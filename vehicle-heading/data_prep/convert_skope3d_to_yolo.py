"""
convert_skope3d_to_yolo.py — SKoPe3D scene zips -> Ultralytics 4-keypoint pose.

We DERIVE the 4 keypoints from each vehicle's 8 projected 3D-bbox corners
(metadata_v.csv) plus kp32 (front logo, keypoint_v.csv) — not from the 33
semantic keypoints, whose ordering is undocumented. This is robust and
viewpoint-agnostic:

  metadata rows 1..8 = 8 cuboid corners in image px, as 4 consecutive
    (bottom, top) vertical-edge pairs.  Ground corner = the larger-y one in
    each pair  ->  4 ground corners.
  kp32 (front logo) full px  ->  picks which 2 ground corners are the FRONT pair.

  Output keypoints (order matches vehicle_pose.yaml):
    0 front-center = midpoint(front ground pair)
    1 rear-center  = midpoint(rear  ground pair)
    2 front-left   } the two front ground corners, split L/R by the sign of
    3 front-right  } cross(forward, corner-front_center)  (flip swaps 2<->3)

Reads the avi + csvs straight from the zip (no 500k-file extraction).

Run:
    python data_prep/convert_skope3d_to_yolo.py \
        --zips datasets/skope3d_raw/scene_0.zip datasets/skope3d_raw/scene_1.zip \
        --val_zips datasets/skope3d_raw/scene_2.zip \
        --stride 5 --min_box_px 18
"""

import argparse
import glob
import os
import tempfile
import zipfile

import cv2
import numpy as np


def parse(b: bytes):
    return [[p for p in ln.split(",") if p != ""]
            for ln in b.decode().splitlines() if ln.strip()]


def derive_keypoints(corners: np.ndarray, kp32: np.ndarray):
    """corners: (8,2) image px. kp32: (2,) front-logo px.
    Returns dict with front_center/rear_center/front_left/front_right (each (2,))
    and the 2D bbox (x1,y1,x2,y2), or None if degenerate."""
    if corners.shape != (8, 2):
        return None
    # 4 consecutive vertical-edge pairs -> ground corner = larger y in each pair
    ground = []
    for i in range(0, 8, 2):
        a, b = corners[i], corners[i + 1]
        ground.append(a if a[1] >= b[1] else b)
    ground = np.array(ground)  # (4,2)

    # front pair = the 2 ground corners nearest the front logo
    d = np.hypot(ground[:, 0] - kp32[0], ground[:, 1] - kp32[1])
    order = np.argsort(d)
    front = ground[order[:2]]
    rear = ground[order[2:]]

    front_center = front.mean(axis=0)
    rear_center = rear.mean(axis=0)
    forward = front_center - rear_center
    if np.hypot(*forward) < 1e-3:
        return None

    # split front pair into L/R by cross(forward, corner-front_center) sign
    def cross(c):
        d = c - front_center
        return forward[0] * d[1] - forward[1] * d[0]

    c0, c1 = front[0], front[1]
    if cross(c0) >= cross(c1):
        front_right, front_left = c0, c1
    else:
        front_right, front_left = c1, c0

    x1, y1 = corners[:, 0].min(), corners[:, 1].min()
    x2, y2 = corners[:, 0].max(), corners[:, 1].max()
    return {
        "kpts": np.array([front_center, rear_center, front_left, front_right]),
        "bbox": (x1, y1, x2, y2),
    }


def process_zip(zpath, split, img_dir, lbl_dir, stride, min_box_px, vis_flag):
    zf = zipfile.ZipFile(zpath)
    scene = zf.namelist()[0].split("/")[0]
    names = zf.namelist()
    frames = sorted({int(n.split("/")[1]) for n in names
                     if len(n.split("/")) > 2 and n.split("/")[1].isdigit()})

    # extract avi once to temp
    tmp = os.path.join(tempfile.gettempdir(), f"_skope_{scene}.avi")
    if not os.path.exists(tmp):
        with open(tmp, "wb") as fh:
            fh.write(zf.read(f"{scene}/original.avi"))
    cap = cv2.VideoCapture(tmp)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_img = n_obj = 0
    want = set(frames[::stride])
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi not in want:
            continue

        metas = sorted(n for n in names if n.startswith(f"{scene}/{fi}/metadata_"))
        lines = []
        for mn in metas:
            vidx = mn.split("_")[-1].split(".")[0]
            mrows = parse(zf.read(mn))
            if len(mrows) < 9:
                continue
            corners = np.array([[float(r[0]), float(r[1])] for r in mrows[1:9]])
            try:
                krows = parse(zf.read(f"{scene}/{fi}/keypoint_{vidx}.csv"))
            except KeyError:
                continue
            ox, oy = float(krows[0][0]), float(krows[0][1])
            kp32 = np.array([ox + float(krows[3 + 32][0]),
                             oy + float(krows[3 + 32][1])])

            res = derive_keypoints(corners, kp32)
            if res is None:
                continue
            x1, y1, x2, y2 = res["bbox"]
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < min_box_px:
                continue
            # clip to image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            nw, nh = (x2 - x1) / W, (y2 - y1) / H
            parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
            for (kx, ky) in res["kpts"]:
                # keypoints projected outside the frame -> mark not-labelled
                # (vis=0) so they don't pull the regression to a clamped edge.
                inside = 0 <= kx < W and 0 <= ky < H
                vx, vy = min(max(kx, 0), W - 1) / W, min(max(ky, 0), H - 1) / H
                parts.append(f"{vx:.6f} {vy:.6f} {vis_flag if inside else 0}")
            lines.append(" ".join(parts))
            n_obj += 1

        if not lines:
            continue
        stem = f"{scene}_f{fi:04d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        n_img += 1

    cap.release()
    print(f"[{split}] {scene}: {n_img} images, {n_obj} vehicles "
          f"(W={W} H={H}, stride={stride})")
    return W, H, n_img, n_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True, help="train scene zips")
    ap.add_argument("--val_zips", nargs="*", default=[], help="val scene zips")
    ap.add_argument("--out", default="datasets/skope3d_yolo")
    ap.add_argument("--stride", type=int, default=5,
                    help="sample every Nth frame (consecutive frames are redundant)")
    ap.add_argument("--min_box_px", type=int, default=18,
                    help="skip vehicles whose smaller bbox side is below this")
    ap.add_argument("--vis", type=int, default=2, help="keypoint visibility flag")
    args = ap.parse_args()

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    tot = {"train": [0, 0], "val": [0, 0]}
    for zp in args.zips:
        _, _, ni, no = process_zip(
            zp, "train",
            os.path.join(args.out, "images", "train"),
            os.path.join(args.out, "labels", "train"),
            args.stride, args.min_box_px, args.vis)
        tot["train"][0] += ni; tot["train"][1] += no
    for zp in args.val_zips:
        _, _, ni, no = process_zip(
            zp, "val",
            os.path.join(args.out, "images", "val"),
            os.path.join(args.out, "labels", "val"),
            args.stride, args.min_box_px, args.vis)
        tot["val"][0] += ni; tot["val"][1] += no

    # if no explicit val, carve 10% of train images into val
    if not args.val_zips:
        import random, shutil
        random.seed(0)
        timg = sorted(glob.glob(os.path.join(args.out, "images", "train", "*.jpg")))
        val = random.sample(timg, max(1, len(timg) // 10))
        for ip in val:
            stem = os.path.splitext(os.path.basename(ip))[0]
            shutil.move(ip, os.path.join(args.out, "images", "val", stem + ".jpg"))
            shutil.move(os.path.join(args.out, "labels", "train", stem + ".txt"),
                        os.path.join(args.out, "labels", "val", stem + ".txt"))
        print(f"[split] moved {len(val)} images train->val (no val_zips given)")

    yaml_path = os.path.join(args.out, "vehicle_pose.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(
            "# SKoPe3D-derived 4-keypoint vehicle pose\n"
            f"path: {os.path.abspath(args.out)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "kpt_shape: [4, 3]\n"
            "flip_idx: [0, 1, 3, 2]\n"
            "names:\n  0: vehicle\n"
        )
    print(f"[done] train={tot['train']} val={tot['val']}  yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
