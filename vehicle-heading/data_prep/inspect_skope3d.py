"""
inspect_skope3d.py — confirm the SKoPe3D annotation layout empirically.

Reads directly from a scene zip (no extraction). For one frame it renders every
vehicle's 33 keypoints with INDEX labels onto the original.avi frame, so we can
SEE which index is front-left / front-right / rear corners / front logo before
committing the 33->4 mapping.

CSV layout (confirmed):
  metadata_v.csv : row0 = (track_id, model_id); rows 1..8 = 3D-bbox corners (px)
  keypoint_v.csv : row0 = (crop_origin_x, crop_origin_y); rows 1..2 = bbox min/max
                   (vis=-1); rows 3..35 = 33 keypoints in CROP-RELATIVE pixels.
                   full_px = crop_origin + (kx, ky)

Run:
    python data_prep/inspect_skope3d.py --zip datasets/skope3d_raw/scene_0.zip --frame 60
"""

import argparse
import io
import os
import zipfile
import cv2
import numpy as np


def parse_csv_bytes(b: bytes):
    rows = []
    for line in b.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([p for p in line.split(",") if p != ""])
    return rows


def keypoints_full(kp_rows):
    """Return (33,3) array of full-image (x, y, vis)."""
    ox, oy = float(kp_rows[0][0]), float(kp_rows[0][1])
    out = []
    for r in kp_rows[3:3 + 33]:
        out.append([ox + float(r[0]), oy + float(r[1]), float(r[2])])
    return np.array(out, dtype=np.float32)


def read_avi_frame(zf: zipfile.ZipFile, scene: str, idx: int):
    """Extract original.avi to a temp file and grab frame idx."""
    import tempfile
    avi_name = f"{scene}/original.avi"
    tmp = os.path.join(tempfile.gettempdir(), f"_skope_{scene}.avi")
    if not os.path.exists(tmp):
        with open(tmp, "wb") as fh:
            fh.write(zf.read(avi_name))
    cap = cv2.VideoCapture(tmp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--frame", type=int, default=60)
    ap.add_argument("--out", default="data_prep/inspect_overlay.png")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    scene = zf.namelist()[0].split("/")[0]
    names = zf.namelist()
    kp_names = [n for n in names if n.startswith(f"{scene}/{args.frame}/keypoint_")]
    meta_names = [n for n in names if n.startswith(f"{scene}/{args.frame}/metadata_")]
    print(f"[inspect] scene={scene} frame={args.frame}: "
          f"{len(kp_names)} vehicles")

    if meta_names:
        print("\n=== metadata_0 (track_id,model_id ; then 8 bbox corners px) ===")
        for r in parse_csv_bytes(zf.read(sorted(meta_names)[0])):
            print("  ", r)
    if kp_names:
        rows = parse_csv_bytes(zf.read(sorted(kp_names)[0]))
        print("\n=== keypoint_0: origin/bboxmin/bboxmax then 33 kpts (crop-rel) ===")
        print("  origin:", rows[0], " bboxmin:", rows[1], " bboxmax:", rows[2])
        kpf = keypoints_full(rows)
        for i, (x, y, v) in enumerate(kpf):
            print(f"  kp{i:2d}: full=({x:7.1f},{y:7.1f}) vis={v:+.0f}")

    frame = read_avi_frame(zf, scene, args.frame)
    if frame is None:
        print("[inspect] could not read avi frame; skip overlay")
        return

    for kf in sorted(kp_names):
        kpf = keypoints_full(parse_csv_bytes(zf.read(kf)))
        for idx, (x, y, v) in enumerate(kpf):
            if v <= 0:
                continue
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (int(x) + 2, int(y) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"\n[inspect] overlay -> {args.out}")


if __name__ == "__main__":
    main()
