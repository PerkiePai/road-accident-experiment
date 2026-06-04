"""Render derived 4-keypoint labels onto converted images to eyeball correctness.
  red=front-center  blue=rear-center  green=front-left  yellow=front-right
  white line = front-center -> rear-center (heading axis)
"""
import argparse, glob, os, cv2, numpy as np

COL = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]  # fc, rc, fl, fr
NAME = ["FC", "RC", "FL", "FR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="datasets/skope3d_yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="data_prep/verify_labels.png")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.ds, "images", args.split, "*.jpg")))
    imgs = imgs[:: max(1, len(imgs) // args.n)][: args.n]
    tiles = []
    for ip in imgs:
        im = cv2.imread(ip)
        H, W = im.shape[:2]
        lp = ip.replace("images", "labels").replace(".jpg", ".txt")
        for ln in open(lp):
            v = ln.split()
            kp = np.array(v[5:], float).reshape(-1, 3)
            pts = [(int(kp[i, 0] * W), int(kp[i, 1] * H)) for i in range(4)]
            cv2.line(im, pts[0], pts[1], (255, 255, 255), 1)
            for i, p in enumerate(pts):
                cv2.circle(im, p, 3, COL[i], -1)
        tiles.append(im)
    # stack into a grid (resize to common height)
    h = 540
    row = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    w = min(t.shape[1] for t in row)
    row = [t[:, :w] for t in row]
    grid = np.vstack(row)
    cv2.imwrite(args.out, grid)
    print("verify ->", args.out, "from", len(imgs), "images")


if __name__ == "__main__":
    main()
