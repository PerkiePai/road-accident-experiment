"""
train_pose.py — fine-tune a 4-keypoint vehicle-pose model on the SKoPe3D-derived
dataset.

Defaults target an RTX 5090 (32 GB). Drop --batch and --cache if on a smaller GPU.

Usage:
    python train_pose.py                              # defaults (batch=128, cache=ram)
    python train_pose.py --batch 4 --cache False      # small GPU fallback
    python train_pose.py --model yolo11n-pose.pt
"""

import argparse
from ultralytics import YOLO

DATA_YAML = "datasets/skope3d_yolo/vehicle_pose.yaml"


def pick_model(requested: str) -> str:
    """Prefer YOLO26 pose (better for non-human keypoints); fall back to YOLO11."""
    if requested:
        return requested
    for cand in ("yolo26n-pose.pt", "yolo11n-pose.pt"):
        try:
            YOLO(cand)  # triggers auto-download; raises if unavailable
            return cand
        except Exception:
            continue
    return "yolo11n-pose.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="base pose weights (.pt)")
    ap.add_argument("--data", default=DATA_YAML)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default="ram", help="'ram', 'disk', or 'False'")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="vehicle_pose4")
    args = ap.parse_args()

    cache = False if args.cache.lower() == "false" else args.cache

    model_name = pick_model(args.model)
    print(f"[train] base model: {model_name}")
    model = YOLO(model_name)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=cache,
        device=args.device,
        name=args.name,
        amp=True,
        cos_lr=True,
        patience=20,
        plots=True,
    )
    print("[train] done. best weights under runs/pose/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
