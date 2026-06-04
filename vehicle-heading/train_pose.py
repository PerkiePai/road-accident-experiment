"""
train_pose.py — fine-tune a 4-keypoint vehicle-pose model on the SKoPe3D-derived
dataset.

VRAM-safe defaults for an RTX 2050 (4 GB): nano backbone, small batch, AMP, no
RAM cache. If you hit CUDA OOM, drop --imgsz to 512 or --batch to 2.

Usage (conda env car-detection):
    python train_pose.py                       # defaults
    python train_pose.py --imgsz 512 --batch 2 # if OOM
    python train_pose.py --model yolo11n-pose.pt
"""

import argparse
from ultralytics import YOLO

DATA_YAML = "data_prep/vehicle_pose.yaml"


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
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="vehicle_pose4")
    args = ap.parse_args()

    model_name = pick_model(args.model)
    print(f"[train] base model: {model_name}")
    model = YOLO(model_name)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        amp=True,
        cache=False,
        cos_lr=True,
        patience=20,
        plots=True,
    )
    print("[train] done. best weights under runs/pose/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
