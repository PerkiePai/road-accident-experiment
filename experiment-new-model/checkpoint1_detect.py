"""
Checkpoint 1 — RT-DETR vs YOLO on the occlusion frame (t=3.5s).

Goal: verify that RT-DETR (no NMS, bipartite matching) produces TWO separate
vehicle detections for the two cars that ByteTrack+YOLO collapsed into one bbox.

Run:
    conda activate car-detection
    cd D:/intern/NT/project/road-accident/experiment-new-model
    python checkpoint1_detect.py
"""

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO

# ── Paths ──────────────────────────────────────────────────────
FRAME_PATH = "t3.5.png"
OUT_DIR    = "out"

VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)
CONF            = 0.25
IOU             = 0.70            # only relevant for YOLO (RT-DETR has no NMS)

import os
os.makedirs(OUT_DIR, exist_ok=True)

frame = cv2.imread(FRAME_PATH)
if frame is None:
    raise SystemExit(f"Could not read {FRAME_PATH}")

# ── Helper ─────────────────────────────────────────────────────
def annotate(img, results, model_name, color):
    out = img.copy()
    boxes = results[0].boxes
    count = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{model_name} cls{cls} {conf:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        count += 1
    cv2.putText(out, f"{model_name}: {count} detection(s)",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    return out, count

# ── Run YOLO (baseline) ────────────────────────────────────────
print("Running YOLO11n …")
yolo   = YOLO("yolo11n.pt")
y_res  = yolo(frame, classes=VEHICLE_CLASSES, conf=CONF, iou=IOU, verbose=False)
y_out, y_count = annotate(frame, y_res, "YOLO11n", (0, 165, 255))  # orange
cv2.imwrite(f"{OUT_DIR}/yolo_t3.5.png", y_out)
print(f"  YOLO11n  → {y_count} detection(s)")

# ── Run RT-DETR (proposed) ─────────────────────────────────────
print("Running RT-DETR-s …")
rtdetr   = RTDETR("rtdetr-l.pt")          # downloads ~80 MB on first run
rt_res   = rtdetr(frame, classes=VEHICLE_CLASSES, conf=CONF, verbose=False)
rt_out, rt_count = annotate(frame, rt_res, "RT-DETR-s", (0, 255, 80))  # green
cv2.imwrite(f"{OUT_DIR}/rtdetr_t3.5.png", rt_out)
print(f"  RT-DETR-s → {rt_count} detection(s)")

# ── Side-by-side comparison ────────────────────────────────────
h = max(y_out.shape[0], rt_out.shape[0])
side = np.zeros((h, y_out.shape[1] + rt_out.shape[1] + 4, 3), dtype=np.uint8)
side[:y_out.shape[0],  :y_out.shape[1]]  = y_out
side[:rt_out.shape[0], y_out.shape[1]+4:] = rt_out
cv2.imwrite(f"{OUT_DIR}/compare_t3.5.png", side)

print(f"\nSaved to {OUT_DIR}/")
print(f"  yolo_t3.5.png        ({y_count} det)")
print(f"  rtdetr_t3.5.png      ({rt_count} det)")
print(f"  compare_t3.5.png     (side-by-side)")
print(f"\nResult: {'RT-DETR sees more vehicles ✓' if rt_count > y_count else 'Same count — check compare image'}")
