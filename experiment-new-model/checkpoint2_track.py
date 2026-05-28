"""
Checkpoint 2 - RT-DETR-l + Deep OC-SORT tracking on full accident video.

Goal: verify both colliding cars keep separate IDs through the t=3.5s occlusion
window where YOLO+ByteTrack previously collapsed them into one track.

Run:
    conda activate car-detection
    cd D:/intern/NT/project/road-accident/experiment-new-model
    python checkpoint2_track.py
"""

import os
import colorsys

import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from collections import defaultdict, deque
from ultralytics import RTDETR
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

# ── Paths ──────────────────────────────────────────────────────
INPUT_VIDEO = r"..\_ in\accident_cm_in_p10.mp4"
INPUT_VIDEO = r"..\_in\accident_cm_in_p10.mp4"
H_PATH      = r"..\vector-tracking\H_manual.npy"
TRACK_PATH  = r"..\vector-tracking\track_manual.npy"
SRC_PATH    = r"..\vector-tracking\src_manual.npy"
OUTPUT_VIDEO = "out/cp2_tracked.mp4"
os.makedirs("out", exist_ok=True)

VEHICLE_CLASSES = [2, 3, 5, 7]
CONF            = 0.30
TRACE_SEC       = 2.5    # seconds of ground-point trail

# ── Calibration ────────────────────────────────────────────────
H         = np.load(H_PATH)
road_poly = np.load(TRACK_PATH).astype(np.int32)
src_rect  = np.load(SRC_PATH).astype(np.int32)

# ── Video I/O ──────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {INPUT_VIDEO}")

fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer       = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

# ── Detector ───────────────────────────────────────────────────
detector = RTDETR("rtdetr-l.pt")

# ── Tracker: Deep OC-SORT ──────────────────────────────────────
# Q_xy_scaling raised from default 0.01 → 0.08 for fast-moving vehicles
tracker = DeepOcSort(
    reid_model=None,          # we supply external ResNet-18 embeddings via embs=
    embedding_off=False,
    w_association_emb=0.6,    # 60% visual, 40% spatial cost
    Q_xy_scaling=0.08,        # wider position process noise for fast cars
    Q_s_scaling=0.0004,
    delta_t=3,
    inertia=0.2,
)

# ── ResNet-18 feature extractor ────────────────────────────────
_feat_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_feat_model  = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
_feat_model.fc = torch.nn.Identity()
_feat_model.eval().to(_feat_device)

_feat_tf = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((128, 64)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

def extract_features(frame_bgr, boxes_xyxy):
    if not boxes_xyxy:
        return np.empty((0, 512), dtype=np.float32)
    fh, fw = frame_bgr.shape[:2]
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops  = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(fw, int(x2)), min(fh, int(y2))
        if x2 <= x1 or y2 <= y1:
            crops.append(torch.zeros(3, 128, 64))
        else:
            crops.append(_feat_tf(rgb[y1:y2, x1:x2]))
    batch = torch.stack(crops).to(_feat_device)
    with torch.no_grad():
        feats = _feat_model(batch).cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms < 1e-6, 1.0, norms)

# ── Helpers ────────────────────────────────────────────────────
def color_for_id(tid):
    h_ = (int(tid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h_, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)

def project_to_ground(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    xz = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xz[0]), float(xz[1])

def draw_road_overlay(frame, poly):
    ov = frame.copy()
    cv2.fillPoly(ov, [poly], (255, 200, 0))
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [poly], True, (255, 220, 0), 2)

# ── State ──────────────────────────────────────────────────────
gp_trace = defaultdict(lambda: deque(maxlen=int(fps * TRACE_SEC) + 1))

# ── Keyframe saver (t=3.0 to 5.0 s) ───────────────────────────
keyframe_secs  = {3.0, 3.5, 4.0, 4.5, 5.0}
saved_keyframes = set()

# ── Main loop ──────────────────────────────────────────────────
print(f"Input : {INPUT_VIDEO}  ({total_frames} frames @ {fps:.1f} fps)")
print(f"Output: {OUTPUT_VIDEO}")
print(f"Device: {_feat_device}")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_now = frame_idx / fps

    # ── Detect ──────────────────────────────────────────────────
    results  = detector(frame, classes=VEHICLE_CLASSES, conf=CONF, verbose=False)
    boxes    = results[0].boxes

    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls  = boxes.cls.cpu().numpy().reshape(-1, 1)
        dets = np.hstack([xyxy, conf, cls]).astype(np.float32)  # (N,6)
        embs = extract_features(frame, xyxy.tolist())            # (N,512)
    else:
        dets = np.empty((0, 6), dtype=np.float32)
        embs = np.empty((0, 512), dtype=np.float32)

    # ── Track ───────────────────────────────────────────────────
    tracks = tracker.update(dets, frame, embs)   # (M,8): x1,y1,x2,y2,id,conf,cls,det_ind

    # ── Draw overlays ───────────────────────────────────────────
    draw_road_overlay(frame, road_poly)
    cv2.polylines(frame, [src_rect], True, (0, 200, 200), 1)

    active_ids = []
    for row in tracks:
        x1, y1, x2, y2, tid, conf_, cls_, _ = row
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid  = int(tid)
        gx   = (x1 + x2) // 2
        gy   = y2

        # road-polygon filter
        if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
            continue

        color   = color_for_id(tid)
        x_m, z_m = project_to_ground(gx, gy)
        active_ids.append(tid)

        # fading ground-point trail
        gp_trace[tid].append((t_now, gx, gy))
        trail = [(px, py) for t, px, py in gp_trace[tid] if t_now - t <= TRACE_SEC]
        n     = len(trail)
        if n >= 2:
            for i in range(1, n):
                alpha     = i / n
                seg_color = tuple(int(ch * alpha) for ch in color)
                cv2.line(frame, trail[i-1], trail[i], seg_color, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (gx, gy), 5, (0, 255, 255), -1)
        cv2.putText(frame, f"id{tid}  d={z_m:.1f}m",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    # prune stale trails
    for tid in list(gp_trace):
        if gp_trace[tid] and t_now - gp_trace[tid][-1][0] > 3.0:
            del gp_trace[tid]

    # ── HUD ─────────────────────────────────────────────────────
    cv2.putText(frame, f"t={t_now:.2f}s  tracks:{len(active_ids)}  ids:{sorted(active_ids)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Save keyframes ──────────────────────────────────────────
    for ks in keyframe_secs:
        if abs(t_now - ks) < (0.5 / fps) and ks not in saved_keyframes:
            cv2.imwrite(f"out/cp2_t{ks:.1f}.png", frame)
            saved_keyframes.add(ks)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames else f"{frame_idx}fr"
        print(f"\r  {pct}  t={t_now:.1f}s  active_ids={sorted(active_ids)}", end="", flush=True)

cap.release()
writer.release()
print(f"\nWrote {OUTPUT_VIDEO}  ({frame_idx} frames)")
print(f"Keyframes saved: {sorted(saved_keyframes)}")
