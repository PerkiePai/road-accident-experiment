"""
Checkpoint 3 — IMM Filter for speed through collision.

Runs RT-DETR-l + Deep OC-SORT on accident_cm_in_p10.mp4 (same as CP2),
then computes speed two ways for each on-road track:
  1. Raw finite-difference (same as vector-tracking baseline)
  2. IMM filter (CV + CA modes) — should stay valid through sudden deceleration

Saves a comparison plot: out/cp3_speed_plot.png

Pass: IMM speed curve shows smooth deceleration at the collision moment
      rather than the spike/drop the finite-diff baseline produces.

Run:
    conda activate car-detection
    cd D:/intern/NT/project/road-accident/experiment-new-model
    python checkpoint3_imm_speed.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from ultralytics import RTDETR
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
import torch, torchvision.models as tv_models, torchvision.transforms as tv_transforms

from imm_filter import IMMFilter

# ── Paths ──────────────────────────────────────────────────────
INPUT_VIDEO = r"..\_in\car_100kmh.mp4"
H_PATH      = "H_manual.npy"
TRACK_PATH  = "track_manual.npy"
OUT_DIR     = "out"
os.makedirs(OUT_DIR, exist_ok=True)

VEHICLE_CLASSES = [2, 3, 5, 7]
CONF            = 0.30
SPEED_WINDOW    = 0.5    # seconds for finite-diff baseline

# ── Calibration ────────────────────────────────────────────────
H         = np.load(H_PATH)
road_poly = np.load(TRACK_PATH).astype(np.int32)

# ── Video ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dt           = 1.0 / fps

# ── Models ─────────────────────────────────────────────────────
detector = RTDETR("rtdetr-l.pt")
tracker  = DeepOcSort(
    reid_model=None, embedding_off=False,
    w_association_emb=0.6, Q_xy_scaling=0.08, Q_s_scaling=0.0004,
    delta_t=3, inertia=0.2,
)

_feat_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_feat_model  = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
_feat_model.fc = torch.nn.Identity()
_feat_model.eval().to(_feat_device)
_feat_tf = tv_transforms.Compose([
    tv_transforms.ToPILImage(), tv_transforms.Resize((128, 64)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def extract_features(frame_bgr, boxes):
    if not boxes: return np.empty((0, 512), dtype=np.float32)
    fh, fw = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops = []
    for x1,y1,x2,y2 in boxes:
        x1,y1 = max(0,int(x1)), max(0,int(y1))
        x2,y2 = min(fw,int(x2)), min(fh,int(y2))
        crops.append(_feat_tf(rgb[y1:y2,x1:x2]) if x2>x1 and y2>y1
                     else torch.zeros(3,128,64))
    with torch.no_grad():
        feats = _feat_model(torch.stack(crops).to(_feat_device)).cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms < 1e-6, 1.0, norms)

def project(px, py):
    xz = cv2.perspectiveTransform(
        np.array([[[float(px), float(py)]]], dtype=np.float32), H)[0,0]
    return float(xz[0]), float(xz[1])

# ── Per-track state ────────────────────────────────────────────
history  = defaultdict(lambda: deque(maxlen=int(fps * 10)))   # raw positions
imm_filt = {}                                                   # tid -> IMMFilter

# ── Recording ──────────────────────────────────────────────────
# {tid: [(t, raw_speed, imm_speed, mu_ca), ...]}
records  = defaultdict(list)

# ── Main loop ──────────────────────────────────────────────────
print(f"Processing {total_frames} frames @ {fps:.0f} fps ...")
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    t_now = frame_idx / fps

    boxes_np = results = detector(frame, classes=VEHICLE_CLASSES,
                                  conf=CONF, verbose=False)
    b = results[0].boxes
    if b is not None and len(b):
        xyxy = b.xyxy.cpu().numpy()
        dets = np.hstack([xyxy, b.conf.cpu().numpy().reshape(-1,1),
                          b.cls.cpu().numpy().reshape(-1,1)]).astype(np.float32)
        embs = extract_features(frame, xyxy.tolist())
    else:
        dets = np.empty((0,6), dtype=np.float32)
        embs = np.empty((0,512), dtype=np.float32)

    tracks = tracker.update(dets, frame, embs)

    for row in tracks:
        x1,y1,x2,y2,tid,*_ = row
        tid = int(tid)
        gx, gy = (int(x1)+int(x2))//2, int(y2)
        if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
            continue

        x_m, z_m = project(gx, gy)

        # raw finite-diff speed
        history[tid].append((t_now, x_m, z_m))
        raw_speed = None
        if len(history[tid]) >= 2:
            t1, x1m, z1m = history[tid][-1]
            t0, x0m, z0m = history[tid][0]
            for entry in history[tid]:
                if t1 - entry[0] <= SPEED_WINDOW:
                    t0, x0m, z0m = entry; break
            dt_w = t1 - t0
            if dt_w > 1e-3:
                raw_speed = np.hypot(x1m-x0m, z1m-z0m) / dt_w * 3.6

        # sanity gate — skip frames with impossible world coordinates
        if abs(x_m) > 100 or abs(z_m) > 200:
            continue

        # IMM speed
        if tid not in imm_filt:
            imm_filt[tid] = IMMFilter(dt=dt)
            imm_filt[tid].init(x_m, z_m)
        imm_speed = imm_filt[tid].update(x_m, z_m)
        mu_ca     = float(imm_filt[tid].weights[1])

        records[tid].append((t_now, raw_speed, imm_speed, mu_ca))

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"\r  {frame_idx/total_frames*100:.0f}%  t={t_now:.1f}s", end="", flush=True)

cap.release()
print(f"\nDone. Tracked IDs: {sorted(records.keys())}")

# ── Plot ───────────────────────────────────────────────────────
# Pick the 3 tracks with the most data points
top_ids = sorted(records, key=lambda k: len(records[k]), reverse=True)[:3]

fig, axes = plt.subplots(len(top_ids), 1,
                         figsize=(12, 4 * len(top_ids)), sharex=True)
if len(top_ids) == 1:
    axes = [axes]

for ax, tid in zip(axes, top_ids):
    data   = records[tid]
    times  = [d[0] for d in data]
    raw    = [d[1] for d in data]
    imm    = [d[2] for d in data]
    mu_ca  = [d[3] for d in data]

    ax2 = ax.twinx()
    ax2.fill_between(times, mu_ca, alpha=0.15, color="red", label="CA weight")
    ax2.set_ylabel("CA weight", color="red", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="red")

    raw_clean = [v if v is not None else float("nan") for v in raw]
    ax.plot(times, raw_clean, color="steelblue", lw=1.2,
            alpha=0.7, label="Finite-diff (baseline)")
    ax.plot(times, imm,       color="orange",    lw=2.0,
            label="IMM (CV+CA)")

    ax.axvline(3.5, color="red", lw=1.5, ls="--", alpha=0.6, label="t=3.5s ref")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Track id{tid}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("CP3 — IMM vs Finite-Diff Speed through Collision", fontsize=13)
plt.tight_layout()
path = f"{OUT_DIR}/cp3_speed_plot_car100kmh.png"
plt.savefig(path, dpi=150)
print(f"Saved {path}")
