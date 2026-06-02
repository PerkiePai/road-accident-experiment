"""
Checkpoint 4 — CTRV heading estimation.

Runs RT-DETR-l + Deep OC-SORT on the accident clip (same front-end as CP2/CP3),
then computes a heading angle two ways for each on-road track:
  1. Raw position-difference heading  atan2(dx, dz) over SPEED_WINDOW
  2. CTRV Extended Kalman Filter (ctrv_filter.CTRVFilter)

Saves a comparison plot: out/cp4_heading_plot.png

Pass: CTRV heading is smooth during normal driving (where the raw position-diff
      heading is noisy) and shows an abrupt jump at the collision moment.

Run:
    conda activate car-detection
    cd D:/intern/NT/project/road-accident/experiment-new-model
    python checkpoint4_heading.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from ctrv_filter import CTRVFilter
# Heavy detector/tracker imports are deferred into the cache-miss branch below,
# so re-plotting from a cached measurement file needs no GPU stack.

# ── Paths ──────────────────────────────────────────────────────
INPUT_VIDEO = r"..\_in\accident_cm_in_p10.mp4"   # collision clip — heading jump at impact
H_PATH      = "H_manual.npy"
TRACK_PATH  = "track_manual.npy"
OUT_DIR     = "out"
MEAS_CACHE  = "out/cp4_measurements.npz"   # cached (frame, t, tid, x_m, z_m)
os.makedirs(OUT_DIR, exist_ok=True)

VEHICLE_CLASSES = [2, 3, 5, 7]
CONF            = 0.30
SPEED_WINDOW    = 0.5    # seconds for raw position-diff heading

# ── Calibration ────────────────────────────────────────────────
H         = np.load(H_PATH)
road_poly = np.load(TRACK_PATH).astype(np.int32)

def project(px, py):
    xz = cv2.perspectiveTransform(
        np.array([[[float(px), float(py)]]], dtype=np.float32), H)[0,0]
    return float(xz[0]), float(xz[1])

# ── Measurements: cached or freshly extracted via the GPU pass ──
def extract_measurements():
    """Run RT-DETR + Deep OC-SORT over the video once and return
    (meas, fps), where meas rows are [frame, t, tid, x_m, z_m] for
    on-road, sane detections. Cached so re-plots skip the GPU pass."""
    from ultralytics import RTDETR
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
    import torch
    import torchvision.models as tv_models, torchvision.transforms as tv_transforms

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps_  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = RTDETR("rtdetr-l.pt")
    tracker  = DeepOcSort(
        reid_model=None, embedding_off=False,
        w_association_emb=0.6, Q_xy_scaling=0.08, Q_s_scaling=0.0004,
        delta_t=3, inertia=0.2,
    )
    dev   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fmod  = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    fmod.fc = torch.nn.Identity()
    fmod.eval().to(dev)
    ftf = tv_transforms.Compose([
        tv_transforms.ToPILImage(), tv_transforms.Resize((128, 64)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def feats(frame_bgr, boxes):
        if not boxes: return np.empty((0, 512), dtype=np.float32)
        fh, fw = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crops = []
        for x1,y1,x2,y2 in boxes:
            x1,y1 = max(0,int(x1)), max(0,int(y1))
            x2,y2 = min(fw,int(x2)), min(fh,int(y2))
            crops.append(ftf(rgb[y1:y2,x1:x2]) if x2>x1 and y2>y1
                         else torch.zeros(3,128,64))
        with torch.no_grad():
            out = fmod(torch.stack(crops).to(dev)).cpu().numpy()
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n < 1e-6, 1.0, n)

    rows = []
    print(f"Processing {total} frames @ {fps_:.0f} fps (no cache) ...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        t_now = frame_idx / fps_
        det = detector(frame, classes=VEHICLE_CLASSES, conf=CONF, verbose=False)
        b = det[0].boxes
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            dets = np.hstack([xyxy, b.conf.cpu().numpy().reshape(-1,1),
                              b.cls.cpu().numpy().reshape(-1,1)]).astype(np.float32)
            embs = feats(frame, xyxy.tolist())
        else:
            dets = np.empty((0,6), dtype=np.float32)
            embs = np.empty((0,512), dtype=np.float32)

        for row in tracker.update(dets, frame, embs):
            x1,y1,x2,y2,tid,*_ = row
            gx, gy = (int(x1)+int(x2))//2, int(y2)
            if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
                continue
            x_m, z_m = project(gx, gy)
            if abs(x_m) > 100 or abs(z_m) > 200:   # sanity gate
                continue
            rows.append((frame_idx, t_now, int(tid), x_m, z_m))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"\r  {frame_idx/total*100:.0f}%  t={t_now:.1f}s", end="", flush=True)
    cap.release()

    meas_ = np.array(rows, dtype=np.float32)
    np.savez(MEAS_CACHE, meas=meas_, fps=fps_)
    print(f"\nCached {len(meas_)} measurements -> {MEAS_CACHE}")
    return meas_, fps_

if os.path.exists(MEAS_CACHE):
    print(f"Loading cached measurements from {MEAS_CACHE} "
          f"(delete it to re-run the GPU detection pass)")
    _d   = np.load(MEAS_CACHE)
    meas = _d["meas"]; fps = float(_d["fps"])
else:
    meas, fps = extract_measurements()
dt = 1.0 / fps

# ── Per-track state & recording ────────────────────────────────
history   = defaultdict(lambda: deque(maxlen=int(fps * 10)))   # raw positions
ctrv_filt = {}                                                  # tid -> CTRVFilter
# {tid: [(t, raw_heading_deg, ctrv_heading_deg, yaw_rate_deg_s), ...]}
records   = defaultdict(list)

# ── Filter pass (cheap; iterate on this without the GPU) ───────
for frame_idx, t_now, tid, x_m, z_m in meas:
    tid = int(tid)

    # raw position-diff heading: atan2(dx, dz), 0 -> +z, toward +x
    history[tid].append((t_now, x_m, z_m))
    raw_heading = None
    if len(history[tid]) >= 2:
        t1, x1m, z1m = history[tid][-1]
        t0, x0m, z0m = history[tid][0]
        for entry in history[tid]:
            if t1 - entry[0] <= SPEED_WINDOW:
                t0, x0m, z0m = entry; break
        dxm, dzm = x1m - x0m, z1m - z0m
        if np.hypot(dxm, dzm) > 1e-2:
            raw_heading = float(np.degrees(np.arctan2(dxm, dzm)) % 360.0)

    # CTRV heading
    if tid not in ctrv_filt:
        ctrv_filt[tid] = CTRVFilter(dt=dt)
        ctrv_filt[tid].init(x_m, z_m)
    ctrv_heading = ctrv_filt[tid].update(x_m, z_m)
    yaw_rate     = ctrv_filt[tid].yaw_rate_deg

    records[tid].append((t_now, raw_heading, ctrv_heading, yaw_rate))

print(f"Tracked IDs: {sorted(records.keys())}")

# ── Plot ───────────────────────────────────────────────────────
def unwrap_deg(vals):
    """Unwrap a 0-360 heading series (may contain None/NaN) onto a continuous
    axis so the 0/360 seam stops faking vertical jumps. Gaps are preserved."""
    out    = [float("nan")] * len(vals)
    offset = 0.0
    prev   = None   # previous *wrapped* value
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if prev is not None:
            d = v - prev
            while d >  180: offset -= 360; d -= 360
            while d < -180: offset += 360; d += 360
        out[i] = v + offset
        prev   = v
    return out

# Pick the 3 tracks that actually MOVE — heading is meaningful only there
# (parked cars are correctly held flat by the filter but make a dull plot).
def _motion(tid):
    pts = [(x, z) for (f, t, m, x, z) in meas if int(m) == tid]
    if len(pts) < 2: return 0.0
    (x0, z0), (x1, z1) = pts[0], pts[-1]
    return float(np.hypot(x1 - x0, z1 - z0))
moving_ids = [k for k in records if _motion(k) > 3.0]   # >3 m net displacement
ranked     = sorted(moving_ids or records,
                    key=lambda k: _motion(k), reverse=True)
top_ids    = ranked[:3]

fig, axes = plt.subplots(len(top_ids), 1,
                         figsize=(12, 4 * len(top_ids)), sharex=True)
if len(top_ids) == 1:
    axes = [axes]

for ax, tid in zip(axes, top_ids):
    data  = records[tid]
    times = [d[0] for d in data]
    raw   = [d[1] for d in data]
    ctrv  = [d[2] for d in data]
    yaw   = [d[3] for d in data]

    ax2 = ax.twinx()
    ax2.plot(times, yaw, color="green", lw=1.0, alpha=0.5, label="yaw rate")
    ax2.set_ylabel("yaw rate (deg/s)", color="green", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="green")

    raw_u  = unwrap_deg(raw)
    ctrv_u = unwrap_deg(ctrv)
    ax.plot(times, raw_u,  color="steelblue", lw=1.0, marker=".", ms=2,
            alpha=0.5, label="Position-diff (raw)")
    ax.plot(times, ctrv_u, color="orange",    lw=2.0,
            label="CTRV EKF")

    ax.axvline(3.5, color="red", lw=1.5, ls="--", alpha=0.6, label="t=3.5s ref")
    ax.set_ylabel("Heading (deg, unwrapped)")
    ax.set_title(f"Track id{tid}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("CP4 — CTRV vs Position-Diff Heading", fontsize=13)
plt.tight_layout()
path = f"{OUT_DIR}/cp4_heading_plot.png"
plt.savefig(path, dpi=150)
print(f"Saved {path}")
