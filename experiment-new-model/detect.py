import os
import colorsys
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from ultralytics import RTDETR
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

# ─── Config ────────────────────────────────────────────────────
INPUT_VIDEO  = "../_in/accident_cm_in_p10.mp4"
OUTPUT_VIDEO = "out/accident_cm_in_p10_exp.mp4"
H_PATH       = "H_manual.npy"
SRC_PATH     = "src_manual.npy"
TRACK_PATH   = "track_manual.npy"

VEHICLE_CLASSES      = [2, 3, 5, 7]   # car, motorcycle, bus, truck
CONF                 = 0.30
HISTORY_SEC          = 5.0
TRACE_SEC            = 2.5
SPEED_WINDOW         = 0.5
EMA_ALPHA            = 0.1
PANEL_SIZE           = (340, 200)
PANEL_MARGIN         = 20
PANEL_VMAX_KMH_FLOOR = 120

# ─── Calibration ───────────────────────────────────────────────
for p in (H_PATH, SRC_PATH, TRACK_PATH):
    if not os.path.exists(p):
        raise SystemExit(
            f"Missing calibration file: {p}\n"
            f"Run `python manual_calibrate.py` first."
        )

H         = np.load(H_PATH)
src_rect  = np.load(SRC_PATH).astype(np.int32)
road_poly = np.load(TRACK_PATH).astype(np.int32)

# ─── Video I/O ─────────────────────────────────────────────────
os.makedirs("out", exist_ok=True)
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {INPUT_VIDEO}")

fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer       = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ─── Detector: RT-DETR-l (no NMS) ─────────────────────────────
detector = RTDETR("rtdetr-l.pt")

# ─── Tracker: Deep OC-SORT ─────────────────────────────────────
# Q_xy_scaling raised from default 0.01 → 0.08 for fast-moving vehicles.
# We supply ResNet-18 embeddings externally via embs= so reid_model=None.
tracker = DeepOcSort(
    reid_model=None,
    embedding_off=False,
    w_association_emb=0.6,
    Q_xy_scaling=0.08,
    Q_s_scaling=0.0004,
    delta_t=3,
    inertia=0.2,
)

# ─── Feature extractor: ResNet-18 ──────────────────────────────
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

# ─── State ─────────────────────────────────────────────────────
history      = defaultdict(lambda: deque(maxlen=int(fps * HISTORY_SEC)))
ema_speed    = {}
speed_history = defaultdict(lambda: deque(maxlen=int(fps * HISTORY_SEC)))
gp_trace     = defaultdict(lambda: deque(maxlen=int(fps * TRACE_SEC) + 1))

# ─── Helpers ───────────────────────────────────────────────────
def color_for_id(tid):
    h_ = (int(tid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h_, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)

def project_to_ground(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    xz = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xz[0]), float(xz[1])

def speed_from_history(hist, window_sec):
    if len(hist) < 2:
        return None
    t_now, x_now, z_now = hist[-1]
    t_old, x_old, z_old = hist[0]
    for entry in hist:
        if t_now - entry[0] <= window_sec:
            t_old, x_old, z_old = entry
            break
    dt = t_now - t_old
    if dt < 1e-3:
        return None
    return np.hypot(x_now - x_old, z_now - z_old) / dt * 3.6

def draw_road_overlay(frame, poly):
    ov = frame.copy()
    cv2.fillPoly(ov, [poly], (255, 200, 0))
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [poly], True, (255, 220, 0), 2)

def draw_perspective_grid(frame, H, x_step=7.0, z_step=10.0, alpha=0.4):
    H_inv   = np.linalg.inv(H)
    fh, fw  = frame.shape[:2]
    overlay = frame.copy()
    rect    = (0, 0, fw - 1, fh - 1)
    color   = (0, 255, 255)

    corners = np.array(
        [[[0., 0.]], [[fw-1., 0.]], [[fw-1., fh-1.]], [[0., fh-1.]]],
        dtype=np.float32)
    wc    = cv2.perspectiveTransform(corners, H)[:, 0, :]
    x_min = float(wc[:, 0].min()) - x_step
    x_max = float(wc[:, 0].max()) + x_step
    z_min = float(wc[:, 1].min()) - z_step
    z_max = float(wc[:, 1].max()) + z_step

    def to_img(xm, zm):
        p = cv2.perspectiveTransform(
            np.array([[[xm, zm]]], dtype=np.float32), H_inv)[0, 0]
        return (int(np.clip(round(p[0]), -32767, 32767)),
                int(np.clip(round(p[1]), -32767, 32767)))

    def draw_line(x0, z0, x1, z1):
        ok, p0, p1 = cv2.clipLine(rect, to_img(x0, z0), to_img(x1, z1))
        if ok:
            cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)

    xi = np.floor(x_min / x_step) * x_step
    while xi <= x_max:
        draw_line(xi, z_min, xi, z_max); xi += x_step
    zi = np.floor(z_min / z_step) * z_step
    while zi <= z_max:
        draw_line(x_min, zi, x_max, zi); zi += z_step

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_speed_panel(frame, speed_history, ema_speed, t_now):
    pw, ph = PANEL_SIZE
    x0 = w - pw - PANEL_MARGIN
    y0 = h - ph - PANEL_MARGIN
    x1, y1 = x0 + pw, y0 + ph

    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

    pad_l, pad_r, pad_t, pad_b = 38, 8, 22, 18
    px0, py0 = x0 + pad_l, y0 + pad_t
    px1, py1 = x1 - pad_r, y1 - pad_b
    plot_w, plot_h = px1 - px0, py1 - py0

    vmax = PANEL_VMAX_KMH_FLOOR
    if ema_speed:
        vmax = max(vmax, max(ema_speed.values()) * 1.2)

    cv2.putText(frame, "km/h", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    for i in range(5):
        v  = vmax * i / 4
        yy = py1 - int(plot_h * i / 4)
        cv2.line(frame, (px0, yy), (px1, yy), (70, 70, 70), 1)
        cv2.putText(frame, f"{int(v)}", (x0 + 6, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, f"-{HISTORY_SEC:.0f}s", (px0, py1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "0", (px1 - 8, py1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    active_ids = []
    for tid, sh in speed_history.items():
        if len(sh) < 2 or t_now - sh[-1][0] > 1.0:
            continue
        pts = []
        for t, v in sh:
            age = t_now - t
            if age > HISTORY_SEC:
                continue
            fx = px1 - int(plot_w * (age / HISTORY_SEC))
            fy = py1 - int(plot_h * min(v / vmax, 1.0))
            pts.append((fx, fy))
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)],
                          False, color_for_id(tid), 2, cv2.LINE_AA)
            active_ids.append(tid)

    lx = x0 + 50
    for tid in active_ids[:5]:
        cv2.rectangle(frame, (lx, y0 + 6), (lx + 10, y0 + 14), color_for_id(tid), -1)
        cv2.putText(frame, f"id{tid}", (lx + 14, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        lx += 50
    if len(active_ids) > 5:
        cv2.putText(frame, f"+{len(active_ids)-5}", (lx, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

# ─── Main loop ─────────────────────────────────────────────────
print(f"Input : {INPUT_VIDEO}  ({total_frames} frames @ {fps:.1f} fps)")
print(f"Output: {OUTPUT_VIDEO}")
print(f"Device: {_feat_device}")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_now = frame_idx / fps

    # ── Detect (RT-DETR-l, no NMS) ──────────────────────────────
    results = detector(frame, classes=VEHICLE_CLASSES, conf=CONF, verbose=False)
    boxes   = results[0].boxes

    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls  = boxes.cls.cpu().numpy().reshape(-1, 1)
        dets = np.hstack([xyxy, conf, cls]).astype(np.float32)
        embs = extract_features(frame, xyxy.tolist())
    else:
        dets = np.empty((0, 6), dtype=np.float32)
        embs = np.empty((0, 512), dtype=np.float32)

    # ── Track (Deep OC-SORT + ResNet-18 ReID) ───────────────────
    tracks = tracker.update(dets, frame, embs)  # (M,8): x1,y1,x2,y2,id,conf,cls,det_ind

    # ── Draw road overlays ───────────────────────────────────────
    draw_road_overlay(frame, road_poly)
    cv2.polylines(frame, [src_rect], True, (0, 200, 200), 1)
    draw_perspective_grid(frame, H)

    # ── Per-track drawing ────────────────────────────────────────
    for row in tracks:
        x1, y1, x2, y2, tid, conf_, cls_, _ = row
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid = int(tid)
        gx  = (x1 + x2) // 2
        gy  = y2

        # off-road → dim ghost, skip speed/trail
        if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
            continue

        color     = color_for_id(tid)
        x_m, z_m = project_to_ground(gx, gy)

        # speed
        history[tid].append((t_now, x_m, z_m))
        v = speed_from_history(history[tid], SPEED_WINDOW)
        if v is not None:
            prev_ema     = ema_speed.get(tid, v)
            ema_speed[tid] = (1 - EMA_ALPHA) * prev_ema + EMA_ALPHA * v
            speed_history[tid].append((t_now, ema_speed[tid]))
            label = f"id{tid}  {v:.0f}km/h  d={z_m:.1f}m"
        else:
            label = f"id{tid}  d={z_m:.1f}m"

        # fading trail
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
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # ── Prune stale state ────────────────────────────────────────
    for tid in list(ema_speed):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            ema_speed.pop(tid, None)
    for tid in list(speed_history):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            speed_history.pop(tid, None)
    for tid in list(gp_trace):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            gp_trace.pop(tid, None)

    draw_speed_panel(frame, speed_history, ema_speed, t_now)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames else f"{frame_idx}fr"
        print(f"\r  {pct}  t={t_now:.1f}s", end="", flush=True)

cap.release()
writer.release()
print(f"\nWrote {OUTPUT_VIDEO} ({frame_idx} frames)")
