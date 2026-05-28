import os
import colorsys
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ─── Fix 3: Kalman Filter patch ───────────────────────────────
# Ultralytics KF is tuned for pedestrians: _std_weight_velocity = 1/160.
# At 100 km/h a car moves ~28 px/frame; the default velocity std
# (~0.6 px for a 100 px tall box) is far too tight → IoU hits 0 → track dropped.
# Patching before any model.track() call so ByteTrack picks up the new values.
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH as _KFXYAH
_orig_kf_init = _KFXYAH.__init__
def _patched_kf_init(self):
    _orig_kf_init(self)
    self._std_weight_position = 1.0 / 10   # 2× default (was 1/20)
    self._std_weight_velocity = 1.0 / 16   # 10× default (was 1/160)
_KFXYAH.__init__ = _patched_kf_init

# ─── Config ────────────────────────────────────────────────────
INPUT_VIDEO   = "../_in/accident_cm_in_p10.mp4"
model = YOLO("yolo11n.pt")
OUTPUT_VIDEO  = "out/accident_cm_in_p10_tracked_3.mp4"
H_PATH        = "H_manual.npy"
SRC_PATH      = "src_manual.npy"
TRACK_PATH    = "track_manual.npy"
HISTORY_SEC   = 5.0          # how much speed history to plot
TRACE_SEC     = 2.5          # seconds of ground-point trail to draw
SPEED_WINDOW  = 0.5          # seconds, finite-diff window for speed
EMA_ALPHA     = 0.1          # smoothing factor for per-track speed
BBOX_EMA      = 0.6          # bbox smoothing — visual only
GP_EMA        = 0.75         # ground-point smoothing — drives projection & speed
PANEL_SIZE    = (340, 200)   # (w, h) of the speed panel
PANEL_MARGIN  = 20
PANEL_VMAX_KMH_FLOOR = 120

# ─── Re-ID stitcher config ────────────────────────────────────
REID_DIST_BASE_M    = 2.0    # match radius at z=0
REID_DIST_PER_M     = 0.10   # +metres of slack per metre of depth
REID_MAX_GAP_S      = 2.0    # max time a lost track stays a candidate
REID_LOW_VEL_GAP_S  = 1.0    # tighter window for ~stopped cars
REID_LOW_VEL_THRESH = 1.0    # m/s — below this counts as "stopped"
REID_SPATIAL_W      = 0.4    # weight of normalised spatial cost in combined cost
REID_VISUAL_W       = 0.6    # weight of cosine distance in combined cost

# ─── Fix 2: Feature extractor (ResNet-18, ImageNet pretrained) ─
_feat_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_feat_model  = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
_feat_model.fc = torch.nn.Identity()   # strip classifier → 512-d
_feat_model.eval().to(_feat_device)

_feat_tf = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((128, 64)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

def extract_features(frame_bgr, boxes_xyxy):
    """Return L2-normalised 512-d ResNet-18 features for each (x1,y1,x2,y2) box."""
    if not boxes_xyxy:
        return []
    fh, fw = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            crops.append(torch.zeros(3, 128, 64))
        else:
            crops.append(_feat_tf(frame_rgb[y1:y2, x1:x2]))
    batch = torch.stack(crops).to(_feat_device)
    with torch.no_grad():
        feats = _feat_model(batch).cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms < 1e-6, 1.0, norms)

# ─── Calibration ───────────────────────────────────────────────
if not (os.path.exists(H_PATH) and os.path.exists(SRC_PATH) and os.path.exists(TRACK_PATH)):
    raise SystemExit(
        f"Missing calibration files ({H_PATH}, {SRC_PATH}, {TRACK_PATH}).\n"
        f"Run `python manual_calibrate.py` first."
    )

H = np.load(H_PATH)
src_rect   = np.load(SRC_PATH).astype(np.int32)
road_poly  = np.load(TRACK_PATH).astype(np.int32)

# ─── Video I/O ─────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Could not open {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ─── State ─────────────────────────────────────────────────────
history      = defaultdict(lambda: deque(maxlen=int(fps * HISTORY_SEC)))
ema_speed    = {}
speed_history = defaultdict(lambda: deque(maxlen=int(fps * HISTORY_SEC)))
bbox_smooth  = {}
gp_smooth    = {}
gp_trace     = defaultdict(lambda: deque(maxlen=int(fps * TRACE_SEC) + 1))
stitcher     = None


def color_for_id(tid: int):
    h_ = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h_, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def project_to_ground(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    xz = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xz[0]), float(xz[1])


# ─── Fix 1 + 2: ReID Stitcher with Hungarian + visual cost ────
class ReIDStitcher:
    """
    Spatial + Visual Stitcher.

    Per-frame call: .update(t_now, dets)
      dets = [(raw_id, x_m, z_m, feat_vec), ...]
      feat_vec: L2-normalised 512-d array, or None

    Returns {raw_id: canonical_id}.

    Matching uses the Hungarian algorithm on a combined cost:
      cost = REID_SPATIAL_W * (euclidean / threshold)
           + REID_VISUAL_W  * cosine_distance
    Pairs whose spatial distance exceeds the depth-scaled gate are
    forbidden (cost = INF) before the solver runs.
    """

    def __init__(self):
        self.alias  = {}   # raw_id  -> canonical_id
        self.active = {}   # cid -> {t, pos, vel, feat}
        self.lost   = {}   # cid -> {t, pos, vel, feat}

    def _dist_thresh(self, x_m, z_m):
        return REID_DIST_BASE_M + REID_DIST_PER_M * (max(z_m, 0.0) + abs(x_m))

    def _update_state(self, cid, t, x, z, feat=None):
        prev = self.active.get(cid)
        if prev:
            dt = t - prev['t']
            if dt > 1e-3:
                vx = (x - prev['pos'][0]) / dt
                vz = (z - prev['pos'][1]) / dt
                pvx, pvz = prev['vel']
                vx = 0.5 * pvx + 0.5 * vx
                vz = 0.5 * pvz + 0.5 * vz
            else:
                vx, vz = prev['vel']
            # EMA-blend feature vector so the stored embedding tracks appearance changes
            if feat is not None and prev.get('feat') is not None:
                blended = 0.7 * prev['feat'] + 0.3 * feat
                norm = np.linalg.norm(blended)
                blended = blended / norm if norm > 1e-6 else blended
            else:
                blended = feat if feat is not None else prev.get('feat')
            self.active[cid] = {'t': t, 'pos': (x, z), 'vel': (vx, vz), 'feat': blended}
        else:
            self.active[cid] = {'t': t, 'pos': (x, z), 'vel': (0.0, 0.0), 'feat': feat}

    def update(self, t_now, dets):
        # dets: [(raw_id, x_m, z_m, feat), ...]

        # Phase 0: prune stale lost candidates
        for cid in list(self.lost):
            st = self.lost[cid]
            speed = np.hypot(*st['vel'])
            max_gap = REID_LOW_VEL_GAP_S if speed < REID_LOW_VEL_THRESH else REID_MAX_GAP_S
            if t_now - st['t'] > max_gap:
                del self.lost[cid]

        out_map = {}
        seen    = set()
        new_dets = []

        # Phase 1: sticky aliases for already-known raw IDs
        for raw_id, x_m, z_m, feat in dets:
            if raw_id in self.alias:
                cid = self.alias[raw_id]
                out_map[raw_id] = cid
                self._update_state(cid, t_now, x_m, z_m, feat)
                seen.add(cid)
            else:
                new_dets.append((raw_id, x_m, z_m, feat))

        # Phase 2: Hungarian matching of unknowns against lost tracks
        lost_cids = list(self.lost.keys())
        if lost_cids and new_dets:
            INF = 1e9
            cost = np.full((len(lost_cids), len(new_dets)), INF)

            for i, cid in enumerate(lost_cids):
                st  = self.lost[cid]
                dt  = t_now - st['t']
                px  = st['pos'][0] + st['vel'][0] * dt
                pz  = st['pos'][1] + st['vel'][1] * dt
                thr = self._dist_thresh(px, pz)

                for j, (_, x_m, z_m, feat) in enumerate(new_dets):
                    spatial = float(np.hypot(x_m - px, z_m - pz))
                    if spatial >= thr:
                        continue  # outside gate — leave as INF
                    norm_spatial = spatial / thr

                    # cosine distance [0, 1] — 0 means identical appearance
                    if feat is not None and st.get('feat') is not None:
                        cos_dist = float(1.0 - np.dot(feat, st['feat']))
                        cos_dist = max(0.0, min(1.0, cos_dist))
                    else:
                        cos_dist = 0.5  # neutral when one side has no embedding

                    cost[i, j] = REID_SPATIAL_W * norm_spatial + REID_VISUAL_W * cos_dist

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_cols = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= INF:
                    continue
                cid = lost_cids[r]
                raw_id, x_m, z_m, feat = new_dets[c]
                matched_cols.add(c)
                del self.lost[cid]
                self.alias[raw_id] = cid
                out_map[raw_id]    = cid
                self._update_state(cid, t_now, x_m, z_m, feat)
                seen.add(cid)

            # Unmatched new dets become new canonical tracks
            for j, (raw_id, x_m, z_m, feat) in enumerate(new_dets):
                if j not in matched_cols:
                    self.alias[raw_id] = raw_id
                    out_map[raw_id]    = raw_id
                    self._update_state(raw_id, t_now, x_m, z_m, feat)
                    seen.add(raw_id)
        else:
            # No lost tracks to match against — every new det is a new track
            for raw_id, x_m, z_m, feat in new_dets:
                self.alias[raw_id] = raw_id
                out_map[raw_id]    = raw_id
                self._update_state(raw_id, t_now, x_m, z_m, feat)
                seen.add(raw_id)

        # Phase 3: active not seen this frame → move to lost
        for cid in list(self.active):
            if cid not in seen:
                self.lost[cid] = self.active.pop(cid)

        return out_map


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
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (255, 200, 0))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [poly], isClosed=True, color=(255, 220, 0), thickness=2)


def draw_perspective_grid(frame, H, x_step=7.0, z_step=10.0, alpha=0.4):
    H_inv = np.linalg.inv(H)
    fh, fw = frame.shape[:2]
    color   = (0, 255, 255)
    overlay = frame.copy()
    rect    = (0, 0, fw - 1, fh - 1)

    corners = np.array(
        [[[0., 0.]], [[fw - 1., 0.]], [[fw - 1., fh - 1.]], [[0., fh - 1.]]],
        dtype=np.float32,
    )
    wc    = cv2.perspectiveTransform(corners, H)[:, 0, :]
    x_min = float(wc[:, 0].min()) - x_step
    x_max = float(wc[:, 0].max()) + x_step
    z_min = float(wc[:, 1].min()) - z_step
    z_max = float(wc[:, 1].max()) + z_step

    def to_img(xm, zm):
        p = cv2.perspectiveTransform(np.array([[[xm, zm]]], dtype=np.float32), H_inv)[0, 0]
        return (int(np.clip(round(p[0]), -32767, 32767)),
                int(np.clip(round(p[1]), -32767, 32767)))

    def draw_world_line(x0, z0, x1, z1):
        ok, p0, p1 = cv2.clipLine(rect, to_img(x0, z0), to_img(x1, z1))
        if ok:
            cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)

    xi = np.floor(x_min / x_step) * x_step
    while xi <= x_max:
        draw_world_line(xi, z_min, xi, z_max)
        xi += x_step
    zi = np.floor(z_min / z_step) * z_step
    while zi <= z_max:
        draw_world_line(x_min, zi, x_max, zi)
        zi += z_step

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_speed_panel(frame, speed_history, ema_speed, t_now):
    pw, ph = PANEL_SIZE
    x0 = w - pw - PANEL_MARGIN
    y0 = h - ph - PANEL_MARGIN
    x1, y1 = x0 + pw, y0 + ph

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
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
                          isClosed=False, color=color_for_id(tid),
                          thickness=2, lineType=cv2.LINE_AA)
            active_ids.append(tid)

    lx = x0 + 50
    for tid in active_ids[:5]:
        cv2.rectangle(frame, (lx, y0 + 6), (lx + 10, y0 + 14), color_for_id(tid), -1)
        cv2.putText(frame, f"id{tid}", (lx + 14, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        lx += 50
    if len(active_ids) > 5:
        cv2.putText(frame, f"+{len(active_ids) - 5}", (lx, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


# ─── Main loop ─────────────────────────────────────────────────
stitcher  = ReIDStitcher()
frame_idx = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
print(f"Input : {INPUT_VIDEO}  ({total_frames} frames @ {fps:.1f} fps)")
print(f"Output: {OUTPUT_VIDEO}")
print(f"Device: {_feat_device}  |  writer open: {out.isOpened()}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_now = frame_idx / fps

    results = model.track(frame, classes=[2, 3, 5, 7], persist=True,
                          tracker="bytetrack.yaml", imgsz=960,
                          conf=0.25, iou=0.9,
                          device=0, verbose=False)

    draw_road_overlay(frame, road_poly)
    cv2.polylines(frame, [src_rect], isClosed=True, color=(0, 200, 200), thickness=1)
    draw_perspective_grid(frame, H)

    boxes = results[0].boxes
    ids   = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

    # Phase 1: collect on-road detections; draw off-road as dim ghosts
    on_road = []  # (raw_id, x1, y1, x2, y2, gx, gy, x_m, z_m)
    for box, tid in zip(boxes, ids):
        x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
        gx, gy = (x1b + x2b) // 2, y2b
        if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (100, 100, 100), 1)
            continue
        x_m, z_m = project_to_ground(gx, gy)
        on_road.append((tid, x1b, y1b, x2b, y2b, gx, gy, x_m, z_m))

    # Phase 2: extract visual features for on-road detections (one batched forward pass)
    feat_boxes = [(x1b, y1b, x2b, y2b) for _, x1b, y1b, x2b, y2b, _, _, _, _ in on_road]
    feats = extract_features(frame, feat_boxes)  # list of 512-d arrays

    # Phase 3: stitch raw IDs → canonical IDs (Hungarian + visual + spatial)
    stitch_input = [
        (tid, x_m, z_m, feats[i] if len(feats) > 0 else None)
        for i, (tid, _, _, _, _, _, _, x_m, z_m) in enumerate(on_road)
        if tid is not None
    ]
    remap = stitcher.update(t_now, stitch_input)

    # Phase 4: draw using canonical IDs
    for i, (raw_id, x1b, y1b, x2b, y2b, gx, gy, x_m, z_m) in enumerate(on_road):
        cid = remap.get(raw_id, raw_id)

        sx1, sy1, sx2, sy2 = float(x1b), float(y1b), float(x2b), float(y2b)
        dx1, dy1, dx2, dy2 = int(sx1), int(sy1), int(sx2), int(sy2)
        sgx, sgy = float(gx), float(gy)
        dgx, dgy = int(sgx), int(sgy)
        s_x_m, s_z_m = project_to_ground(sgx, sgy)

        label_parts = []
        if cid is not None:
            history[cid].append((t_now, s_x_m, s_z_m))
            v = speed_from_history(history[cid], SPEED_WINDOW)
            if v is not None:
                prev_ema = ema_speed.get(cid, v)
                ema_speed[cid] = (1 - EMA_ALPHA) * prev_ema + EMA_ALPHA * v
                speed_history[cid].append((t_now, ema_speed[cid]))
                label_parts += [f"id{cid}", f"{v:.0f}km/h"]
            else:
                label_parts.append(f"id{cid}")
            color = color_for_id(cid)
        else:
            color = (0, 255, 0)

        label_parts.append(f"d={s_z_m:.1f}m")
        label = " ".join(label_parts)

        # Ground-point trail (fades from dim oldest → bright newest)
        if cid is not None:
            gp_trace[cid].append((t_now, dgx, dgy))
            trail = [(px, py) for t, px, py in gp_trace[cid] if t_now - t <= TRACE_SEC]
            n = len(trail)
            if n >= 2:
                for k in range(1, n):
                    alpha = k / n
                    seg_color = tuple(int(ch * alpha) for ch in color)
                    cv2.line(frame, trail[k - 1], trail[k], seg_color, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
        cv2.circle(frame, (dgx, dgy), 4, (0, 255, 255), -1)
        cv2.putText(frame, label, (dx1, dy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Prune stale entries (track gone > 2 s)
    for tid in list(ema_speed):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            ema_speed.pop(tid, None)
    for tid in list(bbox_smooth):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            bbox_smooth.pop(tid, None)
    for tid in list(speed_history):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            speed_history.pop(tid, None)
    for tid in list(gp_smooth):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            gp_smooth.pop(tid, None)
    for tid in list(gp_trace):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            gp_trace.pop(tid, None)

    draw_speed_panel(frame, speed_history, ema_speed, t_now)

    out.write(frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames else f"{frame_idx}fr"
        print(f"\r  {pct}  frame {frame_idx}", end="", flush=True)

cap.release()
out.release()
print(f"\nWrote {OUTPUT_VIDEO} ({frame_idx} frames)")
