import os
import colorsys
from collections import deque, defaultdict
import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from ultralytics import RTDETR
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

# ─── Config ────────────────────────────────────────────────────
INPUT_VIDEO  = "../_in/car_100kmh.mp4"
OUTPUT_VIDEO = "out/car_100kmh_exp_4.mp4"
H_PATH       = "H_manual.npy"
SRC_PATH     = "src_manual.npy"
TRACK_PATH   = "track_manual.npy"

VEHICLE_CLASSES      = [2, 3, 5, 7]   # car, motorcycle, bus, truck
CONF                 = 0.40
IOU                  = 0.40
HISTORY_SEC          = 5.0
TRACE_SEC            = 2.5
SPEED_WINDOW         = 0.5
EMA_ALPHA            = 0.1
PANEL_SIZE           = (340, 200)
PANEL_MARGIN         = 20
PANEL_VMAX_KMH_FLOOR = 120
WORLD_MERGE_DIST_M   = 5.0    # ground-plane match radius (metres) — generous since no two vehicles are close in this video
WORLD_MERGE_GAP_S    = 1.5   # how long a lost canonical stays a match candidate

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
    w_association_emb=0.2,   # autotune best: rely heavily on position
    Q_xy_scaling=0.08,
    Q_s_scaling=0.0004,
    delta_t=3,
    inertia=0.2,
    min_hits=1,
    max_age=60,
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
def nms_tracks(tracks, iou_thresh=0.50):
    """Post-tracking NMS: if two track boxes overlap > iou_thresh, drop the
    one with lower confidence so the same vehicle doesn't get two drawn IDs."""
    if len(tracks) == 0:
        return tracks
    boxes = tracks[:, :4]
    scores = tracks[:, 5]
    order = scores.argsort()[::-1]
    keep = []
    suppressed = set()
    for i in range(len(order)):
        idx = order[i]
        if idx in suppressed:
            continue
        keep.append(idx)
        x1a, y1a, x2a, y2a = boxes[idx]
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if jdx in suppressed:
                continue
            x1b, y1b, x2b, y2b = boxes[jdx]
            ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
            ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (x2a - x1a) * (y2a - y1a)
            area_b = (x2b - x1b) * (y2b - y1b)
            union = area_a + area_b - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed.add(jdx)
    return tracks[sorted(keep)]

class WorldMerger:
    """Stable world-coordinate ID layer.

    Two operating modes every frame:
    1. Cross-frame: a new raw ID whose ground position falls within
       WORLD_MERGE_DIST_M of a recently-lost canonical (extrapolated forward)
       is aliased to that canonical.  This catches the alternating-track case
       where the two IDs never appear in the same frame simultaneously.
    2. Same-frame: two simultaneously-active canonicals within
       WORLD_MERGE_DIST_M are merged (older absorbs newer).

    Alias table is sticky — once assigned, a raw ID never changes canonical.
    """

    def __init__(self):
        self.alias      = {}   # raw_tid -> canonical_tid
        self.first_seen = {}   # canonical_tid -> frame_idx
        # canonical_tid -> {'t': float, 'pos': (x,z), 'vel': (vx,vz)}
        self.recent     = {}

    def _extrapolate(self, state, t_now):
        dt = t_now - state['t']
        return (state['pos'][0] + state['vel'][0] * dt,
                state['pos'][1] + state['vel'][1] * dt)

    def _update_vel(self, cid, t, x, z):
        prev = self.recent.get(cid)
        if prev and (t - prev['t']) > 1e-3:
            dt = t - prev['t']
            vx = 0.5 * prev['vel'][0] + 0.5 * (x - prev['pos'][0]) / dt
            vz = 0.5 * prev['vel'][1] + 0.5 * (z - prev['pos'][1]) / dt
        else:
            vx, vz = prev['vel'] if prev else (0.0, 0.0)
        self.recent[cid] = {'t': t, 'pos': (x, z), 'vel': (vx, vz)}

    def update(self, t_now, frame_idx, tid_pos_list):
        """
        tid_pos_list: [(tid, x_m, z_m), ...]
        Returns: {tid: canonical_tid}
        """
        # Prune canonicals that have been lost too long
        for cid in list(self.recent):
            if t_now - self.recent[cid]['t'] > WORLD_MERGE_GAP_S:
                del self.recent[cid]

        # Phase 1 — resolve known aliases
        frame_tracks = []   # (tid, canonical, x_m, z_m)
        new_raw      = []   # (tid, x_m, z_m) — first time we see this raw ID
        for tid, x_m, z_m in tid_pos_list:
            if tid in self.alias:
                frame_tracks.append((tid, self.alias[tid], x_m, z_m))
            else:
                new_raw.append((tid, x_m, z_m))

        # Phase 2 — cross-frame match: new raw IDs vs recently-lost canonicals
        active_cids = {can for _, can, _, _ in frame_tracks}
        used        = set()
        for tid, x_m, z_m in new_raw:
            best_cid, best_dist = None, float('inf')
            for cid, state in self.recent.items():
                if cid in active_cids or cid in used:
                    continue
                px, pz = self._extrapolate(state, t_now)
                dist   = float(np.hypot(x_m - px, z_m - pz))
                if dist < WORLD_MERGE_DIST_M and dist < best_dist:
                    best_cid, best_dist = cid, dist
            if best_cid is not None:
                used.add(best_cid)
                self.alias[tid] = best_cid
                frame_tracks.append((tid, best_cid, x_m, z_m))
                active_cids.add(best_cid)
            else:
                self.alias[tid] = tid
                self.first_seen.setdefault(tid, frame_idx)
                frame_tracks.append((tid, tid, x_m, z_m))
                active_cids.add(tid)

        # Phase 3 — same-frame merge
        seen_can = {}
        for _, can, x_m, z_m in frame_tracks:
            seen_can.setdefault(can, (x_m, z_m))

        redirect = {}
        can_list = list(seen_can.items())
        for i in range(len(can_list)):
            can_i, (xi, zi) = can_list[i]
            root_i = redirect.get(can_i, can_i)
            for j in range(i + 1, len(can_list)):
                can_j, (xj, zj) = can_list[j]
                root_j = redirect.get(can_j, can_j)
                if root_i == root_j:
                    continue
                if np.hypot(xi - xj, zi - zj) < WORLD_MERGE_DIST_M:
                    senior = root_i if self.first_seen.get(root_i, 0) <= self.first_seen.get(root_j, 0) else root_j
                    junior = root_j if senior == root_i else root_i
                    redirect[junior] = senior
                    self.first_seen.pop(junior, None)

        # Apply redirects through alias table
        for k in list(self.alias):
            c = self.alias[k]
            if c in redirect:
                self.alias[k] = redirect[c]

        out = {}
        for tid, can, x_m, z_m in frame_tracks:
            final = redirect.get(can, can)
            self.alias[tid] = final
            self._update_vel(final, t_now, x_m, z_m)
            out[tid] = final
        return out


merger = WorldMerger()


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


def draw_speed_panel(frame, speed_history, ema_speed, t_now, frame_w, frame_h):
    pw, ph = PANEL_SIZE
    x0 = frame_w - pw - PANEL_MARGIN
    y0 = frame_h - ph - PANEL_MARGIN
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
    results = detector(frame, classes=VEHICLE_CLASSES, conf=CONF, iou=IOU,
                       agnostic_nms=True, verbose=False)
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
    tracks = nms_tracks(tracks, iou_thresh=0.40)  # autotune best config

    # ── Draw road overlays ───────────────────────────────────────
    draw_road_overlay(frame, road_poly)
    cv2.polylines(frame, [src_rect], True, (0, 200, 200), 1)

    # ── Pass 1: collect on-road tracks with ground positions ─────
    on_road = []  # (tid, x1, y1, x2, y2, gx, gy, x_m, z_m)
    for row in tracks:
        x1, y1, x2, y2, tid, conf_, cls_, _ = row
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid = int(tid)
        gx  = (x1 + x2) // 2
        gy  = y2
        if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
            continue
        x_m, z_m = project_to_ground(gx, gy)
        on_road.append((tid, x1, y1, x2, y2, gx, gy, x_m, z_m))

    # ── World-coordinate merge: collapse same-vehicle duplicate IDs ─
    world_remap = merger.update(
        t_now, frame_idx,
        [(tid, x_m, z_m) for tid, _, _, _, _, _, _, x_m, z_m in on_road],
    )

    # ── Pass 2: draw with canonical IDs ──────────────────────────
    # Guard: skip duplicate canonical IDs within this frame (both tracks
    # mapped to the same canonical — only draw once, prefer first occurrence)
    drawn_cids = set()
    for tid, x1, y1, x2, y2, gx, gy, x_m, z_m in on_road:
        cid = world_remap[tid]
        if cid in drawn_cids:
            continue
        drawn_cids.add(cid)

        color = color_for_id(cid)

        history[cid].append((t_now, x_m, z_m))
        v = speed_from_history(history[cid], SPEED_WINDOW)
        if v is not None:
            prev_ema       = ema_speed.get(cid, v)
            ema_speed[cid] = (1 - EMA_ALPHA) * prev_ema + EMA_ALPHA * v
            speed_history[cid].append((t_now, ema_speed[cid]))
            label = f"id{cid}  {v:.0f}km/h  d={z_m:.1f}m"
        else:
            label = f"id{cid}  d={z_m:.1f}m"

        gp_trace[cid].append((t_now, gx, gy))
        trail = [(px, py) for t, px, py in gp_trace[cid] if t_now - t <= TRACE_SEC]
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

    draw_speed_panel(frame, speed_history, ema_speed, t_now, w, h)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 30 == 0:
        pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames else f"{frame_idx}fr"
        print(f"\r  {pct}  t={t_now:.1f}s", end="", flush=True)

cap.release()
writer.release()
print(f"\nWrote {OUTPUT_VIDEO} ({frame_idx} frames)")
