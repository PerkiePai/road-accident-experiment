import os
import colorsys
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ─── Config ────────────────────────────────────────────────────
INPUT_VIDEO   = "../_in/car_100kmh.mp4"
model = YOLO("yolo11n.pt")
OUTPUT_VIDEO  = "out/car_100kmh_tracked.mp4"
H_PATH        = "H_manual.npy"
SRC_PATH      = "src_manual.npy"
TRACK_PATH    = "track_manual.npy"
HISTORY_SEC   = 5.0          # how much speed history to plot
SPEED_WINDOW  = 0.5          # seconds, finite-diff window for speed
EMA_ALPHA     = 0.1          # smoothing factor for per-track speed
BBOX_EMA      = 0.6          # bbox smoothing — visual only
GP_EMA        = 0.75         # ground-point (tracking point) smoothing — drives projection & speed
PANEL_SIZE    = (340, 200)   # (w, h) of the speed panel
PANEL_MARGIN  = 20
PANEL_VMAX_KMH_FLOOR = 120  # y-axis upper bound floor

# ─── Re-ID stitcher (Layer B) ─────────────────────────────────
REID_DIST_BASE_M    = 2.0    # match radius at z=0
REID_DIST_PER_M     = 0.10   # +meters of slack per meter of depth
REID_MAX_GAP_S      = 2.0    # max time a lost track stays a candidate
REID_LOW_VEL_GAP_S  = 1.0    # tighter window for ~stopped cars
REID_LOW_VEL_THRESH = 1.0    # m/s — below this counts as "stopped"

# ─── Calibration ───────────────────────────────────────────────
if not (os.path.exists(H_PATH) and os.path.exists(SRC_PATH) and os.path.exists(TRACK_PATH)):
    raise SystemExit(
        f"Missing calibration files ({H_PATH}, {SRC_PATH}, {TRACK_PATH}).\n"
        f"Run `python manual_calibrate.py` first."
    )

H = np.load(H_PATH)
src_rect   = np.load(SRC_PATH).astype(np.int32)    # 4-pt homography rectangle (reference)
road_poly  = np.load(TRACK_PATH).astype(np.int32)  # N-pt tracking region

# ─── Video I/O ─────────────────────────────────────────────────

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Could not open {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ─── State ─────────────────────────────────────────────────────
# canonical_id -> deque of (t_sec, x_m, z_m)
history = defaultdict(lambda: deque(maxlen=int(fps * HISTORY_SEC)))
# canonical_id -> EMA km/h
ema_speed = {}
# canonical_id -> deque of (t_sec, ema_kmh) for plotting
speed_history = defaultdict(lambda: deque(maxlen=int(fps * HISTORY_SEC)))
# canonical_id -> smoothed (x1, y1, x2, y2) as floats
bbox_smooth = {}
# canonical_id -> smoothed (gx, gy) ground point (the actual tracking target)
gp_smooth = {}
stitcher = None  # instantiated after class is defined


def color_for_id(tid: int):
    h_ = (tid * 0.61803398875) % 1.0  # golden-ratio hue spread
    r, g, b = colorsys.hsv_to_rgb(h_, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def project_to_ground(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    xz = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xz[0]), float(xz[1])  # x_m, z_m


class ReIDStitcher:
    """Re-associate fresh ByteTrack IDs to recently-lost canonical IDs using
    metric ground-plane position + linear-velocity extrapolation.

    Workflow per frame: call .update(t_now, [(raw_id, x_m, z_m), ...]).
    Returns {raw_id: canonical_id}. Once a raw_id is mapped, the mapping
    sticks for future frames (sticky alias).
    """

    def __init__(self):
        self.alias = {}    # raw_id -> canonical_id
        self.active = {}   # canonical_id -> {t, pos, vel}
        self.lost = {}     # canonical_id -> {t, pos, vel}

    def _dist_thresh(self, x_m, z_m):
        return REID_DIST_BASE_M + REID_DIST_PER_M * (max(z_m, 0.0) + abs(x_m))

    def _update_state(self, cid, t, x, z):
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
            self.active[cid] = {'t': t, 'pos': (x, z), 'vel': (vx, vz)}
        else:
            self.active[cid] = {'t': t, 'pos': (x, z), 'vel': (0.0, 0.0)}

    def update(self, t_now, dets):
        # Phase 0: prune stale lost candidates
        for cid in list(self.lost):
            st = self.lost[cid]
            speed = np.hypot(*st['vel'])
            max_gap = REID_LOW_VEL_GAP_S if speed < REID_LOW_VEL_THRESH else REID_MAX_GAP_S
            if t_now - st['t'] > max_gap:
                del self.lost[cid]

        out = {}
        seen = set()
        new_dets = []

        # Phase 1: sticky aliases for known raw IDs
        for raw_id, x_m, z_m in dets:
            if raw_id in self.alias:
                cid = self.alias[raw_id]
                out[raw_id] = cid
                self._update_state(cid, t_now, x_m, z_m)
                seen.add(cid)
            else:
                new_dets.append((raw_id, x_m, z_m))

        # Phase 2: greedy match unknowns against lost tracks
        used = set()
        for raw_id, x_m, z_m in new_dets:
            best_cid, best_dist = None, float('inf')
            for cid, st in self.lost.items():
                if cid in used:
                    continue
                dt = t_now - st['t']
                px = st['pos'][0] + st['vel'][0] * dt
                pz = st['pos'][1] + st['vel'][1] * dt
                dist = float(np.hypot(x_m - px, z_m - pz))
                if dist < self._dist_thresh(x_m, z_m) and dist < best_dist:
                    best_cid, best_dist = cid, dist
            if best_cid is not None:
                used.add(best_cid)
                del self.lost[best_cid]
                self.alias[raw_id] = best_cid
                out[raw_id] = best_cid
                self._update_state(best_cid, t_now, x_m, z_m)
                seen.add(best_cid)
            else:
                self.alias[raw_id] = raw_id
                out[raw_id] = raw_id
                self._update_state(raw_id, t_now, x_m, z_m)
                seen.add(raw_id)

        # Phase 3: active not seen → move to lost
        for cid in list(self.active):
            if cid not in seen:
                self.lost[cid] = self.active.pop(cid)

        return out


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
    v_ms = np.hypot(x_now - x_old, z_now - z_old) / dt
    return v_ms * 3.6  # km/h


def draw_road_overlay(frame, poly):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], (255, 200, 0))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [poly], isClosed=True, color=(255, 220, 0), thickness=2)


def draw_perspective_grid(frame, H, x_step=7.0, z_step=10.0, alpha=0.4):
    H_inv = np.linalg.inv(H)
    fh, fw = frame.shape[:2]
    color = (0, 255, 255)  # BGR yellow
    overlay = frame.copy()
    rect = (0, 0, fw - 1, fh - 1)

    # Visible world bounds from frame corners
    corners = np.array(
        [[[0., 0.]], [[fw - 1., 0.]], [[fw - 1., fh - 1.]], [[0., fh - 1.]]],
        dtype=np.float32,
    )
    wc = cv2.perspectiveTransform(corners, H)[:, 0, :]
    x_min = float(wc[:, 0].min()) - x_step
    x_max = float(wc[:, 0].max()) + x_step
    z_min = float(wc[:, 1].min()) - z_step
    z_max = float(wc[:, 1].max()) + z_step

    def to_img(xm, zm):
        p = cv2.perspectiveTransform(
            np.array([[[xm, zm]]], dtype=np.float32), H_inv
        )[0, 0]
        return (int(np.clip(round(p[0]), -32767, 32767)),
                int(np.clip(round(p[1]), -32767, 32767)))

    def draw_world_line(x0, z0, x1, z1):
        ok, p0, p1 = cv2.clipLine(rect, to_img(x0, z0), to_img(x1, z1))
        if ok:
            cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)

    # Vertical lines every x_step metres (constant X)
    xi = np.floor(x_min / x_step) * x_step
    while xi <= x_max:
        draw_world_line(xi, z_min, xi, z_max)
        xi += x_step

    # Horizontal lines every z_step metres (constant Z)
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

    # background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

    # plot area
    pad_l, pad_r, pad_t, pad_b = 38, 8, 22, 18
    px0, py0 = x0 + pad_l, y0 + pad_t
    px1, py1 = x1 - pad_r, y1 - pad_b
    plot_w, plot_h = px1 - px0, py1 - py0

    # determine y-max from current EMA speeds
    vmax = PANEL_VMAX_KMH_FLOOR
    if ema_speed:
        vmax = max(vmax, max(ema_speed.values()) * 1.2)

    # grid + y-ticks
    cv2.putText(frame, "km/h", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    n_ticks = 4
    for i in range(n_ticks + 1):
        v = vmax * i / n_ticks
        yy = py1 - int(plot_h * i / n_ticks)
        cv2.line(frame, (px0, yy), (px1, yy), (70, 70, 70), 1)
        cv2.putText(frame, f"{int(v)}", (x0 + 6, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # time axis labels
    cv2.putText(frame, f"-{HISTORY_SEC:.0f}s", (px0, py1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "0", (px1 - 8, py1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # one polyline per active track — pre-EMA-smoothed series
    active_ids = []
    for tid, sh in speed_history.items():
        if len(sh) < 2:
            continue
        if t_now - sh[-1][0] > 1.0:
            continue  # stale (track gone)
        poly_pts = []
        for t, v in sh:
            age = t_now - t
            if age > HISTORY_SEC:
                continue
            fx = px1 - int(plot_w * (age / HISTORY_SEC))
            fy = py1 - int(plot_h * min(v / vmax, 1.0))
            poly_pts.append((fx, fy))
        if len(poly_pts) >= 2:
            cv2.polylines(frame, [np.array(poly_pts, dtype=np.int32)],
                          isClosed=False, color=color_for_id(tid),
                          thickness=2, lineType=cv2.LINE_AA)
            active_ids.append(tid)

    # legend (top of panel)
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
stitcher = ReIDStitcher()
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t_now = frame_idx / fps

    results = model.track(frame, classes=[2, 3, 5, 7], persist=True,
                          tracker="bytetrack.yaml", imgsz=960,
                          device=0, verbose=False)

    draw_road_overlay(frame, road_poly)
    cv2.polylines(frame, [src_rect], isClosed=True, color=(0, 200, 200), thickness=1)
    draw_perspective_grid(frame, H)

    boxes = results[0].boxes
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

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

    # Phase 2: stitch raw IDs → canonical IDs using metric position
    stitch_input = [(tid, x_m, z_m)
                    for tid, _, _, _, _, _, _, x_m, z_m in on_road
                    if tid is not None]
    remap = stitcher.update(t_now, stitch_input)

    # Phase 3: draw using canonical IDs
    for raw_id, x1b, y1b, x2b, y2b, gx, gy, x_m, z_m in on_road:
        cid = remap.get(raw_id, raw_id)  # None stays None

        # Smooth bbox per canonical id (visual only; metric flow uses raw)
        # if cid is not None and cid in bbox_smooth:
        #     px1, py1_, px2, py2 = bbox_smooth[cid]
        #     sx1 = BBOX_EMA * px1 + (1 - BBOX_EMA) * x1b
        #     sy1 = BBOX_EMA * py1_ + (1 - BBOX_EMA) * y1b
        #     sx2 = BBOX_EMA * px2 + (1 - BBOX_EMA) * x2b
        #     sy2 = BBOX_EMA * py2 + (1 - BBOX_EMA) * y2b
        # else:
        #     sx1, sy1, sx2, sy2 = float(x1b), float(y1b), float(x2b), float(y2b)
        # if cid is not None:
        #     bbox_smooth[cid] = (sx1, sy1, sx2, sy2)
        sx1, sy1, sx2, sy2 = float(x1b), float(y1b), float(x2b), float(y2b)

        dx1, dy1, dx2, dy2 = int(sx1), int(sy1), int(sx2), int(sy2)

        # Smooth ground point (the actual tracking target) per cid — separate from bbox
        # if cid is not None and cid in gp_smooth:
        #     pgx, pgy = gp_smooth[cid]
        #     sgx = GP_EMA * pgx + (1 - GP_EMA) * gx
        #     sgy = GP_EMA * pgy + (1 - GP_EMA) * gy
        # else:
        #     sgx, sgy = float(gx), float(gy)
        # if cid is not None:
        #     gp_smooth[cid] = (sgx, sgy)
        sgx, sgy = float(gx), float(gy)
        dgx, dgy = int(sgx), int(sgy)

        # Project smoothed ground point → smoothed metric coords
        s_x_m, s_z_m = project_to_ground(sgx, sgy)

        label_parts = []
        if cid is not None:
            history[cid].append((t_now, s_x_m, s_z_m))
            v = speed_from_history(history[cid], SPEED_WINDOW)
            if v is not None:
                prev = ema_speed.get(cid, v)
                ema_speed[cid] = (1 - EMA_ALPHA) * prev + EMA_ALPHA * v
                speed_history[cid].append((t_now, ema_speed[cid]))
                label_parts.append(f"id{cid}")
                label_parts.append(f"{v:.0f}km/h")  # bbox shows real (instantaneous)
            else:
                label_parts.append(f"id{cid}")
            color = color_for_id(cid)
        else:
            color = (0, 255, 0)

        label_parts.append(f"d={s_z_m:.1f}m")
        label = " ".join(label_parts)

        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
        cv2.circle(frame, (dgx, dgy), 4, (0, 255, 255), -1)
        cv2.putText(frame, label, (dx1, dy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # prune stale ema + bbox + speed_history entries (track gone for > 2s)
    for tid in list(ema_speed.keys()):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            ema_speed.pop(tid, None)
    for tid in list(bbox_smooth.keys()):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            bbox_smooth.pop(tid, None)
    for tid in list(speed_history.keys()):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            speed_history.pop(tid, None)
    for tid in list(gp_smooth.keys()):
        if not history[tid] or t_now - history[tid][-1][0] > 2.0:
            gp_smooth.pop(tid, None)

    draw_speed_panel(frame, speed_history, ema_speed, t_now)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Wrote {OUTPUT_VIDEO} ({frame_idx} frames)")
