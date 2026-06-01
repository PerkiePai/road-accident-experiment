"""
Edge-clip diagnostic.

Question this answers: are the speed errors actually caused by bbox clipping at
the frame border (the "partial car -> shrunk box -> wrong ground point" theory),
or are they just RT-DETR's general frame-to-frame box jitter happening everywhere?

It re-runs the SAME detect/track/WorldMerger pipeline as detect.py, and for every
on-road track records per frame:
    frame_idx, t, canonical_id, x_m, z_m, gx, gy, box edges, edge_clipped?

"edge_clipped" here means the part of the box that feeds the GROUND point is
touching the image border:
    bottom (y2 ~ H)  -> corrupts gy directly
    left   (x1 ~ 0)  -> corrupts the bottom-centre x
    right  (x2 ~ W)  -> corrupts the bottom-centre x
(top clipping is ignored — it does not move the ground point.)

Outputs:
    out/edgeclip_trace.csv          full per-sample table
    out/edgeclip_trace.png          per-track depth(t) + speed(t), clips marked red
    stdout summary                  % frames clipped per track, and what fraction of
                                    speed spikes coincide (+/-2 frames) with a clip.

Run:
    conda activate car-detection
    cd D:/intern/NT/project/road-accident/experiment-new-model
    python diag_edgeclip.py
"""

import os, csv, logging
from collections import defaultdict, deque

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("boxmot").setLevel(logging.CRITICAL)
from ultralytics import RTDETR
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

# ─── Config (kept in sync with detect.py) ──────────────────────
INPUT_VIDEO = "../_in/car_100kmh.mp4"
H_PATH      = "H_manual.npy"
SRC_PATH    = "src_manual.npy"
TRACK_PATH  = "track_manual.npy"

VEHICLE_CLASSES    = [2, 3, 5, 7]
CONF               = 0.40
IOU                = 0.40
SPEED_WINDOW       = 0.5
WORLD_MERGE_DIST_M = 5.0
WORLD_SAME_FRAME_M = 1.5
WORLD_MERGE_GAP_S  = 1.5

EDGE_EPS_PX = 3        # how close to the border counts as "clipped"
TOP_K       = 6        # number of longest tracks to plot
SPIKE_KMH   = 15.0     # |Δ windowed-speed| between samples that counts as a "spike"
SPIKE_NEAR  = 2        # a spike "coincides" if a clip is within ±this many frames

OUT_CSV = "out/edgeclip_trace.csv"
OUT_PNG = "out/edgeclip_trace.png"

# ─── Calibration ───────────────────────────────────────────────
for p in (H_PATH, SRC_PATH, TRACK_PATH):
    if not os.path.exists(p):
        raise SystemExit(f"Missing calibration file: {p}\nRun manual_calibrate.py first.")
H         = np.load(H_PATH)
road_poly = np.load(TRACK_PATH).astype(np.int32)
H_inv     = np.linalg.inv(H)

os.makedirs("out", exist_ok=True)


def project_to_ground(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    xz = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xz[0]), float(xz[1])


# ─── nms_tracks (verbatim from detect.py) ──────────────────────
def nms_tracks(tracks, iou_thresh=0.40):
    if len(tracks) == 0:
        return tracks
    boxes = tracks[:, :4]
    scores = tracks[:, 5]
    order = scores.argsort()[::-1]
    keep, suppressed = [], set()
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
            union = (x2a-x1a)*(y2a-y1a) + (x2b-x1b)*(y2b-y1b) - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed.add(jdx)
    return tracks[sorted(keep)]


# ─── WorldMerger (verbatim from detect.py) ─────────────────────
class WorldMerger:
    def __init__(self, H_inv=None, road_poly=None):
        self.alias = {}
        self.alias_last_seen = {}
        self.first_seen = {}
        self.recent = {}
        self._H_inv = H_inv
        self._road_poly = road_poly

    def _extrapolate(self, state, t_now):
        dt = t_now - state['t']
        return (state['pos'][0] + state['vel'][0] * dt,
                state['pos'][1] + state['vel'][1] * dt)

    def _in_road(self, x_m, z_m):
        if self._H_inv is None or self._road_poly is None:
            return True
        px = cv2.perspectiveTransform(
            np.array([[[x_m, z_m]]], dtype=np.float32), self._H_inv)[0, 0]
        return cv2.pointPolygonTest(
            self._road_poly, (float(px[0]), float(px[1])), False) >= 0

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
        for cid in list(self.recent):
            st = self.recent[cid]
            if t_now - st['t'] > WORLD_MERGE_GAP_S:
                del self.recent[cid]
                continue
            px, pz = self._extrapolate(st, t_now)
            if not self._in_road(px, pz):
                del self.recent[cid]

        frame_tracks, new_raw = [], []
        for tid, x_m, z_m in tid_pos_list:
            if tid in self.alias:
                can = self.alias[tid]
                last_seen = self.alias_last_seen.get(tid, 0)
                still_fresh = t_now - last_seen <= WORLD_MERGE_GAP_S
                still_active = can in self.recent
                behind = False
                if still_fresh and still_active:
                    state = self.recent[can]
                    px, pz = self._extrapolate(state, t_now)
                    vel_speed = float(np.hypot(*state['vel']))
                    if vel_speed > 1.0:
                        dot = ((x_m - px) * state['vel'][0] +
                               (z_m - pz) * state['vel'][1]) / vel_speed
                        if dot < -WORLD_MERGE_DIST_M:
                            behind = True
                if still_fresh and still_active and not behind:
                    self.alias_last_seen[tid] = t_now
                    frame_tracks.append((tid, can, x_m, z_m))
                else:
                    del self.alias[tid]
                    new_raw.append((tid, x_m, z_m))
            else:
                new_raw.append((tid, x_m, z_m))

        active_cids = {can for _, can, _, _ in frame_tracks}
        used = set()
        for tid, x_m, z_m in new_raw:
            best_cid, best_dist = None, float('inf')
            for cid, state in self.recent.items():
                if cid in active_cids or cid in used:
                    continue
                px, pz = self._extrapolate(state, t_now)
                dist = float(np.hypot(x_m - px, z_m - pz))
                vel_speed = float(np.hypot(*state['vel']))
                dot = None
                if vel_speed > 1.0:
                    dot = ((x_m - px) * state['vel'][0] +
                           (z_m - pz) * state['vel'][1]) / vel_speed
                if dist >= WORLD_MERGE_DIST_M or dist >= best_dist:
                    continue
                if dot is not None and dot < -WORLD_MERGE_DIST_M:
                    continue
                best_cid, best_dist = cid, dist
            if best_cid is not None:
                used.add(best_cid)
                self.alias[tid] = best_cid
                self.alias_last_seen[tid] = t_now
                frame_tracks.append((tid, best_cid, x_m, z_m))
                active_cids.add(best_cid)
            else:
                self.alias[tid] = tid
                self.alias_last_seen[tid] = t_now
                self.first_seen.setdefault(tid, frame_idx)
                frame_tracks.append((tid, tid, x_m, z_m))
                active_cids.add(tid)

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
                if np.hypot(xi - xj, zi - zj) < WORLD_SAME_FRAME_M:
                    senior = root_i if self.first_seen.get(root_i, 0) <= self.first_seen.get(root_j, 0) else root_j
                    junior = root_j if senior == root_i else root_i
                    redirect[junior] = senior
                    self.first_seen.pop(junior, None)
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


# ─── Feature extractor (ResNet-18) ─────────────────────────────
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

_feat_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_feat_model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
_feat_model.fc = torch.nn.Identity()
_feat_model.eval().to(_feat_device)
_feat_tf = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((128, 64)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features(frame_bgr, boxes_xyxy):
    if not boxes_xyxy:
        return np.empty((0, 512), dtype=np.float32)
    fh, fw = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crops = []
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


# ─── Run pipeline + record trace ───────────────────────────────
detector = RTDETR("rtdetr-l.pt")
tracker = DeepOcSort(
    reid_model=None, embedding_off=False, w_association_emb=0.2,
    Q_xy_scaling=0.08, Q_s_scaling=0.0004, delta_t=3, inertia=0.2,
    min_hits=1, max_age=60,
)
merger = WorldMerger(H_inv=H_inv, road_poly=road_poly)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {INPUT_VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Hgt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input : {INPUT_VIDEO}  ({total_frames} frames @ {fps:.1f} fps, {W}x{Hgt})")
print(f"Device: {_feat_device}")

# rows: dicts; trace[cid] = list of (frame_idx, t, x_m, z_m, clipped_ground)
rows = []
trace = defaultdict(list)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    t_now = frame_idx / fps

    results = detector(frame, classes=VEHICLE_CLASSES, conf=CONF, iou=IOU,
                       agnostic_nms=True, verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        cf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cl = boxes.cls.cpu().numpy().reshape(-1, 1)
        dets = np.hstack([xyxy, cf, cl]).astype(np.float32)
        embs = extract_features(frame, xyxy.tolist())
    else:
        dets = np.empty((0, 6), dtype=np.float32)
        embs = np.empty((0, 512), dtype=np.float32)

    tracks = tracker.update(dets, frame, embs)
    tracks = nms_tracks(tracks, iou_thresh=0.40)

    on_road = []  # (tid, x1,y1,x2,y2, gx,gy, x_m,z_m)
    for row in tracks:
        x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gx, gy = (x1 + x2) // 2, y2
        if cv2.pointPolygonTest(road_poly, (float(gx), float(gy)), False) < 0:
            continue
        x_m, z_m = project_to_ground(gx, gy)
        on_road.append((tid, x1, y1, x2, y2, gx, gy, x_m, z_m))

    remap = merger.update(t_now, frame_idx,
                          [(tid, x_m, z_m) for tid, _, _, _, _, _, _, x_m, z_m in on_road])

    drawn = set()
    for tid, x1, y1, x2, y2, gx, gy, x_m, z_m in on_road:
        cid = remap[tid]
        if cid in drawn:
            continue
        drawn.add(cid)

        clip_bottom = y2 >= Hgt - EDGE_EPS_PX
        clip_left   = x1 <= EDGE_EPS_PX
        clip_right  = x2 >= W - EDGE_EPS_PX
        clip_top    = y1 <= EDGE_EPS_PX
        clip_ground = clip_bottom or clip_left or clip_right

        rows.append(dict(
            frame=frame_idx, t=round(t_now, 4), cid=cid,
            x_m=round(x_m, 4), z_m=round(z_m, 4), gx=gx, gy=gy,
            x1=x1, y1=y1, x2=x2, y2=y2,
            clip_bottom=int(clip_bottom), clip_left=int(clip_left),
            clip_right=int(clip_right), clip_top=int(clip_top),
            clip_ground=int(clip_ground),
        ))
        trace[cid].append((frame_idx, t_now, x_m, z_m, clip_ground))

    frame_idx += 1
    if frame_idx % 30 == 0:
        pct = f"{frame_idx/total_frames*100:.1f}%" if total_frames else f"{frame_idx}fr"
        print(f"\r  {pct}  t={t_now:.1f}s", end="", flush=True)

cap.release()
print()

# ─── Write CSV ─────────────────────────────────────────────────
fields = ["frame", "t", "cid", "x_m", "z_m", "gx", "gy",
          "x1", "y1", "x2", "y2",
          "clip_bottom", "clip_left", "clip_right", "clip_top", "clip_ground"]
with open(OUT_CSV, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=fields)
    wcsv.writeheader()
    wcsv.writerows(rows)
print(f"Wrote {OUT_CSV}  ({len(rows)} samples)")


# ─── Windowed speed (same finite-diff scheme as detect.py) ─────
def windowed_speed_series(samples, window_sec):
    """samples: list of (frame, t, x_m, z_m, clip). Returns list of (t, kmh)."""
    out = []
    hist = deque()
    for (fi, t, x, z, clip) in samples:
        hist.append((t, x, z))
        while hist and t - hist[0][0] > max(window_sec, 1e-6) and len(hist) > 1:
            # keep at least one sample older than window for the diff
            if t - hist[1][0] > window_sec:
                hist.popleft()
            else:
                break
        if len(hist) >= 2:
            t0, x0, z0 = hist[0]
            dt = t - t0
            kmh = np.hypot(x - x0, z - z0) / dt * 3.6 if dt > 1e-3 else None
        else:
            kmh = None
        out.append((t, kmh))
    return out


# ─── Coincidence stats ─────────────────────────────────────────
order = sorted(trace.keys(), key=lambda c: len(trace[c]), reverse=True)
print(f"\nTracks: {len(order)} canonical IDs.  Per-track clip rate & spike coincidence:")
print(f"{'cid':>5} {'frames':>7} {'clipped':>8} {'clip%':>6} {'spikes':>7} {'near_clip':>9} {'coincide%':>9}")

tot_spikes = tot_coincide = 0
for cid in order:
    s = trace[cid]
    n = len(s)
    nclip = sum(1 for r in s if r[4])
    clip_frames = {r[0] for r in s if r[4]}
    sp = windowed_speed_series(s, SPEED_WINDOW)
    spikes = 0
    coincide = 0
    prev = None
    for (fi, _, _, _, _), (_, kmh) in zip(s, sp):
        if kmh is None:
            prev = None
            continue
        if prev is not None and abs(kmh - prev) > SPIKE_KMH:
            spikes += 1
            if any((fi + d) in clip_frames for d in range(-SPIKE_NEAR, SPIKE_NEAR + 1)):
                coincide += 1
        prev = kmh
    tot_spikes += spikes
    tot_coincide += coincide
    cpct = 100.0 * nclip / n if n else 0
    copct = 100.0 * coincide / spikes if spikes else 0
    print(f"{cid:>5} {n:>7} {nclip:>8} {cpct:>5.0f}% {spikes:>7} {coincide:>9} {copct:>8.0f}%")

print(f"\nTOTAL spikes (|Δspeed|>{SPIKE_KMH:.0f} km/h between samples): {tot_spikes}")
if tot_spikes:
    print(f"  within ±{SPIKE_NEAR} frames of an edge clip: {tot_coincide} "
          f"({100.0*tot_coincide/tot_spikes:.0f}%)")
    print("  -> high % supports the edge-clip theory; low % means spikes are "
          "general box jitter, not clipping.")

# ─── Exit-drop analysis (the effect actually observed) ─────────
# When a car drives out of view its box pins to the frame border, the ground
# point freezes, and speed falls. A spike test misses this (it is gradual under
# smoothing), so here we directly compare each track's speed in its final frames
# to its steady-state speed, and report whether those final frames were clipped.
EXIT_N    = 4      # how many final frames count as "exit"
DROP_FRAC = 0.6    # exit speed below this fraction of median = a drop
print(f"\nExit-drop check (last {EXIT_N} frames vs median speed):")
print(f"{'cid':>5} {'frames':>7} {'med_kmh':>8} {'exit_kmh':>9} {'drop?':>6} {'exit_clipped':>13}")
n_drop = n_drop_clipped = 0
for cid in order:
    s = trace[cid]
    if len(s) < EXIT_N + 6:
        continue
    sp = windowed_speed_series(s, SPEED_WINDOW)
    vals = [v for _, v in sp if v is not None]
    if len(vals) < EXIT_N + 1:
        continue
    med      = float(np.median(vals))
    exit_v   = [v for _, v in sp[-EXIT_N:] if v is not None]
    exit_kmh = float(np.mean(exit_v)) if exit_v else float('nan')
    exit_clipped = any(r[4] for r in s[-EXIT_N:])
    is_drop  = med > 5 and exit_kmh < DROP_FRAC * med
    if is_drop:
        n_drop += 1
        if exit_clipped:
            n_drop_clipped += 1
    print(f"{cid:>5} {len(s):>7} {med:>8.0f} {exit_kmh:>9.0f} "
          f"{('YES' if is_drop else '-'):>6} {('YES' if exit_clipped else '-'):>13}")
print(f"\n{n_drop} tracks drop >{(1-DROP_FRAC)*100:.0f}% at exit; "
      f"of those, {n_drop_clipped} were edge-clipped in their final frames.")

# ─── Plot top-K tracks ─────────────────────────────────────────
K = min(TOP_K, len(order))
if K == 0:
    raise SystemExit("No on-road tracks recorded — nothing to plot.")

fig, axes = plt.subplots(K, 2, figsize=(13, 2.6 * K), squeeze=False)
for r, cid in enumerate(order[:K]):
    s = trace[cid]
    ts = [x[1] for x in s]
    zs = [x[3] for x in s]
    xs = [x[2] for x in s]
    clip_t = [x[1] for x in s if x[4]]
    clip_z = [x[3] for x in s if x[4]]

    ax = axes[r][0]
    ax.plot(ts, zs, '-', color='tab:blue', lw=1.2, label='z_m (depth)')
    ax.plot(ts, xs, '-', color='tab:green', lw=1.0, alpha=0.7, label='x_m (lateral)')
    ax.scatter(clip_t, clip_z, color='red', s=18, zorder=5, label='edge-clipped')
    ax.set_ylabel(f"id{cid}\nworld (m)")
    ax.grid(alpha=0.3)
    if r == 0:
        ax.legend(fontsize=7, loc='upper right')

    sp = windowed_speed_series(s, SPEED_WINDOW)
    sp_t = [t for t, v in sp if v is not None]
    sp_v = [v for t, v in sp if v is not None]
    clip_set = {x[0] for x in s if x[4]}
    ax2 = axes[r][1]
    ax2.plot(sp_t, sp_v, '-', color='tab:orange', lw=1.2, label='windowed speed')
    csp_t = [t for (fi, t, _, _, _), (_, v) in zip(s, sp)
             if v is not None and fi in clip_set]
    csp_v = [v for (fi, _, _, _, _), (_, v) in zip(s, sp)
             if v is not None and fi in clip_set]
    ax2.scatter(csp_t, csp_v, color='red', s=18, zorder=5, label='edge-clipped')
    ax2.set_ylabel("km/h")
    ax2.grid(alpha=0.3)
    if r == 0:
        ax2.legend(fontsize=7, loc='upper right')

axes[-1][0].set_xlabel("t (s)")
axes[-1][1].set_xlabel("t (s)")
fig.suptitle(f"Edge-clip diagnostic — {os.path.basename(INPUT_VIDEO)}  "
             f"(red = ground point's box edge touching frame border)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_PNG, dpi=110)
print(f"\nWrote {OUT_PNG}")
