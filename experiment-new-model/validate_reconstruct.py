"""
Validate the "reconstruct the true bottom from the visible top" idea (Idea 3)
on frames where the TRUE bottom is known.

Reads out/edgeclip_trace.csv (produced by diag_edgeclip.py). For each fully
visible track segment we pick many clip-start points; from that point on we
pretend the bottom edge is off-screen and must reconstruct it as

    y2_recon = y1_observed + H_box_predicted

where H_box (= y2 - y1) is extrapolated from history BEFORE the clip point
(extrapolated against time, never against the now-invalid depth). We compare
y2_recon to the real y2 at horizon k = 1, 2, ... frames after the clip starts,
and convert the error to world metres and to km/h speed error.

Three extrapolation models are compared:
    hold    H_box held at its last pre-clip value (naive baseline)
    lin     linear fit of H_box(t) over last L frames
    invlin  linear fit of 1/H_box(t)  (perspective-correct: 1/H_box ~ depth,
            which falls ~linearly in time for constant speed)

Only APPROACHING segments (H_box growing) are used, because a bottom clip
physically means the car is coming toward the camera and exiting the bottom —
that is the scenario the reconstruction is meant to serve.

Run:
    conda activate car-detection
    cd D:/intern/NT/project/road-accident/experiment-new-model
    python validate_reconstruct.py
"""

import os, csv
from collections import defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV   = "out/edgeclip_trace.csv"
H     = np.load("H_manual.npy")
FPS   = 29.9            # car_100kmh; only used to label, dt comes from t column
L     = 8               # history frames used to fit the extrapolation
K     = 12              # reconstruction horizon (frames after clip start)
W_SEC = 0.5             # speed window (matches detect.py SPEED_WINDOW)
OUT_PNG = "out/reconstruct_validation.png"

if not os.path.exists(CSV):
    raise SystemExit(f"{CSV} not found — run diag_edgeclip.py first.")


def proj(cx, y2):
    pt = np.array([[[float(cx), float(y2)]]], dtype=np.float32)
    w = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(w[0]), float(w[1])


# ─── Load trace, group by track, keep only fully-visible rows ──
by_cid = defaultdict(list)
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        by_cid[int(r["cid"])].append(dict(
            frame=int(r["frame"]), t=float(r["t"]),
            gx=float(r["gx"]), y1=float(r["y1"]), y2=float(r["y2"]),
            x_m=float(r["x_m"]), z_m=float(r["z_m"]),
            clip=int(r["clip_ground"]),
        ))

# Build longest fully-visible, frame-contiguous runs per track.
runs = []
for cid, rows in by_cid.items():
    rows.sort(key=lambda d: d["frame"])
    cur = []
    for d in rows:
        if d["clip"] == 0 and (not cur or d["frame"] == cur[-1]["frame"] + 1):
            cur.append(d)
        else:
            if len(cur) >= L + K + 2:
                runs.append((cid, cur))
            cur = [d] if d["clip"] == 0 else []
    if len(cur) >= L + K + 2:
        runs.append((cid, cur))

print(f"Usable visible runs (>= {L+K+2} frames): {len(runs)} "
      f"from {len(by_cid)} tracks")


def fit_eval(ts, ys, t_query, mode):
    ts = np.asarray(ts); ys = np.asarray(ys)
    if mode == "hold":
        return ys[-1]
    if mode == "lin":
        b, a = np.polyfit(ts, ys, 1)
        return a + b * t_query
    if mode == "invlin":
        inv = 1.0 / np.maximum(ys, 1e-6)
        b, a = np.polyfit(ts, inv, 1)
        val = a + b * t_query
        return 1.0 / val if val > 1e-6 else ys[-1]
    raise ValueError(mode)


MODES = ["hold", "lin", "invlin"]
err_px = {m: defaultdict(list) for m in MODES}
err_m  = {m: defaultdict(list) for m in MODES}
err_kmh = {m: defaultdict(list) for m in MODES}
n_candidates = 0

for cid, run in runs:
    Hbox = [d["y2"] - d["y1"] for d in run]
    # candidate clip-start index i: need L history before, K horizon after
    for i in range(L, len(run) - K):
        # approaching only: box height must be growing over the history window
        if Hbox[i - 1] <= Hbox[i - L]:
            continue
        n_candidates += 1
        hist_t = [run[j]["t"] for j in range(i - L, i)]
        hist_H = [Hbox[j] for j in range(i - L, i)]

        for m in MODES:
            # reconstructed world positions from clip-start onward
            pos_recon = {}
            for k in range(1, K + 1):
                j = i + k - 1
                d = run[j]
                Hpred = fit_eval(hist_t, hist_H, d["t"], m)
                y2_recon = d["y1"] + Hpred
                x_r, z_r = proj(d["gx"], y2_recon)
                pos_recon[j] = (x_r, z_r)

                # pixel + metre error vs truth
                err_px[m][k].append(abs(y2_recon - d["y2"]))
                err_m[m][k].append(float(np.hypot(x_r - d["x_m"], z_r - d["z_m"])))

                # windowed speed error: position at j is reconstructed,
                # earlier window endpoint may be pre-clip (true) or recon
                t_now = d["t"]
                e = j
                while e > 0 and t_now - run[e]["t"] < W_SEC:
                    e -= 1
                dt = t_now - run[e]["t"]
                if dt < 1e-3:
                    continue
                # true speed
                vt = np.hypot(d["x_m"] - run[e]["x_m"],
                              d["z_m"] - run[e]["z_m"]) / dt * 3.6
                # recon speed: use recon pos at j, and at e if e>=i else true
                xe, ze = (pos_recon[e] if e >= i else (run[e]["x_m"], run[e]["z_m"]))
                vr = np.hypot(x_r - xe, z_r - ze) / dt * 3.6
                err_kmh[m][k].append(abs(vr - vt))

print(f"Clip-start candidates (approaching segments): {n_candidates}\n")
if n_candidates == 0:
    raise SystemExit("No approaching segments found — cannot validate.")

# ─── Report ────────────────────────────────────────────────────
def summary(d, k):
    a = np.array(d[k])
    return (np.median(a), np.percentile(a, 90)) if len(a) else (float('nan'),) * 2

print("Reconstruction error by horizon (median / 90th pct):")
for m in MODES:
    print(f"\n  model = {m}")
    print(f"   {'k':>3} {'frames':>7} {'y2_px med':>10} {'p90':>7} "
          f"{'world_m med':>12} {'p90':>7} {'kmh med':>9} {'p90':>7}")
    for k in [1, 2, 3, 4, 6, 8, 10, 12]:
        if k > K:
            continue
        pxm, pxp = summary(err_px[m], k)
        mm, mp = summary(err_m[m], k)
        km, kp = summary(err_kmh[m], k)
        tsec = k / FPS
        print(f"   {k:>3} {tsec:>6.2f}s {pxm:>10.1f} {pxp:>7.1f} "
              f"{mm:>12.2f} {mp:>7.2f} {km:>9.1f} {kp:>7.1f}")

# baseline for context: error if you just FROZE the ground point at clip start
# (i.e. did nothing) — measured as world drift of the true point over horizon
froze_m = defaultdict(list)
for cid, run in runs:
    Hbox = [d["y2"] - d["y1"] for d in run]
    for i in range(L, len(run) - K):
        if Hbox[i - 1] <= Hbox[i - L]:
            continue
        x0, z0 = run[i - 1]["x_m"], run[i - 1]["z_m"]
        for k in range(1, K + 1):
            d = run[i + k - 1]
            froze_m[k].append(float(np.hypot(d["x_m"] - x0, d["z_m"] - z0)))
print("\n  baseline = FREEZE ground point at clip start (do nothing):")
print(f"   {'k':>3} {'world_m med':>12} {'p90':>7}")
for k in [1, 2, 3, 4, 6, 8, 10, 12]:
    if k > K:
        continue
    a = np.array(froze_m[k])
    print(f"   {k:>3} {np.median(a):>12.2f} {np.percentile(a,90):>7.2f}")

# ─── Plot ──────────────────────────────────────────────────────
ks = list(range(1, K + 1))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for m in MODES:
    ax[0].plot(ks, [np.median(err_m[m][k]) for k in ks], marker="o", label=m)
    ax[1].plot(ks, [np.median(err_kmh[m][k]) for k in ks], marker="o", label=m)
ax[0].plot(ks, [np.median(froze_m[k]) for k in ks], "k--", label="freeze (do nothing)")
ax[0].set_title("Ground-point error vs horizon")
ax[0].set_xlabel("frames since clip start"); ax[0].set_ylabel("world error (m), median")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[1].set_title("Windowed-speed error vs horizon")
ax[1].set_xlabel("frames since clip start"); ax[1].set_ylabel("|speed err| (km/h), median")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
fig.suptitle(f"Bottom reconstruction validation — {n_candidates} approaching clip-starts "
             f"(L={L} history frames)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=120)
print(f"\nWrote {OUT_PNG}")
