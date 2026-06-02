"""
Plot the CTRV heading logged by detect.py (per canonical ID) straight from
out/detect_speed_trace.csv.  No GPU — reads heading_deg / yaw_deg_s / ema_kmh
that detect.py already wrote (run detect.py once with DUMP_TRACE=1).

Shows the tracks that actually move; heading is unwrapped so the 0/360 seam
doesn't fake vertical jumps.  Speed is overlaid so the speed-drop + heading-
jump that CP5 keys on are visible together.

Run:  python plot_heading.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

CSV  = "out/detect_speed_trace.csv"
OUT  = "out/cp4_detect_heading.png"
MOVE_MIN_M = 3.0   # net displacement (m) to count a track as "moving"

rows = defaultdict(list)
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows[int(r["cid"])].append(r)

def fget(r, k):
    v = r[k]
    return float(v) if v not in ("", "nan") else float("nan")

def unwrap(vals):
    out, off, prev = [], 0.0, None
    for v in vals:
        if np.isnan(v):
            out.append(np.nan); continue
        if prev is not None:
            d = v - prev
            while d > 180: off -= 360; d -= 360
            while d < -180: off += 360; d += 360
        out.append(v + off); prev = v
    return out

def net_disp(d):
    xs = [fget(r, "x_m") for r in d]; zs = [fget(r, "z_m") for r in d]
    return float(np.hypot(xs[-1]-xs[0], zs[-1]-zs[0]))

movers = [c for c in sorted(rows) if len(rows[c]) >= 5 and net_disp(rows[c]) > MOVE_MIN_M]
if not movers:
    movers = sorted(rows, key=lambda c: len(rows[c]), reverse=True)[:3]

fig, axes = plt.subplots(len(movers), 1, figsize=(12, 3.2 * len(movers)),
                         sharex=True)
if len(movers) == 1:
    axes = [axes]

for ax, cid in zip(axes, movers):
    d   = rows[cid]
    t   = [fget(r, "t") for r in d]
    hd  = unwrap([fget(r, "heading_deg") for r in d])
    yaw = [fget(r, "yaw_deg_s") for r in d]
    spd = [fget(r, "ema_kmh") for r in d]

    ax.plot(t, hd, color="orange", lw=2.0, label="CTRV heading")
    ax.set_ylabel("heading (deg, unwrapped)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(t, spd, color="steelblue", lw=1.2, alpha=0.6, label="speed (km/h)")
    ax2.plot(t, yaw, color="green", lw=0.8, alpha=0.4, label="yaw (deg/s)")
    ax2.set_ylabel("km/h  /  deg/s", fontsize=8)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8)
    ax.set_title(f"cid{cid}  (net {net_disp(d):.1f} m)", fontsize=9)

axes[-1].set_xlabel("time (s)")
fig.suptitle("CP4 heading from detect.py (integrated, canonical IDs)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"Saved {OUT}  (tracks: {movers})")
