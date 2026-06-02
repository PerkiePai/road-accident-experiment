"""
Offline CTRV tuning harness (no GPU).

Replays the per-track world positions already logged by detect.py in
out/detect_speed_trace.csv (cid, t, x_m, z_m) through CTRVFilter, so the
heading filter can be tuned/debugged instantly without re-running the
RT-DETR + tracker pass over the video.

Stability metrics per track:
  n_pts, dur, mean_speed   — how much the vehicle actually moves
  winding                  — unwrapped CTRV heading range (max-min), deg.
                             A near-stationary car should NOT wind up;
                             winding >> 360 means the filter invents rotation.
  net_dir                  — heading of net displacement (reference truth)
  end_err                  — |final CTRV heading - net_dir| for moving tracks

Run:  python cp4_tune.py
"""
import csv
import numpy as np
from collections import defaultdict
import importlib, ctrv_filter
importlib.reload(ctrv_filter)
from ctrv_filter import CTRVFilter

CSV   = "out/detect_speed_trace.csv"
MOVING_SPEED = 2.0   # m/s; below this a track is "stationary" for scoring

def load():
    rows = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows[int(r["cid"])].append(
                (float(r["t"]), float(r["x_m"]), float(r["z_m"])))
    for cid in rows:
        rows[cid].sort(key=lambda p: p[0])
    return rows

def angdiff(a, b):
    """smallest signed difference a-b in degrees, (-180,180]"""
    return (a - b + 180) % 360 - 180

def run_track(pts, **kw):
    """Return (times, headings_deg, speeds_kmh, yaw_deg_s)."""
    if not pts:
        return [], [], [], []
    dts = np.diff([p[0] for p in pts])
    dt  = float(np.median(dts)) if len(dts) else 1/30
    f   = CTRVFilter(dt=dt, **kw)
    f.init(pts[0][1], pts[0][2])
    T, H, S, Y = [], [], [], []
    for t, x, z in pts:
        h = f.update(x, z)
        T.append(t); H.append(h); S.append(f.speed_kmh); Y.append(f.yaw_rate_deg)
    return T, H, S, Y

def unwrap_deg(vals):
    out = np.degrees(np.unwrap(np.radians(vals)))
    return out

def score(rows, **kw):
    print(f"\n=== params: {kw if kw else 'defaults'} ===")
    print(f"{'cid':>4} {'n':>4} {'dur':>5} {'mspd':>6} {'winding':>9} "
          f"{'net_dir':>8} {'end_err':>8}  verdict")
    worst_wind = 0.0
    for cid in sorted(rows):
        pts = rows[cid]
        if len(pts) < 5:
            continue
        T, H, S, Y = run_track(pts, **kw)
        dur = pts[-1][0] - pts[0][0]
        # NET-displacement speed (jitter cancels) decides moving vs stationary
        seg = np.array([(p[1], p[2]) for p in pts])
        net  = seg[-1] - seg[0]
        mspd = float(np.linalg.norm(net) / max(dur, 1e-6))   # net speed m/s
        net_dir = float(np.degrees(np.arctan2(net[0], net[1])) % 360)
        Hu   = unwrap_deg(H)
        winding = float(Hu.max() - Hu.min())
        end_err = abs(angdiff(H[-1], net_dir))
        moving  = mspd > MOVING_SPEED
        worst_wind = max(worst_wind, winding if not moving else 0)
        verdict = ""
        if not moving and winding > 90:
            verdict = "WIND-UP (stationary)"
        elif moving and end_err > 45:
            verdict = "heading off"
        else:
            verdict = "ok"
        tag = "MOV" if moving else "sta"
        print(f"{cid:>4} {len(pts):>4} {dur:>5.1f} {mspd:>5.1f}{tag:>1} "
              f"{winding:>9.0f} {net_dir:>8.0f} {end_err:>8.0f}  {verdict}")
    print(f"worst stationary winding: {worst_wind:.0f} deg "
          f"(target < 90)")
    return worst_wind

def plot(rows, path="out/cp4_tune_plot.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot (scoring above is the "
              "key output; run with the car-detection env to render).")
        return
    ids = [c for c in sorted(rows) if len(rows[c]) >= 5]
    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 2.6 * len(ids)),
                             sharex=True)
    if len(ids) == 1:
        axes = [axes]
    for ax, cid in zip(axes, ids):
        pts = rows[cid]
        T, H, S, Y = run_track(pts)
        seg = np.array([(p[1], p[2]) for p in pts])
        net = np.linalg.norm(seg[-1] - seg[0]) / max(pts[-1][0]-pts[0][0], 1e-6)
        ax.plot(T, unwrap_deg(H), color="orange", lw=2.0, label="CTRV heading")
        ax2 = ax.twinx()
        ax2.plot(T, Y, color="green", lw=0.8, alpha=0.5, label="yaw rate")
        ax2.set_ylabel("yaw (deg/s)", color="green", fontsize=8)
        tag = "MOVING" if net > MOVING_SPEED else "stationary"
        ax.set_title(f"cid{cid}  ({tag}, net {net:.1f} m/s)", fontsize=9)
        ax.set_ylabel("heading (deg)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("CP4 tuned — CTRV heading on real trace (unwrapped)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"Saved {path}")


if __name__ == "__main__":
    rows = load()
    score(rows)   # current defaults
    plot(rows)
