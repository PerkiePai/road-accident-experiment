"""Analyse CP5 accident triggers against the per-frame trace.

For every trigger logged in out/{stem}_accidents.csv, reconstruct the speed /
yaw trajectory of that canonical ID in the frames leading up to the trigger so
we can tell a genuine impact deceleration from an EMA-settling artefact.
"""
import csv
from pathlib import Path
from collections import defaultdict

STEMS = ["accident_cm_in_p1", "accident_cm_in_p2", "accident_cm_in_p3",
         "accident_cm_in_p5", "accident_cm_in_p6", "accident_cm_in_p10"]

# Mirror of accident_detector.py constants
YAW_SPIKE = 15.0
DECEL_ONLY = 55.0
DECEL_COMBO = 20.0
FPS = 30.0
DECEL_WINDOW_S = 0.15
NBUF = max(2, int(FPS * DECEL_WINDOW_S))   # 4

OUT = Path("out")


def load_trace(stem):
    """cid -> list of rows (dict) ordered by frame."""
    rows = defaultdict(list)
    with open(OUT / f"{stem}_trace.csv", newline="") as f:
        for r in csv.DictReader(f):
            cid = int(r["cid"])
            def fnum(k):
                v = r[k]
                if v in ("", "nan"):
                    return None
                return float(v)
            rows[cid].append({
                "frame": int(r["frame"]), "t": fnum("t"),
                "v": fnum("v_kmh"), "ema": fnum("ema_kmh"),
                "yaw": fnum("yaw_deg_s"), "recon": fnum("recon_conf"),
            })
    return rows


def classify(yaw, decel):
    conds = []
    if decel > DECEL_ONLY:
        conds.append("DECEL")
    if abs(yaw) > YAW_SPIKE:
        conds.append("YAW")          # may be sustained
    if abs(yaw) > YAW_SPIKE and decel > DECEL_COMBO:
        conds.append("COMBO")
    return conds or ["?"]


print(f"buffer = {NBUF} frames  (~{NBUF/FPS*1000:.0f} ms)\n")

summary = defaultdict(int)
for stem in STEMS:
    acc_path = OUT / f"{stem}_accidents.csv"
    trace = load_trace(stem)
    print("=" * 78)
    print(stem)
    print("=" * 78)
    with open(acc_path, newline="") as f:
        triggers = list(csv.DictReader(f))
    for tr in triggers:
        fr  = int(tr["frame"])
        cid = int(tr["cid"])
        hist = trace.get(cid, [])
        # frames for this cid up to & including trigger frame
        upto = [h for h in hist if h["frame"] <= fr]
        first_seen = hist[0]["frame"] if hist else None
        age = fr - first_seen if first_seen is not None else None

        # reconstruct decel from the EMA buffer used by the detector
        win = upto[-NBUF:]
        decel = None
        if len(win) >= NBUF and win[0]["ema"] and win[-1]["ema"]:
            dt = win[-1]["t"] - win[0]["t"]
            if dt > 1e-3:
                decel = -(win[-1]["ema"] - win[0]["ema"]) / dt
        yaw = float(tr["yaw_deg_s"])
        conds = classify(yaw, decel if decel else 0.0)
        for c in conds:
            summary[c] += 1
        summary["TOTAL"] += 1

        # ema trajectory string (last ~7 frames)
        tail = upto[-7:]
        ema_str = " ".join(f"{h['ema']:.0f}" if h['ema'] else "·" for h in tail)
        v_str   = " ".join(f"{h['v']:.0f}"  if h['v']  else "·" for h in tail)
        decel_s = f"{decel:5.1f}" if decel is not None else "  n/a"
        flag = "  <-- fr4-class (track age %s)" % age if fr <= 7 else ""
        print(f" fr{fr:<4} cid{cid:<3} spd={float(tr['speed_kmh']):6.1f} "
              f"yaw={yaw:6.1f}  decel={decel_s}  age={str(age):>4}  "
              f"[{'+'.join(conds)}]{flag}")
        print(f"        ema : {ema_str}")
        print(f"        vraw: {v_str}")
    print()

print("=" * 78)
print("TRIGGER CONDITION BREAKDOWN")
print("=" * 78)
for k in ("DECEL", "YAW", "COMBO", "?", "TOTAL"):
    print(f"  {k:8} {summary[k]}")
