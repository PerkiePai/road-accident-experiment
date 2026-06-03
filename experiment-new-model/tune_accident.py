"""Offline replay / tuning harness for the CP5 AccidentDetector.

The detector consumes only per-frame (t, yaw_deg_s, ema_kmh) per canonical ID —
exactly the columns DUMP_TRACE writes to out/{stem}_trace.csv.  So we can replay
those traces through AccidentDetector with no GPU and no RT-DETR, faithfully
reproducing detect.py's call contract:

    for each frame a cid is drawn (recon>0 or coasting), detect.py calls
        accident_det.update(cid, t_now, ctrv_yaw, ema_speed[cid])
    but ONLY when ema_speed[cid] is not None (the trace logs it as 'nan' until
    the first EMA is set).

Replaying per-cid in frame order reproduces this exactly (per-cid state is
independent, so cross-cid interleaving doesn't matter).

Usage:
    python tune_accident.py            # replay + compare to logged accidents CSV
    python tune_accident.py --write    # also write out/{stem}_accidents_v2.csv
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

import accident_detector
from accident_detector import AccidentDetector

STEMS = ["accident_cm_in_p1", "accident_cm_in_p2", "accident_cm_in_p3",
         "accident_cm_in_p5", "accident_cm_in_p6", "accident_cm_in_p10"]
FPS = 30.0
OUT = Path("out")


def load_trace_rows(stem):
    """Return per-cid lists of frame-ordered dicts from the trace CSV."""
    by_cid = defaultdict(list)
    with open(OUT / f"{stem}_trace.csv", newline="") as f:
        for r in csv.DictReader(f):
            def fnum(k):
                v = r[k]
                return None if v in ("", "nan") else float(v)
            by_cid[int(r["cid"])].append({
                "frame": int(r["frame"]), "t": fnum("t"),
                "v": fnum("v_kmh"), "ema": fnum("ema_kmh"),
                "yaw": fnum("yaw_deg_s"), "recon": fnum("recon_conf"),
                "heading": r["heading_deg"],
            })
    for cid in by_cid:
        by_cid[cid].sort(key=lambda d: d["frame"])
    return by_cid


def replay(stem):
    """Replay one trace through a fresh AccidentDetector. Returns trigger rows."""
    by_cid = load_trace_rows(stem)
    det = AccidentDetector(FPS)
    # interleave by frame across cids to match detect.py's per-frame ordering
    events = []
    for cid, rows in by_cid.items():
        for d in rows:
            events.append((d["frame"], cid, d))
    events.sort(key=lambda e: (e[0], e[1]))

    triggers = []
    for frame, cid, d in events:
        if d["ema"] is None:          # detect.py skips update() until EMA is set
            continue
        yaw = d["yaw"] if d["yaw"] is not None else 0.0
        is_acc, just_trig, _ = det.update(cid, d["t"], yaw, d["ema"])
        if just_trig:
            stt = det._states[cid]
            sb  = list(stt['speed_buf'])
            ddt = sb[-1][0] - sb[0][0]
            decel = -(sb[-1][1] - sb[0][1]) / ddt if ddt > 1e-3 else 0.0
            triggers.append({
                "frame": frame, "t": round(d["t"], 3), "cid": cid,
                "speed_kmh": round(d["ema"], 1), "yaw_deg_s": round(yaw, 1),
                "heading_deg": d["heading"],
                "age": stt['frames_seen'], "decel": round(decel, 1),
            })
    return triggers


def load_logged(stem):
    p = OUT / f"{stem}_accidents.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return [(int(r["frame"]), int(r["cid"])) for r in csv.DictReader(f)]


def sweep():
    """Grid over MIN_TRACK_FRAMES x DECEL_MAX_PHYS; report counts + clip recall."""
    print(f"{'MIN_FR':>6} {'MAXPHYS':>8} | "
          + " ".join(f"{s.split('_')[-1]:>4}" for s in STEMS)
          + f" | {'TOT':>4} {'clips':>5}")
    print("-" * 72)
    for min_fr in (4, 5, 6, 7, 8, 10, 12):
        for max_phys in (130.0,):
            accident_detector.MIN_TRACK_FRAMES = min_fr
            accident_detector.DECEL_MAX_PHYS   = max_phys
            counts, total, clips = [], 0, 0
            for stem in STEMS:
                n = len(replay(stem))
                counts.append(n)
                total += n
                clips += (n > 0)
            print(f"{min_fr:>6} {max_phys:>8.0f} | "
                  + " ".join(f"{c:>4}" for c in counts)
                  + f" | {total:>4} {clips:>4}/6")


def main():
    if "--sweep" in sys.argv:
        sweep()
        return
    write = "--write" in sys.argv
    # echo the active config so tuning runs are self-documenting
    cfg = {k: getattr(accident_detector, k) for k in dir(accident_detector)
           if k.isupper()}
    print("CONFIG:", {k: cfg[k] for k in (
        "DECEL_ONLY_THRESH", "DECEL_MAX_PHYS", "MIN_TRACK_FRAMES",
        "YAW_SPIKE_DEG_S", "N_CONFIRM", "CLEAR_FRAMES") if k in cfg})
    print()

    grand_new = grand_old = 0
    for stem in STEMS:
        new = replay(stem)
        old = load_logged(stem)
        grand_new += len(new)
        grand_old += len(old)
        new_keys = {(t["frame"], t["cid"]) for t in new}
        old_keys = set(old)
        dropped = sorted(old_keys - new_keys)
        added   = sorted(new_keys - old_keys)
        print(f"{stem:22} logged={len(old)}  replay={len(new)}", end="")
        if dropped:
            print(f"   -dropped {dropped}", end="")
        if added:
            print(f"   +added {added}", end="")
        print()
        for t in new:
            chan = ("YAW" if abs(t['yaw_deg_s']) > accident_detector.YAW_SPIKE_DEG_S
                    else "DECEL")
            print(f"      fr{t['frame']:<4} cid{t['cid']:<3} "
                  f"spd={t['speed_kmh']:6.1f} yaw={t['yaw_deg_s']:6.1f} "
                  f"age={t['age']:>3} decel={t['decel']:6.1f}  [{chan}]")
        if write:
            with open(OUT / f"{stem}_accidents_v2.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t", "cid", "speed_kmh", "yaw_deg_s", "heading_deg"])
                for t in new:
                    w.writerow([t["frame"], t["t"], t["cid"], t["speed_kmh"],
                                t["yaw_deg_s"], t["heading_deg"]])
    print(f"\nTOTAL  logged={grand_old}  replay={grand_new}")


if __name__ == "__main__":
    main()
