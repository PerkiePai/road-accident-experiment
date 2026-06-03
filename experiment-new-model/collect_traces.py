"""Collect DUMP_TRACE CSVs for all videos in run.py's INPUT_FILES."""
import os, subprocess, sys
from pathlib import Path

files = [
    "../_in/accident_cm_in_p1.mp4",
    "../_in/accident_cm_in_p2.mp4",
    "../_in/accident_cm_in_p3.mp4",
    "../_in/accident_cm_in_p5.mp4",
    "../_in/accident_cm_in_p6.mp4",
    "../_in/accident_cm_in_p10.mp4",
]
calib_dir = Path("calib")

for video in files:
    stem = Path(video).stem
    dst  = Path(f"out/{stem}_trace.csv")
    if dst.exists():
        print(f"  [{stem}] trace already exists — skipping.")
        continue
    print(f"  [{stem}] running...")
    env = {
        **os.environ,
        "INPUT_VIDEO":  video,
        "OUTPUT_VIDEO": f"out/{stem}_trace.mp4",
        "H_PATH":       str(calib_dir / f"{stem}_H.npy"),
        "SRC_PATH":     str(calib_dir / f"{stem}_src.npy"),
        "TRACK_PATH":   str(calib_dir / f"{stem}_track.npy"),
        "DUMP_TRACE":   "1",
    }
    subprocess.run([sys.executable, "detect.py"], env=env, cwd=".")
    src = Path("out/detect_speed_trace.csv")
    if src.exists():
        src.rename(dst)
    print(f"  [{stem}] -> {dst}")
