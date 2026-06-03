"""
run.py — multi-video launcher for experiment-new-model.

Edit INPUT_FILES to list every clip you want processed, then:
    python run.py

Or pass files directly on the command line (overrides INPUT_FILES):
    python run.py ../_in/clip_a.mp4 ../_in/clip_b.mp4

Calibration (interactive) runs sequentially — one window at a time.
Detection also runs sequentially — one clip at a time on the GPU.

Calibration files are stored in calib/<stem>_H/src/track.npy so each
video keeps its own calibration and recalibration never overwrites others.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── File list ──────────────────────────────────────────────────
# Add or remove entries here.  Paths are relative to this file.
INPUT_FILES = [
    "../_in/accident_cm_in_p1.mp4",
    "../_in/accident_cm_in_p2.mp4",
    "../_in/accident_cm_in_p3.mp4",
    "../_in/accident_cm_in_p5.mp4",
    "../_in/accident_cm_in_p6.mp4",
    "../_in/accident_cm_in_p10.mp4"
]

# ── Internals ──────────────────────────────────────────────────
HERE      = Path(__file__).parent
CALIB_DIR = HERE / "calib"


def calib_paths(stem: str):
    return (
        CALIB_DIR / f"{stem}_H.npy",
        CALIB_DIR / f"{stem}_src.npy",
        CALIB_DIR / f"{stem}_track.npy",
    )


def active_paths():
    return (
        HERE / "H_manual.npy",
        HERE / "src_manual.npy",
        HERE / "track_manual.npy",
    )


def ensure_calibration(input_video: str) -> tuple:
    """Return (h_path, src_path, track_path) for this video, calibrating if needed."""
    stem               = Path(input_video).stem
    h_c, src_c, track_c = calib_paths(stem)

    if h_c.exists() and src_c.exists() and track_c.exists():
        print(f"  [{stem}] calibration found — skipping.")
        return h_c, src_c, track_c

    missing = [p.name for p in (h_c, src_c, track_c) if not p.exists()]
    print(f"  [{stem}] missing {', '.join(missing)} — opening calibration window...")

    env = {**os.environ, "CALIB_VIDEO": input_video}
    ret = subprocess.run(
        [sys.executable, str(HERE / "manual_calibrate.py")],
        env=env, cwd=str(HERE),
    )
    if ret.returncode != 0:
        sys.exit(f"Calibration for '{stem}' exited with code {ret.returncode}. Aborting.")

    h_a, src_a, track_a = active_paths()
    for p in (h_a, src_a, track_a):
        if not p.exists():
            sys.exit(f"Expected {p.name} after calibration for '{stem}' but it is missing.")

    CALIB_DIR.mkdir(exist_ok=True)
    shutil.copy(h_a,     h_c)
    shutil.copy(src_a,   src_c)
    shutil.copy(track_a, track_c)
    print(f"  [{stem}] calibration saved to calib/ for future runs.")
    return h_c, src_c, track_c


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else INPUT_FILES
    if not files:
        sys.exit("No input files specified. Edit INPUT_FILES in run.py or pass paths as arguments.")

    print(f"Processing {len(files)} file(s).\n")

    # ── Step 1: calibration (sequential — requires user interaction) ──
    print("=== Calibration ===")
    calib_map = {}   # input_video -> (h_path, src_path, track_path)
    for video in files:
        calib_map[video] = ensure_calibration(video)
    print()

    # ── Step 2: detection (sequential) ────────────────────────────────
    print("=== Detection ===")
    failed = []
    for i, video in enumerate(files, 1):
        stem         = Path(video).stem
        output_video = f"out/{stem}_cp5.mp4"
        h_c, src_c, track_c = calib_map[video]

        print(f"  [{i}/{len(files)}] {stem} -> {output_video}")
        env = {
            **os.environ,
            "INPUT_VIDEO":  video,
            "OUTPUT_VIDEO": output_video,
            "H_PATH":       str(h_c),
            "SRC_PATH":     str(src_c),
            "TRACK_PATH":   str(track_c),
        }
        ret = subprocess.run(
            [sys.executable, str(HERE / "detect.py")],
            env=env, cwd=str(HERE),
        ).returncode
        status = "done" if ret == 0 else f"FAILED (exit {ret})"
        print(f"  [{i}/{len(files)}] {stem} {status}\n")
        if ret != 0:
            failed.append(stem)

    if failed:
        sys.exit(f"Detection failed for: {', '.join(failed)}")
    print("All done.")


if __name__ == "__main__":
    main()
