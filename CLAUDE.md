# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vehicle speed-tracking system using YOLO + ByteTrack on dashcam/road footage. A perspective homography maps pixel positions to real-world ground-plane coordinates (metres), and per-track position history is differentiated to produce km/h estimates.

## Workflow

The pipeline is two-step — calibration must come before detection:

**Step 1 — Calibrate** (interactive, run once per video setup):
```
cd speed-tracking
python manual_calibrate.py
```
Opens an OpenCV window. Click 4 points (bottom-left → bottom-right → top-right → top-left) on a known rectangular road patch, then click N ≥ 3 points to outline the driveable tracking region and press Enter. Saves three `.npy` files to the working directory: `H_manual.npy`, `src_manual.npy`, `track_manual.npy`.

**Step 2 — Detect & track**:
```
cd speed-tracking
python detect.py
```
Reads the three `.npy` calibration files and produces an annotated output video (`car_100kmh_tracked.mp4` by default).

## Dependencies

- `opencv-python` (`cv2`)
- `numpy`
- `ultralytics` (YOLO — downloads `yolo11n.pt` on first run)
- GPU recommended; set `device=0` in `detect.py:321` for CUDA, or change to `"cpu"`

## Input Videos

Input videos live in `_in/` at the repo root. The scripts reference the path `in/car_100kmh.mp4` (relative to `speed-tracking/`), so either create a symlink `speed-tracking/in → ../_in` or update `INPUT_VIDEO` in `detect.py` and the `VideoCapture` path in `manual_calibrate.py`.

## Architecture

### `manual_calibrate.py` — `ManualCalibrator`
- **Phase 1**: 4-point click → `cv2.findHomography` → H matrix mapping pixel (x, y) → world (x_m, z_m) in metres, where z is depth along the road.
- **Phase 2**: N-point polygon click → tracking region saved as `track_manual.npy`.
- Saved files: `H_manual.npy`, `src_manual.npy`, `track_manual.npy`.

### `detect.py` — main detection loop
- **YOLO + ByteTrack**: `model.track(...)` with `classes=[2,3,5,7]` (car, motorcycle, bus, truck). Raw track IDs are short-lived when targets leave/re-enter frame.
- **`ReIDStitcher`**: wraps ByteTrack IDs in stable *canonical IDs*. Matches newly-appeared raw IDs against recently-lost tracks using predicted ground-plane position (linear velocity extrapolation). Sticky alias table ensures consistent IDs across the video.
- **Speed calculation**: `speed_from_history()` uses finite difference over a configurable `SPEED_WINDOW` (seconds), then EMA-smoothed per canonical ID.
- **On-road filter**: detections whose bottom-centre falls outside `road_poly` are drawn as dim grey boxes and excluded from speed computation.
- **Speed panel**: bottom-right overlay draws per-track speed time series over the last `HISTORY_SEC` seconds.

### Key tuning constants (top of `detect.py`)
| Constant | Purpose |
|---|---|
| `SPEED_WINDOW` | Seconds of history used for finite-diff speed |
| `EMA_ALPHA` | Smoothing weight for per-track EMA speed |
| `REID_DIST_BASE_M` | Base match radius (metres) for Re-ID stitcher |
| `REID_MAX_GAP_S` | Max seconds a lost track stays a candidate |
| `PANEL_VMAX_KMH_FLOOR` | Minimum y-axis ceiling on the speed chart |

### Calibration geometry
`ManualCalibrator(lane_width_m=7.0, road_depth_m=10.0)` — the destination rectangle for `findHomography` is always `[0,0] – [lane_width_m, road_depth_m]` in metres. Adjust these to match the actual road dimensions you clicked on.
