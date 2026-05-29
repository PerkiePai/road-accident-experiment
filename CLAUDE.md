# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vehicle speed-tracking system using dashcam/road footage. A perspective homography maps pixel positions to real-world ground-plane coordinates (metres), and per-track position history is differentiated to produce km/h estimates.

There are two implementations:
- **`speed-tracking/`** — YOLO11n + ByteTrack (lightweight, CPU-friendly)
- **`experiment-new-model/`** — RT-DETR-l + Deep OC-SORT + ResNet-18 ReID (higher accuracy, GPU required)

## Workflow

The pipeline is two-step — calibration must come before detection:

**Step 1 — Calibrate** (interactive, run once per video setup):
```
cd speed-tracking        # or experiment-new-model
python manual_calibrate.py
```
Opens an OpenCV window. Click 4 points (bottom-left → bottom-right → top-right → top-left) on a known rectangular road patch, then click N ≥ 3 points to outline the driveable tracking region and press Enter. Saves three `.npy` files to the working directory: `H_manual.npy`, `src_manual.npy`, `track_manual.npy`.

**Step 2 — Detect & track**:
```
cd speed-tracking        # or experiment-new-model
python detect.py
```
Reads the three `.npy` calibration files and produces an annotated output video.

## Dependencies

- `opencv-python` (`cv2`)
- `numpy`
- `ultralytics` (YOLO11 / RT-DETR — downloads weights on first run)
- `torch`, `torchvision` (ResNet-18 feature extractor in `experiment-new-model`)
- `boxmot` (Deep OC-SORT tracker in `experiment-new-model`)
- GPU strongly recommended for `experiment-new-model`; set `device=0` for CUDA

## Input Videos

Input videos live in `_in/` at the repo root. Update `INPUT_VIDEO` at the top of each `detect.py` to point to the desired file.

## Architecture — `speed-tracking/`

### `manual_calibrate.py` — `ManualCalibrator`
- **Phase 1**: 4-point click → `cv2.findHomography` → H matrix mapping pixel (x, y) → world (x_m, z_m) in metres, where z is depth along the road.
- **Phase 2**: N-point polygon click → tracking region saved as `track_manual.npy`.
- Saved files: `H_manual.npy`, `src_manual.npy`, `track_manual.npy`.

### `detect.py` — main detection loop
- **YOLO + ByteTrack**: `model.track(...)` with `classes=[2,3,5,7]` (car, motorcycle, bus, truck).
- **`ReIDStitcher`**: wraps ByteTrack IDs in stable *canonical IDs*. Matches newly-appeared raw IDs against recently-lost tracks using predicted ground-plane position (linear velocity extrapolation). Sticky alias table ensures consistent IDs across the video.
- **Speed calculation**: `speed_from_history()` uses finite difference over a configurable `SPEED_WINDOW` (seconds), then EMA-smoothed per canonical ID.
- **On-road filter**: detections whose bottom-centre falls outside `road_poly` are drawn as dim grey boxes and excluded from speed computation.
- **Speed panel**: bottom-right overlay draws per-track speed time series over the last `HISTORY_SEC` seconds.

### Key tuning constants (`speed-tracking/detect.py`)
| Constant | Purpose |
|---|---|
| `SPEED_WINDOW` | Seconds of history used for finite-diff speed |
| `EMA_ALPHA` | Smoothing weight for per-track EMA speed |
| `REID_DIST_BASE_M` | Base match radius (metres) for Re-ID stitcher |
| `REID_MAX_GAP_S` | Max seconds a lost track stays a candidate |
| `PANEL_VMAX_KMH_FLOOR` | Minimum y-axis ceiling on the speed chart |

## Architecture — `experiment-new-model/`

### `detect.py` — main detection loop
- **RT-DETR-l**: transformer-based detector, no NMS required. Bipartite matching produces one box per object but box size/position can vary significantly frame-to-frame.
- **Deep OC-SORT + ResNet-18 ReID**: `tracker.update(dets, frame, embs)` — appearance embeddings reduce ID switches. Tuned with `Q_xy_scaling=0.08` (higher process noise) for fast-moving vehicles.
- **`nms_tracks()`**: pixel-space IoU NMS applied after the tracker to suppress any same-vehicle duplicate boxes.
- **`WorldMerger`**: world-coordinate ID stabilisation layer (see below).
- **Speed calculation** and **speed panel**: same approach as `speed-tracking/`.
- **Fading trail**: ground-contact point history drawn as a colour-fading polyline per track.

### `WorldMerger` — world-coordinate ID stabilisation

RT-DETR's inconsistent box sizes cause the Kalman filter to lose confidence and spawn a second track for the same vehicle. The two tracker IDs then alternate frame-by-frame (never appearing simultaneously), so pixel-space IoU NMS cannot catch them. `WorldMerger` solves this with two operating modes each frame:

**Cross-frame match (primary fix):**
Every canonical ID's last ground position and velocity are stored in `recent`. When a brand-new raw tracker ID appears, its ground position is compared against each recently-lost canonical extrapolated forward by its velocity:
```
extrapolated = last_pos + last_vel × Δt
dist = hypot(new_x − extrap_x, new_z − extrap_z)
```
If `dist < WORLD_MERGE_DIST_M`, the new raw ID is aliased to the existing canonical — catching the alternating-track case even though the two IDs never share a frame.

**Same-frame merge (fallback):**
If two canonicals are simultaneously active and within `WORLD_MERGE_DIST_M`, the older one (lower `first_seen` frame index) absorbs the newer.

The alias table is **sticky** — a raw ID's canonical never changes once assigned. Redirects are propagated back through the full alias table so no stale pointer survives.

### Key tuning constants (`experiment-new-model/detect.py`)
| Constant | Purpose |
|---|---|
| `SPEED_WINDOW` | Seconds of history used for finite-diff speed |
| `EMA_ALPHA` | Smoothing weight for per-track EMA speed |
| `WORLD_MERGE_DIST_M` | Phase 2 cross-frame match radius for `WorldMerger` (metres) |
| `WORLD_SAME_FRAME_M` | Phase 3 same-frame merge radius — must be tight (≤2 m) to avoid merging adjacent-lane vehicles that appear close at long range |
| `WORLD_MERGE_GAP_S` | How long a lost canonical stays a match candidate (seconds) |
| `PANEL_VMAX_KMH_FLOOR` | Minimum y-axis ceiling on the speed chart |

### Calibration geometry
`ManualCalibrator(lane_width_m=7.0, road_depth_m=10.0)` — the destination rectangle for `findHomography` is always `[0,0] – [lane_width_m, road_depth_m]` in metres. Adjust these to match the actual road dimensions you clicked on.
