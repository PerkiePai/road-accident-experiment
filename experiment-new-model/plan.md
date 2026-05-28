# Accident Detection Pipeline — Experiment Plan

## Context

The existing `vector-tracking/` pipeline (YOLO11n + ByteTrack + custom ReIDStitcher)
has three compounding failures at the accident moment:

1. **YOLO NMS merges overlapping cars** — two vehicles with >70% pixel IoU collapse
   into one bounding box (confirmed at t=3.5s in `accident_cm_in_p10.mp4`).
2. **ByteTrack KF diverges** — constant-velocity model cannot handle sudden
   deceleration/direction reversal at impact; track is dropped.
3. **Speed/direction signal breaks** — the tracking point disappears exactly when
   the abrupt change in velocity is the accident evidence.

This experiment replaces each broken component with a better-suited alternative and
adds a dedicated accident trigger based on simultaneous speed + heading change.

---

## Target Pipeline

```
Frame
  │
  ▼
[RT-DETR-l]          — no NMS, bipartite matching → separate boxes even during overlap
  │ vehicle boxes
  ▼
[ResNet-18 ReID]     — 512-d appearance embedding per crop (already in vector-tracking)
  │ boxes + features
  ▼
[Deep OC-SORT]       — Hungarian + virtual-trajectory recovery during occlusion
  │ canonical track IDs + positions
  ▼
[Homography H]       — pixel → world coordinates (metres)
  │ (x_m, z_m) per track
  ▼
[IMM Filter]         — CV + CA modes; switches automatically at sudden deceleration
  │ smoothed position + velocity + acceleration
  ▼
[CTRV-state KF]      — heading ψ and turn rate dψ/dt as explicit state variables
  │ speed (km/h) + heading (degrees)
  ▼
[Accident Trigger]   — flag when |Δspeed| > threshold AND |Δheading| > threshold
  │                    simultaneously, with IMM confidence weighting
  ▼
Annotated output video
```

---

## Checkpoints

### CP1 — Detector comparison on occlusion frame  ✅ READY
**Script**: `checkpoint1_detect.py`  
**Input**: `t3.5.png` (local copy)  
**Goal**: Confirm RT-DETR-l produces 2 separate vehicle detections where YOLO11n
produced 1 merged box.  
**Pass**: `rt_count > yolo_count` at t=3.5s  
**Output**: `out/yolo_t3.5.png`, `out/rtdetr_t3.5.png`, `out/compare_t3.5.png`  
**Run**:
```
conda activate car-detection
cd experiment-new-model
python checkpoint1_detect.py
```

---

### CP2 — Deep OC-SORT tracking on full video
**Script**: `checkpoint2_track.py` (to implement)  
**Input**: `../_in/accident_cm_in_p10.mp4` + calibration `.npy` files from `../vector-tracking/`  
**Goal**: Track all vehicles through the accident with stable canonical IDs.
Verify that both cars retain separate IDs through the t=3.5s occlusion window.  
**Key config**:
- OC-SORT with appearance cost from ResNet-18
- `track_buffer` ~90 frames (3 s at 30 fps)
- KF patch: `_std_weight_velocity = 1/16` (carried over from vector-tracking)  
**Pass**: Car IDs do not swap or merge between t=3.0s and t=5.0s  
**Output**: `out/cp2_tracked.mp4`

---

### CP3 — IMM Filter for speed through collision
**Script**: `checkpoint3_imm_speed.py` (to implement)  
**Module**: `imm_filter.py` (shared)  
**Input**: Ground-plane positions from CP2  
**Goal**: Show speed estimate stays valid through the collision deceleration
rather than going to noise (CV-only KF failure mode).  
**Implementation notes**:
- Two parallel KFs: CV (state=[x, z, vx, vz]) and CA (state=[x, z, vx, vz, ax, az])
- Mixing weights updated each frame via likelihood ratio
- When CA weight dominates → collision deceleration detected  
**Pass**: Speed curve shows smooth deceleration rather than spike/drop at t=3.5s  
**Output**: `out/cp3_speed_plot.png`

---

### CP4 — CTRV heading estimation
**Script**: `checkpoint4_heading.py` (to implement)  
**Module**: `ctrv_filter.py` (shared)  
**Input**: Ground-plane positions from CP2  
**Goal**: Compute stable heading angle ψ (degrees, 0=north) for each track.  
**Implementation notes**:
- Extended KF state: [x, z, v, ψ, dψ/dt]
- State transition: nonlinear CTRV equations with Jacobian
- Fallback to position-diff heading when v < 2 m/s  
**Pass**: Heading smooth during normal driving; shows abrupt jump at impact  
**Output**: `out/cp4_heading_plot.png`

---

### CP5 — Accident trigger logic
**Script**: `checkpoint5_trigger.py` (to implement)  
**Input**: Speed + heading time series from CP3 + CP4  
**Goal**: Flag the accident event with a timestamp.  
**Trigger condition**:
```python
delta_speed   = speed[t] - speed[t-1]          # km/h per frame
delta_heading = abs(heading[t] - heading[t-1])  # degrees per frame

if delta_speed < -SPEED_THRESH and delta_heading > HEADING_THRESH:
    accident_flag = True
    accident_time = t_now
```
**Tuning targets**:
- `SPEED_THRESH`   = 7 km/h/frame
- `HEADING_THRESH` = 15 degrees/frame  
**Pass**: Trigger fires within ±0.5s of visible collision; no false positives on
normal driving sections  
**Output**: `out/cp5_trigger_log.txt`, `out/cp5_timeline.png`

---

### CP6 — Full integrated pipeline video output
**Script**: `detect_new.py` (final integrated script)  
**Output**: `out/accident_detected.mp4` annotated with:
- Per-track bounding boxes (RT-DETR)
- Ground-point fading trail
- Speed (km/h) + heading (°) overlaid on each box
- Speed panel chart
- Red flash overlay when accident trigger fires  
**Pass**: No ID swaps at t=3.5s; accident alert visible at correct timestamp

---

## File Structure

```
experiment-new-model/
├── plan.md                    ← this file
├── t3.5.png                   ← local test frame (occlusion moment)
├── checkpoint1_detect.py      ← CP1 ✅ ready
├── checkpoint2_track.py       ← CP2 (to implement)
├── checkpoint3_imm_speed.py   ← CP3 (to implement)
├── checkpoint4_heading.py     ← CP4 (to implement)
├── checkpoint5_trigger.py     ← CP5 (to implement)
├── detect_new.py              ← CP6 final (to implement)
├── imm_filter.py              ← IMM KF module  (shared by CP3, CP6)
├── ctrv_filter.py             ← CTRV KF module (shared by CP4, CP6)
└── out/                       ← all output images and videos
```

---

## Dependencies

All in `car-detection` conda env — no new installs needed:
- `ultralytics` — RT-DETR + YOLO
- `torch` + `torchvision` — ResNet-18 ReID
- `opencv-python` — video I/O + drawing
- `numpy`, `scipy` — IMM/CTRV math + Hungarian assignment

Weight files (auto-downloaded on first run per script):
- `yolo11n.pt` — ~6 MB (CP1 baseline)
- `rtdetr-l.pt` — ~80 MB (CP1+)

---

## Calibration

Read-only from `vector-tracking/` — no files copied, no re-calibration needed:
- `../vector-tracking/H_manual.npy`
- `../vector-tracking/src_manual.npy`
- `../vector-tracking/track_manual.npy`

---

## Notes

- All scripts run from inside `experiment-new-model/` with `conda activate car-detection`.
- RT-DETR does not use `iou=` (no NMS); only `conf=` applies.
- `../bytetrack.yaml` exists at repo root; Deep OC-SORT config `ocsort.yaml` will be
  added here when CP2 is started.
- IMM mixing weights logged per frame in CP3 for debugging alongside speed curve.
- CP3 and CP4 can be developed in parallel once CP2 positions are available.
