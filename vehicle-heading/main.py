"""
vehicle-heading/main.py — per-track heading overlay on a calibrated video.

Runs a two-source heading pipeline:
  1. PoseHeading   — instantaneous front->rear keypoint vector (trained pose model)
  2. TrajectoryHeading — CTRV EKF on BEV ground-contact point history
  3. HeadingFusion — confidence-weighted circular blend + EMA

Draws on each tracked vehicle:
  - coloured bounding box (colour = track ID)
  - heading arrow (yellow = pose dominant, cyan = trajectory dominant, grey = hold)
  - label: ID, speed km/h, heading deg, source tag

Usage (run from vehicle-heading/):
    conda activate car-detection
    python main.py                                         # defaults
    python main.py --video ../_in/thai_road_full_cut.mp4   # different clip
    python main.py --pose_weights runs/pose/vehicle_pose4/weights/best.pt
    python main.py --no_pose    # trajectory-only (useful before training is done)

Calibration files default to ../experiment-new-model/H_manual.npy etc.
Set --h_path / --track_path to point at a different calibration.
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# ── heading modules (same directory) ──────────────────────────────
from geometry import Homography, heading_from_vec
from heading_estimator import HeadingFusion, PoseHeading, TrajectoryHeading
from pose_inference import PoseInference

# ── default paths ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

DEFAULT_VIDEO      = os.path.join(REPO_ROOT, "_in", "thai_road_full_cut.mp4")
DEFAULT_POSE_W     = os.path.join(SCRIPT_DIR, "runs", "pose", "vehicle_pose4", "weights", "best.pt")
DEFAULT_H_PATH     = os.path.join(REPO_ROOT, "experiment-new-model", "H_manual.npy")
DEFAULT_TRACK_PATH = os.path.join(REPO_ROOT, "experiment-new-model", "track_manual.npy")
DEFAULT_DET_W      = os.path.join(REPO_ROOT, "experiment-new-model", "yolo11n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck
ARROW_M         = 3.0             # heading arrow length in world metres
SPEED_WINDOW_S  = 0.5             # seconds for finite-diff speed estimate
EMA_ALPHA_SPD   = 0.15
CONF_DET        = 0.30
KP_CONF_MIN     = 0.40            # minimum keypoint confidence to use pose


# ── helpers ───────────────────────────────────────────────────────
def track_colour(tid: int) -> tuple:
    import colorsys
    h = (tid * 0.618033988) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_arrow(frame, geo: Homography, gx: float, gz: float,
               heading_deg: float, arrow_m: float, colour, thickness=2):
    psi = math.radians(heading_deg)
    tip_x = gx + arrow_m * math.sin(psi)
    tip_z = gz + arrow_m * math.cos(psi)
    src_px = geo.to_pixel(gx, gz)
    dst_px = geo.to_pixel(tip_x, tip_z)
    cv2.arrowedLine(frame, src_px, dst_px, colour, thickness, tipLength=0.3)


def src_colour(src: str):
    if src and "pose" in src and "traj" not in src:
        return (0, 255, 255)   # yellow — pose dominant
    if src and "traj" in src and "pose" not in src:
        return (255, 255, 0)   # cyan   — traj dominant
    if src and "+" in src:
        return (0, 200, 255)   # orange — fused
    return (160, 160, 160)     # grey   — hold / reject


# ── speed from BEV position history ──────────────────────────────
class SpeedTracker:
    def __init__(self, fps: float):
        from collections import deque
        self._fps = fps
        self._buf: dict = {}  # tid -> deque of (t, x, z)
        self._ema: dict = {}
        self._win = max(2, int(SPEED_WINDOW_S * fps))

    def update(self, tid: int, t: float, x: float, z: float) -> float:
        from collections import deque
        if tid not in self._buf:
            self._buf[tid] = deque(maxlen=self._win)
            self._ema[tid] = 0.0
        self._buf[tid].append((t, x, z))
        buf = self._buf[tid]
        if len(buf) < 2:
            return 0.0
        t0, x0, z0 = buf[0]; t1, x1, z1 = buf[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return self._ema[tid]
        raw = math.hypot(x1 - x0, z1 - z0) / dt * 3.6
        self._ema[tid] = EMA_ALPHA_SPD * raw + (1 - EMA_ALPHA_SPD) * self._ema[tid]
        return self._ema[tid]

    def drop(self, tid: int):
        self._buf.pop(tid, None)
        self._ema.pop(tid, None)


# ── main loop ─────────────────────────────────────────────────────
def run(args):
    # calibration
    if not os.path.exists(args.h_path):
        raise SystemExit(
            f"Calibration not found: {args.h_path}\n"
            "Run manual_calibrate.py first (see experiment-new-model/)."
        )
    geo = Homography(args.h_path, args.track_path if os.path.exists(args.track_path) else None)

    # detector (YOLO11n — lightweight, ByteTrack built-in)
    detector = YOLO(args.det_weights)

    # pose model (optional — skip with --no_pose or if weights missing)
    use_pose = not args.no_pose and os.path.exists(args.pose_weights)
    if not args.no_pose and not os.path.exists(args.pose_weights):
        print(f"[warn] pose weights not found: {args.pose_weights}")
        print("[warn] running trajectory-only (use --no_pose to suppress this warning)")
    pose_inf = PoseInference(args.pose_weights, device=args.device,
                             conf=CONF_DET, imgsz=args.imgsz) if use_pose else None

    # heading estimators
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[main] video: {args.video}  {W}x{H}@{fps:.1f}fps  {n_fr} frames")
    print(f"[main] pose: {'ON  ' + args.pose_weights if use_pose else 'OFF (trajectory only)'}")

    pose_h  = PoseHeading(geo, kp_conf_min=KP_CONF_MIN) if use_pose else None
    traj_h  = TrajectoryHeading(dt=1.0 / fps)
    fusion  = HeadingFusion(ema_alpha=0.25)
    speed_t = SpeedTracker(fps)

    os.makedirs("out", exist_ok=True)
    out_path = args.output or os.path.join(
        "out", os.path.splitext(os.path.basename(args.video))[0] + "_heading.mp4"
    )
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps

        # ── detection + ByteTrack ──────────────────────────────
        det_res = detector.track(
            frame, persist=True,
            classes=VEHICLE_CLASSES,
            conf=CONF_DET, iou=0.45,
            device=args.device,
            imgsz=args.imgsz,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        # ── pose inference on same frame ───────────────────────
        pose_dets = pose_inf(frame) if use_pose else []

        # build a lookup: nearest pose-det to each track bbox center
        def best_pose_for_box(bx1, by1, bx2, by2):
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            best, best_d = None, 1e9
            for pd in pose_dets:
                px1, py1, px2, py2 = pd["bbox"]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                d = math.hypot(pcx - bcx, pcy - bcy)
                if d < best_d:
                    best_d, best = d, pd
            # only accept if centers are close relative to box size
            box_diag = math.hypot(bx2 - bx1, by2 - by1)
            return best if best_d < box_diag * 0.6 else None

        # ── per-track update ───────────────────────────────────
        active_ids = set()
        if det_res.boxes is not None and det_res.boxes.id is not None:
            boxes = det_res.boxes.xyxy.cpu().numpy()
            ids   = det_res.boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = box
                active_ids.add(tid)

                # ground-contact point = bottom-center of bbox
                gx, gz = geo.to_ground((x1 + x2) / 2, y2)

                # speed
                spd = speed_t.update(tid, t, gx, gz)

                # trajectory heading (CTRV EKF)
                traj_est = traj_h.update(tid, gx, gz)

                # pose heading (if available)
                pose_est = None
                if use_pose:
                    pd = best_pose_for_box(x1, y1, x2, y2)
                    if pd is not None:
                        pose_est = pose_h.estimate(pd)

                # fuse
                result = fusion.fuse(tid, pose=pose_est, traj=traj_est)

                # ── draw ──────────────────────────────────────
                col = track_colour(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)

                if result is not None:
                    hdeg, hconf, hsrc = result
                    arrow_col = src_colour(hsrc)
                    if hsrc not in ("hold", "reject"):
                        draw_arrow(frame, geo, gx, gz, hdeg, ARROW_M, arrow_col, 2)
                    label = f"ID{tid} {spd:.0f}km/h {hdeg:.0f}deg [{hsrc[0].upper()}]"
                else:
                    label = f"ID{tid} {spd:.0f}km/h --deg"

                # label above box
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                lx, ly = int(x1), max(int(y1) - 4, th + 2)
                cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 2, ly + 2), col, -1)
                cv2.putText(frame, label, (lx + 1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 0), 1, cv2.LINE_AA)

                # keypoint dots (debug)
                if use_pose and args.show_kpts:
                    pd = best_pose_for_box(x1, y1, x2, y2)
                    if pd is not None:
                        kp_cols = [(0,0,255),(255,0,0),(0,255,0),(0,255,255)]
                        for ki, (kx, ky) in enumerate(pd["kpts"]):
                            if pd["kpt_conf"][ki] >= KP_CONF_MIN:
                                cv2.circle(frame, (int(kx), int(ky)), 3, kp_cols[ki], -1)

        # prune dead tracks
        for tid in list(fusion._state):
            if tid not in active_ids:
                fusion.drop(tid)
                traj_h.drop(tid)
                speed_t.drop(tid)

        # ── HUD ───────────────────────────────────────────────
        mode = "POSE+TRAJ" if use_pose else "TRAJ ONLY"
        cv2.putText(frame, f"heading: {mode}  frame {fi}/{n_fr}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(frame)
        fi += 1
        if fi % 100 == 0:
            print(f"  frame {fi}/{n_fr} ({100*fi/max(1,n_fr):.0f}%)")

    cap.release()
    writer.release()
    print(f"[main] done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        default=DEFAULT_VIDEO)
    ap.add_argument("--pose_weights", default=DEFAULT_POSE_W)
    ap.add_argument("--det_weights",  default=DEFAULT_DET_W)
    ap.add_argument("--h_path",       default=DEFAULT_H_PATH)
    ap.add_argument("--track_path",   default=DEFAULT_TRACK_PATH)
    ap.add_argument("--output",       default="")
    ap.add_argument("--imgsz",        type=int, default=640)
    ap.add_argument("--device",       default="0")
    ap.add_argument("--no_pose",      action="store_true",
                    help="skip pose model; use trajectory heading only")
    ap.add_argument("--show_kpts",    action="store_true",
                    help="draw raw pose keypoints for debugging")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
