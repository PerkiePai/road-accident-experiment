"""
heading_estimator.py — per-track vehicle heading from two complementary sources.

  PoseHeading       instantaneous, single-frame: front->rear keypoint vector in
                    BEV. Works when stationary / newly-detected (trajectory fails).
  TrajectoryHeading motion-based: reuses the proven ctrv_filter.CTRVFilter
                    (Constant Turn Rate & Velocity EKF) on the BEV ground-contact
                    point. Smooth at speed; undefined when slow/new.
  HeadingFusion     confidence-weighted unit-circle blend + circular EMA, with a
                    low-confidence outlier guard and hold-on-coast.

All angles are degrees [0, 360), psi=0 -> +z, increasing toward +x
(see geometry.heading_from_vec / ctrv_filter).
"""

import os
import sys
import math
import numpy as np

from geometry import Homography, heading_from_vec
from pose_inference import (
    KP_FRONT_CENTER, KP_REAR_CENTER, KP_FRONT_LEFT, KP_FRONT_RIGHT,
)

# Reuse the existing CTRV EKF from experiment-new-model without copying it.
_EXP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiment-new-model",
)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)
from ctrv_filter import CTRVFilter  # noqa: E402


# ── circular helpers ────────────────────────────────────────────────
def _to_vec(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


def _from_vec(c: float, s: float) -> float:
    return math.degrees(math.atan2(s, c)) % 360.0


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (degrees), in [0, 180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


# ── 1. instantaneous pose heading ───────────────────────────────────
class PoseHeading:
    """Heading from the front->rear keypoint vector projected into BEV."""

    def __init__(self, geo: Homography, kp_conf_min: float = 0.5):
        self.geo = geo
        self.kp_conf_min = kp_conf_min

    def estimate(self, det: dict):
        """Return (heading_deg, confidence) or None if keypoints unreliable."""
        kp, kc = det["kpts"], det["kpt_conf"]
        cf, cr = kc[KP_FRONT_CENTER], kc[KP_REAR_CENTER]
        if cf < self.kp_conf_min or cr < self.kp_conf_min:
            return None

        fx, fz = self.geo.to_ground(*kp[KP_FRONT_CENTER])
        rx, rz = self.geo.to_ground(*kp[KP_REAR_CENTER])
        if not math.isfinite(fx) or not math.isfinite(rx):
            return None

        heading = heading_from_vec(fx - rx, fz - rz)
        conf = float(min(cf, cr))
        return heading, conf


# ── 2. trajectory heading (CTRV EKF) ────────────────────────────────
class TrajectoryHeading:
    """One CTRVFilter per track; confidence scales with net displacement."""

    SAT_DISP_M = 5.0  # displacement (m) at which confidence saturates to 1.0

    def __init__(self, dt: float):
        self.dt = dt
        self._filters: dict[int, CTRVFilter] = {}
        self._first: dict[int, tuple[float, float]] = {}
        self._last: dict[int, tuple[float, float]] = {}

    def update(self, tid: int, x_m: float, z_m: float):
        """Feed a BEV ground point; return (heading_deg, confidence) or None."""
        f = self._filters.get(tid)
        if f is None:
            f = CTRVFilter(dt=self.dt)
            f.init(x_m, z_m)
            self._filters[tid] = f
            self._first[tid] = (x_m, z_m)

        f.update(x_m, z_m)
        self._last[tid] = (x_m, z_m)
        if not f.has_heading:
            return None

        fx, fz = self._first[tid]
        disp = math.hypot(x_m - fx, z_m - fz)
        conf = min(1.0, disp / self.SAT_DISP_M)
        return f.heading_deg, conf

    def drop(self, tid: int):
        self._filters.pop(tid, None)
        self._first.pop(tid, None)
        self._last.pop(tid, None)


# ── 3. fusion + smoothing ───────────────────────────────────────────
class HeadingFusion:
    """Confidence-weighted circular blend of available sources, EMA-smoothed."""

    def __init__(self, ema_alpha: float = 0.3,
                 outlier_deg: float = 45.0, outlier_conf: float = 0.4):
        self.ema_alpha = ema_alpha
        self.outlier_deg = outlier_deg
        self.outlier_conf = outlier_conf
        self._state: dict[int, dict] = {}  # tid -> {heading, conf, src}

    def fuse(self, tid: int, pose=None, traj=None, coasted: bool = False):
        """
        pose / traj : (heading_deg, conf) or None.
        coasted     : track is predicted-through-occlusion -> hold last heading.
        Returns (heading_deg, conf, src) or None when nothing is available.
        """
        prev = self._state.get(tid)

        # Coasted: never update from a stale/absent measurement; hold last good.
        if coasted and prev is not None:
            return prev["heading"], prev["conf"], "hold"

        sources = [("pose", pose), ("traj", traj)]
        cx = sy = wsum = 0.0
        used = []
        for name, est in sources:
            if est is None:
                continue
            deg, w = est
            if w <= 0:
                continue
            c, s = _to_vec(deg)
            cx += w * c
            sy += w * s
            wsum += w
            used.append(name)

        if wsum <= 0:
            # nothing this frame; keep prior if any
            if prev is not None:
                return prev["heading"], prev["conf"], "hold"
            return None

        raw = _from_vec(cx, sy)
        conf = min(1.0, wsum)
        src = "+".join(used)

        # Outlier guard: reject a big jump that arrives with weak confidence.
        if prev is not None and conf < self.outlier_conf:
            if _ang_diff(raw, prev["heading"]) > self.outlier_deg:
                return prev["heading"], prev["conf"], "reject"

        # Circular EMA toward the new reading.
        if prev is None:
            sm = raw
        else:
            pc, ps = _to_vec(prev["heading"])
            nc, ns = _to_vec(raw)
            a = self.ema_alpha
            sm = _from_vec((1 - a) * pc + a * nc, (1 - a) * ps + a * ns)

        self._state[tid] = {"heading": sm, "conf": conf, "src": src}
        return sm, conf, src

    def drop(self, tid: int):
        self._state.pop(tid, None)

    def prune(self, active_ids):
        for tid in list(self._state):
            if tid not in active_ids:
                self._state.pop(tid, None)
