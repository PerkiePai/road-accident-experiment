"""
Geometry helpers — pixel <-> BEV (bird's-eye, ground-plane metres) projection.

The homography H maps image pixels (px, py) to world ground coordinates
(x_m, z_m) in metres, where z is depth/forward along the road and x is lateral.
This mirrors `experiment-new-model/detect.py:project_to_ground` so heading is
computed in the same world frame the speed/CTRV stack already uses.

Heading convention (shared with ctrv_filter.CTRVFilter):
    psi = 0  -> +z (forward/away), increasing toward +x (lateral).
    heading = atan2(dx, dz)   # note: x first, z second
"""

import numpy as np
import cv2


class Homography:
    """Loads a calibration triple (H/src/track .npy) and projects points."""

    def __init__(self, h_path: str, track_path: str | None = None):
        self.H = np.load(h_path)
        self.H_inv = np.linalg.inv(self.H)
        self.road_poly = (
            np.load(track_path).astype(np.int32) if track_path else None
        )

    def to_ground(self, px: float, py: float) -> tuple[float, float]:
        """Image pixel -> world (x_m, z_m)."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        xz = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return float(xz[0]), float(xz[1])

    def to_pixel(self, x_m: float, z_m: float) -> tuple[int, int]:
        """World (x_m, z_m) -> image pixel."""
        pt = np.array([[[x_m, z_m]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, self.H_inv)[0, 0]
        return int(round(px[0])), int(round(px[1]))

    def on_road(self, px: float, py: float) -> bool:
        """Is image pixel inside the driveable tracking polygon?"""
        if self.road_poly is None:
            return True
        return cv2.pointPolygonTest(self.road_poly, (float(px), float(py)), False) >= 0


def heading_from_vec(dx: float, dz: float) -> float:
    """World displacement (dx, dz) -> heading in degrees [0, 360).
    Matches CTRVFilter: psi=0 -> +z, increasing toward +x."""
    return float(np.degrees(np.arctan2(dx, dz)) % 360.0)
