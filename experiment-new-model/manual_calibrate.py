import cv2
import numpy as np


class ManualCalibrator:
    """Two-phase manual calibration on a single frame.

    Phase 1 — Homography rectangle (4 clicks):
        Click bl, br, tr, tl on a clean rectangular road segment (preferably
        bounded by two parallel lane stripes). Used to fit H.

    Phase 2 — Tracking polygon (N clicks, N >= 3):
        Click any number of points outlining the *entire driveable area*
        where cars should be tracked. Press Enter to finish.
    """

    POINT_LABELS = ["bottom-left", "bottom-right", "top-right", "top-left"]
    DISPLAY_MAX  = (1280, 720)   # max window size on screen

    def __init__(self, lane_width_m=7.0, road_depth_m=10.0):
        self.lane_width_m = lane_width_m
        self.road_depth_m = road_depth_m

        self.H = None
        self.src_pts = None      # 4 homography points
        self.track_poly = None   # N tracking-region points
        self.shape = None

    def _scale_for_display(self, frame):
        """Return (display_frame, scale) where scale = display/original."""
        mw, mh = self.DISPLAY_MAX
        h, w = frame.shape[:2]
        scale = min(mw / w, mh / h, 1.0)
        if scale < 1.0:
            dw, dh = int(w * scale), int(h * scale)
            return cv2.resize(frame, (dw, dh)), scale
        return frame.copy(), 1.0

    # ─── Phase 1: 4-point rectangle for homography ───────────
    def _pick_homography_points(self, frame):
        display, scale = self._scale_for_display(frame)
        clicks_d = []   # clicks in display coords
        window = "Phase 1/2 — Homography: click bl, br, tr, tl  (r=reset, q=quit)"

        def redraw():
            img = display.copy()
            for i, (x, y) in enumerate(clicks_d):
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(img, self.POINT_LABELS[i], (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(clicks_d) >= 2:
                for i in range(len(clicks_d) - 1):
                    cv2.line(img, clicks_d[i], clicks_d[i + 1], (0, 255, 0), 2)
            if len(clicks_d) == 4:
                cv2.line(img, clicks_d[3], clicks_d[0], (0, 255, 0), 2)
            next_idx = len(clicks_d)
            if next_idx < 4:
                msg = f"Phase 1/2 — Click {self.POINT_LABELS[next_idx]}"
            else:
                msg = "Phase 1/2 — ENTER to confirm, r to reset"
            cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.imshow(window, img)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks_d) < 4:
                clicks_d.append((x, y))
                redraw()

        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, display)
        cv2.waitKey(1)
        cv2.setMouseCallback(window, on_mouse)
        redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                clicks_d.clear()
                redraw()
            elif key == ord('q'):
                cv2.destroyWindow(window)
                raise RuntimeError("Calibration cancelled by user")
            elif key in (13, 10) and len(clicks_d) == 4:
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return np.float32([(x / scale, y / scale) for x, y in clicks_d])

    # ─── Phase 2: N-point polygon for tracking region ─────────
    def _pick_tracking_polygon(self, frame, src_pts):
        display, scale = self._scale_for_display(frame)
        clicks_d = []
        src_d = (src_pts * scale).astype(np.int32)
        window = "Phase 2/2 — Tracking region: click N points  (Enter=done, r=reset, u=undo, q=quit)"

        def redraw():
            img = display.copy()
            cv2.polylines(img, [src_d], True, (0, 200, 200), 1)
            if len(clicks_d) >= 1:
                for (x, y) in clicks_d:
                    cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
            if len(clicks_d) >= 2:
                for i in range(len(clicks_d) - 1):
                    cv2.line(img, clicks_d[i], clicks_d[i + 1], (0, 255, 255), 2)
            if len(clicks_d) >= 3:
                cv2.line(img, clicks_d[-1], clicks_d[0], (0, 255, 255), 1)
                overlay = img.copy()
                cv2.fillPoly(overlay, [np.array(clicks_d, dtype=np.int32)],
                             (0, 255, 255))
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            msg = f"Phase 2/2 — {len(clicks_d)} pts  (Enter when done, need >=3)"
            cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.imshow(window, img)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks_d.append((x, y))
                redraw()

        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, display)
        cv2.waitKey(1)
        cv2.setMouseCallback(window, on_mouse)
        redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                clicks_d.clear()
                redraw()
            elif key == ord('u') and clicks_d:
                clicks_d.pop()
                redraw()
            elif key == ord('q'):
                cv2.destroyWindow(window)
                raise RuntimeError("Calibration cancelled by user")
            elif key in (13, 10) and len(clicks_d) >= 3:
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return np.array([(int(x / scale), int(y / scale)) for x, y in clicks_d],
                        dtype=np.int32)

    # ─── Homography fit ───────────────────────────────────────
    def compute_homography(self, src):
        dst = np.float32([
            [0,                 self.road_depth_m],
            [self.lane_width_m, self.road_depth_m],
            [self.lane_width_m, 0                ],
            [0,                 0                ],
        ])
        H, _ = cv2.findHomography(src, dst)
        return H

    # ─── Main entry ───────────────────────────────────────────
    def calibrate(self, cap, save_path="H_manual.npy", frame_idx=0):
        if not cap.isOpened():
            raise RuntimeError("VideoCapture is not opened — check the input path")

        original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx != original_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if original_pos > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} for calibration")

        self.shape = frame.shape[:2]
        self.src_pts = self._pick_homography_points(frame)
        self.H = self.compute_homography(self.src_pts)
        self.track_poly = self._pick_tracking_polygon(frame, self.src_pts)

        if save_path:
            np.save(save_path, self.H)
            print(f"[Calibration] H saved to {save_path}")
            stem = save_path.replace("H_", "") if "H_" in save_path else "manual.npy"
            src_path   = "src_"   + stem
            track_path = "track_" + stem
            np.save(src_path, self.src_pts)
            np.save(track_path, self.track_poly)
            print(f"[Calibration] src_pts saved to {src_path}")
            print(f"[Calibration] track_poly saved to {track_path} ({len(self.track_poly)} pts)")

        return self.H, self.src_pts

    def warp(self, frame, out_size=(500, 800)):
        if self.H is None:
            raise RuntimeError("Calibrator has not been run yet — call .calibrate(cap) first")
        return cv2.warpPerspective(frame, self.H, out_size)


if __name__ == "__main__":
    cap = cv2.VideoCapture("../_in/accident_cm_in_p10.mp4")

    calib = ManualCalibrator(lane_width_m=7.0, road_depth_m=10.0)
    H, src = calib.calibrate(cap, save_path="H_manual.npy")
    print("H =")
    print(H)
    print("src_pts =")
    print(src)
    print(f"track_poly = {len(calib.track_poly)} points")

    # ── BEV verification ─────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        dbg = frame.copy()
        pts = src.astype(int)
        for i in range(4):
            cv2.line(dbg, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            cv2.circle(dbg, tuple(pts[i]), 6, (0, 0, 255), -1)
        cv2.polylines(dbg, [calib.track_poly], True, (0, 255, 255), 2)
        cv2.imwrite("calib_debug.png", dbg)

        px_per_m = 20
        out_w = int(calib.lane_width_m * px_per_m)
        out_h = int(calib.road_depth_m * px_per_m)
        S    = np.diag([px_per_m, px_per_m, 1.0]).astype(np.float64)
        H_px = S @ calib.H
        bev  = cv2.warpPerspective(frame, H_px, (out_w, out_h))
        cv2.imwrite("bev_test.png", bev)
        print(f"saved calib_debug.png and bev_test.png ({out_w}x{out_h} px)")
    cap.release()
