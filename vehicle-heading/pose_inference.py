"""
pose_inference.py — wraps the trained 4-keypoint vehicle-pose model.

Single forward pass per frame: detection + 4 keypoints per vehicle. The model
is custom-trained (SKoPe3D, single class `vehicle`), so no COCO class filter is
needed.

Keypoint order (must match training / vehicle_pose.yaml):
    0 front-center, 1 rear-center, 2 front-left, 3 front-right

Returns, per detection, a dict:
    {
      "bbox":     (x1, y1, x2, y2)   # pixels
      "conf":     float              # detection confidence
      "kpts":     np.ndarray (4, 2)  # keypoint pixel coords
      "kpt_conf": np.ndarray (4,)    # per-keypoint confidence [0, 1]
    }
"""

import numpy as np
from ultralytics import YOLO

# keypoint indices (semantic names for readers / consumers)
KP_FRONT_CENTER = 0
KP_REAR_CENTER = 1
KP_FRONT_LEFT = 2
KP_FRONT_RIGHT = 3


class PoseInference:
    def __init__(self, weights: str, device: int | str = 0,
                 conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame) -> list[dict]:
        res = self.model.predict(
            frame, device=self.device, conf=self.conf,
            imgsz=self.imgsz, verbose=False,
        )[0]

        out: list[dict] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        # keypoints: (N, 4, 2) xy, and (N, 4) confidence
        kxy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        kcf = (
            res.keypoints.conf.cpu().numpy()
            if (res.keypoints is not None and res.keypoints.conf is not None)
            else None
        )

        for i in range(len(boxes)):
            kp = kxy[i] if kxy is not None else np.zeros((4, 2), np.float32)
            kc = kcf[i] if kcf is not None else np.ones(4, np.float32)
            out.append({
                "bbox": tuple(float(v) for v in boxes[i]),
                "conf": float(confs[i]),
                "kpts": kp.astype(np.float32),
                "kpt_conf": kc.astype(np.float32),
            })
        return out
