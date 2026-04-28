"""
Vision Module — Ripe-Tomato Detection (importable)
====================================================
Provides TomatoDetector class that can be used by both the integrated
main.py and the simulation GUI without opening a camera or running a loop.
"""

import cv2
import math
import os
import numpy as np
from ultralytics import YOLO

# ── Model path resolution ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_model_path():
    candidates = [
        os.path.join(_SCRIPT_DIR, 'runs/detect/train_v2/weights/best.pt'),
        os.path.join(_SCRIPT_DIR, 'runs/detect/train6/weights/best.pt'),
        os.path.join(_SCRIPT_DIR, 'yolov8s.pt'),
        os.path.join(_SCRIPT_DIR, 'yolov8n.pt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


class TomatoDetector:
    """Encapsulates YOLO model + false-positive filters + 3D coordinate math."""

    def __init__(self, model_path=None, confidence=0.70,
                 focal_length_px=700, real_diameter_cm=7.0):
        self.model_path = model_path or _resolve_model_path()
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.focal_length_px = focal_length_px
        self.real_diameter_cm = real_diameter_cm

        # Filter parameters
        self.min_bbox_area = 1_500
        self.max_bbox_area = 180_000
        self.min_aspect = 0.45
        self.max_aspect = 2.20
        self.min_red_ratio = 0.18
        self.edge_margin_px = 5   # pixels — skip bboxes this close to frame edge

        # HSV red ranges
        self._lo1 = np.array([0,   80,  50], dtype=np.uint8)
        self._hi1 = np.array([15, 255, 255], dtype=np.uint8)
        self._lo2 = np.array([160, 80,  50], dtype=np.uint8)
        self._hi2 = np.array([179, 255, 255], dtype=np.uint8)

    # ── Depth estimation ───────────────────────────────────────────────────────
    def estimate_depth_cm(self, pixel_width, pixel_height=None):
        """
        Estimate depth via the pinhole model.
        Uses the geometric mean of width and height when both are given —
        this is more robust than width alone for off-centre or skewed bboxes.
        """
        if pixel_height is not None and pixel_height > 0:
            pixel_diameter = math.sqrt(float(pixel_width) * float(pixel_height))
        else:
            pixel_diameter = float(pixel_width)
        if pixel_diameter <= 0:
            return None
        return (self.real_diameter_cm * self.focal_length_px) / pixel_diameter

    # ── Filters ────────────────────────────────────────────────────────────────
    def _passes_size_shape(self, x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        if h <= 0:
            return False
        area = w * h
        aspect = w / h
        return (self.min_bbox_area <= area <= self.max_bbox_area and
                self.min_aspect <= aspect <= self.max_aspect)

    def _passes_colour(self, roi_bgr):
        if roi_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, self._lo1, self._hi1)
        mask |= cv2.inRange(hsv, self._lo2, self._hi2)
        ratio = np.count_nonzero(mask) / mask.size
        return ratio >= self.min_red_ratio

    def _pixel_to_xyz(self, cx, cy, z_cm, fw, fh):
        x_cm = (cx - fw / 2.0) * z_cm / self.focal_length_px
        y_cm = (cy - fh / 2.0) * z_cm / self.focal_length_px
        return round(x_cm, 2), round(y_cm, 2), round(z_cm, 2)

    # ── Main detection on a single frame ───────────────────────────────────────
    def detect(self, frame):
        """
        Run detection on a BGR frame.
        Returns list of dicts, each with:
          confidence, bbox_px, center_px, xyz_cm
        """
        fh, fw = frame.shape[:2]
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if not self._passes_size_shape(x1, y1, x2, y2):
                continue

            # Skip detections whose bounding box touches the frame edge —
            # a clipped bbox gives an unreliable centre and depth estimate.
            m = self.edge_margin_px
            if x1 < m or y1 < m or x2 > fw - m or y2 > fh - m:
                continue

            roi = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
            if not self._passes_colour(roi):
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Use geometric mean of width and height for more accurate depth
            z_cm = self.estimate_depth_cm(x2 - x1, y2 - y1)
            if z_cm is None:
                continue

            x_cm, y_cm, z_cm = self._pixel_to_xyz(cx, cy, z_cm, fw, fh)

            detections.append({
                "confidence": round(conf, 3),
                "bbox_px":    [x1, y1, x2, y2],
                "center_px":  [cx, cy],
                "xyz_cm":     {"x": x_cm, "y": y_cm, "z": z_cm},
            })

        return detections

    # ── Annotate frame with detections ─────────────────────────────────────────
    def annotate(self, frame, detections):
        """Draw detection boxes and labels on a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_px"]
            cx, cy = det["center_px"]
            conf = det["confidence"]
            xyz = det["xyz_cm"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 200, 0), -1)

            top = f"Ripe {conf:.2f} | Z:{xyz['z']:.1f}cm"
            cv2.putText(annotated, top, (x1, max(y1 - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

            bot = f"X:{xyz['x']:+.1f}  Y:{xyz['y']:+.1f}"
            cv2.putText(annotated, bot, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 210, 0), 2)
        return annotated

    # ── Convenience: convert detection xyz_cm to metres ────────────────────────
    @staticmethod
    def xyz_cm_to_metres(xyz_cm_dict):
        """Convert {x, y, z} in cm to metres as a numpy array."""
        return np.array([
            xyz_cm_dict["x"] / 100.0,
            xyz_cm_dict["y"] / 100.0,
            xyz_cm_dict["z"] / 100.0,
        ], dtype=float)
