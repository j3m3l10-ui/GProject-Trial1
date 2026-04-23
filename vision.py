"""
Vision Module — Ripe-Tomato Detection (importable)
====================================================
Provides TomatoDetector class that can be used by both the integrated
main.py and the simulation GUI without opening a camera or running a loop.

False-positive rejection pipeline (applied after YOLO inference):
  1. Confidence threshold (default 0.70)
  2. Bounding-box size & aspect-ratio gates
  3. HSV red-colour ratio gate
  4. *Texture* gate — reject flat, uniformly-coloured surfaces (e.g. a red ball
     or a red cup).  Real tomatoes have a visible stem patch, surface gloss and
     colour gradients, so their hue & saturation standard deviations are
     noticeably higher than those of a painted/plastic sphere.
  5. *Shape* gate via red-blob contour — rejects objects that are TOO perfectly
     circular (a ball's silhouette has circularity > ~0.92; ripe tomatoes tend
     to sit between 0.70 and 0.90 because of the calyx, shoulders and stem).
  6. *Depth* from the red-blob diameter, not the raw bbox — gives a much more
     reliable pixel size (and therefore distance) when the YOLO bbox is loose.

Depth / 3-D mapping:
  Pinhole model:  Z = (real_diameter_cm * focal_length_px) / pixel_diameter
  Pixel→camera: X = (cx - W/2) * Z / f,  Y = (cy - H/2) * Z / f
  FOCAL_LENGTH_PX must be calibrated per camera (see calibrate_focal_length()).
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO

# ── Model path resolution ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Physical reach of the 5-DOF arm (tip of gripper to base) — used by main.py
# to flag tomatoes that are visible but too far away to be harvested.
ARM_REACH_CM = 36.0


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
                 focal_length_px=700, real_diameter_cm=7.0,
                 infer_imgsz=416):
        """
        Args:
            model_path: path to .pt weights (auto-resolved if None)
            confidence: YOLO confidence threshold
            focal_length_px: camera focal length in pixels — MUST be calibrated
                for accurate distance measurement (see calibrate_focal_length).
            real_diameter_cm: assumed real-world tomato diameter (cm)
            infer_imgsz: YOLO inference image size.  On a Raspberry Pi 5, 416
                runs roughly 3x faster than 800 with minimal recall loss —
                this is the single biggest lever for live-camera FPS.
        """
        self.model_path = model_path or _resolve_model_path()
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.focal_length_px = focal_length_px
        self.real_diameter_cm = real_diameter_cm
        self.infer_imgsz = infer_imgsz

        # Size / aspect gates
        self.min_bbox_area = 1_500
        self.max_bbox_area = 180_000
        self.min_aspect = 0.45
        self.max_aspect = 2.20

        # Red-ratio gate
        self.min_red_ratio = 0.18

        # Texture gates — reject flat plastic/painted red surfaces.
        # A ripe tomato has stem/highlight variation, so its saturation stddev
        # inside the red blob is typically >= ~12, hue stddev >= ~3.
        self.min_sat_std = 12.0
        self.min_hue_std = 2.5

        # Shape gate — reject "too perfect" spheres (balls).
        # Circularity = 4*pi*A / P^2.  Perfect circle = 1.0.
        # Real ripe tomatoes measured at 0.70–0.90 (calyx + shoulders break it).
        # We reject anything ABOVE max_circularity.
        self.max_circularity = 0.93
        self.min_circularity = 0.55

        # HSV red ranges (red wraps around H=0)
        self._lo1 = np.array([0,   80,  50], dtype=np.uint8)
        self._hi1 = np.array([15, 255, 255], dtype=np.uint8)
        self._lo2 = np.array([160, 80,  50], dtype=np.uint8)
        self._hi2 = np.array([179, 255, 255], dtype=np.uint8)

    # ── Depth estimation ───────────────────────────────────────────────────────
    def estimate_depth_cm(self, pixel_diameter):
        """Pinhole-model depth: Z = (real_size * f) / pixel_size."""
        if pixel_diameter <= 0:
            return None
        return (self.real_diameter_cm * self.focal_length_px) / pixel_diameter

    # ── Filters ────────────────────────────────────────────────────────────────
    def _passes_size_shape(self, x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        if h <= 0 or w <= 0:
            return False
        area = w * h
        aspect = w / h
        return (self.min_bbox_area <= area <= self.max_bbox_area and
                self.min_aspect <= aspect <= self.max_aspect)

    def _red_mask(self, roi_bgr):
        """Binary mask of red pixels in ROI, cleaned with a morphological open."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lo1, self._hi1) | \
               cv2.inRange(hsv, self._lo2, self._hi2)
        # Remove salt-and-pepper noise so the contour/blob is meaningful
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return hsv, mask

    def _passes_colour(self, mask):
        if mask.size == 0:
            return False
        return (np.count_nonzero(mask) / mask.size) >= self.min_red_ratio

    def _passes_texture(self, hsv, mask):
        """Reject flat red surfaces (plastic balls, painted cups)."""
        if np.count_nonzero(mask) < 200:
            return False
        h, s, _ = cv2.split(hsv)
        red_px_s = s[mask > 0]
        red_px_h = h[mask > 0]
        if red_px_s.size < 200:
            return False
        sat_std = float(np.std(red_px_s))
        # Hue is circular; unwrap the wrap-around reds (values near 179) to
        # a negative range so std is meaningful.
        hue_unwrapped = np.where(red_px_h > 90,
                                 red_px_h.astype(np.int16) - 180,
                                 red_px_h.astype(np.int16))
        hue_std = float(np.std(hue_unwrapped))
        return sat_std >= self.min_sat_std and hue_std >= self.min_hue_std

    def _blob_diameter_and_shape(self, mask):
        """
        Fit the largest red blob. Returns (pixel_diameter, circularity) or
        (None, None) if no usable blob is found.  pixel_diameter is derived
        from the equivalent-area circle, which is far more robust to bbox
        looseness than (x2 - x1).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, closed=True)
        if area < 150 or perim <= 0:
            return None, None
        # Equivalent-area diameter: A = pi*(d/2)^2  ⇒  d = 2*sqrt(A/pi)
        diam = 2.0 * np.sqrt(area / np.pi)
        circularity = 4.0 * np.pi * area / (perim * perim)
        return diam, circularity

    def _pixel_to_xyz(self, cx, cy, z_cm, fw, fh):
        x_cm = (cx - fw / 2.0) * z_cm / self.focal_length_px
        y_cm = (cy - fh / 2.0) * z_cm / self.focal_length_px
        return round(x_cm, 2), round(y_cm, 2), round(z_cm, 2)

    # ── Main detection on a single frame ───────────────────────────────────────
    def detect(self, frame):
        """
        Run detection on a BGR frame.
        Returns list of dicts, each with:
          confidence, bbox_px, center_px, xyz_cm, distance_cm, reachable
        """
        fh, fw = frame.shape[:2]
        # A smaller imgsz is the single biggest speed win on Raspberry Pi 5.
        results = self.model(frame, conf=self.confidence,
                             imgsz=self.infer_imgsz, verbose=False)
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Clip to frame before any ROI work.
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(fw, x2), min(fh, y2)

            if not self._passes_size_shape(x1c, y1c, x2c, y2c):
                continue

            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                continue

            hsv, mask = self._red_mask(roi)

            # Colour gate
            if not self._passes_colour(mask):
                continue

            # Texture gate — rejects plain red balls / cups / cloth
            if not self._passes_texture(hsv, mask):
                continue

            # Shape + diameter from the red blob itself
            blob_diam_px, circ = self._blob_diameter_and_shape(mask)
            if blob_diam_px is None:
                continue
            if circ > self.max_circularity or circ < self.min_circularity:
                # "Too round" → likely a ball;  "too jagged" → likely clutter.
                continue

            # Depth estimation — drive it from the red-blob's equivalent-area
            # diameter (much more robust than a loose YOLO bbox width).  We
            # still cap at the min bbox side to guard against the rare case
            # where the red blob bleeds into adjacent red clutter inside the
            # bbox and becomes artificially large.
            bbox_min_side = float(min(x2c - x1c, y2c - y1c))
            pixel_diameter = min(blob_diam_px, bbox_min_side)
            z_cm = self.estimate_depth_cm(pixel_diameter)
            if z_cm is None:
                continue

            cx, cy = (x1c + x2c) // 2, (y1c + y2c) // 2
            x_cm, y_cm, z_cm = self._pixel_to_xyz(cx, cy, z_cm, fw, fh)

            # 3-D Euclidean distance from camera origin — used for ranking
            # the tomatoes (nearest first) and for the arm-reach check.
            distance_cm = float(round(
                np.linalg.norm([x_cm, y_cm, z_cm]), 2))
            reachable = distance_cm <= ARM_REACH_CM

            detections.append({
                "confidence":  round(conf, 3),
                "bbox_px":     [x1c, y1c, x2c, y2c],
                "center_px":   [cx, cy],
                "xyz_cm":      {"x": x_cm, "y": y_cm, "z": z_cm},
                "distance_cm": distance_cm,
                "circularity": round(float(circ), 3),
                "reachable":   bool(reachable),
            })

        return detections

    # ── Annotate frame with detections ─────────────────────────────────────────
    def annotate(self, frame, detections):
        """Draw detection boxes and labels on a copy of the frame.

        Detections that are beyond the arm's reach are drawn in red with an
        explicit "OUT OF REACH" warning so the operator can move closer.
        Detections that are harvestable are drawn in green with a rank number.
        """
        annotated = frame.copy()
        # Rank is assigned by main.py (see key 'rank'); fall back to no rank.
        for det in detections:
            x1, y1, x2, y2 = det["bbox_px"]
            cx, cy = det["center_px"]
            conf = det["confidence"]
            xyz = det["xyz_cm"]
            dist = det.get("distance_cm",
                           float(np.linalg.norm(
                               [xyz["x"], xyz["y"], xyz["z"]])))
            reachable = det.get("reachable", dist <= ARM_REACH_CM)
            rank = det.get("rank")

            colour = (0, 200, 0) if reachable else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(annotated, (cx, cy), 4, colour, -1)

            rank_str = f"#{rank} " if rank else ""
            top = f"{rank_str}Ripe {conf:.2f} | d:{dist:.1f}cm"
            cv2.putText(annotated, top, (x1, max(y1 - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

            bot = f"X:{xyz['x']:+.1f} Y:{xyz['y']:+.1f} Z:{xyz['z']:+.1f}"
            cv2.putText(annotated, bot, (x1, min(y2 + 18, frame.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 210, 0), 2)

            if not reachable:
                warn = f"OUT OF REACH (>{ARM_REACH_CM:.0f}cm) — move closer"
                cv2.putText(annotated, warn,
                            (x1, min(y2 + 38, frame.shape[0] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
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

    # ── Calibration helper ─────────────────────────────────────────────────────
    def calibrate_focal_length(self, pixel_diameter, known_distance_cm,
                               known_diameter_cm=None):
        """One-shot focal-length calibration.

        Place a tomato of known diameter at a known distance, measure its
        pixel diameter in the image, then call this to derive f (pixels).

            f = (pixel_diameter * Z) / real_diameter

        Example:
            det = TomatoDetector()
            # Measured: 7cm tomato at 20cm shows ~245 px across
            det.focal_length_px = det.calibrate_focal_length(245, 20.0)
        """
        if known_diameter_cm is None:
            known_diameter_cm = self.real_diameter_cm
        f = (pixel_diameter * known_distance_cm) / known_diameter_cm
        self.focal_length_px = f
        return f
