"""
Vision Module — AI-Enhanced Ripe-Tomato Detection for RPi5
============================================================
Provides:
  - ThreadedCamera: low-latency camera capture via dedicated thread
  - TomatoDetector: YOLO detection with AI-enhanced verification
  - TomatoTracker: temporal multi-frame tracking with occlusion handling
  - snapshot_scan(): captures for N seconds, clusters detections, returns
    up to 3 unique tomatoes sorted by distance (nearest first)

AI Enhancements:
  - Temporal tracking: detections persist across frames, survive brief occlusion
  - Ripeness classification: 6-stage HSV-based ripeness scoring
  - Enhanced false-positive rejection: texture, calyx, edge density, specularity
  - Occlusion handling: partial-view detection, track prediction during occlusion
  - Multi-frame consensus: only confirmed detections are reported

Optimised for Raspberry Pi 5 (8 GB):
  - imgsz=320 for faster YOLO inference on CPU
  - Threaded camera eliminates V4L2 buffer lag
  - Frame resolution set to 640×480
"""

import cv2
import math
import os
import time
import threading
import logging
import numpy as np
from collections import deque
from ultralytics import YOLO

logger = logging.getLogger(__name__)

ARM_REACH_CM = 36.0

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


# ── Threaded Camera ────────────────────────────────────────────────────────────

class ThreadedCamera:
    """Low-latency camera capture using a dedicated thread.

    Supports two backends:
      1. picamera2 (preferred on RPi 5 — works with both CSI and USB cameras)
      2. OpenCV V4L2 fallback (for non-RPi systems)

    Continuously grabs frames so the caller always gets the most recent one,
    eliminating pipeline buffering lag.
    """

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.cap = None
        self._use_picamera2 = False
        self._picam = None

    def start(self):
        """Open camera and start the background capture thread."""
        # Try picamera2 first (best on RPi 5 for both CSI and USB cameras)
        try:
            from picamera2 import Picamera2
            self._picam = Picamera2(self.camera_index)
            # Try RGB888 first; USB cameras may only support YUYV
            for fmt in ("RGB888", "YUYV"):
                try:
                    config = self._picam.create_video_configuration(
                        main={"size": (self.width, self.height), "format": fmt},
                        buffer_count=2,
                    )
                    self._picam.configure(config)
                    break
                except Exception:
                    continue
            self._picam.start()
            # Verify we can actually capture a frame
            import time
            time.sleep(0.3)
            test_frame = self._picam.capture_array()
            if test_frame is None:
                raise RuntimeError("picamera2 started but returned no frame")
            # Detect native format from frame shape
            if test_frame.ndim == 2 or (test_frame.ndim == 3 and test_frame.shape[2] == 2):
                self._picam_color_code = cv2.COLOR_YUV2BGR_YUYV
            else:
                self._picam_color_code = cv2.COLOR_RGB2BGR
            self._use_picamera2 = True
            logger.info(f"[CAM] picamera2 started: index={self.camera_index}, "
                        f"{self.width}x{self.height}, "
                        f"frame_shape={test_frame.shape}")
        except Exception as e:
            logger.warning(f"[CAM] picamera2 failed ({e}), trying OpenCV")
            # Clean up picamera2 properly before falling back
            if self._picam is not None:
                try:
                    self._picam.stop()
                except Exception:
                    pass
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None
            self._use_picamera2 = False
            # Fallback to OpenCV
            # On RPi5 the pispbe ISP creates many /dev/videoN nodes that
            # confuse OpenCV's index-based enumeration.  Open by device
            # path first, then fall back to integer index.
            dev_path = f"/dev/video{self.camera_index}"
            self.cap = cv2.VideoCapture(dev_path)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {self.camera_index}. "
                    f"Check that no other program is using the camera "
                    f"(run: fuser /dev/video{self.camera_index})")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # Use MJPEG for better quality at higher FPS
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            for _ in range(5):
                self.cap.read()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        if not self._use_picamera2:
            logger.info(f"[CAM] OpenCV camera started: index={self.camera_index}, "
                        f"{self.width}x{self.height}")
        return self

    def _capture_loop(self):
        """Background thread: continuously grab the latest frame."""
        while self._running:
            if self._use_picamera2:
                frame = self._picam.capture_array()
                # Convert to BGR for OpenCV (handles both RGB888 and YUYV)
                frame = cv2.cvtColor(frame, self._picam_color_code)
                with self._lock:
                    self._frame = frame
            else:
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame

    def read(self):
        """Get the latest frame.  Returns (success, frame)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def is_opened(self):
        if self._use_picamera2:
            return self._picam is not None and self._running
        return self.cap is not None and self.cap.isOpened() and self._running

    def stop(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._use_picamera2 and self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("[CAM] Camera stopped")

    def release(self):
        self.stop()


# ── Ripeness Classifier ────────────────────────────────────────────────────────

class RipenessClassifier:
    """HSV-based 6-stage ripeness classifier for tomatoes.
    
    USDA stages:
      1. Green       – entirely green
      2. Breaker     – first hint of pink/red (<10% red)
      3. Turning     – 10-30% red
      4. Pink        – 30-60% red
      5. Light Red   – 60-90% red
      6. Red (Ripe)  – >90% red surface
    
    Only stages 5-6 are considered 'ripe' for harvest.
    """

    # HSV ranges for red/orange (ripe) pixels
    _RED_LO1 = np.array([0,   70,  50], dtype=np.uint8)
    _RED_HI1 = np.array([15, 255, 255], dtype=np.uint8)
    _RED_LO2 = np.array([160, 70,  50], dtype=np.uint8)
    _RED_HI2 = np.array([179, 255, 255], dtype=np.uint8)
    # Orange (transitional)
    _ORG_LO  = np.array([10,  80,  80], dtype=np.uint8)
    _ORG_HI  = np.array([25, 255, 255], dtype=np.uint8)
    # Green (unripe)
    _GRN_LO  = np.array([30,  40,  30], dtype=np.uint8)
    _GRN_HI  = np.array([85, 255, 255], dtype=np.uint8)

    STAGES = ["Green", "Breaker", "Turning", "Pink", "Light Red", "Ripe"]

    @classmethod
    def classify(cls, roi_bgr):
        """Return (stage_index 0-5, stage_name, red_ratio)."""
        if roi_bgr.size == 0:
            return 0, "Green", 0.0
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        if total == 0:
            return 0, "Green", 0.0

        red_mask  = cv2.inRange(hsv, cls._RED_LO1, cls._RED_HI1)
        red_mask |= cv2.inRange(hsv, cls._RED_LO2, cls._RED_HI2)
        orange_mask = cv2.inRange(hsv, cls._ORG_LO, cls._ORG_HI)
        green_mask = cv2.inRange(hsv, cls._GRN_LO, cls._GRN_HI)

        red_r = np.count_nonzero(red_mask) / total
        org_r = np.count_nonzero(orange_mask) / total
        grn_r = np.count_nonzero(green_mask) / total

        # Combined warm color ratio (red + orange)
        warm_r = red_r + org_r * 0.5

        if warm_r >= 0.90:
            return 5, "Ripe", red_r
        elif warm_r >= 0.60:
            return 4, "Light Red", red_r
        elif warm_r >= 0.30:
            return 3, "Pink", red_r
        elif warm_r >= 0.10:
            return 2, "Turning", red_r
        elif warm_r > 0.0 or grn_r < 0.50:
            return 1, "Breaker", red_r
        else:
            return 0, "Green", red_r


# ── Temporal Tracker ───────────────────────────────────────────────────────────

class TomatoTracker:
    """Multi-frame temporal tracker for tomato detections.
    
    Features:
      - Assigns persistent IDs to tracked tomatoes
      - Maintains confidence history (exponential moving average)
      - Handles brief occlusion: tracks survive for `max_missing_frames`
        without a detection match
      - Predicts position during occlusion using velocity estimate
      - Only promotes tracks to 'confirmed' after `min_hits` consecutive frames
      - Suppresses flickering detections (appear/disappear rapidly)
    """

    def __init__(self, max_missing_frames=8, min_hits=3,
                 match_distance_px=80, ema_alpha=0.3):
        self._next_id = 0
        self.tracks = {}          # track_id → track dict
        self.max_missing = max_missing_frames
        self.min_hits = min_hits
        self.match_dist = match_distance_px
        self.ema_alpha = ema_alpha

    def _new_track(self, det):
        tid = self._next_id
        self._next_id += 1
        cx, cy = det["center_px"]
        return {
            "id": tid,
            "center_px": [cx, cy],
            "velocity": [0.0, 0.0],
            "det": det,
            "hits": 1,
            "missing": 0,
            "ema_conf": det["confidence"],
            "confirmed": False,
            "occluded": False,
            "history": deque(maxlen=30),
        }

    def update(self, detections):
        """Match new detections to existing tracks, return confirmed tracks.
        
        Args:
            detections: list of detection dicts from TomatoDetector.detect()
        
        Returns:
            list of detection dicts for confirmed, active tracks
        """
        # Predict positions for existing tracks
        for tid, trk in self.tracks.items():
            trk["center_px"][0] += trk["velocity"][0]
            trk["center_px"][1] += trk["velocity"][1]

        # Hungarian-style greedy matching (distance-based)
        used_dets = set()
        used_trks = set()

        # Build cost matrix
        pairs = []
        for ti, (tid, trk) in enumerate(self.tracks.items()):
            for di, det in enumerate(detections):
                dx = det["center_px"][0] - trk["center_px"][0]
                dy = det["center_px"][1] - trk["center_px"][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.match_dist:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda x: x[0])

        # Greedy match
        for dist, tid, di in pairs:
            if tid in used_trks or di in used_dets:
                continue
            trk = self.tracks[tid]
            det = detections[di]
            cx_old = trk["center_px"]
            cx_new = det["center_px"]
            # Update velocity (smoothed)
            trk["velocity"][0] = 0.7 * trk["velocity"][0] + 0.3 * (cx_new[0] - cx_old[0])
            trk["velocity"][1] = 0.7 * trk["velocity"][1] + 0.3 * (cx_new[1] - cx_old[1])
            trk["center_px"] = list(cx_new)
            trk["det"] = det
            trk["hits"] += 1
            trk["missing"] = 0
            trk["occluded"] = False
            trk["ema_conf"] = self.ema_alpha * det["confidence"] + (1 - self.ema_alpha) * trk["ema_conf"]
            trk["history"].append(det)
            if trk["hits"] >= self.min_hits:
                trk["confirmed"] = True
            used_trks.add(tid)
            used_dets.add(di)

        # Handle unmatched tracks → occlusion or gone
        for tid, trk in list(self.tracks.items()):
            if tid not in used_trks:
                trk["missing"] += 1
                trk["occluded"] = True
                if trk["missing"] > self.max_missing:
                    del self.tracks[tid]

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in used_dets:
                trk = self._new_track(det)
                self.tracks[trk["id"]] = trk

        # Return confirmed, non-stale tracks
        result = []
        for tid, trk in self.tracks.items():
            if trk["confirmed"] and trk["missing"] <= self.max_missing:
                det = dict(trk["det"])
                det["track_id"] = tid
                det["track_hits"] = trk["hits"]
                det["track_conf"] = round(trk["ema_conf"], 3)
                det["occluded"] = trk["occluded"]
                if trk["occluded"]:
                    # Use predicted position
                    det["center_px"] = list(trk["center_px"])
                result.append(det)
        result.sort(key=lambda d: d.get("distance_cm", 999))
        return result

    def reset(self):
        self.tracks.clear()
        self._next_id = 0


# ── Tomato Detector ────────────────────────────────────────────────────────────

class TomatoDetector:
    """Encapsulates YOLO model + AI-enhanced verification pipeline.

    Pipeline:
      1. YOLO detection (trained on ripe-tomato dataset)
      2. Size/shape filter
      3. Color filter (red ratio)
      4. Authenticity check (texture, calyx, uniformity, contour, diversity)
      5. Edge density analysis (rejects smooth artificial objects)
      6. Specular highlight check (rejects shiny balls/plastic)
      7. Ripeness classification (6-stage USDA scale)
      8. Temporal tracking with occlusion handling
      9. Multi-frame consensus (min_hits before reporting)

    Optimised for RPi 5:
      - imgsz=320 for faster CPU inference
      - Returns detections sorted by distance (nearest first)
      - snapshot_scan() for batch multi-tomato detection
    """

    def __init__(self, model_path=None, confidence=0.35,
                 focal_length_px=700, real_diameter_cm=7.0,
                 imgsz=320):
        self.model_path = model_path or _resolve_model_path()
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.focal_length_px = focal_length_px
        self.focal_length_x_px = float(os.getenv("TOMATO_FX_PX", focal_length_px))
        self.focal_length_y_px = float(os.getenv("TOMATO_FY_PX", focal_length_px))
        self.cx_offset_px = float(os.getenv("TOMATO_CX_OFFSET_PX", "0.0"))
        self.cy_offset_px = float(os.getenv("TOMATO_CY_OFFSET_PX", "0.0"))
        self.depth_scale = float(os.getenv("TOMATO_DEPTH_SCALE", "1.00"))
        self.depth_bias_cm = float(os.getenv("TOMATO_DEPTH_BIAS_CM", "0.0"))
        self.real_diameter_cm = real_diameter_cm
        self.imgsz = imgsz

        # Filter parameters (relaxed for real-world RPi camera conditions)
        self.min_bbox_area = 800
        self.max_bbox_area = 200_000
        self.min_aspect = 0.35
        self.max_aspect = 2.80
        self.min_red_ratio = 0.08

        # Red-blob gates (explicit ball rejection)
        self.min_blob_circularity = 0.35
        self.max_blob_circularity = 0.97
        self.min_blob_hue_std = 4.0
        self.min_blob_sat_std = 12.0

        # HSV red ranges
        self._lo1 = np.array([0,   80,  50], dtype=np.uint8)
        self._hi1 = np.array([15, 255, 255], dtype=np.uint8)
        self._lo2 = np.array([160, 80,  50], dtype=np.uint8)
        self._hi2 = np.array([179, 255, 255], dtype=np.uint8)

        # HSV green range (for calyx / stem detection)
        self._green_lo = np.array([30, 30, 25], dtype=np.uint8)
        self._green_hi = np.array([95, 255, 255], dtype=np.uint8)

        # HSV yellow-green range (some calyxes are yellowish-green)
        self._ygreen_lo = np.array([20, 30, 25], dtype=np.uint8)
        self._ygreen_hi = np.array([35, 255, 255], dtype=np.uint8)

        # Tomato-authenticity thresholds (tightened to reject red balls)
        self.min_texture_var = 18.0       # Laplacian variance
        self.min_calyx_ratio = 0.002      # green pixels in upper 40% of ROI
        self.max_color_uniformity = 0.90  # too-uniform red → likely a ball
        self.min_authenticity_score = 4   # out of 7 core checks

        # Edge density thresholds
        self.min_edge_density = 0.010     # real tomatoes have skin detail
        self.max_specularity = 0.20       # balls have bright specular spots

        # Ripeness: harvest stage 3+ (Pink, Light Red, Ripe)
        self.min_ripeness_stage = 2

        # Performance knobs
        self.snapshot_process_every_n = max(
            1, int(os.getenv("TOMATO_SCAN_PROCESS_EVERY_N", "2")))

        # Temporal tracker
        self.tracker = TomatoTracker(
            max_missing_frames=8,
            min_hits=2,
            match_distance_px=80,
        )

        # Ripeness classifier
        self.ripeness = RipenessClassifier()

    # ── Depth estimation ───────────────────────────────────────────────────────
    def estimate_depth_cm(self, pixel_diameter):
        if pixel_diameter <= 0:
            return None
        raw = (self.real_diameter_cm * self.focal_length_px) / pixel_diameter
        return max(0.0, raw * self.depth_scale + self.depth_bias_cm)

    def calibrate_focal_length(self, pixel_diameter, known_distance_cm):
        """One-shot focal-length calibration from a known tomato distance."""
        if pixel_diameter <= 0 or known_distance_cm <= 0:
            raise ValueError("pixel_diameter and known_distance_cm must be > 0")
        self.focal_length_px = float((pixel_diameter * known_distance_cm) / self.real_diameter_cm)
        self.focal_length_x_px = self.focal_length_px
        self.focal_length_y_px = self.focal_length_px
        return self.focal_length_px

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

    def _red_blob_metrics(self, roi_bgr):
        """Return core blob metrics for geometry, texture, and depth sizing."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, self._lo1, self._hi1)
        red_mask |= cv2.inRange(hsv, self._lo2, self._hi2)

        # Clean the mask so tiny speckles do not affect contour metrics.
        kernel = np.ones((3, 3), dtype=np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        if area <= 1.0 or perim <= 1.0:
            return None

        circularity = float(min(1.0, 4.0 * math.pi * area / (perim * perim)))
        eq_diameter_px = float(np.sqrt(4.0 * area / math.pi))

        blob_pixels = hsv[red_mask > 0]
        if blob_pixels.size == 0:
            return None

        hue_std = float(np.std(blob_pixels[:, 0].astype(np.float32)))
        sat_std = float(np.std(blob_pixels[:, 1].astype(np.float32)))

        return {
            "mask": red_mask,
            "area": area,
            "circularity": circularity,
            "eq_diameter_px": eq_diameter_px,
            "hue_std": hue_std,
            "sat_std": sat_std,
        }

    def _passes_blob_gates(self, metrics):
        if metrics is None:
            return False
        if not (self.min_blob_circularity <= metrics["circularity"] <= self.max_blob_circularity):
            return False
        if metrics["hue_std"] < self.min_blob_hue_std and metrics["sat_std"] < self.min_blob_sat_std:
            return False
        return True

    # ── Tomato-authenticity checks (reject balls, cups, etc.) ──────────────

    def _texture_score(self, roi_bgr):
        """Laplacian variance — real tomatoes have natural surface texture
        (skin pores, subtle colour gradients, calyx edges) whereas a ball
        or plastic object is unnaturally smooth."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _has_calyx(self, roi_bgr):
        """Check for green/yellow-green pixels in the upper 40 % of the ROI.
        Real tomatoes almost always have a green calyx / stem cap."""
        h = roi_bgr.shape[0]
        upper = roi_bgr[:max(1, int(h * 0.40)), :]
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self._green_lo, self._green_hi)
        green_mask |= cv2.inRange(hsv, self._ygreen_lo, self._ygreen_hi)
        if green_mask.size == 0:
            return 0.0
        return float(np.count_nonzero(green_mask)) / green_mask.size

    def _color_uniformity(self, roi_bgr):
        """Measure how uniform the red channel is.  Balls tend to have very
        uniform colour; real tomatoes show natural gradients and variation.
        Returns 0-1 where 1 = perfectly uniform."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # Only look at the Hue & Saturation of red pixels
        red_mask  = cv2.inRange(hsv, self._lo1, self._hi1)
        red_mask |= cv2.inRange(hsv, self._lo2, self._hi2)
        red_pixels = hsv[red_mask > 0]
        if len(red_pixels) < 20:
            return 1.0  # not enough data → treat as suspicious
        # Coefficient of variation of Saturation channel
        sat = red_pixels[:, 1].astype(float)
        std = np.std(sat)
        mean = np.mean(sat)
        if mean < 1:
            return 1.0
        cv_val = std / mean
        # Normalize to 0-1 (lower cv → more uniform)
        uniformity = max(0.0, 1.0 - cv_val)
        return uniformity

    def _contour_regularity(self, roi_bgr):
        """Measure how perfectly circular the contour is.
        A man-made ball is near-perfect; a tomato has slight bumps and
        the calyx breaks the circular outline.
        Returns circularity 0-1 where 1 = perfect circle."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.5
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            return 0.5
        circularity = 4 * math.pi * area / (perim * perim)
        return min(circularity, 1.0)

    def _color_diversity(self, roi_bgr):
        """Measure the range of hues present in the ROI.
        Real tomatoes show red + orange + hints of yellow/green (especially
        near calyx).  A ball is almost exclusively one hue band.
        Returns the number of distinct hue 'zones' with significant presence."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 1]  # saturation to filter out low-sat noise
        s_mask = h_chan > 40
        if np.count_nonzero(s_mask) < 20:
            return 0
        hues = hsv[:, :, 0][s_mask]
        # Count presence in hue zones: red-low(0-10), red-high(170-179),
        # orange(11-25), yellow(26-35), green(36-85)
        zones = 0
        total = len(hues)
        for lo, hi in [(0, 10), (170, 179), (11, 25), (26, 35), (36, 85)]:
            count = np.count_nonzero((hues >= lo) & (hues <= hi))
            if count / total > 0.03:  # at least 3% presence
                zones += 1
        return zones

    def _edge_density(self, roi_bgr):
        """Measure edge density using Canny edge detection.
        Real tomatoes have fine surface detail (skin texture, calyx edges,
        subtle color transitions). Smooth plastic balls have very few edges.
        Returns ratio of edge pixels to total pixels."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        return float(np.count_nonzero(edges)) / max(1, edges.size)

    def _specularity_ratio(self, roi_bgr):
        """Detect specular highlights — shiny balls reflect light as bright
        concentrated spots. Real tomatoes have a matte/waxy surface with
        diffuse, softer reflections.
        Returns ratio of very bright pixels (near-white in HSV)."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # Specular highlights: low saturation + very high value
        spec_mask = cv2.inRange(hsv,
                                np.array([0, 0, 230], dtype=np.uint8),
                                np.array([179, 50, 255], dtype=np.uint8))
        return float(np.count_nonzero(spec_mask)) / max(1, spec_mask.size)

    def _stem_scar_detected(self, roi_bgr):
        """Check for the characteristic stem scar (blossom end) at the bottom
        of the tomato — a small darker circular mark that balls lack.
        Returns True if a plausible stem scar is found."""
        h = roi_bgr.shape[0]
        # Look at bottom 30% of ROI
        bottom = roi_bgr[max(0, int(h * 0.70)):, :]
        if bottom.size == 0:
            return False
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        # Stem scar is a darker region in a lighter red area
        mean_val = float(np.mean(gray))
        # Look for a darker patch
        dark_mask = gray < (mean_val * 0.65)
        dark_ratio = np.count_nonzero(dark_mask) / max(1, dark_mask.size)
        return 0.005 < dark_ratio < 0.25

    def _is_authentic_tomato(self, roi_bgr):
        """Score-based tomato verification.  Returns (is_tomato, score, details).
        A detection must pass at least `min_authenticity_score` of 7 core checks.
        
        Also applies hard-reject rules for obvious ball signatures:
          - No calyx + high circularity → instant reject
          - No calyx + high uniformity + low edge density → instant reject
        """
        score = 0
        details = {}

        # 1. Texture: tomatoes have richer surface texture than smooth balls
        tex = self._texture_score(roi_bgr)
        details["texture"] = round(tex, 1)
        if tex >= self.min_texture_var:
            score += 1

        # 2. Calyx: green stem area on top — strong indicator of real tomato
        calyx = self._has_calyx(roi_bgr)
        details["calyx"] = round(calyx, 4)
        has_calyx = calyx >= self.min_calyx_ratio
        if has_calyx:
            score += 1

        # 3. Color uniformity: too-uniform → likely artificial
        unif = self._color_uniformity(roi_bgr)
        details["uniformity"] = round(unif, 3)
        is_uniform = unif >= self.max_color_uniformity
        if not is_uniform:
            score += 1

        # 4. Contour: too-perfect circle → likely a ball
        circ = self._contour_regularity(roi_bgr)
        details["circularity"] = round(circ, 3)
        is_circular = circ >= 0.85
        if not is_circular:
            score += 1

        # 5. Color diversity: tomatoes have multiple hue zones, balls have one
        diversity = self._color_diversity(roi_bgr)
        details["hue_zones"] = diversity
        if diversity >= 2:  # at least 2 distinct hue zones
            score += 1

        # 6. Edge density: real tomato skin has fine detail; balls are smooth
        edge_dens = self._edge_density(roi_bgr)
        details["edge_density"] = round(edge_dens, 4)
        low_edges = edge_dens < self.min_edge_density
        if not low_edges:
            score += 1

        # 7. Specularity: shiny balls have bright specular highlights
        spec = self._specularity_ratio(roi_bgr)
        details["specularity"] = round(spec, 4)
        if spec < self.max_specularity:
            score += 1

        # Stem-scar heuristic is diagnostic only; not used for scoring because
        # smooth objects can occasionally mimic this pattern under lighting.
        stem_scar = self._stem_scar_detected(roi_bgr)
        details["stem_scar"] = bool(stem_scar)

        details["score"] = f"{score}/7"

        # ── Hard-reject rules for obvious ball signatures ──
        # Rule A: No calyx + very circular = ball (tomatoes always have a stem)
        if not has_calyx and circ >= 0.90:
            details["hard_reject"] = "no_calyx+circular"
            logger.debug(f"[FILTER] HARD REJECT (no calyx + circular={circ:.3f}): {details}")
            return False, score, details

        # Rule B: No calyx + uniform color + very low edges = artificial object
        if not has_calyx and is_uniform and edge_dens < 0.008:
            details["hard_reject"] = "no_calyx+uniform+smooth"
            logger.debug(f"[FILTER] HARD REJECT (no calyx + uniform + smooth): {details}")
            return False, score, details

        # Rule C: Very high circularity + very uniform = ball regardless
        if circ >= 0.93 and is_uniform:
            details["hard_reject"] = "perfect_circle+uniform"
            logger.debug(f"[FILTER] HARD REJECT (perfect circle + uniform): {details}")
            return False, score, details

        # Rule D: No calyx and low hue diversity is a common red-ball signature.
        if not has_calyx and diversity < 2:
            details["hard_reject"] = "no_calyx+single_hue_band"
            logger.debug(f"[FILTER] HARD REJECT (no calyx + single hue band): {details}")
            return False, score, details

        # Rule E: If calyx is absent, demand a stronger overall score.
        if not has_calyx and score < 5:
            details["hard_reject"] = "no_calyx+weak_score"
            logger.debug(f"[FILTER] HARD REJECT (no calyx + weak score): {details}")
            return False, score, details

        # Rule F: Without calyx, require stronger non-color evidence to avoid
        # confusing smooth red balls with tomatoes.
        if not has_calyx and (circ > 0.88 and spec > 0.18):
            details["hard_reject"] = "no_calyx+round_or_shiny"
            logger.debug(f"[FILTER] HARD REJECT (no calyx + round/shiny): {details}")
            return False, score, details

        is_tomato = score >= self.min_authenticity_score
        if not is_tomato:
            logger.debug(f"[FILTER] Rejected (score={score}/7): {details}")
        else:
            logger.debug(f"[FILTER] Accepted (score={score}/7): {details}")
        return is_tomato, score, details

    def _pixel_to_xyz(self, cx, cy, z_cm, fw, fh):
        cx0 = fw / 2.0 + self.cx_offset_px
        cy0 = fh / 2.0 + self.cy_offset_px
        x_cm = (cx - cx0) * z_cm / self.focal_length_x_px
        y_cm = (cy - cy0) * z_cm / self.focal_length_y_px
        return round(x_cm, 2), round(y_cm, 2), round(z_cm, 2)

    # ── Fast detection (lightweight — for scan loops) ──────────────────────────
    def detect_fast(self, frame):
        """Lightweight detection: YOLO + size/colour filter only.

        Skips the expensive authenticity checks (texture, calyx, contour,
        edge density, specularity) so each frame is processed ~3-4× faster.
        Used inside snapshot_scan(); the heavy checks run once on final
        candidates if needed.
        """
        fh, fw = frame.shape[:2]
        results = self.model(frame, conf=self.confidence, verbose=False,
                             imgsz=self.imgsz)
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if not self._passes_size_shape(x1, y1, x2, y2):
                continue

            roi = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
            if roi.size == 0:
                continue

            if not self._passes_colour(roi):
                continue

            metrics = self._red_blob_metrics(roi)
            if not self._passes_blob_gates(metrics):
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            # Equivalent-area diameter is more stable than raw bbox width.
            effective_diam_px = min(metrics["eq_diameter_px"], float(min(w, h)))
            z_cm = self.estimate_depth_cm(effective_diam_px)
            if z_cm is None:
                continue

            x_cm, y_cm, z_cm = self._pixel_to_xyz(cx, cy, z_cm, fw, fh)
            distance_cm = float(np.linalg.norm([x_cm, y_cm, z_cm]))

            detections.append({
                "confidence": round(conf, 3),
                "bbox_px":    [x1, y1, x2, y2],
                "center_px":  [cx, cy],
                "xyz_cm":     {"x": x_cm, "y": y_cm, "z": z_cm},
                "distance_cm": round(distance_cm, 2),
                "reachable": bool(distance_cm <= ARM_REACH_CM),
            })

        detections.sort(key=lambda d: d["distance_cm"])
        return detections

    # ── Main detection on a single frame ───────────────────────────────────────
    def detect(self, frame, use_tracker=True):
        """
        Run AI-enhanced detection on a BGR frame.
        
        Pipeline:
          1. YOLO inference
          2. Size/shape filter
          3. Color filter (red ratio)
          4. Authenticity verification (7-point scoring)
          5. Ripeness classification (6-stage)
          6. Temporal tracking with occlusion handling (if use_tracker=True)
        
        Returns list of dicts sorted by distance (nearest first), each with:
          confidence, bbox_px, center_px, xyz_cm, distance_cm,
          ripeness_stage, ripeness_name, authenticity, track_id (if tracked)
        """
        fh, fw = frame.shape[:2]
        results = self.model(frame, conf=self.confidence, verbose=False,
                             imgsz=self.imgsz)
        raw_detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if not self._passes_size_shape(x1, y1, x2, y2):
                continue

            roi = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
            if roi.size == 0:
                continue

            if not self._passes_colour(roi):
                continue

            metrics = self._red_blob_metrics(roi)
            if not self._passes_blob_gates(metrics):
                continue

            # Verify this is a real tomato, not a ball or other red object
            is_tomato, auth_score, auth_details = self._is_authentic_tomato(roi)
            if not is_tomato:
                continue

            # Classify ripeness stage
            stage_idx, stage_name, red_ratio = self.ripeness.classify(roi)

            # Skip unripe tomatoes (harvest Pink, Light Red, and Ripe)
            if stage_idx < self.min_ripeness_stage:
                logger.debug(f"[RIPENESS] Skipped: {stage_name} "
                            f"(stage {stage_idx}, red={red_ratio:.2f})")
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            effective_diam_px = min(metrics["eq_diameter_px"], float(min(w, h)))
            z_cm = self.estimate_depth_cm(effective_diam_px)
            if z_cm is None:
                continue

            x_cm, y_cm, z_cm = self._pixel_to_xyz(cx, cy, z_cm, fw, fh)
            distance_cm = float(np.linalg.norm([x_cm, y_cm, z_cm]))

            raw_detections.append({
                "confidence": round(conf, 3),
                "bbox_px":    [x1, y1, x2, y2],
                "center_px":  [cx, cy],
                "xyz_cm":     {"x": x_cm, "y": y_cm, "z": z_cm},
                "distance_cm": round(distance_cm, 2),
                "reachable": bool(distance_cm <= ARM_REACH_CM),
                "authenticity": auth_details,
                "ripeness_stage": stage_idx,
                "ripeness_name": stage_name,
            })

        # Apply temporal tracking for multi-frame consensus + occlusion handling
        if use_tracker:
            detections = self.tracker.update(raw_detections)
        else:
            detections = raw_detections

        detections.sort(key=lambda d: d.get("distance_cm", 999))
        return detections

    # ── Snapshot scan — batch detection over a time window ─────────────────────
    def snapshot_scan(self, camera, duration_s=5.0, max_tomatoes=3,
                      cluster_radius_cm=8.0, min_sightings=3):
        """Scan for tomatoes over a time window and return unique detections.

        Captures frames for *duration_s* seconds, clusters nearby detections
        (same tomato seen across frames), and returns up to *max_tomatoes*
        sorted by distance (nearest first).

        Args:
            camera: ThreadedCamera or cv2.VideoCapture with .read()
            duration_s: scanning window in seconds (≤5 s)
            max_tomatoes: maximum tomatoes to return (default 3)
            cluster_radius_cm: merge detections within this radius
            min_sightings: minimum frames a tomato must appear in

        Returns:
            List of dicts: {xyz_cm, confidence, distance_cm, sightings}
        """
        clusters = []
        start = time.time()
        frame_count = 0
        last_frame = None

        while time.time() - start < duration_s:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            last_frame = frame
            # Process every Nth frame to keep scan responsive on CPU.
            if frame_count % self.snapshot_process_every_n != 0:
                frame_count += 1
                continue
            # Use full detection here so scan confirmation includes
            # authenticity/ripeness checks (not color-only fast filtering).
            detections = self.detect(frame, use_tracker=False)
            frame_count += 1

            for det in detections:
                pos = det["xyz_cm"]
                merged = False

                for cluster in clusters:
                    avg = cluster["avg_pos"]
                    dist = math.sqrt(
                        (pos["x"] - avg["x"])**2 +
                        (pos["y"] - avg["y"])**2 +
                        (pos["z"] - avg["z"])**2)
                    if dist < cluster_radius_cm:
                        n = cluster["count"]
                        cluster["avg_pos"] = {
                            "x": (avg["x"] * n + pos["x"]) / (n + 1),
                            "y": (avg["y"] * n + pos["y"]) / (n + 1),
                            "z": (avg["z"] * n + pos["z"]) / (n + 1),
                        }
                        cluster["max_conf"] = max(cluster["max_conf"],
                                                   det["confidence"])
                        cluster["bbox_px"] = det.get("bbox_px", cluster.get("bbox_px"))
                        cluster["count"] += 1
                        merged = True
                        break

                if not merged:
                    clusters.append({
                        "avg_pos": dict(pos),
                        "max_conf": det["confidence"],
                        "bbox_px": det.get("bbox_px"),
                        "count": 1,
                    })

            # Small yield to avoid hogging CPU on RPi5
            time.sleep(0.005)

        # Filter by minimum sightings and compute distances
        valid = [c for c in clusters if c["count"] >= min_sightings]
        for c in valid:
            p = c["avg_pos"]
            c["distance_cm"] = round(
                math.sqrt(p["x"]**2 + p["y"]**2 + p["z"]**2), 2)
        valid.sort(key=lambda c: c["distance_cm"])

        logger.info(f"[SCAN] {frame_count} frames in {duration_s}s, "
                    f"{len(clusters)} clusters, {len(valid)} valid, "
                f"returning top {min(max_tomatoes, len(valid))}; "
                f"process_every_n={self.snapshot_process_every_n}")

        return [
            {"xyz_cm": c["avg_pos"],
             "confidence": c["max_conf"],
             "bbox_px": c.get("bbox_px"),
             "distance_cm": c["distance_cm"],
             "sightings": c["count"]}
            for c in valid[:max_tomatoes]
        ], last_frame

    # ── Annotate frame with detections ─────────────────────────────────────────
    def annotate(self, frame, detections):
        """Draw detection boxes with tracking, ripeness, and occlusion info."""
        annotated = frame.copy()
        colors = [(0, 255, 0), (0, 200, 255), (0, 100, 255)]  # green, yellow, orange
        occluded_color = (128, 128, 255)  # light red for occluded tracks

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox_px"]
            cx, cy = int(det["center_px"][0]), int(det["center_px"][1])
            conf = det.get("track_conf", det["confidence"])
            xyz = det["xyz_cm"]
            dist = det.get("distance_cm", 0)
            is_occluded = det.get("occluded", False)
            track_id = det.get("track_id", None)
            ripeness = det.get("ripeness_name", "")
            auth = det.get("authenticity", {})
            auth_score = auth.get("score", "")

            # Use different color if occluded
            if is_occluded:
                color = occluded_color
                line_style = 1  # thin dashed effect
            else:
                color = colors[min(i, len(colors) - 1)]
                line_style = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_style)
            cv2.circle(annotated, (cx, cy), 4, color, -1)

            # Top label: rank, confidence, distance, track ID
            rank = f"#{i+1} " if len(detections) > 1 else ""
            tid_str = f"T{track_id}" if track_id is not None else ""
            occ_str = " [OCC]" if is_occluded else ""
            top = f"{rank}{tid_str} {conf:.2f} | D:{dist:.1f}cm{occ_str}"
            cv2.putText(annotated, top, (x1, max(y1 - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

            # Middle label: ripeness + auth score
            if ripeness:
                mid = f"{ripeness} | Auth:{auth_score}"
                cv2.putText(annotated, mid, (x1, max(y1 - 28, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)

            # Bottom label: 3D coordinates
            bot = f"X:{xyz['x']:+.1f}  Y:{xyz['y']:+.1f}  Z:{xyz['z']:.1f}"
            cv2.putText(annotated, bot, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 210, 0), 1)

        # Show tracking stats in corner
        n_tracks = len(self.tracker.tracks)
        n_confirmed = sum(1 for t in self.tracker.tracks.values() if t["confirmed"])
        n_occluded = sum(1 for t in self.tracker.tracks.values() if t["occluded"])
        stats = f"Tracks:{n_tracks} Confirmed:{n_confirmed} Occluded:{n_occluded}"
        cv2.putText(annotated, stats, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

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
