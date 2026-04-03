"""
Ripe-Tomato Detection — Agricultural Harvesting Robot
======================================================
Detects only ripe tomatoes and outputs structured 3-D coordinates
(x, y, z in cm) for the robotic-arm controller.

False-positive rejection pipeline (applied after YOLO inference):
  1. High confidence threshold (default 0.70)
  2. HSV colour gate  — region must contain enough red/orange pixels
  3. Aspect-ratio gate — bounding box must be roughly square (tomato-like)
  4. Size gate        — bounding box area must be within plausible range

Coordinate frame:
  - Origin: camera optical axis projected onto the image plane (image centre)
  - X_cm : positive → right
  - Y_cm : positive → down
  - Z_cm : depth estimated via pinhole model (requires FOCAL_LENGTH calibration)
"""

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

# ── Model ──────────────────────────────────────────────────────────────────────
# After running train.py with hard negatives, switch to the new weights:
#   MODEL_PATH = 'runs/detect/train_v2/weights/best.pt'
# Current: last trained model (before hard-negative retraining)
MODEL_PATH = 'runs/detect/train_v2/weights/best.pt' \
    if os.path.exists('runs/detect/train_v2/weights/best.pt') \
    else 'runs/detect/train6/weights/best.pt'
model = YOLO(MODEL_PATH)

# ── Camera / optics ────────────────────────────────────────────────────────────
FOCAL_LENGTH_PX  = 700    # pixels — run calibration with a known object to tune
REAL_DIAMETER_CM = 7.0    # average ripe-tomato diameter (cm)

# ── Detection twin-filters ─────────────────────────────────────────────────────
# Raise this if you still see false positives; lower it only if real tomatoes
# start being missed.
CONFIDENCE_THRESHOLD = 0.70

# Bounding-box area limits in pixels² (rejects dust specks and full-frame blobs)
MIN_BBOX_AREA_PX = 1_500
MAX_BBOX_AREA_PX = 180_000

# Aspect ratio: w/h — tomatoes are round; allow slight deviation for perspective
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20

# HSV colour gate — ripe tomatoes are red/orange
# OpenCV HSV: H [0-179], S [0-255], V [0-255]
# Red wraps in HSV, so we use two sub-ranges:
_HSV_RED_LO1 = np.array([0,   80,  50], dtype=np.uint8)
_HSV_RED_HI1 = np.array([15, 255, 255], dtype=np.uint8)
_HSV_RED_LO2 = np.array([160, 80,  50], dtype=np.uint8)
_HSV_RED_HI2 = np.array([179, 255, 255], dtype=np.uint8)

# Fraction of pixels inside the bbox that must be tomato-coloured
MIN_RED_PIXEL_RATIO = 0.18


# ── Helper functions ───────────────────────────────────────────────────────────

def estimate_depth_cm(pixel_diameter: float) -> float | None:
    """Pinhole-model depth estimate: Z = (real_size * f) / pixel_size."""
    if pixel_diameter <= 0:
        return None
    return (REAL_DIAMETER_CM * FOCAL_LENGTH_PX) / pixel_diameter


def passes_size_and_shape(x1: int, y1: int, x2: int, y2: int) -> bool:
    """True if the bounding box is within plausible tomato size and shape."""
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    area   = w * h
    aspect = w / h
    return (MIN_BBOX_AREA_PX <= area <= MAX_BBOX_AREA_PX and
            MIN_ASPECT_RATIO   <= aspect <= MAX_ASPECT_RATIO)


def passes_colour_gate(roi_bgr: np.ndarray) -> bool:
    """True if the region contains enough ripe-tomato (red/orange) colour."""
    if roi_bgr.size == 0:
        return False
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, _HSV_RED_LO1, _HSV_RED_HI1)
    mask |= cv2.inRange(hsv, _HSV_RED_LO2, _HSV_RED_HI2)
    ratio = np.count_nonzero(mask) / mask.size
    return ratio >= MIN_RED_PIXEL_RATIO


def pixel_to_camera_xyz(
    cx_px: int, cy_px: int, z_cm: float,
    frame_w: int, frame_h: int
) -> tuple[float, float, float]:
    """
    De-project a pixel centre (cx, cy) + depth Z into camera-frame 3D.
    Returns (X_cm, Y_cm, Z_cm) — ready to send to the robotic arm.
    """
    x_cm = (cx_px - frame_w / 2.0) * z_cm / FOCAL_LENGTH_PX
    y_cm = (cy_px - frame_h / 2.0) * z_cm / FOCAL_LENGTH_PX
    return round(x_cm, 2), round(y_cm, 2), round(z_cm, 2)


# ── Main detection loop ────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check connection or camera_index.")

frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"[INFO] Camera opened: {frame_w}x{frame_h}")
print(f"[INFO] Model: {MODEL_PATH}  |  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("[INFO] Press 'q' to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break

        # Run YOLO — confidence threshold applied inside the model
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = frame.copy()
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── Filter 1: size and aspect ratio ───────────────────────────────
            if not passes_size_and_shape(x1, y1, x2, y2):
                continue

            # ── Filter 2: colour must be red/orange ───────────────────────────
            roi = frame[max(0, y1):min(frame_h, y2),
                        max(0, x1):min(frame_w, x2)]
            if not passes_colour_gate(roi):
                continue

            # ── All filters passed — compute 3-D position ─────────────────────
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            z_cm   = estimate_depth_cm(x2 - x1)
            if z_cm is None:
                continue

            x_cm, y_cm, z_cm = pixel_to_camera_xyz(cx, cy, z_cm, frame_w, frame_h)

            det = {
                "confidence":  round(conf, 3),
                "bbox_px":     [x1, y1, x2, y2],
                "center_px":   [cx, cy],
                "xyz_cm":      {"x": x_cm, "y": y_cm, "z": z_cm},
            }
            detections.append(det)

            # ── Draw annotation ───────────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 200, 0), -1)

            top_label = f"Ripe {conf:.2f} | Z:{z_cm:.1f}cm"
            cv2.putText(annotated, top_label, (x1, max(y1 - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

            bot_label = f"X:{x_cm:+.1f}  Y:{y_cm:+.1f}"
            cv2.putText(annotated, bot_label, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 210, 0), 2)

        # ── Emit structured payload for the robotic-arm controller ────────────
        if detections:
            payload = {"ripe_tomatoes": detections, "count": len(detections)}
            print(json.dumps(payload))

        cv2.imshow("Ripe Tomato Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
