"""
Ripe-Tomato Detection — Agricultural Harvesting Robot (standalone debug tool)
==============================================================================
Detects only ripe tomatoes and outputs structured 3-D coordinates
(x, y, z in cm) plus a Euclidean distance and a "reachable" flag for the
robotic-arm controller.

This script is a thin wrapper around `vision.TomatoDetector`, which contains
the full false-positive rejection pipeline (confidence, size, aspect, red
ratio, texture / hue variance, and red-blob circularity).  Run `main.py` for
the full harvest workflow — this file is useful for calibration and quick
visual testing.

Coordinate frame:
  - Origin: camera optical axis projected onto the image plane (image centre)
  - X_cm : positive → right
  - Y_cm : positive → down
  - Z_cm : depth estimated via pinhole model (requires focal-length calibration)
"""

import cv2
import json

from vision import TomatoDetector, ARM_REACH_CM

# Capture tuning for Raspberry Pi 5 — keep latency low so the feed is live.
CAMERA_INDEX    = 0
CAPTURE_WIDTH   = 640
CAPTURE_HEIGHT  = 480
CAPTURE_FOURCC  = "MJPG"
# Discard this many buffered frames per iteration so we always process the
# freshest frame (see main.py for rationale).
FRAME_DRAIN_COUNT = 2


def _open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check connection or camera_index.")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def _grab_fresh_frame(cap):
    for _ in range(FRAME_DRAIN_COUNT):
        cap.grab()
    return cap.read()


def main():
    detector = TomatoDetector()
    cap = _open_camera(CAMERA_INDEX)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {fw}x{fh}")
    print(f"[INFO] Model: {detector.model_path}  | "
          f"confidence={detector.confidence}  imgsz={detector.infer_imgsz}")
    print(f"[INFO] Arm reach (max harvest distance): {ARM_REACH_CM:.0f} cm")
    print("[INFO] Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = _grab_fresh_frame(cap)
            if not ret:
                print("[ERROR] Failed to grab frame from camera.")
                break

            detections = detector.detect(frame)
            # Sort nearest-first so rank in the log matches rank in the overlay.
            detections.sort(key=lambda d: d["distance_cm"])
            for i, d in enumerate(detections, start=1):
                d["rank"] = i

            annotated = detector.annotate(frame, detections)

            if detections:
                payload = {
                    "ripe_tomatoes": detections,
                    "count": len(detections),
                    "reach_cm": ARM_REACH_CM,
                }
                print(json.dumps(payload))

            # Warn the operator if any visible tomato is beyond the arm reach.
            oor = [d for d in detections if not d["reachable"]]
            if oor:
                cv2.putText(annotated,
                            f"{len(oor)} tomato(es) beyond {ARM_REACH_CM:.0f}cm"
                            f" — MOVE CLOSER",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            cv2.imshow("Ripe Tomato Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
