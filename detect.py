"""
Debug Detection Runner — shared TomatoDetector pipeline
======================================================
Uses the same `TomatoDetector` implementation as `main.py` so filtering,
distance mapping, and reach checks are consistent.
"""

import json
import cv2

from vision import TomatoDetector, ARM_REACH_CM

FRAME_DRAIN_COUNT = 2
CAMERA_INDEX = 0


def open_low_latency_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def drain_frames(cap, n=FRAME_DRAIN_COUNT):
    for _ in range(max(0, n)):
        cap.grab()


def main():
    detector = TomatoDetector(imgsz=416)
    cap = open_low_latency_camera(CAMERA_INDEX)

    print("[INFO] Running detect.py with shared TomatoDetector")
    print(f"[INFO] Reach threshold: {ARM_REACH_CM:.1f} cm")
    print("[INFO] Press 'q' to quit")

    try:
        while True:
            drain_frames(cap)
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame")
                continue

            detections = detector.detect(frame, use_tracker=False)
            payload = {
                "ripe_tomatoes": detections,
                "count": len(detections),
            }

            if detections:
                print(json.dumps(payload))
                try:
                    with open("detected_tomatoes.json", "w") as f:
                        json.dump(payload, f)
                except Exception as exc:
                    print(f"[ERROR] Cannot write detected_tomatoes.json: {exc}")

                for i, det in enumerate(detections, start=1):
                    dist = float(det.get("distance_cm", 1e9))
                    if dist > ARM_REACH_CM:
                        print(
                            f"[WARN] Tomato #{i} OUT OF REACH ({dist:.1f}cm) "
                            "— move closer"
                        )

            annotated = detector.annotate(frame, detections)
            cv2.imshow("Ripe Tomato Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
