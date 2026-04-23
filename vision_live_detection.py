import cv2
import time
from vision import ThreadedCamera, TomatoDetector

# Initialize camera and detector
camera = ThreadedCamera(camera_index=0, width=640, height=480)
detector = TomatoDetector()
camera.start()

print("[VISION] Camera started. Press 'q' to quit.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("[VISION] Waiting for camera frame...")
            time.sleep(0.1)
            continue
        detections = detector.detect(frame)
        for i, det in enumerate(detections):
            bbox = det['bbox_px']
            ripeness = det['ripeness_name']
            dist = det['distance_cm']
            print(f"Tomato #{i+1}: bbox={bbox}, ripeness={ripeness}, distance={dist}cm")
            # Draw bounding box and info
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(frame, f"{ripeness} {dist:.1f}cm", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Tomato Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("[VISION] Camera stopped.")
