#!/usr/bin/env python3
import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Test a V4L2/OpenCV camera device for streaming frames.")
    parser.add_argument("--camera", type=int, required=True,
                        help="Camera index, e.g. 0 for /dev/video0")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of frames to read")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open /dev/video{args.camera}")

    ok_count = 0
    start = time.time()
    last_shape = None
    for _ in range(args.frames):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            ok_count += 1
            last_shape = frame.shape
        else:
            print("WARN: frame read failed")

    cap.release()
    elapsed = max(1e-6, time.time() - start)
    print(f"Read {ok_count}/{args.frames} frames from /dev/video{args.camera}")
    print(f"Measured FPS: {ok_count / elapsed:.2f}")
    print(f"Last frame shape: {last_shape}")

    if ok_count == 0:
        raise SystemExit("ERROR: camera opened but did not stream frames")


if __name__ == "__main__":
    main()
