#!/usr/bin/env bash
set -euo pipefail

CAMERA_INDEX="${1:-0}"
SERVO_BACKEND="${2:-auto}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2
  exit 1
fi

echo "Testing /dev/video${CAMERA_INDEX} before starting harvester..."
python3 scripts/test_opencv_camera.py --camera "$CAMERA_INDEX" --frames 10

echo
echo "Starting hardware harvester with OpenCV/V4L2 camera index ${CAMERA_INDEX}"
python3 main.py \
  --hardware \
  --camera-source opencv \
  --camera "$CAMERA_INDEX" \
  --servo-backend "$SERVO_BACKEND"
