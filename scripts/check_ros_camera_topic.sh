#!/usr/bin/env bash
set -euo pipefail

TOPIC="${1:-/hiwonder_usb_camera/image_raw}"

if ! command -v rostopic >/dev/null 2>&1; then
  echo "ERROR: rostopic not found. Source ROS first." >&2
  exit 1
fi

echo "Available image topics:"
rostopic list | grep -E 'image|camera' || true

echo
echo "Waiting for one frame on $TOPIC ..."
rostopic echo -n 1 "$TOPIC" header
echo
echo "OK: $TOPIC is publishing frames."
