#!/usr/bin/env bash
set -euo pipefail

echo "Detected video devices:"
if compgen -G "/dev/video*" >/dev/null; then
  ls -l /dev/video*
else
  echo "  none"
fi

echo
if command -v v4l2-ctl >/dev/null 2>&1; then
  echo "v4l2 device list:"
  v4l2-ctl --list-devices || true
  echo
  echo "Capture-capable devices and formats:"
  for dev in /dev/video*; do
    [[ -e "$dev" ]] || continue
    if v4l2-ctl --device="$dev" --all 2>/dev/null | grep -q "Video Capture"; then
      echo "==== $dev ===="
      v4l2-ctl --device="$dev" --list-formats-ext || true
    fi
  done
else
  echo "v4l2-ctl is not installed. Install with:"
  echo "  sudo apt update && sudo apt install -y v4l-utils"
fi
