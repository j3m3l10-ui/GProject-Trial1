#!/usr/bin/env bash
set -euo pipefail

VIDEO_DEVICE="${1:-/dev/video0}"
WIDTH="${2:-640}"
HEIGHT="${3:-480}"
FPS="${4:-30}"
PIXEL_FORMAT="${5:-mjpeg}"
IMAGE_TOPIC="/hiwonder_usb_camera/image_raw"

if ! command -v roscore >/dev/null 2>&1; then
  echo "ERROR: ROS is not sourced. Run: source /opt/ros/<distro>/setup.bash" >&2
  exit 1
fi

if ! command -v roslaunch >/dev/null 2>&1; then
  echo "ERROR: roslaunch not found. Install ROS launch tools first." >&2
  exit 1
fi

if ! rospack find usb_cam >/dev/null 2>&1; then
  echo "ERROR: usb_cam is not installed." >&2
  echo "Install it with one of:" >&2
  echo "  sudo apt update && sudo apt install ros-\${ROS_DISTRO}-usb-cam" >&2
  echo "  cd ~/catkin_ws/src && git clone https://github.com/ros-drivers/usb_cam.git && cd .. && catkin_make" >&2
  exit 1
fi

if [[ ! -e "$VIDEO_DEVICE" ]]; then
  echo "ERROR: $VIDEO_DEVICE does not exist. Check: ls /dev/video*" >&2
  exit 1
fi

echo "Starting Hiwonder USB camera on $VIDEO_DEVICE -> $IMAGE_TOPIC"
echo "Resolution=${WIDTH}x${HEIGHT}, fps=$FPS, pixel_format=$PIXEL_FORMAT"
roslaunch "$(dirname "$0")/../launch/hiwonder_usb_camera.launch" \
  video_device:="$VIDEO_DEVICE" \
  image_width:="$WIDTH" \
  image_height:="$HEIGHT" \
  framerate:="$FPS" \
  pixel_format:="$PIXEL_FORMAT"
