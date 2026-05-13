#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d /opt/ros ]]; then
  echo "ERROR: /opt/ros does not exist. ROS is not installed on this system." >&2
  echo "Install ROS1 first, then rerun this script." >&2
  exit 1
fi

mapfile -t DISTROS < <(ls /opt/ros | sort)
if [[ ${#DISTROS[@]} -eq 0 ]]; then
  echo "ERROR: /opt/ros exists but contains no ROS distributions." >&2
  exit 1
fi

ROS_DISTRO="${ROS_DISTRO:-${DISTROS[0]}}"
SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ ! -f "$SETUP" ]]; then
  echo "ERROR: $SETUP does not exist." >&2
  echo "Available ROS distributions under /opt/ros:" >&2
  printf '  %s\n' "${DISTROS[@]}" >&2
  echo "Run with one explicitly, for example:" >&2
  echo "  ROS_DISTRO=${DISTROS[0]} $0" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$SETUP"

if ! command -v roscore >/dev/null 2>&1 || ! command -v rostopic >/dev/null 2>&1; then
  echo "ERROR: ${ROS_DISTRO} does not look like ROS1 (missing roscore/rostopic)." >&2
  echo "This project currently uses ROS1 Python APIs (rospy + cv_bridge)." >&2
  echo "If your robot image only has ROS2, install a ROS1 image/package set or run the OpenCV camera path." >&2
  exit 1
fi

echo "Using ROS distro: ${ROS_DISTRO}"
echo "Installing usb_cam and cv_bridge packages..."
sudo apt update
sudo apt install -y "ros-${ROS_DISTRO}-usb-cam" "ros-${ROS_DISTRO}-cv-bridge"

echo
echo "OK. Source ROS before running camera/harvester commands:"
echo "  source ${SETUP}"
