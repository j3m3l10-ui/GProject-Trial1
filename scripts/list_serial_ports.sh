#!/usr/bin/env bash
set -euo pipefail

echo "Candidate serial/UART devices:"
for pattern in /dev/ttyAMA* /dev/ttyS* /dev/ttyUSB* /dev/ttyACM*; do
  for dev in $pattern; do
    [[ -e "$dev" ]] && ls -l "$dev"
  done
done

echo
if [[ -d /dev/serial/by-id ]]; then
  echo "/dev/serial/by-id:"
  ls -l /dev/serial/by-id
else
  echo "/dev/serial/by-id does not exist."
fi

echo
echo "Current user groups:"
id
echo
echo "If the UART device exists but permission is denied, add your user to dialout:"
echo "  sudo usermod -aG dialout $USER"
echo "  # then log out/in or reboot"
