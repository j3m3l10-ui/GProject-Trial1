#!/usr/bin/env bash
set -euo pipefail

echo "Searching for Hiwonder Board.py / SDK folders..."
SEARCH_ROOTS=(
  "$PWD"
  "$HOME"
  "/home/pi"
  "/home/ubuntu"
  "/home/hiwonder"
  "/usr/local/lib"
  "/usr/lib"
)

FOUND=0
for root in "${SEARCH_ROOTS[@]}"; do
  [[ -d "$root" ]] || continue
  while IFS= read -r path; do
    FOUND=1
    echo "  $path"
  done < <(find "$root" -maxdepth 5 \( -path "*/HiwonderSDK/Board.py" -o -path "*/hiwonder/Board.py" -o -name "Board.py" \) 2>/dev/null | sort)
done

if [[ "$FOUND" -eq 0 ]]; then
  echo "No Hiwonder SDK Board.py file found."
  echo "Copy/install the HiwonderSDK folder from the ArmPi Pro software image or source package."
  exit 1
fi

echo
echo "If Board.py is at /path/to/HiwonderSDK/Board.py, run:"
echo "  export HIWONDER_SDK_PATH=/path/to"
echo "If Board.py is standalone at /path/to/Board.py, run:"
echo "  export HIWONDER_SDK_PATH=/path/to"
