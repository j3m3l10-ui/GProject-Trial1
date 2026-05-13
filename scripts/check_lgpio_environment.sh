#!/usr/bin/env bash
set -euo pipefail

echo "Checking Raspberry Pi GPIO / lgpio environment..."
echo

echo "GPIO character devices:"
if compgen -G "/dev/gpiochip*" >/dev/null; then
  ls -l /dev/gpiochip*
else
  echo "  none"
fi

echo
echo "lgpio daemon/process:"
if pgrep -a lgd >/dev/null 2>&1; then
  pgrep -a lgd
else
  echo "  lgd is not running"
fi

echo
echo "Python lgpio import:"
python3 - <<'PY'
try:
    import lgpio
    print("  OK: lgpio imported from", getattr(lgpio, "__file__", "built-in"))
except Exception as exc:
    print("  ERROR:", exc)
PY

echo
echo "If SDK backend fails with .lgd-nfy-* missing:"
echo "  1) Prefer auto fallback: python3 main.py --servo-backend auto --servo-self-test-only"
echo "  2) Or bypass SDK:       python3 main.py --servo-backend uart --uart-port auto --servo-self-test-only"
echo "  3) If you require SDK, install/fix lgpio for your Raspberry Pi OS image."
