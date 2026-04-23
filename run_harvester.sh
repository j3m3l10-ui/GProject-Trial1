#!/usr/bin/env bash
set -euo pipefail

# Proven ready-to-run command for current project state.
# Default behavior: one full real hardware pass with automatic confirmation.
# Extra arguments are appended, e.g.:
#   ./run_harvester.sh --no-cut
#   ./run_harvester.sh --test-cycles 3

cd "$(dirname "$0")"
python main.py --single-pass --no-confirm "$@"
