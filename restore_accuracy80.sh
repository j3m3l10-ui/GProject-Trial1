#!/usr/bin/env bash
set -euo pipefail

# Restore repository to accuracy80 checkpoint snapshot.
# Usage:
#   ./restore_accuracy80.sh           # hard-restore current branch
#   ./restore_accuracy80.sh --branch  # create/switch to accuracy80-restore branch

cd "$(dirname "$0")"

if [[ ! -d .git ]]; then
  echo "[ERROR] Not a git repository."
  exit 1
fi

if ! git rev-parse -q --verify "refs/tags/accuracy80" >/dev/null; then
  echo "[ERROR] Tag 'accuracy80' not found."
  echo "If needed, recover from bundle: git clone backups/accuracy80.bundle recovered_repo"
  exit 1
fi

mode="hard"
if [[ "${1:-}" == "--branch" ]]; then
  mode="branch"
fi

if [[ "$mode" == "branch" ]]; then
  git checkout -B accuracy80-restore accuracy80
  echo "[OK] Switched to branch 'accuracy80-restore' at tag accuracy80."
else
  git reset --hard accuracy80
  git clean -fd
  echo "[OK] Working tree restored to accuracy80."
fi

echo "[INFO] Current commit: $(git rev-parse --short HEAD)"
