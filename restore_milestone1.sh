#!/usr/bin/env bash
set -euo pipefail

# Restore repository to Milestone 1 snapshot.
# Usage:
#   ./restore_milestone1.sh           # hard-restore current branch
#   ./restore_milestone1.sh --branch  # create/switch to milestone1-restore branch

cd "$(dirname "$0")"

if [[ ! -d .git ]]; then
  echo "[ERROR] Not a git repository."
  exit 1
fi

if ! git rev-parse -q --verify "refs/tags/milestone1" >/dev/null; then
  echo "[ERROR] Tag 'milestone1' not found."
  echo "If needed, recover from bundle: git clone backups/milestone1.bundle recovered_repo"
  exit 1
fi

mode="hard"
if [[ "${1:-}" == "--branch" ]]; then
  mode="branch"
fi

if [[ "$mode" == "branch" ]]; then
  git checkout -B milestone1-restore milestone1
  echo "[OK] Switched to branch 'milestone1-restore' at tag milestone1."
else
  git reset --hard milestone1
  git clean -fd
  echo "[OK] Working tree restored to milestone1."
fi

echo "[INFO] Current commit: $(git rev-parse --short HEAD)"
