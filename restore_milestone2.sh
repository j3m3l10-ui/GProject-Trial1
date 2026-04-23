#!/usr/bin/env bash
set -euo pipefail

# Restore repository to Milestone 2 snapshot.
# Usage:
#   ./restore_milestone2.sh           # hard-restore current branch
#   ./restore_milestone2.sh --branch  # create/switch to milestone2-restore branch

cd "$(dirname "$0")"

if [[ ! -d .git ]]; then
  echo "[ERROR] Not a git repository."
  exit 1
fi

if ! git rev-parse -q --verify "refs/tags/milestone2" >/dev/null; then
  echo "[ERROR] Tag 'milestone2' not found."
  echo "If needed, recover from bundle: git clone backups/milestone2.bundle recovered_repo"
  exit 1
fi

mode="hard"
if [[ "${1:-}" == "--branch" ]]; then
  mode="branch"
fi

if [[ "$mode" == "branch" ]]; then
  git checkout -B milestone2-restore milestone2
  echo "[OK] Switched to branch 'milestone2-restore' at tag milestone2."
else
  git reset --hard milestone2
  git clean -fd
  echo "[OK] Working tree restored to milestone2."
fi

echo "[INFO] Current commit: $(git rev-parse --short HEAD)"
