#!/usr/bin/env bash
# Restore the project to the "fixed april27" checkpoint.
# Usage:
#   ./restore_fixed_april27.sh           # hard reset working tree to the tag
#   ./restore_fixed_april27.sh --branch  # check out the tag in a new branch (safe)

set -euo pipefail

TAG="fixed-april27"
BRANCH_NAME="fixed-april27-restore"

cd "$(dirname "$0")"

if ! git rev-parse --verify "$TAG" >/dev/null 2>&1; then
    echo "ERROR: git tag '$TAG' not found." >&2
    exit 1
fi

if [[ "${1:-}" == "--branch" ]]; then
    echo "Creating branch '$BRANCH_NAME' at tag '$TAG'..."
    git checkout -B "$BRANCH_NAME" "$TAG"
    echo "Done. You are now on branch '$BRANCH_NAME' at $TAG."
else
    echo "Hard-resetting working tree to tag '$TAG'..."
    git checkout "$TAG"
    echo "Done. HEAD is detached at $TAG."
    echo "Tip: pass --branch to create a working branch instead."
fi
