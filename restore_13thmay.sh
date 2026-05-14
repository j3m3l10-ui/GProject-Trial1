#!/bin/bash
# Restore workspace to 13thmay checkpoint (ad834f4d1b497bd6cab47bec070b1e7091de3f9c)
# Usage: bash ./restore_13thmay.sh

git checkout cursor/fix-vision-arm-integration-3a7c && git reset --hard ad834f4d1b497bd6cab47bec070b1e7091de3f9c
