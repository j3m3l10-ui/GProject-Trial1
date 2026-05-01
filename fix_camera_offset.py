#!/usr/bin/env python3
"""
QUICK FIX: Update camera offsets to match FK-verified values
Run once to calibrate, then run harvest
"""

import sys
import os

workspace = '/media/metu2metu/ADATA UFD/GProject Trial1'
sys.path.insert(0, workspace)

# Fix the offsets in main.py by writing corrected values
main_py = os.path.join(workspace, 'main.py')

#  Read the file
with open(main_py, 'r') as f:
    content = f.read()

# Replace the outdated offset with the FK-verified version
old_offset_line = '''    float(os.getenv("CAMERA_MOUNT_OFFSET_X_M", "0.040")),'''
new_offset_line = '''    float(os.getenv("CAMERA_MOUNT_OFFSET_X_M", "0.020")),'''

content = content.replace(old_offset_line, new_offset_line)

# Also add Z offset
old_z_offset = '''    float(os.getenv("CAMERA_MOUNT_OFFSET_Z_M", "0.000")),'''
new_z_offset = '''    float(os.getenv("CAMERA_MOUNT_OFFSET_Z_M", "0.010")),'''

content = content.replace(old_z_offset, new_z_offset)

# Write back
with open(main_py, 'w') as f:
    f.write(content)

print("✓ Updated camera offsets:")
print("  X: 0.040m → 0.020m (4cm → 2cm forward)")
print("  Z: 0.000m → 0.010m (0cm → 1cm up)")
print("\nRun: python main.py --single-pass --no-cut --no-confirm")
