#!/usr/bin/env python3
"""
Diagnostic script to verify camera-to-arm frame transformation
Shows the exact FK computation at home and the resulting transform
"""

import numpy as np
from arm_controller import FiveDOFArm
from arm_home_position import get_home_angles
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ── Test the FK at home pose ──
logger.info("="*70)
logger.info("FORWARD KINEMATICS AT HOME POSE")
logger.info("="*70)

arm = FiveDOFArm()
home_angles = get_home_angles()
arm.set_joint_angles(home_angles)

logger.info(f"\nHome angles (radians): {home_angles}")
logger.info(f"Home angles (degrees): {np.degrees(home_angles)}")

positions, rotations = arm.forward_kinematics(home_angles)

logger.info(f"\nFK yields {len(positions)} frames (0-indexed):")
for i, (pos, rot) in enumerate(zip(positions, rotations)):
    logger.info(f"\n  Frame {i}: position={pos}")
    logger.info(f"  Frame {i}: rotation=\n{rot}")

logger.info("\n" + "="*70)
logger.info("CAMERA MOUNT LOCATION (ID2 bracket - wrist output)")
logger.info("="*70)

# Camera is at wrist output (index 4)
# This is after wrist pitch (ID3) and before gripper roll (ID1)
wrist_idx = 4
wrist_pos = positions[wrist_idx]
wrist_rot = rotations[wrist_idx]

logger.info(f"\nWrist (ID3) output position: {wrist_pos}")
logger.info(f"Wrist (ID3) output rotation:\n{wrist_rot}")

# Compute camera position with small forward offset
camera_offset_local = np.array([0.020, 0.000, 0.010])  # 2cm fwd, 1cm up
camera_pos = wrist_pos + wrist_rot @ camera_offset_local

logger.info(f"\nCamera local offset in wrist frame: {camera_offset_local}")
logger.info(f"Camera position in ARM BASE FRAME: {camera_pos}")

# Camera rotation: wrist rotation + camera's look direction
# Camera looks +X forward from the wrist
camera_rot = wrist_rot

logger.info(f"Camera rotation matrix:\n{camera_rot}")

logger.info("\n" + "="*70)
logger.info("TRANSFORM VERIFICATION")
logger.info("="*70)

# Test a sample point
test_point_cm = {"x": 25.0, "y": 0.0, "z": 30.0}  # 30cm ahead in camera frame
test_point_m = np.array([test_point_cm["x"], test_point_cm["y"], test_point_cm["z"]]) / 100.0

logger.info(f"\nTest point in camera frame: {test_point_cm} (cm)")
logger.info(f"Test point in camera frame: {test_point_m} (m)")

# Transform using FK-based camera model
arm_point = camera_rot @ test_point_m + camera_pos

logger.info(f"\nTransformed to ARM BASE FRAME: {arm_point} (m)")
logger.info(f"Transformed to ARM BASE FRAME: {arm_point * 100} (cm)")

logger.info("\nExpected behavior:")
logger.info(f"  - If detection is 30cm ahead (z), arm should move ~30cm forward")
logger.info(f"  - If detection is 25cm right (x), arm should move ~25cm in arm-x direction")
logger.info(f"  - Height should match camera height at home + detection offset")

logger.info("\n" + "="*70)
