#!/usr/bin/env python3
"""
Simple validation of camera-to-arm transform without moving hardware.
Tests the transform math and FK computation.
"""

import sys
sys.path.insert(0, '/media/metu2metu/ADATA UFD/GProject Trial1')

import numpy as np
from main import (
    camera_to_arm_frame,
    _coarse_camera_pose_at_home,
    get_home_angles,
    ARM_BASE_POS,
    COARSE_CAMERA_TO_ARM_ROT,
    COARSE_CAMERA_ORIGIN_M,
    COARSE_CAMERA_POSE_MODE,
)
from arm_controller import FiveDOFArm

def main():
    print("\n" + "="*70)
    print("CAMERA TRANSFORM VALIDATION (NO HARDWARE)")
    print("="*70)
    
    # Show what was computed
    print(f"\n1. Coarse Camera Pose Computation")
    print("-"*70)
    print(f"   Mode: {COARSE_CAMERA_POSE_MODE}")
    print(f"   Camera origin: {COARSE_CAMERA_ORIGIN_M}")
    print(f"   Camera rotation matrix:")
    print(COARSE_CAMERA_TO_ARM_ROT)
    
    # Verify eigenvalues of rotation matrix (should be 1 + 2 complex conjugates on unit circle)
    eigs = np.linalg.eigvals(COARSE_CAMERA_TO_ARM_ROT)
    det = np.linalg.det(COARSE_CAMERA_TO_ARM_ROT)
    print(f"   Det(R): {det:.6f} (should be ~1.0)")
    print(f"   Eigenvalues: {eigs}")
    if abs(det - 1.0) < 0.01:
        print(f"   ✓ Looks like a valid rotation matrix")
    else:
        print(f"   ✗ WARNING: Not a valid rotation matrix!")
    
    # Verify orthogonality: R^T @ R = I
    orth_err = np.linalg.norm(COARSE_CAMERA_TO_ARM_ROT.T @ COARSE_CAMERA_TO_ARM_ROT - np.eye(3))
    print(f"   Orthogonality error ||R^T·R - I||: {orth_err:.8f} (should be ~0)")
    
    # Test transform on known points
    print(f"\n2. Transform Test - Known Points")
    print("-"*70)
    
    # Test 1: Camera origin (should map to camera position)
    cam_origin_cam_frame = np.array([0, 0, 0])  # Camera looking at itself
    transformed = COARSE_CAMERA_TO_ARM_ROT @ cam_origin_cam_frame + COARSE_CAMERA_ORIGIN_M
    print(f"   Camera origin in camera frame [0,0,0] → arm frame {transformed}")
    print(f"   Should be camera position: {COARSE_CAMERA_ORIGIN_M}")
    err_origin = np.linalg.norm(transformed - COARSE_CAMERA_ORIGIN_M)
    print(f"   Error: {err_origin:.8f} (should be ~0)")
    
    # Test 2: Point directly in front of camera
    cam_fwd = np.array([0, 0, 0.5])  # 50cm forward in camera +Z
    transformed_fwd = COARSE_CAMERA_TO_ARM_ROT @ cam_fwd + COARSE_CAMERA_ORIGIN_M
    print(f"\n   Point 50cm forward in camera frame [0,0,0.5]")
    print(f"   → arm frame {transformed_fwd}")
    print(f"   Should be camera_origin + camera_rot @ [0,0,0.5]")
    print(f"   Expected: {COARSE_CAMERA_ORIGIN_M + COARSE_CAMERA_TO_ARM_ROT @ cam_fwd}")
    
    # Test 3: Arm base from camera frame
    # Camera is at [-0.1149, 0, 0.1252] in arm frame
    # So arm base [0,0,0] should be at [-CAMERA_ORIGIN] in camera frame
    # But we need to account for rotation
    arm_base_vec_in_arm_frame = np.array(ARM_BASE_POS) - COARSE_CAMERA_ORIGIN_M
    base_in_cam_frame = np.linalg.inv(COARSE_CAMERA_TO_ARM_ROT) @ arm_base_vec_in_arm_frame
    print(f"\n   Arm base [{ARM_BASE_POS[0]}, {ARM_BASE_POS[1]}, {ARM_BASE_POS[2]}] in arm frame")
    print(f"   → should appear at {base_in_cam_frame} in camera frame")
    print(f"   (i.e., -camera_origin rotated into camera frame)")
    
    # Test example detection
    print(f"\n3. Example Detection Transform")
    print("-"*70)
    
    # Simulate a detection: tomato at 5cm right, 30cm forward
    test_detection = {'x': 5, 'y': 0, 'z': 30}  # cm
    test_cam_xyz_m = np.array([0.05, 0, 0.30])  # m
    
    arm_target_m = camera_to_arm_frame(test_detection)
    arm_target_direct = COARSE_CAMERA_TO_ARM_ROT @ test_cam_xyz_m + COARSE_CAMERA_ORIGIN_M
    
    print(f"   Detection: {test_detection}")
    print(f"   Via camera_to_arm_frame(): {arm_target_m}")
    print(f"   Via direct transform: {arm_target_direct}")
    
    if np.allclose(arm_target_m, arm_target_direct, atol=1e-6):
        print(f"   ✓ Functions agree")
    else:
        print(f"   ✗ MISMATCH: {np.linalg.norm(np.array(arm_target_m) - arm_target_direct):.6f}m")
    
    # Test with FK at home to understand arm geometry
    print(f"\n4. FK Geometry at Home Pose")
    print("-"*70)
    
    arm = FiveDOFArm()
    home_angles = get_home_angles()
    positions, rotations = arm.forward_kinematics(home_angles)
    
    print(f"   Home angles (rad): {home_angles}")
    print(f"   Joint angles (deg): {np.degrees(home_angles)}")
    
    # Show all frames
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        print(f"\n   Frame {i} (index {i}):")
        print(f"      Position: {pos}")
        print(f"      Rotation matrix:")
        for row in rot:
            print(f"         {row}")
    
    # Verify camera is at frame 4
    print(f"\n5. Camera Location Verification")
    print("-"*70)
    
    frame4_pos = np.array(positions[4])
    frame4_rot = np.array(rotations[4])
    
    print(f"   Frame 4 position: {frame4_pos}")
    print(f"   Frame 4 rotation:\n{frame4_rot}")
    
    # Apply local offset
    local_offset = np.array([0.04, 0, 0])  # 4cm forward
    camera_pos_from_fk = frame4_pos + frame4_rot @ local_offset
    
    print(f"\n   Local offset: {local_offset}")
    print(f"   Camera position (frame4 + offset): {camera_pos_from_fk}")
    print(f"   COARSE_CAMERA_ORIGIN_M from config: {COARSE_CAMERA_ORIGIN_M}")
    
    err_cam_pos = np.linalg.norm(camera_pos_from_fk - COARSE_CAMERA_ORIGIN_M)
    print(f"   Difference: {err_cam_pos:.6f}m")
    
    if err_cam_pos < 0.001:
        print(f"   ✓ FK matches configured camera position")
    else:
        print(f"   ✗ Mismatch! Config may be outdated.")
    
    print(f"\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
