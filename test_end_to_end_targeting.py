#!/usr/bin/env python3
"""
End-to-end diagnostic: validates transform chain from detection → IK → arm movement → visual feedback.

Tests:1. Detect tomato in camera frame
2. Transform to arm frame
3. Solve IK
4. Execute tiny motion
5. Re-detect to see if arm moved toward target
6. Measure visual error and transform error
"""

import sys
sys.path.insert(0, '/media/metu2metu/ADATA UFD/GProject Trial1')

import numpy as np
import cv2
import time

from main import (
    camera_to_arm_frame,
    _coarse_camera_pose_at_home,
    get_home_angles,
    COARSE_CAMERA_TO_ARM_ROT,
    COARSE_CAMERA_ORIGIN_M,
    ARM_BASE_POS,
)
from arm_controller import FiveDOFArm
from servo_driver import ServoDriver
from vision import YOLODetector

def main():
    print("\n" + "="*70)
    print("END-TO-END TARGETING DIAGNOSTIC")
    print("="*70)
    
    # Get FK camera pose at home
    print("\n1. FK CAMERA COMPUTATION AT HOME")
    print("-" * 70)
    cam_rot, cam_origin, cam_mode = _coarse_camera_pose_at_home()
    print(f"   Mode: {cam_mode}")
    print(f"   Camera origin (arm frame): {cam_origin}")
    print(f"   Camera rotation matrix:\n{cam_rot}")
    
    # Initialize arm
    print("\n2. ARM INITIALIZATION")
    print("-" * 70)
    arm = FiveDOFArm()
    home_angles = get_home_angles()
    arm.set_joint_angles(home_angles)
    ee_pos = arm.end_effector_pos(home_angles)
    print(f"   Home angles: {home_angles}")
    print(f"   End effector pos at home: {ee_pos}")
    print(f"   ARM_BASE_POS: {ARM_BASE_POS}")
    
    # Initialize camera
    print("\n3. CAMERA INITIALIZATION")
    print("-" * 70)
    import os
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.5)
    
    detector = YOLODetector()
    print("   Camera and detector initialized")
    
    # Wait for user to position tomato
    print("\n4. USER SETUP")
    print("-" * 70)
    print("   Position a tomato in the camera view.")
    print("   Press ENTER when ready...")
    input()
    
    # Capture frames and detect
    print("\n5. DETECTION & TRANSFORM CHAIN")
    print("-" * 70)
    detections = []
    for frame_idx in range(30):  # 1 second at 30fps
        ret, frame = cap.read()
        if not ret:
            print("   ERROR: Cannot read frame!")
            return
        
        results = detector.detect(frame, conf_thresh=0.4)
        if results and len(results) > 0:
            det = results[0]  # Closest or first detection
            detections.append(det)
            print(f"   Frame {frame_idx}: detected tomato at "
                  f"x_cm={det['x']:.1f}, y_cm={det['y']:.1f}, z_cm={det['z']:.1f}")
    
    if not detections:
        print("   ERROR: No tomato detected!")
        cap.release()
        return
    
    # Use median of detections for stability
    det_x_vals = [d['x'] for d in detections]
    det_y_vals = [d['y'] for d in detections]
    det_z_vals = [d['z'] for d in detections]
    
    tomato_xyz_cm = {
        'x': np.median(det_x_vals),
        'y': np.median(det_y_vals),
        'z': np.median(det_z_vals),
    }
    
    print(f"\n   Median detection ({len(detections)} frames):")
    print(f"   Camera frame: x={tomato_xyz_cm['x']:.2f}cm, "
          f"y={tomato_xyz_cm['y']:.2f}cm, z={tomato_xyz_cm['z']:.2f}cm")
    
    # Step 1: Raw detection
    cam_xyz_m = np.array([
        float(tomato_xyz_cm['x']) / 100.0,
        float(tomato_xyz_cm['y']) / 100.0,
        float(tomato_xyz_cm['z']) / 100.0,
    ], dtype=float)
    print(f"   Camera frame (m): {cam_xyz_m}")
    
    # Step 2: Transform to arm frame (show both methods)
    print(f"\n6. CAMERA → ARM FRAME TRANSFORM")
    print("-" * 70)
    
    # Method 1: Using coarse model
    target_m_coarse = COARSE_CAMERA_TO_ARM_ROT @ cam_xyz_m + COARSE_CAMERA_ORIGIN_M
    print(f"   Method 1 (coarse FK-based):")
    print(f"   target = cam_rot @ cam_xyz + cam_origin")
    print(f"   target = {target_m_coarse}")
    
    # Method 2: Using main.py function
    target_m_canon = camera_to_arm_frame(tomato_xyz_cm)
    print(f"   Method 2 (camera_to_arm_frame function):")
    print(f"   target = {target_m_canon}")
    
    if np.allclose(target_m_coarse, target_m_canon, atol=1e-6):
        print(f"   ✓ Both methods agree!")
    else:
        print(f"   ✗ MISMATCH! Difference: {np.linalg.norm(target_m_canon - target_m_coarse):.6f}m")
    
    target_m = np.array(target_m_canon, dtype=float)
    
    # Step 3: Check reachability
    print(f"\n7. REACHABILITY CHECK")
    print("-" * 70)
    is_reach = arm.is_reachable(target_m)
    dist_to_base = np.linalg.norm(target_m - ARM_BASE_POS)
    dist_to_ee = np.linalg.norm(target_m - ee_pos)
    print(f"   Is reachable: {is_reach}")
    print(f"   Distance from base: {dist_to_base:.4f}m")
    print(f"   Distance from current EE: {dist_to_ee:.4f}m")
    
    if not is_reach:
        print(f"   ✗ TARGET NOT REACHABLE! Skipping IK test.")
        cap.release()
        return
    
    # Step 4: Solve IK
    print(f"\n8. INVERSE KINEMATICS SOLVE")
    print("-" * 70)
    q_home = home_angles
    accepted, solved, err, iters, q_sol, used_target = arm.inverse_kinematics(
        target_m,
        max_iters=300,
        tol=5e-4,
    )
    print(f"   Solved: {solved}")
    print(f"   Accepted: {accepted}")
    print(f"   Error: {err:.6f}m")
    print(f"   Iterations: {iters}")
    print(f"   Solution angles: {q_sol}")
    
    if not accepted:
        print(f"   ✗ IK DID NOT CONVERGE! Skipping motion test.")
        cap.release()
        return
    
    # Verify FK at solution
    ee_sol = arm.end_effector_pos(q_sol)
    ik_residual = np.linalg.norm(ee_sol - target_m)
    print(f"   EE pos at solution: {ee_sol}")
    print(f"   IK residual: {ik_residual:.6f}m")
    
    # Step 5: Execute small motion toward target
    print(f"\n9. EXECUTE SMALL MOTION (HOME → INTERMEDIATE)")
    print("-" * 70)
    q_interp = 0.3 * q_sol + 0.7 * q_home  # 30% toward solution
    print(f"   Interpolated angles (30% toward target): {q_interp}")
    
    try:
        driver = ServoDriver(backend="uart.pi")
        driver.move_servos_smooth(q_interp, duration_ms=2000)
        time.sleep(2.5)
        arm.set_joint_angles(q_interp)
        print(f"   ✓ Motion executed")
    except Exception as e:
        print(f"   ✗ Motion failed: {e}")
        cap.release()
        return
    
    # Step 6: Re-detect tomato
    print(f"\n10. POST-MOTION RE-DETECTION")
    print("-" * 70)
    detections2 = []
    for frame_idx in range(30):
        ret, frame = cap.read()
        if not ret:
            continue
        results = detector.detect(frame, conf_thresh=0.4)
        if results and len(results) > 0:
            det = results[0]
            detections2.append(det)
            print(f"   Frame {frame_idx}: x={det['x']:.1f}, y={det['y']:.1f}, z={det['z']:.1f}")
    
    if not detections2:
        print("   ✗ Tomato lost after motion!")
        cap.release()
        return
    
    # Median post-motion detection
    tomato_xyz_cm_post = {
        'x': np.median([d['x'] for d in detections2]),
        'y': np.median([d['y'] for d in detections2]),
        'z': np.median([d['z'] for d in detections2]),
    }
    
    print(f"\n   Post-motion detection ({len(detections2)} frames):")
    print(f"   Camera frame: x={tomato_xyz_cm_post['x']:.2f}cm, "
          f"y={tomato_xyz_cm_post['y']:.2f}cm, z={tomato_xyz_cm_post['z']:.2f}cm")
    
    # Step 7: Analyze motion
    print(f"\n11. MOTION ANALYSIS")
    print("-" * 70)
    
    # Pixel change
    px_delta = tomato_xyz_cm_post['x'] - tomato_xyz_cm['x']
    py_delta = tomato_xyz_cm_post['y'] - tomato_xyz_cm['y']
    pz_delta = tomato_xyz_cm_post['z'] - tomato_xyz_cm['z']
    
    print(f"   Pixel delta (cm): x={px_delta:.2f}, y={py_delta:.2f}, z={pz_delta:.2f}")
    
    # Did it move toward center?
    if abs(px_delta) < 2 and abs(py_delta) < 2:
        print(f"   ✓ Tomato stayed roughly centered in frame (good!)")
    elif px_delta < 0 and py_delta > 0:
        print(f"   ✓ Tomato moved in expected direction")
    else:
        print(f"   ? Tomato moved: checking if direction is reasonable...")
    
    # New arm frame target
    target_m_post = camera_to_arm_frame(tomato_xyz_cm_post)
    target_shift_m = np.linalg.norm(target_m_post - target_m)
    
    print(f"   Arm frame shift: {target_shift_m*100:.2f}mm")
    print(f"   Original target:  {target_m}")
    print(f"   New target:       {target_m_post}")
    
    # Current EE position
    ee_interp = arm.end_effector_pos(q_interp)
    ee_to_orig = np.linalg.norm(ee_interp - target_m)
    ee_to_new = np.linalg.norm(ee_interp - target_m_post)
    
    print(f"\n   Distance from new EE to original target: {ee_to_orig:.4f}m")
    print(f"   Distance from new EE to new target: {ee_to_new:.4f}m")
    
    if ee_to_orig < ee_to_new:
        print(f"   ✓ Arm moved TOWARD target (distance decreased)")
    else:
        print(f"   ✗ Arm moved AWAY or orthogonal to target!")
        print(f"   This suggests camera transform or IK may be wrong!")
    
    # Cleanup
    cap.release()
    driver.close_gripper()
    print(f"\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
