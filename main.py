"""
Integrated Tomato-Harvesting System — main.py
===============================================
Unifies the vision subsystem (YOLOv8 + filters) with the 5-DOF robotic arm
subsystem for autonomous ripe-tomato detection, approach, cut, and collection.

Workflow:
  1. ARM → Search Home (camera faces outward)
  2. VISION → Continuous detection loop on live camera
  3. LOCK → When a ripe tomato is detected consistently for CONFIRM_SECONDS,
             lock its 3D position
  4. ARM → Solve IK for the cut point (1 cm from tomato surface toward stem)
  5. ARM → Move from Search Home → cut pose over safe trajectory
  6. ARM → Close gripper / scissors to cut
  7. ARM → Retract to Search Home
  8. Repeat

Usage:
  python main.py                  # real hardware (RPi5 + Hiwonder servos)
  python main.py --sim            # simulation mode (no servos, log only)
  python main.py --gui            # launch 3D simulation GUI instead

Hand-in-Eye: camera is mounted between wrist (ID3) and gripper (ID1).
The arm must be at Search Home for the camera to see the workspace.
"""

import argparse
import json
import logging
import math
import sys
import time
import cv2
import numpy as np

from vision import TomatoDetector
from arm_controller import (
    FiveDOFArm, angles_to_pulses, search_home_angles,
    compute_cut_point, SERVO_IDS, SEARCH_HOME_PULSES,
)
from servo_driver import ServoDriver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIRM_SECONDS    = 5.0     # seconds a tomato must be visible before acting
CONFIRM_FRAMES     = 8       # minimum detections within the confirm window
MOVE_DURATION_MS   = 800     # servo move duration for arm motion
GRIPPER_DELAY_S    = 0.6     # wait after gripper command
RETRACT_DELAY_S    = 1.0     # wait after retract before next scan
TOMATO_RADIUS_M    = 0.035   # average tomato radius in metres
CAMERA_INDEX       = 0       # default camera

# ── Camera-to-arm transform ───────────────────────────────────────────────────
# The camera is Hand-in-Eye (mounted between the wrist and gripper).
# During scanning the arm is at Search Home; we use the arm's own forward
# kinematics to compute the exact camera origin in the arm-base frame so the
# transform stays correct for any calibrated link lengths.
#
# CAMERA_TILT_RAD — downward tilt of the camera from the arm's forward
#   (link) axis, in radians.  Positive = camera nose tilts toward the floor.
#   Measure or tune on the real robot; 0 = perfectly level (good first guess).
#
# CAMERA_MOUNT_AHEAD_M — how many metres the camera optical centre sits ahead
#   of the wrist joint along the link axis.  Defaults to half the last link.
CAMERA_TILT_RAD = 0.0       # radians — tune on real robot
CAMERA_MOUNT_AHEAD_M = 0.035  # metres  — approx half the last link (0.070 m)


def camera_to_arm_frame(xyz_cm_dict, arm):
    """
    Convert a camera-frame detection (x, y, z in cm) to arm-base frame (metres).

    Coordinate conventions
    ----------------------
    Camera frame  : X = right,   Y = down,    Z = forward (into scene)
    Arm base frame: X = forward, Y = left,     Z = up

    The camera origin in arm-base frame is computed from the arm's FK at
    Search Home, then shifted forward by CAMERA_MOUNT_AHEAD_M along the
    end-effector link axis to place it at the camera optical centre.

    A small pitch correction (CAMERA_TILT_RAD) rotates the depth-to-X and
    height-to-Z mappings so the transform stays accurate even when the camera
    is tilted slightly downward to view the workspace.
    """
    cam = np.array([xyz_cm_dict["x"], xyz_cm_dict["y"], xyz_cm_dict["z"]],
                   dtype=float) / 100.0

    # ── Step 1: camera origin from FK at Search Home ──────────────────────────
    home_q = search_home_angles()
    positions, rotations = arm.forward_kinematics(home_q)
    R_ee = rotations[-1]                          # end-effector rotation (arm frame)
    ee_pos = np.array(positions[-1], dtype=float) # end-effector position (arm frame)

    # Move back from the gripper tip toward the wrist by (last link – mount ahead)
    last_link = arm.link_lengths[-1]
    cam_origin = ee_pos + R_ee.dot(
        [CAMERA_MOUNT_AHEAD_M - last_link, 0.0, 0.0]
    )

    # ── Step 2: axis permutation + tilt correction ────────────────────────────
    # Base mapping (zero tilt):  cam_Z → arm_X,  -cam_X → arm_Y,  -cam_Y → arm_Z
    # With a downward camera tilt θ, cam_Z and cam_Y are mixed in the X/Z plane:
    ct = math.cos(CAMERA_TILT_RAD)
    st = math.sin(CAMERA_TILT_RAD)
    arm_x = cam[2] * ct + cam[1] * st + cam_origin[0]   # depth  (+tilt)
    arm_y = -cam[0]            + cam_origin[1]            # lateral (unchanged)
    arm_z = -cam[1] * ct + cam[2] * st + cam_origin[2]   # vertical (+tilt)

    return np.array([arm_x, arm_y, arm_z], dtype=float)


# ── Trajectory interpolation ──────────────────────────────────────────────────
def interpolate_trajectory(q_start, q_end, steps=20):
    """Linear joint-space interpolation."""
    traj = []
    for s in range(steps + 1):
        alpha = s / steps
        q = (1 - alpha) * np.array(q_start) + alpha * np.array(q_end)
        traj.append(q)
    return traj


# ── Main harvesting loop ──────────────────────────────────────────────────────
def run_harvesting(sim_mode=False):
    mode = "sim" if sim_mode else "real"
    logger.info(f"Starting harvesting system in {mode.upper()} mode")

    # Initialise subsystems
    detector = TomatoDetector()
    arm = FiveDOFArm()
    driver = ServoDriver(mode=mode)

    # Move to Search Home
    logger.info("Moving arm to Search Home position...")
    home_angles = search_home_angles()
    arm.set_joint_angles(home_angles)
    driver.go_search_home(duration_ms=1200)
    time.sleep(1.5)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open camera!")
        return
    logger.info(f"Camera opened: index={CAMERA_INDEX}")

    # Detection confirmation buffer
    confirm_buffer = []  # list of (timestamp, detection_dict)
    state = "SCANNING"   # SCANNING → CONFIRMING → ACTING → RETRACTING

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed")
                continue

            if state in ("SCANNING", "CONFIRMING"):
                detections = detector.detect(frame)
                annotated = detector.annotate(frame, detections)
                now = time.time()

                if detections:
                    # Pick the highest-confidence detection
                    best = max(detections, key=lambda d: d["confidence"])

                    confirm_buffer.append((now, best))
                    # Purge old entries outside the confirm window
                    confirm_buffer = [(t, d) for t, d in confirm_buffer
                                      if now - t <= CONFIRM_SECONDS]

                    if state == "SCANNING":
                        state = "CONFIRMING"
                        logger.info(f"Tomato spotted! Confirming for {CONFIRM_SECONDS}s...")

                    # Check if we have enough consistent detections
                    if (len(confirm_buffer) >= CONFIRM_FRAMES and
                            now - confirm_buffer[0][0] >= CONFIRM_SECONDS):
                        # Median position across confirmed frames (robust to outliers)
                        positions = [d["xyz_cm"] for _, d in confirm_buffer]
                        med_x = float(np.median([p["x"] for p in positions]))
                        med_y = float(np.median([p["y"] for p in positions]))
                        med_z = float(np.median([p["z"] for p in positions]))
                        locked_xyz_cm = {"x": med_x, "y": med_y, "z": med_z}

                        logger.info(f"LOCKED tomato at (cm): "
                                    f"X={med_x:.1f} Y={med_y:.1f} Z={med_z:.1f}")

                        state = "ACTING"
                        confirm_buffer.clear()

                        # ── Execute harvest sequence ───────────────────────────
                        _execute_harvest(arm, driver, locked_xyz_cm, annotated)

                        state = "RETRACTING"
                        # Retract to Search Home
                        logger.info("Retracting to Search Home...")
                        arm.set_joint_angles(home_angles)
                        pulses = angles_to_pulses(home_angles)
                        driver.move_servos(pulses, duration_ms=1000)
                        time.sleep(RETRACT_DELAY_S)

                        state = "SCANNING"
                        logger.info("Ready for next tomato.\n")
                else:
                    # No detection this frame — decay buffer
                    confirm_buffer = [(t, d) for t, d in confirm_buffer
                                      if now - t <= CONFIRM_SECONDS]
                    if not confirm_buffer:
                        state = "SCANNING"

                # Show status on annotated frame
                status_text = f"State: {state}"
                if state == "CONFIRMING":
                    elapsed = now - confirm_buffer[0][0] if confirm_buffer else 0
                    status_text += f"  ({elapsed:.1f}/{CONFIRM_SECONDS}s, " \
                                   f"{len(confirm_buffer)} frames)"
                cv2.putText(annotated, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Tomato Harvester", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q' — exiting.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        driver.go_park(duration_ms=1000)
        driver.close()
        logger.info("System shut down.")


def _execute_harvest(arm, driver, locked_xyz_cm, frame):
    """Execute the full harvest: IK solve → move → cut → open gripper."""
    # Convert camera-frame coords to arm-base frame (uses FK-computed camera pose)
    target_m = camera_to_arm_frame(locked_xyz_cm, arm)
    logger.info(f"Arm-frame target (m): {target_m}")

    # Check reachability
    if not arm.is_reachable(target_m):
        logger.warning("Target is OUT OF REACH — skipping.")
        return

    # Compute cut point (1 cm from tomato surface toward base/stem)
    edge_pt, cut_pt = compute_cut_point(target_m, TOMATO_RADIUS_M, arm.base_pos)
    logger.info(f"Cut point (m): {cut_pt}")

    # Save current (search-home) joint config
    q_home = arm.joint_angles.copy()

    # Open gripper before approach
    driver.gripper_open(duration_ms=400)
    time.sleep(0.5)

    # Initialise joint angles toward the target before running IK —
    # this prevents the DLS solver from getting trapped in a poor local
    # minimum when the target is to the side or in a corner.
    arm.initialise_for_target(cut_pt)

    # Solve IK for the cut point
    solved, err, iters = arm.inverse_kinematics(cut_pt, max_iters=400, tol=5e-4)
    q_cut = arm.joint_angles.copy()
    logger.info(f"IK solved={solved}, error={err:.4f}m, iters={iters}")

    # FK sanity check: verify the end-effector will actually reach the cut point
    fk_pos = arm.end_effector_pos(q_cut)
    fk_err = float(np.linalg.norm(fk_pos - cut_pt))
    logger.info(f"FK validation error: {fk_err:.4f}m")

    if not solved and err > 0.02:
        logger.warning(f"IK error too large ({err:.4f}m) — aborting harvest.")
        arm.set_joint_angles(q_home)
        return

    # Move arm along interpolated trajectory: home → cut
    traj = interpolate_trajectory(q_home, q_cut, steps=25)
    for q in traj:
        arm.set_joint_angles(q)
        pulses = angles_to_pulses(q)
        driver.move_servos(pulses, duration_ms=MOVE_DURATION_MS // 25)
        time.sleep(MOVE_DURATION_MS / 25000.0)

    # Small settling delay at cut position
    time.sleep(0.3)

    # Close gripper / scissors to cut the stem
    logger.info("CUTTING — closing gripper...")
    driver.gripper_close(duration_ms=500)
    time.sleep(GRIPPER_DELAY_S)
    logger.info("Cut complete.")

    # Open gripper to release (tomato falls into net below)
    driver.gripper_open(duration_ms=400)
    time.sleep(0.3)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tomato Harvesting Robot — Integrated System")
    parser.add_argument("--sim", action="store_true",
                        help="Run in simulation mode (no real servos)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch the 3D simulation GUI")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    args = parser.parse_args()

    global CAMERA_INDEX
    CAMERA_INDEX = args.camera

    if args.gui:
        from simulation_gui import launch_gui
        launch_gui()
    else:
        run_harvesting(sim_mode=args.sim)


if __name__ == "__main__":
    main()
