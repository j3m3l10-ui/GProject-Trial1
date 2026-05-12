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
  4. ARM → Solve IK for the cut point (1 cm above tomato surface along stem)
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
import fcntl
import logging
import os
import struct
import sys
import time
import cv2
import numpy as np

from vision import TomatoDetector
from arm_controller import (
    FiveDOFArm, angles_to_pulses, search_home_angles,
    compute_cut_point, SERVO_IDS, SEARCH_HOME_PULSES,
)
from servo_driver import (
    DEFAULT_BAUD, DEFAULT_SERVO_BACKEND, DEFAULT_UART_PORT,
    GRIPPER_OPEN_PULSE, ServoDriver,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIRM_SECONDS    = 1.0     # seconds a tomato must be visible before acting
CONFIRM_FRAMES     = 3       # minimum detections within the confirm window
CONFIRM_POSITION_TOLERANCE_CM = 6.0  # restart confirmation if target jumps
MOVE_DURATION_MS   = 800     # servo move duration for arm motion
GRIPPER_DELAY_S    = 0.6     # wait after gripper command
RETRACT_DELAY_S    = 1.0     # wait after retract before next scan
TOMATO_RADIUS_M    = 0.035   # average tomato radius in metres
CAMERA_INDEX       = -1      # -1 = auto-detect camera
CAMERA_READ_PROBE_FRAMES = 1 # frames required before accepting a camera
FRAME_GRAB_FAILURE_LIMIT = 3 # reopen camera after this many failed reads
IK_TOLERANCE_M     = 5e-4    # cutter target must solve to sub-millimetre error
IK_MAX_ERROR_M     = 0.005   # never cut with more than 5 mm residual error
STEM_DIRECTION_ARM_FRAME = np.array([0.0, 0.0, 1.0], dtype=float)

VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
V4L2_CAP_DEVICE_CAPS = 0x80000000

# ── Camera-to-arm transform ───────────────────────────────────────────────────
# Because our camera is Hand-in-Eye (mounted on the wrist), the detected
# xyz_cm is in the camera frame.  At Search Home the camera has a known
# pose relative to the arm base.  This offset transforms camera coords
# into arm-base coords.  CALIBRATE THESE VALUES on the real robot.
#
# Approximate offsets when arm is at Search Home position:
#   Camera is ~0.15m in front of the base, ~0.18m above ground, angled down.
CAMERA_OFFSET_M = np.array([0.15, 0.0, 0.18], dtype=float)


def camera_to_arm_frame(xyz_cm_dict):
    """Convert vision xyz_cm (camera frame) → arm base frame (metres)."""
    cam = np.array([xyz_cm_dict["x"], xyz_cm_dict["y"], xyz_cm_dict["z"]],
                   dtype=float) / 100.0
    # In hand-in-eye at Search Home: camera Z ≈ arm X (forward),
    # camera X ≈ arm -Y, camera Y ≈ arm -Z
    arm_x = cam[2] + CAMERA_OFFSET_M[0]   # depth → forward
    arm_y = -cam[0] + CAMERA_OFFSET_M[1]  # left/right
    arm_z = -cam[1] + CAMERA_OFFSET_M[2]  # up/down
    return np.array([arm_x, arm_y, arm_z], dtype=float)


def _camera_candidates(camera_index):
    """Return camera indexes to try. -1 means auto-detect."""
    if camera_index >= 0:
        return [camera_index]

    candidates = []
    try:
        video_names = [
            name for name in os.listdir("/dev")
            if name.startswith("video") and name[5:].isdigit()
        ]
        for name in sorted(video_names, key=lambda n: int(n[5:])):
            if name.startswith("video") and name[5:].isdigit():
                candidates.append(int(name[5:]))
    except OSError:
        pass

    for idx in range(6):
        if idx not in candidates:
            candidates.append(idx)
    return candidates


def _video_device_supports_capture(idx):
    """Return False for V4L2 metadata/control nodes that cannot stream frames."""
    path = f"/dev/video{idx}"
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf, True)
        capabilities = struct.unpack_from("I", buf, 84)[0]
        device_caps = struct.unpack_from("I", buf, 88)[0]
        caps = device_caps if capabilities & V4L2_CAP_DEVICE_CAPS else capabilities
        return bool(caps & (V4L2_CAP_VIDEO_CAPTURE |
                            V4L2_CAP_VIDEO_CAPTURE_MPLANE))
    except OSError:
        # If querying fails, let OpenCV try the device and report the real error.
        return True
    finally:
        if fd is not None:
            os.close(fd)


def _camera_can_read_frame(cap, idx, probe_frames=CAMERA_READ_PROBE_FRAMES):
    """Verify an opened capture actually returns frames."""
    for attempt in range(1, probe_frames + 1):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            return True
        logger.warning(
            "Camera index %s opened but did not return a frame "
            "(probe %s/%s).", idx, attempt, probe_frames)
    return False


def _open_camera(camera_index=CAMERA_INDEX):
    """Open the first usable camera and return (capture, selected_index)."""
    attempted = []
    for idx in _camera_candidates(camera_index):
        attempted.append(idx)
        if not _video_device_supports_capture(idx):
            logger.warning("Camera index %s is not a capture device; skipping.", idx)
            continue

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened() and _camera_can_read_frame(cap, idx):
            logger.info(f"Camera opened and read frame: index={idx}")
            return cap, idx
        cap.release()
        logger.warning(f"Camera index {idx} did not open or did not stream frames.")

    logger.error(
        "Cannot open camera. Tried indexes: %s. Use --camera N if your camera "
        "is on a known index, or check that /dev/video* exists and the user has "
        "video permissions.", attempted)
    return None, None


def _position_vector_cm(detection):
    xyz = detection["xyz_cm"]
    return np.array([xyz["x"], xyz["y"], xyz["z"]], dtype=float)


def _buffer_mean_position_cm(confirm_buffer):
    positions = np.array([_position_vector_cm(d) for _, d in confirm_buffer],
                         dtype=float)
    return np.mean(positions, axis=0)


def _select_detection_for_confirmation(detections, confirm_buffer,
                                       tolerance_cm=CONFIRM_POSITION_TOLERANCE_CM):
    """
    Keep the confirmation window attached to one physical tomato.
    Returns (selected_detection, same_target).
    """
    if not detections:
        return None, False

    if not confirm_buffer:
        return max(detections, key=lambda d: d["confidence"]), True

    reference = _buffer_mean_position_cm(confirm_buffer)
    selected = min(
        detections,
        key=lambda d: np.linalg.norm(_position_vector_cm(d) - reference),
    )
    distance_cm = np.linalg.norm(_position_vector_cm(selected) - reference)
    if distance_cm <= tolerance_cm:
        return selected, True

    return max(detections, key=lambda d: d["confidence"]), False


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
def run_harvesting(sim_mode=False, confirm_seconds=CONFIRM_SECONDS,
                   confirm_frames=CONFIRM_FRAMES,
                   camera_index=CAMERA_INDEX,
                   servo_backend=DEFAULT_SERVO_BACKEND,
                   uart_port=DEFAULT_UART_PORT,
                   baud=DEFAULT_BAUD):
    mode = "sim" if sim_mode else "real"
    logger.info(f"Starting harvesting system in {mode.upper()} mode")
    if not sim_mode:
        logger.info(
            "Hardware mode enabled: servo backend=%s, uart=%s @ %s",
            servo_backend, uart_port, baud)

    # Initialise subsystems
    detector = TomatoDetector()
    arm = FiveDOFArm()
    driver = ServoDriver(mode=mode, backend=servo_backend,
                         uart_port=uart_port, baud=baud)

    # Move to Search Home
    logger.info("Moving arm to Search Home position...")
    home_angles = search_home_angles()
    arm.set_joint_angles(home_angles)
    driver.go_search_home(duration_ms=1200)
    time.sleep(1.5)

    # Open camera before entering the harvest loop. Auto mode tries /dev/video*.
    cap, selected_camera = _open_camera(camera_index)
    if cap is None:
        driver.go_park(duration_ms=1000)
        driver.close()
        return

    # Detection confirmation buffer
    confirm_buffer = []  # list of (timestamp, detection_dict)
    confirm_started_at = None
    state = "SCANNING"   # SCANNING → CONFIRMING → ACTING → RETRACTING
    last_status_message = ""
    frame_grab_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_grab_failures += 1
                logger.warning(
                    "Frame grab failed (%s/%s) on camera index %s.",
                    frame_grab_failures, FRAME_GRAB_FAILURE_LIMIT,
                    selected_camera)
                if frame_grab_failures >= FRAME_GRAB_FAILURE_LIMIT:
                    logger.warning("Reopening camera after repeated frame failures.")
                    cap.release()
                    cap, selected_camera = _open_camera(camera_index)
                    confirm_buffer.clear()
                    confirm_started_at = None
                    state = "SCANNING"
                    frame_grab_failures = 0
                    if cap is None:
                        break
                continue
            frame_grab_failures = 0

            if state in ("SCANNING", "CONFIRMING"):
                detections = detector.detect(frame)
                annotated = detector.annotate(frame, detections)
                now = time.time()

                # Purge old entries outside the confirm window before matching.
                confirm_buffer = [(t, d) for t, d in confirm_buffer
                                  if now - t <= confirm_seconds]

                if detections:
                    best, same_target = _select_detection_for_confirmation(
                        detections, confirm_buffer)

                    if same_target:
                        if not confirm_buffer:
                            confirm_started_at = now
                        confirm_buffer.append((now, best))
                    else:
                        logger.info("Tomato target changed; restarting confirmation.")
                        confirm_buffer = [(now, best)]
                        confirm_started_at = now

                    if state == "SCANNING":
                        state = "CONFIRMING"
                        if confirm_started_at is None:
                            confirm_started_at = now
                        logger.info(f"Tomato spotted! Confirming for {confirm_seconds}s...")

                    confirm_elapsed = (
                        now - confirm_started_at
                        if confirm_started_at is not None else 0.0)

                    # Check if we have enough consistent detections
                    if (len(confirm_buffer) >= confirm_frames and
                            confirm_elapsed >= confirm_seconds):
                        # Average the confirmed position
                        avg_x, avg_y, avg_z = _buffer_mean_position_cm(confirm_buffer)
                        locked_xyz_cm = {"x": avg_x, "y": avg_y, "z": avg_z}

                        logger.info(f"LOCKED tomato at (cm): "
                                    f"X={avg_x:.1f} Y={avg_y:.1f} Z={avg_z:.1f}")

                        state = "ACTING"
                        confirm_buffer.clear()
                        confirm_started_at = None

                        # ── Execute harvest sequence ───────────────────────────
                        harvest_ok, harvest_message = _execute_harvest(
                            arm, driver, locked_xyz_cm, annotated)
                        last_status_message = harvest_message
                        if harvest_ok:
                            logger.info(f"Harvest result: {harvest_message}")
                        else:
                            logger.warning(f"Harvest skipped: {harvest_message}")

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
                                      if now - t <= confirm_seconds]
                    if not confirm_buffer:
                        state = "SCANNING"
                        confirm_started_at = None

                # Show status on annotated frame
                status_text = (
                    f"Mode: {mode.upper()}  Camera: {selected_camera}  "
                    f"State: {state}")
                if state == "CONFIRMING":
                    elapsed = (
                        now - confirm_started_at
                        if confirm_started_at is not None else 0.0)
                    status_text += f"  ({elapsed:.1f}/{confirm_seconds}s, " \
                                   f"{len(confirm_buffer)} frames)"
                if last_status_message:
                    status_text += f" | {last_status_message[:60]}"
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
    # Convert camera-frame coords to arm-base frame
    target_m = camera_to_arm_frame(locked_xyz_cm)
    logger.info(f"Arm-frame target (m): {target_m}")

    # Compute cut point (1 cm above tomato surface along stem / +Z)
    edge_pt, cut_pt = compute_cut_point(
        target_m, TOMATO_RADIUS_M, arm.base_pos,
        stem_direction=STEM_DIRECTION_ARM_FRAME)
    logger.info(f"Cut point (m): {cut_pt}")

    # Check reachability on the actual cutter target, not just the tomato centre.
    if not arm.is_reachable(cut_pt):
        reach = arm.max_reach() if hasattr(arm, "max_reach") else None
        reason = (
            f"cut point out of reach: cut={np.round(cut_pt, 3).tolist()}m"
            + (f", max reach={reach:.3f}m" if reach is not None else ""))
        logger.warning(reason)
        return False, reason

    # Save current (search-home) joint config
    q_home = arm.joint_angles.copy()

    # Open gripper before approach
    driver.gripper_open(duration_ms=400)
    time.sleep(0.5)

    # Solve IK for the cut point
    solved, err, iters = arm.inverse_kinematics(
        cut_pt, max_iters=500, tol=IK_TOLERANCE_M)
    q_cut = arm.joint_angles.copy()
    logger.info(f"IK solved={solved}, error={err:.4f}m, iters={iters}")

    if not solved or err > IK_MAX_ERROR_M:
        reason = (
            f"IK failed for cut point: solved={solved}, error={err:.4f}m, "
            f"limit={IK_MAX_ERROR_M:.4f}m, iterations={iters}")
        logger.warning(reason)
        arm.set_joint_angles(q_home)
        return False, reason

    # Move arm along interpolated trajectory: home → cut
    target_pulses = angles_to_pulses(q_cut)
    target_pulses[1] = GRIPPER_OPEN_PULSE
    logger.info(f"Moving all 5 servos toward cut point: pulses={target_pulses}")
    traj = interpolate_trajectory(q_home, q_cut, steps=25)
    for q in traj:
        arm.set_joint_angles(q)
        pulses = angles_to_pulses(q)
        # The fifth servo is the gripper, so keep it open while still sending a
        # synchronized command for every servo on each approach step.
        pulses[1] = GRIPPER_OPEN_PULSE
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
    return True, "cut complete"


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tomato Harvesting Robot — Integrated System")
    parser.add_argument("--sim", action="store_true",
                        help="Run in simulation mode (no real servos)")
    parser.add_argument("--hardware", action="store_true",
                        help="Run real hardware mode (default; sends UART servo commands)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch the 3D simulation GUI")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Camera index, or -1 to auto-detect (default: -1)")
    parser.add_argument("--confirm-seconds", type=float, default=CONFIRM_SECONDS,
                        help=f"Seconds to confirm a tomato before moving (default: {CONFIRM_SECONDS})")
    parser.add_argument("--confirm-frames", type=int, default=CONFIRM_FRAMES,
                        help=f"Detection frames required before moving (default: {CONFIRM_FRAMES})")
    parser.add_argument("--servo-backend",
                        choices=("auto", "sdk", "uart"),
                        default=DEFAULT_SERVO_BACKEND,
                        help="Real servo backend: auto tries Hiwonder SDK before raw UART")
    parser.add_argument("--uart-port", default=DEFAULT_UART_PORT,
                        help=f"Raw UART serial port (default: {DEFAULT_UART_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Raw UART baud rate (default: {DEFAULT_BAUD})")
    args = parser.parse_args()
    if args.hardware and args.sim:
        parser.error("--hardware and --sim cannot be used together")

    if args.gui:
        from simulation_gui import launch_gui
        launch_gui()
    else:
        run_harvesting(sim_mode=args.sim,
                       confirm_seconds=args.confirm_seconds,
                       confirm_frames=args.confirm_frames,
                       camera_index=args.camera,
                       servo_backend=args.servo_backend,
                       uart_port=args.uart_port,
                       baud=args.baud)


if __name__ == "__main__":
    main()
