"""
Integrated Tomato-Harvesting System — main.py
===============================================
Unifies the vision subsystem (YOLOv8 + filters) with a generic 5-DOF
robotic arm for autonomous ripe-tomato detection, approach, stem-cut,
and collection.

3-Tomato Batch Harvest Workflow:
  1. ARM  → Search Home (camera faces outward)
  2. SCAN → Snapshot scan for ≤5 s — detect all ripe tomatoes
  3. RANK → Sort by distance, pick up to 3 nearest
  4. For each tomato (nearest → farthest):
     a. Solve IK for stem cut point
     b. Move arm directly to cut pose (no return home between cuts)
     c. Close scissors to cut stem
     d. Open scissors, proceed to next tomato
  5. RETURN → Retract to default (search home) position
  6. Repeat from step 1

Servo control:  direct I2C via smbus2 to the PWM servo controller at
                0x34 on bus 1.  No vendor-specific SDK required.

Usage:
  python main.py                  # real hardware (5-DOF arm + camera)
  python main.py --sim            # simulation mode (no real servos)

Hand-in-Eye: camera is mounted between wrist (ID3) and gripper (ID1).
"""

import argparse
import logging
import sys
import time
import os
import cv2
import numpy as np

from vision import TomatoDetector, ARM_REACH_CM
from arm_controller import (
    FiveDOFArm, angles_to_pulses,
    compute_stem_cut_point, SERVO_IDS,
)
from arm_home_position import get_home_pulses, get_home_angles
from servo_driver import ServoDriver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global Safety Flags ────────────────────────────────────────────────────────
DRY_RUN = False        # If True, log all moves but don't send servo commands
NO_CUT = False         # If True, skip the actual gripper close (cutting) action
NO_CONFIRM = False     # If True, skip confirmation prompts
SINGLE_PASS = False    # If True, run only one cycle then exit
TEST_CYCLES = None     # If set, limit total cycles to this number

# ── Configuration ──────────────────────────────────────────────────────────────
SCAN_DURATION_S      = 3.0    # seconds to scan for tomatoes
SCAN_ANALYSIS_S      = 1.2    # seconds for heavy authenticity scan pass
CONFIRM_DELAY_S      = 3.0    # seconds camera must confirm before arm moves
MAX_HARVEST_PER_CYCLE = 3     # max tomatoes per harvest cycle
MIN_SIGHTINGS        = 2      # min frames a tomato must appear in to be confirmed
MOVE_DURATION_MS     = 800    # servo move duration for trajectory steps
GRIPPER_DELAY_S      = 0.6    # wait after gripper close (cutting)
RETRACT_DELAY_S      = 1.0    # wait after returning home
TOMATO_RADIUS_M      = 0.035  # average tomato radius in metres
CAMERA_INDEX         = 0      # default camera index
TRAJ_STEPS_APPROACH  = 25     # interpolation steps for approach
TRAJ_STEPS_RETRACT   = 30     # interpolation steps for retract to home
LIVE_DETECT_EVERY_N  = 3      # run live overlay inference every N frames
FRAME_DRAIN_COUNT    = 2      # flush stale buffered frames each iteration
MIN_CLUSTER_FRAMES   = 2      # drop one-frame fluke targets
ARM_MAX_REACH_FROM_CAMERA_M = 0.36  # hard physical limit from camera lens

# ── Camera-to-arm transform ───────────────────────────────────────────────────
# Hand-in-Eye: at Search Home the camera has a known pose relative to
# the arm base.  CALIBRATE THESE VALUES on the real robot.
# Arm reach is 36.3cm — camera is between wrist (ID3) and gripper (ID1),
# roughly 8cm forward from base and 12cm above when at Search Home.
CAMERA_OFFSET_M = np.array([0.08, 0.0, 0.12], dtype=float)
CAMERA_AXIS_SCALE = np.array([
    float(os.getenv("CAMERA_SCALE_X", "1.0")),
    float(os.getenv("CAMERA_SCALE_Y", "1.0")),
    float(os.getenv("CAMERA_SCALE_Z", "1.0")),
], dtype=float)


def _rotation_matrix_xyz(roll_deg, pitch_deg, yaw_deg):
    rx, ry, rz = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m


CAM_TO_ARM_FINE_ROT = _rotation_matrix_xyz(
    float(os.getenv("CAM_ROLL_DEG", "0.0")),
    float(os.getenv("CAM_PITCH_DEG", "0.0")),
    float(os.getenv("CAM_YAW_DEG", "0.0")),
)


def camera_to_arm_frame(xyz_cm_dict):
    """Convert vision xyz_cm (camera frame) → arm base frame (metres)."""
    cam = np.array([xyz_cm_dict["x"], xyz_cm_dict["y"], xyz_cm_dict["z"]],
                   dtype=float) / 100.0
    cam = cam * CAMERA_AXIS_SCALE
    # Base mapping: Camera Z ≈ arm X, camera X ≈ arm -Y, camera Y ≈ arm -Z
    arm_nominal = np.array([cam[2], -cam[0], -cam[1]], dtype=float)
    arm_rot = CAM_TO_ARM_FINE_ROT @ arm_nominal
    return arm_rot + CAMERA_OFFSET_M


def camera_distance_m(xyz_cm_dict):
    """Euclidean distance from camera lens to target point (metres)."""
    x_m = float(xyz_cm_dict["x"]) / 100.0
    y_m = float(xyz_cm_dict["y"]) / 100.0
    z_m = float(xyz_cm_dict["z"]) / 100.0
    return float(np.linalg.norm([x_m, y_m, z_m]))


def _project_to_reachable(target_m, arm, safety_margin_m=0.02):
    """Project a target to the nearest reachable point if needed."""
    target = np.array(target_m, dtype=float)
    base = np.array(arm.base_pos, dtype=float)
    vec = target - base
    dist = float(np.linalg.norm(vec))
    max_r = max(0.05, float(arm.max_reach()) - float(safety_margin_m))

    if dist <= max_r:
        return target, False

    if dist < 1e-9:
        return target, False

    projected = base + (vec / dist) * max_r
    return projected, True


# ── Trajectory interpolation ──────────────────────────────────────────────────
def interpolate_trajectory(q_start, q_end, steps=20):
    """Linear joint-space interpolation."""
    traj = []
    for s in range(steps + 1):
        alpha = s / steps
        q = (1 - alpha) * np.array(q_start) + alpha * np.array(q_end)
        traj.append(q)
    return traj


# ── Execute trajectory on hardware ────────────────────────────────────────────
def execute_trajectory(arm, driver, q_start, q_end, steps, duration_ms):
    """Smoothly move the arm from q_start to q_end via interpolation."""
    q_start = np.array(q_start, dtype=float)
    q_end = np.array(q_end, dtype=float)

    # I2C relay boards often ignore very short burst commands; keep timing conservative.
    backend = getattr(driver, "_backend", "")
    if backend.startswith("i2c"):
        steps = max(4, min(int(steps), 10))
        min_step_ms = 120
    else:
        steps = max(4, int(steps))
        min_step_ms = 40

    traj = interpolate_trajectory(q_start, q_end, steps)
    step_duration = max(min_step_ms, duration_ms // steps)
    step_sleep = step_duration / 1000.0

    start_pulses = angles_to_pulses(q_start)
    end_pulses = angles_to_pulses(q_end)
    max_delta = max(abs(end_pulses[sid] - start_pulses[sid]) for sid in end_pulses)
    logger.info(
        f"Trajectory plan: backend={backend}, steps={steps}, step_ms={step_duration}, "
        f"max_pulse_delta={max_delta}"
    )
    
    if DRY_RUN:
        q_str = ", ".join([f"{q:.2f}" for q in q_end])
        _log_dry_run(f"Trajectory: {steps} steps to angles [{q_str}] in {duration_ms}ms")
        time.sleep(0.2)  # Simulate brief delay
        return
    
    for q in traj:
        arm.set_joint_angles(q)
        pulses = angles_to_pulses(q)
        driver.move_servos(pulses, duration_ms=step_duration)
        time.sleep(step_sleep)

    # Final absolute settle command ensures the last pose is reached on relay boards.
    arm.set_joint_angles(q_end)
    driver.move_servos(end_pulses, duration_ms=max(200, step_duration * 2))
    time.sleep(max(0.2, step_sleep))


# ── Single tomato harvest action ──────────────────────────────────────────────
def harvest_single_tomato(arm, driver, tomato_xyz_cm, index, total):
    """
    Move arm from its current pose to the tomato's stem, cut, and leave
    the arm at the cut pose (caller decides whether to retract or continue).

    Returns True if successfully cut, False if skipped.
    """
    dist_cam_m = camera_distance_m(tomato_xyz_cm)
    if dist_cam_m > ARM_MAX_REACH_FROM_CAMERA_M:
        logger.warning(
            f"[{index+1}/{total}] Target at {dist_cam_m*100.0:.1f}cm from lens "
            f"is beyond 36cm reach. Cannot harvest. Move closer."
        )
        return False

    target_m = camera_to_arm_frame(tomato_xyz_cm)
    logger.info(f"[{index+1}/{total}] Arm-frame target: "
                f"[{target_m[0]:.3f}, {target_m[1]:.3f}, {target_m[2]:.3f}]m")

    # If calibration drift pushes target just beyond reach, project to a safe
    # reachable point so the arm still executes a meaningful motion.
    target_m, was_projected = _project_to_reachable(target_m, arm)
    if was_projected:
        logger.warning(
            f"[{index+1}/{total}] Target outside arm reach; projecting to "
            f"[{target_m[0]:.3f}, {target_m[1]:.3f}, {target_m[2]:.3f}]m"
        )

    # Compute stem cut point (above the tomato)
    stem_pt, cut_pt = compute_stem_cut_point(target_m, TOMATO_RADIUS_M,
                                              arm.base_pos)
    cut_pt, cut_projected = _project_to_reachable(cut_pt, arm)
    if cut_projected:
        logger.warning(
            f"[{index+1}/{total}] Cut point adjusted to reachable point: "
            f"[{cut_pt[0]:.3f}, {cut_pt[1]:.3f}, {cut_pt[2]:.3f}]m"
        )
    logger.info(f"[{index+1}/{total}] Stem cut point: "
                f"[{cut_pt[0]:.3f}, {cut_pt[1]:.3f}, {cut_pt[2]:.3f}]m")

    # Save current pose
    q_current = arm.joint_angles.copy()

    # Open gripper before approach
    if DRY_RUN:
        _log_dry_run(f"[{index+1}/{total}] Opening gripper (400ms)")
    else:
        driver.gripper_open(duration_ms=400)
    time.sleep(0.4)

    # Solve IK for cut point
    arm.set_joint_angles(q_current)
    solved, err, iters = arm.inverse_kinematics(cut_pt, max_iters=400, tol=5e-4)
    q_cut = arm.joint_angles.copy()
    logger.info(f"[{index+1}/{total}] IK: solved={solved}, "
                f"err={err:.4f}m, iters={iters}")

    if not solved and err > 0.02:
        logger.warning(f"[{index+1}/{total}] IK error too large — skipping")
        arm.set_joint_angles(q_current)
        return False

    # Move: current position → cut point
    logger.info(f"[{index+1}/{total}] Approaching stem...")
    if DRY_RUN:
        _log_dry_run(f"[{index+1}/{total}] Moving to cut point")
    execute_trajectory(arm, driver, q_current, q_cut,
                       steps=TRAJ_STEPS_APPROACH, duration_ms=MOVE_DURATION_MS)

    # Settling delay
    time.sleep(0.3)

    # Close gripper / scissors to cut the stem
    logger.info(f"[{index+1}/{total}] {'[SKIPPED-DRY-RUN/NO-CUT]' if DRY_RUN or NO_CUT else 'CUTTING'} stem...")
    if not (DRY_RUN or NO_CUT):
        if _user_confirm(f"READY TO CUT STEM for tomato #{index+1}. Proceed?", 
                        default_yes=False):
            driver.gripper_close(duration_ms=500)
            time.sleep(GRIPPER_DELAY_S)
            logger.info(f"[{index+1}/{total}] Cut complete!")
            driver.gripper_open(duration_ms=400)
            time.sleep(0.3)
        else:
            logger.warning(f"[{index+1}/{total}] Cut cancelled by user — skipping")
            return False
    else:
        if DRY_RUN:
            _log_dry_run(f"[{index+1}/{total}] Closing gripper (500ms)")
            time.sleep(0.5)
            _log_dry_run(f"[{index+1}/{total}] Opening gripper (400ms)")
        time.sleep(0.3)

    return True


def _open_low_latency_camera(camera_index):
    """Open camera with low-latency settings suitable for Raspberry Pi 5."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ── Safety utilities ──────────────────────────────────────────────────────────
def _user_confirm(prompt, default_yes=False):
    """Prompt user for confirmation. Returns True if confirmed."""
    if NO_CONFIRM:
        return True
    default_str = "[Y/n]" if default_yes else "[y/N]"
    response = input(f"{prompt} {default_str}: ").strip().lower()
    if default_yes:
        return response != "n"
    else:
        return response == "y"


def _log_dry_run(message):
    """Log message with [DRY RUN] prefix if in dry-run mode."""
    if DRY_RUN:
        logger.info(f"[DRY RUN] {message}")


def _drain_camera_frames(cap, count=FRAME_DRAIN_COUNT):
    for _ in range(max(0, count)):
        cap.grab()


def _show_status_banner(message, color=(0, 0, 255), hold_s=0.8):
    frame = np.zeros((180, 960, 3), dtype=np.uint8)
    cv2.putText(frame, message, (20, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, color, 2)
    cv2.imshow("Tomato Harvester", frame)
    cv2.waitKey(1)
    time.sleep(hold_s)


def _collect_ranked_targets(detector, cap, window_s):
    """Collect stable targets over a confirm window and return nearest-first."""
    clusters = []
    deadline = time.time() + window_s

    while time.time() < deadline:
        _drain_camera_frames(cap)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        detections = detector.detect(frame, use_tracker=False)
        annotated = detector.annotate(frame, detections)
        cv2.putText(annotated, "CONFIRMING TARGETS...", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Tomato Harvester", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        for det in detections:
            cx, cy = det["center_px"]
            xyz = det["xyz_cm"]
            conf = float(det["confidence"])
            matched = None
            best_d = 1e9
            for cl in clusters:
                dx = cx - cl["cx"]
                dy = cy - cl["cy"]
                d = float(np.hypot(dx, dy))
                if d < 45.0 and d < best_d:
                    best_d = d
                    matched = cl

            if matched is None:
                clusters.append({
                    "cx": float(cx),
                    "cy": float(cy),
                    "sum_x": float(xyz["x"]),
                    "sum_y": float(xyz["y"]),
                    "sum_z": float(xyz["z"]),
                    "count": 1,
                    "best_conf": conf,
                    "bbox_px": det.get("bbox_px"),
                })
            else:
                n = matched["count"]
                matched["cx"] = (matched["cx"] * n + cx) / (n + 1)
                matched["cy"] = (matched["cy"] * n + cy) / (n + 1)
                matched["sum_x"] += float(xyz["x"])
                matched["sum_y"] += float(xyz["y"])
                matched["sum_z"] += float(xyz["z"])
                matched["count"] += 1
                matched["best_conf"] = max(matched["best_conf"], conf)
                matched["bbox_px"] = det.get("bbox_px", matched["bbox_px"])

    stable = [cl for cl in clusters if cl["count"] >= MIN_CLUSTER_FRAMES]
    if not stable:
        return []

    ranked = []
    for cl in stable:
        n = float(cl["count"])
        xyz = {
            "x": cl["sum_x"] / n,
            "y": cl["sum_y"] / n,
            "z": cl["sum_z"] / n,
        }
        dist = float(np.linalg.norm([xyz["x"], xyz["y"], xyz["z"]]))
        ranked.append({
            "xyz_cm": xyz,
            "confidence": round(cl["best_conf"], 3),
            "bbox_px": cl.get("bbox_px"),
            "distance_cm": round(dist, 2),
            "sightings": int(cl["count"]),
            "reachable": bool(dist <= ARM_REACH_CM),
        })

    ranked.sort(key=lambda d: d["distance_cm"])
    return ranked[:MAX_HARVEST_PER_CYCLE]


def _harvest_sequence(arm, driver, tomatoes):
    harvested = 0
    n = len(tomatoes)
    for i, tomato in enumerate(tomatoes):
        dist = float(tomato.get("distance_cm", 1e9))
        if dist > ARM_REACH_CM:
            msg = f"Tomato #{i+1}: OUT OF REACH ({dist:.1f}cm) - move closer"
            logger.warning(msg)
            _show_status_banner(msg, color=(0, 0, 255), hold_s=1.0)
            continue

        logger.info(f"\n--- Harvesting tomato #{i+1}/{n} (distance: {dist:.1f}cm) ---")
        success = harvest_single_tomato(arm, driver, tomato["xyz_cm"], i, n)
        if success:
            harvested += 1
            logger.info(f"Tomato #{i+1} harvested successfully!")
        else:
            logger.info(f"Tomato #{i+1} skipped.")
        time.sleep(0.2)
    return harvested


# ── Main harvesting loop ──────────────────────────────────────────────────────
def run_harvesting(sim_mode=False):
    """Main loop: scan → detect 3 nearest tomatoes → harvest each → return home."""
    mode = "sim" if sim_mode else "real"
    logger.info(f"Starting 3-tomato batch harvest system in {mode.upper()} mode")
    if DRY_RUN:
        logger.warning("████████████████████████████████████████████████████████")
        logger.warning("                 ⚠️  DRY RUN MODE ACTIVE ⚠️")
        logger.warning("  All movements will be LOGGED NOT EXECUTED on hardware")
        logger.warning("████████████████████████████████████████████████████████")
    if NO_CUT:
        logger.warning("NO-CUT MODE ACTIVE — Gripper will not cut tomatoes")
    if SINGLE_PASS:
        logger.info("SINGLE PASS MODE — Will run 1 cycle then exit")
    if TEST_CYCLES:
        logger.info(f"TEST MODE — Limiting to {TEST_CYCLES} cycle(s)")
    logger.info(f"3D mapping: CAMERA_OFFSET_M={CAMERA_OFFSET_M.tolist()}, "
                f"CAMERA_AXIS_SCALE={CAMERA_AXIS_SCALE.tolist()}")

    # Initialise subsystems
    detector = TomatoDetector(imgsz=416)
    arm = FiveDOFArm()
    driver = ServoDriver(mode=mode)

    # Move to exact saved home base
    home_pulses = get_home_pulses()
    home_angles = get_home_angles()
    logger.info("Moving arm to home base position...")
    if not DRY_RUN:
        arm.set_joint_angles(home_angles)
        driver.move_servos(home_pulses, duration_ms=2000)
    else:
        _log_dry_run(f"Setting arm to home: [{', '.join([f'{a:.2f}' for a in home_angles])}]")
    time.sleep(2.5)

    # Open low-latency camera
    cap = _open_low_latency_camera(CAMERA_INDEX)

    cycle = 0
    try:
        while True:
            cycle += 1
            
            # Check cycle limits
            if TEST_CYCLES and cycle > TEST_CYCLES:
                logger.info(f"Reached test cycle limit ({TEST_CYCLES}). Exiting.")
                break
            
            logger.info("=" * 60)
            logger.info(f"CYCLE {cycle}:  SCANNING for ripe tomatoes "
                        f"({SCAN_DURATION_S}s window)...")

            # ── Phase 1: SCAN ──────────────────────────────────────────────
            # Ensure arm is at saved home base so camera can see the workspace
            if not DRY_RUN:
                arm.set_joint_angles(home_angles)
                driver.move_servos(home_pulses, duration_ms=800)
            else:
                _log_dry_run("Resetting arm to home base for scan")
            time.sleep(1.0)

            # Live camera window during scan
            logger.info(f"Scanning for {SCAN_DURATION_S}s (live feed active)...")
            scan_end = time.time() + SCAN_DURATION_S
            while time.time() < scan_end:
                _drain_camera_frames(cap)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                dets = detector.detect_fast(frame)
                annotated = detector.annotate(frame, dets)
                cv2.putText(annotated, "SCANNING...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow("Tomato Harvester", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # Confirm and cluster stable targets over the confirm window.
            tomatoes = _collect_ranked_targets(detector, cap, CONFIRM_DELAY_S)

            if not tomatoes:
                logger.info("No stable tomatoes detected. Waiting 2s and rescanning...")
                time.sleep(2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            n = len(tomatoes)
            logger.info(f"Detected {n} stable tomato(es), nearest-first.")
            for i, t in enumerate(tomatoes):
                logger.info(
                    f"  #{i+1}: distance={t['distance_cm']:.1f}cm, "
                    f"reachable={t.get('reachable')}, conf={t['confidence']:.2f}, "
                    f"sightings={t['sightings']}"
                )

            harvested = _harvest_sequence(arm, driver, tomatoes)

            # ── Phase 3: RETURN to saved home base ────────────────────────
            logger.info(f"\n{harvested}/{n} tomatoes harvested. "
                        f"Returning to home base...")
            if not DRY_RUN:
                q_current = arm.joint_angles.copy()
                execute_trajectory(arm, driver, q_current, home_angles,
                                   steps=TRAJ_STEPS_RETRACT,
                                   duration_ms=MOVE_DURATION_MS)
                arm.set_joint_angles(home_angles)
                driver.move_servos(home_pulses, duration_ms=800)
            else:
                _log_dry_run("Returning arm to home base")
            time.sleep(RETRACT_DELAY_S)

            logger.info(f"Cycle {cycle} complete. Ready for next scan.\n")
            
            if SINGLE_PASS:
                logger.info("Single-pass mode: exiting after 1 cycle")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Safe shutdown: park all servos
        logger.info("Parking arm (safe shutdown)...")
        if not DRY_RUN:
            driver.go_park(duration_ms=1200)
            time.sleep(1.5)
        else:
            _log_dry_run("Parking arm")
        driver.close()
        logger.info("System shut down safely.")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tomato Harvesting Robot — 5-DOF Arm + Vision System")
    parser.add_argument("--sim", action="store_true",
                        help="Run in simulation mode (no real servos)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log all moves but don't send servo commands (safe preview mode)")
    parser.add_argument("--no-cut", action="store_true",
                        help="Skip actual cutting (gripper won't close)")
    parser.add_argument("--single-pass", action="store_true",
                        help="Run exactly 1 harvest cycle then exit")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip all confirmation prompts (auto-confirm)")
    parser.add_argument("--test-cycles", type=int, default=None,
                        help="Limit execution to N cycles (for testing)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--from-file", action="store_true",
                        help="Read detected tomato positions from detected_tomatoes.json and move arm in real time")
    args = parser.parse_args()

    # Set global safety flags
    global DRY_RUN, NO_CUT, NO_CONFIRM, SINGLE_PASS, TEST_CYCLES, CAMERA_INDEX
    DRY_RUN = args.dry_run
    NO_CUT = args.no_cut
    NO_CONFIRM = args.no_confirm
    SINGLE_PASS = args.single_pass
    TEST_CYCLES = args.test_cycles
    CAMERA_INDEX = args.camera

    if args.from_file:
        import json, time, os
        from arm_home_position import get_home_pulses as _ghp, get_home_angles as _gha
        
        if DRY_RUN:
            logger.warning("████████████████████████████████████████████████████████")
            logger.warning("                 ⚠️  DRY RUN MODE ACTIVE ⚠️")
            logger.warning("  All movements will be LOGGED NOT EXECUTED on hardware")
            logger.warning("████████████████████████████████████████████████████████")
        if NO_CUT:
            logger.warning("NO-CUT MODE ACTIVE — Gripper will not cut tomatoes")
        
        arm = FiveDOFArm()
        driver = ServoDriver(mode="sim" if args.sim else "real")
        home_angles = _gha()
        home_pulses = _ghp()
        
        if not DRY_RUN:
            arm.set_joint_angles(home_angles)
            driver.move_servos(home_pulses, duration_ms=2000)
        else:
            _log_dry_run(f"Setting arm to home: [{', '.join([f'{a:.2f}' for a in home_angles])}]")
        time.sleep(2.5)
        
        logger.info("[INFO] Waiting for detected_tomatoes.json updates...")
        last_payload = None
        while True:
            try:
                if not os.path.exists("detected_tomatoes.json"):
                    time.sleep(0.5)
                    continue
                with open("detected_tomatoes.json", "r") as f:
                    payload = json.load(f)
                if payload == last_payload or payload.get("count", 0) == 0:
                    time.sleep(0.5)
                    continue
                last_payload = payload
                tomatoes = payload["ripe_tomatoes"]
                tomatoes.sort(
                    key=lambda t: float(t.get("distance_cm", 1e9))
                )
                tomatoes = tomatoes[:MAX_HARVEST_PER_CYCLE]
                logger.info(f"[INFO] Moving to {len(tomatoes)} nearest detected tomato(es)...")
                for i, tomato in enumerate(tomatoes):
                    xyz_cm = tomato["xyz_cm"]
                    logger.info(f"[INFO] Moving arm to tomato #{i+1} at {xyz_cm}")
                    dist_cam_m = camera_distance_m(xyz_cm)
                    if dist_cam_m > ARM_MAX_REACH_FROM_CAMERA_M:
                        logger.warning(
                            "[WARN] Target is "
                            f"{dist_cam_m*100.0:.1f}cm from lens (>36cm). "
                            "Cannot harvest. Move closer."
                        )
                        continue
                    target_m = camera_to_arm_frame(xyz_cm)
                    if not arm.is_reachable(target_m):
                        logger.warning(f"[WARN] Target {target_m} out of reach, skipping.")
                        continue
                    stem_pt, cut_pt = compute_stem_cut_point(target_m, TOMATO_RADIUS_M, arm.base_pos)
                    q_current = arm.joint_angles.copy()
                    if not DRY_RUN:
                        arm.set_joint_angles(q_current)
                    solved, err, iters = arm.inverse_kinematics(cut_pt, max_iters=400, tol=5e-4)
                    q_cut = arm.joint_angles.copy()
                    if not solved and err > 0.02:
                        logger.warning(f"[WARN] IK error too large, skipping.")
                        if not DRY_RUN:
                            arm.set_joint_angles(q_current)
                        continue
                    if DRY_RUN:
                        _log_dry_run(f"[INFO] Tomato #{i+1}: Moving to cut point")
                    else:
                        execute_trajectory(arm, driver, q_current, q_cut, 
                                         steps=TRAJ_STEPS_APPROACH, duration_ms=MOVE_DURATION_MS)
                    time.sleep(0.3)
                    
                    if not (DRY_RUN or NO_CUT):
                        logger.info(f"[INFO] CUTTING stem of tomato #{i+1}...")
                        driver.gripper_close(duration_ms=500)
                        time.sleep(GRIPPER_DELAY_S)
                        driver.gripper_open(duration_ms=400)
                    else:
                        if DRY_RUN:
                            _log_dry_run(f"[INFO] Tomato #{i+1}: Closing gripper (500ms)")
                            time.sleep(0.5)
                            _log_dry_run(f"[INFO] Tomato #{i+1}: Opening gripper (400ms)")
                        elif NO_CUT:
                            logger.info(f"[INFO] Tomato #{i+1}: [NO-CUT MODE] Skipping cut")
                    
                    time.sleep(0.3)
                    logger.info(f"[INFO] Tomato #{i+1} processed.")
                
                logger.info("[INFO] Returning to home position...")
                if not DRY_RUN:
                    q_current = arm.joint_angles.copy()
                    execute_trajectory(arm, driver, q_current, home_angles, 
                                     steps=TRAJ_STEPS_RETRACT, duration_ms=MOVE_DURATION_MS)
                    arm.set_joint_angles(home_angles)
                    driver.move_servos(home_pulses, duration_ms=800)
                else:
                    _log_dry_run("Returning arm to home position")
                time.sleep(RETRACT_DELAY_S)
            except KeyboardInterrupt:
                logger.info("[INFO] Interrupted by user.")
                break
            except Exception as e:
                logger.error(f"[ERROR] {e}")
                time.sleep(1)
        
        if not DRY_RUN:
            driver.go_park(duration_ms=1200)
            time.sleep(1.5)
        else:
            _log_dry_run("Parking arm")
        driver.close()
        logger.info("[INFO] System shut down safely.")
    else:
        run_harvesting(sim_mode=args.sim)


if __name__ == "__main__":
    main()
