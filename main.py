"""
Integrated Tomato-Harvesting System — main.py
===============================================
Unifies the vision subsystem (YOLOv8 + filters) with the 5-DOF robotic arm
subsystem for autonomous ripe-tomato detection, approach, cut, and collection.

Workflow:
  1. ARM  → Search Home (camera faces outward)
  2. VISION → Continuous detection loop on live camera
  3. LOCK → When ripe tomatoes are detected consistently for CONFIRM_SECONDS,
             lock a ranked list of up to MAX_TOMATOES_PER_RUN (3) targets,
             sorted by 3-D distance (nearest first)
  4. ARM  → For each target in order:
              a. If distance > ARM_REACH_CM, show "out of reach — move closer"
                 and skip
              b. Otherwise: solve IK → approach → cut → release
  5. ARM  → Return to Search Home (home base)
  6. Repeat

Usage:
  python main.py                  # real hardware (RPi5 + Hiwonder servos)
  python main.py --sim            # simulation mode (no servos, log only)
  python main.py --gui            # launch 3D simulation GUI instead

Hand-in-Eye: camera is mounted between wrist (ID3) and gripper (ID1).
The arm must be at Search Home for the camera to see the workspace.

Performance notes (Raspberry Pi 5, 8GB):
  - Capture at 640x480 with `CAP_PROP_BUFFERSIZE=1` and MJPG FOURCC to keep
    the grab queue short and avoid multi-hundred-ms latency.
  - Run YOLOv8 inference at imgsz=416 (set on TomatoDetector).  This is the
    single biggest lever for live-camera FPS on the Pi.
  - The capture/inference loop drains the OS buffer each iteration so we
    always work on the freshest frame.
"""

import argparse
import logging
import time
import cv2
import numpy as np

from vision import TomatoDetector, ARM_REACH_CM
from arm_controller import (
    FiveDOFArm, angles_to_pulses, search_home_angles,
    compute_cut_point,
)
from servo_driver import ServoDriver

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIRM_SECONDS       = 3.0    # seconds the scene must be stable before acting
CONFIRM_FRAMES        = 6      # minimum detections within the confirm window
MAX_TOMATOES_PER_RUN  = 3      # harvest up to this many in a single sweep
MOVE_DURATION_MS      = 800    # servo move duration for arm motion
GRIPPER_DELAY_S       = 0.6    # wait after gripper command
RETRACT_DELAY_S       = 1.0    # wait after retract before next scan
TOMATO_RADIUS_M       = 0.035  # average tomato radius in metres
CAMERA_INDEX          = 0      # default camera

# Capture tuning for Raspberry Pi 5 — keep the pipeline latency low.
CAPTURE_WIDTH         = 640
CAPTURE_HEIGHT        = 480
CAPTURE_FOURCC        = "MJPG"
# On the Pi 5 V4L2 driver the capture queue holds up to ~2 frames even with
# CAP_PROP_BUFFERSIZE=1 (the property is advisory, not enforced), so we call
# grab() twice before read() to discard stale frames and keep end-to-end
# latency below ~60 ms.
FRAME_DRAIN_COUNT     = 2
# When a cluster of same-position detections survives this many frames within
# the confirm window, we trust it.  Anything seen fewer times is treated as a
# transient false positive and dropped.
MIN_CLUSTER_FRAMES    = 2

# ── Camera-to-arm transform ───────────────────────────────────────────────────
# Hand-in-Eye: the camera sits between the wrist (ID3) and the gripper (ID1).
# At Search Home the camera has a known pose relative to the arm base.
# Axis convention used here at Search Home:
#   camera +X (right in image)  →  arm -Y
#   camera +Y (down in image)   →  arm -Z
#   camera +Z (depth forward)   →  arm +X
# The offset below is the position of the camera optical centre, expressed
# in the arm-base frame, when the arm is at Search Home.
# CALIBRATE THESE on the real robot (e.g. by commanding the arm to a known
# pose and measuring the camera centre relative to the base).
CAMERA_OFFSET_M = np.array([0.15, 0.0, 0.18], dtype=float)


def camera_to_arm_frame(xyz_cm_dict):
    """Convert vision xyz_cm (camera frame) → arm base frame (metres)."""
    cam = np.array([xyz_cm_dict["x"], xyz_cm_dict["y"], xyz_cm_dict["z"]],
                   dtype=float) / 100.0
    arm_x = cam[2] + CAMERA_OFFSET_M[0]   # depth → forward
    arm_y = -cam[0] + CAMERA_OFFSET_M[1]  # left/right (mirror X)
    arm_z = -cam[1] + CAMERA_OFFSET_M[2]  # up/down    (mirror Y)
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


# ── Camera helpers ────────────────────────────────────────────────────────────
def _open_camera(index):
    """Open the camera with low-latency settings tuned for Raspberry Pi 5."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    # MJPG is dramatically faster than YUYV on most USB webcams / Pi cameras.
    try:
        fourcc = cv2.VideoWriter_fourcc(*CAPTURE_FOURCC)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    # Keep only the latest frame in the driver queue — otherwise we process
    # stale frames and the feed looks "laggy".
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def _grab_fresh_frame(cap):
    """Drain any buffered frames so we always process the newest one."""
    for _ in range(FRAME_DRAIN_COUNT):
        cap.grab()
    ret, frame = cap.read()
    return ret, frame


# ── Ranking + reach helpers ───────────────────────────────────────────────────
def _rank_targets(detections):
    """
    Sort detections by 3-D Euclidean distance from the camera (nearest first)
    and tag each with a 1-based rank.  Returns a new list.
    """
    ranked = sorted(detections, key=lambda d: d["distance_cm"])
    for i, d in enumerate(ranked, start=1):
        d["rank"] = i
    return ranked


def _select_harvest_targets(confirm_buffer):
    """
    Cluster the detections across the confirm window by pixel centre so that
    the same physical tomato, seen across multiple frames, collapses into one
    averaged target.  Returns a ranked list of up to MAX_TOMATOES_PER_RUN
    averaged targets.
    """
    # Flatten (every frame contributed a list of detections).
    all_dets = []
    for _, dets in confirm_buffer:
        all_dets.extend(dets)
    if not all_dets:
        return []

    # Simple greedy clustering on pixel centres. A tomato seen across the
    # confirm window will wobble at most a few pixels, so 60 px is generous.
    CLUSTER_RADIUS_PX = 60
    clusters = []   # each cluster: list of dets
    for d in all_dets:
        cx, cy = d["center_px"]
        placed = False
        for cl in clusters:
            # cluster centroid = mean of centres so far
            mxs = np.mean([x["center_px"][0] for x in cl])
            mys = np.mean([x["center_px"][1] for x in cl])
            if (cx - mxs) ** 2 + (cy - mys) ** 2 <= CLUSTER_RADIUS_PX ** 2:
                cl.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])

    # Require a cluster to be seen at least MIN_CLUSTER_FRAMES times — this
    # rejects single-frame flukes that slipped through the per-frame filters.
    # If nothing reaches the threshold, we intentionally return no targets
    # rather than acting on an unstable detection.
    clusters = [cl for cl in clusters if len(cl) >= MIN_CLUSTER_FRAMES]
    if not clusters:
        return []

    # Average each cluster into a single stable target.
    averaged = []
    for cl in clusters:
        xs = [d["xyz_cm"]["x"] for d in cl]
        ys = [d["xyz_cm"]["y"] for d in cl]
        zs = [d["xyz_cm"]["z"] for d in cl]
        dists = [d["distance_cm"] for d in cl]
        confs = [d["confidence"] for d in cl]
        # Use the most recent bbox for display.
        last = cl[-1]
        averaged.append({
            "confidence":  float(np.mean(confs)),
            "bbox_px":     last["bbox_px"],
            "center_px":   last["center_px"],
            "xyz_cm":      {
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "z": float(np.mean(zs)),
            },
            "distance_cm": float(np.mean(dists)),
            "reachable":   float(np.mean(dists)) <= ARM_REACH_CM,
            "frames_seen": len(cl),
        })

    ranked = _rank_targets(averaged)
    return ranked[:MAX_TOMATOES_PER_RUN]


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
    cap = _open_camera(CAMERA_INDEX)
    if cap is None:
        logger.error("Cannot open camera!")
        return
    logger.info(f"Camera opened: index={CAMERA_INDEX} "
                f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT} {CAPTURE_FOURCC}")

    # Detection confirmation buffer: [(timestamp, [detections in that frame])]
    confirm_buffer = []
    state = "SCANNING"   # SCANNING → CONFIRMING → ACTING → RETRACTING

    try:
        while True:
            ret, frame = _grab_fresh_frame(cap)
            if not ret:
                logger.warning("Frame grab failed")
                continue

            if state in ("SCANNING", "CONFIRMING"):
                detections = detector.detect(frame)
                # Rank & tag every frame so the overlay shows live numbering.
                detections = _rank_targets(detections)
                annotated = detector.annotate(frame, detections)
                now = time.time()

                if detections:
                    confirm_buffer.append((now, detections))
                    # Purge entries older than the confirm window.
                    confirm_buffer = [(t, d) for t, d in confirm_buffer
                                      if now - t <= CONFIRM_SECONDS]

                    if state == "SCANNING":
                        state = "CONFIRMING"
                        logger.info(f"Tomato(es) spotted! Confirming for "
                                    f"{CONFIRM_SECONDS}s...")

                    # Enough consistent frames? → lock & act.
                    if (len(confirm_buffer) >= CONFIRM_FRAMES and
                            now - confirm_buffer[0][0] >= CONFIRM_SECONDS):

                        targets = _select_harvest_targets(confirm_buffer)
                        logger.info(f"LOCKED {len(targets)} tomato(es) "
                                    f"(max {MAX_TOMATOES_PER_RUN}):")
                        for t in targets:
                            logger.info(
                                f"  #{t['rank']}  d={t['distance_cm']:.1f}cm "
                                f"xyz={t['xyz_cm']}  "
                                f"reachable={t['reachable']}")

                        state = "ACTING"
                        confirm_buffer.clear()

                        # ── Sequential harvest ───────────────────────────────
                        _harvest_sequence(arm, driver, home_angles, targets)

                        state = "RETRACTING"
                        logger.info("Returning to home base (Search Home)...")
                        arm.set_joint_angles(home_angles)
                        pulses = angles_to_pulses(home_angles)
                        driver.move_servos(pulses, duration_ms=1000)
                        time.sleep(RETRACT_DELAY_S)

                        state = "SCANNING"
                        logger.info("Ready for next sweep.\n")
                else:
                    # No detection this frame — decay buffer.
                    confirm_buffer = [(t, d) for t, d in confirm_buffer
                                      if now - t <= CONFIRM_SECONDS]
                    if not confirm_buffer:
                        state = "SCANNING"

                # Status overlay
                status_text = f"State: {state}"
                if state == "CONFIRMING" and confirm_buffer:
                    elapsed = now - confirm_buffer[0][0]
                    status_text += (f"  ({elapsed:.1f}/{CONFIRM_SECONDS}s, "
                                    f"{len(confirm_buffer)} frames)")
                cv2.putText(annotated, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # If anything was flagged out-of-reach this frame, tell the
                # operator at the top of the window (beyond the per-box label).
                oor = [d for d in detections if not d.get("reachable", True)]
                if oor:
                    cv2.putText(annotated,
                                f"{len(oor)} tomato(es) beyond {ARM_REACH_CM:.0f}cm"
                                f" — MOVE CLOSER",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

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


def _harvest_sequence(arm, driver, home_angles, targets):
    """
    Harvest up to MAX_TOMATOES_PER_RUN tomatoes in ranked order.
    Handles the 1-, 2- and 3-tomato cases transparently — we just iterate
    over whatever the vision pipeline gave us.
    """
    if not targets:
        logger.info("No tomatoes to harvest.")
        return

    for t in targets:
        rank = t.get("rank", "?")
        dist = t["distance_cm"]
        logger.info(f"--- Tomato #{rank}  d={dist:.1f}cm ---")

        # Reach check based on the physical arm length (36 cm).
        if dist > ARM_REACH_CM:
            logger.warning(
                f"Tomato #{rank} is {dist:.1f}cm away — beyond the "
                f"{ARM_REACH_CM:.0f}cm arm reach. Cannot harvest; "
                f"please move the robot closer.")
            continue

        _execute_harvest(arm, driver, t["xyz_cm"])

        # Small settling pause before moving to the next tomato.
        time.sleep(0.4)


def _execute_harvest(arm, driver, locked_xyz_cm):
    """Execute a single harvest: IK solve → approach → cut → release."""
    # Convert camera-frame coords to arm-base frame.
    target_m = camera_to_arm_frame(locked_xyz_cm)
    logger.info(f"Arm-frame target (m): {target_m}")

    # Kinematic reach check (independent of the 36cm camera-distance check).
    if not arm.is_reachable(target_m):
        logger.warning("Target unreachable by IK — skipping.")
        return

    # Compute cut point (1 cm from tomato surface toward the stem).
    _edge_pt, cut_pt = compute_cut_point(target_m, TOMATO_RADIUS_M,
                                         arm.base_pos)
    logger.info(f"Cut point (m): {cut_pt}")

    # Save current (search-home) joint config.
    q_home = arm.joint_angles.copy()

    # Open gripper before approach.
    driver.gripper_open(duration_ms=400)
    time.sleep(0.5)

    # Solve IK for the cut point.
    solved, err, iters = arm.inverse_kinematics(cut_pt, max_iters=400,
                                                tol=5e-4)
    q_cut = arm.joint_angles.copy()
    logger.info(f"IK solved={solved}, error={err:.4f}m, iters={iters}")

    if not solved and err > 0.02:
        logger.warning(f"IK error too large ({err:.4f}m) — aborting harvest.")
        arm.set_joint_angles(q_home)
        return

    # Move arm along interpolated trajectory: home → cut.
    traj = interpolate_trajectory(q_home, q_cut, steps=25)
    for q in traj:
        arm.set_joint_angles(q)
        pulses = angles_to_pulses(q)
        driver.move_servos(pulses, duration_ms=MOVE_DURATION_MS // 25)
        time.sleep(MOVE_DURATION_MS / 25000.0)

    # Small settling delay at cut position.
    time.sleep(0.3)

    # Close gripper / scissors to cut the stem.
    logger.info("CUTTING — closing gripper...")
    driver.gripper_close(duration_ms=500)
    time.sleep(GRIPPER_DELAY_S)
    logger.info("Cut complete.")

    # Open gripper to release (tomato falls into net below).
    driver.gripper_open(duration_ms=400)
    time.sleep(0.3)

    # Retract this single tomato back to Search Home before the next one,
    # so the arm always starts each approach from a known, safe pose.
    q_back = arm.joint_angles.copy()
    back_traj = interpolate_trajectory(q_back, q_home, steps=25)
    for q in back_traj:
        arm.set_joint_angles(q)
        pulses = angles_to_pulses(q)
        driver.move_servos(pulses, duration_ms=MOVE_DURATION_MS // 25)
        time.sleep(MOVE_DURATION_MS / 25000.0)


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
