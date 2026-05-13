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
import threading
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
CONFIRM_SECONDS    = 3.0     # seconds used for the reference validation snapshot
CONFIRM_FRAMES     = 3       # minimum detections within the confirm window
CONFIRM_POSITION_TOLERANCE_CM = 6.0  # restart confirmation if target jumps
MOVE_DURATION_MS   = 800     # servo move duration for arm motion
GUIDED_MOVE_DURATION_MS = 180 # short moves for live vision-guided approach
GRIPPER_DELAY_S    = 0.6     # wait after gripper command
RETRACT_DELAY_S    = 1.0     # wait after retract before next scan
TOMATO_RADIUS_M    = 0.035   # average tomato radius in metres
REFERENCE_TARGET_LIMIT = 3
VALIDATION_TRACK_TOLERANCE_CM = 5.0
VALIDATION_MIN_OBSERVATIONS = 2
GUIDED_APPROACH_MAX_STEPS = 45
GUIDED_APPROACH_STEP_M = 0.012
GUIDED_LATERAL_STEP_M = 0.008
GUIDED_VERTICAL_STEP_M = 0.008
GUIDED_APPROACH_FILL_RATIO = 0.42
GUIDED_APPROACH_STOP_DEPTH_CM = 10.0
GUIDED_TARGET_LOST_LIMIT = 8
GUIDED_CUT_OFFSET_M = 0.01
GUIDED_CENTER_TOLERANCE_RATIO = 0.08
CAMERA_INDEX       = -1      # -1 = auto-detect camera
DEFAULT_CAMERA_SOURCE = "opencv"
DEFAULT_ROS_IMAGE_TOPIC = "/camera/image_raw"
ROS_FRAME_TIMEOUT_S = 5.0
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


class RosImageSource:
    """OpenCV-like frame source backed by a ROS image topic."""

    def __init__(self, image_topic=DEFAULT_ROS_IMAGE_TOPIC,
                 timeout_s=ROS_FRAME_TIMEOUT_S):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.image_topic = image_topic
        self.timeout_s = timeout_s
        self.rospy = rospy
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._subscriber = None
        self._opened = False

        if not rospy.core.is_initialized():
            rospy.init_node("tomato_harvester", anonymous=True,
                            disable_signals=True)

        self._subscriber = rospy.Subscriber(
            image_topic, Image, self._callback, queue_size=1,
            buff_size=2 ** 24)

        deadline = time.time() + timeout_s
        while time.time() < deadline and not rospy.is_shutdown():
            with self._lock:
                if self._latest_frame is not None:
                    self._opened = True
                    logger.info("ROS camera topic opened: %s", image_topic)
                    return
            rospy.sleep(0.05)

        logger.error(
            "ROS camera topic %s did not publish a frame within %.1fs.",
            image_topic, timeout_s)

    def _callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            logger.warning("Could not convert ROS image frame: %s", e)
            return
        with self._lock:
            self._latest_frame = frame

    def isOpened(self):
        return self._opened

    def read(self):
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self):
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None

    def set(self, *_args):
        return False


def _open_ros_camera(image_topic=DEFAULT_ROS_IMAGE_TOPIC,
                     timeout_s=ROS_FRAME_TIMEOUT_S):
    try:
        source = RosImageSource(image_topic=image_topic, timeout_s=timeout_s)
    except ImportError as e:
        logger.error(
            "ROS camera source unavailable: %s. Source your ROS environment "
            "and install cv_bridge/sensor_msgs.", e)
        return None, None

    if source.isOpened():
        return source, image_topic

    source.release()
    return None, None


def _open_frame_source(camera_source=DEFAULT_CAMERA_SOURCE,
                       camera_index=CAMERA_INDEX,
                       ros_image_topic=DEFAULT_ROS_IMAGE_TOPIC,
                       ros_frame_timeout=ROS_FRAME_TIMEOUT_S):
    if camera_source == "ros":
        return _open_ros_camera(ros_image_topic, ros_frame_timeout)
    if camera_source == "opencv":
        return _open_camera(camera_index)
    raise ValueError("camera_source must be one of: opencv, ros")


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


def _detection_diameter_px(detection):
    x1, y1, x2, y2 = detection["bbox_px"]
    return max(0.0, ((x2 - x1) + (y2 - y1)) / 2.0)


def _detection_fill_ratio(detection, frame_shape):
    if frame_shape is None or len(frame_shape) < 2:
        return 0.0
    frame_h, frame_w = frame_shape[:2]
    frame_min = max(1, min(frame_w, frame_h))
    return _detection_diameter_px(detection) / frame_min


def _new_validation_track(detection):
    xyz = _position_vector_cm(detection)
    return {
        "count": 1,
        "xyz_sum": xyz.copy(),
        "center_sum": np.array(detection["center_px"], dtype=float),
        "best": detection,
        "last": detection,
    }


def _add_detection_to_tracks(tracks, detection,
                             tolerance_cm=VALIDATION_TRACK_TOLERANCE_CM):
    xyz = _position_vector_cm(detection)
    if not tracks:
        tracks.append(_new_validation_track(detection))
        return

    distances = [
        np.linalg.norm(xyz - (track["xyz_sum"] / track["count"]))
        for track in tracks
    ]
    nearest_idx = int(np.argmin(distances))
    if distances[nearest_idx] > tolerance_cm:
        tracks.append(_new_validation_track(detection))
        return

    track = tracks[nearest_idx]
    track["count"] += 1
    track["xyz_sum"] += xyz
    track["center_sum"] += np.array(detection["center_px"], dtype=float)
    track["last"] = detection
    if detection["confidence"] > track["best"]["confidence"]:
        track["best"] = detection


def _tracks_to_ranked_targets(tracks, max_targets=REFERENCE_TARGET_LIMIT,
                              min_observations=VALIDATION_MIN_OBSERVATIONS):
    targets = []
    for track in tracks:
        if track["count"] < min_observations:
            continue
        xyz = track["xyz_sum"] / track["count"]
        center = track["center_sum"] / track["count"]
        best = track["best"]
        targets.append({
            "rank": 0,
            "observations": track["count"],
            "confidence": best["confidence"],
            "bbox_px": best["bbox_px"],
            "center_px": [int(round(center[0])), int(round(center[1]))],
            "xyz_cm": {
                "x": float(round(xyz[0], 2)),
                "y": float(round(xyz[1], 2)),
                "z": float(round(xyz[2], 2)),
            },
            "distance_cm": float(round(xyz[2], 2)),
        })

    targets.sort(key=lambda target: target["distance_cm"])
    for rank, target in enumerate(targets[:max_targets], start=1):
        target["rank"] = rank
    return targets[:max_targets]


def _build_reference_snapshot(detection_samples, arm, captured_at=None,
                              max_targets=REFERENCE_TARGET_LIMIT,
                              min_observations=VALIDATION_MIN_OBSERVATIONS):
    tracks = []
    for detection in detection_samples:
        _add_detection_to_tracks(tracks, detection)

    return {
        "captured_at": captured_at if captured_at is not None else time.time(),
        "initial_joint_angles": arm.joint_angles.copy(),
        "initial_pulses": angles_to_pulses(arm.joint_angles),
        "targets": _tracks_to_ranked_targets(
            tracks, max_targets=max_targets,
            min_observations=min_observations),
    }


def _collect_validation_snapshot(cap, detector, arm,
                                 validation_seconds=CONFIRM_SECONDS,
                                 min_observations=CONFIRM_FRAMES):
    """Collect live detections for the validation window and rank closest targets."""
    logger.info("Starting %.1fs validation snapshot.", validation_seconds)
    deadline = time.time() + validation_seconds
    samples = []
    last_frame = None

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame grab failed during validation snapshot.")
            continue
        last_frame = frame
        detections = detector.detect(frame)
        samples.extend(detections)

        annotated = detector.annotate(frame, detections)
        remaining = max(0.0, deadline - time.time())
        cv2.putText(annotated, f"Validation snapshot: {remaining:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow("Tomato Harvester", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    snapshot = _build_reference_snapshot(
        samples, arm, min_observations=min_observations)
    logger.info(
        "Validation snapshot saved: %s target(s), initial pulses=%s",
        len(snapshot["targets"]), snapshot["initial_pulses"])
    for target in snapshot["targets"]:
        logger.info(
            "Snapshot rank %s: z=%.1fcm center=%s xyz=%s observations=%s",
            target["rank"], target["distance_cm"], target["center_px"],
            target["xyz_cm"], target["observations"])
    return snapshot, last_frame


def _select_live_target(detections, reference_target, last_detection=None):
    if not detections:
        return None

    anchor = last_detection or reference_target
    anchor_center = np.array(anchor["center_px"], dtype=float)
    anchor_xyz = np.array([
        anchor["xyz_cm"]["x"], anchor["xyz_cm"]["y"], anchor["xyz_cm"]["z"],
    ], dtype=float)

    def score(detection):
        center = np.array(detection["center_px"], dtype=float)
        xyz = _position_vector_cm(detection)
        center_score = np.linalg.norm(center - anchor_center) / 100.0
        xyz_score = np.linalg.norm(xyz - anchor_xyz) / 25.0
        confidence_bonus = detection["confidence"] * 0.2
        return center_score + xyz_score - confidence_bonus

    return min(detections, key=score)


def _solve_and_command_point(arm, driver, target_m,
                             duration_ms=GUIDED_MOVE_DURATION_MS,
                             error_limit_m=IK_MAX_ERROR_M):
    q_before = arm.joint_angles.copy()
    solved, err, iters = arm.inverse_kinematics(
        target_m, max_iters=500, tol=IK_TOLERANCE_M)
    q_target = arm.joint_angles.copy()
    logger.info("Guided IK solved=%s, error=%.4fm, iters=%s",
                solved, err, iters)
    if not solved or err > error_limit_m:
        arm.set_joint_angles(q_before)
        return False, f"IK failed: solved={solved}, error={err:.4f}m"

    pulses = angles_to_pulses(q_target)
    pulses[1] = GRIPPER_OPEN_PULSE
    driver.move_servos(pulses, duration_ms=duration_ms)
    time.sleep(duration_ms / 1000.0)
    return True, "moved"


def _visual_servo_target_point(arm, detection, frame_shape):
    """
    Compute the next small base-frame step from image error.

    The wrist camera moves with the arm, so live guidance must use relative
    image-centering corrections instead of the fixed search-home camera
    transform. Positive image X maps to arm -Y and positive image Y maps to
    arm -Z for the current mounting convention.
    """
    frame_h, frame_w = frame_shape[:2]
    frame_w = max(1, frame_w)
    frame_h = max(1, frame_h)
    cx, cy = detection["center_px"]
    norm_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
    norm_y = (cy - frame_h / 2.0) / (frame_h / 2.0)
    fill_ratio = _detection_fill_ratio(detection, frame_shape)

    current_m = arm.end_effector_pos()
    approach_scale = max(0.0, 1.0 - fill_ratio / GUIDED_APPROACH_FILL_RATIO)
    if detection["xyz_cm"]["z"] <= GUIDED_APPROACH_STOP_DEPTH_CM:
        approach_scale = 0.0

    delta = np.array([
        GUIDED_APPROACH_STEP_M * min(1.0, approach_scale),
        -GUIDED_LATERAL_STEP_M * float(np.clip(norm_x, -1.0, 1.0)),
        -GUIDED_VERTICAL_STEP_M * float(np.clip(norm_y, -1.0, 1.0)),
    ], dtype=float)
    return current_m + delta


def _is_detection_close_enough(detection, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    cx, cy = detection["center_px"]
    norm_x = abs((cx - frame_w / 2.0) / max(1.0, frame_w / 2.0))
    norm_y = abs((cy - frame_h / 2.0) / max(1.0, frame_h / 2.0))
    centered = (norm_x <= GUIDED_CENTER_TOLERANCE_RATIO and
                norm_y <= GUIDED_CENTER_TOLERANCE_RATIO)
    return (
        centered and (
            _detection_fill_ratio(detection, frame_shape) >=
            GUIDED_APPROACH_FILL_RATIO or
            detection["xyz_cm"]["z"] <= GUIDED_APPROACH_STOP_DEPTH_CM
        )
    )


def _guided_harvest_target(arm, driver, detector, cap, reference_target):
    """Use live detections to slowly approach and cut one ranked target."""
    logger.info("Guided harvest for snapshot rank %s started.",
                reference_target["rank"])
    driver.gripper_open(duration_ms=400)
    time.sleep(0.3)

    last_detection = None
    lost_count = 0
    saw_target = False

    for step in range(1, GUIDED_APPROACH_MAX_STEPS + 1):
        ret, frame = cap.read()
        if not ret:
            lost_count += 1
            logger.warning("No frame while guiding rank %s (%s/%s).",
                           reference_target["rank"], lost_count,
                           GUIDED_TARGET_LOST_LIMIT)
            if lost_count >= GUIDED_TARGET_LOST_LIMIT:
                return False, "target lost: no frames"
            continue

        detections = detector.detect(frame)
        target = _select_live_target(detections, reference_target,
                                     last_detection)
        annotated = detector.annotate(frame, detections)

        if target is None:
            lost_count += 1
            logger.warning("No live target match for rank %s (%s/%s).",
                           reference_target["rank"], lost_count,
                           GUIDED_TARGET_LOST_LIMIT)
            if lost_count >= GUIDED_TARGET_LOST_LIMIT:
                return False, "target lost: no detections"
            cv2.imshow("Tomato Harvester", annotated)
            cv2.waitKey(1)
            continue

        lost_count = 0
        last_detection = target
        saw_target = True
        fill_ratio = _detection_fill_ratio(target, frame.shape)
        logger.info(
            "Guided rank %s step %s: fill=%.2f z=%.1fcm center=%s",
            reference_target["rank"], step, fill_ratio,
            target["xyz_cm"]["z"], target["center_px"])

        if _is_detection_close_enough(target, frame.shape):
            logger.info("Rank %s centered and close enough for cut "
                        "(fill=%.2f, z=%.1fcm).",
                        reference_target["rank"], fill_ratio,
                        target["xyz_cm"]["z"])
            cv2.imshow("Tomato Harvester", annotated)
            cv2.waitKey(1)
            break

        next_point_m = _visual_servo_target_point(arm, target, frame.shape)
        ok, reason = _solve_and_command_point(arm, driver, next_point_m)
        if not ok:
            return False, reason

        cv2.putText(annotated, f"Guiding rank {reference_target['rank']} step {step}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow("Tomato Harvester", annotated)
        cv2.waitKey(1)
    else:
        logger.warning("Guided approach reached max steps for rank %s.",
                       reference_target["rank"])

    if not saw_target:
        return False, "no final target center"

    cut_point_m = arm.end_effector_pos() + (
        STEM_DIRECTION_ARM_FRAME / np.linalg.norm(STEM_DIRECTION_ARM_FRAME)
    ) * GUIDED_CUT_OFFSET_M
    logger.info("Rank %s cut point 1cm vertically above current end-effector: %s",
                reference_target["rank"], np.round(cut_point_m, 4))

    ok, reason = _solve_and_command_point(
        arm, driver, cut_point_m, duration_ms=MOVE_DURATION_MS)
    if not ok:
        return False, reason

    logger.info("CUTTING rank %s — closing gripper.", reference_target["rank"])
    driver.gripper_close(duration_ms=500)
    time.sleep(GRIPPER_DELAY_S)
    driver.gripper_open(duration_ms=400)
    time.sleep(0.2)
    return True, f"rank {reference_target['rank']} harvested"


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
                   camera_source=DEFAULT_CAMERA_SOURCE,
                   ros_image_topic=DEFAULT_ROS_IMAGE_TOPIC,
                   ros_frame_timeout=ROS_FRAME_TIMEOUT_S,
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

    # Open camera before entering the harvest loop.
    cap, selected_camera = _open_frame_source(
        camera_source=camera_source,
        camera_index=camera_index,
        ros_image_topic=ros_image_topic,
        ros_frame_timeout=ros_frame_timeout)
    if cap is None:
        driver.go_park(duration_ms=1000)
        driver.close()
        return

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
                    cap, selected_camera = _open_frame_source(
                        camera_source=camera_source,
                        camera_index=camera_index,
                        ros_image_topic=ros_image_topic,
                        ros_frame_timeout=ros_frame_timeout)
                    state = "SCANNING"
                    frame_grab_failures = 0
                    if cap is None:
                        break
                continue
            frame_grab_failures = 0

            if state in ("SCANNING", "CONFIRMING"):
                detections = detector.detect(frame)
                annotated = detector.annotate(frame, detections)

                if detections:
                    state = "CONFIRMING"
                    logger.info(
                        "Tomato detected; collecting %.1fs validation snapshot.",
                        confirm_seconds)
                    snapshot, _last_frame = _collect_validation_snapshot(
                        cap, detector, arm,
                        validation_seconds=confirm_seconds,
                        min_observations=confirm_frames)
                    if not snapshot["targets"]:
                        logger.warning(
                            "Validation snapshot did not confirm any stable "
                            "targets; continuing live scan.")
                        state = "SCANNING"
                        continue

                    state = "ACTING"
                    logger.info(
                        "Reference snapshot saved with %s target(s). "
                        "Harvesting by distance rank.",
                        len(snapshot["targets"]))
                    for target in snapshot["targets"]:
                        harvest_ok, harvest_message = _guided_harvest_target(
                            arm, driver, detector, cap, target)
                        last_status_message = harvest_message
                        if harvest_ok:
                            logger.info("Harvest result: %s", harvest_message)
                        else:
                            logger.warning("Harvest skipped: %s",
                                           harvest_message)

                    state = "RETRACTING"
                    logger.info("Returning to Search Home after ranked targets.")
                    arm.set_joint_angles(home_angles)
                    pulses = angles_to_pulses(home_angles)
                    driver.move_servos(pulses, duration_ms=1000)
                    time.sleep(RETRACT_DELAY_S)

                    state = "SCANNING"
                    logger.info("Ready for next validation snapshot.\n")

                # Show status on annotated frame
                status_text = (
                    f"Mode: {mode.upper()}  Camera: {selected_camera}  "
                    f"State: {state}")
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
    parser.add_argument("--camera-source",
                        choices=("opencv", "ros"),
                        default=DEFAULT_CAMERA_SOURCE,
                        help="Frame source: opencv uses /dev/video*, ros subscribes to an image topic")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Camera index, or -1 to auto-detect (default: -1)")
    parser.add_argument("--ros-image-topic", default=DEFAULT_ROS_IMAGE_TOPIC,
                        help=f"ROS sensor_msgs/Image topic (default: {DEFAULT_ROS_IMAGE_TOPIC})")
    parser.add_argument("--ros-frame-timeout", type=float,
                        default=ROS_FRAME_TIMEOUT_S,
                        help=f"Seconds to wait for first ROS frame (default: {ROS_FRAME_TIMEOUT_S})")
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
                       camera_source=args.camera_source,
                       ros_image_topic=args.ros_image_topic,
                       ros_frame_timeout=args.ros_frame_timeout,
                       servo_backend=args.servo_backend,
                       uart_port=args.uart_port,
                       baud=args.baud)


if __name__ == "__main__":
    main()
