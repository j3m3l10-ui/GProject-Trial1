"""
Single-file hardware controller for a tomato harvesting robot.

Run on the robot with:
    python tomato_harvester.py

This script intentionally has no GUI, ROS dependency, argparse, or external
configuration file. Calibrate the constants below for the deployed camera,
camera mount, arm geometry, and servo IDs before field use.
"""

import logging
import math
import signal
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera calibration and capture constants
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAMERA_FPS = 30

# Replace these placeholder intrinsics with measured calibration values.
K = np.array(
    [
        [1400.0, 0.0, 960.0],
        [0.0, 1400.0, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Detection and depth constants
# ---------------------------------------------------------------------------

MODEL_PATH = "tomato_ripe_yolov8n.pt"
RIPE_TOMATO_CLASS_ID = 0
CONF_THRESHOLD = 0.60
IOU_THRESHOLD = 0.45
REAL_TOMATO_DIAMETER_M = 0.07
APPROACH_OFFSET_M = 0.05
DEPTH_PATCH_SIZE_PX = 10
MIN_DEPTH_M = 0.02


# ---------------------------------------------------------------------------
# Camera-to-base transform
# ---------------------------------------------------------------------------

# Hand-in-eye style starting transform: camera Z -> base X, camera X -> -base Y,
# camera Y -> -base Z, plus a calibrated mount offset. Replace after calibration.
T_CAM2BASE = np.array(
    [
        [0.0, 0.0, 1.0, 0.15],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.18],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Arm geometry, limits, and home pose
# ---------------------------------------------------------------------------

DH_PARAMS = {
    "d1": 0.072,
    "a2": 0.104,
    "a3": 0.096,
    "d5": 0.070,
}

JOINT_LIMITS_RAD = np.array(
    [
        [math.radians(-180.0), math.radians(180.0)],
        [math.radians(-150.0), math.radians(150.0)],
        [math.radians(-150.0), math.radians(150.0)],
        [math.radians(-150.0), math.radians(150.0)],
        [math.radians(-180.0), math.radians(180.0)],
    ],
    dtype=np.float64,
)

HOME_ANGLES_RAD = np.array(
    [
        0.0,
        math.radians(-30.0),
        math.radians(60.0),
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Dynamixel constants
# ---------------------------------------------------------------------------

DYNAMIXEL_PORT = "/dev/ttyUSB0"
DYNAMIXEL_BAUDRATE = 1_000_000
DYNAMIXEL_PROTOCOL_VERSION = 2.0

SERVO_IDS = [1, 2, 3, 4, 5]
CUTTER_SERVO_ID = 6
ALL_SERVO_IDS = SERVO_IDS + [CUTTER_SERVO_ID]

OPERATING_MODE_POSITION = 3
PROFILE_ACCELERATION = 40
PROFILE_VELOCITY = 120

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

POSITION_MIN = 0
POSITION_MAX = 4095
POSITION_CENTER = 2048
POSITION_PER_PI = 2048
POSITION_SETTLE_TOLERANCE = 20
POSITION_SETTLE_TIMEOUT_S = 5.0

CUTTER_CLOSED_POSITION = 3000
CUTTER_OPEN_POSITION = 2048
CUTTER_HOLD_S = 0.8


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    u: float
    v: float
    width: float
    height: float
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class Camera:
    """Captures BGR frames, undistorts them, and resolves depth/back-projection."""

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera at index {CAMERA_INDEX}")

        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        self.rs = None
        self.rs_pipeline = None
        self.rs_align = None
        self.rs_depth_scale = 1.0
        self._try_start_realsense()

    def _try_start_realsense(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            logger.info("pyrealsense2 not installed; using apparent-size depth.")
            return

        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, CAMERA_FPS)
            config.enable_stream(
                rs.stream.color,
                FRAME_WIDTH,
                FRAME_HEIGHT,
                rs.format.bgr8,
                CAMERA_FPS,
            )
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.rs_depth_scale = float(depth_sensor.get_depth_scale())
            self.rs = rs
            self.rs_pipeline = pipeline
            self.rs_align = rs.align(rs.stream.color)
            logger.info("RealSense depth enabled and aligned to color stream.")
        except Exception as exc:
            logger.warning("RealSense unavailable; using apparent-size depth: %s", exc)
            self.rs = None
            self.rs_pipeline = None
            self.rs_align = None

    def capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame from USB camera")

        undistorted = cv2.undistort(frame, K, DIST, None, K)
        depth_m = self._capture_realsense_depth_m()
        return undistorted, depth_m

    def _capture_realsense_depth_m(self) -> Optional[np.ndarray]:
        if self.rs_pipeline is None or self.rs_align is None:
            return None

        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.rs_align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                return None

            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_m *= self.rs_depth_scale
            if depth_m.shape != (FRAME_HEIGHT, FRAME_WIDTH):
                depth_m = cv2.resize(
                    depth_m,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
            return depth_m
        except Exception as exc:
            logger.warning("RealSense depth grab failed: %s", exc)
            return None

    def resolve_depth(self, detection: Detection, depth_frame_m: Optional[np.ndarray]) -> Optional[float]:
        if depth_frame_m is not None:
            depth_m = self._median_depth_patch(depth_frame_m, detection.u, detection.v)
            if depth_m is not None:
                return self._apply_approach_offset(depth_m)

        pixel_diameter = max(detection.width, detection.height)
        if pixel_diameter <= 0:
            return None

        estimated_depth_m = (REAL_TOMATO_DIAMETER_M * self.fx) / pixel_diameter
        return self._apply_approach_offset(estimated_depth_m)

    def _median_depth_patch(self, depth_frame_m: np.ndarray, u: float, v: float) -> Optional[float]:
        half = DEPTH_PATCH_SIZE_PX // 2
        x = int(round(u))
        y = int(round(v))
        h, w = depth_frame_m.shape[:2]
        x0 = max(0, x - half)
        x1 = min(w, x + half)
        y0 = max(0, y - half)
        y1 = min(h, y + half)

        patch = depth_frame_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _apply_approach_offset(depth_m: float) -> Optional[float]:
        adjusted_depth_m = float(depth_m) - APPROACH_OFFSET_M
        if adjusted_depth_m < MIN_DEPTH_M:
            return None
        return adjusted_depth_m

    def back_project(self, u: float, v: float, depth_m: float) -> np.ndarray:
        x_c = (u - self.cx) * depth_m / self.fx
        y_c = (v - self.cy) * depth_m / self.fy
        z_c = depth_m
        return np.array([x_c, y_c, z_c], dtype=np.float64)

    def close(self):
        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
        self.cap.release()


class TomatoDetector:
    """Loads YOLOv8 and returns the highest-confidence ripe tomato detection."""

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(MODEL_PATH)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        results = self.model(
            frame_bgr,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[RIPE_TOMATO_CLASS_ID],
            verbose=False,
        )

        detections: list[Detection] = []
        boxes = getattr(results[0], "boxes", None) if results else None
        if boxes is None:
            return None

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            confidence = float(box.conf[0])
            detections.append(
                Detection(
                    u=(x1 + x2) / 2.0,
                    v=(y1 + y2) / 2.0,
                    width=width,
                    height=height,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[0] if detections else None


class ServoController:
    """Dynamixel Protocol 2.0 controller for five joints plus cutter servo."""

    def __init__(self):
        import dynamixel_sdk as dxl

        self.dxl = dxl
        self.port_handler = dxl.PortHandler(DYNAMIXEL_PORT)
        self.packet_handler = dxl.PacketHandler(DYNAMIXEL_PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port {DYNAMIXEL_PORT}")
        if not self.port_handler.setBaudRate(DYNAMIXEL_BAUDRATE):
            raise RuntimeError(f"Failed to set baudrate {DYNAMIXEL_BAUDRATE}")

        self._initialize_servos()

    def _initialize_servos(self):
        for servo_id in ALL_SERVO_IDS:
            self._write1(servo_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self._write1(servo_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
            self._write4(servo_id, ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION)
            self._write4(servo_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)
            self._write1(servo_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        self.set_cutter(CUTTER_OPEN_POSITION)
        self.go_home(wait=False)
        logger.info("Dynamixel servos initialized.")

    def _check_comm(self, result: int, error: int, action: str):
        if result != self.dxl.COMM_SUCCESS:
            message = self.packet_handler.getTxRxResult(result)
            raise RuntimeError(f"{action} failed: {message}")
        if error:
            message = self.packet_handler.getRxPacketError(error)
            raise RuntimeError(f"{action} servo error: {message}")

    def _write1(self, servo_id: int, address: int, value: int):
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler,
            servo_id,
            address,
            int(value),
        )
        self._check_comm(result, error, f"write1 ID {servo_id} addr {address}")

    def _write4(self, servo_id: int, address: int, value: int):
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler,
            servo_id,
            address,
            int(value),
        )
        self._check_comm(result, error, f"write4 ID {servo_id} addr {address}")

    def read_position(self, servo_id: int) -> int:
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler,
            servo_id,
            ADDR_PRESENT_POSITION,
        )
        self._check_comm(result, error, f"read position ID {servo_id}")
        return int(position)

    def send_joint_angles(self, angles_rad: np.ndarray) -> dict[int, int]:
        positions = {
            servo_id: angle_to_dynamixel_position(angle)
            for servo_id, angle in zip(SERVO_IDS, angles_rad, strict=True)
        }
        self.send_joint_positions(positions)
        return positions

    def send_joint_positions(self, positions: dict[int, int]):
        group = self.dxl.GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        for servo_id, position in positions.items():
            param = self._position_to_little_endian_bytes(position)
            if not group.addParam(int(servo_id), param):
                group.clearParam()
                raise RuntimeError(f"Failed to add ID {servo_id} to sync write")

        result = group.txPacket()
        group.clearParam()
        if result != self.dxl.COMM_SUCCESS:
            message = self.packet_handler.getTxRxResult(result)
            raise RuntimeError(f"GroupSyncWrite failed: {message}")

    def _position_to_little_endian_bytes(self, position: int) -> list[int]:
        position = clamp_position(position)
        return [
            self.dxl.DXL_LOBYTE(self.dxl.DXL_LOWORD(position)),
            self.dxl.DXL_HIBYTE(self.dxl.DXL_LOWORD(position)),
            self.dxl.DXL_LOBYTE(self.dxl.DXL_HIWORD(position)),
            self.dxl.DXL_HIBYTE(self.dxl.DXL_HIWORD(position)),
        ]

    def wait_until_settled(
        self,
        goal_positions: dict[int, int],
        timeout_s: float = POSITION_SETTLE_TIMEOUT_S,
        tolerance: int = POSITION_SETTLE_TOLERANCE,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            settled = True
            for servo_id, goal in goal_positions.items():
                current = self.read_position(servo_id)
                if abs(current - goal) > tolerance:
                    settled = False
                    break
            if settled:
                return True
            time.sleep(0.05)
        logger.warning("Servo settle timeout after %.1fs", timeout_s)
        return False

    def set_cutter(self, position: int):
        self._write4(CUTTER_SERVO_ID, ADDR_GOAL_POSITION, clamp_position(position))

    def cut(self):
        self.set_cutter(CUTTER_CLOSED_POSITION)
        time.sleep(CUTTER_HOLD_S)
        self.set_cutter(CUTTER_OPEN_POSITION)

    def go_home(self, wait: bool = True):
        goal_positions = self.send_joint_angles(HOME_ANGLES_RAD)
        if wait:
            self.wait_until_settled(goal_positions)

    def disable_torque(self):
        for servo_id in ALL_SERVO_IDS:
            try:
                self._write1(servo_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            except Exception as exc:
                logger.warning("Failed to disable torque on ID %s: %s", servo_id, exc)

    def close(self):
        self.port_handler.closePort()


def clamp_position(position: int) -> int:
    return max(POSITION_MIN, min(POSITION_MAX, int(position)))


def angle_to_dynamixel_position(angle_rad: float) -> int:
    position = int(POSITION_CENTER + (float(angle_rad) / math.pi) * POSITION_PER_PI)
    return clamp_position(position)


def camera_to_base(point_camera_m: np.ndarray) -> np.ndarray:
    point_h = np.array(
        [point_camera_m[0], point_camera_m[1], point_camera_m[2], 1.0],
        dtype=np.float64,
    )
    point_base_h = T_CAM2BASE @ point_h
    return point_base_h[:3]


def inverse_kinematics(target_base_m: np.ndarray) -> Optional[np.ndarray]:
    x_b, y_b, z_b = map(float, target_base_m)
    d1 = DH_PARAMS["d1"]
    a2 = DH_PARAMS["a2"]
    a3 = DH_PARAMS["a3"]

    theta1 = math.atan2(y_b, x_b)
    r = math.sqrt(x_b * x_b + y_b * y_b)
    z = z_b - d1

    d = (r * r + z * z - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
    if d < -1.0 or d > 1.0:
        logger.warning("IK target unreachable; cosine-rule D=%.3f", d)
        return None

    theta3 = math.atan2(-math.sqrt(max(0.0, 1.0 - d * d)), d)
    theta2 = math.atan2(z, r) - math.atan2(
        a3 * math.sin(theta3),
        a2 + a3 * math.cos(theta3),
    )
    theta4 = -(theta2 + theta3)
    theta5 = theta1

    angles = np.array([theta1, theta2, theta3, theta4, theta5], dtype=np.float64)
    if not _within_joint_limits(angles):
        logger.warning("IK solution outside joint limits: %s", np.rad2deg(angles))
        return None
    return angles


def _within_joint_limits(angles_rad: np.ndarray) -> bool:
    return bool(
        np.all(angles_rad >= JOINT_LIMITS_RAD[:, 0])
        and np.all(angles_rad <= JOINT_LIMITS_RAD[:, 1])
    )


def forward_kinematics(angles_rad: np.ndarray) -> np.ndarray:
    theta1, theta2, theta3, theta4, _theta5 = map(float, angles_rad)
    d1 = DH_PARAMS["d1"]
    a2 = DH_PARAMS["a2"]
    a3 = DH_PARAMS["a3"]
    d5 = DH_PARAMS["d5"]

    pitch23 = theta2 + theta3
    pitch234 = pitch23 + theta4
    planar_r = (
        a2 * math.cos(theta2)
        + a3 * math.cos(pitch23)
        + d5 * math.cos(pitch234)
    )
    z = (
        d1
        + a2 * math.sin(theta2)
        + a3 * math.sin(pitch23)
        + d5 * math.sin(pitch234)
    )

    x = planar_r * math.cos(theta1)
    y = planar_r * math.sin(theta1)
    return np.array([x, y, z], dtype=np.float64)


class HarvestRobot:
    """Owns camera, detector, servos, and the harvest loop."""

    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_sigint)

        self.camera = Camera()
        self.detector = TomatoDetector()
        self.servos = ServoController()

    def _handle_sigint(self, _signum, _frame):
        logger.info("SIGINT received; returning home and shutting down.")
        self.shutdown_requested = True

    def run(self):
        logger.info("Starting tomato harvesting loop.")
        try:
            while not self.shutdown_requested:
                frame, depth_frame_m = self.camera.capture()
                detection = self.detector.detect(frame)
                if detection is None:
                    continue

                depth_m = self.camera.resolve_depth(detection, depth_frame_m)
                if depth_m is None:
                    logger.info("Skipping detection with invalid depth.")
                    continue

                point_camera = self.camera.back_project(detection.u, detection.v, depth_m)
                target_base = camera_to_base(point_camera)
                angles = inverse_kinematics(target_base)
                if angles is None:
                    continue

                logger.info(
                    "Harvest target base xyz=%s, conf=%.2f",
                    np.round(target_base, 4),
                    detection.confidence,
                )
                goal_positions = self.servos.send_joint_angles(angles)
                self.servos.wait_until_settled(goal_positions)
                self.servos.cut()
                self.servos.go_home()
        finally:
            self.shutdown()

    def shutdown(self):
        try:
            self.servos.go_home()
        except Exception as exc:
            logger.warning("Failed to go home during shutdown: %s", exc)
        try:
            self.servos.disable_torque()
        except Exception as exc:
            logger.warning("Failed to disable torque during shutdown: %s", exc)
        self.servos.close()
        self.camera.close()
        logger.info("Tomato harvester stopped.")


if __name__ == "__main__":
    HarvestRobot().run()
