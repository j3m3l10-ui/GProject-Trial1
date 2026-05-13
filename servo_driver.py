"""
Servo Driver — Hiwonder ArmPi Pro Serial Bus Servos
=====================================================
Provides a unified interface to move servos, with two backends:
  1. REAL mode  — sends commands via UART to Hiwonder expansion board on RPi5
  2. SIM  mode  — logs commands and updates an in-memory state (for GUI/testing)

The Hiwonder LX-series serial bus servos use a single-wire half-duplex UART
protocol on /dev/ttyAMA0 (RPi5) at 115200 baud.

Protocol frame:
  [0x55][0x55][ID][LEN][CMD][PARAM1][PARAM2]...[CHECKSUM]
  CMD 1 = SERVO_MOVE_TIME_WRITE: move servo to position over duration
"""

import importlib
import os
import sys
import time
import struct
import logging
from glob import glob
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Protocol constants ─────────────────────────────────────────────────────────
FRAME_HEADER     = bytes([0x55, 0x55])
CMD_SERVO_MOVE   = 1    # SERVO_MOVE_TIME_WRITE
CMD_SERVO_READ   = 28   # SERVO_POS_READ
BROADCAST_ID     = 254
GRIPPER_OPEN_PULSE = 200
GRIPPER_CLOSE_PULSE = 700

# ── Default UART settings ─────────────────────────────────────────────────────
DEFAULT_UART_PORT = "auto"
DEFAULT_DIRECTION_GPIO = 17
FALLBACK_UART_PORTS = (
    "/dev/ttyUSB0",
    "/dev/ttyUSB1",
    "/dev/ttyACM0",
    "/dev/ttyACM1",
    "/dev/ttyAMA0",
    "/dev/ttyS0",
)
DEFAULT_BAUD      = 115200
DEFAULT_SERVO_BACKEND = "auto"
SDK_BOARD_MODULES = (
    "HiwonderSDK.Board",
    "hiwonder.Board",
    "Board",
)
SDK_ENV_VAR = "HIWONDER_SDK_PATH"
SDK_SEARCH_ROOTS = (
    ".",
    "/home/pi",
    "/home/pi/ArmPi",
    "/home/pi/ArmPi/HiwonderSDK",
    "/home/pi/ArmPi/hiwonder",
    "/home/ubuntu",
    "/home/ubuntu/ArmPi",
    "/home/ubuntu/ArmPi/HiwonderSDK",
    "/home/ubuntu/ArmPi/hiwonder",
    "/home/hiwonder",
    "/home/hiwonder/ArmPi",
    "/home/hiwonder/ArmPi/HiwonderSDK",
)


def _checksum(buf: bytes) -> int:
    """Hiwonder checksum: ~(ID + LEN + CMD + params) & 0xFF."""
    return (~sum(buf[2:])) & 0xFF


def _build_move_cmd(servo_id: int, position: int, duration_ms: int) -> bytes:
    """Build a single-servo move command packet."""
    position = max(0, min(1000, position))
    duration_ms = max(0, min(30000, duration_ms))
    payload = struct.pack('<BHH', servo_id, duration_ms, position)
    length = len(payload) + 2   # LEN = data bytes + cmd byte + checksum
    pkt = bytearray(FRAME_HEADER)
    pkt.append(servo_id)
    pkt.append(length)
    pkt.append(CMD_SERVO_MOVE)
    pkt.extend(struct.pack('<HH', position, duration_ms))
    pkt.append(_checksum(pkt))
    return bytes(pkt)


def _candidate_sdk_paths():
    """Return sys.path candidates for common Hiwonder SDK layouts."""
    roots = []
    env_value = os.environ.get(SDK_ENV_VAR, "")
    roots.extend([p for p in env_value.split(os.pathsep) if p])
    roots.extend(SDK_SEARCH_ROOTS)

    candidates = []
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.exists(root):
            continue

        checks = (
            root,
            os.path.dirname(root),
            os.path.join(root, "HiwonderSDK"),
            os.path.join(root, "hiwonder"),
        )
        for path in checks:
            if not path or not os.path.isdir(path):
                continue
            parent = os.path.dirname(path)
            if os.path.exists(os.path.join(path, "Board.py")):
                candidates.extend([path, parent])
            if os.path.exists(os.path.join(path, "HiwonderSDK", "Board.py")):
                candidates.append(path)
            if os.path.exists(os.path.join(path, "hiwonder", "Board.py")):
                candidates.append(path)

    unique = []
    for path in candidates:
        if path and path not in unique:
            unique.append(path)
    return unique


def _add_sdk_search_paths():
    paths = _candidate_sdk_paths()
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    return paths


def _candidate_uart_ports(uart_port: str):
    """Return UART devices to try. 'auto' means discover common serial names."""
    if uart_port != "auto":
        return [uart_port]

    candidates = []
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/ttyAMA*",
        "/dev/ttyS*",
    ):
        candidates.extend(sorted(glob(pattern)))

    candidates.extend(FALLBACK_UART_PORTS)
    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


class ServoDriver:
    """Unified servo driver with real/sim backends."""

    def __init__(self, mode: str = "sim", uart_port: str = DEFAULT_UART_PORT,
                 baud: int = DEFAULT_BAUD,
                 backend: str = DEFAULT_SERVO_BACKEND,
                 direction_gpio: int = DEFAULT_DIRECTION_GPIO):
        """
        Args:
            mode: "real" for hardware UART, "sim" for simulation/logging
            uart_port: serial port path (RPi5: /dev/ttyAMA0)
            baud: baud rate
            backend: "auto", "sdk", "uart-gpio", or "uart" for real hardware mode
            direction_gpio: BCM GPIO pin controlling half-duplex bus direction
        """
        self.mode = mode.lower()
        self.backend_requested = backend.lower()
        self.backend = "sim" if self.mode != "real" else None
        self.uart_port = uart_port
        self.baud = baud
        self.direction_gpio = direction_gpio
        self._gpio_handle = None
        self._gpio_module = None
        self.serial_conn = None
        self.board = None
        self.state: Dict[int, int] = {}  # servo_id → current pulse
        self._move_log = []               # history of moves (for GUI playback)

        if self.mode == "real":
            if self.backend_requested not in ("auto", "sdk", "uart-gpio", "uart"):
                raise ValueError("backend must be one of: auto, sdk, uart-gpio, uart")

            sdk_error = None
            if self.backend_requested in ("auto", "sdk"):
                try:
                    self._open_sdk_backend()
                except Exception as e:
                    sdk_error = e
                    if self.backend_requested == "sdk":
                        logger.error(f"[SERVO] Cannot open Hiwonder SDK backend: {e}")
                        raise
                    logger.warning(
                        "[SERVO] Hiwonder SDK backend unavailable (%s); "
                        "falling back to raw UART.", e)

            if self.backend is None and self.backend_requested in ("auto", "uart-gpio"):
                try:
                    self._open_uart_backend(uart_port, baud, use_gpio=True)
                except Exception as e:
                    if self.backend_requested == "uart-gpio":
                        raise
                    logger.warning(
                        "[SERVO] GPIO-controlled UART unavailable (%s); "
                        "falling back to plain UART.", e)

            if self.backend is None and self.backend_requested in ("auto", "uart"):
                try:
                    self._open_uart_backend(uart_port, baud, use_gpio=False)
                except Exception:
                    if sdk_error is not None:
                        logger.error(
                            "[SERVO] SDK and raw UART backends both failed.")
                    raise
        else:
            logger.info("[SERVO] Running in SIMULATION mode")

    def _open_sdk_backend(self):
        """Use Hiwonder's official Board.setBusServoPulse API."""
        errors = []
        sdk_paths = _add_sdk_search_paths()
        for module_name in SDK_BOARD_MODULES:
            try:
                board = importlib.import_module(module_name)
            except ImportError as e:
                errors.append(f"{module_name}: {e}")
                continue

            if not hasattr(board, "setBusServoPulse"):
                errors.append(f"{module_name}: missing setBusServoPulse")
                continue

            self.board = board
            self.backend = "sdk"
            logger.info(f"[SERVO] Using Hiwonder SDK backend: {module_name}")
            return

        searched = f"; searched paths={sdk_paths}" if sdk_paths else ""
        raise ImportError(
            ("; ".join(errors) or "no SDK module candidates") + searched)

    def _open_uart_backend(self, uart_port: str, baud: int, use_gpio: bool = False):
        """Use direct serial bus-servo packets as a fallback backend."""
        errors = []
        last_error = None
        for candidate in _candidate_uart_ports(uart_port):
            try:
                self._open_uart_candidate(candidate, baud)
                if use_gpio:
                    self._open_direction_gpio()
                    self.backend = "uart-gpio"
                self.uart_port = candidate
                return
            except Exception as e:
                errors.append(f"{candidate}: {e}")
                last_error = e

        raise last_error or RuntimeError(
            "No UART candidates available: " + "; ".join(errors))

    def _open_uart_candidate(self, uart_port: str, baud: int):
        try:
            import serial
            self.serial_conn = serial.Serial(
                uart_port, baud, timeout=0.5,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.backend = "uart"
            logger.info(f"[SERVO] Opened raw UART backend {uart_port} @ {baud}")
        except ImportError:
            logger.error("[SERVO] pyserial not installed. pip install pyserial")
            raise
        except Exception as e:
            logger.error(f"[SERVO] Cannot open raw UART {uart_port}: {e}")
            raise

    def _open_direction_gpio(self):
        """
        Enable TX/RX direction control for Hiwonder half-duplex bus boards.
        The common ArmPi/Hiwonder expansion-board direction pin is BCM GPIO17.
        """
        try:
            import lgpio
            handle = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(handle, self.direction_gpio, 0)
            self._gpio_module = lgpio
            self._gpio_handle = handle
            logger.info(
                "[SERVO] Enabled UART direction GPIO BCM%d via lgpio",
                self.direction_gpio)
            return
        except Exception as lgpio_error:
            try:
                import RPi.GPIO as GPIO
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.direction_gpio, GPIO.OUT)
                GPIO.output(self.direction_gpio, 0)
                self._gpio_module = GPIO
                self._gpio_handle = "RPi.GPIO"
                logger.info(
                    "[SERVO] Enabled UART direction GPIO BCM%d via RPi.GPIO",
                    self.direction_gpio)
                return
            except Exception as gpio_error:
                raise RuntimeError(
                    f"Cannot enable direction GPIO BCM{self.direction_gpio}: "
                    f"lgpio={lgpio_error}; RPi.GPIO={gpio_error}")

    def _set_direction_tx(self):
        if self._gpio_module is None:
            return
        if self._gpio_handle == "RPi.GPIO":
            self._gpio_module.output(self.direction_gpio, 1)
        else:
            self._gpio_module.gpio_write(
                self._gpio_handle, self.direction_gpio, 1)

    def _set_direction_rx(self):
        if self._gpio_module is None:
            return
        if self._gpio_handle == "RPi.GPIO":
            self._gpio_module.output(self.direction_gpio, 0)
        else:
            self._gpio_module.gpio_write(
                self._gpio_handle, self.direction_gpio, 0)

    def _fallback_to_uart_after_sdk_error(self, error):
        if self.backend_requested != "auto":
            raise RuntimeError(
                "Hiwonder SDK servo command failed. If this mentions lgpio "
                "or .lgd-nfy files, run with --servo-backend auto or uart, "
                "or fix the Raspberry Pi GPIO/lgpio daemon environment."
            ) from error

        logger.warning(
            "[SERVO] SDK command failed at runtime (%s). "
            "Falling back to raw UART.", error)
        self.board = None
        self.backend = None
        try:
            self._open_uart_backend(self.uart_port, self.baud, use_gpio=True)
        except Exception as gpio_error:
            logger.warning(
                "[SERVO] GPIO-controlled UART fallback failed (%s); "
                "trying plain UART.", gpio_error)
            self._open_uart_backend(self.uart_port, self.baud, use_gpio=False)

    # ── Move a single servo ────────────────────────────────────────────────────
    def move_servo(self, servo_id: int, pulse: int, duration_ms: int = 500):
        """Move a single servo to the given pulse over duration_ms."""
        pulse = max(0, min(1000, pulse))
        entry = {"id": servo_id, "pulse": pulse, "duration_ms": duration_ms,
                 "time": time.time()}
        self._move_log.append(entry)
        self.state[servo_id] = pulse

        if self.mode == "real" and self.backend == "sdk":
            try:
                self.board.setBusServoPulse(servo_id, pulse, duration_ms)
                logger.debug(f"[SERVO SDK] ID{servo_id} → {pulse}  ({duration_ms}ms)")
            except Exception as e:
                self._fallback_to_uart_after_sdk_error(e)
                self.move_servo(servo_id, pulse, duration_ms)
        elif (self.mode == "real" and self.backend in ("uart", "uart-gpio") and
              self.serial_conn):
            pkt = _build_move_cmd(servo_id, pulse, duration_ms)
            self._set_direction_tx()
            time.sleep(0.001)
            self.serial_conn.write(pkt)
            try:
                self.serial_conn.flush()
            except Exception:
                pass
            time.sleep(0.003)
            self._set_direction_rx()
            logger.debug(f"[SERVO UART] ID{servo_id} → {pulse}  ({duration_ms}ms)")
        else:
            logger.debug(f"[SIM] ID{servo_id} → {pulse}  ({duration_ms}ms)")

    # ── Move multiple servos simultaneously ────────────────────────────────────
    def move_servos(self, pulses: Dict[int, int], duration_ms: int = 500):
        """Move multiple servos at once. pulses = {servo_id: pulse_value}."""
        if self.mode == "real":
            logger.info(
                "[SERVO] Commanding backend=%s duration=%sms pulses=%s",
                self.backend, duration_ms, dict(pulses))
        for sid, pulse in pulses.items():
            self.move_servo(sid, pulse, duration_ms)

    # ── Gripper control ────────────────────────────────────────────────────────
    def gripper_open(self, duration_ms: int = 400):
        """Open the gripper (ID 1) — wide open for approach."""
        self.move_servo(1, GRIPPER_OPEN_PULSE, duration_ms)

    def gripper_close(self, duration_ms: int = 400):
        """Close the gripper (ID 1) — cutting/gripping action."""
        self.move_servo(1, GRIPPER_CLOSE_PULSE, duration_ms)

    # ── Search Home position ───────────────────────────────────────────────────
    def go_search_home(self, duration_ms: int = 1000):
        """Move arm to the Search Home position for camera scanning."""
        from arm_controller import SEARCH_HOME_PULSES
        self.move_servos(SEARCH_HOME_PULSES, duration_ms)

    # ── Park (safe rest) ───────────────────────────────────────────────────────
    def go_park(self, duration_ms: int = 1200):
        """Move all servos to neutral (parked / safe)."""
        park = {6: 500, 5: 500, 4: 500, 3: 500, 1: 500}
        self.move_servos(park, duration_ms)

    # ── Query ──────────────────────────────────────────────────────────────────
    def get_position(self, servo_id: int) -> Optional[int]:
        return self.state.get(servo_id, None)

    def get_move_log(self):
        return list(self._move_log)

    def clear_log(self):
        self._move_log.clear()

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def close(self):
        if self.serial_conn:
            self.serial_conn.close()
            logger.info("[SERVO] UART closed")
        if self._gpio_module is not None and self._gpio_handle not in (None, "RPi.GPIO"):
            try:
                self._gpio_module.gpiochip_close(self._gpio_handle)
            except Exception:
                pass
