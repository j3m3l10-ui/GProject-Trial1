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

import time
import struct
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Protocol constants ─────────────────────────────────────────────────────────
FRAME_HEADER     = bytes([0x55, 0x55])
CMD_SERVO_MOVE   = 1    # SERVO_MOVE_TIME_WRITE
CMD_SERVO_READ   = 28   # SERVO_POS_READ
BROADCAST_ID     = 254

# ── Default UART settings ─────────────────────────────────────────────────────
DEFAULT_UART_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD      = 115200


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


class ServoDriver:
    """Unified servo driver with real/sim backends."""

    def __init__(self, mode: str = "sim", uart_port: str = DEFAULT_UART_PORT,
                 baud: int = DEFAULT_BAUD):
        """
        Args:
            mode: "real" for hardware UART, "sim" for simulation/logging
            uart_port: serial port path (RPi5: /dev/ttyAMA0)
            baud: baud rate
        """
        self.mode = mode.lower()
        self.serial_conn = None
        self.state: Dict[int, int] = {}  # servo_id → current pulse
        self._move_log = []               # history of moves (for GUI playback)

        if self.mode == "real":
            try:
                import serial
                self.serial_conn = serial.Serial(
                    uart_port, baud, timeout=0.5,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                logger.info(f"[SERVO] Opened {uart_port} @ {baud}")
            except ImportError:
                logger.error("[SERVO] pyserial not installed. pip install pyserial")
                raise
            except Exception as e:
                logger.error(f"[SERVO] Cannot open {uart_port}: {e}")
                raise
        else:
            logger.info("[SERVO] Running in SIMULATION mode")

    # ── Move a single servo ────────────────────────────────────────────────────
    def move_servo(self, servo_id: int, pulse: int, duration_ms: int = 500):
        """Move a single servo to the given pulse over duration_ms."""
        pulse = max(0, min(1000, pulse))
        entry = {"id": servo_id, "pulse": pulse, "duration_ms": duration_ms,
                 "time": time.time()}
        self._move_log.append(entry)
        self.state[servo_id] = pulse

        if self.mode == "real" and self.serial_conn:
            pkt = _build_move_cmd(servo_id, pulse, duration_ms)
            self.serial_conn.write(pkt)
            logger.debug(f"[SERVO] ID{servo_id} → {pulse}  ({duration_ms}ms)")
        else:
            logger.debug(f"[SIM] ID{servo_id} → {pulse}  ({duration_ms}ms)")

    # ── Move multiple servos simultaneously ────────────────────────────────────
    def move_servos(self, pulses: Dict[int, int], duration_ms: int = 500):
        """Move multiple servos at once. pulses = {servo_id: pulse_value}."""
        for sid, pulse in pulses.items():
            self.move_servo(sid, pulse, duration_ms)

    # ── Gripper control ────────────────────────────────────────────────────────
    def gripper_open(self, duration_ms: int = 400):
        """Open the gripper (ID 1) — wide open for approach."""
        self.move_servo(1, 200, duration_ms)

    def gripper_close(self, duration_ms: int = 400):
        """Close the gripper (ID 1) — cutting/gripping action."""
        self.move_servo(1, 700, duration_ms)

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
