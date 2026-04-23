"""
Servo Driver — 5-DOF Arm on RasAdapter5A (P4 port)
====================================================
Supports three communication backends (tried in order):
  1. Direct UART — LX-16A half-duplex serial protocol on available
     /dev/ttyAMA* ports.  Auto-probes to find the correct one.
  2. I2C relay  — expansion board MCU at 0x34 (register 21 = bus servo,
     register 40 = PWM servo).
  3. SIM mode   — logs commands, no hardware.

Pulse range: 0–1000 (centre 500).

Servo IDs (daisy chain on P4):
  ID 1 = Gripper / Scissors
  ID 3 = Wrist Pitch
  ID 4 = Elbow
  ID 5 = Shoulder
  ID 6 = Base Yaw
"""

import struct
import time
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Pulse limits ──────────────────────────────────────────────────────────────
PULSE_CENTER = 500
PULSE_MIN    = 0
PULSE_MAX    = 1000

SAFE_LIMITS = {
    1: (0, 1000),     # Gripper — full range
    3: (0, 1000),     # Wrist — full range
    4: (150, 850),    # Elbow — restricted
    5: (150, 850),    # Shoulder — restricted
    6: (0, 1000),     # Base — full range
}

# ── LX-16A bus servo commands ─────────────────────────────────────────────────
_CMD_MOVE_TIME_WRITE       = 1
_CMD_POS_READ              = 28
_CMD_LOAD_UNLOAD_WRITE     = 31
_CMD_ID_READ               = 14

# ── I2C constants ─────────────────────────────────────────────────────────────
_I2C_BUS       = 1
_I2C_ADDR      = 0x34
_I2C_REG_BUS   = 21    # bus servo register
_I2C_REG_PWM   = 40    # PWM servo register

# ── UART ports to probe (P4 could be any of these) ───────────────────────────
_UART_PORTS = [
    '/dev/ttyAMA0',    # UART0 — GPIO14/15
    '/dev/ttyAMA3',    # UART3 — GPIO4/5
    '/dev/ttyAMA4',    # UART4 — GPIO8/9
    '/dev/ttyAMA10',   # serial0 alias
    '/dev/ttyAMA2',    # UART2 — GPIO0/1
]
_UART_BAUD = 115200

# SDK UART controller (used by legacy working scripts)
_SDK_UART_PORTS = [
    '/dev/ttyAMA0',
    '/dev/serial0',
    '/dev/ttyS0',
    '/dev/ttyAMA3',
    '/dev/ttyAMA4',
    '/dev/ttyAMA10',
    '/dev/ttyAMA2',
]
_SDK_UART_BAUD = 1000000


# ── LX-16A packet helpers ────────────────────────────────────────────────────
def _lx_packet(servo_id: int, cmd: int, params: bytes = b'') -> bytes:
    """Build an LX-16A half-duplex serial packet."""
    length = len(params) + 3
    data = bytes([servo_id, length, cmd]) + params
    checksum = (~sum(data)) & 0xFF
    return b'\x55\x55' + data + bytes([checksum])


def _lx_move(servo_id: int, position: int, duration_ms: int) -> bytes:
    """Build a SERVO_MOVE_TIME_WRITE packet."""
    params = struct.pack('<HH', position, duration_ms)
    return _lx_packet(servo_id, _CMD_MOVE_TIME_WRITE, params)


def _lx_torque_enable(servo_id: int, enable: bool = True) -> bytes:
    """Build a SERVO_LOAD_OR_UNLOAD_WRITE packet."""
    return _lx_packet(servo_id, _CMD_LOAD_UNLOAD_WRITE,
                      b'\x01' if enable else b'\x00')


class ServoDriver:
    """5-DOF arm driver — auto-detects UART or falls back to I2C."""

    def __init__(self, mode: str = "sim"):
        self.mode = mode.lower()
        self.state: Dict[int, int] = {}
        self._move_log: List[dict] = []
        self._serial = None          # serial.Serial if UART found
        self._sdk_board = None       # ros_robot_controller_sdk.Board instance
        self._backend = "sim"        # "uart", "i2c_bus", "i2c_pwm", "sim"

        if self.mode != "real":
            logger.info("[SERVO] Running in SIMULATION mode")
            return

        # ── Try UART first ────────────────────────────────────────────────
        if self._try_uart():
            return

        # ── Try ROS SDK UART backend (known good in this repo) ───────────
        if self._try_sdk_uart():
            return

        # ── Fall back to I2C relay ────────────────────────────────────────
        if self._try_i2c():
            return

        logger.warning("[SERVO] No servo backend found — falling back to SIM")

    # ── UART auto-probe ───────────────────────────────────────────────────────
    def _try_uart(self) -> bool:
        """Probe each UART port: send ID read, check for any response."""
        try:
            import serial as _serial_mod
        except ImportError:
            logger.warning("[SERVO] pyserial not installed — skipping UART")
            return False

        for port in _UART_PORTS:
            try:
                ser = _serial_mod.Serial(port, _UART_BAUD, timeout=0.1)
                # Send broadcast torque-enable + ID read
                ser.reset_input_buffer()
                ser.write(_lx_torque_enable(254, True))
                ser.flush()
                time.sleep(0.02)
                ser.write(_lx_packet(254, _CMD_ID_READ))
                ser.flush()
                time.sleep(0.05)
                resp = ser.read(100)
                # Look for 0x55 0x55 header in response
                if resp and b'\x55\x55' in resp:
                    self._serial = ser
                    self._backend = "uart"
                    logger.info(f"[SERVO] UART servo detected on {port} "
                                f"(response: {resp.hex()})")
                    self._enable_all_torque()
                    return True
                ser.close()
            except Exception as e:
                logger.debug(f"[SERVO] UART probe {port}: {e}")
        logger.info("[SERVO] No UART servo response on any port")
        return False

    def _try_sdk_uart(self) -> bool:
        """Probe ROS SDK UART transport used by legacy arm scripts."""
        try:
            from ros_robot_controller_sdk import Board as SDKBoard
        except Exception as e:
            logger.debug(f"[SERVO] SDK UART import unavailable: {e}")
            return False

        forced_port = os.getenv("SERVO_SDK_PORT", "").strip()
        ports = [forced_port] if forced_port else list(_SDK_UART_PORTS)

        for port in ports:
            try:
                board = SDKBoard(device=port, baudrate=_SDK_UART_BAUD, timeout=0.2)
                # Send a harmless hold command to validate write path.
                board.bus_servo_set_position(0.3, [[6, 500]])
                self._sdk_board = board
                self._backend = "sdk_uart"
                logger.info(f"[SERVO] Using SDK UART backend on {port} @ {_SDK_UART_BAUD}")
                return True
            except Exception as e:
                logger.debug(f"[SERVO] SDK UART probe {port}: {e}")

        logger.info("[SERVO] No SDK UART backend available")
        return False

    # ── I2C fallback ──────────────────────────────────────────────────────────
    def _try_i2c(self) -> bool:
        """Try I2C bus-servo (reg 21) then PWM-servo (reg 40)."""
        try:
            from smbus2 import SMBus, i2c_msg
            with SMBus(_I2C_BUS) as bus:
                bus.read_byte_data(_I2C_ADDR, 0)
            self._backend = "i2c_bus"
            logger.info(f"[SERVO] Using I2C relay at 0x{_I2C_ADDR:02X} "
                        f"(bus servo register {_I2C_REG_BUS})")
            return True
        except Exception as e:
            logger.warning(f"[SERVO] I2C probe failed: {e}")
            return False

    # ── Enable torque on all known servo IDs ──────────────────────────────────
    def _enable_all_torque(self):
        """Send torque-enable to all arm servo IDs via UART."""
        if self._serial is None:
            return
        for sid in [1, 3, 4, 5, 6]:
            self._serial.write(_lx_torque_enable(sid, True))
            self._serial.flush()
            time.sleep(0.01)
        logger.info("[SERVO] Torque enabled on IDs 1,3,4,5,6")

    # ── Low-level write ───────────────────────────────────────────────────────
    def _hw_write(self, servo_id: int, pulse: int, duration_ms: int):
        """Send a move command via the active backend."""
        if self._backend == "uart" and self._serial:
            self._serial.write(_lx_move(servo_id, pulse, duration_ms))
            self._serial.flush()
        elif self._backend == "sdk_uart" and self._sdk_board is not None:
            self._sdk_board.bus_servo_set_position(duration_ms / 1000.0,
                                                   [[int(servo_id), int(pulse)]])
        elif self._backend.startswith("i2c"):
            reg = _I2C_REG_BUS if self._backend == "i2c_bus" else _I2C_REG_PWM
            try:
                from smbus2 import SMBus, i2c_msg
                buf = ([reg, 1]
                       + list(duration_ms.to_bytes(2, 'little'))
                       + [servo_id]
                       + list(pulse.to_bytes(2, 'little')))
                with SMBus(_I2C_BUS) as bus:
                    msg = i2c_msg.write(_I2C_ADDR, buf)
                    bus.i2c_rdwr(msg)
            except Exception as e:
                logger.error(f"[SERVO] I2C write failed ID{servo_id}: {e}")

    # ── Move a single servo ────────────────────────────────────────────────────
    def move_servo(self, servo_id: int, pulse: int, duration_ms: int = 500):
        lo, hi = SAFE_LIMITS.get(servo_id, (PULSE_MIN, PULSE_MAX))
        pulse = max(lo, min(hi, int(pulse)))
        duration_ms = max(0, min(30000, int(duration_ms)))

        self._move_log.append({"id": servo_id, "pulse": pulse,
                               "duration_ms": duration_ms,
                               "time": time.time()})
        self.state[servo_id] = pulse

        if self.mode == "real":
            self._hw_write(servo_id, pulse, duration_ms)
            logger.debug(f"[SERVO] ID{servo_id} → {pulse}  ({duration_ms}ms) "
                         f"[{self._backend}]")
        else:
            logger.debug(f"[SIM] ID{servo_id} → {pulse}  ({duration_ms}ms)")

    # ── Move multiple servos ──────────────────────────────────────────────────
    def move_servos(self, pulses: Dict[int, int], duration_ms: int = 500):
        if self.mode == "real" and self._backend == "sdk_uart" and self._sdk_board is not None:
            positions = []
            duration_ms = max(0, min(30000, int(duration_ms)))
            for sid, pulse in pulses.items():
                lo, hi = SAFE_LIMITS.get(sid, (PULSE_MIN, PULSE_MAX))
                pulse = max(lo, min(hi, int(pulse)))
                self._move_log.append({"id": sid, "pulse": pulse,
                                       "duration_ms": duration_ms,
                                       "time": time.time()})
                self.state[sid] = pulse
                positions.append([int(sid), int(pulse)])
            try:
                self._sdk_board.bus_servo_set_position(duration_ms / 1000.0, positions)
                logger.debug(f"[SERVO] BATCH {positions} ({duration_ms}ms) [sdk_uart]")
            except Exception as e:
                logger.error(f"[SERVO] SDK batch write failed: {e}")
            return

        for sid, pulse in pulses.items():
            self.move_servo(sid, pulse, duration_ms)
            if self.mode == "real":
                time.sleep(0.02)

    # ── Gripper control ────────────────────────────────────────────────────────
    def gripper_open(self, duration_ms: int = 400):
        self.move_servo(1, 350, duration_ms)

    def gripper_close(self, duration_ms: int = 400):
        self.move_servo(1, 700, duration_ms)

    # ── Preset positions ──────────────────────────────────────────────────────
    def go_search_home(self, duration_ms: int = 1000):
        from arm_controller import SEARCH_HOME_PULSES
        self.move_servos(SEARCH_HOME_PULSES, duration_ms)

    def go_default(self, duration_ms: int = 1000):
        from arm_home_position import get_home_pulses
        self.move_servos(get_home_pulses(), duration_ms)

    def go_park(self, duration_ms: int = 1200):
        from arm_home_position import get_home_pulses
        self.move_servos(get_home_pulses(), duration_ms)

    def wait_for_move(self, duration_ms: int):
        time.sleep((duration_ms + 50) / 1000.0)

    # ── Read servo position (UART only) ───────────────────────────────────────
    def read_position(self, servo_id: int) -> Optional[int]:
        """Read the current position of a servo. Returns pulse or None."""
        if self._backend == "uart" and self._serial:
            self._serial.reset_input_buffer()
            self._serial.write(_lx_packet(servo_id, _CMD_POS_READ))
            self._serial.flush()
            time.sleep(0.05)
            resp = self._serial.read(20)
            if len(resp) >= 8 and resp[0] == 0x55 and resp[1] == 0x55:
                # Skip echo, find response header
                idx = resp.find(b'\x55\x55', 2)
                if idx >= 0 and idx + 7 <= len(resp):
                    pos = struct.unpack_from('<h', resp, idx + 5)[0]
                    return max(0, min(1000, pos))
        return self.state.get(servo_id, None)

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_position(self, servo_id: int) -> Optional[int]:
        return self.state.get(servo_id, None)

    def get_move_log(self):
        return list(self._move_log)

    def clear_log(self):
        self._move_log.clear()

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def close(self):
        if self._sdk_board is not None:
            try:
                self._sdk_board.port.close()
            except Exception:
                pass
            self._sdk_board = None
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        logger.info(f"[SERVO] Driver closed (backend was {self._backend})")
