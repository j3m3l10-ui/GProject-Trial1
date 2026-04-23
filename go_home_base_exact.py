import time
from ros_robot_controller_sdk import Board
from bus_servo_control import BusServoControl

# Exact validated home base pose (do not modify unless recalibrated)
HOME_BASE_PULSES = {
    6: 500,
    5: 850,
    4: 850,
    3: 124,
    1: 350,
}

SAFE_PULSE_MIN = {1: 150, 3: 0, 4: 150, 5: 150, 6: 0}
SAFE_PULSE_MAX = {1: 850, 3: 1000, 4: 850, 5: 850, 6: 1000}
SERVO_IDS = [6, 5, 4, 3, 1]
MOVE_TIME_MS = 8000  # slow move for safety


def clamp_pulse(servo_id: int, pulse: int) -> int:
    lo = SAFE_PULSE_MIN.get(servo_id, 0)
    hi = SAFE_PULSE_MAX.get(servo_id, 1000)
    return max(lo, min(hi, int(pulse)))


def main() -> None:
    board = Board()
    bsc = BusServoControl(board)

    print("[HOME_BASE] Moving to exact validated home base pose...")
    for sid in SERVO_IDS:
        pulse = clamp_pulse(sid, HOME_BASE_PULSES[sid])
        print(f"[HOME_BASE] ID {sid} -> {pulse} ({MOVE_TIME_MS} ms)")
        bsc.setBusServoPulse(sid, pulse, MOVE_TIME_MS)
        time.sleep(0.1)

    time.sleep(MOVE_TIME_MS / 1000.0)
    print("[HOME_BASE] Reached exact home base pose.")


if __name__ == "__main__":
    main()
