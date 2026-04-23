import threading, time
from ros_robot_controller_sdk import Board
from bus_servo_control import BusServoControl
from action_group_controller import ActionGroupController
import numpy as np
from arm_controller import SERVO_IDS, angles_to_pulses

board = Board()
bsc = BusServoControl(board)

def setServoPulse(servo_id, pulse, use_time):
    if servo_id == 2:
        return
    bsc.setBusServoPulse(servo_id, pulse, use_time)

if __name__ == "__main__":
    # Move all servos (except ID 2) to home position
    print("[INFO] Moving all servos (except ID 2) to HOME position with speed safety...")
    HOME_POSITION = [0, 0, 0, -np.pi, -np.pi]
    pulses = angles_to_pulses(HOME_POSITION)
    pulses = {sid: pulse for sid, pulse in pulses.items() if sid != 2}
    SAFE_PULSE_MIN = {1: 150, 3: 0, 4: 150, 5: 150, 6: 0}
    SAFE_PULSE_MAX = {1: 850, 3: 1000, 4: 850, 5: 850, 6: 1000}
    for sid in pulses:
        lo = SAFE_PULSE_MIN.get(sid, 0)
        hi = SAFE_PULSE_MAX.get(sid, 1000)
        pulses[sid] = max(lo, min(hi, pulses[sid]))
    move_time = 8000  # ms, extra slow for safety
    MIN_SAFE_MOVE_TIME = 2000
    if move_time < MIN_SAFE_MOVE_TIME:
        print(f"[WARN] move_time too low ({move_time} ms), setting to minimum safe value {MIN_SAFE_MOVE_TIME} ms")
        move_time = MIN_SAFE_MOVE_TIME
    for sid, pulse in pulses.items():
        print(f"[INFO] Moving servo ID {sid} to pulse {pulse} for HOME position with move_time={move_time} ms")
        setServoPulse(sid, pulse, move_time)
        time.sleep(0.05)
    print("[INFO] All servos commanded to HOME position (except ID 2) with speed safety.")
    print("[INFO] Servos will now hold this position. Keeping script alive to maintain hold. Press Ctrl+C to exit.")
    # Test: send a pulse to the 2nd servo motor (ID 2)
    print("[TEST] Attempting to send a pulse to servo ID 2...")
    test_pulse = 500  # Center position
    test_time = 2000  # 2 seconds
    try:
        bsc.setBusServoPulse(2, test_pulse, test_time)
        print(f"[TEST] Pulse {test_pulse} sent to servo ID 2 with move_time={test_time} ms.")
    except Exception as e:
        print(f"[ERROR] Failed to send pulse to servo ID 2: {e}")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[INFO] Exiting and releasing script hold.")
