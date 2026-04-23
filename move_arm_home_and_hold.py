import time
from ros_robot_controller_sdk import Board
from bus_servo_control import BusServoControl
from arm_controller import SERVO_IDS
from arm_home_position import get_home_pulses

# Use stored default home pulses directly.
pulses = get_home_pulses()

# Remove ID 2 if present (not used)
if 2 in pulses:
    del pulses[2]

# Clamp to safe pulse ranges
SAFE_PULSE_MIN = {1: 150, 3: 0, 4: 150, 5: 150, 6: 0}
SAFE_PULSE_MAX = {1: 850, 3: 1000, 4: 850, 5: 850, 6: 1000}
for sid in pulses:
    pulses[sid] = max(SAFE_PULSE_MIN.get(sid, 0), min(SAFE_PULSE_MAX.get(sid, 1000), pulses[sid]))

move_time = 8000  # ms, extra slow for safety

board = Board()
bsc = BusServoControl(board)

print("[HOME] Moving all servos to forward-facing home position (gripper open) and holding...")
for sid in SERVO_IDS:
    if sid == 2:
        continue
    pulse = pulses[sid]
    print(f"[HOME] Moving servo ID {sid} to pulse {pulse} (home) with move_time={move_time} ms")
    bsc.setBusServoPulse(sid, pulse, move_time)
    time.sleep(0.1)
time.sleep(move_time / 1000)

print("[HOME] Holding position. Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("[HOME] Exiting and releasing hold.")
