import time
from ros_robot_controller_sdk import Board
from bus_servo_control import BusServoControl

# Safe pulse limits for each servo (based on your SAFE_PULSE_MIN/MAX)
safe_min_pulses = {1: 150, 3: 0, 4: 150, 5: 150, 6: 0}
safe_max_pulses = {1: 850, 3: 1000, 4: 850, 5: 850, 6: 1000}
servo_ids = [1, 3, 4, 5, 6]
move_time = 8000  # ms, extra slow for safety

board = Board()
bsc = BusServoControl(board)

print("[TEST] Moving all servos to their minimum safe positions...")
for sid in servo_ids:
    pulse = safe_min_pulses[sid]
    print(f"[TEST] Moving servo ID {sid} to pulse {pulse} (min) with move_time={move_time} ms")
    bsc.setBusServoPulse(sid, pulse, move_time)
    time.sleep(0.1)
time.sleep(move_time / 1000)

print("[TEST] Moving all servos to their maximum safe positions...")
for sid in servo_ids:
    pulse = safe_max_pulses[sid]
    print(f"[TEST] Moving servo ID {sid} to pulse {pulse} (max) with move_time={move_time} ms")
    bsc.setBusServoPulse(sid, pulse, move_time)
    time.sleep(0.1)
time.sleep(move_time / 1000)

print("[TEST] Done. If the arm did not move, check power, wiring, and servo IDs.")
