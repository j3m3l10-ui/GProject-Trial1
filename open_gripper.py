import time
from ros_robot_controller_sdk import Board
from bus_servo_control import BusServoControl

# Gripper (ID 1) open pulse (use a safe value, e.g., 150 for fully open)
GRIPPER_ID = 1
OPEN_PULSE = 150
MOVE_TIME = 1000  # ms

board = Board()
bsc = BusServoControl(board)

print(f"[GRIPPER] Opening gripper (servo ID {GRIPPER_ID}) to pulse {OPEN_PULSE}...")
bsc.setBusServoPulse(GRIPPER_ID, OPEN_PULSE, MOVE_TIME)
time.sleep(MOVE_TIME / 1000)
print("[GRIPPER] Gripper should now be open.")
