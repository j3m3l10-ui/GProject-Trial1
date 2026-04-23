import time
from Board import Board
from bus_servo_control import BusServoControl

# Initialize the board and bus servo control
board = Board()
bus_servo = BusServoControl(board)

# Example: Move all bus servos (IDs 1, 3, 4, 5, 6) to pulse 250, then to 750
servo_ids = [1, 3, 4, 5, 6]
positions = [250, 750]
duration = 1000  # ms

for pos in positions:
    for sid in servo_ids:
        print(f"Moving servo {sid} to position {pos}")
        bus_servo.setBusServoPulse(sid, pos, duration)
    time.sleep(2)

print("Bus servos moved to test positions.")
