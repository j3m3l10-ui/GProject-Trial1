import time
import Board

# Move servo with ID 3 to pulse 800
servo_id = 3
pulse = 800
duration = 1000  # ms

print(f"Moving servo {servo_id} to position {pulse}")
Board.setBusServoPulse(servo_id, pulse, duration)
time.sleep(2)
print("Done.")
