import time
from arm_controller import SERVO_IDS, angles_to_pulses
import Board
import numpy as np

# Move all joints to -pi radians (within joint limits)
angles = np.full(5, -np.pi)
# Convert angles to pulses for each servo
pulses = angles_to_pulses(angles)

print("Moving all arm joints to -pi position (within safe limits):")
for sid in SERVO_IDS:
    pulse = pulses[sid]
    print(f"Servo {sid}: pulse {pulse}")
    Board.setBusServoPulse(sid, pulse, 1200)
    time.sleep(0.5)

print("Done.")
