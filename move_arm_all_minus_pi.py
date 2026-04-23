from arm_controller import FiveDOFArm, SERVO_IDS, angles_to_pulses
from servo_driver import ServoDriver
import numpy as np
import time

# Initialize arm and driver
arm = FiveDOFArm()
driver = ServoDriver(mode="real")  # Use "sim" for simulation if needed

# Set all joint angles to -pi
all_minus_pi = np.full(5, -np.pi)
arm.set_joint_angles(all_minus_pi)
pulses = angles_to_pulses(all_minus_pi)
driver.move_servos(pulses, duration_ms=1200)

print("Arm moved to all -pi angles.")
time.sleep(2)
