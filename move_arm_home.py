from arm_controller import FiveDOFArm
from arm_home_position import get_home_angles, get_home_pulses
from servo_driver import ServoDriver
import time

# Initialize arm and driver
arm = FiveDOFArm()
driver = ServoDriver(mode="real")  # Use "sim" for simulation if needed

# Get front-facing home angles and move arm
home_angles = get_home_angles()
arm.set_joint_angles(home_angles)
driver.move_servos(get_home_pulses(), duration_ms=1200)

print("Arm moved to home position.")
time.sleep(2)
