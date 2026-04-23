from servo_driver import ServoDriver
import time

if __name__ == "__main__":
    driver = ServoDriver(mode="real")  # Use "sim" for simulation
    servo_id = 6  # Base/Yaw
    pulse = 200   # Move to a visible position (adjust as needed)
    duration_ms = 1200
    print(f"Moving servo {servo_id} to pulse {pulse}")
    driver.move_servo(servo_id, pulse, duration_ms)
    time.sleep(2)
    print("Done.")
