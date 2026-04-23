import threading, os, time
from ros_robot_controller_sdk import Board
from bus_servo_control import BusServoControl
from action_group_controller import ActionGroupController

board = Board()

agc = ActionGroupController(board)
bsc = BusServoControl(board)

def getServoPulse(servo_id):
    if servo_id == 2:
        return None
    data = bsc.getBusServoPulse(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def getServoDeviation(servo_id):
    if servo_id == 2:
        return None
    data = bsc.getBusServoDeviation(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def setServoPulse(servo_id, pulse, use_time):
    if servo_id == 2:
        return
    bsc.setBusServoPulse(servo_id, pulse, use_time)

def setServoDeviation(servo_id ,dev):
    if servo_id == 2:
        return
    bsc.setBusServoDeviation(servo_id, dev)
    
def saveServoDeviation(servo_id):
    if servo_id == 2:
        return
    bsc.saveBusServoDeviation(servo_id)

def unloadServo(servo_id):
    if servo_id == 2:
        return
    bsc.unloadBusServo(servo_id)

def runActionGroup(num):
    threading.Thread(target=agc.runAction, args=(num, )).start()    

def stopActionGroup():    
    agc.stop_action_group()

def enable_reception(enable=True):
    if enable:
        board.enable_reception(not enable)
        time.sleep(1)
        threading.Thread(target=os.system, args=("/bin/zsh -c 'source $HOME/armpi_pro/src/armpi_pro_bringup/scripts/source_env.bash && rostopic pub /ros_robot_controller/enable_reception std_msgs/Bool \"data: true\"'",), daemon=True).start()
        time.sleep(1)


# --- Move all servos (except ID 2) fully down ---
if __name__ == "__main__":
    print("[INFO] Moving all servos (except ID 2) fully down with extra slow speed and speed safety...")
    # Safe minimum pulses for each servo (based on your SAFE_PULSE_MIN logic)
    safe_min_pulses = {1: 150, 3: 0, 4: 150, 5: 150, 6: 0}
    move_time = 8000  # ms, extra slow for safety
    MIN_SAFE_MOVE_TIME = 2000  # ms, do not allow below this
    if move_time < MIN_SAFE_MOVE_TIME:
        print(f"[WARN] move_time too low ({move_time} ms), setting to minimum safe value {MIN_SAFE_MOVE_TIME} ms")
        move_time = MIN_SAFE_MOVE_TIME
    for sid, pulse in safe_min_pulses.items():
        print(f"[INFO] Moving servo ID {sid} to pulse {pulse} with move_time={move_time} ms")
        setServoPulse(sid, pulse, move_time)
        time.sleep(0.05)







    # --- Move servos to user-requested pose and HOLD ---
    import numpy as np
    from arm_controller import SERVO_IDS, angles_to_pulses

    HOME_POSITION = [0, 0, 0, -np.pi, -np.pi]
    print("[INFO] Moving servos to HOME POSITION: base 0, next 0, next 0, rest -π (excluding ID 2) with speed safety...")
    print("[INFO] This pose is set as the home position.")
    # Angles in radians: [0, 0, 0, -π, -π]
    target_angles = HOME_POSITION
    pulses = angles_to_pulses(target_angles)
    # Remove ID 2 if present
    pulses = {sid: pulse for sid, pulse in pulses.items() if sid != 2}
    # Clamp to safe range
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
        print(f"[INFO] Moving servo ID {sid} to pulse {pulse} for user-requested pose with move_time={move_time} ms")
        setServoPulse(sid, pulse, move_time)
        time.sleep(0.05)
    print("[INFO] All servos commanded to user-requested pose (except ID 2) with speed safety.")
    print("[INFO] Servos will now hold this position. Keeping script alive to maintain hold. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[INFO] Exiting and releasing script hold.")

# --- Remove or comment out all subsequent test code to prevent overwriting the pose ---
# (All test pulses below are now commented out to ensure the arm holds the requested pose)
# Pulse sent to servo 4 (pulse=500)
# Pulse sent to servo 5 (pulse=500) - back to home position
# Pulse sent to servo 5 (pulse=500) with safety margins and slow speed
# Pulse sent to servos [1, 3, 4, 5, 6] (pulse=600) with safety margins and extra slow speed

# Test: send a pulse to servo 4 (not 2)
servo_id = 4
pulse = 500
use_time = 1000

# Example: set position for bus servo
board.bus_servo_set_position(1, [[servo_id, pulse]])
time.sleep(2)
print(f"Pulse sent to servo {servo_id} (pulse={pulse})")

# Test: send a pulse to servo 5 only (ID 2 is unplugged)
servo_id = 5
pulse = 500  # Home position
use_time = 1000

board.bus_servo_set_position(1, [[servo_id, pulse]])
time.sleep(2)
print(f"Pulse sent to servo {servo_id} (pulse={pulse}) - back to home position")

# Safety margins for servo 5
SERVO_ID = 5
PULSE_HOME = 500
PULSE_MIN = 300  # Safe lower bound
PULSE_MAX = 700  # Safe upper bound

# Move to a safe position within margins, slowly
pulse = PULSE_HOME  # Home position
use_time = 3000     # Slower movement (3 seconds)

# Clamp pulse to safety margins
pulse = max(PULSE_MIN, min(PULSE_MAX, pulse))

board.bus_servo_set_position(use_time // 1000, [[SERVO_ID, pulse]])
time.sleep(use_time / 1000)
print(f"Pulse sent to servo {SERVO_ID} (pulse={pulse}) with safety margins and slow speed")

# Test servo 6 with safety margins and extra slow speed
PULSE_MIN = 300
PULSE_MAX = 700
pulse = 600  # Target position for all servos
use_time = 8000  # 8 seconds for extra slow movement
MIN_SAFE_MOVE_TIME = 2000
if use_time < MIN_SAFE_MOVE_TIME:
    print(f"[WARN] use_time too low ({use_time} ms), setting to minimum safe value {MIN_SAFE_MOVE_TIME} ms")
    use_time = MIN_SAFE_MOVE_TIME
# List of servo IDs except 2
servo_ids = [1, 3, 4, 5, 6]
# Clamp pulse to safety margins
pulse = max(PULSE_MIN, min(PULSE_MAX, pulse))
positions = [[sid, pulse] for sid in servo_ids]
board.bus_servo_set_position(use_time // 1000, positions)
time.sleep(use_time / 1000)
print(f"Pulse sent to servos {servo_ids} (pulse={pulse}) with safety margins and extra slow speed")
