import numpy as np
from arm_controller import SERVO_IDS, pulse_to_radians

# Exact validated default home pose in servo pulses.
HOME_PULSES = {6: 500, 5: 850, 4: 850, 3: 124, 1: 350}

# Derive angles from the stored pulse home pose for IK/state users.
HOME_ANGLES = np.array([
    pulse_to_radians(idx, HOME_PULSES[sid])
    for idx, sid in enumerate(SERVO_IDS)
], dtype=float)

# Remove ID 2 if present
if 2 in HOME_PULSES:
    del HOME_PULSES[2]

# Clamp to safe pulse ranges
SAFE_PULSE_MIN = {1: 150, 3: 0, 4: 150, 5: 150, 6: 0}
SAFE_PULSE_MAX = {1: 850, 3: 1000, 4: 850, 5: 850, 6: 1000}
for sid in HOME_PULSES:
    HOME_PULSES[sid] = max(SAFE_PULSE_MIN.get(sid, 0), min(SAFE_PULSE_MAX.get(sid, 1000), HOME_PULSES[sid]))

# Export for use in other scripts
def get_home_angles():
    return HOME_ANGLES.copy()

def get_home_pulses():
    return HOME_PULSES.copy()
