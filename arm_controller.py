"""
5-DOF Robotic Arm Controller — Kinematics + Inverse Kinematics
===============================================================
Adapted for Hiwonder ArmPi Pro (modified to 5DOF from 6DOF).

Servo index → Serial Bus ID mapping:
  [0] → ID 6  (Base / Yaw)
  [1] → ID 5  (Shoulder)  — INVERTED, pulse limits 150–850
  [2] → ID 4  (Elbow)     — INVERTED, pulse limits 150–850
  [3] → ID 3  (Wrist Pitch)
  [4] → ID 1  (Gripper)   — independent of IK chain

Neutral pulse = 500 (0 radians).
Hand-in-Eye camera between ID3 and ID1.
Search Home pulses: ID6:500, ID5:700, ID4:250, ID3:400
"""

import math
import numpy as np

# ── Servo mapping ──────────────────────────────────────────────────────────────
SERVO_IDS       = [6, 5, 4, 3, 1]   # index → serial bus ID
INVERTED_JOINTS = {1, 2}            # indices whose direction is negated
NEUTRAL_PULSE   = 500
PULSE_MIN       = 0
PULSE_MAX       = 1000
SAFE_PULSE_MIN  = {1: 150, 2: 150}  # IDs 5,4 need tighter limits
SAFE_PULSE_MAX  = {1: 850, 2: 850}

# Search-Home pulses (arm looks outward for camera to scan)
SEARCH_HOME_PULSES = {6: 500, 5: 700, 4: 250, 3: 400, 1: 500}

# Pulse-per-radian conversion (Hiwonder LX-series ≈ 240 steps/rad)
PULSE_PER_RAD = 240.0

# ── Link lengths (metres) ─────────────────────────────────────────────────────
DEFAULT_LINKS = [0.072, 0.104, 0.096, 0.046, 0.070]
#                base↑  shoulder  elbow   wrist   tool/gripper


def rodrigues(axis, theta):
    """Rotation matrix via Rodrigues (quaternion shortcut)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
        [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]
    ], dtype=float)


class FiveDOFArm:
    """5-DOF robotic arm with forward kinematics, numerical Jacobian, and DLS IK."""

    def __init__(self, link_lengths=None):
        self.link_lengths = np.array(link_lengths or DEFAULT_LINKS, dtype=float)
        self.joint_angles = np.zeros(5, dtype=float)
        self.joint_limits = np.array([
            [-np.pi,      np.pi],        # base yaw
            [-math.pi/2,  math.pi/2],    # shoulder
            [-math.pi/2,  math.pi/2],    # elbow
            [-math.pi/2,  math.pi/2],    # wrist pitch
            [-math.pi/2,  math.pi/2],    # gripper (not used by IK)
        ], dtype=float)
        self.base_pos = np.array([0.0, 0.0, 0.0], dtype=float)

    # ── Forward kinematics ─────────────────────────────────────────────────────
    def forward_kinematics(self, joint_angles=None):
        if joint_angles is None:
            joint_angles = self.joint_angles
        q = np.asarray(joint_angles, dtype=float)
        pos = self.base_pos.copy()
        R = np.eye(3, dtype=float)
        positions = [pos.copy()]
        rotations = [R.copy()]

        # Joint 0: base yaw (z-axis)
        R = R.dot(rodrigues([0, 0, 1], q[0]))
        pos = pos + R.dot([0, 0, self.link_lengths[0]])
        positions.append(pos.copy()); rotations.append(R.copy())

        # Joint 1: shoulder (y-axis)
        R = R.dot(rodrigues([0, 1, 0], q[1]))
        pos = pos + R.dot([self.link_lengths[1], 0, 0])
        positions.append(pos.copy()); rotations.append(R.copy())

        # Joint 2: elbow (y-axis)
        R = R.dot(rodrigues([0, 1, 0], q[2]))
        pos = pos + R.dot([self.link_lengths[2], 0, 0])
        positions.append(pos.copy()); rotations.append(R.copy())

        # Joint 3: wrist pitch (y-axis)
        R = R.dot(rodrigues([0, 1, 0], q[3]))
        pos = pos + R.dot([self.link_lengths[3], 0, 0])
        positions.append(pos.copy()); rotations.append(R.copy())

        # Joint 4: gripper roll (x-axis)
        R = R.dot(rodrigues([1, 0, 0], q[4]))
        pos = pos + R.dot([self.link_lengths[4], 0, 0])
        positions.append(pos.copy()); rotations.append(R.copy())

        return positions, rotations

    def end_effector_pos(self, joint_angles=None):
        positions, _ = self.forward_kinematics(joint_angles)
        return np.array(positions[-1], dtype=float)

    # ── Joint handling ─────────────────────────────────────────────────────────
    def set_joint_angles(self, angles):
        angles = np.array(angles, dtype=float)
        for i in range(len(angles)):
            angles[i] = np.clip(angles[i], self.joint_limits[i, 0],
                                self.joint_limits[i, 1])
        self.joint_angles = angles

    # ── Numerical Jacobian ─────────────────────────────────────────────────────
    def jacobian(self, joint_angles=None, eps=1e-6):
        q = self.joint_angles.copy() if joint_angles is None \
            else np.array(joint_angles, dtype=float).copy()
        J = np.zeros((3, len(q)), dtype=float)
        f0 = self.end_effector_pos(q)
        for j in range(len(q)):
            dq = np.zeros_like(q); dq[j] = eps
            J[:, j] = (self.end_effector_pos(q + dq) - f0) / eps
        return J

    # ── Damped Least-Squares IK ────────────────────────────────────────────────
    # Named constants for the IK algorithm to aid tuning and readability
    _IK_INIT_ELBOW_BEND      = 0.3   # radians — moderate initial elbow bend
    _IK_DAMPING_ERR_SCALE    = 0.05  # metres  — error level at which damping = lam
    _IK_MAX_STEP_RAD         = 0.2   # radians — per-iteration step clamp

    def initialise_for_target(self, target):
        """
        Set a good initial joint configuration before running IK.

        Aligns the base yaw toward the horizontal direction of the target and
        sets a moderate shoulder/elbow configuration so the DLS solver starts
        from a configuration that is already pointing roughly at the target.
        This is critical for left/right/corner targets where starting from the
        Search Home angles can trap the solver in a poor local minimum.
        """
        target = np.array(target, dtype=float)
        q = self.joint_angles.copy()

        # Base yaw: point directly at the target in the XY plane
        q[0] = math.atan2(target[1], target[0])

        # Shoulder: half the geometric pitch toward target height — halved to
        # avoid over-pitching on the initial estimate before IK refinement.
        dist_xy = math.sqrt(target[0] ** 2 + target[1] ** 2)
        q[1] = np.clip(-math.atan2(target[2], max(dist_xy, 1e-6)) * 0.5,
                       self.joint_limits[1, 0], self.joint_limits[1, 1])

        # Elbow: moderate forward bend (initial estimate, refined by IK)
        q[2] = np.clip(self._IK_INIT_ELBOW_BEND,
                       self.joint_limits[2, 0], self.joint_limits[2, 1])

        # Wrist: level
        q[3] = 0.0
        q[4] = 0.0
        self.set_joint_angles(q)

    def inverse_kinematics(self, target, max_iters=400, tol=5e-4, lam=5e-3):
        """
        Damped Least-Squares (DLS) inverse kinematics with adaptive damping.

        Adaptive damping reduces λ as the error decreases so the solver is
        stable when far from the target and precise when close to it.
        """
        target = np.array(target, dtype=float)
        q = self.joint_angles.copy()
        for it in range(max_iters):
            err_vec = target - self.end_effector_pos(q)
            err_norm = np.linalg.norm(err_vec)
            if err_norm < tol:
                self.set_joint_angles(q)
                return True, err_norm, it

            # Adaptive damping: scale λ up when error is large (stability),
            # scale it down when error is small (precision).
            lam_adaptive = lam * max(err_norm / self._IK_DAMPING_ERR_SCALE, 1.0)

            J = self.jacobian(q)
            A = J @ J.T + (lam_adaptive ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err_vec)

            # Clamp the step to avoid overshooting
            step_norm = np.linalg.norm(dq)
            if step_norm > self._IK_MAX_STEP_RAD:
                dq = dq * (self._IK_MAX_STEP_RAD / step_norm)

            q += dq
            for i in range(len(q)):
                q[i] = np.clip(q[i], self.joint_limits[i, 0],
                                self.joint_limits[i, 1])
        self.set_joint_angles(q)
        final_err = np.linalg.norm(target - self.end_effector_pos())
        return False, final_err, max_iters

    # ── Workspace reach check ──────────────────────────────────────────────────
    def max_reach(self):
        return float(np.sum(self.link_lengths))

    def is_reachable(self, target):
        return np.linalg.norm(np.array(target) - self.base_pos) <= self.max_reach()


# ── Pulse conversion helpers ───────────────────────────────────────────────────

def radians_to_pulse(joint_idx, angle_rad):
    """Convert a joint angle (radians) to a servo pulse value."""
    sign = -1 if joint_idx in INVERTED_JOINTS else 1
    pulse = NEUTRAL_PULSE + int(sign * angle_rad * PULSE_PER_RAD)
    lo = SAFE_PULSE_MIN.get(joint_idx, PULSE_MIN)
    hi = SAFE_PULSE_MAX.get(joint_idx, PULSE_MAX)
    return max(lo, min(hi, pulse))


def pulse_to_radians(joint_idx, pulse):
    """Convert a servo pulse to radians."""
    sign = -1 if joint_idx in INVERTED_JOINTS else 1
    return sign * (pulse - NEUTRAL_PULSE) / PULSE_PER_RAD


def angles_to_pulses(joint_angles):
    """Convert all 5 joint angles to a dict {servo_id: pulse}."""
    pulses = {}
    for idx, angle in enumerate(joint_angles):
        sid = SERVO_IDS[idx]
        pulses[sid] = radians_to_pulse(idx, angle)
    return pulses


def search_home_angles():
    """Return joint angles (radians) corresponding to Search Home position."""
    angles = np.zeros(5, dtype=float)
    for idx, sid in enumerate(SERVO_IDS):
        if sid in SEARCH_HOME_PULSES:
            angles[idx] = pulse_to_radians(idx, SEARCH_HOME_PULSES[sid])
    return angles


# ── Cut-point computation ──────────────────────────────────────────────────────

def compute_cut_point(tomato_center_m, tomato_radius_m, arm_base):
    """
    Given tomato position in arm-frame (metres), compute the point
    1 cm away from the tomato surface toward the base (stem cut location).
    Returns (edge_point, cut_point) both as numpy arrays.
    """
    base = np.array(arm_base, dtype=float)
    center = np.array(tomato_center_m, dtype=float)
    direction = center - base
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / dist
    edge_point = center - direction * tomato_radius_m
    cut_gap = 0.01  # 1 cm
    cut_point = center - direction * (tomato_radius_m + cut_gap)
    return edge_point, cut_point
