"""
5-DOF Robotic Arm Controller — Kinematics + Inverse Kinematics
===============================================================
Generic 5-DOF arm with daisy-chained serial bus servos (LX-16A/LX-225)
connected via port P4 on the RPi5 expansion board.

Servo index → ID mapping (on the daisy chain):
  [0] → ID 6  (Base / Yaw)
  [1] → ID 5  (Shoulder)  — INVERTED
  [2] → ID 4  (Elbow)     — INVERTED
  [3] → ID 3  (Wrist Pitch)
  [4] → ID 1  (Gripper / Scissors)

Bus servo pulse range: 0–1000 (centre 500).
Hand-in-Eye camera mounted at the missing ID2 bracket between wrist (ID3)
and gripper (ID1).
"""

import math
import numpy as np

# ── Servo mapping ──────────────────────────────────────────────────────────────
SERVO_IDS       = [6, 5, 4, 3, 1]   # index → serial bus ID on daisy chain
INVERTED_JOINTS = {1, 2}            # indices whose direction is negated
NEUTRAL_PULSE   = 500               # Bus servo centre (0–1000 range)
PULSE_MIN       = 0
PULSE_MAX       = 1000
SAFE_PULSE_MIN  = {1: 150, 2: 150}  # Shoulder, Elbow restricted
SAFE_PULSE_MAX  = {1: 850, 2: 850}

# Default/Search-Home pulses (validated front-facing home base)
# This pose is also used as the default home base.
SEARCH_HOME_PULSES = {6: 500, 5: 850, 4: 850, 3: 124, 1: 350}

# Pulse-per-radian conversion (~240 steps/rad in 0–1000 range)
PULSE_PER_RAD = 240.0

# ── Link lengths (metres) ─────────────────────────────────────────────────────
# Arm reach (excl. base) ≈ 36.3 cm.  Adjust for your arm.
DEFAULT_LINKS = [0.072, 0.120, 0.110, 0.053, 0.080]
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


def _point_to_segment_distance(point, seg_start, seg_end):
    """Shortest distance from a point to a line segment in 3D."""
    point = np.asarray(point, dtype=float)
    seg_start = np.asarray(seg_start, dtype=float)
    seg_end = np.asarray(seg_end, dtype=float)
    seg = seg_end - seg_start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(point - seg_start))
    t = float(np.dot(point - seg_start, seg) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    closest = seg_start + t * seg
    return float(np.linalg.norm(point - closest))


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

    def _heuristic_seed(self, target, elbow_sign=-1.0):
        """Planar geometric seed to help the Jacobian solver converge."""
        target = np.asarray(target, dtype=float)
        q = np.zeros(5, dtype=float)
        q[0] = math.atan2(target[1], target[0])

        radial = math.hypot(target[0], target[1])
        z = target[2] - self.link_lengths[0]
        l1 = float(self.link_lengths[1])
        l2 = float(self.link_lengths[2] + self.link_lengths[3] + self.link_lengths[4])
        dist = math.hypot(radial, z)
        dist = max(1e-6, min(dist, l1 + l2 - 1e-6))

        cos_elbow = (dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow = elbow_sign * math.acos(cos_elbow)
        shoulder = math.atan2(z, radial) - math.atan2(
            l2 * math.sin(elbow),
            l1 + l2 * math.cos(elbow),
        )
        wrist = -(shoulder + elbow)
        q[1] = shoulder
        q[2] = elbow
        q[3] = wrist
        self.set_joint_angles(q)
        return self.joint_angles.copy()

    def link_clearance(self, point, joint_angles=None, ignore_last_segments=1):
        """Minimum distance from arm links to a point obstacle."""
        positions, _ = self.forward_kinematics(joint_angles)
        segment_count = len(positions) - 1
        active_segments = max(0, segment_count - int(ignore_last_segments))
        distances = []
        for idx in range(active_segments):
            distances.append(_point_to_segment_distance(
                point, positions[idx], positions[idx + 1]))
        if not distances:
            return float("inf")
        return float(min(distances))

    def _collision_cost(self, joint_angles, obstacle_center, obstacle_radius,
                        clearance=0.015, ignore_last_segments=1):
        if obstacle_center is None or obstacle_radius <= 0.0:
            return 0.0
        min_dist = self.link_clearance(
            obstacle_center,
            joint_angles=joint_angles,
            ignore_last_segments=ignore_last_segments,
        )
        safe_dist = float(obstacle_radius) + float(clearance)
        if min_dist >= safe_dist:
            return 0.0
        deficit = safe_dist - min_dist
        return float(deficit * deficit)

    def avoidance_gradient(self, joint_angles, obstacle_center, obstacle_radius,
                           clearance=0.015, eps=1e-4,
                           ignore_last_segments=1):
        """Finite-difference gradient of the collision penalty."""
        q = np.array(joint_angles, dtype=float).copy()
        grad = np.zeros_like(q)
        base_cost = self._collision_cost(
            q,
            obstacle_center,
            obstacle_radius,
            clearance=clearance,
            ignore_last_segments=ignore_last_segments,
        )
        if base_cost <= 0.0:
            return grad
        for idx in range(len(q)):
            dq = np.zeros_like(q)
            dq[idx] = eps
            q_plus = q + dq
            q_minus = q - dq
            q_plus[idx] = np.clip(q_plus[idx], self.joint_limits[idx, 0], self.joint_limits[idx, 1])
            q_minus[idx] = np.clip(q_minus[idx], self.joint_limits[idx, 0], self.joint_limits[idx, 1])
            c_plus = self._collision_cost(
                q_plus,
                obstacle_center,
                obstacle_radius,
                clearance=clearance,
                ignore_last_segments=ignore_last_segments,
            )
            c_minus = self._collision_cost(
                q_minus,
                obstacle_center,
                obstacle_radius,
                clearance=clearance,
                ignore_last_segments=ignore_last_segments,
            )
            grad[idx] = (c_plus - c_minus) / (2.0 * eps)
        return grad

    # ── Damped Least-Squares IK ────────────────────────────────────────────────
    def _ik_refine(self, target, seed, max_iters=300, tol=5e-4, lam=1e-2,
                   obstacle_center=None, obstacle_radius=0.0,
                   obstacle_clearance=0.015, avoid_gain=0.12,
                   joint_center_gain=0.02, max_step=0.18):
        target = np.array(target, dtype=float)
        q = np.array(seed, dtype=float).copy()
        q_mid = np.mean(self.joint_limits, axis=1)
        q_span = np.maximum(self.joint_limits[:, 1] - self.joint_limits[:, 0], 1e-6)
        for it in range(max_iters):
            err_vec = target - self.end_effector_pos(q)
            err_norm = np.linalg.norm(err_vec)
            collision_cost = self._collision_cost(
                q,
                obstacle_center,
                obstacle_radius,
                clearance=obstacle_clearance,
            )
            if err_norm < tol and collision_cost <= 1e-8:
                return True, err_norm, it, q.copy()
            J = self.jacobian(q)
            A = J @ J.T + (lam ** 2) * np.eye(3)
            J_pinv = J.T @ np.linalg.solve(A, np.eye(3))
            dq_primary = J_pinv @ err_vec

            secondary = -joint_center_gain * (q - q_mid) / (q_span * q_span)
            if obstacle_center is not None and obstacle_radius > 0.0:
                secondary -= avoid_gain * self.avoidance_gradient(
                    q,
                    obstacle_center,
                    obstacle_radius,
                    clearance=obstacle_clearance,
                )

            nullspace = np.eye(len(q)) - J_pinv @ J
            dq = dq_primary + nullspace @ secondary
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > max_step:
                dq *= max_step / dq_norm

            current_cost = err_norm + collision_cost
            accepted = False
            for step_scale in (1.0, 0.5, 0.25, 0.1):
                q_candidate = q + dq * step_scale
                for i in range(len(q_candidate)):
                    q_candidate[i] = np.clip(
                        q_candidate[i],
                        self.joint_limits[i, 0],
                        self.joint_limits[i, 1],
                    )
                candidate_err = np.linalg.norm(target - self.end_effector_pos(q_candidate))
                candidate_cost = candidate_err + self._collision_cost(
                    q_candidate,
                    obstacle_center,
                    obstacle_radius,
                    clearance=obstacle_clearance,
                )
                if candidate_cost <= current_cost or candidate_err < err_norm:
                    q = q_candidate
                    accepted = True
                    break
            if not accepted:
                q = q + 0.1 * dq_primary
                for i in range(len(q)):
                    q[i] = np.clip(q[i], self.joint_limits[i, 0],
                                   self.joint_limits[i, 1])
        final_err = np.linalg.norm(target - self.end_effector_pos(q))
        return False, final_err, max_iters, q.copy()

    def inverse_kinematics(self, target, max_iters=300, tol=5e-4, lam=1e-2,
                           obstacle_center=None, obstacle_radius=0.0,
                           obstacle_clearance=0.015, avoid_gain=0.12,
                           joint_center_gain=0.02, max_step=0.18):
        target = np.array(target, dtype=float)
        q_mid = np.mean(self.joint_limits, axis=1)
        seeds = [
            self.joint_angles.copy(),
            self._heuristic_seed(target, elbow_sign=-1.0),
            self._heuristic_seed(target, elbow_sign=1.0),
            q_mid,
        ]

        best = {
            "solved": False,
            "err": float("inf"),
            "iters": max_iters,
            "q": self.joint_angles.copy(),
        }

        for seed in seeds:
            solved, err, iters, q_sol = self._ik_refine(
                target,
                seed,
                max_iters=max_iters,
                tol=tol,
                lam=lam,
                obstacle_center=obstacle_center,
                obstacle_radius=obstacle_radius,
                obstacle_clearance=obstacle_clearance,
                avoid_gain=avoid_gain,
                joint_center_gain=joint_center_gain,
                max_step=max_step,
            )
            if err < best["err"]:
                best.update({
                    "solved": solved,
                    "err": err,
                    "iters": iters,
                    "q": q_sol,
                })
            if solved:
                self.set_joint_angles(q_sol)
                return True, err, iters

        self.set_joint_angles(best["q"])
        return False, best["err"], best["iters"]

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


def compute_stem_cut_point(tomato_center_m, tomato_radius_m, arm_base,
                           stem_up_offset=0.01):
    """
    Compute the cut point targeting the stem at the top of the tomato.

    The stem is assumed to attach at the top (positive Z) of the tomato.
    The end-effector approaches from the arm base direction but biased
    upward to align the scissors with the stem.

    Returns (stem_point, cut_point) both as numpy arrays.
    """
    center = np.array(tomato_center_m, dtype=float)
    base = np.array(arm_base, dtype=float)

    # Stem target: top of the tomato, slightly biased upward
    stem_target = center.copy()
    stem_target[2] += tomato_radius_m * 0.5  # bias toward stem

    # Approach direction from base to stem target
    direction = stem_target - base
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / dist

    # Edge of tomato nearest to base
    edge_point = center - direction * tomato_radius_m
    # Cut point: 1 cm before the edge + upward stem bias
    cut_point = center - direction * (tomato_radius_m + 0.01)
    cut_point[2] += stem_up_offset  # offset upward toward stem

    return edge_point, cut_point
