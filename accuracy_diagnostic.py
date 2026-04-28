"""
Accuracy Diagnostic & Correction Pipeline
==========================================
5-DOF Tomato Harvesting Arm — standalone, no ROS dependency.

Usage:
    python accuracy_diagnostic.py            # run all 3 steps
    python accuracy_diagnostic.py --step 1   # diagnostics only
    python accuracy_diagnostic.py --step 2   # auto-fix only
    python accuracy_diagnostic.py --step 3   # validate before vs after
    python accuracy_diagnostic.py --fix-links 120 110 53 80   # supply measured link mm
    python accuracy_diagnostic.py --hand-eye  # run hand-eye calibration routine
    python accuracy_diagnostic.py --calibrate-camera  # camera calibration instructions
"""

import argparse
import json
import math
import os
import sys
import numpy as np

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import (
    FiveDOFArm, DEFAULT_LINKS, SEARCH_HOME_PULSES, SERVO_IDS,
    search_home_angles, pulse_to_radians, radians_to_pulse,
)

# ── colour helpers ───────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(msg):    print(f"  {_GREEN}[OK]  {_RESET}{msg}")
def _warn(msg):  print(f"  {_YELLOW}[WARN]{_RESET} {msg}")
def _fail(msg):  print(f"  {_RED}[FAIL]{_RESET} {msg}")
def _info(msg):  print(f"        {msg}")
def _hdr(title): print(f"\n{_BOLD}{'─'*62}\n  {title}\n{'─'*62}{_RESET}")

# ── thresholds ───────────────────────────────────────────────────────────────
FK_WARN_MM         = 5.0    # mm — FK round-trip error worth flagging
IK_WARN_MM         = 2.0    # mm — IK round-trip residual threshold
COND_WARN          = 50.0   # Jacobian condition number warning
REPROJ_WARN_PX     = 0.5    # pixels
HAND_EYE_WARN_MM   = 3.0    # mm
JAC_CORR_THRESH_M  = 0.001  # 1 mm target
JAC_CORR_MAX_ITERS = 50
JAC_CORR_LAM       = 0.01   # DLS damping for correction loop

# ── file paths ───────────────────────────────────────────────────────────────
_DIR              = os.path.dirname(os.path.abspath(__file__))
HAND_EYE_FILE     = os.path.join(_DIR, "hand_eye_transform.json")
CALIB_FILES       = [
    os.path.join(_DIR, "camera_calibration.json"),
    os.path.join(_DIR, "camera_calibration.npz"),
]
HAND_EYE_POSES    = os.path.join(_DIR, "hand_eye_poses.json")
ARM_BIAS_FILE     = os.path.join(_DIR, "arm_bias_calibration.json")

# ═══════════════════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════════════════

# Five joint configs covering the normal harvest workspace (radians).
# NOTE: all-zeros config is a geometric singularity (fully-extended arm);
#       use search-home and realistic working configs instead.
KNOWN_CONFIGS = np.array([
    [ 0.00, -1.40,  1.40, -0.31, 0.00],  # search-home equivalent
    [ 0.00, -0.52,  0.52,  0.00, 0.00],  # elbow raised ~30°
    [ 0.52,  0.00, -0.26,  0.26, 0.00],  # yaw 30°, mild flex
    [ 0.00, -0.79,  0.79, -0.26, 0.00],  # elbow ~45°, wrist back
    [-0.52,  0.26, -0.26,  0.00, 0.00],  # negative yaw
], dtype=float)

# ⚠  Replace with ruler measurements for real-world FK accuracy.
# Format: [[x,y,z], ...] metres, same order as KNOWN_CONFIGS.
# Leave as None to run model-internal round-trips instead.
MEASURED_POSITIONS_M = None

# IK round-trip targets (arm-frame, metres)
IK_TEST_TARGETS = np.array([
    [0.25,  0.00, 0.10],
    [0.28, -0.05, 0.08],
    [0.22,  0.04, 0.12],
    [0.30,  0.00, 0.07],
    [0.20, -0.03, 0.15],
], dtype=float)

# Hand-eye verification: 3D points in camera frame (metres).
# ⚠  Replace both arrays with real measurements to enable residual check.
HAND_EYE_TEST_POINTS_CAM = np.array([
    [ 0.05,  0.02, 0.25],
    [-0.03,  0.01, 0.30],
    [ 0.00,  0.00, 0.20],
    [ 0.04, -0.02, 0.28],
    [-0.02,  0.03, 0.22],
], dtype=float)
HAND_EYE_TEST_POINTS_ARM = None  # set to measured arm-frame positions to enable


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _rot_to_euler_deg(R: np.ndarray) -> np.ndarray:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def _load_hand_eye() -> np.ndarray:
    if os.path.isfile(HAND_EYE_FILE):
        try:
            with open(HAND_EYE_FILE) as f:
                data = json.load(f)
            T = np.array(data["T"], dtype=float)
            if T.shape == (4, 4):
                return T
        except Exception as e:
            _warn(f"Could not parse {HAND_EYE_FILE}: {e}")
    return np.eye(4, dtype=float)


def _robust_axis_bias_mm(samples_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(samples_mm, axis=0)
    mad = np.median(np.abs(samples_mm - median), axis=0)
    return median, mad


def _run_bias_calibration():
    _hdr("ARM BIAS CALIBRATION")
    _info("Enter the correction the robot should have taken from the landed cut point")
    _info("to reach the true stem, in arm-frame millimetres: dx dy dz")
    _info("Example: if the blade landed 4mm low, enter: 0 0 4")
    _info("Enter at least 3 samples. Press Enter on an empty line to finish.")

    samples = []
    while True:
        raw = input(f"sample {len(samples)+1} dx dy dz (mm): ").strip()
        if not raw:
            if len(samples) >= 3:
                break
            _warn("Need at least 3 samples before finishing.")
            continue

        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            _warn("Please enter exactly 3 numbers: dx dy dz")
            continue
        try:
            vec = [float(v) for v in parts]
        except ValueError:
            _warn("Could not parse numbers. Try again.")
            continue
        samples.append(vec)

    samples_mm = np.array(samples, dtype=float)
    bias_mm, mad_mm = _robust_axis_bias_mm(samples_mm)
    abs_err_mm = np.linalg.norm(samples_mm - bias_mm, axis=1)
    rms_mm = float(np.sqrt(np.mean(abs_err_mm ** 2)))
    payload = {
        "bias_m": (bias_mm / 1000.0).tolist(),
        "bias_mm": bias_mm.tolist(),
        "axis_mad_mm": mad_mm.tolist(),
        "rms_residual_mm": rms_mm,
        "num_samples": int(samples_mm.shape[0]),
        "samples_mm": samples_mm.tolist(),
        "notes": "Positive dx/dy/dz means the robot should have moved further in +X/+Y/+Z arm frame.",
    }
    with open(ARM_BIAS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _ok(f"Saved arm bias calibration to {ARM_BIAS_FILE}")
    _ok(f"Estimated bias: dx={bias_mm[0]:.2f} dy={bias_mm[1]:.2f} dz={bias_mm[2]:.2f} mm")
    _info(f"Axis MAD: {np.round(mad_mm, 2).tolist()} mm")
    _info(f"Residual RMS around bias: {rms_mm:.2f} mm")
    _info("main.py will load this file automatically on startup.")


def _jacobian_correction(arm: FiveDOFArm, q_start: np.ndarray,
                          target: np.ndarray) -> tuple:
    """
    Iterative Jacobian correction loop (damped least-squares).
    Returns (converged, q_final, final_err_mm, iterations).
    """
    q = q_start.copy()
    last_err = float("inf")
    for it in range(1, JAC_CORR_MAX_ITERS + 1):
        p_curr = arm.end_effector_pos(q)
        delta_p = target - p_curr
        err = float(np.linalg.norm(delta_p))
        last_err = err
        if err <= JAC_CORR_THRESH_M:
            return True, q, err * 1000.0, it
        J = arm.jacobian(q)
        A = J @ J.T + (JAC_CORR_LAM ** 2) * np.eye(3)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3))
        delta_q = J_pinv @ delta_p
        step = float(np.linalg.norm(delta_q))
        if step > 0.15:
            delta_q *= 0.15 / step
        q_next = q + delta_q
        for j in range(len(q_next)):
            q_next[j] = np.clip(q_next[j],
                                arm.joint_limits[j, 0], arm.joint_limits[j, 1])
        q = q_next
    return False, q, last_err * 1000.0, JAC_CORR_MAX_ITERS


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DIAGNOSE
# ═══════════════════════════════════════════════════════════════════════════════

def step1_diagnose(arm: FiveDOFArm) -> dict:
    """Run all diagnostic checks. Returns results dict for Step 2/3."""
    _hdr("STEP 1 — ACCURACY DIAGNOSTICS")
    results = {}

    # ── 1a. FK accuracy check ─────────────────────────────────────────────
    print(f"\n{_BOLD}1. FK Accuracy Check{_RESET}")
    fk_errors_mm = []
    for i, q in enumerate(KNOWN_CONFIGS):
        p_fk = arm.end_effector_pos(q)
        if MEASURED_POSITIONS_M is not None:
            p_ref = np.array(MEASURED_POSITIONS_M[i], dtype=float)
            err_mm = float(np.linalg.norm(p_fk - p_ref)) * 1000.0
            tag = "vs measured"
        else:
            # model round-trip: IK→FK
            _, _, _, q_rt = arm._ik_refine(p_fk, seed=q, max_iters=300, tol=1e-5)
            p_rt = arm.end_effector_pos(q_rt)
            err_mm = float(np.linalg.norm(p_fk - p_rt)) * 1000.0
            tag = "round-trip"
        fk_errors_mm.append(err_mm)
        _info(f"Config {i+1}: p_fk={np.round(p_fk*1000,1).tolist()} mm  "
              f"{tag} err={err_mm:.2f} mm")

    max_fk = max(fk_errors_mm)
    results["fk_max_err_mm"] = max_fk
    if MEASURED_POSITIONS_M is None:
        _info("(No measured positions supplied — model round-trip only)")
        _info(" → Set MEASURED_POSITIONS_M in accuracy_diagnostic.py for real values.")
    (_ok if max_fk <= FK_WARN_MM else _warn)(
        f"FK max error: {max_fk:.2f} mm")

    # ── 1b. IK round-trip check ───────────────────────────────────────────
    print(f"\n{_BOLD}2. IK Round-Trip Accuracy Check{_RESET}")
    ik_errors_mm = []
    for i, target in enumerate(IK_TEST_TARGETS):
        # Use multi-seed IK for a fair round-trip test
        solved, err_m, iters = arm.inverse_kinematics(
            target, max_iters=500, tol=1e-5)
        q_sol = arm.joint_angles.copy()
        p_fk = arm.end_effector_pos(q_sol)
        err_mm = float(np.linalg.norm(p_fk - target)) * 1000.0
        ik_errors_mm.append(err_mm)
        flag = " ⚠" if err_mm > IK_WARN_MM else ""
        _info(f"Target {i+1}: {np.round(target*1000,1).tolist()} mm  "
              f"→ err={err_mm:.2f} mm  (solved={solved}, iters={iters}){flag}")

    max_ik = max(ik_errors_mm)
    results["ik_max_err_mm"] = max_ik
    (_ok if max_ik <= IK_WARN_MM else _warn)(
        f"IK round-trip error: {max_ik:.2f} mm")

    # ── 1c. Camera intrinsic check ────────────────────────────────────────
    print(f"\n{_BOLD}3. Camera Intrinsic Check{_RESET}")
    reproj_err = None
    for cf in CALIB_FILES:
        if not os.path.isfile(cf):
            continue
        try:
            if cf.endswith(".json"):
                with open(cf) as f:
                    cal = json.load(f)
                reproj_err = float(cal.get("reprojection_error",
                                           cal.get("rms", -1)))
            else:
                cal = np.load(cf)
                reproj_err = float(cal.get("rms",
                                           cal.get("reprojection_error", -1)))
            _info(f"Loaded: {cf}")
            break
        except Exception as e:
            _warn(f"Could not load {cf}: {e}")

    results["reproj_err_px"] = reproj_err
    if reproj_err is None:
        _warn("No camera calibration file found.")
        _info(" → Run: python accuracy_diagnostic.py --calibrate-camera")
    elif reproj_err < 0:
        _warn("Reprojection error field missing from calibration file.")
    else:
        (_ok if reproj_err <= REPROJ_WARN_PX else _warn)(
            f"Reprojection error: {reproj_err:.3f} px")

    # ── 1d. Hand-eye calibration check ───────────────────────────────────
    print(f"\n{_BOLD}4. Hand-Eye Calibration Check{_RESET}")
    T_he = _load_hand_eye()
    he_identity = np.allclose(T_he, np.eye(4), atol=1e-6)

    if he_identity:
        _warn("Hand-eye transform is identity — not calibrated.")
        _info(" → Run: python accuracy_diagnostic.py --hand-eye")
        results["hand_eye_residual_mm"] = float("inf")
    else:
        _info(f"Loaded from {HAND_EYE_FILE}")
        _info(f"  Translation : {np.round(T_he[:3,3]*1000, 2).tolist()} mm")
        _info(f"  Rotation RPY: {np.round(_rot_to_euler_deg(T_he[:3,:3]), 2).tolist()} deg")

        if HAND_EYE_TEST_POINTS_ARM is not None:
            residuals = []
            for p_cam, p_arm_exp in zip(HAND_EYE_TEST_POINTS_CAM,
                                        HAND_EYE_TEST_POINTS_ARM):
                p_pred = (T_he @ np.append(p_cam, 1.0))[:3]
                res_mm = float(np.linalg.norm(p_pred - p_arm_exp)) * 1000.0
                residuals.append(res_mm)
                _info(f"  pred={np.round(p_pred*1000,1).tolist()} mm  "
                      f"expected={np.round(p_arm_exp*1000,1).tolist()} mm  "
                      f"residual={res_mm:.2f} mm")
            max_res = max(residuals)
            results["hand_eye_residual_mm"] = max_res
            (_ok if max_res <= HAND_EYE_WARN_MM else _warn)(
                f"Hand-eye residual: {max_res:.2f} mm")
        else:
            _info("No ground-truth arm points — skipping residual check.")
            _info(" → Set HAND_EYE_TEST_POINTS_ARM in accuracy_diagnostic.py to enable.")
            results["hand_eye_residual_mm"] = None

    # ── 1e. Jacobian condition number check ───────────────────────────────
    print(f"\n{_BOLD}5. Jacobian Condition Number Check{_RESET}")
    all_conds = []
    for i, q in enumerate(KNOWN_CONFIGS):
        J = arm.jacobian(q)
        sv = np.linalg.svd(J, compute_uv=False)
        cond = float(sv[0] / (sv[-1] + 1e-12))
        all_conds.append(cond)
        flag = "  ⚠ NEAR SINGULARITY" if cond > COND_WARN else ""
        _info(f"Config {i+1}: cond={cond:.1f}  min_sv={sv[-1]:.5f}{flag}")

    max_cond = max(all_conds)
    results["max_jacobian_cond"] = max_cond
    (_ok if max_cond <= COND_WARN else _warn)(
        f"Jacobian condition: {max_cond:.1f}")

    # ── Summary ───────────────────────────────────────────────────────────
    _hdr("DIAGNOSTIC SUMMARY")
    _print_summary(results)
    return results


def _print_summary(results):
    rows = [
        ("fk_max_err_mm",        "FK max error",              "mm",  FK_WARN_MM),
        ("ik_max_err_mm",        "IK round-trip error",        "mm",  IK_WARN_MM),
        ("reproj_err_px",        "Camera reprojection error",  "px",  REPROJ_WARN_PX),
        ("hand_eye_residual_mm", "Hand-eye residual",          "mm",  HAND_EYE_WARN_MM),
        ("max_jacobian_cond",    "Jacobian condition number",  "",    COND_WARN),
    ]
    for key, label, unit, thresh in rows:
        v = results.get(key)
        if v is None:
            _info(f"{label}: not measured")
        elif v == float("inf"):
            _warn(f"{label}: not calibrated")
        elif v < 0:
            _warn(f"{label}: field missing from calibration file")
        else:
            suffix = f"{v:.3f} {unit}".strip()
            (_ok if v <= thresh else _warn)(f"{label}: {suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — AUTO FIX
# ═══════════════════════════════════════════════════════════════════════════════

def step2_autofix(arm: FiveDOFArm, results: dict, args) -> list:
    """Apply fixes based on diagnostic results. Returns list of applied fixes."""
    _hdr("STEP 2 — AUTO FIX")
    fixes = []

    # ── Fix A: Jacobian iterative correction if IK residual > threshold ───
    max_ik = results.get("ik_max_err_mm", 0.0)
    if max_ik is not None and max_ik > IK_WARN_MM:
        print(f"\n{_BOLD}Fix A — Jacobian Iterative Correction  "
              f"(IK err={max_ik:.2f}mm > {IK_WARN_MM}mm){_RESET}")
        all_conv = True
        for i, target in enumerate(IK_TEST_TARGETS):
            solved, _, _ = arm.inverse_kinematics(target, max_iters=500, tol=1e-4)
            q0 = arm.joint_angles.copy()
            converged, q_final, err_mm, iters = _jacobian_correction(arm, q0, target)
            if converged:
                _ok(f"Target {i+1}: converged in {iters} iter(s), "
                    f"final err={err_mm:.2f} mm")
            else:
                _warn(f"Target {i+1}: did not converge "
                      f"({JAC_CORR_MAX_ITERS} iters), err={err_mm:.2f} mm")
                all_conv = False
        fixes.append(f"Jacobian correction validated "
                     f"(all converged={all_conv}, thresh={JAC_CORR_THRESH_M*1000:.1f}mm)")
        _info(f"→ The correction loop is already wired in main.py "
              f"(_iterative_jacobian_correction).")
        _info(f"→ Tune via env: JAC_CORR_THRESH_M  JAC_CORR_MAX_ITERS  "
              f"JAC_CORR_GAIN  JAC_CORR_STEP_MS")

    # ── Fix B: increase DLS lambda for near-singular configs ──────────────
    max_cond = results.get("max_jacobian_cond", 0.0)
    if max_cond is not None and max_cond > COND_WARN:
        print(f"\n{_BOLD}Fix B — DLS Lambda Recommendation  "
              f"(cond={max_cond:.1f} > {COND_WARN}){_RESET}")
        rec_lam = float(np.clip(max_cond / 500.0, 0.05, 0.50))
        _warn(f"Jacobian condition {max_cond:.1f} exceeds threshold.")
        _ok(f"Recommended DLS lambda: {rec_lam:.3f}  (current default {JAC_CORR_LAM})")
        _info(f"→ Apply: export IK_LAM={rec_lam:.3f}  before running main.py")
        _info("→ Or update lam= in _ik_refine() call inside arm_controller.py.")
        fixes.append(f"DLS lambda recommendation: {rec_lam:.3f}")

    # ── Fix C: link length update (CLI supplied) ───────────────────────────
    if args.fix_links:
        print(f"\n{_BOLD}Fix C — Link Length Update{_RESET}")
        vals = args.fix_links
        if len(vals) == 4:
            new_links = [DEFAULT_LINKS[0]] + [v / 1000.0 for v in vals]
        elif len(vals) == 5:
            new_links = [v / 1000.0 for v in vals]
        else:
            _warn(f"Expected 4 or 5 values, got {len(vals)}. Skipping.")
            new_links = None

        if new_links:
            arm.link_lengths = np.array(new_links, dtype=float)
            _ok(f"Link lengths updated: {[round(v*1000, 1) for v in new_links]} mm")
            _info("→ Make permanent by updating DEFAULT_LINKS in arm_controller.py:")
            _info(f"  DEFAULT_LINKS = {new_links}")
            fixes.append(f"Link lengths updated: {new_links}")

    # ── Fix D: hand-eye calibration ───────────────────────────────────────
    he_res = results.get("hand_eye_residual_mm")
    needs_he = (he_res is None or
                (isinstance(he_res, float) and
                 (he_res == float("inf") or he_res > HAND_EYE_WARN_MM)))
    if args.hand_eye or needs_he:
        print(f"\n{_BOLD}Fix D — Hand-Eye Calibration{_RESET}")
        _run_hand_eye_calibration()
        fixes.append("Hand-eye calibration routine run")

    # ── Fix E: camera calibration instructions ────────────────────────────
    rp = results.get("reproj_err_px")
    if rp is None or (isinstance(rp, float) and rp > REPROJ_WARN_PX):
        print(f"\n{_BOLD}Fix E — Camera Calibration{_RESET}")
        _print_camera_calibration_instructions()

    if not fixes:
        _ok("All checks within threshold — no automatic fixes required.")
    else:
        print(f"\n  Applied {len(fixes)} fix(es):")
        for f in fixes:
            _info(f"• {f}")
    return fixes


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════

def step3_validate(arm: FiveDOFArm, before: dict):
    """Re-run Step 1 and compare to before-results."""
    _hdr("STEP 3 — VALIDATION  (before → after)")
    after = step1_diagnose(arm)

    print(f"\n{_BOLD}Improvement Report{_RESET}")
    fields = [
        ("fk_max_err_mm",        "FK max error",              "mm"),
        ("ik_max_err_mm",        "IK round-trip error",        "mm"),
        ("reproj_err_px",        "Camera reproj error",        "px"),
        ("hand_eye_residual_mm", "Hand-eye residual",          "mm"),
        ("max_jacobian_cond",    "Jacobian condition",         ""),
    ]
    for key, label, unit in fields:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            continue
        if not (isinstance(b, float) and isinstance(a, float)):
            continue
        if b == float("inf") or a == float("inf"):
            _info(f"{label}: not measurable before/after")
            continue
        delta = a - b
        symbol = "▼" if delta < 0 else ("▲" if delta > 0 else "═")
        suffix = unit.strip()
        (_ok if delta <= 0 else _warn)(
            f"{label:30s}  {b:.3f} → {a:.3f} {suffix}  "
            f"{symbol} {abs(delta):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Hand-eye calibration routine
# ═══════════════════════════════════════════════════════════════════════════════

def _run_hand_eye_calibration():
    if not os.path.isfile(HAND_EYE_POSES):
        _warn(f"No pose file found at {HAND_EYE_POSES}")
        _info("")
        _info("To collect hand-eye pose pairs:")
        _info("  1. Print a 9×6 chessboard (25 mm squares).")
        _info("  2. Mount it stably in front of the arm.")
        _info("  3. Move the arm to ≥15 distinct poses where the board is visible.")
        _info("  4. For each pose record:")
        _info("       • gripper-to-base 4×4 transform  (computed from FK)")
        _info("       • board-to-camera 4×4 transform  (from cv2.solvePnP)")
        _info("  5. Save as hand_eye_poses.json:")
        _info('       {"gripper2base": [[[...4×4...]], ...],')
        _info('        "target2cam":   [[[...4×4...]], ...]}')
        _info("  6. Re-run: python accuracy_diagnostic.py --hand-eye")
        return

    try:
        import cv2
    except ImportError:
        _warn("cv2 not available — install opencv-contrib-python for calibrateHandEye.")
        return

    try:
        with open(HAND_EYE_POSES) as f:
            data = json.load(f)

        R_g2b = [np.array(T, dtype=float)[:3, :3] for T in data["gripper2base"]]
        t_g2b = [np.array(T, dtype=float)[:3,  3].reshape(3, 1)
                 for T in data["gripper2base"]]
        R_t2c = [np.array(T, dtype=float)[:3, :3] for T in data["target2cam"]]
        t_t2c = [np.array(T, dtype=float)[:3,  3].reshape(3, 1)
                 for T in data["target2cam"]]

        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        T_he = np.eye(4, dtype=float)
        T_he[:3, :3] = R_c2g
        T_he[:3,  3] = t_c2g.flatten()

        with open(HAND_EYE_FILE, "w") as f:
            json.dump({"T": T_he.tolist(),
                       "method": "TSAI",
                       "n_poses": len(R_g2b)}, f, indent=2)

        _ok(f"Hand-eye transform computed from {len(R_g2b)} pose pairs.")
        _ok(f"Saved → {HAND_EYE_FILE}")
        _info(f"  Translation: {np.round(t_c2g.flatten()*1000, 2).tolist()} mm")
        _info(f"  Rotation RPY: {np.round(_rot_to_euler_deg(R_c2g), 2).tolist()} deg")

    except Exception as e:
        _fail(f"Hand-eye calibration failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Camera calibration instructions
# ═══════════════════════════════════════════════════════════════════════════════

def _print_camera_calibration_instructions():
    _info("")
    _info("Camera Calibration Instructions")
    _info("─" * 54)
    _info("1. Print a 9×6 chessboard at exactly 25 mm square size.")
    _info("2. Collect ≥20 images of the board at varied angles/distances.")
    _info("   Place images in:  calib_images/*.jpg")
    _info("3. Run the snippet below (or adapt it):")
    _info("")
    _info("   import cv2, glob, numpy as np, json")
    _info("   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)")
    _info("   objp = np.zeros((6*9, 3), np.float32)")
    _info("   objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025")
    _info("   objpoints, imgpoints = [], []")
    _info("   for fname in glob.glob('calib_images/*.jpg'):")
    _info("       img  = cv2.imread(fname)")
    _info("       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
    _info("       ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)")
    _info("       if ret:")
    _info("           objpoints.append(objp)")
    _info("           corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)")
    _info("           imgpoints.append(corners2)")
    _info("   rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(")
    _info("       objpoints, imgpoints, gray.shape[::-1], None, None)")
    _info("   json.dump({'K': K.tolist(), 'dist': dist.tolist(),")
    _info("             'reprojection_error': rms},")
    _info("            open('camera_calibration.json', 'w'))")
    _info("   print(f'RMS = {rms:.4f} px')")
    _info("")
    _info("4. Re-run: python accuracy_diagnostic.py")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Accuracy diagnostic & auto-fix pipeline for 5-DOF tomato arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", type=int, choices=[1, 2, 3],
                        help="Run only step 1 (diagnose), 2 (fix), or 3 (validate). "
                             "Default: all three.")
    parser.add_argument("--fix-links", type=float, nargs="+", metavar="MM",
                        help="Measured link lengths in mm. Provide 4 values "
                             "(shoulder elbow wrist tool) or 5 (base shoulder elbow wrist tool).")
    parser.add_argument("--hand-eye", action="store_true",
                        help="Force hand-eye calibration routine.")
    parser.add_argument("--calibrate-camera", action="store_true",
                        help="Print camera calibration instructions and exit.")
    parser.add_argument("--fit-bias", action="store_true",
                        help="Interactively estimate a constant arm-frame XYZ bias and save it.")
    args = parser.parse_args()

    if args.calibrate_camera:
        _print_camera_calibration_instructions()
        return
    if args.fit_bias:
        _run_bias_calibration()
        return

    arm = FiveDOFArm()
    arm.set_joint_angles(search_home_angles())

    if args.step == 1:
        step1_diagnose(arm)
    elif args.step == 2:
        results = step1_diagnose(arm)
        step2_autofix(arm, results, args)
    elif args.step == 3:
        before = step1_diagnose(arm)
        step2_autofix(arm, before, args)
        step3_validate(arm, before)
    else:
        before = step1_diagnose(arm)
        step2_autofix(arm, before, args)
        step3_validate(arm, before)


if __name__ == "__main__":
    main()
