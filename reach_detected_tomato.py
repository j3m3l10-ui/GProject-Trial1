#!/usr/bin/env python3
"""Image-Based Visual Servoing (IBVS) reach: detect tomato → coarse approach → pixel-error servo → cut.

Pipeline
--------
1. Wait for N stable detections at home (initial lock).
2. Coarse approach: move arm forward by ~(depth * DEPTH_FRAC) using IK.
3. IBVS loop (up to VS_MAX_ITERS):
   a. Re-detect tomato in current frame.
   b. Compute pixel error:  e_x = center_px - frame_cx,  e_y = center_px - frame_cy
   c. If |e_x| < PIX_THRESH and |e_y| < PIX_THRESH → converged, proceed to cut.
   d. Otherwise apply proportional joint adjustments:
        base yaw  ← -Kp * e_x   (pan left/right)
        shoulder  ← -Kp * e_y   (tilt up/down)
4. Stem cut: move +STEM_OFFSET_M on arm-Z then close gripper.
5. Open gripper, return home.

Why pixel error (IBVS) instead of 3D world error?
  Depth from a monocular camera is noisy. Pixel position of the tomato
  centre is stable and directly maps to servo axes: X-pixels → yaw,
  Y-pixels → pitch/shoulder. No camera-to-arm calibration needed for the
  correction loop.

Usage
-----
  /usr/bin/python reach_detected_tomato.py
  /usr/bin/python reach_detected_tomato.py --dry-run
  /usr/bin/python reach_detected_tomato.py --no-cut      # approach & servo, skip cut
  /usr/bin/python reach_detected_tomato.py --show        # live preview window
"""

import argparse
import os
import time

import cv2
import numpy as np

from vision import TomatoDetector
from arm_controller import FiveDOFArm, PULSE_PER_RAD
from arm_home_position import get_home_angles, get_home_pulses
from servo_driver import ServoDriver
from main import camera_to_arm_frame, _solve_ik_with_fallback, execute_trajectory, _project_to_reachable

# ── Tunable constants (override via env) ─────────────────────────────────────
DEPTH_FRAC      = float(os.getenv("VS_DEPTH_FRAC",    "0.60"))   # fraction of initial depth for coarse pan/tilt standoff
VS_MAX_ITERS    = int(os.getenv("VS_MAX_ITERS",        "6"))      # IBVS correction iterations
PIX_THRESH      = float(os.getenv("VS_PIX_THRESH",    "20.0"))   # pixel convergence radius
KP_BASE         = float(os.getenv("VS_KP_BASE",       "0.0008")) # rad per pixel for base yaw
KP_SHOULDER     = float(os.getenv("VS_KP_SHOULDER",   "0.0008")) # rad per pixel for shoulder/elbow
MAX_STEP_RAD    = float(os.getenv("VS_MAX_STEP_RAD",  "0.10"))   # max joint delta per iteration (rad)
STEM_OFFSET_M   = float(os.getenv("VS_STEM_OFFSET_M", "0.01"))   # +Z to reach stem before cut
SETTLE_S        = float(os.getenv("VS_SETTLE_S",      "1.2"))    # pause after each move (s)


# ── helpers ──────────────────────────────────────────────────────────────────

def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def move_home(arm, driver, dry_run=False):
    arm.set_joint_angles(get_home_angles())
    if dry_run:
        print("[DRY RUN] move_home")
        return
    driver.move_servos(get_home_pulses(), duration_ms=6000)
    time.sleep(3.0)


def grab_stable_detection(cap, detector, min_hits, show=False, window="VS"):
    """Block until *min_hits* consecutive stable detections; return nearest det."""
    stable_hits = 0
    last_center = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        dets = detector.detect(frame, use_tracker=False)
        nearest = dets[0] if dets else None
        if nearest is None:
            stable_hits = 0
            last_center = None
        else:
            c = nearest.get("center_px", [0, 0])
            cxy = np.array([float(c[0]), float(c[1])], dtype=float)
            if last_center is None or np.linalg.norm(cxy - last_center) < 8.0:
                stable_hits += 1
            else:
                stable_hits = 1
            last_center = cxy
            if stable_hits >= min_hits:
                return nearest
        if show:
            view = detector.annotate(frame, dets)
            pct = min(100, int(stable_hits / min_hits * 100))
            cv2.putText(view, f"Locking {pct}%  ({stable_hits}/{min_hits})", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt


def detect_from_frame(cap, detector, n_good=4, timeout_s=5.0, show=False, window="VS"):
    """Read frames for up to *timeout_s*, return mean detection or None."""
    cxs, cys, dets_list = [], [], []
    deadline = time.time() + timeout_s
    while len(cxs) < n_good and time.time() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        dets = detector.detect(frame, use_tracker=False)
        if dets:
            c = dets[0].get("center_px", [0, 0])
            cxs.append(float(c[0]))
            cys.append(float(c[1]))
            dets_list.append(dets[0])
        if show:
            view = detector.annotate(frame, dets)
            cv2.putText(view, f"IBVS detect {len(cxs)}/{n_good}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.imshow(window, view)
            cv2.waitKey(1)
    if not cxs:
        return None, None, None
    return float(np.mean(cxs)), float(np.mean(cys)), dets_list[-1]


def apply_joint_step(arm, driver, dq, dry_run=False):
    """Apply a small joint delta; clamp to limits; execute with short trajectory."""
    q = np.array(arm.joint_angles, dtype=float)
    q_new = q + np.array(dq, dtype=float)
    # Clamp to joint limits
    for i, (lo, hi) in enumerate(arm.joint_limits):
        q_new[i] = float(np.clip(q_new[i], lo, hi))
    if dry_run:
        print(f"  [DRY RUN] joint step dq={np.round(dq,4)}  q_new={np.round(q_new,4)}")
        arm.set_joint_angles(q_new)
        return
    execute_trajectory(arm, driver, q, q_new, steps=8, duration_ms=1000)
    time.sleep(SETTLE_S)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="IBVS visual-servoing tomato reach + cut")
    ap.add_argument("--camera",   type=int,   default=0)
    ap.add_argument("--imgsz",    type=int,   default=416)
    ap.add_argument("--min-hits", type=int,   default=8)
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--no-cut",   action="store_true",
                    help="Approach and servo but skip gripper close")
    ap.add_argument("--show",     action="store_true")
    args = ap.parse_args()

    focal_length_px = float(os.getenv("FOCAL_LENGTH_PX", "700"))
    detector = TomatoDetector(imgsz=args.imgsz, focal_length_px=focal_length_px)
    arm      = FiveDOFArm()
    driver   = ServoDriver(mode="real")
    cap      = open_camera(args.camera)

    frame_cx, frame_cy = 320.0, 240.0   # image centre (640×480)

    try:
        # ── 0. Home ────────────────────────────────────────────────────────
        move_home(arm, driver, dry_run=args.dry_run)

        # ── 1. Initial stable detection ────────────────────────────────────
        print(f"[VS] Waiting for {args.min_hits} stable detections...")
        init_det = grab_stable_detection(cap, detector, args.min_hits,
                                         show=args.show, window="Visual Servoing")
        init_xyz   = init_det["xyz_cm"]
        init_depth = float(init_xyz["z"]) / 100.0          # metres
        init_cx    = float(init_det.get("center_px", [frame_cx, frame_cy])[0])
        init_cy    = float(init_det.get("center_px", [frame_cx, frame_cy])[1])
        print(f"[VS] Locked: depth={init_depth*100:.1f}cm  px=({init_cx:.0f},{init_cy:.0f})"
              f"  conf={init_det.get('confidence',0):.2f}")

        # ── 2. Coarse approach via IK on 3D target ─────────────────────────
        coarse_target_m = camera_to_arm_frame(init_xyz)
        # Scale toward arm base by (1 - DEPTH_FRAC) to stop short
        ee_home = np.array(arm.end_effector_pos(), dtype=float)
        coarse_target_m = ee_home + (coarse_target_m - ee_home) * DEPTH_FRAC
        # Clamp to reachable workspace
        coarse_target_m, was_projected = _project_to_reachable(coarse_target_m, arm, safety_margin_m=0.04)
        if was_projected:
            print(f"[VS] Coarse target projected to reachable boundary.")

        print(f"[VS] Coarse approach → {np.round(coarse_target_m,4)} m  (depth_frac={DEPTH_FRAC})")
        q_home = np.array(arm.joint_angles, dtype=float)
        accepted, _, err, _, q_coarse, _ = _solve_ik_with_fallback(arm, q_home, coarse_target_m)
        if not accepted:
            print("[VS] Coarse IK failed – aborting.")
            return 1
        if args.dry_run:
            print(f"[VS][DRY RUN] Coarse move  err={err*1000:.1f}mm")
            arm.set_joint_angles(q_coarse)
        else:
            execute_trajectory(arm, driver, q_home, q_coarse, steps=50, duration_ms=8000)
            time.sleep(SETTLE_S)

        # ── 3. Phase 1: Pan/tilt IBVS to centre tomato in frame ───────────
        # Run at current standoff distance - no forward movement yet.
        # Once the tomato is centred we know the arm is aligned with it.
        print("\n[VS] Phase 1: Aligning camera to tomato (pan/tilt servo)...")
        converged = False
        last_pix_err = float("inf")

        for iteration in range(1, VS_MAX_ITERS + 1):
            print(f"\n[VS] ── IBVS Align Iteration {iteration}/{VS_MAX_ITERS} ──────────")

            px, py, det = detect_from_frame(cap, detector, n_good=4,
                                            timeout_s=5.0, show=args.show,
                                            window="Visual Servoing")
            if det is None:
                print("  [VS] No detection – adjusting pan/tilt may have moved tomato out of frame.")
                break

            ex = px - frame_cx
            ey = py - frame_cy
            pix_err = float(np.linalg.norm([ex, ey]))
            last_pix_err = pix_err
            print(f"  Tomato px=({px:.0f},{py:.0f})  error=({ex:.1f},{ey:.1f})px  |e|={pix_err:.1f}px")

            if pix_err < PIX_THRESH:
                print(f"  [VS] Aligned ✓  |e|={pix_err:.1f} < {PIX_THRESH:.0f}px")
                converged = True
                break

            # Base yaw corrects X pixel error; shoulder/elbow corrects Y pixel error
            d_base     = float(np.clip(-KP_BASE     * ex, -MAX_STEP_RAD, MAX_STEP_RAD))
            d_shoulder = float(np.clip(-KP_SHOULDER * ey, -MAX_STEP_RAD, MAX_STEP_RAD))
            d_elbow    = float(np.clip(-KP_SHOULDER * ey * 0.4, -MAX_STEP_RAD * 0.4, MAX_STEP_RAD * 0.4))

            dq = np.zeros(5, dtype=float)
            dq[0] = d_base
            dq[1] = d_shoulder
            dq[2] = d_elbow
            print(f"  Step: base={np.degrees(d_base):.1f}°  "
                  f"shoulder={np.degrees(d_shoulder):.1f}°  "
                  f"elbow={np.degrees(d_elbow):.1f}°")
            apply_joint_step(arm, driver, dq, dry_run=args.dry_run)

        if not converged:
            print(f"\n[VS] Alignment did not converge (|e|={last_pix_err:.1f}px).")
            if last_pix_err > PIX_THRESH * 5:
                print("[VS] Could not align to tomato – aborting.")
                return 1
            print("[VS] Partial alignment – proceeding.")

        # ── 4: Phase 2: Forward approach now that camera is aligned ────────
        print("\n[VS] Phase 2: Forward approach to target depth...")
        px2, py2, det2 = detect_from_frame(cap, detector, n_good=4,
                                           timeout_s=5.0, show=args.show,
                                           window="Visual Servoing")
        if det2 is not None:
            new_xyz = det2["xyz_cm"]
            approach_target = camera_to_arm_frame(new_xyz)
            approach_target, _ = _project_to_reachable(approach_target, arm, safety_margin_m=0.06)
            q_now = np.array(arm.joint_angles, dtype=float)
            acc, _, err2, _, q_approach, _ = _solve_ik_with_fallback(arm, q_now, approach_target)
            if acc:
                print(f"  Depth approach → {np.round(approach_target,4)} m")
                if args.dry_run:
                    print(f"  [DRY RUN] Approach move err={err2*1000:.1f}mm")
                    arm.set_joint_angles(q_approach)
                else:
                    execute_trajectory(arm, driver, q_now, q_approach, steps=20, duration_ms=3000)
                    time.sleep(SETTLE_S)
            else:
                print("  [VS] Approach IK failed – staying at aligned pose.")
        else:
            print("  [VS] No detection after alignment – staying at current pose.")

        # ── 5: Phase 3: Fine IBVS after approach ───────────────────────────
        print("\n[VS] Phase 3: Fine pixel alignment after approach...")
        for iteration in range(1, 3):
            px, py, det = detect_from_frame(cap, detector, n_good=4,
                                            timeout_s=4.0, show=args.show,
                                            window="Visual Servoing")
            if det is None:
                print("  [VS] No detection for fine tune – skipping.")
                break
            ex = px - frame_cx
            ey = py - frame_cy
            pix_err = float(np.linalg.norm([ex, ey]))
            print(f"  Fine iter {iteration}: px=({px:.0f},{py:.0f}) error=({ex:.1f},{ey:.1f})px |e|={pix_err:.1f}")
            if pix_err < PIX_THRESH:
                print(f"  [VS] Fine aligned ✓")
                break
            d_base     = float(np.clip(-KP_BASE * 0.5     * ex, -MAX_STEP_RAD * 0.5, MAX_STEP_RAD * 0.5))
            d_shoulder = float(np.clip(-KP_SHOULDER * 0.5 * ey, -MAX_STEP_RAD * 0.5, MAX_STEP_RAD * 0.5))
            dq = np.zeros(5, dtype=float)
            dq[0] = d_base
            dq[1] = d_shoulder
            apply_joint_step(arm, driver, dq, dry_run=args.dry_run)

        # ── 4. Stem cut: +STEM_OFFSET_M on Z then close gripper ───────────
        print(f"\n[VS] Approaching stem (+{STEM_OFFSET_M*100:.1f}cm on Z)...")
        q_now = np.array(arm.joint_angles, dtype=float)
        stem_target = np.array(arm.end_effector_pos(), dtype=float)
        stem_target[2] += STEM_OFFSET_M
        acc, _, _, _, q_stem, _ = _solve_ik_with_fallback(arm, q_now, stem_target)
        if acc:
            if args.dry_run:
                print(f"[VS][DRY RUN] Stem move to {np.round(stem_target,4)}")
                arm.set_joint_angles(q_stem)
            else:
                execute_trajectory(arm, driver, q_now, q_stem, steps=8, duration_ms=1200)
                time.sleep(0.3)
        else:
            print("[VS] Stem IK rejected – cutting at current position.")

        if args.no_cut or args.dry_run:
            tag = "[DRY RUN] " if args.dry_run else ""
            print(f"[VS] {tag}Cut skipped.")
        else:
            print("[VS] Cutting: closing gripper...")
            driver.move_servo(1, 700, duration_ms=200)
            time.sleep(0.3)
            driver.move_servo(1, 350, duration_ms=400)
            time.sleep(0.4)
            print("[VS] Cut done. Gripper open.")

        # ── 5. Return home ─────────────────────────────────────────────────
        print("[VS] Returning to home...")
        move_home(arm, driver, dry_run=args.dry_run)
        print("[VS] Done.")

    except KeyboardInterrupt:
        print("\n[VS] Stopped by user.")
    finally:
        try:
            driver.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


