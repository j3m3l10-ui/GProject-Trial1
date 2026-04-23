#!/usr/bin/env python3
"""
Simulation GUI — 3-Tomato Batch Harvest with 3D Robotic Arm
=============================================================
Interactive 3D simulation for the full detection → IK → stem-cut → collect
pipeline.  Supports up to 3 tomatoes per harvest cycle, visited nearest-first.

Features:
  - 3D animated 5-DOF arm with scissor end-effector
  - 3 tomatoes with stems, sorted and colored by distance
  - Sequential harvest: nearest → 2nd → 3rd → return home
  - Net/basket for catching cut tomatoes
  - Live camera feed panel (optional) with real YOLO detection
  - Works with both Simulation and Hardware modes

Launch:
  python main.py --gui
  or directly: python simulation_gui.py
"""

import math
import sys
import time
import threading
import numpy as np
import cv2
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import random

# Enable vision filter logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from arm_controller import (
    FiveDOFArm, angles_to_pulses, search_home_angles,
    compute_stem_cut_point, SERVO_IDS, SEARCH_HOME_PULSES,
)
from servo_driver import ServoDriver

# Try to import vision (may fail if no model weights yet)
_VISION_AVAILABLE = False
try:
    from vision import TomatoDetector, ThreadedCamera
    _VISION_AVAILABLE = True
except Exception:
    pass


def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


# ── Default tomato positions (within arm reach) ───────────────────────────────
DEFAULT_TOMATOES = [
    np.array([0.20, -0.05, 0.15], dtype=float),
    np.array([0.18,  0.06, 0.12], dtype=float),
    np.array([0.22,  0.00, 0.19], dtype=float),
]


class SimulationGUI:
    """Full 3D simulation GUI for the 3-tomato batch harvesting robot."""

    MAX_TOMATOES = 3

    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Harvester — 3-Tomato Batch Simulation")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Subsystems
        self.arm = FiveDOFArm()
        self.driver = ServoDriver(mode="real")
        self.detector = TomatoDetector() if _VISION_AVAILABLE else None

        # Home angles
        self.home_angles = search_home_angles()
        self.arm.set_joint_angles(self.home_angles)

        # Tomato state: list of 3 tomatoes
        self.tomato_radius = 0.035
        self.tomato_states = []
        for pos in DEFAULT_TOMATOES:
            self.tomato_states.append({
                "center": pos.copy(),
                "pos": pos.copy(),
                "detached": False,
            })

        # Net
        self.net_center = np.array([0.20, 0.0, 0.0], dtype=float)
        self.net_radius = 0.10
        self.net_rim_z = 0.04
        self.net_depth = 0.05

        # Scissor state
        self.scissor_open_angle = math.radians(25)
        self.scissor_blade_len = 0.04
        self.scissor_pivot_back = 0.01

        # Animation / state machine
        self.animation_running = False
        self.harvest_queue = []
        self.harvest_index = 0
        self.camera_active = False
        self.cap = None
        self._closing = False

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_ui()
        self.draw_arm()

    # ══════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── 3D Plot (left) ────────────────────────────────────────────────
        plot_frame = ttk.LabelFrame(main_frame, text="3D Arm Simulation")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Right panel ───────────────────────────────────────────────────
        right = ttk.Frame(main_frame, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # ── Tomato positions (3 rows) ─────────────────────────────────────
        pos_frame = ttk.LabelFrame(right, text="Tomato Positions (metres)")
        pos_frame.pack(fill=tk.X, pady=5)

        self.tomato_vars = []  # list of (x_var, y_var, z_var)
        colors_label = ["#1 (nearest)", "#2", "#3"]
        for idx in range(self.MAX_TOMATOES):
            row_frame = ttk.Frame(pos_frame)
            row_frame.pack(fill=tk.X, padx=3, pady=1)
            ttk.Label(row_frame, text=colors_label[idx], width=10).pack(
                side=tk.LEFT)
            x_var = tk.StringVar(
                value=f"{self.tomato_states[idx]['center'][0]:.3f}")
            y_var = tk.StringVar(
                value=f"{self.tomato_states[idx]['center'][1]:.3f}")
            z_var = tk.StringVar(
                value=f"{self.tomato_states[idx]['center'][2]:.3f}")
            for label, var in [("X", x_var), ("Y", y_var), ("Z", z_var)]:
                ttk.Label(row_frame, text=label).pack(side=tk.LEFT, padx=1)
                ttk.Entry(row_frame, textvariable=var, width=6).pack(
                    side=tk.LEFT, padx=1)
            self.tomato_vars.append((x_var, y_var, z_var))

        ttk.Button(pos_frame, text="Randomise Positions",
                   command=self._randomise_tomatoes).pack(
            fill=tk.X, pady=2, padx=3)

        # ── Action buttons ────────────────────────────────────────────────
        btn_frame = ttk.LabelFrame(right, text="Actions")
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Harvest 3 Tomatoes (Nearest-First)",
                   command=self._start_multi_harvest).pack(
            fill=tk.X, pady=2, padx=3)
        ttk.Button(btn_frame, text="Harvest Single Tomato (#1)",
                   command=self._start_single_harvest).pack(
            fill=tk.X, pady=2, padx=3)
        ttk.Button(btn_frame, text="Move to Search Home",
                   command=self._go_home).pack(fill=tk.X, pady=2, padx=3)
        ttk.Button(btn_frame, text="Reset Arm & Tomatoes",
                   command=self._reset).pack(fill=tk.X, pady=2, padx=3)

        # ── Camera integration ────────────────────────────────────────────
        cam_frame = ttk.LabelFrame(right, text="Live Camera")
        cam_frame.pack(fill=tk.X, pady=5)

        self.cam_btn_text = tk.StringVar(value="Start Camera Detection")
        ttk.Button(cam_frame, textvariable=self.cam_btn_text,
                   command=self._toggle_camera).pack(fill=tk.X, pady=2, padx=3)

        ttk.Label(cam_frame, text="60 FPS • Detects up to 3 tomatoes → fills positions",
                  font=("", 8)).pack(padx=3)

        # ── Status ────────────────────────────────────────────────────────
        status_frame = ttk.LabelFrame(right, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        self.status_var = tk.StringVar(
            value="Ready. Set tomato positions or start camera.")
        ttk.Label(status_frame, textvariable=self.status_var,
                  wraplength=270).pack(padx=5, pady=5)

        # ── Joint angles display ──────────────────────────────────────────
        joint_frame = ttk.LabelFrame(right, text="Joint Angles (deg) / Pulses")
        joint_frame.pack(fill=tk.X, pady=5)
        self.joint_labels = []
        names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]
        for name in names:
            lbl = ttk.Label(joint_frame, text=f"{name}: 0.0° / 500")
            lbl.pack(anchor=tk.W, padx=5)
            self.joint_labels.append(lbl)

    # ══════════════════════════════════════════════════════════════════════
    #  3D Drawing
    # ══════════════════════════════════════════════════════════════════════

    def draw_arm(self, highlight_targets=None):
        self.ax.clear()

        positions, rotations = self.arm.forward_kinematics()
        positions = np.array(positions, dtype=float)

        # Arm links
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                     '-o', color='#2196F3', linewidth=3, markersize=6)

        # Scissor end-effector
        tip = positions[-1]
        R_tip = rotations[-1]
        self._draw_scissors(tip, R_tip)

        # Net basket
        self._draw_net()

        # Draw all tomatoes with stems
        tomato_colors = ['#FF0000', '#FF6600', '#FF9900']  # red, orange, gold
        stem_color = '#228B22'

        # Sort tomatoes by distance for ranking display
        ranked = self._get_ranked_tomatoes()

        for rank, (dist, orig_idx) in enumerate(ranked):
            ts = self.tomato_states[orig_idx]
            tp = ts["pos"]
            tc = ts["center"]
            color = tomato_colors[min(rank, len(tomato_colors) - 1)]

            # Tomato sphere
            u, v = np.mgrid[0:2*np.pi:16j, 0:np.pi:9j]
            r = self.tomato_radius
            x = tp[0] + r * np.cos(u) * np.sin(v)
            y = tp[1] + r * np.sin(u) * np.sin(v)
            z = tp[2] + r * np.cos(v)
            alpha = 0.3 if ts["detached"] else 0.7
            self.ax.plot_surface(x, y, z, color=color, alpha=alpha,
                                 linewidth=0)

            # Stem (small green line at top of tomato — only if not detached)
            if not ts["detached"]:
                stem_base = tc.copy()
                stem_base[2] += r
                stem_top = stem_base.copy()
                stem_top[2] += 0.02  # 2cm stem
                self.ax.plot([stem_base[0], stem_top[0]],
                             [stem_base[1], stem_top[1]],
                             [stem_base[2], stem_top[2]],
                             color=stem_color, linewidth=2.5)
                # Small leaf at top of stem
                self.ax.scatter(*stem_top, color=stem_color, s=20, marker='^')

            # Distance label
            label = f"#{rank+1} D:{dist:.0f}cm"
            self.ax.text(tc[0], tc[1], tc[2] + r + 0.03, label,
                         fontsize=7, ha='center',
                         color=color if not ts["detached"] else 'gray')

        # Highlight targets (cut points)
        if highlight_targets:
            for ht in highlight_targets:
                self.ax.scatter(*ht, color='yellow', s=80, marker='*',
                                zorder=5)

        # Base platform
        self.ax.scatter(0, 0, 0, color='gray', s=100, marker='s')

        # Axis formatting
        reach = self.arm.max_reach()
        lim = reach * 0.8
        self.ax.set_xlim(-lim * 0.3, lim)
        self.ax.set_ylim(-lim * 0.6, lim * 0.6)
        self.ax.set_zlim(0, lim)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("5-DOF Arm — 3-Tomato Batch Harvest")

        try:
            self.ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        self.canvas.draw()
        self._update_joint_display()

    def _draw_scissors(self, tip, R_tip):
        pivot = tip + R_tip.dot([-self.scissor_pivot_back, 0, 0])
        half = 0.5 * self.scissor_open_angle
        v1 = rot_x(+half).dot([0, self.scissor_blade_len, 0])
        v2 = rot_x(-half).dot([0, self.scissor_blade_len, 0])
        p1 = pivot + R_tip.dot(v1)
        p2 = pivot + R_tip.dot(v2)
        self.ax.plot(*zip(pivot, p1), color='#FF5722', linewidth=2.5)
        self.ax.plot(*zip(pivot, p2), color='#FF5722', linewidth=2.5)
        self.ax.scatter(*pivot, color='#FF5722', s=25)

    def _draw_net(self):
        cx, cy = self.net_center[0], self.net_center[1]
        z0 = self.net_rim_z
        t = np.linspace(0, 2*np.pi, 50)
        xr = cx + self.net_radius * np.cos(t)
        yr = cy + self.net_radius * np.sin(t)
        self.ax.plot(xr, yr, np.full_like(t, z0), color='#8D6E63',
                     linewidth=1.5)
        zb = max(0, z0 - self.net_depth)
        for k in range(0, len(t), 5):
            self.ax.plot([xr[k], xr[k]], [yr[k], yr[k]], [z0, zb],
                         color='#8D6E63', linewidth=0.8, alpha=0.6)

    def _update_joint_display(self):
        names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]
        pulses = angles_to_pulses(self.arm.joint_angles)
        for i, name in enumerate(names):
            deg = math.degrees(self.arm.joint_angles[i])
            sid = SERVO_IDS[i]
            p = pulses[sid]
            self.joint_labels[i].config(text=f"{name} (ID{sid}): {deg:+.1f}° / {p}")

    # ══════════════════════════════════════════════════════════════════════
    #  Tomato Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _get_ranked_tomatoes(self):
        """Return list of (distance_cm, original_index) sorted nearest-first."""
        ranked = []
        for i, ts in enumerate(self.tomato_states):
            dist_m = np.linalg.norm(ts["center"] - self.arm.base_pos)
            dist_cm = dist_m * 100.0
            ranked.append((dist_cm, i))
        ranked.sort(key=lambda x: x[0])
        return ranked

    def _read_all_tomato_positions(self):
        """Read tomato positions from GUI entries. Returns list of arrays."""
        positions = []
        for idx, (x_var, y_var, z_var) in enumerate(self.tomato_vars):
            try:
                x = float(x_var.get())
                y = float(y_var.get())
                z = float(z_var.get())
                positions.append(np.array([x, y, z], dtype=float))
            except ValueError:
                messagebox.showerror("Input Error",
                                     f"Tomato #{idx+1}: invalid X/Y/Z.")
                return None
        return positions

    def _update_tomato_states_from_gui(self):
        """Sync tomato_states from the GUI text fields."""
        positions = self._read_all_tomato_positions()
        if positions is None:
            return False
        for i, pos in enumerate(positions):
            self.tomato_states[i]["center"] = pos.copy()
            self.tomato_states[i]["pos"] = pos.copy()
            self.tomato_states[i]["detached"] = False
        return True

    def _randomise_tomatoes(self):
        """Generate 3 random tomato positions within arm reach."""
        if self.animation_running:
            return
        reach = self.arm.max_reach() * 0.7
        for idx in range(self.MAX_TOMATOES):
            x = random.uniform(0.12, reach)
            y = random.uniform(-reach * 0.4, reach * 0.4)
            z = random.uniform(0.06, reach * 0.6)
            self.tomato_vars[idx][0].set(f"{x:.3f}")
            self.tomato_vars[idx][1].set(f"{y:.3f}")
            self.tomato_vars[idx][2].set(f"{z:.3f}")
        self._update_tomato_states_from_gui()
        self.draw_arm()
        self.status_var.set("Random tomato positions generated.")

    # ══════════════════════════════════════════════════════════════════════
    #  Multi-Tomato Harvest Animation
    # ══════════════════════════════════════════════════════════════════════

    def _start_multi_harvest(self):
        """Start the 3-tomato batch harvest simulation (nearest-first)."""
        if self.animation_running:
            return
        if not self._update_tomato_states_from_gui():
            return

        # Sort by distance (nearest first)
        ranked = self._get_ranked_tomatoes()

        # Build harvest queue: list of original indices, nearest first
        self.harvest_queue = [orig_idx for _, orig_idx in ranked]
        self.harvest_index = 0
        self.harvested_count = 0

        # Start from search home
        self.arm.set_joint_angles(self.home_angles)
        self.driver.go_search_home(duration_ms=1200)
        self.scissor_open_angle = math.radians(25)
        self.animation_running = True

        distances = [f"#{r+1}={d:.0f}cm" for r, (d, _) in enumerate(ranked)]
        self.status_var.set(f"Harvest order (nearest-first): {', '.join(distances)}")
        self.draw_arm()

        self.root.after(500, self._harvest_next_tomato)

    def _harvest_next_tomato(self):
        """Process the next tomato in the harvest queue."""
        if self._closing:
            return
        if self.harvest_index >= len(self.harvest_queue):
            # All done — retract to home
            self._animate_retract_home()
            return

        tidx = self.harvest_queue[self.harvest_index]
        pos = self.tomato_states[tidx]["center"]
        rank = self.harvest_index + 1

        # Reachability check
        if not self.arm.is_reachable(pos):
            self.status_var.set(
                f"Tomato #{rank} (idx {tidx}) out of reach — skipping")
            self.harvest_index += 1
            self.root.after(500, self._harvest_next_tomato)
            return

        # Compute stem cut point
        stem_pt, cut_pt = compute_stem_cut_point(
            pos, self.tomato_radius, self.arm.base_pos)

        # Save current pose
        q_current = self.arm.joint_angles.copy()

        # Solve IK
        self.arm.set_joint_angles(q_current)
        solved, err, iters = self.arm.inverse_kinematics(
            cut_pt, max_iters=500, tol=5e-4)
        q_cut = self.arm.joint_angles.copy()

        if not solved and err > 0.02:
            self.status_var.set(
                f"Cannot reach tomato #{rank} "
                f"(IK err={err:.4f}m) — skipping")
            self.arm.set_joint_angles(q_current)
            self.harvest_index += 1
            self.root.after(500, self._harvest_next_tomato)
            return

        # Reset to current pose for animation
        self.arm.set_joint_angles(q_current)
        self.scissor_open_angle = math.radians(25)

        self.status_var.set(
            f"Approaching tomato #{rank} "
            f"(D={np.linalg.norm(pos)*100:.0f}cm)...")

        # Animate approach
        self._current_q_cut = q_cut
        self._current_tidx = tidx
        self._animate_to_target(q_current, q_cut, 0, 40,
                                 on_complete=self._cut_current_tomato)

    def _animate_to_target(self, q_start, q_end, step, steps, on_complete):
        """Animate arm from q_start to q_end, then call on_complete."""
        if self._closing:
            return
        alpha = step / steps
        q = (1 - alpha) * q_start + alpha * q_end
        self.arm.set_joint_angles(q)
        # Send to real servos
        pulses = angles_to_pulses(q)
        self.driver.move_servos(pulses, duration_ms=30)
        self.draw_arm()

        if step < steps:
            self.root.after(30, lambda: self._animate_to_target(
                q_start, q_end, step + 1, steps, on_complete))
        else:
            self.root.after(200, on_complete)

    def _cut_current_tomato(self):
        """Animate scissors closing on the current tomato's stem."""
        rank = self.harvest_index + 1
        self.status_var.set(f"Cutting tomato #{rank} stem...")
        self.driver.gripper_close(duration_ms=500)
        self._animate_cut(0, 20)

    def _animate_cut(self, step, steps):
        if self._closing:
            return
        t = step / steps
        self.scissor_open_angle = (1 - t) * math.radians(25)
        self.draw_arm()

        if step < steps:
            self.root.after(30, lambda: self._animate_cut(step + 1, steps))
        else:
            # Mark tomato as cut
            tidx = self._current_tidx
            self.tomato_states[tidx]["detached"] = True
            rank = self.harvest_index + 1
            self.harvested_count += 1
            self.status_var.set(f"Tomato #{rank} cut! Falling into net...")
            self.root.after(100, lambda: self._animate_fall(0, 30))

    def _animate_fall(self, step, steps):
        """Animate the cut tomato falling into the net."""
        if self._closing:
            return
        tidx = self._current_tidx
        t = step / steps
        original = self.tomato_states[tidx]["center"]
        z_target = max(0, self.net_rim_z - self.net_depth + 0.01)
        # Fall toward net center (x, y) and down (z)
        x = (1 - t) * original[0] + t * self.net_center[0]
        y = (1 - t) * original[1] + t * self.net_center[1]
        z = (1 - t) * original[2] + t * z_target
        self.tomato_states[tidx]["pos"] = np.array([x, y, z])
        self.draw_arm()

        if step < steps:
            self.root.after(25, lambda: self._animate_fall(step + 1, steps))
        else:
            # Reopen scissors and proceed to next tomato
            self.scissor_open_angle = math.radians(25)
            self.driver.gripper_open(duration_ms=400)
            self.harvest_index += 1

            remaining = len(self.harvest_queue) - self.harvest_index
            if remaining > 0:
                self.status_var.set(
                    f"Tomato caught! Moving to next ({remaining} remaining)...")
            else:
                self.status_var.set("All tomatoes cut! Returning home...")

            self.root.after(300, self._harvest_next_tomato)

    def _animate_retract_home(self):
        """Animate arm returning to search home (default position)."""
        q_current = self.arm.joint_angles.copy()
        self.scissor_open_angle = math.radians(25)
        self.status_var.set(
            f"All done! {self.harvested_count} tomato(es) harvested. "
            f"Returning to default position...")
        self._animate_to_target(q_current, self.home_angles, 0, 35,
                                 on_complete=self._harvest_complete)

    def _harvest_complete(self):
        """Called when the full harvest cycle is finished."""
        self.animation_running = False
        self.status_var.set(
            f"Harvest cycle complete! "
            f"{self.harvested_count}/{len(self.harvest_queue)} "
            f"tomato(es) harvested. Arm at default position.")

    # ══════════════════════════════════════════════════════════════════════
    #  Single Tomato Harvest (backward compat)
    # ══════════════════════════════════════════════════════════════════════

    def _start_single_harvest(self):
        """Harvest only tomato #1 (nearest)."""
        if self.animation_running:
            return
        if not self._update_tomato_states_from_gui():
            return

        ranked = self._get_ranked_tomatoes()
        tidx = ranked[0][1]  # nearest tomato index
        self.harvest_queue = [tidx]
        self.harvest_index = 0
        self.harvested_count = 0

        self.arm.set_joint_angles(self.home_angles)
        self.driver.go_search_home(duration_ms=1200)
        self.scissor_open_angle = math.radians(25)
        self.animation_running = True

        self.status_var.set("Single tomato harvest (nearest)...")
        self.draw_arm()
        self.root.after(500, self._harvest_next_tomato)

    # ══════════════════════════════════════════════════════════════════════
    #  Simple Actions
    # ══════════════════════════════════════════════════════════════════════

    def _go_home(self):
        if self.animation_running:
            return
        self.arm.set_joint_angles(self.home_angles)
        self.driver.go_search_home(duration_ms=1200)
        self.scissor_open_angle = math.radians(25)
        self.draw_arm()
        self.status_var.set("Arm at Search Home (default) position.")

    def _reset(self):
        self.animation_running = False
        self.arm.set_joint_angles(np.zeros(5))
        pulses = angles_to_pulses(np.zeros(5))
        self.driver.move_servos(pulses, duration_ms=1000)
        self.scissor_open_angle = math.radians(25)

        # Reset all tomato states
        for i, ts in enumerate(self.tomato_states):
            ts["pos"] = ts["center"].copy()
            ts["detached"] = False

        self.draw_arm()
        self.status_var.set("Arm and tomatoes reset.")

    # ══════════════════════════════════════════════════════════════════════
    #  Live Camera Integration
    # ══════════════════════════════════════════════════════════════════════

    def _toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.cam_btn_text.set("Start Camera Detection")
            if self.cap:
                if _VISION_AVAILABLE and isinstance(self.cap, ThreadedCamera):
                    self.cap.stop()
                else:
                    self.cap.release()
                self.cap = None
            try:
                cv2.destroyWindow("Camera Feed")
            except cv2.error:
                pass
        else:
            try:
                if _VISION_AVAILABLE:
                    self.cap = ThreadedCamera(camera_index=0,
                                              width=640, height=480)
                    self.cap.start()
                else:
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture('/dev/video0')
                    if not self.cap.isOpened():
                        raise RuntimeError("Cannot open camera 0")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except RuntimeError as e:
                messagebox.showerror("Camera Error", str(e))
                return
            self.camera_active = True
            self.cam_btn_text.set("Stop Camera Detection")
            self._camera_loop()

    def _camera_loop(self):
        if not self.camera_active or self._closing:
            return
        ret, frame = self.cap.read()
        if ret:
            if self.detector:
                dets = self.detector.detect(frame)

                # Fill up to 3 tomato positions from non-occluded detections
                visible_dets = [d for d in dets if not d.get("occluded", False)]
                if visible_dets:
                    from main import camera_to_arm_frame
                    for i in range(min(len(visible_dets), self.MAX_TOMATOES)):
                        arm_pos = camera_to_arm_frame(visible_dets[i]["xyz_cm"])
                        self.tomato_vars[i][0].set(f"{arm_pos[0]:.3f}")
                        self.tomato_vars[i][1].set(f"{arm_pos[1]:.3f}")
                        self.tomato_vars[i][2].set(f"{arm_pos[2]:.3f}")
                        self.tomato_states[i]["center"] = arm_pos.copy()
                        self.tomato_states[i]["pos"] = arm_pos.copy()
                        self.tomato_states[i]["detached"] = False

                    # --- AUTO-MOVE ARM TO FIRST DETECTED TOMATO ---
                    if not self.animation_running:
                        # Move to the first detected tomato
                        target = camera_to_arm_frame(visible_dets[0]["xyz_cm"])
                        # Use IK to move the arm
                        solved, err, iters = self.arm.inverse_kinematics(target)
                        if solved or err < 0.02:
                            self.status_var.set(f"Moved to detected tomato at {target} (err={err:.3f})")
                        else:
                            self.status_var.set(f"IK failed for detected tomato (err={err:.3f})")
                        self.draw_arm()

                if dets:
                    n = len(dets)
                    n_occ = sum(1 for d in dets if d.get("occluded", False))
                    ripeness = [d.get("ripeness_name", "?") for d in dets[:3]]
                    confs = [f"{d.get('track_conf', d['confidence']):.2f}"
                             for d in dets[:3]]
                    status_parts = [f"AI: {n} ripe tomato(es)"]
                    if n_occ:
                        status_parts.append(f"({n_occ} occluded)")
                    status_parts.append(
                        f"[{', '.join(ripeness)}] conf=[{','.join(confs)}]")
                    self.status_var.set(" ".join(status_parts))

                annotated = self.detector.annotate(frame, dets)
                cv2.imshow("Camera Feed", annotated)
            else:
                # No detector — show raw camera feed
                cv2.imshow("Camera Feed", frame)
                self.status_var.set("Camera: live feed (no YOLO model)")
            cv2.waitKey(1)

        # Poll camera at 60 FPS (~16ms)
        self.root.after(16, self._camera_loop)

    # ══════════════════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════════════════

    def _on_close(self):
        self._closing = True
        self.camera_active = False
        if self.cap:
            if _VISION_AVAILABLE and isinstance(self.cap, ThreadedCamera):
                self.cap.stop()
            else:
                self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def launch_gui():
    """Entry point called by main.py --gui."""
    root = tk.Tk()
    root.geometry("1200x700")
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
