"""
Simulation GUI — 3D Robotic Arm + Vision Tomato Harvesting
============================================================
Provides a safe, interactive 3D simulation for testing the full
detection → IK → cut → collect pipeline WITHOUT real hardware.

Features:
  - 3D animated arm with scissor end-effector
  - Simulated tomato at user-specified or vision-detected position
  - Full IK solving and trajectory animation
  - Net/basket always positioned under the tomato
  - Live camera feed panel (optional) with real YOLO detection
  - State machine: SCAN → CONFIRM → APPROACH → CUT → RETRACT

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
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from arm_controller import (
    FiveDOFArm, angles_to_pulses, search_home_angles,
    compute_cut_point, SERVO_IDS, SEARCH_HOME_PULSES,
)
from servo_driver import ServoDriver

# Try to import vision (may fail if no model weights yet — GUI still works)
_VISION_AVAILABLE = False
try:
    from vision import TomatoDetector
    _VISION_AVAILABLE = True
except Exception:
    pass


def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)


class SimulationGUI:
    """Full 3D simulation GUI for the tomato-harvesting robot."""

    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Harvester — 3D Simulation GUI")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Subsystems
        self.arm = FiveDOFArm()
        self.driver = ServoDriver(mode="sim")
        self.detector = TomatoDetector() if _VISION_AVAILABLE else None

        # Home angles
        self.home_angles = search_home_angles()
        self.arm.set_joint_angles(self.home_angles)

        # Tomato
        self.tomato_radius = 0.035
        self.tomato_center = np.array([0.20, 0.0, 0.15], dtype=float)
        self.tomato_pos = self.tomato_center.copy()
        self.tomato_detached = False

        # Net
        self.net_radius = 0.08
        self.net_rim_z = 0.04
        self.net_depth = 0.05

        # Scissor state
        self.scissor_open_angle = math.radians(25)
        self.scissor_blade_len = 0.04
        self.scissor_pivot_back = 0.01

        # State machine
        self.state = "IDLE"
        self.animation_running = False
        self.camera_active = False
        self.cap = None
        self._closing = False

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self.draw_arm()

    def _build_ui(self):
        # Main layout: left = 3D plot, right = controls
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 3D Plot
        plot_frame = ttk.LabelFrame(main_frame, text="3D Arm Simulation")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right panel
        right = ttk.Frame(main_frame, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # Tomato position inputs
        pos_frame = ttk.LabelFrame(right, text="Tomato Position (metres)")
        pos_frame.pack(fill=tk.X, pady=5)

        self.x_var = tk.StringVar(value=f"{self.tomato_center[0]:.3f}")
        self.y_var = tk.StringVar(value=f"{self.tomato_center[1]:.3f}")
        self.z_var = tk.StringVar(value=f"{self.tomato_center[2]:.3f}")

        for i, (label, var) in enumerate([("X:", self.x_var),
                                           ("Y:", self.y_var),
                                           ("Z:", self.z_var)]):
            ttk.Label(pos_frame, text=label).grid(row=i, column=0, padx=3, pady=2)
            ttk.Entry(pos_frame, textvariable=var, width=10).grid(
                row=i, column=1, padx=3, pady=2)

        # Action buttons
        btn_frame = ttk.LabelFrame(right, text="Actions")
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Simulate Full Harvest",
                   command=self._start_harvest_sim).pack(fill=tk.X, pady=2, padx=3)
        ttk.Button(btn_frame, text="Move to Search Home",
                   command=self._go_home).pack(fill=tk.X, pady=2, padx=3)
        ttk.Button(btn_frame, text="Solve IK Only (no motion)",
                   command=self._solve_ik_only).pack(fill=tk.X, pady=2, padx=3)
        ttk.Button(btn_frame, text="Reset Arm",
                   command=self._reset).pack(fill=tk.X, pady=2, padx=3)

        # Camera integration
        cam_frame = ttk.LabelFrame(right, text="Live Camera")
        cam_frame.pack(fill=tk.X, pady=5)

        self.cam_btn_text = tk.StringVar(value="Start Camera Detection")
        ttk.Button(cam_frame, textvariable=self.cam_btn_text,
                   command=self._toggle_camera).pack(fill=tk.X, pady=2, padx=3)
        ttk.Label(cam_frame, text="Detects tomato → sets position",
                  font=("", 8)).pack(padx=3)

        # Status
        status_frame = ttk.LabelFrame(right, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        self.status_var = tk.StringVar(value="Ready. Enter tomato position or start camera.")
        ttk.Label(status_frame, textvariable=self.status_var,
                  wraplength=250).pack(padx=5, pady=5)

        # Joint angles display
        joint_frame = ttk.LabelFrame(right, text="Joint Angles (deg) / Pulses")
        joint_frame.pack(fill=tk.X, pady=5)
        self.joint_labels = []
        names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]
        for i, name in enumerate(names):
            lbl = ttk.Label(joint_frame, text=f"{name}: 0.0° / 500")
            lbl.pack(anchor=tk.W, padx=5)
            self.joint_labels.append(lbl)

    # ── Drawing ────────────────────────────────────────────────────────────────
    def draw_arm(self, highlight_target=None):
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

        # Net under tomato
        self._draw_net()

        # Tomato sphere
        u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:10j]
        r = self.tomato_radius
        tp = self.tomato_pos
        x = tp[0] + r * np.cos(u) * np.sin(v)
        y = tp[1] + r * np.sin(u) * np.sin(v)
        z = tp[2] + r * np.cos(v)
        self.ax.plot_surface(x, y, z, color='red', alpha=0.6, linewidth=0)

        # Markers
        self.ax.scatter(*self.tomato_center, color='darkred', s=25, zorder=5)

        if highlight_target is not None:
            self.ax.scatter(*highlight_target, color='yellow', s=80, marker='*', zorder=5)

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
        self.ax.set_title("5-DOF Arm — Tomato Harvester Simulation")

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
        cx, cy = self.tomato_center[0], self.tomato_center[1]
        z0 = self.net_rim_z
        t = np.linspace(0, 2*np.pi, 50)
        xr = cx + self.net_radius * np.cos(t)
        yr = cy + self.net_radius * np.sin(t)
        self.ax.plot(xr, yr, np.full_like(t, z0), color='#8D6E63', linewidth=1.5)
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

    # ── Read tomato position from GUI ──────────────────────────────────────────
    def _read_tomato_pos(self):
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            return np.array([x, y, z], dtype=float)
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid X, Y, Z values.")
            return None

    # ── Actions ────────────────────────────────────────────────────────────────
    def _start_harvest_sim(self):
        if self.animation_running:
            return
        pos = self._read_tomato_pos()
        if pos is None:
            return

        self.tomato_center = pos.copy()
        self.tomato_pos = pos.copy()
        self.tomato_detached = False

        if not self.arm.is_reachable(pos):
            messagebox.showwarning("Out of Reach",
                                   f"Target {pos} is beyond arm reach "
                                   f"({self.arm.max_reach():.3f}m).")
            return

        edge_pt, cut_pt = compute_cut_point(pos, self.tomato_radius, self.arm.base_pos)

        # Start from Search Home
        self.arm.set_joint_angles(self.home_angles)
        q_home = self.arm.joint_angles.copy()

        # Solve IK for cut point
        solved, err, iters = self.arm.inverse_kinematics(cut_pt, max_iters=500, tol=5e-4)
        q_cut = self.arm.joint_angles.copy()

        self.status_var.set(f"IK: solved={solved}, err={err:.4f}m, iters={iters}\n"
                           f"Cut point: [{cut_pt[0]:.3f}, {cut_pt[1]:.3f}, {cut_pt[2]:.3f}]")

        if not solved and err > 0.02:
            messagebox.showwarning("IK Failed", f"Cannot reach cut point (err={err:.4f}m)")
            return

        # Reset arm to home for animation
        self.arm.set_joint_angles(q_home)
        self.scissor_open_angle = math.radians(25)
        self.animation_running = True

        # Animate: home → cut → close scissors → tomato falls → retract
        self._animate_approach(q_home, q_cut, step=0, steps=40)

    def _animate_approach(self, q_start, q_end, step, steps):
        if self._closing:
            return
        alpha = step / steps
        q = (1 - alpha) * q_start + alpha * q_end
        self.arm.set_joint_angles(q)
        self.draw_arm(highlight_target=self.tomato_center)

        if step < steps:
            self.root.after(30, lambda: self._animate_approach(
                q_start, q_end, step + 1, steps))
        else:
            self.status_var.set(self.status_var.get() + "\nApproach complete. Cutting...")
            self.root.after(200, lambda: self._animate_cut(0, 20))

    def _animate_cut(self, step, steps):
        if self._closing:
            return
        t = step / steps
        self.scissor_open_angle = (1 - t) * math.radians(25)
        self.draw_arm()

        if step < steps:
            self.root.after(30, lambda: self._animate_cut(step + 1, steps))
        else:
            self.tomato_detached = True
            self.status_var.set(self.status_var.get() + "\nCUT DONE. Tomato falling...")
            self.root.after(100, lambda: self._animate_fall(0, 35))

    def _animate_fall(self, step, steps):
        if self._closing:
            return
        t = step / steps
        z_target = max(0, self.net_rim_z - self.net_depth + 0.01)
        z = (1 - t) * self.tomato_center[2] + t * z_target
        self.tomato_pos = np.array([self.tomato_center[0],
                                    self.tomato_center[1], z])
        self.draw_arm()

        if step < steps:
            self.root.after(25, lambda: self._animate_fall(step + 1, steps))
        else:
            self.status_var.set(self.status_var.get() + "\nCaught in net! Retracting...")
            self.root.after(300, lambda: self._animate_retract(
                self.arm.joint_angles.copy(), self.home_angles, 0, 35))

    def _animate_retract(self, q_start, q_end, step, steps):
        if self._closing:
            return
        alpha = step / steps
        q = (1 - alpha) * q_start + alpha * q_end
        self.arm.set_joint_angles(q)
        self.scissor_open_angle = math.radians(25)  # reopen scissors
        self.draw_arm()

        if step < steps:
            self.root.after(30, lambda: self._animate_retract(
                q_start, q_end, step + 1, steps))
        else:
            self.animation_running = False
            self.status_var.set(self.status_var.get() + "\nHarvest cycle complete!")

    def _go_home(self):
        if self.animation_running:
            return
        self.arm.set_joint_angles(self.home_angles)
        self.scissor_open_angle = math.radians(25)
        self.draw_arm()
        self.status_var.set("Arm at Search Home position.")

    def _solve_ik_only(self):
        pos = self._read_tomato_pos()
        if pos is None:
            return
        self.tomato_center = pos.copy()
        self.tomato_pos = pos.copy()
        edge_pt, cut_pt = compute_cut_point(pos, self.tomato_radius, self.arm.base_pos)
        self.arm.set_joint_angles(self.home_angles)
        solved, err, iters = self.arm.inverse_kinematics(cut_pt, max_iters=500, tol=5e-4)
        self.draw_arm(highlight_target=cut_pt)
        self.status_var.set(f"IK: solved={solved}, err={err:.4f}m, iters={iters}\n"
                           f"Arm jumped to cut pose (no animation).")

    def _reset(self):
        self.animation_running = False
        self.arm.set_joint_angles(np.zeros(5))
        self.tomato_pos = self.tomato_center.copy()
        self.tomato_detached = False
        self.scissor_open_angle = math.radians(25)
        self.draw_arm()
        self.status_var.set("Arm reset to zero angles.")

    # ── Live camera integration ────────────────────────────────────────────────
    def _toggle_camera(self):
        if not _VISION_AVAILABLE:
            messagebox.showinfo("No Model", "Vision module unavailable (no model weights).")
            return

        if self.camera_active:
            self.camera_active = False
            self.cam_btn_text.set("Start Camera Detection")
            if self.cap:
                self.cap.release()
                self.cap = None
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot open camera.")
                return
            self.camera_active = True
            self.cam_btn_text.set("Stop Camera Detection")
            self._camera_loop()

    def _camera_loop(self):
        if not self.camera_active or self._closing:
            return
        ret, frame = self.cap.read()
        if ret and self.detector:
            dets = self.detector.detect(frame)
            if dets:
                best = max(dets, key=lambda d: d["confidence"])
                # Convert to arm frame and update GUI fields
                xyz = best["xyz_cm"]
                from main import camera_to_arm_frame
                arm_pos = camera_to_arm_frame(xyz, self.arm)
                self.x_var.set(f"{arm_pos[0]:.3f}")
                self.y_var.set(f"{arm_pos[1]:.3f}")
                self.z_var.set(f"{arm_pos[2]:.3f}")
                self.tomato_center = arm_pos.copy()
                self.tomato_pos = arm_pos.copy()
                self.draw_arm()
                self.status_var.set(
                    f"Camera: tomato at X={arm_pos[0]:.3f} "
                    f"Y={arm_pos[1]:.3f} Z={arm_pos[2]:.3f}m  "
                    f"(conf {best['confidence']:.2f})")

            annotated = self.detector.annotate(frame, dets)
            cv2.imshow("Camera Feed", annotated)
            cv2.waitKey(1)

        self.root.after(100, self._camera_loop)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._closing = True
        self.camera_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


def launch_gui():
    """Entry point called by main.py --gui."""
    root = tk.Tk()
    root.geometry("1100x650")
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
