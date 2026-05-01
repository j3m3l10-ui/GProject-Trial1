#!/usr/bin/env python3
"""Capture camera->arm correspondence points for rigid/affine calibration."""

import csv
import os

import cv2

from main import _open_low_latency_camera
from vision import TomatoDetector


CSV_PATH = os.path.join(os.path.dirname(__file__), "cam_arm_points.csv")
CAMERA_INDEX = 0


def _ensure_csv_header(path):
	if os.path.isfile(path) and os.path.getsize(path) > 0:
		return
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"cam_x_cm",
			"cam_y_cm",
			"cam_z_cm",
			"arm_x_m",
			"arm_y_m",
			"arm_z_m",
			"note",
		])


def _append_row(path, cam_xyz, arm_xyz, note="manual"):
	with open(path, "a", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			f"{cam_xyz['x']:.6f}",
			f"{cam_xyz['y']:.6f}",
			f"{cam_xyz['z']:.6f}",
			f"{arm_xyz[0]:.6f}",
			f"{arm_xyz[1]:.6f}",
			f"{arm_xyz[2]:.6f}",
			note,
		])


def _parse_arm_xyz(text):
	parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
	if len(parts) != 3:
		raise ValueError("Please enter exactly 3 numbers: arm_x_m arm_y_m arm_z_m")
	return [float(parts[0]), float(parts[1]), float(parts[2])]


def main():
	_ensure_csv_header(CSV_PATH)
	detector = TomatoDetector(imgsz=416)
	cap = _open_low_latency_camera(CAMERA_INDEX)

	print("Calibration capture started.")
	print("Keys: c=capture nearest tomato, q=quit")
	print(f"Saving points to: {CSV_PATH}")

	try:
		while True:
			ok, frame = cap.read()
			if not ok or frame is None:
				continue

			detections = detector.detect(frame, use_tracker=False)
			annotated = detector.annotate(frame, detections)

			nearest = detections[0] if detections else None
			if nearest is not None:
				xyz = nearest["xyz_cm"]
				info = (
					f"Nearest xyz_cm=({xyz['x']:.1f}, {xyz['y']:.1f}, {xyz['z']:.1f}) "
					f"dist={nearest.get('distance_cm', 0):.1f}"
				)
				cv2.putText(
					annotated,
					info,
					(10, 24),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 255, 255),
					2,
					cv2.LINE_AA,
				)

			cv2.imshow("Capture Cam-Arm Points", annotated)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
			if key == ord("c"):
				if nearest is None:
					print("No tomato detection to capture.")
					continue

				cam_xyz = nearest["xyz_cm"]
				print(
					"Captured camera xyz_cm:",
					f"({cam_xyz['x']:.3f}, {cam_xyz['y']:.3f}, {cam_xyz['z']:.3f})",
				)
				raw = input("Enter arm_x_m arm_y_m arm_z_m for this same point: ").strip()
				try:
					arm_xyz = _parse_arm_xyz(raw)
				except ValueError as exc:
					print(f"Invalid input: {exc}")
					continue

				_append_row(CSV_PATH, cam_xyz, arm_xyz)
				print("Saved calibration row.")
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
