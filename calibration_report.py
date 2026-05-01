#!/usr/bin/env python3
"""Report camera-to-arm calibration quality from point correspondences."""

import argparse
import csv
import os

import numpy as np


def _load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            try:
                cam = np.array([
                    float(row["cam_x_cm"]),
                    float(row["cam_y_cm"]),
                    float(row["cam_z_cm"]),
                ], dtype=float) / 100.0
                arm = np.array([
                    float(row["arm_x_m"]),
                    float(row["arm_y_m"]),
                    float(row["arm_z_m"]),
                ], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"line": i, "cam": cam, "arm": arm})
    return rows


def _fit_affine(cam_pts, arm_pts):
    cam_h = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1), dtype=float)])
    coeffs, *_ = np.linalg.lstsq(cam_h, arm_pts, rcond=None)
    pred = cam_h @ coeffs
    residual_vec = pred - arm_pts
    residual_norm = np.linalg.norm(residual_vec, axis=1)
    rms_m = float(np.sqrt(np.mean(residual_norm ** 2)))
    mean_m = float(np.mean(residual_norm))
    max_m = float(np.max(residual_norm))
    return coeffs, pred, residual_vec, residual_norm, rms_m, mean_m, max_m


def main():
    parser = argparse.ArgumentParser(description="Evaluate camera-arm calibration CSV")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "cam_arm_points.csv"),
        help="Path to calibration CSV",
    )
    parser.add_argument(
        "--rms-threshold-cm",
        type=float,
        default=3.0,
        help="Pass/fail threshold for RMS error in cm",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = _load_rows(csv_path)
    if len(rows) < 4:
        raise SystemExit(f"Need at least 4 valid points, found {len(rows)} in {csv_path}")

    cam_pts = np.array([r["cam"] for r in rows], dtype=float)
    arm_pts = np.array([r["arm"] for r in rows], dtype=float)
    coeffs, pred, residual_vec, residual_norm, rms_m, mean_m, max_m = _fit_affine(cam_pts, arm_pts)

    print("Calibration Report")
    print(f"CSV: {csv_path}")
    print(f"Points: {len(rows)}")
    print("-")
    print(f"RMS error:  {rms_m*100.0:.2f} cm")
    print(f"Mean error: {mean_m*100.0:.2f} cm")
    print(f"Max error:  {max_m*100.0:.2f} cm")
    print("-")

    threshold_m = max(0.0, float(args.rms_threshold_cm)) / 100.0
    if rms_m <= threshold_m:
        print(f"Status: PASS (RMS <= {args.rms_threshold_cm:.2f} cm)")
    else:
        print(f"Status: FAIL (RMS > {args.rms_threshold_cm:.2f} cm)")

    print("-")
    print("Affine matrix B (arm = [cam,1] @ B):")
    print(np.array2string(coeffs, precision=6, suppress_small=True))
    print("-")
    print("Per-point residuals:")
    print("line | cam_xyz_cm -> arm_xyz_m | pred_arm_xyz_m | err_cm")
    for i, row in enumerate(rows):
        cam_cm = row["cam"] * 100.0
        arm = row["arm"]
        pred_i = pred[i]
        err_cm = residual_norm[i] * 100.0
        print(
            f"{row['line']:>4} | "
            f"[{cam_cm[0]:6.2f}, {cam_cm[1]:6.2f}, {cam_cm[2]:6.2f}] -> "
            f"[{arm[0]: .4f}, {arm[1]: .4f}, {arm[2]: .4f}] | "
            f"[{pred_i[0]: .4f}, {pred_i[1]: .4f}, {pred_i[2]: .4f}] | "
            f"{err_cm:6.2f}"
        )

    worst_idx = int(np.argmax(residual_norm))
    worst = rows[worst_idx]
    worst_vec_cm = residual_vec[worst_idx] * 100.0
    print("-")
    print(
        "Worst point: "
        f"line {worst['line']} with vector error "
        f"[{worst_vec_cm[0]:.2f}, {worst_vec_cm[1]:.2f}, {worst_vec_cm[2]:.2f}] cm"
    )


if __name__ == "__main__":
    main()