"""
Training script — Ripe Tomato Detector
=======================================
Upgrade notes vs. the original script
---------------------------------------
- Backbone: yolov8s.pt  (small — significantly more accurate than nano with
  manageable extra compute cost).
- Higher resolution (imgsz=800) to better resolve small / distant tomatoes.
- More epochs (100) with early stopping (patience=20).
- `copy_paste` + `mosaic` + `mixup` augmentation forces the model to handle
  partial/occluded tomatoes and cluttered scenes — the key to telling a
  tomato apart from red balls, cups, fabric, etc.
- `hsv_h / hsv_s / hsv_v` jitter makes the colour decision robust to both
  lighting changes AND to "plain red" objects (saturated balls) because the
  model learns to rely on more than hue alone.
- `erasing` randomly blanks patches of real tomatoes so the network must
  learn the full texture (stem / shoulders / gloss), not just a red disc.
- `dropout=0.1` and `label_smoothing=0.05` reduce single-class over-
  confidence — which is what was producing "ball = tomato" false positives.

To add hard negatives (red balls, red cups, tomatoes-on-a-poster, faces, etc.)
to the dataset:
  1. Drop background images (no label file, or empty label file) into
     images/train/  — YOLO treats images with no labels as background-only
     and will learn to suppress detections on them.
  2. Re-run this script.  No code change needed.  The `collect_negatives.py`
     helper in this repo is the recommended way to build that set.
"""

from ultralytics import YOLO


def main():
    # yolov8s gives a much better precision/recall trade-off than yolov8n
    # while still running comfortably on a mid-range GPU or modern CPU.
    model = YOLO('yolov8s.pt')

    model.train(
        data='data.yaml',
        epochs=100,
        patience=20,          # early-stop if no improvement for 20 epochs
        imgsz=800,            # larger input → better small-tomato recall
        batch=8,              # lower batch for larger imgsz; increase if VRAM allows
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        dropout=0.1,          # head dropout — reduces single-class overfitting
        label_smoothing=0.05, # prevents over-confident false positives
        # ── Augmentation ────────────────────────────────────────────────────
        hsv_h=0.02,           # hue jitter — slightly stronger than before
        hsv_s=0.8,            # saturation jitter — critical for colour robustness
        hsv_v=0.5,            # brightness jitter — handles lighting changes
        degrees=15.0,         # rotation
        translate=0.1,
        scale=0.5,            # scale jitter — handles near/far tomatoes
        shear=2.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,           # mosaic on (default) — improves generalisation
        mixup=0.15,           # blends two images — strong regulariser
        copy_paste=0.3,       # paste random tomatoes onto random backgrounds
        erasing=0.4,          # random erasing — forces full-texture learning
        # ── Output ──────────────────────────────────────────────────────────
        project='runs/detect',
        name='train_v2',
        save=True,
        plots=True,
    )

    print("\n[DONE] Best weights saved to: runs/detect/train_v2/weights/best.pt")
    print("detect.py / vision.py auto-resolve this path — no code change needed.")


if __name__ == "__main__":
    main()
