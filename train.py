"""
Training script — Ripe Tomato Detector
=======================================
Upgrade notes vs. the original script
---------------------------------------
- Backbone: yolov8s.pt  (small — significantly more accurate than nano with
  manageable extra compute cost)
- Higher resolution (imgsz=800) to better resolve small / distant tomatoes
- More epochs (100) with early stopping (patience=20)
- copy_paste + mosaic augmentation forces the model to handle partial / occluded
  tomatoes, reducing false positives on similar-looking non-tomato objects
- flipud, degrees, hsv_h/s/v variation all increase colour/pose robustness
- dropout=0.1 in the head reduces overfitting on a small dataset
- label_smoothing=0.05 prevents over-confidence on the single class

To add hard negatives (faces, random objects) to the dataset:
  1. Drop background images (no label file, or empty label file) into
     images/train/  — YOLO treats images with no labels as background-only
     and will learn to suppress detections on them.
  2. Re-run this script.  No code change needed.
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
        hsv_h=0.015,          # hue jitter
        hsv_s=0.7,            # saturation jitter — critical for colour robustness
        hsv_v=0.4,            # brightness jitter
        degrees=10.0,         # rotation
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,           # mosaic on (default) — improves generalisation
        copy_paste=0.3,       # paste random tomatoes onto random backgrounds
        # ── Output ──────────────────────────────────────────────────────────
        project='runs/detect',
        name='train_v2',
        save=True,
        plots=True,
    )

    print("\n[DONE] Best weights saved to: runs/detect/train_v2/weights/best.pt")
    print("Update MODEL_PATH in detect.py to use the new weights.")


if __name__ == "__main__":
    main()
