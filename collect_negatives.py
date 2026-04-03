"""
Hard-Negative Collector for Ripe-Tomato Detector
=================================================
Adds background-only (label-less) images to train/images/ so YOLOv8 learns
to suppress detections on non-tomato objects (faces, hands, red objects, etc.)

YOLO rule: if an image in images/train/ has NO matching .txt file in
labels/train/, it is treated as a background-only sample and the model is
penalised for firing any detection on it.

Three modes
-----------
  python collect_negatives.py download   # pull ~100 diverse images from net
  python collect_negatives.py webcam     # live capture: press S to save frame
  python collect_negatives.py generate   # create synthetic hard negatives
  python collect_negatives.py all        # run download + generate together

Usage example
-------------
  python collect_negatives.py all
  # Then re-train:
  python train.py
  # Then update MODEL_PATH in detect.py to runs/detect/train_v2/weights/best.pt
"""

import sys
import os
import time
import urllib.request
import urllib.error
import uuid
import cv2
import numpy as np

# ── Output directory (must match data.yaml  train: ../train/images) ──────────
NEG_DIR = os.path.join(os.path.dirname(__file__), "train", "images")
os.makedirs(NEG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(img_bgr: np.ndarray, tag: str) -> str:
    """Save a BGR image to NEG_DIR with a unique name; return the path."""
    name = f"neg_{tag}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(NEG_DIR, name)
    cv2.imwrite(path, img_bgr)
    return path


def _download_jpg(url: str, timeout: int = 10) -> np.ndarray | None:
    """Download an image from url, decode to BGR ndarray, or return None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        print(f"  [warn] {url} — {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Download
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple: (label_tag, url_template, count)
# picsum.photos gives random CC-licensed photos (landscapes, people, objects)
# thispersondoesnotexist.com gives AI-generated faces (not real people, CC0)
DOWNLOAD_SOURCES = [
    # --- Faces & people (most important — direct fix for the face FP) ---
    ("face",    "https://thispersondoesnotexist.com",                        25),

    # --- General scenes (outdoor, indoor, clutter — reduces background FP) -
    ("scene",   "https://picsum.photos/640/480?random={i}",                  30),

    # --- Specific red-obj seeds via picsum (red ≠ tomato shapes) ----------
    ("scene2",  "https://picsum.photos/seed/{i}a/640/480",                   20),

    # --- Portrait-crop sized images (same aspect ratio as tomato close-up) -
    ("portrait","https://picsum.photos/480/480?random={i}99",                15),
]


def run_download() -> None:
    print("\n=== MODE: download ===")
    saved = 0
    try:
        for tag, url_tpl, count in DOWNLOAD_SOURCES:
            print(f"\n  Downloading {count}× '{tag}' images …")
            for i in range(count):
                url = url_tpl.format(i=i)
                img = _download_jpg(url)
                if img is not None:
                    path = _save(img, tag)
                    print(f"    [{saved+1:>3}] saved → {os.path.basename(path)}")
                    saved += 1
                time.sleep(0.3)   # be polite to the servers
    except KeyboardInterrupt:
        print(f"\n  [interrupted] Saved {saved} images so far.")

    print(f"\n  ✓ Downloaded {saved} hard-negative images into {NEG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Webcam capture
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam() -> None:
    print("\n=== MODE: webcam ===")
    print("  Point the camera at objects that trigger false positives.")
    print("  Controls:  S = save frame  |  Q = quit\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [error] Cannot open camera.")
        return

    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, "CAPTURING NEGATIVES — Press S to save, Q to quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        cv2.putText(disp, f"Saved: {saved}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
        cv2.imshow("Negative Collector", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            path = _save(frame, "webcam")
            print(f"  [{saved+1:>3}] saved → {os.path.basename(path)}")
            saved += 1
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  ✓ Captured {saved} frames into {NEG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Synthetic generation
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(seed=42)


def _rand_color():
    return tuple(int(c) for c in RNG.integers(0, 255, 3))


def _make_random_background(w=640, h=480) -> np.ndarray:
    """Solid or gradient background — teaches the model that plain colours ≠ tomato."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    c1 = _rand_color()
    c2 = _rand_color()
    for y in range(h):
        t = y / h
        img[y] = tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3))
    return img


def _make_red_non_tomato(w=640, h=480) -> np.ndarray:
    """
    Red/orange shapes that are NOT round — elongated blobs, stars, etc.
    These specifically attack the colour gate (they pass the HSV test) so the
    model must learn to rely on shape and context, not colour alone.
    """
    img = RNG.integers(40, 160, (h, w, 3), dtype=np.uint8)
    # Draw 2-5 elongated red rectangles / polygons
    n = int(RNG.integers(2, 6))
    for _ in range(n):
        x = int(RNG.integers(0, w - 1))
        y = int(RNG.integers(0, h - 1))
        # Very elongated (aspect ratio > 3  → not circular → fails real tomato shape)
        rw = int(RNG.integers(80, 250))
        rh = int(RNG.integers(10, 40))
        angle = float(RNG.integers(0, 180))
        color = (int(RNG.integers(0, 40)),            # B (low)
                 int(RNG.integers(0, 80)),             # G (low)
                 int(RNG.integers(180, 255)))          # R (high→ red)
        center = (x, y)
        axes = (rw // 2, rh // 2)
        cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
    return img


def _make_noisy_red_texture(w=640, h=480) -> np.ndarray:
    """Red/brown noisy texture — e.g., brick wall, terracotta, skin."""
    # HSV: reddish hue, moderate saturation, variable brightness
    h_base = int(RNG.integers(0, 15))   # hue ≈ red
    img_hsv = np.zeros((h, w, 3), dtype=np.uint8)
    img_hsv[:, :, 0] = np.clip(
        h_base + RNG.integers(-5, 5, (h, w), dtype=np.int16), 0, 179
    ).astype(np.uint8)
    img_hsv[:, :, 1] = np.clip(
        100 + RNG.integers(-40, 40, (h, w), dtype=np.int16), 0, 255
    ).astype(np.uint8)
    img_hsv[:, :, 2] = np.clip(
        150 + RNG.integers(-80, 80, (h, w), dtype=np.int16), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def _make_cluttered_scene(w=640, h=480) -> np.ndarray:
    """Many small, multi-coloured circles — teaches 'not every circle is a tomato'."""
    bg = RNG.integers(20, 200, (h, w, 3), dtype=np.uint8)
    for _ in range(int(RNG.integers(15, 40))):
        cx = int(RNG.integers(0, w))
        cy = int(RNG.integers(0, h))
        r  = int(RNG.integers(5, 35))
        col = _rand_color()
        cv2.circle(bg, (cx, cy), r, col, -1)
    return bg


def _make_face_oval(w=640, h=480) -> np.ndarray:
    """
    Simplified synthetic 'skin-tone oval' — mimics the aspect ratio of a face.
    Forces the model to learn that skin-coloured ovals ≠ tomato.
    """
    img = RNG.integers(100, 180, (h, w, 3), dtype=np.uint8)
    # skin tone in BGR
    skin_b = int(RNG.integers(80, 130))
    skin_g = int(RNG.integers(100, 160))
    skin_r = int(RNG.integers(160, 220))
    cx, cy = w // 2, h // 2
    ax = int(RNG.integers(100, 180))   # wider
    ay = int(RNG.integers(150, 230))   # taller → aspect ratio > 1.3  (not tomato)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (skin_b, skin_g, skin_r), -1)
    return img


GENERATORS = [
    ("bg",       _make_random_background),
    ("redshape",  _make_red_non_tomato),
    ("redtex",   _make_noisy_red_texture),
    ("clutter",  _make_cluttered_scene),
    ("faceoval", _make_face_oval),
]
PER_GENERATOR = 20   # images per generator type


def run_generate() -> None:
    print("\n=== MODE: generate (synthetic negatives) ===")
    saved = 0
    for tag, fn in GENERATORS:
        print(f"  Generating {PER_GENERATOR}× '{tag}' …")
        for _ in range(PER_GENERATOR):
            img = fn()
            _save(img, tag)
            saved += 1
    print(f"\n  ✓ Generated {saved} synthetic negatives in {NEG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print(f"[INFO] Saving hard-negative images (no labels) → {NEG_DIR}")

    if mode in ("download", "all"):
        run_download()
    if mode in ("generate", "all"):
        run_generate()
    if mode == "webcam":
        run_webcam()

    if mode not in ("download", "generate", "webcam", "all"):
        print(__doc__)
        sys.exit(1)

    # Final sanity check: confirm no stray .txt files were created
    label_dir = NEG_DIR.replace(
        os.path.join("train", "images"),
        os.path.join("train", "labels")
    )
    if os.path.isdir(label_dir):
        neg_labels = [f for f in os.listdir(label_dir) if f.startswith("neg_")]
        if neg_labels:
            print(f"\n[WARN] Found {len(neg_labels)} label files for negatives — "
                  "deleting them now so YOLO treats them as background.")
            for f in neg_labels:
                os.remove(os.path.join(label_dir, f))

    print("\n[DONE] Hard negatives added.")
    print("Next step → python train.py")
    print("Then update MODEL_PATH in detect.py to the new best.pt path.")


if __name__ == "__main__":
    main()
