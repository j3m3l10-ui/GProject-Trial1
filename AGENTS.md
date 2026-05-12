# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Tomato Harvesting Robot — Python project using YOLOv8 for ripe-tomato detection and a 5-DOF robotic arm (Hiwonder ArmPi Pro) for autonomous harvesting. See `README.md` for general usage.

### Running the application

- **Simulation (headless, no hardware):** `python3 main.py --sim` — initializes vision + arm + servo driver in sim mode. Requires a camera device (`/dev/video0`); will exit early in cloud VMs with "Cannot open camera" (expected).
- **GUI simulation:** `python3 main.py --gui` — launches Tkinter/Matplotlib 3D GUI. Requires `DISPLAY` set to a running X server (use `Xvfb :99 -screen 0 1280x1024x24 &` then `export DISPLAY=:99`, or use `:1` if the desktop is available). Requires `python3-tk` system package.
- **Standalone detection:** `python3 detect.py` — requires a camera device and display.

### Key gotchas

- `detect.py` opens a camera and display at **module import time** (top-level code), so importing it in a headless environment will raise `RuntimeError`. Use `vision.py` (`TomatoDetector` class) instead for programmatic detection.
- The pre-trained model weights live at `runs/detect/train6/weights/best.pt` (committed). A retrained model would go to `runs/detect/train_v2/weights/best.pt`; `vision.py` auto-resolves whichever exists.
- `python3-tk` is a system package (not in `requirements.txt`) needed for `--gui` mode.
- No automated test suite exists. Verify correctness by running the pipeline in sim mode or instantiating `FiveDOFArm` / `TomatoDetector` programmatically.

### Linting

No linter is configured in the project. `ruff check *.py` can be used; existing code has minor style warnings (unused imports, semicolons) that are pre-existing.

### Compile check

`python3 -m py_compile <file>` on all `*.py` files should pass cleanly.
