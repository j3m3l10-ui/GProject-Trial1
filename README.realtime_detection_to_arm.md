# Real-Time Tomato Detection to Arm Movement Workflow

This workflow enables your robot to detect ripe tomatoes using the camera and immediately move the arm to harvest them, using two coordinated scripts:

## 1. Detection Script (`detect.py`)
- Runs YOLOv8 detection on the camera feed.
- Writes detected tomato positions (3D, in cm) to `detected_tomatoes.json` in real time.

## 2. Arm Controller Script (`main.py --from-file`)
- Reads `detected_tomatoes.json` for new detections.
- Moves the arm to each detected tomato position and performs the harvest action.

---

## How to Use

### Step 1: Start Detection
Open a terminal and run:
```sh
python detect.py
```
This will start the camera, run detection, and continuously update `detected_tomatoes.json` with the latest ripe tomato positions.

### Step 2: Start Arm Controller
Open a second terminal and run:
```sh
python main.py --from-file
```
This will:
- Wait for new detections in `detected_tomatoes.json`.
- Move the arm to each detected tomato and perform the harvest sequence.
- Return the arm to the home position after each cycle.

### Notes
- You can run both scripts in simulation mode by adding `--sim` to the `main.py` command.
- Make sure only one process is using the camera at a time.
- The system will skip unreachable tomatoes and log warnings if inverse kinematics cannot solve a pose.
- The workflow is robust to repeated detections and will only act on new data.

---

## Troubleshooting
- If the arm does not move, check the logs for warnings about reachability or IK errors.
- If no tomatoes are detected, ensure the camera is unobstructed and a ripe tomato (or red object) is visible.
- If you see file access errors, ensure both scripts are running in the same directory.

---

For further customization or integration (e.g., networked control, advanced filtering), contact your developer or request additional features.
