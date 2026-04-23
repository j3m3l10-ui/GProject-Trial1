# Tomato Detection Project (YOLOv8)

This project uses YOLOv8 to detect ripe and unripe tomatoes in real-time using your webcam or an external camera (e.g., iPhone as webcam). It also estimates the distance to the detected tomato and simulates sending a command to a robotic arm to cut and collect ripe tomatoes.

## Project Structure

- `train.py`: Train YOLOv8 on your dataset.
- `detect.py`: Run real-time detection, estimate distance, and simulate robot commands.
- `requirements.txt`: Python dependencies.
- `data.yaml`: Dataset configuration for YOLOv8.
- `images/` and `labels/`: Your dataset (already provided).

## How to Use

### 1. Install Python and Dependencies

Open PowerShell in this folder and run:

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```
python train.py
```

This will create a trained model at `runs/detect/train/weights/best.pt`.

### 3. Run Real-Time Detection

```
python detect.py
```

- Press `q` to quit.
- The script will show bounding boxes, class (ripe/unripe), estimated distance, and print/simulate robot commands.

### 4. Using Your iPhone as a Webcam

- Install an app like EpocCam or iVCam on your iPhone and PC.
- Change the camera index in `detect.py` from `0` to `1` or `2` if needed.

## Sources
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_video_display.html)
- [YOLOv8 Custom Data Training](https://docs.ultralytics.com/tasks/detect/#train-on-custom-data)
- [How to Use iPhone as Webcam](https://www.pcmag.com/how-to/how-to-use-your-phone-as-a-webcam)
