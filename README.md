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

### 3. Run Hardware Harvesting

On the Raspberry Pi with the camera and Hiwonder arm connected:

```
python main.py --hardware
```

or:

```
python detect.py
```

Both commands run the integrated hardware pipeline: camera detection → IK →
servo commands. The default servo backend tries Hiwonder's
`Board.setBusServoPulse` SDK first, then falls back to raw UART on
`/dev/ttyAMA0`.

When a ripe tomato appears, the robot now records a 3-second validation
snapshot, ranks up to the 3 closest tomatoes, and harvests them in that order.
During each approach the camera keeps streaming live detections so the arm
moves in small relative image-centering IK steps until the tomato nearly fills
the frame, then cuts 1 cm vertically above the live tomato centre.

If the arm still does not move, force the hardware backend explicitly:

```
python main.py --hardware --servo-backend sdk
```

or:

```
python main.py --hardware --servo-backend uart --uart-port /dev/ttyAMA0
```

Camera auto-detection is enabled by default. If OpenCV still cannot open the
camera, pass the known index explicitly:

```
python main.py --hardware --camera 1
```

If V4L2/OpenCV frame grabbing fails but your camera is already published by
ROS, run from a sourced ROS shell and subscribe to the image topic instead:

```
python main.py --hardware --camera-source ros --ros-image-topic /usb_cam/image_raw
```

If `rostopic list | grep image` shows no image topics, start the Hiwonder
USB/UVC camera through ROS first. These cameras are driver-free USB cameras, so
the standard ROS1 `usb_cam` node is the expected bridge:

```
ls /opt/ros/
chmod +x scripts/install_ros_usb_camera_deps.sh scripts/start_hiwonder_camera_ros.sh scripts/check_ros_camera_topic.sh
./scripts/install_ros_usb_camera_deps.sh
export ROS_DISTRO=$(ls /opt/ros | sort | head -n 1)
source /opt/ros/$ROS_DISTRO/setup.bash
./scripts/start_hiwonder_camera_ros.sh /dev/video0 640 480 30 mjpeg
```

If `/opt/ros` does not exist, ROS is not installed. Install a ROS1 distro for
your OS first. This project uses ROS1 APIs (`rospy`, `rostopic`, `roslaunch`),
not ROS2.

On Debian 13, do not install ROS Noetic manually unless you are prepared to
build unsupported packages from source. Use the direct OpenCV/V4L2 path for the
Hiwonder driver-free USB camera instead:

```
sudo apt update
sudo apt install -y v4l-utils
chmod +x scripts/list_hiwonder_cameras.sh scripts/run_harvester_debian_opencv.sh scripts/test_opencv_camera.py
./scripts/list_hiwonder_cameras.sh
python3 scripts/test_opencv_camera.py --camera 0
./scripts/run_harvester_debian_opencv.sh 0 auto
```

If `/dev/video0` does not stream frames, replace `0` with the capture-capable
index shown by `scripts/list_hiwonder_cameras.sh`.

In a second terminal:

```
source /opt/ros/$ROS_DISTRO/setup.bash
./scripts/check_ros_camera_topic.sh /hiwonder_usb_camera/image_raw
python main.py --hardware --camera-source ros --ros-image-topic /hiwonder_usb_camera/image_raw --servo-backend auto
```

### 4. Run Camera-Only Detection

```
python detect.py --vision-only
```

- Press `q` to quit.
- The script will show bounding boxes and print detections without moving the arm.

### 5. Using Your iPhone as a Webcam

- Install an app like EpocCam or iVCam on your iPhone and PC.
- Change the camera index in `detect.py` from `0` to `1` or `2` if needed.

## Sources
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_video_display.html)
- [YOLOv8 Custom Data Training](https://docs.ultralytics.com/tasks/detect/#train-on-custom-data)
- [How to Use iPhone as Webcam](https://www.pcmag.com/how-to/how-to-use-your-phone-as-a-webcam)
