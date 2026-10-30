# ROS 2 Hand-Eye Calibration

Scripts for calibrating an Intel RealSense camera mounted on a 6-DOF robotic arm.

Uses a 7x5 ChArUco board with 30 mm squares and 20 mm markers, and solves the transformation via cv2.calibrateHandEye.

## Dependencies

    pip install opencv-contrib-python numpy scipy rosbags

## 1. Offline Calibration from ROS 2 Bag (bag_calib.py)

Extracts camera parameters, images, the /tf kinematic chain, and joint states directly from a ROS 2 bag directory without running ROS 2:
- Filters out frames where the manipulator is moving based on joint velocity.
- Skips duplicate static poses with less than 1.5 cm translation and 3 deg rotation difference.
- Computes the 4x4 transformation matrix, translation vector, quaternion, and Euler angles, saving the output to hand_eye_calibration_result.json.

Run:

    python3 bag_calib.py

## 2. Live ROS 2 Calibration (live_calib.py)

Subscribes to live ROS 2 camera and /tf topics and opens an interactive OpenCV window with real-time ChArUco pose visualization.

Run:

    python3 live_calib.py

Controls:
- SPACE or S: Capture current robot and board pose
- C: Compute Hand-Eye calibration and save to hand_eye_calibration_result.json
- Q: Quit
