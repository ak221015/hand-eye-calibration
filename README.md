# ROS 2 Hand-Eye Calibration (ChArUco)

Extrinsic calibration scripts (eye_in_hand / eye_to_hand) for an Intel RealSense camera mounted on a 6-DOF robotic arm (arm_base_link -> arm_6_link).

Uses a 7x5 ChArUco board (30 mm square size, 20 mm marker size, DICT_6X6_250) and cv2.calibrateHandEye (CALIB_HAND_EYE_PARK).

## Dependencies

    pip install opencv-contrib-python numpy scipy rosbags

## 1. Offline Calibration from ROS 2 Bag (bag_calib.py)

Extracts camera intrinsics (/camera/camera/color/camera_info), images (/camera/camera/color/image_raw), /tf chain, and /joint_states directly from a ROS 2 bag directory without running ROS 2:
- Filters out frames where the manipulator is moving (joint velocity > 0.02).
- Skips duplicate static poses (< 1.5 cm translation and < 3 deg rotation difference).
- Computes the 4x4 transformation matrix, translation vector, quaternion, and Euler angles, saving the output to hand_eye_calibration_result.json.

Run:

    python3 bag_calib.py

## 2. Live ROS 2 Calibration (live_calib.py)

Subscribes to live ROS 2 topics (/camera/camera/color/image_raw, /camera/camera/color/camera_info, /tf) and opens an interactive OpenCV window with real-time ChArUco pose visualization.

Run:

    python3 live_calib.py

Controls in OpenCV Window:
- SPACE or S: Capture current robot and board pose
- C: Compute Hand-Eye calibration and save to hand_eye_calibration_result.json
- Q: Quit
