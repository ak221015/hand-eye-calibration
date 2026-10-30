import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
CALIB_MODE = "eye_in_hand"  # "eye_in_hand" or "eye_to_hand"
OUTPUT_JSON = "hand_eye_calibration_result.json"

IMAGE_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"

LINK_CHAIN = [
    ("arm_base_link", "arm_1_link"),
    ("arm_1_link", "arm_2_link"),
    ("arm_2_link", "arm_3_link"),
    ("arm_3_link", "arm_4_link"),
    ("arm_4_link", "arm_5_link"),
    ("arm_5_link", "arm_6_link"),
]

# ChArUco board parameters (7x5, square 30mm, marker 20mm)
board = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.020, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)
detector = cv2.aruco.CharucoDetector(board)
# =================================================


class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__("hand_eye_calib_node")
        self.K = None
        self.D = None
        self.links = {}

        # Lists for cv2.calibrateHandEye
        self.R_gripper2base, self.t_gripper2base = [], []
        self.R_target2cam, self.t_target2cam = [], []

        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)
        self.create_subscription(TFMessage, "/tf", self.tf_cb, 50)
        self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)
        self.get_logger().info("Node started. Controls: [SPACE] = capture pose, [C] = calibrate, [Q] = quit")

    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(f"Camera intrinsics received ({msg.width}x{msg.height})")

    def tf_cb(self, msg: TFMessage):
        for tr in msg.transforms:
            key = (tr.header.frame_id.strip("/"), tr.child_frame_id.strip("/"))
            if key in LINK_CHAIN:
                T = np.eye(4)
                q = [tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w]
                T[:3, :3] = R.from_quat(q).as_matrix()
                T[:3, 3] = [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z]
                self.links[key] = T

    def get_robot_pose(self):
        """Multiplies T_base_1 * T_1_2 * ... * T_5_6 to get T_base_ee."""
        if not all(k in self.links for k in LINK_CHAIN):
            return None
        T_base_ee = np.eye(4)
        for k in LINK_CHAIN:
            T_base_ee = T_base_ee @ self.links[k]
        return T_base_ee

    def image_cb(self, msg: Image):
        if self.K is None:
            return

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)

        pose_ok = False
        rvec, tvec = None, None
        n_corners = len(ids) if ids is not None else 0

        if ids is not None and n_corners >= 6 and not board.checkCharucoCornersCollinear(ids):
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (255, 0, 0))
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            pose_ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D)
            if pose_ok:
                cv2.drawFrameAxes(vis, self.K, self.D, rvec, tvec, 0.06)

        T_rob = self.get_robot_pose()
        tf_ready = T_rob is not None

        cv2.putText(
            vis,
            f"Saved: {len(self.R_target2cam)} | Corners: {n_corners}/24 | TF: {'OK' if tf_ready else 'WAIT'}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if (pose_ok and tf_ready) else (0, 165, 255), 2
        )
        cv2.imshow("Live Hand-Eye Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        # [SPACE] or [S] -> Capture current sample
        if key in (ord(" "), ord("s")):
            if not pose_ok or not tf_ready:
                self.get_logger().warn("Cannot capture: board pose or robot /tf not ready!")
                return

            R_cam, _ = cv2.Rodrigues(rvec)
            R_ee, t_ee = T_rob[:3, :3], T_rob[:3, 3].reshape(3, 1)

            if CALIB_MODE == "eye_in_hand":
                self.R_gripper2base.append(R_ee)
                self.t_gripper2base.append(t_ee)
            else:
                self.R_gripper2base.append(R_ee.T)
                self.t_gripper2base.append(-R_ee.T @ t_ee)

            self.R_target2cam.append(R_cam)
            self.t_target2cam.append(tvec.reshape(3, 1))
            self.get_logger().info(f"Captured pose #{len(self.R_target2cam)} ({n_corners} corners)")

        # [C] -> Compute calibration
        elif key == ord("c"):
            if len(self.R_target2cam) < 5:
                self.get_logger().warn("Collect at least 5 poses (12-15 recommended) before calibrating!")
                return
            self.compute_calibration()

        # [Q] -> Quit
        elif key == ord("q"):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def compute_calibration(self):
        R_est, t_est = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK
        )
        T = np.eye(4)
        T[:3, :3] = R_est
        T[:3, 3] = t_est.flatten()

        quat = R.from_matrix(R_est).as_quat()
        euler = R.from_matrix(R_est).as_euler("xyz", degrees=True)

        print("\n================ CALIBRATION RESULT ================")
        print(f"Transformation Matrix 4x4 (arm_6_link -> camera_color_optical_frame):\n{np.round(T, 6)}")
        print(f"Translation [x, y, z] (m):   {np.round(t_est.flatten(), 6)}")
        print(f"Quaternion [qx, qy, qz, qw]: {np.round(quat, 6)}")
        print(f"Euler XYZ (deg):             {np.round(euler, 4)}")
        print("====================================================\n")

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "calibration_mode": CALIB_MODE,
                "used_poses": len(self.R_target2cam),
                "transformation_matrix_4x4": T.tolist(),
                "translation_xyz_m": t_est.flatten().tolist(),
                "quaternion_xyzw": quat.tolist(),
                "euler_xyz_deg": euler.tolist(),
            }, f, indent=4)
        self.get_logger().info(f"Saved result to {OUTPUT_JSON}")


def main():
    rclpy.init()
    node = HandEyeCalibNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()