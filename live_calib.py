from collections import deque
import json
import os
import sys
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
CUSTOM_INTRINSICS_JSON = "camera_intrinsics.json"  # Loaded automatically if present

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

# ChArUco board: 7x5 squares, 30mm square, 20mm marker, with subpixel corner refinement
board = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.020, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

# Filtering & Outlier Rejection thresholds
STATIC_HISTORY_SEC = (
    0.40  # Time window (seconds) to verify the robot is completely stationary
)
MAX_STATIC_SHIFT_M = (
    0.0005  # Max allowed flange translation within static window (0.5 mm)
)
MAX_STATIC_ROT_DEG = 0.05  # Max allowed flange rotation within static window (0.05 deg)
MIN_POSE_SHIFT_M = 0.015  # Minimum translation (1.5 cm) required between captured poses
MIN_POSE_ROT_DEG = 3.0  # Minimum rotation (3.0 deg) required between captured poses
MIN_CHARUCO_CORNERS = (
    12  # Minimum detected ChArUco corners required per frame (out of 24)
)
MAX_REPROJ_ERR_PX = 0.6  # Max allowed PnP reprojection error in pixels

MAX_OUTLIER_RATIO = (
    0.30  # Raise an error if more than 30% of captured poses are outliers
)
MIN_INLIER_POSES = 8  # Minimum required valid inlier poses for calibration
# =================================================


def stamp_to_ns(stamp):
    """Convert ROS timestamp message to nanoseconds."""
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    return int(stamp.sec) * 1_000_000_000 + int(nsec)


def pose_diff(T1, T2):
    """Compute Euclidean distance (m) and angular difference (deg) between two 4x4 poses."""
    dist = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    ang = np.degrees(
        np.linalg.norm(R.from_matrix(T1[:3, :3].T @ T2[:3, :3]).as_rotvec())
    )
    return dist, ang


def evaluate_calibration(R_g2b, t_g2b, R_t2c, t_t2c, R_est, t_est):
    """Evaluate target position consistency in the robot base frame."""
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_est
    T_cam[:3, 3] = t_est.flatten()

    target_pos = []
    for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        Tg = np.eye(4)
        Tg[:3, :3], Tg[:3, 3] = Rg, tg.flatten()
        Tt = np.eye(4)
        Tt[:3, :3], Tt[:3, 3] = Rt, tt.flatten()
        # Base -> Gripper -> Camera -> Board
        target_pos.append((Tg @ T_cam @ Tt)[:3, 3])

    target_pos = np.array(target_pos)
    center = np.median(target_pos, axis=0)
    pos_err_mm = np.linalg.norm(target_pos - center, axis=1) * 1000.0
    rmse_mm = np.sqrt(np.mean(pos_err_mm**2))
    max_err_mm = np.max(pos_err_mm)
    return rmse_mm, max_err_mm, pos_err_mm


class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__("hand_eye_calib_node")
        self.K = None
        self.D = None
        self.links = {}
        self.tf_history = deque(maxlen=200)  # Stores (timestamp_ns, T_base_ee)
        self.last_captured_T = None

        # Lists for cv2.calibrateHandEye
        self.R_gripper2base, self.t_gripper2base = [], []
        self.R_target2cam, self.t_target2cam = [], []

        # Load custom calibrated intrinsics if available
        if os.path.isfile(CUSTOM_INTRINSICS_JSON):
            with open(CUSTOM_INTRINSICS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.K = np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
                self.D = np.array(data["distortion_coefficients"], dtype=np.float64)
                self.get_logger().info(
                    f"Loaded custom camera intrinsics from {CUSTOM_INTRINSICS_JSON}: "
                    f"D={np.round(self.D, 5)}"
                )

        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)
        self.create_subscription(TFMessage, "/tf", self.tf_cb, 50)
        self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)
        self.get_logger().info(
            "Node started. Controls: [SPACE/S] = capture, [X] = undo last, [C] = calibrate, [Q] = quit"
        )

    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(
                f"Camera intrinsics received from topic ({msg.width}x{msg.height}), D={np.round(self.D, 5)}"
            )
            if np.allclose(self.D, 0.0):
                self.get_logger().warn(
                    "Camera distortion coefficients D are all zeros! "
                    "Run camera_intrinsics_calib.py first for higher accuracy."
                )

    def tf_cb(self, msg: TFMessage):
        updated_chain = False
        chain_stamp_ns = None
        for tr in msg.transforms:
            key = (tr.header.frame_id.strip("/"), tr.child_frame_id.strip("/"))
            if key in LINK_CHAIN:
                T = np.eye(4)
                q = [
                    tr.transform.rotation.x,
                    tr.transform.rotation.y,
                    tr.transform.rotation.z,
                    tr.transform.rotation.w,
                ]
                T[:3, :3] = R.from_quat(q).as_matrix()
                T[:3, 3] = [
                    tr.transform.translation.x,
                    tr.transform.translation.y,
                    tr.transform.translation.z,
                ]
                self.links[key] = T
                updated_chain = True
                chain_stamp_ns = stamp_to_ns(tr.header.stamp)

        # Append pose to history only when a link in LINK_CHAIN was actually updated
        if updated_chain and all(k in self.links for k in LINK_CHAIN):
            T_base_ee = np.eye(4)
            for k in LINK_CHAIN:
                T_base_ee = T_base_ee @ self.links[k]
            now_ns = chain_stamp_ns or self.get_clock().now().nanoseconds
            self.tf_history.append((now_ns, T_base_ee))

    def get_static_robot_pose(self):
        """Return (T_rob, is_static) by checking pose stability over STATIC_HISTORY_SEC."""
        if len(self.tf_history) < 3:
            return None, False

        latest_ns, T_latest = self.tf_history[-1]
        window_ns = int(STATIC_HISTORY_SEC * 1_000_000_000)
        recent_poses = [T for ts, T in self.tf_history if latest_ns - ts <= window_ns]

        if len(recent_poses) < 3:
            return T_latest, False

        T_ref = recent_poses[0]
        for T_cur in recent_poses[1:]:
            dist, ang = pose_diff(T_ref, T_cur)
            if dist > MAX_STATIC_SHIFT_M or ang > MAX_STATIC_ROT_DEG:
                return T_latest, False

        return T_latest, True

    def image_cb(self, msg: Image):
        if self.K is None:
            return

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)

        pose_ok = False
        rvec, tvec = None, None
        reproj_err = float("inf")
        n_corners = len(ids) if ids is not None else 0

        if (
            ids is not None
            and n_corners >= MIN_CHARUCO_CORNERS
            and not board.checkCharucoCornersCollinear(ids)
        ):
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (255, 0, 0))
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                # Non-linear Levenberg-Marquardt refinement for board pose
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_pts, img_pts, self.K, self.D, rvec, tvec
                )
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
                reproj_err = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (img_pts.reshape(-1, 2) - proj_pts.reshape(-1, 2)) ** 2,
                                axis=1,
                            )
                        )
                    )
                )
                if reproj_err <= MAX_REPROJ_ERR_PX:
                    pose_ok = True
                    cv2.drawFrameAxes(vis, self.K, self.D, rvec, tvec, 0.06)

        T_rob, is_static = self.get_static_robot_pose()
        tf_ready = T_rob is not None and is_static

        status_color = (0, 255, 0) if (pose_ok and tf_ready) else (0, 165, 255)
        tf_status_str = (
            "STATIC" if tf_ready else ("MOVING" if T_rob is not None else "WAIT")
        )
        err_str = f"{reproj_err:.2f}px" if reproj_err < 100 else "N/A"

        cv2.putText(
            vis,
            f"Saved: {len(self.R_target2cam)} | Corners: {n_corners}/24 | Err: {err_str} | TF: {tf_status_str}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2,
        )
        cv2.imshow("Live Hand-Eye Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        # [SPACE] or [S] -> Capture current sample
        if key in (ord(" "), ord("s")):
            if not pose_ok:
                self.get_logger().warn(
                    f"Cannot capture: need >= {MIN_CHARUCO_CORNERS} corners and reproj error <= {MAX_REPROJ_ERR_PX} px "
                    f"(current: {n_corners} corners, err: {err_str})"
                )
                return
            if not tf_ready:
                self.get_logger().warn(
                    "Cannot capture: robot /tf is not ready or still moving/vibrating!"
                )
                return

            if self.last_captured_T is not None:
                dist, ang = pose_diff(self.last_captured_T, T_rob)
                if dist < MIN_POSE_SHIFT_M and ang < MIN_POSE_ROT_DEG:
                    self.get_logger().warn(
                        f"Pose too close to previous capture (shift: {dist * 100:.2f} cm, angle: {ang:.2f} deg). "
                        f"Move at least {MIN_POSE_SHIFT_M * 100:.1f} cm or {MIN_POSE_ROT_DEG:.1f} deg!"
                    )
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
            self.last_captured_T = T_rob.copy()
            self.get_logger().info(
                f"Captured pose #{len(self.R_target2cam)} ({n_corners} corners, reproj: {reproj_err:.3f} px)"
            )

        # [X] -> Undo last captured sample
        elif key == ord("x"):
            if self.R_target2cam:
                self.R_gripper2base.pop()
                self.t_gripper2base.pop()
                self.R_target2cam.pop()
                self.t_target2cam.pop()
                self.last_captured_T = None
                self.get_logger().info(
                    f"Removed last pose. Remaining poses: {len(self.R_target2cam)}"
                )

        # [C] -> Compute calibration
        elif key == ord("c"):
            if len(self.R_target2cam) < MIN_INLIER_POSES:
                self.get_logger().warn(
                    f"Collect at least {MIN_INLIER_POSES} poses (15-20 recommended) before calibrating!"
                )
                return
            try:
                self.compute_calibration()
            except RuntimeError as err:
                self.get_logger().error(str(err))

        # [Q] -> Quit
        elif key == ord("q"):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def compute_calibration(self):
        R_g2b = list(self.R_gripper2base)
        t_g2b = list(self.t_gripper2base)
        R_t2c = list(self.R_target2cam)
        t_t2c = list(self.t_target2cam)

        # Initial pass for Outlier Rejection
        R_init, t_init = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK
        )
        _, _, err_init = evaluate_calibration(
            R_g2b, t_g2b, R_t2c, t_t2c, R_init, t_init
        )

        threshold = max(3.5, float(np.median(err_init) * 2.0))
        inliers = [i for i, e in enumerate(err_init) if e <= threshold]

        num_total = len(R_t2c)
        num_outliers = num_total - len(inliers)
        outlier_ratio = num_outliers / num_total

        # Raise an error if outlier ratio is too high or too few valid inliers remain
        if outlier_ratio > MAX_OUTLIER_RATIO or len(inliers) < MIN_INLIER_POSES:
            raise RuntimeError(
                f"[ERROR] Outlier rejection failed: {num_outliers} out of {num_total} poses "
                f"({outlier_ratio * 100:.1f}%) exceeded the error threshold of {threshold:.2f} mm "
                f"(max allowed outlier ratio: {MAX_OUTLIER_RATIO * 100:.0f}%, "
                f"min required inliers: {MIN_INLIER_POSES}, remaining inliers: {len(inliers)}). "
                "Calibration aborted! Check if the board moved or capture more accurate poses."
            )

        if num_outliers > 0:
            self.get_logger().info(
                f"Filtered out {num_outliers}/{num_total} outlier poses "
                f"({outlier_ratio * 100:.1f}%, threshold: {threshold:.2f} mm)"
            )
            R_g2b = [R_g2b[i] for i in inliers]
            t_g2b = [t_g2b[i] for i in inliers]
            R_t2c = [R_t2c[i] for i in inliers]
            t_t2c = [t_t2c[i] for i in inliers]

        methods = {
            "TSAI": cv2.CALIB_HAND_EYE_TSAI,
            "PARK": cv2.CALIB_HAND_EYE_PARK,
            "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
            "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        best_method, best_rmse, best_max, best_R, best_t = (
            None,
            float("inf"),
            float("inf"),
            None,
            None,
        )
        for name, flag in methods.items():
            try:
                R_m, t_m = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
                rmse_m, max_m, _ = evaluate_calibration(
                    R_g2b, t_g2b, R_t2c, t_t2c, R_m, t_m
                )
                print(
                    f"  Method {name:10s} -> RMSE: {rmse_m:.3f} mm | Max: {max_m:.3f} mm"
                )
                if rmse_m < best_rmse:
                    best_method, best_rmse, best_max, best_R, best_t = (
                        name,
                        rmse_m,
                        max_m,
                        R_m,
                        t_m,
                    )
            except cv2.error:
                continue

        if best_R is None:
            raise RuntimeError(
                "[ERROR] All Hand-Eye calibration solvers failed to converge."
            )

        T_link6_cam = np.eye(4)
        T_link6_cam[:3, :3] = best_R
        T_link6_cam[:3, 3] = best_t.flatten()

        # Transform from arm_tool0 (arm_6_link) to gripper_tcp (+0.167m along Z, +90 deg around Z)
        T_tool0_tcp = np.eye(4)
        T_tool0_tcp[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
        T_tool0_tcp[:3, 3] = [0.0, 0.0, 0.167]

        # Target calibration matrix: gripper_tcp -> camera_color_optical_frame
        T_tcp_cam = np.linalg.inv(T_tool0_tcp) @ T_link6_cam

        R_tcp = T_tcp_cam[:3, :3]
        t_tcp = T_tcp_cam[:3, 3]
        quat_tcp = R.from_matrix(R_tcp).as_quat()
        euler_tcp = R.from_matrix(R_tcp).as_euler("xyz", degrees=True)

        print(f"\n================ BEST RESULT ({best_method}) ================")
        print(np.array2string(T_tcp_cam, precision=6, suppress_small=True))
        print(f"T_link6_cam translation (m): {np.round(T_link6_cam[:3, 3], 6)}")
        print(f"T_tcp_cam translation (m):   {np.round(t_tcp, 6)}")
        print(f"Quaternion [qx, qy, qz, qw]: {np.round(quat_tcp, 6)}")
        print(f"Euler XYZ (deg):             {np.round(euler_tcp, 4)}")
        print(
            f"Accuracy (RMSE):             {best_rmse:.2f} mm (Max error: {best_max:.2f} mm)"
        )
        print("=============================================================\n")

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "calibration_mode": CALIB_MODE,
                    "solver_method": best_method,
                    "parent_frame": "gripper_tcp",
                    "child_frame": "camera_color_optical_frame",
                    "used_unique_poses": len(R_t2c),
                    "rejected_outliers": num_outliers,
                    "rmse_error_mm": round(float(best_rmse), 3),
                    "max_error_mm": round(float(best_max), 3),
                    "transformation_matrix_4x4": T_tcp_cam.tolist(),
                    "translation_xyz_m": t_tcp.tolist(),
                    "quaternion_xyzw": quat_tcp.tolist(),
                    "euler_xyz_deg": euler_tcp.tolist(),
                },
                f,
                indent=4,
            )
        self.get_logger().info(f"Saved calibration result to {OUTPUT_JSON}")


def main():
    rclpy.init()
    node = HandEyeCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
