from collections import deque
from functools import reduce
import json, os
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
CUSTOM_INTRINSICS_JSON = "camera_intrinsics.json"

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

board = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.020, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

STATIC_HISTORY_SEC = 0.40  # Time window (seconds) to verify the robot is stationary
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
MAX_OUTLIER_RATIO = 0.30  # Raise error if >30% of captured poses are outliers
MIN_INLIER_POSES = 8  # Minimum required valid inlier poses for calibration
# =================================================


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(
        getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    )


def make_tf(rot_mat, t_vec):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot_mat, np.asarray(t_vec).flatten()
    return T


def pose_diff(T1, T2):
    return np.linalg.norm(T1[:3, 3] - T2[:3, 3]), np.degrees(
        R.from_matrix(T1[:3, :3].T @ T2[:3, :3]).magnitude()
    )


def evaluate_calibration(Tg, Tt, R_est, t_est):
    pts = (Tg @ make_tf(R_est, t_est) @ Tt)[:, :3, 3]
    err_mm = np.linalg.norm(pts - np.median(pts, axis=0), axis=1) * 1000.0
    return float(np.sqrt(np.mean(err_mm**2))), float(np.max(err_mm)), err_mm


class HandEyeCalibNode(Node):
    def __init__(self):
        super().__init__("hand_eye_calib_node")
        self.K, self.D, self.links = None, None, {}
        self.tf_history = deque(maxlen=200)
        self.last_captured_T = None
        self.Tg_list, self.Tt_list = [], []

        if os.path.isfile(CUSTOM_INTRINSICS_JSON):
            with open(CUSTOM_INTRINSICS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.K = np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
                self.D = np.array(data["distortion_coefficients"], dtype=np.float64)
                self.get_logger().info(
                    f"Loaded custom intrinsics from {CUSTOM_INTRINSICS_JSON}: D={np.round(self.D, 5)}"
                )

        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)
        self.create_subscription(TFMessage, "/tf", self.tf_cb, 50)
        self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)
        self.get_logger().info(
            "Node started. Controls: [SPACE/S] = capture, [X] = undo last, [C] = calibrate, [Q] = quit"
        )

    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K, self.D = (
                np.array(msg.k, dtype=np.float64).reshape(3, 3),
                np.array(msg.d, dtype=np.float64),
            )
            self.get_logger().info(
                f"Camera intrinsics received ({msg.width}x{msg.height}), D={np.round(self.D, 5)}"
            )
            if np.allclose(self.D, 0.0):
                self.get_logger().warn(
                    "Distortion coefficients D are all zeros! Run camera_intrinsics_calib.py first."
                )

    def tf_cb(self, msg: TFMessage):
        chain_stamp_ns = None
        for tr in msg.transforms:
            key = (tr.header.frame_id.strip("/"), tr.child_frame_id.strip("/"))
            if key in LINK_CHAIN:
                q, t = tr.transform.rotation, tr.transform.translation
                self.links[key] = make_tf(
                    R.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [t.x, t.y, t.z]
                )
                chain_stamp_ns = stamp_to_ns(tr.header.stamp)
        if chain_stamp_ns is not None and len(self.links) == len(LINK_CHAIN):
            now_ns = chain_stamp_ns or self.get_clock().now().nanoseconds
            self.tf_history.append(
                (now_ns, reduce(np.matmul, (self.links[k] for k in LINK_CHAIN)))
            )

    def get_static_robot_pose(self):
        if len(self.tf_history) < 3:
            return None, False
        latest_ns, T_latest = self.tf_history[-1]
        win = np.array(
            [
                T
                for ts, T in self.tf_history
                if latest_ns - ts <= int(STATIC_HISTORY_SEC * 1e9)
            ]
        )
        if len(win) < 3:
            return T_latest, False
        shift = np.linalg.norm(win[:, :3, 3] - win[0, :3, 3], axis=1).max()
        rot = np.degrees(
            R.from_matrix(win[0, :3, :3].T @ win[:, :3, :3]).magnitude().max()
        )
        return T_latest, bool(shift <= MAX_STATIC_SHIFT_M and rot <= MAX_STATIC_ROT_DEG)

    def image_cb(self, msg: Image):
        if self.K is None:
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vis, gray = img.copy(), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _, _ = detector.detectBoard(gray)
        pose_ok, rvec, tvec, reproj_err = False, None, None, float("inf")
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
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_pts, img_pts, self.K, self.D, rvec, tvec
                )
                reproj_err = float(
                    cv2.norm(
                        img_pts,
                        cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)[0],
                        cv2.NORM_L2,
                    )
                    / np.sqrt(n_corners)
                )
                if reproj_err <= MAX_REPROJ_ERR_PX:
                    pose_ok = True
                    cv2.drawFrameAxes(vis, self.K, self.D, rvec, tvec, 0.06)

        T_rob, is_static = self.get_static_robot_pose()
        tf_ready = T_rob is not None and is_static
        tf_status = (
            "STATIC" if tf_ready else ("MOVING" if T_rob is not None else "WAIT")
        )
        err_str = f"{reproj_err:.2f}px" if reproj_err < 100 else "N/A"

        cv2.putText(
            vis,
            f"Saved: {len(self.Tg_list)} | Corners: {n_corners}/24 | Err: {err_str} | TF: {tf_status}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0) if (pose_ok and tf_ready) else (0, 165, 255),
            2,
        )
        cv2.imshow("Live Hand-Eye Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord(" "), ord("s")):
            if not pose_ok:
                self.get_logger().warn(
                    f"Cannot capture: need >= {MIN_CHARUCO_CORNERS} corners and err <= {MAX_REPROJ_ERR_PX} px (current: {n_corners}, {err_str})"
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
                        f"Pose too close (shift: {dist * 100:.2f} cm, angle: {ang:.2f} deg). Move >= {MIN_POSE_SHIFT_M * 100:.1f} cm or {MIN_POSE_ROT_DEG:.1f} deg!"
                    )
                    return

            self.Tg_list.append(
                T_rob.copy() if CALIB_MODE == "eye_in_hand" else np.linalg.inv(T_rob)
            )
            self.Tt_list.append(make_tf(cv2.Rodrigues(rvec)[0], tvec))
            self.last_captured_T = T_rob.copy()
            self.get_logger().info(
                f"Captured pose #{len(self.Tg_list)} ({n_corners} corners, reproj: {reproj_err:.3f} px)"
            )

        elif key == ord("x") and self.Tg_list:
            self.Tg_list.pop(), self.Tt_list.pop()
            self.last_captured_T = None
            self.get_logger().info(
                f"Removed last pose. Remaining poses: {len(self.Tg_list)}"
            )

        elif key == ord("c"):
            if len(self.Tg_list) < MIN_INLIER_POSES:
                self.get_logger().warn(
                    f"Collect at least {MIN_INLIER_POSES} poses (15-20 recommended) before calibrating!"
                )
                return
            try:
                self.compute_calibration()
            except RuntimeError as err:
                self.get_logger().error(str(err))

        elif key == ord("q"):
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def compute_calibration(self):
        Tg, Tt = np.array(self.Tg_list), np.array(self.Tt_list)
        R_init, t_init = cv2.calibrateHandEye(
            Tg[:, :3, :3],
            Tg[:, :3, 3],
            Tt[:, :3, :3],
            Tt[:, :3, 3],
            method=cv2.CALIB_HAND_EYE_PARK,
        )
        _, _, err_init = evaluate_calibration(Tg, Tt, R_init, t_init)

        threshold = max(3.5, float(np.median(err_init) * 2.0))
        mask = err_init <= threshold
        num_total, num_outliers = len(Tg), int(np.sum(~mask))
        outlier_ratio = num_outliers / num_total

        if outlier_ratio > MAX_OUTLIER_RATIO or np.sum(mask) < MIN_INLIER_POSES:
            raise RuntimeError(
                f"[ERROR] Outlier rejection failed: {num_outliers}/{num_total} poses ({outlier_ratio * 100:.1f}%) "
                f"exceeded {threshold:.2f} mm (max ratio: {MAX_OUTLIER_RATIO * 100:.0f}%, min inliers: {MIN_INLIER_POSES})."
            )
        if num_outliers > 0:
            self.get_logger().info(
                f"Filtered out {num_outliers}/{num_total} outlier poses ({outlier_ratio * 100:.1f}%, threshold: {threshold:.2f} mm)"
            )
            Tg, Tt = Tg[mask], Tt[mask]

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
                R_m, t_m = cv2.calibrateHandEye(
                    Tg[:, :3, :3],
                    Tg[:, :3, 3],
                    Tt[:, :3, :3],
                    Tt[:, :3, 3],
                    method=flag,
                )
                rmse_m, max_m, _ = evaluate_calibration(Tg, Tt, R_m, t_m)
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

        T_link6_cam = make_tf(best_R, best_t)
        T_tool0_tcp = make_tf(
            R.from_euler("z", 90, degrees=True).as_matrix(), [0.0, 0.0, 0.167]
        )
        T_tcp_cam = np.linalg.inv(T_tool0_tcp) @ T_link6_cam
        t_tcp, rot_tcp = T_tcp_cam[:3, 3], R.from_matrix(T_tcp_cam[:3, :3])
        quat_tcp, euler_tcp = rot_tcp.as_quat(), rot_tcp.as_euler("xyz", degrees=True)

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
                    "used_unique_poses": len(Tg),
                    "rejected_outliers": num_outliers,
                    "rmse_error_mm": round(best_rmse, 3),
                    "max_error_mm": round(best_max, 3),
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
