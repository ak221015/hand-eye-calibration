import bisect
import json
import os
import sys
import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
BAG_PATH = "./calib_run_1"
OUTPUT_JSON = os.path.join(BAG_PATH, "hand_eye_calibration_result.json")
CALIB_MODE = "eye_in_hand"  # "eye_in_hand" or "eye_to_hand"

IMAGE_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"
TF_TOPIC = "/tf"

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
STATIC_WINDOW_NS = (
    250_000_000  # +/- 250 ms window to verify the robot is completely stationary
)
MAX_STATIC_SHIFT_M = (
    0.0005  # Max allowed flange translation within static window (0.5 mm)
)
MAX_STATIC_ROT_DEG = 0.05  # Max allowed flange rotation within static window (0.05 deg)
MIN_CHARUCO_CORNERS = (
    12  # Minimum detected ChArUco corners required per frame (out of 24)
)
MAX_REPROJ_ERR_PX = 0.6  # Max allowed PnP reprojection error in pixels

MAX_OUTLIER_RATIO = 0.30  # Raise an error if more than 30% of static poses are outliers
MIN_INLIER_POSES = 8  # Minimum required valid inlier poses after outlier rejection
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


def get_closest_pose(tf_ts, tf_poses, target_ns, max_diff_ns=50_000_000):
    """Find the closest robot TF pose in time using binary search on sorted timestamps."""
    if not tf_ts:
        return None
    i = bisect.bisect_left(tf_ts, target_ns)
    candidates = []
    if i < len(tf_ts):
        candidates.append((abs(tf_ts[i] - target_ns), tf_poses[i]))
    if i > 0:
        candidates.append((abs(tf_ts[i - 1] - target_ns), tf_poses[i - 1]))
    diff, val = min(candidates, key=lambda x: x[0])
    return val if diff <= max_diff_ns else None


def is_robot_truly_static(tf_ts, tf_poses, target_ns, window_ns=STATIC_WINDOW_NS):
    """Verify that the robot did not move within [target_ns - window_ns, target_ns + window_ns]."""
    i_start = bisect.bisect_left(tf_ts, target_ns - window_ns)
    i_end = bisect.bisect_right(tf_ts, target_ns + window_ns)
    if i_end - i_start < 3:
        return False
    T_ref = tf_poses[i_start]
    for idx in range(i_start + 1, i_end):
        dist, ang = pose_diff(T_ref, tf_poses[idx])
        if dist > MAX_STATIC_SHIFT_M or ang > MAX_STATIC_ROT_DEG:
            return False
    return True


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
    # Use median center for robustness against individual outliers
    center = np.median(target_pos, axis=0)
    pos_err_mm = np.linalg.norm(target_pos - center, axis=1) * 1000.0
    rmse_mm = np.sqrt(np.mean(pos_err_mm**2))
    max_err_mm = np.max(pos_err_mm)
    return rmse_mm, max_err_mm, pos_err_mm


def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    K, D = None, None
    raw_tf_records = []

    with Reader(BAG_PATH) as reader:
        # 1. Read Camera Intrinsics
        for conn, _, raw in reader.messages(
            [c for c in reader.connections if c.topic == INFO_TOPIC]
        ):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            print(
                f"[1/3] Camera intrinsics loaded ({msg.width}x{msg.height}), D={np.round(D, 5)}"
            )
            break

        if os.path.isfile("camera_intrinsics.json"):
            with open("camera_intrinsics.json", "r", encoding="utf-8") as f:
                intr = json.load(f)
                K = np.array(intr["camera_matrix"], dtype=np.float64).reshape(3, 3)
                D = np.array(intr["distortion_coefficients"], dtype=np.float64)
                print(
                    f"[1/3] Overrode with custom camera_intrinsics.json: D={np.round(D, 5)}"
                )

        # 2. Read Robot TF (record pose only when LINK_CHAIN transforms are updated)
        links = {}
        for conn, bag_ts, raw in reader.messages(
            [c for c in reader.connections if c.topic == TF_TOPIC]
        ):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
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
                    links[key] = T
                    updated_chain = True
                    chain_stamp_ns = stamp_to_ns(tr.header.stamp) or bag_ts

            if updated_chain and all(k in links for k in LINK_CHAIN):
                T_total = np.eye(4)
                for k in LINK_CHAIN:
                    T_total = T_total @ links[k]
                raw_tf_records.append((chain_stamp_ns, T_total))

        # Sort TF records chronologically to ensure valid binary search in bisect
        raw_tf_records.sort(key=lambda x: x[0])
        tf_ts = [r[0] for r in raw_tf_records]
        tf_poses = [r[1] for r in raw_tf_records]
        print(f"[2/3] Loaded and sorted {len(tf_poses)} valid robot TF states")

        # 3. Process Images & Synchronize
        R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
        last_T = None
        skipped_moving = 0
        skipped_reproj = 0

        img_conns = [c for c in reader.connections if c.topic == IMAGE_TOPIC]
        for conn, bag_ts, raw in reader.messages(img_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            ts = stamp_to_ns(msg.header.stamp) or bag_ts

            # Skip frame if the robot is moving or vibrating within the static time window
            if not is_robot_truly_static(tf_ts, tf_poses, ts):
                skipped_moving += 1
                continue

            T_rob = get_closest_pose(tf_ts, tf_poses, ts)
            if T_rob is None:
                continue

            # Skip duplicate poses (require >= 1.5 cm translation or >= 3 deg rotation)
            if last_T is not None:
                dist, ang = pose_diff(last_T, T_rob)
                if dist < 0.015 and ang < 3.0:
                    continue

            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY,
            )

            corners, ids, _, _ = detector.detectBoard(gray)
            if (
                ids is None
                or len(ids) < MIN_CHARUCO_CORNERS
                or board.checkCharucoCornersCollinear(ids)
            ):
                continue

            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue

            # Refine board pose using Levenberg-Marquardt optimization
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)

            # Filter out frames with high PnP reprojection error
            proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            reproj_err = np.sqrt(
                np.mean(
                    np.sum(
                        (img_pts.reshape(-1, 2) - proj_pts.reshape(-1, 2)) ** 2, axis=1
                    )
                )
            )
            if reproj_err > MAX_REPROJ_ERR_PX:
                skipped_reproj += 1
                continue

            R_cam, _ = cv2.Rodrigues(rvec)
            R_ee, t_ee = T_rob[:3, :3], T_rob[:3, 3].reshape(3, 1)

            if CALIB_MODE == "eye_in_hand":
                R_g2b.append(R_ee)
                t_g2b.append(t_ee)
            else:
                R_g2b.append(R_ee.T)
                t_g2b.append(-R_ee.T @ t_ee)

            R_t2c.append(R_cam)
            t_t2c.append(tvec.reshape(3, 1))
            last_T = T_rob

    print(
        f"[3/3] Selected {len(R_t2c)} static poses (skipped moving: {skipped_moving}, high reproj: {skipped_reproj})"
    )
    if len(R_t2c) < MIN_INLIER_POSES:
        raise RuntimeError(
            f"[ERROR] Only {len(R_t2c)} valid static poses found. "
            f"At least {MIN_INLIER_POSES} poses are required for reliable calibration."
        )

    # 4. Outlier Rejection & Hand-Eye Calibration Solver Selection
    R_init, t_init = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK
    )
    _, _, err_init = evaluate_calibration(R_g2b, t_g2b, R_t2c, t_t2c, R_init, t_init)

    threshold = max(3.5, float(np.median(err_init) * 2.0))
    inliers = [i for i, e in enumerate(err_init) if e <= threshold]

    num_total = len(R_t2c)
    num_outliers = num_total - len(inliers)
    outlier_ratio = num_outliers / num_total

    # Abort calibration if there are too many outliers or too few remaining inliers
    if outlier_ratio > MAX_OUTLIER_RATIO or len(inliers) < MIN_INLIER_POSES:
        raise RuntimeError(
            f"[ERROR] Outlier rejection failed: {num_outliers} out of {num_total} poses "
            f"({outlier_ratio * 100:.1f}%) exceeded the error threshold of {threshold:.2f} mm "
            f"(max allowed outlier ratio: {MAX_OUTLIER_RATIO * 100:.0f}%, "
            f"min required inliers: {MIN_INLIER_POSES}, remaining inliers: {len(inliers)}). "
            "Calibration aborted. Please check if the calibration board moved during recording "
            "or if camera/TF timestamps are desynchronized."
        )

    if num_outliers > 0:
        print(
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
            print(f"  Method {name:10s} -> RMSE: {rmse_m:.3f} mm | Max: {max_m:.3f} mm")
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
    t_tcp = T_tcp_cam[:3, 3]
    quat_tcp = R.from_matrix(T_tcp_cam[:3, :3]).as_quat()
    euler_tcp = R.from_matrix(T_tcp_cam[:3, :3]).as_euler("xyz", degrees=True)

    print(f"\n================ BEST RESULT ({best_method}) ================")
    print(f"T_link6_cam translation (m): {np.round(T_link6_cam[:3, 3], 6)}")
    print(f"T_tcp_cam translation (m):   {np.round(t_tcp, 6)}")
    print(f"Quaternion [qx, qy, qz, qw]: {np.round(quat_tcp, 6)}")
    print(f"Euler XYZ (deg):             {np.round(euler_tcp, 4)}")
    print(f"Accuracy (RMSE):             {best_rmse:.2f} mm (Max: {best_max:.2f} mm)")
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
    print(f"Saved calibration result to: {OUTPUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
