from functools import reduce
import json, os, sys
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

board = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.020, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

STATIC_WINDOW_NS = 250_000_000  # +/- 250 ms window to verify the robot is stationary
MAX_SYNC_DIFF_NS = (
    50_000_000  # Max allowed timestamp difference (50 ms) between image and TF
)
MAX_STATIC_SHIFT_M = (
    0.0005  # Max allowed flange translation within static window (0.5 mm)
)
MAX_STATIC_ROT_DEG = 0.05  # Max allowed flange rotation within static window (0.05 deg)
MIN_CHARUCO_CORNERS = (
    12  # Minimum detected ChArUco corners required per frame (out of 24)
)
MAX_REPROJ_ERR_PX = 0.6  # Max allowed PnP reprojection error in pixels
MAX_OUTLIER_RATIO = 0.30  # Raise error if >30% of static poses are outliers
MIN_INLIER_POSES = 8  # Minimum required valid inlier poses after outlier rejection
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


def is_window_static(win):
    if len(win) < 3:
        return False
    shift = np.linalg.norm(win[:, :3, 3] - win[0, :3, 3], axis=1).max()
    rot = np.degrees(R.from_matrix(win[0, :3, :3].T @ win[:, :3, :3]).magnitude().max())
    return shift <= MAX_STATIC_SHIFT_M and rot <= MAX_STATIC_ROT_DEG


def evaluate_calibration(Tg, Tt, R_est, t_est):
    """Vectorized target position consistency check in the robot base frame."""
    pts = (Tg @ make_tf(R_est, t_est) @ Tt)[:, :3, 3]
    err_mm = np.linalg.norm(pts - np.median(pts, axis=0), axis=1) * 1000.0
    return float(np.sqrt(np.mean(err_mm**2))), float(np.max(err_mm)), err_mm


def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    K, D, raw_tf_records, links = None, None, [], {}
    Tg_list, Tt_list = [], []

    with Reader(BAG_PATH) as reader:
        # 1. Read Camera Intrinsics & override if custom JSON exists
        for conn, _, raw in reader.messages(
            [c for c in reader.connections if c.topic == INFO_TOPIC]
        ):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            K, D = (
                np.array(msg.k, dtype=np.float64).reshape(3, 3),
                np.array(msg.d, dtype=np.float64),
            )
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

        # 2. Read & Sort Robot TF (record only when LINK_CHAIN updates)
        for conn, bag_ts, raw in reader.messages(
            [c for c in reader.connections if c.topic == TF_TOPIC]
        ):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            chain_stamp_ns = None
            for tr in msg.transforms:
                key = (tr.header.frame_id.strip("/"), tr.child_frame_id.strip("/"))
                if key in LINK_CHAIN:
                    q, t = tr.transform.rotation, tr.transform.translation
                    links[key] = make_tf(
                        R.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [t.x, t.y, t.z]
                    )
                    chain_stamp_ns = stamp_to_ns(tr.header.stamp) or bag_ts
            if chain_stamp_ns and len(links) == len(LINK_CHAIN):
                raw_tf_records.append(
                    (chain_stamp_ns, reduce(np.matmul, (links[k] for k in LINK_CHAIN)))
                )

        raw_tf_records.sort(key=lambda x: x[0])
        tf_ts = np.array([r[0] for r in raw_tf_records], dtype=np.int64)
        tf_poses = np.array([r[1] for r in raw_tf_records], dtype=np.float64)
        print(f"[2/3] Loaded and sorted {len(tf_poses)} valid robot TF states")

        # 3. Process Images & Synchronize
        last_T, skipped_moving, skipped_reproj = None, 0, 0
        for conn, bag_ts, raw in reader.messages(
            [c for c in reader.connections if c.topic == IMAGE_TOPIC]
        ):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            ts = stamp_to_ns(msg.header.stamp) or bag_ts

            # Fast O(log N) static window check
            i0, i1 = np.searchsorted(
                tf_ts, [ts - STATIC_WINDOW_NS, ts + STATIC_WINDOW_NS]
            )
            if not is_window_static(tf_poses[i0:i1]):
                skipped_moving += 1
                continue

            # Find closest TF pose within MAX_SYNC_DIFF_NS (50 ms)
            idx = int(np.searchsorted(tf_ts, ts))
            cands = [
                k
                for k in (idx - 1, idx)
                if 0 <= k < len(tf_ts) and abs(tf_ts[k] - ts) <= MAX_SYNC_DIFF_NS
            ]
            if not cands:
                continue
            T_rob = tf_poses[min(cands, key=lambda k: abs(tf_ts[k] - ts))]

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

            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)
            reproj_err = cv2.norm(
                img_pts, cv2.projectPoints(obj_pts, rvec, tvec, K, D)[0], cv2.NORM_L2
            ) / np.sqrt(len(img_pts))
            if reproj_err > MAX_REPROJ_ERR_PX:
                skipped_reproj += 1
                continue

            Tg_list.append(
                T_rob if CALIB_MODE == "eye_in_hand" else np.linalg.inv(T_rob)
            )
            Tt_list.append(make_tf(cv2.Rodrigues(rvec)[0], tvec))
            last_T = T_rob

    print(
        f"[3/3] Selected {len(Tg_list)} static poses (skipped moving: {skipped_moving}, high reproj: {skipped_reproj})"
    )
    if len(Tg_list) < MIN_INLIER_POSES:
        raise RuntimeError(
            f"[ERROR] Only {len(Tg_list)} valid static poses found. Need >= {MIN_INLIER_POSES}."
        )

    # 4. Outlier Rejection & Hand-Eye Calibration Solver Selection
    Tg, Tt = np.array(Tg_list), np.array(Tt_list)
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
        print(
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
                Tg[:, :3, :3], Tg[:, :3, 3], Tt[:, :3, :3], Tt[:, :3, 3], method=flag
            )
            rmse_m, max_m, _ = evaluate_calibration(Tg, Tt, R_m, t_m)
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

    T_link6_cam = make_tf(best_R, best_t)
    T_tool0_tcp = make_tf(
        R.from_euler("z", 90, degrees=True).as_matrix(), [0.0, 0.0, 0.167]
    )
    T_tcp_cam = np.linalg.inv(T_tool0_tcp) @ T_link6_cam
    t_tcp, rot_tcp = T_tcp_cam[:3, 3], R.from_matrix(T_tcp_cam[:3, :3])
    quat_tcp, euler_tcp = rot_tcp.as_quat(), rot_tcp.as_euler("xyz", degrees=True)

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
    print(f"Saved calibration result to: {OUTPUT_JSON}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
