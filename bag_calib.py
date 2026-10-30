import bisect
import json
import os
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
JOINT_TOPIC = "/joint_states"

LINK_CHAIN = [
    ("arm_base_link", "arm_1_link"),
    ("arm_1_link", "arm_2_link"),
    ("arm_2_link", "arm_3_link"),
    ("arm_3_link", "arm_4_link"),
    ("arm_4_link", "arm_5_link"),
    ("arm_5_link", "arm_6_link"),
]

# ChArUco board: 7x5 squares, 30mm square, 20mm marker
board = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.020, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)
detector = cv2.aruco.CharucoDetector(board)
# =================================================


def stamp_to_ns(stamp):
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    return int(stamp.sec) * 1_000_000_000 + int(nsec)


def get_closest(ts_list, val_list, target_ns, max_diff_ns=50_000_000):
    if not ts_list:
        return None
    i = bisect.bisect_left(ts_list, target_ns)
    candidates = []
    if i < len(ts_list):
        candidates.append((abs(ts_list[i] - target_ns), val_list[i]))
    if i > 0:
        candidates.append((abs(ts_list[i - 1] - target_ns), val_list[i - 1]))
    diff, val = min(candidates, key=lambda x: x[0])
    return val if diff <= max_diff_ns else None


def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    K, D = None, None
    tf_ts, tf_poses = [], []
    js_ts, js_vels = [], []

    with Reader(BAG_PATH) as reader:
        # 1. Read Camera Intrinsics
        for conn, _, raw in reader.messages([c for c in reader.connections if c.topic == INFO_TOPIC]):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            print(f"[1/3] Camera intrinsics loaded ({msg.width}x{msg.height})")
            break

        # 2. Read Robot TF & Joint Velocities
        links = {}
        for conn, bag_ts, raw in reader.messages([c for c in reader.connections if c.topic == TF_TOPIC]):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            for tr in msg.transforms:
                key = (tr.header.frame_id.strip("/"), tr.child_frame_id.strip("/"))
                if key in LINK_CHAIN:
                    T = np.eye(4)
                    q = [tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w]
                    T[:3, :3] = R.from_quat(q).as_matrix()
                    T[:3, 3] = [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z]
                    links[key] = T
            if all(k in links for k in LINK_CHAIN):
                T_total = np.eye(4)
                for k in LINK_CHAIN:
                    T_total = T_total @ links[k]
                tf_ts.append(stamp_to_ns(msg.transforms[0].header.stamp) or bag_ts)
                tf_poses.append(T_total)

        for conn, bag_ts, raw in reader.messages([c for c in reader.connections if c.topic == JOINT_TOPIC]):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            js_ts.append(stamp_to_ns(msg.header.stamp) or bag_ts)
            js_vels.append(float(np.max(np.abs(msg.velocity))) if len(msg.velocity) else 0.0)

        print(f"[2/3] Loaded {len(tf_poses)} TF states and {len(js_vels)} joint states")

        # 3. Process Images & Synchronize
        R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
        last_T = None

        img_conns = [c for c in reader.connections if c.topic == IMAGE_TOPIC]
        for conn, bag_ts, raw in reader.messages(img_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            ts = stamp_to_ns(msg.header.stamp) or bag_ts

            # Skip if robot is moving
            vel = get_closest(js_ts, js_vels, ts)
            if vel is not None and vel > 0.02:
                continue

            T_rob = get_closest(tf_ts, tf_poses, ts)
            if T_rob is None:
                continue

            # Skip duplicate poses (require >= 1.5cm shift or >= 3 deg rotation)
            if last_T is not None:
                dist = np.linalg.norm(T_rob[:3, 3] - last_T[:3, 3])
                ang = np.degrees(np.linalg.norm(R.from_matrix(last_T[:3, :3].T @ T_rob[:3, :3]).as_rotvec()))
                if dist < 0.015 and ang < 3.0:
                    continue

            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY)

            corners, ids, _, _ = detector.detectBoard(gray)
            if ids is None or len(ids) < 8 or board.checkCharucoCornersCollinear(ids):
                continue

            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
            if not ok:
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

    if len(R_t2c) < 5:
        print(f"[ERROR] Only {len(R_t2c)} valid poses found. Hand-eye calibration requires at least 5 poses.")
        return

    print(f"[3/3] Selected {len(R_t2c)} unique static poses. Running cv2.calibrateHandEye...")

    # 4. Solve Hand-Eye Calibration
    R_est, t_est = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)

    T_link6_cam = np.eye(4)
    T_link6_cam[:3, :3] = R_est
    T_link6_cam[:3, 3] = t_est.flatten()

    # Target position consistency check in robot base frame
    target_pos = []
    for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        Tg = np.eye(4)
        Tg[:3, :3], Tg[:3, 3] = Rg, tg.flatten()
        Tt = np.eye(4)
        Tt[:3, :3], Tt[:3, 3] = Rt, tt.flatten()
        # Base -> Gripper -> Camera -> Board
        target_pos.append((Tg @ T_link6_cam @ Tt)[:3, 3])

    pos_err_mm = np.linalg.norm(np.array(target_pos) - np.mean(target_pos, axis=0), axis=1) * 1000.0
    rmse_mm = np.sqrt(np.mean(pos_err_mm**2))
    max_err_mm = np.max(pos_err_mm)

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

    print("\n================ CALIBRATION RESULT (gripper_tcp -> camera) ================")
    print(np.array2string(T_tcp_cam, precision=6, suppress_small=True))
    print(f"Translation [x, y, z] (m):   {np.round(t_tcp, 6)}")
    print(f"Quaternion [qx, qy, qz, qw]: {np.round(quat_tcp, 6)}")
    print(f"Euler XYZ (deg):             {np.round(euler_tcp, 4)}")
    print(f"Calibration accuracy (RMSE): {rmse_mm:.2f} mm (Max error: {max_err_mm:.2f} mm)")
    print("============================================================================\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "calibration_mode": CALIB_MODE,
            "parent_frame": "gripper_tcp",
            "child_frame": "camera_color_optical_frame",
            "used_unique_poses": len(R_t2c),
            "rmse_error_mm": round(float(rmse_mm), 3),
            "max_error_mm": round(float(max_err_mm), 3),
            "transformation_matrix_4x4": T_tcp_cam.tolist(),
            "translation_xyz_m": t_tcp.tolist(),
            "quaternion_xyzw": quat_tcp.tolist(),
            "euler_xyz_deg": euler_tcp.tolist(),
        }, f, indent=4)
    print(f"Saved calibration result to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
