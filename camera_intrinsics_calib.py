import argparse
import json
import os
import sys
import cv2
import numpy as np

# ================= CONFIGURATION =================
IMAGE_TOPIC = "/camera/camera/color/image_raw"
OUTPUT_INTRINSICS_JSON = "camera_intrinsics.json"

# ChArUco board: 7x5 squares, 30mm square, 20mm marker
board = cv2.aruco.CharucoBoard(
    (7, 5), 0.030, 0.020, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

MIN_CORNERS_PER_FRAME = 12  # Minimum ChArUco corners required to accept a frame
MIN_SAMPLES_FOR_CALIB = 12  # Minimum frames required to run intrinsic calibration
MIN_CENTROID_SHIFT_PX = (
    25.0  # Minimum board shift (pixels) between auto-sampled bag frames
)
MAX_OUTLIER_RATIO = (
    0.30  # Raise error if >30% of frames have excessive reprojection error
)
MAX_FRAME_REPROJ_PX = (
    1.0  # Per-frame reprojection error threshold for outlier rejection
)
# =================================================


def run_intrinsic_calibration(all_obj_pts, all_img_pts, image_size):
    """Calibrate camera intrinsics (K, D) with outlier rejection."""
    if len(all_obj_pts) < MIN_SAMPLES_FOR_CALIB:
        raise RuntimeError(
            f"[ERROR] Collected only {len(all_obj_pts)} frames. "
            f"At least {MIN_SAMPLES_FOR_CALIB} diverse views are required."
        )

    print(f"Running initial camera calibration on {len(all_obj_pts)} frames...")
    rms_init, K_init, D_init, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts, all_img_pts, image_size, None, None
    )

    # Compute per-frame reprojection error for Outlier Rejection
    frame_errors = []
    for i in range(len(all_obj_pts)):
        proj_pts, _ = cv2.projectPoints(
            all_obj_pts[i], rvecs[i], tvecs[i], K_init, D_init
        )
        err = np.sqrt(
            np.mean(
                np.sum(
                    (all_img_pts[i].reshape(-1, 2) - proj_pts.reshape(-1, 2)) ** 2,
                    axis=1,
                )
            )
        )
        frame_errors.append(err)

    frame_errors = np.array(frame_errors)
    threshold = max(MAX_FRAME_REPROJ_PX, float(np.median(frame_errors) * 2.0))
    inliers = [i for i, e in enumerate(frame_errors) if e <= threshold]

    num_total = len(all_obj_pts)
    num_outliers = num_total - len(inliers)
    outlier_ratio = num_outliers / num_total

    if outlier_ratio > MAX_OUTLIER_RATIO or len(inliers) < MIN_SAMPLES_FOR_CALIB:
        raise RuntimeError(
            f"[ERROR] Camera calibration outlier rejection failed: {num_outliers}/{num_total} frames "
            f"({outlier_ratio * 100:.1f}%) exceeded {threshold:.2f} px reprojection error. "
            "Check board dimensions or motion blur."
        )

    if num_outliers > 0:
        print(
            f"Filtered out {num_outliers}/{num_total} outlier frames (threshold: {threshold:.2f} px)"
        )
        all_obj_pts = [all_obj_pts[i] for i in inliers]
        all_img_pts = [all_img_pts[i] for i in inliers]
        rms_final, K_final, D_final, _, _ = cv2.calibrateCamera(
            all_obj_pts, all_img_pts, image_size, None, None
        )
    else:
        rms_final, K_final, D_final = rms_init, K_init, D_init

    print("\n================ CAMERA INTRINSICS RESULT ================")
    print(f"Image resolution:        {image_size[0]}x{image_size[1]}")
    print(
        f"Used valid frames:       {len(all_obj_pts)} (rejected outliers: {num_outliers})"
    )
    print(f"Reprojection RMSE (px):  {rms_final:.4f}")
    print(
        f"Camera Matrix K:\n{np.array2string(K_final, precision=4, suppress_small=True)}"
    )
    print(f"Distortion Coeffs D:     {np.round(D_final.flatten(), 6)}")
    print("==========================================================\n")

    with open(OUTPUT_INTRINSICS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "width": image_size[0],
                "height": image_size[1],
                "reprojection_rmse_px": round(float(rms_final), 5),
                "used_frames": len(all_obj_pts),
                "rejected_outliers": num_outliers,
                "camera_matrix": K_final.tolist(),
                "distortion_coefficients": D_final.flatten().tolist(),
            },
            f,
            indent=4,
        )
    print(f"Saved calibrated intrinsics to: {OUTPUT_INTRINSICS_JSON}")


def calibrate_from_bag(bag_path):
    """Extract diverse ChArUco views from a ROS 2 bag and calibrate intrinsics."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    all_obj_pts, all_img_pts = [], []
    image_size = None
    last_centroid = None
    last_area = None

    with Reader(bag_path) as reader:
        img_conns = [c for c in reader.connections if c.topic == IMAGE_TOPIC]
        for conn, _, raw in reader.messages(img_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            if image_size is None:
                image_size = (int(msg.width), int(msg.height))

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
                or len(ids) < MIN_CORNERS_PER_FRAME
                or board.checkCharucoCornersCollinear(ids)
            ):
                continue

            pts_2d = corners.reshape(-1, 2)
            centroid = np.mean(pts_2d, axis=0)
            span = np.ptp(pts_2d, axis=0)
            area = float(span[0] * span[1])

            # Ensure diverse board positions and scales across the image sensor
            if last_centroid is not None:
                shift = np.linalg.norm(centroid - last_centroid)
                area_change = abs(area - last_area) / max(last_area, 1.0)
                if shift < MIN_CENTROID_SHIFT_PX and area_change < 0.12:
                    continue

            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            all_obj_pts.append(obj_pts.astype(np.float32))
            all_img_pts.append(img_pts.astype(np.float32))
            last_centroid = centroid
            last_area = area

    print(f"Extracted {len(all_obj_pts)} diverse board views from bag: {bag_path}")
    run_intrinsic_calibration(all_obj_pts, all_img_pts, image_size)


def calibrate_live():
    """Interactive ROS 2 node for live camera intrinsic calibration."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class LiveIntrinsicCalibNode(Node):
        def __init__(self):
            super().__init__("camera_intrinsic_calib_node")
            self.all_obj_pts = []
            self.all_img_pts = []
            self.image_size = None
            self.coverage_mask = None
            self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, 10)
            self.get_logger().info(
                "Live Camera Calibration started. [SPACE/S] = capture view, [C] = calibrate, [Q] = quit"
            )

        def image_cb(self, msg: Image):
            if self.image_size is None:
                self.image_size = (int(msg.width), int(msg.height))
                self.coverage_mask = np.zeros(
                    (msg.height, msg.width, 3), dtype=np.uint8
                )

            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            vis = cv2.addWeighted(img, 0.85, self.coverage_mask, 0.4, 0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _, _ = detector.detectBoard(gray)

            n_corners = len(ids) if ids is not None else 0
            valid = (
                ids is not None
                and n_corners >= MIN_CORNERS_PER_FRAME
                and not board.checkCharucoCornersCollinear(ids)
            )

            if valid:
                cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 255, 0))

            cv2.putText(
                vis,
                f"Views: {len(self.all_obj_pts)}/{MIN_SAMPLES_FOR_CALIB}+ | Corners: {n_corners}/24",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if valid else (0, 165, 255),
                2,
            )
            cv2.imshow("Camera Intrinsic Calibration", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord(" "), ord("s")):
                if not valid:
                    self.get_logger().warn(
                        f"Need at least {MIN_CORNERS_PER_FRAME} non-collinear corners!"
                    )
                    return
                obj_pts, img_pts = board.matchImagePoints(corners, ids)
                self.all_obj_pts.append(obj_pts.astype(np.float32))
                self.all_img_pts.append(img_pts.astype(np.float32))
                for pt in img_pts.reshape(-1, 2):
                    cv2.circle(
                        self.coverage_mask,
                        (int(pt[0]), int(pt[1])),
                        6,
                        (0, 255, 100),
                        -1,
                    )
                self.get_logger().info(
                    f"Captured view #{len(self.all_obj_pts)} ({n_corners} corners)"
                )

            elif key == ord("c"):
                try:
                    run_intrinsic_calibration(
                        self.all_obj_pts, self.all_img_pts, self.image_size
                    )
                except RuntimeError as err:
                    self.get_logger().error(str(err))

            elif key == ord("q"):
                cv2.destroyAllWindows()
                rclpy.shutdown()

    rclpy.init()
    node = LiveIntrinsicCalibNode()
    rclpy.spin(node)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Camera Intrinsic Calibration using ChArUco"
    )
    parser.add_argument(
        "--bag", type=str, default=None, help="Path to ROS 2 bag directory (optional)"
    )
    args = parser.parse_args()

    try:
        if args.bag:
            calibrate_from_bag(args.bag)
        else:
            calibrate_live()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
