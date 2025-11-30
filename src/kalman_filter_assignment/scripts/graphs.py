#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib.pyplot as plt

import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

def angle_diff(a, b):
        d = a - b
        # Normalize to -pi to +pi
        return (d + np.pi) % (2 * np.pi) - np.pi

def get_latest_robot_position_file(dir):
    files = [
        f
        for f in os.listdir(dir)
        if re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.bag$", f)
    ]
    return (
        max(files, key=lambda f: datetime.strptime(f[:-4], "%Y-%m-%d-%H-%M-%S"))
        if files
        else None
    )


# Config
BAG_PATH = "bags/" + get_latest_robot_position_file("bags")
# BAG_PATH = "bags/end_result.bag"

blacklist = [
    "/kalman_cmd",
    "/kalman_enc",
    "/kalman_imu",
    "/kalman_enc_gps",
    "/kalman_enc_imu",
    "/kalman_full_trust_process",
    "/kalman_full_no_trust_process",
    # "/kalman_enc_gps_imu",
    "/kalman_full_process_0_30",
    "/kalman_full_process_0_20",
    "/kalman_full_process_0_10",
    "/kalman_full_process_0_05",
]

GPS_TOPIC = "/fake_gps"
CMD_VEL_TOPIC = "/cmd_vel"
ENCODER_TOPIC = "/odom1"
GT_TOPIC = "/odom"


DT = 0.1  # time step for cmd_vel integration


@dataclass
class OdomData:
    t: List[float]
    x: List[float]
    y: List[float]
    yaw: List[float]

    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []


# Helpers
def extract_xy_from_odom(msg):
    return (
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.orientation.z,
    )


def extract_xy_from_twist(msg):
    return msg.linear.x, msg.linear.y, msg.angular.z


def time_in_seconds(t0, t):
    return (t - t0).to_sec()


def main():
    bag = rosbag.Bag(BAG_PATH)

    gps = OdomData()
    gt = OdomData()

    all_kal: Dict[str, OdomData] = {}
    kal_topics = [
        topic for topic, _, _ in bag.read_messages() if topic.startswith("/kalman")
    ]
    for topic in kal_topics:
        all_kal[topic] = OdomData()

    t0 = None

    for topic, msg, t in bag.read_messages():
        if t0 is None:
            t0 = t

        if topic == GPS_TOPIC:
            x, y, _ = extract_xy_from_odom(msg)
            gps.x.append(x)
            gps.y.append(y)
            gps.t.append(time_in_seconds(t0, t))

        elif topic in kal_topics:
            x, y, yaw = extract_xy_from_odom(msg)
            all_kal[topic].x.append(x)
            all_kal[topic].y.append(y)
            all_kal[topic].yaw.append(yaw)
            all_kal[topic].t.append(time_in_seconds(t0, t))

        elif topic == GT_TOPIC:
            x, y, yaw = extract_xy_from_odom(msg)
            gt.x.append(x)
            gt.y.append(y)
            gt.yaw.append(yaw)
            gt.t.append(time_in_seconds(t0, t))

    bag.close()

    #   PLOT 1: 2D TRAJECTORY (X vs Y)
    f = plt.figure()
    f.canvas.set_window_title("2D Trajectory Comparison")
    for topic, kal in all_kal.items():
        if topic in blacklist:
            continue
        plt.plot(kal.x, kal.y, "-", linewidth=1.5, label=topic)
    plt.plot(gt.x, gt.y, "-", linewidth=1.5, label="Ground Truth")
    plt.plot(gps.x, gps.y, "o", markersize=3, label="GPS (sparse)")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("2D Trajectory Comparison")
    plt.legend()
    plt.grid(True)

    #   PLOT 2: ERROR in EUCLIDIAN POSITION vs TIME
    # Interpolate ground truth for error calculation
    f = plt.figure()
    f.canvas.set_window_title("Position Error Over Time")
    for topic, kal in all_kal.items():
        if topic in blacklist:
            continue
        gt_x_interp = np.interp(kal.t, gt.t, gt.x)
        gt_y_interp = np.interp(kal.t, gt.t, gt.y)
        error_x = np.array(kal.x) - gt_x_interp
        error_y = np.array(kal.y) - gt_y_interp
        error_squared = error_x**2 + error_y**2
        error = np.sqrt(error_squared)
        rmse = np.sqrt(np.mean(error_squared))
        plt.plot(kal.t, error, "-", linewidth=1.5, label=topic + f" (rmse: {rmse:.3f} m)")

    plt.title("Position Error Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend()
    plt.grid(True)

    f = plt.figure()
    f.canvas.set_window_title("Yaw Error Over Time")
    #   PLOT 3: ERROR in YAW vs TIME
    for topic, kal in all_kal.items():
        if topic in blacklist:
            continue
        gt_sin = np.sin(gt.yaw)
        gt_cos = np.cos(gt.yaw)
        interp_sin = np.interp(kal.t, gt.t, gt_sin)
        interp_cos = np.interp(kal.t, gt.t, gt_cos)
        gt_yaw_interp = np.arctan2(interp_sin, interp_cos)
        error_yaw = angle_diff(np.array(kal.yaw), gt_yaw_interp)
        mean_yaw_error = np.mean(np.abs(error_yaw))
        plt.plot(kal.t, error_yaw, "-", linewidth=1.5, label=topic + f" (mean: {mean_yaw_error:.4f} rad)")

    plt.xlabel("Time (s)")
    plt.ylabel("Yaw Error (rad)")
    plt.title("Yaw Error Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
