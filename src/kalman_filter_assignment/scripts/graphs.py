#!/usr/bin/env python3
import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry

import os
import re
from datetime import datetime

def get_latest_robot_position_file(dir):
    files=[f for f in os.listdir(dir) if re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.bag$",f)]
    return max(files, key=lambda f: datetime.strptime(f[:-4], "%Y-%m-%d-%H-%M-%S")) if files else None

# Config
BAG_PATH = "bags/" + get_latest_robot_position_file("bags")

GPS_TOPIC = "/fake_gps"
KALMAN_TOPIC = "/kalman_estimate"
CMD_VEL_TOPIC = "/cmd_vel"
ENCODER_TOPIC = "/odom1"
GT_TOPIC = "/odom"


DT = 0.1  # time step for cmd_vel integration

# Helpers
def extract_xy_from_odom(msg):
    return msg.pose.pose.position.x, msg.pose.pose.position.y

def extract_xy_from_twist(msg):
    return msg.linear.x, msg.linear.y, msg.angular.z

def time_in_seconds(t0, t):
    return (t - t0).to_sec()


def main():
    bag = rosbag.Bag(BAG_PATH)

    gps_t, gps_x, gps_y = [], [], []
    kal_t, kal_x, kal_y = [], [], []
    gt_t, gt_x, gt_y = [], [], []
    encoder_t, encoder_x, encoder_y = [], [], []
    cmdvel_t, cmdvel_x, cmdvel_y, cmdvel_theta = [], [], [], []

    t0 = None

    for topic, msg, t in bag.read_messages():
        if t0 is None:
            t0 = t

        if topic == GPS_TOPIC:
            x, y = extract_xy_from_odom(msg)
            gps_x.append(x)
            gps_y.append(y)
            gps_t.append(time_in_seconds(t0, t))

        elif topic == KALMAN_TOPIC:
            x, y = extract_xy_from_odom(msg)
            kal_x.append(x)
            kal_y.append(y)
            kal_t.append(time_in_seconds(t0, t))

        elif topic == GT_TOPIC:
            x, y = extract_xy_from_odom(msg)
            gt_x.append(x)
            gt_y.append(y)
            gt_t.append(time_in_seconds(t0, t))

        elif topic == ENCODER_TOPIC:
            x, y = extract_xy_from_odom(msg)
            encoder_x.append(x)
            encoder_y.append(y)
            encoder_t.append(time_in_seconds(t0, t))

    bag.close()

    #   PLOT 1: 2D TRAJECTORY (X vs Y)
    plt.figure()
    plt.plot(gps_x, gps_y, "o", markersize=3, label="GPS (sparse)")
    plt.plot(kal_x, kal_y, "-", linewidth=1.5, label="Kalman (smooth)")
    plt.plot(gt_x, gt_y, "-", linewidth=1.5, label="Ground Truth")
    # plt.plot(encoder_x, encoder_y, "-", linewidth=0.2, label="Encoder")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("2D Trajectory Comparison")
    plt.legend()
    plt.grid(True)

    #   PLOT 2: X POSITION vs TIME
    plt.figure()
    plt.plot(gps_t, gps_x, "o", markersize=3, label="GPS X")
    plt.plot(kal_t, kal_x, "-", linewidth=1.5, label="Kalman X")
    plt.plot(gt_t, gt_x, "-", linewidth=1.5, label="Ground Truth X")
    plt.xlabel("Time (s)")
    plt.ylabel("X position (m)")
    plt.title("X Position Over Time")
    plt.legend()
    plt.grid(True)

    #   PLOT 3: ERROR in X POSITION vs TIME
    plt.figure()
    # Interpolate ground truth for error calculation
    gt_x_interp = np.interp(kal_t, gt_t, gt_x)
    error_x = np.array(kal_x) - gt_x_interp
    plt.plot(kal_t, error_x, "-", linewidth=1.5, label="Kalman X Error")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position Error (m)")
    mean_error = np.mean(np.abs(error_x))
    plt.title(f"X Position Error Over Time (Mean Error: {mean_error:.2f} m)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
