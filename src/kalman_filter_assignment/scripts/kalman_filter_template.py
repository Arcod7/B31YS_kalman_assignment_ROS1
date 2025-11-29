#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler
from dataclasses import dataclass

@dataclass
class KalmanFilterConfig:
    config_name: str

    output_topic: str

    ## Noise Covariances
    # Process noise - uncertainty in motion model
    Q: np.ndarray  # [x, y, yaw] process noise
    # GPS measurement noise - uncertainty in GPS measurements
    R_gps: np.ndarray  # [x, y] measurement noise

    ## Fusion for velocity prediction, the sum of weights should be 1
    imu_weight: float
    encoder_weight: float
    cmdvel_weight: float

    ## Fusion for yaw_rate prediction, the sum of weights should be 1
    imu_yawrate_weight: float
    cmdvel_yawrate_weight: float
    encoder_yawrate_weight: float

    use_gps: bool = True


node_configs = [
    KalmanFilterConfig(
        config_name="cmd",
        output_topic="/kalman_cmd",
        Q=np.diag([0.1, 0.1, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([30, 30]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.0,
        cmdvel_weight=1.0,
        imu_yawrate_weight=0.0,
        cmdvel_yawrate_weight=1.0,
        encoder_yawrate_weight=0.0,
        use_gps=False,
    ),
    KalmanFilterConfig(
        config_name="cmd",
        output_topic="/kalman_enc",
        Q=np.diag([0.1, 0.1, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([30, 30]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=1.0,
        cmdvel_weight=0.0,
        imu_yawrate_weight=0.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=1.0,
        use_gps=False,
    ),
    KalmanFilterConfig(
        config_name="cmd",
        output_topic="/kalman_imu",
        Q=np.diag([0.1, 0.1, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([30, 30]),  # [x, y] measurement noise
        imu_weight=1.0,
        encoder_weight=0.0,
        cmdvel_weight=0.0,
        imu_yawrate_weight=0.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=1.0,
        use_gps=False,
    ),
    KalmanFilterConfig(
        config_name="cmd",
        output_topic="/kalman_full1",
        Q=np.diag([0.1, 0.1, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([20, 20]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.8,
        cmdvel_weight=0.2,
        imu_yawrate_weight=0.7,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.2,
        use_gps=True,
    ),
    KalmanFilterConfig(
        config_name="cmd",
        output_topic="/kalman_full2",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([20, 20]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.8,
        cmdvel_weight=0.2,
        imu_yawrate_weight=0.7,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.2,
        use_gps=True,
    ),
]


class SimpleKalmanFilterNode:
    def __init__(self, config=KalmanFilterConfig):
        rospy.init_node("kalman_filter_simple", anonymous=True)

        # param
        self.dt = rospy.get_param("~dt", 0.1)  # Time step

        # subscriber
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)
        if config.use_gps:
            rospy.Subscriber("/fake_gps", Odometry, self.gps_callback)
        rospy.Subscriber("/odom1", Odometry, self.odom_callback)
        rospy.Subscriber("/imu", Imu, self.imu_callback)

        # Publisher
        self.pub = rospy.Publisher(config.output_topic, Odometry, queue_size=10)

        # Initial State: [x, y, yaw]
        self.x = None  # np.zeros((3,1))
        self.P = np.eye(3) * 0.1  # Initial covariance
        rospy.loginfo(f"Initial state: {self.x}")
        rospy.loginfo(f"Initial covariance: {self.P}")

        self.config = config

        # Latest command velocities
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

        # Latest GPS measurement
        self.gps = None

        # Latest Encoder data
        self.odom_vx = None
        self.odom_vy = None
        self.odom_yaw_rate = None

        # Latest IMU data
        self.imu_vx = None
        self.imu_vy = None
        self.imu_yaw_rate = None

        # Timer for Kalman update
        rospy.Timer(rospy.Duration(self.dt), self.update_kalman)

        # Ground truth
        rospy.Subscriber(
            "/odom", Odometry, self.real_odom_callback
        )  # To plot against real odom
        self.real_odom = None

        rospy.loginfo("Kalman Filter start")

    def cmd_vel_callback(self, msg):
        """Store the latest cmd velocities."""
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.yaw_rate = msg.angular.z

    def odom_callback(self, msg):
        """Store the latest odom measurement."""
        self.odom_vx = msg.twist.twist.linear.x
        self.odom_vy = msg.twist.twist.linear.y
        self.odom_yaw_rate = msg.twist.twist.angular.z

    def imu_callback(self, msg):
        """Store the latest IMU measurement."""
        # Yaw rate from IMU angular velocity around Z axis
        self.imu_vx = msg.linear_acceleration.x
        self.imu_vy = msg.linear_acceleration.y
        self.imu_yaw_rate = msg.angular_velocity.z

    def gps_callback(self, msg):
        """Store the latest GPS measurement."""
        self.gps = np.array(
            [
                [msg.pose.pose.position.x],
                [msg.pose.pose.position.y],
            ]
        )
        # --- Step1: Init ---
        H = np.array([[1, 0, 0], [0, 1, 0]])  # Measurement model

        # --- Step5: Correction ---
        if self.gps is not None and self.x is not None:
            rospy.logdebug(f"GPS measurement: {self.gps.T}")
            rospy.logdebug(f"Predicted state before correction: {self.x.T}")
            self.measurement_update(self.gps, self.config.R_gps, H)
            rospy.logdebug(f"Corrected state: {self.x.T}")
        else:
            rospy.logwarn("No GPS measurement available, using prediction only")

    def real_odom_callback(self, msg):
        """Store the real odom for error plotting."""
        self.real_odom = np.array(
            [
                [msg.pose.pose.position.x],
                [msg.pose.pose.position.y],
                [
                    2
                    * np.arctan2(
                        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
                    )
                ],  # Yaw from quaternion
            ]
        )
        if self.x is None:
            self.x = self.real_odom.copy()
            rospy.loginfo(f"Initialized state to real odom: {self.x.T}")

    def measurement_update(self, z, R, H):
        I = np.eye(3)

        # --- Step3: Estimation ---
        S = H @ self.P @ H.T + R

        # --- Step4: Compute Kalman Gain ---
        K_gain = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K_gain @ (z - H @ self.x)
        # Joseph form for stability:
        self.P = (I - K_gain @ H) @ self.P @ (I - K_gain @ H).T + K_gain @ R @ K_gain.T

    def predict_state(self):
        """
        Predict the next state based on current state x and control input u.
        Named g in the lecture.
        x: current state [x, y, yaw]
        u: control input [vx, vy, yaw_rate]
        """
        if self.imu_vx is not None and self.odom_vx is not None:
            # Mean cmd_vel and encoders with weights
            vx = (
                self.vx * self.config.cmdvel_weight
                + self.imu_vx * self.config.imu_weight
                + self.odom_vx * self.config.encoder_weight
            )
            vy = (
                self.vy * self.config.cmdvel_weight
                + self.imu_vy * self.config.imu_weight
                + self.odom_vy * self.config.encoder_weight
            )
        else:
            vx = self.vx
            vy = self.vy
        if self.imu_yaw_rate is not None and self.odom_yaw_rate is not None:
            yaw_rate = (
                self.yaw_rate * self.config.cmdvel_yawrate_weight
                + self.imu_yaw_rate * self.config.imu_yawrate_weight
                + self.odom_yaw_rate * self.config.encoder_yawrate_weight
            )
        else:
            yaw_rate = self.yaw_rate

        u = np.array([vx, vy, yaw_rate])
        cos_yaw = np.cos(self.x[2, 0])
        sin_yaw = np.sin(self.x[2, 0])

        x_t = self.x[0, 0] + u[0] * self.dt * cos_yaw - u[1] * self.dt * sin_yaw
        y_t = self.x[1, 0] + u[0] * self.dt * sin_yaw + u[1] * self.dt * cos_yaw
        yaw_t = self.x[2, 0] + u[2] * self.dt
        return np.array([[x_t], [y_t], [yaw_t]])

    def compute_jacobian_G(self):
        """
        Compute Jacobian of motion model g with respect to state.
        G = dg/dx where g is the motion model.
        """
        cos_yaw = np.cos(self.x[2, 0])
        sin_yaw = np.sin(self.x[2, 0])

        G = np.array(
            [
                [1, 0, -self.vx * self.dt * sin_yaw - self.vy * self.dt * cos_yaw],
                [0, 1, self.vx * self.dt * cos_yaw - self.vy * self.dt * sin_yaw],
                [0, 0, 1],
            ]
        )
        return G

    def update_kalman(self, event):
        """
        This is the main Kalman filter loop. In this function
        you should do a prediction plus a correction step.

        Pseudo-code:
        xt = g(ut, xt-1)
        Pt = Gt*Pt-1*Gt' + R
        Kt = Pt*Ht'*(Ht*Pt*Ht' + Q)^-1
        xt' = xt + Kt*(z - h(xt))
        Pt' = (I - Kt*Ht)*Pt

        Note: R and Q are probably inverted


        Pseudo-code found [online](https://www.researchgate.net/publication/342479702_Accurate_indoor_positioning_with_ultra-wide_band_sensors/figures?lo=1):
        Inputs: x_est, P_est, z, Q, R
        Outputs: x_updated, P_updated
        Step 1: Initialize G matrix and H matrix
        Step 2: Predict state vector and covariance
            x_pred = g(u, x_est)
            P_pred = G * P_est * GT + Q
        Step 3: Estimation
            S = H * P_pred * HT + R
        Step 4: Compute Kalman gain factor
            K_gain = P_pred * HT * S^-1
        Step 5: Correction based on observation
            x_updated = x_pred + K_gain * (z - H * x_pred)
            P_updated = P_pred - K_gain * H * P_pred
                For simplicity, changed to -> P_updated = (I - K_gain * H) * P_pred
        Step 6: Return x_updated, P_updated
        """
        if self.x is None:
            rospy.logwarn("State not initialized, skipping Kalman update")
            return
        # --- Step1: Init ---
        G = self.compute_jacobian_G()  # Jacobian of motion model

        # --- Step2: Prediction ---
        x_pred = self.predict_state()
        P_pred = G @ self.P @ G.T + self.config.Q

        self.x = x_pred
        self.P = P_pred

        self.publish_estimate()

    def publish_estimate(self):
        """Publish the current state estimate as Odometry message."""
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = float(self.x[0])
        msg.pose.pose.position.y = float(self.x[1])

        q = quaternion_from_euler(0, 0, float(self.x[2]))
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.twist.twist.linear.x = float(self.vx)
        msg.twist.twist.linear.y = float(self.vy)
        msg.twist.twist.angular.z = float(self.yaw_rate)

        self.pub.publish(msg)


if __name__ == "__main__":
    try:
        for node_config in node_configs:
            rospy.loginfo(f"Starting Kalman Filter Node: {node_config.config_name}")
            node = SimpleKalmanFilterNode(config=node_config)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
