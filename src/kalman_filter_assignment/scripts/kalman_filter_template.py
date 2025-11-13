#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler

class SimpleKalmanFilterNode:
    def __init__(self):
        rospy.init_node('kalman_filter_simple', anonymous=True)

        # param
        self.dt = rospy.get_param('~dt', 0.1)  # Time step

        #subscriber
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/fake_gps', Odometry, self.gps_callback)
        rospy.Subscriber('/odom1', Odometry, self.odom_callback)

        # Publisher 
        self.pub = rospy.Publisher('/kalman_estimate', Odometry, queue_size=10)

        # Initial State: [x, y, yaw]
        self.x = np.zeros((3,1))
        self.P = np.eye(3) * 0.1  # Initial covariance
        rospy.loginfo(f"Initial state: {self.x}")
        rospy.loginfo(f"Initial covariance: {self.P}")

        # Noise Covariances
        # Process noise - uncertainty in motion model
        self.Q = np.diag([0.1, 0.1, 0.05])  # [x, y, yaw] process noise
        # GPS measurement noise - uncertainty in GPS measurements
        self.R_gps = np.diag([0.5, 0.5, 0.2])  # [x, y, yaw] measurement noise
        self.R_odom = np.diag([30, 30, 10])  # [x, y, yaw] odom measurement noise

        # Latest command velocities
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

        # Latest GPS measurement
        self.gps = None

        # Latest IMU/Odom data
        self.odom = None

        # Timer for Kalman update
        rospy.Timer(rospy.Duration(self.dt), self.update_kalman)

        rospy.loginfo("Kalman Filter start")

    def cmd_vel_callback(self, msg):
        """Store the latest cmd velocities."""
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.yaw_rate = msg.angular.z

    def gps_callback(self, msg):
        """Store the latest GPS measurement."""
        self.gps = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)]  # Yaw from quaternion
        ])

    def odom_callback(self, msg):
        """Store the latest odom measurement."""
        self.odom = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)]  # Yaw from quaternion
        ])

    def measurement_update(self, z, R):
        I = np.eye(3)
        H = np.eye(3)

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
        u = np.array([self.vx, self.vy, self.yaw_rate])
        cos_yaw = np.cos(self.x[2,0])
        sin_yaw = np.sin(self.x[2,0])

        x_t = self.x[0,0] + u[0] * self.dt * cos_yaw - u[1] * self.dt * sin_yaw
        y_t = self.x[1,0] + u[0] * self.dt * sin_yaw + u[1] * self.dt * cos_yaw
        yaw_t = self.x[2,0] + u[2] * self.dt
        return np.array([[x_t],
                         [y_t],
                         [yaw_t]])

    def compute_jacobian_G(self):
        """
        Compute Jacobian of motion model g with respect to state.
        G = dg/dx where g is the motion model.
        """
        cos_yaw = np.cos(self.x[2,0])
        sin_yaw = np.sin(self.x[2,0])
        
        G = np.array([
            [1, 0, -self.vx * self.dt * sin_yaw - self.vy * self.dt * cos_yaw],
            [0, 1,  self.vx * self.dt * cos_yaw - self.vy * self.dt * sin_yaw],
            [0, 0, 1]
        ])
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
        # --- Step1: Init ---
        G = self.compute_jacobian_G() # Jacobian of motion model
        H = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])  # Measurement model

        # --- Step2: Prediction ---
        x_pred = self.predict_state()
        P_pred = G @ self.P @ G.T + self.Q

        self.x = x_pred
        self.P = P_pred
        if self.gps is not None:
            # --- Step5: Correction ---
            rospy.logdebug(f"GPS measurement: {self.gps.T}")
            rospy.logdebug(f"Predicted state before correction: {self.x.T}")
            self.measurement_update(self.gps, self.R_gps)
            self.measurement_update(self.odom, self.R_odom)
            rospy.logdebug(f"Corrected state: {self.x.T}")
        else:
            rospy.logwarn("No GPS measurement available, using prediction only")

        self.publish_estimate()

    def publish_estimate(self):
        """Publish the current state estimate as Odometry message."""
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = float(self.x[0])
        msg.pose.pose.position.y = float(self.x[1])

        q = quaternion_from_euler(0,0,float(self.x[2]))
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.twist.twist.linear.x = float(self.vx)
        msg.twist.twist.linear.y = float(self.vy)
        msg.twist.twist.angular.z = float(self.yaw_rate)

        self.pub.publish(msg)


if __name__ == '__main__':
    try:
        node = SimpleKalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

