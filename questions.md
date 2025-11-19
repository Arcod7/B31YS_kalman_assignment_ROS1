Does the covariance need to change over time or is it fixed?
Do we need to use the /odom1 twist or pose ? I don't see how we can use the twist information and the pose is drifted over time


imu data is much more reliable

integrate imu in the prediction step


plot odom (ground truth)
result of the kalman filter
different combinaison of sensors

We can pick imu gyroscope as the ground truth for orientation
and use odometry velocity to predict position and correct it with gps