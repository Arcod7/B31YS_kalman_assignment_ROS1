# Assignment 3
https://canvas.hw.ac.uk/courses/30401/assignments/190260

A PDF containing:
    A 2D plot showing X vs Y trajectories for GPS, Kalman, and ground truth
        Or: A 2D plot showing X-axis: Time (seconds); Y-axis: Position (meters); And three lines: GPS measurements (sparse), Kalman estimate (smooth), Ground truth (reference).
A screenshot of RViz showing the odometry estimation running
A screenshot of the code you created

- X vs Y trajectory
- X over Time
- Y over Time
- Yaw over Time

imu data is much more reliable

integrate imu in the prediction step


plot odom (ground truth)
result of the kalman filter
different combinaison of sensors

We can pick imu gyroscope as the ground truth for orientation
and use odometry velocity to predict position and correct it with gps