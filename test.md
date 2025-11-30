# Test1
KalmanFilterConfig(
        output_topic="/kalman_full",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([30, 30]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=0.5,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.5,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full1",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([100, 100]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=0.5,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.5,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full2",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([30, 30]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=1.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full3",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([30, 30]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=0.0,
        cmdvel_yawrate_weight=1.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),

# Test2
KalmanFilterConfig(
        output_topic="/kalman_full",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([100, 100]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=1.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full1",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([200, 200]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=1.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full2",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([100, 100]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=1.0,
        cmdvel_weight=0.0,
        imu_yawrate_weight=1.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full3",
        Q=np.diag([0.2, 0.2, 0.1]),  # [x, y, yaw] process noise
        R_gps=np.diag([100, 100]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=0.9,
        cmdvel_weight=0.1,
        imu_yawrate_weight=1.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),

# Test3 imu vs encoder for yaw

KalmanFilterConfig(
        output_topic="/kalman_full_yaw_imu",
        Q=np.diag([0.15, 0.15, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([200, 200]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=1.0,
        cmdvel_weight=0.0,
        imu_yawrate_weight=1.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.0,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full_yaw_enc",
        Q=np.diag([0.15, 0.15, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([200, 200]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=1.0,
        cmdvel_weight=0.0,
        imu_yawrate_weight=0.0,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=1.0,
        use_gps=True,
    ),
    KalmanFilterConfig(
        output_topic="/kalman_full_yaw_mix",
        Q=np.diag([0.15, 0.15, 0.05]),  # [x, y, yaw] process noise
        R_gps=np.diag([200, 200]),  # [x, y] measurement noise
        imu_weight=0.0,
        encoder_weight=1.0,
        cmdvel_weight=0.0,
        imu_yawrate_weight=0.5,
        cmdvel_yawrate_weight=0.0,
        encoder_yawrate_weight=0.5,
        use_gps=True,
    ),