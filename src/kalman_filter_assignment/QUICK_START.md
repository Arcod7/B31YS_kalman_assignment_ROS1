## Terminal 1:
Launch the roscore

```bash
roscore
```

## Terminal 2:
Launch the kalman_estimate

Init: `rosparam set use_sim_time true`

```bash
roslaunch kalman_filter_assignment kalman_execution.launch student_name:=AntoineEsman
```

## Terminal 3:
Record the kalman estimate

Init: `cd ~/catkin2_ws/src/B31YS_kalman_assignment_ROS1/src/kalman_filter_assignment/scripts/bags`

```bash
rosbag record -O result_config.bag /kalman_estimate /tf /odom
```

## Terminal 4:
Play the robot movement for 60 sec

Init: `cd ~/catkin2_ws/src/B31YS_kalman_assignment_ROS1/src/kalman_filter_assignment/scripts/bags`

```bash
rosbag play --clock moving_robot.bag
```

## Terminal 5:
Plot some graphs

Init: `cd ~/catkin2_ws/src/B31YS_kalman_assignment_ROS1/src/kalman_filter_assignment/scripts`

```bash
python3 graph.py
```
