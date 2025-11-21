## ROS1 ↔ ROS2 桥接快速指南

本指南记录了 `sim2real`（ROS1）与 `hightorque_rl_inference_ros2`（ROS2）之间的桥接流程，确保 `/imu/data`、`/sim2real_master_node/rbt_state` 等话题能够正确流向 ROS2 推理节点，同时将 ROS2 发布的关节命令回传给 ROS1。

---

### 1. 前置条件
- 已安装 ROS1 Noetic 与 ROS2 Foxy，并可分别运行。
- `ros1_bridge` 已安装（`sudo apt install ros-foxy-ros1-bridge`）。
- `sim2real` 相关 ROS1 包能够正常启动并发布传感器/状态话题。
- `hightorque_rl_inference_ros2` 已编译（`colcon build`）并可运行。

---

### 2. 启动顺序（关键）
1. **终端 A – 启动 ROS1**
   ```bash
   source /opt/ros/noetic/setup.bash
   # 另开终端启动 sim2real 相关节点，确保 /imu/data、/sim2real_master_node/* 等话题存在
   ```

2. **终端 B – 启动 ROS2 推理节点**
   ```bash
   source /opt/ros/foxy/setup.bash
   source XXX/install/setup.bash  # 使用本仓库编译结果
   ros2 launch hightorque_rl_inference_ros2 hightorque_rl_inference.launch.py
   ```
   > 推理节点会订阅 `/imu/data`、`/sim2real_master_node/rbt_state`、`/sim2real_master_node/mtr_state` 等话题，使 dynamic_bridge 有“订阅者”可检测。

3. **终端 C – 启动桥接（dynamic_bridge）**
   ```bash
   source /opt/ros/noetic/setup.bash
   source /opt/ros/foxy/setup.bash
   ros2 run ros1_bridge dynamic_bridge
   ```
   - 该命令会自动桥接所有在 ROS1/ROS2 之间存在兼容发布者与订阅者的话题。
   - 若需要限制话题，可改用 `ros2 run ros1_bridge parameter_bridge ...` 指定。

