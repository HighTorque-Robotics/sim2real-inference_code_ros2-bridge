#!/bin/bash
# 验证话题配置和数据流

echo "=========================================="
echo "验证 AMP 20DOF 话题配置"
echo "=========================================="

echo ""
echo "1. 检查 ROS2 话题列表..."
ros2 topic list | grep -E "(rbt_state|mtr_state|imu|cmd_vel|pi_plus)"

echo ""
echo "2. 检查 robot state 话题（应该有20个关节）..."
timeout 2s ros2 topic echo /sim2real_master_node/rbt_state --once | grep -E "position|velocity" | head -5

echo ""
echo "3. 检查 IMU 话题..."
timeout 2s ros2 topic echo /imu/data --once | grep -E "orientation|angular_velocity" | head -5

echo ""
echo "4. 检查推理节点发布的控制命令（应该有20个关节）..."
timeout 2s ros2 topic echo /pi_plus_all --once | grep "position" | head -5

echo ""
echo "5. 检查话题频率..."
echo "Robot state frequency:"
timeout 5s ros2 topic hz /sim2real_master_node/rbt_state

echo ""
echo "IMU frequency:"
timeout 5s ros2 topic hz /imu/data

echo ""
echo "Control command frequency (应该约50Hz):"
timeout 5s ros2 topic hz /pi_plus_all

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

