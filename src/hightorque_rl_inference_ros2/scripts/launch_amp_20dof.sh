#!/bin/bash
# 快速启动 AMP 20DOF 推理节点

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PKG_DIR/config/config_example.yaml"

echo "=========================================="
echo "启动 AMP 20DOF 推理节点"
echo "=========================================="
echo ""
echo "配置文件: $CONFIG_FILE"
echo "模型类型: pi_plus"
echo "控制频率: 50Hz"
echo "关节自由度: 20 (12腿 + 8手臂)"
echo ""
echo "=========================================="
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查是否在 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "错误: 未检测到 ROS2 环境，请先 source ROS2 工作空间"
    exit 1
fi

echo "启动推理节点..."
echo ""

ros2 run hightorque_rl_inference_ros2 hightorque_rl_inference_ros2 \
    --ros-args \
    -p config_file:="$CONFIG_FILE" \
    -p model_type:=pi_plus \
    -p steps_period:=60.0 \
    -p joy_topic:=/joy \
    -p joy_axis2:=2 \
    -p joy_axis5:=5 \
    -p joy_button_start:=7 \
    -p joy_button_lb:=4 \
    -p reset_duration:=2.0

echo ""
echo "节点已退出"

