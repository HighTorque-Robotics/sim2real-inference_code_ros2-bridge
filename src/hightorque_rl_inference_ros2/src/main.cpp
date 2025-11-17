/**
 * @file main.cpp
 * @brief Main entry point for HighTorque RL Inference ROS2 node
 */

#include <rclcpp/rclcpp.hpp>
#include "hightorque_rl_inference_ros2/hightorque_rl_inference_ros2.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<hightorque_rl_inference_ros2::HighTorqueRLInference>();

    if (!node->init())
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize HighTorque RL Inference node");
        rclcpp::shutdown();
        return 1;
    }

    RCLCPP_INFO(node->get_logger(), "HighTorque RL Inference node initialized successfully");
    RCLCPP_INFO(node->get_logger(), "Starting inference loop...");

    node->run();

    node->stop();
    rclcpp::shutdown();
    return 0;
}

