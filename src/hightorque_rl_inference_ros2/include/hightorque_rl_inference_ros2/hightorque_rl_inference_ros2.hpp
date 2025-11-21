#ifndef HIGHTORQUE_RL_INFERENCE_ROS2_HPP
#define HIGHTORQUE_RL_INFERENCE_ROS2_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <Eigen/Dense>
#include <memory>
#include <shared_mutex>
#include <deque>
#include <vector>
#include <atomic>

#ifdef PLATFORM_ARM
#include "rknn_api.h"
#endif

namespace hightorque_rl_inference_ros2
{

class HighTorqueRLInference : public rclcpp::Node
{
public:
    HighTorqueRLInference();
    ~HighTorqueRLInference();

    bool init();
    void run();
    void stop() { quit_ = true; }

private:
    bool loadPolicy();
    void updateObservation();
    void updateAction();
    void quat2euler();
    Eigen::Vector3d rotateVectorByQuatVec4(const Eigen::Vector4d& q, const Eigen::Vector3d& v);

    void robotStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void motorStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointCmdPub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr presetPub_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr robotStateSub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr motorStateSub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imuSub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmdVelSub_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joySub_;

    // Configuration
    std::string modelType_;
    std::string policyPath_;
    int numActions_;          // 实际控制的关节数（如12）
    int policyDofs_;          // 策略输出的自由度（如20，包含手部）
    int numSingleObs_;
    double rlCtrlFreq_;
    double clipObs_;

    double cmdLinVelScale_;
    double cmdAngVelScale_;
    double rbtLinPosScale_;
    double rbtLinVelScale_;
    double rbtAngVelScale_;
    double actionScale_;

    std::vector<float> clipActionsLower_;
    std::vector<float> clipActionsUpper_;

    // Robot joint data (relative position from rbt_state topic)
    Eigen::VectorXd robotJointPositions_;
    Eigen::VectorXd robotJointVelocities_;

    // Motor joint data (absolute angle from mtr_state topic)
    Eigen::VectorXd motorJointPositions_;
    Eigen::VectorXd motorJointVelocities_;
    std::shared_timed_mutex mutex_;

    Eigen::Quaterniond quat_;
    Eigen::Vector3d eulerAngles_;
    Eigen::Vector3d baseAngVel_;
    Eigen::Vector3d command_;

    Eigen::VectorXd observations_;
    std::deque<Eigen::VectorXd> histObs_;  // 历史观测缓冲区（5帧）
    Eigen::MatrixXd obsInput_;
    Eigen::VectorXd action_;
    Eigen::VectorXd policyFullOutput_;  // 缓存完整的策略输出（20dof）用于输出手部关节

    std::vector<double> urdfOffset_;
    std::vector<int> motorDirection_;
    std::vector<int> actualToPolicyMap_;
    std::vector<int> policyToControlMap_;  // 从策略输出(20)映射到控制输出(12)的索引

    double stepsPeriod_;
    double step_;

    // AMP Policy specific members
    Eigen::VectorXd lastAction_;           // 上次输出的动作
    Eigen::Vector3d lastCmdVel_;           // 上次的命令速度 (用于滤波)
    std::vector<double> actionScales_;     // 动作缩放系数
    std::vector<double> defaultPose_;      // 默认姿态
    
    // 控制模式和速度参数（与 amp_policy.cpp 一致）
    std::string controlMode_;              // 控制模式 (Soccer/Fight)
    double cmdVelFilterScale_;             // 速度滤波系数（兼容旧配置）
    double cmdVelFilterScaleSoccer_;       // Soccer模式滤波系数
    double cmdVelFilterScaleFight_;        // Fight模式滤波系数
    double cmdVelXMin_;                    // x速度下限
    double cmdVelXMax_;                    // x速度上限（兼容旧配置）
    double cmdVelXMaxSoccerNormal_;        // Soccer模式正常速度上限
    double cmdVelXMaxSoccerBoost_;         // Soccer模式加速速度上限
    double cmdVelXMaxFight_;               // Fight模式速度上限
    double cmdVelYMin_, cmdVelYMax_;       // y速度限制
    double cmdVelYawMin_, cmdVelYawMax_;   // yaw速度限制
    double actionFilterScale_;             // 动作滤波系数

    std::atomic<bool> quit_;
    std::atomic<bool> stateReceived_;
    std::atomic<bool> imuReceived_;

    // Joystick state
    std::atomic<bool> joyReady_;
    sensor_msgs::msg::Joy joyMsg_;
    std::mutex joyMutex_;
    rclcpp::Time lastTrigger_;

#ifdef PLATFORM_ARM
    rknn_context ctx_{};
    rknn_input_output_num ioNum_{};
    rknn_input rknnInputs_[1]{};
    rknn_output rknnOutputs_[1]{};
#endif

    // State machine
    enum State
    {
        NOT_READY,
        STANDBY,
        RUNNING
    };
    State currentState_ = NOT_READY;
};

} // namespace hightorque_rl_inference_ros2

#endif

