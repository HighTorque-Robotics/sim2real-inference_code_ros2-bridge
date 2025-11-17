#ifndef HIGHTORQUE_RL_INFERENCE_ROS2_HPP
#define HIGHTORQUE_RL_INFERENCE_ROS2_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <Eigen/Dense>
#include <memory>
#include <deque>
#include <shared_mutex>
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
    int numActions_;
    int numSingleObs_;
    int frameStack_;
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
    std::deque<Eigen::VectorXd> histObs_;
    Eigen::MatrixXd obsInput_;
    Eigen::VectorXd action_;

    std::vector<double> urdfOffset_;
    std::vector<int> motorDirection_;
    std::vector<int> actualToPolicyMap_;

    double stepsPeriod_;
    double step_;

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

