/**
 * @file hightorque_rl_inference_ros2.cpp
 * @brief HighTorque RL Inference Package - ROS2 Foxy implementation
 *        高擎机电强化学习推理功能包 - ROS2 Foxy 实现
 * 
 * This file implements the core reinforcement learning inference system for
 * humanoid robot control using ROS2 Foxy.
 * 
 * @author HighTorque Robotics
 * @date 2025
 * @copyright Copyright (c) 2025 HighTorque Robotics
 */

#include "hightorque_rl_inference_ros2/hightorque_rl_inference_ros2.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>

namespace hightorque_rl_inference_ros2
{

/**
 * @brief Load data from file at specific offset
 */
static unsigned char* loadData(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

/**
 * @brief Read entire file into memory
 */
static unsigned char* readFileData(const char* filename, int* modelSize)
{
    FILE* fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = loadData(fp, 0, size);
    fclose(fp);

    *modelSize = size;
    return data;
}

/**
 * @brief Constructor - loads configuration from YAML file
 */
HighTorqueRLInference::HighTorqueRLInference()
    : Node("hightorque_rl_inference_ros2"),
      quit_(false),
      stateReceived_(false),
      imuReceived_(false),
      joyReady_(false),
      lastTrigger_(this->now())
{
    RCLCPP_INFO(this->get_logger(), "=== Loading configuration from YAML ===");

    // Declare parameters
    this->declare_parameter<std::string>("config_file", "");
    this->declare_parameter<std::string>("model_type", "pi_plus");
    this->declare_parameter<double>("steps_period", 60.0);
    this->declare_parameter<std::string>("joy_topic", "/joy");
    this->declare_parameter<int>("joy_axis2", 2);
    this->declare_parameter<int>("joy_axis5", 5);
    this->declare_parameter<int>("joy_button_start", 7);
    this->declare_parameter<int>("joy_button_lb", 4);
    this->declare_parameter<double>("reset_duration", 2.0);

    // Get package path
    std::string pkgPath;
    try
    {
        pkgPath = ament_index_cpp::get_package_share_directory("hightorque_rl_inference_ros2");
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get package path: %s", e.what());
        pkgPath = ".";
    }

    // Load config file
    std::string configFile;
    this->get_parameter("config_file", configFile);
    if (configFile.empty())
    {
        configFile = pkgPath + "/config/config_example.yaml";
    }

    RCLCPP_INFO(this->get_logger(), "Loading config from: %s", configFile.c_str());

    try
    {
        YAML::Node config = YAML::LoadFile(configFile);

        // Read basic parameters
        numActions_ = config["num_actions"].as<int>(12);
        numSingleObs_ = config["num_single_obs"].as<int>(36);
        frameStack_ = config["frame_stack"].as<int>(1);
        clipObs_ = config["clip_obs"].as<double>(18.0);

        // Read policy name and build path
        std::string policyName = config["policy_name"].as<std::string>("policy_0322_12dof_4000.rknn");
        policyPath_ = pkgPath + "/policy/" + policyName;

        // Read control frequency
        double dt = config["dt"].as<double>(0.001);
        int decimation = config["decimation"].as<int>(10);
        rlCtrlFreq_ = 1.0 / (dt * decimation);

        // Read scaling parameters
        cmdLinVelScale_ = config["cmd_lin_vel_scale"].as<double>(1.0);
        cmdAngVelScale_ = config["cmd_ang_vel_scale"].as<double>(1.25);
        rbtLinPosScale_ = config["rbt_lin_pos_scale"].as<double>(1.0);
        rbtLinVelScale_ = config["rbt_lin_vel_scale"].as<double>(1.0);
        rbtAngVelScale_ = config["rbt_ang_vel_scale"].as<double>(1.0);
        actionScale_ = config["action_scale"].as<double>(1.0);

        // Read action limits
        std::vector<double> clipLower = config["clip_actions_lower"].as<std::vector<double>>();
        std::vector<double> clipUpper = config["clip_actions_upper"].as<std::vector<double>>();

        // Read motor configuration
        if (config["motor_direction"])
        {
            motorDirection_ = config["motor_direction"].as<std::vector<int>>();
        }
        if (config["urdf_dof_pos_offset"])
        {
            urdfOffset_ = config["urdf_dof_pos_offset"].as<std::vector<double>>();
        }
        if (config["map_index"])
        {
            actualToPolicyMap_ = config["map_index"].as<std::vector<int>>();
        }

        // Model type (can be overridden from parameters)
        this->get_parameter("model_type", modelType_);

        RCLCPP_INFO(this->get_logger(), "YAML config loaded successfully:");
        RCLCPP_INFO(this->get_logger(), "  num_actions: %d", numActions_);
        RCLCPP_INFO(this->get_logger(), "  num_single_obs: %d", numSingleObs_);
        RCLCPP_INFO(this->get_logger(), "  frame_stack: %d", frameStack_);
        RCLCPP_INFO(this->get_logger(), "  rl_ctrl_freq: %.1f Hz", rlCtrlFreq_);
        RCLCPP_INFO(this->get_logger(), "  policy_path: %s", policyPath_.c_str());
        RCLCPP_INFO(this->get_logger(), "  action_scale: %.2f", actionScale_);

        clipActionsLower_.resize(numActions_);
        clipActionsUpper_.resize(numActions_);
        for (int i = 0; i < numActions_ && i < (int)clipLower.size(); ++i)
        {
            clipActionsLower_[i] = static_cast<float>(clipLower[i]);
            clipActionsUpper_[i] = static_cast<float>(clipUpper[i]);
        }
    }
    catch (const YAML::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "YAML parsing error: %s", e.what());
        RCLCPP_ERROR(this->get_logger(), "Using default parameters");

        // Use default values
        numActions_ = 12;
        numSingleObs_ = 36;
        frameStack_ = 1;
        rlCtrlFreq_ = 100.0;
        clipObs_ = 18.0;
        cmdLinVelScale_ = 1.0;
        cmdAngVelScale_ = 1.25;
        rbtLinPosScale_ = 1.0;
        rbtLinVelScale_ = 1.0;
        rbtAngVelScale_ = 1.0;
        actionScale_ = 1.0;
        policyPath_ = pkgPath + "/policy/policy_0322_12dof_4000.rknn";
        this->get_parameter("model_type", modelType_);
        stepsPeriod_ = 60.0;

        // Default limits
        std::vector<float> lower = {-1.00, -0.40, -0.60, -1.30, -0.75, -0.30, -1.00, -0.40, -0.60, -1.30, -0.75, -0.30};
        std::vector<float> upper = {1.00, 0.40, 0.60, 1.30, 0.75, 0.30, 1.00, 0.40, 0.60, 1.30, 0.75, 0.30};
        clipActionsLower_ = lower;
        clipActionsUpper_ = upper;

        // Default motor configuration
        motorDirection_ = {1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1};
        actualToPolicyMap_ = {5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6};
    }

    // Initialize Eigen vectors
    robotJointPositions_ = Eigen::VectorXd::Zero(numActions_);
    robotJointVelocities_ = Eigen::VectorXd::Zero(numActions_);
    motorJointPositions_ = Eigen::VectorXd::Zero(numActions_);
    motorJointVelocities_ = Eigen::VectorXd::Zero(numActions_);
    eulerAngles_ = Eigen::Vector3d::Zero();
    baseAngVel_ = Eigen::Vector3d::Zero();
    command_ = Eigen::Vector3d::Zero();
    action_ = Eigen::VectorXd::Zero(numActions_);

    observations_ = Eigen::VectorXd::Zero(numSingleObs_);
    for (int i = 0; i < frameStack_; ++i)
    {
        histObs_.push_back(Eigen::VectorXd::Zero(numSingleObs_));
    }
    obsInput_ = Eigen::MatrixXd::Zero(1, numSingleObs_ * frameStack_);

    quat_ = Eigen::Quaterniond::Identity();

    // Check vector sizes
    if (urdfOffset_.size() != static_cast<size_t>(numActions_))
    {
        urdfOffset_.assign(numActions_, 0.0);
    }
    if (motorDirection_.size() != static_cast<size_t>(numActions_))
    {
        motorDirection_.assign(numActions_, 1);
    }
    if (actualToPolicyMap_.size() != static_cast<size_t>(numActions_))
    {
        std::vector<int> defaultMap(numActions_);
        for (int i = 0; i < numActions_; ++i)
            defaultMap[i] = i;
        actualToPolicyMap_ = defaultMap;
    }

    this->get_parameter("steps_period", stepsPeriod_);
    step_ = 0.0;
}

HighTorqueRLInference::~HighTorqueRLInference()
{
    quit_ = true;
#ifdef PLATFORM_ARM
    rknn_destroy(ctx_);
#endif
}

bool HighTorqueRLInference::init()
{
    // Create publishers with QoS settings for real-time control
    // 为实时控制创建具有 QoS 设置的发布者
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1000))
        .reliable()  // 确保消息可靠传输
        .durability_volatile();  // 不保存历史消息，降低延迟
    
    std::string topicName = "/" + modelType_ + "_all";
    jointCmdPub_ = this->create_publisher<sensor_msgs::msg::JointState>(topicName, qos);

    auto preset_qos = rclcpp::QoS(rclcpp::KeepLast(10))
        .reliable()
        .durability_volatile();
    std::string presetTopic = "/" + modelType_ + "_preset";
    presetPub_ = this->create_publisher<sensor_msgs::msg::JointState>(presetTopic, preset_qos);

    // Create subscribers with QoS settings for high-frequency control (100Hz)
    // 为高频控制（100Hz）创建具有 QoS 设置的订阅者
    // 注意：从 ROS1 桥接的话题需要使用兼容的 QoS 策略
    
    // 对于关节状态话题，使用 reliable + volatile
    auto joint_qos = rclcpp::QoS(rclcpp::KeepLast(100))
        .reliable()  // ROS1 bridge 默认使用 reliable
        .durability_volatile();
    
    robotStateSub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/sim2real_master_node/rbt_state", joint_qos,
        std::bind(&HighTorqueRLInference::robotStateCallback, this, std::placeholders::_1));

    motorStateSub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/sim2real_master_node/mtr_state", joint_qos,
        std::bind(&HighTorqueRLInference::motorStateCallback, this, std::placeholders::_1));

    // 对于 IMU 话题，使用 reliable + volatile（与 simple_bridge 兼容）
    // simple_bridge 默认使用 reliable，必须匹配
    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(200))  // 增大队列
        .reliable()  // 与 simple_bridge 兼容
        .durability_volatile();
    
    imuSub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", imu_qos,
        std::bind(&HighTorqueRLInference::imuCallback, this, std::placeholders::_1));

    auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(50))
        .best_effort()
        .durability_volatile();
    cmdVelSub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", cmd_qos,
        std::bind(&HighTorqueRLInference::cmdVelCallback, this, std::placeholders::_1));

    std::string joy_topic;
    this->get_parameter("joy_topic", joy_topic);
    joySub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        joy_topic, 10,
        std::bind(&HighTorqueRLInference::joyCallback, this, std::placeholders::_1));

    // Attempt to wait for initial state but allow the node to continue if data has
    // not yet arrived. dynamic_bridge 可能延迟创建话题桥接，因此不再强制退出。
    rclcpp::Rate rate(100);
    int timeout = 50;
    while (rclcpp::ok() && (!stateReceived_ || !imuReceived_) && timeout > 0)
    {
        rclcpp::spin_some(this->get_node_base_interface());
        rate.sleep();
        timeout--;
    }
    if (!stateReceived_)
    {
        RCLCPP_WARN(this->get_logger(),
            "Initial robot or motor state not received yet, continuing without data.");
    }
    if (!imuReceived_)
    {
        RCLCPP_WARN(this->get_logger(),
            "Initial IMU data not received yet, continuing without data.");
    }

    if (!loadPolicy())
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to load policy");
        return false;
    }
    return true;
}

bool HighTorqueRLInference::loadPolicy()
{
    RCLCPP_INFO(this->get_logger(), "Loading RKNN policy from: %s", policyPath_.c_str());
    
    int modelSize = 0;
    unsigned char* modelData = readFileData(policyPath_.c_str(), &modelSize);
    if (!modelData)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to read policy file: %s", policyPath_.c_str());
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "Policy file loaded, size: %d bytes", modelSize);
    
    int ret = rknn_init(&ctx_, modelData, modelSize, 0, nullptr);
    free(modelData);
    if (ret < 0)
    {
        RCLCPP_ERROR(this->get_logger(), "RKNN init failed with error code: %d", ret);
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "RKNN context initialized successfully");
    
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, sizeof(ioNum_));
    if (ret < 0)
    {
        RCLCPP_ERROR(this->get_logger(), "RKNN query failed with error code: %d", ret);
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "RKNN query successful: %d inputs, %d outputs", ioNum_.n_input, ioNum_.n_output);

    memset(rknnInputs_, 0, sizeof(rknnInputs_));
    rknnInputs_[0].index = 0;
    rknnInputs_[0].size = obsInput_.size() * sizeof(float);
    rknnInputs_[0].pass_through = false;
    rknnInputs_[0].type = RKNN_TENSOR_FLOAT32;
    rknnInputs_[0].fmt = RKNN_TENSOR_NHWC;

    memset(rknnOutputs_, 0, sizeof(rknnOutputs_));
    rknnOutputs_[0].want_float = true;
    
    RCLCPP_INFO(this->get_logger(), "RKNN policy loaded successfully");
    return true;
}

/**
 * @brief Update observation vector for RL policy
 */
void HighTorqueRLInference::updateObservation()
{
    if (observations_.size() != numSingleObs_)
    {
        observations_.resize(numSingleObs_);
    }

    step_ += 1.0 / stepsPeriod_;

    observations_[0] = currentState_ == STANDBY ? 1.0 : std::sin(2 * M_PI * step_);
    observations_[1] = currentState_ == STANDBY ? -1.0 : std::cos(2 * M_PI * step_);

    double cmdX = currentState_ == STANDBY ? 0.0 : command_[0];
    double cmdY = currentState_ == STANDBY ? 0.0 : command_[1];
    double cmdYaw = currentState_ == STANDBY ? 0.0 : command_[2];

    observations_[2] = cmdX * cmdLinVelScale_ * (cmdX < 0 ? 0.5 : 1.0);
    observations_[3] = cmdY * cmdLinVelScale_;
    observations_[4] = cmdYaw * cmdAngVelScale_;
    
    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
    observations_.segment(5, numActions_) = robotJointPositions_ * rbtLinPosScale_;
    observations_.segment(17, numActions_) = robotJointVelocities_ * rbtLinVelScale_;
    lk.unlock();

    observations_.segment(29, 3) = baseAngVel_ * rbtAngVelScale_;
    observations_.segment(32, 3) = eulerAngles_;

    for (int i = 0; i < numSingleObs_; ++i)
    {
        observations_[i] = std::clamp(observations_[i], -clipObs_, clipObs_);
    }

    histObs_.push_back(observations_);
    histObs_.pop_front();
}

/**
 * @brief Run RKNN inference to generate actions
 */
void HighTorqueRLInference::updateAction()
{
    for (int i = 0; i < frameStack_; ++i)
    {
        obsInput_.block(0, i * numSingleObs_, 1, numSingleObs_) = histObs_[i].transpose();
    }

    std::vector<float> inputData(obsInput_.size());
    Eigen::Index obsSize = obsInput_.size();
    for (Eigen::Index i = 0; i < obsSize; ++i)
    {
        inputData[i] = obsInput_(i);
    }

    rknnInputs_[0].buf = inputData.data();
    rknn_inputs_set(ctx_, ioNum_.n_input, rknnInputs_);
    rknn_run(ctx_, nullptr);
    rknn_outputs_get(ctx_, ioNum_.n_output, rknnOutputs_, nullptr);

    float* outputData = static_cast<float*>(rknnOutputs_[0].buf);
    for (int i = 0; i < numActions_; ++i)
    {
        action_[i] = std::clamp(outputData[i], clipActionsLower_[i], clipActionsUpper_[i]);
    }

    rknn_outputs_release(ctx_, ioNum_.n_output, rknnOutputs_);

}

void HighTorqueRLInference::quat2euler()
{
    double x = quat_.x();
    double y = quat_.y();
    double z = quat_.z();
    double w = quat_.w();

    double t0 = 2.0 * (w * x + y * z);
    double t1 = 1.0 - 2.0 * (x * x + y * y);
    double roll = std::atan2(t0, t1);

    double t2 = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
    double pitch = std::asin(t2);

    double t3 = 2.0 * (w * z + x * y);
    double t4 = 1.0 - 2.0 * (y * y + z * z);
    double yaw = std::atan2(t3, t4);

    eulerAngles_ << roll, pitch, yaw;
}

void HighTorqueRLInference::robotStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->position.size() < static_cast<size_t>(numActions_))
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Robot state size mismatch: expected %d, got %zu", numActions_, msg->position.size());
        return;
    }

    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
    for (int i = 0; i < numActions_; ++i)
    {
        robotJointPositions_[i] = msg->position[i];
        if (msg->velocity.size() >= static_cast<size_t>(numActions_))
        {
            robotJointVelocities_[i] = msg->velocity[i];
        }
    }

    if (!stateReceived_)
    {
        stateReceived_ = true;
        RCLCPP_INFO(this->get_logger(), "Robot state callback: first data received");
    }
}

void HighTorqueRLInference::motorStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->position.size() < static_cast<size_t>(numActions_))
    {
        return;
    }

    for (int i = 0; i < numActions_; ++i)
    {
        int policyIdx = actualToPolicyMap_[i];
        if (policyIdx >= 0 && policyIdx < numActions_)
        {
            motorJointPositions_[policyIdx] = msg->position[i] * motorDirection_[i];
            if (msg->velocity.size() >= static_cast<size_t>(numActions_))
            {
                motorJointVelocities_[policyIdx] = msg->velocity[i] * motorDirection_[i];
            }
        }
    }

    if (!stateReceived_)
    {
        stateReceived_ = true;
    }
}

void HighTorqueRLInference::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    quat_.x() = msg->orientation.x;
    quat_.y() = msg->orientation.y;
    quat_.z() = msg->orientation.z;
    quat_.w() = msg->orientation.w;
    quat2euler();
    baseAngVel_[0] = msg->angular_velocity.x;
    baseAngVel_[1] = msg->angular_velocity.y;
    baseAngVel_[2] = msg->angular_velocity.z;
    
    // 调试输出：每2秒打印一次 IMU 数据
    static auto lastImuDebugTime = this->now();
    if ((this->now() - lastImuDebugTime).seconds() > 2.0)
    {
        lastImuDebugTime = this->now();
        std::stringstream ss;
        ss << "IMU Data Received:" << std::endl;
        ss << "  Orientation (quaternion): x=" << quat_.x() << ", y=" << quat_.y() 
           << ", z=" << quat_.z() << ", w=" << quat_.w() << std::endl;
        ss << "  Euler Angles (rad): roll=" << eulerAngles_[0] 
           << ", pitch=" << eulerAngles_[1] << ", yaw=" << eulerAngles_[2] << std::endl;
        ss << "  Angular Velocity (rad/s): x=" << baseAngVel_[0] 
           << ", y=" << baseAngVel_[1] << ", z=" << baseAngVel_[2];
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }
    
    if (!imuReceived_)
    {
        imuReceived_ = true;
        RCLCPP_INFO(this->get_logger(), "✓ IMU callback: first data received");
    }
}

void HighTorqueRLInference::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    command_[0] = msg->linear.x;
    command_[1] = msg->linear.y;
    command_[2] = msg->angular.z;
}

void HighTorqueRLInference::joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(joyMutex_);
    joyMsg_ = *msg;
    joyReady_ = true;
}

void HighTorqueRLInference::run()
{
    rclcpp::Rate rate(rlCtrlFreq_);

    while (rclcpp::ok() && !quit_)
    {
        rclcpp::spin_some(this->get_node_base_interface());

        if (currentState_ != NOT_READY)
        {
            updateObservation();
            updateAction();
        }

        if (joyReady_.load())
        {
            std::lock_guard<std::mutex> lock(joyMutex_);
            
            int axis2, axis5, btn_start, btn_lb;
            this->get_parameter("joy_axis2", axis2);
            this->get_parameter("joy_axis5", axis5);
            this->get_parameter("joy_button_start", btn_start);
            this->get_parameter("joy_button_lb", btn_lb);

            bool ltPressed = (axis2 >= 0 && axis2 < (int)joyMsg_.axes.size()) && (std::abs(joyMsg_.axes[axis2]) > 0.8);
            bool rtPressed = (axis5 >= 0 && axis5 < (int)joyMsg_.axes.size()) && (std::abs(joyMsg_.axes[axis5]) > 0.8);
            bool startPressed = (btn_start >= 0 && btn_start < (int)joyMsg_.buttons.size()) && (joyMsg_.buttons[btn_start] == 1);
            bool lbPressed = (btn_lb >= 0 && btn_lb < (int)joyMsg_.buttons.size()) && (joyMsg_.buttons[btn_lb] == 1);

            bool triggerReset = ltPressed && rtPressed && startPressed;
            bool triggerToggle = ltPressed && rtPressed && lbPressed;

            auto now = this->now();
            double timeSinceLastTrigger = (now - lastTrigger_).seconds();

            // LT+RT+START: Enter RL mode (STANDBY) from NOT_READY
            if (triggerReset && timeSinceLastTrigger > 1.0)
            {
                if (currentState_ == NOT_READY)
                {
                    lastTrigger_ = now;

                    double reset_duration;
                    this->get_parameter("reset_duration", reset_duration);

                    auto preset = sensor_msgs::msg::JointState();
                    preset.header.frame_id = "zero";
                    preset.header.stamp = rclcpp::Time(static_cast<int64_t>(reset_duration * 1e9));

                    presetPub_->publish(preset);
                    auto sleep_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::duration<double>(reset_duration));
                    rclcpp::sleep_for(sleep_duration);
                    currentState_ = STANDBY;
                    
                    RCLCPP_INFO(this->get_logger(), "Entered STANDBY mode");
                }
            }

            if (triggerToggle && timeSinceLastTrigger > 1.0)
            {
                if (currentState_ == STANDBY || currentState_ == RUNNING)
                {
                    lastTrigger_ = now;

                    if (currentState_ == STANDBY)
                    {
                        currentState_ = RUNNING;
                        RCLCPP_INFO(this->get_logger(), "Entered RUNNING mode");
                    }
                    else if (currentState_ == RUNNING)
                    {
                        currentState_ = STANDBY;
                        RCLCPP_INFO(this->get_logger(), "Entered STANDBY mode");
                    }
                }
            }
        }

        if (currentState_ == NOT_READY)
        {
            rate.sleep();
            continue;
        }

        auto msg = sensor_msgs::msg::JointState();
        msg.header.stamp = rclcpp::Time(0);

        msg.position.resize(22);
        // Scale action based on state: RUNNING uses actionScale_, other states use 0.05
        double scale = (currentState_ == RUNNING) ? actionScale_ : 0.05;

        for (int i = 0; i < 12; ++i)
        {
            msg.position[i] = action_[i] * scale;
        }
        for (int i = 12; i < 22; ++i)
        {
            msg.position[i] = 0.0;
        }

        jointCmdPub_->publish(msg);
        rate.sleep();
    }
}

} // namespace hightorque_rl_inference_ros2

