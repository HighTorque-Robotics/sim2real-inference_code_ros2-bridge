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
#include <iomanip>
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
        numActions_ = config["num_actions"].as<int>();          // 实际控制的关节数
        policyDofs_ = config["policy_dofs"].as<int>(); // 策略输出维度
        numSingleObs_ = config["num_single_obs"].as<int>();
        clipObs_ = config["clip_obs"].as<double>();

        // Read policy name and build path
        std::string policyName = config["policy_name"].as<std::string>();
        policyPath_ = pkgPath + "/policy/" + policyName;
        RCLCPP_INFO(this->get_logger(), "Policy name: %s", policyName.c_str());

        // Read control frequency
        double dt = config["dt"].as<double>();
        int decimation = config["decimation"].as<int>();
        rlCtrlFreq_ = 1.0 / (dt * decimation);

        // Read scaling parameters
        cmdLinVelScale_ = config["cmd_lin_vel_scale"].as<double>();
        cmdAngVelScale_ = config["cmd_ang_vel_scale"].as<double>();
        rbtLinPosScale_ = config["rbt_lin_pos_scale"].as<double>();
        rbtLinVelScale_ = config["rbt_lin_vel_scale"].as<double>();
        rbtAngVelScale_ = config["rbt_ang_vel_scale"].as<double>();
        actionScale_ = config["action_scale"].as<double>();

        // Read action limits
        std::vector<double> clipLower = config["clip_actions_lower"].as<std::vector<double>>();
        std::vector<double> clipUpper = config["clip_actions_upper"].as<std::vector<double>>();

        // Read AMP-specific parameters (与 amp_pi_plus_20dof.yaml 对应)
        
        // 控制模式
        controlMode_ = config["control_mode"].as<std::string>();
        
        // 速度限制
        cmdVelXMin_ = config["cmd_vel_x_min"].as<double>();
        cmdVelYMin_ = config["cmd_vel_y_min"].as<double>();
        cmdVelYMax_ = config["cmd_vel_y_max"].as<double>();
        cmdVelYawMin_ = config["cmd_vel_yaw_min"].as<double>();
        cmdVelYawMax_ = config["cmd_vel_yaw_max"].as<double>();
        
        // Soccer/Fight 模式参数
        cmdVelFilterScaleSoccer_ = config["cmd_vel_filter_scale_soccer"].as<double>();
        cmdVelFilterScaleFight_ = config["cmd_vel_filter_scale_fight"].as<double>();
        cmdVelXMaxSoccerNormal_ = config["cmd_vel_x_max_soccer_normal"].as<double>();
        cmdVelXMaxSoccerBoost_ = config["cmd_vel_x_max_soccer_boost"].as<double>();
        cmdVelXMaxFight_ = config["cmd_vel_x_max_fight"].as<double>();
        
        // 兼容旧配置：单个速度限制和滤波参数（可选）
        if (config["cmd_vel_filter_scale"])
        {
            cmdVelFilterScale_ = config["cmd_vel_filter_scale"].as<double>();
        }
        
        if (config["cmd_vel_x_max"])
        {
            cmdVelXMax_ = config["cmd_vel_x_max"].as<double>();
        }
        
        // 动作滤波和缩放
        actionFilterScale_ = config["action_filter_scale"].as<double>();
        
        if (config["action_scales"])
        {
            actionScales_ = config["action_scales"].as<std::vector<double>>();
        }
        if (config["default_pose"])
        {
            defaultPose_ = config["default_pose"].as<std::vector<double>>();
        }

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
        
        // Read policy to control mapping (for extracting leg joints from 20dof policy output)
        if (config["policy_to_control_map"])
        {
            policyToControlMap_ = config["policy_to_control_map"].as<std::vector<int>>();
        }

        // Model type (can be overridden from parameters)
        this->get_parameter("model_type", modelType_);

        RCLCPP_INFO(this->get_logger(), "YAML config loaded successfully:");
        RCLCPP_INFO(this->get_logger(), "  num_actions: %d", numActions_);
        RCLCPP_INFO(this->get_logger(), "  num_single_obs: %d", numSingleObs_);
        RCLCPP_INFO(this->get_logger(), "  rl_ctrl_freq: %.1f Hz", rlCtrlFreq_);
        RCLCPP_INFO(this->get_logger(), "  policy_path: %s", policyPath_.c_str());
        RCLCPP_INFO(this->get_logger(), "  action_scale: %.2f", actionScale_);

        // Action limits 应该与策略输出维度一致
        clipActionsLower_.resize(policyDofs_);
        clipActionsUpper_.resize(policyDofs_);
        for (int i = 0; i < policyDofs_ && i < (int)clipLower.size(); ++i)
        {
            clipActionsLower_[i] = static_cast<float>(clipLower[i]);
            clipActionsUpper_[i] = static_cast<float>(clipUpper[i]);
        }
    }
    catch (const YAML::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "YAML parsing error: %s", e.what());
        RCLCPP_FATAL(this->get_logger(), "Failed to load required configuration parameters. Please check your YAML file.");
        throw;
    }

    // Initialize Eigen vectors
    // 注意：所有关节相关的向量都应该是 policyDofs_ 维度（策略输出维度）
    robotJointPositions_ = Eigen::VectorXd::Zero(policyDofs_);
    robotJointVelocities_ = Eigen::VectorXd::Zero(policyDofs_);
    motorJointPositions_ = Eigen::VectorXd::Zero(policyDofs_);
    motorJointVelocities_ = Eigen::VectorXd::Zero(policyDofs_);
    eulerAngles_ = Eigen::Vector3d::Zero();
    baseAngVel_ = Eigen::Vector3d::Zero();
    command_ = Eigen::Vector3d::Zero();
    action_ = Eigen::VectorXd::Zero(policyDofs_);

    observations_ = Eigen::VectorXd::Zero(numSingleObs_);
    // 初始化历史观测缓冲区（5帧），与 PolicyBase 保持一致
    for (int i = 0; i < 5; ++i)
    {
        histObs_.emplace_back(Eigen::VectorXd::Zero(numSingleObs_));
    }
    obsInput_ = Eigen::MatrixXd::Zero(1, numSingleObs_ * 5);

    quat_ = Eigen::Quaterniond::Identity();

    // Check vector sizes (都应该与 policyDofs_ 对齐)
    if (urdfOffset_.size() != static_cast<size_t>(policyDofs_))
    {
        urdfOffset_.assign(policyDofs_, 0.0);
        RCLCPP_WARN(this->get_logger(), "urdf_offset resized to %d", policyDofs_);
    }
    if (motorDirection_.size() != static_cast<size_t>(policyDofs_))
    {
        motorDirection_.assign(policyDofs_, 1);
        RCLCPP_WARN(this->get_logger(), "motor_direction resized to %d", policyDofs_);
    }
    if (actualToPolicyMap_.size() != static_cast<size_t>(policyDofs_))
    {
        std::vector<int> defaultMap(policyDofs_);
        for (int i = 0; i < policyDofs_; ++i)
            defaultMap[i] = i;
        actualToPolicyMap_ = defaultMap;
        RCLCPP_WARN(this->get_logger(), "actual_to_policy_map resized to %d", policyDofs_);
    }
    
    // 验证 policy_to_control_map 的大小
    // 策略输出 policyDofs_ (20)个关节，需要重新排序到 numActions_ (22)的前20个位置
    if (policyToControlMap_.empty())
    {
        RCLCPP_ERROR(this->get_logger(), 
            "policy_to_control_map is empty! This is required to map policy output (interleaved order) to control output (grouped order)");
        RCLCPP_ERROR(this->get_logger(), 
            "Please add policy_to_control_map in config file");
    }
    else if (policyToControlMap_.size() != static_cast<size_t>(policyDofs_))
    {
        RCLCPP_ERROR(this->get_logger(), 
            "policy_to_control_map size (%zu) != policy_dofs (%d)",
            policyToControlMap_.size(), policyDofs_);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), 
            "policy_to_control_map loaded: %d policy joints -> %d control joints (first %d positions)",
            policyDofs_, numActions_, policyDofs_);
    }

    this->get_parameter("steps_period", stepsPeriod_);
    step_ = 0.0;

    // Initialize AMP-specific members（都应该是 policyDofs_ 维度）
    lastAction_ = Eigen::VectorXd::Zero(policyDofs_);
    lastCmdVel_ = Eigen::Vector3d::Zero();
    
    // 确保动作缩放和默认姿态的大小正确（应该是 policyDofs_ = numActions_ = 20）
    if (actionScales_.size() < static_cast<size_t>(policyDofs_))
    {
        actionScales_.resize(policyDofs_, 1.0);
        RCLCPP_WARN(this->get_logger(), "Action scales resized to %d, using default 1.0 for missing values", policyDofs_);
    }
    
    if (defaultPose_.size() < static_cast<size_t>(policyDofs_))
    {
        defaultPose_.resize(policyDofs_, 0.0);
        RCLCPP_WARN(this->get_logger(), "Default pose resized to %d, using default 0.0 for missing values", policyDofs_);
    }
    
    RCLCPP_INFO(this->get_logger(), "AMP Policy parameters:");
    RCLCPP_INFO(this->get_logger(), "  control_mode: %s", controlMode_.c_str());
    if (controlMode_ == "Soccer")
    {
        RCLCPP_INFO(this->get_logger(), "  Soccer mode - filter_scale: %.2f, normal_speed: %.2f, boost_speed: %.2f",
                    cmdVelFilterScaleSoccer_, cmdVelXMaxSoccerNormal_, cmdVelXMaxSoccerBoost_);
    }
    else if (controlMode_ == "Fight")
    {
        RCLCPP_INFO(this->get_logger(), "  Fight mode - filter_scale: %.2f, max_speed: %.2f",
                    cmdVelFilterScaleFight_, cmdVelXMaxFight_);
    }
    RCLCPP_INFO(this->get_logger(), "  action_filter_scale: %.2f", actionFilterScale_);
    RCLCPP_INFO(this->get_logger(), "  cmd_vel limits: x[%.2f, --], y[%.2f, %.2f], yaw[%.2f, %.2f]",
                cmdVelXMin_, cmdVelYMin_, cmdVelYMax_, cmdVelYawMin_, cmdVelYawMax_);
    
    currentState_ = RUNNING;
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
    std::string policyName = policyPath_.substr(policyPath_.find_last_of("/") + 1);
    RCLCPP_INFO(this->get_logger(), "Policy name: %s", policyName.c_str());
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
 * @brief Update observation vector for AMP policy
 * 观测顺序（完整20dof）：base_ang_vel(3) + projected_gravity(3) + commands(3) + dof_pos(20) + dof_vel(20) + actions(20) = 69维
 * 与 AMPPolicy::updateObservation 完全对齐
 */
void HighTorqueRLInference::updateObservation()
{
    if (observations_.size() != numSingleObs_)
    {
        observations_.resize(numSingleObs_);
    }

    // ========== 测试模式：使用固定观测数据 ==========
    static const double fixedObsData[] = {
        0.1, -0.2, 0.3,    // base_ang_vel (3)
        -0.4, 0.5, -0.6,   // projected_gravity (3)
        0.2, -0.1, 0.0,    // cmd_vel (3)
        0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8, 0.9, -0.9, 1.0,  // dof_pos (20)
        -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8, -0.9, 0.9, -1.0, 1.0,  // dof_vel (20)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   // last_action (20) - 全部为0
    };
    const int fixedObsSize = sizeof(fixedObsData) / sizeof(fixedObsData[0]);
    int copySize = std::min((int)observations_.size(), fixedObsSize);
    for (int i = 0; i < copySize; ++i)
    {
        observations_[i] = fixedObsData[i];
    }
    for (int i = copySize; i < (int)observations_.size(); ++i)
    {
        observations_[i] = 0.0;
    }
    // 注意：历史缓冲区更新在 updateAction() 中进行，与 PolicyBase::getFinalObsInput() 保持一致
    return;
    // ========== 测试模式结束 ==========

    /*
    // ========== 真实数据计算（已注释，测试完成后恢复） ==========
    int idx = 0;
    
    // 1. base_ang_vel (3维) - 基础角速度（与 amp_policy.cpp 第127行对应）
    observations_.segment(idx, 3) = baseAngVel_ * rbtAngVelScale_;
    idx += 3;
    
    // 2. projected_gravity (3维) - 重力投影（与 amp_policy.cpp 第130-133行对应）
    Eigen::Vector3d gravityVec(0, 0, -1);
    Eigen::Vector4d quatVec(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    observations_.segment(idx, 3) = rotateVectorByQuatVec4(quatVec, gravityVec);
    idx += 3;
    
    // 3. commands (3维) - 速度命令（带滤波和限制，与 amp_policy.cpp 第136-168行对应）
    double cmdX = currentState_ == STANDBY ? 0.0 : command_[0];
    double cmdY = currentState_ == STANDBY ? 0.0 : command_[1];
    double cmdYaw = currentState_ == STANDBY ? 0.0 : command_[2];
    
    // 获取LT按键状态（与 amp_policy.cpp 第137行一致）
    bool ltPressed = false;
    if (joyReady_.load())
    {
        std::lock_guard<std::mutex> lock(joyMutex_);
        int axis2;
        this->get_parameter("joy_axis2", axis2);
        ltPressed = (axis2 >= 0 && axis2 < (int)joyMsg_.axes.size()) && (std::abs(joyMsg_.axes[axis2]) > 0.8);
    }
    
    // 根据控制模式确定x速度上限（与 amp_policy.cpp 第139-153行一致）
    double maxVelX, dynamicCmdVelFilterScale;
    if (controlMode_ == "Soccer")
    {
        // Soccer模式：按LT加速
        maxVelX = ltPressed ? cmdVelXMaxSoccerBoost_ : cmdVelXMaxSoccerNormal_;
        dynamicCmdVelFilterScale = ltPressed ? 1.0 : cmdVelFilterScaleSoccer_; // 按下直接氮气加速
    }
    else if (controlMode_ == "Fight")
    {
        // Fight模式：固定速度，不响应LT
        maxVelX = cmdVelXMaxFight_;
        dynamicCmdVelFilterScale = cmdVelFilterScaleFight_;
    }
    else
    {
        // 默认模式（兼容旧配置）
        maxVelX = cmdVelXMax_;
        dynamicCmdVelFilterScale = cmdVelFilterScale_;
    }
    
    // 应用速度限制（与 amp_policy.cpp 第156-159行一致）
    cmdX = std::clamp(cmdX, cmdVelXMin_, maxVelX);
    cmdY = std::clamp(cmdY, cmdVelYMin_, cmdVelYMax_);
    cmdYaw = std::clamp(cmdYaw, cmdVelYawMin_, cmdVelYawMax_);
    
    // 应用滤波（与 amp_policy.cpp 第163-165行一致）
    observations_[idx + 0] = (cmdX * dynamicCmdVelFilterScale + lastCmdVel_[0] * (1 - dynamicCmdVelFilterScale)) * cmdLinVelScale_;
    observations_[idx + 1] = (cmdY * dynamicCmdVelFilterScale + lastCmdVel_[1] * (1 - dynamicCmdVelFilterScale)) * cmdLinVelScale_;
    observations_[idx + 2] = (cmdYaw * dynamicCmdVelFilterScale + lastCmdVel_[2] * (1 - dynamicCmdVelFilterScale)) * cmdAngVelScale_;
    idx += 3;
    
    // 保存当前命令速度供下次滤波使用（与 amp_policy.cpp 第168行一致）
    lastCmdVel_ << cmdX, cmdY, cmdYaw;
    
    // 4. dof_pos (policyDofs_维=20) - 关节位置（与 amp_policy.cpp 第171行对应）
    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
    observations_.segment(idx, policyDofs_) = robotJointPositions_.head(policyDofs_) * rbtLinPosScale_;
    idx += policyDofs_;
    
    // 5. dof_vel (policyDofs_维=20) - 关节速度（与 amp_policy.cpp 第174行对应）
    observations_.segment(idx, policyDofs_) = robotJointVelocities_.head(policyDofs_) * rbtLinVelScale_;
    idx += policyDofs_;
    lk.unlock();
    
    // 6. actions (policyDofs_维=20) - 上次动作输出（与 amp_policy.cpp 第177行对应）
    observations_.segment(idx, policyDofs_) = lastAction_.head(policyDofs_);
    idx += policyDofs_;

    // 裁剪观测值（与 amp_policy.cpp 第181行对应）
    for (int i = 0; i < numSingleObs_; ++i)
    {
        observations_[i] = std::clamp(observations_[i], -clipObs_, clipObs_);
    }
    // ========== 真实数据计算结束 ==========
    */
}

/**
 * @brief Run RKNN inference to generate actions
 * 直接输出完整20dof，不需要映射
 * 注意：模型需要5帧堆叠输入（345 = 69 * 5），与 PolicyBase::getFinalObsInput() 逻辑一致
 */
void HighTorqueRLInference::updateAction()
{
    // ========== 测试模式：使用固定的345维观测（5帧×69维） ==========
    static bool fixed345ObsInitialized = false;
    static Eigen::MatrixXd fixed345Obs;
    if (!fixed345ObsInitialized)
    {
        // 定义固定的69维单帧观测数据（与 sim2real.cpp 完全一致）
        static const double singleFrameObs[] = {
            0.1, -0.2, 0.3,    // base_ang_vel (3)
            -0.4, 0.5, -0.6,   // projected_gravity (3)
            0.2, -0.1, 0.0,    // cmd_vel (3)
            0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8, 0.9, -0.9, 1.0,  // dof_pos (20)
            -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8, -0.9, 0.9, -1.0, 1.0,  // dof_vel (20)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   // last_action (20)
        };
        const int singleFrameSize = sizeof(singleFrameObs) / sizeof(singleFrameObs[0]);
        
        // 构造345维观测（5帧堆叠，每帧69维）
        fixed345Obs = Eigen::MatrixXd::Zero(1, 345);
        for (int frame = 0; frame < 5; ++frame)
        {
            for (int i = 0; i < singleFrameSize && i < 69; ++i)
            {
                fixed345Obs(0, frame * 69 + i) = singleFrameObs[i];
            }
        }
        fixed345ObsInitialized = true;
    }
    obsInput_ = fixed345Obs;  // 直接使用固定的345维观测
    // ========== 测试模式结束 ==========
    
    /*
    // ========== 真实数据堆叠（已注释，测试完成后恢复） ==========
    // 与 PolicyBase::getFinalObsInput() 逻辑一致：
    // 1. 将当前观测添加到历史缓冲区
    histObs_.push_back(observations_);
    // 2. 保持缓冲区大小为5帧
    while (histObs_.size() > 5)
    {
        histObs_.pop_front();
    }
    // 3. 将历史缓冲区中的5帧数据堆叠成最终输入
    for (int i = 0; i < 5; ++i)
    {
        obsInput_.block(0, i * numSingleObs_, 1, numSingleObs_) = histObs_[i].transpose();
    }
    // ========== 真实数据堆叠结束 ==========
    */

    std::vector<float> inputData(obsInput_.size());
    Eigen::Index obsSize = obsInput_.size();
    for (Eigen::Index i = 0; i < obsSize; ++i)
    {
        inputData[i] = obsInput_(i);
    }

    // 打印传入推理引擎的完整输入
    std::stringstream inputStr;
    inputStr << "Input(" << inputData.size() << "): ";
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputStr << std::fixed << std::setprecision(6) << inputData[i];
        if (i < inputData.size() - 1)
            inputStr << ", ";
    }
    RCLCPP_INFO(this->get_logger(), "%s", inputStr.str().c_str());

    rknnInputs_[0].buf = inputData.data();
    rknn_inputs_set(ctx_, ioNum_.n_input, rknnInputs_);
    rknn_run(ctx_, nullptr);
    rknn_outputs_get(ctx_, ioNum_.n_output, rknnOutputs_, nullptr);

    float* outputData = static_cast<float*>(rknnOutputs_[0].buf);
    
    // 打印原始推理输出（未裁剪）
    std::stringstream rawOutputStr;
    rawOutputStr << "RawOutput: ";
    for (int i = 0; i < policyDofs_; ++i)
    {
        rawOutputStr << std::fixed << std::setprecision(6) << outputData[i];
        if (i < policyDofs_ - 1)
            rawOutputStr << ", ";
    }
    RCLCPP_INFO(this->get_logger(), "%s", rawOutputStr.str().c_str());
    
    // 直接使用策略输出（完整20dof）
    for (int i = 0; i < policyDofs_; ++i)
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

Eigen::Vector3d HighTorqueRLInference::rotateVectorByQuatVec4(const Eigen::Vector4d& q, const Eigen::Vector3d& v)
{
    double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    Eigen::Vector3d qVec(qx, qy, qz);

    Eigen::Vector3d term1 = v * (2.0 * qw * qw - 1.0);
    Eigen::Vector3d term2 = qVec.cross(v) * qw * 2.0;
    Eigen::Vector3d term3 = qVec * qVec.dot(v) * 2.0;

    return term1 - term2 + term3;
}

void HighTorqueRLInference::robotStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->position.size() < static_cast<size_t>(policyDofs_))
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Robot state size mismatch: expected %d, got %zu", policyDofs_, msg->position.size());
        return;
    }

    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
    for (int i = 0; i < policyDofs_; ++i)
    {
        robotJointPositions_[i] = msg->position[i];
        if (msg->velocity.size() >= static_cast<size_t>(policyDofs_))
        {
            robotJointVelocities_[i] = msg->velocity[i];
        }
    }

    if (!stateReceived_)
    {
        stateReceived_ = true;
        RCLCPP_INFO(this->get_logger(), "Robot state callback: first data received (20 dof)");
    }
}

void HighTorqueRLInference::motorStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->position.size() < static_cast<size_t>(policyDofs_))
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Motor state size mismatch: expected %d, got %zu", policyDofs_, msg->position.size());
        return;
    }

    std::unique_lock<std::shared_timed_mutex> lk(mutex_);
    for (int i = 0; i < policyDofs_; ++i)
    {
        int policyIdx = actualToPolicyMap_[i];
        if (policyIdx >= 0 && policyIdx < policyDofs_)
        {
            motorJointPositions_[policyIdx] = msg->position[i] * motorDirection_[i];
            if (msg->velocity.size() >= static_cast<size_t>(policyDofs_))
            {
                motorJointVelocities_[policyIdx] = msg->velocity[i] * motorDirection_[i];
            }
        }
    }
    lk.unlock();

    if (!stateReceived_)
    {
        stateReceived_ = true;
        RCLCPP_INFO(this->get_logger(), "Motor state callback: first data received (20 dof)");
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

        updateObservation();
        updateAction();

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

        auto msg = sensor_msgs::msg::JointState();
        msg.header.stamp = rclcpp::Time(0);

        // 发布22dof到话题（20个控制关节 + 2个头部关节固定为0）
        msg.position.resize(numActions_);  // numActions_ = 22
        
        // ========== AMP动作后处理：完全对齐 AMPPolicy::postModifyOutput (amp_policy.cpp 第187-201行) ==========
        
        // 步骤1：保存当前原始动作（用于下次滤波，与 amp_policy.cpp 第189行对应）
        Eigen::VectorXd currentRawAction = action_;
        
        // 步骤2：动作后处理 - 滤波 + 缩放 + 默认姿态（与 amp_policy.cpp 第192-195行对应）
        Eigen::VectorXd processedAction(policyDofs_);
        for (int i = 0; i < policyDofs_; ++i)
        {
            // 公式：(action[i] * filterScale + lastAction[i] * (1 - filterScale)) * actionScales[i] + defaultPose[i]
            processedAction[i] = (action_[i] * actionFilterScale_ + lastAction_[i] * (1 - actionFilterScale_)) 
                               * actionScales_[i] + defaultPose_[i];
        }
        
        // 步骤3：根据状态调整全局缩放（与 sim2real.cpp 第1165-1167行对应）
        // RUNNING 或 PRE_POLICY_CHANGE 使用 actionScale_，其他状态（STANDBY等）使用 0.05
        double stateScale = (currentState_ == RUNNING) ? actionScale_ : 0.05;
        
        // 步骤4：使用 policy_to_control_map 重新排序输出
        // 策略输出(交错顺序) -> PD配置(分组顺序)
        // 策略输出20个关节，通过映射表重新排列到PD配置期望的顺序
        if (!policyToControlMap_.empty() && policyToControlMap_.size() == static_cast<size_t>(policyDofs_))
        {
            for (int i = 0; i < policyDofs_; ++i)
            {
                int targetIdx = policyToControlMap_[i];
                msg.position[targetIdx] = processedAction[i] * stateScale;
            }
        }
        else
        {
            // 如果映射表不可用，直接按顺序填充（不推荐）
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "policy_to_control_map not available, using direct mapping (may cause wrong joint order)");
            for (int i = 0; i < policyDofs_; ++i)
            {
                msg.position[i] = processedAction[i] * stateScale;
            }
        }
        
        // 最后2个关节（头部）固定为0
        msg.position[20] = 0.0;  // head_yaw_joint
        msg.position[21] = 0.0;  // head_pitch_joint
        
        // 步骤5：更新 lastAction_ 为原始动作（用于下次滤波和观测，与 amp_policy.cpp 第189行对应）
        lastAction_ = currentRawAction;

        jointCmdPub_->publish(msg);
        rate.sleep();
    }
}

} // namespace hightorque_rl_inference_ros2

