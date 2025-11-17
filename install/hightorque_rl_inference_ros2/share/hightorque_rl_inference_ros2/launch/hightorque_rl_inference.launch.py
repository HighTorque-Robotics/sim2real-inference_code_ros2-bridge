"""
Launch file for HighTorque RL Inference ROS2 node
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('hightorque_rl_inference_ros2')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_dir, 'config', 'config_example.yaml'),
        description='Path to configuration YAML file'
    )
    
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='pi_plus',
        description='Robot model type'
    )
    
    joy_topic_arg = DeclareLaunchArgument(
        'joy_topic',
        default_value='/joy',
        description='Joystick topic name'
    )
    
    steps_period_arg = DeclareLaunchArgument(
        'steps_period',
        default_value='60.0',
        description='Steps period for gait phase'
    )
    
    reset_duration_arg = DeclareLaunchArgument(
        'reset_duration',
        default_value='2.0',
        description='Duration for reset/preset transitions'
    )
    
    # Create node
    inference_node = Node(
        package='hightorque_rl_inference_ros2',
        executable='hightorque_rl_inference_ros2_node',
        name='hightorque_rl_inference_ros2',
        output='screen',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'model_type': LaunchConfiguration('model_type'),
            'joy_topic': LaunchConfiguration('joy_topic'),
            'steps_period': LaunchConfiguration('steps_period'),
            'reset_duration': LaunchConfiguration('reset_duration'),
            'joy_axis2': 2,
            'joy_axis5': 5,
            'joy_button_start': 7,
            'joy_button_lb': 4,
        }],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        model_type_arg,
        joy_topic_arg,
        steps_period_arg,
        reset_duration_arg,
        inference_node
    ])

