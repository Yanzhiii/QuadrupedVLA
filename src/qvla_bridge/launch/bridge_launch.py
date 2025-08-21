#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution


def generate_launch_description():
    # Launch args
    args = [
        DeclareLaunchArgument('model_type', default_value='navila', description='Model type'),
        DeclareLaunchArgument('debug_enabled', default_value='true', description='Enable debug'),
        DeclareLaunchArgument('image_buffer_size', default_value='32', description='Image buffer size'),
        DeclareLaunchArgument('target_frames', default_value='8', description='Sequence frames'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw', description='Camera topic'),
        DeclareLaunchArgument('base_linear_speed', default_value='0.5', description='Linear speed'),
        DeclareLaunchArgument('base_angular_speed', default_value='0.8', description='Angular speed'),
        DeclareLaunchArgument('move_duration_s', default_value='1.5', description='Move duration'),
        DeclareLaunchArgument('turn_duration_s', default_value='1.2', description='Turn duration'),
    ]

    # Use ExecuteProcess to run python -m directly (no libexec needed)
    cmd = [
        TextSubstitution(text='python3'),
        TextSubstitution(text='-m'), TextSubstitution(text='qvla_bridge.bridge_node'),
    ]

    bridge_proc = ExecuteProcess(cmd=cmd, output='screen')

    return LaunchDescription(args + [bridge_proc])