#!/usr/bin/env python3
"""
最小 Bridge 启动（中英注释）
Minimal Bridge launch (CN/EN)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution


def generate_launch_description():
    # 参数 / Launch args
    args = [
        DeclareLaunchArgument('model_type', default_value='navila', description='模型类型/Model type'),
        DeclareLaunchArgument('debug_enabled', default_value='true', description='启用调试/Enable debug'),
        DeclareLaunchArgument('image_buffer_size', default_value='32', description='图像缓冲/Image buffer size'),
        DeclareLaunchArgument('target_frames', default_value='8', description='序列帧数/Sequence frames'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw', description='相机话题/Camera topic'),
        DeclareLaunchArgument('base_linear_speed', default_value='0.5', description='线速度/Linear speed'),
        DeclareLaunchArgument('base_angular_speed', default_value='0.8', description='角速度/Angular speed'),
        DeclareLaunchArgument('move_duration_s', default_value='1.5', description='前进时长/Move duration'),
        DeclareLaunchArgument('turn_duration_s', default_value='1.2', description='转向时长/Turn duration'),
    ]

    # 直接用 ExecuteProcess 调用 python -m（不依赖包内可执行）
    # Use ExecuteProcess to run python -m directly (no libexec needed)
    # 简化：使用默认参数（可运行后用 ros2 param set 动态调整）
    cmd = [
        TextSubstitution(text='python3'),
        TextSubstitution(text='-m'), TextSubstitution(text='qvla_bridge.bridge_node'),
    ]

    bridge_proc = ExecuteProcess(cmd=cmd, output='screen')

    return LaunchDescription(args + [bridge_proc])