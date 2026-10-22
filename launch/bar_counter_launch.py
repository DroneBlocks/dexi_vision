#!/usr/bin/env python3
"""
Launch file for the bar counter node

Usage:
    ros2 launch dexi_vision bar_counter_launch.py

    # Your own settings
    ros2 launch dexi_vision bar_counter_launch.py params:=/home/dexi/my_bar_counter.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('dexi_vision'),
                                  'config', 'bar_counter_params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params',
            default_value=default_params,
            description='Parameter file for the bar counter'
        ),
        Node(
            package='dexi_vision',
            executable='bar_counter_node.py',
            name='bar_counter_node',
            output='screen',
            parameters=[LaunchConfiguration('params')],
        ),
    ])
