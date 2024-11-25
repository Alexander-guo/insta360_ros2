#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the 'config' launch argument with default value 'config.yaml'
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='config.yaml',
        description='Path to the configuration file'
    )

    # Construct the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('insta360_ros_driver'),
        'config',
        LaunchConfiguration('config')
    ])

    # Define the compression_node
    compression_node = Node(
        package='insta360_ros_driver',
        executable='compression_node',  # Ensure this matches the executable name in setup.py
        name='compression_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        config_arg,
        compression_node
    ])
