#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    bag_type_arg = DeclareLaunchArgument(
        'bag_type',
        default_value='compressed',
        description='Type of bag to record: raw, compressed, or undistorted'
    )
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='config.yaml',
        description='Path to the configuration file'
    )

    # Define the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('insta360_ros_driver'),
        'config',
        LaunchConfiguration('config')
    ])

    # Define the directory_verification node
    directory_verification_node = Node(
        package='insta360_ros_driver',
        executable='directory_verification',
        name='directory_verification',
        output='screen',
        parameters=[config_file]
    )

    # Define the record node
    record_node = Node(
        package='insta360_ros_driver',
        executable='record',
        name='record',
        output='screen',
        parameters=[
            config_file,
            {'bag_type': LaunchConfiguration('bag_type')}
        ]
    )

    return LaunchDescription([
        bag_type_arg,
        config_arg,
        directory_verification_node,
        record_node
    ])
