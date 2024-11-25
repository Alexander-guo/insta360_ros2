#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the 'bag_file' launch argument with default value '~/bag/undistorted/record.bag'
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value=PathJoinSubstitution([
            EnvironmentVariable('HOME'),
            'bag',
            'undistorted',
            'record.bag'
        ]),
        description='Path to the bag file to play'
    )

    # Define the rviz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('insta360_ros_driver'),
            'config',
            'preview.rviz'
        ])]
    )

    # Define the rosbag play process
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '--clock', LaunchConfiguration('bag_file')],
        output='screen'
    )

    return LaunchDescription([
        bag_file_arg,
        rviz_node,
        rosbag_play
    ])
