#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    undistort_arg = DeclareLaunchArgument(
        'undistort',
        default_value='false',
        description='Enable undistortion'
    )

    equirectangular_arg = DeclareLaunchArgument(
        'equirectangular',
        default_value='true',
        description='Enable equirectangular projection'
    )

    config_arg = DeclareLaunchArgument(
        'config',
        default_value='config.yaml',
        description='Path to the configuration file'
    )

    # Define the bringup node with parameters
    bringup_node = Node(
        package='insta360_ros_driver',
        executable='insta360_ros_driver',
        name='insta360_bringup',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('insta360_ros_driver'),
                'config',
                LaunchConfiguration('config')
            ])
        ],
        output='screen'
    )

    imu_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('insta360_ros_driver'),
                'config',
                'imu_filter.yaml'
            ])
        ]
    )

    equirectangular_node = Node(
        package='insta360_ros_driver',
        executable='equirectangular.py',
        name='equirectangular_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('equirectangular')),
        # arguments=['--gpu']
    )


    ld = LaunchDescription()

    ld.add_action(undistort_arg)
    ld.add_action(config_arg)
    ld.add_action(equirectangular_arg)
    ld.add_action(bringup_node)
    ld.add_action(imu_node)
    ld.add_action(equirectangular_node)

    return ld
