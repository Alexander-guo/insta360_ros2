#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    undistort_arg = DeclareLaunchArgument(
        'undistort',
        default_value='false',
        description='Enable undistortion'
    )
    get_images_arg = DeclareLaunchArgument(
        'get_images',
        default_value='false',
        description='Enable image retrieval'
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
        name='bringup',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('insta360_ros_driver'),
                'config',
                LaunchConfiguration('config')
            ])
        ],
        output='screen'
    )

    live_processing_node = Node(
        package='insta360_ros_driver',
        executable='live_processing.py',
        name='live_processing',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('insta360_ros_driver'),
                'config',
                LaunchConfiguration('config')
            ])
        ],
        output='screen'
    )

    # Define the get_images node
    get_images_node = Node(
        package='insta360_ros_driver',
        executable='get_images.py',
        name='get_images',
        output='screen',
        parameters=[
            {'topic': '/back_camera_image/compressed'}
        ],
        condition=IfCondition(LaunchConfiguration('get_images'))
    )

    # Create and populate the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(undistort_arg)
    ld.add_action(get_images_arg)
    ld.add_action(config_arg)

    # Add nodes to the launch description
    ld.add_action(bringup_node)
    ld.add_action(live_processing_node)
    ld.add_action(get_images_node)

    return ld
