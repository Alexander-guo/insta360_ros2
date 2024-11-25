#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node


def verify_directories():
    rclpy.init()
    node = Node('verify_directories_node')

    default_dir = "/home/bag"

    node.declare_parameter('raw_bag_folder', os.path.join(default_dir, 'raw'))
    node.declare_parameter('compressed_bag_folder', os.path.join(default_dir, 'compressed'))
    node.declare_parameter('undistorted_bag_folder', os.path.join(default_dir, 'undistorted'))

    raw_bag_folder = node.get_parameter('raw_bag_folder').value
    compressed_bag_folder = node.get_parameter('compressed_bag_folder').value
    undistorted_bag_folder = node.get_parameter('undistorted_bag_folder').value

    folders = [raw_bag_folder, compressed_bag_folder, undistorted_bag_folder]

    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                node.get_logger().info(f"Created folder {folder}")
            except OSError as e:
                node.get_logger().error(f"Failed to create folder {folder}: {e}")
        else:
            node.get_logger().info(f"Folder already exists: {folder}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    verify_directories()
