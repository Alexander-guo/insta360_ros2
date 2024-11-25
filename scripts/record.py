#!/usr/bin/env python3

import os
import subprocess
import sys
import datetime

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

class RecordNode(Node):
    def __init__(self):
        super().__init__('record')

        # Declare parameters with default values
        self.declare_parameter('bag_type', 'compressed')
        self.declare_parameter('raw_bag_folder', '/home/bag/raw')
        self.declare_parameter('compressed_bag_folder', '/home/bag/compressed')
        self.declare_parameter('undistorted_bag_folder', '/home/bag/undistorted')
        self.declare_parameter('image_save_directory', '/home/saved_images')

        # Retrieve parameters
        self.bag_type = self.get_parameter('bag_type').get_parameter_value().string_value

        self.raw_bag_folder = self.get_parameter('raw_bag_folder').get_parameter_value().string_value
        self.compressed_bag_folder = self.get_parameter('compressed_bag_folder').get_parameter_value().string_value
        self.undistorted_bag_folder = self.get_parameter('undistorted_bag_folder').get_parameter_value().string_value
        self.image_save_directory = self.get_parameter('image_save_directory').get_parameter_value().string_value

        # Get current timestamp
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Determine output directory and topics based on bag_type
        if self.bag_type == 'raw':
            output_dir = os.path.join(self.raw_bag_folder, f"{current_time}_raw")
            topics = ['/insta_image_yuv']
        elif self.bag_type == 'compressed':
            output_dir = os.path.join(self.compressed_bag_folder, f"{current_time}_compressed")
            topics = ['/back_camera_image/compressed', '/front_camera_image/compressed']
        elif self.bag_type == 'undistorted':
            output_dir = os.path.join(self.undistorted_bag_folder, f"{current_time}_undistorted")
            topics = ['/back_camera_image/compressed', '/front_camera_image/compressed']
        else:
            self.get_logger().error('Invalid `bag_type` parameter. Valid options are: raw, compressed, undistorted.')
            rclpy.shutdown()
            return

        # Ensure the image_save_directory exists
        if not os.path.exists(self.image_save_directory):
            try:
                os.makedirs(self.image_save_directory)
                self.get_logger().info(f'Created image save directory: {self.image_save_directory}')
            except OSError as e:
                self.get_logger().error(f'Failed to create image save directory {self.image_save_directory}: {e}')
                rclpy.shutdown()
                return

        # Ensure the bag output directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.get_logger().info(f'Created bag output directory: {output_dir}')
            except OSError as e:
                self.get_logger().error(f'Failed to create bag output directory {output_dir}: {e}')
                rclpy.shutdown()
                return
        else:
            self.get_logger().warn(f'Bag output directory {output_dir} already exists.')

        self.get_logger().warn(f"Recording to: {output_dir}")

        # Build the ros2 bag record command
        command = ['ros2', 'bag', 'record', '-o', output_dir] + topics

        # Start the recording subprocess
        try:
            self.process = subprocess.Popen(command)
            self.get_logger().info(f"Started ros2 bag record with command: {' '.join(command)}")
        except Exception as e:
            self.get_logger().error(f"Failed to start ros2 bag record: {e}")
            rclpy.shutdown()
            return

    def destroy_node(self):
        self.get_logger().warn('Stopping recording...')
        if hasattr(self, 'process') and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.get_logger().warn('Recording stopped gracefully.')
            except subprocess.TimeoutExpired:
                self.get_logger().error('Recording process did not terminate in time. Killing process.')
                self.process.kill()
                self.process.wait()
            except Exception as e:
                self.get_logger().error(f'Error while terminating recording process: {e}')
        else:
            self.get_logger().info('Recording process already terminated.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RecordNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received. Shutting down node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
