#!/usr/bin/env python3
import os
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from sensor_msgs.msg import CompressedImage

# Removed import for get_package_share_directory as it's no longer needed

class GetImagesNode(Node):
    def __init__(self):
        super().__init__('image_capture')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Declare and get parameters
        self.declare_parameter('topic', '/back_camera_image/compressed')
        self.declare_parameter('image_save_directory', '/home/saved_images')

        # Retrieve 'topic' parameter
        try:
            self.topic = self.get_parameter('topic').get_parameter_value().string_value
        except ParameterNotDeclaredException:
            self.get_logger().warn(
                'Parameter "topic" not declared. Using default "/back_camera_image/compressed".')
            self.topic = '/back_camera_image/compressed'

        # Retrieve 'image_save_directory' parameter
        try:
            self.img_path = self.get_parameter('image_save_directory').get_parameter_value().string_value
        except ParameterNotDeclaredException:
            self.get_logger().warn(
                'Parameter "image_save_directory" not declared. Using default "/home/saved_images".')
            self.img_path = '/home/saved_images'

        # Ensure the image save directory exists
        if not os.path.exists(self.img_path):
            try:
                os.makedirs(self.img_path)
                self.get_logger().info(f'Created directory: {self.img_path}')
            except OSError as e:
                self.get_logger().error(f'Failed to create directory {self.img_path}: {e}')
                raise

        self.get_logger().info(f'Images will be saved to: {self.img_path}')

        # Initialize image counter
        self.img_counter = 0

        # Set up subscriber to the compressed image topic
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.subscription = self.create_subscription(
            CompressedImage,
            self.topic,
            self.sub_callback,
            qos_profile
        )
        self.subscription  # Prevent unused variable warning

        self.get_logger().info(f'Subscribed to topic: {self.topic}')

    def sub_callback(self, msg):
        try:
            # Convert compressed image data to NumPy array
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn('Received an empty image.')
                return

            window_title = "Image Capture - Press 'q' to quit, 'SPACE' to capture"
            cv2.imshow(window_title, image)

            # Wait for 1 ms for a key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.get_logger().info('\'q\' pressed. Shutting down node.')
                rclpy.shutdown()

            elif key == 32:  # SPACE key
                img_name = f"frame_{self.img_counter}.jpg"
                img_full_path = os.path.join(self.img_path, img_name)
                success = cv2.imwrite(img_full_path, image)
                if success:
                    self.get_logger().info(f"Captured and saved image: {img_full_path}")
                    self.img_counter += 1
                else:
                    self.get_logger().error(f"Failed to save image: {img_full_path}")

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in callback: {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        get_images_node = GetImagesNode()
    except Exception as e:
        print(f"Failed to initialize node: {e}")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(get_images_node)
    except KeyboardInterrupt:
        get_images_node.get_logger().info('Keyboard interrupt received. Shutting down node.')
    finally:
        get_images_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
