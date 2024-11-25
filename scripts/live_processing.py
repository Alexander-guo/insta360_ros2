#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from insta360_ros_driver.tools import split_image, undistort_image, compress_image_to_msg

class LiveProcessingNode(Node):
    def __init__(self):
        super().__init__('live_processing')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Declare parameters with default values
        self.declare_parameter('undistort', False)
        self.declare_parameter('K', [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ])
        self.declare_parameter('D', [0.0, 0.0, 0.0, 0.0])

        # Retrieve parameters
        self.undistort = self.get_parameter('undistort').value
        K_param = self.get_parameter('K').value
        D_param = self.get_parameter('D').value

        # Convert parameters to numpy arrays
        try:
            self.K = np.array(K_param, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"K matrix:\n{self.K}")
        except Exception as e:
            self.get_logger().error(f"Error parsing K matrix: {e}")
            self.K = np.eye(3)  # Fallback to identity

        try:
            self.D = np.array(D_param, dtype=np.float64).flatten()
            if self.D.size < 4:
                raise ValueError(f"D vector must have at least 4 elements, but got {self.D.size}")
            self.get_logger().info(f"D vector: {self.D}")
        except Exception as e:
            self.get_logger().error(f"Error parsing D vector: {e}")
            self.D = np.zeros(4)  # Fallback to zero vector

        # Set up subscriber to the YUV image topic
        self.subscription = self.create_subscription(
            Image,
            '/insta_image_yuv',
            self.processing_callback,
            10  # QoS history depth
        )
        self.subscription  # Prevent unused variable warning

        # Set up publishers for front and back compressed images
        self.front_image_pub = self.create_publisher(
            CompressedImage,
            'front_camera_image/compressed',
            10  # QoS history depth
        )
        self.back_image_pub = self.create_publisher(
            CompressedImage,
            'back_camera_image/compressed',
            10  # QoS history depth
        )

        self.get_logger().info('LiveProcessingNode has been initialized.')

    def processing_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (YUV format)
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Convert the YUV image to BGR format
            bgr_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
            self.get_logger().info(f"Image Size: {bgr_image.shape}")

            # Split the image horizontally into front and back halves
            front_image, back_image = split_image(bgr_image)

            # Apply undistortion if enabled
            if self.undistort:
                front_image = undistort_image(front_image, self.K, self.D)
                back_image = undistort_image(back_image, self.K, self.D)
                self.get_logger().debug('Undistortion applied to images.')

            # Convert processed images to CompressedImage messages
            front_compressed_msg = compress_image_to_msg(front_image, msg.header.stamp)
            back_compressed_msg = compress_image_to_msg(back_image, msg.header.stamp)

            # Publish the compressed images
            self.front_image_pub.publish(front_compressed_msg)
            self.back_image_pub.publish(back_compressed_msg)

            self.get_logger().debug('Published front and back compressed images.')

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LiveProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('LiveProcessingNode has been stopped manually.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
