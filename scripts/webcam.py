#!/usr/bin/env python3

# This is designed for Insta360 Air

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        # Declare parameters
        self.declare_parameter('camera_index', 4)
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('width', 2176)
        self.declare_parameter('height', 1088)

        # Get parameters
        self.camera_index = self.get_parameter('camera_index').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value

        # Create publisher
        self.publisher = self.create_publisher(
            Image,
            '/dual_fisheye/image',
            10  # QoS profile depth
        )

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize camera
        self.get_logger().info(f"Opening camera {self.camera_index}")
        self.cap = cv2.VideoCapture(self.camera_index)

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)

        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            return

        # Create timer for capturing frames
        self.timer = self.create_timer(1.0/self.frame_rate, self.timer_callback)
        
        self.get_logger().info(
            f"Webcam publisher started. Publishing to /dual_fisheye/image at {self.frame_rate} FPS"
        )
        
        # Frame counter
        self.frame_count = 0

    def timer_callback(self):
        """Capture and publish a frame"""
        try:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to capture frame!")
                return

            # Convert to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
            # Set header
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera"

            # Publish
            self.publisher.publish(img_msg)
            
            # Update frame counter
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames
                self.get_logger().debug(f"Published frame {self.frame_count}")

        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {str(e)}")

    def destroy_node(self):
        """Release resources when node is destroyed"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("Camera released")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()