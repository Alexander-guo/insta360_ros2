#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImageUndistortNode(Node):
    def __init__(self):
        super().__init__('image_undistort_node')

        # Check for CUDA availability with PyTorch
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # Create CV bridge
        self.cv_bridge = CvBridge()

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('fx', 254.87597174247458),
                ('fy', 254.06164868269013),
                ('cx', 476.66470100570314),
                ('cy', 482.477893158099),
                ('k1', 0.08),
                ('k2', -0.02),
                ('p1', 0.0),
                ('p2', 0.0),
            ]
        )

        # Get parameters
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.k1 = self.get_parameter('k1').value
        self.k2 = self.get_parameter('k2').value
        self.p1 = self.get_parameter('p1').value
        self.p2 = self.get_parameter('p2').value

        # Create camera matrix and distortion coefficients
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        self.D = np.array([self.k1, self.k2, self.p1, self.p2])

        # Create subscription to dual fisheye image
        self.dual_fisheye_sub = self.create_subscription(
            Image,
            '/dual_fisheye/image',
            self.dual_fisheye_callback,
            10
        )

        # Create publishers with updated topic names
        self.front_pub = self.create_publisher(
            Image,
            '/front_camera/undistorted/image',
            10
        )
        self.back_pub = self.create_publisher(
            Image,
            '/back_camera/undistorted/image',
            10
        )

        # Log info
        self.get_logger().info('Image undistort node has been started')
        self.get_logger().info(f'Camera matrix: {self.K}')
        self.get_logger().info(f'Distortion coefficients: {self.D}')
        self.get_logger().info(f'Subscribing to dual fisheye image at /dual_fisheye/image')
        self.get_logger().info(f'Publishing front camera undistorted image to /front_camera/undistorted/image')
        self.get_logger().info(f'Publishing back camera undistorted image to /back_camera/undistorted/image')

        # Store undistortion maps
        self.undistortion_maps = {}

        # Check for GPU availability
        if self.use_gpu:
            self.get_logger().info('PyTorch GPU acceleration is available')
        else:
            self.get_logger().info('PyTorch GPU acceleration is NOT available, using CPU')

    def get_undistortion_map(self, height, width):
        """Create or retrieve cached undistortion map"""
        key = f"{height}_{width}"
        
        if key not in self.undistortion_maps:
            self.get_logger().info(f"Creating undistortion map for {width}x{height}")
            
            # Create undistortion maps using OpenCV
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, (width, height), cv2.CV_32FC1)
            
            if self.use_gpu:
                # Convert maps to PyTorch tensors on GPU
                map1_tensor = torch.from_numpy(map1).to(self.device)
                map2_tensor = torch.from_numpy(map2).to(self.device)
                
                # Normalize coordinates for grid_sample
                map1_normalized = 2.0 * map1_tensor / (width - 1) - 1.0
                map2_normalized = 2.0 * map2_tensor / (height - 1) - 1.0
                
                # Combine into grid
                grid = torch.stack([map1_normalized, map2_normalized], dim=-1).unsqueeze(0)
                self.undistortion_maps[key] = grid
            else:
                self.undistortion_maps[key] = (map1, map2)
        
        return self.undistortion_maps[key]

    def undistort_image(self, image):
        """Undistort image using PyTorch GPU or OpenCV CPU"""
        h, w = image.shape[:2]
        
        if self.use_gpu:
            # Get undistortion grid
            grid = self.get_undistortion_map(h, w)
            
            # Convert image to PyTorch tensor
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            
            # Perform undistortion using grid_sample
            undistorted = torch.nn.functional.grid_sample(
                img_tensor, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )
            
            # Convert back to numpy
            result = undistorted.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            return result
        else:
            # CPU fallback using OpenCV
            map1, map2 = self.get_undistortion_map(h, w)
            return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    def dual_fisheye_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Split the dual fisheye image into front and back halves
            h, w = cv_image.shape[:2]
            mid_point = w // 2
            
            front_image = cv_image[:, :mid_point]
            back_image = cv_image[:, mid_point:]
            
            # Process front image
            front_undistorted = self.undistort_image(front_image)
            front_msg = self.cv_bridge.cv2_to_imgmsg(front_undistorted, encoding='bgr8')
            front_msg.header = msg.header
            front_msg.header.frame_id = "front_camera_frame"
            self.front_pub.publish(front_msg)
            
            # Process back image
            back_undistorted = self.undistort_image(back_image)
            back_msg = self.cv_bridge.cv2_to_imgmsg(back_undistorted, encoding='bgr8')
            back_msg.header = msg.header
            back_msg.header.frame_id = "back_camera_frame"
            self.back_pub.publish(back_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing dual fisheye image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageUndistortNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()