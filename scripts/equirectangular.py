#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import message_filters
import math
import argparse
import json
import os
from rcl_interfaces.msg import SetParametersResult


def parse_args():
    # Filter out ros arguments before parsing
    ros_args = []
    regular_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('--ros-args') or arg.startswith('-r'):
            ros_args.append(arg)
        else:
            regular_args.append(arg)

    parser = argparse.ArgumentParser(description='Convert dual fisheye images to equirectangular format')
    parser.add_argument('--calibrate', action='store_true', help='Enable calibration mode')
    parser.add_argument('--calibration_file', type=str, default='/home/abanesjo/ros2_ws/src/Triton/dependencies/insta360_ros_driver/config/extrinsics.json', help='Path to calibration JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration for equirectangular projection')
    
    # Parse only non-ROS arguments
    return parser.parse_args(regular_args)


class EquirectangularNode(Node):
    def __init__(self, enable_calibration=False, calibration_file='extrinsics.json', use_gpu=False):
        super().__init__('equirectangular_node')

        self.params_changed = True
        
        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available() and use_gpu
        self.get_logger().info(f"GPU acceleration requested: {use_gpu}")
        self.get_logger().info(f"CUDA available: {torch.cuda.is_available()}")
        self.get_logger().info(f"Using GPU acceleration: {self.use_cuda}")
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Set calibration mode flag early
        self.calibration_mode = enable_calibration

        # Flag to track if mapping is initialized
        self.maps_initialized = False
        
        # Store image dimensions
        self.img_height = None
        self.img_width = None
        
        # Store last images for potential reprocessing
        self.last_front_img = None
        self.last_back_img = None
        
        # Default parameter values
        self.cx_offset = 0.0
        self.cy_offset = 0.0
        self.crop_size = 960
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.out_width = 1920
        self.out_height = 960
        
        # Try to load calibration from file
        self.calibration_file = calibration_file
        self.load_calibration()
        
        # Declare parameters with defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                ('cx_offset', self.cx_offset),
                ('cy_offset', self.cy_offset),
                ('crop_size', self.crop_size),
                ('tx', self.tx),
                ('ty', self.ty),
                ('tz', self.tz),
                ('roll', self.roll),
                ('pitch', self.pitch),
                ('yaw', self.yaw),
                ('out_width', self.out_width),
                ('out_height', self.out_height)
            ]
        )
        
        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Read initial parameters
        self.update_camera_parameters()
        
        # Create subscriber for the combined dual fisheye image
        self.bridge = CvBridge()
        self.dual_fisheye_sub = self.create_subscription(
            Image, 
            '/dual_fisheye/image', 
            self.image_callback, 
            10
        )
        
        # Publisher
        self.equirect_pub = self.create_publisher(
            Image, '/equirectangular/image', 10)
        
        # Setup calibration UI if enabled
        if self.calibration_mode:
            self.get_logger().info("Calibration mode enabled")
            self.setup_calibration_ui()
    
    def load_calibration(self):
        """Load calibration parameters from JSON file"""
        if not os.path.isfile(self.calibration_file):
            self.get_logger().warn(f"Calibration file not found: {self.calibration_file}")
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                params = json.load(f)
            
            self.cx_offset = params.get('cx_offset', self.cx_offset)
            self.cy_offset = params.get('cy_offset', self.cy_offset)
            self.crop_size = params.get('crop_size', self.crop_size)
            
            translation = params.get('translation', [self.tx, self.ty, self.tz])
            self.tx, self.ty, self.tz = translation
            
            rotation_deg = params.get('rotation_deg', [0.0, 0.0, 0.0])
            self.roll = math.radians(rotation_deg[0])
            self.pitch = math.radians(rotation_deg[1])
            self.yaw = math.radians(rotation_deg[2])
            
            self.get_logger().info(f"Loaded calibration parameters from {self.calibration_file}")
            self.get_logger().info(f"  Crop size: {self.crop_size}")
            self.get_logger().info(f"  Center offset: ({self.cx_offset}, {self.cy_offset})")
            self.get_logger().info(f"  Translation: [{self.tx}, {self.ty}, {self.tz}]")
            self.get_logger().info(f"  Rotation (deg): {rotation_deg}")
            
            return True
        except Exception as e:
            self.get_logger().error(f"Error loading calibration file: {e}")
            return False
    
    def save_calibration(self):
        """Save current calibration parameters to JSON file"""
        params = {
            'cx_offset': self.cx_offset,
            'cy_offset': self.cy_offset,
            'crop_size': self.crop_size,
            'translation': [self.tx, self.ty, self.tz],
            'rotation_deg': [
                math.degrees(self.roll),
                math.degrees(self.pitch),
                math.degrees(self.yaw)
            ]
        }
        
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(params, f, indent=2)
                
            self.get_logger().info(f"Parameters saved to {self.calibration_file}")
            return True
        except Exception as e:
            self.get_logger().error(f"Error saving calibration file: {e}")
            return False

    def parameters_callback(self, params):
        """Parameter update callback for dynamic reconfiguration"""
        update_needed = False
        
        for param in params:
            # Check if a camera parameter was changed
            if param.name in ['cx_offset', 'cy_offset', 'crop_size', 'tx', 'ty', 'tz',
                             'roll', 'pitch', 'yaw', 'out_width', 'out_height']:
                update_needed = True
                
        if update_needed:
            self.update_camera_parameters()
            
        return SetParametersResult(successful=True)
    
    def update_trackbar_positions(self):
        """Update trackbar positions to match current parameter values"""
        if not hasattr(self, 'control_window'):
            return
            
        # Update trackbar positions without triggering callbacks
        cv2.setTrackbarPos("CX Offset [-100,100]", self.control_window, int(self.cx_offset) + 100)
        cv2.setTrackbarPos("CY Offset [-100,100]", self.control_window, int(self.cy_offset) + 100)
        cv2.setTrackbarPos("Crop Size", self.control_window, self.crop_size)
        cv2.setTrackbarPos("TX [-0.5,0.5]", self.control_window, int(self.tx * 1000) + 500)
        cv2.setTrackbarPos("TY [-0.5,0.5]", self.control_window, int(self.ty * 1000) + 500)
        cv2.setTrackbarPos("TZ [-0.5,0.5]", self.control_window, int(self.tz * 1000) + 500)
        cv2.setTrackbarPos("Roll [-180°,180°]", self.control_window, int(math.degrees(self.roll) * 10) + 1800)
        cv2.setTrackbarPos("Pitch [-180°,180°]", self.control_window, int(math.degrees(self.pitch) * 10) + 1800)
        cv2.setTrackbarPos("Yaw [-180°,180°]", self.control_window, int(math.degrees(self.yaw) * 10) + 1800)

    def update_camera_parameters(self):
        # Do not re-read cx_offset and cy_offset so they remain as set by the UI.
        # self.cx_offset = self.get_parameter('cx_offset').value
        # self.cy_offset = self.get_parameter('cy_offset').value

        self.crop_size = self.get_parameter('crop_size').value
        self.out_width = self.get_parameter('out_width').value
        self.out_height = self.get_parameter('out_height').value

        self.tx = self.get_parameter('tx').value
        self.ty = self.get_parameter('ty').value
        self.tz = self.get_parameter('tz').value
        self.roll = self.get_parameter('roll').value
        self.pitch = self.get_parameter('pitch').value
        self.yaw = self.get_parameter('yaw').value

        # Rebuild the rotation matrix etc. as before
        Rx = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(self.roll), -math.sin(self.roll)],
            [0.0, math.sin(self.roll), math.cos(self.roll)]
        ], device=self.device)
        
        Ry = torch.tensor([
            [math.cos(self.pitch), 0.0, math.sin(self.pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(self.pitch), 0.0, math.cos(self.pitch)]
        ], device=self.device)
        
        Rz = torch.tensor([
            [math.cos(self.yaw), -math.sin(self.yaw), 0.0],
            [math.sin(self.yaw), math.cos(self.yaw), 0.0],
            [0.0, 0.0, 1.0]
        ], device=self.device)
        
        self.back_to_front_rotation = torch.matmul(torch.matmul(Rz, Ry), Rx)
        self.back_to_front_translation = torch.tensor([self.tx, self.ty, self.tz], device=self.device)
        
        if self.maps_initialized and not self.calibration_mode:
            self.maps_initialized = False
            self.get_logger().info("Parameters updated, remapping will occur on next image")

    def init_mapping(self, img_height, img_width):
        """Initialize mapping matrices for equirectangular projection"""
        self.get_logger().info(f"Initializing mapping matrices for {self.out_width}x{self.out_height}")
        
        # Store dimensions for future use
        self.img_height = img_height
        self.img_width = img_width
        
        # Set cx and cy based on crop size and offset
        self.cx = img_width / 2 + self.cx_offset
        self.cy = img_height / 2 + self.cy_offset
        
        # Create coordinate grid - ensure consistent dtype
        y, x = torch.meshgrid(
            torch.arange(self.out_height, dtype=torch.float32, device=self.device),
            torch.arange(self.out_width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Convert to normalized coordinates
        longitude = (x / self.out_width) * 2 * math.pi - math.pi
        latitude = (y / self.out_height) * math.pi - math.pi/2
        
        # Convert to 3D points on unit sphere
        X = torch.cos(latitude) * torch.sin(longitude)
        Y = torch.sin(latitude)
        Z = torch.cos(latitude) * torch.cos(longitude)
        
        # Create front and back masks
        self.front_mask = (Z >= 0)
        self.back_mask = (Z < 0)
        
        # Convert masks to NumPy once for reuse
        self.front_mask_np = self.front_mask.cpu().numpy()
        self.back_mask_np = self.back_mask.cpu().numpy()
        
        # Calculate mapping for front camera
        r_front = torch.sqrt(X[self.front_mask]**2 + Y[self.front_mask]**2)
        # Avoid division by zero
        r_front = torch.clamp(r_front, min=1e-6)
        theta_front = torch.atan2(r_front, torch.abs(Z[self.front_mask]))
        r_fisheye_front = 2 * theta_front / math.pi * (img_width / 2)
        
        # Initialize map arrays with consistent dtype - both PyTorch and NumPy versions
        self.front_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.front_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        # Create NumPy arrays directly
        self.front_map_x_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.front_map_y_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        
        # Assign values to PyTorch tensors
        self.front_map_x[self.front_mask] = self.cx + X[self.front_mask]/r_front * r_fisheye_front
        self.front_map_y[self.front_mask] = self.cy + Y[self.front_mask]/r_front * r_fisheye_front
        
        # Copy values to NumPy arrays
        self.front_map_x_np[self.front_mask_np] = self.front_map_x[self.front_mask].cpu().numpy()
        self.front_map_y_np[self.front_mask_np] = self.front_map_y[self.front_mask].cpu().numpy()
        
        # Calculate mapping for back camera (mirror X)
        back_X = X[self.back_mask]
        back_Y = Y[self.back_mask]
        back_Z = Z[self.back_mask]
        
        # Stack to form 3D points
        back_points = torch.stack([back_X, back_Y, back_Z], dim=1)
        
        # Create rotation and translation with explicit dtype
        rotation = self.back_to_front_rotation.to(torch.float32)
        translation = self.back_to_front_translation.to(torch.float32)
        
        # Apply rotation from back to front camera frame
        transformed_points = torch.matmul(back_points, rotation.transpose(0, 1))
        
        # Apply translation vector
        transformed_points = transformed_points + translation
        
        # Extract transformed coordinates
        X_back = -transformed_points[:, 0]  # Still negate X for back camera view
        Y_back = transformed_points[:, 1]
        Z_back = transformed_points[:, 2]
        
        # Continue with back camera mapping using transformed points
        r_back = torch.sqrt(X_back**2 + Y_back**2)
        r_back = torch.clamp(r_back, min=1e-6)
        theta_back = torch.atan2(r_back, torch.abs(Z_back))
        r_fisheye_back = 2 * theta_back / math.pi * (img_width / 2)
        
        # Initialize with consistent dtype - both PyTorch and NumPy versions
        self.back_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.back_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        # Create NumPy arrays directly for back maps
        self.back_map_x_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.back_map_y_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        
        # Assign values to PyTorch tensors
        self.back_map_x[self.back_mask] = self.cx + X_back/r_back * r_fisheye_back
        self.back_map_y[self.back_mask] = self.cy + Y_back/r_back * r_fisheye_back
        
        # Copy values to NumPy arrays
        self.back_map_x_np[self.back_mask_np] = self.back_map_x[self.back_mask].cpu().numpy()
        self.back_map_y_np[self.back_mask_np] = self.back_map_y[self.back_mask].cpu().numpy()
        
        # Prepare seam blending masks (edge detection for front/back boundary)
        # Dilate and erode to find the edge regions
        self.blend_kernel = np.ones((5, 5), np.uint8)
        self.front_edge = cv2.dilate(self.front_mask_np.astype(np.uint8), self.blend_kernel, iterations=2) - \
                        cv2.erode(self.front_mask_np.astype(np.uint8), self.blend_kernel, iterations=2)
        
        # Create distance maps for blending weights
        self.front_distance = cv2.distanceTransform(self.front_mask_np.astype(np.uint8), cv2.DIST_L2, 3)
        self.front_distance = np.clip(self.front_distance * 0.3, 0, 1)
        
        # Create a tensor for gpu operations if needed
        self.front_mask_torch = self.front_mask.unsqueeze(0).unsqueeze(0)
        
        self.maps_initialized = True
        self.get_logger().info("Mapping matrices initialized")

        # Additional GPU-specific initializations
        if self.use_cuda:
            try:
                # Use the faster [x,y] coordinate order
                front_map_x_norm = 2.0 * (self.front_map_x / self.img_width) - 1.0
                front_map_y_norm = 2.0 * (self.front_map_y / self.img_height) - 1.0
                
                # Stack in the faster order (the one that gave 20fps)
                self.front_grid = torch.stack([front_map_x_norm, front_map_y_norm], dim=-1).unsqueeze(0)
                
                back_map_x_norm = 2.0 * (self.back_map_x / self.img_width) - 1.0
                back_map_y_norm = 2.0 * (self.back_map_y / self.img_height) - 1.0
                self.back_grid = torch.stack([back_map_x_norm, back_map_y_norm], dim=-1).unsqueeze(0)
                
                # Create GPU mask tensors for faster processing
                self.front_mask_gpu = self.front_mask.float().unsqueeze(0).unsqueeze(0)
                self.back_mask_gpu = (~self.front_mask).float().unsqueeze(0).unsqueeze(0)
                
                # Create edge and blend masks
                self.edge_mask_gpu = torch.from_numpy(self.front_edge.astype(np.float32)).to(self.device)
                self.edge_mask_gpu = self.edge_mask_gpu.unsqueeze(0).unsqueeze(0)
                
                # Create blend weights
                self.blend_weight_gpu = torch.from_numpy(self.front_distance).to(self.device)
                self.blend_weight_gpu = self.blend_weight_gpu.unsqueeze(0).unsqueeze(0)
                
                self.get_logger().info("GPU acceleration initialized successfully")
            except Exception as e:
                self.use_cuda = False
                self.get_logger().error(f"Error initializing GPU acceleration, falling back to CPU: {e}")
                    
                self.maps_initialized = True
                self.get_logger().info(f"Mapping matrices initialized (GPU: {self.use_cuda})")
        
    def image_callback(self, dual_fisheye_msg):
        """Process the dual fisheye image to create equirectangular image"""
        try:
            # Convert ROS Image message to OpenCV image
            dual_fisheye_img = self.bridge.imgmsg_to_cv2(dual_fisheye_msg, "rgb8")
            
            # Split the dual fisheye image into front and back
            img_width = dual_fisheye_img.shape[1]
            midpoint = img_width // 2
            front_img = dual_fisheye_img[:, 0:midpoint].copy()
            back_img = dual_fisheye_img[:, midpoint:].copy()
            
            # Store original uncropped images
            self.original_front_img = front_img.copy()
            self.original_back_img = back_img.copy()
            
            # Crop images if needed to match crop_size
            if front_img.shape[0] != self.crop_size or front_img.shape[1] != self.crop_size:
                orig_height, orig_width = front_img.shape[:2]
                y_start = (orig_height - self.crop_size) // 2
                x_start = (orig_width - self.crop_size) // 2
                
                if y_start >= 0 and x_start >= 0 and y_start + self.crop_size <= orig_height and x_start + self.crop_size <= orig_width:
                    front_img = front_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]
                    back_img = back_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]
                    self.get_logger().debug(f"Cropped images to {self.crop_size}x{self.crop_size}")
            
            # Store cropped images for potential reuse
            self.last_front_img = front_img.copy()
            self.last_back_img = back_img.copy()
            
            # Initialize mapping if needed
            if not self.maps_initialized:
                self.init_mapping(front_img.shape[0], front_img.shape[1])
            elif front_img.shape[0] != self.img_height or front_img.shape[1] != self.img_width:
                self.init_mapping(front_img.shape[0], front_img.shape[1])
            
            # Create equirectangular image
            start_time = self.get_clock().now()
            if self.use_cuda:
                try:
                    equirect_img = self.create_equirectangular_gpu(front_img, back_img)
                except Exception as e:
                    self.get_logger().warn(f"GPU processing error: {e}, falling back to CPU")
                    equirect_img = self.create_equirectangular(front_img, back_img)
            else:
                equirect_img = self.create_equirectangular(front_img, back_img)
            
            # Publish result
            equirect_msg = self.bridge.cv2_to_imgmsg(equirect_img, encoding="rgb8")
            equirect_msg.header = dual_fisheye_msg.header
            self.equirect_pub.publish(equirect_msg)
            
            # Log processing time
            process_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            self.get_logger().debug(f"Processing time: {process_time:.3f} seconds")
            
            # Update calibration view if in calibration mode
            if self.calibration_mode:
                self.update_calibration_view()
            
        except Exception as e:
            self.get_logger().error(f"Error processing images: {str(e)}")
    
    def create_equirectangular(self, front_img, back_img):
        """Create equirectangular image from front and back fisheye images"""
        # Check if we need to initialize the mapping
        if not self.maps_initialized or self.params_changed:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            self.params_changed = False
        elif front_img.shape[0] != self.img_height or front_img.shape[1] != self.img_width:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
                
        # Use the pre-computed NumPy maps directly
        front_result = cv2.remap(front_img, self.front_map_x_np, self.front_map_y_np, 
                            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        back_result = cv2.remap(back_img, self.back_map_x_np, self.back_map_y_np,
                            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        # Create a more gradual blending at the seams (instead of abrupt mask-based combination)
        equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
        front_mask_np = self.front_mask.cpu().numpy()
        
        # Create a blend region around the seams (approximately 10 pixels wide)
        blend_kernel = np.ones((5, 5), np.uint8)
        front_edge = cv2.dilate(front_mask_np.astype(np.uint8), blend_kernel, iterations=3) - \
                    cv2.erode(front_mask_np.astype(np.uint8), blend_kernel, iterations=3)
        
        # Apply mask for non-blending regions
        equirect[front_mask_np & (front_edge==0)] = front_result[front_mask_np & (front_edge==0)]
        equirect[~front_mask_np & (front_edge==0)] = back_result[~front_mask_np & (front_edge==0)]
        
        # Blend the seam regions
        # blend_region = (front_edge == 1)
        # if np.any(blend_region):
        #     # Create gradient-based alpha for smooth transition
        #     alpha = cv2.distanceTransform(front_mask_np.astype(np.uint8), cv2.DIST_L2, 3)
        #     alpha = np.clip(alpha[blend_region] * 0.3, 0, 1)  # Scale to get smooth gradient
            
        #     # Alpha blend at seams
        #     equirect[blend_region] = (alpha[:, np.newaxis] * front_result[blend_region] + 
        #                             (1 - alpha[:, np.newaxis]) * back_result[blend_region]).astype(np.uint8)
        
        return equirect
    
    def create_equirectangular_gpu(self, front_img, back_img):
        """Create equirectangular image using GPU acceleration"""
        # Check if we need to initialize the mapping
        if not self.maps_initialized or self.params_changed:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            self.params_changed = False
        elif front_img.shape[0] != self.img_height or front_img.shape[1] != self.img_width:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
        
        # Convert images to PyTorch tensors and move to device
        front_tensor = torch.from_numpy(front_img).to(self.device).float()
        back_tensor = torch.from_numpy(back_img).to(self.device).float()
        
        # Reshape for grid_sample (N,C,H,W format)
        front_tensor = front_tensor.permute(2, 0, 1).unsqueeze(0)
        back_tensor = back_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Apply grid_sample with proper output size
        front_remapped = F.grid_sample(
            front_tensor, 
            self.front_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        back_remapped = F.grid_sample(
            back_tensor, 
            self.back_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Resize to match output dimensions if needed
        if front_remapped.shape[2:] != (self.out_height, self.out_width):
            front_remapped = F.interpolate(
                front_remapped, 
                size=(self.out_height, self.out_width), 
                mode='bilinear', 
                align_corners=True
            )
        
        if back_remapped.shape[2:] != (self.out_height, self.out_width):
            back_remapped = F.interpolate(
                back_remapped, 
                size=(self.out_height, self.out_width), 
                mode='bilinear', 
                align_corners=True
            )
        
        # Simple binary mask approach - no blending
        output = front_remapped * self.front_mask_gpu + back_remapped * (1.0 - self.front_mask_gpu)
        
        # Convert back to numpy array for ROS
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Efficient fix of orientation directly on CPU array
        # output_np = np.flip(output_np, axis=1).copy()
        
        return np.clip(output_np, 0, 255).astype(np.uint8)
        
    def setup_calibration_ui(self):
        """Set up UI for calibration mode"""
        # Window names
        self.window_name = "Equirectangular Calibration"
        self.control_window = "Calibration Controls"
        
        # Create windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.out_width // 2, self.out_height // 2)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        
        # Create trackbars with more intuitive ranges
        # Camera center offsets: range -100 to +100
        cv2.createTrackbar("CX Offset [-100,100]", self.control_window, 100, 200, self.update_cx)
        cv2.setTrackbarPos("CX Offset [-100,100]", self.control_window, int(self.cx_offset) + 100)
        
        cv2.createTrackbar("CY Offset [-100,100]", self.control_window, 100, 200, self.update_cy)
        cv2.setTrackbarPos("CY Offset [-100,100]", self.control_window, int(self.cy_offset) + 100)
        
        # Crop size (not centered around zero)
        cv2.createTrackbar("Crop Size", self.control_window, 960, 1280, self.update_crop)
        cv2.setTrackbarPos("Crop Size", self.control_window, self.crop_size)
        
        # Translation parameters: range -0.5 to +0.5
        cv2.createTrackbar("TX [-0.5,0.5]", self.control_window, 500, 1000, self.update_tx)
        cv2.setTrackbarPos("TX [-0.5,0.5]", self.control_window, int(self.tx * 1000) + 500)
        
        cv2.createTrackbar("TY [-0.5,0.5]", self.control_window, 500, 1000, self.update_ty)
        cv2.setTrackbarPos("TY [-0.5,0.5]", self.control_window, int(self.ty * 1000) + 500)
        
        cv2.createTrackbar("TZ [-0.5,0.5]", self.control_window, 500, 1000, self.update_tz)
        cv2.setTrackbarPos("TZ [-0.5,0.5]", self.control_window, int(self.tz * 1000) + 500)
        
        # Rotation parameters: range -180 to +180 degrees
        cv2.createTrackbar("Roll [-180°,180°]", self.control_window, 1800, 3600, self.update_roll)
        cv2.setTrackbarPos("Roll [-180°,180°]", self.control_window, int(math.degrees(self.roll) * 10) + 1800)
        
        cv2.createTrackbar("Pitch [-180°,180°]", self.control_window, 1800, 3600, self.update_pitch)
        cv2.setTrackbarPos("Pitch [-180°,180°]", self.control_window, int(math.degrees(self.pitch) * 10) + 1800)
        
        cv2.createTrackbar("Yaw [-180°,180°]", self.control_window, 1800, 3600, self.update_yaw)
        cv2.setTrackbarPos("Yaw [-180°,180°]", self.control_window, int(math.degrees(self.yaw) * 10) + 1800)

   # Trackbar update callbacks for calibration
    def update_cx(self, value):
        self.cx_offset = float(value - 100)  # -100 to +100 range
        self.set_parameters([rclpy.parameter.Parameter('cx_offset', rclpy.Parameter.Type.DOUBLE, self.cx_offset)])
        self.params_changed = True

    def update_cy(self, value):
        self.cy_offset = float(value - 100)  # -100 to +100 range
        self.set_parameters([rclpy.parameter.Parameter('cy_offset', rclpy.Parameter.Type.DOUBLE, self.cy_offset)])
        self.params_changed = True
        
    def update_crop(self, value):
        self.crop_size = value
        self.set_parameters([rclpy.parameter.Parameter('crop_size', rclpy.Parameter.Type.INTEGER, self.crop_size)])
        
        # Re-crop original images with new crop size
        if hasattr(self, 'original_front_img') and self.original_front_img is not None:
            orig_height, orig_width = self.original_front_img.shape[:2]
            y_start = (orig_height - self.crop_size) // 2
            x_start = (orig_width - self.crop_size) // 2
            
            if y_start >= 0 and x_start >= 0 and y_start + self.crop_size <= orig_height and x_start + self.crop_size <= orig_width:
                self.last_front_img = self.original_front_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size].copy()
                self.last_back_img = self.original_back_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size].copy()
                self.get_logger().debug(f"Re-cropped images to {self.crop_size}x{self.crop_size}")
        
        # Force remapping on next image
        if self.maps_initialized:
            self.maps_initialized = False
        self.params_changed = True
        
    def update_tx(self, value):
        self.tx = float((value - 500) / 1000.0)  # -0.5 to +0.5 range
        self.set_parameters([rclpy.parameter.Parameter('tx', rclpy.Parameter.Type.DOUBLE, self.tx)])
        self.params_changed = True

    def update_ty(self, value):
        self.ty = float((value - 500) / 1000.0)  # -0.5 to +0.5 range
        self.set_parameters([rclpy.parameter.Parameter('ty', rclpy.Parameter.Type.DOUBLE, self.ty)])
        self.params_changed = True

    def update_tz(self, value):
        self.tz = float((value - 500) / 1000.0)  # -0.5 to +0.5 range
        self.set_parameters([rclpy.parameter.Parameter('tz', rclpy.Parameter.Type.DOUBLE, self.tz)])
        self.params_changed = True

    def update_roll(self, value):
        self.roll = float(math.radians((value - 1800) / 10.0))  # -180 to +180 degrees
        self.set_parameters([rclpy.parameter.Parameter('roll', rclpy.Parameter.Type.DOUBLE, self.roll)])
        self.params_changed = True

    def update_pitch(self, value):
        self.pitch = float(math.radians((value - 1800) / 10.0))  # -180 to +180 degrees
        self.set_parameters([rclpy.parameter.Parameter('pitch', rclpy.Parameter.Type.DOUBLE, self.pitch)])
        self.params_changed = True

    def update_yaw(self, value):
        self.yaw = float(math.radians((value - 1800) / 10.0))  # -180 to +180 degrees
        self.set_parameters([rclpy.parameter.Parameter('yaw', rclpy.Parameter.Type.DOUBLE, self.yaw)])
        self.params_changed = True
    
    def update_calibration_view(self):
        """Update the calibration view with current images and parameters"""
        if not hasattr(self, 'window_name') or self.last_front_img is None or self.last_back_img is None:
            return
        
        if self.params_changed:
            # Call update_camera_parameters to sync with ROS parameters
            self.update_camera_parameters()
            self.maps_initialized = False
            # Don't reset params_changed here - let create_equirectangular handle it
            
        # Force remapping for calibration view
        if self.maps_initialized:
            self.maps_initialized = False
        
        # Create a copy of equirectangular image to add text
        equirect_bgr = cv2.cvtColor(self.create_equirectangular(self.last_front_img, self.last_back_img), cv2.COLOR_RGB2BGR)
        
        # Now that create_equirectangular has run, we can reset params_changed
        self.params_changed = False
        
        # Display current parameters
        info_text = (
            f"cx: {self.crop_size/2 + self.cx_offset:.1f}, cy: {self.crop_size/2 + self.cy_offset:.1f} | "
            f"crop: {self.crop_size} | "
            f"t: [{self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}] | "
            f"r: [{math.degrees(self.roll):.1f}, {math.degrees(self.pitch):.1f}, {math.degrees(self.yaw):.1f}]"
        )
        
        # Add info text to frame
        cv2.putText(
            equirect_bgr, 
            info_text,
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow(self.window_name, equirect_bgr)
        
        # Process key presses immediately
        key = cv2.waitKey(1)
        if key == ord('s'):  # Save parameters
            self.save_calibration()
        elif key == ord('q'):  # Quit calibration mode
            self.get_logger().info("Exiting calibration mode")
            cv2.destroyAllWindows()
            self.calibration_mode = False


def main(args=None):
    # Initialize ROS first with all its args
    rclpy.init(args=args)
    
    # Process arguments regardless of launch method
    enable_calibration = '--calibrate' in sys.argv
    use_gpu = '--gpu' in sys.argv
    calibration_file = '/home/abanesjo/ros2_ws/src/Triton/dependencies/insta360_ros_driver/config/extrinsics_air.json'
    
    # Check for custom calibration file
    for i, arg in enumerate(sys.argv[1:-1], 1):
        if arg == '--calibration_file' and i < len(sys.argv)-1:
            calibration_file = sys.argv[i+1]
    
    node = EquirectangularNode(
        enable_calibration=enable_calibration,
        calibration_file=calibration_file,
        use_gpu=use_gpu
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        if node.calibration_mode and hasattr(node, 'window_name'):
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()