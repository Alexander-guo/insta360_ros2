#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math
import os
import json
from rcl_interfaces.msg import SetParametersResult
import argparse


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
    parser.add_argument('--calibration_file', type=str, default='/home/triton/ros2_ws/src/Triton/dependencies/insta360_ros_driver/config/extrinsics_x2.json', help='Path to calibration JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration for equirectangular projection')
    
    # Parse only non-ROS arguments
    return parser.parse_args(regular_args)


class EquirectangularNode(Node):
    def __init__(self, enable_calibration=False, calibration_file='extrinsics.json', use_gpu=False):
        super().__init__('equirectangular_node')

        self.params_changed = True
        
        self.use_cuda = torch.cuda.is_available() and use_gpu
        self.get_logger().info(f"GPU acceleration: requested={use_gpu}, available={torch.cuda.is_available()}, using={self.use_cuda}")
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.calibration_mode = enable_calibration
        self.maps_initialized = False
        
        self.img_height: int | None = None
        self.img_width: int | None = None
        
        self.last_front_img: np.ndarray | None = None
        self.last_back_img: np.ndarray | None = None
        self.original_front_img: np.ndarray | None = None
        self.original_back_img: np.ndarray | None = None

        # Precomputed masks and kernels
        self.front_mask: torch.Tensor | None = None
        self.back_mask: torch.Tensor | None = None
        self.front_mask_np: np.ndarray | None = None
        self.back_mask_np: np.ndarray | None = None
        self.blend_kernel: np.ndarray = np.ones((5, 5), np.uint8)
        self.front_edge: np.ndarray | None = None
        self.front_distance: np.ndarray | None = None

        # GPU specific masks and grids
        self.front_grid: torch.Tensor | None = None
        self.back_grid: torch.Tensor | None = None
        self.front_mask_gpu: torch.Tensor | None = None
        self.back_mask_gpu: torch.Tensor | None = None
        self.edge_mask_gpu: torch.Tensor | None = None # Retained if used elsewhere, otherwise can be removed
        self.blend_weight_gpu: torch.Tensor | None = None # Retained if used elsewhere
        
        # Default parameter values
        self.cx_offset: float = 0.0
        self.cy_offset: float = 0.0
        self.crop_size: int = 960
        self.tx: float = 0.0
        self.ty: float = 0.0
        self.tz: float = 0.0
        self.roll: float = 0.0
        self.pitch: float = 0.0
        self.yaw: float = math.radians(90.0) # Initialize in radians
        self.out_width: int = 1920
        self.out_height: int = 960
        
        self.calibration_file = calibration_file
        self.load_calibration() # Load calibration before declaring parameters to use loaded values as defaults
        
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
        
        self.add_on_set_parameters_callback(self.parameters_callback)
        self.update_camera_parameters() # Read initial parameters after declaration
        
        self.bridge = CvBridge()
        
        # Configure QoS for reliable communication with buffer size 1
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE
        )
        
        self.dual_fisheye_sub = self.create_subscription(
            Image, '/dual_fisheye/image', self.image_callback, qos)
        self.equirect_pub = self.create_publisher(
            Image, '/equirectangular/image', qos)
        
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

        # For parameters controlled by UI trackbars (crop_size, tx, ty, tz, roll, pitch, yaw),
        # their instance variables are updated directly by their respective callbacks.
        # We avoid re-reading them from self.get_parameter() here to prevent using a
        # potentially stale value if the ROS parameter service hasn't processed the update yet.
        # self.crop_size = self.get_parameter('crop_size').value # Updated by update_crop callback
        
        # Read and cast out_width and out_height
        out_width_param = self.get_parameter('out_width')
        if out_width_param.value is not None:
            self.out_width = int(out_width_param.value)
        else:
            self.get_logger().warn("out_width parameter is None, using default or previous value if available.")
            if not hasattr(self, 'out_width') or self.out_width is None: # Ensure it has some value
                 self.out_width = 1920 # Default fallback

        out_height_param = self.get_parameter('out_height')
        if out_height_param.value is not None:
            self.out_height = int(out_height_param.value)
        else:
            self.get_logger().warn("out_height parameter is None, using default or previous value if available.")
            if not hasattr(self, 'out_height') or self.out_height is None: # Ensure it has some value
                self.out_height = 960 # Default fallback

        # self.tx = self.get_parameter('tx').value # Updated by update_tx callback
        # self.ty = self.get_parameter('ty').value # Updated by update_ty callback
        # self.tz = self.get_parameter('tz').value # Updated by update_tz callback
        # self.roll = self.get_parameter('roll').value # Updated by update_roll callback
        # self.pitch = self.get_parameter('pitch').value # Updated by update_pitch callback
        # self.yaw = self.get_parameter('yaw').value # Updated by update_yaw callback

        # Rebuild the rotation matrix etc. as before, using the current instance variables
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

    def init_mapping(self, img_height: int, img_width: int):
        """Initialize mapping matrices for equirectangular projection."""
        if self.out_width is None or self.out_height is None:
            self.get_logger().error("Output dimensions (out_width, out_height) are not set. Cannot initialize mapping.")
            return

        self.get_logger().info(f"Initializing mapping for {img_width}x{img_height} to {self.out_width}x{self.out_height}")
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.cx = img_width / 2 + self.cx_offset
        self.cy = img_height / 2 + self.cy_offset
        
        y, x = torch.meshgrid(
            torch.arange(self.out_height, dtype=torch.float32, device=self.device),
            torch.arange(self.out_width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        longitude = (x / self.out_width) * 2 * math.pi - math.pi
        latitude = (y / self.out_height) * math.pi - math.pi / 2
        
        X = torch.cos(latitude) * torch.sin(longitude)
        Y = torch.sin(latitude)
        Z = torch.cos(latitude) * torch.cos(longitude)
        
        self.front_mask = (Z >= 0)
        self.back_mask = (Z < 0)
        
        self.front_mask_np = self.front_mask.cpu().numpy()
        self.back_mask_np = self.back_mask.cpu().numpy()
        
        r_front = torch.sqrt(X[self.front_mask]**2 + Y[self.front_mask]**2).clamp_(min=1e-6)
        theta_front = torch.atan2(r_front, torch.abs(Z[self.front_mask]))
        r_fisheye_front = 2 * theta_front / math.pi * (self.img_width / 2) # Use self.img_width
        
        self.front_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.front_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        self.front_map_x[self.front_mask] = self.cx + X[self.front_mask] / r_front * r_fisheye_front
        self.front_map_y[self.front_mask] = self.cy + Y[self.front_mask] / r_front * r_fisheye_front
        
        back_X_tensor = X[self.back_mask]
        back_Y_tensor = Y[self.back_mask]
        back_Z_tensor = Z[self.back_mask]
        
        back_points = torch.stack([back_X_tensor, back_Y_tensor, back_Z_tensor], dim=1)
        
        rotation = self.back_to_front_rotation.to(dtype=torch.float32)
        translation = self.back_to_front_translation.to(dtype=torch.float32)
        
        transformed_points = torch.matmul(back_points, rotation.transpose(0, 1)) + translation
        
        X_back = -transformed_points[:, 0]
        Y_back = transformed_points[:, 1]
        Z_back = transformed_points[:, 2]
        
        r_back = torch.sqrt(X_back**2 + Y_back**2).clamp_(min=1e-6)
        theta_back = torch.atan2(r_back, torch.abs(Z_back))
        r_fisheye_back = 2 * theta_back / math.pi * (self.img_width / 2) # Use self.img_width
        
        self.back_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.back_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)

        self.back_map_x[self.back_mask] = self.cx + X_back / r_back * r_fisheye_back
        self.back_map_y[self.back_mask] = self.cy + Y_back / r_back * r_fisheye_back

        self.front_map_x_np = self.front_map_x.cpu().numpy()
        self.front_map_y_np = self.front_map_y.cpu().numpy()
        self.back_map_x_np = self.back_map_x.cpu().numpy()
        self.back_map_y_np = self.back_map_y.cpu().numpy()
        
        if self.front_mask_np is not None: # self.blend_kernel is pre-initialized
            front_mask_uint8 = self.front_mask_np.astype(np.uint8)
            self.front_edge = cv2.dilate(front_mask_uint8, self.blend_kernel, iterations=2) - \
                              cv2.erode(front_mask_uint8, self.blend_kernel, iterations=2)
            self.front_distance = cv2.distanceTransform(front_mask_uint8, cv2.DIST_L2, 3)
            self.front_distance = np.clip(self.front_distance * 0.3, 0, 1)
        
        self.maps_initialized = True

        if self.use_cuda:
            try:
                if self.img_width is None or self.img_height is None: 
                    self.get_logger().error("Image dimensions (img_width, img_height) are None. Cannot initialize GPU grids.")
                    self.use_cuda = False # Fallback if critical dimensions are missing
                else:
                    front_map_x_norm = 2.0 * (self.front_map_x / self.img_width) - 1.0
                    front_map_y_norm = 2.0 * (self.front_map_y / self.img_height) - 1.0
                    self.front_grid = torch.stack([front_map_x_norm, front_map_y_norm], dim=-1).unsqueeze(0)
                    
                    back_map_x_norm = 2.0 * (self.back_map_x / self.img_width) - 1.0
                    back_map_y_norm = 2.0 * (self.back_map_y / self.img_height) - 1.0
                    self.back_grid = torch.stack([back_map_x_norm, back_map_y_norm], dim=-1).unsqueeze(0)
                    
                    if self.front_mask is not None: # Check front_mask before use
                        self.front_mask_gpu = self.front_mask.float().unsqueeze(0).unsqueeze(0)
                    if self.back_mask is not None: # Check back_mask before use
                        self.back_mask_gpu = self.back_mask.float().unsqueeze(0).unsqueeze(0)
                    
                    # GPU edge and blend masks (if needed for GPU blending later)
                    if self.front_edge is not None:
                        self.edge_mask_gpu = torch.from_numpy(self.front_edge.astype(np.float32)).to(self.device, non_blocking=True).unsqueeze(0).unsqueeze(0)
                    if self.front_distance is not None:
                        self.blend_weight_gpu = torch.from_numpy(self.front_distance.astype(np.float32)).to(self.device, non_blocking=True).unsqueeze(0).unsqueeze(0)
                    
                    self.get_logger().info("GPU acceleration resources initialized successfully")

            except Exception as e:
                self.use_cuda = False # Fallback if GPU init fails
                self.get_logger().error(f"Error initializing GPU acceleration resources, falling back to CPU: {e}")
        
        self.get_logger().info(f"Mapping matrices initialization complete (GPU: {self.use_cuda})")

    def image_callback(self, dual_fisheye_msg: Image):
        """Process the dual fisheye image to create equirectangular image"""
        try:
            dual_fisheye_img = self.bridge.imgmsg_to_cv2(dual_fisheye_msg, "rgb8")
            
            img_height, img_width_full, _ = dual_fisheye_img.shape
            midpoint = img_width_full // 2
            front_img_full = dual_fisheye_img[:, midpoint:]
            back_img_full = dual_fisheye_img[:, :midpoint]

            front_img_full = cv2.rotate(front_img_full, cv2.ROTATE_90_COUNTERCLOCKWISE)
            back_img_full = cv2.rotate(back_img_full, cv2.ROTATE_90_CLOCKWISE)
            
            # Store original uncropped images if not already stored or if they change
            # This is useful if crop_size can change dynamically via parameters after startup
            if self.original_front_img is None or self.original_front_img.shape != front_img_full.shape:
                self.original_front_img = front_img_full.copy()
            if self.original_back_img is None or self.original_back_img.shape != back_img_full.shape:
                self.original_back_img = back_img_full.copy()

            # Determine current front_img and back_img based on crop_size
            # This logic assumes self.crop_size is an int and has been validated
            current_crop_size = self.crop_size
            orig_h, orig_w = front_img_full.shape[:2]

            if orig_h != current_crop_size or orig_w != current_crop_size:
                y_start = (orig_h - current_crop_size) // 2
                x_start = (orig_w - current_crop_size) // 2
                
                if y_start >= 0 and x_start >= 0 and \
                   y_start + current_crop_size <= orig_h and x_start + current_crop_size <= orig_w:
                    front_img = front_img_full[y_start:y_start+current_crop_size, x_start:x_start+current_crop_size]
                    back_img = back_img_full[y_start:y_start+current_crop_size, x_start:x_start+current_crop_size]
                else:
                    self.get_logger().warn(f"Cannot crop to {current_crop_size}x{current_crop_size}. Using uncropped images.")
                    front_img = front_img_full
                    back_img = back_img_full
            else:
                front_img = front_img_full
                back_img = back_img_full
            
            self.last_front_img = front_img.copy() # For calibration view
            self.last_back_img = back_img.copy()   # For calibration view
            
            # Initialize mapping if needed or if input image dimensions/params changed
            # Check self.img_height and self.img_width against current front_img dimensions
            if not self.maps_initialized or self.params_changed or \
               (self.img_height is not None and front_img.shape[0] != self.img_height) or \
               (self.img_width is not None and front_img.shape[1] != self.img_width):
                self.init_mapping(front_img.shape[0], front_img.shape[1])
                self.params_changed = False # Reset flag after re-initialization
            
            start_time = self.get_clock().now()
            if self.use_cuda and self.maps_initialized: # Ensure maps are ready for GPU
                try:
                    equirect_img = self.create_equirectangular_gpu(front_img, back_img)
                except Exception as e:
                    self.get_logger().warn(f"GPU processing error: {e}, falling back to CPU")
                    self.use_cuda = False # Disable GPU for subsequent frames if it errors
                    equirect_img = self.create_equirectangular(front_img, back_img)
            else:
                if not self.maps_initialized:
                    self.get_logger().warn("Maps not initialized, attempting CPU processing with on-the-fly init.")
                equirect_img = self.create_equirectangular(front_img, back_img)
            
            equirect_msg = self.bridge.cv2_to_imgmsg(equirect_img, encoding="rgb8")
            equirect_msg.header = dual_fisheye_msg.header
            self.equirect_pub.publish(equirect_msg)
            
            process_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            self.get_logger().debug(f"Processing time: {process_time:.3f} seconds (GPU: {self.use_cuda if self.maps_initialized else 'N/A'})")
            
            if self.calibration_mode:
                self.update_calibration_view()
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing images: {e}", exc_info=True)

    def create_equirectangular(self, front_img: np.ndarray, back_img: np.ndarray) -> np.ndarray:
        """Create equirectangular image from front and back fisheye images using CPU."""
        # Ensure maps are initialized, re-initialize if necessary
        if not self.maps_initialized or self.params_changed or \
           (self.img_height is not None and front_img.shape[0] != self.img_height) or \
           (self.img_width is not None and front_img.shape[1] != self.img_width):
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            self.params_changed = False
        
        # Fallback if maps are still not ready (e.g., output dimensions were None)
        if not self.maps_initialized or self.front_map_x_np is None or self.front_map_y_np is None or \
           self.back_map_x_np is None or self.back_map_y_np is None or \
           self.front_mask_np is None or self.front_edge is None or \
           self.out_height is None or self.out_width is None: # out_height/width check
            self.get_logger().error("CPU Mapping arrays or output dimensions not properly initialized. Returning black image.")
            # Attempt to use declared defaults if instance attributes are None
            h = self.out_height if self.out_height is not None else 1920
            w = self.out_width if self.out_width is not None else 3840
            return np.zeros((h, w, 3), dtype=np.uint8)
                
        front_result = cv2.remap(front_img, self.front_map_x_np, self.front_map_y_np, 
                                 cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        back_result = cv2.remap(back_img, self.back_map_x_np, self.back_map_y_np,
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
        
        non_edge_front = self.front_mask_np & (self.front_edge == 0)
        non_edge_back = ~self.front_mask_np & (self.front_edge == 0)

        equirect[non_edge_front] = front_result[non_edge_front]
        equirect[non_edge_back] = back_result[non_edge_back]
        
        if self.front_distance is not None:
            blend_region = (self.front_edge == 1)
            if np.any(blend_region):
                # Ensure alpha is correctly shaped for broadcasting
                alpha = self.front_distance[blend_region][..., np.newaxis] 
                equirect[blend_region] = (alpha * front_result[blend_region].astype(np.float32) + 
                                        (1 - alpha) * back_result[blend_region].astype(np.float32)).astype(np.uint8)
        
        return equirect
    
    def create_equirectangular_gpu(self, front_img: np.ndarray, back_img: np.ndarray) -> np.ndarray:
        """Create equirectangular image using GPU acceleration."""
        if not self.use_cuda: # Safeguard
            self.get_logger().warn("GPU processing called but not enabled/initialized. Falling back to CPU.")
            return self.create_equirectangular(front_img, back_img)

        # Ensure maps are initialized, re-initialize if necessary
        if not self.maps_initialized or self.params_changed or \
           (self.img_height is not None and front_img.shape[0] != self.img_height) or \
           (self.img_width is not None and front_img.shape[1] != self.img_width):
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            self.params_changed = False
        
        # Fallback if maps or GPU resources are still not ready
        if not self.maps_initialized or self.front_grid is None or self.back_grid is None or \
           self.front_mask_gpu is None or self.back_mask_gpu is None or \
           self.out_height is None or self.out_width is None: # out_height/width check
            self.get_logger().error("GPU grids or masks not initialized. Falling back to CPU processing.")
            self.use_cuda = False 
            return self.create_equirectangular(front_img, back_img)

        front_tensor = torch.from_numpy(front_img).to(self.device, non_blocking=True).float().permute(2, 0, 1).unsqueeze(0)
        back_tensor = torch.from_numpy(back_img).to(self.device, non_blocking=True).float().permute(2, 0, 1).unsqueeze(0)
        
        front_remapped = F.grid_sample(front_tensor, self.front_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        back_remapped = F.grid_sample(back_tensor, self.back_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Interpolation might be needed if grid_sample output size doesn't match out_height/out_width
        # This depends on how front_grid and back_grid are constructed relative to out_height/out_width
        if front_remapped.shape[2:] != (self.out_height, self.out_width):
            front_remapped = F.interpolate(front_remapped, size=(self.out_height, self.out_width), mode='bilinear', align_corners=True)
        if back_remapped.shape[2:] != (self.out_height, self.out_width):
            back_remapped = F.interpolate(back_remapped, size=(self.out_height, self.out_width), mode='bilinear', align_corners=True)
        
        # Combine using masks
        output = front_remapped * self.front_mask_gpu + back_remapped * self.back_mask_gpu
        
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
        cv2.createTrackbar("Crop Size", self.control_window, 1920, 3840, self.update_crop)
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
        self.set_parameters([Parameter('cx_offset', Parameter.Type.DOUBLE, self.cx_offset)])
        self.params_changed = True

    def update_cy(self, value):
        self.cy_offset = float(value - 100)  # -100 to +100 range
        self.set_parameters([Parameter('cy_offset', Parameter.Type.DOUBLE, self.cy_offset)])
        self.params_changed = True
        
    def update_crop(self, value: int):
        self.crop_size = value
        self.set_parameters([Parameter('crop_size', Parameter.Type.INTEGER, self.crop_size)])
        
        # Re-crop original images with new crop size
        if self.original_front_img is not None and self.original_back_img is not None:
            orig_height, orig_width = self.original_front_img.shape[:2]
            y_start = (orig_height - self.crop_size) // 2
            x_start = (orig_width - self.crop_size) // 2
            
            if y_start >= 0 and x_start >= 0 and \
               y_start + self.crop_size <= orig_height and x_start + self.crop_size <= orig_width:
                self.last_front_img = self.original_front_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size].copy()
                self.last_back_img = self.original_back_img[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size].copy()
                self.get_logger().debug(f"Re-cropped images to {self.crop_size}x{self.crop_size} for calibration view.")
            else:
                self.get_logger().warn(f"Cannot re-crop to {self.crop_size} for calibration view. Using previous last_images.")
        else:
            self.get_logger().warn("Original images not available for re-cropping in update_crop.")
        
        self.params_changed = True # Signal that mapping might need update
        # No direct call to init_mapping here; let image_callback handle it if dimensions change

    def update_tx(self, value):
        self.tx = float((value - 500) / 1000.0)  # -0.5 to +0.5 range
        self.set_parameters([Parameter('tx', Parameter.Type.DOUBLE, self.tx)])
        self.params_changed = True

    def update_ty(self, value):
        self.ty = float((value - 500) / 1000.0)  # -0.5 to +0.5 range
        self.set_parameters([Parameter('ty', Parameter.Type.DOUBLE, self.ty)])
        self.params_changed = True

    def update_tz(self, value):
        self.tz = float((value - 500) / 1000.0)  # -0.5 to +0.5 range
        self.set_parameters([Parameter('tz', Parameter.Type.DOUBLE, self.tz)])
        self.params_changed = True

    def update_roll(self, value):
        self.roll = float(math.radians((value - 1800) / 10.0))  # -180 to +180 degrees
        self.set_parameters([Parameter('roll', Parameter.Type.DOUBLE, self.roll)])
        self.params_changed = True

    def update_pitch(self, value):
        self.pitch = float(math.radians((value - 1800) / 10.0))  # -180 to +180 degrees
        self.set_parameters([Parameter('pitch', Parameter.Type.DOUBLE, self.pitch)])
        self.params_changed = True

    def update_yaw(self, value):
        self.yaw = float(math.radians((value - 1800) / 10.0))  # -180 to +180 degrees
        self.set_parameters([Parameter('yaw', Parameter.Type.DOUBLE, self.yaw)])
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
    calibration_file = '/home/triton/ros2_ws/src/Triton/dependencies/insta360_ros_driver/config/extrinsics_x2.json'
    
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