#!/usr/bin/env python3
# filepath: /home/abanesjo/ros2_ws/src/Triton/dependencies/insta360_ros_driver/scripts/equirectangular_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import message_filters
import math
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange, IntegerRange


class EquirectangularNode(Node):
    def __init__(self):
        super().__init__('equirectangular_node')
        
        # Check for CUDA availability
        self.use_cuda = torch.cuda.is_available()
        self.get_logger().info(f"CUDA available: {self.use_cuda}")
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Flag to track if mapping is initialized
        self.maps_initialized = False
        
        # Store image dimensions
        self.img_height = None
        self.img_width = None
        
        # Store last images for potential reprocessing
        self.last_front_img = None
        self.last_back_img = None
        
        # Default parameter values - store for reset functionality
        self.default_params = {
            'fx': 254.87597174247458,
            'fy': 254.06164868269013,
            'cx': 476.66470100570314,
            'cy': 482.477893158099,
            'k1': 0.08,
            'k2': -0.02,
            'p1': 0.00,
            'p2': -0.00,
            'equirect_width': 1920,
            'equirect_height': 960,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'tx': 0.03,
            'ty': 0.0,
            'tz': -0.129
        }
        
        # Create parameter descriptors for dynamic reconfiguration
        int_desc = ParameterDescriptor(dynamic_typing=True)
        
        # Create reset descriptor
        reset_desc = ParameterDescriptor(
            dynamic_typing=True,
            description="Check to reset all parameters to default values"
        )
        
        # Camera parameters (with simplified naming for dynamic reconfiguration)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('fx', self.default_params['fx']),
                ('fy', self.default_params['fy']),
                ('cx', self.default_params['cx']),
                ('cy', self.default_params['cy']),
                ('k1', self.default_params['k1']),
                ('k2', self.default_params['k2']),
                ('p1', self.default_params['p1']),
                ('p2', self.default_params['p2']),
                ('equirect_width', self.default_params['equirect_width'], int_desc),
                ('equirect_height', self.default_params['equirect_height'], int_desc),
                ('roll', self.default_params['roll']),
                ('pitch', self.default_params['pitch']),
                ('yaw', self.default_params['yaw']),
                ('tx', self.default_params['tx']),  # Special range for translation
                ('ty', self.default_params['ty']),  # Special range for translation
                ('tz', self.default_params['tz']),  # Special range for translation
                ('reset_to_defaults', False, reset_desc),        # Reset checkbox
                ('calibration_mode', False, 
                    ParameterDescriptor(dynamic_typing=True, description="Enable calibration visualization")),
                ('start_calibration', False,
                    ParameterDescriptor(dynamic_typing=True, description="Start automatic calibration")),
                ('calibration_progress', 0.0,
                    ParameterDescriptor(dynamic_typing=True, description="Calibration progress (0-100%)")),
                ('calib_resolution', 'high', 
                    ParameterDescriptor(dynamic_typing=True, 
                                       description="Calibration resolution (low, medium, high)")),
                ('calib_range', 'very narrow',
                    ParameterDescriptor(dynamic_typing=True, 
                                       description="Calibration search range (narrow, normal, wide)")),
                ('calib_grid_search', False, 
                    ParameterDescriptor(dynamic_typing=True, 
                                       description="Use grid search for translations (slow but thorough)"))                  
            ]
        )

        self.calibrating = False
        self.calib_thread = None
        
        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Read initial parameters
        self.update_camera_parameters()
        
        # Create subscribers with synchronization
        self.bridge = CvBridge()
        self.front_sub = message_filters.Subscriber(self, Image, '/front_camera/image')
        self.back_sub = message_filters.Subscriber(self, Image, '/back_camera/image')
        
        # Synchronize messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.front_sub, self.back_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)
        
        # Publisher
        self.equirect_pub = self.create_publisher(
            Image, '/equirectangular/image', 10)
        
    def reset_parameters(self):
        """Reset all parameters to default values"""
        self.get_logger().info("Resetting parameters to default values")
        
        # Set parameters back to default values
        for param_name, default_value in self.default_params.items():
            self.set_parameters([rclpy.parameter.Parameter(
                param_name, 
                type_=rclpy.Parameter.Type.DOUBLE if isinstance(default_value, float) else rclpy.Parameter.Type.INTEGER,
                value=default_value
            )])
        
        # Reset the checkbox
        self.set_parameters([rclpy.parameter.Parameter(
            'reset_to_defaults', 
            type_=rclpy.Parameter.Type.BOOL,
            value=False
        )])
        
        # Force update with new parameters
        self.update_camera_parameters()

    def update_camera_parameters(self):
        """Update camera parameters from ROS parameters"""
        # Get intrinsic parameters
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        
        # Get distortion parameters
        self.D = [
            self.get_parameter('k1').value,
            self.get_parameter('k2').value,
            self.get_parameter('p1').value,
            self.get_parameter('p2').value
        ]
        
        # Get output dimensions
        self.out_width = self.get_parameter('equirect_width').value
        self.out_height = self.get_parameter('equirect_height').value
        
        # Get extrinsic parameters
        roll = math.radians(self.get_parameter('roll').value)
        pitch = math.radians(self.get_parameter('pitch').value)
        yaw = math.radians(self.get_parameter('yaw').value)
        
        # Create rotation matrix from Euler angles (roll, pitch, yaw)
        # Roll (rotation around X)
        Rx = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)]
        ], device=self.device)
        
        # Pitch (rotation around Y)
        Ry = torch.tensor([
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)]
        ], device=self.device)
        
        # Yaw (rotation around Z)
        Rz = torch.tensor([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ], device=self.device)
        
        # Combined rotation matrix (R = Rz @ Ry @ Rx)
        self.back_to_front_rotation = torch.matmul(torch.matmul(Rz, Ry), Rx)
        
        # Translation vector
        self.back_to_front_translation = torch.tensor([
            self.get_parameter('tx').value,
            self.get_parameter('ty').value,
            self.get_parameter('tz').value
        ], device=self.device)
        
        # Reset map initialization flag to force remapping
        if self.maps_initialized:
            self.maps_initialized = False
            self.get_logger().info("Parameters updated, remapping will occur on next image")
            
        # Immediately reinitialize mapping if we have dimensions
        if not self.maps_initialized and self.img_height is not None and self.img_width is not None:
            self.init_mapping(self.img_height, self.img_width)

    def parameters_callback(self, params):
        """Parameter update callback for dynamic reconfiguration"""
        reset_needed = False
        update_needed = False
        
        for param in params:
            # Check for reset request
            if param.name == 'reset_to_defaults' and param.value:
                reset_needed = True
                break
            
            # Check if a camera parameter was changed
            if param.name in self.default_params:
                update_needed = True
                
        if reset_needed:
            self.reset_parameters()
        elif update_needed:
            self.update_camera_parameters()
            
        return SetParametersResult(successful=True)

    def init_mapping(self, img_height, img_width):
        """Initialize mapping matrices for equirectangular projection"""
        self.get_logger().info(f"Initializing mapping matrices for {self.out_width}x{self.out_height}")
        
        # Store dimensions for future use
        self.img_height = img_height
        self.img_width = img_width
        
        # Create coordinate grid - ensure consistent dtype
        y, x = torch.meshgrid(
            torch.arange(self.out_height, dtype=torch.float32, device=self.device),
            torch.arange(self.out_width, dtype=torch.float32, device=self.device)
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
        
        # Calculate mapping for front camera
        r_front = torch.sqrt(X[self.front_mask]**2 + Y[self.front_mask]**2)
        # Avoid division by zero
        r_front = torch.clamp(r_front, min=1e-6)
        theta_front = torch.atan2(r_front, torch.abs(Z[self.front_mask]))
        r_fisheye_front = 2 * theta_front / math.pi * (img_width / 2)
        
        # Initialize map arrays with consistent dtype
        self.front_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.front_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        self.front_map_x[self.front_mask] = self.cx + X[self.front_mask]/r_front * r_fisheye_front
        self.front_map_y[self.front_mask] = self.cy + Y[self.front_mask]/r_front * r_fisheye_front
        
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
        
        # Initialize with consistent dtype
        self.back_map_x = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        self.back_map_y = torch.zeros((self.out_height, self.out_width), dtype=torch.float32, device=self.device)
        
        self.back_map_x[self.back_mask] = self.cx + X_back/r_back * r_fisheye_back
        self.back_map_y[self.back_mask] = self.cy + Y_back/r_back * r_fisheye_back
        
        # Create normalized mapping for grid_sample
        h, w = img_height, img_width
        self.front_norm_x = 2 * self.front_map_x / w - 1
        self.front_norm_y = 2 * self.front_map_y / h - 1
        self.front_grid = torch.stack([self.front_norm_x, self.front_norm_y], dim=-1).unsqueeze(0)
        
        self.back_norm_x = 2 * self.back_map_x / w - 1
        self.back_norm_y = 2 * self.back_map_y / h - 1
        self.back_grid = torch.stack([self.back_norm_x, self.back_norm_y], dim=-1).unsqueeze(0)
        
        self.front_mask_torch = self.front_mask.unsqueeze(0).unsqueeze(0)
        self.maps_initialized = True
        self.get_logger().info("Mapping matrices initialized")
        
    def image_callback(self, front_msg, back_msg):
        """Process front and back images to create equirectangular image"""
        try:
            # Convert ROS Image messages to OpenCV images
            front_img = self.bridge.imgmsg_to_cv2(front_msg, "rgb8")
            back_img = self.bridge.imgmsg_to_cv2(back_msg, "rgb8")
            
            # Store images for potential reuse
            self.last_front_img = front_img.copy()
            self.last_back_img = back_img.copy()
            
            # Initialize mapping if needed
            if not self.maps_initialized:
                self.init_mapping(front_img.shape[0], front_img.shape[1])
            
            # Create equirectangular image
            start_time = self.get_clock().now()
            equirect_img = self.create_equirectangular(front_img, back_img)
            
            # In calibration mode, visualize the seam
            if self.get_parameter('calibration_mode').value:
                equirect_img = self.visualize_seam(equirect_img)
            
            # Check if calibration should be started
            if self.get_parameter('start_calibration').value and not self.calibrating:
                # Reset the parameter
                self.set_parameters([rclpy.parameter.Parameter(
                    'start_calibration', rclpy.Parameter.Type.BOOL, False
                )])
                # Start calibration in a separate thread
                self.start_calibration()
            
            # Publish result
            equirect_msg = self.bridge.cv2_to_imgmsg(equirect_img, encoding="rgb8")
            equirect_msg.header = front_msg.header
            self.equirect_pub.publish(equirect_msg)
            
            # Log processing time
            process_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            self.get_logger().debug(f"Processing time: {process_time:.3f} seconds")
            
        except Exception as e:
            self.get_logger().error(f"Error processing images: {str(e)}")
    
    def create_equirectangular(self, front_img, back_img):
        """Create equirectangular image from front and back fisheye images"""
        if self.use_cuda:
            # GPU implementation using PyTorch
            # Convert images to PyTorch tensors
            front_tensor = torch.from_numpy(front_img).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
            back_tensor = torch.from_numpy(back_img).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
            
            # Apply the mapping using grid_sample
            front_result = torch.nn.functional.grid_sample(
                front_tensor, 
                self.front_grid,
                mode='bicubic',
                padding_mode='zeros',
                align_corners=True
            )
            
            back_result = torch.nn.functional.grid_sample(
                back_tensor,
                self.back_grid,
                mode='bicubic',
                padding_mode='zeros',
                align_corners=True
            )
            
            # Combine results based on masks
            result = torch.where(self.front_mask_torch.expand_as(front_result), front_result, back_result)
            
            # Convert back to numpy
            equirect = result.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            return equirect
        else:
            # CPU fallback using OpenCV
            front_map_x = self.front_map_x.cpu().numpy().astype(np.float32)
            front_map_y = self.front_map_y.cpu().numpy().astype(np.float32)
            back_map_x = self.back_map_x.cpu().numpy().astype(np.float32)
            back_map_y = self.back_map_y.cpu().numpy().astype(np.float32)
            
            # Create front and back portions
            front_result = cv2.remap(front_img, front_map_x, front_map_y, 
                                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            back_result = cv2.remap(back_img, back_map_x, back_map_y,
                                   cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            
            # Combine using masks
            equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
            front_mask_np = self.front_mask.cpu().numpy()
            equirect[front_mask_np] = front_result[front_mask_np]
            equirect[~front_mask_np] = back_result[~front_mask_np]
            
            return equirect
        
    def visualize_seam(self, equirect_img):
        """Highlight the seam between front and back cameras for visualization"""
        result = equirect_img.copy()
        
        # Get front/back mask as numpy array
        front_mask_np = self.front_mask.cpu().numpy()
        
        # Dilate and erode to get boundary
        kernel = np.ones((5, 5), np.uint8)
        front_dilated = cv2.dilate(front_mask_np.astype(np.uint8), kernel)
        front_eroded = cv2.erode(front_mask_np.astype(np.uint8), kernel)
        seam_mask = (front_dilated - front_eroded) > 0
        
        # Highlight the seam in red
        result[seam_mask] = [255, 0, 0]  # Red color
        
        return result

    def evaluate_stitch_quality(self):
        """Evaluate the quality of stitching at the seam (lower is better)"""
        if self.last_front_img is None or self.last_back_img is None:
            return float('inf')
            
        # Create equirectangular image
        equirect_img = self.create_equirectangular(self.last_front_img, self.last_back_img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(equirect_img, cv2.COLOR_RGB2GRAY)
        
        # Find the seam between front and back
        front_mask_np = self.front_mask.cpu().numpy()
        
        # Dilate and erode to get boundary
        kernel = np.ones((3, 3), np.uint8)
        front_dilated = cv2.dilate(front_mask_np.astype(np.uint8), kernel)
        front_eroded = cv2.erode(front_mask_np.astype(np.uint8), kernel)
        seam_mask = (front_dilated - front_eroded) > 0
        
        # Compute gradient magnitude at the seam
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient magnitude along the seam
        seam_grad = grad_mag[seam_mask]
        
        if len(seam_grad) == 0:
            return float('inf')
            
        score = np.mean(seam_grad)
        
        return score

    def start_calibration(self):
        """Start automatic calibration in a separate thread"""
        import threading
        
        if self.calibrating:
            self.get_logger().warn("Calibration already in progress")
            return
            
        if self.last_front_img is None or self.last_back_img is None:
            self.get_logger().error("No images available for calibration")
            return
            
        self.calibrating = True
        self.get_logger().info("Starting automatic calibration")
        
        # Reset progress
        self.set_parameters([rclpy.parameter.Parameter(
            'calibration_progress', rclpy.Parameter.Type.DOUBLE, 0.0
        )])
        
        # Create a thread to run the calibration
        self.calib_thread = threading.Thread(target=self.calibration_worker)
        self.calib_thread.daemon = True
        self.calib_thread.start()

    def calibration_worker(self):
        """Worker function for automatic calibration - translation parameters only"""
        try:
            import numpy as np
            
            # Store original parameters
            original_params = {
                'tx': self.get_parameter('tx').value,
                'ty': self.get_parameter('ty').value,
                'tz': self.get_parameter('tz').value
            }
            
            # Get calibration settings
            resolution = self.get_parameter('calib_resolution').value.lower()
            range_setting = self.get_parameter('calib_range').value.lower()
            use_grid_search = self.get_parameter('calib_grid_search').value
            
            self.get_logger().info("Translation-only calibration mode")
            if use_grid_search:
                self.get_logger().info("Using grid search (all combinations of tx, ty, tz)")
            else:
                self.get_logger().info("Using direct search (faster)")
            
            # Define step sizes based on resolution
            if resolution == 'high':
                coarse_trans_step = 0.001
                fine_trans_step = 0.0005
                self.get_logger().info("Using high resolution calibration (very precise but slow)")
            elif resolution == 'low':
                coarse_trans_step = 0.01
                fine_trans_step = 0.005
                self.get_logger().info("Using low resolution calibration (faster but less precise)")
            else:  # medium (default)
                coarse_trans_step = 0.005
                fine_trans_step = 0.002
                self.get_logger().info("Using medium resolution calibration (balanced)")
                
            # Define search ranges based on range setting
            if range_setting == 'very narrow':
                trans_range = (-0.01, 0.01)  # ±0.01 from current
                self.get_logger().info("Using very narrow search range (for precision fine-tuning)")
            elif range_setting == 'narrow':
                trans_range = (-0.05, 0.05)  # ±0.05 from current
                self.get_logger().info("Using narrow search range (for fine-tuning)")
            elif range_setting == 'wide':
                trans_range = (-0.3, 0.3)  # ±0.3 from current
                self.get_logger().info("Using wide search range (for initial calibration)")
            else:  # normal (default)
                trans_range = (-0.15, 0.15)  # ±0.15 from current
                self.get_logger().info("Using normal search range")
                
            # Define parameter bounds for translation
            tx_bounds = (original_params['tx'] + trans_range[0], original_params['tx'] + trans_range[1])
            ty_bounds = (original_params['ty'] + trans_range[0], original_params['ty'] + trans_range[1])
            tz_bounds = (original_params['tz'] + trans_range[0], original_params['tz'] + trans_range[1])
            
            # For sequential search (grid or direct)
            translation_ranges = {
                'tx': (tx_bounds[0], tx_bounds[1], coarse_trans_step),
                'ty': (ty_bounds[0], ty_bounds[1], coarse_trans_step),
                'tz': (tz_bounds[0], tz_bounds[1], coarse_trans_step)
            }
            
            param_bounds = {
                'tx': tx_bounds,
                'ty': ty_bounds,
                'tz': tz_bounds
            }
            
            # Initialize best parameters and score
            best_params = original_params.copy()
            best_score = self.evaluate_stitch_quality()
            self.get_logger().info(f"Initial score: {best_score:.4f}")
            
            # DEBUG: Show initial parameters
            self.get_logger().info(f"Initial parameters: tx={original_params['tx']}, ty={original_params['ty']}, tz={original_params['tz']}")
            
            # Phase 1: Parameter optimization
            if use_grid_search:
                # Grid search implementation
                # Generate parameter values with explicit steps
                tx_values = np.linspace(tx_bounds[0], tx_bounds[1], int((tx_bounds[1] - tx_bounds[0])/coarse_trans_step) + 1).tolist()
                ty_values = np.linspace(ty_bounds[0], ty_bounds[1], int((ty_bounds[1] - ty_bounds[0])/coarse_trans_step) + 1).tolist()
                tz_values = np.linspace(tz_bounds[0], tz_bounds[1], int((tz_bounds[1] - tz_bounds[0])/coarse_trans_step) + 1).tolist()
                
                # Calculate total combinations
                total_combinations = len(tx_values) * len(ty_values) * len(tz_values)
                self.get_logger().info(f"Testing {total_combinations} translation combinations")
                current_combination = 0
                
                # Try all combinations
                for tx in tx_values:
                    for ty in ty_values:
                        for tz in tz_values:
                            # Force all parameters to be set with explicit values (avoid potential floating point issues)
                            tx_value = float(tx)
                            ty_value = float(ty)
                            tz_value = float(tz)
                            
                            # Update all translation parameters at once
                            self.set_parameters([
                                rclpy.parameter.Parameter('tx', rclpy.Parameter.Type.DOUBLE, tx_value),
                                rclpy.parameter.Parameter('ty', rclpy.Parameter.Type.DOUBLE, ty_value),
                                rclpy.parameter.Parameter('tz', rclpy.Parameter.Type.DOUBLE, tz_value)
                            ])
                            self.update_camera_parameters()
                            
                            # Verify parameters were set correctly
                            actual_tx = self.get_parameter('tx').value
                            actual_ty = self.get_parameter('ty').value
                            actual_tz = self.get_parameter('tz').value
                            
                            if abs(actual_tx - tx_value) > 1e-6 or abs(actual_ty - ty_value) > 1e-6 or abs(actual_tz - tz_value) > 1e-6:
                                self.get_logger().warn(f"Parameter mismatch! Set: ({tx_value}, {ty_value}, {tz_value}), Got: ({actual_tx}, {actual_ty}, {actual_tz})")
                            
                            # Evaluate stitch quality
                            score = self.evaluate_stitch_quality()
                            
                            # Update best if better
                            if score < best_score:
                                best_score = score
                                best_params['tx'] = tx_value
                                best_params['ty'] = ty_value
                                best_params['tz'] = tz_value
                                self.get_logger().info(
                                    f"  New best: tx={tx_value:.4f}, ty={ty_value:.4f}, tz={tz_value:.4f}, score={score:.4f}"
                                )
                            
                            # Update progress
                            current_combination += 1
                            if current_combination % 10 == 0:  # Update less frequently to reduce overhead
                                progress = 50.0 * current_combination / total_combinations
                                self.set_parameters([rclpy.parameter.Parameter(
                                    'calibration_progress', rclpy.Parameter.Type.DOUBLE, progress
                                )])
            else:
                # Direct search - one parameter at a time with verbose output
                self.get_logger().info("Phase 1: Direct parameter optimization")
                
                # Count total steps for progress tracking
                total_steps = sum(int((r[1] - r[0]) / r[2]) + 1 for r in translation_ranges.values())
                current_step = 0
                
                # Iterate through translation parameters one at a time
                param_names = ['tx', 'ty', 'tz']
                for param_name in param_names:
                    min_val, max_val, step = translation_ranges[param_name]
                    self.get_logger().info(f"Optimizing {param_name}: {min_val:.4f} to {max_val:.4f} in {step} steps")
                    
                    # Create explicit list of values to try
                    test_values = np.linspace(min_val, max_val, int((max_val - min_val)/step) + 1).tolist()
                    
                    for value in test_values:
                        # Explicit float casting to avoid potential issues
                        param_value = float(value)
                        
                        # Update parameter
                        self.set_parameters([rclpy.parameter.Parameter(
                            param_name, rclpy.Parameter.Type.DOUBLE, param_value
                        )])
                        self.update_camera_parameters()
                        
                        # Verify parameter was set
                        actual_value = self.get_parameter(param_name).value
                        if abs(actual_value - param_value) > 1e-6:
                            self.get_logger().warn(f"{param_name} mismatch! Set: {param_value}, Got: {actual_value}")
                        
                        # Evaluate stitch quality
                        score = self.evaluate_stitch_quality()
                        
                        # Update best if better
                        if score < best_score:
                            best_score = score
                            best_params[param_name] = param_value
                            self.get_logger().info(f"  New best {param_name} = {param_value:.6f}, score = {score:.6f}")
                        
                        # Update progress
                        current_step += 1
                        progress = 50.0 * current_step / total_steps
                        self.set_parameters([rclpy.parameter.Parameter(
                            'calibration_progress', rclpy.Parameter.Type.DOUBLE, progress
                        )])
                    
                    # Set to best value for this parameter before moving to next
                    self.set_parameters([rclpy.parameter.Parameter(
                        param_name, rclpy.Parameter.Type.DOUBLE, best_params[param_name]
                    )])
                    self.update_camera_parameters()
                    self.get_logger().info(f"  Best {param_name} set to {best_params[param_name]}")
            
            # Ensure best parameters from phase 1 are applied
            self.get_logger().info("Applying best parameters from Phase 1")
            for param_name, value in best_params.items():
                self.set_parameters([rclpy.parameter.Parameter(
                    param_name, rclpy.Parameter.Type.DOUBLE, value
                )])
            self.update_camera_parameters()
            
            # Phase 2: Fine-tuning (simplified to ensure it works)
            self.get_logger().info("Phase 2: Fine-tuning with pattern search")
            
            # Smaller steps for fine-tuning
            param_names = ['tx', 'ty', 'tz']
            
            # Multiple iterations of pattern search
            for iteration in range(10):
                improved = False
                
                # Try each parameter
                for param_name in param_names:
                    current_value = self.get_parameter(param_name).value
                    
                    # Try positive direction
                    test_value = current_value + fine_trans_step
                    self.set_parameters([rclpy.parameter.Parameter(
                        param_name, rclpy.Parameter.Type.DOUBLE, test_value
                    )])
                    self.update_camera_parameters()
                    score_plus = self.evaluate_stitch_quality()
                    
                    # Try negative direction
                    test_value = current_value - fine_trans_step
                    self.set_parameters([rclpy.parameter.Parameter(
                        param_name, rclpy.Parameter.Type.DOUBLE, test_value
                    )])
                    self.update_camera_parameters()
                    score_minus = self.evaluate_stitch_quality()
                    
                    # Find best direction
                    best_direction = 0
                    if score_plus < best_score and score_plus <= score_minus:
                        best_score = score_plus
                        best_params[param_name] = current_value + fine_trans_step
                        best_direction = 1
                        improved = True
                    elif score_minus < best_score:
                        best_score = score_minus
                        best_params[param_name] = current_value - fine_trans_step
                        best_direction = -1
                        improved = True
                    
                    # Set best value
                    if best_direction != 0:
                        self.set_parameters([rclpy.parameter.Parameter(
                            param_name, rclpy.Parameter.Type.DOUBLE, best_params[param_name]
                        )])
                        self.update_camera_parameters()
                        self.get_logger().info(f"  Fine-tune: {param_name} = {best_params[param_name]:.6f}, score = {best_score:.6f}")
                
                # Update progress
                progress = 50.0 + 50.0 * (iteration + 1) / 10
                self.set_parameters([rclpy.parameter.Parameter(
                    'calibration_progress', rclpy.Parameter.Type.DOUBLE, progress
                )])
                
                # Stop if no improvement
                if not improved:
                    self.get_logger().info("No further improvement found, stopping fine-tuning")
                    break
            
            # Final update with best parameters
            self.get_logger().info("Setting final best parameters")
            for param_name, value in best_params.items():
                self.set_parameters([rclpy.parameter.Parameter(
                    param_name, rclpy.Parameter.Type.DOUBLE, value
                )])
            self.update_camera_parameters()
            
            # Compare initial vs final
            improvement = original_params['tx'] != best_params['tx'] or \
                         original_params['ty'] != best_params['ty'] or \
                         original_params['tz'] != best_params['tz']
            
            if improvement:
                self.get_logger().info("Calibration complete. Parameters changed:")
                self.get_logger().info(f"  tx: {original_params['tx']:.6f} → {best_params['tx']:.6f}")
                self.get_logger().info(f"  ty: {original_params['ty']:.6f} → {best_params['ty']:.6f}")
                self.get_logger().info(f"  tz: {original_params['tz']:.6f} → {best_params['tz']:.6f}")
                self.get_logger().info(f"  score: {self.evaluate_stitch_quality():.6f} (improved by {self.evaluate_stitch_quality() - best_score:.6f})")
            else:
                self.get_logger().warn("Calibration complete, but parameters did not change! Current values may already be optimal.")
            
            # Set progress to 100%
            self.set_parameters([rclpy.parameter.Parameter(
                'calibration_progress', rclpy.Parameter.Type.DOUBLE, 100.0
            )])
            
            # Save calibration results
            self.save_calibration(best_params)
            
        except Exception as e:
            self.get_logger().error(f"Calibration error: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.calibrating = False
    
    def save_calibration(self, params):
        """Save calibration results to a file"""
        try:
            import json
            import os
            
            # Create calibration directory if it doesn't exist
            calib_dir = os.path.join(os.path.expanduser('~'), '.ros', 'insta360_calib')
            os.makedirs(calib_dir, exist_ok=True)
            
            # Save to JSON file
            calib_file = os.path.join(calib_dir, 'calibration.json')
            with open(calib_file, 'w') as f:
                json.dump(params, f, indent=4)
                
            self.get_logger().info(f"Calibration saved to {calib_file}")
        except Exception as e:
            self.get_logger().error(f"Error saving calibration: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = EquirectangularNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()