#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import math
import argparse
import json
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dual fisheye videos to equirectangular format')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing front/back video pairs')
    parser.add_argument('--output', type=str, required=True, help='Output folder for equirectangular videos')
    parser.add_argument('--width', type=int, default=1920, help='Width of output equirectangular video')
    parser.add_argument('--height', type=int, default=960, help='Height of output equirectangular video')
    parser.add_argument('--crop_size', type=int, default=1080, help='Size to crop input frames to (square)')
    parser.add_argument('--calibration', type=str, help='Path to calibration JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--preview', action='store_true', help='Display preview window of processed frames')
    return parser.parse_args()


class EquirectangularConverter:
    def __init__(self, out_width=1920, out_height=960, use_gpu=False):
        # Default camera parameters (from the ROS node)
        self.cx = None  # Will be set to input_width/2
        self.cy = None  # Will be set to input_height/2
        self.cx_offset = 0
        self.cy_offset = 0
        self.out_width = out_width
        self.out_height = out_height
        
        # GPU flag
        self.use_gpu = use_gpu
        if use_gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available. Falling back to CPU.")
            self.use_gpu = False
            
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU for processing")
        
        # Extrinsic parameters
        self.roll = math.radians(0.0)
        self.pitch = math.radians(0.0)
        self.yaw = math.radians(0.0)
        
        # Update rotation matrices
        self.update_rotation_matrices()
        
        # Translation vector
        self.back_to_front_translation = np.array([0.0, 0.0, -0.180])
        if self.use_gpu:
            self.back_to_front_translation_gpu = torch.tensor(self.back_to_front_translation, 
                                                           device=self.device, dtype=torch.float32)
        
        # Flag to track if mapping is initialized
        self.maps_initialized = False
        self.img_height = None
        self.img_width = None
    
    def update_rotation_matrices(self):
        # Create rotation matrix from Euler angles
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(self.roll), -math.sin(self.roll)],
            [0.0, math.sin(self.roll), math.cos(self.roll)]
        ])
        
        Ry = np.array([
            [math.cos(self.pitch), 0.0, math.sin(self.pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(self.pitch), 0.0, math.cos(self.pitch)]
        ])
        
        Rz = np.array([
            [math.cos(self.yaw), -math.sin(self.yaw), 0.0],
            [math.sin(self.yaw), math.cos(self.yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Combined rotation matrix
        self.back_to_front_rotation = Rz @ Ry @ Rx
        
        # Create GPU version if needed
        if self.use_gpu:
            self.back_to_front_rotation_gpu = torch.tensor(self.back_to_front_rotation, 
                                                        device=self.device, dtype=torch.float32)
    
    def init_mapping(self, img_height, img_width):
        """Initialize mapping matrices for equirectangular projection"""
        print(f"Initializing mapping matrices for {self.out_width}x{self.out_height} on {'GPU' if self.use_gpu else 'CPU'}")
        
        # Store dimensions for future use
        self.img_height = img_height
        self.img_width = img_width
        
        # Set cx and cy to the center of the input image plus offset
        if self.cx is None or self.cy is None:
            self.cx = img_width / 2 + self.cx_offset
            self.cy = img_height / 2 + self.cy_offset
            print(f"Setting camera center to ({self.cx}, {self.cy})")
        
        # Create coordinate grid for the output equirectangular image
        if self.use_gpu:
            # Create on GPU using PyTorch
            h_range = torch.arange(0, self.out_height, device=self.device, dtype=torch.float32)
            w_range = torch.arange(0, self.out_width, device=self.device, dtype=torch.float32)
            Y, X = torch.meshgrid(h_range, w_range, indexing='ij')
            
            # Convert to normalized coordinates
            longitude = (X / self.out_width) * 2 * math.pi - math.pi
            latitude = (Y / self.out_height) * math.pi - math.pi/2
            
            # Convert to 3D points on unit sphere
            X_sphere = torch.cos(latitude) * torch.sin(longitude)
            Y_sphere = torch.sin(latitude)
            Z_sphere = torch.cos(latitude) * torch.cos(longitude)
            
            # Create front and back masks
            self.front_mask_gpu = (Z_sphere >= 0)
            self.back_mask_gpu = (Z_sphere < 0)
            
            # For OpenCV compatibility, also create numpy versions
            self.front_mask = self.front_mask_gpu.cpu().numpy()
            self.back_mask = ~self.front_mask
            
            # Calculate mapping for front camera
            r_front = torch.sqrt(X_sphere[self.front_mask_gpu]**2 + Y_sphere[self.front_mask_gpu]**2)
            # Avoid division by zero
            r_front = torch.clamp(r_front, min=1e-6)
            theta_front = torch.atan2(r_front, torch.abs(Z_sphere[self.front_mask_gpu]))
            r_fisheye_front = 2 * theta_front / math.pi * (img_width / 2)
            
            # Initialize map arrays
            self.front_map_x = torch.zeros((self.out_height, self.out_width), device=self.device, dtype=torch.float32)
            self.front_map_y = torch.zeros((self.out_height, self.out_width), device=self.device, dtype=torch.float32)
            
            self.front_map_x[self.front_mask_gpu] = self.cx + X_sphere[self.front_mask_gpu]/r_front * r_fisheye_front
            self.front_map_y[self.front_mask_gpu] = self.cy + Y_sphere[self.front_mask_gpu]/r_front * r_fisheye_front
            
            # Calculate mapping for back camera
            back_X = X_sphere[self.back_mask_gpu]
            back_Y = Y_sphere[self.back_mask_gpu]
            back_Z = Z_sphere[self.back_mask_gpu]
            
            # Stack to form 3D points
            back_points = torch.stack([back_X, back_Y, back_Z], dim=1)
            
            # Apply rotation from back to front camera frame
            transformed_points = torch.matmul(back_points, self.back_to_front_rotation_gpu.T)
            
            # Apply translation vector
            transformed_points = transformed_points + self.back_to_front_translation_gpu
            
            # Extract transformed coordinates
            X_back = -transformed_points[:, 0]  # Negate X for back camera view
            Y_back = transformed_points[:, 1]
            Z_back = transformed_points[:, 2]
            
            # Continue with back camera mapping using transformed points
            r_back = torch.sqrt(X_back**2 + Y_back**2)
            r_back = torch.clamp(r_back, min=1e-6)
            theta_back = torch.atan2(r_back, torch.abs(Z_back))
            r_fisheye_back = 2 * theta_back / math.pi * (img_width / 2)
            
            # Initialize back map arrays
            self.back_map_x = torch.zeros((self.out_height, self.out_width), device=self.device, dtype=torch.float32)
            self.back_map_y = torch.zeros((self.out_height, self.out_width), device=self.device, dtype=torch.float32)
            
            self.back_map_x[self.back_mask_gpu] = self.cx + X_back/r_back * r_fisheye_back
            self.back_map_y[self.back_mask_gpu] = self.cy + Y_back/r_back * r_fisheye_back
            
            # Convert to numpy arrays for OpenCV
            self.front_map_x_np = self.front_map_x.cpu().numpy().astype(np.float32)
            self.front_map_y_np = self.front_map_y.cpu().numpy().astype(np.float32)
            self.back_map_x_np = self.back_map_x.cpu().numpy().astype(np.float32)
            self.back_map_y_np = self.back_map_y.cpu().numpy().astype(np.float32)
        else:
            # Create on CPU using NumPy (original code)
            y, x = np.mgrid[0:self.out_height, 0:self.out_width]
            
            # Convert to normalized coordinates
            longitude = (x / self.out_width) * 2 * math.pi - math.pi
            latitude = (y / self.out_height) * math.pi - math.pi/2
            
            # Convert to 3D points on unit sphere
            X = np.cos(latitude) * np.sin(longitude)
            Y = np.sin(latitude)
            Z = np.cos(latitude) * np.cos(longitude)
            
            # Create front and back masks
            self.front_mask = (Z >= 0)
            self.back_mask = (Z < 0)
            
            # Calculate mapping for front camera
            r_front = np.sqrt(X[self.front_mask]**2 + Y[self.front_mask]**2)
            # Avoid division by zero
            r_front = np.maximum(r_front, 1e-6)
            theta_front = np.arctan2(r_front, np.abs(Z[self.front_mask]))
            r_fisheye_front = 2 * theta_front / math.pi * (img_width / 2)
            
            # Initialize map arrays
            self.front_map_x_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
            self.front_map_y_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
            
            self.front_map_x_np[self.front_mask] = self.cx + X[self.front_mask]/r_front * r_fisheye_front
            self.front_map_y_np[self.front_mask] = self.cy + Y[self.front_mask]/r_front * r_fisheye_front
            
            # Calculate mapping for back camera
            back_X = X[self.back_mask]
            back_Y = Y[self.back_mask]
            back_Z = Z[self.back_mask]
            
            # Stack to form 3D points
            back_points = np.column_stack([back_X, back_Y, back_Z])
            
            # Apply rotation from back to front camera frame
            transformed_points = np.dot(back_points, self.back_to_front_rotation.T)
            
            # Apply translation vector
            transformed_points = transformed_points + self.back_to_front_translation
            
            # Extract transformed coordinates
            X_back = -transformed_points[:, 0]  # Negate X for back camera view
            Y_back = transformed_points[:, 1]
            Z_back = transformed_points[:, 2]
            
            # Continue with back camera mapping using transformed points
            r_back = np.sqrt(X_back**2 + Y_back**2)
            r_back = np.maximum(r_back, 1e-6)
            theta_back = np.arctan2(r_back, np.abs(Z_back))
            r_fisheye_back = 2 * theta_back / math.pi * (img_width / 2)
            
            # Initialize back map arrays
            self.back_map_x_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
            self.back_map_y_np = np.zeros((self.out_height, self.out_width), dtype=np.float32)
            
            self.back_map_x_np[self.back_mask] = self.cx + X_back/r_back * r_fisheye_back
            self.back_map_y_np[self.back_mask] = self.cy + Y_back/r_back * r_fisheye_back
        
        self.maps_initialized = True
        print("Mapping matrices initialized")
        
    def create_equirectangular(self, front_img, back_img):
        """Create equirectangular image from front and back fisheye images"""
        # Check if we need to initialize the mapping
        if not self.maps_initialized:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
        
        if self.use_gpu:
            # Convert images to GPU tensors
            front_tensor = torch.from_numpy(front_img).to(self.device)
            back_tensor = torch.from_numpy(back_img).to(self.device)
            
            # On GPU, we need to use OpenCV's remap on CPU
            # PyTorch doesn't have a direct equivalent to cv2.remap
            front_result = cv2.remap(front_img, self.front_map_x_np, self.front_map_y_np, 
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            back_result = cv2.remap(back_img, self.back_map_x_np, self.back_map_y_np,
                            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            
            # Combine using masks - still using numpy for simplicity
            equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
            equirect[self.front_mask] = front_result[self.front_mask]
            equirect[self.back_mask] = back_result[self.back_mask]
        else:
            # Original CPU implementation
            front_result = cv2.remap(front_img, self.front_map_x_np, self.front_map_y_np, 
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            back_result = cv2.remap(back_img, self.back_map_x_np, self.back_map_y_np,
                            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            
            # Combine using masks
            equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
            equirect[self.front_mask] = front_result[self.front_mask]
            equirect[self.back_mask] = back_result[self.back_mask]
        
        return equirect


def load_calibration(calibration_file):
    """Load calibration parameters from JSON file"""
    if not os.path.isfile(calibration_file):
        print(f"Warning: Calibration file not found: {calibration_file}")
        return None
    
    try:
        with open(calibration_file, 'r') as f:
            params = json.load(f)
        
        print(f"Loaded calibration parameters from {calibration_file}")
        print(f"  Crop size: {params['crop_size']}")
        print(f"  Center offset: ({params['cx_offset']}, {params['cy_offset']})")
        print(f"  Translation: {params['translation']}")
        print(f"  Rotation (deg): {params['rotation_deg']}")
        
        return params
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return None


def apply_calibration(converter, params):
    """Apply calibration parameters to converter instance"""
    if not params:
        return
    
    # Set camera center offsets
    converter.cx_offset = params['cx_offset']
    converter.cy_offset = params['cy_offset']
    
    # Set translation vector
    tx, ty, tz = params['translation']
    converter.back_to_front_translation = np.array([tx, ty, tz])
    if converter.use_gpu:
        converter.back_to_front_translation_gpu = torch.tensor(converter.back_to_front_translation, 
                                                           device=converter.device, dtype=torch.float32)
    
    # Set rotation parameters
    roll_deg, pitch_deg, yaw_deg = params['rotation_deg']
    converter.roll = math.radians(roll_deg)
    converter.pitch = math.radians(pitch_deg)
    converter.yaw = math.radians(yaw_deg)
    
    # Update rotation matrices
    converter.update_rotation_matrices()
    
    # Reset mapping to force reinitialization
    converter.maps_initialized = False


def find_video_pairs(input_folder):
    """Find front and back video pairs in the input folder"""
    videos = {}
    
    # Get all video files in the directory
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.AVI'))]
    
    # Group by base name
    for video_file in video_files:
        base_name = video_file.split('_')[0]
        cam_type = 'front' if 'front' in video_file.lower() else 'back' if 'back' in video_file.lower() else None
        
        if cam_type and base_name:
            if base_name not in videos:
                videos[base_name] = {}
            videos[base_name][cam_type] = os.path.join(input_folder, video_file)
    
    # Filter to ensure we only have pairs with both front and back
    return {k: v for k, v in videos.items() if 'front' in v and 'back' in v}


def convert_videos(input_folder, output_folder, width=1920, height=960, crop_size=1080, 
                   calibration_file=None, use_gpu=False, preview=False):
    """Convert all video pairs to equirectangular format"""
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find video pairs
    video_pairs = find_video_pairs(input_folder)
    print(f"Found {len(video_pairs)} video pairs to process")
    
    # Load calibration parameters if available
    calibration_params = None
    if calibration_file:
        calibration_params = load_calibration(calibration_file)
        if calibration_params:
            # Use crop size from calibration if available
            crop_size = calibration_params.get('crop_size', crop_size)
    
    # Create converter instance
    converter = EquirectangularConverter(out_width=width, out_height=height, use_gpu=use_gpu)
    
    # Apply calibration parameters if loaded
    if calibration_params:
        apply_calibration(converter, calibration_params)
    
    # Create preview window if needed
    if preview:
        cv2.namedWindow("Equirectangular Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Equirectangular Preview", width // 2, height // 2)
        print("Preview window enabled. Press 'q' to quit, any other key to continue.")
    
    # Process each pair
    for base_name, videos in video_pairs.items():
        front_path = videos['front']
        back_path = videos['back']
        output_path = os.path.join(output_folder, f"{base_name}_equirect.mp4")
        
        print(f"Processing {base_name}...")
        front_cap = cv2.VideoCapture(front_path)
        back_cap = cv2.VideoCapture(back_path)
        
        # Check if videos were opened successfully
        if not front_cap.isOpened() or not back_cap.isOpened():
            print(f"Error opening video files for {base_name}")
            continue
        
        # Get video properties
        front_fps = front_cap.get(cv2.CAP_PROP_FPS)
        front_frame_count = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, front_fps, (width, height))
        
        # Process frames
        with tqdm(total=front_frame_count, desc=f"Converting {base_name}") as pbar:
            while front_cap.isOpened() and back_cap.isOpened():
                front_ret, front_frame = front_cap.read()
                back_ret, back_frame = back_cap.read()
                
                if not front_ret or not back_ret:
                    break
                
                # Crop frames from original size to crop_size x crop_size (centered)
                if front_frame.shape[0] == 1280 and front_frame.shape[1] == 1280:
                    y_start = (1280 - crop_size) // 2
                    x_start = (1280 - crop_size) // 2
                    front_frame = front_frame[y_start:y_start+crop_size, x_start:x_start+crop_size]
                    back_frame = back_frame[y_start:y_start+crop_size, x_start:x_start+crop_size]
                
                # Convert to RGB (OpenCV uses BGR)
                front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
                back_rgb = cv2.cvtColor(back_frame, cv2.COLOR_BGR2RGB)
                
                # Create equirectangular frame
                equirect_rgb = converter.create_equirectangular(front_rgb, back_rgb)
                
                # Convert back to BGR for OpenCV
                equirect_bgr = cv2.cvtColor(equirect_rgb, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out_writer.write(equirect_bgr)
                
                # Show preview if enabled
                if preview:
                    # Display current frame number on preview
                    frame_with_info = equirect_bgr.copy()
                    cv2.putText(
                        frame_with_info,
                        f"Frame: {pbar.n+1}/{front_frame_count} | {base_name}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow("Equirectangular Preview", frame_with_info)
                    key = cv2.waitKey(1)
                    if key == ord('q'):  # Quit on 'q'
                        print("Preview terminated by user")
                        front_cap.release()
                        back_cap.release()
                        out_writer.release()
                        cv2.destroyAllWindows()
                        return
                
                # Update progress bar
                pbar.update(1)
        
        # Release resources
        front_cap.release()
        back_cap.release()
        out_writer.release()
        
        print(f"Saved equirectangular video to {output_path}")
    
    print("All videos processed")
    
    # Clean up preview window
    if preview:
        cv2.destroyAllWindows()
    
    # Free GPU memory if used
    if use_gpu:
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    convert_videos(
        args.input, 
        args.output, 
        args.width, 
        args.height, 
        args.crop_size,
        args.calibration,
        args.gpu,
        args.preview
    )


if __name__ == "__main__":
    main()