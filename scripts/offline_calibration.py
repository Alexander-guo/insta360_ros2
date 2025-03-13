#!/usr/bin/env python3

import os
import cv2
import numpy as np
import math
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate dual fisheye to equirectangular conversion parameters')
    parser.add_argument('--front', type=str, required=True, help='Front fisheye video')
    parser.add_argument('--back', type=str, required=True, help='Back fisheye video')
    parser.add_argument('--width', type=int, default=1920, help='Width of output equirectangular video')
    parser.add_argument('--height', type=int, default=960, help='Height of output equirectangular video')
    parser.add_argument('--output', type=str, default='calibration.json', help='Output file for calibration parameters')
    return parser.parse_args()


class CalibrationGUI:
    def __init__(self, front_video, back_video, out_width=1920, out_height=960):
        # Window names
        self.window_name = "Equirectangular Calibration"
        self.control_window = "Calibration Controls"
        
        # Video properties
        self.front_video = front_video
        self.back_video = back_video
        self.out_width = out_width
        self.out_height = out_height
        
        # Initialize video captures
        self.front_cap = cv2.VideoCapture(front_video)
        self.back_cap = cv2.VideoCapture(back_video)
        
        if not self.front_cap.isOpened() or not self.back_cap.isOpened():
            raise ValueError("Could not open video files")
            
        # Get frame properties
        self.frame_count = min(
            int(self.front_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self.back_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.fps = self.front_cap.get(cv2.CAP_PROP_FPS)
        
        # Read first frame to get dimensions
        ret, first_frame = self.front_cap.read()
        if not ret:
            raise ValueError("Could not read frame from front video")
        self.input_height, self.input_width = first_frame.shape[:2]
        
        # Reset video captures
        self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.back_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Current frame index
        self.current_frame = 0
        
        # Default parameters (can be adjusted via GUI)
        self.cx_offset = 0  # Offset from center
        self.cy_offset = 0  # Offset from center
        self.crop_size = 1080
        self.tx = 0.03
        self.ty = 0.0
        self.tz = -0.129
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Frame storage
        self.front_frame = None
        self.back_frame = None
        self.equirect_frame = None
        
        # Play/pause state
        self.playing = True
        
        # Create windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, out_width // 2, out_height // 2)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        
        # Create trackbars
        self.create_trackbars()
        
    def create_trackbars(self):
        """Create trackbars for parameter adjustment"""
        # Camera center offsets
        cv2.createTrackbar("CX Offset", self.control_window, 0, 200, self.update_cx)
        cv2.setTrackbarPos("CX Offset", self.control_window, 100)  # Set to middle (0 offset)
        
        cv2.createTrackbar("CY Offset", self.control_window, 0, 200, self.update_cy)
        cv2.setTrackbarPos("CY Offset", self.control_window, 100)  # Set to middle (0 offset)
        
        # Crop size
        cv2.createTrackbar("Crop Size", self.control_window, 960, 1280, self.update_crop)
        cv2.setTrackbarPos("Crop Size", self.control_window, self.crop_size)
        
        # Translation parameters (x100 for UI precision)
        cv2.createTrackbar("TX (x100)", self.control_window, 0, 100, self.update_tx)
        cv2.setTrackbarPos("TX (x100)", self.control_window, int(self.tx * 100) + 50)
        
        cv2.createTrackbar("TY (x100)", self.control_window, 0, 100, self.update_ty)
        cv2.setTrackbarPos("TY (x100)", self.control_window, int(self.ty * 100) + 50)
        
        cv2.createTrackbar("TZ (x100)", self.control_window, 0, 100, self.update_tz)
        cv2.setTrackbarPos("TZ (x100)", self.control_window, int(self.tz * 100) + 50)
        
        # Rotation parameters (in degrees for UI)
        cv2.createTrackbar("Roll (deg)", self.control_window, 0, 360, self.update_roll)
        cv2.setTrackbarPos("Roll (deg)", self.control_window, 180)  # 180 = 0 degrees
        
        cv2.createTrackbar("Pitch (deg)", self.control_window, 0, 360, self.update_pitch)
        cv2.setTrackbarPos("Pitch (deg)", self.control_window, 180)  # 180 = 0 degrees
        
        cv2.createTrackbar("Yaw (deg)", self.control_window, 0, 360, self.update_yaw)
        cv2.setTrackbarPos("Yaw (deg)", self.control_window, 180)  # 180 = 0 degrees
        
        # Frame control
        cv2.createTrackbar("Frame", self.control_window, 0, max(1, self.frame_count-1), self.jump_to_frame)
        
    # Trackbar update callbacks
    def update_cx(self, value):
        self.cx_offset = (value - 100) * 2  # -200 to 200 range
        self.update_view()
        
    def update_cy(self, value):
        self.cy_offset = (value - 100) * 2  # -200 to 200 range
        self.update_view()
        
    def update_crop(self, value):
        self.crop_size = value
        self.update_view()
        
    def update_tx(self, value):
        self.tx = (value - 50) / 100.0  # -0.5 to 0.5 range
        self.update_view()
        
    def update_ty(self, value):
        self.ty = (value - 50) / 100.0  # -0.5 to 0.5 range
        self.update_view()
        
    def update_tz(self, value):
        self.tz = (value - 50) / 100.0  # -0.5 to 0.5 range
        self.update_view()
        
    def update_roll(self, value):
        self.roll = (value - 180) * math.pi / 180.0  # Convert to radians
        self.update_view()
        
    def update_pitch(self, value):
        self.pitch = (value - 180) * math.pi / 180.0  # Convert to radians
        self.update_view()
        
    def update_yaw(self, value):
        self.yaw = (value - 180) * math.pi / 180.0  # Convert to radians
        self.update_view()
        
    def jump_to_frame(self, frame_idx):
        if frame_idx != self.current_frame:
            self.current_frame = frame_idx
            self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.back_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.read_frames()
            self.update_view()
            
    def read_frames(self):
        """Read current frames from both videos"""
        front_ret, front_frame = self.front_cap.read()
        back_ret, back_frame = self.back_cap.read()
        
        if not front_ret or not back_ret:
            # Loop back to beginning if we reached the end
            self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.back_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            cv2.setTrackbarPos("Frame", self.control_window, 0)
            front_ret, front_frame = self.front_cap.read()
            back_ret, back_frame = self.back_cap.read()
            
        # Crop frames to square
        if front_frame.shape[0] == 1280 and front_frame.shape[1] == 1280:
            y_start = (1280 - self.crop_size) // 2
            x_start = (1280 - self.crop_size) // 2
            self.front_frame = front_frame[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]
            self.back_frame = back_frame[y_start:y_start+self.crop_size, x_start:x_start+self.crop_size]
        else:
            self.front_frame = front_frame
            self.back_frame = back_frame
            
        return True
            
    def update_view(self):
        """Update the equirectangular view with current parameters"""
        if self.front_frame is None or self.back_frame is None:
            return
            
        # Create equirectangular converter with current parameters
        converter = self.create_converter()
        
        # Convert to RGB (OpenCV uses BGR)
        front_rgb = cv2.cvtColor(self.front_frame, cv2.COLOR_BGR2RGB)
        back_rgb = cv2.cvtColor(self.back_frame, cv2.COLOR_BGR2RGB)
        
        # Create equirectangular frame
        equirect_rgb = converter.create_equirectangular(front_rgb, back_rgb)
        
        # Convert back to BGR for OpenCV
        self.equirect_frame = cv2.cvtColor(equirect_rgb, cv2.COLOR_RGB2BGR)
        
        # Display current frame and parameters
        info_text = (
            f"Frame: {self.current_frame}/{self.frame_count} | "
            f"cx: {self.crop_size/2 + self.cx_offset:.1f}, cy: {self.crop_size/2 + self.cy_offset:.1f} | "
            f"crop: {self.crop_size} | "
            f"t: [{self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}] | "
            f"r: [{math.degrees(self.roll):.1f}, {math.degrees(self.pitch):.1f}, {math.degrees(self.yaw):.1f}]"
        )
        
        # Add info text to frame
        frame_with_info = self.equirect_frame.copy()
        cv2.putText(
            frame_with_info, 
            info_text,
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow(self.window_name, frame_with_info)
        
    def create_converter(self):
        """Create an equirectangular converter with the current parameters"""
        converter = EquirectangularConverter(out_width=self.out_width, out_height=self.out_height)
        
        # Set camera center with offset
        converter.cx = self.crop_size / 2 + self.cx_offset
        converter.cy = self.crop_size / 2 + self.cy_offset
        
        # Set translation vector
        converter.back_to_front_translation = np.array([self.tx, self.ty, self.tz])
        
        # Set rotation matrices
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
        converter.back_to_front_rotation = Rz @ Ry @ Rx
        
        return converter
        
    def run(self):
        """Main loop for the GUI application"""
        # Initial frame read
        if not self.read_frames():
            print("Failed to read initial frames")
            return False
        
        self.update_view()
        last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        while True:
            if self.playing:
                # Calculate appropriate delay to maintain video FPS
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed = current_time - last_frame_time
                wait_time = max(1, int((1.0/self.fps - elapsed) * 1000))
            else:
                wait_time = 100  # Longer wait when paused
                
            key = cv2.waitKey(wait_time)
            
            # Process key presses
            if key == 27:  # ESC key
                break
            elif key == ord(' '):  # Space bar - toggle play/pause
                self.playing = not self.playing
            elif key == ord('s'):  # 's' key - save parameters
                self.save_parameters()
            
            if self.playing:
                # Advance to next frame
                self.current_frame += 1
                cv2.setTrackbarPos("Frame", self.control_window, self.current_frame)
                if not self.read_frames():
                    break
                self.update_view()
                last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
                
        # Cleanup
        self.front_cap.release()
        self.back_cap.release()
        cv2.destroyAllWindows()
        return True
        
    def save_parameters(self, filename='calibration.json'):
        """Save current calibration parameters to a JSON file"""
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
        
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)
            
        print(f"Parameters saved to {filename}")


class EquirectangularConverter:
    def __init__(self, out_width=1920, out_height=960):
        # Default camera parameters (will be set by calibration)
        self.cx = None
        self.cy = None
        self.out_width = out_width
        self.out_height = out_height
        
        # Rotation matrix and translation vector (will be set by calibration)
        self.back_to_front_rotation = np.eye(3)
        self.back_to_front_translation = np.zeros(3)
        
        # Flag to track if mapping is initialized
        self.maps_initialized = False
        self.img_height = None
        self.img_width = None
    
    def init_mapping(self, img_height, img_width):
        """Initialize mapping matrices for equirectangular projection"""
        # Store dimensions for future use
        self.img_height = img_height
        self.img_width = img_width
        
        # Set cx and cy if not already set
        if self.cx is None or self.cy is None:
            self.cx = img_width / 2
            self.cy = img_height / 2
        
        # Create coordinate grid for the output equirectangular image
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
        self.front_map_x = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.front_map_y = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        
        self.front_map_x[self.front_mask] = self.cx + X[self.front_mask]/r_front * r_fisheye_front
        self.front_map_y[self.front_mask] = self.cy + Y[self.front_mask]/r_front * r_fisheye_front
        
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
        self.back_map_x = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        self.back_map_y = np.zeros((self.out_height, self.out_width), dtype=np.float32)
        
        self.back_map_x[self.back_mask] = self.cx + X_back/r_back * r_fisheye_back
        self.back_map_y[self.back_mask] = self.cy + Y_back/r_back * r_fisheye_back
        
        self.maps_initialized = True
        
    def create_equirectangular(self, front_img, back_img):
        """Create equirectangular image from front and back fisheye images"""
        # Check if we need to initialize the mapping
        if not self.maps_initialized:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
        # Reinitialize if the image dimensions have changed
        elif front_img.shape[0] != self.img_height or front_img.shape[1] != self.img_width:
            self.init_mapping(front_img.shape[0], front_img.shape[1])
            
        # Create front and back portions using OpenCV remap
        front_result = cv2.remap(front_img, self.front_map_x, self.front_map_y, 
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        back_result = cv2.remap(back_img, self.back_map_x, self.back_map_y,
                            cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        # Combine using masks
        equirect = np.zeros((self.out_height, self.out_width, 3), dtype=np.uint8)
        equirect[self.front_mask] = front_result[self.front_mask]
        equirect[~self.front_mask] = back_result[~self.front_mask]
        
        return equirect


def main():
    args = parse_args()
    
    # Check if files exist
    if not os.path.isfile(args.front):
        print(f"Error: Front video file not found: {args.front}")
        return
    if not os.path.isfile(args.back):
        print(f"Error: Back video file not found: {args.back}")
        return
        
    print(f"Starting calibration with videos: {args.front} and {args.back}")
    print("Controls:")
    print("  Space: Play/Pause")
    print("  S: Save current parameters")
    print("  ESC: Exit")
    
    # Start calibration GUI
    gui = CalibrationGUI(
        args.front,
        args.back,
        out_width=args.width,
        out_height=args.height
    )
    
    # Run main loop
    gui.run()
    
    print("Calibration complete")


if __name__ == "__main__":
    main()