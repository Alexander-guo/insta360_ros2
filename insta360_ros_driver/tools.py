#!/usr/bin/env python3
import cv2
from sensor_msgs.msg import CompressedImage, Image
import numpy as np

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def split_image(image):
    height, width = image.shape[:2]
    mid_point = width // 2
    front_image = image[:, :mid_point]
    back_image = image[:, mid_point:]

    return back_image, front_image

def undistort_image(image, K, D, use_gpu=False):
    """
    Undistort fisheye image using either CPU or GPU
    
    Args:
        image: Input image
        K: Camera matrix
        D: Distortion coefficients
        use_gpu: Whether to use GPU acceleration (if available)
        
    Returns:
        Undistorted image
    """
    h, w = image.shape[:2]
    new_K = K.copy()
    
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            # Create undistortion maps
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
            
            # Convert maps to GPU
            gpu_map1 = cv2.cuda_GpuMat()
            gpu_map2 = cv2.cuda_GpuMat()
            gpu_map1.upload(map1)
            gpu_map2.upload(map2)
            
            # Upload image to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # Perform remap on GPU
            gpu_result = cv2.cuda.remap(gpu_image, gpu_map1, gpu_map2, cv2.INTER_LINEAR)
            
            # Download result back to CPU
            undistorted_img = gpu_result.download()
            return undistorted_img
        
        except cv2.error as e:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fall back to CPU implementation
    
    # CPU implementation (original)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_img

def compress_image_to_msg(image, timestamp):
    _, buffer = cv2.imencode('.jpg', image)
    image_msg = CompressedImage()
    image_msg.header.stamp = timestamp
    image_msg.format = 'jpeg'
    image_msg.data = buffer.tobytes()
    return image_msg

def image_to_msg(image, timestamp, frame_id="camera_frame"):
    """Convert raw OpenCV image directly to Image message"""
    image_msg = Image()
    image_msg.header.stamp = timestamp
    image_msg.header.frame_id = frame_id
    
    if len(image.shape) == 3:  # Color image
        image_msg.height = image.shape[0]
        image_msg.width = image.shape[1]
        image_msg.encoding = 'bgr8'
        image_msg.is_bigendian = 0
        image_msg.step = 3 * image.shape[1]
    else:  # Grayscale image
        image_msg.height = image.shape[0]
        image_msg.width = image.shape[1]
        image_msg.encoding = 'mono8'
        image_msg.is_bigendian = 0
        image_msg.step = image.shape[1]
        
    image_msg.data = image.tobytes()
    return image_msg