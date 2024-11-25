#!/usr/bin/env python3

import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from tqdm import tqdm

import rclpy
from rclpy.node import Node

from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from insta360_ros_driver.tools import undistort_image, compress_image_to_msg
from insta360_ros_driver.directory_verification import verify_directories

class UndistortionNode(Node):
    def __init__(self):
        super().__init__('undistortion')
        self.bridge = CvBridge()

        self.topic_names = ['/front_camera_image/compressed', '/back_camera_image/compressed']

        # Declare and get parameters
        self.declare_parameter('K', [1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0])
        self.declare_parameter('D', [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('compressed_bag_folder', '/home/bag/compressed')
        self.declare_parameter('undistorted_bag_folder', '/home/bag/undistorted')

        K_param = self.get_parameter('K').value
        D_param = self.get_parameter('D').value
        self.K = np.array(K_param).reshape(3, 3)
        self.D = np.array(D_param[:4])

        verify_directories()
        default_dir = "/home/bag"
        self.compressed_bag_folder = self.get_parameter('compressed_bag_folder').value
        self.undistorted_bag_folder = self.get_parameter('undistorted_bag_folder').value

        bag_filenames = os.listdir(self.compressed_bag_folder)
        bag_filenames = [f for f in bag_filenames if f.endswith('.db3') and not f.startswith('.')]
        bag_filenames.sort()
        bag_filenames.sort(key=len)
        print(bag_filenames)

        bag_paths = [os.path.join(self.compressed_bag_folder, bag_filename) for bag_filename in bag_filenames]

        outbag_filenames = [os.path.splitext(filename)[0] + '_undistorted.db3' for filename in bag_filenames]
        outbag_paths = [os.path.join(self.undistorted_bag_folder, outbag_filename) for outbag_filename in outbag_filenames]

        for i in tqdm(range(len(bag_paths))):
            try:
                self.process_bag(bag_paths[i], outbag_paths[i])
            except Exception as e:
                print(e)
                continue

    def process_bag(self, input_bag_path, output_bag_path):
        storage_options = StorageOptions(uri=input_bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        writer_storage_options = StorageOptions(uri=output_bag_path, storage_id='sqlite3')
        writer_converter_options = ConverterOptions('', '')
        writer = SequentialWriter()
        writer.open(writer_storage_options, writer_converter_options)

        # Get topic information from reader and register with writer
        topic_types = reader.get_all_topics_and_types()
        for topic in topic_types:
            writer.create_topic(TopicMetadata(name=topic.name, type=topic.type, serialization_format='cdr'))

        # Read and process messages
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            if topic in self.topic_names:
                from rclpy.serialization import deserialize_message, serialize_message
                from sensor_msgs.msg import CompressedImage

                msg = deserialize_message(data, CompressedImage)

                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                image = undistort_image(image, self.K, self.D)
                undistorted_msg = compress_image_to_msg(image, msg.header.stamp)

                data = serialize_message(undistorted_msg)
                writer.write(topic, data, t)
            else:
                writer.write(topic, data, t)

        reader.close()
        writer.close()

def main(args=None):
    rclpy.init(args=args)
    node = UndistortionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
