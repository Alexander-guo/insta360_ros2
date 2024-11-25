#!/usr/bin/env python3

import os
import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.parameter import Parameter

from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from sensor_msgs.msg import CompressedImage

from tqdm import tqdm

from insta360_ros_driver.tools import split_image, compress_image_to_msg
from insta360_ros_driver.directory_verification import verify_directories


def compression_node():
    rclpy.init()
    node = Node('compression')
    bridge = CvBridge()

    topic_name = '/insta_image_yuv'

    # Declare parameters
    node.declare_parameter('raw_bag_folder', '/home/bag/raw')
    node.declare_parameter('compressed_bag_folder', '/home/bag/compressed')
    node.declare_parameter('K', [1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0])
    node.declare_parameter('D', [0.0, 0.0, 0.0, 0.0])

    # Retrieve parameters
    raw_bag_folder = node.get_parameter('raw_bag_folder').value
    compressed_bag_folder = node.get_parameter('compressed_bag_folder').value
    K_param = node.get_parameter('K').value
    D_param = node.get_parameter('D').value

    # Convert K and D to NumPy arrays
    K = np.array(K_param).reshape(3, 3) if len(K_param) == 9 else np.eye(3)
    D = np.array(D_param[:4]) if len(D_param) >= 4 else np.zeros(4)

    # Verify directories
    verify_directories()

    # Log directories
    node.get_logger().info(f"Raw Bag Folder: {raw_bag_folder}")
    node.get_logger().info(f"Compressed Bag Folder: {compressed_bag_folder}")

    # Gather bag files
    try:
        bag_filenames = os.listdir(raw_bag_folder)
    except FileNotFoundError:
        node.get_logger().error(f"Raw bag folder not found: {raw_bag_folder}")
        node.destroy_node()
        rclpy.shutdown()
        return

    bag_filenames = [f for f in bag_filenames if f.endswith('.db3') and not f.startswith('.')]
    bag_filenames.sort()
    bag_filenames.sort(key=len)

    node.get_logger().info(f"Bag files to process: {bag_filenames}")

    bag_paths = [os.path.join(raw_bag_folder, bag_filename) for bag_filename in bag_filenames]
    outbag_filenames = [os.path.splitext(filename)[0] + '_compressed.db3' for filename in bag_filenames]
    outbag_paths = [os.path.join(compressed_bag_folder, outbag_filename) for outbag_filename in outbag_filenames]

    for i in tqdm(range(len(bag_paths)), desc="Processing bag files"):
        try:
            input_bag_path = bag_paths[i]
            output_bag_path = outbag_paths[i]

            storage_options = StorageOptions(uri=input_bag_path, storage_id='sqlite3')
            converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

            reader = SequentialReader()
            reader.open(storage_options, converter_options)

            writer_storage_options = StorageOptions(uri=output_bag_path, storage_id='sqlite3')
            writer_converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

            writer = SequentialWriter()
            writer.open(writer_storage_options, writer_converter_options)

            # Get topic information from reader and register with writer
            topics_metadata = reader.get_all_topics_and_types()
            for topic in topics_metadata:
                writer.create_topic(TopicMetadata(name=topic.name, type=topic.type, serialization_format='cdr'))

            while reader.has_next():
                (topic, data, t) = reader.read_next()
                if topic == topic_name:
                    try:
                        msg = deserialize_message(data, CompressedImage)
                        # Decode compressed image
                        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        # Convert YUV to BGR
                        bgr_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)

                        # Split image into front and back
                        front_image, back_image = split_image(bgr_image)

                        # Compress images to messages
                        front_compressed_msg = compress_image_to_msg(front_image, t)
                        back_compressed_msg = compress_image_to_msg(back_image, t)

                        # Serialize messages
                        front_data = serialize_message(front_compressed_msg)
                        back_data = serialize_message(back_compressed_msg)

                        # Write to output bag
                        writer.write('/front_camera_image/compressed', front_data, t)
                        writer.write('/back_camera_image/compressed', back_data, t)
                    except Exception as e:
                        node.get_logger().error(f"Error processing message on {topic}: {e}")
                else:
                    # Write original message for other topics
                    writer.write(topic, data, t)

            reader.close()
            writer.close()
            node.get_logger().info(f"Successfully processed {bag_filenames[i]}")

        except Exception as e:
            node.get_logger().error(f"Failed to process {bag_filenames[i]}: {e}")
            continue

    node.get_logger().info("All bag files processed. Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


def main():
    compression_node()


if __name__ == '__main__':
    main()
