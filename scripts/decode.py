#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import av
import threading
import queue
import time

class H264DecoderNode(Node):
    def __init__(self):
        super().__init__('h264_decoder_node')
        
        # --- Parameters ---
        self.declare_parameter('subscribe_topic', '/dual_fisheye/image/compressed')
        self.declare_parameter('publish_topic', '/dual_fisheye/image')
        
        subscribe_topic = self.get_parameter('subscribe_topic').get_parameter_value().string_value
        publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value

        # --- FFmpeg/PyAV Initialization ---
        # Try hardware accelerated decoder first
        try:
            self.codec = av.CodecContext.create('h264_cuvid', 'r')
            self.get_logger().info("Using hardware H.264 decoder (NVDEC)")
        except:
            try:
                self.codec = av.CodecContext.create('h264', 'r')
                self.get_logger().info("Using software H.264 decoder")
            except Exception as e:
                self.get_logger().error(f"Failed to create H.264 decoder: {e}")
                raise

        # --- Threading and Queue Setup ---
        self.frame_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()

        
        # --- ROS2 Communication ---
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            CompressedImage,
            subscribe_topic,
            self.compressed_image_callback,
            10
        )
        
        self.publisher = self.create_publisher(Image, publish_topic, 10)
        
        self.get_logger().info(f"Subscribing to compressed images on: '{subscribe_topic}'")
        self.get_logger().info(f"Publishing raw images on: '{publish_topic}'")

    def _publish_loop(self):
        """Separate thread for publishing to avoid blocking decoder"""
        while rclpy.ok():
            try:
                ros_image_msg = self.frame_queue.get(timeout=1.0)
                if ros_image_msg is not None:
                    self.publisher.publish(ros_image_msg)
                    self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in publish loop: {e}")

    def compressed_image_callback(self, msg):
        if msg.format != "h264":
            return
        
        try:
            # Parse packets directly from message data
            packets = self.codec.parse(bytes(msg.data))
            
            for packet in packets:
                if packet:
                    frames = self.codec.decode(packet)
                    for frame in frames:
                        cv_image = frame.to_ndarray(format='bgr24')
                        ros_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                        ros_image_msg.header = msg.header
                        
                        try:
                            self.frame_queue.put_nowait(ros_image_msg)
                        except queue.Full:
                            pass
                
        except Exception as e:
            self.get_logger().warn(f"Decode error: {e}")

def main(args=None):
    rclpy.init(args=args)
    decoder_node = H264DecoderNode()
    rclpy.spin(decoder_node)
    decoder_node.destroy_node()
    rclcpy.shutdown()

if __name__ == '__main__':
    main()