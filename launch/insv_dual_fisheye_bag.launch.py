from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    file_path = LaunchConfiguration("file_path")
    bag_path = LaunchConfiguration("bag_path")
    front_topic = LaunchConfiguration("front_topic")
    rear_topic = LaunchConfiguration("rear_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    frame_id_front = LaunchConfiguration("frame_id_front")
    frame_id_rear = LaunchConfiguration("frame_id_rear")
    imu_frame_id = LaunchConfiguration("imu_frame_id")
    compressed_images = LaunchConfiguration("compressed_images")
    image_transport_format = LaunchConfiguration("image_transport_format")
    storage_id = LaunchConfiguration("storage_id")
    time_window_margin_sec = LaunchConfiguration("time_window_margin_sec")
    jpeg_quality = LaunchConfiguration("jpeg_quality")
    encoding_threads = LaunchConfiguration("encoding_threads")
    decoder_threads = LaunchConfiguration("decoder_threads")
    crop_ratio = LaunchConfiguration("crop_ratio")
    save_images = LaunchConfiguration("save_images")
    verbose = LaunchConfiguration("verbose")

    return LaunchDescription([
        DeclareLaunchArgument(
            "file_path",
            description="Absolute path to the source .insv file"
        ),
        DeclareLaunchArgument(
            "bag_path",
            description="Destination rosbag2 folder"
        ),
        DeclareLaunchArgument(
            "front_topic",
            default_value="/insta360/front/image_raw",
            description="Topic for front fisheye frames"
        ),
        DeclareLaunchArgument(
            "rear_topic",
            default_value="/insta360/rear/image_raw",
            description="Topic for rear fisheye frames"
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/insta360/imu",
            description="Topic for IMU messages"
        ),
        DeclareLaunchArgument(
            "frame_id_front",
            default_value="front_frame",
            description="Frame ID for front images"
        ),
        DeclareLaunchArgument(
            "frame_id_rear",
            default_value="rear_frame",
            description="Frame ID for rear images"
        ),
        DeclareLaunchArgument(
            "imu_frame_id",
            default_value="imu_frame",
            description="Frame ID for IMU data"
        ),
        DeclareLaunchArgument(
            "compressed_images",
            default_value="true",
            description="If true, store sensor_msgs/CompressedImage under <topic>/compressed"
        ),
        DeclareLaunchArgument(
            "image_transport_format",
            default_value="jpeg",
            description="Encoding format when writing compressed images (jpeg/png)"
        ),
        DeclareLaunchArgument(
            "storage_id",
            default_value="db3",
            description="Rosbag2 storage id: 'db3' (sqlite3) or 'mcap'"
        ),
        DeclareLaunchArgument(
            "time_window_margin_sec",
            default_value="0.05",
            description="IMU Margin for time window (seconds)"
        ),
        DeclareLaunchArgument(
            "jpeg_quality",
            default_value="50",
            description="JPEG encoding quality (1-100). Ignored if image_transport_format is 'png'"
        ),
        DeclareLaunchArgument(
            "encoding_threads",
            default_value="0",
            description="Worker threads for image encoding (0 = auto)"
        ),
        DeclareLaunchArgument(
            "decoder_threads",
            default_value="0",
            description="FFmpeg decode threads for probing/streaming (0 = auto)"
        ),
        DeclareLaunchArgument(
            "crop_ratio",
            default_value="0.75",
            description="Center crop ratio (0 or 1 = disabled, (0,1) crops square)"
        ),
        DeclareLaunchArgument(
            "save_images",
            default_value="true",
            description="If true, save images to the rosbag2; if false, only save IMU data (useful for debugging or when images are not needed)"
        ),
        DeclareLaunchArgument(
            "verbose",
            default_value="false",
            description="If true, print verbose logs during trailer parsing, helpful for debugging or understanding the trailer structure"
        ),
        Node(
            package="insta360_ros_driver",
            executable="insv_dual_fisheye_bag_node",
            name="insv_dual_fisheye_bag_node",
            output="screen",
            parameters=[{
                "file_path": file_path,
                "bag_path": bag_path,
                "front_topic": front_topic,
                "rear_topic": rear_topic,
                "imu_topic": imu_topic,
                "frame_id_front": frame_id_front,
                "frame_id_rear": frame_id_rear,
                "imu_frame_id": imu_frame_id,
                "verbose": verbose,
                "compressed_images": compressed_images,
                "image_transport_format": image_transport_format,
                "storage_id": storage_id,
                "time_window_margin_sec": time_window_margin_sec,
                "jpeg_quality": jpeg_quality,
                "encoding_threads": encoding_threads,
                "decoder_threads": decoder_threads,
                "crop_ratio": crop_ratio,
                "save_images": save_images,
            }]
        )
    ])
