#include "insv/insta360_trailer_parser.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <rclcpp/serialization.hpp>
#include <rmw/rmw.h>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_options.hpp>

#include "insv/insv_video_decoder.hpp"

class InsvDualFisheyeBagNode : public rclcpp::Node {
public:
    InsvDualFisheyeBagNode() : rclcpp::Node("insv_dual_fisheye_bag_node") {
        declare_parameter<std::string>("file_path", "");
        declare_parameter<std::string>("bag_path", "");
        declare_parameter<std::string>("front_topic", "/insta360/front/image_raw");
        declare_parameter<std::string>("rear_topic", "/insta360/rear/image_raw");
        declare_parameter<std::string>("imu_topic", "/insta360/imu");
        declare_parameter<std::string>("frame_id_front", "front_frame");
        declare_parameter<std::string>("frame_id_rear", "rear_frame");
        declare_parameter<std::string>("imu_frame_id", "imu_frame");

        file_path_ = get_parameter("file_path").as_string();
        bag_path_ = get_parameter("bag_path").as_string();
        front_topic_ = get_parameter("front_topic").as_string();
        rear_topic_ = get_parameter("rear_topic").as_string();
        imu_topic_ = get_parameter("imu_topic").as_string();
        frame_id_front_ = get_parameter("frame_id_front").as_string();
        frame_id_rear_ = get_parameter("frame_id_rear").as_string();
        imu_frame_id_ = get_parameter("imu_frame_id").as_string();

        if (file_path_.empty() || bag_path_.empty()) {
            RCLCPP_ERROR(get_logger(), "Parameters 'file_path' and 'bag_path' are required");
            return;
        }

        if (!InitBagWriter()) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize rosbag2 writer");
            return;
        }

        // Parse trailer IMU and VideoTimeStamp
        insta360_insv::TrailerParser parser;
        std::vector<insta360_insv::ImuSample> trailer_samples;
        std::string err;
        if (!parser.ParseFile(file_path_, trailer_samples, &err)) {
            RCLCPP_ERROR(get_logger(), "Failed to parse trailer: %s", err.c_str());
            return;
        }
        for (const auto& s : trailer_samples) {
            if (s.is_video_ts) {
                video_ts_sec_.push_back(s.video_ts_sec);
            } else {
                imu_samples_.push_back(s);
            }
        }
        if (video_ts_sec_.empty()) {
            RCLCPP_WARN(get_logger(), "No video timestamps (0x600) found; alignment will use zero offset");
        }

        // Write IMU samples first
        WriteImuSamples();

        // Decode and write dual fisheye frames
        DecodeAndWriteVideo();
    }

private:
    bool InitBagWriter() {
        try {
            rosbag2_storage::StorageOptions storage_options;
            storage_options.uri = bag_path_;
            storage_options.storage_id = "sqlite3";

            rosbag2_cpp::ConverterOptions converter_options;
            converter_options.input_serialization_format = rmw_get_serialization_format();
            converter_options.output_serialization_format = rmw_get_serialization_format();

            writer_.open(storage_options, converter_options);

            // Create topics
            rosbag2_storage::TopicMetadata meta_front;
            meta_front.name = front_topic_;
            meta_front.type = "sensor_msgs/msg/Image";
            meta_front.serialization_format = rmw_get_serialization_format();
            writer_.create_topic(meta_front);

            rosbag2_storage::TopicMetadata meta_rear;
            meta_rear.name = rear_topic_;
            meta_rear.type = "sensor_msgs/msg/Image";
            meta_rear.serialization_format = rmw_get_serialization_format();
            writer_.create_topic(meta_rear);

            rosbag2_storage::TopicMetadata meta_imu;
            meta_imu.name = imu_topic_;
            meta_imu.type = "sensor_msgs/msg/Imu";
            meta_imu.serialization_format = rmw_get_serialization_format();
            writer_.create_topic(meta_imu);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error initializing rosbag2 writer: %s", e.what());
            return false;
        }
        return true;
    }

    void WriteImuSamples() {
        rclcpp::Serialization<sensor_msgs::msg::Imu> serializer;
        for (const auto& s : imu_samples_) {
            sensor_msgs::msg::Imu msg;
            msg.header.stamp = rclcpp::Time(static_cast<int64_t>(s.time_sec * 1e9));
            msg.header.frame_id = imu_frame_id_;

            msg.angular_velocity.x = s.gx;
            msg.angular_velocity.y = s.gy;
            msg.angular_velocity.z = s.gz;

            msg.linear_acceleration.x = s.ax * 9.80665;
            msg.linear_acceleration.y = s.ay * 9.80665;
            msg.linear_acceleration.z = s.az * 9.80665;

            msg.orientation_covariance[0] = -1.0;
            for (int i = 0; i < 9; ++i) {
                msg.angular_velocity_covariance[i] = 0.0;
                msg.linear_acceleration_covariance[i] = 0.0;
            }

            rclcpp::SerializedMessage serialized;
            serializer.serialize_message(&msg, &serialized);
            writer_.write(imu_topic_, serialized, msg.header.stamp.nanoseconds());
        }
        RCLCPP_INFO(get_logger(), "Wrote %zu IMU samples", imu_samples_.size());
    }

    void DecodeAndWriteVideo() {
        insta360_insv::InsvVideoDecoder decoder(file_path_);
        std::string derr;
        if (!decoder.open(&derr)) {
            RCLCPP_ERROR(get_logger(), "Decoder open failed: %s", derr.c_str());
            return;
        }

        std::vector<insta360_insv::DecodedFrame> frames;
        if (!decoder.decode_all(frames, &derr)) {
            RCLCPP_ERROR(get_logger(), "Decode failed: %s", derr.c_str());
            return;
        }

        double offset_sec = 0.0;
        if (!frames.empty() && !video_ts_sec_.empty()) {
            offset_sec = frames.front().t_video - video_ts_sec_.front();
            RCLCPP_INFO(get_logger(), "Computed offset_sec=%.6f (t_video=%.6f, trailer_ts=%.6f)", offset_sec, frames.front().t_video, video_ts_sec_.front());
        }

        rclcpp::Serialization<sensor_msgs::msg::Image> img_serializer;
        int64_t frame_idx = 0;
        for (const auto& f : frames) {
            rclcpp::Time stamp(static_cast<int64_t>((f.t_video - offset_sec) * 1e9));

            // rear
            {
                std_msgs::msg::Header header;
                header.stamp = stamp;
                header.frame_id = frame_id_rear_;
                cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, f.rear);
                sensor_msgs::msg::Image msg;
                cv_img.toImageMsg(msg);
                rclcpp::SerializedMessage serialized;
                img_serializer.serialize_message(&msg, &serialized);
                writer_.write(rear_topic_, serialized, msg.header.stamp.nanoseconds());
            }
            // front
            {
                std_msgs::msg::Header header;
                header.stamp = stamp;
                header.frame_id = frame_id_front_;
                cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, f.front);
                sensor_msgs::msg::Image msg;
                cv_img.toImageMsg(msg);
                rclcpp::SerializedMessage serialized;
                img_serializer.serialize_message(&msg, &serialized);
                writer_.write(front_topic_, serialized, msg.header.stamp.nanoseconds());
            }
            frame_idx++;
        }
        RCLCPP_INFO(get_logger(), "Wrote %ld video frames (front+rear)", (long)frame_idx);
    }

    // Params
    std::string file_path_;
    std::string bag_path_;
    std::string front_topic_;
    std::string rear_topic_;
    std::string imu_topic_;
    std::string frame_id_front_;
    std::string frame_id_rear_;
    std::string imu_frame_id_;

    // Trailer data
    std::vector<double> video_ts_sec_;
    std::vector<insta360_insv::ImuSample> imu_samples_;

    rosbag2_cpp::Writer writer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InsvDualFisheyeBagNode>();
    rclcpp::shutdown();
    return 0;
}
