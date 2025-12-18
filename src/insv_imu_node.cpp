#include "insv/insta360_trailer_parser.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/serialization.hpp>
#include <rmw/rmw.h>

#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_options.hpp>

#include <string>
#include <vector>
#include <chrono>
#include <thread>

class InsvImuNode : public rclcpp::Node {
public:
    InsvImuNode() : rclcpp::Node("insv_imu_parser_node") {
        declare_parameter<std::string>("file_path", "");
        declare_parameter<std::string>("imu_topic", "imu/data_raw");
        declare_parameter<std::string>("frame_id", "imu_frame");
        declare_parameter<bool>("realtime", false);
        declare_parameter<bool>("write_to_bag", false);
        declare_parameter<std::string>("bag_path", "");

        file_path_ = get_parameter("file_path").as_string();
        topic_ = get_parameter("imu_topic").as_string();
        frame_id_ = get_parameter("frame_id").as_string();
        realtime_ = get_parameter("realtime").as_bool();
        write_to_bag_ = get_parameter("write_to_bag").as_bool();
        bag_path_ = get_parameter("bag_path").as_string();

        publisher_ = create_publisher<sensor_msgs::msg::Imu>(topic_, rclcpp::SensorDataQoS());

        if (file_path_.empty()) {
            RCLCPP_ERROR(get_logger(), "Parameter 'file_path' is required");
            return;
        }

        RCLCPP_INFO(get_logger(), "Parsing INSV file: %s", file_path_.c_str());
        insta360_insv::TrailerParser parser;
        std::vector<insta360_insv::ImuSample> samples;
        std::string err;
        if (!parser.ParseFile(file_path_, samples, &err)) {
            RCLCPP_ERROR(get_logger(), "Failed to parse INSV: %s", err.c_str());
            return;
        }

        if (samples.empty()) {
            RCLCPP_WARN(get_logger(), "No IMU samples found in trailer");
            return;
        }

        if (write_to_bag_) {
            if (bag_path_.empty()) {
                RCLCPP_ERROR(get_logger(), "'write_to_bag' is true but 'bag_path' is empty");
                return;
            }
            if (!InitBagWriter()) {
                RCLCPP_ERROR(get_logger(), "Failed to initialize rosbag2 writer");
                return;
            }
            RCLCPP_INFO(get_logger(), "Writing IMU samples to bag: %s", bag_path_.c_str());
        }

        RCLCPP_INFO(get_logger(), "Publishing %zu IMU samples to '%s'%s%s", samples.size(), topic_.c_str(),
                    realtime_ ? " in realtime" : " as fast as possible",
                    write_to_bag_ ? " and recording to rosbag2" : "");

        PublishSamples(samples);

        RCLCPP_INFO(get_logger(), "Publishing complete");
    }

private:
    void PublishSamples(const std::vector<insta360_insv::ImuSample>& samples) {
        if (!publisher_) return;

        // Establish a base time for simulated realtime playback
        double first_time = samples.front().time_sec;
        rclcpp::Time start_ros_time = this->get_clock()->now();

        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& s = samples[i];

            // Only publish IMU records; skip pure video timestamp records (kept for alignment elsewhere).
            if (s.is_video_ts) {
                continue;
            }

            // Build Imu message
            auto msg = sensor_msgs::msg::Imu();

            // Timestamp: use the original absolute time if available. If realtime=true, map to wall clock.
            rclcpp::Time stamp;
            if (realtime_) {
                // Map recorded time offsets onto wall clock
                double dt = s.time_sec - first_time; // seconds
                stamp = rclcpp::Time(start_ros_time.nanoseconds() + static_cast<int64_t>(dt * 1e9));
            } else {
                // Use recorded time directly
                stamp = rclcpp::Time(static_cast<int64_t>(s.time_sec * 1e9));
            }

            msg.header.stamp = stamp;
            msg.header.frame_id = frame_id_;

            msg.angular_velocity.x = s.gx;
            msg.angular_velocity.y = s.gy;
            msg.angular_velocity.z = s.gz;

            // Convert linear acceleration to m/s^2 if the parser provides g's; here parser already outputs in g (scaled shorts) or raw doubles.
            // To keep consistent with previous usage, multiply ax/ay/az by 9.80665.
            msg.linear_acceleration.x = s.ax * 9.80665;
            msg.linear_acceleration.y = s.ay * 9.80665;
            msg.linear_acceleration.z = s.az * 9.80665;

            // No orientation data
            msg.orientation_covariance[0] = -1.0;
            for (int c = 0; c < 9; ++c) {
                msg.angular_velocity_covariance[c] = 0.0;
                msg.linear_acceleration_covariance[c] = 0.0;
            }

            publisher_->publish(msg);

            if (write_to_bag_) {
                // Serialize and write to rosbag2
                rclcpp::Serialization<sensor_msgs::msg::Imu> serializer;
                rclcpp::SerializedMessage serialized;
                serializer.serialize_message(&msg, &serialized);
                try {
                    writer_.write(topic_, serialized, msg.header.stamp.nanoseconds());
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(get_logger(), "Failed to write to bag: %s", e.what());
                }
            }

            if (realtime_ && i + 1 < samples.size()) {
                double dt = samples[i + 1].time_sec - s.time_sec;
                if (dt > 0.0 && dt < 5.0) {
                    // Sleep roughly according to recorded delta
                    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
                }
            }
        }
    }

    bool InitBagWriter() {
        try {
            rosbag2_storage::StorageOptions storage_options;
            storage_options.uri = bag_path_;
            storage_options.storage_id = "sqlite3";

            rosbag2_cpp::ConverterOptions converter_options;
            converter_options.input_serialization_format = rmw_get_serialization_format();
            converter_options.output_serialization_format = rmw_get_serialization_format();

            writer_.open(storage_options, converter_options);

            rosbag2_storage::TopicMetadata meta;
            meta.name = topic_;
            meta.type = "sensor_msgs/msg/Imu";
            meta.serialization_format = rmw_get_serialization_format();
            writer_.create_topic(meta);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error initializing rosbag2 writer: %s", e.what());
            return false;
        }
        return true;
    }

    std::string file_path_;
    std::string topic_;
    std::string frame_id_;
    bool realtime_{false};
    bool write_to_bag_{false};
    std::string bag_path_;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
    rosbag2_cpp::Writer writer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InsvImuNode>();
    // If node failed (e.g., missing file), it simply finishes after construction.
    rclcpp::shutdown();
    return 0;
}
