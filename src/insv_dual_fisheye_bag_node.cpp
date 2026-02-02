#include "insv/insta360_trailer_parser.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
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
        declare_parameter<bool>("compressed_images", true);
        declare_parameter<std::string>("image_transport_format", "jpeg");
        declare_parameter<std::string>("storage_id", "db3");
        declare_parameter<double>("time_window_margin_sec", 0.05);

        file_path_ = get_parameter("file_path").as_string();
        bag_path_ = get_parameter("bag_path").as_string();
        front_topic_ = get_parameter("front_topic").as_string();
        rear_topic_ = get_parameter("rear_topic").as_string();
        imu_topic_ = get_parameter("imu_topic").as_string();
        frame_id_front_ = get_parameter("frame_id_front").as_string();
        frame_id_rear_ = get_parameter("frame_id_rear").as_string();
        imu_frame_id_ = get_parameter("imu_frame_id").as_string();
        compressed_images_ = get_parameter("compressed_images").as_bool();
        image_transport_format_ = get_parameter("image_transport_format").as_string();
        storage_id_param_ = get_parameter("storage_id").as_string();
        time_window_margin_sec_ = get_parameter("time_window_margin_sec").as_double();

        // Map user-friendly values to rosbag2 storage IDs
        if (storage_id_param_ == "db3" || storage_id_param_ == "sqlite" || storage_id_param_ == "sqlite3") {
            storage_id_param_ = "sqlite3";
        } else if (storage_id_param_ == "mcap" || storage_id_param_ == "MCAP") {
            storage_id_param_ = "mcap";
        } else {
            RCLCPP_WARN(get_logger(), "Unknown storage_id '%s', defaulting to sqlite3", storage_id_param_.c_str());
            storage_id_param_ = "sqlite3";
        }

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
                video_ts_raw_.push_back(s.raw_time);
            } else {
                imu_samples_.push_back(s);
            }
        }
        // Stats on IMU record encoding
        if (!imu_samples_.empty()) {
            size_t hp = 0, sp = 0;
            for (const auto& s : imu_samples_) {
                if (s.high_precision) hp++; else sp++;
            }
            RCLCPP_INFO(get_logger(), "IMU records parsed: %zu total (double=%zu, short=%zu)", imu_samples_.size(), hp, sp);
        }
        if (!video_ts_raw_.empty()) {
            std::sort(video_ts_raw_.begin(), video_ts_raw_.end());
            video_ts_raw_.erase(std::unique(video_ts_raw_.begin(), video_ts_raw_.end()), video_ts_raw_.end());
        }
        if (!imu_samples_.empty()) {
            std::sort(imu_samples_.begin(), imu_samples_.end(), [](const insta360_insv::ImuSample& lhs, const insta360_insv::ImuSample& rhs) {
                return lhs.raw_time < rhs.raw_time;
            });
        }
        if (video_ts_raw_.empty()) {
            RCLCPP_WARN(get_logger(), "No video timestamps (0x600) found; alignment will use zero offset");
        }

        // Decode frames first to derive reliable video time window from frame PTS
        {
            insta360_insv::InsvVideoDecoder decoder(file_path_);
            std::string derr;
            if (!decoder.open(&derr)) {
                RCLCPP_ERROR(get_logger(), "Decoder open failed: %s", derr.c_str());
                return;
            }

            if (!decoder.decode_all(frames_, &derr)) {
                RCLCPP_ERROR(get_logger(), "Decode failed: %s", derr.c_str());
                return;
            }

            if (frames_.empty()) {
                RCLCPP_ERROR(get_logger(), "No video frames decoded; cannot determine video duration");
                return;
            }

            // Use decoded frame timestamps as ground truth window
            video_min_sec_ = frames_.front().t_video;
            video_max_sec_ = frames_.back().t_video;
            video_dur_sec_ = std::max(0.0, video_max_sec_ - video_min_sec_);
        }

        // Align IMU raw timestamps to video timestamps with robust filtering.
        if (!imu_samples_.empty()) {
            // Use video window from decoded frames
            const double video_start_sec = video_min_sec_;
            const double video_dur = video_dur_sec_;
            // Build a list of plausible raw times to infer scale (reject extremes poisoning span)
            std::vector<double> valid_raw;
            valid_raw.reserve(imu_samples_.size());
            for (const auto& s : imu_samples_) {
                if (std::isfinite(s.raw_time) && s.raw_time > 0.0 && s.raw_time < 1e12) {
                    valid_raw.push_back(s.raw_time);
                }
            }
            double min_raw = imu_samples_.front().raw_time;
            double max_raw = imu_samples_.front().raw_time;
            if (!valid_raw.empty()) {
                auto [mn_it, mx_it] = std::minmax_element(valid_raw.begin(), valid_raw.end());
                min_raw = *mn_it;
                max_raw = *mx_it;
            } else {
                for (const auto& s : imu_samples_) {
                    if (s.raw_time < min_raw) min_raw = s.raw_time;
                    if (s.raw_time > max_raw) max_raw = s.raw_time;
                }
            }
            const double span = max_raw - min_raw;
            // Detect IMU scale from its own span only (µs vs ms)
            imu_time_scale_ = (span > 1e6 ? 1e-6 : 1e-3);

            // Estimate offset to anchor IMU start to video start (first valid IMU)
            imu_time_offset_ = 0.0;
            if (std::isfinite(video_start_sec)) {
                double imu_ref = std::numeric_limits<double>::quiet_NaN();
                for (const auto& s : imu_samples_) {
                    if (s.raw_time > 0.0 && s.raw_time < 1e16) { // sanity
                        imu_ref = s.raw_time * imu_time_scale_;
                        break;
                    }
                }
                if (std::isfinite(imu_ref)) {
                    imu_time_offset_ = video_start_sec - imu_ref;
                }
            }

            // Filter by duration bound and monotonicity on raw_time; then compute aligned time
            const double video_min = std::isfinite(video_start_sec) ? video_start_sec : std::numeric_limits<double>::quiet_NaN();
            const double video_max = (std::isfinite(video_start_sec) && std::isfinite(video_dur)) ? (video_start_sec + video_dur) : std::numeric_limits<double>::quiet_NaN();
            // Keep existing member window (already set from frames)

            std::vector<insta360_insv::ImuSample> cleaned;
            cleaned.reserve(imu_samples_.size());
            size_t dropped_ts_range = 0;
            size_t dropped_non_mono = 0;
            double last_raw = -1.0;
            for (const auto& s : imu_samples_) {
                const double rt = s.raw_time;
                if (!(rt > 0.0 && std::isfinite(rt))) {
                    ++dropped_ts_range;
                    continue;
                }
                if (rt < min_raw || rt > max_raw) {
                    ++dropped_ts_range;
                    continue;
                }
                // Use aligned IMU time against absolute video window
                if (std::isfinite(video_min) && std::isfinite(video_max)) {
                    const double sec_aligned = rt * imu_time_scale_ + imu_time_offset_;
                    if (!(sec_aligned >= video_min - time_window_margin_sec_ && sec_aligned <= video_max + time_window_margin_sec_)) {
                        ++dropped_ts_range;
                        continue;
                    }
                }
                if (last_raw >= 0.0 && rt <= last_raw) {
                    ++dropped_non_mono;
                    continue;
                }
                last_raw = rt;
                cleaned.push_back(s);
            }

            if (!cleaned.empty()) {
                imu_samples_.swap(cleaned);
            }

            for (auto& s : imu_samples_) {
                s.time_sec = s.raw_time * imu_time_scale_ + imu_time_offset_;
            }

            // Drop IMU samples that still fall outside the computed video window after alignment
            size_t dropped_beyond_window = 0;
            if (std::isfinite(video_min) && std::isfinite(video_max)) {
                std::vector<insta360_insv::ImuSample> windowed;
                windowed.reserve(imu_samples_.size());
                for (const auto& s : imu_samples_) {
                    if (s.time_sec < video_min - time_window_margin_sec_ || s.time_sec > video_max + time_window_margin_sec_) {
                        ++dropped_beyond_window;
                        continue;
                    }
                    windowed.push_back(s);
                }
                if (!windowed.empty()) {
                    imu_samples_.swap(windowed);
                }
            }

            RCLCPP_INFO(get_logger(),
                        "IMU time alignment: raw_min=%.3f raw_max=%.3f span=%.3f scale=%.6g offset=%.6f video_start=%.6f video_dur=%.3f dropped_range=%zu dropped_nonmono=%zu dropped_outside=%zu kept=%zu",
                        min_raw, max_raw, span, imu_time_scale_, imu_time_offset_,
                        video_min, std::isfinite(video_dur) ? video_dur : -1.0, dropped_ts_range, dropped_non_mono, dropped_beyond_window, imu_samples_.size());
        }

        double imu_min = imu_samples_.front().time_sec;
        double imu_max = imu_samples_.back().time_sec;

        RCLCPP_WARN(
            get_logger(),
            "FINAL WINDOW: video [%.3f, %.3f], imu [%.3f, %.3f]",
            video_min_sec_, video_max_sec_, imu_min, imu_max);

        // Write IMU samples first (now clamped to the video window)
        WriteImuSamples();

        // Write dual fisheye frames (reuse decoded frames)
        DecodeAndWriteVideo();
    }

private:
    bool InitBagWriter() {
        try {
            rosbag2_storage::StorageOptions storage_options;
            storage_options.uri = bag_path_;
            storage_options.storage_id = storage_id_param_;

            rosbag2_cpp::ConverterOptions converter_options;
            converter_options.input_serialization_format = rmw_get_serialization_format();
            converter_options.output_serialization_format = rmw_get_serialization_format();

            writer_.open(storage_options, converter_options);

            // Create topics
            if (compressed_images_) {
                rosbag2_storage::TopicMetadata meta_front;
                meta_front.name = front_topic_ + "/compressed";
                meta_front.type = "sensor_msgs/msg/CompressedImage";
                meta_front.serialization_format = rmw_get_serialization_format();
                writer_.create_topic(meta_front);

                rosbag2_storage::TopicMetadata meta_rear;
                meta_rear.name = rear_topic_ + "/compressed";
                meta_rear.type = "sensor_msgs/msg/CompressedImage";
                meta_rear.serialization_format = rmw_get_serialization_format();
                writer_.create_topic(meta_rear);
            } else {
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
            }

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
        size_t skipped = 0;
        size_t reason_not_finite = 0;
        size_t reason_negative = 0;
        size_t sanity_warned = 0;
        size_t logged_examples = 0;
        for (const auto& s : imu_samples_) {
            if (!std::isfinite(s.time_sec)) {
                skipped++;
                reason_not_finite++;
                if (logged_examples < 10) {
                    RCLCPP_WARN(get_logger(),
                                "Skip IMU: non-finite time_sec (raw=%.3f, scale=%.6g, offset=%.6f)",
                                s.raw_time, imu_time_scale_, imu_time_offset_);
                    logged_examples++;
                }
                continue;
            }
            if (std::isfinite(video_max_sec_) && s.time_sec > video_max_sec_ + time_window_margin_sec_ && sanity_warned < 10) {
                RCLCPP_WARN(get_logger(),
                            "IMU timestamp %.3f exceeds video end + margin (%.3f + %.3f)",
                            s.time_sec, video_max_sec_, time_window_margin_sec_);
                ++sanity_warned;
            }
            if (!(s.time_sec >= -1.0 && s.time_sec <= (static_cast<double>(std::numeric_limits<int64_t>::max())/1e9))) {
                skipped++;
                reason_negative++;
                if (logged_examples < 10) {
                    RCLCPP_WARN(get_logger(),
                                "Skip IMU: out-of-range stamp (time_sec=%.6f raw=%.3f scale=%.6g offset=%.6f)",
                                s.time_sec, s.raw_time, imu_time_scale_, imu_time_offset_);
                    logged_examples++;
                }
                continue;
            }
            const int64_t stamp_ns = static_cast<int64_t>(s.time_sec * 1e9);
            const rclcpp::Time stamp(stamp_ns);
            sensor_msgs::msg::Imu msg;
            msg.header.stamp = stamp;
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
            std::shared_ptr<const rclcpp::SerializedMessage> serialized_ptr =
                std::make_shared<rclcpp::SerializedMessage>(serialized);
            writer_.write(serialized_ptr, imu_topic_, "sensor_msgs/msg/Imu", stamp);
        }
        if (skipped > 0) {
            RCLCPP_WARN(get_logger(), "Skipped %zu IMU samples with invalid timestamps", skipped);
        }
        RCLCPP_INFO(get_logger(), "Wrote %zu IMU samples", imu_samples_.size() - skipped);
    }

    void DecodeAndWriteVideo() {
        if (frames_.empty()) {
            RCLCPP_ERROR(get_logger(), "No decoded frames available to write");
            return;
        }

        double offset_sec = 0.0;
        if (!frames_.empty() && !video_ts_raw_.empty()) {
            const double vmin = video_ts_raw_.front();
            const double vmax = video_ts_raw_.back();
            const double vspan = std::max(0.0, vmax - vmin);
            const double vscale = (vspan > 1e6 ? 1e-6 : 1e-3);
            const double base_ts = vmin * vscale;
            offset_sec = frames_.front().t_video - base_ts;
            RCLCPP_INFO(get_logger(), "Computed offset_sec=%.6f (t_video=%.6f, trailer_ts=%.6f)", offset_sec, frames_.front().t_video, base_ts);
        }

        const std::string rear_topic_to_write = compressed_images_ ? rear_topic_ + "/compressed" : rear_topic_;
        const std::string front_topic_to_write = compressed_images_ ? front_topic_ + "/compressed" : front_topic_;

        rclcpp::Serialization<sensor_msgs::msg::Image> img_serializer;
        rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed_serializer;
        int64_t frame_idx = 0;
        size_t skipped_frames = 0;
        for (const auto& f : frames_) {
            const double stamp_sec = f.t_video - offset_sec;
            if (!std::isfinite(stamp_sec)) {
                skipped_frames++;
                continue;
            }
            const int64_t stamp_ns = static_cast<int64_t>(stamp_sec * 1e9);
            if (stamp_ns < 0) {
                skipped_frames++;
                continue;
            }
            rclcpp::Time stamp(stamp_ns);

            // rear
            {
                if (compressed_images_) {
                    sensor_msgs::msg::CompressedImage msg;
                    msg.header.stamp = stamp;
                    msg.header.frame_id = frame_id_rear_;
                    msg.format = image_transport_format_;
                    std::vector<int> params;
                    if (image_transport_format_ == "jpeg" || image_transport_format_ == "jpg") {
                        params = { cv::IMWRITE_JPEG_QUALITY, 90 };
                        msg.format = "jpeg";
                    } else if (image_transport_format_ == "png") {
                        params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
                        msg.format = "png";
                    }
                    cv::imencode("." + msg.format, f.rear, msg.data, params);
                    rclcpp::SerializedMessage serialized;
                    compressed_serializer.serialize_message(&msg, &serialized);
                    std::shared_ptr<const rclcpp::SerializedMessage> serialized_ptr =
                        std::make_shared<rclcpp::SerializedMessage>(serialized);
                    writer_.write(serialized_ptr, rear_topic_to_write, "sensor_msgs/msg/CompressedImage", stamp);
                } else {
                    std_msgs::msg::Header header;
                    header.stamp = stamp;
                    header.frame_id = frame_id_rear_;
                    cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, f.rear);
                    sensor_msgs::msg::Image msg;
                    cv_img.toImageMsg(msg);
                    rclcpp::SerializedMessage serialized;
                    img_serializer.serialize_message(&msg, &serialized);
                    std::shared_ptr<const rclcpp::SerializedMessage> serialized_ptr =
                        std::make_shared<rclcpp::SerializedMessage>(serialized);
                    writer_.write(serialized_ptr, rear_topic_to_write, "sensor_msgs/msg/Image", stamp);
                }
            }
            // front
            {
                if (compressed_images_) {
                    sensor_msgs::msg::CompressedImage msg;
                    msg.header.stamp = stamp;
                    msg.header.frame_id = frame_id_front_;
                    msg.format = image_transport_format_;
                    std::vector<int> params;
                    if (image_transport_format_ == "jpeg" || image_transport_format_ == "jpg") {
                        params = { cv::IMWRITE_JPEG_QUALITY, 90 };
                        msg.format = "jpeg";
                    } else if (image_transport_format_ == "png") {
                        params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
                        msg.format = "png";
                    }
                    cv::imencode("." + msg.format, f.front, msg.data, params);
                    rclcpp::SerializedMessage serialized;
                    compressed_serializer.serialize_message(&msg, &serialized);
                    std::shared_ptr<const rclcpp::SerializedMessage> serialized_ptr =
                        std::make_shared<rclcpp::SerializedMessage>(serialized);
                    writer_.write(serialized_ptr, front_topic_to_write, "sensor_msgs/msg/CompressedImage", stamp);
                } else {
                    std_msgs::msg::Header header;
                    header.stamp = stamp;
                    header.frame_id = frame_id_front_;
                    cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, f.front);
                    sensor_msgs::msg::Image msg;
                    cv_img.toImageMsg(msg);
                    rclcpp::SerializedMessage serialized;
                    img_serializer.serialize_message(&msg, &serialized);
                    std::shared_ptr<const rclcpp::SerializedMessage> serialized_ptr =
                        std::make_shared<rclcpp::SerializedMessage>(serialized);
                    writer_.write(serialized_ptr, front_topic_to_write, "sensor_msgs/msg/Image", stamp);
                }
            }
            frame_idx++;
        }
        if (skipped_frames > 0) {
            RCLCPP_WARN(get_logger(), "Skipped %zu frames with invalid timestamps", skipped_frames);
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
    std::vector<double> video_ts_raw_;
    std::vector<insta360_insv::ImuSample> imu_samples_;

    bool compressed_images_{false};
    std::string image_transport_format_ = "jpeg";
    std::string storage_id_param_ = "sqlite3";

    // Alignment parameters
    double imu_time_scale_{1e-3};
    double imu_time_offset_{0.0};
    double video_min_sec_{std::numeric_limits<double>::quiet_NaN()};
    double video_max_sec_{std::numeric_limits<double>::quiet_NaN()};
    double video_dur_sec_{std::numeric_limits<double>::quiet_NaN()};
    double time_window_margin_sec_{0.05};

    std::vector<insta360_insv::DecodedFrame> frames_;

    rosbag2_cpp::Writer writer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InsvDualFisheyeBagNode>();
    rclcpp::shutdown();
    return 0;
}
