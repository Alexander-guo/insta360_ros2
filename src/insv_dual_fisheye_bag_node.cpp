#include "insv/insta360_trailer_parser.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
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

namespace {
constexpr double STANDARD_GRAVITY_MS2 = 9.80665;

struct EncodedFrameResult {
    size_t seq{0};
    bool valid{false};
    rclcpp::Time stamp;
    std::shared_ptr<const rclcpp::SerializedMessage> rear_msg;
    std::shared_ptr<const rclcpp::SerializedMessage> front_msg;
    std::shared_ptr<const rclcpp::SerializedMessage> rear_cropped_msg;
    std::shared_ptr<const rclcpp::SerializedMessage> front_cropped_msg;
};

class FrameTaskPool {
public:
    explicit FrameTaskPool(size_t thread_count) {
        Start(thread_count);
    }

    FrameTaskPool(const FrameTaskPool&) = delete;
    FrameTaskPool& operator=(const FrameTaskPool&) = delete;

    ~FrameTaskPool() {
        Stop();
    }

    void Enqueue(std::packaged_task<EncodedFrameResult()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

private:
    void Start(size_t count) {
        if (count == 0) {
            count = 1;
        }
        threads_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            threads_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        threads_.clear();
    }

    void WorkerLoop() {
        while (true) {
            std::packaged_task<EncodedFrameResult()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::packaged_task<EncodedFrameResult()>> tasks_;
    bool stopping_{false};
};
}

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
        declare_parameter<double>("crop_ratio", 0.0);
        declare_parameter<int>("jpeg_quality", 90);
        declare_parameter<bool>("save_images", true);
        declare_parameter<bool>("verbose", false);
        const unsigned int hw_threads = std::thread::hardware_concurrency();
        const int default_threads = hw_threads > 0 ? static_cast<int>(hw_threads) : 4;
        const int decoder_default_threads = hw_threads > 0 ? std::min(default_threads, 16) : 4;
        declare_parameter<int>("encoding_threads", default_threads);
        declare_parameter<int>("decoder_threads", decoder_default_threads);

        const std::string file_path_param = get_parameter("file_path").as_string();
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
        crop_ratio_ = get_parameter("crop_ratio").as_double();
        save_images_ = get_parameter("save_images").as_bool();
        verbose_ = get_parameter("verbose").as_bool();
        {
            int q = get_parameter("jpeg_quality").as_int();
            if (q < 1) q = 1;
            if (q > 100) q = 100;
            jpeg_quality_ = q;
        }
        encoding_threads_ = get_parameter("encoding_threads").as_int();
        if (encoding_threads_ < 1) {
            encoding_threads_ = default_threads;
            if (encoding_threads_ < 1) {
                encoding_threads_ = 1;
            }
        }
        decoder_threads_ = get_parameter("decoder_threads").as_int();
        if (decoder_threads_ < 1) {
            decoder_threads_ = decoder_default_threads;
            if (decoder_threads_ < 1) {
                decoder_threads_ = 1;
            }
        }

        if (!std::isfinite(crop_ratio_)) {
            RCLCPP_WARN(get_logger(), "crop_ratio is non-finite; disabling cropping");
            crop_ratio_ = 0.0;
        }
        crop_enabled_ = (crop_ratio_ > 0.0 && crop_ratio_ < 1.0);
        if (crop_enabled_) {
            const std::string crop_suffix = compressed_images_ ? "/cropped/compressed" : "/cropped";
            front_cropped_topic_ = front_topic_ + crop_suffix;
            rear_cropped_topic_ = rear_topic_ + crop_suffix;
            RCLCPP_INFO(get_logger(), "Cropping enabled with ratio %.3f", crop_ratio_);
        } else if (crop_ratio_ < 0.0 || crop_ratio_ > 1.0) {
            RCLCPP_WARN(get_logger(), "crop_ratio %.3f outside (0,1); cropping disabled", crop_ratio_);
            crop_ratio_ = 0.0;
        }

        if (bag_path_.empty()) {
            RCLCPP_ERROR(get_logger(), "Parameter 'bag_path' is required");
            return;
        }

        // Map user-friendly values to rosbag2 storage IDs
        if (storage_id_param_ == "db3" || storage_id_param_ == "sqlite" || storage_id_param_ == "sqlite3") {
            storage_id_param_ = "sqlite3";
        } else if (storage_id_param_ == "mcap" || storage_id_param_ == "MCAP") {
            storage_id_param_ = "mcap";
        } else {
            RCLCPP_WARN(get_logger(), "Unknown storage_id '%s', defaulting to sqlite3", storage_id_param_.c_str());
            storage_id_param_ = "sqlite3";
        }

        if (!InitBagWriter()) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize rosbag2 writer");
            return;
        }

        if (!LoadFilePaths(file_path_param)) {
            RCLCPP_ERROR(get_logger(), "Parameter 'file_path' must be a valid file or directory containing .insv/.lrv files");
            return;
        }
        
        if (file_paths_.size() > 1) {
            multi_files_ = true;
            RCLCPP_INFO(get_logger(), "Found %zu files to process", file_paths_.size());        
        }

        for (const auto& path : file_paths_) {
            RCLCPP_INFO(get_logger(), "\nProcessing file: %s", path.c_str());
            if (!ProcessFile(path)) {
                RCLCPP_ERROR(get_logger(), "Failed to process file: %s", path.c_str());
            }
        }
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

                if (crop_enabled_) {
                    rosbag2_storage::TopicMetadata meta_front_cropped;
                    meta_front_cropped.name = front_cropped_topic_;
                    meta_front_cropped.type = "sensor_msgs/msg/CompressedImage";
                    meta_front_cropped.serialization_format = rmw_get_serialization_format();
                    writer_.create_topic(meta_front_cropped);

                    rosbag2_storage::TopicMetadata meta_rear_cropped;
                    meta_rear_cropped.name = rear_cropped_topic_;
                    meta_rear_cropped.type = "sensor_msgs/msg/CompressedImage";
                    meta_rear_cropped.serialization_format = rmw_get_serialization_format();
                    writer_.create_topic(meta_rear_cropped);
                }
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

                if (crop_enabled_) {
                    rosbag2_storage::TopicMetadata meta_front_cropped;
                    meta_front_cropped.name = front_cropped_topic_;
                    meta_front_cropped.type = "sensor_msgs/msg/Image";
                    meta_front_cropped.serialization_format = rmw_get_serialization_format();
                    writer_.create_topic(meta_front_cropped);

                    rosbag2_storage::TopicMetadata meta_rear_cropped;
                    meta_rear_cropped.name = rear_cropped_topic_;
                    meta_rear_cropped.type = "sensor_msgs/msg/Image";
                    meta_rear_cropped.serialization_format = rmw_get_serialization_format();
                    writer_.create_topic(meta_rear_cropped);
                }
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

    bool LoadFilePaths(const std::string& path) {
        file_paths_.clear();
        if (path.empty()) {
            return false;
        }

        std::error_code ec;
        const std::filesystem::path p(path);
        if (std::filesystem::is_directory(p, ec)) {
            for (const auto& entry : std::filesystem::directory_iterator(p, ec)) {
                if (ec) {
                    break;
                }
                if (!entry.is_regular_file()) {
                    continue;
                }
                const auto ext = entry.path().extension().string();
                std::string ext_lower;
                ext_lower.resize(ext.size());
                std::transform(ext.begin(), ext.end(), ext_lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (ext_lower == ".insv" || ext_lower == ".lrv") {
                    file_paths_.push_back(entry.path().string());
                }
            }
            std::sort(file_paths_.begin(), file_paths_.end());
        } else if (std::filesystem::is_regular_file(p, ec)) {
            file_paths_.push_back(path);
        }

        if (file_paths_.empty()) {
            return false;
        }
        return true;
    }

    bool ProcessFile(const std::string& path) {
        ResetPerFileState();
        file_path_ = path;
        RCLCPP_INFO(get_logger(), "Current global_time_offset_sec: %.3f", global_time_offset_sec_);

        if (!ParseTrailer(path)) {
            return false;
        }
        if (!ProbeVideoWindow()) {
            return false;
        }
        AlignImu(); // trim IMU samples using video time window, since IMU ts are already aligned to video timestamps using first_frame_timestamp
        WriteImuSamples();
        if (save_images_) {
            DecodeAndWriteVideo();
        }

        if (std::isfinite(video_dur_sec_)) {
            global_time_offset_sec_ += video_dur_sec_ + frame_interval_sec_;
        }
        return true;
    }

    void ResetPerFileState() {
        imu_samples_.clear();
        video_min_sec_ = std::numeric_limits<double>::quiet_NaN();
        video_max_sec_ = std::numeric_limits<double>::quiet_NaN();
        video_dur_sec_ = std::numeric_limits<double>::quiet_NaN();
        frame_interval_sec_ = 0.001;
    }

    bool ParseTrailer(const std::string& file) {
        insta360_insv::TrailerParser parser;
        parser.SetVerbose(verbose_);
        std::string err;
        if (!parser.ParseFile(file, imu_samples_, &err)) {
            RCLCPP_ERROR(get_logger(), "Failed to parse trailer: %s", err.c_str());
            return false;
        }
        if (!imu_samples_.empty()) {
            RCLCPP_INFO(get_logger(), "IMU records parsed: %zu total, timestamp interval=[%.3f, %.3f]", 
                imu_samples_.size(), imu_samples_.front().time_sec, imu_samples_.back().time_sec);
        } else {
            RCLCPP_WARN(get_logger(), "No IMU records parsed from trailer!");
        }
        return true;
    }

    bool ProbeVideoWindow() {
        insta360_insv::InsvVideoDecoder decoder(file_path_, decoder_threads_);
        std::string derr;
        if (!decoder.open(&derr)) {
            RCLCPP_ERROR(get_logger(), "Decoder open failed: %s", derr.c_str());
            return false;
        }

        double vmin = 0.0;
        double vmax = 0.0;
        std::size_t frame_count = 0;
        if (!decoder.probe_time_window(vmin, vmax, &frame_count, &derr)) {
            RCLCPP_ERROR(get_logger(), "Failed to probe video timestamps: %s", derr.c_str());
            return false;
        }

        if (!std::isfinite(vmin) || !std::isfinite(vmax)) {
            RCLCPP_ERROR(get_logger(), "Video timestamps are not finite (min=%.3f max=%.3f)", vmin, vmax);
            return false;
        }

        video_min_sec_ = vmin;
        video_max_sec_ = vmax;
        video_dur_sec_ = std::max(0.0, video_max_sec_ - video_min_sec_);

        // Estimate frame interval 
        double avg_vid_gap_sec = video_dur_sec_ / std::max<size_t>(frame_count - 1, 1);
        frame_interval_sec_ = avg_vid_gap_sec;
        if (multi_files_) { 
            time_window_margin_sec_ = avg_vid_gap_sec; 
        }
        RCLCPP_INFO(get_logger(), "Probed video window: [%.3f, %.3f] sec (duration=%.3f sec, frame_count=%zu), estimated frame interval=%.3f sec, time_window_margin_sec_: %.3f", \
            video_min_sec_, video_max_sec_, video_dur_sec_, frame_count, avg_vid_gap_sec, time_window_margin_sec_);

        return true;
    }

    void AlignImu() {
        if (imu_samples_.empty()) {
            RCLCPP_WARN(get_logger(), "IMU empty before IMU Alignment - FINAL WINDOW: video [%.3f, %.3f], imu [no samples remain after filtering]", video_min_sec_, video_max_sec_);
            return;
        }

        const double imu_rh_sec = video_max_sec_ + time_window_margin_sec_;

        const size_t original_imu_count = imu_samples_.size();
        std::vector<insta360_insv::ImuSample> valid_imu;
        valid_imu.reserve(imu_samples_.size());
        for (const auto& s : imu_samples_) {
            if (std::isfinite(s.raw_time) && s.time_sec >= video_min_sec_&& s.time_sec <= imu_rh_sec) {
                valid_imu.push_back(s);
            }
        }

        if (!valid_imu.empty()) {
            imu_samples_.swap(valid_imu);
        } else {
            RCLCPP_WARN(get_logger(), "No IMU samples within video time window [%.3f, %.3f]; all %zu samples are outside the window", 
                video_min_sec_, video_max_sec_, original_imu_count);
        }

        RCLCPP_INFO(get_logger(), "IMU records truncated: Video window with rh margin [%.3f, %.3f(%.3f + %.3f)] sec, imu valid samples within window=%zu (dropped %zu outside window)",
            video_min_sec_, imu_rh_sec, video_max_sec_, time_window_margin_sec_, imu_samples_.size(), original_imu_count - imu_samples_.size());
    }

    void WriteImuSamples() {
        rclcpp::Serialization<sensor_msgs::msg::Imu> serializer;
        size_t skipped = 0;
        for (const auto& s : imu_samples_) {
            const double aligned_time = s.time_sec + global_time_offset_sec_;
            if (aligned_time <= last_written_imu_time_) {
                ++skipped;
                continue;
            }
            const int64_t stamp_ns = static_cast<int64_t>(aligned_time * 1e9);
            const rclcpp::Time stamp(stamp_ns);
            sensor_msgs::msg::Imu msg;
            msg.header.stamp = stamp;
            msg.header.frame_id = imu_frame_id_;

            // Always publish angular velocity in rad/s as required by ROS.
            msg.angular_velocity.x = s.gx;
            msg.angular_velocity.y = s.gy;
            msg.angular_velocity.z = s.gz;

            msg.linear_acceleration.x = s.ax * STANDARD_GRAVITY_MS2;
            msg.linear_acceleration.y = s.ay * STANDARD_GRAVITY_MS2;
            msg.linear_acceleration.z = s.az * STANDARD_GRAVITY_MS2;

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
            last_written_imu_time_ = aligned_time;
        }
        if (skipped > 0) {
            RCLCPP_WARN(get_logger(), "Skipped %zu IMU samples with invalid timestamps", skipped);
        }
        RCLCPP_INFO(get_logger(), "Wrote %zu IMU samples", imu_samples_.size() - skipped);
    }

    void DecodeAndWriteVideo() {
        insta360_insv::InsvVideoDecoder decoder(file_path_, decoder_threads_);
        std::string derr;
        if (!decoder.open(&derr)) {
            RCLCPP_ERROR(get_logger(), "Decoder open failed: %s", derr.c_str());
            return;
        }

        const std::string rear_topic_to_write = compressed_images_ ? rear_topic_ + "/compressed" : rear_topic_;
        const std::string front_topic_to_write = compressed_images_ ? front_topic_ + "/compressed" : front_topic_;
        const bool crop_enabled = crop_enabled_;
        const std::string rear_cropped_topic_to_write = crop_enabled ? rear_cropped_topic_ : std::string();
        const std::string front_cropped_topic_to_write = crop_enabled ? front_cropped_topic_ : std::string();
        const std::string msg_type = compressed_images_ ? "sensor_msgs/msg/CompressedImage" : "sensor_msgs/msg/Image";

        FrameTaskPool pool(static_cast<size_t>(encoding_threads_));
        struct PendingFuture {
            size_t seq{0};
            std::future<EncodedFrameResult> future;
        };
        std::deque<PendingFuture> inflight;
        std::map<size_t, EncodedFrameResult> ready_results;
        size_t next_frame_seq = 0;
        size_t next_write_seq = 0;
        size_t frames_written = 0;
        size_t skipped_frames = 0;
        const size_t max_inflight = std::max<size_t>(static_cast<size_t>(encoding_threads_) * 3, 6);

        auto write_serialized = [&](const std::shared_ptr<const rclcpp::SerializedMessage>& msg,
                                    const std::string& topic,
                                    const rclcpp::Time& stamp) {
            if (!msg) {
                return;
            }
            writer_.write(msg, topic, msg_type, stamp);
        };

        auto drain_ready = [&](bool block_first) {
            if (block_first && !inflight.empty()) {
                inflight.front().future.wait();
            }
            for (auto it = inflight.begin(); it != inflight.end();) {
                if (it->future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    EncodedFrameResult result = it->future.get();
                    ready_results.emplace(result.seq, std::move(result));
                    it = inflight.erase(it);
                } else {
                    ++it;
                }
            }
            while (true) {
                auto ready_it = ready_results.find(next_write_seq);
                if (ready_it == ready_results.end()) {
                    break;
                }
                EncodedFrameResult result = std::move(ready_it->second);
                ready_results.erase(ready_it);
                if (result.valid) {
                    write_serialized(result.rear_msg, rear_topic_to_write, result.stamp);
                    write_serialized(result.front_msg, front_topic_to_write, result.stamp);
                    if (crop_enabled) {
                        write_serialized(result.rear_cropped_msg, rear_cropped_topic_to_write, result.stamp);
                        write_serialized(result.front_cropped_msg, front_cropped_topic_to_write, result.stamp);
                    }
                    ++frames_written;
                } else {
                    ++skipped_frames;
                }
                ++next_write_seq;
            }
        };

        auto make_task = [&](size_t seq, insta360_insv::DecodedFrame frame) {
            return std::packaged_task<EncodedFrameResult()>(
                [seq,
                 frame = std::move(frame),
                 compressed = compressed_images_,
                 fmt = image_transport_format_,
                 jpeg_q = jpeg_quality_,
                 frame_id_front = frame_id_front_,
                 frame_id_rear = frame_id_rear_,
                 crop_enabled = crop_enabled_,
                 crop_ratio = crop_ratio_,
                 global_offset = global_time_offset_sec_]() mutable {
                    EncodedFrameResult result;
                    result.seq = seq;
                    const double stamp_sec = frame.t_video + global_offset;
                    if (!std::isfinite(stamp_sec)) {
                        return result;
                    }
                    const int64_t stamp_ns = static_cast<int64_t>(stamp_sec * 1e9);
                    if (stamp_ns < 0) {
                        return result;
                    }
                    result.stamp = rclcpp::Time(stamp_ns);

                    auto encode_compressed = [&](const cv::Mat& image, const std::string& frame_id) -> std::shared_ptr<const rclcpp::SerializedMessage> {
                        sensor_msgs::msg::CompressedImage msg;
                        msg.header.stamp = result.stamp;
                        msg.header.frame_id = frame_id;
                        std::string local_fmt = fmt;
                        std::vector<int> params;
                        if (local_fmt == "jpeg" || local_fmt == "jpg") {
                            params = { cv::IMWRITE_JPEG_QUALITY, jpeg_q };
                            local_fmt = "jpeg";
                        } else if (local_fmt == "png") {
                            params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
                            local_fmt = "png";
                        }
                        if (!cv::imencode("." + local_fmt, image, msg.data, params)) {
                            return nullptr;
                        }
                        msg.format = local_fmt;
                        auto serialized = std::make_shared<rclcpp::SerializedMessage>();
                        rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serializer;
                        serializer.serialize_message(&msg, serialized.get());
                        return std::shared_ptr<const rclcpp::SerializedMessage>(serialized);
                    };

                    auto encode_raw = [&](const cv::Mat& image, const std::string& frame_id) -> std::shared_ptr<const rclcpp::SerializedMessage> {
                        std_msgs::msg::Header header;
                        header.stamp = result.stamp;
                        header.frame_id = frame_id;
                        cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, image);
                        sensor_msgs::msg::Image msg;
                        cv_img.toImageMsg(msg);
                        auto serialized = std::make_shared<rclcpp::SerializedMessage>();
                        rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
                        serializer.serialize_message(&msg, serialized.get());
                        return std::shared_ptr<const rclcpp::SerializedMessage>(serialized);
                    };

                    auto center_crop = [&](const cv::Mat& image) -> cv::Mat {
                        if (!crop_enabled || image.empty()) {
                            return cv::Mat();
                        }
                        const int min_dim = std::min(image.cols, image.rows);
                        if (min_dim <= 0) {
                            return cv::Mat();
                        }
                        const int edge = static_cast<int>(std::round(static_cast<double>(min_dim) * crop_ratio));
                        if (edge <= 0 || edge >= min_dim) {
                            return cv::Mat();
                        }
                        const int x = (image.cols - edge) / 2;
                        const int y = (image.rows - edge) / 2;
                        return image(cv::Rect(x, y, edge, edge));
                    };

                    if (compressed) {
                        result.rear_msg = encode_compressed(frame.rear, frame_id_rear);
                        result.front_msg = encode_compressed(frame.front, frame_id_front);
                        if (crop_enabled) {
                            const cv::Mat rear_cropped = center_crop(frame.rear);
                            if (!rear_cropped.empty()) {
                                result.rear_cropped_msg = encode_compressed(rear_cropped, frame_id_rear);
                            }
                            const cv::Mat front_cropped = center_crop(frame.front);
                            if (!front_cropped.empty()) {
                                result.front_cropped_msg = encode_compressed(front_cropped, frame_id_front);
                            }
                        }
                    } else {
                        result.rear_msg = encode_raw(frame.rear, frame_id_rear);
                        result.front_msg = encode_raw(frame.front, frame_id_front);
                        if (crop_enabled) {
                            const cv::Mat rear_cropped = center_crop(frame.rear);
                            if (!rear_cropped.empty()) {
                                result.rear_cropped_msg = encode_raw(rear_cropped, frame_id_rear);
                            }
                            const cv::Mat front_cropped = center_crop(frame.front);
                            if (!front_cropped.empty()) {
                                result.front_cropped_msg = encode_raw(front_cropped, frame_id_front);
                            }
                        }
                    }
                    result.valid = (result.rear_msg && result.front_msg);
                    return result;
                });
        };

        auto schedule_frame = [&](insta360_insv::DecodedFrame&& frame) {
            const size_t seq = next_frame_seq++;
            auto task = make_task(seq, std::move(frame));
            std::future<EncodedFrameResult> future = task.get_future();
            pool.Enqueue(std::move(task));
            inflight.push_back(PendingFuture{seq, std::move(future)});
            drain_ready(false);
            if (inflight.size() >= max_inflight) {
                drain_ready(true);
            }
            return true;
        };

        if (!decoder.stream_decode(schedule_frame, &derr)) {
            RCLCPP_ERROR(get_logger(), "Decode failed: %s", derr.c_str());
            return;
        }

        while (!inflight.empty()) {
            drain_ready(true);
        }
        drain_ready(false);

        if (frames_written == 0) {
            RCLCPP_WARN(get_logger(), "Stream decode produced zero frames");
        }
        if (skipped_frames > 0) {
            RCLCPP_WARN(get_logger(), "Skipped %zu frames with invalid timestamps or encoding failures", skipped_frames);
        }
        RCLCPP_INFO(get_logger(), "Wrote %zu video frames (front+rear)", frames_written);
    }

    // Params
    std::vector<std::string> file_paths_;
    std::string file_path_;
    std::string bag_path_;
    std::string front_topic_;
    std::string rear_topic_;
    std::string imu_topic_;
    std::string frame_id_front_;
    std::string frame_id_rear_;
    std::string imu_frame_id_;

    bool save_images_{true};
    bool multi_files_{false};

    double global_time_offset_sec_{0.0};
    double last_written_imu_time_{-std::numeric_limits<double>::infinity()};

    // Trailer data
    std::vector<insta360_insv::ImuSample> imu_samples_;

    bool verbose_{false};
    bool compressed_images_{false};
    std::string image_transport_format_ = "jpeg";
    std::string storage_id_param_ = "sqlite3";

    // Alignment parameters
    double video_min_sec_{std::numeric_limits<double>::quiet_NaN()};
    double video_max_sec_{std::numeric_limits<double>::quiet_NaN()};
    double video_dur_sec_{std::numeric_limits<double>::quiet_NaN()};
    double frame_interval_sec_{0.001};
    double time_window_margin_sec_{0};
    double crop_ratio_{0.0};
    int jpeg_quality_{90};
    int encoding_threads_{1};
    int decoder_threads_{1};
    bool crop_enabled_{false};
    std::string front_cropped_topic_;
    std::string rear_cropped_topic_;

    rosbag2_cpp::Writer writer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InsvDualFisheyeBagNode>();
    rclcpp::shutdown();
    return 0;
}
