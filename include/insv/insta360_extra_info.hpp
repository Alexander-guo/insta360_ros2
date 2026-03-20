#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace insta360_insv::extra_info {

namespace RecordType {
constexpr uint8_t Offsets = 0;
constexpr uint8_t Metadata = 1;
constexpr uint8_t Thumbnail = 2;
constexpr uint8_t Gyro = 3;
constexpr uint8_t Exposure = 4;
constexpr uint8_t ThumbnailExt = 5;
constexpr uint8_t TimelapseTimestamp = 6;   // timelapse / interval photos or interval video, not normal continuous video.
constexpr uint8_t Gps = 7;
constexpr uint8_t StarNum = 8;
constexpr uint8_t AAAData = 9;
constexpr uint8_t Anchors = 10;
constexpr uint8_t AAASimulation = 11;
constexpr uint8_t ExposureSecondary = 12;
constexpr uint8_t Magnetic = 13;
constexpr uint8_t Euler = 14;
constexpr uint8_t SecGyro = 15;
constexpr uint8_t Speed = 16;
constexpr uint8_t TBox = 17;
constexpr uint8_t Quaternions = 18;
constexpr uint8_t TimeMap = 128;
}  // namespace RecordType

namespace RecordFormat {
constexpr uint8_t Binary = 0;
constexpr uint8_t Protobuf = 1;
constexpr uint8_t Json = 2;
}  // namespace RecordFormat

enum class ExtraType : int32_t {
    All = 0,
    Metadata = 1,
    Thumbnail = 2,
    Gyro = 3,
    Exposure = 4,
    ExtThumbnail = 5,
    FramePts = 6,
    Gps = 7,
    StarNum = 8,
    AaaData = 9,
    Highlight = 10,
    AaaSim = 11,
    ExposureSecondary = 12,
    Magnetic = 13,
    Euler = 14,
    SecGyro = 15,
    Speed = 16,
    TBox = 17,
    Quaternions = 18,
    TimeMap = 128,
};

struct GyroConfigInfo {
    uint32_t acc_range{0};
    uint32_t gyro_range{0};
};

struct ExtraMetadata {
    std::string serial_number;
    std::string camera_type;
    std::string fw_version;
    std::string file_type;
    std::string offset;
    std::string ip;
    uint64_t creation_time{0};
    uint64_t export_time{0};
    uint64_t file_size{0};
    uint32_t total_time{0};

    std::vector<uint8_t> gps;
    std::vector<uint8_t> orientation;
    std::vector<uint8_t> gyro;
    int32_t frame_rate{0};
    int64_t first_frame_timestamp{0};
    double rolling_shutter_time{0.0};
    double gyro_timestamp{0.0};
    bool is_has_gyro_timestamp{false};
    uint32_t timelapse_interval{0};
    std::vector<uint8_t> gyro_calib;

    std::string original_offset;
    std::string original_offset_3d;
    std::string offset_v2;
    std::string offset_v3;
    std::string original_offset_v2;
    std::string original_offset_v3;

    bool is_collected{false};
    uint64_t recycle_time{0};
    uint32_t total_frames{0};
    bool is_selfie{false};
    bool is_flowstate_online{false};
    bool is_dewarp{false};

    std::vector<float> photo_rot;
    bool is_raw_gyro{false};
    int32_t raw_capture_type{0};
    int32_t pts_type{0};
    GyroConfigInfo gyro_cfg_info{};
};

struct ParsedGyroCalib {
    std::array<double, 6> numbers{};
    uint64_t unix_timestamp{0};
};

struct ParsedGyro {
    std::array<double, 6> numbers{};
    uint64_t timestamp{0};
};

bool ParseGyroCalib(const uint8_t* data, size_t len, ParsedGyroCalib& out);
bool ParseGyro(const uint8_t* data, size_t len, ParsedGyro& out);
bool ParseOffset(const std::string& text, std::vector<double>& out);

}  // namespace insta360_insv::extra_info
