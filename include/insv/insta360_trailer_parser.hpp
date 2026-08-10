#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <limits>

namespace insta360_insv {

struct ImuSample {
    // Final aligned time in seconds (after matching to video timestamps).
    double time_sec{std::numeric_limits<double>::quiet_NaN()};
    // Raw IMU timestamp, stored unscaled.
    double raw_time{std::numeric_limits<double>::quiet_NaN()};
    // Accelerometer readings in g, gyroscope readings in rad/s, all after applying scale and offset from metadata.
    double ax{0.0};
    double ay{0.0};
    double az{0.0};
    double gx{0.0};
    double gy{0.0};
    double gz{0.0};
};

class TrailerParser {
public:
    struct OffsetEntry {
        uint8_t id{0};
        uint8_t format{0};
        uint32_t size{0};
        uint32_t rel_offset{0};
    };

    struct ExtraMetadataState {
        std::string model;
        bool have_first_frame_timestamp{false};
        double first_frame_timestamp{0.0};
        double first_frame_timestamp_sec{0.0};
        bool have_gyro_timestamp{false};
        double gyro_timestamp_ms{0.0};
        bool have_frame_readout_time{false};
        double frame_readout_time{0.0};
        bool has_offset_v3{false};
    };

    // Parse an INSV/LRV/MP4 file containing an Insta360 trailer and collect samples.
    // Returns true on success. When false, error_out (if provided) contains a brief reason.
    bool ParseFile(const std::string& path, std::vector<ImuSample>& out_samples, std::string* error_out = nullptr) const;

    void SetVerbose(bool verbose) const { verbose_ = verbose; }

    const ExtraMetadataState& GetMetadataState() const { return metadata_state_; }

private:
    struct ImuCalibration {
        bool have_ranges{false};
        bool is_raw_gyro{false};
        double gyro_range{2000.0};  // deg/s
        double acc_range{16.0};     // g
    };

    mutable ImuCalibration imu_calib_{};
    mutable ExtraMetadataState metadata_state_{};
    mutable bool verbose_{false};

    static uint16_t ReadU16LE(const uint8_t* data);
    static uint32_t ReadU32LE(const uint8_t* data);
    static uint64_t ReadU64LE(const uint8_t* data);
    static double ReadF64LE(const uint8_t* data);

    // New-style Insta360 trailer parser based on the official extra data
    // layout used by AdrianEddy/telemetry-parser (HEADER_SIZE + records
    // stored as [data][format][id][size] at the end of the file).
    bool ParseInsta360Extra(const std::string& path,
                            std::streamoff file_size,
                            std::vector<ImuSample>& out_samples,
                            std::string* error_out) const;

    void ApplyMetadataPostProcess(std::vector<ImuSample>& out_samples) const;

    bool ParseMetadataRecord(const uint8_t* data, size_t len) const;
    bool ParseGyroConfigInfo(const uint8_t* data, size_t len, double& acc_range, double& gyro_range) const;

    void ParseRecordByType(uint8_t id,
                           uint8_t format,
                           const uint8_t* data,
                           size_t len,
                           std::vector<ImuSample>& out_samples) const;

    bool ParseTrailer(const std::vector<uint8_t>& trailer, std::vector<ImuSample>& out_samples, std::string* error_out) const;
    bool ScanTrailerWindow(const std::string& path, size_t start_offset, size_t window_size, std::vector<ImuSample>& out_samples, std::string* error_out) const;
    std::vector<OffsetEntry> ParseOffsetsRecord(const uint8_t* data, size_t len) const;
    void ParseTimeMapRecord(const uint8_t* data, size_t len) const;
    void ParseThumbnailRecord(const uint8_t* data, size_t len, bool is_ext) const;
    void ParseImuRecord(uint16_t id, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const;
    void ParseExposureRecord(const uint8_t* data, size_t len) const;
    void ParseTimelapseTimestampRecord(const uint8_t *data, size_t len) const;
    void ParseGpsRecord(const uint8_t* data, size_t len) const;
    void ParseAaaDataRecord(const uint8_t* data, size_t len) const;
    void ParseAnchorsRecord(const uint8_t* data, size_t len) const;
};

}  // namespace insta360_insv
