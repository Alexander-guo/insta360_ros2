#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <limits>

namespace insta360_insv {

struct ImuSample {
    double time_sec{0.0};
    double ax{0.0};
    double ay{0.0};
    double az{0.0};
    double gx{0.0};
    double gy{0.0};
    double gz{0.0};
    bool high_precision{false};

    // For 0x600 video timestamp records; when false, fields below are NaN.
    bool is_video_ts{false};
    double video_ts_sec{std::numeric_limits<double>::quiet_NaN()};
};

class TrailerParser {
public:
    // Parse an INSV/INSP/MP4 file containing an Insta360 trailer footer and collect IMU samples.
    // Returns true on success. When false, error_out (if provided) contains a brief reason.
    bool ParseFile(const std::string& path, std::vector<ImuSample>& out_samples, std::string* error_out = nullptr) const;

private:
    static uint16_t ReadU16LE(const uint8_t* data);
    static uint32_t ReadU32LE(const uint8_t* data);
    static uint64_t ReadU64LE(const uint8_t* data);
    static double ReadF64LE(const uint8_t* data);

    bool ParseTrailer(const std::vector<uint8_t>& trailer, std::vector<ImuSample>& out_samples, std::string* error_out) const;
    void ParseImuRecord(uint16_t id, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const;
    void ParseVideoTimestampRecord(uint16_t id, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const;
};

}  // namespace insta360_insv
