// Authors: Jinyuan Guo
// In Reference to ExifTool ProcessInsta360(): https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/QuickTimeStream.pl

#include "insv/insta360_trailer_parser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

namespace insta360_insv {
namespace {
constexpr size_t kFooterSize = 6;            // 2-byte id + 4-byte length
constexpr size_t kTrailerProbeSize = 78;     // minimum footer span used by ExifTool logic
constexpr const char* kMagicHex = "8db42d694ccc418790edff439fe026bf";
constexpr size_t kMagicAsciiLen = 32;   // length of kMagicHex string
constexpr size_t kMagicBinaryLen = kMagicAsciiLen / 2;
constexpr size_t kMaxTailSearchBytes = 32 * 1024 * 1024;
constexpr std::array<uint8_t, kMagicBinaryLen> kMagicBinary = {
    0x8d, 0xb4, 0x2d, 0x69, 0x4c, 0xcc, 0x41, 0x87,
    0x90, 0xed, 0xff, 0x43, 0x9f, 0xe0, 0x26, 0xbf};

std::string ToHex(const uint8_t* data, size_t len) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(hex[(data[i] >> 4) & 0xF]);
        out.push_back(hex[data[i] & 0xF]);
    }
    return out;
}

size_t FindPatternReverse(const std::vector<uint8_t>& buffer, const uint8_t* pattern, size_t pattern_len) {
    if (pattern_len == 0 || buffer.size() < pattern_len) {
        return std::string::npos;
    }
    for (size_t pos = buffer.size() - pattern_len;; --pos) {
        if (std::memcmp(buffer.data() + pos, pattern, pattern_len) == 0) {
            return pos;
        }
        if (pos == 0) {
            break;
        }
    }
    return std::string::npos;
}
}  // namespace

uint16_t TrailerParser::ReadU16LE(const uint8_t* data) {
    return static_cast<uint16_t>(data[0] | (static_cast<uint16_t>(data[1]) << 8));
}

uint32_t TrailerParser::ReadU32LE(const uint8_t* data) {
    return static_cast<uint32_t>(data[0] | (static_cast<uint32_t>(data[1]) << 8) |
                                 (static_cast<uint32_t>(data[2]) << 16) |
                                 (static_cast<uint32_t>(data[3]) << 24));
}

uint64_t TrailerParser::ReadU64LE(const uint8_t* data) {
    return static_cast<uint64_t>(ReadU32LE(data)) |
           (static_cast<uint64_t>(ReadU32LE(data + 4)) << 32);
}

double TrailerParser::ReadF64LE(const uint8_t* data) {
    uint64_t raw = ReadU64LE(data);
    double value;
    std::memcpy(&value, &raw, sizeof(double));
    return value;
}

bool TrailerParser::ParseFile(const std::string& path, std::vector<ImuSample>& out_samples, std::string* error_out) const {
    out_samples.clear();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error_out) *error_out = "Failed to open file";
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamoff file_size = file.tellg();
    if (file_size < static_cast<std::streamoff>(kTrailerProbeSize)) {
        if (error_out) *error_out = "File too small to contain Insta360 trailer";
        return false;
    }

    // Read the probe buffer containing trailer footer + magic.
    std::string footer_error;
    {
        file.seekg(file_size - static_cast<std::streamoff>(kTrailerProbeSize));
        std::vector<uint8_t> probe(kTrailerProbeSize);
        file.read(reinterpret_cast<char*>(probe.data()), probe.size());
        if (file.gcount() == static_cast<std::streamsize>(probe.size())) {
            bool has_magic = false;
            bool ascii_magic_found = false;
            bool binary_magic_found = false;
            if (probe.size() >= kMagicAsciiLen) {
                const char* ascii_ptr = reinterpret_cast<const char*>(probe.data() + probe.size() - kMagicAsciiLen);
                const std::string ascii_magic_str(ascii_ptr, kMagicAsciiLen);
                if (ascii_magic_str == kMagicHex) {
                    has_magic = true;
                    ascii_magic_found = true;
                }
            }
            if (!has_magic && probe.size() >= kMagicBinaryLen) {
                const std::string binary_magic_str = ToHex(probe.data() + probe.size() - kMagicBinaryLen, kMagicBinaryLen);
                if (binary_magic_str == kMagicHex) {
                    has_magic = true;
                    binary_magic_found = true;
                }
            }

            if (has_magic) {
                auto read_len_valid = [&](const uint8_t* ptr) -> std::optional<uint32_t> {
                    const uint32_t candidate = ReadU32LE(ptr);
                    if (candidate == 0 || candidate > static_cast<uint64_t>(file_size)) {
                        return std::nullopt;
                    }
                    return candidate;
                };

                std::optional<uint32_t> trailer_len;
                if (binary_magic_found && probe.size() >= 0x38 + sizeof(uint32_t)) {
                    trailer_len = read_len_valid(probe.data() + 0x38);
                }
                if (!trailer_len && ascii_magic_found && probe.size() >= 2 + sizeof(uint32_t)) {
                    trailer_len = read_len_valid(probe.data() + 2);
                }
                if (!trailer_len) {
                    trailer_len = read_len_valid(probe.data());
                }

                if (trailer_len) {
                    file.seekg(file_size - static_cast<std::streamoff>(*trailer_len));
                    std::vector<uint8_t> trailer(*trailer_len);
                    file.read(reinterpret_cast<char*>(trailer.data()), trailer.size());
                    if (file.gcount() == static_cast<std::streamsize>(trailer.size())) {
                        std::vector<ImuSample> tmp_samples;
                        if (ParseTrailer(trailer, tmp_samples, &footer_error) && !tmp_samples.empty()) {
                            out_samples = std::move(tmp_samples);
                            return true;
                        }
                    } else {
                        footer_error = "Failed to read full trailer";
                    }
                } else {
                    footer_error = "Invalid trailer length";
                }
            } else {
                footer_error = "Insta360 magic trailer not found";
            }
        } else {
            footer_error = "Failed to read trailer probe";
        }
    }

    out_samples.clear();
    std::string scan_error;
    if (ScanTailForRecords(path, file_size, out_samples, &scan_error)) {
        return true;
    }

    if (error_out) {
        if (!scan_error.empty()) {
            *error_out = scan_error;
        } else if (!footer_error.empty()) {
            *error_out = footer_error;
        } else {
            *error_out = "Failed to parse trailer";
        }
    }
    return false;
}

bool TrailerParser::ParseTrailer(const std::vector<uint8_t>& trailer, std::vector<ImuSample>& out_samples, std::string* error_out) const {
    size_t cursor = trailer.size();

    while (cursor >= kFooterSize) {
        const size_t footer_pos = cursor - kFooterSize;
        const uint8_t* footer = trailer.data() + footer_pos;
        const uint16_t id = ReadU16LE(footer);
        const uint32_t len = ReadU32LE(footer + 2);

        if (len == 0) {
            cursor = footer_pos;
            continue;
        }

        if (footer_pos < len) {
            if (error_out) *error_out = "Trailer parse underflow";
            return false;
        }

        const size_t data_pos = footer_pos - len;
        const uint8_t* data = trailer.data() + data_pos;

        if (id == 0x300) {
            ParseImuRecord(id, data, len, out_samples);
        } else if (id == 0x600) {
            ParseVideoTimestampRecord(id, data, len, out_samples);
        }
        // Other record types can be added here when needed (0x400 exposure, 0x600 video ts, 0x700 GPS, etc.)

        cursor = data_pos;
    }

    return true;
}

bool TrailerParser::ScanTailForRecords(const std::string& path, std::streamoff file_size, std::vector<ImuSample>& out_samples, std::string* error_out) const {
    if (file_size <= 0) {
        if (error_out) *error_out = "File is empty";
        return false;
    }

    const size_t tail_size = static_cast<size_t>(std::min<std::streamoff>(file_size, static_cast<std::streamoff>(kMaxTailSearchBytes)));
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error_out) *error_out = "Failed to reopen file for tail scan";
        return false;
    }

    file.seekg(file_size - static_cast<std::streamoff>(tail_size));
    std::vector<uint8_t> buffer(tail_size);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (file.gcount() != static_cast<std::streamsize>(buffer.size())) {
        if (error_out) *error_out = "Failed to read tail segment";
        return false;
    }

    size_t search_end = buffer.size();
    const size_t ascii_pos = FindPatternReverse(buffer, reinterpret_cast<const uint8_t*>(kMagicHex), kMagicAsciiLen);
    if (ascii_pos != std::string::npos) {
        search_end = ascii_pos;
    }
    const size_t binary_pos = FindPatternReverse(buffer, kMagicBinary.data(), kMagicBinary.size());
    if (binary_pos != std::string::npos) {
        search_end = std::min(search_end, binary_pos);
    }

    bool found_supported = false;
    size_t cursor = search_end;
    while (cursor >= kFooterSize) {
        const size_t footer_pos = cursor - kFooterSize;
        const uint16_t id = ReadU16LE(buffer.data() + footer_pos);
        if (id != 0x300 && id != 0x600) {
            if (cursor == 0) {
                break;
            }
            cursor -= 1;
            continue;
        }

        const uint32_t len = ReadU32LE(buffer.data() + footer_pos + 2);
        if (len == 0 || len > footer_pos) {
            if (cursor == 0) {
                break;
            }
            cursor -= 1;
            continue;
        }

        const size_t data_pos = footer_pos - len;
        const uint8_t* data = buffer.data() + data_pos;
        if (id == 0x300) {
            ParseImuRecord(id, data, len, out_samples);
            found_supported = true;
        } else {
            ParseVideoTimestampRecord(id, data, len, out_samples);
            found_supported = true;
        }
        cursor = data_pos;
    }

    if (!found_supported) {
        if (error_out) *error_out = "No supported trailer records found near file end";
        return false;
    }
    return true;
}

void TrailerParser::ParseImuRecord(uint16_t /*id*/, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const {
    // Determine record stride: 56-byte (double precision) or 20-byte (scaled shorts).
    size_t stride = 0;
    if (len % 56 == 0) {
        stride = 56;
    } else if (len % 20 == 0) {
        stride = 20;
    } else if (len >= 56 && (len - 56) % 20 == 0) {
        // Mixed types are unlikely; prefer double block if present.
        stride = 56;
    } else if (len >= 20) {
        stride = 20;
    }

    if (stride == 0) {
        return;
    }

    for (size_t offset = 0; offset + stride <= len; offset += stride) {
        const uint8_t* rec = data + offset;
        const uint64_t raw_time = ReadU64LE(rec);
        if (raw_time == 0 || raw_time == std::numeric_limits<uint64_t>::max()) {
            // Skip padding/marker entries commonly present in trailers
            continue;
        }
        ImuSample sample;
        sample.raw_time = static_cast<double>(raw_time);
        sample.time_sec = std::numeric_limits<double>::quiet_NaN();

        if (stride == 56) {
            sample.ax = ReadF64LE(rec + 8);
            sample.ay = ReadF64LE(rec + 16);
            sample.az = ReadF64LE(rec + 24);
            sample.gx = ReadF64LE(rec + 32);
            sample.gy = ReadF64LE(rec + 40);
            sample.gz = ReadF64LE(rec + 48);
            sample.high_precision = true;
        } else {
            // 6x uint16_t, stored after 8-byte timestamp.
            const uint16_t* raw = reinterpret_cast<const uint16_t*>(rec + 8);
            const double scale = 1.0 / 1000.0;
            auto convert = [scale](uint16_t v) {
                return (static_cast<int>(v) - 0x8000) * scale;
            };
            sample.ax = convert(raw[0]);
            sample.ay = convert(raw[1]);
            sample.az = convert(raw[2]);
            sample.gx = convert(raw[3]);
            sample.gy = convert(raw[4]);
            sample.gz = convert(raw[5]);
            sample.high_precision = false;
        }

        // Optional physical plausibility filter to suppress junk
        const double abs_ax = std::fabs(sample.ax);
        const double abs_ay = std::fabs(sample.ay);
        const double abs_az = std::fabs(sample.az);
        const double abs_gx = std::fabs(sample.gx);
        const double abs_gy = std::fabs(sample.gy);
        const double abs_gz = std::fabs(sample.gz);
        const bool accel_ok = (abs_ax < 16.0 && abs_ay < 16.0 && abs_az < 16.0);
        const bool gyro_ok = (abs_gx < 2000.0 && abs_gy < 2000.0 && abs_gz < 2000.0);
        if (!(accel_ok && gyro_ok)) {
            continue;
        }

        out_samples.push_back(sample);
    }
}

void TrailerParser::ParseVideoTimestampRecord(uint16_t /*id*/, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const {
    // Video timestamp records are observed as 0x600, with an 8-byte millisecond timestamp at the start.
    size_t stride = 0;
    if (len % 12 == 0) {
        stride = 12; // common pattern: 8-byte ts + 4 bytes padding/flags
    } else if (len % 8 == 0) {
        stride = 8; // fallback: only timestamp
    } else if (len >= 8) {
        stride = 8; // best-effort
    }

    if (stride == 0) {
        return;
    }

    for (size_t offset = 0; offset + stride <= len; offset += stride) {
        const uint8_t* rec = data + offset;
        const uint64_t raw = ReadU64LE(rec);
        if (raw == 0 || raw == std::numeric_limits<uint64_t>::max()) {
            continue;
        }

        ImuSample sample;
        sample.raw_time = static_cast<double>(raw);
        sample.time_sec = std::numeric_limits<double>::quiet_NaN();
        sample.is_video_ts = true;
        sample.video_ts_sec = std::numeric_limits<double>::quiet_NaN();

        out_samples.push_back(sample);
    }
}

}  // namespace insta360_insv
