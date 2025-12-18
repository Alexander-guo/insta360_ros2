// Authors: Jinyuan Guo
// In Reference to: https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/QuickTimeStream.pl

#include "insv/insta360_trailer_parser.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace insta360_insv {
namespace {
constexpr size_t kFooterSize = 6;            // 2-byte id + 4-byte length
constexpr size_t kTrailerProbeSize = 78;     // minimum footer span used by ExifTool logic
constexpr const char* kMagicHex = "8db42d694ccc418790edff439fe026bf";

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
    file.seekg(file_size - static_cast<std::streamoff>(kTrailerProbeSize));
    std::vector<uint8_t> probe(kTrailerProbeSize);
    file.read(reinterpret_cast<char*>(probe.data()), probe.size());
    if (file.gcount() != static_cast<std::streamsize>(probe.size())) {
        if (error_out) *error_out = "Failed to read trailer probe";
        return false;
    }

    // Magic number lives in the last 32 bytes of the probe.
    const std::string magic = ToHex(probe.data() + probe.size() - 16, 16);
    if (magic != kMagicHex) {
        if (error_out) *error_out = "Insta360 magic trailer not found";
        return false;
    }

    // Trailer length is at offset 0x38 within the probe (little-endian uint32).
    const uint32_t trailer_len = ReadU32LE(probe.data() + 0x38);
    if (trailer_len == 0 || trailer_len > static_cast<uint64_t>(file_size)) {
        if (error_out) *error_out = "Invalid trailer length";
        return false;
    }

    // Load the entire trailer block.
    file.seekg(file_size - static_cast<std::streamoff>(trailer_len));
    std::vector<uint8_t> trailer(trailer_len);
    file.read(reinterpret_cast<char*>(trailer.data()), trailer.size());
    if (file.gcount() != static_cast<std::streamsize>(trailer.size())) {
        if (error_out) *error_out = "Failed to read full trailer";
        return false;
    }

    return ParseTrailer(trailer, out_samples, error_out);
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
        const uint64_t time_ms = ReadU64LE(rec);
        ImuSample sample;
        sample.time_sec = static_cast<double>(time_ms) / 1000.0;

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
        const uint64_t time_ms = ReadU64LE(rec);

        ImuSample sample;
        sample.time_sec = static_cast<double>(time_ms) / 1000.0;
        sample.video_ts_sec = sample.time_sec;
        sample.is_video_ts = true;

        out_samples.push_back(sample);
    }
}

}  // namespace insta360_insv
