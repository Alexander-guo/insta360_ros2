// Authors: Jinyuan Guo
// In Reference to ExifTool ProcessInsta360(): https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/QuickTimeStream.pl
// and Telemetry-parser's Insta360 module: https://github.com/AdrianEddy/telemetry-parser/tree/master/src/insta360

#include "insv/insta360_trailer_parser.hpp"
#include "insv/insta360_extra_info.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace insta360_insv {
namespace {
constexpr size_t kTrailerProbeSize = 78;     // minimum footer span used by ExifTool logic
constexpr const char* kMagicHex = "8db42d694ccc418790edff439fe026bf";
constexpr size_t kMagicAsciiLen = 32;   // length of kMagicHex string
constexpr size_t kInstaHeaderSize = 32 + 4 + 4 + 32;  // padding(32) + size(4) + version(4) + magic(32)

double PickAxis(double x, double y, double z, char axis_char) {
    const bool positive = (axis_char >= 'A' && axis_char <= 'Z');
    const char axis = static_cast<char>(positive ? (axis_char - 'A' + 'a') : axis_char);
    double v = 0.0;
    if (axis == 'x') v = x;
    else if (axis == 'y') v = y;
    else if (axis == 'z') v = z;
    return positive ? v : -v;
}

void ApplyOrientation(double& x, double& y, double& z, const std::string& orientation) {
    if (orientation.size() < 3) {
        return;
    }
    const double ox = x;
    const double oy = y;
    const double oz = z;
    x = PickAxis(ox, oy, oz, orientation[0]);
    y = PickAxis(ox, oy, oz, orientation[1]);
    z = PickAxis(ox, oy, oz, orientation[2]);
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
    // Reset per-call IMU calibration derived from Insta360 metadata.
    imu_calib_ = ImuCalibration{};
    metadata_state_ = ExtraMetadataState{};

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

    // Preferred path: parse the official Insta360 extra data block as
    // implemented in AdrianEddy/telemetry-parser. This relies on the
    // fixed HEADER layout at the end of the file and records stored as
    // [data][format][id][size]. If this fails or is not present, we fall
    // back to the older QuickTime-style trailer scanning below.
    std::string insta360_error;
    if (ParseInsta360Extra(path, file_size, out_samples, &insta360_error) && !out_samples.empty()) {
        ApplyMetadataPostProcess(out_samples);
        return true;
    }

    if (error_out) {
        if (!insta360_error.empty()) {
            *error_out = insta360_error;
        } else {
            *error_out = "Failed to parse trailer";
        }
    }
    return false;
}

void TrailerParser::ApplyMetadataPostProcess(std::vector<ImuSample>& out_samples) const {
    if (out_samples.empty()) {
        return;
    }

    const double fft = metadata_state_.have_first_frame_timestamp
                               ? (metadata_state_.first_frame_timestamp / 1000.0)
                               : 0.0;
    const double gyro_ts_sec = metadata_state_.have_gyro_timestamp
                                   ? (metadata_state_.gyro_timestamp_ms / 1000.0)
                                   : 0.0;
    const bool is_raw = imu_calib_.is_raw_gyro;

    std::string imu_orientation = "Xyz";
    if (metadata_state_.has_offset_v3) {
        if (metadata_state_.model == "Insta360 GO 2") imu_orientation = "XYZ";
        else if (metadata_state_.model == "Insta360 GO 3") imu_orientation = "XYZ";
        else if (metadata_state_.model == "Insta360 GO 3S") imu_orientation = "yXZ";
        else if (metadata_state_.model == "Insta360 GO Ultra") imu_orientation = "YxZ";
        else if (metadata_state_.model == "Insta360 OneR") imu_orientation = "Xyz";
        else if (metadata_state_.model == "Insta360 OneRS") imu_orientation = "Xyz";
        else if (metadata_state_.model == "Insta360 X4") imu_orientation = "yzX";
        else if (metadata_state_.model == "Insta360 X5") imu_orientation = "yzX";
        else imu_orientation = "Xyz";
    } else {
        if (metadata_state_.model == "Insta360 Go") imu_orientation = "xyZ";
        else if (metadata_state_.model == "Insta360 GO 2") imu_orientation = "yXZ";
        else if (metadata_state_.model == "Insta360 OneR") imu_orientation = "yXZ";
        else if (metadata_state_.model == "Insta360 OneRS") imu_orientation = "yxz";
        else if (metadata_state_.model == "Insta360 ONE X2") imu_orientation = "xZy";
        else imu_orientation = "yXZ";
    }
    printf("Applying IMU axis remapping with orientation '%s' based on camera model '%s' and offset_v3=%d\n",
           imu_orientation.c_str(), metadata_state_.model.c_str(), metadata_state_.has_offset_v3 ? 1 : 0);
    fflush(stdout);

    for (auto& s : out_samples) {
        // Telemetry-parser style timestamp normalization for IMU samples:
        //   1) t1 = t_raw - first_frame_timestamp -- align to video start, makes video start at t = 0
        //   2) raw-only: t2 = t1 / 1000
        //   3) t_final = t2 - gyro_timestamp -- remove gyro offset, aligns IMU clock with video clock
        // We preserve raw_time and write normalized time into time_sec.
        double t = s.raw_time / 1000.0;
        t -= fft;
        if (is_raw) {
            t /= 1000.0;
        }
        t -= gyro_ts_sec;
        s.time_sec = t;
        
        // Apply orientation remapping to IMU samples based on the model-specific layout used by telemetry-parser. 
        // This ensures that the final IMU axes are consistently oriented across different camera models.
        ApplyOrientation(s.ax, s.ay, s.az, imu_orientation);
        ApplyOrientation(s.gx, s.gy, s.gz, imu_orientation);
    }
}

// Parse ExtraMetadata protobuf (insta360::extra_info::ExtraMetadata) just
// enough to retrieve:
//   - is_raw_gyro (field 62, bool)
//   - gyro_cfg_info (field 65, message GyroConfigInfo)
// which provides:
//   - acc_range (field 1, uint32)
//   - gyro_range (field 2, uint32)
// Parse the Insta360 "extra" trailer block as implemented in
// AdrianEddy/telemetry-parser's insta360 module. The layout is:
//   - A fixed-size header at EOF: padding(32) + extra_size(4) + version(4) + magic(32)
//   - Immediately before the header: a sequence of records stored as
//       [record_data (size bytes)] [format u8] [id u8] [size u32],
//     packed back-to-back inside the extra_size region.
//
// We scan this region from the end towards the front, decoding
//   - Gyro records (id = 3) as IMU data
// and append them into out_samples. On any structural error we stop and
// let the caller fall back to the older heuristic logic.
bool TrailerParser::ParseInsta360Extra(const std::string& path,
                                       std::streamoff file_size,
                                       std::vector<ImuSample>& out_samples,
                                       std::string* error_out) const {
    if (file_size <= static_cast<std::streamoff>(kInstaHeaderSize)) {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error_out) *error_out = "Failed to open file in Insta360 extra parser";
        return false;
    }

    // Read the fixed-size header at the end of the file.
    file.seekg(file_size - static_cast<std::streamoff>(kInstaHeaderSize));
    std::array<uint8_t, kInstaHeaderSize> header{};
    file.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (file.gcount() != static_cast<std::streamsize>(header.size())) {
        return false;
    }

    // Check for the Insta360 magic at the end of the header (ASCII hex string).
    if (std::memcmp(header.data() + kInstaHeaderSize - kMagicAsciiLen, kMagicHex, kMagicAsciiLen) != 0) {
        // Not an Insta360 extra-data footer; let caller fall back.
        return false;
    }

    const uint32_t extra_size32 = ReadU32LE(header.data() + 32);
    const uint32_t version = ReadU32LE(header.data() + 36);

    if (extra_size32 < kInstaHeaderSize + 4 + 1 + 1 ||
        static_cast<std::streamoff>(extra_size32) > file_size) {
        if (error_out) *error_out = "Invalid Insta360 extra size in header";
        return false;
    }

    const size_t extra_size = static_cast<size_t>(extra_size32);
    const size_t extra_start = static_cast<size_t>(file_size) - extra_size;

    printf("Insta360 extra header detected: version=%u extra_size=%u extra_start=%zu file_size=%lld\n",
           version, extra_size32, extra_start, static_cast<long long>(file_size));
    fflush(stdout);

    // Records are stored inside the [extra_start, file_size) region as
    // [data][format][id][size], packed back-to-back before the header.
    // We iterate from the end towards the beginning, mirroring
    // telemetry-parser's approach.
    constexpr size_t kRecordTrailerSize = 4 + 1 + 1;  // size (u32) + id (u8) + format (u8)
    if (extra_size <= kInstaHeaderSize + kRecordTrailerSize) {
        if (error_out) *error_out = "Insta360 extra region too small for any records";
        return false;
    }

    // First, attempt to use the Offsets directory record (id = 0) if it is
    // present as the first record, mirroring telemetry-parser's logic. This
    // gives us direct offsets to each record inside the extra region.
    size_t offset = kInstaHeaderSize + kRecordTrailerSize;
    bool found_supported = false;

    if (offset + kRecordTrailerSize <= extra_size) {
        const std::streamoff first_hdr_pos =
            file_size - static_cast<std::streamoff>(offset);
        if (first_hdr_pos >= 0) {
            char rec_hdr[static_cast<size_t>(kRecordTrailerSize)] = {};
            file.seekg(first_hdr_pos);
            file.read(rec_hdr, static_cast<std::streamsize>(kRecordTrailerSize));
            if (file.gcount() == static_cast<std::streamsize>(kRecordTrailerSize)) {
                const uint8_t first_format = static_cast<uint8_t>(rec_hdr[0]);
                const uint8_t first_id     = static_cast<uint8_t>(rec_hdr[1]);
                const uint32_t first_size  = ReadU32LE(reinterpret_cast<const uint8_t*>(rec_hdr + 2));

                if (first_id == 0 && first_size > 0 &&
                    offset + kRecordTrailerSize + static_cast<size_t>(first_size) <= extra_size) {
                    printf("Offsets record found in Insta360 extra data, parsing with Offsets Table ...\n");
                    fflush(stdout);
                    const std::streamoff first_data_pos =
                        file_size - static_cast<std::streamoff>(offset + static_cast<size_t>(first_size));
                    if (first_data_pos >= 0) {
                        std::vector<uint8_t> offsets_buf(first_size);
                        file.seekg(first_data_pos);
                        file.read(reinterpret_cast<char*>(offsets_buf.data()), static_cast<std::streamsize>(first_size));
                        if (file.gcount() == static_cast<std::streamsize>(first_size)) {
                            (void)first_format;  // Currently unused, kept for completeness.

                            const auto dir_entries = ParseOffsetsRecord(offsets_buf.data(), offsets_buf.size());

                            for (const auto& e : dir_entries) {
                                const size_t record_size = static_cast<size_t>(e.size);
                                const size_t rel_offset  = static_cast<size_t>(e.rel_offset);

                                if (record_size == 0) {
                                    continue;
                                }

                                const std::streamoff data_pos =
                                    static_cast<std::streamoff>(extra_start + rel_offset);
                                const std::streamoff hdr_pos =
                                    data_pos + static_cast<std::streamoff>(record_size);
                                if (data_pos < 0 || hdr_pos < 0) {
                                    continue;
                                }
                                if (hdr_pos + static_cast<std::streamoff>(kRecordTrailerSize) > file_size) {
                                    continue;
                                }

                                std::vector<uint8_t> buf(record_size);
                                file.seekg(data_pos);
                                file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(record_size));
                                if (file.gcount() != static_cast<std::streamsize>(record_size)) {
                                    continue;
                                }

                                char rec_hdr2[static_cast<size_t>(kRecordTrailerSize)] = {};
                                file.seekg(hdr_pos);
                                file.read(rec_hdr2, static_cast<std::streamsize>(kRecordTrailerSize));
                                if (file.gcount() != static_cast<std::streamsize>(kRecordTrailerSize)) {
                                    continue;
                                }

                                const uint8_t format = static_cast<uint8_t>(rec_hdr2[0]);
                                const uint8_t id_u8  = static_cast<uint8_t>(rec_hdr2[1]);
                                const uint32_t size2 = ReadU32LE(reinterpret_cast<const uint8_t*>(rec_hdr2 + 2));
                                if (size2 != e.size || id_u8 != e.id || id_u8 == 0) {
                                    continue;
                                }

                                ParseRecordByType(id_u8, format, buf.data(), buf.size(), out_samples);
                                if (id_u8 == insta360_insv::extra_info::RecordType::Gyro) {
                                    found_supported = true;
                                }
                            }

                            if (found_supported) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: scan records linearly from the end of the extra region,
    // as in the original implementation and telemetry-parser's non-offsets
    // path.
    printf("Offsets record not found or invalid, falling back to linear scan of Insta360 extra region\n");
    fflush(stdout);

    offset = kInstaHeaderSize + kRecordTrailerSize;
    found_supported = false;

    while (offset < extra_size) {
        if (offset + kRecordTrailerSize > extra_size) {
            break;
        }

        const std::streamoff rec_hdr_pos =
            file_size - static_cast<std::streamoff>(offset);
        if (rec_hdr_pos < 0) {
            break;
        }

        char rec_hdr[static_cast<size_t>(kRecordTrailerSize)] = {};
        file.seekg(rec_hdr_pos);
        file.read(rec_hdr, static_cast<std::streamsize>(kRecordTrailerSize));
        if (file.gcount() != static_cast<std::streamsize>(kRecordTrailerSize)) {
            break;
        }

        const uint8_t format = static_cast<uint8_t>(rec_hdr[0]);
        const uint8_t id_u8  = static_cast<uint8_t>(rec_hdr[1]);
        const uint32_t size  = ReadU32LE(reinterpret_cast<const uint8_t*>(rec_hdr + 2));

        if (size == 0 || size > extra_size) {
            if (error_out) *error_out = "Malformed Insta360 record size";
            break;
        }

        if (offset + kRecordTrailerSize + static_cast<size_t>(size) > extra_size) {
            if (error_out) *error_out = "Insta360 record would exceed extra region";
            break;
        }

        const std::streamoff data_pos =
            file_size - static_cast<std::streamoff>(offset + static_cast<size_t>(size));
        if (data_pos < 0) {
            break;
        }

        std::vector<uint8_t> buf(size);
        file.seekg(data_pos);
        file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
        if (file.gcount() != static_cast<std::streamsize>(size)) {
            if (error_out) *error_out = "Failed to read Insta360 record payload";
            break;
        }

        ParseRecordByType(id_u8, format, buf.data(), buf.size(), out_samples);
        if (id_u8 == insta360_insv::extra_info::RecordType::Gyro) {
            found_supported = true;
        }

        offset += static_cast<size_t>(size) + kRecordTrailerSize;
    }

    if (!found_supported) {
        if (error_out && error_out->empty()) {
            *error_out = "No supported Insta360 records (gyro) found in extra region";
        }
        return false;
    }

    return true;
}
}  // namespace insta360_insv
