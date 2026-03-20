#include "insv/insta360_trailer_parser.hpp"

#include "insv/insta360_extra_info.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace insta360_insv {
namespace {

bool ReadVarint(const uint8_t* data, size_t len, size_t& pos, uint64_t& value) {
    value = 0;
    uint32_t shift = 0;
    while (pos < len && shift < 64) {
        const uint8_t byte = data[pos++];
        value |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            return true;
        }
        shift += 7;
    }
    return false;
}

bool SkipField(const uint8_t* data, size_t len, size_t& pos, uint32_t wire_type) {
    switch (wire_type) {
    case 0: {
        uint64_t tmp;
        return ReadVarint(data, len, pos, tmp);
    }
    case 1:
        if (pos + 8 > len) return false;
        pos += 8;
        return true;
    case 2: {
        uint64_t l = 0;
        if (!ReadVarint(data, len, pos, l)) return false;
        if (pos + l > len) return false;
        pos += static_cast<size_t>(l);
        return true;
    }
    case 5:
        if (pos + 4 > len) return false;
        pos += 4;
        return true;
    default:
        return false;
    }
}

float ReadF32LE(const uint8_t* data) {
    const uint32_t raw = static_cast<uint32_t>(data[0]) |
                         (static_cast<uint32_t>(data[1]) << 8) |
                         (static_cast<uint32_t>(data[2]) << 16) |
                         (static_cast<uint32_t>(data[3]) << 24);
    float value;
    std::memcpy(&value, &raw, sizeof(float));
    return value;
}

const char* RecordTypeName(uint8_t id) {
    switch (id) {
    case extra_info::RecordType::Offsets: return "Offsets";
    case extra_info::RecordType::Metadata: return "Metadata";
    case extra_info::RecordType::Thumbnail: return "Thumbnail";
    case extra_info::RecordType::Gyro: return "Gyro";
    case extra_info::RecordType::Exposure: return "Exposure";
    case extra_info::RecordType::ThumbnailExt: return "ThumbnailExt";
    case extra_info::RecordType::TimelapseTimestamp: return "TimelapseTimestamp";
    case extra_info::RecordType::Gps: return "Gps";
    case extra_info::RecordType::StarNum: return "StarNum";
    case extra_info::RecordType::AAAData: return "AAAData";
    case extra_info::RecordType::Anchors: return "Anchors";
    case extra_info::RecordType::AAASimulation: return "AAASimulation";
    case extra_info::RecordType::ExposureSecondary: return "ExposureSecondary";
    case extra_info::RecordType::Magnetic: return "Magnetic";
    case extra_info::RecordType::Euler: return "Euler";
    case extra_info::RecordType::SecGyro: return "SecGyro";
    case extra_info::RecordType::Speed: return "Speed";
    case extra_info::RecordType::TBox: return "TBox";
    case extra_info::RecordType::Quaternions: return "Quaternions";
    case extra_info::RecordType::TimeMap: return "TimeMap";
    default: return "Unknown";
    }
}

bool DumpRecordsEnabled(bool verbose) {
    if (verbose) {
        return true;
    }
    static const bool enabled = []() {
        const char* env = std::getenv("INSTA360_DUMP_RECORDS");
        return env && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

}  // namespace

void TrailerParser::ParseRecordByType(uint8_t id,
                                      uint8_t format,
                                      const uint8_t* data,
                                      size_t len,
                                      std::vector<ImuSample>& out_samples) const {
    if (!data || len == 0) {
        return;
    }

    if (DumpRecordsEnabled(verbose_)) {
        std::printf("Record id=%u (%s) format=%u size=%zu\n",
                    static_cast<unsigned>(id),
                    RecordTypeName(id),
                    static_cast<unsigned>(format),
                    len);
        std::fflush(stdout);
    }

    switch (id) {
    case extra_info::RecordType::Offsets:
        (void)ParseOffsetsRecord(data, len);
        break;
    case extra_info::RecordType::Metadata:
        ParseMetadataRecord(data, len);
        break;
    case extra_info::RecordType::Thumbnail:
        ParseThumbnailRecord(data, len, false);
        break;
    case extra_info::RecordType::Gyro:
        ParseImuRecord(0x0300, data, len, out_samples);
        break;
    case extra_info::RecordType::Exposure:
    case extra_info::RecordType::ExposureSecondary:
        ParseExposureRecord(data, len);
        break;
    case extra_info::RecordType::ThumbnailExt:
        ParseThumbnailRecord(data, len, true);
        break;
    case extra_info::RecordType::TimelapseTimestamp:
        ParseTimelapseTimestampRecord(data, len);
        break;
    case extra_info::RecordType::Gps:
        ParseGpsRecord(data, len);
        break;
    case extra_info::RecordType::AAAData:
        ParseAaaDataRecord(data, len);
        break;
    case extra_info::RecordType::Anchors:
        ParseAnchorsRecord(data, len);
        break;
    case extra_info::RecordType::TimeMap:
        ParseTimeMapRecord(data, len);
        break;
    default:
        break;
    }
}

bool TrailerParser::ParseMetadataRecord(const uint8_t* data, size_t len) const {
    size_t pos = 0;
    bool is_raw_gyro = imu_calib_.is_raw_gyro;
    double acc_range = imu_calib_.acc_range;
    double gyro_range = imu_calib_.gyro_range;
    std::string camera_type = metadata_state_.model;
    bool have_first_frame_timestamp = metadata_state_.have_first_frame_timestamp;
    double first_frame_timestamp = metadata_state_.first_frame_timestamp;
    bool have_gyro_timestamp = metadata_state_.have_gyro_timestamp;
    double gyro_timestamp_ms = metadata_state_.gyro_timestamp_ms;
    bool have_frame_readout_time = metadata_state_.have_frame_readout_time;
    double frame_readout_time = metadata_state_.frame_readout_time;
    bool has_offset_v3 = metadata_state_.has_offset_v3;

    while (pos < len) {
        uint64_t key = 0;
        if (!ReadVarint(data, len, pos, key)) {
            break;
        }
        const uint32_t field_number = static_cast<uint32_t>(key >> 3);
        const uint32_t wire_type = static_cast<uint32_t>(key & 0x7);

        if (field_number == 2 && wire_type == 2) {  // camera_type: string
            uint64_t l = 0;
            if (!ReadVarint(data, len, pos, l)) break;
            if (pos + l > len) break;
            camera_type.assign(reinterpret_cast<const char*>(data + pos), static_cast<size_t>(l));
            pos += static_cast<size_t>(l);
            continue;
        }
        if (field_number == 24 && wire_type == 0) {  // first_frame_timestamp
            uint64_t v = 0;
            if (!ReadVarint(data, len, pos, v)) break;
            first_frame_timestamp = static_cast<double>(static_cast<int64_t>(v));
            have_first_frame_timestamp = true;
            continue;
        }
        if (field_number == 25 && wire_type == 1) {  // rolling_shutter_time
            if (pos + 8 > len) break;
            frame_readout_time = ReadF64LE(data + pos);
            have_frame_readout_time = true;
            pos += 8;
            continue;
        }
        if (field_number == 28 && wire_type == 1) {  // gyro_timestamp
            if (pos + 8 > len) break;
            gyro_timestamp_ms = ReadF64LE(data + pos);
            pos += 8;
            continue;
        }
        if (field_number == 29 && wire_type == 0) {  // is_has_gyro_timestamp
            uint64_t v = 0;
            if (!ReadVarint(data, len, pos, v)) break;
            have_gyro_timestamp = (v != 0);
            continue;
        }
        if ((field_number == 54 || field_number == 53) && wire_type == 2) {  // offset_v3 / offset_v2
            uint64_t l = 0;
            if (!ReadVarint(data, len, pos, l)) break;
            if (pos + l > len) break;
            if (field_number == 54) {
                const char* s = reinterpret_cast<const char*>(data + pos);
                size_t n = static_cast<size_t>(l);
                size_t count = 0;
                bool in_token = false;
                for (size_t i = 0; i < n; ++i) {
                    const char c = s[i];
                    if (c == '_') {
                        if (in_token) {
                            ++count;
                            in_token = false;
                        }
                    } else {
                        in_token = true;
                    }
                }
                if (in_token) {
                    ++count;
                }
                has_offset_v3 = (count >= 20);
            }
            pos += static_cast<size_t>(l);
            continue;
        }

        if (field_number == 62 && wire_type == 0) {
            uint64_t v = 0;
            if (!ReadVarint(data, len, pos, v)) break;
            is_raw_gyro = (v != 0);
            continue;
        }
        if (field_number == 65 && wire_type == 2) {
            uint64_t l = 0;
            if (!ReadVarint(data, len, pos, l)) break;
            if (pos + l > len) break;
            const uint8_t* sub = data + pos;
            const size_t sub_len = static_cast<size_t>(l);
            double acc_r = acc_range;
            double gyro_r = gyro_range;
            if (ParseGyroConfigInfo(sub, sub_len, acc_r, gyro_r)) {
                acc_range = acc_r;
                gyro_range = gyro_r;
            }
            pos += sub_len;
            continue;
        }

        if (!SkipField(data, len, pos, wire_type)) {
            break;
        }
    }

    imu_calib_.is_raw_gyro = is_raw_gyro;
    if (acc_range > 0.0 && gyro_range > 0.0) {
        imu_calib_.acc_range = acc_range;
        imu_calib_.gyro_range = gyro_range;
        imu_calib_.have_ranges = true;
    }

    metadata_state_.model = camera_type;
    metadata_state_.have_first_frame_timestamp = have_first_frame_timestamp;
    metadata_state_.first_frame_timestamp = first_frame_timestamp;
    metadata_state_.have_gyro_timestamp = have_gyro_timestamp;
    metadata_state_.gyro_timestamp_ms = gyro_timestamp_ms;
    metadata_state_.have_frame_readout_time = have_frame_readout_time;
    metadata_state_.frame_readout_time = frame_readout_time;
    metadata_state_.has_offset_v3 = has_offset_v3;

    if (DumpRecordsEnabled(verbose_)) {
        std::printf(
            "MetadataState: model='%s' is_raw_gyro=%d acc_range=%.6f gyro_range=%.6f "
            "have_first_frame_timestamp=%d first_frame_timestamp=%.6f "
            "have_gyro_timestamp=%d gyro_timestamp_ms=%.6f "
            "have_frame_readout_time=%d frame_readout_time=%.9f has_offset_v3=%d\n",
            metadata_state_.model.c_str(),
            imu_calib_.is_raw_gyro ? 1 : 0,
            imu_calib_.acc_range,
            imu_calib_.gyro_range,
            metadata_state_.have_first_frame_timestamp ? 1 : 0,
            metadata_state_.first_frame_timestamp,
            metadata_state_.have_gyro_timestamp ? 1 : 0,
            metadata_state_.gyro_timestamp_ms,
            metadata_state_.have_frame_readout_time ? 1 : 0,
            metadata_state_.frame_readout_time,
            metadata_state_.has_offset_v3 ? 1 : 0);
        std::fflush(stdout);
    }

    return true;
}

bool TrailerParser::ParseGyroConfigInfo(const uint8_t* data, size_t len, double& acc_range, double& gyro_range) const {
    size_t pos = 0;
    while (pos < len) {
        uint64_t key = 0;
        if (!ReadVarint(data, len, pos, key)) {
            break;
        }
        const uint32_t field_number = static_cast<uint32_t>(key >> 3);
        const uint32_t wire_type = static_cast<uint32_t>(key & 0x7);

        if ((field_number == 1 || field_number == 2) && wire_type == 0) {
            uint64_t v = 0;
            if (!ReadVarint(data, len, pos, v)) break;
            if (field_number == 1 && v > 0) {
                acc_range = static_cast<double>(v);
            } else if (field_number == 2 && v > 0) {
                gyro_range = static_cast<double>(v);
            }
            continue;
        }

        if (!SkipField(data, len, pos, wire_type)) {
            break;
        }
    }
    return true;
}

std::vector<TrailerParser::OffsetEntry> TrailerParser::ParseOffsetsRecord(const uint8_t* data, size_t len) const {
    std::vector<OffsetEntry> entries;
    entries.reserve(len / 10);

    size_t pos = 0;
    while (pos + 10 <= len) {
        OffsetEntry e;
        e.id = data[pos + 0];
        e.format = data[pos + 1];
        e.size = ReadU32LE(data + pos + 2);
        e.rel_offset = ReadU32LE(data + pos + 6);

        if (e.id > 0 && e.size > 0) {
            entries.push_back(e);
        }
        pos += 10;
    }

    return entries;
}

void TrailerParser::ParseTimeMapRecord(const uint8_t* data, size_t len) const {
    if (len < 16) {
        return;
    }

    size_t pos = 0;
    (void)ReadU32LE(data + pos);
    pos += 4;
    const uint32_t num_trims = ReadU32LE(data + pos);
    pos += 4;
    (void)ReadU32LE(data + pos);
    pos += 4;
    (void)ReadU32LE(data + pos);
    pos += 4;

    for (uint32_t i = 0; i < num_trims; ++i) {
        if (pos + 32 > len) return;
        (void)ReadF64LE(data + pos + 0);
        (void)ReadF64LE(data + pos + 8);
        (void)ReadF64LE(data + pos + 16);
        (void)ReadF64LE(data + pos + 24);
        pos += 32;
    }

    if (pos + 8 > len) return;
    (void)ReadF64LE(data + pos);
    pos += 8;

    while (pos + 16 <= len) {
        (void)ReadF64LE(data + pos + 0);
        (void)ReadF64LE(data + pos + 8);
        pos += 16;
    }
}

void TrailerParser::ParseThumbnailRecord(const uint8_t* data, size_t len, bool is_ext) const {
    (void)data;
    (void)len;
    (void)is_ext;
}

void TrailerParser::ParseImuRecord(uint16_t /*id*/, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const {
    const bool is_raw_gyro = imu_calib_.is_raw_gyro;
    const size_t item_size = is_raw_gyro ? (8 + 6 * 2) : (8 + 6 * 8);
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

    if (item_size == 0 || len < item_size) {
        return;
    }

    const double acc_range_g = (imu_calib_.have_ranges && imu_calib_.acc_range > 0.0)
                                   ? imu_calib_.acc_range
                                   : 16.0;
    printf("Using accelerometer range: %.6f g\n", acc_range_g);
    fflush(stdout);
    const double gyro_range_dps = (imu_calib_.have_ranges && imu_calib_.gyro_range > 0.0)
                                      ? imu_calib_.gyro_range
                                      : 2000.0;
    const double acc_scale_g = acc_range_g / 32768.0;
    const double gyro_scale_rad_s = (gyro_range_dps / 32768.0) * kDegToRad;

    for (size_t offset = 0; offset + item_size <= len; offset += item_size) {
        const uint8_t* rec = data + offset;
        const uint64_t timestamp_raw = ReadU64LE(rec);

        ImuSample sample;
        sample.raw_time = static_cast<double>(timestamp_raw);

        if (!is_raw_gyro) {
            sample.ax = ReadF64LE(rec + 8);
            sample.ay = ReadF64LE(rec + 16);
            sample.az = ReadF64LE(rec + 24);
            sample.gx = ReadF64LE(rec + 32);
            sample.gy = ReadF64LE(rec + 40);
            sample.gz = ReadF64LE(rec + 48);
        } else {
            const double ax_raw = static_cast<double>(static_cast<int>(ReadU16LE(rec + 8)) - 32768);
            const double ay_raw = static_cast<double>(static_cast<int>(ReadU16LE(rec + 10)) - 32768);
            const double az_raw = static_cast<double>(static_cast<int>(ReadU16LE(rec + 12)) - 32768);
            const double gx_raw = static_cast<double>(static_cast<int>(ReadU16LE(rec + 14)) - 32768);
            const double gy_raw = static_cast<double>(static_cast<int>(ReadU16LE(rec + 16)) - 32768);
            const double gz_raw = static_cast<double>(static_cast<int>(ReadU16LE(rec + 18)) - 32768);

            // Keep accelerometer in g.
            sample.ax = ax_raw * acc_scale_g;
            sample.ay = ay_raw * acc_scale_g;
            sample.az = az_raw * acc_scale_g;

            // Always keep gyroscope in rad/s.
            sample.gx = gx_raw * gyro_scale_rad_s;
            sample.gy = gy_raw * gyro_scale_rad_s;
            sample.gz = gz_raw * gyro_scale_rad_s;
        }

        out_samples.push_back(sample);
    }
}

void TrailerParser::ParseExposureRecord(const uint8_t* data, size_t len) const {
    constexpr size_t kItemSize = 16;
    if (len < kItemSize) {
        return;
    }

    for (size_t pos = 0; pos + kItemSize <= len; pos += kItemSize) {
        (void)ReadU64LE(data + pos + 0);
        (void)ReadF64LE(data + pos + 8);
    }
}

void TrailerParser::ParseTimelapseTimestampRecord(const uint8_t* data, size_t len) const {
    constexpr size_t kTimestampSize = 8;
    if (len < kTimestampSize) {
        return;
    }

    for (size_t offset = 0; offset + kTimestampSize <= len; offset += kTimestampSize) {
        const uint64_t timestamp = ReadU64LE(data + offset) / 1000.0;
        (void)timestamp;
    }
}

void TrailerParser::ParseGpsRecord(const uint8_t* data, size_t len) const {
    constexpr size_t kItemSize = 53;
    if (len < kItemSize) {
        return;
    }

    for (size_t pos = 0; pos + kItemSize <= len; pos += kItemSize) {
        const double unix_ts = static_cast<double>(ReadU64LE(data + pos + 0)) +
                               static_cast<double>(ReadU16LE(data + pos + 8)) / 1000.0;
        const char fix = static_cast<char>(data[pos + 10]);
        double lat = ReadF64LE(data + pos + 11);
        const char lat_dir = static_cast<char>(data[pos + 19]);
        double lon = ReadF64LE(data + pos + 20);
        const char lon_dir = static_cast<char>(data[pos + 28]);
        const double speed_kmh = ReadF64LE(data + pos + 29) * 3.6;
        const double track = ReadF64LE(data + pos + 37);
        const double altitude = ReadF64LE(data + pos + 45);

        if (lat_dir == 'S') lat = -std::fabs(lat);
        if (lon_dir == 'W') lon = -std::fabs(lon);

        (void)unix_ts;
        (void)fix;
        (void)lat;
        (void)lon;
        (void)speed_kmh;
        (void)track;
        (void)altitude;
    }
}

void TrailerParser::ParseAaaDataRecord(const uint8_t* data, size_t len) const {
    constexpr size_t kItemSize = 48;
    if (len < kItemSize) {
        return;
    }

    for (size_t pos = 0; pos + kItemSize <= len; pos += kItemSize) {
        const uint32_t timestamp = ReadU32LE(data + pos + 0);
        const float ev_target = ReadF32LE(data + pos + 4);
        const float exp_time = ReadF32LE(data + pos + 8);
        const uint32_t data_stat = ReadU32LE(data + pos + 12);
        const uint32_t luma_struct = ReadU32LE(data + pos + 16);

        const uint32_t luma_wg_grid = luma_struct & 0x7F;
        const uint32_t luma_wg_y = (luma_struct & 0x3F80) >> 7;
        const uint32_t sum_wg_y = (0x7C000 & luma_struct) >> 14;
        const uint32_t iso_value = (100 * ((luma_struct & 0xFFF80000) >> 19)) >> 6;

        (void)timestamp;
        (void)ev_target;
        (void)exp_time;
        (void)data_stat;
        (void)luma_wg_grid;
        (void)luma_wg_y;
        (void)sum_wg_y;
        (void)iso_value;
    }
}

void TrailerParser::ParseAnchorsRecord(const uint8_t* data, size_t len) const {
    size_t pos = 0;
    while (pos + 5 <= len) {
        const uint8_t type = data[pos];
        const uint32_t count = ReadU32LE(data + pos + 1);
        pos += 5;

        const size_t per_item = (type != 2 && type != 18) ? 8 : 16;
        const size_t need = static_cast<size_t>(count) * per_item;
        if (pos + need > len) {
            return;
        }
        pos += need;
    }
}

}  // namespace insta360_insv
