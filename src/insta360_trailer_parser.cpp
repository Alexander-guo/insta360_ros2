// Authors: Jinyuan Guo
// In Reference to ExifTool ProcessInsta360(): https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/QuickTimeStream.pl

#include "insv/insta360_trailer_parser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace insta360_insv {
namespace {
constexpr size_t kFooterSize = 6;            // 2-byte id + 4-byte length
constexpr size_t kTrailerProbeSize = 78;     // minimum footer span used by ExifTool logic
constexpr const char* kMagicHex = "8db42d694ccc418790edff439fe026bf";
constexpr size_t kMagicAsciiLen = 32;   // length of kMagicHex string
constexpr size_t kMagicBinaryLen = kMagicAsciiLen / 2;
constexpr size_t kMaxTailSearchBytes = 32 * 1024 * 1024;
constexpr size_t kMaxTrailerBackscan = 300 * 1024 * 1024;  // scan up to ~300MB before magic
constexpr size_t kMagicSearchChunk = 4 * 1024 * 1024;
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

// Heuristic stride guess based on ExifTool logic: prefer divisible length, otherwise
// peek at first 20 bytes to decide between 20-byte (short) and 56-byte (double) records.
size_t GuessImuStride(size_t len, const uint8_t* data) {
    if (len == 0 || data == nullptr) {
        return 0;
    }

    const bool div20 = (len % 20) == 0;
    const bool div56 = (len % 56) == 0;

    if (!div20 && div56) {
        return 56;
    }
    if (!div56 && div20) {
        return 20;
    }

    if (len >= 20) {
        const bool last_three_zero = (data[16] == 0 && data[17] == 0 && data[18] == 0);
        return last_three_zero ? 56 : 20;
    }

    return 0;
}

void DebugScanAllIMURegions(const uint8_t* data, size_t len) {
    constexpr size_t stride = 20;
    const size_t min_region_records = []() {
        const char* env = std::getenv("INSTA360_DEBUG_IMU_MIN_REGION");
        if (!env || !*env) {
            return static_cast<size_t>(1000);
        }
        char* end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end == env || v <= 0) {
            return static_cast<size_t>(1000);
        }
        return static_cast<size_t>(v);
    }();
    auto read_u64_le = [](const uint8_t* p) -> uint64_t {
        return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
               (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
               (static_cast<uint64_t>(p[4]) << 32) | (static_cast<uint64_t>(p[5]) << 40) |
               (static_cast<uint64_t>(p[6]) << 48) | (static_cast<uint64_t>(p[7]) << 56);
    };
    size_t region_start = SIZE_MAX;
    size_t valid_count = 0;
    uint64_t prev_ts = 0;
    size_t max_region_start = SIZE_MAX;
    size_t max_region_len = 0;
    bool printed_any = false;

    for (size_t offset = 0; offset + stride <= len; ++offset) {
        const uint8_t* rec = data + offset;
        const uint64_t ts = read_u64_le(rec);

        if (ts == 0 || ts == std::numeric_limits<uint64_t>::max()) {
            continue;
        }

        const bool monotonic = (prev_ts == 0 || ts > prev_ts);

        if (monotonic) {
            if (region_start == SIZE_MAX) {
                region_start = offset;
            }
            ++valid_count;
            prev_ts = ts;
        } else {
            if (valid_count >= min_region_records) {
                printf("IMU region at offset %zu, records=%zu\n", region_start, valid_count);
                printed_any = true;
            }
            if (valid_count > max_region_len) {
                max_region_len = valid_count;
                max_region_start = region_start;
            }
            region_start = SIZE_MAX;
            valid_count = 0;
            prev_ts = 0;
        }
    }

    if (valid_count >= min_region_records && region_start != SIZE_MAX) {
        printf("IMU region at offset %zu, records=%zu\n", region_start, valid_count);
        printed_any = true;
    }
    if (valid_count > max_region_len) {
        max_region_len = valid_count;
        max_region_start = region_start;
    }

    if (!printed_any) {
        printf("No IMU region reached min threshold (%zu records). Max region: offset=%zu len=%zu\n",
               min_region_records,
               max_region_start == SIZE_MAX ? 0 : max_region_start,
               max_region_len);
    }
}

std::optional<size_t> FindMagicOffset(const std::string& path, std::streamoff file_size, bool& ascii_magic_found) {
    ascii_magic_found = false;
    if (file_size <= 0) {
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    const size_t overlap = kMagicAsciiLen;
    std::vector<uint8_t> buffer(kMagicSearchChunk + overlap);

    size_t pos = static_cast<size_t>(file_size);
    while (pos > 0) {
        const size_t to_read = std::min<size_t>(pos, kMagicSearchChunk + overlap);
        const size_t start = pos - to_read;
        file.seekg(static_cast<std::streamoff>(start));
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(to_read));
        const size_t got = static_cast<size_t>(file.gcount());
        if (got == 0) {
            break;
        }

        if (got >= kMagicAsciiLen) {
            const size_t ascii_limit = got - kMagicAsciiLen + 1;
            for (size_t i = 0; i < ascii_limit; ++i) {
                if (std::memcmp(buffer.data() + i, kMagicHex, kMagicAsciiLen) == 0) {
                    ascii_magic_found = true;
                    return start + i;
                }
            }
        }

        if (got >= kMagicBinaryLen) {
            const size_t bin_limit = got - kMagicBinaryLen + 1;
            for (size_t i = 0; i < bin_limit; ++i) {
                if (std::memcmp(buffer.data() + i, kMagicBinary.data(), kMagicBinaryLen) == 0) {
                    ascii_magic_found = false;
                    return start + i;
                }
            }
        }

        if (start == 0) {
            break;
        }
        pos = start;
    }

    return std::nullopt;
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
    
    // First attempt: scan backwards for magic and parse trailer from there. 
    // This is more robust, as it exhaustively scans for valid trailers and IMU records, and can find trailers located far from the end of the file. 
    bool ascii_magic_found = false;
    std::string trailer_error;
    if (auto magic_offset = FindMagicOffset(path, file_size, ascii_magic_found)) {
        const size_t magic_pos = *magic_offset;
        // const size_t scan_start = (magic_pos > kMaxTrailerBackscan) ? magic_pos - kMaxTrailerBackscan : 0;
        const size_t scan_start = 0;    // Exhaustive scan from the start of the file
        const size_t window_size = magic_pos - scan_start;
        printf("Found Insta360 magic at offset %zu (%s). Scanning %zu MB window from offset %zu\n",
               magic_pos,
               ascii_magic_found ? "ascii" : "binary",
               window_size / (1024 * 1024),
               scan_start);
        fflush(stdout);

        if (window_size > 0) {
            std::string scan_error;
            if (ScanTrailerWindow(path, scan_start, window_size, out_samples, &scan_error) && !out_samples.empty()) {
                return true;
            }
            trailer_error = scan_error.empty() ? "Trailer window scan produced no samples" : scan_error;
        } else {
            trailer_error = "Trailer magic at offset 0 produced zero-sized window";
        }
    } else {
        trailer_error = "Insta360 magic not found during backward scan";
    }

    // Fallback: old probe logic on the tiny tail plus limited tail scan.
    // This preserves the previous behavior when magic search fails or yields no samples.
    std::string footer_error;
    {
        file.seekg(file_size - static_cast<std::streamoff>(kTrailerProbeSize));
        std::vector<uint8_t> probe(kTrailerProbeSize);
        file.read(reinterpret_cast<char*>(probe.data()), probe.size());
        if (file.gcount() == static_cast<std::streamsize>(probe.size())) {
            bool has_magic = false;
            bool ascii_magic_tail = false;
            bool binary_magic_tail = false;
            if (probe.size() >= kMagicAsciiLen) {
                const char* ascii_ptr = reinterpret_cast<const char*>(probe.data() + probe.size() - kMagicAsciiLen);
                const std::string ascii_magic_str(ascii_ptr, kMagicAsciiLen);
                if (ascii_magic_str == kMagicHex) {
                    has_magic = true;
                    ascii_magic_tail = true;
                }
            }
            if (!has_magic && probe.size() >= kMagicBinaryLen) {
                const std::string binary_magic_str = ToHex(probe.data() + probe.size() - kMagicBinaryLen, kMagicBinaryLen);
                if (binary_magic_str == kMagicHex) {
                    has_magic = true;
                    binary_magic_tail = true;
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
                if (binary_magic_tail && probe.size() >= 0x38 + sizeof(uint32_t)) {
                    trailer_len = read_len_valid(probe.data() + 0x38);
                }
                if (!trailer_len && ascii_magic_tail && probe.size() >= 2 + sizeof(uint32_t)) {
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
    printf("Trailer fallback probe error: %s\n", footer_error.c_str());
    fflush(stdout);

    out_samples.clear();
    const size_t tail_window = static_cast<size_t>(std::min<std::streamoff>(file_size, static_cast<std::streamoff>(kMaxTailSearchBytes)));
    const size_t tail_start = static_cast<size_t>(file_size) - tail_window;
    std::string scan_error;
    if (ScanTrailerWindow(path, tail_start, tail_window, out_samples, &scan_error)) {
        return true;
    }

    if (error_out) {
        if (!scan_error.empty()) {
            *error_out = scan_error;
        } else if (!footer_error.empty()) {
            *error_out = footer_error;
        } else if (!trailer_error.empty()) {
            *error_out = trailer_error;
        } else {
            *error_out = "Failed to parse trailer";
        }
    }
    return false;
}

// Trailer records are stored back-to-front: [data][footer=id+len][data][footer]... ending with magic.
// Walk backwards by reading each 6-byte footer (id, len), then dispatch on id to parse its payload.
bool TrailerParser::ParseTrailer(const std::vector<uint8_t>& trailer, std::vector<ImuSample>& out_samples, std::string* error_out) const {
    size_t cursor = trailer.size();
    const size_t trailer_len = trailer.size();
    const uint8_t* trailer_data = trailer.data();

    const uint8_t* dir_table = nullptr;
    size_t dir_table_len = 0;

    while (cursor >= kFooterSize) {
        const size_t footer_pos = cursor - kFooterSize;
        const uint8_t* footer = trailer_data + footer_pos;
        const uint16_t id = ReadU16LE(footer);
        const uint32_t len = ReadU32LE(footer + 2);

        if (len == 0) {
            cursor = footer_pos;
            continue;
        }

        if (footer_pos < len) {
            if (error_out) *error_out = "Trailer parse underflow";
            printf("Trailer parse underflow at cursor=%zu (id=0x%x len=%u footer_pos=%zu)\n", cursor, id, len, footer_pos);
            fflush(stdout);
            return false;
        }

        const size_t data_pos = footer_pos - len;
        const uint8_t* data = trailer_data + data_pos;

        if (id == 0x0000) {
            if (len == 0) {
                break;
            }
            if (dir_table == nullptr) {
                dir_table = data;
                dir_table_len = len;
                printf("Directory table found (len=%u)\n", len);
                fflush(stdout);
            }
        } else {
            if (len < 64 || len > 50 * 1024 * 1024) {
                cursor--;
                continue;
            }

            if (id == 0x300) {
            printf("Parse Trailer Record id=0x%x len=%u\n", id, len);
            fflush(stdout);
            ParseImuRecord(id, data, len, out_samples);
            } else if (id == 0x600) {
            ParseVideoTimestampRecord(id, data, len, out_samples);
            }
        }
        // Other record types can be added here when needed (0x400 exposure, 0x600 video ts, 0x700 GPS, etc.)

        cursor = data_pos;
    }

    if (dir_table && dir_table_len >= 10) {
        std::unordered_set<size_t> parsed_offsets;
        size_t pos = 0;
        while (pos + 10 <= dir_table_len) {
            const uint16_t entry_id = ReadU16LE(dir_table + pos);
            const uint32_t entry_size = ReadU32LE(dir_table + pos + 2);
            const uint32_t entry_off = ReadU32LE(dir_table + pos + 6);
            pos += 10;

            if (entry_id == 0 || entry_size == 0) {
                continue;
            }

            if (entry_size < 64 || entry_size > 50 * 1024 * 1024) {
                continue;
            }

            if (entry_off + entry_size > trailer_len || entry_off + kFooterSize > trailer_len) {
                continue;
            }

            const size_t footer_pos = entry_off;
            const size_t data_pos = footer_pos >= entry_size ? footer_pos - entry_size : trailer_len;
            if (data_pos == trailer_len) {
                continue;
            }

            if (parsed_offsets.count(footer_pos)) {
                continue;
            }

            const uint16_t footer_id = ReadU16LE(trailer_data + footer_pos);
            const uint32_t footer_len = ReadU32LE(trailer_data + footer_pos + 2);
            if (footer_id != entry_id || footer_len != entry_size) {
                continue;
            }

            parsed_offsets.insert(footer_pos);

            const uint8_t* entry_data = trailer_data + data_pos;
            if (entry_id == 0x300) {
                printf("DirTable: Record id=0x%x len=%u at footer_pos=%zu\n", entry_id, entry_size, footer_pos);
                fflush(stdout);
                ParseImuRecord(entry_id, entry_data, entry_size, out_samples);
            } else if (entry_id == 0x600) {
                ParseVideoTimestampRecord(entry_id, entry_data, entry_size, out_samples);
            }
        }
    }

    return true;
}

// Scans a specified slice of the file (start_offset, window_size) backwards for trailer footers
// and parses supported records (0x300 IMU, 0x600 video timestamps). Length sanity is applied
// to reject implausible records and IMU blocks must align to 20- or 56-byte strides.
bool TrailerParser::ScanTrailerWindow(const std::string& path, size_t start_offset, size_t window_size, std::vector<ImuSample>& out_samples, std::string* error_out) const {
    if (window_size == 0) {
        if (error_out) *error_out = "Trailer window size is zero";
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error_out) *error_out = "Failed to reopen file for trailer scan";
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    if (start_offset >= static_cast<size_t>(file_size)) {
        if (error_out) *error_out = "Trailer window start beyond file size";
        return false;
    }

    const size_t clamped_window = std::min(window_size, static_cast<size_t>(file_size - static_cast<std::streamoff>(start_offset)));
    file.seekg(static_cast<std::streamoff>(start_offset));
    std::vector<uint8_t> buffer(clamped_window);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (file.gcount() != static_cast<std::streamsize>(buffer.size())) {
        if (error_out) *error_out = "Failed to read trailer window";
        return false;
    }

    printf("Scanning trailer window: offset=%zu size=%zu MB\n", start_offset, clamped_window / (1024 * 1024));
    fflush(stdout);

    bool found_supported = false;
    size_t cursor = buffer.size();
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

        if (len < 64 || len > 50 * 1024 * 1024) {
            cursor--;
            continue;
        }

        const size_t data_pos = footer_pos - len;
        const uint8_t* data = buffer.data() + data_pos;
        if (id == 0x300) {
            if (len % 20 != 0 && len % 56 != 0) {
                cursor--;
                continue;
            }
            // printf("Scan Trailer Window: Accepted Record id=0x%x len=%u\n", id, len);
            // fflush(stdout);
            ParseImuRecord(id, data, len, out_samples);
            found_supported = true;
        } else if (id == 0x600) {
            ParseVideoTimestampRecord(id, data, len, out_samples);
            found_supported = true;
        }
        cursor = data_pos;
    }

    if (!found_supported) {
        if (error_out) *error_out = "No supported trailer records found in window";
        return false;
    }
    return true;
}

void TrailerParser::ParseImuRecord(uint16_t /*id*/, const uint8_t* data, size_t len, std::vector<ImuSample>& out_samples) const {
    constexpr size_t kShortStride = 20;
    constexpr size_t kDoubleStride = 56;
    constexpr bool kAllowHighPrecisionStride = false;  // X5 trailers are short stride only
    // constexpr bool kAllowHighPrecisionStride = true; 

    if (len < kShortStride) {
        return;
    }

    static const bool kDebugScanRegions = []() {
        const char* env = std::getenv("INSTA360_DEBUG_IMU_REGIONS");
        return env && std::strcmp(env, "0") != 0;
    }();
    if (kDebugScanRegions) {
        static bool announced = false;
        if (!announced) {
                        printf("INSTA360_DEBUG_IMU_REGIONS enabled; scanning IMU records (min region env: INSTA360_DEBUG_IMU_MIN_REGION)\n");
            fflush(stdout);
            announced = true;
        }
        DebugScanAllIMURegions(data, len);
        fflush(stdout);
    }

    const auto decode_record = [&](const uint8_t* rec, size_t stride, ImuSample& sample) -> bool {
        sample.high_precision = (stride == kDoubleStride);
        if (sample.high_precision) {
            sample.ax = ReadF64LE(rec + 8);
            sample.ay = ReadF64LE(rec + 16);
            sample.az = ReadF64LE(rec + 24);
            sample.gx = ReadF64LE(rec + 32);
            sample.gy = ReadF64LE(rec + 40);
            sample.gz = ReadF64LE(rec + 48);
        } else {
            uint16_t raw[6];
            std::memcpy(raw, rec + 8, sizeof(raw));
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
        }

        const double abs_ax = std::fabs(sample.ax);
        const double abs_ay = std::fabs(sample.ay);
        const double abs_az = std::fabs(sample.az);
        const double abs_gx = std::fabs(sample.gx);
        const double abs_gy = std::fabs(sample.gy);
        const double abs_gz = std::fabs(sample.gz);
        const bool accel_ok = (abs_ax < 16.0 && abs_ay < 16.0 && abs_az < 16.0);
        const bool gyro_ok = (abs_gx < 2000.0 && abs_gy < 2000.0 && abs_gz < 2000.0);
        return accel_ok && gyro_ok;
    };

    struct ProbeResult {
        size_t stride{0};
        size_t offset{0};
        size_t inspected_records{0};
        size_t accepted_records{0};
        size_t monotonic_errors{0};
        size_t jump_outliers{0};
        size_t invalid_ts{0};
        size_t invalid_values{0};
        double first_ts{std::numeric_limits<double>::quiet_NaN()};
        double last_ts{std::numeric_limits<double>::quiet_NaN()};
        double median_dt{std::numeric_limits<double>::quiet_NaN()};
        double coverage{0.0};
        double score{0.0};
        bool dt_reasonable{false};
        bool viable{false};
    };

    auto compute_median = [](std::vector<double>& values) -> double {
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        double median = values[mid];
        if ((values.size() % 2) == 0) {
            std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
            median = 0.5 * (median + values[mid - 1]);
        }
        return median;
    };

    auto probe_stride = [&](size_t stride, size_t start_offset) {
        ProbeResult result;
        result.stride = stride;
        result.offset = start_offset;
        if (stride == 0 || start_offset >= len || len - start_offset < stride) {
            return result;
        }

        const size_t available = len - start_offset;
        const size_t max_records = available / stride;
        if (max_records == 0) {
            return result;
        }

        const size_t max_probe = std::min<size_t>(max_records, static_cast<size_t>(512));
        const size_t max_inspected = std::min<size_t>(max_records, max_probe * 4);
        if (max_probe == 0) {
            return result;
        }

        std::vector<double> dt_samples;
        dt_samples.reserve(max_probe);

        double prev_ts = std::numeric_limits<double>::quiet_NaN();
        for (size_t idx = 0; idx < max_inspected; ++idx) {
            if (result.accepted_records >= max_probe) {
                break;
            }
            const size_t offset = start_offset + idx * stride;
            if (offset + stride > len) {
                break;
            }

            ++result.inspected_records;
            const uint8_t* rec = data + offset;
            const uint64_t raw_time = ReadU64LE(rec);
            if (raw_time == 0 || raw_time == std::numeric_limits<uint64_t>::max()) {
                ++result.invalid_ts;
                continue;
            }

            const double ts = static_cast<double>(raw_time);
            if (!std::isfinite(ts)) {
                ++result.invalid_ts;
                continue;
            }

            ImuSample tmp;
            if (!decode_record(rec, stride, tmp)) {
                ++result.invalid_values;
                continue;
            }

            if (!std::isfinite(result.first_ts)) {
                result.first_ts = ts;
            } else {
                if (ts <= prev_ts) {
                    ++result.monotonic_errors;
                    continue;
                }
                const double dt = ts - prev_ts;
                if (dt <= 0.0) {
                    ++result.monotonic_errors;
                    continue;
                }
                if (dt > 5.0e6) {
                    ++result.jump_outliers;
                    continue;
                }
                if (dt_samples.size() < max_probe) {
                    dt_samples.push_back(dt);
                }
            }

            prev_ts = ts;
            result.last_ts = ts;
            ++result.accepted_records;
        }

        if (result.accepted_records == 0 || result.inspected_records == 0) {
            return result;
        }

        if (!dt_samples.empty()) {
            result.median_dt = compute_median(dt_samples);
            if (std::isfinite(result.median_dt)) {
                result.dt_reasonable = (result.median_dt >= 5.0 && result.median_dt <= 5.0e5);
            }
        }

        result.coverage = static_cast<double>(result.accepted_records) /
                          static_cast<double>(result.inspected_records);

        const double penalties = static_cast<double>(result.monotonic_errors) * 6.0 +
                                 static_cast<double>(result.jump_outliers) * 8.0 +
                                 static_cast<double>(result.invalid_ts + result.invalid_values) * 0.5;
        double base = static_cast<double>(result.accepted_records);
        if (result.accepted_records >= 64) {
            base += std::log1p(static_cast<double>(result.accepted_records));
        }
        double quality = std::max(0.05, result.coverage);
        if (!result.dt_reasonable) {
            quality *= 0.1;
        }
        if (!std::isfinite(result.last_ts) || !std::isfinite(result.first_ts) ||
            result.last_ts <= result.first_ts) {
            quality *= 0.1;
        }

        result.score = std::max(0.0, base * quality - penalties);
        result.viable = (result.accepted_records >= 12) && (result.coverage >= 0.2) &&
                        (result.score > 0.0);
        return result;
    };

    std::vector<ProbeResult> candidates;
    candidates.reserve(32);

    auto try_stride = [&](size_t stride) {
        if (stride == 0 || stride > len) {
            return;
        }
        const size_t max_offset = std::min<size_t>(4096, len - stride);
        // const size_t max_offset = 1;
        for (size_t start = 0; start < max_offset; ++start) {
            auto result = probe_stride(stride, start);
            if (result.inspected_records == 0) {
                continue;
            }
            candidates.push_back(std::move(result));
        }
    };

    auto try_stride_ordered = [&](size_t primary, size_t secondary) {
        if (primary) {
            try_stride(primary);
        }
        if (secondary && secondary != primary) {
            try_stride(secondary);
        }
    };

    // Prefer the ExifTool-style stride hint; still fall back to probing both if needed.
    const size_t hinted_stride = GuessImuStride(len, data);

    if (hinted_stride == kDoubleStride) {
        try_stride_ordered(kDoubleStride, kShortStride);
    } else if (hinted_stride == kShortStride) {
        try_stride_ordered(kShortStride, kAllowHighPrecisionStride ? kDoubleStride : 0);
    } else {
        try_stride(kShortStride);
        if (kAllowHighPrecisionStride) {
            try_stride(kDoubleStride);
        }
    }

    // try_stride(kShortStride);
    // if (kAllowHighPrecisionStride) {
    //     try_stride(kDoubleStride);
    // }

    const ProbeResult* best = nullptr;
    for (const auto& cand : candidates) {
        if (!cand.viable) {
            continue;
        }
        if (cand.stride == kDoubleStride && !kAllowHighPrecisionStride) {
            continue;
        }
        if (best == nullptr) {
            best = &cand;
            continue;
        }
        double score = cand.score;
        double best_score = best->score;
        if (std::fabs(score - best_score) < 1e-6) {
            if (cand.stride == kShortStride && best->stride != kShortStride) {
                best = &cand;
            } else if (cand.coverage > best->coverage + 0.05) {
                best = &cand;
            }
            continue;
        }
        if (score > best_score) {
            best = &cand;
        }
    }

    if (best == nullptr) {
        // printf("Stride probe: len=%zu hint=%zu candidates=%zu -> no viable stride\n",
        //        len, hinted_stride, candidates.size());
        // fflush(stdout);
        return;
    }

    const size_t stride = best->stride;
    const size_t start_offset = best->offset;
    printf("Selected stride %zu with offset %zu: accepted=%zu coverage=%.2f%% median_dt=%.3f score=%.1f\n",
           stride, start_offset, best->accepted_records, best->coverage * 100.0, best->median_dt, best->score);
    fflush(stdout);

    for (size_t offset = start_offset; offset + stride <= len; offset += stride) {
        const uint8_t* rec = data + offset;
        const uint64_t raw_time = ReadU64LE(rec);
        if (raw_time == 0 || raw_time == std::numeric_limits<uint64_t>::max()) {
            continue;
        }

        ImuSample sample;
        sample.raw_time = static_cast<double>(raw_time);
        sample.time_sec = std::numeric_limits<double>::quiet_NaN();
        sample.is_video_ts = false;

        if (!decode_record(rec, stride, sample)) {
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
