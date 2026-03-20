#include "insv/insta360_extra_info.hpp"

#include <cstring>
#include <sstream>

namespace insta360_insv::extra_info {
namespace {
uint64_t ReadU64LE(const uint8_t* p) {
    return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
           (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
           (static_cast<uint64_t>(p[4]) << 32) | (static_cast<uint64_t>(p[5]) << 40) |
           (static_cast<uint64_t>(p[6]) << 48) | (static_cast<uint64_t>(p[7]) << 56);
}

double ReadF64LE(const uint8_t* p) {
    uint64_t raw = ReadU64LE(p);
    double out;
    std::memcpy(&out, &raw, sizeof(double));
    return out;
}
}  // namespace

bool ParseGyroCalib(const uint8_t* data, size_t len, ParsedGyroCalib& out) {
    if (!data || len < 56) {
        return false;
    }
    for (size_t i = 0; i < 6; ++i) {
        out.numbers[i] = ReadF64LE(data + i * 8);
    }
    out.unix_timestamp = ReadU64LE(data + 48);
    return true;
}

bool ParseGyro(const uint8_t* data, size_t len, ParsedGyro& out) {
    if (!data || len < 56) {
        return false;
    }
    out.timestamp = ReadU64LE(data);
    for (size_t i = 0; i < 6; ++i) {
        out.numbers[i] = ReadF64LE(data + 8 + i * 8);
    }
    return true;
}

bool ParseOffset(const std::string& text, std::vector<double>& out) {
    out.clear();
    if (text.empty()) {
        return false;
    }

    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, '_')) {
        try {
            out.push_back(std::stod(part));
        } catch (...) {
            out.clear();
            return false;
        }
    }
    return !out.empty();
}

}  // namespace insta360_insv::extra_info
