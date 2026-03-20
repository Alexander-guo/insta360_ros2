#include "insv/insta360_trailer_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: trailer_parser_dump <file_path>" << std::endl;
        return 2;
    }

    const std::string path = argv[1];
    setenv("INSTA360_DUMP_RECORDS", "1", 1);

    insta360_insv::TrailerParser parser;
    std::vector<insta360_insv::ImuSample> samples;
    std::string err;

    if (!parser.ParseFile(path, samples, &err)) {
        std::cerr << "Parse failed: " << err << std::endl;
        return 1;
    }

    size_t imu_count = samples.size();

    std::cout << "\nParse summary:\n";
    std::cout << "  total samples: " << samples.size() << "\n";
    std::cout << "  imu samples:   " << imu_count << "\n";

    
    const size_t preview = std::min<size_t>(samples.size(), 10);
    // const size_t preview = samples.size();
    std::cout << "\nFirst " << preview << " samples:\n";
    for (size_t i = 0; i < preview; ++i) {
        const auto& s = samples[i];
        std::cout << "  [" << i << "] "
                  << "IMU"
                  << " raw_time=" << s.raw_time << " [camera ticks/raw]"
                  << " time_sec=" << s.time_sec << " [s]"
                  << " delta_time_sec=" << (i > 0 ? (s.time_sec - samples[i - 1].time_sec) : 0.0) << " [s]"
                  << " ax=" << s.ax << " [g]"
                  << " ay=" << s.ay << " [g]"
                  << " az=" << s.az << " [g]"
                  << " gx=" << s.gx << " [rad/s]"
                  << " gy=" << s.gy << " [rad/s]"
                  << " gz=" << s.gz << " [rad/s]"
                  << "\n";
    }

    return 0;
}
