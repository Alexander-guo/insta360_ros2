#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace insta360_insv {

struct DecodedFrame {
    cv::Mat front;   // right half of SxS frame
    cv::Mat rear;    // left half of SxS frame
    double t_video;  // seconds, computed from pts * time_base
};

class InsvVideoDecoder {
public:
    explicit InsvVideoDecoder(const std::string& path);
    ~InsvVideoDecoder();

    bool open(std::string* error_out = nullptr);
    bool decode_all(std::vector<DecodedFrame>& out_frames, std::string* error_out = nullptr);

private:
    std::string path_;
};

} // namespace insta360_insv
