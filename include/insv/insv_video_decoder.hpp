#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace insta360_insv {

struct DecodedFrame {
    cv::Mat front;   
    cv::Mat rear;    
    double t_video;  // seconds, computed from pts * time_base
};

class InsvVideoDecoder {
public:
    explicit InsvVideoDecoder(const std::string& path, int decode_threads = 0);
    ~InsvVideoDecoder();

    void set_decode_threads(int decode_threads);

    bool open(std::string* error_out = nullptr);
    bool probe_time_window(double& min_sec, double& max_sec, std::size_t* frame_count = nullptr, std::string* error_out = nullptr);
    bool stream_decode(const std::function<bool(DecodedFrame&&)>& on_frame,
                      std::string* error_out = nullptr);

private:
    std::string path_;
    int decode_threads_{0};
};

} // namespace insta360_insv
