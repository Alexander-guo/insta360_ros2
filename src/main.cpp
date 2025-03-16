#include <iostream>
#include <thread>
#include <regex>
#include <opencv2/opencv.hpp>

#include "camera/camera.h"
#include "camera/photography_settings.h"
#include "camera/device_discovery.h"
#include "camera/ins_types.h"

#include "stream/stream_delegate.h"
#include "stream/stream_types.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <atomic>
#include <signal.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    FILE* file1_;
    FILE* file2_;
    int64_t last_timestamp = 0;
    AVCodec* codec;
    AVCodecContext* codecCtx;
    AVFrame* avFrame;
    AVPacket* pkt;
    struct SwsContext* img_convert_ctx;
    

    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dual_fisheye_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

    double fx, fy, cx, cy;
    std::vector<double> distCoeffs;

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {
        dual_fisheye_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("/dual_fisheye/image", rclcpp::QoS(0));
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::QoS(0));

        file1_ = fopen("./01.h264", "wb");
        file2_ = fopen("./02.h264", "wb");
        codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec) {
            std::cerr << "Codec not found\n";
            exit(1);
        }

        codecCtx = avcodec_alloc_context3(codec);
        codecCtx->flags2 |= AV_CODEC_FLAG2_FAST;
        if (!codecCtx) {
            std::cerr << "Could not allocate video codec context\n";
            exit(1);
        }
        if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
            std::cerr << "Could not open codec\n";
            exit(1);
        }
        avFrame = av_frame_alloc();
        pkt = av_packet_alloc();
    }

    ~TestStreamDelegate() {
        fclose(file1_);
        fclose(file2_);
        av_frame_free(&avFrame);
        av_packet_free(&pkt);
        avcodec_free_context(&codecCtx);
    }

    cv::Mat rotateImage(const cv::Mat& src, double angle) {
        cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat dst;
        cv::warpAffine(src, dst, rot, src.size());
        return dst;
    }

    sensor_msgs::msg::Image::SharedPtr matToImgMsg(const cv::Mat& image, const std::string& frame_id) {
        auto img_msg = std::make_shared<sensor_msgs::msg::Image>();
        img_msg->header.stamp = node_->get_clock()->now();
        img_msg->header.frame_id = frame_id;
        img_msg->height = image.rows;
        img_msg->width = image.cols;
        img_msg->encoding = "rgb8";
        img_msg->is_bigendian = 0;
        img_msg->step = image.cols * image.elemSize();
        size_t size = img_msg->step * image.rows;
        img_msg->data.resize(size);
        memcpy(img_msg->data.data(), image.data, size);
        return img_msg;
    }

    cv::Mat avframeToCvmat(const AVFrame *frame) {
        int width = frame->width;
        int height = frame->height;
        cv::Mat yuv(height + height / 2, width, CV_8UC1, frame->data[0]);
        cv::Mat rgb;
        cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);
        return rgb;
    }

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {
        // std::cout << "on audio data:" << std::endl;
    }

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index = 0) override {
        if (stream_index == 0)
        {
            pkt->data = const_cast<uint8_t*>(data);
            pkt->size = size;
        }
        if (avcodec_send_packet(codecCtx, pkt) == 0) {
            while (avcodec_receive_frame(codecCtx, avFrame) == 0) {
                int width = avFrame->width;
                int height = avFrame->height;

                int chromaHeight = height / 2;
                int chromaWidth = width / 2;
                cv::Mat yuv(height + chromaHeight, width, CV_8UC1);
                memcpy(yuv.data, avFrame->data[0], width * height);
                memcpy(yuv.data + width * height, avFrame->data[1], chromaWidth * chromaHeight);
                memcpy(yuv.data + width * height + chromaWidth * chromaHeight, avFrame->data[2], chromaWidth * chromaHeight);

                cv::Mat rgb;
                cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_I420);

                int midPoint = width / 2;
                cv::Mat frontImage = rgb(cv::Rect(midPoint, 0, midPoint, height));
                cv::Mat backImage = rgb(cv::Rect(0, 0, midPoint, height));

                frontImage = rotateImage(frontImage, 90);
                backImage = rotateImage(backImage, -90);

                cv::Mat dualFisheyeImage;
                cv::hconcat(frontImage, backImage, dualFisheyeImage);
                
                auto dualFisheyeMsg = matToImgMsg(dualFisheyeImage, "dual_fisheye_frame");
                dual_fisheye_pub_->publish(*dualFisheyeMsg);
            }
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        for (auto& gyro : data) {
            sensor_msgs::msg::Imu msg;
            msg.header.stamp = node_->get_clock()->now();
            msg.header.frame_id = "imu_frame";
            msg.angular_velocity.x = gyro.gx;
            msg.angular_velocity.y = gyro.gy;
            msg.angular_velocity.z = gyro.gz;
            msg.linear_acceleration.x = gyro.ax * 9.81;
            msg.linear_acceleration.y = gyro.ay * 9.81;
            msg.linear_acceleration.z = gyro.az * 9.81;
            imu_pub_->publish(msg);
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<rclcpp::Node> node_;

public:
    CameraWrapper(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {
        RCLCPP_ERROR(node_->get_logger(), "Opened Camera");
    }

    ~CameraWrapper() {
        RCLCPP_ERROR(node_->get_logger(), "Closing Camera");
        cam->Close();
    }

    int run_camera() {
        ins_camera::DeviceDiscovery discovery;
        auto list = discovery.GetAvailableDevices();
        for (int i = 0; i < list.size(); ++i) {
            auto desc = list[i];
            std::cout << "serial:" << desc.serial_number << "\t"
                << "camera type:" << int(desc.camera_type) << "\t"
                << "lens type:" << int(desc.lens_type) << std::endl;
        }
        if (list.size() <= 0) {
            std::cerr << "no device found." << std::endl;
            return -1;
        }
        cam = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam->Open()) {
            std::cerr << "failed to open camera" << std::endl;
            return -1;
        }
        std::cout << "http base url:" << cam->GetHttpBaseUrl() << std::endl;
        std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>(node_);
        cam->SetStreamDelegate(delegate);
        discovery.FreeDeviceDescriptors(list);
        std::cout << "Successfully opened camera..." << std::endl;
        auto camera_type = cam->GetCameraType();
        auto start = time(NULL);
        cam->SyncLocalTimeToCamera(start);
        ins_camera::LiveStreamParam param;
        // param.video_resolution = ins_camera::VideoResolution::RES_2560_1280P30;
        // param.video_resolution = ins_camera::VideoResolution::RES_1152_1152P30;
        param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30;
        param.video_bitrate = 1024 * 1024 * 10;
        param.using_lrv = false;
        do {} while (!cam->StartLiveStreaming(param));
        std::cout << "successfully started live stream" << std::endl;
        return 0;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("insta");
    {
        CameraWrapper camera(node);
        camera.run_camera();
        rclcpp::spin(node);
    }
    rclcpp::shutdown();
    return 0;
}
