#pragma once
#include <string>
#include <vector>
#include <functional>

#if WIN32
#ifdef USE_EXPORTDLL
#define CAMERASDK_API _declspec(dllexport)
#else
#define CAMERASDK_API _declspec(dllimport)
#endif
#else
#define CAMERASDK_API
#endif

namespace ins_camera {
    enum class CameraType {
        Insta360OneX,
        Insta360OneR,
        Insta360OneRS,
        Insta360OneX2,
        Insta360X3,
        Insta360X4,
        Insta360X5,
        Insta360X4Air,
        Unknown,
    };

    enum class ConnectionType {
        USB,
        Wifi,
        Bluetooth
    };

    enum class VideoEncodeType {
        H264,
        H265
    };

    enum class CameraLensType {
        PanoDefault,
        Wide577,
        Pano577,
        Wide283,
        Pano283,
        Wide586,
        Pano586,
        Action577,
    };

    struct DeviceConnectionInfo {
        ConnectionType connection_type;
        void* native_connection_info;
    };

    struct DeviceDescriptor {
        CameraType camera_type;
        std::string serial_number;
        std::string camera_name;
        std::string fw_version;
        DeviceConnectionInfo info;
    };

    struct WindowCropInfo {
        uint32_t src_width;
        uint32_t src_height;
        uint32_t dst_width;
        uint32_t dst_height;
        int32_t crop_offset_x;   //自拍模式偏移量
        int32_t crop_offset_y;
    };

    struct PreviewParam {
        std::string camera_name;
        WindowCropInfo crop_info;
        VideoEncodeType encode_type;
        int64_t delay_timestamp;
        int64_t sweep_time;
        int32_t acceleration_range;
        int32_t gyro_range;
        std::vector<std::string> offset;
    };

    enum SensorDevice {
        SENSOR_DEVICE_FRONT = 1,
        SENSOR_DEVICE_REAR = 2,
        SENSOR_DEVICE_ALL = 3
    };

    enum CardState {
        STOR_CS_PASS = 0,
        STOR_CS_NOCARD = 1,
        STOR_CS_NOSPACE = 2,
        STOR_CS_INVALID_FORMAT = 3,
        STOR_CS_WPCARD = 4,
        STOR_CS_OTHER_ERROR = 5
    };

    struct StorageStatus {
        CardState state;
        uint64_t free_space;
        uint64_t total_space;
    };

    enum LogLevel {
        VERBOSE = 0,
        INFO,
        WARNING,
        ERR,
        FATAL,
    };

    enum class CaptureStoppedErrorCode {
        OVER_TIME_LIMIT = 0,
        STORAGE_FULL = 1,
        OTHER_SITUATION = 2,
        OVER_FILE_NUMBER_LIMIT = 3,
        LOW_CARD_SPEED = 4,
        MUXER_STREAM_ERROR = 5,
        DROP_FRAMES = 6,
        LOW_BATTERY = 7,
        STORAGEFRGMT = 8,
        HIGH_TEMP = 9,
        LOW_POWER_START = 10,
        STORAGE_RUNOUT_START = 11,
        HIGH_TEMP_START = 12,
        TASK_CONFLICT_START = 13,
        FW_UPDATE = 14
    };

    using DownloadProgressCallBack = std::function<void(int64_t download_size, int64_t total_size)>;
    using UploadProgressCallBack = std::function<void(int64_t total_size, int64_t upload_size)>;

    using BatteryLowCallBack = std::function<void(int battery_level)>;
    using StorageFullCallBack = std::function<void()>;
    using CaptureStoppedCallBack = std::function<void(const std::string& url, int err_code)>;
    using TemperatureHighCallBack = std::function<void()>;
    using CaptureStateCallBack = std::function<void(bool is_capture)>;

    class CAMERASDK_API MediaUrl {
    public:
        MediaUrl(const std::vector<std::string>& uris, const std::vector<std::string>& lrv_uris = std::vector<std::string>());
        bool Empty() const;
        bool IsSingleOrigin() const;
        bool IsSingleLRV() const;
        std::string GetSingleOrigin() const;
        std::string GetSingleLRV() const;
        const std::vector<std::string>& OriginUrls() const;
        const std::vector<std::string>& LRVUrls() const;
    private:
        std::vector<std::string> uris_;
        std::vector<std::string> lrv_uris_;
    };
}
