#pragma once

#include <string>
#include <memory>
#include <vector>
#include "ins_types.h"
#include "photography_settings.h"
#include "../stream/stream_delegate.h"
namespace ins_camera {
    class CameraImpl;
    /**
     * \class Camera
     * \brief A class that represents Insta360 camera,support ONE X, ONE R, ONERS，ONEX2，X3，X4, X5
     */
     // set path of the log
    void CAMERASDK_API SetLogPath(const std::string& log_path);

    // set log level
    void CAMERASDK_API SetLogLevel(LogLevel level);

    class CAMERASDK_API Camera {
    public:
        /**
         * \brief see also DeviceDiscovery to get DeviceConnectionInfo.
         */
        Camera(const DeviceConnectionInfo& info);
        /**
         * \brief Open camera and start session
         * \return true if succeed, otherwise return false;
         */
        bool Open() const;
        /**
         * \brief Close camera and release resources
         */
        void Close() const;

        /**
        * \brief get capture current status of camera.
        * be sure to call this function after Open()
        * \return lens type of the camera
        */
        bool CaptureCurrentStatus() const;

        /**
        * \brief set service port of download service
        * be sure to call this function before Open()
        */
        void SetServicePort(int port);

        /**
         * \brief Control camera to take normal photo
         * \return The url of the photo if success, otherwise empty
         */
        MediaUrl TakePhoto() const;

        /**
         * \brief Control camera to take hdr photo
         * \return The url of the photo if success, otherwise empty
         */
        MediaUrl StartHDRCapture(const PhotoSize& photo_size, bool is_raw = false);

        /**
         * \brief set exposure settings, the settings will only be applied to specified mode
         * \param mode the target mode you want to apply exposure settings, for normal video recording,
         *      use CameraFunctionMode::FUNCTION_MODE_NORMAL_VIDEO, for normal still image capture,
         *      use CaperaFunctionMode::FUNCTION_MODE_NORMAL_IMAGE.
         * \param settings ExposureSettings containing exposure mode/iso/shutter/ev to be applied.
         * \return true on success, false otherwise.
         */
        bool SetExposureSettings(CameraFunctionMode mode, const std::shared_ptr<ExposureSettings>& settings);

        std::shared_ptr<ExposureSettings> GetExposureSettings(CameraFunctionMode mode) const;

        /**
         * \brief set capture settings, the settings will only be applied to specified mode
         * \param mode the target mode you want to apply exposure settings, for normal video recording,
         *      use CameraFunctionMode::FUNCTION_MODE_NORMAL_VIDEO, for normal still image capture,
         *      use CaperaFunctionMode::FUNCTION_MODE_NORMAL_IMAGE.
         * \param settings CaptureSettings containing capture settings like saturation,contrast,whitebalance,sharpness,brightness and etc.
         * \return true on success, false otherwise.
         */
        bool SetCaptureSettings(CameraFunctionMode mode, std::shared_ptr<CaptureSettings> settings);

        std::shared_ptr<CaptureSettings> GetCaptureSettings(CameraFunctionMode mode) const;

        /**
         * \brief set capture settings such as resolutions, bitrate,
         * \param params RecordParams containing settings you want to apply
         * \param mode the target mode you want to apply capture settings, the mode must be one of video modes.
         * \return true on success, false otherwise.
         */
        bool SetVideoCaptureParams(RecordParams params, CameraFunctionMode mode = CameraFunctionMode::FUNCTION_MODE_NORMAL_VIDEO);

        /**
         * \brief Start Recording.
         * \return true on success, false otherwise
         */
        bool StartRecording();

        /**
         * \brief Stop Recording.
         * \return video uri of the video, it may contain low bitrate proxy video
         */
        MediaUrl StopRecording();

        /**
        * \brief start preview stream
        */
        bool StartLiveStreaming(const LiveStreamParam& param);

        /**
        * \brief stop preview stream
        */
        bool StopLiveStreaming();

        /**
            \brief set a stream delegate, you may implement StreamDelegate to handle stream data.
         */
        void SetStreamDelegate(std::shared_ptr<StreamDelegate>& delegate);

        /**
         * \brief Delete the specified file from camera
         */
        bool DeleteCameraFile(const std::string& filePath) const;

        /**
         * \brief Download the specified file from camera
         */
        bool DownloadCameraFile(const std::string& remoteFilePath, const std::string& localFilePath, DownloadProgressCallBack callback = nullptr) const;

        /**
         * \brief upload Firmware to camera. only support X4
         */
        bool UploadFile(const std::string& localFilePath, const std::string& remoteFilePath, UploadProgressCallBack callback = nullptr) const;

        /**
         * \brief cancal the file that downloading, and it will not be deleted it
         * need delete it by caller.
         */
        void CancelDownload();

        /**
         * \brief Get the list of files stored in camera storage card
         */
        std::vector<std::string> GetCameraFilesList() const;

        /**
         * \brief Get the count of files stored in camera storage card
         */
        bool GetCameraFilesCount(int& count) const;

        /**
        * \brief set timelapse param
        */
        bool SetTimeLapseOption(TimelapseParam params);

        /**
        * \brief start timelapse
        */
        bool StartTimeLapse(CameraTimelapseMode mode);

        /**
        * \brief stop timelapse
        * \return video uri of the video, it may contain low bitrate proxy video
        */
        MediaUrl StopTimeLapse(CameraTimelapseMode mode);

        /**
        * \brief sync local time to camera
        */
        bool SyncLocalTimeToCamera(uint64_t utc_time, uint32_t offset_time);

        /**
        * \brief device is connected
        */
        bool IsConnected();

        /**
        * \brief SetTimeout, default is 10000ms
        */
        void SetTimeout(int time_out_ms);

        /**
        * \brief get battry info
        */
        bool GetBatteryStatus(BatteryStatus& status);

        /**
        * \brief get storage state
        */
        bool GetStorageState(StorageStatus& status);

        /**
        * \brief set photo size on photo mode
        */
        bool SetPhotoSize(CameraFunctionMode mode, const PhotoSize& photo_size);

        /**
        * \brief switch sensor
        */
        bool SetActiveSensor(SensorDevice sensor_device);

        /**
        * \brief get media time from camera
        */
        int64_t GetCameraMediaTime() const;

        /**
        * \brief set sub mode on video mode
        */
        bool SetVideoSubMode(SubVideoMode video_mode);

        /**
        * \brief set sub mode on photo mode
        */
        bool SetPhotoSubMode(SubPhotoMode photo_mode);

        /**
        * \brief Obtain the information of the file being recorded
        */
        bool GetRecordingFiles(std::vector<std::string>& file_list);

        /**
        * \brief preview param of camera
        * be sure to call this function after Open()
        */

        PreviewParam GetPreviewParam() const;

        /**
       * \brief get offset of camera
       * be sure to call this function after Open()
       */
        std::vector<std::string> GetCameraOffset() const;

        /**
        * \brief get current camera encode type(h264 or h265)
        */
        VideoEncodeType GetVideoEncodeType() const;

        /**
        * \brief enable In-camera stitching
        */
        bool EnableInCameraStitching(bool enable);

        /**
        * \brief shutdown camera
        */
        bool ShutdownCamera() const;

        /**
        * \brief Get url of camera log
        */
        std::string GetCameraLogFileUrl() const;

        /**
        * \brief Set the notification callback for low battery
        */
        void SetBatteryLowNotification(BatteryLowCallBack callback);

        /**
        * \brief Set the notification callback for storage full
        */
        void SetStorageFullNotification(StorageFullCallBack callback);

        /**
        * \brief Set the notification callback for recording termination
        * \ param: error_code:  "CaptureStoppedErrorCode"
        */
        void SetCaptureStoppedNotification(CaptureStoppedCallBack callback);

        /**
        * \brief Set the notification callback for temperature high
        */
        void SetTemperatureHighNotification(TemperatureHighCallBack callback);

        void SetCaptureStateNotification(CaptureStateCallBack callback);

        CameraLensType GetCameraLensType() const;

        std::string GetHttpBaseUrl() const;

    private:
        std::shared_ptr<CameraImpl> impl_;
    };
}
