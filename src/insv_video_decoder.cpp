#include "insv/insv_video_decoder.hpp"

#include <algorithm>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
}

namespace insta360_insv {

InsvVideoDecoder::InsvVideoDecoder(const std::string& path) : path_(path) {}
InsvVideoDecoder::~InsvVideoDecoder() {}

bool InsvVideoDecoder::open(std::string* /*error_out*/) {
    // No persistent open state currently; handled in decode_all()
    return !path_.empty();
}

namespace {
struct RawFrame {
    double t_video;
    cv::Mat image;
};
} // namespace

static bool DecodeStreamOrdinal(const std::string& path,
                                int video_ordinal,
                                std::vector<RawFrame>& out_frames,
                                std::string* error_out) {
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        if (error_out) *error_out = "Failed to reopen input";
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        if (error_out) *error_out = "Failed to find stream info";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    int target_stream = -1;
    int seen = 0;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (seen == video_ordinal) {
                target_stream = static_cast<int>(i);
                break;
            }
            ++seen;
        }
    }
    if (target_stream < 0) {
        if (error_out) *error_out = "Video stream ordinal not found";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    AVStream* vstream = fmt_ctx->streams[target_stream];
    const AVCodec* codec = avcodec_find_decoder(vstream->codecpar->codec_id);
    if (!codec) {
        if (error_out) *error_out = "Decoder not found";
        avformat_close_input(&fmt_ctx);
        return false;
    }
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        if (error_out) *error_out = "Failed to alloc codec context";
        avformat_close_input(&fmt_ctx);
        return false;
    }
    avcodec_parameters_to_context(codec_ctx, vstream->codecpar);
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        if (error_out) *error_out = "Failed to open codec";
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return false;
    }

    SwsContext* sws_ctx = nullptr;
    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != target_stream) {
            av_packet_unref(pkt);
            continue;
        }

        if (avcodec_send_packet(codec_ctx, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }
        av_packet_unref(pkt);

        while (true) {
            int rr = avcodec_receive_frame(codec_ctx, frame);
            if (rr == AVERROR(EAGAIN) || rr == AVERROR_EOF) break;
            if (rr < 0) break;

            double tb = av_q2d(vstream->time_base);
            int64_t pts = (frame->best_effort_timestamp == AV_NOPTS_VALUE) ? frame->pts : frame->best_effort_timestamp;
            double t_video = (pts == AV_NOPTS_VALUE) ? 0.0 : pts * tb;

            if (!sws_ctx) {
                sws_ctx = sws_getContext(
                    frame->width, frame->height, (AVPixelFormat)frame->format,
                    frame->width, frame->height, AV_PIX_FMT_BGR24,
                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws_ctx) {
                    av_frame_unref(frame);
                    continue;
                }
            }

            cv::Mat bgr(frame->height, frame->width, CV_8UC3);
            uint8_t* dst_data[4] = { bgr.data, nullptr, nullptr, nullptr };
            int dst_linesize[4] = { static_cast<int>(bgr.step[0]), 0, 0, 0 };
            sws_scale(sws_ctx,
                      (const uint8_t* const*)frame->data, frame->linesize,
                      0, frame->height,
                      dst_data, dst_linesize);

            RawFrame rf;
            rf.t_video = t_video;
            rf.image = bgr.clone();
            out_frames.push_back(std::move(rf));

            av_frame_unref(frame);
        }
    }

    if (sws_ctx) sws_freeContext(sws_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    return true;
}

bool InsvVideoDecoder::decode_all(std::vector<DecodedFrame>& out_frames, std::string* error_out) {
    out_frames.clear();

    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, path_.c_str(), nullptr, nullptr);
    if (ret < 0) {
        if (error_out) *error_out = "Failed to open input";
        return false;
    }
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        if (error_out) *error_out = "Failed to find stream info";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    int video_stream_count = 0;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            ++video_stream_count;
        }
    }
    avformat_close_input(&fmt_ctx);

    if (video_stream_count == 0) {
        if (error_out) *error_out = "No video stream found";
        return false;
    }

    std::vector<RawFrame> stream_a;
    std::vector<RawFrame> stream_b;

    if (!DecodeStreamOrdinal(path_, 0, stream_a, error_out)) {
        return false;
    }

    // If there is only one video stream, split the side-by-side frame into rear/front halves. 
    // TODO: remove this split logic for single stream INSV files.
    if (video_stream_count == 1) {
        for (const auto& rf : stream_a) {
            DecodedFrame df;
            int half_w = rf.image.cols / 2;
            df.rear = rf.image(cv::Rect(0, 0, half_w, rf.image.rows)).clone();
            df.front = rf.image(cv::Rect(half_w, 0, half_w, rf.image.rows)).clone();
            df.t_video = rf.t_video;
            out_frames.push_back(std::move(df));
        }
    } else {
        if (!DecodeStreamOrdinal(path_, 1, stream_b, error_out)) {
            return false;
        }
        size_t paired = std::min(stream_a.size(), stream_b.size());
        for (size_t i = 0; i < paired; ++i) {
            DecodedFrame df;
            df.front = stream_a[i].image;
            df.rear = stream_b[i].image;
            df.t_video = std::min(stream_a[i].t_video, stream_b[i].t_video);
            out_frames.push_back(std::move(df));
        }
    }

    return true;
}

} // namespace insta360_insv
