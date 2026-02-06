#include "insv/insv_video_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>

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
    // No persistent open state currently; handled in probe/stream helpers
    return !path_.empty();
}

namespace {

struct RawFrame {
    double t_video;
    cv::Mat image;
};

double FrameTimestampSec(const AVFrame* frame, const AVStream* stream) {
    if (!frame || !stream) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    int64_t pts = (frame->best_effort_timestamp == AV_NOPTS_VALUE) ? frame->pts : frame->best_effort_timestamp;
    if (pts == AV_NOPTS_VALUE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return pts * av_q2d(stream->time_base);
}

} // namespace

bool InsvVideoDecoder::probe_time_window(double& min_sec, double& max_sec, std::string* error_out) {
    min_sec = std::numeric_limits<double>::quiet_NaN();
    max_sec = std::numeric_limits<double>::quiet_NaN();

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path_.c_str(), nullptr, nullptr) < 0) {
        if (error_out) *error_out = "Failed to open input";
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        if (error_out) *error_out = "Failed to find stream info";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    struct ProbeDecoder {
        int stream_index;
        AVStream* stream;
        AVCodecContext* codec_ctx;
    };

    std::vector<ProbeDecoder> probes;
    probes.reserve(fmt_ctx->nb_streams);
    auto free_probe_contexts = [&]() {
        for (auto& probe_ctx : probes) {
            if (probe_ctx.codec_ctx) {
                avcodec_free_context(&probe_ctx.codec_ctx);
                probe_ctx.codec_ctx = nullptr;
            }
        }
    };
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
            continue;
        }
        const AVCodec* codec = avcodec_find_decoder(fmt_ctx->streams[i]->codecpar->codec_id);
        if (!codec) {
            if (error_out) *error_out = "Decoder not found";
            free_probe_contexts();
            avformat_close_input(&fmt_ctx);
            return false;
        }
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            if (error_out) *error_out = "Failed to alloc codec context";
            free_probe_contexts();
            avformat_close_input(&fmt_ctx);
            return false;
        }
        if (avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[i]->codecpar) < 0 ||
            avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            if (error_out) *error_out = "Failed to open codec";
            avcodec_free_context(&codec_ctx);
            free_probe_contexts();
            avformat_close_input(&fmt_ctx);
            return false;
        }
        probes.push_back({static_cast<int>(i), fmt_ctx->streams[i], codec_ctx});
    }

    if (probes.empty()) {
        if (error_out) *error_out = "No video stream found";
        free_probe_contexts();
        avformat_close_input(&fmt_ctx);
        return false;
    }

    std::vector<ProbeDecoder*> lookup(fmt_ctx->nb_streams, nullptr);
    for (auto& probe : probes) {
        lookup[probe.stream_index] = &probe;
    }

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    if (!frame || !pkt) {
        if (error_out) *error_out = "Failed to allocate ffmpeg buffers";
        if (frame) {
            av_frame_free(&frame);
        }
        if (pkt) {
            av_packet_free(&pkt);
        }
        free_probe_contexts();
        avformat_close_input(&fmt_ctx);
        return false;
    }
    bool saw_frame = false;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    auto drain_probe = [&](ProbeDecoder& probe_ctx) -> bool {
        while (true) {
            int rr = avcodec_receive_frame(probe_ctx.codec_ctx, frame);
            if (rr == AVERROR(EAGAIN) || rr == AVERROR_EOF) {
                break;
            }
            if (rr < 0) {
                if (error_out) *error_out = "Failed to receive frame";
                return false;
            }
            double t = FrameTimestampSec(frame, probe_ctx.stream);
            if (std::isfinite(t)) {
                if (!saw_frame) {
                    local_min = t;
                    local_max = t;
                    saw_frame = true;
                } else {
                    local_min = std::min(local_min, t);
                    local_max = std::max(local_max, t);
                }
            }
            av_frame_unref(frame);
        }
        return true;
    };

    bool ok = true;
    while (ok && av_read_frame(fmt_ctx, pkt) >= 0) {
        ProbeDecoder* target = lookup[pkt->stream_index];
        if (!target) {
            av_packet_unref(pkt);
            continue;
        }
        if (avcodec_send_packet(target->codec_ctx, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }
        av_packet_unref(pkt);
        ok = drain_probe(*target);
    }

    for (auto& probe_ctx : probes) {
        if (!ok) {
            break;
        }
        avcodec_send_packet(probe_ctx.codec_ctx, nullptr);
        ok = drain_probe(probe_ctx);
    }

    if (ok && !saw_frame) {
        ok = false;
        if (error_out) *error_out = "No video frames decoded";
    }

    if (ok) {
        min_sec = local_min;
        max_sec = local_max;
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    free_probe_contexts();
    avformat_close_input(&fmt_ctx);

    return ok;
}

bool InsvVideoDecoder::stream_decode(const std::function<bool(DecodedFrame&&)>& on_frame,
                                     std::string* error_out) {
    if (!on_frame) {
        if (error_out) *error_out = "Frame callback missing";
        return false;
    }

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path_.c_str(), nullptr, nullptr) < 0) {
        if (error_out) *error_out = "Failed to open input";
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        if (error_out) *error_out = "Failed to find stream info";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    struct StreamState {
        int stream_index{-1};
        AVStream* stream{nullptr};
        AVCodecContext* codec_ctx{nullptr};
        SwsContext* sws_ctx{nullptr};
        std::deque<RawFrame> ready_frames;
    };

    std::vector<int> video_indices;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_indices.push_back(static_cast<int>(i));
        }
    }
    if (video_indices.empty()) {
        if (error_out) *error_out = "No video stream found";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    const size_t stream_count = std::min<size_t>(2, video_indices.size());
    std::vector<StreamState> streams;
    streams.reserve(stream_count);
    bool ok = true;
    for (size_t ordinal = 0; ordinal < stream_count; ++ordinal) {
        int idx = video_indices[ordinal];
        AVStream* av_stream = fmt_ctx->streams[idx];
        const AVCodec* codec = avcodec_find_decoder(av_stream->codecpar->codec_id);
        if (!codec) {
            if (error_out) *error_out = "Decoder not found";
            ok = false;
            break;
        }
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            if (error_out) *error_out = "Failed to alloc codec context";
            ok = false;
            break;
        }
        if (avcodec_parameters_to_context(codec_ctx, av_stream->codecpar) < 0 ||
            avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            if (error_out) *error_out = "Failed to open codec";
            avcodec_free_context(&codec_ctx);
            ok = false;
            break;
        }
        StreamState state;
        state.stream_index = idx;
        state.stream = av_stream;
        state.codec_ctx = codec_ctx;
        streams.push_back(std::move(state));
    }

    if (!ok || streams.empty()) {
        for (auto& state : streams) {
            if (state.sws_ctx) {
                sws_freeContext(state.sws_ctx);
            }
            if (state.codec_ctx) {
                avcodec_free_context(&state.codec_ctx);
            }
        }
        avformat_close_input(&fmt_ctx);
        return false;
    }

    std::vector<StreamState*> lookup(fmt_ctx->nb_streams, nullptr);
    for (auto& state : streams) {
        lookup[state.stream_index] = &state;
    }

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    bool stop_requested = false;

    auto convert_frame = [&](StreamState& state) -> bool {
        while (true) {
            int rr = avcodec_receive_frame(state.codec_ctx, frame);
            if (rr == AVERROR(EAGAIN) || rr == AVERROR_EOF) {
                break;
            }
            if (rr < 0) {
                if (error_out) *error_out = "Failed to receive frame";
                return false;
            }

            state.sws_ctx = sws_getCachedContext(
                state.sws_ctx,
                frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                frame->width, frame->height, AV_PIX_FMT_BGR24,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!state.sws_ctx) {
                if (error_out) *error_out = "Failed to init scaler";
                return false;
            }

            cv::Mat bgr(frame->height, frame->width, CV_8UC3);
            uint8_t* dst_data[4] = { bgr.data, nullptr, nullptr, nullptr };
            int dst_linesize[4] = { static_cast<int>(bgr.step[0]), 0, 0, 0 };
            sws_scale(state.sws_ctx,
                      (const uint8_t* const*)frame->data, frame->linesize,
                      0, frame->height,
                      dst_data, dst_linesize);

            RawFrame rf;
            rf.t_video = FrameTimestampSec(frame, state.stream);
            rf.image = std::move(bgr);
            state.ready_frames.push_back(std::move(rf));
            av_frame_unref(frame);
        }
        return true;
    };

    auto emit_frames = [&]() -> bool {
        if (streams.empty()) {
            return true;
        }
        if (streams.size() == 1) {
            auto& queue = streams[0].ready_frames;
            while (!queue.empty()) {
                RawFrame raw = std::move(queue.front());
                queue.pop_front();
                DecodedFrame df;
                int half_w = raw.image.cols / 2;
                int other_w = raw.image.cols - half_w;
                if (half_w <= 0 || other_w <= 0) {
                    continue;
                }
                df.rear = raw.image(cv::Rect(0, 0, half_w, raw.image.rows)).clone();
                df.front = raw.image(cv::Rect(half_w, 0, other_w, raw.image.rows)).clone();
                df.t_video = raw.t_video;
                if (!on_frame(std::move(df))) {
                    return false;
                }
            }
        } else {
            auto& front_q = streams[0].ready_frames;
            auto& rear_q = streams[1].ready_frames;
            while (!front_q.empty() && !rear_q.empty()) {
                RawFrame front = std::move(front_q.front());
                RawFrame rear = std::move(rear_q.front());
                front_q.pop_front();
                rear_q.pop_front();
                DecodedFrame df;
                df.front = std::move(front.image);
                df.rear = std::move(rear.image);
                df.t_video = std::min(front.t_video, rear.t_video);
                if (!on_frame(std::move(df))) {
                    return false;
                }
            }
        }
        return true;
    };

    while (ok && !stop_requested && av_read_frame(fmt_ctx, pkt) >= 0) {
        StreamState* target = lookup[pkt->stream_index];
        if (!target) {
            av_packet_unref(pkt);
            continue;
        }
        if (avcodec_send_packet(target->codec_ctx, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }
        av_packet_unref(pkt);
        ok = convert_frame(*target);
        if (ok) {
            stop_requested = !emit_frames();
        }
    }

    for (auto& state : streams) {
        if (!ok || stop_requested) {
            break;
        }
        avcodec_send_packet(state.codec_ctx, nullptr);
        ok = convert_frame(state);
        if (ok) {
            stop_requested = !emit_frames();
        }
    }

    if (!ok && error_out && error_out->empty()) {
        *error_out = "Video decode failed";
    }

    if (!stop_requested && ok) {
        stop_requested = !emit_frames();
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    for (auto& state : streams) {
        if (state.sws_ctx) {
            sws_freeContext(state.sws_ctx);
        }
        if (state.codec_ctx) {
            avcodec_free_context(&state.codec_ctx);
        }
    }
    avformat_close_input(&fmt_ctx);

    return ok || stop_requested;
}

} // namespace insta360_insv
