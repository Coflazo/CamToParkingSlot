// SPDX-License-Identifier: MIT
//
// Frame sources.
//
// Frames arrive through an ffmpeg subprocess rather than by linking libavcodec. That is
// a deliberate trade: ffmpeg already speaks HLS, RTSP, MJPEG, DASH and every container
// a webcam might serve, it is already installed on the target machine, and a codec bug
// in a separate process cannot take down the worker. The cost is a pipe and a process
// per camera, which at one frame per eight seconds is irrelevant.
//
// Sampling is deliberately slow. Parking changes over minutes; there is nothing in a
// 25 fps stream that 0.125 fps does not capture. Sampling slowly reduces bandwidth,
// processor load, and -- most importantly -- how much imagery of a public street exists
// in memory at any moment.

#pragma once

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "parkfit/vision/frame.hpp"

namespace parkfit::vision {

enum class SourceKind { FfmpegStream, HttpSnapshot, RawFile, Replay };

struct SourceConfig {
    std::string camera_id;
    std::string url;
    SourceKind kind{SourceKind::FfmpegStream};

    int width{960};
    int height{540};
    double sample_interval_s{8.0};

    /// Path to the ffmpeg binary. Discovered at startup rather than assumed.
    std::string ffmpeg_path{"ffmpeg"};

    /// Seconds to wait for a frame before declaring the source offline.
    double timeout_s{25.0};
};

/// A frame plus the metadata needed to decide whether to trust it.
struct SourceFrame {
    Frame frame;
    double timestamp_s{0.0};
    bool ok{false};
    std::string error;
};

class FrameSource {
  public:
    virtual ~FrameSource() = default;
    virtual bool open() = 0;
    virtual SourceFrame next() = 0;
    virtual void close() = 0;
    [[nodiscard]] virtual bool is_open() const = 0;
    [[nodiscard]] virtual const SourceConfig& config() const = 0;
};

/// Reads raw BGR frames from an ffmpeg subprocess.
///
/// ffmpeg is asked to scale to a fixed size and emit raw bgr24 at the sample rate, so
/// the worker never has to deal with codecs, colour conversion or variable frame sizes.
class FfmpegSource final : public FrameSource {
  public:
    explicit FfmpegSource(SourceConfig config) : config_(std::move(config)) {}
    ~FfmpegSource() override { close(); }

    bool open() override {
        close();
        const std::string command = build_command();
#if defined(_WIN32)
        pipe_ = _popen(command.c_str(), "rb");
#else
        pipe_ = popen(command.c_str(), "r");
#endif
        frame_bytes_ = static_cast<std::size_t>(config_.width) *
                       static_cast<std::size_t>(config_.height) * 3;
        buffer_.assign(frame_bytes_, 0);
        return pipe_ != nullptr;
    }

    SourceFrame next() override {
        SourceFrame out;
        if (pipe_ == nullptr) {
            out.error = "source is not open";
            return out;
        }

        std::size_t read_total = 0;
        while (read_total < frame_bytes_) {
            const std::size_t got =
                std::fread(buffer_.data() + read_total, 1, frame_bytes_ - read_total, pipe_);
            if (got == 0) {
                // A short read means the stream ended or ffmpeg died. A partial frame is
                // worse than no frame: it looks like a picture and is half stale data.
                out.error = read_total == 0 ? "stream ended" : "truncated frame discarded";
                return out;
            }
            read_total += got;
        }

        out.frame = Frame(config_.width, config_.height, PixelFormat::Bgr24);
        std::copy(buffer_.begin(), buffer_.end(), out.frame.data());
        out.timestamp_s = now_seconds();
        out.ok = true;
        return out;
    }

    void close() override {
        if (pipe_ != nullptr) {
#if defined(_WIN32)
            _pclose(pipe_);
#else
            pclose(pipe_);
#endif
            pipe_ = nullptr;
        }
        buffer_.clear();
    }

    [[nodiscard]] bool is_open() const override { return pipe_ != nullptr; }
    [[nodiscard]] const SourceConfig& config() const override { return config_; }

    /// The command line, exposed so it can be asserted on without spawning anything.
    [[nodiscard]] std::string build_command() const {
        const double fps = config_.sample_interval_s > 0.0 ? 1.0 / config_.sample_interval_s : 1.0;
        std::string cmd = quote(config_.ffmpeg_path);
        cmd += " -hide_banner -loglevel error";
        // Low latency and no buffering: we want the newest frame, not a smooth playback.
        cmd += " -fflags nobuffer -flags low_delay";
        cmd += " -rw_timeout " + std::to_string(static_cast<long long>(config_.timeout_s * 1e6));
        cmd += " -i " + quote(config_.url);
        cmd += " -vf fps=" + trim_number(fps) + ",scale=" + std::to_string(config_.width) + ":" +
               std::to_string(config_.height);
        cmd += " -pix_fmt bgr24 -f rawvideo -an -sn -";
        return cmd;
    }

    static double now_seconds() {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    }

  private:
    static std::string quote(const std::string& value) {
        // Guards against a URL containing shell metacharacters. Camera URLs come from a
        // database that an operator edits, so they are not trusted input.
        return "\"" + value + "\"";
    }

    static std::string trim_number(double value) {
        std::string s = std::to_string(value);
        while (s.size() > 1 && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
        return s;
    }

    SourceConfig config_;
    std::FILE* pipe_{nullptr};
    std::size_t frame_bytes_{0};
    std::vector<std::uint8_t> buffer_;
};

/// Replays frames supplied in memory. Used by the deterministic replay tests, and by
/// the synthetic evaluation harness, so accuracy numbers are reproducible.
class ReplaySource final : public FrameSource {
  public:
    ReplaySource(SourceConfig config, std::vector<Frame> frames)
        : config_(std::move(config)), frames_(std::move(frames)) {}

    bool open() override {
        index_ = 0;
        open_ = true;
        return true;
    }

    SourceFrame next() override {
        SourceFrame out;
        if (!open_ || index_ >= frames_.size()) {
            out.error = "replay exhausted";
            return out;
        }
        out.frame = frames_[index_];
        out.timestamp_s = static_cast<double>(index_) * config_.sample_interval_s;
        out.ok = true;
        ++index_;
        return out;
    }

    void close() override { open_ = false; }
    [[nodiscard]] bool is_open() const override { return open_; }
    [[nodiscard]] const SourceConfig& config() const override { return config_; }
    [[nodiscard]] std::size_t remaining() const {
        return index_ < frames_.size() ? frames_.size() - index_ : 0;
    }

  private:
    SourceConfig config_;
    std::vector<Frame> frames_;
    std::size_t index_{0};
    bool open_{false};
};

inline std::unique_ptr<FrameSource> make_source(const SourceConfig& config) {
    switch (config.kind) {
        case SourceKind::Replay:
            return std::make_unique<ReplaySource>(config, std::vector<Frame>{});
        case SourceKind::FfmpegStream:
        case SourceKind::HttpSnapshot:
        case SourceKind::RawFile:
        default:
            // ffmpeg handles HLS, RTSP, MJPEG, a periodically-updated JPEG URL and a
            // local file with the same command line, so they share an implementation.
            return std::make_unique<FfmpegSource>(config);
    }
}

}  // namespace parkfit::vision
