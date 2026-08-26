// SPDX-License-Identifier: MIT
//
// Frame buffer.
//
// Deliberately a plain owned byte buffer rather than an OpenCV Mat. The vision worker
// needs to build and run without OpenCV present, and everything on the critical path
// here -- brightness, blur, perceptual hashing, homography -- is arithmetic over a
// contiguous buffer. OpenCV earns its place in the training pipeline, not in the
// production sampler.
//
// Frames are held in memory and dropped as soon as they have been processed. Nothing
// here writes pixels to disk; see the privacy notes in ObservationPublisher.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace parkfit::vision {

enum class PixelFormat { Gray8, Bgr24, Rgb24 };

/// An owned image buffer.
class Frame {
  public:
    Frame() = default;

    Frame(int width, int height, PixelFormat format)
        : width_(width), height_(height), format_(format),
          data_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                channels_for(format)) {}

    static int channels_for(PixelFormat format) {
        return format == PixelFormat::Gray8 ? 1 : 3;
    }

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }
    [[nodiscard]] PixelFormat format() const { return format_; }
    [[nodiscard]] int channels() const { return channels_for(format_); }
    [[nodiscard]] bool empty() const { return data_.empty() || width_ <= 0 || height_ <= 0; }
    [[nodiscard]] std::size_t size_bytes() const { return data_.size(); }

    [[nodiscard]] std::uint8_t* data() { return data_.data(); }
    [[nodiscard]] const std::uint8_t* data() const { return data_.data(); }

    [[nodiscard]] std::size_t stride() const {
        return static_cast<std::size_t>(width_) * static_cast<std::size_t>(channels());
    }

    [[nodiscard]] const std::uint8_t* row(int y) const {
        return data_.data() + static_cast<std::size_t>(y) * stride();
    }
    [[nodiscard]] std::uint8_t* row(int y) {
        return data_.data() + static_cast<std::size_t>(y) * stride();
    }

    /// Luminance at a pixel, whatever the source format.
    ///
    /// Uses the Rec. 601 weights rather than a plain mean: the eye is far more
    /// sensitive to green, and a flat average makes a green-tinted night frame look
    /// brighter than it is, which is exactly when the darkness check matters most.
    [[nodiscard]] std::uint8_t luma(int x, int y) const {
        if (x < 0 || y < 0 || x >= width_ || y >= height_) return 0;
        const std::uint8_t* p = row(y) + static_cast<std::size_t>(x) * channels();
        if (format_ == PixelFormat::Gray8) return p[0];
        const double b = format_ == PixelFormat::Bgr24 ? p[0] : p[2];
        const double g = p[1];
        const double r = format_ == PixelFormat::Bgr24 ? p[2] : p[0];
        const double y_value = 0.299 * r + 0.587 * g + 0.114 * b;
        return static_cast<std::uint8_t>(std::clamp(y_value, 0.0, 255.0));
    }

    /// Copy into a single-channel greyscale frame.
    [[nodiscard]] Frame to_gray() const {
        if (format_ == PixelFormat::Gray8) return *this;
        Frame out(width_, height_, PixelFormat::Gray8);
        for (int y = 0; y < height_; ++y) {
            std::uint8_t* dst = out.row(y);
            for (int x = 0; x < width_; ++x) dst[x] = luma(x, y);
        }
        return out;
    }

    void fill(std::uint8_t value) { std::fill(data_.begin(), data_.end(), value); }

    /// Release the pixels. Called as soon as a frame has been processed, so image data
    /// spends the minimum possible time resident.
    void release() {
        data_.clear();
        data_.shrink_to_fit();
        width_ = height_ = 0;
    }

  private:
    int width_{0};
    int height_{0};
    PixelFormat format_{PixelFormat::Bgr24};
    std::vector<std::uint8_t> data_;
};

}  // namespace parkfit::vision
