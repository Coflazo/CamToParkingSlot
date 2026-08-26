// SPDX-License-Identifier: MIT
//
// Frame health.
//
// A camera that has gone dark, frozen, been rained on or been knocked out of alignment
// still produces frames. It just produces frames that mean nothing. The single most
// dangerous failure in this product is telling a driver a space is free because the
// detector saw an empty patch of a frozen image from four hours ago -- so nothing is
// published until the frame itself has been shown to be worth believing.
//
// Every check here is cheap enough to run on every sampled frame. At one frame per
// eight seconds that is not a constraint, but the same code runs across many cameras
// on one machine, so it stays arithmetic over a byte buffer.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "parkfit/vision/frame.hpp"

namespace parkfit::vision {

enum class FrameHealth {
    Healthy,
    Dark,
    Blurred,
    Obstructed,
    Frozen,
    PoseChanged,
    Offline,
};

inline const char* to_string(FrameHealth h) {
    switch (h) {
        case FrameHealth::Healthy: return "HEALTHY";
        case FrameHealth::Dark: return "DARK";
        case FrameHealth::Blurred: return "BLURRED";
        case FrameHealth::Obstructed: return "OBSTRUCTED";
        case FrameHealth::Frozen: return "FROZEN";
        case FrameHealth::PoseChanged: return "POSE_CHANGED";
        case FrameHealth::Offline: return "OFFLINE";
    }
    return "OFFLINE";
}

/// A 64-bit difference hash, used to notice a stream that has stopped advancing.
///
/// dHash compares each pixel with its right-hand neighbour on a 9x8 thumbnail, so it
/// encodes *gradients* rather than absolute values. That makes it stable under the slow
/// exposure drift every outdoor camera shows, while still moving the moment anything in
/// the scene actually changes -- which is exactly the distinction between "quiet street"
/// and "frozen stream".
using PerceptualHash = std::uint64_t;

inline PerceptualHash difference_hash(const Frame& frame) {
    constexpr int kW = 9;
    constexpr int kH = 8;
    if (frame.empty()) return 0;

    std::array<double, kW * kH> thumb{};
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            // Box-average the source region rather than point-sample: a single pixel is
            // dominated by sensor noise, which would make the hash flicker on a static
            // scene and hide a genuine freeze.
            const int x0 = x * frame.width() / kW;
            const int x1 = std::max(x0 + 1, (x + 1) * frame.width() / kW);
            const int y0 = y * frame.height() / kH;
            const int y1 = std::max(y0 + 1, (y + 1) * frame.height() / kH);
            double sum = 0.0;
            int count = 0;
            for (int yy = y0; yy < y1; ++yy) {
                for (int xx = x0; xx < x1; ++xx) {
                    sum += frame.luma(xx, yy);
                    ++count;
                }
            }
            thumb[static_cast<std::size_t>(y) * kW + x] = count ? sum / count : 0.0;
        }
    }

    PerceptualHash hash = 0;
    int bit = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW - 1; ++x) {
            const double left = thumb[static_cast<std::size_t>(y) * kW + x];
            const double right = thumb[static_cast<std::size_t>(y) * kW + x + 1];
            if (left > right) hash |= (1ULL << bit);
            ++bit;
        }
    }
    return hash;
}

inline int hamming_distance(PerceptualHash a, PerceptualHash b) {
    std::uint64_t diff = a ^ b;
    int count = 0;
    while (diff) {
        diff &= diff - 1;
        ++count;
    }
    return count;
}

/// Variance of the Laplacian: the standard focus measure.
///
/// A sharp image has strong second derivatives at edges, so their variance is high. Rain
/// on the lens, condensation and defocus all suppress edges and collapse the variance.
inline double laplacian_variance(const Frame& frame) {
    if (frame.width() < 3 || frame.height() < 3) return 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;
    for (int y = 1; y < frame.height() - 1; ++y) {
        for (int x = 1; x < frame.width() - 1; ++x) {
            const double value = -4.0 * frame.luma(x, y) + frame.luma(x - 1, y) +
                                 frame.luma(x + 1, y) + frame.luma(x, y - 1) +
                                 frame.luma(x, y + 1);
            sum += value;
            sum_sq += value * value;
            ++count;
        }
    }
    if (count == 0) return 0.0;
    const double mean = sum / static_cast<double>(count);
    return sum_sq / static_cast<double>(count) - mean * mean;
}

inline double mean_luma(const Frame& frame) {
    if (frame.empty()) return 0.0;
    double sum = 0.0;
    std::size_t count = 0;
    // Every fourth pixel. The mean of a quarter-million samples is indistinguishable
    // from the mean of a million, and this runs on every frame of every camera.
    for (int y = 0; y < frame.height(); y += 2) {
        for (int x = 0; x < frame.width(); x += 2) {
            sum += frame.luma(x, y);
            ++count;
        }
    }
    return count ? sum / static_cast<double>(count) : 0.0;
}

/// Fraction of pixels that are pinned at black or white.
///
/// A lens covered by a leaf, a van parked across the view, or a blown-out sun flare all
/// show up as a large region with no information in it at all.
inline double clipped_fraction(const Frame& frame, int dark = 6, int bright = 249) {
    if (frame.empty()) return 1.0;
    std::size_t clipped = 0;
    std::size_t count = 0;
    for (int y = 0; y < frame.height(); y += 2) {
        for (int x = 0; x < frame.width(); x += 2) {
            const int value = frame.luma(x, y);
            if (value <= dark || value >= bright) ++clipped;
            ++count;
        }
    }
    return count ? static_cast<double>(clipped) / static_cast<double>(count) : 1.0;
}

struct HealthThresholds {
    double min_mean_luma{22.0};
    double max_mean_luma{242.0};
    double min_laplacian_variance{28.0};
    double max_clipped_fraction{0.55};
    /// Two hashes within this distance are the same picture. Small but not zero:
    /// compression noise moves a bit or two between identical frames.
    int frozen_hamming{3};
    /// Consecutive identical frames before the stream is called frozen. A genuinely
    /// quiet street at 04:00 does produce near-identical frames, so a single repeat
    /// proves nothing; several in a row do.
    int frozen_repeats{4};
    /// Reprojection drift, in pixels, beyond which the camera is assumed to have moved
    /// and every calibration built for it is void.
    double pose_drift_px{6.0};
};

struct HealthReport {
    FrameHealth state{FrameHealth::Offline};
    double mean_luma{0.0};
    double sharpness{0.0};
    double clipped{0.0};
    double pose_drift_px{0.0};
    int repeat_count{0};
    std::string detail;

    [[nodiscard]] bool usable() const { return state == FrameHealth::Healthy; }
};

/// Stateful health checker. One per camera; it remembers the previous frame hash.
class FrameHealthChecker {
  public:
    explicit FrameHealthChecker(HealthThresholds thresholds = {}) : thresholds_(thresholds) {}

    /// `pose_drift_px` comes from the pose validator, or a negative value if unchecked.
    HealthReport check(const Frame& frame, double pose_drift_px = -1.0) {
        HealthReport report;
        if (frame.empty()) {
            report.state = FrameHealth::Offline;
            report.detail = "no frame";
            return report;
        }

        report.mean_luma = mean_luma(frame);
        report.sharpness = laplacian_variance(frame);
        report.clipped = clipped_fraction(frame);
        report.pose_drift_px = pose_drift_px;

        const PerceptualHash hash = difference_hash(frame);
        if (has_previous_ && hamming_distance(hash, previous_hash_) <= thresholds_.frozen_hamming) {
            ++repeats_;
        } else {
            repeats_ = 0;
        }
        previous_hash_ = hash;
        has_previous_ = true;
        report.repeat_count = repeats_;

        // Ordered by severity. A camera that has moved invalidates the geometry, which
        // matters more than the picture being slightly soft.
        if (pose_drift_px >= 0.0 && pose_drift_px > thresholds_.pose_drift_px) {
            report.state = FrameHealth::PoseChanged;
            report.detail = "camera pose has drifted; calibration is void";
            return report;
        }
        if (repeats_ >= thresholds_.frozen_repeats) {
            report.state = FrameHealth::Frozen;
            report.detail = "identical frames repeating; stream is not advancing";
            return report;
        }
        if (report.mean_luma < thresholds_.min_mean_luma) {
            report.state = FrameHealth::Dark;
            report.detail = "too dark to read the scene";
            return report;
        }
        if (report.clipped > thresholds_.max_clipped_fraction) {
            report.state = FrameHealth::Obstructed;
            report.detail = "most of the view carries no information";
            return report;
        }
        if (report.mean_luma > thresholds_.max_mean_luma) {
            report.state = FrameHealth::Obstructed;
            report.detail = "view washed out";
            return report;
        }
        if (report.sharpness < thresholds_.min_laplacian_variance) {
            report.state = FrameHealth::Blurred;
            report.detail = "edges too soft; lens obscured or out of focus";
            return report;
        }

        report.state = FrameHealth::Healthy;
        return report;
    }

    void reset() {
        has_previous_ = false;
        repeats_ = 0;
        previous_hash_ = 0;
    }

    [[nodiscard]] const HealthThresholds& thresholds() const { return thresholds_; }

  private:
    HealthThresholds thresholds_;
    PerceptualHash previous_hash_{0};
    bool has_previous_{false};
    int repeats_{0};
};

}  // namespace parkfit::vision
