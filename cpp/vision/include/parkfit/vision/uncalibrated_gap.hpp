// SPDX-License-Identifier: MIT
//
// Kerb gaps from an uncalibrated camera.
//
// This is the poorer of the two gap finders in this module, and it exists because the
// better one cannot always run. `CurbGapEstimator` in gap.hpp projects detections onto a
// surveyed kerb centreline through a validated homography and measures in real metres.
// It needs calibration: four or more world points whose ground positions are known. For
// a camera somebody points at a street and publishes, nobody has surveyed anything.
//
// So this measures in the image instead, and converts with one number: the median
// detected car width against a typical car. That is a genuinely worse instrument. It
// assumes cars at one depth along one kerb, it has no notion of perspective, and its
// distances are estimates rather than measurements. Everything it returns is labelled
// that way, and it is never used where a calibrated reading exists.
//
// It lives in C++ because it runs on a two-second loop per watched camera rather than
// once per search, and because the occlusion guard below is quadratic in the detection
// count. The Python original in `parkfit/services/camera_analysis.py` stays as the
// reference the parity tests compare against.
//
// Three guards here each exist because the Python version, without them, reported
// something false on a real frame:
//
//  * **the gap ceiling** stopped 37.6 m of open Groningen road being offered as parking;
//  * **the occlusion scan** stopped a confident 22.4 m "space" that ran straight across
//    two dozen parked cars at Kijkduin, where a perpendicular car park broke the
//    one-depth assumption the kerb band relies on;
//  * **the depth floor** stopped a bicycle-flanked stretch of pavement being offered.
//
// None of the three is an optimisation. Each is the difference between a claim that is
// wrong and one that is not made.

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace parkfit::vision {

/// One detection in image space, in pixels.
struct ImageBox {
    double x1{};
    double y1{};
    double x2{};
    double y2{};
    /// True when this class can stand at the side of a gap. A bicycle cannot: two bikes
    /// with a stretch of pavement between them is not a parking space.
    bool flanking{false};
    /// True when this detection is a car, which is what the scale is derived from. A
    /// lorry is a vehicle and a terrible ruler.
    bool is_car{false};

    [[nodiscard]] double width() const { return x2 - x1; }
    [[nodiscard]] double height() const { return y2 - y1; }
    [[nodiscard]] double centre_y() const { return (y1 + y2) / 2.0; }
};

struct UncalibratedGapConfig {
    /// The median car this scale assumes, in metres. Taken from the real RDW fleet used
    /// elsewhere in the product rather than from a round number.
    double typical_car_width_m{1.80};
    /// Below this a gap is not a parking space, whatever it measures.
    double min_gap_m{4.2};
    /// Above this it is not a gap between two parked cars; it is a stretch of road with
    /// nothing on it, and offering it as parking is how a driver ends up on a junction.
    double max_gap_m{15.0};
    /// A space shallower than this is not deep enough to hold a car, so whatever the two
    /// flanking objects are, they are not parked cars either side of one.
    double min_depth_m{1.2};
    /// Fewer cars than this and the median is one or two boxes, which is not a
    /// measurement. The scale is still returned; it is just not called confident.
    std::size_t min_cars_for_confident_scale{3};
    /// A detected box narrower than this is noise, not a car.
    double min_car_width_px{4.0};
    /// Floor for the kerb band's half-height, so a frame of small distant cars still
    /// admits a band rather than collapsing to a line.
    double min_band_tolerance_px{12.0};
    /// How close to the frame edge a gap may end before it is treated as clipped.
    double frame_edge_margin_px{2.0};
};

/// A measured free stretch, in image coordinates with estimated real dimensions.
struct ImageGap {
    double x1{};
    double y1{};
    double x2{};
    double y2{};
    /// Estimated, from the detected-car scale. Never a measurement.
    double length_m{};
    double depth_m{};
};

struct Scale {
    double pixels_per_metre{0.0};
    /// False when the median rests on too few cars to mean anything.
    bool confident{false};

    [[nodiscard]] bool usable() const { return pixels_per_metre > 0.0; }
};

/// Pixels per metre, from the median detected car width.
///
/// Median rather than mean, and deliberately so. One lorry, or one box that swallowed
/// two cars at once, drags a mean far enough to make every distance in the frame wrong.
/// A median ignores it.
inline Scale estimate_scale(const std::vector<ImageBox>& boxes,
                            const UncalibratedGapConfig& config = {}) {
    std::vector<double> widths;
    for (const auto& box : boxes) {
        if (box.is_car && box.width() > config.min_car_width_px) widths.push_back(box.width());
    }
    if (widths.empty()) return Scale{};
    std::sort(widths.begin(), widths.end());
    const double middle = widths[widths.size() / 2];
    if (middle <= 0.0) return Scale{};
    return Scale{middle / config.typical_car_width_m,
                 widths.size() >= config.min_cars_for_confident_scale};
}

struct Band {
    double low{};
    double high{};
    bool valid{false};
};

/// The horizontal band the parked cars occupy.
///
/// Cars at one kerb sit at roughly one depth in the image, so their vertical centres
/// cluster. Taking the median and a tolerance from the cars' own heights keeps a car
/// crossing the far side of the square out of the row being measured, which would
/// otherwise produce a gap spanning the whole picture.
inline Band kerb_band(const std::vector<ImageBox>& boxes,
                      const UncalibratedGapConfig& config = {}) {
    if (boxes.size() < 2) return Band{};
    std::vector<double> centres;
    std::vector<double> heights;
    centres.reserve(boxes.size());
    heights.reserve(boxes.size());
    for (const auto& box : boxes) {
        centres.push_back(box.centre_y());
        heights.push_back(box.height());
    }
    std::sort(centres.begin(), centres.end());
    std::sort(heights.begin(), heights.end());
    const double median = centres[centres.size() / 2];
    const double tolerance =
        std::max(config.min_band_tolerance_px, heights[heights.size() / 2]);
    return Band{median - tolerance, median + tolerance, true};
}

/// Gaps between consecutive vehicles along the kerb, left to right.
///
/// Built for parallel kerb parking, which is what a street camera usually looks at. A
/// perpendicular car park breaks the one-depth assumption the band relies on, so every
/// candidate is checked against the **full** detection list rather than the band. That
/// is the whole point of the check: the vehicles standing in the way are precisely the
/// ones the band discarded for sitting at a different depth.
/// `frame_width` is required, not optional. A gap whose right edge touches the edge of
/// the picture continues past it, so its measured length is a lower bound rather than a
/// length, and reporting it overstates the space. Passing 0 disables the check, which is
/// only ever right in a test that has no frame.
inline std::vector<ImageGap> find_free_spaces(const std::vector<ImageBox>& boxes,
                                              double pixels_per_metre, double frame_width,
                                              const UncalibratedGapConfig& config = {}) {
    std::vector<ImageGap> gaps;
    if (pixels_per_metre <= 0.0) return gaps;

    const Band band = kerb_band(boxes, config);
    if (!band.valid) return gaps;

    // Indices rather than copies, so the occlusion scan below can compare identity the
    // way the Python original compares object identity.
    std::vector<std::size_t> row;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        if (box.flanking && band.low <= box.centre_y() && box.centre_y() <= band.high) {
            row.push_back(i);
        }
    }
    if (row.size() < 2) return gaps;

    std::sort(row.begin(), row.end(),
              [&](std::size_t a, std::size_t b) { return boxes[a].x1 < boxes[b].x1; });

    for (std::size_t k = 0; k + 1 < row.size(); ++k) {
        const std::size_t li = row[k];
        const std::size_t ri = row[k + 1];
        const ImageBox& left = boxes[li];
        const ImageBox& right = boxes[ri];

        const double gap_px = right.x1 - left.x2;
        if (gap_px <= 0.0) continue;

        const double length_m = gap_px / pixels_per_metre;
        if (length_m < config.min_gap_m || length_m > config.max_gap_m) continue;

        // The box spans the gap and takes its vertical extent from the cars either side,
        // which is the depth a car parked there would occupy.
        const double y1 = std::min(left.y1, right.y1);
        const double y2 = std::max(left.y2, right.y2);

        bool blocked = false;
        for (std::size_t j = 0; j < boxes.size() && !blocked; ++j) {
            if (j == li || j == ri) continue;
            const auto& other = boxes[j];
            blocked = other.x2 > left.x2 && other.x1 < right.x1 && other.y2 > y1 &&
                      other.y1 < y2;
        }
        if (blocked) continue;

        const double depth_m = (y2 - y1) / pixels_per_metre;
        if (depth_m < config.min_depth_m) continue;

        // A gap running off the right of the frame has its far end outside the picture,
        // so its length is a lower bound and drawing it would overstate. Two pixels of
        // margin, because a box that ends exactly at the edge was almost certainly
        // clipped by it.
        if (frame_width > 0.0 && right.x1 >= frame_width - config.frame_edge_margin_px) {
            continue;
        }

        gaps.push_back(ImageGap{left.x2, y1, right.x1, y2, length_m, depth_m});
    }
    return gaps;
}

}  // namespace parkfit::vision
