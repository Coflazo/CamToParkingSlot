// SPDX-License-Identifier: MIT
//
// Kerb-gap estimation: from vehicle detections to a measured gap in metres.
//
// The pipeline is deliberately geometric rather than learned. A detector says "there is
// a car roughly here in the image"; everything that turns that into "there is 5.8 m of
// free kerb between these two cars" is projection and interval arithmetic, which can be
// tested against exact ground truth. Asking a network to regress gap length directly
// would be both harder to train and impossible to audit when it is wrong.
//
// Six steps:
//   1. Take each detection's *ground contact*, not its bounding box.
//   2. Project that to the ground plane through the homography.
//   3. Project the ground point onto the kerb centreline, giving a 1-D interval.
//   4. Merge overlapping occupied intervals.
//   5. Subtract occupied and prohibited intervals from the legal segment.
//   6. Report what is left, with a confidence that reflects what could not be seen.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "parkfit/vision/homography.hpp"

namespace parkfit::vision {

/// A detected object in image space.
struct Detection {
    double x1{}, y1{}, x2{}, y2{};  ///< bounding box, pixels
    double score{};
    int class_id{};
    std::string label;

    [[nodiscard]] double width() const { return x2 - x1; }
    [[nodiscard]] double height() const { return y2 - y1; }

    /// Midpoint of the bottom edge: where the object meets the ground.
    ///
    /// This is the only point of a bounding box that lies on the ground plane, and the
    /// homography maps the ground plane and nothing else. Projecting the box centre,
    /// which floats at roughly half the vehicle height, would place every car several
    /// metres further from the camera than it is, and the error grows with distance.
    [[nodiscard]] Point2d ground_contact() const { return Point2d{(x1 + x2) * 0.5, y2}; }

    /// Bottom-left and bottom-right corners, which bound the footprint along the kerb.
    [[nodiscard]] Point2d ground_left() const { return Point2d{x1, y2}; }
    [[nodiscard]] Point2d ground_right() const { return Point2d{x2, y2}; }
};

/// A closed interval along the kerb centreline, in metres from the segment start.
struct Interval {
    double start{};
    double end{};

    [[nodiscard]] double length() const { return std::max(0.0, end - start); }
    [[nodiscard]] bool overlaps(const Interval& other, double tolerance = 0.0) const {
        return start <= other.end + tolerance && other.start <= end + tolerance;
    }
};

/// The kerb a camera is watching: an ordered polyline in world metres.
struct CurbSegment {
    std::string id;
    std::vector<Point2d> centreline;  ///< RD New metres
    double usable_width_m{2.0};
    /// Stretches where parking is forbidden regardless of what is parked there:
    /// driveways, crossings, bus stops, loading zones. Measured along the centreline.
    std::vector<Interval> prohibited;

    [[nodiscard]] double length_m() const {
        double total = 0.0;
        for (std::size_t i = 1; i < centreline.size(); ++i) {
            total += std::hypot(centreline[i].x - centreline[i - 1].x,
                                centreline[i].y - centreline[i - 1].y);
        }
        return total;
    }

    /// Distance along the centreline of the closest point to `p`, plus how far `p` sits
    /// from the line. The offset is what tells us whether a detection is even on this
    /// kerb: a car in the traffic lane projects onto the line just as neatly as one
    /// parked against it, and only its offset distinguishes them.
    struct Projection {
        double along_m{0.0};
        double offset_m{0.0};
        bool valid{false};
    };

    [[nodiscard]] Projection project(const Point2d& p) const {
        Projection best;
        if (centreline.size() < 2) return best;

        double travelled = 0.0;
        double best_offset = std::numeric_limits<double>::max();
        for (std::size_t i = 1; i < centreline.size(); ++i) {
            const Point2d& a = centreline[i - 1];
            const Point2d& b = centreline[i];
            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            const double seg_len_sq = dx * dx + dy * dy;
            const double seg_len = std::sqrt(seg_len_sq);
            if (seg_len_sq < 1e-12) continue;

            double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / seg_len_sq;
            t = std::clamp(t, 0.0, 1.0);
            const double px = a.x + t * dx;
            const double py = a.y + t * dy;
            const double offset = std::hypot(p.x - px, p.y - py);

            if (offset < best_offset) {
                best_offset = offset;
                best.along_m = travelled + t * seg_len;
                best.offset_m = offset;
                best.valid = true;
            }
            travelled += seg_len;
        }
        return best;
    }
};

/// A measured free stretch of kerb.
struct Gap {
    double start_m{};
    double end_m{};
    double length_m{};
    double width_m{};
    double confidence{};
    bool occluded_start{false};
    bool occluded_end{false};

    [[nodiscard]] double length_cm() const { return length_m * 100.0; }
};

struct GapConfig {
    /// How far from the centreline a detection may sit and still count as parked here.
    /// Wide enough for a car parked untidily, narrow enough to exclude the traffic lane.
    double max_offset_m{3.2};
    /// Vehicles closer than this along the kerb are treated as one blockage. Below it,
    /// the "gap" is a bumper interval, not a parking space.
    double merge_tolerance_m{0.35};
    /// Shorter than this is not worth reporting to anyone.
    double min_gap_m{3.0};
    /// A gap touching the end of what the camera can see may continue past it, so its
    /// measured length is a lower bound. Reported, and penalised, rather than trusted.
    double edge_margin_m{0.5};
    /// Beyond this distance from the camera, a pixel of vertical error becomes more
    /// than a few centimetres of ground error, and the measurement stops being useful.
    double max_range_m{45.0};
};

/// Merge overlapping or near-touching intervals. Input need not be sorted.
inline std::vector<Interval> merge_intervals(std::vector<Interval> intervals,
                                             double tolerance = 0.0) {
    if (intervals.empty()) return intervals;
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& a, const Interval& b) { return a.start < b.start; });

    std::vector<Interval> merged;
    merged.push_back(intervals.front());
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        Interval& last = merged.back();
        if (intervals[i].start <= last.end + tolerance) {
            last.end = std::max(last.end, intervals[i].end);
        } else {
            merged.push_back(intervals[i]);
        }
    }
    return merged;
}

/// Subtract a set of blocked intervals from `[0, total]`.
inline std::vector<Interval> free_intervals(double total, std::vector<Interval> blocked,
                                            double tolerance = 0.0) {
    std::vector<Interval> free;
    if (total <= 0.0) return free;
    blocked = merge_intervals(std::move(blocked), tolerance);

    double cursor = 0.0;
    for (const auto& block : blocked) {
        if (block.start > cursor) {
            free.push_back(Interval{cursor, std::min(block.start, total)});
        }
        cursor = std::max(cursor, block.end);
        if (cursor >= total) break;
    }
    if (cursor < total) free.push_back(Interval{cursor, total});

    free.erase(std::remove_if(free.begin(), free.end(),
                              [](const Interval& i) { return i.length() <= 1e-9; }),
               free.end());
    return free;
}

/// Projects detections onto a kerb segment and measures what is left free.
class CurbGapEstimator {
  public:
    explicit CurbGapEstimator(GapConfig config = {}) : config_(config) {}

    struct Result {
        std::vector<Gap> gaps;
        std::vector<Interval> occupied;
        int projected_vehicles{0};
        int rejected_off_kerb{0};
        int rejected_out_of_range{0};
        bool usable{false};
        std::string reason;
    };

    /// `camera_world` is the camera position in world metres, used for range checks.
    Result estimate(const CurbSegment& segment, const Homography& homography,
                    const std::vector<Detection>& detections,
                    const Point2d& camera_world) const {
        Result result;
        const double total = segment.length_m();
        if (segment.centreline.size() < 2 || total <= 0.0) {
            result.reason = "kerb segment has no usable geometry";
            return result;
        }
        if (!homography.valid()) {
            result.reason = "calibration is not valid";
            return result;
        }

        std::vector<Interval> blocked = segment.prohibited;

        for (const auto& detection : detections) {
            // Both bottom corners are projected, so the occupied interval reflects the
            // vehicle's true extent along the kerb rather than a point at its centre.
            const Point2d left = homography.apply(detection.ground_left());
            const Point2d right = homography.apply(detection.ground_right());
            const Point2d centre = homography.apply(detection.ground_contact());

            const double range = std::hypot(centre.x - camera_world.x, centre.y - camera_world.y);
            if (range > config_.max_range_m) {
                ++result.rejected_out_of_range;
                continue;
            }

            const auto pl = segment.project(left);
            const auto pr = segment.project(right);
            const auto pc = segment.project(centre);
            if (!pc.valid) continue;

            if (pc.offset_m > config_.max_offset_m) {
                // Projects onto the kerb line but sits too far from it: this is traffic,
                // not parking. Counting it would invent a blockage that is not there.
                ++result.rejected_off_kerb;
                continue;
            }

            double start = pl.valid ? pl.along_m : pc.along_m;
            double end = pr.valid ? pr.along_m : pc.along_m;
            if (start > end) std::swap(start, end);
            if (end - start < 1.0) {
                // A near-degenerate footprint means the vehicle is almost end-on to the
                // camera and its width along the kerb cannot be read from the box. Assume
                // a conservative car length rather than a sliver.
                const double centre_along = (start + end) * 0.5;
                start = centre_along - 2.1;
                end = centre_along + 2.1;
            }

            blocked.push_back(Interval{std::max(0.0, start), std::min(total, end)});
            ++result.projected_vehicles;
        }

        result.occupied = merge_intervals(blocked, config_.merge_tolerance_m);
        const auto free = free_intervals(total, result.occupied, config_.merge_tolerance_m);

        for (const auto& interval : free) {
            if (interval.length() < config_.min_gap_m) continue;

            Gap gap;
            gap.start_m = interval.start;
            gap.end_m = interval.end;
            gap.length_m = interval.length();
            gap.width_m = segment.usable_width_m;
            gap.occluded_start = interval.start <= config_.edge_margin_m;
            gap.occluded_end = interval.end >= total - config_.edge_margin_m;
            gap.confidence = confidence_for(gap, result);
            result.gaps.push_back(gap);
        }

        std::sort(result.gaps.begin(), result.gaps.end(),
                  [](const Gap& a, const Gap& b) { return a.length_m > b.length_m; });
        result.usable = true;
        return result;
    }

    [[nodiscard]] const GapConfig& config() const { return config_; }

  private:
    [[nodiscard]] double confidence_for(const Gap& gap, const Result& result) const {
        double confidence = 0.92;

        // A gap running off the edge of the visible segment might continue, or might be
        // blocked by something just out of frame. Either way the measurement is a lower
        // bound, and a lower bound deserves less confidence than a measurement.
        if (gap.occluded_start) confidence -= 0.18;
        if (gap.occluded_end) confidence -= 0.18;

        // Detections thrown away for being out of range mean part of the scene was not
        // assessed, so something could be parked in the gap without having been seen.
        if (result.rejected_out_of_range > 0) confidence -= 0.10;

        // A gap far longer than a car is more robust to a small projection error than
        // one that only just fits, so tight gaps are reported less confidently.
        if (gap.length_m < 5.5) confidence -= 0.08;

        return std::clamp(confidence, 0.05, 0.95);
    }

    GapConfig config_;
};

}  // namespace parkfit::vision
