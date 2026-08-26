// SPDX-License-Identifier: MIT
//
// Vehicle profile. All dimensions in centimetres so that integer-ish source data
// (RDW publishes lengths and widths in cm) survives without rounding drift.

#pragma once

#include <algorithm>
#include <string>

namespace parkfit::fit {

/// Which physical dimensions we actually trust for this vehicle.
///
/// This exists because RDW's registered-vehicle dataset is genuinely incomplete:
/// `lengte` and `breedte` are usually present, height almost never is. Rather than
/// silently substituting a guess, we track provenance per dimension and let the
/// fit engine downgrade its verdict to "unverified" instead of inventing confidence.
struct DimensionProvenance {
    bool length_confirmed{false};
    bool width_confirmed{false};
    bool height_confirmed{false};
    bool weight_confirmed{false};
};

struct Vehicle {
    std::string id;
    std::string nickname;

    double length_cm{0.0};
    double body_width_cm{0.0};
    /// Mirrors stick out 15-25 cm per side on a typical car and are what actually
    /// scrapes a garage pillar, so this (not body width) governs lateral fit.
    double width_with_mirrors_cm{0.0};
    double height_cm{0.0};
    /// Roof box, roof rack, bicycle carrier or aerial. This is the number that gets
    /// compared against a garage height barrier.
    double height_with_accessories_cm{0.0};
    double weight_kg{0.0};

    bool is_ev{false};
    bool has_trailer{false};
    bool has_roof_box{false};

    /// Extra room the driver personally wants when parallel parking, on top of the
    /// safety floor. Users who dislike tight parking raise this; it can never lower
    /// the hard minimum.
    double extra_parallel_clearance_cm{0.0};

    DimensionProvenance provenance{};

    /// Effective height for barrier checks: accessories if known, else body height.
    [[nodiscard]] double effective_height_cm() const {
        return height_with_accessories_cm > 0.0 ? height_with_accessories_cm : height_cm;
    }

    /// Width across the mirrors. This is the number that governs a real physical
    /// aperture: a garage entrance, a ramp, the gap between two pillars.
    [[nodiscard]] double effective_width_cm() const {
        if (width_with_mirrors_cm > 0.0) return width_with_mirrors_cm;
        if (body_width_cm > 0.0) return body_width_cm + kMirrorAllowanceCm;
        return 0.0;
    }

    /// Width across the bodywork, excluding mirrors.
    ///
    /// This governs a painted bay or a kerbside strip, where the boundary is a line
    /// rather than a wall. Mirrors overhang that line into the airspace above the
    /// neighbouring bay, which is precisely why a 2.10 m Amsterdam bay comfortably
    /// holds a car with a 1.94 m mirror span. Judging painted bays on mirror width
    /// would reject most of the on-street supply in the country.
    [[nodiscard]] double effective_body_width_cm() const {
        if (body_width_cm > 0.0) return body_width_cm;
        if (width_with_mirrors_cm > 0.0) {
            return std::max(0.0, width_with_mirrors_cm - kMirrorAllowanceCm);
        }
        return 0.0;
    }

    /// Combined protrusion of both wing mirrors on a typical passenger car.
    static constexpr double kMirrorAllowanceCm = 36.0;

    /// Total road length occupied, including anything towed.
    [[nodiscard]] double effective_length_cm() const { return length_cm; }

    [[nodiscard]] bool has_usable_dimensions() const {
        return length_cm > 0.0 && effective_width_cm() > 0.0;
    }
};

}  // namespace parkfit::fit
