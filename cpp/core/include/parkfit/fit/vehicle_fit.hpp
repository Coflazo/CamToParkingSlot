// SPDX-License-Identifier: MIT
//
// The vehicle-fit engine: the product promise reduced to arithmetic.
//
// Design rule that governs this whole file: when a required dimension is missing from
// the source data, the answer is UNVERIFIED, never FITS. Telling a driver a garage
// works when we simply did not know its height barrier is the one failure mode that
// destroys trust in a parking app, and it is entirely avoidable.

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "parkfit/fit/vehicle.hpp"

namespace parkfit::fit {

enum class Verdict {
    Fits,        ///< Clears every checked constraint with margin to spare.
    TightFit,    ///< Physically fits, but at least one clearance is uncomfortably small.
    DoesNotFit,  ///< Violates a hard constraint.
    Unverified,  ///< A dimension needed to decide was missing from the data.
};

inline const char* to_string(Verdict v) {
    switch (v) {
        case Verdict::Fits: return "FITS";
        case Verdict::TightFit: return "TIGHT_FIT";
        case Verdict::DoesNotFit: return "DOES_NOT_FIT";
        case Verdict::Unverified: return "UNVERIFIED";
    }
    return "UNVERIFIED";
}

/// How a bay is laid out relative to the kerb. Amsterdam publishes this directly in
/// the type field of its parkeervakken dataset, which is why we can reason about it
/// at all: Langs = parallel to the kerb, Haaks = perpendicular, Visgraat = angled.
enum class BayOrientation { Parallel, Perpendicular, Angled, Unknown };

inline BayOrientation orientation_from_dutch(const std::string& s) {
    if (s == "Langs" || s == "langs" || s == "LANGS") return BayOrientation::Parallel;
    if (s == "Haaks" || s == "haaks" || s == "HAAKS") return BayOrientation::Perpendicular;
    if (s == "Visgraat" || s == "visgraat" || s == "VISGRAAT") return BayOrientation::Angled;
    return BayOrientation::Unknown;
}

/// Clearance policy. Defaults are deliberately conservative; every value can be raised
/// by the user but none can be lowered past its floor.
struct Margins {
    double vertical_cm{10.0};

    /// Clearance across a real physical aperture: a garage entrance or pillar gap.
    /// Compared against mirror width, because a wall does not yield.
    double lateral_total_cm{40.0};

    /// Clearance across a *perpendicular* painted bay. Compared against bodywork, and
    /// sized for opening a door between two neighbouring cars. Mirrors overhang the
    /// line harmlessly, so charging them here would reject most of the on-street
    /// supply in the Netherlands.
    double bay_lateral_total_cm{25.0};

    /// Clearance across a *parallel* kerbside bay, where the physics are different:
    /// there is no car beside you. One flank is the pavement -- opening the passenger
    /// door onto it is the entire point -- and the other is the traffic lane. NEN 2443
    /// specifies parallel bays at 1.80 to 2.00 m for cars 1.75 to 1.85 m wide, so the
    /// standard itself assumes a near-zero lateral margin.
    ///
    /// This is not a detail. With the perpendicular allowance applied to parallel bays,
    /// the engine rejected a Volkswagen Polo from the median Amsterdam kerb bay (1.96 m)
    /// by four centimetres, and 1846 of 2427 rejections near one destination were width.
    double parallel_lateral_total_cm{5.0};

    /// End clearance in a perpendicular bay. NEN 2443 sets Haaks depth at 5.00 m
    /// standard and 4.50 m minimum, with front overhang over a kerb or planting
    /// strip expressly permitted, so a 4.05 m car in a 4.50 m bay is ordinary
    /// parking rather than a rejection.
    double longitudinal_total_cm{30.0};

    /// End clearance in a *marked* parallel bay, which you simply park inside.
    double bay_parallel_end_cm{20.0};

    /// End clearance for an *open kerb gap* between two parked cars, which has to be
    /// reversed into. Manoeuvring room, not merely bumper room, which is why it is
    /// more than twice the marked-bay figure.
    double parallel_front_cm{50.0};
    double parallel_rear_cm{50.0};

    /// A tight result is reported when slack falls below this, so the UI can warn
    /// without excluding the option outright.
    double tight_threshold_cm{15.0};

    // Hard floors. Reducing clearance below these to manufacture more search results
    // would trade the bumper of the driver for our conversion rate.
    static constexpr double kMinVerticalCm = 5.0;
    static constexpr double kMinLateralTotalCm = 20.0;
    static constexpr double kMinBayLateralTotalCm = 10.0;
    static constexpr double kMinParallelLateralTotalCm = 0.0;
    static constexpr double kMinBayParallelEndCm = 10.0;
    static constexpr double kMinLongitudinalTotalCm = 30.0;
    static constexpr double kMinParallelEndCm = 30.0;

    [[nodiscard]] Margins clamped() const {
        Margins m = *this;
        m.vertical_cm = std::max(m.vertical_cm, kMinVerticalCm);
        m.lateral_total_cm = std::max(m.lateral_total_cm, kMinLateralTotalCm);
        m.bay_lateral_total_cm = std::max(m.bay_lateral_total_cm, kMinBayLateralTotalCm);
        m.parallel_lateral_total_cm =
            std::max(m.parallel_lateral_total_cm, kMinParallelLateralTotalCm);
        m.bay_parallel_end_cm = std::max(m.bay_parallel_end_cm, kMinBayParallelEndCm);
        m.longitudinal_total_cm = std::max(m.longitudinal_total_cm, kMinLongitudinalTotalCm);
        m.parallel_front_cm = std::max(m.parallel_front_cm, kMinParallelEndCm);
        m.parallel_rear_cm = std::max(m.parallel_rear_cm, kMinParallelEndCm);
        return m;
    }
};

/// Which constraint drove the verdict. Surfacing this lets the UI say
/// "too tall for this garage by 12 cm" instead of a bare "does not fit".
struct FitReason {
    std::string constraint;  ///< height, width, length, weight, gap_length, gap_width
    double required_cm{0.0};
    double available_cm{0.0};
    double slack_cm{0.0};
    bool binding{false};  ///< true when this constraint is the one that decided it
};

struct FitResult {
    Verdict verdict{Verdict::Unverified};
    double min_slack_cm{0.0};
    std::vector<FitReason> reasons;
    std::vector<std::string> unverified_dimensions;

    [[nodiscard]] bool acceptable() const {
        return verdict == Verdict::Fits || verdict == Verdict::TightFit;
    }
};

/// Physical limits declared by a parking facility. A value <= 0 means "not published",
/// which is materially different from "unlimited" and is treated as such.
struct FacilityLimits {
    double max_height_cm{0.0};
    double max_width_cm{0.0};
    double max_length_cm{0.0};
    double max_weight_kg{0.0};
};

namespace detail {

inline void record(std::vector<FitReason>& out, const char* name, double required,
                   double available, bool& fatal, double& min_slack, bool& any_checked) {
    FitReason r;
    r.constraint = name;
    r.required_cm = required;
    r.available_cm = available;
    r.slack_cm = available - required;
    if (r.slack_cm < 0.0) fatal = true;
    if (!any_checked || r.slack_cm < min_slack) min_slack = r.slack_cm;
    any_checked = true;
    out.push_back(r);
}

inline void finalise(FitResult& res, bool fatal, bool any_checked, double min_slack,
                     double tight_threshold) {
    res.min_slack_cm = any_checked ? min_slack : 0.0;
    if (fatal) {
        res.verdict = Verdict::DoesNotFit;
    } else if (!res.unverified_dimensions.empty() || !any_checked) {
        res.verdict = Verdict::Unverified;
    } else if (min_slack < tight_threshold) {
        res.verdict = Verdict::TightFit;
    } else {
        res.verdict = Verdict::Fits;
    }
    // Mark the binding constraint (smallest slack) so the UI can explain itself.
    if (!res.reasons.empty()) {
        auto it = std::min_element(
            res.reasons.begin(), res.reasons.end(),
            [](const FitReason& a, const FitReason& b) { return a.slack_cm < b.slack_cm; });
        it->binding = true;
    }
}

}  // namespace detail

/// Can this vehicle physically enter and occupy this facility?
inline FitResult check_facility(const Vehicle& v, const FacilityLimits& lim,
                                const Margins& raw_margins) {
    const Margins m = raw_margins.clamped();
    FitResult res;
    bool fatal = false;
    bool any = false;
    double min_slack = 0.0;

    const double h = v.effective_height_cm();
    if (lim.max_height_cm > 0.0) {
        if (h <= 0.0) {
            res.unverified_dimensions.emplace_back("vehicle_height");
        } else {
            detail::record(res.reasons, "height", h + m.vertical_cm, lim.max_height_cm, fatal,
                           min_slack, any);
        }
    } else if (h > 0.0) {
        // The facility published no barrier height. That is normal for a surface lot,
        // where it really is unlimited, but it is also what a garage with missing data
        // looks like. We cannot tell the two apart, so we decline to guess and let the
        // caller see that the check never happened.
        res.unverified_dimensions.emplace_back("facility_max_height");
    }

    const double w = v.effective_width_cm();
    if (lim.max_width_cm > 0.0 && w > 0.0) {
        detail::record(res.reasons, "width", w + m.lateral_total_cm, lim.max_width_cm, fatal,
                       min_slack, any);
    }
    if (lim.max_length_cm > 0.0 && v.effective_length_cm() > 0.0) {
        detail::record(res.reasons, "length", v.effective_length_cm() + m.longitudinal_total_cm,
                       lim.max_length_cm, fatal, min_slack, any);
    }
    if (lim.max_weight_kg > 0.0 && v.weight_kg > 0.0) {
        FitReason r;
        r.constraint = "weight";
        r.required_cm = v.weight_kg;
        r.available_cm = lim.max_weight_kg;
        r.slack_cm = lim.max_weight_kg - v.weight_kg;
        if (r.slack_cm < 0.0) fatal = true;
        res.reasons.push_back(r);
    }

    detail::finalise(res, fatal, any, min_slack, m.tight_threshold_cm);
    return res;
}

/// Can this vehicle occupy a specific marked bay of known dimensions?
///
/// Orientation matters and is not cosmetic. In a perpendicular bay the mirrors are the
/// binding constraint and modest length overhang is tolerated; in a parallel bay the
/// length binds instead, because the car has to be manoeuvred in between two neighbours.
/// That is why parallel bays get front and rear end clearances rather than a single
/// shared longitudinal margin.
inline FitResult check_bay(const Vehicle& v, double bay_length_cm, double bay_width_cm,
                           BayOrientation orientation, const Margins& raw_margins) {
    const Margins m = raw_margins.clamped();
    FitResult res;
    bool fatal = false;
    bool any = false;
    double min_slack = 0.0;

    if (!v.has_usable_dimensions()) {
        res.unverified_dimensions.emplace_back("vehicle_dimensions");
        res.verdict = Verdict::Unverified;
        return res;
    }
    if (bay_length_cm <= 0.0 || bay_width_cm <= 0.0) {
        res.unverified_dimensions.emplace_back("bay_dimensions");
        res.verdict = Verdict::Unverified;
        return res;
    }

    // Painted lines, not walls: bodywork is the binding dimension here.
    const double vw = v.effective_body_width_cm();
    const double vl = v.effective_length_cm();
    const double lat = m.bay_lateral_total_cm;

    switch (orientation) {
        case BayOrientation::Parallel: {
            // A marked bay is parked *into*, not reversed into between two cars, so it
            // needs bumper room rather than manoeuvring room. An open kerb gap is the
            // other case and is handled by check_gap with far larger clearances.
            const double ends = 2.0 * m.bay_parallel_end_cm +
                                std::max(0.0, v.extra_parallel_clearance_cm);
            detail::record(res.reasons, "length", vl + ends, bay_length_cm, fatal, min_slack, any);
            detail::record(res.reasons, "width", vw + m.parallel_lateral_total_cm, bay_width_cm,
                           fatal, min_slack, any);
            break;
        }
        case BayOrientation::Perpendicular:
        case BayOrientation::Angled: {
            detail::record(res.reasons, "width", vw + lat, bay_width_cm, fatal,
                           min_slack, any);
            detail::record(res.reasons, "length", vl + m.longitudinal_total_cm, bay_length_cm,
                           fatal, min_slack, any);
            break;
        }
        case BayOrientation::Unknown: {
            // Without orientation we apply the stricter of the two readings, so an
            // unknown layout can never come out more permissive than a known one.
            const double ends = 2.0 * m.bay_parallel_end_cm;
            detail::record(res.reasons, "length", vl + std::max(ends, m.longitudinal_total_cm),
                           bay_length_cm, fatal, min_slack, any);
            detail::record(res.reasons, "width", vw + std::max(lat, m.parallel_lateral_total_cm),
                           bay_width_cm, fatal, min_slack, any);
            res.unverified_dimensions.emplace_back("bay_orientation");
            break;
        }
    }

    detail::finalise(res, fatal, any, min_slack, m.tight_threshold_cm);
    return res;
}

/// Required kerbside length for parallel parking into an open gap between two vehicles.
///   L_required = L_vehicle + C_front + C_rear
inline double required_gap_length_cm(const Vehicle& v, const Margins& raw_margins) {
    const Margins m = raw_margins.clamped();
    return v.effective_length_cm() + m.parallel_front_cm + m.parallel_rear_cm +
           std::max(0.0, v.extra_parallel_clearance_cm);
}

/// Does a measured kerb gap accommodate this vehicle?
inline FitResult check_gap(const Vehicle& v, double gap_length_cm, double gap_width_cm,
                           const Margins& raw_margins) {
    const Margins m = raw_margins.clamped();
    FitResult res;
    bool fatal = false;
    bool any = false;
    double min_slack = 0.0;

    if (!v.has_usable_dimensions()) {
        res.unverified_dimensions.emplace_back("vehicle_dimensions");
        res.verdict = Verdict::Unverified;
        return res;
    }
    if (gap_length_cm <= 0.0) {
        res.unverified_dimensions.emplace_back("gap_length");
        res.verdict = Verdict::Unverified;
        return res;
    }

    detail::record(res.reasons, "gap_length", required_gap_length_cm(v, m), gap_length_cm, fatal,
                   min_slack, any);
    if (gap_width_cm > 0.0) {
        detail::record(res.reasons, "gap_width",
                       v.effective_body_width_cm() + m.parallel_lateral_total_cm, gap_width_cm,
                       fatal, min_slack, any);
    } else {
        res.unverified_dimensions.emplace_back("gap_width");
    }

    detail::finalise(res, fatal, any, min_slack, m.tight_threshold_cm);
    return res;
}

}  // namespace parkfit::fit
