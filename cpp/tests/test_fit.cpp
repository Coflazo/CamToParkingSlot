// SPDX-License-Identifier: MIT
//
// Vehicle-fit tests.
//
// The most important cases here are the negative ones. A parking app that occasionally
// fails to surface a valid garage is mildly annoying; one that sends a 2.05 m van into
// a 2.00 m barrier causes real damage. Every test that asserts DoesNotFit or Unverified
// is guarding that asymmetry.

#include "test_framework.hpp"

#include "parkfit/fit/vehicle_fit.hpp"

using namespace parkfit::fit;

namespace {

/// A Volkswagen Polo as returned by the live RDW register for plate XT-994-N,
/// with the height the owner confirmed manually (RDW does not publish it).
Vehicle polo() {
    Vehicle v;
    v.id = "veh_polo";
    v.nickname = "Polo";
    v.length_cm = 405.3;
    v.body_width_cm = 175.1;
    v.width_with_mirrors_cm = 194.0;
    v.height_cm = 145.1;
    v.height_with_accessories_cm = 145.1;
    v.weight_kg = 1105.0;
    v.provenance = {true, true, true, true};
    return v;
}

/// A tall van with a roof rack: the case that height filtering exists for.
Vehicle tall_van() {
    Vehicle v;
    v.id = "veh_van";
    v.nickname = "Transporter";
    v.length_cm = 590.0;
    v.body_width_cm = 190.4;
    v.width_with_mirrors_cm = 246.0;
    v.height_cm = 199.0;
    v.height_with_accessories_cm = 232.0;  // roof rack adds 33 cm
    v.weight_kg = 2000.0;
    v.provenance = {true, true, true, true};
    return v;
}

}  // namespace

TEST_CASE("facility: a normal car clears a 2.00 m barrier") {
    FacilityLimits lim;
    lim.max_height_cm = 200.0;  // RDW maximumvehicleheight, in cm
    const FitResult r = check_facility(polo(), lim, Margins{});
    CHECK(r.verdict == Verdict::Fits);
    CHECK(r.acceptable());
    CHECK_NEAR(r.min_slack_cm, 200.0 - (145.1 + 10.0), 1e-9);
}

TEST_CASE("facility: a roof rack is what fails the barrier, not the roof") {
    FacilityLimits lim;
    lim.max_height_cm = 230.0;
    Vehicle v = tall_van();

    // The bare van at 199 cm clears a 230 cm barrier with 21 cm to spare. With the rack
    // it stands 232 cm and must not. Reading body height instead of accessory height
    // here is the classic bug this case guards against.
    const FitResult with_rack = check_facility(v, lim, Margins{});
    CHECK(with_rack.verdict == Verdict::DoesNotFit);

    v.height_with_accessories_cm = 0.0;  // rack removed, fall back to body height
    const FitResult without = check_facility(v, lim, Margins{});
    CHECK(without.verdict == Verdict::Fits);
}

TEST_CASE("facility: the binding constraint is identified for the UI") {
    FacilityLimits lim;
    lim.max_height_cm = 165.0;   // 145.1 + 10 = 155.1 required, 9.9 cm of slack
    lim.max_width_cm = 300.0;    // 194.0 + 40 = 234.0 required, 66 cm of slack
    const FitResult r = check_facility(polo(), lim, Margins{});
    CHECK(r.verdict == Verdict::TightFit);
    bool found = false;
    for (const auto& reason : r.reasons) {
        if (reason.binding) {
            CHECK(reason.constraint == "height");
            found = true;
        }
    }
    CHECK(found);
}

TEST_CASE("facility: an unpublished barrier height yields UNVERIFIED, never FITS") {
    FacilityLimits lim;  // max_height_cm left at 0, meaning "not published"
    const FitResult r = check_facility(tall_van(), lim, Margins{});
    CHECK(r.verdict == Verdict::Unverified);
    CHECK(!r.unverified_dimensions.empty());
}

TEST_CASE("facility: an unknown vehicle height also yields UNVERIFIED") {
    Vehicle v = polo();
    v.height_cm = 0.0;
    v.height_with_accessories_cm = 0.0;
    FacilityLimits lim;
    lim.max_height_cm = 200.0;
    const FitResult r = check_facility(v, lim, Margins{});
    CHECK(r.verdict == Verdict::Unverified);
}

TEST_CASE("facility: weight limits are enforced") {
    FacilityLimits lim;
    lim.max_height_cm = 250.0;
    lim.max_weight_kg = 1500.0;
    const FitResult r = check_facility(tall_van(), lim, Margins{});
    CHECK(r.verdict == Verdict::DoesNotFit);
}

TEST_CASE("bay: a parallel bay is judged on length with end clearances") {
    // A generous Dutch Langs bay: 5.8 m x 2.3 m.
    // Length 405.3 + 50 + 50 = 505.3 against 580, width 175.1 + 25 = 200.1 against 230.
    const FitResult r = check_bay(polo(), 580.0, 230.0, BayOrientation::Parallel, Margins{});
    CHECK(r.verdict == Verdict::Fits);
    CHECK_NEAR(required_gap_length_cm(polo(), Margins{}), 505.3, 1e-9);
}

TEST_CASE("bay: a narrow bay leaves too little room to open a door") {
    // The same car in a 2.10 m bay has only 9.9 cm of door clearance. It physically
    // fits and should still be offered, but the driver deserves the warning.
    const FitResult r = check_bay(polo(), 580.0, 210.0, BayOrientation::Parallel, Margins{});
    CHECK(r.verdict == Verdict::TightFit);
    CHECK(r.acceptable());
    CHECK_NEAR(r.min_slack_cm, 9.9, 1e-9);
}

TEST_CASE("bay: the same bay rejects a long van") {
    const FitResult r = check_bay(tall_van(), 580.0, 230.0, BayOrientation::Parallel, Margins{});
    CHECK(r.verdict == Verdict::DoesNotFit);  // 590 + 100 = 690 needed, 580 available
}

TEST_CASE("bay: orientation changes the verdict for the same rectangle") {
    // A 4.8 m x 2.4 m bay. Read as perpendicular the Polo fits (405.3 + 60 = 465.3);
    // read as parallel it cannot, because swinging in needs 405.3 + 50 + 50 = 505.3.
    const FitResult perp =
        check_bay(polo(), 480.0, 240.0, BayOrientation::Perpendicular, Margins{});
    const FitResult para = check_bay(polo(), 480.0, 240.0, BayOrientation::Parallel, Margins{});
    CHECK(perp.acceptable());
    CHECK(para.verdict == Verdict::DoesNotFit);
}

TEST_CASE("bay: unknown orientation is never more permissive than a known one") {
    // 4.8 m bay: perpendicular would pass, parallel would not. With the orientation
    // unknown we must land on the stricter reading and reject.
    const FitResult unk = check_bay(polo(), 480.0, 240.0, BayOrientation::Unknown, Margins{});
    CHECK(!unk.acceptable());
    // A violated hard constraint is definitive, so it outranks "we were unsure about
    // the layout" -- an unknown orientation must never soften a physical rejection.
    CHECK(unk.verdict == Verdict::DoesNotFit);

    bool length_negative = false;
    for (const auto& reason : unk.reasons) {
        if (reason.constraint == "length" && reason.slack_cm < 0.0) length_negative = true;
    }
    CHECK(length_negative);
}

TEST_CASE("bay: unknown orientation that clears both readings is merely unverified") {
    // 6.0 m x 2.4 m clears the parallel reading (505.3) and the perpendicular one
    // (465.3) alike. Nothing is violated, so the only reservation left is that we do
    // not know how the bay is laid out.
    const FitResult unk = check_bay(polo(), 600.0, 240.0, BayOrientation::Unknown, Margins{});
    CHECK(unk.verdict == Verdict::Unverified);
    CHECK(unk.unverified_dimensions.size() == 1);
    CHECK(unk.unverified_dimensions[0] == "bay_orientation");

    // The recorded length requirement used the stricter of the two margins.
    for (const auto& reason : unk.reasons) {
        if (reason.constraint == "length") CHECK_NEAR(reason.required_cm, 505.3, 1e-9);
    }
}

TEST_CASE("width: mirrors govern apertures, bodywork governs painted bays") {
    Vehicle v = polo();

    // A 2.20 m physical aperture. Mirrors at 194.0 plus the 40 cm aperture allowance
    // need 234.0, so this must fail even though the bodywork is only 175.1 wide.
    FacilityLimits narrow_gate;
    narrow_gate.max_height_cm = 250.0;
    narrow_gate.max_width_cm = 220.0;
    CHECK(check_facility(v, narrow_gate, Margins{}).verdict == Verdict::DoesNotFit);

    // The same 2.20 m as a painted bay is routine parking: 175.1 + 25 = 200.1.
    // Mirrors overhang the line into the neighbouring airspace, which is why every
    // Dutch city centre is full of cars parked exactly like this.
    const FitResult bay = check_bay(v, 600.0, 220.0, BayOrientation::Perpendicular, Margins{});
    CHECK(bay.acceptable());
}

TEST_CASE("width: an unknown bodywork width is derived from the mirror span") {
    Vehicle v = polo();
    v.body_width_cm = 0.0;
    // 194.0 - 36 = 158.0 inferred.
    CHECK_NEAR(v.effective_body_width_cm(), 158.0, 1e-9);

    v.width_with_mirrors_cm = 0.0;
    v.body_width_cm = 175.1;
    // Symmetrically, an unknown mirror span is inferred as bodywork + 36.
    CHECK_NEAR(v.effective_width_cm(), 211.1, 1e-9);
}

TEST_CASE("margins: clearances cannot be lowered past their safety floors") {
    Margins m;
    m.parallel_front_cm = 0.0;
    m.parallel_rear_cm = -50.0;
    m.vertical_cm = 0.0;
    m.lateral_total_cm = 1.0;
    const Margins c = m.clamped();
    CHECK_NEAR(c.parallel_front_cm, Margins::kMinParallelEndCm, 1e-9);
    CHECK_NEAR(c.parallel_rear_cm, Margins::kMinParallelEndCm, 1e-9);
    CHECK_NEAR(c.vertical_cm, Margins::kMinVerticalCm, 1e-9);
    CHECK_NEAR(c.lateral_total_cm, Margins::kMinLateralTotalCm, 1e-9);

    // Even with every margin zeroed, a van cannot be squeezed into a compact bay.
    const FitResult r = check_bay(tall_van(), 560.0, 200.0, BayOrientation::Parallel, m);
    CHECK(r.verdict == Verdict::DoesNotFit);
}

TEST_CASE("margins: a nervous driver can ask for more room and gets fewer results") {
    Vehicle v = polo();
    // 5.20 m bay: 405.3 + 50 + 50 = 505.3 required, so 14.7 cm of slack -- a tight fit.
    const FitResult relaxed = check_bay(v, 520.0, 220.0, BayOrientation::Parallel, Margins{});
    CHECK(relaxed.acceptable());

    v.extra_parallel_clearance_cm = 60.0;  // wants 60 cm more than the default
    const FitResult cautious = check_bay(v, 520.0, 220.0, BayOrientation::Parallel, Margins{});
    CHECK(cautious.verdict == Verdict::DoesNotFit);
}

TEST_CASE("gap: a measured kerb gap is checked against the required length") {
    // 5.82 m gap, the worked example from the availability contract.
    const FitResult r = check_gap(polo(), 582.0, 219.0, Margins{});
    CHECK(r.verdict == Verdict::Fits);

    const FitResult tight = check_gap(polo(), 512.0, 219.0, Margins{});
    CHECK(tight.verdict == Verdict::TightFit);  // 6.7 cm of slack

    const FitResult no = check_gap(polo(), 480.0, 219.0, Margins{});
    CHECK(no.verdict == Verdict::DoesNotFit);
}

TEST_CASE("gap: an unmeasurable gap width is reported rather than assumed") {
    const FitResult r = check_gap(polo(), 582.0, 0.0, Margins{});
    CHECK(r.verdict == Verdict::Unverified);
}

TEST_CASE("orientation parsing accepts the Dutch source values") {
    CHECK(orientation_from_dutch("Langs") == BayOrientation::Parallel);
    CHECK(orientation_from_dutch("Haaks") == BayOrientation::Perpendicular);
    CHECK(orientation_from_dutch("Visgraat") == BayOrientation::Angled);
    CHECK(orientation_from_dutch("") == BayOrientation::Unknown);
    CHECK(orientation_from_dutch("something else") == BayOrientation::Unknown);
}

PF_TEST_MAIN()
