// SPDX-License-Identifier: MIT
//
// Clearance policy per country.
//
// The obvious way to write this file is to invent four sets of numbers, one per national
// parking standard, and call it internationalisation. That would be wrong twice over, and
// the second reason is the interesting one.
//
// **First, I have not read three of the four standards.** NEN 2443 is quoted in
// vehicle_fit.hpp with its actual figures because those were checked; EAR/RASt-06,
// NF P 91-100 and the Otopark Yonetmeligi are names I know rather than documents I have
// transcribed. Encoding numbers from memory is exactly what this project refuses to do
// with road law, and a bay dimension is no different from a setback distance.
//
// **Second, and more usefully: the margins are not national.** Look at what Margins
// actually holds. It is not how wide a country builds its bays; it is how much room a
// driver needs beyond their own bodywork to open a door, swing a mirror past a pillar, or
// reverse into a kerb gap. A door needs the same room in Lyon as in Leiden. What varies
// between countries is the *supply*, the bays themselves, and this product reads those
// from surveyed polygons and published dimensions rather than from a standard.
//
// So the national standard enters in one place only: as the evidence that a margin is
// small enough to be safe. NEN 2443 specifies Dutch parallel bays at 1.80 to 2.00 m for
// cars 1.75 to 1.85 m wide, which is what justifies a 5 cm lateral allowance there. A
// country that builds *wider* parallel bays does not need a different margin; it produces
// more slack against the same one. That direction is safe. A country that builds narrower
// ones would need a smaller margin, and none of the four does.
//
// The result is one physical policy, applied everywhere, with each country recording
// which standard was actually consulted. `verified` is false where none was, and that is
// a statement about this file rather than about the country.

#pragma once

#include <string>

#include "parkfit/fit/vehicle_fit.hpp"

namespace parkfit::fit {

/// The clearance rules for one country, and the provenance of those rules.
struct ClearancePolicy {
    /// ISO 3166-1 alpha-2.
    const char* country{""};
    /// The national parking-design standard, named whether or not it was transcribed.
    const char* standard{""};
    /// True only where the standard's own figures were read and used. False means the
    /// physical margins below are in force and the standard has not been consulted, which
    /// is a gap in this file rather than a claim about the country.
    bool verified{false};
    /// What the standard contributes, or what is missing. Shown in the evidence trail.
    const char* note{""};
    Margins margins{};
};

/// The physical baseline, in force everywhere.
///
/// Derived from what a driver's body and doors need, and checked against NEN 2443 where
/// that standard has something to say. Every figure here has its reasoning in
/// `vehicle_fit.hpp`, including the one that matters most: the parallel lateral allowance
/// is 5 cm because a kerbside bay has a pavement on one side and a traffic lane on the
/// other, so there is no neighbouring car to open a door against.
inline Margins physical_margins() { return Margins{}; }

/// The policy for an ISO 3166-1 alpha-2 code.
///
/// An unknown country gets the same physical margins rather than a refusal, because
/// unlike road law these are not jurisdiction-specific: a car that physically fits a bay
/// in an uncatalogued country still fits it. What the caller loses is the confirmation
/// that a national standard agrees, which `verified` reports.
inline ClearancePolicy clearance_for(const std::string& country) {
    if (country == "NL") {
        return ClearancePolicy{
            "NL", "NEN 2443", /*verified=*/true,
            "Parallel bays 1.80-2.00 m for cars 1.75-1.85 m wide, and Haaks depth 5.00 m "
            "standard with 4.50 m minimum and front overhang permitted. Both figures were "
            "read and are what the parallel and longitudinal allowances rest on.",
            physical_margins()};
    }
    if (country == "DE") {
        return ClearancePolicy{
            "DE", "EAR 05 / RASt 06", /*verified=*/false,
            "Standard not transcribed. The physical margins apply. German Senkrechtparken "
            "bays are not narrower than Dutch ones, so a margin sized against NEN 2443 is "
            "conservative here rather than optimistic.",
            physical_margins()};
    }
    if (country == "FR") {
        return ClearancePolicy{
            "FR", "NF P 91-100 / NF P 91-120", /*verified=*/false,
            "Standard not transcribed. The physical margins apply. French off-street "
            "height limits are published per site and read from the data rather than "
            "assumed from the standard; the median is 1.90 m.",
            physical_margins()};
    }
    if (country == "TR") {
        return ClearancePolicy{
            "TR", "Otopark Yonetmeligi", /*verified=*/false,
            "Standard not transcribed. The physical margins apply. Turkish parallel bays "
            "are not narrower than Dutch ones, so the same allowance is conservative.",
            physical_margins()};
    }
    return ClearancePolicy{"??", "no national standard consulted", /*verified=*/false,
                           "The physical margins apply. A car that fits a bay fits it "
                           "wherever the bay is, which is why an uncatalogued country "
                           "still gets an answer here, unlike in the legal rulebook.",
                           physical_margins()};
}

}  // namespace parkfit::fit
