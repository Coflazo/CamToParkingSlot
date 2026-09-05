// SPDX-License-Identifier: MIT
//
// Netherlands: RVV 1990, articles 23 to 25.
//
// Transcribed from the consolidated text on wetten.overheid.nl (BWBR0004825), not from a
// summary. That matters more than it sounds: most secondary sources give the bus-stop
// distance as 15 m by analogy with Germany, and the Dutch figure is 12 m.
//
// The Dutch scheme splits stopping (article 23, "stilstaan") from parking (article 24,
// "parkeren"), and parking inherits every stopping prohibition. Article 25 adds the disc
// zone, which is a condition rather than a prohibition.
//
// One rule here is worth more than all the setbacks put together. Article 24(4) says
// that where a parking facility has marked bays, you may park only inside them. That is
// the legal basis for this product's whole approach: Amsterdam publishes 210,247 surveyed
// bay polygons, and on those streets a gap between two cars is not a parking space no
// matter how long it measures. It is handled in Context rather than here, because it is
// read off the bay record rather than measured against the map.

#pragma once

#include "parkfit/legal/rulebook.hpp"

namespace parkfit::legal::nl {

/// RVV 1990 as a rule table.
inline Rulebook rulebook() {
    return Rulebook{
        "NL",
        "Reglement verkeersregels en verkeerstekens 1990 (RVV 1990)",
        {
            // --- article 23: stilstaan (stopping), which parking inherits -----------
            {AnchorKind::Junction, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "RVV 1990 art. 23(1)(a)", "stopping on a junction"},
            {AnchorKind::LevelCrossing, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "RVV 1990 art. 23(1)(a)", "stopping on a level crossing"},
            {AnchorKind::CycleLane, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "RVV 1990 art. 23(1)(b)", "stopping on or alongside a cycle lane"},
            // "op een oversteekplaats of binnen een afstand van vijf meter daarvan"
            {AnchorKind::PedestrianCrossing, Manoeuvre::Stopping, Scope::Any, m(5.0), true,
             "RVV 1990 art. 23(1)(c)", "within five metres of a pedestrian crossing"},
            {AnchorKind::Tunnel, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "RVV 1990 art. 23(1)(d)", "stopping in a tunnel"},
            // "op een afstand van minder dan 12 meter van het bord". Twelve, not fifteen.
            {AnchorKind::BusStopSign, Manoeuvre::Stopping, Scope::Any, m(12.0), true,
             "RVV 1990 art. 23(1)(e)", "within twelve metres of a bus stop sign"},
            {AnchorKind::BusLane, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "RVV 1990 art. 23(1)(f)", "stopping on the carriageway alongside a bus lane"},
            {AnchorKind::YellowLineSolid, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "RVV 1990 art. 23(1)(g)", "stopping along a solid yellow line"},

            // --- article 24: parkeren (parking) -------------------------------------
            // "bij een kruispunt op een afstand van minder dan vijf meter daarvan"
            {AnchorKind::Junction, Manoeuvre::Parking, Scope::Any, m(5.0), true,
             "RVV 1990 art. 24(1)(a)", "within five metres of a junction"},
            {AnchorKind::Driveway, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "RVV 1990 art. 24(1)(b)", "in front of an entrance or exit"},
            {AnchorKind::YellowLineBroken, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "RVV 1990 art. 24(1)(e)", "parking along a broken yellow line"},
            {AnchorKind::LoadingBay, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "RVV 1990 art. 24(1)(f)", "in a bay reserved for loading and unloading"},
            {AnchorKind::DisabledBay, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "RVV 1990 art. 24(1)(d)",
             "in a bay reserved for a vehicle category this vehicle is not in"},
        },
    };
}

}  // namespace parkfit::legal::nl
