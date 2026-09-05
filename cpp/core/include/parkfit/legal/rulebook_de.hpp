// SPDX-License-Identifier: MIT
//
// Germany: StVO paragraph 12, plus Zeichen 224 from Anlage 2.
//
// Transcribed from gesetze-im-internet.de/stvo_2013/__12.html.
//
// Two things here came out differently from the obvious assumption, and both would have
// been wrong if taken from a summary.
//
// **The junction setback is 5 m, or 8 m where a structurally separate cycle path runs on
// the right in the direction of travel.** StVO 12(3) Nr. 1 says so in one sentence, and
// almost every popular summary drops the second half. It is not a footnote either: it is
// the difference between offering and refusing a space on any German street with a
// separated Radweg, which is most residential streets in a modern German city.
//
// **The 15 m bus-stop rule is not in paragraph 12.** It belongs to Zeichen 224
// (Haltestelle) in Anlage 2 to paragraph 41, and citing 12 for it would put a wrong
// article in front of a user. This is exactly why every rule in this file carries the
// citation it was transcribed with.
//
// Deliberate gaps, recorded rather than guessed:
//
// * **Pedestrian crossings.** German crossing rules live in paragraph 26 and Zeichen 293
//   rather than in paragraph 12, and the distance is not stated in the text fetched here.
//   No entry is better than an invented one, so there is none, and a German result near a
//   crossing is therefore less complete than a Dutch one. Fill it from the primary text
//   for paragraph 26 and Anlage 2 before relying on it.
// * **Paragraph 12(1) Nr. 1 and Nr. 2** ("enge und unuebersichtliche Strassenstellen",
//   "scharfe Kurven") are genuinely qualitative. They have no distance to encode and are
//   not modelled; a narrow or blind spot is something this system cannot see.
// * **Paragraph 12(3a)** restricts night and Sunday parking for vehicles over 7.5 t in
//   residential, recreation, spa and clinic areas. That is a vehicle-mass plus land-use
//   plus time-of-day rule rather than a setback, so it belongs with the restriction
//   evaluator, not here.

#pragma once

#include "parkfit/legal/rulebook.hpp"

namespace parkfit::legal::de {

/// StVO as a rule table.
inline Rulebook rulebook() {
    return Rulebook{
        "DE",
        "Strassenverkehrs-Ordnung (StVO)",
        {
            // --- paragraph 12(1): Halten unzulaessig, inherited by parking -----------
            {AnchorKind::LevelCrossing, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "StVO 12(1) Nr. 4", "stopping on a level crossing"},
            {AnchorKind::EmergencyAccess, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "StVO 12(1) Nr. 5",
             "in front of or inside an officially marked fire brigade access"},

            // --- paragraph 12(3): Parken unzulaessig --------------------------------
            // "vor und hinter Kreuzungen und Einmuendungen bis zu je 5 m von den
            // Schnittpunkten der Fahrbahnkanten"
            {AnchorKind::Junction, Manoeuvre::Parking, Scope::Any, m(5.0), true,
             "StVO 12(3) Nr. 1", "within five metres of a junction"},
            // "soweit in Fahrtrichtung rechts neben der Fahrbahn ein Radweg baulich
            // angelegt ist, vor Kreuzungen und Einmuendungen bis zu je 8 m"
            {AnchorKind::JunctionWithCyclePath, Manoeuvre::Parking, Scope::Any, m(8.0), true,
             "StVO 12(3) Nr. 1",
             "within eight metres of a junction with a separate cycle path on the right"},
            {AnchorKind::Driveway, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "StVO 12(3) Nr. 3", "in front of a property entrance or exit"},
            // "vor Bordsteinabsenkungen": a dropped kerb is a wheelchair or pram crossing
            // point, and blocking one is the German rule most often broken by visitors.
            {AnchorKind::PublicEntrance, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "StVO 12(3) Nr. 5", "in front of a dropped kerb"},

            // --- Anlage 2 to paragraph 41: Zeichen 224 (Haltestelle) ----------------
            {AnchorKind::BusStopSign, Manoeuvre::Parking, Scope::Any, m(15.0), true,
             "StVO Anlage 2 zu 41 Abs. 1, Zeichen 224",
             "within fifteen metres of a bus or tram stop sign"},
            {AnchorKind::TramStop, Manoeuvre::Parking, Scope::Any, m(15.0), true,
             "StVO Anlage 2 zu 41 Abs. 1, Zeichen 224",
             "within fifteen metres of a tram stop sign"},
        },
    };
}

}  // namespace parkfit::legal::de
