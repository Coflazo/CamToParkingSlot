// SPDX-License-Identifier: MIT
//
// France: Code de la route, articles R417-9 to R417-13. NOT YET TRANSCRIBED.
//
// This book is deliberately empty and marked incomplete, so every French evaluation
// returns Unknown rather than Legal. That is the point of the flag: an empty table
// breaks no rules, and a country whose statute nobody has read yet would otherwise
// declare every space in Paris perfectly legal.
//
// Why it is empty: legifrance.gouv.fr sits behind Cloudflare and refuses automated
// requests, so the primary text could not be fetched the way the Dutch, German and
// Turkish ones were. The other three books in this directory were transcribed from
// official consolidated texts, and France will be too rather than from a driving-school
// summary. Getting the German bus-stop citation right depended on reading the actual
// statute (the 15 m rule is in Anlage 2, not in paragraph 12, and every summary says
// otherwise), which is the argument for not shortcutting this one.
//
// How to finish it, in order of preference:
//
//  1. **The DILA LEGI open-data dump** at echanges.dila.gouv.fr, which publishes the
//     consolidated codes as XML with no gate in front of it. Extract section
//     LEGISCTA000006177136 ("Arret ou stationnement dangereux, genant ou abusif").
//  2. **The PISTE API** (piste.gouv.fr), the official Legifrance API. Needs a free
//     account and an OAuth client, which is why it is second rather than first.
//
// What is known and still needs its exact wording and distances confirmed:
//
// * R417-9   arret ou stationnement dangereux
// * R417-10  stationnement genant: pavements, cycle paths, bus lanes, and the rule that
//            a stopped vehicle must obstruct traffic as little as possible
// * R417-11  stationnement tres genant: crossings, disabled bays, bus stops
// * R417-12  stationnement abusif (the seven-day rule)
// * R417-13  removal
//
// One French distance is widely reported and still deliberately absent here: five metres
// before a pedestrian crossing in the direction of travel. It is very likely right, and
// "very likely right" is not the standard the other three books were held to.

#pragma once

#include "parkfit/legal/rulebook.hpp"

namespace parkfit::legal::fr {

/// An intentionally empty book. `complete` is false, so evaluation returns Unknown.
inline Rulebook rulebook() {
    return Rulebook{
        "FR",
        "Code de la route (articles R417-9 a R417-13), not yet transcribed",
        {},
        /*complete=*/false,
    };
}

}  // namespace parkfit::legal::fr
