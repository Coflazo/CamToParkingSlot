// SPDX-License-Identifier: MIT
//
// Turkiye: Karayollari Trafik Kanunu No. 2918, articles 60 and 61.
//
// Transcribed from the official consolidated PDF at mevzuat.gov.tr (1.5.2918.pdf).
//
// The article numbers are 60 and 61, not 61 and 62. Secondary sources disagree about
// this often enough that it is worth stating: article 60 is "Duraklamanin yasak oldugu
// yerler" (where stopping is prohibited) and article 61 is "Park etmenin yasak oldugu
// yerler ve haller" (where and when parking is prohibited).
//
// This is the strictest of the four books, and that is useful rather than inconvenient:
// it exercises parts of the rule model that the Dutch and German statutes never touch.
//
// * **Article 61(a) states the inheritance outright**, prohibiting parking "in places
//   where stopping is prohibited". The general rule in rulebook.hpp that a parking query
//   also evaluates stopping rules is Turkish law written down, not an approximation.
// * **Built-up scope changes the number by a factor of twenty.** Five metres from a
//   junction inside a built-up area, one hundred metres outside it. No other book here
//   needs Scope, and without it Turkey would be wrong on every rural road.
// * **Fire hydrants and the ten-metre bridge rule have no Dutch or German equivalent.**
//   A gap that is perfectly legal in Amsterdam can be a fine in Istanbul.
//
// Not modelled, and recorded rather than guessed:
//
// * Article 60(b) (left lane), 60(e) (blind crests and bends) and 60(g) (double parking)
//   are either qualitative or about the vehicle's own manoeuvre rather than a place.
// * Article 61(f) (the middle carriageway of a road with three or more) needs a road
//   cross-section this system does not model.
// * Article 61(i) and 61(l) are time and duration rules, which belong with the
//   restriction evaluator rather than in a setback table.

#pragma once

#include "parkfit/legal/rulebook.hpp"

namespace parkfit::legal::tr {

/// KTK 2918 as a rule table.
inline Rulebook rulebook() {
    return Rulebook{
        "TR",
        "Karayollari Trafik Kanunu No. 2918",
        {
            // --- article 60: duraklama (stopping), which parking inherits ------------
            // "Yaya ve okul gecitleri ile diger gecitlerde"
            {AnchorKind::PedestrianCrossing, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "KTK 2918 md. 60(c)", "stopping on a pedestrian or school crossing"},
            // "Kavsaklar, tuneller, rampalar, koprüler ve baglanti yollarinda ve buralara,
            // yerlesim birimleri icinde bes metre ve yerlesim birimleri disinda yuz metre
            // mesafede"
            {AnchorKind::Junction, Manoeuvre::Stopping, Scope::BuiltUp, m(5.0), true,
             "KTK 2918 md. 60(d)", "within five metres of a junction in a built-up area"},
            {AnchorKind::Junction, Manoeuvre::Stopping, Scope::Outside, m(100.0), true,
             "KTK 2918 md. 60(d)",
             "within one hundred metres of a junction outside a built-up area"},
            {AnchorKind::Tunnel, Manoeuvre::Stopping, Scope::BuiltUp, m(5.0), true,
             "KTK 2918 md. 60(d)", "within five metres of a tunnel in a built-up area"},
            {AnchorKind::Tunnel, Manoeuvre::Stopping, Scope::Outside, m(100.0), true,
             "KTK 2918 md. 60(d)",
             "within one hundred metres of a tunnel outside a built-up area"},
            {AnchorKind::Bridge, Manoeuvre::Stopping, Scope::BuiltUp, m(5.0), true,
             "KTK 2918 md. 60(d)", "within five metres of a bridge in a built-up area"},
            {AnchorKind::Bridge, Manoeuvre::Stopping, Scope::Outside, m(100.0), true,
             "KTK 2918 md. 60(d)",
             "within one hundred metres of a bridge outside a built-up area"},
            // "Otobus, tramvay ve taksi duraklarinda"
            {AnchorKind::BusStopSign, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "KTK 2918 md. 60(f)", "stopping at a bus, tram or taxi stop"},
            {AnchorKind::TramStop, Manoeuvre::Stopping, Scope::Any, 0.0, true,
             "KTK 2918 md. 60(f)", "stopping at a tram stop"},
            // "Isaret levhalarina, yaklasim yonunde ... yerlesim birimi icinde onbes metre
            // ve yerlesim birimi disinda yuz metre mesafede". Approach direction only,
            // which is the one rule in any of these books that is genuinely one-sided.
            {AnchorKind::PublicEntrance, Manoeuvre::Stopping, Scope::BuiltUp, m(15.0), false,
             "KTK 2918 md. 60(h)",
             "within fifteen metres of a sign board on the approach side, in a built-up area"},

            // --- article 61: park etme (parking) ------------------------------------
            // "Gecis yollari onunde veya uzerinde"
            {AnchorKind::Driveway, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "KTK 2918 md. 61(c)", "in front of or on an access way"},
            // "Belirlenmis yangin musluklarina her iki yonden bes metrelik mesafe icinde"
            {AnchorKind::FireHydrant, Manoeuvre::Parking, Scope::Any, m(5.0), true,
             "KTK 2918 md. 61(d)",
             "within five metres of a designated fire hydrant, in either direction"},
            // "Kamu hizmeti yapan yolcu tasitlarinin duraklarini belirten levhalara iki
            // yonden onbes metrelik mesafe icinde"
            {AnchorKind::BusStopSign, Manoeuvre::Parking, Scope::Any, m(15.0), true,
             "KTK 2918 md. 61(e)",
             "within fifteen metres of a public transport stop sign, in either direction"},
            // "Gecis ustunlugu olan araclarin giris ve cikisinin yapildiginin belirlendigi
            // isaret levhasindan onbes metre mesafe icinde"
            {AnchorKind::EmergencyAccess, Manoeuvre::Parking, Scope::Any, m(15.0), true,
             "KTK 2918 md. 61(h)",
             "within fifteen metres of a marked emergency vehicle access"},
            // "Kamunun faydalandigi ... giris ve cikis kapilarinin her iki yonde bes
            // metrelik mesafe icinde"
            {AnchorKind::PublicEntrance, Manoeuvre::Parking, Scope::Any, m(5.0), true,
             "KTK 2918 md. 61(j)",
             "within five metres of the entrance or exit gate of a public building"},
            // "... alt gecit, ust gecit ve koprüler uzerinde veya bunlara on metrelik
            // mesafe icinde"
            {AnchorKind::Underpass, Manoeuvre::Parking, Scope::Any, m(10.0), true,
             "KTK 2918 md. 61(k)", "on or within ten metres of an underpass"},
            {AnchorKind::Bridge, Manoeuvre::Parking, Scope::Any, m(10.0), true,
             "KTK 2918 md. 61(k)", "on or within ten metres of an overpass or bridge"},
            // "yaya yollarda"
            {AnchorKind::Footway, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "KTK 2918 md. 61(n)", "parking on a pedestrian way"},
            // "Engellilerin araclari icin ayrilmis park yerlerinde". The statute doubles
            // the fine for this one, which is worth surfacing to a driver.
            {AnchorKind::DisabledBay, Manoeuvre::Parking, Scope::Any, 0.0, true,
             "KTK 2918 md. 61(o)",
             "in a bay reserved for disabled drivers, where the fine is doubled"},
        },
    };
}

}  // namespace parkfit::legal::tr
