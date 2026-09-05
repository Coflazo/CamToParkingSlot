// SPDX-License-Identifier: MIT
//
// Legality tests.
//
// The fit engine already answers "does the car go there". These cover the other half:
// whether a driver may legally leave it there. The cases that matter most are the ones
// where a space is perfectly usable and still forbidden, because those are the ones a
// geometry-only system gets confidently wrong.
//
// Three properties are worth more than the rest, and each has a test that fails loudly:
// Unknown must never decay into Legal, parking must inherit stopping prohibitions, and
// an untranscribed rulebook must refuse to judge rather than wave everything through.

#include "test_framework.hpp"

#include <chrono>
#include <cstdio>
#include <string>

#include "parkfit/legal/rulebook.hpp"
#include "parkfit/legal/rulebook_de.hpp"
#include "parkfit/legal/rulebook_fr.hpp"
#include "parkfit/legal/rulebook_nl.hpp"
#include "parkfit/legal/rulebook_tr.hpp"

using namespace parkfit::legal;

namespace {

/// One anchor at a measured distance, in metres, which is how statutes are written.
std::vector<AnchorHit> at(AnchorKind kind, double metres) {
    return {AnchorHit{kind, metres * 100.0}};
}

bool cites(const LegalFinding& finding, const std::string& fragment) {
    return std::string(finding.citation).find(fragment) != std::string::npos;
}

}  // namespace

// ------------------------------------------------------------------ Turkiye

TEST_CASE("legal: a space three metres from an Istanbul hydrant is refused, and cited") {
    // The headline case. The space can be empty, measured and wide enough, and it is
    // still a fine. Nothing in the fit engine can see this.
    const auto book = tr::rulebook();
    const auto finding = evaluate(book, Manoeuvre::Parking, at(AnchorKind::FireHydrant, 3.0));

    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK_EQ(finding.anchor, AnchorKind::FireHydrant);
    CHECK(cites(finding, "md. 61(d)"));
    CHECK_NEAR(finding.required_cm, 500.0, 1e-9);
    CHECK_NEAR(finding.distance_cm, 300.0, 1e-9);
}

TEST_CASE("legal: six metres from the same hydrant is fine") {
    const auto book = tr::rulebook();
    const auto finding = evaluate(book, Manoeuvre::Parking, at(AnchorKind::FireHydrant, 6.0));
    CHECK_EQ(finding.verdict, LegalVerdict::Legal);
}

TEST_CASE("legal: the hydrant rule binds exactly at five metres, not just inside it") {
    // "bes metrelik mesafe icinde" includes the boundary, so 5.00 m is still prohibited
    // and 5.01 m is not. Off-by-one here is a real fine.
    const auto book = tr::rulebook();
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::FireHydrant, 5.0)).verdict,
             LegalVerdict::Prohibited);
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::FireHydrant, 5.01)).verdict,
             LegalVerdict::Legal);
}

TEST_CASE("legal: Turkish junction distance is twenty times larger outside a built-up area") {
    const auto book = tr::rulebook();
    Context town;
    town.built_up = true;
    Context country;
    country.built_up = false;

    // Fifty metres from a junction: nothing at all in a city, prohibited on a rural road.
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::Junction, 50.0), town).verdict,
             LegalVerdict::Legal);
    const auto rural =
        evaluate(book, Manoeuvre::Parking, at(AnchorKind::Junction, 50.0), country);
    CHECK_EQ(rural.verdict, LegalVerdict::Prohibited);
    CHECK(cites(rural, "md. 60(d)"));
    CHECK_NEAR(rural.required_cm, 10000.0, 1e-9);
}

TEST_CASE("legal: parking inherits the Turkish stopping prohibitions") {
    // Article 61(a) says so outright. A tram stop is an article 60 rule, and it must
    // still bind a parking query.
    const auto book = tr::rulebook();
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::TramStop, 0.0)).verdict,
             LegalVerdict::Prohibited);
}

TEST_CASE("legal: a parking-only rule does not bind a stopping query") {
    // The inheritance runs one way. Stopping briefly by a hydrant is not article 61(d).
    const auto book = tr::rulebook();
    CHECK_EQ(evaluate(book, Manoeuvre::Stopping, at(AnchorKind::FireHydrant, 3.0)).verdict,
             LegalVerdict::Legal);
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::FireHydrant, 3.0)).verdict,
             LegalVerdict::Prohibited);
}

TEST_CASE("legal: the Turkish bridge rule has no Dutch or German equivalent") {
    // Eight metres from a bridge: a fine in Istanbul, nothing anywhere else here.
    const auto hits = at(AnchorKind::Bridge, 8.0);
    CHECK_EQ(evaluate(tr::rulebook(), Manoeuvre::Parking, hits).verdict,
             LegalVerdict::Prohibited);
    CHECK_EQ(evaluate(nl::rulebook(), Manoeuvre::Parking, hits).verdict, LegalVerdict::Legal);
    CHECK_EQ(evaluate(de::rulebook(), Manoeuvre::Parking, hits).verdict, LegalVerdict::Legal);
}

// -------------------------------------------------------------- Netherlands

TEST_CASE("legal: four metres from a Dutch pedestrian crossing is refused, and cited") {
    const auto book = nl::rulebook();
    const auto finding =
        evaluate(book, Manoeuvre::Parking, at(AnchorKind::PedestrianCrossing, 4.0));
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK(cites(finding, "art. 23(1)(c)"));
    CHECK_NEAR(finding.required_cm, 500.0, 1e-9);
}

TEST_CASE("legal: the Dutch bus stop distance is twelve metres, not fifteen") {
    // Every popular summary says fifteen by analogy with Germany. The statute says
    // twelve, and thirteen metres from a Dutch bus stop is a legal space this product
    // would otherwise refuse to offer.
    const auto book = nl::rulebook();
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::BusStopSign, 11.0)).verdict,
             LegalVerdict::Prohibited);
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::BusStopSign, 13.0)).verdict,
             LegalVerdict::Legal);
    // Germany, at the same spot, still refuses.
    CHECK_EQ(evaluate(de::rulebook(), Manoeuvre::Parking, at(AnchorKind::BusStopSign, 13.0))
                 .verdict,
             LegalVerdict::Prohibited);
}

TEST_CASE("legal: a marked-bay road allows parking only inside the bays") {
    // RVV art. 24(4). This is the legal basis for the whole bay-polygon approach: on
    // these streets a gap between two cars is not a space, however long it measures.
    const auto book = nl::rulebook();
    Context outside_bay;
    outside_bay.road_has_marked_bays = true;
    outside_bay.inside_marked_bay = false;

    const auto finding = evaluate(book, Manoeuvre::Parking, {}, outside_bay);
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK(cites(finding, "art. 24(4)"));

    Context inside_bay;
    inside_bay.road_has_marked_bays = true;
    inside_bay.inside_marked_bay = true;
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, {}, inside_bay).verdict, LegalVerdict::Legal);
}

TEST_CASE("legal: a permit zone without a permit is a refusal, not a suggestion") {
    const auto book = nl::rulebook();
    Context context;
    context.permit_zone_without_permit = true;
    const auto finding = evaluate(book, Manoeuvre::Parking, {}, context);
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK(cites(finding, "art. 24(1)(g)"));
}

TEST_CASE("legal: a disc zone stays conditional and is never resolved into legal") {
    // Conditional is a real third answer. Collapsing it into Legal promises a space the
    // driver cannot take without a disc; collapsing it into Prohibited hides a usable one.
    const auto book = nl::rulebook();
    Context context;
    context.disc_zone = true;
    const auto finding = evaluate(book, Manoeuvre::Parking, {}, context);
    CHECK_EQ(finding.verdict, LegalVerdict::Conditional);
    CHECK(cites(finding, "art. 25"));
}

// ------------------------------------------------------------------ Germany

TEST_CASE("legal: six metres from a German junction with a cycle path is still refused") {
    // StVO 12(3) Nr. 1 extends 5 m to 8 m wherever a structurally separate cycle path
    // runs on the right. Most summaries drop that half of the sentence, and dropping it
    // would offer a space on almost every modern German residential street.
    const auto book = de::rulebook();

    // A plain junction at 6 m is fine.
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::Junction, 6.0)).verdict,
             LegalVerdict::Legal);

    // The same junction, with a separate cycle path, is not. A caller emits both anchor
    // kinds for such a feature and the stricter rule has to win.
    const std::vector<AnchorHit> both{AnchorHit{AnchorKind::Junction, 600.0},
                                      AnchorHit{AnchorKind::JunctionWithCyclePath, 600.0}};
    const auto finding = evaluate(book, Manoeuvre::Parking, both);
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK_EQ(finding.anchor, AnchorKind::JunctionWithCyclePath);
    CHECK_NEAR(finding.required_cm, 800.0, 1e-9);
    CHECK(cites(finding, "StVO 12(3)"));
}

TEST_CASE("legal: the German cycle-path anchor is ignored by books that do not use it") {
    // A Dutch or Turkish evaluation must not trip over an anchor kind only Germany needs.
    const std::vector<AnchorHit> both{AnchorHit{AnchorKind::Junction, 600.0},
                                      AnchorHit{AnchorKind::JunctionWithCyclePath, 600.0}};
    CHECK_EQ(evaluate(nl::rulebook(), Manoeuvre::Parking, both).verdict, LegalVerdict::Legal);
    CHECK_EQ(evaluate(tr::rulebook(), Manoeuvre::Parking, both).verdict, LegalVerdict::Legal);
}

TEST_CASE("legal: the German bus stop rule cites Anlage 2, not paragraph 12") {
    // The 15 m rule belongs to Zeichen 224. Citing paragraph 12 would put a wrong article
    // in front of a user, which is worse than giving no citation at all.
    const auto finding =
        evaluate(de::rulebook(), Manoeuvre::Parking, at(AnchorKind::BusStopSign, 10.0));
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK(cites(finding, "Zeichen 224"));
    CHECK(!cites(finding, "StVO 12"));
}

// ------------------------------------------------------------------- France

TEST_CASE("legal: an untranscribed rulebook refuses to judge instead of allowing") {
    // The failure this guards against is specific and silent: an empty rule table breaks
    // no rules, so a naive evaluation returns Legal and the product declares every space
    // in Paris usable on the strength of having done no legal work at all.
    const auto book = fr::rulebook();
    CHECK(!book.complete);
    CHECK(book.rules.empty());

    // Even standing on a fire hydrant, the answer is Unknown, never Legal.
    const auto finding = evaluate(book, Manoeuvre::Parking, at(AnchorKind::FireHydrant, 0.0));
    CHECK_EQ(finding.verdict, LegalVerdict::Unknown);
    CHECK(finding.verdict != LegalVerdict::Legal);
}

// ------------------------------------------------------------ general rules

TEST_CASE("legal: missing anchor data yields Unknown, never Legal") {
    const auto book = nl::rulebook();
    Context nothing_loaded;
    nothing_loaded.anchors_loaded = false;
    const auto finding = evaluate(book, Manoeuvre::Parking, {}, nothing_loaded);
    CHECK_EQ(finding.verdict, LegalVerdict::Unknown);
}

TEST_CASE("legal: a clean sweep with anchors loaded is Legal") {
    const auto book = nl::rulebook();
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::Junction, 40.0)).verdict,
             LegalVerdict::Legal);
}

TEST_CASE("legal: a zero-distance rule triggers only when the point is on the feature") {
    // "op een fietsstrook" is about being on it, not near it. A car parked two metres
    // from a cycle lane is not on the cycle lane.
    const auto book = nl::rulebook();
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::CycleLane, 0.0)).verdict,
             LegalVerdict::Prohibited);
    CHECK_EQ(evaluate(book, Manoeuvre::Parking, at(AnchorKind::CycleLane, 2.0)).verdict,
             LegalVerdict::Legal);
}

TEST_CASE("legal: every violation is reported, worst first") {
    // A driver deserves both facts, and the one hardest to argue with should lead.
    const std::vector<AnchorHit> hits{
        AnchorHit{AnchorKind::FireHydrant, 450.0},    // 0.5 m inside a 5 m rule
        AnchorHit{AnchorKind::BusStopSign, 200.0},    // 13 m inside a 15 m rule
    };
    const auto broken = violations(tr::rulebook(), Manoeuvre::Parking, hits);
    CHECK(broken.size() >= 2);
    CHECK_EQ(broken.front().anchor, AnchorKind::BusStopSign);
    CHECK(cites(broken.front(), "md. 61(e)"));
}

TEST_CASE("legal: every rule in every complete book carries a citation") {
    // A distance with no provenance is a number nobody can audit, and the product is
    // built on being auditable. This is cheap to check and easy to forget.
    for (const auto& book : {nl::rulebook(), de::rulebook(), tr::rulebook()}) {
        CHECK(book.complete);
        CHECK(!book.rules.empty());
        for (const auto& rule : book.rules) {
            CHECK(std::string(rule.citation).size() > 4);
            CHECK(std::string(rule.reason).size() > 4);
            CHECK(rule.distance_cm >= 0.0);
        }
    }
}

TEST_CASE("legal: measuring positions agrees with hand-computed distances") {
    const parkfit::geo::LatLon at_point{52.3700, 4.9000};
    // Roughly 111.3 m east at this latitude.
    const parkfit::geo::LatLon hydrant{52.3700, 4.9016362};
    const auto hits = measure(at_point, {{AnchorKind::FireHydrant, hydrant}});
    CHECK_EQ(hits.size(), static_cast<std::size_t>(1));
    CHECK_NEAR(hits[0].distance_cm / 100.0, 111.3, 1.5);
}

TEST_CASE("legal: verdict and anchor names round-trip to strings") {
    CHECK_EQ(std::string(to_string(LegalVerdict::Prohibited)), "prohibited");
    CHECK_EQ(std::string(to_string(LegalVerdict::Unknown)), "unknown");
    CHECK_EQ(std::string(to_string(AnchorKind::FireHydrant)), "fire_hydrant");
    CHECK_EQ(std::string(to_string(AnchorKind::JunctionWithCyclePath)),
             "junction_with_cycle_path");
    CHECK_EQ(std::string(to_string(Manoeuvre::Parking)), "parking");
}

// ------------------------------------------------------- the anchor index

TEST_CASE("legal: the index finds a hydrant and the verdict matches a hand measurement") {
    using parkfit::geo::LatLon;
    AnchorIndex anchors;
    // A hydrant about 3.3 m north of the candidate point.
    anchors.add(AnchorKind::FireHydrant, LatLon{52.3700300, 4.9000000});
    anchors.build();

    const auto finding =
        evaluate_at(tr::rulebook(), Manoeuvre::Parking, anchors, LatLon{52.3700, 4.9000});
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK_EQ(finding.anchor, AnchorKind::FireHydrant);
    CHECK(finding.distance_cm < 500.0);
}

TEST_CASE("legal: an empty index forces Unknown rather than Legal") {
    using parkfit::geo::LatLon;
    AnchorIndex anchors;
    anchors.build();
    CHECK_EQ(evaluate_at(nl::rulebook(), Manoeuvre::Parking, anchors, LatLon{52.37, 4.90}).verdict,
             LegalVerdict::Unknown);
}

TEST_CASE("legal: the sweep radius comes from the book, so long rules are not missed") {
    using parkfit::geo::LatLon;
    // Turkey reaches 100 m outside a built-up area. A fixed short radius would sweep
    // 15 m, find nothing, and report Legal for the wrong reason.
    CHECK_NEAR(max_distance_cm(tr::rulebook()), 10000.0, 1e-9);
    CHECK_NEAR(max_distance_cm(nl::rulebook()), 1200.0, 1e-9);
    CHECK_NEAR(max_distance_cm(de::rulebook()), 1500.0, 1e-9);

    AnchorIndex anchors;
    // A junction roughly 55 m away: inside Turkey's rural rule, outside every other.
    anchors.add(AnchorKind::Junction, LatLon{52.3705, 4.9000});
    anchors.build();

    Context countryside;
    countryside.built_up = false;
    const auto finding =
        evaluate_at(tr::rulebook(), Manoeuvre::Parking, anchors, LatLon{52.37, 4.90}, countryside);
    CHECK_EQ(finding.verdict, LegalVerdict::Prohibited);
    CHECK(cites(finding, "md. 60(d)"));
}

TEST_CASE("legal: evaluate_many answers one finding per point, in order") {
    using parkfit::geo::LatLon;
    AnchorIndex anchors;
    anchors.add(AnchorKind::FireHydrant, LatLon{52.3700300, 4.9000000});
    anchors.build();

    const std::vector<LatLon> points{
        LatLon{52.3700, 4.9000},   // right beside the hydrant
        LatLon{52.3710, 4.9000},   // about 111 m away, clear of it
    };
    const auto findings = evaluate_many(tr::rulebook(), Manoeuvre::Parking, anchors, points);
    CHECK_EQ(findings.size(), static_cast<std::size_t>(2));
    CHECK_EQ(findings[0].verdict, LegalVerdict::Prohibited);
    CHECK_EQ(findings[1].verdict, LegalVerdict::Legal);
}

TEST_CASE("legal: a candidate-scale legality sweep is not the slow part of a search") {
    using parkfit::geo::LatLon;
    // 20k anchors is the order of the bus stops, crossings and hydrants in a city
    // centre, and 400 candidates is what a search scores. If this costs milliseconds the
    // legality layer would be more expensive than the routing it sits beside.
    AnchorIndex anchors;
    anchors.reserve(20000);
    for (int i = 0; i < 20000; ++i) {
        const double lat = 52.30 + 0.0001 * static_cast<double>(i % 200);
        const double lon = 4.80 + 0.0001 * static_cast<double>(i / 200);
        anchors.add(static_cast<AnchorKind>(i % 6), LatLon{lat, lon});
    }
    anchors.build();

    std::vector<LatLon> points;
    points.reserve(400);
    for (int i = 0; i < 400; ++i) {
        points.push_back(LatLon{52.30 + 0.0001 * (i % 200), 4.80 + 0.0001 * (i / 200)});
    }

    const auto book = nl::rulebook();
    auto warm = evaluate_many(book, Manoeuvre::Parking, anchors, points);
    CHECK_EQ(warm.size(), static_cast<std::size_t>(400));

    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kIters = 20;
    for (int i = 0; i < kIters; ++i) {
        auto findings = evaluate_many(book, Manoeuvre::Parking, anchors, points);
        CHECK_EQ(findings.size(), static_cast<std::size_t>(400));
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / kIters;
    std::printf("[perf] 400 candidates against 20k anchors: %.2f ms\n", ms);
    CHECK(ms < 50.0);
}

PF_TEST_MAIN()
