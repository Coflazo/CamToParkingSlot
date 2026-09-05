// SPDX-License-Identifier: MIT
//
// Where a car may legally stop or park, by country.
//
// The fit engine answers "does this car physically go there". That is necessary and it
// is not sufficient. A space can be empty, measured, and wide enough, and still be one
// a driver must not use: in front of a fire hydrant, inside a bus stop's blocked
// marking, five metres from a junction. Offering one of those costs a fine, and it is a
// worse failure than missing a valid space, because the driver acts on it.
//
// Two design choices carry the whole module.
//
// **A rule is data, and it carries its citation.** Every entry names the article it
// comes from, and the citation travels with the verdict all the way to the interface.
// If this product refuses a space it can say why, in the words of the statute, and a
// user can check it. A hard-coded distance with no provenance is a number nobody can
// audit, and the whole product is built on being auditable.
//
// **The rulebook does the work, not the map.** Explicit parking restrictions are mapped
// densely in almost nowhere: Berlin has 2,762 ways tagged with the OSM parking schema,
// Paris 88, Istanbul 50, Amsterdam 33. But the things the statutes measure *from* are
// everywhere, because they are ordinary map features: bus stops, crossings, hydrants,
// junctions. Encoding the statute and measuring to those anchors turns a data-coverage
// problem into a geometry problem, which is why this generalises to a new country by
// adding a table rather than by waiting for someone to map it.
//
// Unknown is never Legal. Missing anchor data lowers confidence; it does not grant
// permission. This mirrors BayState::Unknown in the vision module, and for the same
// reason: the honest answer to "I cannot see" is not "yes".

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "parkfit/geo/primitives.hpp"
#include "parkfit/index/grid.hpp"

namespace parkfit::legal {

using geo::LatLon;

/// Stopping and parking are separate offences in every statute modelled here, and the
/// prohibited sets are different. The Netherlands splits them across RVV articles 23 and
/// 24; Turkey across KTK 2918 articles 60 and 61; Germany within StVO paragraph 12.
enum class Manoeuvre {
    /// Halting with the driver present, to set someone down or wait briefly.
    Stopping,
    /// Leaving the vehicle. Always at least as restricted as stopping.
    Parking,
};

/// Some distances change outside a built-up area. Turkey is the sharp case: five metres
/// from a junction inside a built-up area, one hundred metres outside it.
enum class Scope { Any, BuiltUp, Outside };

/// The map features a statute measures from.
///
/// These are deliberately the things an ordinary map already contains. Anything that
/// needed a bespoke survey to locate would not generalise past one city.
enum class AnchorKind {
    Junction,
    /// A junction that also has a structurally separate cycle path on the right in the
    /// direction of travel. Germany extends its junction setback from 5 m to 8 m for
    /// exactly this case, so the distinction has to survive into the rule table. A
    /// caller emits *both* this and Junction for such a feature, and the stricter rule
    /// wins; that way a country whose statute does not care never has to know about it.
    JunctionWithCyclePath,
    PedestrianCrossing,
    LevelCrossing,
    BusStopSign,
    TramStop,
    FireHydrant,
    Driveway,
    Bridge,
    Underpass,
    Tunnel,
    CycleLane,
    Footway,
    EmergencyAccess,
    PublicEntrance,
    DisabledBay,
    LoadingBay,
    BusLane,
    YellowLineSolid,
    YellowLineBroken,
};

enum class LegalVerdict {
    /// No rule in the book prohibits it, and the anchor data was good enough to say so.
    Legal,
    /// A rule prohibits it. The finding names which and cites the article.
    Prohibited,
    /// Allowed only subject to something the driver must satisfy: a permit, a parking
    /// disc, staying inside a marked bay. Never silently resolved into Legal.
    Conditional,
    /// The rulebook could not be applied, usually because the anchors for this area were
    /// not loaded. Not a synonym for Legal.
    Unknown,
};

/// One statutory prohibition.
struct LegalRule {
    AnchorKind anchor{};
    Manoeuvre manoeuvre{};
    Scope scope{Scope::Any};
    /// Prohibited within this distance of the anchor. Zero means the prohibition applies
    /// only when the vehicle is *on* the feature rather than near it, which is how a
    /// cycle lane or a pedestrian way works.
    double distance_cm{0.0};
    /// True when the statute measures in both directions from the feature. Turkey says
    /// so explicitly for hydrants and transit stops ("her iki yonden"); the Netherlands
    /// and Germany measure before and after a junction alike.
    bool both_directions{true};
    /// The article this comes from, verbatim enough to look up.
    const char* citation{""};
    /// What the rule protects, in plain words, for an interface to show.
    const char* reason{""};
};

/// A candidate point's measured distance to one nearby anchor.
///
/// The caller measures, because it owns the spatial index and the polygon geometry. A
/// distance of zero means the point lies on or inside the feature.
struct AnchorHit {
    AnchorKind kind{};
    double distance_cm{0.0};
};

/// Conditions that come from the bay's own attributes rather than from proximity.
///
/// These are read off sign codes and regimes during ingest, not measured. Amsterdam
/// already parses them out of the parkeervakken eType field.
struct Context {
    bool built_up{true};
    /// The road has marked bays. Where it does, several statutes allow parking only
    /// inside them, which turns "there is room here" into "there is room, but not a bay".
    bool road_has_marked_bays{false};
    bool inside_marked_bay{false};
    /// Signed for permit holders, and this driver has no permit for that zone.
    bool permit_zone_without_permit{false};
    /// A disc zone: legal, but only with a parking disc displayed.
    bool disc_zone{false};
    /// True when the anchors for this area were actually loaded. False forces Unknown,
    /// because a rulebook with nothing to measure against proves nothing.
    bool anchors_loaded{true};
};

/// One reason a point is or is not usable.
struct LegalFinding {
    LegalVerdict verdict{LegalVerdict::Unknown};
    AnchorKind anchor{};
    /// Measured distance to the anchor, or -1 when the finding is not distance-based.
    double distance_cm{-1.0};
    /// What the statute required. -1 when not distance-based.
    double required_cm{-1.0};
    const char* citation{""};
    const char* reason{""};
};

/// A country's rules, plus the metadata needed to attribute them.
struct Rulebook {
    /// ISO 3166-1 alpha-2, so this keys the same way as the rest of the country plumbing.
    const char* country{""};
    /// The instrument these rules come from, for the attribution line.
    const char* instrument{""};
    std::vector<LegalRule> rules;
    /// False while the statute has not been transcribed from its primary source yet.
    ///
    /// This exists because of a specific way to be silently wrong. An empty rule table
    /// breaks no rules, so a naive evaluation of one returns Legal, and a country nobody
    /// has done the legal work for would confidently declare every space usable. That is
    /// the worst possible failure for this module, so an incomplete book forces Unknown
    /// no matter what the anchors say.
    bool complete{true};
};

// ---------------------------------------------------------------- helpers

inline const char* to_string(LegalVerdict v) {
    switch (v) {
        case LegalVerdict::Legal: return "legal";
        case LegalVerdict::Prohibited: return "prohibited";
        case LegalVerdict::Conditional: return "conditional";
        case LegalVerdict::Unknown: return "unknown";
    }
    return "unknown";
}

inline const char* to_string(AnchorKind a) {
    switch (a) {
        case AnchorKind::Junction: return "junction";
        case AnchorKind::JunctionWithCyclePath: return "junction_with_cycle_path";
        case AnchorKind::PedestrianCrossing: return "pedestrian_crossing";
        case AnchorKind::LevelCrossing: return "level_crossing";
        case AnchorKind::BusStopSign: return "bus_stop_sign";
        case AnchorKind::TramStop: return "tram_stop";
        case AnchorKind::FireHydrant: return "fire_hydrant";
        case AnchorKind::Driveway: return "driveway";
        case AnchorKind::Bridge: return "bridge";
        case AnchorKind::Underpass: return "underpass";
        case AnchorKind::Tunnel: return "tunnel";
        case AnchorKind::CycleLane: return "cycle_lane";
        case AnchorKind::Footway: return "footway";
        case AnchorKind::EmergencyAccess: return "emergency_access";
        case AnchorKind::PublicEntrance: return "public_entrance";
        case AnchorKind::DisabledBay: return "disabled_bay";
        case AnchorKind::LoadingBay: return "loading_bay";
        case AnchorKind::BusLane: return "bus_lane";
        case AnchorKind::YellowLineSolid: return "yellow_line_solid";
        case AnchorKind::YellowLineBroken: return "yellow_line_broken";
    }
    return "unknown";
}

inline const char* to_string(Manoeuvre m) {
    return m == Manoeuvre::Stopping ? "stopping" : "parking";
}

/// Parking inherits every stopping prohibition.
///
/// This is not an approximation, it is what the statutes say. Turkey states it outright:
/// KTK 2918 article 61(a) prohibits parking "in places where stopping is prohibited". The
/// Dutch and German schemes reach the same result, since a vehicle that may not stop
/// somewhere plainly may not be left there. So a stopping rule binds a parking query, and
/// a parking rule does not bind a stopping query.
inline bool applies_to(const LegalRule& rule, Manoeuvre manoeuvre) {
    if (rule.manoeuvre == manoeuvre) return true;
    return manoeuvre == Manoeuvre::Parking && rule.manoeuvre == Manoeuvre::Stopping;
}

inline bool applies_in(const LegalRule& rule, bool built_up) {
    switch (rule.scope) {
        case Scope::Any: return true;
        case Scope::BuiltUp: return built_up;
        case Scope::Outside: return !built_up;
    }
    return true;
}

/// Metres to centimetres, so rule tables can be written in the units the statute uses.
inline constexpr double m(double metres) { return metres * 100.0; }

// ------------------------------------------------------------- evaluation

/// Every rule this point breaks, worst first.
///
/// Worst is measured by how far inside the prohibited distance the point sits, so the
/// finding a user sees first is the one hardest to argue with. All of them are returned
/// rather than just the first, because "you are too close to a hydrant" and "you are
/// also in a bus stop" are different facts and a driver deserves both.
inline std::vector<LegalFinding> violations(const Rulebook& book, Manoeuvre manoeuvre,
                                            const std::vector<AnchorHit>& hits,
                                            const Context& context = Context{}) {
    std::vector<LegalFinding> found;
    for (const auto& rule : book.rules) {
        if (!applies_to(rule, manoeuvre) || !applies_in(rule, context.built_up)) continue;
        for (const auto& hit : hits) {
            if (hit.kind != rule.anchor) continue;
            // A zero-distance rule triggers only when the point is on the feature; any
            // positive rule triggers at or inside its distance.
            if (hit.distance_cm > rule.distance_cm) continue;
            found.push_back(LegalFinding{LegalVerdict::Prohibited, rule.anchor, hit.distance_cm,
                                         rule.distance_cm, rule.citation, rule.reason});
        }
    }
    std::sort(found.begin(), found.end(), [](const LegalFinding& a, const LegalFinding& b) {
        const double slack_a = a.required_cm - a.distance_cm;
        const double slack_b = b.required_cm - b.distance_cm;
        if (slack_a != slack_b) return slack_a > slack_b;
        return a.distance_cm < b.distance_cm;
    });
    return found;
}

/// Conditions that do not come from proximity: permits, discs, marked bays.
///
/// Kept separate from the setback rules because they are read from the bay's own
/// attributes during ingest rather than measured against the map, and because they
/// produce Conditional rather than Prohibited. A conditional answer is genuinely
/// different from a refusal, and collapsing the two would either hide usable spaces or
/// promise ones the driver cannot legally take.
inline std::vector<LegalFinding> conditions(const Rulebook& book, Manoeuvre manoeuvre,
                                            const Context& context) {
    std::vector<LegalFinding> found;
    if (manoeuvre != Manoeuvre::Parking) return found;

    if (context.permit_zone_without_permit) {
        found.push_back(LegalFinding{LegalVerdict::Prohibited, AnchorKind::PublicEntrance, -1.0,
                                     -1.0,
                                     std::string(book.country) == "NL" ? "RVV 1990 art. 24(1)(g)"
                                                                       : "permit zone",
                                     "signed for permit holders and no permit is held"});
    }
    if (context.road_has_marked_bays && !context.inside_marked_bay) {
        found.push_back(LegalFinding{
            LegalVerdict::Prohibited, AnchorKind::LoadingBay, -1.0, -1.0,
            std::string(book.country) == "NL" ? "RVV 1990 art. 24(4)" : "marked bays only",
            "this road has marked bays, and parking is allowed only inside them"});
    }
    if (context.disc_zone) {
        found.push_back(LegalFinding{
            LegalVerdict::Conditional, AnchorKind::PublicEntrance, -1.0, -1.0,
            std::string(book.country) == "NL" ? "RVV 1990 art. 25" : "disc zone",
            "a parking disc must be displayed"});
    }
    return found;
}

/// The single verdict for a point, and the reason behind it.
///
/// Order matters: Unknown outranks everything when the anchors were never loaded,
/// because a clean sweep of a rulebook against no data is not evidence of legality.
inline LegalFinding evaluate(const Rulebook& book, Manoeuvre manoeuvre,
                             const std::vector<AnchorHit>& hits,
                             const Context& context = Context{}) {
    if (!book.complete) {
        return LegalFinding{LegalVerdict::Unknown, AnchorKind::Junction, -1.0, -1.0,
                            book.instrument,
                            "the rulebook for this country is not transcribed yet, so legality "
                            "cannot be judged"};
    }
    if (!context.anchors_loaded) {
        return LegalFinding{LegalVerdict::Unknown, AnchorKind::Junction, -1.0, -1.0,
                            book.instrument,
                            "no restriction data loaded for this area, so legality is unknown"};
    }

    const auto broken = violations(book, manoeuvre, hits, context);
    if (!broken.empty()) return broken.front();

    const auto conditional = conditions(book, manoeuvre, context);
    for (const auto& finding : conditional) {
        if (finding.verdict == LegalVerdict::Prohibited) return finding;
    }
    if (!conditional.empty()) return conditional.front();

    return LegalFinding{LegalVerdict::Legal, AnchorKind::Junction, -1.0, -1.0, book.instrument,
                        "no rule in this book prohibits it"};
}

/// Turn anchor positions into measured hits.
///
/// A convenience for callers that already have positions rather than distances. The hot
/// path uses the spatial index and measures once, so this exists for tests and for small
/// candidate sets rather than for bulk work.
inline std::vector<AnchorHit> measure(const LatLon& at,
                                      const std::vector<std::pair<AnchorKind, LatLon>>& anchors) {
    std::vector<AnchorHit> hits;
    hits.reserve(anchors.size());
    for (const auto& [kind, position] : anchors) {
        hits.push_back(AnchorHit{kind, geo::haversine_m(at, position) * 100.0});
    }
    return hits;
}

/// The furthest any rule in this book reaches.
///
/// This is the search radius a caller must use. Anything smaller silently misses rules:
/// query 20 m around a Turkish rural junction and the hundred-metre prohibition in
/// article 60(d) never fires, and the answer comes back Legal for the wrong reason.
inline double max_distance_cm(const Rulebook& book) {
    double furthest = 0.0;
    for (const auto& rule : book.rules) furthest = std::max(furthest, rule.distance_cm);
    return furthest;
}

/// A spatial index over the map features the statutes measure from.
///
/// Built once per area and reused across every candidate in a search. A search scores a
/// few hundred candidates and each one needs its own anchor sweep, so doing this in
/// Python would mean a few hundred radius queries plus a few thousand haversines per
/// request, which is exactly the shape of work that belongs on this side of the boundary.
class AnchorIndex {
  public:
    void reserve(std::size_t n) {
        kinds_.reserve(n);
        grid_.reserve(n);
    }

    void add(AnchorKind kind, const LatLon& position) {
        grid_.insert(position, static_cast<std::uint32_t>(kinds_.size()));
        kinds_.push_back(kind);
    }

    void build() { grid_.build(); }
    void clear() {
        kinds_.clear();
        grid_.clear();
    }

    [[nodiscard]] std::size_t size() const { return kinds_.size(); }
    [[nodiscard]] bool empty() const { return kinds_.empty(); }

    /// Every anchor within `radius_m`, already measured.
    std::vector<AnchorHit> hits_near(const LatLon& at, double radius_m) {
        std::vector<AnchorHit> hits;
        for (const auto& hit : grid_.query_radius(at, radius_m, 0)) {
            hits.push_back(AnchorHit{kinds_[hit.payload], hit.distance_m * 100.0});
        }
        return hits;
    }

  private:
    std::vector<AnchorKind> kinds_;
    index::SpatialGrid grid_{100.0};
};

/// Judge one point against the book, sweeping the index for the anchors that matter.
///
/// The radius comes from the book itself rather than from the caller, so a country whose
/// statute reaches a hundred metres is swept a hundred metres and one that reaches
/// fifteen is not swept further than it needs.
inline LegalFinding evaluate_at(const Rulebook& book, Manoeuvre manoeuvre, AnchorIndex& anchors,
                                const LatLon& at, Context context = Context{}) {
    if (anchors.empty()) context.anchors_loaded = false;
    if (!book.complete || !context.anchors_loaded) {
        return evaluate(book, manoeuvre, {}, context);
    }
    const double radius_m = std::max(1.0, max_distance_cm(book) / 100.0);
    return evaluate(book, manoeuvre, anchors.hits_near(at, radius_m), context);
}

/// The same, for many points at once.
///
/// One crossing of the language boundary instead of one per candidate, matching the
/// `insert_many` idiom the spatial index already uses. `contexts` may be empty, in which
/// case `shared` applies to every point; otherwise it must have one entry per point.
inline std::vector<LegalFinding> evaluate_many(const Rulebook& book, Manoeuvre manoeuvre,
                                               AnchorIndex& anchors,
                                               const std::vector<LatLon>& points,
                                               const std::vector<Context>& contexts = {},
                                               const Context& shared = Context{}) {
    std::vector<LegalFinding> out;
    out.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Context& context = contexts.empty() ? shared : contexts[i];
        out.push_back(evaluate_at(book, manoeuvre, anchors, points[i], context));
    }
    return out;
}

}  // namespace parkfit::legal
