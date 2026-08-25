// SPDX-License-Identifier: MIT
//
// Recommendation scoring.
//
// The central idea: rank by expected total inconvenience, not by distance. A free kerb
// space 200 m away that will probably be gone in four minutes is worse than a guaranteed
// garage 600 m away, and a ranking built on metres cannot express that. So every
// candidate is reduced to a single generalised cost in "equivalent minutes plus euros",
// which is comparable across garages, kerb gaps and park-and-ride alike.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "parkfit/fit/vehicle_fit.hpp"

namespace parkfit::rank {

/// Where an availability claim came from. Ordered weakest to strongest so that the
/// enum value doubles as a confidence tier; the resolver in the Python layer relies
/// on this ordering when several sources disagree about the same facility.
enum class EvidenceSource : std::uint8_t {
    OsmOnly = 0,        ///< Inferred from OpenStreetMap tags alone.
    StaticDatabase = 1, ///< Published capacity, no live signal.
    PredictiveModel = 2,
    UserConfirmation = 3,
    MunicipalSensor = 4,
    CameraObservation = 5,
    OperatorFeed = 6,   ///< The operator itself reporting its own free spaces.
};

inline const char* to_string(EvidenceSource s) {
    switch (s) {
        case EvidenceSource::OsmOnly: return "OSM_ONLY";
        case EvidenceSource::StaticDatabase: return "STATIC_DATABASE";
        case EvidenceSource::PredictiveModel: return "PREDICTIVE_MODEL";
        case EvidenceSource::UserConfirmation: return "USER_CONFIRMATION";
        case EvidenceSource::MunicipalSensor: return "MUNICIPAL_SENSOR";
        case EvidenceSource::CameraObservation: return "CAMERA_OBSERVATION";
        case EvidenceSource::OperatorFeed: return "OPERATOR_FEED";
    }
    return "STATIC_DATABASE";
}

/// What the user is allowed to be told about a candidate. Deriving this centrally,
/// rather than letting each surface invent its own wording, is what keeps the app from
/// quietly promising more than the evidence supports.
enum class ConfidenceLabel : std::uint8_t {
    CameraConfirmed,
    ReportedByOperator,
    LikelyAvailable,
    StaticInformationOnly,
    DataStale,
};

inline const char* to_string(ConfidenceLabel c) {
    switch (c) {
        case ConfidenceLabel::CameraConfirmed: return "CAMERA_CONFIRMED";
        case ConfidenceLabel::ReportedByOperator: return "AVAILABILITY_REPORTED_BY_OPERATOR";
        case ConfidenceLabel::LikelyAvailable: return "LIKELY_AVAILABLE";
        case ConfidenceLabel::StaticInformationOnly: return "STATIC_INFORMATION_ONLY";
        case ConfidenceLabel::DataStale: return "DATA_CURRENTLY_STALE";
    }
    return "STATIC_INFORMATION_ONLY";
}

/// Tunables for the generalised-cost model. Defaults are per-minute euro equivalents
/// calibrated so that one minute of driving costs about the same as 0.20 EUR, which is
/// roughly the Dutch statutory value of travel time for commuting.
struct ScoringConfig {
    double value_of_time_eur_per_min{0.20};

    /// What a failed recommendation actually costs the driver.
    ///
    /// Not just the wasted approach: you arrive, discover the space is gone, re-enter
    /// traffic, re-route, and start searching again -- typically in the busiest part of
    /// the city, since that is where marginal recommendations live. Observed parking
    /// search times in dense European centres run 8-15 minutes, and a failure puts you
    /// back at the start of that distribution rather than at its mean. Setting this too
    /// low makes the ranking gamble on long-shot kerb spaces; too high and it never
    /// recommends on-street parking at all.
    double failure_penalty_min{14.0};
    /// Walking is experienced as more costly than sitting in a car, minute for minute.
    double walk_discomfort_multiplier{1.35};

    double tight_fit_penalty_eur{1.50};
    double unverified_fit_penalty_eur{2.50};

    /// Charged against candidates whose evidence is weak or old.
    double staleness_penalty_eur_per_min{0.05};
    double max_staleness_penalty_eur{3.00};

    /// Observations older than this are never presented as live.
    double stale_after_s{300.0};

    /// Anti-herding: each additional active user recently pointed at the same exact
    /// space multiplies its survival probability down. Without this the app manufactures
    /// its own congestion by sending everyone to the one space it can see.
    double herding_decay_per_recommendation{0.55};

    /// Exact single-space recommendations expire quickly; a kerb space seen 60 s ago
    /// is not a space, it is a memory.
    double exact_space_ttl_s{45.0};
};

/// A parking option being considered, with everything needed to price it.
struct Candidate {
    std::string id;
    std::string group_key;  ///< Street or facility identity, used for diversification.

    double drive_time_min{0.0};
    double walk_time_min{0.0};
    double price_eur{0.0};

    /// Probability the space is free right now, given the evidence.
    double p_available_now{0.0};
    /// Per-minute rate at which visible vacancies disappear on this segment.
    double lambda_per_min{0.0};
    /// Minutes until the driver actually arrives.
    double eta_min{0.0};
    /// Age of the observation backing p_available_now.
    double observation_age_s{0.0};

    EvidenceSource evidence{EvidenceSource::StaticDatabase};
    fit::Verdict fit_verdict{fit::Verdict::Unverified};

    /// How many other active users have recently been shown this same exact space.
    int recent_recommendation_count{0};
    /// True for a single identified space (a kerb gap or one marked bay), as opposed
    /// to a facility with many interchangeable spaces.
    bool is_exact_space{false};
};

/// The scored result, carrying its own explanation.
struct ScoredCandidate {
    std::string id;
    double generalised_cost{0.0};
    double expected_time_min{0.0};
    double p_available_at_eta{0.0};
    ConfidenceLabel confidence{ConfidenceLabel::StaticInformationOnly};
    bool expired{false};
    double fit_penalty_eur{0.0};
    double uncertainty_penalty_eur{0.0};
};

/// Survival probability of a vacancy over the time until arrival.
///
///   P(available at ETA) = P(now) * exp(-lambda * t)
///
/// The exponential is the memoryless assumption: given the space is still free, the
/// chance it survives another minute does not depend on how long it has already been
/// free. That is a simplification -- a space free at 03:00 is far more durable than one
/// free at 17:30 -- which is exactly why lambda is estimated per street segment, per
/// weekday and per 15-minute bucket rather than being a single global constant.
inline double survival_probability(double p_now, double lambda_per_min, double eta_min) {
    if (p_now <= 0.0) return 0.0;
    const double lam = std::max(0.0, lambda_per_min);
    const double t = std::max(0.0, eta_min);
    return std::clamp(p_now * std::exp(-lam * t), 0.0, 1.0);
}

/// Reduce a probability to account for other drivers we have already sent here.
inline double apply_anti_herding(double p, int recent_recommendations, double decay) {
    if (recent_recommendations <= 0) return p;
    const double d = std::clamp(decay, 0.0, 1.0);
    return p * std::pow(d, static_cast<double>(recent_recommendations));
}

/// Choose the wording the user is allowed to see for this candidate.
inline ConfidenceLabel label_for(const Candidate& c, const ScoringConfig& cfg) {
    const bool live_source = c.evidence == EvidenceSource::CameraObservation ||
                             c.evidence == EvidenceSource::OperatorFeed ||
                             c.evidence == EvidenceSource::MunicipalSensor;
    if (live_source && c.observation_age_s > cfg.stale_after_s) return ConfidenceLabel::DataStale;

    switch (c.evidence) {
        case EvidenceSource::CameraObservation: return ConfidenceLabel::CameraConfirmed;
        case EvidenceSource::OperatorFeed:
        case EvidenceSource::MunicipalSensor: return ConfidenceLabel::ReportedByOperator;
        case EvidenceSource::PredictiveModel:
        case EvidenceSource::UserConfirmation: return ConfidenceLabel::LikelyAvailable;
        case EvidenceSource::StaticDatabase:
        case EvidenceSource::OsmOnly: return ConfidenceLabel::StaticInformationOnly;
    }
    return ConfidenceLabel::StaticInformationOnly;
}

/// Price one candidate.
///
///   E[T] = T_drive + w * T_walk + (1 - P_eta) * T_failure
///   G    = v_t * E[T] + C_parking + R_fit + R_uncertainty
inline ScoredCandidate score(const Candidate& c, const ScoringConfig& cfg) {
    ScoredCandidate out;
    out.id = c.id;

    double p = survival_probability(c.p_available_now, c.lambda_per_min, c.eta_min);
    p = apply_anti_herding(p, c.recent_recommendation_count, cfg.herding_decay_per_recommendation);

    // An exact space whose observation has aged past its time-to-live is no longer a
    // claim we can make. Drop its probability to zero rather than quietly decaying it,
    // so it can never win a ranking on stale evidence.
    if (c.is_exact_space && c.observation_age_s > cfg.exact_space_ttl_s) {
        out.expired = true;
        p = 0.0;
    }
    out.p_available_at_eta = p;

    const double expected =
        c.drive_time_min + cfg.walk_discomfort_multiplier * c.walk_time_min +
        (1.0 - p) * cfg.failure_penalty_min;
    out.expected_time_min = expected;

    double fit_penalty = 0.0;
    if (c.fit_verdict == fit::Verdict::TightFit) fit_penalty = cfg.tight_fit_penalty_eur;
    if (c.fit_verdict == fit::Verdict::Unverified) fit_penalty = cfg.unverified_fit_penalty_eur;
    out.fit_penalty_eur = fit_penalty;

    // Uncertainty penalty grows with how old the evidence is, capped so that a very old
    // observation degrades to "no better than static" rather than to minus infinity.
    double staleness = 0.0;
    if (c.evidence >= EvidenceSource::MunicipalSensor) {
        staleness = std::min(cfg.max_staleness_penalty_eur,
                             cfg.staleness_penalty_eur_per_min * (c.observation_age_s / 60.0));
    } else if (c.evidence <= EvidenceSource::StaticDatabase) {
        staleness = cfg.max_staleness_penalty_eur * 0.5;
    }
    out.uncertainty_penalty_eur = staleness;

    out.generalised_cost =
        cfg.value_of_time_eur_per_min * expected + c.price_eur + fit_penalty + staleness;
    out.confidence = label_for(c, cfg);
    return out;
}

/// Score, sort, and spread the results across distinct streets or facilities.
///
/// Diversification is not cosmetic. Three kerb gaps on one street share a single failure
/// mode -- if the street is actually full, all three are wrong together. Forcing variety
/// into the top slots means the backup option fails independently of the primary.
inline std::vector<ScoredCandidate> rank_and_diversify(const std::vector<Candidate>& candidates,
                                                       const ScoringConfig& cfg,
                                                       std::size_t max_results = 10,
                                                       std::size_t max_per_group = 2) {
    std::vector<std::pair<ScoredCandidate, const Candidate*>> scored;
    scored.reserve(candidates.size());
    for (const auto& c : candidates) scored.emplace_back(score(c, cfg), &c);

    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.first.generalised_cost < b.first.generalised_cost;
    });

    std::vector<ScoredCandidate> out;
    std::vector<std::pair<std::string, std::size_t>> group_counts;
    out.reserve(std::min(max_results, scored.size()));

    // Two passes: first honour the per-group cap, then backfill from what was held back
    // so we never return fewer results than we could just because of diversification.
    for (int pass = 0; pass < 2 && out.size() < max_results; ++pass) {
        for (const auto& [s, c] : scored) {
            if (out.size() >= max_results) break;
            if (std::any_of(out.begin(), out.end(),
                            [&](const ScoredCandidate& e) { return e.id == s.id; })) {
                continue;
            }
            if (pass == 0) {
                const std::string& key = c->group_key;
                auto it = std::find_if(group_counts.begin(), group_counts.end(),
                                       [&](const auto& g) { return g.first == key; });
                if (it != group_counts.end() && it->second >= max_per_group) continue;
                if (it == group_counts.end()) {
                    group_counts.emplace_back(key, 1);
                } else {
                    ++it->second;
                }
            }
            out.push_back(s);
        }
    }
    return out;
}

}  // namespace parkfit::rank
