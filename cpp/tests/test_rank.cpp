// SPDX-License-Identifier: MIT
//
// Ranking tests.
//
// These encode the product judgement calls: that a probable space beats a distant
// certain one only up to a point, that stale evidence is discounted, that the app does
// not send every driver to the same kerb gap, and that an expired exact-space
// observation is dropped outright rather than merely faded.

#include "test_framework.hpp"

#include <cmath>

#include "parkfit/rank/score.hpp"

using namespace parkfit::rank;
using parkfit::fit::Verdict;

namespace {

Candidate base_candidate(const char* id, const char* group = "g") {
    Candidate c;
    c.id = id;
    c.group_key = group;
    c.drive_time_min = 8.0;
    c.walk_time_min = 4.0;
    c.price_eur = 3.0;
    c.p_available_now = 1.0;
    c.lambda_per_min = 0.0;
    c.eta_min = 8.0;
    c.observation_age_s = 5.0;
    c.evidence = EvidenceSource::OperatorFeed;
    c.fit_verdict = Verdict::Fits;
    return c;
}

}  // namespace

TEST_CASE("survival: no decay leaves the probability untouched") {
    CHECK_NEAR(survival_probability(0.8, 0.0, 30.0), 0.8, 1e-12);
}

TEST_CASE("survival: probability decays exponentially with time to arrival") {
    // lambda = 0.1/min over 10 min -> factor e^-1
    CHECK_NEAR(survival_probability(1.0, 0.1, 10.0), std::exp(-1.0), 1e-12);
    CHECK_NEAR(survival_probability(0.5, 0.2, 5.0), 0.5 * std::exp(-1.0), 1e-12);
}

TEST_CASE("survival: a longer drive erodes a kerb space more than a short one") {
    const double near = survival_probability(0.9, 0.08, 3.0);
    const double far = survival_probability(0.9, 0.08, 25.0);
    CHECK(near > far);
    CHECK(far < 0.15);
}

TEST_CASE("survival: never returns a value outside [0, 1]") {
    CHECK_NEAR(survival_probability(0.0, 0.5, 10.0), 0.0, 1e-12);
    CHECK(survival_probability(1.0, -5.0, 10.0) <= 1.0);
    CHECK(survival_probability(1.0, 0.1, -10.0) <= 1.0);
}

TEST_CASE("anti-herding: each extra driver sent here lowers the odds") {
    const double p0 = apply_anti_herding(0.9, 0, 0.55);
    const double p1 = apply_anti_herding(0.9, 1, 0.55);
    const double p3 = apply_anti_herding(0.9, 3, 0.55);
    CHECK_NEAR(p0, 0.9, 1e-12);
    CHECK_NEAR(p1, 0.9 * 0.55, 1e-12);
    CHECK_NEAR(p3, 0.9 * 0.55 * 0.55 * 0.55, 1e-12);
    CHECK(p3 < p1);
}

TEST_CASE("scoring: a guaranteed space beats an unlikely one at equal travel time") {
    ScoringConfig cfg;
    Candidate sure = base_candidate("sure");
    Candidate unlikely = base_candidate("unlikely");
    unlikely.p_available_now = 0.2;

    const ScoredCandidate a = score(sure, cfg);
    const ScoredCandidate b = score(unlikely, cfg);
    CHECK(a.generalised_cost < b.generalised_cost);
}

TEST_CASE("scoring: walking is weighted more heavily than driving") {
    ScoringConfig cfg;
    Candidate drive = base_candidate("drive");
    drive.drive_time_min = 12.0;
    drive.walk_time_min = 2.0;
    Candidate walk = base_candidate("walk");
    walk.drive_time_min = 2.0;
    walk.walk_time_min = 12.0;

    // Same 14 minutes total, but the one that makes you walk should cost more.
    CHECK(score(walk, cfg).generalised_cost > score(drive, cfg).generalised_cost);
}

TEST_CASE("scoring: a close but uncertain kerb space can lose to a far certain garage") {
    ScoringConfig cfg;
    Candidate kerb = base_candidate("kerb", "street_a");
    kerb.drive_time_min = 3.0;
    kerb.walk_time_min = 1.0;
    kerb.price_eur = 4.0;
    kerb.p_available_now = 0.85;
    kerb.lambda_per_min = 0.25;   // busy street, spaces evaporate
    kerb.eta_min = 12.0;
    kerb.evidence = EvidenceSource::CameraObservation;
    kerb.is_exact_space = true;

    Candidate garage = base_candidate("garage", "garage_b");
    garage.drive_time_min = 7.0;
    garage.walk_time_min = 5.0;
    garage.price_eur = 4.5;
    garage.p_available_now = 0.99;

    // e^(-0.25*12) is about 0.05, so the kerb space is almost certainly gone by the
    // time the driver arrives. Even though it is nearer and cheaper on paper, the
    // 95 % chance of a wasted trip has to outweigh four minutes and one euro.
    CHECK(score(kerb, cfg).p_available_at_eta < 0.06);
    CHECK(score(garage, cfg).generalised_cost < score(kerb, cfg).generalised_cost);
}

TEST_CASE("scoring: a near-certain kerb space still beats a distant garage") {
    // The mirror image of the case above. If the model always preferred garages it
    // would be useless for on-street parking, so a fresh, high-probability kerb space
    // on a quiet street must win.
    ScoringConfig cfg;
    Candidate kerb = base_candidate("kerb", "street_a");
    kerb.drive_time_min = 3.0;
    kerb.walk_time_min = 1.0;
    kerb.price_eur = 4.0;
    kerb.p_available_now = 0.95;
    kerb.lambda_per_min = 0.01;  // quiet residential street
    kerb.eta_min = 3.0;
    kerb.evidence = EvidenceSource::CameraObservation;
    kerb.is_exact_space = true;
    kerb.observation_age_s = 5.0;

    Candidate garage = base_candidate("garage", "garage_b");
    garage.drive_time_min = 7.0;
    garage.walk_time_min = 5.0;
    garage.price_eur = 4.5;
    garage.p_available_now = 0.99;

    CHECK(score(kerb, cfg).p_available_at_eta > 0.9);
    CHECK(score(kerb, cfg).generalised_cost < score(garage, cfg).generalised_cost);
}

TEST_CASE("scoring: an expired exact-space observation is dropped, not faded") {
    ScoringConfig cfg;
    Candidate stale = base_candidate("stale");
    stale.is_exact_space = true;
    stale.evidence = EvidenceSource::CameraObservation;
    stale.observation_age_s = cfg.exact_space_ttl_s + 1.0;

    const ScoredCandidate s = score(stale, cfg);
    CHECK(s.expired);
    CHECK_NEAR(s.p_available_at_eta, 0.0, 1e-12);
}

TEST_CASE("scoring: a facility with many spaces does not expire like a single space") {
    ScoringConfig cfg;
    Candidate garage = base_candidate("garage");
    garage.is_exact_space = false;
    garage.observation_age_s = cfg.exact_space_ttl_s + 60.0;
    const ScoredCandidate s = score(garage, cfg);
    CHECK(!s.expired);
    CHECK(s.p_available_at_eta > 0.5);
}

TEST_CASE("scoring: an unverified fit is penalised more than a tight one") {
    ScoringConfig cfg;
    Candidate ok = base_candidate("ok");
    Candidate tight = base_candidate("tight");
    tight.fit_verdict = Verdict::TightFit;
    Candidate unknown = base_candidate("unknown");
    unknown.fit_verdict = Verdict::Unverified;

    const double c_ok = score(ok, cfg).generalised_cost;
    const double c_tight = score(tight, cfg).generalised_cost;
    const double c_unknown = score(unknown, cfg).generalised_cost;
    CHECK(c_ok < c_tight);
    CHECK(c_tight < c_unknown);
}

TEST_CASE("labels: live evidence that has aged out reads as stale") {
    ScoringConfig cfg;
    Candidate fresh = base_candidate("fresh");
    fresh.evidence = EvidenceSource::CameraObservation;
    fresh.observation_age_s = 10.0;
    CHECK(label_for(fresh, cfg) == ConfidenceLabel::CameraConfirmed);

    Candidate old = fresh;
    old.observation_age_s = cfg.stale_after_s + 1.0;
    CHECK(label_for(old, cfg) == ConfidenceLabel::DataStale);
}

TEST_CASE("labels: static data is never dressed up as a live claim") {
    ScoringConfig cfg;
    Candidate c = base_candidate("static");
    c.evidence = EvidenceSource::StaticDatabase;
    c.observation_age_s = 0.0;
    CHECK(label_for(c, cfg) == ConfidenceLabel::StaticInformationOnly);

    c.evidence = EvidenceSource::OsmOnly;
    CHECK(label_for(c, cfg) == ConfidenceLabel::StaticInformationOnly);

    c.evidence = EvidenceSource::PredictiveModel;
    CHECK(label_for(c, cfg) == ConfidenceLabel::LikelyAvailable);
}

TEST_CASE("evidence ordering encodes the source-priority ladder") {
    CHECK(EvidenceSource::OperatorFeed > EvidenceSource::CameraObservation);
    CHECK(EvidenceSource::CameraObservation > EvidenceSource::MunicipalSensor);
    CHECK(EvidenceSource::MunicipalSensor > EvidenceSource::UserConfirmation);
    CHECK(EvidenceSource::UserConfirmation > EvidenceSource::PredictiveModel);
    CHECK(EvidenceSource::PredictiveModel > EvidenceSource::StaticDatabase);
    CHECK(EvidenceSource::StaticDatabase > EvidenceSource::OsmOnly);
}

TEST_CASE("diversify: the top results are spread across different streets") {
    ScoringConfig cfg;
    std::vector<Candidate> cands;
    // Five excellent options all on the same street, plus one mediocre option elsewhere.
    for (int i = 0; i < 5; ++i) {
        Candidate c = base_candidate(("same_" + std::to_string(i)).c_str(), "street_a");
        c.drive_time_min = 2.0 + 0.1 * i;
        cands.push_back(c);
    }
    Candidate other = base_candidate("other", "street_b");
    other.drive_time_min = 7.0;
    cands.push_back(other);

    const auto ranked = rank_and_diversify(cands, cfg, 3, 2);
    CHECK_EQ(ranked.size(), static_cast<std::size_t>(3));
    // With a cap of two per street, the third slot has to come from street_b, so the
    // backup option fails independently of the primary rather than alongside it.
    CHECK(ranked[2].id == "other");
}

TEST_CASE("diversify: backfills rather than returning fewer results than requested") {
    ScoringConfig cfg;
    std::vector<Candidate> cands;
    for (int i = 0; i < 5; ++i) {
        cands.push_back(base_candidate(("only_" + std::to_string(i)).c_str(), "street_a"));
    }
    // Every candidate shares one group, so the cap alone would yield 2. We still want 4.
    const auto ranked = rank_and_diversify(cands, cfg, 4, 2);
    CHECK_EQ(ranked.size(), static_cast<std::size_t>(4));
}

TEST_CASE("diversify: results come back cheapest first") {
    ScoringConfig cfg;
    std::vector<Candidate> cands;
    for (int i = 0; i < 6; ++i) {
        Candidate c = base_candidate(("c" + std::to_string(i)).c_str(),
                                     ("g" + std::to_string(i)).c_str());
        c.drive_time_min = 10.0 - static_cast<double>(i);
        cands.push_back(c);
    }
    const auto ranked = rank_and_diversify(cands, cfg, 6, 2);
    for (std::size_t i = 1; i < ranked.size(); ++i) {
        CHECK(ranked[i - 1].generalised_cost <= ranked[i].generalised_cost);
    }
}

TEST_CASE("diversify: an empty candidate set is handled") {
    ScoringConfig cfg;
    const auto ranked = rank_and_diversify({}, cfg, 5, 2);
    CHECK(ranked.empty());
}

PF_TEST_MAIN()
