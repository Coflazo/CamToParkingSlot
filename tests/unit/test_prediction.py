"""Occupancy prediction: demand curves, decay-rate estimation, features, model.

The tests that matter most here are the ones about *estimation*, not plumbing. A decay
rate is easy to compute and easy to compute wrongly, censoring, empty cells and a grid
too fine to support an estimate each produce a confident number that is simply false, and
none of them raise. So each has a test that would fail if the estimator regressed to the
naive form.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from parkfit.domain.evidence import ResolvedAvailability
from parkfit.numeric import limit_numeric_threads
from parkfit.prediction import lambda_est
from parkfit.prediction.demand import (
    lambda_table,
    occupancy_rate,
    occupancy_table,
    profile_for,
    vacancy_lambda,
)
from parkfit.prediction.features import FEATURE_NAMES, TargetStatics, row
from parkfit.prediction.history import TransitionCounts, weekday_type
from parkfit.prediction.model import OccupancyModel, _auc, _brier
from parkfit.storage.models import EvidenceSource, OccupancyState

CENTRE = (52.3730, 4.8926)
OUTER = (52.3550, 4.8100)


def centre_bay():
    return profile_for(*CENTRE, is_facility=False, metered=True, capacity=None, seed_value=17)


def outer_bay():
    return profile_for(*OUTER, is_facility=False, metered=False, capacity=None, seed_value=17)


# ---------------------------------------------------------------------------
# demand model
# ---------------------------------------------------------------------------
def test_vectorised_occupancy_matches_the_scalar_form():
    """The numpy table and the scalar function must not drift apart.

    They exist separately only for speed; the table is what the simulation uses and the
    scalar is what the docstrings describe, so a divergence would be invisible.
    """
    profile = centre_bay()
    table = occupancy_table(profile)
    for weekday in range(7):
        for minute in (0, 137, 599, 720, 1021, 1439):
            assert table[weekday, minute] == pytest.approx(
                occupancy_rate(profile, weekday, minute), abs=1e-12
            )


def test_vectorised_lambda_matches_the_scalar_form():
    profile = outer_bay()
    table = lambda_table(profile)
    for weekday in (0, 5, 6):
        for minute in (0, 361, 900, 1439):
            assert table[weekday, minute] == pytest.approx(
                vacancy_lambda(profile, weekday, minute), abs=1e-12
            )


def test_occupancy_never_saturates_across_the_day():
    """Composing effects in log-odds keeps the diurnal signal alive.

    The earlier multiplicative form pinned a centre bay at 0.99 for most of the day, which
    is not merely inaccurate, it erases the pattern the learned model exists to find.
    """
    profile = centre_bay()
    values = [occupancy_rate(profile, 5, m) for m in range(0, 1440, 15)]
    assert max(values) < 0.99
    assert min(values) > 0.01
    assert max(values) - min(values) > 0.10


def test_residential_and_destination_streets_peak_at_opposite_times():
    """The interaction no per-target constant can express.

    An outer residential street is fullest overnight, when its residents are home. A
    centre street is fullest in the evening, when visitors arrive. This is the structure
    the occupancy model has to learn, so if it ever stopped holding the model's advantage
    over a per-target constant would be meaningless.
    """
    centre, outer = centre_bay(), outer_bay()
    night, evening = 3 * 60, 20 * 60

    assert occupancy_rate(outer, 5, night) > occupancy_rate(outer, 5, evening)
    assert occupancy_rate(centre, 5, evening) > occupancy_rate(centre, 5, night)


def test_metered_bays_are_more_contested_than_free_ones():
    metered = profile_for(*OUTER, is_facility=False, metered=True, capacity=None, seed_value=5)
    free = profile_for(*OUTER, is_facility=False, metered=False, capacity=None, seed_value=5)
    assert metered.baseline > free.baseline


def test_lambda_rises_with_occupancy():
    """A space on a full street is retaken faster than one on an empty street."""
    profile = centre_bay()
    busy = vacancy_lambda(profile, 5, 20 * 60)
    quiet = vacancy_lambda(profile, 6, 5 * 60)
    assert busy > quiet
    assert 1.0 / busy < 1.0 / quiet


def test_profiles_are_deterministic_in_their_seed():
    """An evaluation you cannot reproduce is an anecdote."""
    a = profile_for(*CENTRE, is_facility=False, metered=True, capacity=None, seed_value=99)
    b = profile_for(*CENTRE, is_facility=False, metered=True, capacity=None, seed_value=99)
    assert a == b


# ---------------------------------------------------------------------------
# decay-rate estimation
# ---------------------------------------------------------------------------
def test_weekday_types_group_monday_to_thursday():
    assert [weekday_type(d) for d in range(7)] == [0, 0, 0, 0, 1, 2, 3]


def test_rate_is_events_over_exposure_not_one_over_mean_dwell():
    """The censoring correction, stated as a test.

    Ten events over 1000 vacant minutes is 0.01/min. The naive estimator, one over the
    mean of the dwell times that happened to complete, ignores that most of those 1000
    minutes belong to intervals still running, and lands far higher. With a weak prior the
    correct estimator stays within a few percent of the true rate.
    """
    counts = TransitionCounts()
    counts.events[0, 12] = 10.0
    counts.exposure_min[0, 12] = 1000.0

    pooled = np.full((4, 24), 0.01)
    estimate = lambda_est._shrink(counts, pooled)[0, 12]
    assert estimate == pytest.approx(0.01, rel=0.05)


def test_a_cell_with_no_events_returns_the_pooled_rate():
    """Zero events over real exposure must not mean "free forever".

    ``0 / exposure`` is the answer that would make a quiet street look permanently
    available. Shrinkage returns the pool instead, with no special case for empty.
    """
    counts = TransitionCounts()
    counts.exposure_min[2, 4] = 0.0
    pooled = np.full((4, 24), 0.037)
    assert lambda_est._shrink(counts, pooled)[2, 4] == pytest.approx(0.037, rel=1e-6)


def test_plentiful_data_dominates_the_prior():
    """A cell with real support lands near its own rate, not near the pool.

    Not *exactly* at its own rate, and that is correct rather than a rounding artefact. A
    Gamma prior with mean ``lambda_pool`` carries pseudo-exposure ``k / lambda_pool``, so a
    prior concentrated at a very low rate implies a long imaginary observation and pulls
    harder. Here 400 events over 2000 minutes says 0.2/min, a pool of 0.01/min adds 200
    pseudo-minutes, and the posterior is 0.183, 91% of the way from the pool to the data.
    """
    counts = TransitionCounts()
    counts.events[1, 9] = 400.0
    counts.exposure_min[1, 9] = 2000.0  # 0.2/min, far from the pool
    pool = 0.01
    estimate = lambda_est._shrink(counts, np.full((4, 24), pool))[1, 9]

    assert abs(estimate - 0.2) < abs(estimate - pool)
    assert (estimate - pool) / (0.2 - pool) > 0.9


def test_estimates_stay_inside_the_plausible_band():
    counts = TransitionCounts()
    counts.events[0, 0] = 10_000.0
    counts.exposure_min[0, 0] = 1.0  # absurd rate
    pooled = np.full((4, 24), 0.05)
    estimate = lambda_est._shrink(counts, pooled)
    assert estimate.max() <= lambda_est.LAMBDA_MAX
    assert estimate.min() >= lambda_est.LAMBDA_MIN


def test_quarter_hour_expansion_wraps_around_midnight():
    """No cliff at 00:00, because a driver cannot see one.

    Two searches a minute either side of midnight must not price the same space
    differently for a reason that exists only in the storage grid.
    """
    coarse = np.zeros((4, 24))
    coarse[0] = np.linspace(0.01, 0.24, 24)
    fine = lambda_est._expand_to_quarter_hours(coarse)

    last, first = fine[0, 95], fine[0, 0]
    interior = np.abs(np.diff(fine[0]))
    assert abs(last - first) <= interior.max() * 1.5


def test_quarter_hour_expansion_is_monotone_where_the_hours_are():
    coarse = np.zeros((4, 24))
    coarse[0] = np.linspace(0.02, 0.30, 24)
    fine = lambda_est._expand_to_quarter_hours(coarse)
    # Between the 02:00 and 20:00 midpoints the hourly series only rises, so the
    # interpolation must too.
    stretch = fine[0, 10:80]
    assert np.all(np.diff(stretch) >= -1e-12)


def test_every_weekday_maps_onto_its_type_when_expanded():
    coarse = np.zeros((4, 24))
    for wtype in range(4):
        coarse[wtype] = wtype + 1.0
    fine = lambda_est._expand_to_quarter_hours(coarse)
    for weekday in range(7):
        assert fine[weekday, 40] == pytest.approx(weekday_type(weekday) + 1.0)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def test_feature_row_matches_the_declared_names():
    """A row of a different length than FEATURE_NAMES would silently shift every column."""
    from datetime import datetime

    statics = TargetStatics(
        key=("bay", 1),
        lat=CENTRE[0],
        lon=CENTRE[1],
        is_facility=False,
        metered=True,
        capacity=1.0,
        tariff_eur_per_hour=0.0,
        bay_length_cm=550.0,
        bay_width_cm=195.0,
        fill_ratio=0.97,
    )
    values = row(statics, datetime(2026, 8, 26, 14, 30))
    assert len(values) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in values)


def test_time_is_encoded_cyclically():
    """23:50 and 00:10 must be near neighbours, not opposite ends of a range."""
    from datetime import datetime

    statics = TargetStatics(
        key=("bay", 1),
        lat=CENTRE[0],
        lon=CENTRE[1],
        is_facility=False,
        metered=True,
        capacity=1.0,
        tariff_eur_per_hour=0.0,
        bay_length_cm=0.0,
        bay_width_cm=0.0,
        fill_ratio=1.0,
    )
    idx_sin = FEATURE_NAMES.index("hour_sin")
    idx_cos = FEATURE_NAMES.index("hour_cos")

    late = row(statics, datetime(2026, 8, 26, 23, 50))
    early = row(statics, datetime(2026, 8, 27, 0, 10))
    midday = row(statics, datetime(2026, 8, 26, 12, 0))

    def distance(a, b):
        return np.hypot(a[idx_sin] - b[idx_sin], a[idx_cos] - b[idx_cos])

    assert distance(late, early) < 0.1
    assert distance(late, midday) > 1.5


def test_distance_to_centre_is_derived_not_stored():
    statics = TargetStatics(
        key=("bay", 1),
        lat=OUTER[0],
        lon=OUTER[1],
        is_facility=False,
        metered=False,
        capacity=1.0,
        tariff_eur_per_hour=0.0,
        bay_length_cm=0.0,
        bay_width_cm=0.0,
        fill_ratio=1.0,
    )
    assert 5.0 < statics.km_to_centre < 8.0


# ---------------------------------------------------------------------------
# model scoring and degradation
# ---------------------------------------------------------------------------
def test_auc_of_a_constant_predictor_is_one_half():
    """Tied scores must average their ranks, or a constant would score a perfect 1.0."""
    labels = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    assert _auc(np.full(5, 0.7), labels) == pytest.approx(0.5)


def test_auc_of_a_perfect_predictor_is_one():
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert _auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)


def test_brier_rewards_calibration_not_just_ordering():
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    confident = np.array([0.95, 0.95, 0.05, 0.05])
    timid = np.array([0.55, 0.55, 0.45, 0.45])
    assert _brier(confident, labels) < _brier(timid, labels)


def test_an_absent_model_predicts_nothing_rather_than_guessing():
    """Degradation has to be explicit, or a missing file becomes a silent constant."""
    model = OccupancyModel()
    assert model.available is False
    assert model.probability_occupied([]) is None


def test_loading_a_missing_model_file_is_not_an_error(scratch_dir):
    model = OccupancyModel.load(scratch_dir / "does-not-exist.lgb")
    assert model.available is False


# ---------------------------------------------------------------------------
# how a prediction reaches the ranking
# ---------------------------------------------------------------------------
def _unknown(kind: str = "bay", metered: bool = True) -> ResolvedAvailability:
    return ResolvedAvailability(
        target_kind=kind,
        target_id=1,
        state=OccupancyState.UNKNOWN,
        evidence=EvidenceSource.STATIC_DATABASE,
        observed_at=None,
        age_s=float("inf"),
        confidence=0.0,
        metered=metered,
    )


def test_without_a_model_the_static_base_rate_applies():
    assert _unknown().prior == pytest.approx(ResolvedAvailability.PRIOR_SINGLE_METERED_BAY)
    assert _unknown(metered=False).prior == pytest.approx(
        ResolvedAvailability.PRIOR_SINGLE_FREE_BAY
    )
    assert _unknown("facility").prior == pytest.approx(ResolvedAvailability.PRIOR_FACILITY)


def test_a_model_prior_replaces_the_base_rate():
    from dataclasses import replace

    availability = replace(_unknown(), model_prior=0.42)
    assert availability.prior == pytest.approx(0.42)
    assert availability.probability_available == pytest.approx(0.42)


def test_a_model_prior_is_clamped_to_a_probability():
    from dataclasses import replace

    assert replace(_unknown(), model_prior=1.7).prior == pytest.approx(0.99)
    assert replace(_unknown(), model_prior=-0.3).prior == pytest.approx(0.01)


def test_a_model_prior_never_overrides_a_live_observation():
    """The ordering that keeps a prediction from displacing a measurement."""
    from datetime import UTC, datetime

    seen = ResolvedAvailability(
        target_kind="bay",
        target_id=1,
        state=OccupancyState.OCCUPIED,
        evidence=EvidenceSource.MUNICIPAL_SENSOR,
        observed_at=datetime.now(UTC),
        age_s=20.0,
        confidence=0.95,
        model_prior=0.9,
    )
    # The bay was seen occupied twenty seconds ago; an optimistic model must not turn
    # that into a nine-in-ten chance of being free.
    assert seen.probability_available == pytest.approx(0.02)


def test_predictive_model_outranks_the_static_register_and_nothing_else():
    assert EvidenceSource.PREDICTIVE_MODEL > EvidenceSource.STATIC_DATABASE
    assert EvidenceSource.PREDICTIVE_MODEL < EvidenceSource.USER_CONFIRMATION
    assert EvidenceSource.PREDICTIVE_MODEL < EvidenceSource.CAMERA_OBSERVATION


# ---------------------------------------------------------------------------
# numeric thread guard
# ---------------------------------------------------------------------------
def test_thread_limit_respects_an_explicit_setting(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    applied = limit_numeric_threads(1)
    assert "OPENBLAS_NUM_THREADS" not in applied
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_thread_limit_sets_unset_variables(monkeypatch):
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    applied = limit_numeric_threads(2)
    assert applied.get("MKL_NUM_THREADS") == "2"
