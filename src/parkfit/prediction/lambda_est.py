"""Estimating how fast a free space disappears.

``lambda`` is the rate at which a vacant space is taken, per minute. The ranking model
uses it for the survival term ``P(still free at arrival) = P(free now) * exp(-lambda * t)``,
which means a wrong lambda does not just add noise, it systematically misprices every
option whose drive time is long.

Three things make this harder than dividing observations by time.

**Censoring.** The obvious estimator, ``1 / mean(observed vacant dwell)``, is wrong. A
vacant interval that is still vacant when the observation window closes is right-censored:
you know it lasted *at least* that long, not how long. Averaging only the intervals that
completed throws away exactly the long survivals and biases the rate upward, and the
bias is worst on quiet streets, where long survivals are the whole point. The maximum
likelihood estimator for an exponential rate under right-censoring is::

    lambda_hat = (number of take events) / (total time spent vacant)

Both numerator and denominator come from :class:`TransitionCounts`, which accumulates
them separately for this reason.

**Sparsity.** The schema keys decay on (target, weekday, quarter-hour). That is 672 cells
per target, and three weeks of history puts about three observations in each. An estimate
from three observations is noise wearing a number's clothing. So estimation happens on a
coarse grid, 4 weekday types x 24 hours, 96 cells, and the fine grid is *interpolated*
from it, never estimated directly.

**Empty cells.** Even coarsened, plenty of cells have no events at all, and ``0 / exposure``
would claim a space stays free forever. Each cell is shrunk toward a pooled rate for its
kind with a Gamma-conjugate posterior mean::

    lambda_hat = (k + events) / (k / lambda_pool + exposure)

``k`` is the prior strength in pseudo-events. A cell with plenty of data barely moves; a
cell with none returns the pooled rate exactly. There is no special case for "no data",
which is what makes it safe to run over every target.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.prediction.demand import vacancy_lambda
from parkfit.prediction.history import (
    SYNTHETIC_SOURCE,
    WEEKDAY_TYPES,
    SimulatedTarget,
    TransitionCounts,
    delete_in_chunks,
    weekday_type,
)
from parkfit.storage.models import AvailabilityObservation, OccupancyState, SegmentDynamics

log = logging.getLogger(__name__)

#: Prior strength, in pseudo-events. Two is deliberately weak: it lets a cell with a
#: dozen real events speak for itself while still rescuing one with none.
PRIOR_STRENGTH = 2.0

#: Rates outside this range are not estimates, they are artefacts. 0.002/min is a mean
#: survival of eight hours; 0.5/min is two minutes.
LAMBDA_MIN = 0.002
LAMBDA_MAX = 0.5


@dataclass
class EstimateReport:
    targets: int = 0
    rows_written: int = 0
    cells_with_events: int = 0
    cells_total: int = 0
    pooled_lambda: dict[str, float] | None = None
    mean_absolute_error: float | None = None
    coarse_sampling_error: float | None = None

    def describe(self) -> str:
        cover = 0.0 if not self.cells_total else 100.0 * self.cells_with_events / self.cells_total
        parts = [
            f"{self.rows_written:,} decay rows for {self.targets:,} targets",
            f"{cover:.0f}% of coarse cells had at least one event",
        ]
        if self.mean_absolute_error is not None:
            parts.append(f"mean |error| vs truth {self.mean_absolute_error:.4f}/min")
        return "; ".join(parts)


def _pool(counts: dict[tuple[str, int], TransitionCounts]) -> dict[str, np.ndarray]:
    """Pooled rate per target kind, on the coarse grid.

    Pooling is by kind rather than globally because a garage and a kerb bay differ by an
    order of magnitude, and a pool that mixes them helps neither.
    """
    events: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((WEEKDAY_TYPES, 24)))
    exposure: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((WEEKDAY_TYPES, 24)))
    for (kind, _), c in counts.items():
        events[kind] += c.events
        exposure[kind] += c.exposure_min

    pooled: dict[str, np.ndarray] = {}
    for kind in events:
        # The pool itself is shrunk toward its own grand mean, so a quiet hour with no
        # events anywhere does not hand back a zero for every target at once.
        total_rate = events[kind].sum() / max(1.0, exposure[kind].sum())
        pooled[kind] = np.clip(
            (PRIOR_STRENGTH + events[kind])
            / (PRIOR_STRENGTH / max(total_rate, LAMBDA_MIN) + exposure[kind]),
            LAMBDA_MIN,
            LAMBDA_MAX,
        )
    return pooled


def _shrink(counts: TransitionCounts, pooled: np.ndarray) -> np.ndarray:
    """Posterior-mean rate for one target on the coarse grid."""
    return np.clip(
        (PRIOR_STRENGTH + counts.events)
        / (PRIOR_STRENGTH / np.maximum(pooled, LAMBDA_MIN) + counts.exposure_min),
        LAMBDA_MIN,
        LAMBDA_MAX,
    )


def _expand_to_quarter_hours(coarse: np.ndarray) -> np.ndarray:
    """Lift a 4x24 estimate to the 7x96 grid the schema stores.

    Interpolated rather than held constant across the hour, because a step change at
    every hour boundary would show up directly in the ranking: two searches a minute
    apart would price the same space differently for no reason a driver could observe.

    Hourly estimates are treated as the value at the hour's midpoint and interpolated
    circularly, so 23:45 blends into 00:15 rather than falling off a cliff at midnight.
    """
    fine = np.empty((7, 96), dtype=np.float64)
    quarter_centres = np.arange(96) * 15.0 + 7.5
    hour_positions = quarter_centres / 60.0 - 0.5  # hour midpoints sit at h + 0.5

    lo = np.floor(hour_positions).astype(int) % 24
    hi = (lo + 1) % 24
    frac = hour_positions - np.floor(hour_positions)

    for weekday in range(7):
        row = coarse[weekday_type(weekday)]
        fine[weekday] = row[lo] * (1.0 - frac) + row[hi] * frac
    return fine


def _occupancy_grid(
    session: Session, keys: list[tuple[str, int]], *, source_name: str | None
) -> dict[tuple[str, int], np.ndarray]:
    """Observed occupancy fraction per (weekday type, hour), from persisted samples.

    This is the marginal, not the rate, and 15-minute sampling estimates it perfectly
    well, which is the whole reason the two quantities are computed from different
    resolutions.
    """
    if not keys:
        return {}
    ids = [k[1] for k in keys]
    kinds = {k[0] for k in keys}

    stmt = select(
        AvailabilityObservation.target_kind,
        AvailabilityObservation.target_id,
        AvailabilityObservation.observed_at,
        AvailabilityObservation.state,
    ).where(
        AvailabilityObservation.target_kind.in_(kinds),
        AvailabilityObservation.target_id.in_(ids),
    )
    if source_name:
        stmt = stmt.where(AvailabilityObservation.source_name == source_name)

    occupied: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros((WEEKDAY_TYPES, 24)))
    total: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros((WEEKDAY_TYPES, 24)))

    wanted = set(keys)
    for kind, tid, observed_at, state in session.execute(stmt).yield_per(20000):
        key = (kind, tid)
        if key not in wanted or observed_at is None:
            continue
        wt = weekday_type(observed_at.weekday())
        hour = observed_at.hour
        total[key][wt, hour] += 1.0
        if state == OccupancyState.OCCUPIED.value:
            occupied[key][wt, hour] += 1.0

    return {key: occupied[key] / np.maximum(total[key], 1.0) for key in wanted if key in total}


def counts_from_observations(
    session: Session,
    keys: list[tuple[str, int]],
    *,
    source_name: str | None = SYNTHETIC_SOURCE,
) -> dict[tuple[str, int], TransitionCounts]:
    """Reconstruct transition counts from persisted observations.

    This is the path real data has to take: nobody hands you the transitions, only the
    samples. It is included because it is the honest production estimator, and because
    comparing it against the simulation's own fine-grained counts *measures* how much a
    given polling interval costs you. See :func:`estimate_and_store`, which reports that
    difference rather than hiding it.
    """
    if not keys:
        return {}
    ids = [k[1] for k in keys]
    kinds = {k[0] for k in keys}

    stmt = (
        select(
            AvailabilityObservation.target_kind,
            AvailabilityObservation.target_id,
            AvailabilityObservation.observed_at,
            AvailabilityObservation.state,
        )
        .where(
            AvailabilityObservation.target_kind.in_(kinds),
            AvailabilityObservation.target_id.in_(ids),
        )
        .order_by(AvailabilityObservation.target_id, AvailabilityObservation.observed_at)
    )
    if source_name:
        stmt = stmt.where(AvailabilityObservation.source_name == source_name)

    wanted = set(keys)
    counts: dict[tuple[str, int], TransitionCounts] = {}
    previous: dict[tuple[str, int], tuple[datetime, str]] = {}

    for kind, tid, observed_at, state in session.execute(stmt).yield_per(20000):
        key = (kind, tid)
        if key not in wanted or observed_at is None:
            continue
        counts.setdefault(key, TransitionCounts())
        prior = previous.get(key)
        previous[key] = (observed_at, state)
        if prior is None:
            continue

        prior_at, prior_state = prior
        gap_min = (observed_at - prior_at).total_seconds() / 60.0
        # A gap far longer than the polling interval means the feed dropped out. Counting
        # it as exposure would claim the space sat vacant through an outage nobody saw.
        if gap_min <= 0 or gap_min > 120:
            continue
        if prior_state != OccupancyState.VACANT.value:
            continue

        wt = weekday_type(prior_at.weekday())
        hour = prior_at.hour
        counts[key].exposure_min[wt, hour] += gap_min
        if state == OccupancyState.OCCUPIED.value:
            counts[key].events[wt, hour] += 1.0

    return counts


def estimate_and_store(
    session: Session,
    counts: dict[tuple[str, int], TransitionCounts],
    *,
    truth: dict[tuple[str, int], SimulatedTarget] | None = None,
    replace: bool = True,
    observation_source: str | None = SYNTHETIC_SOURCE,
) -> EstimateReport:
    """Estimate decay rates and write them to ``segment_dynamics``."""
    report = EstimateReport(targets=len(counts))
    if not counts:
        return report

    pooled = _pool(counts)
    report.pooled_lambda = {k: float(v.mean()) for k, v in pooled.items()}
    report.cells_total = len(counts) * WEEKDAY_TYPES * 24
    report.cells_with_events = int(sum(float((c.events > 0).sum()) for c in counts.values()))

    keys = list(counts.keys())
    occupancy = _occupancy_grid(session, keys, source_name=observation_source)

    if replace:
        delete_in_chunks(
            session,
            SegmentDynamics,
            SegmentDynamics.target_id.in_([k[1] for k in keys]),
        )

    errors: list[float] = []
    pending: list[dict] = []
    now = datetime.now(UTC).replace(tzinfo=None)

    for key, c in counts.items():
        coarse = _shrink(c, pooled[key[0]])
        fine = _expand_to_quarter_hours(coarse)
        occ_coarse = occupancy.get(key)
        occ_fine = (
            _expand_to_quarter_hours(occ_coarse)
            if occ_coarse is not None
            else np.full((7, 96), 0.75)
        )
        samples = int(c.exposure_min.sum())

        if truth is not None and key in truth:
            target = truth[key]
            # Compare against the rate that actually generated the data, at a spread of
            # times, rather than at one convenient hour.
            for weekday in range(7):
                for quarter in range(0, 96, 8):
                    minute_of_day = quarter * 15
                    actual = vacancy_lambda(target.profile, weekday, minute_of_day)
                    errors.append(abs(fine[weekday, quarter] - actual))

        for weekday in range(7):
            for quarter in range(96):
                pending.append(
                    {
                        "target_kind": key[0],
                        "target_id": key[1],
                        "weekday": weekday,
                        "quarter_hour": quarter,
                        "lambda_per_min": float(fine[weekday, quarter]),
                        "base_occupancy": float(occ_fine[weekday, quarter]),
                        "sample_count": samples,
                        "updated_at": now,
                    }
                )

        if len(pending) >= 8000:
            session.bulk_insert_mappings(SegmentDynamics, pending)
            session.commit()
            report.rows_written += len(pending)
            pending.clear()

    if pending:
        session.bulk_insert_mappings(SegmentDynamics, pending)
        session.commit()
        report.rows_written += len(pending)

    if errors:
        report.mean_absolute_error = float(np.mean(errors))
    return report


def measure_sampling_cost(
    session: Session,
    simulated: dict[tuple[str, int], SimulatedTarget],
    *,
    source_name: str | None = SYNTHETIC_SOURCE,
) -> dict[str, float]:
    """How much decay-rate accuracy a coarse polling interval costs.

    The simulation counted every transition at one-minute resolution. The persisted
    observations were sampled far more sparsely. Estimating from each and comparing is a
    direct measurement of what the polling interval throws away, and it is worth printing
    rather than assuming: it is the number that says whether a 15-minute municipal feed
    can support a survival model at all.
    """
    keys = list(simulated.keys())
    coarse_counts = counts_from_observations(session, keys, source_name=source_name)
    if not coarse_counts:
        return {}

    fine_pooled = _pool({k: v.counts for k, v in simulated.items()})
    coarse_pooled = _pool(coarse_counts)

    ratios: list[float] = []
    fine_rates: list[float] = []
    coarse_rates: list[float] = []
    for key, target in simulated.items():
        if key not in coarse_counts:
            continue
        fine = _shrink(target.counts, fine_pooled[key[0]])
        crs = _shrink(coarse_counts[key], coarse_pooled[key[0]])
        mask = target.counts.exposure_min > 30  # cells with real support only
        if not mask.any():
            continue
        fine_rates.append(float(fine[mask].mean()))
        coarse_rates.append(float(crs[mask].mean()))
        ratios.append(float((crs[mask] / np.maximum(fine[mask], 1e-9)).mean()))

    if not ratios:
        return {}
    return {
        "fine_lambda_mean": float(np.mean(fine_rates)),
        "coarse_lambda_mean": float(np.mean(coarse_rates)),
        "coarse_over_fine": float(np.mean(ratios)),
        "targets_compared": float(len(ratios)),
    }


def lambda_lookup(
    session: Session, keys: list[tuple[str, int]], when: datetime
) -> dict[tuple[str, int], float]:
    """Stored decay rate for these targets at this moment. Used by search."""
    if not keys:
        return {}
    weekday = when.weekday()
    quarter = (when.hour * 60 + when.minute) // 15
    rows = session.execute(
        select(SegmentDynamics).where(
            SegmentDynamics.target_kind.in_({k[0] for k in keys}),
            SegmentDynamics.target_id.in_([k[1] for k in keys]),
            SegmentDynamics.weekday == weekday,
            SegmentDynamics.quarter_hour == quarter,
        )
    ).scalars()
    return {(r.target_kind, r.target_id): r.lambda_per_min for r in rows}


def default_window(days: int = 21) -> tuple[datetime, datetime]:
    end = datetime.now(UTC)
    return end - timedelta(days=days), end
