"""Simulating parking history, so the learned models have something to fit.

Occupancy is simulated as a two-state continuous-time Markov chain per target: a free
space is taken at rate ``lambda``, an occupied space is released at rate ``mu``. The two
rates are not independent parameters. The demand model already fixes both the stationary
occupancy ``p`` and the take rate ``lambda``, and a two-state chain in equilibrium
satisfies ``p = lambda / (lambda + mu)``, so::

    mu = lambda * (1 - p) / p

falls out. Choosing ``mu`` freely would let the simulation drift away from the occupancy
curve it is supposed to realise.

**Why two different time resolutions.**

The occupancy model and the decay-rate estimator need different things from this
simulation, and forcing both through one sampling interval breaks one of them.

*The decay-rate estimator needs transitions.* It counts how often a vacant space was
taken and divides by the time spent vacant. That only works while the sampling interval
is short relative to the dwell time. A centre bay on a Saturday evening has a mean vacant
dwell of about five minutes, so 15-minute polling sees ``V, V, V`` and misses two
complete turnovers in between. The chain equilibrates inside one sample and the estimate
collapses. This is a real property of the problem rather than an artefact of simulating
it; it is why municipal bay sensors report every minute.

*The occupancy model needs marginals.* It learns "how likely is this target to be free at
this hour", which only requires the state at each sample. Its samples can be sparse.

So the chain steps at one-minute resolution, transition counts are accumulated during the
walk and never stored, and only every Nth state is persisted as an observation. Four
million rows do not go into the database to answer a question that a quarter of a million
can answer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from parkfit.prediction.demand import (
    DemandProfile,
    lambda_table,
    occupancy_rate,
    occupancy_table,
    profile_for,
    vacancy_lambda,
)
from parkfit.storage.models import (
    AvailabilityObservation,
    EvidenceSource,
    OccupancyState,
    ParkingBay,
    ParkingFacility,
)

log = logging.getLogger(__name__)

#: Marks every row this module writes, so a regeneration can delete exactly its own
#: output and never touch a real observation from NDW or a camera.
SYNTHETIC_SOURCE = "synthetic-history"

#: Weekday grouping used for the decay-rate bins. Monday to Thursday behave alike in the
#: Dutch data; Friday, Saturday and Sunday each do not.
WEEKDAY_TYPES = 4


def weekday_type(weekday: int) -> int:
    if weekday <= 3:
        return 0
    return weekday - 3  # Fri -> 1, Sat -> 2, Sun -> 3


@dataclass
class TransitionCounts:
    """Vacant-to-occupied events and time at risk, per (weekday type, hour).

        Both halves are needed because the exponential rate estimator is ``events / exposure``
    , not ``1 / mean(observed dwell)``. The difference matters: dwell times still running
        when the window ends are right-censored, and averaging only the completed ones throws
        away precisely the long survivals, biasing the rate upward.
    """

    events: np.ndarray = field(
        default_factory=lambda: np.zeros((WEEKDAY_TYPES, 24), dtype=np.float64)
    )
    exposure_min: np.ndarray = field(
        default_factory=lambda: np.zeros((WEEKDAY_TYPES, 24), dtype=np.float64)
    )

    def add(self, other: TransitionCounts) -> None:
        self.events += other.events
        self.exposure_min += other.exposure_min


@dataclass
class SimulatedTarget:
    """What is kept about one simulated target after its history is written.

    Deliberately not the samples. Those are written to the database and then dropped:
    holding half a million ``(datetime, bool, int)`` tuples costs more than the feature
    matrix they eventually become, and nothing reads them again. What later stages need
    is the transition counts (for the decay estimator) and the latent profile (to score
    the estimate against the truth that generated it).
    """

    key: tuple[str, int]
    profile: DemandProfile
    capacity: int | None
    counts: TransitionCounts
    sample_count: int = 0


@dataclass
class HistoryReport:
    targets: int = 0
    observations: int = 0
    days: int = 0
    sample_interval_min: int = 0
    deleted: int = 0

    def describe(self) -> str:
        return (
            f"{self.observations:,} observations over {self.days} days "
            f"for {self.targets:,} targets "
            f"(every {self.sample_interval_min} min; {self.deleted:,} replaced)"
        )


def _rate_grid(profile: DemandProfile, start: datetime, minutes: int) -> tuple[np.ndarray, ...]:
    """Pre-compute per-minute occupancy, take rate and release rate.

    Vectorised because the chain walk itself cannot be: each step depends on the previous
    state. Everything that does not depend on the state is computed up front, which is
    what keeps a fourteen-day, one-minute walk to a fraction of a second per target.
    """
    offsets = np.arange(minutes, dtype=np.int64)
    weekday = np.empty(minutes, dtype=np.int64)
    minute_of_day = np.empty(minutes, dtype=np.int64)

    start_dow = start.weekday()
    start_minute = start.hour * 60 + start.minute
    absolute = start_minute + offsets
    minute_of_day[:] = absolute % 1440
    weekday[:] = (start_dow + absolute // 1440) % 7

    # A 7x1440 table covers every distinct (weekday, minute-of-day) pair, so the demand
    # curve is evaluated ten thousand times per target instead of once per simulated
    # minute, three weeks at one-minute resolution is thirty thousand minutes.
    occ = occupancy_table(profile)[weekday, minute_of_day]
    lam = lambda_table(profile)[weekday, minute_of_day]
    # Equilibrium identity: p = lambda / (lambda + mu).
    mu = lam * (1.0 - occ) / np.maximum(occ, 1e-6)
    return occ, lam, mu, weekday, minute_of_day


def _walk_binary(
    profile: DemandProfile,
    start: datetime,
    minutes: int,
    rng: np.random.Generator,
    sample_every: int,
) -> tuple[TransitionCounts, list[tuple[int, bool]]]:
    """Walk one bay's occupancy minute by minute.

    Returns the transition statistics and the sampled states, the latter as
    ``(minute offset, occupied)``.
    """
    occ, lam, mu, weekday, _ = _rate_grid(profile, start, minutes)

    # Probability of at least one transition during a one-minute step.
    p_take = 1.0 - np.exp(-lam)
    p_release = 1.0 - np.exp(-mu)
    draws = rng.random(minutes)

    counts = TransitionCounts()
    samples: list[tuple[int, bool]] = []

    occupied = bool(rng.random() < occ[0])
    wtype = np.array([weekday_type(int(d)) for d in weekday], dtype=np.int64)
    hours = np.array([(start.hour * 60 + start.minute + i) % 1440 // 60 for i in range(minutes)])

    for i in range(minutes):
        if occupied:
            if draws[i] < p_release[i]:
                occupied = False
        else:
            # Time at risk accrues only while the space is actually vacant; that is what
            # makes events/exposure the maximum-likelihood rate under censoring.
            counts.exposure_min[wtype[i], hours[i]] += 1.0
            if draws[i] < p_take[i]:
                counts.events[wtype[i], hours[i]] += 1.0
                occupied = True
        if i % sample_every == 0:
            samples.append((i, occupied))

    return counts, samples


def _walk_facility(
    profile: DemandProfile,
    capacity: int,
    start: datetime,
    minutes: int,
    rng: np.random.Generator,
    sample_every: int,
) -> tuple[TransitionCounts, list[tuple[int, bool, int]]]:
    """Walk a facility's free-space count.

    A garage is not one space, it is ``capacity`` of them, and what a driver cares about
    is whether *any* is free. Simulating each space individually would be honest and
    pointless at 600 spaces; instead the occupied count follows the same two-state rates
    aggregated over the whole facility, which gives the right mean and a plausible
    variance without a per-space state vector.
    """
    occ, lam, mu, weekday, _ = _rate_grid(profile, start, minutes)
    counts = TransitionCounts()
    samples: list[tuple[int, bool, int]] = []

    wtype = np.array([weekday_type(int(d)) for d in weekday], dtype=np.int64)
    hours = np.array([(start.hour * 60 + start.minute + i) % 1440 // 60 for i in range(minutes)])

    occupied_count = round(capacity * float(occ[0]))
    for i in range(minutes):
        free = capacity - occupied_count
        # Arrivals scale with how many drivers are hunting; departures with how many cars
        # are inside. Poisson counts rather than Bernoulli flips, because a large garage
        # sees several of each per minute.
        arrivals = rng.poisson(lam[i] * max(0.0, free) * 0.25) if free > 0 else 0
        departures = rng.poisson(mu[i] * occupied_count * 0.25) if occupied_count > 0 else 0
        occupied_count = min(capacity, max(0, occupied_count + arrivals - departures))

        if free > 0:
            counts.exposure_min[wtype[i], hours[i]] += 1.0
            if occupied_count >= capacity:
                counts.events[wtype[i], hours[i]] += 1.0
        if i % sample_every == 0:
            samples.append((i, occupied_count >= capacity, capacity - occupied_count))

    return counts, samples


def delete_in_chunks(session: Session, model, condition, *, chunk: int = 20000) -> int:
    """Delete matching rows a chunk at a time.

    A single ``DELETE`` over half a million rows makes SQLite build one rollback journal
    covering all of them, and the statement fails outright on a machine short of memory.
    Chunking by primary key bounds the journal to one batch, and each batch is committed,
    so an interruption leaves fewer rows rather than an aborted transaction.
    """
    removed = 0
    while True:
        ids = [row[0] for row in session.execute(select(model.id).where(condition).limit(chunk))]
        if not ids:
            break
        session.execute(delete(model).where(model.id.in_(ids)))
        session.commit()
        removed += len(ids)
    return removed


def _load_targets(
    session: Session, *, bays: int, facilities: int, seed: int
) -> list[tuple[tuple[str, int], float, float, bool, bool, int | None]]:
    """Pick a reproducible spread of real targets to give a history to.

    Sampled across the whole set rather than taking the first N rows, because the first N
    bays share a neighbourhood and would teach the model that all of Amsterdam looks like
    one street.
    """
    out: list[tuple[tuple[str, int], float, float, bool, bool, int | None]] = []

    bay_rows = session.execute(
        select(ParkingBay.id, ParkingBay.lat, ParkingBay.lon, ParkingBay.fiscal).where(
            ParkingBay.lat.is_not(None)
        )
    ).all()
    if bay_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(bay_rows), size=min(bays, len(bay_rows)), replace=False)
        for i in idx:
            bid, lat, lon, fiscal = bay_rows[int(i)]
            out.append((("bay", bid), lat, lon, False, bool(fiscal), None))

    fac_rows = session.execute(
        select(
            ParkingFacility.id,
            ParkingFacility.lat,
            ParkingFacility.lon,
            ParkingFacility.capacity,
        ).where(ParkingFacility.lat.is_not(None))
    ).all()
    if fac_rows:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(fac_rows), size=min(facilities, len(fac_rows)), replace=False)
        for i in idx:
            fid, lat, lon, capacity = fac_rows[int(i)]
            out.append((("facility", fid), lat, lon, True, True, capacity))

    return out


def generate_history(
    session: Session,
    *,
    days: int = 21,
    bays: int = 150,
    facilities: int = 40,
    sample_interval_min: int = 30,
    seed: int = 20260826,
    end: datetime | None = None,
    replace: bool = True,
) -> tuple[HistoryReport, dict[tuple[str, int], SimulatedTarget]]:
    """Simulate and persist a parking history for a sample of real targets."""
    end = end or datetime.now(UTC).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    minutes = days * 24 * 60

    report = HistoryReport(days=days, sample_interval_min=sample_interval_min)

    if replace:
        report.deleted = delete_in_chunks(
            session,
            AvailabilityObservation,
            AvailabilityObservation.source_name == SYNTHETIC_SOURCE,
        )

    targets = _load_targets(session, bays=bays, facilities=facilities, seed=seed)
    if not targets:
        log.warning("no geolocated targets in the database; ingest before generating history")
        return report, {}

    simulated: dict[tuple[str, int], SimulatedTarget] = {}
    pending: list[dict] = []

    for n, (key, lat, lon, is_facility, metered, capacity) in enumerate(targets):
        rng = np.random.default_rng(seed + 977 * n)
        profile = profile_for(
            lat,
            lon,
            is_facility=is_facility,
            metered=metered,
            capacity=capacity,
            seed_value=key[1],
        )

        if is_facility and capacity and capacity > 0:
            counts, fac_samples = _walk_facility(
                profile, capacity, start, minutes, rng, sample_interval_min
            )
            samples = [
                (start + timedelta(minutes=off), occupied, free)
                for off, occupied, free in fac_samples
            ]
        else:
            counts, bin_samples = _walk_binary(profile, start, minutes, rng, sample_interval_min)
            samples = [
                (start + timedelta(minutes=off), occupied, None) for off, occupied in bin_samples
            ]

        simulated[key] = SimulatedTarget(
            key=key,
            profile=profile,
            capacity=capacity,
            counts=counts,
            sample_count=len(samples),
        )

        evidence = (
            EvidenceSource.OPERATOR_FEED if is_facility else EvidenceSource.MUNICIPAL_SENSOR
        ).value
        for observed_at, occupied, free in samples:
            pending.append(
                {
                    "target_kind": key[0],
                    "target_id": key[1],
                    "state": (
                        OccupancyState.OCCUPIED.value if occupied else OccupancyState.VACANT.value
                    ),
                    "evidence_source": evidence,
                    "source_name": SYNTHETIC_SOURCE,
                    "observed_at": observed_at.replace(tzinfo=None),
                    "confidence": 0.9,
                    "vacant_spaces": free,
                    "total_spaces": capacity,
                    "occupancy_ratio": (
                        None if not capacity else 1.0 - (free or 0) / max(1, capacity)
                    ),
                }
            )

        samples.clear()

        # Commit in batches. A single transaction spanning a quarter of a million rows is
        # how an earlier ingest silently rolled back everything it claimed to have done.
        if len(pending) >= 8000:
            session.bulk_insert_mappings(AvailabilityObservation, pending)
            session.commit()
            report.observations += len(pending)
            pending.clear()
            log.info("history: %d/%d targets simulated", n + 1, len(targets))

    if pending:
        session.bulk_insert_mappings(AvailabilityObservation, pending)
        session.commit()
        report.observations += len(pending)

    report.targets = len(targets)
    return report, simulated


def true_occupancy(target: SimulatedTarget, when: datetime) -> float:
    """The latent rate that generated this target's history.

    Only for evaluation: it is what the learned model is scored against, and it is the
    one number the model is never allowed to see.
    """
    return occupancy_rate(target.profile, when.weekday(), when.hour * 60 + when.minute)


def true_lambda(target: SimulatedTarget, when: datetime) -> float:
    return vacancy_lambda(target.profile, when.weekday(), when.hour * 60 + when.minute)


def expected_bay_count(days: int, sample_interval_min: int, targets: int) -> int:
    """Row count a run will produce, for sizing before committing to it."""
    return int(math.ceil(days * 24 * 60 / sample_interval_min) * targets)
