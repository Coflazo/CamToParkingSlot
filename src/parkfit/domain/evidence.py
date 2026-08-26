"""Resolving what is actually known about a parking option, right now.

This module is the honesty layer. Several sources can describe the same car park and
they routinely disagree: the operator feed says twelve free, a camera saw none, the
static register says it has ninety spaces. The product promise is not "we always find a
space"; it is "we show you what we know, where it came from, and how old it is". That
is only possible if disagreement is resolved explicitly rather than by whichever write
happened last.

Two rules govern everything here:

**Source priority is fixed and ordered.** An operator reporting its own barrier count
outranks a camera, which outranks a municipal sensor, which outranks a user report,
which outranks a model, which outranks the static register. Never overwrite: keep every
observation and resolve on read.

**Freshness is a first-class property.** A live source whose last observation is older
than the staleness window stops being a live source and is presented as stale, not as
current. The alternative, showing a five-minute-old count as though it were now, is
how a parking app teaches its users not to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.storage.models import AvailabilityObservation, EvidenceSource, OccupancyState


@dataclass(frozen=True)
class ResolvedAvailability:
    """The current best answer for one parking target, with its provenance."""

    target_kind: str
    target_id: int
    state: OccupancyState
    evidence: EvidenceSource
    observed_at: datetime | None
    age_s: float
    confidence: float
    vacant_spaces: int | None = None
    total_spaces: int | None = None
    occupancy_ratio: float | None = None
    gap_length_m: float | None = None
    source_name: str = ""
    stale: bool = False
    conflicting_sources: int = 0
    #: Metered bays are far more contested than unmetered ones, which changes the
    #: prior by a factor of two.
    metered: bool = True

    #: Probability of being free supplied by the learned occupancy model, when one is
    #: trained and this target has nothing live to go on. ``None`` means the static base
    #: rate applies. It replaces the prior, never an observation:
    #: :func:`resolve_availability` still returns whatever was actually seen, and the
    #: caller that sets this also raises the evidence label to ``PREDICTIVE_MODEL`` so a
    #: prediction is never presented as a measurement.
    model_prior: float | None = None

    #: Prior probability a target is free when nothing has been observed.
    #:
    #: These are not shrugs, they are base rates, and the difference between a kerb bay
    #: and a garage is large enough that one number for both is actively misleading. A
    #: metered bay in a Dutch city centre is occupied most of the day, so the chance any
    #: *one* named bay is free is low. A garage has many interchangeable spaces, so the
    #: chance *some* space is free is high. Learned per-segment rates replace these as
    #: history accumulates; see :class:`SegmentDynamics`.
    PRIOR_SINGLE_METERED_BAY = 0.15
    PRIOR_SINGLE_FREE_BAY = 0.30
    PRIOR_FACILITY = 0.62

    @property
    def probability_available(self) -> float:
        """Probability the target has a free space *right now*.

        Derived rather than stored, because the same observation means different things
        depending on which source made it: an operator counting twelve free spaces at a
        barrier is near-certain, while a model saying "probably" is not.
        """
        if self.stale or self.state is OccupancyState.UNKNOWN:
            return self.prior
        if self.state is OccupancyState.OCCUPIED:
            return 0.02
        if self.vacant_spaces is not None:
            if self.vacant_spaces <= 0:
                return 0.05
            # More free spaces means more resilience to the ones taken before arrival.
            return min(0.99, 0.55 + 0.05 * min(self.vacant_spaces, 9))
        return min(0.95, 0.55 + 0.4 * self.confidence)

    @property
    def prior(self) -> float:
        """Base rate for this kind of target when there is nothing live to go on.

        A learned per-target, per-time-of-day estimate replaces the flat base rate where
        one is available. The base rates remain the fallback rather than being deleted:
        the model covers only the targets it was trained on, and a system that returns
        nothing for the rest would be worse than one that returns a defensible average.
        """
        if self.model_prior is not None:
            return max(0.01, min(0.99, self.model_prior))
        if self.target_kind == "facility":
            return self.PRIOR_FACILITY
        return self.PRIOR_SINGLE_METERED_BAY if self.metered else self.PRIOR_SINGLE_FREE_BAY


def _to_aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; treat them as the UTC they were written as."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def resolve_availability(
    session: Session,
    targets: list[tuple[str, int]],
    *,
    now: datetime | None = None,
    stale_after_s: float = 300.0,
    lookback: timedelta = timedelta(hours=6),
) -> dict[tuple[str, int], ResolvedAvailability]:
    """Resolve the current state of many targets in one query.

    Batched deliberately: a search resolves several hundred candidates, and one query
    per candidate would dominate the entire request budget.
    """
    now = now or datetime.now(UTC)
    if not targets:
        return {}

    kinds = {kind for kind, _ in targets}
    ids = [tid for _, tid in targets]
    since = now - lookback

    rows = (
        session.execute(
            select(AvailabilityObservation)
            .where(
                AvailabilityObservation.target_kind.in_(kinds),
                AvailabilityObservation.target_id.in_(ids),
                AvailabilityObservation.observed_at >= since.replace(tzinfo=None),
            )
            .order_by(AvailabilityObservation.observed_at.desc())
        )
        .scalars()
        .all()
    )

    # Keep the strongest source, and within a source the most recent observation.
    best: dict[tuple[str, int], AvailabilityObservation] = {}
    seen_sources: dict[tuple[str, int], set[int]] = {}
    for row in rows:
        key = (row.target_kind, row.target_id)
        seen_sources.setdefault(key, set()).add(row.evidence_source)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if row.evidence_source > current.evidence_source:
            best[key] = row
        elif row.evidence_source == current.evidence_source:
            row_at = _to_aware(row.observed_at)
            cur_at = _to_aware(current.observed_at)
            if row_at and cur_at and row_at > cur_at:
                best[key] = row

    resolved: dict[tuple[str, int], ResolvedAvailability] = {}
    for key in targets:
        row = best.get(key)
        if row is None:
            resolved[key] = ResolvedAvailability(
                target_kind=key[0],
                target_id=key[1],
                state=OccupancyState.UNKNOWN,
                evidence=EvidenceSource.STATIC_DATABASE,
                observed_at=None,
                age_s=float("inf"),
                confidence=0.0,
                stale=False,
            )
            continue

        observed_at = _to_aware(row.observed_at)
        age = (now - observed_at).total_seconds() if observed_at else float("inf")
        evidence = EvidenceSource(row.evidence_source)
        # Only a source that claims to be live can go stale. The static register is not
        # stale at five minutes old; it is simply static, and is labelled as such.
        live = evidence >= EvidenceSource.MUNICIPAL_SENSOR
        stale = live and age > stale_after_s

        resolved[key] = ResolvedAvailability(
            target_kind=row.target_kind,
            target_id=row.target_id,
            state=OccupancyState(row.state),
            evidence=evidence,
            observed_at=observed_at,
            age_s=age,
            confidence=row.confidence,
            vacant_spaces=row.vacant_spaces,
            total_spaces=row.total_spaces,
            occupancy_ratio=row.occupancy_ratio,
            gap_length_m=row.gap_length_m,
            source_name=row.source_name,
            stale=stale,
            conflicting_sources=max(0, len(seen_sources.get(key, ())) - 1),
        )
    return resolved


def describe_freshness(availability: ResolvedAvailability) -> str:
    """Human wording for how old a claim is. Never implies more precision than exists."""
    if availability.observed_at is None:
        return "no live data"
    if availability.age_s == float("inf"):
        return "no live data"
    if availability.stale:
        return f"last seen {_humanise(availability.age_s)} ago (stale)"
    if availability.evidence <= EvidenceSource.STATIC_DATABASE:
        return "static information"
    return f"updated {_humanise(availability.age_s)} ago"


def _humanise(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} s"
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h"
    return f"{int(seconds // 86400)} d"
