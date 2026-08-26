"""Is this driver allowed to park here, at this time, in this vehicle?

An empty space and a usable space are not the same thing. A bay can be visibly free and
still be reserved for permit holders, marked for disabled badges, inside a loading
window, or an EV charging bay that a diesel must not occupy. Amsterdam publishes the
sign code and time regimes for every bay, so this is knowable rather than guessable --
and offering an illegal space is worse than offering none, because the driver pays for
the mistake with a fine.

The rule that matters most here: **a restriction whose applicability is uncertain is
treated as applying.** If we cannot tell whether the permit window covers the intended
stay, the answer is no. That direction costs a driver one option; the other direction
costs them ninety euros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.domain.vehicle import VehicleProfile
from parkfit.storage.models import ParkingRestriction


@dataclass
class RestrictionVerdict:
    allowed: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_duration_minutes: int | None = None
    requires_permit: bool = False
    disabled_only: bool = False
    ev_only: bool = False

    def deny(self, reason: str) -> None:
        self.allowed = False
        self.reasons.append(reason)


def _overlaps(rule: ParkingRestriction, arrival: datetime, departure: datetime) -> bool:
    """Does the stay intersect the window this rule applies to?

    Checked per calendar day of the stay, because a rule keyed to a weekday mask must be
    evaluated against each day the car is actually parked. An overnight stay from Friday
    evening to Saturday morning is governed by both days.
    """
    day = arrival.date()
    end_day = departure.date()
    while day <= end_day:
        weekday = day.weekday()
        if rule.weekday_mask & (1 << weekday):
            window_start = datetime.combine(day, datetime.min.time(), tzinfo=arrival.tzinfo)
            start = window_start + timedelta(minutes=rule.start_minute)
            end = window_start + timedelta(minutes=rule.end_minute)
            if arrival < end and departure > start:
                return True
        day += timedelta(days=1)
    return False


def _in_validity_period(rule: ParkingRestriction, arrival: datetime) -> bool:
    valid_from = rule.valid_from
    valid_until = rule.valid_until
    if valid_from is not None and valid_from.tzinfo is None:
        valid_from = valid_from.replace(tzinfo=arrival.tzinfo)
    if valid_until is not None and valid_until.tzinfo is None:
        valid_until = valid_until.replace(tzinfo=arrival.tzinfo)
    if valid_from and arrival < valid_from:
        return False
    return not (valid_until and arrival > valid_until)


def evaluate_restrictions(
    session: Session,
    targets: list[tuple[str, int]],
    *,
    arrival: datetime,
    departure: datetime,
    vehicle: VehicleProfile,
    needs_ev_charging: bool = False,
    needs_disabled_bay: bool = False,
) -> dict[tuple[str, int], RestrictionVerdict]:
    """Evaluate every restriction for many targets in one query."""
    verdicts: dict[tuple[str, int], RestrictionVerdict] = {
        key: RestrictionVerdict() for key in targets
    }
    if not targets:
        return verdicts

    kinds = {kind for kind, _ in targets}
    ids = [tid for _, tid in targets]
    rules = (
        session.execute(
            select(ParkingRestriction).where(
                ParkingRestriction.target_kind.in_(kinds),
                ParkingRestriction.target_id.in_(ids),
            )
        )
        .scalars()
        .all()
    )

    stay_minutes = max(1, int((departure - arrival).total_seconds() // 60))

    for rule in rules:
        key = (rule.target_kind, rule.target_id)
        verdict = verdicts.get(key)
        if verdict is None:
            continue
        if not _in_validity_period(rule, arrival):
            continue
        if not _overlaps(rule, arrival, departure):
            continue

        label = rule.description or rule.rule_type

        if rule.forbids_parking:
            verdict.deny(f"parking not permitted here during your stay ({label})")
            continue

        if rule.disabled_only:
            verdict.disabled_only = True
            if not needs_disabled_bay:
                verdict.deny(f"reserved for disabled badge holders ({label})")
            continue

        if rule.ev_only:
            verdict.ev_only = True
            if not vehicle.is_ev:
                verdict.deny(f"reserved for charging electric vehicles ({label})")
            elif not needs_ev_charging:
                # An EV that is not charging still may not block a charging bay.
                verdict.deny("charging bay: only usable while actually charging")
            continue

        if rule.permit_required:
            verdict.requires_permit = True
            verdict.deny(f"permit holders only during your stay ({label})")
            continue

        if rule.max_duration_minutes:
            current = verdict.max_duration_minutes
            verdict.max_duration_minutes = (
                rule.max_duration_minutes
                if current is None
                else min(current, rule.max_duration_minutes)
            )

    for verdict in verdicts.values():
        limit = verdict.max_duration_minutes
        if limit is not None and stay_minutes > limit:
            verdict.deny(f"maximum stay here is {limit} minutes, you asked for {stay_minutes}")
        elif limit is not None:
            verdict.warnings.append(f"maximum stay {limit} minutes")

    if needs_ev_charging:
        for key, verdict in verdicts.items():
            if key[0] == "bay" and not verdict.ev_only:
                verdict.warnings.append("no charging point recorded for this bay")

    return verdicts
