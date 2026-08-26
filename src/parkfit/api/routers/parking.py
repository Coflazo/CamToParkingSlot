"""Facility detail, user confirmations and the live availability stream."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from parkfit.api.schemas import (
    EvidenceDetail,
    FacilityDetail,
    ObservationResponse,
    UserConfirmationRequest,
)
from parkfit.api.security import get_optional_user
from parkfit.config import get_settings
from parkfit.domain.evidence import describe_freshness, resolve_availability
from parkfit.storage.models import (
    AvailabilityObservation,
    EvidenceSource,
    OccupancyState,
    ParkingFacility,
    SourceLicence,
    User,
    UserConfirmation,
)
from parkfit.storage.session import get_session, session_scope

log = logging.getLogger(__name__)
router = APIRouter(tags=["parking"])


@router.get("/parking/{facility_id}", response_model=FacilityDetail)
async def facility_detail(
    facility_id: int, session: AsyncSession = Depends(get_session)
) -> FacilityDetail:
    facility = (
        await session.execute(select(ParkingFacility).where(ParkingFacility.id == facility_id))
    ).scalar_one_or_none()
    if facility is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown facility")

    licence = (
        await session.execute(
            select(SourceLicence).where(SourceLicence.source_name == facility.source_name)
        )
    ).scalar_one_or_none()

    with session_scope() as sync_session:
        resolved = resolve_availability(
            sync_session,
            [("facility", facility_id)],
            stale_after_s=get_settings().stale_after_s,
        )
    availability = resolved.get(("facility", facility_id))

    evidence = None
    if availability is not None:
        evidence = EvidenceDetail(
            source=availability.evidence.name,
            observed_at=availability.observed_at,
            age_seconds=(
                round(availability.age_s, 1) if availability.age_s != float("inf") else None
            ),
            freshness=describe_freshness(availability),
            confidence_label=(
                "DATA_CURRENTLY_STALE"
                if availability.stale
                else "AVAILABILITY_REPORTED_BY_OPERATOR"
            ),
            stale=availability.stale,
            conflicting_sources=availability.conflicting_sources,
            vacant_spaces=availability.vacant_spaces,
            total_spaces=availability.total_spaces,
        )

    return FacilityDetail(
        id=facility.id,
        name=facility.name,
        kind=facility.kind,
        lat=facility.lat,
        lon=facility.lon,
        city=facility.city,
        capacity=facility.capacity,
        charging_capacity=facility.charging_capacity,
        max_vehicle_height_cm=facility.max_vehicle_height_cm,
        source_name=facility.source_name,
        evidence=evidence,
        # Attribution is a licence obligation for OpenStreetMap, not a courtesy, so it
        # travels with the data rather than living in a footer somewhere.
        attribution=licence.attribution_text if licence else None,
    )


@router.post(
    "/observations/user-confirmation",
    response_model=ObservationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def user_confirmation(
    payload: UserConfirmationRequest,
    session: AsyncSession = Depends(get_session),
    user: User | None = Depends(get_optional_user),
) -> ObservationResponse:
    """Record what actually happened when a driver acted on a recommendation.

    This is the only sensor that is always present. It is also the ground truth the
    accuracy metrics are measured against: a "was_occupied" report on a space the system
    called free is a false-free event, which is the error this product cares about most.
    """
    now = datetime.now(UTC)
    session.add(
        UserConfirmation(
            user_id=user.id if user else None,
            target_kind=payload.target_kind,
            target_id=payload.target_id,
            outcome=payload.outcome,
            reported_at=now.replace(tzinfo=None),
            recommendation_id=payload.recommendation_id,
            note=payload.note,
        )
    )

    # A confirmation is also an observation. It enters the same append-only table as
    # every other source, at user-report priority, so it can correct a stale feed
    # without being able to override an operator counting spaces at its own barrier.
    state = {
        "parked": OccupancyState.VACANT,
        "was_occupied": OccupancyState.OCCUPIED,
        "not_found": OccupancyState.UNKNOWN,
        "illegal": OccupancyState.UNKNOWN,
    }[payload.outcome]

    session.add(
        AvailabilityObservation(
            target_kind=payload.target_kind,
            target_id=payload.target_id,
            observed_at=now.replace(tzinfo=None),
            evidence_source=int(EvidenceSource.USER_CONFIRMATION),
            state=state.value,
            confidence=0.8 if payload.outcome in {"parked", "was_occupied"} else 0.3,
            source_name="user",
        )
    )
    await session.flush()
    return ObservationResponse(accepted=True, message="Thank you, that improves the next search.")


@router.get("/availability/stream")
async def availability_stream(
    targets: str = Query(
        description="Comma-separated target ids, e.g. facility:12,bay:9981", max_length=2000
    ),
    interval_s: float = Query(default=10.0, ge=2.0, le=60.0),
) -> StreamingResponse:
    """Server-sent events for an active search.

    Only for a search a driver is currently acting on. Map browsing uses cached HTTP,
    because holding a stream open for every idle map view is a cost with no benefit.
    """
    parsed: list[tuple[str, int]] = []
    for token in targets.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        kind, _, raw_id = token.partition(":")
        if kind not in {"facility", "bay", "curb"} or not raw_id.isdigit():
            continue
        parsed.append((kind, int(raw_id)))

    if not parsed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No valid targets supplied"
        )
    if len(parsed) > 50:
        parsed = parsed[:50]

    async def event_source():
        settings = get_settings()
        last_payload: str | None = None
        deadline = datetime.now(UTC) + timedelta(minutes=20)
        try:
            while datetime.now(UTC) < deadline:
                with session_scope() as session:
                    resolved = resolve_availability(
                        session, parsed, stale_after_s=settings.stale_after_s
                    )
                items = [
                    {
                        "target": f"{k}:{i}",
                        "state": r.state.value,
                        "vacant_spaces": r.vacant_spaces,
                        "probability": round(r.probability_available, 3),
                        "evidence": r.evidence.name,
                        "observed_at": r.observed_at.isoformat() if r.observed_at else None,
                        "freshness": describe_freshness(r),
                        "stale": r.stale,
                    }
                    for (k, i), r in resolved.items()
                ]
                payload = json.dumps({"items": items}, separators=(",", ":"))
                # Only emit on change. A stream that repeats itself every ten seconds
                # burns battery on a phone in someone's car for no information at all.
                if payload != last_payload:
                    yield f"event: availability\ndata: {payload}\n\n"
                    last_payload = payload
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:  # pragma: no cover - client disconnected
            return

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
