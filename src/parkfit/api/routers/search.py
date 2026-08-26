"""Geocoding and parking search endpoints."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from parkfit.api.schemas import (
    EvidenceDetail,
    FitDetail,
    GeocodeResponse,
    GeocodeResult,
    LegDetail,
    RecommendationResponse,
    SearchCreate,
    SearchResponse,
)
from parkfit.api.security import get_optional_user
from parkfit.config import get_settings
from parkfit.domain.evidence import describe_freshness
from parkfit.domain.vehicle import VehicleProfile
from parkfit.services.geocoding import HybridGeocoder
from parkfit.services.search import (
    Candidate,
    SearchEngine,
    SearchPreferences,
    SearchRequest,
)
from parkfit.storage.models import User, Vehicle
from parkfit.storage.session import get_session, session_scope

log = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


@router.get("/geocode", response_model=GeocodeResponse)
async def geocode(
    q: str = Query(min_length=2, max_length=200, description="Free-text destination"),
    city: str | None = Query(default=None),
    limit: int = Query(default=5, ge=1, le=20),
) -> GeocodeResponse:
    """Resolve a destination.

    Points of interest are searched before the address register, which is what makes
    "Rembrandt House Museum" work: the official Dutch geocoder returns nothing for it,
    because it indexes addresses rather than places.
    """
    # The geocoder is synchronous and touches both a local table and an upstream API,
    # so it runs in its own session rather than borrowing the request one.
    with session_scope() as session:
        geocoder = HybridGeocoder(session)
        try:
            hits = geocoder.geocode(q, city=city, limit=limit)
        finally:
            geocoder.close()
    return GeocodeResponse(
        query=q,
        results=[
            GeocodeResult(
                label=h.label,
                lat=h.lat,
                lon=h.lon,
                kind=h.kind,
                source=h.source,
                confidence=round(h.confidence, 3),
                city=h.city,
            )
            for h in hits
        ],
    )


@router.post("/searches", response_model=SearchResponse, status_code=status.HTTP_201_CREATED)
async def create_search(
    payload: SearchCreate,
    session: AsyncSession = Depends(get_session),
    user: User | None = Depends(get_optional_user),
) -> SearchResponse:
    """Run a parking search.

    Anonymous searches are allowed. A driver should not have to create an account to
    find out where they can park, and a search without a vehicle still works -- it just
    cannot verify fit, and every result says so.
    """
    vehicle_profile = VehicleProfile()
    if payload.vehicle_id is not None:
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sign in to search with a saved vehicle",
            )
        vehicle = (
            await session.execute(
                select(Vehicle).where(Vehicle.id == payload.vehicle_id, Vehicle.user_id == user.id)
            )
        ).scalar_one_or_none()
        if vehicle is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown vehicle")
        vehicle_profile = _profile_from_row(vehicle)

    preferences = SearchPreferences(
        max_walk_minutes=payload.preferences.max_walk_minutes,
        prefer_covered=payload.preferences.prefer_covered,
        prefer_cheapest=payload.preferences.prefer_cheapest,
        needs_ev_charging=payload.preferences.needs_ev_charging,
        needs_disabled_bay=payload.preferences.needs_disabled_bay,
        include_on_street=payload.preferences.include_on_street,
        value_of_time_eur_per_min=payload.preferences.value_of_time_eur_per_min,
    )

    request = SearchRequest(
        destination=payload.destination,
        vehicle=vehicle_profile,
        origin_lat=payload.origin_lat,
        origin_lon=payload.origin_lon,
        arrival_time=payload.arrival_time,
        duration_minutes=payload.expected_duration_minutes,
        preferences=preferences,
        user_id=user.id if user else None,
        city_hint=payload.city_hint,
    )

    with session_scope() as sync_session:
        engine = SearchEngine(sync_session)
        try:
            result = engine.search(request)
        finally:
            engine.close()

        return SearchResponse(
            search_id=result.search_id,
            destination=(
                GeocodeResult(
                    label=result.destination.label,
                    lat=result.destination.lat,
                    lon=result.destination.lon,
                    kind=result.destination.kind,
                    source=result.destination.source,
                    confidence=round(result.destination.confidence, 3),
                    city=result.destination.city,
                )
                if result.destination
                else None
            ),
            results=[
                _to_recommendation(c, index, payload.expected_duration_minutes)
                for index, c in enumerate(result.results)
            ],
            considered=result.considered,
            merged_duplicates=result.merged_duplicates,
            rejected_illegal=result.rejected_illegal,
            rejected_fit=result.rejected_fit,
            rejected_walk=result.rejected_walk,
            radius_m=round(result.radius_m, 1),
            routing_provider=result.routing_provider,
            warnings=result.warnings,
            elapsed_ms=round(result.elapsed_ms, 1),
        )


def _profile_from_row(vehicle: Vehicle) -> VehicleProfile:
    return VehicleProfile(
        id=str(vehicle.id),
        nickname=vehicle.nickname,
        make=vehicle.make,
        model=vehicle.model,
        length_cm=vehicle.length_cm,
        body_width_cm=vehicle.body_width_cm,
        width_with_mirrors_cm=vehicle.width_with_mirrors_cm,
        height_cm=vehicle.height_cm,
        height_with_accessories_cm=vehicle.height_with_accessories_cm,
        weight_kg=vehicle.weight_kg,
        length_confirmed=vehicle.length_confirmed,
        width_confirmed=vehicle.width_confirmed,
        height_confirmed=vehicle.height_confirmed,
        weight_confirmed=vehicle.weight_confirmed,
        is_ev=vehicle.is_ev,
        charging_connector=vehicle.charging_connector,
        has_trailer=vehicle.has_trailer,
        has_roof_box=vehicle.has_roof_box,
        extra_parallel_clearance_cm=vehicle.extra_parallel_clearance_cm,
    )


def _explain_fit(candidate: Candidate) -> str:
    """Plain wording for why a result fits, is tight, or could not be checked."""
    if candidate.fit_verdict == "FITS":
        return f"clears every checked limit with {candidate.fit_slack_cm:.0f} cm to spare"
    if candidate.fit_verdict == "TIGHT_FIT":
        constraint = candidate.fit_binding or "clearance"
        return f"fits, but only {candidate.fit_slack_cm:.0f} cm of {constraint} clearance"
    if candidate.fit_verdict == "DOES_NOT_FIT":
        constraint = candidate.fit_binding or "a limit"
        return f"too large: {constraint} is short by {abs(candidate.fit_slack_cm):.0f} cm"
    # The commonest case by far is that no vehicle was selected, and telling someone
    # that "vehicle_dimensions" are unknown is accurate and useless. Say what to do.
    if "vehicle_dimensions" in candidate.fit_unverified:
        return "select a vehicle to check whether it fits here"
    if "facility_max_height" in candidate.fit_unverified:
        return "this car park does not publish a height limit, so fit is unconfirmed"
    if "bay_orientation" in candidate.fit_unverified:
        return "bay layout unknown, so the stricter of the two readings was applied"
    missing = ", ".join(candidate.fit_unverified) or "required dimensions"
    return f"fit not fully verified ({missing} unknown)"


def _to_recommendation(
    candidate: Candidate, index: int, duration_minutes: int
) -> RecommendationResponse:
    settings = get_settings()
    availability = candidate.availability
    now = datetime.now(UTC)

    evidence = EvidenceDetail(
        source=availability.evidence.name if availability else "STATIC_DATABASE",
        observed_at=availability.observed_at if availability else None,
        age_seconds=(
            round(availability.age_s, 1)
            if availability and availability.age_s != float("inf")
            else None
        ),
        freshness=describe_freshness(availability) if availability else "no live data",
        confidence_label=candidate.confidence_label or "STATIC_INFORMATION_ONLY",
        stale=bool(availability and availability.stale),
        conflicting_sources=availability.conflicting_sources if availability else 0,
        vacant_spaces=availability.vacant_spaces if availability else None,
        total_spaces=availability.total_spaces if availability else None,
    )

    # An exact single space is only a claim for as long as the observation behind it is
    # current. Handing the client an expiry stops a stale space being shown as live.
    expires_at = (
        now + timedelta(seconds=settings.exact_space_ttl_s) if candidate.is_exact_space else None
    )

    return RecommendationResponse(
        id=f"{candidate.key[0]}:{candidate.key[1]}",
        kind=candidate.kind,
        name=candidate.name,
        lat=candidate.lat,
        lon=candidate.lon,
        rank=index,
        generalised_cost_eur=round(candidate.generalised_cost, 2),
        probability_at_arrival=round(candidate.probability_at_eta, 3),
        price_eur=candidate.price_eur,
        price_note=candidate.price_note,
        drive=_leg(candidate.drive),
        walk=_leg(candidate.walk),
        fit=FitDetail(
            verdict=candidate.fit_verdict,
            slack_cm=round(candidate.fit_slack_cm, 1),
            binding_constraint=candidate.fit_binding,
            unverified=candidate.fit_unverified,
            explanation=_explain_fit(candidate),
        ),
        evidence=evidence,
        capacity=candidate.capacity,
        max_height_cm=candidate.max_height_cm,
        bay_length_cm=round(candidate.bay_length_cm, 1),
        bay_width_cm=round(candidate.bay_width_cm, 1),
        orientation=candidate.orientation,
        restriction_warnings=(candidate.restriction.warnings if candidate.restriction else []),
        is_exact_space=candidate.is_exact_space,
        expires_at=expires_at,
    )


def _leg(route) -> LegDetail | None:
    if route is None:
        return None
    return LegDetail(
        distance_m=round(route.distance_m, 1),
        duration_min=round(route.duration_min, 2),
        provider=route.provider,
        is_estimate=route.is_estimate,
        geometry=route.geometry,
    )
