"""Vehicle registration and management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from parkfit.api.schemas import (
    PlateLookupRequest,
    PlateLookupResponse,
    VehicleCreate,
    VehicleResponse,
    VehicleUpdate,
)
from parkfit.api.security import get_current_user
from parkfit.domain.vehicle import RdwVehicleClient
from parkfit.storage.models import User, Vehicle
from parkfit.storage.session import get_session

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


@router.post("/lookup-rdw", response_model=PlateLookupResponse)
async def lookup_plate(payload: PlateLookupRequest) -> PlateLookupResponse:
    """Look up a Dutch licence plate in the RDW open register.

    The plate is used for this request and then discarded. It is a direct identifier
    for a person and the product only needs the dimensions; keeping it would build a
    licence-plate database as a side effect of a parking search.

    Height is never returned, because the register does not publish it, and height is
    the dimension a barrier physically stops. It is always in ``unconfirmed_fields``.
    """
    with RdwVehicleClient() as client:
        profile = client.lookup(payload.plate)

    if profile is None:
        return PlateLookupResponse(
            found=False,
            note="No vehicle found for that plate. You can still enter dimensions manually.",
        )

    return PlateLookupResponse(
        found=True,
        make=profile.make,
        model=profile.model,
        length_cm=profile.length_cm,
        body_width_cm=profile.body_width_cm,
        width_with_mirrors_cm=profile.width_with_mirrors_cm,
        weight_kg=profile.weight_kg,
        fuel_type=profile.fuel_type,
        is_ev=profile.is_ev,
        unconfirmed_fields=profile.unconfirmed_fields,
        note=(
            "RDW does not publish vehicle height, and height is what a garage barrier "
            "stops. Please measure it, including anything on the roof."
        ),
    )


@router.get("", response_model=list[VehicleResponse])
async def list_vehicles(
    user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)
) -> list[VehicleResponse]:
    rows = (
        (await session.execute(select(Vehicle).where(Vehicle.user_id == user.id))).scalars().all()
    )
    return [VehicleResponse.model_validate(v) for v in rows]


@router.post("", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
async def create_vehicle(
    payload: VehicleCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> VehicleResponse:
    vehicle = Vehicle(user_id=user.id, **payload.model_dump())

    # Anything the user typed is confirmed by definition; that is the whole point of
    # the confirmation step. What matters is that it is recorded as confirmed, so the
    # fit engine can tell a measured height from an absent one.
    vehicle.length_confirmed = True
    vehicle.width_confirmed = payload.width_with_mirrors_cm > 0
    vehicle.height_confirmed = True
    vehicle.weight_confirmed = payload.weight_kg > 0

    if vehicle.width_with_mirrors_cm <= 0:
        # Mirrors stick out roughly 18 cm per side. Assumed, and flagged unconfirmed,
        # because mirror span varies far more between models than bodywork does.
        vehicle.width_with_mirrors_cm = vehicle.body_width_cm + 36.0
    if vehicle.height_with_accessories_cm <= 0:
        vehicle.height_with_accessories_cm = vehicle.height_cm

    session.add(vehicle)
    await session.flush()
    return VehicleResponse.model_validate(vehicle)


@router.patch("/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(
    vehicle_id: int,
    payload: VehicleUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> VehicleResponse:
    vehicle = await _owned_vehicle(session, user, vehicle_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(vehicle, field, value)
    if payload.height_cm is not None:
        vehicle.height_confirmed = True
        if vehicle.height_with_accessories_cm < vehicle.height_cm:
            vehicle.height_with_accessories_cm = vehicle.height_cm
    if payload.width_with_mirrors_cm is not None:
        vehicle.width_confirmed = True
    await session.flush()
    return VehicleResponse.model_validate(vehicle)


@router.delete("/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vehicle(
    vehicle_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    vehicle = await _owned_vehicle(session, user, vehicle_id)
    await session.delete(vehicle)
    await session.flush()


async def _owned_vehicle(session: AsyncSession, user: User, vehicle_id: int) -> Vehicle:
    vehicle = (
        await session.execute(
            select(Vehicle).where(Vehicle.id == vehicle_id, Vehicle.user_id == user.id)
        )
    ).scalar_one_or_none()
    if vehicle is None:
        # 404 rather than 403 for a vehicle owned by someone else: a different status
        # would confirm the id exists, which is information the caller has no claim to.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown vehicle")
    return vehicle
