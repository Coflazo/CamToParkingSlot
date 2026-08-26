"""Price estimation.

Dutch parking tariffs are published as a tangle of per-operator fare codes and time
windows, and coverage in the open data is patchy. Rather than pretend to a precision
that is not there, this module returns a price *and a note saying where it came from*:
"operator tariff", "typical for this city", "unknown". The user interface shows the
note, so an estimate never masquerades as a quoted fare.

Fallback rates come from published municipal tariffs and are deliberately coarse. They
exist so the ranking can compare a EUR 7.50/hour canal-ring bay against a EUR 1.50/hour
park-and-ride, which is a real and important distinction, without claiming to know the
exact fare of either.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from parkfit.storage.models import FacilityKind, ParkingBay, ParkingFacility

#: Indicative on-street hourly rates by city, in euros. Amsterdam centre is the most
#: expensive on-street parking in the Netherlands by a wide margin, and a model that
#: ignores that will rank kerb spaces far too favourably against garages.
CITY_ONSTREET_EUR_PER_HOUR = {
    "amsterdam": 7.50,
    "utrecht": 5.00,
    "rotterdam": 3.75,
    "den haag": 4.50,
    "haarlem": 4.10,
    "leiden": 3.60,
    "groningen": 3.00,
    "eindhoven": 3.00,
    "maastricht": 3.10,
    "delft": 3.30,
}
DEFAULT_ONSTREET_EUR_PER_HOUR = 2.60

#: Indicative rates by facility type when nothing better is known.
KIND_EUR_PER_HOUR = {
    FacilityKind.GARAGE.value: 4.50,
    FacilityKind.SURFACE_LOT.value: 2.50,
    FacilityKind.PARK_AND_RIDE.value: 1.00,
    FacilityKind.ON_STREET_ZONE.value: 3.50,
    FacilityKind.TRUCK_PARKING.value: 2.00,
    FacilityKind.UNKNOWN.value: 3.00,
}

#: Park and ride in the Randstad is deliberately cheap, and usually a flat day rate
#: rather than an hourly one, which changes the ranking for a long stay entirely.
PARK_AND_RIDE_DAY_CAP_EUR = 8.00


def estimate_price(
    session: Session,
    key: tuple[str, int],
    *,
    kind: str,
    arrival: datetime,
    duration_minutes: int,
) -> tuple[float, str]:
    """Estimate what this stay will cost, and say how confident that is."""
    hours = max(0.25, duration_minutes / 60.0)

    if key[0] == "bay":
        bay = session.get(ParkingBay, key[1])
        if bay is None:
            return 0.0, "unknown"
        if not bay.fiscal:
            # NIET FISCAAL means the bay is not metered. Free to park, though it may
            # still be restricted by sign, which the legality check handles separately.
            return 0.0, "free (unmetered bay)"
        city = _city_for_bay(session, bay)
        rate = CITY_ONSTREET_EUR_PER_HOUR.get((city or "").lower(), DEFAULT_ONSTREET_EUR_PER_HOUR)
        return round(rate * hours, 2), f"typical on-street rate for {city or 'this area'}"

    facility = session.get(ParkingFacility, key[1])
    if facility is None:
        return 0.0, "unknown"

    if facility.tariff_eur_per_hour:
        total = facility.tariff_eur_per_hour * hours
        if facility.tariff_day_max_eur and hours >= 4:
            total = min(total, facility.tariff_day_max_eur)
        return round(total, 2), "operator tariff"

    if facility.tariff_note and "no" in str(facility.tariff_note).lower():
        return 0.0, "free (per source)"

    rate = KIND_EUR_PER_HOUR.get(facility.kind, KIND_EUR_PER_HOUR[FacilityKind.UNKNOWN.value])
    if facility.city and facility.kind == FacilityKind.ON_STREET_ZONE.value:
        rate = CITY_ONSTREET_EUR_PER_HOUR.get(facility.city.lower(), rate)

    total = rate * hours
    if facility.kind == FacilityKind.PARK_AND_RIDE.value:
        total = min(total, PARK_AND_RIDE_DAY_CAP_EUR)
        return round(total, 2), "typical park-and-ride rate (day capped)"

    return round(total, 2), f"estimated from typical {facility.kind.replace('_', ' ')} rates"


def _city_for_bay(session: Session, bay: ParkingBay) -> str | None:
    """Bays carry a street but no city; the source is city-scoped, so infer from it."""
    if bay.source_name.lower().startswith("amsterdam"):
        return "Amsterdam"
    nearby = session.execute(
        select(ParkingFacility.city)
        .where(
            ParkingFacility.city.is_not(None),
            ParkingFacility.lat.between(bay.lat - 0.02, bay.lat + 0.02),
            ParkingFacility.lon.between(bay.lon - 0.03, bay.lon + 0.03),
        )
        .limit(1)
    ).scalar_one_or_none()
    return nearby


def estimate_prices(
    session: Session,
    targets: list[tuple[tuple[str, int], str]],
    *,
    arrival: datetime,
    duration_minutes: int,
) -> dict[tuple[str, int], tuple[float, str]]:
    """Price many candidates in two queries instead of one per candidate.

    A search prices several hundred options, and the per-candidate version issued a
    round-trip each: 456 queries for one search, costing more than the routing did.
    """
    hours = max(0.25, duration_minutes / 60.0)
    out: dict[tuple[str, int], tuple[float, str]] = {}

    bay_ids = [key[1] for key, _ in targets if key[0] == "bay"]
    facility_ids = [key[1] for key, _ in targets if key[0] != "bay"]

    bays = {}
    if bay_ids:
        bays = {
            b.id: b
            for b in session.execute(select(ParkingBay).where(ParkingBay.id.in_(bay_ids))).scalars()
        }
    facilities = {}
    if facility_ids:
        facilities = {
            f.id: f
            for f in session.execute(
                select(ParkingFacility).where(ParkingFacility.id.in_(facility_ids))
            ).scalars()
        }

    city_cache: dict[tuple[int, int], str | None] = {}

    for key, _kind in targets:
        if key[0] == "bay":
            bay = bays.get(key[1])
            if bay is None:
                out[key] = (0.0, "unknown")
                continue
            if not bay.fiscal:
                # NIET FISCAAL means the bay is not metered. In a Dutch city centre that
                # usually means permit-controlled rather than genuinely free, and the
                # regime data does not always say which, so it is never presented as
                # confirmed free parking.
                out[key] = (0.0, "not metered: check the signs on arrival")
                continue
            cell = (int(bay.lat * 50), int(bay.lon * 50))
            if cell not in city_cache:
                city_cache[cell] = _city_for_bay(session, bay)
            city = city_cache[cell]
            rate = CITY_ONSTREET_EUR_PER_HOUR.get(
                (city or "").lower(), DEFAULT_ONSTREET_EUR_PER_HOUR
            )
            out[key] = (round(rate * hours, 2), f"typical on-street rate for {city or 'this area'}")
            continue

        facility = facilities.get(key[1])
        if facility is None:
            out[key] = (0.0, "unknown")
            continue
        if facility.tariff_eur_per_hour:
            total = facility.tariff_eur_per_hour * hours
            if facility.tariff_day_max_eur and hours >= 4:
                total = min(total, facility.tariff_day_max_eur)
            out[key] = (round(total, 2), "operator tariff")
            continue

        rate = KIND_EUR_PER_HOUR.get(facility.kind, KIND_EUR_PER_HOUR[FacilityKind.UNKNOWN.value])
        if facility.city and facility.kind == FacilityKind.ON_STREET_ZONE.value:
            rate = CITY_ONSTREET_EUR_PER_HOUR.get(facility.city.lower(), rate)
        total = rate * hours
        if facility.kind == FacilityKind.PARK_AND_RIDE.value:
            out[key] = (
                round(min(total, PARK_AND_RIDE_DAY_CAP_EUR), 2),
                "typical park-and-ride rate (day capped)",
            )
            continue
        out[key] = (
            round(total, 2),
            f"estimated from typical {facility.kind.replace('_', ' ')} rates",
        )
    return out
