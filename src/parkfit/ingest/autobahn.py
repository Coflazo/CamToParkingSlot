"""Autobahn GmbH: parking on the German motorway network.

The third country, and the first where the national source gives capacity without
occupancy. That distinction runs through this whole module. `api.ibb.gov.tr/ispark/Park`
publishes how many spaces are free right now, and NDW publishes the same for Dutch truck
parking; Autobahn GmbH publishes how many spaces exist. Those are different claims, and
conflating them would turn "this rest area has 20 car spaces" into "this rest area has 20
free car spaces", which is a lie at exactly the moment a driver is deciding to stop.

So every observation from here lands at ``EvidenceSource.STATIC_DATABASE``, the rung this
product uses for "somebody wrote this down once", never at ``OPERATOR_FEED``, and it
records ``total_spaces`` with ``vacant_spaces`` left unset rather than guessed.

**The endpoint is called ``parking_lorry`` and is not only about lorries.** Each record's
description carries both ``PKW Stellplätze`` (car spaces) and ``LKW Stellplätze`` (truck
spaces), and it is the car number this product needs. A site with lorry spaces and no car
spaces is genuinely a truck stop and is stored as one, so a car search never offers it.

Two shapes here would fail quietly rather than loudly:

* coordinates arrive as GeoJSON, which is **longitude first**, and the same mistake in
  the Istanbul adapter would have put Turkey in Somalia;
* ``isBlocked`` is the **string** ``"false"``, and every non-empty string is truthy in
  Python, so a naive read closes every rest area on the network.

Coverage is the motorway network only. German city parking needs the municipal
Parkleitsystem feeds, which are per-city and are a separate adapter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta
from parkfit.storage.models import (
    AvailabilityObservation,
    EvidenceSource,
    FacilityKind,
    OccupancyState,
    ParkingFacility,
    SourceLicence,
    utcnow,
)
from parkfit.storage.session import session_scope

log = logging.getLogger(__name__)

BASE_URL = "https://verkehr.autobahn.de/o/autobahn"

#: "PKW Stellplätze: 20" and "LKW Stellplätze: 16".
#:
#: Matched loosely on case and spacing because this is a display string, not a schema,
#: and display strings drift. The vowel alternation is not padding: German transliterates
#: ä as "ae", so the same field legitimately appears as Stellplätze, Stellplaetze and
#: occasionally Stellplatze depending on which system last touched it, and matching only
#: the umlaut form silently reads zero spaces from a record that has plenty.
_SPACES = re.compile(r"(PKW|LKW)\s*Stellpl(?:ae|ä|a)tze\s*:\s*(\d+)", re.IGNORECASE)

#: A rest area with more spaces than this is a data error, not a very large rest area.
#: The biggest German Autohof is a few hundred.
_MAX_PLAUSIBLE_SPACES = 3000


@dataclass(frozen=True)
class RestArea:
    """One motorway parking site, as the Autobahn API describes it."""

    identifier: str
    road: str
    name: str
    lat: float
    lon: float
    car_spaces: int
    lorry_spaces: int
    blocked: bool
    problems: tuple[str, ...] = ()

    @property
    def kind(self) -> FacilityKind:
        # Car spaces make it usable by this product's users. Without them it is a truck
        # stop, and storing it as a surface lot would offer a car a space that does not
        # exist for it.
        if self.car_spaces > 0:
            return FacilityKind.SURFACE_LOT
        return FacilityKind.TRUCK_PARKING


def parse_spaces(description: list[str] | None) -> tuple[int, int]:
    """Car and lorry space counts from the German description lines.

    Returns ``(0, 0)`` when the record says nothing, which is distinct from a record that
    says zero: both mean "do not offer this to a car", and neither is treated as unknown
    capacity that might be large.
    """
    car = lorry = 0
    for line in description or []:
        for kind, count in _SPACES.findall(str(line)):
            value = int(count)
            if kind.upper() == "PKW":
                car = value
            else:
                lorry = value
    return car, lorry


def parse_record(row: dict[str, Any], road: str) -> RestArea | None:
    """One API record, checked. None when it cannot be placed on a map."""
    identifier = str(row.get("identifier") or "").strip()
    if not identifier:
        return None

    coordinate = row.get("coordinate") or {}
    pair = coordinate.get("coordinates") or []
    if len(pair) < 2:
        return None
    try:
        # GeoJSON: longitude first. Reading these in the other order silently relocates
        # every German rest area, and nothing downstream would raise.
        lon, lat = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    if lat == 0.0 and lon == 0.0:
        return None

    car, lorry = parse_spaces(row.get("description"))
    problems: list[str] = []
    if car > _MAX_PLAUSIBLE_SPACES or lorry > _MAX_PLAUSIBLE_SPACES:
        problems.append(f"implausible capacity: {car} car, {lorry} lorry")
        car = lorry = 0

    # The API's `title` is "A8 | undefined", with the word undefined coming straight from
    # its own frontend. The subtitle is the human name.
    name = str(row.get("subtitle") or "").strip() or identifier

    return RestArea(
        identifier=identifier,
        road=road,
        name=name,
        lat=lat,
        lon=lon,
        car_spaces=car,
        lorry_spaces=lorry,
        # A JSON string, not a boolean. Every non-empty string is truthy, so reading this
        # without the comparison closes every rest area on the network.
        blocked=str(row.get("isBlocked", "")).strip().lower() == "true",
        problems=tuple(problems),
    )


class AutobahnAdapter(BaseAdapter):
    """Static parking capacity across the German motorway network."""

    meta = SourceMeta(
        name="Autobahn",
        url=f"{BASE_URL}/",
        licence="Datenlizenz Deutschland Namensnennung 2.0",
        licence_url="https://www.govdata.de/dl-de/by-2-0",
        attribution="Autobahn GmbH des Bundes",
        commercial_use=True,
        share_alike=False,
        refresh="daily",
        contact="https://verkehr.autobahn.de/",
        notes=(
            "Capacity only, never occupancy: the feed says how many spaces exist, not "
            "how many are free. Observations are recorded as STATIC_DATABASE."
        ),
    )

    def roads(self) -> list[str]:
        payload = self.fetch_json(f"{BASE_URL}/")
        return [str(r).strip() for r in (payload.get("roads") or []) if str(r).strip()]

    def parking_for(self, road: str) -> list[RestArea]:
        payload = self.fetch_json(f"{BASE_URL}/{road}/services/parking_lorry")
        rows = payload.get("parking_lorry") or []
        out: list[RestArea] = []
        for row in rows:
            record = parse_record(row, road)
            if record is not None:
                out.append(record)
        return out

    def run(self, *, roads: list[str] | None = None, **_: Any) -> IngestResult:
        """Walk the network, one request per road.

        108 roads is 108 requests, which is why this is a daily job rather than something
        a search triggers. The list is fetched rather than hard-coded so a new autobahn
        appears without a code change.
        """
        result = IngestResult(source=self.meta.name)
        try:
            names = roads or self.roads()
        except Exception as exc:
            result.errors.append(f"road list unavailable: {exc}")
            result.finished_at = utcnow()
            return result

        observed_at = utcnow()
        seen: dict[str, RestArea] = {}

        for road in names:
            try:
                for record in self.parking_for(road):
                    result.fetched += 1
                    # The same rest area appears on both carriageways of some roads under
                    # one identifier. Last one wins; they carry the same capacity.
                    seen[record.identifier] = record
            except Exception as exc:
                # One road failing is a gap in coverage, not a failed ingest.
                log.warning("autobahn %s: %s", road, exc)
                result.errors.append(f"{road}: {exc}")

        with session_scope() as session:
            self._register_licence(session)
            existing = {
                facility.external_id: facility
                for facility in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for identifier, record in seen.items():
                facility = existing.get(identifier)
                if facility is None:
                    facility = ParkingFacility(
                        source_name=self.meta.name, external_id=identifier
                    )
                    existing[identifier] = facility
                    session.add(facility)
                    result.created += 1
                else:
                    result.updated += 1

                facility.name = f"{record.road} {record.name}"[:300]
                facility.kind = record.kind.value
                facility.lat = record.lat
                facility.lon = record.lon
                facility.capacity = record.car_spaces or None
                facility.country = "DE"
                facility.currency = "EUR"
                facility.active = not record.blocked
                facility.fetched_at = observed_at
                facility.source_record_id = identifier
                for problem in record.problems:
                    result.errors.append(f"{identifier}: {problem}")

            session.flush()

            for identifier, record in seen.items():
                facility = existing.get(identifier)
                if facility is None or facility.id is None or record.car_spaces <= 0:
                    continue
                session.add(
                    AvailabilityObservation(
                        target_kind="facility",
                        target_id=facility.id,
                        observed_at=observed_at,
                        # Capacity, not occupancy. The feed never says how many are free,
                        # so this is the rung for "somebody wrote this down", and
                        # vacant_spaces stays unset rather than being invented.
                        evidence_source=int(EvidenceSource.STATIC_DATABASE),
                        state=OccupancyState.UNKNOWN.value,
                        total_spaces=record.car_spaces,
                        confidence=0.35,
                        source_name=self.meta.name,
                    )
                )

        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    def _register_licence(self, session) -> None:
        row = session.execute(
            select(SourceLicence).where(SourceLicence.source_name == self.meta.name)
        ).scalar_one_or_none()
        if row is None:
            row = SourceLicence(source_name=self.meta.name)
            session.add(row)
        row.dataset_url = self.meta.url
        row.licence = self.meta.licence
        row.licence_url = self.meta.licence_url
        row.attribution_text = self.meta.attribution
        row.commercial_use = self.meta.commercial_use
        row.share_alike = self.meta.share_alike
        row.refresh_frequency = self.meta.refresh
        row.data_contact = self.meta.contact
        row.notes = self.meta.notes
        row.last_reviewed = utcnow()
