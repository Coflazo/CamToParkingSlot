"""ISPARK: Istanbul's municipal parking operator, live.

This is the second country, and it arrives in better shape than the first did. The
Netherlands took two sources to assemble: Amsterdam publishes bay polygons but no
occupancy, and NDW publishes occupancy but only for truck parking. Istanbul publishes
both from one endpoint, keyless, with a timestamp on every reading.

``api.ibb.gov.tr/ispark/Park`` returns every ISPARK site with ``capacity`` and
``emptyCapacity``. That is not a model, an estimate or a camera reading; it is the
operator's own count of its own spaces, which is why observations land at
``EvidenceSource.OPERATOR_FEED``, the same rung NDW occupies and one above anything this
product infers for itself.

``/ParkDetay?id=`` adds the parts that only matter once a driver has chosen: the tariff
ladder, the address, and an ``areaPolygon`` in WKT. The polygon is the interesting one,
because it is the Istanbul counterpart of Amsterdam's surveyed bays and it means the
product can eventually say "this lot" rather than "somewhere around this pin".

Three things about the data are worth knowing before trusting it.

**52 of the sites are ``YOL USTU``, on-street.** Those are exactly the spaces this product
exists for, and they are metered kerbside bays rather than lots. They are kept as
``ON_STREET_ZONE`` rather than flattened into ``SURFACE_LOT``, because the fit and
legality questions differ: a kerbside bay is subject to the setback rules in KTK 2918
articles 60 and 61, and a lot behind a barrier is not.

**Prices are in lira and are never converted.** A search happens in one city, so every
candidate it ranks shares a currency, and inventing an exchange rate in the middle of a
generalised cost would be a made-up number dressed as a measurement.

**``isOpen`` does not mean what it looks like.** The observed values are 0 and 1 across
sites that are plainly operating, including 24-hour ones, so it is recorded and not acted
on. Treating it as "closed" would hide most of the network. This is flagged rather than
guessed at, and the field is stored verbatim for whoever works out what it means.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta
from parkfit.storage.models import (
    AvailabilityObservation,
    DataQualityIncident,
    EvidenceSource,
    FacilityKind,
    OccupancyState,
    ParkingFacility,
    SourceLicence,
    utcnow,
)
from parkfit.storage.session import session_scope

log = logging.getLogger(__name__)

BASE_URL = "https://api.ibb.gov.tr/ispark"

#: ISPARK's own site classification.
#:
#: ``YOL USTU`` is "on the road", which is kerbside metered parking rather than a lot.
#: Keeping the distinction matters: the legality engine applies statutory setbacks to
#: kerbside space and not to a lot behind a barrier.
PARK_TYPE_KIND = {
    "KAPALI OTOPARK": FacilityKind.GARAGE,          # covered
    "ACIK OTOPARK": FacilityKind.SURFACE_LOT,       # open
    "YOL USTU": FacilityKind.ON_STREET_ZONE,        # on-street
}

#: Turkish upper-case folding that ``str.upper()`` gets wrong.
#:
#: Python folds the dotless i and the dotted I by English rules, so the Turkish forms
#: do not round-trip. Comparing park types without this silently drops all 111 open
#: lots into UNKNOWN. The characters below are deliberate, which is why the ambiguous
#: character lint is suppressed rather than the text being sanitised: sanitising it
#: would delete the bug it exists to prevent.
_TR_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "CGIOSUCGIOSU")

#: A lot reporting more free spaces than it has is not a lot with extra spaces; it is a
#: broken feed. Recorded as an incident and clamped rather than propagated.
_MAX_PLAUSIBLE_CAPACITY = 20000


def _fold(value: str) -> str:
    return (value or "").translate(_TR_FOLD).upper().strip()


@dataclass(frozen=True)
class ParkReading:
    """One site's live state, after the sanity checks."""

    park_id: int
    name: str
    lat: float
    lon: float
    capacity: int
    empty: int
    park_type: str
    district: str
    work_hours: str
    free_minutes: int
    problems: tuple[str, ...] = ()

    @property
    def occupied(self) -> int:
        return max(0, self.capacity - self.empty)

    @property
    def ratio(self) -> float | None:
        return (self.occupied / self.capacity) if self.capacity > 0 else None

    @property
    def state(self) -> OccupancyState:
        if self.capacity <= 0:
            return OccupancyState.UNKNOWN
        if self.empty <= 0:
            return OccupancyState.OCCUPIED
        return OccupancyState.VACANT

    @property
    def kind(self) -> FacilityKind:
        return PARK_TYPE_KIND.get(_fold(self.park_type), FacilityKind.UNKNOWN)


def parse_reading(row: dict[str, Any]) -> ParkReading | None:
    """Turn one ``/Park`` row into a checked reading, or None if it is unusable.

    Coordinates arrive as strings and occasionally as empty ones. A site with no position
    cannot be routed to, ranked, or drawn, so it is dropped rather than placed at the
    origin, which is in the Atlantic.
    """
    try:
        park_id = int(row["parkID"])
        lat = float(row["lat"])
        lon = float(row["lon"] if "lon" in row else row["lng"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    if lat == 0.0 and lon == 0.0:
        return None

    capacity = _int(row.get("capacity"))
    empty = _int(row.get("emptyCapacity"))
    problems: list[str] = []

    if capacity > _MAX_PLAUSIBLE_CAPACITY:
        problems.append(f"capacity {capacity} is beyond anything plausible")
        capacity = 0
    if empty > capacity:
        # Seen in feeds of this shape when a site is being reconfigured. Reporting more
        # free spaces than the site has is a bug in the source, and passing it through
        # would put an impossible number in front of a driver.
        problems.append(f"emptyCapacity {empty} exceeds capacity {capacity}")
        empty = capacity
    if empty < 0:
        problems.append(f"negative emptyCapacity {empty}")
        empty = 0

    return ParkReading(
        park_id=park_id,
        name=str(row.get("parkName") or f"ISPARK {park_id}").strip(),
        lat=lat,
        lon=lon,
        capacity=capacity,
        empty=empty,
        park_type=str(row.get("parkType") or ""),
        district=str(row.get("district") or "").strip(),
        work_hours=str(row.get("workHours") or "").strip(),
        free_minutes=_int(row.get("freeTime")),
        problems=tuple(problems),
    )


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


#: "0-1 Saat : 110,00;1-2 Saat : 140,00;..." is the tariff shape ISPARK publishes.
_TARIFF_ROW = re.compile(r"(\d+)\s*-\s*(\d+)\s*Saat\s*:\s*([\d.,]+)", re.IGNORECASE)


def parse_tariff(text: str) -> tuple[float | None, str]:
    """The first hour's price in lira, plus the ladder kept verbatim.

    Only the first band is turned into a number. The ladder is not linear (110 lira for
    the first hour, 370 for a full day), so multiplying an hourly rate by eight would
    overstate a day by more than double. The full string is stored so the real price can
    be shown, and the single number is only ever a first-hour estimate.

    Turkish decimals use a comma, and a naive ``float("110,00")`` raises while
    ``float("1.234,00")`` would silently read as 1.234.
    """
    if not text:
        return None, ""
    match = _TARIFF_ROW.search(text)
    if not match:
        return None, text.strip()
    raw = match.group(3).replace(".", "").replace(",", ".")
    try:
        price = float(raw)
    except ValueError:
        return None, text.strip()
    return (price if price > 0 else None), text.strip()


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    """The outer ring of a WKT ``POLYGON``, as (lat, lon) pairs.

    ISPARK writes coordinates in WKT order, which is longitude first, and everything
    downstream in this product is latitude first. Getting that backwards puts Istanbul in
    Somalia, so the swap happens here, once, rather than at each call site.
    """
    if not wkt or "POLYGON" not in wkt.upper():
        return []
    start = wkt.find("((")
    end = wkt.find("))", start)
    if start < 0 or end < 0:
        return []
    ring: list[tuple[float, float]] = []
    for pair in wkt[start + 2 : end].split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            continue
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        ring.append((lat, lon))
    return ring


#: ISPARK stamps its detail records "dd.MM.yyyy HH:mm:ss" in Istanbul local time.
_UPDATED_AT = "%d.%m.%Y %H:%M:%S"
#: Turkey has been on permanent UTC+3 since 2016, with no daylight saving, which is why a
#: fixed offset is correct here rather than a lazy shortcut.
_ISTANBUL_OFFSET_HOURS = 3


def parse_updated_at(text: str) -> datetime | None:
    if not text:
        return None
    try:
        naive = datetime.strptime(text.strip(), _UPDATED_AT)
    except ValueError:
        return None
    from datetime import timedelta

    return (naive - timedelta(hours=_ISTANBUL_OFFSET_HOURS)).replace(tzinfo=UTC)


class IsparkAdapter(BaseAdapter):
    """Live occupancy, geometry and tariffs for Istanbul's municipal parking."""

    meta = SourceMeta(
        name="ISPARK",
        url=f"{BASE_URL}/Park",
        licence="IBB Open Data Licence",
        licence_url="https://data.ibb.gov.tr/license",
        attribution="Istanbul Buyuksehir Belediyesi (IBB) / ISPARK",
        commercial_use=None,  # the IBB licence is not a standard SPDX one; not asserted
        share_alike=False,
        refresh="continuous",
        contact="https://data.ibb.gov.tr/",
        notes=(
            "Keyless public endpoint operated by the municipality. capacity and "
            "emptyCapacity are the operator's own counts. Prices are in TRY and are "
            "never converted."
        ),
    )

    def fetch_sites(self) -> list[dict[str, Any]]:
        payload = self.fetch_json(f"{BASE_URL}/Park")
        # The endpoint answers with a bare list, not an envelope.
        return payload if isinstance(payload, list) else []

    def fetch_detail(self, park_id: int) -> dict[str, Any] | None:
        """Tariff, address and polygon for one site.

        A detail call per site is 249 requests, so this is deliberately not part of the
        live status run. It is for enrichment, on its own schedule.
        """
        payload = self.fetch_json(f"{BASE_URL}/ParkDetay", {"id": park_id})
        rows = payload if isinstance(payload, list) else []
        if not rows:
            return None
        row = rows[0]
        # An unknown id answers 200 with a placeholder whose parkID is 0 rather than a
        # 404, so the shape has to be checked rather than the status code.
        return row if _int(row.get("parkID")) == park_id else None

    # ------------------------------------------------------------- ingest
    def run(self, **kwargs: Any) -> IngestResult:
        """Refresh the site list and record one observation per site."""
        result = IngestResult(source=self.meta.name)
        rows = self.fetch_sites()
        result.fetched = len(rows)
        observed_at = utcnow()

        with session_scope() as session:
            self._register_licence(session)
            existing = {
                facility.external_id: facility
                for facility in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for row in rows:
                reading = parse_reading(row)
                if reading is None:
                    result.skipped += 1
                    continue

                external_id = str(reading.park_id)
                facility = existing.get(external_id)
                if facility is None:
                    facility = ParkingFacility(
                        source_name=self.meta.name, external_id=external_id
                    )
                    existing[external_id] = facility
                    session.add(facility)
                    result.created += 1
                else:
                    result.updated += 1

                facility.name = reading.name
                facility.kind = reading.kind.value
                facility.lat = reading.lat
                facility.lon = reading.lon
                facility.city = "Istanbul"
                facility.province = reading.district or None
                facility.capacity = reading.capacity or None
                facility.currency = "TRY"
                facility.country = "TR"
                facility.active = True
                facility.fetched_at = observed_at
                facility.source_record_id = external_id

                for problem in reading.problems:
                    result.errors.append(f"{external_id}: {problem}")
                    session.add(
                        DataQualityIncident(
                            source_name=self.meta.name,
                            severity="warning",
                            kind="implausible_occupancy",
                            detail=f"{external_id}: {problem}",
                            target_kind="facility",
                            target_id=facility.id or 0,
                        )
                    )

            # Flush so new facilities have ids before observations reference them.
            session.flush()

            for row in rows:
                reading = parse_reading(row)
                if reading is None or reading.capacity <= 0:
                    continue
                facility = existing.get(str(reading.park_id))
                if facility is None or facility.id is None:
                    continue
                session.add(
                    AvailabilityObservation(
                        target_kind="facility",
                        target_id=facility.id,
                        observed_at=observed_at,
                        evidence_source=int(EvidenceSource.OPERATOR_FEED),
                        state=reading.state.value,
                        vacant_spaces=reading.empty,
                        occupied_spaces=reading.occupied,
                        total_spaces=reading.capacity,
                        occupancy_ratio=reading.ratio,
                        # The operator counting its own spaces is the best evidence this
                        # product ever gets, short of a camera pointed at the bay.
                        confidence=0.95,
                        source_name=self.meta.name,
                    )
                )

        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    def run_details(self, *, limit: int | None = None) -> IngestResult:
        """Enrich stored sites with tariffs, addresses and polygons.

        Separate from :meth:`run` because it is one request per site. The live status run
        has to stay a single call so it can be repeated often; this can take its time.
        """
        result = IngestResult(source=f"{self.meta.name}-Detail")

        with session_scope() as session:
            facilities = list(
                session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            )
            if limit is not None:
                facilities = facilities[:limit]

            for facility in facilities:
                result.fetched += 1
                try:
                    detail = self.fetch_detail(int(facility.external_id))
                except Exception as exc:
                    result.errors.append(f"{facility.external_id}: {exc}")
                    continue
                if detail is None:
                    result.skipped += 1
                    continue

                hourly, ladder = parse_tariff(str(detail.get("tariff") or ""))
                if hourly is not None:
                    facility.tariff_eur_per_hour = hourly  # lira; see ParkingFacility.currency
                if ladder:
                    facility.tariff_note = ladder
                address = str(detail.get("address") or "").strip()
                if address:
                    facility.street = address[:200]

                ring = parse_wkt_polygon(str(detail.get("areaPolygon") or ""))
                if ring:
                    import json

                    facility.geometry_geojson = json.dumps(
                        {
                            "type": "Polygon",
                            # GeoJSON is longitude first, which is the order the WKT
                            # arrived in and the opposite of what parse_wkt_polygon
                            # returns, so it is swapped back here.
                            "coordinates": [[[lon, lat] for lat, lon in ring]],
                        },
                        separators=(",", ":"),
                    )
                result.updated += 1

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
