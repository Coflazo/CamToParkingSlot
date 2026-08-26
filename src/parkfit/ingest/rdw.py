"""RDW / National Parking Register adapter.

RDW publishes the national parking register as eight separate Socrata tables that all
join on ``(areamanagerid, areaid)``. This adapter reassembles them into whole facilities.

Two facts about the real data shape the design:

**Most facilities have no published height barrier.** Of 3137 specification rows only
909 carry a non-zero ``maximumvehicleheight``; the other 2225 store a literal ``0``.
Several geolocated Amsterdam garages -- IJ-oever, De Bijenkorf -- have no specification
row at all. So ``UNVERIFIED`` is the common case rather than an edge case, and treating
a missing or zero height as "unlimited" would cheerfully route a van into a 2.0 m
barrier. Zero is mapped to ``None``, never to infinity.

**Only 371 of 14748 areas carry coordinates.** The GEO garage and park-and-ride layers
are the only ones with a location. Everything else has to be geocoded from its name and
municipality, which is a separate, slower pass -- so areas land in the database
un-geocoded and inactive rather than being silently dropped.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from parkfit.ingest.base import (
    IngestResult,
    SocrataAdapter,
    SourceMeta,
    parse_bool,
    parse_dutch_weekday,
    parse_float,
    parse_hhmm,
    parse_int,
    parse_rdw_datetime,
)
from parkfit.storage.models import (
    AreaManager,
    FacilityKind,
    OpeningHours,
    ParkingFacility,
    SourceLicence,
    utcnow,
)
from parkfit.storage.session import session_scope

log = logging.getLogger(__name__)

# Verified reachable 2026-08-26.
DATASET_INDEX = "f6v7-gjpa"  # organisations and their static/dynamic feed URLs
DATASET_AREA = "8u4d-s4q7"  # GEBIED            14748 rows
DATASET_SPECS = "b3us-f26s"  # SPECIFICATIES      3137 rows
DATASET_USAGE = "mz4f-59fw"  # GEBRUIKSDOEL      14691 rows
DATASET_GEO_GARAGE = "t5pc-eb34"  # GEO garages         237 rows
DATASET_GEO_PR = "6wzd-evwu"  # GEO park-and-ride   134 rows
DATASET_MANAGER = "2uc2-nnv3"  # GEBIEDSBEHEERDER    461 rows
DATASET_OPEN = "figd-gux7"  # PARKING OPEN       2824 rows
DATASET_ACCESS = "edv8-qiyg"  # PARKING TOEGANG    4826 rows

#: Exact usage codes, checked before the prefix heuristic below.
USAGE_EXACT: dict[str, FacilityKind] = {
    "GARAGEP": FacilityKind.GARAGE,
    "PARKRIDE": FacilityKind.PARK_AND_RIDE,
    "CARPOOL": FacilityKind.PARK_AND_RIDE,
    "TERREINP": FacilityKind.SURFACE_LOT,
    "TEREINP": FacilityKind.SURFACE_LOT,  # a real typo in the upstream data
    "PRTERREIN": FacilityKind.SURFACE_LOT,
    "TRUCKP": FacilityKind.TRUCK_PARKING,
    "BETAALDP": FacilityKind.ON_STREET_ZONE,
    "BETAALDPO": FacilityKind.ON_STREET_ZONE,
    "PARKEREN": FacilityKind.ON_STREET_ZONE,
    "BLAUWEZ": FacilityKind.ON_STREET_ZONE,
    "BLAUWEZN": FacilityKind.ON_STREET_ZONE,
    "BLAUWZONE": FacilityKind.ON_STREET_ZONE,
    "BLAUWEZONE": FacilityKind.ON_STREET_ZONE,
    "BLZONE": FacilityKind.ON_STREET_ZONE,
    "BZONE": FacilityKind.ON_STREET_ZONE,
}

#: Prefixes for the long tail. RDW lets each municipality mint its own codes, so the
#: 14691 usage rows contain over 200 distinct values, most appearing once or twice.
#: Anything permit-shaped is still on-street parking as far as a driver is concerned;
#: what makes it unusable is the permit requirement, which is recorded as a restriction.
USAGE_PREFIX: tuple[tuple[str, FacilityKind], ...] = (
    ("VERGUN", FacilityKind.ON_STREET_ZONE),
    ("VERG_", FacilityKind.ON_STREET_ZONE),
    ("VERGP", FacilityKind.ON_STREET_ZONE),
    ("BEWONER", FacilityKind.ON_STREET_ZONE),
    ("BEWO", FacilityKind.ON_STREET_ZONE),
    ("BEZOEK", FacilityKind.ON_STREET_ZONE),
    ("BEZ", FacilityKind.ON_STREET_ZONE),
    ("BEDRIJF", FacilityKind.ON_STREET_ZONE),
    ("ONTHEF", FacilityKind.ON_STREET_ZONE),
    ("ONTH", FacilityKind.ON_STREET_ZONE),
    ("GPK", FacilityKind.ON_STREET_ZONE),
    ("GEHANDIC", FacilityKind.ON_STREET_ZONE),
    ("GEHNDIC", FacilityKind.ON_STREET_ZONE),
    ("DEELAUT", FacilityKind.ON_STREET_ZONE),
    ("AUTODEL", FacilityKind.ON_STREET_ZONE),
    ("AUTODP", FacilityKind.ON_STREET_ZONE),
)

#: Usage codes that imply the bays are reserved and a visiting driver cannot use them.
PERMIT_USAGE_MARKERS = ("VERGUN", "VERG_", "BEWONER", "BEWO", "ONTHEF", "ONTH", "GPK", "GEHANDIC")


def classify_usage(usage_id: str | None) -> FacilityKind:
    if not usage_id:
        return FacilityKind.UNKNOWN
    code = usage_id.strip().upper()
    if code in USAGE_EXACT:
        return USAGE_EXACT[code]
    for prefix, kind in USAGE_PREFIX:
        if code.startswith(prefix):
            return kind
    return FacilityKind.UNKNOWN


def usage_requires_permit(usage_id: str | None) -> bool:
    if not usage_id:
        return False
    code = usage_id.strip().upper()
    return any(code.startswith(p) for p in PERMIT_USAGE_MARKERS)


def height_cm_from_rdw(raw: Any) -> float | None:
    """Interpret ``maximumvehicleheight``.

    RDW stores this in centimetres and uses ``0`` for "not published". Mapping zero to
    ``None`` rather than to a number is the whole point: the fit engine must be able to
    tell "no barrier recorded" apart from "a barrier of height X", because only one of
    those is safe to clear a vehicle against.
    """
    value = parse_float(raw)
    if value is None or value <= 0:
        return None
    # A handful of rows carry metres (2.1) instead of centimetres (210).
    if value < 10:
        value *= 100.0
    return value


class RdwAdapter(SocrataAdapter):
    """Builds parking facilities from the RDW national register."""

    meta = SourceMeta(
        name="RDW-NPR",
        url="https://opendata.rdw.nl/",
        licence="CC0-1.0 / Public Domain",
        licence_url="https://www.rdw.nl/over-rdw/dienstverlening/open-data",
        attribution="Data: RDW / Nationaal Parkeer Register",
        commercial_use=True,
        share_alike=False,
        refresh="daily",
        contact="https://www.nationaalparkeerregister.nl/",
        notes="Eight Socrata tables joined on (areamanagerid, areaid).",
    )

    # -- lookups ------------------------------------------------------------
    def _load_geo(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Coordinates, from the two GEO layers. Only 371 areas have any."""
        geo: dict[tuple[str, str], dict[str, Any]] = {}
        for dataset in (DATASET_GEO_GARAGE, DATASET_GEO_PR):
            for row in self.socrata_rows(dataset):
                loc = row.get("location") or {}
                lat = parse_float(loc.get("latitude"))
                lon = parse_float(loc.get("longitude"))
                if lat is None or lon is None:
                    coords = loc.get("coordinates")
                    if isinstance(coords, list) and len(coords) == 2:
                        lon, lat = parse_float(coords[0]), parse_float(coords[1])
                if lat is None or lon is None:
                    continue
                key = (str(row.get("areamanagerid", "")), str(row.get("areaid", "")))
                geo[key] = {
                    "lat": lat,
                    "lon": lon,
                    "usage": row.get("usageid"),
                    "name": row.get("areadesc"),
                }
        return geo

    def _load_specs(self) -> dict[tuple[str, str], dict[str, Any]]:
        specs: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.socrata_rows(DATASET_SPECS):
            key = (str(row.get("areamanagerid", "")), str(row.get("areaid", "")))
            # Several areas carry more than one specification row over time; the newest
            # start date wins, because an old barrier height is worse than none.
            existing = specs.get(key)
            started = parse_rdw_datetime(row.get("startdatespecifications"))
            if (
                existing
                and existing.get("_started")
                and started
                and (started < existing["_started"])
            ):
                continue
            specs[key] = {
                "capacity": parse_int(row.get("capacity")),
                "charging": parse_int(row.get("chargingpointcapacity")),
                "disabled": parse_int(row.get("disabledaccess")),
                "max_height_cm": height_cm_from_rdw(row.get("maximumvehicleheight")),
                "limited_access": parse_bool(row.get("limitedaccess")) or False,
                "_started": started,
            }
        return specs

    def _load_usage(self) -> dict[tuple[str, str], list[str]]:
        usage: dict[tuple[str, str], list[str]] = {}
        for row in self.socrata_rows(DATASET_USAGE):
            key = (str(row.get("areamanagerid", "")), str(row.get("areaid", "")))
            code = row.get("usageid")
            if code:
                usage.setdefault(key, []).append(str(code))
        return usage

    def _load_open(self) -> dict[tuple[str, str], dict[str, Any]]:
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.socrata_rows(DATASET_OPEN):
            key = (str(row.get("areamanagerid", "")), str(row.get("areaid", "")))
            out[key] = {
                "exit_all_day": parse_bool(row.get("exitpossibleallday")),
                "open_all_year": parse_bool(row.get("openallyear")),
            }
        return out

    def _load_access(self) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
        """Access windows as ``(weekday, open_minute, close_minute)``."""
        out: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
        for row in self.socrata_rows(DATASET_ACCESS):
            key = (str(row.get("areamanagerid", "")), str(row.get("areaid", "")))
            weekday = parse_dutch_weekday(row.get("days"))
            start = parse_hhmm(row.get("enterfrom"))
            end = parse_hhmm(row.get("enteruntil"))
            if weekday is None or start is None or end is None:
                continue
            # A window that ends at or before it starts runs past midnight; express it
            # as minutes past the window start so overnight stays are a comparison.
            if end <= start:
                end += 24 * 60
            out.setdefault(key, []).append((weekday, start, end))
        return out

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Static and dynamic feed URLs per organisation."""
        out: dict[str, dict[str, Any]] = {}
        for row in self.socrata_rows(DATASET_INDEX):
            org_id = str(row.get("organization_id", ""))
            if not org_id:
                continue
            out[org_id] = {
                "static_url": row.get("url_static_parking_data"),
                "dynamic_url": (
                    row.get("url_dynamic_parking_data")
                    if parse_bool(row.get("dynamic_parking_data"))
                    else None
                ),
            }
        return out

    # -- run ----------------------------------------------------------------
    def run(self, *, limit: int | None = None, geocoded_only: bool = False) -> IngestResult:
        result = IngestResult(source=self.meta.name)

        log.info("RDW: loading lookup tables")
        geo = self._load_geo()
        specs = self._load_specs()
        usage = self._load_usage()
        opening = self._load_open()
        access = self._load_access()
        index = self._load_index()
        log.info(
            "RDW: geo=%d specs=%d usage=%d open=%d access=%d",
            len(geo),
            len(specs),
            len(usage),
            len(opening),
            len(access),
        )

        with session_scope() as session:
            self._register_licence(session)
            managers = self._sync_managers(session, index)

            existing = {
                (f.source_name, f.external_id): f
                for f in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for row in self.socrata_rows(DATASET_AREA, limit=limit):
                result.fetched += 1
                amid = str(row.get("areamanagerid", ""))
                aid = str(row.get("areaid", ""))
                if not amid or not aid:
                    result.skipped += 1
                    continue

                key = (amid, aid)
                external_id = f"{amid}:{aid}"
                geo_row = geo.get(key)
                if geocoded_only and geo_row is None:
                    result.skipped += 1
                    continue

                spec = specs.get(key, {})
                codes = usage.get(key, [])
                primary_code = (geo_row or {}).get("usage") or (codes[0] if codes else None)

                kind = classify_usage(primary_code)
                if kind is FacilityKind.UNKNOWN:
                    for code in codes:
                        kind = classify_usage(code)
                        if kind is not FacilityKind.UNKNOWN:
                            break

                name = row.get("areadesc") or (geo_row or {}).get("name") or aid
                city = _city_from_areadesc(name) or managers.get(amid, {}).get("city")

                facility = existing.get((self.meta.name, external_id))
                created = facility is None
                if facility is None:
                    facility = ParkingFacility(source_name=self.meta.name, external_id=external_id)
                    session.add(facility)

                facility.area_manager_id = amid
                facility.name = str(name)[:300]
                facility.kind = kind.value
                facility.city = city
                facility.capacity = spec.get("capacity")
                facility.charging_capacity = spec.get("charging")
                facility.disabled_capacity = spec.get("disabled")
                facility.max_vehicle_height_cm = spec.get("max_height_cm")
                facility.limited_access = bool(spec.get("limited_access", False))
                facility.source_updated_at = parse_rdw_datetime(row.get("startdatearea"))
                facility.fetched_at = utcnow()

                if geo_row:
                    facility.lat = geo_row["lat"]
                    facility.lon = geo_row["lon"]
                    facility.geocode_precision = "source"

                hours = opening.get(key, {})
                facility.open_all_year = hours.get("open_all_year")
                facility.exit_possible_all_day = hours.get("exit_all_day")

                # A facility is only searchable once it has a position. The rest stay in
                # the table so a later geocoding pass can find them, but they never reach
                # a result list in a state where we cannot say where they are.
                facility.active = facility.lat is not None and facility.lon is not None

                session.flush()
                self._sync_opening_hours(session, facility, access.get(key, []))

                if created:
                    result.created += 1
                    existing[(self.meta.name, external_id)] = facility
                else:
                    result.updated += 1

                if result.fetched % 2000 == 0:
                    session.flush()
                    log.info("RDW: %d areas processed", result.fetched)

        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    # -- helpers ------------------------------------------------------------
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

    def _sync_managers(self, session, index: dict[str, dict[str, Any]]) -> dict[str, dict]:
        existing = {m.area_manager_id: m for m in session.execute(select(AreaManager)).scalars()}
        info: dict[str, dict] = {}
        for row in self.socrata_rows(DATASET_MANAGER):
            amid = str(row.get("areamanagerid", ""))
            if not amid:
                continue
            manager = existing.get(amid)
            if manager is None:
                manager = AreaManager(area_manager_id=amid, source_name=self.meta.name)
                session.add(manager)
                existing[amid] = manager
            manager.name = str(row.get("areamanagerdesc") or amid)[:200]
            manager.url = row.get("url")
            manager.valid_from = parse_rdw_datetime(row.get("startdateareamanagerid"))
            manager.valid_until = parse_rdw_datetime(row.get("enddateareamanagerid"))
            feeds = index.get(amid, {})
            manager.static_data_url = feeds.get("static_url")
            manager.dynamic_data_url = feeds.get("dynamic_url")
            # The manager name is almost always the municipality, which is the best
            # available city hint for areas whose own name does not carry one.
            info[amid] = {"city": manager.name}
        session.flush()
        return info

    @staticmethod
    def _sync_opening_hours(session, facility: ParkingFacility, windows) -> None:
        if not windows:
            return
        for existing in list(facility.opening_hours):
            session.delete(existing)
        seen: set[tuple[int, int, int]] = set()
        for weekday, start, end in windows:
            if (weekday, start, end) in seen:
                continue
            seen.add((weekday, start, end))
            session.add(
                OpeningHours(
                    facility_id=facility.id,
                    weekday=weekday,
                    open_minute=start,
                    close_minute=end,
                )
            )


def _city_from_areadesc(name: str | None) -> str | None:
    """Extract the municipality from names like ``Garage De Bijenkorf (Amsterdam)``.

    RDW has no city column on the area table, but the convention of appending the
    municipality in parentheses is followed widely enough to be worth mining -- and it
    is what lets the geocoding pass disambiguate a "Centrum" garage in one of forty
    Dutch towns that all have one.
    """
    if not name or "(" not in name or not name.rstrip().endswith(")"):
        return None
    inner = name[name.rfind("(") + 1 : -1].strip()
    if not inner or len(inner) > 60 or inner.isdigit():
        return None
    return inner
