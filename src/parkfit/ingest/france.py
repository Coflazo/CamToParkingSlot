"""France: the national off-street parking base, from the transport data access point.

The fourth country, and the one whose dataset answers the question this product exists
for better than any of the other three. **Every one of its 826 sites publishes a height
limit**, where RDW publishes one for 909 of 3,137 Dutch rows and Autobahn publishes none
at all. Height is the dimension a barrier physically stops a van at, so a source that
states it for every site is worth more here than one with ten times the rows.

What it does not have is occupancy. Like the German feed and unlike the Dutch and Turkish
ones, it says how many spaces exist, so observations land at ``STATIC_DATABASE`` and
``vacant_spaces`` stays unset rather than being invented.

Three columns need care.

**``hauteur_max`` is centimetres, and zero means unpublished.** The median is 190, which
is the classic French underground garage limit, not 1.9 m read as metres. Zero is the same
trap RDW sets: it means nobody wrote a limit down, never that there is no limit, and
reading it as unlimited routes a van into a barrier. One row reads 1905, which is not a
19-metre car park, so implausible values become NULL rather than a claim.

**``Xlong`` and ``Ylat`` are named, which is a kindness.** The Istanbul and German feeds
both put longitude first without saying so, and this one says so.

**The file is semicolon-delimited with a BOM**, which is the French CSV convention and
which a default ``csv.reader`` gets wrong in two separate ways.
"""

from __future__ import annotations

import csv
import io
import logging
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

#: Base nationale des lieux de stationnement hors voirie, via the national access point.
BNLS_URL = "https://transport.data.gouv.fr/resources/78899/download"

#: Above this a "height limit" is not a barrier a car meets; it is a typing error. The
#: real file contains a 1905, which would otherwise be recorded as a 19 metre clearance.
_MAX_PLAUSIBLE_HEIGHT_CM = 600.0
#: Below this it is not a car park entrance either. A 50 cm limit is a bollard.
_MIN_PLAUSIBLE_HEIGHT_CM = 120.0

#: A site larger than this is a data error. The largest real one in the file is 3,427.
_MAX_PLAUSIBLE_SPACES = 20000

#: ``type_ouvrage``. "ouvrage" is a structure, "enclos_en_surface" a fenced surface lot,
#: and a third of rows say nothing, which stays UNKNOWN rather than being guessed at.
_KIND_BY_TYPE = {
    "ouvrage": FacilityKind.GARAGE,
    "enclos_en_surface": FacilityKind.SURFACE_LOT,
}


@dataclass(frozen=True)
class FrenchSite:
    """One row of the national base, checked."""

    external_id: str
    name: str
    lat: float
    lon: float
    spaces: int
    disabled_spaces: int
    ev_spaces: int
    max_height_cm: float | None
    free: bool
    kind: FacilityKind
    address: str
    tariff_1h_eur: float | None
    tariff_24h_eur: float | None
    problems: tuple[str, ...] = ()


def _number(value: str | None) -> float | None:
    """A French-formatted number, or None. Accepts both decimal separators."""
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _count(value: str | None) -> int:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else 0


def parse_height_cm(value: str | None) -> tuple[float | None, str | None]:
    """The barrier height in centimetres, plus a complaint if the value is not credible.

    Returns ``(None, None)`` for an unpublished limit, which is deliberately the same
    answer as an implausible one from the fit engine's point of view: both mean the height
    is unverified, and unverified is not the same as unlimited.
    """
    height = _number(value)
    if height is None or height <= 0.0:
        return None, None
    if height > _MAX_PLAUSIBLE_HEIGHT_CM:
        return None, f"height {height:.0f} cm is not a barrier a car meets"
    if height < _MIN_PLAUSIBLE_HEIGHT_CM:
        return None, f"height {height:.0f} cm is a bollard, not a car park entrance"
    return height, None


def parse_row(row: dict[str, str]) -> FrenchSite | None:
    """One CSV row. None when it cannot be placed on a map."""
    external_id = (row.get("id") or "").strip()
    if not external_id:
        return None

    lat = _number(row.get("Ylat"))
    lon = _number(row.get("Xlong"))
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    if lat == 0.0 and lon == 0.0:
        return None

    problems: list[str] = []
    height, complaint = parse_height_cm(row.get("hauteur_max"))
    if complaint:
        problems.append(complaint)

    spaces = _count(row.get("nb_places"))
    if spaces > _MAX_PLAUSIBLE_SPACES:
        problems.append(f"capacity {spaces} is beyond anything plausible")
        spaces = 0

    return FrenchSite(
        external_id=external_id,
        name=(row.get("nom") or "").strip() or external_id,
        lat=lat,
        lon=lon,
        spaces=spaces,
        disabled_spaces=_count(row.get("nb_pmr")),
        ev_spaces=_count(row.get("nb_voitures_electriques")),
        max_height_cm=height,
        # "1" is free. Anything else, including blank, is treated as paid, because
        # telling a driver a paid car park is free is the more expensive mistake.
        free=(row.get("gratuit") or "").strip() == "1",
        kind=_KIND_BY_TYPE.get((row.get("type_ouvrage") or "").strip(), FacilityKind.UNKNOWN),
        address=(row.get("adresse") or "").strip(),
        tariff_1h_eur=_number(row.get("tarif_1h")),
        tariff_24h_eur=_number(row.get("tarif_24h")),
        problems=tuple(problems),
    )


def parse_csv(text: str) -> list[FrenchSite]:
    """Every usable row of the national base.

    Semicolon-delimited with a UTF-8 BOM, which is the French convention. A default
    reader gets the delimiter wrong and leaves the BOM glued to the first column name,
    so the first field silently never matches.
    """
    handle = io.StringIO(text.lstrip("﻿"))
    reader = csv.DictReader(handle, delimiter=";")
    sites: list[FrenchSite] = []
    for row in reader:
        site = parse_row(row)
        if site is not None:
            sites.append(site)
    return sites


class FranceAdapter(BaseAdapter):
    """The French national off-street parking base."""

    meta = SourceMeta(
        name="BNLS",
        url=BNLS_URL,
        licence="Licence Ouverte 2.0",
        licence_url="https://www.etalab.gouv.fr/licence-ouverte-open-licence",
        attribution="Base nationale des lieux de stationnement, transport.data.gouv.fr",
        commercial_use=True,
        share_alike=False,
        refresh="irregular",
        contact="https://transport.data.gouv.fr/",
        notes=(
            "Capacity and height limits, never occupancy. Explicitly non-exhaustive: it "
            "is the sites that have been declared, not every car park in France."
        ),
    )

    def fetch_sites(self) -> list[FrenchSite]:
        return parse_csv(self.fetch_text(BNLS_URL))

    def run(self, **_: Any) -> IngestResult:
        result = IngestResult(source=self.meta.name)
        try:
            sites = self.fetch_sites()
        except Exception as exc:
            result.errors.append(f"download failed: {exc}")
            result.finished_at = utcnow()
            return result

        result.fetched = len(sites)
        observed_at = utcnow()

        with session_scope() as session:
            self._register_licence(session)
            existing = {
                facility.external_id: facility
                for facility in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for site in sites:
                facility = existing.get(site.external_id)
                if facility is None:
                    facility = ParkingFacility(
                        source_name=self.meta.name, external_id=site.external_id
                    )
                    existing[site.external_id] = facility
                    session.add(facility)
                    result.created += 1
                else:
                    result.updated += 1

                facility.name = site.name[:300]
                facility.kind = site.kind.value
                facility.lat = site.lat
                facility.lon = site.lon
                facility.street = site.address[:200] or None
                facility.capacity = site.spaces or None
                facility.disabled_capacity = site.disabled_spaces or None
                facility.charging_capacity = site.ev_spaces or None
                # None where the limit is unpublished or implausible. The fit engine
                # answers UNVERIFIED for that, which is the honest verdict; inventing a
                # limit here is what drives a van into a barrier.
                facility.max_vehicle_height_cm = site.max_height_cm
                facility.tariff_eur_per_hour = 0.0 if site.free else site.tariff_1h_eur
                facility.tariff_day_max_eur = 0.0 if site.free else site.tariff_24h_eur
                facility.country = "FR"
                facility.currency = "EUR"
                facility.active = True
                facility.fetched_at = observed_at
                facility.source_record_id = site.external_id

                for problem in site.problems:
                    result.errors.append(f"{site.external_id}: {problem}")

            session.flush()

            for site in sites:
                facility = existing.get(site.external_id)
                if facility is None or facility.id is None or site.spaces <= 0:
                    continue
                session.add(
                    AvailabilityObservation(
                        target_kind="facility",
                        target_id=facility.id,
                        observed_at=observed_at,
                        evidence_source=int(EvidenceSource.STATIC_DATABASE),
                        state=OccupancyState.UNKNOWN.value,
                        total_spaces=site.spaces,
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
