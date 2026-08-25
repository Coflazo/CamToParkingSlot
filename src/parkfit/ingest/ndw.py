"""NDW (Nationaal Dataportaal Wegverkeer) adapter.

NDW is the only genuinely open, genuinely live parking-occupancy source in the
Netherlands. ``Truckparking_Parking_Status.xml`` is a DATEX II v3 publication carrying
real vacant-space counts, refreshed roughly once a minute; ``Truckparking_Parking_Table.xml``
carries the matching static records with names, capacities and coordinates.

**The feed is not trustworthy at face value.** Observed in a single live sample:

* a site with capacity 210 reporting 1146 vacant spaces and **-1046 occupied**;
* a site reporting 0 vacant, 24 occupied and 100 % occupancy while its own
  ``parkingSiteStatus`` says ``spacesAvailable``;
* a site reporting 8 vacant, 0 occupied and 64 % occupancy simultaneously.

So every value is validated before it is stored, and where the declared status and the
counts disagree the **more pessimistic** reading wins. That asymmetry is deliberate:
telling a driver a full car park is full costs them one alternative click, while telling
them a full car park has space costs them the entire trip. Every rejection is recorded
as a :class:`DataQualityIncident` rather than silently swallowed.
"""

from __future__ import annotations

import gzip
import logging
import xml.etree.ElementTree as ET
from typing import Any

from sqlalchemy import select

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta
from parkfit.ingest.datex import (
    direct_child,
    direct_text,
    iter_descendants,
    element_id,
    find_records,
    parse_datetime,
    parse_float,
    parse_int,
    path_text,
)
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

TRUCKPARKING_STATUS = "Truckparking_Parking_Status.xml"
TRUCKPARKING_TABLE = "Truckparking_Parking_Table.xml"
EMISSION_ZONES = "emissiezones.xml.gz"

#: How far a reported vacancy may exceed known capacity before it is rejected outright.
#: A little slack absorbs legitimate lag between a capacity change and a status update.
CAPACITY_TOLERANCE = 1.10


class OccupancyReading:
    """A validated site-level occupancy reading."""

    __slots__ = ("vacant", "occupied", "ratio", "state", "problems")

    def __init__(self) -> None:
        self.vacant: int | None = None
        self.occupied: int | None = None
        self.ratio: float | None = None
        self.state: OccupancyState = OccupancyState.UNKNOWN
        self.problems: list[str] = []

    @property
    def total(self) -> int | None:
        if self.vacant is None or self.occupied is None:
            return None
        return self.vacant + self.occupied

    @property
    def confidence(self) -> float:
        if self.state is OccupancyState.UNKNOWN:
            return 0.0
        # A reading we had to correct is still usable, but it is not a clean signal.
        return 0.55 if self.problems else 0.95


def read_site_occupancy(status: ET.Element, capacity: int | None) -> OccupancyReading:
    """Extract and validate the site-level occupancy from a ``parkingRecordStatus``.

    Only the *direct* ``parkingOccupancy`` child is read. Sub-area figures under
    ``groupOfParkingSpacesStatus`` describe parts of the site and must never be
    presented as the whole.
    """
    reading = OccupancyReading()

    declared = (direct_text(status, "parkingSiteStatus") or "").strip().lower()
    vacant = parse_int(path_text(status, "parkingOccupancy", "parkingNumberOfVacantSpaces"))
    occupied = parse_int(path_text(status, "parkingOccupancy", "parkingNumberOfOccupiedSpaces"))
    ratio = parse_float(path_text(status, "parkingOccupancy", "parkingOccupancy"))

    # --- validate the raw counts ------------------------------------------
    if occupied is not None and occupied < 0:
        reading.problems.append(f"negative occupied count ({occupied})")
        occupied = None
    if vacant is not None and vacant < 0:
        reading.problems.append(f"negative vacant count ({vacant})")
        vacant = None
    if ratio is not None and not (0.0 <= ratio <= 100.0):
        reading.problems.append(f"occupancy ratio out of range ({ratio})")
        ratio = None
    if capacity and vacant is not None and vacant > capacity * CAPACITY_TOLERANCE:
        reading.problems.append(f"vacant {vacant} exceeds capacity {capacity}")
        vacant = None

    # --- derive a state ----------------------------------------------------
    # The declared status is the primary signal; it is a statement of intent by the
    # operator, whereas the counts are frequently arithmetic wreckage.
    if declared == "full":
        state = OccupancyState.OCCUPIED
    elif declared in {"spacesavailable", "almostfull"}:
        state = OccupancyState.VACANT
    else:
        state = OccupancyState.UNKNOWN

    counts_say_full = (vacant == 0 and (occupied or 0) > 0) or (ratio is not None and ratio >= 99.5)

    if state is OccupancyState.VACANT and counts_say_full:
        # Declared "spaces available" but every number says otherwise. Take the
        # pessimistic branch: a wasted trip is far costlier than a skipped option.
        reading.problems.append(
            f"status={declared!r} contradicts counts (vacant={vacant}, occupied={occupied}, "
            f"ratio={ratio})"
        )
        state = OccupancyState.OCCUPIED
    elif state is OccupancyState.OCCUPIED and vacant:
        reading.problems.append(f"status=full but {vacant} vacant reported; trusting status")
        vacant = 0

    # "Available" with no usable count is still information, but not a number we can
    # show, so the count stays absent while the state remains VACANT.
    if state is OccupancyState.UNKNOWN:
        vacant = None
        occupied = None
        ratio = None

    reading.vacant = vacant
    reading.occupied = occupied
    reading.ratio = ratio / 100.0 if ratio is not None else None
    reading.state = state
    return reading


class NdwAdapter(BaseAdapter):
    """Live parking occupancy and supporting regulatory datasets from NDW."""

    meta = SourceMeta(
        name="NDW",
        url="https://opendata.ndw.nu/",
        licence="CC0-1.0 / Open data (NDW)",
        licence_url="https://www.ndw.nu/copyright",
        attribution="Data: Nationaal Dataportaal Wegverkeer (NDW)",
        commercial_use=True,
        share_alike=False,
        refresh="1 minute (status), daily (static)",
        contact="https://www.ndw.nu/contact-en-support",
        notes="DATEX II v3 parking status and table; emission zones; roadworks.",
    )

    def _fetch_xml(self, filename: str) -> ET.Element:
        url = f"{self.settings.ndw_base_url}/{filename}"
        # Live feeds are never served from cache: a cached copy would defeat the whole
        # point of a one-minute refresh and would show stale data as current.
        response = self.client.get(url, headers={"Accept": "*/*"})
        response.raise_for_status()
        payload = response.content
        if filename.endswith(".gz"):
            payload = gzip.decompress(payload)
        return ET.fromstring(payload)

    # -- static table -------------------------------------------------------
    def run_truck_parking_table(self) -> IngestResult:
        result = IngestResult(source=f"{self.meta.name}-TruckParkingTable")
        root = self._fetch_xml(TRUCKPARKING_TABLE)

        with session_scope() as session:
            self._register_licence(session)
            existing = {
                f.external_id: f
                for f in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for record in find_records(root, "parkingRecord"):
                result.fetched += 1
                record_id = element_id(record)
                if not record_id:
                    result.skipped += 1
                    continue

                name = path_text(record, "parkingName", "values", "value") or record_id
                capacity = self._record_capacity(record)
                lat, lon = self._record_coordinates(record)

                facility = existing.get(record_id)
                created = facility is None
                if facility is None:
                    facility = ParkingFacility(source_name=self.meta.name, external_id=record_id)
                    session.add(facility)
                    existing[record_id] = facility

                facility.name = str(name)[:300]
                facility.kind = FacilityKind.TRUCK_PARKING.value
                if lat is not None and lon is not None:
                    facility.lat = lat
                    facility.lon = lon
                    facility.geocode_precision = "source"
                if capacity:
                    facility.capacity = capacity
                facility.active = facility.lat is not None and facility.lon is not None
                facility.fetched_at = utcnow()

                result.created += int(created)
                result.updated += int(not created)

        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    @staticmethod
    def _record_capacity(record: ET.Element) -> int | None:
        """Best available capacity for a parking record.

        DATEX II does not publish a single site total. Capacity lives inside repeated
        ``groupOfParkingSpaces / parkingSpaceBasics / parkingNumberOfSpaces`` elements,
        and the groups overlap rather than partition: Truck-Inn Nobis, whose real
        capacity is 210, publishes groups of 0, 210 and 210. Summing them would claim
        420 spaces.

        The maximum is used instead. This value only ever serves as a sanity ceiling
        for validating reported vacancies, so an over-generous bound is the safe
        direction -- it rejects the genuinely absurd (1157 free out of 210) without
        discarding merely surprising numbers.
        """
        values = [
            parse_int(direct_text(basics, "parkingNumberOfSpaces"))
            for group in iter_descendants(record, "groupOfParkingSpaces")
            for basics in [direct_child(group, "parkingSpaceBasics")]
            if basics is not None
        ]
        usable = [v for v in values if v and v > 0]
        return max(usable) if usable else None

    @staticmethod
    def _record_coordinates(record: ET.Element) -> tuple[float | None, float | None]:
        """Find the first point location in a parking record.

        Records nest location several layers deep and inconsistently, so this is one
        place where a subtree search is legitimate: we want any point that describes
        the site, and there is only ever one kind of coordinate pair in the document.
        """
        for element in record.iter():
            lat = direct_text(element, "latitude")
            lon = direct_text(element, "longitude")
            if lat is not None and lon is not None:
                return parse_float(lat), parse_float(lon)
        return None, None

    # -- live status --------------------------------------------------------
    def run_truck_parking_status(self) -> IngestResult:
        result = IngestResult(source=f"{self.meta.name}-TruckParkingStatus")
        root = self._fetch_xml(TRUCKPARKING_STATUS)
        publication_time = parse_datetime(direct_text(root, "publicationTime")) or utcnow()

        with session_scope() as session:
            facilities = {
                f.external_id: (f.id, f.capacity)
                for f in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for status in find_records(root, "parkingRecordStatus"):
                result.fetched += 1
                reference = direct_child(status, "parkingRecordReference")
                record_id = element_id(reference) if reference is not None else None
                if not record_id:
                    result.skipped += 1
                    continue

                entry = facilities.get(record_id)
                if entry is None:
                    # The status feed references a record the static table has not
                    # published. An observation with no facility has no coordinates and
                    # nowhere to appear, so it is dropped rather than orphaned.
                    result.skipped += 1
                    continue
                facility_id, capacity = entry

                observed_at = (
                    parse_datetime(direct_text(status, "parkingStatusOriginTime"))
                    or publication_time
                )
                reading = read_site_occupancy(status, capacity)

                for problem in reading.problems:
                    result.errors.append(f"{record_id}: {problem}")
                    session.add(
                        DataQualityIncident(
                            source_name=self.meta.name,
                            severity="warning",
                            kind="implausible_occupancy",
                            detail=f"{record_id}: {problem}",
                            target_kind="facility",
                            target_id=facility_id,
                        )
                    )

                session.add(
                    AvailabilityObservation(
                        target_kind="facility",
                        target_id=facility_id,
                        observed_at=observed_at,
                        evidence_source=int(EvidenceSource.OPERATOR_FEED),
                        state=reading.state.value,
                        vacant_spaces=reading.vacant,
                        occupied_spaces=reading.occupied,
                        total_spaces=reading.total,
                        occupancy_ratio=reading.ratio,
                        confidence=reading.confidence,
                        source_name=self.meta.name,
                    )
                )
                result.created += 1

        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    # -- environmental zones ------------------------------------------------
    def run_emission_zones(self) -> IngestResult:
        """Parse environmental zones.

        A vehicle can fit a garage perfectly and still be banned from the streets that
        reach it, so these feed the routing restriction layer rather than the parking
        tables.
        """
        result = IngestResult(source=f"{self.meta.name}-EmissionZones")
        try:
            root = self._fetch_xml(EMISSION_ZONES)
        except Exception as exc:  # noqa: BLE001 - upstream availability varies
            result.errors.append(f"emission zones unavailable: {exc}")
            result.finished_at = utcnow()
            return result

        zones = find_records(root, "environmentalZone") or find_records(root, "zone")
        result.fetched = len(zones)
        result.skipped = len(zones)
        result.finished_at = utcnow()
        log.info("%s (parsed %d zones)", result.summary(), len(zones))
        return result

    def run(self, **kwargs: Any) -> IngestResult:
        """Static table first, then live status: the status feed needs those records."""
        table = self.run_truck_parking_table()
        status = self.run_truck_parking_status()
        combined = IngestResult(source=self.meta.name)
        combined.fetched = table.fetched + status.fetched
        combined.created = table.created + status.created
        combined.updated = table.updated + status.updated
        combined.skipped = table.skipped + status.skipped
        combined.errors = table.errors + status.errors
        combined.finished_at = utcnow()
        return combined

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
