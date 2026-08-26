"""City of Amsterdam ``parkeervakken`` adapter.

This is the most valuable dataset in the whole product. Amsterdam publishes *every
individual parking bay* as an exact polygon in RD New, tagged with its layout
(``Langs`` parallel / ``Haaks`` perpendicular), whether it is metered, its traffic-sign
code, and its time regimes.

That changes what computer vision has to do. Without it, a camera would have to answer
"where is a legal space and how long is it", the research-grade problem. With it,
geometry is a solved data problem and vision is left with the tractable question:
*is this known bay occupied right now?*

The bay corners are also surveyed ground-truth points in a metric coordinate system,
which is exactly what camera homography calibration needs and would otherwise have to
be measured by hand in the street.

Dimensions are computed once here, at ingest, by the rotating-calipers routine, and
stored. A search never re-derives geometry.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from sqlalchemy import delete, select

from parkfit.geo.rd import rd_in_range, rd_to_wgs84, ring_centroid_rd
from parkfit.geo.shapes import measure_bay
from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta
from parkfit.storage.models import (
    BayOrientation,
    ParkingBay,
    ParkingRestriction,
    SourceLicence,
    utcnow,
)
from parkfit.storage.session import session_scope

log = logging.getLogger(__name__)


class _NullContext:
    """Yield a caller-supplied session without owning its lifetime."""

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *exc):
        return False


ORIENTATION_BY_TYPE = {
    "langs": BayOrientation.PARALLEL,
    "haaks": BayOrientation.PERPENDICULAR,
    "visgraat": BayOrientation.ANGLED,
}

DUTCH_DAY_INDEX = {"ma": 0, "di": 1, "wo": 2, "do": 3, "vr": 4, "za": 5, "zo": 6}

#: Dutch traffic-sign codes that matter for whether a visiting driver may park.
#: A bay can be perfectly empty and perfectly illegal, and only the sign says which.
SIGN_DISABLED = {"E6", "E6a", "E6b"}
SIGN_PERMIT = {"E9", "E9a", "E9b"}
SIGN_LOADING = {"E7"}
SIGN_NO_PARKING = {"E1", "E2"}
SIGN_VEHICLE_CATEGORY = {"E8"}

#: Words appearing in the ``bord`` (sign text) field that indicate an EV-only bay.
EV_SIGN_MARKERS = ("opladen", "elektrisch", "laadpunt", "oplaad")


def parse_time_to_minutes(value: str | None) -> int | None:
    """Amsterdam writes regime times as ``HH:MM:SS``."""
    if not value:
        return None
    parts = str(value).strip().split(":")
    try:
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return hours * 60 + minutes


def weekday_mask(days: list[str] | None) -> int:
    """Pack Dutch day abbreviations into a 7-bit mask. Absent means every day."""
    if not days:
        return 0b1111111
    mask = 0
    for day in days:
        idx = DUTCH_DAY_INDEX.get(str(day).strip().lower()[:2])
        if idx is not None:
            mask |= 1 << idx
    return mask or 0b1111111


#: Aspect ratio separating the two layouts when the source omits ``type``.
#: Measured over 57k Amsterdam bays: parallel bays sit near 3.0 (5.51 m / 1.84 m) and
#: perpendicular near 1.9 (4.71 m / 2.48 m), so the two populations barely overlap.
ORIENTATION_RATIO_PARALLEL = 2.4
ORIENTATION_RATIO_PERPENDICULAR = 2.1


def infer_orientation(length_m: float, width_m: float) -> BayOrientation:
    """Derive the layout from the shape of the bay.

    About a third of Amsterdam bays carry no ``type``, and treating those as unknown
    meant applying the strictest reading of both layouts, which rejected almost all of
    them. The geometry itself is unambiguous: a bay three times longer than it is wide
    is kerb-parallel, and one only twice as long is perpendicular.
    """
    if length_m <= 0 or width_m <= 0:
        return BayOrientation.UNKNOWN
    ratio = length_m / width_m
    if ratio >= ORIENTATION_RATIO_PARALLEL:
        return BayOrientation.PARALLEL
    if ratio <= ORIENTATION_RATIO_PERPENDICULAR:
        return BayOrientation.PERPENDICULAR
    return BayOrientation.UNKNOWN


class AmsterdamAdapter(BaseAdapter):
    """Ingests individual parking-bay polygons for Amsterdam."""

    meta = SourceMeta(
        name="Amsterdam-Parkeervakken",
        url="https://api.data.amsterdam.nl/v1/parkeervakken/parkeervakken/",
        licence="CC-BY-4.0 (Gemeente Amsterdam open data)",
        licence_url="https://data.amsterdam.nl/",
        attribution="Data: Gemeente Amsterdam",
        commercial_use=True,
        share_alike=False,
        refresh="weekly",
        contact="https://data.amsterdam.nl/",
        notes="Per-bay polygons in EPSG:28992 with layout, sign code and time regimes.",
    )

    PAGE_SIZE = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hit_page_ceiling = False

    def _pages(self, *, page_size: int, limit: int | None, params: dict[str, Any] | None):
        """Walk the DSO-API pages via their ``next`` link.

        Two upstream behaviours are handled here.

        The API refuses ``Accept: application/json`` outright with a 406 and wants
        ``_format=json`` in the query string instead.

        And it enforces a **deep-pagination ceiling**: page 101 returns 403 Forbidden,
        so plain paging tops out at 100 pages. That is a limit, not a fault, and it is
        reported as such rather than raised, see :meth:`run_all` for the partitioned
        ingest that reaches the whole city.
        """
        url: str | None = self.meta.url
        query: dict[str, Any] | None = {"_pageSize": page_size, "_format": "json"}
        if params:
            query.update(params)

        fetched = 0
        page = 1
        while url:
            try:
                body = self.fetch_json(url, query, headers={"Accept": "*/*"})
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {400, 403} and page > 1:
                    self._hit_page_ceiling = True
                    log.warning(
                        "Amsterdam: pagination ceiling reached at page %d (%s); "
                        "partition the query to read further",
                        page,
                        exc.response.status_code,
                    )
                    return
                raise

            rows = (body.get("_embedded") or {}).get("parkeervakken") or []
            if not rows:
                return
            for row in rows:
                yield row
                fetched += 1
                if limit is not None and fetched >= limit:
                    return
            nxt = (body.get("_links") or {}).get("next") or {}
            url = nxt.get("href")
            # Must be None, not {}. httpx *replaces* a URL query string with `params`,
            # so an empty dict strips the page cursor off the next link and silently
            # re-fetches page 1 forever.
            query = None
            page += 1

    def run(
        self,
        *,
        limit: int | None = None,
        page_size: int | None = None,
        street: str | None = None,
        neighbourhood: str | None = None,
        session=None,
    ) -> IngestResult:
        """Ingest one slice of the dataset.

        Commits every batch. The unit of durability has to be the batch rather than the
        run: an earlier version held a single transaction across 200,000 rows and four
        minutes of network I/O, and when the final request failed the rollback discarded
        every one of them. The logs said "200,000 processed" and the database was empty.
        """
        result = IngestResult(source=self.meta.name)
        params: dict[str, Any] = {}
        if street:
            params["straatnaam"] = street
        if neighbourhood:
            params["buurtcode"] = neighbourhood
        self._hit_page_ceiling = False

        owns_session = session is None
        ctx = session_scope() if owns_session else _NullContext(session)
        with ctx as session:
            if owns_session:
                self._register_licence(session)
            existing = {
                b.external_id: b
                for b in session.execute(
                    select(ParkingBay).where(ParkingBay.source_name == self.meta.name)
                ).scalars()
            }

            batch: list[tuple[ParkingBay, list[dict]]] = []
            for row in self._pages(
                page_size=page_size or self.PAGE_SIZE, limit=limit, params=params
            ):
                result.fetched += 1
                bay, regimes = self._build_bay(row, existing, result)
                if bay is None:
                    continue
                batch.append((bay, regimes))

                if len(batch) >= 500:
                    session.flush()
                    self._sync_restrictions(session, batch)
                    session.commit()
                    batch.clear()
                    if result.fetched % 10000 == 0:
                        log.info("Amsterdam: %d bays processed", result.fetched)

            session.flush()
            self._sync_restrictions(session, batch)
            session.commit()

        if self._hit_page_ceiling:
            result.errors.append("stopped at the API pagination ceiling")
        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    def neighbourhoods(self) -> list[str]:
        """Every buurtcode that has parking bays.

        Read from the bays themselves rather than the gebieden dataset, so the
        partition covers exactly the rows that exist and nothing else.
        """
        codes: set[str] = set()
        body = self.fetch_json(
            self.meta.url,
            {"_pageSize": 1, "_format": "json", "_fields": "buurtcode"},
            headers={"Accept": "*/*"},
        )
        if body:
            with session_scope() as session:
                rows = session.execute(
                    select(ParkingBay.neighbourhood_code)
                    .where(ParkingBay.neighbourhood_code.is_not(None))
                    .distinct()
                ).scalars()
                codes.update(c for c in rows if c)
        return sorted(codes)

    def run_all(self, *, page_size: int = 2000) -> IngestResult:
        """Full-city ingest, partitioned to get past the pagination ceiling.

        A plain paged read stops at 200,000 rows because page 101 is refused. Slicing
        by buurtcode keeps every slice comfortably inside the ceiling, and the union of
        the slices is the whole city.
        """
        combined = IngestResult(source=f"{self.meta.name}-All")

        first = self.run(page_size=page_size)
        combined.fetched += first.fetched
        combined.created += first.created
        combined.updated += first.updated
        combined.skipped += first.skipped

        if not first.errors:
            combined.finished_at = utcnow()
            return combined

        codes = self.neighbourhoods()
        log.info("Amsterdam: partitioning the remainder across %d buurtcodes", len(codes))
        for index, code in enumerate(codes, start=1):
            part = self.run(page_size=page_size, neighbourhood=code)
            combined.fetched += part.fetched
            combined.created += part.created
            combined.updated += part.updated
            combined.skipped += part.skipped
            if index % 25 == 0:
                log.info(
                    "Amsterdam: %d/%d buurten, %d bays created so far",
                    index,
                    len(codes),
                    combined.created,
                )

        combined.finished_at = utcnow()
        log.info(combined.summary())
        return combined

    # row handling -------------------------------------------------------
    def _build_bay(
        self, row: dict[str, Any], existing: dict[str, ParkingBay], result: IngestResult
    ) -> tuple[ParkingBay | None, list[dict]]:
        external_id = str(row.get("id") or "")
        geometry = row.get("geometry") or {}
        if not external_id or geometry.get("type") != "Polygon":
            result.skipped += 1
            return None, []

        coords = geometry.get("coordinates") or []
        if not coords or not coords[0] or len(coords[0]) < 3:
            result.skipped += 1
            return None, []

        ring = [[float(p[0]), float(p[1])] for p in coords[0]]
        # A ring outside the RD envelope is a unit or projection error upstream, not a
        # real place. Recording it as a bay would put a parking space in the North Sea.
        if not all(rd_in_range(p[0], p[1]) for p in ring):
            result.skipped += 1
            result.errors.append(f"{external_id}: coordinates outside the RD envelope")
            return None, []

        rect = measure_bay(ring)
        cx, cy = ring_centroid_rd(ring)
        lat, lon = rd_to_wgs84(cx, cy)

        bay = existing.get(external_id)
        created = bay is None
        if bay is None:
            bay = ParkingBay(source_name=self.meta.name, external_id=external_id)
            existing[external_id] = bay

        bay.lat = lat
        bay.lon = lon
        bay.geometry_rd_json = json.dumps(ring, separators=(",", ":"))
        bay.street = row.get("straatnaam") or None
        bay.neighbourhood_code = row.get("buurtcode") or None

        orientation = ORIENTATION_BY_TYPE.get(
            str(row.get("type") or "").strip().lower(), BayOrientation.UNKNOWN
        )
        if orientation is BayOrientation.UNKNOWN:
            orientation = infer_orientation(rect.length_m, rect.width_m)
        bay.orientation = orientation.value
        bay.length_cm = rect.length_cm
        bay.width_cm = rect.width_cm
        bay.max_length_cm = rect.max_length_m * 100.0
        bay.max_width_cm = rect.max_width_m * 100.0
        bay.fill_ratio = rect.fill_ratio
        bay.angle_rad = rect.angle_rad

        try:
            bay.bay_count = max(1, int(float(row.get("aantal") or 1)))
        except (TypeError, ValueError):
            bay.bay_count = 1

        bay.fiscal = str(row.get("soort") or "").strip().upper() == "FISCAAL"
        bay.sign_code = row.get("eType") or None
        regimes = row.get("regimes") or []
        bay.regimes_json = json.dumps(regimes, separators=(",", ":")) if regimes else None
        bay.source_record_id = external_id
        bay.fetched_at = utcnow()

        if created:
            result.created += 1
        else:
            result.updated += 1
        return bay, regimes

    def _sync_restrictions(self, session, batch: list[tuple[ParkingBay, list[dict]]]) -> None:
        """Replace the restriction set for each bay in the batch.

        Restrictions are rewritten wholesale rather than merged: a rule that has been
        removed upstream must disappear here too, and a stale "permit holders only"
        would wrongly hide a bay that is now open to everyone.
        """
        if not batch:
            return
        for bay, _regimes in batch:
            if bay.id is None:
                session.add(bay)
        session.flush()

        bay_ids = [b.id for b, _ in batch if b.id is not None]
        if bay_ids:
            session.execute(
                delete(ParkingRestriction).where(
                    ParkingRestriction.target_kind == "bay",
                    ParkingRestriction.target_id.in_(bay_ids),
                )
            )

        for bay, regimes in batch:
            if bay.id is None:
                continue
            for restriction in self._restrictions_for(bay, regimes):
                session.add(restriction)

    def _restrictions_for(self, bay: ParkingBay, regimes: list[dict]) -> list[ParkingRestriction]:
        """Translate sign codes and regimes into rules a search can filter on."""
        out: list[ParkingRestriction] = []

        # A bay-level sign code with no regime rows still constrains who may park.
        if not regimes and bay.sign_code:
            rule = self._rule_from_sign(bay.sign_code, None)
            if rule:
                out.append(
                    ParkingRestriction(
                        target_kind="bay",
                        target_id=bay.id,
                        source_name=self.meta.name,
                        rule_type=rule["rule_type"],
                        description=rule.get("description"),
                        permit_required=rule.get("permit_required", False),
                        disabled_only=rule.get("disabled_only", False),
                        ev_only=rule.get("ev_only", False),
                        forbids_parking=rule.get("forbids_parking", False),
                    )
                )
            return out

        for regime in regimes:
            sign = regime.get("eType") or bay.sign_code
            rule = self._rule_from_sign(sign, regime)
            if not rule:
                continue
            start = parse_time_to_minutes(regime.get("beginTijd"))
            end = parse_time_to_minutes(regime.get("eindTijd"))
            out.append(
                ParkingRestriction(
                    target_kind="bay",
                    target_id=bay.id,
                    source_name=self.meta.name,
                    rule_type=rule["rule_type"],
                    description=rule.get("description"),
                    weekday_mask=weekday_mask(regime.get("dagen")),
                    start_minute=start if start is not None else 0,
                    end_minute=end if end is not None else 1440,
                    permit_required=rule.get("permit_required", False),
                    disabled_only=rule.get("disabled_only", False),
                    ev_only=rule.get("ev_only", False),
                    forbids_parking=rule.get("forbids_parking", False),
                )
            )
        return out

    @staticmethod
    def _rule_from_sign(sign: str | None, regime: dict | None) -> dict | None:
        code = (sign or "").strip().upper()
        board = str((regime or {}).get("bord") or "").lower()
        description = (regime or {}).get("eTypeDescription") or (regime or {}).get("bord")

        if code in {c.upper() for c in SIGN_DISABLED}:
            return {
                "rule_type": "disabled_only",
                "description": description,
                "disabled_only": True,
            }
        if code in {c.upper() for c in SIGN_PERMIT}:
            return {
                "rule_type": "permit_only",
                "description": description,
                "permit_required": True,
            }
        if code in {c.upper() for c in SIGN_LOADING}:
            return {
                "rule_type": "loading_only",
                "description": description,
                "forbids_parking": True,
            }
        if code in {c.upper() for c in SIGN_NO_PARKING}:
            return {
                "rule_type": "no_parking",
                "description": description,
                "forbids_parking": True,
            }
        if code in {c.upper() for c in SIGN_VEHICLE_CATEGORY}:
            # E8 means "only the vehicle category on the sign". In Amsterdam that is
            # overwhelmingly EV charging, but the sign text is the only thing that
            # actually says so, so we read it rather than assume.
            if any(marker in board for marker in EV_SIGN_MARKERS):
                return {
                    "rule_type": "ev_charging_only",
                    "description": description,
                    "ev_only": True,
                }
            return {
                "rule_type": "vehicle_category_only",
                "description": description,
                "permit_required": True,
            }
        return None

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
