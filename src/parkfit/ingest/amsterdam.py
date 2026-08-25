"""City of Amsterdam ``parkeervakken`` adapter.

This is the most valuable dataset in the whole product. Amsterdam publishes *every
individual parking bay* as an exact polygon in RD New, tagged with its layout
(``Langs`` parallel / ``Haaks`` perpendicular), whether it is metered, its traffic-sign
code, and its time regimes.

That changes what computer vision has to do. Without it, a camera would have to answer
"where is a legal space and how long is it" -- the research-grade problem. With it,
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

    def _pages(self, *, page_size: int, limit: int | None, params: dict[str, Any] | None):
        """Walk the DSO-API pages via their ``next`` link.

        The API refuses ``Accept: application/json`` outright with a 406 and wants
        ``_format=json`` in the query string instead, which is unusual enough to be
        worth pinning down here rather than rediscovering at 3 a.m.
        """
        url = self.meta.url
        query: dict[str, Any] | None = {"_pageSize": page_size, "_format": "json"}
        if params:
            query.update(params)

        fetched = 0
        while url:
            body = self.fetch_json(url, query, headers={"Accept": "*/*"})
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

    def run(
        self,
        *,
        limit: int | None = None,
        page_size: int | None = None,
        street: str | None = None,
        neighbourhood: str | None = None,
    ) -> IngestResult:
        result = IngestResult(source=self.meta.name)
        params: dict[str, Any] = {}
        if street:
            params["straatnaam"] = street
        if neighbourhood:
            params["buurtcode"] = neighbourhood

        with session_scope() as session:
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
                    batch.clear()
                    log.info("Amsterdam: %d bays processed", result.fetched)

            session.flush()
            self._sync_restrictions(session, batch)

        result.finished_at = utcnow()
        log.info(result.summary())
        return result

    # -- row handling -------------------------------------------------------
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
        bay.street = (row.get("straatnaam") or None)
        bay.neighbourhood_code = (row.get("buurtcode") or None)

        orientation = ORIENTATION_BY_TYPE.get(
            str(row.get("type") or "").strip().lower(), BayOrientation.UNKNOWN
        )
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
        bay.sign_code = (row.get("eType") or None)
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
