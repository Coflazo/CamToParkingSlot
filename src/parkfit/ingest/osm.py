"""OpenStreetMap adapter, via the Overpass API.

OSM fills two gaps that no Dutch government source covers.

**Points of interest.** PDOK cannot find "Rembrandthuis" because it indexes addresses,
not places people actually name when they say where they are going. OSM knows museums,
parks, stadiums, restaurants and hospitals by name, which is what turns a destination
box into something a human can type into.

**Parking outside the national register.** Plenty of real car parks -- supermarket lots,
hospital decks, small private garages -- never reach RDW. ``amenity=parking`` covers many
of them, including access rules, height limits and surface type.

Two operational notes. Overpass rejects a POST request with 406 but serves the identical
query over GET, and it rejects clients that do not identify themselves, so a User-Agent
is mandatory rather than polite. And OSM is **ODbL**: it carries share-alike obligations
that the other sources here do not, which is exactly why the licence registry records
per-source terms instead of assuming they are all equivalent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from parkfit.ingest.base import BaseAdapter, IngestResult, SourceMeta
from parkfit.storage.models import FacilityKind, ParkingFacility, SourceLicence, utcnow
from parkfit.storage.session import session_scope

log = logging.getLogger(__name__)

#: OSM tags that make a node or way a plausible destination someone would type.
POI_SELECTORS = (
    # `nwr` matches nodes, ways AND relations. Using node/way alone silently drops the
    # largest venues in the country: the Van Gogh Museum, Johan Cruijff ArenA and Ziggo
    # Dome are all multipolygon relations, so the most-searched destinations were
    # exactly the ones missing from the index.
    'nwr["tourism"~"museum|attraction|gallery|zoo|theme_park|aquarium|viewpoint"]',
    'nwr["amenity"~"theatre|cinema|hospital|university|townhall|arts_centre"]',
    'nwr["amenity"~"conference_centre|events_venue|music_venue|exhibition_centre|library"]',
    'nwr["leisure"~"park|stadium|sports_centre|garden|water_park|ice_rink"]',
    'nwr["building"~"stadium|train_station"]',
    'nwr["shop"="mall"]',
    'nwr["railway"="station"]',
    'nwr["historic"~"monument|memorial|castle"]',
)

PARKING_SELECTORS = ('nwr["amenity"="parking"]',)

PARKING_KIND_BY_TAG = {
    "underground": FacilityKind.GARAGE,
    "multi-storey": FacilityKind.GARAGE,
    "garage_boxes": FacilityKind.GARAGE,
    "carports": FacilityKind.SURFACE_LOT,
    "surface": FacilityKind.SURFACE_LOT,
    "rooftop": FacilityKind.SURFACE_LOT,
    "street_side": FacilityKind.ON_STREET_ZONE,
    "lane": FacilityKind.ON_STREET_ZONE,
}


@dataclass(frozen=True)
class OsmPoi:
    """A named place someone might use as a destination."""

    osm_id: str
    name: str
    lat: float
    lon: float
    category: str
    tags: dict[str, str]


def parse_height_to_cm(value: str | None) -> float | None:
    """Parse an OSM ``maxheight`` tag.

    OSM allows ``2.1``, ``2.1 m``, ``210 cm`` and imperial forms in the same field.
    A bare number is metres by OSM convention, which is the opposite of RDW, where a
    bare number is centimetres -- get that backwards and every garage looks either
    impassable or unlimited.
    """
    if not value:
        return None
    text = str(value).strip().lower().replace(",", ".")
    if "'" in text or "ft" in text:
        return None  # imperial is rare in NL and not worth guessing at
    try:
        if text.endswith("cm"):
            return float(text[:-2].strip())
        if text.endswith("m"):
            return float(text[:-1].strip()) * 100.0
        metres = float(text)
    except ValueError:
        return None
    # A bare value above 10 is already centimetres in practice, whatever the convention.
    return metres * 100.0 if metres < 10 else metres


class OsmAdapter(BaseAdapter):
    """Fetches points of interest and parking facilities from Overpass."""

    meta = SourceMeta(
        name="OpenStreetMap",
        url="https://overpass-api.de/api/interpreter",
        licence="ODbL-1.0",
        licence_url="https://opendatacommons.org/licenses/odbl/1-0/",
        attribution="(c) OpenStreetMap contributors",
        commercial_use=True,
        share_alike=True,  # derived databases inherit ODbL obligations
        refresh="continuous",
        contact="https://www.openstreetmap.org/",
        notes="Share-alike applies to derived databases. Attribution is mandatory.",
    )

    def query(self, overpass_ql: str) -> dict[str, Any]:
        """Run an Overpass QL query.

        Sent as GET with the query in the ``data`` parameter: Overpass answers 406 to a
        POST with a raw body, and rejects requests without a User-Agent outright.
        """
        return self.fetch_json(
            self.settings.overpass_url,
            {"data": overpass_ql},
            headers={"Accept": "*/*"},
        )

    @staticmethod
    def _bbox(south: float, west: float, north: float, east: float) -> str:
        return f"{south},{west},{north},{east}"

    @staticmethod
    def _element_point(element: dict[str, Any]) -> tuple[float, float] | None:
        if element.get("lat") is not None and element.get("lon") is not None:
            return float(element["lat"]), float(element["lon"])
        centre = element.get("center")
        if centre:
            return float(centre["lat"]), float(centre["lon"])
        return None

    # -- points of interest -------------------------------------------------
    def fetch_pois(
        self, south: float, west: float, north: float, east: float, *, timeout: int = 90
    ) -> list[OsmPoi]:
        bbox = self._bbox(south, west, north, east)
        selectors = "".join(f"{sel}({bbox});" for sel in POI_SELECTORS)
        ql = f"[out:json][timeout:{timeout}];({selectors});out center tags;"
        payload = self.query(ql)

        pois: list[OsmPoi] = []
        for element in payload.get("elements", []):
            tags = element.get("tags") or {}
            name = tags.get("name") or tags.get("name:nl") or tags.get("name:en")
            if not name:
                continue  # an unnamed place cannot be searched for by name
            point = self._element_point(element)
            if point is None:
                continue
            category = (
                tags.get("tourism")
                or tags.get("amenity")
                or tags.get("leisure")
                or tags.get("historic")
                or tags.get("shop")
                or tags.get("railway")
                or "place"
            )
            pois.append(
                OsmPoi(
                    osm_id=f"{element.get('type')}/{element.get('id')}",
                    name=str(name),
                    lat=point[0],
                    lon=point[1],
                    category=str(category),
                    tags={k: str(v) for k, v in tags.items()},
                )
            )
        return pois

    # -- parking ------------------------------------------------------------
    def run(
        self,
        *,
        south: float = 52.30,
        west: float = 4.72,
        north: float = 52.43,
        east: float = 5.02,
        timeout: int = 120,
        **_: Any,
    ) -> IngestResult:
        """Ingest ``amenity=parking`` facilities within a bounding box.

        The default box is greater Amsterdam. A nationwide Overpass query would be both
        slow and impolite; the OSM project asks that large extracts be taken from a
        Geofabrik download instead, which is what a production refresh should do.
        """
        result = IngestResult(source=self.meta.name)
        bbox = self._bbox(south, west, north, east)
        selectors = "".join(f"{sel}({bbox});" for sel in PARKING_SELECTORS)
        ql = f"[out:json][timeout:{timeout}];({selectors});out center tags;"

        payload = self.query(ql)
        elements = payload.get("elements", [])
        result.fetched = len(elements)

        with session_scope() as session:
            self._register_licence(session)
            existing = {
                f.external_id: f
                for f in session.execute(
                    select(ParkingFacility).where(ParkingFacility.source_name == self.meta.name)
                ).scalars()
            }

            for element in elements:
                tags = element.get("tags") or {}
                point = self._element_point(element)
                if point is None:
                    result.skipped += 1
                    continue

                # A private staff car park is not a result; offering it wastes a trip.
                access = (tags.get("access") or "").lower()
                if access in {"private", "no", "permit"}:
                    result.skipped += 1
                    continue

                external_id = f"{element.get('type')}/{element.get('id')}"
                facility = existing.get(external_id)
                created = facility is None
                if facility is None:
                    facility = ParkingFacility(
                        source_name=self.meta.name, external_id=external_id
                    )
                    session.add(facility)
                    existing[external_id] = facility

                facility.name = str(tags.get("name") or "Parking")[:300]
                facility.kind = PARKING_KIND_BY_TAG.get(
                    (tags.get("parking") or "").lower(), FacilityKind.SURFACE_LOT
                ).value
                facility.lat, facility.lon = point
                facility.geocode_precision = "osm_centroid"
                facility.max_vehicle_height_cm = parse_height_to_cm(tags.get("maxheight"))
                facility.capacity = _int_or_none(tags.get("capacity"))
                facility.charging_capacity = _int_or_none(tags.get("capacity:charging"))
                facility.disabled_capacity = _int_or_none(tags.get("capacity:disabled"))
                facility.street = tags.get("addr:street")
                facility.house_number = tags.get("addr:housenumber")
                facility.postcode = tags.get("addr:postcode")
                facility.city = tags.get("addr:city")
                facility.tariff_note = tags.get("fee:conditional") or tags.get("fee")
                facility.geometry_geojson = json.dumps(
                    {k: v for k, v in tags.items() if k.startswith(("parking", "surface", "fee"))},
                    separators=(",", ":"),
                )
                facility.active = True
                facility.fetched_at = utcnow()

                result.created += int(created)
                result.updated += int(not created)

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


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def ingest_pois(
    adapter: OsmAdapter,
    *,
    south: float = 52.30,
    west: float = 4.72,
    north: float = 52.43,
    east: float = 5.02,
) -> IngestResult:
    """Index named places so users can type a destination instead of an address.

    See :mod:`parkfit.services.geocoding` for why this exists: the official Dutch
    geocoder returns zero results for "Rembrandthuis".
    """
    from parkfit.services.geocoding import CATEGORY_IMPORTANCE, normalise
    from parkfit.storage.models import PointOfInterest

    result = IngestResult(source=f"{adapter.meta.name}-POI")
    pois = adapter.fetch_pois(south, west, north, east)
    result.fetched = len(pois)

    with session_scope() as session:
        existing = {
            p.external_id: p
            for p in session.execute(
                select(PointOfInterest).where(
                    PointOfInterest.source_name == adapter.meta.name
                )
            ).scalars()
        }
        for poi in pois:
            row = existing.get(poi.osm_id)
            created = row is None
            if row is None:
                row = PointOfInterest(
                    source_name=adapter.meta.name, external_id=poi.osm_id
                )
                session.add(row)
                existing[poi.osm_id] = row

            row.name = poi.name[:300]
            row.normalised_name = normalise(poi.name)[:300]
            row.category = poi.category[:60]
            row.lat = poi.lat
            row.lon = poi.lon
            row.city = poi.tags.get("addr:city")
            row.street = poi.tags.get("addr:street")
            row.house_number = poi.tags.get("addr:housenumber")
            row.postcode = poi.tags.get("addr:postcode")
            row.importance = CATEGORY_IMPORTANCE.get(poi.category, 0.4)

            aliases = [
                poi.tags.get(key)
                for key in ("name:en", "name:nl", "alt_name", "short_name", "official_name")
            ]
            aliases = [a for a in aliases if a and a != poi.name]
            row.aliases_json = json.dumps(aliases, separators=(",", ":")) if aliases else None
            row.fetched_at = utcnow()

            result.created += int(created)
            result.updated += int(not created)

    result.finished_at = utcnow()
    log.info(result.summary())
    return result


ROAD_SELECTOR = (
    'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|'
    'living_street|service|road|pedestrian|footway|path|steps|cycleway)$"]'
)


def ingest_roads(
    adapter: OsmAdapter,
    *,
    south: float = 52.33,
    west: float = 4.82,
    north: float = 52.41,
    east: float = 4.97,
    timeout: int = 180,
) -> IngestResult:
    """Build and cache the routable road graph for a bounding box.

    ``out body`` plus a recursed node pass is required: ways carry node references, not
    coordinates, so without ``>;`` every way would reference nodes we do not have and
    the graph would come out empty.

    The default box is central Amsterdam. A nationwide graph should come from a
    Geofabrik extract rather than Overpass, which the OSM project asks not be used for
    bulk downloads.
    """
    from parkfit.routing.graph import GraphBuilder, NativeGraphProvider

    result = IngestResult(source=f"{adapter.meta.name}-Roads")
    bbox = f"{south},{west},{north},{east}"
    ql = f"[out:json][timeout:{timeout}];({ROAD_SELECTOR}({bbox}););out body;>;out skel qt;"

    payload = adapter.query(ql)
    elements = payload.get("elements", [])
    result.fetched = len(elements)

    graph = GraphBuilder(adapter.settings).build(elements)
    if not graph.nodes:
        result.errors.append("graph came out empty; check the bounding box")
        result.finished_at = utcnow()
        return result

    path = NativeGraphProvider(adapter.settings).save_graph(graph)
    result.created = len(graph.nodes)
    result.finished_at = utcnow()
    log.info("road graph cached at %s (%d nodes)", path, len(graph.nodes))
    return result
