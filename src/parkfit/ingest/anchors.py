"""The map features that road law measures distances from.

Every statute in ``cpp/core/include/parkfit/legal/`` says the same kind of thing: not
within five metres of *this*, not within fifteen metres of *that*. The things it points
at are ordinary map features, and that is the whole reason the legality engine
generalises. Explicit parking restrictions are mapped densely in almost nowhere (Berlin
has 2,762 ways carrying the OSM parking schema, Paris 88, Istanbul 50, Amsterdam 33), but
bus stops, crossings and hydrants are everywhere, in every country, because they are what
a map is for.

So this module does not ingest parking rules. It ingests the anchors, and the rulebook
supplies the law.

**Junctions come from the road graph, not from Overpass.** OSM has no "junction" feature;
a junction is an emergent property of the topology, a node where three or more road
segments meet. The cached road graph already knows that exactly, for free, and asking
Overpass for an approximation would be both slower and worse.

**Germany's cycle-path junctions get two anchors.** StVO 12(3) extends the junction
setback from 5 m to 8 m wherever a structurally separate cycle path runs on the right, so
a junction that has a cycleway nearby is emitted as both ``JUNCTION`` and
``JUNCTION_WITH_CYCLE_PATH``. Books that do not have the second rule never match it.

Coverage is honest rather than complete. What is sourced here:

===========================  ==========================================================
anchor                        source
===========================  ==========================================================
JUNCTION                      road graph nodes with three or more distinct neighbours
JUNCTION_WITH_CYCLE_PATH      those junctions with a cycleway within ``CYCLE_PATH_M``
PEDESTRIAN_CROSSING           ``highway=crossing``
BUS_STOP_SIGN                 ``highway=bus_stop``, ``public_transport=platform`` + bus
TRAM_STOP                     ``railway=tram_stop``
FIRE_HYDRANT                  ``emergency=fire_hydrant``
LEVEL_CROSSING                ``railway=level_crossing``, ``railway=crossing``
PUBLIC_ENTRANCE               ``kerb=lowered`` (StVO 12(3) Nr. 5 dropped kerbs)
DRIVEWAY                      endpoints of ``highway=service`` + ``service=driveway``
===========================  ==========================================================

And what is not, recorded rather than quietly skipped. ``EMERGENCY_ACCESS`` (a German
Feuerwehrzufahrt or a Turkish emergency exit) has no reliable OSM tag; ``BRIDGE``,
``TUNNEL`` and ``UNDERPASS`` are way attributes rather than points and need geometry this
module does not yet fetch, which matters most for Turkey because article 61(k) has a ten
metre rule for them; ``DISABLED_BAY``, ``LOADING_BAY`` and the yellow lines come from the
bay records themselves during the Amsterdam ingest, not from here. A missing anchor kind
lowers confidence rather than granting permission, because the legality engine treats an
absent anchor as unproven, never as clear.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parkfit.config import Settings, get_settings
from parkfit.ingest.base import IngestResult
from parkfit.storage.models import utcnow

log = logging.getLogger(__name__)

#: Overpass selectors that yield one anchor per element, paired with the anchor name the
#: C++ enum uses. Kept as strings so this table reads like the statute it serves.
POINT_SELECTORS: tuple[tuple[str, str], ...] = (
    ('node["highway"="crossing"]', "PEDESTRIAN_CROSSING"),
    ('node["highway"="bus_stop"]', "BUS_STOP_SIGN"),
    ('node["public_transport"="platform"]["bus"="yes"]', "BUS_STOP_SIGN"),
    ('node["railway"="tram_stop"]', "TRAM_STOP"),
    ('node["emergency"="fire_hydrant"]', "FIRE_HYDRANT"),
    ('node["railway"="level_crossing"]', "LEVEL_CROSSING"),
    ('node["railway"="crossing"]', "LEVEL_CROSSING"),
    # A dropped kerb is a wheelchair and pram crossing point. Germany prohibits parking
    # in front of one outright, and it is the rule visitors break most often.
    ('node["kerb"="lowered"]', "PUBLIC_ENTRANCE"),
)

#: Driveways are ways, and what matters is where they meet the road, so their endpoints
#: are taken rather than their geometry.
DRIVEWAY_SELECTOR = 'way["highway"="service"]["service"="driveway"]'

#: A junction counts as having a separate cycle path when a cycleway node sits within
#: this distance. StVO says "baulich angelegt" (structurally laid out) on the right in
#: the direction of travel, which OSM does not record precisely enough to reproduce, so
#: this is an approximation and it is deliberately generous: over-applying the 8 m rule
#: refuses a legal space, while under-applying it offers an illegal one, and of the two
#: errors only the second costs the driver money.
CYCLE_PATH_M = 15.0

#: A node needs at least this many distinct neighbours to be a junction. Two means a
#: bend or a way split, which is not a junction and would carpet a city in false anchors.
JUNCTION_MIN_DEGREE = 3


@dataclass
class AnchorSet:
    """Anchors for one area, ready to be handed to the C++ index."""

    #: (anchor_name, lat, lon) triples, in the shape ``AnchorIndex.add_many`` wants.
    anchors: list[tuple[str, float, float]] = field(default_factory=list)
    #: Which country's rulebook these were collected for.
    country: str = "NL"
    bbox: tuple[float, float, float, float] | None = None
    generated_at: str = ""
    #: The anchor kinds this ingest **asked for**, which is not the same as the kinds it
    #: found. "We looked for hydrants and this district has none" and "nobody ever looked
    #: for hydrants here" produce identical anchor lists and opposite conclusions, and
    #: only this field can tell them apart. A rule whose anchor was never queried cannot
    #: be cleared, so the legality engine downgrades to Unknown rather than to Legal.
    queried_kinds: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.anchors)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind, _lat, _lon in self.anchors:
            out[kind] = out.get(kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def anchor_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / "osm" / "anchors"


def region_key(country: str, bbox: tuple[float, float, float, float]) -> str:
    """A stable filename for one ingested area.

    Anchors are per-region, not global. One cache file would mean ingesting Istanbul
    erased Amsterdam, and a product covering four countries cannot hold one city at a
    time. The key is the country plus the rounded box, so re-ingesting the same area
    replaces it and a neighbouring area sits beside it.
    """
    south, west, north, east = bbox
    return f"{country.upper()}_{south:.3f}_{west:.3f}_{north:.3f}_{east:.3f}".replace("-", "m")


def cache_path(
    settings: Settings | None = None,
    *,
    country: str = "NL",
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Where one region's anchors live.

    With no bbox this returns the legacy single-file path, which is still read so an
    existing cache keeps working rather than silently becoming invisible.
    """
    if bbox is None:
        return (settings or get_settings()).data_dir / "osm" / "legal_anchors.json.gz"
    return anchor_dir(settings) / f"{region_key(country, bbox)}.json.gz"


def save(anchor_set: AnchorSet, settings: Settings | None = None) -> Path:
    path = cache_path(settings, country=anchor_set.country, bbox=anchor_set.bbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "country": anchor_set.country,
        "bbox": list(anchor_set.bbox) if anchor_set.bbox else None,
        "generated_at": anchor_set.generated_at,
        "queried_kinds": list(anchor_set.queried_kinds),
        # Coordinates are rounded to about a centimetre. Anything finer is noise against
        # a five metre rule, and it halves the file.
        "anchors": [[k, round(lat, 7), round(lon, 7)] for k, lat, lon in anchor_set.anchors],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def _read(path: Path) -> AnchorSet | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        log.warning("unreadable anchor cache %s: %s", path.name, exc)
        return None
    return AnchorSet(
        anchors=[(row[0], float(row[1]), float(row[2])) for row in payload.get("anchors", [])],
        country=payload.get("country", "NL"),
        bbox=tuple(payload["bbox"]) if payload.get("bbox") else None,
        generated_at=payload.get("generated_at", ""),
        queried_kinds=tuple(payload.get("queried_kinds", ())),
    )


def load_all(settings: Settings | None = None) -> list[AnchorSet]:
    """Every cached region, newest last. Empty when nothing has been ingested."""
    out: list[AnchorSet] = []
    legacy = cache_path(settings)
    if legacy.exists():
        found = _read(legacy)
        if found is not None:
            out.append(found)
    directory = anchor_dir(settings)
    if directory.exists():
        for path in sorted(directory.glob("*.json.gz")):
            found = _read(path)
            if found is not None:
                out.append(found)
    return out


def load(settings: Settings | None = None) -> AnchorSet | None:
    """The first cached region, for callers that only want one.

    Kept because a single region is the common case in tests and scripts. Anything that
    has to answer for more than one area uses :func:`load_all`.
    """
    found = load_all(settings)
    return found[0] if found else None


def junctions_from_graph(
    graph: Any, bbox: tuple[float, float, float, float] | None = None
) -> list[tuple[str, float, float]]:
    """Road-graph nodes where three or more distinct roads meet.

    Distinct is the operative word. The adjacency holds one entry per direction, so a
    plain two-way street gives every mid-block node a degree of two; counting raw edges
    would call every node in the city a junction. Counting the set of neighbours instead
    gives two for a through node and three or more for a real fork.

    **The car graph only.** The first version unioned the foot graph too, on the reasoning
    that a junction is a junction. It is not, legally. Every statute here says roads: RVV
    says "kruispunt", StVO "Kreuzungen und Einmuendungen", KTK "kavsaklar". A footpath
    meeting a street is not one of those, and the foot graph has 424,600 edges against the
    car graph's 114,646, so including it turned every path connection into a five-metre
    no-parking zone. On central Amsterdam that inflated the junction count roughly
    threefold, and each false junction refuses a legal space.

    Service roads and driveways are still in the car graph and so still count here, which
    over-restricts slightly. That is the safe direction of the two, and a driveway mouth
    already has its own anchor and its own article.
    """
    from parkfit.routing.provider import Profile

    out: list[tuple[str, float, float]] = []
    edges = graph.edges_for(Profile.CAR)
    for node_id, rows in edges.items():
        if len({neighbour for neighbour, _length, _seconds in rows}) < JUNCTION_MIN_DEGREE:
            continue
        position = graph.nodes.get(node_id)
        if position is None:
            continue
        # The cached road graph usually covers a wider area than the anchors being built,
        # and an anchor outside the requested box is dead weight in the index.
        if bbox is not None:
            south, west, north, east = bbox
            if not (south <= position[0] <= north and west <= position[1] <= east):
                continue
        out.append(("JUNCTION", position[0], position[1]))
    return out


def mark_cycle_path_junctions(
    junctions: list[tuple[str, float, float]],
    cycleway_points: list[tuple[float, float]],
    *,
    radius_m: float = CYCLE_PATH_M,
) -> list[tuple[str, float, float]]:
    """Emit a second anchor for junctions that have a cycle path beside them.

    Germany needs this and nobody else does, which is exactly why it is a second anchor
    rather than a flag: a country whose statute does not mention cycle paths simply never
    matches ``JUNCTION_WITH_CYCLE_PATH`` and does not have to know it exists.
    """
    if not cycleway_points or not junctions:
        return []

    from parkfit.native import native

    if native is None:
        # Without the spatial index this would be a few hundred thousand haversines.
        # Skipping is honest: the 8 m rule then never fires, the 5 m one still does, and
        # a German result near a cycle path is under-restricted rather than wrong about
        # anything it does claim.
        log.warning("parkfit_native is not built, so cycle-path junctions are not marked")
        return []

    grid = native.SpatialGrid(50.0)
    grid.insert_many([(lat, lon, i) for i, (lat, lon) in enumerate(cycleway_points)])
    grid.build()

    out: list[tuple[str, float, float]] = []
    for _kind, lat, lon in junctions:
        if grid.query_radius(lat, lon, radius_m, 1):
            out.append(("JUNCTION_WITH_CYCLE_PATH", lat, lon))
    return out


def _point_of(element: dict[str, Any]) -> tuple[float, float] | None:
    if element.get("lat") is not None and element.get("lon") is not None:
        return float(element["lat"]), float(element["lon"])
    centre = element.get("center")
    if centre:
        return float(centre["lat"]), float(centre["lon"])
    return None


def ingest_anchors(
    adapter: Any,
    *,
    south: float = 52.33,
    west: float = 4.82,
    north: float = 52.41,
    east: float = 4.97,
    country: str = "NL",
    timeout: int = 180,
) -> IngestResult:
    """Collect the legal anchors for a bounding box and cache them.

    The default box is central Amsterdam, matching ``ingest_roads``. The two belong
    together: junctions are derived from the road graph, so anchors for an area are only
    as complete as the road graph for that area.
    """
    result = IngestResult(source="OpenStreetMap-LegalAnchors")
    bbox = f"{south},{west},{north},{east}"

    selectors = ";".join(f"{selector}({bbox})" for selector, _kind in POINT_SELECTORS)
    ql = f"[out:json][timeout:{timeout}];({selectors};);out body;"
    payload = adapter.query(ql)
    elements = payload.get("elements", [])
    result.fetched = len(elements)

    # One selector per anchor kind would be several round trips, so the union comes back
    # in one response and the tags decide which kind each element is.
    anchors: list[tuple[str, float, float]] = []
    for element in elements:
        point = _point_of(element)
        if point is None:
            continue
        tags = element.get("tags") or {}
        kind = _classify(tags)
        if kind is None:
            continue
        anchors.append((kind, point[0], point[1]))

    # Driveways: ways, reduced to the endpoints where they meet the road.
    driveway_ql = (
        f"[out:json][timeout:{timeout}];({DRIVEWAY_SELECTOR}({bbox}););out body;>;out skel qt;"
    )
    try:
        driveways = adapter.query(driveway_ql)
        anchors.extend(_driveway_endpoints(driveways))
    except Exception as exc:  # a missing driveway layer is a gap, not a failure
        log.warning("driveway anchors unavailable: %s", exc)
        result.errors.append(f"driveways: {exc}")

    # Junctions from the cached road graph, plus the German cycle-path variant.
    junctions = _junctions_for(adapter, (south, west, north, east), timeout)
    anchors.extend(junctions)
    cycleways = _cycleway_points(adapter, bbox, timeout, result)
    anchors.extend(mark_cycle_path_junctions(junctions, cycleways))

    # What this run *asked for*, so a later reader can tell an absent hydrant from an
    # un-queried one. Junctions only count as queried when a road graph actually covered
    # the box; without one the whole kind is missing rather than merely empty.
    queried = {kind for _selector, kind in POINT_SELECTORS} | {"DRIVEWAY"}
    if junctions:
        queried |= {"JUNCTION"}
    if cycleways:
        queried |= {"JUNCTION_WITH_CYCLE_PATH"}

    anchor_set = AnchorSet(
        anchors=anchors,
        country=country,
        bbox=(south, west, north, east),
        generated_at=utcnow().isoformat(),
        queried_kinds=tuple(sorted(queried)),
    )
    path = save(anchor_set, adapter.settings)
    result.created = len(anchors)
    result.finished_at = utcnow()
    log.info("legal anchors: %d written to %s", len(anchors), path)
    log.info("anchor mix: %s", anchor_set.counts())
    return result


def _classify(tags: dict[str, Any]) -> str | None:
    """Which anchor kind an OSM element is, or None if it is not one.

    Order matters where an element could match twice. A tram stop that is also tagged as
    a bus platform is a tram stop first, because the tram rule is the stricter of the two
    in every book here.
    """
    if tags.get("railway") == "tram_stop":
        return "TRAM_STOP"
    if tags.get("railway") in {"level_crossing", "crossing"}:
        return "LEVEL_CROSSING"
    if tags.get("emergency") == "fire_hydrant":
        return "FIRE_HYDRANT"
    if tags.get("highway") == "bus_stop":
        return "BUS_STOP_SIGN"
    if tags.get("public_transport") == "platform" and tags.get("bus") == "yes":
        return "BUS_STOP_SIGN"
    if tags.get("highway") == "crossing":
        return "PEDESTRIAN_CROSSING"
    if tags.get("kerb") == "lowered":
        return "PUBLIC_ENTRANCE"
    return None


def _driveway_endpoints(payload: dict[str, Any]) -> list[tuple[str, float, float]]:
    """First and last node of each driveway way.

    A driveway's midpoint is inside somebody's garden and no statute cares about it. The
    ends are where it meets the carriageway, which is the thing you must not park across.
    """
    coords: dict[int, tuple[float, float]] = {}
    for element in payload.get("elements", []):
        if element.get("type") == "node":
            coords[int(element["id"])] = (float(element["lat"]), float(element["lon"]))

    out: list[tuple[str, float, float]] = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        nodes = [int(n) for n in element.get("nodes", [])]
        for node_id in {nodes[0], nodes[-1]} if nodes else set():
            position = coords.get(node_id)
            if position is not None:
                out.append(("DRIVEWAY", position[0], position[1]))
    return out


def _junctions_for(
    adapter: Any, bbox: tuple[float, float, float, float], timeout: int
) -> list[tuple[str, float, float]]:
    """Junctions from the cached road graph, falling back to fetching one."""
    from parkfit.routing.graph import Graph, NativeGraphProvider

    south, west, north, east = bbox
    centre_lat, centre_lon = (south + north) / 2.0, (west + east) / 2.0

    provider = NativeGraphProvider(adapter.settings)
    covering = [r for r in provider.regions() if r.contains(centre_lat, centre_lon)]
    if not covering:
        # Junctions are the one anchor kind that comes from the road graph rather than
        # from Overpass, so without a graph for this area the whole kind is absent. That
        # is recorded in queried_kinds, which is what stops a later evaluation reporting
        # a space near an uncollected junction as legal.
        log.info(
            "no cached road graph covers (%.4f, %.4f), so junction anchors are skipped; "
            "run: pf ingest roads for this area",
            centre_lat,
            centre_lon,
        )
        return []

    with gzip.open(covering[0].path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    graph = Graph(
        nodes={int(k): (v[0], v[1]) for k, v in payload["nodes"].items()},
        car_edges={
            int(k): [(int(n), float(d), float(s)) for n, d, s in v]
            for k, v in payload["car_edges"].items()
        },
        foot_edges={
            int(k): [(int(n), float(d), float(s)) for n, d, s in v]
            for k, v in payload["foot_edges"].items()
        },
    )
    return junctions_from_graph(graph, bbox)


def _cycleway_points(
    adapter: Any, bbox: str, timeout: int, result: IngestResult
) -> list[tuple[float, float]]:
    ql = f'[out:json][timeout:{timeout}];(way["highway"="cycleway"]({bbox}););out body;>;out skel qt;'
    try:
        payload = adapter.query(ql)
    except Exception as exc:
        log.warning("cycleway layer unavailable: %s", exc)
        result.errors.append(f"cycleways: {exc}")
        return []
    return [
        (float(e["lat"]), float(e["lon"]))
        for e in payload.get("elements", [])
        if e.get("type") == "node" and e.get("lat") is not None
    ]
