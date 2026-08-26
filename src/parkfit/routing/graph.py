"""Road-network routing over a cached OpenStreetMap graph.

This exists so the product does not need OSRM. OSRM is excellent, but standing it up
means Docker, a 1.5 GB extract and a twenty-minute preprocessing step, and this
machine has no running Docker daemon. A parking search needs distances of a few
kilometres over a few hundred candidates, which a plain bidirectional A* handles in
milliseconds without any of that.

The graph is built once from an Overpass extract and cached on disk. Two profiles:

* **car** honours ``oneway``, skips footways and cycleways, and uses per-road-class
  speeds rather than a single average: the difference between a canal-side
  ``residential`` at 15 km/h and an ``a-road`` at 60 km/h dominates any Amsterdam ETA.
* **foot** ignores ``oneway`` entirely, since pedestrians do not obey it, and includes
  footpaths, bridges and pedestrian squares that a car cannot use.

Both directions matter: the drive leg and the walk leg of the same search take genuinely
different paths through the same city.
"""

from __future__ import annotations

import gzip
import heapq
import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from parkfit.config import Settings, get_settings
from parkfit.geo.rd import haversine_m
from parkfit.routing.provider import Profile, RouteResult, RoutingProvider, RoutingUnavailableError

log = logging.getLogger(__name__)

#: Free-flow speeds in km/h by OSM highway class. Deliberately below the legal limit:
#: these are journey speeds including junctions, lights and the general friction of a
#: Dutch city centre, not signposted maxima.
CAR_SPEED_KMH = {
    "motorway": 100.0,
    "motorway_link": 60.0,
    "trunk": 70.0,
    "trunk_link": 45.0,
    "primary": 45.0,
    "primary_link": 35.0,
    "secondary": 38.0,
    "secondary_link": 30.0,
    "tertiary": 32.0,
    "tertiary_link": 25.0,
    "unclassified": 25.0,
    "residential": 20.0,
    "living_street": 12.0,
    "service": 12.0,
    "road": 22.0,
}

FOOT_WAYS = {
    "footway",
    "path",
    "pedestrian",
    "steps",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "track",
    "tertiary",
    "secondary",
    "primary",
    "road",
    "cycleway",
    "corridor",
    "platform",
}

CAR_WAYS = set(CAR_SPEED_KMH)

WALK_SPEED_KMH = 4.8
#: Stairs are passable on foot but slow, and a router that ignores that will happily
#: send someone with luggage down three flights to save forty metres.
STEPS_PENALTY = 3.5


@dataclass
class Graph:
    """An adjacency-list road graph in memory."""

    #: node id -> (lat, lon)
    nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
    #: node id -> list of (neighbour, length_m, seconds)
    car_edges: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)
    foot_edges: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def edges_for(self, profile: Profile) -> dict[int, list[tuple[int, float, float]]]:
        return self.car_edges if profile is Profile.CAR else self.foot_edges


class GraphBuilder:
    """Turns an Overpass ``way`` extract into a routable graph."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def build(self, elements: list[dict]) -> Graph:
        graph = Graph()
        coords: dict[int, tuple[float, float]] = {}
        for element in elements:
            if element.get("type") == "node":
                coords[int(element["id"])] = (float(element["lat"]), float(element["lon"]))

        for element in elements:
            if element.get("type") != "way":
                continue
            tags = element.get("tags") or {}
            highway = tags.get("highway")
            if not highway:
                continue
            node_ids = [int(n) for n in element.get("nodes", [])]
            if len(node_ids) < 2:
                continue

            car_ok = highway in CAR_WAYS and tags.get("access") not in {"private", "no"}
            if tags.get("motor_vehicle") in {"no", "private"}:
                car_ok = False
            foot_ok = highway in FOOT_WAYS and tags.get("foot") not in {"no", "private"}
            if highway in {"motorway", "motorway_link", "trunk", "trunk_link"}:
                foot_ok = False
            if not car_ok and not foot_ok:
                continue

            oneway = str(tags.get("oneway", "")).lower()
            forward_only = oneway in {"yes", "true", "1"}
            reverse_only = oneway == "-1"

            speed = self._car_speed(highway, tags)
            walk_penalty = STEPS_PENALTY if highway == "steps" else 1.0

            for a, b in itertools.pairwise(node_ids):
                pa, pb = coords.get(a), coords.get(b)
                if pa is None or pb is None:
                    continue
                length = haversine_m(pa[0], pa[1], pb[0], pb[1])
                if length <= 0.0:
                    continue
                graph.nodes.setdefault(a, pa)
                graph.nodes.setdefault(b, pb)

                if car_ok:
                    seconds = length / (speed / 3.6)
                    if not reverse_only:
                        graph.car_edges.setdefault(a, []).append((b, length, seconds))
                    if not forward_only:
                        graph.car_edges.setdefault(b, []).append((a, length, seconds))
                if foot_ok:
                    # One-way restrictions apply to vehicles, not to people on foot.
                    seconds = length / (WALK_SPEED_KMH / 3.6) * walk_penalty
                    graph.foot_edges.setdefault(a, []).append((b, length, seconds))
                    graph.foot_edges.setdefault(b, []).append((a, length, seconds))

        log.info(
            "graph built: %d nodes, %d car edges, %d foot edges",
            len(graph.nodes),
            sum(len(v) for v in graph.car_edges.values()),
            sum(len(v) for v in graph.foot_edges.values()),
        )
        return graph

    @staticmethod
    def _car_speed(highway: str, tags: dict) -> float:
        base = CAR_SPEED_KMH.get(highway, 22.0)
        raw = tags.get("maxspeed")
        if raw:
            try:
                signed = float(str(raw).split()[0])
            except (ValueError, IndexError):
                return base
            # Signposted limits are an upper bound, never a promise. Real journey speed
            # in a city sits well below them, so take the lower of the two.
            return min(base, signed * 0.8) if signed > 0 else base
        return base


class RoadRouter:
    """Heuristic A* over the road graph.

    Two things here are not incidental detail; without either, routing is broken.

    **Connectivity-aware snapping.** A bounding-box extract of OSM is full of islands:
    parking aisles, service yards, and in Amsterdam the whole of Noord, which really is
    unreachable by car without a ferry. The car graph for central Amsterdam has 323
    disconnected components. Snapping each endpoint to its geometrically nearest node
    lands you on an eleven-node service island, and A* then correctly reports no path.
    So the destination is snapped first, and the origin is snapped afterwards *within
    the destination component*.

    **Indexed snapping.** A linear scan over 188k nodes costs about 30 ms. A search
    scoring 400 candidates over two legs would spend 24 seconds snapping coordinates
    alone, so the C++ spatial grid does it instead.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._components: dict[Profile, dict[int, int]] = {}
        self._component_sizes: dict[Profile, dict[int, int]] = {}
        self._grids: dict[Profile, object] = {}
        self._grid_nodes: dict[Profile, list[int]] = {}

    # connectivity -------------------------------------------------------
    def components(self, profile: Profile) -> dict[int, int]:
        """Map every routable node to a **strongly** connected component id.

        Strong connectivity is the requirement, not an optimisation. The car graph is
        directed because one-way streets are directed, and in a directed graph
        "reachable from a seed" is not the same relation as "mutually reachable".
        Labelling by forward reachability puts two nodes in the same group when the
        seed can reach both, even though neither can reach the other, and Amsterdam,
        whose canal ring is a dense web of one-ways, is precisely the topology where
        that goes wrong. Snapping both endpoints into such a group produced a pair with
        no path between them, and A* then correctly reported failure.

        Tarjan, run iteratively: 188k nodes would overflow the recursion limit.
        """
        cached = self._components.get(profile)
        if cached is not None:
            return cached

        edges = self.graph.edges_for(profile)
        index: dict[int, int] = {}
        lowlink: dict[int, int] = {}
        on_stack: dict[int, bool] = {}
        stack: list[int] = []
        component_of: dict[int, int] = {}
        sizes: dict[int, int] = {}
        counter = 0
        next_component = 0

        for seed in edges:
            if seed in index:
                continue
            work: list[tuple[int, int]] = [(seed, 0)]
            while work:
                node, child_index = work[-1]
                if child_index == 0:
                    index[node] = lowlink[node] = counter
                    counter += 1
                    stack.append(node)
                    on_stack[node] = True

                descended = False
                neighbours = edges.get(node, ())
                for i in range(child_index, len(neighbours)):
                    neighbour = neighbours[i][0]
                    if neighbour not in index:
                        work[-1] = (node, i + 1)
                        work.append((neighbour, 0))
                        descended = True
                        break
                    if on_stack.get(neighbour):
                        lowlink[node] = min(lowlink[node], index[neighbour])
                if descended:
                    continue

                if lowlink[node] == index[node]:
                    size = 0
                    while True:
                        member = stack.pop()
                        on_stack[member] = False
                        component_of[member] = next_component
                        size += 1
                        if member == node:
                            break
                    sizes[next_component] = size
                    next_component += 1

                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

        self._components[profile] = component_of
        self._component_sizes[profile] = sizes
        log.debug(
            "%s graph: %d strongly connected components, largest %d",
            profile.value,
            len(sizes),
            max(sizes.values()) if sizes else 0,
        )
        return component_of

    def largest_component(self, profile: Profile) -> int | None:
        self.components(profile)
        sizes = self._component_sizes.get(profile) or {}
        if not sizes:
            return None
        return max(sizes, key=lambda cid: sizes[cid])

    # snapping -----------------------------------------------------------
    def _grid(self, profile: Profile):
        """A spatial index over the routable nodes for this profile."""
        if profile in self._grids:
            return self._grids[profile]

        from parkfit.native import native

        edges = self.graph.edges_for(profile)
        node_ids = [n for n in self.graph.nodes if n in edges]
        self._grid_nodes[profile] = node_ids

        if native is None:
            self._grids[profile] = None
            return None

        grid = native.SpatialGrid(200.0)
        grid.insert_many(
            [(self.graph.nodes[n][0], self.graph.nodes[n][1], i) for i, n in enumerate(node_ids)]
        )
        grid.build()
        self._grids[profile] = grid
        return grid

    def nearest_node(
        self, lat: float, lon: float, profile: Profile, *, component: int | None = None
    ) -> int | None:
        """Closest routable node, optionally restricted to one connected component."""
        component_of = self.components(profile) if component is not None else {}
        grid = self._grid(profile)

        if grid is not None:
            node_ids = self._grid_nodes[profile]
            radius = 150.0
            while radius <= 12000.0:
                # No result cap. Hits come back sorted by distance, so a capped query
                # returns the same nearest N however far the radius grows, widening
                # the search would never surface a node in the component we need.
                for hit in grid.query_radius(lat, lon, radius, 0):
                    node_id = node_ids[hit.payload]
                    if component is None or component_of.get(node_id) == component:
                        return node_id
                radius *= 2.5
            return None

        # Pure-Python fallback for a checkout that has not been compiled.
        edges = self.graph.edges_for(profile)
        best_id: int | None = None
        best_d = float("inf")
        for node_id, (nlat, nlon) in self.graph.nodes.items():
            if node_id not in edges:
                continue
            if component is not None and component_of.get(node_id) != component:
                continue
            d = (nlat - lat) ** 2 + ((nlon - lon) * 0.61) ** 2
            if d < best_d:
                best_d = d
                best_id = node_id
        return best_id

    # one-to-many --------------------------------------------------------
    def costs_from(
        self, lat: float, lon: float, profile: Profile, *, max_seconds: float = 1500.0
    ) -> tuple[dict[int, tuple[float, float]], int | None]:
        """Dijkstra from one point, labelling every node within a time budget.

        This is the shape a parking search actually has. It is not N independent
        point-to-point queries; it is two one-to-many queries: drive time from one
        origin to many entrances, and walk time from many exits to one destination.
        Running A* per candidate re-explores the same city several hundred times, at
        roughly 30 ms each.

        One capped Dijkstra labels every reachable node in a single sweep. The cap
        matters as much as the sweep: without it the search settles the entire graph,
        including places no driver would ever consider.

        Returns ``({node_id: (seconds, metres)}, origin_node)``.
        """
        # Snap into the largest strongly connected component, not merely to the closest
        # node. The nearest node to Amsterdam Centraal sits in a tiny forecourt
        # component; a sweep launched from there reaches almost nothing, and every
        # candidate silently degrades to a straight-line estimate.
        main = self.largest_component(profile)
        origin = self.nearest_node(lat, lon, profile, component=main)
        if origin is None:
            origin = self.nearest_node(lat, lon, profile)
        if origin is None:
            return {}, None

        edges = self.graph.edges_for(profile)
        best: dict[int, tuple[float, float]] = {origin: (0.0, 0.0)}
        heap: list[tuple[float, float, int]] = [(0.0, 0.0, origin)]
        settled: set[int] = set()

        while heap:
            seconds, metres, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            if seconds > max_seconds:
                break
            for neighbour, length, cost in edges.get(node, ()):
                if neighbour in settled:
                    continue
                new_seconds = seconds + cost
                if new_seconds > max_seconds:
                    continue
                known = best.get(neighbour)
                if known is None or new_seconds < known[0]:
                    best[neighbour] = (new_seconds, metres + length)
                    heapq.heappush(heap, (new_seconds, metres + length, neighbour))

        return best, origin

    def many_costs(
        self,
        lat: float,
        lon: float,
        targets: list[tuple[float, float]],
        profile: Profile,
        *,
        max_seconds: float = 1500.0,
    ) -> list[RouteResult | None]:
        """Route from one point to many, in a single graph sweep.

        A target with no entry is genuinely unreachable inside the budget, and the
        caller is expected to fall back rather than be handed a fabricated number.
        """
        costs, origin = self.costs_from(lat, lon, profile, max_seconds=max_seconds)
        if origin is None or not costs:
            return [None] * len(targets)

        # Targets snap inside the component the sweep actually covered, so a candidate
        # is not lost to a node twenty metres nearer on an unreachable service island.
        origin_component = self.components(profile).get(origin)
        out: list[RouteResult | None] = []
        for target_lat, target_lon in targets:
            node = self.nearest_node(target_lat, target_lon, profile, component=origin_component)
            entry = costs.get(node) if node is not None else None
            if entry is None:
                out.append(None)
                continue
            seconds, metres = entry
            out.append(
                RouteResult(
                    distance_m=metres,
                    duration_min=max(0.5, seconds / 60.0),
                    profile=profile,
                    provider="native-graph",
                    confidence=0.88,
                )
            )
        return out

    def route(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: Profile
    ) -> RouteResult:
        # Snap the destination first: it is the fixed point of the query. Then snap the
        # origin inside the destination component, so the endpoints are reachable from
        # one another by construction rather than by luck.
        goal = self.nearest_node(to_lat, to_lon, profile)
        if goal is None:
            raise RoutingUnavailableError("no routable node near the destination")
        goal_component = self.components(profile).get(goal)
        start = self.nearest_node(from_lat, from_lon, profile, component=goal_component)
        if start is None:
            # The origin cannot reach that component at all, which happens across the IJ
            # where the only link is a ferry. Fall back to the main road network so the
            # search still returns a usable estimate rather than nothing at all.
            main = self.largest_component(profile)
            goal = self.nearest_node(to_lat, to_lon, profile, component=main)
            start = self.nearest_node(from_lat, from_lon, profile, component=main)
        if start is None or goal is None:
            raise RoutingUnavailableError("no routable node near one of the endpoints")
        if start == goal:
            straight = haversine_m(from_lat, from_lon, to_lat, to_lon)
            speed = 20.0 if profile is Profile.CAR else WALK_SPEED_KMH
            return RouteResult(
                distance_m=straight,
                duration_min=max(0.5, straight / 1000.0 / speed * 60.0),
                profile=profile,
                provider="native-graph",
                geometry=[[from_lon, from_lat], [to_lon, to_lat]],
                confidence=0.7,
            )

        edges = self.graph.edges_for(profile)
        goal_pos = self.graph.nodes[goal]
        speed_ms = (60.0 if profile is Profile.CAR else WALK_SPEED_KMH) / 3.6

        def heuristic(node_id: int) -> float:
            lat, lon = self.graph.nodes[node_id]
            # Admissible: straight-line time at the fastest speed this profile can
            # achieve can never exceed the true remaining time, so A* stays optimal.
            return haversine_m(lat, lon, goal_pos[0], goal_pos[1]) / speed_ms

        open_heap: list[tuple[float, int]] = [(heuristic(start), start)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start: 0.0}
        distance: dict[int, float] = {start: 0.0}
        closed: set[int] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._build_result(
                    came_from, current, g_score[current], distance[current], profile
                )
            if current in closed:
                continue
            closed.add(current)

            for neighbour, length, seconds in edges.get(current, ()):
                if neighbour in closed:
                    continue
                tentative = g_score[current] + seconds
                if tentative < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    distance[neighbour] = distance[current] + length
                    heapq.heappush(open_heap, (tentative + heuristic(neighbour), neighbour))

        raise RoutingUnavailableError("no path between the endpoints in this graph")

    def _build_result(
        self, came_from: dict[int, int], goal: int, seconds: float, metres: float, profile: Profile
    ) -> RouteResult:
        path = [goal]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        geometry = [[self.graph.nodes[n][1], self.graph.nodes[n][0]] for n in path]
        return RouteResult(
            distance_m=metres,
            duration_min=max(0.5, seconds / 60.0),
            profile=profile,
            provider="native-graph",
            geometry=geometry,
            confidence=0.9,
        )


class NativeGraphProvider(RoutingProvider):
    """Routing provider backed by a locally cached OSM graph."""

    name = "native-graph"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._router: RoadRouter | None = None
        self._load_attempted = False

    @property
    def cache_path(self) -> Path:
        return self.settings.data_dir / "osm" / "road_graph.json.gz"

    def available(self) -> bool:
        if self._router is not None:
            return True
        if self._load_attempted:
            return False
        return self.cache_path.exists()

    def _ensure_router(self) -> RoadRouter:
        if self._router is not None:
            return self._router
        self._load_attempted = True
        if not self.cache_path.exists():
            raise RoutingUnavailableError(
                f"no cached road graph at {self.cache_path}; run: pf ingest roads"
            )
        with gzip.open(self.cache_path, "rt", encoding="utf-8") as handle:
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
        log.info("loaded cached road graph: %d nodes", len(graph))
        self._router = RoadRouter(graph)
        return self._router

    def route(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: Profile
    ) -> RouteResult:
        return self._ensure_router().route(from_lat, from_lon, to_lat, to_lon, profile)

    def save_graph(self, graph: Graph) -> Path:
        path = self.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": {str(k): [v[0], v[1]] for k, v in graph.nodes.items()},
            "car_edges": {
                str(k): [[n, round(d, 2), round(s, 3)] for n, d, s in v]
                for k, v in graph.car_edges.items()
            },
            "foot_edges": {
                str(k): [[n, round(d, 2), round(s, 3)] for n, d, s in v]
                for k, v in graph.foot_edges.items()
            },
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        self._router = RoadRouter(graph)
        self._load_attempted = False
        return path


def haversine_bearing_fallback(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line metres. Kept here so callers need not import the geo module."""
    return math.hypot((lat2 - lat1) * 110540.0, (lon2 - lon1) * 111320.0 * 0.61)
