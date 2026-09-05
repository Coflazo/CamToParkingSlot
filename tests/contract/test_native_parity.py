"""The C++ core and its Python reference must answer identically.

Several modules exist twice on purpose: a readable Python implementation that runs on an
uncompiled checkout, and a C++ one that runs in production. That arrangement is only safe
while the two agree, and nothing enforced it until this file. ``src/parkfit/native.py``
and ``src/parkfit/geo/shapes.py`` have both cited this suite in their docstrings for a
while; it did not exist, and the duplicates were free to drift.

The comparisons here are deliberately not "close enough". Routing costs and bay
measurements are arithmetic over the same inputs in the same order, so they should match
to floating-point noise, and a tolerance wide enough to hide a real divergence would
defeat the point.
"""

from __future__ import annotations

import math

import pytest

from parkfit.native import native
from parkfit.routing.graph import Graph, NativeRoadRouter, RoadRouter
from parkfit.routing.provider import Profile

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(native is None, reason="parkfit_native is not built"),
]


def _lattice(side: int = 14) -> Graph:
    """A deterministic grid city.

    A lattice rather than a chain because a chain has exactly one path between any two
    points, so it cannot catch a router that picks a different equally-cheap route. A
    grid has many, and the two implementations still have to agree on the cost.
    """
    graph = Graph()
    step = 0.0009
    for row in range(side):
        for col in range(side):
            graph.nodes[row * side + col] = (52.30 + step * row, 4.80 + step * col)

    def link(a: int, b: int) -> None:
        lat_a, lon_a = graph.nodes[a]
        lat_b, lon_b = graph.nodes[b]
        metres = math.hypot((lat_b - lat_a) * 110540.0, (lon_b - lon_a) * 111320.0 * 0.61)
        for start, end in ((a, b), (b, a)):
            graph.car_edges.setdefault(start, []).append((end, metres, metres / 10.0))
            graph.foot_edges.setdefault(start, []).append((end, metres, metres / 1.33))

    for row in range(side):
        for col in range(side):
            if col + 1 < side:
                link(row * side + col, row * side + col + 1)
            if row + 1 < side:
                link(row * side + col, (row + 1) * side + col)
    return graph


@pytest.fixture(scope="module")
def routers() -> tuple[RoadRouter, NativeRoadRouter]:
    graph = _lattice()
    return RoadRouter(graph), NativeRoadRouter.from_graph(graph)


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_components_partition_matches(routers, profile):
    """Component *ids* may differ; the partition they induce may not.

    The two implementations seed Tarjan in different orders, so component numbering is an
    implementation detail. What has to hold is that two nodes are grouped together in one
    exactly when they are grouped together in the other.
    """
    python_router, native_router = routers
    py_labels = python_router.components(profile)
    cpp_labels = native_router.components(profile)

    assert py_labels.keys() == cpp_labels.keys()

    py_groups = {frozenset(n for n, c in py_labels.items() if c == g) for g in set(py_labels.values())}
    cpp_groups = {
        frozenset(n for n, c in cpp_labels.items() if c == g) for g in set(cpp_labels.values())
    }
    assert py_groups == cpp_groups


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_largest_component_holds_the_same_nodes(routers, profile):
    python_router, native_router = routers
    py_labels = python_router.components(profile)
    cpp_labels = native_router.components(profile)

    py_main = python_router.largest_component(profile)
    cpp_main = native_router.largest_component(profile)
    assert py_main is not None and cpp_main is not None

    assert {n for n, c in py_labels.items() if c == py_main} == {
        n for n, c in cpp_labels.items() if c == cpp_main
    }


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_nearest_node_agrees(routers, profile):
    python_router, native_router = routers
    for lat, lon in ((52.3000, 4.8000), (52.3055, 4.8060), (52.3110, 4.8115), (52.2990, 4.7990)):
        assert python_router.nearest_node(lat, lon, profile) == native_router.nearest_node(
            lat, lon, profile
        )


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_costs_from_agrees_node_for_node(routers, profile):
    """The whole sweep, not a sample of it.

    Comparing only a handful of nodes would miss a divergence at the frontier, which is
    exactly where a budget or a relaxation bug would show up first.
    """
    python_router, native_router = routers
    py_costs, py_origin = python_router.costs_from(52.3000, 4.8000, profile, max_seconds=100000.0)
    cpp_costs, cpp_origin = native_router.costs_from(
        52.3000, 4.8000, profile, max_seconds=100000.0
    )

    assert py_origin == cpp_origin
    assert py_costs.keys() == cpp_costs.keys()
    for node in py_costs:
        assert py_costs[node][0] == pytest.approx(cpp_costs[node][0], abs=1e-9)
        assert py_costs[node][1] == pytest.approx(cpp_costs[node][1], abs=1e-9)


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_the_time_budget_cuts_both_off_at_the_same_place(routers, profile):
    python_router, native_router = routers
    py_costs, _ = python_router.costs_from(52.3000, 4.8000, profile, max_seconds=60.0)
    cpp_costs, _ = native_router.costs_from(52.3000, 4.8000, profile, max_seconds=60.0)
    assert py_costs.keys() == cpp_costs.keys()


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_many_costs_agrees(routers, profile):
    python_router, native_router = routers
    targets = [
        (52.3108, 4.8112),
        (52.3050, 4.8050),
        (52.3000, 4.8110),
        (54.0000, 7.5000),  # nowhere near the graph: both must return None
    ]
    py_rows = python_router.many_costs(52.3000, 4.8000, targets, profile, max_seconds=100000.0)
    cpp_rows = native_router.many_costs(52.3000, 4.8000, targets, profile, max_seconds=100000.0)

    assert len(py_rows) == len(cpp_rows) == len(targets)
    for py_row, cpp_row in zip(py_rows, cpp_rows, strict=True):
        assert (py_row is None) == (cpp_row is None)
        if py_row is None:
            continue
        assert py_row.distance_m == pytest.approx(cpp_row.distance_m, abs=1e-9)
        assert py_row.duration_min == pytest.approx(cpp_row.duration_min, abs=1e-9)
    assert py_rows[-1] is None


@pytest.mark.parametrize("profile", [Profile.CAR, Profile.FOOT])
def test_route_agrees_on_cost(routers, profile):
    """Cost, not path.

    A lattice has many shortest paths of identical cost, and which one a router returns
    depends on how its heap breaks ties. Demanding the same node sequence would be
    testing the tie-break, not the routing.
    """
    python_router, native_router = routers
    py_route = python_router.route(52.3000, 4.8000, 52.3108, 4.8112, profile)
    cpp_route = native_router.route(52.3000, 4.8000, 52.3108, 4.8112, profile)

    assert py_route.duration_min == pytest.approx(cpp_route.duration_min, abs=1e-9)
    assert py_route.distance_m == pytest.approx(cpp_route.distance_m, abs=1e-9)
    assert len(cpp_route.geometry) >= 2


def test_the_provider_string_says_which_engine_answered(routers):
    """A result must not imply work that did not happen."""
    python_router, native_router = routers
    assert python_router.route(52.3000, 4.8000, 52.3108, 4.8112, Profile.CAR).provider == (
        "python-graph"
    )
    assert native_router.route(52.3000, 4.8000, 52.3108, 4.8112, Profile.CAR).provider == (
        "native-graph"
    )


# ------------------------------------------------------- batched fit checks

def _vehicle():
    v = native.Vehicle()
    v.length_cm = 445.0
    v.body_width_cm = 180.0
    v.width_with_mirrors_cm = 201.0
    v.height_cm = 149.0
    v.weight_kg = 1300.0
    return v


@pytest.mark.parametrize(
    ("length_cm", "width_cm", "orientation"),
    [
        (600.0, 220.0, "parallel"),
        (480.0, 200.0, "parallel"),
        (500.0, 250.0, "perpendicular"),
        (430.0, 175.0, "perpendicular"),   # too small for this car
        (520.0, 230.0, "angled"),
        (600.0, 220.0, "unknown"),
    ],
)
def test_batched_bay_checks_match_one_at_a_time(length_cm, width_cm, orientation):
    """The batched entry point must be a pure speed-up, never a different answer."""
    vehicle = _vehicle()
    margins = native.Margins()
    margins.tight_threshold_cm = 15.0

    single = native.check_bay(
        vehicle, length_cm, width_cm, native.orientation_from_string(orientation), margins
    )
    batched = native.check_bays(vehicle, [length_cm], [width_cm], [orientation], margins)[0]

    assert single.verdict == batched.verdict
    assert single.min_slack_cm == pytest.approx(batched.min_slack_cm, abs=1e-9)
    assert list(single.unverified_dimensions) == list(batched.unverified_dimensions)


def test_batched_facility_checks_match_one_at_a_time():
    vehicle = _vehicle()
    margins = native.Margins()
    heights = [0.0, 190.0, 200.0, 260.0]

    batched = native.check_facilities(vehicle, heights, margins)
    assert len(batched) == len(heights)

    for height, result in zip(heights, batched, strict=True):
        limits = native.FacilityLimits()
        if height > 0.0:
            limits.max_height_cm = height
        single = native.check_facility(vehicle, limits, margins)
        assert single.verdict == result.verdict
        assert single.min_slack_cm == pytest.approx(result.min_slack_cm, abs=1e-9)


def test_an_unpublished_height_is_unverified_not_unlimited():
    """Zero means the operator published nothing. Reading it as unlimited routes a van
    into a barrier, which is the single most expensive mistake this engine can make."""
    verdict = native.check_facilities(_vehicle(), [0.0], native.Margins())[0]
    assert verdict.verdict != native.Verdict.FITS or verdict.unverified_dimensions


def test_batched_calls_reject_mismatched_list_lengths():
    """Silently zipping to the shortest list would drop candidates without a trace."""
    with pytest.raises(ValueError):
        native.check_bays(_vehicle(), [600.0, 500.0], [220.0], ["parallel", "parallel"])


def test_orientation_no_longer_requires_a_dutch_round_trip():
    """The search used to translate "parallel" back into "Langs" to satisfy the parser.

    Both spellings still work, because an adapter reading raw Amsterdam data legitimately
    has the Dutch one, but nothing downstream has to know Dutch any more.
    """
    assert native.orientation_from_string("parallel") == native.BayOrientation.PARALLEL
    assert native.orientation_from_string("Langs") == native.BayOrientation.PARALLEL
    assert native.orientation_from_string("perpendicular") == native.BayOrientation.PERPENDICULAR
    assert native.orientation_from_string("Haaks") == native.BayOrientation.PERPENDICULAR
    assert native.orientation_from_string("angled") == native.BayOrientation.ANGLED
    assert native.orientation_from_string("nonsense") == native.BayOrientation.UNKNOWN
