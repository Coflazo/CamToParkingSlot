"""Road graphs are per region, and the right one has to be chosen.

A single cached graph was fine while the product was Dutch. With two cities it is a bug
with a plausible output: an Istanbul query answered from the Amsterdam extract snaps both
endpoints to Dutch streets and returns a perfectly reasonable-looking distance for a
route on the wrong continent. Nothing downstream can detect that, which is why the
selection is tested rather than assumed.
"""

from __future__ import annotations

import math

import pytest

from parkfit.config import Settings
from parkfit.routing.graph import (
    Graph,
    NativeGraphProvider,
    region_key,
)
from parkfit.routing.provider import Profile, RoutingUnavailableError

AMSTERDAM = (52.33, 4.82, 52.41, 4.97)
ISTANBUL = (41.02, 28.98, 41.08, 29.05)


def tiny_graph(lat: float, lon: float, n: int = 6) -> Graph:
    """A short two-way chain running east from a point."""
    graph = Graph()
    step = 0.0009
    for i in range(n):
        graph.nodes[i] = (lat, lon + step * i)
    for i in range(n - 1):
        a, b = graph.nodes[i], graph.nodes[i + 1]
        metres = math.hypot((b[0] - a[0]) * 110540.0, (b[1] - a[1]) * 111320.0 * 0.61)
        for start, end in ((i, i + 1), (i + 1, i)):
            graph.car_edges.setdefault(start, []).append((end, metres, metres / 10.0))
            graph.foot_edges.setdefault(start, []).append((end, metres, metres / 1.33))
    return graph


@pytest.fixture
def provider(tmp_path) -> NativeGraphProvider:
    p = NativeGraphProvider(Settings(data_dir=tmp_path))
    p.save_graph(tiny_graph(52.37, 4.90), country="NL", bbox=AMSTERDAM)
    p.save_graph(tiny_graph(41.04, 29.00), country="TR", bbox=ISTANBUL)
    return p


def test_region_keys_are_stable_and_filename_safe():
    key = region_key("nl", AMSTERDAM)
    assert key.startswith("NL_")
    assert "/" not in key and " " not in key
    # Negative coordinates must not produce a leading dash, which reads as a CLI flag.
    assert not region_key("FR", (-1.0, -2.0, 3.0, 4.0)).startswith("-")
    assert "m1.000" in region_key("FR", (-1.0, -2.0, 3.0, 4.0))


def test_saving_a_second_city_does_not_replace_the_first(provider):
    """The whole reason regions exist."""
    countries = {r.country for r in provider.regions()}
    assert countries == {"NL", "TR"}
    assert len(provider.regions()) == 2


def test_the_graph_covering_the_origin_is_the_one_used(provider):
    dutch = provider.router_for(52.37, 4.90)
    turkish = provider.router_for(41.04, 29.00)
    assert dutch is not turkish

    # Each router only knows its own city, which is the point.
    assert dutch.nearest_node(52.37, 4.90, Profile.CAR) is not None
    assert turkish.nearest_node(41.04, 29.00, Profile.CAR) is not None


def test_a_point_no_graph_covers_is_refused_rather_than_answered(provider):
    """Refusing is what makes the routing service fall through to its estimator.

    Answering from whichever graph happened to load would produce a plausible distance
    for a route through the wrong country, and a plausible wrong number is worse than an
    admitted gap because the fallback chain cannot see it.
    """
    with pytest.raises(RoutingUnavailableError) as caught:
        provider.router_for(52.52, 13.405)  # Berlin
    assert "no cached road graph covers" in str(caught.value)


def test_routing_end_to_end_uses_the_local_graph(provider):
    result = provider.route(41.04, 29.00, 41.04, 29.0036, Profile.CAR)
    assert result.distance_m > 0
    assert result.provider == "native-graph"


def test_a_region_is_only_read_when_it_covers_the_point(provider, tmp_path):
    """Reading a city extract to answer a query in another city is pure waste.

    A fresh provider over the same directory, because the one that wrote the graphs
    already holds their routers: it built them in memory, so keeping them costs nothing
    and dropping them would mean parsing straight back what was just written.
    """
    fresh = NativeGraphProvider(Settings(data_dir=tmp_path))
    assert not fresh._routers

    fresh.router_for(41.04, 29.00)
    loaded = {p.name for p in fresh._routers}
    assert any(name.startswith("TR_") for name in loaded)
    assert not any(name.startswith("NL_") for name in loaded)


def test_an_unusable_cache_does_not_take_down_the_others(provider, tmp_path):
    """One corrupt file must not make a second city unroutable."""
    broken = provider.graph_dir / "DE_52.000_13.000_53.000_14.000.json.gz"
    broken.write_bytes(b"this is not gzip")

    # The broken region claims Berlin, and asking for Berlin fails cleanly.
    with pytest.raises(RoutingUnavailableError):
        provider.router_for(52.5, 13.5)
    # The healthy regions still answer.
    assert provider.router_for(41.04, 29.00) is not None


def test_a_legacy_single_graph_is_still_read(tmp_path):
    """An existing install must not silently lose its graph on upgrade."""
    p = NativeGraphProvider(Settings(data_dir=tmp_path))
    p.save_graph(tiny_graph(52.37, 4.90))  # no bbox: the legacy path
    assert p.cache_path.exists()

    regions = p.regions()
    assert len(regions) == 1
    assert regions[0].bbox is None
    # With no recorded extent it is tried for any point rather than assumed to cover
    # none, which is how this behaved before regions existed.
    assert regions[0].contains(52.37, 4.90)
    assert p.router_for(52.37, 4.90) is not None


def test_available_is_true_as_soon_as_any_region_exists(provider, tmp_path):
    assert provider.available()
    assert not NativeGraphProvider(Settings(data_dir=tmp_path / "empty")).available()
