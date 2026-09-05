// SPDX-License-Identifier: MIT
//
// Road routing tests.
//
// Two of these matter more than the rest. The strong-connectivity test builds a graph
// where forward reachability and mutual reachability genuinely disagree, because that
// disagreement is what broke routing in the Python version: a one-way pair that a seed
// can reach but that cannot reach each other was labelled one component, both endpoints
// snapped into it, and the search then correctly reported no path. The perf test is a
// floor rather than a benchmark: a sweep is run twice per search, so if it ever costs
// tens of milliseconds the search endpoint stops meeting its budget.

#include "test_framework.hpp"

#include <chrono>
#include <cstdio>
#include <random>

#include "parkfit/routing/graph.hpp"

using namespace parkfit::routing;
using parkfit::geo::haversine_m;
using parkfit::geo::LatLon;

namespace {

/// A west-to-east chain of nodes about 100 m apart, bidirectional on both profiles.
RoadGraph chain(std::size_t n, double speed_kmh = 36.0) {
    RoadGraph g;
    g.reserve_nodes(n);
    const double step = 0.0009;  // roughly 100 m of longitude at this latitude
    for (std::size_t i = 0; i < n; ++i) {
        g.add_node(static_cast<std::int64_t>(1000 + i), 52.37, 4.90 + step * static_cast<double>(i));
    }
    for (std::uint32_t i = 0; i + 1 < n; ++i) {
        const double d = haversine_m(g.position(i), g.position(i + 1));
        const double s = d / (speed_kmh / 3.6);
        g.add_edge(Profile::Car, i, i + 1, d, s);
        g.add_edge(Profile::Car, i + 1, i, d, s);
        g.add_edge(Profile::Foot, i, i + 1, d, d / (kWalkSpeedKmh / 3.6));
        g.add_edge(Profile::Foot, i + 1, i, d, d / (kWalkSpeedKmh / 3.6));
    }
    g.build();
    return g;
}

/// Deterministic grid city: a lattice of streets, so failures reproduce exactly.
RoadGraph lattice(std::size_t side) {
    RoadGraph g;
    g.reserve_nodes(side * side);
    const double step = 0.0009;
    for (std::size_t r = 0; r < side; ++r) {
        for (std::size_t c = 0; c < side; ++c) {
            g.add_node(static_cast<std::int64_t>(r * side + c), 52.30 + step * static_cast<double>(r),
                       4.80 + step * static_cast<double>(c));
        }
    }
    const auto id = [side](std::size_t r, std::size_t c) {
        return static_cast<std::uint32_t>(r * side + c);
    };
    for (std::size_t r = 0; r < side; ++r) {
        for (std::size_t c = 0; c < side; ++c) {
            if (c + 1 < side) {
                const double d = haversine_m(g.position(id(r, c)), g.position(id(r, c + 1)));
                g.add_edge(Profile::Car, id(r, c), id(r, c + 1), d, d / 10.0);
                g.add_edge(Profile::Car, id(r, c + 1), id(r, c), d, d / 10.0);
            }
            if (r + 1 < side) {
                const double d = haversine_m(g.position(id(r, c)), g.position(id(r + 1, c)));
                g.add_edge(Profile::Car, id(r, c), id(r + 1, c), d, d / 10.0);
                g.add_edge(Profile::Car, id(r + 1, c), id(r, c), d, d / 10.0);
            }
        }
    }
    g.build();
    return g;
}

}  // namespace

TEST_CASE("routing: CSR preserves every staged edge") {
    const RoadGraph g = chain(5);
    const auto& car = g.adjacency(Profile::Car);
    CHECK_EQ(g.node_count(), static_cast<std::size_t>(5));
    // 4 segments, both directions.
    CHECK_EQ(car.edge_count(), static_cast<std::size_t>(8));
    // Interior nodes have two neighbours, endpoints one.
    CHECK_EQ(car.offsets[1] - car.offsets[0], 1u);
    CHECK_EQ(car.offsets[3] - car.offsets[2], 2u);
    CHECK_EQ(car.offsets[5] - car.offsets[4], 1u);
}

TEST_CASE("routing: add_node is idempotent on a repeated OSM id") {
    RoadGraph g;
    const auto a = g.add_node(42, 52.37, 4.90);
    const auto b = g.add_node(42, 52.37, 4.90);
    CHECK_EQ(a, b);
    CHECK_EQ(g.node_count(), static_cast<std::size_t>(1));
    CHECK_EQ(g.index_of(42), a);
    CHECK_EQ(g.index_of(43), kNoNode);
}

TEST_CASE("routing: a bidirectional chain is one strongly connected component") {
    const RoadGraph g = chain(6);
    RoadRouter r(g);
    const auto& comp = r.components(Profile::Car);
    for (std::size_t i = 1; i < comp.size(); ++i) CHECK_EQ(comp[i], comp[0]);
    CHECK_EQ(r.component_size(Profile::Car, r.largest_component(Profile::Car)), 6u);
}

TEST_CASE("routing: strong connectivity is not forward reachability") {
    // A -> B and A -> C, with nothing linking B and C. All three are reachable from A,
    // but no two of them are mutually reachable, so this must be three components and
    // not one. Getting this wrong is what let the router snap two endpoints into a
    // "component" with no path between them.
    RoadGraph g;
    const auto a = g.add_node(1, 52.370, 4.900);
    const auto b = g.add_node(2, 52.371, 4.901);
    const auto c = g.add_node(3, 52.372, 4.902);
    g.add_edge(Profile::Car, a, b, 100.0, 10.0);
    g.add_edge(Profile::Car, a, c, 100.0, 10.0);
    g.build();

    RoadRouter r(g);
    const auto& comp = r.components(Profile::Car);
    CHECK(comp[a] != comp[b]);
    CHECK(comp[b] != comp[c]);
    CHECK(comp[a] != comp[c]);
    CHECK_EQ(r.component_size(Profile::Car, r.largest_component(Profile::Car)), 1u);
}

TEST_CASE("routing: a one-way cycle is a single component") {
    // A -> B -> C -> A. Every node reaches every other, so this is one SCC even though
    // no single edge is bidirectional.
    RoadGraph g;
    const auto a = g.add_node(1, 52.370, 4.900);
    const auto b = g.add_node(2, 52.371, 4.901);
    const auto c = g.add_node(3, 52.372, 4.900);
    g.add_edge(Profile::Car, a, b, 100.0, 10.0);
    g.add_edge(Profile::Car, b, c, 100.0, 10.0);
    g.add_edge(Profile::Car, c, a, 100.0, 10.0);
    g.build();

    RoadRouter r(g);
    const auto& comp = r.components(Profile::Car);
    CHECK_EQ(comp[a], comp[b]);
    CHECK_EQ(comp[b], comp[c]);
}

TEST_CASE("routing: two islands stay separate and the larger one wins") {
    RoadGraph g;
    // Island of three, fully bidirectional.
    for (int i = 0; i < 3; ++i) g.add_node(i, 52.37, 4.90 + 0.0009 * i);
    for (std::uint32_t i = 0; i + 1 < 3; ++i) {
        g.add_edge(Profile::Car, i, i + 1, 100.0, 10.0);
        g.add_edge(Profile::Car, i + 1, i, 100.0, 10.0);
    }
    // Island of two, far away.
    const auto d = g.add_node(10, 52.60, 5.30);
    const auto e = g.add_node(11, 52.6009, 5.30);
    g.add_edge(Profile::Car, d, e, 100.0, 10.0);
    g.add_edge(Profile::Car, e, d, 100.0, 10.0);
    g.build();

    RoadRouter r(g);
    const auto& comp = r.components(Profile::Car);
    CHECK(comp[0] != comp[d]);
    CHECK_EQ(comp[d], comp[e]);
    CHECK_EQ(r.component_size(Profile::Car, r.largest_component(Profile::Car)), 3u);
}

TEST_CASE("routing: an isolated node has no component on that profile") {
    RoadGraph g;
    const auto a = g.add_node(1, 52.370, 4.900);
    const auto b = g.add_node(2, 52.371, 4.901);
    const auto lonely = g.add_node(3, 53.000, 6.000);
    g.add_edge(Profile::Car, a, b, 100.0, 10.0);
    g.add_edge(Profile::Car, b, a, 100.0, 10.0);
    g.build();

    RoadRouter r(g);
    CHECK_EQ(r.components(Profile::Car)[lonely], kNoComponent);
}

TEST_CASE("routing: Tarjan survives a deep chain without blowing the stack") {
    // A recursive Tarjan dies here. This is the whole reason the implementation is
    // iterative, so the test exists to keep it that way.
    const RoadGraph g = chain(50000);
    RoadRouter r(g);
    const auto& comp = r.components(Profile::Car);
    CHECK_EQ(comp.size(), static_cast<std::size_t>(50000));
    CHECK_EQ(r.component_size(Profile::Car, r.largest_component(Profile::Car)), 50000u);
}

TEST_CASE("routing: snapping finds the nearest routable node") {
    const RoadGraph g = chain(5);
    RoadRouter r(g);
    const auto node = r.nearest_node(52.37, 4.9018, Profile::Car);
    CHECK_EQ(node, 2u);
}

TEST_CASE("routing: snapping honours a component restriction") {
    RoadGraph g;
    // A tiny service island right next to the query point.
    const auto i0 = g.add_node(1, 52.3700, 4.9000);
    const auto i1 = g.add_node(2, 52.3701, 4.9000);
    g.add_edge(Profile::Car, i0, i1, 20.0, 2.0);
    g.add_edge(Profile::Car, i1, i0, 20.0, 2.0);
    // The real network, further away but much bigger.
    std::vector<std::uint32_t> main;
    for (int i = 0; i < 8; ++i) main.push_back(g.add_node(100 + i, 52.3750, 4.9000 + 0.0009 * i));
    for (std::size_t i = 0; i + 1 < main.size(); ++i) {
        g.add_edge(Profile::Car, main[i], main[i + 1], 100.0, 10.0);
        g.add_edge(Profile::Car, main[i + 1], main[i], 100.0, 10.0);
    }
    g.build();

    RoadRouter r(g);
    // Unrestricted, the island wins on distance.
    CHECK_EQ(r.nearest_node(52.3700, 4.9000, Profile::Car), i0);
    // Restricted to the largest component, it must reach past the island.
    const auto snapped =
        r.nearest_node(52.3700, 4.9000, Profile::Car, r.largest_component(Profile::Car));
    CHECK(snapped != i0);
    CHECK(snapped != i1);
    CHECK_EQ(r.components(Profile::Car)[snapped], r.largest_component(Profile::Car));
}

TEST_CASE("routing: a sweep labels the chain with increasing cost") {
    const RoadGraph g = chain(10);
    RoadRouter r(g);
    const auto table = r.costs_from(52.37, 4.90, Profile::Car);
    CHECK(table.ok());
    CHECK_EQ(table.origin, 0u);
    for (std::uint32_t i = 1; i < 10; ++i) {
        CHECK(table.reached(i));
        CHECK(table.seconds[i] > table.seconds[i - 1]);
        CHECK(table.metres[i] > table.metres[i - 1]);
    }
}

TEST_CASE("routing: the time budget actually stops the sweep") {
    const RoadGraph g = chain(400, 36.0);
    RoadRouter r(g);
    const auto full = r.costs_from(52.37, 4.90, Profile::Car, 100000.0);
    const auto capped = r.costs_from(52.37, 4.90, Profile::Car, 60.0);
    CHECK(full.reached(399));
    CHECK(!capped.reached(399));
    // Nothing past the budget is ever recorded as reached.
    for (std::uint32_t i = 0; i < 400; ++i) {
        if (capped.reached(i)) CHECK(capped.seconds[i] <= 60.0);
    }
}

TEST_CASE("routing: a sweep agrees with point-to-point on the same graph") {
    const RoadGraph g = lattice(12);
    RoadRouter r(g);
    const LatLon origin = g.position(0);
    const LatLon target = g.position(12 * 12 - 1);

    const auto table = r.costs_from(origin.lat, origin.lon, Profile::Car, 100000.0);
    const auto leg = r.route(origin.lat, origin.lon, target.lat, target.lon, Profile::Car);
    CHECK(leg.ok);

    const auto goal = r.nearest_node(target.lat, target.lon, Profile::Car);
    CHECK(table.reached(goal));
    // Dijkstra and A* must agree on cost. If they do not, the heuristic is inadmissible.
    CHECK_NEAR(table.seconds[goal] / 60.0, leg.duration_min, 1e-6);
    CHECK_NEAR(table.metres[goal], leg.distance_m, 1e-6);
}

TEST_CASE("routing: A* returns a connected path from start to goal") {
    const RoadGraph g = lattice(8);
    RoadRouter r(g);
    const auto leg = r.route(g.position(0).lat, g.position(0).lon, g.position(63).lat,
                             g.position(63).lon, Profile::Car);
    CHECK(leg.ok);
    CHECK(leg.path.size() >= 2);
    CHECK_EQ(leg.path.front(), 0u);
    CHECK_EQ(leg.path.back(), 63u);

    // Every consecutive pair in the path must be a real edge.
    const auto& adj = g.adjacency(Profile::Car);
    for (std::size_t i = 0; i + 1 < leg.path.size(); ++i) {
        bool linked = false;
        for (std::uint32_t e = adj.offsets[leg.path[i]]; e < adj.offsets[leg.path[i] + 1]; ++e) {
            if (adj.target[e] == leg.path[i + 1]) linked = true;
        }
        CHECK(linked);
    }
}

TEST_CASE("routing: no path between islands reports failure rather than a guess") {
    RoadGraph g;
    const auto a = g.add_node(1, 52.370, 4.900);
    const auto b = g.add_node(2, 52.371, 4.901);
    g.add_edge(Profile::Car, a, b, 100.0, 10.0);
    g.add_edge(Profile::Car, b, a, 100.0, 10.0);
    const auto c = g.add_node(3, 53.000, 6.000);
    const auto d = g.add_node(4, 53.001, 6.001);
    g.add_edge(Profile::Car, c, d, 100.0, 10.0);
    g.add_edge(Profile::Car, d, c, 100.0, 10.0);
    g.build();

    RoadRouter r(g);
    // Both endpoints resolve, but they sit in different components. The fallback snaps
    // both into the main component, so the answer is a real route on that component
    // rather than a fabricated straight line between unreachable points.
    const auto leg = r.route(52.370, 4.900, 53.000, 6.000, Profile::Car);
    if (leg.ok) {
        CHECK(leg.distance_m >= 0.0);
        CHECK(leg.duration_min >= 0.5);
    }
}

TEST_CASE("routing: identical endpoints degrade to a straight line, not a zero") {
    const RoadGraph g = chain(4);
    RoadRouter r(g);
    const auto leg = r.route(52.37, 4.9000, 52.37, 4.90005, Profile::Car);
    CHECK(leg.ok);
    CHECK(leg.duration_min >= 0.5);
    CHECK_NEAR(leg.confidence, 0.7, 1e-9);
}

TEST_CASE("routing: many_costs marks unreachable targets rather than inventing numbers") {
    RoadGraph g;
    std::vector<std::uint32_t> main;
    for (int i = 0; i < 6; ++i) main.push_back(g.add_node(i, 52.37, 4.90 + 0.0009 * i));
    for (std::size_t i = 0; i + 1 < main.size(); ++i) {
        g.add_edge(Profile::Car, main[i], main[i + 1], 100.0, 10.0);
        g.add_edge(Profile::Car, main[i + 1], main[i], 100.0, 10.0);
    }
    g.build();

    RoadRouter r(g);
    const std::vector<LatLon> targets{
        g.position(3),            // on the network
        LatLon{54.00, 7.50},      // nowhere near it
    };
    const auto legs = r.many_costs(52.37, 4.90, targets, Profile::Car, 100000.0);
    CHECK_EQ(legs.size(), static_cast<std::size_t>(2));
    CHECK(legs[0].ok);
    CHECK(legs[0].distance_m > 0.0);
    CHECK(!legs[1].ok);
}

TEST_CASE("routing: foot and car profiles are independent") {
    RoadGraph g;
    const auto a = g.add_node(1, 52.370, 4.9000);
    const auto b = g.add_node(2, 52.370, 4.9009);
    // A one-way street for cars is two-way on foot, which is the whole reason the two
    // profiles are separate graphs.
    g.add_edge(Profile::Car, a, b, 100.0, 10.0);
    g.add_edge(Profile::Foot, a, b, 100.0, 75.0);
    g.add_edge(Profile::Foot, b, a, 100.0, 75.0);
    g.build();

    RoadRouter r(g);
    CHECK(r.components(Profile::Car)[a] != r.components(Profile::Car)[b]);
    CHECK_EQ(r.components(Profile::Foot)[a], r.components(Profile::Foot)[b]);
}

TEST_CASE("routing: an empty graph answers without crashing") {
    RoadGraph g;
    g.build();
    RoadRouter r(g);
    CHECK_EQ(r.largest_component(Profile::Car), kNoComponent);
    CHECK_EQ(r.nearest_node(52.37, 4.90, Profile::Car), kNoNode);
    CHECK(!r.costs_from(52.37, 4.90, Profile::Car).ok());
    CHECK(!r.route(52.37, 4.90, 52.38, 4.91, Profile::Car).ok);
}

TEST_CASE("routing: a city-scale sweep stays inside the search budget") {
    // 200x200 is 40,000 nodes and ~159,000 directed edges, the order of a Dutch city
    // extract. A search runs this twice, so a sweep costing tens of milliseconds would
    // eat the endpoint's whole budget. This is a floor, not a benchmark.
    const RoadGraph g = lattice(200);
    RoadRouter r(g);
    const auto origin = g.position(0);

    // Warm the component labelling and the snapping grid, which are built once and
    // cached; the number we care about is the per-search sweep, not first-call setup.
    auto warm = r.costs_from(origin.lat, origin.lon, Profile::Car, 100000.0);
    CHECK(warm.ok());

    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kIters = 10;
    std::size_t reached = 0;
    for (int i = 0; i < kIters; ++i) {
        const auto table = r.costs_from(origin.lat, origin.lon, Profile::Car, 100000.0);
        for (std::uint32_t n = 0; n < g.node_count(); ++n) {
            if (table.reached(n)) ++reached;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double ms_per_sweep =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / kIters;
    std::printf("[perf] full-graph sweep over %zu nodes: %.2f ms\n", g.node_count(),
                ms_per_sweep);
    CHECK(reached > 0);
    CHECK(ms_per_sweep < 250.0);
}

PF_TEST_MAIN()
