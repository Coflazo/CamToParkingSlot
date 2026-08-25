// SPDX-License-Identifier: MIT
//
// Spatial index tests, including a brute-force cross-check: the grid must return
// exactly what a linear scan would, or a driver silently loses candidate spaces
// near a cell boundary.

#include "test_framework.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

#include "parkfit/index/grid.hpp"

using namespace parkfit::index;
using parkfit::geo::LatLon;
using parkfit::geo::haversine_m;

namespace {

/// Deterministic scatter of points across Amsterdam, so failures reproduce exactly.
std::vector<LatLon> amsterdam_points(std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> lat(52.30, 52.43);
    std::uniform_real_distribution<double> lon(4.75, 5.02);
    std::vector<LatLon> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) pts.push_back(LatLon{lat(rng), lon(rng)});
    return pts;
}

}  // namespace

TEST_CASE("index: an empty grid answers queries without crashing") {
    SpatialGrid g;
    CHECK(g.query_radius(LatLon{52.37, 4.90}, 500.0).empty());
    CHECK(g.query_knn(LatLon{52.37, 4.90}, 5).empty());
    CHECK_EQ(g.size(), static_cast<std::size_t>(0));
}

TEST_CASE("index: a single point is found within range and missed outside it") {
    SpatialGrid g;
    const LatLon p{52.3676, 4.9041};
    g.insert(p, 7);
    const auto near = g.query_radius(p, 10.0);
    CHECK_EQ(near.size(), static_cast<std::size_t>(1));
    CHECK_EQ(near[0].payload, static_cast<std::uint32_t>(7));
    CHECK_NEAR(near[0].distance_m, 0.0, 1e-6);

    const LatLon far{52.3676, 4.9500};  // roughly 3.1 km east
    CHECK(g.query_radius(far, 500.0).empty());
}

TEST_CASE("index: results agree exactly with a brute-force scan") {
    const auto pts = amsterdam_points(4000);
    SpatialGrid g(250.0);
    for (std::uint32_t i = 0; i < pts.size(); ++i) g.insert(pts[i], i);

    const LatLon queries[] = {{52.3676, 4.9041}, {52.3200, 4.7600}, {52.4250, 5.0150},
                              {52.3800, 4.8500}, {52.4300, 4.7501}};
    for (const auto& q : queries) {
        for (double radius : {150.0, 600.0, 2000.0}) {
            std::vector<std::uint32_t> expected;
            for (std::uint32_t i = 0; i < pts.size(); ++i) {
                if (haversine_m(q, pts[i]) <= radius) expected.push_back(i);
            }
            auto got = g.query_radius(q, radius);
            std::vector<std::uint32_t> got_ids;
            got_ids.reserve(got.size());
            for (const auto& h : got) got_ids.push_back(h.payload);
            std::sort(expected.begin(), expected.end());
            std::sort(got_ids.begin(), got_ids.end());
            CHECK(expected == got_ids);
        }
    }
}

TEST_CASE("index: radius results are ordered nearest first") {
    const auto pts = amsterdam_points(2000);
    SpatialGrid g;
    for (std::uint32_t i = 0; i < pts.size(); ++i) g.insert(pts[i], i);
    const auto hits = g.query_radius(LatLon{52.3676, 4.9041}, 2500.0);
    CHECK(!hits.empty());
    for (std::size_t i = 1; i < hits.size(); ++i) {
        CHECK(hits[i - 1].distance_m <= hits[i].distance_m);
    }
}

TEST_CASE("index: max_results truncates to the nearest few") {
    const auto pts = amsterdam_points(2000);
    SpatialGrid g;
    for (std::uint32_t i = 0; i < pts.size(); ++i) g.insert(pts[i], i);
    const LatLon q{52.3676, 4.9041};
    const auto all = g.query_radius(q, 3000.0);
    const auto few = g.query_radius(q, 3000.0, 5);
    CHECK(all.size() > 5);
    CHECK_EQ(few.size(), static_cast<std::size_t>(5));
    for (std::size_t i = 0; i < few.size(); ++i) CHECK_EQ(few[i].payload, all[i].payload);
}

TEST_CASE("index: knn expands its radius until it finds enough") {
    SpatialGrid g;
    // Three points, all several kilometres from the query, so the initial 500 m
    // radius must widen. A fixed radius here would return nothing at all, which is
    // exactly the failure mode outside the dense city centres.
    g.insert(LatLon{52.4000, 4.9500}, 1);
    g.insert(LatLon{52.4200, 4.9700}, 2);
    g.insert(LatLon{52.4400, 4.9900}, 3);
    const auto hits = g.query_knn(LatLon{52.3676, 4.9041}, 2);
    CHECK_EQ(hits.size(), static_cast<std::size_t>(2));
    CHECK_EQ(hits[0].payload, static_cast<std::uint32_t>(1));
}

TEST_CASE("index: knn gives up at the maximum radius rather than looping forever") {
    SpatialGrid g;
    g.insert(LatLon{52.3676, 4.9041}, 1);
    const auto hits = g.query_knn(LatLon{53.5, 6.5}, 10, 500.0, 5000.0);
    CHECK(hits.empty());
}

TEST_CASE("index: clear resets the structure") {
    SpatialGrid g;
    g.insert(LatLon{52.3676, 4.9041}, 1);
    CHECK_EQ(g.size(), static_cast<std::size_t>(1));
    g.clear();
    CHECK_EQ(g.size(), static_cast<std::size_t>(0));
    CHECK(g.query_radius(LatLon{52.3676, 4.9041}, 100.0).empty());
}

TEST_CASE("index: inserting after a query rebuilds correctly") {
    SpatialGrid g;
    g.insert(LatLon{52.3676, 4.9041}, 1);
    CHECK_EQ(g.query_radius(LatLon{52.3676, 4.9041}, 50.0).size(), static_cast<std::size_t>(1));
    g.insert(LatLon{52.3677, 4.9042}, 2);
    CHECK_EQ(g.query_radius(LatLon{52.3676, 4.9041}, 50.0).size(), static_cast<std::size_t>(2));
}

TEST_CASE("index: a bay-scale workload answers in well under a millisecond") {
    // 250k entries is the order of the Amsterdam parkeervakken set. This is a
    // performance floor, not a benchmark: if a radius query ever costs milliseconds
    // the search endpoint stops meeting its 500 ms budget under load.
    const auto pts = amsterdam_points(250000, 7);
    SpatialGrid g(250.0);
    g.reserve(pts.size());
    for (std::uint32_t i = 0; i < pts.size(); ++i) g.insert(pts[i], i);
    g.build();

    const LatLon q{52.3676, 4.9041};
    auto warm = g.query_radius(q, 500.0);
    CHECK(!warm.empty());

    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kIters = 200;
    std::size_t total = 0;
    for (int i = 0; i < kIters; ++i) total += g.query_radius(q, 500.0).size();
    const auto t1 = std::chrono::steady_clock::now();

    const double us_per_query =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
    std::printf("       250k entries, 500 m radius: %.1f us/query, %zu hits\n", us_per_query,
                total / kIters);
    CHECK(us_per_query < 1000.0);
}

PF_TEST_MAIN()
