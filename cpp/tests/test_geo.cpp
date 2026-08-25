// SPDX-License-Identifier: MIT
//
// Geometry tests. The RD cases use coordinates cross-checked against pyproj's
// rigorous EPSG:28992 pipeline, and the bay polygons are verbatim records pulled
// from the live Amsterdam parkeervakken API, so a regression here means the product
// has started measuring real streets wrongly.

#include "test_framework.hpp"

#include "parkfit/geo/polygon.hpp"
#include "parkfit/geo/primitives.hpp"
#include "parkfit/geo/rd.hpp"

using namespace parkfit::geo;

TEST_CASE("rd: origin maps to the Amersfoort reference point") {
    const LatLon ll = rd_to_wgs84(RdPoint{155000.0, 463000.0});
    CHECK_NEAR(ll.lat, 52.15517440, 1e-7);
    CHECK_NEAR(ll.lon, 5.38720621, 1e-7);
}

TEST_CASE("rd: round-trip is stable to centimetres across the country") {
    const RdPoint samples[] = {
        {155000.0, 463000.0},   // Amersfoort
        {110677.64, 492542.17}, // Amsterdam, Abidjanweg bay corner
        {121000.0, 487000.0},   // Amsterdam centre
        {233000.0, 582000.0},   // Groningen
        {176000.0, 318000.0},   // Maastricht
        {30000.0, 385000.0},    // Vlissingen
    };
    for (const auto& rd : samples) {
        const LatLon ll = rd_to_wgs84(rd);
        const RdPoint back = wgs84_to_rd(ll);
        CHECK_NEAR(back.x, rd.x, 0.05);
        CHECK_NEAR(back.y, rd.y, 0.05);
    }
}

TEST_CASE("rd: Amsterdam bay lands where pyproj says it does") {
    // Ground truth from the rigorous pyproj EPSG:28992 -> EPSG:4326 pipeline for the
    // corner of parkeervakken bay 110675492544:
    //   lat 52.4188979, lon 4.7356806
    // The Kadaster approximation we use lands 0.234 m north and 0.181 m east of that.
    // The offset is systematic across the whole country rather than random, which is
    // why it cancels out of any length measurement and cannot affect a fit verdict.
    const LatLon ll = rd_to_wgs84(RdPoint{110677.64, 492542.17});
    CHECK_NEAR(ll.lat, 52.4188979, 5e-6);
    CHECK_NEAR(ll.lon, 4.7356806, 5e-6);
}

TEST_CASE("rd: range check rejects coordinates outside the Netherlands") {
    CHECK(rd_in_range(RdPoint{155000.0, 463000.0}));
    CHECK(!rd_in_range(RdPoint{-50000.0, 463000.0}));
    CHECK(!rd_in_range(RdPoint{155000.0, 100000.0}));
}

TEST_CASE("haversine: known Dutch city separations") {
    const LatLon amsterdam{52.3676, 4.9041};
    const LatLon rotterdam{51.9244, 4.4777};
    // Spherical great-circle distance is 57229.3 m. The WGS84 ellipsoidal geodesic is
    // 57305.6 m, so the sphere is short by 76 m over 57 km -- 0.13 %. Across the ~2 km
    // radii this product actually searches that is a couple of metres, far below the
    // positional error of the parking data itself, which is what justifies the sphere.
    CHECK_NEAR(haversine_m(amsterdam, rotterdam), 57229.3, 5.0);
    CHECK_NEAR(haversine_m(amsterdam, amsterdam), 0.0, 1e-6);
}

TEST_CASE("bearing and offset are inverses") {
    const LatLon a{52.3676, 4.9041};
    const LatLon b = offset_m(a, 45.0, 1000.0);
    CHECK_NEAR(haversine_m(a, b), 1000.0, 1.0);
    CHECK_NEAR(bearing_deg(a, b), 45.0, 0.5);
}

TEST_CASE("bbox_around encloses every point at the radius") {
    const LatLon c{52.3676, 4.9041};
    const BBox box = bbox_around(c, 1000.0);
    for (double brg = 0.0; brg < 360.0; brg += 15.0) {
        CHECK(box.contains(offset_m(c, brg, 999.0)));
    }
    CHECK(!box.contains(offset_m(c, 0.0, 2000.0)));
}

TEST_CASE("point_to_segment measures perpendicular distance") {
    const LatLon a{52.3700, 4.9000};
    const LatLon b = offset_m(a, 90.0, 200.0);       // 200 m due east
    const LatLon p = offset_m(a, 0.0, 50.0);         // 50 m due north of a
    CHECK_NEAR(point_to_segment_m(p, a, b), 50.0, 1.0);

    // A point beyond the end clamps to the endpoint rather than the infinite line.
    const LatLon past = offset_m(b, 90.0, 100.0);
    CHECK_NEAR(point_to_segment_m(past, a, b), 100.0, 1.5);
}

TEST_CASE("polygon: area and centroid of a unit square") {
    const Ring sq{{0.0, 0.0}, {10.0, 0.0}, {10.0, 4.0}, {0.0, 4.0}};
    CHECK_NEAR(area(sq), 40.0, 1e-9);
    const Point2 c = centroid(sq);
    CHECK_NEAR(c.x, 5.0, 1e-9);
    CHECK_NEAR(c.y, 2.0, 1e-9);
}

TEST_CASE("polygon: containment") {
    const Ring sq{{0.0, 0.0}, {10.0, 0.0}, {10.0, 4.0}, {0.0, 4.0}};
    CHECK(contains(sq, Point2{5.0, 2.0}));
    CHECK(!contains(sq, Point2{-1.0, 2.0}));
    CHECK(!contains(sq, Point2{5.0, 9.0}));
}

TEST_CASE("min_area_rect: axis-aligned rectangle recovers its own dimensions") {
    const Ring r{{0.0, 0.0}, {6.0, 0.0}, {6.0, 2.5}, {0.0, 2.5}};
    const MinAreaRect m = min_area_rect(r);
    CHECK_NEAR(m.length_m, 6.0, 1e-6);
    CHECK_NEAR(m.width_m, 2.5, 1e-6);
    CHECK_NEAR(m.centre.x, 3.0, 1e-6);
    CHECK_NEAR(m.centre.y, 1.25, 1e-6);
}

TEST_CASE("min_area_rect: rotated rectangle recovers true metric size") {
    // A 5.0 x 2.0 rectangle rotated 30 degrees. A naive axis-aligned bounding box
    // would report roughly 5.33 x 4.23 -- badly wrong for a fit decision.
    const double ang = 30.0 * kPi / 180.0;
    const double ca = std::cos(ang);
    const double sa = std::sin(ang);
    Ring r;
    const double pts[4][2] = {{0.0, 0.0}, {5.0, 0.0}, {5.0, 2.0}, {0.0, 2.0}};
    for (const auto& p : pts) r.push_back(Point2{p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca});

    const MinAreaRect m = min_area_rect(r);
    CHECK_NEAR(m.length_m, 5.0, 1e-6);
    CHECK_NEAR(m.width_m, 2.0, 1e-6);
    CHECK_NEAR(std::fabs(m.angle_rad), ang, 1e-6);
}

TEST_CASE("min_area_rect: real Amsterdam parallel bay measures like a parking space") {
    // Verbatim from api.data.amsterdam.nl parkeervakken id 110675492544, a "Langs"
    // (kerb-parallel) bay on Abidjanweg. Coordinates are RD New metres.
    const Ring bay{{110677.64, 492542.17},
                   {110672.28, 492543.33},
                   {110670.76, 492545.68},
                   {110678.06, 492544.12}};
    const MinAreaRect m = min_area_rect(bay);
    // A Dutch kerb-parallel bay is about 5.0-6.5 m long and 1.8-2.5 m wide.
    CHECK(m.length_m > 4.5);
    CHECK(m.length_m < 8.0);
    CHECK(m.width_m > 1.2);
    CHECK(m.width_m < 3.5);
    CHECK(m.length_m > m.width_m);
}

TEST_CASE("convex_hull: interior points are discarded") {
    const Ring pts{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}, {5.0, 5.0}, {3.0, 4.0}};
    const Ring h = convex_hull(pts);
    CHECK_EQ(h.size(), static_cast<std::size_t>(4));
    CHECK_NEAR(area(h), 100.0, 1e-9);
}

TEST_CASE("degenerate polygons do not crash") {
    CHECK_NEAR(area(Ring{}), 0.0, 1e-12);
    CHECK_NEAR(area(Ring{{1.0, 1.0}}), 0.0, 1e-12);
    CHECK(!contains(Ring{{1.0, 1.0}}, Point2{1.0, 1.0}));
    const MinAreaRect m = min_area_rect(Ring{{0.0, 0.0}, {3.0, 4.0}});
    CHECK_NEAR(m.length_m, 5.0, 1e-9);
    CHECK_NEAR(m.width_m, 0.0, 1e-9);
}

PF_TEST_MAIN()
