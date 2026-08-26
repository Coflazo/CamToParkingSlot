"""Geometry and coordinate tests.

The RD transform sits on the critical path of the entire product: every Amsterdam bay
is published in it, and every bay dimension is derived from it. These cases use real
polygons from the live API and check against ``pyproj``'s rigorous pipeline, so a
regression shows up as a wrong measurement rather than as a plausible-looking number.
"""

from __future__ import annotations

import math

import pytest

from parkfit.geo.rd import (
    haversine_m,
    rd_in_range,
    rd_to_wgs84,
    rd_to_wgs84_approx,
    ring_centroid_rd,
    wgs84_to_rd,
    wgs84_to_rd_approx,
)
from parkfit.geo.shapes import (
    convex_hull,
    measure_bay,
    min_area_rect,
    polyline_length,
    ring_area,
)

# Points spanning the country, so an error that only shows at the edges is caught.
RD_SAMPLES = [
    ("Amersfoort origin", 155000.0, 463000.0),
    ("Amsterdam bay", 110677.64, 492542.17),
    ("Amsterdam centre", 121000.0, 487000.0),
    ("Groningen", 233000.0, 582000.0),
    ("Maastricht", 176000.0, 318000.0),
    ("Vlissingen", 30000.0, 385000.0),
]


class TestRdTransform:
    def test_origin_maps_to_the_amersfoort_reference_point(self):
        """The RD origin is the Onze Lieve Vrouwetoren in Amersfoort.

        The two transforms disagree here by exactly the offset documented elsewhere in
        this file: the Kadaster approximation returns its own reference constant by
        construction, while the rigorous pipeline places the tower 23 cm away. Asserting
        the constant against pyproj would be asserting that the offset does not exist.
        """
        approx_lat, approx_lon = rd_to_wgs84_approx(155000.0, 463000.0)
        assert approx_lat == pytest.approx(52.15517440, abs=1e-9)
        assert approx_lon == pytest.approx(5.38720621, abs=1e-9)

        exact_lat, exact_lon = rd_to_wgs84(155000.0, 463000.0)
        assert exact_lat == pytest.approx(52.1551723, abs=1e-6)
        assert exact_lon == pytest.approx(5.3872035, abs=1e-6)
        assert haversine_m(approx_lat, approx_lon, exact_lat, exact_lon) < 0.5

    @pytest.mark.parametrize(("name", "x", "y"), RD_SAMPLES)
    def test_round_trip_is_stable_to_centimetres(self, name, x, y):
        lat, lon = rd_to_wgs84(x, y)
        back_x, back_y = wgs84_to_rd(lat, lon)
        assert back_x == pytest.approx(x, abs=0.02), name
        assert back_y == pytest.approx(y, abs=0.02), name

    @pytest.mark.parametrize(("name", "x", "y"), RD_SAMPLES)
    def test_kadaster_approximation_tracks_the_rigorous_pipeline(self, name, x, y):
        """The polynomial the C++ core uses must agree with pyproj to well under a metre.

        The residual is a known systematic offset of roughly 0.23 m north and 0.18 m
        east, near-constant across the country. It cancels out of any length measurement,
        which is why bay dimensions are computed in RD and never in degrees.
        """
        exact_lat, exact_lon = rd_to_wgs84(x, y)
        approx_lat, approx_lon = rd_to_wgs84_approx(x, y)
        separation = haversine_m(exact_lat, exact_lon, approx_lat, approx_lon)
        assert separation < 0.5, f"{name}: {separation:.3f} m apart"

    def test_approximation_round_trips_to_itself(self):
        for _name, x, y in RD_SAMPLES:
            lat, lon = rd_to_wgs84_approx(x, y)
            back_x, back_y = wgs84_to_rd_approx(lat, lon)
            assert back_x == pytest.approx(x, abs=0.05)
            assert back_y == pytest.approx(y, abs=0.05)

    def test_range_check_rejects_coordinates_outside_the_netherlands(self):
        assert rd_in_range(155000.0, 463000.0)
        assert not rd_in_range(-50000.0, 463000.0)
        assert not rd_in_range(155000.0, 100000.0)


class TestHaversine:
    def test_known_city_separation(self):
        # Spherical great-circle Amsterdam to Rotterdam. The WGS84 geodesic is 57305.6 m,
        # so the sphere is short by 76 m over 57 km: 0.13 %, and a couple of metres over
        # the radii this product searches.
        assert haversine_m(52.3676, 4.9041, 51.9244, 4.4777) == pytest.approx(57229.3, abs=5.0)

    def test_zero_distance(self):
        assert haversine_m(52.3676, 4.9041, 52.3676, 4.9041) == pytest.approx(0.0, abs=1e-9)

    def test_symmetry(self):
        a = haversine_m(52.3676, 4.9041, 52.36, 4.88)
        b = haversine_m(52.36, 4.88, 52.3676, 4.9041)
        assert a == pytest.approx(b, abs=1e-9)


class TestBayMeasurement:
    def test_true_rectangle_measures_exactly(self):
        m = measure_bay([[0, 0], [6, 0], [6, 2.5], [0, 2.5]])
        assert m.length_m == pytest.approx(6.0, abs=1e-9)
        assert m.width_m == pytest.approx(2.5, abs=1e-9)
        assert m.fill_ratio == pytest.approx(1.0, abs=1e-9)

    def test_rotated_rectangle_recovers_true_size(self):
        """An axis-aligned bounding box would report 5.33 x 4.23 for this."""
        angle = math.radians(30.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        ring = [
            [x * cos_a - y * sin_a, x * sin_a + y * cos_a]
            for x, y in ((0, 0), (5, 0), (5, 2), (0, 2))
        ]
        m = measure_bay(ring)
        assert m.length_m == pytest.approx(5.0, abs=1e-6)
        assert m.width_m == pytest.approx(2.0, abs=1e-6)

    def test_skewed_parallelogram_uses_its_own_geometry(self, prinsengracht_ring):
        """The case that made ordinary cars 'not fit' ordinary Amsterdam canal bays.

        This bay has sides of 5.66 m and 2.61 m at 48 degrees. Its minimum-area
        *enclosing* rectangle is 7.40 x 1.89 m, which is a box rotated to hug a diagonal
        and matches neither dimension. The usable width is the perpendicular height:
        area divided by the long side.
        """
        m = measure_bay(prinsengracht_ring)
        assert m.length_m == pytest.approx(5.66, abs=0.02)
        assert m.width_m == pytest.approx(1.94, abs=0.02)
        # The enclosing rectangle is recorded too, and is visibly different.
        assert m.max_length_m == pytest.approx(7.40, abs=0.05)
        assert m.max_length_m > m.length_m + 1.5

    def test_trapezoid_is_measured_conservatively(self, amsterdam_bay_ring):
        """A real Abidjanweg bay: long sides of 5.48 m and 7.46 m.

        The enclosing rectangle takes the maximum extent, which overstates the bay by
        two metres in the optimistic direction, for the number that decides whether a
        car fits. The mean of the parallel sides is the honest answer.
        """
        m = measure_bay(amsterdam_bay_ring)
        rect = min_area_rect(amsterdam_bay_ring)
        assert m.length_m == pytest.approx(6.47, abs=0.05)
        assert m.length_m < rect.length_m
        assert m.width_m == pytest.approx(1.99, abs=0.05)

    def test_measured_dimensions_never_exceed_the_enclosing_rectangle(self, amsterdam_bay_ring):
        m = measure_bay(amsterdam_bay_ring)
        assert m.length_m <= m.max_length_m + 1e-9
        assert m.width_m <= m.max_width_m + 1e-9

    def test_degenerate_rings_do_not_raise(self):
        for ring in ([], [[0, 0]], [[0, 0], [1, 1]]):
            m = measure_bay(ring)
            assert m.length_m >= 0.0
            assert m.width_m >= 0.0

    def test_closing_vertex_is_ignored(self):
        """GeoJSON rings repeat the first point; that must not change the measurement."""
        open_ring = [[0, 0], [6, 0], [6, 2.5], [0, 2.5]]
        closed_ring = [*open_ring, [0, 0]]
        assert measure_bay(closed_ring).length_m == pytest.approx(
            measure_bay(open_ring).length_m, abs=1e-9
        )


class TestPlanarHelpers:
    def test_ring_area_shoelace(self):
        assert ring_area([[0, 0], [10, 0], [10, 4], [0, 4]]) == pytest.approx(40.0)

    def test_centroid_of_a_rectangle(self):
        cx, cy = ring_centroid_rd([[0, 0], [10, 0], [10, 4], [0, 4]])
        assert (cx, cy) == pytest.approx((5.0, 2.0))

    def test_convex_hull_drops_interior_points(self):
        hull = convex_hull([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [3, 4]])
        assert len(hull) == 4
        assert ring_area(hull) == pytest.approx(100.0)

    def test_polyline_length(self):
        assert polyline_length([[0, 0], [3, 4], [3, 8]]) == pytest.approx(9.0)


@pytest.mark.native
class TestNativeParity:
    """The C++ core and the Python fallback must agree, or a bay's verdict depends on
    whether the project happened to be compiled."""

    def test_bay_measurement_is_identical(self, amsterdam_bay_ring, prinsengracht_ring):
        from parkfit.native import native

        if native is None:
            pytest.skip("native module not built")
        for ring in (amsterdam_bay_ring, prinsengracht_ring, [[0, 0], [6, 0], [6, 2.5], [0, 2.5]]):
            py = measure_bay(ring)
            cpp = native.measure_bay([(p[0], p[1]) for p in ring])
            assert cpp.length_m == pytest.approx(py.length_m, abs=1e-9)
            assert cpp.width_m == pytest.approx(py.width_m, abs=1e-9)
            assert cpp.fill_ratio == pytest.approx(py.fill_ratio, abs=1e-9)

    def test_rd_transform_agrees_within_the_documented_offset(self):
        from parkfit.native import native

        if native is None:
            pytest.skip("native module not built")
        for _name, x, y in RD_SAMPLES:
            cpp_lat, cpp_lon = native.rd_to_wgs84(x, y)
            py_lat, py_lon = rd_to_wgs84(x, y)
            assert haversine_m(cpp_lat, cpp_lon, py_lat, py_lon) < 0.5

    def test_haversine_agrees_exactly(self):
        from parkfit.native import native

        if native is None:
            pytest.skip("native module not built")
        assert native.haversine_m(52.3676, 4.9041, 51.9244, 4.4777) == pytest.approx(
            haversine_m(52.3676, 4.9041, 51.9244, 4.4777), abs=1e-6
        )
