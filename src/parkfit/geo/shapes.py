"""Planar geometry on RD-metre rings.

:func:`min_area_rect` is the function that turns a raw Amsterdam bay polygon into the
two numbers the fit engine needs. It mirrors ``parkfit::geo::min_area_rect`` in the C++
core, and ``tests/contract/test_geometry_parity.py`` asserts the two agree, so the same
bay produces the same verdict whichever side of the boundary computes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Ring = list[list[float]]


@dataclass(frozen=True)
class RectMeasurement:
    """Minimum-area enclosing rectangle of a bay polygon, in metres."""

    length_m: float
    width_m: float
    angle_rad: float
    centre_x: float
    centre_y: float

    @property
    def length_cm(self) -> float:
        return self.length_m * 100.0

    @property
    def width_cm(self) -> float:
        return self.width_m * 100.0


def _cross(o: list[float], a: list[float], b: list[float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Ring) -> Ring:
    """Monotone-chain convex hull, counter-clockwise, no repeated endpoint."""
    pts = sorted({(round(p[0], 9), round(p[1], 9)) for p in points})
    if len(pts) < 3:
        return [list(p) for p in pts]

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(list(lower[-2]), list(lower[-1]), list(p)) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(list(upper[-2]), list(upper[-1]), list(p)) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def min_area_rect(ring: Ring) -> RectMeasurement:
    """Rotating-calipers minimum-area enclosing rectangle.

    By the Freeman-Shapira theorem this rectangle shares an edge with the convex hull,
    so trying one orientation per hull edge is exhaustive rather than a sampled guess.

    Using an axis-aligned bounding box instead would be badly wrong for any bay that is
    not aligned to the RD grid, and almost no Amsterdam street is. A 5.0 x 2.0 m bay
    rotated 30 degrees measures 5.33 x 4.23 m axis-aligned, which would pass a van into
    a space that cannot hold it.
    """
    hull = convex_hull(ring)
    if len(hull) < 3:
        if len(hull) == 2:
            dx = hull[1][0] - hull[0][0]
            dy = hull[1][1] - hull[0][1]
            return RectMeasurement(
                length_m=math.hypot(dx, dy),
                width_m=0.0,
                angle_rad=math.atan2(dy, dx),
                centre_x=(hull[0][0] + hull[1][0]) / 2,
                centre_y=(hull[0][1] + hull[1][1]) / 2,
            )
        if len(hull) == 1:
            return RectMeasurement(0.0, 0.0, 0.0, hull[0][0], hull[0][1])
        return RectMeasurement(0.0, 0.0, 0.0, 0.0, 0.0)

    best: RectMeasurement | None = None
    best_area = float("inf")
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        elen = math.hypot(ex, ey)
        if elen < 1e-9:
            continue
        ux, uy = ex / elen, ey / elen
        vx, vy = -uy, ux

        us = [(p[0] - ax) * ux + (p[1] - ay) * uy for p in hull]
        vs = [(p[0] - ax) * vx + (p[1] - ay) * vy for p in hull]
        su = max(us) - min(us)
        sv = max(vs) - min(vs)
        area = su * sv
        if area >= best_area:
            continue

        best_area = area
        cu = (min(us) + max(us)) / 2
        cv = (min(vs) + max(vs)) / 2
        best = RectMeasurement(
            length_m=max(su, sv),
            width_m=min(su, sv),
            angle_rad=math.atan2(uy, ux) if su >= sv else math.atan2(vy, vx),
            centre_x=ax + ux * cu + vx * cv,
            centre_y=ay + uy * cu + vy * cv,
        )
    return best or RectMeasurement(0.0, 0.0, 0.0, 0.0, 0.0)


def ring_area(ring: Ring) -> float:
    """Absolute area via the shoelace formula, in square metres."""
    n = len(ring)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        s += x0 * y1 - x1 * y0
    return abs(s) * 0.5


def polyline_length(points: Ring) -> float:
    """Total length of an open polyline, in metres."""
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


@dataclass(frozen=True)
class BayMeasurement:
    """Conservative usable dimensions of a bay polygon.

    The minimum-area rectangle is the smallest rectangle that *encloses* the polygon,
    which makes it the wrong number for a fit decision. Amsterdam bays drawn against a
    curving kerb are trapezoids: one real Abidjanweg bay has long sides of 5.48 m and
    7.46 m, and the enclosing rectangle reports 7.46, claiming two metres of kerb that
    do not exist, in the optimistic direction, for the number that decides whether a car
    fits.

    ``area / extent`` instead yields the mean of the parallel sides. It is exact for a
    true rectangle and conservative for a trapezoid, which is the direction an error
    has to point when the consequence is a driver arriving at a space too small.

    ``fill_ratio`` reports how rectangular the bay actually is. A low ratio means the
    two measures disagree substantially and the fit verdict deserves less confidence.
    """

    length_m: float
    width_m: float
    max_length_m: float
    max_width_m: float
    angle_rad: float
    fill_ratio: float
    centre_x: float
    centre_y: float

    @property
    def length_cm(self) -> float:
        return self.length_m * 100.0

    @property
    def width_cm(self) -> float:
        return self.width_m * 100.0


def measure_bay(ring: Ring) -> BayMeasurement:
    """Measure a bay polygon by its own geometry, in metres, RD frame.

    Bay polygons are overwhelmingly quadrilaterals, and frequently *skewed* ones:
    Amsterdam canal-side parking is drawn as angled parallelograms. For those, an
    enclosing rectangle is the wrong tool entirely. A real Prinsengracht bay has sides
    of 5.66 m and 2.61 m at 48 degrees; its minimum-area enclosing rectangle is
    7.40 x 1.89 m, which is a box rotated to hug a diagonal and matches neither
    dimension of the actual bay. Measured that way, cars "did not fit" spaces they park
    in every day.

    So a quadrilateral is measured by its own edges: pair the opposite sides, take the
    mean of the longer pair as the length, and derive the width as ``area / length``.
    That width is the polygon perpendicular height, which is exactly the clearance a
    car occupies, 10.98 / 5.66 = 1.94 m for that Prinsengracht bay, ample for a
    1.75 m car. The result is exact for rectangles and parallelograms, and conservative
    for trapezoids: the Abidjanweg bay measures 6.47 x 1.99 m rather than the enclosing
    box 7.46 x 1.72 m.

    Anything that is not a quadrilateral falls back to the enclosing-rectangle estimate,
    still corrected by area so it cannot overstate.
    """
    rect = min_area_rect(ring)
    area = ring_area(ring)

    pts = _dedupe_closing_point(ring)
    length_m = width_m = 0.0
    angle_rad = rect.angle_rad

    if len(pts) == 4 and area > 1e-9:
        edges = [
            (
                math.hypot(pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]),
                math.atan2(pts[(i + 1) % 4][1] - pts[i][1], pts[(i + 1) % 4][0] - pts[i][0]),
            )
            for i in range(4)
        ]
        pair_a = (edges[0][0] + edges[2][0]) / 2.0
        pair_b = (edges[1][0] + edges[3][0]) / 2.0
        if pair_a >= pair_b:
            length_m, angle_rad = pair_a, edges[0][1]
        else:
            length_m, angle_rad = pair_b, edges[1][1]
        if length_m > 1e-9:
            width_m = area / length_m

    if length_m <= 1e-9 or width_m <= 1e-9:
        rect_area = rect.length_m * rect.width_m
        if rect_area <= 1e-9 or area <= 1e-9:
            return BayMeasurement(
                length_m=rect.length_m,
                width_m=rect.width_m,
                max_length_m=rect.length_m,
                max_width_m=rect.width_m,
                angle_rad=rect.angle_rad,
                fill_ratio=0.0,
                centre_x=rect.centre_x,
                centre_y=rect.centre_y,
            )
        length_m = min(area / rect.width_m if rect.width_m > 1e-9 else rect.length_m, rect.length_m)
        width_m = min(area / rect.length_m if rect.length_m > 1e-9 else rect.width_m, rect.width_m)

    if width_m > length_m:
        length_m, width_m = width_m, length_m

    rect_area = rect.length_m * rect.width_m
    return BayMeasurement(
        length_m=length_m,
        width_m=width_m,
        max_length_m=rect.length_m,
        max_width_m=rect.width_m,
        angle_rad=angle_rad,
        fill_ratio=min(1.0, area / rect_area) if rect_area > 1e-9 else 0.0,
        centre_x=rect.centre_x,
        centre_y=rect.centre_y,
    )


def _dedupe_closing_point(ring: Ring) -> Ring:
    """Drop a repeated final vertex, which GeoJSON rings carry by convention."""
    if (
        len(ring) >= 2
        and abs(ring[0][0] - ring[-1][0]) < 1e-9
        and abs(ring[0][1] - ring[-1][1]) < 1e-9
    ):
        return ring[:-1]
    return ring
