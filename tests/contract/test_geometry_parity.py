"""``parkfit.geo.shapes`` and ``parkfit::geo`` must measure a bay the same way.

``src/parkfit/geo/shapes.py`` says in its own docstring that it is a port of
``cpp/core/include/parkfit/geo/polygon.hpp`` and points at this file. Until now the file
did not exist, and the two could have drifted silently. That matters more here than
almost anywhere else in the codebase: the number these functions return is the number
that decides whether a driver is told their car fits, and the two implementations run in
different places, Python during ingest and C++ during search.

The bays below are real Amsterdam shapes, not random polygons, because the interesting
cases are the ones that broke the enclosing-rectangle approach: skewed canal-side
parallelograms and trapezoids drawn against a curving kerb.
"""

from __future__ import annotations

import math

import pytest

from parkfit.geo.shapes import measure_bay as measure_bay_python
from parkfit.geo.shapes import ring_area as ring_area_python
from parkfit.native import native

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(native is None, reason="parkfit_native is not built"),
]

#: A skewed canal-side bay: sides of 5.66 m and 2.61 m at roughly 48 degrees. Its
#: minimum-area enclosing rectangle is 7.40 x 1.89 m, which matches neither real
#: dimension, and measuring it that way is what made cars "not fit" spaces they park in
#: every day.
PRINSENGRACHT = [
    (121000.00, 487000.00),
    (121003.79, 487004.21),
    (121005.53, 487002.26),
    (121001.74, 486998.05),
    (121000.00, 487000.00),
]

#: A trapezoid against a curving kerb: long sides of unequal length, so the enclosing
#: rectangle overstates the usable length in the optimistic direction.
ABIDJANWEG = [
    (122000.00, 488000.00),
    (122007.46, 488000.00),
    (122005.48, 488002.00),
    (122000.00, 488002.00),
    (122000.00, 488000.00),
]

#: The easy case, kept because a parity suite that only covers hard cases will not notice
#: an implementation that gets the easy one wrong.
RECTANGLE = [
    (123000.00, 489000.00),
    (123005.00, 489000.00),
    (123005.00, 489002.20),
    (123000.00, 489002.20),
    (123000.00, 489000.00),
]

#: Not a quadrilateral, so both implementations must fall back to the enclosing-rectangle
#: estimate, still corrected by area.
PENTAGON = [
    (124000.00, 490000.00),
    (124006.00, 490000.00),
    (124006.50, 490001.20),
    (124003.00, 490002.40),
    (124000.00, 490002.00),
    (124000.00, 490000.00),
]

#: Wound the other way. Signed area flips sign; the measurement must not.
RECTANGLE_CLOCKWISE = list(reversed(RECTANGLE))

ALL_BAYS = {
    "prinsengracht": PRINSENGRACHT,
    "abidjanweg": ABIDJANWEG,
    "rectangle": RECTANGLE,
    "pentagon": PENTAGON,
    "rectangle_clockwise": RECTANGLE_CLOCKWISE,
}


@pytest.mark.parametrize("name", sorted(ALL_BAYS))
def test_measure_bay_agrees(name):
    ring = ALL_BAYS[name]
    py = measure_bay_python(ring)
    cpp = native.measure_bay(ring)

    assert py.length_m == pytest.approx(cpp.length_m, abs=1e-9)
    assert py.width_m == pytest.approx(cpp.width_m, abs=1e-9)
    assert py.max_length_m == pytest.approx(cpp.max_length_m, abs=1e-9)
    assert py.max_width_m == pytest.approx(cpp.max_width_m, abs=1e-9)
    assert py.fill_ratio == pytest.approx(cpp.fill_ratio, abs=1e-9)
    # Angle is modulo a half turn: a rectangle at 0 and one at pi describe the same box,
    # so compare the orientation rather than the raw number.
    assert math.sin(2 * py.angle_rad) == pytest.approx(math.sin(2 * cpp.angle_rad), abs=1e-9)
    assert math.cos(2 * py.angle_rad) == pytest.approx(math.cos(2 * cpp.angle_rad), abs=1e-9)


@pytest.mark.parametrize("name", sorted(ALL_BAYS))
def test_ring_area_agrees(name):
    ring = ALL_BAYS[name]
    assert ring_area_python(ring) == pytest.approx(native.ring_area(ring), abs=1e-9)


def test_the_skewed_bay_is_not_measured_as_its_enclosing_box():
    """The regression that motivated the edge-pair measurement in the first place.

    If either implementation ever reverts to the enclosing rectangle, the length jumps to
    about 7.4 m and the width drops to about 1.9 m, and this fails in both.
    """
    for measure in (measure_bay_python, native.measure_bay):
        bay = measure(PRINSENGRACHT)
        assert 5.4 < bay.length_m < 5.9
        assert bay.width_m > 1.9


def test_the_trapezoid_is_measured_conservatively():
    """A fit error has to point towards "too small", never "big enough"."""
    for measure in (measure_bay_python, native.measure_bay):
        bay = measure(ABIDJANWEG)
        # The long side is 7.46 m; the honest usable length is the mean of the parallel
        # pair, which is shorter. Claiming the long side would invent kerb.
        assert bay.length_m < 7.46
        assert bay.length_m <= bay.max_length_m


def test_winding_direction_does_not_change_the_measurement():
    counter_clockwise = native.measure_bay(RECTANGLE)
    clockwise = native.measure_bay(RECTANGLE_CLOCKWISE)
    assert counter_clockwise.length_m == pytest.approx(clockwise.length_m, abs=1e-9)
    assert counter_clockwise.width_m == pytest.approx(clockwise.width_m, abs=1e-9)


def test_the_ingest_module_picks_the_native_implementation_when_it_is_built():
    """The switch itself is worth a test: it is one line and easy to lose in a refactor."""
    from parkfit.ingest import amsterdam

    assert amsterdam.measure_bay is native.measure_bay
