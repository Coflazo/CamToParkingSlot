"""The kerb-gap finder exists twice, and the two must agree.

The geometry runs in C++ because it is on a two-second loop per watched camera and its
occlusion guard is quadratic in the detection count. The Python version stays as the
reference and as the answer on an uncompiled checkout, which only works while they give
the same answer.

Three of these are regressions from real frames rather than invented cases. Each is a
claim the finder made before its guard existed, and each was found by looking at rendered
output rather than by reading code:

* a confident 22.4 m "space" running straight across two dozen parked cars at Kijkduin,
  where a perpendicular car park broke the one-depth assumption the kerb band relies on;
* 37.6 m of open Groningen road offered as parking;
* a stretch of pavement flanked by two bicycles.
"""

from __future__ import annotations

import pytest

from parkfit.native import native
from parkfit.services.camera_analysis import (
    DetectedVehicle,
    _find_free_spaces_python,
    find_free_spaces,
)

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(native is None, reason="parkfit_native is not built"),
]

#: 40 px per metre, so a 1.80 m car is 72 px wide and the arithmetic stays readable.
PPM = 40.0


def car(x1: float, y1: float, width: float = 72.0, height: float = 60.0, label: str = "car"):
    return DetectedVehicle(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height, label=label, score=0.9)


def kerb_row(*offsets: float, y: float = 300.0) -> list[DetectedVehicle]:
    return [car(x, y) for x in offsets]


def assert_same(vehicles: list[DetectedVehicle], ppm: float = PPM) -> list:
    """Both implementations, compared field for field, and the shared answer returned."""
    cpp = find_free_spaces(vehicles, ppm, 1280)
    py = _find_free_spaces_python(vehicles, ppm, 1280)

    assert len(cpp) == len(py), f"C++ found {len(cpp)} gaps, Python found {len(py)}"
    for a, b in zip(cpp, py, strict=True):
        assert a.x1 == pytest.approx(b.x1, abs=1e-9)
        assert a.x2 == pytest.approx(b.x2, abs=1e-9)
        assert a.y1 == pytest.approx(b.y1, abs=1e-9)
        assert a.y2 == pytest.approx(b.y2, abs=1e-9)
        assert a.length_m == pytest.approx(b.length_m, abs=1e-9)
        assert a.depth_m == pytest.approx(b.depth_m, abs=1e-9)
        assert a.fits == b.fits
    return cpp


# ---------------------------------------------------------------- agreement

def test_an_ordinary_gap_is_found_identically():
    # Two cars with 240 px between them: 6.0 m, comfortably a space.
    spaces = assert_same(kerb_row(100.0, 412.0))
    assert len(spaces) == 1
    assert spaces[0].length_m == pytest.approx(6.0, abs=0.05)
    assert spaces[0].fits  # something in the fleet clears 6 m


def test_a_row_with_several_gaps_agrees_on_all_of_them():
    assert_same(kerb_row(100.0, 412.0, 700.0, 1000.0))


def test_no_gap_where_the_cars_are_touching():
    assert assert_same(kerb_row(100.0, 174.0)) == []


def test_a_single_car_yields_nothing():
    assert assert_same(kerb_row(100.0)) == []


def test_an_empty_frame_yields_nothing():
    assert assert_same([]) == []


def test_an_unusable_scale_yields_nothing():
    assert assert_same(kerb_row(100.0, 412.0), ppm=0.0) == []


# --------------------------------------------------------------- the guards

def test_a_gap_shorter_than_the_smallest_car_is_refused():
    # 120 px is 3.0 m, under the 4.2 m floor.
    assert assert_same(kerb_row(100.0, 292.0)) == []


def test_open_road_is_not_a_parking_space():
    """The Groningen case: 37.6 m of empty road reported as parking.

    Two cars a long way apart are not a parking space with room for eight cars; they are
    two cars with a road between them, and the far end is usually a junction.
    """
    # 1600 px is 40 m, well past the 15 m ceiling.
    assert assert_same(kerb_row(100.0, 1772.0)) == []


def test_a_gap_running_across_parked_cars_is_refused():
    """The Kijkduin case: a confident 22.4 m space across two dozen parked cars.

    A perpendicular car park puts cars at many depths. The kerb band keeps a scattered
    subset at one depth, and the naive gap between two of those runs straight over
    everything in between. The occlusion scan checks the full detection list rather than
    the band, because the vehicles in the way are exactly the ones the band discarded.
    """
    vehicles = kerb_row(100.0, 412.0)
    # A car sitting in the middle of the gap, at a different depth so the band drops it.
    vehicles.append(car(250.0, 250.0))
    assert assert_same(vehicles) == []

    # Remove the obstruction and the same pair is a space again, which shows the guard
    # is discriminating rather than simply suppressing.
    assert len(assert_same(kerb_row(100.0, 412.0))) == 1


def test_bicycles_cannot_flank_a_parking_space():
    """The pavement case. Two bicycles with a stretch between them is a pavement."""
    bikes = [
        DetectedVehicle(x1=100.0, y1=300.0, x2=140.0, y2=360.0, label="bicycle", score=0.9),
        DetectedVehicle(x1=412.0, y1=300.0, x2=452.0, y2=360.0, label="bicycle", score=0.9),
    ]
    assert assert_same(bikes) == []


def test_a_space_too_shallow_to_hold_a_car_is_refused():
    """Depth comes from the flanking cars, and a car needs more than a stripe of it."""
    shallow = [car(100.0, 300.0, height=20.0), car(412.0, 300.0, height=20.0)]
    # 20 px at 40 px/m is 0.5 m, under the 1.2 m floor.
    assert assert_same(shallow) == []


def test_vans_and_trucks_may_flank_a_space_but_do_not_set_the_scale():
    """A lorry is a vehicle and a terrible ruler.

    It can stand at the side of a gap, so it is a flanking class, but the pixels-per-metre
    estimate is taken from cars only. Both implementations have to make that same split.
    """
    mixed = [
        car(100.0, 300.0, width=110.0, label="truck"),
        car(450.0, 300.0, label="car"),
    ]
    assert_same(mixed)

    boxes = [(v.x1, v.y1, v.x2, v.y2, True, v.label == "car") for v in mixed]
    scale = native.estimate_gap_scale(boxes)
    # One car only, so the scale exists but is not called confident.
    assert scale.usable()
    assert not scale.confident


# ---------------------------------------------------- the shared primitives

def test_the_scale_is_the_median_car_width_not_the_mean():
    """One box that swallowed two cars must not move every distance in the frame."""
    widths = [70.0, 72.0, 74.0, 400.0]  # the last is a merged detection
    boxes = [(0.0, 0.0, w, 60.0, True, True) for w in widths]
    scale = native.estimate_gap_scale(boxes)
    # Median of the four is 74; a mean would be 154 and halve every reported length.
    assert scale.pixels_per_metre == pytest.approx(74.0 / 1.80, abs=1e-9)
    assert scale.confident


def test_the_kerb_band_needs_at_least_two_vehicles():
    assert native.kerb_band([(0.0, 0.0, 72.0, 60.0, True, True)]) is None
    band = native.kerb_band([(0.0, 0.0, 72.0, 60.0, True, True), (100.0, 0.0, 172.0, 60.0, True, True)])
    assert band is not None
    low, high = band
    assert low < 30.0 < high  # both centres are at y=30


def test_the_band_tolerance_has_a_floor_so_distant_cars_still_form_a_row():
    """Small far-away cars would otherwise collapse the band to nearly a line."""
    tiny = [(0.0, 100.0, 8.0, 104.0, True, True), (20.0, 100.0, 28.0, 104.0, True, True)]
    low, high = native.kerb_band(tiny)
    assert high - low >= 24.0  # 2 x the 12 px floor


# ------------------------------------------------------- the randomised sweep

def test_a_gap_running_off_the_frame_edge_is_refused():
    """Its far end is outside the picture, so its length is a lower bound.

    This was a real porting bug. The Python version ends with a frame-edge filter and the
    first C++ port did not have one, so the two disagreed on any frame with a gap near
    the right edge. Every hand-written case above happened to sit well inside the frame,
    which is exactly why hand-written cases are not enough.
    """
    # A gap ending at x=1278 in a 1280-wide frame: clipped.
    assert assert_same(kerb_row(1000.0, 1278.0), ppm=PPM) == []
    # The same pair in a wider frame is a real space.
    cpp = find_free_spaces(kerb_row(1000.0, 1278.0), PPM, 1600)
    py = _find_free_spaces_python(kerb_row(1000.0, 1278.0), PPM, 1600)
    assert len(cpp) == len(py) == 1


@pytest.mark.parametrize("seed", range(40))
def test_random_frames_agree(seed):
    """Forty pseudo-random street frames, compared field for field.

    Hand-written cases test what the author already thought of. This tests what they did
    not: it is what caught the missing frame-edge filter, on a frame no one would have
    written by hand.
    """
    import random

    rng = random.Random(seed)
    frame_width = rng.choice([640, 1280, 1920])
    vehicles: list[DetectedVehicle] = []

    # A kerb row with irregular spacing.
    x = rng.uniform(0.0, 120.0)
    while x < frame_width + 200.0:
        width = rng.uniform(40.0, 120.0)
        height = rng.uniform(20.0, 90.0)
        label = rng.choice(["car", "car", "car", "van", "truck", "bicycle"])
        vehicles.append(
            DetectedVehicle(x1=x, y1=300.0, x2=x + width, y2=300.0 + height, label=label, score=0.9)
        )
        x += width + rng.choice([5.0, 30.0, 90.0, 200.0, 400.0, 900.0])

    # Scattered traffic at other depths, which is what breaks the one-depth assumption.
    for _ in range(rng.randint(0, 12)):
        vx = rng.uniform(0.0, frame_width)
        vy = rng.uniform(120.0, 380.0)
        vehicles.append(
            DetectedVehicle(
                x1=vx, y1=vy, x2=vx + rng.uniform(30.0, 100.0), y2=vy + rng.uniform(20.0, 80.0),
                label=rng.choice(["car", "van", "bus", "motorcycle"]), score=0.7,
            )
        )

    ppm = rng.choice([20.0, 40.0, 63.5])
    cpp = find_free_spaces(vehicles, ppm, frame_width)
    py = _find_free_spaces_python(vehicles, ppm, frame_width)

    assert len(cpp) == len(py), (
        f"seed {seed}: C++ {len(cpp)} gaps, Python {len(py)} "
        f"({len(vehicles)} detections, frame {frame_width}, {ppm} px/m)"
    )
    for a, b in zip(cpp, py, strict=True):
        assert a.x1 == pytest.approx(b.x1, abs=1e-9)
        assert a.x2 == pytest.approx(b.x2, abs=1e-9)
        assert a.length_m == pytest.approx(b.length_m, abs=1e-9)
        assert a.depth_m == pytest.approx(b.depth_m, abs=1e-9)
        assert a.fits == b.fits
