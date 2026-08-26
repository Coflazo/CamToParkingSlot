"""The vehicle detector: dataset construction, target encoding, decoding, export.

The test that earns its place here is the encode/decode round trip. Every other number
this pipeline produces is measured *through* those two functions, so if they disagree the
model can be trained perfectly and score badly for reasons no amount of staring at the
loss curve will explain.

The C++ decoder in ``cpp/vision/src/onnx_detector.cpp`` has to match
:func:`parkfit.ml.train.detector.decode` exactly. Both are tested against hand-built
tensors with arithmetic answers rather than against each other, so neither can drift into
agreeing on something wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from parkfit.ml.datasets import scenes
from parkfit.ml.train import detector


def _box(x1, y1, x2, y2, cls=0):
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "class": cls,
        "label": scenes.CLASS_NAMES[cls],
    }


# ---------------------------------------------------------------------------
# the contract with C++
# ---------------------------------------------------------------------------
def test_class_order_matches_the_cpp_enum():
    """Channel i of the heatmap is ``VehicleClass(i)`` on the other side of ONNX.

    ``parkfit::vision::VehicleClass`` declares Car, Van, Truck, Bus, Motorcycle, Bicycle,
    Trailer, Unknown. Unknown is not predicted, so the first seven are the model's
    channels, in that order. Reordering this tuple silently relabels every detection the
    worker ever publishes.
    """
    assert scenes.CLASS_NAMES == (
        "car",
        "van",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "trailer",
    )


def test_every_renderer_body_style_maps_to_a_class():
    """The renderer names body styles; an unmapped one would be dropped silently."""
    from parkfit.ml.synthetic.scene import VEHICLE_TYPES

    for kind in VEHICLE_TYPES:
        assert kind in scenes.KIND_TO_CLASS, f"{kind!r} has no class mapping"
        assert scenes.KIND_TO_CLASS[kind] in scenes.CLASS_NAMES


def test_grid_dimensions_follow_from_the_stride():
    assert scenes.GRID_WIDTH == scenes.INPUT_WIDTH // scenes.OUTPUT_STRIDE
    assert scenes.GRID_HEIGHT == scenes.INPUT_HEIGHT // scenes.OUTPUT_STRIDE


# ---------------------------------------------------------------------------
# target encoding
# ---------------------------------------------------------------------------
def test_encode_then_decode_returns_the_original_boxes():
    """The round trip that everything else is measured through."""
    boxes = [_box(100, 60, 180, 100), _box(260, 64, 350, 104, cls=1)]
    heatmap, size, offset, _mask = scenes.encode_targets(boxes)

    recovered = detector.decode(heatmap, size, offset, threshold=0.99)
    assert len(recovered) == len(boxes)

    tp, fp, fn, mae = detector.match(recovered, boxes)
    assert (tp, fp, fn) == (2, 0, 0)
    assert mae < 1e-6


def test_encoded_peaks_are_exactly_one_at_each_centre():
    boxes = [_box(100, 60, 180, 100)]
    heatmap, _, _, mask = scenes.encode_targets(boxes)
    assert heatmap.max() == pytest.approx(1.0)
    assert mask.sum() == 1


def test_overlapping_vehicles_do_not_build_a_brighter_peak():
    """Splatting takes the maximum, not the sum.

    Summed Gaussians would make a crowded kerb produce confidences above one, and the
    model would learn that crowding means certainty.
    """
    boxes = [_box(100, 60, 180, 100), _box(150, 60, 230, 100)]
    heatmap, _, _, _ = scenes.encode_targets(boxes)
    assert heatmap.max() <= 1.0 + 1e-6


def test_offsets_capture_the_fraction_lost_to_rounding():
    # Centre at (150.5, 80.5) input pixels is cell (37, 20) plus a fraction.
    boxes = [_box(110.5, 60.5, 190.5, 100.5)]
    _, _, offset, mask = scenes.encode_targets(boxes)
    y, x = np.argwhere(mask > 0)[0]
    assert 0.0 <= offset[0, y, x] < 1.0
    assert 0.0 <= offset[1, y, x] < 1.0


def test_size_is_stored_in_input_pixels_not_grid_cells():
    """The decoder multiplies by no scale factor, so the encoder must not either."""
    boxes = [_box(100, 60, 180, 100)]
    _, size, _, mask = scenes.encode_targets(boxes)
    y, x = np.argwhere(mask > 0)[0]
    assert size[0, y, x] == pytest.approx(80.0)
    assert size[1, y, x] == pytest.approx(40.0)


def test_a_box_centred_outside_the_grid_is_not_encoded():
    boxes = [_box(scenes.INPUT_WIDTH + 50, 60, scenes.INPUT_WIDTH + 130, 100)]
    _, _, _, mask = scenes.encode_targets(boxes)
    assert mask.sum() == 0


def test_gaussian_radius_grows_with_the_object():
    """A fixed radius would smear a motorcycle and over-sharpen a truck."""
    small = scenes.gaussian_radius(5.0, 8.0)
    large = scenes.gaussian_radius(20.0, 60.0)
    assert large > small > 0.0


# ---------------------------------------------------------------------------
# frame clipping
# ---------------------------------------------------------------------------
def test_a_box_fully_inside_the_frame_is_untouched():
    x1, y1, x2, y2, visible = scenes._clip_box(10.0, 20.0, 110.0, 80.0)
    assert (x1, y1, x2, y2) == (10.0, 20.0, 110.0, 80.0)
    assert visible == pytest.approx(1.0)


def test_a_box_half_off_the_edge_reports_half_visible():
    _, _, _, _, visible = scenes._clip_box(scenes.INPUT_WIDTH - 50.0, 20.0,
                                           scenes.INPUT_WIDTH + 50.0, 80.0)
    assert visible == pytest.approx(0.5, abs=0.02)


def test_a_box_entirely_outside_the_frame_reports_nothing_visible():
    """The bug this catches was real.

    The renderer places a kerb far longer than the camera can see, and ``detections()``
    reports every vehicle on it, in frame or not. The first version of the dataset builder
    trained on all of them: four of six boxes in the first validation scene sat outside
    the image entirely.
    """
    _, _, _, _, visible = scenes._clip_box(
        scenes.INPUT_WIDTH + 100.0, 20.0, scenes.INPUT_WIDTH + 200.0, 80.0
    )
    assert visible == pytest.approx(0.0)


def test_the_visible_kerb_shrinks_as_the_camera_comes_closer():
    """Closer camera, bigger vehicles, less kerb in frame."""
    from parkfit.ml.synthetic.scene import CameraModel

    camera = CameraModel()
    near = scenes.visible_kerb_m(camera, 14.0)
    far = scenes.visible_kerb_m(camera, 30.0)
    assert far > near > 0.0
    # The default 18 m installation sees far less than the 40 m kerb the generator draws.
    assert scenes.visible_kerb_m(camera, 18.0) < 20.0


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def _blank_head(classes=len(scenes.CLASS_NAMES), rows=8, cols=12):
    return (
        np.zeros((classes, rows, cols), dtype=np.float32),
        np.zeros((2, rows, cols), dtype=np.float32),
        np.zeros((2, rows, cols), dtype=np.float32),
    )


def test_decode_places_a_single_peak_by_arithmetic():
    heatmap, size, offset = _blank_head()
    heatmap[0, 3, 5] = 0.9
    size[0, 3, 5], size[1, 3, 5] = 40.0, 20.0

    found = detector.decode(heatmap, size, offset, threshold=0.3)
    assert len(found) == 1
    # Centre (5*4, 3*4) = (20, 12), box 40x20.
    assert found[0]["x1"] == pytest.approx(0.0)
    assert found[0]["y1"] == pytest.approx(2.0)
    assert found[0]["x2"] == pytest.approx(40.0)
    assert found[0]["y2"] == pytest.approx(22.0)


def test_decode_reports_a_bright_neighbourhood_once():
    """The local-maximum test, which is why no NMS is needed in the graph."""
    heatmap, size, offset = _blank_head()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            heatmap[0, 3 + dy, 5 + dx] = 0.95 if (dx or dy) == 0 else 0.6
            size[0, 3 + dy, 5 + dx] = 40.0
            size[1, 3 + dy, 5 + dx] = 20.0

    found = detector.decode(heatmap, size, offset, threshold=0.3)
    assert len(found) == 1
    assert found[0]["score"] == pytest.approx(0.95)


def test_decode_respects_the_threshold():
    heatmap, size, offset = _blank_head()
    heatmap[0, 3, 5] = 0.25
    size[0, 3, 5], size[1, 3, 5] = 40.0, 20.0

    assert detector.decode(heatmap, size, offset, threshold=0.3) == []
    assert len(detector.decode(heatmap, size, offset, threshold=0.2)) == 1


def test_decode_labels_each_channel_with_its_class():
    heatmap, size, offset = _blank_head()
    heatmap[4, 2, 3] = 0.8  # motorcycle
    size[0, 2, 3], size[1, 2, 3] = 12.0, 14.0

    found = detector.decode(heatmap, size, offset, threshold=0.3)
    assert len(found) == 1
    assert found[0]["label"] == "motorcycle"
    assert found[0]["class"] == 4


def test_decode_returns_the_strongest_first():
    heatmap, size, offset = _blank_head()
    for x, score in ((2, 0.5), (6, 0.9), (10, 0.7)):
        heatmap[0, 3, x] = score
        size[0, 3, x], size[1, 3, x] = 30.0, 18.0

    found = detector.decode(heatmap, size, offset, threshold=0.3)
    assert [round(d["score"], 2) for d in found] == [0.9, 0.7, 0.5]


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def test_matching_counts_a_wrong_class_as_both_a_miss_and_a_false_alarm():
    truth = [_box(100, 60, 180, 100, cls=0)]
    predicted = [{**_box(100, 60, 180, 100, cls=1), "score": 0.9}]
    tp, fp, fn, _ = detector.match(predicted, truth)
    assert (tp, fp, fn) == (0, 1, 1)


def test_matching_rejects_a_box_that_overlaps_too_little():
    truth = [_box(100, 60, 180, 100)]
    predicted = [{**_box(170, 60, 250, 100), "score": 0.9}]
    tp, fp, fn, _ = detector.match(predicted, truth)
    assert (tp, fp, fn) == (0, 1, 1)


def test_matching_uses_each_truth_box_only_once():
    """Two predictions on one car is one hit and one false alarm, not two hits."""
    truth = [_box(100, 60, 180, 100)]
    predicted = [
        {**_box(100, 60, 180, 100), "score": 0.9},
        {**_box(102, 62, 182, 102), "score": 0.8},
    ]
    tp, fp, fn, _ = detector.match(predicted, truth)
    assert (tp, fp, fn) == (1, 1, 0)


def test_iou_of_identical_boxes_is_one_and_of_disjoint_boxes_is_zero():
    a = _box(0, 0, 10, 10)
    assert detector.iou(a, a) == pytest.approx(1.0)
    assert detector.iou(a, _box(100, 100, 110, 110)) == pytest.approx(0.0)
