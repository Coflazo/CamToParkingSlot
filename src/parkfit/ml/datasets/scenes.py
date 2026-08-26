"""Turning rendered scenes into a detection dataset.

The synthetic renderer already knows where every vehicle is, because it put them there.
That is the whole reason it exists: a real camera gives you pixels and no answer, so any
accuracy number measured against real footage is really a number about whoever drew the
boxes. Here the boxes are exact by construction.

**Storage is memory-mapped.** A few hundred 512x288 RGB frames is over a hundred
megabytes held as one array, and this machine does not reliably have that spare. Images
go to a single ``uint8`` ``.npy`` on disk and are read back a batch at a time, so peak
memory is a batch rather than a dataset.

**The split is by scene, in contiguous blocks.** Scenes are independent draws, so a
random split would not leak in the way consecutive video frames would. Blocks are used
anyway because it costs nothing and it keeps the habit: the moment this dataset grows a
temporal dimension, a random split silently starts reporting a number that means nothing.

**Class indices are the C++ enum.** ``CLASS_NAMES`` is ordered to match
``parkfit::vision::VehicleClass`` exactly, so channel *i* of the model's heatmap is
``VehicleClass(i)`` on the other side of the ONNX boundary with no lookup table in
between. A test asserts the two stay in step.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parkfit.ml.synthetic.scene import SceneGenerator

log = logging.getLogger(__name__)

#: Ordered to match ``parkfit::vision::VehicleClass``. Index is the contract.
CLASS_NAMES = ("car", "van", "truck", "bus", "motorcycle", "bicycle", "trailer")
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

#: The renderer names body styles; the detector predicts vehicle classes. An estate and a
#: hatchback are both a car as far as a parking space is concerned.
KIND_TO_CLASS = {
    "car": "car",
    "compact": "car",
    "estate": "car",
    "sedan": "car",
    "hatchback": "car",
    "suv": "car",
    "van": "van",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "trailer": "trailer",
}

INPUT_WIDTH = 512
INPUT_HEIGHT = 288

#: A box the frame only clips a sliver of is not a training target. Below this visible
#: fraction the model is being shown a few pixels of bumper and told it is a van.
MIN_VISIBLE_FRACTION = 0.45
OUTPUT_STRIDE = 4
GRID_WIDTH = INPUT_WIDTH // OUTPUT_STRIDE
GRID_HEIGHT = INPUT_HEIGHT // OUTPUT_STRIDE


@dataclass
class DatasetSplit:
    """One split, as paths rather than arrays. Nothing is loaded until it is needed."""

    name: str
    images_path: Path
    labels: list[list[dict]] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    #: Where this split starts inside the shared image file. Both splits share one array,
    #: so a split's local index i is image ``index_offset + i``. Stored rather than
    #: recomputed: deriving it by subtraction works until someone adds a third split.
    index_offset: int = 0

    def __len__(self) -> int:
        return len(self.labels)

    def images(self) -> np.ndarray:
        """Memory-mapped view. Slicing a batch out of this reads only that batch."""
        return np.load(self.images_path, mmap_mode="r")


@dataclass
class DatasetReport:
    train: int = 0
    val: int = 0
    boxes: int = 0
    dropped_outside_frame: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    per_condition: dict[str, int] = field(default_factory=dict)
    root: str = ""

    def describe(self) -> str:
        classes = ", ".join(f"{k} {v}" for k, v in sorted(self.per_class.items()) if v)
        return (
            f"{self.train} train / {self.val} val scenes, {self.boxes} boxes ({classes}); "
            f"{self.dropped_outside_frame} dropped as outside the frame"
        )


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize.

    Deliberately not an interpolating resize. The renderer draws flat-shaded boxes with
    hard edges, so smoothing them invents gradients the model would learn to rely on and
    a real camera would never produce.
    """
    src_h, src_w = image.shape[:2]
    rows = (np.arange(height) * (src_h / height)).astype(np.int32).clip(0, src_h - 1)
    cols = (np.arange(width) * (src_w / width)).astype(np.int32).clip(0, src_w - 1)
    return image[rows][:, cols]


def visible_kerb_m(camera, kerb_offset_m: float) -> float:
    """How many metres of kerb the camera can actually see.

    The renderer places vehicles from the camera's own x origin rightward, and that origin
    projects to the image centre, so the kerb occupies the right half of the frame and
    runs off the edge. At the default 18 m offset only about 13 m of a 40 m kerb is ever
    drawn. Asking for 40 m and training on all of it means training on vehicles that were
    never rendered, which is what the first version of this builder did: four of six boxes
    in the first validation scene sat outside the image.
    """
    tilt = math.radians(camera.tilt_deg)
    depth = kerb_offset_m * math.cos(tilt) + camera.height_m * math.sin(tilt)
    return (camera.width_px / 2.0) * depth / camera.focal_px


def _clip_box(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float, float]:
    """Clip to the frame and report what fraction of the original area survived."""
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    cx1 = min(max(x1, 0.0), INPUT_WIDTH)
    cy1 = min(max(y1, 0.0), INPUT_HEIGHT)
    cx2 = min(max(x2, 0.0), INPUT_WIDTH)
    cy2 = min(max(y2, 0.0), INPUT_HEIGHT)
    kept = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return cx1, cy1, cx2, cy2, kept / area


def build(
    root: Path,
    *,
    train_scenes: int = 600,
    val_scenes: int = 150,
    seed: int = 7,
) -> DatasetReport:
    """Render scenes and write a memory-mapped detection dataset.

    Camera distance is swept rather than fixed, so the model sees vehicles at a range of
    apparent sizes: a car is 60 px/m of kerb at 14 m and 30 px/m at 30 m. A dataset shot
    entirely at one distance teaches a scale, not a shape.
    """
    root.mkdir(parents=True, exist_ok=True)
    total = train_scenes + val_scenes

    generator = SceneGenerator(seed=seed)
    conditions_cycle = SceneGenerator.CONDITIONS
    camera_offsets = (14.0, 18.0, 22.0, 26.0, 30.0)
    clipped_away = 0
    report = DatasetReport(root=str(root))
    report.per_class = dict.fromkeys(CLASS_NAMES, 0)

    images_path = root / "images.npy"
    images = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, INPUT_HEIGHT, INPUT_WIDTH, 3),
    )

    all_labels: list[list[dict]] = []
    conditions: list[str] = []

    for index in range(total):
        condition = conditions_cycle[index % len(conditions_cycle)]
        # Sweep occupancy rather than fixing it. A dataset rendered entirely at 60%
        # teaches the model that a kerb always has roughly that many cars on it.
        occupancy = 0.45 + 0.5 * ((index * 7) % 11) / 10.0
        kerb_offset_m = camera_offsets[(index // len(conditions_cycle)) % len(camera_offsets)]
        # Ask for exactly the kerb the camera can see, with a little margin so the last
        # vehicle is not always half off the right edge.
        kerb_length_m = visible_kerb_m(generator.camera, kerb_offset_m) * 0.95
        scene = generator.build(
            kerb_length_m=kerb_length_m,
            condition=condition,
            occupancy=occupancy,
            kerb_offset_m=kerb_offset_m,
        )

        source = scene.image
        scale_x = INPUT_WIDTH / source.shape[1]
        scale_y = INPUT_HEIGHT / source.shape[0]
        images[index] = _resize_nearest(source, INPUT_WIDTH, INPUT_HEIGHT)

        boxes: list[dict] = []
        for detection in scene.detections():
            name = KIND_TO_CLASS.get(detection["label"])
            if name is None:
                # A body style the mapping does not cover is a bug in the mapping, not a
                # sample to quietly drop into "car".
                log.warning("unmapped vehicle kind %r; skipping box", detection["label"])
                continue
            x1, y1, x2, y2, visible = _clip_box(
                detection["x1"] * scale_x,
                detection["y1"] * scale_y,
                detection["x2"] * scale_x,
                detection["y2"] * scale_y,
            )
            if visible < MIN_VISIBLE_FRACTION or x2 - x1 < 4.0 or y2 - y1 < 3.0:
                clipped_away += 1
                continue
            boxes.append(
                {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                    "class": CLASS_INDEX[name],
                    "label": name,
                }
            )
            report.per_class[name] += 1

        all_labels.append(boxes)
        conditions.append(condition)
        report.boxes += len(boxes)
        report.per_condition[condition] = report.per_condition.get(condition, 0) + 1

    images.flush()
    del images
    if clipped_away:
        log.info("%d boxes dropped as outside or barely inside the frame", clipped_away)
    report.dropped_outside_frame = clipped_away

    report.train = train_scenes
    report.val = val_scenes
    (root / "labels.json").write_text(
        json.dumps(
            {
                "class_names": list(CLASS_NAMES),
                "input_width": INPUT_WIDTH,
                "input_height": INPUT_HEIGHT,
                "output_stride": OUTPUT_STRIDE,
                "train_scenes": train_scenes,
                "val_scenes": val_scenes,
                "conditions": conditions,
                "labels": all_labels,
            }
        ),
        encoding="utf-8",
    )
    return report


def load(root: Path) -> tuple[DatasetSplit, DatasetSplit]:
    """Read a built dataset back as two splits."""
    meta = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    images_path = root / "images.npy"
    cut = meta["train_scenes"]

    train = DatasetSplit(
        name="train",
        images_path=images_path,
        labels=meta["labels"][:cut],
        conditions=meta["conditions"][:cut],
        index_offset=0,
    )
    val = DatasetSplit(
        name="val",
        images_path=images_path,
        labels=meta["labels"][cut:],
        conditions=meta["conditions"][cut:],
        index_offset=cut,
    )
    return train, val


# ---------------------------------------------------------------------------
# CenterNet targets
# ---------------------------------------------------------------------------
def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Radius at which a predicted box still overlaps the truth by ``min_overlap``.

    From the CornerNet derivation. A fixed radius would treat a 20-pixel motorcycle and a
    200-pixel truck the same, which either smears the small object's peak across its
    neighbours or gives the large one a peak too sharp to ever hit.
    """
    a1, b1, c1 = 1.0, height + width, width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4 * a1 * c1))
    r1 = (b1 - sq1) / (2 * a1)

    a2, b2, c2 = 4.0, 2 * (height + width), (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2**2 - 4 * a2 * c2))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


def _draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Splat a Gaussian peak, keeping the maximum where two vehicles overlap.

    Maximum rather than sum: two adjacent cars must not build a peak brighter than one,
    or the model learns that crowding means higher confidence.
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0

    height, width = heatmap.shape
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    masked = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    masked_kernel = kernel[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked, masked_kernel, out=masked)


def encode_targets(boxes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the CenterNet training targets for one image.

    Returns ``(heatmap, size, offset, mask)``. ``size`` is in input pixels rather than
    grid cells so the decoder needs no scale factor, and ``offset`` recovers the fraction
    of a cell lost when the centre was rounded to an integer grid position.
    """
    heatmap = np.zeros((len(CLASS_NAMES), GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    size = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    offset = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

    for box in boxes:
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]
        if w <= 0 or h <= 0:
            continue
        cx = (box["x1"] + box["x2"]) / 2.0 / OUTPUT_STRIDE
        cy = (box["y1"] + box["y2"]) / 2.0 / OUTPUT_STRIDE
        cxi, cyi = int(cx), int(cy)
        if not (0 <= cxi < GRID_WIDTH and 0 <= cyi < GRID_HEIGHT):
            continue

        radius = max(1, int(gaussian_radius(h / OUTPUT_STRIDE, w / OUTPUT_STRIDE)))
        _draw_gaussian(heatmap[box["class"]], cxi, cyi, radius)

        size[0, cyi, cxi] = w
        size[1, cyi, cxi] = h
        offset[0, cyi, cxi] = cx - cxi
        offset[1, cyi, cxi] = cy - cyi
        mask[cyi, cxi] = 1.0

    return heatmap, size, offset, mask
