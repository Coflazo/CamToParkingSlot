"""Turning live camera frames into a training set the detector can actually learn from.

The rendered-scene dataset in :mod:`parkfit.ml.datasets.scenes` gives exact ground truth,
which is the only way to measure gap-length error honestly. What it cannot give is a real
street: flat-shaded boxes never taught the model what a tree shadow, a tram rail or a
low sun through a windscreen looks like, and the first real frame it saw produced two
motorcycles in a tree and no police vans at all.

So the labels here come from a teacher instead of from a renderer. A COCO-pretrained
Faster R-CNN has seen 118,000 real photographs; it is far too heavy to run per frame in
the vision worker, but it runs once, offline, over harvested frames and writes down what
it saw. The small CenterNet then learns from those labels and keeps the ONNX contract the
C++ worker already speaks.

**These labels are not ground truth and are never treated as such.** The teacher is wrong
sometimes, and a student trained on its output inherits those mistakes. What the arrangement
buys is domain: every pixel is a real Dutch street under real light. Gap-length accuracy is
still measured against rendered scenes, where the true answer is known by construction.

Two of the seven classes have no source here. COCO has no "van" (they land in ``car`` or
``truck``) and no "trailer", so the student sees five of the seven. That is a real gap and
it is recorded rather than papered over.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parkfit.ml.datasets import scenes

log = logging.getLogger(__name__)

#: COCO category name to our class name. COCO's "truck" covers panel vans and box trucks
#: alike, so a Dutch bestelbus arrives labelled truck; that is the teacher's vocabulary,
#: not ours, and pretending otherwise would just move the error somewhere less visible.
COCO_TO_CLASS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}

#: Below this the teacher is guessing, and a guess promoted to a label is worse than no
#: label at all: the student learns to reproduce it confidently.
TEACHER_SCORE_THRESHOLD = 0.60

#: A box this small at 512x288 is a handful of pixels. The renderer already clips these
#: for the same reason: nothing can be localised inside four pixels.
MIN_BOX_PIXELS = 6.0


@dataclass(frozen=True)
class LabelledFrame:
    """One real frame and what the teacher found in it."""

    path: Path
    boxes: list[dict]

    @property
    def camera_id(self) -> str:
        # Harvested files are named "<camera_id>_<stamp>_<index>.jpg", and the camera id
        # itself contains underscores, so split from the right.
        return self.path.stem.rsplit("_", 2)[0]


def load_teacher(device: str = "cuda"):
    """The COCO-pretrained detector used to label real frames.

    Kept behind a function because torchvision is an optional extra and importing it at
    module scope would make the whole dataset package require it.
    """
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA was asked for but is not available, labelling on CPU instead")
        device = "cpu"

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=TEACHER_SCORE_THRESHOLD)
    model.eval().to(device)
    return model, weights.meta["categories"], device


def label_frames(
    frames: list[Path],
    *,
    device: str = "cuda",
    score_threshold: float = TEACHER_SCORE_THRESHOLD,
) -> list[LabelledFrame]:
    """Run the teacher over real frames and keep only the vehicle classes.

    Boxes come back in the frame's own pixel coordinates and are rescaled to the model
    input here, so a caller never has to know what resolution the camera happens to serve.
    """
    import torch
    from torchvision.io import read_image

    model, categories, device = load_teacher(device)
    out: list[LabelledFrame] = []

    for path in frames:
        try:
            image = read_image(str(path)).float() / 255.0
        except Exception as exc:
            log.warning("unreadable frame %s: %s", path.name, exc)
            continue

        _, height, width = image.shape
        with torch.inference_mode():
            prediction = model([image.to(device)])[0]

        scale_x = scenes.INPUT_WIDTH / float(width)
        scale_y = scenes.INPUT_HEIGHT / float(height)

        boxes: list[dict] = []
        for box, label, score in zip(
            prediction["boxes"].tolist(),
            prediction["labels"].tolist(),
            prediction["scores"].tolist(),
            strict=True,
        ):
            if score < score_threshold:
                continue
            name = COCO_TO_CLASS.get(categories[label])
            if name is None:
                continue
            x1, y1, x2, y2 = box
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            x1 = max(0.0, min(x1, scenes.INPUT_WIDTH))
            x2 = max(0.0, min(x2, scenes.INPUT_WIDTH))
            y1 = max(0.0, min(y1, scenes.INPUT_HEIGHT))
            y2 = max(0.0, min(y2, scenes.INPUT_HEIGHT))
            if x2 - x1 < MIN_BOX_PIXELS or y2 - y1 < MIN_BOX_PIXELS:
                continue
            boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class": scenes.CLASS_INDEX[name],
                    "score": round(float(score), 4),
                }
            )

        out.append(LabelledFrame(path=path, boxes=boxes))
        log.debug("%s: %d vehicles", path.name, len(boxes))

    return out


def write_labels(labelled: list[LabelledFrame], destination: Path) -> Path:
    """Persist the labels next to the frames so training need not re-run the teacher."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"path": str(item.path.as_posix()), "camera_id": item.camera_id, "boxes": item.boxes}
        for item in labelled
    ]
    destination.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return destination


def read_labels(source: Path) -> list[LabelledFrame]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [LabelledFrame(path=Path(row["path"]), boxes=row["boxes"]) for row in payload]


def to_arrays(labelled: list[LabelledFrame]) -> tuple[np.ndarray, ...]:
    """Encode labelled frames into the CenterNet targets the trainer expects.

    Images come back as float32 NCHW in [0, 1] at the model's input size, and the four
    target tensors match :func:`parkfit.ml.datasets.scenes.encode_targets` exactly, so the
    same training loop consumes rendered and real data without knowing which it has.
    """
    from PIL import Image

    n = len(labelled)
    images = np.zeros((n, 3, scenes.INPUT_HEIGHT, scenes.INPUT_WIDTH), dtype=np.float32)
    heatmaps = np.zeros(
        (n, len(scenes.CLASS_NAMES), scenes.GRID_HEIGHT, scenes.GRID_WIDTH), dtype=np.float32
    )
    sizes = np.zeros((n, 2, scenes.GRID_HEIGHT, scenes.GRID_WIDTH), dtype=np.float32)
    offsets = np.zeros((n, 2, scenes.GRID_HEIGHT, scenes.GRID_WIDTH), dtype=np.float32)
    masks = np.zeros((n, scenes.GRID_HEIGHT, scenes.GRID_WIDTH), dtype=np.float32)

    for i, item in enumerate(labelled):
        with Image.open(item.path) as handle:
            resized = handle.convert("RGB").resize(
                (scenes.INPUT_WIDTH, scenes.INPUT_HEIGHT), Image.BILINEAR
            )
            images[i] = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        heatmaps[i], sizes[i], offsets[i], masks[i] = scenes.encode_targets(item.boxes)

    return images, heatmaps, sizes, offsets, masks


def split_by_camera(
    labelled: list[LabelledFrame], holdout: set[str]
) -> tuple[list[LabelledFrame], list[LabelledFrame]]:
    """Split train and test by camera, never by frame.

    Two frames from the same camera five seconds apart are nearly the same picture. Split
    those at random and the test set is memorised rather than predicted, which is how a
    detector reports a number it cannot reproduce on a street it has not seen.
    """
    train = [f for f in labelled if f.camera_id not in holdout]
    test = [f for f in labelled if f.camera_id in holdout]
    return train, test
