"""A small anchor-free vehicle detector, trained in PyTorch and exported to ONNX.

The architecture is CenterNet: predict a per-class heatmap whose peaks are object
centres, plus a box size and a sub-pixel offset at each peak. Chosen over an anchor-based
detector for one practical reason above all others, which is that decoding it needs no
anchor bookkeeping and no non-maximum suppression inside the graph. A 3x3 local-maximum
test over the heatmap is the whole decoder, which is fifteen lines of C++ on the other
side of the ONNX boundary rather than a port of an anchor generator that has to agree
with the Python one bit for bit.

The backbone is deliberately tiny, roughly 200k parameters. The job is not open-world
detection; it is finding flat-shaded vehicles against a road in a fixed camera. Spending
a ResNet on that would train slower, export bigger, and measure nothing extra.

**Three losses, and only one of them is dense.** The heatmap uses the CornerNet focal
variant, which down-weights the enormous background rather than letting it drown the few
positive cells. Size and offset are L1 and are evaluated *only* at true centre cells,
because "what size is the object at this empty patch of road" has no answer to regress
toward.

**Size is predicted in input pixels, not grid cells.** It costs nothing here and means
the C++ decoder multiplies by no scale factor it could get wrong.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parkfit.ml.datasets import scenes

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_WEIGHTS = DEFAULT_MODEL_DIR / "detector.pt"
DEFAULT_ONNX = DEFAULT_MODEL_DIR / "detector.onnx"

#: Bumped whenever the graph's inputs, outputs or class order change. The worker records
#: it on every published observation, so a detection can always be traced to the model
#: that made it.
MODEL_VERSION = "curb-detector-0.1.0"


@dataclass
class EpochStat:
    epoch: int
    loss: float
    heatmap_loss: float
    size_loss: float
    offset_loss: float
    seconds: float


@dataclass
class TrainReport:
    trained: bool = False
    reason: str = ""
    epochs: int = 0
    parameters: int = 0
    history: list[EpochStat] = field(default_factory=list)
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_box_mae_px: float = 0.0
    per_condition: dict[str, float] = field(default_factory=dict)
    weights_path: str = ""

    def describe(self) -> str:
        if not self.trained:
            return f"not trained: {self.reason}"
        last = self.history[-1] if self.history else None
        head = f"{self.parameters:,} parameters, {self.epochs} epochs"
        if last:
            head += f", final loss {last.loss:.4f}"
        return (
            f"{head}\n"
            f"  validation: precision {self.val_precision:.3f}, "
            f"recall {self.val_recall:.3f}, F1 {self.val_f1:.3f}, "
            f"box MAE {self.val_box_mae_px:.2f} px"
        )


def _torch():
    """Import torch lazily, so the package imports on a machine without it."""
    try:
        import torch

        return torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return None


def build_model(num_classes: int = len(scenes.CLASS_NAMES)):
    """The detector graph.

    A downsampling trunk to stride 16, then two upsampling steps back to stride 4 with
    skip connections. Stride 4 matters: at stride 8 a distant motorcycle occupies a
    single cell and its centre cannot be localised to better than eight pixels, which is
    most of the vehicle.
    """
    import torch
    from torch import nn

    def conv_bn(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(conv_bn(3, 16, stride=2), conv_bn(16, 16))
            self.down1 = nn.Sequential(conv_bn(16, 32, stride=2), conv_bn(32, 32))
            self.down2 = nn.Sequential(conv_bn(32, 64, stride=2), conv_bn(64, 64))
            self.down3 = nn.Sequential(conv_bn(64, 96, stride=2), conv_bn(96, 96))

            self.up2 = conv_bn(96, 64)
            self.up1 = conv_bn(64, 32)
            self.fuse = conv_bn(32, 32)

            self.heatmap_head = nn.Sequential(conv_bn(32, 32), nn.Conv2d(32, num_classes, 1))
            self.size_head = nn.Sequential(conv_bn(32, 32), nn.Conv2d(32, 2, 1))
            self.offset_head = nn.Sequential(conv_bn(32, 32), nn.Conv2d(32, 2, 1))

            # Start the heatmap strongly negative. Without it the first steps predict
            # roughly 0.5 everywhere, the focal loss sees tens of thousands of confident
            # false positives, and the gradient blows the run apart before it starts.
            nn.init.constant_(self.heatmap_head[-1].bias, -4.6)  # sigmoid(-4.6) ~ 0.01

        def forward(self, x):
            s2 = self.stem(x)  # stride 2
            s4 = self.down1(s2)  # stride 4
            s8 = self.down2(s4)  # stride 8
            s16 = self.down3(s8)  # stride 16

            u8 = self.up2(nn.functional.interpolate(s16, size=s8.shape[-2:], mode="nearest"))
            u8 = u8 + s8
            u4 = self.up1(nn.functional.interpolate(u8, size=s4.shape[-2:], mode="nearest"))
            u4 = u4 + s4
            feat = self.fuse(u4)

            # Sigmoid inside the graph, so the exported model hands C++ a probability and
            # the threshold on the other side means what it says.
            heatmap = torch.sigmoid(self.heatmap_head(feat))
            # Sizes are positive by definition; softplus removes a whole class of
            # nonsense box from the decoder's problem.
            size = nn.functional.softplus(self.size_head(feat))
            offset = torch.sigmoid(self.offset_head(feat))
            return heatmap, size, offset

    return Detector()


def focal_loss(pred, target):
    """CornerNet focal loss for a Gaussian-splatted heatmap.

    Cells near a true centre are penalised less for being bright, scaled by how close
    they are. A plain binary cross-entropy would treat the cell one pixel from a car's
    centre as a full negative, which is a label the geometry contradicts.
    """
    import torch

    pred = pred.clamp(1e-4, 1 - 1e-4)
    positives = target.eq(1.0).float()
    negatives = 1.0 - positives

    positive_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * positives
    negative_loss = (
        -torch.log(1 - pred)
        * torch.pow(pred, 2)
        * torch.pow(1 - target, 4)  # the Gaussian falloff
        * negatives
    )

    count = positives.sum()
    if count == 0:
        return negative_loss.sum()
    return (positive_loss.sum() + negative_loss.sum()) / count


def masked_l1(pred, target, mask):
    """L1 at true centre cells only."""
    import torch

    expanded = mask.unsqueeze(1).expand_as(pred)
    total = torch.abs(pred - target) * expanded
    return total.sum() / (expanded.sum() + 1e-4)


def decode(
    heatmap: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    *,
    threshold: float = 0.3,
    max_detections: int = 64,
) -> list[dict]:
    """Turn model output into boxes. The reference the C++ decoder must match.

    Peak extraction is a 3x3 local-maximum test rather than a sort over every cell: a
    single vehicle lights up a small neighbourhood, and taking the top-K cells directly
    would return the same car several times.
    """
    classes, rows, cols = heatmap.shape
    detections: list[dict] = []

    for c in range(classes):
        plane = heatmap[c]
        for y in range(rows):
            for x in range(cols):
                score = float(plane[y, x])
                if score < threshold:
                    continue
                y0, y1 = max(0, y - 1), min(rows, y + 2)
                x0, x1 = max(0, x - 1), min(cols, x + 2)
                if score < float(plane[y0:y1, x0:x1].max()):
                    continue  # not a local maximum

                w = float(size[0, y, x])
                h = float(size[1, y, x])
                cx = (x + float(offset[0, y, x])) * scenes.OUTPUT_STRIDE
                cy = (y + float(offset[1, y, x])) * scenes.OUTPUT_STRIDE
                detections.append(
                    {
                        "x1": cx - w / 2.0,
                        "y1": cy - h / 2.0,
                        "x2": cx + w / 2.0,
                        "y2": cy + h / 2.0,
                        "score": score,
                        "class": c,
                        "label": scenes.CLASS_NAMES[c],
                    }
                )

    detections.sort(key=lambda d: -d["score"])
    return detections[:max_detections]


def iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def match(
    predicted: list[dict], truth: list[dict], *, iou_threshold: float = 0.5
) -> tuple[int, int, int, float]:
    """Greedy highest-score-first matching. Returns ``(tp, fp, fn, box mae)``.

    Class-aware: a van detected as a truck is a miss and a false alarm, not a hit. That
    is stricter than it needs to be for gap measurement, where any vehicle blocks a
    space, but it keeps the number honest about what the model actually learned.
    """
    unmatched = list(truth)
    tp = 0
    errors: list[float] = []

    for prediction in sorted(predicted, key=lambda d: -d["score"]):
        best, best_iou = None, iou_threshold
        for candidate in unmatched:
            if candidate["class"] != prediction["class"]:
                continue
            score = iou(prediction, candidate)
            if score >= best_iou:
                best, best_iou = candidate, score
        if best is None:
            continue
        unmatched.remove(best)
        tp += 1
        errors.append(
            (
                abs(prediction["x1"] - best["x1"])
                + abs(prediction["y1"] - best["y1"])
                + abs(prediction["x2"] - best["x2"])
                + abs(prediction["y2"] - best["y2"])
            )
            / 4.0
        )

    fp = len(predicted) - tp
    fn = len(unmatched)
    return tp, fp, fn, float(np.mean(errors)) if errors else 0.0


def _batches(split: scenes.DatasetSplit, batch_size: int, shuffle: bool, seed: int):
    """Yield ``(images, heatmap, size, offset, mask)`` batches from the memmap."""
    images = split.images()
    base = split.index_offset
    order = np.arange(len(split))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)

    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        if len(chunk) == 0:
            continue
        # Read the batch out of the memmap, then normalise. Channels-first for torch.
        raw = np.stack([images[base + int(i)] for i in chunk]).astype(np.float32)
        batch_images = np.transpose(raw / 255.0, (0, 3, 1, 2))

        targets = [scenes.encode_targets(split.labels[int(i)]) for i in chunk]
        yield (
            batch_images,
            np.stack([t[0] for t in targets]),
            np.stack([t[1] for t in targets]),
            np.stack([t[2] for t in targets]),
            np.stack([t[3] for t in targets]),
            chunk,
        )


def train(
    dataset_root: Path,
    *,
    epochs: int = 24,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    weights_path: Path = DEFAULT_WEIGHTS,
    seed: int = 11,
    threads: int = 4,
) -> TrainReport:
    """Fit the detector and score it on the held-out scenes."""
    torch = _torch()
    if torch is None:
        return TrainReport(reason="pytorch is not installed")
    if not (dataset_root / "labels.json").exists():
        return TrainReport(reason=f"no dataset at {dataset_root}; run `pf detect dataset`")

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))

    train_split, val_split = scenes.load(dataset_root)
    model = build_model()
    parameters = sum(p.numel() for p in model.parameters())

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=learning_rate,
        total_steps=max(1, epochs * ((len(train_split) + batch_size - 1) // batch_size)),
    )

    report = TrainReport(trained=True, epochs=epochs, parameters=parameters)
    report.weights_path = str(weights_path)

    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        totals = np.zeros(4)
        steps = 0

        for images, heat, size, offset, mask, _ in _batches(
            train_split, batch_size, shuffle=True, seed=seed + epoch
        ):
            x = torch.from_numpy(images)
            heat_t = torch.from_numpy(heat)
            size_t = torch.from_numpy(size)
            offset_t = torch.from_numpy(offset)
            mask_t = torch.from_numpy(mask)

            pred_heat, pred_size, pred_offset = model(x)
            loss_heat = focal_loss(pred_heat, heat_t)
            loss_size = masked_l1(pred_size, size_t, mask_t)
            loss_offset = masked_l1(pred_offset, offset_t, mask_t)
            # Size is in pixels and runs to a couple of hundred, so its raw L1 dwarfs the
            # other two. 0.1 puts the three terms on comparable footing.
            loss = loss_heat + 0.1 * loss_size + loss_offset

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            schedule.step()

            # detach before reading: these are only for the log, and converting a
            # tensor that still carries grad keeps the graph alive for no reason.
            totals += [
                loss.detach().item(),
                loss_heat.detach().item(),
                loss_size.detach().item(),
                loss_offset.detach().item(),
            ]
            steps += 1

        steps = max(1, steps)
        report.history.append(
            EpochStat(
                epoch=epoch + 1,
                loss=totals[0] / steps,
                heatmap_loss=totals[1] / steps,
                size_loss=totals[2] / steps,
                offset_loss=totals[3] / steps,
                seconds=time.perf_counter() - started,
            )
        )
        log.info(
            "epoch %d/%d loss %.4f (heat %.4f size %.3f offset %.4f) in %.1fs",
            epoch + 1,
            epochs,
            report.history[-1].loss,
            report.history[-1].heatmap_loss,
            report.history[-1].size_loss,
            report.history[-1].offset_loss,
            report.history[-1].seconds,
        )

    evaluate(model, val_split, report)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    weights_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "class_names": list(scenes.CLASS_NAMES),
                "input_width": scenes.INPUT_WIDTH,
                "input_height": scenes.INPUT_HEIGHT,
                "output_stride": scenes.OUTPUT_STRIDE,
                "parameters": parameters,
                "val_precision": report.val_precision,
                "val_recall": report.val_recall,
                "val_f1": report.val_f1,
                "val_box_mae_px": report.val_box_mae_px,
                "per_condition_f1": report.per_condition,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def evaluate(model, split: scenes.DatasetSplit, report: TrainReport) -> None:
    """Score the model on held-out scenes, overall and per lighting condition."""
    torch = _torch()
    model.eval()

    tp = fp = fn = 0
    errors: list[float] = []
    by_condition: dict[str, list[int]] = {}

    with torch.no_grad():
        for images, _, _, _, _, chunk in _batches(split, 4, shuffle=False, seed=0):
            heat, size, offset = model(torch.from_numpy(images))
            heat = heat.numpy()
            size = size.numpy()
            offset = offset.numpy()

            for row, index in enumerate(chunk):
                predicted = decode(heat[row], size[row], offset[row])
                truth = split.labels[int(index)]
                a, b, c, mae = match(predicted, truth)
                tp += a
                fp += b
                fn += c
                if a:
                    errors.append(mae)

                condition = split.conditions[int(index)]
                bucket = by_condition.setdefault(condition, [0, 0, 0])
                bucket[0] += a
                bucket[1] += b
                bucket[2] += c

    report.val_precision = tp / (tp + fp) if (tp + fp) else 0.0
    report.val_recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = report.val_precision + report.val_recall
    report.val_f1 = (
        2 * report.val_precision * report.val_recall / denominator if denominator else 0.0
    )
    report.val_box_mae_px = float(np.mean(errors)) if errors else 0.0

    for condition, (a, b, c) in sorted(by_condition.items()):
        precision = a / (a + b) if (a + b) else 0.0
        recall = a / (a + c) if (a + c) else 0.0
        total = precision + recall
        report.per_condition[condition] = 2 * precision * recall / total if total else 0.0
