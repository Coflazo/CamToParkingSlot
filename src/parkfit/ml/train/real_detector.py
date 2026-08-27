"""Training the detector on real camera frames instead of rendered ones.

Same architecture, same decode, same ONNX contract as
:mod:`parkfit.ml.train.detector`. The only thing that changes is where the pixels come
from, and that turned out to be the thing that mattered: the rendered-scene model scored
F1 0.994 on rendered scenes and found two motorcycles in a tree on the first real frame.

**Held out by camera, never by frame.** Frames from one feed five seconds apart are very
nearly the same picture, so a random split lets the model memorise the test set and
report a number it cannot reproduce on a street it has not seen. Whole cameras are held
out instead, which is the same discipline the occupancy model uses for targets and time.

**Augmentation is deliberately thin.** Horizontal flip and photometric jitter only. A
fixed camera sees one geometry forever, so rotation and scale jitter would teach the
model to expect variation the deployment never has, and colour jitter is doing the work
that matters: the same street at 08:00 and 19:00 is the actual distribution shift.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parkfit.ml.datasets import real, scenes
from parkfit.ml.train.detector import (
    DEFAULT_WEIGHTS,
    build_model,
    decode,
    focal_loss,
    masked_l1,
    match,
)

log = logging.getLogger(__name__)

DEFAULT_REAL_WEIGHTS = DEFAULT_WEIGHTS.parent / "detector_real.pt"


@dataclass
class RealTrainReport:
    trained: bool = False
    reason: str = ""
    device: str = ""
    epochs: int = 0
    parameters: int = 0
    backbone: str = ""
    input_width: int = 0
    input_height: int = 0
    train_frames: int = 0
    test_frames: int = 0
    train_boxes: int = 0
    test_boxes: int = 0
    holdout_cameras: list[str] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    box_mae_px: float = 0.0
    per_class: dict[str, float] = field(default_factory=dict)
    peak_vram_mb: float = 0.0
    weights_path: str = ""

    def describe(self) -> str:
        if not self.trained:
            return f"not trained: {self.reason}"
        return (
            f"{self.parameters:,} parameters ({self.backbone} trunk) on {self.device}, "
            f"{self.epochs} epochs at {self.input_width}x{self.input_height}, "
            f"{self.train_frames} real train frames ({self.train_boxes} boxes), "
            f"held out {', '.join(self.holdout_cameras)} "
            f"({self.test_frames} frames, {self.test_boxes} boxes)\n"
            f"  precision {self.precision:.3f}  recall {self.recall:.3f}  "
            f"F1 {self.f1:.3f}  box MAE {self.box_mae_px:.2f} px  "
            f"peak VRAM {self.peak_vram_mb:.0f} MB"
        )


def build_pretrained_model(num_classes: int = len(scenes.CLASS_NAMES)):
    """CenterNet heads on an ImageNet-pretrained MobileNetV3 trunk.

    The from-scratch 322k-parameter trunk fits nine fixed camera views perfectly and
    finds nothing at all on a tenth. That is not a capacity problem, it is a features
    problem: with a few hundred frames from a handful of viewpoints there is no way for
    a randomly initialised trunk to learn what a car looks like as opposed to what
    *this* street looks like. A trunk that has already seen 1.2 million photographs
    starts with edges, texture and shape, and only the heads have to learn the task.

    Output shapes, activations and channel order are identical to
    :func:`parkfit.ml.train.detector.build_model`, so the ONNX contract and the C++
    decoder do not know or care which trunk produced the tensors.
    """
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    features = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features

    class PretrainedDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Stage boundaries taken from the trunk's own strides: 4, 8, 16, 32.
            self.stage4 = nn.Sequential(*features[0:2])
            self.stage8 = nn.Sequential(*features[2:4])
            self.stage16 = nn.Sequential(*features[4:9])
            self.stage32 = nn.Sequential(*features[9:12])

            def lateral(in_ch: int, out_ch: int = 64) -> nn.Conv2d:
                return nn.Conv2d(in_ch, out_ch, 1, bias=False)

            self.lat32, self.lat16 = lateral(96), lateral(48)
            self.lat8, self.lat4 = lateral(24), lateral(16)

            def smooth() -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )

            self.smooth16, self.smooth8, self.smooth4 = smooth(), smooth(), smooth()

            def head(out_ch: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, out_ch, 1),
                )

            self.heatmap_head = head(num_classes)
            self.size_head = head(2)
            self.offset_head = head(2)
            # Same negative prior as the scratch model: without it the first steps predict
            # roughly 0.5 everywhere and the focal loss sees tens of thousands of
            # confident false positives.
            nn.init.constant_(self.heatmap_head[-1].bias, -4.6)

        def forward(self, x):
            c4 = self.stage4(x)
            c8 = self.stage8(c4)
            c16 = self.stage16(c8)
            c32 = self.stage32(c16)

            def up(small, lateral_out):
                return (
                    nn.functional.interpolate(small, size=lateral_out.shape[-2:], mode="nearest")
                    + lateral_out
                )

            p16 = self.smooth16(up(self.lat32(c32), self.lat16(c16)))
            p8 = self.smooth8(up(p16, self.lat8(c8)))
            p4 = self.smooth4(up(p8, self.lat4(c4)))

            heatmap = torch.sigmoid(self.heatmap_head(p4))
            size = nn.functional.softplus(self.size_head(p4))
            offset = torch.sigmoid(self.offset_head(p4))
            return heatmap, size, offset

    return PretrainedDetector()


def _augment(
    images: np.ndarray,
    heat: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Horizontal flip only, on uint8. Brightness and contrast happen on the GPU.

    Splitting them matters for memory: jittering here would force a float32 copy of the
    batch on the CPU, and the whole point of holding the set as uint8 is not to make
    those copies.
    """
    images = images.copy()
    heat, size, offset, mask = heat.copy(), size.copy(), offset.copy(), mask.copy()

    for i in range(images.shape[0]):
        if rng.random() < 0.5:
            images[i] = images[i][:, :, ::-1]
            heat[i] = heat[i][:, :, ::-1]
            size[i] = size[i][:, :, ::-1]
            mask[i] = mask[i][:, ::-1]
            # The x offset is a fraction of a cell measured left to right, so mirroring
            # the grid without inverting it puts every centre on the wrong side of its
            # cell.
            offset[i] = offset[i][:, :, ::-1]
            offset[i][0] = np.where(mask[i] > 0, 1.0 - offset[i][0], 0.0)

    return images, heat, size, offset, mask


def _augment(
    images: np.ndarray,
    heat: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Horizontal flip plus brightness and contrast jitter, applied per sample."""
    images = images.copy()
    heat, size, offset, mask = heat.copy(), size.copy(), offset.copy(), mask.copy()

    for i in range(images.shape[0]):
        if rng.random() < 0.5:
            images[i] = images[i][:, :, ::-1]
            heat[i] = heat[i][:, :, ::-1]
            size[i] = size[i][:, :, ::-1]
            mask[i] = mask[i][:, ::-1]
            # The x offset is a fraction of a cell measured left to right, so mirroring
            # the grid without negating it puts every centre on the wrong side of its cell.
            offset[i] = offset[i][:, :, ::-1]
            offset[i][0] = np.where(mask[i] > 0, 1.0 - offset[i][0], 0.0)

        if rng.random() < 0.8:
            brightness = rng.uniform(0.65, 1.35)
            contrast = rng.uniform(0.8, 1.25)
            frame = images[i] * brightness
            frame = (frame - frame.mean()) * contrast + frame.mean()
            images[i] = np.clip(frame, 0.0, 1.0)

    return images, heat, size, offset, mask


def train_real(
    labels_path: Path,
    *,
    holdout_cameras: set[str] | None = None,
    epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    weights_path: Path = DEFAULT_REAL_WEIGHTS,
    seed: int = 11,
    device: str = "cuda",
    backbone: str = "pretrained",
    width: int = 960,
    height: int = 544,
) -> RealTrainReport:
    """Fit the detector on teacher-labelled real frames and score it on unseen cameras."""
    try:
        import torch
    except ImportError:
        return RealTrainReport(reason="pytorch is not installed")

    if not labels_path.exists():
        return RealTrainReport(reason=f"no labels at {labels_path}; run `pf detect harvest` first")

    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA unavailable, training on CPU")
        device = "cpu"

    labelled = real.read_labels(labels_path)
    labelled = [f for f in labelled if f.path.exists()]
    if not labelled:
        return RealTrainReport(reason="labels reference no frames that still exist")

    cameras = sorted({f.camera_id for f in labelled})
    if holdout_cameras is None:
        # Hold out roughly a fifth of the cameras, at least one.
        count = max(1, len(cameras) // 5)
        holdout_cameras = set(cameras[-count:])

    train_frames, test_frames = real.split_by_camera(labelled, holdout_cameras)
    if not train_frames or not test_frames:
        return RealTrainReport(
            reason=f"split left {len(train_frames)} train / {len(test_frames)} test frames"
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    log.info("encoding %d train / %d test frames", len(train_frames), len(test_frames))
    tr_images, tr_heat, tr_size, tr_offset, tr_mask = real.to_arrays(train_frames, width, height)
    te_images, *_ = real.to_arrays(test_frames, width, height)
    log.info(
        "dataset in memory: %.2f GB as uint8",
        (tr_images.nbytes + te_images.nbytes) / 1024**3,
    )

    model = (build_pretrained_model() if backbone == "pretrained" else build_model()).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps_per_epoch = max(1, (len(train_frames) + batch_size - 1) // batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=epochs * steps_per_epoch
    )

    report = RealTrainReport(
        trained=True,
        device=device,
        epochs=epochs,
        parameters=parameters,
        backbone=backbone,
        input_width=width,
        input_height=height,
        train_frames=len(train_frames),
        test_frames=len(test_frames),
        train_boxes=sum(len(f.boxes) for f in train_frames),
        test_boxes=sum(len(f.boxes) for f in test_frames),
        holdout_cameras=sorted(holdout_cameras),
        weights_path=str(weights_path),
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    order = np.arange(len(train_frames))
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        started = time.perf_counter()
        running = 0.0

        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            images, heat, size, offset, mask = _augment(
                tr_images[index],
                tr_heat[index],
                tr_size[index],
                tr_offset[index],
                tr_mask[index],
                rng,
            )
            # uint8 to float on the device, so the CPU never holds a float32 batch.
            x = torch.from_numpy(np.ascontiguousarray(images)).to(device).float().div_(255.0)
            if rng.random() < 0.8:
                brightness = float(rng.uniform(0.65, 1.35))
                contrast = float(rng.uniform(0.8, 1.25))
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                x = ((x * brightness - mean) * contrast + mean).clamp_(0.0, 1.0)
            heat_t = torch.from_numpy(np.ascontiguousarray(heat)).to(device)
            size_t = torch.from_numpy(np.ascontiguousarray(size)).to(device)
            offset_t = torch.from_numpy(np.ascontiguousarray(offset)).to(device)
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(device)

            pred_heat, pred_size, pred_offset = model(x)
            loss = (
                focal_loss(pred_heat, heat_t)
                + 0.1 * masked_l1(pred_size, size_t, mask_t)
                + masked_l1(pred_offset, offset_t, mask_t)
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            running += loss.detach().item()

        mean_loss = running / steps_per_epoch
        report.losses.append(round(mean_loss, 5))
        log.info(
            "epoch %d/%d loss %.4f (%.1fs)",
            epoch + 1,
            epochs,
            mean_loss,
            time.perf_counter() - started,
        )

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)

    # ------------------------------------------------------------------ evaluation
    model.eval()
    tp = fp = fn = 0
    errors: list[float] = []
    per_class_tp: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    with torch.inference_mode():
        for start in range(0, len(test_frames), batch_size):
            chunk = test_frames[start : start + batch_size]
            x = (
                torch.from_numpy(np.ascontiguousarray(te_images[start : start + batch_size]))
                .float()
                .div_(255.0)
            )
            # build_model applies sigmoid to the heatmap inside forward(), so the model
            # already hands back probabilities. Squashing them again puts every cell in
            # the 0.50 to 0.73 band, which is above any sensible decode threshold and
            # turns the whole grid into detections.
            heat, size, offset = model(x.to(device))
            heat = heat.cpu().numpy()
            size = size.cpu().numpy()
            offset = offset.cpu().numpy()

            for i, frame in enumerate(chunk):
                predicted = decode(heat[i], size[i], offset[i])
                hit, miss_fp, miss_fn, mae = match(predicted, frame.boxes)
                tp += hit
                fp += miss_fp
                fn += miss_fn
                if mae == mae:  # not NaN
                    errors.append(mae)
                # Per class, matched against only that class's truths and predictions.
                # Counting a hit whenever any truth shared the predicted class made the
                # rate exceed 1.0, which is how the bug announced itself.
                for cls in {b["class"] for b in frame.boxes} | {b["class"] for b in predicted}:
                    truths = [b for b in frame.boxes if b["class"] == cls]
                    preds = [b for b in predicted if b["class"] == cls]
                    hit_c, _, _, _ = match(preds, truths)
                    per_class_tp[cls] = per_class_tp.get(cls, 0) + hit_c
                    per_class_total[cls] = per_class_total.get(cls, 0) + len(truths)

    report.precision = tp / (tp + fp) if tp + fp else 0.0
    report.recall = tp / (tp + fn) if tp + fn else 0.0
    report.f1 = (
        2 * report.precision * report.recall / (report.precision + report.recall)
        if report.precision + report.recall
        else 0.0
    )
    report.box_mae_px = float(np.mean(errors)) if errors else 0.0
    report.per_class = {
        scenes.CLASS_NAMES[c]: round(per_class_tp.get(c, 0) / total, 3)
        for c, total in sorted(per_class_total.items())
        if total
    }
    if device == "cuda":
        report.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    return report


def write_report(report: RealTrainReport, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report.__dict__)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
