"""The occupancy classifier: MobileNetV3 on real parking-space crops.

Small on purpose. This runs once per bay per frame in the C++ worker, and a bay-level
question over a few hundred spaces has to fit in the gap between two frames, so the
budget is a couple of million parameters rather than a couple of hundred.

**The metric that matters is not accuracy.** It is the false-free rate: how often the
model says a space is empty when a car is in it. A classifier that calls everything
occupied scores well on a balanced set and is useless, while one that occasionally
invents an empty space sends a driver across a city for nothing. The whole product is
built to be wrong in the safe direction, so this reports the unsafe direction separately
and does not average it away.

Thresholding follows from that. 0.5 is the default only because it has to be something;
the trainer sweeps the threshold on the validation split and reports the operating point
that holds false-free under the target, which is the number the worker should actually
run at.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parkfit.ml.datasets import occupancy

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path("data/models/occupancy_cnn.pt")

#: The specification's ceiling for calling an occupied space free.
FALSE_FREE_TARGET = 0.02


@dataclass
class OccupancyReport:
    trained: bool = False
    reason: str = ""
    device: str = ""
    protocol: str = ""
    epochs: int = 0
    parameters: int = 0
    train_patches: int = 0
    test_patches: int = 0
    holdout: list[str] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    false_free_rate: float = 0.0
    threshold: float = 0.5
    accuracy_at_threshold: float = 0.0
    per_weather: dict = field(default_factory=dict)
    per_camera: dict = field(default_factory=dict)
    peak_vram_mb: float = 0.0
    weights_path: str = ""

    def describe(self) -> str:
        if not self.trained:
            return f"not trained: {self.reason}"
        return (
            f"{self.parameters:,} parameters on {self.device}, {self.epochs} epochs, "
            f"{self.protocol} split\n"
            f"  train {self.train_patches:,} patches, test {self.test_patches:,} on "
            f"{', '.join(self.holdout) or 'the official held-out days'}\n"
            f"  accuracy {self.accuracy:.4f}  precision {self.precision:.4f}  "
            f"recall {self.recall:.4f}  F1 {self.f1:.4f}  AUC {self.auc:.4f}\n"
            f"  false-free {self.false_free_rate:.4f} at threshold {self.threshold:.2f} "
            f"(accuracy there {self.accuracy_at_threshold:.4f}), "
            f"peak VRAM {self.peak_vram_mb:.0f} MB"
        )


def build_model(width_mult: str = "small"):
    """MobileNetV3 with ImageNet weights and a two-class head.

    The trunk is pretrained because a parking space crop is a natural image and the
    first layers of an ImageNet model already know edges, glass and shadow. Training
    those from scratch on 94,000 crops of nine car parks would spend most of the budget
    relearning what a photograph looks like.
    """
    import torch
    from torch import nn
    from torchvision.models import (
        MobileNet_V3_Small_Weights,
        mobilenet_v3_small,
    )

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    # A small positive bias on the occupied logit starts the model slightly pessimistic,
    # which is the direction this product is allowed to be wrong in.
    with torch.no_grad():
        model.classifier[-1].bias.copy_(torch.tensor([0.0, 0.25]))
    return model


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U as AUC, with tie-averaged ranks. No sklearn dependency."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pick_threshold(scores: np.ndarray, labels: np.ndarray, target: float) -> tuple[float, float]:
    """Lowest threshold whose false-free rate clears the target, and its accuracy.

    Swept on validation, never on test. Choosing an operating point by looking at the
    test set is how a model reports a false-free rate it will not reproduce in the street.
    """
    best = (0.5, 0.0)
    for threshold in np.arange(0.05, 0.96, 0.01):
        predicted = (scores >= threshold).astype(np.int64)
        occupied = labels == 1
        if occupied.sum() == 0:
            continue
        false_free = float((predicted[occupied] == 0).mean())
        if false_free <= target:
            accuracy = float((predicted == labels).mean())
            if accuracy > best[1]:
                best = (float(threshold), accuracy)
    return best


def train(
    root: Path = occupancy.DEFAULT_ROOT,
    *,
    protocol: str = "official",
    holdout: set[str] | None = None,
    epochs: int = 6,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    workers: int = 4,
    size: int = occupancy.PATCH_SIZE,
    device: str = "cuda",
    weights_path: Path = DEFAULT_WEIGHTS,
    seed: int = 11,
) -> OccupancyReport:
    """Fit the occupancy classifier and score it on held-out days, cameras or weather."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError:
        return OccupancyReport(reason="pytorch is not installed")

    splits_dir = root / "splits" / "CNRPark-EXT"
    patches_root = root / "PATCHES"
    if not splits_dir.exists() or not patches_root.exists():
        return OccupancyReport(reason=f"CNRPark-EXT not under {root}; run `pf occupancy fetch`")

    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA unavailable, training on CPU")
        device = "cpu"

    torch.manual_seed(seed)

    if protocol == "official":
        train_patches = occupancy.read_split(splits_dir / "train.txt", patches_root)
        val_patches = occupancy.read_split(splits_dir / "val.txt", patches_root)
        test_patches = occupancy.read_split(splits_dir / "test.txt", patches_root)
        holdout_names = ["official test days"]
    else:
        every = occupancy.read_split(splits_dir / "all.txt", patches_root)
        if protocol == "camera":
            holdout = holdout or {"camera8", "camera9"}
            kept, test_patches = occupancy.split_by_camera(every, holdout)
        elif protocol == "weather":
            holdout = holdout or {"rainy"}
            kept, test_patches = occupancy.split_by_weather(every, holdout)
        else:
            return OccupancyReport(reason=f"unknown protocol {protocol!r}")
        # A tenth of the training cameras is kept back to choose the threshold on.
        rng = np.random.default_rng(seed)
        mask = rng.random(len(kept)) < 0.1
        val_patches = [p for p, m in zip(kept, mask, strict=True) if m]
        train_patches = [p for p, m in zip(kept, mask, strict=True) if not m]
        holdout_names = sorted(holdout)

    if not train_patches or not test_patches:
        return OccupancyReport(
            reason=f"split left {len(train_patches)} train / {len(test_patches)} test"
        )

    log.info(
        "train %d, val %d, test %d patches",
        len(train_patches),
        len(val_patches),
        len(test_patches),
    )

    loaders = {
        name: DataLoader(
            occupancy.build_dataset(items, train=(name == "train"), size=size),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=workers,
            pin_memory=(device == "cuda"),
            persistent_workers=workers > 0,
        )
        for name, items in (("train", train_patches), ("val", val_patches), ("test", test_patches))
        if items
    }

    model = build_model().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate * 4, total_steps=epochs * max(1, len(loaders["train"]))
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    report = OccupancyReport(
        trained=True,
        device=device,
        protocol=protocol,
        epochs=epochs,
        parameters=parameters,
        train_patches=len(train_patches),
        test_patches=len(test_patches),
        holdout=holdout_names,
        weights_path=str(weights_path),
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        running, steps = 0.0, 0
        for images, labels in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            running += loss.detach().item()
            steps += 1
        mean = running / max(1, steps)
        report.losses.append(round(mean, 5))
        log.info(
            "epoch %d/%d loss %.4f (%.0fs)",
            epoch + 1,
            epochs,
            mean,
            time.perf_counter() - started,
        )

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)

    def score(loader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        probabilities, truth = [], []
        with torch.inference_mode():
            for images, labels in loader:
                logits = model(images.to(device, non_blocking=True))
                probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                truth.append(labels.numpy())
        return np.concatenate(probabilities), np.concatenate(truth)

    if "val" in loaders:
        val_scores, val_labels = score(loaders["val"])
        report.threshold, _ = _pick_threshold(val_scores, val_labels, FALSE_FREE_TARGET)

    scores, labels = score(loaders["test"])
    predicted = (scores >= 0.5).astype(np.int64)

    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())

    report.accuracy = float((predicted == labels).mean())
    report.precision = tp / (tp + fp) if tp + fp else 0.0
    report.recall = tp / (tp + fn) if tp + fn else 0.0
    report.f1 = (
        2 * report.precision * report.recall / (report.precision + report.recall)
        if report.precision + report.recall
        else 0.0
    )
    report.auc = _rank_auc(scores, labels)

    at_threshold = (scores >= report.threshold).astype(np.int64)
    occupied = labels == 1
    report.false_free_rate = float((at_threshold[occupied] == 0).mean()) if occupied.any() else 0.0
    report.accuracy_at_threshold = float((at_threshold == labels).mean())

    by_weather: dict[str, list] = {}
    by_camera: dict[str, list] = {}
    for patch, ok in zip(test_patches, (at_threshold == labels), strict=True):
        by_weather.setdefault(patch.weather, []).append(bool(ok))
        by_camera.setdefault(patch.camera, []).append(bool(ok))
    report.per_weather = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_weather.items())}
    report.per_camera = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_camera.items())}

    if device == "cuda":
        report.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    return report


def write_report(report: OccupancyReport, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(report.__dict__), indent=2), encoding="utf-8")
    return destination
