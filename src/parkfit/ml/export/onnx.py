"""Exporting the detector to ONNX, and proving the export did not change it.

An export is a translation between two implementations, and translations go wrong
quietly. A fused batch-norm with the wrong epsilon, an interpolate that resolves to a
different rounding rule, a softplus approximated where it was exact: none of these raise,
and every one of them shifts boxes by a few pixels in a way that only shows up as a
mysteriously worse accuracy number weeks later.

So exporting is not finished when the file is written. :func:`verify` runs the same
frames through PyTorch and through ONNX Runtime and compares the raw tensors, then
compares the decoded boxes. The tensor check catches numerical drift; the box check
catches the case where a small tensor difference lands either side of a decision
boundary and moves a detection.

The graph deliberately contains no decoding. Peak extraction stays in C++ because a
3x3 local-maximum test is trivial there, while expressing top-K-with-suppression in ONNX
operators produces a graph that is harder to read, slower on CPU, and pinned to a fixed
maximum detection count baked in at export time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parkfit.ml.datasets import scenes
from parkfit.ml.train import detector

log = logging.getLogger(__name__)

#: Opset 17 covers everything this graph uses and is old enough that any ONNX Runtime
#: from the last few years will load it.
OPSET = 17

INPUT_NAME = "image"
OUTPUT_NAMES = ("heatmap", "size", "offset")


@dataclass
class ExportReport:
    exported: bool = False
    reason: str = ""
    onnx_path: str = ""
    size_kb: float = 0.0
    opset: int = OPSET
    max_abs_diff: dict[str, float] = field(default_factory=dict)
    max_rel_diff: dict[str, float] = field(default_factory=dict)
    frames_checked: int = 0
    boxes_pytorch: int = 0
    boxes_onnx: int = 0
    box_max_shift_px: float = 0.0
    agrees: bool = False

    def describe(self) -> str:
        if not self.exported:
            return f"not exported: {self.reason}"
        diffs = ", ".join(
            f"{k} {v:.1e} abs / {self.max_rel_diff.get(k, 0.0):.1e} rel"
            for k, v in sorted(self.max_abs_diff.items())
        )
        verdict = "agree" if self.agrees else "DISAGREE"
        return (
            f"{self.onnx_path} ({self.size_kb:.0f} KB, opset {self.opset})\n"
            f"  PyTorch vs ONNX Runtime over {self.frames_checked} frames: {verdict}\n"
            f"  max |difference| per output: {diffs}\n"
            f"  boxes {self.boxes_pytorch} vs {self.boxes_onnx}, "
            f"largest corner shift {self.box_max_shift_px:.4f} px"
        )


def export(
    weights_path: Path = detector.DEFAULT_WEIGHTS,
    onnx_path: Path = detector.DEFAULT_ONNX,
    *,
    dataset_root: Path = Path("data/detector"),
    verify_frames: int = 12,
    tolerance: float = 1e-5,
) -> ExportReport:
    """Write the ONNX graph and check it against PyTorch."""
    try:
        import torch
    except ImportError:
        return ExportReport(reason="pytorch is not installed")
    if not weights_path.exists():
        return ExportReport(reason=f"no weights at {weights_path}; run `pf detect train`")

    model = detector.build_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, scenes.INPUT_HEIGHT, scenes.INPUT_WIDTH, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        opset_version=OPSET,
        # Batch is dynamic; height and width are not. The worker feeds one frame at a
        # time, and fixing the spatial dims lets the runtime pick its kernels once
        # instead of re-planning on every call.
        dynamic_axes={INPUT_NAME: {0: "batch"}, **{n: {0: "batch"} for n in OUTPUT_NAMES}},
        do_constant_folding=True,
        # The TorchScript exporter rather than the dynamo one. This graph is a plain
        # convolutional stack with no control flow, so the newer path buys nothing here
        # and costs an onnxscript dependency. Passed explicitly because the default
        # flipped in torch 2.9 and a silent switch would change the emitted graph.
        dynamo=False,
    )

    report = ExportReport(
        exported=True,
        onnx_path=str(onnx_path),
        size_kb=onnx_path.stat().st_size / 1024.0,
    )

    _write_sidecar(onnx_path, weights_path)
    verify(model, onnx_path, dataset_root, report, frames=verify_frames, tolerance=tolerance)
    return report


def _write_sidecar(onnx_path: Path, weights_path: Path) -> None:
    """Record what the graph expects, next to the graph.

    The C++ side reads this to learn the input size and class order rather than having
    them compiled in, so retraining with a different input size does not silently feed
    the model a wrongly-scaled image.
    """
    trained = {}
    meta_path = weights_path.with_suffix(".json")
    if meta_path.exists():
        trained = json.loads(meta_path.read_text(encoding="utf-8"))

    onnx_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_version": detector.MODEL_VERSION,
                "input_name": INPUT_NAME,
                "output_names": list(OUTPUT_NAMES),
                "input_width": scenes.INPUT_WIDTH,
                "input_height": scenes.INPUT_HEIGHT,
                "output_stride": scenes.OUTPUT_STRIDE,
                "class_names": list(scenes.CLASS_NAMES),
                "opset": OPSET,
                "val_f1": trained.get("val_f1"),
                "val_precision": trained.get("val_precision"),
                "val_recall": trained.get("val_recall"),
                "val_box_mae_px": trained.get("val_box_mae_px"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def verify(
    model,
    onnx_path: Path,
    dataset_root: Path,
    report: ExportReport,
    *,
    frames: int = 12,
    tolerance: float = 1e-5,
) -> None:
    """Run the same frames through both runtimes and compare.

    Tolerance is **relative to each output's own scale**, not one absolute number across
    all three. The heatmap and the offset live in [0, 1] while the size head emits box
    dimensions in input pixels running to a couple of hundred, so a single absolute
    threshold is either far too tight for the size head or meaningless for the other two.
    The first version of this check used 1e-4 absolute and reported a mismatch on a size
    difference of 2.9e-4, which is a relative error of one and a half parts per million,
    while the decoded boxes were identical to the last decimal place.
    """
    try:
        import onnxruntime as ort
        import torch
    except ImportError:
        report.reason = "onnxruntime is not installed; export written but unverified"
        return

    if not (dataset_root / "labels.json").exists():
        report.reason = "no dataset to verify against"
        return

    _, val = scenes.load(dataset_root)
    images = val.images()
    count = min(frames, len(val))
    if count == 0:
        report.reason = "validation split is empty"
        return

    raw = np.stack([images[val.index_offset + i] for i in range(count)]).astype(np.float32)
    batch = np.transpose(raw / 255.0, (0, 3, 1, 2)).astype(np.float32)

    with torch.no_grad():
        torch_out = [t.numpy() for t in model(torch.from_numpy(batch))]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(list(OUTPUT_NAMES), {INPUT_NAME: batch})

    report.frames_checked = count
    worst = 0.0
    for name, a, b in zip(OUTPUT_NAMES, torch_out, onnx_out, strict=True):
        diff = float(np.max(np.abs(a - b)))
        scale = max(1e-6, float(np.max(np.abs(a))))
        report.max_abs_diff[name] = diff
        report.max_rel_diff[name] = diff / scale
        worst = max(worst, diff / scale)

    # Tensors agreeing is necessary but not sufficient. A difference of 1e-6 on a score
    # sitting exactly at the threshold changes whether a box exists at all, so the
    # decoded output is compared too.
    shift = 0.0
    total_torch = total_onnx = 0
    for i in range(count):
        boxes_a = detector.decode(torch_out[0][i], torch_out[1][i], torch_out[2][i])
        boxes_b = detector.decode(onnx_out[0][i], onnx_out[1][i], onnx_out[2][i])
        total_torch += len(boxes_a)
        total_onnx += len(boxes_b)
        for a, b in zip(
            sorted(boxes_a, key=lambda d: (d["class"], d["x1"])),
            sorted(boxes_b, key=lambda d: (d["class"], d["x1"])),
            strict=False,
        ):
            shift = max(
                shift,
                abs(a["x1"] - b["x1"]),
                abs(a["y1"] - b["y1"]),
                abs(a["x2"] - b["x2"]),
                abs(a["y2"] - b["y2"]),
            )

    report.boxes_pytorch = total_torch
    report.boxes_onnx = total_onnx
    report.box_max_shift_px = shift
    report.agrees = worst <= tolerance and total_torch == total_onnx and shift <= 0.5
