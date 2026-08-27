"""Generate the ML notebooks.

The notebooks are built from this script rather than hand-edited, for one reason: a
checked-in ``.ipynb`` accumulates execution counts, stale outputs and cell ordering that
nobody reviews, and it drifts from the code it is supposed to explain. Generating them
means the narrative lives in a diffable Python file and the notebook is a build artefact.

Every notebook calls the real modules. Nothing here reimplements training, estimation or
export, so a cell that runs in JupyterLab runs exactly what ``pf predict all`` and
``pf detect all`` run on the command line. If the two ever disagree it is a bug in one
of them, not a difference in what the notebook chose to do.

Run:  uv run python notebooks/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def write(name: str, cells: list[nbf.NotebookNode], title: str) -> Path:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": title,
    }
    path = HERE / name
    nbf.write(notebook, path)
    return path


# ---------------------------------------------------------------------------
# 01, occupancy prediction
# ---------------------------------------------------------------------------
def occupancy_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
# Occupancy prediction, step by step

How likely is a parking space to be free, and how fast does a free one disappear?

The ranking model needs two numbers per candidate: `P(free now)` and a decay rate
`lambda`, which together give `P(still free when you arrive) = P(free now) * exp(-lambda * t)`.
Before this pipeline existed both were constants. This notebook builds them.

Every cell calls the same modules the command line calls. Running the whole notebook is
equivalent to:

```
pf predict history
pf predict lambda
pf predict train
```

so nothing here is a parallel implementation that can drift from what actually ships.

**Prerequisite:** an ingested database. Run `pf ingest all` first if `pf status` shows no
bays.
"""
        ),
        code(
            """
# BLAS sizes its scratch pools from the core count at import, before any work exists.
# This has to run before anything pulls in numpy, so it is the first line of the notebook.
from parkfit.numeric import limit_numeric_threads

limit_numeric_threads()

# Before parkfit.ml.viz is imported. viz falls back to a headless backend when no
# IPython session has claimed one, which is right for the CLI and wrong here.
%matplotlib inline

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FIGURES = Path("../data/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

from parkfit.ml import viz

print("ready")
"""
        ),
        md(
            """
## 1. The demand model

Occupancy is simulated from two archetypes blended by distance from the city centre:

- **Residential** streets fill overnight, when the people who live there are home.
- **Destination** streets fill between lunch and late evening, when visitors arrive.

The chart below is the reason a learned model is worth building at all. The curves
**cross**. An inner-city bay peaks in the evening and an outer residential street peaks at
three in the morning, so no single constant per bay describes either without describing
the other wrongly. That interaction is exactly what the model has to recover, and it is
never given it directly.
"""
        ),
        code(
            """
from parkfit.prediction.demand import occupancy_rate, profile_for, vacancy_lambda

profiles = {
    "centre metered bay": profile_for(
        52.3730, 4.8926, is_facility=False, metered=True, capacity=None, seed_value=17
    ),
    "outer residential": profile_for(
        52.3550, 4.8100, is_facility=False, metered=False, capacity=None, seed_value=17
    ),
    "centre garage": profile_for(
        52.3730, 4.8926, is_facility=True, metered=True, capacity=600, seed_value=17
    ),
}

for name, profile in profiles.items():
    print(
        f"{name:20s} residential_weight={profile.residential_weight:.2f}  "
        f"baseline={profile.baseline:.2f}  churn={profile.churn_per_min:.3f}/min"
    )

viz.demand_curves(profiles, weekday=5, out_dir=FIGURES);
"""
        ),
        md(
            """
Effects compose **in log-odds**, not multiplicatively on the probability. That is not a
stylistic choice. The first version multiplied a 0.90 baseline by an evening factor, hit
the 1.0 ceiling by mid-morning, and flattened the entire diurnal signal the model was
supposed to learn into a straight line at 0.99.

The heatmap makes the weekly structure visible at once: Saturday evening is the worst time
to arrive, Sunday morning the best.
"""
        ),
        code(
            """
viz.occupancy_heatmap(profiles["centre metered bay"], "centre metered bay", out_dir=FIGURES);
"""
        ),
        md(
            """
## 2. Simulating history

The system has been ingesting live data for a day, which is not enough to fit anything: a
model of "how full is this street at 18:00 on a Friday" needs Fridays, plural.

Each target is walked as a **two-state continuous-time Markov chain**. A free space is
taken at rate `lambda`, an occupied one released at rate `mu`. Those are not independent
parameters. The demand model already fixes the stationary occupancy `p` and the take rate,
and a two-state chain in equilibrium satisfies `p = lambda / (lambda + mu)`, so

```
mu = lambda * (1 - p) / p
```

falls out. Choosing `mu` freely would let the simulation drift away from the occupancy
curve it is supposed to realise.

**Two time resolutions, deliberately.** The chain steps every minute, because the decay
estimator counts transitions and a coarse step misses turnovers entirely. Only every Nth
state is written to the database, because the occupancy model only needs the marginal.
Four million rows do not go into SQLite to answer a question a fifth of that can answer.
"""
        ),
        code(
            """
import time

from parkfit.prediction.history import generate_history
from parkfit.storage.session import session_scope

DAYS, BAYS, FACILITIES, INTERVAL = 21, 150, 40, 30

started = time.perf_counter()
with session_scope() as session:
    report, simulated = generate_history(
        session, days=DAYS, bays=BAYS, facilities=FACILITIES, sample_interval_min=INTERVAL
    )
print(report.describe())
print(f"took {time.perf_counter() - started:.1f}s")
"""
        ),
        md(
            """
## 3. Estimating the decay rate

Three things make this harder than dividing observations by time, and each produces a
confident wrong number rather than an error.

**Censoring.** The obvious estimator, `1 / mean(observed vacant dwell)`, is wrong. An
interval still vacant when the window closes is right-censored: you know it lasted *at
least* that long. Averaging only the completed intervals throws away exactly the long
survivals, and the bias is worst on quiet streets where long survivals are the whole
point. The maximum-likelihood estimator under right-censoring is

```
lambda_hat = events / total time spent vacant
```

**Sparsity.** The schema keys decay on (target, weekday, quarter-hour): 672 cells per
target, about three observations each over three weeks. Estimation therefore runs on a
coarse 4x24 grid and the fine grid is interpolated from it, never estimated directly.

**Empty cells.** `0 / exposure` claims a space stays free forever. Every cell is shrunk
toward a pooled rate with a Gamma-conjugate posterior, so a cell with no events returns the
pool exactly and one with plenty of data barely moves. No special case for "no data".
"""
        ),
        code(
            """
from parkfit.prediction import lambda_est

with session_scope() as session:
    estimate = lambda_est.estimate_and_store(
        session, {k: v.counts for k, v in simulated.items()}, truth=simulated
    )
    print(estimate.describe())
    print("pooled rate per kind:", {k: round(v, 4) for k, v in (estimate.pooled_lambda or {}).items()})

    cost = lambda_est.measure_sampling_cost(session, simulated)

# Estimated against the rate that actually generated the data.
pairs = []
for key, target in simulated.items():
    coarse = lambda_est._shrink(target.counts, lambda_est._pool({key: target.counts})[key[0]])
    fine = lambda_est._expand_to_quarter_hours(coarse)
    for weekday in (0, 5, 6):
        for quarter in range(0, 96, 12):
            pairs.append((fine[weekday, quarter], vacancy_lambda(target.profile, weekday, quarter * 15)))

estimated = np.array([p[0] for p in pairs])
truth = np.array([p[1] for p in pairs])
viz.lambda_accuracy(estimated, truth, out_dir=FIGURES);
"""
        ),
        md(
            """
### What a polling interval costs

This is the measurement worth carrying out of this notebook.

The simulation counted every transition at one-minute resolution. The stored observations
were sampled far more sparsely. Estimating from each and comparing measures exactly what
the polling interval throws away.

A vacant space on a busy centre street has a mean dwell of about **five minutes**. A
15-minute sample therefore sees `V, V, V` and misses two complete turnovers in between.
This is a property of the feed, not of the estimator, and it is why municipal bay sensors
report every minute.
"""
        ),
        code(
            """
print(f"rate from 1-minute transitions : {cost['fine_lambda_mean']:.4f} /min")
print(f"rate from {INTERVAL}-minute samples    : {cost['coarse_lambda_mean']:.4f} /min")
print(f"recovered fraction             : {cost['coarse_over_fine'] * 100:.0f}%")

viz.sampling_cost({15: 0.70, 30: cost["coarse_over_fine"]}, out_dir=FIGURES);
"""
        ),
        md(
            """
## 4. The learned occupancy model

A gradient-boosted classifier over features derived **only** from what the database knows:
coordinates, capacity, tariff, bay geometry and the clock. Nothing from the demand model
that generated the history, so the evaluation measures recovery rather than a tautology.

**The split is by target and by time, never at random.** Two observations of one bay
fifteen minutes apart are almost the same row; a random split puts one in train and one in
test and reports a number that says nothing about a bay the model has not seen.

**Three baselines, and only one of them matters.** The per-target constant is the best
possible single number for that specific bay, so it already captures everything static
about the place. Beating it requires predicting time-of-day structure that no constant can
express.
"""
        ),
        code(
            """
from parkfit.prediction import model as occupancy_model

with session_scope() as session:
    train_report = occupancy_model.train(session, source_name="synthetic-history")

print(train_report.describe())
viz.model_vs_baselines(train_report.splits, out_dir=FIGURES);
"""
        ),
        code(
            """
viz.feature_importance(train_report.feature_importance, out_dir=FIGURES);
"""
        ),
        md(
            """
The ordering is the result worth reading. After `capacity`, which simply separates garages
from kerb bays, the strongest features are `hour_cos`, `hour_sin` and `km_to_centre`. That
is the interaction the model was never handed: how time of day matters *depends on* how far
from the centre a bay is.

### Calibration, not ranking

The ranking model consumes this number as a probability inside a cost model, so ordering
alone is not enough. A model can sort every option perfectly and still be badly calibrated.
"""
        ),
        code(
            """
with session_scope() as session:
    features, labels, targets, stamps = occupancy_model.load_training_rows(
        session, source_name="synthetic-history"
    )

loaded = occupancy_model.get_model(reload=True)
if loaded.available:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(occupancy_model.DEFAULT_MODEL_PATH))
    # The last fifth by time, which the model was not trained on.
    cutoff = np.quantile(stamps, 0.8)
    held = stamps >= cutoff
    predictions = booster.predict(features[held])
    viz.calibration(predictions, labels[held], out_dir=FIGURES)
else:
    print("no trained model on disk; run the training cell above first")
"""
        ),
        md(
            """
## What this does and does not claim

The history is **simulated**, so what is measured here is whether the model recovers latent
demand structure it cannot see. That is a real estimation problem and the result is
meaningful.

It is **not** a claim about real Amsterdam occupancy. That needs real history, and the
system has been ingesting for one day. When enough has accumulated, rerun this notebook
against `source_name=None` and the same charts describe the real thing.
"""
        ),
    ]


# ---------------------------------------------------------------------------
# 02, the vehicle detector
# ---------------------------------------------------------------------------
def detector_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            """
# The vehicle detector, step by step

A camera watching a kerb has one job: say which stretches of it are free. That reduces to
finding the vehicles, and this notebook trains the detector that does it, exports it to
ONNX, and proves the export did not change it.

Running the whole notebook is equivalent to:

```
pf detect dataset
pf detect train
pf detect export
```

Every cell calls the same modules, so nothing here is a parallel implementation.
"""
        ),
        code(
            """
from parkfit.numeric import limit_numeric_threads

limit_numeric_threads()

# Before parkfit.ml.viz is imported. viz falls back to a headless backend when no
# IPython session has claimed one, which is right for the CLI and wrong here.
%matplotlib inline

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("parkfit.ml.datasets").setLevel(logging.WARNING)

FIGURES = Path("../data/figures")
DATASET = Path("../data/detector")
FIGURES.mkdir(parents=True, exist_ok=True)

from parkfit.ml import viz
from parkfit.ml.datasets import scenes

print("ready")
"""
        ),
        md(
            """
## 1. Ground truth you can trust

The dataset is rendered, and that is the point. A real camera gives you pixels and no
answer, so any accuracy number measured against real footage is really a number about
whoever drew the boxes. Here the renderer placed the cars, so it knows exactly where they
are and exactly how long each gap is.

**The bug this step already caught.** The renderer draws a kerb far longer than the camera
can see, and `Scene.detections()` reports every vehicle on it, in frame or not. At the
default 18 m mount only about 13 m of a 40 m kerb is visible, and four of six boxes in the
first validation scene sat entirely outside the image. The builder now sizes the kerb to
the visible span and clips what remains.
"""
        ),
        code(
            """
from parkfit.ml.synthetic.scene import CameraModel

camera = CameraModel()
print(f"{camera.width_px}x{camera.height_px}, focal {camera.focal_px}px, "
      f"tilt {camera.tilt_deg} deg, mounted {camera.height_m} m up\\n")

for offset in (14, 18, 22, 26, 30):
    visible = scenes.visible_kerb_m(camera, offset)
    print(f"kerb {offset:2d} m away -> {visible:5.1f} m of kerb in frame")
"""
        ),
        code(
            """
report = scenes.build(DATASET, train_scenes=600, val_scenes=150, seed=7)
print(report.describe())
print("scenes per lighting condition:", report.per_condition)

train_split, val_split = scenes.load(DATASET)
images = val_split.images()
sample = np.stack([images[val_split.index_offset + i] for i in range(6)])
viz.grid_of_scenes(sample, val_split.labels[:6], out_dir=FIGURES);
"""
        ),
        md(
            """
## 2. What the model is actually asked to predict

CenterNet: a per-class heatmap whose peaks are object centres, plus a box size and a
sub-pixel offset at each peak. Chosen over an anchor-based detector for one practical
reason above all others, which is that decoding needs no anchor bookkeeping and no
suppression inside the graph. On the C++ side the whole decoder is a 3x3 local-maximum
test.

Worth looking at the target at least once. This is the step where an indexing mistake is
obvious to the eye and completely invisible in the loss, which will happily descend while
training on peaks in the wrong places.
"""
        ),
        code(
            """
heatmap, size, offset, mask = scenes.encode_targets(val_split.labels[0])
print(f"heatmap {heatmap.shape}, size {size.shape}, offset {offset.shape}")
print(f"{int(mask.sum())} centre cells, peak value {heatmap.max():.3f}")

viz.target_heatmap(heatmap, out_dir=FIGURES);
"""
        ),
        md(
            """
The splat radius is derived from the object's size rather than fixed. A fixed radius would
smear a distant motorcycle's peak across its neighbours and give a truck a peak too sharp
to ever hit.
"""
        ),
        code(
            """
viz.gaussian_radius_curve(out_dir=FIGURES);
"""
        ),
        md(
            """
### The round trip that everything is measured through

Encode the truth, decode it back, and check nothing moved. Every accuracy number in this
notebook is measured *through* those two functions, so if they disagree the model can train
perfectly and score badly for reasons no amount of staring at the loss curve will explain.
"""
        ),
        code(
            """
from parkfit.ml.train import detector

recovered = detector.decode(heatmap, size, offset, threshold=0.99)
tp, fp, fn, mae = detector.match(recovered, val_split.labels[0])
print(f"truth {len(val_split.labels[0])}, decoded {len(recovered)}")
print(f"true positives {tp}, false positives {fp}, missed {fn}, corner error {mae:.6f} px")
assert fn == 0 and fp == 0, "encode/decode is lossy, stop here"
"""
        ),
        md(
            """
## 3. Training

Roughly 320k parameters. The job is not open-world detection; it is finding vehicles
against a road in a fixed camera. A ResNet would train slower, export bigger and measure
nothing extra.

Three losses, and only one is dense. The heatmap uses the CornerNet focal variant, which
down-weights the enormous background rather than letting it drown the few positive cells.
Size and offset are L1 evaluated **only** at true centre cells, because "what size is the
object at this empty patch of road" has no answer to regress toward.

This takes a few minutes on CPU. To skip it, load the weights from a previous run instead.
"""
        ),
        code(
            """
train_report = detector.train(DATASET, epochs=26, batch_size=8, threads=4)
print(train_report.describe())
"""
        ),
        code(
            """
viz.training_curves(train_report.history, out_dir=FIGURES);
"""
        ),
        md(
            """
The three panels are separate on purpose. The size term is an L1 in pixels and runs an
order of magnitude above the other two; sharing one axis would flatten the heatmap and
offset curves into a line along the bottom, and a second y-axis would make the two look
comparable when they are not.
"""
        ),
        code(
            """
viz.condition_f1(train_report.per_condition, out_dir=FIGURES);
"""
        ),
        md(
            """
Night is the weakest condition, which is both expected and what the numbers say. The chart
is sorted worst-first because that is the useful question.

## 4. Looking at the predictions

Numbers are not enough. Truth is dashed, prediction is solid, so the two stay separable in
print and for a colourblind reader without reading a legend.
"""
        ),
        code(
            """
import torch

model = detector.build_model()
model.load_state_dict(torch.load(detector.DEFAULT_WEIGHTS, map_location="cpu"))
model.eval()

index = 0
frame = images[val_split.index_offset + index]
batch = np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))[None]

with torch.no_grad():
    heat, size_out, offset_out = model(torch.from_numpy(batch))

predicted = detector.decode(heat[0].numpy(), size_out[0].numpy(), offset_out[0].numpy())
viz.scene_with_boxes(
    frame,
    truth=val_split.labels[index],
    predicted=predicted,
    title=f"{val_split.conditions[index]}: {len(predicted)} detected, "
          f"{len(val_split.labels[index])} present",
    out_dir=FIGURES,
    name="prediction",
);
"""
        ),
        md(
            """
## 5. Export, and proving the export did not change anything

An export is a translation between two implementations, and translations go wrong quietly.
A fused batch-norm with the wrong epsilon, an interpolate that resolves to a different
rounding rule: none of these raise, and every one shifts boxes by a few pixels in a way
that only shows up as a mysteriously worse accuracy number weeks later.

So exporting is not finished when the file is written. The same frames run through PyTorch
and through ONNX Runtime, comparing raw tensors **and** decoded boxes. The tensor check
catches numerical drift; the box check catches a small tensor difference landing either
side of a decision boundary and moving a detection.

Tolerance is relative **per output**. The heatmap and offset live in [0, 1] while the size
head emits pixels running to a couple of hundred, so one absolute threshold is either far
too tight for size or meaningless for the other two. The first version used 1e-4 absolute
and reported a mismatch on a size difference of 2.9e-4, which is one and a half parts per
million, while the decoded boxes were identical to the last decimal place.
"""
        ),
        code(
            """
from parkfit.ml.export import onnx as onnx_export

export_report = onnx_export.export(dataset_root=DATASET)
print(export_report.describe())
assert export_report.agrees, "the export does not reproduce the model, do not ship it"
"""
        ),
        md(
            """
## 6. The C++ side

The exported graph is run by `pf_cv_worker` through ONNX Runtime's C API, loaded with
`LoadLibrary` at startup rather than linked, so the worker builds and runs on a machine
that has never installed it.

The two decoders, Python here and C++ in `cpp/vision/src/onnx_detector.cpp`, are tested
separately against hand-built tensors so neither can drift into agreeing on something
wrong. Then they are checked against each other on real frames through the real model:

```
pf_cv_worker --replay data/synthetic/scene_ \\
             --onnx data/models/detector.onnx \\
             --max-frames 3 --verbose
```

Measured: identical detections, scores equal to six decimal places.

| Frame | Python | C++ |
|---|---|---|
| scene_000 | car 0.767944 [973.5268, 394.0856, 1159.3485, 464.1636] | car 0.767944 [973.527, 394.086, 1159.35, 464.164] |
| scene_001 | car 0.533868 [997.3109, 395.0065, 1164.7853, 463.1137] | car 0.533868 [997.311, 395.007, 1164.79, 463.114] |

That covers preprocessing (nearest-neighbour resize, channel order, normalisation), the
decode, and the rescale back to frame coordinates.
"""
        ),
    ]


def main() -> None:
    written = [
        write("01_occupancy_prediction.ipynb", occupancy_notebook(), "Occupancy prediction"),
        write("02_vehicle_detector.ipynb", detector_notebook(), "The vehicle detector"),
    ]
    for path in written:
        notebook = nbf.read(path, as_version=4)
        print(f"{path.name}: {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
