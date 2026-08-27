"""Figures for the machine-learning pipelines.

This module exists so there is exactly one set of charts. The notebooks call these
functions and the command line calls the same ones with ``--figures``, which means a
figure someone screenshots from a notebook is the figure CI produced, rather than two
implementations that drift until they disagree about what the model did.

**Colour carries a job, never decoration.** Categorical hues are assigned in a fixed
order and never cycled, so a series keeps its colour when another is filtered away.
Magnitude uses one hue light to dark; a rainbow ramp invents boundaries in continuous
data that are not there. Nothing here uses a second y-axis: two measures on two scales
in one frame is the chart mistake that most reliably produces a wrong conclusion, and the
answer is always two panels.

The palette is validated rather than eyeballed, against colour-vision deficiency and
against the surface. Two of the hues sit below 3:1 contrast on white, which is why every
chart here carries direct labels rather than relying on colour alone to be read.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

# Fixed categorical order. A fifth series takes slot 5, never a generated hue.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948")
#: One hue, light to dark, for magnitude.
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281")

INK = "#1a1a19"
INK_SOFT = "#5c5c57"
INK_FAINT = "#8f8f88"
SURFACE = "#fcfcfb"
GRID = "#e6e6e1"


def _plt():
    """Import matplotlib lazily, headless on the command line and inline in a notebook.

    The subtlety is worth spelling out, because getting it wrong is silent. The CLI writes
    PNGs on machines with no display, so it needs Agg. A notebook needs whatever backend
    IPython has already installed, or figures never appear: an earlier version forced Agg
    unconditionally and produced a notebook that executed without a single error and showed
    not one chart.

    So Agg is selected only when nothing else has claimed a backend and there is no IPython
    session to defer to.
    """
    import matplotlib

    in_notebook = False
    try:
        from IPython import get_ipython

        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False

    if not in_notebook:
        matplotlib.use("Agg", force=False)

    import matplotlib.pyplot as plt

    return plt


def style(ax, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive frame. The data should be the only assertive thing in the figure."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)


def save(fig, out_dir: Path | None, name: str):
    """Write the figure when a directory is given, and hand it back either way."""
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------
# P4, occupancy prediction
# ---------------------------------------------------------------------------
def demand_curves(profiles: dict, weekday: int = 5, out_dir: Path | None = None):
    """Occupancy over a day for several target archetypes.

    The chart that shows why a per-target constant cannot work: the curves cross. An
    inner-city bay peaks in the evening and an outer residential street peaks overnight,
    so no single number describes either without describing the other wrongly.
    """
    from parkfit.prediction.demand import occupancy_rate

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    hours = np.arange(0, 24.01, 0.25)

    for index, (label, profile) in enumerate(profiles.items()):
        values = [occupancy_rate(profile, weekday, int(h * 60) % 1440) for h in hours]
        colour = SERIES[index % len(SERIES)]
        ax.plot(hours, values, color=colour, linewidth=2, label=label)
        # Direct label at the curve's own peak. Two of these hues fall below 3:1 on white,
        # so identity never rests on colour alone. The label flips to the left of the peak
        # once that peak is late in the day, because otherwise it runs off the axes, which
        # is what the first render of this chart did.
        peak = int(np.argmax(values))
        late = hours[peak] > 17.0
        ax.annotate(
            label,
            (hours[peak], values[peak]),
            textcoords="offset points",
            xytext=(-8 if late else 8, 7),
            ha="right" if late else "left",
            color=INK,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    style(
        ax,
        title=f"Occupancy through a {'Saturday' if weekday == 5 else 'weekday'}",
        xlabel="time of day",
        ylabel="probability occupied",
    )
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="lower left")
    fig.tight_layout()
    return save(fig, out_dir, "demand_curves")


def occupancy_heatmap(profile, label: str = "", out_dir: Path | None = None):
    """Weekday against hour, as one hue light to dark.

    Sequential rather than a rainbow: occupancy is a continuous magnitude, and a rainbow
    ramp invents banding at hue boundaries that the data does not have.
    """
    from matplotlib.colors import LinearSegmentedColormap

    from parkfit.prediction.demand import occupancy_table

    plt = _plt()
    table = occupancy_table(profile)
    hourly = table.reshape(7, 24, 60).mean(axis=2)

    cmap = LinearSegmentedColormap.from_list("parkfit_seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    image = ax.imshow(hourly, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="upper")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.grid(False)
    style(ax, title=f"Occupancy by weekday and hour{f': {label}' if label else ''}",
          xlabel="hour")

    bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.03)
    bar.set_label("probability occupied", color=INK_SOFT, fontsize=9)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    fig.tight_layout()
    return save(fig, out_dir, "occupancy_heatmap")


def lambda_accuracy(estimated: np.ndarray, truth: np.ndarray, out_dir: Path | None = None):
    """Estimated decay rate against the rate that generated the data.

    Points on the diagonal mean the estimator recovered what produced the history. The
    diagonal is drawn rather than implied, because a scatter without it invites the eye to
    fit its own line.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 5.2))

    limit = float(max(estimated.max(), truth.max())) * 1.05
    ax.plot([0, limit], [0, limit], color=INK_FAINT, linewidth=1.2, linestyle="--",
            label="exact recovery", zorder=1)
    ax.scatter(truth, estimated, s=14, color=SERIES[0], alpha=0.55, linewidths=0, zorder=2)

    mae = float(np.mean(np.abs(estimated - truth)))
    ax.annotate(
        f"mean |error| {mae:.4f} /min",
        (0.04, 0.94),
        xycoords="axes fraction",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    style(ax, title="Decay rate: estimated against truth",
          xlabel="true lambda (per minute)", ylabel="estimated lambda (per minute)")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="lower right")
    fig.tight_layout()
    return save(fig, out_dir, "lambda_accuracy")


def sampling_cost(recovered: dict[int, float], out_dir: Path | None = None):
    """What a polling interval costs the decay estimate.

    The finding this chart exists for: a vacant space on a busy street has a mean dwell of
    about five minutes, so a 15-minute sample sees the space still free and misses two
    complete turnovers in between. That is a property of the feed, not the estimator.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    intervals = sorted(recovered)
    values = [recovered[i] * 100 for i in intervals]
    bars = ax.bar([str(i) for i in intervals], values, color=SERIES[1], width=0.55)

    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.0f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(100, color=INK_FAINT, linewidth=1.2, linestyle="--")
    ax.annotate("one-minute sensor", (0.02, 0.93), xycoords="axes fraction",
                color=INK_SOFT, fontsize=9)
    ax.set_ylim(0, 118)
    style(ax, title="Fraction of the true decay rate a polling interval recovers",
          xlabel="polling interval (minutes)", ylabel="recovered (%)")
    fig.tight_layout()
    return save(fig, out_dir, "sampling_cost")


def model_vs_baselines(splits, out_dir: Path | None = None):
    """Brier score for the model against the baselines it has to beat.

    The per-target constant is the bar that matters. It already captures everything static
    about a bay, so beating it requires time-of-day structure that no constant can express.
    Lower is better, and the axis says so, because a reader should not have to remember it.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))

    labels = ["model", "flat prior", "per kind", "per target"]
    colours = [SERIES[0], INK_FAINT, SERIES[3], SERIES[1]]
    width = 0.2
    positions = np.arange(len(splits))

    for index, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        values = []
        for split in splits:
            if label == "model":
                values.append(split.model_brier)
            elif label == "flat prior":
                values.append(split.flat_prior_brier)
            elif label == "per kind":
                values.append(split.per_kind_brier)
            else:
                values.append(split.per_target_brier if split.per_target_brier else 0.0)

        offset = (index - 1.5) * width
        bars = ax.bar(positions + offset, values, width * 0.9, color=colour, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value <= 0:
                continue
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                color=INK_SOFT,
                fontsize=8,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([s.name for s in splits])
    style(ax, title="Occupancy model against its baselines", ylabel="Brier score (lower is better)")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, ncol=4, loc="upper left")
    fig.tight_layout()
    return save(fig, out_dir, "model_vs_baselines")


def feature_importance(importance: dict[str, float], top: int = 10,
                       out_dir: Path | None = None):
    """Which features the model actually leaned on.

    Horizontal because feature names are words, and rotating words to fit a vertical axis
    is a layout failing dressed up as a chart choice.
    """
    plt = _plt()
    ranked = sorted(importance.items(), key=lambda kv: kv[1])[-top:]
    names = [name for name, _ in ranked]
    values = [value for _, value in ranked]

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(names) + 1.4))
    ax.barh(names, values, color=SERIES[0], height=0.62)
    for index, value in enumerate(values):
        ax.annotate(f"{value:,.0f}", (value, index), textcoords="offset points",
                    xytext=(6, 0), va="center", color=INK_SOFT, fontsize=9)

    ax.set_xlim(0, max(values) * 1.16)
    style(ax, title="Feature importance (gain)", xlabel="total gain")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, out_dir, "feature_importance")


def calibration(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10,
                out_dir: Path | None = None):
    """Predicted probability against observed frequency.

    Calibration, not ranking. A model can order every option correctly and still be badly
    calibrated, and this product consumes the number as a probability in a cost model, so
    ordering alone is not enough.
    """
    plt = _plt()
    edges = np.linspace(0, 1, bins + 1)
    centres, observed, counts = [], [], []

    for lo, hi in itertools.pairwise(edges):
        mask = (probabilities >= lo) & (probabilities < hi)
        if mask.sum() < 5:
            continue
        centres.append(float(probabilities[mask].mean()))
        observed.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_FAINT, linewidth=1.2, linestyle="--",
            label="perfect calibration")
    ax.plot(centres, observed, color=SERIES[0], linewidth=2, marker="o", markersize=7,
            label="model")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style(ax, title="Calibration", xlabel="predicted probability occupied",
          ylabel="observed frequency")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="upper left")
    fig.tight_layout()
    return save(fig, out_dir, "calibration")


# ---------------------------------------------------------------------------
# P7, the detector
# ---------------------------------------------------------------------------
def scene_with_boxes(image: np.ndarray, truth: list[dict] | None = None,
                     predicted: list[dict] | None = None, title: str = "",
                     out_dir: Path | None = None, name: str = "scene"):
    """A rendered frame with truth and prediction drawn over it.

    Truth is a dashed outline and prediction is solid, so the two are separable in print
    and for a colourblind reader without reading the legend.
    """
    from matplotlib.patches import Rectangle

    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 10 * image.shape[0] / image.shape[1]))
    ax.imshow(image)

    for box in truth or []:
        ax.add_patch(Rectangle(
            (box["x1"], box["y1"]), box["x2"] - box["x1"], box["y2"] - box["y1"],
            fill=False, edgecolor="#ffffff", linewidth=2.4, linestyle="--"))

    for box in predicted or []:
        ax.add_patch(Rectangle(
            (box["x1"], box["y1"]), box["x2"] - box["x1"], box["y2"] - box["y1"],
            fill=False, edgecolor=SERIES[1], linewidth=2.2))
        ax.annotate(
            f"{box.get('label', '?')} {box.get('score', 0):.2f}",
            (box["x1"], box["y1"] - 4),
            color=SERIES[1],
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    fig.tight_layout()
    return save(fig, out_dir, name)


def training_curves(history, out_dir: Path | None = None):
    """Loss per epoch, split into panels rather than stacked on two axes.

    The size term is an L1 in pixels and runs an order of magnitude above the other two.
    Sharing one axis would flatten the heatmap and offset curves into a line along the
    bottom, and a second y-axis would make the two look comparable when they are not.
    """
    plt = _plt()
    epochs = [h.epoch for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    panels = [
        ("total", [h.loss for h in history], SERIES[0]),
        ("heatmap (focal)", [h.heatmap_loss for h in history], SERIES[2]),
        ("size (L1, pixels)", [h.size_loss for h in history], SERIES[1]),
    ]
    for ax, (label, values, colour) in zip(axes, panels, strict=True):
        ax.plot(epochs, values, color=colour, linewidth=2)
        ax.annotate(f"final {values[-1]:.3f}", (0.96, 0.9), xycoords="axes fraction",
                    ha="right", color=INK, fontsize=9, fontweight="bold")
        style(ax, title=label, xlabel="epoch")

    axes[0].set_ylabel("loss", color=INK_SOFT, fontsize=10)
    fig.tight_layout()
    return save(fig, out_dir, "training_curves")


def condition_f1(per_condition: dict[str, float], out_dir: Path | None = None):
    """Detection F1 per lighting condition.

    Sorted worst first, because the useful question is which condition is weakest, and a
    chart ordered alphabetically makes the reader do that sort themselves.
    """
    plt = _plt()
    ranked = sorted(per_condition.items(), key=lambda kv: kv[1])
    names = [name for name, _ in ranked]
    values = [value for _, value in ranked]

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    # The weakest condition takes the warning hue; everything else is one series colour.
    colours = [SERIES[3] if index == 0 else SERIES[0] for index in range(len(values))]
    bars = ax.bar(names, values, color=colours, width=0.58)

    for bar, value in zip(bars, values, strict=True):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK, fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.1)
    style(ax, title="Detection F1 by lighting condition", ylabel="F1")
    fig.tight_layout()
    return save(fig, out_dir, "condition_f1")


def target_heatmap(heatmap: np.ndarray, image: np.ndarray | None = None,
                   out_dir: Path | None = None):
    """The CenterNet target: where the model is told object centres are.

    Worth looking at at least once. It is the step where an indexing mistake is obvious to
    the eye and invisible in the loss, which will happily descend while training on peaks
    in the wrong places.
    """
    from matplotlib.colors import LinearSegmentedColormap

    plt = _plt()
    cmap = LinearSegmentedColormap.from_list("parkfit_seq", SEQUENTIAL)
    collapsed = heatmap.max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 10 * collapsed.shape[0] / collapsed.shape[1]))
    if image is not None:
        ax.imshow(image, alpha=0.5, extent=(0, collapsed.shape[1], collapsed.shape[0], 0))
    image_handle = ax.imshow(collapsed, cmap=cmap, alpha=0.85, vmin=0, vmax=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Centre heatmap target (all classes collapsed)", color=INK, fontsize=11,
                 fontweight="bold", loc="left", pad=8)
    bar = fig.colorbar(image_handle, ax=ax, pad=0.015, fraction=0.03)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    fig.tight_layout()
    return save(fig, out_dir, "target_heatmap")


def gaussian_radius_curve(out_dir: Path | None = None):
    """Splat radius against object size.

    A fixed radius would smear a distant motorcycle's peak across its neighbours and give a
    truck a peak too sharp to ever hit. The curve is why the radius is derived.
    """
    from parkfit.ml.datasets.scenes import gaussian_radius

    plt = _plt()
    sizes = np.arange(4, 80, 1.0)
    radii = [gaussian_radius(s * 0.5, s) for s in sizes]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(sizes, radii, color=SERIES[0], linewidth=2)
    for marker, note in ((10, "motorcycle"), (46, "truck")):
        value = gaussian_radius(marker * 0.5, marker)
        ax.scatter([marker], [value], s=48, color=SERIES[1], zorder=3, linewidths=0)
        ax.annotate(f"{note}  r={value:.1f}", (marker, value), textcoords="offset points",
                    xytext=(8, -2), color=INK, fontsize=9, fontweight="bold")

    style(ax, title="Gaussian splat radius grows with the object",
          xlabel="box width (grid cells)", ylabel="radius (cells)")
    fig.tight_layout()
    return save(fig, out_dir, "gaussian_radius")


def denormalise(chw: np.ndarray) -> np.ndarray:
    """Turn a CHW float tensor back into an HWC image for display."""
    return np.clip(np.transpose(chw, (1, 2, 0)), 0.0, 1.0)


def grid_of_scenes(images: np.ndarray, labels: list[list[dict]], columns: int = 3,
                   out_dir: Path | None = None):
    """A contact sheet of training scenes with their ground truth."""
    from matplotlib.patches import Rectangle

    plt = _plt()
    count = len(images)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 2.7 * rows))
    flat = np.atleast_1d(axes).ravel()

    for index, ax in enumerate(flat):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if index >= count:
            ax.axis("off")
            continue
        ax.imshow(images[index])
        for box in labels[index]:
            ax.add_patch(Rectangle(
                (box["x1"], box["y1"]), box["x2"] - box["x1"], box["y2"] - box["y1"],
                fill=False, edgecolor=SERIES[1], linewidth=1.8))
        ax.set_title(f"{len(labels[index])} vehicles", color=INK_SOFT, fontsize=9,
                     loc="left")

    fig.tight_layout()
    return save(fig, out_dir, "scene_grid")
