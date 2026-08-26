"""Accuracy evaluation.

Reports the metric table the project specification asks for, measured rather than
asserted. The numbers come from synthetic scenes whose gap lengths are known to the
millimetre by construction, so "gap-length mean absolute error" is a measurement and not
a hope.

One metric matters more than the others and is listed first: the **false-free rate**,
how often the system calls a space free when it is not. Overall accuracy hides it. A
detector that reports every space occupied scores well on accuracy and is useless; one
that reports every space free scores identically and is actively harmful. Only the
directional errors distinguish them.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from parkfit.ml.synthetic.scene import CameraModel, SceneGenerator

log = logging.getLogger(__name__)


@dataclass
class MetricTarget:
    name: str
    target: float
    higher_is_better: bool
    unit: str = ""

    def passes(self, value: float | None) -> bool | None:
        if value is None:
            return None
        return value >= self.target if self.higher_is_better else value <= self.target


#: The targets from the specification. Kept here so a run reports pass or fail rather
#: than a number the reader has to look up.
TARGETS: dict[str, MetricTarget] = {
    "false_free_rate": MetricTarget("False-free rate", 0.02, False, "%"),
    "vacant_precision": MetricTarget("Vacant precision", 0.98, True, "%"),
    "vacant_recall": MetricTarget("Vacant recall", 0.90, True, "%"),
    "gap_mae_m": MetricTarget("Gap-length MAE", 0.25, False, "m"),
    "gap_p95_m": MetricTarget("Gap-length 95th percentile error", 0.50, False, "m"),
    "false_fit_rate": MetricTarget("False 'fits' rate", 0.02, False, "%"),
    "search_p95_ms": MetricTarget("Cached search p95", 500.0, False, "ms"),
}


@dataclass
class EvaluationResult:
    metrics: dict[str, float | None] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    per_condition: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def verdict(self, key: str) -> bool | None:
        target = TARGETS.get(key)
        return target.passes(self.metrics.get(key)) if target else None


def evaluate_gap_measurement(
    *, scenes: int = 60, seed: int = 7, camera: CameraModel | None = None
) -> EvaluationResult:
    """Measure gap-length error over synthetic scenes with exact ground truth.

    The detector is perfect here by design: detections come from the scene geometry.
    That isolates the *projection and interval arithmetic*, which is what this measures.
    Running the same harness with a real detector's boxes measures the two together, and
    the difference between the runs is the detector's contribution.
    """
    from parkfit.native import native

    result = EvaluationResult()
    if native is None:
        result.notes.append("native module not built; gap measurement not evaluated")
        return result

    camera = camera or CameraModel()
    generator = SceneGenerator(camera=camera, seed=seed)

    # Calibrate once from the surveyed control points, exactly as a real camera would be.
    control = camera.control_points()
    image_points = [p["image"] for p in control]
    world_points = [p["world"] for p in control]

    errors: list[float] = []
    by_condition: dict[str, list[float]] = {}
    matched = 0
    missed = 0
    spurious = 0

    for index in range(scenes):
        condition = SceneGenerator.CONDITIONS[index % len(SceneGenerator.CONDITIONS)]
        scene = generator.build(condition=condition)

        measured = _measure_scene(scene, image_points, world_points)
        truth = [g for g in scene.gap_lengths_m if g >= 3.0]

        # Pair each true gap with its nearest measurement. A greedy nearest match is
        # adequate because gaps along one kerb are well separated by construction.
        remaining = list(measured)
        for true_length in truth:
            if not remaining:
                missed += 1
                continue
            closest = min(remaining, key=lambda m: abs(m - true_length))
            if abs(closest - true_length) > 2.0:
                missed += 1
                continue
            remaining.remove(closest)
            error = abs(closest - true_length)
            errors.append(error)
            by_condition.setdefault(condition, []).append(error)
            matched += 1
        spurious += len(remaining)

    if errors:
        errors.sort()
        result.metrics["gap_mae_m"] = sum(errors) / len(errors)
        result.metrics["gap_p95_m"] = errors[min(len(errors) - 1, int(0.95 * len(errors)))]
        result.metrics["gap_max_m"] = errors[-1]

    result.counts.update(
        {
            "scenes": scenes,
            "gaps_matched": matched,
            "gaps_missed": missed,
            "gaps_spurious": spurious,
        }
    )
    for condition, values in sorted(by_condition.items()):
        result.per_condition[condition] = {
            "mae_m": sum(values) / len(values),
            "count": len(values),
        }
    return result


def _measure_scene(scene, image_points, world_points) -> list[float]:
    """Run one scene through the same geometry the worker uses."""
    from parkfit.ml.evaluate import _cpp_bridge
    from parkfit.native import native  # noqa: F401  (guarded by the caller)

    return _cpp_bridge.measure_gaps(
        image_points=image_points,
        world_points=world_points,
        detections=scene.detections(),
        kerb_start=(scene.camera.origin_x, scene.kerb_y),
        kerb_end=(scene.camera.origin_x + scene.kerb_length_m, scene.kerb_y),
        camera_world=(scene.camera.origin_x, scene.camera.origin_y),
    )


def evaluate_fit_engine(*, samples: int = 4000, seed: int = 11) -> EvaluationResult:
    """Measure the false-'fits' rate against exact bay and vehicle geometry.

    A false 'fits' is the fit-engine analogue of a false-free: the system said a vehicle
    would go into a space that it physically cannot. Ground truth here is arithmetic:
    the vehicle either occupies less than the bay or it does not.
    """
    import random

    from parkfit.native import native

    result = EvaluationResult()
    if native is None:
        result.notes.append("native module not built; fit engine not evaluated")
        return result

    rng = random.Random(seed)
    margins = native.Margins()
    false_fits = 0
    false_rejects = 0
    accepted = 0

    for _ in range(samples):
        length = rng.uniform(330.0, 700.0)
        body_width = rng.uniform(150.0, 220.0)
        vehicle = native.Vehicle()
        vehicle.length_cm = length
        vehicle.body_width_cm = body_width
        vehicle.width_with_mirrors_cm = body_width + rng.uniform(28.0, 46.0)
        vehicle.height_cm = rng.uniform(135.0, 280.0)
        vehicle.height_with_accessories_cm = vehicle.height_cm

        parallel = rng.random() < 0.55
        bay_length = rng.uniform(380.0, 720.0)
        bay_width = rng.uniform(170.0, 270.0)
        orientation = (
            native.BayOrientation.PARALLEL if parallel else native.BayOrientation.PERPENDICULAR
        )

        verdict = native.check_bay(vehicle, bay_length, bay_width, orientation, margins)

        # Physical truth: the bodywork has to be inside the bay, full stop. Any verdict
        # that accepts a vehicle failing this is a false 'fits'.
        physically_possible = length <= bay_length and body_width <= bay_width

        if verdict.acceptable:
            accepted += 1
            if not physically_possible:
                false_fits += 1
        elif physically_possible and length <= bay_length - 60 and body_width <= bay_width - 30:
            # Comfortably possible yet rejected: a false reject. Costs an option, not a
            # bumper, so it is tracked but not a headline failure.
            false_rejects += 1

    result.metrics["false_fit_rate"] = false_fits / max(1, accepted)
    result.metrics["false_reject_rate"] = false_rejects / max(1, samples)
    result.counts.update(
        {
            "samples": samples,
            "accepted": accepted,
            "false_fits": false_fits,
            "false_rejects": false_rejects,
        }
    )
    return result


def evaluate_state_machine(*, trials: int = 3000, seed: int = 5) -> EvaluationResult:
    """Measure the false-free rate of the temporal filter under noisy detections.

    The scenario is the one that matters: a space that is genuinely occupied, with a
    detector that intermittently misses the car. A filter that publishes VACANT on a
    single miss is exactly the failure the asymmetric transitions exist to prevent.
    """
    import random

    result = EvaluationResult()
    rng = random.Random(seed)

    try:
        from parkfit.ml.evaluate import _cpp_bridge
    except ImportError:  # pragma: no cover
        result.notes.append("state machine bridge unavailable")
        return result

    false_free = 0
    true_occupied = 0
    detected_vacant = 0
    true_vacant = 0
    missed_vacant = 0

    for _ in range(trials):
        occupied = rng.random() < 0.6
        # 12 % miss rate on an occupied space, 6 % false alarm on an empty one: a
        # deliberately mediocre detector, because a filter that only works with a
        # perfect one is not doing anything.
        scores = []
        for _frame in range(6):
            if occupied:
                scores.append(0.05 if rng.random() < 0.12 else rng.uniform(0.55, 0.97))
            else:
                scores.append(rng.uniform(0.5, 0.9) if rng.random() < 0.06 else 0.02)

        published = _cpp_bridge.run_state_machine(scores)

        if occupied:
            true_occupied += 1
            if published == "VACANT":
                false_free += 1
        else:
            true_vacant += 1
            if published == "VACANT":
                detected_vacant += 1
            else:
                missed_vacant += 1

    result.metrics["false_free_rate"] = false_free / max(1, true_occupied)
    result.metrics["vacant_recall"] = detected_vacant / max(1, true_vacant)
    result.metrics["vacant_precision"] = detected_vacant / max(1, detected_vacant + false_free)
    result.counts.update(
        {
            "trials": trials,
            "truly_occupied": true_occupied,
            "truly_vacant": true_vacant,
            "false_free": false_free,
            "missed_vacant": missed_vacant,
        }
    )
    return result


def evaluate_search_latency(*, runs: int = 12) -> EvaluationResult:
    """Measure end-to-end search latency against whatever data is loaded locally."""
    import time

    result = EvaluationResult()
    try:
        from datetime import UTC, datetime

        from parkfit.domain.vehicle import VehicleProfile
        from parkfit.services.search import SearchEngine, SearchPreferences, SearchRequest
        from parkfit.storage.session import session_scope
    except ImportError as exc:  # pragma: no cover
        result.notes.append(f"search not available: {exc}")
        return result

    destinations = ["Rembrandthuis", "Van Gogh Museum", "Artis", "Dam", "Rijksmuseum", "Vondelpark"]
    vehicle = VehicleProfile(
        id="eval",
        length_cm=405.0,
        body_width_cm=175.0,
        width_with_mirrors_cm=194.0,
        height_cm=145.0,
        height_with_accessories_cm=145.0,
        weight_kg=1100.0,
        length_confirmed=True,
        width_confirmed=True,
        height_confirmed=True,
    )

    timings: list[float] = []
    empty = 0
    with session_scope() as session:
        engine = SearchEngine(session)
        try:
            # One warm-up: the first call loads the road graph and builds the spatial
            # index, which is a process-start cost rather than a per-search one.
            engine.search(
                SearchRequest(
                    destination=destinations[0],
                    vehicle=vehicle,
                    origin_lat=52.3789,
                    origin_lon=4.9002,
                    arrival_time=datetime.now(UTC),
                )
            )
            for index in range(runs):
                destination = destinations[index % len(destinations)]
                started = time.perf_counter()
                response = engine.search(
                    SearchRequest(
                        destination=destination,
                        vehicle=vehicle,
                        origin_lat=52.3789,
                        origin_lon=4.9002,
                        arrival_time=datetime.now(UTC),
                        duration_minutes=120,
                        preferences=SearchPreferences(max_walk_minutes=15),
                    )
                )
                timings.append((time.perf_counter() - started) * 1000.0)
                if not response.results:
                    empty += 1
        finally:
            engine.close()

    if timings:
        timings.sort()
        result.metrics["search_median_ms"] = timings[len(timings) // 2]
        result.metrics["search_p95_ms"] = timings[min(len(timings) - 1, int(0.95 * len(timings)))]
        result.metrics["search_max_ms"] = timings[-1]
    result.counts.update({"runs": len(timings), "empty_results": empty})
    if empty:
        result.notes.append(
            f"{empty} searches returned nothing; run `pf ingest all` for local data"
        )
    return result


def run_all(*, scenes: int = 60, quick: bool = False) -> EvaluationResult:
    """Run every evaluation and merge the results into one table."""
    combined = EvaluationResult()

    for part in (
        evaluate_state_machine(trials=800 if quick else 3000),
        evaluate_fit_engine(samples=1200 if quick else 4000),
        evaluate_gap_measurement(scenes=18 if quick else scenes),
        evaluate_search_latency(runs=6 if quick else 12),
    ):
        combined.metrics.update(part.metrics)
        combined.counts.update(part.counts)
        combined.per_condition.update(part.per_condition)
        combined.notes.extend(part.notes)

    return combined


def format_report(result: EvaluationResult) -> str:
    """Render the metric table, ordered with the metric that matters first."""
    lines: list[str] = []
    order = [
        "false_free_rate",
        "vacant_precision",
        "vacant_recall",
        "false_fit_rate",
        "gap_mae_m",
        "gap_p95_m",
        "search_p95_ms",
    ]

    header = f"{'metric':<38} {'measured':>12} {'target':>10}   verdict"
    lines.append(header)
    lines.append("-" * len(header))

    for key in order:
        target = TARGETS.get(key)
        value = result.metrics.get(key)
        if target is None:
            continue
        if value is None:
            lines.append(f"{target.name:<38} {'not measured':>12} {'':>10}   -")
            continue

        if target.unit == "%":
            measured = f"{value * 100:.2f} %"
            goal = f"{'>=' if target.higher_is_better else '<='} {target.target * 100:.0f} %"
        elif target.unit == "m":
            measured = f"{value:.3f} m"
            goal = f"<= {target.target:.2f} m"
        else:
            measured = f"{value:.0f} ms"
            goal = f"<= {target.target:.0f} ms"

        passed = target.passes(value)
        verdict = "PASS" if passed else "FAIL"
        lines.append(f"{target.name:<38} {measured:>12} {goal:>10}   {verdict}")

    extras = {k: v for k, v in result.metrics.items() if k not in TARGETS}
    if extras:
        lines.append("")
        lines.append("supporting measurements")
        for key, value in sorted(extras.items()):
            if value is None:
                continue
            suffix = " m" if key.endswith("_m") else (" ms" if key.endswith("_ms") else "")
            lines.append(f"  {key:<36} {value:.3f}{suffix}")

    if result.per_condition:
        lines.append("")
        lines.append("gap error by lighting condition")
        for condition, stats in result.per_condition.items():
            lines.append(
                f"  {condition:<14} MAE {stats['mae_m']:.3f} m over {int(stats['count'])} gaps"
            )

    if result.counts:
        lines.append("")
        lines.append("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(result.counts.items())))

    for note in result.notes:
        lines.append(f"note: {note}")

    return "\n".join(lines)


def write_report(result: EvaluationResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in result.metrics.items()
        },
        "counts": result.counts,
        "per_condition": result.per_condition,
        "notes": result.notes,
        "targets": {
            k: {"target": t.target, "higher_is_better": t.higher_is_better, "unit": t.unit}
            for k, t in TARGETS.items()
        },
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
