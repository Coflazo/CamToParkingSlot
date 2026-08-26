"""The learned occupancy model.

A gradient-boosted classifier over :mod:`parkfit.prediction.features`, answering one
question: *how likely is this target to be occupied at this moment?* Its output feeds the
``P(free now)`` term of the ranking survival model, and it is labelled
``PREDICTIVE_MODEL`` evidence, which sits below a user confirmation and far below a
sensor, so a prediction never displaces something somebody actually saw.

**Splitting.** Rows are split by target *and* by time, never at random. Two observations
of the same bay fifteen minutes apart are almost the same row; a random split puts one in
train and one in test and reports a score that says nothing about a bay the model has not
seen. The split here holds out whole targets (does it generalise across space?) and the
final stretch of the window (does it generalise forward in time?), and scores them
separately, because those are different failures.

**Baselines.** A model is only worth its complexity if it beats the simpler thing. Three
baselines are always reported alongside it:

* the flat prior the system uses today, one number for everything;
* the best constant per target kind, bays and facilities differ enormously;
* the best constant per *target*, the mean occupancy of that specific bay.

The third is the one that matters. Beating it requires predicting time-of-day structure,
because a per-target constant already captures everything static about the place. On the
held-out-targets split that baseline is unavailable by construction, so the comparison
there is against the per-kind constant.

**Degradation.** If LightGBM is missing, or no model file has been trained, prediction
returns ``None`` and every caller falls back to the analytic prior. The system is designed
to work without this model; it is an improvement, not a dependency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from parkfit.prediction.features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TargetStatics,
    batch,
    load_statics,
    row,
)
from parkfit.storage.models import AvailabilityObservation, OccupancyState

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/models/occupancy.lgb")
DEFAULT_META_PATH = Path("data/models/occupancy.json")


@dataclass
class SplitScore:
    """How the model and its baselines did on one held-out slice."""

    name: str
    rows: int = 0
    model_brier: float = 0.0
    model_logloss: float = 0.0
    model_auc: float = 0.0
    flat_prior_brier: float = 0.0
    per_kind_brier: float = 0.0
    per_target_brier: float | None = None

    @property
    def beats_per_target(self) -> bool | None:
        if self.per_target_brier is None:
            return None
        return self.model_brier < self.per_target_brier

    def describe(self) -> str:
        parts = [
            f"{self.name}: {self.rows:,} rows",
            f"model Brier {self.model_brier:.4f}",
            f"flat {self.flat_prior_brier:.4f}",
            f"per-kind {self.per_kind_brier:.4f}",
        ]
        if self.per_target_brier is not None:
            verdict = "beaten" if self.beats_per_target else "NOT beaten"
            parts.append(f"per-target {self.per_target_brier:.4f} ({verdict})")
        parts.append(f"AUC {self.model_auc:.3f}")
        return " | ".join(parts)


@dataclass
class TrainReport:
    trained: bool = False
    reason: str = ""
    rows: int = 0
    targets: int = 0
    trees: int = 0
    splits: list[SplitScore] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    model_path: str = ""

    def describe(self) -> str:
        if not self.trained:
            return f"not trained: {self.reason}"
        lines = [f"{self.rows:,} rows over {self.targets:,} targets, {self.trees} trees"]
        lines.extend(f"  {s.describe()}" for s in self.splits)
        top = sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:6]
        lines.append("  top features: " + ", ".join(f"{k} {v:.0f}" for k, v in top))
        return "\n".join(lines)


def _brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _logloss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def _auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, computed directly rather than pulled in from sklearn.

    Equivalent to the Mann-Whitney U statistic over the two label groups; ties are
    handled by averaging ranks, which is what makes a constant predictor score exactly
    0.5 instead of accidentally scoring 1.0.
    """
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(probabilities), dtype=np.float64)
    ranks[order] = np.arange(1, len(probabilities) + 1, dtype=np.float64)

    sorted_p = probabilities[order]
    start = 0
    for i in range(1, len(sorted_p) + 1):
        if i == len(sorted_p) or sorted_p[i] != sorted_p[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_training_rows(
    session: Session,
    *,
    source_name: str | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the training matrix from persisted observations.

    Returns ``(features, labels, target_index, timestamps)``. The last two are not
    features; they exist so the split can be made by target and by time.

    The scan is streamed into a pre-allocated array rather than collected into lists.
    SQLAlchemy materialises a result set into ``Row`` objects by default, and half a
    million of those exhausts memory before a single tree is grown; the array they end up
    in is only sixty megabytes.
    """
    base = select(
        AvailabilityObservation.target_kind,
        AvailabilityObservation.target_id,
        AvailabilityObservation.observed_at,
        AvailabilityObservation.state,
    )
    if source_name:
        base = base.where(AvailabilityObservation.source_name == source_name)

    count_stmt = select(func.count()).select_from(AvailabilityObservation)
    if source_name:
        count_stmt = count_stmt.where(AvailabilityObservation.source_name == source_name)
    total = int(session.execute(count_stmt).scalar_one() or 0)
    if limit:
        total = min(total, limit)

    if total == 0:
        empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return empty, np.empty(0), np.empty(0, dtype=np.int64), np.empty(0)

    key_stmt = select(
        AvailabilityObservation.target_kind, AvailabilityObservation.target_id
    ).distinct()
    if source_name:
        key_stmt = key_stmt.where(AvailabilityObservation.source_name == source_name)
    keys = sorted({(kind, tid) for kind, tid in session.execute(key_stmt)})
    statics = load_statics(session, keys)
    index_of = {key: i for i, key in enumerate(keys)}

    features = np.empty((total, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(total, dtype=np.float64)
    target_index = np.empty(total, dtype=np.int64)
    stamps = np.empty(total, dtype=np.float64)

    stmt = base.order_by(AvailabilityObservation.observed_at)
    if limit:
        stmt = stmt.limit(limit)

    n = 0
    for kind, tid, observed_at, state in session.execute(stmt).yield_per(20000):
        static = statics.get((kind, tid))
        if static is None or observed_at is None or n >= total:
            continue
        features[n] = row(static, observed_at)
        labels[n] = 1.0 if state == OccupancyState.OCCUPIED.value else 0.0
        target_index[n] = index_of[(kind, tid)]
        stamps[n] = observed_at.timestamp()
        n += 1

    return features[:n], labels[:n], target_index[:n], stamps[:n]


def _baselines(
    train_y: np.ndarray,
    train_targets: np.ndarray,
    train_kind: np.ndarray,
    test_y: np.ndarray,
    test_targets: np.ndarray,
    test_kind: np.ndarray,
    *,
    per_target_available: bool,
) -> tuple[float, float, float | None]:
    """Brier scores for the three baselines, all fitted on train only."""
    flat = np.full(len(test_y), 0.75)  # what the system assumes with no history at all

    kind_means = {int(k): float(train_y[train_kind == k].mean()) for k in np.unique(train_kind)}
    overall = float(train_y.mean())
    per_kind = np.array([kind_means.get(int(k), overall) for k in test_kind])

    per_target = None
    if per_target_available:
        means: dict[int, float] = {}
        for t in np.unique(train_targets):
            means[int(t)] = float(train_y[train_targets == t].mean())
        per_target_pred = np.array([means.get(int(t), overall) for t in test_targets])
        per_target = _brier(per_target_pred, test_y)

    return _brier(flat, test_y), _brier(per_kind, test_y), per_target


def train(
    session: Session,
    *,
    source_name: str | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    holdout_target_fraction: float = 0.2,
    holdout_time_fraction: float = 0.2,
    num_trees: int = 400,
    num_threads: int = 4,
    seed: int = 20260826,
) -> TrainReport:
    """Fit the occupancy model and score it against its baselines."""
    try:
        import lightgbm as lgb
    except ImportError:
        return TrainReport(reason="lightgbm is not installed")

    x_all, y, targets, stamps = load_training_rows(session, source_name=source_name)
    if len(y) < 500:
        return TrainReport(reason=f"only {len(y)} observations; need at least 500")

    kind_col = FEATURE_NAMES.index("is_facility")
    kinds = x_all[:, kind_col]
    unique_targets = np.unique(targets)

    rng = np.random.default_rng(seed)
    held_targets = {
        int(t)
        for t in rng.choice(
            unique_targets,
            size=max(1, int(len(unique_targets) * holdout_target_fraction)),
            replace=False,
        )
    }
    target_held = np.array([int(t) in held_targets for t in targets])

    # The time cut is taken on the timestamp quantile of the *remaining* rows, so the two
    # holdouts do not overlap and each measures one thing.
    cutoff = float(np.quantile(stamps, 1.0 - holdout_time_fraction))
    time_held = (stamps >= cutoff) & ~target_held

    train_mask = ~target_held & ~time_held
    if train_mask.sum() < 200:
        return TrainReport(reason="not enough rows left after holdout")

    categorical = [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]
    dataset = lgb.Dataset(
        x_all[train_mask],
        label=y[train_mask],
        feature_name=list(FEATURE_NAMES),
        categorical_feature=categorical,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 48,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        # Occupancy is a probability, and the ranking model consumes it as one. An
        # uncalibrated but well-ranked score would sort correctly and price wrongly, so
        # the objective stays binary log-loss rather than a ranking objective.
        "num_threads": max(1, num_threads),
        "verbosity": -1,
        "seed": seed,
    }
    booster = lgb.train(params, dataset, num_boost_round=num_trees)

    report = TrainReport(
        trained=True,
        rows=int(train_mask.sum()),
        targets=len(unique_targets),
        trees=booster.num_trees(),
        model_path=str(model_path),
    )

    for name, mask, per_target_available in (
        ("unseen time", time_held, True),
        ("unseen targets", target_held, False),
    ):
        if not mask.any():
            continue
        predictions = booster.predict(x_all[mask])
        flat_b, kind_b, target_b = _baselines(
            y[train_mask],
            targets[train_mask],
            kinds[train_mask],
            y[mask],
            targets[mask],
            kinds[mask],
            per_target_available=per_target_available,
        )
        report.splits.append(
            SplitScore(
                name=name,
                rows=int(mask.sum()),
                model_brier=_brier(predictions, y[mask]),
                model_logloss=_logloss(predictions, y[mask]),
                model_auc=_auc(predictions, y[mask]),
                flat_prior_brier=flat_b,
                per_kind_brier=kind_b,
                per_target_brier=target_b,
            )
        )

    gains = booster.feature_importance(importance_type="gain")
    report.feature_importance = {
        name: float(g) for name, g in zip(FEATURE_NAMES, gains, strict=True)
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "features": list(FEATURE_NAMES),
                "trained_at": datetime.now().astimezone().isoformat(),
                "rows": report.rows,
                "targets": report.targets,
                "splits": [asdict(s) for s in report.splits],
                "source_name": source_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


class OccupancyModel:
    """Loaded model, or an honest absence of one."""

    def __init__(self, booster=None, features: tuple[str, ...] = FEATURE_NAMES):
        self._booster = booster
        self._features = features

    @property
    def available(self) -> bool:
        return self._booster is not None

    @classmethod
    def load(cls, model_path: Path = DEFAULT_MODEL_PATH) -> OccupancyModel:
        if not model_path.exists():
            return cls()
        try:
            import lightgbm as lgb
        except ImportError:
            log.info("lightgbm not installed; occupancy predictions fall back to the prior")
            return cls()
        try:
            booster = lgb.Booster(model_file=str(model_path))
        except Exception:
            log.warning("could not load %s; falling back to the prior", model_path)
            return cls()

        meta_path = model_path.with_suffix(".json")
        features = FEATURE_NAMES
        if meta_path.exists():
            try:
                stored = tuple(json.loads(meta_path.read_text(encoding="utf-8"))["features"])
            except (ValueError, KeyError):
                stored = FEATURE_NAMES
            if stored != FEATURE_NAMES:
                # A model trained on a different feature set would silently read the
                # wrong column for every row. Refuse it rather than predict nonsense.
                log.warning("%s was trained on a different feature set; ignoring it", model_path)
                return cls()
            features = stored
        return cls(booster, features)

    def probability_occupied(
        self, pairs: list[tuple[TargetStatics, datetime]]
    ) -> np.ndarray | None:
        """Predicted occupancy probability per pair, or ``None`` if unavailable."""
        if self._booster is None or not pairs:
            return None
        return np.asarray(self._booster.predict(batch(pairs)), dtype=np.float64)


_cached: OccupancyModel | None = None


def get_model(model_path: Path = DEFAULT_MODEL_PATH, *, reload: bool = False) -> OccupancyModel:
    """Process-wide model. Loading a booster per request costs more than predicting."""
    global _cached
    if _cached is None or reload:
        _cached = OccupancyModel.load(model_path)
    return _cached
