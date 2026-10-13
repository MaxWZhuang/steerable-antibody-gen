"""Train-only fixed ridge probe and grouped regression metrics (numpy float64).

The probe minimises ``(1/2n)||y - ybar - Z w||^2 + (alpha/2)||w||^2`` where ``Z`` standardises each
column with the TRAIN mean and population std (ddof=0) and ``ybar`` is the TRAIN label mean; the
intercept is ``ybar`` and is not penalised. Exactly constant train columns get scale 1, are centred to
zero and carry coefficient 0 (they are counted). ``w`` comes from the dual solve when ``n < d`` and the
primal solve otherwise. Because the penalty is ``n * alpha``, duplicating every training row leaves the
fit unchanged. Held-out rows contribute nothing to any fitted quantity: predictions are row-wise
(elementwise product then a per-row sum), so adding rows to a prediction batch cannot move another row.

No sklearn, no calibration on held-out data, no assay-selected coefficients, no confidence intervals,
thresholds or pass flags. ``scipy`` is imported lazily, only for Spearman ranks.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from smallAntibodyGen.experiments.interaction_qualification.config import PARTITIONS
from smallAntibodyGen.experiments.interaction_qualification.records import Record

RIDGE_RECIPE = (
    "train-only column standardisation (ddof=0, constant columns scale 1), train-mean intercept not "
    "penalised, minimise (1/2n)||yc - Zw||^2 + (alpha/2)||w||^2, dual solve if n < d else primal, "
    "one fixed alpha for every arm, no tuning"
)


class NoTrainingSupport(ValueError):
    """There are no training rows, so no probe can be fitted (the runner skips the fit)."""


def _matrix(x: object, name: str) -> np.ndarray:
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {raw.dtype}")
    out = np.array(raw, dtype=np.float64)  # always a copy: the caller's array is never touched
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {out.shape}")
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _vector(y: object, name: str) -> np.ndarray:
    raw = np.asarray(y)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {raw.dtype}")
    out = np.array(raw, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"alpha must be a positive finite real, got {value!r}")
    alpha = float(value)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a positive finite real, got {value!r}")
    return alpha


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class RidgeFit:
    alpha: float
    n_train: int
    n_features: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    zero_std_columns: int
    y_mean: float  # the (unpenalised) intercept of the standardised model
    coefficients: np.ndarray  # in standardised-feature units
    solver: str  # "dual" or "primal"
    objective: float

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "recipe": RIDGE_RECIPE,
            "alpha": self.alpha,
            "n_train": self.n_train,
            "n_features": self.n_features,
            "zero_std_columns": self.zero_std_columns,
            "y_mean": self.y_mean,
            "solver": self.solver,
            "objective": self.objective,
            "coefficient_l2": float(np.sqrt(self.coefficients @ self.coefficients)),
            "feature_mean_sha256": _array_sha256(self.feature_mean),
            "feature_scale_sha256": _array_sha256(self.feature_scale),
            "coefficients_sha256": _array_sha256(self.coefficients),
        }
        if include_arrays:
            out["feature_mean"] = self.feature_mean.tolist()
            out["feature_scale"] = self.feature_scale.tolist()
            out["coefficients"] = self.coefficients.tolist()
        return out


def fit_ridge(x: object, y: object, alpha: float) -> RidgeFit:
    """Fit on TRAIN rows only. Inputs are copied, never mutated."""
    alpha = _alpha(alpha)
    features = _matrix(x, "x")
    labels = _vector(y, "y")
    n, d = features.shape
    if n == 0:
        raise NoTrainingSupport("no training rows")
    if d == 0:
        raise ValueError("x has no feature columns")
    if labels.shape[0] != n:
        raise ValueError(f"y has {labels.shape[0]} entries for {n} training rows")
    constant = np.all(features == features[0], axis=0)
    mean = features.mean(axis=0)
    mean[constant] = features[0][constant]
    scale = features.std(axis=0)
    scale[constant] = 1.0
    if not bool(np.all(np.isfinite(scale))) or bool(np.any(scale <= 0.0)):
        raise ValueError("a non-constant training column has no representable standard deviation")
    z = (features - mean) / scale
    z[:, constant] = 0.0
    y_mean = float(labels.mean())
    centred = labels - y_mean
    if n < d:
        solver = "dual"
        gram = z @ z.T
        gram[np.diag_indices(n)] += n * alpha
        weights = z.T @ np.linalg.solve(gram, centred)
    else:
        solver = "primal"
        gram = z.T @ z
        gram[np.diag_indices(d)] += n * alpha
        weights = np.linalg.solve(gram, z.T @ centred)
    weights[constant] = 0.0
    if not bool(np.all(np.isfinite(weights))):
        raise ValueError("the ridge solve produced non-finite coefficients")
    residual = centred - z @ weights
    objective = 0.5 / n * float(residual @ residual) + 0.5 * alpha * float(weights @ weights)
    return RidgeFit(
        alpha=alpha,
        n_train=n,
        n_features=d,
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        zero_std_columns=int(constant.sum()),
        y_mean=y_mean,
        coefficients=_readonly(weights),
        solver=solver,
        objective=objective,
    )


def predict(fit: RidgeFit, x: object) -> np.ndarray:
    """Row-wise predictions with the TRAIN statistics; a row's value never depends on other rows."""
    features = _matrix(x, "x")
    if features.shape[1] != fit.n_features:
        raise ValueError(f"x has {features.shape[1]} columns, the fit has {fit.n_features}")
    z = (features - fit.feature_mean) / fit.feature_scale
    return np.sum(z * fit.coefficients, axis=1) + fit.y_mean


def predict_by_id(fit: RidgeFit, ids: Sequence[str], x: object) -> Dict[str, float]:
    """Predictions keyed by the record ids naming the rows of ``x`` (ids must be unique)."""
    if isinstance(ids, str):
        raise TypeError("pass a sequence of record ids, not a string")
    ordered = list(ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("record ids must be unique")
    values = predict(fit, x)
    if len(ordered) != values.shape[0]:
        raise ValueError(f"{len(ordered)} ids for {values.shape[0]} rows")
    return {record_id: float(value) for record_id, value in zip(ordered, values)}


# --- metrics --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredRow:
    record_id: str
    partition: str
    source: str
    assay: str
    pair_fingerprint: str
    y_true: float
    y_pred: float

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError(f"unknown partition {self.partition!r}")
        for name in ("y_true", "y_pred"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real, got {value!r}")


def scored_rows(records: Sequence[Record], predictions: Mapping[str, float]) -> List[ScoredRow]:
    """Join predictions to records by id; the id sets must match exactly. Order follows ``records``."""
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids must be unique")
    if set(ids) != set(predictions):
        raise ValueError(
            f"predictions and records cover different ids (missing {sorted(set(ids) - set(predictions))[:3]}, "
            f"extra {sorted(set(predictions) - set(ids))[:3]})"
        )
    return [
        ScoredRow(
            record_id=r.record_id,
            partition=r.partition,
            source=r.source,
            assay=r.assay,
            pair_fingerprint=r.pair_fingerprint,
            y_true=r.pkd,
            y_pred=float(predictions[r.record_id]),
        )
        for r in records
    ]


def training_counts(records: Iterable[Record]) -> Dict[Tuple[str, str], int]:
    """Training rows per (source, assay), for the training-support flags."""
    counts: Dict[Tuple[str, str], int] = {}
    for record in records:
        counts[(record.source, record.assay)] = counts.get((record.source, record.assay), 0) + 1
    return counts


@dataclass(frozen=True)
class GroupMetrics:
    partition: str
    source: Optional[str]  # None for the pooled-over-sources/assays row of a partition
    assay: Optional[str]
    n_rows: int
    n_unique_constructs: int  # distinct (heavy, light, antigen) pairs
    n_train_rows: int  # training rows of the same (source, assay); all training rows for a pooled row
    training_support: bool
    mae: float
    rmse: float
    train_mean_baseline_rmse: float
    pearson: Optional[float]
    spearman: Optional[float]
    correlation_unavailable: Optional[str]  # why pearson/spearman are None, else None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Evaluation:
    groups: Tuple[GroupMetrics, ...]  # PRIMARY: per (partition, source, assay)
    pooled: Tuple[GroupMetrics, ...]  # secondary: per partition over all of its sources/assays


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = math.sqrt(float(ac @ ac) * float(bc @ bc))
    if not math.isfinite(denominator) or denominator == 0.0:
        return None
    return min(1.0, max(-1.0, float(ac @ bc) / denominator))


def _correlations(y: np.ndarray, p: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if y.shape[0] < 2:
        return None, None, "fewer than 2 rows"
    constant_y = bool(np.all(y == y[0]))
    constant_p = bool(np.all(p == p[0]))
    if constant_y or constant_p:
        which = "labels and predictions" if constant_y and constant_p else "labels" if constant_y else "predictions"
        return None, None, f"constant {which}"
    from scipy.stats import rankdata  # lazy: only Spearman needs scipy

    pearson = _pearson(y, p)
    spearman = _pearson(rankdata(y), rankdata(p))
    if pearson is None or spearman is None:
        return None, None, "numerically degenerate variance"
    return pearson, spearman, None


def _metrics(
    partition: str,
    source: Optional[str],
    assay: Optional[str],
    rows: Sequence[ScoredRow],
    train_mean: float,
    n_train_rows: int,
) -> GroupMetrics:
    y = np.array([row.y_true for row in rows], dtype=np.float64)
    p = np.array([row.y_pred for row in rows], dtype=np.float64)
    pearson, spearman, reason = _correlations(y, p)
    return GroupMetrics(
        partition=partition,
        source=source,
        assay=assay,
        n_rows=len(rows),
        n_unique_constructs=len({row.pair_fingerprint for row in rows}),
        n_train_rows=n_train_rows,
        training_support=n_train_rows > 0,
        mae=float(np.mean(np.abs(p - y))),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        train_mean_baseline_rmse=float(np.sqrt(np.mean((y - train_mean) ** 2))),
        pearson=pearson,
        spearman=spearman,
        correlation_unavailable=reason,
    )


def evaluate(
    rows: Sequence[ScoredRow], *, train_mean: float, train_counts: Mapping[Tuple[str, str], int]
) -> Evaluation:
    """Metrics for exactly the rows handed in, grouped by (partition, source, assay) -- never by assay alone."""
    if isinstance(train_mean, bool) or not isinstance(train_mean, (int, float)) or not math.isfinite(train_mean):
        raise ValueError(f"train_mean must be a finite real, got {train_mean!r}")
    total_train = sum(train_counts.values())
    by_group: Dict[Tuple[str, str, str], List[ScoredRow]] = {}
    by_partition: Dict[str, List[ScoredRow]] = {}
    for row in rows:
        by_group.setdefault((row.partition, row.source, row.assay), []).append(row)
        by_partition.setdefault(row.partition, []).append(row)
    groups = tuple(
        _metrics(key[0], key[1], key[2], by_group[key], float(train_mean), train_counts.get((key[1], key[2]), 0))
        for key in sorted(by_group, key=lambda k: (PARTITIONS.index(k[0]), k[1], k[2]))
    )
    pooled = tuple(
        _metrics(partition, None, None, by_partition[partition], float(train_mean), total_train)
        for partition in sorted(by_partition, key=PARTITIONS.index)
    )
    return Evaluation(groups=groups, pooled=pooled)
