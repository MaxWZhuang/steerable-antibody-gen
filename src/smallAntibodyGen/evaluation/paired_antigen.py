"""
Paired cross-antigen evaluation: does a score differential track a measured one?

Step 2 of the agreed order of work in ``docs/research-followup/MEMO.md`` §4e. The evaluator
is deliberately model-agnostic. It accepts **two scores per variant** -- one per antigen
condition -- from anything: a released affinity regressor, a language-model pseudo-likelihood,
a conditioned policy's per-position readout reduced to a scalar, or a constant. Nothing here
imports a model.

The question it answers, for one antibody background and one antigen pair:

    Does the model's predicted human-versus-mouse difference track the measured difference?

Three rules from the memo review are load-bearing and are enforced here rather than left to
the caller.

**The target-wide shift is removed before anything is asked of a variant.** Nearly every
variant in the reference cohort binds one antigen more tightly than the other, so the raw
differential is dominated by an offset carrying no variant-specific information. A model can
reproduce that offset while knowing nothing about any variant.

**Sensitivity is a screen, never a criterion.** ``sensitivity`` reports whether a model's
differential varies with the variant at all. Clearing it licenses nothing about usefulness;
a mechanism can move its output further than another and be confidently wrong. Agreement with
measurement is the criterion, and it is a separate field.

**Flat arms stay in the output.** An antigen-blind arm has a constant differential by
construction, and a real arm can come out constant too. Rank correlation on a constant vector
is undefined, not zero. Undefined statistics are returned as ``None`` beside an explicit
``undefined_reason``, never as 0.0 and never by dropping the row -- those baselines are the
reference the whole comparison is read against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .contrasts import concordance

SCHEMA = "paired-antigen-evaluation/1"

DEFAULT_PERMUTATIONS = 10_000
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 20260910


class SequencePairScorer(Protocol):
    """Anything that turns one (binder, antigen) pair into one scalar.

    Deliberately minimal: a scorer needs no residue head, no sampling interface, and no
    structural input. A policy with a per-position head participates by reducing its readout
    to a scalar, and must say in ``score_semantics`` how it did so.
    """

    def score(self, binder_sequence: str, antigen_sequence: str) -> float: ...


@dataclass(frozen=True)
class PairedMeasurement:
    """One variant's measured pair. ``delta`` follows the manifest's stated direction."""

    variant: str
    measured_a: float
    measured_b: float
    delta: float
    paired_standard_error: float | None = None
    exploratory: bool = False


def measurements_from_manifest(payload: Mapping[str, Any]) -> list[PairedMeasurement]:
    """Read the cohort manifest written by ``scripts/reproduce_ym0693_cohort.py``."""
    if payload.get("schema") != "ym0693-paired-cohort/1":
        raise ValueError(f"unexpected manifest schema {payload.get('schema')!r}")
    return [
        PairedMeasurement(
            variant=row["variant"],
            measured_a=row["affinity_human"],
            measured_b=row["affinity_mouse"],
            delta=row["delta_mouse_minus_human"],
            paired_standard_error=row.get("paired_standard_error"),
            exploratory=bool(row.get("exploratory_subset", False)),
        )
        for row in payload["variants"]
    ]


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rho, or None when either vector is constant.

    Returning None rather than 0.0 or NaN is the point: a constant vector has no ranking, so
    the statistic is undefined, and an undefined statistic rendered as zero is indistinguishable
    from a measured null.
    """
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _permutation_p(x: np.ndarray, y: np.ndarray, observed: float, *, n: int, rng: np.random.Generator) -> float | None:
    """Two-sided permutation p for the rank association, shuffling the paired unit."""
    if observed is None or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    target = abs(float(np.corrcoef(rx, ry)[0, 1]))
    hits = 0
    for _ in range(n):
        if abs(float(np.corrcoef(rng.permutation(rx), ry)[0, 1])) >= target:
            hits += 1
    # Add-one correction: with a finite number of permutations a p of exactly zero is not
    # attainable evidence, only an unresolved upper bound.
    return float((hits + 1) / (n + 1))


def _bootstrap_ci(
    x: np.ndarray, y: np.ndarray, *, n: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for Spearman rho, resampling variants."""
    if np.ptp(x) == 0 or np.ptp(y) == 0 or len(x) < 3:
        return None
    values: list[float] = []
    size = len(x)
    for _ in range(n):
        idx = rng.integers(0, size, size)
        rho = _spearman(x[idx], y[idx])
        if rho is not None and not np.isnan(rho):
            values.append(rho)
    if len(values) < n // 10:
        return None
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def evaluate_arm(
    *,
    arm: str,
    measurements: Sequence[PairedMeasurement],
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    score_semantics: str,
    score_direction: str = "lower_is_tighter",
    permutations: int = DEFAULT_PERMUTATIONS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate one arm's paired scores against the measured differential.

    ``scores_a`` / ``scores_b`` map variant to that model's scalar under antigen condition A
    and B, in the same order the manifest defines ``delta``. Every variant in ``measurements``
    must be present in both; a partial arm is a different cohort and is refused rather than
    silently reduced, because a shrinking denominator across arms makes them incomparable.
    """
    missing = [m.variant for m in measurements if m.variant not in scores_a or m.variant not in scores_b]
    if missing:
        raise ValueError(
            f"arm {arm!r} is missing scores for {len(missing)} variant(s), e.g. {missing[:3]}; "
            "score every variant in the cohort or build a smaller cohort explicitly"
        )
    if score_direction not in {"lower_is_tighter", "higher_is_tighter"}:
        raise ValueError("score_direction must be 'lower_is_tighter' or 'higher_is_tighter'")

    variants = [m.variant for m in measurements]
    measured_delta = np.array([m.delta for m in measurements], dtype=float)
    a = np.array([float(scores_a[v]) for v in variants], dtype=float)
    b = np.array([float(scores_b[v]) for v in variants], dtype=float)
    predicted_delta = b - a
    if score_direction == "higher_is_tighter":
        # Re-orient so a positive predicted delta means the same thing as a positive measured
        # one. Skipping this silently inverts every correlation reported below.
        predicted_delta = -predicted_delta

    rng = np.random.default_rng(seed)
    is_constant = bool(np.ptp(predicted_delta) == 0)
    undefined_reason = "predicted differential is constant across variants" if is_constant else None

    rho = None if is_constant else _spearman(predicted_delta, measured_delta)
    result_primary: dict[str, Any] = {
        "n": len(variants),
        "spearman_rho": rho,
        "permutation_p": None
        if is_constant
        else _permutation_p(predicted_delta, measured_delta, rho, n=permutations, rng=rng),
        "bootstrap_ci_95": None
        if is_constant
        else _bootstrap_ci(predicted_delta, measured_delta, n=bootstrap, rng=rng),
        "sd_ratio": None if float(np.std(measured_delta)) == 0 else float(np.std(predicted_delta) / np.std(measured_delta)),
        # concordance() returns its full decomposition (wins/losses/ties/measured ties), which is
        # the point of using it here -- a bare agreement fraction hides how many pairs were
        # excluded as measured ties. "higher" means a larger measured delta should get a larger
        # predicted delta, which holds after the re-orientation above.
        "concordance": None
        if is_constant
        else concordance(list(measured_delta), list(predicted_delta), direction="higher"),
        "undefined_reason": undefined_reason,
    }

    exploratory_idx = [i for i, m in enumerate(measurements) if m.exploratory]
    if exploratory_idx and not is_constant:
        ex_pred, ex_meas = predicted_delta[exploratory_idx], measured_delta[exploratory_idx]
        ex_rho = _spearman(ex_pred, ex_meas)
        result_exploratory: dict[str, Any] = {
            "n": len(exploratory_idx),
            "spearman_rho": ex_rho,
            "sd_ratio": None if float(np.std(ex_meas)) == 0 else float(np.std(ex_pred) / np.std(ex_meas)),
            "claim_limit": (
                "Selected on the outcome variable, so it inflates apparent agreement. Secondary to "
                "the full-cohort result above, never the headline."
            ),
        }
    else:
        result_exploratory = {
            "n": len(exploratory_idx),
            "spearman_rho": None,
            "sd_ratio": None,
            "undefined_reason": undefined_reason or "no variants flagged exploratory",
        }

    measured_a = np.array([m.measured_a for m in measurements], dtype=float)
    measured_b = np.array([m.measured_b for m in measurements], dtype=float)
    within = {
        "condition_a": _spearman(a, measured_a),
        "condition_b": _spearman(b, measured_b),
        "note": (
            "Reported beside the differential, never as a gate on it. A model with no usable "
            "within-condition ranking makes a null differential ambiguous between 'no antigen "
            "conditioning' and 'no usable signal at all'."
        ),
    }

    return {
        "schema": SCHEMA,
        "arm": arm,
        "score_semantics": score_semantics,
        "score_direction": score_direction,
        "sensitivity": {
            "sd_predicted_delta": float(np.std(predicted_delta)),
            "is_constant": is_constant,
            "claim_limit": (
                "A screen, not a criterion. It reports rung (a) only: whether the model's output "
                "varies with the variant. A larger value does not make an arm better."
            ),
        },
        "agreement_primary": result_primary,
        "agreement_exploratory": result_exploratory,
        "within_condition": within,
        "seed": seed,
    }


def sensitivity_screen(
    arm_result: Mapping[str, Any], null_sds: Sequence[float], *, quantile: float = 0.95
) -> dict[str, Any]:
    """Compare an arm's differential spread against decoy arms.

    ``null_sds`` are ``sensitivity.sd_predicted_delta`` values from composition-matched decoy
    runs. Passing means the arm's differential varies more with the variant than under decoys.

    It does **not** mean the model read the antigen. Decoy controls narrow that gap; they do
    not close it, and no result here should be described as separating antigen-reading from
    generic perturbation.
    """
    if not null_sds:
        return {"passes": None, "undefined_reason": "no decoy runs supplied"}
    threshold = float(np.quantile(np.asarray(null_sds, dtype=float), quantile))
    observed = float(arm_result["sensitivity"]["sd_predicted_delta"])
    return {
        "observed_sd": observed,
        "decoy_threshold": threshold,
        "n_decoys": len(null_sds),
        "passes": bool(observed > threshold),
        "claim_limit": (
            "Clearing this screen establishes rung (a) for this cohort. It licenses nothing about "
            "agreement with measurement, and does not identify the mechanism."
        ),
    }


def compare_arms(arm_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank arms by agreement with measurement, keeping flat arms visible.

    Arms with an undefined primary correlation are retained in ``undefined`` with their reason,
    never sorted into the ranking as zeros.
    """
    ranked, undefined = [], []
    for result in arm_results:
        rho = result["agreement_primary"]["spearman_rho"]
        row = {
            "arm": result["arm"],
            "spearman_rho": rho,
            "sd_ratio": result["agreement_primary"]["sd_ratio"],
            "permutation_p": result["agreement_primary"]["permutation_p"],
            "sd_predicted_delta": result["sensitivity"]["sd_predicted_delta"],
            "is_constant": result["sensitivity"]["is_constant"],
        }
        (undefined if rho is None else ranked).append(
            row if rho is not None else {**row, "undefined_reason": result["agreement_primary"]["undefined_reason"]}
        )
    ranked.sort(key=lambda r: r["spearman_rho"], reverse=True)
    return {
        "schema": SCHEMA,
        "criterion": "agreement with the measured differential; sensitivity is reported but does not rank",
        "ranked": ranked,
        "undefined": undefined,
        "note": (
            "Arms in 'undefined' are baselines, not failures. An antigen-blind arm has a constant "
            "differential by construction and is the reference the ranking is read against."
        ),
    }
