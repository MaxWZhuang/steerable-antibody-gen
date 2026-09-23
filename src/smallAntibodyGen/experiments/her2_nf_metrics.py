"""Ranking, yield, tail and uncertainty statistics, under one stated convention each.

Nothing here is new mathematics. What it adds is one place where each convention
is fixed, so two call sites cannot disagree quietly:

* **average precision** is threshold-grouped: equal scores form one block and a
  constant scorer returns exactly the prevalence. This is the campaign's
  corrected definition and it agrees with scikit-learn; the cross-check is run
  against the *installed* library rather than asserted, and the argument order
  differs between the two (``average_precision(scores, labels)`` here,
  ``average_precision_score(y_true, y_score)`` there), which is itself a mistake
  worth catching once in a test instead of silently in a report.
* **expected distinct yield** ``Y(N) = sum_x [1 - (1-q_x)^N]`` is computed on
  log probabilities with the branch that keeps precision at both ends, and it is
  defined on a fixed set of unique identities. Yields on two different panels are
  not comparable counts.
* **Wilson** intervals are two-sided 95% score intervals. A "worst upper" is the
  maximum of pointwise endpoints, not a simultaneous certificate.
* **paired seed uncertainty** uses the three paired differences with ``df = 2``.
  Row bootstraps answer a different question and are never pooled into a
  fictitious larger training sample.
* **crossovers** retain the initial sign, the search domain and every sign change
  found on the declared grid. "No crossing found" is not "always better".
"""
from __future__ import annotations

import math

import numpy as np

from .her2_eval import auroc as _inherited_auroc
from .her2_eval import average_precision as _inherited_average_precision
from .her2_nf_contract import NF_SCHEMA, tail_counts
from .her2_runtime import require

Z_95 = 1.959963984540054

#: Student-t two-sided 95% critical values for the small df this flight can have.
#: df = 2 is the three-paired-seed case and is the number the reference suite uses.
T_CRITICAL_95 = {1: 12.706204736432095, 2: 4.302652729911275, 3: 3.182446305284263,
                 4: 2.776445105198231, 5: 2.5705818366147395}

#: The published yield budgets, plus the denser grid used only for crossover search.
YIELD_BUDGETS = (1_000, 10_000, 100_000, 300_000, 1_000_000, 3_000_000)


def dense_yield_grid(*, low=1_000, high=3_000_000, points=61):
    """A log-spaced crossover-diagnostic grid. Declared, so "the grid" is not a choice."""
    return tuple(int(round(value)) for value in
                 np.unique(np.round(np.logspace(math.log10(low), math.log10(high), int(points)))))


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def average_precision(scores, labels):
    """Threshold-grouped AP. Argument order is ``(scores, labels)``, as in the campaign."""
    return _inherited_average_precision(scores, labels)


def auroc(scores, labels):
    """Tie-aware AUROC; ``None`` when a single class is present."""
    return _inherited_auroc(scores, labels)


def sklearn_average_precision(scores, labels):
    """The installed scikit-learn value, or a recorded reason it is unavailable.

    Deliberately not a fallback: if the library is present the comparison is made
    against it, and if it is absent that fact is recorded rather than replaced by
    the repository implementation checking itself.
    """
    try:
        from sklearn.metrics import average_precision_score
    except ImportError as error:                                # pragma: no cover - env dependent
        return {"available": False, "value": None, "reason": f"{type(error).__name__}: {error}"}
    import sklearn
    value = float(average_precision_score(np.asarray(labels).astype(int),
                                          np.asarray(scores, dtype=np.float64)))
    return {"available": True, "value": value, "sklearn_version": str(sklearn.__version__),
            "argument_order": "sklearn takes (y_true, y_score); this module takes (scores, labels)"}


def average_precision_crosscheck(scores, labels, *, atol=1e-12):
    """Compare the repository AP with scikit-learn's on the same vectors.

    A divergence is a failure, not a convention difference: the two definitions
    agree on tied scores, and 80 tie-heavy cases were measured to agree to
    1.11e-16. This returns the comparison rather than asserting it so a caller
    can record the measured error.
    """
    ours = average_precision(scores, labels)
    theirs = sklearn_average_precision(scores, labels)
    if not theirs["available"]:
        return {"repository": ours, "sklearn": theirs, "agrees": None,
                "difference": None, "atol": float(atol)}
    difference = abs(ours - theirs["value"])
    return {"repository": ours, "sklearn": theirs, "difference": float(difference),
            "atol": float(atol), "agrees": bool(difference <= float(atol)),
            "note": ("threshold-grouped AP and sklearn's average_precision_score are the same "
                     "statistic. A disagreement is a defect in one of the two implementations.")}


def rank_block(scores, labels, *, prevalence_note=True):
    """AP, AUROC, prevalence and counts for one population and one scorer."""
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels).astype(bool)
    require(values.shape == positive.shape, "Scores and labels must align")
    block = {"rows": int(values.size), "positives": int(positive.sum()),
             "prevalence": float(positive.mean()) if values.size else None,
             "average_precision": average_precision(values, positive),
             "auroc": auroc(values, positive)}
    if prevalence_note:
        block["tie_convention"] = ("threshold-grouped: equal scores are one block, so a constant "
                                   "scorer returns the prevalence and not 0.5")
    return block


def stratified_rank_block(scores, labels, strata, *, categories=None):
    """Per-stratum AP/AUROC plus the pooled value and a macro average.

    The macro average is reported with the strata it averaged over named, because
    after an exact purge the old d1/d2/far macro is not the same statistic: every
    E row is distance >= 3 from ``T_purge``.
    """
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels).astype(bool)
    keys = np.asarray(strata)
    names = tuple(categories) if categories is not None else tuple(sorted(set(keys.tolist())))
    per_stratum = {}
    for name in names:
        mask = keys == name
        if not mask.any():
            per_stratum[str(name)] = {"rows": 0, "average_precision": None, "auroc": None,
                                      "reason": "no rows in this stratum"}
            continue
        per_stratum[str(name)] = rank_block(values[mask], positive[mask], prevalence_note=False)
    usable = [block["average_precision"] for block in per_stratum.values()
              if block.get("average_precision") is not None and block.get("rows")]
    return {"per_stratum": per_stratum,
            "pooled": rank_block(values, positive, prevalence_note=False),
            "macro_average_precision": float(np.mean(usable)) if usable else None,
            "macro_over": [str(name) for name in names],
            "macro_note": ("the macro average is over the named strata only. It is not comparable "
                           "with a macro over different strata, and after the exact purge the "
                           "original proximity bins are diagnostics, not the primary endpoint.")}


def class_mass(cores, *, class_of, denominator_label):
    """Mass on each assayed class, with the denominator that was actually used.

    ``class_of`` maps a row id to a class or to ``None`` for an unassayed core.
    Purity numerators and denominators are returned together; an "ANY-hit
    fraction times purity" estimate built from mismatched denominators is the
    error this shape exists to prevent.
    """
    total = 0
    counts = {}
    unassayed = 0
    for core in cores:
        total += 1
        label = class_of.get(core)
        if label is None:
            unassayed += 1
            continue
        counts[label] = counts.get(label, 0) + 1
    assayed = total - unassayed
    return {"draws": int(total), "assayed_hits": int(assayed), "unassayed": int(unassayed),
            "counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "mass_over_all_draws": {str(k): v / float(total) for k, v in sorted(counts.items())}
            if total else {},
            "purity_over_assayed_hits": {str(k): v / float(assayed)
                                         for k, v in sorted(counts.items())} if assayed else {},
            "denominator_label": str(denominator_label),
            "denominator_note": ("mass is over ALL draws; purity is over assayed hits only. The "
                                 "two denominators are different and are never multiplied "
                                 "together to estimate a known-high yield.")}


# ---------------------------------------------------------------------------
# yield
# ---------------------------------------------------------------------------

def expected_distinct_yield(log_probabilities, draws):
    """``sum_x [1 - (1-q_x)^N]`` over a fixed set of unique identities.

    The branch at ``ell <= -ln 2`` is not cosmetic: ``log1p(-exp(ell))`` loses all
    precision as ``q -> 1`` and ``log(-expm1(ell))`` loses it as ``q -> 0``, and
    the rare-identity end is exactly where the sum lives.
    """
    values = np.asarray(log_probabilities, dtype=np.float64)
    draws = int(draws)
    require(draws >= 0, "Negative draw budget")
    require(bool((values <= 0).all()), "A log probability above zero is not a probability")
    if draws == 0 or values.size == 0:
        return 0.0
    log_one_minus = np.where(values <= -math.log(2.0),
                             np.log1p(-np.exp(np.minimum(values, -1e-300))),
                             np.log(-np.expm1(np.where(values < 0, values, -1e-300))))
    terms = -np.expm1(draws * log_one_minus)
    terms = np.where(np.isneginf(values), 0.0, terms)
    terms = np.where(values == 0.0, 1.0, terms)
    return float(np.sum(terms))


def yield_curve(log_probabilities, *, budgets=YIELD_BUDGETS):
    """``{N: Y(N)}`` on the declared grid, plus the identity count it is defined on."""
    values = np.asarray(log_probabilities, dtype=np.float64)
    return {"identities": int(values.size),
            "curve": {str(int(n)): expected_distinct_yield(values, int(n)) for n in budgets},
            "budgets": [int(n) for n in budgets],
            "panel_note": ("Y(N) counts distinct identities in a FIXED set. Two panels of "
                           "different size produce incomparable absolute counts; compare "
                           "parent-relative curves on the same panel.")}


def mixture_yield_bound(parent_yield, policy_yield, alpha):
    """``Y_M >= (1-alpha) Y_P + alpha Y_Q``, from concavity of ``1-(1-q)^N``.

    An analytic bound conditional on the probability contract, not a measurement
    of the mixture. The actual mixture value is computed from the mixture scores.
    """
    alpha = float(alpha)
    require(0.0 <= alpha <= 1.0, "alpha outside [0, 1]")
    return {"alpha": alpha, "parent_yield": float(parent_yield),
            "policy_yield": float(policy_yield),
            "lower_bound": (1.0 - alpha) * float(parent_yield) + alpha * float(policy_yield),
            "basis": "f(q) = 1 - (1-q)^N is concave in q for integer N >= 1",
            "status": "analytic lower bound, not a measured mixture yield"}


def crossing_brackets(budgets, differences, *, atol=1e-9):
    """Every sign change on THIS grid, with the initial sign and the search domain."""
    budgets = [float(value) for value in budgets]
    differences = [float(value) for value in differences]
    require(len(budgets) == len(differences) and len(budgets) >= 2, "Invalid crossover grid")
    nonzero = [(n, 1 if d > 0 else -1) for n, d in zip(budgets, differences) if abs(d) > atol]
    brackets = []
    for (left, a), (right, b) in zip(nonzero, nonzero[1:]):
        if a != b:
            brackets.append({"left": left, "right": right, "from_sign": a, "to_sign": b})
    return {"domain": [budgets[0], budgets[-1]],
            "first_nonzero_sign": nonzero[0][1] if nonzero else 0,
            "brackets": brackets, "atol": float(atol),
            "skipped_numerical_ties": int(len(budgets) - len(nonzero)),
            "interpretation": ("sign changes found on this grid. 'No crossing found' does NOT "
                               "mean 'always better'; near the common support ceiling a numerical "
                               "tie is skipped explicitly rather than reported as a crossing.")}


def refine_bracket(bracket, difference_at, *, iterations=20, atol=1e-9):
    """Bisect one bracket in log-budget space and report the refined interval.

    ``difference_at(n)`` returns the signed difference at an integer budget. The
    refinement narrows the interval; it does not prove uniqueness of the root
    inside it.
    """
    left, right = float(bracket["left"]), float(bracket["right"])
    sign_left = int(bracket["from_sign"])
    history = []
    for _ in range(int(iterations)):
        middle = int(round(math.exp(0.5 * (math.log(left) + math.log(right)))))
        if middle <= left or middle >= right:
            break
        value = float(difference_at(middle))
        history.append({"budget": middle, "difference": value})
        if abs(value) <= atol:
            left = right = float(middle)
            break
        if (1 if value > 0 else -1) == sign_left:
            left = float(middle)
        else:
            right = float(middle)
    return {"bracket": dict(bracket), "refined": [left, right], "evaluations": history,
            "iterations": len(history),
            "claim": "a narrowed interval containing at least one sign change on this function; "
                     "not a proof that it contains exactly one root"}


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------

def wilson_interval(count, total, *, z=Z_95):
    """Two-sided 95% Wilson score endpoints, as fractions."""
    count, total = int(count), int(total)
    require(total > 0 and 0 <= count <= total, "Invalid count/total for a Wilson interval")
    p = count / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": count, "total": total, "rate": p,
            "lower": max(0.0, center - half), "upper": min(1.0, center + half),
            "kind": "two-sided 95% Wilson score interval, no continuity correction", "z": float(z)}


def worst_upper(intervals):
    """The maximum pointwise upper endpoint, labelled as what it is."""
    values = [float(block["upper"]) for block in intervals]
    require(values, "No intervals supplied")
    return {"worst_upper": max(values), "entries": len(values),
            "claim": ("the maximum of pointwise two-sided 95% endpoints. This is NOT a "
                      "simultaneous 95% guarantee across arms, seeds and thresholds; a "
                      "simultaneous certificate needs a predeclared family and adjusted bounds.")}


def paired_t(values, *, confidence=0.95):
    """Mean paired difference with a nominal Student-t interval, ``df = n - 1``."""
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size >= 2, "A paired t interval needs at least two pairs")
    degrees = int(array.size - 1)
    require(degrees in T_CRITICAL_95 or confidence != 0.95,
            f"No tabulated 95% critical value for df={degrees}")
    critical = T_CRITICAL_95[degrees]
    center = float(array.mean())
    half = critical * float(array.std(ddof=1)) / math.sqrt(array.size)
    return {"raw_differences": [float(value) for value in array],
            "mean": center, "lower": center - half, "upper": center + half,
            "degrees_of_freedom": degrees, "t_critical": critical,
            "excludes_zero": bool((center - half) * (center + half) > 0),
            "power_note": ("three trained seeds give df = 2. Failure to exclude zero is not "
                           "equivalence, and this interval is not a familywise statement.")}


def paired_bootstrap(values_a, values_b, *, draws=2000, seed, confidence=0.95):
    """Paired sequence resampling of a within-bank difference. A different question.

    This describes Monte Carlo sensitivity of a statistic computed on one bank.
    It does not describe variation across trained seeds and is never pooled with
    the seed-level interval.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1, "Paired bootstrap needs aligned vectors")
    generator = np.random.default_rng(int(seed))
    difference = a - b
    samples = np.empty(int(draws), dtype=np.float64)
    for position in range(int(draws)):
        rows = generator.integers(0, difference.size, difference.size)
        samples[position] = difference[rows].mean()
    low = float(np.quantile(samples, (1.0 - confidence) / 2.0))
    high = float(np.quantile(samples, 1.0 - (1.0 - confidence) / 2.0))
    return {"mean": float(difference.mean()), "lower": low, "upper": high,
            "draws": int(draws), "rows": int(difference.size), "seed": int(seed),
            "kind": "paired row bootstrap within one bank",
            "not": "a statement about training-seed variability; rows are not extra models"}


def drop_block(parent_log_probability, policy_log_probability, *, inclusive=True):
    """Probability-loss summary: mean log ratio, quantiles, inclusive tail rates.

    The mean log ratio is a Monte-Carlo estimate of ``KL(P || Q)`` when the rows
    are independent parent draws. The per-position conditional KL is a different
    estimator of the same population quantity and is computed elsewhere; both are
    saved with their difference rather than forced to agree on a finite bank.
    """
    parent = np.asarray(parent_log_probability, dtype=np.float64)
    policy = np.asarray(policy_log_probability, dtype=np.float64)
    require(parent.shape == policy.shape, "Parent and policy vectors must align")
    finite = np.isfinite(parent) & np.isfinite(policy)
    require(bool(finite.all()),
            f"{int((~finite).sum())} nonfinite score(s) on the preservation bank. That is an "
            "observation about the policy, not a tail count, and the caller must record it.")
    drop = parent - policy
    quantiles = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)
    tails = tail_counts(drop, inclusive=inclusive)
    intervals = {name: wilson_interval(block["count"], int(drop.size))
                 for name, block in tails["events"].items()}
    return {"rows": int(drop.size),
            "mean_log_ratio": float(drop.mean()),
            "standard_error": float(drop.std(ddof=1) / math.sqrt(drop.size)),
            "quantiles": {str(q): float(np.quantile(drop, q)) for q in quantiles},
            "modest_loss_rates": {"gt_0": float((drop > 0).mean()),
                                  "ge_0p5_nats": float((drop >= 0.5).mean()),
                                  "ge_1_nat": float((drop >= 1.0).mean()),
                                  "ge_2_nats": float((drop >= 2.0).mean())},
            "tails": tails, "wilson": intervals,
            "estimator_note": ("mean(log p_P - log p_Q) over independent parent draws estimates "
                               "KL(P||Q). The summed per-position conditional KL estimates the "
                               "same population quantity and is not equal on a finite bank.")}


def summary_record(label, block):
    """Wrap any of the blocks above with the flight's schema and a label."""
    return {"schema_version": NF_SCHEMA, "record_kind": "metric_block", "label": str(label),
            **dict(block)}
