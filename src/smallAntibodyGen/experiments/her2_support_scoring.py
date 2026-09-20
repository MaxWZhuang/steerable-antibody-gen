"""Signed parent-support drops, their tails, their strata and the declared decision rule.

The estimand is one line of arithmetic and everything else here exists to keep it
honest. For a parent bank row ``i`` and a checkpoint ``theta``::

    drop[i]    = log p0(y_i | scaffold) - log p_theta(y_i | scaffold)
    forward_kl = mean(drop)                       # nats per ten-residue sequence

The properties that are not obvious from that line:

* **The sign is parent minus policy**, so a positive drop means the policy made a
  parent-likely sequence less likely. Individual drops may be negative and so may
  a finite-bank mean; both are preserved rather than clipped, because clipping a
  slightly negative KL estimate to zero manufactures a one-sided claim.
* **Nonfinite scores fail the artifact.** Dropping the offending rows would leave a
  mean computed over a different population than the one the coverage table
  reports, and capping infinities would report a number the model did not produce.
* **Bootstrap indices are shared within a parent seed.** The same ``(draws, rows)``
  integer matrix is reused for every checkpoint of one seed, which is what makes a
  difference between two checkpoints *paired*. Across seeds the rows are different
  objects and sharing would be meaningless.
* **Rows are never pooled across training seeds.** Three seeds and ten thousand
  shared bank rows are different uncertainty sources; the cross-seed view is an
  equal-weight descriptive summary of three numbers, not a 30,000-row sample.
* **Ties stay together in every stratum.** Quartile edges are value thresholds, so
  a bin can be larger or smaller than a quarter and the real count is reported.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .her2_runtime import require

#: 95% two-sided normal quantile, pinned so a library change cannot move an interval.
Z_95 = 1.959963984540054

#: Declared drop thresholds. ``tenfold``/``hundredfold`` are log ratios, so a drop
#: above ``ln 10`` means the policy assigns the sequence under a tenth of the
#: parent's probability.
LN10 = math.log(10.0)
LN100 = math.log(100.0)
DEFAULT_THRESHOLDS = {"gt1": 1.0, "gt5": 5.0, "tenfold": LN10, "hundredfold": LN100}
DEFAULT_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
QUANTILE_CONVENTION = "linear"


# ---------------------------------------------------------------------------
# finiteness and the signed drop
# ---------------------------------------------------------------------------

def require_finite(values, *, label):
    """A nonfinite score fails the whole artifact, with the offending rows named."""
    array = np.asarray(values, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(array))
    require(bad.size == 0,
            f"{label}: {bad.size} of {array.size} values are nonfinite (first rows "
            f"{bad[:5].tolist()}). The scoring artifact fails; rows are not dropped and "
            "infinities are not replaced with a numerical cap.")
    return array


def signed_drop(parent_log_probability, policy_log_probability, *, label):
    """``parent - policy`` per row, in float64, both sides checked for finiteness."""
    parent = require_finite(parent_log_probability, label=f"{label}: parent log probability")
    policy = require_finite(policy_log_probability, label=f"{label}: policy log probability")
    require(parent.shape == policy.shape,
            f"{label}: parent {parent.shape} and policy {policy.shape} are not aligned")
    require(parent.ndim == 1 and parent.size > 0, f"{label}: expected a non-empty 1-D vector")
    return parent - policy


# ---------------------------------------------------------------------------
# the audit's own scoring route: every logit checked, not just the answer
# ---------------------------------------------------------------------------

def strict_sequence_log_probabilities(policy, index, *, batch_size=256, label, cached=True,
                                      progress=None):
    """Sum log probability per row, with **all twenty** logits checked at each step.

    The historical scorer checks its output. That is not sufficient here: a
    ``-inf`` on an *unselected* residue leaves the selected log probability
    perfectly finite after ``log_softmax``, and a ``+inf`` anywhere drives every
    other category to ``-inf`` while the selected one may still read as a plausible
    number. The audit's claim is about a distribution, so the whole distribution is
    checked -- raw logits, the normalized log probabilities, and the gathered
    values -- and any failure fails the artifact rather than a row.

    The forward stays float32 (that is the computation being measured); the sum
    over the ten positions is accumulated in float64. This is a strict wrapper: no
    historical scientific module is modified.
    """
    import torch
    values = np.asarray(index)
    require(values.ndim == 2 and values.shape[1] == policy.core_length,
            f"{label}: expected (N, {policy.core_length}) canonical cores, got {values.shape}")
    require(values.shape[0] > 0, f"{label}: nothing to score")
    total = np.empty(values.shape[0], dtype=np.float64)
    checked = {"rows": int(values.shape[0]), "batches": 0, "logits_checked": 0,
               "min_logit": None, "max_logit": None,
               "mode": "eval + inference_mode", "forward_dtype": "float32",
               "reduction_dtype": "float64", "path": "cached" if cached else "full"}
    was_training = bool(getattr(policy.model, "training", False))
    policy.model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, values.shape[0], int(batch_size)):
                chunk = values[start:start + int(batch_size)]
                canonical = policy.core_index(chunk)
                core_ids = policy.token_ids(chunk)
                logits = (policy.core_logits(core_ids) if cached
                          else policy.full_logits(core_ids)).float()
                _require_finite_tensor(logits, label=f"{label}: canonical logits", rows=start)
                log_probabilities = torch.log_softmax(logits, dim=-1)
                _require_finite_tensor(log_probabilities,
                                       label=f"{label}: 20-way log probabilities", rows=start)
                selected = log_probabilities.gather(2, canonical.unsqueeze(-1)).squeeze(-1)
                _require_finite_tensor(selected, label=f"{label}: selected log probabilities",
                                       rows=start)
                total[start:start + len(chunk)] = selected.double().sum(dim=1).cpu().numpy()
                checked["batches"] += 1
                checked["logits_checked"] += int(logits.numel())
                batch_min, batch_max = float(logits.min()), float(logits.max())
                checked["min_logit"] = (batch_min if checked["min_logit"] is None
                                        else min(checked["min_logit"], batch_min))
                checked["max_logit"] = (batch_max if checked["max_logit"] is None
                                        else max(checked["max_logit"], batch_max))
                if progress is not None:
                    progress.heartbeat(f"{label} {start + len(chunk)}/{values.shape[0]}")
    finally:
        if was_training:
            policy.model.train()
    require_finite(total, label=f"{label}: summed log probability")
    return {"sum_log_probability": total, "checks": checked}


def _require_finite_tensor(tensor, *, label, rows):
    import torch
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    bad = (~finite).nonzero()
    first = bad[0].tolist() if bad.numel() else []
    require(False,
            f"{label}: {int((~finite).sum())} nonfinite values, first at batch position {first} "
            f"(bank row offset {rows}). A nonfinite logit on an unselected residue still produces "
            "a finite selected value, so the whole artifact fails here; no row is dropped and no "
            "value is capped.")


# ---------------------------------------------------------------------------
# intervals
# ---------------------------------------------------------------------------

def wilson_interval(successes, total, *, z=Z_95):
    """Wilson score interval, **no** continuity correction. Convention pinned in tests."""
    successes, total = int(successes), int(total)
    require(0 <= successes <= total, f"Wilson: {successes} successes out of {total}")
    if total == 0:
        return {"successes": 0, "total": 0, "proportion": None, "lower": None, "upper": None,
                "reason": "no rows in this population", "z": z,
                "method": "wilson_score_no_continuity_correction"}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total
                           + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "total": total, "proportion": proportion,
            "lower": max(0.0, centre - spread), "upper": min(1.0, centre + spread),
            "z": z, "method": "wilson_score_no_continuity_correction"}


def bootstrap_index_matrix(rows, *, draws, seed):
    """The ``(draws, rows)`` resample matrix shared by every checkpoint of one seed.

    Materialized rather than regenerated per checkpoint: two checkpoints compared
    within a parent must be resampled on bitwise identical indices, and "same RNG
    seed, same call order" is a promise the calling code would have to keep by hand
    at every future call site.
    """
    require(isinstance(rows, int) and rows > 0, "Bootstrap needs a positive row count")
    require(isinstance(draws, int) and draws > 0, "Bootstrap needs a positive replicate count")
    generator = np.random.default_rng(int(seed))
    return generator.integers(0, rows, size=(draws, rows), dtype=np.int32)


def bootstrap_means(values, index_matrix, *, chunk=250):
    """Replicate means of ``values`` under a fixed index matrix, chunked for memory."""
    array = np.asarray(values, dtype=np.float64)
    matrix = np.asarray(index_matrix)
    require(matrix.ndim == 2 and matrix.shape[1] == array.size,
            f"Bootstrap indices {matrix.shape} do not match {array.size} rows")
    out = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], max(1, int(chunk))):
        block = matrix[start:start + max(1, int(chunk))]
        out[start:start + block.shape[0]] = array[block].mean(axis=1)
    return out


def bootstrap_summary(values, index_matrix, *, confidence=0.95, chunk=250):
    """Percentile interval of the mean, with the point estimate computed on the sample."""
    array = np.asarray(values, dtype=np.float64)
    replicates = bootstrap_means(array, index_matrix, chunk=chunk)
    low, high = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    return {"mean": float(array.mean()),
            "standard_error": (float(array.std(ddof=1) / math.sqrt(array.size))
                               if array.size > 1 else None),
            "standard_error_reason": None if array.size > 1 else "fewer than two rows",
            "bootstrap_draws": int(replicates.size),
            "bootstrap_mean_of_replicates": float(replicates.mean()),
            "ci_low": float(np.percentile(replicates, low)),
            "ci_high": float(np.percentile(replicates, high)),
            "confidence": float(confidence),
            "interval_kind": "bootstrap_percentile_confidence_interval",
            "note": ("finite-bank sampling variation at fixed models; this does not repair "
                     "historical selection and gives no simultaneous coverage over the screen")}


def paired_difference(values_a, values_b, index_matrix, *, confidence=0.95, chunk=250):
    """Mean of ``a - b`` with the *same* resampled rows on both sides."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    require(a.shape == b.shape, f"Paired comparison needs aligned vectors, got {a.shape}/{b.shape}")
    summary = bootstrap_summary(a - b, index_matrix, confidence=confidence, chunk=chunk)
    summary["pairing"] = "same bank rows and the same bootstrap indices on both sides"
    return summary


# ---------------------------------------------------------------------------
# quantiles and tails
# ---------------------------------------------------------------------------

def quantile_block(values, quantiles=DEFAULT_QUANTILES):
    """Linear-convention quantiles plus the effective row count behind the far tail."""
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, "No values to take quantiles of")
    block = {"convention": QUANTILE_CONVENTION, "rows": int(array.size), "values": {}}
    for q in quantiles:
        value = float(np.quantile(array, q, method=QUANTILE_CONVENTION))
        block["values"][repr(float(q))] = value
        block.setdefault("effective_tail_rows", {})[repr(float(q))] = {
            "rows_strictly_above": int((array > value).sum()),
            "rows_at_or_above": int((array >= value).sum()),
            "expected_rows_above": float(array.size * (1.0 - float(q)))}
    return block


def tail_block(values, thresholds=None, *, z=Z_95):
    """Strictly-greater counts, fractions and Wilson intervals for each declared cut."""
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, "No values to count tails over")
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    out = {"rows": int(array.size), "comparison": "strictly greater; exact equality is not counted",
           "thresholds": {name: float(value) for name, value in sorted(thresholds.items())},
           "counts": {}}
    for name, cut in sorted(thresholds.items()):
        successes = int((array > float(cut)).sum())
        out["counts"][name] = dict(wilson_interval(successes, array.size, z=z),
                                   threshold=float(cut),
                                   fraction=successes / array.size)
    return out


# ---------------------------------------------------------------------------
# one checkpoint against one bank
# ---------------------------------------------------------------------------

def drop_statistics(drop, index_matrix, *, thresholds=None, quantiles=DEFAULT_QUANTILES,
                    confidence=0.95, z=Z_95, chunk=250, label="drop"):
    """Everything the audit reports about one signed drop vector."""
    array = require_finite(drop, label=label)
    return {"rows": int(array.size),
            "estimator": "monte_carlo_forward_kl_from_parent_draws",
            "units": "nats per ten-residue sequence",
            "forward_kl": bootstrap_summary(array, index_matrix, confidence=confidence,
                                            chunk=chunk),
            "quantiles": quantile_block(array, quantiles),
            "tails": tail_block(array, thresholds, z=z),
            "negative_rows": int((array < 0).sum()),
            "zero_rows": int((array == 0).sum()),
            "sign_note": ("positive = the policy assigns less probability than the parent. "
                          "Negative rows and a negative mean are preserved, not clipped.")}


def log_probability_block(values, *, label):
    """Parent and policy sequence log-probability summaries, kept apart from drops."""
    array = require_finite(values, label=label)
    return {"rows": int(array.size), "mean": float(array.mean()),
            "standard_error": (float(array.std(ddof=1) / math.sqrt(array.size))
                               if array.size > 1 else None),
            "min": float(array.min()), "max": float(array.max()),
            "quantiles": quantile_block(array),
            "units": "sum log probability over the ten core positions, 20-way renormalized"}


def self_control(parent_a, parent_b, *, atol, rtol, label="parent versus itself"):
    """The parent scored against its own bank twice: per-row drops must be ~zero."""
    a = require_finite(parent_a, label=f"{label}: first pass")
    b = require_finite(parent_b, label=f"{label}: second pass")
    require(a.shape == b.shape, f"{label}: {a.shape} vs {b.shape}")
    difference = np.abs(a - b)
    allowance = float(atol) + float(rtol) * np.abs(b)
    worst = int(np.argmax(difference - allowance))
    record = {"rows": int(a.size), "max_abs_drop": float(difference.max()),
              "worst_row": worst, "worst_abs_drop": float(difference[worst]),
              "worst_allowance": float(allowance[worst]), "atol": float(atol), "rtol": float(rtol),
              "within_tolerance": bool((difference <= allowance).all())}
    require(record["within_tolerance"],
            f"{label}: max |drop| {record['max_abs_drop']:.6g} exceeds the declared route "
            f"tolerance (atol {atol:.1e}, rtol {rtol:.1e}) at row {worst}. This is a numerical "
            "investigation, not a tolerance to widen.")
    return record


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------

#: Which side of a quantile edge a row that sits exactly on it belongs to. Declared
#: once, prospectively, and shared with :func:`her2_ches.value_bins` so the audit's
#: quartiles and its CHES deciles cannot disagree about the same tie.
#:
#: ``"right"`` means: a row whose value EQUALS an edge goes into the bin ABOVE that
#: edge. Ties therefore stay together -- which is the requirement -- and the
#: consequence is that a *low* bin can be empty when the lower quantile is tied.
#: That is a real property of the data (many rows share one value), not a defect to
#: be papered over by flipping the comparison until the bins look balanced.
BIN_EDGE_SIDE = "right"


def quartile_labels(values, *, parts=4):
    """Value-threshold bins that keep ties together; bin sizes are therefore unequal.

    Equal-probability bins and equal-count bins are not the same thing when values
    repeat, and this returns the first. The edges are the ``parts``-quantiles; a row
    equal to an edge falls in the bin above it (:data:`BIN_EDGE_SIDE`), so with, say,
    four rows tied at the minimum, all four land in one bin together and ``q1`` is
    reported empty rather than split.
    """
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, "No values to bin")
    cuts = [float(np.quantile(array, k / parts, method=QUANTILE_CONVENTION))
            for k in range(1, parts)]
    assignment = np.searchsorted(np.asarray(cuts), array, side=BIN_EDGE_SIDE)
    labels = np.array([f"q{int(value) + 1}" for value in assignment], dtype=object)
    return labels, {"parts": int(parts), "edges": cuts, "convention": QUANTILE_CONVENTION,
                    "edge_side": BIN_EDGE_SIDE,
                    "ties": ("rows with equal values share a bin, so bin sizes are unequal and a "
                             "bin below a tied edge can be empty. A row equal to an edge belongs "
                             "to the bin above it.")}


def membership_labels(in_catalogue):
    """``in_training`` / ``not_in_training``; an unlabelled identity stays unlabelled."""
    flags = np.asarray(in_catalogue)
    return np.where(flags, "in_training", "not_in_training").astype(object)


def stratum_statistics(drop, labels, index_matrix, *, categories=None, thresholds=None,
                       confidence=0.95, z=Z_95, chunk=250):
    """The same drop statistics inside each stratum, with real counts and tie notes.

    A stratum with fewer than two rows reports its mean and ``null`` uncertainty
    with a reason. An empty declared category is reported as empty rather than
    omitted, so a table cannot quietly lose a bin.
    """
    array = require_finite(drop, label="stratified drop")
    labels = np.asarray(labels, dtype=object)
    require(labels.shape == array.shape, "One stratum label per row")
    matrix = np.asarray(index_matrix)
    present = sorted({str(value) for value in labels.tolist()})
    wanted = list(categories) if categories is not None else present
    for value in present:
        if value not in wanted:
            wanted.append(value)
    out = {}
    for category in wanted:
        rows = np.flatnonzero(labels.astype(str) == str(category))
        block = {"rows": int(rows.size),
                 "parent_draw_fraction": float(rows.size / array.size)}
        if rows.size == 0:
            block.update(mean=None, reason="no parent draws fall in this stratum",
                         quantiles=None, tails=None)
        elif rows.size < 2:
            block.update(mean=float(array[rows].mean()),
                         standard_error=None,
                         reason="fewer than two rows: no sampling uncertainty is estimable",
                         quantiles=quantile_block(array[rows]),
                         tails=tail_block(array[rows], thresholds, z=z))
        else:
            sub = array[rows]
            block.update(
                mean=float(sub.mean()),
                standard_error=float(sub.std(ddof=1) / math.sqrt(sub.size)),
                reason=None,
                quantiles=quantile_block(sub),
                tails=tail_block(sub, thresholds, z=z),
                forward_kl=bootstrap_summary(
                    sub, bootstrap_index_matrix(sub.size, draws=matrix.shape[0],
                                                seed=_stratum_seed(category, matrix)),
                    confidence=confidence, chunk=chunk))
        out[str(category)] = block
    return {"strata": out,
            "note": ("stratified drop statistics describe the parent draws that fall in a "
                     "stratum. They are not a measurement of total policy mass in it.")}


def stable_seed(*parts, purpose):
    """A reproducible 32-bit seed from a named purpose and stable string parts.

    ``hash()`` of a str is randomized per process by PEP 456, so a seed derived
    from it produces different bootstrap replicates on every run -- and the
    difference is invisible, because both runs look deterministic from inside.
    A digest is used instead, and the purpose string is part of it so two
    different resamples of the same category cannot collide.
    """
    payload = "|".join([str(purpose)] + [str(part) for part in parts])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _stratum_seed(category, matrix):
    """A deterministic per-stratum bootstrap seed, stable across processes.

    Within-stratum row counts differ from the full bank, so the shared full-bank
    index matrix cannot be reused directly. The seed is derived from the matrix's
    own first row and the category name by digest, so every checkpoint measured
    against the same parent bank resamples a stratum on identical indices -- which
    is what makes two checkpoints comparable inside a stratum -- while the stratum
    resample stays distinct from the full-bank one.
    """
    base = int(np.asarray(matrix)[0, :8].astype(np.int64).sum())
    return stable_seed(base, category, purpose="her2-support-audit/stratum-bootstrap")


# ---------------------------------------------------------------------------
# across seeds, and the declared decision
# ---------------------------------------------------------------------------

def equal_weight_summary(per_seed_values, *, label):
    """Descriptive mean/spread over the three training seeds. Not a 30,000-row sample."""
    values = [float(value) for value in per_seed_values if value is not None]
    if not values:
        return {"seeds": 0, "mean": None, "min": None, "max": None,
                "reason": f"no seed reported {label}"}
    array = np.asarray(values, dtype=np.float64)
    return {"seeds": int(array.size), "mean": float(array.mean()), "min": float(array.min()),
            "max": float(array.max()),
            "spread": float(array.max() - array.min()),
            "weighting": "equal weight per training seed",
            "note": ("three training seeds are three replicates. Bank rows are not pooled across "
                     "seeds and no thousands-of-replicates claim is made.")}


#: The escalation rule, verbatim, so the report and the code cannot drift apart.
ESCALATION_RULE = (
    "For each intended surviving method at its 600-second endpoint, escalate if in at least two "
    "of the three seeds the lower endpoint of a Wilson 95% interval exceeds either 1% for tenfold "
    "drops or 0.1% for hundredfold drops. These are practical investigation thresholds, not "
    "validated biological thresholds and not a multiplicity-corrected hypothesis test.")

DECISION_ESCALATE = "escalate"
DECISION_NO_ESCALATION = "no_escalation_at_this_resolution"
DECISION_INSUFFICIENT = "insufficient_coverage"


def method_decision(seed_records, *, settings):
    """Apply the declared rule to one method's per-seed 600 s endpoints.

    A seed whose endpoint is missing, unverified or incomplete is *not* a seed that
    failed the threshold. It is missing coverage, and the method's outcome becomes
    ``insufficient_coverage`` rather than a negative preservation finding.
    """
    tenfold_cut = float(settings["tenfold_wilson_lower_threshold"])
    hundredfold_cut = float(settings["hundredfold_wilson_lower_threshold"])
    needed = int(settings["min_seeds"])
    declared = int(settings["of_seeds"])
    declared_seeds = [int(seed) for seed in settings["seeds"]]
    require(len(set(declared_seeds)) == declared,
            f"The decision declares {declared} seeds but names {sorted(declared_seeds)}")
    present = [record.get("seed") for record in seed_records]
    duplicates = sorted({seed for seed in present if present.count(seed) > 1})
    require(not duplicates,
            f"Seeds {duplicates} appear more than once in this method's endpoints. Two rows for "
            "one seed would let a single training run satisfy a two-of-three rule.")
    unexpected = sorted({int(seed) for seed in present if seed is not None}
                        - set(declared_seeds))
    require(not unexpected,
            f"Endpoints for undeclared seeds {unexpected} were supplied to the decision rule")
    rows, crossing, unusable = [], 0, []
    for seed in declared_seeds:
        if seed not in [None if s is None else int(s) for s in present]:
            unusable.append({"seed": seed, "reason": "no endpoint record was produced for this "
                                                     "declared seed"})
            rows.append({"seed": seed, "usable": False,
                         "reason": "no endpoint record was produced for this declared seed",
                         "tenfold_wilson_lower": None, "hundredfold_wilson_lower": None,
                         "crosses": None})
    for record in seed_records:
        seed = record.get("seed")
        if not record.get("usable", False):
            unusable.append({"seed": seed, "reason": record.get("reason") or "not verified"})
            rows.append({"seed": seed, "usable": False, "reason": record.get("reason"),
                         "tenfold_wilson_lower": None, "hundredfold_wilson_lower": None,
                         "crosses": None})
            continue
        tenfold = record.get("tenfold_wilson_lower")
        hundredfold = record.get("hundredfold_wilson_lower")
        for name, value in (("tenfold_wilson_lower", tenfold),
                            ("hundredfold_wilson_lower", hundredfold),
                            ("tenfold_fraction", record.get("tenfold_fraction")),
                            ("hundredfold_fraction", record.get("hundredfold_fraction")),
                            ("forward_kl", record.get("forward_kl"))):
            require(value is None or math.isfinite(float(value)),
                    f"seed {seed}: {name} is {value!r}. A usable endpoint reports finite decision "
                    "inputs; a nonfinite one is a scoring failure, not a threshold result.")
        require(tenfold is not None and hundredfold is not None,
                f"seed {seed} is marked usable but supplies no Wilson lower bound for "
                f"{'tenfold' if tenfold is None else 'hundredfold'} drops")
        # Strictly greater, both here and in the tail counts: a lower bound exactly
        # equal to the threshold does not cross it.
        crosses = bool(tenfold > tenfold_cut or hundredfold > hundredfold_cut)
        crossing += int(crosses)
        rows.append({"seed": seed, "usable": True, "reason": None,
                     "tenfold_wilson_lower": tenfold,
                     "hundredfold_wilson_lower": hundredfold,
                     "tenfold_effect": record.get("tenfold_fraction"),
                     "hundredfold_effect": record.get("hundredfold_fraction"),
                     "forward_kl": record.get("forward_kl"),
                     "crosses": crosses})
    rows.sort(key=lambda row: (row["seed"] is None, row["seed"]))
    usable = sum(1 for row in rows if row["usable"])
    if crossing >= needed:
        outcome = DECISION_ESCALATE
    elif usable < declared:
        outcome = DECISION_INSUFFICIENT
    else:
        outcome = DECISION_NO_ESCALATION
    if outcome == DECISION_ESCALATE:
        note = f"the declared rule is met: {crossing} of {declared} seeds crossed, {needed} needed"
    elif outcome == DECISION_INSUFFICIENT:
        note = (f"coverage is incomplete: {usable} of {declared} declared seeds are usable. A "
                "missing, unverified or incomplete endpoint is not evidence of preservation.")
    elif crossing:
        # Reporting "none crossed" here would be false. One seed over the threshold
        # and a two-of-three rule is a rule that was not met, which is a different
        # statement from a measurement that found nothing.
        note = (f"the rule was not met: {crossing} of {declared} seeds crossed a threshold and "
                f"{needed} are required. This is not 'no seed crossed'.")
    else:
        note = ("no detected abandonment at the resolution of this audit for this method: every "
                "declared seed was measured and none crossed")
    return {"seeds": rows, "seeds_declared": declared, "seeds_usable": usable,
            "seeds_crossing": crossing, "seeds_required": needed,
            "declared_seeds": declared_seeds,
            "tenfold_wilson_lower_threshold": tenfold_cut,
            "hundredfold_wilson_lower_threshold": hundredfold_cut,
            "comparison": "strictly greater than the threshold",
            "outcome": outcome, "unusable": unusable, "outcome_note": note}


def decision_record(per_method, *, settings, coverage, completion=None, ches_summary=None):
    """The go/no-go record: every method, seed, effect, coverage and the intended target.

    Coverage is *enforced* here, not merely attached. The measured rule and launch
    eligibility are separate questions: a method can measurably fail to cross its
    threshold while the audit as a whole is still incomplete, and in that case the
    audit does not get to conclude preservation. CHES is not part of either test --
    it is not a positive gate -- but a missing required CHES stage does leave the
    audit incomplete, which is recorded below rather than folded into the rule.
    """
    methods = {name: method_decision(records, settings=settings)
               for name, records in sorted(per_method.items())}
    declared = list(settings["methods"])
    missing = [name for name in declared if name not in methods]
    outcomes = {name: block["outcome"] for name, block in methods.items()}
    coverage_complete = bool((coverage or {}).get("complete", False))
    completion_block = dict(completion or {})
    audit_complete = bool(completion_block.get("complete", False))
    blocking = []
    if missing:
        blocking.append(f"no measured endpoints for declared methods {missing}")
    if not coverage_complete:
        blocking.append("the coverage table reports gaps")
    if not audit_complete:
        blocking.append(f"required audit work is unfinished: {completion_block.get('unmet')}")
    if any(value == DECISION_INSUFFICIENT for value in outcomes.values()):
        blocking.append("at least one method has unusable seeds")
    if blocking:
        overall = DECISION_INSUFFICIENT
    elif any(value == DECISION_ESCALATE for value in outcomes.values()):
        overall = DECISION_ESCALATE
    else:
        overall = DECISION_NO_ESCALATION
    measured = ({name: block["outcome"] for name, block in methods.items()})
    return {
        "rule": ESCALATION_RULE,
        "endpoint_gpu_seconds": float(settings["endpoint_gpu_seconds"]),
        "declared_methods": declared, "methods_missing": missing,
        "methods": methods, "outcome": overall,
        "measured_rule_by_method": measured,
        "coverage_complete": coverage_complete,
        "audit_complete": audit_complete,
        "completion": completion_block,
        "blocking": blocking,
        "eligibility_note": ("the measured rule and launch eligibility are reported apart. Only a "
                             "complete audit with complete coverage can support a preservation "
                             "finding; an incomplete one reports insufficient coverage whatever "
                             "the measured tails say."),
        "intended_preservation_target": settings.get("intended_preservation_target"),
        "coverage": coverage,
        "ches": (None if ches_summary is None else
                 dict(ches_summary, role="descriptive mechanism check")),
        "ches_note": ("CHES is not a required positive gate for replay, and high CHES without "
                      "measured loss is not sufficient justification for it."),
        "resolution_caveat": ("at 10,000 draws a parent region of mass 1e-4 contributes one draw "
                              "on average and can be missed entirely. A small mean KL is not proof "
                              "that no rare region was severely suppressed."),
        "retrospective_caveat": ("these banks already contributed to the historical diversity "
                                 "assessments, so this audit is retrospective and is not an "
                                 "independent confirmation of a previously selected winner."),
        "biology_caveat": ("preservation is a distributional property. Nothing here is a binding "
                           "affinity measurement or a claim about newly generated designs.")}
