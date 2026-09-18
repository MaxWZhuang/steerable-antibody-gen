"""One metric implementation for every HER2 scorer, plus generation and assay endpoints.

Every comparator -- prior, distance, nearest neighbour, linear, CNN, zero-shot
p-IgGen, both post-trained arms -- passes through :func:`rank_metrics` on the
*same rows*. That is the point of the module: a difference between two scorers
here can never be a difference between two metric implementations.

Conventions that are fixed rather than discovered:

* **Ties are handled, not broken by luck.** AUROC uses average ranks; average
  precision accumulates at distinct-score group boundaries; precision@k sorts by
  ``(-score, sequence)`` so the reported set is reproducible byte for byte.
* **Degenerate strata are stated, not hidden.** With one class present AUROC is
  undefined and reported as ``None`` with a reason; average precision is the
  prevalence (1.0 for a positives-only cohort), which is a constant, not a score.
* **A bootstrap interval and a permutation interval are different objects.** The
  Spearman confidence interval comes from resampling pairs; a permutation
  distribution is a null, and if it is reported it is reported as a p-value under
  a separate name.
* **Monte Carlo KL and entropy use the SUM log probability** over the ten
  positions -- the quantity whose expectation is the entropy -- while ranking and
  per-residue NLL use the mean. Both are recorded under names that say which.
* **No measured affinity is ever attached to an unassayed sequence.**
  :func:`generation_record` has a closed key set and refuses affinity-valued
  fields outright.

Scope note (Codex, round 3): the three-class CNN and the additive linear model are
**auxiliary ranking comparators on labelled populations only** -- the reserved test
split and the independent assay cohort. They are not the generator, not a reward
model, not a preference source and not a selection criterion. Scoring *generated*
draws with them is optional and is deliberately **not** produced: a proxy
probability on an unassayed design is not a measurement, and leaving it out keeps
the generation tables to counts, distances and measured catalogue lookups.
:data:`GENERATION_RECORD_FIELDS` still reserves ``classifier_proxy_p_high`` for a
caller that wants it, but nothing in this pipeline emits it.
"""
from __future__ import annotations

import numpy as np

from .her2_data import (CORE_LENGTH, WT_CORE, decode_cores, encode_cores, hamming_to,
                        mean_pairwise_hamming, nearest_training_labels, site_entropy)
from .her2_runtime import require

DEFAULT_K_VALUES = (32, 100, 1000)
AUROC_UNDEFINED = "auroc_undefined_single_class"


def _positives(labels):
    values = np.asarray(labels)
    if values.dtype == bool:
        return values
    require(bool(np.isin(values, (0, 1)).all()), "Binary labels must be 0/1 or boolean")
    return values.astype(bool)


def auroc(scores, labels):
    """Tie-aware AUROC by average ranks. ``None`` when only one class is present."""
    from scipy.stats import rankdata
    values = np.asarray(scores, dtype=np.float64)
    positive = _positives(labels)
    require(values.shape == positive.shape, "Scores and labels must align")
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(values)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores, labels):
    """Tie-aware AP: precision/recall are evaluated at distinct-score boundaries.

    Ranking within a block of equal scores is arbitrary, so the block is scored as
    a block. A constant scorer therefore returns exactly the prevalence -- not
    0.5, which is the AUROC null and a different number.
    """
    values = np.asarray(scores, dtype=np.float64)
    positive = _positives(labels)
    n_pos = int(positive.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    hits = np.cumsum(positive[order])
    ends = np.concatenate([np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1,
                           [values.size]])
    true_positive = hits[ends - 1]
    precision = true_positive / ends
    recall = true_positive / n_pos
    previous = np.concatenate([[0.0], recall[:-1]])
    return float(((recall - previous) * precision).sum())


def precision_at_k(scores, labels, sequences, k):
    """Deterministic top-k precision: ``(-score, sequence)`` ordering, no random tie-break."""
    values = np.asarray(scores, dtype=np.float64)
    positive = _positives(labels)
    keys = np.asarray(sequences, dtype=str)
    require(keys.shape == values.shape, "One sequence per score is required for a stable top-k")
    if k > values.size:
        return None
    order = np.lexsort((keys, -values))[:k]
    return float(positive[order].mean())


def rank_metrics(scores, labels, sequences, *, k_values=DEFAULT_K_VALUES):
    """The single metric function. Every scorer and every stratum goes through it."""
    values = np.asarray(scores, dtype=np.float64)
    positive = _positives(labels)
    require(values.size > 0, "rank_metrics needs at least one row")
    require(bool(np.isfinite(values).all()), "Nonfinite score")
    n_pos = int(positive.sum())
    document = {
        "n": int(values.size), "positives": n_pos, "negatives": int(values.size - n_pos),
        "prevalence": float(positive.mean()) if values.size else None,
        "auroc": auroc(values, positive),
        "average_precision": average_precision(values, positive),
        "precision_at_k": {str(k): precision_at_k(values, positive, sequences, k)
                           for k in k_values},
        "distinct_scores": int(np.unique(values).size),
    }
    if document["auroc"] is None:
        document["auroc_note"] = AUROC_UNDEFINED
    if document["distinct_scores"] == 1:
        document["constant_scorer"] = True
    return document


def require_reference_diagnostics(records, artifacts):
    """Refuse a freeze made after individual scoring but before the reference pass.

    Validation persists each policy before measuring KL and relative rankings.
    An interruption between those steps leaves valid files but incomplete evidence.
    Both the freezer and final evaluator call this before reserved outcomes open.
    """
    zero = "piggen_zeroshot"
    for name, entry in artifacts.items():
        record = records[name]
        generation = record.get("generation") or {}
        expected = [("kl_to_zero_shot", zero)]
        parent = entry.get("parent_name")
        if parent:
            expected.append(("kl_to_sft_parent", parent))
        for key, reference in expected:
            measured = generation.get(key) or {}
            require(measured.get("reference_name") == reference
                    and measured.get("reference_sha256") == artifacts[reference]["sha256"]
                    and measured.get("draws_sha256") == record["generation_samples"]["sha256"]
                    and isinstance(measured.get("kl_nats"), (int, float))
                    and np.isfinite(measured["kl_nats"]),
                    f"{name}: incomplete or stale {key}; finish validation before freezing")
        for key, reference in (("val_metrics_minus_zero_shot", zero),
                               ("val_metrics_minus_parent", parent)):
            if reference and reference != name:
                measured = record.get(key) or {}
                require(measured.get("reference_name") == reference
                        and "average_precision" in measured,
                        f"{name}: incomplete {key}; finish validation before freezing")


def implicit_reward_metrics(policy_scores, reference_scores, labels, sequences, *,
                            k_values=DEFAULT_K_VALUES, reference_name=None):
    """Ranking metrics for ``log pi - log pi_ref`` -- DPO's implicit reward.

    DPO's implicit reward is ``beta * (log pi - log pi_ref)``, and ``beta`` is
    positive, so it is a rescaling that cannot reorder anything: the difference is
    the ranking, and it is reported without a beta rather than with an arbitrary
    one. Which reference is subtracted decides what the number means -- the
    policy's *own* SFT parent gives the quantity the objective actually optimized,
    while the zero-shot model gives the effect of the whole post-training path.
    Both are reported, under names that say which.

    This is a diagnostic. Selection is by raw-density validation average precision
    and that rule is unchanged; picking whichever scoring rule ranks best would be
    the selection-on-the-outcome this whole protocol is arranged to avoid.
    """
    left = np.asarray(policy_scores, dtype=np.float64)
    right = np.asarray(reference_scores, dtype=np.float64)
    require(left.shape == right.shape, "Policy and reference scores must align")
    document = rank_metrics(left - right, labels, sequences, k_values=k_values)
    document["scoring_rule"] = "policy_minus_reference_log_density"
    document["reference_name"] = reference_name
    document["note"] = ("DPO implicit-reward ranking, beta omitted because a positive scale "
                        "cannot change an order; diagnostic only, never the selection rule")
    return document


def stratified_metrics(scores, labels, sequences, strata, *, k_values=(32,), categories=None):
    """Per-stratum metrics; a declared-but-empty stratum is reported, not dropped.

    Pass ``categories`` when the stratum set is fixed in advance -- an absent
    stratum then shows up as ``n: 0`` instead of silently vanishing from the
    table, which is how a missing population becomes invisible.
    """
    keys = np.asarray(strata, dtype=str)
    values = np.asarray(scores, dtype=np.float64)
    positive = _positives(labels)
    sequence_array = np.asarray(sequences, dtype=str)
    wanted = sorted(set(keys.tolist()) | set(categories or ()))
    out = {}
    for key in wanted:
        mask = keys == key
        if not mask.any():
            out[key] = {"n": 0, "note": "empty_stratum"}
            continue
        out[key] = rank_metrics(values[mask], positive[mask], sequence_array[mask],
                                k_values=k_values)
    return out


def bootstrap_paired_differences(score_map, labels, sequences, pairs, *, draws=1000, seed,
                                 confidence=0.95):
    """Paired bootstrap over rows, with common random numbers across scorers.

    The same resampled row set is used for every scorer inside one draw, which is
    what makes the difference paired. Percentile intervals; no bias correction is
    claimed.
    """
    positive = _positives(labels)
    keys = np.asarray(sequences, dtype=str)
    needed = sorted({name for pair in pairs for name in pair})
    for name in needed:
        require(name in score_map, f"Bootstrap needs a score vector for {name!r}")
    rng = np.random.default_rng(seed)
    total = positive.size
    # The OBSERVED difference is computed once on the original sample. The mean of
    # the bootstrap replicates is a different quantity -- it carries the resampling
    # bias -- so it is reported separately under its own name rather than standing
    # in for the point estimate.
    point = {name: (auroc(score_map[name], positive), average_precision(score_map[name], positive))
             for name in needed}
    collected = {f"{a}_minus_{b}": {"auroc": [], "average_precision": []} for a, b in pairs}
    usable = 0
    for _ in range(draws):
        rows = rng.integers(0, total, total)
        sample_labels = positive[rows]
        if sample_labels.all() or not sample_labels.any():
            continue
        usable += 1
        sample_sequences = keys[rows]
        cache = {name: (auroc(score_map[name][rows], sample_labels),
                        average_precision(score_map[name][rows], sample_labels))
                 for name in needed}
        for a, b in pairs:
            collected[f"{a}_minus_{b}"]["auroc"].append(cache[a][0] - cache[b][0])
            collected[f"{a}_minus_{b}"]["average_precision"].append(cache[a][1] - cache[b][1])
    low, high = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    summary = {"draws_requested": int(draws), "draws_used": usable, "confidence": confidence,
               "observed_is": "difference on the original sample, not the bootstrap mean",
               "differences": {}}
    metric_position = {"auroc": 0, "average_precision": 1}
    for (a, b) in pairs:
        name = f"{a}_minus_{b}"
        entry = {}
        for metric, samples in collected[name].items():
            left, right = point[a][metric_position[metric]], point[b][metric_position[metric]]
            observed = None if left is None or right is None else float(left - right)
            entry[metric] = {
                "observed": observed,
                "bootstrap_mean": float(np.mean(samples)) if samples else None,
                "ci_low": float(np.percentile(samples, low)) if samples else None,
                "ci_high": float(np.percentile(samples, high)) if samples else None,
                "excludes_zero": bool(samples) and bool(
                    np.percentile(samples, low) > 0 or np.percentile(samples, high) < 0)}
        summary["differences"][name] = entry
    return summary


# ---------------------------------------------------------------------------
# correlation with bootstrap interval
# ---------------------------------------------------------------------------

def spearman(x, y):
    from scipy.stats import rankdata
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.size >= 3, "Spearman needs at least three aligned pairs")
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_spearman(x, y, *, draws=2000, seed, confidence=0.95):
    """Percentile CI by resampling pairs. This is a confidence interval.

    A permutation distribution answers a different question (what rho looks like
    under no association) and its quantiles are a null interval, not a CI. If a
    permutation p-value is wanted it is computed separately and named separately.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    point = spearman(a, b)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        rows = rng.integers(0, a.size, a.size)
        value = spearman(a[rows], b[rows])
        if value is not None:
            samples.append(value)
    low, high = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    return {"spearman": point, "n": int(a.size), "draws_requested": int(draws),
            "draws_used": len(samples), "confidence": confidence,
            "ci_low": float(np.percentile(samples, low)) if samples else None,
            "ci_high": float(np.percentile(samples, high)) if samples else None,
            "interval_kind": "bootstrap_percentile_confidence_interval"}


def permutation_p_value(x, y, *, draws=2000, seed):
    """Two-sided permutation p-value for Spearman. A null, reported as a p-value."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    observed = spearman(a, b)
    if observed is None:
        return {"p_value": None, "note": "degenerate_ranks"}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(draws):
        value = spearman(a, rng.permutation(b))
        if value is not None and abs(value) >= abs(observed):
            extreme += 1
    return {"p_value": (extreme + 1) / (draws + 1), "draws": int(draws),
            "interval_kind": "permutation_null_not_a_confidence_interval"}


def clopper_pearson(successes, trials, *, confidence=0.95):
    """Exact binomial interval, including the ``successes == 0`` branch."""
    from scipy.stats import beta
    k, n = int(successes), int(trials)
    require(0 <= k <= n, "Invalid binomial counts")
    if n == 0:
        return {"rate": None, "ci_low": None, "ci_high": None, "successes": k, "trials": 0}
    alpha = 1 - confidence
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"rate": k / n, "ci_low": low, "ci_high": high, "successes": k, "trials": n,
            "confidence": confidence}


# ---------------------------------------------------------------------------
# generation diagnostics
# ---------------------------------------------------------------------------

def monte_carlo_kl(policy_sum_log_probability, reference_sum_log_probability):
    """MC estimate of KL(policy || reference) from policy draws, with its SE.

    This is an estimate from a finite draw count, not an exact KL: the support is
    20^10, so no exhaustive sum exists. Under fixed IID draws the sample mean is
    unbiased for the KL, so the SE below is the whole story about the sampling
    error -- there is no separate finite-N bias term to quote. The estimate can
    come out negative when the true KL is near zero; that is sampling error and
    is reported as measured rather than clamped to a false exact nonnegativity.
    """
    policy = np.asarray(policy_sum_log_probability, dtype=np.float64)
    reference = np.asarray(reference_sum_log_probability, dtype=np.float64)
    require(policy.shape == reference.shape and policy.size > 1, "Aligned draw log probabilities")
    difference = policy - reference
    return {"kl_nats": float(difference.mean()),
            "standard_error": float(difference.std(ddof=1) / np.sqrt(difference.size)),
            "draws": int(difference.size), "estimator": "monte_carlo_sum_log_probability"}


def empirical_entropy(sum_log_probability):
    values = np.asarray(sum_log_probability, dtype=np.float64)
    return {"entropy_nats": float(-values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(values.size)),
            "draws": int(values.size), "estimator": "monte_carlo_sum_log_probability"}


def split_conditional_hits(cores, split_of, class_of, *, heldout_splits=("val", "test")):
    """Exact catalogue hits and conditional high rate, **per split** and held-out only.

    Pooling the three splits is the specific mistake this replaces: a policy that
    has memorized its training positives produces a high pooled conditional rate
    that says nothing about generalization. Every split is reported separately,
    the held-out splits are aggregated on their own, and the pooled number is kept
    only under a name that says it is pooled.

    These are exact catalogue lookups of sequences that already carry a published
    bin. No new assay is performed and no unmeasured sequence acquires a label.
    """
    draws = len(cores)
    per_split, matched_total, high_total, labelled_total = {}, 0, 0, 0
    heldout_labelled, heldout_high, heldout_matched = 0, 0, 0
    labelled_heldout_splits = []
    for name in sorted(set(split_of.values())):
        hits = [core for core in cores if split_of.get(core) == name]
        labelled = [core for core in hits if core in class_of]
        high = sum(1 for core in labelled if class_of[core] == "high")
        per_split[name] = {"draws_matching": len(hits),
                           "unique_cores_matching": len(set(hits)),
                           "labels_available": bool(len(labelled) == len(hits)),
                           "draws_with_labels": len(labelled),
                           "conditional_high": clopper_pearson(high, len(labelled)),
                           "hit_rate_over_all_draws": clopper_pearson(len(hits), draws)}
        matched_total += len(hits)
        labelled_total += len(labelled)
        high_total += high
        if name in heldout_splits:
            heldout_matched += len(hits)
            heldout_labelled += len(labelled)
            heldout_high += high
            if len(labelled) == len(hits):
                labelled_heldout_splits.append(name)
    return {
        "draws": draws,
        "by_split": per_split,
        "heldout": {"splits": list(heldout_splits),
                    "splits_with_labels_read": sorted(labelled_heldout_splits),
                    "draws_matching": heldout_matched,
                    "draws_with_labels": heldout_labelled,
                    "conditional_high": clopper_pearson(heldout_high, heldout_labelled),
                    "hit_rate_over_all_draws": clopper_pearson(heldout_matched, draws)},
        "pooled": {"draws_matching": matched_total,
                   "draws_with_labels": labelled_total,
                   "conditional_high": clopper_pearson(high_total, labelled_total),
                   "note": ("pooled across every split whose labels this stage may read; it "
                            "includes memorized training rows and is NOT a generalization "
                            "measurement. Use `heldout`.")},
        "note": ("counts over exact matches to published catalogue rows, which already carry a "
                 "published bin -- no new assay is performed. An empty intersection is reported "
                 "as zero with an exact interval, never as a missing key."),
    }


def diversity_reference(index):
    """The three distribution statistics every diversity gate is expressed against."""
    values = np.asarray(index)
    return {"mean_pairwise_hamming": mean_pairwise_hamming(values),
            "sum_site_entropy_nats": float(sum(site_entropy(values))),
            "unique_fraction": float(len(set(decode_cores(values))) / values.shape[0])}


def generation_diagnostics(index, sum_log_probability, *, catalogs, train_index, train_labels,
                           split_of, class_of, reference_sum_log_probability=None,
                           parent_sum_log_probability=None, max_train_distance=2,
                           heldout_splits=("val", "test")):
    """Distribution, novelty and measured-overlap statistics for one policy's draws.

    ``split_of``/``class_of`` map a core to the split it came from and its
    published bin, so overlap is reported per split rather than pooled. Draws that
    match nothing are reported as an explicit zero with an interval, never as a
    missing key, because "no overlap" is the outcome most likely to be silently
    dropped.
    """
    values = np.asarray(index)
    cores = decode_cores(values)
    counts = {}
    for core in cores:
        counts[core] = counts.get(core, 0) + 1
    unique = set(cores)
    lookup = nearest_training_labels(values, np.asarray(train_index), np.asarray(train_labels),
                                     max_distance=max_train_distance)
    wt_distance = hamming_to(values, encode_cores([WT_CORE])[0])
    train_catalog = set(catalogs.get("train", ()))
    not_exact_train = sum(1 for core in cores if core not in train_catalog)
    entropy_per_site = site_entropy(values)
    document = {
        "draws": len(cores), "unique_cores": len(unique),
        "duplicate_draws": len(cores) - len(unique),
        "unique_fraction": float(len(unique) / len(cores)),
        "max_single_core_frequency": float(max(counts.values()) / len(cores)),
        "max_single_core_count": int(max(counts.values())),
        "duplicates_retained": True,
        "catalog_overlap": {name: {"draws_in_catalog": int(sum(1 for c in cores if c in catalog)),
                                   "unique_cores_in_catalog": int(len(unique & set(catalog)))}
                            for name, catalog in catalogs.items()},
        "exact_train_novelty": {
            "draws_not_in_training_cores": int(not_exact_train),
            "fraction_not_in_training_cores": float(not_exact_train / len(cores))},
        "labelled_hits": split_conditional_hits(cores, split_of, class_of,
                                                heldout_splits=heldout_splits),
        "min_train_hamming": {
            "distribution": {str(k): int(v) for k, v in
                             zip(*np.unique(lookup.strata(), return_counts=True))},
            "mean_resolved": (float(lookup.distance[lookup.distance >= 0].mean())
                              if bool((lookup.distance >= 0).any()) else None),
            "cap": max_train_distance,
            "note": f"distances above {max_train_distance} are reported as >= and not resolved"},
        "wt_hamming": {"mean": float(wt_distance.mean()),
                       "distribution": {str(int(k)): int(v) for k, v in
                                        zip(*np.unique(wt_distance, return_counts=True))}},
        "site_entropy_nats": entropy_per_site,
        "sum_site_entropy_nats": float(sum(entropy_per_site)),
        "mean_pairwise_hamming": mean_pairwise_hamming(values),
        "entropy": empirical_entropy(sum_log_probability),
    }
    if reference_sum_log_probability is not None:
        document["kl_to_zero_shot"] = monte_carlo_kl(sum_log_probability,
                                                     reference_sum_log_probability)
    if parent_sum_log_probability is not None:
        document["kl_to_sft_parent"] = monte_carlo_kl(sum_log_probability,
                                                      parent_sum_log_probability)
    return document


# ---------------------------------------------------------------------------
# preregistered diversity eligibility
# ---------------------------------------------------------------------------

#: Fixed before any checkpoint was fitted. These are operational heuristics for
#: "this policy has not collapsed", not a guarantee of anything biological.
DIVERSITY_GATES = {"relative_floor": 0.90, "min_unique_fraction": 0.90,
                   "max_single_core_frequency": 0.01, "min_fraction_not_in_training": 0.50}


def diversity_eligibility(diagnostics, *, training_reference, parent_reference,
                          gates=DIVERSITY_GATES):
    """Apply the preregistered eligibility gates to one checkpoint's draws.

    ``training_reference`` is computed from the exact full training-high one-site
    counts; ``parent_reference`` from the policy's own initial-SFT parent, drawn
    under the same rule and draw count. A checkpoint must clear the relative floor
    against **both**, so improving on one reference cannot excuse collapsing
    relative to the other.

    Thresholds were fixed in advance and are not revisited after looking at the
    results. A checkpoint that fails is reported as failing, with the numbers.
    """
    floor = gates["relative_floor"]
    checks = {}
    for metric in ("mean_pairwise_hamming", "sum_site_entropy_nats"):
        observed = float(diagnostics[metric])
        for label, reference in (("training", training_reference), ("parent", parent_reference)):
            bound = floor * float(reference[metric])
            checks[f"{metric}_vs_{label}"] = {
                "observed": observed, "reference": float(reference[metric]),
                "threshold": bound, "passes": bool(observed >= bound)}
    checks["unique_fraction"] = {
        "observed": float(diagnostics["unique_fraction"]),
        "threshold": gates["min_unique_fraction"],
        "passes": bool(diagnostics["unique_fraction"] >= gates["min_unique_fraction"])}
    checks["max_single_core_frequency"] = {
        "observed": float(diagnostics["max_single_core_frequency"]),
        "threshold": gates["max_single_core_frequency"],
        "passes": bool(diagnostics["max_single_core_frequency"]
                       <= gates["max_single_core_frequency"])}
    checks["fraction_not_in_training_cores"] = {
        "observed": float(diagnostics["exact_train_novelty"]["fraction_not_in_training_cores"]),
        "threshold": gates["min_fraction_not_in_training"],
        "passes": bool(diagnostics["exact_train_novelty"]["fraction_not_in_training_cores"]
                       >= gates["min_fraction_not_in_training"])}
    failed = sorted(name for name, check in checks.items() if not check["passes"])
    return {"eligible": not failed, "failed_gates": failed, "checks": checks,
            "gates": dict(gates),
            "note": ("preregistered operational heuristics for distributional collapse; not a "
                     "guarantee of functional diversity")}


def select_within_budget(records, *, budgets, metric="val_average_precision"):
    """Best eligible checkpoint at or below each budget, earlier budget breaking ties.

    ``records`` is ``{checkpoint_name: {"budget_seconds": float, "eligible": bool,
    metric: float|None}}``, where ``budget_seconds`` is the **nominal** target the
    checkpoint was dropped at -- not the measured time, which overshoots it by at
    most one update. Comparing against the nominal target is what keeps a
    checkpoint from being excluded from its own budget by the overshoot that
    defines it; the measured value travels alongside under
    ``actual_gpu_seconds`` and is reported, never compared.

    The rule is fixed: among checkpoints at or below the budget whose diversity
    gates passed, take the highest ``metric``; break ties toward the cheaper budget
    and then the lexicographically smaller name, so the outcome does not depend on
    dict order.

    A budget at which nothing is eligible yields ``None`` with the reason, and the
    caller is expected to report that rather than quietly promoting the zero-budget
    parent into the winner's slot.
    """
    out = {}
    for budget in budgets:
        candidates = [(name, record) for name, record in records.items()
                      if record["budget_seconds"] <= budget + 1e-9
                      and record.get("eligible") and record.get(metric) is not None]
        if not candidates:
            considered = [name for name, record in records.items()
                          if record["budget_seconds"] <= budget + 1e-9]
            out[str(budget)] = {
                "selected": None, "metric": metric,
                "reason": ("no eligible checkpoint at or below this budget"
                           if considered else "no checkpoint reached this budget"),
                "checkpoints_considered": sorted(considered)}
            continue
        best = min(candidates, key=lambda item: (-item[1][metric], item[1]["budget_seconds"],
                                                 item[0]))
        out[str(budget)] = {"selected": best[0], "metric": metric,
                            "value": float(best[1][metric]),
                            "budget_seconds": float(best[1]["budget_seconds"]),
                            "budget_basis": "nominal target; actual GPU seconds reported beside it",
                            "actual_gpu_seconds": (None if best[1].get("actual_gpu_seconds") is None
                                                   else float(best[1]["actual_gpu_seconds"])),
                            "checkpoints_considered": sorted(name for name, _ in candidates)}
    return out


# ---------------------------------------------------------------------------
# campaign naming: one source of truth for what the freeze must contain
# ---------------------------------------------------------------------------

def checkpoint_name(method, seed, budget):
    """The name a raw budget checkpoint carries everywhere: artifact, freeze, table."""
    return f"{method}_seed{seed}_budget{int(budget)}"


def initial_policy_names(config):
    return sorted(f"policy_{arm}_seed{seed}" for arm in config["policy"]["arms"]
                  for seed in config["policy"]["seeds"])


def raw_budget_names(config):
    continuation = config["continuation"]
    return sorted(checkpoint_name(method, seed, budget)
                  for method in continuation["methods"]
                  for seed in continuation["seeds"]
                  for budget in continuation["budgets_gpu_seconds"][method])


def required_artifact_sets(config):
    """The exact names every stage must account for -- sets, not counts.

    A count check passes when one arm is missing and another is duplicated. Both
    the freeze that writes this block and the evaluation that reads it compare the
    same sorted name lists, built here from the config, so "24 raw checkpoints"
    cannot stand in for "these 24 raw checkpoints".
    """
    continuation = config["continuation"]
    return {"methods": list(continuation["methods"]),
            "seeds": list(continuation["seeds"]),
            "budgets_gpu_seconds": dict(continuation["budgets_gpu_seconds"]),
            # "initial" is the ROLE, which covers the fitted initial-SFT policies and
            # the pinned zero-shot model: both sit at budget 0 and both are scored.
            "initial_policies": sorted(initial_policy_names(config) + ["piggen_zeroshot"]),
            "raw_budget_checkpoints": raw_budget_names(config)}


def expected_selection_names(config):
    """Every artifact the FINAL freeze must name, policies and comparators alike."""
    names = set(initial_policy_names(config)) | set(raw_budget_names(config))
    names |= {f"cnn_3class_seed{seed}" for seed in config["classifier"]["seeds"]}
    names |= {"piggen_zeroshot", "linear_3class"}
    return sorted(names)


#: The closed key set for a per-draw record. ``classifier_proxy_p_high`` is
#: reserved but **not produced** by this pipeline: the CNN scores measured
#: populations only, and a proxy probability on an unassayed design is not a
#: measurement. The persisted draw files carry the four density/identity columns.
GENERATION_RECORD_FIELDS = ("draw_index", "core", "sum_log_probability", "mean_log_probability",
                            "classifier_proxy_p_high", "library_split", "library_class",
                            "min_train_hamming", "wt_hamming")
#: Name fragments that would mean somebody attached an affinity to an unassayed design.
FORBIDDEN_RECORD_TOKENS = ("kd", "affinity", "ic50", "ec50", "molar", "nm")

# The guard is on the FIELD LIST itself, checked at import. A later edit that adds
# an affinity-shaped column to the closed set fails on import rather than on the
# day somebody reads the artifact and believes the number.
for _field in GENERATION_RECORD_FIELDS:
    if set(_field.lower().split("_")) & set(FORBIDDEN_RECORD_TOKENS):
        raise ValueError(f"{_field!r} is an affinity-shaped field; generated sequences are "
                         "unassayed and carry no measurement")


def generation_record(**fields):
    """Build one per-draw record. A KD-valued field is not expressible here.

    The library class is present only for a draw that exactly matches a library
    row, and is ``None`` otherwise -- there is no path by which a generated,
    unassayed sequence acquires a measurement.
    """
    unknown = sorted(set(fields) - set(GENERATION_RECORD_FIELDS))
    require(not unknown, f"Unknown generation field(s) {unknown}; the key set is closed")
    missing = sorted(set(GENERATION_RECORD_FIELDS) - set(fields))
    require(not missing, f"Missing generation field(s) {missing}")
    if fields["library_split"] is None:
        require(fields["library_class"] is None,
                "library_class is only defined for a draw that matches a library row")
    return {name: fields[name] for name in GENERATION_RECORD_FIELDS}


# ---------------------------------------------------------------------------
# independent assay endpoint
# ---------------------------------------------------------------------------

def assay_endpoints(frame, score_column, *, seed, draws=2000, method_column="design_label"):
    """Quantitative and binary endpoints on one cohort, plus per-method strata.

    Quantitative: Spearman between the score and ``-log10(KD)`` over rows with a
    finite KD only. Binary: binding (quantitative or ``I.C.``) versus ``N.B.``.
    ``I.C.`` rows enter the binary endpoint as observed binders and are excluded
    from the correlation; they never receive a fabricated KD and are never
    counted as negatives.

    Within-method strata reduce the confound that each design method has its own
    generative bias. They do not remove selection bias: these designs were chosen
    by their authors, not drawn at random from sequence space.
    """
    quantitative = frame[frame.outcome_class == "quantitative"]
    document = {"cohort_rows": int(len(frame)),
                "outcome_class_counts": {k: int(v) for k, v in
                                         frame.outcome_class.value_counts().items()}}
    if len(quantitative) >= 3:
        strength = -np.log10(quantitative.kd_molar.to_numpy(dtype=np.float64))
        document["quantitative"] = bootstrap_spearman(
            quantitative[score_column].to_numpy(dtype=np.float64), strength,
            draws=draws, seed=seed)
        document["quantitative"]["endpoint"] = "spearman_score_vs_negative_log10_kd"
    else:
        document["quantitative"] = {"spearman": None, "n": int(len(quantitative)),
                                    "note": "fewer than three finite KD fits"}

    binary = frame[frame.outcome_class.isin(["quantitative", "binding_unquantified",
                                             "non_binding"])]
    positive = binary.outcome_class.isin(["quantitative", "binding_unquantified"]).to_numpy()
    if positive.any() and not positive.all():
        document["binary"] = rank_metrics(binary[score_column].to_numpy(dtype=np.float64),
                                          positive, binary.core.to_numpy(dtype=str),
                                          k_values=(32,))
    else:
        document["binary"] = {"n": int(len(binary)), "positives": int(positive.sum()),
                              "auroc": None, "auroc_note": AUROC_UNDEFINED}
    document["binary"]["definition"] = "binding = quantitative or I.C.; non-binding = N.B."

    document["by_method"] = {}
    for method, group in frame.groupby(method_column):
        rows = group[group.outcome_class == "quantitative"]
        document["by_method"][str(method)] = {
            "rows": int(len(group)), "quantitative_rows": int(len(rows)),
            "spearman": spearman(rows[score_column].to_numpy(dtype=np.float64),
                                 -np.log10(rows.kd_molar.to_numpy(dtype=np.float64)))
            if len(rows) >= 3 else None}
    document["selection_bias_note"] = (
        "Within-method comparison mitigates cross-method generative bias; it does NOT make this "
        "cohort selection-bias free. These designs were produced and chosen by the Buzz authors, "
        "and this endpoint tests our scorer on other people's designs -- it measures nothing "
        "about sequences this policy generated.")
    return document


def wt_distance_strata(index):
    return hamming_to(np.asarray(index), encode_cores([WT_CORE])[0]).astype(str)


def core_length_check(cores):
    require(all(len(core) == CORE_LENGTH for core in cores), "Every core must be a 10-mer")
    return True
