"""Train-only, uncertainty-adjusted sampling for bounded affinity-weighted SFT."""
from __future__ import annotations

import numpy as np


def affinity_population(records, *, quantile=.75, uncertainty_multiplier=1.,
                        max_weight_ratio=20., uniform_fraction=.25):
    """Return the original SFT-positive pool and two sampling distributions.

    Temperature is the pooled training replicate SD. The uncertainty adjustment
    is a heuristic, not a calibrated lower confidence bound. Weighted sampling
    implements weighted likelihood: use an UNWEIGHTED minibatch NLL afterward.
    No development measurements may be passed, even if they would be filtered.
    """
    required = {"genotype", "split", "mean", "sample_variance", "replicate_count"}
    if not required <= set(records.columns) or len(records) < 2:
        raise ValueError("Missing training measurements")
    if set(records.split) != {"train"} or not records.genotype.is_unique:
        raise ValueError("Expected unique training records only")
    if not all(isinstance(g, str) and len(g) == 16 and set(g) <= {"0", "1"}
               for g in records.genotype):
        raise ValueError("Genotypes must retain all 16 binary digits")
    parameters = [quantile, uncertainty_multiplier, max_weight_ratio, uniform_fraction]
    if (not np.isfinite(parameters).all() or not 0 <= quantile < 1
            or uncertainty_multiplier < 0 or max_weight_ratio < 1
            or not 0 <= uniform_fraction <= 1):
        raise ValueError("Invalid sampling parameters")
    values = records[["mean", "sample_variance", "replicate_count"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, 1] < 0).any()
            or (values[:, 2] < 2).any() or (values[:, 2] != np.floor(values[:, 2])).any()):
        raise ValueError("Invalid training replicate measurements")
    variance = float(np.average(values[:, 1], weights=values[:, 2] - 1))
    if variance <= 0:
        raise ValueError("Positive pooled replicate variance required")
    threshold = float(records["mean"].quantile(quantile))
    pool = records[records["mean"] >= threshold].sort_values("genotype").copy().reset_index(drop=True)
    pool["affinity_effective_sem"] = np.sqrt(
        np.maximum(pool.sample_variance, variance) / pool.replicate_count)
    pool["utility"] = pool["mean"] - uncertainty_multiplier * pool.affinity_effective_sem
    logits = (pool.utility.to_numpy() - pool.utility.max()) / np.sqrt(variance)
    weights = np.exp(np.maximum(logits, -np.log(max_weight_ratio)))
    pool["uniform_probability"] = 1. / len(pool)
    pool["affinity_probability"] = ((1 - uniform_fraction) * weights / weights.sum()
                                     + uniform_fraction / len(pool))
    p = pool.affinity_probability.to_numpy()
    audit = {"training_records": len(records), "positive_records": len(pool),
             "mean_threshold": threshold, "pooled_sample_variance": variance,
             "temperature": float(np.sqrt(variance)),
             "effective_sampling_population": float(1 / np.sum(p ** 2)),
             "probability_ratio": float(p.max() / p.min()),
             "uniform_expected_affinity": float(pool["mean"].mean()),
             "weighted_expected_affinity": float(p @ pool["mean"].to_numpy()),
             "block_probability": {str(int(b)): float(g.affinity_probability.sum())
                                   for b, g in pool.groupby("block")} if "block" in pool else {}}
    return pool, audit


def likelihood_schedule(population, *, weighted, steps, batch_size, seed):
    """Common-random-number inverse-CDF draws; never apply weights twice."""
    if set(population.split) != {"train"}:
        raise ValueError("Training schedule cannot include held-out records")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
               for v in (steps, batch_size)):
        raise ValueError("Positive integer schedule dimensions required")
    column = "affinity_probability" if weighted else "uniform_probability"
    p = population[column].to_numpy(float)
    if (not len(p) or not np.isfinite(p).all() or (p <= 0).any()
            or not np.isclose(p.sum(), 1.)):
        raise ValueError("Invalid sampling probabilities")
    cdf = np.cumsum(p / p.sum())
    cdf[-1] = 1.
    draws = np.random.default_rng(seed).random((steps, batch_size))
    return np.searchsorted(cdf, draws, side="right")
