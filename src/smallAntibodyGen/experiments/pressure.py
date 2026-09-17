"""Continuous distribution and assay diagnostics for a fixed legal edit space."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

MEASUREMENT_COLUMNS = ("mean", "effective_sem", "sample_variance", "sample_sem", "replicate_count")
CATEGORY_NAMES = ("measured_train", "measured_development", "train_ineligible",
                  "development_ineligible", "eligible_development_withheld", "test")


def _checked(document):
    """Every reported float must be finite; None still means "not measured"."""
    for key, value in document.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Nonfinite reported statistic: {key}")
    return document


def probabilities(log_q, tolerance=2e-5):
    values = np.asarray(log_q, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values > 1e-6).any():
        raise ValueError("Finite nonpositive log probabilities required")
    log_mass = float(np.logaddexp.reduce(values))
    mass = float(np.exp(log_mass))
    if abs(mass - 1.) > tolerance:
        raise ValueError("Incomplete or unnormalized policy support")
    normalized_log = values - log_mass
    return np.exp(normalized_log), normalized_log, mass


def exact_distribution(log_q, reference_log_q, tolerance=2e-5):
    """Exhaustive-support movement. Atoms may underflow to exactly zero mass.

    Both log vectors stay finite, so every ``p * log`` term is a product of a
    finite log probability with a possibly-zero mass. ``kl_from_sft_nats`` is
    unbounded as the policy drains an atom the reference still holds; total
    variation and ``kl_to_sft_nats`` are the bounded movement statistics.
    """
    p, logp, mass = probabilities(log_q, tolerance)
    r, logr, reference_mass = probabilities(reference_log_q, tolerance)
    if p.shape != r.shape:
        raise ValueError("Policy/reference supports differ")
    entropy = float(-p @ logp)
    return _checked({"normalization_mass_before_correction": mass, "reference_normalization_mass": reference_mass,
            "entropy_nats": entropy, "entropy_effective_support": float(np.exp(entropy)),
            "collision_probability": float(p @ p), "collision_effective_support": float(1 / (p @ p)),
            "maximum_genotype_probability": float(p.max()), "kl_to_sft_nats": float(p @ (logp - logr)),
            "kl_from_sft_nats": float(r @ (logr - logp)), "total_variation_from_sft": float(np.abs(p - r).sum() / 2)})


def lexicographic_support(sites=16):
    """The identity order every exhaustive array in this study is stored in."""
    if not isinstance(sites, int) or isinstance(sites, bool) or not 1 <= sites <= 20:
        raise ValueError("Unsupported support width")
    return [f"{i:0{sites}b}" for i in range(2 ** sites)]


def site_marginals(p):
    """Exact per-site allele-1 probability over the lexicographic support.

    Site j is digit j of the genotype string, so it is bit ``sites-1-j`` of the
    support index. Marginals are exact sums, not sampled frequencies.
    """
    values = np.asarray(p, dtype=np.float64)
    sites = int(round(float(np.log2(len(values))))) if values.ndim == 1 and len(values) >= 2 else 0
    if sites < 1 or 2 ** sites != len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Exact distribution over a full binary support required")
    index = np.arange(len(values))
    return [float(values[((index >> (sites - 1 - j)) & 1).astype(bool)].sum()) for j in range(sites)]


def block_labels(genotypes, loci):
    """Split-defining block identifier: the loci digits, most significant first."""
    positions = list(loci)
    if (not positions or len(set(positions)) != len(positions)
            or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in positions)):
        raise ValueError("Invalid split-defining loci")
    labels = []
    for genotype in genotypes:
        if (not isinstance(genotype, str) or set(genotype) - {"0", "1"}
                or max(positions) >= len(genotype)):
            raise ValueError("Invalid genotype for block assignment")
        labels.append(int("".join(genotype[i] for i in positions), 2))
    return labels


def group_masses(p, labels):
    """Exact mass per label; labels align with the distribution's support order."""
    values = np.asarray(p, dtype=np.float64)
    keys = np.asarray(labels)
    if (values.ndim != 1 or keys.shape != values.shape or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError("Aligned finite nonnegative masses required")
    return {str(key): float(values[keys == key].sum()) for key in sorted(set(keys.tolist()))}


def category_table(identities, assignments, measured, eligible_development, weights=None):
    """Place every identity in exactly one measurement-availability category.

    Membership alone decides the category: split assignment, eligibility for
    measurement, and the evaluated-development whitelist. No withheld or
    reserved value is read and no missing measurement is imputed, so the
    omitted populations stay visible as counts or mass rather than disappearing.
    """
    ids = list(identities)
    if not ids or not assignments.index.is_unique or not set(ids) <= set(assignments.index):
        raise ValueError("Unique split assignments covering every identity required")
    seen = set(measured)
    known = assignments.reindex(sorted(seen))
    if known.isna().any() or (known == "test").any():
        raise ValueError("Reserved or unassigned identity carries a measurement")
    eligible = set(eligible_development)
    if not set(known[known == "development"].index) <= eligible:
        raise ValueError("Measured development identity outside the eligible population")
    splits = assignments.loc[ids].tolist()

    def category(genotype, split):
        if split == "test":
            return "test"
        if genotype in seen:
            return f"measured_{split}"
        if split == "train":
            return "train_ineligible"
        return "eligible_development_withheld" if genotype in eligible else "development_ineligible"

    labels = [category(genotype, split) for genotype, split in zip(ids, splits)]
    if weights is None:
        counts = Counter(labels)
        return dict({name: int(counts.get(name, 0)) for name in CATEGORY_NAMES}, total=len(ids))
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(ids),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid category weights")
    table = {name: 0. for name in CATEGORY_NAMES}
    for name, value in zip(labels, values):
        table[name] += float(value)
    return dict(table, total=float(values.sum()))


def evaluation_records(records, allowed_development):
    """Strip unused development labels before passing data to any evaluator."""
    if (not records.genotype.is_unique or not set(records.split) <= {"train", "development"}
            or not set(allowed_development) <= set(records.loc[records.split == "development", "genotype"])):
        raise ValueError("Invalid or test-contaminated measurement table")
    return records[(records.split == "train") | records.genotype.isin(allowed_development)].copy()


def label_free(frame, columns):
    """Artifacts saved before evaluation carry identities and model scores only."""
    if tuple(frame.columns) != tuple(columns) or set(frame.columns) & set(MEASUREMENT_COLUMNS):
        raise ValueError("Pre-evaluation artifact carries measurements or unexpected columns")
    return frame


def development_whitelist(previous, fresh):
    """Union of the development identities already evaluated; overlap is an error."""
    earlier, recent = list(previous), list(fresh)
    if len(set(earlier)) != len(earlier) or len(set(recent)) != len(recent) or not earlier or not recent:
        raise ValueError("Duplicate or empty development cohort")
    if set(earlier) & set(recent):
        raise ValueError("Development cohorts overlap")
    return sorted(set(earlier) | set(recent))


def replay_exposures(identities, seed, steps, batch_size):
    """Reconstruct the original SFT draws over a table's OWN row order.

    The pilot indexed ``training_positives.csv`` as written, which is landscape
    row order. ``affinity_population`` returns the same identities sorted by
    genotype, so passing that order reconstructs a plausible but wrong exposure
    set. The returned bit-generator state is the decisive provenance check
    against the state stored in the pilot checkpoint.
    """
    ordered = list(identities)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("Unique ordered identities required")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (steps, batch_size)):
        raise ValueError("Positive integer replay dimensions required")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(ordered), size=(steps, batch_size))
    return [[ordered[int(i)] for i in row] for row in draws], generator.bit_generator.state


def monte_carlo_kl(log_q, reference_log_q):
    """Sampling estimate of KL(q||reference) from on-policy draws.

    Per-draw log ratios are exact, but this remains an estimate with sampling
    error at every checkpoint, including the exhaustively scored endpoints.
    """
    ratio = np.asarray(log_q, dtype=np.float64) - np.asarray(reference_log_q, dtype=np.float64)
    if ratio.ndim != 1 or len(ratio) < 2 or not np.isfinite(ratio).all():
        raise ValueError("Finite paired log probabilities required")
    return _checked({"nats": float(ratio.mean()), "draws": len(ratio),
                     "standard_error": float(ratio.std(ddof=1) / np.sqrt(len(ratio)))})


def exposure_overlap(genotypes, exposures):
    """Draws and unique identities that already carried a training label."""
    ids = list(genotypes)
    seen = set(exposures)
    if not ids:
        raise ValueError("No identities to compare")
    return {"draw_count": sum(g in seen for g in ids), "unique_count": len(set(ids) & seen),
            "exposure_unique_count": len(seen)}


def weighted_affinity(records, weights, threshold):
    """Conditional affinity, coverage and independent-genotype assay-SEM proxy.

    Weights are unconditional probability masses or sampling frequencies keyed
    by identity; they are never renormalized before the join, so measured_mass
    is the true mass the measured identities carry. Repeated observations share
    the same assay uncertainty, and the effective genotype count shows how few
    identities a concentrated policy leaves the conditional mean resting on.
    """
    if (not records.genotype.is_unique or not set(records.split) <= {"train", "development"}
            or not weights.index.is_unique or not np.isfinite(weights.to_numpy()).all()
            or (weights < 0).any() or not np.isfinite(threshold)):
        raise ValueError("Invalid measurement identities or weights")
    table = records.set_index("genotype")
    if not np.isfinite(table[["mean", "effective_sem"]].to_numpy()).all() or (table.effective_sem < 0).any():
        raise ValueError("Invalid assay measurements")
    joined = table.join(weights.rename("weight"), how="inner")
    joined = joined[joined.weight > 0]
    mass = float(joined.weight.sum())
    if mass == 0:
        return {"measured_mass": 0., "conditional_mean_affinity": None, "assay_sem_proxy": None,
                "positive_mass": 0., "conditional_positive_fraction": None, "distinct_measured_genotypes": 0,
                "conditional_effective_genotypes": None}
    conditional = joined.weight.to_numpy() / mass
    positive = joined["mean"].to_numpy() >= threshold
    return _checked({"measured_mass": mass, "conditional_mean_affinity": float(conditional @ joined["mean"].to_numpy()),
            "assay_sem_proxy": float(np.linalg.norm(conditional * joined.effective_sem.to_numpy())),
            "positive_mass": float(joined.weight.to_numpy() @ positive),
            "conditional_positive_fraction": float(conditional @ positive),
            "distinct_measured_genotypes": len(joined),
            "conditional_effective_genotypes": float(1 / np.square(conditional).sum())})


def affinity_difference(records, first_weights, reference_weights):
    """Shared genotype assay errors cancel in a difference of conditional means."""
    if (not records.genotype.is_unique or not set(records.split) <= {"train", "development"}
            or not first_weights.index.is_unique or not reference_weights.index.is_unique):
        raise ValueError("Invalid identities or held-out measurements")
    table = records.set_index("genotype")
    if not np.isfinite(table[["mean", "effective_sem"]].to_numpy()).all() or (table.effective_sem < 0).any():
        raise ValueError("Invalid assay measurements")
    first = first_weights.reindex(table.index, fill_value=0).to_numpy(float)
    reference = reference_weights.reindex(table.index, fill_value=0).to_numpy(float)
    if (not np.isfinite(first).all() or not np.isfinite(reference).all() or (first < 0).any()
            or (reference < 0).any()):
        raise ValueError("Invalid probability weights")
    if first.sum() == 0 or reference.sum() == 0:
        return {"conditional_affinity_delta": None, "assay_sem_proxy_for_delta": None}
    coefficients = first / first.sum() - reference / reference.sum()
    return _checked({"conditional_affinity_delta": float(coefficients @ table["mean"].to_numpy()),
            "assay_sem_proxy_for_delta": float(np.linalg.norm(coefficients * table.effective_sem.to_numpy()))})


def sampled_affinity(genotypes, assignments, records, threshold, seen_training):
    """Per-split conditional affinity over the actual draws, duplicates retained.

    The whole measurement table is validated before any split filtering. A
    reserved, duplicated, unassigned or misassigned measurement identity is an
    error rather than a row that silently vanishes from the per-split summaries
    while still counting as covered in ``unscored_draw_count``: that is how
    coverage comes to disagree with the split assignment.

    The *draws* are a different matter. A draw may legitimately land on an
    identity that carries no usable measurement, including a reserved one; those
    stay visible in ``split_counts`` and ``unscored_draw_count`` instead of
    being dropped.
    """
    ids = list(genotypes)
    if not ids or not assignments.index.is_unique or not set(ids) <= set(assignments.index):
        raise ValueError("Unassigned sampled identities")
    if (not records.genotype.is_unique or not set(records.split) <= {"train", "development"}
            or not set(records.genotype) <= set(assignments.index)
            or assignments.loc[records.genotype.tolist()].tolist() != records.split.tolist()):
        raise ValueError("Duplicate, unassigned or misassigned measurement identity")
    counts = pd.Series(Counter(ids), dtype=float)
    weights = counts / len(ids)
    split_counts = assignments.loc[ids].value_counts().to_dict()
    summaries = {}
    lookup = records.set_index("genotype")
    for split in ("train", "development"):
        subset = records[records.split == split]
        available = set(subset.genotype)
        measured_draws = [g for g in ids if g in available]
        measured = weighted_affinity(subset, weights, threshold)
        measured["measured_draw_count"] = len(measured_draws)
        values = lookup.loc[measured_draws, "mean"].to_numpy() if measured_draws else np.array([])
        measured["sampling_sem_conditional_mean"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else None
        summaries[split] = measured
    overlap = exposure_overlap(ids, seen_training)
    return {"sample_count": len(ids), "unique_draw_count": len(counts),
            "split_counts": {str(k): int(v) for k, v in split_counts.items()},
            "training_exposure_overlap_draw_count": overlap["draw_count"],
            "training_exposure_overlap_unique_count": overlap["unique_count"],
            "unscored_draw_count": sum(g not in lookup.index for g in ids), "by_split": summaries}
