"""Label-blind portfolio selection and descriptive sequence diversity."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_shortlist(candidates, k, admission_multiplier=1):
    """Top score first, then greedy max-min Hamming within the top m*k scores.

    Exactly two input columns are accepted: genotype and score. Ties use higher
    score then lexicographic genotype. m=1 returns the ordinary ranked top-k set.
    Score admission is a quality proxy, not an affinity guarantee.
    """
    if set(candidates.columns) != {"genotype", "score"}:
        raise ValueError("Selector accepts identities and scores only, never labels")
    if (not isinstance(k, int) or isinstance(k, bool) or k < 2 or k > len(candidates)
            or not isinstance(admission_multiplier, int) or isinstance(admission_multiplier, bool)
            or admission_multiplier < 1):
        raise ValueError("Invalid shortlist size or admission multiplier")
    ids = candidates.genotype
    if (not ids.is_unique or ids.isna().any()
            or not ids.map(lambda g: isinstance(g, str) and len(g) == 16 and set(g) <= {"0", "1"}).all()
            or not np.isfinite(candidates.score.to_numpy(dtype=float)).all()):
        raise ValueError("Invalid identities or scores")
    admitted = candidates.sort_values(["score", "genotype"], ascending=[False, True]).head(k * admission_multiplier)
    if admission_multiplier == 1:
        return admitted.genotype.tolist()
    bits = np.array([list(g) for g in admitted.genotype])
    chosen = [0]
    distances = np.count_nonzero(bits != bits[0], axis=1)
    distances[0] = -1
    while len(chosen) < k:
        index = int(np.argmax(distances))
        chosen.append(index)
        distances = np.minimum(distances, np.count_nonzero(bits != bits[index], axis=1))
        distances[chosen] = -1
    return admitted.iloc[chosen].genotype.tolist()


def sequence_diversity(genotypes):
    """Unordered-pair distances and connected components at Hamming <= 1.

    Components are descriptive single-linkage clusters, not biological modes.
    Empirical identity entropy on a unique shortlist is always log(k); report
    uniqueness explicitly instead of presenting that constant as model entropy.
    """
    ids = list(genotypes)
    if (len(ids) < 2 or any(not isinstance(g, str) or len(g) != 16 or set(g) - {"0", "1"} for g in ids)):
        raise ValueError("At least two valid binary genotypes required")
    bits = np.array([list(g) for g in ids])
    distances = np.count_nonzero(bits[:, None, :] != bits[None, :, :], axis=2)
    pairs = distances[np.triu_indices(len(ids), 1)]
    remaining = set(range(len(ids)))
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = {i for i in remaining if distances[current, i] <= 1}
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    return {"unique": len(set(ids)), "mean_hamming": float(pairs.mean()),
            "minimum_hamming": int(pairs.min()), "hamming_le_one_pair_fraction": float((pairs <= 1).mean()),
            "identity_collision_pair_fraction": float((pairs == 0).mean()),
            "hamming_le_one_components": len(sizes), "largest_component": max(sizes)}


def portfolio_metrics(records, genotypes, quality_threshold):
    """Evaluation only: join measurements after identities have been selected."""
    if (not records.genotype.is_unique or len(set(genotypes)) != len(genotypes)
            or not set(genotypes) <= set(records.genotype)
            or not set(records.split) <= {"train", "development"}):
        raise ValueError("Invalid or test-contaminated evaluation")
    selected = records.set_index("genotype").loc[genotypes]
    if not np.isfinite(selected[["mean", "effective_sem"]].to_numpy()).all():
        raise ValueError("Nonfinite measurements")
    qualified = selected[selected["mean"] >= quality_threshold]
    return {"selected_genotypes": list(genotypes), "mean_affinity": float(selected["mean"].mean()),
            "mean_minus_sem": float((selected["mean"] - selected.effective_sem).mean()),
            "quality_threshold": float(quality_threshold), "quality_qualified_count": len(qualified),
            "quality_qualified_diversity": sequence_diversity(qualified.index) if len(qualified) >= 2 else None,
            "block_counts": {str(int(b)): int(n) for b, n in selected.block.value_counts().sort_index().items()},
            "diversity": sequence_diversity(genotypes)}


def calibrate_admission(training, scores, config):
    """Choose one multiplier using training-only, identity-defined cohorts.

    Acceptable options retain mean affinity within the declared tolerance for
    both budgets in every calibration cohort. Prefer the highest mean Hamming
    gain, breaking ties toward the smaller admission window. m=1 is a fallback.
    """
    if set(training.split) != {"train"} or not training.genotype.is_unique:
        raise ValueError("Calibration requires unique training records only")
    if not scores.index.is_unique or set(scores.index) != set(training.genotype):
        raise ValueError("Calibration score identities differ")
    threshold = config["quality_threshold"]
    options = []
    for multiplier in config["admission_multipliers"]:
        cells = []
        for cohort, group in training.groupby("calibration_cohort", sort=True):
            candidates = pd.DataFrame({"genotype": group.genotype, "score": group.genotype.map(scores)})
            for k in config["budgets"]:
                baseline = portfolio_metrics(group, select_shortlist(candidates, k), threshold)
                selected = portfolio_metrics(group, select_shortlist(candidates, k, multiplier), threshold)
                cells.append({"cohort": int(cohort), "k": k,
                    "affinity_delta": selected["mean_affinity"] - baseline["mean_affinity"],
                    "hamming_delta": selected["diversity"]["mean_hamming"] - baseline["diversity"]["mean_hamming"]})
        options.append({"multiplier": multiplier, "cells": cells,
            "passes_quality": all(c["affinity_delta"] >= -config["affinity_loss_tolerance"] for c in cells),
            "mean_hamming_gain": float(np.mean([c["hamming_delta"] for c in cells]))})
    acceptable = [o for o in options if o["passes_quality"]]
    if not acceptable:
        raise ValueError("Calibration must include the ordinary top-k fallback")
    winner = max(acceptable, key=lambda o: (o["mean_hamming_gain"], -o["multiplier"]))
    return {"selected_multiplier": winner["multiplier"], "options": options,
            "affinity_loss_tolerance": config["affinity_loss_tolerance"], "quality_threshold": threshold}
