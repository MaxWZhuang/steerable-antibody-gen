#!/usr/bin/env python
"""Recompute affinity-pilot evidence without importing its trainer or metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def shortlist_hamming(genotypes):
    """Post-hoc shortlist diagnostic; not part of the predeclared decision gate."""
    bits = np.array([list(map(int, g)) for g in genotypes])
    ones = bits.sum(axis=0)
    n = len(bits)
    return float(np.sum(2 * ones * (n - ones)) / (n * (n - 1)))


def run(directory):
    destination = directory / "independent_audit.json"
    require(not destination.exists(), "Audit evidence already exists")
    results = json.loads((directory / "results.json").read_text())
    require(results["status"] == "completed" and not results["reserved_test_labels_evaluated"], "Incomplete or test-contaminated run")
    config = results["config"]
    for relative, digest in results["output_sha256"].items():
        require(sha256(directory / relative) == digest, f"Artifact changed: {relative}")
    records_path = ROOT / "outputs/cr9114_preferences_verified_20260916/eligible_non_test_records.csv"
    require(sha256(records_path) == results["input_records_sha256"], "Measurement input changed")
    records = pd.read_csv(records_path, dtype={"genotype": "string"})
    require(set(records.split) == {"train", "development"}, "Test measurements in input")
    population = pd.read_csv(directory / "training_population.csv", dtype={"genotype": "string"})
    fresh = pd.read_csv(directory / "fresh_development.csv", dtype={"genotype": "string"})
    pairs = pd.read_csv(directory / "development_pairs.csv", dtype={"chosen_genotype": "string", "rejected_genotype": "string"})
    assignments = pd.read_csv(ROOT / "outputs/cr9114_5cjq_pilot_20260916/split.csv", dtype={"genotype": "string"}).set_index("genotype").split
    require(set(assignments.loc[population.genotype]) == set(population.split) == {"train"}, "Training identity crosses split")
    require(set(assignments.loc[fresh.genotype]) == set(fresh.split) == {"development"}, "Evaluation identity crosses split")
    require(population.genotype.is_unique and fresh.genotype.is_unique and not set(population.genotype) & set(fresh.genotype), "Identity overlap")
    old = set()
    for relative in ("outputs/cr9114_5cjq_pilot_20260916/development_scores.csv",
                     "outputs/cr9114_dpo_diagnostics_20260916/fresh_development.csv",
                     "outputs/cr9114_diversity_pilot_20260916_v3/fresh_development.csv"):
        old.update(pd.read_csv(ROOT / relative, dtype={"genotype": "string"}).genotype)
    require(len(old) == 4608 and not old & set(fresh.genotype), "Previous evaluation identity overlap")
    candidates = records[(records.split == "development") & ~records.genotype.isin(old)]
    expected_ids = sorted(candidates.genotype, key=lambda g: hashlib.sha256(
        f"{config['fresh_development_seed']}:{g}".encode()).hexdigest())[:config["fresh_development_count"]]
    require(fresh.genotype.tolist() == expected_ids, "Cohort was not selected by declared identity hash")
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), records.set_index("genotype").loc[expected_ids].reset_index(),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    train = records[records.split == "train"]
    threshold = train["mean"].quantile(config["positive_quantile"])
    expected_pool = train[train["mean"] >= threshold].sort_values("genotype")
    require(population.genotype.tolist() == expected_pool.genotype.tolist(), "Positive population changed")
    for column in ("mean", "sample_variance", "replicate_count"):
        check_close(population[column], expected_pool[column])
    variance = float(((train.replicate_count - 1) * train.sample_variance).sum() / (train.replicate_count - 1).sum())
    sem = np.sqrt(np.maximum(expected_pool.sample_variance.to_numpy(), variance) / expected_pool.replicate_count.to_numpy())
    utility = expected_pool["mean"].to_numpy() - config["uncertainty_multiplier"] * sem
    weight = np.maximum(np.exp((utility - utility.max()) / np.sqrt(variance)), 1 / config["max_weight_ratio"])
    probability = (1 - config["uniform_fraction"]) * weight / weight.sum() + config["uniform_fraction"] / len(weight)
    check_close(population.affinity_effective_sem, sem)
    check_close(population.utility, utility)
    check_close(population.affinity_probability, probability)
    check_close(population.uniform_probability, np.full(len(population), 1 / len(population)))
    check_close(results["population"]["pooled_sample_variance"], variance)
    check_close(results["population"]["mean_threshold"], threshold)
    check_close(results["population"]["effective_sampling_population"], 1 / np.sum(probability ** 2))
    check_close(results["population"]["probability_ratio"], probability.max() / probability.min())
    check_close(results["population"]["uniform_expected_affinity"], expected_pool["mean"].mean())
    check_close(results["population"]["weighted_expected_affinity"], probability @ expected_pool["mean"].to_numpy())
    schedule_summary = {}
    for seed in config["seeds"]:
        for weighted in (False, True):
            key = f"{seed}_{'weighted' if weighted else 'uniform'}"
            p = probability if weighted else np.full(len(population), 1 / len(population))
            expected = np.random.default_rng(seed).choice(len(population), p=p / p.sum(),
                size=(config["steps"], config["batch_size"]))
            actual = np.load(directory / f"schedule_{key}.npy", allow_pickle=False)
            np.testing.assert_array_equal(actual, expected)
            exposed = population.iloc[actual.ravel()]
            schedule_summary[key] = {"labelled_draws": len(exposed), "unique_genotypes": exposed.genotype.nunique(),
                                     "mean_measured_affinity": float(exposed["mean"].mean())}
    require(set(pairs.split) == {"development"}, "Wrong pair split")
    fresh_by_id = fresh.set_index("genotype")
    require(set(pairs.chosen_genotype) | set(pairs.rejected_genotype) <= set(fresh.genotype), "Pair outside cohort")
    for column in ("chosen_genotype", "rejected_genotype"):
        np.testing.assert_array_equal(fresh_by_id.loc[pairs[column], "block"], pairs.block)
    checks = {}
    metrics = {"sft": results["sft"], **results["arms"]}
    for name, result in metrics.items():
        folder = directory / name
        scores = pd.read_csv(folder / "development_scores.csv", dtype={"genotype": "string"}).set_index("genotype").log_q
        require(scores.index.is_unique and set(scores.index) == set(fresh.genotype), "Scored cohort changed")
        require(np.isfinite(scores).all() and (scores <= 1e-6).all(), "Invalid policy scores")
        ordered = fresh.assign(score=scores.loc[fresh.genotype].to_numpy()).sort_values(["score", "genotype"], ascending=[False, True])
        top = {str(k): float(ordered.head(k)["mean"].mean()) for k in (16, 32)}
        shortlist_diversity = {str(k): shortlist_hamming(ordered.head(k).genotype) for k in (16, 32)}
        for k, mean in top.items():
            check_close(mean, result["development"]["top_k_mean"][k])
            require(ordered.head(int(k)).genotype.tolist() == result["selection_details"][k]["selected_genotypes"], "Selection mismatch")
            check_close((ordered.head(int(k))["mean"] - ordered.head(int(k)).effective_sem).mean(),
                        result["selection_details"][k]["mean_minus_effective_sem"])
            baseline_ids = set(results["sft"]["selection_details"][k]["selected_genotypes"])
            require(len(set(ordered.head(int(k)).genotype) & baseline_ids) == result["selection_details"][k]["overlap_with_sft"], "Selection overlap mismatch")
            for block, group in ordered.groupby("block"):
                check_close(group.head(int(k))["mean"].mean(), result["selection_details"][k]["per_block_top_k_mean"][str(int(block))])
                check_close(ordered[ordered.block != block].head(int(k))["mean"].mean(),
                            result["selection_details"][k]["leave_one_block_out_top_k_mean"][str(int(block))])
        margin = scores.loc[pairs.chosen_genotype].to_numpy() - scores.loc[pairs.rejected_genotype].to_numpy()
        credit = (margin > 1e-6).astype(float) + .5 * (np.abs(margin) <= 1e-6)
        check_close(np.average(credit, weights=pairs.pair_weight), result["development"]["pair_accuracy"])
        samples = pd.read_csv(folder / "samples.csv", dtype={"genotype": "string"})
        n = len(samples)
        require(n == config["evaluation_samples"], "Sample count changed")
        require(np.isfinite(samples.log_q).all() and (samples.log_q <= 1e-6).all(), "Invalid sample probabilities")
        counts = np.array(list(Counter(samples.genotype).values()))
        one_counts = np.array([list(map(int, g)) for g in samples.genotype]).sum(axis=0)
        hamming = float(np.sum(2 * one_counts * (n - one_counts)) / (n * (n - 1)))
        require(len(counts) == result["diversity"]["unique_genotypes"], "Unique count mismatch")
        check_close(hamming, result["diversity"]["mean_pairwise_hamming_unbiased"])
        check_close(-samples.log_q.mean(), result["diversity"]["entropy_nats_mc"])
        check_close(np.sum(counts * (counts - 1)) / (n * (n - 1)), result["diversity"]["collision_probability_unbiased"])
        check_close(one_counts / n, result["diversity"]["allele_one_frequencies"])
        if name == "sft":
            checks[name] = {"top_k": top, "unique": len(counts), "hamming": hamming,
                            "posthoc_top_k_mean_hamming": shortlist_diversity}
            continue
        require(sha256(folder / f"decoder_step_{config['steps']:04d}.pt") == result["checkpoint_sha256"], "Checkpoint changed")
        check_close((samples.log_q - samples.reference_log_q).mean(), result["kl_to_sft_mc"]["nats"])
        history = json.loads((folder / "history.json").read_text())
        require([h["step"] for h in history] == list(range(1, config["steps"] + 1)), "Training history incomplete")
        require(all(np.isfinite([h["nll_per_site"], h["gradient_norm"]]).all() and h["gradient_norm"] > 0 for h in history), "Nonfinite training")
        coefficient = result["arm"]["entropy_coefficient"]
        expected_steps = list(range(config["entropy_every"], config["steps"] + 1, config["entropy_every"])) if coefficient else []
        require([h["step"] for h in history if "entropy_mc" in h] == expected_steps, "Entropy update mismatch")
        require(all(h.get("entropy_rescore_max_error", 0) < 2e-4 for h in history), "Entropy parity failure")
        if coefficient:
            regularization = pd.read_csv(folder / "entropy_training_samples.csv", dtype={"genotype": "string"})
            require(len(regularization) == result["regularization_sample_count"] == len(expected_steps) * config["entropy_samples"], "Entropy sample count mismatch")
            require(set(regularization.columns) == {"step", "genotype", "log_q"}, "Labels in entropy samples")
            require(regularization.groupby("step").size().to_dict() == {s: config["entropy_samples"] for s in expected_steps}, "Entropy batch count mismatch")
            require(len(set(regularization.genotype) & set(fresh.genotype)) == result["unlabelled_regularization_unique_overlap_with_development"], "Unlabelled overlap mismatch")
        else:
            require(result["regularization_sample_count"] == 0, "Unexpected entropy samples")
        baseline = checks["sft"]
        gate = {f"top_{k}_at_least_sft": top[k] >= baseline["top_k"][k] for k in ("16", "32")}
        gate.update(unique_genotypes_retained=len(counts) >= config["diversity_retention_fraction"] * baseline["unique"],
                    mean_pairwise_hamming_unbiased_retained=hamming >= config["diversity_retention_fraction"] * baseline["hamming"])
        require(gate == result["screen"]["checks"] and all(gate.values()) == result["screen"]["passes"], "SFT screen mismatch")
        if result["arm"]["weighted"]:
            control = checks[f"seed_{result['seed']}_continued_sft_entropy"]["top_k"]
            comparison = {f"top_{k}_at_least_control": top[k] >= control[k] for k in ("16", "32")}
            comparison["at_least_one_strict_improvement"] = any(top[k] > control[k] for k in ("16", "32"))
            require(comparison == result["control_comparison"]["checks"] and all(comparison.values()) == result["control_comparison"]["passes"], "Control comparison mismatch")
        checks[name] = {"top_k": top, "unique": len(counts), "hamming": hamming, "screen_passes": all(gate.values()),
                        "posthoc_top_k_mean_hamming": shortlist_diversity}
    joint = all(results["arms"][f"seed_{seed}_affinity_entropy"]["screen"]["passes"] and
                results["arms"][f"seed_{seed}_affinity_entropy"]["control_comparison"]["passes"] for seed in config["seeds"])
    require(joint == results["affinity_entropy_two_seed_screen"], "Joint decision mismatch")
    audit = {"status": "passed", "audit_script_sha256": sha256(Path(__file__)),
             "results_sha256": sha256(directory / "results.json"), "verified_artifact_hashes": len(results["output_sha256"]),
             "verified_checkpoint_hashes": len(results["arms"]), "train_only_weights_recomputed": True,
             "schedules_reproduced": True, "schedules": schedule_summary, "fresh_cohort_exclusion_verified": True,
             "reserved_test_labels_evaluated": False, "affinity_entropy_two_seed_screen": joint, "models": checks}
    save_json(destination, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)
