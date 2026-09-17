#!/usr/bin/env python
"""Independently recompute matched-control metrics, schedules and checkpoint identity."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_cr9114_shortlist import independent_distances, independent_selection
from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cosine(features):
    z = np.asarray(features, dtype=np.float64)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1., rtol=1e-5, atol=1e-5)
    # Independently sum explicit rows against all other rows (bounded batches).
    n = len(z)
    total = z.sum(0)
    return float(sum(np.dot(row, total - row) for row in z) / (n * (n - 1)))


def split_distance_decomposition(ids, loci=(2, 5, 8, 10)):
    """Post-hoc descriptive check: diversity on split-defining versus other sites."""
    pairs = list(itertools.combinations(ids, 2))
    split = [sum(a[i] != b[i] for i in loci) for a, b in pairs]
    other = [sum(a[i] != b[i] for i in range(16) if i not in loci) for a, b in pairs]
    within = [d for s, d in zip(split, other) if s == 0]
    return {"split_defining_loci_0based": list(loci), "split_locus_mean_hamming": float(np.mean(split)),
            "other_locus_mean_hamming": float(np.mean(other)), "within_block_pair_count": len(within),
            "within_block_mean_hamming": float(np.mean(within)) if within else None}


def audit(directory):
    result = json.loads((directory / "results.json").read_text())
    config = result["config"]
    require(result["status"] == "completed" and len(result["arms"]) == 12, "Incomplete comparison")
    require(not result["reserved_test_labels_evaluated"] and result["development_is_reused"], "Invalid evaluation scope")
    for relative, digest in result["output_sha256"].items():
        require(sha256(directory / relative) == digest, f"Artifact changed: {relative}")
    source = ROOT / "outputs/cr9114_preferences_verified_20260916/eligible_non_test_records.csv"
    require(sha256(source) == result["input_records_sha256"], "Source records changed")
    records = pd.read_csv(source, dtype={"genotype": "string"})
    require(set(records.split) == {"train", "development"}, "Test measurements present")
    train = records[records.split == "train"]
    threshold = float(train["mean"].quantile(config["positive_quantile"]))
    expected_pool = train[train["mean"] >= threshold].sort_values("genotype")
    population = pd.read_csv(directory / "training_population.csv", dtype={"genotype": "string"})
    require(population.genotype.tolist() == expected_pool.genotype.tolist() and set(population.split) == {"train"}, "Population changed")
    variance = float(np.average(train.sample_variance, weights=train.replicate_count - 1))
    sem = np.sqrt(np.maximum(expected_pool.sample_variance, variance) / expected_pool.replicate_count)
    utility = expected_pool["mean"] - config["uncertainty_multiplier"] * sem
    weights = np.maximum(np.exp((utility - utility.max()) / np.sqrt(variance)), 1 / config["max_weight_ratio"])
    p = ((1 - config["uniform_fraction"]) * weights / weights.sum() + config["uniform_fraction"] / len(weights)).to_numpy()
    close(population.affinity_probability, p)
    for seed in config["seeds"]:
        expected = np.random.default_rng(seed).choice(len(p), size=(config["steps"], config["batch_size"]), p=p / p.sum())
        np.testing.assert_array_equal(np.load(directory / f"schedule_{seed}.npy", allow_pickle=False), expected)
    shortlist_dir = ROOT / config["shortlist_directory"]
    require(sha256(shortlist_dir / "results.json") == result["shortlist_result_sha256"], "Selector input changed")
    shortlist = json.loads((shortlist_dir / "results.json").read_text())
    fresh = pd.read_csv(directory / "development_records.csv", dtype={"genotype": "string"})
    original = pd.read_csv(shortlist_dir / "fresh_development.csv", dtype={"genotype": "string"})
    pd.testing.assert_frame_equal(fresh, original)
    require(set(fresh.split) == {"development"} and not set(fresh.genotype) & set(population.genotype), "Split leakage")
    require([int("".join(g[i] for i in (2, 5, 8, 10)), 2) for g in fresh.genotype] == fresh.block.tolist(), "Split-defining loci changed")
    frozen = np.load(directory / "frozen_development_embeddings.npy", allow_pickle=False)
    ids_to_position = {g: i for i, g in enumerate(fresh.genotype)}
    reports = {}
    decomposition = {}
    for name, measured in [("sft", result["sft"]), *result["arms"].items()]:
        folder = directory / name
        scores = pd.read_csv(folder / "development_scores.csv", dtype={"genotype": "string"})
        require(scores.genotype.tolist() == fresh.genotype.tolist(), "Score order changed")
        features = np.load(folder / "development_embeddings.npy", allow_pickle=False)
        decomposition[name] = {}
        for mode, multiplier in (("ordinary", 1), ("diverse", shortlist["calibration"]["selected_multiplier"])):
            decomposition[name][mode] = {}
            for k in (16, 32):
                ids = independent_selection(scores, k, multiplier)
                cell = measured["portfolios"][mode][str(k)]
                require(ids == cell["selected_genotypes"], "Selector changed")
                decomposition[name][mode][str(k)] = split_distance_decomposition(ids)
                components = decomposition[name][mode][str(k)]
                close(components["split_locus_mean_hamming"] + components["other_locus_mean_hamming"], cell["diversity"]["mean_hamming"])
                subset = fresh.set_index("genotype").loc[ids]
                close(cell["mean_affinity"], subset["mean"].mean())
                close(cell["mean_minus_sem"], (subset["mean"] - subset.effective_sem).mean())
                distances = np.array(independent_distances(ids))
                close(cell["diversity"]["mean_hamming"], distances.mean())
                close(cell["diversity"]["minimum_hamming"], distances.min())
                close(cell["diversity"]["hamming_le_one_pair_fraction"], (distances <= 1).mean())
                qualified = subset[subset["mean"] >= threshold].index.tolist()
                require(cell["quality_qualified_count"] == len(qualified), "Quality-qualified count differs")
                if len(qualified) >= 2:
                    close(cell["quality_qualified_diversity"]["mean_hamming"], np.mean(independent_distances(qualified)))
                positions = [ids_to_position[g] for g in ids]
                close(cell["frozen_sft_embedding_cosine"], cosine(frozen[positions]))
                close(cell["live_embedding_cosine"], cosine(features[positions]))
        samples = pd.read_csv(folder / "samples.csv", dtype={"genotype": "string"})
        n = len(samples)
        require(n == config["evaluation_samples"], "Wrong sample budget")
        counts = np.array(list(Counter(samples.genotype).values()))
        bits = np.array([list(map(int, g)) for g in samples.genotype])
        ones = bits.sum(0)
        close(measured["diversity"]["mean_pairwise_hamming_unbiased"], np.sum(2 * ones * (n - ones)) / (n * (n - 1)))
        close(measured["diversity"]["unique_genotypes"], len(counts))
        close(measured["diversity"]["entropy_nats_mc"], -samples.log_q.mean())
        close(measured["diversity"]["collision_probability_unbiased"], np.sum(counts * (counts - 1)) / (n * (n - 1)))
        ratio = samples.log_q - samples.reference_log_q
        close(measured["kl_to_sft_mc"]["nats"], ratio.mean())
        close(measured["kl_to_sft_mc"]["standard_error"], ratio.std(ddof=1) / np.sqrt(n))
        close(measured["frozen_sft_embedding_cosine"], cosine(np.load(folder / "sample_frozen_embeddings.npy", allow_pickle=False)))
        close(measured["live_embedding_cosine"], cosine(np.load(folder / "sample_live_embeddings.npy", allow_pickle=False)))
        if name != "sft":
            checkpoint_path = folder / f"decoder_step_{config['steps']:04d}.pt"
            require(sha256(checkpoint_path) == measured["checkpoint_sha256"], "Checkpoint changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            require(tensor_digest(checkpoint["decoder"]) == measured["decoder_state_sha256"], "Decoder state changed")
            require(checkpoint["initial_identity"] == result["initial_identity"] and checkpoint["config"] == config
                    and checkpoint["arm"] == measured["arm"] and checkpoint["seed"] == measured["seed"], "Checkpoint provenance differs")
            require(checkpoint["schedule_sha256"] == result["schedule_sha256"][str(measured["seed"])] == measured["schedule_sha256"], "Unequal labelled schedules")
            del checkpoint
            gc.collect()
            history = json.loads((folder / "history.json").read_text())
            require([h["step"] for h in history] == list(range(1, config["steps"] + 1)), "Training step gap")
            expected_count = config["steps"] // config["regularization_every"] * config["regularization_samples"] if measured["arm"]["kl"] else 0
            require(measured["regularization_sample_count"] == expected_count, "Regularization budget differs")
            if expected_count:
                training_samples = pd.read_csv(folder / "regularization_samples.csv", dtype={"genotype": "string"})
                require(len(training_samples) == expected_count, "Missing regularization samples")
                require(training_samples.groupby("step").size().to_dict() == {s: config["regularization_samples"] for s in range(4, 257, 4)}, "Regularization schedule differs")
                close(measured["unlabelled_regularization_unique_development_overlap"], len(set(training_samples.genotype) & set(fresh.genotype)))
        reports[name] = {"metrics_recomputed": True, "selection_recomputed": True, "checkpoint_verified": name != "sft"}
        print(f"Audited {name}", flush=True)
    evidence = {"passes": True, "output_artifacts_checked": len(result["output_sha256"]), "checkpoints_checked": 12,
        "training_weights_and_schedules_recomputed": True, "frozen_and_live_embedding_metrics_recomputed": True,
        "posthoc_split_locus_decomposition": decomposition,
        "reserved_test_labels_evaluated": False, "arms": reports}
    save_json(directory / "independent_audit.json", evidence)
    print(json.dumps(evidence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/cr9114_regularization_20260917")
    audit(parser.parse_args().directory)
