#!/usr/bin/env python
"""Build split-isolated preferences and evaluate existing CR9114 development scores.

CPU only. No model fitting, reference scoring, or reserved-test label aggregation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cr9114_esmif1_pilot import sha256, split_genotypes, stable_subset  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def replicate_statistics(frame):
    require(set(frame.split) <= {"train", "development"}, "Reserved or unknown split in statistics")
    reps = frame[["h1_repa", "h1_repb", "h1_repc"]].to_numpy(dtype=float)
    require(not np.isinf(reps).any(), "Infinite assay replicate")
    count = np.isfinite(reps).sum(axis=1)
    eligible = (count >= 2) & np.all(np.isnan(reps) | (reps > 7.0), axis=1)
    result = frame.loc[eligible, ["genotype", "split", "block"]].copy()
    result["replicate_count"] = count[eligible]
    result["mean"] = np.nanmean(reps[eligible], axis=1)
    result["sample_variance"] = np.nanvar(reps[eligible], axis=1, ddof=1)
    result["sample_sem"] = np.sqrt(result.sample_variance / result.replicate_count)
    return result.reset_index(drop=True)


def fit_uncertainty(records):
    train = records[records.split == "train"]
    require(len(train) > 1, "Insufficient training observations for variance floor")
    df = train.replicate_count.to_numpy() - 1
    pooled = float(np.sum(df * train.sample_variance) / df.sum())
    require(np.isfinite(pooled) and pooled > 0, "Invalid training variance floor")
    result = records.copy()
    result["effective_sem"] = np.sqrt(np.maximum(result.sample_variance, pooled) / result.replicate_count)
    return result, pooled


def construct_pairs(records, config):
    require(len(records) > 1 and records.genotype.is_unique, "Duplicate or insufficient pair candidates")
    require(len(set(records.split)) == 1 and set(records.split) <= {"train", "development"},
            "Pairs require one non-test split")
    cap, rounds = config["maximum_pairs_per_variant"], config["matching_rounds"]
    require(isinstance(cap, int) and cap > 0 and isinstance(rounds, int) and rounds > 0,
            "Pair cap and matching rounds must be positive integers")
    table = records.sort_values("genotype").reset_index(drop=True)
    degrees = np.zeros(len(table), dtype=int)
    edges, seen = [], set()
    for block, group in table.groupby("block", sort=True):
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], int(block)]))
        indices = group.index.to_numpy()
        for _ in range(rounds):
            order = rng.permutation(indices)
            left, right = order[:len(order) // 2], order[len(order) // 2:2 * (len(order) // 2)]
            gap = table["mean"].to_numpy()[left] - table["mean"].to_numpy()[right]
            uncertainty = np.hypot(table.effective_sem.to_numpy()[left], table.effective_sem.to_numpy()[right])
            keep = ((np.abs(gap) > config["minimum_mean_gap"])
                    & (np.abs(gap) > config["uncertainty_multiplier"] * uncertainty)
                    & (degrees[left] < cap) & (degrees[right] < cap))
            for a, b in zip(left[keep], right[keep]):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                if key in seen:
                    continue
                seen.add(key)
                chosen, rejected = (int(a), int(b)) if table.at[a, "mean"] > table.at[b, "mean"] else (int(b), int(a))
                edges.append((chosen, rejected))
                degrees[a] += 1
                degrees[b] += 1
    require(bool(edges), "No pairs pass the declared uncertainty rule")
    chosen, rejected = np.array(edges).T
    c, r = table.iloc[chosen].reset_index(drop=True), table.iloc[rejected].reset_index(drop=True)
    pairs = pd.DataFrame({"chosen_genotype": c.genotype, "rejected_genotype": r.genotype,
                          "split": c.split, "block": c.block,
                          "chosen_mean": c["mean"], "rejected_mean": r["mean"],
                          "mean_gap": c["mean"] - r["mean"],
                          "combined_effective_sem": np.hypot(c.effective_sem, r.effective_sem),
                          "chosen_degree": degrees[chosen], "rejected_degree": degrees[rejected]})
    active = table.loc[degrees > 0]
    counts = active.groupby("block").size()
    pairs["pair_weight"] = ((1 / pairs.chosen_degree + 1 / pairs.rejected_degree)
                             / pairs.block.map(counts) / len(counts))
    require(np.isclose(pairs.pair_weight.sum(), 1.0), "Pair weights do not normalize")
    require(int(degrees.max()) <= cap and (c.block == r.block).all() and (c.split == r.split).all(),
            "Pair cap or block/split isolation violated")
    audit = {"candidate_variants": len(table), "represented_variants": len(active),
             "unrepresented_variants": int((degrees == 0).sum()), "pair_count": len(pairs),
             "maximum_variant_degree": int(degrees.max()), "represented_blocks": len(counts),
             "pair_weight_sum": float(pairs.pair_weight.sum()),
             "hamming_distance_histogram": {str(int(k)): int(v) for k, v in
                 pd.Series([sum(a != b for a, b in zip(x, y)) for x, y in
                            zip(pairs.chosen_genotype, pairs.rejected_genotype)]).value_counts().sort_index().items()},
             "per_block": [{"block": int(block), "candidates": len(group),
                            "represented_variants": int((degrees[group.index] > 0).sum()),
                            "pairs": int((pairs.block == block).sum())}
                           for block, group in table.groupby("block", sort=True)]}
    return pairs, audit


def evaluate_pairs(pairs, scores, config):
    require(scores.genotype.is_unique and set(scores.split) == {"development"}, "Invalid development score identities")
    require(set(pairs.split) == {"development"}, "Evaluation requires development pairs")
    indexed = scores.set_index("genotype")
    columns = {"parent": "parent_log_q", "sft": "final_log_q", "additive_ridge": "additive_ridge_prediction"}
    require(np.isfinite(scores[list(columns.values())].to_numpy()).all(), "Nonfinite model score")
    result, scored = {}, pairs.copy()
    for name, column in columns.items():
        margin = (indexed.loc[pairs.chosen_genotype, column].to_numpy()
                  - indexed.loc[pairs.rejected_genotype, column].to_numpy())
        correct = np.where(margin > config["score_tie_tolerance"], 1.0,
                           np.where(margin < -config["score_tie_tolerance"], 0.0, 0.5))
        scored[f"{name}_correct"] = correct
        by_block = {str(int(block)): float(np.average(correct[group.index], weights=group.pair_weight))
                    for block, group in pairs.groupby("block", sort=True)}
        values = list(by_block.values())
        leave_one_out = [(sum(values) - v) / (len(values) - 1) for v in values] if len(values) > 1 else []
        result[name] = {"variant_and_block_balanced_accuracy": float(np.sum(correct * pairs.pair_weight)),
                        "unweighted_pair_accuracy": float(correct.mean()), "per_block_accuracy": by_block,
                        "leave_one_block_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
                        "score_ties": int((correct == 0.5).sum())}
    result["sft_minus_parent_pp"] = 100 * (result["sft"]["variant_and_block_balanced_accuracy"] - result["parent"]["variant_and_block_balanced_accuracy"])
    result["sft_minus_additive_pp"] = 100 * (result["sft"]["variant_and_block_balanced_accuracy"] - result["additive_ridge"]["variant_and_block_balanced_accuracy"])
    result["sft_mathematical_headroom_pp"] = 100 * (1 - result["sft"]["variant_and_block_balanced_accuracy"])
    result["paired_weight_mass"] = {
        "sft_better_than_parent": float(scored.loc[scored.sft_correct > scored.parent_correct, "pair_weight"].sum()),
        "sft_worse_than_parent": float(scored.loc[scored.sft_correct < scored.parent_correct, "pair_weight"].sum()),
        "equal_credit": float(scored.loc[scored.sft_correct == scored.parent_correct, "pair_weight"].sum())}
    topk = {}
    for k in config["top_k"]:
        require(0 < k <= len(scores), "Invalid top-K budget")
        oracle = float(scores.h1_mean_recomputed.nlargest(k).mean())
        models = {}
        for name, column in columns.items():
            selected = scores.sort_values([column, "genotype"], ascending=[False, True]).head(k)
            average = float(selected.h1_mean_recomputed.mean())
            models[name] = {"mean_measured_h1": average, "regret_to_same_pool_oracle": oracle - average}
        topk[str(k)] = {"same_pool_oracle_mean": oracle, "models": models}
    result["measured_top_k"] = topk
    result["uncertainty_note"] = "Pairs share variants. Leave-one-block-out ranges are sensitivity summaries, not confidence intervals. Only three development blocks: no confirmatory cluster interval or power claim."
    return result, scored


def run(config_path, pilot_dir, output):
    require(not output.exists(), "Output directory exists; choose a fresh directory")
    settings = json.loads(config_path.read_text())
    require(settings["schema_version"] == "cr9114-preferences/1" and settings["reserved_test_evaluation"] is False,
            "Unsupported preference configuration")
    require(settings["minimum_mean_gap"] > 0 and settings["uncertainty_multiplier"] > 0,
            "Preference thresholds must be positive")
    run_record = json.loads((pilot_dir / "run.json").read_text())
    config = run_record["config"]
    cohort = json.loads((pilot_dir / "cohort.json").read_text())
    require(sha256(ROOT / config["data"]) == config["data_sha256"], "Raw data hash changed")
    require(sha256(pilot_dir / "split.csv") == cohort["split_csv_sha256"], "Pilot split hash changed")
    frame = pd.read_csv(ROOT / config["data"], dtype={"genotype": "string"})
    split = pd.read_csv(pilot_dir / "split.csv", dtype={"genotype": "string"})
    require(frame.genotype.is_unique and split.genotype.is_unique and len(frame) == len(split) == 65536,
            "Duplicate or missing genotype identities")
    frame = frame.merge(split, on="genotype", validate="one_to_one")
    require(len(frame) == 65536, "Split does not cover the raw release")
    expected, split_info = split_genotypes(frame.genotype.tolist(), config)
    require(frame.split.tolist() == expected, "Split disagrees with the recorded sequence-only assignment")
    # Discard test rows before computing any replicate summary or pair.
    frame = frame.loc[frame.split != "test"].copy()
    frame["block"] = frame.genotype.map(lambda g: int("".join(g[i] for i in split_info["loci_0based"]), 2))
    records, pooled = fit_uncertainty(replicate_statistics(frame))
    train = records[records.split == "train"].copy()
    dev = records[records.split == "development"].copy()
    scores = pd.read_csv(pilot_dir / "development_scores.csv", dtype={"genotype": "string"})
    require(scores.genotype.is_unique and set(scores.split) == {"development"}, "Invalid score records")
    expected_dev = stable_subset(dev, config["development_sample_size"], config["seed"])
    require(set(expected_dev.genotype) == set(scores.genotype), "Scored development cohort changed")
    actual = dev.set_index("genotype").loc[scores.genotype]
    require(np.allclose(actual["mean"], scores.h1_mean_recomputed, rtol=0, atol=1e-12)
            and np.allclose(actual.sample_sem, scores.h1_sem_recomputed, rtol=0, atol=1e-12),
            "Saved score measurements disagree with raw replicates")
    # Independently reproduce the train-only additive baseline used by the pilot.
    def design(table):
        return np.array([[1.0, *map(float, g)] for g in table.genotype])
    x = design(train)
    penalty = np.eye(17)
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ train["mean"].to_numpy())
    require(np.allclose(design(scores) @ coefficients, scores.additive_ridge_prediction, rtol=0, atol=1e-10),
            "Additive baseline does not reproduce")
    print(f"Variance floor fitted on {len(train)} eligible training genotypes: SD={np.sqrt(pooled):.6f}", flush=True)
    train_pairs, train_audit = construct_pairs(train, settings)
    dev_pairs, dev_audit = construct_pairs(expected_dev, settings)
    require(not set(train.genotype).intersection(expected_dev.genotype), "Training/development overlap")
    metrics, scored = evaluate_pairs(dev_pairs, scores, settings)
    output.mkdir(parents=True)
    train_pairs.to_csv(output / "training_pairs.csv", index=False)
    dev_pairs.to_csv(output / "development_pairs.csv", index=False)
    scored.to_csv(output / "development_pair_scores.csv", index=False)
    records.to_csv(output / "eligible_non_test_records.csv", index=False)
    manifest = {"kind": "cr9114_preference_pairs_and_development_evaluation", "date": "2026-09-16",
                "config": settings, "config_sha256": sha256(config_path),
                "script_sha256": sha256(Path(__file__)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_worktree_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "scope": "Development exploration on existing pilot scores; no new training or final-test evaluation.",
                "pilot_source_commit": run_record["git_commit"],
                "prepared_artifact": config["prepared_artifact"],
                "prepared_artifact_sha256": run_record["prepared_artifact_sha256"],
                "data_sha256": config["data_sha256"],
                "input_files": {name: sha256(pilot_dir / name) for name in
                                ("run.json", "cohort.json", "split.csv", "development_scores.csv", "checkpoint_reload.json")},
                "training_pooled_sample_variance": pooled, "training_pooled_sd": float(np.sqrt(pooled)),
                "train": train_audit, "development": dev_audit, "development_metrics": metrics,
                "split_isolation_verified": True, "reserved_test_evaluated": False,
                "output_files": {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size}
                                 for p in sorted(output.iterdir())}}
    write_json(output / "manifest.json", manifest)
    print(f"Constructed {len(train_pairs)} training pairs and {len(dev_pairs)} development pairs.", flush=True)
    for name in ("parent", "sft", "additive_ridge"):
        print(f"{name} balanced pair accuracy: {metrics[name]['variant_and_block_balanced_accuracy']:.4%}", flush=True)
    print(f"SFT mathematical headroom: {metrics['sft_mathematical_headroom_pp']:.3f} percentage points", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_preferences.json")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "outputs/cr9114_5cjq_pilot_20260916")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.pilot_dir, args.output_dir)


if __name__ == "__main__":
    main()
