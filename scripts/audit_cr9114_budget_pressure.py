#!/usr/bin/env python
"""Independently recompute the budget-pressure run's statistics from saved artifacts.

Nothing here calls the runner's metric helpers. Each statistic this script
re-derives is computed from the saved arrays, tables and checkpoints with its
own formula, so an agreement is evidence about the numbers rather than about
shared code.

What this script does NOT establish: recorded metadata is reported, not
re-derived. Torch/CUDA versions, the deterministic-algorithm flag, the
strict-reload / optimizer-carried-forward / encoder-unchanged / RNG-invariance
assertions and the timing fields were checked live inside the run and are only
read back here for internal consistency. Reproducing the fitted trajectory
itself would require rerunning training, which this audit deliberately does not
do.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_cr9114_shortlist import independent_distances, independent_selection
from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]

RECORDS = "outputs/cr9114_preferences_verified_20260916/eligible_non_test_records.csv"
COHORTS = [("outputs/cr9114_5cjq_pilot_20260916/development_scores.csv", None),
           ("outputs/cr9114_dpo_diagnostics_20260916/fresh_development.csv",
            "reference/evidence/cr9114-dpo-diagnostics-2026-09-16.json"),
           ("outputs/cr9114_diversity_pilot_20260916_v3/fresh_development.csv",
            "reference/evidence/cr9114-diversity-pilot-2026-09-16.json"),
           ("outputs/cr9114_affinity_pilot_20260916/fresh_development.csv",
            "reference/evidence/cr9114-affinity-pilot-2026-09-17.json")]
CATEGORIES = ("measured_train", "measured_development", "train_ineligible",
              "development_ineligible", "eligible_development_withheld", "test")
# Score CSVs are compared bit for bit against their .npy and pinned sources, and
# default pandas float parsing is not a binary round trip. Same reader as the
# runner, so a saved score and its source stay exactly equal.
SCORE_CSV = {"dtype": {"genotype": "string"}, "float_precision": "round_trip"}
PINNED_RUN_ARTIFACTS = ("history.json", "schedule.npy", "training_population.csv", "development_records.csv",
                        "allowed_development.csv", "sft_exposures.csv", "continuation_exposures.csv")
PINNED_CHECKPOINT_ARTIFACTS = ("development_scores.csv", "selections_before_evaluation.json",
                               "samples.csv", "result.json")


def close(actual, expected, tolerance=1e-9):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def log_sum_exp(values):
    """Deliberately not numpy's pairwise logaddexp reduction."""
    peak = float(values.max())
    return peak + float(np.log(np.exp(values - peak).sum()))


def bit_matrix(genotypes):
    return np.frombuffer("".join(genotypes).encode(), dtype=np.uint8).reshape(len(genotypes), -1) - ord("0")


def build_evaluation(records, allowed):
    """Only training labels and the whitelisted development labels may pass."""
    require(records.genotype.is_unique, "Duplicate measurement identities")
    require(set(records.split) <= {"train", "development"}, "Reserved measurements present in the source table")
    require(set(allowed) <= set(records.loc[records.split == "development", "genotype"]), "Unknown whitelist identity")
    keep = (records.split == "train") | records.genotype.isin(set(allowed))
    return records.loc[keep, ["genotype", "split", "block", "mean", "effective_sem"]].reset_index(drop=True)


def conditional_affinity(frame, weights, threshold):
    """Conditional mean over measured identities, weights never renormalized first."""
    values = np.array([float(weights.get(genotype, 0.)) for genotype in frame.genotype])
    keep = values > 0
    mass = float(values[keep].sum())
    means, sems = frame["mean"].to_numpy()[keep], frame.effective_sem.to_numpy()[keep]
    if mass == 0:
        return {"measured_mass": 0., "conditional_mean_affinity": None, "assay_sem_proxy": None,
                "positive_mass": 0., "conditional_positive_fraction": None, "distinct_measured_genotypes": 0,
                "conditional_effective_genotypes": None}
    share = values[keep] / mass
    positive = means >= threshold
    return {"measured_mass": mass, "conditional_mean_affinity": float((share * means).sum()),
            "assay_sem_proxy": float(np.sqrt(((share * sems) ** 2).sum())),
            "positive_mass": float((values[keep] * positive).sum()),
            "conditional_positive_fraction": float((share * positive).sum()),
            "distinct_measured_genotypes": int(keep.sum()),
            "conditional_effective_genotypes": float(1. / (share ** 2).sum())}


def affinity_delta(frame, weights, reference):
    first = np.array([float(weights.get(genotype, 0.)) for genotype in frame.genotype])
    second = np.array([float(reference.get(genotype, 0.)) for genotype in frame.genotype])
    if first.sum() == 0 or second.sum() == 0:
        return {"conditional_affinity_delta": None, "assay_sem_proxy_for_delta": None}
    coefficients = first / first.sum() - second / second.sum()
    return {"conditional_affinity_delta": float((coefficients * frame["mean"].to_numpy()).sum()),
            "assay_sem_proxy_for_delta": float(np.sqrt(((coefficients * frame.effective_sem.to_numpy()) ** 2).sum()))}


def category_counts(genotypes, split_of, measured, eligible, weights=None):
    table = {name: 0. for name in CATEGORIES}
    for position, genotype in enumerate(genotypes):
        assignment = split_of[genotype]
        if assignment == "test":
            name = "test"
        elif genotype in measured:
            name = f"measured_{assignment}"
        elif assignment == "train":
            name = "train_ineligible"
        else:
            name = "eligible_development_withheld" if genotype in eligible else "development_ineligible"
        table[name] += 1. if weights is None else float(weights[position])
    total = float(len(genotypes)) if weights is None else float(np.sum(weights))
    if weights is None:
        return dict({name: int(value) for name, value in table.items()}, total=int(total))
    return dict(table, total=total)


def recompute_exhaustive(log_q, reference_log_q, support, context):
    """Every exact endpoint statistic, from the saved log-probability arrays."""
    total, reference_total = log_sum_exp(log_q), log_sum_exp(reference_log_q)
    mass = float(np.exp(total))
    logp, logr = log_q - total, reference_log_q - reference_total
    p, r = np.exp(logp), np.exp(logr)
    entropy = float(-(p * logp).sum())
    weights = dict(zip(support, p.tolist()))
    bits = context["bits"]
    marginals = [float(p[bits[:, j] == 1].sum()) for j in range(bits.shape[1])]
    document = {"normalization_mass_before_correction": mass,
                "reference_normalization_mass": float(np.exp(reference_total)),
                "entropy_nats": entropy, "entropy_effective_support": float(np.exp(entropy)),
                "collision_probability": float((p * p).sum()), "collision_effective_support": float(1 / (p * p).sum()),
                "maximum_genotype_probability": float(p.max()),
                "kl_to_sft_nats": float((p * (logp - logr)).sum()),
                "kl_from_sft_nats": float((r * (logr - logp)).sum()),
                "total_variation_from_sft": float(np.abs(p - r).sum() / 2),
                "site_allele_one_marginals": marginals,
                "split_defining_site_marginals": {str(i): marginals[i] for i in context["loci"]},
                "other_site_marginals": {str(i): marginals[i] for i in range(len(marginals))
                                         if i not in context["loci"]},
                "block_mass": {str(int(b)): float(p[context["blocks"] == b].sum())
                               for b in sorted(set(context["blocks"].tolist()))},
                "split_mass": {str(s): float(p[context["split_array"] == s].sum())
                               for s in sorted(set(context["split_array"].tolist()))},
                "category_mass": category_counts(support, context["split_of"], context["measured"],
                                                 context["eligible"], p),
                "seen_labelled_mass": {name: float(sum(weights[g] for g in identities)) if identities else 0.
                                       for name, identities in context["seen"].items()},
                "conditional_affinity": {}, "conditional_affinity_delta_from_sft": {}}
    reference_weights = context["reference_weights"] if context["reference_weights"] is not None else weights
    for name, frame in context["populations"].items():
        document["conditional_affinity"][name] = conditional_affinity(frame, weights, context["threshold"])
        document["conditional_affinity_delta_from_sft"][name] = affinity_delta(frame, weights, reference_weights)
    document["measured_threshold_exceedance_mass"] = float(
        document["conditional_affinity"]["train"]["positive_mass"]
        + document["conditional_affinity"]["development"]["positive_mass"])
    return document, weights


def compare(recomputed, reported, path=""):
    """Structural comparison so a missing or renamed reported field is a failure."""
    require(set(recomputed) <= set(reported), f"Reported document lacks recomputed fields at {path}")
    for key, value in recomputed.items():
        where = f"{path}.{key}"
        if isinstance(value, dict):
            compare(value, reported[key], where)
        elif value is None or isinstance(value, (str, bool)):
            require(value == reported[key], f"Mismatch at {where}")
        elif isinstance(value, list):
            close(np.asarray(value, dtype=float), np.asarray(reported[key], dtype=float))
        else:
            close(value, reported[key])


def audit(directory):
    result = json.loads((directory / "results.json").read_text())
    config = result["config"]
    require(result["status"] == "completed" and not result["reserved_test_labels_evaluated"]
            and not result["withheld_development_evaluated"] and not result["checkpoint_promoted"],
            "Run is incomplete or claims an out-of-scope evaluation")
    require(result["schema_version"] == "cr9114-budget-pressure-result/1", "Unexpected result schema")
    for relative, digest in result["output_sha256"].items():
        require(sha256(directory / relative) == digest, f"Artifact changed: {relative}")
    require("results.json" not in result["output_sha256"], "The hash table must not hash the document carrying it")
    require(set(PINNED_RUN_ARTIFACTS) <= set(result["output_sha256"]),
            "A saved run artifact is missing from output_sha256")
    require(sha256(ROOT / "configs/experiments/cr9114_budget_pressure.json") == result["config_sha256"]
            and sha256(ROOT / "scripts/run_cr9114_budget_pressure.py") == result["script_sha256"],
            "Protocol config or runner changed after the run")
    require(config["kl_coefficient"] == config["entropy_coefficient"] == config["embedding_coefficient"] == 0.,
            "Declared arm carried an explicit regularizer")

    records = pd.read_csv(ROOT / RECORDS, dtype={"genotype": "string"})
    require(sha256(ROOT / RECORDS) == result["input_records_sha256"], "Measurement source changed")
    train = records[records.split == "train"]
    threshold = float(train["mean"].quantile(config["positive_quantile"]))
    pool_frame = train[train["mean"] >= threshold].sort_values("genotype").reset_index(drop=True)
    variance = float(np.average(train.sample_variance, weights=train.replicate_count - 1))
    sem = np.sqrt(np.maximum(pool_frame.sample_variance, variance) / pool_frame.replicate_count)
    utility = pool_frame["mean"] - config["uncertainty_multiplier"] * sem
    raw = np.maximum(np.exp((utility - utility.max()) / np.sqrt(variance)), 1 / config["max_weight_ratio"])
    probability = ((1 - config["uniform_fraction"]) * raw / raw.sum()
                   + config["uniform_fraction"] / len(raw)).to_numpy()
    population = pd.read_csv(directory / "training_population.csv", dtype={"genotype": "string"})
    require(population.genotype.tolist() == pool_frame.genotype.tolist()
            and len(population) == config["training_population_count"], "Training population changed")
    close(population.affinity_probability, probability)
    close(result["population"]["weighted_expected_affinity"], float(probability @ pool_frame["mean"].to_numpy()))
    close(result["population"]["mean_threshold"], threshold)

    cumulative = np.cumsum(probability / probability.sum())
    cumulative[-1] = 1.
    draws = np.random.default_rng(config["seed"]).random((config["steps"], config["batch_size"]))
    schedule = np.searchsorted(cumulative, draws, side="right")
    saved = np.load(directory / "schedule.npy", allow_pickle=False)
    np.testing.assert_array_equal(saved, schedule)
    prior_dir = ROOT / config["prior_directory"]
    prefix = np.load(prior_dir / config["prior_schedule_artifact"], allow_pickle=False)
    require(sha256(prior_dir / config["prior_schedule_artifact"]) == config["prior_schedule_sha256"],
            "Pinned prior schedule changed")
    np.testing.assert_array_equal(schedule[:len(prefix)], prefix)

    positives = pd.read_csv(ROOT / config["pilot_directory"] / config["training_positives_artifact"],
                            dtype={"genotype": "string"})
    require(sha256(ROOT / config["pilot_directory"] / config["training_positives_artifact"])
            == config["training_positives_sha256"], "Original SFT positives changed")
    split = pd.read_csv(ROOT / config["pilot_directory"] / "split.csv", dtype={"genotype": "string"})
    require(sha256(ROOT / config["pilot_directory"] / "split.csv") == config["split_csv_sha256"], "Split changed")
    require(positives.genotype.tolist()
            == split[split.genotype.isin(set(positives.genotype))].genotype.tolist(), "Positives row order changed")
    generator = np.random.default_rng(config["sft_exposure_seed"])
    identifiers = positives.genotype.tolist()
    replay = [identifiers[int(i)] for i in generator.integers(
        0, len(identifiers), size=(config["sft_exposure_steps"], config["sft_exposure_batch_size"])).ravel()]
    exposures = pd.read_csv(directory / "sft_exposures.csv", dtype={"genotype": "string"})
    require(exposures.genotype.tolist() == replay, "Saved SFT exposures differ from an independent replay")
    np.testing.assert_array_equal(exposures.step.to_numpy(), np.repeat(
        np.arange(1, config["sft_exposure_steps"] + 1), config["sft_exposure_batch_size"]))
    require(len(replay) == result["original_sft_exposure_draws"] == config["expected_sft_exposure_draws"]
            and len(set(replay)) == result["original_sft_exposure_unique"]
            == config["expected_sft_exposure_unique"], "SFT exposure counts changed")
    pilot_checkpoint = torch.load(ROOT / "outputs/cr9114_5cjq_pilot_20260916/decoder_step_0256.portable.pt",
                                  map_location="cpu", weights_only=True)
    require(pilot_checkpoint.get("numpy_rng_state") == generator.bit_generator.state,
            "Replayed generator state differs from the original SFT checkpoint")
    del pilot_checkpoint
    gc.collect()

    previous = set()
    for relative, evidence_path in COHORTS:
        if evidence_path:
            document = json.loads((ROOT / evidence_path).read_text())
            require(sha256(ROOT / relative) == document.get("run", document)["fresh_cohort_sha256"],
                    f"Earlier cohort changed: {relative}")
        require(sha256(ROOT / relative) == result["prior_cohort_sha256"][relative], f"Cohort hash differs: {relative}")
        previous.update(pd.read_csv(ROOT / relative, dtype={"genotype": "string"}).genotype)
    shortlist_dir = ROOT / config["shortlist_directory"]
    shortlist = json.loads((shortlist_dir / "results.json").read_text())
    require(sha256(shortlist_dir / "results.json") == result["shortlist_result_sha256"]
            == config["shortlist_results_sha256"], "Selector input changed")
    require(sha256(shortlist_dir / "fresh_development.csv") == shortlist["output_sha256"]["fresh_development.csv"],
            "Pinned fresh development cohort changed")
    # The locally saved pool is a copy; compare it to the pinned cohort and to the
    # safe source labels rather than trusting the copy the run wrote for itself.
    cohort = pd.read_csv(shortlist_dir / "fresh_development.csv", **SCORE_CSV)
    pool = pd.read_csv(directory / "development_records.csv", **SCORE_CSV)
    require(pool.columns.tolist() == cohort.columns.tolist() and pool.genotype.tolist() == cohort.genotype.tolist()
            and pool.split.tolist() == cohort.split.tolist() and pool.block.tolist() == cohort.block.tolist()
            and pool.replicate_count.tolist() == cohort.replicate_count.tolist()
            and set(pool.split) == {"development"} and len(pool) == config["fresh_development_count"],
            "Saved development pool differs from the pinned shortlist cohort")
    for column in ("mean", "sample_variance", "sample_sem", "effective_sem"):
        close(pool[column].to_numpy(), cohort[column].to_numpy(), 0.)
    source = records.set_index("genotype").loc[pool.genotype.tolist()]
    require(source.split.tolist() == pool.split.tolist() and source.block.tolist() == pool.block.tolist(),
            "Saved development pool disagrees with the safe source split or block labels")
    close(pool["mean"].to_numpy(), source["mean"].to_numpy())
    close(pool.effective_sem.to_numpy(), source.effective_sem.to_numpy())
    require(len(previous) == config["prior_development_count"] and not previous & set(pool.genotype),
            "Whitelist cohorts overlap or changed")
    allowed = sorted(previous | set(pool.genotype))
    require(len(allowed) == config["allowed_development_total"] == result["allowed_development_count"],
            "Allowed development population changed")
    require(pd.read_csv(directory / "allowed_development.csv", dtype={"genotype": "string"}).genotype.tolist()
            == allowed, "Saved whitelist differs")
    eligible = set(records.loc[records.split == "development", "genotype"])
    withheld = sorted(eligible - set(allowed))
    require(len(withheld) == config["withheld_development_count"] == result["withheld_development_count"],
            "Withheld development count changed")
    evaluation = build_evaluation(records, allowed)
    require(not set(evaluation.genotype) & set(withheld), "A withheld development identity reached evaluation")
    require(len(evaluation) == config["eligible_training_count"] + config["allowed_development_total"],
            "Evaluation population size changed")
    support = [f"{i:016b}" for i in range(config["legal_identity_count"])]
    require(sorted(split.genotype) == support, "Split file does not cover the legal support")
    split_of = dict(zip(split.genotype, split.split))
    require([split_of[g] for g in evaluation.genotype] == evaluation.split.tolist(), "Split assignment disagrees")
    loci = config["split_defining_loci_0based"]
    require([int("".join(g[i] for i in loci), 2) for g in evaluation.genotype] == evaluation.block.tolist(),
            "Split-defining loci changed")

    bits = bit_matrix(support)
    context = {"bits": bits, "blocks": np.array([int("".join(g[i] for i in loci), 2) for g in support]),
               "split_array": np.array([split_of[g] for g in support]), "split_of": split_of,
               "measured": set(evaluation.genotype), "eligible": eligible, "threshold": threshold,
               "reference_weights": None, "loci": loci,
               "populations": {"train": evaluation[evaluation.split == "train"].reset_index(drop=True),
                               "development": evaluation[evaluation.split == "development"].reset_index(drop=True),
                               **{f"development_block_{block}": evaluation[(evaluation.split == "development")
                                  & (evaluation.block == block)].reset_index(drop=True)
                                  for block in config["development_blocks"]}}}
    require(threshold == result["population"]["mean_threshold"], "Threshold differs from the recorded population")

    multiplier = shortlist["calibration"]["selected_multiplier"]
    steps = [0, *config["checkpoints"]]
    require(sorted(int(key) for key in result["checkpoints"]) == steps, "Checkpoint set changed")
    sft_log_q = np.load(directory / "sft/exhaustive_log_q.npy", allow_pickle=False)
    require(sft_log_q.shape == (config["legal_identity_count"],) and np.isfinite(sft_log_q).all()
            and (sft_log_q <= 0).all(), "Invalid exhaustive SFT scores")
    close(result["sft_support_mass_before_normalization"], float(np.exp(log_sum_exp(sft_log_q))))
    require(abs(float(np.exp(log_sum_exp(sft_log_q))) - 1) <= config["exact_probability_tolerance"],
            "SFT support mass is outside the declared tolerance")
    require(sha256(shortlist_dir / "development_scores.csv") == shortlist["output_sha256"]["development_scores.csv"],
            "Pinned shortlist development scores changed")
    pinned_scores = pd.read_csv(shortlist_dir / "development_scores.csv", **SCORE_CSV)
    pinned_error = float(np.max(np.abs(sft_log_q[[int(g, 2) for g in pinned_scores.genotype]]
                                       - pinned_scores.score.to_numpy())))
    close(result["sft_pool_vs_pinned_shortlist_max_error"], pinned_error)
    require(pinned_error < config["pool_score_tolerance"], "SFT scores drifted from the pinned shortlist pass")
    continuation = population.genotype.to_numpy()[schedule]
    exposed = pd.read_csv(directory / "continuation_exposures.csv", dtype={"genotype": "string"})
    require(exposed.genotype.tolist() == continuation.ravel().tolist(),
            "Saved continuation exposures differ from the independently recomputed schedule")
    np.testing.assert_array_equal(exposed.step.to_numpy(), np.repeat(
        np.arange(1, config["steps"] + 1), config["batch_size"]))
    checks, sft_selected, sft_weights, previous_selected = {}, None, None, None
    for step in steps:
        folder = directory / ("sft" if step == 0 else f"step_{step:05d}")
        measured = result["checkpoints"][str(step)]
        require(measured["step"] == step, "Checkpoint step label differs")
        require(all(f"{folder.name}/{item}" in result["output_sha256"] for item in PINNED_CHECKPOINT_ARTIFACTS),
                f"A {folder.name} artifact is missing from output_sha256")
        scores = pd.read_csv(folder / "development_scores.csv", **SCORE_CSV)
        require(scores.genotype.tolist() == pool.genotype.tolist(), "Score order changed")
        require(set(scores.columns) == {"genotype", "score"}, "Label-blind scores carry extra columns")
        selections = json.loads((folder / "selections_before_evaluation.json").read_text())
        for mode, factor in (("ordinary", 1), ("diverse", multiplier)):
            for budget in config["budgets"]:
                key = str(budget)
                chosen = independent_selection(scores, budget, factor)
                require(chosen == selections[mode][key], "Saved pre-join selection differs")
                cell = measured["portfolios"][mode][key]
                require(chosen == cell["selected_genotypes"], "Reported portfolio differs from the selector")
                subset = pool.set_index("genotype").loc[chosen]
                close(cell["mean_affinity"], subset["mean"].mean())
                close(cell["mean_minus_sem"], (subset["mean"] - subset.effective_sem).mean())
                distances = np.array(independent_distances(chosen))
                close(cell["diversity"]["mean_hamming"], distances.mean())
                close(cell["diversity"]["minimum_hamming"], distances.min())
                if sft_selected is not None:
                    require(cell["shared_with_sft"] == len(set(chosen) & set(sft_selected[mode][key])), "Overlap")
                    require(cell["swapped_from_sft"] == len(set(chosen) - set(sft_selected[mode][key])), "Swaps")
                if previous_selected is not None:
                    require(cell["shared_with_previous_checkpoint"]
                            == len(set(chosen) & set(previous_selected[mode][key])), "Sequential overlap")
        pool_weights = dict(zip(scores.genotype, np.exp(scores.score.to_numpy()).tolist()))
        compare(conditional_affinity(pool, pool_weights, threshold), measured["pool_conditional_affinity"], "pool")
        compare(affinity_delta(pool, pool_weights, sft_weights if sft_weights is not None else pool_weights),
                measured["pool_conditional_delta_from_sft"], "pool_delta")

        draws = pd.read_csv(folder / "samples.csv", **SCORE_CSV)
        require(set(draws.columns) == {"genotype", "log_q", "reference_log_q"}, "Draw file carries extra columns")
        require(len(draws) == config["evaluation_samples"]
                and measured["sample_rescore_max_error"] < config["sample_rescore_tolerance"],
                "Draw count or teacher-forced rescore agreement changed")
        reuse = config["reused_sample_artifacts"].get(str(step))
        if reuse is None:
            require(measured["sample_source"] == "generated", "Undeclared sample reuse")
        else:
            require(sha256(prior_dir / reuse) == config["reused_sample_sha256"][str(step)], "Pinned draws changed")
            pinned = pd.read_csv(prior_dir / reuse, **SCORE_CSV)
            require(draws.genotype.tolist() == pinned.genotype.tolist(), "Reused draw identities differ")
            close(draws.log_q.to_numpy(), pinned.log_q.to_numpy(), 0.)
        close(draws.reference_log_q.to_numpy(), sft_log_q[[int(g, 2) for g in draws.genotype]], 0.)
        multiplicity = Counter(draws.genotype)
        counts = np.array(sorted(multiplicity.values()))
        n = len(draws)
        frequencies = bit_matrix(draws.genotype.tolist()).mean(axis=0)
        diversity = measured["diversity"]
        close(diversity["sample_count"], n)
        close(diversity["unique_genotypes"], len(counts))
        close(diversity["duplicate_fraction"], 1 - len(counts) / n)
        close(diversity["largest_observed_genotype_fraction"], counts.max() / n)
        close(diversity["entropy_nats_mc"], -draws.log_q.mean())
        close(diversity["entropy_mc_standard_error"], draws.log_q.std(ddof=1) / np.sqrt(n))
        close(diversity["entropy_effective_support_mc"], np.exp(-draws.log_q.mean()))
        close(diversity["collision_probability_unbiased"], np.sum(counts * (counts - 1)) / (n * (n - 1)))
        close(diversity["mean_pairwise_hamming_unbiased"],
              2 * (frequencies * (1 - frequencies)).sum() * n / (n - 1))
        close(np.asarray(diversity["allele_one_frequencies"], dtype=float), frequencies)
        close(diversity["sites_with_minor_allele_below_1pct"],
              (np.minimum(frequencies, 1 - frequencies) < .01).sum())
        ratio = draws.log_q.to_numpy() - draws.reference_log_q.to_numpy()
        close(measured["kl_to_sft_mc"]["nats"], ratio.mean())
        close(measured["kl_to_sft_mc"]["standard_error"], ratio.std(ddof=1) / np.sqrt(n))
        require(measured["kl_to_sft_mc"]["draws"] == n, "Monte Carlo draw count differs")
        seen = {"sft": set(replay), "continuation": set(continuation[:step].ravel().tolist()) if step else set()}
        seen["union"] = seen["sft"] | seen["continuation"]
        for name, identities in seen.items():
            overlap = measured["sampled_exposure_overlap"][name]
            require(overlap["draw_count"] == sum(g in identities for g in draws.genotype)
                    and overlap["unique_count"] == len(set(draws.genotype) & identities)
                    and overlap["exposure_unique_count"] == len(identities), "Exposure overlap differs")
        exposure = measured["exposure_counts"]
        require(exposure["sft_unique"] == len(seen["sft"]) and exposure["union_unique"] == len(seen["union"])
                and exposure["continuation_unique"] == len(seen["continuation"]), "Exposure identity count differs")
        if step:
            require(exposure["continuation_draws"] == step * config["batch_size"]
                    and exposure["sft_continuation_shared_unique"] == len(seen["sft"] & seen["continuation"]),
                    "Continuation exposure count differs")
        sampled = measured["sampled_affinity"]
        require(sampled["sample_count"] == n and sampled["unique_draw_count"] == len(counts)
                and sampled["split_counts"] == {name: int(value) for name, value
                                                in Counter(split_of[g] for g in draws.genotype).items()},
                "Sampled draw or split counts differ")
        require(sampled["training_exposure_overlap_draw_count"] == sum(g in seen["union"] for g in draws.genotype)
                and sampled["training_exposure_overlap_unique_count"] == len(set(draws.genotype) & seen["union"]),
                "Sampled training-exposure overlap differs")
        frequency = {g: c / n for g, c in multiplicity.items()}
        for name in ("train", "development"):
            frame = context["populations"][name]
            compare(conditional_affinity(frame, frequency, threshold),
                    sampled["by_split"][name], f"sampled.{name}")
            available = set(frame.genotype)
            hits = [g for g in draws.genotype if g in available]
            require(sampled["by_split"][name]["measured_draw_count"] == len(hits),
                    f"Measured draw count differs for {name}")
            values = frame.set_index("genotype").loc[hits, "mean"].to_numpy() if hits else np.array([])
            if len(values) >= 2:
                close(sampled["by_split"][name]["sampling_sem_conditional_mean"],
                      values.std(ddof=1) / np.sqrt(len(values)))
            else:
                require(sampled["by_split"][name]["sampling_sem_conditional_mean"] is None,
                        f"A sampling SE was reported for fewer than two measured draws in {name}")
        compare(category_counts(draws.genotype.tolist(), split_of, context["measured"], eligible),
                measured["sampled_draw_categories"], "draw_categories")
        require(sampled["unscored_draw_count"]
                == sum(g not in context["measured"] for g in draws.genotype), "Unscored draw count differs")

        if step in config["exhaustive_checkpoints"]:
            log_q = np.load(folder / "exhaustive_log_q.npy", allow_pickle=False)
            require(log_q.shape == sft_log_q.shape and np.isfinite(log_q).all() and (log_q <= 1e-6).all(),
                    "Invalid exhaustive scores")
            require(f"{folder.name}/exhaustive_log_q.npy" in result["output_sha256"],
                    "Exhaustive score array is not pinned in output_sha256")
            context["seen"] = seen
            recomputed, weights = recompute_exhaustive(log_q, sft_log_q, support, context)
            compare(recomputed, measured["exhaustive"], f"exhaustive.{step}")
            # The normalization gate the run claims, re-derived at every scored endpoint.
            require(abs(recomputed["normalization_mass_before_correction"] - 1)
                    <= config["exact_probability_tolerance"],
                    f"Endpoint {step} support mass is outside the declared tolerance")
            close(sum(recomputed["category_mass"][name] for name in CATEGORIES), 1.)
            close(sum(recomputed["split_mass"].values()), 1.)
            close(sum(recomputed["block_mass"].values()), 1.)
            require(len(recomputed["block_mass"]) == 16 and len(recomputed["site_allele_one_marginals"]) == 16,
                    "Incomplete block or site table")
            if step == 0:
                context["reference_weights"] = weights
            checks[str(step)] = {"exhaustive_recomputed": True}
        else:
            checks[str(step)] = {"exhaustive_recomputed": False}
        if step:
            path = folder / f"decoder_step_{step:05d}.pt"
            require(sha256(path) == measured["checkpoint_sha256"], "Checkpoint file changed")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            require(tensor_digest(checkpoint["decoder"]) == measured["decoder_state_sha256"], "Decoder state changed")
            require(all(bool(torch.isfinite(t).all()) for t in checkpoint["decoder"].values()
                        if t.is_floating_point()), "Nonfinite saved parameter")
            require(checkpoint["step"] == step and checkpoint["config"] == config
                    and checkpoint["initial_identity"] == result["initial_identity"]
                    and checkpoint["git_commit"] == result["git_commit"]
                    and checkpoint["schedule_sha256"] == result["schedule_sha256"], "Checkpoint provenance differs")
            require({int(entry["step"]) for entry in checkpoint["optimizer"]["state"].values()} == {step},
                    "Optimizer step counter disagrees with the checkpoint")
            if step == config["checkpoints"][0]:
                require(measured["decoder_state_sha256"] == config["expected_step_256_decoder_state_sha256"],
                        "Step-256 decoder state does not reproduce the prior affinity-only arm")
                checks[str(step)]["reproduces_prior_arm_decoder_state"] = True
            del checkpoint
            gc.collect()
        sft_selected = sft_selected or selections
        sft_weights = sft_weights or pool_weights
        previous_selected = selections
        print(f"Audited checkpoint {step}", flush=True)

    history = json.loads((directory / "history.json").read_text())
    require([entry["step"] for entry in history] == list(range(1, config["steps"] + 1)), "Incomplete history")
    require(all(np.isfinite(entry["nll_per_site"]) and np.isfinite(entry["gradient_norm"])
                and entry["gradient_norm"] > 0 and np.isfinite(entry["elapsed_seconds"])
                and np.isfinite(entry["training_seconds"]) for entry in history),
            "Nonfinite loss, nonfinite gradient norm or vanished gradient in the recorded history")
    require(all(a["elapsed_seconds"] <= b["elapsed_seconds"] and a["training_seconds"] <= b["training_seconds"]
                for a, b in zip(history, history[1:])), "History is not in wall-clock order")
    # training_seconds must be the optimizer loop only, so it cannot exceed wall clock.
    require(history[-1]["training_seconds"] == result["training_seconds"]
            and result["training_seconds"] <= result["wall_clock_seconds"]
            and all(entry["training_seconds"] <= entry["elapsed_seconds"] for entry in history),
            "Reported training seconds are not the accumulated optimizer-loop time")
    close(result["evaluation_and_io_seconds"], result["wall_clock_seconds"] - result["training_seconds"])
    for step in config["checkpoints"]:
        cell = result["checkpoints"][str(step)]
        require(cell["training_seconds"] == history[step - 1]["training_seconds"]
                and cell["training_seconds"] <= cell["wall_clock_seconds"],
                f"Checkpoint {step} timing disagrees with the recorded history")
        close(cell["evaluation_and_io_seconds"], cell["wall_clock_seconds"] - cell["training_seconds"])
    require(sha256(directory / "schedule.npy") == result["schedule_sha256"], "Schedule artifact changed")

    perturbed = records.copy()
    perturbed.loc[perturbed.genotype.isin(set(withheld)), ["mean", "effective_sem"]] = [-999., 999.]
    isolated = build_evaluation(perturbed, allowed)
    pd.testing.assert_frame_equal(isolated, evaluation)
    sentinel_context = dict(context, populations={name: isolated[isolated.genotype.isin(set(frame.genotype))
                                                                 ].reset_index(drop=True)
                                                  for name, frame in context["populations"].items()})
    endpoint = np.load(directory / f"step_{config['steps']:05d}/exhaustive_log_q.npy", allow_pickle=False)
    again, _ = recompute_exhaustive(endpoint, sft_log_q, support, sentinel_context)
    require(again == recompute_exhaustive(endpoint, sft_log_q, support, context)[0],
            "Withheld development labels changed a reported statistic")
    synthetic = pd.DataFrame({"genotype": ["0" * 16, "1" * 16], "split": ["train", "test"], "block": [0, 0],
                              "mean": [9., 99.], "effective_sem": [.1, .1]})
    try:
        build_evaluation(synthetic, [])
        raise AssertionError("Test-labelled rows must be rejected, not filtered away")
    except ValueError:
        pass

    evidence = {"passes": True, "audited_directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
        "output_artifacts_checked": len(result["output_sha256"]), "checkpoints_checked": len(config["checkpoints"]),
        "independent_formulas": True, "schedule_and_population_recomputed": True,
        "original_sft_exposures_recomputed": True, "continuation_exposures_recomputed": True,
        "whitelist_recomputed": True, "development_pool_compared_to_pinned_cohort_and_source": True,
        "endpoint_normalization_regated": True, "timing_fields_internally_consistent": True,
        "allowed_development_total": len(allowed), "withheld_development_total": len(withheld),
        "withheld_label_perturbation_changed_nothing": True, "test_labelled_rows_rejected": True,
        "step_256_reproduces_prior_arm_decoder_state": True,
        "reserved_test_labels_evaluated": False, "withheld_development_evaluated": False,
        "checkpoint_promoted": False, "checkpoints": checks,
        "recomputed": "Exact endpoint distributions, per-site and per-block mass, category mass, conditional and "
                      "paired affinity, selections and portfolios, sampled diversity/KL/Hamming and their standard "
                      "errors, split and exposure counts, the training population, both schedules, the SFT exposure "
                      "replay and generator state, the development whitelist and every pinned artifact hash.",
        "not_recomputed": "Recorded metadata is read back, not re-derived: torch/CUDA versions, the "
                          "deterministic-algorithm flag, and the strict-reload, optimizer-carried-forward, "
                          "encoder-unchanged and RNG-invariance assertions, which were checked live during the run. "
                          "Timing fields are only checked for internal consistency. Training itself is not replayed.",
        "scope": "Recomputation agreement is numerical evidence only. Conditional affinity remains conditional on "
                 "measured identities, Monte Carlo KL and entropy remain estimates, and the endpoint normalization "
                 "check bounds float accumulation at the endpoints only, not at the intermediate checkpoints."}
    save_json(directory / "independent_audit.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/cr9114_budget_pressure_20260917")
    # Resolve first: a relative --directory must still be expressible under ROOT.
    audit(parser.parse_args().directory.resolve())
