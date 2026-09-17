#!/usr/bin/env python
"""Affinity-only 16,384-update continuation with exhaustive endpoint measurement."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.affinity import affinity_population, likelihood_schedule
from smallAntibodyGen.experiments.pressure import (
    affinity_difference, block_labels, category_table, development_whitelist, evaluation_records,
    exact_distribution, exposure_overlap, group_masses, label_free, lexicographic_support,
    monte_carlo_kl, probabilities, replay_exposures, sampled_affinity, site_marginals, weighted_affinity,
)
from smallAntibodyGen.experiments.shortlist import portfolio_metrics, select_shortlist
from diagnose_cr9114_dpo import diversity_metrics
from run_cr9114_diversity_pilot import sample_table
from run_cr9114_dpo_pilot import score_sequences
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest
from run_cr9114_shortlist import load_non_test, load_sft, prior_development_ids

EVALUATION_COLUMNS = ("genotype", "split", "block", "mean", "effective_sem")
# Default pandas float parsing is not a binary round trip: reading a saved score
# and writing it back can move the last bit, which then fails an exact audit
# comparison against the .npy source. Every CSV whose floats are model scores or
# are copied into a saved artifact is read with the round-trip parser instead.
SCORE_CSV = {"dtype": {"genotype": "string"}, "float_precision": "round_trip"}


def portable(mapping):
    """Recorded artifact keys carry the separator of the platform that wrote them."""
    return {key.replace("\\", "/"): value for key, value in mapping.items()}


def plain_identities(frame):
    """One identity dtype everywhere, so every weight join aligns by value."""
    return frame.assign(genotype=frame.genotype.astype(object))


def optimizer_digest(optimizer):
    """Identity of the live optimizer, so evaluation cannot silently perturb it."""
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for key in sorted(state["state"]):
        for field in sorted(state["state"][key]):
            value = state["state"][key][field]
            digest.update(f"{key}:{field}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes()
                          if torch.is_tensor(value) else repr(value).encode())
    return digest.hexdigest()


def optimizer_step_counts(optimizer):
    state = optimizer.state_dict()["state"]
    return {int(entry["step"]) for entry in state.values() if "step" in entry}


def sequences_for(bound, genotypes):
    return [bound.space.sequence_for(tuple(map(int, genotype))) for genotype in genotypes]


def exhaustive_metrics(log_q, ctx, config):
    """Exact distribution statistics over all 65,536 legal identities.

    The support mass is checked against one before any normalization, and only
    then is the exact probability vector used as the weight for conditional
    affinity. Normalization at an endpoint bounds float accumulation error
    there; it says nothing about intermediate checkpoints, which are never
    exhaustively scored.
    """
    p, _, _ = probabilities(log_q, config["exact_probability_tolerance"])
    document = exact_distribution(log_q, ctx["sft_log_q"], config["exact_probability_tolerance"])
    weights = pd.Series(p, index=ctx["support"])
    reference = ctx["sft_exact_weights"] if ctx["sft_exact_weights"] is not None else weights
    marginals = site_marginals(p)
    loci = config["split_defining_loci_0based"]
    document["site_allele_one_marginals"] = marginals
    document["split_defining_site_marginals"] = {str(i): marginals[i] for i in loci}
    document["other_site_marginals"] = {str(i): marginals[i] for i in range(len(marginals)) if i not in loci}
    document["block_mass"] = group_masses(p, ctx["block_labels"])
    document["split_mass"] = group_masses(p, ctx["assignments"].loc[ctx["support"]].to_numpy())
    document["category_mass"] = category_table(ctx["support"], ctx["assignments"], ctx["measured"],
                                               ctx["eligible_development"], p)
    document["seen_labelled_mass"] = {label: float(weights.loc[sorted(identities)].sum()) if identities else 0.
                                      for label, identities in ctx["seen"].items()}
    conditional, deltas, populations = {}, {}, ctx["conditional_populations"]
    for name, subset in populations.items():
        conditional[name] = weighted_affinity(subset, weights, ctx["threshold"])
        deltas[name] = affinity_difference(subset, weights, reference)
    document["conditional_affinity"] = conditional
    document["conditional_affinity_delta_from_sft"] = deltas
    document["measured_threshold_exceedance_mass"] = float(
        conditional["train"]["positive_mass"] + conditional["development"]["positive_mass"])
    return document, weights


@torch.no_grad()
def evaluate(bound, ctx, config, directory, name, step, exhaustive_log_q=None):
    """Label-blind scoring and sampling first; every assay join happens afterwards.

    Nothing above the ASSAY JOINS marker reads a measured affinity: the pool
    scores, the selections and the actual draws are computed, rescored and
    written to disk first. The precommitted plan requires every draw and every
    selected identity to be frozen on disk before the first join, so no later
    join can influence which candidates or draws were reported.
    """
    document = {"step": step, "exhaustive_support_scored": exhaustive_log_q is not None}
    pool_scores = score_sequences(bound.policy, bound.geometry, ctx["pool_sequences"],
                                  config["exhaustive_score_batch_size"],
                                  progress=directory / "pool_scoring_progress.json")
    candidates = pd.DataFrame({"genotype": ctx["pool"].genotype.tolist(), "score": pool_scores})
    label_free(candidates, ("genotype", "score")).to_csv(directory / "development_scores.csv", index=False)
    selected = {"ordinary": {str(k): select_shortlist(candidates, k) for k in config["budgets"]},
                "diverse": {str(k): select_shortlist(candidates, k, ctx["multiplier"]) for k in config["budgets"]}}
    save_json(directory / "selections_before_evaluation.json", selected)

    reuse = ctx["reuse_samples"].get(str(step))
    if reuse is None:
        draws = sample_table(bound, config["evaluation_samples"], config["evaluation_sample_seed"], name)
        document["sample_source"] = "generated"
    else:
        draws = pd.read_csv(reuse, **SCORE_CSV)[["genotype", "log_q"]].copy()
        document["sample_source"] = str(reuse.relative_to(ROOT)).replace("\\", "/")
    require(len(draws) == config["evaluation_samples"], "Wrong evaluation draw count")
    rescored = score_sequences(bound.policy, bound.geometry, sequences_for(bound, draws.genotype),
                               config["exhaustive_score_batch_size"])
    rescore_error = float(np.max(np.abs(rescored - draws.log_q.to_numpy())))
    require(rescore_error < config["sample_rescore_tolerance"], "Sampling/rescoring mismatch")
    draws = draws.assign(reference_log_q=ctx["sft_log_q"][[int(g, 2) for g in draws.genotype]])
    label_free(draws, ("genotype", "log_q", "reference_log_q")).to_csv(directory / "samples.csv", index=False)
    document["sample_rescore_max_error"] = rescore_error
    document["diversity"] = diversity_metrics(draws.genotype.tolist(), draws.log_q)
    document["kl_to_sft_mc"] = monte_carlo_kl(draws.log_q.to_numpy(), draws.reference_log_q.to_numpy())

    # ---- ASSAY JOINS. Selections and draws above are already on disk. ----
    reference_selected = ctx["sft_selected"] if ctx["sft_selected"] is not None else selected
    previous_selected = ctx["previous_selected"] if ctx["previous_selected"] is not None else selected
    portfolios = {}
    for mode, budgets in selected.items():
        portfolios[mode] = {}
        for k, chosen in budgets.items():
            metrics = portfolio_metrics(ctx["pool"], chosen, ctx["threshold"])
            against_sft, against_previous = set(reference_selected[mode][k]), set(previous_selected[mode][k])
            metrics.update(shared_with_sft=len(set(chosen) & against_sft),
                           swapped_from_sft=len(set(chosen) - against_sft),
                           identical_to_sft=set(chosen) == against_sft,
                           shared_with_previous_checkpoint=len(set(chosen) & against_previous))
            portfolios[mode][k] = metrics
    document["portfolios"] = portfolios
    pool_weights = pd.Series(np.exp(pool_scores), index=candidates.genotype)
    reference_pool = ctx["sft_pool_weights"] if ctx["sft_pool_weights"] is not None else pool_weights
    document["pool_conditional_affinity"] = weighted_affinity(ctx["pool"], pool_weights, ctx["threshold"])
    document["pool_conditional_delta_from_sft"] = affinity_difference(ctx["pool"], pool_weights, reference_pool)
    document["sampled_affinity"] = sampled_affinity(draws.genotype.tolist(), ctx["assignments"],
                                                    ctx["evaluation"], ctx["threshold"], ctx["seen"]["union"])
    document["sampled_draw_categories"] = category_table(draws.genotype.tolist(), ctx["assignments"],
                                                         ctx["measured"], ctx["eligible_development"])
    document["sampled_exposure_overlap"] = {label: exposure_overlap(draws.genotype.tolist(), identities)
                                            for label, identities in ctx["seen"].items()}
    exact_weights = None
    if exhaustive_log_q is not None:
        document["exhaustive"], exact_weights = exhaustive_metrics(exhaustive_log_q, ctx, config)
    return document, {"pool_weights": pool_weights, "selected": selected, "exact_weights": exact_weights}


def prepare(config, output):
    """Provenance, whitelist and schedule checks: everything gated before fitting."""
    evidence = json.loads((ROOT / config["prior_evidence"]).read_text())
    prior_dir = ROOT / config["prior_directory"]
    require(evidence["run_results_sha256"] == config["prior_results_sha256"]
            == sha256(prior_dir / "results.json"), "Prior affinity-only results are not the audited ones")
    prior = evidence["run"]
    prior_hashes = portable(prior["output_sha256"])
    consumed = [config["prior_schedule_artifact"], *config["reused_sample_artifacts"].values()]
    for relative in consumed:
        require(sha256(prior_dir / relative) == prior_hashes[relative], f"Reused prior artifact changed: {relative}")
    require(prior_hashes[config["prior_schedule_artifact"]] == config["prior_schedule_sha256"], "Schedule pin differs")
    for step, relative in config["reused_sample_artifacts"].items():
        require(prior_hashes[relative] == config["reused_sample_sha256"][step], "Reused sample pin differs")
    require(prior["arms"][config["prior_arm"]]["decoder_state_sha256"]
            == config["expected_step_256_decoder_state_sha256"], "Step-256 decoder pin differs from the audited arm")

    shortlist_evidence = json.loads((ROOT / config["shortlist_evidence"]).read_text())
    shortlist_dir = ROOT / config["shortlist_directory"]
    require(shortlist_evidence["results_sha256"] == config["shortlist_results_sha256"]
            == sha256(shortlist_dir / "results.json"), "Shortlist input is not the audited one")
    shortlist = json.loads((shortlist_dir / "results.json").read_text())
    for relative, digest in shortlist["output_sha256"].items():
        require(sha256(shortlist_dir / relative) == digest, f"Shortlist artifact changed: {relative}")

    prior_config, pilot, records, record_path = load_non_test()
    require(sha256(record_path) == prior["input_records_sha256"], "Measurement source differs from the prior arm")
    pool = plain_identities(pd.read_csv(shortlist_dir / "fresh_development.csv", **SCORE_CSV))
    require(set(pool.split) == {"development"} and len(pool) == config["fresh_development_count"], "Pool changed")
    previous, cohort_hashes = prior_development_ids()
    require(len(previous) == config["prior_development_count"], "Earlier development population changed")
    allowed = development_whitelist(previous, pool.genotype.tolist())
    require(len(allowed) == config["allowed_development_total"], "Allowed development population changed")

    pilot_dir = ROOT / config["pilot_directory"]
    require(sha256(pilot_dir / "split.csv") == config["split_csv_sha256"], "Split assignment changed")
    split = pd.read_csv(pilot_dir / "split.csv", dtype={"genotype": "string"})[["genotype", "split"]]
    require(len(split) == config["legal_identity_count"] and split.genotype.is_unique, "Incomplete split table")
    support = lexicographic_support()
    require(set(support) == set(split.genotype), "Support differs from the assigned identity set")
    assignments = plain_identities(split).set_index("genotype").split
    eligible_development = set(records.loc[records.split == "development", "genotype"])
    require(len(eligible_development) == config["eligible_development_count"], "Eligible development count changed")
    evaluation = plain_identities(evaluation_records(records, allowed)[list(EVALUATION_COLUMNS)])
    withheld = sorted(eligible_development - set(allowed))
    require(len(withheld) == config["withheld_development_count"]
            and not set(withheld) & set(evaluation.genotype), "Withheld development labels reached evaluation")
    require(int((evaluation.split == "train").sum()) == config["eligible_training_count"]
            and int((evaluation.split == "development").sum()) == config["allowed_development_total"]
            and len(evaluation) == config["eligible_training_count"] + config["allowed_development_total"],
            "Evaluation population is not the declared one")
    require(assignments.loc[evaluation.genotype].tolist() == evaluation.split.tolist(), "Split assignment disagrees")
    require(block_labels(evaluation.genotype, config["split_defining_loci_0based"]) == evaluation.block.tolist(),
            "Split-defining loci changed")

    population, population_audit = affinity_population(records[records.split == "train"],
        quantile=config["positive_quantile"], uncertainty_multiplier=config["uncertainty_multiplier"],
        max_weight_ratio=config["max_weight_ratio"], uniform_fraction=config["uniform_fraction"])
    require(population_audit == prior["population"], "Training population differs from the prior arm")
    require(len(population) == config["training_population_count"], "Training population size changed")
    population = plain_identities(population)
    threshold = population_audit["mean_threshold"]
    schedule = likelihood_schedule(population, weighted=True, steps=config["steps"],
                                   batch_size=config["batch_size"], seed=config["seed"])
    prefix = np.load(prior_dir / config["prior_schedule_artifact"], allow_pickle=False)
    require(prefix.shape == (config["checkpoints"][0], config["batch_size"])
            and np.array_equal(schedule[:len(prefix)], prefix), "Schedule prefix differs from the prior arm")

    positives_path = pilot_dir / config["training_positives_artifact"]
    require(sha256(positives_path) == config["training_positives_sha256"], "Original SFT positives changed")
    positives = pd.read_csv(positives_path, dtype={"genotype": "string"})
    ordered = split[split.genotype.isin(set(positives.genotype))].genotype.tolist()
    require(positives.genotype.tolist() == ordered, "Original positives are not in pinned split-file order")
    require(set(positives.genotype) == set(population.genotype), "Original positives differ from the training pool")
    exposures, rng_state = replay_exposures(positives.genotype.tolist(), config["sft_exposure_seed"],
                                            config["sft_exposure_steps"], config["sft_exposure_batch_size"])
    flat = [genotype for row in exposures for genotype in row]
    require(len(flat) == config["expected_sft_exposure_draws"]
            and len(set(flat)) == config["expected_sft_exposure_unique"], "Original SFT exposure replay changed")
    sft_path = ROOT / prior_config["sft_checkpoint"]
    require(sha256(sft_path) == prior_config["sft_checkpoint_sha256"], "Portable SFT checkpoint changed")
    checkpoint = torch.load(sft_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("numpy_rng_state") == rng_state,
            "Original SFT exposure replay does not match the checkpoint's recorded generator state")
    del checkpoint
    continuation = population.genotype.to_numpy()[schedule]
    # Nothing is written until every read-only gate above has passed.
    output.mkdir(parents=True)
    np.save(output / "schedule.npy", schedule, allow_pickle=False)
    population.to_csv(output / "training_population.csv", index=False)
    pool.to_csv(output / "development_records.csv", index=False)
    pd.Series(sorted(allowed), name="genotype").to_csv(output / "allowed_development.csv", index=False)
    pd.DataFrame({"step": np.repeat(np.arange(1, config["sft_exposure_steps"] + 1), config["sft_exposure_batch_size"]),
                  "genotype": flat}).to_csv(output / "sft_exposures.csv", index=False)
    pd.DataFrame({"step": np.repeat(np.arange(1, config["steps"] + 1), config["batch_size"]),
                  "genotype": continuation.ravel()}).to_csv(output / "continuation_exposures.csv", index=False)
    return {"prior": prior, "prior_config": prior_config, "pilot": pilot, "shortlist": shortlist, "pool": pool,
            "record_path": record_path, "allowed": allowed, "withheld": withheld, "support": support,
            "assignments": assignments, "evaluation": evaluation, "eligible_development": eligible_development,
            "population": population, "population_audit": population_audit, "threshold": threshold,
            "schedule": schedule, "continuation": continuation, "sft_exposures": flat,
            "cohort_hashes": cohort_hashes, "shortlist_dir": shortlist_dir, "prior_dir": prior_dir}


def run(config_path, output):
    require(not output.exists(), "Choose a fresh output directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit implementation and protocol first")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-budget-pressure/1", "Unsupported protocol")
    require((config["steps"], config["batch_size"], config["sequence_normalizer"], config["seed"],
             tuple(config["checkpoints"]), tuple(config["exhaustive_checkpoints"]),
             config["evaluation_samples"], config["evaluation_sample_seed"])
            == (16384, 4, 16, 20260925, (256, 2048, 8192, 16384), (0, 16384), 1024, 20260928),
            "Declared training/evaluation budget changed")
    require((config["learning_rate"], config["weight_decay"], config["gradient_clip"]) == (1e-5, .01, 1.),
            "Declared optimization settings changed")
    require(config["kl_coefficient"] == config["entropy_coefficient"] == config["embedding_coefficient"] == 0.,
            "Budget is the only intervention; explicit regularizers must stay disabled")
    setup = prepare(config, output)
    prior, pool, population = setup["prior"], setup["pool"], setup["population"]

    model, bound, sft_state, identity = load_sft(setup["prior_config"], setup["pilot"])
    require(identity == prior["initial_identity"], "Initialization differs from the prior affinity-only arm")
    support_sequences = sequences_for(bound, setup["support"])
    roundtrip = all(bound.space.alleles_for(sequence) == tuple(map(int, genotype))
                    for sequence, genotype in zip(support_sequences, setup["support"]))
    require(roundtrip, "Genotype/sequence round trip failed on the legal support")
    population_sequences = [support_sequences[int(g, 2)] for g in population.genotype]
    pool_sequences = [support_sequences[int(g, 2)] for g in pool.genotype]
    ctx = {"pool": pool, "pool_sequences": pool_sequences, "support": setup["support"],
           "multiplier": setup["shortlist"]["calibration"]["selected_multiplier"],
           "threshold": setup["shortlist"]["calibration"]["quality_threshold"],
           "assignments": setup["assignments"], "evaluation": setup["evaluation"],
           "measured": set(setup["evaluation"].genotype), "eligible_development": setup["eligible_development"],
           "block_labels": np.asarray(block_labels(setup["support"], config["split_defining_loci_0based"])),
           "reuse_samples": {step: setup["prior_dir"] / relative
                             for step, relative in config["reused_sample_artifacts"].items()},
           "sft_log_q": None, "sft_exact_weights": None, "sft_pool_weights": None,
           "sft_selected": None, "previous_selected": None,
           "seen": {"sft": set(setup["sft_exposures"]), "continuation": set(), "union": set(setup["sft_exposures"])}}
    ctx["conditional_populations"] = {
        "train": setup["evaluation"][setup["evaluation"].split == "train"],
        "development": setup["evaluation"][setup["evaluation"].split == "development"],
        **{f"development_block_{block}": setup["evaluation"][(setup["evaluation"].split == "development")
                                                             & (setup["evaluation"].block == block)]
           for block in config["development_blocks"]}}
    require(all(len(subset) for subset in ctx["conditional_populations"].values()), "Empty conditional population")
    require(ctx["threshold"] == setup["threshold"], "Quality threshold differs from the training-only threshold")

    baseline = output / "sft"
    baseline.mkdir()
    print("Scoring the full 65,536-identity support under SFT", flush=True)
    sft_log_q = score_sequences(bound.policy, bound.geometry, support_sequences,
                                config["exhaustive_score_batch_size"],
                                progress=baseline / "exhaustive_progress.json")
    np.save(baseline / "exhaustive_log_q.npy", sft_log_q, allow_pickle=False)
    canary = score_sequences(bound.policy, bound.geometry, support_sequences[:16],
                             config["exhaustive_score_batch_size"])
    canary_error = float(np.max(np.abs(canary - sft_log_q[:16])))
    require(canary_error < 1e-6, "Repeated scoring of the same batch disagreed")
    _, _, support_mass = probabilities(sft_log_q, config["exact_probability_tolerance"])
    ctx["sft_log_q"] = sft_log_q
    pinned = pd.read_csv(setup["shortlist_dir"] / "development_scores.csv", **SCORE_CSV)
    require(pinned.genotype.tolist() == pool.genotype.tolist(), "Development score order differs")
    exhaustive_pool = sft_log_q[[int(g, 2) for g in pool.genotype]]
    np.testing.assert_allclose(exhaustive_pool, pinned.score, atol=config["pool_score_tolerance"], rtol=0)

    result = {"schema_version": "cr9114-budget-pressure-result/1", "status": "running", "config": config,
        "git_commit": commit, "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
        "initial_identity": identity, "input_records_sha256": sha256(setup["record_path"]),
        "prior_results_sha256": config["prior_results_sha256"], "prior_cohort_sha256": setup["cohort_hashes"],
        "shortlist_result_sha256": config["shortlist_results_sha256"], "population": setup["population_audit"],
        "training_population_max_measured_affinity": float(
            setup["evaluation"].loc[setup["evaluation"].split == "train", "mean"].max()),
        "schedule_sha256": sha256(output / "schedule.npy"), "schedule_prefix_matches_prior_arm": True,
        "expected_step_256_decoder_state_sha256": config["expected_step_256_decoder_state_sha256"],
        "allowed_development_count": len(setup["allowed"]), "withheld_development_count": len(setup["withheld"]),
        "original_sft_exposure_draws": len(setup["sft_exposures"]),
        "original_sft_exposure_unique": len(set(setup["sft_exposures"])),
        "original_sft_exposure_rng_state_verified": True,
        "genotype_sequence_round_trip_verified": True,
        "sft_support_mass_before_normalization": support_mass,
        "sft_repeat_scoring_max_error": canary_error,
        "sft_pool_vs_pinned_shortlist_max_error": float(np.max(np.abs(exhaustive_pool - pinned.score.to_numpy()))),
        "reserved_test_labels_evaluated": False, "withheld_development_evaluated": False,
        "checkpoint_promoted": False, "development_is_reused": True, "checkpoints": {},
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "deterministic_algorithms": True}}
    save_json(output / "results.json", result)

    measured, extras = evaluate(bound, ctx, config, baseline, "sft", 0, sft_log_q)
    measured["exposure_counts"] = {"sft_unique": len(ctx["seen"]["sft"]), "continuation_unique": 0,
                                   "union_unique": len(ctx["seen"]["union"])}
    ctx.update(sft_exact_weights=extras["exact_weights"], sft_pool_weights=extras["pool_weights"],
               sft_selected=extras["selected"], previous_selected=extras["selected"])
    result["checkpoints"]["0"] = measured
    save_json(baseline / "result.json", measured)
    save_json(output / "results.json", result)
    print(f"SFT evaluated; support mass {support_mass:.9f}", flush=True)

    model.zero_grad(set_to_none=True)
    model.decoder.load_state_dict(sft_state, strict=True)
    require(state_digest(model.decoder) == identity["decoder_state_sha256"], "Initialization drift")
    torch.manual_seed(config["seed"])
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    checkpoints = set(config["checkpoints"])
    # training_seconds accumulates only the optimizer loop. Intermediate sampling,
    # exhaustive scoring and checkpoint I/O are wall clock but not training.
    history, started, training_seconds = [], time.perf_counter(), 0.
    print(f"Training {config['steps']} updates on {config['steps'] * config['batch_size']} labelled exposures",
          flush=True)
    for step, indices in enumerate(setup["schedule"], 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        nll = -bound.policy.log_prob([population_sequences[i] for i in indices],
                                     bound.geometry).mean() / config["sequence_normalizer"]
        require(bool(torch.isfinite(nll)), "Nonfinite NLL")
        nll.backward()
        require(all(p.grad is None for p in model.encoder.parameters()), "Frozen parameter gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"],
                                              error_if_nonfinite=True)
        require(float(norm) > 0, "Zero training gradient")
        optimizer.step()
        # CUDA launches are asynchronous, so take the logged host transfers before stopping the
        # timer: they synchronize the stream and keep trailing optimizer work inside training_seconds.
        nll_per_site, gradient_norm = float(nll.detach()), float(norm)
        training_seconds += time.perf_counter() - step_started
        history.append({"step": step, "nll_per_site": nll_per_site, "gradient_norm": gradient_norm,
                        "training_seconds": training_seconds,
                        "elapsed_seconds": time.perf_counter() - started})
        if step == 1 or step % config["history_flush_every"] == 0 or step in checkpoints:
            save_json(output / "history.json", history)
            print(f"step {step}/{config['steps']}, nll/site={history[-1]['nll_per_site']:.6f}, "
                  f"{history[-1]['elapsed_seconds']:.1f}s", flush=True)
        if step not in checkpoints:
            continue
        model.zero_grad(set_to_none=True)
        digest = state_digest(model.decoder)
        if step == config["checkpoints"][0]:
            require(digest == config["expected_step_256_decoder_state_sha256"],
                    "Step-256 decoder state does not reproduce the prior affinity-only arm")
        require(optimizer_step_counts(optimizer) == {step}, "Optimizer step counter disagrees with the loop")
        require(all(bool(torch.isfinite(tensor).all()) for tensor in model.decoder.state_dict().values()
                    if tensor.is_floating_point()), "Nonfinite decoder parameter")
        directory = output / f"step_{step:05d}"
        directory.mkdir()
        path = directory / f"decoder_step_{step:05d}.pt"
        optimizer_before = optimizer_digest(optimizer)
        torch.save({"schema_version": "cr9114-budget-pressure-decoder/1", "step": step,
                    "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "seed": config["seed"], "initial_identity": identity, "git_commit": commit,
                    "schedule_sha256": result["schedule_sha256"], "torch_rng_state": torch.get_rng_state()}, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        model.decoder.load_state_dict(restored["decoder"], strict=True)
        del restored
        # load_state_dict copies in place, so the live optimizer keeps the same Parameters.
        require(state_digest(model.decoder) == digest and optimizer_digest(optimizer) == optimizer_before,
                "Strict reload changed decoder or optimizer state")
        require([id(p) for p in optimizer.param_groups[0]["params"]]
                == [id(p) for p in model.decoder.parameters()],
                "Reload replaced decoder parameters and detached the optimizer")
        before = (digest, optimizer_before, torch.get_rng_state(), torch.cuda.get_rng_state())
        ctx["seen"]["continuation"] = set(setup["continuation"][:step].ravel().tolist())
        ctx["seen"]["union"] = ctx["seen"]["sft"] | ctx["seen"]["continuation"]
        endpoint_log_q = None
        if step in config["exhaustive_checkpoints"]:
            print(f"step {step}: scoring the full 65,536-identity support", flush=True)
            endpoint_log_q = score_sequences(bound.policy, bound.geometry, support_sequences,
                                             config["exhaustive_score_batch_size"],
                                             progress=directory / "exhaustive_progress.json")
            np.save(directory / "exhaustive_log_q.npy", endpoint_log_q, allow_pickle=False)
        measured, extras = evaluate(bound, ctx, config, directory, f"step_{step:05d}", step, endpoint_log_q)
        require(state_digest(model.decoder) == before[0] and optimizer_digest(optimizer) == before[1]
                and torch.equal(torch.get_rng_state(), before[2])
                and torch.equal(torch.cuda.get_rng_state(), before[3]), "Evaluation mutated training state")
        require(state_digest(model.encoder) == identity["encoder_state_sha256"], "Frozen encoder changed")
        ctx["previous_selected"] = extras["selected"]
        elapsed = time.perf_counter() - started
        measured.update(decoder_state_sha256=digest, checkpoint_sha256=sha256(path), strict_reload_verified=True,
            encoder_unchanged=True, optimizer_carried_forward=True, training_seconds=training_seconds,
            wall_clock_seconds=elapsed, evaluation_and_io_seconds=elapsed - training_seconds,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
            exposure_counts={"sft_unique": len(ctx["seen"]["sft"]),
                             "continuation_unique": len(ctx["seen"]["continuation"]),
                             "union_unique": len(ctx["seen"]["union"]),
                             "continuation_draws": step * config["batch_size"],
                             "sft_continuation_shared_unique": len(ctx["seen"]["sft"] & ctx["seen"]["continuation"])})
        save_json(directory / "result.json", measured)
        result["checkpoints"][str(step)] = measured
        save_json(output / "results.json", result)
        print(f"step {step}: evaluated, decoder {digest[:12]}", flush=True)
    save_json(output / "history.json", history)
    require([entry["step"] for entry in history] == list(range(1, config["steps"] + 1)), "Training step gap")
    require(sorted(int(k) for k in result["checkpoints"]) == [0, *config["checkpoints"]], "Missing checkpoint")
    result["training_seconds"] = training_seconds
    result["wall_clock_seconds"] = time.perf_counter() - started
    result["evaluation_and_io_seconds"] = result["wall_clock_seconds"] - training_seconds
    result["timing_note"] = ("training_seconds is the optimizer loop only. wall_clock_seconds additionally covers "
                             "intermediate sampling, rescoring, exhaustive endpoint scoring and checkpoint I/O; the "
                             "prior arm reported a training-only number measured over a different budget, so neither "
                             "field is a like-for-like throughput comparison against it.")
    result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    # results.json is excluded: it is the document that carries these hashes.
    result["output_sha256"] = {str(p.relative_to(output)).replace("\\", "/"): sha256(p)
                               for p in sorted(output.rglob("*"))
                               if p.is_file() and p.name != "results.json"
                               and p.suffix in (".csv", ".npy", ".pt", ".json")}
    result["status"] = "completed"
    save_json(output / "results.json", result)
    print("Completed; no checkpoint selected or promoted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_budget_pressure.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cr9114_budget_pressure_20260917")
    args = parser.parse_args()
    run(args.config, args.output)
