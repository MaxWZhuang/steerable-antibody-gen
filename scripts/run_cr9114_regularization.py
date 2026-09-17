#!/usr/bin/env python
"""Matched affinity-only/KL/KL+entropy/KL+embedding continuation experiments."""
from __future__ import annotations

import argparse
import copy
import gc
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
from smallAntibodyGen.experiments.diversity import negative_entropy_surrogate
from smallAntibodyGen.experiments.regularization import decoder_statistics, off_diagonal_cosine, reverse_kl_surrogate
from smallAntibodyGen.experiments.shortlist import portfolio_metrics, select_shortlist
from diagnose_cr9114_dpo import diversity_metrics
from run_cr9114_diversity_pilot import sample_table
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest
from run_cr9114_shortlist import load_non_test, load_sft


@torch.no_grad()
def score_and_embed(decoder, bound, genotypes, name):
    scores, features = [], []
    for start in range(0, len(genotypes), 16):
        sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in genotypes[start:start + 16]]
        values, z = decoder_statistics(decoder, bound, sequences)
        require(bool(torch.isfinite(values).all()), "Nonfinite scores")
        scores.extend(values.cpu().tolist())
        features.append(z.cpu().numpy())
        if len(scores) % 512 == 0:
            print(f"{name}: {len(scores)}/{len(genotypes)} scored", flush=True)
    return np.asarray(scores), np.concatenate(features)


def cosine_summary(features):
    # Float64 accumulation makes near-one similarities easier to compare.
    z = torch.tensor(features, dtype=torch.float64)
    return float(off_diagonal_cosine(z))


def evaluate(model, reference, bound, fresh, frozen_dev_features, config, shortlist, directory, name):
    ids = fresh.genotype.tolist()
    scores, live_features = score_and_embed(model.decoder, bound, ids, name)
    candidates = pd.DataFrame({"genotype": ids, "score": scores})
    candidates.to_csv(directory / "development_scores.csv", index=False)
    np.save(directory / "development_embeddings.npy", live_features, allow_pickle=False)
    multiplier = shortlist["calibration"]["selected_multiplier"]
    threshold = shortlist["calibration"]["quality_threshold"]
    selected = {mode: {str(k): select_shortlist(candidates, k, m) for k in (16, 32)}
                for mode, m in (("ordinary", 1), ("diverse", multiplier))}
    save_json(directory / "selections_before_evaluation.json", selected)
    index = {g: i for i, g in enumerate(ids)}
    portfolios = {}
    for mode, budgets in selected.items():
        portfolios[mode] = {}
        for k, chosen in budgets.items():
            metrics = portfolio_metrics(fresh, chosen, threshold)
            positions = [index[g] for g in chosen]
            metrics["frozen_sft_embedding_cosine"] = cosine_summary(frozen_dev_features[positions])
            metrics["live_embedding_cosine"] = cosine_summary(live_features[positions])
            portfolios[mode][k] = metrics
    samples = sample_table(bound, config["evaluation_samples"], config["evaluation_sample_seed"], name)
    ref_scores, ref_features = score_and_embed(reference, bound, samples.genotype.tolist(), name + " frozen sample features")
    sampled_scores, sampled_features = score_and_embed(model.decoder, bound, samples.genotype.tolist(), name + " live sample features")
    error = float(np.max(np.abs(sampled_scores - samples.log_q.to_numpy())))
    require(error < 1e-4, "Sampling/rescoring mismatch")
    ratio = samples.log_q.to_numpy() - ref_scores
    samples.assign(reference_log_q=ref_scores).to_csv(directory / "samples.csv", index=False)
    np.save(directory / "sample_frozen_embeddings.npy", ref_features, allow_pickle=False)
    np.save(directory / "sample_live_embeddings.npy", sampled_features, allow_pickle=False)
    return {"portfolios": portfolios, "diversity": diversity_metrics(samples.genotype.tolist(), samples.log_q),
        "kl_to_sft_mc": {"nats": float(ratio.mean()), "standard_error": float(ratio.std(ddof=1) / np.sqrt(len(ratio)))},
        "frozen_sft_embedding_cosine": cosine_summary(ref_features), "live_embedding_cosine": cosine_summary(sampled_features),
        "sample_rescore_max_error": error}


def run(config_path, output):
    require(not output.exists(), "Choose a fresh output directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit implementation first")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-regularization/1" and config["sequence_normalizer"] == 16, "Unsupported protocol")
    require((config["steps"], config["batch_size"], config["regularization_every"], config["regularization_samples"],
             config["evaluation_samples"], config["evaluation_sample_seed"])
            == (256, 4, 4, 8, 1024, 20260928), "Declared training/evaluation budget changed")
    require(config["seeds"] == [20260925, 20260926, 20260927]
            and [(a["name"], a["kl"], a["entropy"], a["embedding"]) for a in config["arms"]]
            == [("affinity", 0., 0., 0.), ("kl", .1, 0., 0.), ("kl_entropy", .1, .1, 0.), ("kl_embedding", .1, 0., .1)],
            "Declared arms changed")
    prior, pilot, records, record_path = load_non_test()
    shortlist_dir = ROOT / config["shortlist_directory"]
    shortlist = json.loads((shortlist_dir / "results.json").read_text())
    audit = json.loads((shortlist_dir / "independent_audit.json").read_text())
    evidence = json.loads((ROOT / "reference/evidence/cr9114-shortlist-2026-09-17.json").read_text())
    require(audit["passes"] and sha256(shortlist_dir / "results.json") == evidence["results_sha256"], "Unaudited shortlist input")
    for relative, digest in shortlist["output_sha256"].items():
        require(sha256(shortlist_dir / relative) == digest, "Shortlist artifact changed")
    fresh = pd.read_csv(shortlist_dir / "fresh_development.csv", dtype={"genotype": "string"})
    require(set(fresh.split) == {"development"}, "Evaluation crossed split")
    population, population_audit = affinity_population(records[records.split == "train"],
        quantile=config["positive_quantile"], uncertainty_multiplier=config["uncertainty_multiplier"],
        max_weight_ratio=config["max_weight_ratio"], uniform_fraction=config["uniform_fraction"])
    output.mkdir(parents=True)
    population.to_csv(output / "training_population.csv", index=False)
    fresh.to_csv(output / "development_records.csv", index=False)
    schedules = {}
    for seed in config["seeds"]:
        schedules[seed] = likelihood_schedule(population, weighted=True, steps=config["steps"], batch_size=config["batch_size"], seed=seed)
        np.save(output / f"schedule_{seed}.npy", schedules[seed], allow_pickle=False)
    model, bound, sft_state, identity = load_sft(prior, pilot)
    require(identity == shortlist["initial_identity"], "Initialization differs from shortlist study")
    reference = copy.deepcopy(model.decoder).eval().requires_grad_(False)
    require(state_digest(reference) == identity["decoder_state_sha256"], "Reference differs")
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in population.genotype]
    with torch.no_grad():
        expected = bound.policy.log_prob(sequences[:16], bound.geometry)
        actual, _ = decoder_statistics(reference, bound, sequences[:16])
        parity_error = float((expected - actual).abs().max())
        require(parity_error < 1e-5, "Feature/scoring path differs from native policy")
    ref_scores, frozen_dev_features = score_and_embed(reference, bound, fresh.genotype.tolist(), "SFT frozen development")
    prior_scores = pd.read_csv(shortlist_dir / "development_scores.csv", dtype={"genotype": "string"})
    require(prior_scores.genotype.tolist() == fresh.genotype.tolist(), "Development score order differs")
    np.testing.assert_allclose(ref_scores, prior_scores.score, atol=1e-5, rtol=0)
    np.save(output / "frozen_development_embeddings.npy", frozen_dev_features, allow_pickle=False)
    result = {"schema_version": "cr9114-regularization-result/1", "status": "running", "config": config,
        "git_commit": commit, "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
        "initial_identity": identity, "input_records_sha256": sha256(record_path), "population": population_audit,
        "shortlist_result_sha256": sha256(shortlist_dir / "results.json"), "development_is_reused": True,
        "reserved_test_labels_evaluated": False, "checkpoint_promoted": False, "native_feature_path_parity_max_error": parity_error,
        "schedule_sha256": {str(s): sha256(output / f"schedule_{s}.npy") for s in schedules}, "arms": {},
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "deterministic_algorithms": True}}
    baseline_dir = output / "sft"
    baseline_dir.mkdir()
    result["sft"] = evaluate(model, reference, bound, fresh, frozen_dev_features, config, shortlist, baseline_dir, "sft")
    save_json(output / "results.json", result)
    for seed in config["seeds"]:
        for arm in config["arms"]:
            name = f"seed_{seed}_{arm['name']}"
            directory = output / name
            directory.mkdir()
            model.zero_grad(set_to_none=True)
            model.decoder.load_state_dict(sft_state, strict=True)
            require(state_digest(model.decoder) == identity["decoder_state_sha256"], "Initialization drift")
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            history, regularization_rows = [], []
            print(f"{name}: training", flush=True)
            for step, indices in enumerate(schedules[seed], 1):
                optimizer.zero_grad(set_to_none=True)
                nll = -bound.policy.log_prob([sequences[i] for i in indices], bound.geometry).mean() / config["sequence_normalizer"]
                require(bool(torch.isfinite(nll)), "Nonfinite NLL")
                nll.backward()
                entry = {"step": step, "nll_per_site": float(nll.detach())}
                if arm["kl"] > 0 and step % config["regularization_every"] == 0:
                    sampled = bound.policy.sample(bound.geometry, num_samples=config["regularization_samples"], generator=generator)
                    with torch.no_grad():
                        ref_logq, ref_z = decoder_statistics(reference, bound, sampled.sequences)
                    logq, z = decoder_statistics(model.decoder, bound, sampled.sequences)
                    error = float((logq.detach() - sampled.log_probability).abs().max())
                    require(error < 2e-4, "Training sampling/rescoring mismatch")
                    penalty = arm["kl"] * reverse_kl_surrogate(logq, ref_logq) / config["sequence_normalizer"]
                    if arm["entropy"] > 0:
                        penalty = penalty + arm["entropy"] * negative_entropy_surrogate(logq) / config["sequence_normalizer"]
                    if arm["embedding"] > 0:
                        penalty = penalty + arm["embedding"] * off_diagonal_cosine(z)
                    (config["regularization_every"] * penalty).backward()
                    entry.update(kl_mc=float((logq.detach() - ref_logq).mean()), entropy_mc=float(-logq.detach().mean()),
                                 live_embedding_cosine=float(off_diagonal_cosine(z.detach())),
                                 frozen_embedding_cosine=float(off_diagonal_cosine(ref_z)), rescore_max_error=error)
                    regularization_rows.extend({"step": step, "genotype": "".join(map(str, g)), "log_q": q, "reference_log_q": ref}
                        for g, q, ref in zip(sampled.alleles, sampled.log_probability.cpu().tolist(), ref_logq.cpu().tolist()))
                    del sampled, logq, z, ref_logq, ref_z, penalty
                require(all(p.grad is None for p in model.encoder.parameters()) and all(p.grad is None for p in reference.parameters()), "Frozen parameter gradient")
                norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                require(float(norm) > 0, "Zero training gradient")
                optimizer.step()
                entry.update(gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - started)
                history.append(entry)
                if step == 1 or step % 32 == 0:
                    save_json(directory / "history.json", history)
                    print(f"{name}: {step}/{config['steps']}, {entry['elapsed_seconds']:.1f}s", flush=True)
            training_seconds = time.perf_counter() - started
            peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20
            save_json(directory / "history.json", history)
            model.zero_grad(set_to_none=True)
            final_hash = state_digest(model.decoder)
            require(final_hash != identity["decoder_state_sha256"] and state_digest(model.encoder) == identity["encoder_state_sha256"]
                    and state_digest(reference) == identity["decoder_state_sha256"], "Model/reference state audit failed")
            path = directory / f"decoder_step_{config['steps']:04d}.pt"
            torch.save({"schema_version": "cr9114-regularization-decoder/1", "decoder": model.decoder.state_dict(),
                        "optimizer": optimizer.state_dict(), "config": config, "seed": seed, "arm": arm,
                        "initial_identity": identity, "git_commit": commit, "schedule_sha256": result["schedule_sha256"][str(seed)],
                        "torch_rng_state": torch.get_rng_state(), "regularization_generator_state": generator.get_state()}, path)
            del optimizer
            restored = torch.load(path, map_location="cpu", weights_only=True)
            model.decoder.load_state_dict(restored["decoder"], strict=True)
            del restored
            require(state_digest(model.decoder) == final_hash, "Strict reload changed state")
            if regularization_rows:
                pd.DataFrame(regularization_rows).to_csv(directory / "regularization_samples.csv", index=False)
            measured = evaluate(model, reference, bound, fresh, frozen_dev_features, config, shortlist, directory, name)
            measured.update(seed=seed, arm=arm, training_seconds=training_seconds, peak_cuda_allocated_mib=peak_mib,
                checkpoint_sha256=sha256(path), decoder_state_sha256=final_hash, strict_reload_verified=True,
                encoder_and_reference_unchanged=True, schedule_sha256=result["schedule_sha256"][str(seed)],
                regularization_sample_count=len(regularization_rows),
                unlabelled_regularization_unique_development_overlap=len(set(r["genotype"] for r in regularization_rows) & set(fresh.genotype)))
            save_json(directory / "result.json", measured)
            result["arms"][name] = measured
            save_json(output / "results.json", result)
            print(f"{name}: completed; top16={measured['portfolios']['ordinary']['16']['mean_affinity']:.6f}, top32={measured['portfolios']['ordinary']['32']['mean_affinity']:.6f}", flush=True)
    result["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
        if p.is_file() and p.suffix in (".csv", ".npy")}
    result["status"] = "completed"
    save_json(output / "results.json", result)
    print("All twelve arms completed; no checkpoint promoted", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_regularization.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cr9114_regularization_20260917")
    args = parser.parse_args()
    run(args.config, args.output)
