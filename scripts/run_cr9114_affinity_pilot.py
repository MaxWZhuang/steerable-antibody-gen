#!/usr/bin/env python
"""Fixed affinity-weighted likelihood comparison with independent dev measurements."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.affinity import affinity_population, likelihood_schedule  # noqa: E402
from smallAntibodyGen.experiments.diversity import negative_entropy_surrogate  # noqa: E402
from run_cr9114_dpo_pilot import load_inputs, score_sequences  # noqa: E402
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest  # noqa: E402
from run_cr9114_diversity_pilot import evaluate, fresh_cohort, screen_against_sft  # noqa: E402
from prepare_cr9114_preferences import construct_pairs  # noqa: E402


def selection_details(records, scores, baseline_scores):
    table = records.set_index("genotype").assign(score=scores, baseline_score=baseline_scores).reset_index()
    ordered = table.sort_values(["score", "genotype"], ascending=[False, True])
    baseline = table.sort_values(["baseline_score", "genotype"], ascending=[False, True])
    return {str(k): {
        "selected_genotypes": ordered.head(k).genotype.tolist(),
        "overlap_with_sft": len(set(ordered.head(k).genotype) & set(baseline.head(k).genotype)),
        "mean_minus_effective_sem": float((ordered.head(k)["mean"] - ordered.head(k).effective_sem).mean()),
        "per_block_top_k_mean": {str(int(b)): float(g.head(k)["mean"].mean()) for b, g in ordered.groupby("block")},
        "leave_one_block_out_top_k_mean": {str(int(b)): float(ordered[ordered.block != b].head(k)["mean"].mean())
                                            for b in sorted(table.block.unique())}}
        for k in (16, 32)}


def control_comparison(result, control):
    candidate = result["development"]["top_k_mean"]
    reference = control["development"]["top_k_mean"]
    checks = {f"top_{k}_at_least_control": candidate[k] >= reference[k] for k in ("16", "32")}
    checks["at_least_one_strict_improvement"] = any(candidate[k] > reference[k] for k in ("16", "32"))
    return {"checks": checks, "passes": all(checks.values())}


def run(config_path, output):
    require(not output.exists(), "Choose a fresh output directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit implementation and protocol before fitting")
    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-affinity-pilot/1", "Unsupported configuration")
    require(config["likelihood_normalizer"] == 16 and config["entropy_samples"] >= 2
            and config["entropy_every"] >= 1 and config["steps"] % config["entropy_every"] == 0
            and config["evaluation_samples"] >= 32, "Invalid objective or schedule")
    require([a["name"] for a in config["arms"]] == ["continued_sft_entropy", "affinity_entropy", "affinity_plain"]
            and [a["weighted"] for a in config["arms"]] == [False, True, True]
            and [a["entropy_coefficient"] for a in config["arms"]] == [.03, .03, 0.]
            and len(config["seeds"]) == len(set(config["seeds"])) == 2, "Expected six declared arms")
    require(config["deterministic_algorithms"] is True, "Deterministic operations required")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), "CUDA required")
    prior = json.loads((ROOT / "configs/experiments/cr9114_dpo_pilot.json").read_text())
    pilot, manifest, _, _, old_scores, _ = load_inputs(prior)
    record_path = ROOT / prior["preference_dir"] / "eligible_non_test_records.csv"
    require(sha256(record_path) == manifest["output_files"][record_path.name]["sha256"], "Measurement records changed")
    records = pd.read_csv(record_path, dtype={"genotype": "string"})
    require(set(records.split) == {"train", "development"}, "Test labels in input")
    population, population_audit = affinity_population(records[records.split == "train"],
        quantile=config["positive_quantile"], uncertainty_multiplier=config["uncertainty_multiplier"],
        max_weight_ratio=config["max_weight_ratio"], uniform_fraction=config["uniform_fraction"])
    positives = pd.read_csv(ROOT / prior["pilot_dir"] / "training_positives.csv", dtype={"genotype": "string"})
    require(set(population.genotype) == set(positives.genotype), "Original SFT-positive population changed")
    require(np.isclose(population_audit["pooled_sample_variance"], manifest["training_pooled_sample_variance"]),
            "Training uncertainty estimate changed")
    excluded = set(old_scores.genotype)
    earlier = [
        ("outputs/cr9114_dpo_diagnostics_20260916/fresh_development.csv", "reference/evidence/cr9114-dpo-diagnostics-2026-09-16.json"),
        ("outputs/cr9114_diversity_pilot_20260916_v3/fresh_development.csv", "reference/evidence/cr9114-diversity-pilot-2026-09-16.json")]
    for path, evidence in earlier:
        expected = json.loads((ROOT / evidence).read_text())["run"]["fresh_cohort_sha256"]
        require(sha256(ROOT / path) == expected, "Earlier evaluation cohort changed")
        excluded.update(pd.read_csv(ROOT / path, dtype={"genotype": "string"}).genotype)
    require(len(excluded) == 4608, "Unexpected previous evaluation population")
    fresh = fresh_cohort(records, excluded, config["fresh_development_count"], config["fresh_development_seed"])
    require(not set(fresh.genotype) & set(population.genotype), "Labelled train/dev overlap")
    pairs, pair_audit = construct_pairs(fresh, manifest["config"])
    output.mkdir(parents=True)
    population.to_csv(output / "training_population.csv", index=False)
    fresh.to_csv(output / "fresh_development.csv", index=False)
    pairs.to_csv(output / "development_pairs.csv", index=False)
    schedules = {}
    for seed in config["seeds"]:
        for weighted in (False, True):
            key = f"{seed}_{'weighted' if weighted else 'uniform'}"
            schedules[key] = likelihood_schedule(population, weighted=weighted, steps=config["steps"],
                batch_size=config["batch_size"], seed=seed)
            np.save(output / f"schedule_{key}.npy", schedules[key], allow_pickle=False)
    results = {"status": "running", "config": config, "population": population_audit,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
        "input_records_sha256": sha256(record_path), "fresh_cohort_sha256": sha256(output / "fresh_development.csv"),
        "fresh_pair_audit": pair_audit, "excluded_previous_development_count": len(excluded),
        "schedule_sha256": {key: sha256(output / f"schedule_{key}.npy") for key in schedules},
        "reserved_test_labels_evaluated": False, "checkpoint_promoted": False, "arms": {},
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "deterministic_algorithms": True}}
    save_json(output / "results.json", results)
    print(f"Frozen population: {population_audit}", flush=True)

    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy
    install()
    import esm
    base = pilot["config"]
    prepared_path = ROOT / base["prepared_artifact"]
    require(sha256(prepared_path) == pilot["prepared_artifact_sha256"], "Context changed")
    prepared = load_prepared_structure(prepared_path)
    require(not verify_against_source(prepared, ROOT / base["structure"]), "Structure source changed")
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    sft_path = ROOT / prior["sft_checkpoint"]
    require(sha256(weights) == base["weights_sha256"] and sha256(sft_path) == prior["sft_checkpoint_sha256"], "Weights changed")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    repeated = bind_policy(prepared, model, alphabet)
    require(bound.geometry.digest == repeated.geometry.digest, "Geometry encoding did not repeat")
    del repeated
    checkpoint = torch.load(sft_path, map_location="cpu", weights_only=True)
    sft_state = checkpoint["decoder"]
    del checkpoint
    model.decoder.load_state_dict(sft_state, strict=True)
    identity = {"decoder_state_sha256": state_digest(model.decoder), "encoder_state_sha256": state_digest(model.encoder),
                "geometry_digest": bound.geometry.digest, "prepared_artifact_sha256": sha256(prepared_path),
                "sft_checkpoint_sha256": sha256(sft_path), "weights_sha256": base["weights_sha256"]}
    results.update(initial_identity=identity, deterministic_geometry_repeat_verified=True)
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in population.genotype]
    baseline_dir = output / "sft"
    baseline_dir.mkdir()
    baseline, baseline_scores, _ = evaluate(bound, fresh, pairs, None, config, baseline_dir, "sft")
    baseline["selection_details"] = selection_details(fresh, baseline_scores, baseline_scores)
    results["sft"] = baseline
    save_json(output / "results.json", results)

    for seed in config["seeds"]:
        for arm in config["arms"]:
            name = f"seed_{seed}_{arm['name']}"
            directory = output / name
            directory.mkdir()
            schedule_key = f"{seed}_{'weighted' if arm['weighted'] else 'uniform'}"
            model.zero_grad(set_to_none=True)
            model.decoder.load_state_dict(sft_state, strict=True)
            require(state_digest(model.decoder) == identity["decoder_state_sha256"], "Initialization drift")
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            history, regularization_rows = [], []
            started = time.perf_counter()
            coefficient = arm["entropy_coefficient"]
            print(f"{name}: training", flush=True)
            for step, indices in enumerate(schedules[schedule_key], 1):
                optimizer.zero_grad(set_to_none=True)
                logq = bound.policy.log_prob([sequences[i] for i in indices], bound.geometry)
                # Sampling already implements the weights; do NOT weight this loss again.
                loss = -logq.mean() / config["likelihood_normalizer"]
                require(bool(torch.isfinite(loss)), "Nonfinite likelihood loss")
                loss.backward()
                entry = {"step": step, "nll_per_site": float(loss.detach())}
                if coefficient > 0 and step % config["entropy_every"] == 0:
                    sampled = bound.policy.sample(bound.geometry, num_samples=config["entropy_samples"], generator=generator)
                    rescored = bound.policy.log_prob(sampled.sequences, bound.geometry)
                    error = float((rescored.detach() - sampled.log_probability).abs().max())
                    require(error < 2e-4, "Training sample/rescore mismatch")
                    (coefficient * config["entropy_every"] * negative_entropy_surrogate(rescored)).backward()
                    entry.update(entropy_mc=float(-rescored.detach().mean()), entropy_rescore_max_error=error)
                    regularization_rows.extend({"step": step, "genotype": "".join(map(str, g)), "log_q": float(v)}
                        for g, v in zip(sampled.alleles, sampled.log_probability.cpu().tolist()))
                    del sampled, rescored
                require(all(p.grad is None for p in model.encoder.parameters()), "Encoder gradient detected")
                norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                require(float(norm) > 0, "Zero decoder gradient")
                optimizer.step()
                entry.update(gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - started)
                history.append(entry)
                if step == 1 or step % 16 == 0:
                    save_json(directory / "history.json", history)
                    print(f"{name}: {step}/{config['steps']}, NLL/site={entry['nll_per_site']:.4f}, {entry['elapsed_seconds']:.1f}s", flush=True)
            training_seconds = time.perf_counter() - started
            peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20
            save_json(directory / "history.json", history)
            model.zero_grad(set_to_none=True)
            final_hash = state_digest(model.decoder)
            require(final_hash != identity["decoder_state_sha256"] and state_digest(model.encoder) == identity["encoder_state_sha256"], "Model state audit failed")
            destination = directory / f"decoder_step_{config['steps']:04d}.pt"
            torch.save({"schema_version": "cr9114-affinity-decoder/1", "decoder": model.decoder.state_dict(),
                        "optimizer": optimizer.state_dict(), "config": config, "seed": seed, "arm": arm,
                        "initial_identity": identity, "git_commit": results["git_commit"],
                        "schedule_sha256": results["schedule_sha256"][schedule_key],
                        "torch_rng_state": torch.get_rng_state(), "entropy_generator_state": generator.get_state()}, destination)
            del optimizer
            if regularization_rows:
                pd.DataFrame(regularization_rows).to_csv(directory / "entropy_training_samples.csv", index=False)
            before = score_sequences(bound.policy, bound.geometry, sequences[:16], 4)
            reloaded = torch.load(destination, map_location="cpu", weights_only=True)
            model.decoder.load_state_dict(reloaded["decoder"], strict=True)
            del reloaded
            after = score_sequences(bound.policy, bound.geometry, sequences[:16], 4)
            require(state_digest(model.decoder) == final_hash, "Reload tensor mismatch")
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-5)
            result, scores, samples = evaluate(bound, fresh, pairs, None, config, directory, name)
            result.update(seed=seed, arm=arm, training_seconds=training_seconds, peak_cuda_allocated_mib=peak_mib,
                checkpoint_sha256=sha256(destination), decoder_state_sha256=final_hash, safe_strict_reload=True,
                reload_max_error=float(np.max(np.abs(before - after))), encoder_unchanged=True,
                selection_details=selection_details(fresh, scores, baseline_scores), schedule_key=schedule_key,
                regularization_sample_count=len(regularization_rows),
                unlabelled_regularization_unique_overlap_with_development=len(
                    set(row["genotype"] for row in regularization_rows) & set(fresh.genotype)))
            result["screen"] = screen_against_sft(result, baseline, config["diversity_retention_fraction"])
            if arm["weighted"]:
                control = results["arms"][f"seed_{seed}_continued_sft_entropy"]
                result["control_comparison"] = control_comparison(result, control)
            model.decoder.load_state_dict(sft_state, strict=True)
            refs = score_sequences(bound.policy, bound.geometry,
                [bound.space.sequence_for(tuple(map(int, g))) for g in samples.genotype], 16)
            ratio = samples.log_q.to_numpy() - refs
            result["kl_to_sft_mc"] = {"nats": float(ratio.mean()), "standard_error": float(ratio.std(ddof=1) / np.sqrt(len(ratio)))}
            samples.assign(reference_log_q=refs).to_csv(directory / "samples.csv", index=False)
            save_json(directory / "result.json", result)
            results["arms"][name] = result
            save_json(output / "results.json", results)
            print(f"{name}: topK={result['development']['top_k_mean']}, unique={result['diversity']['unique_genotypes']}, screen={result['screen']['passes']}", flush=True)
    results["affinity_entropy_two_seed_screen"] = all(
        results["arms"][f"seed_{seed}_affinity_entropy"]["screen"]["passes"] and
        results["arms"][f"seed_{seed}_affinity_entropy"]["control_comparison"]["passes"] for seed in config["seeds"])
    results["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
        if p.is_file() and p.suffix in (".csv", ".npy")}
    results["status"] = "completed"
    save_json(output / "results.json", results)
    print("All six arms completed; reserved-test labels untouched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_affinity_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cr9114_affinity_pilot_20260916")
    args = parser.parse_args()
    run(args.config, args.output)
