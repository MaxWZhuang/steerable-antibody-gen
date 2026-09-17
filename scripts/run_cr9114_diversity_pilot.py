#!/usr/bin/env python
"""Matched SFT-to-DPO pilots with explicit on-policy sequence entropy control."""
from __future__ import annotations

import argparse
import gc
import json
import math
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
from smallAntibodyGen.experiments.diversity import negative_entropy_surrogate  # noqa: E402
from smallAntibodyGen.experiments.dpo import (  # noqa: E402
    dpo_per_pair_loss, genotype_digest, load_reference_cache, write_reference_cache,
)
from run_cr9114_dpo_pilot import load_inputs, score_sequences, training_schedule  # noqa: E402
from run_cr9114_esmif1_pilot import require, save_json, sha256, stable_subset, state_digest  # noqa: E402
from prepare_cr9114_preferences import construct_pairs  # noqa: E402
from diagnose_cr9114_dpo import cohort_metrics, diversity_metrics, pair_metrics  # noqa: E402


def fresh_cohort(records, excluded, count, seed):
    require(set(records.split) <= {"train", "development"}, "Test measurements in input")
    require(records.genotype.is_unique, "Duplicate record identities")
    result = stable_subset(records[(records.split == "development")
                                  & ~records.genotype.isin(excluded)], count, seed)
    require(len(result) == count and not set(result.genotype) & set(excluded), "Insufficient fresh development records")
    return result


def screen_against_sft(result, sft, retention):
    require(0 < retention <= 1, "Invalid diversity retention threshold")
    checks = {f"top_{k}_at_least_sft": result["development"]["top_k_mean"][k]
              >= sft["development"]["top_k_mean"][k] for k in ("16", "32")}
    checks.update({f"{metric}_retained": result["diversity"][metric] >= retention * sft["diversity"][metric]
                   for metric in ("unique_genotypes", "mean_pairwise_hamming_unbiased")})
    return {"checks": checks, "passes": all(checks.values()), "biological_significance_claim": False}


def sample_table(bound, count, seed, name):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    genotypes, values = [], []
    for start in range(0, count, 32):
        sampled = bound.policy.sample(bound.geometry, num_samples=min(32, count - start), generator=generator)
        genotypes.extend("".join(map(str, g)) for g in sampled.alleles)
        values.extend(sampled.log_probability.cpu().tolist())
        if (start + 32) % 256 == 0:
            print(f"{name}: evaluation samples {min(start + 32, count)}/{count}", flush=True)
    return pd.DataFrame({"genotype": genotypes, "log_q": values})


def evaluate(bound, records, pairs, sft_scores, config, destination, name):
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in records.genotype]
    values = score_sequences(bound.policy, bound.geometry, sequences, 16)
    scores = pd.Series(values, index=records.genotype)
    scores.rename("log_q").rename_axis("genotype").to_csv(destination / "development_scores.csv")
    samples = sample_table(bound, config["evaluation_samples"], config["evaluation_sample_seed"], name)
    rescore = score_sequences(bound.policy, bound.geometry,
        [bound.space.sequence_for(tuple(map(int, g))) for g in samples.genotype[:32]], 16)
    error = float(np.max(np.abs(rescore - samples.log_q[:32].to_numpy())))
    require(error < 1e-4, "Evaluation sample/rescore mismatch")
    samples.to_csv(destination / "samples.csv", index=False)
    return {"development": cohort_metrics(records, pairs, scores, sft_scores),
            "diversity": diversity_metrics(samples.genotype.tolist(), samples.log_q),
            "sample_rescore_max_error": error}, scores, samples


def run(config_path, output):
    require(not output.exists(), "Choose a fresh output directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit code and protocol before running")
    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-diversity-pilot/1" and config["beta"] == .1,
            "Unsupported configuration or evaluation beta")
    require(config["entropy_samples"] >= 2 and config["entropy_every"] >= 1
            and config["evaluation_samples"] >= 32 and config["steps"] % config["entropy_every"] == 0,
            "Invalid entropy schedule")
    require(len(set(config["seeds"])) == len(config["seeds"])
            and len(set(config["entropy_coefficients"])) == len(config["entropy_coefficients"])
            and all(np.isfinite(v) and v >= 0 for v in config["entropy_coefficients"]), "Invalid arms")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    require(torch.cuda.is_available(), "CUDA required")
    prior_config = json.loads((ROOT / "configs/experiments/cr9114_dpo_pilot.json").read_text())
    pilot, manifest, pairs, old_pairs, old_scores, genotypes = load_inputs(prior_config)
    record_path = ROOT / prior_config["preference_dir"] / "eligible_non_test_records.csv"
    require(sha256(record_path) == manifest["output_files"][record_path.name]["sha256"], "Measurement records changed")
    records = pd.read_csv(record_path, dtype={"genotype": "string"})
    diagnostic_path = ROOT / "outputs/cr9114_dpo_diagnostics_20260916/fresh_development.csv"
    diagnostic_evidence = json.loads((ROOT / "reference/evidence/cr9114-dpo-diagnostics-2026-09-16.json").read_text())["run"]
    require(sha256(diagnostic_path) == diagnostic_evidence["fresh_cohort_sha256"], "Earlier diagnostic cohort changed")
    excluded = set(old_scores.genotype) | set(pd.read_csv(diagnostic_path, dtype={"genotype": "string"}).genotype)
    fresh = fresh_cohort(records, excluded, config["fresh_development_count"], config["fresh_development_seed"])
    dev_pairs, pair_audit = construct_pairs(fresh, manifest["config"])
    require(not set(fresh.genotype) & set(genotypes), "Training preference/evaluation identity overlap")
    source_cache_path = ROOT / "outputs/cr9114_dpo_pilot_20260916/sft_dpo/reference_cache.json"
    source_cache_sha = "e1aea46a8b7e05bfa26b51a9293b0cd8837a05d45fc14f768fef7927344f07ac"
    require(sha256(source_cache_path) == source_cache_sha, "SFT reference cache changed")
    source_identity = json.loads(source_cache_path.read_text())["identity"]
    identity = source_identity.copy()
    source_ref_values = load_reference_cache(source_cache_path, source_identity, genotypes)
    output.mkdir(parents=True)
    fresh.to_csv(output / "fresh_development.csv", index=False)
    dev_pairs.to_csv(output / "development_pairs.csv", index=False)
    schedules = {}
    for seed in config["seeds"]:
        schedules[seed] = training_schedule(pairs, config["steps"], config["pairs_per_batch"], seed)
        np.save(output / f"schedule_{seed}.npy", schedules[seed], allow_pickle=False)
    results = {"status": "running", "config": config,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
               "fresh_cohort_sha256": sha256(output / "fresh_development.csv"),
               "fresh_pairs_sha256": sha256(output / "development_pairs.csv"), "fresh_pair_audit": pair_audit,
               "source_reference_cache_sha256": source_cache_sha, "source_reference_identity": source_identity,
               "schedule_sha256": {str(seed): sha256(output / f"schedule_{seed}.npy") for seed in config["seeds"]},
               "reserved_test_labels_evaluated": False, "checkpoint_promoted": False, "arms": {}}
    results["runtime"] = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0),
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    save_json(output / "results.json", results)
    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy
    install()
    import esm
    torch.set_num_threads(4)
    base = pilot["config"]
    prepared_path = ROOT / base["prepared_artifact"]
    require(sha256(prepared_path) == pilot["prepared_artifact_sha256"] == identity["prepared_artifact_sha256"], "Context changed")
    prepared = load_prepared_structure(prepared_path)
    require(not verify_against_source(prepared, ROOT / base["structure"]), "Invalid structure")
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    sft_path = ROOT / prior_config["sft_checkpoint"]
    require(sha256(weights) == base["weights_sha256"] and sha256(sft_path) == prior_config["sft_checkpoint_sha256"], "Weights changed")
    print("Loading released encoder and fixed SFT initialization", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    repeated = bind_policy(prepared, model, alphabet)
    require(bound.geometry.digest == repeated.geometry.digest, "Deterministic geometry encoding did not repeat")
    del repeated
    results["deterministic_geometry_repeat_verified"] = True
    checkpoint = torch.load(sft_path, map_location="cpu", weights_only=True)
    sft_state = checkpoint["decoder"]
    del checkpoint
    model.decoder.load_state_dict(sft_state, strict=True)
    require(state_digest(model.decoder) == identity["decoder_state_sha256"]
            and state_digest(model.encoder) == identity["encoder_state_sha256"]
            and sha256(ROOT / "src/smallAntibodyGen/models/esmif1_policy.py") == identity["policy_source_sha256"]
            and sha256(ROOT / "src/smallAntibodyGen/esmif1_compat.py") == identity["compat_source_sha256"],
            "Reference identity mismatch")
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in genotypes]
    idx = {g: i for i, g in enumerate(genotypes)}
    chosen_idx = pairs.chosen_genotype.map(idx).to_numpy(dtype=int)
    rejected_idx = pairs.rejected_genotype.map(idx).to_numpy(dtype=int)
    # GPU encoding bytes need not reproduce across processes. Bind the new cache
    # to THIS encoding and score every endpoint in BOTH fixed training schedules.
    scheduled_rows = np.concatenate([s.ravel() for s in schedules.values()])
    used = np.unique(np.concatenate([chosen_idx[scheduled_rows], rejected_idx[scheduled_rows]]))
    print(f"Rebuilding exact reference scores for {len(used)} scheduled variants", flush=True)
    fresh_values = score_sequences(bound.policy, bound.geometry, [sequences[i] for i in used], 16)
    np.testing.assert_allclose(fresh_values, source_ref_values[used], rtol=2e-5, atol=2e-4)
    used_genotypes = [genotypes[i] for i in used]
    identity.update(geometry_digest=bound.geometry.digest, genotype_order_sha256=genotype_digest(used_genotypes))
    cache_path = output / "reference_cache.json"
    write_reference_cache(cache_path, identity, used_genotypes, fresh_values)
    fresh_values = load_reference_cache(cache_path, identity, used_genotypes)
    cache_sha = sha256(cache_path)
    ref_values = np.full(len(genotypes), np.nan)
    ref_values[used] = fresh_values
    results.update(reference_identity=identity, reference_cache_sha256=cache_sha,
                   rebuilt_reference_variants=len(used),
                   source_reference_max_score_error=float(np.max(np.abs(fresh_values - source_ref_values[used]))))
    save_json(output / "results.json", results)
    old_sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in old_scores.genotype]
    initial_old = score_sequences(bound.policy, bound.geometry, old_sequences, 16)
    np.testing.assert_allclose(initial_old, old_scores.final_log_q, rtol=2e-5, atol=2e-4)
    baseline_dir = output / "sft"
    baseline_dir.mkdir()
    print("Evaluating SFT on the fixed fresh cohort", flush=True)
    baseline, baseline_scores, _ = evaluate(bound, fresh, dev_pairs, None, config, baseline_dir, "sft")
    results["sft"] = baseline
    train_records = records[records.split == "train"]
    def design(table):
        return np.array([[1., *map(float, g)] for g in table.genotype])
    x = design(train_records)
    penalty = np.eye(17)
    penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ train_records["mean"].to_numpy())
    results["additive_ridge"] = cohort_metrics(fresh, dev_pairs, pd.Series(design(fresh) @ coef, index=fresh.genotype))
    save_json(output / "results.json", results)

    for seed in config["seeds"]:
        for coefficient in config["entropy_coefficients"]:
            name = f"seed_{seed}_entropy_{coefficient:g}"
            directory = output / name
            directory.mkdir()
            model.zero_grad(set_to_none=True)
            model.decoder.load_state_dict(sft_state, strict=True)
            require(state_digest(model.decoder) == identity["decoder_state_sha256"], "Arm initialization drift")
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            history, regularization_rows = [], []
            started = time.perf_counter()
            print(f"{name}: starting fixed training schedule", flush=True)
            for step, indices in enumerate(schedules[seed], 1):
                c, r = chosen_idx[indices], rejected_idx[indices]
                optimizer.zero_grad(set_to_none=True)
                logq = bound.policy.log_prob([sequences[i] for i in np.concatenate([c, r])], bound.geometry)
                size = len(indices)
                ref_c = torch.tensor(ref_values[c], dtype=logq.dtype, device=logq.device)
                ref_r = torch.tensor(ref_values[r], dtype=logq.dtype, device=logq.device)
                loss = dpo_per_pair_loss(logq[:size], logq[size:], ref_c, ref_r, beta=config["beta"]).mean()
                require(bool(torch.isfinite(loss)), "Nonfinite preference loss")
                if step == 1:
                    require(abs(float(loss.detach()) - math.log(2)) < 1e-4, "Initial reference mismatch")
                loss.backward()
                entry = {"step": step, "dpo_loss": float(loss.detach())}
                if coefficient > 0 and step % config["entropy_every"] == 0:
                    sampled = bound.policy.sample(bound.geometry, num_samples=config["entropy_samples"], generator=generator)
                    rescored = bound.policy.log_prob(sampled.sequences, bound.geometry)
                    error = float((rescored.detach() - sampled.log_probability).abs().max())
                    require(error < 2e-4, "Training sample/rescore mismatch")
                    carrier = negative_entropy_surrogate(rescored)
                    (coefficient * config["entropy_every"] * carrier).backward()
                    entry.update(entropy_mc=float(-rescored.detach().mean()), entropy_rescore_max_error=error)
                    regularization_rows.extend({"step": step, "genotype": "".join(map(str, g)), "log_q": float(v)}
                        for g, v in zip(sampled.alleles, sampled.log_probability.cpu().tolist()))
                    del sampled, rescored, carrier
                require(all(p.grad is None for p in model.encoder.parameters()), "Encoder gradient detected")
                norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                require(float(norm) > 0, "Zero decoder gradient")
                optimizer.step()
                entry.update(gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - started)
                history.append(entry)
                if step == 1 or step % 16 == 0:
                    save_json(directory / "history.json", history)
                    print(f"{name}: step {step}/{config['steps']}, DPO={entry['dpo_loss']:.4f}, {entry['elapsed_seconds']:.1f}s", flush=True)
            training_seconds = time.perf_counter() - started
            peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20
            model.zero_grad(set_to_none=True)
            final_hash = state_digest(model.decoder)
            require(final_hash != identity["decoder_state_sha256"] and state_digest(model.encoder) == identity["encoder_state_sha256"], "Model state audit failed")
            destination = directory / f"decoder_step_{config['steps']:04d}.pt"
            torch.save({"schema_version": "cr9114-diversity-decoder/1", "decoder": model.decoder.state_dict(),
                        "optimizer": optimizer.state_dict(), "config": config, "seed": seed,
                        "entropy_coefficient": coefficient, "reference_identity": identity,
                        "reference_cache_sha256": cache_sha, "git_commit": results["git_commit"],
                        "schedule_sha256": results["schedule_sha256"][str(seed)],
                        "torch_rng_state": torch.get_rng_state(), "entropy_generator_state": generator.get_state()}, destination)
            del optimizer
            if regularization_rows:
                pd.DataFrame(regularization_rows).to_csv(directory / "entropy_training_samples.csv", index=False)
            before = score_sequences(bound.policy, bound.geometry, old_sequences[:16], 4)
            reloaded = torch.load(destination, map_location="cpu", weights_only=True)
            model.decoder.load_state_dict(reloaded["decoder"], strict=True)
            del reloaded
            after = score_sequences(bound.policy, bound.geometry, old_sequences[:16], 4)
            require(state_digest(model.decoder) == final_hash, "Reload tensor mismatch")
            np.testing.assert_allclose(before, after, rtol=0, atol=1e-5)
            result, _, samples = evaluate(bound, fresh, dev_pairs, baseline_scores, config, directory, name)
            result.update(seed=seed, entropy_coefficient=coefficient, training_seconds=training_seconds,
                          peak_cuda_allocated_mib=peak_mib, checkpoint_sha256=sha256(destination),
                          decoder_state_sha256=final_hash, safe_strict_reload=True,
                          reload_max_error=float(np.max(np.abs(before - after))), encoder_unchanged=True,
                          regularization_sample_count=len(regularization_rows),
                          unlabelled_regularization_unique_overlap_with_development=len(
                              set(row["genotype"] for row in regularization_rows) & set(fresh.genotype)))
            result["screen"] = screen_against_sft(result, baseline, config["diversity_retention_fraction"])
            if seed == 20260916 and coefficient == 0:
                prior = pd.read_csv(ROOT / "outputs/cr9114_dpo_pilot_20260916/sft_dpo/development_scores.csv", dtype={"genotype": "string"}).set_index("genotype")
                values = score_sequences(bound.policy, bound.geometry, old_sequences, 16)
                original = prior.loc[old_scores.genotype, "adapted_log_q"].to_numpy()
                result["historical_control_comparison"] = {
                    "max_log_q_difference": float(np.max(np.abs(values - original))),
                    "mean_absolute_log_q_difference": float(np.mean(np.abs(values - original))),
                    "score_rank_correlation": float(pd.Series(values).corr(pd.Series(original), method="spearman")),
                    "historical_pair_accuracy": pair_metrics(old_pairs, pd.Series(original, index=old_scores.genotype))["pair_accuracy"],
                    "retrained_pair_accuracy": pair_metrics(old_pairs, pd.Series(values, index=old_scores.genotype))["pair_accuracy"],
                    "note": "Historical training used nondeterministic GPU operations. Record drift; this is not a checkpoint reload or within-experiment matched-arm check."}
            model.decoder.load_state_dict(sft_state, strict=True)
            refs = score_sequences(bound.policy, bound.geometry,
                [bound.space.sequence_for(tuple(map(int, g))) for g in samples.genotype], 16)
            ratio = samples.log_q.to_numpy() - refs
            result["kl_to_sft_mc"] = {"nats": float(ratio.mean()), "standard_error": float(ratio.std(ddof=1) / np.sqrt(len(ratio)))}
            samples.assign(reference_log_q=refs).to_csv(directory / "samples.csv", index=False)
            save_json(directory / "result.json", result)
            results["arms"][name] = result
            save_json(output / "results.json", results)
            print(f"{name}: accuracy={result['development']['pair_accuracy']:.3%}, unique={result['diversity']['unique_genotypes']}, screen={result['screen']['passes']}", flush=True)
    require(sha256(cache_path) == cache_sha and sha256(source_cache_path) == source_cache_sha, "Reference cache changed")
    results["two_seed_screen"] = {str(coefficient): all(r["screen"]["passes"] for r in results["arms"].values()
        if r["entropy_coefficient"] == coefficient) for coefficient in config["entropy_coefficients"]}
    results["reference_cache_unchanged"] = True
    results["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                                 if p.is_file() and p.suffix in (".csv", ".npy")}
    results["status"] = "completed"
    save_json(output / "results.json", results)
    print("All six arms completed; reserved-test labels untouched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_diversity_pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cr9114_diversity_pilot_20260916")
    args = parser.parse_args()
    run(args.config, args.output)
