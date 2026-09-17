#!/usr/bin/env python
"""Cache fixed references and run matched direct-DPO and SFT-to-DPO pilots."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.dpo import (  # noqa: E402
    dpo_per_pair_loss, genotype_digest, load_reference_cache, write_reference_cache,
)
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest  # noqa: E402
from prepare_cr9114_preferences import evaluate_pairs  # noqa: E402


def training_schedule(pairs, steps, batch_size, seed):
    require(set(pairs.split) == {"train"}, "DPO schedule must use training pairs only")
    weights = pairs.pair_weight.to_numpy(dtype=float)
    require(np.isfinite(weights).all() and (weights > 0).all() and np.isclose(weights.sum(), 1),
            "Training pair weights must be positive, finite and normalized")
    require(isinstance(steps, int) and steps > 0 and isinstance(batch_size, int) and batch_size > 0,
            "Invalid DPO schedule shape")
    return np.random.default_rng(seed).choice(len(pairs), size=(steps, batch_size),
                                             replace=True, p=weights / weights.sum())


def load_inputs(config):
    pilot_dir, pair_dir = ROOT / config["pilot_dir"], ROOT / config["preference_dir"]
    require(sha256(pair_dir / "manifest.json") == config["preference_manifest_sha256"], "Preference manifest changed")
    pair_manifest = json.loads((pair_dir / "manifest.json").read_text())
    require(pair_manifest["reserved_test_evaluated"] is False and pair_manifest["split_isolation_verified"] is True,
            "Preference manifest lacks split isolation")
    for name in ("training_pairs.csv", "development_pairs.csv"):
        require(sha256(pair_dir / name) == pair_manifest["output_files"][name]["sha256"], f"Changed {name}")
    for name in ("run.json", "split.csv", "development_scores.csv"):
        require(sha256(pilot_dir / name) == pair_manifest["input_files"][name], f"Changed pilot {name}")
    dtypes = {"chosen_genotype": "string", "rejected_genotype": "string"}
    train = pd.read_csv(pair_dir / "training_pairs.csv", dtype=dtypes)
    dev_pairs = pd.read_csv(pair_dir / "development_pairs.csv", dtype=dtypes)
    scores = pd.read_csv(pilot_dir / "development_scores.csv", dtype={"genotype": "string"})
    split = pd.read_csv(pilot_dir / "split.csv", dtype={"genotype": "string"}).set_index("genotype").split
    for pairs, expected in ((train, "train"), (dev_pairs, "development")):
        require(set(pairs.split) == {expected}, "Cross-split pair table")
        for column in ("chosen_genotype", "rejected_genotype"):
            require(set(split.loc[pairs[column]]) == {expected}, "Pair endpoint crosses its declared split")
    require(scores.genotype.is_unique and set(split.loc[scores.genotype]) == {"development"}, "Invalid development scores")
    genotypes = sorted(set(train.chosen_genotype) | set(train.rejected_genotype))
    require(not set(genotypes).intersection(scores.genotype), "Reference cache would include development variants")
    run_record = json.loads((pilot_dir / "run.json").read_text())
    require(run_record["prepared_artifact_sha256"] == pair_manifest["prepared_artifact_sha256"],
            "Pilot and preferences use different structural contexts")
    return run_record, pair_manifest, train, dev_pairs, scores, genotypes


def score_sequences(policy, geometry, sequences, batch_size, *, progress=None):
    values = []
    started = time.perf_counter()
    last_report = started
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            scores = policy.log_prob(sequences[start:start + batch_size], geometry)
            require(bool(torch.isfinite(scores).all()), "Nonfinite model score")
            values.extend(scores.cpu().tolist())
            now = time.perf_counter()
            if progress is not None and (now - last_report >= 20 or len(values) == len(sequences)):
                entry = {"phase": "reference_scoring", "scored": len(values), "total": len(sequences),
                         "elapsed_seconds": now - started}
                save_json(progress, entry)
                print(f"Reference scores {len(values)}/{len(sequences)} ({now - started:.1f}s)", flush=True)
                last_report = now
    return np.array(values)


def development_evaluation(scores, dev_pairs, settings, initial, adapted):
    evaluation = scores.copy()
    evaluation["parent_log_q"], evaluation["final_log_q"] = initial, adapted
    metrics, pair_scores = evaluate_pairs(dev_pairs, evaluation, settings)
    from scipy.stats import spearmanr
    result = {"initial_policy": metrics["parent"], "adapted_policy": metrics["sft"],
              "additive_ridge": metrics["additive_ridge"],
              "adapted_minus_initial_pp": metrics["sft_minus_parent_pp"],
              "adapted_minus_additive_pp": metrics["sft_minus_additive_pp"],
              "spearman_initial": float(spearmanr(initial, scores.h1_mean_recomputed).statistic),
              "spearman_adapted": float(spearmanr(adapted, scores.h1_mean_recomputed).statistic),
              "measured_top_k": {}, "uncertainty_note": metrics["uncertainty_note"]}
    for k, entry in metrics["measured_top_k"].items():
        result["measured_top_k"][k] = {"same_pool_oracle_mean": entry["same_pool_oracle_mean"],
            "initial_policy": entry["models"]["parent"], "adapted_policy": entry["models"]["sft"],
            "additive_ridge": entry["models"]["additive_ridge"]}
    return result, pair_scores.rename(columns={"parent_correct": "initial_correct", "sft_correct": "adapted_correct"})


def run(config_path, output):
    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy

    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-dpo-pilot/1", "Unsupported DPO configuration")
    require(config["arms"] == ["direct_dpo", "sft_dpo"], "Expected the two matched DPO arms")
    require(config["device"] == "cuda" and torch.cuda.is_available() and config["precision"] == "float32",
            "This pilot requires CUDA and float32")
    require(config["checkpoint_every"] > 0 and config["score_batch_size"] > 0
            and 0 < config["beta"] < float("inf") and config["learning_rate"] > 0, "Invalid pilot settings")
    require(not output.exists(), "Output exists; choose a fresh directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit the implementation and protocol before running")
    pilot_run, pair_manifest, pairs, dev_pairs, dev_scores, genotypes = load_inputs(config)
    base = pilot_run["config"]
    prepared_path = ROOT / base["prepared_artifact"]
    require(sha256(prepared_path) == pilot_run["prepared_artifact_sha256"], "Prepared artifact changed")
    prepared = load_prepared_structure(prepared_path)
    require(not verify_against_source(prepared, ROOT / base["structure"]), "Structure verification failed")
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    require(sha256(weights) == base["weights_sha256"], "Released checkpoint changed")
    sft_path = ROOT / config["sft_checkpoint"]
    require(sha256(sft_path) == config["sft_checkpoint_sha256"], "SFT checkpoint changed")
    schedule = training_schedule(pairs, config["steps"], config["pairs_per_batch"], config["seed"])
    output.mkdir(parents=True)
    np.save(output / "pair_schedule.npy", schedule, allow_pickle=False)
    provenance = {"config": config, "config_sha256": sha256(config_path),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
                  "prepared_artifact_sha256": sha256(prepared_path),
                  "pair_schedule_sha256": sha256(output / "pair_schedule.npy"),
                  "training_variants": len(genotypes), "training_pairs": len(pairs),
                  "sampled_distinct_pairs": int(np.unique(schedule).size),
                  "reserved_test_evaluated": False}
    save_json(output / "run.json", provenance)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    install()
    import esm
    print("Loading released ESM-IF1 and verified 5CJQ input...", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    policy, geometry = bound.policy, bound.geometry
    encoder_hash = state_digest(model.encoder)
    genotype_index = {g: i for i, g in enumerate(genotypes)}
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in genotypes]
    dev_sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in dev_scores.genotype]
    chosen_indices = pairs.chosen_genotype.map(genotype_index).to_numpy(dtype=int)
    rejected_indices = pairs.rejected_genotype.map(genotype_index).to_numpy(dtype=int)
    results = {}
    for arm in config["arms"]:
        arm_dir = output / arm
        arm_dir.mkdir()
        model.zero_grad(set_to_none=True)
        if arm == "sft_dpo":
            checkpoint = torch.load(sft_path, map_location="cpu", weights_only=True)
            require(checkpoint["schema_version"] == "cr9114-decoder-pilot/1"
                    and checkpoint["provenance"]["prepared_artifact_sha256"] == sha256(prepared_path),
                    "SFT checkpoint is incompatible with this geometry")
            model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
            del checkpoint
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        decoder_hash = state_digest(model.decoder)
        initial_checkpoint_sha = base["weights_sha256"] if arm == "direct_dpo" else config["sft_checkpoint_sha256"]
        identity = {"reference": arm, "source_checkpoint_sha256": initial_checkpoint_sha,
                    "decoder_state_sha256": decoder_hash, "encoder_state_sha256": encoder_hash,
                    "prepared_artifact_sha256": sha256(prepared_path), "geometry_digest": geometry.digest,
                    "genotype_order_sha256": genotype_digest(genotypes),
                    "probability_contract": "ESM-IF1 binary 16-site constrained summed log q; temperature 1; dropout off",
                    "policy_source_sha256": sha256(ROOT / "src/smallAntibodyGen/models/esmif1_policy.py"),
                    "compat_source_sha256": sha256(ROOT / "src/smallAntibodyGen/esmif1_compat.py"),
                    "torch_version": str(torch.__version__), "esm_version": str(esm.__version__),
                    "score_batch_size": config["score_batch_size"], "precision": "float32"}
        print(f"{arm}: scoring all {len(genotypes)} training reference variants", flush=True)
        reference_started = time.perf_counter()
        reference_values = score_sequences(policy, geometry, sequences, config["score_batch_size"],
                                            progress=arm_dir / "progress.json")
        reference_seconds = time.perf_counter() - reference_started
        cache_path = arm_dir / "reference_cache.json"
        write_reference_cache(cache_path, identity, genotypes, reference_values)
        reference_values = load_reference_cache(cache_path, identity, genotypes)
        cache_sha = sha256(cache_path)
        check_indices = np.linspace(0, len(genotypes) - 1, 16, dtype=int)
        fresh = score_sequences(policy, geometry, [sequences[i] for i in check_indices], config["pairs_per_batch"] * 2)
        np.testing.assert_allclose(fresh, reference_values[check_indices], rtol=2e-5, atol=2e-4)
        initial_scores = score_sequences(policy, geometry, dev_sequences, config["score_batch_size"])
        saved_column = "parent_log_q" if arm == "direct_dpo" else "final_log_q"
        np.testing.assert_allclose(initial_scores, dev_scores[saved_column], rtol=2e-5, atol=2e-4)
        save_json(arm_dir / "reference_checks.json", {
            "identity": identity, "cache_sha256": cache_sha, "reference_seconds": reference_seconds,
            "cache_fresh_max_error": float(np.max(np.abs(fresh - reference_values[check_indices]))),
            "initial_development_max_error": float(np.max(np.abs(initial_scores - dev_scores[saved_column]))),
            "initial_decoder_state_sha256": decoder_hash})
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        history = []
        started = time.perf_counter()
        print(f"{arm}: reference verified; starting {config['steps']} DPO updates", flush=True)
        for step, indices in enumerate(schedule, 1):
            c, r = chosen_indices[indices], rejected_indices[indices]
            batch = [sequences[i] for i in np.concatenate([c, r])]
            optimizer.zero_grad(set_to_none=True)
            logq = policy.log_prob(batch, geometry)
            ref_c = torch.tensor(reference_values[c], dtype=logq.dtype, device=logq.device)
            ref_r = torch.tensor(reference_values[r], dtype=logq.dtype, device=logq.device)
            size = len(indices)
            loss = dpo_per_pair_loss(logq[:size], logq[size:], ref_c, ref_r, beta=config["beta"]).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite DPO loss")
            if step == 1:
                require(abs(float(loss.detach()) - math.log(2)) < 1e-4, "Initial policy/reference loss is not log(2)")
            loss.backward()
            require(all(p.grad is None for p in model.encoder.parameters()), "Frozen encoder received gradients")
            norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            require(float(norm) > 0, "No decoder learning signal")
            optimizer.step()
            entry = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                     "elapsed_seconds": time.perf_counter() - started}
            history.append(entry)
            if step == 1 or step % 16 == 0:
                save_json(arm_dir / "progress.json", {"phase": "dpo_training", **entry})
                print(f"{arm} step {step}/{config['steps']}: loss={entry['loss']:.5f}, elapsed={entry['elapsed_seconds']:.1f}s", flush=True)
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                destination = arm_dir / f"decoder_step_{step:04d}.pt"
                temporary = destination.with_suffix(".tmp")
                torch.save({"schema_version": "cr9114-dpo-decoder/1", "arm": arm, "step": step,
                            "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(),
                            "provenance": provenance, "reference_identity": identity, "reference_cache_sha256": cache_sha,
                            "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state()}, temporary)
                temporary.replace(destination)
        training_seconds = time.perf_counter() - started
        model.zero_grad(set_to_none=True)
        del optimizer
        adapted_scores = score_sequences(policy, geometry, dev_sequences, config["score_batch_size"])
        metrics, scored_pairs = development_evaluation(dev_scores, dev_pairs, pair_manifest["config"], initial_scores, adapted_scores)
        table = dev_scores.copy()
        table["initial_policy_log_q"], table["adapted_log_q"] = initial_scores, adapted_scores
        table["implicit_reward"] = config["beta"] * (adapted_scores - initial_scores)
        table.to_csv(arm_dir / "development_scores.csv", index=False)
        scored_pairs.to_csv(arm_dir / "development_pair_scores.csv", index=False)
        require(state_digest(model.encoder) == encoder_hash, "Encoder changed during DPO")
        final_decoder_hash = state_digest(model.decoder)
        require(final_decoder_hash != decoder_hash and sha256(cache_path) == cache_sha, "Decoder unchanged or reference cache changed")
        final_checkpoint = arm_dir / f"decoder_step_{config['steps']:04d}.pt"
        reloaded = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
        model.decoder.load_state_dict(reloaded["decoder"], strict=True)
        del reloaded
        require(state_digest(model.decoder) == final_decoder_hash, "Checkpoint reload changes decoder tensors")
        reload_scores = score_sequences(policy, geometry, dev_sequences[:16], config["pairs_per_batch"] * 2)
        np.testing.assert_allclose(reload_scores, adapted_scores[:16], rtol=2e-5, atol=2e-4)
        result = {"status": "completed", "arm": arm, "steps": config["steps"],
                  "pair_exposures": int(schedule.size), "sequence_exposures": int(2 * schedule.size),
                  "reference_scoring_seconds": reference_seconds, "training_seconds": training_seconds,
                  "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                  "initial_loss": history[0]["loss"], "mean_last_32_training_losses": float(np.mean([h["loss"] for h in history[-32:]])),
                  "encoder_unchanged": True, "decoder_changed": True, "reference_cache_unchanged": True,
                  "decoder_state_sha256": final_decoder_hash,
                  "final_checkpoint": final_checkpoint.name, "final_checkpoint_sha256": sha256(final_checkpoint),
                  "safe_strict_checkpoint_reload": True, "reload_max_error": float(np.max(np.abs(reload_scores - adapted_scores[:16]))),
                  "development": metrics, "reserved_test_evaluated": False, "checkpoint_promoted": False}
        save_json(arm_dir / "history.json", history)
        save_json(arm_dir / "result.json", result)
        results[arm] = result
        save_json(output / "results.json", results)
        print(f"{arm} completed: pair accuracy {metrics['initial_policy']['variant_and_block_balanced_accuracy']:.3%} -> {metrics['adapted_policy']['variant_and_block_balanced_accuracy']:.3%}", flush=True)
    print(f"Both bounded DPO pilots completed. Results: {output / 'results.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_dpo_pilot.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
