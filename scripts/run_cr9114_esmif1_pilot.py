#!/usr/bin/env python
"""Released-model scoring checks followed by a bounded CR9114 decoder SFT pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_genotypes(genotypes, config):
    import numpy as np
    rng = np.random.default_rng(config["seed"])
    loci = sorted(rng.choice(16, size=config["split_loci_count"], replace=False).tolist())
    blocks = rng.permutation(2 ** len(loci)).tolist()
    ntrain, ndev = config["train_blocks"], config["development_blocks"]
    require(ntrain > 0 and ndev > 0 and ntrain + ndev < len(blocks), "Invalid block counts")
    names = {b: "train" if i < ntrain else "development" if i < ntrain + ndev else "test"
             for i, b in enumerate(blocks)}
    assignments = [names[int("".join(g[i] for i in loci), 2)] for g in genotypes]
    return assignments, {"loci_0based": loci, "blocks": names, "seed": config["seed"]}


def eligible_records(frame):
    """Called only after reserved-test rows have been removed."""
    import numpy as np
    reps = frame[["h1_repa", "h1_repb", "h1_repc"]].to_numpy(dtype=float)
    require(not np.isinf(reps).any(), "Infinite assay replicate")
    counts = np.isfinite(reps).sum(axis=1)
    above_floor = np.all(np.isnan(reps) | (reps > 7.0), axis=1)
    keep = (counts >= 2) & above_floor
    result = frame.loc[keep, ["genotype", "split"]].copy()
    values = reps[keep]
    result["h1_mean_recomputed"] = np.nanmean(values, axis=1)
    result["h1_sem_recomputed"] = np.nanstd(values, axis=1, ddof=1) / np.sqrt(counts[keep])
    return result


def stable_subset(frame, count, seed):
    ordered = frame.assign(_key=frame.genotype.map(
        lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()))
    return ordered.sort_values("_key").head(count).drop(columns="_key").copy()


def load_cohort(config, space, output):
    import numpy as np
    import pandas as pd
    path = ROOT / config["data"]
    require(sha256(path) == config["data_sha256"], "Landscape SHA-256 changed")
    frame = pd.read_csv(path, dtype={"genotype": "string"})
    require(len(frame) == 65536 and frame.genotype.nunique() == 65536, "Incomplete genotype set")
    require(frame.genotype.str.fullmatch("[01]{16}").all(), "Invalid genotype strings")
    expected = frame[[f"pos{i}" for i in range(1, 17)]].astype(str).agg("".join, axis=1)
    require((frame.genotype == expected).all(), "Genotype columns disagree")
    frame["split"], split = split_genotypes(frame.genotype.tolist(), config)
    frame[["genotype", "split"]].to_csv(output / "split.csv", index=False)
    # Labels from reserved blocks never enter aggregation or any fitted statistic.
    eligible = eligible_records(frame.loc[frame.split != "test"])
    train = eligible[eligible.split == "train"].copy()
    dev = eligible[eligible.split == "development"].copy()
    threshold = float(train.h1_mean_recomputed.quantile(config["positive_quantile"]))
    positives = train[train.h1_mean_recomputed >= threshold].copy()
    require(len(positives) >= config["batch_size"] and len(dev) >= config["development_sample_size"],
            "Insufficient eligible pilot data")
    dev = stable_subset(dev, config["development_sample_size"], config["seed"])
    train_score = stable_subset(positives, config["train_score_sample_size"], config["seed"])
    for table in (positives, dev, train_score):
        table["sequence"] = table.genotype.map(lambda g: space.sequence_for(tuple(map(int, g))))
    # A conventional additive baseline, with fixed ridge penalty (no dev tuning).
    def design(table):
        return np.array([[1.0, *map(float, g)] for g in table.genotype])
    x, y = design(train), train.h1_mean_recomputed.to_numpy()
    penalty = np.eye(17)
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    dev["additive_ridge_prediction"] = design(dev) @ coefficients
    positives.drop(columns="sequence").to_csv(output / "training_positives.csv", index=False)
    split.update({"all_split_counts": {str(k): int(v) for k, v in frame.split.value_counts().items()},
                  "eligible_training": len(train), "eligible_development": int((eligible.split == "development").sum()),
                  "training_positives": len(positives), "training_only_positive_threshold": threshold,
                  "evaluated_development": len(dev), "reserved_test_evaluated": False,
                  "split_csv_sha256": sha256(output / "split.csv")})
    save_json(output / "cohort.json", split)
    return positives, train_score, dev


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(config_path, output):
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy, build_edit_space

    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-sft-pilot/1", "Unknown pilot schema")
    require(config["precision"] == "float32", "Only validated float32 pilot is supported")
    require(config["device"] == "cuda" and torch.cuda.is_available(), "CUDA training device unavailable")
    require(config["steps"] > 0 and config["batch_size"] > 0 and config["checkpoint_every"] > 0,
            "Steps, batch size and checkpoint interval must be positive")
    require(not output.exists(), "Output directory exists; choose a fresh run directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit the implementation and config before starting a recorded run")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    require(weights.is_file() and sha256(weights) == config["weights_sha256"], "Missing or changed released weights")
    prepared = load_prepared_structure(ROOT / config["prepared_artifact"])
    require(not verify_against_source(prepared, ROOT / config["structure"]), "Prepared structure/source mismatch")
    require(list(prepared.manifest.policy_site_ids) == [f"pos{i}" for i in range(1, 17)], "Unexpected site order")
    output.mkdir(parents=True)
    provenance = {"config": config, "config_sha256": sha256(config_path), "git_commit": revision,
                  "prepared_artifact_sha256": sha256(ROOT / config["prepared_artifact"]),
                  "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    save_json(output / "run.json", provenance)
    train, train_score, dev = load_cohort(config, build_edit_space(prepared), output)
    print(f"Cohort: {len(train)} training positives, {len(dev)} development scoring records; test reserved.", flush=True)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(4)
    install()
    import esm
    from esm.inverse_folding.util import CoordBatchConverter
    print("Loading hash-verified ESM-IF1 checkpoint and encoding 5CJQ context...", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model = model.eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    bound = bind_policy(prepared, model, alphabet)
    policy, geometry = bound.policy, bound.geometry
    encoder_before = state_digest(model.encoder)
    decoder_before = state_digest(model.decoder)
    parity_errors = []
    with torch.no_grad():
        for sequence in train_score.sequence.iloc[:3]:
            coords, confidence, _, tokens, padding = CoordBatchConverter(alphabet)(
                [(prepared.coordinates, prepared.confidence, sequence)], device="cuda")
            native, _ = model(coords, padding, confidence, tokens[:, :-1])
            cached = policy.native_logits(sequence, geometry)
            torch.testing.assert_close(cached, native, atol=2e-4, rtol=2e-5)
            parity_errors.append(float((cached - native).abs().max()))
        sample = policy.sample(geometry, num_samples=2)
        rescored = policy.log_prob(sample.sequences, geometry)
        torch.testing.assert_close(sample.log_probability, rescored, atol=2e-4, rtol=2e-5)
        sample_error = float((sample.log_probability - rescored).abs().max())
    parity = {"status": "pass", "native_cached_logit_max_errors": parity_errors,
              "sample_rescore_max_error": sample_error, "atol": 2e-4, "rtol": 2e-5,
              "encoder_frozen": all(not p.requires_grad for p in model.encoder.parameters()),
              "geometry_rows": prepared.num_rows}
    save_json(output / "scoring_checks.json", parity)
    print(f"Scoring parity passed: max native-logit error {max(parity_errors):.3g}; sample/rescore {sample_error:.3g}.", flush=True)

    def score(table):
        values = []
        with torch.no_grad():
            for start in range(0, len(table), config["batch_size"]):
                logq = policy.log_prob(table.sequence.iloc[start:start + config["batch_size"]].tolist(), geometry)
                require(bool(torch.isfinite(logq).all()), "Nonfinite candidate score")
                values.extend(logq.cpu().tolist())
        return np.array(values)

    dev["parent_log_q"] = score(dev)
    train_score["parent_log_q"] = score(train_score)
    dev.drop(columns="sequence").to_csv(output / "parent_development_scores.csv", index=False)
    print("Parent scoring complete. Starting decoder-only supervised training...", flush=True)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config["seed"])
    started = time.perf_counter()
    history = []

    def checkpoint(step):
        destination = output / f"decoder_step_{step:04d}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save({"schema_version": "cr9114-decoder-pilot/1", "step": step,
                    "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(),
                    "provenance": provenance, "geometry_digest": geometry.digest,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                    "numpy_rng_state": rng.bit_generator.state}, temporary)
        temporary.replace(destination)

    for step in range(1, config["steps"] + 1):
        indices = rng.integers(0, len(train), size=config["batch_size"])
        sequences = train.sequence.iloc[indices].tolist()
        optimizer.zero_grad(set_to_none=True)
        loss = -policy.log_prob(sequences, geometry).mean()
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        loss.backward()
        require(all(p.grad is None for p in model.encoder.parameters()), "Frozen encoder received gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"],
                                                  error_if_nonfinite=True)
        require(float(grad_norm) > 0, "Zero decoder gradient")
        optimizer.step()
        entry = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                 "elapsed_seconds": time.perf_counter() - started}
        history.append(entry)
        if step == 1 or step % 16 == 0:
            save_json(output / "progress.json", entry)
            print(f"step {step}/{config['steps']} loss={entry['loss']:.4f} elapsed={entry['elapsed_seconds']:.1f}s", flush=True)
        if step % config["checkpoint_every"] == 0 or step == config["steps"]:
            checkpoint(step)
    training_seconds = time.perf_counter() - started
    dev["final_log_q"] = score(dev)
    train_score["final_log_q"] = score(train_score)
    require(state_digest(model.encoder) == encoder_before, "Frozen encoder weights changed")
    require(state_digest(model.decoder) != decoder_before, "Decoder weights did not change")
    dev.drop(columns="sequence").to_csv(output / "development_scores.csv", index=False)
    train_score.drop(columns="sequence").to_csv(output / "training_scores.csv", index=False)
    result = {"status": "completed", "steps": config["steps"],
              "training_seconds": training_seconds, "encoder_unchanged": True, "decoder_changed": True,
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
              "training_positive_mean_nll_parent": float(-train_score.parent_log_q.mean()),
              "training_positive_mean_nll_final": float(-train_score.final_log_q.mean()),
              "development_spearman_parent": float(spearmanr(dev.parent_log_q, dev.h1_mean_recomputed).statistic),
              "development_spearman_final": float(spearmanr(dev.final_log_q, dev.h1_mean_recomputed).statistic),
              "development_spearman_additive_ridge": float(spearmanr(dev.additive_ridge_prediction, dev.h1_mean_recomputed).statistic),
              "reserved_test_evaluated": False, "checkpoint_promoted": False,
              "interpretation": "Single-seed bounded development pilot, no confirmatory or biological improvement claim.",
              "final_checkpoint": f"decoder_step_{config['steps']:04d}.pt"}
    save_json(output / "history.json", history)
    save_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_5cjq_pilot.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
