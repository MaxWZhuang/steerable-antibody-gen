#!/usr/bin/env python
"""Post-hoc checkpoint, fresh-development and on-policy diversity diagnostics.

Never loads raw assay data or reserved-test labels. Does not train or select a
checkpoint. Sampling covers the whole constrained space without label joins.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_cr9114_dpo_pilot import load_inputs, score_sequences  # noqa: E402
from run_cr9114_esmif1_pilot import require, save_json, sha256, stable_subset  # noqa: E402
from prepare_cr9114_preferences import construct_pairs  # noqa: E402


def diversity_metrics(genotypes, log_q):
    """Entropy uses on-policy log probabilities, not a softmax over a subset."""
    require(len(genotypes) > 1 and all(len(g) == 16 and set(g) <= {"0", "1"}
                                     for g in genotypes), "Invalid sample genotypes")
    log_q = np.asarray(log_q, dtype=float)
    require(log_q.shape == (len(genotypes),) and np.isfinite(log_q).all()
            and (log_q <= 1e-6).all(), "Invalid sample log probabilities")
    n = len(genotypes)
    counts = np.array(list(Counter(genotypes).values()))
    p = np.array([list(map(int, g)) for g in genotypes]).mean(axis=0)
    entropy = float(-log_q.mean())
    return {"sample_count": n, "unique_genotypes": len(counts),
            "duplicate_fraction": 1 - len(counts) / n,
            "largest_observed_genotype_fraction": float(counts.max() / n),
            "entropy_nats_mc": entropy,
            "entropy_mc_standard_error": float(log_q.std(ddof=1) / np.sqrt(n)),
            "entropy_effective_support_mc": float(np.exp(entropy)),
            "collision_probability_unbiased": float(np.sum(counts * (counts - 1)) / (n * (n - 1))),
            "mean_pairwise_hamming_unbiased": float(2 * np.sum(p * (1 - p)) * n / (n - 1)),
            "allele_one_frequencies": p.tolist(),
            "sites_with_minor_allele_below_1pct": int((np.minimum(p, 1 - p) < .01).sum())}


def pair_metrics(pairs, scores, reference=None, beta=.1, weighted=True):
    margin = (scores.loc[pairs.chosen_genotype].to_numpy()
              - scores.loc[pairs.rejected_genotype].to_numpy())
    require(np.isfinite(margin).all(), "Nonfinite pair margins")
    w = pairs.pair_weight.to_numpy() if weighted else np.ones(len(pairs))
    credit = np.where(margin > 1e-6, 1., np.where(margin < -1e-6, 0., .5))
    result = {"pair_accuracy": float(np.average(credit, weights=w)), "pairs": len(pairs),
              "per_block_accuracy": {str(int(b)): float(np.average(credit[m], weights=w[m]))
                  for b in sorted(pairs.block.unique()) for m in [np.array(pairs.block == b)]}}
    if reference is not None:
        ref_margin = (reference.loc[pairs.chosen_genotype].to_numpy()
                      - reference.loc[pairs.rejected_genotype].to_numpy())
        result["dpo_loss"] = float(np.average(np.logaddexp(0, -beta * (margin - ref_margin)), weights=w))
    return result


def cohort_metrics(records, pairs, scores, reference=None):
    indexed = records.set_index("genotype")
    ordered = indexed.assign(score=scores.reindex(indexed.index)).reset_index()
    result = pair_metrics(pairs, scores, reference)
    result["spearman"] = float(spearmanr(ordered.score, ordered["mean"]).statistic)
    result["top_k_mean"] = {str(k): float(ordered.sort_values(
        ["score", "genotype"], ascending=[False, True]).head(k)["mean"].mean()) for k in (16, 32)}
    result["per_block_spearman"] = {str(int(b)): float(spearmanr(g.score, g["mean"]).statistic)
                                      for b, g in ordered.groupby("block")}
    return result


def run(output):
    require(not output.exists(), "Choose a fresh output directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit the diagnostic protocol before running")
    config = json.loads((ROOT / "configs/experiments/cr9114_dpo_pilot.json").read_text())
    settings = json.loads((ROOT / "configs/experiments/cr9114_preferences.json").read_text())
    pilot, manifest, train_pairs, old_pairs, old_scores, _ = load_inputs(config)
    pair_dir = ROOT / config["preference_dir"]
    records_path = pair_dir / "eligible_non_test_records.csv"
    require(sha256(records_path) == manifest["output_files"][records_path.name]["sha256"], "Records changed")
    records = pd.read_csv(records_path, dtype={"genotype": "string"})
    require(set(records.split) == {"train", "development"}, "Test labels in diagnostic inputs")
    old = records[records.genotype.isin(old_scores.genotype)].copy()
    fresh = stable_subset(records[(records.split == "development")
                                 & ~records.genotype.isin(old_scores.genotype)], 2048, 20260917)
    require(len(fresh) == 2048 and not set(fresh.genotype) & set(old.genotype), "Fresh cohort overlap")
    fresh_pairs, fresh_audit = construct_pairs(fresh, settings)
    run_dir = ROOT / "outputs/cr9114_dpo_pilot_20260916"
    schedule_path = run_dir / "pair_schedule.npy"
    require(sha256(schedule_path) == "d1717fd2c654e6015a478fefab7d924061e9ea63f1c5f2fcac7e94fc81f4eb6e",
            "Training schedule changed")
    sampled_train = train_pairs.iloc[np.load(schedule_path).ravel()].reset_index(drop=True)
    train_genotypes = sorted(set(sampled_train.chosen_genotype) | set(sampled_train.rejected_genotype))
    require(not set(train_genotypes) & (set(old.genotype) | set(fresh.genotype)), "Train/dev overlap")
    output.mkdir(parents=True)
    fresh.to_csv(output / "fresh_development.csv", index=False)
    fresh_pairs.to_csv(output / "fresh_development_pairs.csv", index=False)
    evidence = {"schema": "cr9114-dpo-diagnostics/1", "reserved_test_evaluated": False,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "script_sha256": sha256(Path(__file__)), "fresh_cohort": fresh_audit,
                "fresh_cohort_sha256": sha256(output / "fresh_development.csv"),
                "fresh_pairs_sha256": sha256(output / "fresh_development_pairs.csv"),
                "sample_count_per_model": 1024, "sample_seed": 20260917,
                "sampling_temperature": 1, "training_schedule_variants": len(train_genotypes),
                "train_development_identity_overlap": 0, "models": {}, "checkpoint_sha256": {},
                "scope": "Post-hoc diagnostics; fresh variants share the same three development blocks. No checkpoint selection, model update or test-label evaluation. Monte Carlo errors are conditional on fixed checkpoints, not training-seed uncertainty."}
    save_json(output / "results.json", evidence)

    # Reproduce the existing train-only additive ridge baseline on both pools.
    train = records[records.split == "train"]
    def design(table):
        return np.array([[1., *map(float, g)] for g in table.genotype])
    x = design(train)
    penalty = np.eye(17)
    penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ train["mean"].to_numpy())
    evidence["additive_ridge"] = {name: cohort_metrics(table, pairs, pd.Series(design(table) @ coef, index=table.genotype))
        for name, table, pairs in (("original", old, old_pairs), ("fresh", fresh, fresh_pairs))}
    # Distances use identities only, including the complete eligible training pool.
    popcount = np.array([i.bit_count() for i in range(65536)], dtype=np.uint8)
    for label, gs in (("exposed_dpo", train_genotypes), ("eligible_train", train.genotype)):
        train_codes = np.array([int(g, 2) for g in gs], dtype=np.uint16)
        nearest = [int(popcount[np.bitwise_xor(int(g, 2), train_codes)].min()) for g in fresh.genotype]
        evidence[f"fresh_min_hamming_to_{label}"] = dict(sorted(Counter(map(str, nearest)).items()))

    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy
    install()
    import esm
    torch.set_num_threads(4)
    torch.manual_seed(20260917)
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    base = pilot["config"]
    require(sha256(weights) == base["weights_sha256"], "Parent checkpoint changed")
    prepared_path = ROOT / base["prepared_artifact"]
    require(sha256(prepared_path) == pilot["prepared_artifact_sha256"], "Context changed")
    prepared = load_prepared_structure(prepared_path)
    require(not verify_against_source(prepared, ROOT / base["structure"]), "Context verification failed")
    print("Loading model for frozen-checkpoint diagnostics", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    parent_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    states = [("parent", None), ("sft", ROOT / config["sft_checkpoint"])]
    states += [(f"{arm}_{step:04d}", run_dir / arm / f"decoder_step_{step:04d}.pt")
               for arm in ("direct_dpo", "sft_dpo") for step in (64, 128, 192, 256)]
    all_scores, samples = {}, {}
    for name, path in states:
        print(f"Scoring {name}", flush=True)
        if path is not None:
            digest = sha256(path)
            if name == "sft":
                require(digest == config["sft_checkpoint_sha256"], "SFT checkpoint changed")
            elif name.endswith("0256"):
                prior = json.loads((path.parent / "result.json").read_text())
                require(digest == prior["final_checkpoint_sha256"], "Final DPO checkpoint changed")
            evidence["checkpoint_sha256"][name] = digest
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
            del checkpoint
        final = name in ("parent", "sft", "direct_dpo_0256", "sft_dpo_0256")
        gs = train_genotypes + old.genotype.tolist() + (fresh.genotype.tolist() if final else [])
        sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in gs]
        scores = pd.Series(score_sequences(bound.policy, bound.geometry, sequences, 16,
                           progress=output / "progress.json"), index=gs)
        all_scores[name] = scores
        scores.rename("log_q").rename_axis("genotype").to_csv(output / f"{name}_scores.csv")
        ref_name = "sft" if name.startswith("sft_dpo") else "parent"
        reference = all_scores[ref_name]
        entry = {"original_development": cohort_metrics(old, old_pairs, scores, reference),
                 "fixed_training_schedule_pool": pair_metrics(sampled_train, scores, reference, weighted=False)}
        if "dpo_" in name:
            step = int(name[-4:])
            entry["already_exposed_training_pairs"] = pair_metrics(sampled_train.iloc[:2 * step], scores, reference, weighted=False)
        if final:
            entry["fresh_development"] = cohort_metrics(fresh, fresh_pairs, scores, reference)
            # Validate unchanged scoring against the original run before sampling.
            if name in ("parent", "sft"):
                expected = old_scores.set_index("genotype")["parent_log_q" if name == "parent" else "final_log_q"]
            else:
                expected = pd.read_csv(path.parent / "development_scores.csv", dtype={"genotype": "string"}).set_index("genotype").adapted_log_q
            err = float(np.max(np.abs(scores.loc[expected.index] - expected)))
            require(err < 1e-4, "Saved score parity failed")
            entry["saved_score_max_error"] = err
            generator = torch.Generator(device="cuda").manual_seed(20260917)
            genotypes, log_q = [], []
            for start in range(0, 1024, 32):
                sample = bound.policy.sample(bound.geometry, num_samples=32, generator=generator)
                genotypes.extend("".join(map(str, g)) for g in sample.alleles)
                log_q.extend(sample.log_probability.cpu().tolist())
                if (start + 32) % 256 == 0:
                    print(f"{name}: sampled {start + 32}/1024", flush=True)
            rescore = score_sequences(bound.policy, bound.geometry,
                [bound.space.sequence_for(tuple(map(int, g))) for g in genotypes[:32]], 16)
            err = float(np.max(np.abs(rescore - np.array(log_q[:32]))))
            require(err < 1e-4, "Sample/teacher-forced score mismatch")
            entry["sample_score_max_error"] = err
            entry["diversity"] = diversity_metrics(genotypes, log_q)
            samples[name] = pd.DataFrame({"genotype": genotypes, "log_q": log_q})
            samples[name].to_csv(output / f"{name}_samples.csv", index=False)
        evidence["models"][name] = entry
        save_json(output / "results.json", evidence)

    # KL estimates use samples drawn from each adapted policy and its actual reference.
    for ref_name, names in (("parent", ("sft", "direct_dpo_0256")), ("sft", ("sft_dpo_0256",))):
        if ref_name == "parent":
            model.decoder.load_state_dict(parent_state, strict=True)
        else:
            checkpoint = torch.load(ROOT / config["sft_checkpoint"], map_location="cpu", weights_only=True)
            model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
            del checkpoint
        for name in names:
            sample = samples[name]
            ref = score_sequences(bound.policy, bound.geometry,
                [bound.space.sequence_for(tuple(map(int, g))) for g in sample.genotype], 16)
            ratios = sample.log_q.to_numpy() - ref
            evidence["models"][name]["kl_to_reference_mc"] = {"reference": ref_name,
                "nats": float(ratios.mean()), "standard_error": float(ratios.std(ddof=1) / np.sqrt(len(ratios)))}
            sample.assign(reference_log_q=ref).to_csv(output / f"{name}_samples.csv", index=False)
    evidence["output_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.csv"))}
    evidence["status"] = "completed"
    save_json(output / "results.json", evidence)
    print("Diagnostics completed; reserved-test labels untouched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cr9114_dpo_diagnostics_20260916")
    run(parser.parse_args().output)
