#!/usr/bin/env python
"""Calibrate a label-blind shortlist selector, then evaluate one fresh cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.shortlist import calibrate_admission, portfolio_metrics, select_shortlist
from run_cr9114_dpo_pilot import load_inputs, score_sequences
from run_cr9114_esmif1_pilot import require, save_json, sha256, stable_subset, state_digest
from run_cr9114_diversity_pilot import fresh_cohort


def load_non_test():
    prior = json.loads((ROOT / "configs/experiments/cr9114_dpo_pilot.json").read_text())
    pilot, manifest, _, _, old_scores, _ = load_inputs(prior)
    path = ROOT / prior["preference_dir"] / "eligible_non_test_records.csv"
    require(sha256(path) == manifest["output_files"][path.name]["sha256"], "Measurements changed")
    records = pd.read_csv(path, dtype={"genotype": "string"})
    require(set(records.split) == {"train", "development"}, "Test measurements in input")
    return prior, pilot, records, path


def load_sft(prior, pilot):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), "CUDA required")
    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy
    install()
    import esm
    base = pilot["config"]
    path = ROOT / base["prepared_artifact"]
    require(sha256(path) == pilot["prepared_artifact_sha256"], "Structure artifact changed")
    prepared = load_prepared_structure(path)
    require(not verify_against_source(prepared, ROOT / base["structure"]), "Structure source changed")
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    sft_path = ROOT / prior["sft_checkpoint"]
    require(sha256(weights) == base["weights_sha256"] and sha256(sft_path) == prior["sft_checkpoint_sha256"], "Weights changed")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    require(bound.geometry.digest == bind_policy(prepared, model, alphabet).geometry.digest, "Encoding did not repeat")
    state = torch.load(sft_path, map_location="cpu", weights_only=True)["decoder"]
    model.decoder.load_state_dict(state, strict=True)
    identity = {"decoder_state_sha256": state_digest(model.decoder), "encoder_state_sha256": state_digest(model.encoder),
                "geometry_digest": bound.geometry.digest, "sft_checkpoint_sha256": sha256(sft_path),
                "weights_sha256": base["weights_sha256"], "prepared_artifact_sha256": sha256(path)}
    return model, bound, state, identity


def prior_development_ids():
    paths = [("outputs/cr9114_5cjq_pilot_20260916/development_scores.csv", None),
             ("outputs/cr9114_dpo_diagnostics_20260916/fresh_development.csv", "reference/evidence/cr9114-dpo-diagnostics-2026-09-16.json"),
             ("outputs/cr9114_diversity_pilot_20260916_v3/fresh_development.csv", "reference/evidence/cr9114-diversity-pilot-2026-09-16.json"),
             ("outputs/cr9114_affinity_pilot_20260916/fresh_development.csv", "reference/evidence/cr9114-affinity-pilot-2026-09-17.json")]
    excluded, hashes = set(), {}
    for relative, evidence in paths:
        path = ROOT / relative
        if evidence:
            doc = json.loads((ROOT / evidence).read_text())
            require(sha256(path) == doc.get("run", doc)["fresh_cohort_sha256"], "Earlier cohort changed")
        hashes[relative] = sha256(path)
        excluded.update(pd.read_csv(path, dtype={"genotype": "string"}).genotype)
    require(len(excluded) == 6656, "Unexpected earlier development population")
    return excluded, hashes


def scores_for(bound, genotypes, name):
    seqs = [bound.space.sequence_for(tuple(map(int, g))) for g in genotypes]
    values = []
    for start in range(0, len(seqs), 512):
        values.extend(score_sequences(bound.policy, bound.geometry, seqs[start:start + 512], 16))
        print(f"{name}: {len(values)}/{len(seqs)} scored", flush=True)
    return pd.Series(values, index=genotypes, name="score")


def run(config_path, output):
    require(not output.exists(), "Choose a fresh output directory")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit protocol first")
    config = json.loads(config_path.read_text())
    require(config["schema_version"] == "cr9114-shortlist/1", "Unsupported protocol")
    prior, pilot, records, record_path = load_non_test()
    train = records[records.split == "train"]
    threshold = float(train["mean"].quantile(config["quality_threshold_training_quantile"]))
    count = config["calibration_count_per_cohort"]
    calibration = stable_subset(train, count * config["calibration_cohorts"], config["calibration_seed"])
    require(len(calibration) == count * config["calibration_cohorts"], "Insufficient calibration records")
    calibration["calibration_cohort"] = np.arange(len(calibration)) // count
    excluded, previous_hashes = prior_development_ids()
    fresh = fresh_cohort(records, excluded, config["fresh_development_count"], config["fresh_development_seed"])
    output.mkdir(parents=True)
    model, bound, _, identity = load_sft(prior, pilot)
    train_scores = scores_for(bound, calibration.genotype, "training calibration")
    calibration.assign(score=calibration.genotype.map(train_scores)).to_csv(output / "calibration.csv", index=False)
    decision = calibrate_admission(calibration, train_scores, dict(config, quality_threshold=threshold))
    # This file exists before any fresh development scoring, selection, or label join.
    save_json(output / "locked_selection.json", decision)
    print(f"Training-only locked admission multiplier: {decision['selected_multiplier']}", flush=True)
    dev_scores = scores_for(bound, fresh.genotype, "fresh development")
    candidates = pd.DataFrame({"genotype": fresh.genotype, "score": fresh.genotype.map(dev_scores)})
    selections = {name: {str(k): select_shortlist(candidates, k, multiplier) for k in config["budgets"]}
                  for name, multiplier in (("ordinary", 1), ("diverse", decision["selected_multiplier"]))}
    save_json(output / "selections_before_evaluation.json", selections)
    results = {"schema_version": "cr9114-shortlist-result/1", "status": "completed", "config": config,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config_path), "input_records_sha256": sha256(record_path), "initial_identity": identity,
        "prior_cohort_sha256": previous_hashes, "previous_development_count": len(excluded),
        "remaining_unevaluated_eligible_development": int((records.split == "development").sum()) - len(excluded) - len(fresh),
        "calibration": decision, "locked_selection_sha256": sha256(output / "locked_selection.json"),
        "reserved_test_labels_evaluated": False, "model_trained": False, "checkpoint_promoted": False,
        "portfolios": {name: {k: portfolio_metrics(fresh, ids, threshold) for k, ids in budgets.items()}
                       for name, budgets in selections.items()}}
    checks = {}
    for k in map(str, config["budgets"]):
        base, diverse = results["portfolios"]["ordinary"][k], results["portfolios"]["diverse"][k]
        checks[k] = {"affinity_within_tolerance": diverse["mean_affinity"] >= base["mean_affinity"] - config["affinity_loss_tolerance"],
                     "mean_hamming_gain": diverse["diversity"]["mean_hamming"] >= base["diversity"]["mean_hamming"] + config["minimum_mean_hamming_gain"],
                     "near_duplicate_fraction_not_increased": diverse["diversity"]["hamming_le_one_pair_fraction"] <= base["diversity"]["hamming_le_one_pair_fraction"]}
    results["screen"] = {"checks": checks, "passes": all(all(c.values()) for c in checks.values())}
    require(state_digest(model.decoder) == identity["decoder_state_sha256"], "SFT model changed")
    fresh.to_csv(output / "fresh_development.csv", index=False)
    candidates.to_csv(output / "development_scores.csv", index=False)
    results["output_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    save_json(output / "results.json", results)
    print(json.dumps({"screen": results["screen"], "portfolios": results["portfolios"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/cr9114_shortlist.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cr9114_shortlist_20260917")
    args = parser.parse_args()
    run(args.config, args.output)
