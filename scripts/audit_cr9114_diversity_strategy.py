#!/usr/bin/env python
"""Audit saved diversity evidence and metric counterexamples; no model training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    require(not output.exists(), "Choose a fresh audit output")
    evidence_path = ROOT / "reference/evidence/cr9114-affinity-pilot-2026-09-17.json"
    evidence = json.loads(evidence_path.read_text())
    run = evidence["run"]
    shortlist = evidence["independent_audit"]["models"]
    config = run["config"]
    require(run["status"] == "completed" and evidence["independent_audit"]["status"] == "passed", "Unaudited inputs")
    require(not run["reserved_test_labels_evaluated"], "Unexpected test-label evaluation")
    result = {"schema_version": "cr9114-diversity-strategy-audit/1", "date": "2026-09-17",
        "source_evidence_sha256": sha256(evidence_path), "script_sha256": sha256(Path(__file__)),
        "new_training_performed": False, "new_assay_labels_read": False,
        "best_method_established": False, "comparisons": {}}
    for seed in config["seeds"]:
        with_name, without_name = f"seed_{seed}_affinity_entropy", f"seed_{seed}_affinity_plain"
        with_entropy, without = run["arms"][with_name], run["arms"][without_name]
        result["comparisons"][str(seed)] = {
            "entropy_minus_plain_top_k_affinity": {k: with_entropy["development"]["top_k_mean"][k]
                - without["development"]["top_k_mean"][k] for k in ("16", "32")},
            "entropy_minus_plain_sample_hamming": with_entropy["diversity"]["mean_pairwise_hamming_unbiased"]
                - without["diversity"]["mean_pairwise_hamming_unbiased"],
            "entropy_unique_count": with_entropy["diversity"]["unique_genotypes"],
            "plain_unique_count": without["diversity"]["unique_genotypes"],
            "entropy_minus_sft_shortlist_hamming": {k: shortlist[with_name]["posthoc_top_k_mean_hamming"][k]
                - shortlist["sft"]["posthoc_top_k_mean_hamming"][k] for k in ("16", "32")},
            "entropy_passes_sft_screen": with_entropy["screen"]["passes"],
            "plain_passes_sft_diversity_gates": all(without["screen"]["checks"][k] for k in
                ("unique_genotypes_retained", "mean_pairwise_hamming_unbiased_retained"))}
    dimension = config["likelihood_normalizer"]
    support = 2 ** dimension
    draws = config["evaluation_samples"]
    result["objective_scale"] = {
        "variable_sites": dimension, "entropy_coefficient_per_site_nll": .03,
        "equivalent_entropy_coefficient_sequence_nll": .03 * dimension,
        "note": "Multiply the entire nominal objective by 16; this does not assert identical Adam/clipping trajectories or an implementation bug.",
        "explicit_reference_kl_in_affinity_training_loss": False}
    result["uniform_policy_benchmark"] = {
        "support": support, "draws": draws,
        "expected_unique": float(-support * math.expm1(draws * math.log1p(-1 / support))),
        "entropy_nats": dimension * math.log(2), "expected_hamming": dimension / 2,
        "collision_probability": 1 / support,
        "note": "Analytic comparison only; no genotypes were scored or joined to assay labels."}
    # Two equiprobable opposite binary strings have maximum IID mean Hamming,
    # despite retaining only two modes. This checks the metric, not our model.
    binary = np.array([[0] * dimension, [1] * dimension])
    distances = (binary[:, None, :] != binary[None, :, :]).sum(axis=-1)
    two_mode_hamming = float(distances.mean())
    require(two_mode_hamming == dimension / 2, "Counterexample calculation failed")
    result["hamming_counterexample"] = {"support": 2, "expected_hamming": two_mode_hamming,
        "entropy_nats": math.log(2), "collision_probability": .5,
        "interpretation": "Mean Hamming alone does not certify broad joint support."}
    two = np.array([[1., 0.], [-1., 0.]])
    four = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    cosine = [float((z @ z.T).mean()) for z in (two, four)]
    require(cosine == [0., 0.], "Cosine counterexample calculation failed")
    result["cosine_counterexample"] = {"mean_iid_cosine_support_two": cosine[0],
        "mean_iid_cosine_support_four": cosine[1],
        "interpretation": "Average cosine similarity alone is not a mode-count or functional-diversity guarantee; this is not an empirical criticism of combined objectives."}
    save_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reference/evidence/cr9114-diversity-strategy-audit-2026-09-17.json")
    run(parser.parse_args().output)
