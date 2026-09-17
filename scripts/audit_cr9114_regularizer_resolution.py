#!/usr/bin/env python
"""Audit prior affinity resolution, set overlaps and continuous policy movement."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.pressure import affinity_difference
from run_cr9114_esmif1_pilot import require, save_json, sha256


def run():
    directory = ROOT / "outputs/cr9114_regularization_20260917"
    evidence = json.loads((ROOT / "reference/evidence/cr9114-regularization-2026-09-17.json").read_text())
    require(sha256(directory / "results.json") == evidence["run_results_sha256"], "Prior results changed")
    result = evidence["run"]
    # The results hash covers the metrics; the cohort file needs its own pinned hash.
    require(sha256(directory / "development_records.csv") == result["output_sha256"]["development_records.csv"],
            "Development cohort changed")
    records = pd.read_csv(directory / "development_records.csv", dtype={"genotype": "string"})
    require(set(records.split) == {"development"}, "Wrong evaluation split")
    contrasts = {}
    for seed in result["config"]["seeds"]:
        control = result["arms"][f"seed_{seed}_kl"]
        for arm in ("kl_entropy", "kl_embedding"):
            name = f"seed_{seed}_{arm}"
            measured = result["arms"][name]
            cells = {}
            for k in ("16", "32"):
                first = measured["portfolios"]["ordinary"][k]["selected_genotypes"]
                reference = control["portfolios"]["ordinary"][k]["selected_genotypes"]
                cells[k] = dict(affinity_difference(records, pd.Series(1 / int(k), index=first),
                    pd.Series(1 / int(k), index=reference)), shared_candidates=len(set(first) & set(reference)),
                    swapped_candidates=len(set(first) - set(reference)), identical_set=set(first) == set(reference))
            cells["entropy"] = {"delta_nats": measured["diversity"]["entropy_nats_mc"] - control["diversity"]["entropy_nats_mc"],
                "independent_draws_sem_proxy": float(np.hypot(measured["diversity"]["entropy_mc_standard_error"], control["diversity"]["entropy_mc_standard_error"]))}
            contrasts[name] = cells
    movement = {}
    for seed in result["config"]["seeds"]:
        name = f"seed_{seed}_affinity"
        measured = result["arms"][name]
        movement[name] = {"kl_to_sft_mc": measured["kl_to_sft_mc"], "overlap_with_sft": {
            k: len(set(measured["portfolios"]["ordinary"][k]["selected_genotypes"]) & set(result["sft"]["portfolios"]["ordinary"][k]["selected_genotypes"])) for k in ("16", "32")}}
    output = {"schema_version": "cr9114-regularizer-resolution/1", "prior_results_sha256": evidence["run_results_sha256"],
        "regularizer_contrasts_against_kl": contrasts, "affinity_only_movement_from_sft": movement,
        "uncertainty_scope": "Assay-error proxy assumes independent genotype errors and cancels shared identities. Entropy SE proxy ignores shared-RNG covariance. Neither is a calibrated noise floor or formal significance test.",
        "interpretation": "Affinity contrasts are below their assay-SEM proxy when any candidates change. Embedding sets match at both budgets for the third seed; other seeds have small swaps. Entropy increases are about 1.9-3.6 independent-MC-SE proxies. Affinity-only policies move modestly but measurably from SFT; the extra regularizers mostly perturb those policies weakly."}
    save_json(ROOT / "reference/evidence/cr9114-regularizer-resolution-2026-09-17.json", output)
    print(json.dumps(movement, indent=2))


if __name__ == "__main__": run()
