#!/usr/bin/env python
"""Does a policy's log probability beat the model-free baselines on the cohort?

``scripts/her2_affinity_scoring.py`` reports each policy's rank correlation with
measured affinity and, separately, each baseline's. Two correlations printed
side by side are not a comparison: the same 152 clones produce both, so the
difference has to be tested on paired resamples of those clones.

That is the whole content of this script. It reads the persisted per-row scores
-- no GPU, no rescoring -- and, under a shared row resample, differences every
checkpoint's Spearman against:

* **negative distance to wild type.** The cohort's affinity falls off with
  mutational load. Any scorer that merely prefers near-wild-type cores inherits
  that correlation without knowing anything about binding.
* **nearest labelled-library neighbour class.** Simple memorization of the
  adaptation data, extended one Hamming step.

A policy that does not clear both is not reporting an independent signal.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))
from her2_affinity_scoring import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED,  # noqa: E402
                                   bootstrap_rows, interval, spearman)

from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    args = parser.parse_args(argv)

    out = ROOT / args.output
    cohort = {r["clone_id"]: r for r in
              csv.DictReader((out / "affinity_152.csv").open(encoding="utf-8"))}
    rows = list(csv.DictReader((out / "affinity_scores.csv").open(encoding="utf-8")))
    checkpoints = [name for name in rows[0]
                   if "_seed" in name and name.split("_seed")[0] != "clone"]

    affinity = np.array([float(r["neg_log10_kd"]) for r in rows])
    wild_type = np.array([-float(r["wt_distance"]) for r in rows])
    neighbour = np.array([float(cohort[r["clone_id"]]["nearest_neighbour_score"])
                          for r in rows])
    masks = {"primary_152": np.array([r["primary_152"] == "True" for r in rows]),
             "sensitivity_153_with_range_outlier":
                 np.array([r["sensitivity_153"] == "True" for r in rows]),
             "sensitivity_187_all_in_support": np.ones(len(rows), dtype=bool)}

    document = {"record_kind": "affinity_policy_versus_baseline",
                "question": ("whether a policy's mean log probability orders the cohort by "
                             "measured affinity better than a scorer that only counts "
                             "mutations, or one that copies the labelled library"),
                "row_bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                                  "pairing": "the same resampled clones score the policy and "
                                             "the baseline within a draw"},
                "populations": {}, "recorded_at": paths.utc_now()}

    for population, mask in masks.items():
        block = {"rows": int(mask.sum()), "baselines": {}, "per_checkpoint": []}
        target = affinity[mask]
        wt_samples, nn_samples = bootstrap_rows(target, wild_type[mask], neighbour[mask])
        block["baselines"] = {
            "negative_distance_to_wild_type": {
                "spearman": spearman(wild_type[mask], target), **interval(wt_samples)},
            "nearest_labelled_neighbour_class": {
                "spearman": spearman(neighbour[mask], target), **interval(nn_samples)}}
        for name in checkpoints:
            values = np.array([float(r[name]) for r in rows])[mask]
            policy_samples, wt_paired, nn_paired = bootstrap_rows(
                target, values, wild_type[mask], neighbour[mask])
            arm, rest = name.split("_seed")
            seed, update = rest.split("_u")
            block["per_checkpoint"].append({
                "arm": arm, "seed": int(seed), "update": int(update),
                "spearman": spearman(values, target),
                **{f"policy_{k}": v for k, v in interval(policy_samples).items()},
                "over_wild_type_distance": {
                    "difference": spearman(values, target)
                                  - block["baselines"]["negative_distance_to_wild_type"]["spearman"],
                    **interval(policy_samples - wt_paired)},
                "over_nearest_neighbour": {
                    "difference": spearman(values, target)
                                  - block["baselines"]["nearest_labelled_neighbour_class"]["spearman"],
                    **interval(policy_samples - nn_paired)}})
        document["populations"][population] = block

    paths.write_json(out / "affinity_vs_baseline.json", document)

    primary = document["populations"]["primary_152"]
    print("primary 152 rows\n")
    for name, entry in primary["baselines"].items():
        print("  baseline %-34s %+.4f [%+.4f, %+.4f]" % (
            name, entry["spearman"], entry["lower"], entry["upper"]))
    print("\n%-16s %6s %5s %8s %26s %26s" % (
        "arm", "seed", "upd", "rho", "over WT-distance", "over nearest-neighbour"))
    for entry in primary["per_checkpoint"]:
        wt, nn = entry["over_wild_type_distance"], entry["over_nearest_neighbour"]
        print("%-16s %6d %5d %+8.4f   %+.4f [%+.4f, %+.4f]   %+.4f [%+.4f, %+.4f]" % (
            entry["arm"], entry["seed"], entry["update"], entry["spearman"],
            wt["difference"], wt["lower"], wt["upper"],
            nn["difference"], nn["lower"], nn["upper"]))
    beaten = sum(1 for e in primary["per_checkpoint"]
                 if e["over_wild_type_distance"]["lower"] > 0)
    print(f"\n{beaten}/{len(primary['per_checkpoint'])} checkpoints clear the "
          "wild-type-distance baseline with an interval excluding zero")
    print(f"wrote {out / 'affinity_vs_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
