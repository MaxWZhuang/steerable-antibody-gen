#!/usr/bin/env python
"""Recompute shortlist choices and metrics without importing the selector."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_cr9114_esmif1_pilot import require, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def independent_selection(scores, k, multiplier):
    ranked = sorted(zip(scores.genotype, scores.score), key=lambda pair: (-pair[1], pair[0]))[:k * multiplier]
    if multiplier == 1:
        return [g for g, _ in ranked[:k]]
    chosen = [ranked[0][0]]
    while len(chosen) < k:
        candidates = [pair for pair in ranked if pair[0] not in chosen]
        best = min(candidates, key=lambda pair: (-min(sum(x != y for x, y in zip(pair[0], g)) for g in chosen), -pair[1], pair[0]))
        chosen.append(best[0])
    return chosen


def independent_distances(ids):
    return [sum(a != b for a, b in zip(x, y)) for x, y in itertools.combinations(ids, 2)]


def audit(directory):
    result = json.loads((directory / "results.json").read_text())
    config = result["config"]
    require(result["status"] == "completed" and not result["reserved_test_labels_evaluated"], "Invalid run")
    for relative, digest in result["output_sha256"].items():
        require(sha256(directory / relative) == digest, f"Changed artifact {relative}")
    records_path = ROOT / "outputs/cr9114_preferences_verified_20260916/eligible_non_test_records.csv"
    require(sha256(records_path) == result["input_records_sha256"], "Records changed")
    records = pd.read_csv(records_path, dtype={"genotype": "string"})
    calibration = pd.read_csv(directory / "calibration.csv", dtype={"genotype": "string"})
    fresh = pd.read_csv(directory / "fresh_development.csv", dtype={"genotype": "string"})
    scores = pd.read_csv(directory / "development_scores.csv", dtype={"genotype": "string"})
    require(set(calibration.split) == {"train"} and set(fresh.split) == {"development"}, "Crossed splits")
    train = records[records.split == "train"]
    threshold = float(train["mean"].quantile(config["quality_threshold_training_quantile"]))
    require(threshold == result["calibration"]["quality_threshold"], "Threshold changed")
    expected_train = sorted(train.genotype, key=lambda g: hashlib.sha256(f"{config['calibration_seed']}:{g}".encode()).hexdigest())[:len(calibration)]
    require(calibration.genotype.tolist() == expected_train, "Calibration cohort changed")
    np.testing.assert_array_equal(calibration.calibration_cohort, np.arange(len(calibration)) // config["calibration_count_per_cohort"])
    excluded = set()
    for path, digest in result["prior_cohort_sha256"].items():
        require(sha256(ROOT / path) == digest, "Prior cohort changed")
        excluded.update(pd.read_csv(ROOT / path, dtype={"genotype": "string"}).genotype)
    candidates = records[(records.split == "development") & ~records.genotype.isin(excluded)]
    expected_fresh = sorted(candidates.genotype, key=lambda g: hashlib.sha256(f"{config['fresh_development_seed']}:{g}".encode()).hexdigest())[:config["fresh_development_count"]]
    require(fresh.genotype.tolist() == scores.genotype.tolist() == expected_fresh, "Development cohort changed")
    for frame in (calibration, fresh):
        source = records.set_index("genotype").loc[frame.genotype]
        for column in ("mean", "effective_sem", "block"):
            np.testing.assert_allclose(frame[column], source[column], rtol=1e-12, atol=1e-12)
    summaries = []
    for option in result["calibration"]["options"]:
        deltas, gains = [], []
        for cohort, group in calibration.groupby("calibration_cohort"):
            indexed = group.set_index("genotype")
            for k in config["budgets"]:
                base = independent_selection(group, k, 1)
                ids = independent_selection(group, k, option["multiplier"])
                deltas.append(float(indexed.loc[ids, "mean"].mean() - indexed.loc[base, "mean"].mean()))
                gains.append(float(np.mean(independent_distances(ids)) - np.mean(independent_distances(base))))
        np.testing.assert_allclose(deltas, [cell["affinity_delta"] for cell in option["cells"]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(gains, [cell["hamming_delta"] for cell in option["cells"]], atol=1e-12, rtol=0)
        passes = all(delta >= -config["affinity_loss_tolerance"] for delta in deltas)
        require(passes == option["passes_quality"], "Calibration quality screen differs")
        if passes: summaries.append((float(np.mean(gains)), -option["multiplier"]))
    selected_multiplier = -max(summaries)[1]
    require(selected_multiplier == result["calibration"]["selected_multiplier"], "Wrong calibration winner")
    screen = {}
    for name, multiplier in (("ordinary", 1), ("diverse", selected_multiplier)):
        for k in config["budgets"]:
            ids = independent_selection(scores, k, multiplier)
            measured = result["portfolios"][name][str(k)]
            require(ids == measured["selected_genotypes"], "Selected identities differ")
            subset = fresh.set_index("genotype").loc[ids]
            distances = np.array(independent_distances(ids))
            np.testing.assert_allclose(measured["mean_affinity"], subset["mean"].mean(), atol=1e-12, rtol=0)
            np.testing.assert_allclose(measured["diversity"]["mean_hamming"], distances.mean(), atol=1e-12, rtol=0)
            require(measured["diversity"]["minimum_hamming"] == int(distances.min()), "Wrong minimum distance")
            require(measured["quality_qualified_count"] == int((subset["mean"] >= threshold).sum()), "Wrong quality count")
    for k in map(str, config["budgets"]):
        b, d = result["portfolios"]["ordinary"][k], result["portfolios"]["diverse"][k]
        screen[k] = (d["mean_affinity"] >= b["mean_affinity"] - config["affinity_loss_tolerance"]
                     and d["diversity"]["mean_hamming"] >= b["diversity"]["mean_hamming"] + config["minimum_mean_hamming_gain"]
                     and d["diversity"]["hamming_le_one_pair_fraction"] <= b["diversity"]["hamming_le_one_pair_fraction"])
    require(all(screen.values()) == result["screen"]["passes"], "Screen differs")
    evidence = {"passes": True, "artifacts_checked": len(result["output_sha256"]),
                "calibration_recomputed": True, "selections_recomputed": True, "fresh_identity_exclusion_verified": True,
                "metrics_recomputed": True, "screen_recomputed": True, "reserved_test_labels_evaluated": False}
    save_json(directory / "independent_audit.json", evidence)
    print(json.dumps(evidence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/cr9114_shortlist_20260917")
    audit(parser.parse_args().directory)
