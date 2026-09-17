"""Pilot data rules: sequence-only blocks and conservative replicate eligibility."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location(
    "cr9114_pilot", Path(__file__).resolve().parents[3] / "scripts/run_cr9114_esmif1_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_block_split_is_order_invariant_and_keeps_whole_combinations():
    config = {"seed": 20260916, "split_loci_count": 4, "train_blocks": 10, "development_blocks": 3}
    genotypes = [f"{i:016b}" for i in range(65536)]
    split, info = pilot.split_genotypes(genotypes, config)
    reversed_split, reversed_info = pilot.split_genotypes(genotypes[::-1], config)
    assert reversed_split[::-1] == split
    assert reversed_info == info
    assert {s: split.count(s) for s in set(split)} == {
        "train": 40960, "development": 12288, "test": 12288}
    groups = {}
    for genotype, assignment in zip(genotypes, split):
        block = tuple(genotype[i] for i in info["loci_0based"])
        groups.setdefault(block, set()).add(assignment)
    assert all(len(assignments) == 1 for assignments in groups.values())


def test_replicates_recomputed_and_censored_or_single_observations_excluded():
    frame = pd.DataFrame({
        "genotype": [f"{i:016b}" for i in range(5)], "split": ["train"] * 5,
        "h1_repa": [8.0, 8.0, 7.0, np.nan, 8.0],
        "h1_repb": [9.0, np.nan, 9.0, np.nan, 8.0],
        "h1_repc": [np.nan, np.nan, 9.0, np.nan, 8.0],
        "h1_mean": [999.0] * 5, "h1_sem": [0.0] * 5,
    })
    eligible = pilot.eligible_records(frame)
    assert eligible.genotype.tolist() == ["0000000000000000", "0000000000000100"]
    np.testing.assert_allclose(eligible.h1_mean_recomputed, [8.5, 8.0])
    np.testing.assert_allclose(eligible.h1_sem_recomputed, [0.5, 0.0])


def test_scoring_subset_does_not_depend_on_labels_or_input_order():
    frame = pd.DataFrame({"genotype": [f"{i:016b}" for i in range(64)],
                          "h1_mean_recomputed": np.arange(64)})
    first = pilot.stable_subset(frame, 12, 42).genotype.tolist()
    frame["h1_mean_recomputed"] = -999
    second = pilot.stable_subset(frame.iloc[::-1], 12, 42).genotype.tolist()
    assert first == second
