"""Preference isolation, uncertainty regularization, and balanced scoring contracts."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("cr9114_preferences", SCRIPTS / "prepare_cr9114_preferences.py")
preferences = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    spec.loader.exec_module(preferences)
finally:
    sys.path[:] = original_path


def settings():
    return {"seed": 42, "maximum_pairs_per_variant": 3, "matching_rounds": 32,
            "minimum_mean_gap": 0.1, "uncertainty_multiplier": 2.0,
            "score_tie_tolerance": 1e-6, "top_k": [2]}


def candidates():
    return pd.DataFrame({"genotype": [f"{i:016b}" for i in range(16)],
                         "split": ["development"] * 16, "block": [1] * 8 + [2] * 8,
                         "mean": [8 + i * 0.25 for i in range(8)] * 2,
                         "effective_sem": [0.05] * 16})


def test_pairs_are_unique_capped_correctly_oriented_and_block_isolated():
    records = candidates()
    pairs, audit = preferences.construct_pairs(records, settings())
    lookup = records.set_index("genotype")
    unordered = [tuple(sorted(p)) for p in zip(pairs.chosen_genotype, pairs.rejected_genotype)]
    assert len(unordered) == len(set(unordered))
    for row in pairs.itertuples():
        assert lookup.loc[row.chosen_genotype, "block"] == lookup.loc[row.rejected_genotype, "block"]
        assert row.chosen_mean > row.rejected_mean
        assert row.mean_gap > 2 * row.combined_effective_sem
    degrees = pd.concat([pairs.chosen_genotype, pairs.rejected_genotype]).value_counts()
    assert degrees.max() <= 3
    assert audit["maximum_variant_degree"] == degrees.max()
    assert pairs.pair_weight.sum() == pytest.approx(1.0)
    assert pairs.groupby("block").pair_weight.sum().tolist() == pytest.approx([0.5, 0.5])


def test_pair_construction_is_independent_of_input_order():
    forward, _ = preferences.construct_pairs(candidates(), settings())
    backward, _ = preferences.construct_pairs(candidates().iloc[::-1], settings())
    pd.testing.assert_frame_equal(forward, backward)


@pytest.mark.parametrize("splits", [["train", "development"], ["test", "test"]])
def test_cross_split_or_reserved_test_candidates_are_rejected(splits):
    records = candidates().iloc[:2].copy()
    records["split"] = splits
    with pytest.raises(ValueError, match="non-test split"):
        preferences.construct_pairs(records, settings())


def test_training_only_variance_floor_protects_zero_spread_replicates():
    records = pd.DataFrame({"split": ["train", "train", "development"],
                            "sample_variance": [0.04, 0.0, 0.0], "replicate_count": [3, 3, 3]})
    result, floor = preferences.fit_uncertainty(records)
    assert floor == pytest.approx(0.02)
    assert result.effective_sem.iloc[1] == pytest.approx(np.sqrt(0.02 / 3))
    records.loc[2, "sample_variance"] = 1000
    _, changed_floor = preferences.fit_uncertainty(records)
    assert changed_floor == floor


def test_no_pair_for_a_gap_inside_uncertainty_screen():
    records = candidates().iloc[:2].copy()
    records["effective_sem"] = 1.0
    with pytest.raises(ValueError, match="No pairs pass"):
        preferences.construct_pairs(records, settings())


def test_metrics_use_declared_weights_ties_and_shared_pairs():
    records = candidates()
    pairs, _ = preferences.construct_pairs(records, settings())
    scores = records.rename(columns={"mean": "h1_mean_recomputed"})
    scores["parent_log_q"] = 0.0  # all ties get half credit
    scores["final_log_q"] = scores.h1_mean_recomputed
    scores["additive_ridge_prediction"] = -scores.h1_mean_recomputed
    metrics, _ = preferences.evaluate_pairs(pairs, scores, settings())
    assert metrics["parent"]["variant_and_block_balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["sft"]["variant_and_block_balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["additive_ridge"]["variant_and_block_balanced_accuracy"] == 0
    assert metrics["sft_minus_parent_pp"] == pytest.approx(50)
    assert metrics["sft_mathematical_headroom_pp"] == pytest.approx(0)
    assert metrics["measured_top_k"]["2"]["models"]["sft"]["regret_to_same_pool_oracle"] == 0


def test_test_rows_cannot_enter_replicate_statistics():
    with pytest.raises(ValueError, match="Reserved"):
        preferences.replicate_statistics(pd.DataFrame({"split": ["test"]}))
