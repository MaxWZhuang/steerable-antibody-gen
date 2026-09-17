"""Matched weighted schedules and immediate-parent evaluation for DPO pilots."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("cr9114_dpo_pilot", SCRIPTS / "run_cr9114_dpo_pilot.py")
runner = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(runner)
finally:
    sys.path[:] = original_path


def test_weighted_schedule_is_reproducible_and_uses_weights_once():
    pairs = pd.DataFrame({"split": ["train"] * 3, "pair_weight": [0.1, 0.2, 0.7]})
    schedule = runner.training_schedule(pairs, 25000, 4, 42)
    np.testing.assert_array_equal(schedule, runner.training_schedule(pairs, 25000, 4, 42))
    np.testing.assert_allclose(np.bincount(schedule.ravel()) / schedule.size, [0.1, 0.2, 0.7], atol=0.004)


@pytest.mark.parametrize("weights", [[0.1, 0.2], [-0.1, 1.1], [float("nan"), 1], [0, 1]])
def test_invalid_pair_sampling_weights_rejected(weights):
    pairs = pd.DataFrame({"split": ["train"] * 2, "pair_weight": weights})
    with pytest.raises(ValueError, match="weights"):
        runner.training_schedule(pairs, 10, 2, 42)


def test_development_pairs_cannot_enter_training_schedule():
    with pytest.raises(ValueError, match="training pairs only"):
        runner.training_schedule(pd.DataFrame({"split": ["development"], "pair_weight": [1]}), 2, 1, 42)


def test_evaluation_compares_explicit_initial_policy_not_saved_parent_column():
    genotypes = [f"{i:016b}" for i in range(4)]
    scores = pd.DataFrame({"genotype": genotypes, "split": ["development"] * 4,
                           "h1_mean_recomputed": [8, 9, 8.2, 9.2],
                           "parent_log_q": [-4, -3, -2, -1],
                           "final_log_q": [-4, -3, -2, -1],
                           "additive_ridge_prediction": [8, 9, 8.2, 9.2]})
    pairs = pd.DataFrame({"chosen_genotype": [genotypes[1], genotypes[3]],
                          "rejected_genotype": [genotypes[0], genotypes[2]],
                          "split": ["development"] * 2, "block": [1, 2], "pair_weight": [0.5, 0.5]})
    result, scored = runner.development_evaluation(scores, pairs, {"score_tie_tolerance": 1e-6, "top_k": [1]},
                                                  np.array([-1, -2, -3, -4]), np.array([-4, -3, -2, -1]))
    assert result["initial_policy"]["variant_and_block_balanced_accuracy"] == 0
    assert result["adapted_policy"]["variant_and_block_balanced_accuracy"] == 1
    assert result["adapted_minus_initial_pp"] == 100
    assert "initial_correct" in scored and "adapted_correct" in scored
