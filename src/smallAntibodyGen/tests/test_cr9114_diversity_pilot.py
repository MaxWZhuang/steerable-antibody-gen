"""Fresh measurements and fixed quality/diversity acceptance gates."""
import copy
import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("cr9114_diversity_pilot", SCRIPTS / "run_cr9114_diversity_pilot.py")
runner = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(runner)
finally:
    sys.path[:] = original_path


def test_fresh_cohort_excludes_prior_candidates_and_training_rows():
    records = pd.DataFrame({"genotype": [f"{i:016b}" for i in range(12)],
                            "split": ["train"] * 4 + ["development"] * 8})
    excluded = records.genotype.iloc[4:7].tolist()
    cohort = runner.fresh_cohort(records, excluded, 5, 1)
    assert set(cohort.genotype) == set(records.genotype.iloc[7:])
    assert set(cohort.split) == {"development"}
    pd.testing.assert_frame_equal(cohort, runner.fresh_cohort(records, excluded, 5, 1))


def test_fresh_cohort_rejects_test_measurements():
    with pytest.raises(ValueError, match="Test measurements"):
        runner.fresh_cohort(pd.DataFrame({"genotype": ["0" * 16], "split": ["test"]}), [], 1, 1)


def test_screen_requires_quality_and_actual_diversity_together():
    sft = {"development": {"top_k_mean": {"16": 9.5, "32": 9.4}},
           "diversity": {"unique_genotypes": 1000, "mean_pairwise_hamming_unbiased": 7.}}
    result = copy.deepcopy(sft)
    result["diversity"].update(unique_genotypes=800, mean_pairwise_hamming_unbiased=5.6 + 1e-12)
    assert runner.screen_against_sft(result, sft, .8)["passes"]
    for category, metric, value in (("quality", "16", 9.49), ("quality", "32", 9.39),
                                    ("diversity", "unique_genotypes", 799),
                                    ("diversity", "mean_pairwise_hamming_unbiased", 5.59)):
        bad = copy.deepcopy(result)
        target = bad["development"]["top_k_mean"] if category == "quality" else bad["diversity"]
        target[metric] = value
        assert not runner.screen_against_sft(bad, sft, .8)["passes"]
