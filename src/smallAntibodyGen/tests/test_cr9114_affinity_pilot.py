"""Top-candidate reporting and the continued-training comparison gate."""
import importlib.util
from pathlib import Path
import sys

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("cr9114_affinity_pilot", SCRIPTS / "run_cr9114_affinity_pilot.py")
runner = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(runner)
finally:
    sys.path[:] = original_path


def metrics(a, b):
    return {"development": {"top_k_mean": {"16": a, "32": b}}}


def test_comparison_requires_no_regression_and_at_least_one_improvement():
    control = metrics(9.5, 9.4)
    assert runner.control_comparison(metrics(9.6, 9.4), control)["passes"]
    assert not runner.control_comparison(control, control)["passes"]
    assert not runner.control_comparison(metrics(9.6, 9.3), control)["passes"]
    assert not runner.control_comparison(metrics(9.4, 9.5), control)["passes"]


def test_selection_uses_scores_with_identity_ties_not_affinity_for_ranking():
    gs = [f"{i:016b}" for i in range(48)]
    records = pd.DataFrame({"genotype": gs, "mean": range(48), "effective_sem": .1,
                            "block": [i // 16 for i in range(48)]})
    scores = pd.Series(0., index=gs)
    baseline = pd.Series(range(48), index=gs)
    result = runner.selection_details(records, scores, baseline)
    assert result["16"]["selected_genotypes"] == gs[:16]
    assert result["16"]["overlap_with_sft"] == 0
    assert result["16"]["mean_minus_effective_sem"] == 7.4
    assert result["16"]["leave_one_block_out_top_k_mean"]["0"] == 23.5
