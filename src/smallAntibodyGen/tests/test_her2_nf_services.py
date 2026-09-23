"""The runtime's decision helpers, on CPU, without the dataset or the weights.

These are the pieces of :mod:`her2_nf_services` that decide what gets audited,
what counts as feasible across seeds, and what a class mass is. Each of them was
wrong in a way that produced a plausible number rather than an error, which is
exactly the class of defect a native run does not surface.

The native fixtures exercise the rest; see ``test_her2_nf_native.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_services as services


def _metrics(ap, *, tenfold=0.002, hundredfold=0.0002, y10k=500.0, y1m=5000.0):
    return {"macro_average_precision": ap, "tenfold_rate": tenfold,
            "hundredfold_rate": hundredfold, "yield_10k": y10k, "yield_1m": y1m}


SEEDS = [20260918, 20260919, 20260920]


def test_a_one_seed_family_is_not_all_seed_feasible():
    """Reproduced: an IPO-tail family carrying only seed 18 was accepted.

    ``all()`` over a one-element mapping is ``True``, so a family whose other two
    seeds stopped or failed read as feasible in all seeds.
    """
    choice = services._choose_tail_family(
        {"ipo_tail": {20260918: _metrics(0.9, tenfold=0.005, hundredfold=0.0005)},
         "dpo_tail": {}},
        declared_seeds=SEEDS)
    assert choice["summary"]["ipo_tail"]["feasible"] is False
    assert choice["summary"]["ipo_tail"]["missing_seeds"] == [20260919, 20260920]
    assert choice["summary"]["ipo_tail"]["all_seed_results"] is False
    assert "stops or failures and stay visible" in choice["summary"]["ipo_tail"]["reason"]
    assert choice["feasible"] is False, "audited under the fallback rule, not as feasible"


def test_all_three_declared_seeds_present_and_feasible_is_all_seed_feasible():
    choice = services._choose_tail_family(
        {"ipo_tail": {seed: _metrics(0.88) for seed in SEEDS},
         "dpo_tail": {seed: _metrics(0.87) for seed in SEEDS}},
        declared_seeds=SEEDS)
    assert choice["summary"]["ipo_tail"]["feasible"] is True
    assert choice["summary"]["ipo_tail"]["missing_seeds"] == []
    assert choice["family"] == "ipo_tail"
    assert choice["feasible"] is True
    assert "all-seed point-rate feasibility" in choice["rule"]


def test_one_infeasible_seed_disqualifies_the_whole_family():
    choice = services._choose_tail_family(
        {"ipo_tail": {20260918: _metrics(0.9), 20260919: _metrics(0.9, tenfold=0.5),
                      20260920: _metrics(0.9)},
         "dpo_tail": {seed: _metrics(0.86) for seed in SEEDS}},
        declared_seeds=SEEDS)
    assert choice["summary"]["ipo_tail"]["feasible"] is False
    assert choice["summary"]["dpo_tail"]["feasible"] is True
    assert choice["family"] == "dpo_tail"


def test_neither_family_feasible_minimizes_the_normalized_violation_and_says_so():
    choice = services._choose_tail_family(
        {"ipo_tail": {seed: _metrics(0.9, tenfold=0.5) for seed in SEEDS},
         "dpo_tail": {seed: _metrics(0.85, tenfold=0.02) for seed in SEEDS}},
        declared_seeds=SEEDS)
    assert choice["feasible"] is False
    assert choice["family"] == "dpo_tail", "the smaller maximum normalized violation"
    assert "neither family is feasible" in choice["rule"]
    assert "retained and reported" in choice["retained"]


def test_the_ap_tie_is_broken_by_yield_then_by_ipo_tail():
    tie = services._choose_tail_family(
        {"ipo_tail": {seed: _metrics(0.8700, y10k=500.0) for seed in SEEDS},
         "dpo_tail": {seed: _metrics(0.8705, y10k=400.0) for seed in SEEDS}},
        declared_seeds=SEEDS)
    assert tie["family"] == "ipo_tail" and tie["tie_broken_by"] == "Y@10k"
    exact = services._choose_tail_family(
        {"ipo_tail": {seed: _metrics(0.87, y10k=500.0, y1m=5000.0) for seed in SEEDS},
         "dpo_tail": {seed: _metrics(0.87, y10k=500.0, y1m=5000.0) for seed in SEEDS}},
        declared_seeds=SEEDS)
    assert exact["family"] == "ipo_tail" and exact["tie_broken_by"] == "IPO-tail"


def test_class_mass_is_a_sum_of_exponentials_not_a_hit_rate():
    """Exact mass on each class of the declared assayed panel."""
    scores = np.log(np.array([0.10, 0.20, 0.05, 0.01, 0.04]))
    classes = np.array(["high", "high", "mid", "low", "low"])
    block = services._class_mass(scores, classes)
    assert block["high"]["mass"] == pytest.approx(0.30)
    assert block["mid"]["mass"] == pytest.approx(0.05)
    assert block["low"]["mass"] == pytest.approx(0.05)
    assert block["high"]["rows"] == 2
    assert block["_panel"]["mass"] == pytest.approx(0.40)
    assert block["_panel"]["purity"]["high"] == pytest.approx(0.75)
    assert "not a draw-hit rate" in block["_panel"]["definition"]


def test_the_difference_in_differences_needs_all_four_cells():
    missing = services._difference_in_differences({"DPO_FKL@purge": {1: 0.9}})
    assert missing["available"] is False
    assert set(missing["missing"]) == {"IPO_FKL@purge", "DPO_FKL@matched", "IPO_FKL@matched"}
    per_arm = {"DPO_FKL@purge": {1: 0.90, 2: 0.88, 3: 0.89},
               "IPO_FKL@purge": {1: 0.87, 2: 0.86, 3: 0.86},
               "DPO_FKL@matched": {1: 0.91, 2: 0.90, 3: 0.90},
               "IPO_FKL@matched": {1: 0.87, 2: 0.87, 3: 0.86}}
    block = services._difference_in_differences(per_arm)
    assert block["available"] is True
    assert block["paired_seeds"] == [1, 2, 3]
    assert block["degrees_of_freedom"] == 2
    assert block["mean"] == pytest.approx(np.mean([0.03 - 0.04, 0.02 - 0.03, 0.03 - 0.04]))
    assert "collapsing after neighbourhood removal" in block["reading"]


def test_the_slug_keeps_the_model_readable_and_the_path_safe():
    assert services._slug("A_IPO_FKL@u1000_seed20260918") == "A_IPO_FKL_u1000_seed20260918"
    assert services._slug("B_purge_DPO_FKL@u1000") == "B_purge_DPO_FKL_u1000"


def test_high_versus_one_named_class_reports_an_absent_class_rather_than_a_number():
    classes = np.array(["high", "high", "low"])
    block = services._versus(np.array([1.0, 2.0, 0.5]), classes, "mid")
    assert block["average_precision"] is None
    assert "no mid rows" in block["reason"]
    present = services._versus(np.array([1.0, 2.0, 0.5]), classes, "low")
    assert present["average_precision"] is not None


def test_the_diversity_block_retains_duplicates_and_reports_them():
    index = np.zeros((10, 10), dtype=np.int8)
    index[5:] = 1
    block = services._diversity_block(index)
    assert block["rows"] == 10
    assert block["unique_cores"] == 2
    assert block["unique_fraction"] == pytest.approx(0.2)
    assert block["max_single_core_frequency"] == pytest.approx(0.5)
    assert len(block["site_entropy_nats"]) == 10
    assert "duplicates are retained and never filtered" in block["basis"]
