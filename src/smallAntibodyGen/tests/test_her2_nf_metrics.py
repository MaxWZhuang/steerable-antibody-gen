"""Ranking, yield, tails and uncertainty -- checked against the installed library.

The scikit-learn cross-check runs against whatever is actually installed rather
than against a remembered agreement. A divergence here is a defect in one of the
two implementations, not a convention difference: threshold-grouped AP and
``average_precision_score`` are the same statistic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_metrics as metrics

sklearn = pytest.importorskip("sklearn", reason="the AP cross-check runs against the installed "
                                               "library, never against a remembered agreement")


def test_average_precision_agrees_with_the_installed_sklearn_on_tie_heavy_cases():
    generator = np.random.default_rng(3)
    worst = 0.0
    for _ in range(80):
        size = int(generator.choice([2, 10, 100, 1000]))
        labels = generator.integers(0, 2, size)
        if labels.sum() == 0:
            labels[0] = 1
        scores = generator.integers(0, 3, size).astype(float)
        block = metrics.average_precision_crosscheck(scores, labels)
        assert block["sklearn"]["available"] is True
        assert block["agrees"], block
        worst = max(worst, block["difference"])
    assert worst < 1e-12


def test_the_two_apis_take_their_arguments_in_different_orders():
    from sklearn.metrics import average_precision_score
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.3, 0.3])
    assert metrics.average_precision(scores, labels) == pytest.approx(
        float(average_precision_score(labels, scores)), abs=1e-12)


def test_a_constant_scorer_returns_the_prevalence_not_one_half():
    labels = np.array([1, 0, 0])
    assert metrics.average_precision(np.zeros(3), labels) == pytest.approx(1 / 3)
    assert metrics.auroc(np.zeros(3), labels) == pytest.approx(0.5)


def test_rank_block_reports_counts_and_prevalence():
    block = metrics.rank_block(np.array([3.0, 1.0, 2.0, 0.0]), np.array([1, 0, 1, 0]))
    assert block["positives"] == 2 and block["rows"] == 4
    assert block["prevalence"] == pytest.approx(0.5)
    assert block["auroc"] == pytest.approx(1.0)


def test_stratified_block_names_what_it_averaged_over():
    scores = np.arange(9, dtype=float)
    labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    strata = np.array(["1"] * 3 + ["2"] * 3 + [">=3"] * 3)
    block = metrics.stratified_rank_block(scores, labels, strata,
                                          categories=("1", "2", ">=3"))
    assert block["macro_over"] == ["1", "2", ">=3"]
    assert block["macro_average_precision"] is not None
    assert "not comparable" in block["macro_note"]


def test_an_empty_stratum_is_a_row_with_a_reason_not_a_dropped_row():
    block = metrics.stratified_rank_block(np.array([1.0, 2.0]), np.array([1, 0]),
                                          np.array(["1", "1"]), categories=("1", "2"))
    assert block["per_stratum"]["2"]["rows"] == 0
    assert "no rows" in block["per_stratum"]["2"]["reason"]


def test_expected_distinct_yield_keeps_precision_at_both_ends():
    # q -> 1: every draw finds it, so Y -> 1 per identity.
    assert metrics.expected_distinct_yield([-1e-12], 10) == pytest.approx(1.0, abs=1e-9)
    # q -> 0: Y ~= N*q and the naive 1 - (1-q)^N would round to zero.
    tiny = -40.0
    expected = 1e6 * math.exp(tiny)
    assert metrics.expected_distinct_yield([tiny], 1_000_000) == pytest.approx(expected,
                                                                               rel=1e-6)
    assert metrics.expected_distinct_yield([-2.0], 0) == 0.0


def test_yield_is_defined_on_a_fixed_identity_set():
    block = metrics.yield_curve([-3.0, -4.0, -5.0])
    assert block["identities"] == 3
    assert "incomparable absolute counts" in block["panel_note"]


def test_the_concavity_bound_is_labelled_as_a_bound():
    block = metrics.mixture_yield_bound(451.3, 573.3, 0.89)
    assert block["lower_bound"] == pytest.approx(0.11 * 451.3 + 0.89 * 573.3)
    assert block["status"].startswith("analytic lower bound")


def test_crossovers_retain_the_initial_sign_and_every_sign_change():
    budgets = [1e3, 1e4, 1e5, 1e6, 3e6]
    differences = [2.0, 0.5, -0.5, -0.1, 0.4]
    block = metrics.crossing_brackets(budgets, differences)
    assert block["first_nonzero_sign"] == 1
    assert len(block["brackets"]) == 2
    assert block["domain"] == [1e3, 3e6]
    assert "always better" in block["interpretation"]


def test_a_numerical_tie_is_skipped_explicitly_rather_than_reported_as_a_crossing():
    block = metrics.crossing_brackets([1.0, 2.0, 3.0], [1.0, 0.0, 1.0], atol=1e-9)
    assert block["brackets"] == []
    assert block["skipped_numerical_ties"] == 1


def test_bracket_refinement_narrows_without_claiming_uniqueness():
    def difference(n):
        return 100.0 - float(n)

    bracket = {"left": 1.0, "right": 1000.0, "from_sign": 1, "to_sign": -1}
    block = metrics.refine_bracket(bracket, difference, iterations=25)
    low, high = block["refined"]
    assert low <= 100.0 <= high + 1
    assert high - low < 900
    assert "not a proof" in block["claim"]


def test_wilson_is_two_sided_and_zero_events_are_not_zero_width():
    block = metrics.wilson_interval(0, 50_000)
    assert block["lower"] == 0.0
    assert block["upper"] == pytest.approx(7.68e-5, rel=0.05)
    assert "two-sided" in block["kind"]


def test_worst_upper_refuses_to_be_called_simultaneous():
    block = metrics.worst_upper([metrics.wilson_interval(370, 50_000),
                                 metrics.wilson_interval(401, 50_000)])
    assert block["worst_upper"] == pytest.approx(
        metrics.wilson_interval(401, 50_000)["upper"])
    assert "NOT a simultaneous" in block["claim"]


def test_paired_t_uses_df_two_for_three_seeds_and_says_what_it_cannot_say():
    block = metrics.paired_t([-132.0, -66.0, 0.84])
    assert block["degrees_of_freedom"] == 2
    assert block["t_critical"] == pytest.approx(4.302652729911275)
    assert block["excludes_zero"] is False
    assert "not equivalence" in block["power_note"]


def test_paired_bootstrap_is_labelled_a_different_question():
    generator = np.random.default_rng(1)
    a = generator.normal(size=500)
    b = a + 0.2
    block = metrics.paired_bootstrap(a, b, draws=200, seed=2)
    assert block["mean"] == pytest.approx(-0.2, abs=0.02)
    assert "not" in block and "extra models" in block["not"]


def test_drop_block_reports_inclusive_tails_modest_losses_and_the_estimator_caveat():
    parent = np.zeros(1000)
    policy = np.concatenate([np.full(10, -math.log(10.0)), np.full(5, -math.log(100.0)),
                             np.full(985, -0.1)])
    block = metrics.drop_block(parent, policy)
    assert block["tails"]["events"]["tenfold"]["count"] == 15      # inclusive: ln100 > ln10 too
    assert block["tails"]["events"]["hundredfold"]["count"] == 5
    assert block["modest_loss_rates"]["gt_0"] == pytest.approx(1.0)
    assert "estimates KL(P||Q)" in block["estimator_note"]
    assert set(block["wilson"]) == {"tenfold", "hundredfold"}


def test_a_nonfinite_score_is_an_observation_and_stops_the_block():
    with pytest.raises(ValueError, match="nonfinite"):
        metrics.drop_block(np.zeros(3), np.array([-1.0, -np.inf, -2.0]))


def test_class_mass_keeps_its_denominators_apart():
    block = metrics.class_mass(["a", "b", "c", "d"],
                               class_of={"a": "high", "b": "high", "c": "low"},
                               denominator_label="10k screen")
    assert block["draws"] == 4 and block["assayed_hits"] == 3 and block["unassayed"] == 1
    assert block["mass_over_all_draws"]["high"] == pytest.approx(0.5)
    assert block["purity_over_assayed_hits"]["high"] == pytest.approx(2 / 3)
    assert "never multiplied together" in block["denominator_note"]
