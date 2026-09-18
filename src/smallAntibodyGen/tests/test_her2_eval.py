"""Metrics, bootstrap point estimates, generation diagnostics, diversity gates, selection.

Pure numpy; no model, no GPU, no download.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_eval as evaluation

SCORES_A = np.array([0.8, 0.7, 0.1, 0.2])
SCORES_B = np.array([0.1, 0.2, 0.8, 0.3])
LABELS = np.array([1, 0, 1, 0])
KEYS = np.array(["a", "b", "c", "d"])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_a_constant_scorer_gets_the_prevalence_not_one_half():
    """At 50% prevalence the AP null and the AUROC null are both 0.5, so that
    fixture could not tell the two constants apart. Use 25% prevalence, where the
    right answer (0.25) and the wrong one (0.5) are different numbers."""
    labels = np.array([1, 0, 0, 0])
    constant = np.full(4, 0.5)
    assert evaluation.average_precision(constant, labels) == pytest.approx(0.25)
    assert evaluation.auroc(constant, labels) == pytest.approx(0.5)
    document = evaluation.rank_metrics(constant, labels, KEYS, k_values=(2,))
    assert document["constant_scorer"] is True
    assert document["prevalence"] == pytest.approx(0.25)
    assert document["average_precision"] == pytest.approx(0.25)


def test_auroc_is_undefined_with_a_single_class():
    assert evaluation.auroc(SCORES_A, np.ones(4)) is None
    document = evaluation.rank_metrics(SCORES_A, np.ones(4), KEYS, k_values=(2,))
    assert document["auroc"] is None
    assert document["auroc_note"] == evaluation.AUROC_UNDEFINED
    assert document["average_precision"] == pytest.approx(1.0)


def test_precision_at_k_breaks_ties_deterministically():
    tied = np.array([1.0, 1.0, 1.0, 1.0])
    first = evaluation.precision_at_k(tied, LABELS, KEYS, 2)
    second = evaluation.precision_at_k(tied, LABELS, KEYS[::-1], 2)
    assert first == pytest.approx(0.5)
    assert second == pytest.approx(0.5)
    assert evaluation.precision_at_k(tied, LABELS, KEYS, 99) is None


def test_bootstrap_observed_is_the_point_difference_not_the_bootstrap_mean():
    """Both scorers rank one positive first and one last, so the AP difference is 0."""
    summary = evaluation.bootstrap_paired_differences(
        {"a": SCORES_A, "b": SCORES_B}, LABELS, KEYS, [("a", "b")], draws=20, seed=3)
    entry = summary["differences"]["a_minus_b"]["average_precision"]
    assert entry["observed"] == pytest.approx(0.0)
    assert entry["bootstrap_mean"] is not None
    assert entry["bootstrap_mean"] != entry["observed"]
    assert summary["draws_used"] <= 20


def test_bootstrap_reports_an_interval_and_whether_it_excludes_zero():
    summary = evaluation.bootstrap_paired_differences(
        {"a": SCORES_A, "b": SCORES_B}, LABELS, KEYS, [("a", "b")], draws=50, seed=5)
    entry = summary["differences"]["a_minus_b"]["auroc"]
    assert entry["ci_low"] <= entry["ci_high"]
    assert isinstance(entry["excludes_zero"], bool)


# ---------------------------------------------------------------------------
# generation diagnostics
# ---------------------------------------------------------------------------

def test_per_split_hits_separate_memorized_training_rows_from_generalization():
    drawn = ["AAAAAAAAAA", "AAAAAAAAAA", "CCCCCCCCCC", "DDDDDDDDDD", "EEEEEEEEEE"]
    split_of = {"AAAAAAAAAA": "train", "CCCCCCCCCC": "val", "DDDDDDDDDD": "test"}
    class_of = {"AAAAAAAAAA": "high", "CCCCCCCCCC": "low", "DDDDDDDDDD": "high"}
    document = evaluation.split_conditional_hits(drawn, split_of, class_of)
    assert document["by_split"]["train"]["draws_matching"] == 2
    assert document["by_split"]["train"]["conditional_high"]["rate"] == pytest.approx(1.0)
    assert document["by_split"]["val"]["conditional_high"]["rate"] == pytest.approx(0.0)
    assert document["heldout"]["draws_matching"] == 2
    assert document["heldout"]["conditional_high"]["rate"] == pytest.approx(0.5)
    assert document["pooled"]["draws_matching"] == 4
    # the pooled rate is higher than the held-out one precisely because of the
    # memorized training positives, which is the confusion this split-out prevents
    assert document["pooled"]["conditional_high"]["rate"] > \
        document["heldout"]["conditional_high"]["rate"]


def test_zero_hits_are_reported_as_zero_with_an_interval():
    document = evaluation.split_conditional_hits(["AAAAAAAAAA"], {"CCCCCCCCCC": "val"},
                                                 {"CCCCCCCCCC": "high"})
    entry = document["by_split"]["val"]
    assert entry["draws_matching"] == 0
    assert entry["conditional_high"]["rate"] is None
    assert entry["hit_rate_over_all_draws"]["successes"] == 0
    assert entry["hit_rate_over_all_draws"]["ci_high"] > 0


def test_unlabelled_splits_are_counted_but_not_rated():
    """During validation the test split's sequences are known and its labels are not."""
    document = evaluation.split_conditional_hits(
        ["AAAAAAAAAA"], {"AAAAAAAAAA": "test"}, {})
    entry = document["by_split"]["test"]
    assert entry["draws_matching"] == 1
    assert entry["draws_with_labels"] == 0
    assert entry["labels_available"] is False
    assert entry["conditional_high"]["rate"] is None


def test_clopper_pearson_handles_the_zero_and_full_branches():
    assert evaluation.clopper_pearson(0, 10)["ci_low"] == 0.0
    assert evaluation.clopper_pearson(10, 10)["ci_high"] == 1.0
    assert evaluation.clopper_pearson(0, 0)["rate"] is None


def test_generation_record_cannot_carry_an_affinity():
    with pytest.raises(ValueError, match="key set is closed"):
        evaluation.generation_record(kd_molar=1e-9)
    with pytest.raises(ValueError, match="Missing generation field"):
        evaluation.generation_record(draw_index=0, core="AAAAAAAAAA")
    record = evaluation.generation_record(
        draw_index=0, core="AAAAAAAAAA", sum_log_probability=-1.0, mean_log_probability=-0.1,
        classifier_proxy_p_high=0.2, library_split=None, library_class=None,
        min_train_hamming=2, wt_hamming=5)
    assert record["library_class"] is None
    with pytest.raises(ValueError, match="only defined for a draw that matches"):
        evaluation.generation_record(
            draw_index=0, core="AAAAAAAAAA", sum_log_probability=-1.0, mean_log_probability=-0.1,
            classifier_proxy_p_high=0.2, library_split=None, library_class="high",
            min_train_hamming=2, wt_hamming=5)


def test_monte_carlo_kl_may_come_out_negative_and_is_not_clamped():
    """Assert the sign, not just finiteness: a clamp to zero would pass 'is finite'.

    The estimator is a sample mean of ``log pi - log ref`` under the policy's own
    draws, so near KL 0 it lands on either side. Here the reference is uniformly
    the larger, which forces a negative estimate that must be reported as measured.
    """
    rng = np.random.default_rng(0)
    policy = rng.normal(size=500)
    reference = policy + np.abs(rng.normal(scale=0.1, size=500)) + 0.05
    document = evaluation.monte_carlo_kl(policy, reference)
    assert document["kl_nats"] < 0, "a negative estimate must survive, not be clamped to 0"
    assert document["kl_nats"] == pytest.approx(float(np.mean(policy - reference)))
    assert document["standard_error"] > 0
    assert document["draws"] == 500


# ---------------------------------------------------------------------------
# diversity gates
# ---------------------------------------------------------------------------

def diagnostics(**overrides):
    document = {"mean_pairwise_hamming": 8.0, "sum_site_entropy_nats": 20.0,
                "unique_fraction": 0.96, "max_single_core_frequency": 0.001,
                "exact_train_novelty": {"fraction_not_in_training_cores": 0.8}}
    document.update(overrides)
    return document


REFERENCE = {"mean_pairwise_hamming": 8.0, "sum_site_entropy_nats": 20.0, "unique_fraction": 0.96}


def test_a_healthy_policy_clears_every_gate():
    verdict = evaluation.diversity_eligibility(diagnostics(), training_reference=REFERENCE,
                                               parent_reference=REFERENCE)
    assert verdict["eligible"] is True
    assert verdict["failed_gates"] == []


def test_a_collapsed_policy_fails_every_gate_and_says_which():
    collapsed = diagnostics(mean_pairwise_hamming=0.5, sum_site_entropy_nats=1.0,
                            unique_fraction=0.05, max_single_core_frequency=0.7,
                            exact_train_novelty={"fraction_not_in_training_cores": 0.1})
    verdict = evaluation.diversity_eligibility(collapsed, training_reference=REFERENCE,
                                               parent_reference=REFERENCE)
    assert verdict["eligible"] is False
    assert set(verdict["failed_gates"]) == set(verdict["checks"])


def test_a_gate_must_clear_both_references_not_just_the_easier_one():
    easy = dict(REFERENCE, mean_pairwise_hamming=1.0, sum_site_entropy_nats=2.0)
    verdict = evaluation.diversity_eligibility(
        diagnostics(mean_pairwise_hamming=4.0, sum_site_entropy_nats=10.0),
        training_reference=REFERENCE, parent_reference=easy)
    assert verdict["eligible"] is False
    assert "mean_pairwise_hamming_vs_training" in verdict["failed_gates"]
    assert "mean_pairwise_hamming_vs_parent" not in verdict["failed_gates"]


# ---------------------------------------------------------------------------
# within-budget selection
# ---------------------------------------------------------------------------

def records():
    return {
        "b180": {"budget_seconds": 180.0, "eligible": True, "val_average_precision": 0.30},
        "b360": {"budget_seconds": 360.0, "eligible": True, "val_average_precision": 0.40},
        "b600": {"budget_seconds": 600.0, "eligible": False, "val_average_precision": 0.90},
    }


def test_selection_takes_the_best_eligible_checkpoint_within_the_budget():
    chosen = evaluation.select_within_budget(records(), budgets=[180, 360, 600])
    assert chosen["180"]["selected"] == "b180"
    assert chosen["360"]["selected"] == "b360"
    # b600 scores highest but failed the diversity gates, so the 600 s budget falls
    # back to the best eligible checkpoint at or below it, not to the ineligible one.
    assert chosen["600"]["selected"] == "b360"


def test_a_budget_with_nothing_eligible_is_recorded_as_a_failure_not_promoted():
    table = {"b180": {"budget_seconds": 180.0, "eligible": False,
                      "val_average_precision": 0.9}}
    chosen = evaluation.select_within_budget(table, budgets=[180])
    assert chosen["180"]["selected"] is None
    assert "no eligible checkpoint" in chosen["180"]["reason"]


def test_a_tie_is_broken_toward_the_earlier_budget():
    table = {"late": {"budget_seconds": 600.0, "eligible": True, "val_average_precision": 0.5},
             "early": {"budget_seconds": 180.0, "eligible": True, "val_average_precision": 0.5}}
    chosen = evaluation.select_within_budget(table, budgets=[600])
    assert chosen["600"]["selected"] == "early"


def test_a_missing_metric_is_not_selectable():
    table = {"b180": {"budget_seconds": 180.0, "eligible": True, "val_average_precision": None}}
    chosen = evaluation.select_within_budget(table, budgets=[180])
    assert chosen["180"]["selected"] is None
    assert chosen["180"]["checkpoints_considered"] == ["b180"]


def test_selection_compares_the_nominal_budget_and_reports_the_measured_one():
    """The measured time overshoots its own target by design; comparing it would
    exclude a checkpoint from the budget that defines it."""
    table = {"b180": {"budget_seconds": 180.0, "actual_gpu_seconds": 180.42, "eligible": True,
                      "val_average_precision": 0.3}}
    chosen = evaluation.select_within_budget(table, budgets=[180])
    assert chosen["180"]["selected"] == "b180"
    assert chosen["180"]["budget_seconds"] == 180.0
    assert chosen["180"]["actual_gpu_seconds"] == pytest.approx(180.42)
    assert "nominal" in chosen["180"]["budget_basis"]


# ---------------------------------------------------------------------------
# reference-relative ranking diagnostic
# ---------------------------------------------------------------------------

def test_the_implicit_reward_ranking_is_a_different_order_from_raw_density():
    """Subtracting the parent can reorder completely; that is why it is separate.

    Raw density ranks the two positives last here; the difference from the parent
    ranks them first. Reporting only one of the two would be a choice about what
    the number means, so both are reported and neither selects anything.
    """
    policy = np.array([-5.0, -4.0, -3.0, -2.0])
    parent = np.array([-9.0, -4.1, -3.0, -1.0])
    labels = np.array([1, 1, 0, 0])
    raw = evaluation.rank_metrics(policy, labels, KEYS, k_values=(1,))
    relative = evaluation.implicit_reward_metrics(policy, parent, labels, KEYS, k_values=(1,),
                                                  reference_name="policy_sft_seed1")
    assert raw["auroc"] == pytest.approx(0.0)
    assert raw["precision_at_k"]["1"] == pytest.approx(0.0)
    assert relative["auroc"] == pytest.approx(1.0)
    assert relative["precision_at_k"]["1"] == pytest.approx(1.0)
    assert relative["reference_name"] == "policy_sft_seed1"
    assert relative["scoring_rule"] == "policy_minus_reference_log_density"


def test_a_positive_beta_cannot_change_the_implicit_reward_ranking():
    policy = np.array([-5.0, -4.0, -3.0, -2.0])
    parent = np.array([-9.0, -4.1, -3.0, -1.0])
    labels = np.array([1, 1, 0, 0])
    plain = evaluation.implicit_reward_metrics(policy, parent, labels, KEYS, k_values=(1,))
    scaled = evaluation.rank_metrics(0.1 * (policy - parent), labels, KEYS, k_values=(1,))
    assert plain["auroc"] == pytest.approx(scaled["auroc"])
    assert plain["average_precision"] == pytest.approx(scaled["average_precision"])


def test_implicit_reward_metrics_refuse_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        evaluation.implicit_reward_metrics(np.zeros(4), np.zeros(3), LABELS, KEYS)
