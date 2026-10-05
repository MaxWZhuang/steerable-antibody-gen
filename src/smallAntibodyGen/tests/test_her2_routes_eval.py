"""I22 and the evaluation contract: panels, frozen clustering, gates, the guided sampler.

The frozen-clustering test is the one that protects a measurement rather than a value: a
clustering rebuilt at each draw count changes what "family coverage" means, so every arm
and every nested prefix must read the same assignment.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_decoding as decoding
from smallAntibodyGen.experiments import her2_routes_eval as eval_lib
from smallAntibodyGen.experiments import her2_routes_report as report_lib


# --- the panel envelope ---------------------------------------------------

def test_unsupported_panels_carry_a_reason_and_no_value():
    for panel in ("E12", "E20"):
        record = eval_lib.panel_record(panel, population=0, supported=False)
        assert record["supported"] is False
        assert record["blocking_reason"]
        assert record["value"] is None
    with pytest.raises(ValueError, match="cannot also report a value"):
        eval_lib.panel_record("E12", population=5, value=0.9, supported=False,
                              reason="stated")


def test_e12_makes_no_specificity_claim():
    assert "NO specificity claim is made" in eval_lib.UNSUPPORTED["E12"]
    assert "not automatically" in eval_lib.UNSUPPORTED["E12"]


def test_e20_reason_carries_the_measured_leakage():
    assert "78.6%" in eval_lib.UNSUPPORTED["E20"]
    assert "800x" in eval_lib.UNSUPPORTED["E20"]


def test_panel_record_reports_the_distance_distribution_and_missing_policy():
    record = eval_lib.panel_record("E6", population=3, value=0.5, interval=(0.2, 0.8),
                                   assay="a", distances=[1, 1, 3])
    assert record["distance_distribution"] == {"1": 2, "3": 1}
    assert "never a negative" in record["missing_data_policy"]
    assert record["interval"] == [0.2, 0.8]


def test_an_unknown_panel_is_refused():
    with pytest.raises(ValueError, match="Unknown panel"):
        eval_lib.panel_record("E99", population=1)


# --- ranking statistics ---------------------------------------------------

def test_spearman_handles_ties_and_degenerate_input():
    assert eval_lib.spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert eval_lib.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert eval_lib.spearman([1, 1, 1], [1, 2, 3]) is None
    assert eval_lib.spearman([1.0], [1.0]) is None
    # Average ranks for ties: a perfectly tied pair does not break monotonicity.
    assert eval_lib.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_spearman_refuses_invalid_scores_and_outcomes(bad):
    with pytest.raises(ValueError, match="scores"):
        eval_lib.spearman([1, 2, bad], [1, 2, 3])
    with pytest.raises(ValueError, match="outcomes"):
        eval_lib.spearman([1, 2, 3], [1, 2, bad])
    with pytest.raises(ValueError, match="scores"):
        eval_lib.spearman([bad], [1.0])


@pytest.mark.parametrize("scores, outcomes", [
    ([[1, 2], [3, 4]], [1, 2, 3, 4]),
    (5.0, [1.0, 2.0]),
    ([1, 2, 3], [1, 2]),
    ([1, 2, 3], "abc"),
    ([[1, 2], [3]], [1, 2]),
])
def test_spearman_refuses_invalid_shapes(scores, outcomes):
    with pytest.raises(ValueError):
        eval_lib.spearman(scores, outcomes)


def test_spearman_keeps_none_for_valid_undersupported_input():
    assert eval_lib.spearman([], []) is None
    assert eval_lib.spearman([2.0, 2.0], [1.0, 3.0]) is None
    assert eval_lib.spearman([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) is None


def test_censoring_aware_concordance_skips_unorderable_pairs():
    scores = [1.0, 2.0, 3.0]
    lower = [9.5, None, 7.0]      # right-censored, left-censored, point
    upper = [None, 5.0, 7.0]
    block = eval_lib.censoring_aware_concordance(scores, lower, upper)
    assert block["pairs"] + block["skipped_unorderable"] == 3
    assert "not orderable" in block["rule"]
    # The right-censored row is orderable against the left-censored one.
    assert block["pairs"] >= 1


def test_censoring_aware_concordance_does_not_penalise_a_detection_limit():
    # Two right-censored rows at the same limit cannot be ordered; counting them as
    # discordant would penalise the model for the assay.
    block = eval_lib.censoring_aware_concordance([1.0, 2.0], [9.5, 9.5], [None, None])
    assert block["pairs"] == 0
    assert block["skipped_unorderable"] == 1
    assert block["concordance"] is None


def test_paired_bootstrap_is_the_primary_endpoint_and_names_its_unit():
    block = eval_lib.paired_bootstrap([0.1] * 40, seed=1)
    assert block["mean_difference"] == pytest.approx(0.1)
    assert block["excludes_zero"]
    assert "sequence neighbourhood" in block["unit"]
    assert "selection batch" in block["not_a_unit"]
    assert "NOT a statement about training-seed variability" in block["seed_variation_note"]
    noisy = eval_lib.paired_bootstrap(list(np.linspace(-1, 1, 41)), seed=1)
    assert not noisy["excludes_zero"]


def test_average_precision_is_none_without_a_positive():
    assert eval_lib.average_precision([0.9, 0.1], [False, False]) is None
    assert eval_lib.average_precision([0.9, 0.1], [True, False]) == pytest.approx(1.0)


# --- frozen clustering ----------------------------------------------------

def test_clustering_is_built_once_from_every_arm_and_records_which():
    arms = {"declared": ["AAAAAAAAAA", "AAAAAAAAAW"],
            "easy_first": ["WWWWWWWWWW", "AAAAAAAAAA"]}
    clustering = eval_lib.freeze_clustering(arms, radius=1, separation=3)
    assert clustering.frozen
    assert clustering.built_from["arms"] == ["declared", "easy_first"]
    assert clustering.built_from["pooled_unique_unassigned"] == 3
    assert "Rebuilding per sample count changes the measurement" in clustering.document()["rule"]
    assert "not demonstrated binding mechanisms" in clustering.document()["meaning"]


def test_frozen_assignment_is_identical_for_every_nested_prefix():
    arms = {"a": ["AAAAAAAAAA", "AAAAAAAAAW", "WWWWWWWWWW"]}
    clustering = eval_lib.freeze_clustering(arms, radius=1, separation=3)
    full = clustering.occupancy(arms["a"])
    prefix = clustering.occupancy(arms["a"][:2])
    # The same medoids answer both, so the cluster indices agree.
    assert set(prefix["counts"]) <= set(full["counts"])
    assert prefix["clusters_total"] == full["clusters_total"]


def test_assignment_ties_are_deterministic_and_radius_bounded():
    clustering = eval_lib.freeze_clustering({"a": ["AAAAAAAAAA", "WWWWWWWWWW"]}, radius=1)
    assert clustering.assign("AAAAAAAAAA") == clustering.assign("AAAAAAAAAA")
    assert clustering.assign("AAAAAAAAAW") is not None      # distance 1
    assert clustering.assign("AAAAAAAAWW") is None          # distance 2, outside radius


def test_training_medoids_respect_the_declared_separation():
    medoids = eval_lib.training_medoids(["AAAAAAAAAA", "AAAAAAAAAW", "WWWWWWWWWW"],
                                        separation=3)
    assert "AAAAAAAAAA" in medoids
    assert "AAAAAAAAAW" not in medoids      # one substitution away
    assert "WWWWWWWWWW" in medoids


def test_pre_assigned_training_medoids_absorb_nearby_candidates_before_new_clusters():
    clustering = eval_lib.freeze_clustering({"a": ["AAAAAAAAAA", "CAAAAAAAAA", "WWWWWWWWWW"]},
                                           radius=1, assigned=["AAAAAAAAAA"])
    assert clustering.medoids == ("AAAAAAAAAA", "WWWWWWWWWW")
    assert clustering.assign("CAAAAAAAAA") == clustering.assign("AAAAAAAAAA")
    assert clustering.built_from["known_training_medoids"] == 1
    assert clustering.built_from["new_candidate_medoids"] == 1
    assert clustering.built_from["pre_assigned_excluded"] == 1


# --- raw generation -------------------------------------------------------

def test_generation_report_separates_duplicates_novelty_and_recovery():
    windows = ["AAAAAAAAAA", "AAAAAAAAAA", "WWWWWWWWWW", "CCCCCCCCCC"]
    clustering = eval_lib.freeze_clustering({"a": windows}, radius=1)
    report = eval_lib.generation_report(windows, training=["AAAAAAAAAA"],
                                        catalogue=["AAAAAAAAAA", "WWWWWWWWWW"],
                                        clustering=clustering, wild_type="WGGDGFYAMD",
                                        draw_prefixes=[2, 4])
    rows = {row["draws"]: row for row in report["rows"]}
    assert rows[2]["unique"] == 1 and rows[2]["exact_duplicates"] == 1
    assert rows[4]["unique"] == 3 and rows[4]["novel_unique"] == 2
    assert rows[4]["catalogue_recovered"] == 2
    assert rows[4]["catalogue_recovery_rate"] == pytest.approx(1.0)
    assert "in-silico candidates" in report["labels"]
    assert "Nothing generated by this project has been experimentally assayed" in report["labels"]


def test_generation_report_refuses_a_prefix_longer_than_the_draws():
    clustering = eval_lib.freeze_clustering({"a": ["AAAAAAAAAA"]}, radius=1)
    with pytest.raises(ValueError, match="reported as the measured one rather than padded"):
        eval_lib.generation_report(["AAAAAAAAAA"], training=[], catalogue=[],
                                   clustering=clustering, wild_type="WGGDGFYAMD",
                                   draw_prefixes=[10])


def test_learning_curve_plots_against_population_keys_not_parents_or_rows():
    block = eval_lib.learning_curve({0.1: {"score": 0.2, "population_keys": 10},
                                     1.0: {"score": 0.5, "population_keys": 100}})
    assert block["fractions"] == [0.1, 1.0]
    assert block["monotone"] is True and block["monotone_reason"] is None
    assert block["scored_points"] == 2 and block["unscored_fractions"] == []
    assert "POPULATION KEYS" in block["x_axis"] and "not parent families" in block["x_axis"]
    assert block["scope"] is None


def test_a_missing_or_nonfinite_score_is_never_read_as_zero():
    """``score or 0`` turned a failed fit into a point at zero, so a good curve could 'decrease'."""
    for bad in (None, float("nan"), float("inf"), float("-inf")):
        block = eval_lib.learning_curve({0.1: {"score": 0.3}, 0.5: {"score": bad},
                                         1.0: {"score": 0.4}})
        assert block["unscored_fractions"] == [0.5], bad
        assert block["scored_points"] == 2 and block["monotone"] is True
        assert [row["score"] for row in block["rows"]] == [0.3, None, 0.4], (
            "an unscored row reports None, the value the ordering used, never the raw nonfinite")
    negative = eval_lib.learning_curve({0.1: {"score": -0.4}, 0.5: {"score": None},
                                        1.0: {"score": -0.2}})
    assert negative["monotone"] is True, "-0.4 -> -0.2 rises; a None read as 0 would have fallen"


def test_every_missing_score_form_gives_a_strict_json_block_that_keeps_every_fraction():
    """``rows[*].score`` used to keep the original NaN/inf, so ``allow_nan=False`` raised."""
    for bad in (None, float("nan"), float("inf"), float("-inf")):
        points = {0.1: {"score": 0.3, "population_keys": 10},
                  0.5: {"score": bad, "population_keys": 50},
                  1.0: {"score": 0.4, "population_keys": 100}}
        block = eval_lib.learning_curve(points, scope={"kind": "incremental"})
        decoded = json.loads(json.dumps(block, allow_nan=False))
        assert decoded["fractions"] == [0.1, 0.5, 1.0], bad
        assert [row["fraction"] for row in decoded["rows"]] == [0.1, 0.5, 1.0]
        assert decoded["rows"][1]["score"] is None
        assert decoded["rows"][1]["population_keys"] == 50, "the rest of the row is carried"
        assert decoded["unscored_fractions"] == [0.5] and decoded["scored_points"] == 2
        assert decoded["monotone"] is True and decoded["monotone_reason"] is None
        assert decoded["largest_decrease"] == 0.0
        assert points[0.5]["score"] is bad, "the caller's own block is not rewritten"
    absent = eval_lib.learning_curve({0.1: {"score": 0.3}, 0.5: {"population_keys": 5},
                                      1.0: {"score": 0.4}})
    assert [row["score"] for row in absent["rows"]] == [0.3, None, 0.4]
    assert absent["unscored_fractions"] == [0.5]
    json.dumps(absent, allow_nan=False)


def test_an_unknown_monotonicity_stays_unknown_and_strict_json_when_every_score_is_missing():
    points = {0.1: {"score": float("nan")}, 0.5: {"score": float("inf")},
              1.0: {"score": float("-inf")}}
    block = eval_lib.learning_curve(points)
    decoded = json.loads(json.dumps(block, allow_nan=False))
    assert decoded["fractions"] == [0.1, 0.5, 1.0]
    assert [row["score"] for row in decoded["rows"]] == [None, None, None]
    assert decoded["monotone"] is None and decoded["largest_decrease"] is None
    assert decoded["scored_points"] == 0 and decoded["unscored_fractions"] == [0.1, 0.5, 1.0]
    assert "0 of 3" in decoded["monotone_reason"] and "unknown" in decoded["monotone_reason"]


def test_a_nonfinite_score_neither_invents_nor_hides_a_degradation_diagnosis():
    rising = eval_lib.learning_curve({0.1: {"score": 0.3}, 0.5: {"score": float("nan")},
                                      1.0: {"score": 0.4}})
    diagnosis = report_lib.no_gain_diagnosis(panels={}, controls={}, exposure={},
                                             learning_curve=rising)
    assert [item["explanation"] for item in diagnosis["diagnosis"]] == [
        "no single explanation is isolated by the controls"], "a missing score is not a decrease"
    falling = eval_lib.learning_curve({0.1: {"score": 0.5}, 0.5: {"score": float("inf")},
                                       1.0: {"score": 0.4}})
    assert falling["monotone"] is False and falling["largest_decrease"] == pytest.approx(0.1)
    diagnosis = report_lib.no_gain_diagnosis(panels={}, controls={}, exposure={},
                                             learning_curve=falling)
    entry, = [item for item in diagnosis["diagnosis"] if "did not rise" in item["explanation"]]
    assert entry["evidence"]["scores"] == [0.5, None, 0.4]
    json.dumps(diagnosis, allow_nan=False)


def test_monotonicity_is_unknown_not_false_with_fewer_than_two_scores():
    for points in ({}, {1.0: {"score": 0.5}}, {0.1: {"score": None}, 1.0: {"score": None}},
                   {0.1: {"score": 0.2}, 1.0: {"score": float("nan")}}):
        block = eval_lib.learning_curve(points)
        assert block["monotone"] is None, points
        assert "at least two" in block["monotone_reason"] and "unknown" in block["monotone_reason"]
        assert block["largest_decrease"] is None
    dropped = eval_lib.learning_curve({0.1: {"score": 0.5}, 1.0: {"score": 0.3}})
    assert dropped["monotone"] is False and dropped["largest_decrease"] == pytest.approx(0.2)
    assert "no interval" in dropped["monotone_basis"]


def test_learning_curve_refuses_an_invalid_or_repeated_fraction():
    for bad in (0.0, -0.1, 1.5, float("nan"), None, "x"):
        with pytest.raises(ValueError, match="fraction"):
            eval_lib.learning_curve({bad: {"score": 0.1}})
    with pytest.raises(ValueError, match="appears twice"):
        eval_lib.learning_curve({0.5: {"score": 0.1}, "0.5": {"score": 0.2}})


def test_learning_curve_carries_its_scope_verbatim():
    scope = {"kind": "incremental_summary_supervision"}
    assert eval_lib.learning_curve({1.0: {"score": 0.1}}, scope=scope)["scope"] == scope


def test_readout_separation_reports_residual_imbalance():
    pairs = [{"expression_left": 1.0, "expression_right": 1.02,
              "affinity_left": 8.0, "affinity_right": 9.0}]
    block = eval_lib.readout_separation(pairs, tolerance=0.05)
    assert block["expression_matched_pairs"] == 1
    assert block["affinity_matched_pairs"] == 0
    assert block["residual_expression_imbalance"] == pytest.approx(0.02)
    assert block["residual_affinity_imbalance"] is None
    assert "approximate" in block["matching"]
    assert block["affinity_spread_among_expression_matched"] == pytest.approx(1.0)
    assert block["expression_spread_among_affinity_matched"] is None


def test_the_legacy_tolerance_is_an_explicit_recorded_broadcast_not_a_common_unit():
    pairs = [{"expression_left": 0.0, "expression_right": 0.05,
              "affinity_left": 3.0, "affinity_right": 3.05}]
    legacy = eval_lib.readout_separation(pairs, tolerance=0.1)
    assert legacy["tolerance_source"] == "legacy_broadcast" and legacy["tolerance"] == 0.1
    assert legacy["affinity_tolerance"] == legacy["expression_tolerance"] == 0.1
    assert "broadcast" in legacy["tolerance_note"] and "different units" in legacy["tolerance_note"]
    named = eval_lib.readout_separation(
        pairs, affinity_tolerance=0.01, expression_tolerance=0.1,
        units={"affinity": "pKD", "expression": "enrichment ratio"})
    assert named["tolerance_source"] == "named" and "tolerance" not in named
    assert named["affinity_matched_pairs"] == 0 and named["expression_matched_pairs"] == 1
    assert named["tolerance_units"] == {"affinity": "pKD", "expression": "enrichment ratio"}


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), None, True, "x"])
def test_a_tolerance_must_be_finite_and_non_negative(bad):
    with pytest.raises(ValueError):
        eval_lib.readout_separation([], affinity_tolerance=bad, expression_tolerance=0.1)
    with pytest.raises(ValueError):
        eval_lib.readout_separation([], affinity_tolerance=0.1, expression_tolerance=bad)
    with pytest.raises(ValueError):
        eval_lib.readout_separation([], tolerance=bad)


def test_the_two_tolerance_forms_cannot_be_mixed_or_half_declared():
    with pytest.raises(ValueError, match="not both"):
        eval_lib.readout_separation([], tolerance=0.1, affinity_tolerance=0.1,
                                    expression_tolerance=0.1)
    with pytest.raises(ValueError, match="needs both"):
        eval_lib.readout_separation([], affinity_tolerance=0.1)
    with pytest.raises(ValueError, match="needs both"):
        eval_lib.readout_separation([])


# --- G1 -------------------------------------------------------------------

def test_gate_is_pareto_not_beat_everything():
    margins = {"backbone": 0.02, "distance": 0.02, "generation": 0.0}
    # A CNN out-ranks the decoder by 0.03, inside delta = 0.05. The route still advances.
    decision = eval_lib.gate_decision(
        route="r1", ranking=0.60, generation=0.1, backbone_ranking=0.50,
        distance_ranking=0.45, comparators={"cnn": 0.63}, delta=0.05, margins=margins)
    assert decision["advances"]
    assert decision["checks"]["not_dominated_beyond_delta"]
    assert "does not fail the gate by itself" in decision["criterion"]
    assert decision["spr152"].endswith("never gated on")


def test_gate_refuses_a_route_that_only_beats_its_backbone():
    margins = {"backbone": 0.02, "distance": 0.02, "generation": 0.0}
    decision = eval_lib.gate_decision(
        route="r2", ranking=0.46, generation=0.1, backbone_ranking=0.40,
        distance_ranking=0.45, comparators={}, delta=0.05, margins=margins)
    assert not decision["advances"]
    assert not decision["checks"]["beats_distance_only"]
    assert decision["checks"]["beats_own_backbone"]


def test_gate_refuses_a_route_dominated_beyond_delta():
    decision = eval_lib.gate_decision(
        route="r3", ranking=0.60, generation=0.1, backbone_ranking=0.50,
        distance_ranking=0.45, comparators={"cnn": 0.80}, delta=0.05,
        margins={"backbone": 0.02, "distance": 0.02, "generation": 0.0})
    assert not decision["advances"]
    assert not decision["checks"]["not_dominated_beyond_delta"]


GATE_VALID = dict(route="rx", ranking=0.8, generation=0.1, backbone_ranking=0.0,
                  distance_ranking=0.0, comparators={"c": 0.8}, delta=0.05)
GATE_MARGINS = {"backbone": 0.02, "distance": 0.02, "generation": 0.0}


def _gate(**overrides):
    arguments = {**GATE_VALID, "margins": dict(GATE_MARGINS), **overrides}
    return eval_lib.gate_decision(**arguments)


def test_gate_valid_decision_is_unchanged_and_zero_is_legitimate():
    assert _gate()["advances"]
    zero = _gate(generation=0.0, comparators={})
    assert zero["advances"] and zero["best_discriminative_comparator"] is None
    assert _gate(ranking=0.0, comparators={"c": 0.0}, delta=0.0)["checks"]["not_dominated_beyond_delta"]


@pytest.mark.parametrize("field", ["ranking", "generation", "backbone_ranking",
                                   "distance_ranking", "delta"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_gate_refuses_invalid_scalar_evidence(field, bad):
    with pytest.raises(ValueError, match=field):
        _gate(**{field: bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_gate_refuses_invalid_comparator_and_margin(bad):
    with pytest.raises(ValueError, match="comparator 'bad'"):
        _gate(comparators={"ok": 0.5, "bad": bad})
    with pytest.raises(ValueError, match="margin 'generation'"):
        _gate(margins={**GATE_MARGINS, "generation": bad})


def test_gate_refuses_a_missing_margin():
    with pytest.raises(ValueError, match="missing 'generation'"):
        _gate(margins={"backbone": 0.02, "distance": 0.02})


# --- decoding functionals -------------------------------------------------

def test_ranking_uses_the_mean_never_the_raw_sum():
    assert decoding.rank_score(-30.0, 10) == pytest.approx(-3.0)
    assert decoding.rank_score(-30.0, 15) == pytest.approx(-2.0)
    # The sum would rank the 15-site window as worse purely for being longer.
    assert decoding.rank_score(-30.0, 15) > decoding.rank_score(-30.0, 10)


def test_mixture_of_orders_is_a_distribution_and_the_mean_is_not():
    values = [-2.0, -3.0]
    mixture = decoding.mixture_of_orders_log_probability(values)
    mean = decoding.mean_log_probability_over_orders(values)
    assert mixture != pytest.approx(mean)
    # A degenerate mixture over one order returns that order's value exactly.
    assert decoding.mixture_of_orders_log_probability([-2.0], weights=[1.0]) == \
        pytest.approx(-2.0)
    assert "NOT a distribution" in decoding.FUNCTIONALS[
        "mean_log_probability_over_orders"]


def test_mixture_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        decoding.mixture_of_orders_log_probability([-1.0, -2.0], weights=[0.5, 0.2])


def test_resolve_order_covers_the_four_strategies():
    sites = [("H", 1), ("H", 2), ("H", 3)]
    assert decoding.resolve_order(sites, strategy="declared") == tuple(sites)
    rng = np.random.default_rng(0)
    assert set(decoding.resolve_order(sites, strategy="random", rng=rng)) == set(sites)
    easy = decoding.resolve_order(sites, strategy="easy_first", confidence=[0.1, 0.9, 0.5])
    hard = decoding.resolve_order(sites, strategy="hard_first", confidence=[0.1, 0.9, 0.5])
    assert easy[0] == ("H", 2) and hard[0] == ("H", 1)
    assert tuple(reversed(easy)) == hard
    with pytest.raises(ValueError, match="needs an explicit generator"):
        decoding.resolve_order(sites, strategy="random")


def test_remask_schedule_is_declared_and_seeded():
    rng = np.random.default_rng(1)
    schedule = decoding.remask_schedule(10, rounds=3, fraction=0.3, rng=rng)
    assert len(schedule) == 3
    assert all(len(block) == 3 for block in schedule)
    assert all(sorted(block) == list(block) for block in schedule)


def test_draw_accounting_keeps_the_three_numbers_apart():
    block = decoding.draw_accounting(draws=100, forward_calls=1000, seconds=5.0,
                                     mode="full_mask")
    assert block["forward_calls_per_draw"] == pytest.approx(10.0)
    assert block["seconds_per_draw"] == pytest.approx(0.05)
    assert "three different procedures" in block["distinction"]
    with pytest.raises(ValueError, match="Unknown draw mode"):
        decoding.draw_accounting(draws=1, forward_calls=1, seconds=1.0, mode="invented")


# --- I22: the guided sampler ----------------------------------------------

def test_i22_guided_weights_normalise_over_the_enumerated_block():
    weights = decoding.guided_log_weights([-1.0, -2.0], [0.5, 0.0], kappa=2.0)
    conditional = decoding.block_conditional(weights)
    assert float(np.sum(conditional)) == pytest.approx(1.0)
    assert conditional[0] > conditional[1]
    with pytest.raises(ValueError, match="non-finite"):
        decoding.guided_log_weights([float("nan")], [0.0], kappa=1.0)


def test_i22_kernel_invariance_is_verified_by_enumeration():
    states = ["a", "b"]
    target = {"a": 0.25, "b": 0.75}
    # The exact Gibbs kernel for this target.
    kernel = {"a": {"a": 0.25, "b": 0.75}, "b": {"a": 0.25, "b": 0.75}}
    block = decoding.verify_stationarity(states=states, target=target, transition=kernel)
    assert block["invariant"], block
    wrong = {"a": {"a": 0.9, "b": 0.1}, "b": {"a": 0.1, "b": 0.9}}
    assert not decoding.verify_stationarity(states=states, target=target,
                                            transition=wrong)["invariant"]
    assert "Mixing is a separate measurement" in block["claim"]


def test_i22_a_non_stochastic_kernel_is_refused():
    with pytest.raises(ValueError, match="not stochastic"):
        decoding.verify_stationarity(states=["a", "b"], target={"a": 0.5, "b": 0.5},
                                     transition={"a": {"a": 0.5, "b": 0.9},
                                                 "b": {"a": 0.5, "b": 0.5}})


def test_i22_adaptive_block_selection_is_refused_not_corrected_in_passing():
    assert decoding.require_fixed_block_selection("fixed")
    assert decoding.require_fixed_block_selection("state_independent")
    with pytest.raises(ValueError, match="needs its own correction"):
        decoding.require_fixed_block_selection("confidence_based")


def test_i22_mixing_is_measured_separately_from_invariance():
    chain = list(np.sin(np.arange(200) / 5.0))
    block = decoding.mixing_report(chain, target=0.0)
    assert "NOT invariance" in block["measures"]
    assert block["autocorrelation"]["1"] is not None
    assert block["burn_in_signal"] >= 0.0
    assert block["absolute_bias"] >= 0.0
    # A lag longer than the chain reports None rather than a fabricated number.
    short = decoding.mixing_report([1.0, 2.0, 3.0], lags=(1, 99))
    assert short["autocorrelation"]["99"] is None
