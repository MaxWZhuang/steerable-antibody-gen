"""The report's two rules: a panel reports a number, or a DATA fact that blocks it.

The distinction is the whole point. "No measured HER2 counter-target exists" is a
support gate; "not implemented" is unfinished code wearing a support gate's clothes,
and :func:`her2_routes_report.require_no_placeholder` refuses it.

The reserved endpoint is also pinned here. It was a bootstrap of the mean raw
log-score difference, which an arbitrary additive offset in one arm's scores moves
while moving no ordering whatsoever -- so a shifted scorer passed the primary
endpoint. The replacement is the bootstrapped difference of Spearmans over sequence
neighbourhoods, and the offset invariance is asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_eval as eval_lib
from smallAntibodyGen.experiments import her2_routes_report as report_lib


# --- the support gate -----------------------------------------------------

def test_an_unsupported_panel_carries_a_data_reason_and_no_value():
    record = report_lib.panel("E12", population=0, supported=False)
    assert record["value"] is None
    assert "no measured HER2 counter-target" in record["blocking_reason"]


def test_a_blocking_reason_naming_unfinished_code_is_refused():
    for reason in ("not implemented yet", "TODO: wire the head", "placeholder for now",
                   "the objective is not wired"):
        with pytest.raises(ValueError, match="unfinished code rather than unavailable data"):
            report_lib.panel("E8", population=0, supported=False, reason=reason)


def test_a_data_reason_is_accepted():
    record = report_lib.panel(
        "E8", population=0, supported=False,
        reason=("no complete measured quartet exists in the enrolled training partitions, so no "
                "numerical interaction is computable"))
    assert record["supported"] is False
    assert record["value"] is None


def test_the_assembled_report_separates_completeness_from_the_panels():
    panels = {"E3": report_lib.panel("E3", population=10, value=0.4),
              "E12": report_lib.panel("E12", population=0, supported=False)}
    document = report_lib.assemble(
        panels=panels, stages={"bakeoff": "completed", "confirm": "partial"},
        unsupported=["E12"], caveats=["nothing generated here has been assayed"])
    assert document["numeric_panels"] == ["E3"]
    assert document["completion"]["complete"] is False
    assert "three different facts" in document["completion"]["rule"]


def test_the_assembled_report_refuses_a_placeholder_panel():
    bad = dict(report_lib.panel("E3", population=1, value=0.1))
    bad["blocking_reason"] = "not implemented"
    with pytest.raises(ValueError, match="unfinished code"):
        report_lib.assemble(panels={"E3": bad}, stages={}, unsupported=[], caveats=[])


# --- the paired delta Spearman --------------------------------------------

def panel_vectors(n=40, *, seed=3):
    rng = np.random.default_rng(seed)
    outcomes = rng.normal(size=n)
    good = outcomes + rng.normal(scale=0.4, size=n)
    poor = rng.normal(size=n)
    groups = [f"n{index // 4}" for index in range(n)]
    return good, poor, outcomes, groups


def test_the_delta_spearman_is_invariant_to_an_additive_offset():
    """The exact defect. A shifted scorer changes no ordering and must change nothing."""
    good, poor, outcomes, groups = panel_vectors()
    base = eval_lib.paired_delta_spearman(good, poor, outcomes, groups=groups, seed=1,
                                         draws=200)
    shifted = eval_lib.paired_delta_spearman(good + 100.0, poor, outcomes, groups=groups,
                                            seed=1, draws=200)
    assert base["delta"] == pytest.approx(shifted["delta"])
    assert base["interval"] == pytest.approx(shifted["interval"])
    assert "moves no ordering" in base["not_this"]


def test_the_delta_spearman_finds_the_better_scorer_and_names_its_unit():
    good, poor, outcomes, groups = panel_vectors()
    block = eval_lib.paired_delta_spearman(good, poor, outcomes, groups=groups, seed=1,
                                          draws=400)
    assert block["delta"] > 0
    assert block["excludes_zero"]
    assert block["unit"].startswith("sequence neighbourhood")
    assert "not families and not bootstrap units" in block["not_a_unit"]
    assert block["neighbourhoods"] == len(set(groups))
    assert block["usable_draws"] <= 400


def test_the_delta_spearman_resamples_NEIGHBOURHOODS_not_rows():
    """Two rows of one neighbourhood must travel together, so the unit count is groups."""
    good, poor, outcomes, groups = panel_vectors(n=40)
    block = eval_lib.paired_delta_spearman(good, poor, outcomes, groups=groups, seed=7,
                                          draws=200)
    assert block["neighbourhoods"] == 10 and block["rows"] == 40


def test_one_neighbourhood_is_not_an_interval():
    good, poor, outcomes, _groups = panel_vectors(n=8)
    with pytest.raises(ValueError, match="One unit is not an interval"):
        eval_lib.paired_delta_spearman(good, poor, outcomes, groups=["a"] * 8, seed=1)


# --- the tie conventions --------------------------------------------------

def test_a_constant_scorer_earns_half_credit_not_zero():
    block = eval_lib.censoring_aware_concordance([1.0, 1.0, 1.0],
                                                [8.0, 9.0, 10.0], [8.0, 9.0, 10.0])
    assert block["concordance"] == pytest.approx(0.5)
    assert block["score_ties"] == block["pairs"]
    assert "earns half credit" in block["rule"]


def test_average_precision_is_tie_invariant():
    """Delegated to the repository's threshold-grouped implementation."""
    assert eval_lib.average_precision([0.9, 0.9], [True, False]) == pytest.approx(0.5)
    assert eval_lib.average_precision([0.9, 0.9], [False, True]) == pytest.approx(0.5)
    assert eval_lib.average_precision([1.0, 1.0, 1.0, 1.0],
                                      [True, True, False, False]) == pytest.approx(0.5)


# --- the no-gain diagnosis ------------------------------------------------

def test_a_head_only_arm_that_matches_the_full_arm_is_diagnosed():
    block = report_lib.no_gain_diagnosis(
        panels={}, controls={"rankings": {"full": 0.40, "head_only": 0.399,
                                          "reconstruction_only": 0.20}},
        exposure={}, learning_curve={})
    explanations = [entry["explanation"] for entry in block["diagnosis"]]
    assert "the gain is in the head, not the decoder" in explanations
    assert "not a successful campaign" in block["rule"]


def test_a_falling_learning_curve_is_described_not_attributed_to_capacity():
    """A nonmonotone single-fit curve is a description of points. It does not establish a cause."""
    block = report_lib.no_gain_diagnosis(
        panels={}, controls={"rankings": {"full": 0.5, "head_only": 0.2,
                                          "reconstruction_only": 0.1}},
        exposure={},
        learning_curve={"monotone": False, "largest_decrease": 0.1,
                        "rows": [{"fraction": 0.1, "score": 0.5},
                                 {"fraction": 1.0, "score": 0.4}]})
    entry, = [item for item in block["diagnosis"]
              if "did not rise" in item["explanation"]]
    assert entry["evidence"]["scores"] == [0.5, 0.4]
    assert "does not by itself establish a capacity" in entry["interpretation"]
    assert "capacity or conditioning limit rather than a data limit" not in str(block)
    assert "repeat the curve over seeds" in entry["next"]
    assert all(item["next"] for item in block["diagnosis"])
    assert block["learning_curve"]["monotone"] is False


def test_an_unknown_monotonicity_is_not_interpreted_as_a_degradation():
    """``bool(None)`` was False, so a curve with no scored fractions read as 'did not help'."""
    for curve in ({}, {"rows": [{"fraction": 1.0, "score": None}], "monotone": None,
                       "monotone_reason": "0 of 1 fraction(s) carry a finite score"},
                  {"rows": [], "monotone": None}):
        block = report_lib.no_gain_diagnosis(panels={}, controls={}, exposure={},
                                             learning_curve=curve)
        explanations = [item["explanation"] for item in block["diagnosis"]]
        assert not any("did not rise" in text or "did not help" in text for text in explanations)
        assert explanations == ["no single explanation is isolated by the controls"]
        assert block["learning_curve"]["monotone"] is None
    reported = report_lib.no_gain_diagnosis(
        panels={}, controls={}, exposure={},
        learning_curve={"rows": [{"fraction": 1.0, "score": None}], "monotone": None,
                        "monotone_reason": "0 of 1 fraction(s) carry a finite score"})
    assert "0 of 1" in reported["learning_curve"]["reason"]


def test_a_rising_learning_curve_adds_no_data_limit_diagnosis():
    block = report_lib.no_gain_diagnosis(
        panels={}, controls={}, exposure={},
        learning_curve={"monotone": True, "rows": [{"fraction": 0.1, "score": 0.1},
                                                   {"fraction": 1.0, "score": 0.4}]})
    assert [item["explanation"] for item in block["diagnosis"]] == [
        "no single explanation is isolated by the controls"]


def test_no_isolated_explanation_still_names_a_finite_next_step():
    block = report_lib.no_gain_diagnosis(panels={}, controls={}, exposure={},
                                         learning_curve={})
    assert len(block["diagnosis"]) == 1
    assert "5,000-update increments" in block["diagnosis"][0]["next"]


# --- the gate -------------------------------------------------------------

def test_the_gate_reads_only_scalar_comparators():
    """A nested fit block in the comparator table used to raise inside float()."""
    decision = eval_lib.gate_decision(
        route="ablang2_masked_editor", ranking=0.40, generation=0.05,
        backbone_ranking=0.30, distance_ranking=0.30,
        comparators={"additive_linear": 0.42, "negative_wt_distance": 0.31},
        delta=0.05, margins={"backbone": 0.02, "distance": 0.02, "generation": 0.0})
    assert decision["advances"]
    assert decision["best_discriminative_comparator"] == pytest.approx(0.42)


def test_a_route_that_does_not_beat_its_own_backbone_does_not_advance():
    decision = eval_lib.gate_decision(
        route="piggen_causal_infiller", ranking=0.31, generation=0.05,
        backbone_ranking=0.30, distance_ranking=0.10, comparators={"additive_linear": 0.2},
        delta=0.05, margins={"backbone": 0.02, "distance": 0.02, "generation": 0.0})
    assert not decision["advances"]
    assert decision["checks"]["beats_own_backbone"] is False


def test_a_negative_generation_endpoint_does_not_advance():
    """The earlier endpoint was a negated positive loss, so it was always negative."""
    decision = eval_lib.gate_decision(
        route="ablang2_masked_editor", ranking=0.40, generation=-1.5,
        backbone_ranking=0.30, distance_ranking=0.30, comparators={},
        delta=0.05, margins={"backbone": 0.02, "distance": 0.02, "generation": 0.0})
    assert decision["checks"]["raw_generation_improves"] is False
    assert not decision["advances"]
