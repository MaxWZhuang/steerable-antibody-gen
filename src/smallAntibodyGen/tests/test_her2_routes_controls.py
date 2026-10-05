"""E17 and E19: matched exposure, permutations that preserve what they should, reversals.

The permutation tests are the ones worth reading closely. A global shuffle destroys the
outcome's relationship to distance as well as to sequence, and an arm that beat *that*
control would only have learned about distance. Within-stratum permutation is what makes
the control hard.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_controls as controls


# --- matched exposure -----------------------------------------------------

def test_matched_exposure_requires_equal_rows_and_reports_unequal_forwards():
    ledgers = {
        "full": {"totals": {"microbatch_rows": 640, "student_forwards": 900}},
        "reconstruction_only": {"totals": {"microbatch_rows": 640, "student_forwards": 640}},
    }
    record = controls.require_matched_exposure(ledgers)
    assert record["row_spread"] == 0
    assert record["forward_spread"] == 260
    assert "legitimately DIFFER" in record["forward_note"]


def test_matched_exposure_refuses_a_shorter_control():
    with pytest.raises(ValueError, match="confounds the removed loss"):
        controls.require_matched_exposure({
            "full": {"totals": {"microbatch_rows": 640, "student_forwards": 640}},
            "short": {"totals": {"microbatch_rows": 320, "student_forwards": 320}}})


def test_arm_coefficients_zero_only_the_excluded_terms():
    base = {"reconstruction": 1.0, "local_effect": 1.0, "whole_window": 1.0, "heads": 0.5}
    full = controls.arm_coefficients(controls.ARM_FULL, base=base)
    assert full == base
    head_only = controls.arm_coefficients(controls.ARM_HEAD_ONLY, base=base)
    assert head_only == {"reconstruction": 1.0, "local_effect": 0.0, "whole_window": 0.0,
                         "heads": 0.5}
    recon = controls.arm_coefficients(controls.ARM_RECONSTRUCTION_ONLY, base=base)
    assert recon["heads"] == 0.0, "reconstruction-only removes HEAD supervision too"
    assert recon["reconstruction"] == 1.0
    local = controls.arm_coefficients(controls.ARM_LOCAL_ONLY, base=base)
    assert local["whole_window"] == 0.0 and local["local_effect"] == 1.0


def test_every_control_arm_has_a_declared_term_set():
    for arm in controls.CONTROL_ARMS:
        assert arm in controls.ARM_TERMS
    with pytest.raises(ValueError, match="Unknown control arm"):
        controls.arm_coefficients("invented", base={})


# --- permutations ---------------------------------------------------------

def test_within_stratum_permutation_preserves_every_stratum_marginal():
    values = np.arange(20.0)
    strata = ["a"] * 10 + ["b"] * 10
    permuted = controls.permute_within_strata(values, strata, seed=1)
    report = controls.permutation_preserved_marginals(values, permuted, strata)
    assert report["global_multiset_preserved"]
    for block in report["per_stratum"].values():
        assert block["multiset_preserved"]
        assert block["mean_preserved"]
    assert report["values_moved"] > 0
    # No value crossed a stratum boundary.
    assert set(permuted[:10]) == set(values[:10])
    assert set(permuted[10:]) == set(values[10:])


def test_a_stratum_with_one_row_is_left_alone_rather_than_faked():
    values = np.asarray([1.0, 2.0, 3.0])
    permuted = controls.permute_within_strata(values, ["a", "b", "b"], seed=1)
    assert permuted[0] == 1.0


def test_oriented_substitution_group_is_directional_and_single_site():
    assert controls.oriented_substitution_group("AAAA", "AWAA") == (1, "A", "W")
    assert controls.oriented_substitution_group("AWAA", "AAAA") == (1, "W", "A")
    assert controls.oriented_substitution_group("AAAA", "AWWA") is None
    assert controls.oriented_substitution_group("AAAA", "AAAA") is None


# --- E19 ------------------------------------------------------------------

def test_distance_profile_reports_every_declared_summary():
    profile = controls.distance_profile("AAAAAAAAAA", wild_type="WGGDGFYAMD",
                                        bank=["AAAAAAAAAA", "AAAAAAAAAW"],
                                        groups={"high": ["AAAAAAAAAW"]}, radius=2)
    assert profile.to_nearest_training == 0
    # AAAAAAAAAA and WGGDGFYAMD agree at position 7 (A), so the distance is 9, not 10.
    # The earlier expectation of 10 assumed two random 10-mers differ everywhere.
    assert profile.to_wild_type == 9
    assert profile.to_nearest_by_group["high"] == 1
    assert profile.neighbourhood_density == 2
    assert set(profile.document()) == {"to_wild_type", "to_nearest_training",
                                       "to_nearest_by_group", "neighbourhood_density"}


def test_out_of_fold_excludes_the_query_from_its_own_bank():
    bank = ["AAAAAAAAAA", "AAAAAAAAAW", "AAAAAAAAWW"]
    values = [1.0, 2.0, 3.0]
    leaky = controls.out_of_fold_nearest("AAAAAAAAAA", bank=bank, values=values,
                                        exclude_self=False)
    honest = controls.out_of_fold_nearest("AAAAAAAAAA", bank=bank, values=values)
    # Without the exclusion the query retrieves itself at distance zero and the control
    # scores perfectly, which makes it useless AS a control.
    assert leaky["distance"] == 0 and leaky["prediction"] == pytest.approx(1.0)
    assert honest["distance"] == 1 and honest["prediction"] == pytest.approx(2.0)


def test_out_of_fold_reports_an_empty_bank_rather_than_guessing():
    block = controls.out_of_fold_nearest("AAAAAAAAAA", bank=["AAAAAAAAAA"], values=[1.0])
    assert block["prediction"] is None
    assert "nothing but the query" in block["reason"]


def test_stratified_concordance_withholds_the_pooled_figure():
    scores = [1.0, 2.0, 3.0, 4.0]
    outcomes = [1.0, 2.0, 3.0, 4.0]
    strata = ["0", "0", "1", "1"]
    block = controls.stratified_concordance(scores, outcomes, strata)
    assert block["macro_average"] == pytest.approx(1.0)
    assert block["within_stratum_pairs"] == 2
    assert "pooled figure is deliberately absent" in block["note"]
    # A distance-only rule that is constant within a stratum scores 0.5 there, not 1.0.
    flat = controls.stratified_concordance([1.0, 1.0, 2.0, 2.0], outcomes, strata)
    assert flat["macro_average"] is None or flat["macro_average"] < 1.0


def test_stratified_concordance_reports_a_stratum_with_no_support():
    block = controls.stratified_concordance([1.0, 2.0], [1.0, 1.0], ["0", "0"])
    assert block["per_stratum"]["0"]["concordance"] is None
    assert block["strata_with_support"] == 0
    assert block["macro_average"] is None


def test_distance_only_baseline_is_the_measured_shortcut():
    scores = controls.distance_only_baseline(["WGGDGFYAMD", "WGGDGFYAMA", "AAAAAAAAAA"],
                                             wild_type="WGGDGFYAMD")
    # -9.0 and not -10.0: the two share position 7. Hand-computed, in one place.
    assert scores == [0.0, -1.0, -9.0]
    context = controls.shortcut_context()
    assert context["spr152_distance_only_spearman"] == pytest.approx(0.4312)
    assert context["spr152_parent_spearman"] == pytest.approx(0.5634)
    assert "E19 or it is not made" in context["consequence"]


# --- matched reversals ----------------------------------------------------

def _pair(left, right, outcome_delta, score_delta):
    return {"left_window": left, "right_window": right,
            "left_profile": {"wt": 1, "train": 2}, "right_profile": {"wt": 2, "train": 2},
            "outcome_delta": outcome_delta, "score_delta": score_delta}


# A reversal needs the SAME substitution (here A<->W at position 1) with matched endpoint profiles
# measured in two DIFFERENT physical contexts -- two distinct pairs of windows. One physical pair
# cannot stand in both directions, so the fixtures below use two backgrounds, "AAAA"/"AWAA" and
# "CAAA"/"CWAA", not the same pair twice.
def test_matched_reversals_are_support_gated_when_a_group_is_one_directional():
    block = controls.matched_reversals([_pair("AAAA", "AWAA", 1.0, 1.0)])
    assert block["groups_seen"] == 1
    assert block["groups_in_both_directions"] == 0
    assert block["support_gated"]
    assert block["macro_accuracy"] is None
    assert "0-3 repeated development groups" in block["her2_caveat"]


def test_an_additive_scalar_score_hits_the_half_ceiling():
    # One group seen in both directions. An additive score assigns the SAME signed
    # margin to both, so it is right once and wrong once.
    pairs = [_pair("AAAA", "AWAA", 1.0, 1.0), _pair("CAAA", "CWAA", -1.0, 1.0)]
    block = controls.matched_reversals(pairs)
    assert block["groups_in_both_directions"] == 1
    assert block["macro_accuracy"] == pytest.approx(0.5)
    assert "0.5 ceiling" in block["ceiling_claim"]


def test_a_context_dependent_score_can_exceed_the_ceiling():
    pairs = [_pair("AAAA", "AWAA", 1.0, 1.0), _pair("CAAA", "CWAA", -1.0, -1.0)]
    block = controls.matched_reversals(pairs)
    assert block["macro_accuracy"] == pytest.approx(1.0)


def test_ties_get_half_credit_under_the_declared_tolerance():
    pairs = [_pair("AAAA", "AWAA", 1.0, 0.0), _pair("CAAA", "CWAA", -1.0, 0.0)]
    block = controls.matched_reversals(pairs, tolerance=0.01)
    assert block["macro_accuracy"] == pytest.approx(0.5)


def test_distance_strata_are_declared_not_fitted():
    assert controls.distance_stratum(0) == "0"
    assert controls.distance_stratum(3) == "3"
    assert controls.distance_stratum(9) == ">3"
