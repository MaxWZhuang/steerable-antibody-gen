"""
Tests for the paired cross-antigen evaluator.

The behaviours pinned here are the ones the memo review corrected, so a regression would
quietly reinstate a decision rule the owner rejected. Each test names the rule it protects.
"""

from __future__ import annotations

import math

import pytest

from smallAntibodyGen.evaluation.paired_antigen import (
    PairedMeasurement,
    compare_arms,
    evaluate_arm,
    measurements_from_manifest,
    sensitivity_screen,
)


def _cohort(n: int = 24) -> list[PairedMeasurement]:
    """A synthetic cohort with a target-wide shift plus a variant-dependent component."""
    out = []
    for i in range(n):
        variant_effect = (i - n / 2) / n
        delta = 0.5 + variant_effect  # 0.5 is the target-wide offset
        out.append(
            PairedMeasurement(
                variant=f"v{i}",
                measured_a=1.0,
                measured_b=1.0 + delta,
                delta=delta,
                paired_standard_error=0.1,
                exploratory=abs(variant_effect) > 0.25,
            )
        )
    return out


def _arm(measurements, fn, **kw):
    return evaluate_arm(
        arm=kw.pop("arm", "test"),
        measurements=measurements,
        scores_a={m.variant: 0.0 for m in measurements},
        scores_b={m.variant: fn(m) for m in measurements},
        score_semantics="synthetic",
        permutations=200,
        bootstrap=200,
        **kw,
    )


def test_constant_differential_yields_undefined_not_zero():
    """A0-style flat arm: rank correlation is undefined, and must never be emitted as 0.0."""
    result = _arm(_cohort(), lambda m: 0.0, arm="A0")

    assert result["sensitivity"]["is_constant"] is True
    assert result["agreement_primary"]["spearman_rho"] is None
    assert result["agreement_primary"]["permutation_p"] is None
    assert result["agreement_primary"]["concordance"] is None
    assert result["agreement_primary"]["undefined_reason"]


def test_flat_arm_is_retained_in_comparison_not_dropped():
    """Flat arms are baselines. Ranking must keep them visible with their reason."""
    flat = _arm(_cohort(), lambda m: 0.0, arm="A0")
    real = _arm(_cohort(), lambda m: m.delta, arm="A1")

    table = compare_arms([flat, real])

    assert [row["arm"] for row in table["ranked"]] == ["A1"]
    assert [row["arm"] for row in table["undefined"]] == ["A0"]
    assert table["undefined"][0]["undefined_reason"]
    # The flat arm must not have been coerced into the ranking as a zero.
    assert all(row["spearman_rho"] is not None for row in table["ranked"])


def test_perfect_agreement_recovers_rho_one():
    result = _arm(_cohort(), lambda m: m.delta, arm="oracle")

    assert result["agreement_primary"]["spearman_rho"] == pytest.approx(1.0)
    assert result["agreement_primary"]["sd_ratio"] == pytest.approx(1.0)
    low, high = result["agreement_primary"]["bootstrap_ci_95"]
    assert low > 0.9 and high <= 1.0


def test_target_wide_offset_alone_gives_no_agreement():
    """A model reproducing only the shift knows nothing about any variant."""
    result = _arm(_cohort(), lambda m: 0.5, arm="offset-only")

    assert result["sensitivity"]["is_constant"] is True
    assert result["agreement_primary"]["spearman_rho"] is None


def test_score_direction_reorients_rather_than_inverting():
    """A higher-is-tighter scorer must not come out anti-correlated."""
    measurements = _cohort()
    lower = _arm(measurements, lambda m: m.delta, arm="lower")
    higher = evaluate_arm(
        arm="higher",
        measurements=measurements,
        scores_a={m.variant: 0.0 for m in measurements},
        scores_b={m.variant: -m.delta for m in measurements},
        score_semantics="synthetic",
        score_direction="higher_is_tighter",
        permutations=200,
        bootstrap=200,
    )

    assert higher["agreement_primary"]["spearman_rho"] == pytest.approx(
        lower["agreement_primary"]["spearman_rho"]
    )


def test_sd_ratio_detects_narrowed_predictions_that_correlation_misses():
    """The range statistic exists because correlation survives a collapsed dynamic range."""
    result = _arm(_cohort(), lambda m: m.delta * 0.1, arm="narrow")

    assert result["agreement_primary"]["spearman_rho"] == pytest.approx(1.0)
    assert result["agreement_primary"]["sd_ratio"] == pytest.approx(0.1)


def test_missing_scores_are_refused_not_silently_reduced():
    measurements = _cohort()
    with pytest.raises(ValueError, match="missing scores"):
        evaluate_arm(
            arm="partial",
            measurements=measurements,
            scores_a={m.variant: 0.0 for m in measurements},
            scores_b={m.variant: 1.0 for m in measurements[:-1]},
            score_semantics="synthetic",
        )


def test_sensitivity_screen_is_undefined_without_decoys():
    result = _arm(_cohort(), lambda m: m.delta)
    assert sensitivity_screen(result, [])["passes"] is None


def test_sensitivity_screen_compares_against_decoy_spread():
    result = _arm(_cohort(), lambda m: m.delta)
    observed = result["sensitivity"]["sd_predicted_delta"]

    assert sensitivity_screen(result, [observed * 0.1] * 20)["passes"] is True
    assert sensitivity_screen(result, [observed * 10] * 20)["passes"] is False


def test_exploratory_subset_is_reported_separately_from_primary():
    measurements = _cohort()
    result = _arm(measurements, lambda m: m.delta)

    assert result["agreement_primary"]["n"] == len(measurements)
    assert 0 < result["agreement_exploratory"]["n"] < len(measurements)
    assert "claim_limit" in result["agreement_exploratory"]


def test_within_condition_is_reported_but_never_gates():
    result = _arm(_cohort(), lambda m: m.delta)
    within = result["within_condition"]

    # Condition A scores are constant here, so its ranking is undefined -- and that must not
    # suppress or invalidate the differential result, which is the D1 correction.
    assert within["condition_a"] is None
    assert result["agreement_primary"]["spearman_rho"] is not None


def test_manifest_schema_is_checked():
    with pytest.raises(ValueError, match="unexpected manifest schema"):
        measurements_from_manifest({"schema": "something-else/1", "variants": []})


def test_manifest_round_trip_from_real_cohort_shape():
    payload = {
        "schema": "ym0693-paired-cohort/1",
        "variants": [
            {
                "variant": "candidate_1",
                "affinity_human": 0.5,
                "affinity_mouse": 1.0,
                "delta_mouse_minus_human": 0.5,
                "paired_standard_error": 0.1,
                "exploratory_subset": True,
            }
        ],
    }
    (measurement,) = measurements_from_manifest(payload)

    assert measurement.variant == "candidate_1"
    assert math.isclose(measurement.delta, 0.5)
    assert measurement.exploratory is True
