"""Exact continuous metrics and assay-error accounting beyond candidate swaps."""
import numpy as np
import pandas as pd
import pytest

from smallAntibodyGen.experiments.pressure import (
    affinity_difference, block_labels, evaluation_records, exact_distribution, exposure_overlap, group_masses,
    label_free, lexicographic_support, monte_carlo_kl, probabilities, sampled_affinity, site_marginals,
    weighted_affinity,
)


def records():
    return pd.DataFrame({"genotype": [f"{i:016b}" for i in range(4)],
                         "split": ["train", "development", "development", "development"],
                         "mean": [8., 9., 10., 7.], "effective_sem": [.1, .2, .3, .4]})


def test_exact_distribution_matches_hand_computation():
    p, r = np.array([.8, .2]), np.array([.5, .5])
    result = exact_distribution(np.log(p), np.log(r))
    assert result["entropy_nats"] == pytest.approx(-sum(p * np.log(p)))
    assert result["collision_probability"] == pytest.approx(.68)
    assert result["total_variation_from_sft"] == pytest.approx(.3)
    assert result["kl_to_sft_nats"] == pytest.approx(sum(p * np.log(p / r)))
    assert result["kl_from_sft_nats"] == pytest.approx(sum(r * np.log(r / p)))
    assert exact_distribution(np.log(r), np.log(r))["total_variation_from_sft"] == 0.


@pytest.mark.parametrize("values", [[0.], [0., 0.], [np.nan, -1.], [1., -1.], [-10., -10.]])
def test_rejects_incomplete_or_invalid_probability_support(values):
    with pytest.raises(ValueError): probabilities(values)


def test_conditional_affinity_is_not_unconditional_and_missing_mass_is_visible():
    frame = records().iloc[:2]
    weights = pd.Series([.2, .3, .5], index=[f"{i:016b}" for i in range(3)])
    result = weighted_affinity(frame, weights, 8.5)
    assert result["measured_mass"] == pytest.approx(.5)
    assert result["conditional_mean_affinity"] == pytest.approx(8.6)
    assert result["positive_mass"] == pytest.approx(.3)
    assert result["assay_sem_proxy"] == pytest.approx(np.hypot(.4 * .1, .6 * .2))


def test_shared_identities_cancel_assay_error_for_a_single_swap():
    frame = records()
    a = pd.Series([.5, .5], index=frame.genotype[:2])
    b = pd.Series([.5, .5], index=frame.genotype[[0, 2]])
    result = affinity_difference(frame, a, b)
    assert result["conditional_affinity_delta"] == pytest.approx(-.5)
    assert result["assay_sem_proxy_for_delta"] == pytest.approx(np.hypot(.2, .3) / 2)
    assert affinity_difference(frame, a, a)["assay_sem_proxy_for_delta"] == 0.


def test_continuous_affinity_moves_without_changing_top_candidate():
    frame = records().iloc[:2]
    a = pd.Series([.6, .4], index=frame.genotype)
    b = pd.Series([.7, .3], index=frame.genotype)
    assert a.idxmax() == b.idxmax()
    assert affinity_difference(frame, a, b)["conditional_affinity_delta"] == pytest.approx(.1)


def test_unused_development_labels_are_excluded_and_test_labels_rejected():
    frame = records()
    allowed = {frame.genotype[1]}
    filtered = evaluation_records(frame, allowed)
    assert filtered.genotype.tolist() == frame.genotype[:2].tolist()
    altered = frame.copy()
    altered.loc[2:, "mean"] = 10000.
    pd.testing.assert_frame_equal(filtered, evaluation_records(altered, allowed))
    with pytest.raises(ValueError): evaluation_records(frame.assign(split="test"), allowed)


def test_duplicate_draws_share_assay_error_and_unscored_outcomes_remain_visible():
    frame = records()
    assignments = frame.set_index("genotype").split.copy()
    assignments.iloc[3] = "test"
    filtered = evaluation_records(frame, {frame.genotype[1]})
    draws = [frame.genotype[i] for i in (1, 1, 2, 3)]
    result = sampled_affinity(draws, assignments, filtered, 8.5, set())
    dev = result["by_split"]["development"]
    assert dev["measured_draw_count"] == 2
    assert dev["assay_sem_proxy"] == pytest.approx(.2)  # same assay used twice
    assert dev["sampling_sem_conditional_mean"] == 0.
    assert result["unscored_draw_count"] == 2
    assert result["split_counts"] == {"development": 3, "test": 1}


def test_zero_coverage_has_no_invented_affinity():
    frame = records()
    result = weighted_affinity(frame, pd.Series([1.], index=["1111111111111111"]), 9.)
    assert result["measured_mass"] == 0 and result["conditional_mean_affinity"] is None


def test_unnormalized_weights_report_true_subset_mass():
    # Joint log probabilities are already normalized over the full support, so a
    # subset's exp-sum is its unconditional mass. Scaling must not move the mean.
    frame = records().iloc[:2]
    weights = pd.Series([.02, .03, .95], index=[f"{i:016b}" for i in range(3)])
    result = weighted_affinity(frame, weights, 8.5)
    scaled = weighted_affinity(frame, weights * 7, 8.5)
    assert result["measured_mass"] == pytest.approx(.05)
    assert scaled["measured_mass"] == pytest.approx(.35)
    assert scaled["conditional_mean_affinity"] == pytest.approx(result["conditional_mean_affinity"])


@pytest.mark.parametrize("weights,expected", [([1., 1., 1.], 3.), ([1., 0., 0.], 1.), ([3., 1.], 1.6)])
def test_effective_genotype_count_tracks_concentration(weights, expected):
    frame = records().iloc[:len(weights)]
    series = pd.Series(weights, index=frame.genotype)
    assert weighted_affinity(frame, series, 8.5)["conditional_effective_genotypes"] == pytest.approx(expected)


def test_near_degenerate_distribution_stays_finite():
    # With no KL penalty an atom can reach p == 0 exactly while its log stays finite.
    result = exact_distribution(np.array([-1e-9, -800.]), np.log([.5, .5]))
    assert np.isfinite(list(result.values())).all()
    assert result["maximum_genotype_probability"] == pytest.approx(1.)
    assert result["entropy_nats"] == pytest.approx(0., abs=1e-6)
    assert result["kl_from_sft_nats"] > 300  # unbounded as the policy drains an atom


@pytest.mark.parametrize("mass,accepted", [(1 + 1.5e-5, True), (1 - 1.5e-5, True), (1 + 3e-5, False), (1 - 3e-5, False)])
def test_normalization_tolerance_boundary(mass, accepted):
    values = np.log([mass / 2, mass / 2])
    if accepted:
        assert probabilities(values)[0].sum() == pytest.approx(1.)
    else:
        with pytest.raises(ValueError): probabilities(values)


def test_site_marginals_recover_a_known_product_distribution():
    q = np.array([.2, .5, .9])
    support = lexicographic_support(3)
    p = np.array([np.prod([q[j] if bit == "1" else 1 - q[j] for j, bit in enumerate(g)]) for g in support])
    assert support[:2] == ["000", "001"] and len(support) == 8
    np.testing.assert_allclose(site_marginals(p), q, atol=1e-12)


def test_block_labels_and_group_masses_partition_the_support():
    assert block_labels(["0101", "1110", "0000"], [0, 2]) == [0, 3, 0]
    masses = group_masses([.1, .2, .3, .4], ["a", "b", "a", "b"])
    assert masses == {"a": pytest.approx(.4), "b": pytest.approx(.6)}
    with pytest.raises(ValueError): block_labels(["0101"], [0, 9])


def test_monte_carlo_kl_is_an_estimate_with_reported_error():
    result = monte_carlo_kl([-1., -2., -3.], [-2., -2., -1.])
    ratio = np.array([1., 0., -2.])
    assert result["nats"] == pytest.approx(ratio.mean()) and result["draws"] == 3
    assert result["standard_error"] == pytest.approx(ratio.std(ddof=1) / np.sqrt(3))
    assert result["standard_error"] > 0
    assert monte_carlo_kl([-1., -2.], [-1., -2.])["nats"] == 0.


def test_exposure_overlap_separates_draws_from_identities():
    result = exposure_overlap(["a", "a", "b"], {"a", "c"})
    assert (result["draw_count"], result["unique_count"], result["exposure_unique_count"]) == (2, 1, 2)


def test_pre_evaluation_artifacts_cannot_carry_measurements():
    frame = pd.DataFrame({"genotype": ["0" * 16], "score": [-1.], "mean": [9.]})
    with pytest.raises(ValueError): label_free(frame, ("genotype", "score", "mean"))
    with pytest.raises(ValueError): label_free(frame, ("genotype", "score"))
    assert label_free(frame[["genotype", "score"]], ("genotype", "score")).columns.tolist() == ["genotype", "score"]


def test_sampled_affinity_rejects_test_labelled_records_before_filtering():
    frame = records().iloc[:3].copy()
    frame.loc[2, "split"] = "test"
    assignments = frame.set_index("genotype").split
    with pytest.raises(ValueError):
        sampled_affinity([frame.genotype[2]], assignments, frame, 9., set())
