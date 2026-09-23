"""Dependence: four quantities that are routinely confused, kept apart and checked.

The parity construction is the decisive one. A distribution where every pair is
independent but the whole is not has **zero** pairwise mutual information and
**positive** total correlation, so any implementation that reports the MI sum as
total correlation fails here rather than in a report.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_coupling as coupling
from smallAntibodyGen.tests.her2_nf_support import SmallModel, TinyPolicy, tiny_cores

sklearn = pytest.importorskip("sklearn")


def _parity_draws(rows, *, seed=0):
    """``x3 = x1 xor x2`` over three binary positions, padded to ten.

    Pairwise independent by construction; jointly maximally dependent.
    """
    generator = np.random.default_rng(seed)
    first = generator.integers(0, 2, rows)
    second = generator.integers(0, 2, rows)
    third = first ^ second
    block = np.zeros((rows, 10), dtype=np.int8)
    block[:, 0], block[:, 1], block[:, 2] = first, second, third
    return block


def test_pairwise_mi_is_zero_while_total_correlation_is_positive():
    rows = 20_000
    draws = _parity_draws(rows, seed=1)
    pairwise = coupling.pairwise_mutual_information(draws)
    assert abs(pairwise["sum"]) < 0.01, "pairwise MI is ~0 for a parity construction"
    # The exact conditionals of the generating process: position 2 is determined
    # by the first two, and the padding positions are deterministic.
    conditionals = np.full((rows, 10, 20), 1e-300)
    conditionals[:, 0, 0] = conditionals[:, 0, 1] = 0.5
    conditionals[:, 1, 0] = conditionals[:, 1, 1] = 0.5
    conditionals[np.arange(rows), 2, draws[:, 2]] = 1.0
    conditionals[:, 3:, 0] = 1.0
    block = coupling.total_correlation(np.log(conditionals))
    assert block["total_correlation"] == pytest.approx(math.log(2.0), abs=0.02)
    assert block["not"].startswith("the sum of pairwise")


def test_total_correlation_matches_an_exactly_enumerated_model():
    policy = TinyPolicy(SmallModel(seed=3, length=3, alphabet=3))
    words = np.asarray(list(itertools.product(range(3), repeat=3)), dtype=np.int8)
    probabilities = np.exp(policy.sequence_log_probs(words).detach().numpy())
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    # Exact TC = sum_i H(X_i) - H(X).
    marginal_entropy = 0.0
    for position in range(3):
        marginal = np.zeros(3)
        for row, word in enumerate(words):
            marginal[word[position]] += probabilities[row]
        marginal_entropy += float(-(marginal * np.log(marginal)).sum())
    joint_entropy = float(-(probabilities * np.log(probabilities)).sum())
    exact = marginal_entropy - joint_entropy
    # And the estimator, on a large own-context bank.
    from smallAntibodyGen.experiments import her2_nf_mixture as mixture
    index, _ = policy.sample(60_000, seed=4)
    conditionals = mixture.position_conditionals(policy, index, batch_size=4096)
    estimate = coupling.total_correlation(np.log(conditionals))
    assert estimate["total_correlation"] == pytest.approx(exact, abs=0.02)
    assert estimate["per_position"][0] == pytest.approx(0.0, abs=1e-9), \
        "the first editable position contributes zero under a fixed conditioning prefix"


def test_the_estimator_is_identically_zero_at_one_draw():
    conditionals = np.log(np.full((1, 10, 20), 1.0 / 20))
    block = coupling.total_correlation(conditionals)
    assert block["total_correlation"] == pytest.approx(0.0, abs=1e-12)
    assert "DOWNWARD biased" in block["properties"]


def test_generated_marginals_use_logsumexp_not_a_pseudocount():
    conditionals = np.log(np.full((5, 10, 20), 1.0 / 20))
    log_marginals = coupling.generated_log_marginals(conditionals)
    assert np.allclose(np.exp(log_marginals).sum(axis=1), 1.0, atol=1e-12)
    # An underflowed residue stays representable rather than becoming a zero
    # that would need an undocumented floor.
    conditionals[:, 0, 3] = -800.0
    block = coupling.generated_log_marginals(conditionals)
    assert np.isfinite(block).all()
    assert block[0, 3] < -700


def test_the_permutation_floor_removes_dependence_and_keeps_the_marginals():
    draws = _parity_draws(4000, seed=5)
    observed = coupling.pairwise_mutual_information(draws)
    floor = coupling.permutation_floor(draws, seed=6, draws=10)
    assert floor["mean"] >= 0
    assert "not a bias correction" in floor["status"]
    # On a dependent construction the floor sits below the observed sum.
    dependent = draws.copy()
    dependent[:, 3] = dependent[:, 0]
    assert coupling.pairwise_mutual_information(dependent)["sum"] > observed["sum"]


def test_the_score_decomposition_is_exact():
    index = tiny_cores(200, seed=7)
    policy = TinyPolicy(seed=8)
    parent = TinyPolicy(seed=9)
    from smallAntibodyGen.experiments import her2_nf_mixture as mixture
    policy_marginals = coupling.generated_log_marginals(
        np.log(mixture.position_conditionals(policy, index, batch_size=64)))
    parent_marginals = coupling.generated_log_marginals(
        np.log(mixture.position_conditionals(parent, index, batch_size=64)))
    policy_scores = policy.score(index)["sum_log_probability"]
    parent_scores = parent.score(index)["sum_log_probability"]
    left = coupling.score_decomposition(index, policy_scores, policy_marginals)
    right = coupling.score_decomposition(index, parent_scores, parent_marginals)
    identity = (left["full_score"] - right["full_score"]) - (
        (left["marginal_score"] - right["marginal_score"])
        + (left["residual_score"] - right["residual_score"]))
    assert np.abs(identity).max() < 1e-12


def test_hybrids_are_labelled_as_ranking_diagnostics():
    index = tiny_cores(40, seed=10)
    policy = TinyPolicy(seed=11)
    marginals = np.log(np.full((10, 20), 1.0 / 20))
    left = coupling.score_decomposition(index, policy.score(index)["sum_log_probability"],
                                        marginals)
    block = coupling.hybrid_scores(left, left)
    assert np.allclose(block["marginal_hybrid"], left["full_score"])
    assert "Never exponentiated into a density" in block["status"]


def test_the_decomposition_report_refuses_to_add_ap_differences():
    index = tiny_cores(120, seed=12)
    policy, parent = TinyPolicy(seed=13), TinyPolicy(seed=14)
    marginals = np.log(np.full((10, 20), 1.0 / 20))
    labels = np.arange(120) % 3 == 0
    classes = np.where(labels, "high", "low")
    block = coupling.decomposition_report(
        index, parent.score(index)["sum_log_probability"],
        policy.score(index)["sum_log_probability"], marginals, marginals, labels, classes)
    assert set(block["ranking"]) >= {"full_parent", "full_policy", "marginal_hybrid",
                                     "residual_hybrid"}
    for name, entry in block["within_class_mean_changes"].items():
        assert abs(entry["additivity_residual"]) < 1e-12, name
    assert "AP differences are not additive" in block["forbidden"]


def test_parent_context_drift_reports_T_B_and_their_gap():
    generator = np.random.default_rng(15)
    logits = generator.normal(size=(50, 10, 20))
    teacher_log = logits - np.log(np.exp(logits).sum(axis=2, keepdims=True))
    student_log = teacher_log + generator.normal(scale=0.2, size=teacher_log.shape)
    student_log = student_log - np.log(np.exp(student_log).sum(axis=2, keepdims=True))
    block = coupling.parent_context_drift(np.exp(teacher_log), teacher_log, student_log)
    assert block["B_le_T_holds"] is True
    assert len(block["conditional_T"]) == 10
    assert block["sum_T"] >= block["sum_B"]
    assert "not biological epistasis" in block["interpretation"]
    assert "not a proof" in block["sanity_check_note"]


def test_design_matrix_has_the_declared_feature_counts():
    index = tiny_cores(30, seed=16)
    additive = coupling.design_matrix(index, pairwise=False)
    full = coupling.design_matrix(index, pairwise=True)
    assert additive.shape[1] == coupling.MAIN_EFFECTS == 190
    assert full.shape[1] == coupling.MAIN_EFFECTS + coupling.PAIR_FEATURES == 190 + 16245
    # Reference coding: residue 0 contributes no indicator at its position.
    row = np.zeros((1, 10), dtype=np.int8)
    assert coupling.design_matrix(row, pairwise=True).nnz == 0


def test_int8_residue_codes_do_not_overflow_the_column_arithmetic():
    """The reported ``OverflowError: Python integer 133 out of bounds for int8``.

    A bank stores residue codes as ``int8``; position 7's offset alone is 133.
    NumPy 2 raises on the addition and a pair product would wrap silently.
    """
    index = np.full((2, 10), 19, dtype=np.int8)
    assert index.dtype == np.int8
    matrix = coupling.design_matrix(index, pairwise=True)
    assert matrix.shape == (2, 190 + 16245)
    # Ten main effects plus all 45 pair terms are active on every row.
    assert matrix.nnz == 2 * (10 + 45)


def test_encoded_columns_match_an_independent_feature_enumeration():
    """The encoder against the coding rule, written out separately."""
    index = tiny_cores(25, seed=161)
    for pairwise in (False, True):
        matrix = coupling.design_matrix(index, pairwise=pairwise).toarray()
        for row in range(index.shape[0]):
            expected = coupling.enumerate_features(index[row], pairwise=pairwise)
            observed = sorted(int(value) for value in np.flatnonzero(matrix[row]))
            assert observed == expected, (row, pairwise)


def test_generated_marginals_return_minus_infinity_for_an_all_zero_column():
    """A bank supported only on residue zero produced 190 NaN columns."""
    conditionals = np.full((64, 10, 20), -np.inf)
    conditionals[:, :, 0] = 0.0
    block = coupling.generated_log_marginals(conditionals)
    assert not np.isnan(block).any()
    assert np.isneginf(block[:, 1:]).all()
    assert np.allclose(block[:, 0], 0.0)
    # And the total correlation over that bank is finite and zero.
    total = coupling.total_correlation(conditionals)
    assert np.isfinite(total["total_correlation"])
    assert total["total_correlation"] == pytest.approx(0.0, abs=1e-12)


def test_streamed_drift_matches_the_whole_bank_computation():
    """Chunked sufficient statistics reproduce the single-shot record exactly."""
    generator = np.random.default_rng(1601)
    teacher = np.log(_normalize(generator.random((300, 10, 20)) + 0.05))
    student = np.log(_normalize(generator.random((300, 10, 20)) + 0.05))
    single = coupling.parent_context_drift(np.exp(teacher), teacher, student)
    accumulator = coupling.DriftAccumulator()
    for start in range(0, 300, 37):
        accumulator.add(teacher[start:start + 37], student[start:start + 37])
    streamed = accumulator.finish()
    assert streamed["rows"] == 300 == single["rows"]
    assert np.allclose(streamed["conditional_T"], single["conditional_T"], atol=1e-12)
    assert np.allclose(streamed["averaged_B"], single["averaged_B"], atol=1e-12)
    assert "FULL declared bank" in streamed["estimator"]


def _normalize(values):
    return values / values.sum(axis=-1, keepdims=True)


def test_the_interaction_classifier_runs_on_the_installed_sklearn_without_multi_class():
    generator = np.random.default_rng(17)
    index = generator.integers(0, 3, size=(300, 10)).astype(np.int8)
    classes = np.where(index[:, 0] == index[:, 1], "high",
                       np.where(index[:, 2] == 0, "mid", "low"))
    fit = coupling.fit_interaction_classifier(
        index[:200], classes[:200], development_index=index[200:],
        development_classes=classes[200:], pairwise=True, regularization_grid=(0.1, 1.0),
        max_iterations=60)
    assert fit["pairwise"] is True
    assert fit["features"] == 190 + 16245
    assert fit["tuned_on"] == "development/C rows only"
    assert len(fit["attempts"]) == 2
    scores = coupling.classifier_scores(fit, index[200:])
    assert scores["p_high"].shape == (100,)
    assert "not a generative probability" in scores["status"]
    additive = coupling.fit_interaction_classifier(
        index[:200], classes[:200], development_index=index[200:],
        development_classes=classes[200:], pairwise=False, regularization_grid=(1.0,),
        max_iterations=60)
    assert additive["pair_features"] == 0


def test_bank_size_sensitivity_subsamples_one_bank_and_says_so():
    conditionals = np.log(np.full((500, 10, 20), 1.0 / 20))
    block = coupling.bank_size_sensitivity(conditionals, sizes=(100, 250), seed=18)
    assert block["sizes"] == [100, 250]
    assert "independent banks answer the repeat question" in block["note"]
