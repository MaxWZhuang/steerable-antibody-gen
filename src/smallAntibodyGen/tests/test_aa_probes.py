from __future__ import annotations

import json
import math

import numpy as np
import pytest

from smallAntibodyGen.experiments.interaction_qualification import probes
from smallAntibodyGen.experiments.interaction_qualification import records as rec
from smallAntibodyGen.tests import aa_qualification_world as world


def random_problem(n, d, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 3.0, size=d) + rng.normal(size=d)
    return x, rng.normal(size=n)


# --- oracle 1: one feature, hand computed ----------------------------------------------------


def test_a_one_feature_fit_matches_the_hand_computation_and_the_intercept_is_free():
    x = np.array([[0.0], [2.0]])
    y = np.array([1.0, 5.0])
    fit = probes.fit_ridge(x, y, 1.0)
    assert fit.feature_mean.tolist() == [1.0] and fit.feature_scale.tolist() == [1.0]
    assert fit.y_mean == 3.0 and fit.coefficients.tolist() == [1.0]
    assert fit.solver == "primal" and fit.n_train == 2 and fit.n_features == 1
    np.testing.assert_allclose(probes.predict(fit, [[0.0], [2.0], [4.0]]), [2.0, 4.0, 6.0])
    shifted = probes.fit_ridge(x, y + 1000.0, 1.0)
    assert shifted.coefficients.tolist() == [1.0] and shifted.y_mean == 1003.0
    np.testing.assert_allclose(probes.predict(shifted, [[0.0], [2.0], [4.0]]), [1002.0, 1004.0, 1006.0])


# --- oracle 2: dual solve, objective, duplication, no mutation ---------------------------------


def test_a_dual_fit_matches_the_hand_computation_and_duplicating_every_row_changes_nothing():
    x = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    y = np.array([1.0, 5.0])
    x_before, y_before = x.copy(), y.copy()
    fit = probes.fit_ridge(x, y, 1.0)
    assert fit.solver == "dual"
    np.testing.assert_allclose(fit.coefficients, [0.5, 0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(probes.predict(fit, x), [1.5, 4.5])
    assert fit.objective == pytest.approx(0.5)
    assert probes.predict(fit, [[2.0, 0.0, -2.0]])[0] == pytest.approx(3.0)
    assert np.array_equal(x, x_before) and np.array_equal(y, y_before)

    twice = probes.fit_ridge(np.vstack([x, x]), np.concatenate([y, y]), 1.0)
    assert twice.solver == "primal" and twice.n_train == 4
    np.testing.assert_allclose(twice.coefficients, fit.coefficients, atol=1e-12)
    np.testing.assert_allclose(twice.feature_mean, fit.feature_mean)
    np.testing.assert_allclose(twice.feature_scale, fit.feature_scale)
    assert twice.y_mean == fit.y_mean and twice.objective == pytest.approx(fit.objective)


@pytest.mark.parametrize("copies", [2, 3, 5])
def test_duplicating_every_training_row_leaves_a_random_fit_unchanged(copies):
    x, y = random_problem(6, 4)
    reference = probes.fit_ridge(x, y, 0.3)
    again = probes.fit_ridge(np.tile(x, (copies, 1)), np.tile(y, copies), 0.3)
    np.testing.assert_allclose(again.coefficients, reference.coefficients, atol=1e-10)
    np.testing.assert_allclose(again.feature_mean, reference.feature_mean)
    np.testing.assert_allclose(again.feature_scale, reference.feature_scale)
    assert again.y_mean == pytest.approx(reference.y_mean)
    assert again.objective == pytest.approx(reference.objective)


# --- the fit is the ridge minimiser, whichever solver ran ------------------------------------


@pytest.mark.parametrize("n,d", [(6, 3), (3, 6), (5, 5), (1, 4)])
def test_the_fit_satisfies_the_ridge_optimality_condition(n, d):
    x, y = random_problem(n, d)
    alpha = 0.7
    fit = probes.fit_ridge(x, y, alpha)
    assert fit.solver == ("dual" if n < d else "primal")
    mean, std = x.mean(axis=0), x.std(axis=0)
    constant = std == 0.0
    z = (x - mean) / np.where(constant, 1.0, std)
    z[:, constant] = 0.0
    gradient = -(z.T @ ((y - y.mean()) - z @ fit.coefficients)) / n + alpha * fit.coefficients
    np.testing.assert_allclose(gradient, 0.0, atol=1e-10)
    np.testing.assert_allclose(fit.feature_mean, mean)
    np.testing.assert_allclose(fit.feature_scale, np.where(constant, 1.0, std))
    assert fit.y_mean == pytest.approx(y.mean())


def test_training_row_order_does_not_change_the_fit():
    x, y = random_problem(8, 3)
    order = np.random.default_rng(1).permutation(8)
    first = probes.fit_ridge(x, y, 1.0)
    second = probes.fit_ridge(x[order], y[order], 1.0)
    np.testing.assert_allclose(second.coefficients, first.coefficients, atol=1e-10)
    assert second.y_mean == pytest.approx(first.y_mean)


def test_constant_train_columns_get_scale_one_zero_weight_and_are_counted():
    x = np.array([[3.0, 0.0, 1.0], [3.0, 2.0, 5.0], [3.0, 4.0, 9.0]])
    fit = probes.fit_ridge(x, [1.0, 2.0, 4.0], 1.0)
    assert fit.zero_std_columns == 1
    assert fit.feature_scale[0] == 1.0 and fit.feature_mean[0] == 3.0 and fit.coefficients[0] == 0.0
    same = probes.predict(fit, [[3.0, 1.0, 2.0]])
    shifted = probes.predict(fit, [[1e6, 1.0, 2.0]])
    assert np.array_equal(same, shifted)


def test_a_single_training_row_predicts_its_own_label_everywhere():
    fit = probes.fit_ridge([[1.0, 2.0]], [5.0], 1.0)
    assert fit.zero_std_columns == 2
    np.testing.assert_allclose(probes.predict(fit, [[9.0, -3.0], [0.0, 0.0]]), [5.0, 5.0])


# --- oracle 3: held-out data cannot move the fit ---------------------------------------------


def test_heldout_rows_and_labels_cannot_change_the_fit_or_another_rows_prediction():
    fit = probes.fit_ridge([[0.0], [2.0]], [1.0, 5.0], 1.0)
    alone = probes.predict(fit, [[4.0]])
    np.testing.assert_allclose(alone, [6.0])
    batch = np.array([[4.0], [1e6], [-1e6]])
    batch_before = batch.copy()
    np.testing.assert_allclose(probes.predict(fit, batch)[:1], alone, rtol=0, atol=1e-12)
    assert np.array_equal(batch, batch_before)

    x, y = random_problem(6, 4)
    dev, _ = random_problem(5, 4, seed=11)
    fit = probes.fit_ridge(x, y, 1.0)
    snapshot = fit.to_dict(include_arrays=True)
    predictions = probes.predict(fit, dev)
    huge = np.vstack([dev, np.full((1, 4), 1e6)])
    np.testing.assert_allclose(probes.predict(fit, huge)[:5], predictions, rtol=0, atol=1e-12)
    np.testing.assert_allclose(probes.predict(fit, dev[::-1]), predictions[::-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(probes.predict(fit, dev[2:3]), predictions[2:3], rtol=0, atol=1e-12)
    assert fit.to_dict(include_arrays=True) == snapshot
    assert probes.fit_ridge(x, y, 1.0).to_dict(include_arrays=True) == snapshot


def test_fit_arrays_are_read_only_and_the_export_is_auditable():
    x, y = random_problem(6, 3)
    fit = probes.fit_ridge(x, y, 2.0)
    for array in (fit.feature_mean, fit.feature_scale, fit.coefficients):
        with pytest.raises(ValueError):
            array[0] = 1.0
    summary = fit.to_dict()
    assert summary["alpha"] == 2.0 and summary["n_train"] == 6 and summary["n_features"] == 3
    assert summary["solver"] == "primal" and summary["zero_std_columns"] == 0
    assert "coefficients" not in summary and summary["recipe"] == probes.RIDGE_RECIPE
    full = fit.to_dict(include_arrays=True)
    assert full["coefficients"] == fit.coefficients.tolist()
    json.dumps(full, allow_nan=False)


# --- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0, -1.0, float("nan"), float("inf"), True, "1", None])
def test_alpha_must_be_a_positive_finite_real(alpha):
    with pytest.raises(ValueError):
        probes.fit_ridge([[0.0], [1.0]], [0.0, 1.0], alpha)


def test_inputs_are_validated_not_coerced():
    good_x, good_y = [[0.0], [1.0]], [0.0, 1.0]
    with pytest.raises(probes.NoTrainingSupport):
        probes.fit_ridge(np.zeros((0, 3)), np.zeros(0), 1.0)
    for bad_x in ([[float("nan")], [1.0]], [[float("inf")], [1.0]], [0.0, 1.0], np.zeros((2, 0)), np.zeros((2, 1, 1))):
        with pytest.raises(ValueError):
            probes.fit_ridge(bad_x, good_y, 1.0)
    for bad_x in (np.array([[True], [False]]), [["a"], ["b"]]):
        with pytest.raises(TypeError):
            probes.fit_ridge(bad_x, good_y, 1.0)
    for bad_y in ([0.0], [0.0, 1.0, 2.0], [[0.0], [1.0]], [0.0, float("nan")], [0.0, float("inf")]):
        with pytest.raises(ValueError):
            probes.fit_ridge(good_x, bad_y, 1.0)
    fit = probes.fit_ridge(good_x, good_y, 1.0)
    with pytest.raises(ValueError):
        probes.predict(fit, [[0.0, 1.0]])
    with pytest.raises(ValueError):
        probes.predict(fit, [[float("nan")]])
    assert probes.predict(fit, np.zeros((0, 1))).shape == (0,)


# --- id joins --------------------------------------------------------------------------------


def dev_records():
    return rec.parse_records(world.jsonl_bytes(world.grid("development", ["ACD", "ACE", "ACF"], ["EFG"])))


def test_predictions_join_to_records_by_id_and_a_mismatch_is_an_error():
    records = dev_records()
    ids = [r.record_id for r in records]
    fit = probes.fit_ridge([[0.0], [2.0]], [1.0, 5.0], 1.0)
    predictions = probes.predict_by_id(fit, ids, [[0.0], [2.0], [4.0]])
    assert predictions == {ids[0]: pytest.approx(2.0), ids[1]: pytest.approx(4.0), ids[2]: pytest.approx(6.0)}
    rows = probes.scored_rows(records, predictions)
    assert [r.record_id for r in rows] == ids
    assert [r.y_pred for r in rows] == pytest.approx([2.0, 4.0, 6.0])
    assert [r.y_true for r in rows] == [r.pkd for r in records]
    assert [r.pair_fingerprint for r in rows] == [r.pair_fingerprint for r in records]
    # the same numbers keyed to swapped ids give swapped rows: nothing is joined by position
    swapped = probes.scored_rows(records, dict(zip(ids, [6.0, 4.0, 2.0])))
    assert [r.y_pred for r in swapped] == [6.0, 4.0, 2.0]
    with pytest.raises(ValueError, match="different ids"):
        probes.scored_rows(records[:-1], predictions)
    with pytest.raises(ValueError, match="different ids"):
        probes.scored_rows(records, dict(predictions, extra=1.0))
    with pytest.raises(ValueError):
        probes.predict_by_id(fit, [ids[0], ids[0]], [[0.0], [1.0]])
    with pytest.raises(ValueError):
        probes.predict_by_id(fit, ids[:2], [[0.0], [1.0], [2.0]])
    with pytest.raises(TypeError):
        probes.predict_by_id(fit, "abc", [[0.0], [1.0], [2.0]])


def test_training_counts_are_per_source_and_assay():
    records = rec.parse_records(
        world.jsonl_bytes(
            world.grid("train", ["ACD", "ACE"], ["EFG"], assay="assay1")
            + world.grid("train", ["ACD"], ["EFG"], assay="assay2")
            + world.grid("train", ["ACD"], ["EFG"], source="study2", assay="assay1")
        )
    )
    assert probes.training_counts(records) == {("study1", "assay1"): 2, ("study1", "assay2"): 1, ("study2", "assay1"): 1}


# --- oracle 4: grouping by (partition, source, assay), never by assay alone ---------------------


def srow(record_id, partition, source, assay, pair, y_true, y_pred):
    return probes.ScoredRow(record_id, partition, source, assay, pair, y_true, y_pred)


def oracle_rows():
    return [
        srow("a1", "development", "sourceA", "KD", "p1", 6.0, 5.0),
        srow("a2", "development", "sourceA", "KD", "p2", 5.0, 6.0),
        srow("b1", "development", "sourceB", "KD", "p3", 11.0, 10.0),
        srow("b2", "development", "sourceB", "KD", "p4", 10.0, 11.0),
    ]


def test_the_same_assay_name_in_two_sources_is_two_groups_and_pooling_is_secondary():
    evaluation = probes.evaluate(oracle_rows(), train_mean=8.0, train_counts={("sourceA", "KD"): 3})
    assert [(g.partition, g.source, g.assay) for g in evaluation.groups] == [
        ("development", "sourceA", "KD"),
        ("development", "sourceB", "KD"),
    ]
    for group in evaluation.groups:
        assert group.n_rows == 2 and group.n_unique_constructs == 2
        assert group.mae == pytest.approx(1.0) and group.rmse == pytest.approx(1.0)
        assert group.spearman == pytest.approx(-1.0) and group.pearson == pytest.approx(-1.0)
        assert group.correlation_unavailable is None
        assert group.train_mean_baseline_rmse == pytest.approx(math.sqrt(6.5))
    (pooled,) = evaluation.pooled
    assert pooled.partition == "development" and pooled.source is None and pooled.assay is None
    assert pooled.n_rows == 4 and pooled.n_unique_constructs == 4
    assert pooled.spearman == pytest.approx(0.6)
    assert pooled.pearson == pytest.approx(24.0 / 26.0)
    assert pooled.mae == pytest.approx(1.0) and pooled.rmse == pytest.approx(1.0)
    assert pooled.train_mean_baseline_rmse == pytest.approx(math.sqrt(6.5))


def test_training_support_is_reported_per_heldout_group():
    evaluation = probes.evaluate(oracle_rows(), train_mean=8.0, train_counts={("sourceA", "KD"): 3})
    group_a, group_b = evaluation.groups
    assert (group_a.n_train_rows, group_a.training_support) == (3, True)
    assert (group_b.n_train_rows, group_b.training_support) == (0, False)
    assert evaluation.pooled[0].n_train_rows == 3 and evaluation.pooled[0].training_support is True
    none = probes.evaluate(oracle_rows(), train_mean=8.0, train_counts={})
    assert not any(g.training_support for g in none.groups + none.pooled)


def test_undefined_correlations_are_null_with_a_reason_never_zero():
    rows = [
        srow("s1", "development", "src", "single", "p1", 5.0, 5.5),
        srow("c1", "development", "src", "const_y", "p2", 5.0, 1.0),
        srow("c2", "development", "src", "const_y", "p3", 5.0, 2.0),
        srow("q1", "development", "src", "const_p", "p4", 1.0, 3.0),
        srow("q2", "development", "src", "const_p", "p5", 2.0, 3.0),
        srow("z1", "development", "src", "const_both", "p6", 1.0, 1.0),
        srow("z2", "development", "src", "const_both", "p7", 1.0, 1.0),
    ]
    evaluation = probes.evaluate(rows, train_mean=0.0, train_counts={})
    by_assay = {g.assay: g for g in evaluation.groups}
    reasons = {assay: g.correlation_unavailable for assay, g in by_assay.items()}
    assert "fewer than 2 rows" in reasons["single"]
    assert reasons["const_y"] == "constant labels"
    assert reasons["const_p"] == "constant predictions"
    assert reasons["const_both"] == "constant labels and predictions"
    for group in by_assay.values():
        assert group.pearson is None and group.spearman is None
    assert by_assay["single"].mae == pytest.approx(0.5) and by_assay["single"].n_rows == 1
    assert by_assay["const_both"].rmse == 0.0
    assert evaluation.pooled[0].correlation_unavailable is None and evaluation.pooled[0].spearman is not None


def test_groups_and_pooled_rows_are_sorted_by_partition_then_source_and_assay():
    rows = [
        srow("t1", "test", "b", "z", "p1", 1.0, 1.0),
        srow("t2", "test", "b", "z", "p2", 2.0, 2.5),
        srow("d1", "development", "b", "z", "p3", 1.0, 1.0),
        srow("d2", "development", "a", "y", "p4", 1.0, 1.0),
        srow("d3", "development", "a", "x", "p5", 1.0, 1.0),
    ]
    for ordered in (rows, rows[::-1]):
        evaluation = probes.evaluate(ordered, train_mean=0.0, train_counts={})
        assert [(g.partition, g.source, g.assay, g.n_rows) for g in evaluation.groups] == [
            ("development", "a", "x", 1),
            ("development", "a", "y", 1),
            ("development", "b", "z", 1),
            ("test", "b", "z", 2),
        ]
        assert [(g.partition, g.n_rows) for g in evaluation.pooled] == [("development", 3), ("test", 2)]
    only_dev = probes.evaluate(rows[2:], train_mean=0.0, train_counts={})
    assert {g.partition for g in only_dev.groups + only_dev.pooled} == {"development"}


def test_permuting_heldout_labels_changes_the_metrics_and_nothing_else():
    train_x, train_y = random_problem(6, 3)
    dev_x, _ = random_problem(3, 3, seed=11)
    fit = probes.fit_ridge(train_x, train_y, 1.0)
    predictions = probes.predict(fit, dev_x)
    assert len(set(predictions.tolist())) == 3
    labels = predictions.copy()  # perfect held-out labels, so any permutation must make the metrics worse

    def metrics(order):
        rows = [
            srow(f"d{i}", "development", "s", "a", f"p{i}", float(labels[order[i]]), float(predictions[i]))
            for i in range(3)
        ]
        return probes.evaluate(rows, train_mean=fit.y_mean, train_counts={("s", "a"): 6}).groups[0]

    original, permuted = metrics([0, 1, 2]), metrics([2, 0, 1])
    assert np.array_equal(predictions, probes.predict(fit, dev_x))  # labels never enter prediction
    assert (original.n_rows, original.n_unique_constructs) == (permuted.n_rows, permuted.n_unique_constructs)
    assert original.mae == 0.0 and permuted.mae > 0.0
    assert original.spearman == pytest.approx(1.0) and permuted.spearman < 1.0


def test_metric_inputs_are_validated():
    with pytest.raises(ValueError):
        srow("x", "validation", "s", "a", "p", 1.0, 1.0)
    with pytest.raises(ValueError):
        srow("x", "development", "s", "a", "p", float("nan"), 1.0)
    with pytest.raises(ValueError):
        srow("x", "development", "s", "a", "p", 1.0, True)
    with pytest.raises(ValueError):
        probes.evaluate(oracle_rows(), train_mean=float("nan"), train_counts={})
    json.dumps(
        [g.to_dict() for g in probes.evaluate(oracle_rows(), train_mean=8.0, train_counts={}).groups], allow_nan=False
    )
