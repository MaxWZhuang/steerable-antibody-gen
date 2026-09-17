"""Sampling semantics, uncertainty handling, and held-out-label exclusion."""
import numpy as np
import pandas as pd
import pytest

from smallAntibodyGen.experiments.affinity import affinity_population, likelihood_schedule


def records():
    return pd.DataFrame({"genotype": [f"{i:016b}" for i in range(8)], "split": "train",
        "mean": np.linspace(8., 10., 8), "sample_variance": .04, "replicate_count": 3.})


def test_sampling_distribution_uses_training_noise_and_bounded_weights():
    frame = records()
    pool, audit = affinity_population(frame, quantile=0.)
    assert audit["temperature"] == pytest.approx(.2)
    np.testing.assert_allclose(pool.affinity_effective_sem, np.sqrt(.04 / 3))
    assert pool.affinity_probability.sum() == pytest.approx(1.)
    assert pool.affinity_probability.min() >= .25 / len(pool)
    assert audit["probability_ratio"] <= 20.
    assert audit["weighted_expected_affinity"] > audit["uniform_expected_affinity"]
    assert np.all(np.diff(pool.affinity_probability) >= 0)
    assert 1 < audit["effective_sampling_population"] <= len(pool)


def test_high_uncertainty_reduces_weight_at_equal_affinity():
    frame = records()
    frame["mean"] = 9.
    frame.loc[7, "sample_variance"] = 1.
    pool, _ = affinity_population(frame, quantile=0.)
    assert pool.loc[7, "affinity_probability"] < pool.loc[0, "affinity_probability"]
    assert pool.loc[7, "affinity_effective_sem"] > pool.loc[0, "affinity_effective_sem"]


def test_pooled_variance_uses_replicate_degrees_of_freedom():
    frame = records()
    frame.loc[0, ["sample_variance", "replicate_count"]] = [.5, 2]
    pool, audit = affinity_population(frame, quantile=.75)
    assert audit["pooled_sample_variance"] == pytest.approx((.5 + 7 * 2 * .04) / 15)
    assert len(pool) == 2
    assert set(pool.genotype) == set(frame.genotype.iloc[-2:])


@pytest.mark.parametrize("column,value", [("split", "development"), ("split", "test"),
    ("genotype", "0"), ("genotype", "0000000000000001"), ("mean", np.nan),
    ("sample_variance", -1.), ("replicate_count", 1), ("replicate_count", 2.5)])
def test_rejects_leakage_malformed_identities_and_measurements(column, value):
    frame = records()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        affinity_population(frame)


@pytest.mark.parametrize("kwargs", [{"quantile": 1.}, {"uncertainty_multiplier": -1.},
    {"max_weight_ratio": .5}, {"uniform_fraction": 1.1}, {"uniform_fraction": np.nan}])
def test_rejects_invalid_weighting(kwargs):
    with pytest.raises(ValueError):
        affinity_population(records(), **kwargs)


def test_uniform_mixture_and_equal_utilities_recover_uniform_distribution():
    pool, _ = affinity_population(records(), quantile=0., uniform_fraction=1.)
    np.testing.assert_allclose(pool.affinity_probability, pool.uniform_probability)
    frame = records()
    frame["mean"] = 9.
    pool, _ = affinity_population(frame)
    np.testing.assert_allclose(pool.affinity_probability, pool.uniform_probability)


def test_schedule_matches_declared_distribution_and_repeats_exactly():
    pool, _ = affinity_population(records(), quantile=0.)
    schedule = likelihood_schedule(pool, weighted=True, steps=25000, batch_size=4, seed=123)
    np.testing.assert_array_equal(schedule, likelihood_schedule(pool, weighted=True, steps=25000, batch_size=4, seed=123))
    frequencies = np.bincount(schedule.ravel(), minlength=len(pool)) / schedule.size
    np.testing.assert_allclose(frequencies, pool.affinity_probability, atol=.004, rtol=0)
    # Weighted sampling + ordinary mean equals weighted NLL; weighting twice would fail.
    nll = np.arange(len(pool)) / 8
    assert nll[schedule].mean() == pytest.approx(float(pool.affinity_probability @ nll), abs=.004)
    uniform = likelihood_schedule(pool, weighted=False, steps=25000, batch_size=4, seed=123)
    assert nll[uniform].mean() == pytest.approx(nll.mean(), abs=.004)


def test_schedule_rejects_held_out_rows_and_invalid_probabilities():
    pool, _ = affinity_population(records())
    pool.loc[0, "split"] = "development"
    with pytest.raises(ValueError, match="held-out"):
        likelihood_schedule(pool, weighted=True, steps=1, batch_size=4, seed=1)
    pool["split"] = "train"
    pool["affinity_probability"] = 0.
    with pytest.raises(ValueError, match="probabilities"):
        likelihood_schedule(pool, weighted=True, steps=1, batch_size=4, seed=1)
