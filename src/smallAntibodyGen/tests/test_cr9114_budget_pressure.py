"""Dataflow guards for the larger-budget affinity-only continuation.

Everything here is synthetic: no checkpoint, no GPU and no real measurement file.
These tests cover the properties the run's integrity gates rest on -- schedule
prefix identity, exposure ordering, label isolation, missingness accounting and
the freeze-before-any-assay-join ordering inside the evaluator -- rather than
restating what the implementations already say.
"""
import numpy as np
import pandas as pd
import pytest

from smallAntibodyGen.experiments.affinity import likelihood_schedule
from smallAntibodyGen.experiments.pressure import (
    block_labels, category_table, development_whitelist, evaluation_records, group_masses,
    lexicographic_support, replay_exposures, site_marginals, weighted_affinity,
)

DEVELOPMENT_BLOCKS = [5, 7, 11]
LOCI = [2, 5, 8, 10]


def population(probabilities):
    return pd.DataFrame({"genotype": [f"{i:016b}" for i in range(len(probabilities))],
                         "split": ["train"] * len(probabilities),
                         "uniform_probability": [1 / len(probabilities)] * len(probabilities),
                         "affinity_probability": probabilities})


def measurements():
    ids = [f"{i:016b}" for i in range(10)]
    return pd.DataFrame({"genotype": ids, "split": ["train"] * 4 + ["development"] * 6,
                         "block": [0, 0, 0, 0, 5, 5, 7, 7, 11, 11],
                         "mean": np.linspace(8., 10., 10), "effective_sem": [.1] * 10})


@pytest.mark.parametrize("long,short", [(16384, 256), (64, 8), (1000, 999)])
def test_a_longer_draw_extends_the_shorter_one(long, short):
    # The step-256 bitwise gate depends on this numpy property; it is load bearing.
    full = np.random.default_rng(20260925).random((long, 4))
    np.testing.assert_array_equal(full[:short], np.random.default_rng(20260925).random((short, 4)))


def test_schedule_prefix_matches_the_shorter_declared_run():
    frame = population([.5, .3, .2])
    long = likelihood_schedule(frame, weighted=True, steps=512, batch_size=4, seed=20260925)
    short = likelihood_schedule(frame, weighted=True, steps=256, batch_size=4, seed=20260925)
    np.testing.assert_array_equal(long[:256], short)
    assert not np.array_equal(long[:256], likelihood_schedule(frame, weighted=True, steps=256,
                                                              batch_size=4, seed=20260926))


def test_population_weights_set_the_schedule_draw_frequencies():
    # Weighted likelihood is implemented by WHICH identities the schedule draws,
    # and this checks only that: the sampler's frequencies match the weights.
    # Whether the runner's minibatch loss then multiplies by those weights a
    # second time is not observable here and is settled by reading the loss.
    probabilities = [.6, .3, .1]
    schedule = likelihood_schedule(population(probabilities), weighted=True, steps=20000, batch_size=4, seed=7)
    observed = np.bincount(schedule.ravel(), minlength=3) / schedule.size
    np.testing.assert_allclose(observed, probabilities, atol=.01)
    uniform = likelihood_schedule(population(probabilities), weighted=False, steps=20000, batch_size=4, seed=7)
    np.testing.assert_allclose(np.bincount(uniform.ravel(), minlength=3) / uniform.size, 1 / 3, atol=.01)


def test_exposure_replay_follows_table_order_not_sorted_order():
    written = ["1001", "0001", "1111", "0100"]
    exposed, state = replay_exposures(written, 20260916, 4, 2)
    assert [genotype for row in exposed for genotype in row] != [
        genotype for row in replay_exposures(sorted(written), 20260916, 4, 2)[0] for genotype in row]
    assert replay_exposures(written, 20260916, 4, 2)[1] == state
    assert set(genotype for row in exposed for genotype in row) <= set(written)
    with pytest.raises(ValueError): replay_exposures(["0001", "0001"], 20260916, 4, 2)


def test_development_whitelist_is_a_disjoint_union():
    assert development_whitelist(["b", "a"], ["c"]) == ["a", "b", "c"]
    with pytest.raises(ValueError): development_whitelist(["a", "b"], ["b"])
    with pytest.raises(ValueError): development_whitelist(["a", "a"], ["b"])


def test_withheld_development_measurements_cannot_reach_any_statistic():
    frame = measurements()
    allowed = frame.genotype[4:8].tolist()
    filtered = evaluation_records(frame, allowed)
    assert len(filtered) == 8 and not set(frame.genotype[8:]) & set(filtered.genotype)
    assert filtered.split.value_counts().to_dict() == {"train": 4, "development": 4}
    perturbed = frame.copy()
    perturbed.loc[8:, ["mean", "effective_sem"]] = [1e6, 1e6]
    pd.testing.assert_frame_equal(filtered, evaluation_records(perturbed, allowed))
    weights = pd.Series(.1, index=frame.genotype)
    assert (weighted_affinity(filtered, weights, 9.) ==
            weighted_affinity(evaluation_records(perturbed, allowed), weights, 9.))
    with pytest.raises(ValueError): evaluation_records(frame.assign(split="test"), allowed)


def test_every_identity_lands_in_exactly_one_missingness_category():
    support = lexicographic_support(3)
    assignments = pd.Series(["train"] * 3 + ["development"] * 3 + ["test"] * 2, index=support)
    measured, eligible = {support[0], support[1], support[3]}, {support[3], support[4]}
    counts = category_table(support, assignments, measured, eligible)
    assert counts == {"measured_train": 2, "measured_development": 1, "train_ineligible": 1,
                      "development_ineligible": 1, "eligible_development_withheld": 1, "test": 2, "total": 8}
    assert sum(v for k, v in counts.items() if k != "total") == counts["total"]
    mass = category_table(support, assignments, measured, eligible, np.full(8, .125))
    assert mass["total"] == pytest.approx(1.) and mass["test"] == pytest.approx(.25)
    assert sum(v for k, v in mass.items() if k != "total") == pytest.approx(1.)


def test_categories_reject_measurements_on_reserved_or_ineligible_identities():
    support = lexicographic_support(3)
    assignments = pd.Series(["train"] * 3 + ["development"] * 3 + ["test"] * 2, index=support)
    with pytest.raises(ValueError):  # a reserved identity carrying a label is contamination
        category_table(support, assignments, {support[7]}, {support[3]})
    with pytest.raises(ValueError):  # an unwhitelisted development label is not silently dropped
        category_table(support, assignments, {support[5]}, {support[3]})


def test_duplicate_draws_are_counted_once_per_draw_in_the_category_table():
    support = lexicographic_support(3)
    assignments = pd.Series(["train"] * 3 + ["development"] * 3 + ["test"] * 2, index=support)
    draws = [support[0], support[0], support[7]]
    counts = category_table(draws, assignments, {support[0]}, {support[3]})
    assert counts == {"measured_train": 2, "measured_development": 0, "train_ineligible": 0,
                      "development_ineligible": 0, "eligible_development_withheld": 0, "test": 1, "total": 3}


def test_uniform_policy_spreads_evenly_over_sites_and_split_blocks():
    support = lexicographic_support(16)
    uniform = np.full(len(support), 1 / len(support))
    np.testing.assert_allclose(site_marginals(uniform), .5, atol=1e-12)
    blocks = group_masses(uniform, np.asarray(block_labels(support, LOCI)))
    assert len(blocks) == 16
    np.testing.assert_allclose(list(blocks.values()), 1 / 16, atol=1e-12)
    assert set(DEVELOPMENT_BLOCKS) <= {int(key) for key in blocks}


def test_evaluator_freezes_samples_and_selections_before_any_assay_join(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import run_cr9114_budget_pressure as runner

    ids = [f"{i:016b}" for i in range(4)]
    frame = pd.DataFrame({"genotype": ids[:3], "split": ["development", "development", "train"],
                          "block": [5, 5, 0], "mean": [8., 9., 10.], "effective_sem": [.1] * 3})
    draws = pd.DataFrame({"genotype": ids, "log_q": [np.log(.25)] * 4})
    assignments = pd.Series(["development", "development", "train", "test"], index=ids)
    bound = SimpleNamespace(policy=None, geometry=None,
                            space=SimpleNamespace(sequence_for=lambda alleles: "".join(map(str, alleles))))
    monkeypatch.setattr(runner, "score_sequences", lambda policy, geometry, sequences, batch_size, **kw:
                        np.full(len(sequences), np.log(.25)))
    monkeypatch.setattr(runner, "sample_table", lambda *args: draws.copy())

    def guard(original):
        def checked(*args, **kwargs):
            for name in ("development_scores.csv", "selections_before_evaluation.json", "samples.csv"):
                assert (tmp_path / name).is_file(), f"Assay join preceded frozen artifact: {name}"
            return original(*args, **kwargs)
        return checked

    for name in ("portfolio_metrics", "weighted_affinity", "affinity_difference", "sampled_affinity"):
        monkeypatch.setattr(runner, name, guard(getattr(runner, name)))
    ctx = {"pool": frame.iloc[:2], "pool_sequences": ids[:2], "multiplier": 1, "threshold": 8.5,
           "sft_selected": None, "previous_selected": None, "sft_pool_weights": None,
           "reuse_samples": {}, "sft_log_q": np.full(4, np.log(.25)), "assignments": assignments,
           "evaluation": frame, "measured": set(frame.genotype), "eligible_development": set(ids[:2]),
           "seen": {"sft": set(), "continuation": set(), "union": set()}}
    config = {"exhaustive_score_batch_size": 16, "budgets": [2], "evaluation_samples": 4,
              "evaluation_sample_seed": 0, "sample_rescore_tolerance": 1e-4}
    measured, _ = runner.evaluate(bound, ctx, config, tmp_path, "synthetic", 0)
    assert measured["sampled_affinity"]["unscored_draw_count"] == 1
