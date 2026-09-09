"""Invariants for the linear-ranker positive control.

The load-bearing one is leakage: the `cv` arm must never fit on a group it then
predicts. If that breaks, the probe reports the `ceiling` number under the `cv`
label and silently turns an upper bound into a result.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from probe_linear_hcdr3_ranker import (  # noqa: E402
    AMINO_ACIDS, featurise, group_stats_from_predictions, load_population, ridge_fit,
    ridge_predict, run_arm,
)


def synthetic_population(n_groups=20, per_group=4, seed=0):
    """Groups whose measurement is an exact linear function of residue counts."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=len(AMINO_ACIDS))
    groups = []
    for g in range(n_groups):
        hcdr3s, values = [], []
        for _ in range(per_group):
            seq = "".join(rng.choice(list(AMINO_ACIDS), size=12))
            hcdr3s.append(seq)
            values.append(sum(weights[AMINO_ACIDS.index(c)] for c in seq))
        groups.append(("g{:03d}".format(g), hcdr3s, np.array(values)))
    return {"t1": groups}


def test_featurise_shape_and_composition_block():
    X = featurise(["AAAC", "WWWWWW"])
    assert X.shape[0] == 2
    assert X[0, -1] == 4 and X[1, -1] == 6  # length feature
    composition = X[0, -1 - len(AMINO_ACIDS):-1]
    assert composition[AMINO_ACIDS.index("A")] == 3
    assert composition[AMINO_ACIDS.index("C")] == 1


def test_ridge_recovers_a_planted_linear_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    truth = rng.normal(size=8)
    y = X @ truth + 3.0
    weights = ridge_fit(X, y, 1e-8)
    assert np.allclose(weights[:8], truth, atol=1e-4)
    assert weights[-1] == pytest.approx(3.0, abs=1e-4)  # intercept is unpenalised
    assert np.allclose(ridge_predict(X, weights), y, atol=1e-4)


def test_a_perfect_predictor_scores_one_and_a_reversed_one_scores_zero():
    groups = [("g1", ["AA", "CC", "DD"], np.array([1.0, 2.0, 3.0]))]
    perfect = group_stats_from_predictions("t1", groups, [np.array([1.0, 2.0, 3.0])])
    reversed_ = group_stats_from_predictions("t1", groups, [np.array([3.0, 2.0, 1.0])])
    assert perfect[0].concordance == pytest.approx(1.0)
    assert reversed_[0].concordance == pytest.approx(0.0)


def test_cv_arm_never_fits_on_the_group_it_predicts():
    """Leakage guard: a group memorised by the fit would score far above its CV value."""
    population = synthetic_population()
    calls = {"fit_groups": []}
    import probe_linear_hcdr3_ranker as probe

    original = probe.ridge_fit

    def spy(X, y, lam):
        calls["fit_groups"].append(len(X))
        return original(X, y, lam)

    probe.ridge_fit = spy
    try:
        cv = run_arm(population, 10.0, folds=5, arm="cv")
        ceiling = run_arm(population, 10.0, folds=5, arm="ceiling")
    finally:
        probe.ridge_fit = original

    total_variants = sum(len(h) for _, h, _ in population["t1"])
    # Each CV fit must see strictly fewer variants than the whole population.
    cv_fits = calls["fit_groups"][:5]
    assert all(n < total_variants for n in cv_fits), cv_fits
    assert len(cv) == len(ceiling) == len(population["t1"])
    # On an exactly-linear task the in-sample ceiling cannot be beaten by CV.
    from smallAntibodyGen.evaluation.contrast_statistics import primary_endpoint
    assert primary_endpoint(ceiling) >= primary_endpoint(cv) - 1e-9


def test_recoverable_signal_is_actually_recovered_out_of_sample():
    # If the probe could not recover a signal that is there by construction, a
    # null result on real data would be uninformative.
    cv = run_arm(synthetic_population(), 1.0, folds=5, arm="cv")
    from smallAntibodyGen.evaluation.contrast_statistics import primary_endpoint
    # 0.75 is "clearly recovers", not "recovers perfectly": 20 groups x 4 variants
    # is a small fit for a 20-residue alphabet, so exact recovery is not expected
    # (observed ~0.84). The invariant is distance from the 0.5 floor.
    assert primary_endpoint(cv) > 0.75


def test_missing_direction_is_fatal(tmp_path):
    import gzip, json
    scores = {"groups": [{"group_id": "g1", "target": "t1", "fold": "validation",
                          "variants": [{"hcdr3": "AA", "measurement": 1.0},
                                       {"hcdr3": "CC", "measurement": 2.0}]}]}
    manifest = {"groups": [{"group_id": "g1", "target": "t1"}]}
    sp, mp = tmp_path / "s.json", tmp_path / "m.json"
    sp.write_text(json.dumps(scores), encoding="utf8")
    mp.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="direction"):
        load_population(sp, mp)


def test_sequence_disjoint_folds_never_split_a_shared_hcdr3():
    """The whole point: no HCDR3 string may appear in two different folds."""
    import numpy as np
    from probe_linear_hcdr3_ranker import sequence_disjoint_folds
    groups = [
        ("g0", ["AAA", "CCC"], np.array([1.0, 2.0])),
        ("g1", ["CCC", "DDD"], np.array([1.0, 2.0])),   # shares CCC with g0
        ("g2", ["EEE", "FFF"], np.array([1.0, 2.0])),   # disjoint
        ("g3", ["GGG", "HHH"], np.array([1.0, 2.0])),   # disjoint
    ]
    assignment, components, largest = sequence_disjoint_folds(groups, 3)
    assert assignment[0] == assignment[1], "groups linked by CCC must share a fold"
    assert components == 3
    assert largest == pytest.approx(0.5)
    placement = {}
    for index, (_, hcdr3s, _) in enumerate(groups):
        for sequence in hcdr3s:
            placement.setdefault(sequence, set()).add(assignment[index])
    assert all(len(folds) == 1 for folds in placement.values()), placement


def test_sequence_disjoint_is_not_more_optimistic_than_background_cv():
    """Controlling sequence overlap can only remove optimism, never add it."""
    from smallAntibodyGen.evaluation.contrast_statistics import primary_endpoint
    population = synthetic_population()
    background = primary_endpoint(run_arm(population, 10.0, folds=5, arm="cv"))
    disjoint = primary_endpoint(run_arm(population, 10.0, folds=5, arm="cv_seqdisjoint"))
    assert disjoint <= background + 1e-9
