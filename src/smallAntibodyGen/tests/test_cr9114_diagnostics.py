"""Guard against misleading diversity and preference diagnostic estimators."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
spec = importlib.util.spec_from_file_location("cr9114_diagnostics", SCRIPTS / "diagnose_cr9114_dpo.py")
diagnostics = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(diagnostics)
finally:
    sys.path[:] = original_path


def test_point_mass_has_zero_entropy_diversity_and_unit_collision():
    m = diagnostics.diversity_metrics(["0" * 16] * 8, np.zeros(8))
    assert m["entropy_nats_mc"] == 0
    assert m["entropy_effective_support_mc"] == 1
    assert m["mean_pairwise_hamming_unbiased"] == 0
    assert m["collision_probability_unbiased"] == 1
    assert m["sites_with_minor_allele_below_1pct"] == 16


def test_entropy_uses_true_policy_probability_not_empirical_sample_support():
    # Two observations from a uniform 65,536-member policy still estimate 16 bits.
    m = diagnostics.diversity_metrics(["0" * 16, "1" * 16], [-16 * np.log(2)] * 2)
    assert m["entropy_effective_support_mc"] == pytest.approx(65536)
    assert m["unique_genotypes"] == 2
    assert m["mean_pairwise_hamming_unbiased"] == 16
    assert m["collision_probability_unbiased"] == 0


def test_pairwise_hamming_and_collision_match_explicit_pairs():
    gs = ["0" * 16, "0" * 16, "1" * 16, "01" * 8]
    m = diagnostics.diversity_metrics(gs, [-3] * 4)
    distances = [sum(a != b for a, b in zip(gs[i], gs[j]))
                 for i in range(4) for j in range(i)]
    assert m["mean_pairwise_hamming_unbiased"] == np.mean(distances)
    assert m["collision_probability_unbiased"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("gs,logq", [(["x" * 16] * 2, [-1, -1]),
                                    (["0" * 16] * 2, [0, np.nan]),
                                    (["0" * 16] * 2, [0, 1]),
                                    (["0" * 16] * 2, [-1])])
def test_invalid_samples_fail(gs, logq):
    with pytest.raises(ValueError):
        diagnostics.diversity_metrics(gs, logq)


def test_raw_pair_accuracy_differs_from_reference_relative_dpo_loss():
    pairs = pd.DataFrame({"chosen_genotype": ["a", "c"], "rejected_genotype": ["b", "d"],
                          "block": [1, 2], "pair_weight": [.8, .2]})
    scores = pd.Series({"a": -2., "b": -1., "c": -1., "d": -3.})
    same = diagnostics.pair_metrics(pairs, scores, scores)
    assert same["pair_accuracy"] == pytest.approx(.2)
    assert same["dpo_loss"] == pytest.approx(np.log(2))
    assert diagnostics.pair_metrics(pairs, scores, weighted=False)["pair_accuracy"] == .5
    ref = pd.Series({"a": -4., "b": -1., "c": -1., "d": -2.})
    improved = diagnostics.pair_metrics(pairs, scores, ref)
    assert improved["dpo_loss"] < np.log(2)
    assert improved["pair_accuracy"] == same["pair_accuracy"]
