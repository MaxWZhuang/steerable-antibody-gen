"""Statistical contracts for the v2 contrast-benchmark endpoints.

Every test here pins a property the *protocol* must have, independent of any
checkpoint: the floor a trivial scorer earns, the weighting that keeps one
oversized panel from becoming the benchmark, and the determinism of every
resampling step. See docs/HCDR3_CONTRAST_BENCHMARK.md for why the v1
mean-group-concordance headline was replaced.
"""
from __future__ import annotations

import pytest

from smallAntibodyGen.evaluation.contrasts import concordance
from smallAntibodyGen.evaluation.contrast_statistics import (
    cluster_bootstrap, drop_largest_group, equal_group_by_target, group_stats,
    paired_contrast, pair_pooled_by_target, permutation_null, power_simulation,
    primary_endpoint, primary_endpoint_se, weight_concentration,
)


def make_group(group_id, target, measurements, native, substituted=None, direction="higher"):
    """One synthetic contrast group whose cached ranking is internally consistent."""
    substituted = list(substituted if substituted is not None else native)
    variants = [{"hcdr3": "A" * (i + 3), "measurement": float(m), "is_strong_binder": False,
                 "record_id": "{}/{}".format(group_id, i),
                 "scores": {"native": float(n), "substituted_unmeasured": float(s)}}
                for i, (m, n, s) in enumerate(zip(measurements, native, substituted))]
    values = [v["measurement"] for v in variants]
    manifest_group = {"group_id": group_id, "target": target, "direction": direction}
    scores_group = {"group_id": group_id, "target": target, "fold": "validation",
                    "variants": variants,
                    "native_ranking": concordance(values, [v["scores"]["native"] for v in variants],
                                                  direction=direction)}
    return manifest_group, scores_group


def make_docs(specs):
    pairs = [make_group(*spec) for spec in specs]
    return ({"groups": [p[0] for p in pairs]}, {"groups": [p[1] for p in pairs]})


def ladder(n):
    """n variants whose measurements strictly increase, so every pair is comparable."""
    return [float(i) for i in range(n)]


# --- the floor (change-control Rule 4: no metric claim without its floor) ---

def test_constant_scorer_earns_exactly_the_floor():
    manifest, scores = make_docs([("g1", "t1", ladder(4), [7.0] * 4),
                                  ("g2", "t2", ladder(3), [0.0] * 3)])
    stats = group_stats(scores, manifest)
    assert primary_endpoint(stats) == pytest.approx(0.5)
    assert all(v == pytest.approx(0.5) for v in pair_pooled_by_target(stats).values())


def test_perfect_and_reversed_scorers_reach_the_endpoints():
    for native, expected in ((ladder(5), 1.0), (list(reversed(ladder(5))), 0.0)):
        manifest, scores = make_docs([("g1", "t1", ladder(5), native)])
        assert primary_endpoint(group_stats(scores, manifest)) == pytest.approx(expected)


def test_lower_is_stronger_direction_is_honoured():
    manifest, scores = make_docs([("g1", "t1", ladder(4), list(reversed(ladder(4))), None, "lower")])
    assert primary_endpoint(group_stats(scores, manifest)) == pytest.approx(1.0)


# --- weighting: the v1 defect this module exists to fix ---

def test_one_huge_panel_cannot_dominate_the_primary_endpoint():
    # t1 holds a 12-variant perfect panel (66 pairs); t2 holds two reversed 3-variant
    # panels (3 pairs each). Pooling every pair would report ~0.90; the primary
    # endpoint weights targets equally and reports the floor.
    manifest, scores = make_docs([("big", "t1", ladder(12), ladder(12)),
                                  ("s1", "t2", ladder(3), list(reversed(ladder(3)))),
                                  ("s2", "t2", ladder(3), list(reversed(ladder(3))))])
    stats = group_stats(scores, manifest)
    assert primary_endpoint(stats) == pytest.approx(0.5)
    pooled = pair_pooled_by_target(stats)
    assert pooled["t1"] == pytest.approx(1.0) and pooled["t2"] == pytest.approx(0.0)


def test_groups_are_weighted_equally_within_a_target():
    manifest, scores = make_docs([("big", "t1", ladder(10), ladder(10)),
                                  ("small", "t1", ladder(2), list(reversed(ladder(2))))])
    stats = group_stats(scores, manifest)
    assert equal_group_by_target(stats)["t1"] == pytest.approx(0.5)
    assert pair_pooled_by_target(stats)["t1"] == pytest.approx(45 / 46)


def test_weight_concentration_reports_the_dominant_panel_share():
    manifest, scores = make_docs([("big", "t1", ladder(10), ladder(10)),
                                  ("small", "t1", ladder(2), ladder(2))])
    stats = group_stats(scores, manifest)
    report = weight_concentration(stats)["t1"]
    assert report["groups"] == 2 and report["comparable_pairs"] == 46
    assert report["largest_group_pair_share"] == pytest.approx(45 / 46)
    assert report["largest_group_id"] == "big"
    assert drop_largest_group(stats) == [s for s in stats if s.group_id == "small"]


def test_min_variants_keeps_single_pair_groups_but_drops_unrankable_ones():
    manifest, scores = make_docs([("pair", "t1", ladder(2), ladder(2)),
                                  ("solo", "t1", ladder(1), [1.0])])
    assert {s.group_id for s in group_stats(scores, manifest)} == {"pair"}
    assert {s.group_id for s in group_stats(scores, manifest, min_variants=3)} == set()


# --- inference ---

def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    manifest, scores = make_docs([("g{}".format(i), "t1", ladder(3),
                                   ladder(3) if i % 2 else list(reversed(ladder(3))))
                                  for i in range(24)])
    stats = group_stats(scores, manifest)
    first = cluster_bootstrap(stats, n_resamples=500, seed=7)
    assert first == cluster_bootstrap(stats, n_resamples=500, seed=7)
    assert first != cluster_bootstrap(stats, n_resamples=500, seed=8)
    assert first["low"] <= first["estimate"] <= first["high"]
    assert first["estimate"] == pytest.approx(primary_endpoint(stats))


def test_permutation_null_centres_on_the_floor():
    manifest, scores = make_docs([("g{}".format(i), "t1", ladder(4), ladder(4))
                                  for i in range(12)])
    null = permutation_null(scores, manifest, n_permutations=400, seed=3)
    assert null["null_mean"] == pytest.approx(0.5, abs=0.05)
    assert null["low"] < 0.5 < null["high"]
    assert null["p_value"] < 0.05  # a perfect scorer must beat its own permuted null
    assert null == permutation_null(scores, manifest, n_permutations=400, seed=3)


def test_power_rises_with_effect_size_and_is_labelled_an_upper_bound():
    # Mixed panels, so the observed groups actually carry between-group variance.
    manifest, scores = make_docs([("g{}".format(i), "t1", ladder(3),
                                   ladder(3) if i % 3 else list(reversed(ladder(3))))
                                  for i in range(40)])
    stats = group_stats(scores, manifest)
    weak = power_simulation(stats, effect=0.01, n_simulations=300, seed=5)
    strong = power_simulation(stats, effect=0.20, n_simulations=300, seed=5)
    assert weak["power"] < strong["power"]
    assert "upper bound" in weak["caveat"]
    assert primary_endpoint_se(stats) > 0


def test_paired_contrast_compares_native_against_substituted_on_the_same_groups():
    manifest, scores = make_docs([("g1", "t1", ladder(4), ladder(4), list(reversed(ladder(4)))),
                                  ("g2", "t1", ladder(4), ladder(4), list(reversed(ladder(4))))])
    native = group_stats(scores, manifest, score_key="native")
    substituted = group_stats(scores, manifest, score_key="substituted_unmeasured")
    result = paired_contrast(native, substituted, n_resamples=400, seed=11)
    assert result["estimate"] == pytest.approx(1.0)
    assert result["n_groups"] == 2


# --- the artifact must not have drifted underneath the analysis ---

def test_recomputation_disagreeing_with_the_frozen_artifact_is_fatal():
    manifest, scores = make_docs([("g1", "t1", ladder(4), ladder(4))])
    scores["groups"][0]["native_ranking"]["concordant"] += 1
    with pytest.raises(ValueError, match="disagrees with the frozen"):
        group_stats(scores, manifest)
    assert group_stats(scores, manifest, verify=False)


def test_missing_direction_is_fatal_rather_than_assumed():
    manifest, scores = make_docs([("g1", "t1", ladder(3), ladder(3))])
    del manifest["groups"][0]["direction"]
    with pytest.raises(ValueError, match="direction"):
        group_stats(scores, manifest)
