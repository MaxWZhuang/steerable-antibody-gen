"""Differential tests against the specification's standard-library reference suite.

The reference module is loaded **by path** so the supplied blueprint artifact is
not modified, not imported as a package member and not copied. Every production
implementation that has a scalar reference is compared to it here; where the
argument orders differ -- ``average_precision(labels, scores)`` there,
``average_precision(scores, labels)`` in the campaign -- the difference is
exercised deliberately, because reversing them raises rather than returning a
plausible number.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_metrics as metrics
from smallAntibodyGen.experiments import her2_nf_mixture as mixture
from smallAntibodyGen.experiments import her2_nf_objectives as objectives
from smallAntibodyGen.experiments import her2_nf_proximity as proximity

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = ROOT / "specs" / "her2_next_flight" / "reference_math.py"

#: ``specs/*`` is git-ignored, so the blueprint artifact is local-only and is not
#: shipped with the package. Loading it unconditionally failed the DEFAULT suite
#: with ``FileNotFoundError`` in any tree but this one. It is not copied into
#: ``tests/`` to fix that: copying would publish a local-only specification
#: artifact to a pushed repository. These differential tests are opted out
#: explicitly instead, so their absence reads as a skip with a reason rather than
#: as a broken suite -- and wherever the artifact IS present, every one of them
#: still runs against it.
pytestmark = pytest.mark.skipif(
    not REFERENCE_PATH.is_file(),
    reason=(f"{REFERENCE_PATH} is absent: the reference oracle lives under the git-ignored "
            "specs/ tree, and these differential tests run only where it exists"))


@pytest.fixture(scope="module")
def reference():
    spec = importlib.util.spec_from_file_location("nf_reference_math", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_artifact_is_present_and_unmodified_by_this_import(reference):
    before = REFERENCE_PATH.read_bytes()
    assert hasattr(reference, "ipo_loss")
    assert REFERENCE_PATH.read_bytes() == before


def test_normalized_dpo_multiplier_matches_the_reference(reference):
    for beta in (0.1, 0.5, 1.0, 3.0):
        assert objectives.dpo_scale(beta) == pytest.approx(4.0 * 5.0 / beta, rel=0, abs=0)
        for delta in (-2.0, -0.25, 0.0, 0.75, 4.0):
            expected = reference.normalized_dpo_loss(delta, beta)
            observed = objectives.dpo_scale(beta) * reference.dpo_loss(delta, beta)
            assert observed == pytest.approx(expected, rel=1e-12)


def test_tail_penalty_matches_the_reference_pointwise(reference):
    import torch
    values = [-1.0, 0.0, 0.5, math.log(2.0), math.log(5.0), math.log(10.0), math.log(50.0),
              math.log(100.0), 6.0]
    observed, components = objectives.tail_penalty_per_row(
        torch.tensor(values, dtype=torch.float64))
    for position, value in enumerate(values):
        assert float(observed[position]) == pytest.approx(reference.tail_penalty(value),
                                                          rel=1e-12, abs=1e-12)
    assert float(components["tenfold"][values.index(math.log(10.0))]) == pytest.approx(
        1.0 / 0.01, rel=1e-12)


def test_mixture_log_probability_matches_the_reference_including_endpoints(reference):
    parent = np.array([-3.0, -10.0, -0.5])
    policy = np.array([-2.0, -30.0, -0.75])
    for alpha in mixture.ALPHA_GRID:
        observed = mixture.mixture_log_probability(parent, policy, alpha)
        for position in range(parent.size):
            assert observed[position] == pytest.approx(
                reference.mixture_log_probability(parent[position], policy[position], alpha),
                rel=1e-12, abs=1e-12)


def test_mixture_handles_a_zero_probability_component_without_nan():
    parent = np.array([-4.0])
    policy = np.array([-np.inf])
    value = mixture.mixture_log_probability(parent, policy, 0.89)
    assert np.isfinite(value).all()
    # The certified floor: the mixture keeps at least (1 - alpha) of the parent.
    assert float(value[0]) == pytest.approx(math.log(0.11) + (-4.0), rel=1e-12)


def test_average_precision_matches_the_reference_and_the_argument_orders_differ(reference):
    generator = np.random.default_rng(7)
    for trial in range(40):
        size = int(generator.integers(2, 60))
        labels = generator.integers(0, 2, size)
        if labels.sum() == 0:
            labels[0] = 1
        scores = generator.integers(0, 4, size).astype(float)  # heavy ties on purpose
        assert metrics.average_precision(scores, labels) == pytest.approx(
            reference.average_precision(list(labels), list(scores)), rel=0, abs=1e-12)
    # Reversing the arguments is a mistake that must raise rather than return a number.
    with pytest.raises(Exception):
        metrics.average_precision(np.array([0, 1, 1]), np.array([0.1, 5.0, -2.0]))


def test_constant_scores_give_the_prevalence(reference):
    labels = np.array([1, 0, 0])
    assert metrics.average_precision(np.zeros(3), labels) == pytest.approx(1.0 / 3.0)
    assert reference.average_precision(list(labels), [0.0, 0.0, 0.0]) == pytest.approx(1.0 / 3.0)


def test_expected_distinct_yield_matches_the_reference_at_both_precision_ends(reference):
    values = [-1e-9, -0.1, -math.log(2.0), -5.0, -30.0, -700.0]
    for draws in (0, 1, 10, 10_000, 1_000_000):
        assert metrics.expected_distinct_yield(values, draws) == pytest.approx(
            reference.expected_distinct(values, draws), rel=1e-12, abs=1e-12)


def test_wilson_and_paired_t_match_the_reference(reference):
    for count, total in ((0, 50_000), (370, 50_000), (50_000, 50_000), (1, 3)):
        low, high = reference.wilson(count, total)
        block = metrics.wilson_interval(count, total)
        assert block["lower"] == pytest.approx(low, rel=1e-12, abs=1e-15)
        assert block["upper"] == pytest.approx(high, rel=1e-12, abs=1e-15)
    values = [-132.0, -66.0, 0.5]
    centre, low, high = reference.paired_t3(values)
    block = metrics.paired_t(values)
    assert block["degrees_of_freedom"] == 2
    assert (block["mean"], block["lower"], block["upper"]) == pytest.approx(
        (centre, low, high), rel=1e-12)


def test_projection_search_agrees_with_the_reference_witness_search(reference):
    generator = np.random.default_rng(11)
    panel = generator.integers(0, 3, size=(25, 10))
    queries = generator.integers(0, 3, size=(60, 10))
    for radius in (1, 2):
        lookup = reference.projection_index([tuple(row) for row in panel], radius)
        expected = [reference.within_radius(tuple(row), [tuple(p) for p in panel], radius,
                                            lookup)[0] for row in queries]
        block = proximity.exact_nearest_within(queries, panel, radius=radius)
        assert [bool(value) for value in block["within_radius"]] == expected
        # A non-hit certifies radius + 1 and nothing tighter.
        misses = ~block["within_radius"]
        assert set(block["certified_min_distance"][misses].tolist()) <= {radius + 1}


def test_crossing_brackets_match_the_reference(reference):
    budgets = [1e3, 1e4, 1e5, 1e6]
    differences = [1.0, -0.5, -0.2, 3.0]
    expected = reference.crossing_brackets(budgets, differences)
    observed = metrics.crossing_brackets(budgets, differences)
    assert observed["brackets"] == expected["brackets"]
    assert observed["first_nonzero_sign"] == expected["first_nonzero_sign"]
    assert observed["domain"] == expected["domain"]


def test_production_radius_two_uses_forty_five_masks_and_radius_one_uses_ten():
    assert len(proximity.masks_of_size(2)) == 45
    assert len(proximity.masks_of_size(1)) == 10
    # Sweeping sizes 0..r is what makes the first hit the EXACT distance.
    assert proximity.mask_count(2) == 1 + 10 + 45
