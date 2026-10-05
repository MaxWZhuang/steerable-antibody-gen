"""The common evaluation: one selector, one panel, every route's checkpoints, read-only.

This module tests ``her2_routes_common_eval`` in slices. The first slice is pure logic: the
panel contract, the selector arithmetic and the neighbourhood bootstrap. Later slices cover
the checkpoint policy, the scoring orchestration and the read-only flight contexts.

What the statistical tests pin, because each was a defect class in review:

* The bootstrap unit is a sequence NEIGHBOURHOOD (the repository's own convention,
  ``her2_routes_eval.paired_delta_spearman``), never a row. This panel carries 102 pairs of
  cores within Hamming distance two, so 100 rows are not 100 independent draws.
* The resampling here must be the SAME resampling ``paired_delta_spearman`` performs for the
  same seed, so a single-subject interval and a pairwise contrast are about the same draws.
* An undefined correlation (a constant score vector) is ``None``, never 0 and never NaN, and a
  contrast involving one is undefined rather than computed from the other half.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_common_eval as ce
from smallAntibodyGen.experiments import her2_routes_eval as eval_lib

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

BASE = "WGGDGFYAMD"


def window(index):
    """Ten canonical residues that differ per index (two positions encode the index)."""
    letters = "ACDEFGHIKLMNPQRSTVWY"
    return BASE[:6] + letters[index // 20 % 20] + letters[index % 20] + BASE[8:]


def panel_rows(count=12, *, offset=0):
    return [{"window": window(index + offset), "value": 5.0 + 0.25 * index}
            for index in range(count)]


# ---------------------------------------------------------------------------
# the panel contract
# ---------------------------------------------------------------------------

def test_a_complete_panel_is_returned_sorted_and_as_copies():
    rows = panel_rows(6)
    shuffled = [rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]
    checked = ce.validate_panel(shuffled, development_windows=[row["window"] for row in rows])
    assert [row["window"] for row in checked] == sorted(row["window"] for row in rows)
    checked[0]["value"] = -999.0
    assert all(row["value"] != -999.0 for row in shuffled), "the caller's rows must not alias"


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: [], "empty"),
    (lambda rows: rows + [dict(rows[0])], "duplicate"),
    (lambda rows: [dict(rows[0], value=float("nan"))] + rows[1:], "finite"),
    (lambda rows: [dict(rows[0], value=float("inf"))] + rows[1:], "finite"),
    (lambda rows: [dict(rows[0], value=None)] + rows[1:], "finite"),
    (lambda rows: [dict(rows[0], value=True)] + rows[1:], "finite"),
    (lambda rows: [dict(rows[0], window="WGGDGFYAM")] + rows[1:], "ten"),
    (lambda rows: [dict(rows[0], window="WGGDGFYAMX")] + rows[1:], "canonical"),
    (lambda rows: [{"window": rows[0]["window"]}] + rows[1:], "malformed"),
    (lambda rows: ["not a row"] + rows[1:], "malformed"),
])
def test_a_malformed_panel_is_refused_with_the_reason(mutate, message):
    rows = panel_rows(6)
    with pytest.raises(ValueError, match=message):
        ce.validate_panel(mutate(rows), development_windows=[row["window"] for row in rows])


def test_equal_panels_do_not_establish_complete_panels():
    """The service silently drops a development window with no qualified outcome.

    Two flights can therefore agree on the same INCOMPLETE subset, so coverage of the frozen
    development list is checked in both directions.
    """
    rows = panel_rows(6)
    development = [row["window"] for row in rows]
    with pytest.raises(ValueError, match="missing from the panel"):
        ce.validate_panel(rows[:-1], development_windows=development)
    with pytest.raises(ValueError, match="not in the frozen development"):
        ce.validate_panel(rows + panel_rows(1, offset=40), development_windows=development)


def test_the_panel_digest_moves_with_every_input_that_defines_the_panel():
    rows = ce.validate_panel(panel_rows(6), development_windows=[w["window"] for w in panel_rows(6)])
    labels = ["a", "a", "b", "c", "c", "d"]
    context = {"assay": "alphabind_her2:affinity:HER2", "split_sha256": "s" * 64,
               "scaffold_sha256": "c" * 64, "sites": 10}
    base = ce.panel_digest(rows, labels, context=context)
    assert base == ce.panel_digest(rows, labels, context=dict(context))
    moved_value = [dict(row) for row in rows]
    moved_value[2]["value"] += 1e-9
    assert ce.panel_digest(moved_value, labels, context=context) != base
    assert ce.panel_digest(rows, labels[:-1] + ["e"], context=context) != base
    for key, value in (("assay", "other"), ("split_sha256", "t" * 64),
                       ("scaffold_sha256", "d" * 64), ("sites", 9)):
        assert ce.panel_digest(rows, labels, context=dict(context, **{key: value})) != base


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

def test_the_three_selectors_are_defined_once_and_the_primary_is_the_legacy_one():
    high = np.array([-1.0, -2.0, -3.0])
    null = np.array([-1.5, -1.0, -5.0])
    assert ce.PRIMARY_SELECTOR == "high"
    assert ce.SELECTORS == ("high", "null", "ratio")
    assert np.array_equal(ce.selector_vector(high, null, "high"), high)
    assert np.array_equal(ce.selector_vector(high, null, "null"), null)
    assert np.array_equal(ce.selector_vector(high, null, "ratio"), high - null)
    with pytest.raises(ValueError, match="Unknown selector"):
        ce.selector_vector(high, null, "legacy")
    with pytest.raises(ValueError, match="misaligned"):
        ce.selector_vector(high, null[:2], "ratio")


# ---------------------------------------------------------------------------
# Spearman and the neighbourhood bootstrap
# ---------------------------------------------------------------------------

def test_a_constant_score_has_no_correlation_and_a_nonfinite_one_is_refused():
    outcomes = np.arange(6.0)
    assert ce.spearman_or_none(np.zeros(6), outcomes) is None
    assert ce.spearman_or_none(np.arange(6.0), outcomes) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-finite"):
        ce.spearman_or_none(np.array([0.0, 1.0, float("nan"), 3.0, 4.0, 5.0]), outcomes)


def make_problem(seed=3, groups=8, per_group=3):
    rng = np.random.default_rng(seed)
    labels = [f"g{index // per_group}" for index in range(groups * per_group)]
    outcomes = rng.normal(size=groups * per_group)
    good = outcomes + rng.normal(scale=0.7, size=outcomes.size)
    weak = outcomes * 0.2 + rng.normal(size=outcomes.size)
    return labels, outcomes, good, weak


def test_the_resampling_is_exactly_the_one_the_repository_bootstrap_performs():
    """A single-subject interval and a pairwise contrast must be about the same draws.

    The per-draw difference of the two subjects' correlations, over the draws where both are
    defined, has the same quantile interval ``paired_delta_spearman`` reports for the same
    seed -- which can only hold if the sampled neighbourhoods are the same sequence.
    """
    labels, outcomes, good, weak = make_problem()
    draws = ce.resampled_correlations({"good": good, "weak": weak}, outcomes, labels,
                                      seed=11, draws=300)
    both = np.isfinite(draws["good"]) & np.isfinite(draws["weak"])
    difference = draws["good"][both] - draws["weak"][both]
    reference = eval_lib.paired_delta_spearman(good, weak, outcomes, groups=labels, seed=11,
                                               draws=300)
    assert int(both.sum()) == reference["usable_draws"]
    assert np.quantile(difference, 0.025) == pytest.approx(reference["interval"][0])
    assert np.quantile(difference, 0.975) == pytest.approx(reference["interval"][1])


def test_the_bootstrap_reports_observed_rho_interval_and_usable_draws_per_subject():
    labels, outcomes, good, weak = make_problem()
    result = ce.neighbourhood_bootstrap({"good": good, "weak": weak}, outcomes, labels,
                                        seed=5, draws=400)
    assert result["neighbourhoods"] == 8 and result["rows"] == 24
    assert result["unit"].startswith("sequence neighbourhood")
    assert result["draws"] == 400 and result["confidence"] == 0.95
    for name, vector in (("good", good), ("weak", weak)):
        block = result["subjects"][name]
        assert block["rho"] == pytest.approx(eval_lib.spearman(vector, outcomes))
        low, high = block["interval"]
        assert low < block["rho"] < high
        assert 0 < block["usable_draws"] <= 400
    assert result["subjects"]["good"]["rho"] > result["subjects"]["weak"]["rho"]


def test_the_bootstrap_is_deterministic_in_its_seed():
    labels, outcomes, good, weak = make_problem()
    first = ce.neighbourhood_bootstrap({"good": good}, outcomes, labels, seed=5, draws=200)
    again = ce.neighbourhood_bootstrap({"good": good}, outcomes, labels, seed=5, draws=200)
    other = ce.neighbourhood_bootstrap({"good": good}, outcomes, labels, seed=6, draws=200)
    assert first == again
    assert first["subjects"]["good"]["interval"] != other["subjects"]["good"]["interval"]


def test_an_undefined_correlation_is_none_everywhere_never_zero_or_nan():
    labels, outcomes, good, _ = make_problem()
    result = ce.neighbourhood_bootstrap({"constant": np.zeros(outcomes.size), "good": good},
                                        outcomes, labels, seed=5, draws=100)
    constant = result["subjects"]["constant"]
    assert constant == {"rho": None, "interval": None, "usable_draws": 0}
    assert result["subjects"]["good"]["rho"] is not None
    assert "NaN" not in repr(result) and not any(
        isinstance(value, float) and math.isnan(value)
        for block in result["subjects"].values() for value in block.values()
        if not isinstance(value, (list, type(None))))


def test_one_neighbourhood_is_not_an_interval():
    _, outcomes, good, _ = make_problem()
    with pytest.raises(ValueError, match="more than one neighbourhood"):
        ce.neighbourhood_bootstrap({"good": good}, outcomes, ["only"] * outcomes.size,
                                   seed=1, draws=10)


def test_misaligned_inputs_are_refused():
    labels, outcomes, good, _ = make_problem()
    with pytest.raises(ValueError, match="misaligned"):
        ce.neighbourhood_bootstrap({"good": good[:-1]}, outcomes, labels, seed=1, draws=10)
    with pytest.raises(ValueError, match="misaligned"):
        ce.neighbourhood_bootstrap({"good": good}, outcomes, labels[:-1], seed=1, draws=10)


def test_a_contrast_with_an_undefined_half_is_undefined_not_half_computed():
    labels, outcomes, good, _ = make_problem()
    undefined = ce.paired_contrast(good, np.zeros(outcomes.size), outcomes, labels, seed=2,
                                   draws=100)
    assert undefined["delta"] is None and undefined["interval"] is None
    assert "constant" in undefined["reason"]
    defined = ce.paired_contrast(good, good[::-1].copy(), outcomes, labels, seed=2, draws=100)
    reference = eval_lib.paired_delta_spearman(good, good[::-1].copy(), outcomes, groups=labels,
                                               seed=2, draws=100)
    assert defined["delta"] == pytest.approx(reference["delta"])
    assert defined["interval"] == pytest.approx(reference["interval"])
    assert defined["reason"] is None


# ---------------------------------------------------------------------------
# the bootstrap unit: connected components of the near-neighbour graph
# ---------------------------------------------------------------------------

def test_every_pair_within_the_radius_shares_a_neighbourhood_even_through_a_chain():
    """The repository's greedy-cover labels do NOT do this: on the frozen development list they
    produce 59 groups and separate 45 of the 102 pairs within Hamming distance two, so those
    pairs were resampled as independent units. A connected component keeps every such pair, and
    every chain of them, together.
    """
    chain = ["AAAAAAAAAA", "AAAAAAAACC", "AAAAAACCCC", "AAAACCCCCC"]   # a~b, b~c, c~d; a !~ c
    far = "WWWWWWWWWW"
    windows = chain + [far]
    labels = ce.neighbourhood_components(windows, radius=2)
    assert len({labels[i] for i in range(4)}) == 1, "a chain is one component"
    assert labels[4] != labels[0]
    for left in range(len(windows)):
        for right in range(left + 1, len(windows)):
            near = sum(a != b for a, b in zip(windows[left], windows[right])) <= 2
            if near:
                assert labels[left] == labels[right]


def test_the_components_are_independent_of_input_order_and_labelled_canonically():
    rng = np.random.default_rng(2)
    letters = np.array(list("ACDEFGHIKL"))
    windows = sorted({"".join(rng.choice(letters, 10)) for _ in range(60)})
    shuffled = list(windows)
    rng.shuffle(shuffled)
    forward = dict(zip(windows, ce.neighbourhood_components(windows, radius=3)))
    other = dict(zip(shuffled, ce.neighbourhood_components(shuffled, radius=3)))
    assert forward == other
    for label in set(forward.values()):
        members = [w for w, l in forward.items() if l == label]
        assert label == min(members), "the label is the smallest window of its component"


def test_the_components_refuse_duplicates_and_ragged_windows():
    with pytest.raises(ValueError, match="duplicate"):
        ce.neighbourhood_components(["AAAAAAAAAA", "AAAAAAAAAA"], radius=2)
    with pytest.raises(ValueError, match="same length"):
        ce.neighbourhood_components(["AAAAAAAAAA", "AAAAAAAAA"], radius=2)


def test_the_grouping_record_reports_the_largest_component_honestly():
    windows = ["AAAAAAAAAA", "AAAAAAAACC", "AAAAAACCCC", "WWWWWWWWWW", "YYYYYYYYYY"]
    labels = ce.neighbourhood_components(windows, radius=2)
    record = ce.grouping_record(labels, radius=2)
    assert record == {"method": "connected_components", "radius": 2, "groups": 3,
                      "largest": 3, "rows": 5, "size_histogram": {"1": 2, "3": 1}}
