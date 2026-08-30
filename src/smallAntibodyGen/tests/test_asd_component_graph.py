"""
The indexed similarity graph must equal the naive all-pairs graph, exactly.

The component partition decides which antibodies may appear in a held-out
cohort, so an edge the index fails to find is not an efficiency problem -- it
silently reports MORE independent components than exist and overstates how much
evidence a target can support. Two such defects shipped in the first version of
this script and both pointed the same way:

- **Anchor-star verification.** Each deletion bucket was compared only against
  its first member, so two non-anchor members that neighbour each other but not
  the anchor were never joined. Verified counterexample at >=80%:
  ``{AAAAAB, AAAABA, AAAABB}`` is one component because ``AAAABB`` bridges, and
  the anchor star left ``AAAABA`` isolated.
- **Floating-point flooring.** ``floor((1.0 - 0.80) * length)`` allows one edit
  too few at lengths 5, 10, 15, 20, ... because ``1.0 - 0.80`` evaluates to
  ``0.19999999999999996``.

So the tests here are not spot checks. They enumerate whole small-alphabet
spaces and compare the indexed partition against a brute-force reference, which
is the only check that would have caught either defect without knowing to look
for it.
"""
from __future__ import annotations

import importlib.util
import itertools
import random
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sizer():
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "size_asd_cohorts.py"
    spec = importlib.util.spec_from_file_location("size_asd_cohorts", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _partition(labels: list[int]) -> set[frozenset[int]]:
    """Component membership as a set of frozensets, so root ids need not match."""
    groups: dict[int, set[int]] = {}
    for i, root in enumerate(labels):
        groups.setdefault(root, set()).add(i)
    return {frozenset(v) for v in groups.values()}


# --------------------------------------------------------------------------- #
# Exhaustive equivalence with brute force
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", [4, 5, 6])
@pytest.mark.parametrize("band", [80, 90])
def test_indexed_graph_equals_all_pairs_over_a_whole_binary_space(sizer, length, band):
    """
    Every string of a 2-letter alphabet at this length, all of them at once.

    This is the check that catches a missing edge without anyone guessing which
    edge. The spaces are small (16 to 64 strings) but they are COMPLETE, so every
    adjacency the rule admits is present and every one must be found.
    """
    seqs = ["".join(p) for p in itertools.product("AB", repeat=length)]
    blank = [""] * len(seqs)
    indexed, _ = sizer.components(seqs, blank, band)
    naive = sizer.naive_components(seqs, blank, band)
    assert _partition(indexed) == _partition(naive)


@pytest.mark.parametrize("band", [80, 90])
def test_indexed_graph_equals_all_pairs_over_a_ternary_space(sizer, band):
    """A 3-letter alphabet at length 4: 81 strings, denser adjacency."""
    seqs = ["".join(p) for p in itertools.product("ABC", repeat=4)]
    blank = [""] * len(seqs)
    indexed, _ = sizer.components(seqs, blank, band)
    naive = sizer.naive_components(seqs, blank, band)
    assert _partition(indexed) == _partition(naive)


@pytest.mark.parametrize("seed", range(30))
def test_indexed_graph_equals_all_pairs_on_SPARSE_subsets(sizer, seed):
    """
    Sparse samples from a binary space -- the case with real power.

    Measured: against a reintroduced anchor-star defect the COMPLETE spaces above
    catch nothing (0/6), because a fully enumerated space is so densely connected
    that a star still reaches every member transitively and the partition comes
    out identical. Sparse subsets break that: the same mutant disagrees with
    all-pairs on 29 of these 30 seeds.

    A test that cannot fail against the bug it was written for is decoration, so
    this is the exhaustive-equivalence check that actually guards the invariant.
    """
    rng = random.Random(seed)
    universe = ["".join(p) for p in itertools.product("AB", repeat=7)]
    seqs = rng.sample(universe, 18)
    blank = [""] * len(seqs)
    indexed, _ = sizer.components(seqs, blank, 80)
    naive = sizer.naive_components(seqs, blank, 80)
    assert _partition(indexed) == _partition(naive)


@pytest.mark.parametrize("seed", range(12))
def test_indexed_graph_equals_all_pairs_on_mixed_length_samples(sizer, seed):
    """
    Ragged lengths over a SMALL alphabet, so indels are actually exercised.

    An earlier version drew from 5 letters at lengths 4-9, which is so sparse
    that almost no pair is within the band -- both implementations returned all
    singletons and the comparison was vacuous. Three letters at lengths 5-7
    produces real edges, including length-changing ones.
    """
    rng = random.Random(seed)
    seqs = list({
        "".join(rng.choice("ABC") for _ in range(rng.randint(5, 7)))
        for _ in range(40)
    })
    blank = [""] * len(seqs)
    indexed, _ = sizer.components(seqs, blank, 80)
    naive = sizer.naive_components(seqs, blank, 80)
    assert _partition(indexed) == _partition(naive)
    # guard against the vacuous case returning: this input must contain edges
    assert len(_partition(naive)) < len(seqs), "sample produced no edges at all"


# --------------------------------------------------------------------------- #
# The two shipped defects, pinned by name
# --------------------------------------------------------------------------- #
def test_a_bucket_is_a_graph_not_an_anchor_star(sizer):
    """
    Two non-anchor members neighbouring each other but not the anchor.

    AAAAAB--AAAABB is one edit, AAAABA--AAAABB is one edit, and
    AAAAAB--AAAABA is two, so the three form ONE component through the bridge.
    Comparing only against the first member leaves AAAABA on its own.
    """
    seqs = ["AAAAAB", "AAAABA", "AAAABB"]
    blank = ["", "", ""]
    assert sizer.bounded_levenshtein("AAAAAB", "AAAABB", 3) == 1
    assert sizer.bounded_levenshtein("AAAABA", "AAAABB", 3) == 1
    assert sizer.bounded_levenshtein("AAAAAB", "AAAABA", 3) == 2
    indexed, _ = sizer.components(seqs, blank, 80)
    assert len(set(indexed)) == 1
    assert _partition(indexed) == _partition(sizer.naive_components(seqs, blank, 80))


@pytest.mark.parametrize("length,expected", [(5, 1), (10, 2), (15, 3), (20, 4), (13, 2)])
def test_edit_budget_is_integer_arithmetic(sizer, length, expected):
    """
    `floor((1.0 - 0.80) * L)` is short by one wherever 0.2*L is an integer,
    because 1.0 - 0.80 is 0.19999999999999996. Integer arithmetic is exact.
    """
    import math

    assert sizer.max_edits(length, 80) == expected
    if expected != int(math.floor((1.0 - 0.80) * length)):
        assert int(math.floor((1.0 - 0.80) * length)) == expected - 1


def test_the_80_percent_boundary_is_inclusive(sizer):
    """
    At length 5, one edit is exactly 80% similar and MUST be an edge; two edits
    is 60% and must not be. The float version failed the first of these.
    """
    assert sizer.similar("ABCDE", "ABCDX", 80) is True
    assert sizer.similar("ABCDE", "ABCXY", 80) is False
    # ...and at length 10, exactly two edits is the boundary.
    assert sizer.similar("ABCDEFGHIJ", "ABCDEFGHXY", 80) is True
    assert sizer.similar("ABCDEFGHIJ", "ABCDEFGXYZ", 80) is False


# --------------------------------------------------------------------------- #
# Transitivity, indels, and determinism
# --------------------------------------------------------------------------- #
def test_a_transitive_chain_forms_one_component(sizer):
    """
    Similarity is not transitive; COMPONENTS are. A chain of single edits joins
    endpoints far outside the band, which is the property that makes HER2 one
    component and is exactly what a held-out cohort must respect.
    """
    seqs = ["AAAAAAAAAA", "AAAAAAAAAB", "AAAAAAAABB", "AAAAAAABBB", "AAAAAABBBB"]
    blank = [""] * len(seqs)
    assert not sizer.similar(seqs[0], seqs[-1], 80), "endpoints are NOT directly similar"
    indexed, _ = sizer.components(seqs, blank, 80)
    assert len(set(indexed)) == 1
    assert _partition(indexed) == _partition(sizer.naive_components(seqs, blank, 80))


def test_indels_are_found_not_just_substitutions(sizer):
    """One deletion shifts every later position; positional blocking misses it."""
    a, b = "CARDGGYYYFDYW", "CARDGYYYFDYW"
    assert sizer.bounded_levenshtein(a, b, 3) == 1
    indexed, _ = sizer.components([a, b], ["", ""], 80)
    assert len(set(indexed)) == 1


def test_exact_heavy_chain_joins_dissimilar_hcdr3(sizer):
    """
    The heavy-chain edge is independent of HCDR3 similarity: one antibody cannot
    straddle cohorts even when its loop looks unrelated to its partner's.
    """
    seqs = ["AAAAAAAAAA", "QQQQQQQQQQ"]
    assert not sizer.similar(seqs[0], seqs[1], 80)
    indexed, _ = sizer.components(seqs, ["SHARED_HEAVY", "SHARED_HEAVY"], 80)
    assert len(set(indexed)) == 1


def test_no_bucket_is_ever_skipped(sizer):
    """
    The 20,000-member cap silently dropped edges in exactly the dense libraries
    where they matter. Its absence is reported, not assumed.
    """
    seqs = ["AAAAAAAAAA"] * 1  # trivial input; the contract is the reported field
    _, stats = sizer.components(seqs + ["AAAAAAAAAB"], ["", ""], 80)
    assert stats["buckets_skipped"] == 0
    assert stats["levenshtein_checks"] >= 1


def test_component_ids_are_content_addressed_and_order_independent(sizer):
    """
    Component ids must be a function of content alone. Shuffling the input must
    not change the partition -- otherwise a frozen component id means nothing.
    """
    rng = random.Random(7)
    seqs = list({"".join(rng.choice("ACDE") for _ in range(6)) for _ in range(40)})
    blank = [""] * len(seqs)
    first, _ = sizer.components(seqs, blank, 80)
    order = list(range(len(seqs)))
    rng.shuffle(order)
    shuffled = [seqs[i] for i in order]
    second, _ = sizer.components(shuffled, blank, 80)

    as_sets = {frozenset(seqs[i] for i in group) for group in _partition(first)}
    shuffled_sets = {frozenset(shuffled[i] for i in group) for group in _partition(second)}
    assert as_sets == shuffled_sets


def test_a_singleton_stays_a_singleton(sizer):
    """Degenerate input must not be silently merged."""
    seqs = ["AAAAAAAAAA", "QQQQQQQQQQ", "MMMMMMMMMM"]
    blank = [""] * 3
    indexed, _ = sizer.components(seqs, blank, 80)
    assert len(set(indexed)) == 3


# --------------------------------------------------------------------------- #
# Diversity statistics
# --------------------------------------------------------------------------- #
def test_effective_counts_agree_when_components_are_equal_sized(sizer):
    """With k equal components both measures return k; they diverge only on skew."""
    assert sizer.shannon_effective([10, 10, 10]) == pytest.approx(3.0)
    assert sizer.simpson_effective([10, 10, 10]) == pytest.approx(3.0)


def test_simpson_is_the_conservative_measure_under_skew(sizer):
    """
    Why both are reported: perplexity flatters a skewed distribution, and it is
    the effective SAMPLE SIZE that bounds metric precision.
    """
    skewed = [100, 1, 1, 1, 1]
    assert sizer.simpson_effective(skewed) < sizer.shannon_effective(skewed)
    assert sizer.simpson_effective(skewed) < 1.5
