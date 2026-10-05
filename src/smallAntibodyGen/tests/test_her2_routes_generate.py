"""The scalable index, and the fill orders that must differ from each other.

Two classes of defect are pinned here.

The first is scale. Greedy clustering of 100,000 generated windows written as a
Python scan is ``draws x medoids`` Hamming computations and the nearest-training
query against the 367,042-row display library is another 37 billion; neither
finishes inside a stage. The deletion-key form is exact for the declared radius, and
these tests check the exactness against the brute-force answer on sizes where both
can run.

The second is that ``easy_first`` and ``hard_first`` were the same procedure: the
confidence fed to the order resolver was ``rng.random()``, so the two labels named
two random orders and the comparison between them was a comparison of two seeds. The
order here is derived from a supplied distribution, and the test asserts the two
strategies invert each other.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_decoding as decoding
from smallAntibodyGen.experiments import her2_routes_split as split_lib
from smallAntibodyGen.experiments.her2_routes_controls import hamming

WINDOWS = ["WGGDGFYAMD", "WGGDGFYAMA", "WAGDGFYAMD", "AAAAAAAAAA", "AAAAAAAAAW",
           "CCCCCCCCCC", "WGGDGFYAMV"]


def brute_force_medoids(windows, radius):
    medoids = []
    for window in sorted(set(windows)):
        if all(hamming(window, chosen) > radius for chosen in medoids):
            medoids.append(window)
    return tuple(medoids)


# --- the deletion-key index -----------------------------------------------

def test_deletion_keys_are_exact_for_the_declared_radius():
    """Two windows are within r iff they share a key with at most r blanks."""
    for left, right in itertools.combinations(WINDOWS, 2):
        distance = hamming(left, right)
        for radius in (0, 1, 2, 3):
            shared = bool(set(split_lib.deletion_keys(left, radius))
                          & set(split_lib.deletion_keys(right, radius)))
            assert shared == (distance <= radius), (
                f"{left}/{right} at distance {distance}: the key test disagreed at radius "
                f"{radius}, so the hash form is not exact")


def test_greedy_medoids_agree_with_the_brute_force_cover():
    for radius in (0, 1, 2):
        fresh, existing = split_lib.greedy_medoids(WINDOWS, radius=radius)
        assert existing == ()
        assert fresh == brute_force_medoids(WINDOWS, radius), (
            f"the hashed cover at radius {radius} differs from the pairwise one")


def test_greedy_medoids_respect_pre_existing_medoids():
    fresh, existing = split_lib.greedy_medoids(
        WINDOWS, radius=1, existing=["WGGDGFYAMD"])
    assert existing == ("WGGDGFYAMD",)
    assert "WGGDGFYAMD" not in fresh
    assert "WGGDGFYAMA" not in fresh, "a window inside an existing medoid's radius is covered"
    assert "CCCCCCCCCC" in fresh


def test_greedy_medoids_finish_at_a_hundred_thousand_windows():
    """The size the plan asks for. A pairwise cover here does not finish."""
    rng = np.random.default_rng(20260927)
    alphabet = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
    codes = alphabet[rng.integers(0, alphabet.size, (100000, 10))]
    windows = [row.tobytes().decode("ascii") for row in codes]
    import time
    started = time.perf_counter()
    fresh, _ = split_lib.greedy_medoids(windows, radius=1)
    elapsed = time.perf_counter() - started
    assert elapsed < 120.0, f"the hashed cover took {elapsed:.1f}s at 100k windows"
    assert len(fresh) > 0
    # Every window must be covered by some medoid at the declared radius.
    index = split_lib.HammingIndex(fresh, radius=1)
    sample = [windows[int(position)] for position in rng.integers(0, len(windows), 200)]
    assert all(index.nearest(window) >= 0 for window in sample)


def test_the_hamming_index_matches_min_distance_to_library():
    """Two implementations of one convention. ``-1`` means beyond the radius in both."""
    bank = WINDOWS[:4]
    queries = WINDOWS
    index = split_lib.HammingIndex(bank, radius=2)
    vectorised = split_lib.min_distance_to_library(queries, bank, max_distance=2)
    for window, expected in zip(queries, vectorised):
        assert index.nearest(window) == int(expected)


def test_assign_to_medoids_matches_the_pairwise_occupancy():
    medoids = brute_force_medoids(WINDOWS, 1)
    block = split_lib.assign_to_medoids(WINDOWS, medoids, radius=1)
    manual = {}
    unassigned = 0
    for window in WINDOWS:
        best, chosen = None, None
        for position, medoid in enumerate(medoids):
            distance = hamming(window, medoid)
            if distance <= 1 and (best is None or distance < best
                                  or (distance == best and medoid < medoids[chosen])):
                best, chosen = distance, position
        if chosen is None:
            unassigned += 1
            continue
        manual[str(chosen)] = manual.get(str(chosen), 0) + 1
    assert block["counts"] == manual
    assert block["unassigned"] == unassigned
    assert block["clusters_total"] == len(medoids)


# --- the fill orders ------------------------------------------------------

def test_easy_first_and_hard_first_invert_each_other_on_a_real_confidence():
    sites = tuple(("H", 98 + offset) for offset in range(4))
    confidence = [0.1, 0.9, 0.4, 0.7]
    easy = decoding.resolve_order(sites, strategy=decoding.ORDER_EASY_FIRST,
                                  confidence=confidence)
    hard = decoding.resolve_order(sites, strategy=decoding.ORDER_HARD_FIRST,
                                  confidence=confidence)
    assert easy == tuple(reversed(hard))
    assert easy[0] == sites[1], "the most confident site goes first"
    assert hard[0] == sites[0], "the least confident site goes first"


def test_a_confidence_free_strategy_refuses_to_guess_one():
    sites = tuple(("H", 98 + offset) for offset in range(4))
    with pytest.raises(ValueError, match="needs one confidence value per site"):
        decoding.resolve_order(sites, strategy=decoding.ORDER_EASY_FIRST)
    with pytest.raises(ValueError, match="needs an explicit generator"):
        decoding.resolve_order(sites, strategy=decoding.ORDER_RANDOM)


def test_the_remask_schedule_is_declared_and_bounded():
    rng = np.random.default_rng(1)
    schedule = decoding.remask_schedule(10, rounds=3, fraction=0.3, rng=rng)
    assert len(schedule) == 3
    assert all(len(block) == 3 for block in schedule)
    assert all(0 <= position < 10 for block in schedule for position in block)
    assert all(len(set(block)) == len(block) for block in schedule)


def test_the_draw_accounting_keeps_three_numbers_apart():
    block = decoding.draw_accounting(draws=100, forward_calls=1000, seconds=2.0,
                                     mode="full_mask")
    assert block["forward_calls_per_draw"] == pytest.approx(10.0)
    assert block["seconds_per_draw"] == pytest.approx(0.02)
    assert "never pooled into one" in block["distinction"]
    with pytest.raises(ValueError, match="Unknown draw mode"):
        decoding.draw_accounting(draws=1, forward_calls=1, seconds=1.0, mode="invented")
