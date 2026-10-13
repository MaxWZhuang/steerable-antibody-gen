"""Whole-mask draws: the full/partial mixture, the two geometries, and RNG separation.

The statistical assertions use fixed seeds and bounds several standard deviations wide, so they
are deterministic and are not tuned to the draws the current implementation happens to make.
"""
from __future__ import annotations

import inspect
import itertools
import random
from collections import Counter

import numpy as np
import pytest

from smallAntibodyGen.experiments.her2_multiscale import masks

HEAVY_LENGTH = 40
LIGHT_LENGTH = 20
REGION = tuple(("H", index) for index in range(3, 9))  # six coordinates


def generators(seed):
    return np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])


def draw(region=REGION, *, rngs=None, seed=0, **options):
    count_rng, position_rng = rngs if rngs is not None else generators(seed)
    return masks.draw_mask(region, heavy_length=HEAVY_LENGTH, light_length=LIGHT_LENGTH,
                           count_rng=count_rng, position_rng=position_rng, **options)


def states(rngs):
    return tuple(rng.bit_generator.state for rng in rngs)


# --- region validation ----------------------------------------------------

def test_a_region_is_sorted_heavy_then_light_ascending():
    region = [("L", 2), ("H", 9), ("L", 0), ("H", 1)]
    assert masks.canonical_region(region, heavy_length=HEAVY_LENGTH, light_length=LIGHT_LENGTH) \
        == (("H", 1), ("H", 9), ("L", 0), ("L", 2))


def test_numpy_integers_are_accepted_and_returned_as_python_ints():
    result = masks.canonical_region((("H", np.int64(4)),), heavy_length=HEAVY_LENGTH,
                                    light_length=LIGHT_LENGTH)
    assert result == (("H", 4),) and type(result[0][1]) is int


BAD_REGIONS = [
    ((), "no coordinates"),
    (None, "tuple or list"),
    ("H3", "tuple or list"),
    ({("H", 1)}, "tuple or list"),
    (("H", 1), "pair"),
    ((("H", 1, 2),), "pair"),
    ((("X", 1),), "chain"),
    ((("h", 1),), "chain"),
    (((b"H", 1),), "chain"),
    ((("H", True),), "integer"),
    ((("H", 1.0),), "integer"),
    ((("H", "1"),), "integer"),
    ((("H", None),), "integer"),
    ((("H", np.bool_(True)),), "integer"),
    ((("H", -1),), "non-negative"),
    ((("H", HEAVY_LENGTH),), "outside"),
    ((("L", LIGHT_LENGTH),), "outside"),
    ((("H", 1), ("H", 1)), "Duplicate"),
]


@pytest.mark.parametrize("region, message", BAD_REGIONS)
def test_malformed_regions_are_refused(region, message):
    with pytest.raises(ValueError, match=message):
        masks.canonical_region(region, heavy_length=HEAVY_LENGTH, light_length=LIGHT_LENGTH)


def test_a_light_coordinate_needs_a_light_chain():
    with pytest.raises(ValueError, match="outside"):
        masks.canonical_region((("L", 0),), heavy_length=HEAVY_LENGTH, light_length=0)


@pytest.mark.parametrize("length", [-1, True, 1.5, "40", None])
def test_chain_lengths_must_be_non_negative_integers(length):
    with pytest.raises(ValueError, match="non-negative integer"):
        masks.canonical_region(REGION, heavy_length=length, light_length=LIGHT_LENGTH)


# --- the full probability --------------------------------------------------

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, 2,
                                   True, False, "0.5", None, [0.5]])
def test_invalid_full_probabilities_are_refused(value):
    with pytest.raises(ValueError, match="full_probability"):
        masks.validated_full_probability(value)


@pytest.mark.parametrize("value", [0, 0.0, 0.25, 1, 1.0, np.float32(0.5), np.float64(1.0)])
def test_valid_full_probabilities_are_accepted(value):
    assert masks.validated_full_probability(value) == pytest.approx(float(value))


def test_the_default_full_probability_is_one_half():
    assert masks.DEFAULT_FULL_PROBABILITY == 0.5
    rngs = generators(31)
    modes = Counter(draw(rngs=rngs).mode for _ in range(4000))
    assert 0.45 < modes[masks.MODE_FULL] / 4000 < 0.55


def test_probability_one_always_masks_everything_and_zero_never_does():
    rngs = generators(3)
    for _ in range(200):
        everything = draw(rngs=rngs, full_probability=1.0)
        assert everything.mode == masks.MODE_FULL and everything.sites == REGION
        partial = draw(rngs=rngs, full_probability=0.0)
        assert partial.mode == masks.MODE_PARTIAL and 1 <= partial.count <= len(REGION) - 1


# --- partial draws ---------------------------------------------------------

def test_partial_counts_are_uniform_on_one_to_n_minus_one():
    rngs = generators(8)
    counts = Counter(draw(rngs=rngs, full_probability=0.0).count for _ in range(5000))
    assert set(counts) == {1, 2, 3, 4, 5}
    assert all(800 < counts[count] < 1200 for count in range(1, 6))


@pytest.mark.parametrize("geometry", masks.GEOMETRIES)
def test_a_partial_draw_hides_something_but_never_the_whole_region(geometry):
    rngs = generators(12)
    for _ in range(500):
        result = draw(rngs=rngs, geometry=geometry, full_probability=0.0)
        assert 0 < len(result.sites) < len(REGION)
        assert len(result.sites) == result.count and len(set(result.sites)) == result.count
        assert set(result.sites) < set(REGION)
        assert list(result.sites) == sorted(result.sites)


@pytest.mark.parametrize("geometry", masks.GEOMETRIES)
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_a_one_coordinate_region_is_always_masked_fully(geometry, probability):
    for seed in range(25):
        result = draw((("H", 3),), seed=seed, geometry=geometry, full_probability=probability)
        assert (result.mode, result.count, result.sites) == (masks.MODE_FULL, 1, (("H", 3),))


# --- geometry ----------------------------------------------------------------

def test_the_suffix_geometry_hides_the_last_k_canonical_coordinates():
    shuffled = list(REGION)
    random.Random(1).shuffle(shuffled)
    rngs = generators(4)
    for _ in range(200):
        result = draw(tuple(shuffled), rngs=rngs, geometry=masks.GEOMETRY_SUFFIX,
                      full_probability=0.0)
        assert result.sites == REGION[len(REGION) - result.count:]


def test_the_subset_geometry_reaches_every_subset_of_its_size_roughly_uniformly():
    rngs = generators(21)
    seen = Counter()
    for _ in range(10000):
        result = draw(rngs=rngs, geometry=masks.GEOMETRY_SUBSET, full_probability=0.0)
        if result.count == 3:
            seen[result.sites] += 1
    assert set(seen) == set(itertools.combinations(REGION, 3))
    assert all(50 < number < 170 for number in seen.values())
    scattered = [sites for sites in seen
                 if REGION.index(sites[-1]) - REGION.index(sites[0]) + 1 > len(sites)]
    assert scattered, "no non-contiguous subset was ever drawn"


def test_a_full_draw_is_the_whole_region_in_both_geometries():
    for geometry in masks.GEOMETRIES:
        result = draw(geometry=geometry, full_probability=1.0)
        assert result.sites == REGION and result.count == result.size == len(REGION)


def test_light_chain_coordinates_sort_after_heavy_ones():
    region = (("L", 4), ("H", 30), ("L", 1), ("H", 2))
    result = draw(region, full_probability=1.0)
    assert result.sites == (("H", 2), ("H", 30), ("L", 1), ("L", 4))


def test_the_order_a_region_is_listed_in_changes_nothing():
    shuffled = list(REGION)
    random.Random(2).shuffle(shuffled)
    for geometry in masks.GEOMETRIES:
        left, right = generators(6), generators(6)
        for _ in range(60):
            assert draw(REGION, rngs=left, geometry=geometry).sites \
                == draw(tuple(shuffled), rngs=right, geometry=geometry).sites


# --- the two generators --------------------------------------------------------

def run_arm(geometry, steps=300):
    """Many rows of different region sizes through one pair of generators."""
    regions = [REGION, REGION[:4], REGION[:1], REGION[:3],
               (("H", 2), ("L", 5), ("L", 7)), REGION[:2]]
    rngs = generators(11)
    log, sites = [], []
    for step in range(steps):
        result = draw(regions[step % len(regions)], rngs=rngs, geometry=geometry)
        log.append((result.mode, result.count, result.size))
        sites.append(result.sites)
    return log, sites, states(rngs), [float(rng.random()) for rng in rngs]


def test_the_geometries_draw_identical_modes_counts_and_generator_consumption():
    suffix_log, suffix_sites, suffix_states, suffix_future = run_arm(masks.GEOMETRY_SUFFIX)
    subset_log, subset_sites, subset_states, subset_future = run_arm(masks.GEOMETRY_SUBSET)
    assert suffix_log == subset_log
    assert suffix_states == subset_states
    assert suffix_future == subset_future
    assert suffix_sites != subset_sites, "the two geometries chose identical sites everywhere"


def test_consumption_does_not_depend_on_the_mode_drawn():
    """A run that only ever draws full masks leaves the generators where a mixed run does."""
    mixed, always_full = generators(17), generators(17)
    for _ in range(100):
        draw(rngs=mixed, full_probability=0.5)
        draw(rngs=always_full, full_probability=1.0)
    assert states(mixed) == states(always_full)


def test_the_two_generators_must_be_separate():
    shared = np.random.default_rng(1)
    with pytest.raises(ValueError, match="separate"):
        draw(rngs=(shared, shared))
    bit_generator = np.random.PCG64(2)
    with pytest.raises(ValueError, match="separate"):
        draw(rngs=(np.random.Generator(bit_generator), np.random.Generator(bit_generator)))
    with pytest.raises(ValueError, match="numpy Generator"):
        draw(rngs=(random.Random(1), np.random.default_rng(1)))


def test_a_refused_call_consumes_nothing():
    rngs = generators(5)
    before = states(rngs)
    with pytest.raises(ValueError):
        draw(rngs=rngs, full_probability=float("nan"))
    with pytest.raises(ValueError):
        draw((("H", 99),), rngs=rngs)
    with pytest.raises(ValueError):
        draw(rngs=rngs, geometry="prefix")
    assert states(rngs) == before


# --- logging and labels ------------------------------------------------------------

def test_a_draw_logs_its_mode_and_count():
    full = draw(full_probability=1.0).document()
    assert full == {"mode": "full", "count": 6, "region_size": 6, "geometry": "suffix"}
    partial = draw(full_probability=0.0, geometry="subset").document()
    assert partial["mode"] == "partial" and 1 <= partial["count"] <= 5
    assert partial["region_size"] == 6 and partial["geometry"] == "subset"


def test_no_label_or_measurement_can_reach_the_draw():
    assert set(inspect.signature(masks.draw_mask).parameters) == {
        "region", "heavy_length", "light_length", "count_rng", "position_rng", "geometry",
        "full_probability"}
