"""Measured endpoint pairs: the lattice, refusals, determinism, digests, draws and staleness.

Every synthetic construct is a full VH+VL sequence. The workhorse is a binary lattice: three
region coordinates that each take one of two residues give eight constructs, and every pair of
them is at distance 1, 2 or 3 -- 12, 12 and 4 unordered pairs -- which the tests count
independently of the code that builds the pool.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments.her2_multiscale import pairs as pairs_lib
from smallAntibodyGen.experiments.her2_routes_sources import CENSOR_LEFT, CENSOR_RIGHT, SCALE_PKD
from smallAntibodyGen.experiments.her2_routes_split import ExclusionClosure
from smallAntibodyGen.models.her2_routes_heads import KIND_INTERVAL, KIND_ORDINAL, KIND_POINT

HEAVY = "ACDEFGHIKLMNPQRSTVWY" * 2
LIGHT = "YWVTSRQPNMLKIHGFEDCA"
REGION = (("H", 3), ("H", 4), ("H", 5))
REGION4 = REGION + (("H", 6),)
OTHER_HEAVY = HEAVY[:20] + "W" + HEAVY[21:]  # differs from HEAVY outside every region here
OPEN = ExclusionClosure.build()


def vertex(bits, *, region=REGION, background=HEAVY, light=LIGHT):
    heavy = list(background)
    for (_, index), bit in zip(region, bits):
        heavy[index] = "AC"[bit]
    return "".join(heavy), light


def with_residue(heavy, index, residue):
    return heavy[:index] + residue + heavy[index + 1:]


def make(heavy, light=LIGHT, *, value=7.0, region=REGION, partition="train", study="study",
         family="affinity", assay="assay", target="target", parent="parent", scale=SCALE_PKD,
         kind=KIND_POINT, censoring="none", lower=None, upper=None, prefix="m",
         measurements=None):
    record = hashlib.sha256(f"{heavy}|{light}".encode()).hexdigest()[:8]
    construct = serial.Construct(heavy=heavy, light=light, sites=REGION, source="synthetic",
                                 record_id=record)
    if measurements is None:
        measurements = (data_lib.ReplicateMeasurement(
            value=value, lower=value if lower is None else lower,
            upper=value if upper is None else upper, censoring=censoring, replicate=1,
            measurement_id=f"{prefix}:{record}"),)
    return data_lib.ConstructSummary(
        construct=construct, study=study, assay=assay, target=target, parent=parent,
        family=family, kind=kind, measurements=measurements, region=tuple(region),
        mutated_sites=(), partition=partition, scale=scale, source="synthetic")


def lattice(*, region=REGION, background=HEAVY, light=LIGHT, base=6.0, **options):
    summaries = []
    for number, bits in enumerate(itertools.product((0, 1), repeat=len(region))):
        heavy, light_chain = vertex(bits, region=region, background=background, light=light)
        summaries.append(make(heavy, light_chain, value=base + 0.1 * number, region=region,
                              **options))
    return summaries


def no_window(construct):
    """``window_of`` for constructs outside the closure's edit space: nothing to test."""
    return None


def scoped(construct):
    """``window_of`` for constructs whose closure window is the lattice region."""
    return construct.residues(REGION)


def keys_of(summaries):
    return frozenset(summary.population_key for summary in summaries
                     if isinstance(summary, data_lib.ConstructSummary))


def build(summaries, **options):
    options.setdefault("closure", OPEN)
    options.setdefault("window_of", no_window)
    options.setdefault("training_population_keys", keys_of(summaries))
    return pairs_lib.build_pair_pool(summaries, **options)


def current(pair, **options):
    options.setdefault("closure", OPEN)
    options.setdefault("window_of", no_window)
    options.setdefault("training_population_keys", frozenset(pair.population_keys))
    return pairs_lib.require_pair_current(pair, **options)


def hamming(left, right):
    return sum(one != other for one, other in zip(left, right))


def by_distance(pool):
    return Counter(pair.distance for pair in pool.pairs)


# --- the lattice -----------------------------------------------------------------

def test_three_binary_sites_give_the_expected_unordered_pairs():
    pool = build(lattice())
    assert by_distance(pool) == {1: 12, 2: 12, 3: 4}
    assert pool.counts["qualified_seen"] == {"1": 12, "2": 12, "3": 4}
    assert pool.counts["retained"] == {"1": 12, "2": 12, "3": 4}
    assert len({frozenset(pair.identities) for pair in pool.pairs}) == 28
    for pair in pool.pairs:
        left, right = pair.first.construct, pair.second.construct
        assert pair.distance == hamming(left.heavy, right.heavy) + hamming(left.light, right.light)
        assert set(pair.sites) == {("H", index) for index, (one, other)
                                   in enumerate(zip(left.heavy, right.heavy)) if one != other}


def test_four_binary_sites_count_only_distances_up_to_three():
    pool = build(lattice(region=REGION4))
    assert pool.counts["qualified_seen"] == {"1": 32, "2": 48, "3": 32}  # the 8 antipodes are d=4


def test_different_full_chain_backgrounds_never_cross():
    other_light = "A" + LIGHT[1:]
    pool = build(lattice() + lattice(background=OTHER_HEAVY) + lattice(light=other_light))
    assert by_distance(pool) == {1: 36, 2: 36, 3: 12}
    for pair in pool.pairs:
        assert pair.first.construct.heavy[20] == pair.second.construct.heavy[20]
        assert pair.first.construct.light == pair.second.construct.light
    # 24 constructs, 3 buckets of 8: the pairs the background rule skipped are counted.
    assert pool.counts["cross_background_pairs"] == 24 * 23 // 2 - 3 * 28


@pytest.mark.parametrize("name", ["study", "assay", "target"])
def test_populations_never_cross(name):
    """The same molecules in another study, assay or target are other populations, not conflicts."""
    pool = build(lattice() + lattice(**{name: "other"}))
    assert len(pool.pairs) == 56
    assert pool.counts["eligible_endpoints"] == 16 and pool.counts["duplicates"] == 0
    for pair in pool.pairs:
        assert getattr(pair.first, name) == getattr(pair.second, name)


@pytest.mark.parametrize("name, other", [("parent", "other"), ("family", "display"),
                                         ("scale", "enrichment ratio")])
def test_one_population_with_two_parents_families_or_scales_is_refused(name, other):
    """Parent, family and scale are properties of an observed population, not new groups."""
    with pytest.raises(ValueError, match="Conflicting evidence"):
        build(lattice() + lattice(**{name: other}))


def test_a_different_declared_region_is_a_different_group():
    pool = build(lattice() + lattice(region=REGION4))
    assert len(pool.pairs) == 28 + 120 - 8  # the 4-site lattice drops its 8 antipodal pairs
    for pair in pool.pairs:
        assert pair.first.region == pair.second.region


def test_the_order_a_region_is_listed_in_does_not_change_the_group():
    reference = build(lattice())
    relisted = [replace(summary, region=tuple(reversed(REGION))) for summary in lattice()]
    pool = build(relisted)
    assert pool.structural_digest == reference.structural_digest
    assert pool.supervision_digest == reference.supervision_digest
    assert all(pair.region == REGION for pair in pool.pairs)
    # the same constructs listed both ways are one group, so they deduplicate
    assert build(lattice() + relisted).counts["duplicates"] == 8


# --- malformed and conflicting input ------------------------------------------------

@pytest.mark.parametrize("region", [(("H", 40),), (("X", 1),), (("H", True),), (("H", -1),),
                                    (("H", 3), ("H", 3)), (("H", 3, 1),), (("L", 20),)])
@pytest.mark.parametrize("partition", ["train", "development"])
def test_a_malformed_region_raises_even_on_a_row_that_would_be_filtered(region, partition):
    bad = make(with_residue(HEAVY, 20, "W"), region=region, partition=partition)
    with pytest.raises(ValueError):
        build(lattice() + [bad])


def test_a_summary_with_no_region_is_counted_not_paired():
    pool = build(lattice() + [make(with_residue(HEAVY, 20, "W"), region=())])
    assert pool.counts["refused"] == {"no_region": 1}
    assert len(pool.pairs) == 28


def test_identical_evidence_is_deduplicated_and_counted():
    summaries = lattice()
    pool = build(summaries + [replace(summaries[0])])
    reference = build(summaries)
    assert pool.counts["duplicates"] == 1 and pool.counts["summaries"] == 9
    assert pool.structural_digest == reference.structural_digest
    assert pool.supervision_digest == reference.supervision_digest


def with_value(summary, value):
    return replace(summary, measurements=(replace(summary.measurements[0], value=value,
                                                  lower=value, upper=value),))


CONFLICTS = {
    "value": lambda s: with_value(s, s.exact_value + 1.0),
    "partition": lambda s: replace(s, partition="development"),
    "measurement_id": lambda s: replace(
        s, measurements=(replace(s.measurements[0], measurement_id="renamed"),)),
    "record_id": lambda s: replace(s, construct=replace(s.construct, record_id="renamed")),
    "kind": lambda s: replace(s, kind=KIND_INTERVAL),
    "source": lambda s: replace(s, source="elsewhere"),
}


@pytest.mark.parametrize("name", sorted(CONFLICTS))
def test_conflicting_evidence_for_one_construct_is_refused_before_filtering(name):
    summaries = lattice()
    with pytest.raises(ValueError, match="Conflicting evidence"):
        build(summaries + [CONFLICTS[name](summaries[0])])


@pytest.mark.parametrize("change, fields", [
    ({"partition": "development"}, "['partition']"),
    ({"region": REGION[:2]}, "['region']"),
    ({"partition": "development", "region": REGION[:2]}, "['partition', 'region']")])
def test_one_observed_population_with_a_different_partition_or_region_is_a_conflict(change, fields):
    """A twin that differs in region and partition used to be filtered silently as its own group."""
    summaries = lattice()
    twin = replace(summaries[0], **change)
    assert twin.population_key == summaries[0].population_key
    with pytest.raises(ValueError, match="Conflicting evidence") as raised:
        build(summaries + [twin])
    assert fields in str(raised.value)
    with pytest.raises(ValueError, match="Conflicting evidence"):
        build([twin] + summaries)


def test_the_input_must_be_construct_summaries():
    with pytest.raises(ValueError, match="ConstructSummary"):
        build(lattice() + [{"heavy": HEAVY}])


# --- the training boundary --------------------------------------------------------------

@pytest.mark.parametrize("partition", ["development", "protected", "TRAIN", ""])
def test_only_the_declared_train_partition_is_used(partition):
    summaries = lattice()
    summaries[0] = replace(summaries[0], partition=partition)
    pool = build(summaries)
    assert pool.counts["refused"] == {"not_train_partition": 1}
    assert by_distance(pool) == {1: 9, 2: 9, 3: 3}  # one cube vertex removed
    assert summaries[0].identity not in {item for pair in pool.pairs for item in pair.identities}


def test_the_closure_refuses_identities_and_windows():
    summaries = lattice()
    for closure in (ExclusionClosure.build(reserved_identities=[summaries[0].identity]),
                    ExclusionClosure.build(
                        reserved_windows=[summaries[0].construct.residues(REGION)])):
        pool = build(summaries, closure=closure, window_of=scoped)
        assert pool.counts["refused"] == {"closure_refused": 1}
        assert by_distance(pool) == {1: 9, 2: 9, 3: 3}


def test_the_identity_is_checked_even_when_the_construct_has_no_closure_window():
    summaries = lattice()
    closure = ExclusionClosure.build(reserved_identities=[summaries[0].identity])
    pool = build(summaries, closure=closure, window_of=no_window)
    assert pool.counts["refused"] == {"closure_refused": 1}


def test_region_text_that_spells_a_held_out_core_is_not_refused_without_a_closure_window():
    """An unrelated target whose region text equals a held-out HER2 core stays eligible.

    The radius rule is defined for the HER2 edit window only; the caller says which constructs live
    there, and a region string is never taken for a window.
    """
    summaries = lattice()
    held_out = summaries[0].construct.residues(REGION)
    closure = ExclusionClosure.build(reserved_windows=[held_out], radius=1)
    unrelated = build(summaries, closure=closure, window_of=no_window)
    assert unrelated.counts["refused"] == {} and by_distance(unrelated) == {1: 12, 2: 12, 3: 4}
    # The same constructs inside the closure's edit space are refused, and so are their radius-1
    # neighbours: the held-out vertex and the three vertices one edit away from it. The four
    # vertices left (two or three C's) have no antipodal partner among themselves, so d=3 is gone.
    scoped_pool = build(summaries, closure=closure, window_of=scoped, distances=(1, 2))
    assert scoped_pool.counts["refused"] == {"closure_refused": 4}
    assert by_distance(scoped_pool) == {1: 3, 2: 3}


def test_a_window_callback_is_required_and_must_be_callable():
    with pytest.raises(TypeError):
        pairs_lib.build_pair_pool(lattice(), closure=OPEN,
                                  training_population_keys=keys_of(lattice()))
    with pytest.raises(ValueError, match="window_of"):
        build(lattice(), window_of=None)
    with pytest.raises(ValueError, match="window_of"):
        build(lattice(), window_of="HER2")


def test_a_train_labelled_row_outside_the_training_population_is_refused_at_creation():
    summaries = lattice()
    pool = build(summaries, training_population_keys=keys_of(summaries[1:]))
    assert pool.counts["refused"] == {"not_training_population": 1}
    assert by_distance(pool) == {1: 9, 2: 9, 3: 3}
    assert summaries[0].identity not in {item for pair in pool.pairs for item in pair.identities}
    assert pool.population_digest == pairs_lib.population_keys_digest(keys_of(summaries[1:]))
    assert pool.population_digest != pairs_lib.population_keys_digest(keys_of(summaries))


@pytest.mark.parametrize("keys", [
    None, (), [], set(), frozenset(), "abc", {("a", "b", "c")}, {("a", "b", "c", 1)},
    {("a", "b", "c", "d"), "not a tuple"}, iter([("a", "b", "c", "d")])])
def test_the_training_population_must_be_a_non_empty_set_of_four_string_tuples(keys):
    with pytest.raises(ValueError, match="training_population_keys|training population key"):
        build(lattice(), training_population_keys=keys)


def test_the_training_population_is_a_required_argument_with_no_default():
    with pytest.raises(TypeError):
        pairs_lib.build_pair_pool(lattice(), closure=OPEN, window_of=no_window)
    with pytest.raises(TypeError):
        pairs_lib.require_pair_current(fresh_pair(), closure=OPEN, window_of=no_window)


def test_equal_numeric_endpoints_are_a_valid_pair_with_delta_zero():
    pool = build([make(*vertex((0, 0, 0)), value=7.25), make(*vertex((0, 0, 1)), value=7.25)],
                 distances=(1,))
    assert len(pool.pairs) == 1 and pool.pairs[0].delta == 0.0


def test_exactness_is_read_from_the_replicates_not_from_kind():
    zero = vertex((0, 0, 0))[0]

    def weird(number, **options):
        return make(with_residue(zero, 3, "DEFGHIKL"[number]), **options)

    exact_replicate = data_lib.ReplicateMeasurement(
        value=7.0, lower=7.0, upper=7.0, censoring="none", replicate=1, measurement_id="a")
    left_replicate = data_lib.ReplicateMeasurement(
        value=6.0, lower=None, upper=6.0, censoring=CENSOR_LEFT, replicate=2, measurement_id="b")
    included = weird(0, kind=KIND_INTERVAL, value=7.5)  # a degenerate uncensored interval
    refused = [
        weird(1, kind=KIND_INTERVAL, value=7.0, lower=6.5, upper=7.5),
        weird(2, censoring=CENSOR_RIGHT),
        weird(3, kind=KIND_ORDINAL),
        weird(4, value=float("nan")),
        weird(5, scale="enrichment ratio"),
        weird(6, assay=""),
        weird(7, measurements=(exact_replicate, left_replicate)),
    ]
    pool = build(lattice() + [included] + refused)
    assert pool.counts["refused"] == {
        "nondegenerate_bounds": 1, "censored": 1, "ordinal": 1, "nonfinite": 1,
        "scale_not_pkd": 1, "missing_metadata": 1, "mixed_censoring": 1}
    with_interval = [pair for pair in pool.pairs if included.identity in pair.identities]
    assert with_interval
    for pair in with_interval:
        values = {pair.first.identity: pair.first_value, pair.second.identity: pair.second_value}
        assert values[included.identity] == 7.5


# --- determinism, the reservoir and the digests ----------------------------------------------

def test_the_pool_is_independent_of_input_order():
    summaries = lattice() + lattice(background=OTHER_HEAVY)
    reference = build(summaries, capacity=3)
    for seed in range(5):
        shuffled = list(summaries)
        random.Random(seed).shuffle(shuffled)
        pool = build(shuffled, capacity=3)
        assert pool.structural_digest == reference.structural_digest
        assert pool.supervision_digest == reference.supervision_digest
        assert [pair.structural_digest for pair in pool.pairs] \
            == [pair.structural_digest for pair in reference.pairs]
        assert json.dumps(pool.document(), sort_keys=True) \
            == json.dumps(reference.document(), sort_keys=True)


def test_a_smaller_capacity_keeps_a_subset_and_counts_every_qualified_pair():
    summaries = lattice()
    big, middle, small = build(summaries), build(summaries, capacity=4), \
        build(summaries, capacity=2)
    for pool in (big, middle, small):
        assert pool.counts["qualified_seen"] == {"1": 12, "2": 12, "3": 4}
    assert middle.counts["retained"] == {"1": 4, "2": 4, "3": 4}
    assert small.counts["retained"] == {"1": 2, "2": 2, "3": 2}

    def retained(pool):
        return {pair.structural_digest for pair in pool.pairs}
    assert retained(small) < retained(middle) < retained(big)
    assert retained(build(summaries, capacity=2, seed=1)) != retained(small)


def test_a_label_only_change_moves_the_supervision_digest_not_the_structural_one():
    reference = build(lattice(), capacity=5)
    changes = {"value": lattice(base=9.0), "provenance": lattice(prefix="other"),
               "kind": lattice(kind=KIND_INTERVAL)}
    for name, changed in changes.items():
        pool = build(changed, capacity=5)
        assert pool.structural_digest == reference.structural_digest, name
        assert [pair.structural_digest for pair in pool.pairs] \
            == [pair.structural_digest for pair in reference.pairs], name
        assert pool.supervision_digest != reference.supervision_digest, name
        assert all(new.supervision_digest != old.supervision_digest
                   for new, old in zip(pool.pairs, reference.pairs)), name


def test_a_pair_references_its_summaries_and_copies_no_sequences():
    summaries = lattice()
    pool = build(summaries)
    known = {id(summary) for summary in summaries}
    for pair in pool.pairs:
        assert id(pair.first) in known and id(pair.second) in known
        for field in dataclasses.fields(pair):
            value = getattr(pair, field.name)
            if isinstance(value, str):
                assert len(value) <= 64, field.name


# --- configuration -----------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    {"distances": ()}, {"distances": (0,)}, {"distances": (4,)}, {"distances": (1, 1)},
    {"distances": (True,)}, {"distances": (1.0,)}, {"distances": ("1",)}, {"distances": 3},
    {"distances": {1}}, {"capacity": 0}, {"capacity": -1}, {"capacity": True},
    {"capacity": 2.0}, {"capacity": "8"}, {"capacity": None}, {"seed": True}, {"seed": 1.5},
    {"seed": -1}])
def test_invalid_configuration_is_refused(options):
    with pytest.raises(ValueError):
        build(lattice(), **options)


def test_a_closure_is_required():
    with pytest.raises(ValueError, match="ExclusionClosure"):
        pairs_lib.build_pair_pool(lattice(), closure=None, window_of=no_window,
                                  training_population_keys=keys_of(lattice()))


def test_a_configured_distance_with_no_pair_fails_closed():
    two = [make(*vertex((0, 0, 0))), make(*vertex((0, 0, 1)))]
    with pytest.raises(ValueError, match="distance 2"):
        build(two, distances=(1, 2))
    assert len(build(two, distances=(1,)).pairs) == 1


# --- anchor-uniform draws and the reported exposure ---------------------------------------------

def star(*, extra=()):
    """A hub with three leaves at distance 1 from it and distance 2 from each other."""
    hub = make(*vertex((0, 0, 0)), value=7.0)
    leaves = [make(*vertex(bits), value=7.0 + 0.1 * (number + 1))
              for number, bits in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)])]
    return hub, leaves, build([hub] + leaves + list(extra), distances=(1,))


def test_draws_are_anchor_uniform_not_pair_uniform():
    hub, leaves, pool = star()
    rng = np.random.default_rng(3)
    drawn = [pool.draw(1, rng) for _ in range(4000)]
    anchors = Counter(pair.first.identity for pair in drawn)
    assert 850 < anchors[hub.identity] < 1150  # a quarter, not the 1/2 a pair-uniform draw gives
    assert all(750 < anchors[leaf.identity] < 1250 for leaf in leaves)
    partners = Counter(pair.second.identity for pair in drawn
                       if pair.first.identity == hub.identity)
    assert set(partners) == {leaf.identity for leaf in leaves}
    assert all(250 < number < 420 for number in partners.values())
    assert all(pair.distance == 1 for pair in drawn)


def test_the_same_seed_draws_the_same_pairs():
    _, _, pool = star()

    def sequence(seed):
        rng = np.random.default_rng(seed)
        return [pool.draw(1, rng).structural_digest for _ in range(20)]
    assert sequence(9) == sequence(9)


def test_the_report_states_degree_and_exposure_as_they_are():
    isolated = make(*vertex((1, 1, 1)), value=8.0)  # distance 2 and 3 from everything else
    _, _, pool = star(extra=[isolated])
    report = pool.report()["1"]
    assert report["anchors"] == 4 and report["retained_pairs"] == 3
    assert report["qualified_seen"] == 3
    assert report["endpoints_without_a_retained_pair"] == 1
    assert report["degree"] == {"min": 1, "median": 1.0, "max": 3}
    assert report["exposure"]["max"] == pytest.approx(1.0)  # the hub: 1/4 as anchor + 3 * 1/4
    assert report["exposure"]["min"] == pytest.approx(1 / 3)  # a leaf: 1/4 + 1/4 * 1/3
    assert report["exposure"]["mean"] == pytest.approx(0.5)  # two endpoints per draw over four


def two_assays():
    """Two sequences at distance 1, each measured in two assays: four endpoint populations."""
    summaries = [make(*vertex(bits), assay=assay, value=7.0 + 0.1 * number)
                 for number, (assay, bits) in enumerate(
                     (assay, bits) for assay in ("a1", "a2") for bits in ((0, 0, 0), (0, 0, 1)))]
    return summaries, build(summaries, distances=(1,))


def test_a_molecule_measured_in_two_assays_is_two_endpoints_everywhere():
    """Anchors, the denominator and the coverage all count populations, not molecules.

    Anchor tables keyed by molecule once reported 2 endpoints 'without a pair' here, while the four
    populations are each an endpoint of a retained pair.
    """
    summaries, pool = two_assays()
    assert pool.counts["eligible_endpoints"] == 4 and len(pool.pairs) == 2
    report = pool.report()["1"]
    assert report["unit"] == "population (identity, study, assay, target)"
    assert report["anchors"] == 4 and report["population_endpoints"] == 4
    assert report["coverage"] == 1.0 and report["endpoints_without_a_retained_pair"] == 0
    assert report["retained_pairs"] == 2 and report["degree"] == {"min": 1, "median": 1.0, "max": 1}
    assert report["exposure"] == {"min": pytest.approx(0.5), "mean": pytest.approx(0.5),
                                  "max": pytest.approx(0.5)}
    rng = np.random.default_rng(11)
    drawn = [pool.draw(1, rng) for _ in range(800)]
    anchors = Counter(pair.first.population_key for pair in drawn)
    assert set(anchors) == {summary.population_key for summary in summaries}
    assert all(120 < number < 280 for number in anchors.values())
    for pair in drawn:  # oriented by the anchor's identity; the partner is in the same assay
        assert pair.first.assay == pair.second.assay
        assert pair.first.identity != pair.second.identity


def test_coverage_counts_populations_without_a_retained_pair_at_that_distance():
    summaries, _ = two_assays()
    isolated = make(*vertex((1, 1, 1)), assay="a1", value=9.0)  # distance 2 and 3 from the rest
    pool = build(summaries + [isolated], distances=(1,))
    report = pool.report()["1"]
    assert report["population_endpoints"] == 5 and report["anchors"] == 4
    assert report["endpoints_without_a_retained_pair"] == 1
    assert report["coverage"] == pytest.approx(0.8)


@pytest.mark.parametrize("distance", [2, 0, 4, True, 1.0, "1"])
def test_a_draw_at_an_unconfigured_or_invalid_distance_is_refused(distance):
    _, _, pool = star()
    with pytest.raises(ValueError):
        pool.draw(distance, np.random.default_rng(0))


def test_orientation_swaps_the_pair_without_changing_its_identity():
    _, _, pool = star()
    pair = pool.pairs[0]
    flipped = pair.oriented(pair.second.identity)
    assert flipped.first is pair.second and flipped.second is pair.first
    assert flipped.delta == -pair.delta
    assert flipped.structural_digest == pair.structural_digest
    assert flipped.supervision_digest == pair.supervision_digest
    assert current(flipped) is True
    with pytest.raises(ValueError, match="anchor"):
        pair.oriented("not an endpoint")


# --- pair_between: the full-chain ground truth -------------------------------------------------------

def test_pair_between_is_orientation_independent_in_its_digests():
    a, b = make(*vertex((0, 0, 0)), value=7.0), make(*vertex((1, 0, 1)), value=7.6)
    forward, backward = pairs_lib.pair_between(a, b), pairs_lib.pair_between(b, a)
    assert forward.structural_digest == backward.structural_digest
    assert forward.supervision_digest == backward.supervision_digest
    assert forward.delta == pytest.approx(0.6) and backward.delta == pytest.approx(-0.6)
    assert forward.distance == 2 and forward.sites == (("H", 3), ("H", 5))


def test_pair_between_refuses_what_is_not_a_pair():
    a = make(*vertex((0, 0, 0)))
    near = vertex((0, 0, 1))
    four = (make(*vertex((0, 0, 0, 0), region=REGION4), region=REGION4),
            make(*vertex((1, 1, 1, 1), region=REGION4), region=REGION4))
    cases = [
        (make(with_residue(near[0], 20, "W"), near[1]), "outside the declared common region"),
        (make(near[0], "A" + LIGHT[1:]), "outside the declared common region"),
        (make(near[0] + "A", near[1]), "different length"),
        (make(*near, assay="other"), "do not share"),
        (make(*near, region=REGION[:2]), "different common regions"),
        (make(*near, censoring="right"), "not an exact finite pKD"),
        (make(*near, region=()), "declares no common region"),
        (a, "same construct"),
    ]
    for other, message in cases:
        with pytest.raises(ValueError, match=message):
            pairs_lib.pair_between(a, other)
    with pytest.raises(ValueError, match="distance 1..3"):
        pairs_lib.pair_between(*four)


# --- staleness -------------------------------------------------------------------------------------------

def fresh_pair():
    return next(pair for pair in build(lattice()).pairs if pair.distance == 2)


def set_value(summary, value):
    summary.measurements = (replace(summary.measurements[0], value=value, lower=value,
                                    upper=value),)


def test_a_current_pair_is_accepted():
    assert current(fresh_pair()) is True


STALE = {
    "left the training partition": (
        lambda pair: setattr(pair.first, "partition", "development"),
        "no longer an eligible training endpoint"),
    "became censored": (
        lambda pair: setattr(pair.second, "measurements",
                             (replace(pair.second.measurements[0], censoring="right"),)),
        "no longer an eligible training endpoint"),
    "changed value": (
        lambda pair: set_value(pair.second, pair.second_value + 0.3), "numeric value changed"),
    "changed provenance": (
        lambda pair: setattr(pair.first, "measurements",
                             (replace(pair.first.measurements[0], measurement_id="renamed"),)),
        "measurements, metadata or provenance changed"),
    "region narrowed": (
        lambda pair: [setattr(side, "region", REGION[:1]) for side in (pair.first, pair.second)],
        "stale"),
}

#: Mutations that move a population key (study, assay, target or the sequence itself). Against the
#: keys captured BEFORE the change they leave the training population; against keys re-derived
#: AFTER it (an authority that re-blessed the moved population) the recomputed pair still catches them.
MOVES_THE_POPULATION = {
    "both endpoints changed study": (
        lambda pair: [setattr(side, "study", "elsewhere") for side in (pair.first, pair.second)],
        "support or identity changed"),
    "one endpoint changed assay": (
        lambda pair: setattr(pair.first, "assay", "elsewhere"), "do not share"),
    "sequence changed outside the region": (
        lambda pair: setattr(pair.first, "construct", replace(
            pair.first.construct, heavy=with_residue(pair.first.construct.heavy, 20, "W"))),
        "outside the declared common region"),
}


@pytest.mark.parametrize("name", sorted(STALE))
def test_a_pair_whose_summaries_changed_is_refused_at_consumption(name):
    pair = fresh_pair()
    mutate, message = STALE[name]
    mutate(pair)
    with pytest.raises(ValueError, match=message):
        current(pair)


@pytest.mark.parametrize("name", sorted(MOVES_THE_POPULATION))
def test_a_pair_whose_population_moved_leaves_the_training_population(name):
    pair = fresh_pair()
    keys = frozenset(pair.population_keys)  # the authority, captured before anything changed
    MOVES_THE_POPULATION[name][0](pair)
    with pytest.raises(ValueError, match="no longer an eligible training endpoint"):
        current(pair, training_population_keys=keys)


@pytest.mark.parametrize("name", sorted(MOVES_THE_POPULATION))
def test_the_recomputed_pair_catches_a_moved_population_even_if_the_authority_follows(name):
    pair = fresh_pair()
    mutate, message = MOVES_THE_POPULATION[name]
    mutate(pair)
    with pytest.raises(ValueError, match=message):
        current(pair, training_population_keys=frozenset(pair.population_keys))


@pytest.mark.parametrize("side", ["first", "second"])
def test_a_pair_is_refused_when_an_endpoint_is_not_in_the_training_population(side):
    pair = fresh_pair()
    keys = frozenset(pair.population_keys) - {getattr(pair, side).population_key}
    with pytest.raises(ValueError, match=r"no longer an eligible training endpoint "
                                         r"\(not_training_population\)"):
        current(pair, training_population_keys=keys)


def test_a_train_labelled_endpoint_the_authority_does_not_know_cannot_pass():
    """The partition flag says train, the closure is open, the values are exact -- and still no."""
    pair = fresh_pair()
    assert pair.first.partition == pair.second.partition == "train"
    with pytest.raises(ValueError, match="not_training_population"):
        current(pair, training_population_keys=frozenset({("x", "y", "z", "w")}))


@pytest.mark.parametrize("side", ["first", "second"])
def test_a_pair_is_refused_against_a_closure_that_now_reserves_an_endpoint(side):
    pair = fresh_pair()
    closure = ExclusionClosure.build(reserved_identities=[getattr(pair, side).identity])
    with pytest.raises(ValueError, match="no longer an eligible training endpoint"):
        current(pair, closure=closure)


def test_a_pair_is_refused_against_a_window_the_closure_now_reserves():
    pair = fresh_pair()
    closure = ExclusionClosure.build(reserved_windows=[pair.first.construct.residues(REGION)])
    current(pair, closure=closure, window_of=no_window)          # no closure window: not tested
    with pytest.raises(ValueError, match="closure_refused"):
        current(pair, closure=closure, window_of=scoped)


def test_a_hand_altered_pair_record_is_refused():
    pair = fresh_pair()
    for altered, message in [
            (replace(pair, first_value=pair.first_value + 1.0), "numeric value changed"),
            (replace(pair, supervision_digest="0" * 64), "measurements, metadata or provenance"),
            (replace(pair, structural_digest="0" * 64), "support or identity"),
            (replace(pair, sites=pair.sites[:1]), "support or identity")]:
        with pytest.raises(ValueError, match=message):
            current(altered)
    with pytest.raises(ValueError, match="MeasuredPair"):
        pairs_lib.require_pair_current({"distance": 2}, closure=OPEN, window_of=no_window,
                                       training_population_keys=frozenset({("a", "b", "c", "d")}))
