"""How the arms change a base draw: masks on whole-window draws, pairs in place of pair-only local draws.

The properties that make the four arms comparable are pinned here with no model: two mask arms hide
the same number of sites of the same rows under the same conditions and differ only in which sites; a
replacement decision is the same in every arm; and nothing here reads or advances a generator it does
not own.
"""
from __future__ import annotations

import types
from collections import Counter

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments.her2_multiscale import draws as draws_lib
from smallAntibodyGen.experiments.her2_routes_conditions import BIN_HIGH, BIN_LOW, BIN_NULL, Condition
from smallAntibodyGen.experiments.her2_routes_tasks import TaskDraw
from smallAntibodyGen.tests import her2_multiscale_world as world_lib
from smallAntibodyGen.tests.test_her2_multiscale_pairs import OPEN, build, keys_of, lattice, no_window

SITES = serial.her2_core_edit_map()
SCAFFOLD = world_lib.fake_scaffold()
CONDITIONS = (Condition("affinity", BIN_HIGH, world_lib.ASSAY),
              Condition("affinity", BIN_LOW, world_lib.ASSAY),
              Condition("affinity", BIN_NULL, "null"))


def whole_draw(rows=32, *, sites=SITES):
    cores = world_lib.lattice_cores()
    views = []
    for index in range(rows):
        construct = serial.Construct(heavy=SCAFFOLD.with_core(cores[index % len(cores)]),
                                     light=SCAFFOLD.light, sites=tuple(sites), source="synthetic",
                                     record_id=f"r{index}")
        views.append((construct, serial.ablang_serialization(construct),
                      CONDITIONS[index % len(CONDITIONS)]))
    return TaskDraw(task=data_lib.TASK_WHOLE, family=data_lib.FAMILY_HER2, views=tuple(views),
                    summaries=("kept",), source="synthetic")


def hidden_sites(view):
    """The sites a view hides, read back from its tokens (not from the draw's own record)."""
    construct, serialization, _ = view
    positions = serial.ablang_positions(construct)
    masked = {site for site in construct.sites
              if serialization.tokens[positions[site]] == serial.ABLANG_MASK_ID}
    queried = {site for site in construct.sites if positions[site] in set(serialization.query_slots)}
    assert masked == queried, "the queried slots are exactly the masked ones"
    assert tuple(serialization.order) == tuple(sorted(queried)), "canonical declared order"
    return tuple(sorted(masked))


def replace(draw, geometry, *, seed=3, step=2, micro=1, probability=0.5):
    return draws_lib.replace_masks(draw, seed=seed, step=step, micro=micro, geometry=geometry,
                                   probability=probability)


# --- masks -------------------------------------------------------------------------------------

def test_a_replaced_draw_keeps_every_row_its_construct_its_condition_and_the_rest_of_the_draw():
    draw = whole_draw()
    original = [hidden_sites(view) for view in draw.views]
    assert all(len(sites) == len(SITES) for sites in original)          # the base: all hidden
    fresh, documents = replace(draw, "subset")
    assert len(fresh.views) == len(draw.views) == len(documents)
    for old, new in zip(draw.views, fresh.views):
        assert new[0] is old[0] and new[2] is old[2]
    assert (fresh.task, fresh.family, fresh.summaries, fresh.source) == (
        draw.task, draw.family, draw.summaries, draw.source)
    assert [hidden_sites(view) for view in draw.views] == original      # the base draw is untouched
    for view, document in zip(fresh.views, documents):
        assert len(hidden_sites(view)) == document["count"] <= len(SITES)
        assert document["region_size"] == len(SITES) and document["geometry"] == "subset"


def test_the_two_geometries_hide_the_same_number_of_sites_of_the_same_rows():
    draw = whole_draw()
    suffix, suffix_documents = replace(draw, "suffix")
    subset, subset_documents = replace(draw, "subset")
    assert [(d["mode"], d["count"]) for d in suffix_documents] \
        == [(d["mode"], d["count"]) for d in subset_documents]
    assert {d["mode"] for d in suffix_documents} == {"full", "partial"}
    differing = 0
    for left, right, document in zip(suffix.views, subset.views, suffix_documents):
        hidden_left, hidden_right = hidden_sites(left), hidden_sites(right)
        assert len(hidden_left) == len(hidden_right) == document["count"]
        if document["mode"] == "full":
            assert hidden_left == hidden_right == tuple(SITES)
        else:
            assert hidden_left == tuple(SITES[len(SITES) - document["count"]:])   # the last k
            differing += hidden_left != hidden_right
    assert differing > 0, "a uniform subset is not always the suffix"


def test_every_hidden_set_is_inside_the_chunk_the_base_draw_carried():
    chunk = SITES[2:7]
    draw = whole_draw(16, sites=chunk)
    for geometry in ("suffix", "subset"):
        fresh, documents = replace(draw, geometry, probability=0.0)
        for view, document in zip(fresh.views, documents):
            assert set(hidden_sites(view)) <= set(chunk)
            assert 1 <= document["count"] <= len(chunk) - 1 and document["region_size"] == 5


def test_the_decisions_are_a_function_of_seed_step_micro_and_row_only():
    draw = whole_draw()
    first, again = replace(draw, "subset"), replace(draw, "subset")
    assert first[1] == again[1]
    assert [hidden_sites(view) for view in first[0].views] \
        == [hidden_sites(view) for view in again[0].views]
    other_step, other_micro, other_seed = (replace(draw, "subset", step=3)[1],
                                           replace(draw, "subset", micro=0)[1],
                                           replace(draw, "subset", seed=4)[1])
    assert first[1] != other_step and first[1] != other_micro and first[1] != other_seed


def test_full_probability_one_hides_everything_and_zero_hides_part():
    draw = whole_draw()
    _, everything = replace(draw, "suffix", probability=1.0)
    assert {d["mode"] for d in everything} == {"full"} and {d["count"] for d in everything} == {10}
    _, part = replace(draw, "suffix", probability=0.0)
    assert {d["mode"] for d in part} == {"partial"}
    assert all(1 <= d["count"] <= 9 for d in part) and len({d["count"] for d in part}) > 3


def test_a_single_site_region_is_always_fully_hidden():
    draw = whole_draw(4, sites=SITES[:1])
    for geometry in ("suffix", "subset"):
        _, documents = replace(draw, geometry, probability=0.0)
        assert {(d["mode"], d["count"]) for d in documents} == {("full", 1)}


def test_replacing_masks_reads_and_advances_no_global_generator():
    draw = whole_draw(8)
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state()
    replace(draw, "subset")
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert torch.equal(torch_before, torch.get_rng_state())


def test_only_a_whole_window_draw_has_its_masks_replaced():
    draw = whole_draw(2)
    for task in (data_lib.TASK_LOCAL, data_lib.TASK_HEADS, data_lib.TASK_RECONSTRUCTION):
        other = TaskDraw(task=task, family=draw.family, views=draw.views)
        with pytest.raises(ValueError, match="whole-window draws only"):
            replace(other, "suffix")


def test_the_unknown_geometry_and_a_bad_probability_are_refused():
    draw = whole_draw(2)
    with pytest.raises(ValueError, match="geometry"):
        replace(draw, "prefix")
    for probability in (-0.1, 1.1, float("nan"), True, "0.5"):
        with pytest.raises(ValueError, match="full_probability"):
            replace(draw, "suffix", probability=probability)


def test_the_full_probability_is_one_minus_the_base_progressive_probability():
    assert draws_lib.full_probability({}) == 0.5
    assert draws_lib.full_probability({"masked_progressive_probability": 0.5}) == 0.5
    assert draws_lib.full_probability({"masked_progressive_probability": 0.3}) == pytest.approx(0.7)
    assert draws_lib.full_probability({"masked_progressive_probability": 0.0}) == 1.0
    with pytest.raises(ValueError, match="full_probability"):
        draws_lib.full_probability({"masked_progressive_probability": 1.5})


def test_the_mask_summary_counts_modes_and_the_histogram_of_k():
    documents = [{"mode": "full", "count": 10}, {"mode": "partial", "count": 3},
                 {"mode": "partial", "count": 3}, {"mode": "partial", "count": 7}]
    assert draws_lib.summarise_masks(documents) == {
        "views": 4, "modes": {"full": 1, "partial": 3}, "counts": {"10": 1, "3": 2, "7": 1}}


def test_the_mask_streams_are_keyed_apart_from_each_other_and_from_the_pair_streams():
    keys = [draws_lib.KEY_COUNT, draws_lib.KEY_POSITION, draws_lib.KEY_REPLACE,
            draws_lib.KEY_DISTANCE, draws_lib.KEY_PAIRS, draws_lib.KEY_OBJECTIVE]
    assert len(set(keys)) == len(keys)
    from smallAntibodyGen.experiments.her2_multiscale import evaluation as evaluation_lib
    assert not set(keys) & {evaluation_lib.KEY_SUBSET, evaluation_lib.KEY_FIT_PAIRS,
                            evaluation_lib.KEY_FIT_ORDER}


# --- distances ----------------------------------------------------------------------------------

class Scripted:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.mark.parametrize("draw, expected", [(0.0, 1), (0.49, 1), (0.5, 2), (0.79, 2), (0.81, 3),
                                            (0.999, 3), (0.9999999999999999, 3)])
def test_a_distance_is_chosen_from_the_cumulative_weights(draw, expected):
    weights = {1: 0.5, 2: 0.3, 3: 0.2}
    assert draws_lib.choose_distance(weights, Scripted(draw)) == expected


def test_the_distance_one_arms_always_choose_distance_one():
    rng = np.random.default_rng(0)
    assert {draws_lib.choose_distance({1: 1.0}, rng) for _ in range(50)} == {1}


# --- local replacement -----------------------------------------------------------------------------

def contrast(kind):
    return types.SimpleNamespace(kind=kind)


def local_draw(kinds=(data_lib.CONTRAST_SINGLE, data_lib.CONTRAST_ALTERNATIVES), *,
               family=data_lib.FAMILY_HER2, task=data_lib.TASK_LOCAL):
    return TaskDraw(task=task, family=family, views=tuple(range(2 * len(kinds))),
                    contrasts=tuple(contrast(kind) for kind in kinds))


def local_pairs(distance_weights=None, *, probability=0.5, seed=7, distances=(1, 2, 3)):
    summaries = lattice()
    pool = build(summaries, distances=distances)
    return draws_lib.LocalPairs(
        pool, seed=seed, probability=probability,
        distance_weights={1: 1.0} if distance_weights is None else distance_weights,
        closure=OPEN, window_of=no_window, training_keys=keys_of(summaries), kappa=2.0,
        reconstruction_weight=0.25)


def test_only_a_her2_local_draw_without_a_quartet_is_eligible():
    pairs = (data_lib.CONTRAST_SINGLE, data_lib.CONTRAST_ALTERNATIVES)
    assert draws_lib.LocalPairs.eligible(local_draw(pairs))
    assert draws_lib.LocalPairs.eligible(local_draw((data_lib.CONTRAST_SINGLE,)))
    assert not draws_lib.LocalPairs.eligible(local_draw((data_lib.CONTRAST_COUPLED,)))
    assert not draws_lib.LocalPairs.eligible(local_draw(
        (data_lib.CONTRAST_SINGLE, data_lib.CONTRAST_COUPLED)))
    assert not draws_lib.LocalPairs.eligible(local_draw(pairs, family=data_lib.FAMILY_OTHER))
    assert not draws_lib.LocalPairs.eligible(local_draw(pairs, family=data_lib.FAMILY_REPLAY))
    assert not draws_lib.LocalPairs.eligible(local_draw(pairs, task=data_lib.TASK_WHOLE))
    assert not draws_lib.LocalPairs.eligible(local_draw(()))


def test_a_quartet_or_other_family_draw_is_never_replaced():
    replacer = local_pairs(probability=1.0)
    assert not any(replacer.wants(local_draw((data_lib.CONTRAST_COUPLED,)), step=step, micro=0)
                   for step in range(1, 40))
    assert not any(replacer.wants(local_draw(family=data_lib.FAMILY_OTHER), step=step, micro=0)
                   for step in range(1, 40))
    assert all(replacer.wants(local_draw(), step=step, micro=0) for step in range(1, 40))


def test_the_replacement_probability_is_honoured_at_its_extremes_and_in_between():
    draw = local_draw()
    assert not any(local_pairs(probability=0.0).wants(draw, step=s, micro=m)
                   for s in range(1, 30) for m in range(4))
    decided = [local_pairs(probability=0.5).wants(draw, step=s, micro=m)
               for s in range(1, 101) for m in range(4)]
    assert 140 < sum(decided) < 260


def test_the_replacement_decision_does_not_depend_on_the_distances_an_arm_uses():
    draw = local_draw()
    narrow = local_pairs({1: 1.0})
    wide = local_pairs({1: 0.5, 2: 0.3, 3: 0.2})
    grid = [(s, m) for s in range(1, 60) for m in range(4)]
    assert [narrow.wants(draw, step=s, micro=m) for s, m in grid] \
        == [wide.wants(draw, step=s, micro=m) for s, m in grid]
    assert [local_pairs(seed=8).wants(draw, step=s, micro=m) for s, m in grid] \
        != [narrow.wants(draw, step=s, micro=m) for s, m in grid]


def test_a_replacement_presents_the_base_rows_as_pairs_at_one_distance():
    replacer = local_pairs({1: 0.5, 2: 0.3, 3: 0.2})
    seen = Counter()
    for step in range(1, 80):
        for rows in (2, 4, 6):
            draw = local_draw((data_lib.CONTRAST_SINGLE,) * (rows // 2))
            distance, pairs = replacer.sample(draw, step=step, micro=0)
            assert len(pairs) == rows // 2 and {pair.distance for pair in pairs} == {distance}
            seen[distance] += 1
    assert set(seen) == {1, 2, 3} and seen[1] > seen[3]


def test_a_replacement_is_a_function_of_seed_step_and_micro():
    draw = local_draw()
    replacer = local_pairs({1: 0.5, 2: 0.3, 3: 0.2})

    def digests(**place):
        distance, pairs = replacer.sample(draw, **place)
        return distance, [pair.structural_digest for pair in pairs]
    assert digests(step=5, micro=1) == digests(step=5, micro=1)
    assert digests(step=5, micro=1) != digests(step=6, micro=1) \
        or digests(step=5, micro=1) != digests(step=5, micro=0)


def test_the_distance_one_arms_draw_distance_one_pairs_only():
    replacer = local_pairs({1: 1.0})
    for step in range(1, 40):
        distance, pairs = replacer.sample(local_draw(), step=step, micro=0)
        assert distance == 1 and {pair.distance for pair in pairs} == {1}


def test_an_odd_or_empty_row_count_cannot_be_replaced():
    replacer = local_pairs()
    for views in (0, 1, 3):
        odd = TaskDraw(task=data_lib.TASK_LOCAL, family=data_lib.FAMILY_HER2,
                       views=tuple(range(views)), contrasts=(contrast(data_lib.CONTRAST_SINGLE),))
        with pytest.raises(ValueError, match="even row count"):
            replacer.sample(odd, step=1, micro=0)


def test_a_configured_distance_the_pool_lacks_is_a_preflight_error_not_a_renormalisation():
    with pytest.raises(ValueError, match="never renormalised|does not hold"):
        local_pairs({1: 0.5, 3: 0.5}, distances=(1, 2))
    with pytest.raises(ValueError, match="does not hold"):
        local_pairs({}, distances=(1, 2))
    with pytest.raises(ValueError, match="probability"):
        local_pairs(probability=1.5)


def test_the_pair_streams_use_their_own_keys_and_touch_no_global_generator():
    replacer = local_pairs({1: 0.5, 2: 0.3, 3: 0.2})
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state()
    for step in range(1, 10):
        replacer.wants(local_draw(), step=step, micro=0)
        replacer.sample(local_draw(), step=step, micro=0)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert torch.equal(torch_before, torch.get_rng_state())
