"""The resolved exposure stream, the replay order, the cadence and the seeds.

The failures these cover all look like success: a cycle boundary that silently
drops a partial batch, a cadence that is off by one from the declared one, a
"new" seed that lands inside the historical band, and a replay order that moved
when the task order did.
"""
from __future__ import annotations

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_guard as guard
from smallAntibodyGen.experiments import her2_preferences as preferences
from smallAntibodyGen.experiments import her2_replay_streams as streams


def population(chosen=10, rejected=6):
    """A small distance-matched population with the real dataclass and pairing."""
    generator = np.random.default_rng(20260920)
    chosen_index = generator.integers(0, 20, size=(chosen, 10)).astype(np.int8)
    rejected_index = generator.integers(0, 20, size=(rejected, 10)).astype(np.int8)
    chosen_distance = np.array([1 if row % 2 else 2 for row in range(chosen)])
    rejected_distance = np.array([1 if row % 2 else 2 for row in range(rejected)])
    return preferences.PreferencePopulation(
        split="train", chosen_index=chosen_index, chosen_distance=chosen_distance,
        rejected_index=rejected_index, rejected_distance=rejected_distance,
        excluded_chosen_distances={})


def pairing(seed=20260924):
    return preferences.PreferencePairing(population(), seed=seed)


# ---------------------------------------------------------------------------
# the common ordered stream
# ---------------------------------------------------------------------------

def test_every_arm_at_one_seed_consumes_the_same_resolved_order():
    first = streams.resolve_task_stream(pairing(), seed=1, exposures=20, batch_rows=4,
                                        pairing_seed=20264925)
    second = streams.resolve_task_stream(pairing(), seed=1, exposures=20, batch_rows=4,
                                         pairing_seed=20264925)
    assert np.array_equal(first.chosen_rows, second.chosen_rows)
    assert np.array_equal(first.rejected_rows, second.rejected_rows)
    assert first.document()["chosen_rows_sha256"] == second.document()["chosen_rows_sha256"]


def test_exposures_are_exact_at_every_update_and_endpoint():
    stream = streams.resolve_task_stream(pairing(), seed=1, exposures=20, batch_rows=4,
                                         pairing_seed=1)
    assert stream.exposures == 20 and stream.updates == 5
    for update in range(1, 6):
        chosen, rejected = stream.batch(update)
        assert chosen.size == 4 and rejected.size == 4
    endpoints = streams.exposure_endpoints([2, 5], batch_rows=4)
    assert endpoints == {2: 8, 5: 20}


def test_a_batch_that_spans_a_cycle_boundary_is_filled_rather_than_dropped():
    """Ten chosen rows and batches of four: update 3 straddles the cycle edge."""
    stream = streams.resolve_task_stream(pairing(), seed=1, exposures=20, batch_rows=4,
                                         pairing_seed=1)
    spans = stream.cycle_boundary_updates()
    assert spans == [3], "the boundary lands inside update 3 and the batch is filled across it"
    chosen, _ = stream.batch(3)
    assert chosen.size == 4
    cycles = stream.cycle_of_position[8:12]
    assert set(cycles.tolist()) == {0, 1}
    assert dict(zip(*np.unique(stream.cycle_of_position, return_counts=True)))[0] == 10


def test_every_position_pairs_rows_from_one_wild_type_distance_group():
    stream = streams.resolve_task_stream(pairing(), seed=1, exposures=20, batch_rows=4,
                                         pairing_seed=1)
    people = population()
    assert bool((people.chosen_distance[stream.chosen_rows]
                 == people.rejected_distance[stream.rejected_rows]).all())


def test_an_exposure_count_that_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="do not divide"):
        streams.resolve_task_stream(pairing(), seed=1, exposures=21, batch_rows=4, pairing_seed=1)


def test_the_stream_document_records_the_order_digest_and_the_boundaries():
    stream = streams.resolve_task_stream(pairing(), seed=7, exposures=20, batch_rows=4,
                                         pairing_seed=3)
    document = stream.document()
    assert document["updates"] == 5 and document["batch_rows"] == 4
    assert document["cycles"] == {"0": 10, "1": 10}
    assert document["cycle_boundary_updates"] == [3]
    assert len(document["chosen_rows_sha256"]) == 64


def test_an_update_outside_the_resolved_stream_is_refused():
    stream = streams.resolve_task_stream(pairing(), seed=1, exposures=20, batch_rows=4,
                                         pairing_seed=1)
    with pytest.raises(ValueError, match="outside 1..5"):
        stream.batch(6)
    with pytest.raises(ValueError, match="outside 1..5"):
        stream.batch(0)


# ---------------------------------------------------------------------------
# the replay order
# ---------------------------------------------------------------------------

def test_the_replay_order_cycles_through_independently_shuffled_full_banks():
    order = streams.replay_order(bank_rows=10, exposures=25, seed=777000301, parent_seed=20260918)
    assert order.size == 25
    assert sorted(order[:10].tolist()) == list(range(10))
    assert sorted(order[10:20].tolist()) == list(range(10))
    assert not np.array_equal(order[:10], order[10:20]), "each cycle is shuffled independently"


def test_every_lambda_at_one_seed_reads_the_same_replay_order():
    first = streams.replay_order(bank_rows=10, exposures=20, seed=777000301, parent_seed=1)
    second = streams.replay_order(bank_rows=10, exposures=20, seed=777000301, parent_seed=1)
    assert np.array_equal(first, second)


def test_the_replay_order_is_decoupled_from_the_task_stream():
    """Changing the task pairing seed must not move a single replay row."""
    order = streams.replay_order(bank_rows=10, exposures=20, seed=777000301, parent_seed=1)
    first = streams.resolve_task_stream(pairing(seed=1), seed=1, exposures=20, batch_rows=4,
                                        pairing_seed=1)
    second = streams.resolve_task_stream(pairing(seed=2), seed=1, exposures=20, batch_rows=4,
                                         pairing_seed=2)
    assert not np.array_equal(first.chosen_rows, second.chosen_rows)
    assert np.array_equal(order, streams.replay_order(bank_rows=10, exposures=20,
                                                      seed=777000301, parent_seed=1))


def test_replay_batch_rows_refuses_to_read_past_the_resolved_order():
    order = streams.replay_order(bank_rows=10, exposures=8, seed=777000301, parent_seed=1)
    assert streams.replay_batch_rows(order, 2, batch_rows=4).size == 4
    with pytest.raises(ValueError, match="beyond the resolved order"):
        streams.replay_batch_rows(order, 3, batch_rows=4)


def test_the_replay_order_document_reports_cycles_and_duplicates():
    order = streams.replay_order(bank_rows=10, exposures=25, seed=1, parent_seed=2)
    document = streams.replay_order_document(order, bank_rows=10, seed=1, parent_seed=2,
                                             batch_rows=5)
    assert document["cycles"] == 3 and document["rows"] == 25 and document["distinct_rows"] == 10


# ---------------------------------------------------------------------------
# the cadence
# ---------------------------------------------------------------------------

def test_the_declared_cadence_is_1_25_50_and_not_1_26_51():
    cadence = streams.UpdateCadence(first_update=1, interval=25, endpoints=(100,))
    scheduled = cadence.scheduled(100)
    assert scheduled[:4] == [1, 25, 50, 75]
    assert scheduled[-1] == 100


def test_the_inherited_schedule_would_have_produced_a_different_cadence():
    """The reason a new cadence object exists rather than a config change."""
    schedule = guard.MonitorSchedule(first_update=1, update_interval=25, gpu_second_interval=1e9)
    hit = []
    for update in range(1, 80):
        due, _ = schedule.due(update, gpu_seconds=0.0)
        if due:
            hit.append(update)
            schedule.record(update, 0.0)
    assert hit[:3] == [1, 26, 51], (
        "MonitorSchedule measures its interval from the last check, so it cannot express "
        "1, 25, 50 and the declared cadence needs its own object")
    with pytest.raises(ValueError, match="GPU-second interval must be positive"):
        guard.MonitorSchedule(gpu_second_interval=0)


def test_an_endpoint_that_is_also_an_interval_multiple_produces_one_check():
    cadence = streams.UpdateCadence(first_update=1, interval=25, endpoints=(50, 100))
    due, reason = cadence.due(50)
    assert due and reason == "exposure_endpoint"
    assert cadence.scheduled(100).count(50) == 1


def test_the_final_update_is_always_checked_even_off_cadence():
    cadence = streams.UpdateCadence(first_update=1, interval=25, endpoints=())
    assert cadence.due(37, final=False) == (False, None)
    assert cadence.due(37, final=True) == (True, "final_state")


def test_the_cadence_document_names_the_amendment_and_what_it_did_not_change():
    document = streams.UpdateCadence(endpoints=(1000,)).document()
    assert "five-GPU-second" in document["amendment"]
    assert "threshold are unchanged" in document["amendment"]


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_a_seed_inside_the_historical_band_is_refused():
    with pytest.raises(ValueError, match="historical draw-seed band"):
        streams.require_disjoint_seeds({"replay_draw": 20260930})


@pytest.mark.parametrize("value", [20260921, 20360920, 20300000])
def test_the_whole_historical_band_is_refused(value):
    with pytest.raises(ValueError, match="historical draw-seed band"):
        streams.require_disjoint_seeds({"probe": value})


def test_two_purposes_may_not_share_one_seed():
    with pytest.raises(ValueError, match="more than one purpose"):
        streams.require_disjoint_seeds({"replay_draw": 777000101, "monitor_draw": 777000101})


def test_declared_literals_outside_the_band_are_accepted_and_recorded():
    block = streams.require_disjoint_seeds({"replay_draw": 777000101,
                                            "monitor_draw": 777000201})
    assert block["band"] == [20260921, 20360920]
    assert block["seeds"]["replay_draw"] == 777000101


# ---------------------------------------------------------------------------
# the reviewable manifest block
# ---------------------------------------------------------------------------

def test_the_stream_manifest_carries_identities_and_no_arrays():
    stream = streams.resolve_task_stream(pairing(), seed=5, exposures=20, batch_rows=4,
                                         pairing_seed=5)
    document = streams.stream_manifest({5: stream}, endpoints=[2, 5], batch_rows=4,
                                       cadence=streams.UpdateCadence(endpoints=(2, 5)))
    assert document["exposure_endpoints"] == {"2": 8, "5": 20}
    block = document["task_streams"]["5"]
    assert set(block) >= {"chosen_rows_sha256", "rejected_rows_sha256"}
    assert all(not isinstance(value, (list, np.ndarray)) or key in ("cycle_boundary_updates",)
               for key, value in block.items()), "arrays live in shards, not in the manifest"
    assert len(document["identity_sha256"]) == 64
    again = streams.stream_manifest({5: stream}, endpoints=[2, 5], batch_rows=4,
                                    cadence=streams.UpdateCadence(endpoints=(2, 5)))
    assert again["identity_sha256"] == document["identity_sha256"]
