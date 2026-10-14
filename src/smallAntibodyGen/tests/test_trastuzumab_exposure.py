"""Exposure planning agrees with actual draws without observing their values."""
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc.exposure import plan_post_exposure
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest, MeasuredEndpoint,
    RestrictedPostData, SelectionAllowance, select_budget,
)
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import keyed_random
from smallAntibodyGen.experiments.trastuzumab_poc.runner import TrainingConfig


def restricted_data(assays=("b", "a", "b", "a", "b", "a", "c", "c", "singleton"), *, tied=False, reversed_values=False):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    entries = tuple(EndpointEntry(
        str(index), EndpointConstruct("AC", "A" * 7 + alphabet[index // 400]
                                     + alphabet[(index // 20) % 20] + alphabet[index % 20],
                                     "DE", "FG", "HIK"),
        "post_pool", "synthetic-neighborhood", assay, "synthetic endpoint")
        for index, assay in enumerate(assays))
    manifest = EndpointManifest(entries)
    budget = select_budget(manifest, len(entries), "shared-budget-v1")
    labels = tuple(MeasuredEndpoint(entry.endpoint_id,
                                    0.0 if tied else (-1 if reversed_values else 1) * float(index),
                                    "higher_better", entry.assay_id)
                   for index, entry in enumerate(entries))
    return RestrictedPostData(manifest=manifest, budget=budget, measured=labels,
                              fitting_count=len(entries), development_labels=(),
                              selection_allowance=SelectionAllowance(0, ()))


@pytest.mark.parametrize("seed", [0, 19])
@pytest.mark.parametrize("ordering_weight", [0.0, 1.0])
@pytest.mark.parametrize("tied,reversed_values", [(False, False), (True, False), (False, True)])
def test_plan_matches_actual_schedule_independently_of_outcomes(seed, ordering_weight, tied, reversed_values):
    data = restricted_data(tied=tied, reversed_values=reversed_values)
    config = TrainingConfig(stage="post", seed=seed, total_updates=11,
                            examples_per_update=3, ordering_weight=ordering_weight)
    planned = plan_post_exposure(data.selected_entries, config)
    reconstructed = dict.fromkeys(data.budget.selected_ids, 0)
    paired = dict.fromkeys(data.budget.selected_ids, 0)
    for update in range(config.total_updates):
        for example in range(config.examples_per_update):
            entry = keyed_random(seed, update, example, "endpoint").choice(data.selected_entries)
            reconstructed[entry.endpoint_id] += 1
            if ordering_weight:
                pair = data.draw_pair(keyed_random(seed, update, example, "pair"))
                paired[pair.left_id] += 1
                paired[pair.right_id] += 1
    assert planned["reconstruction"]["per_endpoint_counts"] == reconstructed
    assert planned["pair_labels"]["per_endpoint_counts"] == paired
    assert planned["pair_ineligible_singleton_ids"] == ["8"]
    assert planned["pair_labels"]["eligible_endpoint_count"] == 8
    assert planned["pair_labels"]["minimum_count_all_selected"] == 0
    assert planned["pair_labels"]["total_endpoint_exposures"] == 66 * bool(ordering_weight)
    assert planned["reconstruction"]["total_endpoint_exposures"] == 33
    assert json.loads(json.dumps(planned, allow_nan=False)) == planned
    different_values = restricted_data(tied=not tied, reversed_values=not reversed_values)
    assert plan_post_exposure(different_values.selected_entries, config) == planned


def test_reads_identity_metadata_only_and_preserves_outer_rng():
    class Entry:
        role = "post_pool"
        assay_id = "synthetic-assay"

        def __init__(self, endpoint_id):
            self.endpoint_id = endpoint_id

        @property
        def value(self):
            raise AssertionError("must not read outcomes")

        @property
        def construct(self):
            raise AssertionError("must not read sequence or construct")

    config = TrainingConfig(stage="post", total_updates=4, examples_per_update=2, ordering_weight=1)
    python_rng = random.getstate()
    torch_rng = torch.random.get_rng_state().clone()
    report = plan_post_exposure((Entry("one"), Entry("two")), config)
    assert report["pair_labels"]["per_endpoint_counts"] == {"one": 8, "two": 8}
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_reconstruction_preserves_budget_order_while_pairs_are_order_invariant():
    data = restricted_data()
    config = TrainingConfig(stage="post", total_updates=17, examples_per_update=2, ordering_weight=1)
    original = plan_post_exposure(data.selected_entries, config)
    reversed_plan = plan_post_exposure(tuple(reversed(data.selected_entries)), config)
    assert original["pair_labels"] == reversed_plan["pair_labels"]
    assert original["reconstruction"]["per_endpoint_counts"] != reversed_plan["reconstruction"]["per_endpoint_counts"]


def test_large_budget_reports_unused_measurements_separately_from_reconstruction():
    data = restricted_data(("assay",) * 2146)
    config = TrainingConfig(stage="post", seed=0, total_updates=500,
                            examples_per_update=2, ordering_weight=1)
    planned = plan_post_exposure(data.selected_entries, config)
    reconstruction, pairs = planned["reconstruction"], planned["pair_labels"]
    assert 0 < reconstruction["seen_endpoint_count"] < 1000
    assert 0 < pairs["seen_endpoint_count"] < 2000
    assert pairs["unseen_endpoint_count"] == 2146 - pairs["seen_endpoint_count"]
    assert pairs["fraction_of_selected"] == pairs["seen_endpoint_count"] / 2146
    assert pairs["minimum_count_all_selected"] == 0
    assert set(reconstruction["seen_endpoint_ids"]) - set(pairs["seen_endpoint_ids"])
    assert set(pairs["seen_endpoint_ids"]) - set(reconstruction["seen_endpoint_ids"])


def test_singleton_without_ordering_has_no_eligible_pair_denominator():
    data = restricted_data(("assay",))
    config = TrainingConfig(stage="post", total_updates=3, examples_per_update=2)
    result = plan_post_exposure(data.selected_entries, config)
    assert result["reconstruction"]["per_endpoint_counts"] == {"0": 6}
    assert result["pair_labels"]["fraction_of_eligible"] is None
    assert result["pair_labels"]["minimum_count_eligible"] is None
    assert result["pair_labels"]["fraction_of_selected"] == 0
    with pytest.raises(ValueError, match="two selected endpoints"):
        plan_post_exposure(data.selected_entries, replace(config, ordering_weight=1))


@pytest.mark.parametrize("changes", [
    {"stage": "mid"}, {"seed": True}, {"seed": -1},
    {"total_updates": 0}, {"total_updates": True}, {"examples_per_update": 0},
    {"ordering_weight": True}, {"ordering_weight": -1},
    {"ordering_weight": float("nan")}, {"ordering_weight": float("inf")},
])
def test_invalid_config_rejected(changes):
    config = SimpleNamespace(stage="post", seed=0, total_updates=1,
                             examples_per_update=1, ordering_weight=0)
    for name, value in changes.items():
        setattr(config, name, value)
    with pytest.raises(ValueError):
        plan_post_exposure(restricted_data().selected_entries, config)


def test_empty_duplicate_or_nontraining_entries_rejected():
    data = restricted_data()
    config = TrainingConfig(stage="post")
    for entries in ((), (data.selected_entries[0],) * 2,
                    (replace(data.selected_entries[0], role="reserved"),)):
        with pytest.raises(ValueError):
            plan_post_exposure(entries, config)
