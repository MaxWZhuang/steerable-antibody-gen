"""The pilot runner against a real tiny native DPLM on CPU.

What is real: the frozen-input stage (``her2_multiscale_world``), the base curriculum stream, the
four tasks through ``tasks.dispatch``, collation, the DPLM conditioning wrapper, the diffusion
reconstruction, the sequential joint, the heads, Adafactor, backward, clipping, the snapshot and its
reload. What is replaced: exactly one seam, ``dplm_policy.load_pinned_dplm`` (the pinned weights are
not here), by a recorder that returns ``NativeDPLM.random_init`` (32 wide, 2 layers, 4 heads, 256
positions). Nothing about the new losses, the optimizer or the loop is mocked, and
``FlightServices.train`` is made to fail if anything calls it.

The stream is the real one with its task and family schedule fixed (reconstruction, local, whole
window, heads, repeating) so that every task is exercised in four updates; the draws, rows,
constructs, conditions and contrasts are the real stream's.
"""
from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_multiscale import config as config_lib
from smallAntibodyGen.experiments.her2_multiscale import evaluation as evaluation_lib
from smallAntibodyGen.experiments.her2_multiscale import inputs as inputs_lib
from smallAntibodyGen.experiments.her2_multiscale import objective as objective_lib
from smallAntibodyGen.experiments.her2_multiscale import pairs as pairs_lib
from smallAntibodyGen.experiments.her2_multiscale import runner as runner_lib
from smallAntibodyGen.tests import her2_multiscale_world as world_lib

ROUTE = config_lib.ROUTE
TASKS = set(data_lib.TASKS)
SCRIPT = world_lib.REPO / "scripts" / "her2_multiscale_pilot.py"
EMPTY_TALLY = {"local_draws": 0, "local_replaced": 0, "pair_endpoint_rows": 0,
               "replacement_distances": {}, "local_pair_eligible": 0, "local_with_quartets": 0,
               "local_other_family": 0}


def prepare(world, arm, *, name=None):
    """The loaded inputs, pair pool, plan and fixed-schedule stream of one arm. Builds no model."""
    pilot = world.pilot().with_arm(arm)
    loaded, context = world.load(name or f"arm-{arm}", pilot=pilot)
    services = loaded.services
    pool = pairs_lib.build_pair_pool(
        loaded.pool, closure=loaded.closure, window_of=loaded.window_of,
        training_population_keys=loaded.training_keys, distances=config_lib.DISTANCES,
        capacity=pilot.reservoir_capacity, seed=pilot.seed)
    plan = runner_lib.ArmPlan.from_pilot(pilot, context.config["curriculum"])
    stream = world_lib.cycling_stream(
        services.curriculum_stream(ROUTE, seed=pilot.seed, arm="full", updates=pilot.updates),
        services.selected_shapes()["accumulation"])
    return types.SimpleNamespace(pilot=pilot, loaded=loaded, context=context, services=services,
                                 pool=pool, plan=plan, stream=stream,
                                 directory=context.run_root, built=[], built_heads=[],
                                 identity={"schema": "test", "arm": arm})


def execute(run, **changes):
    services = run.services
    real = run.build_route = services.build_route     # `run.build_route` builds without recording
    services.build_route = lambda route, seed=0: run.built.append(real(route, seed=seed)) or \
        run.built[-1]
    real_heads = run.build_heads = services.build_heads  # `run.build_heads` builds without recording
    services.build_heads = lambda route, model: run.built_heads.append(
        real_heads(route, model)) or run.built_heads[-1]
    arguments = dict(plan=run.plan, stream=run.stream, pool=run.pool, closure=run.loaded.closure,
                     window_of=run.loaded.window_of, keys=run.loaded.training_keys,
                     panel=run.loaded.panel, directory=run.directory, identity=run.identity)
    arguments.update(changes)
    run.record = runner_lib.execute(services, ROUTE, **arguments)
    run.journal = [json.loads(line) for line in
                   (run.directory / runner_lib.PROGRESS).read_text(encoding="utf-8").splitlines()]
    run.evaluation = json.loads((run.directory / "evaluation.json").read_text(encoding="utf-8"))
    return run


def microbatches(run, task):
    return [(step["step"], entry) for step in run.journal for entry in step["microbatches"]
            if not entry["skipped"] and entry["task"] == task]


def forbid_train(patch):
    def refuse(*args, **kwargs):
        raise AssertionError("FlightServices.train must not be called by the pilot")
    patch.setattr(services_mod.FlightServices, "train", refuse)


@pytest.fixture(scope="module")
def pair_world(tmp_path_factory):
    """Pair-only local draws (no quartet), so every local draw is eligible for replacement."""
    with pytest.MonkeyPatch.context() as patch:
        forbid_train(patch)
        yield world_lib.build_world(tmp_path_factory.mktemp("pairs"), patch,
                                    quartet_probability=0.0)


@pytest.fixture(scope="module")
def quartet_world(tmp_path_factory):
    """Local draws that are always a complete quartet: nothing is eligible for replacement."""
    with pytest.MonkeyPatch.context() as patch:
        forbid_train(patch)
        yield world_lib.build_world(tmp_path_factory.mktemp("quartets"), patch,
                                    quartet_probability=1.0)


@pytest.fixture(scope="module")
def runs(pair_world):
    """All four arms, run once, each with the draws ``tasks.dispatch`` was handed (``.dispatched``)."""
    real, log, results = tasks_lib.dispatch, [], {}

    def spy(services, route, model, heads, draw, **kwargs):
        log.append(draw)
        return real(services, route, model, heads, draw, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tasks_lib, "dispatch", spy)
        for arm in config_lib.ARMS:
            start = len(log)
            results[arm] = execute(prepare(pair_world, arm))
            results[arm].dispatched = log[start:]
    return results


# --- one complete run -------------------------------------------------------------------------------

def test_a_combined_run_completes_and_writes_its_records(runs):
    run = runs["combined"]
    record = run.record
    assert record["status"] == "completed" and record["updates"] == 4
    assert record["optimizer"] == "Adafactor"
    assert record["plan"]["mask_geometry"] == "subset"
    assert record["plan"]["distance_weights"] == {"1": 0.5, "2": 0.3, "3": 0.2}
    assert record["plan"]["full_probability"] == 0.5
    assert "no biological-improvement claim" in record["claim"]
    assert [step["step"] for step in run.journal] == [1, 2, 3, 4]
    assert sorted(path.name for path in run.directory.glob("*.pt")) == ["snapshot.pt"]
    assert (run.directory / "evaluation.json").is_file()
    assert record["snapshot"]["sha256"] == paths.sha256_file(run.directory / "snapshot.pt")
    assert record["snapshot"]["evaluation_only"] is True
    for step in run.journal:
        assert np.isfinite(step["loss"]) and np.isfinite(step["gradient_norm"])
        assert step["learning_rate"] > 0


def test_every_task_trains_under_the_declared_coefficients(runs):
    run = runs["combined"]
    entries = [entry for step in run.journal for entry in step["microbatches"]
               if not entry["skipped"]]
    assert {entry["task"] for entry in entries} == TASKS
    for task in TASKS:
        assert len(microbatches(run, task)) == 2
    terms = {name for step in run.journal for name in step["terms"]}
    assert {"reconstruction", "whole_window", "local_effect", "local_reconstruction"} <= terms
    assert run.record["coefficients"] == {"reconstruction": 1.0, "local_effect": 1.0,
                                          "whole_window": 1.0, "heads": 0.5}


def test_pairs_replace_the_pair_only_local_draws_and_nothing_else(runs):
    run = runs["combined"]
    local = microbatches(run, data_lib.TASK_LOCAL)
    assert len(local) == 2
    for _, entry in local:
        replacement = entry["replacement"]
        assert replacement["kind"] == "measured_pairs" and replacement["distance"] in (1, 2, 3)
        assert replacement["pairs"] == 2 and replacement["base_rows"] == 4
        assert len(replacement["structural_digests"]) == 2
        assert len(replacement["identity_digests"]) == 4
    for task in (data_lib.TASK_RECONSTRUCTION, data_lib.TASK_WHOLE, data_lib.TASK_HEADS):
        assert all("replacement" not in entry for _, entry in microbatches(run, task))
    tally = run.record["tally"]
    assert tally["local_draws"] == tally["local_pair_eligible"] == tally["local_replaced"] == 2
    assert tally["local_with_quartets"] == 0 and tally["local_other_family"] == 0
    assert sum(tally["replacement_distances"].values()) == 2
    assert tally["pair_endpoint_rows"] == 8


def test_forwards_are_counted_as_they_ran_for_every_distance(runs):
    """A pair at distance d is 2 rows x (d sequential reveals + 1 reconstruction forward)."""
    seen = set()
    for run in runs.values():
        for _, entry in microbatches(run, data_lib.TASK_LOCAL):
            distance = entry["replacement"]["distance"]
            seen.add(distance)
            assert entry["rows"] == 4
            assert entry["forwards"] == 4 * (distance + 1), (distance, entry["forwards"])
            assert entry["targets"] == 4 * (distance + distance)
            assert entry["term"] == "local_effect"
            assert sum(item["forwards"] for item in entry["contributions"]) == entry["forwards"]
            assert sum(item["rows"] for item in entry["contributions"]) == 4
    assert 1 in seen, "the distance-1 arms ran"


def test_the_ledger_agrees_with_what_the_microbatches_reported(runs):
    for arm, run in runs.items():
        entries = [entry for step in run.journal for entry in step["microbatches"]
                   if not entry["skipped"]]
        totals = run.record["exposure"]["totals"]
        assert totals["optimizer_updates"] == 4, arm
        assert totals["microbatch_rows"] == sum(entry["rows"] for entry in entries), arm
        assert totals["student_forwards"] == sum(entry["forwards"] for entry in entries), arm
        assert totals["supervised_targets"] == sum(entry["targets"] for entry in entries), arm
        assert all(block["additive"] is not False for kind in
                   run.record["exposure"]["additivity"].values() for block in kind.values())


def moved_names(trained, untouched, prefix=None):
    return [name for (name, a), (_, b) in zip(trained.state_dict().items(),
                                              untouched.state_dict().items())
            if (prefix is None or name.startswith(prefix)) and not torch.equal(a, b)]


def starting_point(run):
    """The model and heads exactly as the loop built them (same seed, same RNG consumption)."""
    model = run.build_route(ROUTE, seed=run.plan.seed)
    return model, run.build_heads(ROUTE, model)


def test_trunk_conditioning_and_heads_each_move_and_stay_finite(runs):
    """Three separate claims about three parameter groups of the real native model."""
    run = runs["combined"]
    trained, trained_heads = run.built[0], run.built_heads[0]    # what the loop actually trained
    untouched, untouched_heads = starting_point(run)
    trunk = moved_names(trained, untouched, "native.")
    tables = {name: moved_names(trained, untouched, name) for name in
              ("condition_family", "condition_bin", "condition_assay")}
    heads = moved_names(trained_heads, untouched_heads)
    assert trunk, "no native trunk parameter changed in four updates"
    assert any(tables.values()), f"no condition table changed: {tables}"
    assert heads, "no observation-head parameter changed although the heads task ran twice"
    assert sum(float(table.weight.detach().abs().sum()) for table in trained._tables()) > 0, \
        "the zero-initialised condition tables never left 0"
    assert sum(float(table.weight.detach().abs().sum()) for table in untouched._tables()) == 0
    for module in (trained, trained_heads):
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())


def test_the_snapshot_reloads_strictly_with_its_heads_and_its_scores(runs):
    run = runs["combined"]
    services = run.services
    trained, trained_heads = run.built[0], run.built_heads[0]
    untouched, untouched_heads = starting_point(run)
    model, heads = starting_point(run)
    record = runner_lib.load_snapshot(run.directory / "snapshot.pt", model=model, heads=heads,
                                      identity=run.identity)
    assert record["evaluation_only"] is True and record["step"] == 4
    for (name, a), (_, b) in zip(trained.state_dict().items(), model.state_dict().items()):
        assert torch.equal(a, b), name
    for (name, a), (_, b) in zip(trained_heads.state_dict().items(), heads.state_dict().items()):
        assert torch.equal(a, b), name
    assert moved_names(model, untouched) and moved_names(heads, untouched_heads)

    def scores(one, two):
        terms = services.hr17_selection_terms(ROUTE, one, two, run.loaded.panel, batch_size=2)
        return np.asarray(terms["high"]), np.asarray(terms["null"])

    trained_scores, restored_scores = scores(trained, trained_heads), scores(model, heads)
    for original, restored in zip(trained_scores, restored_scores):
        assert np.array_equal(original, restored), "restored scores differ from the trained ones"
    start_scores = scores(untouched, untouched_heads)
    assert not np.array_equal(trained_scores[0], start_scores[0]), \
        "the trained model scores like its starting point, so the comparison proves nothing"
    # The same holds for the whole development block the evaluation reports.
    assert json.dumps(evaluation_lib.development_ranking(
        services, ROUTE, trained, trained_heads, run.loaded.panel, batch_size=2), sort_keys=True) \
        == json.dumps(evaluation_lib.development_ranking(
            services, ROUTE, model, heads, run.loaded.panel, batch_size=2), sort_keys=True)


def test_the_snapshot_cannot_be_resumed_or_loaded_under_another_identity(runs, tmp_path):
    run = runs["combined"]
    services, path = run.services, run.directory / "snapshot.pt"
    model = run.build_route(ROUTE, seed=run.plan.seed)
    heads = services.build_heads(ROUTE, model)
    with pytest.raises(ValueError, match="another identity"):
        runner_lib.load_snapshot(path, model=model, heads=heads,
                                 identity=dict(run.identity, arm="baseline"))
    with pytest.raises(ValueError, match="her2-routes-checkpoint"):
        checkpoints.restore(path, model=model, heads=heads, optimizer=None, scheduler=None,
                            state=services.fresh_state(1), identity=None)
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    assert payload["evaluation_only"] is True and "optimizer" not in payload
    assert "rng" not in payload and "scheduler" not in payload
    broken = dict(payload, model={key: value for key, value in list(payload["model"].items())[1:]})
    target = tmp_path / "broken.pt"
    torch.save(broken, str(target))
    with pytest.raises(ValueError, match="keys differ"):
        runner_lib.load_snapshot(target, model=model, heads=heads, identity=run.identity)
    unmarked = dict(payload, evaluation_only=False)
    torch.save(unmarked, str(target))
    with pytest.raises(ValueError, match="evaluation-only"):
        runner_lib.load_snapshot(target, model=model, heads=heads, identity=run.identity)


# --- the snapshot loads strictly: nothing is cast and nothing is half-loaded ----------------------------------

def snapshot_payload(run):
    return torch.load(str(run.directory / "snapshot.pt"), map_location="cpu", weights_only=True)


def first_key(payload, block):
    return next(iter(payload[block]))


def swap(block, change):
    def mutate(payload):
        key = first_key(payload, block)
        payload[block][key] = change(payload[block][key])
    return mutate


def grow(tensor):
    return torch.zeros(tensor.shape[0] + 1, *tensor.shape[1:], dtype=tensor.dtype)


def drop_first(block):
    return lambda payload: payload[block].pop(first_key(payload, block))


def add_extra(block):
    return lambda payload: payload[block].update({"extra.weight": torch.zeros(1)})


def tamper_identity(payload):
    payload["model_format"]["base_identity"]["tampered"] = 1


CORRUPTIONS = {
    "model dtype": (swap("model", lambda tensor: tensor.double()), "has dtype torch.float64"),
    "model shape": (swap("model", grow), "has shape"),
    "model missing key": (drop_first("model"), "keys differ"),
    "model extra key": (add_extra("model"), "keys differ"),
    "model not a tensor": (swap("model", lambda tensor: 3), "flat mapping of tensors"),
    "heads dtype": (swap("heads", lambda tensor: tensor.double()), "has dtype torch.float64"),
    "heads shape": (swap("heads", grow), "has shape"),
    "heads missing key": (drop_first("heads"), "keys differ"),
    "heads extra key": (add_extra("heads"), "keys differ"),
    "heads not a tensor": (swap("heads", lambda tensor: None), "flat mapping of tensors"),
    "base identity": (tamper_identity, "different base"),
}


def state_of(model, heads):
    return ({name: value.clone() for name, value in model.state_dict().items()},
            {name: value.clone() for name, value in heads.state_dict().items()})


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_a_bad_snapshot_is_refused_and_leaves_both_model_and_heads_untouched(runs, tmp_path, name):
    """The valid half of the payload is the TRAINED state, so a half-load would be visible."""
    run = runs["combined"]
    mutate, message = CORRUPTIONS[name]
    payload = snapshot_payload(run)
    mutate(payload)
    target = tmp_path / "bad.pt"
    torch.save(payload, str(target))
    model, heads = starting_point(run)
    before = state_of(model, heads)
    with pytest.raises(ValueError, match=message):
        runner_lib.load_snapshot(target, model=model, heads=heads, identity=run.identity)
    for original, current in zip(before, state_of(model, heads)):
        assert original.keys() == current.keys()
        for key, value in original.items():
            assert current[key].dtype == value.dtype and torch.equal(current[key], value), \
                f"{name}: {key} was modified by a refused load"


def test_a_snapshot_without_a_required_block_is_refused(runs, tmp_path):
    run = runs["combined"]
    payload = snapshot_payload(run)
    payload.pop("heads")
    target = tmp_path / "headless.pt"
    torch.save(payload, str(target))
    model, heads = starting_point(run)
    with pytest.raises(ValueError, match="lacks"):
        runner_lib.load_snapshot(target, model=model, heads=heads, identity=run.identity)


def test_the_matching_snapshot_restores_the_trained_dtypes_unchanged(runs):
    run = runs["combined"]
    model, heads = starting_point(run)
    runner_lib.load_snapshot(run.directory / "snapshot.pt", model=model, heads=heads,
                             identity=run.identity)
    payload = snapshot_payload(run)
    for key, value in payload["model"].items():
        assert model.state_dict()[key].dtype == value.dtype
    for key, value in payload["heads"].items():
        assert heads.state_dict()[key].dtype == value.dtype


# --- the masks ------------------------------------------------------------------------------------------------

def mask_summaries(run):
    return [(step, entry["micro"], entry["masks"]) for step, entry in
            microbatches(run, data_lib.TASK_WHOLE)]


def test_the_arms_see_the_same_tasks_families_and_whole_window_rows(runs):
    schedules = {arm: [(step["step"], entry["micro"], entry.get("task"), entry.get("family"))
                       for step in run.journal for entry in step["microbatches"]]
                 for arm, run in runs.items()}
    assert len({json.dumps(value) for value in schedules.values()}) == 1
    rows = {arm: [(entry.get("task"), entry.get("rows")) for step in run.journal
                  for entry in step["microbatches"] if entry.get("task") != data_lib.TASK_LOCAL]
            for arm, run in runs.items()}
    assert len({json.dumps(value) for value in rows.values()}) == 1


def test_a_mask_arm_and_its_control_hide_the_same_number_of_sites_of_the_same_rows(runs):
    baseline, masks, pairs, combined = (mask_summaries(runs[arm]) for arm in
                                        ("baseline", "masks", "pairs", "combined"))
    assert baseline == pairs and masks == combined and baseline    # k and mode per row
    assert baseline == masks, "suffix and subset share the full/partial decisions and every k"
    for _, _, summary in baseline:
        assert summary["views"] == 4 and set(summary["modes"]) <= {"full", "partial"}
        assert all(1 <= int(count) <= 10 for count in summary["counts"])


def test_the_dispatch_received_the_replaced_whole_window_views(runs):
    hidden = set()
    for arm, run in runs.items():
        whole = [draw for draw in run.dispatched if draw.task == data_lib.TASK_WHOLE]
        assert len(whole) == 2, arm
        for draw in whole:
            counts = [len(serialization.query_slots) for _, serialization, _ in draw.views]
            assert len(counts) == 4 and all(1 <= count <= 10 for count in counts)
            hidden.update(counts)
    assert len(hidden) > 1, "a replaced mask is not always the full window"
    local = [draw for run in runs.values() for draw in run.dispatched
             if draw.task == data_lib.TASK_LOCAL]
    assert local == [], "a replaced local draw never reaches dispatch; the pair objective ran"


def test_the_local_replacement_decisions_are_the_same_in_every_arm(runs):
    placed = {arm: [(step, entry["micro"], "replacement" in entry) for step, entry in
                    microbatches(run, data_lib.TASK_LOCAL)] for arm, run in runs.items()}
    assert len({json.dumps(value) for value in placed.values()}) == 1
    assert placed["baseline"] == [(1, 1, True), (3, 1, True)]
    distances = {arm: [entry["replacement"]["distance"] for _, entry in
                       microbatches(run, data_lib.TASK_LOCAL)] for arm, run in runs.items()}
    # The keyed draws of the default seed: the distance-one arms replace both local draws at
    # distance 1; the pair arms draw distance 2 at update 1 and distance 3 at update 3.
    assert distances == {"baseline": [1, 1], "masks": [1, 1], "pairs": [2, 3],
                         "combined": [2, 3]}
    assert set().union(*map(set, distances.values())) == {1, 2, 3}, "all three distances were seen"


def test_the_realized_replacement_counts_are_reported_not_inferred_from_the_config(runs):
    expected = {"baseline": {"1": 2}, "masks": {"1": 2}, "pairs": {"2": 1, "3": 1},
                "combined": {"2": 1, "3": 1}}
    for arm, run in runs.items():
        realized = run.record["realized_training"]
        assert realized["replacement_distances"] == expected[arm], arm
        assert realized["replacement_microbatches"] == 2 and realized["local_draws"] == 2
        assert realized["replacement_endpoint_rows"] == 8
        assert realized["retained_legacy_local"] == {
            "her2_pair_only_not_replaced": 0, "her2_with_quartets_or_without_contrasts": 0,
            "other_family": 0}
        assert sum(realized["replacement_distances"].values()) \
            == run.record["tally"]["local_replaced"]
        fit = run.evaluation["after"]["training_fit_by_distance"]
        assert fit["realized_training"] == realized, "the final evaluation reports the final tally"


def test_before_any_update_the_evaluation_has_configured_distances_and_no_realized_ones(runs):
    configured = {"baseline": [1], "masks": [1], "pairs": [1, 2, 3], "combined": [1, 2, 3]}
    for arm, run in runs.items():
        before = run.evaluation["before"]["training_fit_by_distance"]
        after = run.evaluation["after"]["training_fit_by_distance"]
        assert "distances_trained_by_this_arm" not in before
        assert "distances_trained_by_this_arm" not in after
        assert before["configured_replacement_distances"] == configured[arm]
        assert after["configured_replacement_distances"] == configured[arm]
        assert "not evidence" in before["configured_note"]
        # Configured, but nothing has been drawn yet: the realized block is empty and says so.
        assert before["realized_training"]["replacement_distances"] == {}
        assert before["realized_training"]["replacement_microbatches"] == 0
        assert before["realized_training"]["local_draws"] == 0
        assert after["realized_training"]["replacement_distances"] != {}
        assert "replacement measured-pair objective" in before["objective_scope"]
        assert "retained legacy" in before["objective_scope"]


def test_pair_arms_are_configured_with_distances_the_distance_one_arms_are_not(runs):
    for arm in ("baseline", "masks"):
        assert set(runs[arm].record["plan"]["distance_weights"]) == {"1"}
        configured = runs[arm].evaluation["after"]["training_fit_by_distance"]
        assert configured["configured_replacement_distances"] == [1]
    for arm in ("pairs", "combined"):
        configured = runs[arm].evaluation["after"]["training_fit_by_distance"]
        assert configured["configured_replacement_distances"] == [1, 2, 3]


# --- the shared stream: the arms present the same rows -----------------------------------------------------

def presented(draw):
    """The rows and conditions a draw presents (apart from how its sites are masked)."""
    return [(construct.identity(), condition.family, condition.bin, condition.assay)
            for construct, _serialization, condition in draw.views]


def hidden_sites(draw):
    return [inputs_lib.plain(list(serialization.query_slots))
            for _construct, serialization, _condition in draw.views]


@pytest.mark.parametrize("task", [data_lib.TASK_RECONSTRUCTION, data_lib.TASK_WHOLE,
                                  data_lib.TASK_HEADS])
def test_the_arms_present_identical_constructs_and_conditions_for_every_shared_row(runs, task):
    rows = {arm: [presented(draw) for draw in run.dispatched if draw.task == task]
            for arm, run in runs.items()}
    assert len(rows["baseline"]) == 2 and all(len(draws) == 2 for draws in rows.values())
    assert all(view and all(identity for identity, *_ in view)
               for draws in rows.values() for view in draws)
    for arm in ("masks", "pairs", "combined"):
        assert rows[arm] == rows["baseline"], f"{arm} presents other rows than the baseline"


def test_arms_with_the_same_geometry_hide_the_same_sites_and_all_hide_the_same_number(runs):
    hidden = {arm: [hidden_sites(draw) for draw in run.dispatched
                    if draw.task == data_lib.TASK_WHOLE] for arm, run in runs.items()}
    assert hidden["baseline"] == hidden["pairs"], "the two suffix arms hide the same sites"
    assert hidden["masks"] == hidden["combined"], "the two subset arms hide the same sites"
    sizes = {arm: [[len(view) for view in draw] for draw in draws] for arm, draws in hidden.items()}
    assert sizes["baseline"] == sizes["masks"], "suffix and subset hide the same number of sites"
    assert all(1 <= size <= 10 for draw in sizes["baseline"] for size in draw)


# --- the native pair objective: gradients reach the real model ----------------------------------------------

@pytest.mark.parametrize("distance", [2, 3])
def test_a_pair_only_backward_reaches_the_native_trunk_and_conditioning_at_distance(
        runs, distance):
    """Real native model, real collation and sequential joint; no mocked scorer, no parameter probe."""
    run = runs["combined"]
    services = run.services
    model = run.build_route(ROUTE, seed=run.plan.seed)
    heads = run.build_heads(ROUTE, model)
    model.train()
    rng = np.random.default_rng([3, distance])
    pairs = [run.pool.draw(distance, rng) for _ in range(2)]
    outcome = objective_lib.pair_outcome(
        services, ROUTE, model, pairs, closure=run.loaded.closure, window_of=run.loaded.window_of,
        training_population_keys=run.loaded.training_keys, kappa=services.task_settings()["kappa"],
        rng=np.random.default_rng([5, distance]))
    assert outcome.term == "local_effect" and set(outcome.terms) == {"local_effect"}
    assert outcome.rows == 4 and outcome.forwards == 4 * distance == outcome.targets
    assert outcome.loss.requires_grad and bool(torch.isfinite(outcome.loss))
    outcome.loss.backward()
    grads = {name: parameter.grad for name, parameter in model.named_parameters()
             if parameter.grad is not None}
    trunk = [grad for name, grad in grads.items() if name.startswith("native.")]
    tables = [table.weight.grad for table in model._tables() if table.weight.grad is not None]
    assert trunk and all(bool(torch.isfinite(grad).all()) for grad in trunk)
    assert sum(float(grad.norm()) for grad in trunk) > 0.0, "the trunk received no gradient"
    assert tables and all(bool(torch.isfinite(grad).all()) for grad in tables)
    assert sum(float(grad.norm()) for grad in tables) > 0.0, "the condition tables received none"
    assert all(parameter.grad is None for parameter in heads.parameters()), \
        "a pair-only objective does not touch the observation heads"


def test_flight_services_train_is_never_called_and_the_loop_is_independent_of_it():
    source = Path(runner_lib.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    named = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
             and node.attr in ("train", "train_energy", "smoke") and isinstance(node.value, ast.Name)
             and node.value.id in ("services", "FlightServices")]
    assert named == []
    bases = [ast.unparse(base) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
             for base in node.bases]
    assert "FlightServices" not in bases


# --- evaluation ----------------------------------------------------------------------------------------------

def test_the_evaluation_is_labelled_and_the_initial_ratio_is_undefined_not_zero(runs):
    run = runs["combined"]
    before, after = run.evaluation["before"], run.evaluation["after"]
    ranking = before["development_ranking"]
    assert ranking["rows"] == 2 and ranking["divisor"] == 10 and ranking["assay"] == world_lib.ASSAY
    assert "bin=high" in ranking["high_only"]["label"]
    assert "minus" in ranking["high_minus_null"]["label"]
    undefined = ranking["high_minus_null"]["spearman_vs_pkd"]
    assert undefined["value"] is None and "constant" in undefined["reason"]
    assert ranking["high_minus_null"]["scores"]["mean"] == 0.0
    for block in (ranking["high_only"], after["development_ranking"]["high_only"]):
        assert np.isfinite(block["scores"]["mean"])
    assert set(after) == {"development_ranking", "completion_by_mask_count", "order_sensitivity",
                          "training_fit_by_distance", "heldout_local_gap"}
    assert after["heldout_local_gap"]["status"] == "deferred"
    assert "no number" in after["heldout_local_gap"]["reason"]
    completion = after["completion_by_mask_count"]
    assert set(completion["by_bin"]) == {"null", "high"}
    for bin_block in completion["by_bin"].values():
        assert set(bin_block) == {"suffix", "subset"}
        for geometry in bin_block.values():
            assert set(geometry) == {"1", "4", "10"}
            assert all(np.isfinite(cell["mean"]) and cell["rows"] == 2 for cell in geometry.values())
        assert bin_block["suffix"]["10"] == bin_block["subset"]["10"]
    order = after["order_sensitivity"]
    assert order["rows"] == 2 and np.isfinite(order["mean_abs_difference"])
    fit = after["training_fit_by_distance"]
    assert set(fit["by_distance"]) == {"1", "2", "3"}
    assert fit["population"].startswith("training pairs")
    for block in fit["by_distance"].values():
        assert block["pairs"] == 4 and np.isfinite(block["rmse"])


def test_training_fit_and_development_generalisation_are_separate_blocks(runs):
    after = runs["combined"].evaluation["after"]
    assert "training_fit_by_distance" in after and "development_ranking" in after
    assert "development" not in after["training_fit_by_distance"]["population"]
    assert "not generalisation" in after["training_fit_by_distance"]["population"]
    assert "development windows only" in after["development_ranking"]["note"]


def test_completion_and_order_diagnostics_are_kept_apart_from_the_ranking(runs):
    for when in ("before", "after"):
        record = runs["combined"].evaluation[when]
        ranking, completion, order = (record[name] for name in (
            "development_ranking", "completion_by_mask_count", "order_sensitivity"))
        assert not {"by_bin", "window_sites", "mean_abs_difference",
                    "spearman_forward_vs_reverse"} & set(ranking)
        assert not {"high_only", "high_minus_null", "null_only"} & (set(completion) | set(order))
        assert "spearman" not in json.dumps(completion), "completion is not a ranking"
        assert "not an affinity ranking" in completion["note"]
        assert completion["unit"].startswith("mean negative log likelihood")
        assert order["unit"].startswith("mean log probability per site")
    # At the start the conditioning tables are zero, so high minus null is exactly constant: the
    # record says so with a reason and never reports a number.
    constant = runs["combined"].evaluation["before"]["development_ranking"]["high_minus_null"]
    assert constant["scores"]["min"] == constant["scores"]["max"] == 0.0
    assert constant["spearman_vs_pkd"] == {
        "value": None, "reason": "the scores are constant, so they rank nothing", "rows": 2}


def numpy_state_equal(before, after):
    """The WHOLE legacy state: name, key array, position, gaussian flag and cached gaussian.

    Comparing only the name and the key array would miss a generator that advanced its position
    without changing the array.
    """
    return (before[0] == after[0] and np.array_equal(before[1], after[1])
            and tuple(before[2:]) == tuple(after[2:]))


def test_the_evaluation_restores_every_mode_and_every_generator(pair_world):
    run = prepare(pair_world, "combined", name="isolation")
    services = run.services
    model = services.build_route(ROUTE, seed=run.plan.seed)
    heads = services.build_heads(ROUTE, model)
    model.train()
    heads.train()
    child = next(iter(model.children()))
    child.eval()                                  # one child already in eval under a train parent
    assert model.training and not child.training
    flags = [(module, module.training) for root in (model, heads) for module in root.modules()]
    assert any(flag for _, flag in flags) and not all(flag for _, flag in flags)
    parameters = [value.clone() for value in model.state_dict().values()]
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state()

    def evaluate():
        return evaluation_lib.evaluate(
            services, ROUTE, model, heads, panel=run.loaded.panel, pool=run.pool,
            closure=run.loaded.closure, window_of=run.loaded.window_of,
            keys=run.loaded.training_keys, kappa=2.0, seed=run.plan.seed,
            diagnostics=run.plan.diagnostics, configured_distances={1: 1.0},
            realized=runner_lib.realized_training(EMPTY_TALLY))
    first = evaluate()
    assert all(module.training is flag for module, flag in flags)
    assert all(torch.equal(a, b) for a, b in zip(parameters, model.state_dict().values()))
    assert numpy_state_equal(numpy_before, np.random.get_state())
    assert torch.equal(torch_before, torch.get_rng_state())
    assert json.dumps(first, sort_keys=True) == json.dumps(evaluate(), sort_keys=True), \
        "the evaluation is a function of the model alone"
    assert all(parameter.grad is None for parameter in model.parameters())
    # A failure inside the evaluation still leaves the incoming modes in place -- the mixed ones
    # included -- and the global generators where they were, even after the failing call drew from
    # the torch generator before it raised.
    inside = {}

    def boom(*args, **kwargs):
        torch.rand(3)
        inside["torch"] = torch.get_rng_state()
        raise RuntimeError("evaluation failed")
    services.hr17_selection_terms = boom
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate()
    assert all(module.training is flag for module, flag in flags)
    assert model.training and not child.training
    assert not torch.equal(inside["torch"], torch_before), \
        "the failing call did advance the generator"
    assert torch.equal(torch_before, torch.get_rng_state())
    assert numpy_state_equal(numpy_before, np.random.get_state())


@pytest.mark.parametrize("scores, outcomes, reason", [
    ([1.0], [1.0], "fewer than two"), ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "scores are constant"),
    ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], "outcomes are constant")])
def test_an_undefined_spearman_is_null_with_its_reason(scores, outcomes, reason):
    record = evaluation_lib.spearman_record(scores, outcomes)
    assert record["value"] is None and reason in record["reason"] and record["rows"] == len(scores)


def test_a_defined_spearman_has_no_reason():
    record = evaluation_lib.spearman_record([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert record == {"value": pytest.approx(1.0), "reason": None, "rows": 3}
    with pytest.raises(ValueError):
        evaluation_lib.spearman_record([1.0, float("nan")], [1.0, 2.0])


# --- quartets --------------------------------------------------------------------------------------------------------

def test_quartet_containing_local_draws_pass_through_unchanged(quartet_world, monkeypatch):
    log, real = [], tasks_lib.dispatch

    def spy(services, route, model, heads, draw, **kwargs):
        log.append(draw)
        return real(services, route, model, heads, draw, **kwargs)
    monkeypatch.setattr(tasks_lib, "dispatch", spy)
    run = execute(prepare(quartet_world, "combined", name="quartets"))
    # With nothing replaced, all four tasks reach the real dispatch, and the local draws that do
    # are the base stream's own quartet-bearing draws.
    assert {draw.task for draw in log} == TASKS
    kept = [draw for draw in log if draw.task == data_lib.TASK_LOCAL]
    assert len(kept) == 2 and all(
        any(contrast.kind == data_lib.CONTRAST_COUPLED for contrast in draw.contrasts)
        for draw in kept)
    local = microbatches(run, data_lib.TASK_LOCAL)
    assert len(local) == 2
    for _, entry in local:
        assert "replacement" not in entry
        assert entry["kept_base_contrasts"][data_lib.CONTRAST_COUPLED] >= 1
        assert entry["rows"] == 4 and entry["term"] == "local_effect"
        assert entry["forwards"] > entry["rows"], "the quartet's sequential joint was scored"
    tally = run.record["tally"]
    assert tally["local_with_quartets"] == 2 and tally["local_replaced"] == 0
    assert tally["local_pair_eligible"] == 0 and tally["pair_endpoint_rows"] == 0
    # The other three tasks still ran under this arm's masks.
    assert len(microbatches(run, data_lib.TASK_WHOLE)) == 2


# --- refusals ----------------------------------------------------------------------------------------------------------

def test_only_the_native_150m_route_can_be_executed(pair_world):
    run = prepare(pair_world, "baseline", name="route")
    for route in ("dplm_650m_lora_masked", "ablang2_masked_editor"):
        with pytest.raises(ValueError, match="trains dplm_150m_masked only"):
            runner_lib.execute(run.services, route, plan=run.plan, stream=run.stream,
                               pool=run.pool, closure=run.loaded.closure,
                               window_of=run.loaded.window_of, keys=run.loaded.training_keys,
                               panel=run.loaded.panel, directory=run.directory,
                               identity=run.identity)
    assert list(run.directory.glob("*.pt")) == []
    routes = run.services.config["routes"]
    assert routes["enabled"] == [ROUTE] and "dplm_650m_lora_masked" not in routes
    with pytest.raises((KeyError, ValueError)):
        run.services.build_route("dplm_650m_lora_masked")


def test_a_pair_pool_that_lacks_a_weighted_distance_is_a_preflight_error(pair_world):
    run = prepare(pair_world, "combined", name="distances")
    narrow = pairs_lib.build_pair_pool(
        run.loaded.pool, closure=run.loaded.closure, window_of=run.loaded.window_of,
        training_population_keys=run.loaded.training_keys, distances=(1,), capacity=64, seed=1)
    built = len(pair_world.loads)
    with pytest.raises(ValueError, match="does not hold"):
        runner_lib.execute(run.services, ROUTE, plan=run.plan, stream=run.stream, pool=narrow,
                           closure=run.loaded.closure, window_of=run.loaded.window_of,
                           keys=run.loaded.training_keys, panel=run.loaded.panel,
                           directory=run.directory, identity=run.identity)
    assert len(pair_world.loads) == built, "no model was built before the preflight refused"
    assert list(run.directory.glob("*.pt")) == []
    assert not (run.directory / runner_lib.PROGRESS).exists()


def test_a_non_finite_loss_stops_the_run_and_leaves_no_snapshot(pair_world, monkeypatch):
    run = prepare(pair_world, "baseline", name="nonfinite")
    real = tasks_lib.dispatch

    def poisoned(services, route, model, heads, draw, **kwargs):
        outcome = real(services, route, model, heads, draw, **kwargs)
        outcome.loss = outcome.loss * float("nan")
        return outcome
    monkeypatch.setattr(tasks_lib, "dispatch", poisoned)
    with pytest.raises(ValueError, match="not finite"):
        execute(run)
    assert not (run.directory / runner_lib.SNAPSHOT).exists()


# --- the whole pilot -------------------------------------------------------------------------------------------------

def test_the_whole_pilot_runs_end_to_end_and_records_every_identity(tmp_path, monkeypatch):
    world = world_lib.build_world(tmp_path, monkeypatch)
    world.install_code()
    forbid_train(monkeypatch)
    pilot = world.pilot(world.write_pilot("short", lambda doc: doc.update(updates=2)))
    roots = world.reserve("whole")
    environment = world_lib.prepared_environment()
    present_when_built, real_build = [], services_mod.FlightServices.build_route

    def watched(self, *args, **kwargs):
        present_when_built.append((roots.run_root / "current_bins.json").is_file())
        return real_build(self, *args, **kwargs)
    monkeypatch.setattr(services_mod.FlightServices, "build_route", watched)
    record = runner_lib.run_pilot(pilot, roots, device="cpu", environment=environment)
    root = roots.run_root
    assert record["status"] == "completed" and record["updates"] == 2
    assert json.loads((root / "run.json").read_text(encoding="utf-8"))["snapshot"] == \
        record["snapshot"]
    assert not (root / "failure.json").exists()
    assert (root / "snapshot.pt").is_file() and (root / "evaluation.json").is_file()
    assert len((root / "progress.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    assert inputs["membership"]["split_sha256"] == world.policy["expected_split_sha256"]
    assert inputs["configured_routes"] == {"enabled": [ROUTE]}
    assert inputs["training_population_keys"] == 16 and inputs["development_panel_rows"] == 2
    assert inputs["source_identities"]["manifest_files_verified"] == 1
    assert inputs["pair_pool"]["distances"] == [1, 2, 3]
    identity = inputs["identity"]
    assert identity["pilot_sha256"] == pilot.sha256 and identity["arm"] == "combined"
    assert len(identity["native_sha256"]) == 64
    assert identity["split_sha256"] == world.policy["expected_split_sha256"]
    written = json.loads((root / "pilot_config.json").read_text(encoding="utf-8"))
    assert written["arm"] == "combined" and written["roots"]["run_root"] == str(root)
    assert sorted(path.name for path in root.glob("*.pt")) == ["snapshot.pt"]
    assert world.loads and all(load["variant"] == "dplm_150m" for load in world.loads)

    # The code identity is the hash of every file the entry script can execute, read from THIS code
    # root (a copy of the real closure), and it reaches the snapshot's identity.
    code = identity["code"]
    assert code["file_count"] == len(code["files"]) > 30
    assert code == runner_lib.code_identity(world.code)
    for name, digest in code["files"].items():
        assert digest == paths.sha256_file(world.code / name), name
    assert {"scripts/her2_multiscale_pilot.py",
            "src/smallAntibodyGen/experiments/her2_routes_services.py",
            "src/smallAntibodyGen/experiments/her2_routes_tasks.py",
            "src/smallAntibodyGen/models/dplm_policy.py"} <= set(code["files"])
    payload = torch.load(str(root / "snapshot.pt"), map_location="cpu", weights_only=True)
    assert payload["identity"]["code"]["closure_sha256"] == code["closure_sha256"]
    assert payload["identity"]["bin_policy"] == identity["bin_policy"]
    assert payload["identity"]["runtime"] == identity["runtime"]

    # The bin policy and what it froze are in the inputs and in the identity.
    assert identity["bin_policy"] == inputs["bin_policy"]["frozen_identity"]
    assert identity["bin_policy"]["policy"] == "current_exact_training"
    assert len(identity["bin_policy"]["thresholds_sha256"]) == 64
    assert inputs["bin_policy"]["historical_replay"] is False

    # The current bins as numbers: written before the model was built, bound to the identity (and
    # so to the snapshot's), and listed in the inputs report by run-relative path and file SHA-256.
    assert present_when_built and all(present_when_built)
    bins_file = root / "current_bins.json"
    saved = paths.read_json(bins_file)
    assert inputs["current_bins_file"] == {
        "path": "current_bins.json", "file_sha256": paths.sha256_file(bins_file),
        "content_sha256": paths.digest_document(saved)}
    bound = identity["bin_policy"]
    assert bound["current_bins_sha256"] == paths.digest_document(saved)
    assert bound["assignments_sha256"] == saved["assignments"]["sha256"]
    assert bound["assignments_count"] == saved["assignments"]["count"] == bound["bins_assigned"]
    assert bound["thresholds_sha256"] == paths.digest_document(saved["bin_thresholds"])
    assert bound["support_sha256"] == paths.digest_document(saved["bin_threshold_support"])
    assert saved["bin_thresholds"] and saved["policy"] == "current_exact_training"

    # The prepared environment is persisted, with the interpreter and the libraries actually
    # imported (versions and locations), and the record says what it does not verify.
    runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["environment"] == environment
    assert runtime["runtime"]["python"]["executable"] == sys.executable
    libraries = runtime["runtime"]["libraries"]
    assert {"torch", "numpy"} <= set(libraries)
    assert libraries["torch"]["version"] == importlib.metadata.version("torch")
    assert Path(libraries["numpy"]["location"]).is_dir()
    assert runtime["code"]["closure_sha256"] == code["closure_sha256"]
    assert "not byte-verified" in runtime["verification"]
    # runtime.json says WHEN it was captured; the completion capture is the one in run.json, and
    # the static code identity binds the installed versions of the libraries imported in between.
    stage = runtime["capture_stage"]
    assert "before the inputs, the model and the optimizer" in stage["runtime"]
    assert "libraries_imported_at_completion" in stage["completion"]
    assert "code.external_versions" in stage["late_imports"]
    assert "transformers" in code["external_versions"]
    assert identity["runtime"]["libraries"]["torch"] == libraries["torch"]["version"]
    assert identity["runtime"]["python"] == ".".join(str(part) for part in sys.version_info[:3])
    assert "location" not in json.dumps(identity["runtime"]), "the identity carries no paths"
    assert "transformers" in record["libraries_imported_at_completion"], \
        "Adafactor ran, so the optimizer's library was imported by the end"

    # An inherited source that changes changes the identity (the pilot package alone would not).
    changed = world.code / "src" / "smallAntibodyGen" / "experiments" / "her2_routes_tasks.py"
    changed.write_bytes(changed.read_bytes() + b"\n# a changed inherited source\n")
    after = runner_lib.code_identity(world.code)
    assert after["closure_sha256"] != code["closure_sha256"]
    differing = [name for name in code["files"] if after["files"][name] != code["files"][name]]
    assert differing == ["src/smallAntibodyGen/experiments/her2_routes_tasks.py"]


def test_the_code_identity_names_the_whole_executed_closure_not_only_the_pilot_package():
    code = runner_lib.code_identity(world_lib.REPO)
    names = set(code["files"])
    expected = {"config.py", "draws.py", "evaluation.py", "inputs.py", "masks.py", "membership.py",
                "objective.py", "pairs.py", "roots.py", "runner.py", "__init__.py"}
    package = {name.rsplit("/", 1)[1] for name in names if "her2_multiscale/" in name}
    assert package >= expected
    inherited = {"src/smallAntibodyGen/experiments/" + name for name in (
        "her2_routes_services.py", "her2_routes_tasks.py", "her2_routes_sources.py",
        "her2_routes_data.py", "her2_routes_conditions.py", "her2_routes_objectives.py",
        "her2_runtime.py", "her2_replay_spec.py")}
    inherited |= {"src/smallAntibodyGen/models/dplm_policy.py",
                  "src/smallAntibodyGen/models/her2_routes_heads.py"}
    assert inherited <= names, sorted(inherited - names)
    assert "scripts/her2_multiscale_pilot.py" in names and code["entry_points"] == [
        "scripts/her2_multiscale_pilot.py"]
    assert code["file_count"] == len(names) > 30
    assert all(len(digest) == 64 for digest in code["files"].values())
    assert code["closure_sha256"] == paths.sha256_text("\n".join(
        f"{name} {code['files'][name]}" for name in sorted(names)))
    assert {"torch", "numpy", "transformers"} <= set(code["external_versions"])
    assert code["external_versions"]["json"]["kind"] == "standard_library"
    assert "static" in code["scope"] and "not by bytes" in code["scope"]


def test_an_inherited_source_byte_change_changes_the_identity(tmp_path):
    root = tmp_path / "code"
    world_lib.install_code(root)
    original = runner_lib.code_identity(root)
    assert original == runner_lib.code_identity(root), "the identity is a function of the bytes"
    inherited = "src/smallAntibodyGen/experiments/her2_routes_sources.py"
    assert inherited in original["files"] and "her2_multiscale" not in inherited
    (root / inherited).write_bytes((root / inherited).read_bytes() + b"\n")
    changed = runner_lib.code_identity(root)
    assert changed["closure_sha256"] != original["closure_sha256"]
    assert [name for name in original["files"] if changed["files"][name] != original["files"][name]] \
        == [inherited]
    assert changed["file_count"] == original["file_count"]


def test_a_code_root_without_the_closure_is_refused_not_hashed_as_empty(tmp_path):
    with pytest.raises(ValueError, match="closure"):
        runner_lib.code_identity(tmp_path)                       # no files at all
    script_only = tmp_path / "scripts"
    script_only.mkdir()
    (script_only / "her2_multiscale_pilot.py").write_bytes(SCRIPT.read_bytes())
    with pytest.raises(ValueError, match="does not contain the pilot"):
        runner_lib.code_identity(tmp_path)                       # the script, none of what it runs


def test_a_world_without_real_code_refuses_to_run_and_opens_nothing(tmp_path, monkeypatch):
    world = world_lib.build_world(tmp_path, monkeypatch)
    roots = world.reserve("no-code")
    with pytest.raises(ValueError, match="closure"):
        runner_lib.run_pilot(world.pilot(), roots, device="cpu",
                             environment=world_lib.prepared_environment())
    root = roots.run_root
    assert (root / "failure.json").is_file() and world.loads == []
    for name in ("runtime.json", "pilot_config.json", "inputs.json", "run.json", "snapshot.pt",
                 "condition_vocabulary.json"):
        assert not (root / name).exists(), name


@pytest.mark.parametrize("environment, message", [
    (None, "prepare_environment returned"), ({}, "prepare_environment returned"),
    ({"imported_before_this_call": ["torch"], "declared_paths_absent": []},
     "were imported before the cache variables"),
    ({"imported_before_this_call": [], "declared_paths_absent": ["/missing"]},
     "do not exist")])
def test_a_missing_or_late_environment_record_is_refused_before_any_input(
        tmp_path, monkeypatch, environment, message):
    world = world_lib.build_world(tmp_path, monkeypatch)
    world.install_code()
    roots = world.reserve("environment")
    with pytest.raises(ValueError, match=message):
        runner_lib.run_pilot(world.pilot(), roots, device="cpu", environment=environment)
    assert world.loads == [] and not (roots.run_root / "runtime.json").exists()
    assert (roots.run_root / "failure.json").is_file()


def test_the_command_line_hands_the_prepared_environment_to_the_runner(tmp_path, monkeypatch):
    spec_ = importlib.util.spec_from_file_location("her2_multiscale_pilot_handoff", SCRIPT)
    script = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(script)
    from smallAntibodyGen.experiments import her2_routes_env as env
    layout = {name: tmp_path / name for name in ("data", "artifacts", "frozen", "runs")}
    for directory in layout.values():
        directory.mkdir()
    record = world_lib.prepared_environment()
    seen = {}
    monkeypatch.setattr(env, "prepare_environment", lambda *args, **kwargs: record)
    monkeypatch.setattr(runner_lib, "run_pilot", lambda pilot, roots, **kwargs: seen.update(
        kwargs=kwargs, roots=roots) or {"status": "completed", "arm": pilot.arm,
                                        "snapshot": {"path": "x"}})
    script.main(["--frozen-inputs", str(layout["frozen"]), "--data-repo-root", str(layout["data"]),
                 "--artifact-root", str(layout["artifacts"]), "--run-root",
                 str(layout["runs"] / "cli")])
    assert seen["kwargs"]["environment"] is record, "the environment record was discarded"
    assert seen["kwargs"]["device"] == "cuda" and seen["roots"].run_root == layout["runs"] / "cli"


def test_a_refused_input_is_recorded_as_a_failure_not_a_smaller_success(tmp_path, monkeypatch):
    world = world_lib.build_world(tmp_path, monkeypatch)
    world.install_code()
    document = paths.read_json(world.frozen / "condition_vocabulary.json")
    document["counts"][world_lib.ASSAY] += 1
    paths.write_json(world.frozen / "condition_vocabulary.json", document)
    roots = world.reserve("failing")
    with pytest.raises(ValueError, match="vocabulary changed"):
        runner_lib.run_pilot(world.pilot(), roots, device="cpu",
                             environment=world_lib.prepared_environment())
    root = roots.run_root
    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed" and failure["error_type"] == "ValueError"
    assert "vocabulary" in failure["message"] and failure["traceback"]
    for name in ("run.json", "snapshot.pt", "evaluation.json", "progress.jsonl"):
        assert not (root / name).exists()
    assert world.loads == [], "no model was built for a refused input"
