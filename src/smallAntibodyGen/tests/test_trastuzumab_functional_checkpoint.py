"""Schema-2 checkpoints with the functional arm: exact resume, data-free validation, no mutation
before every check has passed, and inference/export that never opens a measurement (M2)."""
from __future__ import annotations

import copy
import shutil
from types import SimpleNamespace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc import functional_data as fd
from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol as fp
from smallAntibodyGen.experiments.trastuzumab_poc import functional_train as ft
from smallAntibodyGen.experiments.trastuzumab_poc import inputs
from smallAntibodyGen.experiments.trastuzumab_poc import post_budget as pb
from smallAntibodyGen.experiments.trastuzumab_poc.functional_checkpoint import FunctionalRun
from smallAntibodyGen.experiments.trastuzumab_poc.parent import export_parent
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    MeasuredEndpoint, RestrictedPostData, SelectedBudget, select_budget,
)
from smallAntibodyGen.experiments.trastuzumab_poc.runner import (
    StageTrainer, TrainingConfig, deterministic_ordering, learning_rate,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_protocol import MIXED, ordered_pairs_per_update
from smallAntibodyGen.tests.test_trastuzumab_functional_runner import (
    MODES, SMALL_VALUES, build_trainer, find_seed, fresh_models, functional_run, functional_settings,
    restricted_data, small_world, standard_functional_world, training_config,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_transfer import endpoint_manifest, post_data
from smallAntibodyGen.tests.test_trastuzumab_runner import equal

SOURCE = {"source": "synthetic"}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory):
    """A cosine-schedule functional run stopped after two of four updates, with a checkpoint."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, _, population, qualified = small_world()
        base, encoder = fresh_models("lora")
        config = training_config(total_updates=4, schedule="cosine")

        def make(world="ordered", functional=True, **settings):
            run = functional_run(qualified, SMALL_VALUES[world], **settings) if functional else None
            return build_trainer(copy.deepcopy(base), encoder, population, config, functional=run)

        trained = make()
        trained.run(stop_after_updates=2)
        path = tmp_path_factory.mktemp("small") / "latest.pt"
        ck.save_checkpoint(trained, path)
        return SimpleNamespace(make=make, path=path, run=trained, encoder=encoder, base=base)
    finally:
        torch.set_num_threads(previous)


def snapshot(run):
    return {"policy": copy.deepcopy(run.policy.state_dict()), "optimizer": copy.deepcopy(run.optimizer.state_dict()),
            "progress": copy.deepcopy(run.progress()), "next_update": run.next_update, "failed": run.failed,
            "head": None if run.head is None else run.head.weight.detach().clone(),
            "head_optimizer": None if run.head is None else copy.deepcopy(run.head_optimizer.state_dict()),
            "rng": torch.random.get_rng_state().clone()}


def assert_untouched(run, before):
    after = snapshot(run)
    assert torch.equal(before.pop("rng"), after.pop("rng"))
    equal(before, after)


def tamper(path, mutate):
    payload = torch.load(path, weights_only=True)
    mutate(payload)
    torch.save(payload, path)


def head_state(payload):
    return payload["functional"]["optimizer"]["state"][0]


def head_group(payload):
    return payload["functional"]["optimizer"]["param_groups"][0]


def head_weights(payload):
    return payload["functional"]["weights"]


def counters_of(payload):
    return payload["progress"]["functional"]["counters"]


def seen_of(payload):
    return payload["progress"]["functional"]["seen"]


def put(container, key, value):
    container[key] = value


CORRUPTIONS = {
    "moment_nan": lambda p: head_state(p)["exp_avg"].fill_(float("nan")),
    "second_moment_nan": lambda p: head_state(p)["exp_avg_sq"].fill_(float("nan")),
    "second_moment_negative": lambda p: head_state(p)["exp_avg_sq"].fill_(-1.0),
    "moment_dtype": lambda p: put(head_state(p), "exp_avg", head_state(p)["exp_avg"].double()),
    "moment_shape": lambda p: put(head_state(p), "exp_avg", head_state(p)["exp_avg"].flatten()),
    "step_count": lambda p: put(head_state(p), "step", head_state(p)["step"] + 1),
    "step_dtype": lambda p: put(head_state(p), "step", head_state(p)["step"].double()),
    "extra_state_key": lambda p: put(head_state(p), "extra", torch.zeros(1)),
    "missing_state_key": lambda p: head_state(p).pop("exp_avg"),
    "extra_state_entry": lambda p: put(p["functional"]["optimizer"]["state"], 1, copy.deepcopy(head_state(p))),
    "group_parameter_order": lambda p: put(head_group(p), "params", [1]),
    "extra_group": lambda p: p["functional"]["optimizer"]["param_groups"].append(copy.deepcopy(head_group(p))),
    "group_rate": lambda p: put(head_group(p), "lr", head_group(p)["lr"] * 2),
    "group_option": lambda p: put(head_group(p), "weight_decay", 0.5),
    "weights_nan": lambda p: head_weights(p)["weight"].fill_(float("nan")),
    "weights_dtype": lambda p: put(head_weights(p), "weight", head_weights(p)["weight"].double()),
    "weights_shape": lambda p: put(head_weights(p), "weight", head_weights(p)["weight"].flatten()),
    "weights_name": lambda p: put(head_weights(p), "w", head_weights(p).pop("weight")),
    "extra_payload_key": lambda p: put(p["functional"], "extra", 1),
    "missing_head": lambda p: put(p, "functional", None),
    "bool_counter": lambda p: put(counters_of(p), "tie_pairs", False),
    "opportunities_forged": lambda p: put(counters_of(p), "pair_opportunities", counters_of(p)["pair_opportunities"] + 1),
    "ordered_pairs_forged": lambda p: put(counters_of(p), "ordered_pairs", counters_of(p)["ordered_pairs"] + 1),
    "ordered_updates_forged": lambda p: put(counters_of(p), "ordered_updates", 99),
    "forwards_forged": lambda p: put(counters_of(p), "feature_forwards", counters_of(p)["feature_forwards"] + 2),
    "seen_unsorted": lambda p: seen_of(p)["compared"].reverse(),
    "seen_duplicate": lambda p: seen_of(p)["compared"].append(seen_of(p)["compared"][0]),
    "seen_outside_budget": lambda p: seen_of(p)["ordered"].append(["g-x", "zzz"]),
    "seen_missing": lambda p: seen_of(p)["compared"].pop(),
    "budget_order": lambda p: p["identity"]["input"]["functional"]["budget"]["selected"].reverse(),
    "budget_seed": lambda p: put(p["identity"]["input"]["functional"]["budget"], "budget_seed", "other"),
    "qualified_digest": lambda p: put(p["identity"]["functional"], "qualified_digest", "0" * 64),
    "head_width": lambda p: put(p["identity"]["functional"]["head"], "hidden_size", 8),
    "config_edit": lambda p: put(p["identity"]["functional"]["config"], "weight", 3.0),
    "input_record_null": lambda p: put(p["identity"]["input"], "functional", None),
    "progress_record_null": lambda p: put(p["progress"], "functional", None),
}
# A valid config with a different weight cannot be told apart from an honest one without the
# trainer's own identity, so the data-free (inference) reader cannot reject it; resume does.
NOT_DATA_FREE = {"config_edit"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_schema_two_records_the_arm_explicitly_with_and_without_it(tmp_path, mode):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models(mode)
    config = training_config(total_updates=2)
    plain = build_trainer(copy.deepcopy(base), encoder, population, config)
    plain.step()
    ck.save_checkpoint(plain, tmp_path / "plain.pt")
    payload = torch.load(tmp_path / "plain.pt", weights_only=True)
    assert payload["schema"] == "trastuzumab-checkpoint/2" and payload["functional"] is None
    assert payload["identity"]["functional"] is None and payload["progress"]["functional"] is None
    assert payload["identity"]["input"]["functional"] is None and payload["identity"]["input"]["budget"] is None

    functional = build_trainer(copy.deepcopy(base), encoder, population, config,
                               functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    functional.step()
    ck.save_checkpoint(functional, tmp_path / "functional.pt")
    payload = torch.load(tmp_path / "functional.pt", weights_only=True)
    assert set(payload["functional"]) == {"weights", "optimizer"}
    assert set(head_weights(payload)) == {"weight"}
    weight = head_weights(payload)["weight"]
    assert weight.shape == (1, 16) and weight.dtype == torch.float32
    assert torch.equal(weight, functional.head.weight.detach())
    assert set(payload["functional"]["optimizer"]["state"]) == {0}
    assert float(head_state(payload)["step"]) == 1.0 and head_state(payload)["exp_avg"].shape == (1, 16)
    # The head is outside the policy: its weights and moments are separate entries.
    assert set(payload["weights"]) == {name for name, _ in functional.parameters}
    assert set(payload["optimizer"]["state"]) == set(range(len(functional.parameters)))
    assert payload["progress"]["functional"]["counters"]["ordered_updates"] == 1


@pytest.mark.parametrize("change", ["checkpoint_schema", "training_schema", "missing_functional"])
def test_old_schema_checkpoints_are_rejected_rather_than_migrated(tmp_path, small_checkpoint, change):
    def mutate(payload):
        if change == "checkpoint_schema":
            payload["schema"] = "trastuzumab-checkpoint/1"
        elif change == "training_schema":
            payload["identity"]["schema"] = "trastuzumab-training/1"
        else:
            del payload["functional"]

    path = tmp_path / "old.pt"
    shutil.copyfile(small_checkpoint.path, path)
    tamper(path, mutate)
    fresh = small_checkpoint.make()
    before = snapshot(fresh)
    with pytest.raises(ValueError):
        ck.resume_checkpoint(fresh, path)
    assert_untouched(fresh, before)
    policy, encoder = fresh_models("lora")
    with pytest.raises(ValueError):
        ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                     source_identity=SOURCE)


# ---------------------------------------------------------------------------
# Exact resume
# ---------------------------------------------------------------------------

def assert_same_run(full, resumed):
    equal(full.policy.state_dict(), resumed.policy.state_dict())
    equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    equal(full.head.weight.detach(), resumed.head.weight.detach())
    equal(full.head_optimizer.state_dict(), resumed.head_optimizer.state_dict())
    assert full.progress() == resumed.progress()


@pytest.mark.parametrize("mode", MODES)
def test_a_pristine_zero_update_checkpoint_round_trips_and_continues_identically(tmp_path, mode):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models(mode)
    config = training_config(schedule="cosine")

    def make():
        return build_trainer(copy.deepcopy(base), encoder, population, config,
                             functional=functional_run(qualified, SMALL_VALUES["ordered"], head_lr_scale=0.5))

    pristine = make()
    path = tmp_path / "zero.pt"
    ck.save_checkpoint(pristine, path)
    payload = torch.load(path, weights_only=True)
    assert payload["progress"]["next_update"] == 0 and payload["optimizer"]["state"] == {}
    assert payload["functional"]["optimizer"]["state"] == {}
    assert counters_of(payload)["ordered_pairs"] == 0 and seen_of(payload) == {"compared": [], "ordered": []}
    assert torch.equal(head_weights(payload)["weight"], pristine.head.weight.detach())
    assert head_group(payload)["lr"] == learning_rate(config, 0) * 0.5
    resumed, full = make(), make()
    ck.resume_checkpoint(resumed, path)
    resumed.run()
    full.run()
    assert_same_run(full, resumed)
    policy, encoder = fresh_models(mode)
    progress = ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                            source_identity=SOURCE)["progress"]
    assert progress["next_update"] == 0 and progress["functional"]["counters"]["pair_opportunities"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_cosine_resume_across_ordered_and_unresolved_updates_is_exact(tmp_path, mode):
    population, qualified = standard_functional_world()
    data = restricted_data(qualified, MIXED)
    pairs, total, stop, scale = 2, 6, 3, 0.5

    def both_segments_mix(seed):
        per_update = ordered_pairs_per_update(data, data.budget, seed, pairs, total)
        return all(any(part) and not all(part) for part in (per_update[:stop], per_update[stop:]))

    seed = find_seed(both_segments_mix)
    base, encoder = fresh_models(mode)
    config = training_config(seed=seed, total_updates=total, schedule="cosine")

    def make():
        return build_trainer(copy.deepcopy(base), encoder, population, config, functional=FunctionalRun(
            functional_settings(pairs_per_update=pairs, head_lr_scale=scale), qualified, data))

    full = make()
    full.run()
    partial = make()
    partial.run(stop_after_updates=stop)
    path = tmp_path / "latest.pt"
    ck.save_checkpoint(partial, path)
    saved = torch.load(path, weights_only=True)
    steps = counters_of(saved)["ordered_updates"]
    assert 0 < steps <= stop and float(head_state(saved)["step"]) == steps
    # The group rate tracks the last GLOBAL update, whether or not that update stepped the head.
    assert head_group(saved)["lr"] == learning_rate(config, stop - 1) * scale
    resumed = make()
    before = torch.random.get_rng_state().clone()
    ck.resume_checkpoint(resumed, path)
    assert torch.equal(before, torch.random.get_rng_state())
    assert resumed.head_optimizer.param_groups[0]["lr"] == learning_rate(config, stop - 1) * scale
    assert resumed.progress()["functional"] == partial.progress()["functional"]
    resumed.run()
    assert_same_run(full, resumed)
    state = next(iter(resumed.head_optimizer.state.values()))
    assert float(state["step"]) == resumed.progress()["functional"]["counters"]["ordered_updates"]
    ck.save_checkpoint(resumed, path)
    final = make()
    ck.resume_checkpoint(final, path)
    assert final.progress() == full.progress() and final.next_update == total


def test_an_inactive_hinge_after_an_earlier_step_still_advances_momentum_and_decay():
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = build_trainer(base, encoder, population, training_config(total_updates=2),
                        functional=functional_run(qualified, SMALL_VALUES["ordered"], gradient_mode="detached",
                                                  pairs_per_update=1, head_weight_decay=0.1,
                                                  head_gradient_clip=1000.0))
    first = run.step()["functional"]
    assert first["head_stepped"] and first["active_pairs"] == 1
    state = run.head_optimizer.state[run.head.weight]
    moment, second, steps = state["exp_avg"].clone(), state["exp_avg_sq"].clone(), float(state["step"])
    assert bool(moment.abs().sum() > 0)
    # Force a gap of 1000 against a margin of 100: the hinge is exactly zero, the gradient is zero.
    with deterministic_ordering(run.policy):
        better, worse = (ft.endpoint_features(run.policy, run.encoder, qualified.construct(name), "paired",
                                              grad=False) for name in ("a", "b"))
    delta = (better - worse).detach()
    with torch.no_grad():
        run.head.weight.copy_((1000.0 * delta / delta.dot(delta)).unsqueeze(0))
    forced = run.head.weight.detach().clone()
    stats = run.step()["functional"]
    assert stats["head_stepped"] and stats["ordered_pairs"] == 1 and stats["active_pairs"] == 0
    assert stats["loss"] == 0.0
    state = run.head_optimizer.state[run.head.weight]
    assert float(state["step"]) == steps + 1 == run.progress()["functional"]["counters"]["ordered_updates"]
    torch.testing.assert_close(state["exp_avg"], 0.9 * moment, rtol=1e-5, atol=0)
    torch.testing.assert_close(state["exp_avg_sq"], 0.999 * second, rtol=1e-5, atol=0)
    assert not torch.equal(run.head.weight.detach(), forced)
    assert run.progress()["functional"]["counters"]["active_pairs"] == 1


def test_an_all_unresolved_run_checkpoints_an_untouched_head_and_an_empty_head_state(tmp_path):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    config = training_config(total_updates=3, schedule="cosine")

    def make():
        return build_trainer(copy.deepcopy(base), encoder, population, config,
                             functional=functional_run(qualified, SMALL_VALUES["unresolved"]))

    run = make()
    run.run(stop_after_updates=2)
    path = tmp_path / "unresolved.pt"
    ck.save_checkpoint(run, path)
    payload = torch.load(path, weights_only=True)
    assert payload["functional"]["optimizer"]["state"] == {} and counters_of(payload)["ordered_updates"] == 0
    assert torch.equal(head_weights(payload)["weight"], ft.new_head(16, config.seed).weight.detach())
    assert head_group(payload)["lr"] == learning_rate(config, 1)         # assigned although it never stepped
    resumed = make()
    ck.resume_checkpoint(resumed, path)
    resumed.run()
    run.run()
    assert_same_run(run, resumed)
    # A head that changed although no head step happened is a lie the resume can detect ...
    tamper(path, lambda p: head_weights(p)["weight"].add_(1.0))
    fresh = make()
    before = snapshot(fresh)
    with pytest.raises(ValueError, match="no head step"):
        ck.resume_checkpoint(fresh, path)
    assert_untouched(fresh, before)
    # ... while the data-free reader, which drops the head anyway, does not reproduce the discarded
    # initialization (it is reconstructible from the seed and algorithm, but this reader never does).
    policy, encoder = fresh_models("lora")
    ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                 source_identity=SOURCE)


# ---------------------------------------------------------------------------
# Rejection before any live mutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corrupted_head_state_or_functional_records_are_rejected_before_any_live_mutation(
        tmp_path, small_checkpoint, name):
    path = tmp_path / "corrupt.pt"
    shutil.copyfile(small_checkpoint.path, path)
    tamper(path, CORRUPTIONS[name])
    fresh = small_checkpoint.make()
    before = snapshot(fresh)
    with pytest.raises((ValueError, TypeError)):
        ck.resume_checkpoint(fresh, path)
    assert_untouched(fresh, before)
    assert fresh.next_update == 0 and not fresh.failed and not fresh.head_optimizer.state


@pytest.mark.parametrize("name", sorted(set(CORRUPTIONS) - NOT_DATA_FREE))
def test_the_data_free_reader_rejects_the_same_corruptions_without_any_trainer(tmp_path, small_checkpoint, name):
    path = tmp_path / "corrupt.pt"
    shutil.copyfile(small_checkpoint.path, path)
    tamper(path, CORRUPTIONS[name])
    policy, encoder = fresh_models("lora")
    before = copy.deepcopy(policy.state_dict())
    with pytest.raises((ValueError, TypeError)):
        ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                     source_identity=SOURCE)
    equal(before, policy.state_dict())


def test_resume_refuses_a_different_run_before_touching_anything(tmp_path, small_checkpoint):
    variants = {
        "other_measurements": lambda: small_checkpoint.make(world="negated"),
        "other_config": lambda: small_checkpoint.make(weight=3.0),
        "other_detachment": lambda: small_checkpoint.make(gradient_mode="detached"),
        "no_functional_arm": lambda: small_checkpoint.make(functional=False),
    }
    for name, factory in variants.items():
        fresh = factory()
        before = snapshot(fresh)
        with pytest.raises(ValueError, match="identity"):
            ck.resume_checkpoint(fresh, small_checkpoint.path)
        assert_untouched(fresh, before)
    plain = small_checkpoint.make(functional=False)
    plain.step()
    plain_path = tmp_path / "plain.pt"
    ck.save_checkpoint(plain, plain_path)
    functional = small_checkpoint.make()
    with pytest.raises(ValueError, match="identity"):
        ck.resume_checkpoint(functional, plain_path)
    assert functional.next_update == 0


def test_resume_needs_a_pristine_trainer_and_a_hardware_failure_poisons_it(tmp_path, small_checkpoint, monkeypatch):
    used = small_checkpoint.make()
    used.step()
    with pytest.raises(ValueError, match="pristine"):
        ck.resume_checkpoint(used, small_checkpoint.path)
    fresh = small_checkpoint.make()

    def broken(state):
        raise OSError("injected copy failure")

    monkeypatch.setattr(fresh.policy, "load_trainable_state", broken)
    with pytest.raises(OSError, match="injected copy"):
        ck.resume_checkpoint(fresh, small_checkpoint.path)
    assert fresh.failed
    with pytest.raises(RuntimeError, match="failed trainer"):
        fresh.step()


def test_a_trainer_with_inconsistent_functional_counters_cannot_save(tmp_path, small_checkpoint):
    run = small_checkpoint.make()
    run.step()
    run.functional_counters["ordered_pairs"] += 1
    path = tmp_path / "never.pt"
    with pytest.raises(ValueError):
        ck.save_checkpoint(run, path)
    assert not path.exists()
    run.functional_counters["ordered_pairs"] -= 1
    run.functional_seen["ordered"].add(("g-x", "zzz"))
    with pytest.raises(ValueError, match="outside the fixed budget"):
        ck.save_checkpoint(run, path)
    assert not path.exists()


# ---------------------------------------------------------------------------
# The canonical post fitting budget
# ---------------------------------------------------------------------------

def post_run(*, manifest=None, total=2, **changes):
    base, encoder = fresh_models("lora")
    config = TrainingConfig(**{"stage": "post", "arm": "paired", "seed": 7, "total_updates": total,
                               "examples_per_update": 1, "learning_rate": 0.01, "ordering_weight": 1.0,
                               "minimum_ordering_endpoint_coverage": 0.0, **changes})
    return StageTrainer(base, encoder, config, post_data=post_data(manifest or endpoint_manifest()),
                        data_identity=SOURCE, source_identity=SOURCE,
                        initialization_identity={"kind": "release", "base": base.base_identity})


def test_the_post_budget_is_recorded_from_the_validated_data_in_canonical_order():
    run = post_run()
    budget = run.post_data.budget
    recorded = run.identity["input"]["budget"]
    assert recorded == pb.post_budget_identity(budget)
    assert set(recorded) == {"schema", "manifest_digest", "budget_seed", "budget_n", "selected_ids", "selection_digest"}
    assert recorded["selected_ids"] == list(budget.selected_ids) and recorded["budget_n"] == 3
    assert recorded["manifest_digest"] == budget.manifest_digest and run.identity["input"]["functional"] is None
    # The selection digest hashes the SORTED IDs; the canonical selection order is recorded separately.
    assert pb.validate_post_budget_identity(recorded) == tuple(budget.selected_ids)
    with pytest.raises(ValueError, match="derived budget and functional"):
        StageTrainer(run.policy, run.encoder, run.config, post_data=run.post_data,
                     data_identity={"budget": {"claimed": True}}, source_identity=SOURCE,
                     initialization_identity={"kind": "release", "base": run.policy.base_identity})


def test_the_declared_order_matches_the_protocols_hash_selection_order():
    manifest = endpoint_manifest()
    ids = [entry.endpoint_id for entry in manifest.entries]
    for seed in ("shared", "another", "third"):
        chosen = select_budget(manifest, 3, seed)
        assert list(chosen.selected_ids) == sorted(ids, key=lambda i: pb.selection_key(seed, i))
        assert pb.validate_post_budget_identity(pb.post_budget_identity(chosen)) == tuple(chosen.selected_ids)


POST_BUDGET_TAMPER = {
    "order": lambda b: b["selected_ids"].reverse(),
    "count": lambda b: put(b, "budget_n", 2),
    "dropped": lambda b: b["selected_ids"].pop(),
    "duplicate": lambda b: put(b, "selected_ids", [b["selected_ids"][0]] * 3),
    "seed": lambda b: put(b, "budget_seed", "another"),
    "digest": lambda b: put(b, "selection_digest", "0" * 64),
    "manifest": lambda b: put(b, "manifest_digest", "0" * 64),
    "schema": lambda b: put(b, "schema", "trastuzumab-post-budget/0"),
    "extra_key": lambda b: put(b, "extra", 1),
    "boolean_size": lambda b: put(b, "budget_n", True),
}


@pytest.mark.parametrize("name", sorted(POST_BUDGET_TAMPER))
def test_a_tampered_post_budget_record_is_rejected_by_the_data_free_validator_and_by_resume(tmp_path, name):
    run = post_run()
    run.step()
    path = tmp_path / "post.pt"
    ck.save_checkpoint(run, path)
    clean = copy.deepcopy(run.identity["input"]["budget"])
    broken = copy.deepcopy(clean)
    POST_BUDGET_TAMPER[name](broken)
    with pytest.raises(ValueError):
        pb.validate_post_budget_identity(broken)
    tamper(path, lambda payload: payload["identity"]["input"].update(budget=broken))
    fresh = post_run()
    before = snapshot(fresh)
    with pytest.raises(ValueError):
        ck.resume_checkpoint(fresh, path)
    assert_untouched(fresh, before)
    policy, encoder = fresh_models("lora")
    with pytest.raises(ValueError):
        ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                     source_identity=SOURCE)


def test_data_free_validation_cannot_prove_a_prefix_but_the_restricted_data_does():
    manifest = endpoint_manifest()
    real = select_budget(manifest, 3, "shared")
    skipped = list(real.selected_ids[1:3])
    digest = pb._selection_digest(real.manifest_digest, "shared", skipped)
    forged = {"schema": pb.POST_BUDGET_SCHEMA, "manifest_digest": real.manifest_digest, "budget_seed": "shared",
              "budget_n": 2, "selected_ids": skipped, "selection_digest": digest}
    # Self-consistent, so no check without the manifest can see what was left out ...
    assert pb.validate_post_budget_identity(forged) == tuple(skipped)
    assert tuple(skipped) != select_budget(manifest, 2, "shared").selected_ids
    # ... which is exactly what the init-only manifest of RestrictedPostData establishes.
    budget = SelectedBudget(2, "shared", tuple(skipped), digest, real.group_manifest, real.manifest_digest)
    labels = tuple(MeasuredEndpoint(i, float(n), "higher_better", "assay") for n, i in enumerate(skipped))
    with pytest.raises(ValueError, match="budget does not match manifest"):
        RestrictedPostData(manifest=manifest, budget=budget, measured=labels, fitting_count=2,
                           development_labels=(), selection_allowance=post_data(manifest).selection_allowance)


def test_a_completed_post_run_need_not_have_observed_every_available_endpoint(tmp_path):
    run = post_run(total=1, ordering_weight=0.0)
    run.run()
    seen = set(run.progress()["seen"]["endpoints"])
    assert len(seen) == 1 and seen < set(run.post_data.budget.selected_ids)       # one of three: still complete
    path = tmp_path / "post.pt"
    ck.save_checkpoint(run, path)
    resumed = post_run(total=1, ordering_weight=0.0)
    ck.resume_checkpoint(resumed, path)
    assert resumed.progress() == run.progress() and resumed.next_update == 1

    def see_an_unknown_endpoint(payload):
        progress = payload["progress"]
        for key in ("endpoints", "reconstruction_endpoints"):       # counts stay consistent; only the subset breaks
            progress["seen"][key] = ["zzz"]
            progress["unique"][key] = 1

    tamper(path, see_an_unknown_endpoint)
    fresh = post_run(total=1, ordering_weight=0.0)
    with pytest.raises(ValueError, match="endpoint exposure"):
        ck.resume_checkpoint(fresh, path)
    policy, encoder = fresh_models("lora")
    with pytest.raises(ValueError, match="endpoint exposure"):
        ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                     source_identity=SOURCE)


def test_a_midtraining_checkpoint_records_no_post_budget(tmp_path, small_checkpoint):
    path = tmp_path / "mid.pt"
    shutil.copyfile(small_checkpoint.path, path)
    claimed = pb.post_budget_identity(post_data(endpoint_manifest()).budget)
    tamper(path, lambda p: p["identity"]["input"].update(budget=claimed))
    fresh = small_checkpoint.make()
    before = snapshot(fresh)
    with pytest.raises(ValueError):
        ck.resume_checkpoint(fresh, path)
    assert_untouched(fresh, before)
    policy, encoder = fresh_models("lora")
    with pytest.raises(ValueError, match="no post fitting budget"):
        ck.load_inference_checkpoint(policy, path, encoder=encoder, expected_sha256=ck.file_digest(path),
                                     source_identity=SOURCE)


# ---------------------------------------------------------------------------
# Inference and export never open measurements
# ---------------------------------------------------------------------------

def test_inference_and_export_never_touch_the_label_loaders(tmp_path, small_checkpoint, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inference or export consulted a measurement")

    for owner, name in ((fp.RestrictedFunctionalData, "classify"), (fp.RestrictedFunctionalData, "replay_outcomes"),
                        (fp.RestrictedFunctionalData, "from_document"), (fp, "load_restricted_measurements"),
                        (fp, "read_strict_json"), (fd, "read_strict_json"), (fd, "load_functional_constructs"),
                        (inputs, "load_functional_measurements"), (inputs, "load_functional_constructs"),
                        (inputs, "load_restricted_measurements"), (inputs, "qualify_functional_inputs")):
        monkeypatch.setattr(owner, name, forbidden)
    policy, encoder = fresh_models("lora")
    provenance = ck.load_inference_checkpoint(policy, small_checkpoint.path, encoder=encoder,
                                              expected_sha256=ck.file_digest(small_checkpoint.path),
                                              source_identity=SOURCE)
    assert set(provenance) == {"checkpoint_sha256", "training", "progress"}
    assert provenance["progress"]["functional"]["counters"]["ordered_updates"] == 2
    trained = small_checkpoint.run.policy.trainable_state()
    for name, tensor in policy.trainable_state().items():
        assert torch.equal(tensor, trained[name]), name
    result = export_parent(policy, tmp_path / "parent.pt", provenance=provenance)
    assert len(result["sha256"]) == 64
    # The head and both optimizer states are not exported; the provenance of the supervision is.
    parent = torch.load(tmp_path / "parent.pt", weights_only=True)
    assert "functional" not in parent and set(parent["provenance"]) == {"checkpoint_sha256", "training", "progress"}
    assert parent["provenance"]["training"]["functional"]["head"]["parent_behavior"].startswith("discarded")
