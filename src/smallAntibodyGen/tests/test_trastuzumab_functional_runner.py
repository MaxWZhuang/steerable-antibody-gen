"""The production StageTrainer with the functional arm (M2).

Every test runs the real trainer on a tiny FusedDPLM with nonzero native and adapter dropout,
activation checkpointing and an active gradient clip (``test_trastuzumab_ordering_dropout.models``).
The fixtures at the top are shared with the checkpoint, CLI and transfer test modules.
"""
from __future__ import annotations

import copy

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc import functional_data as fd
from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol as fp
from smallAntibodyGen.experiments.trastuzumab_poc import functional_train as ft
from smallAntibodyGen.experiments.trastuzumab_poc.data import _digest_manifest, qualify_manifest
from smallAntibodyGen.experiments.trastuzumab_poc.functional_checkpoint import FunctionalRun
from smallAntibodyGen.experiments.trastuzumab_poc.runner import (
    StageTrainer, TrainingConfig, deterministic_ordering, learning_rate,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_data import (
    KD, broad_records, construct_row, contract_block, declaration, group_row, heavy, qualify,
    standard_world,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_protocol import (
    MIXED, ORDERED, _forged, interval, labels_document, ordered_pairs_per_update, point,
)
from smallAntibodyGen.tests.test_trastuzumab_ordering_dropout import models
from smallAntibodyGen.tests.test_trastuzumab_runner import equal

MODES = ["lora", "full"]
BUDGET_SEED = "budget-seed-1"


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def small_document(decl):
    """Two train constructs in ONE KD group, so every scheduled pair is that pair (a is
    constructed to share its content with broad row b0: consistent train/train overlap)."""
    return {
        "schema": fd.CONSTRUCTS_SCHEMA, "pairs_declaration_digest": _digest_manifest(decl),
        "study_contract": contract_block(),
        "sources": [{"source_id": "func", "description": "functional measurement source"}],
        "constructs": [construct_row("a", heavy(0)), construct_row("b", heavy(1)),
                       construct_row("v0", heavy(6), family="famV", partition="val"),
                       construct_row("v1", heavy(7), family="famV", partition="val")],
        "groups": [group_row("g-x", "ctx-a", KD, ["a", "b"]),
                   group_row("g-val", "ctx-a", KD, ["v0", "v1"])]}


def small_world():
    decl, records = declaration(), broad_records()
    population = qualify_manifest(decl, records)
    return decl, records, population, qualify(small_document(decl), decl, records, population=population)


def standard_functional_world():
    decl, records, document = standard_world()
    population = qualify_manifest(decl, records)
    return population, qualify(document, decl, records, population=population)


# KD is lower_better: in "ordered" a beats b, in "negated" b beats a.
SMALL_VALUES = {
    "ordered": {("g-x", "a"): point(1.0), ("g-x", "b"): point(5.0)},
    "negated": {("g-x", "a"): point(5.0), ("g-x", "b"): point(1.0)},
    "tied": {("g-x", "a"): point(1.0), ("g-x", "b"): point(1.0)},
    "unresolved": {("g-x", "a"): interval(0.0, 10.0), ("g-x", "b"): interval(0.0, 10.0)},
}


def restricted_data(qualified, values, *, n=None, seed=BUDGET_SEED):
    budget = fp.select_functional_budget(qualified, n or len(qualified.train_observations()), seed)
    return fp.RestrictedFunctionalData.from_document(labels_document(budget, values), qualified, budget)


def functional_settings(**changes):
    # margin 100 keeps every ordered hinge active, so the head always has a gradient.
    return ft.FunctionalConfig(**{"weight": 2.0, "pairs_per_update": 2, "margin": 100.0,
                                  "minimum_compared_fraction": 0.0, **changes})


def functional_run(qualified, values, *, n=None, seed=BUDGET_SEED, **settings):
    return FunctionalRun(functional_settings(**settings), qualified,
                         restricted_data(qualified, values, n=n, seed=seed))


def training_config(**changes):
    # A clip this small is active on every update (asserted from the reported pre-clip norm).
    return TrainingConfig(**{"stage": "mid", "arm": "paired", "seed": 11, "total_updates": 3,
                             "examples_per_update": 2, "learning_rate": 0.01, "gradient_clip": 1e-3,
                             **changes})


def fresh_models(mode):
    """The ordering-dropout model (dropout 0.4 native, 0.3 adapter, checkpointing) as a test-only base."""
    policy, encoder = models(mode)
    policy.base_identity = {**policy.base_identity, "variant": "synthetic-test-only"}
    return policy, encoder


def build_trainer(policy, encoder, population, config, *, functional=None, certificate=None,
                  data_identity=None):
    return StageTrainer(policy, encoder, config, population=population, functional=functional,
                        data_identity={"input": "synthetic"} if data_identity is None else data_identity,
                        source_identity={"source": "synthetic"},
                        initialization_identity={"kind": "release", "base": policy.base_identity},
                        transfer_certificate=certificate)


def find_seed(predicate, limit=600):
    """A training seed realizing a scenario; the schedule and the labels are pure functions of it."""
    for seed in range(limit):
        if predicate(seed):
            return seed
    raise AssertionError("no seed in range realizes the scenario")


def capture_gradients(run):
    """Policy gradients as the optimizer sees them (after accumulation and clipping)."""
    captured = {}
    original = run.optimizer.step

    def step(*args, **kwargs):
        captured.update({name: p.grad.detach().clone() for name, p in run.parameters})
        return original(*args, **kwargs)

    run.optimizer.step = step
    return captured


def capture_head_gradients(run):
    captured = []
    original = run.head_optimizer.step

    def step(*args, **kwargs):
        captured.append(run.head.weight.grad.detach().clone())
        return original(*args, **kwargs)

    run.head_optimizer.step = step
    return captured


def keys_of(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_of(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from keys_of(item)


LEGACY_METRICS = {"update", "learning_rate", "gradient_norm", "reconstruction_nll", "ordering_loss",
                  "examples", "masked_residues", "teacher_forced_states", "pair_draws",
                  "informative_pairs", "ties", "antigen_examples"}


# ---------------------------------------------------------------------------
# Records and admission
# ---------------------------------------------------------------------------

def test_a_run_without_the_arm_records_explicit_nulls_and_the_legacy_metrics():
    _, _, population, _ = small_world()
    base, encoder = fresh_models("lora")
    run = build_trainer(base, encoder, population, training_config(total_updates=1))
    assert run.identity["schema"] == "trastuzumab-training/2"
    assert run.identity["functional"] is None and run.identity["input"]["functional"] is None
    assert run.identity["input"]["budget"] is None and run.identity["input"]["input"] == "synthetic"
    assert run.head is None and run.head_optimizer is None and run.functional_plan is None
    assert run.progress()["functional"] is None
    assert set(run.step()) == LEGACY_METRICS


@pytest.mark.parametrize("mode", MODES)
def test_functional_records_are_derived_from_the_data_and_hold_no_measurement_values(mode):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models(mode)
    run = build_trainer(base, encoder, population, training_config(),
                        functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    data, record = run.functional.data, run.identity["functional"]
    assert run.identity["input"]["functional"]["budget"] == fp.functional_budget_identity(data.budget)
    assert run.identity["input"]["functional"]["available"] == {
        "qualified_train_observations": 2, "qualified_train_physical_constructs": 2,
        "selected_observations": 2, "selected_construct_ids": 2, "selected_physical_constructs": 2}
    assert record["selection_digest"] == data.budget.selection_digest
    assert (record["qualified_digest"], record["contract_digest"]) == (qualified.digest, qualified.contract_digest)
    assert record["measurement_digest"] == data.data_digest()
    assert record["head"]["parameter_names"] == ["weight"] and record["head"]["hidden_size"] == 16
    assert "discarded" in record["head"]["parent_behavior"] and "ordered_updates" in record["head"]["optimizer_step_rule"]
    # "kind" is not a measurement field here: head, optimizer and initialization records carry one.
    # A measurement document always carries value, lower or upper.
    assert not {"value", "lower", "upper"} & set(keys_of(run.identity))
    # The head is outside the policy: its parameters are not policy parameters or policy names.
    assert run.identity["parameter_names"] == [n for n, p in run.policy.named_parameters() if p.requires_grad]
    assert all(p is not run.head.weight for _, p in run.parameters)
    assert run.head.weight.shape == (1, 16) and run.head.weight.dtype == torch.float32 and run.head.bias is None
    assert run.progress()["functional"] == {
        "counters": dict.fromkeys(["pair_opportunities", "ordered_pairs", "tie_pairs", "unresolved_pairs",
                                   "active_pairs", "ordered_updates", "feature_forwards"], 0),
        "seen": {"compared": [], "ordered": []}}
    assert run.functional_plan["schedule"]["pair_opportunities"] == 6


def test_labels_are_not_consulted_when_the_trainer_is_admitted(monkeypatch):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = functional_run(qualified, SMALL_VALUES["ordered"])

    def forbidden(*args, **kwargs):
        raise AssertionError("admission consulted a measurement")

    monkeypatch.setattr(fp.RestrictedFunctionalData, "classify", forbidden)
    monkeypatch.setattr(fp.RestrictedFunctionalData, "replay_outcomes", forbidden)
    assert build_trainer(base, encoder, population, training_config(), functional=run).functional is run


def test_a_caller_cannot_claim_the_derived_records_or_use_the_wrong_arm_or_stage():
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = functional_run(qualified, SMALL_VALUES["ordered"])
    for key in ("budget", "functional"):
        with pytest.raises(ValueError, match="derived budget and functional"):
            build_trainer(copy.deepcopy(base), encoder, population, training_config(),
                          data_identity={key: {"claimed": True}})
    with pytest.raises(TypeError, match="FunctionalRun"):
        build_trainer(copy.deepcopy(base), encoder, population, training_config(), functional=run.data)
    post = TrainingConfig(stage="post", arm="paired", seed=1, total_updates=1, examples_per_update=1,
                          ordering_weight=1.0)
    with pytest.raises(ValueError, match="midtraining-only"):
        StageTrainer(copy.deepcopy(base), encoder, post, functional=run, data_identity={},
                     source_identity={}, initialization_identity={"kind": "release", "base": base.base_identity})


def test_the_functional_run_binds_its_inputs_before_any_update():
    population, qualified = standard_functional_world()
    data = restricted_data(qualified, ORDERED)
    flip = {"higher_better": "lower_better", "lower_better": "higher_better"}
    flipped = fp.RestrictedFunctionalData(data.budget, {g: flip[d] for g, d in data.directions.items()}, data.values)
    with pytest.raises(ValueError, match="directions differ"):
        FunctionalRun(functional_settings(), qualified, flipped)
    forged = _forged(qualified, data.budget.selected[1:3], seed=BUDGET_SEED)
    direct = fp.RestrictedFunctionalData(forged, {g: qualified.direction(g) for g in forged.group_members()},
                                         {key: data.values[key] for key in forged.selected})
    with pytest.raises(ValueError, match="not the seeded prefix"):
        FunctionalRun(functional_settings(), qualified, direct)
    other_decl, other_records, document = standard_world()
    document["study_contract"]["assertion"] = "another curator claim"
    other = fd.qualify_functional(fd.parse_functional_constructs(document), other_decl, other_records)
    with pytest.raises(ValueError, match="not selected from this qualified"):
        FunctionalRun(functional_settings(), other, data)
    with pytest.raises(TypeError):
        FunctionalRun("config", qualified, data)


def test_the_qualification_must_belong_to_the_denoising_population():
    _, _, _, qualified = small_world()
    other_population = qualify_manifest(declaration(), broad_records()[:2])
    base, encoder = fresh_models("lora")
    with pytest.raises(ValueError, match="different broad population"):
        build_trainer(base, encoder, other_population, training_config(),
                      functional=functional_run(qualified, SMALL_VALUES["ordered"]))


def test_the_coverage_requirement_is_enforced_before_the_first_update():
    population, qualified = standard_functional_world()
    base, encoder = fresh_models("lora")
    # g-single's only member can never be compared, so full coverage of the 8 selected
    # observations is impossible for any horizon.
    run = functional_run(qualified, ORDERED, minimum_compared_fraction=1.0, pairs_per_update=3)
    with pytest.raises(ValueError, match="declared horizon compares only"):
        build_trainer(base, encoder, population, training_config(total_updates=40), functional=run)
    relaxed = functional_run(qualified, ORDERED, minimum_compared_fraction=0.8, pairs_per_update=3)
    trainer = build_trainer(copy.deepcopy(base), encoder, population, training_config(total_updates=40),
                            functional=relaxed)
    assert trainer.functional_plan["schedule"]["compared_fraction"]["denominator"] == 8


def test_every_admitted_train_construct_is_preflighted_in_both_views_even_when_unselected():
    decl, records, document = standard_world()
    document["constructs"].append(construct_row("tl", "EVQLVESG" + "ACDEFGHIKL" * 7))
    document["groups"].append(group_row("g-long", "ctx-a", KD, ["tl"]))
    population = qualify_manifest(decl, records)
    qualified = qualify(document, decl, records, population=population)
    seed = next(s for s in (f"s{i}" for i in range(80))
                if ("g-long", "tl") not in fp.select_functional_budget(qualified, 8, s).selected_set)
    base, encoder = fresh_models("lora")
    run = functional_run(qualified, {**ORDERED, ("g-long", "tl"): point(3.0)}, n=8, seed=seed)
    assert ("g-long", "tl") not in run.data.budget.selected_set
    with pytest.raises(ValueError, match="native position limit"):
        build_trainer(base, encoder, population, training_config(), functional=run)


# ---------------------------------------------------------------------------
# Gradient flow: detached control and shared accumulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_the_detached_arm_leaves_every_policy_tensor_and_adam_moment_identical_to_no_aux(mode):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models(mode)
    config = training_config(total_updates=4)
    plain = build_trainer(copy.deepcopy(base), encoder, population, config)
    detached = build_trainer(
        copy.deepcopy(base), encoder, population, config,
        functional=functional_run(qualified, SMALL_VALUES["ordered"], gradient_mode="detached",
                                  head_gradient_clip=1e-3))
    dropout = [m for m in detached.policy.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropout and any(m.p > 0 for m in dropout) and detached.policy.config.activation_checkpointing
    head_start = detached.head.weight.detach().clone()
    for _ in range(config.total_updates):
        expected, actual = plain.step(), detached.step()
        assert expected["gradient_norm"] > config.gradient_clip, "the policy clip must be active"
        stats = actual.pop("functional")
        assert stats["head_stepped"] and stats["head_gradient_norm"] > 1e-3, "the head clip must be active"
        assert actual == expected
        equal(plain.policy.state_dict(), detached.policy.state_dict())
        equal(plain.optimizer.state_dict(), detached.optimizer.state_dict())
    assert not torch.equal(detached.head.weight.detach(), head_start)
    assert all(p.grad is None for p in detached.policy.parameters())
    assert detached.progress()["functional"]["counters"]["ordered_pairs"] == 8


@pytest.mark.parametrize("mode", MODES)
def test_the_shared_arm_adds_its_gradient_to_the_clipped_policy_update(mode):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models(mode)
    encoder_state = copy.deepcopy(encoder.state_dict())
    config = training_config()
    plain = build_trainer(copy.deepcopy(base), encoder, population, config)
    shared = build_trainer(copy.deepcopy(base), encoder, population, config,
                           functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    for update in range(config.total_updates):
        left, right = plain.sampler.draw_batch(update), shared.sampler.draw_batch(update)
        assert [i.record_id for i in left.paired_items] == [i.record_id for i in right.paired_items]
        assert [m.seed for m in left.mask_seeds] == [m.seed for m in right.mask_seeds]
    plain_gradients, shared_gradients = capture_gradients(plain), capture_gradients(shared)
    first_plain, first_shared = plain.step(), shared.step()
    # The denoising draws, masks and dropout are the same, so the first loss is the same ...
    assert first_shared["reconstruction_nll"] == first_plain["reconstruction_nll"]
    # ... while the shared functional gradient changes the clipped retained gradients.
    assert first_shared["gradient_norm"] != first_plain["gradient_norm"]
    assert first_shared["gradient_norm"] > config.gradient_clip
    assert any(not torch.equal(plain_gradients[name], shared_gradients[name]) for name in plain_gradients)
    for _ in range(config.total_updates - 1):
        plain.step()
        shared.step()
    changed = [name for name, tensor in plain.policy.state_dict().items()
               if not torch.equal(tensor, shared.policy.state_dict()[name])]
    assert changed
    equal(encoder_state, encoder.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())


def test_reversing_the_measured_direction_reverses_the_head_gradient():
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    gradients = {}
    for world in ("ordered", "negated"):
        run = build_trainer(copy.deepcopy(base), encoder, population, training_config(total_updates=1),
                            functional=functional_run(qualified, SMALL_VALUES[world], pairs_per_update=1,
                                                      head_gradient_clip=1000.0))
        gradients[world] = capture_head_gradients(run)
        run.step()
    cosine = torch.nn.functional.cosine_similarity(
        gradients["ordered"][0].flatten(), gradients["negated"][0].flatten(), dim=0)
    assert float(cosine) < -0.999


def test_functional_constructs_never_enter_the_denoising_population():
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    config = training_config()
    plain = build_trainer(copy.deepcopy(base), encoder, population, config)
    shared = build_trainer(copy.deepcopy(base), encoder, population, config,
                           functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    expected_data = {"manifest": population.manifest_digest, "population": population.population_digest}
    assert shared.identity["data"] == plain.identity["data"] == expected_data
    shared.run()
    records = {r.record_id for r in population.train_records()}
    assert set(shared.sampler._record_by_id) == records
    assert set(shared.progress()["seen"]["records"]) <= records
    assert shared.progress()["seen"]["records"] != [] and shared.counters["examples"] == 6
    # Construct b exists only in the functional document; it is never a denoising record.
    broad_contents = {r.content_key for r in population.train_records()}
    assert qualified.construct("b").content_key not in broad_contents
    assert qualified.construct("a").content_key in broad_contents   # the allowed consistent overlap


# ---------------------------------------------------------------------------
# Opportunities, weighting and counters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world", ["unresolved", "tied"])
@pytest.mark.parametrize("mode", MODES)
def test_updates_without_an_ordered_pair_never_change_the_head_but_still_assign_its_rate(mode, world):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models(mode)
    config = training_config(total_updates=4, schedule="cosine")
    plain = build_trainer(copy.deepcopy(base), encoder, population, config)
    shared = build_trainer(copy.deepcopy(base), encoder, population, config,
                           functional=functional_run(qualified, SMALL_VALUES[world], head_lr_scale=0.5))
    head_start = shared.head.weight.detach().clone()
    for update in range(config.total_updates):
        expected, actual = plain.step(), shared.step()
        stats = actual.pop("functional")
        assert stats["head_stepped"] is False and stats["head_gradient_norm"] is None
        assert stats["head_learning_rate"] == learning_rate(config, update) * 0.5
        assert (stats["ordered_pairs"], stats["feature_forwards"], stats["active_pairs"]) == (0, 0, 0)
        assert actual == expected
        equal(plain.policy.state_dict(), shared.policy.state_dict())
    assert torch.equal(shared.head.weight.detach(), head_start) and not shared.head_optimizer.state
    counters = shared.progress()["functional"]["counters"]
    assert counters["pair_opportunities"] == 8 and counters["ordered_updates"] == 0
    assert counters["tie_pairs" if world == "tied" else "unresolved_pairs"] == 8
    assert shared.progress()["functional"]["seen"] == {"compared": [["g-x", "a"], ["g-x", "b"]], "ordered": []}


def test_ties_and_unresolved_pairs_consume_their_opportunity_and_are_never_redrawn(monkeypatch):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    calls, original = [], fp.RestrictedFunctionalData.classify

    def counting(self, draw):
        calls.append((draw.update, draw.index))
        return original(self, draw)

    monkeypatch.setattr(fp.RestrictedFunctionalData, "classify", counting)
    run = build_trainer(base, encoder, population, training_config(),
                        functional=functional_run(qualified, SMALL_VALUES["tied"]))
    run.run()
    assert calls == [(u, k) for u in range(3) for k in range(2)]
    assert run.progress()["functional"]["counters"]["tie_pairs"] == 6


@pytest.mark.parametrize("mode", MODES)
def test_the_loss_is_scaled_by_weight_over_pairs_per_update_not_by_the_resolved_pairs(mode):
    population, qualified = standard_functional_world()
    data = restricted_data(qualified, MIXED)
    pairs = 2
    seed = find_seed(lambda s: ordered_pairs_per_update(data, data.budget, s, pairs, 1) == [1])
    base, encoder = fresh_models(mode)
    run = build_trainer(base, encoder, population, training_config(seed=seed, total_updates=1),
                        functional=FunctionalRun(functional_settings(pairs_per_update=pairs), qualified, data))
    schedule = fp.FunctionalSchedule(data.budget)
    ordered = [r for r in (data.classify(schedule.draw(seed, 0, k)) for k in range(pairs)) if r.informative]
    assert len(ordered) == 1
    with deterministic_ordering(run.policy):
        better, worse = (ft.endpoint_features(run.policy, run.encoder, qualified.construct(name), "paired",
                                              grad=False)
                         for name in (ordered[0].better_id, ordered[0].worse_id))
    gap = float(run.head.weight.detach()[0] @ (better - worse))
    expected = 2.0 * max(100.0 - gap, 0.0) ** 2 / pairs      # weight / pairs_per_update
    stats = run.step()["functional"]
    assert (stats["pair_opportunities"], stats["ordered_pairs"], stats["unresolved_pairs"]) == (2, 1, 1)
    assert stats["loss"] == pytest.approx(expected, rel=1e-4)
    assert stats["loss"] != pytest.approx(2 * expected, rel=1e-3)


@pytest.mark.parametrize("mode", MODES)
def test_mixed_ordered_and_unresolved_updates_count_pairs_updates_and_head_steps_separately(mode):
    population, qualified = standard_functional_world()
    data = restricted_data(qualified, MIXED)
    pairs, total, scale = 3, 8, 0.5

    def mixed(seed):
        per_update = ordered_pairs_per_update(data, data.budget, seed, pairs, total)
        return sum(1 for n in per_update if n) >= 2 and sum(1 for n in per_update if not n) >= 2 \
            and any(n >= 2 for n in per_update)

    seed = find_seed(mixed)
    per_update = ordered_pairs_per_update(data, data.budget, seed, pairs, total)
    base, encoder = fresh_models(mode)
    config = training_config(seed=seed, total_updates=total, schedule="cosine")
    run = build_trainer(base, encoder, population, config, functional=FunctionalRun(
        functional_settings(pairs_per_update=pairs, head_lr_scale=scale), qualified, data))
    for update in range(total):
        stats = run.step()["functional"]
        ordered = per_update[update]
        assert stats["ordered_pairs"] == stats["active_pairs"] == ordered        # margin 100: all hinges active
        assert stats["pair_opportunities"] == pairs and stats["ordered_updates"] == int(bool(ordered))
        assert stats["head_stepped"] is bool(ordered) and (stats["head_gradient_norm"] is not None) is bool(ordered)
        assert stats["head_learning_rate"] == learning_rate(config, update) * scale
    counters = run.progress()["functional"]["counters"]
    assert counters["ordered_pairs"] == sum(per_update) and counters["tie_pairs"] == 0
    assert counters["ordered_updates"] == sum(1 for n in per_update if n) < counters["ordered_pairs"]
    assert counters["unresolved_pairs"] == pairs * total - sum(per_update)
    assert counters["feature_forwards"] == 2 * sum(per_update)
    # The head's Adam step count is the number of updates that backpropagated, never the pair count.
    assert float(next(iter(run.head_optimizer.state.values()))["step"]) == counters["ordered_updates"]
    assert run.head_optimizer.param_groups[0]["lr"] == learning_rate(config, total - 1) * scale


@pytest.mark.parametrize("arm", ["paired", "abonly"])
def test_the_functional_forwards_use_the_context_arm_of_the_run(monkeypatch, arm):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = build_trainer(base, encoder, population, training_config(arm=arm, total_updates=1),
                        functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    arms, original = [], ft.endpoint_features

    def recording(policy, encoder, construct, requested, **kwargs):
        arms.append(requested)
        return original(policy, encoder, construct, requested, **kwargs)

    monkeypatch.setattr(ft, "endpoint_features", recording)
    stats = run.step()["functional"]
    assert stats["ordered_pairs"] == 2 and stats["feature_forwards"] == 4
    assert arms == [arm] * 4


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------

def test_a_failure_after_the_policy_optimizer_stepped_poisons_the_trainer(tmp_path, monkeypatch):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = build_trainer(base, encoder, population, training_config(),
                        functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    counters = copy.deepcopy(run.progress())

    def explode(*args, **kwargs):
        raise RuntimeError("injected head optimizer failure")

    monkeypatch.setattr(run.head_optimizer, "step", explode)
    with pytest.raises(RuntimeError, match="injected head optimizer"):
        run.step()
    assert run.failed and run.next_update == 0 and run.progress() == counters
    assert all(p.grad is None for _, p in run.parameters) and run.head.weight.grad is None
    with pytest.raises(RuntimeError, match="failed trainer"):
        run.step()
    with pytest.raises(ValueError, match="successful optimizer boundary"):
        ck.save_checkpoint(run, tmp_path / "poisoned.pt")


def test_a_failure_inside_a_functional_backward_clears_both_sets_of_gradients(monkeypatch):
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = build_trainer(base, encoder, population, training_config(),
                        functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    original = ft.pair_loss

    def late(*args, **kwargs):
        loss, active = original(*args, **kwargs)
        return loss * float("nan"), active

    monkeypatch.setattr(ft, "pair_loss", late)
    with pytest.raises(FloatingPointError, match="functional loss"):
        run.step()
    assert run.failed and run.next_update == 0
    assert all(p.grad is None for _, p in run.parameters) and run.head.weight.grad is None
    assert run.progress()["functional"]["counters"]["ordered_pairs"] == 0
