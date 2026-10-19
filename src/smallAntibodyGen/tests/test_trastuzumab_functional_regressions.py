"""Regressions for the M2 review findings: parent input lineage, restricted-data binding, CUDA
head saves and strict AdamW state layout.

Parent promotion goes through the real ``validate_stage_lineage`` with the light fake parent of
the transfer-guard tests; the checkpoint cases corrupt an in-memory copy of a real checkpoint and
require a validation error (never a backend error) before anything live is touched.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc import post_budget as pb
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest
from smallAntibodyGen.experiments.trastuzumab_poc.factory import build_models
from smallAntibodyGen.experiments.trastuzumab_poc.functional_checkpoint import (
    functional_identity, functional_input_identity, functional_progress, new_counters, new_seen,
    validate_functional_training,
)
from smallAntibodyGen.experiments.trastuzumab_poc.runner import learning_rate
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import validate_stage_lineage
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig
from smallAntibodyGen.tests.test_trastuzumab_functional_checkpoint import (
    SOURCE, assert_untouched, head_group, head_weights, snapshot,
)
from smallAntibodyGen.tests.test_trastuzumab_functional_protocol import point
from smallAntibodyGen.tests.test_trastuzumab_functional_runner import (
    SMALL_VALUES, build_trainer, fresh_models, functional_run, restricted_data, small_world,
    training_config,
)
from smallAntibodyGen.tests.test_trastuzumab_runner import equal
from smallAntibodyGen.tests.test_trastuzumab_transfer_guard import _parent, _post, _qualify


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def put(container, key, value):
    container[key] = value


# ---------------------------------------------------------------------------
# CX-15: the parent's derived input records are explicit and agree with everything else
# ---------------------------------------------------------------------------

def parent_lineage(mutate=None):
    """Promote the fake midtraining parent (optionally corrupted) the way a post run does."""
    policy, _ = _parent()
    provenance = policy.base_identity["parent_provenance"]
    if mutate is not None:
        mutate(provenance["training"], provenance["progress"])
    # The origin copies the (possibly mutated) base, so only the corruption itself can reject.
    origin = {"kind": "parent", "sha256": "a" * 64, "base": copy.deepcopy(policy.base_identity)}
    return validate_stage_lineage(policy, SimpleNamespace(stage="post"), origin, post_data=_post())


def functional_input_record():
    _, _, _, qualified = small_world()
    return functional_input_identity(functional_run(qualified, SMALL_VALUES["ordered"]))


def with_functional_binding(certificate):
    """A self-consistent (re-signed) certificate that binds some functional qualification."""
    bound = copy.deepcopy(certificate)
    bound["functional"] = {
        "qualified_digest": "1" * 64, "contract_digest": "2" * 64,
        "declaration_digest": bound["declaration_digest"], "population_digest": bound["population_digest"],
        "broad_input_digest": "3" * 64, "train_constructs": 2, "train_unique_antibodies": 2,
        "train_unique_antigens": 1, "train_antibody_families": ["famF"], "train_target_families": ["her2"]}
    bound["certificate_sha256"] = digest({k: v for k, v in bound.items() if k != "certificate_sha256"})
    return bound


def test_a_healthy_schema_two_parent_without_the_arm_promotes_and_other_input_keys_are_free():
    assert parent_lineage() == _qualify()
    # Informational data_identity keys sit next to the derived records and are not constrained.
    assert parent_lineage(lambda t, p: put(t["input"], "input", "synthetic")) == _qualify()
    assert parent_lineage(lambda t, p: put(t["input"], "anything", {"nested": [1, 2]})) == _qualify()


def test_a_parent_whose_input_declares_a_functional_arm_that_its_identity_does_not_is_rejected():
    claimed = functional_input_record()
    assert claimed["budget"]["budget_n"] == 2          # a real, canonical record, not a stub
    # identity, progress and certificate all say "no arm"; only input.functional claims one
    with pytest.raises(ValueError, match="functional records without a functional identity"):
        parent_lineage(lambda t, p: put(t["input"], "functional", claimed))
    # ... and the same contradiction through the progress record
    claimed_progress = functional_progress(new_counters(), new_seen())
    with pytest.raises(ValueError, match="functional records without a functional identity"):
        parent_lineage(lambda t, p: put(p, "functional", claimed_progress))
    # ... and through a certificate that binds a qualification the identity never declared
    with pytest.raises(ValueError, match="does not declare"):
        parent_lineage(lambda t, p: put(t, "transfer_certificate", with_functional_binding(t["transfer_certificate"])))


def test_a_midtraining_parent_never_records_a_post_fitting_budget():
    budget = pb.post_budget_identity(_post().budget)
    assert pb.validate_post_budget_identity(budget)    # canonical and valid on its own
    # null functional records and zero fitting labels in the progress: only the input says otherwise
    with pytest.raises(ValueError, match="post fitting budget"):
        parent_lineage(lambda t, p: put(t["input"], "budget", budget))


MISSING_FIELDS = {
    "input_absent": lambda t, p: t.pop("input"),
    "input_null": lambda t, p: put(t, "input", None),
    "input_list": lambda t, p: put(t, "input", [{"budget": None, "functional": None}]),
    "budget_absent": lambda t, p: t["input"].pop("budget"),
    "input_functional_absent": lambda t, p: t["input"].pop("functional"),
    "identity_functional_absent": lambda t, p: t.pop("functional"),
    "progress_functional_absent": lambda t, p: p.pop("functional"),
}


@pytest.mark.parametrize("name", sorted(MISSING_FIELDS))
def test_absent_explicit_parent_fields_are_value_errors(name):
    with pytest.raises(ValueError, match="explicit"):
        parent_lineage(MISSING_FIELDS[name])


def test_a_malformed_functional_parent_record_is_a_value_error_not_a_key_error():
    _, _, _, qualified = small_world()
    run = functional_run(qualified, SMALL_VALUES["ordered"])

    def declare(training, progress):
        # Every record is well formed, but the fixture's config has no seed: the validator reads
        # it, and the resulting KeyError must surface as a ValueError.
        assert "seed" not in training["config"]
        training["functional"] = functional_identity(run, hidden_size=16, seed=11)
        training["input"]["functional"] = functional_input_identity(run)
        progress["functional"] = functional_progress(new_counters(), new_seen())

    with pytest.raises(ValueError, match="malformed"):
        parent_lineage(declare)


# ---------------------------------------------------------------------------
# CX-16: the restricted measurements must be the recorded ones
# ---------------------------------------------------------------------------

def test_restricted_measurements_must_hash_to_the_recorded_measurement_digest():
    _, _, population, qualified = small_world()
    base, encoder = fresh_models("lora")
    run = build_trainer(base, encoder, population, training_config(total_updates=2),
                        functional=functional_run(qualified, SMALL_VALUES["ordered"]))
    run.run()
    honest = run.functional.data
    altered = restricted_data(qualified, {("g-x", "a"): point(10.0), ("g-x", "b"): point(20.0)})
    # Same budget, observation order, directions and resolution of every pair: only the numbers differ.
    assert altered.budget == honest.budget and list(altered.values) == list(honest.values)
    seed, pairs = run.config.seed, run.functional.config.pairs_per_update
    assert altered.replay_outcomes(seed, pairs, 2) == honest.replay_outcomes(seed, pairs, 2)
    assert altered.data_digest() != honest.data_digest()
    identity, progress = run.identity, run.progress()
    assert validate_functional_training(identity, progress, restricted=honest)["ordered_updates"] == 2
    with pytest.raises(ValueError, match="measurement digest"):
        validate_functional_training(identity, progress, restricted=altered)
    # The data-free reading never needs the measurements, so it stays label-free.
    assert validate_functional_training(identity, progress)["ordered_updates"] == 2


# ---------------------------------------------------------------------------
# CX-17: a head that never stepped saves and resumes from CUDA
# ---------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
HEAD_SCALE = 0.5


def cuda_trainer(values, *, total_updates=3):
    """A real tiny StageTrainer with the policy, the frozen encoder and the head on CUDA."""
    _, _, population, qualified = small_world()
    policy, encoder, _ = build_models(
        {"kind": "tiny"}, FusionConfig(context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4), device="cuda")
    config = training_config(total_updates=total_updates, schedule="cosine")
    run = build_trainer(policy, encoder, population, config,
                        functional=functional_run(qualified, values, head_lr_scale=HEAD_SCALE))
    return run, config


@needs_cuda
def test_a_pristine_cuda_trainer_saves_and_resumes_its_untouched_head(tmp_path):
    run, config = cuda_trainer(SMALL_VALUES["ordered"])
    assert run.policy.device.type == "cuda" and run.head.weight.device.type == "cuda"
    path = tmp_path / "zero.pt"
    ck.save_checkpoint(run, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["progress"]["next_update"] == 0 and payload["functional"]["optimizer"]["state"] == {}
    assert torch.equal(head_weights(payload)["weight"], run.head.weight.detach().cpu())
    assert head_group(payload)["lr"] == learning_rate(config, 0) * HEAD_SCALE
    resumed, _ = cuda_trainer(SMALL_VALUES["ordered"])
    ck.resume_checkpoint(resumed, path)
    assert resumed.next_update == 0 and not resumed.head_optimizer.state
    assert resumed.head.weight.device.type == "cuda"
    assert torch.equal(resumed.head.weight.detach().cpu(), run.head.weight.detach().cpu())
    resumed.step()
    assert resumed.next_update == 1
    assert float(next(iter(resumed.head_optimizer.state.values()))["step"]) == 1.0


@needs_cuda
@pytest.mark.parametrize("world", ["unresolved", "tied"])
def test_an_entirely_unordered_cuda_run_saves_and_resumes_with_the_initial_head(tmp_path, world):
    run, config = cuda_trainer(SMALL_VALUES[world])
    initial = run.head.weight.detach().cpu().clone()
    run.run(stop_after_updates=2)
    counters = run.progress()["functional"]["counters"]
    assert counters["ordered_pairs"] == 0 and counters["ordered_updates"] == 0
    path = tmp_path / f"{world}.pt"
    ck.save_checkpoint(run, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["functional"]["optimizer"]["state"] == {}
    assert torch.equal(head_weights(payload)["weight"], initial)
    assert head_group(payload)["lr"] == learning_rate(config, 1) * HEAD_SCALE     # assigned, never stepped
    resumed, _ = cuda_trainer(SMALL_VALUES[world])
    ck.resume_checkpoint(resumed, path)
    assert resumed.next_update == 2 and resumed.progress() == run.progress()
    assert torch.equal(resumed.head.weight.detach().cpu(), initial) and not resumed.head_optimizer.state
    trained = run.policy.trainable_state()
    for name, tensor in resumed.policy.trainable_state().items():
        assert torch.equal(tensor, trained[name]), name
    resumed.run()
    assert resumed.next_update == config.total_updates
    # The exact equality check survives the device fix. Save derives the head from the live trainer,
    # so only a forged file can disagree with the deterministic initial head a resume starts from.
    put(head_weights(payload), "weight", initial + 1.0)
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    pristine, _ = cuda_trainer(SMALL_VALUES[world])
    before, cuda_rng = snapshot(pristine), torch.cuda.get_rng_state()
    with pytest.raises(ValueError, match="no head step"):
        ck.resume_checkpoint(pristine, forged)
    assert_untouched(pristine, before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert pristine.next_update == 0 and not pristine.failed
    assert not pristine.optimizer.state and not pristine.head_optimizer.state
    assert pristine.head.weight.device.type == "cuda"
    assert torch.equal(pristine.head.weight.detach().cpu(), initial)


# ---------------------------------------------------------------------------
# CX-18: AdamW state indices are exact integers and the step is a strided tensor
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A functional run stopped after two of four updates, with both optimizers populated."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, _, population, qualified = small_world()
        base, encoder = fresh_models("lora")
        config = training_config(total_updates=4, schedule="cosine")

        def make():
            return build_trainer(copy.deepcopy(base), encoder, population, config,
                                 functional=functional_run(qualified, SMALL_VALUES["ordered"]))

        run = make()
        run.run(stop_after_updates=2)
        path = tmp_path_factory.mktemp("regressions") / "latest.pt"
        ck.save_checkpoint(run, path)
        return SimpleNamespace(make=make, path=path)
    finally:
        torch.set_num_threads(previous)


OPTIMIZERS = {"policy": lambda payload: payload["optimizer"],
              "head": lambda payload: payload["functional"]["optimizer"]}


def assert_rejected_before_mutation(saved, monkeypatch, mutate, match):
    """Resume and the data-free reader both reject the corrupted in-memory payload; resume mutates nothing."""
    payload = torch.load(saved.path, map_location="cpu", weights_only=True)
    mutate(payload)
    monkeypatch.setattr(ck, "read_checkpoint", lambda path, expected_sha256=None: payload)
    fresh = saved.make()
    before = snapshot(fresh)
    with pytest.raises(ValueError, match=match):
        ck.resume_checkpoint(fresh, saved.path)
    assert_untouched(fresh, before)
    assert fresh.next_update == 0 and not fresh.failed
    assert not fresh.optimizer.state and not fresh.head_optimizer.state
    policy, encoder = fresh_models("lora")
    policy_before = copy.deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match=match):
        ck.load_inference_checkpoint(policy, saved.path, encoder=encoder, expected_sha256=None,
                                     source_identity=SOURCE)
    equal(policy_before, policy.state_dict())


@pytest.mark.parametrize("key", [False, 0.0], ids=["bool", "float"])
@pytest.mark.parametrize("which", sorted(OPTIMIZERS))
def test_adam_state_indices_must_be_exact_integers(saved, monkeypatch, which, key):
    def mutate(payload):
        optimizer = OPTIMIZERS[which](payload)
        replaced = {key if index == 0 else index: entry for index, entry in optimizer["state"].items()}
        # A plain set comparison cannot tell these apart: False and 0.0 hash and compare equal to 0.
        assert set(replaced) == set(optimizer["state"]) and any(type(k) is not int for k in replaced)
        optimizer["state"] = replaced

    assert_rejected_before_mutation(saved, monkeypatch, mutate, "state inventory")


@pytest.mark.parametrize("which", sorted(OPTIMIZERS))
def test_a_non_strided_adam_step_is_a_validation_error_not_a_backend_error(saved, monkeypatch, which):
    def mutate(payload):
        entry = OPTIMIZERS[which](payload)["state"][0]
        entry["step"] = entry["step"].to_sparse()
        assert entry["step"].layout == torch.sparse_coo and entry["step"].shape == torch.Size([])

    assert_rejected_before_mutation(saved, monkeypatch, mutate, "invalid AdamW step")
