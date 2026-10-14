"""Actual tiny-model integration, strict resume and fault-injection regressions."""
import copy
from dataclasses import replace
import json
import random

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import checkpoint as ck
from smallAntibodyGen.experiments.trastuzumab_poc.factory import build_models
from smallAntibodyGen.experiments.trastuzumab_poc.runner import StageTrainer, TrainingConfig
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest, MeasuredEndpoint,
    RestrictedPostData, SelectionAllowance, select_budget,
)
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig
from smallAntibodyGen.tests.test_trastuzumab_sampling import _make_pop_two_families


def post_data(ties=False):
    entries = tuple(EndpointEntry(str(i), EndpointConstruct("AC", core, "DE", "FG", "HIK"),
                                  "post_pool", "n", "assay", "synthetic affinity")
                    for i, core in enumerate(("ACDEFGHIKL", "LMNPQRSTVW", "AAAAAAAAAA")))
    manifest = EndpointManifest(entries)
    budget = select_budget(manifest, 3, "shared")
    labels = tuple(MeasuredEndpoint(e.endpoint_id, 1.0 if ties else float(i),
                                    "higher_better", "assay") for i, e in enumerate(entries))
    return RestrictedPostData(manifest=manifest, budget=budget, measured=labels, fitting_count=3,
                              development_labels=(), selection_allowance=SelectionAllowance(0, ()))


def make(mode="lora", arm="paired", stage="mid", total=4, ties=False, **changes):
    policy, encoder, origin = build_models({"kind": "tiny"}, FusionConfig(
        context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4,
        activation_checkpointing=False, update_mode=mode))
    config = TrainingConfig(stage=stage, arm=arm, seed=7, total_updates=total,
                            examples_per_update=2, ordering_weight=1.0 if stage == "post" else 0.0,
                            **changes)
    data = {"population": _make_pop_two_families()} if stage == "mid" else {"post_data": post_data(ties)}
    return StageTrainer(policy, encoder, config, **data, data_identity={"input": "synthetic"},
                        source_identity={"source": "synthetic"}, initialization_identity=origin)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("schedule", ["constant", "cosine"])
def test_resume_matches_uninterrupted_and_can_replace_same_file(tmp_path, mode, schedule):
    full = make(mode=mode, schedule=schedule)
    full.run()
    partial = make(mode=mode, schedule=schedule)
    partial.run(stop_after_updates=2)
    path = tmp_path / "latest.pt"
    ck.save_checkpoint(partial, path)
    resumed = make(mode=mode, schedule=schedule)
    before = torch.random.get_rng_state().clone()
    ck.resume_checkpoint(resumed, path)
    assert torch.equal(before, torch.random.get_rng_state())
    resumed.run()
    equal(full.policy.state_dict(), resumed.policy.state_dict())
    equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    equal(full.progress(), resumed.progress())
    ck.save_checkpoint(resumed, path)
    done = make(mode=mode, schedule=schedule)
    ck.resume_checkpoint(done, path)
    assert done.run() == full.progress()


def test_diagnostic_rng_does_not_change_training():
    a, b = make(), make()
    initial = torch.random.get_rng_state().clone()
    a.step()
    assert torch.equal(initial, torch.random.get_rng_state())
    torch.rand(31)
    random.random()
    b.step()
    equal(a.policy.state_dict(), b.policy.state_dict())
    equal(a.progress(), b.progress())


@pytest.mark.parametrize("corruption", ["moment_dtype", "moment_negative", "step", "names", "rng", "counter", "horizon"])
def test_late_checkpoint_failure_does_not_mutate(tmp_path, corruption):
    trained = make()
    trained.step()
    path = tmp_path / "checkpoint.pt"
    ck.save_checkpoint(trained, path)
    payload = torch.load(path, weights_only=True)
    if corruption == "moment_dtype":
        payload["optimizer"]["state"][0]["exp_avg"] = payload["optimizer"]["state"][0]["exp_avg"].double()
    elif corruption == "moment_negative":
        payload["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif corruption == "step":
        payload["optimizer"]["state"][0]["step"] += 1
    elif corruption == "names":
        payload["identity"]["parameter_names"].reverse()
    elif corruption == "rng":
        payload["rng"]["next_update"] += 1
    elif corruption == "counter":
        payload["progress"]["counters"]["examples"] += 1
    else:
        payload["identity"]["config"]["total_updates"] += 1
    torch.save(payload, path)
    fresh = make()
    state = copy.deepcopy(fresh.policy.state_dict())
    optimizer = copy.deepcopy(fresh.optimizer.state_dict())
    progress = copy.deepcopy(fresh.progress())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        ck.resume_checkpoint(fresh, path)
    equal(state, fresh.policy.state_dict())
    equal(optimizer, fresh.optimizer.state_dict())
    equal(progress, fresh.progress())
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("field", ["data", "source", "initialization"])
def test_semantic_drift_rejects_resume(tmp_path, field):
    trained = make()
    trained.step()
    path = tmp_path / "checkpoint.pt"
    ck.save_checkpoint(trained, path)
    fresh = make()
    fresh.identity[field] = {"different": True}
    with pytest.raises(ValueError, match="identity"):
        ck.resume_checkpoint(fresh, path)


@pytest.mark.parametrize("failure", ["serialize", "publish"])
def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    run = make()
    run.step()
    path = tmp_path / "latest.pt"
    ck.save_checkpoint(run, path)
    old = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("injected")
    monkeypatch.setattr(ck.torch if failure == "serialize" else ck.os,
                        "save" if failure == "serialize" else "replace", fail)
    with pytest.raises(OSError, match="injected"):
        ck.save_checkpoint(run, path)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


def test_arms_match_exposures_and_mask_counts():
    paired, abonly = make(), make(arm="abonly")
    paired.run()
    abonly.run()
    assert paired.progress()["seen"] == abonly.progress()["seen"]
    for key in ("examples", "masked_residues", "teacher_forced_states"):
        assert paired.counters[key] == abonly.counters[key]
    assert paired.counters["antigen_examples"] == 8
    assert abonly.counters["antigen_examples"] == 0


@pytest.mark.parametrize("ties", [False, True])
def test_post_training_uses_only_selected_endpoints_and_preserves_encoder(tmp_path, ties):
    run = make(stage="post", total=2, ties=ties)
    encoder_state = copy.deepcopy(run.encoder.state_dict())
    run.step()
    assert run.counters["pair_draws"] == 2
    assert run.counters["ties"] == (2 if ties else 0)
    assert run.counters["teacher_forced_states"] == (0 if ties else 40)
    assert set(run.progress()["seen"]["endpoints"]) <= set(run.post_data.budget.selected_ids)
    equal(encoder_state, run.encoder.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in run.encoder.parameters())
    ck.save_checkpoint(run, tmp_path / "post.pt")
    resumed = make(stage="post", total=2, ties=ties)
    ck.resume_checkpoint(resumed, tmp_path / "post.pt")
    run.step()
    resumed.step()
    equal(run.policy.state_dict(), resumed.policy.state_dict())


def test_nonfinite_update_cannot_publish(tmp_path, monkeypatch):
    run = make()
    monkeypatch.setattr(run.policy, "reconstruction_nll",
                        lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError):
        run.step()
    assert run.failed
    with pytest.raises(ValueError):
        ck.save_checkpoint(run, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):
        run.step()


def test_inference_load_does_not_require_training_files(tmp_path):
    trained = make()
    trained.step()
    artifact = ck.save_checkpoint(trained, tmp_path / "checkpoint.pt")
    fresh = make()
    ck.load_inference_checkpoint(fresh.policy, artifact["path"], expected_sha256=artifact["sha256"],
                                source_identity={"source": "synthetic"})
    equal(trained.policy.state_dict(), fresh.policy.state_dict())


@pytest.mark.parametrize("field", ["schema", "selection", "optimizer"])
def test_inference_checkpoint_rejects_invalid_recipe_metadata(tmp_path, field):
    trained = make()
    trained.step()
    path = tmp_path / "checkpoint.pt"
    ck.save_checkpoint(trained, path)
    payload = torch.load(path, weights_only=True)
    payload["identity"][field] = "invalid"
    torch.save(payload, path)
    fresh = make()
    before = copy.deepcopy(fresh.policy.state_dict())
    with pytest.raises(ValueError):
        ck.load_inference_checkpoint(fresh.policy, path, expected_sha256=ck.file_digest(path),
                                    source_identity={"source": "synthetic"})
    equal(before, fresh.policy.state_dict())


def test_resume_rejects_invented_inherited_exposure(tmp_path):
    trained = make()
    trained.step()
    path = tmp_path / "checkpoint.pt"
    ck.save_checkpoint(trained, path)
    payload = torch.load(path, weights_only=True)
    payload["progress"]["inherited_exposure"] = {"invented": 1}
    torch.save(payload, path)
    fresh = make()
    with pytest.raises(ValueError, match="inherited exposure"):
        ck.resume_checkpoint(fresh, path)
    assert fresh.next_update == 0


@pytest.mark.parametrize("options", [
    {"learning_rate": float("nan")}, {"examples_per_update": True},
    {"reconstruction_weight": 0}, {"ordering_weight": 1}, {"span_weight": 0.8, "full_region_weight": 0.3},
])
def test_training_config_rejects_unsupported_recipe(options):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(**options)
