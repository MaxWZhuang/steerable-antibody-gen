"""The qualification checks against the REAL tiny-model services: profile, decode, bootstrap.

What is real: ``FlightServices.profile`` (its clip, finite check and optimizer), ``tasks.dispatch``,
``generate.fill_batch`` / ``model.iterative_decode`` and ``services.window_scores``. What is
replaced: the pinned-weights loader (a tiny random-init trunk, as in the DPLM services tests) and,
where a test needs a defect, exactly one seam -- the loss a task returns, or the decoder's output.
Nothing here says anything about a GPU: every process below stays on the CPU.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_env as env
from smallAntibodyGen.experiments import her2_routes_qualify as qualify
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
from smallAntibodyGen.experiments.her2_routes_conditions import BIN_HIGH, Condition
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (
    ROUTE_FULL, ROUTE_LORA, harness, reconstruction_stream)

ROOT = Path(__file__).resolve().parents[3]
JOINT = json.loads((ROOT / "configs/experiments/her2_generative_routes_joint.json")
                   .read_text(encoding="utf-8"))
CONDITION = Condition(family="affinity", bin=BIN_HIGH, assay="a1")


def with_production_tolerances(services):
    """The tiny harness gets the PRODUCTION tolerances; production is not relaxed for a test."""
    services.config["tolerances"] = {key: JOINT["tolerances"][key]
                                     for key in qualify.TOLERANCE_KEYS}
    return services


def decode(services, route, model, **overrides):
    return qualify.verify_native_decode(
        services, route, model, sites=serial.her2_core_edit_map(), condition=CONDITION,
        draws=2, batch_size=2, **overrides)


# ---------------------------------------------------------------------------
# profile: non-finite loss / gradient skip the optimizer, CPU touches no CUDA
# ---------------------------------------------------------------------------

class NanGradient(torch.autograd.Function):
    """Identity forward with a non-finite backward: a finite loss whose gradient is not."""

    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * float("nan")


def profile_with_defect(harness, monkeypatch, corrupt):
    services = harness.services
    steps = []
    original_build = services_mod.FlightServices.build_optimizer

    def counting(self, route, parameters, *, lr, **options):
        optimizer = original_build(self, route, parameters, lr=lr, **options)
        real = optimizer.step
        optimizer.step = lambda *args, **kwargs: (steps.append(1), real(*args, **kwargs))[1]
        return optimizer
    monkeypatch.setattr(services_mod.FlightServices, "build_optimizer", counting)
    real_dispatch = tasks_lib.dispatch

    def defective(*args, **kwargs):
        outcome = real_dispatch(*args, **kwargs)
        return dataclasses.replace(outcome, loss=corrupt(outcome.loss))
    monkeypatch.setattr(tasks_lib, "dispatch", defective)
    draw = reconstruction_stream(services)(1, 0, np.random.default_rng(0))
    profile = services.profile(ROUTE_FULL, microbatches=(2,), steps=3,
                               draws={data_lib.TASK_RECONSTRUCTION: draw})
    return profile, steps


@pytest.mark.parametrize("corrupt,failure", [
    (lambda loss: loss * float("nan"), "nonfinite_loss"),
    (NanGradient.apply, "nonfinite_gradient")], ids=["loss", "gradient"])
def test_a_nonfinite_loss_or_gradient_skips_the_optimizer_and_fails_the_row(
        harness, monkeypatch, corrupt, failure):
    profile, steps = profile_with_defect(harness, monkeypatch, corrupt)
    row = profile["rows"][0]
    assert row["failure"] == failure and row["accepted"] is False
    assert row["completed_steps"] == 0 and steps == [], "optimizer.step ran on a bad step"
    assert row["failure_detail"]["step"] == 0
    assert profile["chosen_microbatch"] is None
    json.dumps(qualify.json_safe(profile))
    status, reason, _ = qualify.analyse_profile(
        json.loads(json.dumps(qualify.json_safe(profile))), route=ROUTE_FULL,
        tasks=(data_lib.TASK_RECONSTRUCTION,), diffusion=True, gpu_required=False)
    assert status == qualify.VERDICT_FAILED and "code defect" in reason, (
        "a non-finite computation is a failure, never a capacity refusal")


def test_a_finite_profile_counts_every_step_it_applied(harness):
    services = harness.services
    draw = reconstruction_stream(services)(1, 0, np.random.default_rng(0))
    profile = services.profile(ROUTE_FULL, microbatches=(2,), steps=3,
                               draws={data_lib.TASK_RECONSTRUCTION: draw})
    row = profile["rows"][0]
    assert row["failure"] is None and row["completed_steps"] == row["steps_requested"] == 3
    assert row["rows_measured"] > 0 and row["accepted"] is True


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_the_profile_trains_an_eval_loaded_native_and_runs_layer_checkpoints(
        harness, monkeypatch, route):
    """load_pinned_dplm returns eval(); the profile must still measure the production train step."""
    from smallAntibodyGen.models import dplm_policy
    real_loader = dplm_policy.load_pinned_dplm

    def eval_loaded(*args, **kwargs):
        loaded = real_loader(*args, **kwargs)
        loaded.model.eval()
        return loaded
    monkeypatch.setattr(dplm_policy, "load_pinned_dplm", eval_loaded)
    services = harness.services
    services.config["routes"][route]["activation_checkpointing"] = True
    seen, layers, checkpointed = [], [], []
    real_dispatch = tasks_lib.dispatch
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def observing(svc, route_name, model, heads, *args, **kwargs):
        layers[:] = list(model.native.esm.encoder.layer)
        seen.append((model.training, model.native.training, heads.training))
        return real_dispatch(svc, route_name, model, heads, *args, **kwargs)

    def counting(function, *args, **kwargs):
        checkpointed.append(function)
        return real_checkpoint(function, *args, **kwargs)
    monkeypatch.setattr(tasks_lib, "dispatch", observing)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting)
    draw = reconstruction_stream(services)(1, 0, np.random.default_rng(0))
    profile = services.profile(route, microbatches=(2,), steps=2,
                               draws={data_lib.TASK_RECONSTRUCTION: draw})
    assert seen and all(mode == (True, True, True) for mode in seen), seen
    assert profile["training_mode"] == {"model": True, "native": True, "heads": True}
    assert {id(layer) for layer in layers} <= {id(fn) for fn in checkpointed}, (
        "the per-layer checkpoints did not execute at real dispatch")
    assert profile["rows"][0]["failure"] is None and profile["device_total_mib"] is None
    status, reason, _ = qualify.analyse_profile(
        json.loads(json.dumps(qualify.json_safe(profile))), route=route,
        tasks=(data_lib.TASK_RECONSTRUCTION,), diffusion=True, gpu_required=False)
    assert status is None, reason


def test_a_cpu_profile_on_a_gpu_capable_host_never_touches_cuda(harness, monkeypatch):
    touched = []

    def forbidden(name):
        def call(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"torch.cuda.{name} was called by a CPU profile")
        return call
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize", "max_memory_allocated",
                 "max_memory_reserved", "get_device_name", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden(name))
    monkeypatch.setattr(services_mod, "vram_reading", forbidden("vram_reading"))
    services = harness.services
    draw = reconstruction_stream(services)(1, 0, np.random.default_rng(0))
    profile = services.profile(ROUTE_FULL, microbatches=(2,), steps=2,
                               draws={data_lib.TASK_RECONSTRUCTION: draw})
    row = profile["rows"][0]
    assert touched == [] and profile["measured_on"] == {"cuda": False, "device": None}
    assert profile["device_total_mib"] is None
    assert row["failure"] is None and row["accepted"] is True
    assert row["peak_allocated_mib"] is None and row["vram"]["nvml_fb_free_mib"] is None
    assert row["signals"]["driver_free_above_floor"] is None


# ---------------------------------------------------------------------------
# ledger_sample: counting is grad-free, eval-mode, and CPU-isolated
# ---------------------------------------------------------------------------

def ledger_services(harness, monkeypatch, route, *, eval_native=True):
    from smallAntibodyGen.models import dplm_policy
    real_loader = dplm_policy.load_pinned_dplm

    def eval_loaded(*args, **kwargs):
        loaded = real_loader(*args, **kwargs)
        if eval_native:
            loaded.model.eval()
        return loaded
    monkeypatch.setattr(dplm_policy, "load_pinned_dplm", eval_loaded)
    services = harness.services
    services.config["routes"]["enabled"] = [route]
    services.config["split"]["seed"] = 20260927
    stream = reconstruction_stream(services)
    monkeypatch.setattr(services, "curriculum_stream", lambda *args, **kwargs: stream)
    return services


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_ledger_sample_counts_real_dispatch_without_grad_and_in_eval(
        harness, monkeypatch, route):
    services = ledger_services(harness, monkeypatch, route)
    observed, outcomes = [], []
    real_dispatch = tasks_lib.dispatch

    def inspect(svc, route_name, model, heads, draw, **kwargs):
        result = real_dispatch(svc, route_name, model, heads, draw, **kwargs)
        observed.append({"grad_enabled": torch.is_grad_enabled(),
                         "loss_requires_grad": result.loss.requires_grad,
                         "loss_finite": bool(torch.isfinite(result.loss).all()),
                         "modes": (model.training, model.native.training, heads.training)})
        outcomes.append((draw.task, result.forwards / max(1, result.rows)))
        return result
    monkeypatch.setattr(tasks_lib, "dispatch", inspect)
    assert torch.is_grad_enabled()
    report = services.ledger_sample(attempts=3)
    assert torch.is_grad_enabled(), "the grad mode must be restored after the sample"
    assert observed and report["tasks_observed"]
    assert all(not row["grad_enabled"] and not row["loss_requires_grad"]
               and row["loss_finite"] and row["modes"] == (False, False, False)
               for row in observed), observed
    expected = {}
    for task, ratio in outcomes:
        expected.setdefault(task, []).append(ratio)
    assert report["measured_from"] == {task: float(np.mean(values))
                                       for task, values in sorted(expected.items())}
    assert sorted(expected) == report["tasks_observed"]


def test_ledger_sample_counts_match_dispatch_run_outside_the_mode_boundary(
        harness, monkeypatch):
    """The eval/no_grad boundary changes what is built, not what is counted."""
    services = ledger_services(harness, monkeypatch, ROUTE_FULL)
    report = services.ledger_sample(attempts=1)
    model = services.build_route(ROUTE_FULL)
    model.native.eval()
    heads = services.build_heads(ROUTE_FULL, model)
    seed = int(services.config["split"]["seed"])
    rng = np.random.default_rng([seed, 41])
    draw = services.curriculum_stream(ROUTE_FULL, seed=seed, updates=1)(1, 0, rng)
    outcome = tasks_lib.dispatch(services, ROUTE_FULL, model, heads, draw, arm="full",
                                 settings=services.task_settings(), rng=rng)
    assert report["measured_from"][draw.task] == float(outcome.forwards / max(1, outcome.rows))


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_a_cpu_ledger_sample_on_a_gpu_capable_host_never_touches_cuda(
        harness, monkeypatch, route):
    services = ledger_services(harness, monkeypatch, route)
    touched = []

    def forbidden(name):
        def call(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"torch.cuda.{name} was called by a CPU ledger sample")
        return call
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize", "max_memory_allocated",
                 "max_memory_reserved", "get_device_name", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden(name))
    monkeypatch.setattr(services_mod, "vram_reading", forbidden("vram_reading"))
    report = services.ledger_sample(attempts=2)
    assert touched == [] and report["tasks_observed"]


# ---------------------------------------------------------------------------
# decode: a real tiny DPLM, its real native decoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_the_real_native_decoder_passes_for_the_full_and_the_lora_route(harness, route):
    services = with_production_tolerances(harness.services)
    model = services.build_route(route).eval()
    report = decode(services, route, model)
    assert report["passed"], (report["failures"], report["max_rescore_difference"])
    assert all(report["checks"].values())
    assert report["rescore_tolerance"]["atol"] == JOINT["tolerances"][
        "sum_log_probability_atol"]
    assert report["rescore_tolerance"]["rtol"] == JOINT["tolerances"][
        "sum_log_probability_rtol"]
    assert report["max_rescore_difference"] is not None
    assert report["max_rescore_excess_over_allowance"] <= 0.0
    assert report["hook_forward_calls"] == report["reported_forward_calls"]
    assert "iterative_decode" not in vars(model), "the wrapper is removed afterwards"


def corrupting(model, monkeypatch, mutate):
    """Replace the decoder with one that calls the real decoder and then ``mutate``s."""
    original = model.iterative_decode

    def wrapped(tokens, query_slots, **kwargs):
        result = original(tokens, query_slots, **kwargs)
        return mutate(tokens, query_slots, result)
    monkeypatch.setattr(model, "iterative_decode", wrapped)


def output_only(tokens, slots, result):
    native = result.native_tokens.clone()
    native[0, 0] = native[0, 1]
    return dataclasses.replace(result, native_tokens=native)


def input_only(tokens, slots, result):
    tokens[0, 0] = tokens[0, 1]
    return result


def input_and_output_identically(tokens, slots, result):
    from smallAntibodyGen.models import dplm_policy as dplm
    tokens[0, 0] = tokens[0, 1]
    native = result.native_tokens.clone()
    native[0, 0] = dplm.ablang_to_dplm(torch.as_tensor(tokens))[0, 0]
    return dataclasses.replace(result, native_tokens=native)


def slots_in_place(tokens, slots, result):
    slots[0, 0], slots[0, 1] = slots[0, 1], slots[0, 0]
    return result


@pytest.mark.parametrize("mutate,expected", [
    (output_only, "scaffold_identical"),
    (input_only, "decoder_left_input_unmodified"),
    (input_and_output_identically, "scaffold_identical"),
    (input_and_output_identically, "decoder_left_input_unmodified"),
    (slots_in_place, "decoder_left_input_unmodified")],
    ids=["output", "input", "both-scaffold", "both-unmodified", "slots"])
@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_a_decoder_that_mutates_what_it_was_given_or_returned_is_caught(
        harness, monkeypatch, route, mutate, expected):
    services = with_production_tolerances(harness.services)
    model = services.build_route(route).eval()
    corrupting(model, monkeypatch, mutate)
    report = decode(services, route, model)
    assert not report["passed"] and expected in report["failures"], report["failures"]


def test_slots_are_checked_against_the_site_mapping_not_just_counted(harness, monkeypatch):
    services = with_production_tolerances(harness.services)
    model = services.build_route(ROUTE_FULL).eval()
    real = services.collate

    def reversed_slots(route, views):
        batch = real(route, views)
        return dataclasses.replace(batch, query_slots=np.ascontiguousarray(
            batch.query_slots[:, ::-1]))
    monkeypatch.setattr(services, "collate", reversed_slots)
    report = decode(services, ROUTE_FULL, model)
    assert report["checks"]["slots_match_sites"] is False, (
        "ten slots in the wrong order still number ten")
    assert not report["passed"]


def test_a_batch_whose_scaffold_differs_from_the_serialization_is_caught(harness, monkeypatch):
    """The decoder is consistent with what it was handed, but what it was handed is wrong."""
    services = with_production_tolerances(harness.services)
    model = services.build_route(ROUTE_FULL).eval()
    real = services.collate

    def edited(route, views):
        batch = real(route, views)
        tokens = np.array(batch.tokens, copy=True)
        assert tokens[0, 0] != tokens[0, 1], "the fake scaffold must make this an actual edit"
        tokens[0, 0] = tokens[0, 1]
        return dataclasses.replace(batch, tokens=tokens)
    monkeypatch.setattr(services, "collate", edited)
    report = decode(services, ROUTE_FULL, model)
    assert report["checks"]["input_matches_serialization"] is False and not report["passed"]


def test_the_rescore_is_held_to_the_declared_tolerance_and_records_it(harness, monkeypatch):
    services = harness.services
    services.config["tolerances"] = {"sum_log_probability_atol": 5e-5,
                                     "sum_log_probability_rtol": 2e-6}
    model = services.build_route(ROUTE_FULL).eval()
    original = services.window_scores

    def shifted(route, model, windows, **kwargs):
        result = original(route, model, windows, **kwargs)
        return result if "base" in kwargs else result + 0.0004
    monkeypatch.setattr(services, "window_scores", shifted)
    report = decode(services, ROUTE_FULL, model)
    assert not report["passed"] and "fresh_rescoring_agrees" in report["failures"]
    assert report["rescore_tolerance"]["atol"] == 5e-5
    assert report["rescore_tolerance"]["rtol"] == 2e-6
    assert report["max_rescore_difference"] == pytest.approx(0.0004, abs=1e-4)
    assert report["max_rescore_excess_over_allowance"] > 0


@pytest.mark.parametrize("block", [
    None, {}, {"sum_log_probability_atol": 1e-3},
    {"sum_log_probability_atol": float("nan"), "sum_log_probability_rtol": 0.0},
    {"sum_log_probability_atol": 1e-5, "sum_log_probability_rtol": -1.0},
    {"sum_log_probability_atol": True, "sum_log_probability_rtol": 0.0},
    {"sum_log_probability_atol": "5e-5", "sum_log_probability_rtol": 0.0}])
def test_a_missing_or_invalid_declared_tolerance_is_refused_before_any_decoding(
        harness, block):
    services = harness.services
    services.config.pop("tolerances", None)
    if block is not None:
        services.config["tolerances"] = block
    model = services.build_route(ROUTE_FULL).eval()
    with pytest.raises(ValueError, match="tolerance"):
        decode(services, ROUTE_FULL, model)


def test_the_qualification_has_no_looser_constant_of_its_own():
    assert not hasattr(qualify, "DECODE_SCORE_ATOL")


# ---------------------------------------------------------------------------
# the qualifier command bootstraps the config it will actually load
# ---------------------------------------------------------------------------

def test_bootstrap_cli_prepares_the_root_of_the_default_config_it_is_given(tmp_path, monkeypatch):
    for name, root in (("base.json", "BASE_ROOT"), ("joint.json", "JOINT_ROOT")):
        (tmp_path / name).write_text(json.dumps({"artifacts": {"root": root}}), encoding="utf-8")
    prepared = []
    monkeypatch.setattr(env, "prepare_environment", lambda root: prepared.append(root) or {})
    monkeypatch.setattr(env, "require_environment_ordering", lambda record, label="": True)
    env.bootstrap_cli(tmp_path, [], default_config="joint.json")
    env.bootstrap_cli(tmp_path, ["--config", "base.json"], default_config="joint.json")
    env.bootstrap_cli(tmp_path, ["--artifact-root", "EXPLICIT"], default_config="joint.json")
    assert prepared == ["JOINT_ROOT", "BASE_ROOT", "EXPLICIT"]


def load_script():
    path = ROOT / "scripts" / "her2_routes_qualify_dplm.py"
    module_spec = importlib.util.spec_from_file_location("her2_routes_qualify_dplm_cli", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_the_qualify_script_bootstraps_its_own_joint_default(monkeypatch):
    script = load_script()
    assert script.DEFAULT_CONFIG.endswith("her2_generative_routes_joint.json")
    seen = []
    monkeypatch.setattr(env, "prepare_environment", lambda root: seen.append(root) or {})
    monkeypatch.setattr(env, "require_environment_ordering", lambda record, label="": True)
    script.bootstrap([])
    script.bootstrap(["--artifact-root", "EXPLICIT"])
    script.bootstrap(["--config", "configs/experiments/her2_generative_routes.json"])
    base = json.loads((ROOT / "configs/experiments/her2_generative_routes.json")
                      .read_text(encoding="utf-8"))
    assert seen == [JOINT["artifacts"]["root"], "EXPLICIT", base["artifacts"]["root"]]
    parser = script.build_parser()
    assert parser.parse_args(["--route", "dplm_150m_masked", "--revision", "joint_qual_x"]
                             ).config == script.DEFAULT_CONFIG
