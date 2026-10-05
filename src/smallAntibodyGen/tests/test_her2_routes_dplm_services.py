"""The native DPLM routes through the REAL route services: build, train, resume, profile,
distil, post-train and generate.

What is real here: ``FlightServices.train`` (its optimizer, accumulation, schedule,
checkpoints, resume), ``profile``, ``distil_teacher``, ``run_posttrain``,
``generation_endpoint``, ``generate.fill_batch``, ``tasks.dispatch`` and the checkpoint
protocol. What is replaced: exactly one seam -- ``dplm_policy.load_pinned_dplm`` is
monkeypatched to return a tiny random-init trunk, because the pinned pretrained files are
neither in this repository nor allowed to be fetched by a test. Production
``build_route`` always calls that loader; a test below proves it has no other way to a DPLM
model.

Nothing here says anything about the quality of a trained DPLM, the GPU fit of either
route, or pretrained numerical equivalence.
"""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_data
from smallAntibodyGen.experiments import her2_routes_campaign as campaign
from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_routes_dplm_pins as pins_lib
from smallAntibodyGen.experiments import her2_routes_generate as generate
from smallAntibodyGen.experiments import her2_routes_lineage as lineage
from smallAntibodyGen.experiments import her2_routes_pairs as pairs_lib
from smallAntibodyGen.experiments import her2_routes_posttrain as posttrain
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.experiments import her2_routes_services as services_mod
from smallAntibodyGen.experiments import her2_routes_tasks as tasks_lib
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_routes_conditions import BIN_HIGH, Condition, ConditionSpec
from smallAntibodyGen.experiments.her2_routes_sources import CENSOR_NONE, KIND_POINT, SCALE_PKD
from smallAntibodyGen.models import dplm_policy as dplm

ROUTE_FULL = services_mod.ROUTE_DPLM_150M
ROUTE_LORA = services_mod.ROUTE_DPLM_650M_LORA
HEAVY = "ACDEFGHIKLMNPQRSTVWY" * 2
LIGHT = "YWVTSRQPNMLKIHGFEDCA"


def tiny_native_config():
    return dplm.DPLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                           intermediate_size=64, max_position_embeddings=256,
                           hidden_dropout_prob=0.0)


def route_block(variant, **extra):
    block = {"variant": variant, "activation_checkpointing": False, "dropout": 0.0,
             "updates": 4, "seed": 5, "max_positions": 256,
             "objective": {"reconstruction": "diffusion"},
             "diffusion": {"num_timesteps": 8, "weighting": "constant", "decode_steps": 4,
                           "decoding_strategy": "reparam-uncond-deterministic-linear"},
             "gate_generation_strategy": "diffusion_reparam"}
    block.update(extra)
    return block


ADAFACTOR = {"name": "adafactor", "relative_step": False, "scale_parameter": False,
             "warmup_init": False, "beta1": None, "eps": [1e-30, 1e-3], "clip_threshold": 1.0,
             "decay_rate": -0.8, "weight_decay": 0.0}


def make_config():
    return {
        "routes": {
            "enabled": [ROUTE_FULL, ROUTE_LORA],
            ROUTE_FULL: route_block("tiny150", base_dtype="fp32", optimizer=dict(ADAFACTOR)),
            ROUTE_LORA: route_block(
                "tiny650", lora={"rank": 4, "alpha": 8, "dropout": 0.0},
                optimizer={"name": "adamw", "learning_rate": 1e-3, "betas": [0.9, 0.999],
                           "weight_decay": 0.0})},
        "dplm": {"pins": "dplm_pins.json"},
        "conditions": {"assays": ["a1"], "bin_dropout": 0.0, "assay_dropout": 0.0},
        "curriculum": {"kappa": 2.0,
                       "weights": {"reconstruction": 1.0, "local_effect": 1.0,
                                   "whole_window": 1.0, "heads": 0.5},
                       "shares": {"paired_reconstruction": 0.25, "measured_local_effects": 0.28125,
                                  "whole_window_generation": 0.28125,
                                  "observation_heads": 0.1875},
                       "family_mix": {"her2": 0.5, "other_functional": 0.25,
                                      "paired_replay": 0.25}},
        "optimization": {"learning_rate": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0,
                         "gradient_clip": 1.0, "warmup": 1, "microbatch": 2, "accumulation": 1,
                         "effective_batch": 2, "eval_every": 1, "save_every": 1,
                         "plateau_delta": 0.001, "plateau_patience_evaluations": 4,
                         "plateau_factor": 0.3, "max_plateau_reductions": 2},
        "storage": {"bytes_per_checkpoint": 1000, "min_free_bytes": 10 ** 9},
        "profile": {"probe_lr": 1e-3, "probe_seed": 7, "min_free_vram_mib": 1,
                    "max_peak_mib": 10 ** 6, "max_per_row_slowdown": 1000.0},
        "her2": {"wild_type_core": "WGGDGFYAMD", "assay": "a1"},
        "gates": {"generation_draws": 3},
        "decoding": {"seed": 1},
        "priors": {"shuffle_seed": 1, "distillation_updates": 2, "distillation_batch": 4,
                   "distillation_temperature": 0.5, "distillation_learning_rate": 1e-3},
        "posttrain": {"contract_version": 2, "methods": ["weighted_conditional_nll", "local_ipo"],
                      "updates": 2, "pairs_per_batch": 2, "reuse_cap": 3, "learning_rate": 1e-3,
                      "tau": 0.1, "beta": 0.1, "winner_likelihood": 0.1, "seed": 3},
        "confirmation": {"updates": 2, "seeds": [1, 2, 3], "bootstrap_seed": 1},
        "split": {"minimum_reserved": 100},
        "smoke": {"updates": 2},
    }


class Context:
    """The slice of ``RoutesContext`` the services touch."""

    def __init__(self, root, config, *, write_config=True):
        self.root = Path(root)
        self.config = config
        self.run_root = self.root / "run"
        self.artifact_root = self.root
        self.repository_root = self.root
        self.config_path = self.root / "config.json"
        self.revision = "rev01"
        self.campaign_id = "probe"
        self.root.mkdir(parents=True, exist_ok=True)
        if write_config:
            self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def path(self, *parts):
        return self.run_root.joinpath(*[str(part) for part in parts])

    def artifact(self, logical):
        return self.root / str(logical)

    def repository(self, logical):
        return self.root / str(logical)

    def condition_spec(self):
        block = self.config["conditions"]
        return ConditionSpec(assays=tuple(block["assays"]),
                             bin_dropout=float(block["bin_dropout"]),
                             assay_dropout=float(block["assay_dropout"]))


def write_pins(root, *, revision="a" * 40):
    files = {name: {"bytes": 10, "sha256": "0" * 64} for name in pins_lib.FILES}
    document = {"schema": pins_lib.PINS_SCHEMA, "variants": {
        variant: {"repo": f"local/{variant}", "revision": revision,
                  "logical_dir": f"models/{variant}", "files": copy.deepcopy(files),
                  "architecture": {}} for variant in ("tiny150", "tiny650")}}
    target = Path(root) / "dplm_pins.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def fake_scaffold():
    heavy = "A" * 98 + "WGGDGFYAMD" + "YWGQGTLVTVSS"
    assert len(heavy) == her2_data.HEAVY_LENGTH
    return her2_data.Scaffold(heavy=heavy, light="D" * her2_data.LIGHT_LENGTH)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config = make_config()
    write_pins(tmp_path)
    context = Context(tmp_path, config)
    loads = []

    def fake_loader(weights_dir, *, pins, variant, device="cpu", storage_dtype=torch.float32,
                    dropout=None):
        loads.append({"directory": Path(weights_dir), "variant": variant, "device": device,
                      "dtype": storage_dtype, "pins": pins, "dropout": dropout})
        return dplm.NativeDPLM.random_init(tiny_native_config(), seed=3, std=0.3, label=variant,
                                           storage_dtype=storage_dtype)
    monkeypatch.setattr(dplm, "load_pinned_dplm", fake_loader)
    services = services_mod.FlightServices(context, device="cpu")
    services._cache["scaffold"] = fake_scaffold()
    return types.SimpleNamespace(services=services, context=context, config=config,
                                 loads=loads, root=tmp_path)


def reconstruction_stream(services):
    """A deterministic-in-(step, rng) stream of unlabelled paired reconstruction rows."""
    constructs = [serial.Construct(heavy=HEAVY, light=LIGHT, sites=((serial.HEAVY, 3),),
                                   source="synthetic", record_id=f"r{index}")
                  for index in range(2)]

    def stream(step, micro, rng):
        views = data_lib.reconstruction_views(
            constructs, rng=rng, mask_fraction=0.15, condition_spec=services.condition_spec,
            family="display", route="masked", max_positions=256)
        return tasks_lib.TaskDraw(task=data_lib.TASK_RECONSTRUCTION,
                                  family=data_lib.FAMILY_REPLAY, views=views,
                                  source="synthetic")
    return stream


def train(harness, route, *, updates, directory, resume=False, evaluate=None, services=None):
    services = services or harness.services
    return services.train(route, stream=reconstruction_stream(services), updates=updates,
                          seed=5, directory=directory, resume=resume, evaluate=evaluate)


# ---------------------------------------------------------------------------
# construction and dispatch
# ---------------------------------------------------------------------------

def test_the_original_three_routes_are_unchanged_and_the_dplm_routes_are_optional():
    assert services_mod.ROUTES == ("ablang2_masked_editor", "piggen_causal_infiller",
                                   "dwjs_energy")
    assert set(services_mod.OPTIONAL_ROUTES) == {ROUTE_FULL, ROUTE_LORA}
    assert set(services_mod.KNOWN_ROUTES) == set(services_mod.ROUTES) | set(
        services_mod.OPTIONAL_ROUTES)
    assert not set(services_mod.OPTIONAL_ROUTES) & set(services_mod.ROUTES)


def test_build_route_reaches_a_dplm_model_only_through_the_pinned_loader(harness):
    services = harness.services
    lora = services.build_route(ROUTE_LORA, seed=1)
    full = services.build_route(ROUTE_FULL, seed=1)
    assert isinstance(lora, dplm.ConditionedDPLM) and lora.is_lora
    assert isinstance(full, dplm.ConditionedDPLM) and not full.is_lora
    assert [call["variant"] for call in harness.loads] == ["tiny650", "tiny150"]
    assert harness.loads[0]["directory"] == harness.root / "models" / "tiny650"
    assert harness.loads[0]["pins"]["variants"]["tiny650"]["revision"] == "a" * 40
    assert services.pad_id(ROUTE_LORA) == serial.ABLANG_PAD_ID == 21, (
        "collation stays in public ids; a native pad of 1 would collide with AbLang M")
    heads = services.build_heads(ROUTE_LORA, lora)
    assert heads is not None and lora.hidden_size == 32
    with pytest.raises(ValueError, match="Unknown route"):
        services.build_route("not_a_route")


def test_there_is_no_random_init_or_override_path_in_production(harness, monkeypatch):
    """A loader failure is a failure. An ``architecture_override`` block is never read."""
    harness.config["routes"][ROUTE_FULL]["architecture_override"] = {"hidden_size": 16}

    def refusing(*args, **kwargs):
        raise ValueError("the pinned files are absent")
    monkeypatch.setattr(dplm, "load_pinned_dplm", refusing)
    with pytest.raises(ValueError, match="pinned files are absent"):
        harness.services.build_route(ROUTE_FULL)
    with pytest.raises(ValueError, match="pinned files are absent"):
        harness.services.build_route(ROUTE_LORA)


def test_the_base_rests_in_fp16_only_for_lora_on_cuda(harness, monkeypatch):
    seen = []

    def recording(weights_dir, *, pins, variant, device, storage_dtype, dropout):
        seen.append((variant, device, storage_dtype))
        raise RuntimeError("stop after recording the request")
    monkeypatch.setattr(dplm, "load_pinned_dplm", recording)
    for device, route in (("cuda", ROUTE_LORA), ("cuda", ROUTE_FULL), ("cpu", ROUTE_LORA)):
        harness.services.device = device
        with pytest.raises(RuntimeError, match="stop after recording"):
            harness.services.build_route(route)
    assert seen == [("tiny650", "cuda", torch.float16), ("tiny150", "cuda", torch.float32),
                    ("tiny650", "cpu", torch.float32)]


def test_max_positions_beyond_the_model_is_refused(harness):
    harness.config["routes"][ROUTE_FULL]["max_positions"] = 4096
    with pytest.raises(ValueError, match="max_positions exceeds"):
        harness.services.build_route(ROUTE_FULL)


# ---------------------------------------------------------------------------
# the optimizer: one builder, Adafactor with fixed-LR options
# ---------------------------------------------------------------------------

def test_the_legacy_routes_get_exactly_the_adamw_they_always_got(harness):
    services = harness.services
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = services.build_optimizer("ablang2_masked_editor", [parameter], lr=3e-5,
                                         betas=(0.9, 0.999), weight_decay=0.01)
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults["lr"] == 3e-5 and optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 0.01
    bare = services.build_optimizer("piggen_causal_infiller", [parameter], lr=1e-5)
    assert type(bare) is torch.optim.AdamW and bare.defaults["weight_decay"] == 0.01
    assert services.route_learning_rate("ablang2_masked_editor") == 1e-3
    assert services.route_learning_rate(ROUTE_LORA) == 1e-3       # named in its own block
    harness.config["routes"][ROUTE_LORA]["optimizer"]["learning_rate"] = 7e-4
    assert services.route_learning_rate(ROUTE_LORA) == 7e-4


def test_adafactor_uses_fixed_lr_options_and_never_allocates_a_full_second_moment(harness):
    services = harness.services
    model = services.build_route(ROUTE_FULL, seed=1)
    heads = services.build_heads(ROUTE_FULL, model)
    parameters = [p for p in list(model.parameters()) + list(heads.parameters())
                  if p.requires_grad]
    optimizer = services.build_optimizer(ROUTE_FULL, parameters, lr=1e-3)
    assert type(optimizer).__name__ == "Adafactor"
    assert type(optimizer).__module__ == "transformers.optimization"
    defaults = optimizer.defaults
    assert defaults["relative_step"] is False and defaults["scale_parameter"] is False
    assert defaults["warmup_init"] is False and defaults["beta1"] is None
    assert defaults["lr"] == 1e-3
    tokens = torch.tensor([serial.ablang_token_ids(
        serial.Construct(heavy=HEAVY, light=LIGHT, sites=((serial.HEAVY, 3),), source="s",
                         record_id="x"), masked_sites=((serial.HEAVY, 3),))])
    for _ in range(2):
        optimizer.zero_grad()
        (-model.canonical_log_probs(tokens, torch.tensor([[4]]))[..., 0].sum()).backward()
        optimizer.step()
    state_numel = 0
    for parameter in parameters:
        state = optimizer.state[parameter]
        if parameter.grad is None:
            assert not state, "a parameter that never received a gradient has no state"
            continue
        assert "exp_avg" not in state, "beta1=None means no first moment"
        if parameter.dim() >= 2:
            assert "exp_avg_sq_row" in state and "exp_avg_sq_col" in state
            assert "exp_avg_sq" not in state, "a matrix must not hold a full second moment"
        state_numel += sum(int(value.numel()) for value in state.values()
                           if isinstance(value, torch.Tensor) and value.dim() > 0)
    assert state_numel < 0.25 * sum(p.numel() for p in parameters)


def test_the_optimizer_block_is_validated_and_never_falls_back_to_adam(harness, monkeypatch):
    services = harness.services
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    block = harness.config["routes"][ROUTE_FULL]["optimizer"]
    block["relative_step"] = True
    with pytest.raises(ValueError, match="relative_step must be false"):
        services.build_optimizer(ROUTE_FULL, [parameter], lr=1e-3)
    block["relative_step"] = False
    block["momentum_magic"] = 1
    with pytest.raises(ValueError, match="unknown Adafactor key"):
        services.build_optimizer(ROUTE_FULL, [parameter], lr=1e-3)
    del block["momentum_magic"]
    block["name"] = "lion"
    with pytest.raises(ValueError, match="is not adamw/adafactor"):
        services.build_optimizer(ROUTE_FULL, [parameter], lr=1e-3)
    block["name"] = "adafactor"
    monkeypatch.setitem(sys.modules, "transformers.optimization", None)
    with pytest.raises(ValueError, match="not importable"):
        services.build_optimizer(ROUTE_FULL, [parameter], lr=1e-3)


# ---------------------------------------------------------------------------
# training: the real loop, diffusion reconstruction, checkpoints, resume
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_the_real_train_loop_runs_the_diffusion_objective_and_stores_the_right_state(
        harness, monkeypatch, route):
    calls = {"diffusion": 0, "masked_window": 0}
    original_diffusion = dplm.ConditionedDPLM.diffusion_loss
    original_window = dplm.ConditionedDPLM.canonical_log_probs

    def counting_diffusion(self, *args, **kwargs):
        calls["diffusion"] += 1
        return original_diffusion(self, *args, **kwargs)

    def counting_window(self, *args, **kwargs):
        calls["masked_window"] += 1
        return original_window(self, *args, **kwargs)
    monkeypatch.setattr(dplm.ConditionedDPLM, "diffusion_loss", counting_diffusion)
    monkeypatch.setattr(dplm.ConditionedDPLM, "canonical_log_probs", counting_window)
    if route == ROUTE_FULL:
        # the Adafactor route must never construct an AdamW
        monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("an AdamW was built for an Adafactor route")))
    directory = harness.root / "trained"
    record = train(harness, route, updates=3, directory=directory)
    assert record["updates"] == 3
    assert calls == {"diffusion": 3, "masked_window": 0}, (
        "reconstruction must use sampled-timestep corruption, not the fixed mask fraction")
    assert all(np.isfinite(entry["loss"]) and entry["gradient_norm"] > 0
               for entry in record["history"])
    assert "reconstruction" in record["history"][0]["terms"]
    assert record["identity"]["native_sha256"] == harness.services.native_identity(route)
    payload = torch.load(str(directory / "last.pt"), map_location="cpu", weights_only=False)
    fmt = payload["model_format"]["format"]
    if route == ROUTE_LORA:
        assert fmt == dplm.FORMAT_ADAPTER
        assert all("lora_" in key or key.startswith("condition_") for key in payload["model"])
        assert not any(key.startswith("native.esm.embeddings") for key in payload["model"])
        assert any(float(value.abs().sum()) > 0 for key, value in payload["model"].items()
                   if key.endswith("lora_B"))
    else:
        assert fmt == dplm.FORMAT_FULL
        assert any(key.startswith("native.esm.embeddings") for key in payload["model"])
    assert float(payload["model"]["condition_bin.weight"].abs().sum()) > 0
    paths.canonical_json(record)              # the run record survives the stage writer


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_resume_reproduces_the_uninterrupted_trajectory_step_for_step(harness, route):
    """Adafactor (150M) and adapter-only AdamW (650M LoRA) both resume to the same run."""
    full = train(harness, route, updates=4, directory=harness.root / "full")
    train(harness, route, updates=2, directory=harness.root / "interrupted")
    resumed = train(harness, route, updates=4, directory=harness.root / "interrupted",
                    resume=True)
    assert resumed["resume"] is not None and resumed["resume"]["step"] == 2
    report = checkpoints.resumed_trajectory_report(full["history"], resumed["history"],
                                                   tolerance=1e-6)
    assert report["compared_steps"] == 4 and report["identical"], report["offenders"]
    assert resumed["exposure"]["totals"]["microbatch_rows"] == \
        full["exposure"]["totals"]["microbatch_rows"]
    final_full = torch.load(str(harness.root / "full" / "last.pt"), map_location="cpu",
                            weights_only=False)
    final_resumed = torch.load(str(harness.root / "interrupted" / "last.pt"),
                               map_location="cpu", weights_only=False)
    assert set(final_full["model"]) == set(final_resumed["model"])
    for key, value in final_full["model"].items():
        assert torch.allclose(value, final_resumed["model"][key], atol=1e-6), key


def test_a_resume_under_a_changed_native_setting_is_refused(harness):
    directory = harness.root / "identity"
    train(harness, ROUTE_LORA, updates=2, directory=directory)
    changed = copy.deepcopy(harness.config)
    changed["routes"][ROUTE_LORA]["lora"]["rank"] = 8
    other = services_mod.FlightServices(Context(harness.root, changed, write_config=False),
                                        device="cpu")
    other._cache["scaffold"] = fake_scaffold()
    with pytest.raises(ValueError, match="different identity"):
        train(harness, ROUTE_LORA, updates=3, directory=directory, resume=True, services=other)


def test_the_native_identity_binds_pins_lora_optimizer_and_diffusion_settings(harness):
    base = harness.services.native_identity(ROUTE_LORA)
    assert base is not None and harness.services.native_identity("ablang2_masked_editor") is None
    identity = harness.services.training_identity("ablang2_masked_editor", arm="full", seed=1)
    assert "native_sha256" not in identity, "the original routes' identity must not change"

    counter = iter(range(1000))

    def variant(mutate, *, pins_revision=None):
        config = copy.deepcopy(harness.config)
        mutate(config["routes"][ROUTE_LORA])
        root = harness.root / f"variant{next(counter)}"
        root.mkdir()
        write_pins(root, revision=pins_revision or "a" * 40)
        return services_mod.FlightServices(Context(root, config), device="cpu").native_identity(
            ROUTE_LORA)
    assert variant(lambda block: None) == base
    assert variant(lambda block: block["lora"].update(rank=8)) != base
    assert variant(lambda block: block["lora"].update(alpha=64)) != base
    assert variant(lambda block: block["optimizer"].update(learning_rate=5e-4)) != base
    assert variant(lambda block: block["diffusion"].update(decode_steps=9)) != base
    assert variant(lambda block: block.update(base_dtype="fp32")) != base
    assert variant(lambda block: None, pins_revision="b" * 40) != base


# ---------------------------------------------------------------------------
# profile, evidence, generation
# ---------------------------------------------------------------------------

def test_the_profiler_builds_its_optimizer_through_the_one_builder(harness, monkeypatch):
    calls = []
    original = services_mod.FlightServices.build_optimizer

    def recording(self, route, parameters, *, lr, **options):
        calls.append((route, lr, sorted(options)))
        return original(self, route, parameters, lr=lr, **options)
    monkeypatch.setattr(services_mod.FlightServices, "build_optimizer", recording)
    services = harness.services
    draw = reconstruction_stream(services)(1, 0, np.random.default_rng(0))
    profile = services.profile(ROUTE_FULL, microbatches=(2,), steps=2,
                               draws={data_lib.TASK_RECONSTRUCTION: draw})
    assert calls == [(ROUTE_FULL, 1e-3, [])]
    assert profile["accepted_microbatches"] == [2] and profile["chosen_microbatch"] == 2
    assert profile["rows"][0]["failure"] is None


@pytest.mark.parametrize("route", [ROUTE_FULL, ROUTE_LORA])
def test_native_evidence_measures_parity_gradients_and_a_decode(harness, route):
    record = harness.services.native_evidence(route)
    assert record["parity_at_zero"]["bit_identical"]
    assert record["reference_parity_nonzero_tables"]["meaningful"]
    assert record["gradient"]["trunk_reached"] is True
    assert record["gradient"]["head_gradient_reaches_trunk"] is True
    assert record["decode_probe"]["canonical_only"] and \
        record["decode_probe"]["immutable_scaffold_verified"]
    assert record["decode_probe"]["trajectory_logprob"] is None
    parameters = record["parameters"]
    if route == ROUTE_LORA:
        assert record["lora_noop_parity"]["max_abs_difference"] == 0.0
        assert parameters["trainable"] < parameters["total"] and parameters["frozen"] > 0
        assert record["gradient"]["frozen_parameters"] > 0
        assert record["gradient"]["parameters_with_gradient"] > 0
    else:
        assert parameters["frozen"] == 0 and "lora" not in record
        assert "frozen_parameters" not in record["gradient"]
    json.dumps(campaign._plain(record))


def test_generation_uses_the_native_decoder_and_never_moves_the_scaffold(harness, monkeypatch):
    services = harness.services
    model = services.build_route(ROUTE_FULL, seed=1)
    model.eval()
    captured = []
    original = dplm.ConditionedDPLM.iterative_decode

    def capture(self, tokens, query_slots, **kwargs):
        result = original(self, tokens, query_slots, **kwargs)
        captured.append((torch.as_tensor(tokens).clone(), torch.as_tensor(query_slots).clone(),
                         result))
        return result
    monkeypatch.setattr(dplm.ConditionedDPLM, "iterative_decode", capture)
    condition = Condition(family="affinity", bin=BIN_HIGH, assay="a1")
    sites = serial.her2_core_edit_map()
    result = generate.fill_batch(services, ROUTE_FULL, model, draws=3, sites=sites,
                                 condition=condition, strategy=generate.ORDER_DIFFUSION, seed=4,
                                 batch_size=2)
    assert result.strategy == generate.ORDER_DIFFUSION == "diffusion_reparam"
    assert len(result.windows) == 3 and len(captured) == 2
    assert all(len(window) == 10 and set(window) <= set(her2_data.CANONICAL)
               for window in result.windows)
    produced = []
    for tokens, slots, decoded in captured:
        native_in = dplm.ablang_to_dplm(tokens)
        scaffold = torch.ones_like(native_in, dtype=torch.bool)
        rows = torch.arange(native_in.shape[0])[:, None].expand_as(slots)
        scaffold[rows, slots] = False
        assert torch.equal(decoded.native_tokens[scaffold], native_in[scaffold])
        assert slots[0].tolist() == list(range(99, 109)), "HCDR3 sits at 1 + 98 .. 1 + 107"
        assert set(decoded.native_tokens[rows, slots].flatten().tolist()) <= set(
            dplm.DPLM_CANONICAL_IDS)
        produced.extend("".join(her2_data.CANONICAL[int(i)] for i in row)
                        for row in decoded.canonical)
    assert tuple(produced) == result.windows
    steps = model.diffusion.decode_steps
    diagnostics = result.diagnostics
    assert diagnostics["denoiser_forward_calls_decoding"] == 2 * steps
    assert diagnostics["final_rescore_forward_calls"] == 2 * len(sites)
    assert result.forward_calls == 2 * steps + 2 * len(sites)
    assert diagnostics["trajectory_logprob"] is None
    assert diagnostics["terminal_distribution_log_probability"] is None
    assert result.log_probabilities.shape == (3,) and bool(np.all(result.log_probabilities <= 0))
    again = services.window_scores(ROUTE_FULL, model, result.windows, condition=condition)
    assert np.allclose(again.detach().numpy(), result.log_probabilities, atol=1e-5)
    accounting = result.accounting()
    assert accounting["forward_calls"] == result.forward_calls and accounting["draws"] == 3


@pytest.mark.parametrize("sites", [
    ((serial.HEAVY, 100), (serial.HEAVY, 101)),
    ((serial.HEAVY, 101), (serial.HEAVY, 100)),
    ((serial.LIGHT, 30), (serial.HEAVY, 100), (serial.LIGHT, 31)),
], ids=["partial", "reversed", "mixed_light"])
def test_diffusion_rescore_scores_a_non_core_site_set_in_declared_order(harness, sites):
    services = harness.services
    model = services.build_route(ROUTE_FULL, seed=1)
    model.eval()
    condition = Condition(family="affinity", bin=BIN_HIGH, assay="a1")
    result = generate.diffusion_fill(services, ROUTE_FULL, model, draws=3, sites=sites,
                                     condition=condition, seed=4, batch_size=2)
    assert all(len(window) == len(sites) for window in result.windows)
    assert result.log_probabilities.shape == (3,)
    assert bool(np.all(result.log_probabilities <= 0))
    diagnostics = result.diagnostics
    assert diagnostics["final_rescore_forward_calls"] == 2 * len(sites)
    assert result.forward_calls == (diagnostics["denoiser_forward_calls_decoding"]
                                    + diagnostics["final_rescore_forward_calls"])
    again = services.window_scores(ROUTE_FULL, model, result.windows, condition=condition,
                                   sites=sites)
    assert np.allclose(again.detach().numpy(), result.log_probabilities, atol=1e-5)
    with pytest.raises(ValueError, match="does not fill"):
        services.window_scores(ROUTE_FULL, model, ["A"], condition=condition, sites=sites)


@pytest.mark.parametrize("sites", [
    ((serial.HEAVY, 100), (serial.HEAVY, 101)),
    ((serial.HEAVY, 101), (serial.HEAVY, 100)),
    ((serial.LIGHT, 30), (serial.HEAVY, 100), (serial.LIGHT, 31)),
], ids=["partial", "reversed", "mixed_light"])
def test_diffusion_rescore_uses_the_seed_window_scaffold_it_decoded_on(harness, sites):
    services = harness.services
    model = services.build_route(ROUTE_FULL, seed=1)
    model.eval()
    condition = Condition(family="affinity", bin=BIN_HIGH, assay="a1")
    wild = services.config["her2"]["wild_type_core"]
    seed_window = "AGGDGFYAMD"
    assert seed_window != wild and seed_window[0] != wild[0], "the seed must move fixed H98"
    result = generate.diffusion_fill(services, ROUTE_FULL, model, draws=3, sites=sites,
                                     condition=condition, seed=4, batch_size=2,
                                     seed_window=seed_window)
    on_seed = services.window_scores(ROUTE_FULL, model, result.windows, condition=condition,
                                     sites=sites, base=seed_window).detach().numpy()
    on_wild = services.window_scores(ROUTE_FULL, model, result.windows, condition=condition,
                                     sites=sites).detach().numpy()
    assert np.allclose(on_seed, result.log_probabilities, atol=1e-5)
    assert not np.allclose(on_wild, result.log_probabilities, atol=1e-5), \
        "the wild-type scaffold would score a different construct than the one decoded"
    assert result.forward_calls == (result.diagnostics["denoiser_forward_calls_decoding"]
                                    + 2 * len(sites))


def test_the_diffusion_strategy_refuses_what_it_cannot_honour(harness):
    services = harness.services
    model = services.build_route(ROUTE_FULL, seed=1)
    condition = Condition(family="affinity", bin=BIN_HIGH, assay="a1")
    sites = serial.her2_core_edit_map()
    with pytest.raises(ValueError, match="no temperature"):
        generate.fill_batch(services, ROUTE_FULL, model, draws=1, sites=sites,
                            condition=condition, strategy=generate.ORDER_DIFFUSION, seed=1,
                            temperature=0.5)
    with pytest.raises(ValueError, match="no iterative diffusion decoder"):
        generate.fill_batch(services, "ablang2_masked_editor", object(), draws=1, sites=sites,
                            condition=condition, strategy=generate.ORDER_DIFFUSION, seed=1)
    assert services.supports_diffusion_decoding(ROUTE_FULL)
    assert not services.supports_diffusion_decoding("ablang2_masked_editor")


def test_the_gate_endpoint_draws_natively_for_the_trained_model_and_its_backbone(harness):
    services = harness.services
    directory = harness.root / "endpoint"
    train(harness, ROUTE_LORA, updates=2, directory=directory)
    services._cache["assessor"] = {
        "score": lambda windows: [float(len(set(window))) for window in windows],
        "fit": {"parameters": 0, "objective_evaluations": 0, "train_cross_entropy": 0.0},
        "independence": "stub"}
    checkpoint = directory / "last.pt"
    result = services.generation_endpoint(ROUTE_LORA, checkpoint, draws=3, seed=2)
    steps = harness.config["routes"][ROUTE_LORA]["diffusion"]["decode_steps"]
    for label in ("trained", "backbone"):
        arm = result["arms"][label]
        assert arm["draws"] == 3
        assert arm["accounting"]["forward_calls"] == steps + 10, (
            "one batch: decode_steps denoiser forwards + one fixed-order rescoring pass")
    assert result["arms"]["trained"]["checkpoint"]["trained"] is True
    assert result["arms"]["backbone"]["checkpoint"]["trained"] is False
    # the checkpoint really loaded: the trained arm is not the backbone under another name
    trained, _heads, _record = services.load_for_scoring(ROUTE_LORA, checkpoint)
    backbone, _heads, _record = services.load_for_scoring(ROUTE_LORA, None)
    tokens = torch.tensor([serial.ablang_token_ids(
        serial.Construct(heavy=HEAVY, light=LIGHT, sites=((serial.HEAVY, 3),), source="s",
                         record_id="x"), masked_sites=((serial.HEAVY, 3),))])
    with torch.no_grad():
        moved = (trained.canonical_log_probs(tokens, torch.tensor([[4]]),
                                             condition_indices=[[0, 3, 1]])
                 - backbone.canonical_log_probs(tokens, torch.tensor([[4]]),
                                                condition_indices=[[0, 3, 1]])).abs().max()
    assert float(moved) > 1e-6
    with pytest.raises(ValueError, match="is absent"):
        services.load_for_scoring(ROUTE_LORA, harness.root / "absent.pt")


# ---------------------------------------------------------------------------
# priors -> posttrain lineage through the real distil and post-training code
# ---------------------------------------------------------------------------

PANEL = [{"window": "WGGDGFYAMD", "value": 8.0}, {"window": "WGGDGFYAMA", "value": 7.0},
         {"window": "AGGDGFYAMD", "value": 6.0}, {"window": "WGGDGFYAMV", "value": 5.0}]
BANK = ["WGGDGFYAMD", "AGGDGFYAMD", "WGGDGFYAMA", "WGGDGFYAMV", "WAGDGFYAMD"]


class FakeTeacher:
    def score(self, windows):
        return [float(sum(ord(character) for character in window) % 7) for window in windows]


def produce_bakeoff_best(harness, route):
    directory = harness.context.path("bakeoff", route)
    train(harness, route, updates=2, directory=directory,
          evaluate=lambda model, heads, step: 1.0 / (step + 1))
    assert (directory / "best.pt").is_file()
    return directory / "best.pt"


def complete_stage(context, stage, **fields):
    campaign.write_stage_record(context, stage, dict({"status": "completed"}, **fields))


def test_distillation_starts_from_the_bakeoff_and_posttraining_inherits_the_distilled_model(
        harness, monkeypatch):
    services, context = harness.services, harness.context
    best = produce_bakeoff_best(harness, ROUTE_FULL)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("an AdamW was built for an Adafactor route")))
    result = services.distil_teacher(ROUTE_FULL, FakeTeacher(), bank=BANK, panel=PANEL)
    assert result["lineage"]["source"] == lineage.SOURCE_BAKEOFF
    assert result["lineage"]["path"] == str(best)
    assert result["lineage"]["sha256"] == paths.sha256_file(best)
    assert result["distillation"]["distilled"] is True
    distilled = context.path("priors", ROUTE_FULL, "distilled.pt")
    assert distilled.is_file() and result["checkpoint"] == str(distilled)
    # priors completed with a distilled checkpoint -> post-training starts from it
    complete_stage(context, "priors")
    paths.write_json(context.path("priors.json"), {
        "route": ROUTE_FULL, "teachers": {"functional_selection": {"qualified": True}},
        "distillations": {"functional_selection": {"checkpoint": str(distilled)}}})
    start = services.checkpoint_lineage("posttrain", ROUTE_FULL)
    assert start["source"] == lineage.SOURCE_PRIORS and start["fallback"] is False
    assert start["path"] == str(distilled)
    model, _heads, record = services.load_for_scoring(ROUTE_FULL, start["path"])
    assert record["trained"] is True and record["training_identity"]["arm"] == "distilled"


def test_a_missing_bakeoff_checkpoint_stops_priors_instead_of_scoring_a_fresh_backbone(harness):
    with pytest.raises(lineage.LineageError, match="bake-off best checkpoint"):
        harness.services.distil_teacher(ROUTE_FULL, FakeTeacher(), bank=BANK, panel=PANEL)
    assert not harness.context.path("priors", ROUTE_FULL, "distilled.pt").exists()


def make_pairs(services, *, chosen_value=8.0, rejected_value=7.0):
    """Four one-site pairs that state the measurement they came from.

    Each carries its own assay, target, family, scale, full masked-context digest and
    measured-target digest, as a pair from ``her2_routes_pairs.build_measured_pairs`` does; the
    stage reads these and defaults none of them. The endpoints are real construct summaries
    with one exact pKD replicate each, and every digest is the one those endpoints produce, so
    ``require_pair_metadata`` can re-derive them. The four backgrounds differ OUTSIDE the edited
    coordinate, so they are four distinct masked contexts.
    """
    scaffold = services.scaffold()
    site = ((serial.HEAVY, 98 + 9),)
    pairs = []
    for index in range(4):
        def build(core, tag, value):
            construct = serial.her2_construct(scaffold, core, record_id=f"{tag}{index}")
            construct = construct.with_residues((("H", 40 + index),), "W")
            replicate = data_lib.ReplicateMeasurement(
                value=value, lower=value, upper=value, censoring=CENSOR_NONE, replicate="r1",
                measurement_id=f"{tag}{index}")
            return data_lib.ConstructSummary(
                construct=construct, study="synthetic", assay="a1", target="HER2",
                parent="synthetic::parent", family="affinity", kind=KIND_POINT,
                measurements=(replicate,), region=site, partition=data_lib.PARTITION_TRAIN,
                scale=SCALE_PKD, source="synthetic")
        chosen = build("WGGDGFYAMD", "c", float(chosen_value))
        rejected = build("WGGDGFYAMA", "r", float(rejected_value))
        heavy, light = pairs_lib.masked_context(chosen.construct, site)
        background = pairs_lib.background_digest(
            heavy=heavy, light=light, assay="a1", target="HER2", family="affinity",
            scale=SCALE_PKD, sites=site)
        pairs.append({
            "id": f"p{index}", "sites": site, "measured": True, "source": "synthetic",
            "assay": "a1", "target": "HER2", "family": "affinity", "scale": SCALE_PKD,
            "chosen": chosen, "rejected": rejected,
            "margin": float(chosen_value) - float(rejected_value),
            "chosen_value": float(chosen_value), "rejected_value": float(rejected_value),
            "substitution": "H107:A->D", "background": background,
            "background_digest": background,
            "pair_digest": pairs_lib.pair_digest(
                background=background, chosen_residues=chosen.construct.residues(site),
                rejected_residues=rejected.construct.residues(site)),
            "target_digest": pairs_lib.target_digest(
                background=background, chosen=chosen, rejected=rejected, sites=site)})
    return pairs


def test_run_posttrain_returns_the_shape_the_stage_wrapper_reads_and_names_its_selection(
        harness, monkeypatch):
    """HR-02 on the REAL ``run_posttrain`` return shape, then through the real stage wrapper."""
    services, context = harness.services, harness.context
    best = produce_bakeoff_best(harness, ROUTE_LORA)
    complete_stage(context, "route_gate", winner=ROUTE_LORA)
    monkeypatch.setattr(services, "measured_pairs", lambda: make_pairs(services))
    monkeypatch.setattr(services, "development_panel_outcomes", lambda: PANEL)
    recorded = []
    original = services_mod.FlightServices.build_optimizer

    def recording(self, route, parameters, *, lr, **options):
        recorded.append(route)
        return original(self, route, parameters, lr=lr, **options)
    monkeypatch.setattr(services_mod.FlightServices, "build_optimizer", recording)
    result = services.run_posttrain()
    assert "pair_bank" not in result, "the bank is per method; there is no top-level one"
    assert set(result["methods"]) == {"weighted_conditional_nll", "local_ipo"}
    assert all("pair_bank" in block for block in result["methods"].values())
    assert recorded == [ROUTE_LORA, ROUTE_LORA], "each arm builds its optimizer through the builder"
    assert result["lineage"]["source"] == lineage.SOURCE_BAKEOFF
    assert result["lineage"]["fallback"] is True
    assert "priors stage has not completed" in result["lineage"]["reason"]
    assert result["lineage"]["path"] == str(best)
    assert result["selected_method"] in result["methods"]
    assert Path(result["selected_checkpoint"]).is_file()
    # the stage wrapper: previously KeyError: 'pair_bank'
    monkeypatch.setattr(campaign, "_guarded", lambda ctx, stage, body, **kw: body())
    monkeypatch.setattr(services, "run_posttrain", lambda heartbeat=None: result)
    record = campaign.stage_posttrain(context, services)
    assert set(record["pair_bank"]) == {"weighted_conditional_nll", "local_ipo"}
    assert record["selected_method"] == result["selected_method"]
    assert record["checkpoint"] == result["selected_checkpoint"]
    assert record["lineage"]["fallback"] is True
    written = paths.read_json(context.path("posttrain.json"))
    assert written["selection"]["best"] == result["selected_method"]
    # decoding and the report then resolve the post-trained checkpoint explicitly
    campaign.write_stage_record(context, "posttrain", dict(record))
    for stage in ("decoding", "report"):
        source = services.checkpoint_lineage(stage, ROUTE_LORA)
        assert source["source"] == lineage.SOURCE_POSTTRAIN
        assert source["path"] == result["selected_checkpoint"]
        model, _heads, loaded = services.load_for_scoring(ROUTE_LORA, source["path"])
        # the identity arm is method + contract version, so v1 weights cannot load as v2
        assert loaded["training_identity"]["arm"] == posttrain.arm_identity(
            result["selected_method"])


def test_run_posttrain_builds_the_dplm_reference_from_the_live_model_not_a_second_load(
        harness, monkeypatch):
    """The REAL service path: one checkpoint load per arm, and a shared-base reference."""
    services, context = harness.services, harness.context
    produce_bakeoff_best(harness, ROUTE_LORA)
    complete_stage(context, "route_gate", winner=ROUTE_LORA)
    monkeypatch.setattr(services, "measured_pairs", lambda: make_pairs(services))
    monkeypatch.setattr(services, "development_panel_outcomes", lambda: PANEL)
    loads, references = [], []
    load_original = services_mod.FlightServices.load_for_scoring
    reference_original = dplm.ConditionedDPLM.frozen_reference

    def counting_load(self, route, checkpoint, **kwargs):
        loads.append(route)
        return load_original(self, route, checkpoint, **kwargs)

    def recording_reference(self):
        reference = reference_original(self)
        references.append((self, reference))
        return reference
    monkeypatch.setattr(services_mod.FlightServices, "load_for_scoring", counting_load)
    monkeypatch.setattr(dplm.ConditionedDPLM, "frozen_reference", recording_reference)
    result = services.run_posttrain()
    arms = len(result["methods"])
    assert arms >= 2
    assert len(loads) == arms, "one base load per arm; the reference must not be a second load"
    assert len(references) == arms
    for live, reference in references:
        base = [key for key, p in dict(live.native.named_parameters()).items()
                if "lora_" not in key]
        theirs = dict(reference.native.named_parameters())
        mine = dict(live.native.named_parameters())
        assert base and all(theirs[key].data_ptr() == mine[key].data_ptr() for key in base), (
            "the reference duplicated the frozen base")
        assert not any(p.requires_grad for p in reference.parameters())
        assert not reference.training


def test_the_priors_stage_record_and_document_carry_what_lineage_needs(harness, monkeypatch):
    services, context = harness.services, harness.context
    distilled = context.path("priors", ROUTE_FULL, "distilled.pt")
    distilled.parent.mkdir(parents=True)
    distilled.write_bytes(b"x")
    result = {"teachers": {"functional_selection": {"qualified": True},
                           "neighbour_mean": {"qualified": False}},
              "skipped": {}, "route": ROUTE_FULL,
              "distillations": {"functional_selection": {
                  "checkpoint": str(distilled), "lineage": {"source": "bakeoff_best"}}}}
    monkeypatch.setattr(campaign, "_guarded", lambda ctx, stage, body, **kw: body())
    monkeypatch.setattr(services, "run_priors", lambda heartbeat=None: result)
    record = campaign.stage_priors(context, services)
    assert record["teachers"] == ["functional_selection", "neighbour_mean"]
    assert record["distilled_checkpoints"] == {"functional_selection": str(distilled)}
    assert paths.read_json(context.path("priors.json"))["teachers"]["neighbour_mean"] == {
        "qualified": False}
