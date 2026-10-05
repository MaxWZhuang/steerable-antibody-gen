"""Resume completeness, identity refusal, and the two schedules that must compose.

Four measured defects are pinned here, each of which passed every weight comparison:

* ``best_metric = inf`` made a completed run's record unserialisable, so a short fit
  with no evaluation lost its whole history at the end;
* the exposure ledger was checked for and never restored, so a resumed control arm
  restarted every counter at zero and the matched-exposure comparison then refused
  both arms;
* the CUDA and Python generator states were absent, so a resume diverged as soon as
  any dropout or sampling ran;
* ``LambdaLR`` recomputed the learning rate from the base value one step after the
  plateau handler multiplied it, so a recorded reduction never happened.

The trajectory check is the one that distinguishes "resumed to the same step" from
"resumed to the same run": a re-seeded mask RNG reaches a similar place by a
different path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments import her2_support_paths as paths


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Linear(4, 2)


def identity(**overrides):
    base = {"route": "ablang2_masked_editor", "arm": "full", "seed": 1,
            "config_sha256": "cfg", "snapshot_sha256": "snap", "external_sha256": "ext",
            "condition_sha256": "cond", "split_sha256": "split"}
    base.update(overrides)
    return checkpoints.training_identity(**base)


def state_with(seed=1, *, step=5, ledger=None):
    state = checkpoints.TrainingState(
        mask_rng=np.random.default_rng([seed, 1]),
        sampler_rng=np.random.default_rng([seed, 2]),
        decode_rng=np.random.default_rng([seed, 3]))
    state.step = int(step)
    state.sampler_position = int(step) * 4
    if ledger is not None:
        state.ledger = ledger
    return state


# --- the serialisation defect ---------------------------------------------

def test_no_evaluation_leaves_a_serialisable_best_metric():
    """``inf`` is not JSON. A run with no evaluation must still record."""
    state = checkpoints.TrainingState()
    assert state.best_metric is None
    document = state.document()
    assert document["best_metric"] is None
    paths.canonical_json(document)          # raises on inf
    with pytest.raises((TypeError, ValueError)):
        paths.canonical_json({"best_metric": float("inf")})


def test_the_first_evaluation_is_an_improvement_and_a_worse_one_is_not():
    state = checkpoints.TrainingState()
    assert state.improved(1.0, delta=0.001)
    state.best_metric = 1.0
    assert not state.improved(1.0, delta=0.001)
    assert not state.improved(1.0005, delta=0.001)
    assert state.improved(0.5, delta=0.001)
    assert not state.improved(None, delta=0.001)


# --- the exposure ledger --------------------------------------------------

def test_the_exposure_ledger_round_trips_through_the_checkpoint():
    ledger = data_lib.ExposureLedger()
    ledger.record(task=data_lib.TASK_LOCAL, source="s", rows=4, forwards=8, targets=40,
                  identities=["a", "b"], measured=True, update=True,
                  family=data_lib.FAMILY_HER2)
    ledger.record(task=data_lib.TASK_RECONSTRUCTION, source="oas", rows=4, forwards=4,
                  targets=30, identities=["c"], measured=False, update=True,
                  family=data_lib.FAMILY_REPLAY)
    restored = data_lib.ExposureLedger.restore(ledger.state())
    assert restored.totals == ledger.totals
    assert restored.by_task == ledger.by_task
    assert restored.by_family == ledger.by_family
    assert restored.constructs == ledger.constructs
    assert restored.document()["forwards_per_row"] == ledger.document()["forwards_per_row"]
    assert restored.family_shares() == ledger.family_shares()


def test_the_realised_family_shares_are_measured_not_declared():
    ledger = data_lib.ExposureLedger()
    ledger.record(task=data_lib.TASK_WHOLE, source="s", rows=6, forwards=6, targets=1,
                  identities=["a"], family=data_lib.FAMILY_HER2, update=True)
    ledger.record(task=data_lib.TASK_WHOLE, source="s", rows=2, forwards=2, targets=1,
                  identities=["b"], family=data_lib.FAMILY_OTHER, update=True)
    shares = ledger.family_shares()
    assert shares[data_lib.FAMILY_HER2] == pytest.approx(0.75)
    assert shares[data_lib.FAMILY_OTHER] == pytest.approx(0.25)


# --- save and restore -----------------------------------------------------

def test_a_full_checkpoint_restores_every_generator_and_the_ledger(tmp_path):
    model, other = TinyModel(), TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ledger = data_lib.ExposureLedger()
    ledger.record(task=data_lib.TASK_HEADS, source="s", rows=3, forwards=3, targets=3,
                  identities=["x"], measured=True, update=True)
    state = state_with(ledger=ledger)
    expected = state.mask_rng.random(3).tolist()
    state.mask_rng = np.random.default_rng([1, 1])          # rewind, then snapshot
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None,
                              optimizer=optimizer, scheduler=None, state=state,
                              identity=identity())
    fresh = state_with(step=0)
    restored, record = checkpoints.restore(target, model=other, heads=None, optimizer=None,
                                          scheduler=None, state=fresh, identity=identity())
    assert restored.step == 5
    assert restored.sampler_position == 20
    assert restored.ledger.totals["microbatch_rows"] == 3
    assert restored.mask_rng.random(3).tolist() == expected, (
        "the mask generator resumed at a different position, so the resumed run would draw "
        "different masks from the same step")
    assert set(record["restored"]) == set(checkpoints.REQUIRED_KEYS)
    assert record["rng"]["numpy_generators"] == ["decode_rng", "mask_rng", "sampler_rng"]


def test_a_partial_checkpoint_is_refused_rather_than_completed(tmp_path):
    model = TinyModel()
    state = state_with()
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state, identity=identity())
    payload = torch.load(str(target), weights_only=False)
    payload.pop("rng")
    torch.save(payload, target)
    with pytest.raises(ValueError, match="is missing 'rng'"):
        checkpoints.restore(target, model=TinyModel(), heads=None, optimizer=None,
                            scheduler=None, state=state_with(), identity=identity())


def test_a_checkpoint_from_a_different_identity_is_refused(tmp_path):
    model = TinyModel()
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state_with(), identity=identity())
    with pytest.raises(ValueError, match="config_sha256"):
        checkpoints.restore(target, model=TinyModel(), heads=None, optimizer=None,
                            scheduler=None, state=state_with(),
                            identity=identity(config_sha256="other"))


def test_the_identity_covers_the_data_not_only_the_code():
    left = identity()
    assert left["identity_sha256"] != identity(split_sha256="other")["identity_sha256"], (
        "a resume under a different SPLIT would attribute the recorded numbers to a "
        "population that did not produce them")
    assert left["identity_sha256"] != identity(condition_sha256="o")["identity_sha256"]
    assert left["identity_sha256"] != identity(external_sha256="o")["identity_sha256"]


def test_a_head_bearing_run_refuses_a_headless_checkpoint(tmp_path):
    model = TinyModel()
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state_with(), identity=identity())
    with pytest.raises(ValueError, match="no observation-head state"):
        checkpoints.restore(target, model=TinyModel(), heads=TinyModel(), optimizer=None,
                            scheduler=None, state=state_with(), identity=identity())


def test_the_write_is_atomic_and_leaves_no_temporary(tmp_path):
    model = TinyModel()
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state_with(), identity=identity())
    assert target.is_file()
    assert not list(tmp_path.glob("*.tmp")), "a leftover temporary means the replace did not run"


# --- the composed schedule ------------------------------------------------

def test_the_plateau_reduction_survives_the_next_scheduler_step():
    """The defect: LambdaLR recomputed from the base rate and undid the cut."""
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    state = checkpoints.TrainingState()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, checkpoints.warmup_plateau_lambda(state, warmup=2, plateau_factor=0.5))
    for _ in range(4):
        scheduler.step()
    warm = optimizer.param_groups[0]["lr"]
    assert warm == pytest.approx(0.01), "warm-up should have finished at the declared rate"
    state.plateau_reductions = 1
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005), (
        "the reduction did not survive the next step, which is the exact defect")
    state.plateau_reductions = 2
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)


def test_warmup_ramps_and_does_not_exceed_one():
    state = checkpoints.TrainingState()
    schedule = checkpoints.warmup_plateau_lambda(state, warmup=4, plateau_factor=0.5)
    values = [schedule(step) for step in range(8)]
    assert values[0] == pytest.approx(0.25)
    assert values[3] == pytest.approx(1.0)
    assert max(values) == pytest.approx(1.0)


# --- the trajectory check -------------------------------------------------

def test_the_trajectory_report_catches_a_resume_that_took_a_different_path():
    uninterrupted = [{"step": 1, "loss": 1.0, "gradient_norm": 0.5},
                     {"step": 2, "loss": 0.9, "gradient_norm": 0.4},
                     {"step": 3, "loss": 0.8, "gradient_norm": 0.3}]
    identical = list(uninterrupted)
    assert checkpoints.resumed_trajectory_report(uninterrupted, identical)["identical"]
    diverged = [dict(entry) for entry in uninterrupted]
    diverged[2]["loss"] = 0.7
    block = checkpoints.resumed_trajectory_report(uninterrupted, diverged)
    assert not block["identical"]
    assert block["offenders"][0]["step"] == 3
    assert block["max_abs_difference"] == pytest.approx(0.1)
    assert "not only the resumed weights" in block["asserts"]


def test_the_trajectory_report_compares_only_the_shared_steps():
    block = checkpoints.resumed_trajectory_report(
        [{"step": 1, "loss": 1.0, "gradient_norm": 0.0},
         {"step": 2, "loss": 0.9, "gradient_norm": 0.0}],
        [{"step": 2, "loss": 0.9, "gradient_norm": 0.0}])
    assert block["compared_steps"] == 1 and block["identical"]


def step_entry(step, loss=1.0, norm=0.5):
    return {"step": step, "loss": loss, "gradient_norm": norm}


@pytest.mark.parametrize("left, right, tolerance", [
    ([], [], 0.0),
    ([step_entry(1)], [step_entry(2)], 0.0),                       # disjoint
    ([step_entry(1)], [step_entry(1), step_entry(2)], 0.0),        # step the reference never took
    ([step_entry(1)], [step_entry(1, loss=0.0), step_entry(1)], 0.0),   # duplicate overwrites
    ([step_entry(1), step_entry(1)], [step_entry(1)], 0.0),
    ([step_entry(1)], [step_entry(1, loss=float("nan"))], 0.0),
    ([step_entry(1, loss=float("inf"))], [step_entry(1, loss=float("inf"))], 0.0),
    ([{"step": 1, "loss": 1.0}], [{"step": 1, "loss": 1.0}], 0.0),  # metric missing
    ([step_entry(1, loss=None)], [step_entry(1, loss=None)], 0.0),
    ([step_entry(1)], [step_entry(1)], float("nan")),
    ([step_entry(1)], [step_entry(1)], float("inf")),
    ([step_entry(1)], [step_entry(1)], -1.0),
    ([step_entry(1.1), step_entry(2)], [step_entry(1.9)], 0.0),     # steps are not truncated
    ([step_entry(1), step_entry(2), step_entry(3)], [step_entry(1), step_entry(3)], 0.0),  # gap
    ([step_entry(1), step_entry(2), step_entry(3)], [step_entry(1), step_entry(2)], 0.0),  # no end
])
def test_the_trajectory_report_never_passes_incomplete_or_invalid_evidence(left, right, tolerance):
    report = checkpoints.resumed_trajectory_report(left, right, tolerance=tolerance)
    assert report["identical"] is False and report["problems"], report
    paths.canonical_json(report)                       # nothing non-finite escapes into JSON


def test_a_resumed_tail_that_ends_at_the_reference_end_is_accepted():
    reference = [step_entry(step, loss=1.0 / step) for step in (1, 2, 3, 4)]
    report = checkpoints.resumed_trajectory_report(reference, reference[2:])
    assert report["identical"] and report["compared_steps"] == 2 and not report["problems"]
    full = checkpoints.resumed_trajectory_report(reference, list(reference), tolerance=1e-9)
    assert full["identical"] and full["compared_steps"] == 4


def test_a_finite_difference_within_tolerance_passes_and_beyond_it_fails():
    reference = [step_entry(1), step_entry(2)]
    near = [step_entry(1), step_entry(2, loss=1.0 + 1e-7)]
    assert checkpoints.resumed_trajectory_report(reference, near, tolerance=1e-6)["identical"]
    report = checkpoints.resumed_trajectory_report(reference, near, tolerance=1e-8)
    assert not report["identical"] and report["offenders"][0]["step"] == 2


def test_an_overflowing_difference_of_finite_values_stays_json_safe_and_fails():
    big = 1.7e308
    report = checkpoints.resumed_trajectory_report(
        [step_entry(1, loss=big)], [step_entry(1, loss=-big)], tolerance=1e308)
    assert not report["identical"] and report["offenders"]
    paths.canonical_json(report)


@pytest.mark.parametrize("field", ["loss", "tolerance"])
def test_an_unrepresentable_integer_yields_a_failed_json_safe_report(field):
    left, right, tolerance = [step_entry(1)], [step_entry(1)], 0.0
    if field == "loss":
        right[0]["loss"] = 10 ** 400
    else:
        tolerance = 10 ** 400
    report = checkpoints.resumed_trajectory_report(left, right, tolerance=tolerance)
    assert report["identical"] is False and report["problems"]
    paths.canonical_json(report)


def test_an_overflowed_difference_is_marked_not_reported_as_a_measurement():
    big = 1.7e308
    report = checkpoints.resumed_trajectory_report(
        [step_entry(1, loss=big)], [step_entry(1, loss=-big)], tolerance=1e308)
    assert report["identical"] is False
    assert report["difference_overflow"] is True and report["max_abs_difference"] is None
    assert report["offenders"][0]["difference_overflow"] is True
    assert report["offenders"][0]["difference"] is None
    paths.canonical_json(report)
    ordinary = checkpoints.resumed_trajectory_report([step_entry(1)], [step_entry(1, loss=1.5)])
    assert ordinary["difference_overflow"] is False and ordinary["max_abs_difference"] == 0.5


# --- the RNG bundle -------------------------------------------------------

@pytest.fixture
def cpu_rng_guard():
    import random
    cpu, python = torch.get_rng_state(), random.getstate()
    yield
    torch.set_rng_state(cpu)
    random.setstate(python)


def seed_cpu(seed):
    """Seed only the global CPU generator. ``torch.manual_seed`` also seeds CUDA, which a mocked
    ``is_initialized`` would send into the real (empty) device generators."""
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())


def mock_cuda(monkeypatch, *, initialized, devices=0, log=None):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: (_ for _ in ()).throw(AssertionError("driver discovery")))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    if log is not None:
        monkeypatch.setattr(torch.cuda, "set_rng_state_all",
                            lambda states: log.append([state.clone() for state in states]))


def test_capture_asks_cuda_for_state_only_when_it_is_initialized(monkeypatch, cpu_rng_guard):
    calls = []
    mock_cuda(monkeypatch, initialized=False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: calls.append(1) or [])
    assert checkpoints.RngBundle.capture({}).torch_cuda is None and calls == []
    mock_cuda(monkeypatch, initialized=True, devices=2)
    states = [torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    bundle = checkpoints.RngBundle.capture({})
    assert bundle.torch_cuda == states and bundle.document()["torch_cuda_devices"] == 2


def test_a_mocked_same_topology_cuda_resume_restores_every_device(monkeypatch, cpu_rng_guard):
    log = []
    mock_cuda(monkeypatch, initialized=True, devices=2, log=log)
    states = [torch.full((4,), 3, dtype=torch.uint8), torch.full((4,), 7, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    bundle = checkpoints.RngBundle.capture({})
    assert bundle.restore({}) is True
    assert len(log) == 1 and all(torch.equal(a, b) for a, b in zip(log[0], states))


@pytest.mark.parametrize("initialized, devices, saved, expected", [
    (True, 1, None, "records no CUDA generator state"),
    (True, 1, [], "holds no CUDA generator state"),
    (True, 2, [torch.zeros(4, dtype=torch.uint8)], "1 CUDA generator state"),
    (True, 1, "bytes", "not a list"),
    (True, 1, [[1, 2]], "1-D uint8 tensor"),
    (True, 1, [torch.zeros(0, dtype=torch.uint8)], "1-D uint8 tensor"),
    (False, 0, [torch.zeros(4, dtype=torch.uint8)], "not initialized"),
])
def test_a_mocked_cuda_resume_is_refused_before_any_generator_moves(
        monkeypatch, cpu_rng_guard, initialized, devices, saved, expected):
    log = []
    mock_cuda(monkeypatch, initialized=False)
    bundle = checkpoints.RngBundle.capture({})
    bundle.torch_cuda = saved
    mock_cuda(monkeypatch, initialized=initialized, devices=devices, log=log)
    before = torch.get_rng_state()
    seed_cpu(123)
    marker = torch.get_rng_state()
    with pytest.raises(ValueError, match=expected):
        bundle.restore({})
    assert torch.equal(torch.get_rng_state(), marker) and not torch.equal(marker, before)
    assert log == []


def test_an_empty_cuda_record_is_a_complete_resume_only_on_an_uninitialized_box(
        monkeypatch, cpu_rng_guard):
    mock_cuda(monkeypatch, initialized=False)
    bundle = checkpoints.RngBundle.capture({})
    bundle.torch_cuda = []
    assert bundle.restore({}) is True


def test_a_full_cpu_resume_restores_python_torch_numpy_and_torch_generators(
        monkeypatch, cpu_rng_guard):
    import random
    mock_cuda(monkeypatch, initialized=False)
    numpy_rng, torch_rng = np.random.default_rng(3), torch.Generator().manual_seed(4)
    bundle = checkpoints.RngBundle.capture({"n": numpy_rng, "t": torch_rng})
    expected = (random.random(), torch.rand(2).tolist(), numpy_rng.random(), torch.rand(
        2, generator=torch_rng).tolist())
    bundle.restore({"n": numpy_rng, "t": torch_rng})
    assert (random.random(), torch.rand(2).tolist(), numpy_rng.random(), torch.rand(
        2, generator=torch_rng).tolist()) == expected


def test_missing_or_malformed_generators_are_refused_without_moving_global_state(
        monkeypatch, cpu_rng_guard):
    import random
    mock_cuda(monkeypatch, initialized=False)
    bundle = checkpoints.RngBundle.capture({"n": np.random.default_rng(1)})
    seed_cpu(9)
    random.seed(9)
    marker = (torch.get_rng_state(), random.getstate())
    for generators, expected in (
            ({"n": torch.Generator()}, "recorded it as a NumPy generator"),
            ({"other": np.random.default_rng(2)}, "no state for the generator"),
            ({"n": np.random.default_rng(1), "x": torch.Generator()}, "no state")):
        with pytest.raises(ValueError, match=expected):
            bundle.restore(generators)
    bundle.torch_cpu = None
    with pytest.raises(ValueError, match="missing the torch CPU"):
        bundle.restore({})
    bundle.torch_cpu = [1, 2]
    with pytest.raises(ValueError, match="1-D uint8"):
        bundle.restore({})
    assert torch.equal(torch.get_rng_state(), marker[0]) and random.getstate() == marker[1]


def _mangled(bundle, generators, mutate, match):
    """Apply ``mutate`` to a fresh-captured bundle, seed markers, and require a clean refusal."""
    import random
    mutate(bundle)
    seed_cpu(123)
    random.seed(123)
    before = (torch.get_rng_state(), random.getstate(),
              {k: g.bit_generator.state for k, g in generators.items()
               if isinstance(g, np.random.Generator)},
              {k: g.get_state() for k, g in generators.items() if isinstance(g, torch.Generator)})
    with pytest.raises(ValueError, match=match):
        bundle.restore(generators)
    assert torch.equal(torch.get_rng_state(), before[0]), "refused restore moved Torch RNG"
    assert random.getstate() == before[1], "refused restore moved Python RNG"
    for name, generator in generators.items():
        if isinstance(generator, torch.Generator):
            assert torch.equal(generator.get_state(), before[3][name])
        else:
            assert generator.bit_generator.state == before[2][name]


@pytest.mark.parametrize("mutate, match", [
    (lambda b: b.numpy.__setitem__("n", {}), "nonempty bit-generator"),
    (lambda b: b.numpy.__setitem__("n", {"bit_generator": "MT19937", "state": {}}), "not 'PCG64'"),
    (lambda b: b.numpy.__setitem__("n", {"bit_generator": "PCG64", "state": {}}),
     "not restorable"),
    (lambda b: setattr(b, "python_random", (99, (1, 2), None)), "Python random"),
    (lambda b: setattr(b, "torch_cpu", torch.zeros(0, dtype=torch.uint8)), "1-D uint8"),
    (lambda b: setattr(b, "torch_cpu", torch.zeros(3, dtype=torch.uint8)), "torch CPU"),
    (lambda b: b.torch_generators.__setitem__("t", torch.zeros(3, dtype=torch.uint8)),
     "torch generator 't'"),
    (lambda b: b.torch_generators.__setitem__("t", torch.zeros(0, dtype=torch.uint8)),
     "1-D uint8"),
    (lambda b: setattr(b, "numpy", []), "must be mappings"),
    (lambda b: setattr(b, "torch_generators", None), "must be mappings"),
    (lambda b: b.numpy.__setitem__("t", b.numpy["n"]), "both NumPy and torch"),
])
def test_a_malformed_rng_bundle_is_refused_before_any_generator_moves(
        monkeypatch, cpu_rng_guard, mutate, match):
    mock_cuda(monkeypatch, initialized=False)
    generators = {"n": np.random.default_rng(1), "t": torch.Generator().manual_seed(2)}
    _mangled(checkpoints.RngBundle.capture(generators), generators, mutate, match)


def test_a_named_cpu_generator_round_trips_with_this_builds_own_state_size(
        monkeypatch, cpu_rng_guard):
    mock_cuda(monkeypatch, initialized=False)
    generator = torch.Generator().manual_seed(2)
    bundle = checkpoints.RngBundle.capture({"t": generator})
    expected = generator.get_state().clone()
    generator.manual_seed(999)
    assert bundle.restore({"t": generator}) is True   # whatever length this Torch build serialises
    assert torch.equal(generator.get_state(), expected)


def _mock_cuda_generator(saved):
    from unittest.mock import Mock
    generator = Mock(spec=torch.Generator)
    generator.device = torch.device("cuda:0")
    generator.get_state.return_value = saved
    return generator


def test_a_named_cuda_generator_is_restored_without_a_cpu_preflight(monkeypatch, cpu_rng_guard):
    saved = torch.zeros(16, dtype=torch.uint8)        # CUDA-style layout, not the CPU size
    generator = _mock_cuda_generator(saved)
    log = []
    mock_cuda(monkeypatch, initialized=True, devices=1, log=log)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [saved])
    bundle = checkpoints.RngBundle.capture({"energy_rng": generator})
    assert bundle.restore({"energy_rng": generator}) is True
    generator.set_state.assert_called_once()
    assert torch.equal(generator.set_state.call_args.args[0], saved)
    assert len(log) == 1


def test_a_named_cuda_generator_without_cuda_state_is_refused_before_anything_moves(
        monkeypatch, cpu_rng_guard):
    saved = torch.zeros(16, dtype=torch.uint8)
    generator = _mock_cuda_generator(saved)
    mock_cuda(monkeypatch, initialized=False)
    bundle = checkpoints.RngBundle.capture({"energy_rng": generator})
    with pytest.raises(ValueError, match="CUDA generator"):
        bundle.restore({"energy_rng": generator})
    generator.set_state.assert_not_called()


def test_a_named_generator_on_an_unsupported_device_is_refused(monkeypatch, cpu_rng_guard):
    generator = _mock_cuda_generator(torch.zeros(16, dtype=torch.uint8))
    generator.device = torch.device("meta")
    mock_cuda(monkeypatch, initialized=False)
    bundle = checkpoints.RngBundle.capture({"g": generator})
    with pytest.raises(ValueError, match="only CPU and CUDA"):
        bundle.restore({"g": generator})
    generator.set_state.assert_not_called()


def test_a_real_cpu_checkpoint_round_trips_and_inference_load_is_rng_neutral(
        monkeypatch, tmp_path, cpu_rng_guard):
    mock_cuda(monkeypatch, initialized=False)        # CPU-only state, whatever ran earlier
    model, other = TinyModel(), TinyModel()
    state = state_with()
    target = checkpoints.save(tmp_path / "last.pt", model=model, heads=None, optimizer=None,
                              scheduler=None, state=state, identity=identity())
    payload = torch.load(str(target), map_location="cpu", weights_only=False)
    assert payload["rng"].torch_cuda is None            # a CPU process records no CUDA stream
    seed_cpu(77)
    marker = torch.get_rng_state()
    record = checkpoints.load_inference(target, model=other, heads=None, identity=identity())
    assert record["rng_restored"] is False and torch.equal(torch.get_rng_state(), marker)
    for key, value in model.state_dict().items():
        assert torch.equal(other.state_dict()[key], value)
    _, loaded = checkpoints.restore(target, model=TinyModel(), heads=None, optimizer=None,
                                    scheduler=None, state=state_with(step=0),
                                    identity=identity())
    assert loaded["rng"]["torch_cuda_devices"] == 0 and loaded["step"] == 5


# --- the cadence ----------------------------------------------------------

def test_the_cadence_writes_on_the_interval_and_at_the_end():
    cadence = checkpoints.Cadence(save_every=10, milestones=(25,), bytes_per_checkpoint=1000)
    assert cadence.should_save_last(10, total=100)
    assert not cadence.should_save_last(11, total=100)
    assert cadence.should_save_last(100, total=100), "the final step always writes"
    assert cadence.is_milestone(25) and not cadence.is_milestone(26)
    assert cadence.retained() == 3
    document = cadence.document(disk_budget_bytes=10000)
    assert document["fits"] and document["estimated_bytes"] == 3000


def test_a_retention_policy_that_does_not_fit_is_refused_not_trimmed():
    cadence = checkpoints.Cadence(save_every=1, milestones=(1, 2, 3),
                                  bytes_per_checkpoint=10 ** 9)
    with pytest.raises(ValueError, match="Reduce the milestones deliberately"):
        cadence.document(disk_budget_bytes=10 ** 8)
