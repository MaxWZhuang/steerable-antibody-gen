"""The joint-competition hold, and the DPLM qualification orchestration and root lifecycle.

Two things are checked here and neither is a GPU claim. (1) The joint config is a contract:
its route blocks are the source blocks and it differs from the base config only where it says it
does, and ``spec.competition_unresolved`` keeps it from resolving until the amendments land in
code. (2) ``her2_routes_qualify`` orchestrates honestly: it refuses a used root, a held lock or a
flight revision BEFORE writing anything, classifies capacity refusal versus failure, never lets a
CPU run emit a GPU verdict, and leaves no production marker behind. Phases are injected fakes
here; the real tiny-model checks (profile, decode, native evidence) are in
``test_her2_routes_dplm_qualification.py``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_campaign as campaign
from smallAntibodyGen.experiments import her2_routes_qualify as qualify
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments.her2_replay_campaign import CampaignLock, lock_state

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs" / "experiments"
JOINT_PATH = CONFIGS / "her2_generative_routes_joint.json"
ROUTE_150 = "dplm_150m_masked"
ROUTE_650 = "dplm_650m_lora_masked"
ORIGINALS = ("ablang2_masked_editor", "piggen_causal_infiller", "dwjs_energy")


def read(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def joint():
    return read("her2_generative_routes_joint.json")


@pytest.fixture(scope="module")
def base():
    return read("her2_generative_routes.json")


@pytest.fixture(scope="module")
def dplm_config():
    return read("her2_generative_routes_dplm.json")


# ---------------------------------------------------------------------------
# the joint config is a contract
# ---------------------------------------------------------------------------

def settings_only(node, key=""):
    """Drop prose. A string over 120 characters, or a note/rule/why/status, is commentary."""
    def prose(name, value):
        return (name.endswith(("_note", "_rule")) or name in ("note", "rule", "why", "status",
                                                              "reason")
                or (isinstance(value, str) and len(value) > 120))
    if isinstance(node, dict):
        return {name: settings_only(value, name) for name, value in node.items()
                if not prose(name, value)}
    if isinstance(node, list):
        return [settings_only(value) for value in node]
    return node


def test_every_joint_route_block_is_its_source_block(joint, base, dplm_config):
    routes = joint["routes"]
    for route in ORIGINALS:
        assert routes[route] == base["routes"][route], route
    for route in (ROUTE_150, ROUTE_650):
        assert routes[route] == dplm_config["routes"][route], route
    assert routes["enabled"] == list(ORIGINALS) + [ROUTE_150, ROUTE_650]
    assert joint["dplm"] == dplm_config["dplm"]
    assert (joint["artifacts"]["dplm_acquisition_manifest"]
            == dplm_config["artifacts"]["dplm_acquisition_manifest"])


def test_the_joint_config_differs_from_the_base_only_where_declared(joint, base, dplm_config):
    assert set(joint) - set(base) == {"dplm", "competition", "opt_in_note"}
    assert set(base) - set(joint) == set()
    for identity in ("campaign_id", "config_path", "run_root_leaf"):
        assert len({joint[identity], base[identity], dplm_config[identity]}) == 3, identity
    assert joint["schema_version"] == base["schema_version"]
    assert joint["protocol"] == base["protocol"]
    untouched = ("her2", "conditions", "sources", "split", "curriculum", "optimization", "smoke",
                 "profile", "gates", "controls", "priors", "posttrain", "tolerances", "roots",
                 "runtime", "confirmation")
    for section in untouched:
        assert settings_only(joint[section]) == settings_only(base[section]), section
    assert joint["decoding"]["strategies"] == base["decoding"]["strategies"] + ["diffusion_reparam"]
    assert (settings_only({**joint["decoding"], "strategies": []})
            == settings_only({**base["decoding"], "strategies": []}))
    assert joint["storage"]["bytes_per_checkpoint"] == dplm_config["storage"][
        "bytes_per_checkpoint"] == 800000000
    assert joint["storage"]["min_free_bytes"] == base["storage"]["min_free_bytes"]
    artifacts = settings_only(joint["artifacts"])
    assert artifacts.pop("dplm_acquisition_manifest")
    assert artifacts == settings_only(base["artifacts"])
    assert joint["source"]["entry_points"] == base["source"]["entry_points"] + [
        "scripts/her2_routes_qualify_dplm.py"]
    assert joint["source"]["extra_files"] == base["source"]["extra_files"] + [
        "configs/experiments/dplm_pins.json"]
    assert joint["revisions"]["current"] == "joint_rev01"
    assert {k: v for k, v in joint["revisions"].items() if k in ("kind", "supersedes",
                                                                 "frozen_after")} == {
        k: v for k, v in base["revisions"].items() if k in ("kind", "supersedes", "frozen_after")}


def test_the_two_existing_configs_are_not_competition_configs(base, dplm_config):
    assert "competition" not in base and "competition" not in dplm_config
    assert spec.competition_unresolved(base) == []
    assert spec.competition_unresolved(dplm_config) == []


# ---------------------------------------------------------------------------
# the hold: fail closed, in code
# ---------------------------------------------------------------------------

def test_the_joint_config_holds_on_every_pending_amendment_and_nothing_else(joint):
    assert spec.competition_unresolved(joint) == [
        "competition.amendment::hr17_high_null_ratio",
        "competition.amendment::hr18_bank_and_distance_strata",
        "competition.amendment::originals_checkpoint_policy"]


def test_editing_the_config_cannot_clear_the_hold(joint):
    fewer_routes = copy.deepcopy(joint)
    fewer_routes["routes"]["enabled"].remove(ROUTE_650)
    assert "competition.route_set" in spec.competition_unresolved(fewer_routes)
    reordered = copy.deepcopy(joint)
    reordered["routes"]["enabled"].reverse()
    assert "competition.route_set" in spec.competition_unresolved(reordered)
    shorter = copy.deepcopy(joint)
    shorter["competition"]["required_amendments"].pop()
    unresolved = spec.competition_unresolved(shorter)
    assert "competition.amendment_list" in unresolved
    assert "competition.amendment::originals_checkpoint_policy" in unresolved, (
        "deleting an entry must not remove it from what the code still requires")
    emptied = copy.deepcopy(joint)
    emptied["competition"]["required_amendments"] = []
    emptied["competition"]["required_routes"] = []
    assert {"competition.amendment_list", "competition.route_set"} <= set(
        spec.competition_unresolved(emptied))
    deleted = copy.deepcopy(joint)
    del deleted["competition"]
    assert "competition.amendment_list" in spec.competition_unresolved(deleted), (
        "a config that enables all five routes is a competition config with or without its block")


class ProtocolContext:
    def __init__(self, root, config):
        self.run_root = Path(root)
        self.config = config
        self.campaign_id = "probe"
        self.run_root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        return self.run_root.joinpath(*[str(part) for part in parts])


def resolve(context):
    return spec.resolved_protocol(
        context, snapshot={"snapshot_sha256": "s"}, external={"external_sha256": "e"},
        split={"n": 1}, profile={"n": 1}, forecast={"n": 1}, revision={"revision": "r"})


def test_the_resolved_protocol_records_and_require_resolved_refuses(joint, base, tmp_path):
    held = ProtocolContext(tmp_path / "joint", joint)
    document = resolve(held)
    assert held.path("resolved_protocol.json").is_file(), "the evidence is retained"
    assert len(document["unresolved"]) == 3
    with pytest.raises(ValueError, match=r"competition\.amendment::hr17_high_null_ratio"):
        spec.require_resolved(held, label="bakeoff")
    free = ProtocolContext(tmp_path / "base", base)
    assert resolve(free)["unresolved"] == []
    assert spec.require_resolved(free)["unresolved"] == []


def test_freeze_writes_the_protocol_then_raises_readiness_for_the_joint_config(
        joint, base, tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "source_snapshot", lambda context: {"snapshot_sha256": "s",
                                                                  "file_count": 1})
    monkeypatch.setattr(spec, "verify_external_sources",
                        lambda context, label="": {"files": 1, "external_sha256": "e"})
    monkeypatch.setattr(spec, "write_revision_marker",
                        lambda context, **kwargs: {"revision": "r", "kind": "development"})
    monkeypatch.setattr(spec, "combined_identity", lambda context: {"combined_sha256": "c"})
    for name, config in (("joint", joint), ("base", base)):
        context = ProtocolContext(tmp_path / name, config)
        for logical in (spec.SPLIT_FREEZE, spec.PROFILE_JSON, spec.FORECAST_JSON):
            paths.write_json(context.path(logical), {"present": True})
        if name == "joint":
            with pytest.raises(campaign.ReadinessGate, match="competition.amendment"):
                campaign.stage_freeze(context, None)
            assert context.path("resolved_protocol.json").is_file()
        else:
            assert campaign.stage_freeze(context, None)["status"] == "completed"


def test_a_selection_stage_refuses_on_the_joint_protocol(joint, tmp_path, monkeypatch):
    context = ProtocolContext(tmp_path / "joint", joint)
    resolve(context)
    monkeypatch.setattr(spec, "verify_source_snapshot", lambda context, label="": {})
    monkeypatch.setattr(spec, "verify_external_sources", lambda context, label="": {})
    ran = []
    with pytest.raises(ValueError, match="unresolved fields"):
        campaign._guarded(context, "bakeoff", lambda: ran.append("ran"))
    assert ran == []


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

class FakeGpu:
    def __init__(self, available=True):
        self._available = available
        self.readings = 0

    def available(self):
        return self._available

    def reading(self):
        self.readings += 1
        return {"nvml_fb_free_mib": 2000.0}


class FakeServices:
    def __init__(self, context, device):
        self.context, self.device = context, device
        self.config = context.config

    def route_block(self, route):
        return dict(self.config["routes"][route])

    def dplm_pins(self):
        return {"pins": "fake"}

    def native_identity(self, route):
        return f"native::{route}"


class Recorder:
    """Phase fakes that record what ran and let a test choose what each does."""

    def __init__(self, *, gpu_measured=True, **overrides):
        self.calls, self.contexts = [], []
        self.overrides = overrides
        self.gpu_measured = gpu_measured

    def phase(self, name):
        def run(run_state):
            self.calls.append(name)
            self.contexts.append(run_state.context)
            if name in self.overrides:
                return self.overrides[name](run_state)
            if name == "profile":
                return {"chosen_microbatch": 2, "gpu_measured": self.gpu_measured}
            if name == "environment":
                return {"cuda_available": run_state.gpu.available(), "nvml_start": None}
            return {"ok": True} if name != "snapshot" else {}
        return run

    def phases(self):
        table = {name: self.phase(name) for name in qualify.PHASE_ORDER}
        table["seal"] = self.overrides.get("seal", lambda run_state: {"problems": []})
        return table


def go(tmp_path, recorder, *, route=ROUTE_150, revision="joint_qual_t", device="cuda",
       gpu=None, built=None, phases=None):
    def build(context, device):
        if built is not None:
            built.append(device)
        return FakeServices(context, device)
    return qualify.run_qualification(
        ROOT, config_path=JOINT_PATH, artifact_root=tmp_path, revision=revision, route=route,
        device=device, build_services=build, phases=phases or recorder.phases(),
        gpu=gpu or FakeGpu(), environment={"cache": "x"})


PROFILE_INDEX = qualify.PHASE_ORDER.index("profile")


def run_root(tmp_path, revision="joint_qual_t"):
    return tmp_path / "runs" / revision / "flight_joint"


def tree(root):
    return sorted(str(path.relative_to(root)) for path in Path(root).rglob("*"))


def test_a_cuda_run_with_cuda_evidence_qualifies_and_leaves_no_production_marker(tmp_path):
    recorder = Recorder()
    built = []
    record = go(tmp_path, recorder, built=built)
    assert record["verdict"] == qualify.VERDICT_QUALIFIED and record["gpu_qualified"] is True
    assert recorder.calls == list(qualify.PHASE_ORDER)
    assert built == ["cuda"]
    root = run_root(tmp_path)
    on_disk = json.loads((root / "qualification.json").read_text(encoding="utf-8"))
    assert on_disk["verdict"] == "qualified" and on_disk["reserved_opened"] is False
    assert on_disk["nvml"]["end"] == {"nvml_fb_free_mib": 2000.0}
    for marker in ("stages", "revision.json", "resolved_protocol.json", "frozen_recipe.json",
                   "reserved_opening.json", spec.FREEZE_MARKER):
        assert not (root / marker).exists(), marker
    assert lock_state(root / campaign.LOCK_FILE)["state"] == "released"
    assert "training has been launched" in on_disk["does_not_establish"]


def test_cuda_evidence_is_required_not_assumed(tmp_path):
    record = go(tmp_path, Recorder(gpu_measured=False))
    assert record["verdict"] == qualify.VERDICT_FAILED and record["gpu_qualified"] is False
    recorder = Recorder()
    phases = recorder.phases()
    phases["environment"] = qualify.phase_environment      # the REAL environment phase
    unavailable = go(tmp_path, recorder, revision="joint_qual_nocuda", gpu=FakeGpu(False),
                     phases=phases)
    assert unavailable["verdict"] == qualify.VERDICT_FAILED
    assert unavailable["phases"]["environment"]["status"] == "failed"
    assert "not available" in unavailable["phases"]["environment"]["reason"]
    assert unavailable["phases"]["profile"]["status"] == "skipped"
    assert recorder.calls == [], "no later phase ran on a host without CUDA"


def test_a_cpu_run_never_emits_gpu_qualified_evidence(tmp_path):
    gpu = FakeGpu(True)
    recorder = Recorder(gpu_measured=True)      # even a fabricated CUDA claim cannot promote it
    record = go(tmp_path, recorder, device="cpu", gpu=gpu)
    assert record["verdict"] == qualify.VERDICT_CPU_ONLY and record["gpu_qualified"] is False
    assert record["nvml"] is None and gpu.readings == 0, "a CPU run reads no GPU memory"
    assert "code check" in record["cpu_note"]


def test_out_of_memory_is_a_bounded_hardware_refusal_and_skips_the_rest(tmp_path):
    def oom(run_state):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    recorder = Recorder(profile=oom)
    record = go(tmp_path, recorder)
    assert record["verdict"] == qualify.VERDICT_REFUSED and record["gpu_qualified"] is False
    assert recorder.calls == list(qualify.PHASE_ORDER[:PROFILE_INDEX + 1]), (
        "smoke and decode must not run after a capacity refusal")
    assert record["phases"]["profile"]["status"] == qualify.VERDICT_REFUSED
    assert "OutOfMemoryError" in record["phases"]["profile"]["reason"]
    assert record["phases"]["smoke"]["status"] == record["phases"]["decode"]["status"] == "skipped"
    assert record["seal"]["problems"] == []
    assert (run_root(tmp_path) / "qualification.json").is_file()


@pytest.mark.parametrize("error", [ValueError("the source table is corrupt"),
                                   RuntimeError("a decode assertion"),
                                   FloatingPointError("non-finite"), KeyError("column")])
def test_a_code_or_data_error_is_a_failure_not_a_capacity_refusal(tmp_path, error):
    def broken(run_state):
        raise error
    for phase in ("sources", "profile", "smoke"):
        recorder = Recorder(**{phase: broken})
        record = go(tmp_path, recorder, revision=f"joint_qual_{phase}_{type(error).__name__}")
        assert record["verdict"] == qualify.VERDICT_FAILED, phase
        assert record["phases"][phase]["status"] == "failed"
        assert type(error).__name__ in record["phases"][phase]["reason"]
        assert record["phases"][phase]["where"], "the frames are recorded"


def test_a_phase_that_reports_its_own_refusal_or_failure_keeps_its_evidence(tmp_path):
    def refuse(run_state):
        raise qualify.PhaseRefusal("no microbatch passed", {"rows": [1, 2]})

    def fail(run_state):
        raise qualify.PhaseFailure("nan", {"loss": float("nan")})
    refused = go(tmp_path, Recorder(profile=refuse))
    assert refused["verdict"] == "refused_by_hardware"
    assert refused["phases"]["profile"]["evidence"] == {"rows": [1, 2]}
    failed = go(tmp_path, Recorder(profile=fail), revision="joint_qual_nan")
    assert failed["verdict"] == "failed"
    text = (run_root(tmp_path, "joint_qual_nan") / "qualification.json").read_text("utf-8")
    assert "NaN" not in text and '"nonfinite": "nan"' in text, "evidence stays valid JSON"


def test_an_interrupt_is_recorded_and_the_lock_is_released(tmp_path):
    def interrupted(run_state):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        go(tmp_path, Recorder(profile=interrupted))
    root = run_root(tmp_path)
    assert json.loads((root / "qualification.json").read_text("utf-8"))["verdict"] == "failed"
    assert lock_state(root / campaign.LOCK_FILE)["state"] == "released"


def test_the_seal_invalidates_evidence_when_a_marker_appears_or_the_source_changes(
        tmp_path, monkeypatch):
    def plant_marker(run_state):
        run_state.context.path("revision.json").write_text("{}", encoding="utf-8")
        return {}
    recorder = Recorder(snapshot=plant_marker)
    phases = recorder.phases()
    phases.pop("seal")                          # the REAL seal
    record = go(tmp_path, recorder, revision="joint_qual_marker", phases=phases)
    assert record["verdict"] == "failed" and record["gpu_qualified"] is False
    assert "revision.json" in record["reason"]

    def drifted(context, label=""):
        raise ValueError("a file changed")
    monkeypatch.setattr(spec, "verify_source_snapshot", drifted)
    recorder = Recorder(snapshot=lambda run_state: {"snapshot_sha256": "x"})
    phases = recorder.phases()
    phases.pop("seal")
    record = go(tmp_path, recorder, revision="joint_qual_drift", phases=phases)
    assert record["verdict"] == "failed" and "no longer matches" in record["reason"], (
        "every phase passed on CUDA and the evidence is still invalid")


# ---------------------------------------------------------------------------
# profile analysis: judged at the SELECTED shape
# ---------------------------------------------------------------------------

def _row(task, size, *, fits=True, **changes):
    row = {"task": task, "microbatch": size, "rows_measured": size, "accepted": fits,
           "failure": None, "completed_steps": 4, "steps_requested": 4, "probe_device": "cuda",
           "dispatched": {"scheduled_task": task,
                          "diagnostic_keys": ["mean_timestep", "num_timesteps"]},
           "vram": {"nvml_fb_free_mib": 500 if fits else 100},
           "signals": {"driver_free_above_floor": fits, "allocator_peak_within_budget": fits,
                       "step_time_not_degraded": True}}
    row.update(changes)
    return row


def _profile(rows, chosen, **changes):
    from smallAntibodyGen.experiments import her2_routes_data as data
    profile = {"chosen_microbatch": chosen, "accepted_microbatches": [] if chosen is None
               else [chosen], "tasks_profiled": list(data.TASKS), "measured_on": {"cuda": True},
               "training_mode": {"model": True, "native": True, "heads": True},
               "device_total_mib": 4096.0, "rows": rows}
    profile.update(changes)
    return profile


def _analyse(profile, *, gpu=True):
    from smallAntibodyGen.experiments import her2_routes_data as data
    return qualify.analyse_profile(profile, route=ROUTE_150, tasks=data.TASKS, diffusion=True,
                                   gpu_required=gpu)


def _tasks():
    from smallAntibodyGen.experiments import her2_routes_data as data
    return data.TASKS


def test_a_smaller_accepted_shape_qualifies_and_the_larger_refusal_is_retained():
    rows = [_row(task, size, fits=size == 1) for size in (1, 2) for task in _tasks()]
    status, reason, evidence = _analyse(_profile(rows, 1))
    assert status is None, reason
    assert evidence["gpu_measured"] is True
    refused = evidence["larger_shape_refusals"]
    assert len(refused) == len(_tasks()) and {r["microbatch"] for r in refused} == {2}


@pytest.mark.parametrize("change,phrase", [
    (lambda rows: rows[0].update(accepted=False), "not accepted"),
    (lambda rows: rows[0]["signals"].update(allocator_peak_within_budget=False), "signals"),
    (lambda rows: rows[0]["signals"].update(allocator_peak_within_budget=None), "signals"),
    (lambda rows: rows[0]["signals"].pop("step_time_not_degraded"), "not recorded"),
    (lambda rows: rows[0].update(rows_measured=0), "row count"),
    (lambda rows: rows[0].update(rows_measured=None), "row count"),
    (lambda rows: rows[0].update(completed_steps=0), "completed_steps"),
    (lambda rows: rows[0].pop("completed_steps"), "completed_steps"),
    (lambda rows: rows[0].pop("steps_requested"), "steps_requested"),
    (lambda rows: rows[0].update(completed_steps=1, steps_requested=2 + 2), "completed_steps"),
    (lambda rows: rows[0].update(completed_steps=1, steps_requested=1), "steps_requested"),
    (lambda rows: rows[0].update(completed_steps=True, steps_requested=True), "integers"),
    (lambda rows: rows[0].update(completed_steps=4.0), "integers"),
    (lambda rows: rows[0].update(failure="OutOfMemoryError: x"), "failure"),
    (lambda rows: rows.append(dict(rows[0])), "duplicated"),
    (lambda rows: rows.pop(), "missing"),
    (lambda rows: rows[0].update(probe_device="cpu"), "no accepted CUDA measurement"),
    (lambda rows: rows[0]["vram"].update(nvml_fb_free_mib=None), "no accepted CUDA measurement")])
def test_every_task_must_hold_at_the_chosen_shape(change, phrase):
    rows = [_row(task, 1) for task in _tasks()]
    change(rows)
    status, reason, _ = _analyse(_profile(rows, 1))
    assert status == qualify.VERDICT_FAILED and phrase in reason, reason


def test_a_nonfinite_row_at_any_shape_is_a_failure_not_a_capacity_refusal():
    rows = [_row(task, size, fits=size == 1) for size in (1, 2) for task in _tasks()]
    rows[-1].update(failure="nonfinite_gradient", accepted=False)
    status, reason, _ = _analyse(_profile(rows, 1))
    assert status == qualify.VERDICT_FAILED and "code defect" in reason
    only_large = [_row(task, 2, fits=False) for task in _tasks()]
    status, _, _ = _analyse(_profile(only_large, None))
    assert status == qualify.VERDICT_REFUSED, "no accepted shape and a failed signal is capacity"


def test_a_cpu_shape_needs_no_gpu_memory_signal_but_every_measured_signal_true():
    rows = [_row(task, 1, probe_device="cpu",
                 signals={"driver_free_above_floor": None, "allocator_peak_within_budget": None,
                          "step_time_not_degraded": True}) for task in _tasks()]
    status, reason, evidence = _analyse(_profile(rows, 1, measured_on={"cuda": False}), gpu=False)
    assert status is None, reason
    assert evidence["gpu_measured"] is False
    rows[0]["signals"]["step_time_not_degraded"] = False
    status, _, _ = _analyse(_profile(rows, 1, measured_on={"cuda": False}), gpu=False)
    assert status == qualify.VERDICT_FAILED


BAD_MODES = [None, {}, {"model": True, "native": True}, {"model": True, "native": False, "heads": True},
             {"model": True, "native": None, "heads": True}, {"model": False, "native": True, "heads": True},
             {"model": True, "native": True, "heads": False}, {"model": 1, "native": True, "heads": True}]


@pytest.mark.parametrize("mode", BAD_MODES)
def test_missing_or_false_training_mode_fails_at_a_chosen_shape(mode):
    rows = [_row(task, 1) for task in _tasks()]
    status, reason, _ = _analyse(_profile(rows, 1, training_mode=mode))
    assert status == qualify.VERDICT_FAILED and "training mode" in reason, reason


@pytest.mark.parametrize("mode", BAD_MODES)
def test_missing_or_false_training_mode_is_never_a_hardware_refusal(mode):
    refused = [_row(task, 2, fits=False) for task in _tasks()]
    refused[0]["failure"] = "OutOfMemoryError: x"
    status, reason, _ = _analyse(_profile(refused, None, training_mode=mode))
    assert status == qualify.VERDICT_FAILED and "training mode" in reason, reason
    status, _, _ = _analyse(_profile(refused, None))
    assert status == qualify.VERDICT_REFUSED, "with valid mode evidence the same rows are capacity"


@pytest.mark.parametrize("free", [4194280.0, 4097.0, -1.0, float("nan"), float("inf")])
def test_an_impossible_nvml_reading_cannot_qualify_the_selected_shape(free):
    rows = [_row(task, 1) for task in _tasks()]
    rows[0]["vram"]["nvml_fb_free_mib"] = free
    status, reason, _ = _analyse(_profile(rows, 1))
    assert status == qualify.VERDICT_FAILED and "NVML" in reason, reason


@pytest.mark.parametrize("total", [None, 0.0, float("nan"), True])
def test_a_gpu_profile_without_a_valid_device_total_cannot_qualify(total):
    rows = [_row(task, 1) for task in _tasks()]
    status, reason, _ = _analyse(_profile(rows, 1, device_total_mib=total))
    assert status == qualify.VERDICT_FAILED and "device total" in reason, reason


def test_a_bad_reading_on_a_larger_refused_shape_does_not_reject_a_viable_smaller_one():
    rows = [_row(task, size, fits=size == 1) for size in (1, 2) for task in _tasks()]
    for row in rows:
        if row["microbatch"] == 2:
            row["vram"]["nvml_fb_free_mib"] = 4194280.0
    status, reason, evidence = _analyse(_profile(rows, 1))
    assert status is None, reason
    assert evidence["gpu_measured"] is True and evidence["device_total_mib"] == 4096.0
    rows[0]["vram"]["nvml_fb_free_mib"] = 4096.0      # exactly the total is possible
    status, reason, _ = _analyse(_profile(rows, 1))
    assert status is None, reason


def test_a_gpu_verdict_needs_unmeasured_memory_signals_to_fail():
    rows = [_row(task, 1) for task in _tasks()]
    rows[0]["signals"]["driver_free_above_floor"] = None
    status, _, _ = _analyse(_profile(rows, 1))
    assert status == qualify.VERDICT_FAILED


# ---------------------------------------------------------------------------
# the effective config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route,variant", [(ROUTE_150, "dplm_150m"), (ROUTE_650, "dplm_650m")])
def test_the_effective_config_holds_exactly_one_route_and_pins_the_stage_identity(
        tmp_path, joint, route, variant):
    recorder = Recorder()
    go(tmp_path, recorder, route=route)
    root = run_root(tmp_path)
    effective = json.loads((root / "effective_config.json").read_text("utf-8"))
    assert effective["routes"]["enabled"] == [route]
    assert spec.dplm_variants(effective) == [variant], "only this route's bytes are bound"
    assert effective["qualification"]["source_config_sha256"] == paths.sha256_file(JOINT_PATH)
    assert effective["qualification"]["route"] == route
    assert effective["config_path"] == joint["config_path"], "the snapshot needs the repo config"
    context = recorder.contexts[0]
    assert context.config_path == root / "effective_config.json", (
        "stage and training identities hash context.config_path, so the subset is pinned in them")
    assert context.run_root == root and context.revision == "joint_qual_t"
    assert context.config["routes"]["enabled"] == [route]


def test_qualification_is_defined_for_the_dplm_routes_only(tmp_path):
    for route in ORIGINALS:
        with pytest.raises(ValueError, match="not a native DPLM route"):
            go(tmp_path, Recorder(), route=route)
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------------------
# lifecycle: every refusal happens before anything is written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("revision", ["rev01", "t1_rev01", "dplm_rev01", "joint_rev01",
                                      "joint_qual_", "joint_qual_../x", "Joint_qual_a"])
def test_a_flight_or_malformed_revision_is_refused_before_any_write(tmp_path, revision):
    built = []
    with pytest.raises(qualify.QualificationRefused, match="revision"):
        go(tmp_path, Recorder(), revision=revision, built=built)
    assert built == [] and not (tmp_path / "runs").exists()


def seed_root(tmp_path, *, files):
    root = run_root(tmp_path)
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


@pytest.mark.parametrize("files,phrase", [
    ({"reserved_opening.json": "{}"}, "reserved-opening"),
    ({"stages/profile.json": "{}"}, "completed stage records"),
    ({"source_snapshot.json": "{}"}, "frozen or completed"),
    ({"revision.json": "{}"}, "frozen or completed"),
    ({"resolved_protocol.json": "{}"}, "frozen or completed"),
    ({"qualification.json": "{}", "effective_config.json": "{}"}, "previous qualification"),
    ({"something_else.txt": "x"}, "already holds content")])
def test_a_used_root_is_refused_untouched(tmp_path, files, phrase):
    root = seed_root(tmp_path, files=files)
    before = {name: (root / name).read_bytes() for name in files}
    listing = tree(root)
    built = []
    with pytest.raises(qualify.QualificationRefused, match=phrase):
        go(tmp_path, Recorder(), built=built)
    assert built == [], "no model or service work before a refusal"
    assert tree(root) == listing, "the refusal wrote nothing, not even a lock file"
    assert {name: (root / name).read_bytes() for name in files} == before


def test_a_second_qualification_in_the_same_revision_is_refused(tmp_path):
    first = go(tmp_path, Recorder())
    root = run_root(tmp_path)
    written = (root / "qualification.json").read_bytes()
    with pytest.raises(qualify.QualificationRefused, match="previous qualification"):
        go(tmp_path, Recorder())
    assert (root / "qualification.json").read_bytes() == written
    assert first["verdict"] == "qualified"


def test_qualify_route_itself_never_overwrites_a_record(tmp_path):
    recorder = Recorder()
    go(tmp_path, recorder)
    context = recorder.contexts[0]
    before = context.path("qualification.json").read_bytes()
    with pytest.raises(qualify.QualificationRefused, match="never overwritten"):
        qualify.qualify_route(context, FakeServices(context, "cuda"), ROUTE_150, device="cuda",
                              phases=Recorder().phases(), gpu=FakeGpu())
    assert context.path("qualification.json").read_bytes() == before


def test_a_held_lock_is_refused_before_anything_is_written(tmp_path):
    root = run_root(tmp_path)
    holder = CampaignLock(root / campaign.LOCK_FILE)
    holder.acquire()
    try:
        listing = tree(root)
        built = []
        with pytest.raises(qualify.QualificationRefused, match="lock is held"):
            go(tmp_path, Recorder(), built=built)
        assert built == [] and tree(root) == listing
        assert not (root / "effective_config.json").exists()
    finally:
        holder.release()


def test_an_unobservable_lock_is_refused_not_assumed_free(tmp_path, monkeypatch):
    monkeypatch.setattr(qualify, "lock_state", lambda path: {"state": "inaccessible"})
    built = []
    with pytest.raises(qualify.QualificationRefused, match="inaccessible"):
        go(tmp_path, Recorder(), built=built)
    assert built == [] and not (tmp_path / "runs").exists()


def test_the_root_is_rechecked_inside_the_lock(tmp_path, monkeypatch):
    real = qualify.root_blockers
    calls = []

    def changed_underneath(root, *, inside_lock=False):
        calls.append(inside_lock)
        return ["someone wrote into the root"] if inside_lock else real(root)
    monkeypatch.setattr(qualify, "root_blockers", changed_underneath)
    built = []
    with pytest.raises(qualify.QualificationRefused, match="after taking the lock"):
        go(tmp_path, Recorder(), built=built)
    assert calls == [False, True] and built == []
    root = run_root(tmp_path)
    assert not (root / "effective_config.json").exists(), (
        "the effective config is written only after both checks pass")
    assert lock_state(root / campaign.LOCK_FILE)["state"] == "released"


def test_the_effective_config_is_written_after_the_lock_and_before_the_context(tmp_path):
    """An existing root is never given an effective config; a new one gets it inside the lock."""
    seen = {}

    def spy(run_state):
        root = run_state.context.run_root
        seen["lock"] = lock_state(root / campaign.LOCK_FILE)["state"]
        seen["effective"] = (root / "effective_config.json").is_file()
        return {}
    go(tmp_path, Recorder(environment=spy))
    assert seen == {"lock": "held", "effective": True}
