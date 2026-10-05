"""Behavioural regressions for the operational repairs HR-01 .. HR-08 of the 2026-09-28 audit.

Each test drives the real function with the input shape that broke it, and asserts the
behaviour, never the source text:

* HR-01 the frozen recipe reads ``priors.json`` (a mapping), not the stage record's name list;
* HR-02 the posttrain wrapper on the real ``run_posttrain`` shape lives in
  ``test_her2_routes_dplm_services.py`` next to the model that produces it;
* HR-03 a missing selected checkpoint stops the stage; the backbone is only ever explicit;
* HR-04 the selection guard refuses all six selection stages under a confirmation revision;
* HR-05 an inaccessible lock is UNKNOWN, never "stopped", and never authorises a launch;
* HR-06 the standalone profiler holds the real lock and refuses three kinds of root;
* HR-07 checkpoint lineage resolves priors -> posttrain -> decoding/report explicitly;
* HR-08 a recipe naming something the confirmation would not apply is refused before any
  marker, opening or training. This is FAIL-CLOSED, not replay: replay of a qualified teacher
  or of a post-training method is not implemented and nothing here claims it is.
"""
from __future__ import annotations

import importlib.util
import json
import os
import types
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_routes_campaign as campaign
from smallAntibodyGen.experiments import her2_routes_lineage as lineage
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.tests.test_her2_routes_dplm_services import (  # noqa: F401  (fixtures)
    ROUTE_FULL,
    ROUTE_LORA,
    complete_stage,
    harness,
)

ROOT = Path(__file__).resolve().parents[3]


class SimpleContext:
    """The slice of ``RoutesContext`` the campaign machinery touches."""

    def __init__(self, root, *, config=None, revision="rev01"):
        self.run_root = Path(root) / "run"
        self.artifact_root = Path(root)
        self.repository_root = Path(root)
        self.config = config or {"revisions": {"current": "rev01"}}
        self.config_path = Path(root) / "config.json"
        self.revision = revision
        self.campaign_id = "probe"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def path(self, *parts):
        return self.run_root.joinpath(*[str(part) for part in parts])

    def artifact(self, logical):
        return self.artifact_root / str(logical)

    def repository(self, logical):
        return self.repository_root / str(logical)


def touch(path, content=b"weights"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# HR-01 -- the recipe reads the per-teacher document
# ---------------------------------------------------------------------------

def stage_everything(context, route):
    complete_stage(context, "route_gate", winner=route)
    complete_stage(context, "controls", survives=True)
    complete_stage(context, "priors", teachers=["functional_selection", "neighbour_mean"],
                   skipped={})
    paths.write_json(context.path("priors.json"), {
        "teachers": {"functional_selection": {"qualified": True},
                     "neighbour_mean": {"qualified": False}}})
    complete_stage(context, "posttrain", selected_method="conditional_fit")
    paths.write_json(context.path("posttrain.json"), {
        "route": route, "selection": {"best": "conditional_fit"}})
    complete_stage(context, "decoding")
    paths.write_json(context.path("decoding.json"), {
        "ranked_by_independent_quality": ["easy_first", "declared"]})


def test_frozen_recipe_yields_the_teacher_mapping_from_priors_json_not_the_name_list(harness):
    services, context = harness.services, harness.context
    stage_everything(context, ROUTE_FULL)
    record = paths.read_json(context.path("stages", "priors.json"))
    assert record["teachers"] == ["functional_selection", "neighbour_mean"], (
        "the realistic stage record carries a NAME LIST")
    with pytest.raises(ValueError, match="dictionary update sequence"):
        dict(record["teachers"])              # the audit's failure, exactly
    recipe = services.frozen_recipe()
    assert recipe["teachers"] == {"functional_selection": True, "neighbour_mean": False}
    assert recipe["posttrain_method"] == "conditional_fit"
    assert recipe["decoding_strategy"] == "easy_first"
    assert recipe["route"] == ROUTE_FULL and recipe["controls_survived"] is True
    body = {key: value for key, value in recipe.items() if key != "sha256"}
    assert recipe["sha256"] == paths.digest_document(body)


def test_frozen_recipe_refuses_a_teacher_document_that_is_not_a_mapping_or_is_absent(harness):
    services, context = harness.services, harness.context
    stage_everything(context, ROUTE_FULL)
    paths.write_json(context.path("priors.json"), {"teachers": ["functional_selection"]})
    with pytest.raises(ValueError, match="must be a mapping"):
        services.frozen_recipe()
    context.path("priors.json").unlink()
    with pytest.raises(lineage.LineageError, match="priors.json"):
        services.frozen_recipe()


@pytest.mark.parametrize("damage", ["posttrain_missing", "posttrain_no_selection",
                                    "posttrain_wrong_route", "decoding_missing",
                                    "decoding_no_ranking"])
def test_frozen_recipe_refuses_a_completed_stage_without_its_selection(harness, damage):
    services, context = harness.services, harness.context
    stage_everything(context, ROUTE_FULL)
    if damage == "posttrain_missing":
        context.path("posttrain.json").unlink()
    elif damage == "posttrain_no_selection":
        paths.write_json(context.path("posttrain.json"), {"route": ROUTE_FULL, "selection": {}})
        complete_stage(context, "posttrain")
    elif damage == "posttrain_wrong_route":
        paths.write_json(context.path("posttrain.json"), {
            "route": "other", "selection": {"best": "conditional_fit"}})
    elif damage == "decoding_missing":
        context.path("decoding.json").unlink()
    else:
        paths.write_json(context.path("decoding.json"), {"ranked_by_independent_quality": []})
    with pytest.raises(lineage.LineageError):
        services.frozen_recipe()


def test_frozen_recipe_keeps_an_unsupported_named_selection_for_the_replay_refusal(harness):
    services, context = harness.services, harness.context
    stage_everything(context, ROUTE_FULL)
    recipe = services.frozen_recipe()
    assert recipe["posttrain_method"] == "conditional_fit"
    assert not lineage.recipe_replay_plan(recipe)["replayable"]


# ---------------------------------------------------------------------------
# HR-03 -- a missing selected checkpoint stops the stage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["route_gate", "controls", "priors"])
def test_a_stage_that_reads_the_bakeoff_winner_stops_when_it_is_absent(tmp_path, stage):
    def path_for(*parts):
        return tmp_path.joinpath(*map(str, parts))
    with pytest.raises(lineage.LineageError, match="bake-off best checkpoint"):
        lineage.resolve_lineage(stage, route="r", path_for=path_for,
                                record_for=lambda name: None, document_for=lambda name: None)
    touch(path_for("bakeoff", "r", "best.pt"))
    found = lineage.resolve_lineage(stage, route="r", path_for=path_for,
                                    record_for=lambda name: None,
                                    document_for=lambda name: None)
    assert found["source"] == lineage.SOURCE_BAKEOFF and found["fallback"] is False


def test_arm_scores_stops_on_a_missing_winner_instead_of_scoring_the_backbone(harness,
                                                                              monkeypatch):
    services, context = harness.services, harness.context
    directory = context.path("bakeoff", ROUTE_FULL)
    directory.mkdir(parents=True)
    paths.write_json(directory / "run.json", {})
    monkeypatch.setattr(services, "development_panel_outcomes",
                        lambda: [{"window": "WGGDGFYAMD", "value": 1.0}])

    def forbidden(*args, **kwargs):
        raise AssertionError("scored a model although the winner's checkpoint is absent")
    monkeypatch.setattr(services, "score_panel", forbidden)
    with pytest.raises(lineage.LineageError, match="bake-off best checkpoint"):
        services.arm_scores(ROUTE_FULL)


def _stub_arm_inputs(harness, monkeypatch, *, trained, backbone, distance, generation):
    """Route arm_scores through stubbed model outputs; only the numeric handling is real."""
    from smallAntibodyGen.experiments import her2_routes_controls as controls_lib
    services, context = harness.services, harness.context
    directory = context.path("bakeoff", ROUTE_FULL)
    directory.mkdir(parents=True)
    paths.write_json(directory / "run.json", {})
    panel = [{"window": f"W{i}", "value": float(i)} for i in range(4)]
    monkeypatch.setattr(services, "development_panel_outcomes", lambda: panel)
    monkeypatch.setattr(services, "require_checkpoint", lambda *a, **k: directory / "best.pt")
    monkeypatch.setattr(services, "score_panel",
                        lambda route, checkpoint, panel_: trained if checkpoint else backbone)
    monkeypatch.setattr(controls_lib, "distance_only_baseline", lambda *a, **k: distance)
    monkeypatch.setattr(services, "comparator_scores",
                        lambda panel_: {"scores": {"cnn": 0.5}})
    monkeypatch.setattr(services, "generation_endpoint", lambda route, checkpoint: generation)
    monkeypatch.setitem(services.config.setdefault("her2", {}), "wild_type_core", "W")
    return services


RISING, FALLING, CONSTANT = [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0] * 4


def test_arm_scores_keeps_valid_values_and_a_genuine_zero_generation(harness, monkeypatch):
    services = _stub_arm_inputs(harness, monkeypatch, trained=RISING, backbone=FALLING,
                                distance=[1.0, 3.0, 2.0, 4.0], generation={"value": 0.0})
    scores = services.arm_scores(ROUTE_FULL)
    assert scores["ranking"] == pytest.approx(1.0)
    assert scores["backbone_ranking"] == pytest.approx(-1.0)
    assert scores["generation"] == 0.0


@pytest.mark.parametrize("generation", [{}, {"value": None}, {"value": float("nan")},
                                        {"value": float("inf")}])
def test_arm_scores_refuses_missing_generation_instead_of_zero(harness, monkeypatch, generation):
    services = _stub_arm_inputs(harness, monkeypatch, trained=RISING, backbone=FALLING,
                                distance=FALLING, generation=generation)
    with pytest.raises(ValueError, match="generation"):
        services.arm_scores(ROUTE_FULL)


@pytest.mark.parametrize("which, field", [("trained", "ranking"),
                                          ("backbone", "backbone_ranking"),
                                          ("distance", "distance_ranking")])
def test_arm_scores_refuses_undefined_or_invalid_correlations(harness, monkeypatch, which, field):
    vectors = dict(trained=RISING, backbone=FALLING, distance=FALLING)
    for bad in (CONSTANT, [1.0, float("nan"), 3.0, 4.0]):
        vectors[which] = bad
        services = _stub_arm_inputs(harness, monkeypatch, generation={"value": 0.1}, **vectors)
        with pytest.raises(ValueError, match=field):
            services.arm_scores(ROUTE_FULL)
        harness.context.path("bakeoff", ROUTE_FULL, "run.json").unlink()
        harness.context.path("bakeoff", ROUTE_FULL).rmdir()


def _stub_comparators(monkeypatch, *, additive, distance):
    from smallAntibodyGen.experiments import her2_baselines as baselines
    from smallAntibodyGen.experiments import her2_data
    monkeypatch.setattr(her2_data, "load_split", lambda root, split: _Split(**{"class": []}))
    monkeypatch.setattr(her2_data, "encode_cores", lambda seqs: list(seqs))
    monkeypatch.setattr(baselines, "class_targets", lambda column: None)
    fitted = types.SimpleNamespace(positive_scores=lambda index: additive)
    record = {"parameters": 1, "objective_evaluations": 1, "train_cross_entropy": 0.0}
    monkeypatch.setattr(baselines, "fit_additive_linear", lambda index, targets: (fitted, record))
    monkeypatch.setattr(baselines, "negative_wt_distance", lambda index: distance)


class _Split(dict):
    seq = []


def _forbid_generation(services, monkeypatch, calls):
    def forbidden(route, checkpoint):
        calls.append(route)
        raise AssertionError("generation began although ranking evidence was invalid")
    monkeypatch.setattr(services, "generation_endpoint", forbidden)


@pytest.mark.parametrize("which", ["trained", "backbone", "distance"])
@pytest.mark.parametrize("bad", [CONSTANT, [1.0, float("nan"), 3.0, 4.0]])
def test_invalid_ranking_refuses_before_generation(harness, monkeypatch, which, bad):
    vectors = dict(trained=RISING, backbone=FALLING, distance=FALLING)
    vectors[which] = bad
    services = _stub_arm_inputs(harness, monkeypatch, generation={"value": 0.1}, **vectors)
    calls = []
    _forbid_generation(services, monkeypatch, calls)
    with pytest.raises(ValueError):
        services.arm_scores(ROUTE_FULL)
    assert calls == []


@pytest.mark.parametrize("comparators", [{}, {"scores": {}}, {"scores": {"cnn": None}},
                                         {"scores": {"cnn": float("nan")}},
                                         {"scores": {"cnn": float("inf")}}])
def test_invalid_comparator_evidence_refuses_before_generation(harness, monkeypatch, comparators):
    services = _stub_arm_inputs(harness, monkeypatch, trained=RISING, backbone=FALLING,
                                distance=FALLING, generation={"value": 0.1})
    monkeypatch.setattr(services, "comparator_scores", lambda panel_: comparators)
    calls = []
    _forbid_generation(services, monkeypatch, calls)
    with pytest.raises(ValueError):
        services.arm_scores(ROUTE_FULL)
    assert calls == []


def test_valid_ranking_and_comparators_reach_generation_once(harness, monkeypatch):
    services = _stub_arm_inputs(harness, monkeypatch, trained=RISING, backbone=FALLING,
                                distance=[1.0, 3.0, 2.0, 4.0], generation={"value": 0.25})
    calls = []
    monkeypatch.setattr(services, "generation_endpoint",
                        lambda route, checkpoint: calls.append(route) or {"value": 0.25})
    scores = services.arm_scores(ROUTE_FULL)
    assert calls == [ROUTE_FULL]
    assert scores["generation"] == 0.25 and scores["comparators"] == {"cnn": 0.5}


def test_comparator_scores_never_turn_undefined_correlation_into_zero(harness, monkeypatch):
    services = harness.services
    monkeypatch.setitem(services.config.setdefault("roots", {}), "her2_raw", "her2_raw")
    panel = [{"window": f"W{i}", "value": float(i)} for i in range(4)]
    monkeypatch.setattr(services.context, "repository", lambda logical: logical, raising=False)
    _stub_comparators(monkeypatch, additive=RISING, distance=FALLING)
    valid = services.comparator_scores(panel)["scores"]
    assert valid == {"additive_linear": pytest.approx(1.0),
                     "negative_wt_distance": pytest.approx(-1.0)}
    _stub_comparators(monkeypatch, additive=CONSTANT, distance=FALLING)
    with pytest.raises(ValueError, match="additive_linear"):
        services.comparator_scores(panel)
    _stub_comparators(monkeypatch, additive=RISING, distance=[0.0, float("inf"), 1.0, 2.0])
    with pytest.raises(ValueError, match="scores"):
        services.comparator_scores(panel)


def test_the_reserved_opening_needs_every_seed_checkpoint_before_it_writes_its_guard(harness):
    services = harness.services
    with pytest.raises(lineage.LineageError, match="confirmation seed 1 best checkpoint"):
        services.open_reserved(ROUTE_FULL, seeds=[1, 2, 3], recipe={"sha256": "x"})
    assert not services.reserved_guard_path().exists(), (
        "the once-only guard must not be written for an opening that had nothing to score")


def test_the_backbone_comparator_is_reachable_only_by_an_explicit_none(harness):
    services = harness.services
    model, heads, record = services.load_for_scoring(ROUTE_FULL, None)
    assert record == {"checkpoint": None, "trained": False}
    with pytest.raises(ValueError, match="is absent"):
        services.load_for_scoring(ROUTE_FULL, harness.root / "no_such.pt")


def test_a_stage_that_lacks_its_checkpoint_is_blocked_not_crashed(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path)
    monkeypatch.setattr(campaign, "_guarded", lambda ctx, stage, body, **kw: body())

    def missing(heartbeat=None):
        raise lineage.LineageError("priors: the bake-off best checkpoint is absent")
    services = types.SimpleNamespace(run_priors=missing)
    record = campaign.run_stage(context, "priors", services, reuse=False)
    assert record["status"] == "blocked_readiness"
    assert "bake-off best checkpoint is absent" in record["reason"]
    assert campaign.read_stage_record(context, "priors")["status"] == "blocked_readiness"
    assert not context.path("priors.json").exists()


# ---------------------------------------------------------------------------
# HR-04 -- the selection guard covers all six stages
# ---------------------------------------------------------------------------

def write_revision(context, kind):
    paths.write_json(context.path(spec.REVISION_MARKER), {"kind": kind, "revision": "r"})


def test_the_six_selection_stages_are_the_ones_the_confirmation_freezes():
    assert spec.SELECTION_STAGES == ("bakeoff", "route_gate", "controls", "priors", "posttrain",
                                     "decoding")
    assert set(campaign.STRICT_PREREQUISITES["confirmation_freeze"]) <= set(
        spec.SELECTION_STAGES)


@pytest.mark.parametrize("stage", spec.SELECTION_STAGES)
def test_a_selection_stage_is_refused_under_the_confirmation_revision(tmp_path, stage):
    context = SimpleContext(tmp_path)
    write_revision(context, spec.REVISION_CONFIRMATION)
    with pytest.raises(ValueError, match="selection stage"):
        spec.require_revision_discipline(context, stage=stage)
    write_revision(context, spec.REVISION_DEVELOPMENT)
    assert spec.require_revision_discipline(context, stage=stage)["kind"] == \
        spec.REVISION_DEVELOPMENT


@pytest.mark.parametrize("stage", ["acquire", "preflight", "report", "verify"])
def test_a_non_selection_stage_is_not_refused_under_the_confirmation_revision(tmp_path, stage):
    context = SimpleContext(tmp_path)
    write_revision(context, spec.REVISION_CONFIRMATION)
    assert spec.require_revision_discipline(context, stage=stage)["stage"] == stage


def test_confirm_still_needs_a_confirmation_revision_and_a_frozen_selection(tmp_path):
    context = SimpleContext(tmp_path)
    write_revision(context, spec.REVISION_DEVELOPMENT)
    with pytest.raises(ValueError, match="requires a 'confirmation' revision"):
        spec.require_revision_discipline(context, stage="confirm", selection_frozen=True)
    write_revision(context, spec.REVISION_CONFIRMATION)
    with pytest.raises(ValueError, match="selection is not frozen"):
        spec.require_revision_discipline(context, stage="confirm")
    assert spec.require_revision_discipline(context, stage="confirm", selection_frozen=True)


# ---------------------------------------------------------------------------
# HR-05 -- an unknown lock is unknown
# ---------------------------------------------------------------------------

def fake_lock(state):
    return lambda path: {"state": state, "path": "campaign.lock", "recorded_owner": None}


def test_an_inaccessible_lock_is_unknown_not_stopped(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path)
    progress = context.path("bakeoff", "route", "progress.json")
    paths.write_json(progress, {"completed": 4324})
    monkeypatch.setattr(campaign, "lock_state", fake_lock("inaccessible"))
    health = campaign.launch_health(context)
    assert health["worker_running"] is None
    assert health["health"] == "unknown_lock_state"
    assert health["journalled_steps"] == 4324 and health["progress_observed"] is True
    monkeypatch.setattr(campaign, "lock_state", fake_lock("held"))
    live = campaign.launch_health(context)
    assert live["worker_running"] is True and live["health"] == "running_with_progress"
    monkeypatch.setattr(campaign, "lock_state", fake_lock("released"))
    stopped = campaign.launch_health(context)
    assert stopped["worker_running"] is False and stopped["health"] == "not_running"


def test_a_held_lock_without_progress_and_a_missing_lock_read_correctly(tmp_path):
    context = SimpleContext(tmp_path)
    assert campaign.launch_health(context)["health"] == "not_running"
    holder = campaign.CampaignLock(context.path(campaign.LOCK_FILE))
    with holder.held():
        health = campaign.launch_health(context)
        assert health["worker_running"] is True
        assert health["health"] == "running_without_progress_yet"


@pytest.mark.parametrize("state", ["inaccessible", "held"])
def test_an_unknown_or_held_lock_never_authorises_a_detached_launch(tmp_path, monkeypatch,
                                                                    state):
    context = SimpleContext(tmp_path)
    touch(tmp_path / "scripts" / "her2_routes_supervisor.py", b"# supervisor")
    launched = []
    monkeypatch.setattr(campaign.subprocess, "Popen",
                        lambda *a, **k: launched.append(a) or types.SimpleNamespace(pid=1))
    monkeypatch.setattr(campaign, "lock_state", fake_lock(state))
    with pytest.raises(ValueError, match=state if state == "inaccessible" else "live writer"):
        campaign.launch_detached(context)
    assert launched == []
    for free in ("missing", "released"):
        monkeypatch.setattr(campaign, "lock_state", fake_lock(free))
        record = campaign.launch_detached(context)
        assert record["pid"] == 1
    assert len(launched) == 2


def load_status_script():
    target = ROOT / "scripts" / "her2_routes_status.py"
    specification = importlib.util.spec_from_file_location("her2_routes_status_script", target)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def tree(root):
    return sorted(str(path.relative_to(root)) for path in Path(root).rglob("*")) \
        if Path(root).exists() else None


def test_status_is_read_only_prints_the_resolved_root_and_names_sibling_revisions(
        tmp_path, capsys):
    artifact = tmp_path / "artifacts"
    sibling = artifact / "runs" / "t1_rev01" / "flight" / "stages"
    sibling.mkdir(parents=True)
    for stage in ("acquire", "preflight", "split"):
        paths.write_json(sibling / f"{stage}.json", {"status": "completed"})
    before, environment = tree(artifact), {name: os.environ.get(name) for name in (
        "HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "CUBLAS_WORKSPACE_CONFIG", "TEMP")}
    script = load_status_script()
    code = script.main(["--artifact-root", str(artifact), "--revision", "rev_empty"])
    output = capsys.readouterr().out
    assert code == 0
    assert tree(artifact) == before, "status created or wrote something"
    assert {name: os.environ.get(name) for name in environment} == environment, (
        "status prepared the environment")
    assert "rev_empty" in output and str(artifact / "runs" / "rev_empty") in output
    assert "0 stage record(s)" in output and "does not mean no flight exists" in output
    assert "t1_rev01 (3 records" in output
    code = script.main(["--artifact-root", str(artifact), "--revision", "rev_empty", "--json"])
    document = json.loads(capsys.readouterr().out)
    assert document["run_root"]["exists"] is False and document["run_root"]["empty"] is True
    assert [row["revision"] for row in document["run_root"]["sibling_revisions"]] == ["t1_rev01"]
    assert document["run_root"]["sibling_revisions"][0]["stage_records"] == 3
    assert tree(artifact) == before


def test_status_never_creates_a_missing_artifact_root_or_reports_unknown_as_stopped(
        tmp_path, capsys, monkeypatch):
    artifact = tmp_path / "does_not_exist"
    script = load_status_script()
    assert script.main(["--artifact-root", str(artifact)]) == 0
    assert not artifact.exists()
    capsys.readouterr()
    monkeypatch.setattr(campaign, "lock_state", fake_lock("inaccessible"))
    assert script.main(["--artifact-root", str(artifact)]) == 0
    output = capsys.readouterr().out
    assert "unknown_lock_state" in output and "UNKNOWN, not 'stopped'" in output
    assert "not_running" not in output


# ---------------------------------------------------------------------------
# HR-06 -- the standalone profiler
# ---------------------------------------------------------------------------

PROFILE_CONFIG = {
    "revisions": {"current": "rev01"},
    "profile": {"microbatches": [1, 2], "steps": 3},
    "routes": {"enabled": ["ablang2_masked_editor", "dwjs_energy"],
               "ablang2_masked_editor": {"updates": 11},
               "dwjs_energy": {"denoiser_updates": 5, "energy_updates": 7}}}


class FakeProfiler:
    def __init__(self):
        self.profile_calls, self.energy_calls, self.schedule = [], 0, None

    def profile(self, route, *, microbatches, steps):
        self.profile_calls.append((route, tuple(microbatches), steps))
        return {"route": route, "rows": [], "chosen_microbatch": 1}

    def profile_energy(self):
        self.energy_calls += 1
        return {"route": "dwjs_energy", "rows": [], "chosen_microbatch": 1}

    def ledger_sample(self):
        return {"forwards_per_row": 1.0}

    def forecast(self, profiles, *, schedule, ledger_sample):
        self.schedule = {name: dict(block) for name, block in schedule.items()}
        return {"per_route": {}, "total_hours": 0.0, "with_reserve_hours": 0.0,
                "drop_order": []}


def test_the_standalone_profiler_and_the_stage_pass_identical_arguments(tmp_path):
    stage_context = SimpleContext(tmp_path / "stage", config=json.loads(json.dumps(PROFILE_CONFIG)))
    stage_fake = FakeProfiler()
    campaign.stage_profile(stage_context, stage_fake)
    lone_context = SimpleContext(tmp_path / "lone", config=json.loads(json.dumps(PROFILE_CONFIG)))
    lone_fake = FakeProfiler()
    profiles, forecast = campaign.standalone_profile(lone_context, lambda: lone_fake)
    assert lone_fake.profile_calls == stage_fake.profile_calls == [
        ("ablang2_masked_editor", (1, 2), 3)], (
        "the script must use the config's microbatches and steps, exactly as the stage does")
    assert lone_fake.schedule == stage_fake.schedule == {
        "ablang2_masked_editor": {"updates": 11}, "dwjs_energy": {"updates": 12}}, (
        "the energy schedule is denoiser + energy updates, not energy alone")
    assert lone_fake.energy_calls == stage_fake.energy_calls == 1
    assert lone_context.path(spec.PROFILE_JSON).is_file()
    assert lone_context.path(spec.FORECAST_JSON).is_file()


def test_the_standalone_profiler_holds_the_real_lock_for_its_whole_run(tmp_path):
    context = SimpleContext(tmp_path, config=json.loads(json.dumps(PROFILE_CONFIG)))
    observed = {}

    def build():
        state = campaign.lock_state(context.path(campaign.LOCK_FILE))
        observed["during"] = state["state"]
        with pytest.raises(ValueError, match="Another campaign process holds the write lock"):
            campaign.CampaignLock(context.path(campaign.LOCK_FILE)).acquire()
        return FakeProfiler()
    campaign.standalone_profile(context, build, routes=["ablang2_masked_editor"])
    assert observed["during"] == "held"
    assert campaign.lock_state(context.path(campaign.LOCK_FILE))["state"] == "released"


def refuse_build():
    raise AssertionError("the services were constructed although the guard should have refused")


def test_a_completed_profile_stage_is_refused_before_any_service_is_built(tmp_path):
    context = SimpleContext(tmp_path, config=json.loads(json.dumps(PROFILE_CONFIG)))
    complete_stage(context, "profile")
    with pytest.raises(ValueError, match="profile stage already completed"):
        campaign.standalone_profile(context, refuse_build)
    assert not context.path(spec.PROFILE_JSON).exists()


def test_a_frozen_root_is_refused_before_any_service_is_built(tmp_path):
    context = SimpleContext(tmp_path, config=json.loads(json.dumps(PROFILE_CONFIG)))
    paths.write_json(context.path(spec.FREEZE_MARKER), {"snapshot_sha256": "s"})
    with pytest.raises(ValueError, match="root is frozen"):
        campaign.standalone_profile(context, refuse_build)
    assert not context.path(spec.PROFILE_JSON).exists()


def test_a_held_lock_is_refused_and_the_holder_is_undisturbed(tmp_path):
    context = SimpleContext(tmp_path, config=json.loads(json.dumps(PROFILE_CONFIG)))
    holder = campaign.CampaignLock(context.path(campaign.LOCK_FILE))
    with holder.held():
        with pytest.raises(ValueError, match="'held'"):
            campaign.standalone_profile(context, refuse_build)
        assert campaign.lock_state(context.path(campaign.LOCK_FILE))["state"] == "held"


def test_an_inaccessible_lock_is_refused_not_treated_as_free(tmp_path, monkeypatch):
    context = SimpleContext(tmp_path, config=json.loads(json.dumps(PROFILE_CONFIG)))
    monkeypatch.setattr(campaign, "lock_state", fake_lock("inaccessible"))
    with pytest.raises(ValueError, match="could not be observed"):
        campaign.standalone_profile(context, refuse_build)


def test_there_is_no_force_flag_on_the_profiler_script():
    target = ROOT / "scripts" / "her2_routes_gpu_profile.py"
    specification = importlib.util.spec_from_file_location("her2_routes_gpu_profile_script",
                                                           target)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    flags = {option for action in module.build_parser()._actions
             for option in action.option_strings}
    assert not {flag for flag in flags if "force" in flag or "override" in flag}
    assert {"--config", "--artifact-root", "--revision", "--device", "--route"} <= flags


# ---------------------------------------------------------------------------
# HR-07 -- checkpoint lineage
# ---------------------------------------------------------------------------

class Lineage:
    """A tiny run root: files, stage records and documents, resolved through the real code."""

    def __init__(self, root, route="r"):
        self.root, self.route = Path(root), route
        self.records, self.documents = {}, {}

    def path_for(self, *parts):
        return self.root.joinpath(*map(str, parts))

    def resolve(self, stage):
        return lineage.resolve_lineage(stage, route=self.route, path_for=self.path_for,
                                       record_for=self.records.get,
                                       document_for=self.documents.get)


def test_priors_start_from_the_bakeoff_best_and_record_its_hash(tmp_path):
    run = Lineage(tmp_path)
    best = touch(run.path_for("bakeoff", "r", "best.pt"), b"bakeoff-bytes")
    found = run.resolve("priors")
    assert found["source"] == "bakeoff_best" and found["path"] == str(best)
    assert found["sha256"] == paths.sha256_file(best) and found["fallback"] is False


def test_posttrain_starts_from_the_distilled_model_when_priors_completed_with_one(tmp_path):
    run = Lineage(tmp_path)
    touch(run.path_for("bakeoff", "r", "best.pt"), b"a")
    distilled = touch(run.path_for("priors", "r", "distilled.pt"), b"b")
    run.records["priors"] = {"status": "completed"}
    run.documents["priors"] = {
        "teachers": {"functional_selection": {"qualified": True}},
        "distillations": {"functional_selection": {"checkpoint": str(distilled)}}}
    found = run.resolve("posttrain")
    assert found["source"] == "priors_distilled" and found["path"] == str(distilled)
    assert found["fallback"] is False and "functional_selection" in found["reason"]
    assert found["sha256"] == paths.sha256_file(distilled)


@pytest.mark.parametrize("case, expected", [
    ("priors_not_run", "priors stage has not completed"),
    ("priors_no_distillation", "without a distilled checkpoint"),
])
def test_posttrain_falls_back_to_the_bakeoff_and_says_so(tmp_path, case, expected):
    run = Lineage(tmp_path)
    best = touch(run.path_for("bakeoff", "r", "best.pt"), b"a")
    if case != "priors_not_run":
        run.records["priors"] = {"status": "completed"}
        run.documents["priors"] = {"distillations": {},
                                   "teachers": {"functional_selection": {"qualified": False}}}
    found = run.resolve("posttrain")
    assert found["source"] == "bakeoff_best" and found["path"] == str(best)
    assert found["fallback"] is True, "a fallback is never silent"
    assert expected in found["reason"]
    assert found["considered"], "what was considered and passed over is recorded"


@pytest.mark.parametrize("case", ["file_missing", "wrong_path", "no_checkpoint", "document_absent",
                                  "document_wrong_route", "qualified_not_distilled",
                                  "empty_document", "teachers_missing", "teachers_not_mapping",
                                  "decision_not_mapping", "decision_without_boolean"])
def test_posttrain_refuses_an_invalid_completed_priors_stage(tmp_path, case):
    run = Lineage(tmp_path)
    touch(run.path_for("bakeoff", "r", "best.pt"), b"a")
    other = touch(run.path_for("elsewhere.pt"), b"x")
    run.records["priors"] = {"status": "completed"}
    named = str(run.path_for("priors", "r", "distilled.pt"))
    document = {"teachers": {"functional_selection": {"qualified": True}},
                "distillations": {"functional_selection": {"checkpoint": named}}}
    if case == "empty_document":
        document = {}  # the reproduced failure: an empty completed document fell back
    elif case == "teachers_missing":
        document = {"distillations": {}}
    elif case == "teachers_not_mapping":
        document = {"teachers": ["functional_selection"], "distillations": {}}
    elif case == "decision_not_mapping":
        document = {"teachers": {"functional_selection": True}, "distillations": {}}
    elif case == "decision_without_boolean":
        document = {"teachers": {"functional_selection": {"qualified": "yes"}},
                    "distillations": {}}
    elif case == "wrong_path":
        document["distillations"]["functional_selection"]["checkpoint"] = str(other)
    elif case == "no_checkpoint":
        document["distillations"]["functional_selection"] = {}
    elif case == "document_wrong_route":
        document["route"] = "other"
        touch(run.path_for("priors", "r", "distilled.pt"), b"b")
    elif case == "qualified_not_distilled":
        document = {"distillations": {}, "teachers": {"functional_selection": {"qualified": True}}}
    if case != "document_absent":
        run.documents["priors"] = document
    with pytest.raises(lineage.LineageError):
        run.resolve("posttrain")


def test_two_distilled_teachers_behind_one_path_are_ambiguous_not_guessed(tmp_path):
    run = Lineage(tmp_path)
    touch(run.path_for("bakeoff", "r", "best.pt"))
    distilled = touch(run.path_for("priors", "r", "distilled.pt"))
    run.records["priors"] = {"status": "completed"}
    run.documents["priors"] = {
        "teachers": {"functional_selection": {"qualified": True},
                     "another_teacher": {"qualified": True}},
        "distillations": {
            "functional_selection": {"checkpoint": str(distilled)},
            "another_teacher": {"checkpoint": str(distilled)}}}
    with pytest.raises(lineage.LineageError, match="ambiguous"):
        run.resolve("posttrain")


@pytest.mark.parametrize("teachers", [
    {"functional_selection": {"qualified": False}},        # explicitly rejected
    {},                                                    # unrecorded
    {"other_teacher": {"qualified": True}},                # a different teacher's decision
])
def test_posttrain_refuses_a_distillation_from_a_teacher_that_did_not_qualify(tmp_path, teachers):
    run = Lineage(tmp_path)
    touch(run.path_for("bakeoff", "r", "best.pt"), b"a")
    distilled = touch(run.path_for("priors", "r", "distilled.pt"), b"b")
    run.records["priors"] = {"status": "completed"}
    document = {"teachers": teachers,
                "distillations": {"functional_selection": {"checkpoint": str(distilled)}}}
    if "other_teacher" in teachers:
        document["distillations"]["other_teacher"] = {}
        document["distillations"]["other_teacher"]["checkpoint"] = str(distilled)
        document["teachers"]["functional_selection"] = {"qualified": False}
    run.documents["priors"] = document
    with pytest.raises(lineage.LineageError):
        run.resolve("posttrain")


@pytest.mark.parametrize("distillations", [None, [], "none", 0])
def test_a_present_but_malformed_distillations_value_is_refused_not_normalised(
        tmp_path, distillations):
    with pytest.raises(lineage.LineageError, match="not a mapping"):
        lineage.distilled_checkpoints(
            {"teachers": {"t": {"qualified": False}}, "distillations": distillations},
            route="r", path_for=lambda *parts: tmp_path.joinpath(*parts))


def test_an_absent_distillations_key_still_means_none_for_a_no_teacher_outcome(tmp_path):
    assert lineage.distilled_checkpoints(
        {"teachers": {"t": {"qualified": False}}}, route="r",
        path_for=lambda *parts: tmp_path.joinpath(*parts)) == []


@pytest.mark.parametrize("decision", [{}, {"qualified": None}, {"qualified": "false"},
                                      {"qualified": 1}, {"qualified": 0}, "yes"])
def test_teacher_decisions_do_not_coerce_a_missing_or_nonboolean_qualification(decision):
    with pytest.raises(ValueError):
        lineage.teacher_decisions({"teachers": {"t": decision}})


def test_teacher_decisions_keep_explicit_booleans_and_feed_the_recipe_plan():
    decisions = lineage.teacher_decisions(
        {"teachers": {"b": {"qualified": False}, "a": {"qualified": True}}})
    assert decisions == {"a": True, "b": False}
    plan = lineage.recipe_replay_plan({"teachers": decisions})
    assert plan["unreplayed"] == ["teacher::a"] and plan["elements"]["teacher::b"] == "not_applicable"
    with pytest.raises(lineage.RecipeNotReplayable):
        lineage.require_replayable({"teachers": decisions}, where="freeze")
    assert lineage.require_replayable({"teachers": {"b": False}}, where="freeze")["replayable"]


def selected_run(tmp_path, method="local_ipo"):
    run = Lineage(tmp_path)
    touch(run.path_for("bakeoff", "r", "best.pt"), b"bakeoff")
    last = touch(run.path_for("posttrain", method, "last.pt"), b"posttrained")
    run.records["posttrain"] = {"status": "completed", "selected_method": method}
    run.documents["posttrain"] = {"route": "r", "selection": {"best": method}}
    return run, last


@pytest.mark.parametrize("stage", ["decoding", "report"])
def test_decoding_and_the_report_read_the_selected_posttrained_checkpoint(tmp_path, stage):
    run, last = selected_run(tmp_path)
    found = run.resolve(stage)
    assert found["source"] == "posttrain_selected" and found["path"] == str(last)
    assert found["sha256"] == paths.sha256_file(last)
    assert "local_ipo" in found["reason"] and found["fallback"] is False


@pytest.mark.parametrize("stage", ["decoding", "report"])
def test_decoding_and_the_report_never_fall_back_to_the_bakeoff(tmp_path, stage):
    run, _ = selected_run(tmp_path)
    run.records["posttrain"] = {"status": "blocked_readiness"}
    with pytest.raises(lineage.LineageError, match="does not fall back|posttrain stage is"):
        run.resolve(stage)
    run.records["posttrain"] = {"status": "completed"}
    run.documents["posttrain"] = {"route": "r", "selection": {"best": None}}
    with pytest.raises(lineage.LineageError, match="no selected method"):
        run.resolve(stage)
    run.documents["posttrain"] = {"route": "other", "selection": {"best": "local_ipo"}}
    with pytest.raises(lineage.LineageError, match="was produced for route"):
        run.resolve(stage)
    run.documents["posttrain"] = {"route": "r", "selection": {"best": "local_dpo"}}
    with pytest.raises(lineage.LineageError, match="post-trained checkpoint \\(local_dpo\\)"):
        run.resolve(stage)
    with pytest.raises(lineage.LineageError, match="No checkpoint lineage is defined"):
        run.resolve("acquire")


# ---------------------------------------------------------------------------
# HR-08 -- fail closed on a recipe that cannot be replayed
# ---------------------------------------------------------------------------

def recipe(**changes):
    body = {"route": "r", "arm": "full", "updates": 2, "teachers": {"functional_selection": False},
            "posttrain_method": None, "decoding_strategy": None}
    body.update(changes)
    body["sha256"] = paths.digest_document({k: v for k, v in body.items() if k != "sha256"})
    return body


def test_the_replay_plan_classifies_every_element():
    clean = lineage.recipe_replay_plan(recipe())
    assert clean["replayable"] and clean["unreplayed"] == []
    assert clean["elements"]["base_curriculum"] == "replayed"
    assert clean["elements"]["teacher::functional_selection"] == "not_applicable"
    named = lineage.recipe_replay_plan(recipe(
        teachers={"functional_selection": True, "neighbour_mean": False},
        posttrain_method="conditional_fit", decoding_strategy="easy_first"))
    assert not named["replayable"]
    assert named["unreplayed"] == ["decoding_strategy", "posttrain_method",
                                   "teacher::functional_selection"]
    assert named["elements"]["teacher::neighbour_mean"] == "not_applicable"
    assert "not implemented" in named["gap"]
    for changed in ({"posttrain_method": "local_dpo"}, {"decoding_strategy": "declared"},
                    {"teachers": {"t": True}}):
        assert not lineage.recipe_replay_plan(recipe(**changed))["replayable"]


def freeze_ready(tmp_path):
    context = SimpleContext(tmp_path)
    for stage in ("route_gate", "controls", "priors", "posttrain", "decoding"):
        complete_stage(context, stage)
    paths.write_json(context.path(spec.FREEZE_MARKER), {"snapshot_sha256": "s", "file_count": 1})
    paths.write_json(context.path(spec.EXTERNAL_MARKER),
                     {"external_sha256": "e", "file_count": 1})
    return context


def test_an_unreplayable_recipe_is_refused_before_any_marker_or_recipe_file_exists(tmp_path):
    context = freeze_ready(tmp_path)
    services = types.SimpleNamespace(frozen_recipe=lambda: recipe(posttrain_method="local_ipo"))
    with pytest.raises(campaign.ReadinessGate, match="posttrain_method"):
        campaign.stage_confirmation_freeze(context, services)
    assert not context.path(spec.REVISION_MARKER).exists()
    assert not context.path("frozen_recipe.json").exists()
    record = campaign.run_stage(context, "confirmation_freeze", services, reuse=False)
    assert record["status"] == "blocked_readiness"
    assert "not implemented" in record["reason"] and "Nothing was frozen" in record["reason"]
    assert not context.path(spec.REVISION_MARKER).exists()
    assert not context.path("frozen_recipe.json").exists()


def test_a_recipe_that_cannot_be_assembled_leaves_no_half_frozen_revision(tmp_path):
    context = freeze_ready(tmp_path)

    def explode():
        raise lineage.LineageError("priors.json is absent")
    record = campaign.run_stage(context, "confirmation_freeze",
                                types.SimpleNamespace(frozen_recipe=explode), reuse=False)
    assert record["status"] == "blocked_readiness" and "priors.json" in record["reason"]
    assert not context.path(spec.REVISION_MARKER).exists(), (
        "the marker used to be written first, so an assembly failure left a confirmation "
        "revision with no recipe under it")
    assert not context.path("frozen_recipe.json").exists()


def test_a_replayable_recipe_freezes_the_revision_and_the_recipe_together(tmp_path):
    context = freeze_ready(tmp_path)
    frozen = recipe()
    record = campaign.stage_confirmation_freeze(
        context, types.SimpleNamespace(frozen_recipe=lambda: frozen))
    assert record["recipe_sha256"] == frozen["sha256"] and record["replay_plan"]["replayable"]
    assert paths.read_json(context.path(spec.REVISION_MARKER))["kind"] == \
        spec.REVISION_CONFIRMATION
    assert paths.read_json(context.path("frozen_recipe.json"))["sha256"] == frozen["sha256"]


def write_frozen(context, document, *, record_sha=None):
    paths.write_json(context.path("frozen_recipe.json"), document)
    complete_stage(context, "confirmation_freeze", recipe_sha256=record_sha or document["sha256"])


def fail_if_trained(*args, **kwargs):
    raise AssertionError("a confirmation seed was trained although the recipe is unreplayable")


def test_the_confirmation_refuses_an_unreplayable_recipe_before_training_or_opening(
        harness, monkeypatch):
    services, context = harness.services, harness.context
    monkeypatch.setattr(services, "train", fail_if_trained)
    monkeypatch.setattr(services, "mint_reserved_unlock", fail_if_trained)
    write_frozen(context, recipe(route=ROUTE_FULL, posttrain_method="conditional_fit"))
    with pytest.raises(lineage.RecipeNotReplayable, match="posttrain_method"):
        services.run_confirmation()
    assert not context.path("confirm").exists()
    assert not services.reserved_guard_path().exists()


def test_the_confirmation_consumes_the_frozen_recipe_and_never_reassembles_it(harness,
                                                                               monkeypatch):
    services, context = harness.services, harness.context
    monkeypatch.setattr(services, "train", fail_if_trained)
    monkeypatch.setattr(services, "frozen_recipe", fail_if_trained)
    with pytest.raises(lineage.LineageError, match="never silently re-assembled"):
        services.run_confirmation()
    document = recipe(route=ROUTE_FULL)
    document["route"] = "tampered"                    # digest no longer matches the content
    write_frozen(context, document, record_sha=document["sha256"])
    with pytest.raises(ValueError, match="edited after it was frozen"):
        services.run_confirmation()
    write_frozen(context, recipe(route=ROUTE_FULL), record_sha="0" * 64)
    with pytest.raises(ValueError, match="names recipe"):
        services.run_confirmation()


def test_a_replayable_confirmation_records_what_it_applied_beside_the_recipe(harness,
                                                                              monkeypatch):
    services, context = harness.services, harness.context
    trained = []

    def fake_train(route, *, stream, updates, seed, directory, heartbeat=None, evaluate=None,
                   **kwargs):
        trained.append((route, seed))
        return {"updates": updates}
    monkeypatch.setattr(services, "train", fake_train)
    monkeypatch.setattr(services, "curriculum_stream", lambda route, *, seed, updates: None)
    monkeypatch.setattr(services, "development_metric", lambda route: None)
    monkeypatch.setattr(services, "split", lambda: types.SimpleNamespace(reserved=("A", "B")))
    frozen = recipe(route=ROUTE_FULL)
    write_frozen(context, frozen)
    result = services.run_confirmation()
    assert trained == [(ROUTE_FULL, 1), (ROUTE_FULL, 2), (ROUTE_FULL, 3)]
    assert result["recipe"]["sha256"] == frozen["sha256"]
    applied = result["applied"]
    assert applied["route"] == ROUTE_FULL and applied["base_curriculum"] is True
    assert applied["teachers"] == [] and applied["posttrain_method"] is None
    assert applied["decoding_strategy"] is None and applied["replay_plan"]["replayable"]
    assert result["complete"] is True and result["reserved_opened"] is False
    assert not services.reserved_guard_path().exists()
