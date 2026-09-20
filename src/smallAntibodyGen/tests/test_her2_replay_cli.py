"""The replay CLI's own decisions: which device, which artifacts must exist, what cost.

The stages themselves need real weights and a real run directory. What can be
checked here is the part of the CLI that decides things: that a ``--device`` flag
cannot quietly move a stage off the declared device, that the expected-output list
comes from what was declared and reached rather than from a directory listing, that
a completion-manifest key resolves back to a file under both of the conventions the
ledger actually holds, and that bank preparation is in the cost table rather than
free.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallAntibodyGen.experiments import her2_replay_campaign as campaign
from smallAntibodyGen.experiments import her2_support_paths as paths


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/posttrain_her2_replay.py"
spec = importlib.util.spec_from_file_location("posttrain_her2_replay", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

CONFIG = {"screen": {"tasks": ["continued_sft", "ipo"],
                     "replay_lambdas": [0.0, 1.0],
                     "parent_seeds": [20260918]},
          "inference": {"device": "cuda"}}


def context_for(tmp_path, config=None):
    run = paths.RunPaths.create(tmp_path, tmp_path / "outputs/replay", logical="outputs/replay")
    return SimpleNamespace(repository_root=tmp_path, run=run,
                           config=dict(CONFIG, **(config or {})),
                           seeds=[20260918])


# ---------------------------------------------------------------------------
# the device is declared, not chosen at the command line
# ---------------------------------------------------------------------------

def test_the_declared_device_is_confirmed_by_the_flag_and_never_changed(tmp_path):
    context = context_for(tmp_path, {"inference": {"device": "cpu"}})
    assert cli.choose_device(context, "auto") == "cpu"
    assert cli.choose_device(context, "cpu") == "cpu"
    with pytest.raises(ValueError, match="different device than the config declares"):
        cli.choose_device(context, "cuda")


def test_a_cpu_override_cannot_move_a_cuda_screen_off_its_declared_device(tmp_path):
    context = context_for(tmp_path)
    with pytest.raises(ValueError, match="different device than the config declares"):
        cli.choose_device(context, "cpu")


# ---------------------------------------------------------------------------
# what a verification is held to
# ---------------------------------------------------------------------------

def staged_run(context):
    """The stage documents that necessarily precede any trajectory directory."""
    paths.write_json(context.run.path("training_spec_frozen.json"),
                     {"record_kind": "training_spec_frozen", "git": {"commit": "a" * 40}})
    for name in ("replay_config.json", "preflight.json"):
        paths.write_json(context.run.path(name), {"record_kind": name})
    paths.write_json(context.run.path("banks_manifest.json"), {"banks": {}})


def monitor_journal(directory, *, update, passing=True, failed=False):
    """A monitor journal holding the check records the loop actually writes."""
    records = [{"record_kind": "gate_verdict", "update": update, "passed": passing}]
    if passing:
        rolling = directory / campaign.LAST_PASSING
        records.append({"record_kind": "snapshot", "update": update, "passed": True,
                        "last_passing": {"file": campaign.LAST_PASSING,
                                         "sha256": paths.sha256_file(rolling),
                                         "state_sha256": "b" * 64, "update": update,
                                         "check": 41}})
    if failed:
        records.append({"record_kind": "snapshot", "update": update, "passed": False,
                        "failed_state": {"file": campaign.FAILED_STATE, "sha256": "c" * 64}})
    (directory / campaign.MONITOR_JSONL).write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8", newline="\n")


def reached_trajectory(context, name, *, update=1000, terminal=True, stop_reason=None):
    """A trajectory directory with one reached endpoint and its artifacts."""
    staged_run(context)
    directory = context.run.path(f"trajectories/{name}")
    (directory / f"endpoints/update{update}").mkdir(parents=True, exist_ok=True)
    for leaf in (campaign.IDENTITY_JSON, campaign.UPDATES_JSONL):
        (directory / leaf).write_text("{}\n", encoding="utf-8", newline="\n")
    (directory / campaign.ENDPOINTS_JSONL).write_text(
        json.dumps({"record_kind": "exposure_endpoint", "update": update}) + "\n",
        encoding="utf-8", newline="\n")
    (directory / f"endpoint_update{update}.pt").write_bytes(b"weights")
    (directory / campaign.LAST_PASSING).write_bytes(b"rolling")
    monitor_journal(directory, update=update)
    if terminal:
        paths.write_json(directory / campaign.STATUS_JSON, {
            "schema_version": campaign.TRAJECTORY_SCHEMA, "record_kind": "trajectory_status",
            "status": campaign.STATUS_COMPLETED, "stop_reason": stop_reason,
            "updates": update, "checks": 41, "exposures": {"chosen": update * 64},
            "endpoints_reached": {str(update): {"record_kind": "exposure_endpoint",
                                                "update": update}}})
    return directory


def stopped_at_first_check(context, name, *, update=25):
    """The legitimate first-gate failure: one check, none passed, a diagnostic state."""
    staged_run(context)
    directory = context.run.path(f"trajectories/{name}")
    directory.mkdir(parents=True, exist_ok=True)
    for leaf in (campaign.IDENTITY_JSON, campaign.UPDATES_JSONL):
        (directory / leaf).write_text("{}\n", encoding="utf-8", newline="\n")
    (directory / campaign.FAILED_STATE).write_bytes(b"diagnostic weights")
    monitor_journal(directory, update=update, passing=False, failed=True)
    paths.write_json(directory / campaign.STATUS_JSON, {
        "schema_version": campaign.TRAJECTORY_SCHEMA, "record_kind": "trajectory_status",
        "status": campaign.STATUS_STOPPED, "stop_reason": "parent_relative_likelihood_breach",
        "updates": update, "checks": 1, "exposures": {"chosen": update * 64},
        "last_passing": None, "endpoints_reached": {}})
    return directory


def test_the_expected_outputs_come_from_what_was_declared_and_reached(tmp_path):
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    reached_trajectory(context, name)
    coverage = cli.expected_run_outputs(context, queue)
    expected = set(coverage["expected"])
    assert context.run.logical(f"trajectories/{name}/endpoint_update1000.pt") in expected
    assert context.run.logical(
        f"trajectories/{name}/endpoints/update1000/endpoint.json") in expected
    assert f"trajectories/{name}/endpoints/update1000/evaluation.npz" in expected, (
        "shard keys are run-relative, which is the convention write_shard records")
    assert context.run.logical(f"trajectories/{name}/{campaign.STATUS_JSON}") in expected, (
        "a terminal status is written once and is what the trajectory claims it reached")
    assert coverage["problems"] == []
    assert [row["kind"] for row in coverage["rolling"]] == ["rolling_latest_state"]
    assert coverage["rolling"][0]["matches_latest_snapshot"] is True


def test_a_first_check_that_failed_is_not_asked_for_a_last_passing_state(tmp_path):
    """One check, no passing check, no rolling state -- and that is not a problem.

    Requiring ``last_passing`` from ``checks > 0`` reported a false problem on the
    one outcome this screen declares in advance. The requirement comes from a
    recorded *passing* snapshot, of which there are none here.
    """
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    stopped_at_first_check(context, name)
    coverage = cli.expected_run_outputs(context, queue)
    assert coverage["problems"] == []
    assert coverage["rolling"] == []
    assert context.run.logical(f"trajectories/{name}/{campaign.FAILED_STATE}") in set(
        coverage["expected"]), "a recorded gate stop requires its diagnostic state"


def test_a_deleted_failed_state_does_not_delete_its_own_requirement(tmp_path):
    """Coverage derived from ``is_file()`` disappears exactly when the file does."""
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    directory = stopped_at_first_check(context, name)
    (directory / campaign.FAILED_STATE).unlink()
    coverage = cli.expected_run_outputs(context, queue)
    assert context.run.logical(f"trajectories/{name}/{campaign.FAILED_STATE}") in set(
        coverage["expected"]), "the recorded failure snapshot is what requires it, not the file"


def test_a_deleted_terminal_status_is_still_expected_from_its_own_record(tmp_path):
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    directory = reached_trajectory(context, name)
    coverage = cli.expected_run_outputs(context, queue)
    status = context.run.logical(f"trajectories/{name}/{campaign.STATUS_JSON}")
    assert status in set(coverage["expected"])
    # And once it is registered, deleting it is caught by the ledger rather than
    # vanishing with the entry.
    paths.record_completion(context.run.run_root, status,
                            directory / campaign.STATUS_JSON, kind="replay_trajectory_status",
                            scientific_digest="d" * 64)
    (directory / campaign.STATUS_JSON).unlink()
    problems = paths.verify_completions(
        context.run.run_root, resolve=lambda logical: cli._resolve_run_logical(context, logical),
        expected=cli.expected_run_outputs(context, queue)["expected"])["problems"]
    assert any(campaign.STATUS_JSON in problem["artifact"] for problem in problems)


def test_the_rolling_state_is_compared_to_the_latest_snapshot_not_merely_hashed(tmp_path):
    """Hashing whatever is there proves the file is itself and nothing more."""
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    directory = reached_trajectory(context, name)
    (directory / campaign.LAST_PASSING).write_bytes(b"replaced by something else")
    coverage = cli.expected_run_outputs(context, queue)
    assert coverage["rolling"][0]["matches_latest_snapshot"] is False
    assert any("most recent passing snapshot" in problem["problem"]
               for problem in coverage["problems"])


def test_a_missing_last_passing_state_after_a_passing_check_is_a_problem(tmp_path):
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    directory = reached_trajectory(context, name)
    (directory / campaign.LAST_PASSING).unlink()
    coverage = cli.expected_run_outputs(context, queue)
    assert any("is not on disk" in problem["problem"] for problem in coverage["problems"])


def test_a_begun_run_whose_stage_documents_are_gone_is_not_silently_immutable(tmp_path):
    """Fitting cannot have happened without the freeze, the preflight and the banks."""
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    reached_trajectory(context, name)
    for leaf in ("training_spec_frozen.json", "preflight.json", "banks_manifest.json"):
        context.run.path(leaf).unlink()
    problems = {problem["artifact"] for problem in
                cli.expected_run_outputs(context, queue)["problems"]}
    assert {"training_spec_frozen.json", "preflight.json", "banks_manifest.json"} <= problems


def test_nothing_is_required_of_a_run_that_has_not_begun_fitting(tmp_path):
    """During prepare there is no freeze and no banks manifest, and that is correct."""
    context = context_for(tmp_path)
    coverage = cli.expected_run_outputs(context, campaign.build_queue(context.config))
    assert coverage["problems"] == [] and coverage["expected"] == []


def test_a_deleted_reached_checkpoint_is_a_problem_rather_than_an_immutable_run(tmp_path):
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    name = queue[0]["trajectory"]
    directory = reached_trajectory(context, name)
    (directory / "endpoint_update1000.pt").unlink()
    coverage = cli.expected_run_outputs(context, queue)
    problems = paths.missing_expected_outputs(
        context.run.run_root, coverage["expected"],
        resolve=lambda logical: cli._resolve_run_logical(context, logical))
    assert any("endpoint_update1000.pt" in problem["artifact"] for problem in problems), (
        "a reached endpoint whose checkpoint is gone cannot still report immutable=true")


def test_a_trajectory_missing_its_own_journal_is_a_problem(tmp_path):
    context = context_for(tmp_path)
    queue = campaign.build_queue(context.config)
    directory = reached_trajectory(context, queue[0]["trajectory"])
    (directory / campaign.UPDATES_JSONL).unlink()
    coverage = cli.expected_run_outputs(context, queue)
    assert any("updates.jsonl" in problem["artifact"] for problem in coverage["problems"])


def test_a_bank_artifact_the_manifest_records_and_the_disk_lacks_is_a_problem(tmp_path):
    context = context_for(tmp_path)
    paths.write_json(context.run.path("banks_manifest.json"),
                     {"banks": {"seed1::replay_bank": {"file": "banks/seed1/replay/bank.npz",
                                                       "sha256": "0" * 64}}})
    coverage = cli.expected_run_outputs(context, [])
    assert any("bank.npz" in problem["artifact"] for problem in coverage["problems"])


def test_a_completion_key_resolves_under_both_conventions_the_ledger_holds(tmp_path):
    context = context_for(tmp_path)
    document = context.run.logical("preflight.json")
    assert cli._resolve_run_logical(context, document) == context.run.path("preflight.json")
    shard = "banks/seed1/replay/bank.npz"
    assert cli._resolve_run_logical(context, shard) == context.run.path(shard), (
        "a run-relative shard key must resolve to a file, not be reported as absent")


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

def test_bank_preparation_and_the_teacher_cache_are_in_the_cost_table():
    state = {"trajectories": [
        {"cost": {"seconds": {"optimizer": 10.0, "gate": 2.0, "teacher_cache": 0.0},
                  "wall_seconds": 15.0}}]}
    banks = {"timings": {"wall_seconds": 900.0,
                         "categories": {"seconds": {"teacher_cache": 600.0, "generation": 250.0,
                                                    "io": 0.0}}}}
    totals = cli.aggregate_costs(state, banks=banks)
    assert totals["optimizer"] == 10.0 and totals["wall"] == 15.0
    assert totals["banks::teacher_cache"] == 600.0, (
        "the teacher cache is what every replay arm pays before any of them runs")
    assert totals["banks::generation"] == 250.0 and totals["banks::wall"] == 900.0
    assert "banks::io" not in totals, "a category that measured nothing is not reported as a row"


def test_the_costs_of_a_campaign_with_no_banks_yet_are_just_the_trajectories():
    totals = cli.aggregate_costs({"trajectories": []}, banks={})
    assert totals == {}
