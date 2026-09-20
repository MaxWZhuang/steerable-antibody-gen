"""The monitor must not turn partial logs or early stops into completed budgets."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/her2_live_dashboard.py"
spec = importlib.util.spec_from_file_location("her2_live_dashboard", SCRIPT)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    output, control = tmp_path / "campaign", tmp_path / "control"
    arm = {"stage": 1, "arm_id": "dpo_beta0p1", "objective": "dpo",
           "coefficients": {"beta": .1}, "reused_from_stage": None}
    write(output / "plan.json", {"arms": [arm, dict(arm, stage=2, reused_from_stage=1)],
                                 "seeds": [1, 2], "budgets_gpu_seconds": [180, 360, 600]})
    write(control / "campaign_status.json", {"status": "running", "active_phase": "stage1_continue",
           "updated_at": datetime.now(timezone.utc).isoformat()})
    reader = dashboard.Campaign(output, control)
    monkeypatch.setattr(reader, "gpu_status", lambda: None)
    return reader


def test_journal_retries_partial_line_once_and_handles_truncation(tmp_path):
    path = tmp_path / "updates.jsonl"
    path.write_bytes(b'{"update":1}\n{"update":')
    journal = dashboard.Journal(path, lambda r: r)
    assert journal.read() == [{"update": 1}]
    with path.open("ab") as handle:
        handle.write(b'2}\n')
    assert journal.read() == [{"update": 1}, {"update": 2}]
    assert journal.read() == [{"update": 1}, {"update": 2}]
    path.write_bytes(b'{"update":9}\n')
    assert journal.read() == [{"update": 9}]
    path.unlink()
    assert journal.read() == []


def test_gate_stop_does_not_invent_later_endpoints_or_count_reused_controls(campaign):
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    write(directory / "trajectory.json", {"status": "stopped", "updates": 51,
          "cost": {"training_gpu_seconds": 80.1}, "budgets": {}, "stop_reason": "likelihood_drop"})
    write(directory / "parent_validation_reference.json", {"gpu_seconds": 3})
    (directory / "monitor.jsonl").write_text(json.dumps({"record_kind": "gate_verdict",
        "D": 1.2, "passed": False, "monitor_gpu_seconds": 2, "update": 51}) + "\n" +
        json.dumps({"record_kind": "snapshot", "D": 1.2, "monitor_gpu_seconds": 2}) + "\n")
    state = campaign.snapshot()
    assert state["total_runs"] == 2
    assert state["counts"] == {"stopped": 1, "queued": 1}
    assert state["runs"][0]["budgets_reached"] == []
    assert state["recorded_training_gpu_seconds"] == 80.1
    assert state["recorded_monitor_gpu_seconds"] == 5
    assert state["runs"][0]["checks"] == 1


def test_stale_supervisor_cannot_appear_live(campaign):
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    write(directory / "run.json", {"status": "running"})
    write(campaign.control / "campaign_status.json", {"status": "running",
          "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()})
    snapshot = campaign.snapshot()
    assert snapshot["status"] == "stale"
    assert snapshot["active_run"] is None
    assert snapshot["runs"][0]["status"] == "interrupted"


def test_live_update_count_and_saved_budgets_are_independent(campaign):
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    write(directory / "trajectory_progress.json", {"updates": 25, "training_gpu_seconds": 179})
    (directory / "updates.jsonl").write_bytes(
        b'{"update":26,"cumulative_gpu_seconds":181}\n{"update":27')
    snapshot = campaign.snapshot()
    run = snapshot["runs"][0]
    assert run["updates"] == 26
    assert run["training_gpu_seconds"] == 181
    assert run["budgets_reached"] == []
    assert run["status"] == "running"


def test_detail_rejects_paths_outside_declared_runs(campaign):
    with pytest.raises(KeyError):
        campaign.detail("../../private")
    assert campaign.detail("stage1/dpo_beta0p1_seed1")["updates"] == []


def test_nonfinite_metric_is_missing_not_invalid_json():
    assert dashboard.clean({"D": float("nan"), "nested": [float("inf"), 1]}) == {
        "D": None, "nested": [None, 1]}


def age(path, seconds):
    """Backdate every artifact in a run directory by ``seconds``."""
    stamp = datetime.now(timezone.utc).timestamp() - seconds
    for item in Path(path).iterdir():
        os.utime(item, (stamp, stamp))


def test_journal_discards_history_when_the_file_is_replaced_not_appended(tmp_path):
    """A replaced journal must not serve the previous run's lines as this run's.

    Size-only invalidation misses both of these: an in-place rewrite of the same
    length keeps the offset, and a longer replacement keeps the old rows AND
    resumes mid-record.
    """
    path = tmp_path / "monitor.jsonl"
    path.write_bytes(b'{"update":1}\n')
    journal = dashboard.Journal(path, lambda r: r)
    assert journal.read() == [{"update": 1}]

    path.write_bytes(b'{"update":7}\n')                    # same size, rewritten in place
    assert journal.read() == [{"update": 7}]
    assert journal.resets == 1

    path.write_bytes(b'{"update":8}\n{"update":9}\n')      # longer than the old offset
    assert journal.read() == [{"update": 8}, {"update": 9}]
    assert journal.resets == 2


def test_journal_discards_history_when_the_path_gets_a_different_file(tmp_path):
    path = tmp_path / "monitor.jsonl"
    path.write_bytes(b'{"update":1}\n{"update":2}\n')
    journal = dashboard.Journal(path, lambda r: r)
    assert journal.read() == [{"update": 1}, {"update": 2}]
    replacement = tmp_path / "other.jsonl"
    replacement.write_bytes(b'{"update":5}\n{"update":6}\n')
    os.replace(replacement, path)                          # new inode, same size
    assert journal.read() == [{"update": 5}, {"update": 6}]


def test_hung_trajectory_is_not_running_under_a_live_supervisor(campaign):
    """A fresh campaign heartbeat says the supervisor is alive, nothing more."""
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    write(directory / "trajectory_progress.json", {"updates": 25, "training_gpu_seconds": 179})
    (directory / "updates.jsonl").write_bytes(b'{"update":26,"cumulative_gpu_seconds":181}\n')
    age(directory, 3600)
    snapshot = campaign.snapshot()
    run = snapshot["runs"][0]
    assert snapshot["status"] == "running"                 # the supervisor IS current
    assert run["status"] == "stalled"                      # the trajectory is not
    assert run["idle_seconds"] > 3000
    assert snapshot["active_run"] is None
    assert any("written nothing" in w for w in snapshot["warnings"])


def test_a_gate_check_still_reads_as_running_within_the_stall_bound(campaign):
    """A check costs ~14 GPU s and writes nothing meanwhile; that is not a stall."""
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "updates.jsonl").write_bytes(b'{"update":26,"cumulative_gpu_seconds":181}\n')
    age(directory, 60)
    assert campaign.snapshot()["runs"][0]["status"] == "running"


def test_artifact_failure_is_visible_and_distinct_from_a_gate_stop(campaign):
    """A save that failed is not a likelihood breach and must not read as one."""
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    write(directory / "trajectory.json", {
        "status": "failed", "updates": 900, "cost": {"training_gpu_seconds": 240.0},
        "artifact_failure": {"type": "OSError", "message": "disk full",
                             "where": "save_last_passing", "update": 900},
        "snapshot_errors": [{"check": 12, "save_failed": "OSError: disk full"}]})
    (directory / "monitor.jsonl").write_text(
        json.dumps({"record_kind": "gate_verdict", "D": 0.4, "passed": True,
                    "monitor_gpu_seconds": 14.3, "update": 900}) + "\n" +
        json.dumps({"record_kind": "snapshot", "check": 12, "update": 900, "passed": True,
                    "D": 0.4, "save_failed": "OSError: disk full",
                    "consequence": "no rolling last-passing bytes were written"}) + "\n")
    run = campaign.snapshot()["runs"][0]
    assert run["status"] == "failed"
    assert run["failure"]["type"] == "OSError"
    assert run["artifact_errors"][0]["save_failed"] == "OSError: disk full"
    assert run["artifact_errors"][0]["consequence"]
    assert run["checks"] == 1                              # the snapshot is not a second check
    assert run["monitor_gpu_seconds"] == 14.3
    detail = campaign.detail("stage1/dpo_beta0p1_seed1")
    assert [row["record_kind"] for row in detail["gates"]] == ["gate_verdict"]
    assert [row["record_kind"] for row in detail["snapshots"]] == ["snapshot"]


def test_rolling_checkpoint_cost_is_reported(campaign):
    """Every passing check serializes and hashes the whole model. Show it."""
    directory = campaign.output / "stage1/dpo_beta0p1_seed1"
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for check in range(3):
        lines.append(json.dumps({"record_kind": "gate_verdict", "D": 0.2, "passed": True,
                                 "monitor_gpu_seconds": 14.0, "update": 25 * check}))
        lines.append(json.dumps({"record_kind": "snapshot", "check": check, "passed": True,
                                 "last_passing": {"path": "last_passing.pt", "sha256": "ab",
                                                  "update": 25 * check, "wall_seconds": 1.5}}))
    (directory / "monitor.jsonl").write_text("\n".join(lines) + "\n")
    state = campaign.snapshot()
    assert state["runs"][0]["rolling_checkpoint_saves"] == 3
    assert state["runs"][0]["rolling_checkpoint_wall_seconds"] == 4.5
    assert state["recorded_checkpoint_saves"] == 3
    assert state["recorded_checkpoint_wall_seconds"] == 4.5
