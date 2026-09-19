"""The monitor must not turn partial logs or early stops into completed budgets."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
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
