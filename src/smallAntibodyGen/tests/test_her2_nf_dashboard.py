"""Monitor the next flight without inventing progress or charging work twice."""
from contextlib import contextmanager
from datetime import datetime, timezone
from http.client import HTTPConnection
import importlib.util
import json
import os
from pathlib import Path
import threading

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/her2_live_dashboard.py"
spec = importlib.util.spec_from_file_location("her2_live_dashboard_nf", SCRIPT)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


@contextmanager
def held_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("w+b") as handle:
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_absent_run_is_not_created(tmp_path):
    root = tmp_path / "absent"
    result = dashboard.NextFlightRun(root).snapshot()
    assert result["state"] == "no_run_directory"
    assert result["complete"] is False
    assert not root.exists()


def test_live_empty_pilot_is_preparing_and_missing_queue_is_normal(tmp_path):
    (tmp_path / "calibration/pilot").mkdir(parents=True)
    (tmp_path / "calibration/budgets").mkdir()
    write(tmp_path / "heartbeat.json", {"stage": "calibrate"})
    write(tmp_path / "stages/recover.json", {"status": "completed"})
    with held_lock(tmp_path):
        result = dashboard.NextFlightRun(tmp_path).snapshot()
        assert result["state"] == "alive"
        assert result["calibration_jobs"][0]["activity"] == "preparing"
        assert len(result["calibration_jobs"]) == 1
        assert result["active_job"] is None
        assert result["production"]["queue_document_present"] is False
        assert result["stages_completed"] == 1
        assert result["stages_total"] == 12
        assert result["stages"][1]["status"] == "not_recorded"
        assert result["complete"] is False
    assert dashboard.NextFlightRun(tmp_path).snapshot()["state"] == "gone"


def test_heartbeat_does_not_turn_quiet_journals_into_progress(tmp_path):
    updates = tmp_path / "calibration/pilot/updates.jsonl"
    lines(updates, [{"update": 25}])
    stale = datetime.now(timezone.utc).timestamp() - 3600
    os.utime(updates, (stale, stale))
    write(tmp_path / "heartbeat.json", {"stage": "calibrate", "liveness_tick": 1000})
    with held_lock(tmp_path):
        result = dashboard.NextFlightRun(tmp_path).snapshot()
    assert result["state"] == "alive"
    assert result["active_job"] is None
    assert result["calibration_jobs"][0]["activity"] == "no_recent_writes"
    assert result["warnings"]


@pytest.mark.parametrize("status,expected", [
    ("running", "gone"), ("completed", "finished"), ("stopped", "finished"),
    ("incomplete", "finished"), ("failed", "failed"),
])
def test_terminal_session_is_distinct_from_verified_completion(tmp_path, status, expected):
    with held_lock(tmp_path):
        pass
    write(tmp_path / "campaign_status.json", {"status": status})
    result = dashboard.NextFlightRun(tmp_path).snapshot()
    assert result["state"] == expected
    assert result["complete"] is False


def test_inaccessible_lock_does_not_claim_writer_is_gone(tmp_path, monkeypatch):
    reader = dashboard.NextFlightRun(tmp_path)
    monkeypatch.setattr(reader, "lock_observation", lambda: {
        "state": "inaccessible", "reason": "access denied"})
    assert reader.snapshot()["state"] == "unknown"


@pytest.mark.parametrize("report,verified,stage,expected", [
    (True, True, "completed", True), (True, False, "completed", False),
    (True, True, "failed", False), (False, True, "completed", False),
    (None, True, "completed", False),
])
def test_completion_requires_all_three_records(tmp_path, report, verified, stage, expected):
    write(tmp_path / "report.json", {"completion": {"complete": report}})
    write(tmp_path / "verification.json", {"passed": verified})
    write(tmp_path / "stages/verify.json", {"status": stage})
    assert dashboard.NextFlightRun(tmp_path).snapshot()["complete"] is expected


def test_cost_matches_real_ledger_identities_and_counts_extension_once(tmp_path):
    original = {"entry_id": "L1:pilot", "coefficients": {"lambda": 10}, "seed": 1}
    extension = dict(original, entry_id="EXT:L1:pilot")
    overhead = {"checkpoint": "parent.pt", "seed": 1, "source": "frozen"}
    for name, identity, seconds, debit in (
        ("L1_pilot", original, 100, 2), ("EXT_L1_pilot", extension, 30, 4),
        ("parent_baseline", overhead, 20, 1),
    ):
        write(tmp_path / f"calibration/budgets/{name}.json", {
            "identity": identity, "measured_seconds": seconds,
            "uncertainty_debit_seconds": debit})
    ledger = {"entries": [{"entry_id": "L1:pilot", "cost": {
        "gpu_work_units": {"identity": original}}}],
        "overhead": {"identity": overhead}, "measured_gpu_seconds": 120,
        "uncertainty_debit_seconds": 3, "gpu_hour_cap": 1}
    write(tmp_path / "calibration_ledger.json", ledger)
    reader = dashboard.NextFlightRun(tmp_path)
    result = reader.snapshot()["calibration"]
    assert result["budgets_already_in_ledger"] == 2
    assert result["live_budget_records"] == 1
    assert result["combined_measured_seconds"] == 150
    assert result["combined_uncertainty_debit_seconds"] == 7
    assert result["remaining_seconds_under_cap"] == 3443
    ledger["entries"].append({"entry_id": "EXT:L1:pilot", "cost": {
        "gpu_work_units": {"identity": extension}}})
    ledger.update(measured_gpu_seconds=150, uncertainty_debit_seconds=7)
    write(tmp_path / "calibration_ledger.json", ledger)
    result = reader.snapshot()["calibration"]
    assert result["combined_measured_seconds"] == 150
    assert result["live_budget_records"] == 0


def test_partial_updates_sentinels_and_snapshots_are_separate(tmp_path):
    base = tmp_path / "calibration/pilot"
    lines(base / "updates.jsonl", [{"update": 1, "weighted_total": 2.5,
                                   "exposures": {"chosen": 32}}])
    with (base / "updates.jsonl").open("ab") as handle:
        handle.write(b'{"update":2,"weighted_total":')
    gate = {"record_kind": "gate_verdict", "kind": "full", "update": 1,
            "D": 0.4, "passed": True, "threshold_nats_per_sequence": 1.25}
    lines(base / "monitor.jsonl", [gate, dict(gate, record_kind="snapshot"),
                                  {"record_kind": "sentinel_check", "update": 2, "D": 0.7}])
    lines(base / "endpoints.jsonl", [{"record_kind": "exposure_endpoint", "update": 1}])
    reader = dashboard.NextFlightRun(tmp_path)
    result = reader.snapshot("calibration/pilot")["detail"]
    assert result["updates"] == 1
    assert result["latest_update"]["weighted_total"] == 2.5
    assert result["exposures"] == {"chosen": 32}
    assert result["endpoints_reached"] == ["1"]
    assert len(result["gates"]) == len(result["sentinels"]) == 1
    assert result["other_monitor_records"] == {"snapshot": 1}
    assert result["threshold_nats_per_sequence"] == 1.25
    assert result["pending_update_bytes"] > 0
    with (base / "updates.jsonl").open("ab") as handle:
        handle.write(b'2.1}\nnot json\n')
    result = reader.snapshot("calibration/pilot")["detail"]
    assert result["updates"] == 2
    assert result["update_count"] == 2
    assert result["unreadable_update_lines"] == 1
    assert result["pending_update_bytes"] == 0
    lines(base / "updates.jsonl", [{"update": 9}])
    result = reader.snapshot("calibration/pilot")["detail"]
    assert [row["update"] for row in result["updates_trace"]] == [9]
    assert result["journal_resets"] >= 1


def test_continuation_uses_live_counters_and_budget_coefficients(tmp_path):
    base = tmp_path / "calibration/L1_pilot"
    write(base / "status_u500.json", {"status": "completed", "updates": 500})
    write(base / "trajectory_progress.json", {"update": 500})
    lines(base / "updates.jsonl", [{"update": 525, "weighted_total": 0.2}])
    write(tmp_path / "calibration/budgets/EXT_L1_pilot.json", {
        "identity": {"entry_id": "EXT:L1:pilot", "coefficients": {"lambda": 10}}})
    with held_lock(tmp_path):
        result = dashboard.NextFlightRun(tmp_path).snapshot("calibration/L1_pilot")
    pilot = result["detail"]
    assert pilot["status"] == "continuation_in_progress"
    assert pilot["updates"] == 525
    assert pilot["progress_file_update"] == 500
    assert pilot["coefficients"] == {"lambda": 10}
    assert result["active_job"] == "calibration/L1_pilot"
    assert result["calibration_jobs"][0]["coefficients"] == {"lambda": 10}
    assert pilot["qualified"] is None
    assert result["coefficient_freeze"]["present"] is False


def test_frozen_outcome_counts_only_qualified_families(tmp_path):
    write(tmp_path / "calibration_outcome.json", {"frozen": {
        "dpo_fkl": {"entry_id": "EXT:L1:pilot", "coefficients": {"lambda": 10}},
        "ipo_tail": {"entry_id": None, "coefficients": None},
        "dpo_beta": 0.1, "block_b_dpo": {"family": "dpo_fkl"},
        "dpo_beta_basis": "ladder"}})
    result = dashboard.NextFlightRun(tmp_path).snapshot()["coefficient_freeze"]
    assert result["present"] is True
    assert result["families"] == ["dpo_fkl"]


def test_parent_observation_reads_metadata_without_opening_weights(tmp_path, monkeypatch):
    base = tmp_path / "parents/seed1"
    write(base / "stage1_history.json", [{"epoch": 1, "loss": 0.5}])
    checkpoint = base / "stage1_state.pt"
    checkpoint.write_bytes(b"test placeholder, not model weights")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.suffix != ".pt", "The dashboard must never deserialize checkpoints"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with held_lock(tmp_path):
        result = dashboard.NextFlightRun(tmp_path).snapshot("parents/seed1")
    assert result["active_job"] == "parents/seed1"
    assert result["detail"]["epochs_recorded"] == 1
    assert result["detail"]["checkpoint_files"] == ["stage1_state.pt"]
    assert result["detail"]["updates_trace"] == []
    assert result["parents"][0]["idle_seconds"] is not None


def test_selected_trace_is_bounded_and_keeps_first_and_last_updates(tmp_path):
    lines(tmp_path / "trajectories/pilot/updates.jsonl", [
        {"update": step, "weighted_total": 0.5} for step in range(1, 5001)])
    result = dashboard.NextFlightRun(tmp_path).snapshot("trajectories/pilot")["detail"]
    assert result["update_count"] == 5000
    assert len(result["updates_trace"]) <= 1301
    assert result["updates_trace"][0]["update"] == 1
    assert result["updates_trace"][-1]["update"] == 5000


def test_http_api_is_additive_read_only_and_rejects_foreign_paths(tmp_path):
    lines(tmp_path / "calibration/pilot/updates.jsonl", [{"update": 1}])
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}

    class Stub:
        def snapshot(self):
            return {"existing_view": True}

    handler = dashboard.make_handler(Stub(), Stub(), Stub(), dashboard.NextFlightRun(tmp_path))
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, method="GET", headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), response.getheaders()
        finally:
            connection.close()

    try:
        status, body, headers = request("/api/next-flight?select=calibration%2Fpilot")
        assert status == 200
        assert json.loads(body)["detail"]["updates"] == 1
        assert dict(headers)["Cache-Control"] == "no-store"
        for endpoint in ("/api/status", "/api/audit", "/api/replay"):
            assert json.loads(request(endpoint)[1]) == {"existing_view": True}
        for selected in ("..%2Fprivate", "calibration%2F..%2Fprivate", "calibration%2Fabsent"):
            assert request("/api/next-flight?select=" + selected)[0] == 404
        assert request("/api/next-flight", headers={"Origin": "https://foreign.example"})[0] == 403
        assert request("/api/next-flight", headers={"Host": "foreign.example"})[0] == 403
        assert request("/api/next-flight", method="POST")[0] == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    after = {path.relative_to(tmp_path): path.read_bytes()
             for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
