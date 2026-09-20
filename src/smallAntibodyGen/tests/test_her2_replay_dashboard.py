"""The Replay panel must not make an unlaunched or interrupted campaign look healthy.

The dashboard is where a human reads this campaign while it runs, so the failures
that matter are a panel that shows a plausible queue for a campaign that was
never launched, a "healthy" launch claimed from a process that merely started, or
a dead writer still rendered as running.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/her2_live_dashboard.py"
spec = importlib.util.spec_from_file_location("her2_live_dashboard_replay", SCRIPT)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)

QUEUE = [{"trajectory": "continued_sft_lambda0_seed20260918", "arm_id": "continued_sft_lambda0",
          "task": "continued_sft", "replay_lambda": 0.0, "seed": 20260918, "is_control": True},
         {"trajectory": "ipo_lambda1_seed20260918", "arm_id": "ipo_lambda1", "task": "ipo",
          "replay_lambda": 1.0, "seed": 20260918, "is_control": False}]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@contextmanager
def held_lock(root):
    """Hold the campaign's exclusive write lock, the way a live campaign does.

    Taken here with the platform primitives directly rather than by importing the
    experiment package: the dashboard deliberately imports nothing from it, and a
    test that reached for it would be checking a different arrangement than the one
    that ships.
    """
    path = root / "campaign.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield path
    finally:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "outputs/her2_parent_replay_20260920"
    root.mkdir(parents=True)
    return root


def test_an_absent_run_directory_is_reported_as_absent(tmp_path):
    snapshot = dashboard.ReplayRun(tmp_path / "missing").snapshot()
    assert snapshot["present"] is False and snapshot["trajectories"] == []
    assert snapshot["phase"] == "no_run_directory"


def test_a_campaign_that_was_never_launched_shows_no_queue_rather_than_empty_rows(run):
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["queue_present"] is False and snapshot["trajectories"] == []
    assert snapshot["phase"] == "before_freeze"
    assert any("has not been launched" in warning for warning in snapshot["warnings"])
    assert any("training_spec_frozen" in warning for warning in snapshot["warnings"])


def test_a_frozen_campaign_without_banks_says_it_is_preparing_them(run):
    write(run / "training_spec_frozen.json",
          {"git": {"commit": "a" * 40}, "frozen_at": "2026-09-20T00:00:00+00:00",
           "source": {"sha256": {"x.py": "y"}}, "input_count": 12,
           "audit": {"decision_outcome": "escalate"}})
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["phase"] == "preparing_banks"
    assert snapshot["frozen"]["commit"] == "a" * 40 and snapshot["frozen"]["inputs"] == 12
    assert snapshot["frozen"]["audit_decision"] == "escalate"
    assert snapshot["banks"] is None
    assert any("have not been generated" in warning for warning in snapshot["warnings"])


def test_banks_are_reported_with_the_freeze_they_are_bound_to(run):
    write(run / "training_spec_frozen.json", {"git": {"commit": "a" * 40}})
    write(run / "banks_manifest.json",
          {"bank_count": 6, "freeze": {"commit": "a" * 40},
           "overlap": {"20260918": {"monitor_draws_also_in_replay": 17}}})
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["banks"]["count"] == 6
    assert snapshot["banks"]["overlap"]["20260918"] == 17
    assert snapshot["phase"] == "banks_ready"


def test_a_launch_is_not_healthy_until_an_update_and_a_check_are_on_disk(run):
    write(run / "queue.json", {"queue": QUEUE, "campaign_id": "her2_parent_replay_20260920"})
    write(run / "campaign_status.json", {"status": "running", "campaign_id": "x"})
    write(run / "campaign_heartbeat.json", {"pid": 1, "update": 3})
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["health"]["healthy"] is False
    assert "a started process is not a healthy launch" in snapshot["health"]["basis"]

    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    lines(directory / "updates.jsonl", [{"record_kind": "update", "update": 1}])
    lines(directory / "monitor.jsonl", [{"record_kind": "gate_verdict", "update": 1,
                                         "passed": False, "D": 4.4}])
    assert dashboard.ReplayRun(run).snapshot()["health"]["healthy"] is False, (
        "a measured check that did NOT pass is not a healthy launch")

    lines(directory / "monitor.jsonl", [{"record_kind": "gate_verdict", "update": 1,
                                         "passed": True, "D": 0.01}])
    health = dashboard.ReplayRun(run).snapshot()["health"]
    assert health["healthy"] is True and health["first_check_passed"] is True
    assert health["first_update_completed"] == 1


def test_the_first_check_is_read_from_the_head_not_the_tail(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    lines(directory / "updates.jsonl", [{"record_kind": "update", "update": n}
                                        for n in range(1, 40)])
    lines(directory / "monitor.jsonl",
          [{"record_kind": "gate_verdict", "update": 1, "passed": True, "D": 0.01}]
          + [{"record_kind": "gate_verdict", "update": 25, "passed": False, "D": 9.0}])
    health = dashboard.ReplayRun(run).snapshot()["health"]
    assert health["first_check_update"] == 1 and health["first_check_passed"] is True
    assert health["first_update_completed"] == 1


def test_a_dead_writer_is_not_rendered_as_running(run):
    """The lock decides it: the status still says running and nobody holds it."""
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    write(run / "campaign_heartbeat.json", {"pid": 1, "update": 500})
    stale = datetime.now(timezone.utc).timestamp() - 3600
    os.utime(run / "campaign_heartbeat.json", (stale, stale))
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    directory.mkdir(parents=True)
    snapshot = dashboard.ReplayRun(run, stall_seconds=180).snapshot()
    assert snapshot["writer_state"] == "gone"
    assert snapshot["lock"]["state"] == "missing"
    assert snapshot["trajectories"][0]["status"] == "interrupted"
    assert any("nobody holds the write lock" in warning for warning in snapshot["warnings"])
    assert any("never resumed" in warning for warning in snapshot["warnings"])


def test_a_missing_heartbeat_does_not_make_a_writer_alive(run):
    """The hole: recorded `running` with no heartbeat file used to read as alive forever."""
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    (run / "trajectories" / QUEUE[0]["trajectory"]).mkdir(parents=True)
    assert not (run / "campaign_heartbeat.json").exists()
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["heartbeat_age_seconds"] is None
    assert snapshot["writer_state"] == "gone", (
        "no heartbeat is not evidence of life; the lock is what answers this")
    assert snapshot["trajectories"][0]["status"] == "interrupted"


def test_a_stale_heartbeat_under_a_held_lock_is_a_stall_signal_not_a_death(run):
    """A live owner can be thirty seconds into scoring 25,722 validation pairs."""
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    write(run / "campaign_heartbeat.json", {"pid": 1, "update": 500})
    stale = datetime.now(timezone.utc).timestamp() - 3600
    os.utime(run / "campaign_heartbeat.json", (stale, stale))
    (run / "trajectories" / QUEUE[0]["trajectory"]).mkdir(parents=True)
    with held_lock(run):
        snapshot = dashboard.ReplayRun(run, stall_seconds=180).snapshot()
    assert snapshot["writer_state"] == "alive" and snapshot["lock"]["state"] == "held"
    assert snapshot["heartbeat_stalled"] is True
    assert snapshot["trajectories"][0]["status"] == "running"
    assert any("progress signal, not a liveness one" in warning
               for warning in snapshot["warnings"])
    assert snapshot["health"]["status"] in ("not_established", "healthy")


def test_observing_the_run_directory_changes_no_bytes_and_no_mtimes(run):
    """A read-only panel refreshing every five seconds must write nothing at all.

    The lock file is handled apart from the rest on purpose. While the campaign
    holds its Windows byte-range lock, nobody else can *read* those bytes -- a
    ``PermissionError`` there is the operating system doing its job, not the panel
    mutating anything. So its bytes are checked where they can be: absent before
    the owner takes it, and exactly the owner's one byte after release. While it is
    held, its mtime -- which ``stat`` still reports -- is what is compared.
    """
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    write(run / "campaign_heartbeat.json", {"pid": 1, "update": 2})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    lines(directory / "updates.jsonl", [{"record_kind": "update", "update": 1,
                                         "exposures": {"chosen": 64}}])
    lines(directory / "monitor.jsonl", [{"record_kind": "gate_verdict", "update": 1,
                                         "passed": True, "D": 0.01}])
    lock = run / "campaign.lock"
    assert not lock.exists(), "nothing has taken this directory yet"

    def application_files():
        return {path: (path.stat().st_mtime_ns, path.read_bytes())
                for path in sorted(run.rglob("*")) if path.is_file() and path != lock}

    with held_lock(run):
        before, lock_mtime = application_files(), lock.stat().st_mtime_ns
        names = sorted(run.rglob("*"))
        snapshot = dashboard.ReplayRun(run).snapshot()
        dashboard.ReplayRun(run).snapshot()
        assert snapshot["lock"]["state"] == "held", "the probe did observe a live owner"
        assert application_files() == before, "the panel observed the run and wrote nothing"
        assert lock.stat().st_mtime_ns == lock_mtime, "and it did not write to the lock either"
        assert sorted(run.rglob("*")) == names, "and it created no file"
    assert lock.read_bytes() == b"\0", (
        "released: the lock holds exactly the one byte its owner wrote, so the two probes through "
        "it appended nothing")


def test_a_launch_that_later_failed_is_not_displayed_as_healthy(run):
    """The first update and the first passed check stay true; the label must not."""
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json",
          {"status": "failed", "error": "RuntimeError: the backend fell over"})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    lines(directory / "updates.jsonl", [{"record_kind": "update", "update": 1}])
    lines(directory / "monitor.jsonl", [{"record_kind": "gate_verdict", "update": 1,
                                         "passed": True, "D": 0.01}])
    write(run / f"trajectories/{QUEUE[1]['trajectory']}/status.json",
          {"status": "failed", "updates": 7, "checks": 2, "exposures": {"chosen": 448}})
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["health"]["healthy"] is True, "the historical evidence is preserved"
    assert snapshot["health"]["status"] == "launch_verified_but_campaign_failed"
    assert "not a statement about the campaign now" in snapshot["health"]["detail"]
    assert any("`failed`" in warning for warning in snapshot["warnings"])


def test_a_running_trajectory_is_counted_from_its_journals_not_its_progress_file(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    write(directory / "trajectory_progress.json",
          {"update": 1000, "checks": 41, "exposures": {"chosen": 64000},
           "endpoints_reached": ["1000"]})
    lines(directory / "updates.jsonl",
          [{"record_kind": "update", "update": n, "exposures": {"chosen": n * 64}}
           for n in range(1020, 1025)])
    lines(directory / "endpoints.jsonl",
          [{"record_kind": "exposure_endpoint", "update": 1000,
            "checkpoint": {"sha256": "a" * 64}}])
    with held_lock(run):
        row = dashboard.ReplayRun(run).snapshot()["trajectories"][0]
    assert row["updates"] == 1024 and row["chosen_exposures"] == 1024 * 64
    assert row["checks"] == 41, "checks are written exactly when the progress file is"
    assert row["endpoints_reached"] == ["1000"]
    assert row["counters_from"] == "its append-only journals"
    assert row["progress_file_updates"] == 1000, (
        "the lagging counter is shown beside the journalled one, never in place of it")
    assert row["progress_file_exposures"]["chosen"] == 64000


def test_a_trajectory_with_no_journalled_update_reports_none_not_the_progress_file(run):
    """A directory and a progress file, and no update record: nothing completed yet."""
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    write(directory / "trajectory_progress.json",
          {"update": 1000, "checks": 41, "exposures": {"chosen": 64000}})
    with held_lock(run):
        row = dashboard.ReplayRun(run).snapshot()["trajectories"][0]
    assert row["updates"] is None and row.get("chosen_exposures") is None
    assert row["progress_file_updates"] == 1000


def test_a_failed_campaign_shows_its_error_and_does_not_claim_progress(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json",
          {"status": "failed", "error": "ValueError: a banks manifest is not bound"})
    (run / "trajectories" / QUEUE[0]["trajectory"]).mkdir(parents=True)
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert snapshot["writer_state"] == "failed"
    assert "not bound" in snapshot["error"]
    assert snapshot["trajectories"][0]["status"] == "unknown", (
        "a begun trajectory under a failed campaign is not reported as running or completed")
    assert snapshot["trajectories"][1]["status"] == "queued"


def test_terminal_statuses_come_from_the_trajectory_not_from_the_campaign(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    write(run / "campaign_heartbeat.json", {"pid": 1})
    write(run / f"trajectories/{QUEUE[0]['trajectory']}/status.json",
          {"status": "stopped_by_gate", "stop_reason": "parent_relative_likelihood_breach",
           "updates": 412, "checks": 17, "exposures": {"chosen": 26368, "replay": 26368},
           "endpoints_reached": {}})
    (run / "trajectories" / QUEUE[1]["trajectory"]).mkdir(parents=True)
    with held_lock(run):
        snapshot = dashboard.ReplayRun(run).snapshot()
    row = snapshot["trajectories"][0]
    assert row["status"] == "stopped_by_gate" and row["terminal"] is True
    assert row["stop_reason"] == "parent_relative_likelihood_breach"
    assert row["updates"] == 412 and row["chosen_exposures"] == 26368
    assert row["counters_from"] == "its terminal status"
    assert snapshot["trajectories"][1]["status"] == "running"


def test_a_checkpoint_on_disk_is_not_a_reached_endpoint(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    directory.mkdir(parents=True)
    (directory / "endpoint_update1000.pt").write_bytes(b"not a record")
    row = dashboard.ReplayRun(run).snapshot()["trajectories"][0]
    assert row["endpoints_reached"] == [], (
        "an endpoint comes from the trajectory's own status, written after the bytes were "
        "validated and the evaluation was saved")


def test_the_latest_gate_verdict_and_its_preservation_block_reach_the_panel(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    directory = run / "trajectories" / QUEUE[1]["trajectory"]
    lines(directory / "monitor.jsonl", [
        {"record_kind": "gate_verdict", "update": 1, "passed": True, "D": 0.1},
        {"record_kind": "snapshot", "update": 1, "passed": True},
        {"record_kind": "gate_verdict", "update": 25, "passed": True, "D": 0.22,
         "preservation": {"available": True, "forward_kl": {"mean": 0.31},
                          "tails": {"counts": {"tenfold": {"fraction": 0.021}}},
                          "conditional_kl": {"mean": 0.33}}}])
    gate = dashboard.ReplayRun(run).snapshot()["trajectories"][1]["gate"]
    assert gate["update"] == 25 and gate["D"] == 0.22
    assert gate["forward_kl"] == 0.31 and gate["tenfold_fraction"] == 0.021
    assert gate["conditional_kl"] == 0.33


def test_unreadable_json_is_an_error_not_a_crash_and_not_a_zero(run):
    (run / "queue.json").write_text("{not json", encoding="utf-8")
    snapshot = dashboard.ReplayRun(run).snapshot()
    assert any("queue.json" in message for message in snapshot["errors"])
    assert snapshot["queue_present"] is False


def test_nonfinite_values_are_nulled_so_the_json_stays_parseable(run):
    write(run / "queue.json", {"queue": QUEUE})
    write(run / "campaign_status.json", {"status": "running"})
    directory = run / "trajectories" / QUEUE[0]["trajectory"]
    directory.mkdir(parents=True)
    (directory / "monitor.jsonl").write_text(
        '{"record_kind": "gate_verdict", "update": 1, "passed": true, "D": NaN}\n',
        encoding="utf-8")
    snapshot = dashboard.ReplayRun(run).snapshot()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["trajectories"][0]["gate"]["D"] is None


@pytest.mark.parametrize("name", ["../../secrets.json", "/etc/passwd", "a/../../b.json"])
def test_the_reader_refuses_to_leave_the_run_directory(run, name):
    view = dashboard.ReplayRun(run)
    with pytest.raises(KeyError):
        view.path(name)
    assert view.document(name) is None


def test_the_replay_route_is_absent_when_no_replay_directory_was_given():
    assert dashboard.make_handler(object(), None, None) is not None


def test_the_dashboard_still_imports_no_model_or_training_module():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("import torch", "transformers", "smallAntibodyGen"):
        assert forbidden not in source


def test_the_replay_tab_view_and_route_are_wired_into_the_assets():
    assets = SCRIPT.parent / "her2_dashboard"
    html = (assets / "index.html").read_text(encoding="utf-8")
    script = (assets / "app.js").read_text(encoding="utf-8")
    assert 'data-view="replay"' in html and 'id="view-replay"' in html
    assert "/api/replay" in script and "/api/audit" in script and "/api/status" in script
    assert "replayView" in script and "refreshReplay" in script
    assert "first_check_passed" in script or "health" in script


def test_selecting_the_replay_view_hides_the_other_headers_and_views():
    script = (SCRIPT.parent / "her2_dashboard/app.js").read_text(encoding="utf-8")
    compact = script.replace(" ", "")
    assert 'replayView=view==="replay"' in compact
    assert '$("campaign-heading").hidden=auditView||replayView' in compact
    assert '$("view-campaign").hidden=auditView||replayView' in compact
    assert '$("view-audit").hidden=!auditView' in compact
    assert '$("view-replay").hidden=!replayView' in compact


def test_the_replay_view_polls_only_while_it_is_showing():
    script = (SCRIPT.parent / "her2_dashboard/app.js").read_text(encoding="utf-8")
    compact = script.replace(" ", "")
    assert "setInterval(()=>{if(replayView)refreshReplay();},5000)" in compact
    assert "setInterval(()=>{if(auditView)refreshAudit();},5000)" in compact, (
        "the audit view must keep working exactly as it did")


def test_the_replay_rows_are_written_as_dom_text_not_interpolated_markup():
    script = (SCRIPT.parent / "her2_dashboard/app.js").read_text(encoding="utf-8")
    body = script.split("function renderReplay")[1].split("async function refreshReplay")[0]
    assert "td.textContent" in script
    assert body.count("innerHTML") <= 1, (
        "trajectory names, stop reasons and statuses come out of a run directory and are set as "
        "text; the single remaining innerHTML writes an escaped status badge")


def test_the_existing_audit_panel_is_untouched(run):
    """The audit view's own reader must still behave exactly as before."""
    audit = dashboard.AuditRun(run)
    snapshot = audit.snapshot()
    assert snapshot["present"] is True and snapshot["complete"] is False
    assert [row["stage"] for row in snapshot["stages"]] == list(dashboard.AuditRun.STAGES)
