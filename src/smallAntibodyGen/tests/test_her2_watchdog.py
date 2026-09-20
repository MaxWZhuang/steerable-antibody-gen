"""The watchdog must page for faults, digest declared outcomes, and never lose one."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/her2_watchdog.py"
spec = importlib.util.spec_from_file_location("her2_watchdog", SCRIPT)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class Recorder:
    """A transport that records, and can be told to fail."""

    def __init__(self, working=True):
        self.working, self.sent, self.attempts = working, [], 0

    def send(self, event):
        self.attempts += 1
        if not self.working:
            return False, "synthetic transport failure"
        self.sent.append(event)
        return True, None


def snapshot(**overrides):
    base = {"status": "running", "recorded_status": "running", "heartbeat_age_seconds": 2,
            "counts": {}, "total_runs": 2, "runs": [], "stages": [], "error": None}
    base.update(overrides)
    return base


def run(run_id="stage1/dpo_beta0p1_seed1", status="running", **overrides):
    base = {"id": run_id, "status": status, "updates": 51, "training_gpu_seconds": 76.0,
            "idle_seconds": 4.0, "gate": None, "failure": None, "artifact_errors": [],
            "stop_reason": None, "budgets_reached": []}
    base.update(overrides)
    return base


@pytest.fixture
def dog(tmp_path):
    recorder = Recorder()
    made = watchdog.Watchdog(tmp_path, transports=[recorder], digest_seconds=1e9)
    made.recorder = recorder
    return made


# ---------------------------------------------------------------------------
# what fires, and what only gets collected
# ---------------------------------------------------------------------------

def test_a_stalled_trajectory_pages_and_a_gate_stop_does_not(dog):
    """A gate stop is the protocol working. Paging thirty times for it trains you to ignore it."""
    events = dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900),
                                     run("b", "stopped", stop_reason="likelihood_breach")]))
    severity = {e["key"]: e["severity"] for e in events}
    assert severity["stalled:a"] == watchdog.URGENT
    assert severity["stopped:b"] == watchdog.DIGEST
    assert [e["key"] for e in dog.recorder.sent] == ["stalled:a"]
    assert [e["key"] for e in dog.state["digest"]] == ["stopped:b"]


def test_an_artifact_failure_pages_even_though_its_gate_check_passed(dog):
    """A save that failed at a passing check is invisible in the trajectory's status."""
    events = dog.tick(snapshot(runs=[run("a", "running", artifact_errors=[
        {"check": 12, "save_failed": "OSError: disk full",
         "consequence": "no rolling last-passing bytes were written"}])]))
    fired = [e for e in events if e["severity"] == watchdog.URGENT]
    assert [e["key"] for e in fired] == ["artifact:a:12"]
    assert "disk full" in fired[0]["body"]


def test_a_failed_trajectory_is_named_as_not_a_gate_stop(dog):
    events = dog.tick(snapshot(runs=[run("a", "failed", failure={
        "type": "OSError", "message": "no space left on device"})]))
    body = next(e["body"] for e in events if e["key"] == "failed:a")
    assert "OSError" in body and "not a gate stop" in body


def test_the_status_file_vanishing_only_alarms_once_the_campaign_has_been_seen_running(dog):
    """Starting the watchdog before the campaign must be quiet."""
    assert dog.tick(snapshot(status="unknown", recorded_status=None)) == []
    dog.tick(snapshot())                                        # now it has been seen running
    events = dog.tick(snapshot(status="unknown", recorded_status=None))
    assert [e["key"] for e in events] == ["campaign_status_missing"]


# ---------------------------------------------------------------------------
# edge triggering
# ---------------------------------------------------------------------------

def test_a_standing_condition_fires_once_not_once_per_poll(dog):
    state = snapshot(runs=[run("a", "stalled", idle_seconds=900)])
    assert len(dog.tick(state)) == 1
    assert dog.tick(state) == []
    assert dog.tick(state) == []
    assert len(dog.recorder.sent) == 1


def test_a_recovered_stall_clears_and_can_alarm_again(dog):
    """A second stall is a second thing worth being told about."""
    stalled = snapshot(runs=[run("a", "stalled", idle_seconds=900)])
    healthy = snapshot(runs=[run("a", "running")])
    dog.tick(stalled)
    resolved = dog.tick(healthy)
    assert [(e["key"], e["transition"]) for e in resolved] == [("stalled:a", "resolved")]
    assert resolved[0]["severity"] == watchdog.URGENT       # stand-down matches the alarm
    again = dog.tick(stalled)
    assert [(e["key"], e["transition"]) for e in again] == [("stalled:a", "raised")]


def test_a_terminal_condition_never_re_arms(dog):
    """A failed trajectory does not un-fail, so it must not re-alert for the rest of the run."""
    dog.tick(snapshot(runs=[run("a", "failed")]))
    assert dog.tick(snapshot(runs=[])) == []
    assert dog.tick(snapshot(runs=[run("a", "failed")])) == []


# ---------------------------------------------------------------------------
# durable delivery
# ---------------------------------------------------------------------------

def test_a_failing_transport_keeps_the_event_pending_and_retries(tmp_path):
    recorder = Recorder(working=False)
    dog = watchdog.Watchdog(tmp_path, transports=[recorder], digest_seconds=1e9)
    dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    assert [e["key"] for e in dog.state["pending"]] == ["stalled:a"]
    assert dog.state["pending"][0]["attempts"] == 1
    dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    assert dog.state["pending"][0]["attempts"] == 2         # retried, not re-raised
    recorder.working = True
    dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    assert dog.state["pending"] == []
    assert [e["key"] for e in recorder.sent] == ["stalled:a"]


def test_the_journal_is_written_even_when_every_transport_is_down(tmp_path):
    """The local record of a condition must not depend on a network."""
    dog = watchdog.Watchdog(tmp_path, transports=[Recorder(working=False)], digest_seconds=1e9)
    dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    lines = [json.loads(l) for l in (tmp_path / "watchdog.jsonl").read_text().splitlines()]
    assert [l["key"] for l in lines] == ["stalled:a"]


def test_an_unreachable_transport_is_abandoned_rather_than_queued_forever(tmp_path):
    dog = watchdog.Watchdog(tmp_path, transports=[Recorder(working=False)],
                            digest_seconds=1e9, max_attempts=3)
    for _ in range(4):
        dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    assert dog.state["pending"] == []
    assert [e["key"] for e in dog.state["abandoned"]] == ["stalled:a"]
    kinds = [json.loads(l)["transition"]
             for l in (tmp_path / "watchdog.jsonl").read_text().splitlines()]
    assert kinds == ["raised", "abandoned"]                 # kept in the journal either way


def test_state_survives_a_restart_without_re_alerting_or_losing_the_outbox(tmp_path):
    first = watchdog.Watchdog(tmp_path, transports=[Recorder(working=False)], digest_seconds=1e9)
    first.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900),
                              run("b", "stopped")]))
    recorder = Recorder()
    second = watchdog.Watchdog(tmp_path, transports=[recorder], digest_seconds=1e9)
    assert "stalled:a" in second.state["active"]            # not re-raised
    assert [e["key"] for e in second.state["pending"]] == ["stalled:a"]
    assert [e["key"] for e in second.state["digest"]] == ["stopped:b"]
    events = second.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900),
                                        run("b", "stopped")]))
    assert events == []                                     # nothing new
    assert [e["key"] for e in recorder.sent] == ["stalled:a"]   # the outbox drained


def test_a_corrupt_state_file_restarts_clean_instead_of_crashing(tmp_path):
    (tmp_path / "watchdog_state.json").write_text("{not json", encoding="utf-8")
    dog = watchdog.Watchdog(tmp_path, transports=[Recorder()])
    assert dog.state["active"] == {} and dog.state["pending"] == []


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------

def test_declared_outcomes_arrive_as_one_message_not_thirty(tmp_path):
    """The recorded campaign produced 30 gate stops. That is one message, not thirty."""
    recorder = Recorder()
    dog = watchdog.Watchdog(tmp_path, transports=[recorder], digest_seconds=1e9, digest_max=100)
    runs = [run(f"r{i}", "stopped", stop_reason="likelihood_breach") for i in range(30)]
    dog.tick(snapshot(runs=runs))
    assert recorder.sent == []                              # nothing urgent, nothing sent yet
    dog.tick(snapshot(runs=runs), force_digest=True)
    assert len(recorder.sent) == 1
    summary = recorder.sent[0]
    assert summary["key"] == "digest" and "30 events" in summary["title"]
    assert "30 x stopped" in summary["body"]
    assert dog.state["digest"] == [] and dog.state["digest_opened_at"] is None


def test_the_digest_goes_early_once_enough_has_queued(tmp_path):
    recorder = Recorder()
    dog = watchdog.Watchdog(tmp_path, transports=[recorder], digest_seconds=1e9, digest_max=3)
    dog.tick(snapshot(runs=[run(f"r{i}", "stopped") for i in range(3)]))
    assert len(recorder.sent) == 1 and "3 events" in recorder.sent[0]["title"]


def test_a_terminal_campaign_flushes_what_is_queued(tmp_path):
    recorder = Recorder()
    dog = watchdog.Watchdog(tmp_path, transports=[recorder], digest_seconds=1e9)
    dog.tick(snapshot(runs=[run("a", "completed")]), force_digest=True)
    assert any("Completed: a" in e["body"] for e in recorder.sent)


# ---------------------------------------------------------------------------
# transports and the heartbeat
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,expected", [
    ("text", b"Title\nbody"),
    ("slack", b'{"text": "Title\\nbody"}'),
    ("discord", b'{"content": "Title\\nbody"}')])
def test_webhook_bodies_match_each_service_convention(fmt, expected):
    body, _ = watchdog.webhook_body({"title": "Title", "body": "body", "severity": "urgent"}, fmt)
    assert body == expected


def test_the_watchdog_records_its_own_liveness(dog):
    dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    beat = json.loads((dog.directory / "watchdog_heartbeat.json").read_text())
    assert beat["active_conditions"] == ["stalled:a"]
    assert beat["pid"] == os.getpid() and beat["updated_at"]


def test_the_watchdog_writes_nothing_into_the_campaign_it_reads(tmp_path):
    """Read-only, like the dashboard: its own three files and nothing else."""
    campaign, state = tmp_path / "campaign", tmp_path / "state"
    campaign.mkdir()
    (campaign / "plan.json").write_text("{}", encoding="utf-8")
    dog = watchdog.Watchdog(state, transports=[Recorder()])
    dog.tick(snapshot(runs=[run("a", "stalled", idle_seconds=900)]))
    assert sorted(p.name for p in campaign.iterdir()) == ["plan.json"]
    assert sorted(p.name for p in state.iterdir()) == [
        "watchdog.jsonl", "watchdog_heartbeat.json", "watchdog_state.json"]
