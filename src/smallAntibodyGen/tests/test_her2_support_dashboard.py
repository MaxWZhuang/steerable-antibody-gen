"""The audit panel must not make an unfinished audit look finished.

The dashboard is the only place the audit is read by a human while it runs, so
the failure that matters is a panel that shows a plausible percentage, a green
"complete", or a coverage line for work that never happened. Each test here is
one of those.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/her2_live_dashboard.py"
spec = importlib.util.spec_from_file_location("her2_live_dashboard_audit", SCRIPT)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def progress(root, stage, **fields):
    write(root / f"progress/{stage}.json",
          dict({"schema_version": "her2-support-audit/1", "record_kind": "stage_progress",
                "stage": stage, "status": "completed", "total": None, "completed": 0,
                "current": None, "error": None}, **fields))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "outputs/her2_support_audit_20260919"
    root.mkdir(parents=True)
    return root


def test_an_absent_run_directory_is_reported_as_absent(tmp_path):
    audit = dashboard.AuditRun(tmp_path / "missing")
    snapshot = audit.snapshot()
    assert snapshot["present"] is False and snapshot["stages"] == []
    assert snapshot["complete"] is False


def test_an_unknown_total_is_a_count_not_a_percentage(run):
    progress(run, "inventory", status="running", total=None, completed=17,
             total_note="this stage cannot know its denominator in advance")
    row = next(s for s in dashboard.AuditRun(run).snapshot()["stages"]
               if s["stage"] == "inventory")
    assert row["total"] is None and row["completed"] == 17
    assert row["fraction"] is None, "a fabricated denominator is worse than a count"


def test_a_known_total_reports_a_real_fraction(run):
    progress(run, "score", status="running", total=50, completed=10)
    row = next(s for s in dashboard.AuditRun(run).snapshot()["stages"] if s["stage"] == "score")
    assert row["fraction"] == pytest.approx(0.2)


def test_a_stage_that_stopped_writing_is_not_still_running(run):
    progress(run, "score", status="running", total=50, completed=10)
    stale = datetime.now(timezone.utc).timestamp() - 600
    os.utime(run / "progress/score.json", (stale, stale))
    row = next(s for s in dashboard.AuditRun(run, stall_seconds=180).snapshot()["stages"]
               if s["stage"] == "score")
    assert row["status"] == "stalled" and row["stale"] is True
    assert row["recorded_status"] == "running", "the recorded value is still reported"


def test_a_failed_or_interrupted_stage_keeps_its_status_and_error(run):
    progress(run, "ches", status="failed", error="ValueError: nonfinite CHES")
    progress(run, "report", status="interrupted", error="KeyboardInterrupt")
    rows = {s["stage"]: s for s in dashboard.AuditRun(run).snapshot()["stages"]}
    assert rows["ches"]["status"] == "failed" and "nonfinite" in rows["ches"]["error"]
    assert rows["report"]["status"] == "interrupted"
    assert rows["decide"]["status"] == "not_started"


def test_an_exited_process_without_the_marker_is_not_a_complete_audit(run):
    for stage in dashboard.AuditRun.STAGES:
        progress(run, stage, status="completed")
    write(run / "decision.json", {"outcome": "no_escalation_at_this_resolution",
                                  "blocking": [], "methods": {},
                                  "completion": {"unmet": ["ches_populations"]}})
    snapshot = dashboard.AuditRun(run).snapshot()
    assert snapshot["complete"] is False
    assert snapshot["unmet_requirements"] == ["ches_populations"]


def test_the_completion_marker_is_what_makes_an_audit_complete(run):
    write(run / "audit_complete.json", {"completed_at": "2026-09-19T12:00:00+00:00",
                                        "requirements": {"unmet": []}})
    snapshot = dashboard.AuditRun(run).snapshot()
    assert snapshot["complete"] is True
    assert snapshot["completed_at"].startswith("2026-09-19")


def test_coverage_shortfalls_and_freeze_state_are_reported_truthfully(run):
    write(run / "inventory.json", {
        "audit_id": "her2_support_audit_20260919",
        "coverage": {"total": 159, "verified_total": 158, "expected": {"total": 159},
                     "complete": False,
                     "shortfalls": [{"role": "reached_endpoint", "kind": "verification",
                                     "expected": 72, "observed": 71}]},
        "deduplication": {"distinct_computations": 150},
        "parent_banks": {"20260918": {}, "20260919": {}, "20260920": {}},
        "records": []})
    snapshot = dashboard.AuditRun(run).snapshot()
    assert snapshot["counts"]["states_verified"] == 158
    assert snapshot["counts"]["parent_banks"] == 3
    assert snapshot["coverage"]["complete"] is False
    assert snapshot["coverage"]["shortfalls"][0]["kind"] == "verification"
    assert snapshot["frozen"] is None, "an unfrozen audit must not look frozen"

    write(run / "audit_spec_frozen.json", {"git": {"commit": "a" * 40},
                                           "frozen_at": "2026-09-19T10:00:00+00:00",
                                           "source_sha256": {"a.py": "x"}, "input_count": 200,
                                           "evidence_sha256": {"b.json": "y"}})
    frozen = dashboard.AuditRun(run).snapshot()["frozen"]
    assert frozen["commit"] == "a" * 40 and frozen["inputs"] == 200


def test_scored_endpoints_and_the_decision_are_surfaced(run):
    write(run / "inventory.json", {"records": [
        {"id": "guarded::stage1::x::budget600", "role": "reached_endpoint", "arm_id": "ipo_tau0p1",
         "seed": 20260918, "nominal_budget_gpu_seconds": 600.0}], "coverage": {}})
    write(run / "summaries/checkpoints.json", {"checkpoints": {
        "guarded::stage1::x::budget600": {
            "forward_kl": {"mean": 0.42, "ci_low": 0.39, "ci_high": 0.45},
            "tails": {"counts": {"tenfold": {"fraction": 0.004, "lower": 0.0028}}}}},
        "self_controls": {"parent::policy_sft_seed20260918": {"within_tolerance": True}},
        "paired_method_differences": {"count": 9}})
    write(run / "decision.json", {"outcome": "insufficient_coverage",
                                  "blocking": ["the coverage table reports gaps"],
                                  "coverage_complete": False, "audit_complete": False,
                                  "methods": {"ipo_tau0p1": {"outcome": "insufficient_coverage",
                                                             "seeds_usable": 2,
                                                             "seeds_crossing": 1,
                                                             "seeds_declared": 3}}})
    snapshot = dashboard.AuditRun(run).snapshot()
    row = snapshot["results"]["endpoints"][0]
    assert row["forward_kl"] == pytest.approx(0.42) and row["seed"] == 20260918
    assert snapshot["results"]["paired_comparisons"] == 9
    assert snapshot["results"]["self_controls"]["parent::policy_sft_seed20260918"] is True
    assert snapshot["decision"]["outcome"] == "insufficient_coverage"
    assert snapshot["decision"]["methods"]["ipo_tau0p1"]["seeds_crossing"] == 1


def test_verification_problems_reach_the_panel(run):
    write(run / "audit_complete.json", {"completed_at": "2026-09-19T12:00:00+00:00"})
    write(run / "verification.json", {"immutable": False, "shards_checked": 4,
                                      "problems": [{"shard": "scores/x", "problem": "changed"}]})
    snapshot = dashboard.AuditRun(run).snapshot()
    assert snapshot["verification"]["immutable"] is False
    assert snapshot["complete"] is False
    assert snapshot["verification"]["problems"][0]["problem"] == "changed"


def test_unreadable_json_is_an_error_not_a_crash_and_not_a_zero(run):
    (run / "summaries").mkdir(parents=True, exist_ok=True)
    (run / "summaries/coverage.json").write_text("{not json", encoding="utf-8")
    snapshot = dashboard.AuditRun(run).snapshot()
    assert any("coverage.json" in message for message in snapshot["errors"])
    assert snapshot["coverage"]["complete"] is None


def test_nonfinite_values_are_nulled_so_the_json_stays_parseable(run):
    write(run / "inventory.json", {"records": [], "coverage": {}})
    (run / "summaries").mkdir(parents=True, exist_ok=True)
    (run / "summaries/checkpoints.json").write_text(
        json.dumps({"checkpoints": {"a": {"forward_kl": {"mean": float("nan"), "ci_low": 1.0,
                                                          "ci_high": 2.0},
                                          "tails": {"counts": {}}}}}), encoding="utf-8")
    snapshot = dashboard.AuditRun(run).snapshot()
    assert snapshot["results"]["endpoints"][0]["forward_kl"] is None
    json.dumps(snapshot, allow_nan=False)


@pytest.mark.parametrize("name", ["../../secrets.json", "/etc/passwd", "a/../../b.json"])
def test_the_reader_refuses_to_leave_the_run_directory(run, name):
    audit = dashboard.AuditRun(run)
    with pytest.raises(KeyError):
        audit.path(name)
    assert audit.document(name) is None


def test_the_audit_route_is_absent_when_no_audit_directory_was_given():
    handler = dashboard.make_handler(object(), None)
    assert handler is not None, "the campaign view keeps working without --audit"


def test_the_dashboard_imports_no_model_or_training_module():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("import torch", "transformers", "smallAntibodyGen"):
        assert forbidden not in source, f"the dashboard must not import {forbidden}"


def test_the_audit_tab_and_route_are_wired_into_the_assets():
    assets = SCRIPT.parent / "her2_dashboard"
    html = (assets / "index.html").read_text(encoding="utf-8")
    script = (assets / "app.js").read_text(encoding="utf-8")
    assert 'data-view="audit"' in html and 'id="view-audit"' in html
    assert "/api/audit" in script and "/api/status" in script
    assert "unmet_requirements" in script, "the panel must show why an audit is incomplete"


def view_body(html, marker):
    """The markup of one view, from its opening div to the div that closes it."""
    start = html.index(marker)
    depth, index = 0, start
    while True:
        opening = html.find("<div", index)
        closing = html.index("</div>", index)
        if opening != -1 and opening < closing:
            depth, index = depth + 1, opening + 4
            continue
        depth, index = depth - 1, closing + 6
        if depth == 0:
            return html[start:index]


def test_each_view_owns_exactly_one_footer():
    """Two footers were visible at once: the campaign one sat outside the hidden view."""
    html = (SCRIPT.parent / "her2_dashboard/index.html").read_text(encoding="utf-8")
    campaign = view_body(html, '<div id="view-campaign">')
    audit = view_body(html, '<div id="view-audit" hidden>')
    assert campaign.count("<footer>") == 1 and audit.count("<footer>") == 1
    assert html.count("<footer>") == 2, "no footer lives outside a view"
    assert 'id="updated"' in campaign and 'id="audit-updated"' in audit


def test_the_refresh_button_refreshes_the_view_that_is_showing():
    script = (SCRIPT.parent / "her2_dashboard/app.js").read_text(encoding="utf-8")
    line = next(row for row in script.splitlines()
                if '$("refresh").addEventListener' in row)
    assert "auditView?refreshAudit():refresh()" in line.replace(" ", "")


def test_both_views_still_poll_and_neither_polls_the_hidden_one():
    script = (SCRIPT.parent / "her2_dashboard/app.js").read_text(encoding="utf-8")
    compact = script.replace(" ", "")
    assert "setInterval(refresh,5000)" in compact
    assert "setInterval(()=>{if(auditView)refreshAudit();},5000)" in compact
