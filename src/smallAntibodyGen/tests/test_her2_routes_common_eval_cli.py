"""The common evaluation, slice E: the orchestration and the command line.

The orchestration functions are exercised with their seams replaced by recorders -- what is
under test is ORDER and REFUSAL: verify before anything is scored, never persist the split, check
the identity-marker tripwire after, refuse mismatched panels, and never score a file the policy
forbids. The command line is exercised for real in a subprocess where that is cheap (``--help``,
``selfcheck``), because the one property that cannot be tested in-process is the one that
matters: a fresh interpreter must not initialise CUDA.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_routes_common_eval as ce
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.tests.test_her2_routes_common_eval import panel_rows
from smallAntibodyGen.tests.test_her2_routes_common_eval_flights import PanelServices, make_flight

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "her2_routes_common_eval.py"


def load_script():
    spec = importlib.util.spec_from_file_location("her2_routes_common_eval_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

class Recorder:
    """Replaces the flight seams and records the order they were called in."""

    def __init__(self, monkeypatch, tmp_path):
        self.calls = []
        self.tmp_path = tmp_path
        for name in ("verify_flight", "prepare_scratch_flight",
                     "open_services", "fingerprint_markers", "assert_markers_unchanged",
                     "extract_panel"):
            monkeypatch.setattr(ce, name, getattr(self, name))

    def verify_flight(self, flight, scratch):
        self.calls.append(("verify", flight.name))
        return {"flight": flight.name, "source": {"verified": True},
                "external": {"verified": True}}

    def prepare_scratch_flight(self, flight, scratch_root):
        self.calls.append(("scratch", flight.name))
        return types.SimpleNamespace(run_root=Path(scratch_root) / f"flight_{flight.name}",
                                     cache=Path(scratch_root) / "cache" / "sources", markers={})

    #: What the (fake) flight yields as its development panel; a test changes it to model a
    #: flight whose split or assay moved after the saved panel was extracted.
    rows = None

    def open_services(self, flight, scratch, *, threads=2):
        self.calls.append(("services", flight.name, threads))
        return PanelServices(self.tmp_path / f"svc_{flight.name}", self.rows or panel_rows(6))

    def fingerprint_markers(self, flight):
        self.calls.append(("fingerprint", flight.name))
        return {"m": "hash"}

    def assert_markers_unchanged(self, flight, before):
        self.calls.append(("tripwire", flight.name, before))

    def extract_panel(self, services, *, flight_name):
        self.calls.append(("extract", flight_name))
        panel_services = services
        rows = ce.validate_panel(panel_services.rows, development_windows=[
            r["window"] for r in panel_services.rows])
        labels = ce.neighbourhood_components([row["window"] for row in rows], radius=2)
        context = {"assay": "a", "split_sha256": "s" * 64, "scaffold_sha256": "c" * 64,
                   "sites": 10, "grouping": "connected_components", "neighbourhood_radius": 2}
        return {"schema": ce.SCHEMA, "record_kind": "panel", "flight": flight_name,
                "digest": ce.panel_digest(rows, labels, context=context), "rows": rows,
                "neighbourhoods": labels, "grouping": ce.grouping_record(labels, radius=2),
                "context": context, "reserved_opened": False, "evaluator": {}}


def test_the_panel_command_verifies_first_never_persists_and_checks_the_tripwire(
        tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    recorder = Recorder(monkeypatch, tmp_path)
    panels = ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)
    kinds = [call[0] for call in recorder.calls]
    assert kinds == ["fingerprint", "scratch", "verify", "services", "extract", "tripwire"], (
        "markers are fingerprinted (through pins) first, the scratch root is seeded from the "
        "captured bytes, and verification runs against that scratch copy")
    assert recorder.calls[-1][2] == {"m": "hash"}, "the tripwire compares against the BEFORE picture"
    written = paths.read_json(tmp_path / "out" / f"panel_{flight.name}.json")
    assert written["digest"] == panels[0]["digest"]
    assert (tmp_path / "out" / "eval_spec.json").is_file()


def test_re_extracting_an_identical_panel_is_allowed_and_a_changed_one_is_refused(
        tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    Recorder(monkeypatch, tmp_path)
    ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)
    ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)
    existing = paths.read_json(tmp_path / "out" / f"panel_{flight.name}.json")
    existing["digest"] = "0" * 64
    paths.write_json(tmp_path / "out" / f"panel_{flight.name}.json", existing)
    with pytest.raises(ValueError, match="does not match its content"):
        ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)


def seed_panels(tmp_path, monkeypatch, flights):
    recorder = Recorder(monkeypatch, tmp_path)
    ce.run_panel(flights, tmp_path / "out", tmp_path / "scratch", threads=2)
    recorder.calls.clear()
    return recorder


def test_the_score_command_refuses_to_run_without_the_panel_for_every_flight(tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    Recorder(monkeypatch, tmp_path)
    ce.ensure_output_directory(tmp_path / "out", [flight])
    with pytest.raises(ValueError, match="no panel"):
        ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None, labels=None,
                     allow_live=False, batch_size=2, threads=2)


def test_the_score_command_verifies_plans_scores_and_checks_the_tripwire(tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    bakeoff = flight.run_root / "bakeoff" / "ablang2_masked_editor"
    bakeoff.mkdir(parents=True)
    (bakeoff / "best.pt").write_bytes(b"w")
    (bakeoff / "run.json").write_text("{}", encoding="utf-8")   # train() writes it last
    recorder = seed_panels(tmp_path, monkeypatch, [flight])
    scored = []

    def fake_ensure(services, subject, rows, **kwargs):
        assert "flight_running" not in kwargs, "a cached liveness answer is never passed down"
        scored.append((subject.key, kwargs["scratch"], kwargs["allow_live"],
                       kwargs["batch_size"], len(rows)))
        return {"key": subject.key}

    monkeypatch.setattr(ce, "ensure_subject", fake_ensure)
    result = ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None,
                          labels=None, allow_live=False, batch_size=3, threads=2)
    kinds = [call[0] for call in recorder.calls]
    assert kinds == ["fingerprint", "scratch", "verify", "services", "extract", "tripwire"], (
        "the saved panel is re-derived from the flight before the first subject is scored")
    assert [entry[0] for entry in scored] == ["t1__ablang2_masked_editor__best",
                                              "t1__ablang2_masked_editor__zero_shot"]
    assert all(entry[1] == tmp_path / "scratch" / "copies" for entry in scored)
    assert all(entry[2] is False and entry[3:] == (3, 6) for entry in scored)
    assert result["scored"] == [entry[0] for entry in scored]


def test_a_live_flight_is_scored_from_pinned_snapshots_and_nothing_is_skipped(
        tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    bakeoff = flight.run_root / "bakeoff" / "dplm_150m_masked"
    bakeoff.mkdir(parents=True)
    for name in ("best.pt", "last.pt", "step5000.pt"):
        (bakeoff / name).write_bytes(b"w")
    seed_panels(tmp_path, monkeypatch, [flight])
    scored = []
    monkeypatch.setattr(ce, "ensure_subject",
                        lambda services, subject, rows, **kw: scored.append(subject.key) or {})
    result = ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None,
                          labels=None, allow_live=False, batch_size=2, threads=2)
    assert scored == [
        "t1__dplm_150m_masked__best", "t1__dplm_150m_masked__last",
        "t1__dplm_150m_masked__step5000", "t1__dplm_150m_masked__zero_shot"]
    assert result["skipped"]["t1"] == {}


def test_a_second_command_on_the_same_output_directory_fails_before_touching_a_flight(
        tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    recorder = seed_panels(tmp_path, monkeypatch, [flight])
    with ce.exclusive_invocation(tmp_path / "out"):
        with pytest.raises(ValueError, match="another evaluator invocation"):
            ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None,
                         labels=None, allow_live=False, batch_size=2, threads=2)
        with pytest.raises(ValueError, match="another evaluator invocation"):
            ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)
    assert recorder.calls == []


def test_a_saved_panel_that_the_flight_no_longer_produces_stops_the_score_command(
        tmp_path, monkeypatch):
    """A digest that is self-consistent proves the FILE is intact, not that it describes THIS
    flight: a panel copied from another evaluation, or a flight whose split moved after the panel
    was extracted, would otherwise be scored and recorded against the wrong rows."""
    flight = make_flight(tmp_path)
    bakeoff = flight.run_root / "bakeoff" / "ablang2_masked_editor"
    bakeoff.mkdir(parents=True)
    (bakeoff / "best.pt").write_bytes(b"w")
    recorder = seed_panels(tmp_path, monkeypatch, [flight])
    moved = panel_rows(6)
    moved[0] = {**moved[0], "value": moved[0]["value"] + 1.0}
    recorder.rows = moved
    scored = []
    monkeypatch.setattr(ce, "ensure_subject",
                        lambda services, subject, rows, **kw: scored.append(subject.key) or {})
    with pytest.raises(ValueError, match="no longer produces"):
        ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None, labels=None,
                     allow_live=False, batch_size=2, threads=2)
    assert scored == [], "nothing is scored against a panel the flight does not produce"


def test_the_subjects_are_handed_the_scratch_markers_of_their_flight(tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    bakeoff = flight.run_root / "bakeoff" / "ablang2_masked_editor"
    bakeoff.mkdir(parents=True)
    (bakeoff / "best.pt").write_bytes(b"w")
    seed_panels(tmp_path, monkeypatch, [flight])
    seen = []
    monkeypatch.setattr(ce, "ensure_subject",
                        lambda services, subject, rows, **kw: seen.append(kw["flight_markers"]) or {})
    ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None, labels=None,
                 allow_live=False, batch_size=2, threads=2)
    assert seen and all(markers == {} for markers in seen), (
        "the markers come from the scratch flight (the recorder returns an empty set)")


def test_two_commands_that_share_a_scratch_root_cannot_run_at_once(tmp_path, monkeypatch):
    """Different output directories hold different output locks, but the scratch flights and the
    scratch source cache live under the scratch root: preparing a flight truncates the markers and
    overwrites the cache the other invocation may be reading."""
    flight = make_flight(tmp_path)
    recorder = seed_panels(tmp_path, monkeypatch, [flight])
    with ce.exclusive_invocation(tmp_path / "scratch", name=ce.SCRATCH_LOCK):
        with pytest.raises(ValueError, match="another evaluator invocation"):
            ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)
        with pytest.raises(ValueError, match="another evaluator invocation"):
            ce.run_score([flight], tmp_path / "out", tmp_path / "scratch", routes=None,
                         labels=None, allow_live=False, batch_size=2, threads=2)
    assert recorder.calls == [], "no flight work may start under a held scratch root"
    ce.run_panel([flight], tmp_path / "out", tmp_path / "scratch", threads=2)   # released again


def test_panels_that_disagree_across_flights_stop_the_score_command(tmp_path, monkeypatch):
    first, second = make_flight(tmp_path / "a", name="t1"), make_flight(tmp_path / "b", name="dplm",
                                                                       revision="dplm_rev")
    seed_panels(tmp_path, monkeypatch, [first, second])
    changed = paths.read_json(tmp_path / "out" / "panel_dplm.json")
    changed["rows"][0]["value"] += 1.0                      # a genuinely different panel ...
    changed["digest"] = ce.panel_digest(changed["rows"], changed["neighbourhoods"],
                                        context=changed["context"])   # ... that is self-consistent
    paths.write_json(tmp_path / "out" / "panel_dplm.json", changed)
    with pytest.raises(ValueError, match="different panel"):
        ce.run_score([first, second], tmp_path / "out", tmp_path / "scratch", routes=None,
                     labels=None, allow_live=False, batch_size=2, threads=2)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_the_parser_requires_an_output_directory_and_at_least_one_flight_where_needed():
    script = load_script()
    parser = script.build_parser()
    args = parser.parse_args(["score", "--out-dir", "o", "--flight", "t1=/runs/r/flight",
                              "--route", "ablang2_masked_editor", "--labels", "best,last",
                              "--allow-live-checkpoints", "--batch-size", "3"])
    assert args.command == "score" and args.flight == ["t1=/runs/r/flight"]
    assert args.route == ["ablang2_masked_editor"] and args.labels == "best,last"
    assert args.allow_live_checkpoints is True and args.batch_size == 3 and args.threads == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--flight", "t1=/x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["panel", "--out-dir", "o"])
    summarize = parser.parse_args(["summarize", "--out-dir", "o"])
    assert summarize.command == "summarize"


def test_a_flight_argument_must_be_a_name_equals_path_pair():
    script = load_script()
    assert script.parse_flight("t1=/a/b") == ("t1", Path("/a/b"))
    for bad in ("nopair", "=/a", "t1=", "bad name=/a"):
        with pytest.raises(ValueError):
            script.parse_flight(bad)


def test_help_runs_without_importing_torch():
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys, runpy; sys.argv=['x','--help']\n"
         "try:\n runpy.run_path(r'%s', run_name='__main__')\n"
         "except SystemExit as stop:\n print('exit', stop.code)\n"
         "print('torch' in sys.modules)" % SCRIPT],
        capture_output=True, text=True, timeout=120)
    assert "exit 0" in result.stdout and result.stdout.strip().endswith("False"), result.stderr


def test_destinations_inside_a_flight_are_refused_before_anything_is_created(tmp_path):
    """The output directory and the scratch root are checked against EVERY protected location
    of EVERY flight before any filesystem mutation. Environment preparation creates ``tmp``
    under the scratch root, and the marker tripwire would not notice a new directory."""
    flight = make_flight(tmp_path)
    for protected in (flight.run_root, flight.artifact_root, flight.worktree):
        inside = protected / "common_eval"
        with pytest.raises(ValueError):
            ce.validate_destinations([flight], inside, tmp_path / "scratch_ok")
        with pytest.raises(ValueError):
            ce.validate_destinations([flight], tmp_path / "out_ok", inside)
        assert not inside.exists()
    ce.validate_destinations([flight], tmp_path / "out_ok", tmp_path / "out_ok" / "scratch")


def test_the_panel_and_score_commands_validate_destinations_before_verifying_anything(
        tmp_path, monkeypatch):
    flight = make_flight(tmp_path)
    recorder = Recorder(monkeypatch, tmp_path)
    inside = flight.run_root / "common_eval"
    with pytest.raises(ValueError):
        ce.run_panel([flight], inside, tmp_path / "scratch", threads=2)
    with pytest.raises(ValueError):
        ce.run_score([flight], tmp_path / "out", inside, routes=None, labels=None,
                     allow_live=False, batch_size=2, threads=2)
    assert recorder.calls == [] and not inside.exists()


def test_the_output_directory_is_initialised_before_the_environment_creates_scratch(
        tmp_path, monkeypatch):
    """The default scratch root is ``<out-dir>/scratch``. Environment preparation creates
    ``scratch/tmp``; if that ran first the output directory would be non-empty without an
    ``eval_spec.json`` and be refused as foreign (reproduced in review)."""
    script = load_script()
    flight = make_flight(tmp_path)
    out = tmp_path / "out"
    seen = {}

    def fake_prepare(flights, scratch_root):
        seen["spec_existed"] = (out / "eval_spec.json").is_file()
        (Path(scratch_root) / "tmp").mkdir(parents=True)        # what prepare_environment does

    monkeypatch.setattr(script, "prepare_environment", fake_prepare)
    monkeypatch.setattr(ce, "run_panel", lambda *a, **k: [{"digest": "d" * 64}])
    assert script.main(["panel", "--out-dir", str(out), "--flight",
                        f"t1={flight.run_root}"]) == 0
    assert seen["spec_existed"] is True
    assert ce.ensure_output_directory(out, [flight])["flights"]["t1"], (
        "the directory is still recognised as the evaluator's own afterwards")


def test_a_scratch_root_inside_a_flight_stops_the_command_line_before_any_mkdir(
        tmp_path, monkeypatch):
    script = load_script()
    flight = make_flight(tmp_path)
    monkeypatch.setattr(script, "prepare_environment",
                        lambda *a, **k: pytest.fail("environment preparation must not run"))
    out = tmp_path / "out2"
    with pytest.raises(ValueError):
        script.main(["panel", "--out-dir", str(out), "--scratch-root",
                     str(flight.run_root / "s"), "--flight", f"t1={flight.run_root}"])
    assert not out.exists() and not (flight.run_root / "s").exists()


def test_selfcheck_in_a_fresh_interpreter_reports_a_guarded_cpu_only_process():
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "selfcheck"],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["guard_active"] is True
    assert report["cuda_initialized"] is False and report["cuda_device_count"] == 0
    assert report["cpu_backward_ok"] is True
