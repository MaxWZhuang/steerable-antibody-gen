"""The common evaluation, slice C: reading a flight WITHOUT writing into it or opening what it replaces.

The hazards these tests pin were found by reading the code paths the evaluator takes:

* ``FlightServices.split()`` calls ``freeze_hard_split(persist=True)``, which REWRITES the flight's
  ``hard_split_freeze.json``. The evaluator therefore runs against a scratch run root seeded with
  byte-copies of the identity markers, and extracts the panel through ``persist=False``.
* A supervisor RESTART rewrites that marker too (``curriculum_stream`` -> ``closure`` ->
  ``freeze_hard_split(persist=True)``), and on Windows an ordinary read handle makes the atomic
  replace fail. So every replaceable flight file (``launch.json``, the four identity markers) is
  CAPTURED once through a hard-link pin, and everything afterwards uses the captured bytes.
* The Li and paired-OAS readers rewrite their shared cache non-atomically on a miss, so the
  services are pointed at a scratch COPY of ``<artifact>/cache/sources``.
* A checkpoint's identity hashes the config FILE BYTES, so the context is built from the flight's
  frozen worktree config, and the frozen source and external bytes are verified before anything is
  scored, because copying a marker cannot verify what it points at.

What is real: the file copies, hard links, hashes and path arithmetic. What is replaced:
``her2_routes_spec``'s verifiers, ``resolve_context`` and ``FlightServices`` (recording fakes),
and the services object used for panel extraction. Nothing here touches a real flight.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from smallAntibodyGen.experiments import her2_nf_spec as nf_spec
from smallAntibodyGen.experiments import her2_routes_common_eval as ce
from smallAntibodyGen.experiments import her2_routes_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.tests.test_her2_routes_common_eval import panel_rows
from smallAntibodyGen.tests.test_her2_routes_common_eval_scoring import forbid_opening

# ---------------------------------------------------------------------------
# a synthetic flight on disk
# ---------------------------------------------------------------------------


def make_flight(tmp_path, *, name="t1", revision="t1_rev01", markers=True, archive=False):
    artifact = tmp_path / "artifact"
    worktree = artifact / "worktrees" / "bakeoff_rev01"
    config = worktree / "configs" / "experiments" / "her2_generative_routes.json"
    run_root = artifact / "runs" / revision / "flight"
    for directory in (config.parent, run_root, artifact / "cache" / "sources" / "li_table_a"):
        directory.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    (artifact / "cache" / "sources" / "li_table_a" / "index.json").write_text("{}", "utf-8")
    command = ["python.exe", str(worktree / "scripts" / "her2_routes_supervisor.py"),
               "--device", "cuda", "--max-restarts", "0", "--config", str(config),
               "--artifact-root", str(artifact), "--revision", revision]
    (run_root / "launch.json").write_text(json.dumps(
        {"record_kind": "launch", "command": command}), encoding="utf-8")
    if markers:
        for index, marker in enumerate(ce.MARKERS):
            (run_root / marker).write_bytes(f"marker {index} é\r\nbytes".encode("utf-8"))
    if archive:
        (run_root / "source_snapshot" / "src").mkdir(parents=True)
        (run_root / "source_snapshot" / "src" / "a.py").write_text("frozen\n", encoding="utf-8")
    return ce.flight_from_run_root(name, run_root, pin_parent=tmp_path / "pins")


def recapture(flight, tmp_path):
    return ce.flight_from_run_root(flight.name, flight.run_root, pin_parent=tmp_path / "pins")


def test_the_marker_list_is_exactly_what_the_training_identity_reads():
    assert set(ce.MARKERS) == {spec.SPLIT_FREEZE, spec.EXTERNAL_MARKER, spec.FREEZE_MARKER,
                               "condition_vocabulary.json"}
    assert ce.SNAPSHOT_DIR == nf_spec.SNAPSHOT_DIR


def test_a_flight_is_recovered_from_its_own_launch_record_with_its_markers_captured(tmp_path):
    flight = make_flight(tmp_path)
    artifact = tmp_path / "artifact"
    assert flight.name == "t1" and flight.revision == "t1_rev01"
    assert flight.artifact_root == artifact
    assert flight.worktree == artifact / "worktrees" / "bakeoff_rev01"
    assert flight.config_path == (flight.worktree / "configs" / "experiments"
                                  / "her2_generative_routes.json")
    assert flight.run_root == artifact / "runs" / "t1_rev01" / "flight"
    assert set(flight.markers) == set(ce.MARKERS)
    for marker in ce.MARKERS:
        assert flight.markers[marker] == (flight.run_root / marker).read_bytes()
    assert not ce.leftover_pins(tmp_path / "pins"), "capturing leaves no pin behind"


@pytest.mark.parametrize("mutate, message", [
    (lambda doc: doc.pop("command"), "command"),
    (lambda doc: doc.update(record_kind="other"), "launch record"),
    (lambda doc: doc.update(command=[a for a in doc["command"] if a != "--config"]), "--config"),
    (lambda doc: doc.update(command=doc["command"][:-2]), "--revision"),
])
def test_an_unusable_launch_record_is_refused(tmp_path, mutate, message):
    flight = make_flight(tmp_path)
    launch = flight.run_root / "launch.json"
    document = json.loads(launch.read_text(encoding="utf-8"))
    mutate(document)
    launch.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        recapture(flight, tmp_path)


def test_a_config_outside_the_flights_worktree_is_refused(tmp_path):
    flight = make_flight(tmp_path)
    launch = flight.run_root / "launch.json"
    document = json.loads(launch.read_text(encoding="utf-8"))
    at = document["command"].index("--config")
    document["command"][at + 1] = str(tmp_path / "elsewhere.json")
    launch.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="worktree"):
        recapture(flight, tmp_path)


def test_a_missing_marker_is_refused_at_capture_not_synthesised(tmp_path):
    flight = make_flight(tmp_path)
    (flight.run_root / ce.MARKERS[0]).unlink()
    with pytest.raises(ValueError, match=ce.MARKERS[0]):
        recapture(flight, tmp_path)


# ---------------------------------------------------------------------------
# replaceable flight files are never opened by their live names
# ---------------------------------------------------------------------------

def test_capturing_and_fingerprinting_never_open_a_replaceable_live_name(tmp_path, monkeypatch):
    """A restarting supervisor atomically replaces ``hard_split_freeze.json`` (and may replace the
    other markers and ``launch.json``); an ordinary read handle makes that replace fail on
    Windows (measured: WinError 5), and the supervisor does not retry. Review finding."""
    flight = make_flight(tmp_path)
    forbid_opening(monkeypatch, names=("launch.json",) + ce.MARKERS)   # armed after the fixtures
    captured = recapture(flight, tmp_path)
    before = ce.fingerprint_markers(captured)
    ce.assert_markers_unchanged(captured, before)
    assert set(before) == set(ce.MARKERS)
    assert not ce.leftover_pins(tmp_path / "pins")


def test_a_marker_that_cannot_be_pinned_is_refused_not_opened_live(tmp_path, monkeypatch):
    flight = make_flight(tmp_path)

    def refuse(*_args, **_kwargs):
        raise OSError(17, "The system cannot move the file to a different disk drive")

    monkeypatch.setattr(ce.os, "link", refuse)
    with pytest.raises(ValueError, match="hard-link snapshot is impossible"):
        recapture(flight, tmp_path)


# ---------------------------------------------------------------------------
# the scratch flight
# ---------------------------------------------------------------------------

def listing(root):
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_the_scratch_run_root_holds_the_captured_markers_a_linked_archive_and_a_cache_copy(
        tmp_path):
    flight = make_flight(tmp_path, archive=True)
    before = listing(flight.artifact_root)
    scratch = ce.prepare_scratch_flight(flight, tmp_path / "scratch")
    for marker in ce.MARKERS:
        assert (scratch.run_root / marker).read_bytes() == flight.markers[marker]
    assert (scratch.run_root / "source_snapshot" / "src" / "a.py").read_text("utf-8") == "frozen\n"
    assert (scratch.cache / "li_table_a" / "index.json").is_file(), "cache hits are preserved"
    assert scratch.cache != flight.artifact_root / "cache" / "sources"
    assert listing(flight.artifact_root) == before, "preparing the scratch flight wrote nothing"


def test_the_scratch_copy_is_the_captured_state_not_whatever_the_live_file_became(tmp_path):
    """The identity is computed from the scratch markers, so they must be the captured bytes: a
    marker rewritten after the capture (a restart) must not leak into the evaluation."""
    flight = make_flight(tmp_path)
    captured = flight.markers[ce.MARKERS[0]]
    (flight.run_root / ce.MARKERS[0]).write_bytes(b"rewritten by a restarted supervisor")
    scratch = ce.prepare_scratch_flight(flight, tmp_path / "scratch")
    assert (scratch.run_root / ce.MARKERS[0]).read_bytes() == captured
    with pytest.raises(ValueError, match=ce.MARKERS[0]):
        ce.assert_markers_unchanged(flight, {m: paths.sha256_file(scratch.run_root / m)
                                             for m in ce.MARKERS})


@pytest.mark.parametrize("where", ["artifact", "run_root", "worktree"])
def test_a_scratch_root_inside_any_protected_location_is_refused(tmp_path, where):
    flight = make_flight(tmp_path)
    protected = {"artifact": flight.artifact_root, "run_root": flight.run_root,
                 "worktree": flight.worktree}[where]
    with pytest.raises(ValueError):
        ce.prepare_scratch_flight(flight, protected / "scratch")


def test_a_flight_without_captured_markers_cannot_seed_a_scratch_root(tmp_path):
    flight = make_flight(tmp_path)
    bare = ce.Flight(name=flight.name, run_root=flight.run_root, artifact_root=flight.artifact_root,
                     revision=flight.revision, config_path=flight.config_path,
                     worktree=flight.worktree)
    with pytest.raises(ValueError, match="captured markers"):
        ce.prepare_scratch_flight(bare, tmp_path / "scratch")


def test_the_marker_tripwire_detects_a_changed_marker_and_only_a_marker(tmp_path):
    flight = make_flight(tmp_path)
    before = ce.fingerprint_markers(flight)
    (flight.run_root / "heartbeat.json").write_text("{}", encoding="utf-8")   # a live flight's own
    ce.assert_markers_unchanged(flight, before)
    (flight.run_root / ce.MARKERS[0]).write_bytes(b"rewritten")
    with pytest.raises(ValueError, match=ce.MARKERS[0]):
        ce.assert_markers_unchanged(flight, before)


def test_destinations_are_checked_from_run_roots_before_any_flight_file_is_read(tmp_path):
    flight = make_flight(tmp_path)
    for protected in (flight.run_root, flight.artifact_root):
        with pytest.raises(ValueError):
            ce.refuse_destinations_near_flights([flight.run_root], [protected / "out"])
    ce.refuse_destinations_near_flights([flight.run_root], [tmp_path / "out", tmp_path / "scratch"])


# ---------------------------------------------------------------------------
# verifying the frozen inputs
# ---------------------------------------------------------------------------

def test_the_flight_is_verified_on_the_scratch_root_without_the_runtime_closure_check(
        tmp_path, monkeypatch):
    """The routes verifier would refuse this evaluator (it imports modules outside the frozen
    closure), so the closure check is skipped -- but the frozen worktree bytes, the archived
    bytes and every external source are still re-hashed. It runs on the SCRATCH run root (the
    captured markers plus the hard-linked frozen archive) because the verifier reads the markers
    by name and a live marker may be replaced by a restarting supervisor."""
    flight = make_flight(tmp_path, archive=True)
    scratch = ce.prepare_scratch_flight(flight, tmp_path / "scratch")
    seen = {}

    def resolve(repository_root, **kwargs):
        seen["resolve"] = (Path(repository_root), kwargs)
        return types.SimpleNamespace(run_root=kwargs["run_root"])

    def verify_source(context, **kwargs):
        seen["source"] = (context, kwargs)
        return {"verified": True}

    def verify_external(context, **kwargs):
        seen["external"] = (context, kwargs)
        return {"verified": True, "files": 3, "external_sha256": "e" * 64}

    monkeypatch.setattr(spec, "resolve_context", resolve)
    monkeypatch.setattr(spec, "verify_source_snapshot", verify_source)
    monkeypatch.setattr(spec, "verify_external_sources", verify_external)
    record = ce.verify_flight(flight, scratch)
    root, kwargs = seen["resolve"]
    assert root == flight.worktree
    assert kwargs["run_root"] == scratch.run_root and kwargs["ensure"] is False
    assert kwargs["config_path"] == flight.config_path
    assert seen["source"][1]["check_runtime_closure"] is False
    assert seen["external"][0].run_root == scratch.run_root
    assert record["source"]["verified"] and record["external"]["verified"]


def test_the_routes_source_verifier_passes_the_closure_switch_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(spec.nf_spec, "verify_source_snapshot",
                        lambda context, **kwargs: seen.update(kwargs) or {"verified": True})
    monkeypatch.setattr(spec, "_nf_context", lambda context: "adapted")
    spec.verify_source_snapshot(object(), label="x")
    assert seen == {"label": "x", "check_runtime_closure": None}, "the default is unchanged"
    spec.verify_source_snapshot(object(), label="x", check_runtime_closure=False)
    assert seen["check_runtime_closure"] is False


def test_the_evaluator_reads_no_lock_owner_heartbeat_or_progress_file_of_a_flight():
    """Review finding: ``lock_state`` is not purely observational. It opens ``campaign.owner.json``
    (which a starting supervisor replaces atomically; a reader's handle can make that replace fail
    on Windows) and briefly takes the trainer's exclusive lock when it is free (a concurrent
    supervisor's non-blocking acquire can then fail). Liveness no longer influences any decision,
    so the probe is gone, and so is ``launch_health``'s reading of heartbeat/progress files."""
    import inspect
    source = inspect.getsource(ce)
    for forbidden in ("lock_state", "launch_health", "flight_liveness", "heartbeat.json",
                      "progress.json", "campaign.owner", "LOCK_FILE"):
        assert forbidden not in source, forbidden
    assert not hasattr(ce, "flight_liveness")


# ---------------------------------------------------------------------------
# opening the services
# ---------------------------------------------------------------------------

def test_the_services_are_opened_on_cpu_against_the_scratch_root_and_cache(tmp_path,
                                                                          monkeypatch):
    import torch
    from smallAntibodyGen.experiments import her2_routes_services as services_module
    flight = make_flight(tmp_path)
    scratch = ce.prepare_scratch_flight(flight, tmp_path / "scratch")
    order = []

    class Recorder:
        def __init__(self, context, *, device, source_cache):
            order.append(("construct", context, device, source_cache))

    def resolve(repository_root, **kwargs):
        order.append(("resolve", Path(repository_root), kwargs))
        return types.SimpleNamespace(run_root=kwargs["run_root"])

    monkeypatch.setattr(spec, "resolve_context", resolve)
    monkeypatch.setattr(services_module, "FlightServices", Recorder)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: order.append(("threads", n)))
    ce.open_services(flight, scratch, threads=2)
    kinds = [entry[0] for entry in order]
    assert kinds == ["resolve", "construct", "threads"], (
        "the thread count is set AFTER construction, which sets runtime.cpu_threads itself")
    _, root, kwargs = order[0]
    assert root == flight.worktree and kwargs["run_root"] == scratch.run_root
    assert kwargs["artifact_root"] == flight.artifact_root and kwargs["ensure"] is True
    assert order[1][2:] == ("cpu", scratch.cache)
    assert order[2] == ("threads", 2)


# ---------------------------------------------------------------------------
# extracting the panel
# ---------------------------------------------------------------------------

class PanelServices:
    """The slice of ``FlightServices`` panel extraction touches, recording its order."""

    def __init__(self, tmp_path, rows, *, development=None):
        self.calls = []
        self.rows = rows
        # No "confirmation" block, like the real DPLM training flight's config: the analysis
        # radius is the evaluator's own declared constant, never read from a flight config
        # (reproduced against the real flights: KeyError: 'confirmation').
        self.config = {"her2": {"assay": "alphabind_her2:affinity:HER2"}}
        run_root = tmp_path / "scratch_run"
        run_root.mkdir(parents=True, exist_ok=True)
        paths.write_json(run_root / spec.SPLIT_FREEZE, {"record_kind": "hard_split_freeze"})
        self.context = types.SimpleNamespace(path=lambda *parts: run_root.joinpath(*parts))
        self._development = tuple(development if development is not None
                                  else [row["window"] for row in rows])

    def freeze_hard_split(self, *, persist=True):
        self.calls.append(("freeze", persist))

    def split(self):
        self.calls.append(("split",))
        return types.SimpleNamespace(development=self._development)

    def development_panel_outcomes(self):
        self.calls.append(("outcomes",))
        return [dict(row) for row in self.rows]

    def neighbourhood_labels(self, windows, *, radius=None):
        self.calls.append(("labels", tuple(windows)))
        return [f"n{index // 2}" for index in range(len(windows))]

    def scaffold(self):
        return types.SimpleNamespace(heavy="H" * 5, light="L" * 3)


def test_the_panel_is_extracted_without_persisting_the_split_and_is_digest_bound(tmp_path):
    rows = panel_rows(6)
    services = PanelServices(tmp_path, rows)
    panel = ce.extract_panel(services, flight_name="t1")
    assert services.calls[0] == ("freeze", False), (
        "persist=False first: the default would rewrite the flight's split record")
    assert ("freeze", True) not in services.calls
    assert not [call for call in services.calls if call[0] == "labels"], (
        "the repository's greedy-cover labels split near pairs; the evaluator computes "
        "connected components itself")
    assert [row["window"] for row in panel["rows"]] == sorted(row["window"] for row in rows)
    assert len(panel["neighbourhoods"]) == 6 and panel["flight"] == "t1"
    assert panel["grouping"]["method"] == "connected_components"
    assert panel["grouping"]["radius"] == 2 and panel["grouping"]["rows"] == 6
    assert panel["context"]["grouping"] == "connected_components"
    assert panel["context"]["split_sha256"] == paths.digest_document(
        {"record_kind": "hard_split_freeze"})
    assert panel["context"]["assay"] == "alphabind_her2:affinity:HER2"
    assert panel["digest"] == ce.panel_digest(panel["rows"], panel["neighbourhoods"],
                                              context=panel["context"])
    assert panel["reserved_opened"] is False


def test_the_neighbourhood_radius_is_the_evaluators_constant_not_a_flight_setting(tmp_path):
    """The two real flights disagree on whether their config even HAS a ``confirmation`` block
    (the bakeoff's does, radius 2; the DPLM training phase's does not), so a flight-config radius
    would either crash or quietly give the two flights different groupings -- and so different
    panels. The radius is part of the declared statistical definition."""
    assert ce.NEIGHBOURHOOD_RADIUS == 2
    services = PanelServices(tmp_path, panel_rows(6))
    services.config["confirmation"] = {"neighbourhood_radius": 5}
    panel = ce.extract_panel(services, flight_name="t1")
    assert panel["context"]["neighbourhood_radius"] == 2
    assert panel["grouping"]["radius"] == 2


def test_an_incomplete_panel_is_refused_at_extraction(tmp_path):
    rows = panel_rows(6)
    services = PanelServices(tmp_path, rows[:-1], development=[row["window"] for row in rows])
    with pytest.raises(ValueError, match="missing from the panel"):
        ce.extract_panel(services, flight_name="t1")


def test_panels_from_two_flights_must_be_identical_not_merely_similar(tmp_path):
    first = ce.extract_panel(PanelServices(tmp_path / "a", panel_rows(6)), flight_name="t1")
    same = ce.extract_panel(PanelServices(tmp_path / "b", panel_rows(6)), flight_name="dplm")
    ce.require_same_panel([first, same])
    moved = panel_rows(6)
    moved[3]["value"] += 0.5
    other = ce.extract_panel(PanelServices(tmp_path / "c", moved), flight_name="dplm")
    with pytest.raises(ValueError, match="different panel"):
        ce.require_same_panel([first, other])
