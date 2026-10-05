"""The common evaluation, slice D: which subjects exist, the summary, and the output directory.

The summary is the only place a number is turned into a claim, so these tests pin three things:
the contrasts are the DECLARED ones (not every pair, which would be a multiple-comparisons
fishing expedition), the document says what it does NOT establish, and the same inputs give
byte-identical outputs (the bootstrap is seeded).
"""
from __future__ import annotations

import copy
import csv
import io
import json

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_common_eval as ce
from smallAntibodyGen.experiments import her2_support_paths as paths

# ---------------------------------------------------------------------------
# which subjects exist
# ---------------------------------------------------------------------------


def make_bakeoff(tmp_path, layout):
    """``layout`` maps route -> (status, [filenames])."""
    run_root = tmp_path / "flight"
    for route, (status, names) in layout.items():
        directory = run_root / "bakeoff" / route
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(b"w")
        (directory / "progress.json").write_text("{not json", encoding="utf-8")   # never opened
        if status == "completed":
            (directory / "run.json").write_text("{}", encoding="utf-8")   # written LAST by train()
    return ce.Flight(name="t1", run_root=run_root, artifact_root=tmp_path, revision="r",
                     config_path=tmp_path / "c.json", worktree=tmp_path)


LAYOUT = {"ablang2_masked_editor": ("completed", ["best.pt", "last.pt", "step5000.pt",
                                                   "step10000.pt", "run.json"]),
          "piggen_causal_infiller": ("completed", ["best.pt", "last.pt"]),
          "dwjs_energy": ("completed", ["best.pt"])}


def test_every_supported_route_gets_its_checkpoints_and_a_zero_shot_comparator(tmp_path):
    flight = make_bakeoff(tmp_path, LAYOUT)
    plan = ce.plan_subjects(flight, routes=None, labels=None)
    keys = [subject.key for subject in plan["subjects"]]
    assert keys == sorted(keys)
    assert "t1__ablang2_masked_editor__step5000" in keys and "t1__ablang2_masked_editor__step10000" in keys
    assert {s.label for s in plan["subjects"] if s.route == "piggen_causal_infiller"} == {
        "best", "last", "zero_shot"}
    assert not any(s.route == "dwjs_energy" for s in plan["subjects"]), (
        "the energy route has no bin; it is skipped by discovery, not scored")
    assert plan["skipped"]["dwjs_energy"].startswith("not supported")


def test_a_running_route_is_planned_in_full_because_checkpoints_are_pinned(tmp_path):
    """Liveness no longer changes the plan: a hard-link snapshot makes a file the trainer is still
    replacing as safe to score as a finished one."""
    flight = make_bakeoff(tmp_path, {"dplm_150m_masked": ("running", ["best.pt", "last.pt",
                                                                       "step5000.pt"])})
    plan = ce.plan_subjects(flight, routes=None, labels=None)
    assert {s.label for s in plan["subjects"]} == {"best", "last", "step5000", "zero_shot"}
    assert plan["skipped"] == {}


def test_planning_opens_no_file_of_the_flight(tmp_path, monkeypatch):
    from smallAntibodyGen.tests.test_her2_routes_common_eval_scoring import MUTABLE, forbid_opening
    flight = make_bakeoff(tmp_path, {"dplm_150m_masked": ("running", ["best.pt", "last.pt"])})
    (flight.run_root / "heartbeat.json").write_text("{not json", encoding="utf-8")
    forbid_opening(monkeypatch, names=MUTABLE + ("best.pt", "last.pt", "run.json"))
    ce.plan_subjects(flight, routes=None, labels=None)       # directory listings and stat only


def test_an_explicit_request_for_a_missing_label_or_unsupported_route_is_refused(tmp_path):
    flight = make_bakeoff(tmp_path, LAYOUT)
    with pytest.raises(ValueError, match="step7777"):
        ce.plan_subjects(flight, routes=None, labels=["step7777"])
    with pytest.raises(ValueError, match="not supported"):
        ce.plan_subjects(flight, routes=["dwjs_energy"], labels=None)
    with pytest.raises(ValueError, match="dplm_650m_lora_masked"):
        ce.plan_subjects(flight, routes=["dplm_650m_lora_masked"], labels=None)


def test_the_route_filter_restricts_the_plan(tmp_path):
    flight = make_bakeoff(tmp_path, LAYOUT)
    plan = ce.plan_subjects(flight, routes=["piggen_causal_infiller"], labels=["best"])
    assert [s.key for s in plan["subjects"]] == ["t1__piggen_causal_infiller__best"]


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------

def clustered_rows(clusters=6):
    """Twelve windows in six pairs; pairs are far apart, so the components are exactly the pairs."""
    rows = []
    for index, letter in enumerate("DEFGHI"[:clusters]):
        for offset, tail in enumerate("AC"):
            rows.append({"window": letter * 9 + tail, "value": 5.0 + 0.25 * (2 * index + offset)})
    return rows


ROWS = ce.validate_panel(clustered_rows(), development_windows=[r["window"] for r in clustered_rows()])
LABELS = ce.neighbourhood_components([row["window"] for row in ROWS], radius=2)
CONTEXT = {"assay": "a", "split_sha256": "s" * 64, "scaffold_sha256": "c" * 64, "sites": 10,
           "grouping": "connected_components", "neighbourhood_radius": 2}
DIGEST = ce.panel_digest(ROWS, LABELS, context=CONTEXT)


def panel_document(flight="t1", **overrides):
    """A fresh document every time: the tamper tests mutate rows, labels and context in place,
    so sharing the module-level objects would let one test corrupt every later one."""
    document = {"schema": ce.SCHEMA, "record_kind": "panel", "flight": flight, "digest": DIGEST,
                "rows": copy.deepcopy(ROWS), "neighbourhoods": list(LABELS),
                "context": copy.deepcopy(CONTEXT),
                "grouping": ce.grouping_record(LABELS, radius=2),
                "reserved_opened": False, "evaluator": {"schema": ce.SCHEMA}}
    document.update(overrides)
    return document


#: A complete evaluator record, as ``evaluator_identity`` writes one. Summarising demands every
#: field: a subject without them cannot establish that it came from the same implementation.
EVALUATOR = {"schema": ce.SCHEMA, "sources": {"her2_routes_common_eval.py": "a" * 64},
             "package_sha256": "b" * 64, "package_files": 3,
             "dependencies": {"transformers": "4.40.0", "einops": None},
             "python": "3.11.0", "numpy": "2.0.0"}


def subject_document(flight, route, label, high, null, *, step=100, **overrides):
    document = {
        "schema": ce.SCHEMA, "record_kind": "subject", "key": f"{flight}__{route}__{label}",
        "flight": flight, "route": route, "label": label,
        "kind": "zero_shot" if label == "zero_shot" else "checkpoint",
        "checkpoint": None if label == "zero_shot" else {
            "original_path": f"/x/{label}.pt", "sha256": "a" * 64, "bytes": 1, "step": step,
            "training_identity": {"route": route, "arm": "full", "seed": 1}, "policy": {}},
        "panel_digest": DIGEST, "rows": 12, "high": list(map(float, high)),
        "null": list(map(float, null)), "cuda_initialized": False,
        "evaluator": copy.deepcopy(EVALUATOR),
        "flight_markers": {"hard_split_freeze.json": "1" * 64}, "seconds": 1.0}
    document.update(overrides)
    return document


def write_run(out, *, subjects=None, panels=None):
    out.mkdir(parents=True, exist_ok=True)
    for panel in (panels if panels is not None else [panel_document()]):
        paths.write_json(out / f"panel_{panel['flight']}.json", panel)
    for document in (subjects if subjects is not None else standard_subjects()):
        paths.write_json(out / f"subject_{document['key']}.json", document)


def standard_subjects():
    rng = np.random.default_rng(4)
    truth = np.array([row["value"] for row in ROWS])
    strong = truth + rng.normal(scale=0.4, size=12)
    medium = truth + rng.normal(scale=1.5, size=12)
    weak = rng.normal(size=12)
    zero = np.zeros(12)
    return [
        subject_document("t1", "ablang2_masked_editor", "best", strong, strong - 0.5),
        subject_document("t1", "ablang2_masked_editor", "last", medium, medium - 0.5),
        subject_document("t1", "ablang2_masked_editor", "zero_shot", weak, weak),
        subject_document("t1", "piggen_causal_infiller", "best", medium, medium - 1.0),
        subject_document("t1", "piggen_causal_infiller", "zero_shot", weak, weak),
        subject_document("dplm", "dplm_150m_masked", "best", medium, medium - truth * 0.3),
        subject_document("dplm", "dplm_150m_masked", "zero_shot", zero - 1.0, zero - 1.0),
    ]


def test_the_summary_reports_every_subject_under_all_three_selectors(tmp_path):
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    summary = ce.summarize(tmp_path / "out", draws=200)
    assert summary["primary_selector"] == "high" and summary["selectors"] == ["high", "null", "ratio"]
    assert summary["panel"]["digest"] == DIGEST and summary["panel"]["rows"] == 12
    assert summary["panel"]["neighbourhoods"] == 6 and summary["panel"]["reserved_opened"] is False
    by_key = {entry["key"]: entry for entry in summary["subjects"]}
    assert len(by_key) == 7
    best = by_key["t1__ablang2_masked_editor__best"]
    assert set(best["rho"]) == {"high", "null", "ratio"}
    assert best["rho"]["high"]["rho"] > 0.8 and best["checkpoint"]["step"] == 100
    zero = by_key["dplm__dplm_150m_masked__zero_shot"]
    assert zero["rho"]["high"] == {"rho": None, "interval": None, "usable_draws": 0}
    assert zero["rho"]["ratio"]["rho"] is None, "a constant zero-shot ratio is undefined, not 0"
    json.dumps(summary, allow_nan=False)


def test_only_the_declared_contrasts_are_computed_on_the_primary_selector(tmp_path):
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    summary = ce.summarize(tmp_path / "out", draws=200)
    names = [(c["a"], c["b"]) for c in summary["contrasts"]]
    assert names == [
        ("dplm__dplm_150m_masked__best", "dplm__dplm_150m_masked__zero_shot"),
        ("t1__ablang2_masked_editor__best", "t1__ablang2_masked_editor__last"),
        ("t1__ablang2_masked_editor__best", "t1__ablang2_masked_editor__zero_shot"),
        ("t1__piggen_causal_infiller__best", "t1__piggen_causal_infiller__zero_shot"),
        ("dplm__dplm_150m_masked__best", "t1__ablang2_masked_editor__best"),
        ("dplm__dplm_150m_masked__best", "t1__piggen_causal_infiller__best"),
        ("t1__ablang2_masked_editor__best", "t1__piggen_causal_infiller__best"),
    ]
    assert all(c["selector"] == "high" for c in summary["contrasts"])
    undefined = next(c for c in summary["contrasts"]
                     if c["b"] == "dplm__dplm_150m_masked__zero_shot")
    assert undefined["delta"] is None and undefined["reason"]
    assert summary["multiplicity"].startswith("no correction")


def test_the_summary_states_what_it_does_not_establish(tmp_path):
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    summary = ce.summarize(tmp_path / "out", draws=100)
    text = " ".join(summary["does_not_establish"]).lower()
    for phrase in ("held-out", "seed", "selection", "generation", "neighbourhood"):
        assert phrase in text, phrase


def test_summarising_twice_gives_byte_identical_outputs(tmp_path):
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    ce.summarize(tmp_path / "out", draws=150)
    first = {name: (tmp_path / "out" / name).read_bytes()
             for name in ("summary.json", "per_row_scores.csv", "report.md")}
    ce.summarize(tmp_path / "out", draws=150)
    assert {name: (tmp_path / "out" / name).read_bytes() for name in first} == first


def test_the_per_row_csv_carries_every_subject_and_row_with_all_three_scores(tmp_path):
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    ce.summarize(tmp_path / "out", draws=100)
    text = (tmp_path / "out" / "per_row_scores.csv").read_text(encoding="utf-8")
    assert "\r" not in text
    records = list(csv.DictReader(io.StringIO(text)))
    assert len(records) == 7 * 12
    assert set(records[0]) == {"subject", "flight", "route", "label", "window", "outcome",
                               "neighbourhood", "high", "null", "ratio"}
    first = next(r for r in records if r["subject"] == "t1__ablang2_masked_editor__best")
    assert float(first["ratio"]) == pytest.approx(float(first["high"]) - float(first["null"]))


def test_the_report_names_the_panel_the_selectors_and_the_limits(tmp_path):
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    ce.summarize(tmp_path / "out", draws=100)
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert DIGEST[:12] in report
    assert "t1__ablang2_masked_editor__best" in report and "undefined" in report
    assert "development panel" in report and "not a held-out" in report.lower()


@pytest.mark.parametrize("mutate, message", [
    (lambda subjects, panels: subjects[0].update(panel_digest="f" * 64), "different panel"),
    (lambda subjects, panels: subjects[0].update(cuda_initialized=True), "CUDA"),
    (lambda subjects, panels: subjects[0].update(high=subjects[0]["high"][:-1]), "rows"),
    (lambda subjects, panels: subjects[0].update(schema="other/1"), "schema"),
    (lambda subjects, panels: subjects.clear(), "no subject"),
    (lambda subjects, panels: panels.clear(), "no panel"),
])
def test_inconsistent_evidence_is_refused_rather_than_summarised(tmp_path, mutate, message):
    subjects, panels = standard_subjects(), [panel_document("t1"), panel_document("dplm")]
    mutate(subjects, panels)
    write_run(tmp_path / "out", subjects=subjects, panels=panels)
    with pytest.raises(ValueError, match=message):
        ce.summarize(tmp_path / "out", draws=50)


def test_subjects_scored_by_different_evaluators_are_not_contrasted(tmp_path):
    """A resumed run after an edit leaves old scores beside new ones; a contrast across them
    would compare two implementations, not two models."""
    for field, value in (("package_sha256", "9" * 64),
                         ("dependencies", {"transformers": "4.41.0", "einops": None})):
        subjects = standard_subjects()
        subjects[3]["evaluator"][field] = value
        write_run(tmp_path / field, subjects=subjects,
                  panels=[panel_document("t1"), panel_document("dplm")])
        with pytest.raises(ValueError, match="different evaluators"):
            ce.summarize(tmp_path / field, draws=50)


def test_subjects_of_one_flight_scored_against_different_markers_are_refused(tmp_path):
    subjects = standard_subjects()
    subjects[1]["flight_markers"] = {"hard_split_freeze.json": "2" * 64}     # t1, ablang2 last
    write_run(tmp_path / "out", subjects=subjects,
              panels=[panel_document("t1"), panel_document("dplm")])
    with pytest.raises(ValueError, match="different flight identity markers"):
        ce.summarize(tmp_path / "out", draws=50)


@pytest.mark.parametrize("mutate", [
    lambda document: document["evaluator"].pop("package_sha256"),     # a three-module fingerprint
    lambda document: document["evaluator"].pop("dependencies"),
    lambda document: document["evaluator"].pop("sources"),
    lambda document: document.pop("evaluator"),
    lambda document: document.pop("flight_markers"),
    lambda document: document.update(flight_markers={}),
])
def test_a_subject_without_complete_provenance_is_refused_even_when_all_match(tmp_path, mutate):
    """Missing fields on EVERY subject compare equal to each other, which would accept a whole
    directory scored before the fingerprint existed. Provenance is required, not compared."""
    subjects = standard_subjects()
    for document in subjects:
        mutate(document)
    write_run(tmp_path / "out", subjects=subjects,
              panels=[panel_document("t1"), panel_document("dplm")])
    with pytest.raises(ValueError, match="provenance"):
        ce.summarize(tmp_path / "out", draws=50)


def test_summarising_holds_the_output_lock_and_creates_nothing_for_a_missing_directory(tmp_path):
    """Two summaries write the same fixed temp paths, and a score finishing between their reads
    would leave summary.json, the CSV and the report describing different subject sets."""
    write_run(tmp_path / "out", panels=[panel_document("t1"), panel_document("dplm")])
    with ce.exclusive_invocation(tmp_path / "out"):
        with pytest.raises(ValueError, match="another evaluator invocation"):
            ce.summarize(tmp_path / "out", draws=50)
    ce.summarize(tmp_path / "out", draws=50)                      # released again afterwards
    with pytest.raises(ValueError, match="not an output directory"):
        ce.summarize(tmp_path / "missing", draws=50)
    assert not (tmp_path / "missing").exists()


def test_two_flights_with_different_panels_are_refused(tmp_path):
    rows = [dict(row) for row in ROWS]
    rows[0]["value"] += 1.0
    other = panel_document("dplm", rows=rows,
                           digest=ce.panel_digest(rows, LABELS, context=CONTEXT))
    write_run(tmp_path / "out", panels=[panel_document("t1"), other])
    with pytest.raises(ValueError, match="different panel"):
        ce.summarize(tmp_path / "out", draws=50)


# ---------------------------------------------------------------------------
# a saved panel is integrity-checked, not trusted
# ---------------------------------------------------------------------------

def tampered(mutate):
    document = panel_document("t1")
    mutate(document)
    return document


@pytest.mark.parametrize("mutate, message", [
    # the reversed panel reviewers reproduced: rows moved, stored digest string untouched
    (lambda d: d.update(rows=list(reversed(d["rows"]))), "does not match its content"),
    (lambda d: d["rows"][0].update(value=d["rows"][0]["value"] + 1), "does not match its content"),
    (lambda d: d.update(neighbourhoods=list(reversed(d["neighbourhoods"]))),
     "does not match its content"),
    (lambda d: d["context"].update(assay="other"), "does not match its content"),
    (lambda d: d.update(digest="0" * 64), "does not match its content"),
    (lambda d: d.update(schema="other/1"), "schema"),
    (lambda d: d.pop("grouping"), "grouping"),
])
def test_a_tampered_panel_is_refused_even_when_its_digest_string_is_untouched(
        tmp_path, mutate, message):
    write_run(tmp_path / "out", panels=[tampered(mutate)])
    with pytest.raises(ValueError, match=message):
        ce.summarize(tmp_path / "out", draws=50)


def test_a_self_consistent_panel_with_labels_that_are_not_the_components_is_refused(tmp_path):
    """Recomputing the digest after editing the labels is not enough: the labels must also be
    the connected components of the windows, so a regrouping cannot be smuggled in."""
    document = panel_document("t1")
    document["neighbourhoods"] = [f"x{index // 3}" for index in range(12)]
    document["grouping"] = ce.grouping_record(document["neighbourhoods"], radius=2)
    document["digest"] = ce.panel_digest(document["rows"], document["neighbourhoods"],
                                         context=document["context"])
    write_run(tmp_path / "out", panels=[document])
    with pytest.raises(ValueError, match="connected components"):
        ce.summarize(tmp_path / "out", draws=50)


def test_the_score_command_refuses_a_tampered_saved_panel_too(tmp_path):
    from smallAntibodyGen.tests.test_her2_routes_common_eval_flights import make_flight
    flight = make_flight(tmp_path)
    out = tmp_path / "out"
    ce.ensure_output_directory(out, [flight])
    document = panel_document(flight.name)
    document["rows"] = list(reversed(document["rows"]))
    paths.write_json(out / f"panel_{flight.name}.json", document)
    with pytest.raises(ValueError, match="does not match its content"):
        ce.run_score([flight], out, tmp_path / "scratch", routes=None, labels=None,
                     allow_live=False, batch_size=2, threads=2)


# ---------------------------------------------------------------------------
# the output directory
# ---------------------------------------------------------------------------

def flight_in(tmp_path, name):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return ce.Flight(name=name, run_root=root / "runs" / "r" / "flight", artifact_root=root,
                     revision="r", config_path=root / "c.json", worktree=root / "w")


def test_a_new_output_directory_is_created_with_a_spec_and_an_existing_foreign_one_is_refused(
        tmp_path):
    out = tmp_path / "out"
    flights = [flight_in(tmp_path, "t1")]
    (flights[0].config_path).write_text("{}", encoding="utf-8")
    spec = ce.ensure_output_directory(out, flights)
    assert (out / "eval_spec.json").is_file() and spec["flights"]["t1"]["revision"] == "r"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "something.txt").write_text("not ours", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        ce.ensure_output_directory(foreign, flights)


def test_a_later_invocation_must_name_the_same_flights_it_started_with(tmp_path):
    out = tmp_path / "out"
    t1 = flight_in(tmp_path, "t1")
    t1.config_path.write_text("{}", encoding="utf-8")
    ce.ensure_output_directory(out, [t1])
    assert ce.ensure_output_directory(out, [t1])["flights"]["t1"]["revision"] == "r"
    moved = ce.Flight(name="t1", run_root=t1.run_root, artifact_root=t1.artifact_root,
                      revision="r2", config_path=t1.config_path, worktree=t1.worktree)
    with pytest.raises(ValueError, match="different"):
        ce.ensure_output_directory(out, [moved])
    t1.config_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="different"):
        ce.ensure_output_directory(out, [t1])
    t1.config_path.write_text("{}", encoding="utf-8")


def test_a_new_flight_may_be_added_but_a_recorded_one_is_never_rewritten(tmp_path):
    out = tmp_path / "out"
    t1, other = flight_in(tmp_path, "t1"), flight_in(tmp_path, "dplm")
    for flight in (t1, other):
        flight.config_path.write_text("{}", encoding="utf-8")
    ce.ensure_output_directory(out, [t1])
    both = ce.ensure_output_directory(out, [t1, other])
    assert set(both["flights"]) == {"t1", "dplm"}
    assert set(paths.read_json(out / "eval_spec.json")["flights"]) == {"t1", "dplm"}
