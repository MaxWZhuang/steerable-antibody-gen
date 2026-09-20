"""The completed-report rebuild helper: verify, reuse, compare, and only then record.

This tool exists because the frozen renderer has a known re-render defect that
cannot be fixed without invalidating a completed audit's frozen source identity.
The audit's original production of that report succeeded; the later re-render is
what failed.
So the tests here are about the things a "rebuild" could do wrong instead: rescue
a report whose completion marker is absent, re-render a figure, accept a figure
whose bytes changed, or mark the stage verified before the comparison ran.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallAntibodyGen.experiments import her2_support as support
from smallAntibodyGen.experiments import her2_support_paths as paths


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/rebuild_her2_support_report.py"
spec = importlib.util.spec_from_file_location("rebuild_her2_support_report", SCRIPT)
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


def fake_context(tmp_path):
    """A context with the two attributes the pure helpers use: config and run paths."""
    run = paths.RunPaths.create(tmp_path, tmp_path / "outputs/run", logical="outputs/run")
    return SimpleNamespace(
        repository_root=tmp_path, run=run,
        config={"audit_id": "her2_support_audit_20260919",
                "evidence_root": "reference/evidence/her2-support-audit-2026-09-19",
                "published_report": "reference/her2-support-audit.md",
                "publication": {"root": "reference",
                                "data_directory": "evidence/audit/published",
                                "figure_directory": "figures",
                                "manifest": "evidence/audit/published/manifest.json"}})


def publish_figure(context, name, payload=b"PNGBYTES"):
    """Write one figure into the run directory, the ledger and the published tree."""
    logical = f"report/figures/{name}"
    target = context.run.path(logical)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    paths.record_completion(context.run.run_root, context.run.logical(logical), target,
                            kind="figure", scientific_digest=paths.sha256_bytes(payload))
    destination = (context.repository_root / context.config["publication"]["root"]
                   / f"{context.config['publication']['figure_directory']}/{name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {f"{context.config['publication']['figure_directory']}/{name}":
            paths.sha256_bytes(payload)}


# ---------------------------------------------------------------------------
# figures are reused, never re-rendered
# ---------------------------------------------------------------------------

def test_verified_figure_bytes_are_reused_with_their_recorded_captions(tmp_path):
    context = fake_context(tmp_path)
    files = {}
    for name in rebuild.FIGURE_CAPTIONS:
        files.update(publish_figure(context, name))
    figures = rebuild.figure_entries(context, {"files": files})
    assert set(figures) == set(rebuild.FIGURE_CAPTIONS)
    for name, entry in figures.items():
        assert entry["written"] is True and entry["caption"] == rebuild.FIGURE_CAPTIONS[name]
        assert entry["logical"] == f"report/figures/{name}"


def test_a_figure_the_tool_has_no_caption_for_is_refused(tmp_path):
    context = fake_context(tmp_path)
    files = publish_figure(context, "her2-support-forward-kl.png")
    files.update(publish_figure(context, "her2-support-new-thing.png"))
    with pytest.raises(ValueError, match="no recorded caption"):
        rebuild.figure_entries(context, {"files": files})


def test_a_figure_that_changed_since_it_completed_is_refused(tmp_path):
    context = fake_context(tmp_path)
    files = {}
    for name in rebuild.FIGURE_CAPTIONS:
        files.update(publish_figure(context, name))
    target = context.run.path("report/figures/her2-support-ches.png")
    target.write_bytes(b"EDITED")
    with pytest.raises(ValueError, match="changed after it was completed"):
        rebuild.figure_entries(context, {"files": files})


def test_a_figure_that_was_never_recorded_complete_is_refused(tmp_path):
    context = fake_context(tmp_path)
    files = {}
    for name in rebuild.FIGURE_CAPTIONS:
        files.update(publish_figure(context, name))
    manifest = paths.read_completion_manifest(context.run.run_root)
    manifest["artifacts"].pop("outputs/run/report/figures/her2-support-ches.png")
    paths.write_json(paths.completion_manifest_path(context.run.run_root), manifest)
    with pytest.raises(ValueError, match="never recorded it"):
        rebuild.figure_entries(context, {"files": files})


def test_a_published_copy_that_differs_from_the_original_is_refused(tmp_path):
    context = fake_context(tmp_path)
    files = {}
    for name in rebuild.FIGURE_CAPTIONS:
        files.update(publish_figure(context, name))
    destination = (context.repository_root / "reference/figures/her2-support-forward-kl.png")
    destination.write_bytes(b"OTHERBYTES")
    with pytest.raises(ValueError, match="differs from the run-directory figure"):
        rebuild.figure_entries(context, {"files": files})


def test_a_publication_that_names_no_figure_is_refused(tmp_path):
    context = fake_context(tmp_path)
    with pytest.raises(ValueError, match="names no figures"):
        rebuild.figure_entries(context, {"files": {}})


# ---------------------------------------------------------------------------
# the byte comparison
# ---------------------------------------------------------------------------

def test_the_comparison_is_byte_for_byte_and_names_the_first_difference():
    same = rebuild._compare("hello\n", b"hello\n", label="x")
    assert same["matches"] is True and same["first_difference"] is None
    assert same["rendered_sha256"] == same["recorded_sha256"]
    changed = rebuild._compare("hellp\n", b"hello\n", label="x")
    assert changed["matches"] is False and changed["first_difference"] == 4
    longer = rebuild._compare("hello!\n", b"hello\n", label="x")
    assert longer["matches"] is False and longer["first_difference"] == 5


def test_a_trailing_newline_difference_is_a_difference():
    assert rebuild._compare("a\n", b"a", label="x")["matches"] is False


# ---------------------------------------------------------------------------
# the operational status is only touched after a pass
# ---------------------------------------------------------------------------

def record(matches=True):
    return {"comparisons": {"run_directory": {"matches": matches},
                            "published": {"matches": matches}},
            "figures": {"her2-support-ches.png": {}},
            "tool": {"sha256": "a" * 64}}


def test_the_original_failed_stage_record_is_preserved_when_the_status_is_updated(tmp_path):
    context = fake_context(tmp_path)
    paths.write_json(context.run.path(rebuild.PROGRESS_RECORD),
                     {"stage": "report", "status": "failed", "started_at": "2026-09-19T01:00:00",
                      "error": "ValueError: Format .rerender is not supported"})
    assert rebuild._mark_report_verified(context, record()) == "written"
    original = paths.read_json(context.run.path(rebuild.PROGRESS_ORIGINAL))
    assert original["status"] == "failed" and "rerender" in original["error"]
    updated = paths.read_json(context.run.path(rebuild.PROGRESS_RECORD))
    assert updated["status"] == "completed" and updated["error"] is None
    assert updated["verified_by"] == "scripts/rebuild_her2_support_report.py"
    assert updated["started_at"] == "2026-09-19T01:00:00"


def test_a_second_verification_run_changes_nothing_on_disk(tmp_path):
    context = fake_context(tmp_path)
    paths.write_json(context.run.path(rebuild.PROGRESS_RECORD),
                     {"stage": "report", "status": "failed", "started_at": "t"})
    rebuild._mark_report_verified(context, record())
    first = context.run.path(rebuild.PROGRESS_RECORD).read_bytes()
    assert rebuild._mark_report_verified(context, record()) == "unchanged"
    assert context.run.path(rebuild.PROGRESS_RECORD).read_bytes() == first


def test_the_preserved_original_is_written_once_and_never_overwritten(tmp_path):
    context = fake_context(tmp_path)
    paths.write_json(context.run.path(rebuild.PROGRESS_RECORD),
                     {"stage": "report", "status": "failed", "started_at": "t"})
    rebuild._mark_report_verified(context, record())
    rebuild._mark_report_verified(context, record())
    original = paths.read_json(context.run.path(rebuild.PROGRESS_ORIGINAL))
    assert original["status"] == "failed", (
        "the preserved record must stay the ORIGINAL failure, not the status this tool wrote")


# ---------------------------------------------------------------------------
# what the tool structurally does not do
# ---------------------------------------------------------------------------

def test_importing_the_tool_rebinds_nothing_in_the_frozen_audit_module():
    """Runtime invariance, not a grep for the word 'monkeypatch'.

    The claim is that this tool imports the audit's pure functions and leaves the
    module exactly as it found it. A source-text ban cannot check that -- it fails
    on a docstring that merely says so, and it passes on a rebinding written any
    other way. So the module's callables are snapshotted by identity, the script is
    executed a second time into a fresh module object, and every binding must still
    be the same object.
    """
    before = {name: id(value) for name, value in vars(support).items() if callable(value)}
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    after = {name: id(value) for name, value in vars(support).items() if callable(value)}
    assert before == after, "executing the tool rebound something in the frozen audit module"
    assert fresh.support is support, "the audit module is imported, not copied or shadowed"


def test_the_tool_never_calls_the_defective_figure_renderer(tmp_path, monkeypatch):
    """The defect it routes around is never exercised, checked by calling it."""
    calls = []
    monkeypatch.setattr(support, "save_figure",
                        lambda *args, **kwargs: calls.append("save_figure"))
    monkeypatch.setattr(support, "render_figures",
                        lambda *args, **kwargs: calls.append("render_figures"))
    context = fake_context(tmp_path)
    files = {}
    for name in rebuild.FIGURE_CAPTIONS:
        files.update(publish_figure(context, name))
    figures = rebuild.figure_entries(context, {"files": files})
    rebuild._mark_report_verified(context, record())
    assert calls == [], "a rebuild reuses figure bytes and renders no figure"
    assert all(entry["written"] and entry["sha256"] for entry in figures.values())


def test_the_markdown_comes_from_the_audits_own_pure_function():
    assert rebuild.support.render_report is support.render_report


# ---------------------------------------------------------------------------
# every published file, not only the report and the figures
# ---------------------------------------------------------------------------

def publish_tree(context, *, extra=None):
    """A published tree holding figures, numeric deliverables and a manifest."""
    files = {}
    for name in rebuild.FIGURE_CAPTIONS:
        files.update(publish_figure(context, name))
    root = context.repository_root / context.config["publication"]["root"]
    for name, document in {"decision.json": {"outcome": "escalate"},
                           "ches.json": {"blocks": 54},
                           **(extra or {})}.items():
        logical = f"{context.config['publication']['data_directory']}/{name}"
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        paths.write_json(target, document)
        files[logical] = paths.sha256_file(target)
    report_logical = "her2-support-audit.md"
    (root / report_logical).write_text("# report\n", encoding="utf-8", newline="\n")
    files[report_logical] = paths.sha256_file(root / report_logical)
    manifest_path = root / context.config["publication"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.write_json(manifest_path, {"record_kind": "publication_manifest",
                                     "root": context.config["publication"]["root"],
                                     "files": files})
    return {"record_kind": "publication_manifest", "files": files}, report_logical


def test_every_published_file_is_hashed_against_the_publication_manifest(tmp_path):
    context = fake_context(tmp_path)
    manifest, report_logical = publish_tree(context)
    verified = rebuild.published_files(context, manifest, report_key=report_logical)
    assert verified["file_count"] == len(manifest["files"]) == 6
    assert all(entry["matches"] for entry in verified["files"].values())
    assert verified["files"][report_logical]["is_report"] is True


def test_a_corrupt_published_numeric_file_is_refused_even_though_the_report_still_matches(
        tmp_path):
    """The gap: report and figures verified, a published JSON beside them corrupt."""
    context = fake_context(tmp_path)
    manifest, report_logical = publish_tree(context)
    corrupt = (context.repository_root / context.config["publication"]["root"]
               / f"{context.config['publication']['data_directory']}/ches.json")
    corrupt.write_text('{"blocks": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="do not match their publication manifest"):
        rebuild.published_files(context, manifest, report_key=report_logical)


def test_a_published_file_that_is_gone_is_refused(tmp_path):
    context = fake_context(tmp_path)
    manifest, report_logical = publish_tree(context)
    (context.repository_root / "reference/figures/her2-support-ches.png").unlink()
    with pytest.raises(ValueError, match="is absent"):
        rebuild.published_files(context, manifest, report_key=report_logical)


def test_a_manifest_path_that_escapes_the_published_tree_is_refused(tmp_path):
    context = fake_context(tmp_path)
    manifest, report_logical = publish_tree(context)
    manifest = dict(manifest, files=dict(manifest["files"], **{"../outside.json": "0" * 64}))
    with pytest.raises(ValueError, match="outside the published tree"):
        rebuild.published_files(context, manifest, report_key=report_logical)


def test_the_completion_ledger_verification_is_required_and_reports_the_original_counts(
        tmp_path, monkeypatch):
    context = fake_context(tmp_path)
    monkeypatch.setattr(support, "verify_outputs", lambda ctx: {
        "shards_checked": 216, "expected_output_count": 462, "problems": [],
        "immutable": True, "completion_manifest": {"artifacts_checked": 462}})
    block = rebuild.completion_ledger(context)
    assert block["shards_checked"] == 216 and block["artifacts_checked"] == 462
    monkeypatch.setattr(support, "verify_outputs", lambda ctx: {
        "shards_checked": 215, "expected_output_count": 462,
        "problems": [{"artifact": "scores/x.npz", "problem": "absent"}],
        "immutable": False, "completion_manifest": {"artifacts_checked": 461}})
    with pytest.raises(ValueError, match="does not verify against its own completion manifest"):
        rebuild.completion_ledger(context)


def test_the_compatibility_record_is_never_appended_to_the_original_completion_ledger(tmp_path):
    """The saved authority 462 completed artifacts are checked against is not written to."""
    context = fake_context(tmp_path)
    paths.write_json(context.run.path(rebuild.PROGRESS_RECORD),
                     {"stage": "report", "status": "failed", "started_at": "t"})
    before = paths.read_completion_manifest(context.run.run_root)
    support.require_new_or_identical(context.run.path(rebuild.RUN_RECORD), record(),
                                     what="the report compatibility record")
    rebuild._mark_report_verified(context, record())
    after = paths.read_completion_manifest(context.run.run_root)
    assert (after.get("artifacts") or {}) == (before.get("artifacts") or {}), (
        "a tool that verifies a finished measurement does not register itself in its ledger")
    assert context.run.path(rebuild.RUN_RECORD).is_file(), (
        "the standalone compatibility evidence is still written")


def test_check_only_returns_before_anything_is_written():
    """``--check-only`` verifies and writes nothing; checked on the syntax, not the prose.

    Every call that can write is required to sit *after* the ``check_only`` return,
    so the flag cannot be satisfied by a docstring while a writer runs above it.
    """
    import ast
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "rebuild")
    guard = next(node for node in ast.walk(function)
                 if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                 and node.test.id == "check_only"
                 and any(isinstance(body, ast.Return) for body in node.body))
    writers = {"require_new_or_identical", "require_new_or_identical_text", "write_json",
               "write_text", "_mark_report_verified"}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None))
        if name in writers:
            assert node.lineno > guard.lineno, (
                f"{name} is called at line {node.lineno}, before the --check-only return at "
                f"{guard.lineno}")


def test_the_tool_requires_the_completion_marker_and_the_bound_summaries(tmp_path):
    context = fake_context(tmp_path)
    with pytest.raises(ValueError, match="does not produce one"):
        rebuild.completed_documents(context)


def test_the_tool_refuses_a_run_whose_completion_marker_is_absent(tmp_path, monkeypatch):
    context = fake_context(tmp_path)
    monkeypatch.setattr(support, "completed_stage", lambda *args, **kwargs: {"inventory": {}})
    with pytest.raises(ValueError, match="no completed report to verify"):
        rebuild.completed_documents(context)


def test_the_legacy_defect_is_documented_rather_than_fixed():
    assert rebuild.LEGACY_DEFECT["function"] == "save_figure"
    assert "AUDIT_SOURCE_FILES" in rebuild.LEGACY_DEFECT["not_fixed_because"]
    frozen = Path(support.__file__).read_text(encoding="utf-8")
    assert "def save_figure" in frozen, "the frozen renderer is not edited by this change"
    assert ".rerender" in frozen, "its defect is still there, and still documented"


def test_the_captions_match_the_frozen_renderer():
    """A caption drifting from the renderer's would change the Markdown bytes."""
    frozen = Path(support.__file__).read_text(encoding="utf-8")
    for caption in rebuild.FIGURE_CAPTIONS.values():
        head = caption.split(";")[0].split(",")[0][:40]
        assert head in " ".join(frozen.split()), (
            f"the recorded caption {head!r} is not the one render_figures attaches")
