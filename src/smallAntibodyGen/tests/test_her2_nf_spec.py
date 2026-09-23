"""The source snapshot, the lineage refusals, and the readiness gates that must bite.

The freeze here is content-addressed rather than a commit, because this flight
is authorized to build and train and not to commit. That makes the *drift check*
load-bearing: if a frozen file can change under a running stage without stopping
it, the snapshot certifies nothing.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_contract as contract
from smallAntibodyGen.experiments import her2_nf_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths


def _write_repo(tmp_path):
    """A miniature repository with a real import closure and a git-less worktree."""
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "leaf.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "entry.py").write_text(
        "from pkg import leaf\n\nprint(leaf.VALUE)\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    return tmp_path


def _config(tmp_path, **overrides):
    document = {
        "schema_version": contract.NF_SCHEMA, "campaign_id": "nf_test",
        "config_path": "configs/flight.json", "run_root": "outputs/nf_test",
        "roots": {"raw": {"logical": "data/raw"}},
        "source": {"entry_points": ["scripts/entry.py"], "extra_files": []},
        "historical": {},
        "storage": {"min_free_bytes": 1},
        "probability": {}, "objectives": {}, "preservation": {}, "optimization": {},
        "banks": {}, "seeds": {"spawn_keys": {"a": [1, 0], "b": [1, 1]}},
        "monitor": {}, "block_a": {}, "block_b": {},
        "analysis": {"primary_contrasts": [], "primary_endpoints": {}},
        "calibration": {}, "split": {},
    }
    document.update(overrides)
    path = tmp_path / "configs" / "flight.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = _write_repo(tmp_path)
    config_path = _config(root)
    # The snapshot records git state; this tree is not a repository, so the read
    # is stubbed rather than skipped -- the record must still carry the fields.
    monkeypatch.setattr(spec.support, "git_state",
                        lambda repository_root: {"commit": "deadbeef", "dirty": True,
                                                 "dirty_entries": ["?? scripts/entry.py"]})
    monkeypatch.setattr(spec.support, "head_blob_sha256", lambda root, logical: None)
    monkeypatch.setattr(spec.support, "git_tracked", lambda root, logical: False)
    return spec.resolve_context(root, config_path=config_path)


def test_a_config_with_an_unresolved_placeholder_is_refused(tmp_path):
    root = _write_repo(tmp_path)
    path = _config(root, campaign_id="TODO")
    with pytest.raises(ValueError, match="placeholder"):
        spec.load_config(path)


def test_the_snapshot_copies_the_real_closure_and_is_content_addressed(context):
    document = spec.source_snapshot(context)
    assert set(document["files"]) >= {"scripts/entry.py", "src/pkg/leaf.py",
                                      "src/pkg/__init__.py", "configs/flight.json"}
    for logical, block in document["files"].items():
        copied = context.path(spec.SNAPSHOT_DIR) / logical
        assert copied.is_file()
        assert paths.sha256_file(copied) == block["sha256"]
        assert block["classification"] == "untracked_new"
    assert len(document["snapshot_sha256"]) == 64
    assert document["git_state"]["head"] == "deadbeef"
    assert document["git_state"]["dirty"] is True
    assert "content-addressed rather than a commit" in document["identity_is"]
    assert "not modified" in document["inherited_freeze_untouched"] or \
        "not called here" in document["inherited_freeze_untouched"]


def test_a_changed_frozen_file_stops_the_next_stage(context):
    spec.source_snapshot(context)
    assert spec.verify_source_snapshot(context, label="probe")["verified"] is True
    (context.repository_root / "src" / "pkg" / "leaf.py").write_text("VALUE = 2\n",
                                                                     encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source changed"):
        spec.verify_source_snapshot(context, label="production")


def test_an_existing_snapshot_is_verified_and_reused_rather_than_rewritten(context):
    """A frozen identity a later call can replace is not a freeze."""
    first = spec.source_snapshot(context)
    marker = paths.sha256_file(context.path(spec.FREEZE_MARKER))
    second = spec.source_snapshot(context)
    assert second["snapshot_sha256"] == first["snapshot_sha256"]
    assert second["reused"] is True
    assert second["verification"]["archived_files"] == first["file_count"]
    assert paths.sha256_file(context.path(spec.FREEZE_MARKER)) == marker


def test_changed_source_refuses_to_re_freeze_under_the_same_identity(context):
    spec.source_snapshot(context)
    (context.repository_root / "src" / "pkg" / "leaf.py").write_text("VALUE = 3\n",
                                                                     encoding="utf-8")
    with pytest.raises(ValueError, match="verified and REUSED, never rewritten"):
        spec.source_snapshot(context)


def test_an_edited_archive_copy_is_caught_even_when_the_worktree_is_clean(context):
    """The archived bytes are what a later reader opens."""
    document = spec.source_snapshot(context)
    logical = "src/pkg/leaf.py"
    assert logical in document["files"]
    (context.path(spec.SNAPSHOT_DIR) / logical).write_text("VALUE = 99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source changed"):
        spec.verify_source_snapshot(context, label="production")
    with pytest.raises(ValueError, match="verified and REUSED"):
        spec.source_snapshot(context)


def test_a_changed_configuration_file_is_a_new_protocol_version(context):
    """The coefficients, bank sizes and endpoints this flight runs under live there."""
    spec.source_snapshot(context)
    context.config_path.write_text(
        context.config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="new protocol version"):
        spec.source_snapshot(context)


def test_a_snapshot_directory_without_its_marker_is_not_deleted(context):
    spec.source_snapshot(context)
    context.path(spec.FREEZE_MARKER).unlink()
    with pytest.raises(ValueError, match="not recursively deleted"):
        spec.source_snapshot(context)
    assert any(context.path(spec.SNAPSHOT_DIR).rglob("*")), "the bytes are still there"


def test_the_runtime_closure_check_is_reported_as_skipped_under_pytest(context):
    spec.source_snapshot(context)
    block = spec.verify_source_snapshot(context, label="probe")
    assert block["runtime_closure_checked"] is False
    assert "PYTEST_CURRENT_TEST" in block["runtime_closure_skip_reason"]
    # Asked for explicitly it still runs, and test modules stay unfrozen.
    forced = spec.verify_source_snapshot(context, label="probe",
                                         check_runtime_closure=True)
    assert forced["runtime_closure_checked"] is True
    assert all(entry.startswith("src/smallAntibodyGen/tests/")
               for entry in forced["non_production_imports"])


def test_a_deleted_frozen_file_is_also_a_stop(context):
    spec.source_snapshot(context)
    (context.repository_root / "src" / "pkg" / "leaf.py").unlink()
    with pytest.raises(ValueError, match="Missing"):
        spec.verify_source_snapshot(context, label="production")


def test_verify_runs_before_a_snapshot_exists_and_says_so(context):
    with pytest.raises(ValueError, match="is absent"):
        spec.verify_source_snapshot(context, label="production")


def test_supersession_is_disclosed_and_its_absence_is_a_gap_not_an_assurance(context):
    block = spec.historical_supersession_disclosure(context)
    assert block["available"] is False
    assert "no supersession provenance is configured" in block["reason"]
    target = context.repository_root / "provenance.json"
    paths.write_json(target, {"record_kind": "supersession", "files": ["a", "b", "c"]})
    context.config["historical"]["supersession_provenance"] = "provenance.json"
    block = spec.historical_supersession_disclosure(context)
    assert block["available"] is True
    assert "does NOT claim" in block.get("consequence", "") or \
        "makes no claim that every historical source byte is identical" in block["statement"]


def test_a_missing_supersession_record_is_reported_as_an_incomplete_disclosure(context):
    context.config["historical"]["supersession_provenance"] = "not/here.json"
    block = spec.historical_supersession_disclosure(context)
    assert block["available"] is False
    assert "does NOT claim" in block["consequence"]


def test_recover_lists_what_it_could_not_recover(context):
    context.config["recovery_probes"] = [
        {"path": "scripts/entry.py"},
        {"path": "outputs/nowhere/parent.pt", "sha256": "0" * 64}]
    block = spec.recover(context)
    assert block["recovered"]["unrecovered"][0]["path"] == "outputs/nowhere/parent.pt"
    assert block["discrepancies"]["entries"]
    assert "never patched in memory" in block["discrepancies"]["policy"]
    assert "never read" in block["recovered"]["reserved_test_labels"]


def test_recover_flags_a_digest_mismatch_rather_than_adopting_the_observed_value(context):
    context.config["recovery_probes"] = [{"path": "scripts/entry.py", "sha256": "1" * 64}]
    block = spec.recover(context)
    entry = block["recovered"]["probes"]["scripts/entry.py"]
    assert entry["matches"] is False
    assert entry["expected_sha256"] == "1" * 64
    assert any(item["kind"] == "digest_mismatch" for item in block["discrepancies"]["entries"])


def test_the_resolved_protocol_lists_its_unresolved_fields_and_blocks_production(context):
    snapshot = spec.source_snapshot(context)
    document = spec.resolved_protocol(context, snapshot=snapshot)
    assert len(document["unresolved"]) == 3
    assert any("MEASURED profile" in entry for entry in document["unresolved"])
    with pytest.raises(ValueError, match="unresolved fields"):
        spec.require_resolved(context, label="production")


def test_a_measured_forecast_clears_the_unresolved_list(context):
    snapshot = spec.source_snapshot(context)
    document = spec.resolved_protocol(context, snapshot=snapshot, geometry={"passed": True},
                                      calibration={"frozen": {}},
                                      forecast={"measured": True, "total_hours": 17.8})
    assert document["unresolved"] == []
    assert spec.require_resolved(context, label="production")["unresolved"] == []


def test_immutable_fields_cannot_be_overridden_after_the_freeze():
    with pytest.raises(ValueError, match="frozen before three-seed production"):
        spec.require_immutable({}, {"block_b": {"regimes": ["purge"]}})
    assert spec.require_immutable({}, {"inference": {"score_batch_size": 128}}) is True


def test_a_post_freeze_change_becomes_a_new_protocol_version_with_a_reason(context):
    snapshot = spec.source_snapshot(context)
    previous = spec.resolved_protocol(context, snapshot=snapshot)
    amended = spec.protocol_amendment(context, previous=previous, reason="storage moved",
                                      changes={"run_root": "elsewhere"})
    assert amended["protocol_version"] == 2
    assert amended["amendment"]["reason"] == "storage moved"
    assert context.path("resolved_protocol.v2.json").is_file()
    assert context.path(spec.RESOLVED_PROTOCOL).is_file(), "the old version is retained"


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------

def _cores(rows, seed):
    return [_word(value) for value in np.random.default_rng(seed).integers(0, 20, size=(rows, 10))]


def _word(row):
    from smallAntibodyGen.experiments.her2_data import CANONICAL
    return "".join(CANONICAL[int(value)] for value in row)


def test_the_lineage_refuses_a_cache_from_a_different_parent():
    lineage = spec.ModelLineage()
    lineage.add(spec.LineageNode(node_id="parent::a", kind="stage1", parent_id=None,
                                 parent_state_sha256=None, population="purge",
                                 forbidden_rows=None, config_sha256="c",
                                 scaffold_prefix_sha256="s"))
    lineage.add(spec.LineageNode(node_id="policy::a", kind="policy", parent_id="parent::a",
                                 parent_state_sha256="aaa", population="purge",
                                 forbidden_rows=None, config_sha256="c",
                                 scaffold_prefix_sha256="s"))
    assert lineage.require_parent("policy::a", observed_state_sha256="aaa", where="cache")
    with pytest.raises(ValueError, match="descends from"):
        lineage.require_parent("policy::a", observed_state_sha256="bbb", where="cache")


def test_the_lineage_refuses_an_unknown_parent_edge():
    lineage = spec.ModelLineage()
    with pytest.raises(ValueError, match="not in the DAG"):
        lineage.add(spec.LineageNode(node_id="policy::x", kind="policy", parent_id="missing",
                                     parent_state_sha256="aaa", population="purge",
                                     forbidden_rows=None, config_sha256="c",
                                     scaffold_prefix_sha256="s"))


def test_forbidden_evaluation_rows_cannot_be_resolved_by_a_challenge_population():
    panel = _cores(6, seed=1)
    others = _cores(6, seed=2)
    guard = contract.ForbiddenRows.from_cores(panel, label="E", reason="E never trains")
    assert guard.check(others, where="challenge loader") is True
    with pytest.raises(ValueError, match="forbidden set 'E'"):
        guard.check(others + panel[:1], where="challenge loader")
    assert guard.document()["rows"] == 6


def test_the_lineage_enforces_forbidden_rows_at_the_node():
    panel = _cores(4, seed=3)
    lineage = spec.ModelLineage()
    lineage.add(spec.LineageNode(
        node_id="challenge::purge", kind="policy", parent_id=None, parent_state_sha256="a",
        population="purge",
        forbidden_rows=contract.ForbiddenRows.from_cores(panel, label="E", reason="E never trains"),
        config_sha256="c", scaffold_prefix_sha256="s"))
    with pytest.raises(ValueError, match="forbidden set"):
        lineage.require_allowed_rows("challenge::purge", panel, where="training population")
    document = lineage.document()
    assert document["nodes"]["challenge::purge"]["forbidden_rows"]["rows"] == 4
    assert "refused at the loader" in document["enforced"][1]


def test_artifact_roots_come_from_the_configuration(context):
    assert context.artifact_root("raw").name == "raw"
    with pytest.raises(ValueError, match="No configured artifact root"):
        context.artifact_root("evidence_copies")
