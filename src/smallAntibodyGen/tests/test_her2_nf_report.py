"""The report: nothing omitted, completion not inferred, and the claims it may not make.

The verification tests are the ones that matter. A report is easy to make look
complete; what has to be checkable is that a shorter-than-the-queue table, a
missing artifact, a changed frozen source and a zero-violation certificate that
is absent all fail rather than pass quietly.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_campaign as campaign
from smallAntibodyGen.experiments import her2_nf_report as report_lib
from smallAntibodyGen.experiments import her2_nf_spec as spec
from smallAntibodyGen.experiments import her2_support_paths as paths

CONFIG_PATH = "configs/experiments/her2_next_flight.json"


@pytest.fixture
def flight(tmp_path):
    import smallAntibodyGen
    root = paths.Path(smallAntibodyGen.__file__).resolve().parents[2]
    context = spec.resolve_context(root, config_path=root / CONFIG_PATH,
                                   run_root=tmp_path / "run")
    context.config["storage"]["min_free_bytes"] = 1
    return context


def test_the_forbidden_claim_guard_covers_the_inherited_and_the_new_phrases():
    report_lib.require_no_forbidden_claim("No global winner is named; equivalence is not claimed.")
    for phrase in ("the winning arm", "proves equivalence", "always better",
                   "converged performance", "guaranteed preservation"):
        with pytest.raises(ValueError):
            report_lib.require_no_forbidden_claim(f"This run shows {phrase} for DPO.")


def test_the_checkpoint_matrix_carries_every_queued_row_and_every_comparator(flight):
    queue = campaign.queue_for_reporting(flight)
    state = campaign.campaign_state(flight.run_root, queue)
    matrix = report_lib.checkpoint_matrix(flight, state)
    trajectories = {row["trajectory"] for row in matrix if row["kind"] == "preference_cell"}
    assert trajectories == {row.trajectory for row in queue}
    comparators = {row["entry"] for row in matrix if row["kind"] == "comparator"}
    assert {"parent", "cnn_single", "cnn_ensemble", "additive_linear",
            "interaction_classifier", "mixture_alpha", "historical_ipo0"} <= comparators
    # An unreached checkpoint is a row with a status, not an absent row.
    statuses = {row["checkpoint_status"] for row in matrix}
    assert "not_reached" in statuses or "missing" in statuses


def test_historical_reuse_rows_mark_their_early_checkpoints_missing(flight):
    queue = campaign.build_queue(flight.config, require_frozen=False, reuse_verified=True)
    state = campaign.campaign_state(flight.run_root, queue)
    matrix = report_lib.checkpoint_matrix(flight, state)
    reused = [row for row in matrix
              if row["kind"] == "preference_cell" and row["status"] == "reuse_verified"]
    assert reused
    # With no registry to consult there is nothing on disk for these rows: a
    # reused path writes no local journal and no local checkpoint.
    assert all(row["checkpoint_status"] == "missing" for row in reused)


def test_a_reused_endpoint_the_registry_resolved_is_not_reported_missing(flight):
    """Reproduced with the real IPO_0 seed20260918 u1000 checkpoint: the matrix
    called every reused endpoint missing while ``checkpoint_registry`` -- in the
    same document -- carried those weights as present and scored them. The
    genuinely absent early checkpoints must still read missing."""
    queue = campaign.build_queue(flight.config, require_frozen=False, reuse_verified=True)
    state = campaign.campaign_state(flight.run_root, queue)
    sample = next(row for row in state["rows"] if row.get("reuse") and row["block"] == "A")
    registry = {"entries": {
        f"A_{sample['arm']}@u1000_seed{sample['seed']}": {"status": "present",
                                                          "reused": "historical"},
        f"A_{sample['arm']}@u250_seed{sample['seed']}": {"status": "missing",
                                                         "reused": None}}}
    matrix = report_lib.checkpoint_matrix(flight, state, registry=registry)
    rows = {row["checkpoint_update"]: row for row in matrix
            if row["kind"] == "preference_cell" and row["trajectory"] == sample["trajectory"]}
    assert rows[1000]["checkpoint_status"] == "reached"
    assert rows[250]["checkpoint_status"] == "missing"


def test_the_failure_ledger_classifies_each_non_outcome(flight):
    queue = campaign.queue_for_reporting(flight)
    state = campaign.campaign_state(flight.run_root, queue)
    ledger = report_lib.failure_ledger(flight, state)
    kinds = {entry["classification"] for entry in ledger["entries"]}
    assert "declared optional work, explicitly deferred" in kinds
    assert "not started" in kinds or "not yet established by parity" in str(kinds)
    assert "a crash is a FAILURE" in ledger["classification_rule"]
    assert ledger["count"] == len(ledger["entries"])


def test_paired_contrasts_refuse_to_run_on_unpaired_values():
    records = {"A@1000": {20260918: 0.87}, "B@1000": {20260919: 0.86}}
    block = report_lib.paired_contrasts(records, contrasts=[{"name": "x", "left": "A@1000",
                                                             "right": "B@1000"}])
    assert block[0]["available"] is False
    assert "not computed from unpaired values" in block[0]["reason"]


def test_paired_contrasts_report_raw_seed_differences_and_the_power_caveat():
    records = {"A@1000": {1: 0.880, 2: 0.870, 3: 0.875},
               "B@1000": {1: 0.872, 2: 0.869, 3: 0.874}}
    block = report_lib.paired_contrasts(records, contrasts=[{"name": "x", "left": "A@1000",
                                                             "right": "B@1000"}])[0]
    assert block["paired_seeds"] == [1, 2, 3]
    assert block["raw_differences"] == pytest.approx([0.008, 0.001, 0.001])
    assert block["degrees_of_freedom"] == 2
    assert "not equivalence" in block["reading"]


def test_the_yield_section_retains_the_grid_the_crossovers_and_the_refinements():
    parent = np.log(np.full(50, 1e-4))
    policy = np.log(np.concatenate([np.full(10, 1e-2), np.full(40, 1e-6)]))
    sections = report_lib.yield_section({"policy_vs_parent": {
        "log_probabilities": policy, "control_log_probabilities": parent,
        "control_label": "parent"}}, dense=(1_000, 10_000, 100_000, 1_000_000))
    block = sections[0]
    assert set(block["curve"]) == {str(int(n)) for n in report_lib.metrics.YIELD_BUDGETS}
    assert len(block["dense_differences"]) == 4
    assert "not 'always better'" in block["reading"]
    assert isinstance(block["crossovers"]["brackets"], list)
    assert len(block["refinements"]) == len(block["crossovers"]["brackets"])


def test_the_coupling_and_decomposition_sections_state_what_they_forbid():
    coupling = report_lib.coupling_section([{"model": "a"}])
    assert "different quantities and are reported apart" in coupling["separation"]
    assert "every MI row carries the column-permutation floor" in coupling["requirements"]
    decomposition = report_lib.decomposition_section([])
    assert "AP differences are not added" in decomposition["forbidden"]
    assert "hybrid s_P + (c_Q - c_P)" in decomposition["reported"]


def test_runs_jsonl_is_one_object_per_line(flight):
    queue = campaign.queue_for_reporting(flight)
    state = campaign.campaign_state(flight.run_root, queue)
    block = report_lib.runs_jsonl(flight, state)
    text = flight.path(report_lib.RUNS_JSONL).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == block["rows"] == len(queue)
    for line in lines:
        record = json.loads(line)
        assert record["record_kind"] == "run"
        assert "status" in record and "run_id" in record


def test_the_artifact_manifest_hashes_what_is_on_disk(flight):
    paths.write_json(flight.path("thing.json"), {"a": 1})
    manifest = report_lib.artifact_manifest(flight)
    entry = next(item for item in manifest["files"] if item["file"] == "thing.json")
    assert entry["sha256"] == paths.sha256_file(flight.path("thing.json"))
    assert "source_snapshot" in manifest["exclusion_rule"]


def test_the_manifest_excludes_mutable_files_and_itself():
    """Otherwise the manifest depends on its own bytes and a rerun never matches."""
    for name in ("heartbeat.json", "campaign.lock", "supervisor.log",
                 report_lib.ARTIFACT_MANIFEST, report_lib.REPORT_JSON,
                 "verification.json", "source_snapshot/her2.py",
                 "trajectories/A_IPO_0_seed1/resume_state.pt",
                 "trajectories/A_IPO_0_seed1/trajectory_progress.json"):
        assert report_lib.is_immutable_artifact(name) is False, name
    for name in ("split_manifest.json", "calibration_ledger.json",
                 "trajectories/A_IPO_0_seed1/endpoint_update1000.pt",
                 "trajectories/A_IPO_0_seed1/updates.jsonl"):
        assert report_lib.is_immutable_artifact(name) is True, name


def test_the_manifest_is_stable_across_a_rerun(flight):
    """A manifest that changes every time it is written cannot be verified."""
    paths.write_json(flight.path("thing.json"), {"a": 1})
    first = report_lib.artifact_manifest(flight)
    second = report_lib.artifact_manifest(flight)
    assert first["files"] == second["files"]
    assert report_lib.verify(flight)["checks"]["artifacts_match_manifest"]["passed"] is True


def test_verification_fails_when_a_manifested_artifact_is_missing(flight):
    paths.write_json(flight.path("thing.json"), {"a": 1})
    report_lib.artifact_manifest(flight)
    flight.path("thing.json").unlink()
    document = report_lib.verify(flight)
    assert document["passed"] is False
    assert document["checks"]["artifacts_match_manifest"]["passed"] is False
    assert "thing.json" in document["checks"]["artifacts_match_manifest"]["detail"]


def test_verification_fails_without_a_zero_violation_certificate(flight):
    document = report_lib.verify(flight)
    assert document["checks"]["neighbor_certificate_zero_violations"]["passed"] is False
    assert "does not assert that the science is correct" in document["claim"]


def test_completion_requires_every_stage_and_every_terminal_row(flight):
    document = report_lib.build_report(flight)
    assert document["completion"]["complete"] is False
    assert document["completion"]["rows_without_terminal_status"]
    assert "never implies completion" in document["completion"]["rule"]
    assert flight.path(report_lib.REPORT_MD).is_file()
    text = flight.path(report_lib.REPORT_MD).read_text(encoding="utf-8")
    assert "in progress or stopped" in text
    assert "What this cannot say" in text


def test_the_report_carries_the_declared_limits(flight):
    document = report_lib.build_report(flight)
    joined = " ".join(document["limits"])
    assert "newly separated at ADAPTATION" in joined
    assert "INCLUSIVE" in joined
    assert "cannot establish converged performance" in joined


def test_the_curve_data_artifact_is_written_even_without_a_plotting_backend(flight):
    sections = report_lib.yield_section({})
    block = report_lib.render_curves(flight, sections)
    assert flight.path("curves.json").is_file()
    assert block["data_artifact"] == "curves.json"


def test_the_finalist_freeze_precedes_the_audit_banks(flight):
    document = report_lib.finalist_freeze(
        flight, named_checkpoints=[{"name": "IPO_FKL@1000", "checkpoint": "x.pt",
                                    "parent_checkpoint": "p.pt", "parent_id": "p"}],
        screen_models=[], tail_family={"chosen": "ipo_tail", "reason": "feasible in all seeds"},
        reason="three-seed development results")
    assert flight.path("finalist_freeze.json").is_file()
    assert "before any final-preservation" in document["ordering"]
    assert "retained and reported" in document["retention"]
    with pytest.raises(ValueError, match="at least one checkpoint"):
        report_lib.finalist_freeze(flight, named_checkpoints=[], screen_models=[],
                                   tail_family={}, reason="x")


def test_a_mixture_diagnostic_uses_the_true_mixture_conditionals():
    parent = np.log(np.full((6, 10, 20), 1.0 / 20))
    policy = np.log(np.full((6, 10, 20), 1.0 / 20))
    log_weights = (np.full((6, 10), np.log(0.11)), np.full((6, 10), np.log(0.89)))
    block = report_lib.mixture_diagnostic(parent, policy, log_weights, label="alpha089")
    assert block["conditionals_used"].startswith("posterior-weighted")
    assert "component identity can itself induce dependence" in block["caveat"].lower()
    # Two identical uniform components mix to the same uniform: zero dependence.
    assert block["mixture_total_correlation"]["total_correlation"] == pytest.approx(0.0,
                                                                                    abs=1e-12)


def test_a_limitation_that_names_a_forbidden_phrase_is_not_a_claim():
    """The report's own limits say "cannot establish converged performance"."""
    report_lib.require_no_forbidden_claim(
        "The challenge's u1000 endpoint is a fixed-exposure question and cannot establish "
        "converged performance.")
    report_lib.require_no_forbidden_claim("'No crossing found' is not 'always better'.")
    report_lib.require_no_forbidden_claim("This flight offers no guaranteed preservation.")
    with pytest.raises(ValueError, match="ASSERTS"):
        report_lib.require_no_forbidden_claim("DPO+FKL reaches converged performance here.")
    with pytest.raises(ValueError, match="ASSERTS"):
        report_lib.require_no_forbidden_claim(
            "Tail training delivers guaranteed preservation at every alpha.")


def test_completion_does_not_depend_on_the_stage_that_is_computing_it(flight):
    """report and verify cannot have completed while report is running."""
    for stage in campaign.STAGES:
        if stage in ("report", "verify"):
            continue
        campaign.write_stage_record(flight.run_root, stage, {"status": "completed"})
    document = report_lib.build_report(flight)
    # Still incomplete, because rows have no terminal status -- but NOT because
    # report and verify are pending by construction.
    assert document["completion"]["complete"] is False
    assert document["completion"]["rows_without_terminal_status"]
    assert document["stages"]["report"]["status"] in ("not_run", "completed")
