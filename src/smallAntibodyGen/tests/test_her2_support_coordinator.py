"""First run, rerun, interruption, tamper, coverage and publication, on real files.

These drive the audit coordinator's *stage* contracts rather than its arithmetic.
Nothing here is stubbed except the numbers themselves: the completion manifest, the
shard records, the ``.npz`` containers, the published tree and the report bytes are
all written and re-read from disk, because every failure mode being checked is a
disagreement between what a summary claims and what is actually on disk. A test
that mocked the ledger would only prove that the mock agrees with itself.

The GPU-bound halves of ``score`` and ``ches`` are not run here -- they need real
weights -- so what is exercised is the decision each of them makes *before*
computing: reuse this shard, recompute that one, refuse this rerun.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_support as support
from smallAntibodyGen.experiments import her2_support_inventory as inventory_lib
from smallAntibodyGen.experiments import her2_support_paths as paths
from smallAntibodyGen.experiments import her2_support_scoring as scoring


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = json.loads((REPOSITORY / "configs/experiments/her2_support_audit.json").read_text(
    encoding="utf-8"))

SCORE_IDENTITY = {"state_digest": "d" * 64, "file_sha256": "f" * 64, "bank_sha256": "b" * 64,
                  "bank_order_sha256": "o" * 64, "score_batch_size": 256, "device": "cpu"}


def context_for(tmp_path, **config):
    run = paths.RunPaths.create(tmp_path, tmp_path / "run", logical="outputs/audit")
    return SimpleNamespace(repository_root=tmp_path, run=run, roots={}, historical={},
                           config_path=tmp_path / "configs/audit.json",
                           config=dict(CONFIG, **config))


def statistics_block(mean=0.2, tenfold=0.004, hundredfold=0.0005):
    return {"rows": 3, "forward_kl": {"mean": mean, "ci_low": mean - 0.01, "ci_high": mean + 0.01},
            "tails": {"rows": 3,
                      "counts": {"tenfold": dict(scoring.wilson_interval(40, 10000),
                                                 threshold=scoring.LN10, fraction=tenfold),
                                 "hundredfold": dict(scoring.wilson_interval(5, 10000),
                                                     threshold=scoring.LN100,
                                                     fraction=hundredfold)}}}


def write_score_shard(context, identifier, *, values=(0.1, 0.2, 0.3), identity=None,
                      statistics=None):
    """One completed sequence-score shard, written exactly as the score stage writes it."""
    prefix = support.shard_directory("scores", identifier)
    return paths.write_shard(
        context.run.path(prefix), "sequence_scores",
        {"drop": np.asarray(values, dtype=np.float64),
         "draw_index": np.arange(len(values), dtype=np.int64)},
        {"record_kind": "sequence_scores", "checkpoint_id": identifier,
         "identity": dict(SCORE_IDENTITY if identity is None else identity),
         "statistics": statistics or statistics_block(),
         "timings": {"inference_seconds": 3.0, "wall_seconds": 4.0}},
        order="bank draw order 0..N-1", logical_prefix=prefix, run_root=context.run.run_root)


# ---------------------------------------------------------------------------
# first run, rerun, interruption
# ---------------------------------------------------------------------------

def test_a_completed_shard_is_verified_and_reused_rather_than_recomputed(tmp_path):
    context = context_for(tmp_path)
    written = write_score_shard(context, "example")
    arrays, record = support.reusable_shard(context, "scores", "example", "sequence_scores",
                                            identity=SCORE_IDENTITY)
    assert np.allclose(arrays["drop"], [0.1, 0.2, 0.3])
    assert record["statistics"] == written["statistics"]
    assert record["timings"] == {"inference_seconds": 3.0, "wall_seconds": 4.0}


def test_an_interrupted_stage_reuses_what_finished_and_computes_only_the_rest(tmp_path):
    context = context_for(tmp_path)
    write_score_shard(context, "finished")
    reusable = {identifier: support.reusable_shard(context, "scores", identifier,
                                                   "sequence_scores", identity=SCORE_IDENTITY)
                for identifier in ("finished", "never_started")}
    assert reusable["finished"] is not None
    assert reusable["never_started"] is None, "an absent shard is recomputed, not invented"


def test_a_partial_npz_without_its_record_never_counts_as_finished(tmp_path):
    context = context_for(tmp_path)
    directory = context.run.path(support.shard_directory("scores", "killed"))
    paths.write_arrays(directory / "sequence_scores.npz", {"drop": np.array([0.1, 0.2])})
    assert support.reusable_shard(context, "scores", "killed", "sequence_scores") is None


def test_a_shard_produced_under_another_identity_is_not_adopted(tmp_path):
    context = context_for(tmp_path)
    write_score_shard(context, "example")
    with pytest.raises(ValueError, match="different identity"):
        support.reusable_shard(context, "scores", "example", "sequence_scores",
                               identity=dict(SCORE_IDENTITY, bank_sha256="9" * 64))


def test_a_tampered_shard_container_is_refused_on_reuse(tmp_path):
    context = context_for(tmp_path)
    write_score_shard(context, "example")
    directory = context.run.path(support.shard_directory("scores", "example"))
    paths.write_arrays(directory / "sequence_scores.npz",
                       {"drop": np.asarray([9.0, 9.0, 9.0]), "draw_index": np.arange(3)})
    with pytest.raises(ValueError, match="completion manifest recorded|container hashes"):
        support.reusable_shard(context, "scores", "example", "sequence_scores")


def test_an_edited_completion_record_is_refused_on_reuse(tmp_path):
    context = context_for(tmp_path)
    write_score_shard(context, "example")
    directory = context.run.path(support.shard_directory("scores", "example"))
    record_path = directory / f"sequence_scores{paths.SHARD_RECORD}"
    record = paths.read_json(record_path)
    record["statistics"]["forward_kl"]["mean"] = 0.0
    paths.write_json(record_path, record)
    with pytest.raises(ValueError, match="completion manifest recorded"):
        support.reusable_shard(context, "scores", "example", "sequence_scores")


# ---------------------------------------------------------------------------
# a completed stage is reused only when the ledger agrees with the disk
# ---------------------------------------------------------------------------

def complete_score_stage(context, identifiers=("a", "b")):
    for identifier in identifiers:
        write_score_shard(context, identifier)
    summary = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "checkpoint_summaries",
               "scored": len(identifiers),
               "checkpoints": {identifier: statistics_block() for identifier in identifiers},
               "self_controls": {}, "timings": {"wall_seconds": 1.0},
               "generated_at": paths.utc_now()}
    coverage = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "coverage",
                "scored": sorted(identifiers), "scored_count": len(identifiers),
                "expected_computations": len(identifiers),
                "scored_all_expected_computations": True, "complete": True}
    support.require_new_or_identical(context.run.path(support.CHECKPOINTS_JSON), summary,
                                     what="the checkpoint summary", context=context,
                                     kind="checkpoint_summaries")
    support.require_new_or_identical(context.run.path(support.COVERAGE_JSON), coverage,
                                     what="the coverage record", context=context, kind="coverage")
    return summary, coverage


def test_a_stage_that_never_ran_is_not_a_completed_one(tmp_path):
    context = context_for(tmp_path)
    assert support.completed_stage(context, {"summary": support.CHECKPOINTS_JSON}) is None


def test_a_completed_stage_returns_its_recorded_documents(tmp_path):
    context = context_for(tmp_path)
    summary, coverage = complete_score_stage(context)
    reused = support.completed_stage(context, {"summary": support.CHECKPOINTS_JSON,
                                               "coverage": support.COVERAGE_JSON})
    assert reused["summary"]["checkpoints"] == summary["checkpoints"]
    assert reused["coverage"]["scored_count"] == coverage["scored_count"]


def test_a_summary_edited_after_completion_is_not_reused(tmp_path):
    context = context_for(tmp_path)
    complete_score_stage(context)
    target = context.run.path(support.CHECKPOINTS_JSON)
    document = paths.read_json(target)
    document["checkpoints"]["a"]["forward_kl"]["mean"] = 99.0
    paths.write_json(target, document)
    with pytest.raises(ValueError, match="does not match the saved completion manifest"):
        support.completed_stage(context, {"summary": support.CHECKPOINTS_JSON})


def test_a_summary_written_outside_the_ledger_is_not_reused(tmp_path):
    """A file that simply appeared is not evidence that a stage completed."""
    context = context_for(tmp_path)
    paths.write_json(context.run.path(support.CHECKPOINTS_JSON),
                     {"record_kind": "checkpoint_summaries", "checkpoints": {}, "scored": 0})
    with pytest.raises(ValueError, match="never recorded complete"):
        support.completed_stage(context, {"summary": support.CHECKPOINTS_JSON})


# ---------------------------------------------------------------------------
# coverage: a summary cannot claim outputs that are not there
# ---------------------------------------------------------------------------

def test_expected_outputs_are_derived_from_the_summaries_not_from_a_scan(tmp_path):
    context = context_for(tmp_path)
    complete_score_stage(context)
    expected = support.expected_run_outputs(context)
    for identifier in ("a", "b"):
        prefix = support.shard_directory("scores", identifier)
        for logical in paths.shard_logicals(prefix, "sequence_scores"):
            assert logical in expected


def test_a_deleted_shard_still_claimed_by_the_summary_fails_verification(tmp_path):
    context = context_for(tmp_path)
    complete_score_stage(context)
    assert support.verify_outputs(context)["immutable"] is True
    directory = context.run.path(support.shard_directory("scores", "a"))
    (directory / "sequence_scores.npz").unlink()
    (directory / f"sequence_scores{paths.SHARD_RECORD}").unlink()
    report = support.verify_outputs(context)
    assert report["immutable"] is False
    assert any("scores/a/" in str(problem.get("artifact")) for problem in report["problems"])


def test_a_deleted_completion_marker_never_allows_overwriting_the_saved_container(tmp_path):
    context = context_for(tmp_path)
    write_score_shard(context, "example")
    directory = context.run.path(support.shard_directory("scores", "example"))
    (directory / f"sequence_scores{paths.SHARD_RECORD}").unlink()
    before = (directory / "sequence_scores.npz").read_bytes()
    with pytest.raises(ValueError, match="completion manifest recorded"):
        support.reusable_shard(context, "scores", "example", "sequence_scores")
    with pytest.raises(ValueError, match="completion manifest recorded"):
        write_score_shard(context, "example", values=(9., 9., 9.))
    assert (directory / "sequence_scores.npz").read_bytes() == before


def test_a_coverage_record_cannot_report_completion_without_its_shards(tmp_path):
    """Coverage says 2 of 2 scored; one shard's files are gone. The audit is not complete."""
    context = context_for(tmp_path)
    complete_score_stage(context)
    directory = context.run.path(support.shard_directory("scores", "b"))
    (directory / "sequence_scores.npz").unlink()
    (directory / f"sequence_scores{paths.SHARD_RECORD}").unlink()
    verification = support.verify_outputs(context)
    coverage = paths.read_json(context.run.path(support.COVERAGE_JSON))
    assert coverage["scored_all_expected_computations"] is True
    assert verification["immutable"] is False, (
        "a coverage table that still claims a vanished output must not read as complete")


# ---------------------------------------------------------------------------
# the report and the published deliverable are written once
# ---------------------------------------------------------------------------

def test_a_report_rerun_verifies_its_bytes_instead_of_overwriting_them(tmp_path):
    context = context_for(tmp_path)
    target = context.run.path(support.REPORT_MD)
    digest = support.require_new_or_identical_text(target, "# report\n", what="the report",
                                                   context=context, kind="report")
    assert support.require_new_or_identical_text(target, "# report\n", what="the report",
                                                 context=context, kind="report") == digest
    assert context.run.logical(support.REPORT_MD) in paths.read_completion_manifest(
        context.run.run_root)["artifacts"]
    with pytest.raises(ValueError, match="different bytes"):
        support.require_new_or_identical_text(target, "# other report\n", what="the report",
                                              context=context, kind="report")


def inventory_records():
    return [{"id": f"guarded::stage1::{arm}_seed{seed}::budget600",
             "role": inventory_lib.ROLE_ENDPOINT, "arm_id": arm, "seed": seed,
             "nominal_budget_gpu_seconds": 600.0}
            for arm in ("continued_sft", "ipo_tau0p1") for seed in CONFIG["decision"]["seeds"]]


def decision_for(context):
    records = inventory_records()
    checkpoints = {record["id"]: statistics_block() for record in records}
    inputs = support.decision_inputs(checkpoints, settings=context.config["decision"],
                                     inventory_records=records)
    decision = scoring.decision_record(
        inputs, settings=context.config["decision"], coverage={"complete": True},
        completion={"complete": True, "unmet": []})
    decision.update(schema_version=paths.AUDIT_SCHEMA, record_kind="decision",
                    audit_id=context.config["audit_id"], generated_at=paths.utc_now())
    return decision, checkpoints, records


def figures_for(context):
    logical = "report/figures/her2-support-forward-kl.png"
    target = context.run.path(logical)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\n not really a png, but bytes that must survive")
    return {"her2-support-forward-kl.png": {
        "written": True, "name": "her2-support-forward-kl.png", "logical": logical,
        "sha256": paths.sha256_file(target), "caption": "per-endpoint forward KL"}}


def publish_once(context):
    decision, checkpoints, records = decision_for(context)
    figures = figures_for(context)
    coverage = {"complete": True, "expected": {"total": 159}, "by_status": {"verified": 159},
                "by_role": {}, "shortfalls": [], "verified_by_role": {}}
    scored = {"checkpoints": checkpoints, "scored": len(checkpoints),
              "paired_method_differences": {"comparisons": {}, "count": 0}}
    inventory = {"records": records, "deduplication": {"distinct_computations": len(checkpoints)},
                 "trajectories": {"completed": 24, "stopped": 30}, "parent_banks": {}}
    verification = {"immutable": True, "shards_checked": 2, "problems": [],
                    "expected_output_count": 4}
    return support.publish_deliverable(
        context,
        render=lambda plan: support.render_report(
            context, inventory=inventory, checkpoints=checkpoints, ches_summary=None,
            decision=decision, coverage=coverage, timings={"score": {"wall_seconds": 10.0}},
            figures=figures, paired=scored["paired_method_differences"], publication=plan),
        decision=decision, coverage=coverage, checkpoints_document=scored, ches_document=None,
        verification=verification, figures=figures)


def test_publishing_writes_the_report_the_small_tables_and_the_figures(tmp_path):
    context = context_for(tmp_path)
    manifest = publish_once(context)
    root = tmp_path / context.config["publication"]["root"]
    for logical in manifest["files"]:
        assert (root / logical).is_file(), logical
    assert any(name.endswith("decision.json") for name in manifest["files"])
    assert any(name.endswith("checkpoints.json") for name in manifest["files"])
    assert any(name.endswith(".png") for name in manifest["files"])
    # the figure is copied byte for byte, never re-rendered into the published tree
    published = root / context.config["publication"]["figure_directory"] / \
        "her2-support-forward-kl.png"
    assert published.read_bytes() == context.run.path(
        "report/figures/her2-support-forward-kl.png").read_bytes()


def test_the_published_report_links_resolve_relative_to_itself(tmp_path):
    context = context_for(tmp_path)
    manifest = publish_once(context)
    report_path = tmp_path / context.config["published_report"]
    text = report_path.read_text(encoding="utf-8")
    linked = [logical for logical in manifest["files"] if not logical.endswith(".md")]
    assert linked, "something other than the report itself must be published"
    for logical in linked:
        assert f"]({logical})" in text, logical
        assert (report_path.parent / logical).is_file(), logical
    assert "![" in text, "the figure is embedded, not named in backticks"


def test_republishing_verifies_byte_for_byte_and_never_rewrites(tmp_path):
    context = context_for(tmp_path)
    first = publish_once(context)
    again = publish_once(context)
    assert first["files"] == again["files"]


def test_a_published_file_that_would_change_is_refused(tmp_path):
    context = context_for(tmp_path)
    publish_once(context)
    target = (tmp_path / context.config["publication"]["root"]
              / context.config["publication"]["data_directory"] / "decision.json")
    document = paths.read_json(target)
    document["outcome"] = "escalate"
    paths.write_json(target, document)
    with pytest.raises(ValueError, match="different scientific content"):
        publish_once(context)


def test_the_report_cost_table_excludes_the_reporting_stage(tmp_path):
    """A report that quoted its own elapsed time would change on every rebuild."""
    context = context_for(tmp_path)
    decision, checkpoints, records = decision_for(context)
    coverage = {"complete": True, "expected": {"total": 159}, "by_status": {"verified": 159},
                "by_role": {}, "shortfalls": [], "verified_by_role": {}}
    inventory = {"records": records, "deduplication": {"distinct_computations": len(checkpoints)},
                 "trajectories": {"completed": 24, "stopped": 30}, "parent_banks": {}}
    kwargs = dict(inventory=inventory, checkpoints=checkpoints, ches_summary=None,
                  decision=decision, coverage=coverage, figures={})
    first = support.render_report(context, timings={"score": {"wall_seconds": 10.0}}, **kwargs)
    second = support.render_report(
        context, timings={"score": {"wall_seconds": 10.0},
                          "report": {"wall_seconds": 2.0},
                          "verify": {"wall_seconds": 1.0}}, **kwargs)
    assert first == second
    assert "report" not in support.COSTED_STAGES


def test_report_cli_first_run_and_rerun_preserve_completed_bytes(tmp_path, monkeypatch):
    """Exercise the actual report command as the completion ledger grows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_cli", REPOSITORY / "scripts/audit_her2_support.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    context = context_for(tmp_path, reporting={"figures": False})
    context.seeds = CONFIG["populations"]["seeds"]
    context.progress = lambda stage, **kw: paths.StageProgress(
        context.run.path(f"progress/{stage}.json"), stage=stage, **kw)
    context.stage_status = lambda stage: paths.read_json(context.run.path(f"progress/{stage}.json"))
    # Freeze/input validators have their own tests. This fixture starts at report;
    # summaries, ledger, completion checks, publication and reruns are all real.
    monkeypatch.setattr(support, "require_frozen_identity", lambda ctx: None)
    monkeypatch.setattr(support, "require_committed_inventory", lambda ctx, inv: None)
    decision, checkpoints, records = decision_for(context)
    coverage = {"complete": True, "expected": {"total": len(records)},
                "verified_total": len(records), "by_status": {"verified": len(records)},
                "by_role": {}, "shortfalls": [], "verified_by_role": {}}
    inventory = {"records": records, "coverage": coverage,
                 "deduplication": {"distinct_computations": len(records)},
                 "trajectories": {"completed": 24, "stopped": 30},
                 "parent_banks": {str(s): {} for s in context.seeds}}
    scored, coverage_document = complete_score_stage(context, tuple(checkpoints))
    # Numerical completion is a synthetic fixture; no native inference is claimed.
    monkeypatch.setattr(cli, "audit_requirements", lambda *a, **kw: {
        "complete": bool(kw["verification"]["immutable"]), "unmet": []})
    for logical, document in [(support.INVENTORY_JSON, inventory),
                               (support.DECISION_JSON, decision)]:
        support.require_new_or_identical(context.run.path(logical), document,
                                         what=logical, context=context)
    coverage_document["inventory_coverage"] = coverage
    # Replace the synthetic fixture's coverage before the report stage is exercised.
    target = context.run.path(support.COVERAGE_JSON)
    paths.write_json(target, coverage_document)
    ledger = paths.read_completion_manifest(context.run.run_root)
    ledger["artifacts"][context.run.logical(support.COVERAGE_JSON)]["sha256"] = paths.sha256_file(target)
    paths.write_json(paths.completion_manifest_path(context.run.run_root), ledger)
    paths.write_json(support.freeze_marker_path(context), {
        "schema_version": paths.AUDIT_SCHEMA, "record_kind": "audit_spec_frozen",
        "git": {"commit": "fixture"}})
    for stage in ("inventory", "prepare", "preflight", "score", "ches", "decide"):
        with context.progress(stage).guard():
            pass
    args = SimpleNamespace(publish=True)
    cli.cmd_report(context, args)
    assert context.run.path(support.COMPLETE_JSON).is_file()
    immutable_paths = [context.run.path(support.REPORT_MD), context.run.path(support.COMPLETE_JSON),
                       context.run.path("timings/report.json")]
    immutable_paths += [p for p in (tmp_path / "reference").rglob("*") if p.is_file()]
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in immutable_paths}
    cli.cmd_report(context, args)
    assert support.verify_outputs(context)["immutable"]
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in immutable_paths}


def test_report_displacement_uses_all_pairs_not_equal_weight_deciles(tmp_path):
    context = context_for(tmp_path)
    decision, checkpoints, records = decision_for(context)
    coverage = {"complete": True, "expected": {"total": 6}, "by_status": {"verified": 6},
                "by_role": {}, "shortfalls": [], "verified_by_role": {}}
    inventory = {"records": records, "deduplication": {"distinct_computations": 6},
                 "trajectories": {"completed": 24, "stopped": 30}, "parent_banks": {}}
    ches = {"associations": {"endpoint::population": {
        "overall": {"n": 10, "spearman": 0.5},
        "by_parent_ches_decile": {"mean_displacement": {
            "1": {"rows": 9, "mean": 0.}, "2": {"rows": 1, "mean": 10.}}}}},
        "endpoints": {"endpoint::population": {"displacement": {"mean": 1.}}}}
    text = support.render_report(context, inventory=inventory, checkpoints=checkpoints,
                                 ches_summary=ches, decision=decision, coverage=coverage,
                                 timings={}, figures={})
    assert "| `endpoint::population` | 10 | 0.5000 | 1.0000 |" in text


class FakeFigure:
    """A stand-in for a matplotlib figure: ``savefig`` writes the next payload."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def savefig(self, path, **_):
        self.calls += 1
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payloads[min(self.calls, len(self.payloads)) - 1])


def test_a_figure_is_written_once_and_bound_into_the_completion_manifest(tmp_path):
    context = context_for(tmp_path)
    target = context.run.path("report/figures/her2-support-forward-kl.png")
    name, entry = support.save_figure(FakeFigure(b"figure bytes"), target, context=context)
    assert name == "her2-support-forward-kl.png" and entry["written"] is True
    artifacts = paths.read_completion_manifest(context.run.run_root)["artifacts"]
    assert artifacts[context.run.logical("report/figures/her2-support-forward-kl.png")][
        "sha256"] == entry["sha256"]


def test_a_rerendered_figure_never_replaces_the_published_one(tmp_path):
    context = context_for(tmp_path)
    target = context.run.path("report/figures/her2-support-forward-kl.png")
    support.save_figure(FakeFigure(b"original"), target, context=context)
    _, same = support.save_figure(FakeFigure(b"original"), target, context=context)
    assert "rerendered_bytes_differ" not in same
    _, differing = support.save_figure(FakeFigure(b"a different rendering"), target,
                                       context=context)
    assert target.read_bytes() == b"original", "the completed figure is not overwritten"
    assert differing["rerendered_bytes_differ"]["kept"] == "the original file"
    assert not list(target.parent.glob("*.rerender")), "no scratch file is left behind"


def test_an_edited_figure_is_caught_by_the_completion_manifest(tmp_path):
    context = context_for(tmp_path)
    target = context.run.path("report/figures/her2-support-forward-kl.png")
    support.save_figure(FakeFigure(b"original"), target, context=context)
    assert support.verify_outputs(context)["immutable"] is True
    target.write_bytes(b"edited after the fact")
    report = support.verify_outputs(context)
    assert report["immutable"] is False
    assert any("figures" in str(problem.get("artifact")) for problem in report["problems"])


# ---------------------------------------------------------------------------
# raw input bindings: the key a freeze re-hashes has to be the file on disk
# ---------------------------------------------------------------------------

#: Exactly the shape ``her2_data.source_digests`` returns: the retrieval record's
#: own ``local_path``/``path``, which already begins with the raw root.
NATIVE_SOURCE_KEYS = (
    "data/raw/her2_functional_20260918/absci/LICENSE",
    "data/raw/her2_functional_20260918/piggen/model.safetensors",
    "data/raw/her2_functional_20260918/buzz/data/affinity_data/her2/her2_aff_large/processed/"
    "remove_overlap/random_split/0.7_0.15_0.15/train.csv",
)


def raw_root_at(directory, *, logical="data/raw/her2_functional_20260918"):
    """A resolved raw root holding the three native-shaped files."""
    written = {}
    for recorded in NATIVE_SOURCE_KEYS:
        suffix = paths.campaign_suffix(recorded, anchor=logical.rsplit("/", 1)[-1])
        target = directory / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"contents of {suffix}".encode("utf-8"))
        written[recorded] = paths.sha256_file(target)
    return paths.ResolvedRoot(name="raw", logical=logical, local_path=directory,
                              probes=(), candidates=()), written


@pytest.mark.parametrize("where", ["default", "override"])
def test_every_pinned_raw_source_resolves_and_hashes_where_the_manifest_says(tmp_path, where):
    """The check the previous revision could not pass: 23 doubled prefixes, 23 misses."""
    logical = "data/raw/her2_functional_20260918"
    directory = (tmp_path / logical) if where == "default" else (tmp_path / "elsewhere/release")
    directory.mkdir(parents=True)
    root, digests = raw_root_at(directory, logical=logical)
    context = context_for(tmp_path)
    context.roots = {"raw": root}
    for recorded, digest in sorted(digests.items()):
        key, entry = support.raw_input_entry(root.logical, anchor="her2_functional_20260918",
                                             recorded=recorded, digest=digest)
        assert key == recorded, "the manifest key stays the one repository-logical name"
        assert not entry["logical_path"].startswith("data/raw/")
        assert support._input_hash(context, entry) == digest


def test_the_doubled_prefix_that_broke_the_freeze_cannot_be_produced(tmp_path):
    logical = "data/raw/her2_functional_20260918"
    directory = tmp_path / logical
    directory.mkdir(parents=True)
    root, digests = raw_root_at(directory, logical=logical)
    recorded = NATIVE_SOURCE_KEYS[0]
    key, entry = support.raw_input_entry(root.logical, anchor="her2_functional_20260918",
                                         recorded=recorded, digest=digests[recorded])
    assert "data/raw/her2_functional_20260918/data/raw" not in key
    assert paths.resolve_under(directory, entry["logical_path"]).is_file()


# ---------------------------------------------------------------------------
# prepare: the immutability check runs before the first write
# ---------------------------------------------------------------------------

def test_pair_evidence_that_would_change_is_refused_before_anything_is_written(tmp_path):
    context = context_for(tmp_path)
    logical = f"{context.config['evidence_root']}/pairs/fixed_validation_pairs.csv"
    target = tmp_path / logical
    paths.write_text(target, "pair_index,chosen_row,rejected_row\n0,1,2\n")
    other = tmp_path / f"{context.config['evidence_root']}/pairs/v1_training_cycle0_seed1.csv"
    paths.write_text(other, "pair_index,cycle,chosen_row,rejected_row\n0,0,1,2\n")
    rendered = {logical: "pair_index,chosen_row,rejected_row\n0,9,9\n",
                str(other.relative_to(tmp_path)).replace("\\", "/"):
                    "pair_index,cycle,chosen_row,rejected_row\n0,0,1,2\n"}
    with pytest.raises(ValueError, match="committed pair files would change"):
        support.verify_pair_evidence(context, rendered)
    assert target.read_text(encoding="utf-8").endswith("0,1,2\n"), "nothing was written"


def test_pair_evidence_that_reproduces_is_left_exactly_as_it_was(tmp_path):
    context = context_for(tmp_path)
    logical = f"{context.config['evidence_root']}/pairs/fixed_validation_pairs.csv"
    text = "pair_index,chosen_row,rejected_row\n0,1,2\n"
    paths.write_text(tmp_path / logical, text)
    before = (tmp_path / logical).read_bytes()
    digests = support.verify_pair_evidence(context, {logical: text})
    support.write_pair_evidence(context, {logical: text})
    assert digests[logical] == paths.sha256_text(text)
    assert (tmp_path / logical).read_bytes() == before
