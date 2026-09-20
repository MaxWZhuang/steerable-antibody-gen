"""The coordinator's contracts: portable evidence, paired differences, reruns, roots.

These exercise the pieces of the stage coordinator that do not need a campaign on
disk. The ones that do -- inventory over 159 real states, scoring, CHES -- are
run natively; what is checkable here is the part that decides *what gets
published*, *what a rerun is allowed to do*, and *which directory is accepted as
a campaign root*.
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


def context_for(tmp_path):
    run = paths.RunPaths.create(tmp_path, tmp_path / "run", logical="outputs/audit")
    return SimpleNamespace(repository_root=tmp_path, run=run, config={}, roots={}, historical={})


# ---------------------------------------------------------------------------
# tracked evidence carries no host paths and no doubled roots
# ---------------------------------------------------------------------------

def inventory_document(**record_fields):
    record = dict({
        "id": "guarded::stage1::x::budget600", "role": inventory_lib.ROLE_ENDPOINT,
        "name": "x", "seed": 20260918, "root": "outputs/camp",
        "logical_path": "stage1/x/budget600.pt",
        "recorded_path": "C:\\Users\\big DAWG\\outputs\\camp\\stage1\\x\\budget600.pt",
        "recorded_path_flavor": paths.WINDOWS_ABSOLUTE, "file_sha256": "a" * 64,
        "status": inventory_lib.STATUS_VERIFIED, "status_reason": None,
        "aliases": [], "historical_metrics_source": {"generation": {"unique_fraction": 0.97}}},
        **record_fields)
    return {"protocol": "specs/plan.md", "audit_id": "audit", "roots": {}, "records": [record],
            "probability_contract": {}, "coverage": {}, "deduplication": {},
            "reused_controls": {}, "parent_banks": {}, "bank_cross_stage": {},
            "trajectories": {}}


def test_the_portable_inventory_publishes_no_host_path_anywhere():
    document = inventory_document(
        status=inventory_lib.STATUS_FAILED,
        status_reason="OSError: [Errno 2] F:/her2/stage1/x/budget600.pt is unreadable")
    portable = support.portable_inventory(document)
    assert paths.host_path_leaks(portable) == []
    record = portable["records"][0]
    assert "recorded_path" not in record, "the host-specific string is not published"
    assert record["recorded_path_flavor"] == paths.WINDOWS_ABSOLUTE
    assert record["recorded_path_sha256"] == paths.sha256_text(
        document["records"][0]["recorded_path"])
    assert "removed" in record["status_reason"]


def test_the_portable_inventory_keeps_the_logical_path_and_the_historical_metrics():
    portable = support.portable_inventory(inventory_document())
    record = portable["records"][0]
    assert record["logical_path"] == "stage1/x/budget600.pt"
    assert record["historical_metrics_source"]["generation"]["unique_fraction"] == 0.97


# ---------------------------------------------------------------------------
# the historical join is populated, or says why it is not
# ---------------------------------------------------------------------------

def test_the_historical_join_carries_the_numbers_the_enumeration_copied():
    record = {"role": inventory_lib.ROLE_ENDPOINT, "gate_passed": True,
              "diversity_eligible": True,
              "historical_metrics_source": {"parent_kl": {"mean": 0.51},
                                            "generation": {"unique_fraction": 0.97},
                                            "val_metrics": {"macro_ap": 0.8772}}}
    joined = support.historical_join(record)
    assert joined["parent_kl"]["value"]["mean"] == 0.51
    assert joined["generation"]["value"]["unique_fraction"] == 0.97
    assert joined["validation"]["value"]["macro_ap"] == 0.8772
    assert "reverse of this audit's estimand" in joined["parent_kl"]["direction"]


def test_a_metric_the_campaign_never_recorded_is_explained_not_left_blank():
    """Guarded endpoints have no ranking key; the column must say so."""
    joined = support.historical_join({"role": inventory_lib.ROLE_ENDPOINT,
                                      "historical_metrics_source": {"ranking": None}})
    assert joined["ranking"]["value"] is None
    assert "records no 'ranking'" in joined["ranking"]["reason"]


def test_a_v1_row_with_no_validation_record_carries_that_reason():
    joined = support.historical_join({
        "role": inventory_lib.ROLE_V1_ENDPOINT,
        "historical_metrics_source": {"reason": "validation_records.json holds no row at 600"}})
    assert joined["validation"]["reason"].startswith("validation_records.json holds no row")


# ---------------------------------------------------------------------------
# paired method differences
# ---------------------------------------------------------------------------

def endpoint_record(identifier, arm, seed, budget):
    return {"id": identifier, "role": inventory_lib.ROLE_ENDPOINT, "arm_id": arm, "seed": seed,
            "nominal_budget_gpu_seconds": budget}


def test_methods_are_compared_only_within_a_seed_and_budget():
    rows = 200
    generator = np.random.default_rng(0)
    base = generator.normal(size=rows)
    drops = {"a": base, "b": base + 0.5, "c": generator.normal(size=rows)}
    records = [endpoint_record("a", "ipo_tau0p1", 1, 600.0),
               endpoint_record("b", "continued_sft", 1, 600.0),
               endpoint_record("c", "ipo_tau0p1", 2, 600.0)]
    matrices = {1: scoring.bootstrap_index_matrix(rows, draws=100, seed=1),
                2: scoring.bootstrap_index_matrix(rows, draws=100, seed=2)}
    block = support.paired_method_differences(
        drops, inventory_records=records, statistics={"confidence": 0.95}, matrices=matrices)
    assert block["count"] == 1, "no cross-seed comparison is formed"
    (label, comparison), = block["comparisons"].items()
    assert comparison["mean"] == pytest.approx(-0.5)
    assert comparison["ci_low"] == pytest.approx(-0.5)      # a constant offset, paired
    assert {comparison["left_arm"], comparison["right_arm"]} == {"ipo_tau0p1", "continued_sft"}
    assert "seed1" in label and "budget600" in label


def test_two_endpoints_of_the_same_arm_are_not_a_method_comparison():
    rows = 50
    drops = {"a": np.zeros(rows), "b": np.ones(rows)}
    records = [endpoint_record("a", "ipo_tau0p1", 1, 600.0),
               endpoint_record("b", "ipo_tau0p1", 1, 600.0)]
    block = support.paired_method_differences(
        drops, inventory_records=records, statistics={"confidence": 0.95},
        matrices={1: scoring.bootstrap_index_matrix(rows, draws=20, seed=1)})
    assert block["count"] == 0


# ---------------------------------------------------------------------------
# reruns verify; they do not refresh
# ---------------------------------------------------------------------------

def test_a_rerun_with_identical_science_and_new_timings_is_accepted(tmp_path):
    context = context_for(tmp_path)
    target = context.run.path("summaries/checkpoints.json")
    first = {"scored": 3, "generated_at": "2026-09-19T10:00:00+00:00",
             "timings": {"wall_seconds": 800.0}}
    digest, state = support.require_new_or_identical(target, first, what="summary",
                                                     context=context, kind="summary")
    assert state == "written"
    again = {"scored": 3, "generated_at": "2026-09-19T18:00:00+00:00",
             "timings": {"wall_seconds": 12.0}}
    _, state = support.require_new_or_identical(target, again, what="summary", context=context,
                                                kind="summary")
    assert state == "verified_identical_scientific_content"
    on_disk = paths.read_json(target)
    assert on_disk["timings"]["wall_seconds"] == 800.0, "the original cost is the record"
    assert paths.sha256_file(target) == digest


def test_a_rerun_that_changes_the_science_is_refused(tmp_path):
    context = context_for(tmp_path)
    target = context.run.path("decision.json")
    support.require_new_or_identical(target, {"outcome": "no_escalation_at_this_resolution"},
                                     what="decision", context=context, kind="decision")
    with pytest.raises(ValueError, match="different scientific content"):
        support.require_new_or_identical(target, {"outcome": "escalate"}, what="decision",
                                         context=context, kind="decision")


def test_original_stage_timings_are_preserved_across_a_rerun(tmp_path):
    context = context_for(tmp_path)
    clock = paths.StageClock()
    clock.charge("inference", 120.0)
    support.record_timings(context, "score", clock)
    rerun = paths.StageClock()
    rerun.charge("inference", 0.5)
    support.record_timings(context, "score", rerun)
    assert paths.read_json(context.run.path("timings/score.json"))[
        "inference_seconds"] == pytest.approx(120.0)
    assert paths.read_json(context.run.path("timings/score.rerun.json"))[
        "inference_seconds"] == pytest.approx(0.5)


def test_verification_reports_an_artifact_edited_after_completion(tmp_path):
    context = context_for(tmp_path)
    context.config = {}
    target = context.run.path("decision.json")
    support.require_new_or_identical(target, {"outcome": "escalate", "timings": {}},
                                     what="decision", context=context, kind="decision")
    assert support.verify_outputs(context)["immutable"] is True
    paths.write_json(target, {"outcome": "no_escalation_at_this_resolution", "timings": {}})
    report = support.verify_outputs(context)
    assert report["immutable"] is False
    assert any("bytes differ" in problem["problem"] for problem in report["problems"])


def write_score_shard(context, identifier):
    """One completed score shard, registered in the run's completion manifest."""
    prefix = support.shard_directory("scores", identifier)
    return paths.write_shard(
        context.run.path(prefix), "sequence_scores",
        {"drop": np.zeros(3), "draw_index": np.arange(3, dtype=np.int64)},
        {"record_kind": "sequence_scores", "identity": {"state_digest": "d" * 64},
         "timings": {"wall_seconds": 1.0}},
        order="bank draw order 0..N-1", logical_prefix=prefix, run_root=context.run.run_root)


def test_a_partial_shard_makes_verification_fail(tmp_path):
    context = context_for(tmp_path)
    write_score_shard(context, "example")
    assert support.verify_outputs(context)["immutable"] is True
    (context.run.path("scores/example") / "sequence_scores.npz").write_bytes(b"truncated")
    assert support.verify_outputs(context)["immutable"] is False


# ---------------------------------------------------------------------------
# the v1 root is accepted by the digests the guarded campaign recorded
# ---------------------------------------------------------------------------

def make_v1_root(tmp_path, name, content=b"weights"):
    root = tmp_path / name
    (root / "policy_sft_seed1").mkdir(parents=True)
    (root / "policy_sft_seed1/epoch_3.pt").write_bytes(content)
    (root / "base_selection.json").write_text(json.dumps({"selected": {
        "policy_sft_seed1": {"checkpoint": f"outputs/{name}/policy_sft_seed1/epoch_3.pt",
                             "epoch": 3, "kind": "policy",
                             "sha256": paths.sha256_bytes(content)}}}), encoding="utf-8")
    return root


def test_the_v1_root_is_accepted_only_when_its_parents_hash_to_the_guarded_record(tmp_path):
    right = make_v1_root(tmp_path, "her2_posttrain_20260918", b"the real parent")
    wrong = make_v1_root(tmp_path, "other_posttrain", b"different weights")
    expected = {1: paths.sha256_bytes(b"the real parent")}
    resolved = support.resolve_original_root(
        logical="outputs/her2_posttrain_20260918", anchor="her2_posttrain_20260918",
        candidates=[right], base_relative="base_selection.json", expected_parents=expected)
    assert resolved.local_path == right
    with pytest.raises(ValueError, match="holds the selected parents"):
        support.resolve_original_root(
            logical="outputs/her2_posttrain_20260918", anchor="her2_posttrain_20260918",
            candidates=[tmp_path / "absent"], base_relative="base_selection.json",
            expected_parents=expected)
    with pytest.raises(ValueError, match="holds the selected parents"):
        support.resolve_original_root(
            logical="outputs/other_posttrain", anchor="other_posttrain",
            candidates=[wrong], base_relative="base_selection.json", expected_parents=expected)


def test_the_guarded_record_is_the_authority_for_the_parent_digests():
    stages = {1: {"parent_draw_references": {"1": {"parent_sha256": "a" * 64}}},
              2: {"parent_draw_references": {"1": {"parent_sha256": "a" * 64}}}}
    assert inventory_lib.parent_digests_from_stages(stages) == {1: "a" * 64}
    stages[2]["parent_draw_references"]["1"]["parent_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different parents for seed 1"):
        inventory_lib.parent_digests_from_stages(stages)


# ---------------------------------------------------------------------------
# frozen sources cover the code that actually computes the numbers
# ---------------------------------------------------------------------------

def test_the_frozen_source_list_includes_the_imported_scientific_modules():
    frozen = set(support.AUDIT_SOURCE_FILES)
    for module in ("her2_policy", "her2_data", "her2_preferences", "her2_runtime",
                   "her2_guarded_eval", "her2_guarded_trajectory", "her2_ches",
                   "her2_objectives", "her2_support", "her2_support_paths",
                   "her2_support_inventory", "her2_support_scoring"):
        assert f"src/smallAntibodyGen/experiments/{module}.py" in frozen, module
    assert "scripts/audit_her2_support.py" in frozen
    # ``expected_trajectories`` resolves arm identities through her2_objectives and
    # the raw root is accepted by digests parsed by benchmarks/provenance; both
    # decide what is enumerated and which tree is scored.
    assert "src/smallAntibodyGen/benchmarks/provenance.py" in frozen


def test_the_frozen_source_list_is_closed_under_module_level_imports():
    """A pinned module that imports an unpinned one is only half pinned."""
    import ast

    root = Path(__file__).resolve().parents[3]
    frozen = set(support.AUDIT_SOURCE_FILES)
    unpinned = []
    for logical in sorted(frozen):
        if not logical.startswith("src/smallAntibodyGen/experiments/"):
            continue
        tree = ast.parse((root / logical).read_text(encoding="utf-8"))
        for node in tree.body:                       # module level only, not lazy imports
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            names = [node.module] if node.module else [alias.name for alias in node.names]
            for name in names:
                candidate = f"src/smallAntibodyGen/experiments/{name}.py"
                if (root / candidate).is_file() and candidate not in frozen:
                    unpinned.append((logical, candidate))
    assert unpinned == []


def test_every_frozen_source_exists_in_the_repository():
    root = Path(__file__).resolve().parents[3]
    missing = [name for name in support.AUDIT_SOURCE_FILES if not (root / name).is_file()]
    assert missing == []


def test_the_config_declares_every_setting_the_coordinator_reads():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/experiments/her2_support_audit.json").read_text(
        encoding="utf-8"))
    for section in ("populations", "tolerances", "statistics", "decision", "inference", "ches",
                    "reporting", "roots", "historical"):
        assert section in config
    assert config["ches"]["control_arm"] == "continued_sft"
    assert config["decision"]["methods"] == ["continued_sft", "ipo_tau0p1"]
    assert config["decision"]["endpoint_gpu_seconds"] == 600.0
    assert config["statistics"]["bootstrap_draws"] == 2000
    assert support._placeholders(config) == []


def test_the_narrow_lf_rules_cover_the_audit_evidence():
    root = Path(__file__).resolve().parents[3]
    rules = (root / ".gitattributes").read_text(encoding="utf-8")
    assert "reference/evidence/her2-support-audit-2026-09-19/** text eol=lf" in rules
    assert "configs/experiments/her2_support_audit.json text eol=lf" in rules


def test_every_frozen_source_has_an_explicit_checkout_rule():
    """core.autocrlf is on here: without a rule, a clone re-hashes the frozen bytes."""
    root = Path(__file__).resolve().parents[3]
    rules = (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    pinned = {line.split(" ", 1)[0] for line in rules
              if line.strip() and not line.startswith("#") and "eol=lf" in line}
    missing = [logical for logical in support.AUDIT_SOURCE_FILES
               if logical not in pinned and logical not in support.LEGACY_RAW_SOURCE_FILES]
    assert missing == [], f"no LF rule pins {missing}"
    for logical in support.LEGACY_RAW_SOURCE_FILES:
        assert any(line.startswith(f"{logical} -text") for line in rules)
    assert support._relative(root / "configs/experiments/her2_support_audit.json", root) in pinned


def test_frozen_sources_preserve_the_declared_historical_bytes():
    root = Path(__file__).resolve().parents[3]
    crlf = [logical for logical in support.AUDIT_SOURCE_FILES
            if b"\r\n" in (root / logical).read_bytes()]
    assert sorted(crlf) == sorted(support.LEGACY_RAW_SOURCE_FILES)


def test_v1_validation_follows_the_verified_selection_freeze(tmp_path):
    original = tmp_path / "original"
    context = context_for(tmp_path)
    context.config = {"historical": {
        "v1_final_integrity": "integrity.json",
        "v1_selection_frozen_relative": "validation_math/selection_frozen.json"}}
    context.historical = {"original_anchor": "v1"}
    context.root = lambda name: SimpleNamespace(path=lambda rel: original / rel)
    rows = {"dpo_seed1_budget1800": {"method": "dpo", "seed": 1, "budget_seconds": 1800}}
    records_sha = paths.write_json(original / "validation_math/validation_records.json", rows)
    numerical_sha = paths.write_json(original / "validation_math/numerical_backend.json",
                                     {"attention_backend": "MATH"})
    freeze_sha = paths.write_json(original / "validation_math/selection_frozen.json", {
        "assay_outcomes_read": False,
        "validation_records": {"path": "C:\\legacy\\outputs\\v1\\validation_math\\validation_records.json",
                               "sha256": records_sha},
        "numerical_evaluation": {"path": "outputs/v1/validation_math/numerical_backend.json",
                                 "sha256": numerical_sha}})
    paths.write_json(tmp_path / "integrity.json", {"selection_freeze_sha256": freeze_sha,
                                                  "numerical_manifest_sha256": numerical_sha})
    # An obsolete partial document cannot win merely by existing at the old path.
    paths.write_json(original / "continuation/validation_records.json", {})
    assert support.v1_validation_records_relative(context) == "validation_math/validation_records.json"
    authority = support.v1_validation_authority(context)
    assert authority["validation_records"]["sha256"] == records_sha
    paths.write_json(original / "validation_math/validation_records.json", {})
    with pytest.raises(ValueError, match="frozen v1 validation_records bytes"):
        support.v1_validation_authority(context)


def test_published_figures_are_never_marked_as_text():
    """A ``text eol=lf`` rule over a PNG corrupts it on checkout."""
    root = Path(__file__).resolve().parents[3]
    rules = [line for line in (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    for line in rules:
        pattern, attributes = line.split(" ", 1)
        if pattern.endswith(".png"):
            assert "text" not in attributes.split(), line
        if "eol=lf" in attributes:
            assert not pattern.endswith(".png"), line


def test_the_original_root_acceptance_text_matches_the_implemented_rule():
    """The config described a launch-manifest probe; the code probes the parents."""
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/experiments/her2_support_audit.json").read_text(
        encoding="utf-8"))
    acceptance = config["roots"]["original"]["acceptance"]
    assert "base_selection" in acceptance and "guarded" in acceptance
    assert "launch manifest" not in acceptance
    assert "launch manifest" in config["roots"]["original"]["acceptance_note"]


def test_the_publication_section_declares_where_a_published_file_goes():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/experiments/her2_support_audit.json").read_text(
        encoding="utf-8"))
    publication = config["publication"]
    for key in ("root", "data_directory", "figure_directory", "manifest"):
        assert publication[key]
    assert publication["data_directory"].startswith("evidence/")
    assert config["published_report"].startswith(publication["root"] + "/")
