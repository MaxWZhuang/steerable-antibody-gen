"""Runner-level contracts: allocation arithmetic, provenance, staging, selection, freeze.

The end-to-end tests build a miniature guarded campaign out of a few bytes on
disk -- trajectory documents, endpoint records, checkpoint files -- and drive the
real freeze, verification and selection code over it. Nothing fits, samples or
downloads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_guard as guard
from smallAntibodyGen.experiments import her2_guarded_eval as selection_lib
from smallAntibodyGen.experiments import her2_lineage as lineage
from smallAntibodyGen.experiments import her2_policy as policy_lib
from smallAntibodyGen.experiments.her2_preferences import array_digest
from smallAntibodyGen.experiments.her2_runtime import HER2_CODE_FILES, code_digests, sha256

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (1, 2, 3)
BUDGET = 180.0
#: The miniature campaign's fixed revision config hash and scaffold prefix. Both
#: enter the parent-reference identity, so they are named once here and the
#: fixtures and the runner have to agree on them.
CONFIG_SHA = "c" * 64
PREFIX = "GUARDEDPREFIX"


def tiny_pairs():
    """Four real ordered validation pairs: enough to hash, digest and average over."""
    chosen = data.encode_cores(["ACDEFGHIKL", "ACDEFGHIKM", "WCDEFGHIKL", "YCDEFGHIKM"])
    rejected = data.encode_cores(["ACDEFGHIKN", "ACDEFGHIKP", "WCDEFGHIKQ", "YCDEFGHIKR"])
    return {"chosen_index": chosen, "rejected_index": rejected, "pairs": 4}

#: The tiny grid the miniature campaign declares. It exists because a fixture that
#: builds two arm-seeds while the config declares fifty-four proves nothing about
#: completeness: the freeze has to be checked against a grid it can actually
#: satisfy, and then against one it cannot.
TINY_STAGES = [
    {"stage": 1, "name": "tiny stage 1",
     "arms": [{"objective": "continued_sft", "coefficients": {}},
              {"objective": "dpo", "coefficients": {"beta": 0.1}}]},
    {"stage": 2, "name": "tiny stage 2",
     "arms": [{"objective": "continued_sft", "coefficients": {}, "reused_from_stage": 1},
              {"objective": "dpop", "coefficients": {"beta": 0.1, "lambda": 1.0}}]},
    {"stage": 3, "name": "tiny stage 3",
     "arms": [{"objective": "continued_sft", "coefficients": {}, "reused_from_stage": 1},
              {"objective": "ipo", "coefficients": {"tau": 0.5}}]},
]


def tiny_config(config):
    """The real config schema, cut down to a grid three seeds of fixtures can satisfy."""
    small = json.loads(json.dumps(config))
    small["seeds"] = list(SEEDS)
    small["budgets_gpu_seconds"] = [180]
    small["stages"] = json.loads(json.dumps(TINY_STAGES))
    small["allocation"] = {
        "total_gpu_seconds": 4 * len(SEEDS) * BUDGET,
        "stage_gpu_seconds": {"1": 2 * len(SEEDS) * BUDGET, "2": len(SEEDS) * BUDGET,
                              "3": len(SEEDS) * BUDGET},
        "basis": "tiny synthetic grid: 4 fitted configurations x 3 seeds x the 180 s maximum"}
    return small


def stage_arms(config, stage):
    """The fitted arms of one stage, resolved exactly as the runner resolves them."""
    return [arm for arm in selection_lib.declared_arms(config)
            if arm["stage"] == int(stage) and not arm["reused_from_stage"]]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.is_file():                                   # pragma: no cover
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config():
    return json.loads((ROOT / "configs/experiments/her2_guarded_continuation.json").read_text(
        encoding="utf-8"))


@pytest.fixture(scope="module")
def original_config():
    return json.loads((ROOT / "configs/experiments/her2_posttrain.json").read_text(
        encoding="utf-8"))


@pytest.fixture(scope="module")
def guarded():
    return load_script("posttrain_her2_guarded")


# ---------------------------------------------------------------------------
# the declared grid and its arithmetic
# ---------------------------------------------------------------------------

def test_the_declared_allocation_reconciles_stage_by_stage(config):
    document = selection_lib.require_allocation_reconciles(config)
    assert document["configurations_fitted"] == 18
    assert document["seeds"] == 3
    assert document["total_charged_training_gpu_seconds"] == 32400
    assert {key: value["allocation_gpu_seconds"] for key, value in document["stages"].items()} == {
        "1": 9000, "2": 10800, "3": 12600}


def test_the_allocation_is_the_max_budget_not_the_sum_of_the_three(config):
    """One trajectory drops three checkpoints; charging it three times inflates by 1.9x."""
    document = selection_lib.stage_allocation(config)
    assert document["incorrect_sum_of_budgets"] == 18 * 3 * (180 + 360 + 600)
    assert document["total_charged_training_gpu_seconds"] != document["incorrect_sum_of_budgets"]
    assert document["total_charged_training_gpu_seconds"] == 18 * 3 * 600


def test_a_config_whose_totals_do_not_match_the_grid_fails_closed(config):
    broken = json.loads(json.dumps(config))
    broken["allocation"]["total_gpu_seconds"] = 5400
    with pytest.raises(ValueError, match="does not reconcile"):
        selection_lib.require_allocation_reconciles(broken)


def test_the_grid_is_the_declared_one_and_the_control_is_fitted_once(config):
    arms = selection_lib.declared_arms(config)
    fitted = [arm for arm in arms if not arm["reused_from_stage"]]
    assert len(fitted) == 18
    assert sum(1 for arm in fitted if arm["objective"] == "continued_sft") == 1
    assert sorted(arm["coefficients"]["beta"] for arm in fitted if arm["objective"] == "dpo") == \
        [0.1, 0.5, 1.0, 5.0]
    assert sorted(arm["coefficients"]["tau"] for arm in fitted if arm["objective"] == "ipo") == \
        [0.1, 0.5, 1.0, 5.0]
    reused = [arm for arm in arms if arm["reused_from_stage"]]
    assert {arm["stage"] for arm in reused} == {2, 3}
    assert all(arm["arm_id"] == "continued_sft" for arm in reused)


def test_fitting_the_same_arm_twice_is_refused(config):
    broken = json.loads(json.dumps(config))
    broken["stages"][1]["arms"].append({"objective": "dpo", "coefficients": {"beta": 0.1}})
    with pytest.raises(ValueError, match="declared as fitted twice"):
        selection_lib.declared_arms(broken)


def test_the_budgets_are_the_three_declared_ones_with_no_higher_control(config):
    assert config["budgets_gpu_seconds"] == [180, 360, 600]
    assert list(selection_lib.BUDGETS) == [180.0, 360.0, 600.0]


# ---------------------------------------------------------------------------
# the gate as the config declares it
# ---------------------------------------------------------------------------

def test_the_config_gate_has_no_subset_knob_at_all(config):
    """CX-01: the monitor-pairs key is deleted from the schema, not defaulted to 'all'."""
    gate = config["gate"]
    assert "monitor_pairs" not in gate
    assert not any("subset" in key or "sample" in key for key in gate)
    assert gate["threshold_nats_per_sequence"] == 1.0
    assert gate["threshold_nats_per_residue"] == 0.1
    assert gate["update_interval"] == 25 and gate["gpu_second_interval"] == 5.0
    assert gate["first_update"] == 1
    assert "every arm" in gate["applies_to"]
    assert "every fixed validation pair" in gate["population"]


def test_no_threshold_ladder_survives_anywhere_in_the_config(config):
    """A run stopped at 1.0 observes nothing about 2 or 5; nothing may imply otherwise."""
    text = json.dumps(config)
    assert "ladder" not in text
    assert "dose_response" not in text and "dose-response" not in text


def test_the_matched_settings_are_byte_equal_to_the_original_campaigns(config, original_config):
    """Matched means matched: the same optimizer and the same pairing, checkable."""
    original = original_config["continuation"]
    assert config["optimization"] == original["optimization"]
    assert config["pairing"] == original["pairing"]
    assert config["expected_counts"] == original["expected_counts"]
    assert config["batch_sequences"] == original["batch_sequences"]
    assert config["diversity_gates"] == original["diversity_gates"]


def test_the_config_diversity_gates_match_the_preregistered_constants(config):
    from smallAntibodyGen.experiments import her2_eval as evaluation
    assert config["diversity_gates"] == dict(evaluation.DIVERSITY_GATES)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def test_the_revision_adds_files_and_edits_none_of_the_originals():
    """The original freeze binds HER2_CODE_FILES; adding to that tuple would break it."""
    for relative in lineage.GUARDED_CODE_FILES:
        assert (ROOT / relative).is_file(), relative
        assert relative not in HER2_CODE_FILES


def test_the_original_scientific_code_still_hashes_as_the_freeze_recorded_it():
    freeze = ROOT / "outputs/her2_posttrain_20260918/base_selection.json"
    if not freeze.is_file():                                  # pragma: no cover
        pytest.skip("the original freeze is under the git-ignored output root")
    recorded = json.loads(freeze.read_text(encoding="utf-8"))["code_digests"]
    assert code_digests(ROOT, HER2_CODE_FILES) == recorded, (
        "an original scientific source changed; the parents would no longer be readable "
        "under their own provenance")


def test_a_guarded_freeze_cannot_be_read_as_a_selection_freeze(tmp_path):
    document = {"schema_version": lineage.GUARDED_SELECTION_SCHEMA, "stage": 1,
                "stage_marker": lineage.GUARDED_STAGE,
                "config_sha256": "a" * 64, "selected": {"x": {"checkpoint": "x", "sha256": "y"}}}
    path = tmp_path / "stage1_complete.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert lineage.assert_cannot_unlock(document) is True
    with pytest.raises(ValueError, match="Unsupported selection schema"):
        data.read_selection_freeze(path, root=tmp_path, expected_stage=data.SELECTION_STAGE_FINAL)


def test_the_inherited_block_is_separate_from_the_revision_block(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps({"b": 2}), encoding="utf-8")
    freeze_path = tmp_path / "base_selection.json"
    freeze_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lineage, "code_digests",
                        lambda root, files=HER2_CODE_FILES: {name: "0" * 64 for name in files})
    identity = lineage.guarded_identity(
        tmp_path, config_path=config_path, original_config_path=original_path,
        base_selection_path=freeze_path, source_digests={"s": "1" * 64}, device="cpu",
        batch_sequences=128)
    assert identity["revision"]["config_sha256"] == sha256(config_path)
    assert identity["inherited"]["original_config_sha256"] == sha256(original_path)
    assert identity["inherited"]["original_config_sha256"] != identity["revision"]["config_sha256"]
    assert "did not produce them" in identity["inherited"]["note"]


def test_an_interrupted_guarded_run_is_refused_and_never_reclaimed(tmp_path):
    identity = {"revision": {"code_digests": {"x.py": "1"}}}
    ledger = lineage.GuardedRunLedger(tmp_path / "run", identity)
    assert ledger.claim() == "start"
    with pytest.raises(ValueError, match="never overwritten"):
        lineage.GuardedRunLedger(tmp_path / "run", identity).claim()
    ledger.finish({"status": "stopped"})
    assert lineage.GuardedRunLedger(tmp_path / "run", identity).claim() == "completed"
    changed = lineage.GuardedRunLedger(tmp_path / "run", {"revision": {"code_digests": {"x.py": "2"}}})
    with pytest.raises(ValueError, match="different identity"):
        changed.claim()


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def stratum(n, positives, average_precision):
    return {"n": n, "positives": positives, "negatives": n - positives,
            "average_precision": average_precision}


def endpoint(arm_id, seed, *, budget=180.0, strata=(0.9, 0.8, 0.7), objective="dpo",
             reached=True, gate=True, diversity=True):
    return {"arm_id": arm_id, "objective": objective, "coefficients": {"beta": 0.1},
            "seed": seed, "nominal_budget": budget, "reached": reached, "gate_passed": gate,
            "diversity_eligible": diversity,
            "val_strata": {"0": {"n": 0, "note": "empty_stratum"},
                           "1": stratum(70856, 23000, strata[0]),
                           "2": stratum(6649, 2000, strata[1]),
                           ">=3": stratum(1147, 300, strata[2])}}


def test_selection_uses_the_macro_stratum_average_not_the_aggregate():
    """The aggregate is dominated by distance 1; a macro average cannot be rescued by it."""
    endpoints = []
    for seed in SEEDS:
        endpoints.append(endpoint("dpo_beta0p1", seed, strata=(0.981, 0.862, 0.656)))
        endpoints.append(endpoint("dpo_beta0p5", seed, strata=(0.983, 0.869, 0.683)))
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                              seeds=SEEDS)
    assert result["selected"] == "dpo_beta0p5", "better in every stratum"
    assert result["selected_record"]["strata"] == ["1", "2", ">=3"]
    assert result["selection_inputs"] == ["validation"]


def test_stratum_zero_is_reported_but_never_averaged_in():
    endpoints = [endpoint("dpo_beta0p1", seed) for seed in SEEDS]
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                              seeds=SEEDS)
    assert result["strata"] == ["1", "2", ">=3"]
    assert result["candidates"][0]["mean_macro_average_precision"] == pytest.approx(
        (0.9 + 0.8 + 0.7) / 3)


def test_a_single_class_stratum_is_dropped_because_its_ap_is_a_constant():
    per_stratum = {"1": stratum(100, 30, 0.8), "2": stratum(50, 50, 1.0),
                   ">=3": stratum(10, 3, 0.4)}
    assert selection_lib.stratum_mask(per_stratum) == ("1", ">=3")
    value, worst, mask = selection_lib.macro_average_precision(per_stratum)
    assert value == pytest.approx(0.6) and worst == pytest.approx(0.4)
    assert mask == ("1", ">=3")


def test_one_ineligible_seed_makes_the_whole_arm_ineligible():
    endpoints = [endpoint("dpo_beta0p1", 1), endpoint("dpo_beta0p1", 2),
                 endpoint("dpo_beta0p1", 3, gate=False),
                 endpoint("dpo_beta0p5", 1, strata=(0.5, 0.5, 0.5)),
                 endpoint("dpo_beta0p5", 2, strata=(0.5, 0.5, 0.5)),
                 endpoint("dpo_beta0p5", 3, strata=(0.5, 0.5, 0.5))]
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                              seeds=SEEDS)
    assert result["selected"] == "dpo_beta0p5", "the better arm lost a seed to the gate"
    assert result["ineligible"]["dpo_beta0p1"]["seeds"]["3"] == ["likelihood_gate"]


def test_a_missing_seed_is_not_a_partial_score():
    endpoints = [endpoint("dpo_beta0p1", 1), endpoint("dpo_beta0p1", 2)]
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                              seeds=SEEDS)
    assert result["selected"] is None
    assert result["ineligible"]["dpo_beta0p1"] == {"reason": "missing_seeds", "seeds": [3]}
    assert "none eligible" in result["reason"]


def test_an_unreached_budget_is_never_backfilled():
    endpoints = [endpoint("dpo_beta0p1", seed, budget=600.0, reached=False) for seed in SEEDS]
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=600.0,
                                              seeds=SEEDS)
    assert result["selected"] is None
    assert result["ineligible"]["dpo_beta0p1"]["seeds"]["1"] == ["budget_not_reached"]


def test_a_failed_diversity_gate_is_reported_separately_from_the_likelihood_gate():
    endpoints = [endpoint("dpo_beta0p1", seed, diversity=False) for seed in SEEDS]
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                              seeds=SEEDS)
    assert result["ineligible"]["dpo_beta0p1"]["seeds"]["2"] == ["diversity_gates"]


def test_ties_break_on_the_worst_stratum_then_on_the_name():
    """Exact binary fractions, so the tie is a real tie and not a rounding artifact."""
    endpoints = []
    for seed in SEEDS:
        endpoints.append(endpoint("dpo_b", seed, strata=(0.625, 0.125, 0.125)))
        endpoints.append(endpoint("dpo_a", seed, strata=(0.5, 0.25, 0.125)))
        endpoints.append(endpoint("dpo_c", seed, strata=(0.375, 0.25, 0.25)))
    result = selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                              seeds=SEEDS)
    means = {candidate["arm_id"]: candidate["mean_macro_average_precision"]
             for candidate in result["candidates"]}
    assert len(set(means.values())) == 1, "all three tie exactly on the primary statistic"
    assert result["selected"] == "dpo_c", "highest worst stratum wins the tie"
    exact = [endpoint(name, seed, strata=(0.5, 0.25, 0.125))
             for name in ("dpo_z", "dpo_a") for seed in SEEDS]
    assert selection_lib.select_coefficient(exact, objective="dpo", budget=180.0,
                                            seeds=SEEDS)["selected"] == "dpo_a", \
        "identical on both statistics: the lexicographic id decides, under any dict order"


def test_a_differing_stratum_mask_aborts_rather_than_dropping_a_stratum():
    endpoints = [endpoint("dpo_a", seed) for seed in SEEDS]
    endpoints += [endpoint("dpo_b", seed) for seed in SEEDS]
    for record in endpoints:
        if record["arm_id"] == "dpo_b":
            record["val_strata"][">=3"] = stratum(1147, 1147, 1.0)
    with pytest.raises(ValueError, match="valid stratum mask"):
        selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0, seeds=SEEDS)


def test_a_record_carrying_a_reserved_outcome_is_refused_by_the_selector():
    endpoints = [endpoint("dpo_beta0p1", seed) for seed in SEEDS]
    endpoints[0]["test_average_precision"] = 0.97
    with pytest.raises(ValueError, match="Selection reads validation only"):
        selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0, seeds=SEEDS)
    endpoints[0].pop("test_average_precision")
    endpoints[1]["spr_spearman"] = 0.5
    with pytest.raises(ValueError, match="Selection reads validation only"):
        selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0, seeds=SEEDS)


def test_ordinary_field_names_are_not_mistaken_for_reserved_ones():
    endpoints = [dict(endpoint("dpo_beta0p1", seed), last_passing={"update": 3},
                      latest_check=4) for seed in SEEDS]
    assert selection_lib.select_coefficient(endpoints, objective="dpo", budget=180.0,
                                            seeds=SEEDS)["selected"] == "dpo_beta0p1"


def test_matched_control_deltas_are_descriptive_and_matched_on_seed_and_budget():
    endpoints = []
    for seed in SEEDS:
        endpoints.append(endpoint("continued_sft", seed, objective="continued_sft",
                                  strata=(0.9, 0.8, 0.7)))
        endpoints.append(endpoint("dpo_beta0p1", seed, strata=(0.9, 0.8, 0.8)))
    table = selection_lib.matched_control_deltas(endpoints)
    assert len(table["rows"]) == 3
    assert all(row["delta"] == pytest.approx(0.1 / 3) for row in table["rows"])
    assert "not a confirmatory affinity claim" in table["interpretation"]


def test_a_raw_matched_delta_carries_its_eligibility_and_the_eligible_subset_is_separate():
    """CX-17: an ineligible endpoint can still top the descriptive table. Both are reported."""
    endpoints = []
    for seed in SEEDS:
        endpoints.append(endpoint("continued_sft", seed, objective="continued_sft",
                                  strata=(0.9, 0.8, 0.7)))
        endpoints.append(endpoint("dpo_beta0p1", seed, strata=(0.99, 0.89, 0.79),
                                  diversity=False))
        endpoints.append(endpoint("dpo_beta0p5", seed, strata=(0.91, 0.81, 0.71)))
    table = selection_lib.matched_control_deltas(endpoints)
    assert len(table["rows"]) == 6, "every raw difference is still described"
    ineligible = [row for row in table["rows"] if row["arm_id"] == "dpo_beta0p1"]
    assert all(row["delta"] > 0 for row in ineligible), "and it is the biggest one"
    assert all(row["eligible"] is False for row in ineligible)
    assert all(row["arm_ineligible_reasons"] == ["diversity_gates"] for row in ineligible)
    assert {row["arm_id"] for row in table["eligible_rows"]} == {"dpo_beta0p5"}
    assert "Only `eligible_rows`" in table["eligibility_note"]


def test_an_arm_with_no_matched_control_at_its_budget_is_reported_unmatched():
    endpoints = [endpoint("dpo_beta0p1", seed) for seed in SEEDS]
    endpoints.append(endpoint("continued_sft", 1, objective="continued_sft"))
    table = selection_lib.matched_control_deltas(endpoints)
    assert [row["seed"] for row in table["rows"]] == [1]
    assert sorted(row["seed"] for row in table["unmatched"]) == [2, 3]
    assert "nothing is substituted" in table["unmatched"][0]["reason"]


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------

def test_the_declared_grid_names_every_arm_seed_the_stage_must_produce(config):
    small = tiny_config(config)
    assert selection_lib.expected_trajectories(small, 1) == sorted(
        [f"{arm}_seed{seed}" for arm in ("continued_sft", "dpo_beta0p1") for seed in SEEDS])
    assert selection_lib.expected_trajectories(small, 2) == sorted(
        f"dpop_beta0p1_lambda1_seed{seed}" for seed in SEEDS), "the reused control is not fitted"
    # The real grid, for the stage the campaign actually declares.
    assert len(selection_lib.expected_trajectories(config, 1)) == 5 * 3
    assert len(selection_lib.expected_trajectories(config, 2)) == 6 * 3
    assert len(selection_lib.expected_trajectories(config, 3)) == 7 * 3


def test_a_cpu_continuation_is_refused_while_the_no_fit_stages_are_not(guarded):
    """CX-30: a GPU budget measured on CPU work is not a GPU budget."""
    assert guarded.require_production_device("cpu", ("inspect", "plan")) == "cpu"
    assert guarded.require_production_device("cuda", ("continue", "validate")) == "cuda"
    with pytest.raises(ValueError, match="fits on CUDA only"):
        guarded.require_production_device("cpu", ("inspect", "plan", "continue"))


def test_stage_one_needs_no_predecessor_but_stage_two_does(tmp_path):
    identity = {"revision": {"code_digests": {"a.py": "1"}}}
    assert selection_lib.require_previous_stage(tmp_path, 1, identity=identity, root=tmp_path) \
        is None
    with pytest.raises(ValueError, match="requires a verified stage1_complete.json"):
        selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)


def marker_document(identity, *, control_artifacts, artifacts=None, control_status=None,
                    control_endpoints=(), stage=1):
    """A stage marker in the shape the freeze writes one."""
    return {"schema_version": lineage.GUARDED_SELECTION_SCHEMA, "stage": stage,
            "stage_marker": lineage.GUARDED_STAGE, "identity": identity,
            "artifacts": artifacts or {},
            "control_artifacts": control_artifacts,
            "control_status": control_status or {"available": True, "source_stage": stage,
                                                 "endpoints": len(control_endpoints),
                                                 "trajectories": {}},
            "control_endpoints": list(control_endpoints)}


def test_stage_two_verifies_the_stage_one_control_bytes(tmp_path):
    identity = {"revision": {"code_digests": {"a.py": "1"}}}
    control = tmp_path / "control.pt"
    control.write_bytes(b"control weights")
    record = {"path": "control.pt", "sha256": sha256(control)}
    marker = marker_document(identity, artifacts={"a": record},
                             control_artifacts={"continued_sft_seed1_budget180": record},
                             control_endpoints=[{"arm_id": "continued_sft"}])
    (tmp_path / "stage1_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    assert selection_lib.require_previous_stage(tmp_path, 2, identity=identity,
                                                root=tmp_path)["stage"] == 1
    control.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="changed on disk"):
        selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)


def test_stage_two_refuses_a_marker_from_different_revision_code(tmp_path):
    control = tmp_path / "control.pt"
    control.write_bytes(b"control weights")
    marker = marker_document({"revision": {"code_digests": {"a.py": "1"}}},
                             control_artifacts={"c": {"path": "control.pt",
                                                      "sha256": sha256(control)}},
                             control_endpoints=[{"arm_id": "continued_sft"}])
    (tmp_path / "stage1_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="different revision code"):
        selection_lib.require_previous_stage(
            tmp_path, 2, identity={"revision": {"code_digests": {"a.py": "2"}}}, root=tmp_path)


def test_stage_two_refuses_a_predecessor_that_continued_different_parents(tmp_path):
    """CX-18: the revision hashes agree and the experiment is still a different one."""
    identity = {"revision": {"code_digests": {"a.py": "1"}},
                "inherited": {"parents": {"1": {"sha256": "a" * 64}}},
                "source_digests": {"buzz": "b" * 64}, "device": "cuda", "batch_sequences": 128}
    control = tmp_path / "control.pt"
    control.write_bytes(b"control weights")
    record = {"path": "control.pt", "sha256": sha256(control)}
    older = json.loads(json.dumps(identity))
    older["inherited"]["parents"]["1"]["sha256"] = "9" * 64
    marker = marker_document(older, control_artifacts={"c": record},
                             control_endpoints=[{"arm_id": "continued_sft"}])
    (tmp_path / "stage1_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="different campaign identity") as error:
        selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)
    assert "inherited" in str(error.value)
    # The same argument for the device and the batch size, which are not code hashes either.
    for key, value in (("device", "cpu"), ("batch_sequences", 64)):
        drifted = json.loads(json.dumps(identity))
        drifted[key] = value
        (tmp_path / "stage1_complete.json").write_text(
            json.dumps(marker_document(drifted, control_artifacts={"c": record},
                                       control_endpoints=[{"arm_id": "continued_sft"}])),
            encoding="utf-8")
        with pytest.raises(ValueError, match="different campaign identity"):
            selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)


def test_a_marker_reporting_no_control_must_say_which_trajectories_failed(tmp_path):
    """An all-stopped control is an outcome; a forgotten one is a bug. They differ here."""
    identity = {"revision": {"code_digests": {"a.py": "1"}}}
    silent = marker_document(identity, control_artifacts={}, control_status={"available": False})
    (tmp_path / "stage1_complete.json").write_text(json.dumps(silent), encoding="utf-8")
    with pytest.raises(ValueError, match="without saying which trajectories"):
        selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)
    explained = marker_document(identity, control_artifacts={}, control_status={
        "available": False, "reason": "every continued_sft trajectory stopped",
        "trajectories": {"continued_sft_seed1": {"status": "stopped"}}})
    (tmp_path / "stage1_complete.json").write_text(json.dumps(explained), encoding="utf-8")
    marker = selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)
    rows, note = selection_lib.reused_control_endpoints(marker, root=tmp_path, source_stage=1)
    assert rows == [] and note["available"] is False
    assert "reported unavailable" in note["note"]


def test_a_marker_with_no_control_status_at_all_is_refused(tmp_path):
    identity = {"revision": {"code_digests": {"a.py": "1"}}}
    marker = marker_document(identity, control_artifacts={})
    marker.pop("control_status")
    (tmp_path / "stage1_complete.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="records no control_status"):
        selection_lib.require_previous_stage(tmp_path, 2, identity=identity, root=tmp_path)


def test_a_stage_cannot_be_frozen_while_a_trajectory_is_still_running():
    with pytest.raises(ValueError, match="not terminal"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={"a": {"status": "completed",
                                                      "budgets": {"180.0": {}}},
                                                "b": {"status": "running",
                                                      "budgets": {"180.0": {}}}},
            artifacts={}, control_artifacts={"c": {}}, selection={}, allocation={},
            expected_trajectories=["a", "b"], budgets=[180.0])


def test_a_stage_missing_one_of_its_declared_runs_does_not_freeze():
    """CX-19: an empty or partial trajectory set is not a stage that ran."""
    with pytest.raises(ValueError, match="have no result"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={"a": {"status": "completed",
                                                      "budgets": {"180.0": {}}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["a", "b"], budgets=[180.0])
    with pytest.raises(ValueError, match="needs the declared arm x seed grid"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={}, artifacts={}, control_artifacts={},
            selection={}, allocation={}, expected_trajectories=[], budgets=[180.0])
    with pytest.raises(ValueError, match="not in the declared stage grid"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={"a": {"status": "completed",
                                                      "budgets": {"180.0": {}}},
                                                "z": {"status": "completed",
                                                      "budgets": {"180.0": {}}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["a"], budgets=[180.0])


def test_a_budget_that_is_neither_reached_nor_recorded_unreached_does_not_freeze():
    with pytest.raises(ValueError, match="neither reached nor recorded"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={"a": {"status": "completed",
                                                      "budgets": {"180.0": {}}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["a"], budgets=[180.0, 360.0])


def test_the_stage_marker_enumerates_stopped_trajectories_with_their_unreached_budgets():
    document = selection_lib.stage_complete_document(
        stage=1, identity={}, trajectories={
            "dpo_beta5_seed1": {"status": "stopped", "stop_reason": "parent_relative_likelihood_breach",
                                "updates": 58, "checks": 3, "budgets": {},
                                "budgets_not_reached": {"180.0": {}, "360.0": {}, "600.0": {}},
                                "last_passing": {"update": 25}, "cost": {}}},
        artifacts={}, control_artifacts={"c": {}}, selection={}, allocation={},
        expected_trajectories=["dpo_beta5_seed1"], budgets=[180.0, 360.0, 600.0],
        objectives=["dpo"])
    entry = document["trajectories"]["dpo_beta5_seed1"]
    assert entry["status"] == "stopped"
    assert entry["budgets_reached"] == []
    assert entry["budgets_not_reached"] == ["180.0", "360.0", "600.0"]
    assert document["declared_trajectories"] == ["dpo_beta5_seed1"]
    assert document["declared_objectives"] == ["dpo"]
    assert document["selection_read_test_or_assay"] is False
    assert document["prior_exposure"]["not_claimed"] == "that these assays are unseen"


def test_an_artificially_capped_trajectory_does_not_advance_the_stage():
    """CX-35: a max_updates cap is an artificial end, not a prespecified likelihood stop."""
    capped = {"dpo_beta5_seed1": {"status": "incomplete", "updates": 1, "checks": 2,
                                  "budgets": {},
                                  "budgets_not_reached": {"180.0": {"reached": False}}}}
    with pytest.raises(ValueError, match="neither completed their declared budgets"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories=capped, artifacts={}, control_artifacts={},
            selection={}, allocation={}, expected_trajectories=["dpo_beta5_seed1"],
            budgets=[180.0])
    failed = {"dpo_beta5_seed1": dict(capped["dpo_beta5_seed1"], status="failed")}
    with pytest.raises(ValueError, match="neither completed their declared budgets"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories=failed, artifacts={}, control_artifacts={},
            selection={}, allocation={}, expected_trajectories=["dpo_beta5_seed1"],
            budgets=[180.0])


def test_a_stop_for_an_undeclared_reason_does_not_advance_the_stage():
    """`stopped` means the gate stopped it. An IO failure wearing that label does not."""
    with pytest.raises(ValueError, match="a reason the gate does not declare"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={
                "dpo_beta5_seed1": {"status": "stopped",
                                    "stop_reason": "last_passing_snapshot_failure",
                                    "budgets": {},
                                    "budgets_not_reached": {"180.0": {}}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["dpo_beta5_seed1"], budgets=[180.0])


def test_budget_keys_must_be_exactly_the_declared_ones_and_disjoint():
    """CX-35: reached and not-reached are a partition of the declared budgets."""
    with pytest.raises(ValueError, match="recorded as reached AND as not reached"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={
                "a": {"status": "completed", "budgets": {"180.0": {}},
                      "budgets_not_reached": {"180.0": {}}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["a"], budgets=[180.0])
    with pytest.raises(ValueError, match="recorded but not declared"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={
                "a": {"status": "completed", "budgets": {"180.0": {}, "900.0": {}},
                      "budgets_not_reached": {}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["a"], budgets=[180.0])
    with pytest.raises(ValueError, match="marked completed with budgets"):
        selection_lib.stage_complete_document(
            stage=1, identity={}, trajectories={
                "a": {"status": "completed", "budgets": {"180.0": {}},
                      "budgets_not_reached": {"360.0": {}}}},
            artifacts={}, control_artifacts={}, selection={}, allocation={},
            expected_trajectories=["a"], budgets=[180.0, 360.0])


# ---------------------------------------------------------------------------
# an end-to-end miniature campaign through the real freeze
# ---------------------------------------------------------------------------

class Campaign:
    """A miniature on-disk stage: trajectories, journals, checkpoints, endpoint records.

    Faithful in every place the runner verifies, because a fixture that fakes the
    evidence tests nothing about the code that checks it:

    * the parents are real checkpoint files carrying their own state digests, and
      the frozen parent reference in each trajectory directory is a real ``.npz``
      whose identity is built by :func:`her2_guard.parent_reference_identity` from
      those parents and the actual ordered pairs;
    * ``monitor.jsonl`` is written by the real :class:`~her2_guard.GateMonitor`, so
      the per-check score vectors exist on disk, hash to what the verdict records,
      and reproduce the journalled D when it is recomputed;
    * each budget checkpoint is a real payload naming its arm, seed, budget,
      campaign identity and progress record.

    Only the model itself is absent: the scorer is injected, which is the one part
    a unit test may replace without replacing the thing under test.
    """

    def __init__(self, root, config, *, stage=1, parents=None):
        self.root = Path(root)
        self.output = self.root / "run"
        self.config = config
        self.stage = int(stage)
        self.pairs = tiny_pairs()
        self.parents = self.parent_entries()
        self.identity = {
            "schema_version": lineage.GUARDED_IDENTITY_SCHEMA,
            "revision": {"config_sha256": CONFIG_SHA,
                         "code_digests": {"scripts/posttrain_her2_guarded.py": "e" * 64}},
            "inherited": {"original_config_sha256": "o" * 64,
                          "original_code_digests": {"x.py": "1" * 64},
                          "parents": parents or {
                              str(seed): {"name": entry["name"],
                                          "checkpoint": entry["checkpoint"],
                                          "sha256": entry["sha256"]}
                              for seed, entry in sorted(self.parents.items())}},
            "source_digests": {"buzz": "b" * 64}, "device": "cuda", "batch_sequences": 128}
        self.arms = stage_arms(config, self.stage)

    # -- inputs the runner verifies against -------------------------------
    def parent_entries(self):
        """Real selected-parent checkpoints: hashed bytes and a recorded state digest."""
        entries = {}
        for seed in SEEDS:
            path = self.root / "checkpoints" / f"policy_sft_seed{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            state = hashlib.sha256(f"parent-state-{seed}".encode()).hexdigest()
            if not path.is_file():
                torch.save({"schema_version": policy_lib.POLICY_SCHEMA, "state": {},
                            "state_sha256": state}, path)
            entries[seed] = {"name": f"policy_sft_seed{seed}",
                             "checkpoint": self.relative(path), "sha256": sha256(path),
                             "state_sha256": state, "config_sha256": CONFIG_SHA}
        return entries

    def context(self):
        """The only data the validation path needs once scoring is injected."""
        return {"scaffold": SimpleNamespace(prefix=PREFIX), "val_pairs": self.pairs}

    def write(self, relative, payload):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, (bytes, bytearray)):
            path.write_bytes(payload)
        else:
            path.write_text(payload, encoding="utf-8")
        return path

    def relative(self, path):
        return str(Path(path).relative_to(self.root)).replace("\\", "/")

    def run_identity(self, arm, seed):
        return dict(self.identity, arm=arm, seed=int(seed), stage=self.stage)

    def reference_identity(self, seed):
        parent = self.parents[seed]
        return guard.parent_reference_identity(
            parent_checkpoint_sha256=parent["sha256"],
            parent_state_sha256=parent["state_sha256"], config_sha256=CONFIG_SHA,
            scaffold_prefix=PREFIX, chosen_index=self.pairs["chosen_index"],
            rejected_index=self.pairs["rejected_index"])

    def parent_reference(self, directory, seed):
        """The frozen parent vectors, written the way the run writes them."""
        chosen = np.array([-12.0, -13.0, -14.0, -15.0])
        rejected = np.array([-20.0, -21.0, -22.0, -23.0])
        identity = dict(self.reference_identity(seed),
                        chosen_values_sha256=array_digest(chosen),
                        rejected_values_sha256=array_digest(rejected))
        chosen.setflags(write=False)
        rejected.setflags(write=False)
        reference = guard.ParentValidationReference(identity=identity, chosen=chosen,
                                                    rejected=rejected, gpu_seconds=3.0,
                                                    wall_seconds=4.0)
        guard.save_parent_reference(Path(directory) / "parent_validation_reference.npz", reference)
        return reference

    def build(self, *, stopped=(), missing=()):
        summaries = {}
        for arm in self.arms:
            for seed in SEEDS:
                key = f"{arm['arm_id']}_seed{seed}"
                if key in missing:
                    continue
                is_stopped = key in stopped
                directory = self.output / f"stage{self.stage}" / key
                directory.mkdir(parents=True, exist_ok=True)
                reference = self.parent_reference(directory, seed)
                run_identity = self.run_identity(arm, seed)
                budgets = {}
                if not is_stopped:
                    progress = {"target_gpu_seconds": BUDGET, "reached": True,
                                "actual_gpu_seconds": 180.4, "updates": 1200, "gate_check": 7,
                                "exposures": {"sequences": 1000},
                                "distinct_exposures": {"distinct_chosen_rows": 900}}
                    checkpoint, state = self.checkpoint(directory, arm, seed, progress,
                                                        run_identity)
                    budgets = {"180.0": dict(progress, checkpoint=self.relative(checkpoint),
                                             checkpoint_sha256=sha256(checkpoint),
                                             state_sha256=state,
                                             checkpoint_wall_seconds=0.5)}
                document = {"schema_version": "her2-guarded-trajectory/1",
                            "identity": run_identity,
                            "status": "stopped" if is_stopped else "completed",
                            "stop_reason": "parent_relative_likelihood_breach" if is_stopped
                                           else None,
                            "updates": 58 if is_stopped else 1200, "checks": 3,
                            "attempted_updates": 58 if is_stopped else 1200,
                            "budgets": budgets,
                            "budgets_not_reached": {} if not is_stopped else {
                                "180.0": {"target_gpu_seconds": BUDGET, "reached": False,
                                          "stopped_at_update": 58}},
                            "last_passing": {"update": 25, "kind": "rolling_file"},
                            "exposures": {"sequences": 1000, "pairs": 500},
                            "cost": {"training_gpu_seconds": 180.4}}
                self.write(f"stage{self.stage}/{key}/trajectory.json", json.dumps(document))
                self.write(f"stage{self.stage}/{key}/updates.jsonl", '{"update": 1}\n')
                self.write(f"stage{self.stage}/{key}/monitor.jsonl",
                           self.monitor_lines(directory, reference, passed=not is_stopped))
                summaries[key] = document
        return summaries

    def checkpoint(self, directory, arm, seed, progress, run_identity):
        """A budget checkpoint that says which run, arm, seed and budget wrote it."""
        state = hashlib.sha256(f"{arm['arm_id']}-{seed}-{self.stage}".encode()).hexdigest()
        path = Path(directory) / "budget_180.pt"
        torch.save({"schema_version": policy_lib.POLICY_SCHEMA, "state": {},
                    "state_sha256": state,
                    "arm": {"arm_id": arm["arm_id"], "objective": arm["objective"],
                            "coefficients": dict(arm.get("coefficients") or {})},
                    "seed": int(seed), "budget_seconds": float(BUDGET),
                    "identity": run_identity, "progress": dict(progress)}, path)
        return path, state

    def monitor_lines(self, directory, reference, *, passed=True, check=7, update=1200,
                      identity=None, drop=None):
        """A real gate check: score vectors on disk, a verdict measured from them."""
        drop = float(drop if drop is not None else (0.4 if passed else 5.0))
        chosen_key = np.asarray(self.pairs["chosen_index"]).tobytes()

        def score(policy, index, *, batch_size=256, progress_every=0):
            base = (reference.chosen if np.asarray(index).tobytes() == chosen_key
                    else reference.rejected)
            return np.asarray(base, dtype=np.float64) - drop

        monitor = guard.GateMonitor(None, reference, directory=directory,
                                    score_sequences_fn=score,
                                    gate=guard.LikelihoodGate(threshold_nats_per_sequence=1.0))
        monitor.checks = int(check)
        record = monitor.check(self.pairs, update=int(update), gpu_seconds=180.4,
                               reason="budget_crossing")
        if identity is not None:
            record["identity"] = identity
        verdict = dict(record, record_kind="gate_verdict", run=dict(self.identity))
        snapshot = {"record_kind": "snapshot", "check": int(check), "update": int(update),
                    "passed": bool(record["passed"]), "D": record["D"]}
        return json.dumps(verdict) + "\n" + json.dumps(snapshot) + "\n"

    def endpoints(self, *, stopped=(), missing=()):
        records = []
        for arm in self.arms:
            for seed in SEEDS:
                key = f"{arm['arm_id']}_seed{seed}"
                if key in stopped or key in missing:
                    continue
                name = selection_lib.endpoint_name(arm["arm_id"], seed, BUDGET)
                scores = self.output / "validation" / "validation_scores" / f"{name}.npy"
                scores.parent.mkdir(parents=True, exist_ok=True)
                np.save(scores, np.array([-3.1, -3.2]))
                draws = self.write(f"validation/generation/draws_{name}.csv",
                                   "draw_index,core\n0,ACDEFGHIKL\n")
                checkpoint = self.output / f"stage{self.stage}/{key}/budget_180.pt"
                base = 0.90 if arm["objective"] == "continued_sft" else 0.92
                records.append({
                    "schema_version": "her2-guarded-endpoint/1", "name": name,
                    "arm_id": arm["arm_id"], "objective": arm["objective"],
                    "coefficients": arm.get("coefficients") or {}, "seed": seed,
                    "nominal_budget": BUDGET, "reached": True, "gate_passed": True,
                    "gate_evidence": {"check": 7, "update": 1200, "D": 0.4,
                        "parent_reference_identity": json.loads((
                            self.output / f"stage{self.stage}" / key
                            / "parent_validation_reference.json").read_text())["identity"]},
                    "diversity_eligible": arm["objective"] != "dpo",
                    "trajectory": key,
                    "checkpoint": self.relative(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "val_metrics": {"average_precision": base, "auroc": 0.97},
                    "val_strata": {"0": {"n": 0}, "1": stratum(70856, 23000, base),
                                   "2": stratum(6649, 2000, base - 0.1),
                                   ">=3": stratum(1147, 300, base - 0.2)},
                    "chosen_nll_per_residue": 1.49, "rejected_nll_per_residue": 2.5,
                    "val_scores": {"path": self.relative(scores),
                                   "sha256": sha256(scores), "rows": 2},
                    "generation_samples": {"path": self.relative(draws),
                                           "sha256": sha256(draws), "draws": 1},
                    "diversity": {"eligible": arm["objective"] != "dpo"},
                    "parent_kl": {"kl_nats": 0.4, "standard_error": 0.02}})
        document = {"schema_version": "her2-guarded-endpoint/1", "stage": self.stage,
                    "identity": self.identity,
                    "endpoints": records, "not_reached": [],
                    "parent_draw_references": self.parent_draw_references(),
                    "cost": {"validation_wall_seconds": 12.0, "parent_draw_wall_seconds": 3.0,
                             "parent_kl_wall_seconds": 2.0}}
        self.write(f"validation/stage{self.stage}_endpoints.json", json.dumps(document))
        return document

    def parent_draw_references(self):
        """The parent's own draws, persisted and hashed the way validation persists them."""
        references = {}
        for seed in SEEDS:
            path = self.write(f"validation/generation/parent_draws_policy_sft_seed{seed}.csv",
                              "draw_index,core\n0,ACDEFGHIKL\n")
            references[str(seed)] = {"path": self.relative(path), "sha256": sha256(path),
                                     "draws": 1, "name": f"policy_sft_seed{seed}"}
        return references

    def inspection(self):
        summary = self.write("original_summary.json", json.dumps({"updates": 3}))
        self.write("inspect_report.json", json.dumps({
            "schema_version": "her2-guarded-inspection/1",
            "identity": {"revision": self.identity["revision"],
                         "inherited": self.identity["inherited"]},
            "history": {"inputs": {"dpo_seed1::summary": {"path": str(summary),
                                                          "sha256": sha256(summary)}}}
        }))
        return summary


@pytest.fixture
def campaign(tmp_path, guarded, config, monkeypatch):
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    built = Campaign(tmp_path, small)
    built.build()
    built.endpoints()
    built.inspection()
    return built, small


def test_the_freeze_selects_on_validation_and_names_every_artifact(campaign, guarded, tmp_path):
    built, small = campaign
    document = guarded.run_freeze(small, built.output, built.identity, stage=1)
    assert document["stage"] == 1
    assert document["schema_version"] == lineage.GUARDED_SELECTION_SCHEMA
    assert len(document["trajectories"]) == 6
    assert set(document["control_artifacts"]) == {
        f"continued_sft_seed{seed}_budget180" for seed in SEEDS}
    # DPO ranks higher but fails diversity, so nothing is eligible for it.
    assert document["selection"]["180.0"]["continued_sft"]["selected"] == "continued_sft"
    assert document["selection"]["180.0"]["dpo"]["selected"] is None
    assert "none eligible" in document["selection"]["180.0"]["dpo"]["reason"]
    assert document["selection_read_test_or_assay"] is False
    assert (built.output / "stage1_complete.json").is_file()


def test_the_freeze_refuses_a_tampered_artifact(campaign, guarded, tmp_path):
    built, small = campaign
    target = built.output / "stage1/continued_sft_seed1/budget_180.pt"
    target.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="changed since the trajectory recorded it"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_the_freeze_refuses_a_tampered_inspection_report(campaign, guarded):
    built, small = campaign
    (built.output / "original_summary.json").write_text('{"updates": 4}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed since it was inspected"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_missing_inspection_report_stops_the_freeze(campaign, guarded):
    built, small = campaign
    (built.output / "inspect_report.json").unlink()
    with pytest.raises(ValueError, match="No inspection evidence"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_second_different_freeze_is_refused_rather_than_overwritten(campaign, guarded):
    built, small = campaign
    first = guarded.run_freeze(small, built.output, built.identity, stage=1)
    again = guarded.run_freeze(small, built.output, built.identity, stage=1)
    assert again["selection"] == first["selection"]
    assert again["artifacts"] == first["artifacts"]
    document = json.loads((built.output / "stage1_complete.json").read_text(encoding="utf-8"))
    document["selection"] = {"changed": True}
    (built.output / "stage1_complete.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_stopped_trajectory_appears_in_the_freeze_with_its_unreached_budgets(tmp_path, guarded,
                                                                               config, monkeypatch):
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    built = Campaign(tmp_path, small)
    stopped = ("dpo_beta0p1_seed2",)
    built.build(stopped=stopped)
    built.endpoints(stopped=stopped)
    built.inspection()
    document = guarded.run_freeze(small, built.output, built.identity, stage=1)
    entry = document["trajectories"]["dpo_beta0p1_seed2"]
    assert entry["status"] == "stopped"
    assert entry["budgets_not_reached"] == ["180.0"]
    assert document["selection"]["180.0"]["dpo"]["selected"] is None
    assert document["selection"]["180.0"]["dpo"]["ineligible"]["dpo_beta0p1"]["reason"] == \
        "missing_seeds"


def test_a_declared_trajectory_that_never_ran_fails_the_freeze(campaign, guarded):
    """CX-19: a subset of the grid is not the grid, and the marker must not say it is."""
    built, small = campaign
    import shutil
    shutil.rmtree(built.output / "stage1" / "dpo_beta0p1_seed3")
    with pytest.raises(ValueError, match="have no trajectory.json"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_trajectory_from_another_campaign_fails_the_freeze(campaign, guarded):
    """CX-18 at the runner boundary: the revision hashes are not the whole identity."""
    built, small = campaign
    path = built.output / "stage1" / "dpo_beta0p1_seed3" / "trajectory.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["identity"]["inherited"]["parents"]["1"]["sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="different campaign identity"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_an_endpoint_whose_checkpoint_bytes_moved_is_not_recertified(campaign, guarded, tmp_path):
    """CX-26: a checkpoint is the weights the gate passed, identified by hash."""
    built, small = campaign
    entry = {"trajectory": "dpo_beta0p1_seed1", "nominal_budget": 180.0,
             "record": json.loads((built.output / "stage1" / "dpo_beta0p1_seed1"
                                   / "trajectory.json").read_text(encoding="utf-8"))
             ["budgets"]["180.0"]}
    assert guarded.verify_budget_checkpoint(entry, root=tmp_path)[1] == \
        entry["record"]["checkpoint_sha256"]
    (built.output / "stage1" / "dpo_beta0p1_seed1" / "budget_180.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="changed since the trajectory recorded it"):
        guarded.verify_budget_checkpoint(entry, root=tmp_path)
    entry["record"] = dict(entry["record"])
    entry["record"].pop("checkpoint_sha256")
    with pytest.raises(ValueError, match="records no checkpoint path and hash"):
        guarded.verify_budget_checkpoint(entry, root=tmp_path)


def gate_fixture(built, small, guarded, trajectory="dpo_beta0p1_seed1", seed=1):
    """The directory, its budget record, the verified reference and the declared gate."""
    directory = built.output / "stage1" / trajectory
    record = json.loads((directory / "trajectory.json").read_text(
        encoding="utf-8"))["budgets"]["180.0"]
    expected = guarded.expected_parent_reference_identity(built.parents[seed], built.context(),
                                                          root=built.root)
    reference = guarded.verified_parent_reference(directory, expected)
    gate = guard.LikelihoodGate(threshold_nats_per_sequence=1.0)
    return directory, record, reference, gate


def test_the_gate_verdict_is_recomputed_from_the_vectors_it_was_measured_on(campaign, guarded):
    """CX-26: gate_passed is a measurement re-derived here, not a number read out of a log."""
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)
    verdict = guarded.verified_gate_record(directory, record, reference=reference, gate=gate)
    assert verdict["check"] == 7
    assert verdict["recomputed"]["D"] == pytest.approx(0.4)
    assert verdict["recomputed"]["passed"] is True
    assert verdict["recomputed"]["pairs"] == built.pairs["pairs"]
    assert verdict["recomputed"]["threshold_nats_per_sequence"] == 1.0
    assert "recomputed" in verdict["recomputed"]["note"]


def test_a_fabricated_verdict_with_no_score_vectors_is_refused(campaign, guarded):
    """passed=true beside D=999 and no arrays: the exact shape a log-reading check accepts."""
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)
    fabricated = {"record_kind": "gate_verdict", "check": 7, "update": 1200, "passed": True,
                  "D": 999.0, "D_per_residue": 99.9, "threshold_nats_per_sequence": 1.0,
                  "pairs": built.pairs["pairs"], "identity": dict(reference.identity)}
    (directory / "monitor.jsonl").write_text(json.dumps(fabricated) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names no retained score vectors"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)


def test_a_verdict_whose_arrays_are_gone_or_edited_is_refused(campaign, guarded):
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)
    stored = directory / "monitor_scores" / "check_00007_update1200.npz"
    moved = stored.rename(stored.with_name("moved.npz"))
    with pytest.raises(ValueError, match="not beside the run"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)
    moved.rename(stored)
    with np.load(stored) as archive:
        chosen, rejected = np.array(archive["chosen"]), np.array(archive["rejected"])
    np.savez(stored, chosen=chosen - 3.0, rejected=rejected)
    with pytest.raises(ValueError, match="changed since the check wrote it"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)


def test_a_journalled_d_that_the_vectors_do_not_produce_is_refused(campaign, guarded):
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)
    lines = [json.loads(line) for line in
             (directory / "monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    verdict = dict(lines[0], D=0.01)
    (directory / "monitor.jsonl").write_text(json.dumps(verdict) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not the measurement"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)
    # And a pass declared over vectors that breach is not a pass. The vectors are
    # real: the check is re-run at a 5 nat drop and only the verdict is edited.
    (directory / "monitor.jsonl").write_text(
        built.monitor_lines(directory, reference, drop=5.0), encoding="utf-8")
    measured = json.loads((directory / "monitor.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert measured["passed"] is False and measured["D"] == pytest.approx(5.0)
    breached = dict(measured, passed=True)
    (directory / "monitor.jsonl").write_text(json.dumps(breached) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="the declared gate applied to these vectors"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)


def test_a_verdict_judged_under_another_threshold_is_not_this_protocols(campaign, guarded):
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)
    lines = [json.loads(line) for line in
             (directory / "monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    (directory / "monitor.jsonl").write_text(
        json.dumps(dict(lines[0], threshold_nats_per_sequence=5.0)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="another threshold"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)


def test_a_tampered_or_absent_gate_verdict_stops_validation(campaign, guarded):
    """CX-26: a checkpoint's verdict must exist, be unique, pass, and be this run's."""
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)

    (directory / "monitor.jsonl").write_text(
        built.monitor_lines(directory, reference, passed=False), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)

    (directory / "monitor.jsonl").write_text(
        built.monitor_lines(directory, reference,
                            identity={"pairs": 12, "population": "a_subset"}), encoding="utf-8")
    with pytest.raises(ValueError, match="different parent reference"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)

    (directory / "monitor.jsonl").write_text(
        built.monitor_lines(directory, reference, check=3), encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly one gate verdict"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)

    (directory / "monitor.jsonl").write_text(
        built.monitor_lines(directory, reference, update=17), encoding="utf-8")
    with pytest.raises(ValueError, match="was taken at update"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)


def test_a_snapshot_line_is_not_mistaken_for_a_gate_verdict(campaign, guarded):
    """The two kinds of monitor line say different things and only one is a verdict."""
    built, small = campaign
    directory, record, reference, gate = gate_fixture(built, small, guarded)
    lines = [json.loads(line) for line in
             (directory / "monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["record_kind"] for line in lines] == ["gate_verdict", "snapshot"]
    # Only the snapshot survives: there is no verdict to verify, so nothing is certified.
    (directory / "monitor.jsonl").write_text(json.dumps(lines[1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly one gate verdict"):
        guarded.verified_gate_record(directory, record, reference=reference, gate=gate)


def test_the_parent_reference_must_be_this_parents_and_must_have_its_arrays(campaign, guarded):
    """CX-26: the sidecar is a claim; the identity is rebuilt and the vectors are loaded."""
    built, small = campaign
    directory = built.output / "stage1" / "dpo_beta0p1_seed1"
    expected = guarded.expected_parent_reference_identity(built.parents[1], built.context(),
                                                          root=built.root)
    assert guarded.verified_parent_reference(directory, expected).pairs == built.pairs["pairs"]
    # Another seed's parent produced a different identity, and the sidecar is held to it.
    other = guarded.expected_parent_reference_identity(built.parents[2], built.context(),
                                                       root=built.root)
    with pytest.raises(ValueError, match="identity mismatch"):
        guarded.verified_parent_reference(directory, other)
    # A different ordered pair set is a different measurement, not the same one re-sorted.
    swapped = built.context()
    swapped["val_pairs"] = {"chosen_index": built.pairs["rejected_index"],
                            "rejected_index": built.pairs["chosen_index"], "pairs": 4}
    with pytest.raises(ValueError, match="identity mismatch"):
        guarded.verified_parent_reference(
            directory, guarded.expected_parent_reference_identity(built.parents[1], swapped,
                                                                  root=built.root))
    (directory / "parent_validation_reference.npz").unlink()
    with pytest.raises(ValueError, match="describes vectors that are not on disk"):
        guarded.verified_parent_reference(directory, expected)


def test_a_missing_parent_reference_stops_validation(campaign, guarded):
    built, small = campaign
    directory = built.output / "stage1" / "dpo_beta0p1_seed1"
    expected = guarded.expected_parent_reference_identity(built.parents[1], built.context(),
                                                          root=built.root)
    (directory / "parent_validation_reference.json").unlink()
    with pytest.raises(ValueError, match="is missing"):
        guarded.verified_parent_reference(directory, expected)


# ---------------------------------------------------------------------------
# stage 1 -> 2 -> 3, through the real freeze
# ---------------------------------------------------------------------------

def test_the_staged_campaign_carries_its_verified_control_from_stage_one_to_three(
        tmp_path, guarded, config, monkeypatch):
    """The control is fitted once and reused twice, under hash verification both times."""
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    first = Campaign(tmp_path, small, stage=1)
    first.build()
    first.endpoints()
    first.inspection()
    stage1 = guarded.run_freeze(small, first.output, first.identity, stage=1)
    assert stage1["control_status"]["available"] is True
    assert len(stage1["control_endpoints"]) == len(SEEDS)
    assert set(stage1["control_artifacts"]) == {f"continued_sft_seed{seed}_budget180"
                                                for seed in SEEDS}

    second = Campaign(tmp_path, small, stage=2)
    second.build()
    second.endpoints()
    stage2 = guarded.run_freeze(small, second.output, second.identity, stage=2)
    assert sorted(stage2["trajectories"]) == [f"dpop_beta0p1_lambda1_seed{seed}"
                                              for seed in SEEDS]
    assert stage2["reused_controls"]["available"] is True
    assert stage2["reused_controls"]["source_stage"] == 1
    assert stage2["control_artifacts"] == stage1["control_artifacts"]
    assert stage2["declared_objectives"] == ["continued_sft", "dpop"]
    # The stage-1 control is compared against, not re-fitted and not re-measured.
    deltas = stage2["matched_control_deltas"]
    assert len(deltas["rows"]) == len(SEEDS)
    assert {row["control_reused_from_stage"] for row in deltas["rows"]} == {1}
    assert deltas["controls_available"] is True
    assert stage2["stage_cost"]["control_reused_from_earlier_stage"] is True
    assert stage2["selection"]["180.0"]["continued_sft"]["selected"] == "continued_sft"
    assert stage2["selection"]["180.0"]["dpop"]["selected"] == "dpop_beta0p1_lambda1"

    third = Campaign(tmp_path, small, stage=3)
    third.build()
    third.endpoints()
    stage3 = guarded.run_freeze(small, third.output, third.identity, stage=3)
    assert sorted(stage3["trajectories"]) == [f"ipo_tau0p5_seed{seed}" for seed in SEEDS]
    assert stage3["control_artifacts"] == stage1["control_artifacts"]
    # Inherited through stage 2, still attributed to the stage that paid for it.
    assert stage3["reused_controls"]["source_stage"] == 2
    assert stage3["reused_controls"]["fitted_in_stage"] == 1
    assert {row["reused_from_stage"] for row in stage3["control_endpoints"]} == {1}
    # CX-25/29: the control trajectories travel with the marker rather than being
    # rebuilt from a stage that fitted no control. Two hops, still all three.
    for marker in (stage2, stage3):
        status = marker["control_status"]
        assert set(status["trajectories"]) == {f"continued_sft_seed{seed}" for seed in SEEDS}
        assert status["source_stage"] == 1
        assert all(entry["budgets_reached"] == ["180.0"]
                   for entry in status["trajectories"].values())


def test_stage_two_will_not_start_from_a_stage_one_whose_control_bytes_moved(
        tmp_path, guarded, config, monkeypatch):
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    first = Campaign(tmp_path, small, stage=1)
    first.build()
    first.endpoints()
    first.inspection()
    guarded.run_freeze(small, first.output, first.identity, stage=1)
    second = Campaign(tmp_path, small, stage=2)
    second.build()
    second.endpoints()
    (first.output / "stage1" / "continued_sft_seed2" / "budget_180.pt").write_bytes(b"moved")
    with pytest.raises(ValueError, match="changed on disk"):
        guarded.run_freeze(small, second.output, second.identity, stage=2)


def test_stage_costs_count_the_parent_reference_and_flag_what_was_not_measured(campaign, guarded):
    """CX-28: the monitoring total includes the parent scoring, and 0 is not 'unmeasured'."""
    built, _ = campaign
    directory = built.output / "stage1" / "dpo_beta0p1_seed1"
    (directory / "summary.json").write_text(json.dumps({
        "status": "completed",
        "cost": {"training_gpu_seconds": 180.4, "checkpoint_wall_seconds": 1.5,
                 "budget_evaluation_wall_seconds": 0.7, "diagnostic_wall_seconds": 2.0},
        "excluded_costs": {"monitor_gpu_seconds": 12.0, "monitor_wall_seconds": 15.0,
                           "monitor_checks": 9,
                           "parent_validation_reference_gpu_seconds": 3.0,
                           "parent_validation_reference_wall_seconds": 4.0},
        "training_reference": {"physically_reused": True,
                               "cold_start_charged_gpu_seconds": 68.8,
                               "warm_reuse_wall_seconds": 0.25},
        "budgets": {"180.0": {"checkpoint_wall_seconds": 0.5}},
        "exposures": {"sequences": 1000, "pairs": 500}, "core_token_exposures": 10000,
        "updates": 1200, "attempted_updates": 1200}), encoding="utf-8")
    trajectories, _ = guarded.reached_checkpoints(built.output, stage=1)
    document = guarded.stage_cost_summary(
        built.output, stage=1, trajectories=trajectories,
        validated={"cost": {"validation_wall_seconds": 12.0}}, reused_controls=False)
    row = document["per_trajectory"]["dpo_beta0p1_seed1"]
    assert row["parent_reference_gpu_seconds"] == 3.0
    assert row["parent_reference_wall_seconds"] == 4.0
    assert row["budget_checkpoint_wall_seconds"] == 0.5, "the nominal endpoint write, separately"
    assert row["rolling_checkpoint_wall_seconds"] == 1.5
    assert row["reference_cold_start_charged_gpu_seconds"] == 68.8
    assert row["reference_warm_reuse_wall_seconds"] == 0.25, "what the reuse physically took"
    assert row["sequence_exposures"] == 1000 and row["pair_exposures"] == 500
    assert row["core_token_exposures"] == 10000
    assert document["totals"]["parent_reference_gpu_seconds"] == 3.0
    assert document["totals"]["monitor_checks"] == 9
    # The five trajectories with no summary are named, not silently free.
    assert len(document["summaries_missing"]) == 5
    assert document["unavailable_measurements"]["failed_attempts"] == ["dpo_beta0p1_seed1"]
    assert "counted as zero" in document["unavailable_note"]


def drive_validation(guarded, built, small, monkeypatch):
    """Run the REAL validate stage with only the model work replaced.

    What is injected: scoring a checkpoint, sampling from it, drawing from the
    parent and the Monte Carlo KL. Everything the review found broken --
    rebuilding the parent-reference identity, loading and re-hashing its vectors,
    recomputing D from the retained arrays, binding the checkpoint payload to its
    arm, seed, budget and progress, persisting the parent draws and checking the
    endpoint set against what was actually reached -- is the production code.
    """
    def fake_validate(name, path, arm, seed, config, context, strata, output, raw_root, device,
                      parent_draws, expected_state_sha256=None):
        scores = Path(output) / "validation" / "validation_scores" / f"{name}.npy"
        scores.parent.mkdir(parents=True, exist_ok=True)
        np.save(scores, np.array([-3.1, -3.2]))
        draws = Path(output) / "validation" / "generation" / f"draws_{name}.csv"
        draws.parent.mkdir(parents=True, exist_ok=True)
        draws.write_text("draw_index,core\n0,ACDEFGHIKL\n", encoding="utf-8")
        base = 0.90 if arm["objective"] == "continued_sft" else 0.92
        return {"schema_version": "her2-guarded-endpoint/1", "name": name,
                "arm_id": arm["arm_id"], "objective": arm["objective"],
                "coefficients": arm.get("coefficients") or {}, "seed": int(seed),
                "checkpoint": built.relative(path), "checkpoint_sha256": sha256(path),
                "val_metrics": {"average_precision": base, "auroc": 0.97},
                "val_strata": {"0": {"n": 0}, "1": stratum(70856, 23000, base),
                               "2": stratum(6649, 2000, base - 0.1),
                               ">=3": stratum(1147, 300, base - 0.2)},
                "chosen_nll_per_residue": 1.49, "rejected_nll_per_residue": 2.5,
                "val_scores": {"path": built.relative(scores), "sha256": sha256(scores),
                               "rows": 2},
                "generation": {"draws": 1},
                "generation_samples": {"path": built.relative(draws), "sha256": sha256(draws),
                                       "draws": 1},
                "diversity": {"eligible": True}, "diversity_eligible": True,
                "state_digest_verified": expected_state_sha256,
                "parent_draw_name": parent_draws["name"], "wall_seconds": 0.1}

    monkeypatch.setattr(guarded, "validation_strata", lambda config, context: {})
    monkeypatch.setattr(guarded, "parent_kl_pass",
                        lambda records, *args, **kwargs: records)
    monkeypatch.setattr(guarded, "_sample_parent_draws",
                        lambda parent, config, context, raw_root, device: data.encode_cores(
                            ["ACDEFGHIKL", "ACDEFGHIKM"]))
    monkeypatch.setattr(guarded, "validate_checkpoint", fake_validate)
    return guarded.run_validate(small, built.context(), built.output, built.identity,
                                built.parents, stage=built.stage, device="cpu",
                                raw_root=built.root)


@pytest.fixture
def validated_campaign(tmp_path, guarded, config, monkeypatch):
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    built = Campaign(tmp_path, small)
    built.build()
    built.inspection()
    document = drive_validation(guarded, built, small, monkeypatch)
    return built, small, document


def test_validation_recomputes_the_gate_binds_the_payload_and_keeps_the_parent_draws(
        validated_campaign, guarded):
    """CX-26 end to end: the evidence is re-derived, and the draws behind it are kept."""
    built, small, document = validated_campaign
    assert len(document["endpoints"]) == 6
    assert document["identity"] == built.identity, "the whole campaign identity, not a block of it"
    for record in document["endpoints"]:
        assert record["gate_evidence"]["D"] == pytest.approx(0.4)
        assert record["gate_evidence"]["journalled_D"] == pytest.approx(0.4)
        assert record["gate_evidence"]["pairs"] == built.pairs["pairs"]
        assert record["gate_evidence"]["parent_reference_pairs"] == built.pairs["pairs"]
        assert record["checkpoint_binding"]["updates"] == 1200
        assert record["checkpoint_binding"]["gate_check"] == 7
        assert record["state_digest_verified"] == record["checkpoint_binding"]["state_sha256"]
    # The parent's own draws are evidence: written, hashed and named per seed.
    references = document["parent_draw_references"]
    assert sorted(references) == [str(seed) for seed in SEEDS]
    for seed, entry in references.items():
        path = built.root / entry["path"]
        assert path.is_file() and sha256(path) == entry["sha256"]
        assert entry["draws"] == 2 and entry["name"] == f"policy_sft_seed{seed}"
    frozen = guarded.run_freeze(small, built.output, built.identity, stage=1)
    assert set(frozen["control_artifacts"]) == {f"continued_sft_seed{seed}_budget180"
                                                for seed in SEEDS}
    assert [key for key in frozen["artifacts"] if key.startswith("parent_draws::")] == [
        f"parent_draws::seed{seed}" for seed in SEEDS]


def test_a_checkpoint_from_another_arm_or_budget_is_not_validated(validated_campaign, guarded,
                                                                   monkeypatch):
    """CX-26: the byte hash says the file did not change, not which run wrote it."""
    built, small, _ = validated_campaign
    directory = built.output / "stage1" / "dpo_beta0p1_seed1"
    trajectory = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
    record = trajectory["budgets"]["180.0"]
    payload = torch.load(directory / "budget_180.pt", map_location="cpu", weights_only=True)
    arm = {"arm_id": "dpo_beta0p1", "objective": "dpo", "coefficients": {"beta": 0.1}}
    assert guarded.verify_checkpoint_payload(
        payload, trajectory="dpo_beta0p1_seed1", arm=arm, seed=1, budget=180.0, record=record,
        identity=built.identity, where="x")["seed"] == 1
    with pytest.raises(ValueError, match="written at seed"):
        guarded.verify_checkpoint_payload(payload, trajectory="dpo_beta0p1_seed2", arm=arm,
                                          seed=2, budget=180.0, record=record,
                                          identity=built.identity, where="x")
    with pytest.raises(ValueError, match="written at budget"):
        guarded.verify_checkpoint_payload(payload, trajectory="dpo_beta0p1_seed1", arm=arm,
                                          seed=1, budget=360.0, record=record,
                                          identity=built.identity, where="x")
    other = {"arm_id": "ipo_tau0p5", "objective": "ipo", "coefficients": {"tau": 0.5}}
    with pytest.raises(ValueError, match="written for arm"):
        guarded.verify_checkpoint_payload(payload, trajectory="ipo_tau0p5_seed1", arm=other,
                                          seed=1, budget=180.0, record=record,
                                          identity=built.identity, where="x")
    with pytest.raises(ValueError, match="completed updates"):
        guarded.verify_checkpoint_payload(payload, trajectory="dpo_beta0p1_seed1", arm=arm,
                                          seed=1, budget=180.0, record=dict(record, updates=7),
                                          identity=built.identity, where="x")
    drifted = json.loads(json.dumps(built.identity))
    drifted["device"] = "cpu"
    with pytest.raises(ValueError, match="different campaign identity"):
        guarded.verify_checkpoint_payload(payload, trajectory="dpo_beta0p1_seed1", arm=arm,
                                          seed=1, budget=180.0, record=record,
                                          identity=drifted, where="x")


def endpoints_document(built):
    path = built.output / "validation" / f"stage{built.stage}_endpoints.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_a_validated_document_with_its_endpoints_deleted_does_not_freeze(validated_campaign,
                                                                         guarded):
    """CX-26(a): six reached endpoints and an empty list is missing evidence, not a result."""
    built, small, _ = validated_campaign
    path, document = endpoints_document(built)
    document["endpoints"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="have no validated endpoint"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_foreign_checkpoint_substituted_into_an_endpoint_does_not_freeze(validated_campaign,
                                                                           guarded):
    """CX-26(b): a matching hash on other weights is exactly what substitution looks like."""
    built, small, _ = validated_campaign
    path, document = endpoints_document(built)
    foreign = built.root / "foreign.pt"
    foreign.write_bytes(b"weights from another run")
    document["endpoints"][0]["checkpoint"] = "foreign.pt"
    document["endpoints"][0]["checkpoint_sha256"] = sha256(foreign)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="not the weights that run wrote"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_an_endpoint_relabelled_to_an_unreached_budget_does_not_freeze(validated_campaign,
                                                                       guarded):
    """CX-26(c): 360 s was never reached, so there is no 360 s checkpoint to certify."""
    built, small, _ = validated_campaign
    path, document = endpoints_document(built)
    document["endpoints"][0]["nominal_budget"] = 360.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="do not record as a reached budget"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_duplicated_endpoint_does_not_freeze(validated_campaign, guarded):
    built, small, _ = validated_campaign
    path, document = endpoints_document(built)
    document["endpoints"].append(json.loads(json.dumps(document["endpoints"][0])))
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="One reached budget is one endpoint"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_validated_document_from_another_campaign_does_not_freeze(validated_campaign, guarded):
    built, small, _ = validated_campaign
    path, document = endpoints_document(built)
    document["identity"]["inherited"]["parents"]["1"]["sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="different campaign identity"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_a_moved_parent_draw_reference_does_not_freeze(validated_campaign, guarded):
    """The diversity verdicts were measured against those draws; they are evidence."""
    built, small, document = validated_campaign
    (built.root / document["parent_draw_references"]["1"]["path"]).write_text(
        "draw_index,core\n0,WWWWWWWWWW\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since validation drew it"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


@pytest.mark.parametrize("pattern", ["monitor_scores/*.npz", "parent_validation_reference.npz"])
def test_freeze_requires_gate_vectors_after_validation(validated_campaign, guarded, pattern):
    built, small, _ = validated_campaign
    directory = built.output / "stage1" / "dpo_beta0p1_seed1"
    paths = list(directory.glob(pattern))
    assert paths
    for path in paths:
        path.unlink()
    with pytest.raises(ValueError, match="missing"):
        guarded.run_freeze(small, built.output, built.identity, stage=1)


def test_stage_three_reverifies_original_control_evidence(tmp_path, guarded, config, monkeypatch):
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    for stage in (1, 2):
        built = Campaign(tmp_path, small, stage=stage)
        built.build()
        built.endpoints()
        built.inspection()
        guarded.run_freeze(small, built.output, built.identity, stage=stage)
    directory = built.output / "stage1" / "continued_sft_seed1"
    next((directory / "monitor_scores").glob("*.npz")).unlink()
    with pytest.raises(ValueError, match="missing"):
        selection_lib.require_previous_stage(built.output, 3, identity=built.identity,
                                             root=tmp_path)


@pytest.mark.parametrize("outcome", ["failed", "exception", "stopped"])
def test_outer_runner_preserves_controller_status_and_work(tmp_path, guarded, config,
                                                         monkeypatch, outcome):
    """The real summary/ledger boundary, with only controller work injected."""
    from smallAntibodyGen.experiments.her2_runtime import GpuBudgetClock
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    policy = SimpleNamespace(model=torch.nn.Linear(1, 1))
    reference = SimpleNamespace(gpu_seconds=3.0, wall_seconds=4.0, reused=False,
                                document=lambda: {})
    monkeypatch.setattr(guarded, "load_parent_policy", lambda *a, **kw: policy)
    monkeypatch.setattr(guarded, "prepare_parent_reference", lambda *a, **kw: reference)
    monkeypatch.setattr(guarded, "build_step", lambda *a, **kw: (None, lambda: {}))
    status = "failed" if outcome == "exception" else outcome
    document = {"status": status, "updates": 3, "attempted_updates": 4,
                "stop_reason": "last_passing_snapshot_failure" if status == "failed"
                               else "parent_relative_likelihood_breach",
                "budgets": {}, "budgets_not_reached": {"180.0": {}, "360.0": {}, "600.0": {}},
                "cost": {"training_gpu_seconds": 1.25, "failed_work_gpu_seconds": 0.25}}

    def controller(**kwargs):
        (Path(kwargs["directory"]) / "trajectory.json").write_text(json.dumps(document))
        if outcome == "exception":
            raise RuntimeError("synthetic controller failure")
        return document

    monkeypatch.setattr(guarded.trajectory_lib, "run_guarded_trajectory", controller)
    arm = {"arm_id": "continued_sft", "objective": "continued_sft", "coefficients": {},
           "stage": 1}
    kwargs = dict(stage=1, device="cpu", batch_sequences=128,
                  parents={1: {"checkpoint": "parent.pt", "sha256": "a" * 64}},
                  identity={"revision": {}}, raw_root=tmp_path,
                  clock_factory=GpuBudgetClock, monitor_clock_factory=GpuBudgetClock,
                  allow_dirty=True)
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="synthetic controller failure"):
            guarded.run_trajectory(arm, 1, config, {}, tmp_path / "run", **kwargs)
    else:
        guarded.run_trajectory(arm, 1, config, {}, tmp_path / "run", **kwargs)
    directory = tmp_path / "run" / "stage1" / "continued_sft_seed1"
    summary = json.loads((directory / "summary.json").read_text())
    ledger = json.loads((directory / "run.json").read_text())
    assert summary["status"] == status
    assert ledger["status"] == ("completed" if status == "stopped" else "failed")
    assert summary["updates"] == 3 and summary["attempted_updates"] == 4
    assert summary["cost"]["training_gpu_seconds"] == 1.25
    assert summary["cost"]["failed_work_gpu_seconds"] == 0.25
    assert set(summary["budgets_not_reached"]) == {"180.0", "360.0", "600.0"}


def test_a_stage_whose_every_control_stopped_still_freezes_with_none_eligible(
        tmp_path, guarded, config, monkeypatch):
    """CX-29: a full stage of valid early stops is a result, and the later stages inherit it."""
    monkeypatch.setattr(guarded, "ROOT", tmp_path)
    small = tiny_config(config)
    built = Campaign(tmp_path, small, stage=1)
    stopped = tuple(f"continued_sft_seed{seed}" for seed in SEEDS)
    built.build(stopped=stopped)
    built.endpoints(stopped=stopped)
    built.inspection()
    document = guarded.run_freeze(small, built.output, built.identity, stage=1)
    # Every declared objective is still in the table, with its reason.
    assert sorted(document["selection"]["180.0"]) == ["continued_sft", "dpo"]
    assert document["selection"]["180.0"]["continued_sft"]["selected"] is None
    assert "none eligible" in document["selection"]["180.0"]["continued_sft"]["reason"]
    # The stopped control trajectories are carried, with their stop reasons.
    status = document["control_status"]
    assert status["available"] is False
    assert "no matched control endpoint" in status["reason"]
    assert set(status["trajectories"]) == set(stopped)
    assert all(entry["stop_reason"] == "parent_relative_likelihood_breach"
               for entry in status["trajectories"].values())
    assert document["control_artifacts"] == {}
    # And no comparison is invented for the arms that would have used it.
    deltas = document["matched_control_deltas"]
    assert deltas["rows"] == [] and deltas["controls_available"] is False
    assert "None is approximated" in deltas["unavailable_reason"]
    assert len(deltas["unmatched"]) == len(SEEDS)
    # A later stage inherits the absence rather than an absent field.
    second = Campaign(tmp_path, small, stage=2)
    second.build()
    second.endpoints()
    stage2 = guarded.run_freeze(small, second.output, second.identity, stage=2)
    assert stage2["reused_controls"]["available"] is False
    assert stage2["matched_control_deltas"]["controls_available"] is False
    assert stage2["selection"]["180.0"]["dpop"]["selected"] == "dpop_beta0p1_lambda1"
    # CX-25/29: and stage 3 still carries WHICH control trajectories stopped and why,
    # two markers away from the stage that fitted them.
    third = Campaign(tmp_path, small, stage=3)
    third.build()
    third.endpoints()
    stage3 = guarded.run_freeze(small, third.output, third.identity, stage=3)
    for marker in (stage2, stage3):
        status = marker["control_status"]
        assert status["available"] is False
        assert set(status["trajectories"]) == set(stopped)
        assert all(entry["stop_reason"] == "parent_relative_likelihood_breach"
                   for entry in status["trajectories"].values())
        assert all(entry["budgets_not_reached"] == ["180.0"]
                   for entry in status["trajectories"].values())
        assert "no matched control endpoint" in status["reason"]
    assert stage3["matched_control_deltas"]["controls_available"] is False
