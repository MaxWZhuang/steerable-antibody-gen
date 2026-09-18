"""Runner-level contracts: run identity, stage contract, artifact bookkeeping.

The HER2 scripts are loaded from ``scripts/`` by path, because they are entry
points rather than package modules. Nothing here fits, samples or downloads: the
stage-contract test below builds a miniature campaign out of a few bytes on disk
and drives the real freeze, verification and unlock code over it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_eval as evaluation
from smallAntibodyGen.experiments import her2_policy as policy_lib
from smallAntibodyGen.experiments.her2_runtime import RunLedger, code_digests, sha256

ROOT = Path(__file__).resolve().parents[3]


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
    return json.loads((ROOT / "configs/experiments/her2_posttrain.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def train():
    return load_script("train_her2")


@pytest.fixture(scope="module")
def posttrain():
    return load_script("posttrain_her2")


@pytest.fixture(scope="module")
def evaluate():
    return load_script("evaluate_her2")


# ---------------------------------------------------------------------------
# run identity
# ---------------------------------------------------------------------------

def test_a_completed_run_is_skipped_not_overwritten(tmp_path):
    identity = {"config_digest": "a", "code_digests": {"x.py": "1"}}
    ledger = RunLedger(tmp_path / "run", identity, metadata={"first_claimed_git_commit": "abc123"})
    assert ledger.start() == "start"
    ledger.complete({"result": 1})
    assert RunLedger(tmp_path / "run", identity).start() == "completed"


def test_edited_scientific_code_is_a_different_run(tmp_path):
    """Same config, changed loss: resuming as though it were the same fit is the bug."""
    ledger = RunLedger(tmp_path / "run", {"config_digest": "a", "code_digests": {"x.py": "1"}})
    ledger.start()
    ledger.complete({"result": 1})
    changed = RunLedger(tmp_path / "run", {"config_digest": "a", "code_digests": {"x.py": "2"}})
    with pytest.raises(ValueError, match="different identity"):
        changed.start()


def test_an_interrupted_run_needs_an_explicit_discard_and_keeps_its_first_commit(tmp_path):
    identity = {"config_digest": "a"}
    RunLedger(tmp_path / "run", identity,
              metadata={"first_claimed_git_commit": "abc123"}).start()
    with pytest.raises(ValueError, match="discard-incomplete"):
        RunLedger(tmp_path / "run", identity).start()
    resumed = RunLedger(tmp_path / "run", identity,
                        metadata={"first_claimed_git_commit": "def456"})
    assert resumed.start(discard_incomplete=True) == "start"
    resumed.complete({"result": 2})
    record = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert record["metadata"]["first_claimed_git_commit"] == "abc123", (
        "the commit that produced the checkpoints on disk must survive a resume")


def test_code_digests_refuse_a_missing_source(tmp_path):
    with pytest.raises(ValueError, match="part of the run identity"):
        code_digests(tmp_path, ["scripts/not_here.py"])


def test_every_scientific_source_in_the_identity_exists():
    digests = code_digests(ROOT)
    assert "scripts/posttrain_her2.py" in digests
    assert "src/smallAntibodyGen/experiments/her2_preferences.py" in digests
    assert all(len(value) == 64 for value in digests.values())


# ---------------------------------------------------------------------------
# artifact bookkeeping
# ---------------------------------------------------------------------------

def test_every_raw_budget_checkpoint_becomes_an_artifact(posttrain, config):
    base = {"selected": {"policy_sft_seed1": {"kind": "policy", "checkpoint": "a.pt",
                                              "sha256": "0" * 64},
                         "piggen_zeroshot": {"kind": "pinned_zero_shot", "checkpoint": "w.st",
                                             "sha256": "1" * 64},
                         "linear_3class": {"kind": "classifier_proxy", "checkpoint": "l.pt",
                                           "sha256": "2" * 64}}}
    def budget(name, digest, target, actual, updates, distinct):
        return {"checkpoint": name, "checkpoint_sha256": digest,
                "target_gpu_seconds": target, "actual_gpu_seconds": actual,
                "overshoot_seconds": actual - target, "overshoot_within_one_update": True,
                "updates": updates, "exposures": {"pairs": updates * 64,
                                                  "sequences": updates * 128},
                "core_token_exposures": updates * 1280,
                "distinct_exposures": {"distinct_chosen_rows": distinct}}

    runs = {"dpo_seed1": {"method": "dpo", "seed": 1, "budgets": {
        "180.0": budget("d180.pt", "3" * 64, 180.0, 180.4, 1200, 76800),
        "360.0": budget("d360.pt", "4" * 64, 360.0, 361.1, 2400, 120477)}}}
    artifacts = posttrain.policy_artifacts(config, base, runs)
    assert set(artifacts) == {"policy_sft_seed1", "piggen_zeroshot",
                              "dpo_seed1_budget180", "dpo_seed1_budget360"}
    entry = artifacts["dpo_seed1_budget180"]
    assert entry["role"] == "raw_budget"
    # Nominal target and measured time both survive into the artifact, and so does
    # what the budget actually consumed.
    assert entry["budget_seconds"] == 180.0
    assert entry["actual_gpu_seconds"] == pytest.approx(180.4)
    assert entry["updates"] == 1200
    assert entry["distinct_exposures"]["distinct_chosen_rows"] == 76800
    assert entry["core_token_exposures"] == 1200 * 1280
    # The lineage is recorded by the artifact builder, not attached by whichever
    # stage happens to run.
    assert entry["parent_name"] == "policy_sft_seed1"
    # The two initial arms are distinguishable: both are SFT, from different starts.
    assert artifacts["policy_sft_seed1"]["method"] == "initial_sft_sft"
    assert artifacts["policy_sft_seed1"]["arm"] == "sft"
    assert artifacts["policy_sft_seed1"]["seed"] == 1
    assert artifacts["piggen_zeroshot"]["method"] == "zero_shot"
    # the classifier is not a policy artifact; it is re-added as a baseline at freeze time
    assert "linear_3class" not in artifacts


def test_checkpoint_keys_match_what_the_evaluation_expects(posttrain, evaluate, config):
    expected = set(evaluate.expected_selection(config))
    for method in config["continuation"]["methods"]:
        for seed in config["continuation"]["seeds"]:
            for budget in config["continuation"]["budgets_gpu_seconds"][method]:
                assert posttrain.checkpoint_key(method, seed, budget) in expected


def test_the_expected_selection_covers_every_declared_artifact(evaluate, config):
    names = evaluate.expected_selection(config)
    raw = sum(len(config["continuation"]["budgets_gpu_seconds"][method])
              for method in config["continuation"]["methods"]) * len(config["continuation"]["seeds"])
    policies = len(config["policy"]["arms"]) * len(config["policy"]["seeds"])
    assert len(names) == raw + policies + len(config["classifier"]["seeds"]) + 2
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# declared comparisons
# ---------------------------------------------------------------------------

def test_bootstrap_pairs_are_matched_not_quadratic(evaluate, config):
    freeze = {"selected": {name: {} for name in evaluate.expected_selection(config)}}
    pairs = evaluate.bootstrap_pairs(config, freeze)
    seeds = config["continuation"]["seeds"]
    shared = set(config["continuation"]["budgets_gpu_seconds"]["dpo"]) & \
        set(config["continuation"]["budgets_gpu_seconds"]["continued_sft"])
    for seed in seeds:
        for budget in shared:
            assert (f"dpo_seed{seed}_budget{int(budget)}",
                    f"continued_sft_seed{seed}_budget{int(budget)}") in pairs
        assert (f"dpo_seed{seed}_budget1800", f"policy_sft_seed{seed}") in pairs
    names = len(evaluate.expected_selection(config))
    assert len(pairs) < names * (names - 1) / 2, "the all-pairs table must not be computed"


def test_declared_budgets_are_the_announced_ones(config):
    budgets = config["continuation"]["budgets_gpu_seconds"]
    assert budgets["continued_sft"] == [180, 360, 600]
    assert budgets["dpo"] == [180, 360, 600, 1200, 1800]
    assert config["continuation"]["dpo"]["beta"] == 0.1
    assert config["continuation"]["optimization"]["learning_rate"] == 1e-5
    assert config["continuation"]["optimization"]["schedule"] == "linear_warmup_then_constant"


def test_the_configs_diversity_gates_match_the_preregistered_constants(config):
    from smallAntibodyGen.experiments import her2_eval as evaluation
    assert config["continuation"]["diversity_gates"] == dict(evaluation.DIVERSITY_GATES)


def test_the_expected_preference_counts_are_the_audited_ones(config):
    counts = config["continuation"]["expected_counts"]
    assert counts["train_chosen"] + counts["excluded_train_chosen"] == 120504
    assert counts["train_chosen"] + counts["train_rejected"] == counts["reference_rows"]
    assert counts["val_chosen"] + counts["excluded_val_chosen"] == 25822


def test_the_declared_tolerances_cover_the_measured_native_errors(config):
    """The probe measured these; the config must not be tighter than the hardware."""
    tolerances = config["tolerances"]
    assert tolerances["sum_log_probability_atol"] == policy_lib.SUM_LOG_PROBABILITY_ATOL
    assert tolerances["sum_log_probability_rtol"] == policy_lib.SUM_LOG_PROBABILITY_RTOL
    assert tolerances["sum_log_probability_atol"] > 3.35e-5, "worst measured full-vs-cached error"
    # The logit/gradient gate stays tight and separate: measured 6.20e-6 / 3.43e-6.
    assert tolerances["cached_vs_full_parity"] == pytest.approx(2e-5)
    assert tolerances["cached_vs_full_parity"] < tolerances["sum_log_probability_atol"]


def test_the_config_declares_the_cpu_threading_the_probes_used(config):
    assert config["runtime"]["cpu_threads"] == 4


# ---------------------------------------------------------------------------
# the shared sum-log-probability comparison
# ---------------------------------------------------------------------------

def test_the_score_comparison_accepts_fp32_noise_and_rejects_a_real_difference():
    expected = np.array([-31.0, -28.5, -36.4])
    noisy = expected + np.array([2.4e-5, -3.3e-5, 2.0e-5])
    report = policy_lib.compare_sum_log_probabilities(noisy, expected, label="probe")
    assert report["max_abs_error"] == pytest.approx(3.3e-5)
    assert report["rows_above_atol"] == 0
    with pytest.raises(ValueError, match="exceeds the declared tolerance"):
        policy_lib.compare_sum_log_probabilities(expected + 1e-3, expected, label="probe")


def test_the_score_comparison_rejects_nonfinite_values():
    """An allclose that ignored NaN would pass: NaN compares equal to nothing."""
    expected = np.array([-31.0, -28.5])
    for bad in (np.array([np.nan, -28.5]), np.array([np.inf, -28.5])):
        with pytest.raises(ValueError, match="nonfinite"):
            policy_lib.compare_sum_log_probabilities(bad, expected, label="probe")
    with pytest.raises(ValueError, match="compared"):
        policy_lib.compare_sum_log_probabilities(np.zeros(3), expected, label="probe")


# ---------------------------------------------------------------------------
# preflight: the batch is frozen, the device reading is not
# ---------------------------------------------------------------------------

PREFLIGHT = {"batch_size": 128, "fallback_batch_size": 64, "primary_min_free_mib": 2000,
             "fallback_min_free_mib": 1400}
FROZEN_AT_64 = {"device": "cuda", "batch_size": 64, "decision": "declared_fallback",
                "free_vram_mib": 1500.0}


def test_a_risen_free_vram_reading_never_raises_the_frozen_batch(monkeypatch):
    monkeypatch.setattr(policy_lib, "free_vram_mib", lambda: 3000.0)
    observation = policy_lib.recheck_preflight(PREFLIGHT, FROZEN_AT_64)
    assert observation["frozen_batch_size"] == 64
    assert observation["current_free_vram_mib"] == 3000.0
    assert observation["original_free_vram_mib"] == 1500.0


def test_a_fallen_free_vram_reading_stops_the_resume(monkeypatch):
    monkeypatch.setattr(policy_lib, "free_vram_mib", lambda: 900.0)
    with pytest.raises(ValueError, match="frozen at batch 64"):
        policy_lib.recheck_preflight(PREFLIGHT, FROZEN_AT_64)


def test_resume_rechecks_the_device_even_once_fitting_has_started(train, tmp_path, monkeypatch):
    """The early return on `fitting_has_started` skipped the check entirely, so a
    resumed run reported another day's free-VRAM reading as if it were current."""
    output = tmp_path / "campaign"
    (output / "policy_sft_seed1").mkdir(parents=True)
    (output / "policy_sft_seed1" / "run.json").write_text("{}", encoding="utf-8")
    (output / "preflight.json").write_text(json.dumps(FROZEN_AT_64), encoding="utf-8")
    assert train.fitting_has_started(output)
    monkeypatch.setattr(policy_lib, "free_vram_mib", lambda: 900.0)
    with pytest.raises(ValueError, match="frozen at batch 64"):
        train.resolve_preflight({"preflight": PREFLIGHT}, output, allow_cpu=False)
    monkeypatch.setattr(policy_lib, "free_vram_mib", lambda: 1600.0)
    resolved = train.resolve_preflight({"preflight": PREFLIGHT}, output, allow_cpu=False)
    assert resolved["batch_size"] == 64, "the frozen decision is what gets reused"
    assert resolved["current_check"]["current_free_vram_mib"] == 1600.0
    assert resolved["free_vram_mib"] == 1500.0, "the original reading stays labelled as original"


# ---------------------------------------------------------------------------
# the production parity probe, with an injected nonfinite gradient
# ---------------------------------------------------------------------------

class StubPolicy:
    """The CorePolicy surface `parity_probe` uses, over a real trainable parameter.

    Two routes to the same logits, exactly as the production policy has: ordinary
    teacher forcing and the shared-prefix cache. ``poison`` makes the cached route
    produce a nonfinite gradient, and ``flat`` makes both routes independent of the
    parameter so every gradient is exactly zero -- the two failures the probe is
    supposed to catch and previously could not.
    """

    def __init__(self, *, poison=False, flat=False):
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn(data.CORE_LENGTH, 20))
        self.model = torch.nn.Module()
        self.model.weight = self.weight
        self.device = torch.device("cpu")
        self.poison = poison
        self.flat = flat

    def token_ids(self, index):
        return torch.as_tensor(np.asarray(index), dtype=torch.long)

    def full_logits(self, core_ids):
        base = self.weight * 0.0 if self.flat else self.weight
        return base.unsqueeze(0).expand(core_ids.shape[0], -1, -1)

    def core_logits(self, core_ids):
        logits = self.full_logits(core_ids)
        return logits * float("nan") if self.poison else logits


def probe_index(rows=3):
    return np.tile(np.arange(data.CORE_LENGTH) % 20, (rows, 1)).astype(np.int64)


def test_the_parity_probe_passes_on_two_agreeing_routes(train):
    report = train.parity_probe(StubPolicy(), probe_index(), 1e-6)
    assert report["parameters_with_gradient"] == 1
    assert report["all_gradients_finite"] is True
    assert report["all_gradients_nonzero"] is True
    assert report["gradient_max_abs"] == pytest.approx(0.0)


def test_the_parity_probe_rejects_an_injected_nonfinite_gradient(train):
    with pytest.raises(ValueError, match="nonfinite gradient"):
        train.parity_probe(StubPolicy(poison=True), probe_index(), 1e-6)


def test_the_parity_probe_rejects_an_all_zero_gradient_comparison(train):
    """Two all-zero gradient sets agree perfectly and prove nothing about the paths."""
    with pytest.raises(ValueError, match="every gradient is exactly zero"):
        train.parity_probe(StubPolicy(flat=True), probe_index(), 1e-6)


# ---------------------------------------------------------------------------
# end-to-end stage contract on a synthetic campaign
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config(config):
    """The real protocol, narrowed to one seed and two budgets. Same code paths."""
    small = json.loads(json.dumps(config))
    small["policy"]["seeds"] = [1]
    small["classifier"]["seeds"] = [1]
    small["continuation"]["seeds"] = [1]
    small["continuation"]["budgets_gpu_seconds"] = {"continued_sft": [180], "dpo": [180, 360]}
    return small


class Campaign:
    """A miniature on-disk campaign: base freeze, records, draws, score vectors."""

    def __init__(self, root, posttrain, config):
        self.root = root
        self.posttrain = posttrain
        self.config = config
        self.output = root / "continuation"
        self.output.mkdir(parents=True, exist_ok=True)
        self.context = {"scaffold": SimpleNamespace(prefix="1AAA"),
                        "val_pairs": {"digest": "pairs-digest"}}
        self.identity = {"config_sha256": "c" * 64, "config_digest": "d" * 64,
                         "code_digests": {"scripts/posttrain_her2.py": "e" * 64},
                         "source_digests": {"specs/benchmarks/buzz_her2_affinity.json": "f" * 64},
                         "base_selection_sha256": "a" * 64}

    def write_checkpoint(self, name):
        path = self.root / "checkpoints" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return {"checkpoint": f"checkpoints/{name}.pt", "sha256": sha256(path)}

    def base_selection(self):
        selected = {}
        for name in self.posttrain.base_selection_names(self.config):
            kind = ("pinned_zero_shot" if name == "piggen_zeroshot"
                    else "policy" if name.startswith("policy_") else "classifier_proxy")
            selected[name] = dict(self.write_checkpoint(name), kind=kind)
        document = {"schema_version": data.SELECTION_SCHEMA,
                    "stage": data.SELECTION_STAGE_BASE,
                    "config_sha256": self.identity["config_sha256"],
                    "code_digests": self.identity["code_digests"],
                    "source_digests": self.identity["source_digests"],
                    "git_commit": "commit0",
                    "shared_initial_cost": {"unit": "wall seconds", "runs": {}},
                    "selected": selected}
        path = self.root / "base_selection.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return document, path

    def artifacts(self, base):
        runs = {}
        for method in self.config["continuation"]["methods"]:
            budgets = {}
            for budget in self.config["continuation"]["budgets_gpu_seconds"][method]:
                entry = self.write_checkpoint(f"{method}_seed1_budget{budget}")
                budgets[str(float(budget))] = {
                    "checkpoint": entry["checkpoint"], "checkpoint_sha256": entry["sha256"],
                    "target_gpu_seconds": float(budget),
                    "actual_gpu_seconds": float(budget) + 0.4, "updates": 12,
                    "overshoot_seconds": 0.4, "overshoot_within_one_update": True,
                    "exposures": {"sequences": 128}, "core_token_exposures": 1280,
                    "distinct_exposures": {"distinct_chosen_rows": 100}}
            runs[f"{method}_seed1"] = {"method": method, "seed": 1, "budgets": budgets}
        return self.posttrain.policy_artifacts(self.config, base, runs)

    def records(self, artifacts):
        self.context["generation_offsets"] = self.posttrain.generation_offsets(artifacts)
        records = {}
        for name, entry in sorted(artifacts.items()):
            draws = self.output / "generation" / f"draws_{name}.csv"
            draws.parent.mkdir(parents=True, exist_ok=True)
            draws.write_text("draw_index,core,sum_log_probability,mean_log_probability\n"
                             "0,ACDEFGHIKL,-31.0,-3.1\n", encoding="utf-8")
            scores = self.output / "validation_scores" / f"{name}.npy"
            scores.parent.mkdir(parents=True, exist_ok=True)
            np.save(scores, np.array([-3.1, -3.2]))
            records[name] = {
                "name": name, "sha256": entry["sha256"], "role": entry["role"],
                "method": entry.get("method"), "seed": entry.get("seed"),
                "budget_seconds": entry.get("budget_seconds", 0.0),
                "actual_gpu_seconds": entry.get("actual_gpu_seconds"),
                "identity": self.posttrain.validation_identity(
                    name, entry, self.config, self.identity, self.context),
                "val_metrics": {"average_precision": 0.4 + 0.001 * len(name)},
                "diversity": {"eligible": True},
                "generation_samples": {"path": f"continuation/generation/draws_{name}.csv",
                                       "sha256": sha256(draws), "draws": 1},
                "val_scores": {"path": f"continuation/validation_scores/{name}.npy",
                               "sha256": sha256(scores), "rows": 2}}
        for name, entry in artifacts.items():
            record = records[name]
            record["generation"] = {}
            references = [("kl_to_zero_shot", "val_metrics_minus_zero_shot", "piggen_zeroshot")]
            if entry.get("parent_name"):
                references.append(("kl_to_sft_parent", "val_metrics_minus_parent",
                                   entry["parent_name"]))
            for kl_key, rank_key, reference in references:
                record["generation"][kl_key] = {
                    "kl_nats": 0.0, "reference_name": reference,
                    "reference_sha256": artifacts[reference]["sha256"],
                    "draws_sha256": record["generation_samples"]["sha256"]}
                if name != reference:
                    record[rank_key] = {"reference_name": reference, "average_precision": 0.5}
        path = self.output / "validation_records.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return records


@pytest.fixture
def campaign(tmp_path, posttrain, small_config, monkeypatch):
    monkeypatch.setattr(posttrain, "ROOT", tmp_path)
    monkeypatch.setattr(posttrain, "git_commit", lambda: "commit0")
    built = Campaign(tmp_path, posttrain, small_config)
    base, base_path = built.base_selection()
    artifacts = built.artifacts(base)
    records = built.records(artifacts)
    return built, base, base_path, artifacts, records


def test_the_stage_contract_runs_base_to_freeze_to_outcome_gate(campaign, posttrain, evaluate,
                                                                small_config, tmp_path):
    """base freeze -> continuation records -> final freeze -> reserved-label gate."""
    built, base, base_path, artifacts, records = campaign
    # 1. The base freeze verifies, and its unlock opens nothing.
    _, base_unlock = data.read_selection_freeze(
        base_path, root=tmp_path, expected_config_sha256=built.identity["config_sha256"],
        expected_stage=data.SELECTION_STAGE_BASE,
        expected_selected=posttrain.base_selection_names(small_config),
        expected_code_digests=built.identity["code_digests"])
    assert base_unlock.stage == data.SELECTION_STAGE_BASE

    # 2. The final freeze accepts the verified records and names every artifact.
    document = posttrain.freeze_final(small_config, built.context, built.output, artifacts,
                                      records, base=base, identity_base=built.identity)
    assert document["stage"] == data.SELECTION_STAGE_FINAL
    assert set(document["selected"]) == set(evaluate.expected_selection(small_config))
    assert document["required"]["raw_budget_checkpoints"] == \
        evaluation.raw_budget_names(small_config)
    assert document["selection_within_budget"]["dpo_seed1"]["180"]["selected"] is not None

    # 3. The evidence check passes BEFORE anything reserved is touched.
    freeze, unlock = data.read_selection_freeze(
        built.output / "selection_frozen.json", root=tmp_path,
        expected_stage=data.SELECTION_STAGE_FINAL,
        expected_selected=evaluate.expected_selection(small_config),
        expected_code_digests=built.identity["code_digests"])
    evidence, validation = evaluate.verify_freeze_evidence(freeze, small_config, root=tmp_path)
    assert evidence["generation_files"] == len(records)
    assert evidence["validation_score_files"] == len(records)
    assert set(validation) == set(records)

    # 4. Only the final unlock opens the reserved split.
    split = tmp_path / data.SPLIT_DIR / "test.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("seq,class,label,edit_distance\n"
                     f"{data.WT_CORE},high,1,0\n", encoding="utf-8")
    with pytest.raises(data.Her2GuardError):
        data.load_split(tmp_path, "test", unlock=base_unlock, expect_counts=False)
    assert len(data.load_split(tmp_path, "test", unlock=unlock, expect_counts=False)) == 1


def test_a_stale_draw_file_stops_the_evaluation_before_any_label(campaign, posttrain, evaluate,
                                                                 small_config, tmp_path):
    built, base, _, artifacts, records = campaign
    posttrain.freeze_final(small_config, built.context, built.output, artifacts, records,
                           base=base, identity_base=built.identity)
    freeze = json.loads((built.output / "selection_frozen.json").read_text(encoding="utf-8"))
    name = sorted(freeze["generation"])[0]
    (tmp_path / freeze["generation"][name]["path"]).write_text("draw_index,core\n1,AAAAAAAAAA\n",
                                                               encoding="utf-8")
    with pytest.raises(ValueError, match="changed on disk"):
        evaluate.verify_freeze_evidence(freeze, small_config, root=tmp_path)


def test_a_missing_validation_score_vector_stops_the_evaluation(campaign, posttrain, evaluate,
                                                                small_config, tmp_path):
    built, base, _, artifacts, records = campaign
    posttrain.freeze_final(small_config, built.context, built.output, artifacts, records,
                           base=base, identity_base=built.identity)
    freeze = json.loads((built.output / "selection_frozen.json").read_text(encoding="utf-8"))
    (tmp_path / "continuation" / "validation_scores").rename(tmp_path / "moved_aside")
    with pytest.raises(ValueError, match="Persisted validation scores are missing"):
        evaluate.verify_freeze_evidence(freeze, small_config, root=tmp_path)


def test_the_freeze_refuses_a_record_whose_artifact_moved(campaign, posttrain, small_config,
                                                          tmp_path):
    """A record is evidence about specific bytes; if they changed it is not evidence."""
    built, base, _, artifacts, records = campaign
    name = sorted(n for n, e in artifacts.items() if e["role"] == "raw_budget")[0]
    (tmp_path / "continuation" / "generation" / f"draws_{name}.csv").write_text(
        "draw_index,core\n0,CCCCCCCCCC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be frozen"):
        posttrain.freeze_final(small_config, built.context, built.output, artifacts, records,
                               base=base, identity_base=built.identity)


def test_the_freeze_refuses_a_missing_budget_checkpoint(campaign, posttrain, small_config):
    built, base, _, artifacts, records = campaign
    name = "dpo_seed1_budget360"
    artifacts.pop(name)
    records.pop(name)
    with pytest.raises(ValueError, match="raw_budget_checkpoints do not match"):
        posttrain.freeze_final(small_config, built.context, built.output, artifacts, records,
                               base=base, identity_base=built.identity)


def test_the_freeze_refuses_a_record_from_a_different_identity(campaign, posttrain, small_config):
    built, base, _, artifacts, records = campaign
    name = sorted(records)[0]
    records[name]["identity"] = dict(records[name]["identity"], config_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity changed"):
        posttrain.freeze_final(small_config, built.context, built.output, artifacts, records,
                               base=base, identity_base=built.identity)


def test_the_base_selection_reader_rejects_stale_code_or_sources(campaign, posttrain,
                                                                 small_config):
    built, _, base_path, _, _ = campaign
    edited = dict(built.identity, code_digests={"scripts/posttrain_her2.py": "9" * 64})
    with pytest.raises(ValueError, match="different scientific code"):
        posttrain.read_base_selection(base_path, small_config, edited)
    moved = dict(built.identity,
                 source_digests={"specs/benchmarks/buzz_her2_affinity.json": "0" * 64})
    with pytest.raises(ValueError, match="different pinned sources"):
        posttrain.read_base_selection(base_path, small_config, moved)


@pytest.mark.parametrize("missing", ["kl_to_zero_shot", "kl_to_sft_parent",
                                     "val_metrics_minus_parent"])
def test_an_interrupted_reference_pass_cannot_be_frozen(campaign, posttrain, small_config, missing):
    built, base, _, artifacts, records = campaign
    record = records["dpo_seed1_budget180"]
    (record["generation"] if missing.startswith("kl_") else record).pop(missing)
    with pytest.raises(ValueError, match="incomplete"):
        posttrain.freeze_final(small_config, built.context, built.output, artifacts, records,
                               base=base, identity_base=built.identity)
    assert not (built.output / "selection_frozen.json").exists()


def test_kl_must_describe_the_persisted_draws(campaign):
    _, _, _, artifacts, records = campaign
    records["dpo_seed1_budget180"]["generation"]["kl_to_sft_parent"]["draws_sha256"] = "old"
    with pytest.raises(ValueError, match="stale kl_to_sft_parent"):
        evaluation.require_reference_diagnostics(records, artifacts)


def test_a_changed_parent_selection_invalidates_validation_reuse(campaign, posttrain, small_config):
    built, _, _, artifacts, records = campaign
    name = "dpo_seed1_budget180"
    changed = dict(built.identity, base_selection_sha256="new-parent-selection")
    identity = posttrain.validation_identity(name, artifacts[name], small_config, changed,
                                             built.context)
    assert posttrain.reusable_record(records[name], identity) == (False, "identity changed")
