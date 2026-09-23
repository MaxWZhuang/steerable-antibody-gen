import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from her2_nf_support import TinyPolicy, tiny_cores
from smallAntibodyGen.experiments import her2_nf_campaign as campaign
from smallAntibodyGen.experiments import her2_nf_coupling as coupling
from smallAntibodyGen.experiments import her2_nf_stage1 as stage1
from smallAntibodyGen.experiments import her2_nf_storage as storage
from smallAntibodyGen.experiments import her2_nf_trajectory as trajectory
from smallAntibodyGen.experiments import her2_nf_calibration as calibration
from smallAntibodyGen.experiments import her2_nf_services as services


def test_total_correlation_bootstrap_matches_explicit_whole_row_resampling():
    rng = np.random.default_rng(55)
    q = rng.dirichlet(np.ones(4), size=(9, 3))
    logs = np.log(q)
    observed = coupling.bootstrap_total_correlation(logs, seed=18, draws=13, batch_size=4)
    rng = np.random.default_rng(18)
    explicit = [coupling.total_correlation(logs[rng.integers(9, size=9)])["total_correlation"]
                for _ in range(13)]
    np.testing.assert_allclose(observed["values"], explicit, atol=2e-15, rtol=1e-13)


def test_stage1_interruption_replays_cursor_and_restores_full_state(tmp_path):
    index = tiny_cores(24)
    settings = {"epochs": 2, "batch_size": 8, "microbatch_rows": 4,
                "checkpoints": [1, 2], "resume_interval_updates": 2}
    optimization = {"learning_rate": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.01,
                    "gradient_clip": 1.0, "warmup_fraction": 0.05,
                    "final_learning_rate_fraction": 0.1}

    def run(directory, *, callback=None, values=index):
        policy = TinyPolicy(seed=7)
        policy.prefix_ids = torch.tensor([[1, 2]])
        return stage1.fit(policy, index=values, seed=18, directory=directory,
                          plan_config=settings, optimization=optimization,
                          selection_index=index[:8], selection_labels=np.ones(8, dtype=bool),
                          source_sha256="fixture", after_update=callback)

    expected = run(tmp_path / "reference")

    def crash(update):
        if update == 3:
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected"):
        run(tmp_path / "resumed", callback=crash)
    actual = run(tmp_path / "resumed")
    a = storage.load_cpu(tmp_path / "reference/stage1_state.pt")
    b = storage.load_cpu(tmp_path / "resumed/stage1_state.pt")
    assert a["state_sha256"] == b["state_sha256"]
    for key in ("optimizer", "scheduler", "rng"):
        assert trajectory._payload_digest({key: a[key]}) == trajectory._payload_digest({key: b[key]})
    assert actual["selected_epoch"] == expected["selected_epoch"]
    with pytest.raises(ValueError, match="population"):
        run(tmp_path / "resumed", values=np.roll(index, 1, axis=0))


def test_unqualified_families_leave_controls_and_freeze_honest_b_tail_fallback():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/experiments/her2_next_flight.json").read_text())
    frozen = campaign._flatten_frozen({
        "dpo_beta": 0.1, "dpo_fkl": {"coefficients": None},
        "ipo_tail": {"coefficients": None}, "dpo_tail": {"coefficients": {"lambda": 0.1}},
        "block_b_dpo": {"family": "dpo_tail"}})
    queue = campaign.build_queue(config, frozen=frozen, reuse_verified=True)
    assert any(row.block == "B" and row.arm == "DPO_TAIL" and row.status == "queued" for row in queue)
    assert not any(row.block == "B" and row.arm == "DPO_FKL" for row in queue)
    assert all(row.status == "no_qualified_configuration" for row in queue
               if row.block == "A" and row.arm in ("DPO_FKL", "IPO_TAIL"))
    assert sum(row.block == "B" and row.arm in ("IPO_0", "IPO_FKL") and row.status == "queued"
               for row in queue) == 12
    with pytest.raises(ValueError, match="calibrated"):
        campaign.build_queue(config, frozen={})


def test_calibration_baseline_cost_survives_restore_without_double_charge():
    ledger = calibration.CalibrationLedger(gpu_hour_cap=1)
    ledger.record_overhead({"measured_seconds": 25.0, "uncertainty_debit_seconds": 30.0})
    restored = calibration.CalibrationLedger.restore(ledger.document())
    restored.record_overhead({"measured_seconds": 25.0, "uncertainty_debit_seconds": 30.0})
    assert restored.measured_gpu_seconds == 25.0
    assert restored.remaining_seconds() == pytest.approx(3545.0)


def test_b_fallback_contrast_and_parent_lineage_use_the_actual_family():
    scores = {f"{arm}@{regime}": {seed: value for seed in (1, 2, 3)}
              for arm, regime, value in [("DPO_TAIL", "purge", .6), ("IPO_FKL", "purge", .5),
                                        ("DPO_TAIL", "matched", .9), ("IPO_FKL", "matched", .5)]}
    result = services._difference_in_differences(scores, dpo_arm="DPO_TAIL")
    assert result["available"] and "DPO_TAIL" in result["estimand"]
    parent = {"checkpoint": "new.pt", "state_sha256": "new"}
    row = {"name": "child", "checkpoint": "child.pt", "regime": "purge", "seed": 1,
           "parent_id": "new", "update": 1000, "arm": "DPO_TAIL", "kind": "preference_cell"}
    frozen = services._freeze_entry(row, {"B_purge_parent_seed1": parent,
                                         "parent_seed1": {"state_sha256": "old"}})
    assert frozen["parent_state_sha256"] == "new" and frozen["parent_checkpoint"] == "new.pt"


def test_gate_journal_before_state_crash_remains_a_stop(tmp_path, monkeypatch):
    from smallAntibodyGen.tests.test_her2_nf_trajectory import Harness
    original = trajectory.TrajectoryStateStore.save

    def fail_save(self, **kwargs):
        if kwargs["state"].update == 2:
            raise RuntimeError("between failed gate and saved state")
        return original(self, **kwargs)

    monkeypatch.setattr(trajectory.TrajectoryStateStore, "save", fail_save)
    first = Harness(tmp_path, updates=4, gate=lambda **kw: {
        "passed": kw["update"] < 2, "D": 2.0, "stop_reason": "recorded scientific stop"})
    with pytest.raises(RuntimeError, match="between failed gate"):
        first.run()
    monkeypatch.setattr(trajectory.TrajectoryStateStore, "save", original)
    resumed = Harness(tmp_path, updates=4, gate=lambda **kw: {"passed": True, "D": 0.0})
    result = resumed.run()
    assert result["status"] == trajectory.STATUS_STOPPED and result["updates"] == 2
    assert trajectory.durable_progress(tmp_path)["updates"] == 2


def test_challenge_mixtures_use_the_matching_fresh_parent_and_fixed_e(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from smallAntibodyGen.experiments import her2_nf_spec as spec
    root = Path(__file__).resolve().parents[3]
    context = spec.resolve_context(root, config_path=root / "configs/experiments/her2_next_flight.json",
                                    run_root=tmp_path)
    runtime = services.FlightServices(context, device="cpu")
    index = tiny_cores(9)
    classes = np.array(["high", "mid", "low"] * 3)
    runtime.load_populations = lambda: {"evaluation_rows": np.arange(9), "manifest": {
        "evaluation_panel": {"original_train_distance": [1, 2, -1] * 3}}}
    runtime.split = lambda _: pd.DataFrame({"class": classes})
    runtime.index = lambda _: index
    runtime._finalist_freeze = lambda: {"named_checkpoints": []}
    runtime._challenge_parent_if_fitted = lambda regime, seed: f"{regime}/parent{seed}"
    runtime._endpoint_checkpoint = lambda name, update: name if "IPO_0" in name else None
    runtime._load_comparator_models = lambda regime: None
    runtime._screen_marginals = lambda name: None

    def scores(path):
        value = (np.linspace(-11, -20, 9) if "parent" in str(path) else np.linspace(-22, -14, 9))
        value = value - (2 if "purge" in str(path) else 3)
        return {"sum_log_probability": value, "mean_log_probability": value / 10}

    runtime.policy = lambda path: SimpleNamespace(score=lambda index, **kwargs: scores(path))
    monkeypatch.setattr(services.nf_reuse, "checkpoint_metadata", lambda path: {"state_sha256": "fixture"})
    result = runtime.evaluate_challenge()
    mixtures = [entry for entry in result["records"] if entry["arm"] == "mixture089"]
    assert len(mixtures) == 6 and len(result["mixture_curves"]) == 6
    assert len(result["parent_contrasts"]) == 4
    assert all(entry["available"] and len(entry["paired_seeds"]) == 3 for entry in result["parent_contrasts"])
    for entry in mixtures:
        with np.load(context.path(entry["row_scores"])) as arrays:
            expected = np.logaddexp(np.log(.11) + arrays["parent_sum_log_probability"],
                                   np.log(.89) + arrays["policy_sum_log_probability"])
            np.testing.assert_allclose(arrays["sum_log_probability"], expected, atol=1e-14)
            assert arrays["source_split"].item() == "val:E"
            assert entry["regime"] in arrays["parent_checkpoint"].item()
            assert len(arrays["row_id"]) == 9
