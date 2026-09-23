"""Independent regressions for failures reproduced during the flight review."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smallAntibodyGen.experiments import her2_nf_calibration as calibration
from smallAntibodyGen.experiments import her2_nf_services as services


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_tail_rates_cannot_qualify(rate):
    entry = calibration.CalibrationEntry(
        "bad", "ipo_tail", "ipo", "tail", {"tau": 0.1, "lambda": 0.1},
        allocated_updates=500, completed_updates=500, status="completed",
        metrics={"macro_average_precision": 0.9, "tenfold_rate": rate,
                 "hundredfold_rate": 0.0})
    assert not calibration.qualifies(entry, parent_macro_ap=0.8)["qualified"]


def test_completed_label_cannot_substitute_for_completed_updates():
    entry = calibration.CalibrationEntry(
        "short", "ipo_tail", "ipo", "tail", {"tau": 0.1, "lambda": 0.1},
        allocated_updates=500, completed_updates=1, status="completed",
        metrics={"macro_average_precision": 0.9, "tenfold_rate": 0.0,
                 "hundredfold_rate": 0.0})
    assert not calibration.qualifies(entry, parent_macro_ap=0.8)["qualified"]


def test_calibration_fallback_selection_survives_ledger_restore():
    calls = []

    def runner(entry, **kwargs):
        calls.append(entry.entry_id)
        smallest = 3.0 if entry.preservation == "fkl" else 0.01
        failed_extension = (entry.allocated_updates == 1000
                            and entry.coefficients["lambda"] == smallest)
        return {"status": "completed", "completed_updates": entry.allocated_updates,
                "metrics": {"macro_average_precision": 0.9,
                            "tenfold_rate": 0.02 if failed_extension else 0.0,
                            "hundredfold_rate": 0.0, "yield_10k": 500.0,
                            "yield_1m": 5000.0}, "cost": {"gpu_seconds": 1.0}}

    first = calibration.run_calibration(runner, parent_macro_ap=0.8)
    before = list(calls)
    restored = calibration.CalibrationLedger.restore(first["ledger"])
    second = calibration.run_calibration(runner, parent_macro_ap=0.8, ledger=restored)
    assert calls == before
    assert second["frozen"] == first["frozen"]
    assert second["frozen"]["dpo_fkl"]["coefficients"]["lambda"] == 10.0
    assert second["frozen"]["ipo_tail"]["coefficients"]["lambda"] == 0.1
    assert second["frozen"]["dpo_tail"]["coefficients"]["lambda"] == 0.1


def test_an_expansion_grant_is_idempotent_even_before_its_pilot_runs():
    ledger = calibration.CalibrationLedger(max_bracket_expansions=1)
    first = ledger.request_expansion(family="dpo_fkl", coefficient="lambda -> 300",
                                     reason="the bracket was infeasible")
    restored = calibration.CalibrationLedger.restore(ledger.document())
    assert restored.request_expansion(family="dpo_fkl", coefficient="lambda -> 300",
                                      reason="resume") == first
    assert len(restored.expansions) == 1
    assert not restored.request_expansion(family="ipo_tail", coefficient="lambda -> 10",
                                          reason="a different request")["granted"]


def test_report_yield_compares_each_policy_to_its_actual_parent(tmp_path):
    runtime = services.FlightServices.__new__(services.FlightServices)
    runtime.context = SimpleNamespace(run_root=tmp_path,
                                      path=lambda *parts: tmp_path.joinpath(*parts))
    runtime.config = {"inference": {"score_batch_size": 3}}
    runtime._cache = {}
    entries = {}
    for name, arm, checkpoint, kind, update in (
            ("A_IPO_0@u1000_seed1", "IPO_0", "q", "preference_cell", 1000),
            ("parent_seed1", "parent", "p", "parent", 0)):
        entries[name] = {"status": "present", "regime": None, "checkpoint": checkpoint,
                         "kind": kind, "arm": arm, "seed": 1, "update": update}
    runtime.checkpoint_registry = lambda: {"entries": entries}
    runtime.split = lambda name: pd.DataFrame({"class": ["high", "mid", "low"]})
    runtime.index = lambda name: np.zeros((3, 10), dtype=np.int8)
    runtime.original_strata = lambda: {"labels": np.array(["1", "2", ">=3"])}

    def policy(checkpoint):
        score = np.array([-3.0, -4.0, -5.0]) if str(checkpoint) == "q" else \
            np.array([-5.0, -6.0, -7.0])
        return SimpleNamespace(score=lambda rows, batch_size: {
            "sum_log_probability": score, "mean_log_probability": score / 10.0})

    runtime.policy = policy
    runtime._mixture_evidence = lambda *args, **kwargs: None
    runtime._decomposition_evidence = lambda *args, **kwargs: []
    runtime._comparator_rankings = lambda *args, **kwargs: None
    report = runtime.report_evidence()
    curve = report["yield_curves"]["A_IPO_0@u1000_seed1"]
    np.testing.assert_array_equal(curve["log_probabilities"], [-3.0])
    np.testing.assert_array_equal(curve["control_log_probabilities"], [-5.0])
    assert "mixture089@1000" in report["metric_records"]
    assert "A_IPO_0@u1000_seed1" in report["rankings"]


def test_a_bracket_expansion_carries_its_yields_into_the_beta_comparison():
    """Reproduced: an expanded beta lost an AP tie to a worse-yielding one.

    ``_maybe_expand`` copied the widened selection's AP but not its yields, and
    ``select_beta`` reads a missing yield as ``-inf``. The expanded candidate
    therefore lost every AP tie no matter what it actually produced.
    """
    ledger = calibration.CalibrationLedger()

    def execute(entry, *, resume_from=None):
        # Everything inside the declared bracket is infeasible; the rung one step
        # outside it is feasible and improves on the parent.
        inside = float(entry.coefficients["lambda"]) <= max(calibration.FKL_LAMBDAS)
        entry.status = "completed"
        entry.completed_updates = entry.allocated_updates
        entry.metrics = {"macro_average_precision": 0.85,
                         "tenfold_rate": 0.5 if inside else 0.0,
                         "hundredfold_rate": 0.0,
                         "yield_10k": 1000.0, "yield_1m": 9000.0}
        entry.cost = {"gpu_seconds": 1.0}
        return ledger.record(entry)

    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS, fixed={"beta": 0.1})
    for entry in entries:
        execute(entry)
    entries = [ledger.completed(entry.entry_id) or entry for entry in entries]
    block = {"beta": 0.1, **calibration.select_lambda(entries, parent_macro_ap=0.8)}
    assert block["selected"] is None, "the declared bracket must produce nothing usable here"

    block = calibration._maybe_expand(
        ledger, execute, family="dpo_fkl_beta0.1", entries=entries, task="dpo",
        preservation="fkl", fixed={"beta": 0.1}, lambdas=calibration.FKL_LAMBDAS,
        parent_macro_ap=0.8, criteria=None, block=block)
    assert block["expansion"]["granted"] and block["selected"]
    assert block["yield_10k"] == 1000.0
    assert block["yield_1m"] == 9000.0

    # The tie-break now sees the measured yields rather than two absent ones.
    unexpanded = {"beta": 0.5, "selected": "L1:dpo_fkl_beta0.5_lambda3",
                  "macro_average_precision": 0.85, "yield_10k": 10.0, "yield_1m": 10.0}
    choice = calibration.select_beta([block, unexpanded])
    assert choice["selected_beta"] == 0.1
    assert choice["tie_broken_by"] == "Y@10k"


def _reuse_registry_runtime(tmp_path, *, reuse_verified):
    """A runtime whose only Block-A arm reuses a historical endpoint."""
    from smallAntibodyGen.experiments import her2_nf_campaign as campaign

    run_root = tmp_path / "run"
    campaign.write_stage_record(run_root, "preflight",
                                {"status": "completed", "reuse_verified": bool(reuse_verified)})
    runtime = services.FlightServices.__new__(services.FlightServices)
    runtime.context = SimpleNamespace(
        run_root=run_root, path=lambda *parts: run_root.joinpath(*parts))
    runtime.config = {
        "block_a": {"parent_seeds": [1],
                    "arms": [{"id": "IPO_0",
                              "reuse_historical": "trajectories/ipo_lambda0_seed{seed}"}],
                    "endpoint_updates": [1000], "checkpoint_updates": [1000]},
        "historical": {"continued_sft_updates": []},
        "block_b": {"regimes": [], "parent_seeds": [],
                    "endpoint_updates": [], "checkpoint_updates": []}}
    runtime._cache = {}
    runtime.parent_checkpoint = lambda seed: (None, {})
    runtime.historical_root = lambda: tmp_path / "historical"
    runtime._missing_checkpoint_reason = lambda trajectory: (
        f"stopped_by_gate at update 200: the replacement for {trajectory} stopped early")

    import torch
    from smallAntibodyGen.experiments import her2_nf_storage as storage
    old = tmp_path / "historical" / "trajectories" / "ipo_lambda0_seed1" / "endpoint_update1000.pt"
    old.parent.mkdir(parents=True, exist_ok=True)
    state = {"weight": torch.zeros(2)}
    torch.save({"state": state, "state_sha256": storage.state_dict_digest(state),
                "schema_version": "her2-next-flight/1", "update": 1000, "exposures": {}}, old)
    return runtime, old


def test_a_rejected_reuse_never_substitutes_the_historical_endpoint(tmp_path):
    """Reproduced: parity failed, the replacement stopped early, and the registry
    quietly published the disqualified historical checkpoint as that cell."""
    runtime, old = _reuse_registry_runtime(tmp_path, reuse_verified=False)
    entry = runtime.checkpoint_registry()["entries"]["A_IPO_0@u1000_seed1"]
    assert entry["status"] == "missing"
    assert entry["reused"] is None
    assert entry["checkpoint"] is None
    assert "stopped_by_gate at update 200" in entry["reason"]


def test_a_verified_reuse_still_supplies_the_historical_endpoint(tmp_path):
    runtime, old = _reuse_registry_runtime(tmp_path, reuse_verified=True)
    entry = runtime.checkpoint_registry()["entries"]["A_IPO_0@u1000_seed1"]
    assert entry["status"] == "present"
    assert entry["reused"] == "historical"
    assert entry["checkpoint"] == str(old)


def test_the_saved_monitor_vector_is_found_under_the_declared_trajectory_path(tmp_path):
    """Reproduced against the real completed campaign: the parity check passes
    ``trajectories/ipo_lambda0_seed{seed}`` -- the form the config declares and
    the form the journal lookup beside it already uses -- and the loader prepended
    a second ``trajectories``. Every saved vector read as absent."""
    from smallAntibodyGen.experiments import her2_nf_reuse as reuse
    from smallAntibodyGen.experiments import her2_support_paths as paths

    trajectory = "trajectories/ipo_lambda0_seed1"
    target = (tmp_path / trajectory / "monitor_scores" / "check_00001_update25.npz")
    paths.write_arrays(target, {"chosen": np.arange(4, dtype=np.float64)})

    loaded = reuse.load_monitor_scores(tmp_path, trajectory, update=25)
    assert loaded is not None, "the declared trajectory path must find the saved vector"
    assert loaded["path"] == str(target)
    assert reuse.load_monitor_scores(tmp_path, trajectory, update=50) is None


def test_an_absent_monitor_vector_cannot_report_itself_as_passed():
    """The gate's record CLAIMS the score vector was compared. When there is no
    vector to compare, that is an uncompared requirement -- the arm becomes an
    external reference and is retrained -- never a silent pass."""
    runtime = services.FlightServices.__new__(services.FlightServices)
    runtime.historical_root = lambda: Path("nowhere")
    monitor = services.FlightServices._monitor_vector_parity(
        runtime, "trajectories/ipo_lambda0_seed1", policy=None, update=25,
        atol=1e-5, rtol=1e-6)
    assert monitor["available"] is False
    assert monitor["passed"] is False
    assert "did not run" in monitor["reason"]


def test_a_superseded_supervisor_failure_is_not_the_current_state():
    """Reproduced: the supervisor writes supervisor_failure.json on a crash and
    never clears it, so a resumed flight that finished kept reporting a failure
    once its lock was released."""
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "her2_live_dashboard", root / "scripts" / "her2_live_dashboard.py")
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)

    crash = {"at": "2026-09-22T23:35:49+00:00", "attempt": 1}
    finished_later = {"status": "completed", "finished_at": "2026-09-23T04:00:00+00:00"}
    finished_earlier = {"status": "completed", "finished_at": "2026-09-22T22:00:00+00:00"}

    assert not dashboard.failure_is_current(crash, finished_later)
    assert dashboard.failure_is_current(crash, finished_earlier)
    assert not dashboard.failure_is_current(None, finished_later)
    # An unreadable pair of stamps is treated as current: not knowing which came
    # last is not evidence that the crash was recovered.
    assert dashboard.failure_is_current(crash, {"finished_at": "not a timestamp"})
    assert dashboard.failure_is_current({"attempt": 1}, finished_later)


def test_the_report_and_verify_stage_records_stay_out_of_the_manifest():
    """Reproduced: run_stage writes stages/<stage>.json AFTER the handler returns,
    so a second reporting pass hashed three files the pipeline was about to
    rewrite and the verification then rejected its own output."""
    from smallAntibodyGen.experiments import her2_nf_report as report_lib

    for mutable in ("queue.json", "stages/report.json", "stages/verify.json"):
        assert not report_lib.is_immutable_artifact(mutable), mutable
    for evidence in ("stages/geometry.json", "stages/calibrate.json", "stages/preflight.json",
                     "calibration_ledger.json", "split_manifest.json"):
        assert report_lib.is_immutable_artifact(evidence), evidence


def test_a_provisional_finalist_freeze_is_retaken_once_production_advances():
    """Reproduced: the freeze taken over a partial queue was returned unchanged
    after production advanced, so the report published a finalist list chosen over
    a queue that had since moved."""
    partial = {"named_checkpoints": [{"name": "A_IPO_FKL@u1000_seed1"}],
               "provisional": True, "production_outcome_sha256": "a" * 64}
    assert services._freeze_still_stands(partial, production_outcome_sha256="a" * 64)
    assert not services._freeze_still_stands(partial, production_outcome_sha256="b" * 64)
    final = dict(partial, provisional=False)
    assert services._freeze_still_stands(final, production_outcome_sha256="b" * 64), \
        "a freeze taken over a terminal queue is final and is never retaken"
