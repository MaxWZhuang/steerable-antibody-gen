"""The bounded ladders: the selection rule, the allowances, and the honest failure.

Every test here runs the real policy with a stub runner. That is the point: the
allocation logic is what decides whether the flight spends its GPU budget on a
rescue search, and it has to be checkable without one.
"""
from __future__ import annotations

import pytest

from smallAntibodyGen.experiments import her2_nf_calibration as calibration


def _runner(table, *, gpu_seconds=60.0, status="completed", seen=None):
    """A stub pilot: metrics looked up by entry id, with a measured cost.

    An extension carries the base setting's id with an ``EXT:`` prefix, so the
    lookup strips it -- a 1000-update continuation of a setting is that setting,
    not a new one with no metrics.
    """

    def run(entry, *, resume_from=None, budget_seconds=None):
        if seen is not None:
            seen.append({"entry_id": entry.entry_id, "resume_from": resume_from,
                         "allocated": entry.allocated_updates})
        metrics = dict(table.get(calibration.extension_of(entry.entry_id)
                                 or entry.entry_id, {}))
        return {"status": status, "completed_updates": entry.allocated_updates,
                "metrics": metrics, "cost": {"gpu_seconds": gpu_seconds}}
    return run


def _feasible(ap, *, tenfold=0.005, hundredfold=0.0005, yield_10k=500.0, yield_1m=5000.0):
    return {"macro_average_precision": ap, "tenfold_rate": tenfold,
            "hundredfold_rate": hundredfold, "yield_10k": yield_10k, "yield_1m": yield_1m}


def test_nominal_plan_matches_the_declared_seventy_five_hundred_updates():
    plan = calibration.nominal_plan()
    assert len(plan["stage_one"]) == 6
    assert plan["nominal_updates"] == 6 * 500 + 6 * 500 + 3 * 500 == 7500


def test_the_smallest_feasible_lambda_that_beats_the_parent_is_selected():
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=(3.0, 10.0, 30.0), betas=(0.5,))
    for entry, metrics in zip(entries, [
            _feasible(0.860, tenfold=0.02),        # infeasible tails
            _feasible(0.872),                      # feasible and better than the parent
            _feasible(0.880)]):                    # better still, but larger lambda
        entry.metrics, entry.status = metrics, "completed"
        entry.completed_updates = entry.allocated_updates
    choice = calibration.select_lambda(entries, parent_macro_ap=0.8539)
    assert choice["coefficients"]["lambda"] == 10.0
    assert "smallest lambda" in choice["rule"]
    assert choice["ranked_alternatives"] == [entries[2].entry_id]
    # Every candidate is listed, including the one that failed.
    assert len(choice["considered"]) == 3
    assert choice["considered"][0]["feasible"] is False


def test_a_feasible_setting_that_does_not_beat_the_parent_is_not_selected():
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=(3.0,), betas=(0.1,))
    entries[0].metrics, entries[0].status = _feasible(0.80), "completed"
    choice = calibration.select_lambda(entries, parent_macro_ap=0.8539)
    assert choice["selected"] is None
    assert choice["outcome"] == "no qualified calibrated configuration for this family"
    assert "never relabelled as a tuned success" in choice["consequence"]


def test_beta_is_compared_by_ap_then_yields_then_the_smaller_beta():
    tied = [{"beta": 0.5, "selected": "b", "macro_average_precision": 0.8700,
             "yield_10k": 500.0, "yield_1m": 5000.0},
            {"beta": 0.1, "selected": "a", "macro_average_precision": 0.8705,
             "yield_10k": 500.0, "yield_1m": 5000.0}]
    choice = calibration.select_beta(tied)
    assert choice["selected_beta"] == 0.1
    assert choice["tie_broken_by"] == "smaller beta"

    separated = [{"beta": 0.5, "selected": "b", "macro_average_precision": 0.880,
                  "yield_10k": 1.0, "yield_1m": 1.0},
                 {"beta": 0.1, "selected": "a", "macro_average_precision": 0.870,
                  "yield_10k": 900.0, "yield_1m": 9000.0}]
    assert calibration.select_beta(separated)["selected_beta"] == 0.5

    by_yield = [{"beta": 0.5, "selected": "b", "macro_average_precision": 0.8700,
                 "yield_10k": 10.0, "yield_1m": 1.0},
                {"beta": 0.1, "selected": "a", "macro_average_precision": 0.8705,
                 "yield_10k": 900.0, "yield_1m": 9000.0}]
    picked = calibration.select_beta(by_yield)
    assert picked["selected_beta"] == 0.1 and picked["tie_broken_by"] == "Y@10k"


def test_no_feasible_beta_is_an_explicit_outcome():
    choice = calibration.select_beta([{"beta": 0.1, "selected": None},
                                      {"beta": 0.5, "selected": None}])
    assert choice["selected_beta"] is None
    assert choice["outcome"] == "no feasible DPO beta"


def test_the_ledger_records_settings_the_cap_prevented_from_starting():
    ledger = calibration.CalibrationLedger(gpu_hour_cap=0.05)      # 180 seconds
    table = {entry.entry_id: _feasible(0.87) for entry in calibration.ladder_entries(
        stage="L1", task="dpo", preservation="fkl", lambdas=calibration.FKL_LAMBDAS,
        betas=calibration.DPO_BETAS)}
    outcome = calibration.run_calibration(_runner(table, gpu_seconds=100.0),
                                          parent_macro_ap=0.85, ledger=ledger)
    document = outcome["ledger"]
    assert document["attempted"] > document["completed"]
    assert document["not_started"] >= 1
    not_started = [entry for entry in document["entries"] if entry["status"] == "not_started"]
    assert "cap was reached" in not_started[0]["reason"]
    assert "survivor-only table is not produced" in document["completeness"]


def test_bracket_expansions_are_bounded_and_the_refusal_says_why():
    ledger = calibration.CalibrationLedger()
    first = ledger.request_expansion(family="ipo_tail", coefficient="lambda", reason="inert")
    second = ledger.request_expansion(family="dpo_tail", coefficient="lambda", reason="inert")
    third = ledger.request_expansion(family="dpo_fkl", coefficient="lambda", reason="inert")
    assert first["granted"] and second["granted"]
    assert third["granted"] is False
    assert "No unbounded rescue search" in third["reason"]
    # Behaviour, not prose: two are granted, the allowance is recorded, and the
    # third is refused with the spent allowance named.
    assert [entry["index"] for entry in ledger.expansions] == [1, 2]
    assert ledger.document()["bracket_expansion_allowance"] == 2
    assert str(ledger.max_bracket_expansions) in third["reason"]


def test_one_fallback_per_family_and_no_more():
    ledger = calibration.CalibrationLedger()
    assert ledger.request_fallback(family="dpo_fkl", reason="x")["granted"] is True
    assert ledger.request_fallback(family="dpo_fkl", reason="x")["granted"] is False
    assert ledger.request_fallback(family="ipo_tail", reason="x")["granted"] is True


def test_a_full_run_freezes_one_coefficient_per_family_and_transfers_it_unchanged():
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS,
                                         betas=calibration.DPO_BETAS)
    table = {}
    for entry in entries:
        beta = entry.coefficients["beta"]
        lam = entry.coefficients["lambda"]
        table[entry.entry_id] = _feasible(0.86 + (0.01 if beta == 0.1 else 0.0) - lam * 1e-4)
    for task in ("ipo", "dpo"):
        for entry in calibration.ladder_entries(
                stage="L2", task=task, preservation="tail", lambdas=calibration.TAIL_LAMBDAS,
                betas=(0.1,) if task == "dpo" else (None,),
                fixed={"tau": 0.1} if task == "ipo" else None):
            table[entry.entry_id] = _feasible(0.865)
    outcome = calibration.run_calibration(_runner(table), parent_macro_ap=0.8539)
    assert outcome["beta_choice"]["selected_beta"] == 0.1
    frozen = outcome["frozen"]
    assert frozen["dpo_beta"] == 0.1
    assert "Block B transfers these coefficients unchanged" in frozen["transfer_rule"]
    assert "no per-seed retuning" in frozen["per_seed_rule"]
    # Extensions to 1000 happened for the extendable families.
    assert outcome["extensions"]
    extended = [entry for entry in outcome["ledger"]["entries"]
                if entry["entry_id"].startswith("EXT:")]
    assert extended and all(entry["allocated_updates"] == 1000 for entry in extended)


def test_an_extension_continues_its_pilot_rather_than_restarting_it():
    """A 1000-update extension resumes at 500; it does not spend 1000 fresh updates."""
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS, betas=(0.1,))
    table = {entry.entry_id: _feasible(0.87) for entry in entries}
    seen = []
    calibration.run_calibration(_runner(table, seen=seen), parent_macro_ap=0.85)
    extensions = [entry for entry in seen if entry["entry_id"].startswith("EXT:")]
    assert extensions, "the selected candidate was extended"
    for entry in extensions:
        assert entry["resume_from"] == calibration.extension_of(entry["entry_id"])
        assert entry["allocated"] == calibration.EXTENDED_UPDATES


def test_an_extended_candidate_that_fails_at_1000_takes_its_one_fallback():
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS, betas=(0.1,))
    table = {entry.entry_id: _feasible(0.87) for entry in entries}
    for task in ("ipo", "dpo"):
        for entry in calibration.ladder_entries(
                stage="L2", task=task, preservation="tail", lambdas=calibration.TAIL_LAMBDAS,
                betas=(0.1,) if task == "dpo" else (None,),
                fixed={"tau": 0.1} if task == "ipo" else None):
            table[entry.entry_id] = _feasible(0.865)
    failing = {entries[0].entry_id}                # only the smallest lambda fails at 1000

    def run(entry, *, resume_from=None, budget_seconds=None):
        base = calibration.extension_of(entry.entry_id) or entry.entry_id
        metrics = dict(table.get(base, {}))
        if calibration.extension_of(entry.entry_id) in failing:
            metrics["tenfold_rate"] = 0.05        # feasible at 500, not at 1000
        return {"status": "completed", "completed_updates": entry.allocated_updates,
                "metrics": metrics, "cost": {"gpu_seconds": 10.0}}

    outcome = calibration.run_calibration(run, parent_macro_ap=0.85)
    block = outcome["extensions"]["dpo_fkl"]
    attempts = [entry for entry in block["attempts"] if entry.get("entry_id")]
    assert attempts[0]["qualified"] is False
    assert outcome["ledger"]["fallbacks"]["dpo_fkl"] == 1
    # The fallback really ran the NEXT ranked candidate and it qualified.
    assert attempts[1]["continued_from"] == entries[1].entry_id
    assert attempts[1]["qualified"] is True
    assert block["frozen_entry"] == attempts[1]["entry_id"]
    assert outcome["frozen"]["dpo_fkl"]["coefficients"]["lambda"] == 10.0


def test_a_family_whose_continuation_fails_freezes_nothing():
    """All 500-update pilots feasible and better than the parent, every 1000 fails.

    The reproduced defect: three families froze coefficients anyway. No qualified
    configuration is a recorded outcome and the arms that cite it stay pending.
    """
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS,
                                         betas=calibration.DPO_BETAS)
    table = {entry.entry_id: _feasible(0.9) for entry in entries}
    for task in ("ipo", "dpo"):
        for entry in calibration.ladder_entries(
                stage="L2", task=task, preservation="tail", lambdas=calibration.TAIL_LAMBDAS,
                betas=(0.1,) if task == "dpo" else (None,),
                fixed={"tau": 0.1} if task == "ipo" else None):
            table[entry.entry_id] = _feasible(0.9)

    def run(entry, *, resume_from=None, budget_seconds=None):
        base = calibration.extension_of(entry.entry_id)
        metrics = dict(table.get(base or entry.entry_id, {}))
        if base is not None:
            metrics["tenfold_rate"] = 0.02       # every 1000-update extension fails
        return {"status": "completed", "completed_updates": entry.allocated_updates,
                "metrics": metrics, "cost": {"gpu_seconds": 5.0}}

    outcome = calibration.run_calibration(run, parent_macro_ap=0.8)
    for family in ("dpo_fkl", "ipo_tail", "dpo_tail"):
        assert outcome["frozen"][family]["coefficients"] is None, family
        assert "no qualified" in outcome["frozen"][family]["outcome"], family
        assert "no_qualified_configuration" in outcome["frozen"][family]["consequence"]
    assert "COMPLETED 1,000 updates" in outcome["freeze_rule"]


def test_a_machinery_failure_aborts_instead_of_becoming_a_failed_pilot():
    """A broken path must not make a whole ladder look infeasible."""

    def run(entry, *, resume_from=None, budget_seconds=None):
        raise calibration.MachineryFailure("the replay bank is missing")

    with pytest.raises(calibration.MachineryFailure, match="replay bank"):
        calibration.run_calibration(run, parent_macro_ap=0.85)


def test_an_unmeasured_pilot_cost_is_refused_rather_than_counted_as_zero():
    ledger = calibration.CalibrationLedger(gpu_hour_cap=1.0)

    def run(entry, *, resume_from=None, budget_seconds=None):
        return {"status": "completed", "completed_updates": 500,
                "metrics": _feasible(0.87), "cost": {}}

    with pytest.raises(ValueError, match="no measured gpu_seconds"):
        calibration.run_calibration(run, parent_macro_ap=0.85, ledger=ledger)


def test_the_ledger_is_persisted_and_a_resumed_stage_does_not_re_spend_it():
    """Restarting with a fresh ledger would spend the three-hour allocation twice."""
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS, betas=(0.1,))
    table = {entry.entry_id: _feasible(0.87) for entry in entries}
    written = []
    ledger = calibration.CalibrationLedger(persist=written.append)
    first_seen = []
    calibration.run_calibration(_runner(table, seen=first_seen), parent_macro_ap=0.85,
                               ledger=ledger)
    assert written, "the ledger persisted after every recorded entry"
    restored = calibration.CalibrationLedger.restore(written[-1])
    assert restored.measured_gpu_hours == pytest.approx(ledger.measured_gpu_hours)
    assert len(restored.entries) == len(ledger.entries)
    second_seen = []
    calibration.run_calibration(_runner(table, seen=second_seen), parent_macro_ap=0.85,
                               ledger=restored)
    assert second_seen == [], "every setting was already measured and was not rerun"
    assert restored.measured_gpu_hours == pytest.approx(ledger.measured_gpu_hours)


def test_a_bracket_expansion_is_requested_when_a_whole_ladder_is_infeasible():
    """``request_expansion`` was never called from anywhere."""
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS, betas=(0.1,))
    table = {entry.entry_id: _feasible(0.9, tenfold=0.5) for entry in entries}
    seen = []
    ledger = calibration.CalibrationLedger()
    outcome = calibration.run_calibration(_runner(table, seen=seen), parent_macro_ap=0.85,
                                          ledger=ledger)
    assert ledger.expansions, "an infeasible ladder asked for its bounded expansion"
    assert any(entry["entry_id"].startswith("X1:") for entry in seen)
    granted = ledger.expansions[0]
    assert "infeasible" in granted["reason"]
    assert outcome["ledger"]["bracket_expansions"]


def test_the_dpo_tail_ladder_is_not_run_at_an_unselected_beta():
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS,
                                         betas=calibration.DPO_BETAS)
    table = {entry.entry_id: _feasible(0.80) for entry in entries}   # nothing beats the parent
    for entry in calibration.ladder_entries(stage="L2", task="ipo", preservation="tail",
                                            lambdas=calibration.TAIL_LAMBDAS,
                                            fixed={"tau": 0.1}):
        table[entry.entry_id] = _feasible(0.87)
    outcome = calibration.run_calibration(_runner(table), parent_macro_ap=0.8539)
    assert outcome["beta_choice"]["selected_beta"] is None
    decisions = {entry["decision"] for entry in outcome["ledger"]["decisions"]}
    assert "dpo_tail_ladder" in decisions
    assert "dpo_tail" not in outcome["family_outcomes"]
    families = [entry for entry in outcome["ledger"]["decisions"]
                if entry["decision"] == "dpo_fkl_family"]
    assert families and "before any E scoring" in families[0]["consequence"]


def test_non_dominated_points_are_retained_across_all_four_axes():
    entries = calibration.ladder_entries(stage="L1", task="ipo", preservation="tail",
                                         lambdas=(0.01, 0.1, 1.0), fixed={"tau": 0.1})
    entries[0].metrics = _feasible(0.88, tenfold=0.002, yield_10k=600.0)
    entries[1].metrics = _feasible(0.86, tenfold=0.001, yield_10k=400.0)
    entries[2].metrics = _feasible(0.85, tenfold=0.004, yield_10k=300.0)
    block = calibration.non_dominated(entries)
    assert entries[0].entry_id in block["non_dominated"]
    assert entries[1].entry_id in block["non_dominated"]   # best tail, so not dominated
    assert entries[2].entry_id not in block["non_dominated"]


def test_feasibility_uses_the_inclusive_criteria_and_records_its_reason():
    block = calibration.is_feasible({"tenfold_rate": 0.02, "hundredfold_rate": 0.0})
    assert block["feasible"] is False
    assert block["reasons"][0].startswith("tenfold_rate")
    assert block["tail_convention"] == ">= (inclusive)"
    missing = calibration.is_feasible({})
    assert "not measured" in missing["reasons"][0]


def test_simpo_allowance_is_recorded_separately_when_it_is_admitted():
    entries = calibration.ladder_entries(stage="L1", task="dpo", preservation="fkl",
                                         lambdas=calibration.FKL_LAMBDAS, betas=(0.1,))
    table = {entry.entry_id: _feasible(0.87) for entry in entries}
    outcome = calibration.run_calibration(_runner(table), parent_macro_ap=0.85, simpo=True)
    simpo = [entry for entry in outcome["ledger"]["decisions"] if entry["decision"] == "simpo"]
    assert simpo and simpo[0]["allowance"]["objective_pilots"] == 2
    assert "after the core and audit reserves are measured" in simpo[0]["status"]
