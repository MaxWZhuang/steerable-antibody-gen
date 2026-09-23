"""The two-tier monitor: who may ask, who may stop, and which rows are gated.

The Block-B test is the load-bearing one. A gate built from C-internal
preference pairs silently drops every high row whose WT distance has no eligible
low partner in C -- a smaller population than the specification names, reached
by a change that looks like an implementation detail.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_nf_monitor as monitor
from smallAntibodyGen.tests.her2_nf_support import TinyPolicy, tiny_cores


def test_full_interval_must_be_a_multiple_of_the_sentinel_interval():
    with pytest.raises(ValueError, match="multiple"):
        monitor.MonitorPlan(sentinel_interval=25, full_interval=30)


def test_cadence_gives_one_reason_per_update_and_endpoints_force_a_full_check():
    plan = monitor.MonitorPlan(endpoints=(1000, 2000), checkpoints=(250, 500))
    assert plan.due(1) == ("full", "first_update")
    assert plan.due(25) == ("sentinel", "sentinel_interval")
    assert plan.due(100) == ("full", "full_interval")
    assert plan.due(250) == ("full", "checkpoint")
    assert plan.due(1000) == ("full", "exposure_endpoint")
    assert plan.due(37) == (None, None)
    assert plan.due(37, final=True) == ("full", "final_state")
    assert plan.due(50, requested=True) == ("full", "sentinel_request")


def test_the_scheduled_cadence_costs_far_fewer_full_checks_than_every_25(tmp_path):
    plan = monitor.MonitorPlan(endpoints=(1000, 2000, 3750), checkpoints=(250, 500))
    scheduled = plan.scheduled(3750)
    full = [entry for entry in scheduled if entry[1] == "full"]
    sentinel = [entry for entry in scheduled if entry[1] == "sentinel"]
    assert len(full) < 3750 // 25
    assert len(sentinel) > len(full)
    assert "sensitivity BETWEEN full checks is changed" in plan.document()["disclosure"]


def _sentinel(policy, *, parent_chosen=None, parent_scores=None):
    chosen = tiny_cores(32, seed=1)
    bank = tiny_cores(64, seed=2)
    return monitor.SentinelGate(
        chosen_index=chosen,
        parent_chosen=(parent_chosen if parent_chosen is not None
                       else policy.score(chosen)["sum_log_probability"]),
        parent_index=bank,
        parent_log_probability=(parent_scores if parent_scores is not None
                                else policy.score(bank)["sum_log_probability"]))


def test_a_quiet_sentinel_asks_for_nothing():
    policy = TinyPolicy(seed=3)
    record = _sentinel(policy).evaluate(policy, update=25, reason="sentinel_interval")
    assert record["request_full_check"] is False
    assert record["trigger_reasons"] == []
    assert record["authority"].startswith("request only")


def test_each_declared_trigger_fires_on_its_own():
    policy = TinyPolicy(seed=4)
    chosen = tiny_cores(32, seed=1)
    bank = tiny_cores(64, seed=2)
    base_chosen = policy.score(chosen)["sum_log_probability"]
    base_bank = policy.score(bank)["sum_log_probability"]

    drifted = monitor.SentinelGate(chosen_index=chosen, parent_chosen=base_chosen + 0.9,
                                   parent_index=bank, parent_log_probability=base_bank)
    assert "chosen_side_D" in drifted.evaluate(policy, update=25, reason="x")["trigger_reasons"]

    tails = monitor.SentinelGate(chosen_index=chosen, parent_chosen=base_chosen,
                                 parent_index=bank,
                                 parent_log_probability=base_bank + math.log(10.0))
    reasons = tails.evaluate(policy, update=25, reason="x")["trigger_reasons"]
    assert "inclusive_tenfold_rate" in reasons

    rising = monitor.SentinelGate(chosen_index=chosen, parent_chosen=base_chosen,
                                  parent_index=bank, parent_log_probability=base_bank)
    rising.evaluate(policy, update=25, reason="x")
    rising.parent_chosen = base_chosen + 0.3
    assert "increase_since_previous" in rising.evaluate(policy, update=50,
                                                        reason="x")["trigger_reasons"]


def test_a_nonfinite_reference_is_refused_at_construction():
    """A nan D compares false against every trigger, so the probe goes blind.

    The earlier behaviour was worse than blind: with an infinite reference the
    mean D was ``inf``, which fired ``chosen_side_D`` and looked like a
    detection while being an arithmetic artefact of the reference.
    """
    policy = TinyPolicy(seed=5)
    with pytest.raises(ValueError, match="nonfinite"):
        _sentinel(policy, parent_chosen=np.full(32, np.inf))
    with pytest.raises(ValueError, match="nonfinite"):
        _sentinel(policy, parent_scores=np.full(64, np.nan))


def test_a_nonfinite_policy_score_asks_immediately_and_still_cannot_stop():
    policy = TinyPolicy(seed=5)
    gate = _sentinel(policy)

    class Blind:
        """A policy whose scores have left the domain where D means anything."""

        model = policy.model

        def __getattr__(self, name):
            return getattr(policy, name)

        def sequence_log_probs(self, index, **kwargs):
            import torch
            return torch.full((np.asarray(index).shape[0],), -np.inf, dtype=torch.float64)

    record = gate.evaluate(Blind(), update=25, reason="x")
    assert record["request_full_check"] is True
    assert record["trigger_reasons"] == ["nonfinite_sentinel_score"]
    assert record["nonfinite_halves"]
    assert "never stops" in record["authority"]
    assert "passed" not in record      # a sentinel has no verdict to give


def test_the_sentinel_scores_both_pair_halves_when_it_is_given_them():
    """512 PAIRS, not 512 chosen rows: the rejected half is half the population."""
    policy = TinyPolicy(seed=51)
    chosen = tiny_cores(16, seed=11)
    rejected = tiny_cores(16, seed=12)
    bank = tiny_cores(32, seed=13)
    gate = monitor.SentinelGate(
        chosen_index=chosen, parent_chosen=policy.score(chosen)["sum_log_probability"],
        rejected_index=rejected,
        parent_rejected=policy.score(rejected)["sum_log_probability"],
        parent_index=bank, parent_log_probability=policy.score(bank)["sum_log_probability"])
    record = gate.evaluate(policy, update=25, reason="x")
    assert record["rejected_rows"] == 16
    assert record["rejected_side_D"] == pytest.approx(0.0, abs=1e-12)
    assert record["implicit_margin_mean"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= record["pair_accuracy"] <= 1.0
    assert gate.document()["rejected_rows"] == 16


def test_the_previous_sentinel_value_survives_a_resume():
    """The 'increase since the previous sentinel' trigger needs its predecessor."""
    policy = TinyPolicy(seed=52)
    gate = _sentinel(policy)
    gate.evaluate(policy, update=25, reason="x")
    remembered = gate.previous_D
    fresh = _sentinel(policy)
    assert fresh.previous_D is None            # a resumed gate starts blank
    fresh.restore_previous_D(remembered)
    assert fresh.previous_D == pytest.approx(remembered)
    fresh.parent_chosen = fresh.parent_chosen + 0.3
    assert "increase_since_previous" in fresh.evaluate(policy, update=50,
                                                       reason="x")["trigger_reasons"]


def test_tail_triggers_use_the_inclusive_convention():
    policy = TinyPolicy(seed=6)
    chosen = tiny_cores(8, seed=1)
    bank = tiny_cores(100, seed=2)
    base = policy.score(bank)["sum_log_probability"]
    offsets = np.zeros(100)
    offsets[:3] = math.log(100.0)               # exactly at the hundredfold threshold
    gate = monitor.SentinelGate(chosen_index=chosen,
                                parent_chosen=policy.score(chosen)["sum_log_probability"],
                                parent_index=bank, parent_log_probability=base + offsets)
    record = gate.evaluate(policy, update=25, reason="x")
    assert record["hundredfold_events"] == 3
    assert "inclusive_hundredfold_events" in record["trigger_reasons"]


def test_the_block_b_gate_includes_high_rows_with_no_pairable_partner():
    """The population is ALL high C rows; a pair-based gate would lose some of them."""
    policy = TinyPolicy(seed=7)
    high = tiny_cores(50, seed=8)
    parent_scores = policy.score(high)["sum_log_probability"]
    gate = monitor.HighRowGate(row_index=high, parent_log_probability=parent_scores)
    record = gate.evaluate(policy, update=25, reason="x")
    assert record["rows"] == 50
    assert record["population"] == "all high C rows"
    assert record["passed"] is True
    assert "no eligible low partner" in record["pairability_note"]


def test_the_block_b_gate_keeps_the_strict_threshold_and_equality_passes():
    policy = TinyPolicy(seed=9)
    high = tiny_cores(20, seed=10)
    base = policy.score(high)["sum_log_probability"]
    exact = monitor.HighRowGate(row_index=high, parent_log_probability=base + 1.0)
    record = exact.evaluate(policy, update=1, reason="x")
    assert record["D"] == pytest.approx(1.0)
    assert record["passed"] is True, "exact equality passes; the rule is STRICT D > threshold"
    breach = monitor.HighRowGate(row_index=high, parent_log_probability=base + 1.0001)
    assert breach.evaluate(policy, update=1, reason="x")["passed"] is False


def test_a_nonfinite_parent_reference_is_refused_rather_than_producing_a_nan_D():
    """A nan D is not a small D, and it would compare False against every threshold."""
    high = tiny_cores(10, seed=12)
    with pytest.raises(ValueError, match="nonfinite score on the gated population"):
        monitor.HighRowGate(row_index=high, parent_log_probability=np.full(10, np.nan))


def test_a_nonfinite_policy_score_stops_the_block_b_gate_with_its_own_reason(monkeypatch):
    policy = TinyPolicy(seed=11)
    high = tiny_cores(10, seed=12)
    gate = monitor.HighRowGate(row_index=high,
                               parent_log_probability=policy.score(high)["sum_log_probability"])
    monkeypatch.setattr(monitor, "score_sequences",
                        lambda *args, **kwargs: np.full(10, -np.inf))
    record = gate.evaluate(policy, update=1, reason="x")
    assert record["passed"] is False
    assert record["stop_reason"] == "nonfinite_validation_score"
    assert record["D"] is None, "a nonfinite score is reported as null, not as a zero"


def test_the_gate_refuses_an_empty_population():
    with pytest.raises(ValueError, match="at least one high row"):
        monitor.HighRowGate(row_index=np.zeros((0, 10), dtype=np.int8),
                            parent_log_probability=np.zeros(0))


def test_pair_diagnostics_are_labelled_as_a_subset_and_cannot_change_the_gate():
    policy = TinyPolicy(seed=13)
    pairs = {"chosen_index": tiny_cores(12, seed=14), "rejected_index": tiny_cores(12, seed=15)}
    parent_chosen = policy.score(pairs["chosen_index"])["sum_log_probability"]
    parent_rejected = policy.score(pairs["rejected_index"])["sum_log_probability"]
    block = monitor.pair_diagnostics(policy, pairs, parent_chosen=parent_chosen,
                                     parent_rejected=parent_rejected)
    assert block["pairs"] == 12
    assert "not affected by this subset" in block["status"]


def test_monitor_population_report_quantifies_the_precision_change():
    block = monitor.monitor_population_report(high_rows=958, pairable_rows=900,
                                              excluded_rows=58, block="B")
    assert block["gate_population"] == "all high rows"
    assert block["standard_error_scale"] == pytest.approx(math.sqrt(25722 / 958))
    assert "no equivalence to the old precision is claimed" in block["precision_change"]


def test_sentinel_sensitivity_is_a_diagnostic_not_a_guarantee():
    drop = np.random.default_rng(1).normal(loc=1.0, scale=1.0, size=5000)
    block = monitor.sentinel_sensitivity(drop, pairs=512, trigger=0.5, true_value=1.0,
                                         draws=200, seed=2)
    assert 0.0 <= block["fire_rate"] <= 1.0
    assert "not a detection guarantee" in block["status"]


def test_preservation_block_re_raises_a_non_observation_failure():
    policy = TinyPolicy(seed=16)
    with pytest.raises(Exception):
        monitor.preservation_block(policy, np.zeros((4, 3), dtype=np.int8),
                                   parent_log_probability=np.zeros(4))
