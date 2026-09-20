"""The parent-relative gate: its units, its strictness, its cadence and its identity.

The scoring function is injected in most of these so the arithmetic can be
asserted exactly; one test runs the real scorer over a toy policy end to end to
show that a parent checked against itself gives D = 0 rather than a small number
that happens to pass.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_data as data
from smallAntibodyGen.experiments import her2_guard as guard
from smallAntibodyGen.experiments.her2_runtime import GpuBudgetClock

PAIRS = 6


def cores(seed, count):
    rng = np.random.default_rng(seed)
    letters = np.array(list(data.CANONICAL))
    seen, rows = set(), []
    while len(rows) < count:
        core = "".join(rng.choice(letters, size=data.CORE_LENGTH))
        if core not in seen:
            seen.add(core)
            rows.append(core)
    return rows


class ToyPolicy:
    """A differentiable stand-in with CorePolicy's log-probability contract."""

    def __init__(self, scale=0.0, seed=0):
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(data.CORE_LENGTH, 20, generator=generator)
        self.weight = torch.nn.Parameter(base * float(scale))
        self.model = torch.nn.Module()
        self.model.weight = self.weight
        self.device = torch.device("cpu")

    def position_log_probs(self, index):
        values = torch.as_tensor(np.asarray(index), dtype=torch.long)
        logits = self.weight.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.log_softmax(logits, dim=-1).gather(2, values.unsqueeze(-1)).squeeze(-1)

    def sequence_log_probs(self, index):
        return self.position_log_probs(index).sum(dim=1)


@pytest.fixture
def pairs():
    return {"chosen_index": data.encode_cores(cores(1, PAIRS)),
            "rejected_index": data.encode_cores(cores(2, PAIRS)),
            "pairs": PAIRS}


@pytest.fixture
def identity(pairs):
    return guard.parent_reference_identity(
        parent_checkpoint_sha256="a" * 64, parent_state_sha256="b" * 64,
        config_sha256="c" * 64, scaffold_prefix="1AAA",
        chosen_index=pairs["chosen_index"], rejected_index=pairs["rejected_index"])


def reference_of(identity, chosen, rejected):
    """A reference built straight from numbers, for the arithmetic tests."""
    chosen = np.array(chosen, dtype=np.float64)
    rejected = np.array(rejected, dtype=np.float64)
    chosen.setflags(write=False)
    rejected.setflags(write=False)
    return guard.ParentValidationReference(identity=identity, chosen=chosen, rejected=rejected,
                                           gpu_seconds=0.0, wall_seconds=0.0)


def injected_scorer():
    """A scorer that returns declared vectors, keyed by the exact rows it was asked for."""
    lookup = {}

    def score(policy, index, *, batch_size=256, progress_every=0):
        return np.array(lookup[np.asarray(index).tobytes()], dtype=np.float64)
    return score, lookup


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_the_identity_binds_both_core_orders_and_the_population(identity, pairs):
    assert identity["population"] == guard.GATE_POPULATION
    assert identity["pairs"] == PAIRS
    reordered = guard.parent_reference_identity(
        parent_checkpoint_sha256="a" * 64, parent_state_sha256="b" * 64, config_sha256="c" * 64,
        scaffold_prefix="1AAA", chosen_index=pairs["chosen_index"][::-1],
        rejected_index=pairs["rejected_index"])
    assert reordered != identity, "D is a mean of PAIRED differences; row order is identity"


def test_a_shorter_pair_set_is_a_different_identity(pairs):
    full = guard.parent_reference_identity(
        parent_checkpoint_sha256="a" * 64, parent_state_sha256="b" * 64, config_sha256="c" * 64,
        scaffold_prefix="1AAA", chosen_index=pairs["chosen_index"],
        rejected_index=pairs["rejected_index"])
    sliced = guard.parent_reference_identity(
        parent_checkpoint_sha256="a" * 64, parent_state_sha256="b" * 64, config_sha256="c" * 64,
        scaffold_prefix="1AAA", chosen_index=pairs["chosen_index"][:3],
        rejected_index=pairs["rejected_index"][:3])
    assert sliced != full
    assert sliced["pairs"] == 3 and full["pairs"] == PAIRS


def test_the_reference_round_trips_and_refuses_tampering(tmp_path, pairs, identity):
    policy = ToyPolicy()
    reference = guard.build_parent_reference(policy, pairs, identity)
    assert reference.pairs == PAIRS
    path = guard.save_parent_reference(tmp_path / "parent.npz", reference)
    with pytest.raises(ValueError, match="immutable"):
        guard.save_parent_reference(path, reference)
    reloaded = guard.load_parent_reference(path, identity)
    assert np.array_equal(reloaded.chosen, reference.chosen)
    assert reloaded.reused is True
    with pytest.raises(ValueError, match="identity mismatch"):
        guard.load_parent_reference(path, dict(identity, parent_state_sha256="9" * 64))
    np.savez(path, chosen=np.zeros(PAIRS), rejected=np.zeros(PAIRS))
    with pytest.raises(ValueError, match="do not match the digests"):
        guard.load_parent_reference(path, identity)


def test_a_training_population_reference_cannot_stand_in_for_the_validation_one(tmp_path, pairs,
                                                                                identity):
    """Same parent weights, different rows. The gate must refuse the substitution."""
    policy = ToyPolicy()
    # It cannot even be built through this module: the gate scores one population,
    # and the rows are verified against the identity before they are scored.
    training_identity = dict(identity, population="training_pairs")
    with pytest.raises(ValueError, match="parent reference construction"):
        guard.build_parent_reference(policy, pairs, training_identity)
    # And a stored reference that claims the training population is refused on load.
    saved = guard.build_parent_reference(policy, pairs, identity)
    path = guard.save_parent_reference(tmp_path / "parent.npz", saved)
    sidecar = path.with_suffix(".json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["identity"]["population"] = "training_pairs"
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="population") as error:
        guard.load_parent_reference(path, identity)
    assert "identity mismatch" in str(error.value)


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------

def test_d_is_nats_per_sequence_and_the_per_residue_value_is_a_tenth(identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    gate = guard.LikelihoodGate(threshold_nats_per_sequence=1.0)
    record = gate.evaluate(reference, np.full(PAIRS, -32.0), np.full(PAIRS, -35.0))
    assert record["D"] == pytest.approx(2.0)
    assert record["D_per_residue"] == pytest.approx(0.2)
    assert gate.threshold_nats_per_residue == pytest.approx(0.1)
    assert record["passed"] is False


def test_a_policy_better_than_its_parent_has_a_negative_d(identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    record = guard.LikelihoodGate().evaluate(reference, np.full(PAIRS, -29.0),
                                             np.full(PAIRS, -36.0))
    assert record["D"] == pytest.approx(-1.0)
    assert record["passed"] is True


def test_exact_equality_passes_and_a_hair_over_stops(identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    gate = guard.LikelihoodGate(threshold_nats_per_sequence=1.0)
    assert gate.evaluate(reference, np.full(PAIRS, -31.0), np.full(PAIRS, -35.0))["passed"] is True
    over = gate.evaluate(reference, np.full(PAIRS, -31.0 - 1e-12), np.full(PAIRS, -35.0))
    assert over["passed"] is False
    assert over["stop_reason"] == "parent_relative_likelihood_breach"


def test_a_nonfinite_score_stops_regardless_of_d_and_stays_json_safe(identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    current = np.full(PAIRS, -30.0)
    current[2] = np.nan
    record = guard.LikelihoodGate().evaluate(reference, current, np.full(PAIRS, -35.0))
    assert record["passed"] is False
    assert record["stop_reason"] == "nonfinite_validation_score"
    assert record["D"] is None and record["D_per_residue"] is None
    assert record["nonfinite_scores"] == 1
    # allow_nan=False is what every artifact is written with; a NaN would fail there.
    assert json.loads(json.dumps(record, allow_nan=False))["D"] is None


def test_the_gate_refuses_a_partially_scored_pair_set(identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    with pytest.raises(ValueError, match="the pair set is fixed"):
        guard.LikelihoodGate().evaluate(reference, np.full(3, -30.0), np.full(3, -35.0))


def test_the_record_carries_the_whole_observed_distribution(identity):
    reference = reference_of(identity, np.linspace(-30, -35, PAIRS), np.full(PAIRS, -40.0))
    record = guard.LikelihoodGate().evaluate(reference, np.linspace(-31, -35, PAIRS),
                                             np.full(PAIRS, -40.0))
    assert set(record["chosen_drop"]["quantiles"]) == {str(q) for q in guard.QUANTILES}
    assert record["chosen_drop"]["count"] == PAIRS
    assert record["rejected_drop"]["mean"] == pytest.approx(0.0)
    assert record["pair_accuracy"] == pytest.approx(1.0)


def test_the_record_carries_the_drop_fractions_a_mean_cannot_express(identity):
    """A small mean drop and a small tail collapse are different states.

    Built to the shape the completed campaign's continued-SFT control actually
    finished in: a comfortably passing mean, most of the population barely moved,
    and a thin tail that has lost a great deal.
    """
    parent = np.full(PAIRS, -28.0)
    current = np.full(PAIRS, -27.0)          # five pairs the policy got BETTER at
    current[0] = -34.0                       # one collapsed, and past uniform
    record = guard.LikelihoodGate().evaluate(reference_of(identity, parent,
                                                          np.full(PAIRS, -40.0)),
                                             current, np.full(PAIRS, -40.0))
    # The mean is what the gate stops on, and it passes comfortably here.
    assert record["D"] == pytest.approx(1 / 6) and record["passed"] is True
    fractions = record["chosen_drop_fractions"]
    assert fractions["count"] == PAIRS
    assert fractions["fraction_below_parent"] == pytest.approx(1 / PAIRS)
    assert fractions["fraction_drop_gt1"] == pytest.approx(1 / PAIRS)
    assert fractions["fraction_drop_gt5"] == pytest.approx(1 / PAIRS)
    assert fractions["fraction_below_uniform"] == pytest.approx(1 / PAIRS)
    assert fractions["uniform_sum_log_probability"] == pytest.approx(-29.9573227, abs=1e-6)
    # The fractions are reported beside the verdict and decide nothing.
    assert json.loads(json.dumps(record, allow_nan=False))["chosen_drop_fractions"]


def test_drop_fractions_report_nonfinite_rows_instead_of_propagating_them():
    fractions = guard.drop_fractions([0.5, 2.0, np.nan, 7.0], [-14.0, -14.0, np.nan, -40.0])
    assert fractions["count"] == 4 and fractions["nonfinite"] == 1
    assert fractions["fraction_below_parent"] == pytest.approx(1.0)      # 3 finite, all > 0
    assert fractions["fraction_drop_gt1"] == pytest.approx(2 / 3)
    assert fractions["fraction_drop_gt5"] == pytest.approx(1 / 3)
    assert fractions["fraction_below_uniform"] == pytest.approx(1 / 3)
    assert guard.drop_fractions([np.nan], [np.nan])["fraction_below_parent"] is None


def test_quantile_summary_counts_nonfinite_entries_instead_of_propagating_them():
    summary = guard.quantile_summary([1.0, 2.0, np.nan, 3.0])
    assert summary["nonfinite"] == 1
    assert summary["mean"] == pytest.approx(2.0)
    empty = guard.quantile_summary([np.nan, np.inf * -1])
    assert empty["mean"] is None and empty["quantiles"]["0.5"] is None


# ---------------------------------------------------------------------------
# cadence
# ---------------------------------------------------------------------------

def test_the_first_check_is_after_update_one():
    schedule = guard.MonitorSchedule()
    assert schedule.due(1, 0.2) == (True, "first_update")
    schedule.record(1, 0.2)
    assert schedule.due(2, 0.4)[0] is False


def test_the_update_interval_fires_at_exactly_twenty_five():
    schedule = guard.MonitorSchedule()
    schedule.record(1, 0.0)
    assert schedule.due(25, 0.5)[0] is False
    due, reason = schedule.due(26, 0.5)
    assert due is True and reason == "update_interval"


def test_the_gpu_second_interval_fires_first_when_updates_are_slow():
    schedule = guard.MonitorSchedule()
    schedule.record(1, 0.0)
    assert schedule.due(3, 4.9)[0] is False
    due, reason = schedule.due(3, 5.0)
    assert due is True and reason == "gpu_second_interval"


def test_the_cadence_is_measured_from_the_last_check_not_from_the_start():
    schedule = guard.MonitorSchedule()
    schedule.record(1, 0.0)
    schedule.record(26, 12.0)
    assert schedule.due(50, 16.0)[0] is False, "24 updates and 4 seconds since the last check"
    assert schedule.due(51, 16.0)[0] is True


def test_the_cadence_refuses_a_nonsense_configuration():
    with pytest.raises(ValueError, match="positive integer"):
        guard.MonitorSchedule(update_interval=0)
    with pytest.raises(ValueError, match="must be positive"):
        guard.MonitorSchedule(gpu_second_interval=0)
    with pytest.raises(ValueError, match="finite and positive"):
        guard.LikelihoodGate(threshold_nats_per_sequence=0.0)


# ---------------------------------------------------------------------------
# the monitor: cost accounting and persistence
# ---------------------------------------------------------------------------

def stepping_clock(step_seconds):
    state = {"now": 0.0, "open": False}

    def clock():
        if not state["open"]:
            state["open"] = True
            return state["now"]
        state["open"] = False
        state["now"] += step_seconds
        return state["now"]
    return clock


def test_the_parent_checked_against_itself_measures_exactly_zero(tmp_path, pairs, identity):
    """The real scorer, a real toy policy, and no tolerance hiding a small drift."""
    policy = ToyPolicy(scale=0.3, seed=5)
    reference = guard.build_parent_reference(policy, pairs, identity)
    monitor = guard.GateMonitor(policy, reference, directory=tmp_path, batch_size=4)
    record = monitor.check(pairs, update=1, gpu_seconds=1.0, reason="first_update")
    assert record["D"] == pytest.approx(0.0, abs=1e-12)
    assert record["passed"] is True
    assert record["pairs"] == PAIRS


def test_monitor_cost_is_measured_and_never_touches_the_training_clock(tmp_path, pairs, identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    score, lookup = injected_scorer()
    lookup[np.asarray(pairs["chosen_index"]).tobytes()] = np.full(PAIRS, -30.0)
    lookup[np.asarray(pairs["rejected_index"]).tobytes()] = np.full(PAIRS, -35.0)
    training = GpuBudgetClock(clock=stepping_clock(1.0))
    with training.segment():
        pass
    monitor_clock = GpuBudgetClock(clock=stepping_clock(0.25))
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score,
                                monitor_clock=monitor_clock)
    for update in (1, 26):
        monitor.check(pairs, update=update, gpu_seconds=training.elapsed_seconds,
                      reason="cadence")
    assert training.elapsed_seconds == pytest.approx(1.0), "monitoring is not training"
    assert monitor.gpu_seconds == pytest.approx(0.5)
    assert monitor.wall_seconds > 0.0
    document = monitor.document()
    assert document["checks"] == 2
    assert document["monitor_gpu_seconds"] == pytest.approx(0.5)
    # The schedule advanced with the checks, so the next ordinary check is 25 later.
    assert monitor.schedule.last_update == 26


def test_every_check_persists_its_score_vectors_with_their_identity(tmp_path, pairs, identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    score, lookup = injected_scorer()
    lookup[np.asarray(pairs["chosen_index"]).tobytes()] = np.full(PAIRS, -30.5)
    lookup[np.asarray(pairs["rejected_index"]).tobytes()] = np.full(PAIRS, -35.0)
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score)
    record = monitor.check(pairs, update=1, gpu_seconds=0.0, reason="first_update")
    stored = record["scores"]
    with np.load(stored["path"]) as archive:
        assert archive["chosen"] == pytest.approx(np.full(PAIRS, -30.5))
    assert len(stored["sha256"]) == 64
    assert record["identity"] == reference.identity
    # D is exactly reconstructible from the persisted vector and the parent's.
    with np.load(stored["path"]) as archive:
        assert float((reference.chosen - archive["chosen"]).mean()) == pytest.approx(record["D"])


def test_a_check_on_reordered_rows_is_refused_rather_than_labelled_with_the_old_identity(
        tmp_path, pairs, identity):
    """D is a mean of PAIRED differences. Same multiset, different order, different number."""
    reference = reference_of(identity, np.linspace(-30, -35, PAIRS), np.full(PAIRS, -40.0))
    score, lookup = injected_scorer()
    reordered = {"chosen_index": np.asarray(pairs["chosen_index"])[::-1].copy(),
                 "rejected_index": pairs["rejected_index"], "pairs": PAIRS}
    lookup[np.asarray(reordered["chosen_index"]).tobytes()] = np.linspace(-30, -35, PAIRS)
    lookup[np.asarray(pairs["rejected_index"]).tobytes()] = np.full(PAIRS, -40.0)
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score)
    with pytest.raises(ValueError, match="not the rows this reference was built on") as error:
        monitor.check(reordered, update=1, gpu_seconds=0.0, reason="first_update")
    assert "chosen_core_order_sha256" in str(error.value)
    assert monitor.checks == 0, "a refused check is not a check"


def test_a_check_on_a_sliced_pair_set_is_refused_before_it_is_scored(tmp_path, pairs, identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    score, lookup = injected_scorer()
    sliced = {"chosen_index": np.asarray(pairs["chosen_index"])[:3],
              "rejected_index": np.asarray(pairs["rejected_index"])[:3], "pairs": 3}
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score)
    with pytest.raises(ValueError, match="not the rows this reference was built on") as error:
        monitor.check(sliced, update=1, gpu_seconds=0.0, reason="first_update")
    assert "pairs" in str(error.value)
    assert not lookup, "the scorer was never reached; verification runs first"


def test_the_parent_reference_verifies_its_own_rows_before_scoring_them(pairs, identity):
    """The identity a reference is bound to must describe the rows that produced it."""
    policy = ToyPolicy(scale=0.2, seed=3)
    swapped = {"chosen_index": pairs["rejected_index"], "rejected_index": pairs["chosen_index"],
               "pairs": PAIRS}
    with pytest.raises(ValueError, match="parent reference construction"):
        guard.build_parent_reference(policy, swapped, identity)
    assert guard.build_parent_reference(policy, pairs, identity).pairs == PAIRS


def test_a_declared_pair_count_that_disagrees_with_the_rows_is_refused(pairs, identity):
    lying = dict(pairs, pairs=PAIRS + 1)
    with pytest.raises(ValueError, match="pairs but carries"):
        guard.verify_pairs_against_identity(lying, identity, where="test")


def test_the_verified_rows_are_recorded_on_every_check(tmp_path, pairs, identity):
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    score, lookup = injected_scorer()
    lookup[np.asarray(pairs["chosen_index"]).tobytes()] = np.full(PAIRS, -30.0)
    lookup[np.asarray(pairs["rejected_index"]).tobytes()] = np.full(PAIRS, -35.0)
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score)
    record = monitor.check(pairs, update=1, gpu_seconds=0.0, reason="first_update")
    assert record["scored_rows"]["pairs"] == PAIRS
    assert record["scored_rows"]["chosen_core_order_sha256"] == \
        identity["chosen_core_order_sha256"]
    assert record["scored_rows"]["population"] == guard.GATE_POPULATION


def test_the_measured_check_wall_time_includes_the_persistence_and_the_hashing(tmp_path, pairs,
                                                                                identity):
    """A wall time sampled before the file is written measures the wrong thing."""
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    score, lookup = injected_scorer()
    lookup[np.asarray(pairs["chosen_index"]).tobytes()] = np.full(PAIRS, -30.0)
    lookup[np.asarray(pairs["rejected_index"]).tobytes()] = np.full(PAIRS, -35.0)
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score)
    record = monitor.check(pairs, update=1, gpu_seconds=0.0, reason="first_update")
    from pathlib import Path
    assert Path(record["scores"]["path"]).is_file()
    assert record["monitor_wall_seconds"] > 0.0
    assert "hashing them" in record["monitor_wall_note"]
    assert monitor.wall_seconds == pytest.approx(record["monitor_wall_seconds"])


def test_the_gate_document_states_the_population_and_both_threshold_units():
    document = guard.LikelihoodGate().document()
    assert document["threshold_nats_per_sequence"] == 1.0
    assert document["threshold_nats_per_residue"] == pytest.approx(0.1)
    assert "every arm including continued_sft" in document["rule"]
    assert document["population"] == guard.GATE_POPULATION


# ---------------------------------------------------------------------------
# re-deriving a journalled verdict from the vectors it was measured on
# ---------------------------------------------------------------------------

def journalled_check(tmp_path, pairs, identity, *, chosen, rejected):
    """One real check, its retained vectors on disk, and the record it produced."""
    reference = reference_of(identity, np.full(PAIRS, -30.0), np.full(PAIRS, -35.0))
    score, lookup = injected_scorer()
    lookup[np.asarray(pairs["chosen_index"]).tobytes()] = chosen
    lookup[np.asarray(pairs["rejected_index"]).tobytes()] = rejected
    monitor = guard.GateMonitor(None, reference, directory=tmp_path, score_sequences_fn=score)
    record = monitor.check(pairs, update=1, gpu_seconds=0.0, reason="first_update")
    return reference, json.loads(json.dumps(record))


def test_a_verdict_is_reproduced_from_its_retained_vectors(tmp_path, pairs, identity):
    reference, record = journalled_check(tmp_path, pairs, identity,
                                          chosen=np.full(PAIRS, -30.4),
                                          rejected=np.full(PAIRS, -35.0))
    recomputed = guard.recompute_gate_verdict(tmp_path, record, reference,
                                              gate=guard.LikelihoodGate())
    assert recomputed["D"] == pytest.approx(0.4)
    assert recomputed["passed"] is True and recomputed["pairs"] == PAIRS
    assert recomputed["scores"]["sha256"] == record["scores"]["sha256"]


def test_a_verdict_recorded_on_windows_still_resolves_its_vectors_on_posix(tmp_path, pairs,
                                                                           identity):
    """This campaign fits on Windows and verifies on macOS, over a copied run directory.

    A check records the absolute path it wrote to, so a Windows-written verdict names
    its vectors with backslashes. A POSIX ``Path`` treats those as ordinary filename
    characters, takes a basename of the whole string, finds nothing -- and the refusal
    that follows says the verdict is not evidence about anything, of a file that is
    sitting in the directory.
    """
    from pathlib import Path
    reference, record = journalled_check(tmp_path, pairs, identity,
                                          chosen=np.full(PAIRS, -30.4),
                                          rejected=np.full(PAIRS, -35.0))
    written = Path(record["scores"]["path"]).name
    windows = dict(record["scores"],
                   path="C:\\Users\\big DAWG\\personal_projs\\run\\monitor_scores\\"
                        + written)
    recomputed = guard.recompute_gate_verdict(tmp_path, dict(record, scores=windows), reference,
                                              gate=guard.LikelihoodGate())
    assert recomputed["D"] == pytest.approx(0.4)
    assert Path(recomputed["scores"]["path"]).name == written


def test_a_verdict_with_no_arrays_a_moved_array_or_an_invented_d_is_refused(tmp_path, pairs,
                                                                             identity):
    """The three shapes a journal-reading check accepts and a recomputation does not."""
    reference, record = journalled_check(tmp_path, pairs, identity,
                                          chosen=np.full(PAIRS, -30.4),
                                          rejected=np.full(PAIRS, -35.0))
    gate = guard.LikelihoodGate()
    fabricated = dict(record, D=999.0, passed=True)
    fabricated.pop("scores")
    with pytest.raises(ValueError, match="names no retained score vectors"):
        guard.recompute_gate_verdict(tmp_path, fabricated, reference, gate=gate)
    from pathlib import Path
    stored = Path(record["scores"]["path"])
    moved = stored.rename(stored.with_name("elsewhere.npz"))
    with pytest.raises(ValueError, match="not beside the run"):
        guard.recompute_gate_verdict(tmp_path, record, reference, gate=gate)
    moved.rename(stored)
    with pytest.raises(ValueError, match="is not the measurement"):
        guard.recompute_gate_verdict(tmp_path, dict(record, D=0.0), reference, gate=gate)


def test_a_verdict_measured_under_another_threshold_or_on_other_rows_is_refused(tmp_path, pairs,
                                                                                 identity):
    reference, record = journalled_check(tmp_path, pairs, identity,
                                          chosen=np.full(PAIRS, -30.4),
                                          rejected=np.full(PAIRS, -35.0))
    with pytest.raises(ValueError, match="another threshold"):
        guard.recompute_gate_verdict(tmp_path, record, reference,
                                     gate=guard.LikelihoodGate(threshold_nats_per_sequence=2.0))
    foreign = dict(record, identity={"population": "something_else", "pairs": PAIRS})
    with pytest.raises(ValueError, match="different parent reference"):
        guard.recompute_gate_verdict(tmp_path, foreign, reference, gate=guard.LikelihoodGate())


def test_a_pass_declared_over_breaching_vectors_is_refused(tmp_path, pairs, identity):
    """D = 5 nats/sequence with `passed: true` beside it is exactly what this catches."""
    reference, record = journalled_check(tmp_path, pairs, identity,
                                          chosen=np.full(PAIRS, -35.0),
                                          rejected=np.full(PAIRS, -35.0))
    assert record["passed"] is False and record["D"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="the declared gate applied to these vectors"):
        guard.recompute_gate_verdict(tmp_path, dict(record, passed=True), reference,
                                     gate=guard.LikelihoodGate())
