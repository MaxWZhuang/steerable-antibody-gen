"""The estimand, its uncertainty, its strata and the declared decision rule.

The arithmetic is checked against enumeration wherever enumeration is possible: a
tiny autoregressive model over a two-symbol alphabet has a closed-form forward KL,
a chain-rule decomposition and an exactly computable sampled-estimator
expectation, so the Monte Carlo estimator this audit uses can be checked against
the number it is supposed to estimate rather than against itself.
"""
from __future__ import annotations

import itertools
import math
import subprocess
import sys

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_support_scoring as scoring


# ---------------------------------------------------------------------------
# a tiny enumerable autoregressive model
# ---------------------------------------------------------------------------

SYMBOLS = (0, 1)
LENGTH = 3


def conditionals(bias):
    """``{prefix: (p0, p1)}`` for every prefix, as a simple deterministic family."""
    table = {}
    for length in range(LENGTH):
        for prefix in itertools.product(SYMBOLS, repeat=length):
            weight = 0.5 + bias * (0.1 * (length + 1) + 0.05 * sum(prefix))
            weight = min(max(weight, 0.01), 0.99)
            table[prefix] = (1.0 - weight, weight)
    return table


def sequence_probabilities(table):
    out = {}
    for sequence in itertools.product(SYMBOLS, repeat=LENGTH):
        probability = 1.0
        for position in range(LENGTH):
            probability *= table[sequence[:position]][sequence[position]]
        out[sequence] = probability
    return out


def enumerated_forward_kl(parent_table, policy_table):
    parent = sequence_probabilities(parent_table)
    policy = sequence_probabilities(policy_table)
    return sum(parent[s] * (math.log(parent[s]) - math.log(policy[s])) for s in parent)


def chain_rule_forward_kl(parent_table, policy_table):
    """The same KL as a sum of expected per-position conditional KLs."""
    total = 0.0
    for length in range(LENGTH):
        for prefix in itertools.product(SYMBOLS, repeat=length):
            weight = 1.0
            for position in range(length):
                weight *= parent_table[prefix[:position]][prefix[position]]
            local = sum(parent_table[prefix][s]
                        * (math.log(parent_table[prefix][s]) - math.log(policy_table[prefix][s]))
                        for s in SYMBOLS)
            total += weight * local
    return total


def test_sequence_kl_equals_its_chain_rule_decomposition():
    parent, policy = conditionals(0.0), conditionals(1.0)
    assert enumerated_forward_kl(parent, policy) == pytest.approx(
        chain_rule_forward_kl(parent, policy), rel=1e-12)


def test_the_sampled_estimator_has_the_enumerated_kl_as_its_expectation():
    """mean(parent_lp - policy_lp) over parent draws is exactly the forward KL."""
    parent, policy = conditionals(0.0), conditionals(1.0)
    parent_p = sequence_probabilities(parent)
    policy_p = sequence_probabilities(policy)
    sequences = sorted(parent_p)
    weights = np.asarray([parent_p[s] for s in sequences])
    drop = np.asarray([math.log(parent_p[s]) - math.log(policy_p[s]) for s in sequences])
    assert float((weights * drop).sum()) == pytest.approx(
        enumerated_forward_kl(parent, policy), rel=1e-12)
    # and a large exact-weight draw converges to it
    counts = np.round(weights * 10 ** 6).astype(int)
    sample = np.repeat(drop, counts)
    assert float(sample.mean()) == pytest.approx(enumerated_forward_kl(parent, policy), abs=1e-4)


def test_the_two_tail_distributions_are_not_interchangeable():
    """Drops measured on parent draws and on policy draws are different quantities."""
    parent, policy = conditionals(0.0), conditionals(1.5)
    parent_p = sequence_probabilities(parent)
    policy_p = sequence_probabilities(policy)
    sequences = sorted(parent_p)
    drop = np.asarray([math.log(parent_p[s]) - math.log(policy_p[s]) for s in sequences])
    under_parent = float((np.asarray([parent_p[s] for s in sequences]) * (drop > 0.2)).sum())
    under_policy = float((np.asarray([policy_p[s] for s in sequences]) * (drop > 0.2)).sum())
    assert under_parent != pytest.approx(under_policy, rel=1e-3)


def realized_log_ratio(parent_table, policy_table, sequence):
    """``D(y)``: the realized parent-minus-policy log ratio of one drawn sequence."""
    total = 0.0
    for position in range(LENGTH):
        prefix = sequence[:position]
        symbol = sequence[position]
        total += (math.log(parent_table[prefix][symbol])
                  - math.log(policy_table[prefix][symbol]))
    return total


def conditional_kl_sum(parent_table, policy_table, sequence):
    """``K(y)``: the sum of conditional categorical KLs at this sequence's prefixes."""
    total = 0.0
    for position in range(LENGTH):
        prefix = sequence[:position]
        total += sum(parent_table[prefix][s]
                     * (math.log(parent_table[prefix][s]) - math.log(policy_table[prefix][s]))
                     for s in SYMBOLS)
    return total


def test_the_realized_log_ratio_and_the_conditional_kl_sum_share_a_mean_not_a_tail():
    """D(y) and K(y) are different random variables under the SAME parent law.

    Both are functions of a sequence drawn from the parent, and both average to the
    sequence-level forward KL -- the first by definition, the second by the chain
    rule. They are still not interchangeable: D is the quantity whose upper tail
    this audit reports as abandoned support, and K, being an average over symbols
    the draw did not take, has a visibly narrower spread. Publishing one and
    describing the other would understate exactly the rare events the audit exists
    to count, and no tolerance or sample size would reveal it.
    """
    parent, policy = conditionals(0.0), conditionals(1.5)
    parent_p = sequence_probabilities(parent)
    sequences = sorted(parent_p)
    weights = np.asarray([parent_p[s] for s in sequences])
    d_values = np.asarray([realized_log_ratio(parent, policy, s) for s in sequences])
    k_values = np.asarray([conditional_kl_sum(parent, policy, s) for s in sequences])
    sequence_kl = enumerated_forward_kl(parent, policy)

    # same parent law, same expectation, and both equal the sequence KL
    assert float((weights * d_values).sum()) == pytest.approx(sequence_kl, rel=1e-12)
    assert float((weights * k_values).sum()) == pytest.approx(sequence_kl, rel=1e-12)

    # ...and different distributions: not equal row by row, and a wider spread
    assert not np.allclose(d_values, k_values)
    d_spread = float((weights * (d_values - sequence_kl) ** 2).sum())
    k_spread = float((weights * (k_values - sequence_kl) ** 2).sum())
    assert d_spread > k_spread

    # the tail the audit reports differs between them at a threshold either could see
    threshold = float(sequence_kl + 0.5 * math.sqrt(d_spread))
    tail_d = float((weights * (d_values > threshold)).sum())
    tail_k = float((weights * (k_values > threshold)).sum())
    assert tail_d != pytest.approx(tail_k, abs=1e-9)


# ---------------------------------------------------------------------------
# the signed drop and its edge cases
# ---------------------------------------------------------------------------

def test_a_parent_against_itself_is_exactly_zero():
    values = np.asarray([-3.0, -4.5, -2.25])
    drop = scoring.signed_drop(values, values, label="self")
    assert np.array_equal(drop, np.zeros(3))
    control = scoring.self_control(values, values, atol=5e-5, rtol=2e-6)
    assert control["within_tolerance"] and control["max_abs_drop"] == 0.0


def test_a_suppressed_mode_is_a_large_positive_drop_and_a_concentrated_one_is_negative():
    parent = np.asarray([-1.0, -1.0, -1.0])
    suppressed = np.asarray([-11.0, -1.0, -1.0])
    concentrated = np.asarray([-0.1, -0.1, -0.1])
    assert scoring.signed_drop(parent, suppressed, label="s")[0] == pytest.approx(10.0)
    assert (scoring.signed_drop(parent, concentrated, label="c") < 0).all()


def test_a_negative_finite_bank_estimate_is_preserved_not_clipped():
    drop = np.asarray([-0.4, -0.2, -0.3, 0.1])
    matrix = scoring.bootstrap_index_matrix(4, draws=200, seed=3)
    block = scoring.drop_statistics(drop, matrix)
    assert block["forward_kl"]["mean"] < 0
    assert block["negative_rows"] == 3
    assert block["tails"]["counts"]["gt1"]["proportion"] == 0.0


def test_duplicated_bank_rows_are_retained_and_weight_the_estimate():
    once = np.asarray([0.0, 4.0])
    twice = np.asarray([0.0, 4.0, 4.0])
    matrix_a = scoring.bootstrap_index_matrix(2, draws=50, seed=1)
    matrix_b = scoring.bootstrap_index_matrix(3, draws=50, seed=1)
    assert scoring.drop_statistics(once, matrix_a)["forward_kl"]["mean"] == pytest.approx(2.0)
    assert scoring.drop_statistics(twice, matrix_b)["forward_kl"]["mean"] == pytest.approx(8 / 3)


@pytest.mark.parametrize("bad", [np.asarray([0.0, np.nan]), np.asarray([np.inf, 1.0]),
                                 np.asarray([-np.inf, 1.0])])
def test_a_nonfinite_score_fails_the_artifact_rather_than_dropping_a_row(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        scoring.require_finite(bad, label="scores")
    with pytest.raises(ValueError, match="nonfinite"):
        scoring.signed_drop(bad, np.zeros(2), label="drop")


def test_empty_and_singleton_inputs_are_refused_or_reported_not_averaged():
    with pytest.raises(ValueError):
        scoring.signed_drop(np.asarray([]), np.asarray([]), label="empty")
    block = scoring.log_probability_block(np.asarray([-2.0]), label="one row")
    assert block["standard_error"] is None and block["rows"] == 1


# ---------------------------------------------------------------------------
# thresholds, Wilson intervals, bootstrap pairing
# ---------------------------------------------------------------------------

def test_exact_threshold_equality_is_not_counted_as_a_crossing():
    drop = np.asarray([scoring.LN10, scoring.LN10 + 1e-12, 0.0])
    block = scoring.tail_block(drop)
    assert block["counts"]["tenfold"]["successes"] == 1
    assert "strictly greater" in block["comparison"]


def test_wilson_matches_the_hand_computed_interval_without_continuity_correction():
    interval = scoring.wilson_interval(2, 100)
    z = scoring.Z_95
    p, n = 0.02, 100
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    assert interval["lower"] == pytest.approx(centre - spread)
    assert interval["upper"] == pytest.approx(centre + spread)
    empty = scoring.wilson_interval(0, 0)
    assert empty["lower"] is None and "no rows" in empty["reason"]


def test_bootstrap_indices_are_shared_so_a_within_parent_difference_is_paired():
    matrix = scoring.bootstrap_index_matrix(500, draws=300, seed=11)
    a = np.random.default_rng(0).normal(size=500)
    b = a + 0.25
    paired = scoring.paired_difference(a, b, matrix)
    assert paired["mean"] == pytest.approx(-0.25)
    # Every replicate of a paired difference of a constant offset is the offset.
    assert paired["ci_low"] == pytest.approx(-0.25) and paired["ci_high"] == pytest.approx(-0.25)
    assert "same bank rows" in paired["pairing"]


def test_the_same_seed_reproduces_the_same_index_matrix():
    first = scoring.bootstrap_index_matrix(50, draws=10, seed=7)
    second = scoring.bootstrap_index_matrix(50, draws=10, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, scoring.bootstrap_index_matrix(50, draws=10, seed=8))


# ---------------------------------------------------------------------------
# strata keep ties, empty bins and real counts
# ---------------------------------------------------------------------------

def test_quartiles_keep_ties_together_so_bins_are_unequal():
    """The requirement is that ties are not split; which side of an edge is declared.

    With ``BIN_EDGE_SIDE == "right"`` a row equal to an edge falls in the bin above
    it, so four rows tied at the minimum land in one bin together and the bin below
    is empty. The empty bin is a property of the data, not a defect: asserting a
    balanced split here would be asserting that ties get broken.
    """
    values = np.asarray([1.0, 1.0, 1.0, 1.0, 5.0, 9.0, 9.0, 9.0])
    labels, meta = scoring.quartile_labels(values)
    assert meta["edge_side"] == scoring.BIN_EDGE_SIDE == "right"

    # no tied value is ever split across two bins
    for value in np.unique(values):
        assert len(set(labels[values == value])) == 1

    sizes = {name: int((labels == name).sum()) for name in ("q1", "q2", "q3", "q4")}
    assert sizes == {"q1": 0, "q2": 4, "q3": 1, "q4": 3}
    assert sum(sizes.values()) == values.size
    assert meta["edges"] == [1.0, 3.0, 9.0]
    assert "bin above it" in meta["ties"]


def test_the_quartile_and_decile_binners_share_one_tie_convention():
    """Two tables in one report must not bin the same tied row differently."""
    from smallAntibodyGen.experiments import her2_ches as ches
    values = np.asarray([1.0, 1.0, 1.0, 1.0, 5.0, 9.0, 9.0, 9.0])
    quartiles, quartile_meta = scoring.quartile_labels(values)
    deciles, decile_meta = ches.value_bins(values, parts=4, prefix="q")
    assert list(quartiles) == list(deciles)
    assert quartile_meta["edge_side"] == decile_meta["edge_side"]


def test_an_empty_declared_stratum_is_reported_as_empty_not_omitted():
    drop = np.asarray([0.5, 1.5, 2.5, 3.5])
    labels = np.asarray(["0", "0", "1", "1"], dtype=object)
    matrix = scoring.bootstrap_index_matrix(4, draws=40, seed=2)
    block = scoring.stratum_statistics(drop, labels, matrix, categories=["0", "1", "2", ">=3"])
    assert block["strata"][">=3"]["rows"] == 0
    assert block["strata"][">=3"]["mean"] is None
    assert block["strata"]["0"]["rows"] == 2


def test_a_single_row_stratum_reports_its_mean_with_no_uncertainty():
    drop = np.asarray([1.0, 2.0, 3.0])
    labels = np.asarray(["a", "b", "b"], dtype=object)
    matrix = scoring.bootstrap_index_matrix(3, draws=20, seed=4)
    block = scoring.stratum_statistics(drop, labels, matrix)["strata"]["a"]
    assert block["mean"] == 1.0 and block["standard_error"] is None
    assert "fewer than two rows" in block["reason"]


def test_stratum_bootstrap_seeds_are_stable_across_processes():
    """hash() of a str is randomized per process; a seed derived from it is not a seed."""
    program = ("import numpy as np;"
               "from smallAntibodyGen.experiments import her2_support_scoring as s;"
               "m = s.bootstrap_index_matrix(16, draws=4, seed=5);"
               "print(s._stratum_seed('q1', m))")
    outputs = set()
    for hash_seed in ("1", "2"):
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                                env={**_environment(), "PYTHONHASHSEED": hash_seed})
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"the stratum seed moved with PYTHONHASHSEED: {outputs}"


def _environment():
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) + (f"{os.pathsep}{existing}" if existing else "")
    return environment


def test_stable_seed_depends_on_purpose_and_parts():
    assert scoring.stable_seed("q1", purpose="a") != scoring.stable_seed("q1", purpose="b")
    assert scoring.stable_seed("q1", purpose="a") == scoring.stable_seed("q1", purpose="a")


# ---------------------------------------------------------------------------
# the declared decision rule
# ---------------------------------------------------------------------------

SETTINGS = {"endpoint_gpu_seconds": 600.0, "methods": ["continued_sft", "ipo_tau0p1"],
            "seeds": [1, 2, 3], "min_seeds": 2, "of_seeds": 3,
            "tenfold_wilson_lower_threshold": 0.01,
            "hundredfold_wilson_lower_threshold": 0.001,
            "intended_preservation_target": "the parent's own distribution"}


def seed_row(seed, tenfold, hundredfold=0.0):
    return {"seed": seed, "usable": True, "tenfold_wilson_lower": tenfold,
            "hundredfold_wilson_lower": hundredfold, "tenfold_fraction": tenfold,
            "hundredfold_fraction": hundredfold, "forward_kl": 0.1}


def test_tenfold_counts_119_and_120_straddle_the_declared_threshold():
    """The rule is on the Wilson lower bound, which 119/10000 does not clear."""
    low = scoring.wilson_interval(119, 10000)["lower"]
    high = scoring.wilson_interval(120, 10000)["lower"]
    assert low <= 0.01 < high
    below = scoring.method_decision([seed_row(s, low) for s in (1, 2, 3)], settings=SETTINGS)
    above = scoring.method_decision([seed_row(s, high) for s in (1, 2, 3)], settings=SETTINGS)
    assert below["outcome"] == scoring.DECISION_NO_ESCALATION
    assert above["outcome"] == scoring.DECISION_ESCALATE


def test_hundredfold_counts_16_and_17_straddle_the_declared_threshold():
    low = scoring.wilson_interval(16, 10000)["lower"]
    high = scoring.wilson_interval(17, 10000)["lower"]
    assert low <= 0.001 < high
    rows_low = [seed_row(s, 0.0, low) for s in (1, 2, 3)]
    rows_high = [seed_row(s, 0.0, high) for s in (1, 2, 3)]
    assert scoring.method_decision(rows_low, settings=SETTINGS)["outcome"] == \
        scoring.DECISION_NO_ESCALATION
    assert scoring.method_decision(rows_high, settings=SETTINGS)["outcome"] == \
        scoring.DECISION_ESCALATE


def test_a_lower_bound_exactly_equal_to_the_threshold_does_not_cross():
    rows = [seed_row(s, 0.01, 0.001) for s in (1, 2, 3)]
    block = scoring.method_decision(rows, settings=SETTINGS)
    assert block["seeds_crossing"] == 0 and block["outcome"] == scoring.DECISION_NO_ESCALATION


def test_one_crossing_seed_is_not_reported_as_no_seed_crossed():
    rows = [seed_row(1, 0.5), seed_row(2, 0.0), seed_row(3, 0.0)]
    block = scoring.method_decision(rows, settings=SETTINGS)
    assert block["seeds_crossing"] == 1
    assert block["outcome"] == scoring.DECISION_NO_ESCALATION
    assert "not 'no seed crossed'" in block["outcome_note"]


def test_two_of_three_crossings_meet_the_rule():
    rows = [seed_row(1, 0.5), seed_row(2, 0.5), seed_row(3, 0.0)]
    assert scoring.method_decision(rows, settings=SETTINGS)["outcome"] == \
        scoring.DECISION_ESCALATE


def test_a_missing_third_seed_cannot_pass_as_a_measured_result():
    rows = [seed_row(1, 0.0), seed_row(2, 0.0)]
    block = scoring.method_decision(rows, settings=SETTINGS)
    assert block["outcome"] == scoring.DECISION_INSUFFICIENT
    assert block["seeds_usable"] == 2 and len(block["seeds"]) == 3
    assert any(row["seed"] == 3 and not row["usable"] for row in block["seeds"])


def test_a_duplicated_seed_cannot_satisfy_the_two_of_three_rule():
    rows = [seed_row(1, 0.5), seed_row(1, 0.5), seed_row(2, 0.0)]
    with pytest.raises(ValueError, match="more than once"):
        scoring.method_decision(rows, settings=SETTINGS)


def test_an_unusable_seed_is_missing_coverage_not_a_negative_result():
    rows = [seed_row(1, 0.5), seed_row(2, 0.5),
            {"seed": 3, "usable": False, "reason": "verification failed"}]
    block = scoring.method_decision(rows, settings=SETTINGS)
    assert block["outcome"] == scoring.DECISION_ESCALATE       # two seeds did cross
    assert block["unusable"][0]["reason"] == "verification failed"


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_nonfinite_decision_input_is_a_failure_not_a_threshold_result(value):
    rows = [seed_row(1, value), seed_row(2, 0.0), seed_row(3, 0.0)]
    with pytest.raises(ValueError, match="finite decision inputs"):
        scoring.method_decision(rows, settings=SETTINGS)


def test_incomplete_coverage_cannot_become_a_preservation_finding():
    per_method = {name: [seed_row(s, 0.0) for s in (1, 2, 3)] for name in SETTINGS["methods"]}
    complete = {"complete": True}
    finished = {"complete": True, "unmet": []}
    good = scoring.decision_record(per_method, settings=SETTINGS, coverage=complete,
                                   completion=finished)
    assert good["outcome"] == scoring.DECISION_NO_ESCALATION
    gapped = scoring.decision_record(per_method, settings=SETTINGS,
                                     coverage={"complete": False}, completion=finished)
    assert gapped["outcome"] == scoring.DECISION_INSUFFICIENT
    assert "coverage table reports gaps" in " ".join(gapped["blocking"])
    unfinished = scoring.decision_record(per_method, settings=SETTINGS, coverage=complete,
                                         completion={"complete": False, "unmet": ["ches"]})
    assert unfinished["outcome"] == scoring.DECISION_INSUFFICIENT
    assert unfinished["measured_rule_by_method"]["continued_sft"] == \
        scoring.DECISION_NO_ESCALATION


def test_an_absent_ches_stage_does_not_block_a_measured_escalation():
    """CHES is not a positive gate; only required coverage is."""
    per_method = {"continued_sft": [seed_row(s, 0.5) for s in (1, 2, 3)],
                  "ipo_tau0p1": [seed_row(s, 0.0) for s in (1, 2, 3)]}
    record = scoring.decision_record(per_method, settings=SETTINGS, coverage={"complete": True},
                                     completion={"complete": True, "unmet": []},
                                     ches_summary=None)
    assert record["outcome"] == scoring.DECISION_ESCALATE
    assert record["ches"] is None
    assert "not a required positive gate" in record["ches_note"]


def test_a_missing_declared_method_is_insufficient_coverage():
    record = scoring.decision_record({"continued_sft": [seed_row(s, 0.0) for s in (1, 2, 3)]},
                                     settings=SETTINGS, coverage={"complete": True},
                                     completion={"complete": True, "unmet": []})
    assert record["outcome"] == scoring.DECISION_INSUFFICIENT
    assert record["methods_missing"] == ["ipo_tau0p1"]


# ---------------------------------------------------------------------------
# the strict scoring route checks every logit, not only the answer
# ---------------------------------------------------------------------------

class StubPolicy:
    """A policy seam whose logits the test controls exactly."""

    core_length = 3

    def __init__(self, logits):
        import torch
        self.model = torch.nn.Identity()
        self._logits = logits

    def core_index(self, index):
        import torch
        return torch.as_tensor(np.asarray(index), dtype=torch.long)

    def token_ids(self, index):
        return self.core_index(index)

    def core_logits(self, core_ids):
        return self._logits[:core_ids.shape[0]]

    def full_logits(self, core_ids):
        return self.core_logits(core_ids)


def stub_logits(value=0.0, position=None, category=None, rows=2):
    torch = pytest.importorskip("torch")
    logits = torch.zeros((rows, StubPolicy.core_length, 20))
    if position is not None:
        logits[0, position, category] = value
    return logits


def test_the_strict_scorer_sums_the_selected_log_probabilities():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, StubPolicy.core_length, 20))
    policy = StubPolicy(logits)
    index = np.zeros((1, StubPolicy.core_length), dtype=np.int64)
    block = scoring.strict_sequence_log_probabilities(policy, index, batch_size=4, label="stub")
    assert block["sum_log_probability"][0] == pytest.approx(3 * math.log(1 / 20))
    assert block["checks"]["logits_checked"] == 60
    assert block["checks"]["mode"] == "eval + inference_mode"


@pytest.mark.parametrize("value,category", [
    (float("nan"), 0), (float("inf"), 0), (float("-inf"), 0),
    # An UNSELECTED category: log_softmax leaves the selected value finite, so a
    # check on the output alone would pass this artifact.
    (float("-inf"), 7), (float("nan"), 7), (float("inf"), 7)])
def test_a_nonfinite_logit_fails_the_artifact_even_on_an_unselected_residue(value, category):
    policy = StubPolicy(stub_logits(value, position=1, category=category, rows=1))
    index = np.zeros((1, StubPolicy.core_length), dtype=np.int64)
    with pytest.raises(ValueError, match="nonfinite"):
        scoring.strict_sequence_log_probabilities(policy, index, batch_size=4, label="stub")


def test_the_strict_scorer_refuses_a_wrong_core_width():
    policy = StubPolicy(stub_logits(rows=1))
    with pytest.raises(ValueError, match="canonical cores"):
        scoring.strict_sequence_log_probabilities(policy, np.zeros((1, 5), dtype=np.int64),
                                                  batch_size=4, label="stub")
