"""I11, I13, I19, I21: normalisation, dependence, preservation, planner, effect fitting.

Tiny synthetic tensors. The two checks that carry the most weight are the ones a
plausible-looking implementation passes anyway:

* an accidentally independent two-site model **normalises perfectly** over 400
  outcomes, so normalisation alone is not the check -- dependence is;
* an undebiased entropic OT has a **gradient** and is strictly positive at identity,
  so a gradient test is not the check -- the identity is.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_objectives as objectives
from smallAntibodyGen.experiments.her2_routes_decoding import kappa_worked_example

_SPEC = Path(__file__).resolve().parents[3] / "specs" / "her2_generative_routes"
if str(_SPEC) not in sys.path:
    sys.path.insert(0, str(_SPEC))
import reference_math  # noqa: E402  -- the standard-library oracle, loaded by path


# --- I11: normalisation and dependence ------------------------------------

def test_i11_one_site_conditionals_sum_to_one():
    logits = torch.randn(4, 3, 20, generator=torch.Generator().manual_seed(0))
    log_probs = torch.log_softmax(logits, dim=-1)
    report = objectives.require_normalised(log_probs, label="one-site")
    assert report["support"] == 20
    assert report["max_abs_sum_error"] < 1e-5


def test_i11_require_normalised_rejects_an_unnormalised_block():
    bad = torch.log_softmax(torch.randn(2, 2, 20), dim=-1) + math.log(2.0)
    with pytest.raises(ValueError, match="does not normalise"):
        objectives.require_normalised(bad, label="doubled")


def test_i11_dependent_two_site_joint_sums_to_one_over_four_hundred():
    generator = torch.Generator().manual_seed(1)
    first = torch.log_softmax(torch.randn(20, generator=generator), dim=-1)
    table = torch.log_softmax(torch.randn(20, 20, generator=generator), dim=-1)

    def step(prefix):
        return first if not prefix else table[prefix[0]]

    joint = objectives.sequential_two_site(step)
    assert joint.shape[-1] == 400
    assert float(joint.exp().sum()) == pytest.approx(1.0, abs=1e-5)
    dependence = objectives.dependence_report(step)
    assert not dependence["independent"]
    assert dependence["max_total_variation"] > 0.0


def test_i11_an_independent_model_also_normalises_which_is_the_point():
    generator = torch.Generator().manual_seed(2)
    first = torch.log_softmax(torch.randn(20, generator=generator), dim=-1)
    second = torch.log_softmax(torch.randn(20, generator=generator), dim=-1)

    def step(prefix):
        return first if not prefix else second

    joint = objectives.sequential_two_site(step)
    # It normalises perfectly...
    assert float(joint.exp().sum()) == pytest.approx(1.0, abs=1e-5)
    # ...and it cannot represent any coupling at all.
    dependence = objectives.dependence_report(step)
    assert dependence["independent"]
    assert dependence["max_total_variation"] == 0.0


def test_i11_aa_vv_toy_reproduces_the_fiftyfold_gap():
    from smallAntibodyGen.experiments.her2_routes_decoding import two_site_allocation_example
    block = two_site_allocation_example()
    assert block["generated"]["AA"] == pytest.approx(0.9702)
    assert block["generated"]["VV"] == pytest.approx(0.0098)
    assert block["vv_underrepresentation"] == pytest.approx(50.0)
    assert block["useful_mass_generated"] == pytest.approx(0.98)


# --- I21: effect fitting --------------------------------------------------

def test_i21_absolute_targets_reproduce_the_worked_example():
    block = kappa_worked_example()
    assert block["absolute_target_odds"]["without_partner"] == pytest.approx(0.368, abs=1e-3)
    assert block["absolute_target_odds"]["with_partner"] == pytest.approx(2.226, abs=1e-3)
    assert block["interaction"] == pytest.approx(0.9)
    assert block["reference_relative_with_partner"] == pytest.approx(0.223, abs=1e-3)


def test_i21_absolute_fit_drives_the_log_odds_to_the_target():
    delta = torch.tensor([0.4, -0.5])
    alternative = torch.zeros(2, requires_grad=True)
    baseline = torch.zeros(2)
    loss, record = objectives.fit_absolute_log_odds(alternative, baseline, delta, kappa=2.0)
    assert record["mean_target_log_odds"] == pytest.approx(-0.1)
    loss.backward()
    # The gradient pushes the first pair up and the second down, which is the sign the
    # measurement asks for.
    assert alternative.grad[0] < 0
    assert alternative.grad[1] > 0


def test_i21_reference_relative_requires_a_frozen_reference():
    live = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="must be frozen"):
        objectives.fit_reference_relative_log_odds(
            torch.zeros(2), torch.zeros(2), live, torch.zeros(2), torch.tensor([0.1, 0.2]))


def test_i21_interaction_support_refuses_an_incomplete_quartet():
    partial = {"background": {"value": 8.0, "assay": "a"},
               "single_a": {"value": 7.5, "assay": "a"}, "single_b": None, "double": None}
    block = objectives.interaction_support(partial)
    assert not block["complete"]
    assert block["supports"] == "contextual ordering only"
    assert "epistasis" in block["does_not_support"]


def test_i21_interaction_support_refuses_two_assays():
    mixed = {"background": {"value": 8.0, "assay": "a"},
             "single_a": {"value": 7.5, "assay": "a"},
             "single_b": {"value": 8.2, "assay": "b"},
             "double": {"value": 8.6, "assay": "a"}}
    assert not objectives.interaction_support(mixed)["complete"]


def test_i21_interaction_support_computes_plus_nine_tenths():
    quartet = {name: {"value": value, "assay": "a"} for name, value in
               (("background", 8.0), ("single_a", 7.5), ("single_b", 8.2), ("double", 8.6))}
    block = objectives.interaction_support(quartet)
    assert block["complete"]
    assert block["interaction"] == pytest.approx(0.9)
    assert block["a_without_b"] == pytest.approx(-0.5)
    assert block["a_with_b"] == pytest.approx(0.4)


# --- I13: preservation ----------------------------------------------------

def _teacher(probabilities):
    tensor = torch.as_tensor(probabilities, dtype=torch.float32)
    return tensor, tensor.log()


def test_i13_forward_kl_is_zero_at_identity_and_uses_the_reference_as_the_weight():
    probabilities, logs = _teacher([[[0.7, 0.2, 0.1], [0.5, 0.25, 0.25]]])
    value, record = objectives.forward_kl(probabilities, logs, logs.clone())
    assert float(value) == pytest.approx(0.0, abs=1e-6)
    assert record["positions"] == 2
    # Asymmetric: swapping the arguments changes the value, which is what catches a
    # reversed-argument bug that a symmetric divergence would hide. The pair has to be
    # chosen for it: KL([.7,.2,.1] || [.2,.7,.1]) equals its own reverse, because the
    # two distributions are a permutation of one another and the permuted entry ratios
    # are reciprocal. A pair that is NOT a permutation is what makes this a test.
    other, other_logs = _teacher([[[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]]])
    forward, _ = objectives.forward_kl(probabilities, logs, other_logs)
    reverse, _ = objectives.forward_kl(other, other_logs, logs)
    assert float(forward) != pytest.approx(float(reverse))
    # And the permutation pair the earlier revision used really is symmetric, so the
    # replacement is not superstition.
    mirror, mirror_logs = _teacher([[[0.2, 0.7, 0.1], [0.25, 0.5, 0.25]]])
    symmetric_forward, _ = objectives.forward_kl(probabilities, logs, mirror_logs)
    symmetric_reverse, _ = objectives.forward_kl(mirror, mirror_logs, logs)
    assert float(symmetric_forward) == pytest.approx(float(symmetric_reverse), rel=1e-5)


def test_i13_forward_kl_refuses_a_live_reference_and_an_unnormalised_one():
    probabilities, logs = _teacher([[[0.5, 0.5]]])
    live = probabilities.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="requires_grad"):
        objectives.forward_kl(live, logs, logs.clone())
    bad = torch.as_tensor([[[0.5, 0.9]]])
    with pytest.raises(ValueError, match="summing to one"):
        objectives.forward_kl(bad, bad.log(), bad.log())


def test_i13_forward_kl_agrees_with_the_tested_ten_position_implementation():
    from smallAntibodyGen.experiments import her2_replay
    generator = torch.Generator().manual_seed(3)
    teacher_logs = torch.log_softmax(torch.randn(3, 10, 20, generator=generator), dim=-1)
    teacher = teacher_logs.exp()
    student = torch.log_softmax(torch.randn(3, 10, 20, generator=generator), dim=-1)
    inherited, _ = her2_replay.replay_term(teacher, teacher_logs, student)
    mine, _ = objectives.forward_kl(teacher, teacher_logs, student)
    assert float(mine) == pytest.approx(float(inherited), rel=1e-6)


def test_i13_forward_kl_generalises_past_ten_positions():
    generator = torch.Generator().manual_seed(4)
    teacher_logs = torch.log_softmax(torch.randn(2, 16, 20, generator=generator), dim=-1)
    value, record = objectives.forward_kl(
        teacher_logs.exp(), teacher_logs,
        torch.log_softmax(torch.randn(2, 16, 20, generator=generator), dim=-1))
    assert record["positions"] == 16
    assert float(value) > 0


def test_i13_sinkhorn_is_zero_at_identity_and_positive_otherwise():
    windows = ["AAA", "AAV", "AVV", "VVV"]
    cost = objectives.hamming_cost(windows)
    log_a = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64))
    value, record = objectives.sinkhorn_divergence(log_a, log_a.clone(), cost)
    assert abs(float(value)) <= 1e-5, record
    log_b = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64))
    other, _ = objectives.sinkhorn_divergence(log_a, log_b, cost)
    assert float(other) > 1e-4
    assert record["debiased"] is True


def test_i13_sinkhorn_has_a_gradient_and_responds_to_cost_rescaling():
    windows = ["AA", "AV", "VA", "VV"]
    cost = objectives.hamming_cost(windows)
    logits = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    log_b = torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64))
    value, _ = objectives.sinkhorn_divergence(torch.log_softmax(logits, dim=-1), log_b, cost)
    value.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
    scaled, _ = objectives.sinkhorn_divergence(
        torch.log_softmax(torch.zeros(4, dtype=torch.float64), dim=-1), log_b, 10.0 * cost)
    plain, _ = objectives.sinkhorn_divergence(
        torch.log_softmax(torch.zeros(4, dtype=torch.float64), dim=-1), log_b, cost)
    assert float(scaled) > float(plain)


def test_i13_sinkhorn_agrees_with_an_independent_standard_library_oracle():
    """Two algorithms, two domains. The campaign's runs in logs with tensors.

    The oracle iterates the scaling vectors in the probability domain with
    ``math.fsum``, so an error shared between the two would have to be an error in
    the definition rather than in one implementation. It is also what shows the
    regularised value is the objective: the transport cost alone disagrees.
    """
    windows = ["AAA", "AAV", "AVV", "VVV"]
    cost = objectives.hamming_cost(windows)
    a = [0.4, 0.3, 0.2, 0.1]
    b = [0.1, 0.2, 0.3, 0.4]
    epsilon = objectives.SINKHORN_EPSILON
    plain_cost = [[float(cost[i][j]) for j in range(4)] for i in range(4)]
    # The nearly diagonal self-couplings converge much more slowly than OT(a,b).
    # At 2,000 iterations the two independent solvers retain different residuals;
    # 20,000 resolves the reference to the precision asserted below.
    expected = reference_math.debiased_sinkhorn(plain_cost, a, b, epsilon=epsilon,
                                               iterations=20000)
    value, record = objectives.sinkhorn_divergence(
        torch.log(torch.tensor(a, dtype=torch.float64)),
        torch.log(torch.tensor(b, dtype=torch.float64)), cost,
        epsilon=epsilon, iterations=20000, tolerance=1e-14)
    assert float(value) == pytest.approx(expected, abs=1e-8)
    regularised, transport, penalty = reference_math.entropic_ot(
        plain_cost, a, b, epsilon=epsilon, iterations=20000)
    assert record["ot_ab"] == pytest.approx(regularised, abs=1e-8)
    assert record["transport_cost_ab"] == pytest.approx(transport, abs=1e-8)
    assert record["entropic_penalty_ab"] == pytest.approx(penalty, abs=1e-8)
    assert penalty != pytest.approx(0.0, abs=1e-6), (
        "the entropic penalty is not negligible, which is why returning the transport cost "
        "alone under the name of the regularised objective was wrong")
    assert record["plan_mass_ab"] == pytest.approx(1.0, abs=1e-8)


def test_i13_the_undebiased_entropic_value_is_not_zero_at_identity():
    """The reason the debiasing terms exist, measured on the oracle."""
    windows = ["AAA", "AAV", "AVV", "VVV"]
    cost = [[float(value) for value in row] for row in objectives.hamming_cost(windows)]
    a = [0.4, 0.3, 0.2, 0.1]
    regularised, _transport, _penalty = reference_math.entropic_ot(
        cost, a, a, epsilon=objectives.SINKHORN_EPSILON)
    assert regularised > 1e-3
    assert reference_math.debiased_sinkhorn(
        cost, a, a, epsilon=objectives.SINKHORN_EPSILON) == pytest.approx(0.0, abs=1e-9)


def test_i13_the_teacher_kl_convention_is_declared_not_implied():
    block = objectives.teacher_kl_convention()
    assert "no epsilon is added" in block["no_epsilon"]
    assert block["direction"].startswith("KL(reference || policy)")
    assert "exactly 0" in block["zero_teacher_mass"]
    assert "not clamped" in block["student_zero"]


# --- I19: planner weighting -----------------------------------------------

def test_i19_zero_planner_weight_reproduces_ordinary_training_exactly():
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(2, 4, 20, generator=generator, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.randint(0, 20, (2, 4), generator=generator)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    plain, _ = objectives.masked_nll(log_probs, targets, mask)
    weights, record = objectives.planner_weights(log_probs.detach(), mask, weight=0.0)
    assert record["identical_to_ordinary_training"]
    assert torch.equal(weights, mask.to(weights.dtype))
    weighted = -(log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                 * weights).sum() / weights.sum()
    assert float(weighted) == pytest.approx(float(plain), rel=1e-9)
    plain_grad = torch.autograd.grad(plain, logits, retain_graph=True)[0]
    weighted_grad = torch.autograd.grad(weighted, logits)[0]
    # Effectively exact: the two expressions are the same reduction, so any residual
    # is floating-point op ordering and not a different objective.
    assert torch.allclose(plain_grad, weighted_grad, atol=1e-12, rtol=1e-12)


def test_i19_planner_weights_are_detached_and_mask_normalised():
    generator = torch.Generator().manual_seed(6)
    log_probs = torch.log_softmax(torch.randn(2, 4, 20, generator=generator), dim=-1)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    weights, record = objectives.planner_weights(log_probs, mask, weight=0.5)
    assert not weights.requires_grad
    assert record["detached"]
    assert float(weights.sum()) == pytest.approx(float(mask.sum()), rel=1e-5)
    # Padding never receives weight, so an inactive target cannot contribute.
    assert float(weights[0, 3]) == pytest.approx(0.0)


def test_i19_easy_first_and_hard_first_order_the_weights_oppositely():
    log_probs = torch.log_softmax(
        torch.tensor([[[3.0, 0.0], [0.0, 0.0]]]), dim=-1)
    mask = torch.ones(1, 2)
    easy, _ = objectives.planner_weights(log_probs, mask, weight=1.0, mode="easy_first")
    hard, _ = objectives.planner_weights(log_probs, mask, weight=1.0, mode="hard_first")
    assert float(easy[0, 0]) > float(easy[0, 1])
    assert float(hard[0, 0]) < float(hard[0, 1])


# --- masked NLL -----------------------------------------------------------

def test_masked_nll_normalises_by_active_targets_not_by_width():
    log_probs = torch.log_softmax(torch.zeros(1, 4, 20), dim=-1)
    targets = torch.zeros(1, 4, dtype=torch.long)
    two, record_two = objectives.masked_nll(log_probs, targets, torch.tensor([[1, 1, 0, 0]]))
    four, record_four = objectives.masked_nll(log_probs, targets, torch.ones(1, 4))
    assert record_two["active_targets"] == 2 and record_four["active_targets"] == 4
    assert float(two) == pytest.approx(float(four))


def test_masked_nll_returns_a_differentiable_zero_when_nothing_is_active():
    logits = torch.zeros(1, 2, 20, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss, record = objectives.masked_nll(log_probs, torch.zeros(1, 2, dtype=torch.long),
                                         torch.zeros(1, 2))
    assert record["active_targets"] == 0
    assert float(loss) == 0.0
    loss.backward()
    assert logits.grad is not None      # zeros, but the graph is intact


def test_combine_records_every_term_and_its_contribution():
    total, record = objectives.combine(
        {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        coefficients={"a": 1.0, "b": 0.5})
    assert float(total) == pytest.approx(3.5)
    assert record["terms"]["b"]["contribution"] == pytest.approx(1.5)
