"""Task losses, the tail penalty, and the gradient claims that are easy to overstate.

Two tolerances, deliberately far apart:

* the **exact** identity between IPO and scaled DPO at ``Delta = 0`` is a
  float64 CPU parameter-level statement and is asserted at ``1e-12``;
* native float32 kernels differ by reduction order -- a measured ~2e-5 on the
  production architecture. That is a separate, native, opt-in measurement and is
  never used to loosen the tolerance above.

Conflating them is how "the gradients match" turns into a claim nobody checked.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_nf_objectives as objectives
from smallAntibodyGen.experiments import her2_replay as replay_lib
from smallAntibodyGen.tests.her2_nf_support import TinyPolicy, tiny_cores


def _policy():
    return TinyPolicy(seed=3)


def test_initial_scale_identity_is_arithmetic_not_prose():
    for beta in (0.1, 0.5, 2.0):
        block = objectives.initial_scale_identity(beta)
        assert block["ipo_derivative_at_zero"] == pytest.approx(-10.0)
        assert block["raw_dpo_derivative_at_zero"] == pytest.approx(-beta / 2.0)
        assert block["scaled_dpo_derivative_at_zero"] == pytest.approx(-10.0, rel=1e-12)


@pytest.mark.parametrize("beta", [0.1, 0.5])
def test_ipo_and_scaled_dpo_share_their_parameter_gradient_at_equal_weights(beta):
    """At Delta = 0 the two losses have the same slope, so their PARAMETER gradients agree.

    The premise is what makes this checkable: the reference is the policy's own
    live scores, so every Delta is exactly zero. A cached reference built at a
    different batch shape breaks the premise by float32 reduction order alone --
    which is measured separately and is not a failure of this identity.
    """
    chosen = tiny_cores(16, seed=1)
    rejected = tiny_cores(16, seed=2)

    def gradient(task, coefficients):
        policy = _policy()
        policy.model.zero_grad(set_to_none=True)
        policy_chosen = policy.sequence_log_probs(chosen)
        policy_rejected = policy.sequence_log_probs(rejected)
        reference_chosen = policy_chosen.detach().clone()
        reference_rejected = policy_rejected.detach().clone()
        mean, block = objectives.task_term(
            task, policy_chosen=policy_chosen, policy_rejected=policy_rejected,
            reference_chosen=reference_chosen, reference_rejected=reference_rejected,
            coefficients=coefficients)
        mean.backward()
        return policy.model.table.grad.detach().clone(), block

    ipo_grad, ipo_block = gradient("ipo", {"tau": 0.1})
    dpo_grad, dpo_block = gradient("dpo", {"beta": beta})
    assert torch.allclose(ipo_grad, dpo_grad, atol=1e-12, rtol=1e-10)
    assert dpo_block["loss_multiplier"] == pytest.approx(20.0 / beta)
    assert dpo_block["raw_task_loss"] * dpo_block["loss_multiplier"] == pytest.approx(
        dpo_block["scaled_task_loss"], rel=1e-12)
    assert ipo_block["loss_multiplier"] == 1.0


def test_a_nonzero_margin_breaks_the_identity_which_is_the_point():
    """The identity is about ONE point. Away from it the two losses differ."""
    chosen = tiny_cores(8, seed=4)
    rejected = tiny_cores(8, seed=5)
    policy = _policy()
    policy_chosen = policy.sequence_log_probs(chosen)
    policy_rejected = policy.sequence_log_probs(rejected)
    reference_chosen = policy_chosen.detach() + 1.0     # Delta != 0 by construction
    reference_rejected = policy_rejected.detach()

    def gradient(task, coefficients):
        policy.model.zero_grad(set_to_none=True)
        mean, _ = objectives.task_term(
            task, policy_chosen=policy.sequence_log_probs(chosen),
            policy_rejected=policy.sequence_log_probs(rejected),
            reference_chosen=reference_chosen, reference_rejected=reference_rejected,
            coefficients=coefficients)
        mean.backward()
        return policy.model.table.grad.detach().clone()

    assert not torch.allclose(gradient("ipo", {"tau": 0.1}), gradient("dpo", {"beta": 0.5}),
                              atol=1e-6)


def test_tail_gradient_matches_the_analytic_derivative_including_both_kinks():
    buffer = objectives.TAIL_BUFFER
    for threshold, tolerance in ((math.log(10.0), 0.01), (math.log(100.0), 0.001)):
        for offset in (-1.0, -buffer, -0.5 * buffer, 0.0, 0.5, 2.0):
            drop = torch.tensor([threshold + offset], dtype=torch.float64, requires_grad=True)
            value = objectives.buffered_squared_hinge(drop, threshold, buffer=buffer)
            value.sum().backward()
            inner = (float(drop) - (threshold - buffer)) / buffer
            expected = 0.0 if inner <= 0 else 2.0 * inner / buffer
            assert float(drop.grad) == pytest.approx(expected, abs=1e-12)
        assert tolerance > 0


def test_the_penalty_upper_bounds_the_threshold_indicator():
    """1{d >= t} <= h_t(d) is what makes the population mean bound the tail RATE."""
    values = torch.linspace(-2.0, 8.0, 400, dtype=torch.float64)
    for threshold in (math.log(10.0), math.log(100.0)):
        hinge = objectives.buffered_squared_hinge(values, threshold)
        indicator = (values >= threshold).double()
        assert bool((hinge >= indicator - 1e-12).all())


def test_the_penalty_and_its_gradient_vanish_at_identical_distributions():
    """Which is exactly why an initial gradient-ratio calibration cannot set its lambda."""
    parent = torch.zeros(32, dtype=torch.float64)
    policy = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    mean, block = objectives.tail_term(parent, policy)
    assert float(mean) == 0.0
    mean.backward()
    assert float(policy.grad.abs().max()) == 0.0
    assert block["active_fraction"] == 0.0
    assert "initial gradient-ratio calibration cannot set this lambda" in block["inert_note"]


def test_tail_term_refuses_a_parent_that_carries_gradient():
    parent = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    policy = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        objectives.tail_term(parent, policy)


def test_tail_diagnostics_expose_an_inactive_ladder():
    """A ladder whose penalty never activated has not distinguished its lambdas."""
    parent = torch.full((64,), 0.0, dtype=torch.float64)
    policy = torch.full((64,), 0.0, dtype=torch.float64)
    _, quiet = objectives.tail_term(parent, policy)
    assert quiet["active_fraction"] == 0.0
    assert quiet["inclusive_event_counts"] == {"hundredfold": 0, "tenfold": 0}
    loud_policy = torch.full((64,), -5.0, dtype=torch.float64)
    _, loud = objectives.tail_term(parent, loud_policy)
    assert loud["active_fraction"] == 1.0
    assert loud["inclusive_event_counts"]["tenfold"] == 64
    assert loud["component_means"]["hundredfold"] > 0


def test_inclusive_counting_at_exactly_the_threshold():
    """``drop >= ln 10`` counts; ``drop`` one nanonat below it does not.

    The drop is ``ell_P - ell_Q``, so a policy log probability of
    ``-ln 10 + 1e-9`` is a drop of ``ln 10 - 1e-9`` -- BELOW the threshold. The
    earlier fixture expected two events from one event and one near miss; the
    inclusive comparison was right and the fixture was wrong.
    """
    parent = torch.zeros(4, dtype=torch.float64)
    policy = torch.tensor([-math.log(10.0),            # exactly at the threshold: counts
                           -math.log(10.0) + 1e-9,     # a nanonat below it: does not
                           -math.log(10.0) - 1e-9,     # a nanonat above it: counts
                           -1.0], dtype=torch.float64)
    _, block = objectives.tail_term(parent, policy)
    assert block["inclusive_event_counts"]["tenfold"] == 2
    # The same fixture under the strict historical convention loses the boundary
    # row, which is exactly the difference the two conventions are named apart
    # for.
    drop = (parent - policy).detach().numpy()
    assert int((drop >= math.log(10.0)).sum()) == 2
    assert int((drop > math.log(10.0)).sum()) == 1


def test_registry_declares_what_each_arm_consumes():
    assert objectives.TASK_OBJECTIVES["dpo"].uses_rejected is True
    assert objectives.TASK_OBJECTIVES["simpo"].uses_reference is False
    assert objectives.TASK_OBJECTIVES["ipo"].uses_reference is True
    # DPO consuming a rejected row is exactly what a task=="ipo" exposure rule
    # would have under-counted.
    assert all(spec.uses_rejected for spec in objectives.TASK_OBJECTIVES.values())


def test_resolve_arm_refuses_a_zero_weighted_preservation_control():
    with pytest.raises(ValueError, match="none"):
        objectives.resolve_arm("ipo", "none", {"tau": 0.1, "lambda": 0.5})
    block = objectives.resolve_arm("dpo", "fkl", {"beta": 0.5, "lambda": 10.0})
    assert block["dpo_scale"] == pytest.approx(40.0)
    assert block["raw_lambda_equivalent"] == pytest.approx(10.0 / 40.0)


def test_simpo_has_no_reference_inside_its_task_loss():
    policy = _policy()
    chosen = policy.sequence_log_probs(tiny_cores(8, seed=6))
    rejected = policy.sequence_log_probs(tiny_cores(8, seed=7))
    mean, block = objectives.simpo_term(policy_chosen=chosen, policy_rejected=rejected,
                                        beta_s=2.0, gamma=1.0)
    manual = torch.nn.functional.softplus(1.0 - 2.0 * (chosen - rejected) / 10.0).mean()
    assert float(mean) == pytest.approx(float(manual), rel=1e-12)
    assert "parent" in block["reference_note"]


def test_microbatch_accumulation_reproduces_the_full_batch_mean_on_a_partial_tail():
    """The n_i/N_i weighting and a 1/k average agree only for an exact partition."""
    policy = _policy()
    chosen = tiny_cores(20, seed=8)
    rejected = tiny_cores(20, seed=9)
    reference_chosen = policy.sequence_log_probs(chosen).detach()
    reference_rejected = policy.sequence_log_probs(rejected).detach()
    full, _ = objectives.task_term(
        "ipo", policy_chosen=policy.sequence_log_probs(chosen),
        policy_rejected=policy.sequence_log_probs(rejected),
        reference_chosen=reference_chosen, reference_rejected=reference_rejected,
        coefficients={"tau": 0.1})
    accumulator = replay_lib.MicrobatchAccumulator({"task": 20}, coefficients={"task": 1.0},
                                                   label="partial tail")
    for start in range(0, 20, 8):                       # 8 + 8 + 4
        stop = min(start + 8, 20)
        mean, _ = objectives.task_term(
            "ipo", policy_chosen=policy.sequence_log_probs(chosen[start:stop]),
            policy_rejected=policy.sequence_log_probs(rejected[start:stop]),
            reference_chosen=reference_chosen[start:stop],
            reference_rejected=reference_rejected[start:stop], coefficients={"tau": 0.1})
        accumulator.add({"task": stop - start}, {"task": mean})
    assert accumulator.finish()["component_means"]["task"] == pytest.approx(float(full), rel=1e-12)


def test_gradient_term_diagnostics_report_a_cosine_not_just_two_norms():
    from smallAntibodyGen.experiments.her2_nf_trajectory import gradient_term_diagnostics
    policy = _policy()
    cores = tiny_cores(8, seed=10)

    def task():
        return -policy.sequence_log_probs(cores).mean()

    def preservation():
        return policy.sequence_log_probs(cores).mean()       # deliberately opposed

    block = gradient_term_diagnostics(policy.model, task, preservation)
    assert block["cosine"] == pytest.approx(-1.0, abs=1e-9)
    assert block["task_gradient_norm"] > 0 and block["preservation_gradient_norm"] > 0
    # The diagnostic must leave no gradient behind for the real update.
    assert all(parameter.grad is None for parameter in policy.model.parameters())


def test_gradient_diagnostics_helper_handles_a_missing_component():
    block = objectives.gradient_diagnostics([torch.ones(3)], [])
    assert block["cosine"] is None
    assert "no gradient" in block["reason"]
    assert np.isfinite(block["task_gradient_norm"])
