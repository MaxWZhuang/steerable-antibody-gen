"""The sequence mixture: its floor, its true conditionals, and the error next door.

The failure this file is built around is a fixed-weight mixture of per-position
conditionals. It looks right, it is normalized, and it defines a *different*
distribution -- so its sequence probability does not equal ``M_alpha``. The
chain-rule test is what separates the two.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_nf_mixture as mixture
from smallAntibodyGen.tests.her2_nf_support import SmallModel, TinyPolicy, tiny_cores


def _small_pair(length=3, alphabet=3):
    return (TinyPolicy(SmallModel(seed=1, length=length, alphabet=alphabet)),
            TinyPolicy(SmallModel(seed=2, length=length, alphabet=alphabet)))


def test_pointwise_floor_is_exactly_one_minus_alpha():
    parent = np.array([-3.0, -12.0, -0.25])
    for alpha in (0.25, 0.89, 0.9, 0.99):
        worst = mixture.mixture_log_probability(parent, np.full(parent.shape, -np.inf), alpha)
        ratio = np.exp(worst - parent)
        assert np.allclose(ratio, 1.0 - alpha, rtol=1e-12)
        block = mixture.floor_record(alpha)
        assert block["ratio_floor"] == pytest.approx(1.0 - alpha)


def test_alpha_089_excludes_an_inclusive_tenfold_loss_and_090_does_not():
    """Which is the whole reason .89 is primary at an inclusive threshold."""
    primary = mixture.floor_record(0.89)
    secondary = mixture.floor_record(0.9)
    assert primary["excludes_inclusive_tenfold"] is True
    assert secondary["excludes_inclusive_tenfold"] is False   # 1/0.1 is EXACTLY tenfold
    assert primary["excludes_inclusive_hundredfold"] is True
    assert secondary["excludes_inclusive_hundredfold"] is True


def test_alpha_099_does_not_exclude_an_inclusive_hundredfold_loss():
    """The floating-point boundary case: ``1 - .99`` is .010000000000000009.

    Comparing that against ``.01`` certifies exclusion at the exact boundary,
    where the inclusive event is attained. The comparison is on the declared
    alpha instead.
    """
    assert (1.0 - 0.99) > 0.01                       # the trap, stated
    block = mixture.floor_record(0.99)
    assert block["excludes_inclusive_hundredfold"] is False
    assert block["excludes_inclusive_tenfold"] is False
    # And one step inside the boundary still excludes it.
    assert mixture.floor_record(0.98)["excludes_inclusive_hundredfold"] is True


def test_the_certified_floor_is_attained_exactly_at_the_boundary_alpha():
    """At alpha = .99 a zero-probability Q loses exactly a hundredfold, not less."""
    parent = np.array([-1.0, -7.5])
    mixed = mixture.mixture_log_probability(parent, np.full(parent.shape, -np.inf), 0.99)
    assert np.allclose(np.exp(parent - mixed), 100.0, rtol=1e-12)


def test_breadth_statistic_is_unchanged_for_every_positive_alpha():
    """M_alpha(x) < P(x) iff Q(x) < P(x): only the magnitudes shrink."""
    generator = np.random.default_rng(5)
    parent = -generator.random(500) * 10.0
    policy = -generator.random(500) * 10.0
    loses_under_q = policy < parent
    for alpha in (0.05, 0.5, 0.89, 0.99):
        mixed = mixture.mixture_log_probability(parent, policy, alpha)
        assert np.array_equal(mixed < parent - 1e-15, loses_under_q)


def test_posterior_weights_start_at_alpha_and_then_move():
    parent, policy = _small_pair()
    cores = tiny_cores(12, seed=3, length=3, alphabet=3)
    parent_positions = parent.position_log_probs(cores).detach().numpy()
    policy_positions = policy.position_log_probs(cores).detach().numpy()
    weights = mixture.posterior_component_weights(parent_positions, policy_positions, 0.89)
    assert np.allclose(weights[:, 0], 0.89)
    assert not np.allclose(weights[:, 1], 0.89)          # the prefix has been observed by now


def test_chain_rule_holds_for_the_posterior_weighted_conditionals():
    parent, policy = _small_pair()
    cores = tiny_cores(24, seed=4, length=3, alphabet=3)
    parent_positions = parent.position_log_probs(cores).detach().numpy()
    policy_positions = policy.position_log_probs(cores).detach().numpy()
    for alpha in mixture.ALPHA_GRID:
        block = mixture.chain_rule_check(parent_positions, policy_positions, alpha)
        assert block["max_abs_difference"] < 1e-12


def test_chain_rule_holds_when_one_component_assigns_zero_mass():
    """P = [.5, .5], Q = [0, .9], alpha = .89 -- the reported 1.0296-nat discrepancy.

    Reconstructing the parent weight as ``log1p(-w_Q)`` loses the zero-mass
    component exactly where the certified floor is about it. Both weights come
    from the same normalizer instead.
    """
    parent = np.log(np.array([[0.5, 0.5]]))
    policy = np.log(np.array([[0.0, 0.9]]))
    block = mixture.chain_rule_check(parent, policy, 0.89)
    assert block["max_abs_difference"] < 1e-12
    assert block["rows_with_zero_mixture_mass"] == 0


def test_chain_rule_holds_for_extreme_but_finite_logs():
    """P = [-100, 0], Q = [0, -100]: the posterior rounds to one in float64.

    The reported discrepancy was .1165338 nats -- all of it from ``log1p(-1.0)``
    returning ``-inf`` where the true log weight is about -100.
    """
    parent = np.array([[-100.0, 0.0]])
    policy = np.array([[0.0, -100.0]])
    block = mixture.chain_rule_check(parent, policy, 0.5)
    assert block["direct"][0] == pytest.approx(-100.0 + math.log(0.5) + math.log(2.0), abs=1e-9)
    assert block["max_abs_difference"] < 1e-9


def test_chain_rule_holds_over_the_whole_enumerated_support():
    """Every sequence of a 3x3 model, at every declared alpha, to 1e-12."""
    parent, policy = _small_pair(length=3, alphabet=3)
    cores = np.asarray(list(itertools.product(range(3), repeat=3)), dtype=np.int8)
    parent_positions = parent.position_log_probs(cores).detach().numpy()
    policy_positions = policy.position_log_probs(cores).detach().numpy()
    for alpha in mixture.ALPHA_GRID:
        block = mixture.chain_rule_check(parent_positions, policy_positions, alpha)
        assert block["max_abs_difference"] < 1e-12
        # The enumerated mixture is a probability distribution at every alpha.
        assert float(np.exp(block["direct"]).sum()) == pytest.approx(1.0, abs=1e-12)


def test_log_conditionals_keep_a_value_that_probability_space_would_lose():
    """A conditional that underflows to zero as a probability is finite as a log.

    The old path was ``softmax -> clip(1e-300) -> log``, which reports every such
    conditional as exactly ``log 1e-300 = -690.78`` -- the floor, not the value.
    """
    policy = TinyPolicy(SmallModel(seed=3, length=3, alphabet=3))
    with torch.no_grad():
        policy.model.table[0, policy.start_context, 0] = -800.0
    cores = np.zeros((2, 3), dtype=np.int8)
    logs = mixture.position_log_conditionals(policy, cores, batch_size=2)
    assert np.isfinite(logs[0, 0, 0])
    assert logs[0, 0, 0] < -700.0                     # below the old clip floor
    # The same number, exponentiated, is exactly zero -- which is what a
    # probability-space pipeline would then have to clip and take the log of.
    assert float(np.exp(logs[0, 0, 0])) == 0.0
    # The remaining residues still sum to one, so this is a real conditional.
    assert float(np.exp(logs[0, 0]).sum()) == pytest.approx(1.0, abs=1e-12)


def test_a_fixed_alpha_at_every_position_is_a_different_distribution():
    """The plausible wrong implementation, shown to be wrong rather than warned about."""
    parent, policy = _small_pair()
    cores = tiny_cores(24, seed=6, length=3, alphabet=3)
    parent_positions = parent.position_log_probs(cores).detach().numpy()
    policy_positions = policy.position_log_probs(cores).detach().numpy()
    alpha = 0.89
    fixed = np.logaddexp(math.log1p(-alpha) + parent_positions,
                         math.log(alpha) + policy_positions).sum(axis=1)
    correct = mixture.mixture_log_probability(parent_positions.sum(axis=1),
                                              policy_positions.sum(axis=1), alpha)
    assert not np.allclose(fixed, correct, atol=1e-6)


def test_mixture_conditionals_are_normalized_and_match_an_enumerated_mixture():
    parent, policy = _small_pair(length=3, alphabet=3)
    cores = np.asarray(list(itertools.product(range(3), repeat=3)), dtype=np.int8)
    parent_cond = mixture.position_conditionals(parent, cores, batch_size=8)
    policy_cond = mixture.position_conditionals(policy, cores, batch_size=8)
    parent_positions = parent.position_log_probs(cores).detach().numpy()
    policy_positions = policy.position_log_probs(cores).detach().numpy()
    weights = mixture.posterior_component_weights(parent_positions, policy_positions, 0.75)
    conditionals = mixture.mixture_conditionals(parent_cond, policy_cond, weights)
    assert np.allclose(conditionals.sum(axis=2), 1.0, atol=1e-12)
    rows = np.arange(cores.shape[0])[:, None]
    positions = np.arange(3)[None, :]
    realized = np.log(conditionals[rows, positions, cores]).sum(axis=1)
    direct = mixture.mixture_log_probability(parent_positions.sum(axis=1),
                                             policy_positions.sum(axis=1), 0.75)
    assert np.allclose(realized, direct, atol=1e-12)
    # And the enumerated mixture is a probability distribution.
    assert float(np.exp(direct).sum()) == pytest.approx(1.0, abs=1e-12)


def test_endpoints_are_the_components_themselves():
    parent = np.array([-2.0, -5.0])
    policy = np.array([-9.0, -1.0])
    assert np.array_equal(mixture.mixture_log_probability(parent, policy, 0.0), parent)
    assert np.array_equal(mixture.mixture_log_probability(parent, policy, 1.0), policy)
    with pytest.raises(ValueError, match="alpha outside"):
        mixture.mixture_log_probability(parent, policy, 1.5)


def test_generation_chooses_the_component_once_per_sequence():
    parent, policy = _small_pair(length=3, alphabet=3)
    block = mixture.sample_mixture(parent, policy, count=400, alpha=0.75, component_seed=11,
                                   parent_seed=12, policy_seed=13)
    assert set(np.unique(block["component"]).tolist()) <= {0, 1}
    assert block["draws_from_parent"] + block["draws_from_policy"] == 400
    # One component per SEQUENCE, so the realized share is binomial around alpha.
    assert 0.65 < block["draws_from_policy"] / 400 < 0.85
    assert block["component"].shape == (400,)


def test_score_mixture_uses_both_models_and_says_so():
    parent, policy = _small_pair(length=3, alphabet=3)
    cores = tiny_cores(16, seed=14, length=3, alphabet=3)
    block = mixture.score_mixture(parent, policy, cores, alpha=0.89)
    expected = mixture.mixture_log_probability(block["parent_sum_log_probability"],
                                               block["policy_sum_log_probability"], 0.89)
    assert np.allclose(block["mixture_sum_log_probability"], expected)
    assert "two full scoring passes" in block["cost_note"]


def test_alpha_curve_persists_every_grid_point():
    parent = np.array([-4.0, -6.0, -2.0, -9.0])
    policy = np.array([-3.0, -8.0, -2.5, -7.0])
    labels = np.array([True, True, False, True])
    curve = mixture.alpha_curve(parent, policy, labels=labels,
                                yield_function=lambda values, draws: float(len(values)))
    assert [point["alpha"] for point in curve["points"]] == list(mixture.ALPHA_GRID)
    assert curve["primary_alpha"] == 0.89
    assert "never presented as prespecified" in curve["selection_note"]


def test_a_stopped_run_stays_stopped():
    note = mixture.stopped_run_note("A_DPO_0_seed20260918", 725)
    assert note["status"] == "stopped"
    assert "does not become a completed arm" in note["use"]


def test_position_conditionals_restore_the_training_mode():
    policy = TinyPolicy(seed=7)
    policy.model.train()
    mixture.position_conditionals(policy, tiny_cores(4, seed=8), batch_size=2)
    assert policy.model.training is True
    policy.model.eval()
    mixture.position_conditionals(policy, tiny_cores(4, seed=8), batch_size=2)
    assert policy.model.training is False


def test_conditional_vectors_for_mixture_uses_posterior_weights():
    parent, policy = _small_pair(length=3, alphabet=3)
    cores = tiny_cores(10, seed=15, length=3, alphabet=3)
    block = mixture.conditional_vectors_for_mixture(parent, policy, cores, alpha=0.5,
                                                    batch_size=4)
    assert block["posterior_weights"].shape == (10, 3)
    assert np.allclose(block["posterior_weights"][:, 0], 0.5)
    assert np.allclose(block["conditionals"].sum(axis=2), 1.0, atol=1e-12)


def test_torch_is_not_required_to_be_in_training_mode_for_scoring():
    policy = TinyPolicy(seed=9)
    cores = tiny_cores(6, seed=10)
    with torch.no_grad():
        assert policy.score(cores)["sum_log_probability"].shape == (6,)
