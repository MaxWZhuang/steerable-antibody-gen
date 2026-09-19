"""The six objectives: closed-form gradients, hinge behaviour, frozen references.

Every gradient below is written out here independently rather than imported from
the module under test -- a test that calls the same helper the implementation
calls checks that the code is self-consistent, not that it is right. ``dpop`` and
``ipo`` get a second, independent route as well: a float64 central finite
difference.

Everything runs on the CPU in a fraction of a second and fits nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_objectives as arms


def vectors(*, chosen=(-30.0, -28.0, -33.0), rejected=(-31.0, -35.0, -30.0),
            reference_chosen=(-29.5, -30.0, -34.0), reference_rejected=(-31.5, -33.0, -29.0),
            requires_grad=True, dtype=torch.float64):
    """Four aligned vectors with a mix of active and inactive hinges."""
    lc = torch.tensor(chosen, dtype=dtype, requires_grad=requires_grad)
    lr = torch.tensor(rejected, dtype=dtype, requires_grad=requires_grad)
    refc = torch.tensor(reference_chosen, dtype=dtype)
    refr = torch.tensor(reference_rejected, dtype=dtype)
    return lc, lr, refc, refr


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_terms(lc, lr, refc, refr):
    delta = (lc - lr) - (refc - refr)
    hinge = np.maximum(refc - lc, 0.0)
    return delta, hinge


NUMBERS = {"lc": np.array([-30.0, -28.0, -33.0]), "lr": np.array([-31.0, -35.0, -30.0]),
           "refc": np.array([-29.5, -30.0, -34.0]), "refr": np.array([-31.5, -33.0, -29.0])}


def test_the_arm_names_are_exactly_the_six_declared_ones():
    assert arms.OBJECTIVES == ("continued_sft", "dpo", "dpop", "dpo_hinge", "ipo", "dpo_nll")
    assert "continued_sft" not in arms.PREFERENCE_OBJECTIVES
    assert set(arms.COEFFICIENT_KEYS) == set(arms.OBJECTIVES)


def test_the_test_fixture_actually_exercises_both_hinge_branches():
    """Guard the guard: a fixture with no active hinge would silently pass dpop."""
    _, hinge = expected_terms(**NUMBERS)
    assert (hinge > 0).any() and (hinge == 0).any()


# ---------------------------------------------------------------------------
# gradients, written out independently
# ---------------------------------------------------------------------------

def test_continued_sft_is_the_site_nll_and_ignores_the_rejected_sequence():
    lc, lr, _, _ = vectors()
    per_pair, diagnostics = arms.per_pair_loss("continued_sft", policy_chosen=lc)
    expected = -NUMBERS["lc"] / arms.CORE_POSITIONS
    assert per_pair.detach().numpy() == pytest.approx(expected)
    per_pair.sum().backward()
    assert lc.grad.numpy() == pytest.approx(np.full(3, -1.0 / arms.CORE_POSITIONS))
    # lr is not an input at all, so its gradient is not "small", it does not exist.
    assert lr.grad is None
    assert diagnostics["nll_per_residue"].numpy() == pytest.approx(expected)


def test_continued_sft_refuses_a_rejected_or_reference_argument():
    """Accepting them would mean the caller paid for inference this arm cannot use."""
    lc, lr, refc, refr = vectors()
    with pytest.raises(ValueError, match="chosen scores only"):
        arms.per_pair_loss("continued_sft", policy_chosen=lc, policy_rejected=lr)
    with pytest.raises(ValueError, match="chosen scores only"):
        arms.per_pair_loss("continued_sft", policy_chosen=lc, reference_chosen=refc,
                           reference_rejected=refr)


def test_dpo_gradient_matches_the_closed_form():
    beta = 0.7
    lc, lr, refc, refr = vectors()
    per_pair, diagnostics = arms.per_pair_loss(
        "dpo", policy_chosen=lc, policy_rejected=lr, reference_chosen=refc,
        reference_rejected=refr, coefficients={"beta": beta})
    per_pair.sum().backward()
    delta, _ = expected_terms(**NUMBERS)
    coefficient = beta * sigmoid(-beta * delta)
    assert lc.grad.numpy() == pytest.approx(-coefficient)
    assert lr.grad.numpy() == pytest.approx(coefficient)
    assert diagnostics["coefficient"].numpy() == pytest.approx(coefficient)
    assert diagnostics["margin"].numpy() == pytest.approx(delta)


def test_dpop_gradient_carries_the_hinge_inside_the_sigmoid():
    beta, lam = 0.4, 2.0
    lc, lr, refc, refr = vectors()
    per_pair, diagnostics = arms.per_pair_loss(
        "dpop", policy_chosen=lc, policy_rejected=lr, reference_chosen=refc,
        reference_rejected=refr, coefficients={"beta": beta, "lambda": lam})
    per_pair.sum().backward()
    delta, hinge = expected_terms(**NUMBERS)
    modified = delta - lam * hinge
    coefficient = beta * sigmoid(-beta * modified)
    active = (NUMBERS["refc"] > NUMBERS["lc"]).astype(float)
    assert lc.grad.numpy() == pytest.approx(-coefficient * (1.0 + lam * active))
    assert lr.grad.numpy() == pytest.approx(coefficient)
    assert diagnostics["modified_margin"].numpy() == pytest.approx(modified)
    assert diagnostics["modified_coefficient"].numpy() == pytest.approx(coefficient)


def test_dpo_hinge_gradient_adds_the_penalty_outside_the_sigmoid():
    beta, lam = 0.4, 2.0
    lc, lr, refc, refr = vectors()
    per_pair, diagnostics = arms.per_pair_loss(
        "dpo_hinge", policy_chosen=lc, policy_rejected=lr, reference_chosen=refc,
        reference_rejected=refr, coefficients={"beta": beta, "lambda": lam})
    per_pair.sum().backward()
    delta, hinge = expected_terms(**NUMBERS)
    coefficient = beta * sigmoid(-beta * delta)
    active = (NUMBERS["refc"] > NUMBERS["lc"]).astype(float)
    assert lc.grad.numpy() == pytest.approx(-coefficient - lam * active)
    assert lr.grad.numpy() == pytest.approx(coefficient)
    assert diagnostics["hinge_component"].numpy() == pytest.approx(lam * hinge)


def test_ipo_gradient_is_the_squared_residual_toward_the_target_margin():
    tau = 0.5
    lc, lr, refc, refr = vectors()
    per_pair, diagnostics = arms.per_pair_loss(
        "ipo", policy_chosen=lc, policy_rejected=lr, reference_chosen=refc,
        reference_rejected=refr, coefficients={"tau": tau})
    per_pair.sum().backward()
    delta, _ = expected_terms(**NUMBERS)
    residual = delta - 1.0 / (2.0 * tau)
    assert lc.grad.numpy() == pytest.approx(2.0 * residual)
    assert lr.grad.numpy() == pytest.approx(-2.0 * residual)
    assert diagnostics["residual"].numpy() == pytest.approx(residual)
    # The target is 1 / (2 tau) on the margin alone: a larger tau is a smaller
    # target, and nothing here anchors the absolute chosen likelihood.
    assert diagnostics["target_margin"].numpy() == pytest.approx(np.full(3, 1.0))


def test_dpo_nll_gradient_adds_the_site_term_with_its_own_coefficient():
    beta, lam = 0.9, 3.0
    lc, lr, refc, refr = vectors()
    per_pair, diagnostics = arms.per_pair_loss(
        "dpo_nll", policy_chosen=lc, policy_rejected=lr, reference_chosen=refc,
        reference_rejected=refr, coefficients={"beta": beta, "lambda_site": lam})
    per_pair.sum().backward()
    delta, _ = expected_terms(**NUMBERS)
    coefficient = beta * sigmoid(-beta * delta)
    assert lc.grad.numpy() == pytest.approx(-coefficient - lam / arms.CORE_POSITIONS)
    assert lr.grad.numpy() == pytest.approx(coefficient)
    assert diagnostics["nll_component"].numpy() == pytest.approx(
        lam * (-NUMBERS["lc"] / arms.CORE_POSITIONS))


@pytest.mark.parametrize("name,coefficients", [
    ("dpop", {"beta": 0.4, "lambda": 2.0}),
    ("ipo", {"tau": 0.5}),
])
def test_finite_differences_agree_with_autograd(name, coefficients):
    """A second, independent route to the same gradient: float64 central differences."""
    step = 1e-6

    def loss_at(chosen):
        lc = torch.tensor(chosen, dtype=torch.float64)
        lr = torch.tensor(NUMBERS["lr"], dtype=torch.float64)
        refc = torch.tensor(NUMBERS["refc"], dtype=torch.float64)
        refr = torch.tensor(NUMBERS["refr"], dtype=torch.float64)
        per_pair, _ = arms.per_pair_loss(name, policy_chosen=lc, policy_rejected=lr,
                                         reference_chosen=refc, reference_rejected=refr,
                                         coefficients=coefficients)
        return float(per_pair.sum())

    numeric = np.zeros(3)
    for position in range(3):
        up, down = NUMBERS["lc"].copy(), NUMBERS["lc"].copy()
        up[position] += step
        down[position] -= step
        numeric[position] = (loss_at(up) - loss_at(down)) / (2 * step)
    lc, lr, refc, refr = vectors()
    per_pair, _ = arms.per_pair_loss(name, policy_chosen=lc, policy_rejected=lr,
                                     reference_chosen=refc, reference_rejected=refr,
                                     coefficients=coefficients)
    per_pair.sum().backward()
    assert lc.grad.numpy() == pytest.approx(numeric, abs=1e-6)


# ---------------------------------------------------------------------------
# the hinge, at and around the kink
# ---------------------------------------------------------------------------

def test_an_inactive_hinge_leaves_dpop_and_dpo_hinge_at_the_plain_dpo_gradient():
    beta, lam = 0.4, 5.0
    numbers = {"lc": np.array([-29.0]), "lr": np.array([-31.0]),
               "refc": np.array([-30.0]), "refr": np.array([-31.5])}
    grads = {}
    for name, coefficients in (("dpo", {"beta": beta}),
                               ("dpop", {"beta": beta, "lambda": lam}),
                               ("dpo_hinge", {"beta": beta, "lambda": lam})):
        lc = torch.tensor(numbers["lc"], dtype=torch.float64, requires_grad=True)
        per_pair, diagnostics = arms.per_pair_loss(
            name, policy_chosen=lc,
            policy_rejected=torch.tensor(numbers["lr"], dtype=torch.float64),
            reference_chosen=torch.tensor(numbers["refc"], dtype=torch.float64),
            reference_rejected=torch.tensor(numbers["refr"], dtype=torch.float64),
            coefficients=coefficients)
        per_pair.sum().backward()
        grads[name] = float(lc.grad[0])
        assert float(diagnostics["hinge"][0]) == 0.0
        assert float(diagnostics["hinge_active"][0]) == 0.0
    assert grads["dpop"] == pytest.approx(grads["dpo"])
    assert grads["dpo_hinge"] == pytest.approx(grads["dpo"])


def test_at_exact_equality_the_hinge_subgradient_is_zero():
    """relu'(0) = 0 in torch; pinning it keeps a later rewrite from drifting to 1."""
    beta, lam = 0.4, 5.0
    lc = torch.tensor([-30.0], dtype=torch.float64, requires_grad=True)
    per_pair, diagnostics = arms.per_pair_loss(
        "dpo_hinge", policy_chosen=lc,
        policy_rejected=torch.tensor([-31.0], dtype=torch.float64),
        reference_chosen=torch.tensor([-30.0], dtype=torch.float64),
        reference_rejected=torch.tensor([-31.0], dtype=torch.float64),
        coefficients={"beta": beta, "lambda": lam})
    per_pair.sum().backward()
    assert float(diagnostics["hinge"][0]) == 0.0
    assert float(lc.grad[0]) == pytest.approx(-beta * sigmoid(0.0))


def test_dpop_and_dpo_hinge_are_different_functions_not_aliases():
    beta, lam = 0.4, 2.0
    lc, lr, refc, refr = vectors()
    inside, _ = arms.per_pair_loss("dpop", policy_chosen=lc, policy_rejected=lr,
                                   reference_chosen=refc, reference_rejected=refr,
                                   coefficients={"beta": beta, "lambda": lam})
    inside.sum().backward()
    inside_grad = lc.grad.clone()
    lc2, lr2, refc2, refr2 = vectors()
    additive, _ = arms.per_pair_loss("dpo_hinge", policy_chosen=lc2, policy_rejected=lr2,
                                     reference_chosen=refc2, reference_rejected=refr2,
                                     coefficients={"beta": beta, "lambda": lam})
    additive.sum().backward()
    assert not torch.allclose(inside.detach(), additive.detach())
    assert not torch.allclose(inside_grad, lc2.grad)


# ---------------------------------------------------------------------------
# the frozen reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,coefficients", [
    ("dpo", {"beta": 0.3}), ("dpop", {"beta": 0.3, "lambda": 1.0}),
    ("dpo_hinge", {"beta": 0.3, "lambda": 1.0}), ("ipo", {"tau": 2.0}),
    ("dpo_nll", {"beta": 0.3, "lambda_site": 1.0})])
def test_no_gradient_reaches_a_reference_even_when_the_caller_asked_for_one(name, coefficients):
    """The detach is the mechanism. A caller that gets it wrong still cannot back-propagate."""
    lc, lr, _, _ = vectors()
    refc = torch.tensor(NUMBERS["refc"], dtype=torch.float64, requires_grad=True)
    refr = torch.tensor(NUMBERS["refr"], dtype=torch.float64, requires_grad=True)
    per_pair, _ = arms.per_pair_loss(name, policy_chosen=lc, policy_rejected=lr,
                                     reference_chosen=refc, reference_rejected=refr,
                                     coefficients=coefficients)
    per_pair.sum().backward()
    assert refc.grad is None and refr.grad is None
    assert lc.grad is not None


def test_the_reference_cache_bytes_are_unchanged_by_a_loss_and_a_backward():
    """detach() shares storage, so an in-place write is the real hazard, not the graph."""
    values = np.array(NUMBERS["refc"], dtype=np.float64)
    before = values.tobytes()
    refc = torch.as_tensor(values)
    lc, lr, _, refr = vectors()
    per_pair, _ = arms.per_pair_loss("dpop", policy_chosen=lc, policy_rejected=lr,
                                     reference_chosen=refc, reference_rejected=refr,
                                     coefficients={"beta": 0.3, "lambda": 1.0})
    per_pair.sum().backward()
    assert values.tobytes() == before


def test_the_caller_tripwire_names_the_live_reference():
    live = torch.tensor(NUMBERS["refc"], dtype=torch.float64, requires_grad=True)
    frozen = torch.tensor(NUMBERS["refr"], dtype=torch.float64)
    assert arms.require_frozen_references(frozen, None, where="batch") is True
    with pytest.raises(ValueError, match="requires_grad"):
        arms.require_frozen_references(frozen, live, where="batch")


def test_the_diagnostics_come_back_detached():
    lc, lr, refc, refr = vectors()
    _, diagnostics = arms.per_pair_loss("dpo", policy_chosen=lc, policy_rejected=lr,
                                        reference_chosen=refc, reference_rejected=refr,
                                        coefficients={"beta": 0.3})
    assert all(not value.requires_grad for value in diagnostics.values())


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,coefficients,message", [
    ("dpo", {}, "coefficient keys"),
    ("dpo", {"beta": 0.1, "lambda": 1.0}, "coefficient keys"),
    ("dpo", {"beta": 0.0}, "must be > 0"),
    ("dpo", {"beta": -1.0}, "must be > 0"),
    ("dpo", {"beta": float("nan")}, "not finite"),
    ("ipo", {"tau": 0.0}, "must be > 0"),
    ("dpop", {"beta": 0.1, "lambda": -0.5}, "must be >= 0"),
    ("dpo_nll", {"beta": 0.1, "lambda_site": -1e-9}, "must be >= 0"),
    ("dpo", {"beta": True}, "must be a real number"),
])
def test_coefficient_validation_raises_naming_the_arm(name, coefficients, message):
    with pytest.raises(ValueError, match=message) as error:
        arms.validate_coefficients(name, coefficients)
    assert name in str(error.value)


def test_a_zero_penalty_is_allowed_because_it_is_a_meaningful_setting():
    assert arms.validate_coefficients("dpop", {"beta": 0.1, "lambda": 0.0}) == {
        "beta": 0.1, "lambda": 0.0}


def test_lambda_and_lambda_site_are_separate_parameters():
    """No arm accepts the other's name, and nothing normalizes one into the other."""
    with pytest.raises(ValueError, match="coefficient keys"):
        arms.validate_coefficients("dpop", {"beta": 0.1, "lambda_site": 1.0})
    with pytest.raises(ValueError, match="coefficient keys"):
        arms.validate_coefficients("dpo_nll", {"beta": 0.1, "lambda": 1.0})


def test_arm_ids_are_stable_filename_safe_and_order_independent():
    assert arms.arm_id("continued_sft", {}) == "continued_sft"
    assert arms.arm_id("dpo", {"beta": 0.1}) == "dpo_beta0p1"
    assert arms.arm_id("dpop", {"lambda": 10.0, "beta": 0.1}) == "dpop_beta0p1_lambda10"
    assert arms.arm_id("ipo", {"tau": 0.5}) == "ipo_tau0p5"


def test_describe_records_the_formula_and_the_normalization():
    document = arms.describe("dpo_nll", {"beta": 0.1, "lambda_site": 1.0})
    assert document["core_positions"] == 10
    assert "lambda_site" in document["formula"]
    assert document["uses_rejected_sequences"] is True
    assert arms.describe("continued_sft", {})["uses_rejected_sequences"] is False


def test_unknown_arms_and_malformed_vectors_are_refused():
    lc, lr, refc, refr = vectors()
    with pytest.raises(ValueError, match="Unknown objective"):
        arms.per_pair_loss("dpo_plus", policy_chosen=lc, coefficients={})
    with pytest.raises(ValueError, match="1-D vector"):
        arms.per_pair_loss("dpo", policy_chosen=lc.reshape(3, 1), policy_rejected=lr,
                           reference_chosen=refc, reference_rejected=refr,
                           coefficients={"beta": 0.1})
    with pytest.raises(ValueError, match="share shape"):
        arms.per_pair_loss("dpo", policy_chosen=lc, policy_rejected=lr[:2],
                           reference_chosen=refc, reference_rejected=refr,
                           coefficients={"beta": 0.1})


def test_batch_loss_is_the_mean_over_pairs_for_every_arm():
    lc, lr, refc, refr = vectors()
    per_pair, _ = arms.per_pair_loss("dpo", policy_chosen=lc, policy_rejected=lr,
                                     reference_chosen=refc, reference_rejected=refr,
                                     coefficients={"beta": 0.2})
    mean, _ = arms.batch_loss("dpo", policy_chosen=lc, policy_rejected=lr,
                              reference_chosen=refc, reference_rejected=refr,
                              coefficients={"beta": 0.2})
    # Detached before the conversion: these tensors are live, and a test should not
    # be the thing that drags a graph into a Python float.
    assert float(mean.detach()) == pytest.approx(float(per_pair.detach().mean()))


# ---------------------------------------------------------------------------
# what a logged batch says about the rejected side
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,coefficients", [
    ("dpo", {"beta": 0.3}), ("dpop", {"beta": 0.3, "lambda": 1.0}),
    ("dpo_hinge", {"beta": 0.3, "lambda": 1.0}), ("ipo", {"tau": 2.0}),
    ("dpo_nll", {"beta": 0.3, "lambda_site": 1.0})])
def test_every_preference_arm_logs_both_sides_and_the_margin_sign(name, coefficients):
    """A chosen-only log cannot tell a rising chosen side from a collapsing rejected one."""
    lc, lr, refc, refr = vectors()
    _, diagnostics = arms.per_pair_loss(name, policy_chosen=lc, policy_rejected=lr,
                                        reference_chosen=refc, reference_rejected=refr,
                                        coefficients=coefficients)
    assert diagnostics["rejected_log_probability"].numpy() == pytest.approx(NUMBERS["lr"])
    assert diagnostics["chosen_log_probability"].numpy() == pytest.approx(NUMBERS["lc"])
    assert diagnostics["rejected_change"].numpy() == pytest.approx(
        NUMBERS["lr"] - NUMBERS["refr"])
    assert diagnostics["chosen_change"].numpy() == pytest.approx(NUMBERS["lc"] - NUMBERS["refc"])
    assert diagnostics["chosen_nll_per_residue"].numpy() == pytest.approx(
        -NUMBERS["lc"] / arms.CORE_POSITIONS)
    assert diagnostics["rejected_nll_per_residue"].numpy() == pytest.approx(
        -NUMBERS["lr"] / arms.CORE_POSITIONS)
    delta, _ = expected_terms(**NUMBERS)
    assert diagnostics["sign_correct"].numpy() == pytest.approx((delta > 0).astype(float))
    assert all(not value.requires_grad for value in diagnostics.values())


def test_the_sign_indicator_is_the_implicit_reward_accuracy_before_reduction():
    """Its mean is the sign accuracy; it is left as an indicator so the logger reduces it."""
    lc = torch.tensor([-30.0, -30.0], dtype=torch.float64, requires_grad=True)
    lr = torch.tensor([-31.0, -20.0], dtype=torch.float64)
    refc = torch.tensor([-30.0, -30.0], dtype=torch.float64)
    refr = torch.tensor([-31.0, -31.0], dtype=torch.float64)
    _, diagnostics = arms.per_pair_loss("dpo", policy_chosen=lc, policy_rejected=lr,
                                        reference_chosen=refc, reference_rejected=refr,
                                        coefficients={"beta": 0.5})
    # Pair 0 sits exactly at Delta = 0, which is not a positive margin; pair 1 is negative.
    assert diagnostics["margin"].numpy() == pytest.approx([0.0, -11.0])
    assert diagnostics["sign_correct"].numpy() == pytest.approx([0.0, 0.0])
    assert float(diagnostics["sign_correct"].mean()) == 0.0


def test_continued_sft_logs_its_chosen_nll_and_no_rejected_field():
    lc, _, _, _ = vectors()
    _, diagnostics = arms.per_pair_loss("continued_sft", policy_chosen=lc)
    assert diagnostics["chosen_nll_per_residue"].numpy() == pytest.approx(
        -NUMBERS["lc"] / arms.CORE_POSITIONS)
    assert not any(key.startswith("rejected") for key in diagnostics), \
        "this arm performs no rejected inference, so it has no rejected number to log"
