"""The six exactly named continuation objectives, their coefficients and diagnostics.

Every arm scores a pair with the *same* quantity the original campaign used: the
SUM of the ten core-position log probabilities under the 20-way renormalized
categorical (:data:`~.her2_preferences.PROBABILITY_CONVENTION`). Nothing here
re-derives a probability; the caller hands in ``lc``/``lr`` from the live policy
and ``refc``/``refr`` from the frozen reference, and this module decides only what
is done with those four numbers.

Fixed here rather than left to convention:

* **The reference is detached inside every loss, unconditionally.** ``.detach()``
  is called on ``refc``/``refr`` before any arithmetic, so a caller that hands in
  a live tensor still cannot back-propagate into the reference. The separate
  :func:`require_frozen_references` tripwire is for the *trainer* to call on the
  tensors it built; it is deliberately **not** the mechanism, because a mechanism
  that depends on the caller having got it right is not a mechanism.
* **``dpop`` names the inside-sigmoid hinge and nothing else.** The additive
  variant is :func:`dpo_hinge`. They are different functions with different
  gradients, there is no alias between them, and no "equivalent up to lambda"
  claim is made anywhere.
* **Coefficients are separate parameters with separate grids.** ``lambda`` and
  ``lambda_site`` are never normalized into one another, ``beta``/``tau`` must be
  finite and strictly positive, ``lambda``/``lambda_site`` finite and ``>= 0``,
  and an extra or missing key raises naming the arm.
* **``/10`` is a named constant.** :data:`CORE_POSITIONS` is the ten editable
  positions; a literal 10 appears nowhere, so a scaffold with a different core
  width cannot silently keep the old normalizer.
* **Per-pair diagnostics come back detached.** They are vectors, not reductions:
  the quantiles are taken by the logger, outside the charged training segment.
  Both sides are reported -- chosen *and* rejected log probabilities, their NLL
  per residue, the implicit-reward margin and the per-pair sign of that margin --
  so a logged batch says what happened to the rejected side too.

What a coefficient does **not** mean, stated here because every one of these is
an easy thing to write and none of them is true of these losses:

* A larger ``beta`` is neither a likelihood floor nor a hard KL bound. It scales
  the implicit-reward margin inside the sigmoid; it constrains nothing about the
  absolute chosen log probability and imposes no bound on KL to the reference.
* ``ipo`` has no absolute anchor either. ``1 / (2 tau)`` is a target for the
  *difference* ``Delta``; a policy can hit it exactly while both sides fall.
* ``lambda_site`` in ``dpo_nll`` scales a per-residue NLL term that is already
  divided by :data:`CORE_POSITIONS`. That division is a normalization of the site
  term, not an "SFT:DPO" scaling identity; the two terms are on different scales
  and no ratio between them is implied by their coefficients.
* Every arm reduces by a mean over pairs, which makes the *reduction* comparable
  and nothing else. Two arms at the same batch size and learning rate are not
  thereby equally strong: the gradient magnitudes differ per arm and per
  coefficient, so a common batch mean is not a common objective strength.

Gradients, for the record (the tests re-derive these independently rather than
importing them). With ``c = beta * sigmoid(-beta * Delta)`` and
``c' = beta * sigmoid(-beta * M)`` where ``M = Delta - lambda * h``:

===============  ==========================================  ==================
arm              dL/d lc                                     dL/d lr
===============  ==========================================  ==================
continued_sft    ``-1 / 10``                                 ``0`` exactly
dpo              ``-c``                                      ``+c``
dpop             ``-c' * (1 + lambda)`` active, else ``-c'``  ``+c'``
dpo_hinge        ``-c - lambda * 1{refc > lc}``               ``+c``
ipo              ``2 * (Delta - 1 / (2 tau))``                ``-2 * (...)``
dpo_nll          ``-c - lambda_site / 10``                    ``+c``
===============  ==========================================  ==================

``h = relu(refc - lc)`` uses torch's relu convention at the kink, so the
subgradient at ``refc == lc`` is 0.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .her2_data import CORE_LENGTH
from .her2_preferences import PROBABILITY_CONVENTION
from .her2_runtime import require

#: The ten editable positions. The ``/10`` in ``continued_sft`` and in the
#: ``dpo_nll`` site term is this constant, never a literal.
CORE_POSITIONS = CORE_LENGTH

#: Exact arm names. These strings appear in configs, artifacts and reports; a
#: rename is a new experiment, not a refactor.
OBJECTIVES = ("continued_sft", "dpo", "dpop", "dpo_hinge", "ipo", "dpo_nll")
#: The arms that consume a rejected sequence. ``continued_sft`` does not, and the
#: trainer must not perform rejected inference for it just to fill a log column.
PREFERENCE_OBJECTIVES = tuple(name for name in OBJECTIVES if name != "continued_sft")

#: The exact coefficient key set each arm declares. Extra keys are as much of an
#: error as missing ones: a ``lambda`` handed to ``dpo`` means the caller believes
#: it is running an arm that it is not.
COEFFICIENT_KEYS = {
    "continued_sft": (),
    "dpo": ("beta",),
    "dpop": ("beta", "lambda"),
    "dpo_hinge": ("beta", "lambda"),
    "ipo": ("tau",),
    "dpo_nll": ("beta", "lambda_site"),
}
#: Strictly positive: a zero or negative temperature is not a weaker setting, it
#: is a different (and degenerate) objective.
POSITIVE_COEFFICIENTS = frozenset({"beta", "tau"})
#: Zero is meaningful here -- it recovers the unpenalized arm -- so ``>= 0``.
NONNEGATIVE_COEFFICIENTS = frozenset({"lambda", "lambda_site"})

#: The scoring contract every arm assumes. Recorded beside results so a cached
#: reference produced under another convention cannot be reused silently.
SCORING_CONVENTION = PROBABILITY_CONVENTION


def validate_coefficients(name, coefficients):
    """Return the arm's coefficients as floats, or raise naming the arm.

    Missing, extra, non-numeric, non-finite and out-of-range values all stop
    here. The error text carries the arm name because these are read out of a
    grid file where "beta" alone does not say which row was wrong.
    """
    require(name in COEFFICIENT_KEYS, f"Unknown objective {name!r}; expected one of {OBJECTIVES}")
    expected = set(COEFFICIENT_KEYS[name])
    supplied = dict(coefficients or {})
    actual = set(supplied)
    require(actual == expected,
            f"{name}: coefficient keys {sorted(actual)} != the declared {sorted(expected)}. "
            f"Missing: {sorted(expected - actual)}; unexpected: {sorted(actual - expected)}")
    out = {}
    for key in sorted(expected):
        value = supplied[key]
        require(not isinstance(value, bool) and isinstance(value, (int, float)),
                f"{name}: coefficient {key!r} must be a real number, got {type(value).__name__}")
        value = float(value)
        require(math.isfinite(value), f"{name}: coefficient {key!r} is not finite ({value!r})")
        if key in POSITIVE_COEFFICIENTS:
            require(value > 0, f"{name}: coefficient {key!r} must be > 0, got {value!r}")
        elif key in NONNEGATIVE_COEFFICIENTS:
            require(value >= 0, f"{name}: coefficient {key!r} must be >= 0, got {value!r}")
        out[key] = value
    return out


def arm_id(name, coefficients):
    """The stable identifier an arm carries through every artifact and table."""
    resolved = validate_coefficients(name, coefficients)
    if not resolved:
        return name
    parts = "_".join(f"{key}{_format_coefficient(resolved[key])}" for key in sorted(resolved))
    return f"{name}_{parts}"


def _format_coefficient(value):
    """A short, exact, filename-safe rendering: 0.1 -> ``0p1``, 10.0 -> ``10``."""
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "p").replace("-", "m")


def require_frozen_references(*tensors, where):
    """Caller tripwire: the trainer's reference tensors must not carry gradient.

    The losses detach regardless, so this cannot be the mechanism -- it is a
    second, earlier signal that whoever built the batch handed in a live tensor
    by mistake. The trajectory runner calls it once per batch; the loss functions
    deliberately do not, so that a test may hand a ``requires_grad`` reference
    straight into every arm and observe that no gradient reaches it.
    """
    for position, tensor in enumerate(tensors):
        if tensor is None:
            continue
        require(not tensor.requires_grad,
                f"{where}: reference tensor {position} carries requires_grad. The reference is the "
                "frozen parent; a live one turns the objective into something else.")
    return True


# ---------------------------------------------------------------------------
# shared terms
# ---------------------------------------------------------------------------

def _check_vector(tensor, name, *, like=None):
    require(isinstance(tensor, torch.Tensor), f"{name} must be a torch tensor")
    require(tensor.is_floating_point(), f"{name} must be floating point")
    require(tensor.ndim == 1 and tensor.numel() > 0,
            f"{name} must be a nonempty 1-D vector, one value per pair; got shape "
            f"{tuple(tensor.shape)}")
    if like is not None:
        require(tensor.shape == like.shape and tensor.device == like.device
                and tensor.dtype == like.dtype,
                f"{name} must share shape, dtype and device with the policy chosen scores")
    return tensor


def preference_terms(policy_chosen, policy_rejected, reference_chosen, reference_rejected):
    """``(Delta, h, refc, refr)`` with the references detached before any arithmetic.

    ``Delta = (lc - lr) - (refc - refr)`` is the implicit-reward margin; ``h =
    relu(refc - lc)`` is the *positive part of the chosen-side drop*, i.e. it is
    active exactly when the policy has made the chosen sequence less likely than
    the parent did.
    """
    _check_vector(policy_chosen, "policy_chosen")
    _check_vector(policy_rejected, "policy_rejected", like=policy_chosen)
    _check_vector(reference_chosen, "reference_chosen", like=policy_chosen)
    _check_vector(reference_rejected, "reference_rejected", like=policy_chosen)
    refc = reference_chosen.detach()
    refr = reference_rejected.detach()
    require(bool(torch.isfinite(refc).all()) and bool(torch.isfinite(refr).all()),
            "Nonfinite reference log probability")
    delta = (policy_chosen - policy_rejected) - (refc - refr)
    hinge = torch.relu(refc - policy_chosen)
    return delta, hinge, refc, refr


def _dpo_coefficient(delta, beta):
    """The actual per-pair DPO coefficient ``beta * sigmoid(-beta * Delta)``.

    This is the number the gradient is actually scaled by, and it is *not*
    recoverable from the mean margin: sigma is nonlinear, so
    ``E[beta sigma(-beta Delta)] != beta sigma(-beta E[Delta])``.
    """
    return beta * torch.sigmoid(-beta * delta)


# ---------------------------------------------------------------------------
# the six arms
# ---------------------------------------------------------------------------

def continued_sft_per_pair(policy_chosen):
    """``-lc / 10``: the initial-SFT objective, per chosen sequence.

    ``lr`` is not an argument. Continued SFT does not look at a rejected
    sequence, so there is no rejected forward pass to charge to its budget and no
    way for one to creep in through a diagnostic.
    """
    _check_vector(policy_chosen, "policy_chosen")
    per_pair = -policy_chosen / CORE_POSITIONS
    diagnostics = {"chosen_log_probability": policy_chosen.detach(),
                   "nll_per_residue": per_pair.detach(),
                   "chosen_nll_per_residue": per_pair.detach()}
    return per_pair, diagnostics


def dpo_per_pair(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta):
    """``softplus(-beta * Delta)``, i.e. ``-log sigmoid(beta * Delta)``."""
    delta, hinge, refc, refr = preference_terms(policy_chosen, policy_rejected, reference_chosen,
                                                reference_rejected)
    per_pair = F.softplus(-beta * delta)
    return per_pair, _preference_diagnostics(per_pair, delta, hinge, policy_chosen,
                                             policy_rejected, refc, refr, beta,
                                             preference=per_pair)


def dpop_per_pair(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta,
                  **coefficients):
    """``softplus(-beta * (Delta - lambda * h))`` -- the hinge INSIDE the sigmoid.

    This is the only arm that carries the name ``dpop``. The penalty moves the
    margin the sigmoid sees, so the whole gradient is scaled by
    ``c' = beta * sigmoid(-beta * M)`` evaluated at the **modified** margin ``M``:
    when ``beta * M`` is large and positive the penalty term saturates along with
    the preference term. The additive variant in :func:`dpo_hinge` is different in
    kind, not in size: its ``lambda`` pull is a constant whenever the hinge is
    active, independent of the sigmoid.

    Neither arm is a likelihood floor. Both penalize the chosen-side drop relative
    to the parent, and both can still lose chosen probability mass on a shared
    update -- the penalty is one term of a gradient, not a constraint. That is
    what the separate parent-relative gate is for.
    """
    lam = float(coefficients.pop("lambda"))
    require(not coefficients, f"dpop: unexpected coefficients {sorted(coefficients)}")
    delta, hinge, refc, refr = preference_terms(policy_chosen, policy_rejected, reference_chosen,
                                                reference_rejected)
    modified = delta - lam * hinge
    per_pair = F.softplus(-beta * modified)
    diagnostics = _preference_diagnostics(per_pair, delta, hinge, policy_chosen, policy_rejected,
                                          refc, refr, beta, preference=per_pair)
    diagnostics["modified_margin"] = modified.detach()
    diagnostics["modified_coefficient"] = _dpo_coefficient(modified.detach(), beta)
    return per_pair, diagnostics


def dpo_hinge_per_pair(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *,
                       beta, **coefficients):
    """``softplus(-beta * Delta) + lambda * h`` -- the ADDITIVE chosen-drop penalty.

    The penalty is outside the sigmoid, so while the hinge is active its gradient
    contribution is exactly ``-lambda`` on ``lc`` whatever the margin does. It does
    not saturate with the preference term, and it does not bound the chosen
    likelihood either: a shared update can still reduce it.
    """
    lam = float(coefficients.pop("lambda"))
    require(not coefficients, f"dpo_hinge: unexpected coefficients {sorted(coefficients)}")
    delta, hinge, refc, refr = preference_terms(policy_chosen, policy_rejected, reference_chosen,
                                                reference_rejected)
    preference = F.softplus(-beta * delta)
    penalty = lam * hinge
    per_pair = preference + penalty
    diagnostics = _preference_diagnostics(per_pair, delta, hinge, policy_chosen, policy_rejected,
                                          refc, refr, beta, preference=preference)
    diagnostics["hinge_component"] = penalty.detach()
    return per_pair, diagnostics


def ipo_per_pair(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, tau):
    """``(Delta - 1 / (2 tau))**2`` -- a squared loss toward a fixed target margin.

    The target is ``1 / (2 tau)``, so a larger ``tau`` names a smaller target
    margin. The target is on the implicit-reward *difference* only: it anchors
    nothing absolute, and a policy sitting exactly on it can still have moved both
    ``lc`` and ``lr`` arbitrarily far from the parent.
    """
    delta, hinge, refc, refr = preference_terms(policy_chosen, policy_rejected, reference_chosen,
                                                reference_rejected)
    target = 1.0 / (2.0 * tau)
    residual = delta - target
    per_pair = residual ** 2
    diagnostics = _preference_diagnostics(per_pair, delta, hinge, policy_chosen, policy_rejected,
                                          refc, refr, beta=None, preference=per_pair)
    diagnostics["target_margin"] = torch.full_like(delta.detach(), float(target))
    diagnostics["residual"] = residual.detach()
    return per_pair, diagnostics


def dpo_nll_per_pair(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta,
                     **coefficients):
    """``softplus(-beta * Delta) + lambda_site * (-lc / 10)`` -- DPO plus the SFT term.

    ``lambda_site`` weights a term that is already per residue (``/10``). It is a
    coefficient on that term and not a declared ratio between "the SFT objective"
    and "the DPO objective": the two terms have different scales and different
    gradients, and ``lambda_site = 1`` does not make them equally weighted.
    """
    lam = float(coefficients.pop("lambda_site"))
    require(not coefficients, f"dpo_nll: unexpected coefficients {sorted(coefficients)}")
    delta, hinge, refc, refr = preference_terms(policy_chosen, policy_rejected, reference_chosen,
                                                reference_rejected)
    preference = F.softplus(-beta * delta)
    nll = lam * (-policy_chosen / CORE_POSITIONS)
    per_pair = preference + nll
    diagnostics = _preference_diagnostics(per_pair, delta, hinge, policy_chosen, policy_rejected,
                                          refc, refr, beta, preference=preference)
    diagnostics["nll_component"] = nll.detach()
    return per_pair, diagnostics


def _preference_diagnostics(per_pair, delta, hinge, policy_chosen, policy_rejected, refc, refr,
                            beta, *, preference):
    """Detached per-pair vectors for the logger. No reduction happens here.

    Both sides are here on purpose, and the reason is narrower than "the original
    campaign could not tell them apart". It could: its histories stored the batch
    **mean** of the chosen and of the rejected log probability, so which side moved
    is recoverable from them. What a pair of means cannot show is the pair-level
    joint -- how many pairs carried the margin, whether the chosen drop was a broad
    shift or a tail, where the hinge was active -- and that is what these per-pair
    vectors are for. ``sign_correct`` is the per-pair indicator ``Delta > 0``; its
    mean is the implicit-reward sign accuracy, and it is left as an indicator so
    the logger reduces it rather than this module inventing a reduction.
    """
    detached_delta = delta.detach()
    chosen = policy_chosen.detach()
    rejected = policy_rejected.detach()
    document = {"loss": per_pair.detach(),
                "preference_component": preference.detach(),
                "margin": detached_delta,
                "hinge": hinge.detach(),
                "hinge_active": (hinge.detach() > 0).to(detached_delta.dtype),
                "chosen_change": chosen - refc,
                "rejected_change": rejected - refr,
                "chosen_log_probability": chosen,
                "rejected_log_probability": rejected,
                "chosen_nll_per_residue": -chosen / CORE_POSITIONS,
                "rejected_nll_per_residue": -rejected / CORE_POSITIONS,
                "sign_correct": (detached_delta > 0).to(detached_delta.dtype)}
    if beta is not None:
        document["coefficient"] = _dpo_coefficient(detached_delta, beta)
    return document


#: One dispatch table, so a caller cannot reach an arm this module does not know.
_IMPLEMENTATIONS = {"dpo": dpo_per_pair, "dpop": dpop_per_pair, "dpo_hinge": dpo_hinge_per_pair,
                    "ipo": ipo_per_pair, "dpo_nll": dpo_nll_per_pair}


def per_pair_loss(name, *, policy_chosen, policy_rejected=None, reference_chosen=None,
                  reference_rejected=None, coefficients=None):
    """``(per_pair_loss, detached diagnostics)`` for one named arm.

    ``continued_sft`` takes ``policy_chosen`` alone; supplying a rejected or
    reference tensor to it raises, because it would mean the caller paid for an
    inference pass this arm does not need and cannot use.
    """
    resolved = validate_coefficients(name, coefficients)
    if name == "continued_sft":
        require(policy_rejected is None and reference_chosen is None
                and reference_rejected is None,
                "continued_sft takes chosen scores only; it performs no rejected inference and "
                "consumes no reference, so passing one would misreport what its budget bought")
        return continued_sft_per_pair(policy_chosen)
    require(policy_rejected is not None and reference_chosen is not None
            and reference_rejected is not None,
            f"{name} needs policy_rejected, reference_chosen and reference_rejected")
    return _IMPLEMENTATIONS[name](policy_chosen, policy_rejected, reference_chosen,
                                  reference_rejected, **resolved)


def batch_loss(name, *, policy_chosen, policy_rejected=None, reference_chosen=None,
               reference_rejected=None, coefficients=None):
    """The mean over the batch, plus the same per-pair diagnostics.

    The reduction is a mean over pairs for every arm, so the batch size does not
    rescale the loss and two batches of different sizes are comparable **within**
    an arm. It does not make two arms comparable: their gradient magnitudes differ
    by arm and by coefficient, so a shared mean reduction is not a shared
    objective strength and the same learning rate does not mean the same step.
    """
    per_pair, diagnostics = per_pair_loss(
        name, policy_chosen=policy_chosen, policy_rejected=policy_rejected,
        reference_chosen=reference_chosen, reference_rejected=reference_rejected,
        coefficients=coefficients)
    return per_pair.mean(), diagnostics


def describe(name, coefficients):
    """The arm's declared identity: name, formula, coefficients, normalization."""
    resolved = validate_coefficients(name, coefficients)
    formulas = {
        "continued_sft": "-lc / 10",
        "dpo": "softplus(-beta * Delta)",
        "dpop": "softplus(-beta * (Delta - lambda * h))",
        "dpo_hinge": "softplus(-beta * Delta) + lambda * h",
        "ipo": "(Delta - 1 / (2 * tau)) ** 2",
        "dpo_nll": "softplus(-beta * Delta) + lambda_site * (-lc / 10)"}
    return {"objective": name, "arm_id": arm_id(name, resolved), "coefficients": resolved,
            "formula": formulas[name],
            "margin": "Delta = (lc - lr) - (refc - refr)",
            "hinge": "h = relu(refc - lc)",
            "core_positions": CORE_POSITIONS,
            "probability_convention": SCORING_CONVENTION,
            "reference": "frozen parent, detached inside the loss",
            "uses_rejected_sequences": name != "continued_sft"}
