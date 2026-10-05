"""Training objectives: reconstruction, absolute effect fitting, preservation, planning.

Four objectives and two regularisers, with the arithmetic that distinguishes them
written out rather than implied.

**Absolute versus reference-relative effect fitting.** The default fits the
decoder's *absolute* conditional log odds to a scaled measured effect:
``log q(alt) - log q(ref) -> kappa * dpKD``. The plan's worked example is the
reason, and it is reproduced as a test: with ``kappa = 2`` and an edit worth
``-0.5`` pKD without a partner and ``+0.4`` with it, the target odds are
``exp(-1) = 0.368`` and ``exp(0.8) = 2.226``. No fixed preference for that edit
satisfies both, which is what "the effect is contextual" means operationally.
Reference-relative fitting is the **ablation**, not the default: multiplying a
biased reference odds of ``0.1`` by ``2.226`` gives ``0.223``, so the better edit
stays less probable than its alternative. :func:`effect_targets` computes both and
:func:`fit_absolute_log_odds` implements the default.

**Preservation.** The forward KL here is the same functional as
``her2_replay.replay_term`` -- mean over replay rows of a sum over positions and
residues of ``p * (log p - log q)``, with no division by the position count or the
residue count -- generalised to a **variable** number of positions, because the
inherited one asserts exactly ten. A test checks the two agree on a ``(B, 10, 20)``
block, so the generalisation is verified against the tested implementation instead
of replacing it on trust.

**Sinkhorn.** Debiased:
``S(a,b) = OT(a,b) - (OT(a,a) + OT(b,b)) / 2``, which is what makes it zero for
identical distributions within solver tolerance. The undebiased entropic OT is not,
and an implementation that returned it would pass a "gradient exists" test and fail
the identity check I13 asks for.

**Planner weighting.** Confidence-derived weights are **detached**, ordinary
reconstruction is retained beside them, and at zero planner weight the loss and the
gradients reproduce ordinary training exactly. That last property is the one I19
asserts, and it is the reason the weight enters multiplicatively on a normalised
loss rather than replacing the normalisation.
"""
from __future__ import annotations

import math

import torch

from .her2_runtime import require

#: Default scale on measured effects in the absolute-log-odds objective.
DEFAULT_KAPPA = 2.0
#: Sinkhorn solver settings. The tolerance is what "zero within solver tolerance"
#: means in I13, so it is a declared constant rather than a call-site default.
SINKHORN_EPSILON = 0.05
SINKHORN_ITERATIONS = 200
SINKHORN_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# reconstruction and whole-window likelihood
# ---------------------------------------------------------------------------

def masked_nll(log_probabilities, targets, mask):
    """Mean negative log likelihood over the ACTIVE targets only.

    Normalised by the active-target count, not by the padded width. Dividing by the
    width would make a batch of short windows look better than a batch of long ones
    for no scientific reason, and it silently reweights sources by their edit-map
    size.
    """
    require(log_probabilities.dim() == 3,
            f"Expected (B, k, residues) log probabilities, got {tuple(log_probabilities.shape)}")
    targets = targets.to(log_probabilities.device).long()
    mask = mask.to(log_probabilities.device).to(log_probabilities.dtype)
    require(targets.shape == mask.shape == log_probabilities.shape[:2],
            "Targets, mask and log probabilities are not aligned")
    active = mask.sum()
    picked = log_probabilities.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    if float(active) == 0.0:
        return picked.sum() * 0.0, {"active_targets": 0}
    loss = -(picked * mask).sum() / active
    with torch.no_grad():
        diagnostics = {"active_targets": int(active),
                       "mean_nll_per_target": float(loss.detach()),
                       "normalisation": "sum over active targets / active target count"}
    return loss, diagnostics


def require_normalised(log_probabilities, *, label, tolerance=1e-5):
    """Every conditional must sum to one over its support. Reports the worst error."""
    total = log_probabilities.exp().sum(dim=-1)
    worst = float((total - 1.0).abs().max())
    require(worst <= float(tolerance),
            f"{label}: the worst conditional deviates from summing to one by {worst:.3g}, above "
            f"{tolerance:.1e}. A distribution that does not normalise is a different "
            "distribution, not a rounding difference.")
    return {"max_abs_sum_error": worst, "support": int(log_probabilities.shape[-1]),
            "tolerance": float(tolerance)}


def sequential_two_site(model_step, *, residues=20):
    """The declared 400-way coupled distribution for a two-site window.

    ``model_step(prefix)`` returns the conditional over ``residues`` for the next
    site given the residues already revealed. The joint is the product along the
    declared order, which is a genuine distribution over ``residues**2`` outcomes.

    Two independent one-site marginals also normalise -- that is exactly why an
    "it sums to one" check is not enough, and why :func:`dependence_report` below
    measures whether revealing the first residue moves the second conditional.
    """
    first = model_step(())
    require(first.shape[-1] == int(residues), "The first conditional has the wrong support")
    rows = []
    for index in range(int(residues)):
        second = model_step((index,))
        require(second.shape[-1] == int(residues), "The second conditional has the wrong support")
        rows.append(first[..., index].unsqueeze(-1) + second)
    joint = torch.cat(rows, dim=-1)
    return joint


def dependence_report(model_step, *, residues=20):
    """Does revealing the first residue change the second conditional? A number.

    An accidentally independent model normalises perfectly and cannot represent the
    coupling the campaign exists to teach. The reported quantity is the maximum
    total-variation distance between the second conditional under two different
    first residues; zero means independent.
    """
    conditionals = [model_step((index,)).exp() for index in range(int(residues))]
    worst, pair = 0.0, None
    for i in range(len(conditionals)):
        for j in range(i + 1, len(conditionals)):
            distance = float(0.5 * (conditionals[i] - conditionals[j]).abs().sum(dim=-1).max())
            if distance > worst:
                worst, pair = distance, (i, j)
    return {"max_total_variation": worst, "worst_pair": pair,
            "independent": worst == 0.0,
            "reading": ("zero means the two sites are modelled independently. Such a model still "
                        "normalises over 400 outcomes and still cannot represent an interaction, "
                        "which is why normalisation alone is not the check.")}


# ---------------------------------------------------------------------------
# measured effect fitting
# ---------------------------------------------------------------------------

def effect_targets(delta_pkd, *, kappa=DEFAULT_KAPPA, reference_odds=None):
    """The target odds for one measured effect, absolute and reference-relative.

    Returns both so the ablation is computed by the same code as the default and
    the difference between them is visible in one record.
    """
    delta = float(delta_pkd)
    absolute = math.exp(float(kappa) * delta)
    record = {"delta_pkd": delta, "kappa": float(kappa),
              "target_log_odds": float(kappa) * delta, "target_odds": absolute}
    if reference_odds is not None:
        record["reference_odds"] = float(reference_odds)
        record["reference_relative_odds"] = float(reference_odds) * absolute
        record["reference_relative_note"] = (
            "reference-relative fitting multiplies the reference odds by the target ratio. A "
            "biased reference of 0.1 times a target of 2.226 is 0.223, so the better edit stays "
            "less probable. This is the ABLATION.")
    return record


def fit_absolute_log_odds(policy_log_alt, policy_log_ref, delta_pkd, *, kappa=DEFAULT_KAPPA,
                          weights=None):
    """Squared error between the decoder's absolute log odds and ``kappa * dpKD``.

    ``policy_log_alt`` / ``policy_log_ref`` are the student's conditional log
    probabilities of the two alternatives at the same masked background, so their
    difference is the absolute log odds of the edit.
    """
    delta = torch.as_tensor(delta_pkd, dtype=policy_log_alt.dtype,
                            device=policy_log_alt.device)
    target = float(kappa) * delta
    predicted = policy_log_alt - policy_log_ref
    residual = predicted - target
    if weights is None:
        loss = (residual ** 2).mean()
    else:
        weight = torch.as_tensor(weights, dtype=residual.dtype, device=residual.device)
        total = torch.clamp(weight.sum(), min=1e-12)
        loss = ((residual ** 2) * weight).sum() / total
    with torch.no_grad():
        diagnostics = {"pairs": int(residual.shape[0]),
                       "mean_predicted_log_odds": float(predicted.detach().mean()),
                       "mean_target_log_odds": float(target.detach().mean()),
                       "mean_abs_residual": float(residual.detach().abs().mean()),
                       "sign_agreement": float((torch.sign(predicted.detach())
                                                == torch.sign(target.detach())).double().mean()),
                       "kappa": float(kappa),
                       "mode": "absolute"}
    return loss, diagnostics


def fit_reference_relative_log_odds(policy_log_alt, policy_log_ref, reference_log_alt,
                                    reference_log_ref, delta_pkd, *, kappa=DEFAULT_KAPPA):
    """The declared ablation. Frozen reference; the residual is measured, not assumed."""
    require(not reference_log_alt.requires_grad and not reference_log_ref.requires_grad,
            "The reference must be frozen. A live one turns this into a joint optimisation of "
            "both sides and the recorded 'reference-relative' label becomes false.")
    delta = torch.as_tensor(delta_pkd, dtype=policy_log_alt.dtype,
                            device=policy_log_alt.device)
    predicted = (policy_log_alt - policy_log_ref) - (reference_log_alt - reference_log_ref)
    residual = predicted - float(kappa) * delta
    loss = (residual ** 2).mean()
    with torch.no_grad():
        diagnostics = {"pairs": int(residual.shape[0]), "mode": "reference_relative_ablation",
                       "mean_abs_residual": float(residual.detach().abs().mean()),
                       "kappa": float(kappa),
                       "why_ablation": ("a biased reference is carried through multiplicatively, "
                                        "so an edit the reference disfavours can stay disfavoured "
                                        "after the target ratio is applied")}
    return loss, diagnostics


def interaction_support(quartet):
    """Can an interaction be *stated* from these four measurements? With the reason.

    A numerical interaction claim needs the background, both single edits and the
    double, all in the same assay. An incomplete set supports contextual ordering
    and nothing more, and this function is what stops the incomplete case from
    being reported as epistasis.
    """
    needed = ("background", "single_a", "single_b", "double")
    present = sorted(key for key in needed if quartet.get(key) is not None)
    if len(present) < len(needed):
        return {"complete": False, "present": present,
                "missing": sorted(set(needed) - set(present)),
                "supports": "contextual ordering only",
                "does_not_support": "a numerical interaction or physical epistasis claim"}
    assays = {quartet[key]["assay"] for key in needed}
    if len(assays) > 1:
        return {"complete": False, "present": present, "assays": sorted(assays),
                "supports": "contextual ordering only",
                "does_not_support": "an interaction across two assay scales"}
    background = float(quartet["background"]["value"])
    a = float(quartet["single_a"]["value"]) - background
    b = float(quartet["single_b"]["value"]) - background
    double = float(quartet["double"]["value"]) - background
    return {"complete": True, "assay": sorted(assays)[0], "background": background,
            "effect_a": a, "effect_b": b, "effect_double": double,
            "interaction": double - (a + b),
            "a_without_b": a, "a_with_b": double - b,
            "supports": "a numerical interaction on the declared quantitative scale"}


# ---------------------------------------------------------------------------
# preservation
# ---------------------------------------------------------------------------

def forward_kl(teacher_probabilities, teacher_log_probabilities, student_log_probabilities, *,
               label="preservation", mask=None):
    """``mean_rows sum_positions sum_residues p * (log p - log q)``, variable positions.

    The same functional as ``her2_replay.replay_term`` with the ten-position
    assertion lifted. Nothing divides by the position count or the residue count:
    the reduction is a mean over rows of a quantity already summed over positions,
    which is what makes it an estimator of ``KL(reference || policy)`` over the
    window.

    The masked positions convention: ``p == 0`` contributes exactly zero and no
    epsilon is added, and the masking happens **before** the multiplication so a
    ``-inf`` teacher log at an underflowed residue cannot produce ``0 * -inf`` on
    the forward pass or ``0 * NaN`` on the backward one.
    """
    for name, tensor in (("probabilities", teacher_probabilities),
                         ("log probabilities", teacher_log_probabilities)):
        require(isinstance(tensor, torch.Tensor), f"{label}: teacher {name} must be a tensor")
        require(not tensor.requires_grad,
                f"{label}: teacher {name} carries requires_grad; the reference is frozen")
        require(bool(torch.isfinite(tensor).all()), f"{label}: teacher {name} is not finite")
    require(teacher_probabilities.shape == student_log_probabilities.shape,
            f"{label}: teacher {tuple(teacher_probabilities.shape)} and student "
            f"{tuple(student_log_probabilities.shape)} are not aligned")
    sums = teacher_probabilities.double().sum(dim=-1)
    worst = float((sums - 1.0).abs().max())
    require(worst <= 1e-5,
            f"{label}: a teacher row deviates from summing to one by {worst:.3g}")
    probabilities = teacher_probabilities.to(student_log_probabilities.dtype)
    teacher_log = teacher_log_probabilities.to(student_log_probabilities.dtype)
    difference = torch.where(probabilities > 0, teacher_log - student_log_probabilities,
                             torch.zeros_like(teacher_log))
    per_position = (probabilities * difference).sum(dim=-1)
    if mask is not None:
        per_position = per_position * mask.to(per_position.dtype)
    per_row = per_position.sum(dim=-1)
    mean = per_row.mean()
    with torch.no_grad():
        diagnostics = {"rows": int(per_row.shape[0]),
                       "positions": int(per_position.shape[-1]),
                       "mean_conditional_kl": float(per_row.detach().double().mean()),
                       "max_conditional_kl": float(per_row.detach().double().max()),
                       "max_abs_teacher_sum_error": worst,
                       "units": "nats per window, summed over positions and residues",
                       "zero_convention": "p == 0 contributes exactly 0; no epsilon is added"}
    return mean, diagnostics


def _log_sinkhorn(cost, log_a, log_b, *, epsilon, iterations, tolerance):
    """Log-domain Sinkhorn. Returns the **regularised** OT value and the iterations.

    The value is the entropic objective, ``<P, C> + epsilon * KL(P || a (x) b)``, not
    the transport cost alone. Returning ``<P, C>`` and calling it the regularised
    objective was wrong in a way that matters for the identity check: at ``a == b``
    the optimal plan is not the independent coupling, so the two quantities differ,
    and the debiasing terms subtract the regularised values.

    The KL term is computed from the dual potentials rather than from the plan, which
    is the standard identity ``KL(P || a (x) b) = (<f, a> + <g, b> - <P, C>) /
    epsilon`` at the optimum and avoids forming ``P log P`` where ``P`` underflows.
    """
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    used = 0
    for used in range(1, int(iterations) + 1):
        f_previous = f
        g = -epsilon * torch.logsumexp((f.unsqueeze(1) - cost) / epsilon + log_a.unsqueeze(1),
                                       dim=0)
        f = -epsilon * torch.logsumexp((g.unsqueeze(0) - cost) / epsilon + log_b.unsqueeze(0),
                                       dim=1)
        if float((f - f_previous).abs().max()) < float(tolerance):
            break
    plan = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon
                     + log_a.unsqueeze(1) + log_b.unsqueeze(0))
    transport = (plan * cost).sum()
    a, b = log_a.exp(), log_b.exp()
    regularised = (f * a).sum() + (g * b).sum()
    divergence = regularised - transport
    return regularised, {"iterations": used, "transport_cost": transport,
                         "entropic_penalty": divergence,
                         "plan_mass": plan.sum(),
                         "identity": ("the regularised objective equals <f, a> + <g, b> at the "
                                      "optimum, and epsilon * KL(P || a (x) b) is that minus the "
                                      "transport cost")}


def sinkhorn_divergence(log_a, log_b, cost, *, epsilon=SINKHORN_EPSILON,
                        iterations=SINKHORN_ITERATIONS, tolerance=SINKHORN_TOLERANCE):
    """Debiased Sinkhorn divergence over a shared discrete support.

    ``S(a, b) = OT(a, b) - (OT(a, a) + OT(b, b)) / 2``. The debiasing terms are what
    make ``S(a, a) = 0`` within solver tolerance; the raw entropic OT is strictly
    positive there, so an implementation that omitted them would pass a gradient
    test and fail the identity check.

    The cost is supplied, not chosen here. Normalised Hamming over the window is
    the declared starting cost; a separately qualified frozen structural cost is the
    declared alternative. Hamming alone does not encode whether a combination binds,
    and the report says so beside every transport result.
    """
    require(cost.dim() == 2 and cost.shape[0] == log_a.shape[0] == log_b.shape[0]
            == cost.shape[1],
            f"The cost matrix {tuple(cost.shape)} must be square over the shared support")
    value_ab, record_ab = _log_sinkhorn(cost, log_a, log_b, epsilon=epsilon,
                                        iterations=iterations, tolerance=tolerance)
    value_aa, record_aa = _log_sinkhorn(cost, log_a, log_a, epsilon=epsilon,
                                        iterations=iterations, tolerance=tolerance)
    value_bb, record_bb = _log_sinkhorn(cost, log_b, log_b, epsilon=epsilon,
                                        iterations=iterations, tolerance=tolerance)
    divergence = value_ab - 0.5 * (value_aa + value_bb)
    with torch.no_grad():
        diagnostics = {"support": int(cost.shape[0]), "epsilon": float(epsilon),
                       "iterations_used": int(record_ab["iterations"]),
                       "ot_ab": float(value_ab.detach()), "ot_aa": float(value_aa.detach()),
                       "ot_bb": float(value_bb.detach()),
                       "transport_cost_ab": float(record_ab["transport_cost"].detach()),
                       "entropic_penalty_ab": float(record_ab["entropic_penalty"].detach()),
                       "plan_mass_ab": float(record_ab["plan_mass"].detach()),
                       "debiased": True,
                       "objective": ("the REGULARISED entropic objective, <P, C> + epsilon * "
                                     "KL(P || a (x) b), debiased. Returning <P, C> alone and "
                                     "calling it regularised understates it by the penalty term, "
                                     "which at a == b is not zero."),
                       "cost_caveat": ("normalised Hamming over the window does not encode whether "
                                       "a combination binds. A structural cost has to be "
                                       "separately qualified before it replaces it.")}
        diagnostics["self_terms_iterations"] = [int(record_aa["iterations"]),
                                                int(record_bb["iterations"])]
    return divergence, diagnostics


def teacher_kl_convention():
    """The declared convention for zero teacher mass in :func:`forward_kl`.

    Stated as a callable fact rather than only as prose, because the alternative --
    adding an epsilon -- changes the value of every KL this campaign reports and is
    the kind of choice that gets made silently inside a loss function.
    """
    return {"zero_teacher_mass": ("p == 0 contributes exactly 0 and the multiplication is masked "
                                 "BEFORE it happens, so a -inf teacher log at an underflowed "
                                 "residue cannot produce 0 * -inf on the forward pass or 0 * NaN "
                                 "on the backward one"),
            "no_epsilon": ("no epsilon is added. An epsilon changes the value of every reported KL "
                           "and makes the zero-at-identity check approximate"),
            "student_zero": ("a student probability of zero where the teacher has mass gives +inf, "
                             "which is the correct answer: the student has excluded something the "
                             "reference requires. It is not clamped."),
            "direction": "KL(reference || policy): forward, mass-covering"}


def hamming_cost(windows):
    """Normalised pairwise Hamming cost over a shared list of equal-length windows."""
    require(windows, "An empty support has no cost matrix")
    length = len(windows[0])
    require(all(len(window) == length for window in windows),
            "A transport cost over windows of different lengths is not defined here")
    size = len(windows)
    cost = torch.zeros(size, size, dtype=torch.float64)
    for i in range(size):
        for j in range(i + 1, size):
            distance = sum(1 for a, b in zip(windows[i], windows[j]) if a != b) / length
            cost[i, j] = cost[j, i] = distance
    return cost


# ---------------------------------------------------------------------------
# planner weighting
# ---------------------------------------------------------------------------

def planner_weights(log_probabilities, mask, *, weight, mode="easy_first"):
    """Detached, mask-normalised per-target weights derived from model confidence.

    At ``weight == 0`` the returned weights are exactly the mask, so the weighted
    loss and its gradients reproduce ordinary training bit for bit. That identity is
    what I19 asserts, and it only holds because the weight enters multiplicatively
    on an already mask-normalised loss rather than replacing the normalisation.
    """
    require(mode in ("easy_first", "hard_first"), f"Unknown planner mode {mode!r}")
    require(0.0 <= float(weight) <= 1.0, "The planner weight is a fraction in [0, 1]")
    mask = mask.to(log_probabilities.dtype)
    if float(weight) == 0.0:
        return mask, {"weight": 0.0, "mode": mode, "identical_to_ordinary_training": True}
    with torch.no_grad():
        confidence = log_probabilities.max(dim=-1).values
        centred = confidence - (confidence * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        signed = centred if mode == "easy_first" else -centred
        raw = torch.sigmoid(signed) * mask
        scale = torch.clamp(mask.sum(), min=1.0) / torch.clamp(raw.sum(), min=1e-12)
        derived = (raw * scale).detach()
    weights = (1.0 - float(weight)) * mask + float(weight) * derived
    return weights, {"weight": float(weight), "mode": mode, "detached": True,
                     "identical_to_ordinary_training": False,
                     "normalisation": ("derived weights are rescaled to sum to the active-target "
                                       "count, so the planner reweights targets without changing "
                                       "the effective loss magnitude"),
                     "retained": "ordinary reconstruction is retained beside the weighted term"}


def combine(terms, *, coefficients):
    """Weighted sum of named loss terms, with the per-term values recorded.

    Returns ``(total, record)``. A term whose coefficient is zero is **not**
    evaluated by the caller; recording it here as present-with-zero would imply a
    forward pass that did not happen.
    """
    total, record = None, {}
    for name, value in sorted(dict(terms).items()):
        coefficient = float(dict(coefficients).get(name, 0.0))
        contribution = coefficient * value
        total = contribution if total is None else total + contribution
        record[name] = {"coefficient": coefficient, "value": float(value.detach()),
                        "contribution": float(contribution.detach())}
    require(total is not None, "No loss terms to combine")
    return total, {"terms": record, "total": float(total.detach())}
