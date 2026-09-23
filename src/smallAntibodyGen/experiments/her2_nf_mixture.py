"""The sequence-level parent/policy mixture ``M_alpha = (1-alpha)P + alpha Q``.

This is a distribution over whole editable sequences. It is not weight
interpolation, it is not an average of log probabilities, and -- the error that
is easiest to write and hardest to see -- it is *not* a fixed-weight mixture of
per-position conditionals. A fixed weight at every position defines a different
distribution, whose sequence probability is not ``M_alpha``. The correct
conditionals carry a posterior over which component is generating, updated after
each editable residue:

    ``w_Q(u) = alpha Q(u) / [(1-alpha) P(u) + alpha Q(u)]``,
    ``M(a|u) = (1 - w_Q(u)) P(a|u) + w_Q(u) Q(a|u)``,

initialized at ``w_Q(empty) = alpha``. The likelihood of the fixed conditioning
prompt is common to both components and is excluded: including it would divide
out anyway in exact arithmetic and would inject an unnecessary large magnitude
into the exponentials.

Two exact facts the flight uses, both derivable in one line:

* ``M_alpha(x) / P(x) >= 1 - alpha`` wherever ``P(x) > 0``. At ``alpha = .89`` a
  loss cannot reach tenfold even if ``Q`` assigns zero probability, which is why
  ``.89`` and not ``.9`` is the primary certified-floor control at an inclusive
  threshold; ``.9`` is reported beside it.
* for every ``alpha > 0``, ``M_alpha(x) < P(x)`` iff ``Q(x) < P(x)``. The
  *fraction* of parent draws losing any probability is therefore identical to
  ``Q``'s. Only the magnitudes shrink, which is exactly the limitation of that
  breadth statistic.

Generation stores two models and runs one component per draw; exact mixture
scoring generally needs both. Those costs are recorded, not hidden.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .her2_data import CORE_LENGTH
from .her2_nf_contract import NF_SCHEMA
from .her2_runtime import require

#: The persisted grid. Every point is computed and saved; presenting one chosen
#: alpha as if it had been prespecified is what this grid exists to prevent.
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 0.89, 0.9, 0.95, 0.99, 1.0)

#: The primary certified-floor control and the reported neighbour.
PRIMARY_ALPHA = 0.89
SECONDARY_ALPHA = 0.9

#: ``alpha`` above which the pointwise floor no longer STRICTLY excludes the
#: inclusive event: ``M/P >= 1-alpha`` excludes ``M/P <= 1/k`` exactly when
#: ``alpha < 1 - 1/k``. These are the exact boundaries, not ``1-alpha`` rounded.
TENFOLD_ALPHA_BOUNDARY = 0.9
HUNDREDFOLD_ALPHA_BOUNDARY = 0.99


def _check_alpha(alpha):
    value = float(alpha)
    require(0.0 <= value <= 1.0, f"alpha outside [0, 1]: {alpha!r}")
    return value


def mixture_log_probability(log_parent, log_policy, alpha):
    """``log[(1-alpha) exp(lP) + alpha exp(lQ)]``, stable, with the endpoints exact.

    ``alpha = 0`` and ``alpha = 1`` are returned as the component itself rather
    than computed: ``log(0)`` is ``-inf`` and ``-inf + (-inf)`` is ``nan`` on the
    path that a generic ``logaddexp`` would take when a component has zero
    probability, which is precisely the case the certified floor is about.
    """
    alpha = _check_alpha(alpha)
    parent = np.asarray(log_parent, dtype=np.float64)
    policy = np.asarray(log_policy, dtype=np.float64)
    require(parent.shape == policy.shape, "Parent and policy log probabilities must align")
    if alpha == 0.0:
        return parent.copy()
    if alpha == 1.0:
        return policy.copy()
    return np.logaddexp(math.log1p(-alpha) + parent, math.log(alpha) + policy)


def floor_record(alpha):
    """The pointwise floor and what it does and does not certify.

    The exclusion flags compare the DECLARED alpha with its exact boundary rather
    than comparing ``1 - alpha`` with ``.1`` / ``.01``. In binary floating point
    ``1 - .99`` is ``.010000000000000009``, which is above ``.01`` -- so the
    naive comparison certifies that ``alpha = .99`` strictly excludes an
    inclusive hundredfold loss, which it does not: at ``alpha = .99`` the floor
    is exactly ``1/100`` and the inclusive event is attained.
    """
    alpha = _check_alpha(alpha)
    floor = 1.0 - alpha
    return {"alpha": alpha, "ratio_floor": floor,
            "max_certified_loss_factor": (1.0 / floor) if floor > 0 else None,
            "excludes_inclusive_tenfold": bool(alpha < TENFOLD_ALPHA_BOUNDARY),
            "excludes_inclusive_hundredfold": bool(alpha < HUNDREDFOLD_ALPHA_BOUNDARY),
            "boundary_rule": ("M/P >= 1-alpha, so an INCLUSIVE k-fold loss (M/P <= 1/k) is "
                              "strictly excluded exactly when alpha < 1 - 1/k. At alpha = 1 - 1/k "
                              "the floor is attained and the inclusive event is not excluded."),
            "boundaries": {"tenfold": TENFOLD_ALPHA_BOUNDARY,
                           "hundredfold": HUNDREDFOLD_ALPHA_BOUNDARY},
            "statement": "M_alpha(x) / P(x) >= 1 - alpha wherever P(x) > 0",
            "holds_even_if": "Q assigns the sequence exactly zero probability",
            "does_not_establish": ["higher average precision", "lower average KL",
                                   "a smaller fraction of parent draws losing any probability"],
            "breadth_note": ("for every alpha > 0, M_alpha(x) < P(x) iff Q(x) < P(x): the fraction "
                             "losing any probability is unchanged from Q while the magnitudes "
                             "shrink.")}


# ---------------------------------------------------------------------------
# the true conditionals
# ---------------------------------------------------------------------------

def posterior_component_log_weights(parent_position_log_probs, policy_position_log_probs, alpha):
    """``(log w_P, log w_Q)``, each ``(N, 10)``, computed DIRECTLY from the prefixes.

    Both weights come from the same normalizer, so neither is reconstructed as
    the complement of the other. That matters exactly where the mixture matters:
    when one component's prefix likelihood is zero (or merely 100 nats below the
    other's) the posterior rounds to one in float64, and ``log1p(-w)`` then
    returns ``0`` where the true value is ``-100``. Computing
    ``log w_P = log(1-alpha) + ell_P - logaddexp(...)`` keeps that term exact.

    Position ``0`` carries the prior ``alpha`` exactly -- no editable residue has
    been observed yet -- and position ``k`` uses the realized prefix ``x_<k``. The
    fixed conditioning prompt is in neither prefix likelihood: it is common to
    both components and would divide out.
    """
    alpha = _check_alpha(alpha)
    parent = np.asarray(parent_position_log_probs, dtype=np.float64)
    policy = np.asarray(policy_position_log_probs, dtype=np.float64)
    require(parent.shape == policy.shape and parent.ndim == 2,
            "Expected aligned (N, length) realized-position log probabilities")
    rows, length = parent.shape
    if alpha == 0.0:
        return (np.zeros((rows, length)), np.full((rows, length), -np.inf))
    if alpha == 1.0:
        return (np.full((rows, length), -np.inf), np.zeros((rows, length)))
    prefix_parent = np.concatenate([np.zeros((rows, 1)), np.cumsum(parent, axis=1)[:, :-1]], axis=1)
    prefix_policy = np.concatenate([np.zeros((rows, 1)), np.cumsum(policy, axis=1)[:, :-1]], axis=1)
    left = math.log1p(-alpha) + prefix_parent
    right = math.log(alpha) + prefix_policy
    normalizer = np.logaddexp(left, right)
    return left - normalizer, right - normalizer


def posterior_component_weights(parent_position_log_probs, policy_position_log_probs, alpha):
    """``(N, 10)`` posterior weight on ``Q`` *before* each editable position."""
    _, log_q = posterior_component_log_weights(parent_position_log_probs,
                                               policy_position_log_probs, alpha)
    return np.exp(log_q)


def mixture_conditionals(parent_conditionals, policy_conditionals, weights):
    """``(N, 10, 20)`` mixture conditional *probabilities* under the posterior weights.

    ``parent_conditionals`` and ``policy_conditionals`` are the full 20-way
    conditional probability vectors at the realized prefixes. The weights come
    from :func:`posterior_component_weights` and are broadcast over the residue
    axis; a scalar alpha broadcast in their place would silently define the wrong
    distribution, which is why the weights are an argument and not a default.

    The residue axis is checked for alignment rather than pinned to twenty: this
    is a math kernel, and the native event's twenty-residue support is proved on
    the live weights by :func:`her2_nf_contract.verify_probability_contract`.
    """
    parent = np.asarray(parent_conditionals, dtype=np.float64)
    policy = np.asarray(policy_conditionals, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    require(parent.shape == policy.shape and parent.ndim == 3,
            "Expected aligned (N, length, alphabet) conditional probability blocks")
    require(weight.shape == parent.shape[:2], "One posterior weight per row and position")
    w = weight[:, :, None]
    return (1.0 - w) * parent + w * policy


def mixture_log_conditionals(parent_log_conditionals, policy_log_conditionals, log_weights):
    """``log M(a | x_<i)`` in log space, from native log vectors and log weights.

    Never ``log(mixture_conditionals(...))``: a component conditional that
    underflows to zero in probability space is a finite log here, and a mixture
    whose weight on one component is ``exp(-745)`` still contributes exactly.
    """
    parent = np.asarray(parent_log_conditionals, dtype=np.float64)
    policy = np.asarray(policy_log_conditionals, dtype=np.float64)
    log_parent_weight, log_policy_weight = log_weights
    require(parent.shape == policy.shape and parent.ndim == 3,
            "Expected aligned (N, length, alphabet) LOG conditional blocks")
    left = np.asarray(log_parent_weight, dtype=np.float64)[:, :, None] + parent
    right = np.asarray(log_policy_weight, dtype=np.float64)[:, :, None] + policy
    return _logaddexp_both_infinite_safe(left, right)


def _logaddexp_both_infinite_safe(left, right):
    """``logaddexp`` that returns ``-inf`` where both arguments are ``-inf``.

    ``numpy.logaddexp(-inf, -inf)`` is ``-inf`` already; what is not safe is
    ``-inf + -inf`` arriving as ``nan`` from an earlier product. Guarding here
    keeps a zero-mass component from turning a whole column into ``nan``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    both_zero = np.isneginf(left) & np.isneginf(right)
    with np.errstate(invalid="ignore"):
        out = np.logaddexp(left, right)
    return np.where(both_zero, -np.inf, out)


def chain_rule_check(parent_position_log_probs, policy_position_log_probs, alpha):
    """``(per-position mixture log probs, their sum, the direct mixture value)``.

    The sum over positions of ``log M(x_i | x_<i)`` must equal
    ``mixture_log_probability`` of the two sequence sums. That equality is the
    whole content of "the posterior weights are the right weights".

    Both component log weights are taken directly from
    :func:`posterior_component_log_weights`. Reconstructing the parent weight as
    ``log1p(-w_Q)`` reintroduces the zero-mass component the certified floor is
    about: with ``P = [.5, .5]``, ``Q = [0, .9]`` and ``alpha = .89`` the
    reconstruction is off by 1.0296 nats, and with the merely extreme finite
    logs ``P = [-100, 0]``, ``Q = [0, -100]`` it is off by .1165 because a
    posterior rounds to one.
    """
    alpha = _check_alpha(alpha)
    parent = np.asarray(parent_position_log_probs, dtype=np.float64)
    policy = np.asarray(policy_position_log_probs, dtype=np.float64)
    if alpha == 0.0:
        per_position = parent.copy()
    elif alpha == 1.0:
        per_position = policy.copy()
    else:
        log_parent_weight, log_policy_weight = posterior_component_log_weights(
            parent, policy, alpha)
        per_position = _logaddexp_both_infinite_safe(log_parent_weight + parent,
                                                     log_policy_weight + policy)
    direct = mixture_log_probability(parent.sum(axis=1), policy.sum(axis=1), alpha)
    chain = per_position.sum(axis=1)
    finite = np.isfinite(chain) & np.isfinite(direct)
    worst = float(np.max(np.abs(chain[finite] - direct[finite]))) if finite.any() else 0.0
    return {"per_position": per_position, "chain_rule_sum": chain, "direct": direct,
            "rows_with_zero_mixture_mass": int((~finite).sum()),
            "max_abs_difference": worst,
            "basis": ("both component log posterior weights are computed from the prefix log "
                      "likelihoods; neither is reconstructed from the other. Rows where the "
                      "mixture itself has zero mass are counted, not silently averaged in.")}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def sample_mixture(parent_policy, policy, *, count, alpha, component_seed, parent_seed,
                   policy_seed, batch_size=256):
    """Draw ``count`` cores, choosing the component **once per sequence**.

    Choosing per position would sample a different distribution entirely. The
    component identity is returned and persisted, because it is part of the
    generation record and because a mixture's dependence structure can come from
    the component indicator rather than from either network.
    """
    alpha = _check_alpha(alpha)
    count = int(count)
    require(count > 0, "Positive draw count required")
    generator = np.random.default_rng(int(component_seed))
    from_policy = generator.random(count) < alpha
    take_policy = int(from_policy.sum())
    take_parent = count - take_policy
    length = int(getattr(parent_policy, "core_length", CORE_LENGTH))
    require(length == int(getattr(policy, "core_length", CORE_LENGTH)),
            "the two mixture components disagree about the modelled sequence length")
    index = np.zeros((count, length), dtype=np.int8)
    component_log_probability = np.zeros(count, dtype=np.float64)
    if take_parent:
        cores, logp = parent_policy.sample(take_parent, seed=int(parent_seed),
                                           batch_size=int(batch_size))
        index[~from_policy] = cores
        component_log_probability[~from_policy] = logp
    if take_policy:
        cores, logp = policy.sample(take_policy, seed=int(policy_seed), batch_size=int(batch_size))
        index[from_policy] = cores
        component_log_probability[from_policy] = logp
    return {"index": index, "component": np.where(from_policy, 1, 0).astype(np.int8),
            "component_names": ["parent", "policy"],
            "component_log_probability": component_log_probability,
            "draws_from_parent": take_parent, "draws_from_policy": take_policy,
            "alpha": alpha,
            "seeds": {"component": int(component_seed), "parent": int(parent_seed),
                      "policy": int(policy_seed)},
            "note": ("the component is chosen once per sequence. The recorded "
                     "component_log_probability is that component's density; the MIXTURE density "
                     "of the same draw requires scoring both models and is computed separately.")}


def score_mixture(parent_policy, policy, index, *, alpha, batch_size=256):
    """Exact mixture sum-log-probability of every row, from both components.

    This is the cost the specification asks to be stated plainly: generation runs
    one component per draw, exact scoring runs both.
    """
    parent_scores = parent_policy.score(index, batch_size=int(batch_size))["sum_log_probability"]
    policy_scores = policy.score(index, batch_size=int(batch_size))["sum_log_probability"]
    mixture = mixture_log_probability(parent_scores, policy_scores, alpha)
    return {"parent_sum_log_probability": np.asarray(parent_scores, dtype=np.float64),
            "policy_sum_log_probability": np.asarray(policy_scores, dtype=np.float64),
            "mixture_sum_log_probability": mixture,
            "alpha": _check_alpha(alpha),
            "cost_note": ("two full scoring passes per bank; generation itself pays for one model "
                          "per draw plus the storage of two checkpoints")}


def conditional_vectors_for_mixture(parent_policy, policy, index, *, alpha, batch_size=64):
    """Mixture conditional vectors at the realized prefixes of ``index``.

    Used wherever a mixture's KL or total correlation is wanted: those are
    functionals of the mixture's conditionals, and substituting either
    component's conditionals would answer a different question.
    """
    parent_log = position_log_conditionals(parent_policy, index, batch_size=batch_size)
    policy_log = position_log_conditionals(policy, index, batch_size=batch_size)
    values = np.asarray(index)
    rows = np.arange(values.shape[0])[:, None]
    positions = np.arange(values.shape[1])[None, :]
    parent_realized = parent_log[rows, positions, values]
    policy_realized = policy_log[rows, positions, values]
    log_weights = posterior_component_log_weights(parent_realized, policy_realized, alpha)
    log_conditionals = mixture_log_conditionals(parent_log, policy_log, log_weights)
    return {"log_conditionals": log_conditionals,
            "conditionals": np.exp(log_conditionals),
            "posterior_weights": np.exp(log_weights[1]),
            "posterior_log_weights": {"parent": log_weights[0], "policy": log_weights[1]},
            "parent_log_conditionals": parent_log, "policy_log_conditionals": policy_log,
            "parent_conditionals": np.exp(parent_log), "policy_conditionals": np.exp(policy_log),
            "alpha": _check_alpha(alpha),
            "basis": ("native log_softmax vectors throughout; the mixture is combined in log "
                      "space under both component log posterior weights.")}


@torch.no_grad()
def position_log_conditionals(policy, index, *, batch_size=64):
    """``(N, length, alphabet)`` **log** conditionals at the realized prefixes.

    ``log_softmax`` natively, then widened to float64. Going through
    ``softmax -> clip -> log`` instead loses every conditional below about
    ``1e-38`` to a float32 underflow and then replaces it with whatever floor the
    clip declared -- which is exactly the range a tail or a total-correlation
    term is made of. The eval/train mode of the model is restored on the way out.
    """
    values = np.asarray(index)
    length = int(getattr(policy, "core_length", CORE_LENGTH))
    require(values.ndim == 2 and values.shape[1] == length,
            f"Expected an (N, {length}) core block for this policy")
    was_training = bool(policy.model.training)
    policy.model.eval()
    chunks = []
    try:
        for start in range(0, values.shape[0], int(batch_size)):
            block = values[start:start + int(batch_size)]
            core_ids = policy.token_ids(block)
            logits = policy.core_logits(core_ids)
            if logits.dtype not in (torch.float32, torch.float64):
                logits = logits.float()
            chunks.append(torch.log_softmax(logits, dim=-1).double().cpu().numpy())
    finally:
        if was_training:
            policy.model.train()
    return np.concatenate(chunks, axis=0)


def position_conditionals(policy, index, *, batch_size=64):
    """``(N, length, alphabet)`` conditional probabilities at the realized prefixes.

    The exponential of :func:`position_log_conditionals`. Anything that then
    takes a logarithm should read the log vectors directly instead.
    """
    return np.exp(position_log_conditionals(policy, index, batch_size=batch_size))


# ---------------------------------------------------------------------------
# the persisted alpha curve
# ---------------------------------------------------------------------------

def alpha_curve(parent_log_probability, policy_log_probability, *, labels=None, grid=ALPHA_GRID,
                yield_budgets=(10_000,), yield_function=None):
    """One record per alpha on the declared grid. Every point is persisted.

    ``yield_function(log_probabilities, draws)`` is injected so this module does
    not import the metrics layer; the campaign passes
    ``her2_nf_metrics.expected_distinct_yield``.
    """
    parent = np.asarray(parent_log_probability, dtype=np.float64)
    policy = np.asarray(policy_log_probability, dtype=np.float64)
    require(parent.shape == policy.shape, "Parent and policy score vectors must align")
    mask = np.ones(parent.shape, dtype=bool) if labels is None else np.asarray(labels).astype(bool)
    points = []
    for alpha in grid:
        mixture = mixture_log_probability(parent, policy, alpha)
        record = {"alpha": _check_alpha(alpha), "floor": floor_record(alpha),
                  "mean_sum_log_probability": float(mixture.mean()),
                  "mean_sum_log_probability_on_high": float(mixture[mask].mean())}
        if yield_function is not None:
            record["expected_distinct_high"] = {
                str(int(budget)): float(yield_function(mixture[mask], int(budget)))
                for budget in yield_budgets}
        points.append(record)
    return {"schema_version": NF_SCHEMA, "record_kind": "mixture_alpha_curve",
            "grid": [float(a) for a in grid], "rows": int(parent.size),
            "high_rows": int(mask.sum()), "points": points,
            "primary_alpha": PRIMARY_ALPHA, "reported_alpha": SECONDARY_ALPHA,
            "selection_note": ("every grid point is persisted. A selectively chosen alpha is never "
                               "presented as prespecified; .89 is primary because its floor "
                               "strictly excludes an INCLUSIVE tenfold loss.")}


def stopped_run_note(trajectory, last_passing_update):
    """A stopped unregularized run stays stopped; its checkpoint is a labelled diagnostic."""
    return {"trajectory": str(trajectory), "last_passing_update": int(last_passing_update),
            "status": "stopped",
            "use": ("its last passing checkpoint supports a LABELLED DIAGNOSTIC mixture. It does "
                    "not become a completed arm, it is not promoted to an endpoint, and no table "
                    "row presents it as one.")}
