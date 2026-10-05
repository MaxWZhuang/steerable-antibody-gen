"""Decoding orders, remasking, the reporting functional, and the optional Gibbs sampler.

A fixed unmasking order defines one normalised joint over the window, and that is
the **reporting functional** for every route: ``declared`` order, temperature 1,
fixed length, no filtering, no deduplication. A declared probability mixture of
orders is also a valid distribution and is available as
:func:`mixture_of_orders_log_probability`; an average of log probabilities is a
different score and is named differently so the two cannot be confused. Masked
pseudo-likelihood and a changing remasking trajectory are interchangeable with
neither, and :data:`FUNCTIONALS` says so in the artifacts.

The optional energy-guided sampler is an **AR prior times a fixed outcome score**,
which is a different object from the dWJS learned noisy density. Its target is
``pi(x) proportional to q(x) * exp(kappa * s(x))`` with ``s`` frozen. Two properties
are checked before it is used for anything: the enumerated transition kernel
preserves that target under fixed or state-independent block selection
(:func:`verify_stationarity`), and mixing is measured separately
(:func:`mixing_report`). Adaptive confidence-based block selection does **not** give
the declared target and is refused rather than corrected in passing.
"""
from __future__ import annotations

import math

import numpy as np

from .her2_runtime import require

ORDER_DECLARED = "declared"
ORDER_RANDOM = "random"
ORDER_EASY_FIRST = "easy_first"
ORDER_HARD_FIRST = "hard_first"
ORDERS = (ORDER_DECLARED, ORDER_RANDOM, ORDER_EASY_FIRST, ORDER_HARD_FIRST)

FUNCTIONALS = {
    "fixed_order_joint": ("the product of conditionals along ONE declared order. A normalised "
                          "distribution over the window and the reporting functional."),
    "mixture_of_orders": ("log sum_pi p(pi) prod_t q(x_{pi(t)} | ...). Also a normalised "
                          "distribution, over a declared mixture of orders."),
    "mean_log_probability_over_orders": ("an average of log probabilities. NOT a distribution; "
                                         "reported under its own name if reported at all."),
    "masked_pseudo_likelihood": ("a product of single-site conditionals given everything else. "
                                 "Not a joint, and not interchangeable with either of the above."),
    "remasking_trajectory": ("the terminal distribution of a stochastic revision process. It has "
                             "no closed form here and is compared by samples only."),
}


def rank_score(sum_log_probability, window_length):
    """The repository's ranking convention: the mean, never the raw sum.

    Sum and mean are monotone-equivalent at one fixed length, but the campaign
    compares across windows of different lengths, where the sum grows with length.
    """
    require(int(window_length) > 0, "A window length must be positive")
    return float(sum_log_probability) / int(window_length)


def resolve_order(sites, *, strategy, rng=None, confidence=None):
    """The site order for one decode. ``easy_first`` reads a supplied confidence vector."""
    require(strategy in ORDERS, f"Unknown decoding order {strategy!r}")
    sites = list(sites)
    if strategy == ORDER_DECLARED:
        return tuple(sites)
    if strategy == ORDER_RANDOM:
        require(rng is not None, "A random order needs an explicit generator")
        return tuple(sites[int(index)] for index in rng.permutation(len(sites)))
    require(confidence is not None and len(confidence) == len(sites),
            f"{strategy} needs one confidence value per site")
    order = np.argsort(np.asarray(confidence, dtype=np.float64))
    if strategy == ORDER_EASY_FIRST:
        order = order[::-1]
    return tuple(sites[int(index)] for index in order)


def mixture_of_orders_log_probability(per_order_log_probabilities, *, weights=None):
    """``log sum_pi p(pi) exp(log q_pi(x))`` -- a normalised distribution.

    Deliberately separate from :func:`mean_log_probability_over_orders`: the two
    differ, and reporting one under the other's name is the error this pair exists
    to make impossible.
    """
    values = np.asarray(per_order_log_probabilities, dtype=np.float64)
    require(values.size > 0, "No per-order log probabilities")
    if weights is None:
        weights = np.full(values.shape, 1.0 / values.size)
    weights = np.asarray(weights, dtype=np.float64)
    require(abs(float(weights.sum()) - 1.0) < 1e-9,
            f"The order mixture weights sum to {float(weights.sum())}, not 1")
    shifted = values + np.log(weights)
    largest = float(shifted.max())
    return largest + float(np.log(np.exp(shifted - largest).sum()))


def mean_log_probability_over_orders(per_order_log_probabilities):
    """An average of log probabilities. Not a distribution; named so."""
    return float(np.mean(np.asarray(per_order_log_probabilities, dtype=np.float64)))


# ---------------------------------------------------------------------------
# remasking
# ---------------------------------------------------------------------------

def remask_schedule(window, *, rounds, fraction, rng):
    """Which sites are re-opened in each stochastic revision round.

    Revisiting earlier decisions is a legitimate experiment and it does **not**
    guarantee monotone probability improvement or family diversity; the published
    protein evidence for it concerns longer sequences and predicted folding, not
    HER2 affinity. So the schedule is declared, the comparison is against fixed,
    random and easy-first filling on the same weights and independent starts, and
    the claim is whatever the samples show.
    """
    require(int(rounds) >= 0, "Rounds cannot be negative")
    require(0.0 < float(fraction) <= 1.0, "The remask fraction is in (0, 1]")
    count = max(1, int(round(float(fraction) * int(window))))
    return tuple(tuple(int(value) for value in
                       sorted(rng.choice(int(window), size=min(count, int(window)),
                                         replace=False)))
                 for _ in range(int(rounds)))


# ---------------------------------------------------------------------------
# the optional energy-guided Gibbs sampler
# ---------------------------------------------------------------------------

def guided_log_weights(prior_log_probabilities, outcome_scores, *, kappa):
    """``log q(x) + kappa * s(x)``: the unnormalised target on an enumerated block.

    ``outcome_scores`` must be a FIXED scalar per candidate -- the frozen head's
    output. A score that moved with the sampler would make the target a function of
    the chain's own history and the stationarity claim false.
    """
    prior = np.asarray(prior_log_probabilities, dtype=np.float64)
    scores = np.asarray(outcome_scores, dtype=np.float64)
    require(prior.shape == scores.shape,
            f"{prior.shape} prior values against {scores.shape} outcome scores")
    require(bool(np.isfinite(prior).all()) and bool(np.isfinite(scores).all()),
            "A non-finite value entered the guided target")
    return prior + float(kappa) * scores


def block_conditional(log_weights):
    """Normalise an enumerated block's log weights into a conditional."""
    values = np.asarray(log_weights, dtype=np.float64)
    largest = float(values.max())
    exponentiated = np.exp(values - largest)
    return exponentiated / float(exponentiated.sum())


def verify_stationarity(*, states, target, transition, tolerance=1e-10):
    """Does the enumerated kernel preserve ``target``? A number, on a tiny state space.

    Verified by enumeration before the sampler is used for anything, because a
    Gibbs-style sampler whose kernel does not preserve its target still produces
    plausible sequences. Fixed or state-independent block selection gives the
    declared target; adaptive confidence-based selection does not, and
    :func:`require_fixed_block_selection` refuses it.
    """
    states = list(states)
    distribution = np.asarray([float(target[state]) for state in states], dtype=np.float64)
    require(abs(float(distribution.sum()) - 1.0) < 1e-9, "The target does not sum to one")
    kernel = np.asarray([[float(transition[a][b]) for b in states] for a in states],
                        dtype=np.float64)
    rows = kernel.sum(axis=1)
    worst_row = float(np.abs(rows - 1.0).max())
    require(worst_row < 1e-9,
            f"A transition row sums to {rows.max():.6f}; the kernel is not stochastic")
    propagated = distribution @ kernel
    residual = float(np.abs(propagated - distribution).max())
    return {"states": len(states), "max_abs_residual": residual,
            "invariant": residual <= float(tolerance), "tolerance": float(tolerance),
            "row_sum_error": worst_row,
            "claim": ("invariance only. Mixing is a separate measurement, and an invariant kernel "
                      "can still take arbitrarily long to reach its target.")}


def require_fixed_block_selection(mode):
    require(mode in ("fixed", "state_independent"),
            f"Block selection {mode!r} does not give the declared Gibbs target. Adaptive, "
            "confidence-based selection makes the proposal depend on the current state and needs "
            "its own correction; it is refused here rather than used with the declared target's "
            "name attached.")
    return True


def mixing_report(chain, *, target=None, lags=(1, 5, 10, 25, 50)):
    """Autocorrelation, burn-in and occupancy for one guided chain. Not invariance.

    Separate from :func:`verify_stationarity` on purpose: the two questions are
    independent, and a sampler that passes one and fails the other is common.
    """
    values = np.asarray(chain, dtype=np.float64)
    require(values.size > 2, "A mixing report needs more than two samples")
    centred = values - values.mean()
    denominator = float((centred ** 2).sum())
    autocorrelation = {}
    for lag in lags:
        if int(lag) >= values.size:
            autocorrelation[str(int(lag))] = None
            continue
        numerator = float((centred[:-int(lag)] * centred[int(lag):]).sum())
        autocorrelation[str(int(lag))] = (None if denominator == 0.0
                                          else numerator / denominator)
    half = values.size // 2
    record = {"samples": int(values.size), "mean": float(values.mean()),
              "autocorrelation": autocorrelation,
              "first_half_mean": float(values[:half].mean()),
              "second_half_mean": float(values[half:].mean()),
              "burn_in_signal": abs(float(values[:half].mean() - values[half:].mean())),
              "measures": "initialisation, burn-in and autocorrelation, NOT invariance"}
    if target is not None:
        record["target_mean"] = float(target)
        record["absolute_bias"] = abs(float(values.mean()) - float(target))
    return record


# ---------------------------------------------------------------------------
# draw bookkeeping
# ---------------------------------------------------------------------------

def draw_accounting(*, draws, forward_calls, seconds, mode):
    """Report latency and forward calls separately from the draw count.

    A reranked output and a raw draw are different objects, and an edit of a
    supplied seed is different from a full-mask generation. Every comparison in this
    campaign is at shared reported draw counts with these three numbers beside them.
    """
    require(mode in ("full_mask", "seed_edit", "reranked"), f"Unknown draw mode {mode!r}")
    return {"draws": int(draws), "forward_calls": int(forward_calls),
            "seconds": float(seconds), "mode": mode,
            "forward_calls_per_draw": (None if int(draws) == 0
                                       else int(forward_calls) / int(draws)),
            "seconds_per_draw": None if int(draws) == 0 else float(seconds) / int(draws),
            "distinction": ("full_mask generation, editing a supplied seed and reranking a pool "
                            "are three different procedures and are never pooled into one "
                            "'generation' number.")}


def two_site_allocation_example():
    """The plan's AA/VV arithmetic, computed rather than quoted.

    Intended joint ``{AA: .49, AV: .01, VA: .01, VV: .49}``. The single-site
    conditionals that joint implies prefer the matching residue at ``.98``. If
    fully masked generation picks first-position ``A`` at ``.99``, sequential filling
    keeps 98% of the mass useful and still underrepresents ``VV`` fifty-fold. This is
    the motivation for a whole-window task; it does not prove global training is
    universally necessary, because repeated Gibbs resampling of the same compatible
    conditionals also approaches the intended distribution.
    """
    intended = {"AA": 0.49, "AV": 0.01, "VA": 0.01, "VV": 0.49}
    first_marginal = {"A": 0.99, "V": 0.01}
    conditional = {"A": {"A": 0.49 / 0.50, "V": 0.01 / 0.50},
                   "V": {"A": 0.01 / 0.50, "V": 0.49 / 0.50}}
    generated = {f"{a}{b}": first_marginal[a] * conditional[a][b]
                 for a in "AV" for b in "AV"}
    return {"intended": intended, "generated": generated,
            "first_position_marginal": first_marginal,
            "matching_conditional": conditional["A"]["A"],
            "useful_mass_generated": generated["AA"] + generated["VV"],
            "vv_underrepresentation": intended["VV"] / generated["VV"],
            "caveat": ("motivates an explicit whole-window task. It does NOT prove global training "
                       "is universally necessary: repeated Gibbs resampling of these same "
                       "compatible conditionals also approaches the intended joint.")}


def kappa_worked_example(*, kappa=2.0, without_partner=-0.5, with_partner=0.4,
                         biased_reference_odds=0.1):
    """The plan's contextual-effect arithmetic, computed. Used by the spec test."""
    absolute = {"without_partner": math.exp(kappa * without_partner),
                "with_partner": math.exp(kappa * with_partner)}
    return {"kappa": float(kappa), "delta_without_partner": float(without_partner),
            "delta_with_partner": float(with_partner),
            "absolute_target_odds": absolute,
            "interaction": float(with_partner - without_partner),
            "reference_relative_with_partner": float(biased_reference_odds)
            * absolute["with_partner"],
            "finding": ("no fixed preference for this edit satisfies both targets, which is what a "
                        "contextual effect means operationally. Reference-relative fitting leaves "
                        "the better edit below one, so it is the ablation.")}
