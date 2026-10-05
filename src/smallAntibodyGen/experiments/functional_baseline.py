"""A runnable CPU baseline for learning from labels on a finite candidate support.

What this is. A small, NumPy-only, finite-support learning/optimisation baseline with a synthetic
qualification. Given a pool of candidate sequences, a declared reference ``p`` over them, and a
few training labels, it fits a ridge reward predictor and a linear-energy DPO policy, allocates
probability mass on the held-out candidates (exact learned tilt, DPO tilts, a finite SIR
approximation), freezes everything in a hashed artifact, and only THEN scores it against the
held-out outcomes, including a hindsight oracle. What it is not: a native generator, a claim of
biological improvement, or evidence about any pretrained model. The normalised restriction of a
declared reference to a finite pool IS the experiment; it is not the native generator's
distribution.

The mathematics (everything is exact on the finite support unless it says otherwise)::

    p       reference, normalised over the held-out candidates (log weight -inf = structural zero)
    qhat    ∝ p · exp(uhat / τ)      exact learned tilt of the frozen ridge predictor
    q*      ∝ p · exp(u / τ)         hindsight oracle (evaluation only)
    J_u(q)  = E_q[u] − τ · KL(q ‖ p)
    J_u(q*) − J_u(q) = τ · KL(q ‖ q*)               (the oracle optimises J, not raw E[u])
    J_u(q~) − J_u(qhat) = −τ · KL(q~ ‖ qhat) + (E_q~ − E_qhat)[u − uhat]
    DPO     q_θ ∝ p · exp(θᵀφ);  margin = β θᵀ(φ_win − φ_lose);
            objective = mean log(1 + exp(−margin)) + λ‖θ‖²/2, no intercept

The last identity is signed on purpose: a sampler that is imperfect for the SAME predictor can
raise true utility by counteracting reward error, so the two terms are not "two nonnegative
additive errors". A reverse KL is named as such everywhere; there is no mutual-information floor
here, because that familiar bound is for the forward KL.

Three stages with one-way dependence, enforced by signatures and a hashed artifact:

1. ``fit_stage``: training features and outcomes only. Produces a frozen ridge predictor, and a
   bounded DPO policy when usable preference pairs exist.
2. ``allocate``: candidates, the declared prior, FROZEN predictions/energies and configuration.
   It has no outcome argument at all.
3. ``evaluate_frozen``: verifies the artifact, then requires COMPLETE held-out outcomes (a missing
   one fails; nothing is renormalised onto an observed subset) and only here computes the oracle.

Labels and provenance. Acquired/offered labels are counted separately from expanded preference
pairs, repeated label USES and validation labels. Both the FEATURE history and the PRIOR-score
history are declared as label-free, documented supervision (assays, label ids, cost) or unknown.
A one-hot encoding cannot launder a prior that was already trained on concealed outcomes, and a
checkpoint trained on 2,146 target labels plus 24 new ones is not a clean 24-shot model, so the
clean low-N/acquisition-replay verdict is false or unavailable for overlapping or unknown
supervision.

A label's identity is the triple (target, assay, label id). Per-component declared uses (feature
history, prior history, the current acquisition) are reported as USES; they are never summed into a
label count, because one measured label can sit in several components. The unique same-target
total is exact only when every component supplies its ids and no history is unknown; otherwise it
is a bound or unavailable, never an invented sum. Acquired labels are tracked by identity, and the
frozen artifact binds the complete ordered offered roster (ids, values, priors, groups, features,
sequences) by compact digests, so an unselected offered label cannot change silently.

A hash proves the artifact was not edited; it does not prove that a manifest's provenance
statements are true, or that the inputs were truthful.

Scope limits (documented, not hidden). The ridge and DPO solvers are dense and low-dimensional.
Each forms a d x d system (the ridge Gram matrix, the DPO Newton Hessian), and the DPO fit also a
(pairs x d) difference matrix. Both are bounded EXPLICITLY and BEFORE anything is allocated:
``MAX_DENSE_SYSTEM_ELEMENTS`` bounds d x d (so d <= 4096 at the default) and
``MAX_PAIR_MATRIX_ELEMENTS`` bounds pairs x d. Neither is a peak-memory guarantee: one solve holds
several such arrays at once (the system, an identity, their sum, LAPACK's working copy) beside the
(rows x d) or (pairs x d) inputs, so the peak is a small multiple of the bounded element count.
Pair sampling is O(n) in memory whatever the strict-pair universe, but the pairwise concordance is
O(n^2) work in chunks. No performance refactor is attempted here; solving in the span of the
acquired rows (which would lift the d bound) is a separate, separately qualified feature.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .her2_runtime import require

ARTIFACT_SCHEMA = "functional-baseline-frozen/1"
REPORT_SCHEMA = "functional-baseline-report/1"
SELFTEST_SCHEMA = "functional-baseline-selftest/1"

SPLIT_TRAIN = "train"
SPLIT_EVAL = "eval"
CANDIDATE_COLUMNS = ("candidate_id", "sequence", "group_id", "split", "prior_log_weight")
OUTCOME_COLUMNS = ("candidate_id", "utility")

PRIOR_UNIFORM = "uniform"
PRIOR_JOINT = "joint_log_probability"
PRIOR_SCORE = "score_normalized"
PRIOR_KINDS = (PRIOR_UNIFORM, PRIOR_JOINT, PRIOR_SCORE)

HISTORY_LABEL_FREE = "label_free"
HISTORY_DOCUMENTED = "documented"
HISTORY_UNKNOWN = "unknown"
HISTORY_STATUSES = (HISTORY_LABEL_FREE, HISTORY_DOCUMENTED, HISTORY_UNKNOWN)
EXPOSURE_STATUSES = ("none", "declared", "unknown")

SIR_PRIOR = "prior"
SIR_UNIFORM = "uniform_over_target_support"
SIR_DECLARED = "declared"
SIR_PROPOSALS = (SIR_PRIOR, SIR_UNIFORM, SIR_DECLARED)

#: Independent, documented random streams: ``default_rng([seed, stream])``.
STREAM_BUDGET, STREAM_PAIRS, STREAM_PROPOSAL, STREAM_RESAMPLE, STREAM_PERMUTATION = 1, 2, 3, 4, 5
STREAMS = {"label_budget": STREAM_BUDGET, "preference_pairs": STREAM_PAIRS,
           "sir_proposal_draws": STREAM_PROPOSAL, "sir_final_resample": STREAM_RESAMPLE,
           "permutation_check": STREAM_PERMUTATION}

IDENTITY_TOLERANCE = 1e-9
#: Upper bound on the (pairs x dimension) difference matrix the DPO fit will build.
MAX_PAIR_MATRIX_ELEMENTS = 20_000_000
#: Upper bound on dimension x dimension, the dense system behind the ridge Gram matrix and the DPO
#: Newton Hessian (4096 x 4096 float64 = 128 MiB each). Checked before either is allocated. It
#: bounds the element count of ONE such matrix; it is not a peak-memory guarantee.
MAX_DENSE_SYSTEM_ELEMENTS = 4096 * 4096
FROZEN_NAME = "frozen_artifact.json"
REPORT_NAME = "evaluation_report.json"

HASHED_SECTIONS = ("manifest", "config", "features", "training", "prior", "support", "fitted",
                   "predictions", "policies", "provenance")

#: Guarantees and non-guarantees repeated in every report, so a number is not read as more.
CLAIMS = {
    "scope": ("A finite-support learning/optimisation baseline with user-supplied or synthetic "
              "utilities. It is not native generation and not a statement of biological "
              "improvement."),
    "restricted_support": ("The normalised restriction of the declared reference to the finite "
                           "held-out pool is the experiment. It is not automatically the native "
                           "generator's distribution."),
    "provenance": ("Label/prior provenance is DECLARED, not verified. The artifact hash proves "
                   "the artifact was not edited and binds the offered inputs by digest; it is "
                   "integrity/provenance, not proof the inputs were truthful, and the manifest "
                   "does not cryptographically prove the absence of contamination."),
    "label_cost": ("Per-component declared label USES are not unique labels and are never summed. "
                   "A unique same-target total is exact only when every component supplies its "
                   "ids and no history is unknown; otherwise it is a bound or unavailable. "
                   "Validation labels are declared separately; without validation ids their "
                   "disjointness from the other labels is not asserted."),
    "oracle": ("The hindsight oracle uses the evaluation outcomes and is not an achievable "
               "policy. It maximises J = E[u] - tau*KL(q||p), not raw expected utility."),
    "dpo_energy": ("The DPO implicit energy is not calibrated measured utility, and beta is not "
                   "a calibrated utility temperature. Its true-J/KL pressure curve is compared "
                   "separately from numeric-reward attribution."),
    "sir": ("Sampling-importance-resampling over the finite pool is a biased self-normalised "
            "approximation, not an exact sampler."),
    "uncertainty": ("Exact finite-panel expectations carry no Monte Carlo interval and none is "
                    "fabricated; permutation p-values are not confidence intervals."),
}

#: How a label is identified when historical supervision, the prior's training and the current
#: acquisition are compared. Stated once and carried into every label-cost record.
LABEL_IDENTITY_CONVENTION = (
    "A label is identified by (target, assay, label id). Historical entries on THIS problem's "
    "target and assay name candidate ids in this problem's id namespace, so their ids are unioned "
    "with the acquired candidate ids. An entry on another target or assay is a different measured "
    "outcome even if a string id coincides, and is never merged into the same-target total. The "
    "conservative overlap checks still compare EVERY listed id, whatever its target, with the "
    "offered training and the evaluation candidates.")


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------

def _text(value, name):
    require(isinstance(value, str) and value.strip(), f"{name} must be a non-empty string")
    return value.strip()


def _integer(value, name, *, minimum=0):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and int(value) >= minimum,
            f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _real(value, name, *, positive=False):
    require(isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)),
            f"{name} must be a real number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    require(math.isfinite(number), f"{name} must be finite, got {value!r}")
    require(not positive or number > 0.0, f"{name} must be positive, got {value!r}")
    return number


def _vector(values, name, *, length=None, finite=True):
    array = np.array(values, dtype=np.float64, copy=True)
    require(array.ndim == 1, f"{name} must be one-dimensional, got shape {array.shape}")
    require(length is None or array.shape[0] == length,
            f"{name} has {array.shape[0]} entries; expected {length}")
    if finite:
        require(bool(np.isfinite(array).all()), f"{name} must be finite (no NaN or infinity)")
    return array


def _frozen(values, dtype=np.float64):
    """An independent, read-only copy: later in-place edits of the source cannot reach it."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _feature_matrix(values, dimension, *, name):
    array = np.array(values, dtype=np.float64, copy=True)
    require(array.ndim == 2 and array.shape[1] == dimension,
            f"{name} must be a (rows, {dimension}) matrix, got shape {array.shape}")
    require(bool(np.isfinite(array).all()), f"{name} must be finite")
    return array


def _unique_ids(ids, name):
    ids = tuple(ids)
    require(all(isinstance(item, str) and item for item in ids),
            f"{name} must be non-empty strings")
    require(len(set(ids)) == len(ids), f"{name} contain a duplicate identifier")
    return ids


# ---------------------------------------------------------------------------
# strict JSON and hashing
# ---------------------------------------------------------------------------

def _builtin(value, path="$"):
    """Plain JSON data from numpy scalars/arrays and containers. NaN/Infinity are refused."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        require(math.isfinite(number),
                f"{path} is {number}; a strict JSON document has no NaN/Infinity literals. "
                "Encode an infinite quantity as an explicit status instead.")
        return number
    if isinstance(value, np.ndarray):
        return _builtin(value.tolist(), path)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            require(isinstance(key, str), f"{path} has the non-string key {key!r}")
            out[key] = _builtin(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_builtin(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{path} holds a {type(value).__name__}, which is not JSON data")


def canonical_json(document):
    return json.dumps(_builtin(document), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(document):
    """SHA-256 of the canonical JSON form. Floats round-trip exactly through JSON text."""
    return hashlib.sha256(canonical_json(document).encode("ascii")).hexdigest()


def _array_sha256(values):
    """Compact content digest of a float array: its shape plus its little-endian float64 bytes.

    ``-inf`` (a structural zero) hashes as itself; NaN never reaches a stored array.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype="<f8")
    return hashlib.sha256(f"{array.shape}|".encode("ascii") + array.tobytes()).hexdigest()


def encode_log(values):
    """JSON form of log weights: ``-inf`` (structural zero) is ``None``; NaN/+inf are refused."""
    array = np.asarray(values, dtype=np.float64)
    require(not np.isnan(array).any() and not (array == np.inf).any(),
            "log weights contain NaN or +inf")
    return [None if item == -math.inf else float(item) for item in array.tolist()]


def decode_log(items, *, name="log weights"):
    out = []
    for item in items:
        if item is None:
            out.append(-math.inf)
            continue
        require(isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(float(item)),
                f"{name} may hold only finite numbers and null (structural zero); got {item!r}")
        out.append(float(item))
    return _frozen(out)


def _temporary_path(path):
    """The sibling file ``write_json`` stages through; output guards must treat it as an output."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".tmp")


def write_json(path, document):
    """Strict JSON through a temporary file and a replace, never a half-written file."""
    path = Path(path)
    text = json.dumps(_builtin(document), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def read_json(path):
    """Strict JSON. NaN/Infinity, a number that overflows to infinity and a DUPLICATE KEY at any
    nesting level are refused; ``json`` would otherwise let the last duplicate win silently."""
    path = Path(path)
    require(path.is_file(), f"{path} does not exist")

    def refuse(token):
        raise ValueError(f"{path}: {token} is not valid in a strict JSON document")

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{path}: the number {token} overflows to infinity; a strict JSON "
                             "document has no infinite values")
        return value

    def unique_keys(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f"{path}: the JSON key {key!r} appears twice in one object; a "
                                 "duplicate would silently overwrite the earlier value")
            seen.add(key)
        return dict(pairs)
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=refuse,
                      parse_float=finite_float, object_pairs_hook=unique_keys)


# ---------------------------------------------------------------------------
# exact finite-support mathematics
# ---------------------------------------------------------------------------

def _check_log_weights(values, name, *, length=None, allow_no_mass=False):
    array = np.array(values, dtype=np.float64, copy=True)
    require(array.ndim == 1 and array.size >= 1, f"{name} must be a non-empty vector")
    require(length is None or array.shape[0] == length,
            f"{name} has {array.shape[0]} entries; expected {length}")
    require(not np.isnan(array).any(), f"{name} contains NaN")
    require(not (array == np.inf).any(),
            f"{name} contains +inf; a log weight of +inf is not a probability")
    if not allow_no_mass:
        require(bool((array > -np.inf).any()),
                f"{name}: every candidate has zero mass, so there is no support to allocate over")
    return array


def _log_probabilities(values, name):
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1, f"{name} must be a vector of log probabilities")
    require(not np.isnan(array).any() and not (array == np.inf).any(),
            f"{name} contains NaN or +inf")
    return array


def logsumexp(values):
    """Stable log-sum-exp; ``-inf`` entries contribute nothing, all ``-inf`` gives ``-inf``."""
    array = np.asarray(values, dtype=np.float64)
    require(array.size >= 1, "logsumexp of an empty array")
    require(not np.isnan(array).any(), "logsumexp input contains NaN")
    top = float(array.max())
    if top == -math.inf or top == math.inf:
        return top
    return top + math.log(float(np.exp(array - top).sum()))


def normalize_log_weights(log_weights, *, name="weights"):
    """Normalised log probabilities. Structural zeros (``-inf``) stay ``-inf``.

    Support is decided by the log weight, never by exponentiating: a finite ``-10000`` underflows
    to a probability of 0.0 but is still reachable, and a tilt can bring it back.

    The maximum is subtracted FIRST and the normaliser is taken of what remains. ``logsumexp``
    returns ``max + log(sum)``; subtracting that from the raw weights re-adds the maximum, and when
    it is large (``1e16``) ``max + log 2`` rounds to ``max`` so the log of the sum is lost and two
    equal weights come out as ``[0, 0]`` (mass 2) instead of ``[-log 2, -log 2]``. The only
    arithmetic left that can fail is the difference ``weight - max`` itself, and that is refused
    rather than turned into an accidental ``-inf``.
    """
    array = _check_log_weights(log_weights, name)
    top = float(array.max())
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = array - top
    require(bool(np.isfinite(shifted[array > -np.inf]).all()),
            f"{name}: the finite log weights span more than a float64 can subtract, so one would "
            "become a structural zero; rescale them")
    return shifted - math.log(float(np.exp(shifted).sum()))


def _rebased_on_mask(values, mask, name):
    """``values`` minus their maximum over ``mask`` (zero off the mask), refusing an unrepresentable
    difference.

    A tilt uses only DIFFERENCES of ``values``: a common constant cancels in the normalisation. But
    ``log p + values`` rounds each small prior log probability away when ``values`` carries a large
    common offset (``1e16 + log 0.25 == 1e16 + log 0.75``), and no later normalisation can recover
    information that was already cancelled. So the offset is removed BEFORE the values meet the
    prior. Only candidates in ``mask`` set the reference: a candidate the prior (or the sampler)
    cannot reach has no weight, so its value must not drag the reachable ones off scale.
    """
    require(bool(mask.any()), f"{name}: no candidate has any mass to rebase on")
    top = float(values[mask].max())
    with np.errstate(over="ignore", invalid="ignore"):
        rebased = values - top
    require(bool(np.isfinite(rebased[mask]).all()),
            f"{name} differ by more than a float64 can hold relative to their maximum, so "
            "rebasing them would overflow")
    return np.where(mask, rebased, 0.0)


def _normalised_tilt(log_p, scaled, name):
    """Normalised ``log p + scaled``, where ``scaled`` is already rebased and scaled and ``-inf``
    in ``log_p`` is a structural zero. A finite candidate must stay finite through the sum."""
    with np.errstate(over="ignore"):
        mixed = log_p + scaled
    require(bool(np.array_equal(np.isfinite(mixed), log_p > -np.inf)),
            f"{name}: adding the tilt to the log prior overflowed a finite candidate")
    return normalize_log_weights(mixed, name=name)


def log_tilt(log_prior, utilities, tau):
    """``log q`` for ``q ∝ p · exp(u / τ)``, computed entirely in log space.

    ``u`` is rebased on the prior's support before it is mixed with ``log p`` (see
    :func:`_rebased_on_mask`), so a large constant added to every utility returns the prior.
    """
    tau = _real(tau, "tau", positive=True)
    log_p = normalize_log_weights(log_prior, name="prior")
    u = _vector(utilities, "utilities", length=log_p.shape[0])
    with np.errstate(over="ignore"):
        scaled = _rebased_on_mask(u, log_p > -np.inf, "utilities") / tau
    require(bool(np.isfinite(scaled).all()), "utilities / tau overflowed; choose a larger tau")
    return _normalised_tilt(log_p, scaled, "tilted weights")


def dpo_log_policy(log_p, energy, pressure, *, name):
    """Normalised ``log q`` for ``q ∝ p · exp(pressure · energy)``: the one definition shared by
    ``allocate`` and ``verify_artifact``, so the stored policy and its recomputation cannot drift.

    The energy is rebased on the prior's support before it is scaled and mixed, for the same reason
    as :func:`log_tilt`: ``θᵀφ`` is defined only up to a per-pool constant, which must not erase the
    prior.
    """
    log_p = _log_probabilities(log_p, "the prior")
    energy = _vector(energy, "DPO energies", length=log_p.shape[0])
    with np.errstate(over="ignore"):
        shifted = pressure * _rebased_on_mask(energy, log_p > -np.inf, "DPO energies")
    require(bool(np.isfinite(shifted).all()), "a DPO pressure overflowed the energy")
    return _normalised_tilt(log_p, shifted, name)


def kl_divergence(log_a, log_b):
    """``KL(a ‖ b)`` in nats from two normalised log-probability vectors; ``math.inf`` if ``a`` has
    mass where ``b`` has none. Works in log space, so underflowed probabilities give no fake
    infinity: ``KL(uniform ‖ (1, e^-1000))`` is ``500 - log 2``, finite."""
    a, b = _log_probabilities(log_a, "log_a"), _log_probabilities(log_b, "log_b")
    require(a.shape == b.shape, "KL needs two log-probability vectors of one length")
    on = a > -np.inf
    if bool(np.any(on & (b == -np.inf))):
        return math.inf
    value = float(np.sum(np.exp(a[on]) * (a[on] - b[on])))
    if value < 0.0:
        require(value > -1e-9,
                f"KL came out as {value}; the inputs are not normalised log probabilities")
        value = 0.0
    return value


def total_variation(log_a, log_b):
    a, b = _log_probabilities(log_a, "log_a"), _log_probabilities(log_b, "log_b")
    require(a.shape == b.shape, "TV needs two log-probability vectors of one length")
    return 0.5 * float(np.abs(np.exp(a) - np.exp(b)).sum())


def expected_value(log_q, values):
    log_q = _log_probabilities(log_q, "log_q")
    return float(np.sum(np.exp(log_q) * _vector(values, "values", length=log_q.shape[0])))


def effective_support(log_q):
    """``exp`` of the Shannon entropy (nats): how many equally likely candidates ``q`` is worth."""
    log_q = _log_probabilities(log_q, "log_q")
    on = log_q > -np.inf
    entropy = -float(np.sum(np.exp(log_q[on]) * log_q[on]))
    return math.exp(max(entropy, 0.0))


def expected_distinct(probabilities, draws):
    """Per-candidate ``1 - (1 - q_i)^M``: the chance candidate i appears in M i.i.d. draws."""
    draws = _integer(draws, "draws", minimum=1)
    q = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return -np.expm1(draws * np.log1p(-q))


def objective(log_q, log_p, utilities, tau):
    """``J_u(q) = E_q[u] - τ KL(q ‖ p)``; ``-inf`` if ``q`` puts mass where ``p`` has none."""
    kl = kl_divergence(log_q, log_p)
    if math.isinf(kl):
        return -math.inf
    return expected_value(log_q, utilities) - _real(tau, "tau", positive=True) * kl


def oracle_objective_closed_form(log_p, utilities, tau):
    """``J_u(q*) = τ · log Σ p exp(u/τ)``: an independent route to the oracle's objective."""
    tau = _real(tau, "tau", positive=True)
    log_p = normalize_log_weights(log_p, name="prior")
    return tau * logsumexp(log_p + _vector(utilities, "utilities", length=log_p.shape[0]) / tau)


def oracle_regret(log_q, log_p, utilities, tau):
    """Regret against the oracle tilt and the quantity the identity says it equals.

    ``J_u(q*) - J_u(q) = τ KL(q ‖ q*)`` for every ``q`` on the support, including those with
    structural zeros. Both sides are ``inf`` together when ``q`` leaves the reference's support.
    """
    tau = _real(tau, "tau", positive=True)
    log_star = log_tilt(log_p, utilities, tau)
    best = objective(log_star, normalize_log_weights(log_p, name="prior"), utilities, tau)
    value = objective(log_q, normalize_log_weights(log_p, name="prior"), utilities, tau)
    regret = math.inf if math.isinf(value) else best - value
    return {"oracle_objective": best, "objective": value, "regret": regret,
            "tau_kl_to_oracle": tau * kl_divergence(log_q, log_star)}


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------

def _score_vector(values, name):
    """Scores to rank: finite or ``-inf`` (a candidate the policy gives no mass, ranked last).

    NaN and ``+inf`` are refused. ``-inf`` is a legitimate score here, not a missing value.
    """
    array = _vector(values, name, finite=False)
    require(not np.isnan(array).any() and not (array == np.inf).any(),
            f"{name} must not hold NaN or +inf (only -inf, a zero-mass score, is allowed)")
    return array


def _average_ranks(values):
    """Average ranks (1-based). Ties are cut by EQUALITY of neighbours, never by subtraction:
    ``-inf - -inf`` is NaN, which would split a run of equal ``-inf`` scores into singletons."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [values.shape[0]]])
    for start, end in zip(starts.tolist(), ends.tolist()):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    return ranks


def spearman(scores, outcomes):
    """Spearman rank correlation with average ranks, or ``value: None`` with the reason.

    Scores may hold ``-inf`` (all such scores tie, ranked last); outcomes must be finite.
    """
    s = _score_vector(scores, "scores")
    o = _vector(outcomes, "outcomes", length=s.shape[0])
    n = int(s.shape[0])
    if n < 2:
        return {"value": None, "n": n, "reason": "fewer than two candidates"}
    if bool(np.all(s == s[0])):
        return {"value": None, "n": n,
                "reason": "the scores are constant, so the rank correlation is undefined"}
    if bool(np.all(o == o[0])):
        return {"value": None, "n": n,
                "reason": "the outcomes are constant, so the rank correlation is undefined"}
    rs = _average_ranks(s) - (n + 1) / 2.0
    ro = _average_ranks(o) - (n + 1) / 2.0
    value = float(rs @ ro) / math.sqrt(float(rs @ rs) * float(ro @ ro))
    return {"value": float(min(1.0, max(-1.0, value))), "n": n, "reason": None}


def concordance(scores, outcomes, *, chunk=256):
    """Pairwise concordance: pairs with tied OUTCOMES are excluded, tied SCORES earn half credit.

    Scores may hold ``-inf`` (equal ``-inf`` scores are TIED, compared by order, not subtraction).
    """
    s = _score_vector(scores, "scores")
    o = _vector(outcomes, "outcomes", length=s.shape[0])
    n = int(s.shape[0])
    comparable = 0
    credit = 0.0
    tied = 0
    columns = np.arange(n)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        rows = np.arange(start, stop)[:, None]
        later = rows < columns[None, :]
        outcome_sign = np.sign(o[None, :] - o[start:stop, None])
        score_sign = ((s[None, :] > s[start:stop, None]).astype(np.int8)
                      - (s[None, :] < s[start:stop, None]).astype(np.int8))
        pairs = later & (outcome_sign != 0)
        comparable += int(pairs.sum())
        tied += int((pairs & (score_sign == 0)).sum())
        credit += float((pairs & (score_sign == outcome_sign)).sum()) \
            + 0.5 * float((pairs & (score_sign == 0)).sum())
    if comparable == 0:
        return {"value": None, "comparable_pairs": 0, "score_tied_pairs": 0,
                "reason": "no pair of candidates has distinct outcomes"}
    return {"value": credit / comparable, "comparable_pairs": comparable,
            "score_tied_pairs": tied, "reason": None}


# ---------------------------------------------------------------------------
# provenance: supervision history, features, prior, utility
# ---------------------------------------------------------------------------

def _supervision_entry(entry):
    require(isinstance(entry, dict), "a supervision entry must be a mapping")
    unknown = sorted(set(entry) - {"assay", "target", "label_count", "label_ids"})
    require(not unknown, f"a supervision entry has unknown key(s) {unknown}")
    count = _integer(entry.get("label_count"), "supervision label_count", minimum=0)
    ids = entry.get("label_ids")
    if ids is not None:
        require(isinstance(ids, (list, tuple)), "supervision label_ids must be a list or null")
        ids = [str(item) for item in ids]
        require(len(set(ids)) == len(ids), "supervision label_ids repeat an identifier")
        require(len(ids) == count,
                f"supervision lists {len(ids)} label ids but declares label_count={count}")
    return {"assay": _text(entry.get("assay"), "supervision assay"),
            "target": _text(entry.get("target"), "supervision target"),
            "label_count": count, "label_ids": ids}


@dataclass(frozen=True)
class SupervisionHistory:
    """What labels, if any, a feature extractor or a prior score was trained on.

    ``label_free`` asserts no labels; ``documented`` lists each supervision (assay, target, cost
    and, ideally, the label ids); ``unknown`` says nobody knows, which makes a clean low-N claim
    unavailable. Sequence-only exposure (pretraining on sequences, no outcomes) is recorded
    separately because it is a different contamination channel.
    """

    status: str
    description: str = ""
    documented: tuple = ()
    exposure_status: str = "unknown"
    exposure_description: str = ""

    def __post_init__(self):
        require(self.status in HISTORY_STATUSES,
                f"supervision status must be one of {HISTORY_STATUSES}, got {self.status!r}")
        require(self.exposure_status in EXPOSURE_STATUSES,
                f"sequence_only_exposure status must be one of {EXPOSURE_STATUSES}")
        entries = tuple(_supervision_entry(entry) for entry in self.documented)
        if self.status == HISTORY_DOCUMENTED:
            require(entries, "documented supervision must list at least one entry "
                             "(assay, target, label_count, optional label_ids)")
        else:
            require(not entries, f"a {self.status!r} history cannot list supervision entries")
        object.__setattr__(self, "documented", entries)
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "exposure_description", str(self.exposure_description))

    @classmethod
    def from_document(cls, document, *, label):
        require(isinstance(document, dict), f"{label} provenance must be a mapping")
        unknown = sorted(set(document) - {"status", "description", "documented_supervision",
                                          "sequence_only_exposure"})
        require(not unknown, f"{label} provenance has unknown key(s) {unknown}")
        require("status" in document, f"{label} provenance needs a 'status'")
        exposure = document.get("sequence_only_exposure", {"status": "unknown"})
        require(isinstance(exposure, dict)
                and not sorted(set(exposure) - {"status", "description"}),
                f"{label} sequence_only_exposure must be a mapping of status/description")
        return cls(status=document["status"], description=document.get("description", ""),
                   documented=tuple(document.get("documented_supervision", ())),
                   exposure_status=exposure.get("status", "unknown"),
                   exposure_description=exposure.get("description", ""))

    def document(self):
        return {"status": self.status, "description": self.description,
                "documented_supervision": copy.deepcopy(list(self.documented)),
                "sequence_only_exposure": {"status": self.exposure_status,
                                           "description": self.exposure_description}}

    def snapshot(self):
        """An independent copy (new entry dicts, new id lists): later edits of this object's
        nested entries cannot reach the copy, and edits of the copy cannot reach this object."""
        return SupervisionHistory(status=self.status, description=self.description,
                                  documented=self.documented, exposure_status=self.exposure_status,
                                  exposure_description=self.exposure_description)


@dataclass(frozen=True)
class FeatureSpace:
    """Columns, how they were built, and what supervised the thing that built them.

    The generic explicit-feature API (any future representation) requires ``supervision`` -- there
    is no default -- so provenance cannot be dropped on the way in.
    """

    kind: str
    names: tuple
    schema: dict
    history: tuple
    supervision: SupervisionHistory

    def __post_init__(self):
        require(isinstance(self.kind, str) and self.kind, "a feature space needs a kind")
        names = tuple(str(name) for name in self.names)
        require(names and all(names) and len(set(names)) == len(names),
                "feature names must be non-empty and unique")
        require(isinstance(self.supervision, SupervisionHistory),
                "feature provenance is required: pass a SupervisionHistory (label_free, "
                "documented or unknown)")
        history = tuple(str(step) for step in self.history)
        require(history, "the preprocessing history must be documented, even if it is one step")
        schema = copy.deepcopy(dict(self.schema))
        _builtin(schema)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "schema", schema)
        # The supervision record is copied too, so no FeatureSpace aliases a caller-owned one.
        object.__setattr__(self, "supervision", self.supervision.snapshot())

    @property
    def dimension(self):
        return len(self.names)

    def document(self):
        return {"kind": self.kind, "names": list(self.names), "schema": copy.deepcopy(self.schema),
                "history": list(self.history), "provenance": self.supervision.document()}

    def snapshot(self):
        """An independent deep copy, taken at every problem/model/fit boundary."""
        return FeatureSpace(kind=self.kind, names=self.names, schema=self.schema,
                            history=self.history, supervision=self.supervision)

    def fingerprint(self):
        return digest(self.document())

    def indices(self, columns):
        columns = tuple(columns)
        require(columns, "name at least one feature column")
        require(len(set(columns)) == len(columns), "a feature column is named twice")
        unknown = [name for name in columns if name not in self.names]
        require(not unknown, f"unknown feature column(s) {unknown}; have {list(self.names)}")
        return [self.names.index(name) for name in columns]

    def select(self, columns):
        """``(sub_space, indices)``: a column subset whose history says it was subset."""
        indices = self.indices(columns)
        return (FeatureSpace(kind=self.kind, names=tuple(columns),
                             schema={**copy.deepcopy(self.schema),
                                     "selected_from": list(self.names)},
                             history=self.history + (f"select columns {list(columns)}",),
                             supervision=self.supervision), indices)


def canonical_sequence(sequence):
    """The physical sequence: whitespace stripped, upper-cased."""
    return str(sequence).strip().upper()


def onehot_feature_space(alphabet, length, supervision):
    require(isinstance(alphabet, str) and len(alphabet) >= 1
            and len(set(alphabet)) == len(alphabet) and not any(c.isspace() for c in alphabet),
            "the alphabet must be a string of distinct non-space characters")
    length = _integer(length, "sequence length", minimum=1)
    alphabet = alphabet.upper()
    require(len(set(alphabet)) == len(alphabet), "the alphabet repeats a character once upper-cased")
    return FeatureSpace(
        kind="sequence_onehot",
        names=tuple(f"pos{position}={residue}" for position in range(length)
                    for residue in alphabet),
        schema={"encoding": "one_hot_position_major", "alphabet": alphabet, "length": length},
        history=("one-hot encode every position over the fixed alphabet (position-major)",),
        supervision=supervision)


def onehot_encode(sequences, space):
    """Fixed-alphabet, fixed-length one-hot features. Invalid residues/lengths are refused."""
    require(isinstance(space, FeatureSpace) and space.kind == "sequence_onehot",
            "onehot_encode needs a sequence_onehot FeatureSpace")
    alphabet, length = space.schema["alphabet"], int(space.schema["length"])
    lookup = {residue: index for index, residue in enumerate(alphabet)}
    sequences = [canonical_sequence(item) for item in sequences]
    matrix = np.zeros((len(sequences), length * len(alphabet)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        require(len(sequence) == length,
                f"sequence #{row} ({sequence!r}) has length {len(sequence)}; every candidate must "
                f"have the fixed length {length}")
        for position, residue in enumerate(sequence):
            require(residue in lookup,
                    f"sequence #{row} ({sequence!r}) has residue {residue!r} outside the "
                    f"alphabet {alphabet!r}")
            matrix[row, position * len(alphabet) + lookup[residue]] = 1.0
    return matrix


@dataclass(frozen=True)
class UtilitySpec:
    """One comparable target/assay utility. Orientation and the useful threshold are DECLARED."""

    units: str
    scale: str
    higher_is_better: bool
    useful_threshold: float
    name: str = "utility"

    def __post_init__(self):
        object.__setattr__(self, "units", _text(self.units, "utility units"))
        object.__setattr__(self, "scale", _text(self.scale, "utility scale"))
        object.__setattr__(self, "name", _text(self.name, "utility name"))
        require(isinstance(self.higher_is_better, bool) and self.higher_is_better,
                "this baseline maximises utility, so the manifest must declare "
                "higher_is_better=true. No conversion is inferred (not from a raw KD, not from a "
                "sign): supply utilities already oriented higher-is-better under an explicit, "
                "documented transform.")
        object.__setattr__(self, "useful_threshold",
                           _real(self.useful_threshold, "useful_threshold"))

    def document(self):
        return {"name": self.name, "units": self.units, "scale": self.scale,
                "higher_is_better": True, "useful_threshold": self.useful_threshold,
                "threshold_rule": ("fixed explicitly in utility units; never fitted from "
                                   "evaluation quantiles"),
                "useful_means": "utility >= useful_threshold"}


@dataclass(frozen=True)
class PriorSpec:
    """The declared reference and what its scores mean. Its label history is declared too."""

    kind: str
    semantics: str
    supervision: SupervisionHistory
    description: str = ""

    def __post_init__(self):
        require(self.kind in PRIOR_KINDS, f"prior kind must be one of {PRIOR_KINDS}")
        object.__setattr__(self, "semantics", _text(self.semantics, "prior semantics"))
        require(isinstance(self.supervision, SupervisionHistory),
                "prior-score provenance is required: pass a SupervisionHistory")
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "supervision", self.supervision.snapshot())

    def snapshot(self):
        """An independent deep copy, taken at every problem/fit boundary."""
        return PriorSpec(kind=self.kind, semantics=self.semantics, supervision=self.supervision,
                         description=self.description)

    def fingerprint(self):
        return digest(self.document())

    def document(self):
        notes = {PRIOR_UNIFORM: "every pool candidate has the same weight",
                 PRIOR_JOINT: ("prior_log_weight is the declared joint log probability of the "
                               "whole sequence under some generator; the pool is a subset, so the "
                               "weights must sum to at most 1 before renormalising"),
                 PRIOR_SCORE: ("prior_log_weight is an arbitrary score; it is soft-maxed over the "
                               "pool and means nothing outside it")}
        return {"kind": self.kind, "semantics": self.semantics, "description": self.description,
                "kind_meaning": notes[self.kind], "provenance": self.supervision.document()}


def validate_prior_log_weights(kind, log_weights, *, name="prior_log_weight"):
    """Structural zeros are explicit ``-inf``; NaN/+inf are refused; kind-specific constraints."""
    require(kind in PRIOR_KINDS, f"prior kind must be one of {PRIOR_KINDS}")
    weights = _check_log_weights(log_weights, name, allow_no_mass=True)
    if kind == PRIOR_UNIFORM:
        require(bool(np.isfinite(weights).all()) and bool((weights == weights[0]).all()),
                f"a uniform prior needs one identical finite {name} for every candidate; use "
                f"{PRIOR_SCORE!r} (with explicit -inf) for a non-uniform or structurally "
                "restricted pool")
    if kind == PRIOR_JOINT:
        total = logsumexp(weights)
        require(total <= 1e-9,
                f"the declared joint log probabilities of distinct sequences sum to "
                f"exp({total:.6g}) > 1, which no probability distribution can do over a subset of "
                "its sequences; check the declaration or use 'score_normalized'")
    return weights


# ---------------------------------------------------------------------------
# the reward predictor and feature-space interventions
# ---------------------------------------------------------------------------

def _require_dense_system(dimension, what):
    """Refuse a dense ``dimension x dimension`` system above ``MAX_DENSE_SYSTEM_ELEMENTS``.

    Pure integer arithmetic: it runs BEFORE the Gram matrix or the Hessian (or anything sized by
    them) is allocated, so the refusal itself costs no memory. ``pairs x dimension`` does not bound
    ``dimension x dimension``: one pair of 16384-wide rows still asks for a 2 GiB system.
    """
    dimension = int(dimension)
    elements = dimension * dimension
    require(elements <= MAX_DENSE_SYSTEM_ELEMENTS,
            f"the dense {what} is {dimension} x {dimension} = {elements} elements, above the "
            f"{MAX_DENSE_SYSTEM_ELEMENTS}-element bound on one dense system (dimension "
            f"{math.isqrt(MAX_DENSE_SYSTEM_ELEMENTS)}); reduce the feature dimension. The bound "
            "limits one system, not total peak memory")


def _ridge_solution(features, utilities, alpha):
    """Intercept-free-penalty ridge: centre, solve ``(XcᵀXc + αI) w = Xcᵀ yc``."""
    _require_dense_system(features.shape[1], "ridge Gram system")
    mean_x = features.mean(axis=0)
    mean_y = float(utilities.mean())
    centred = features - mean_x
    gram = centred.T @ centred + alpha * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, centred.T @ (utilities - mean_y))
    return weights, mean_y - float(mean_x @ weights)


@dataclass(frozen=True, eq=False)
class RewardModel:
    """A frozen ridge predictor. Predictions depend only on the row and the frozen weights."""

    space: FeatureSpace
    alpha: float
    weights: object
    intercept: float
    n_train: int
    train_rmse: float

    def predict(self, features):
        matrix = _feature_matrix(features, self.space.dimension, name="prediction features")
        return matrix @ self.weights + self.intercept

    def document(self):
        return {"alpha": self.alpha, "weights": self.weights.tolist(),
                "intercept": self.intercept, "n_train": self.n_train,
                "train_rmse": self.train_rmse,
                "objective": ("ridge: ||y - Xw - b||^2 + alpha ||w||^2, b unpenalised "
                              "(fit by centring)")}


def fit_reward(features, utilities, space, *, alpha):
    """Fit the ridge predictor on TRAINING features and outcomes only."""
    require(isinstance(space, FeatureSpace), "fit_reward needs the FeatureSpace the columns came from")
    alpha = _real(alpha, "ridge alpha", positive=True)
    _require_dense_system(space.dimension, "ridge Gram system")
    matrix = _feature_matrix(features, space.dimension, name="training features")
    y = _vector(utilities, "training utilities", length=matrix.shape[0])
    require(matrix.shape[0] >= 1, "fit_reward needs at least one labelled candidate")
    weights, intercept = _ridge_solution(matrix, y, alpha)
    residual = matrix @ weights + intercept - y
    return RewardModel(space=space.snapshot(), alpha=alpha, weights=_frozen(weights),
                       intercept=float(intercept), n_train=int(matrix.shape[0]),
                       train_rmse=float(math.sqrt(float(np.mean(residual ** 2)))))


@dataclass(frozen=True, eq=False)
class FeatureReference:
    """Column means of the TRAINING features, frozen and hashed: the reference for ablation.

    An ablation replaces columns with these values, never with the evaluation batch's own mean,
    so an intervened row's prediction does not depend on which other rows were scored with it.
    """

    names: tuple
    mean: object
    n_rows: int
    sha256: str

    @classmethod
    def from_training(cls, features, space):
        matrix = _feature_matrix(features, space.dimension, name="training features")
        require(matrix.shape[0] >= 1, "a feature reference needs at least one training row")
        mean = matrix.mean(axis=0)
        return cls(names=space.names, mean=_frozen(mean), n_rows=int(matrix.shape[0]),
                   sha256=digest({"names": list(space.names), "mean": mean.tolist(),
                                  "n_rows": int(matrix.shape[0])}))

    def verify(self):
        expected = digest({"names": list(self.names), "mean": self.mean.tolist(),
                           "n_rows": self.n_rows})
        require(expected == self.sha256, "the feature reference no longer matches its digest")
        return True


INTERVENTION_KINDS = ("identity", "ablate", "patch")


def intervene_features(features, space, kind, *, columns=(), reference=None, donor=None):
    """Return a NEW feature matrix with an intervention applied, to be predicted afterwards.

    * ``identity`` -- a copy; takes no columns, reference or donor.
    * ``ablate`` -- the named columns are replaced by the frozen training reference's means.
    * ``patch`` -- the named columns are replaced by a donor's values: a ``(d,)`` donor row
      broadcast to every row (sufficiency: introduce a feature where it is absent), or an
      ``(n, d)`` matched donor (rescue: restore the original columns after an ablation).

    These are causal manipulations of a PREDICTOR's inputs, not statements about molecules.
    """
    require(kind in INTERVENTION_KINDS, f"unknown intervention {kind!r}; expected {INTERVENTION_KINDS}")
    matrix = _feature_matrix(features, space.dimension, name="features")
    if kind == "identity":
        require(not tuple(columns) and reference is None and donor is None,
                "an identity intervention takes no columns, reference or donor")
        return matrix
    indices = space.indices(columns)
    if kind == "ablate":
        require(donor is None, "ablate takes no donor")
        require(isinstance(reference, FeatureReference),
                "ablation needs an explicit frozen FeatureReference built from TRAINING "
                "features; there is no fallback to the evaluation batch's own mean")
        require(reference.names == space.names, "the feature reference was built for other columns")
        reference.verify()
        matrix[:, indices] = reference.mean[indices]
        return matrix
    require(reference is None, "patch takes no reference mean")
    require(donor is not None, "patch needs an explicit donor row or matched donor matrix")
    source = np.array(donor, dtype=np.float64, copy=True)
    require(bool(np.isfinite(source).all()), "the donor must be finite")
    if source.ndim == 1:
        require(source.shape == (space.dimension,),
                f"a donor row needs {space.dimension} entries, got {source.shape}")
        source = source[None, :]
    else:
        require(source.shape == matrix.shape,
                f"a matched donor must have the features' shape {matrix.shape}, got {source.shape}")
    matrix[:, indices] = source[:, indices]
    return matrix


# ---------------------------------------------------------------------------
# preference pairs and the bounded DPO policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PreferencePairs:
    winners: object
    losers: object
    diagnostics: dict


def _strict_pair_universe(values):
    """An indexable (never materialised) numbering of every strict pair of ``values``.

    Sort the values. An item at sorted position ``i`` pairs with every item in a LATER outcome
    group, so it owns ``n - end_of_its_group`` strict partners; concatenating those blocks position
    by position numbers each unordered strict pair exactly once by an integer rank in ``[0, E)``.
    Memory is O(n) whatever ``E`` is, and a tie-heavy panel costs nothing extra.
    """
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    count = int(ordered.shape[0])
    cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    ends = np.concatenate([cuts, [count]])
    sizes = np.diff(np.concatenate([[0], ends]))
    group_end = np.repeat(ends, sizes)
    cumulative = np.cumsum(count - group_end, dtype=np.int64)
    return order, group_end, cumulative


def _pairs_from_ranks(ranks, universe):
    """``(lower-outcome index, higher-outcome index)`` into ``values`` for ranks in ``[0, E)``."""
    order, group_end, cumulative = universe
    ranks = np.asarray(ranks, dtype=np.int64)
    low = np.searchsorted(cumulative, ranks, side="right")
    before = np.where(low > 0, cumulative[low - 1], 0)
    high = group_end[low] + (ranks - before)
    return order[low], order[high]


def preference_pairs(utilities, supported, *, max_pairs, seed):
    """Strict-outcome preference pairs among positive-support labelled candidates.

    A pair needs two distinct outcomes (a true tie carries no preference) and positive reference
    support at both ends (``log q_θ/p`` is undefined where ``p = 0``). The pair count is capped at
    ``max_pairs``. When more strict pairs exist, ``max_pairs`` distinct integer ranks are drawn
    uniformly WITHOUT replacement from the strict-pair universe (``E`` ranks) and mapped through
    the sorted outcome-group blocks: there is no rejection loop, so a tie-heavy panel (999 tied
    labels plus one positive has only 999 strict pairs among 499,500 index pairs) cannot exhaust
    attempts, and no pair matrix is ever materialised. Indices refer to the input arrays. The
    diagnostics keep the ORIGINAL unique-label count.
    """
    u = _vector(utilities, "labelled utilities")
    flags = np.asarray(supported, dtype=bool)
    require(flags.shape == u.shape, "supported flags must align with the utilities")
    max_pairs = _integer(max_pairs, "max_pairs", minimum=1)
    seed = _integer(seed, "seed", minimum=0)
    labelled = int(u.shape[0])
    usable = np.flatnonzero(flags)
    count = int(usable.shape[0])
    values = u[usable]
    _, group_sizes = np.unique(values, return_counts=True)
    ties = int(sum(int(size) * (int(size) - 1) // 2 for size in group_sizes))
    eligible = count * (count - 1) // 2 - ties
    diagnostics = {"labelled_candidates": labelled, "positive_support_labels": count,
                   "excluded_zero_prior_labels": labelled - count,
                   "eligible_pairs": int(eligible), "ties_excluded": ties, "max_pairs": max_pairs,
                   "seed": seed, "stream": "preference_pairs"}
    empty = np.empty(0, dtype=np.int64)
    if eligible == 0:
        diagnostics.update(pairs_used=0, subsampled=False, method="none")
        return PreferencePairs(winners=_frozen(empty, np.int64), losers=_frozen(empty, np.int64),
                               diagnostics=diagnostics)
    rng = np.random.default_rng([seed, STREAM_PAIRS])
    universe = _strict_pair_universe(values)
    if eligible <= max_pairs:
        ranks, method = np.arange(eligible, dtype=np.int64), "all_eligible_pairs"
    else:
        ranks = np.sort(rng.choice(eligible, size=max_pairs, replace=False))
        method = "seeded_uniform_rank_sample"
    losers, winners = _pairs_from_ranks(ranks, universe)
    diagnostics.update(pairs_used=int(winners.shape[0]), subsampled=bool(eligible > max_pairs),
                       method=method)
    return PreferencePairs(winners=_frozen(usable[winners], np.int64),
                           losers=_frozen(usable[losers], np.int64), diagnostics=diagnostics)


@dataclass(frozen=True, eq=False)
class DPOPolicy:
    """``q_θ ∝ p · exp(θᵀφ)``. ``θᵀφ`` is an implicit energy, not a calibrated utility."""

    space: FeatureSpace
    theta: object
    beta: float
    l2: float
    diagnostics: dict

    def energy(self, features):
        return _feature_matrix(features, self.space.dimension, name="features") @ self.theta

    def document(self):
        return {"status": "fit", "theta": self.theta.tolist(), "beta": self.beta, "l2": self.l2,
                "diagnostics": copy.deepcopy(self.diagnostics),
                "implicit_energy": ("theta . phi: defined up to a per-pool constant and not a "
                                    "calibrated measured utility")}


def fit_dpo(features, winners, losers, space, *, beta=1.0, l2=1e-2, max_iterations=100,
            tolerance=1e-9):
    """Fit ``θ`` for ``q_θ ∝ p exp(θᵀφ)`` by damped Newton on the regularised DPO objective.

    For a pair sharing one condition and positive reference support, the DPO margin is
    ``β θᵀ(φ_win − φ_lose)`` (the log-normaliser cancels). The objective is the MEAN pair logistic
    loss ``log(1 + exp(-margin))`` plus ``λ‖θ‖²/2`` with ``λ > 0`` (strictly convex, no intercept).
    The optimisation takes real, counted Newton steps from ``θ = 0`` with a backtracking line
    search, and the diagnostics record the initial and final loss, the stop reason and whether the
    gradient tolerance was met.

    Label-use instrumentation (explicit convention): every objective, gradient and Hessian
    evaluation, and the one-off construction of the difference matrix, is a FULL PASS that presents
    all ``2 * pairs`` pair endpoints once. The calls are counted, so repeated consumption of a
    label by the optimiser is reported as USES (``pair_endpoint_presentations``), never as new
    acquired labels.
    """
    require(isinstance(space, FeatureSpace), "fit_dpo needs the FeatureSpace the columns came from")
    beta = _real(beta, "DPO beta", positive=True)
    l2 = _real(l2, "DPO l2", positive=True)
    max_iterations = _integer(max_iterations, "max_iterations", minimum=1)
    tolerance = _real(tolerance, "tolerance", positive=True)
    _require_dense_system(space.dimension, "DPO Newton Hessian system")
    matrix = _feature_matrix(features, space.dimension, name="training features")
    win = np.asarray(winners, dtype=np.int64)
    lose = np.asarray(losers, dtype=np.int64)
    require(win.ndim == 1 and win.shape == lose.shape and win.shape[0] >= 1,
            "fit_dpo needs at least one preference pair")
    require(int(min(win.min(), lose.min())) >= 0
            and int(max(win.max(), lose.max())) < matrix.shape[0],
            "a preference pair indexes a row outside the training features")
    require(not bool(np.any(win == lose)), "a preference pair compares a candidate with itself")
    pairs, dimension = int(win.shape[0]), space.dimension
    require(pairs * dimension <= MAX_PAIR_MATRIX_ELEMENTS,
            f"{pairs} pairs x {dimension} features exceeds the {MAX_PAIR_MATRIX_ELEMENTS}-element "
            "bound on the pair-difference matrix; lower max_pairs")
    delta = matrix[win] - matrix[lose]
    calls = {"objective": 0, "gradient": 0, "hessian": 0, "line_search_trials": 0}

    def evaluate(theta):
        calls["objective"] += 1
        margin = beta * (delta @ theta)
        loss = float(np.mean(np.logaddexp(0.0, -margin)) + 0.5 * l2 * float(theta @ theta))
        return margin, loss

    def gradient_of(theta, margin):
        calls["gradient"] += 1
        pull = np.exp(-np.logaddexp(0.0, margin))
        return -beta * (delta.T @ pull) / pairs + l2 * theta

    theta = np.zeros(dimension)
    margin, loss = evaluate(theta)
    initial_loss, history, updates, stop = loss, [loss], 0, "max_iterations"
    for _ in range(max_iterations):
        gradient = gradient_of(theta, margin)
        norm = float(np.linalg.norm(gradient))
        if norm <= tolerance:
            stop = "gradient_tolerance"
            break
        calls["hessian"] += 1
        curvature = np.exp(-np.logaddexp(0.0, margin) - np.logaddexp(0.0, -margin))
        hessian = (beta ** 2 / pairs) * ((delta.T * curvature) @ delta) + l2 * np.eye(dimension)
        step = np.linalg.solve(hessian, gradient)
        if norm >= 1e-6:
            descent, size, accepted = float(gradient @ step), 1.0, False
            while size >= 1e-12:
                candidate = theta - size * step
                calls["line_search_trials"] += 1
                candidate_margin, candidate_loss = evaluate(candidate)
                if candidate_loss <= loss - 1e-4 * size * descent:
                    accepted = True
                    break
                size *= 0.5
            if not accepted:
                stop = "line_search_failed"
                break
        else:
            # Near the optimum the Armijo test is below float resolution; the full Newton step is
            # then the quadratically convergent one.
            candidate = theta - step
            candidate_margin, candidate_loss = evaluate(candidate)
        theta, margin, loss = candidate, candidate_margin, candidate_loss
        updates += 1
        history.append(loss)
    final_norm = float(np.linalg.norm(gradient_of(theta, margin)))
    converged = final_norm <= tolerance
    if converged:
        stop = "gradient_tolerance"
    passes = 1 + calls["objective"] + calls["gradient"] + calls["hessian"]
    diagnostics = {
        "objective": ("mean pair log(1+exp(-beta*theta.delta)) + l2*||theta||^2/2, no intercept"),
        "pairs": pairs, "beta": beta, "l2": l2, "initial_loss": initial_loss, "final_loss": loss,
        "loss_decreased": bool(loss < initial_loss), "updates": updates, "stop_reason": stop,
        "converged": bool(converged), "final_gradient_norm": final_norm,
        "theta_norm": float(np.linalg.norm(theta)), "max_iterations": max_iterations,
        "tolerance": tolerance, "loss_history": history,
        "evaluations": {"difference_matrix_builds": 1, "objective": calls["objective"],
                        "gradient": calls["gradient"], "hessian": calls["hessian"],
                        "line_search_trial_objectives": calls["line_search_trials"]},
        "full_pair_passes": passes,
        "pair_endpoint_presentations": 2 * pairs * passes,
        "use_convention": ("one full pass (the difference-matrix build and each objective, "
                           "gradient or Hessian evaluation) presents every endpoint of every "
                           "used pair once; these are label USES, not acquired labels")}
    return DPOPolicy(space=space.snapshot(), theta=_frozen(theta), beta=beta, l2=l2,
                     diagnostics=diagnostics)


def pressure_name(pressure):
    """A LOSSLESS policy name: ``repr`` round-trips the float, so 1.0 and 1.0000001 never collide.

    A whole number drops its ``.0`` (``dpo_pressure_1``); ``:g`` would round both to ``1``.
    """
    text = repr(float(pressure))
    if text.endswith(".0"):
        text = text[:-2]
    return f"dpo_pressure_{text}"


# ---------------------------------------------------------------------------
# sampling-importance-resampling over the finite support
# ---------------------------------------------------------------------------

def sir_draw_counts(log_proposal, draws, rng):
    """Occurrence counts ``n_i`` of ``draws`` i.i.d. draws from the proposal. Multiplicity kept."""
    log_r = normalize_log_weights(log_proposal, name="proposal")
    probabilities = np.exp(log_r)
    probabilities = probabilities / probabilities.sum()
    return rng.multinomial(_integer(draws, "draws", minimum=1), probabilities)


def sir_weighted_distribution(log_prior, log_proposal, predicted_utilities, tau, counts):
    """The weighted proposal law of SIR and its occurrence ESS.

    Mass is ``∝ n_i · (p_i / r_i) · exp(û_i / τ)``: each of the K draws carries the importance
    weight of the candidate it landed on, so duplicates add. This is the law the final M
    resamples are drawn from, NOT the final histogram. The occurrence ESS is
    ``(Σ n_i w_i)² / Σ n_i w_i²`` -- it is not the ESS of the collapsed candidate masses -- and it
    is reported beside the distinct-candidate count. The proposal must cover the target's positive
    support, and the total sampled weight must be positive.

    A constant added to every predicted utility, to the prior or to the proposal changes nothing.
    Each of the three is centred on its own over the sampled, reachable ("live") candidates BEFORE
    they meet, so a huge common offset (a rare proposal at ``-1e16``) cannot round ``log p`` away
    from the ratio; the finite importance weights are then rebased to a maximum of zero BEFORE
    ``log n`` is added, so the multiplicity enters at full precision; and the sums are centred at
    the largest weighted mass. Centring late would be too late: rounding has already erased the
    information. A finite candidate whose weight cannot be represented relative to the others is
    refused, never silently given zero mass.
    """
    tau = _real(tau, "tau", positive=True)
    # Validated but NOT normalised here: the law and the ESS are unchanged by a constant over the
    # live candidates, and normalising first would subtract an unsampled candidate's mass from the
    # live ones and round their ratio away. They are centred on the live candidates below instead.
    log_p = _check_log_weights(log_prior, "prior")
    size = log_p.shape[0]
    log_r = _check_log_weights(log_proposal, "proposal")
    require(log_r.shape == log_p.shape, "the proposal and the prior must share one support")
    u = _vector(predicted_utilities, "predicted utilities", length=size)
    n_i = np.asarray(counts)
    require(n_i.shape == (size,) and np.issubdtype(n_i.dtype, np.integer) and bool((n_i >= 0).all()),
            "counts must be a non-negative integer occurrence count per candidate")
    draws = int(n_i.sum())
    require(draws >= 1, "SIR needs at least one proposal draw")
    uncovered = (log_p > -np.inf) & (log_r == -np.inf)
    require(not bool(uncovered.any()),
            f"the proposal gives zero mass to {int(uncovered.sum())} candidate(s) the target can "
            "reach; SIR would silently renormalise onto the covered part")
    require(not bool(np.any((n_i > 0) & (log_r == -np.inf))),
            "counts name candidates the proposal cannot draw")
    live = (n_i > 0) & (log_p > -np.inf)
    require(bool(live.any()),
            "every sampled candidate has zero weight (zero total sampled weight); SIR has "
            "nothing to resample")
    # The utilities, the prior and the proposal are each centred over the candidates that can carry
    # weight BEFORE they meet: ``log p - log r`` is unchanged by a common shift of either, but a
    # shift of 1e16 rounds log 0.25 and log 0.75 to the same float, and nothing downstream can
    # recover that. The three shifts add to one constant over the live candidates, which cancels
    # in the law and in the ESS.
    rebased = _rebased_on_mask(u, live, "predicted utilities")
    centred_p = _rebased_on_mask(log_p, live, "the prior log weights")
    centred_r = _rebased_on_mask(log_r, live, "the proposal log weights")
    with np.errstate(invalid="ignore", over="ignore"):
        raw = np.where(live, centred_p - centred_r + rebased / tau, -np.inf)
    require(bool(np.isfinite(raw[live]).all()),
            "the SIR importance weights are not representable (predicted utilities / tau "
            "overflowed); choose a larger tau")
    # Rebase the finite importance weights to a maximum of zero BEFORE adding log n. Adding the
    # multiplicity first would be absorbed by a large weight whenever the weights ride a big offset.
    log_w = np.where(live, _rebased_on_mask(raw, live, "the SIR importance weights"), -np.inf)
    log_n = np.where(n_i > 0, np.log(np.maximum(n_i, 1)), -np.inf)
    weighted = log_n + log_w
    # Centre at the largest weighted mass before summing: the law is ``weighted - total`` and the
    # ESS is ``exp(2 log S1 - log S2)``, both unchanged by a common shift, but the shift keeps every
    # exponent <= 0 so neither log-sum loses the small terms to a large one. Because ``log n >= 0``
    # and ``log w <= 0`` on a live candidate, ``log w - top <= 0`` as well.
    top = float(weighted.max())
    centered = weighted - top
    relative = log_w - top
    require(bool(np.isfinite(centered[live]).all() and np.isfinite(relative[live]).all()),
            "the SIR weights span more than a float64 can hold relative to the largest")
    total = logsumexp(centered)
    second_moment = logsumexp(log_n + 2.0 * relative)
    ess = math.exp(2.0 * total - second_moment)
    info = {"effective_sample_size": ess, "proposal_draws": draws,
            "distinct_candidates": int((n_i > 0).sum()),
            "positive_weight_distinct_candidates": int(((n_i > 0) & (log_w > -np.inf)).sum())}
    return _frozen(centered - total), info


# ---------------------------------------------------------------------------
# configuration, the problem, fit and allocate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineConfig:
    """Every seed and hyperparameter. All of it is hashed into the frozen artifact."""

    tau: float
    seed: int = 0
    ridge_alpha: float = 1.0
    label_budget: object = None
    declared_validation_labels: int = 0
    dpo_beta: float = 1.0
    dpo_l2: float = 1e-2
    dpo_max_pairs: int = 20000
    dpo_max_iterations: int = 100
    dpo_tolerance: float = 1e-9
    dpo_pressures: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    sir_proposal: str = SIR_PRIOR
    sir_proposal_draws: int = 256
    final_draws: int = 64

    def __post_init__(self):
        object.__setattr__(self, "tau", _real(self.tau, "tau", positive=True))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        object.__setattr__(self, "ridge_alpha", _real(self.ridge_alpha, "ridge alpha", positive=True))
        if self.label_budget is not None:
            object.__setattr__(self, "label_budget",
                               _integer(self.label_budget, "label_budget", minimum=1))
        object.__setattr__(self, "declared_validation_labels",
                           _integer(self.declared_validation_labels, "declared_validation_labels"))
        object.__setattr__(self, "dpo_beta", _real(self.dpo_beta, "dpo_beta", positive=True))
        object.__setattr__(self, "dpo_l2", _real(self.dpo_l2, "dpo_l2", positive=True))
        object.__setattr__(self, "dpo_max_pairs", _integer(self.dpo_max_pairs, "dpo_max_pairs", minimum=1))
        object.__setattr__(self, "dpo_max_iterations",
                           _integer(self.dpo_max_iterations, "dpo_max_iterations", minimum=1))
        object.__setattr__(self, "dpo_tolerance", _real(self.dpo_tolerance, "dpo_tolerance", positive=True))
        pressures = tuple(_real(item, "a DPO pressure", positive=True) for item in self.dpo_pressures)
        require(len(set(pressures)) == len(pressures), "DPO pressures must be distinct")
        object.__setattr__(self, "dpo_pressures", pressures)
        require(self.sir_proposal in SIR_PROPOSALS, f"sir_proposal must be one of {SIR_PROPOSALS}")
        object.__setattr__(self, "sir_proposal_draws",
                           _integer(self.sir_proposal_draws, "sir_proposal_draws", minimum=1))
        object.__setattr__(self, "final_draws", _integer(self.final_draws, "final_draws", minimum=1))

    def document(self):
        return {"tau": self.tau, "seed": self.seed, "ridge_alpha": self.ridge_alpha,
                "label_budget": self.label_budget,
                "declared_validation_labels": self.declared_validation_labels,
                "dpo": {"beta": self.dpo_beta, "l2": self.dpo_l2, "max_pairs": self.dpo_max_pairs,
                        "max_iterations": self.dpo_max_iterations,
                        "tolerance": self.dpo_tolerance, "pressures": list(self.dpo_pressures)},
                "sir": {"proposal": self.sir_proposal,
                        "proposal_draws": self.sir_proposal_draws,
                        "final_draws": self.final_draws},
                "random_streams": {name: {"seed": self.seed, "stream": stream,
                                          "generator": "numpy default_rng([seed, stream])"}
                                   for name, stream in STREAMS.items()},
                "tuning": ("none: alpha, DPO beta/l2, tau and every sampler setting are fixed "
                           "inputs; no labels were held out to tune them")}


def _canonical_sequences(sequences, name):
    sequences = tuple(canonical_sequence(item) for item in sequences)
    require(all(sequences), f"{name} must be non-empty")
    return sequences


@dataclass(frozen=True, eq=False)
class FiniteSupportProblem:
    """Everything a fit and an allocation may see: NO evaluation outcomes.

    ``train_*`` hold the labelled, OFFERED training candidates with their outcomes; ``support_*``
    are the held-out candidates that receive allocation mass. Train rows fit predictors and
    policies but never receive allocation mass; the support never carries an outcome. Rows are
    stored in canonical (sorted by id) order, so permuting the input changes nothing, including the
    seeded draws.

    The problem owns SNAPSHOTS of its ``space`` and ``prior`` (deep copies, nested supervision
    entries included), so editing the caller's objects afterwards cannot rewrite what a fit saw.
    ``binding()`` summarises every input by compact content digests, ``fingerprint()`` hashes that
    summary, and ``fit_stage`` / ``freeze_artifact`` compare it, so a problem edited after its fit
    (even through ``problem.space``) is refused rather than frozen under the old provenance.
    ``unlabelled_train_ids`` and ``candidate_roster`` carry the unlabelled training rows and the
    whole candidate-table digest when the CSV pipeline built the problem.
    """

    target: str
    assay: str
    utility: UtilitySpec
    grouping_rationale: str
    space: FeatureSpace
    prior: PriorSpec
    train_ids: tuple
    train_features: object
    train_utilities: object
    train_log_prior: object
    train_groups: tuple
    support_ids: tuple
    support_features: object
    support_log_prior: object
    support_groups: tuple
    train_sequences: object = None
    support_sequences: object = None
    unlabelled_train_candidates: int = 0
    unlabelled_train_ids: tuple = ()
    candidate_roster: object = None

    def __post_init__(self):
        for name in ("target", "assay", "grouping_rationale"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        require(isinstance(self.utility, UtilitySpec), "utility must be a UtilitySpec")
        require(isinstance(self.space, FeatureSpace), "space must be a FeatureSpace")
        require(isinstance(self.prior, PriorSpec), "prior must be a PriorSpec")
        train_ids = _unique_ids(self.train_ids, "training candidate ids")
        support_ids = _unique_ids(self.support_ids, "held-out candidate ids")
        require(train_ids and support_ids, "need at least one labelled training and one held-out candidate")
        shared = sorted(set(train_ids) & set(support_ids))
        require(not shared, f"train/evaluation overlap in candidate ids, e.g. {shared[:3]}")
        d = self.space.dimension
        x_train = _feature_matrix(self.train_features, d, name="training features")
        x_support = _feature_matrix(self.support_features, d, name="held-out features")
        require(x_train.shape[0] == len(train_ids) and x_support.shape[0] == len(support_ids),
                "feature rows must align with the candidate ids")
        y = _vector(self.train_utilities, "training utilities", length=len(train_ids))
        p_train = _check_log_weights(self.train_log_prior, "training prior_log_weight",
                                     length=len(train_ids), allow_no_mass=True)
        p_support = _check_log_weights(self.support_log_prior, "held-out prior_log_weight",
                                       length=len(support_ids))
        validate_prior_log_weights(self.prior.kind, np.concatenate([p_train, p_support]))
        train_groups = tuple(_text(g, "a group id") for g in self.train_groups)
        support_groups = tuple(_text(g, "a group id") for g in self.support_groups)
        require(len(train_groups) == len(train_ids) and len(support_groups) == len(support_ids),
                "group ids must align with the candidate ids")
        crossing = sorted(set(train_groups) & set(support_groups))
        require(not crossing,
                f"train and evaluation candidates share biological group(s) {crossing[:3]}; the "
                "grouping rationale is violated and the held-out score would be leakage")
        train_seq = support_seq = None
        if self.train_sequences is not None or self.support_sequences is not None:
            require(self.train_sequences is not None and self.support_sequences is not None,
                    "supply sequences for both the training and the held-out candidates, or neither")
            train_seq = _canonical_sequences(self.train_sequences, "training sequences")
            support_seq = _canonical_sequences(self.support_sequences, "held-out sequences")
            require(len(train_seq) == len(train_ids) and len(support_seq) == len(support_ids),
                    "sequences must align with the candidate ids")
            require(len(set(train_seq)) == len(train_seq) and len(set(support_seq)) == len(support_seq),
                    "duplicate physical sequences among the candidates")
            twins = sorted(set(train_seq) & set(support_seq))
            require(not twins,
                    f"the same physical sequence is both a training and an evaluation candidate, "
                    f"e.g. {twins[:2]}")
        order_t = sorted(range(len(train_ids)), key=lambda i: train_ids[i])
        order_s = sorted(range(len(support_ids)), key=lambda i: support_ids[i])
        object.__setattr__(self, "train_ids", tuple(train_ids[i] for i in order_t))
        object.__setattr__(self, "support_ids", tuple(support_ids[i] for i in order_s))
        object.__setattr__(self, "train_features", _frozen(x_train[order_t]))
        object.__setattr__(self, "support_features", _frozen(x_support[order_s]))
        object.__setattr__(self, "train_utilities", _frozen(y[order_t]))
        object.__setattr__(self, "train_log_prior", _frozen(p_train[order_t]))
        object.__setattr__(self, "support_log_prior", _frozen(p_support[order_s]))
        object.__setattr__(self, "train_groups", tuple(train_groups[i] for i in order_t))
        object.__setattr__(self, "support_groups", tuple(support_groups[i] for i in order_s))
        if train_seq is not None:
            object.__setattr__(self, "train_sequences", tuple(train_seq[i] for i in order_t))
            object.__setattr__(self, "support_sequences", tuple(support_seq[i] for i in order_s))
        unlabelled = tuple(self.unlabelled_train_ids)
        require(all(isinstance(item, str) and item for item in unlabelled)
                and len(set(unlabelled)) == len(unlabelled),
                "unlabelled training ids must be unique non-empty strings")
        clash = sorted(set(unlabelled) & (set(train_ids) | set(support_ids)))
        require(not clash, f"unlabelled training candidate(s) {clash[:3]} are also labelled or "
                           "held out")
        unlabelled_count = _integer(self.unlabelled_train_candidates, "unlabelled_train_candidates")
        if unlabelled:
            require(unlabelled_count in (0, len(unlabelled)),
                    "unlabelled_train_candidates disagrees with the listed unlabelled ids")
            unlabelled_count = len(unlabelled)
        roster = self.candidate_roster
        if roster is not None:
            require(isinstance(roster, dict), "candidate_roster must be a mapping of digests")
            roster = _builtin(roster)
        object.__setattr__(self, "unlabelled_train_ids", tuple(sorted(unlabelled)))
        object.__setattr__(self, "unlabelled_train_candidates", unlabelled_count)
        object.__setattr__(self, "candidate_roster", roster)
        # Own snapshots: the caller's FeatureSpace/PriorSpec (and their nested supervision
        # entries) stay editable by the caller without reaching this problem.
        object.__setattr__(self, "space", self.space.snapshot())
        object.__setattr__(self, "prior", self.prior.snapshot())

    def manifest_document(self):
        return {"target": self.target, "assay": self.assay, "utility": self.utility.document(),
                "grouping_rationale": self.grouping_rationale}

    def binding(self):
        """Compact content digests of every input a fit or an allocation can read.

        Includes the WHOLE offered roster (selected or not): ids, outcomes, prior scores, groups,
        features and, when supplied, physical sequences, plus the unlabelled training rows and the
        candidate-table digest. Changing any unselected offered row changes this summary. It holds
        no evaluation outcome. A digest is integrity, not proof the inputs were truthful.
        """
        def sequences(values):
            return None if values is None else digest(list(values))
        return {
            "manifest_sha256": digest(self.manifest_document()),
            "space_sha256": self.space.fingerprint(),
            "prior_spec_sha256": self.prior.fingerprint(),
            "train": {"count": len(self.train_ids), "ids_sha256": digest(list(self.train_ids)),
                      "features_sha256": _array_sha256(self.train_features),
                      "utilities_sha256": _array_sha256(self.train_utilities),
                      "prior_log_weight_sha256": _array_sha256(self.train_log_prior),
                      "groups_sha256": digest(list(self.train_groups)),
                      "sequences_sha256": sequences(self.train_sequences)},
            "support": {"count": len(self.support_ids), "ids_sha256": digest(list(self.support_ids)),
                        "features_sha256": _array_sha256(self.support_features),
                        "prior_log_weight_sha256": _array_sha256(self.support_log_prior),
                        "groups_sha256": digest(list(self.support_groups)),
                        "sequences_sha256": sequences(self.support_sequences)},
            "unlabelled_train": {"count": self.unlabelled_train_candidates,
                                 "ids_sha256": digest(list(self.unlabelled_train_ids))},
            "candidate_roster": copy.deepcopy(self.candidate_roster)}

    def fingerprint(self):
        return digest(self.binding())


def acquire_labels(offered, budget, seed):
    """Sorted indices of the labels acquired out of ``offered``: seeded, evaluation-free.

    NESTED by construction. One seeded permutation of the ``offered`` canonical positions is drawn
    independently of the budget and the budget only sets how long a prefix is sliced (then sorted).
    For a fixed roster and seed, the labels acquired at budget ``a <= b`` are therefore a subset of
    those acquired at ``b``, which a label-efficiency curve needs. ``budget=None`` takes every
    offered label.
    """
    offered = _integer(offered, "offered labels", minimum=1)
    if budget is None:
        return np.arange(offered)
    budget = _integer(budget, "label_budget", minimum=1)
    require(budget <= offered,
            f"label_budget={budget} exceeds the {offered} labels offered; the baseline will not "
            "invent labels")
    rng = np.random.default_rng([_integer(seed, "seed", minimum=0), STREAM_BUDGET])
    return np.sort(rng.permutation(offered)[:budget])


@dataclass(frozen=True, eq=False)
class FitResult:
    """The fit, bound to the problem and configuration it was made from.

    ``problem_sha256`` / ``config_sha256`` are taken when the fit starts; ``freeze_artifact``
    refuses a fit whose problem or configuration no longer matches them.
    """

    acquired: object
    reward: RewardModel
    predictions: object
    pairs: PreferencePairs
    dpo: object
    dpo_unsupported_reason: object
    dpo_energy: object
    problem_sha256: str = ""
    config_sha256: str = ""


def fit_stage(problem, config):
    """Stage 1: training features and outcomes only. No evaluation outcome exists in scope."""
    require(isinstance(problem, FiniteSupportProblem) and isinstance(config, BaselineConfig),
            "fit_stage needs a FiniteSupportProblem and a BaselineConfig")
    problem_sha256, config_sha256 = problem.fingerprint(), digest(config.document())
    acquired = acquire_labels(len(problem.train_ids), config.label_budget, config.seed)
    x = problem.train_features[acquired]
    y = problem.train_utilities[acquired]
    reward = fit_reward(x, y, problem.space, alpha=config.ridge_alpha)
    predictions = _frozen(reward.predict(problem.support_features))
    supported = np.isfinite(problem.train_log_prior[acquired])
    pairs = preference_pairs(y, supported, max_pairs=config.dpo_max_pairs, seed=config.seed)
    dpo, reason, energy = None, None, None
    if pairs.winners.shape[0] == 0:
        diag = pairs.diagnostics
        reason = (f"no usable preference pair: {diag['positive_support_labels']} of "
                  f"{diag['labelled_candidates']} acquired labels have positive reference support "
                  f"({diag['excluded_zero_prior_labels']} zero-prior excluded) and "
                  f"{diag['ties_excluded']} support-eligible pairs are outcome ties")
    else:
        dpo = fit_dpo(x, pairs.winners, pairs.losers, problem.space, beta=config.dpo_beta,
                      l2=config.dpo_l2, max_iterations=config.dpo_max_iterations,
                      tolerance=config.dpo_tolerance)
        energy = _frozen(dpo.energy(problem.support_features))
    return FitResult(acquired=_frozen(acquired, np.int64), reward=reward, predictions=predictions,
                     pairs=pairs, dpo=dpo, dpo_unsupported_reason=reason, dpo_energy=energy,
                     problem_sha256=problem_sha256, config_sha256=config_sha256)


@dataclass(frozen=True, eq=False)
class Allocation:
    support_ids: tuple
    log_prior: object
    policies: dict
    sir: dict


def allocate(support_ids, log_prior, predictions, config, *, dpo_energy=None,
             proposal_log_weights=None):
    """Stage 2: allocate mass from FROZEN predictions. There is no outcome argument.

    Candidates must arrive in canonical (sorted) order so the seeded draws are order-invariant.
    Returns the prior, the exact learned tilt, the DPO tilts at each pressure, and the SIR weighted
    proposal law with its resource accounting.
    """
    require(isinstance(config, BaselineConfig), "allocate needs a BaselineConfig")
    ids = _unique_ids(support_ids, "support ids")
    require(list(ids) == sorted(ids),
            "support ids must be in canonical order (sorted); allocate's seeded draws are "
            "defined on that order")
    size = len(ids)
    log_p = normalize_log_weights(log_prior, name="declared prior")
    require(log_p.shape == (size,), "the prior must align with the support ids")
    uhat = _vector(predictions, "predicted utilities", length=size)
    policies = {"prior": log_p, "ridge_tilt": log_tilt(log_p, uhat, config.tau)}
    if dpo_energy is not None:
        energy = _vector(dpo_energy, "DPO energies", length=size)
        for pressure in config.dpo_pressures:
            policies[pressure_name(pressure)] = dpo_log_policy(
                log_p, energy, pressure, name=pressure_name(pressure))
    require((proposal_log_weights is not None) == (config.sir_proposal == SIR_DECLARED),
            "pass proposal_log_weights if and only if sir_proposal is 'declared'")
    if config.sir_proposal == SIR_PRIOR:
        log_r = log_p.copy()
    elif config.sir_proposal == SIR_UNIFORM:
        reachable = log_p > -np.inf
        log_r = np.where(reachable, -math.log(int(reachable.sum())), -np.inf)
    else:
        log_r = normalize_log_weights(proposal_log_weights, name="declared proposal")
        require(log_r.shape == (size,), "the declared proposal must align with the support")
    counts = sir_draw_counts(log_r, config.sir_proposal_draws,
                             np.random.default_rng([config.seed, STREAM_PROPOSAL]))
    weighted, info = sir_weighted_distribution(log_p, log_r, uhat, config.tau, counts)
    histogram = sir_draw_counts(weighted, config.final_draws,
                                np.random.default_rng([config.seed, STREAM_RESAMPLE]))
    expected_queries = float(expected_distinct(np.exp(log_r), config.sir_proposal_draws).sum())
    sir = {"proposal_kind": config.sir_proposal, "log_proposal": _frozen(log_r),
           "proposal_draws": int(config.sir_proposal_draws), "final_draws": int(config.final_draws),
           "counts": _frozen(counts, np.int64), "final_histogram": _frozen(histogram, np.int64),
           "weighted_log_probability": weighted,
           "effective_sample_size": info["effective_sample_size"],
           "distinct_proposed": info["distinct_candidates"],
           "positive_weight_distinct": info["positive_weight_distinct_candidates"],
           "resource_accounting": {
               "proposal_draws_K": int(config.sir_proposal_draws),
               "final_draws_M": int(config.final_draws),
               "full_support_prediction_calls": int(size),
               "unique_candidates_queried_by_sampler": info["distinct_candidates"],
               "expected_unique_queries_with_caching": expected_queries,
               "final_resample_new_prediction_calls": 0,
               "scores_precomputed_for_full_support": True,
               "scoring_cost_saving_claimed": False,
               "note": ("every held-out candidate was scored offline to build the exact tilt, so "
                        "SIR saves no scoring cost here. A lazily scored sampler would query "
                        "about expected_unique_queries_with_caching distinct candidates, and the "
                        "final resampling makes no new prediction calls.")}}
    policies["sir_weighted_proposal"] = weighted
    return Allocation(support_ids=ids, log_prior=_frozen(log_p),
                      policies={name: _frozen(value) for name, value in policies.items()}, sir=sir)


# ---------------------------------------------------------------------------
# label accounting and the clean low-N verdict
# ---------------------------------------------------------------------------

def _selection_record(config):
    return {"method": ("the first N entries of ONE seeded permutation of the offered labels "
                       "(then sorted): budgets are NESTED for a fixed roster and seed"
                       if config.label_budget is not None else "all offered labels"),
            "label_budget": config.label_budget, "seed": config.seed, "stream": "label_budget",
            "evaluation_outcomes_used": False}


def _label_uses(fit, acquired):
    """Repeated label USES, counted under the optimiser's explicit convention, beside acquisition.

    Ridge reads each acquired label once (the first use of every label). The DPO optimiser then
    presents each used pair's two endpoints once per full pass over the pairs, and ``fit_dpo``
    counts those passes, so a label is NOT free to reuse: the uses are reported, and they are
    never added into the independent-label count.
    """
    if fit.dpo is None:
        dpo = {"status": "not_fit", "pairs": 0, "full_pair_passes": 0,
               "pair_endpoint_presentations": 0}
    else:
        diagnostics = fit.dpo.diagnostics
        dpo = {"status": "instrumented", "pairs": int(diagnostics["pairs"]),
               "full_pair_passes": int(diagnostics["full_pair_passes"]),
               "pair_endpoint_presentations": int(diagnostics["pair_endpoint_presentations"]),
               "evaluations": copy.deepcopy(diagnostics["evaluations"]),
               "convention": diagnostics["use_convention"]}
    repeated = dpo["pair_endpoint_presentations"]
    return {"convention": ("a USE is one acquired label read by one computation. Ridge: one "
                           "closed-form pass reads each acquired label once. DPO: each full pair "
                           "pass presents both endpoints of every used pair once. Uses are not "
                           "acquisition and are never added to the label budget."),
            "independent_labels": acquired, "ridge_label_uses": acquired, "dpo": dpo,
            "total_label_uses": acquired + repeated, "repeated_uses_beyond_first": repeated}


def label_accounting(problem, config, fit):
    offered, acquired = len(problem.train_ids), int(fit.acquired.shape[0])
    acquired_ids = [problem.train_ids[i] for i in fit.acquired.tolist()]
    uses = _label_uses(fit, acquired)
    return {
        "offered_labels": offered, "acquired_labels": acquired,
        "unique_acquired_candidates": acquired,
        "offered_ids_sha256": digest(list(problem.train_ids)),
        "acquired_ids_sha256": digest(acquired_ids),
        "label_uses": uses,
        "repeated_presentations": uses["repeated_uses_beyond_first"],
        "unlabelled_train_candidates": int(problem.unlabelled_train_candidates),
        "declared_validation_labels": config.declared_validation_labels,
        "validation_cost": {
            "declared_labels": config.declared_validation_labels, "ids_supplied": False,
            "disjointness_from_acquired_asserted": False,
            "note": ("reported separately and never added to the acquired count: without "
                     "validation label ids their overlap with the acquired, historical or "
                     "evaluation labels is unknown")},
        "selection": _selection_record(config),
        "expanded_preference_pairs": int(fit.pairs.diagnostics["pairs_used"]),
        "counting_rule": ("acquired labels are tracked by identity and are one budget; preference "
                          "pairs expanded from them, repeated label USES and the declared "
                          "validation labels are counted separately and never added into it; "
                          "historical supervision is accounted in clean_low_n.label_cost")}


def _unique_label_total(components, *, history_unknown):
    """Unique labels over components ``(count, ids or None)`` that share one label namespace.

    Exact only when every component names its ids (the union is counted), or when exactly one
    component is non-empty and hides its ids (it IS the set). A component that declares a count
    without ids can overlap the others by an unknown amount: the total is then only bounded
    (``max(known union, largest hidden count)`` to ``known union + hidden counts``). An unknown
    history is unbounded above, so only the known union is a lower bound.
    """
    known, hidden = set(), []
    for count, ids in components:
        if ids is None:
            if count:
                hidden.append(int(count))
        else:
            known.update(ids)
    lower = max([len(known)] + hidden)
    upper = None if history_unknown else len(known) + sum(hidden)
    exact = None
    if not history_unknown:
        if not hidden:
            exact = len(known)
        elif len(hidden) == 1 and not known:
            exact = hidden[0]
    if exact is not None:
        status, basis = "exact", ("every component names its label ids (or one component is the "
                                  "whole set), so the union is counted")
        lower = upper = exact
    elif upper is not None:
        status, basis = "bounded_only", (
            "a component declares a label count without label ids, so its overlap with the other "
            "components cannot be excluded; the unique total lies between the bounds")
    else:
        status, basis = "lower_bound_only", (
            "a supervision history is unknown, so the labels it encodes are unbounded above; only "
            "the known union is a lower bound")
    return {"status": status, "exact": exact, "lower_bound": lower, "upper_bound": upper,
            "basis": basis}


def clean_low_n_eligibility(*, target, assay, feature_history, prior_history, train_ids,
                            support_ids, acquired_labels, acquired_ids=None,
                            declared_validation_labels=0):
    """True / False / None (unavailable) for a clean low-N or acquisition-replay claim.

    False for supervision on the same target+assay (the effective label count is the historical
    count plus the new labels, not the new labels alone) or whose listed ids overlap the training
    or evaluation candidates. None when a history is unknown, or documented without ids so overlap
    cannot be checked. Sequence-only exposure is reported beside it and does not move the verdict.

    ``label_cost`` separates three things the old single sum conflated: per-component declared
    USES (acquired, feature history, prior history; never added together, because one label can sit
    in several), the UNIQUE same-target total (exact only where identities are supplied for every
    component, otherwise bounds or unavailable) and the declared validation cost (disjointness is
    not asserted without validation ids). ``unknown_history_present`` reports genuinely unknown
    histories even when a known overlap already makes the verdict False. ``acquired_ids`` makes the
    acquisition a tracked identity set; without it only its count is known.
    """
    train, support = set(train_ids), set(support_ids)
    acquired_count = _integer(acquired_labels, "acquired_labels", minimum=0)
    if acquired_ids is not None:
        acquired_ids = _unique_ids(acquired_ids, "acquired ids")
        require(len(acquired_ids) == acquired_count,
                f"{len(acquired_ids)} acquired ids were supplied for acquired_labels="
                f"{acquired_count}")
    validation = _integer(declared_validation_labels, "declared_validation_labels", minimum=0)
    violated, unavailable, notes = [], [], []
    unknown_sources, ids_hidden = [], []
    components = [(acquired_count, acquired_ids)]
    uses = {"feature_history_same_target": 0, "prior_history_same_target": 0,
            "feature_history_other_target_or_assay": 0, "prior_history_other_target_or_assay": 0}
    for source, key, history in (("feature extractor", "feature_history", feature_history),
                                 ("prior score", "prior_history", prior_history)):
        if history.status == HISTORY_UNKNOWN:
            unknown_sources.append(source)
            unavailable.append(f"the {source}'s supervision history is unknown, so labels it may "
                               "already encode cannot be excluded")
            continue
        if history.status != HISTORY_DOCUMENTED:
            continue
        for entry in history.documented:
            same = entry["target"] == target and entry["assay"] == assay
            if same:
                uses[f"{key}_same_target"] += entry["label_count"]
                components.append((entry["label_count"], entry["label_ids"]))
                if entry["label_ids"] is None and source not in ids_hidden:
                    ids_hidden.append(source)
                violated.append(f"the {source} was supervised on {entry['label_count']} labels of "
                                f"this very target/assay, so this is not a clean "
                                f"{acquired_count}-shot model")
            else:
                uses[f"{key}_other_target_or_assay"] += entry["label_count"]
            ids = entry["label_ids"]
            if ids is None:
                if not same:
                    unavailable.append(f"the {source}'s documented supervision lists no label ids, "
                                       "so overlap with the training or evaluation candidates "
                                       "cannot be checked")
                continue
            on_support, on_train = sorted(set(ids) & support), sorted(set(ids) & train)
            if on_support:
                violated.append(f"the {source}'s supervision includes {len(on_support)} evaluation "
                                f"candidate(s), e.g. {on_support[:2]}: leakage")
            if on_train:
                violated.append(f"the {source}'s supervision overlaps {len(on_train)} training "
                                f"label(s), e.g. {on_train[:2]}, so the same labels are paid for twice")
            if not same and not on_support and not on_train:
                notes.append(f"the {source} was supervised on another target/assay with no id "
                             "overlap (transfer, not a few-shot claim by itself)")
    verdict = False if violated else (None if unavailable else True)
    history_unknown = bool(unknown_sources)
    unique = _unique_label_total(components, history_unknown=history_unknown)
    with_validation = dict(unique) if validation == 0 else _unique_label_total(
        components + [(validation, None)], history_unknown=history_unknown)
    return {"clean_low_n_eligible": verdict, "acquisition_replay_eligible": verdict,
            "status": {True: "eligible", False: "ineligible", None: "unavailable"}[verdict],
            "reasons": violated + unavailable, "notes": notes,
            "label_cost": {
                "identity_convention": LABEL_IDENTITY_CONVENTION,
                "acquired_new_labels": acquired_count,
                "acquired_ids_supplied": acquired_ids is not None,
                "declared_label_uses": {
                    "acquired_current": acquired_count, **uses,
                    "note": ("per-component declared USES: a label used by several components "
                             "appears in each, so these are never summed into a label count")},
                "unique_same_target_labels": unique,
                "validation": {
                    "declared_labels": validation, "ids_supplied": False,
                    "disjointness_asserted": False,
                    "unique_same_target_labels_including_validation": with_validation,
                    "note": ("validation labels are a separate declared cost; without validation "
                             "label ids they are not asserted disjoint from the acquired or "
                             "historical labels, so the total including them is not exact")},
                "unknown_history_present": history_unknown,
                "unknown_history_sources": unknown_sources,
                "same_target_history_without_label_ids": ids_hidden},
            "sequence_only_exposure": {
                "features": {"status": feature_history.exposure_status,
                             "description": feature_history.exposure_description},
                "prior": {"status": prior_history.exposure_status,
                          "description": prior_history.exposure_description},
                "note": "recorded separately from label supervision; it does not by itself change "
                        "the verdict above"},
            "declared_not_verified": True}


# ---------------------------------------------------------------------------
# the frozen artifact
# ---------------------------------------------------------------------------

def _section_hashes(document):
    hashes = {name: digest(document[name]) for name in HASHED_SECTIONS}
    hashes["artifact_sha256"] = digest({"schema": document["schema"], "stage": document["stage"],
                                        "sections": {name: hashes[name]
                                                     for name in HASHED_SECTIONS}})
    return hashes


def freeze_artifact(problem, config, fit, allocation):
    """Stage 2 output: predictions, policy probabilities and provenance in one hashed document.

    Everything is copied into plain lists, so the artifact shares no memory with the objects that
    produced it. It contains no evaluation outcome and nothing derived from one.

    The fit must come from THIS problem and configuration: ``fit_stage`` recorded their
    fingerprints, and a problem edited after the fit (inputs or nested metadata) or a different
    configuration is refused rather than frozen under provenance the fit never saw. The complete
    offered roster, the acquired indices and the selection rule are bound: ``training.offered_ids``
    / ``acquired_indices`` / ``selection`` list the identities, and ``provenance.input_binding``
    carries compact digests of every offered value, prior score, group, feature and (when supplied)
    physical sequence, so an UNSELECTED offered row cannot change without changing the hashes.
    """
    require(allocation.support_ids == problem.support_ids,
            "the allocation was built for a different support than the problem")
    binding = problem.binding()
    require(fit.problem_sha256 == digest(binding),
            "the fit was made from different problem inputs, or the problem (including its nested "
            "space/prior metadata) was edited after the fit; refit before freezing")
    require(fit.config_sha256 == digest(config.document()),
            "the fit was made with a different BaselineConfig; refit before freezing")
    require(fit.reward.space.fingerprint() == problem.space.fingerprint(),
            "the fitted predictor was built on a different feature space than the problem's")
    acquired = fit.acquired
    acquired_ids = [problem.train_ids[i] for i in acquired.tolist()]
    reachable = int(np.isfinite(problem.support_log_prior).sum())
    prior_declared = encode_log(problem.support_log_prior)
    pool = np.concatenate([problem.train_log_prior, problem.support_log_prior])
    declared_mass = logsumexp(pool)
    document = {
        "schema": ARTIFACT_SCHEMA, "stage": "frozen",
        "manifest": problem.manifest_document(),
        "config": {**config.document(), "schema": ARTIFACT_SCHEMA},
        "features": {
            "space": problem.space.document(),
            "train_feature_sha256": digest(problem.train_features[acquired].tolist()),
            "support_feature_sha256": digest(problem.support_features.tolist())},
        "training": {
            "ids": acquired_ids,
            "utilities": problem.train_utilities[acquired].tolist(),
            "features": problem.train_features[acquired].tolist(),
            "prior_log_weight": encode_log(problem.train_log_prior[acquired]),
            "groups": [problem.train_groups[i] for i in acquired.tolist()],
            "offered_ids": list(problem.train_ids),
            "offered_order": "sorted by candidate_id (code-point order)",
            "acquired_indices": acquired.tolist(),
            "selection": _selection_record(config)},
        "prior": {
            "spec": problem.prior.document(), "log_weight": prior_declared,
            "log_probability": encode_log(allocation.log_prior),
            "candidate_count": len(problem.support_ids), "reachable_count": reachable,
            "structural_zero_count": len(problem.support_ids) - reachable,
            "declared_log_mass_over_labelled_pool": (None if declared_mass == -math.inf
                                                     else declared_mass),
            "finite_pool_note": ("the support is the finite held-out pool; the reference is its "
                                 "normalised restriction, not the native generator")},
        "support": {"ids": list(problem.support_ids),
                    "order": "sorted by candidate_id (code-point order)",
                    "count": len(problem.support_ids),
                    "identity_sha256": digest(list(problem.support_ids)),
                    "groups": list(problem.support_groups),
                    "sequences_sha256": (None if problem.support_sequences is None
                                         else digest(list(problem.support_sequences))),
                    "sequences_note": ("canonical physical sequences in support-id order, bound "
                                       "by digest when supplied; null when none were supplied")},
        "fitted": {
            "reward": fit.reward.document(),
            "pairs": copy.deepcopy(fit.pairs.diagnostics),
            "dpo": (fit.dpo.document() if fit.dpo is not None
                    else {"status": "unsupported", "reason": fit.dpo_unsupported_reason})},
        "predictions": {"utility": fit.predictions.tolist(),
                        "dpo_energy": (None if fit.dpo_energy is None
                                       else fit.dpo_energy.tolist())},
        "policies": {
            "log_probability": {name: encode_log(value)
                                for name, value in allocation.policies.items()},
            "sir": {"proposal_kind": allocation.sir["proposal_kind"],
                    "log_proposal": encode_log(allocation.sir["log_proposal"]),
                    "proposal_draws": allocation.sir["proposal_draws"],
                    "final_draws": allocation.sir["final_draws"],
                    "counts": allocation.sir["counts"].tolist(),
                    "final_histogram": allocation.sir["final_histogram"].tolist(),
                    "effective_sample_size": allocation.sir["effective_sample_size"],
                    "distinct_proposed": allocation.sir["distinct_proposed"],
                    "positive_weight_distinct": allocation.sir["positive_weight_distinct"],
                    "resource_accounting": copy.deepcopy(allocation.sir["resource_accounting"])}},
        "provenance": {
            "label_accounting": label_accounting(problem, config, fit),
            "split_counts": {"train_candidates_offered": len(problem.train_ids)
                             + int(problem.unlabelled_train_candidates),
                             "train_labelled_offered": len(problem.train_ids),
                             "train_labelled_acquired": int(acquired.shape[0]),
                             "evaluation_candidates": len(problem.support_ids),
                             "evaluation_reachable_support": reachable},
            "effective_data": {
                "ridge": {"labels_used": int(acquired.shape[0])},
                "dpo": {"labels_with_positive_support":
                        fit.pairs.diagnostics["positive_support_labels"],
                        "excluded_zero_prior_labels":
                        fit.pairs.diagnostics["excluded_zero_prior_labels"],
                        "eligible_pairs": fit.pairs.diagnostics["eligible_pairs"],
                        "pairs_used": fit.pairs.diagnostics["pairs_used"],
                        "note": ("the matched acquired/offered budget is shared; each method's "
                                 "effective data differs and is not identical")}},
            "clean_low_n": clean_low_n_eligibility(
                target=problem.target, assay=problem.assay,
                feature_history=problem.space.supervision, prior_history=problem.prior.supervision,
                train_ids=problem.train_ids, support_ids=problem.support_ids,
                acquired_labels=int(acquired.shape[0]), acquired_ids=acquired_ids,
                declared_validation_labels=config.declared_validation_labels),
            "input_binding": {
                "problem_sha256": digest(binding), "components": binding,
                "note": ("compact content digests of the COMPLETE offered inputs (selected or "
                         "not), the unlabelled training rows and, for the CSV pipeline, the whole "
                         "candidate table. Integrity and provenance only: a digest does not prove "
                         "the inputs were truthful")},
            "evaluation_outcomes_used": False, "stages_run": ["fit", "allocate"],
            "declared_not_verified": True}}
    document["hashes"] = _section_hashes(document)
    return _builtin(document)


def _logs_match(stored, recomputed):
    stored, recomputed = np.asarray(stored, dtype=np.float64), np.asarray(recomputed, dtype=np.float64)
    if stored.shape != recomputed.shape or bool(np.any((stored == -np.inf) != (recomputed == -np.inf))):
        return False
    finite = stored > -np.inf
    return bool(np.allclose(stored[finite], recomputed[finite], rtol=1e-6, atol=1e-8))


def _verify_input_binding(document, support_ids):
    """The offered roster, the acquired indices, the selection rule and the input binding must
    agree with each other and with the stored sections (a re-hashed edit is still inconsistent)."""
    training, config = document["training"], document["config"]
    offered, indices = training["offered_ids"], training["acquired_indices"]
    require(isinstance(offered, list) and offered == sorted(offered)
            and len(set(offered)) == len(offered) and all(isinstance(i, str) for i in offered),
            "the offered training roster is not in canonical order or has duplicates")
    require(not set(offered) & set(support_ids),
            "an offered training candidate sits in the held-out support (train/evaluation overlap)")
    require(isinstance(indices, list)
            and all(isinstance(i, int) and not isinstance(i, bool) for i in indices)
            and indices == sorted(set(indices))
            and all(0 <= i < len(offered) for i in indices),
            "the acquired indices are malformed or outside the offered roster")
    require([offered[i] for i in indices] == training["ids"],
            "the acquired ids are not the offered roster at the acquired indices")
    require(indices == acquire_labels(len(offered), config["label_budget"],
                                      config["seed"]).tolist(),
            "the acquired indices are not the seeded selection of the offered roster")
    selection = training["selection"]
    require(selection["label_budget"] == config["label_budget"]
            and selection["seed"] == config["seed"]
            and selection["evaluation_outcomes_used"] is False,
            "the recorded selection rule disagrees with the configuration")
    require(document["features"]["train_feature_sha256"] == digest(training["features"]),
            "the acquired-feature digest does not match the stored acquired features")
    accounting = document["provenance"]["label_accounting"]
    require(accounting["offered_ids_sha256"] == digest(offered)
            and accounting["acquired_ids_sha256"] == digest(training["ids"]),
            "the label accounting does not describe the stored offered/acquired identities")
    binding = document["provenance"]["input_binding"]
    components = binding["components"]
    require(digest(components) == binding["problem_sha256"],
            "the input binding's problem digest does not match its components")
    require(components["train"]["count"] == len(offered)
            and components["train"]["ids_sha256"] == digest(offered)
            and components["support"]["count"] == len(support_ids)
            and components["support"]["ids_sha256"] == document["support"]["identity_sha256"]
            and components["support"]["sequences_sha256"] == document["support"]["sequences_sha256"],
            "the input binding does not describe the stored offered and held-out rosters")
    require(components["manifest_sha256"] == digest(document["manifest"])
            and components["space_sha256"] == digest(document["features"]["space"])
            and components["prior_spec_sha256"] == digest(document["prior"]["spec"]),
            "the input binding does not match the stored manifest, feature space or prior spec")


@contextlib.contextmanager
def _malformed_artifact_as_value_error():
    """Turn a missing key / wrong container type inside an artifact into a clear ``ValueError``.

    A hand-edited or re-hashed artifact can pass the hash comparison and still lack a field the
    checks read; the bare ``KeyError`` / ``TypeError`` would not say that the artifact is the
    problem. The original exception stays chained as the cause.
    """
    try:
        yield
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"the frozen artifact is malformed: a field the verifier or evaluator "
                         f"reads is missing or has the wrong type ({type(error).__name__}: "
                         f"{error})") from error


def verify_artifact(document):
    """Raise ``ValueError`` unless the frozen artifact is untampered AND well-formed; see
    ``_verify_artifact_unguarded`` for the checks. A malformed artifact (a missing key or a
    wrong-typed section, even one that was re-hashed) fails with a clear message rather than a
    bare ``KeyError``."""
    with _malformed_artifact_as_value_error():
        return _verify_artifact_unguarded(document)


def _verify_artifact_unguarded(document):
    """Raise unless the frozen artifact is untampered AND internally consistent.

    Hashes: every section and the whole document are recomputed from the stored values. Beyond
    that, the ridge weights are re-derived from the stored training data, and the policies and the
    SIR law from the stored prior, predictions, energies and counts; the acquired indices are
    re-derived from the seeded selection of the offered roster and the input binding is checked
    against the stored sections, so an edit that was re-hashed but is inconsistent is still
    refused. A hash verifies integrity, not the truth of the declared provenance or that the
    inputs were truthful.
    """
    require(isinstance(document, dict), "a frozen artifact is a mapping")
    require(document.get("schema") == ARTIFACT_SCHEMA and document.get("stage") == "frozen",
            f"not a {ARTIFACT_SCHEMA} frozen artifact")
    expected_keys = set(HASHED_SECTIONS) | {"schema", "stage", "hashes"}
    require(set(document) == expected_keys,
            f"the artifact's keys differ from the hashed layout by "
            f"{sorted(set(document) ^ expected_keys)}; nothing may sit outside the hash")
    stored = document["hashes"]
    recomputed = _section_hashes(document)
    changed = sorted(name for name in recomputed if stored.get(name) != recomputed[name])
    require(not changed, f"the frozen artifact was modified after freezing: {changed} no longer "
                         "match their hashes")
    ids = document["support"]["ids"]
    size = len(ids)
    require(ids == sorted(ids) and len(set(ids)) == size and document["support"]["count"] == size,
            "the support is not in canonical order, has duplicates, or its count is wrong")
    require(document["support"]["identity_sha256"] == digest(ids), "the support identity hash is wrong")
    require(not set(document["training"]["ids"]) & set(ids),
            "a training candidate sits in the held-out support (train/evaluation overlap)")
    require(document["provenance"]["evaluation_outcomes_used"] is False,
            "the artifact claims evaluation outcomes were used before freezing")
    _verify_input_binding(document, ids)
    tau = document["config"]["tau"]
    log_p = decode_log(document["prior"]["log_probability"])
    require(log_p.shape == (size,) and _logs_match(log_p, normalize_log_weights(
        decode_log(document["prior"]["log_weight"]), name="prior")),
        "the stored prior probabilities are not the normalised declared prior")
    train = document["training"]
    weights, intercept = _ridge_solution(np.array(train["features"], dtype=np.float64),
                                         np.array(train["utilities"], dtype=np.float64),
                                         document["config"]["ridge_alpha"])
    reward = document["fitted"]["reward"]
    require(np.allclose(weights, reward["weights"], rtol=1e-6, atol=1e-9)
            and math.isclose(intercept, reward["intercept"], rel_tol=1e-6, abs_tol=1e-9),
            "the stored ridge weights are not the ridge solution of the stored training data")
    uhat = np.array(document["predictions"]["utility"], dtype=np.float64)
    require(uhat.shape == (size,), "the predictions do not align with the support")
    policies = {name: decode_log(value, name=name)
                for name, value in document["policies"]["log_probability"].items()}
    require(_logs_match(policies["ridge_tilt"], log_tilt(log_p, uhat, tau)),
            "the stored exact tilt is not the tilt of the stored prior and predictions")
    energy = document["predictions"]["dpo_energy"]
    pressures = document["config"]["dpo"]["pressures"]
    expected_names = {"prior", "ridge_tilt", "sir_weighted_proposal"}
    if energy is not None:
        energy = np.array(energy, dtype=np.float64)
        for pressure in pressures:
            name = pressure_name(pressure)
            expected_names.add(name)
            require(name in policies and _logs_match(
                policies[name], dpo_log_policy(log_p, energy, pressure, name=name)),
                f"the stored {name} policy is not the tilt of the stored energies")
    require(set(policies) == expected_names, "the stored policy set is not the expected one")
    require(_logs_match(policies["prior"], log_p), "the stored 'prior' policy is not the prior")
    sir = document["policies"]["sir"]
    weighted, _ = sir_weighted_distribution(log_p, decode_log(sir["log_proposal"]), uhat, tau,
                                            np.array(sir["counts"], dtype=np.int64))
    require(_logs_match(policies["sir_weighted_proposal"], weighted),
            "the stored SIR law is not the weighted proposal of the stored counts")
    require(sum(sir["final_histogram"]) == sir["final_draws"],
            "the SIR final histogram does not hold the declared number of draws")
    return True


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _infinite_record(value):
    if math.isinf(value):
        return {"status": "infinite", "value": None}
    return {"status": "finite", "value": float(value)}


def _objective_record(value):
    if math.isinf(value):
        return {"status": "unavailable_infinite_kl", "value": None}
    return {"status": "finite", "value": float(value)}


def _identity_holds(residual, *scales):
    return bool(abs(residual) <= IDENTITY_TOLERANCE * max(1.0, *[abs(item) for item in scales]))


def _policy_report(log_q, *, log_p, utilities, tau, draws, threshold, log_star, best, log_hat=None):
    q = np.exp(log_q)
    useful = utilities >= threshold
    distinct = expected_distinct(q, draws)
    kl_q_p = kl_divergence(log_q, log_p)
    value = objective(log_q, log_p, utilities, tau)
    kl_star = kl_divergence(log_q, log_star)
    regret = math.inf if math.isinf(value) else best - value
    report = {
        "reachable_support": int((log_q > -np.inf).sum()),
        "effective_support": effective_support(log_q),
        "expected_utility": expected_value(log_q, utilities),
        "useful_mass": float(q[useful].sum()),
        "expected_distinct_useful": float(distinct[useful].sum()),
        "expected_distinct_total": float(distinct.sum()),
        "kl_policy_to_prior": _infinite_record(kl_q_p),
        "kl_prior_to_policy": _infinite_record(kl_divergence(log_p, log_q)),
        "tv_to_prior": total_variation(log_q, log_p),
        "objective_J": _objective_record(value),
        "oracle_regret": _objective_record(regret),
        "kl_policy_to_oracle": _infinite_record(kl_star),
        "regret_identity": {"statement": "J(q*) - J(q) = tau * KL(q || q*)"}}
    if math.isinf(regret) or math.isinf(kl_star):
        report["regret_identity"].update(holds=bool(math.isinf(regret) and math.isinf(kl_star)),
                                         residual=None, status="both_sides_infinite")
    else:
        residual = regret - tau * kl_star
        report["regret_identity"].update(holds=_identity_holds(residual, best, value, tau * kl_star),
                                         residual=float(residual), status="finite")
    reach = log_q > -np.inf
    cohort = log_p > -np.inf
    with np.errstate(invalid="ignore"):
        log_ratio = log_q - log_p
    both = reach & cohort

    def ranked(scores, outcomes):
        return {"spearman": spearman(scores, outcomes),
                "concordance": concordance(scores, outcomes)}
    # PRIMARY ranking: one cohort for every policy, the candidates the declared prior can reach.
    # A cohort member the policy gives no mass keeps its -inf score and ties LAST, so a sparse
    # policy cannot look perfect by ranking only the few candidates it chose. Log scores are
    # compared directly (never exponentiated), so a finite underflowed log probability keeps its
    # order and is not turned into a structural zero. The log ratio drops prior-zero candidates
    # (the ratio is undefined where p = 0) but keeps policy-zero ones (ratio -inf).
    report["ranking"] = {
        "cohort": {
            "definition": ("every candidate with positive declared-prior mass: one cohort common "
                           "to all policies; a policy's zero-mass cohort member ranks tied last"),
            "candidates": int(cohort.sum()),
            "policy_zero_mass_in_cohort": int((cohort & ~reach).sum())},
        "policy_log_probability_vs_utility": ranked(log_q[cohort], utilities[cohort]),
        "policy_log_ratio_to_prior_vs_utility": ranked(log_ratio[cohort], utilities[cohort]),
        "note": ("over the common prior-reachable cohort; a constant score has an undefined rank"),
        "conditional_on_policy_support": {
            "definition": ("ONLY the candidates this policy gave mass to: conditional on the "
                           "policy's own selection, so a sparse policy can look perfect here; "
                           "never the headline ranking"),
            "candidates": int(reach.sum()),
            "policy_log_probability_vs_utility": ranked(log_q[reach], utilities[reach]),
            "policy_log_ratio_to_prior_vs_utility": ranked(log_ratio[both], utilities[both])}}
    if log_hat is not None:
        report["vs_learned_exact_tilt"] = {
            "tv": total_variation(log_q, log_hat),
            "kl_policy_to_learned": _infinite_record(kl_divergence(log_q, log_hat)),
            "kl_learned_to_policy": _infinite_record(kl_divergence(log_hat, log_q))}
    return report


def _outcome_vector(ids, outcomes):
    require(isinstance(outcomes, dict), "outcomes must map candidate_id to utility")
    missing = [item for item in ids if item not in outcomes]
    require(not missing,
            f"evaluation outcomes are missing for {len(missing)} of {len(ids)} held-out "
            f"candidate(s), e.g. {missing[:3]}. Nothing is renormalised onto an observed subset: "
            "a complete evaluation needs every held-out candidate's outcome.")
    extra = sorted(set(outcomes) - set(ids))
    require(not extra,
            f"evaluation outcomes name {len(extra)} candidate(s) outside the held-out support, "
            f"e.g. {extra[:3]} (training candidates or unknown ids: train/evaluation overlap)")
    return _vector([outcomes[item] for item in ids], "evaluation outcomes", length=len(ids))


def evaluate_frozen(artifact, outcomes):
    """Stage 3: verify the frozen artifact, then score it against complete held-out outcomes.

    The hindsight oracle is computed here and nowhere else. Every true-J comparison in the report
    shares the artifact's one ``tau``.
    """
    verify_artifact(artifact)
    ids = artifact["support"]["ids"]
    u = _outcome_vector(ids, outcomes)
    with _malformed_artifact_as_value_error():
        return _evaluate_verified(artifact, ids, u)


def _evaluate_verified(artifact, ids, u):
    """The scoring body of ``evaluate_frozen`` for a verified artifact and complete outcomes."""
    tau = float(artifact["config"]["tau"])
    draws = int(artifact["config"]["sir"]["final_draws"])
    threshold = float(artifact["manifest"]["utility"]["useful_threshold"])
    log_p = decode_log(artifact["prior"]["log_probability"])
    uhat = np.array(artifact["predictions"]["utility"], dtype=np.float64)
    policies = {name: decode_log(value, name=name)
                for name, value in artifact["policies"]["log_probability"].items()}
    log_hat = policies["ridge_tilt"]
    log_star = log_tilt(log_p, u, tau)
    best = objective(log_star, log_p, u, tau)
    closed_form = oracle_objective_closed_form(log_p, u, tau)
    common = dict(log_p=log_p, utilities=u, tau=tau, draws=draws, threshold=threshold,
                  log_star=log_star, best=best)
    reports = {name: _policy_report(log_q, log_hat=(log_hat if name != "ridge_tilt" else None),
                                    **common)
               for name, log_q in policies.items()}
    oracle = _policy_report(log_star, **common)
    oracle["note"] = ("hindsight: built from the evaluation outcomes, not achievable; it "
                      "maximises J, so its expected utility is below the best single candidate's")
    reach = log_p > -np.inf
    prediction_ranking = {
        "all_candidates": {"spearman": spearman(uhat, u), "concordance": concordance(uhat, u)},
        "reachable_support": {"spearman": spearman(uhat[reach], u[reach]),
                              "concordance": concordance(uhat[reach], u[reach])},
        "rmse": float(math.sqrt(float(np.mean((uhat - u) ** 2)))),
        "mean_bias": float(np.mean(uhat - u)),
        "note": ("utility-prediction ranking: how well the frozen predictor orders held-out "
                 "candidates, separate from policy-probability ranking in each policy report")}
    sir_doc = artifact["policies"]["sir"]
    sir_log = policies["sir_weighted_proposal"]
    gap = tau * kl_divergence(sir_log, log_hat)
    j_tilde, j_hat = objective(sir_log, log_p, u, tau), objective(log_hat, log_p, u, tau)
    cross = (expected_value(sir_log, u - uhat) - expected_value(log_hat, u - uhat))
    attribution = {
        "predictor": "the SAME frozen ridge predictor builds both the exact tilt and the SIR law",
        "statement": ("J_u(q~) - J_u(qhat) = -tau*KL(q~||qhat) + (E_q~ - E_qhat)[u - uhat]; the "
                      "terms have opposite signs in general, not two nonnegative errors")}
    if math.isinf(gap) or math.isinf(j_tilde) or math.isinf(j_hat):
        attribution.update(status="unavailable_infinite_kl")
    else:
        true_difference = j_tilde - j_hat
        residual = true_difference - (-gap + cross)
        attribution.update(
            status="finite", predicted_objective_gap=float(gap),
            true_J_difference_from_learned_exact=float(true_difference),
            residual_cross_term=float(cross), identity_residual=float(residual),
            identity_holds=_identity_holds(residual, true_difference, gap, cross),
            predicted_gap_is_nonnegative=bool(gap >= 0.0))
    realized = np.array(sir_doc["final_histogram"], dtype=np.int64)
    useful = u >= threshold
    sir_report = {
        "weighted_proposal_law": reports["sir_weighted_proposal"],
        "occurrence_effective_sample_size": float(sir_doc["effective_sample_size"]),
        "proposal_draws_K": int(sir_doc["proposal_draws"]),
        "distinct_proposed_candidates": int(sir_doc["distinct_proposed"]),
        "final_resampled_histogram": {
            "draws_M": int(realized.sum()), "realized_distinct": int((realized > 0).sum()),
            "realized_unique_useful": int(((realized > 0) & useful).sum()),
            "realized_useful_draws": int(realized[useful].sum()),
            "realized_mean_utility": float((realized * u).sum() / realized.sum()),
            "note": ("realized counts of the one final draw. The expected-distinct formulas in "
                     "weighted_proposal_law describe the weighted law BEFORE these M resamples; "
                     "applying them to this histogram would describe a second hypothetical "
                     "resampling")},
        "attribution_against_learned_exact_tilt": attribution,
        "resource_accounting": copy.deepcopy(sir_doc["resource_accounting"])}
    identities = [{"name": f"oracle_regret::{name}", "holds": entry["regret_identity"]["holds"]}
                  for name, entry in sorted(reports.items())]
    identities.append({"name": "oracle_regret::oracle", "holds": oracle["regret_identity"]["holds"]})
    identities.append({"name": "oracle_objective_closed_form",
                       "holds": _identity_holds(best - closed_form, best, closed_form),
                       "residual": float(best - closed_form)})
    if attribution.get("status") == "finite":
        identities.append({"name": "sir_attribution", "holds": attribution["identity_holds"]})
    dpo_doc = artifact["fitted"]["dpo"]
    pressure_names = [pressure_name(c) for c in artifact["config"]["dpo"]["pressures"]]
    curve = []
    for c, name in zip(artifact["config"]["dpo"]["pressures"], pressure_names):
        if name in reports:
            entry = reports[name]
            curve.append({"pressure": c, "policy": name,
                          "kl_policy_to_prior": entry["kl_policy_to_prior"],
                          "expected_utility": entry["expected_utility"],
                          "useful_mass": entry["useful_mass"], "objective_J": entry["objective_J"],
                          "oracle_regret": entry["oracle_regret"]})
    dpo_report = ({"status": "unsupported", "reason": dpo_doc["reason"]}
                  if dpo_doc["status"] == "unsupported" else
                  {"status": "fit", "training_diagnostics": copy.deepcopy(dpo_doc["diagnostics"]),
                   "pressure_curve": curve,
                   "attribution": ("not applicable: the implicit energy is not the numeric "
                                   "predictor, so there is no same-predictor attribution"),
                   "note": CLAIMS["dpo_energy"]})
    table = []
    for name, entry in list(reports.items()) + [("oracle", oracle)]:
        table.append({"policy": name, "expected_utility": entry["expected_utility"],
                      "useful_mass": entry["useful_mass"],
                      "kl_policy_to_prior": entry["kl_policy_to_prior"],
                      "objective_J": entry["objective_J"], "oracle_regret": entry["oracle_regret"]})
    return _builtin({
        "schema": REPORT_SCHEMA, "artifact_sha256": artifact["hashes"]["artifact_sha256"],
        "claims": dict(CLAIMS),
        "tau": tau, "final_draws_M": draws, "useful_threshold": threshold,
        "manifest": copy.deepcopy(artifact["manifest"]),
        "support": {"candidates": len(ids),
                    "reachable_support": int(reach.sum()),
                    "structural_zeros": int((~reach).sum()),
                    "useful_candidates": int(useful.sum()),
                    "max_utility": float(u.max())},
        "truth_coverage": {"required": len(ids), "provided": len(ids), "complete": True,
                           "rule": "every held-out candidate needs an outcome; none is imputed"},
        "prediction_ranking": prediction_ranking,
        "oracle": oracle, "oracle_objective_closed_form": float(closed_form),
        "policies": reports, "sir": sir_report, "dpo": dpo_report,
        "comparison": table,
        "identities": {"all_hold": bool(all(item["holds"] for item in identities)),
                       "checks": identities},
        "label_accounting": copy.deepcopy(artifact["provenance"]["label_accounting"]),
        "effective_data": copy.deepcopy(artifact["provenance"]["effective_data"]),
        "split_counts": copy.deepcopy(artifact["provenance"]["split_counts"]),
        "clean_low_n": copy.deepcopy(artifact["provenance"]["clean_low_n"])})


# ---------------------------------------------------------------------------
# CSV / manifest contracts and the finite-support pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Manifest:
    target: str
    assay: str
    utility: UtilitySpec
    grouping_rationale: str
    alphabet: str
    length: int
    feature_supervision: SupervisionHistory
    prior: PriorSpec
    notes: str = ""


MANIFEST_KEYS = ("target", "assay", "utility", "grouping_rationale", "features", "prior")


def parse_manifest(document):
    """Validate the manifest mapping. One comparable target and assay; nothing is inferred."""
    document = copy.deepcopy(document)
    require(isinstance(document, dict), "the manifest must be a JSON object")
    unknown = sorted(set(document) - set(MANIFEST_KEYS) - {"notes"})
    require(not unknown, f"the manifest has unknown key(s) {unknown}")
    missing = [key for key in MANIFEST_KEYS if key not in document]
    require(not missing, f"the manifest is missing {missing}")
    utility = document["utility"]
    require(isinstance(utility, dict), "manifest.utility must be an object")
    bad = sorted(set(utility) - {"name", "units", "scale", "higher_is_better", "useful_threshold"})
    require(not bad, f"manifest.utility has unknown key(s) {bad}")
    absent = [key for key in ("units", "scale", "higher_is_better", "useful_threshold")
              if key not in utility]
    require(not absent, f"manifest.utility is missing {absent}")
    features = document["features"]
    require(isinstance(features, dict), "manifest.features must be an object")
    bad = sorted(set(features) - {"kind", "alphabet", "length", "provenance"})
    require(not bad, f"manifest.features has unknown key(s) {bad}")
    require(features.get("kind") == "sequence_onehot",
            "manifest.features.kind must be 'sequence_onehot' (the only representation a CSV "
            "import builds; other representations use the explicit-feature API)")
    for key in ("alphabet", "length", "provenance"):
        require(key in features, f"manifest.features is missing {key!r}")
    prior = document["prior"]
    require(isinstance(prior, dict), "manifest.prior must be an object")
    bad = sorted(set(prior) - {"kind", "semantics", "description", "provenance"})
    require(not bad, f"manifest.prior has unknown key(s) {bad}")
    for key in ("kind", "semantics", "provenance"):
        require(key in prior, f"manifest.prior is missing {key!r}")
    return Manifest(
        target=_text(document["target"], "manifest.target"),
        assay=_text(document["assay"], "manifest.assay"),
        utility=UtilitySpec(units=utility["units"], scale=utility["scale"],
                            higher_is_better=utility["higher_is_better"],
                            useful_threshold=utility["useful_threshold"],
                            name=utility.get("name", "utility")),
        grouping_rationale=_text(document["grouping_rationale"], "manifest.grouping_rationale"),
        alphabet=_text(features["alphabet"], "manifest.features.alphabet"),
        length=_integer(features["length"], "manifest.features.length", minimum=1),
        feature_supervision=SupervisionHistory.from_document(features["provenance"],
                                                             label="manifest.features"),
        prior=PriorSpec(kind=prior["kind"], semantics=prior["semantics"],
                        description=prior.get("description", ""),
                        supervision=SupervisionHistory.from_document(prior["provenance"],
                                                                     label="manifest.prior")),
        notes=str(document.get("notes", "")))


def load_manifest(path):
    return parse_manifest(read_json(path))


def _read_rows(path, columns, *, label):
    path = Path(path)
    require(path.is_file(), f"{label} file {path} does not exist")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        require(header is not None, f"{label} file {path} is empty")
        require(len(set(header)) == len(header), f"{label} has a repeated column")
        missing = [name for name in columns if name not in header]
        extra = [name for name in header if name not in columns]
        require(not missing, f"{label} is missing column(s) {missing}")
        require(not extra,
                f"{label} has unexpected column(s) {extra}. The contract is exactly {list(columns)}; "
                "an extra column (for instance an outcome) is refused rather than ignored.")
        rows = []
        for line, row in enumerate(reader, start=2):
            require(None not in row and all(value is not None for value in row.values()),
                    f"{label} line {line} has the wrong number of fields")
            rows.append((line, row))
    return rows


#: The only spellings that declare a structural zero. A literal such as ``-1e999`` overflows to
#: ``-inf`` inside ``float`` and must NOT become one: a structural zero is a deliberate statement.
NEGATIVE_INFINITY_SPELLINGS = ("-inf", "-infinity")


def _parse_number(text, *, label, line, allow_negative_infinity=False):
    text = text.strip()
    require(text, f"{label} line {line}: empty value")
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{label} line {line}: {text!r} is not a number") from None
    require(not math.isnan(number), f"{label} line {line}: NaN is not allowed")
    if number == -math.inf:
        require(text.lower() in NEGATIVE_INFINITY_SPELLINGS,
                f"{label} line {line}: {text!r} overflows to negative infinity; only the explicit "
                f"spelling {' or '.join(NEGATIVE_INFINITY_SPELLINGS)} can declare a structural zero")
        require(allow_negative_infinity,
                f"{label} line {line}: -inf is only allowed as an explicit prior_log_weight")
    else:
        require(math.isfinite(number),
                f"{label} line {line}: {text!r} is not finite (+inf is never allowed)")
    return number


@dataclass(frozen=True, eq=False)
class CandidateTable:
    ids: tuple
    sequences: tuple
    groups: tuple
    splits: tuple
    prior_log_weight: object

    def roster(self):
        """Compact content digests of the WHOLE candidate input: every row, labelled or not, on
        either side of the split, in canonical order. Binds the unlabelled training rows too."""
        return {"count": len(self.ids), "ids_sha256": digest(list(self.ids)),
                "sequences_sha256": digest(list(self.sequences)),
                "groups_sha256": digest(list(self.groups)),
                "splits_sha256": digest(list(self.splits)),
                "prior_log_weight_sha256": _array_sha256(self.prior_log_weight)}


def read_candidates(path, *, alphabet, length):
    """The candidate CSV: exactly ``candidate_id, sequence, group_id, split, prior_log_weight``.

    Refused: duplicate ids, duplicate physical sequences (and when a duplicate straddles the
    split, that is named as train/evaluation overlap), invalid residues or lengths, a biological
    group on both sides of the split, NaN/+inf prior weights, and any unlisted column. Rows are
    returned in canonical (sorted by id) order.
    """
    rows = _read_rows(path, CANDIDATE_COLUMNS, label="candidates")
    require(rows, "the candidates file has no rows")
    ids, sequences, groups, splits, priors = [], [], [], [], []
    for line, row in rows:
        ident = row["candidate_id"].strip()
        require(ident, f"candidates line {line}: empty candidate_id")
        sequence = canonical_sequence(row["sequence"])
        require(len(sequence) == length,
                f"candidates line {line}: sequence {sequence!r} has length {len(sequence)}, not "
                f"the fixed length {length}")
        bad = sorted({residue for residue in sequence if residue not in alphabet})
        require(not bad, f"candidates line {line}: residue(s) {bad} outside the alphabet {alphabet!r}")
        group = row["group_id"].strip()
        require(group, f"candidates line {line}: empty group_id")
        split = row["split"].strip()
        require(split in (SPLIT_TRAIN, SPLIT_EVAL),
                f"candidates line {line}: split must be {SPLIT_TRAIN!r} or {SPLIT_EVAL!r}, got {split!r}")
        ids.append(ident)
        sequences.append(sequence)
        groups.append(group)
        splits.append(split)
        priors.append(_parse_number(row["prior_log_weight"], label="candidates", line=line,
                                    allow_negative_infinity=True))
    require(len(set(ids)) == len(ids), "the candidates file repeats a candidate_id")
    seen = {}
    for ident, sequence, split in zip(ids, sequences, splits):
        if sequence in seen:
            other, other_split = seen[sequence]
            overlap = " (train/evaluation overlap in canonical physical sequence)" \
                if other_split != split else ""
            raise ValueError(f"candidates {other!r} and {ident!r} are the same physical sequence "
                             f"{sequence!r}{overlap}")
        seen[sequence] = (ident, split)
    sides = {}
    for group, split in zip(groups, splits):
        sides.setdefault(group, set()).add(split)
    crossing = sorted(group for group, kinds in sides.items() if len(kinds) > 1)
    require(not crossing,
            f"biological group(s) {crossing[:3]} appear in both the training and the evaluation "
            "split; the grouping rationale is violated and the held-out score would be leakage")
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    return CandidateTable(ids=tuple(ids[i] for i in order),
                          sequences=tuple(sequences[i] for i in order),
                          groups=tuple(groups[i] for i in order),
                          splits=tuple(splits[i] for i in order),
                          prior_log_weight=_frozen([priors[i] for i in order]))


def read_outcomes(path, *, label):
    """An outcome CSV ``candidate_id, utility``: finite utilities, no repeated candidate."""
    rows = _read_rows(path, OUTCOME_COLUMNS, label=label)
    outcomes = {}
    for line, row in rows:
        ident = row["candidate_id"].strip()
        require(ident, f"{label} line {line}: empty candidate_id")
        require(ident not in outcomes,
                f"{label} line {line}: candidate {ident!r} has two outcomes; aggregate replicates "
                "upstream and declare the rule rather than presenting a label twice")
        outcomes[ident] = _parse_number(row["utility"], label=label, line=line)
    return outcomes


def build_problem(manifest, table, train_outcomes):
    """Join a validated manifest, candidate table and TRAINING outcomes into a problem."""
    require(isinstance(manifest, Manifest) and isinstance(table, CandidateTable),
            "build_problem needs a parsed Manifest and CandidateTable")
    index = {ident: position for position, ident in enumerate(table.ids)}
    unknown = sorted(set(train_outcomes) - set(index))
    require(not unknown, f"training outcomes name unknown candidate(s) {unknown[:3]}")
    on_eval = sorted(ident for ident in train_outcomes
                     if table.splits[index[ident]] == SPLIT_EVAL)
    require(not on_eval,
            f"the training outcomes include {len(on_eval)} evaluation-split candidate(s), e.g. "
            f"{on_eval[:3]}: train/evaluation overlap in ids")
    require(train_outcomes, "the training outcomes file has no rows; there is nothing to learn from")
    # The declared prior is a statement about the WHOLE pool, so it is checked over every row,
    # including training candidates that carry no outcome.
    validate_prior_log_weights(manifest.prior.kind, table.prior_log_weight)
    labelled = sorted(train_outcomes)
    support = [ident for ident in table.ids if table.splits[index[ident]] == SPLIT_EVAL]
    require(support, "the candidates file has no evaluation-split candidates")
    space = onehot_feature_space(manifest.alphabet, manifest.length, manifest.feature_supervision)
    lab_idx = [index[ident] for ident in labelled]
    sup_idx = [index[ident] for ident in support]
    unlabelled = [ident for ident in table.ids
                  if table.splits[index[ident]] == SPLIT_TRAIN and ident not in train_outcomes]
    return FiniteSupportProblem(
        target=manifest.target, assay=manifest.assay, utility=manifest.utility,
        grouping_rationale=manifest.grouping_rationale, space=space, prior=manifest.prior,
        train_ids=tuple(labelled),
        train_features=onehot_encode([table.sequences[i] for i in lab_idx], space),
        train_utilities=[train_outcomes[ident] for ident in labelled],
        train_log_prior=table.prior_log_weight[lab_idx],
        train_groups=tuple(table.groups[i] for i in lab_idx),
        support_ids=tuple(support),
        support_features=onehot_encode([table.sequences[i] for i in sup_idx], space),
        support_log_prior=table.prior_log_weight[sup_idx],
        support_groups=tuple(table.groups[i] for i in sup_idx),
        train_sequences=tuple(table.sequences[i] for i in lab_idx),
        support_sequences=tuple(table.sequences[i] for i in sup_idx),
        unlabelled_train_candidates=len(unlabelled), unlabelled_train_ids=tuple(unlabelled),
        candidate_roster=table.roster())


def fit_and_allocate(problem, config):
    """Stages 1 and 2 composed: a frozen artifact from training data only."""
    require(config.sir_proposal != SIR_DECLARED,
            "fit_and_allocate cannot build a declared SIR proposal; call fit_stage, allocate "
            "(with proposal_log_weights) and freeze_artifact directly")
    fit = fit_stage(problem, config)
    allocation = allocate(problem.support_ids, problem.support_log_prior, fit.predictions, config,
                          dpo_energy=fit.dpo_energy)
    return freeze_artifact(problem, config, fit, allocation)


def _same_file(left, right):
    """True when two path spellings name one file: equal resolved paths (symlinks, ``..`` and, on
    Windows, letter case are folded) or the same existing file (hard links, junctions)."""
    left, right = Path(left), Path(right)
    if os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right)):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _refuse_path_aliases(inputs, outputs):
    """Refuse, before anything is written, an output that resolves to an input.

    ``outputs`` must include the temporary files ``write_json`` stages through, not only the final
    names: an input sitting at ``frozen_artifact.json.tmp`` would be clobbered by the atomic write.
    ``--overwrite`` permits replacing a previous OUTPUT, never an input.
    """
    for input_label, source in inputs.items():
        for output_label, target in outputs.items():
            require(not _same_file(source, target),
                    f"the {input_label} {source} resolves to the same file as the {output_label} "
                    f"{target}; writing would destroy an input, even with overwrite. Nothing was "
                    "written: choose an output directory that does not alias an input")


def run_finite_support(*, candidates, train_outcomes, eval_outcomes, manifest, output_dir, config,
                       overwrite=False):
    """The CLI pipeline. The evaluation-outcome file is opened only AFTER the artifact is on disk.

    Order: refuse any input/output path alias (resolved paths, including the atomic temporary
    files; before any write, even with ``overwrite``); read the manifest, candidates and TRAINING
    outcomes; fit; allocate; write and reload the hashed frozen artifact; only then read the
    evaluation outcomes, verify the reloaded artifact, evaluate, and write the report. A missing
    evaluation file therefore leaves a frozen artifact that never saw it.
    """
    out = Path(output_dir)
    frozen_path, report_path = out / FROZEN_NAME, out / REPORT_NAME
    require(not out.exists() or out.is_dir(), f"the output directory {out} exists and is not a "
                                              "directory")
    _refuse_path_aliases(
        {"candidates file": candidates, "training outcomes file": train_outcomes,
         "evaluation outcomes file": eval_outcomes, "manifest file": manifest},
        {"frozen artifact": frozen_path, "evaluation report": report_path,
         "frozen artifact temporary file": _temporary_path(frozen_path),
         "evaluation report temporary file": _temporary_path(report_path)})
    for target in (frozen_path, report_path):
        require(overwrite or not target.exists(),
                f"{target} already exists; pass overwrite=True/--overwrite to replace it")
    parsed = load_manifest(manifest)
    table = read_candidates(candidates, alphabet=parsed.alphabet.upper(), length=parsed.length)
    train = read_outcomes(train_outcomes, label="training outcomes")
    problem = build_problem(parsed, table, train)
    artifact = fit_and_allocate(problem, config)
    write_json(frozen_path, artifact)
    # ---- the frozen boundary: nothing above read an evaluation outcome ----
    reloaded = read_json(frozen_path)
    verify_artifact(reloaded)
    outcomes = read_outcomes(eval_outcomes, label="evaluation outcomes")
    report = evaluate_frozen(reloaded, outcomes)
    write_json(report_path, report)
    return {"frozen_artifact": str(frozen_path), "report": str(report_path),
            "artifact_sha256": reloaded["hashes"]["artifact_sha256"],
            "all_identities_hold": report["identities"]["all_hold"],
            "support": report["support"], "clean_low_n": report["clean_low_n"]["status"],
            "comparison": report["comparison"]}


# ---------------------------------------------------------------------------
# the planted causal-learning experiment (explicit features, no sequences)
# ---------------------------------------------------------------------------

PLANTED_COLUMNS = ("a", "b", "ab", "nuisance")
PLANTED_CELLS = tuple((a, b, n) for a in (0, 1) for b in (0, 1) for n in (0, 1))

#: Stated prominently in the design record, the selftest report and the CLI help: what the toy
#: does and does not qualify.
PLANTED_SCOPE_LABEL = (
    "FIT / INTERVENTION QUALIFICATION, NOT unseen-context generalisation. The training design is a "
    "full factorial over (a, b, nuisance): 8 unique feature contexts, each replicated 3 times (24 "
    "training rows). The interaction coordinate ab = a*b is SUPPLIED as a column, so the "
    "interaction predictor is handed the feature it needs, not asked to discover it. The held-out "
    "evaluation candidates occupy the SAME 8 feature cells (2 replicates each, 16 rows): none has "
    "an unseen feature context.")


def planted_utility(a, b):
    """``u = 1.0 a + 0.5 b - 2.0 a b``: a's effect is +1 when b=0 and -1 when b=1 (a reversal)."""
    return 1.0 * a + 0.5 * b - 2.0 * a * b


def _planted_row(a, b, nuisance):
    return [float(a), float(b), float(a * b), float(nuisance)]


def _planted_prior(a, b, nuisance):
    return -0.5 * a + 0.25 * b + 0.1 * nuisance


def planted_space():
    return FeatureSpace(
        kind="explicit", names=PLANTED_COLUMNS,
        schema={"columns": list(PLANTED_COLUMNS),
                "construction": "a, b in {0,1}; ab = a*b; nuisance in {0,1} is irrelevant to u"},
        history=("explicit planted columns [a, b, a*b, nuisance]; no learned preprocessing",),
        supervision=SupervisionHistory(status=HISTORY_LABEL_FREE,
                                       description="explicit synthetic columns",
                                       exposure_status="none"))


def planted_problem(space, *, columns=PLANTED_COLUMNS, train_replicates=3, support_replicates=2):
    """A full-factorial planted problem. The training design is balanced, so the nuisance column
    is orthogonal to everything and its fitted weight is zero up to rounding.

    Scope: 8 unique feature contexts (a, b, nuisance), replicated 3 times for training (24 rows)
    and 2 times for evaluation (16 rows) on the SAME 8 feature cells, with the interaction
    coordinate ``ab`` supplied. It qualifies fitting and interventions, not generalisation to
    unseen contexts (see ``PLANTED_SCOPE_LABEL``).
    """
    sub_space, indices = (space, list(range(space.dimension))) if tuple(columns) == space.names \
        else space.select(columns)
    train_ids, train_rows, train_y, train_prior, train_groups = [], [], [], [], []
    for a, b, n in PLANTED_CELLS:
        for replicate in range(train_replicates):
            train_ids.append(f"train-a{a}b{b}n{n}-r{replicate}")
            train_rows.append(_planted_row(a, b, n))
            train_y.append(planted_utility(a, b))
            train_prior.append(-math.inf if (a, b, n, replicate) == (0, 0, 0, 2)
                               else _planted_prior(a, b, n))
            train_groups.append(f"train-cell-a{a}b{b}n{n}")
    support_ids, support_rows, support_prior, support_groups, truth = [], [], [], [], {}
    for a, b, n in PLANTED_CELLS:
        for replicate in range(support_replicates):
            ident = f"eval-a{a}b{b}n{n}-r{replicate}"
            support_ids.append(ident)
            support_rows.append(_planted_row(a, b, n))
            support_prior.append(-math.inf if (a, b, n, replicate) == (1, 1, 1, 1)
                                 else _planted_prior(a, b, n))
            support_groups.append(f"eval-cell-a{a}b{b}n{n}")
            truth[ident] = planted_utility(a, b)
    problem = FiniteSupportProblem(
        target="planted-interaction", assay="synthetic-closed-form",
        utility=UtilitySpec(units="arbitrary", scale="linear", higher_is_better=True,
                            useful_threshold=0.25, name="planted utility"),
        grouping_rationale="train and evaluation cells carry distinct group ids by construction",
        space=sub_space,
        prior=PriorSpec(kind=PRIOR_SCORE, semantics="explicit synthetic scores with one structural "
                                                    "zero on each side",
                        supervision=SupervisionHistory(status=HISTORY_LABEL_FREE,
                                                       description="closed-form synthetic scores",
                                                       exposure_status="none")),
        train_ids=tuple(train_ids), train_features=np.array(train_rows)[:, indices],
        train_utilities=train_y, train_log_prior=train_prior, train_groups=tuple(train_groups),
        support_ids=tuple(support_ids), support_features=np.array(support_rows)[:, indices],
        support_log_prior=support_prior, support_groups=tuple(support_groups))
    return problem, truth


def _context_rows():
    """The four (a, b) contexts with nuisance 0, in the order (0,0), (1,0), (0,1), (1,1)."""
    return np.array([_planted_row(a, b, 0) for b in (0, 1) for a in (0, 1)])


def a_effects(predictions):
    """Predicted effect of switching a 0 -> 1 in context b=0 and in context b=1."""
    p = np.asarray(predictions, dtype=np.float64)
    return {"b0": float(p[1] - p[0]), "b1": float(p[3] - p[2])}


def _reversed(effects, *, margin=0.1):
    return bool(effects["b0"] > margin and effects["b1"] < -margin
                or effects["b0"] < -margin and effects["b1"] > margin)


def _planted_scope(problem):
    """The scope label with the numbers counted from the problem itself, not asserted."""
    train_cells = {tuple(row) for row in problem.train_features.tolist()}
    support_cells = {tuple(row) for row in problem.support_features.tolist()}
    return {"label": PLANTED_SCOPE_LABEL, "full_factorial": True,
            "unique_feature_contexts": len(train_cells), "training_rows": len(problem.train_ids),
            "evaluation_rows": len(problem.support_ids),
            "supplied_interaction_coordinate": "ab",
            "evaluation_cells_are_training_cells": bool(support_cells <= train_cells),
            "evaluation_contexts_unseen_in_training": len(support_cells - train_cells)}


def run_planted_experiment(seed=0, *, permutations=200):
    """Fit, allocate and evaluate an additive and an interaction predictor on a planted truth.

    Returns a JSON-able record with the artifacts it froze. Within-predictor causal effects only.
    Scope (``PLANTED_SCOPE_LABEL``): a fit/intervention qualification on a replicated full
    factorial (8 unique feature contexts, 24 training rows) with the interaction coordinate
    supplied and evaluation on the same feature cells, NOT unseen-context generalisation.
    """
    base = planted_space()
    config = BaselineConfig(tau=0.5, seed=seed, ridge_alpha=1e-6, sir_proposal_draws=400,
                            final_draws=50, dpo_l2=1e-2)
    variants = {"additive": ("a", "b", "nuisance"), "interaction": PLANTED_COLUMNS}
    models, artifacts, reports, problems = {}, {}, {}, {}
    for name, columns in variants.items():
        problem, truth = planted_problem(base, columns=columns)
        problems[name] = problem
        fit = fit_stage(problem, config)
        allocation = allocate(problem.support_ids, problem.support_log_prior, fit.predictions,
                              config, dpo_energy=fit.dpo_energy)
        artifacts[name] = freeze_artifact(problem, config, fit, allocation)
        reports[name] = evaluate_frozen(artifacts[name], truth)
        models[name] = fit.reward
    context = _context_rows()
    reference = FeatureReference.from_training(planted_problem(base)[0].train_features, base)
    full_space = base

    def predict(name, matrix):
        sub = models[name].space
        return models[name].predict(matrix[:, full_space.indices(sub.names)])

    effects = {name: a_effects(predict(name, context)) for name in variants}
    interventions = {}
    clean = predict("interaction", context)
    identity = predict("interaction", intervene_features(context, full_space, "identity"))
    ablated_ab = intervene_features(context, full_space, "ablate", columns=("ab",),
                                    reference=reference)
    ablated = predict("interaction", ablated_ab)
    nuisance_x = intervene_features(context, full_space, "ablate", columns=("nuisance",),
                                    reference=reference)
    nuisance = predict("interaction", nuisance_x)
    donor_row = np.array(_planted_row(1, 1, 0))
    recipient = context[[1]]                       # a=1, b=0: the interaction feature is off
    patched = intervene_features(recipient, full_space, "patch", columns=("ab",), donor=donor_row)
    patched_pred = predict("interaction", patched)
    nuisance_donor = np.array(_planted_row(0, 0, 1))
    patched_nuisance = predict("interaction", intervene_features(
        recipient, full_space, "patch", columns=("nuisance",), donor=nuisance_donor))
    restored = predict("interaction", intervene_features(
        ablated_ab, full_space, "patch", columns=("ab",), donor=context))
    w = dict(zip(models["interaction"].space.names, models["interaction"].weights.tolist()))
    baseline_row = float(predict("interaction", recipient)[0])
    # Patching ab into the (a=1, b=0) row moves a's effect in the b=0 context from + to -, so it
    # lands on the b=1 context's sign: the cross-context reversal (+ in one context, - in the
    # other) is REMOVED, not introduced. Both are recorded under their own names.
    patched_effects = {"b0": float(patched_pred[0] - clean[0]), "b1": effects["interaction"]["b1"]}
    interventions = {
        "identity": {"max_abs_prediction_change": float(np.max(np.abs(identity - clean))),
                     "bitwise_equal": bool(np.array_equal(identity, clean))},
        "ablate_interaction": {"a_effects": a_effects(ablated),
                               "reversal_present": _reversed(a_effects(ablated)),
                               "max_abs_prediction_change": float(np.max(np.abs(ablated - clean)))},
        "ablate_nuisance": {"a_effects": a_effects(nuisance),
                            "reversal_present": _reversed(a_effects(nuisance)),
                            "max_abs_prediction_change": float(np.max(np.abs(nuisance - clean)))},
        "donor_patch_sufficiency": {
            "recipient": "a=1, b=0 (interaction feature off)", "donor": "a=1, b=1 (ab=1)",
            "prediction_change": float(patched_pred[0] - baseline_row),
            "interaction_weight_times_delta_ab": float(w["ab"] * (donor_row[2] - recipient[0][2])),
            "a_effect_in_context_b0_before": effects["interaction"]["b0"],
            "a_effect_in_context_b0_after": float(patched_pred[0] - clean[0]),
            "a_effect_in_context_b1": effects["interaction"]["b1"],
            "local_negative_effect_introduced": bool(effects["interaction"]["b0"] > 0.1
                                                     and patched_effects["b0"] < -0.1),
            "cross_context_reversal_before": _reversed(effects["interaction"]),
            "cross_context_reversal_after": _reversed(patched_effects),
            "cross_context_reversal_removed": bool(_reversed(effects["interaction"])
                                                   and not _reversed(patched_effects)),
            "reading": ("the patch introduces a LOCAL negative effect of a in the b=0 context "
                        "where it was positive, and thereby REMOVES the cross-context reversal "
                        "(a is now negative in both contexts); it does not introduce the reversal"),
            "nuisance_patch_prediction_change": float(patched_nuisance[0] - baseline_row)},
        "rescue_after_ablation": {"max_abs_difference_from_clean": float(np.max(np.abs(restored - clean))),
                                  "bitwise_equal_to_clean": bool(np.array_equal(restored, clean)),
                                  "reversal_present": _reversed(a_effects(restored))},
        "claim": ("causal effects WITHIN a predictor's inputs; not a molecular mechanism")}
    rng = np.random.default_rng([seed, STREAM_PERMUTATION])
    extra = rng.normal(size=(5, 4))
    batch = np.vstack([extra[:2], context, extra[2:]])
    order = rng.permutation(batch.shape[0])
    alone = np.array([predict("interaction", context[[i]])[0] for i in range(4)])
    inside = predict("interaction", batch)[2:6]
    reference_batch = predict("interaction", intervene_features(
        batch, full_space, "ablate", columns=("ab",), reference=reference))[2:6]
    reference_alone = np.array([predict("interaction", intervene_features(
        context[[i]], full_space, "ablate", columns=("ab",), reference=reference))[0]
        for i in range(4)])
    in_order = predict("interaction", batch)
    reordered = predict("interaction", batch[order])
    undone = np.empty_like(reordered)
    undone[order] = reordered
    independence = {
        "max_abs_difference_alone_vs_in_batch": float(np.max(np.abs(alone - inside))),
        "max_abs_difference_ablated_alone_vs_in_batch": float(np.max(np.abs(reference_alone
                                                                              - reference_batch))),
        "max_abs_difference_after_reordering": float(np.max(np.abs(undone - in_order))),
        "rule": ("raw predictions use only the row and frozen training-derived values; the "
                 "allocation's normalisation legitimately depends on its candidate support"),
        "batch_rows": int(batch.shape[0])}
    permutation = _permutation_check(problems["interaction"], config, permutations, seed)
    return {"design": {"truth": "u = 1.0*a + 0.5*b - 2.0*a*b; nuisance irrelevant",
                       "training": ("full factorial a,b,nuisance (8 unique feature contexts) x 3 "
                                    "replicates = 24 rows, noise-free"),
                       "support": ("the SAME 8 feature cells x 2 replicates = 16 rows; no unseen "
                                   "context"),
                       "scope": _planted_scope(problems["interaction"]),
                       "tau": config.tau, "useful_threshold": 0.25},
            "a_effects_by_context": {**effects,
                                     "truth": {"b0": planted_utility(1, 0) - planted_utility(0, 0),
                                               "b1": planted_utility(1, 1) - planted_utility(0, 1)}},
            "additive_reversal_present": _reversed(effects["additive"]),
            "interaction_reversal_present": _reversed(effects["interaction"]),
            "prediction_ranking": {name: reports[name]["prediction_ranking"] for name in variants},
            "policy_summary": {name: {entry["policy"]: {"objective_J": entry["objective_J"],
                                                        "oracle_regret": entry["oracle_regret"],
                                                        "expected_utility": entry["expected_utility"]}
                                      for entry in reports[name]["comparison"]}
                               for name in variants},
            "dpo": {name: {"training_pairs": artifacts[name]["fitted"]["pairs"],
                           "excluded_zero_prior_labels":
                           artifacts[name]["fitted"]["pairs"]["excluded_zero_prior_labels"]}
                    for name in variants},
            "interventions": interventions, "batch_independence": independence,
            "permutation_check": permutation,
            "identities_hold": {name: reports[name]["identities"]["all_hold"] for name in variants},
            "artifacts": artifacts, "reports": reports}


def _permutation_check(problem, config, permutations, seed):
    """Label-permutation null for the realised utility gain of the learned tilt.

    A fit to shuffled labels has no relation to the truth, but its realised gain need not be
    exactly zero. So the real gain is compared with the permutation distribution, and the mean rank
    agreement of the permuted predictions with the truth is reported as an independence check.

    A permutation whose predictions are constant has an UNDEFINED Spearman correlation. It stays
    ``None`` with its reason (it is never scored as 0.0, which would bias the mean toward zero):
    the defined and undefined permutation counts are reported and the mean is CONDITIONAL on the
    defined ones, or unavailable when none is defined.
    """
    permutations = _integer(permutations, "permutations", minimum=1)
    truth = np.array([planted_utility(a, b) for (a, b, _n) in (
        _cell_of(ident) for ident in problem.support_ids)])
    log_p = normalize_log_weights(problem.support_log_prior, name="prior")
    baseline = expected_value(log_p, truth)

    def gain(utilities):
        reward = fit_reward(problem.train_features, utilities, problem.space,
                            alpha=config.ridge_alpha)
        predicted = reward.predict(problem.support_features)
        log_q = log_tilt(log_p, predicted, config.tau)
        return expected_value(log_q, truth) - baseline, spearman(predicted, truth)
    real_gain, real_rho = gain(problem.train_utilities)
    rng = np.random.default_rng([seed, STREAM_PERMUTATION])
    gains, rhos = [], []
    for _ in range(permutations):
        g, rho = gain(rng.permutation(problem.train_utilities))
        gains.append(g)
        rhos.append(rho)
    at_least = int(sum(1 for g in gains if g >= real_gain))
    defined = [item["value"] for item in rhos if item["value"] is not None]
    undefined_reasons = {}
    for item in rhos:
        if item["value"] is None:
            undefined_reasons[item["reason"]] = undefined_reasons.get(item["reason"], 0) + 1
    conditional_mean = float(np.mean(defined)) if defined else None
    return {"permutations": permutations, "real_gain": float(real_gain),
            "permuted_gain_mean": float(np.mean(gains)), "permuted_gain_min": float(np.min(gains)),
            "permuted_gain_max": float(np.max(gains)), "permuted_at_least_real": at_least,
            "permutation_p_value": (1 + at_least) / (1 + permutations),
            "real_prediction_spearman": real_rho["value"],
            "real_prediction_spearman_reason": real_rho["reason"],
            "permuted_prediction_spearman": {
                "defined_permutations": len(defined),
                "undefined_permutations": permutations - len(defined),
                "undefined_reasons": undefined_reasons,
                "conditional_mean": conditional_mean,
                "mean_unavailable_reason": (None if defined else
                                            "no permutation had a defined rank correlation")},
            "permuted_prediction_spearman_mean": conditional_mean,
            "note": ("a realised random-label policy need not have exactly zero gain; the check is "
                     "the permutation null and the near-zero mean rank agreement over the "
                     "permutations where it is defined (undefined ones are counted, never scored "
                     "as 0), and the p-value is not a confidence interval")}


def _cell_of(identifier):
    """``(a, b, nuisance)`` from a planted candidate id such as ``eval-a1b0n1-r0``."""
    cell = identifier.split("-")[1]
    return int(cell[1]), int(cell[3]), int(cell[5])


# ---------------------------------------------------------------------------
# the hand-computed analytic cases
# ---------------------------------------------------------------------------

def _close(left, right, tolerance=1e-9):
    return bool(abs(float(left) - float(right)) <= tolerance)


def analytic_checks():
    """The hand-computed cases as executable checks: ``[{name, holds, detail}]``."""
    out = []

    def record(name, holds, detail):
        out.append({"name": name, "holds": bool(holds), "detail": detail})
    tau = 0.7
    star = np.exp(log_tilt(np.log([0.9, 0.1]), [0.0, tau * math.log(9)], tau))
    flat = np.exp(log_tilt(np.log([0.9, 0.1]), [3.0, 3.0], tau))
    record("oracle_tilt_halves_a_9_to_1_prior", _close(star[0], 0.5) and _close(star[1], 0.5)
           and _close(flat[0], 0.9), {"oracle": star.tolist(), "constant_utility": flat.tolist()})
    p = np.log([0.5, 0.5])
    u = [0.0, math.log(3)]
    star2 = np.exp(log_tilt(p, u, 1.0))
    regret = oracle_regret(p, p, u, 1.0)["regret"]
    record("oracle_regret_at_the_prior", _close(star2[1], 0.75)
           and _close(regret, math.log(2) - 0.5 * math.log(3)),
           {"oracle": star2.tolist(), "regret": regret,
            "expected": math.log(2) - 0.5 * math.log(3)})
    true_u, pred_u = np.array([math.log(3), 0.0]), np.array([0.0, math.log(3)])
    log_hat = log_tilt(p, pred_u, 1.0)
    gap = kl_divergence(p, log_hat)
    delta = objective(p, p, true_u, 1.0) - objective(log_hat, p, true_u, 1.0)
    cross = expected_value(p, true_u - pred_u) - expected_value(log_hat, true_u - pred_u)
    record("predicted_gap_is_not_the_true_improvement",
           gap > 0 and _close(delta, math.log(1.5)) and _close(delta, -gap + cross),
           {"predicted_gap": gap, "true_J_difference": delta, "cross_term": cross})
    counts = np.array([3, 1])
    law, info = sir_weighted_distribution(np.log([0.5, 0.5]), np.log([0.75, 0.25]), [0.0, 0.0],
                                          1.0, counts)
    repeated, repeated_info = sir_weighted_distribution(np.log([0.5, 0.5]), np.log([0.5, 0.5]),
                                                        [0.0, 0.0], 1.0, np.array([4, 0]))
    record("sir_ess_counts_occurrences", _close(np.exp(law[0]), 0.5)
           and _close(info["effective_sample_size"], 3.0)
           and _close(repeated_info["effective_sample_size"], 4.0)
           and repeated_info["distinct_candidates"] == 1,
           {"corrected_mass": np.exp(law).tolist(), "ess": info["effective_sample_size"],
            "identical_proposals_ess": repeated_info["effective_sample_size"]})
    log_u = log_tilt(np.log([0.5, 0.5]), [0.0, -1000.0], 1.0)
    kl_underflow = kl_divergence(np.log([0.5, 0.5]), log_u)
    record("underflowed_probability_gives_finite_kl",
           _close(kl_underflow, 500.0 - math.log(2), 1e-9) and math.isfinite(kl_underflow),
           {"kl": kl_underflow, "expected": 500.0 - math.log(2)})
    space = FeatureSpace(kind="explicit", names=("x",), schema={}, history=("explicit",),
                         supervision=SupervisionHistory(status=HISTORY_LABEL_FREE))
    dpo = fit_dpo([[-1.0], [1.0]], [1], [0], space, beta=1.0, l2=1.0 / math.log(3))
    theta = float(dpo.theta[0])
    uniform = np.exp(normalize_log_weights(np.log([0.5, 0.5]) + theta * np.array([-1.0, 1.0])))
    skewed = np.exp(normalize_log_weights(np.log([0.75, 0.25]) + theta * np.array([-1.0, 1.0])))
    record("dpo_optimum_matches_the_closed_form",
           _close(theta, math.log(3) / 2, 1e-8) and _close(uniform[1], 0.75, 1e-8)
           and _close(skewed[0], 0.5, 1e-8),
           {"theta": theta, "expected": math.log(3) / 2, "uniform_prior": uniform.tolist(),
            "prior_.75_.25": skewed.tolist(), "stop_reason": dpo.diagnostics["stop_reason"]})
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(50):
        prior = np.where(rng.random(6) < 0.3, -np.inf, rng.normal(size=6))
        prior[0] = 0.0
        q_log = np.where(np.isfinite(prior), rng.normal(size=6), -np.inf)
        q_log = q_log - logsumexp(q_log)
        outcome = oracle_regret(q_log, prior, rng.normal(size=6), 0.8)
        worst = max(worst, abs(outcome["regret"] - outcome["tau_kl_to_oracle"]))
    record("oracle_regret_identity_on_random_q_with_structural_zeros", worst < 1e-9,
           {"max_abs_residual": worst, "trials": 50})
    return out


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

SELFTEST_ALPHABET = "ACDE"
SELFTEST_LENGTH = 4


def write_synthetic_inputs(directory, seed=0):
    """Small explicit CSV/manifest inputs for the finite-support pipeline. Returns their paths.

    Sequences over a 4-letter alphabet, groups = the first two residues (so no group straddles
    the split), an additive planted utility with mild noise, and a declared joint log probability
    from an independent per-position categorical, with one forbidden motif as a structural zero.
    The grouping is a shared-prefix rule only: it keeps whole groups on one side of the split but
    enforces NO minimum sequence distance, so a training and a held-out sequence can differ by a
    single residue (they do, in the shipped data).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, 100])
    everything = ["".join(item) for item in itertools.product(SELFTEST_ALPHABET,
                                                              repeat=SELFTEST_LENGTH)]
    chosen = sorted(int(i) for i in rng.choice(len(everything), size=160, replace=False))
    sequences = [everything[i] for i in chosen]
    groups = sorted({sequence[:2] for sequence in sequences})
    held_out = {str(g) for g in rng.choice(groups, size=5, replace=False)}
    weights = rng.normal(size=(SELFTEST_LENGTH, len(SELFTEST_ALPHABET)))
    categorical = rng.dirichlet(np.full(len(SELFTEST_ALPHABET), 2.0), size=SELFTEST_LENGTH)
    lookup = {residue: i for i, residue in enumerate(SELFTEST_ALPHABET)}
    candidate_lines, train_lines, eval_lines = [], [], []
    for number, sequence in enumerate(sequences):
        ident = f"c{number:03d}"
        split = SPLIT_EVAL if sequence[:2] in held_out else SPLIT_TRAIN
        utility = float(sum(weights[p, lookup[r]] for p, r in enumerate(sequence))
                        + rng.normal(scale=0.1))
        forbidden = sequence[2:] == "AA"
        prior = "-inf" if forbidden else repr(float(sum(math.log(categorical[p, lookup[r]])
                                                         for p, r in enumerate(sequence))))
        candidate_lines.append(f"{ident},{sequence},g-{sequence[:2]},{split},{prior}")
        (eval_lines if split == SPLIT_EVAL else train_lines).append(f"{ident},{utility!r}")
    paths = {"candidates": directory / "candidates.csv",
             "train_outcomes": directory / "train_outcomes.csv",
             "eval_outcomes": directory / "eval_outcomes.csv",
             "manifest": directory / "manifest.json"}
    paths["candidates"].write_text("candidate_id,sequence,group_id,split,prior_log_weight\n"
                                   + "\n".join(candidate_lines) + "\n", encoding="utf-8")
    paths["train_outcomes"].write_text("candidate_id,utility\n" + "\n".join(train_lines) + "\n",
                                       encoding="utf-8")
    paths["eval_outcomes"].write_text("candidate_id,utility\n" + "\n".join(eval_lines) + "\n",
                                      encoding="utf-8")
    write_json(paths["manifest"], {
        "target": "synthetic-additive-target", "assay": "synthetic-closed-form",
        "utility": {"name": "planted additive utility", "units": "arbitrary", "scale": "linear",
                    "higher_is_better": True, "useful_threshold": 1.5},
        "grouping_rationale": ("groups are the first two residues; whole groups are held out so "
                               "no group straddles the split. This is a shared-prefix grouping "
                               "only: it does not enforce a minimum sequence distance, and "
                               "sequences in different groups can differ by a single residue"),
        "features": {"kind": "sequence_onehot", "alphabet": SELFTEST_ALPHABET,
                     "length": SELFTEST_LENGTH,
                     "provenance": {"status": "label_free",
                                    "description": "fixed one-hot encoding, nothing learned",
                                    "sequence_only_exposure": {"status": "none"}}},
        "prior": {"kind": PRIOR_JOINT,
                  "semantics": ("log probability of the whole sequence under an independent "
                                "per-position categorical; sequences ending in AA are forbidden "
                                "(structural zero)"),
                  "provenance": {"status": "label_free",
                                 "description": "synthetic categorical, no outcomes used",
                                 "sequence_only_exposure": {"status": "none"}}},
        "notes": "synthetic selftest inputs"})
    return paths


def _shuffled_outcomes(path_in, path_out, seed):
    rows = [line for line in Path(path_in).read_text(encoding="utf-8").splitlines()[1:] if line]
    ids = [line.split(",")[0] for line in rows]
    values = [line.split(",")[1] for line in rows]
    order = np.random.default_rng([seed, 101]).permutation(len(values))
    Path(path_out).write_text("candidate_id,utility\n" + "\n".join(
        f"{ident},{values[int(i)]}" for ident, i in zip(ids, order)) + "\n", encoding="utf-8")


def selftest(output_dir, seed=0):
    """Run the whole baseline on synthetic data and write a real JSON report. Returns it.

    Executes the hand-computed analytic cases, the planted interaction experiment (fit, allocate
    and evaluate an additive and an interaction predictor, then interventions), and the CSV
    finite-support pipeline end to end. A second pipeline run with SHUFFLED evaluation outcomes
    must produce a byte-identical frozen artifact: the evaluation outcomes cannot have influenced
    anything frozen. ``passed`` is true only if every named check holds.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, holds, detail=None):
        checks.append({"name": name, "holds": bool(holds), "detail": detail})
    analytic = analytic_checks()
    for item in analytic:
        check(f"analytic::{item['name']}", item["holds"])
    planted = run_planted_experiment(seed)
    for name, artifact in planted["artifacts"].items():
        write_json(out / "planted" / f"{name}_frozen_artifact.json", artifact)
    iv = planted["interventions"]
    effects = planted["a_effects_by_context"]
    check("planted::additive_predictor_misses_the_context_reversal",
          not planted["additive_reversal_present"] and abs(effects["additive"]["b0"]) < 1e-6
          and abs(effects["additive"]["b1"]) < 1e-6, effects["additive"])
    check("planted::interaction_ridge_learns_the_reversal",
          planted["interaction_reversal_present"] and abs(effects["interaction"]["b0"] - 1.0) < 1e-4
          and abs(effects["interaction"]["b1"] + 1.0) < 1e-4, effects["interaction"])
    ranking = planted["prediction_ranking"]
    check("planted::interaction_ranks_the_held_out_better",
          ranking["interaction"]["all_candidates"]["spearman"]["value"]
          > ranking["additive"]["all_candidates"]["spearman"]["value"] + 0.3)
    regrets = {name: planted["policy_summary"][name]["ridge_tilt"]["oracle_regret"]["value"]
               for name in ("additive", "interaction")}
    check("planted::interaction_tilt_has_smaller_oracle_regret",
          regrets["interaction"] < regrets["additive"] and regrets["interaction"] < 1e-6, regrets)
    check("planted::identity_intervention_is_exact", iv["identity"]["bitwise_equal"])
    check("planted::interaction_ablation_removes_the_reversal",
          not iv["ablate_interaction"]["reversal_present"], iv["ablate_interaction"])
    check("planted::nuisance_ablation_has_no_relevant_effect",
          iv["ablate_nuisance"]["max_abs_prediction_change"] < 1e-9
          and iv["ablate_nuisance"]["reversal_present"], iv["ablate_nuisance"])
    suff = iv["donor_patch_sufficiency"]
    check("planted::donor_patch_flips_the_local_effect_and_removes_the_cross_context_reversal",
          suff["local_negative_effect_introduced"] and suff["cross_context_reversal_before"]
          and not suff["cross_context_reversal_after"] and suff["cross_context_reversal_removed"]
          and abs(suff["prediction_change"] - suff["interaction_weight_times_delta_ab"]) < 1e-9
          and abs(suff["nuisance_patch_prediction_change"]) < 1e-9, suff)
    check("planted::restoration_rescues_the_ablation",
          iv["rescue_after_ablation"]["bitwise_equal_to_clean"]
          and iv["rescue_after_ablation"]["reversal_present"], iv["rescue_after_ablation"])
    independence = planted["batch_independence"]
    check("planted::predictions_ignore_unrelated_batch_composition",
          max(independence["max_abs_difference_alone_vs_in_batch"],
              independence["max_abs_difference_ablated_alone_vs_in_batch"],
              independence["max_abs_difference_after_reordering"]) < 1e-12, independence)
    permutation = planted["permutation_check"]
    check("planted::real_gain_beats_the_permutation_null",
          permutation["permutation_p_value"] <= 0.05
          and permutation["permuted_prediction_spearman_mean"] is not None
          and abs(permutation["permuted_prediction_spearman_mean"]) < 0.2, permutation)
    check("planted::every_identity_holds", all(planted["identities_hold"].values()),
          planted["identities_hold"])
    paths = write_synthetic_inputs(out / "synthetic", seed)
    config = BaselineConfig(tau=1.0, seed=seed, ridge_alpha=1.0, label_budget=40,
                            sir_proposal_draws=512, final_draws=64)
    summary = run_finite_support(
        candidates=paths["candidates"], train_outcomes=paths["train_outcomes"],
        eval_outcomes=paths["eval_outcomes"], manifest=paths["manifest"],
        output_dir=out / "finite_support", config=config, overwrite=True)
    shuffled = out / "synthetic" / "eval_outcomes_shuffled.csv"
    _shuffled_outcomes(paths["eval_outcomes"], shuffled, seed)
    second = run_finite_support(
        candidates=paths["candidates"], train_outcomes=paths["train_outcomes"],
        eval_outcomes=shuffled, manifest=paths["manifest"],
        output_dir=out / "finite_support_shuffled_evaluation", config=config, overwrite=True)
    first_report = read_json(summary["report"])
    second_report = read_json(second["report"])
    check("pipeline::every_identity_holds", first_report["identities"]["all_hold"])
    check("pipeline::evaluation_outcomes_cannot_change_the_frozen_artifact",
          summary["artifact_sha256"] == second["artifact_sha256"]
          and (out / "finite_support" / FROZEN_NAME).read_bytes()
          == (out / "finite_support_shuffled_evaluation" / FROZEN_NAME).read_bytes()
          and first_report["oracle"] != second_report["oracle"],
          {"artifact_sha256": summary["artifact_sha256"]})
    check("pipeline::label_budget_is_accounted",
          first_report["label_accounting"]["acquired_labels"] == 40
          and first_report["label_accounting"]["offered_labels"] > 40
          and first_report["clean_low_n"]["status"] == "eligible",
          first_report["label_accounting"])
    tampered = read_json(summary["frozen_artifact"])
    tampered["predictions"]["utility"][0] += 1.0
    try:
        verify_artifact(tampered)
        refused = False
    except ValueError:
        refused = True
    check("pipeline::a_tampered_artifact_is_refused", refused)
    report = {
        "schema": SELFTEST_SCHEMA, "seed": seed, "passed": bool(all(c["holds"] for c in checks)),
        "planted_scope": planted["design"]["scope"],
        "checks": checks, "analytic_cases": analytic,
        "planted_experiment": {key: value for key, value in planted.items()
                               if key not in ("artifacts", "reports")},
        "planted_reports": planted["reports"],
        "finite_support": {"summary": summary, "report": first_report},
        "inputs": {key: str(value) for key, value in paths.items()},
        "claims": dict(CLAIMS),
        "note": ("synthetic qualification only: this shows the machinery computes what it says on "
                 "a known truth, not that any native model or biology improves. The planted toy "
                 "is a fit/intervention qualification on a replicated full factorial (8 unique "
                 "feature contexts, 24 training rows) with the interaction coordinate supplied and "
                 "evaluation on the same feature cells; it is not unseen-context generalisation")}
    write_json(out / "selftest_report.json", report)
    return report
