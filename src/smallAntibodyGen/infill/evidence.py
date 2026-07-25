"""Order-averaged evidence estimator for HCDR3 infill candidates.

This module computes ``E-hat``, the **order-averaged path log-likelihood** of a
filled HCDR3, teacher-forced under the generation model's own MLM head.

What E-hat is (and is NOT)
--------------------------
For a candidate whose HCDR3 has ``L`` residue positions, pick a permutation
``sigma`` of those positions (an *unmasking order*). Starting from a fully-masked
span, reveal one position at a time in that order, and at each step read the
model's log-probability of the candidate's TRUE residue at the just-revealed
position under the current partially-filled state (teacher forcing). Summing
those ``L`` terms gives one order's path log-likelihood; ``E-hat`` averages that
sum over ``K`` random orders::

    E-hat = (1/K) sum_k sum_t log p( x_{sigma_k(t)} | x_{sigma_k(<t)}, scaffold, antigen )

Each fixed order defines a valid autoregressive factorization, so its sum is a
valid joint log-likelihood; the average over orders is the Monte-Carlo estimate
of the *order-averaged* path log-likelihood, a fixed, order-independent scalar
per candidate. It is **order-invariant in expectation** -- the property the
single-path infill/guided scores lack.

``E-hat`` is **never** claimed to equal ``log p(x)``. A masked LM's per-position
conditionals are not guaranteed mutually consistent with a single joint, so
different orders define different (valid) factorizations; by Jensen the average
lower-bounds the log-likelihood of the implied order-mixture.

Why it exists here
------------------
``hcdr3.py``'s ``guided_infill`` docstring pins that ``log_probability`` /
``mean_log_probability`` are single-path scores that must **not** be pooled
across the ``infill`` and ``guided_infill`` samplers -- even at
``guidance_strength == 0`` -- because one sums independent marginals from a
single fully-masked forward while the other sums conditionals along one committed
unmasking path. ``E-hat`` is the separately-defined, sampler-independent quantity
that retires that problem *for evidence only*: it averages over orders rather
than committing to one, so two candidates produced by different samplers are
comparable by ``E-hat``. The non-pooling rule stays fully in force for
``log_probability`` / ``mean_log_probability``.

Encoding parity
---------------
Encoding drift between training and inference is this repo's most expensive bug
class, so this module NEVER re-implements tokenization, span math, or masked-state
construction. It takes a live ``FixedLengthHCDR3Infiller`` and reuses
``_encode_antibody_with_masked_hcdr3`` (the byte-identical masked base state
guided decoding uses), ``_encode_antigen`` (the generation antigen stream),
``canonical_token_ids`` and ``tokenizer``. The evidence term is always scored
under ``infiller.model`` and the *generation* antigen stream, never an attached
guidance model: E-hat is the generation model's own likelihood of the residues.

Determinism
-----------
Orders are sampled from an explicit ``seed`` or ``torch.Generator``; the model
forwards are ``eval()`` + ``no_grad`` and therefore deterministic on CPU. Same
seed => identical orders => identical ``E-hat``. Rows and candidates never share
hidden global RNG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from smallAntibodyGen.infill.hcdr3 import HCDR3Span


@dataclass(frozen=True)
class EvidenceEstimate:
    """
    One candidate's order-averaged evidence and its self-consistency numbers.

    Attributes:
        evidence:
            ``E-hat`` -- the mean over ``num_orders`` random unmasking orders of
            the teacher-forced path log-likelihood sum.
        evidence_se:
            Monte-Carlo standard error of ``E-hat`` across the ``K`` order sums,
            ``sample_std(per_order, ddof=1) / sqrt(K)``. ``nan`` when ``K < 2``.
        num_orders:
            ``K``, the number of orders averaged.
        length:
            ``L``, the number of HCDR3 positions (== candidate length).
        per_order:
            The ``K`` per-order path-log-likelihood sums, in sampled order.
        half1_mean, half2_mean:
            Means of the two disjoint halves of ``per_order`` (first ``K//2`` vs
            the rest) -- the half-sample self-consistency check.
        half_delta:
            ``|half1_mean - half2_mean|``. Self-consistency passes when this sits
            within the reported Monte-Carlo error.
        half_within_2se:
            ``True`` iff ``half_delta <= 2 * evidence_se`` (both finite). This is
            a POPULATION-level diagnostic; it is far too noisy to gate a single
            candidate on.
    """

    evidence: float
    evidence_se: float
    num_orders: int
    length: int
    per_order: tuple[float, ...]
    half1_mean: float
    half2_mean: float
    half_delta: float
    half_within_2se: bool


@dataclass(frozen=True)
class OrderSummary:
    """Pure statistics of a set of per-order path sums (no model involved)."""

    mean: float
    se: float
    half1_mean: float
    half2_mean: float
    half_delta: float
    half_within_2se: bool


def summarize_orders(per_order: Sequence[float]) -> OrderSummary:
    """
    Reduce per-order path sums to ``E-hat``, the MC standard error, and the
    half-sample self-check -- with no model dependency, so the arithmetic is
    unit-testable in isolation.

    ``se`` is the sample standard deviation with ``ddof = 1`` divided by
    ``sqrt(K)``. The two halves are the first ``K // 2`` orders and the remaining
    ``K - K // 2`` (disjoint; equal in size only when ``K`` is even). For
    ``K == 1`` the SE and half-check are ``nan`` / ``False`` -- a single order
    cannot self-check, and reporting 0.0 there would read as "measured, no
    error".
    """
    values = [float(v) for v in per_order]
    k = len(values)
    if k == 0:
        raise ValueError("summarize_orders requires at least one order")
    mean = math.fsum(values) / k

    if k >= 2:
        var = math.fsum((v - mean) ** 2 for v in values) / (k - 1)
        se = math.sqrt(var / k)
    else:
        se = float("nan")

    half = k // 2
    first = values[:half]
    second = values[half:]
    if first and second:
        half1 = math.fsum(first) / len(first)
        half2 = math.fsum(second) / len(second)
        half_delta = abs(half1 - half2)
    else:
        half1 = mean
        half2 = float("nan")
        half_delta = float("nan")

    within = math.isfinite(se) and math.isfinite(half_delta) and half_delta <= 2.0 * se
    return OrderSummary(
        mean=mean,
        se=se,
        half1_mean=half1,
        half2_mean=half2,
        half_delta=half_delta,
        half_within_2se=within,
    )


def sample_orders(
    length: int,
    num_orders: int,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """
    Sample ``num_orders`` uniformly random unmasking permutations of
    ``range(length)``.

    Determinism: pass either an explicit ``seed`` (a fresh CPU ``torch.Generator``
    is seeded with it) or a ``torch.Generator``. For ``length == 1`` every order
    is ``[0]``.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if num_orders <= 0:
        raise ValueError("num_orders must be positive")
    if generator is None:
        if seed is None:
            raise ValueError(
                "provide a seed or a torch.Generator for deterministic orders"
            )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return [torch.randperm(length, generator=generator).tolist() for _ in range(num_orders)]


def _prepare(
    infiller: Any,
    record: Any,
    candidate_hcdr3: str,
):
    """
    Build every tensor the teacher-forcing loop needs, reusing the infiller.

    Returns ``(L, base_ids, base_attn, mask_positions, antigen_ids, antigen_attn,
    true_token_ids, true_canonical_idx)``. All model inputs come from the
    infiller's own encoders -- nothing is re-implemented here.

    ``true_canonical_idx`` maps each candidate residue to its index in the
    infiller's ``canonical_token_ids`` order (so a log-softmax over those ids can
    be read by index); ``true_token_ids`` are the tokenizer ids to teacher-force
    into the state. Fails loudly on an empty candidate, an invalid span, or a
    non-canonical residue.
    """
    candidate = (candidate_hcdr3 or "").strip().upper()
    if not candidate:
        raise ValueError("candidate_hcdr3 is empty")

    span = HCDR3Span.from_record(record)
    length = len(candidate)
    base_ids, base_attn, mask_positions, _prefix, _suffix = (
        infiller._encode_antibody_with_masked_hcdr3(record, span, proposed_length=length)
    )
    antigen_ids, antigen_attn = infiller._encode_antigen(record)

    if len(mask_positions) != length:
        raise ValueError(
            f"encoding produced {len(mask_positions)} mask positions for a "
            f"{length}-residue candidate; span/encoding drift"
        )

    # Map candidate residues to canonical indices via the infiller's OWN canonical
    # id list, so index alignment with a log-softmax over canonical_token_ids
    # holds regardless of any CANONICAL_AA drift.
    canonical_residues = [
        infiller.tokenizer.id_to_token[tid] for tid in infiller.canonical_token_ids
    ]
    index_of = {residue: i for i, residue in enumerate(canonical_residues)}
    true_canonical_idx: list[int] = []
    for char in candidate:
        if char not in index_of:
            raise ValueError(
                f"candidate residue {char!r} is not one of the canonical amino acids"
            )
        true_canonical_idx.append(index_of[char])
    true_token_ids = [infiller.canonical_token_ids[i] for i in true_canonical_idx]
    return (
        length,
        base_ids,
        base_attn,
        mask_positions,
        antigen_ids,
        antigen_attn,
        true_token_ids,
        true_canonical_idx,
    )


def _validate_orders(orders: Sequence[Sequence[int]], length: int) -> list[list[int]]:
    """Validate each order is a permutation of ``range(length)``; return as lists."""
    if len(orders) == 0:
        # Without this guard an empty explicit `orders=` fails deep inside the
        # tensor code (IndexError on a 0-row batch) instead of naming the cause.
        raise ValueError("orders must be a non-empty sequence of permutations")
    validated: list[list[int]] = []
    expected = set(range(length))
    for order in orders:
        order_list = [int(i) for i in order]
        if set(order_list) != expected or len(order_list) != length:
            raise ValueError(
                f"each order must be a permutation of range({length}); got {order_list}"
            )
        validated.append(order_list)
    return validated


@torch.no_grad()
def path_step_logprobs(
    infiller: Any,
    record: Any,
    candidate_hcdr3: str,
    orders: Sequence[Sequence[int]],
) -> torch.Tensor:
    """
    Teacher-forced per-step log-probabilities for a batch of unmasking orders.

    For each of the ``K`` orders it walks the ``L`` reveal steps, teacher-forcing
    the candidate's true residue at each revealed position and recording the
    model's log-probability of that residue under the current partial state.

    Batching: all ``K`` orders advance together as one batch-of-``K`` forward per
    step, so the whole computation costs ``L`` forwards rather than ``K*L``. The
    transformer has no cross-example ops (LayerNorm is per-token, attention is
    within-sequence), so each batch row evolves independently and the batched
    result equals the per-order sequential result. The ~20-way enumeration used
    by *guidance* is not needed here -- only the single true residue is scored per
    step.

    The critical correctness surface: the logits at each step are read while the
    revealed position is still ``[MASK]``. Teacher-forcing happens AFTER the read.
    Revealing first would let the model see the answer it is being scored on.

    Returns a ``[K, L]`` CPU tensor whose ``[k, t]`` entry is
    ``log p(true residue revealed at step t of order k | state)``. Row sums are
    the per-order path log-likelihoods.
    """
    (
        length,
        base_ids,
        base_attn,
        mask_positions,
        antigen_ids,
        antigen_attn,
        true_token_ids,
        true_canonical_idx,
    ) = _prepare(infiller, record, candidate_hcdr3)

    orders = _validate_orders(orders, length)
    k = len(orders)
    device = infiller.device
    model = infiller.model
    model.eval()

    working = base_ids.repeat(k, 1).clone()  # [K, S]
    attn = base_attn.repeat(k, 1)  # [K, S]
    antigen_ids_k = antigen_ids.repeat(k, 1)  # [K, T]
    antigen_attn_k = antigen_attn.repeat(k, 1)  # [K, T]

    canonical_ids = torch.tensor(
        infiller.canonical_token_ids, dtype=torch.long, device=device
    )
    orders_t = torch.tensor(orders, dtype=torch.long, device=device)  # [K, L]
    mask_pos_t = torch.tensor(mask_positions, dtype=torch.long, device=device)  # [L]
    true_tok_t = torch.tensor(true_token_ids, dtype=torch.long, device=device)  # [L]
    true_ci_t = torch.tensor(true_canonical_idx, dtype=torch.long, device=device)  # [L]
    row_idx = torch.arange(k, device=device)

    # The antigen is constant across every step, so encode it once (same
    # amortization, and the same bit-exactness argument, as guided_infill).
    antigen_state = model.encode_antigen(antigen_ids_k, antigen_attn_k)

    step_logprobs = torch.empty(k, length, device=device)
    for step in range(length):
        slot_idx = orders_t[:, step]  # [K] which span position is revealed now
        token_pos = mask_pos_t[slot_idx]  # [K] token index of that position
        mlm_logits, _ = model(
            antibody_input_ids=working,
            antibody_attention_mask=attn,
            antigen_input_ids=antigen_ids_k,
            antigen_attention_mask=antigen_attn_k,
            antigen_state=antigen_state,
        )
        row_logits = mlm_logits[row_idx, token_pos]  # [K, V]
        canonical_logits = row_logits[:, canonical_ids]  # [K, num_canonical]
        logprobs = torch.log_softmax(canonical_logits, dim=-1)
        true_ci = true_ci_t[slot_idx]  # [K]
        step_logprobs[:, step] = logprobs[row_idx, true_ci]
        # Teacher-force AFTER the read: commit the TRUE residue so later steps
        # condition on it.
        working[row_idx, token_pos] = true_tok_t[slot_idx]

    return step_logprobs.cpu()


def path_loglik_for_orders(
    infiller: Any,
    record: Any,
    candidate_hcdr3: str,
    orders: Sequence[Sequence[int]],
) -> list[float]:
    """Per-order path log-likelihood sums (row sums of :func:`path_step_logprobs`)."""
    step = path_step_logprobs(infiller, record, candidate_hcdr3, orders)
    return [float(v) for v in step.sum(dim=1).tolist()]


def estimate_evidence(
    infiller: Any,
    record: Any,
    candidate_hcdr3: str,
    *,
    num_orders: int,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    orders: Sequence[Sequence[int]] | None = None,
) -> EvidenceEstimate:
    """
    Estimate ``E-hat`` for one filled candidate over ``K`` random unmasking orders.

    Args:
        infiller: a ``FixedLengthHCDR3Infiller`` -- its encoders/model are reused.
        record: the antibody-antigen record.
        candidate_hcdr3: the filled HCDR3 string (``L`` canonical residues).
        num_orders: ``K`` (ignored when ``orders`` is given).
        seed: seed for order sampling (mutually sufficient with ``generator``).
        generator: a CPU ``torch.Generator`` for order sampling.
        orders: explicit orders instead of sampling -- a reproducibility/testing
            seam. When given, ``seed``/``generator`` are unused and ``num_orders``
            is taken from its length.
    """
    candidate = (candidate_hcdr3 or "").strip().upper()
    if not candidate:
        raise ValueError("candidate_hcdr3 is empty")
    length = len(candidate)

    if orders is None:
        if generator is None and seed is None:
            raise ValueError(
                "provide seed, generator, or explicit orders for deterministic evidence"
            )
        orders = sample_orders(length, num_orders, seed=seed, generator=generator)

    per_order = path_loglik_for_orders(infiller, record, candidate, orders)
    summary = summarize_orders(per_order)
    return EvidenceEstimate(
        evidence=summary.mean,
        evidence_se=summary.se,
        num_orders=len(per_order),
        length=length,
        per_order=tuple(per_order),
        half1_mean=summary.half1_mean,
        half2_mean=summary.half2_mean,
        half_delta=summary.half_delta,
        half_within_2se=summary.half_within_2se,
    )


# ---------------------------------------------------------------------------
# Pure ranking / demotion helpers.
#
# These live in an importable library module rather than in a scripts/*.py so
# their arithmetic is unit-testable without loading a script, and so every
# consumer routes ranking through ONE definition. Ties are broken by a
# deterministic stable sort (ascending original index among equal scores).
# ---------------------------------------------------------------------------
def to_json_safe(value: Any) -> Any:
    """
    Recursively convert non-finite floats (NaN, +/-inf) to ``None`` for JSON.

    RFC 8259 JSON has no representation for NaN/Infinity; Python's
    ``json.dumps`` default (``allow_nan=True``) emits the non-standard literal
    ``NaN``, producing a file that strict parsers reject. Sanitize at the
    SERIALIZATION boundary only: in-memory values keep their honest NaN semantics
    (``evidence_se`` is NaN exactly when ``num_orders < 2``), and the written JSON
    carries ``null`` there with the accompanying count fields stating why.

    Handles dicts, lists, and tuples recursively; every non-float scalar passes
    through unchanged (bools are not floats and are preserved).
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def rank_topk_stable(scores: Sequence[float], k: int) -> list[int]:
    """
    Indices of the top-``k`` scores, highest first, deterministic under ties.

    The sort key ``(-score, index)`` is a stable, fully-deterministic descending
    sort: among equal scores the lower original index ranks first. Returns
    ``min(k, len(scores))`` indices.
    """
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[: max(0, k)]


def boundary_tie_count(scores: Sequence[float], k: int) -> int:
    """
    Number of candidates tied at the top-``k`` cutoff score.

    This is the tie-group size the deterministic sort had to split when it drew
    the top-``k`` boundary. A large count means the reported top-``k`` is an
    artifact of the tie-break, not of the scores. ``0`` when the cutoff is
    degenerate (``k <= 0`` or ``k >= len``).
    """
    n = len(scores)
    if k <= 0 or k >= n:
        return 0
    cutoff = sorted(scores, reverse=True)[k - 1]
    return sum(1 for s in scores if s == cutoff)


def demotion_rate_for_record(
    judge_scores: Sequence[float],
    evidence_scores: Sequence[float],
    k: int,
) -> tuple[float, int, int]:
    """
    Per-record demotion rate: how much evidence-ranking disagrees with the judge.

    The fraction of the top-``k``-by-judge candidates that are NOT in the
    top-``k``-by-evidence, at the SAME ``k`` on both sides, ties broken by
    :func:`rank_topk_stable`.

    This is the measurement behind a real tension: evidence rewards TYPICALITY
    (a candidate the model finds likely), while the compatibility judge rewards
    predicted binding. Ranking by evidence can therefore demote exactly the
    unusual candidates a steering method exists to find, and this number is how
    you see that happening instead of assuming it does not.

    Returns ``(demotion_fraction, judge_boundary_ties, evidence_boundary_ties)``.
    The denominator is ``min(k, num_candidates)``.
    """
    if len(judge_scores) != len(evidence_scores):
        raise ValueError("judge_scores and evidence_scores must have equal length")
    n = len(judge_scores)
    if n == 0:
        return 0.0, 0, 0
    judge_top = rank_topk_stable(judge_scores, k)
    evidence_top = set(rank_topk_stable(evidence_scores, k))
    denom = len(judge_top)
    demoted = sum(1 for i in judge_top if i not in evidence_top)
    fraction = demoted / denom if denom else 0.0
    return (
        fraction,
        boundary_tie_count(judge_scores, k),
        boundary_tie_count(evidence_scores, k),
    )
