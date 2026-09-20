"""CHES: centered hidden embedding similarity over the ten causally shifted core states.

Reference: Razin et al., *Unintentional Unalignment: Likelihood Displacement in
Direct Preference Optimization* (arXiv:2410.08847), sections 4-5. For one
preference pair::

    s_plus  = sum_t h_plus[t]
    s_minus = sum_t h_minus[t]
    CHES    = dot(s_plus, s_minus) - dot(s_plus, s_plus)

The score is theory-motivated and empirically useful. It is **not** a causal
certificate for the full transformer/AdamW trajectory, and this module never
reports it as one.

Four implementation facts carry the correctness of the number:

* **Exactly ten states, causally shifted.** The last scaffold position (index 98
  of the 99-token prefix) is the state that predicts core residue 1; core input
  positions 0..8 predict residues 2..10. The state *after* the last core residue
  predicts nothing in this scaffold and is excluded, and the other 98 scaffold
  positions are not pooled in.
* **They are the final-normalized states, before the output head.** Proven, not
  asserted: :func:`head_reconstruction` recomputes the canonical logits from the
  extracted states through the model's own output embedding and compares them to
  ordinary teacher forcing at a declared tolerance.
* **Chosen and rejected never share a mutable prefix cache.** The GPTNeoX dynamic
  cache is mutated in place by ``batch_repeat_interleave`` and then *appended* to
  by the second forward. A cache built once and handed to both responses would
  make the second one read the first one's keys. Each call builds its own.
* **The reduction is float64.** The forward stays in the model's native float32 --
  that is the computation being measured -- and the sum over ten positions and the
  two dot products are accumulated in double, where a 768-wide dot of values near
  unit scale would otherwise lose several digits.
"""
from __future__ import annotations

import math

import numpy as np

from .her2_data import CORE_LENGTH, PREFIX_LENGTH
from .her2_runtime import require

CHES_SCHEMA = "her2-ches/1"

#: Index of the prefix state that predicts the first core residue.
PREFIX_PREDICTING_INDEX = PREFIX_LENGTH - 1          # 98
#: The ten extracted positions, named once so a report cannot describe a different set.
EXTRACTED_POSITIONS = ("prefix[98]",) + tuple(f"core_input[{i}]" for i in range(CORE_LENGTH - 1))


# ---------------------------------------------------------------------------
# model seams
# ---------------------------------------------------------------------------

def backbone_and_head(model):
    """``(decoder, output_head)`` for a causal LM, refusing anything ambiguous."""
    decoder = None
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
    if decoder is None:
        decoder = getattr(model, "gpt_neox", None)
    require(decoder is not None and decoder is not model,
            "CHES needs the decoder stack separately from the output head; this model exposes "
            "neither get_decoder() nor a gpt_neox attribute")
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None:
        head = getattr(model, "embed_out", None)
    require(head is not None,
            "CHES needs the output head to prove the extracted states are the pre-head ones")
    return decoder, head


# ---------------------------------------------------------------------------
# the ten states
# ---------------------------------------------------------------------------

def full_hidden_states(policy, core_ids):
    """``(B, 10, H)`` final-normalized states by ordinary full teacher forcing."""
    import torch
    decoder, _ = backbone_and_head(policy.model)
    batch = core_ids.shape[0]
    inputs = torch.cat([policy.prefix_ids.expand(batch, -1), core_ids[:, :-1]], dim=1)
    hidden = decoder(inputs, use_cache=False).last_hidden_state
    start = policy.prefix_ids.shape[1] - 1
    states = hidden[:, start:start + policy.core_length, :]
    require(states.shape[1] == policy.core_length,
            f"Extracted {states.shape[1]} states, expected {policy.core_length}")
    return states


def cached_hidden_states(policy, core_ids):
    """``(B, 10, H)`` states via the shared-prefix path, with a single-use cache.

    The cache is constructed inside this call and dropped on return. Hoisting it to
    score chosen and rejected from one object is the exact bug this design removes.
    """
    import torch
    decoder, _ = backbone_and_head(policy.model)
    batch = core_ids.shape[0]
    prefix = decoder(policy.prefix_ids, use_cache=True)
    first = prefix.last_hidden_state[:, -1:, :].expand(batch, -1, -1)
    cache = prefix.past_key_values
    cache.batch_repeat_interleave(batch)
    rest = decoder(core_ids[:, :-1], past_key_values=cache, use_cache=True)
    return torch.cat([first, rest.last_hidden_state], dim=1)


def head_reconstruction(policy, core_ids, *, atol, rtol, cached=False):
    """Recompute canonical logits from the extracted states and compare to the model.

    This is the proof that the states are the final-normalized, pre-head ones. If a
    future release inserts another normalization between them and the vocabulary
    projection, the reconstruction stops matching and the audit stops.
    """
    import torch
    _, head = backbone_and_head(policy.model)
    # Eval + inference mode, like every other extraction path in this module. No
    # gradient is needed to compare two forward passes, and running the comparison
    # under autograd both builds a graph for nothing and emits a requires_grad
    # warning on the real preflight -- which is a warning about the probe's own
    # setup, not about the model it is supposed to be checking.
    with torch.inference_mode():
        was_training = policy.model.training
        policy.model.eval()
        try:
            states = cached_hidden_states(policy, core_ids) if cached else full_hidden_states(
                policy, core_ids)
            reconstructed = head(states)[..., policy.canonical_ids]
            reference = policy.full_logits(core_ids)
            difference = (reconstructed.double() - reference.double()).abs()
            allowance = float(atol) + float(rtol) * reference.double().abs()
        finally:
            if was_training:
                policy.model.train()
    report = {"rows": int(core_ids.shape[0]), "positions": list(EXTRACTED_POSITIONS),
              "max_abs_error": float(difference.max()),
              "max_allowance": float(allowance.max()),
              "atol": float(atol), "rtol": float(rtol),
              "path": "cached" if cached else "full",
              "mode": "eval_and_inference_mode",
              "requires_grad": bool(reconstructed.requires_grad),
              "within_tolerance": bool(bool((difference <= allowance).all()))}
    require(not report["requires_grad"],
            "The head reconstruction ran with autograd active; the probe is an inference-only "
            "comparison of two forward passes and a graph here means the mode guard did not hold")
    require(torch.isfinite(reconstructed).all(),
            "Reconstructed logits are nonfinite; the extracted states are not usable")
    require(report["within_tolerance"],
            f"Output-head reconstruction failed at atol={atol:.1e}, rtol={rtol:.1e}: max error "
            f"{report['max_abs_error']:.6g}. The extracted states are not the pre-head states "
            "this formula assumes.")
    return report


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def ches_from_states(chosen_states, rejected_states):
    """``dot(s+, s-) - dot(s+, s+)`` in float64 from ``(B, T, H)`` state blocks."""
    plus = np.asarray(chosen_states, dtype=np.float64)
    minus = np.asarray(rejected_states, dtype=np.float64)
    require(plus.ndim == 3 and plus.shape == minus.shape,
            f"CHES needs aligned (B, T, H) blocks, got {plus.shape} and {minus.shape}")
    s_plus = plus.sum(axis=1)
    s_minus = minus.sum(axis=1)
    return (s_plus * s_minus).sum(axis=1) - (s_plus * s_plus).sum(axis=1)


def ches_reference(chosen_states, rejected_states):
    """An independent reference: the explicit triple sum, for tiny tensors in tests.

    Deliberately *not* "sum over t, then dot". That is the same factorization
    :func:`ches_from_states` uses, so agreeing with it would only prove that the
    same algebra was typed twice. This expands the product instead and sums over
    both position indices and the channel::

        CHES = sum_{t, t', h} plus[t,h] * minus[t',h]  -  sum_{t, t', h} plus[t,h] * plus[t',h]

    which is the same quantity by distributivity and a genuinely different
    computation, in Python floats, one term at a time.
    """
    plus = np.asarray(chosen_states, dtype=np.float64)
    minus = np.asarray(rejected_states, dtype=np.float64)
    require(plus.ndim == 3 and plus.shape == minus.shape,
            f"CHES reference needs aligned (B, T, H) blocks, got {plus.shape} and {minus.shape}")
    rows, positions, channels = plus.shape
    out = np.zeros(rows, dtype=np.float64)
    for row in range(rows):
        cross = 0.0
        self_term = 0.0
        for t in range(positions):
            for other in range(positions):
                for h in range(channels):
                    cross += float(plus[row][t][h]) * float(minus[row][other][h])
                    self_term += float(plus[row][t][h]) * float(plus[row][other][h])
        out[row] = cross - self_term
    return out


def ches_batch(policy, chosen_index, rejected_index, *, cached=False):
    """CHES for one batch of pairs, as float64 numpy."""
    import torch
    with torch.inference_mode():
        was_training = policy.model.training
        policy.model.eval()
        try:
            chosen_ids = policy.token_ids(chosen_index)
            rejected_ids = policy.token_ids(rejected_index)
            extract = cached_hidden_states if cached else full_hidden_states
            plus = extract(policy, chosen_ids).float().cpu().numpy()
            minus = extract(policy, rejected_ids).float().cpu().numpy()
        finally:
            if was_training:
                policy.model.train()
    return ches_from_states(plus, minus)


def _require_pairs(policy, chosen_index, rejected_index, *, label):
    chosen = np.asarray(chosen_index)
    rejected = np.asarray(rejected_index)
    require(chosen.shape == rejected.shape and chosen.ndim == 2
            and chosen.shape[1] == policy.core_length,
            f"{label}: chosen/rejected must both be (N, {policy.core_length})")
    return chosen, rejected


def _require_all_finite(vectors, *, label):
    for name, vector in sorted(vectors.items()):
        bad = int((~np.isfinite(vector)).sum())
        require(bad == 0, f"{label}: {bad} nonfinite values in {name}; the artifact fails and no "
                          "row is dropped")


def pair_log_probabilities(policy, chosen_index, rejected_index, *, batch_size=256,
                           label="pair scores"):
    """Chosen and rejected sequence log probabilities for a whole pair population.

    Routed through the audit's strict scorer rather than ``policy.score``: the
    displacement this stage reports is a difference of two log probabilities, and
    a nonfinite logit on an unselected residue would leave both of them finite and
    the difference wrong in a way nothing downstream could detect.
    """
    from .her2_support_scoring import strict_sequence_log_probabilities
    chosen, rejected = _require_pairs(policy, chosen_index, rejected_index, label=label)
    chosen_block = strict_sequence_log_probabilities(
        policy, chosen, batch_size=int(batch_size), label=f"{label}: chosen")
    rejected_block = strict_sequence_log_probabilities(
        policy, rejected, batch_size=int(batch_size), label=f"{label}: rejected")
    chosen_lp = chosen_block["sum_log_probability"]
    rejected_lp = rejected_block["sum_log_probability"]
    _require_all_finite({"chosen_lp": chosen_lp, "rejected_lp": rejected_lp}, label=label)
    return {"chosen_log_probability": np.asarray(chosen_lp, dtype=np.float64),
            "rejected_log_probability": np.asarray(rejected_lp, dtype=np.float64),
            "pairs": int(chosen.shape[0]), "score_batch_size": int(batch_size),
            "logit_checks": {"chosen": chosen_block["checks"],
                             "rejected": rejected_block["checks"]}}


def ches_scores(policy, chosen_index, rejected_index, *, batch_size=64, score_batch_size=256,
                cached=False, progress=None, label="ches"):
    """CHES plus the two sequence log probabilities for a whole pair population.

    The hidden-state extraction runs at ``batch_size`` because it holds ``(B, 10, H)``
    states for both responses; the ordinary scoring pass runs at the larger
    ``score_batch_size`` because it holds logits only. The two are recorded apart so
    a later timing or memory note cannot be attributed to the wrong pass.
    """
    chosen, rejected = _require_pairs(policy, chosen_index, rejected_index, label=label)
    total = chosen.shape[0]
    values = np.empty(total, dtype=np.float64)
    for start in range(0, total, int(batch_size)):
        stop = min(start + int(batch_size), total)
        values[start:stop] = ches_batch(policy, chosen[start:stop], rejected[start:stop],
                                        cached=cached)
        if progress is not None:
            progress.advance(f"{label} {stop}/{total}", completed=stop)
    scores = pair_log_probabilities(policy, chosen, rejected, batch_size=score_batch_size,
                                    label=label)
    _require_all_finite({"ches": values}, label=label)
    return {"ches": values, "pairs": int(total), "path": "cached" if cached else "full",
            "ches_batch_size": int(batch_size), **scores}


# ---------------------------------------------------------------------------
# populations, pair identity and temporal ordering
# ---------------------------------------------------------------------------

def pair_identity(chosen_index, rejected_index, *, population_id, construction):
    """Frozen identity of one ordered pair population.

    Both core orders are hashed separately: a re-sorted pair set is a different
    measurement even when the two multisets agree, because every statistic here is
    a per-row paired quantity.
    """
    from .her2_preferences import core_digest
    chosen = np.asarray(chosen_index)
    rejected = np.asarray(rejected_index)
    require(chosen.shape == rejected.shape, "Pair identity needs aligned chosen/rejected blocks")
    return {"population_id": population_id, "pairs": int(chosen.shape[0]),
            "chosen_core_order_sha256": core_digest(chosen),
            "rejected_core_order_sha256": core_digest(rejected),
            "construction": dict(construction)}


def training_pair_prefix(pairing, *, cycle, count):
    """The first ``count`` ordered pairs of one deterministic training cycle.

    The IDs are fixed *before* scoring and never selected by observed displacement.
    """
    require(int(cycle) >= 0, "Cycle must be >= 0")
    order, partners = pairing.cycle_rows(int(cycle))
    require(count <= order.size,
            f"Cycle {cycle} holds {order.size} pairs, fewer than the declared {count}")
    chosen_rows = order[:int(count)]
    rejected_rows = partners[:int(count)]
    chosen, rejected = pairing.pair_cores(chosen_rows, rejected_rows)
    ids = np.stack([np.full(int(count), int(cycle), dtype=np.int64),
                    chosen_rows.astype(np.int64), rejected_rows.astype(np.int64)], axis=1)
    return {"chosen_index": chosen, "rejected_index": rejected, "pair_ids": ids,
            "cycle": int(cycle), "count": int(count),
            "id_columns": ["cycle", "chosen_row", "rejected_row"],
            "cycle_length": int(order.size)}


def verify_consumption(*, pairs_used, cycle, cycle_length, endpoint_pair_exposures, label):
    """Prove the declared training pairs were actually consumed before every endpoint.

    The population is the first ``pairs_used`` pairs of one cycle. The earliest
    endpoint must therefore have consumed at least that many pairs, and the whole
    population must sit inside the cycle it claims.
    """
    require(int(pairs_used) <= int(cycle_length),
            f"{label}: {pairs_used} pairs do not fit inside cycle {cycle} of length {cycle_length}")
    earliest = min(endpoint_pair_exposures.values()) if endpoint_pair_exposures else None
    require(earliest is not None,
            f"{label}: no endpoint exposure counts were supplied, so consumption is unverified")
    consumed_before_cycle = int(cycle) * int(cycle_length)
    required = consumed_before_cycle + int(pairs_used)
    require(int(earliest) >= required,
            f"{label}: the earliest endpoint consumed {earliest} pairs, fewer than the {required} "
            f"needed to reach pair {pairs_used} of cycle {cycle}. These pairs did not all supply "
            "updates before that endpoint.")
    return {"pairs_used": int(pairs_used), "cycle": int(cycle),
            "cycle_length": int(cycle_length), "required_pair_exposures": required,
            "earliest_endpoint_pair_exposures": int(earliest),
            "endpoint_pair_exposures": {str(k): int(v)
                                        for k, v in sorted(endpoint_pair_exposures.items())},
            "verified": True}


def repeated_core_identities(index):
    """How often each core recurs in an ordered population. Recurrence blocks p-values."""
    from .her2_data import pack_codes
    values = np.asarray(index)
    # A lossless 50-bit packing, so equality of codes IS equality of cores and no
    # structured-array view or collision test is involved.
    _, inverse, counts = np.unique(pack_codes(values), return_inverse=True, return_counts=True)
    per_row = counts[inverse].astype(np.int64).ravel()
    return {"rows": int(values.shape[0]), "distinct_cores": int(counts.size),
            "max_repeats": int(counts.max()), "rows_in_repeated_cores": int((per_row > 1).sum()),
            "per_row_repeat_count": per_row,
            "note": ("cores recur, so the pairs are not independent and no naive independent-pair "
                     "significance claim is attached to any association below")}


# ---------------------------------------------------------------------------
# association analyses
# ---------------------------------------------------------------------------

def average_ranks(values):
    """Average ranks with ties sharing their mean rank (scipy's ``average`` method).

    Written out rather than imported: scipy is not a declared dependency of this
    package, and a rank convention is exactly the kind of thing that must not
    change underneath a published association because an optional library was or
    was not installed. ``mergesort`` is stable, so equal values keep input order
    before their ranks are averaged.
    """
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1, f"Ranks are taken over a 1-D vector, got shape {array.shape}")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def spearman_average_ranks(x, y):
    """Spearman rho on average ranks. Deliberately returns no p-value."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape, f"Spearman needs aligned vectors, got {a.shape}/{b.shape}")
    if a.size < 3:
        return {"spearman": None, "n": int(a.size), "ranks": "average",
                "reason": "fewer than three pairs", "p_value": None,
                "p_value_reason": "cores recur across pairs, so there is no independent-pair null"}
    ra, rb = average_ranks(a), average_ranks(b)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return {"spearman": None, "n": int(a.size), "ranks": "average",
                "reason": "one side is constant under ranking", "p_value": None,
                "p_value_reason": "cores recur across pairs, so there is no independent-pair null"}
    return {"spearman": float(np.corrcoef(ra, rb)[0, 1]), "n": int(a.size), "ranks": "average",
            "reason": None, "p_value": None,
            "p_value_reason": ("cores recur across pairs; an independent-pair p-value would "
                               "overstate the evidence and none is computed")}


def value_bins(values, *, parts, prefix):
    """Tie-preserving equal-probability bins by value thresholds.

    The tie convention is the audit's single declared one
    (:data:`her2_support_scoring.BIN_EDGE_SIDE`): a row equal to an edge belongs to
    the bin above it, so ties are never split and a bin below a tied edge may be
    empty. Deciles here and quartiles there must agree about that, or two tables in
    the same report would bin the same row differently.
    """
    from .her2_support_scoring import BIN_EDGE_SIDE
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, "No values to bin")
    edges = [float(np.quantile(array, k / parts, method="linear")) for k in range(1, parts)]
    assignment = np.searchsorted(np.asarray(edges), array, side=BIN_EDGE_SIDE)
    labels = np.array([f"{prefix}{int(value) + 1}" for value in assignment], dtype=object)
    return labels, {"parts": int(parts), "edges": edges, "convention": "linear",
                    "edge_side": BIN_EDGE_SIDE,
                    "ties": ("equal values share a bin, so bin sizes are unequal and a bin below "
                             "a tied edge can be empty")}


def within_pair_hamming(chosen_index, rejected_index):
    """Chosen-versus-rejected core Hamming distance, 0..10, for the distance-matched pairs.

    This is not the nearest-training distance used to stratify the support audit.
    The two are different quantities and never share a table column or a label.
    """
    chosen = np.asarray(chosen_index)
    rejected = np.asarray(rejected_index)
    require(chosen.shape == rejected.shape, "Hamming needs aligned chosen/rejected blocks")
    return (chosen != rejected).sum(axis=1).astype(np.int64)


def grouped_association(x, y, labels, *, categories=None):
    """Spearman inside each stratum, with real counts and empty bins kept visible."""
    labels = np.asarray(labels, dtype=object)
    wanted = [str(value) for value in (categories if categories is not None
                                       else sorted({str(v) for v in labels.tolist()}))]
    for value in sorted({str(v) for v in labels.tolist()}):
        if value not in wanted:
            wanted.append(value)
    out = {}
    for category in wanted:
        rows = np.flatnonzero(labels.astype(str) == category)
        if rows.size == 0:
            out[category] = {"rows": 0, "spearman": None,
                             "reason": "no pairs fall in this stratum"}
            continue
        block = spearman_average_ranks(np.asarray(x)[rows], np.asarray(y)[rows])
        block["rows"] = int(rows.size)
        block["mean_x"] = float(np.asarray(x, dtype=np.float64)[rows].mean())
        block["mean_y"] = float(np.asarray(y, dtype=np.float64)[rows].mean())
        out[category] = block
    return out


def displacement_analysis(*, parent_ches, displacement, chosen_index, rejected_index,
                          parent_chosen_log_probability, deciles=10, quartiles=4):
    """Relate parent CHES to later chosen-likelihood displacement, within declared strata.

    ``displacement`` is ``log p_parent(chosen) - log p_checkpoint(chosen)``: positive
    means the checkpoint made the preferred example less likely in absolute terms.
    Associations are descriptive; training changed both the representations and the
    likelihoods, so nothing here identifies which examples caused which change.
    """
    ches = np.asarray(parent_ches, dtype=np.float64)
    shift = np.asarray(displacement, dtype=np.float64)
    require(ches.shape == shift.shape, "CHES and displacement must be aligned")
    hamming = within_pair_hamming(chosen_index, rejected_index)
    decile_labels, decile_meta = value_bins(ches, parts=deciles, prefix="d")
    quartile_labels, quartile_meta = value_bins(
        np.asarray(parent_chosen_log_probability, dtype=np.float64), parts=quartiles, prefix="q")
    return {
        "overall": spearman_average_ranks(ches, shift),
        "by_parent_ches_decile": {
            "bins": grouped_association(ches, shift, decile_labels,
                                        categories=[f"d{k}" for k in range(1, deciles + 1)]),
            "binning": decile_meta,
            "mean_displacement": _means(shift, decile_labels,
                                        [f"d{k}" for k in range(1, deciles + 1)])},
        "by_within_pair_hamming": {
            "bins": grouped_association(ches, shift, hamming.astype(object).astype(str),
                                        categories=[str(k) for k in range(0, CORE_LENGTH + 1)]),
            "definition": ("chosen-versus-rejected core Hamming distance from the distance-matched "
                           "pairing; not the nearest-training distance used in the support audit"),
            "mean_displacement": _means(shift, hamming.astype(object).astype(str),
                                        [str(k) for k in range(0, CORE_LENGTH + 1)])},
        "by_parent_chosen_log_probability_quartile": {
            "bins": grouped_association(ches, shift, quartile_labels,
                                        categories=[f"q{k}" for k in range(1, quartiles + 1)]),
            "binning": quartile_meta,
            "mean_displacement": _means(shift, quartile_labels,
                                        [f"q{k}" for k in range(1, quartiles + 1)])},
        "interpretation": ("a positive relationship supports the proposed explanation. A null "
                           "result neither proves displacement is absent nor invalidates the "
                           "direct support audit. No CHES cutoff or pair filtering is applied.")}


def _means(values, labels, categories):
    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=object).astype(str)
    out = {}
    for category in categories:
        rows = np.flatnonzero(labels == str(category))
        out[str(category)] = {"rows": int(rows.size),
                              "mean": float(array[rows].mean()) if rows.size else None,
                              "reason": None if rows.size else "empty stratum"}
    return out


def increment_analysis(early, later, *, label, early_ches=None, early_checkpoint=None,
                       chosen_index=None, rejected_index=None,
                       parent_chosen_log_probability=None, parent_ches=None,
                       deciles=10, quartiles=4):
    """**Early-checkpoint** CHES against the subsequent increment in displacement.

    The spec's temporal claim is that the geometry *at the early checkpoint*
    predicts what happens next (180 -> 360/600/1200/1800), so the CHES used here
    is the early endpoint's own, recorded with that endpoint's identity. Parent
    CHES against total later displacement is a different question and is reported
    beside it, never in place of it: an analysis that substituted the parent's
    CHES here would answer neither.
    """
    early_shift = np.asarray(early, dtype=np.float64)
    later_shift = np.asarray(later, dtype=np.float64)
    require(early_shift.shape == later_shift.shape,
            f"{label}: the two endpoints must cover the same pairs in the same order")
    increment = later_shift - early_shift
    block = {"label": label, "pairs": int(increment.size),
             "mean_increment": float(increment.mean()),
             "standard_error": (float(increment.std(ddof=1) / math.sqrt(increment.size))
                                if increment.size > 1 else None),
             "early_checkpoint": early_checkpoint,
             "increment_definition": "displacement(later) - displacement(earlier), same pair order"}
    if early_ches is None:
        block["early_ches_association"] = None
        block["early_ches_reason"] = ("no CHES was computed at the early checkpoint of this run, "
                                      "so the early-to-later prediction is unavailable and is not "
                                      "substituted by the parent's CHES")
    else:
        block["early_ches_association"] = displacement_analysis(
            parent_ches=early_ches, displacement=increment, chosen_index=chosen_index,
            rejected_index=rejected_index,
            parent_chosen_log_probability=parent_chosen_log_probability,
            deciles=deciles, quartiles=quartiles)
        block["early_ches_summary"] = summarize(np.asarray(early_ches, dtype=np.float64),
                                                label="early checkpoint CHES")
    if parent_ches is not None:
        block["parent_ches_versus_increment"] = spearman_average_ranks(parent_ches, increment)
        block["parent_ches_note"] = ("parent CHES against the later increment is a separate, "
                                     "weaker analysis than the early-checkpoint one above")
    return block


def matched_control_comparison(treated, control, *, label, treated_id, control_id, budget,
                               chosen_index=None, rejected_index=None):
    """Use the matched continued-SFT control's measurements, not just its existence.

    A control that is named but never subtracted is decoration. Both vectors are the
    *same pairs in the same order* at the same budget and the same seed, so the
    per-pair difference is a clean contrast between the two endpoints.

    It is a **descriptive contrast, not a causal estimate**. These historical
    endpoints were matched on GPU seconds, which is what the campaign held fixed --
    not on exposures. A preference objective and continued SFT consume different
    numbers of sequences and pairs in the same wall-clock budget, so the difference
    confounds the objective with how much data each arm saw. Calling it "the
    displacement attributable to the objective" would name a cause this design
    cannot separate, and no exposure-matched control was run later to separate it.
    """
    treated_values = np.asarray(treated, dtype=np.float64)
    control_values = np.asarray(control, dtype=np.float64)
    require(treated_values.shape == control_values.shape,
            f"{label}: the treated and control endpoints must cover the same pairs in the same "
            f"order, got {treated_values.shape} and {control_values.shape}")
    difference = treated_values - control_values
    block = {"label": label, "matched": True, "budget_gpu_seconds": float(budget),
             "treated_checkpoint": treated_id, "control_checkpoint": control_id,
             "pairs": int(difference.size),
             "mean_difference": float(difference.mean()),
             "standard_error": (float(difference.std(ddof=1) / math.sqrt(difference.size))
                                if difference.size > 1 else None),
             "treated_mean": float(treated_values.mean()),
             "control_mean": float(control_values.mean()),
             "definition": ("displacement(treated) - displacement(matched continued-SFT control), "
                            "per pair, same order, same seed, same nominal GPU-second budget"),
             "matched_on": "nominal GPU seconds",
             "not_matched_on": "exposures (sequences and preference pairs consumed)",
             "interpretation": ("a descriptive contrast between two endpoints that were given the "
                                "same compute budget. It is not the effect of the objective: the "
                                "arms differ in how many sequences and pairs they consumed in "
                                "that budget, and no exposure-matched control exists to separate "
                                "the two.")}
    if chosen_index is not None and rejected_index is not None:
        hamming = within_pair_hamming(chosen_index, rejected_index)
        block["by_within_pair_hamming"] = _means(
            difference, hamming.astype(object).astype(str),
            [str(k) for k in range(0, CORE_LENGTH + 1)])
    return block


def unavailable_control(*, label, budget, reason):
    """A control that does not exist. Named, never invented and never called matched."""
    return {"label": label, "matched": False, "budget_gpu_seconds": float(budget),
            "reason": reason,
            "consequence": ("this increment is reported uncontrolled. No control is synthesized "
                            "from a different budget, a different seed or a different arm.")}


def summarize(values, *, label):
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, f"{label}: nothing to summarize")
    return {"label": label, "rows": int(array.size), "mean": float(array.mean()),
            "standard_error": (float(array.std(ddof=1) / math.sqrt(array.size))
                               if array.size > 1 else None),
            "min": float(array.min()), "max": float(array.max()),
            "quantiles": {repr(q): float(np.quantile(array, q, method="linear"))
                          for q in (0.0, 0.1, 0.5, 0.9, 1.0)}}
