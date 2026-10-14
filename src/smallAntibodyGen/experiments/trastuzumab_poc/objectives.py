"""Masking and ranking objectives for the trastuzumab PoC stages.

Exports
-------
draw_mask         -- random mask over an editable union (four equally-likely modes)
whole_core_scores -- per-core mean log probability under an explicit reveal order
ordering_loss     -- mean squared hinge on measured better/worse pairs
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM
from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext
from smallAntibodyGen.models.antibody_antigen.views import (
    Coord, MaskedTargets, VisibleView, validate_coord,
)


@dataclass(frozen=True)
class MaskDraw:
    mode: int             # 0=single  1=multi-small  2=span  3=full
    sites: Tuple[Coord, ...]


def _validate_union(union: object) -> Tuple[Coord, ...]:
    """Validate and return canonical coords; always raises before any RNG call."""
    union = tuple(union)
    if not union:
        raise ValueError("hidden union must be nonempty")
    seen: set = set()
    result: List[Coord] = []
    for c in union:
        coord = validate_coord(c)
        if coord[0] not in ("H", "L"):
            raise ValueError(
                f"union coord {coord!r} has role {coord[0]!r}; only H/L allowed"
            )
        if coord in seen:
            raise ValueError(f"duplicate coord {coord!r} in union")
        seen.add(coord)
        result.append(coord)
    return tuple(result)


def _chain_runs(coords: Tuple[Coord, ...]) -> List[List[int]]:
    """Group by role and return lists of consecutive-index runs as union-index lists."""
    by_role: dict = defaultdict(list)
    for union_pos, (role, res_idx) in enumerate(coords):
        by_role[role].append((res_idx, union_pos))

    runs: List[List[int]] = []
    for role in sorted(by_role):
        sorted_pairs = sorted(by_role[role], key=lambda p: p[0])
        run: List[int] = [sorted_pairs[0][1]]
        for prev, cur in zip(sorted_pairs, sorted_pairs[1:]):
            if cur[0] == prev[0] + 1:
                run.append(cur[1])
            else:
                runs.append(run)
                run = [cur[1]]
        runs.append(run)
    return runs


def draw_mask(union: Sequence[Coord], rng: random.Random) -> MaskDraw:
    """Draw a random mask over *union* using one of four equally-likely modes.

    Validates union (unique, nonempty, H/L only) before consuming *rng*.

    Modes:
        0 -- single site
        1 -- 2 or 3 sites (uniform choice; reduced gracefully for tiny unions)
        2 -- contiguous same-chain span, preferred length 4-8; gracefully shorter
             for small regions; never bridges chain gaps
        3 -- full editable union
    """
    coords = _validate_union(union)
    n = len(coords)

    mode = rng.randrange(4)

    if mode == 0:
        sites: Tuple[Coord, ...] = (rng.choice(coords),)

    elif mode == 1:
        count = min(rng.choice([2, 3]), n)
        sites = tuple(rng.sample(coords, count))

    elif mode == 2:
        runs = _chain_runs(coords)
        # Try preferred span lengths 4-8; fall back to shorter for small regions.
        candidates: List[Tuple[Coord, ...]] = []
        for min_slen in range(4, 0, -1):
            for run in runs:
                rlen = len(run)
                if rlen < min_slen:
                    continue
                max_slen = min(8, rlen)
                for slen in range(min_slen, max_slen + 1):
                    for start in range(rlen - slen + 1):
                        candidates.append(
                            tuple(coords[run[start + j]] for j in range(slen))
                        )
            if candidates:
                break
        sites = rng.choice(candidates) if candidates else (rng.choice(coords),)

    else:  # mode == 3
        sites = coords

    return MaskDraw(mode=mode, sites=sites)


def whole_core_scores(
    policy: FusedDPLM,
    context: StaticContext,
    fully_masked_ab_view: VisibleView,
    targets: Sequence[MaskedTargets],
    order: Sequence[Coord],
    state_microbatch: int = 4,
) -> torch.Tensor:
    """Per-core mean log probability under an explicit reveal order.

    For each core (endpoint in *targets*), teacher-forces the gold predecessors
    along *order* and queries each position in turn.  All K query steps across
    all cores are flattened and dispatched in bounded microbatches via
    ``policy.queried_log_probs``, which checkpoints the entire fused model call.

    Returns a ``[ncores]`` tensor with intact gradients.

    Constraints enforced:
    - *fully_masked_ab_view* masks **exactly** ``context.hidden_union`` (no Ag).
    - *order* is a permutation of ``context.hidden_union``.
    - Every target covers exactly ``context.hidden_union``.
    """
    if not targets:
        raise ValueError("targets must be nonempty")
    if type(state_microbatch) is not int or state_microbatch < 1:
        raise ValueError("state_microbatch must be an integer >= 1")
    if "Ag" in fully_masked_ab_view.roles:
        raise ValueError("Ag must not appear in the antibody query view")

    hidden_union_set = set(context.hidden_union)
    view_hidden = set(fully_masked_ab_view.hidden_sites())
    if view_hidden != hidden_union_set:
        raise ValueError(
            "fully_masked_ab_view must mask exactly context.hidden_union; "
            f"extra masked: {view_hidden - hidden_union_set}, "
            f"not masked: {hidden_union_set - view_hidden}"
        )
    context.check_query(fully_masked_ab_view)

    ordered = tuple(order)
    if len(ordered) != len(hidden_union_set) or set(ordered) != hidden_union_set:
        raise ValueError("order must be a permutation of context.hidden_union")

    for i, tgt in enumerate(targets):
        if set(tgt.coords()) != hidden_union_set:
            raise ValueError(
                f"targets[{i}] coordinates must exactly match context.hidden_union"
            )

    n_cores = len(targets)
    K = len(ordered)

    # Preconstruct all (state, coord, residue) tuples in core-major order.
    all_states: List[VisibleView] = []
    all_coords: List[Coord] = []
    all_residues: List[str] = []

    for tgt in targets:
        state = fully_masked_ab_view
        for coord in ordered:
            all_states.append(state)
            all_coords.append(coord)
            all_residues.append(tgt.residue(coord))
            state = state.reveal(coord, tgt.residue(coord))

    total = n_cores * K
    log_prob_parts: List[torch.Tensor] = []
    for start in range(0, total, state_microbatch):
        end = min(start + state_microbatch, total)
        lp = policy.queried_log_probs(
            all_states[start:end],
            [context] * (end - start),
            all_coords[start:end],
            all_residues[start:end],
        )
        log_prob_parts.append(lp)

    # Shape: [n_cores * K] -> [n_cores, K] -> [n_cores]
    return torch.cat(log_prob_parts).reshape(n_cores, K).mean(dim=1)


def ordering_loss(
    scores: torch.Tensor,
    ordered_pairs: Sequence[Tuple[int, int]],
    margin: float = 0.1,
) -> torch.Tensor:
    """Mean squared hinge over measured better/worse pairs.

    For each (better_index, worse_index): relu(margin - (s_better - s_worse))^2.
    A pair whose gap already exceeds *margin* contributes zero.
    """
    if not isinstance(scores, torch.Tensor) or scores.dim() != 1:
        raise TypeError("scores must be a 1-D Tensor")
    if len(scores) == 0 or not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be nonempty finite floating-point values")
    if not ordered_pairs:
        raise ValueError("ordered_pairs must be nonempty")
    margin_f = float(margin)
    if not (margin_f > 0 and margin_f < float("inf")):
        raise ValueError("margin must be a finite positive number")

    n = len(scores)
    terms: List[torch.Tensor] = []
    for better, worse in ordered_pairs:
        if not (type(better) is int and type(worse) is int):
            raise TypeError(
                f"pair indices must be ints, got "
                f"({type(better).__name__}, {type(worse).__name__})"
            )
        if better == worse:
            raise ValueError(f"self-pair ({better}, {worse}) in ordered_pairs")
        if not (0 <= better < n and 0 <= worse < n):
            raise IndexError(
                f"pair ({better}, {worse}) out of range for scores of length {n}"
            )
        terms.append(
            torch.relu(margin_f - (scores[better] - scores[worse])) ** 2
        )

    return torch.stack(terms).mean()
