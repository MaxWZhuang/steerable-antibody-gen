"""Masking and ranking objectives for the trastuzumab PoC stages.

Exports
-------
draw_mask            -- legacy random mask (four equally-likely modes)
UniformMaskSpec      -- config for the clean uniform-K sampler
UniformMaskDraw      -- result of draw_mask_uniform
draw_mask_uniform    -- uniform-K mask sampler matching the mean-site NLL objective
whole_core_scores    -- per-core mean log probability (static or dynamic context)
ordering_loss        -- mean squared hinge on measured better/worse pairs
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
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


# ---------------------------------------------------------------------------
# Uniform-K masking (clean sampler matching the mean-site NLL objective)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniformMaskSpec:
    """Configuration for the clean uniform-K sampler.

    Base component (mixture weight = 1 - span_weight - full_region_weight):
        K ~ Uniform{1, ..., N} where N = len(union);
        sites ~ Uniform over all C(N, K) K-subsets of union.

    The base component estimates mean sequential reconstruction NLL averaged over
    uniformly random reveal orders: each remaining-set size K occurs once per
    order, its subset is uniform, and its next revealed site is uniform within K.
    Conditionals change with the visible subset. This is not the likelihood of
    the iterative generation sampler.

    Optional auxiliary components have declared mixture weights and declared region coords.
    When auxiliary components are active the overall draw is a MIXTURE, not the same
    unbiased estimator as the base alone; the mean-NLL objective changes accordingly.
    Span draws a contiguous same-chain run within ``span_region``; full_region masks all
    of ``full_region_coords``.

    Region coordinates are validated against the union at draw time in
    :func:`draw_mask_uniform`, but structural checks (nonempty, H/L only, no duplicates
    within the region) run on every call regardless of whether the component weight is
    zero, so malformed input cannot hide behind a disabled weight.
    """
    span_region: Optional[Tuple[Coord, ...]] = None
    span_weight: float = 0.0
    full_region_coords: Optional[Tuple[Coord, ...]] = None
    full_region_weight: float = 0.0

    def __post_init__(self):
        for attr, val in (("span_weight", self.span_weight),
                          ("full_region_weight", self.full_region_weight)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"{attr} must be a numeric float, got {type(val).__name__}"
                )
            # val != val catches NaN; explicit inf check catches ±inf.
            if val != val or val == float("inf") or val == float("-inf"):
                raise ValueError(f"{attr} must be finite")
        if self.span_weight < 0.0 or self.span_weight > 1.0:
            raise ValueError("span_weight must be in [0, 1]")
        if self.full_region_weight < 0.0 or self.full_region_weight > 1.0:
            raise ValueError("full_region_weight must be in [0, 1]")
        if self.span_weight + self.full_region_weight > 1.0:
            raise ValueError("span_weight + full_region_weight must be <= 1.0")
        if self.span_weight > 0.0 and not self.span_region:
            raise ValueError("span_weight > 0 requires span_region")
        if self.full_region_weight > 0.0 and not self.full_region_coords:
            raise ValueError("full_region_weight > 0 requires full_region_coords")


@dataclass(frozen=True)
class UniformMaskDraw:
    kind: str              # "uniform_k" | "span" | "full_region"
    sites: Tuple[Coord, ...]
    count: int             # == len(sites)
    eligible_count: int    # union size for uniform_k; region size for span/full_region


def _validate_region(region: object, name: str, union_set: set) -> Tuple[Coord, ...]:
    """Validate and return a region tuple; all coords must be H/L, in union, and non-duplicate."""
    coords = tuple(region)  # type: ignore[arg-type]
    if not coords:
        raise ValueError(f"{name} must be nonempty")
    seen: set = set()
    result: List[Coord] = []
    for c in coords:
        coord = validate_coord(c)
        if coord[0] not in ("H", "L"):
            raise ValueError(f"{name} coord {coord!r} has role {coord[0]!r}; only H/L allowed")
        if coord in seen:
            raise ValueError(f"duplicate coord {coord!r} in {name}")
        seen.add(coord)
        if coord not in union_set:
            raise ValueError(f"{name} coord {coord!r} is not in the editable union")
        result.append(coord)
    return tuple(result)


def draw_mask_uniform(
    union: Sequence[Coord],
    rng: random.Random,
    spec: Optional[UniformMaskSpec] = None,
) -> UniformMaskDraw:
    """Draw a mask under the uniform-K objective, with optional auxiliary components.

    Validates all inputs (including spec region bounds) before consuming *rng*.

    Base component:
        K ~ Uniform{1, ..., N}; sites ~ Uniform K-subset of union.

    With auxiliary components, a single ``rng.random()`` draw selects the component:
        [0, span_weight)         -> span draw
        [span_weight, span+full) -> full_region draw
        otherwise                -> base uniform_k draw

    Span: one contiguous same-chain run drawn from ``spec.span_region`` (no gap bridging).
    Full region: all coords in ``spec.full_region_coords``.
    """
    coords = _validate_union(union)
    union_set = set(coords)
    n = len(coords)
    spec = spec or UniformMaskSpec()

    # Validate ALL provided regions before any RNG, even when their weight is zero,
    # so malformed auxiliary input cannot hide behind a disabled component.
    span_region: Optional[Tuple[Coord, ...]] = None
    full_region: Optional[Tuple[Coord, ...]] = None
    if spec.span_region is not None:
        span_region = _validate_region(spec.span_region, "span_region", union_set)
    if spec.full_region_coords is not None:
        full_region = _validate_region(spec.full_region_coords, "full_region_coords", union_set)

    # Select component.
    threshold = rng.random()
    if span_region is not None and threshold < spec.span_weight:
        # Span draw within the declared region (no chain-gap bridging).
        span_runs = _chain_runs(span_region)
        candidates: List[Tuple[Coord, ...]] = []
        for run in span_runs:
            for slen in range(1, len(run) + 1):
                for start in range(len(run) - slen + 1):
                    candidates.append(tuple(span_region[run[start + j]] for j in range(slen)))
        sites = rng.choice(candidates)
        return UniformMaskDraw(kind="span", sites=sites, count=len(sites),
                               eligible_count=len(span_region))
    if full_region is not None and threshold < spec.span_weight + spec.full_region_weight:
        return UniformMaskDraw(kind="full_region", sites=full_region, count=len(full_region),
                               eligible_count=len(full_region))

    # Base uniform-K draw.
    k = rng.randint(1, n)
    sites = tuple(rng.sample(coords, k))
    return UniformMaskDraw(kind="uniform_k", sites=sites, count=k, eligible_count=n)


def whole_core_scores(
    policy: FusedDPLM,
    context: Union[StaticContext, DynamicContextProvider],
    fully_masked_ab_view: VisibleView,
    targets: Sequence[MaskedTargets],
    order: Sequence[Coord],
    state_microbatch: int = 4,
) -> torch.Tensor:
    """Per-core mean log probability under an explicit reveal order.

    *context* is either a :class:`StaticContext` (one set of pre-extracted features shared
    across all states) or a :class:`DynamicContextProvider` (frozen encoder re-run on the
    current view at every teacher-forcing step).

    Returns a ``[ncores]`` tensor with intact gradients.

    Constraints enforced (all checked before any forward):
    - *fully_masked_ab_view* masks **exactly** the hidden union (no Ag).
    - *order* is a permutation of the hidden union.
    - Every target covers exactly the hidden union.
    - For a dynamic provider, its encoder_identity must match the policy's.
    """
    if not targets:
        raise ValueError("targets must be nonempty")
    if type(state_microbatch) is not int or state_microbatch < 1:
        raise ValueError("state_microbatch must be an integer >= 1")
    if "Ag" in fully_masked_ab_view.roles:
        raise ValueError("Ag must not appear in the antibody query view")

    is_dynamic = isinstance(context, DynamicContextProvider)
    if is_dynamic:
        provider = context
        if provider.encoder_identity != policy.encoder_identity:
            raise ValueError(
                "DynamicContextProvider encoder_identity differs from the policy's encoder_identity"
            )
        hidden_union = context.hidden_union
        hidden_union_set = set(hidden_union)
        if not hidden_union_set:
            raise ValueError("fully_masked_ab_view must have at least one masked position")
        if set(fully_masked_ab_view.hidden_sites()) != hidden_union_set:
            raise ValueError("fully_masked_ab_view must mask exactly provider.hidden_union")
        provider.check_query(fully_masked_ab_view)
    else:
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
        raise ValueError("order must be a permutation of the hidden union")

    for i, tgt in enumerate(targets):
        if set(tgt.coords()) != hidden_union_set:
            raise ValueError(
                f"targets[{i}] coordinates must exactly match the hidden union"
            )

    policy.preflight([fully_masked_ab_view], [context])

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
        batch_contexts = [context] * (end - start)
        lp = policy.queried_log_probs(
            all_states[start:end],
            batch_contexts,
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
