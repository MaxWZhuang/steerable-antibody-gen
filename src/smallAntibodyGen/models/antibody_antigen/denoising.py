"""Backend contract, likelihood scorers and iterative masked completion.

A backend maps a :class:`VisibleView` (and nothing else) to a :class:`BackendOutput`:
one row per present residue, antigen included, in ``view.coords()`` order and with no
special tokens. Every scorer here normalises with ``log_softmax`` over the 20 selected
canonical logits, never over the full vocabulary followed by a slice; the probability
the full-vocabulary softmax puts outside those 20 is reported separately as
``canonical_mass``.

Scoring is differentiable end to end (nothing detaches a likelihood before the loss).
:func:`complete` is the opposite: a sampler, run under ``no_grad``, that takes no target
object and so cannot see hidden truth. It is iterative masked completion with whatever
backend it is given; it does not claim the backend was trained as a diffusion model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import torch

from smallAntibodyGen.models.antibody_antigen.views import (
    CANONICAL_RESIDUES,
    RESIDUE_INDEX,
    Coord,
    MaskedTargets,
    VisibleView,
    validate_coord,
)

N_CANONICAL = len(CANONICAL_RESIDUES)
_MASS_SLACK = 1e-4


@dataclass(frozen=True, eq=False)
class BackendOutput:
    coords: Tuple[Coord, ...]  # == view.coords(): all present residues, no specials
    raw_logits: torch.Tensor  # [N, 20], columns follow CANONICAL_RESIDUES
    canonical_mass: torch.Tensor  # [N], full-vocabulary softmax mass inside the 20 columns
    features: torch.Tensor  # [N, D], final residue features
    mask_ratio: float  # mask ratio the backend actually observed


class ResidueBackend(Protocol):
    def forward(self, view: VisibleView) -> BackendOutput: ...


def check_output(view: VisibleView, out: BackendOutput) -> BackendOutput:
    """Raise unless ``out`` is aligned with ``view`` and numerically sane; return it unchanged."""
    coords = view.coords()
    n = len(coords)
    if tuple(out.coords) != coords:
        raise ValueError("backend coordinates do not match the view's present residues")
    if tuple(out.raw_logits.shape) != (n, N_CANONICAL):
        raise ValueError(f"raw_logits shape {tuple(out.raw_logits.shape)} != {(n, N_CANONICAL)}")
    if tuple(out.canonical_mass.shape) != (n,):
        raise ValueError(f"canonical_mass shape {tuple(out.canonical_mass.shape)} != {(n,)}")
    if out.features.dim() != 2 or out.features.shape[0] != n:
        raise ValueError(f"features must be [{n}, D], got {tuple(out.features.shape)}")
    for name, tensor in (
        ("raw_logits", out.raw_logits),
        ("canonical_mass", out.canonical_mass),
        ("features", out.features),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
    if bool(((out.canonical_mass < -_MASS_SLACK) | (out.canonical_mass > 1 + _MASS_SLACK)).any()):
        raise ValueError("canonical_mass must lie in [0, 1]")
    if not isinstance(out.mask_ratio, float) or not 0.0 <= out.mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be a float in [0, 1], got {out.mask_ratio!r}")
    return out


def checked_forward(backend: ResidueBackend, view: VisibleView) -> BackendOutput:
    return check_output(view, backend.forward(view))


def check_order(view: VisibleView, order: Sequence[Coord]) -> Tuple[Coord, ...]:
    """``order`` must be exactly a permutation of the view's hidden sites."""
    ordered = tuple(validate_coord(coord) for coord in order)
    if len(set(ordered)) != len(ordered):
        raise ValueError("order repeats a site")
    if set(ordered) != set(view.hidden_sites()):
        raise ValueError("order must be exactly a permutation of the view's hidden sites")
    return ordered


@dataclass(frozen=True, eq=False)
class MaskedNll:
    total: torch.Tensor  # scalar: differentiable sum of -log p over the scored sites
    site_nll: torch.Tensor  # [count], differentiable
    count: int
    coords: Tuple[Coord, ...]
    canonical_mass: torch.Tensor  # [count], diagnostic (detached)
    mask_ratio: float


def masked_nll(backend: ResidueBackend, view: VisibleView, targets: MaskedTargets) -> MaskedNll:
    """One forward pass of ``view``; scores only the target sites, which must be masked in it."""
    if len(targets) == 0:
        raise ValueError("no target sites to score")
    hidden = set(view.hidden_sites())
    for coord in targets.coords():
        if coord not in hidden:
            raise ValueError(f"target {coord!r} is not masked in the view")
    out = checked_forward(backend, view)
    rows = [view.position(coord) for coord in targets.coords()]
    cols = [RESIDUE_INDEX[targets.residue(coord)] for coord in targets.coords()]
    logp = torch.log_softmax(out.raw_logits[rows], dim=-1)
    site_nll = -logp[torch.arange(len(rows), device=logp.device), torch.tensor(cols, device=logp.device)]
    return MaskedNll(
        total=site_nll.sum(),
        site_nll=site_nll,
        count=len(rows),
        coords=targets.coords(),
        canonical_mass=out.canonical_mass[rows].detach(),
        mask_ratio=out.mask_ratio,
    )


@dataclass(frozen=True, eq=False)
class SequentialLogProb:
    order: Tuple[Coord, ...]
    log_prob: torch.Tensor  # scalar: sum of the per-step log-probabilities (differentiable)
    step_log_probs: torch.Tensor  # [T], differentiable
    step_mass: torch.Tensor  # [T], diagnostic (detached)
    step_mask_ratio: Tuple[float, ...]


def sequential_log_prob(
    backend: ResidueBackend, view: VisibleView, targets: MaskedTargets, order: Sequence[Coord]
) -> SequentialLogProb:
    """Teacher-forced ``sum_t log p(x_t | visible background, x_<t)`` along an explicit order.

    The first forward sees every hidden site masked. After each query the site's own gold
    residue is revealed, and the next step is a full forward of the whole current state.
    """
    ordered = check_order(view, order)
    if not ordered:
        raise ValueError("no hidden sites to score")
    if set(targets.coords()) != set(ordered):
        raise ValueError("targets must hold the gold residue of exactly the hidden sites")
    state = view
    steps: List[torch.Tensor] = []
    mass: List[torch.Tensor] = []
    ratios: List[float] = []
    for coord in ordered:
        out = checked_forward(backend, state)
        row = state.position(coord)
        gold = targets.residue(coord)
        steps.append(torch.log_softmax(out.raw_logits[row], dim=-1)[RESIDUE_INDEX[gold]])
        mass.append(out.canonical_mass[row].detach())
        ratios.append(out.mask_ratio)
        state = state.reveal(coord, gold)
    step_log_probs = torch.stack(steps)
    return SequentialLogProb(
        order=ordered,
        log_prob=step_log_probs.sum(),
        step_log_probs=step_log_probs,
        step_mass=torch.stack(mass),
        step_mask_ratio=tuple(ratios),
    )


@dataclass(frozen=True, eq=False)
class PairGap:
    a: SequentialLogProb
    b: SequentialLogProb
    gap: torch.Tensor  # J(B) - J(A), differentiable


def sequential_gap(
    backend: ResidueBackend,
    view: VisibleView,
    order: Sequence[Coord],
    targets_a: MaskedTargets,
    targets_b: MaskedTargets,
) -> PairGap:
    """``J(B) - J(A)`` with ``J(X) = sum_t log p(x_t | visible background, x_<t)``.

    ``view`` masks exactly the sites where the endpoints differ, and both endpoints are
    scored with the same masks and the same ``order``. Sites the endpoints share belong
    in the visible background, not among the masks. The caller turns the gap into a loss;
    this function knows nothing about measurements.

    The order, both endpoints' coordinate coverage and the unchanged-site rule are all
    checked before either endpoint is scored, so a malformed pair costs no model call.
    """
    ordered = check_order(view, order)
    if not ordered:
        raise ValueError("no hidden sites to score")
    for targets in (targets_a, targets_b):
        if set(targets.coords()) != set(ordered):
            raise ValueError("targets must hold the gold residue of exactly the hidden sites")
    for coord in ordered:
        if targets_a.residue(coord) == targets_b.residue(coord):
            raise ValueError(f"{coord!r} is unchanged between the endpoints and must not be masked")
    a = sequential_log_prob(backend, view, targets_a, ordered)
    b = sequential_log_prob(backend, view, targets_b, ordered)
    return PairGap(a=a, b=b, gap=b.log_prob - a.log_prob)


@dataclass(frozen=True)
class Completion:
    view: VisibleView  # fully revealed
    order: Tuple[Coord, ...]
    residues: Tuple[str, ...]
    step_log_probs: Tuple[float, ...]  # log-probability of each chosen residue at its step


def complete(
    backend: ResidueBackend,
    view: VisibleView,
    *,
    order: Optional[Sequence[Coord]] = None,
    sample: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Completion:
    """Fill every hidden site in exactly ``len(view.hidden_sites())`` steps.

    ``order=None`` is easy-first: each step reveals the hidden site whose most likely
    residue has the highest probability (ties go to the earlier coordinate). Otherwise
    ``order`` is followed as given. A site is filled with its argmax residue, or with a
    ``torch.multinomial`` draw from ``generator`` (a CPU generator, required) when
    ``sample=True``. Known residues are never touched and the full current state is
    re-run after every reveal.
    """
    if sample:
        if generator is None:
            raise ValueError("sample=True needs an explicit torch.Generator")
        if generator.device.type != "cpu":
            raise ValueError("the sampling generator must be a CPU generator")
    pending = list(view.hidden_sites() if order is None else check_order(view, order))
    easy_first = order is None
    state = view
    chosen_coords: List[Coord] = []
    chosen: List[str] = []
    step_log_probs: List[float] = []
    with torch.no_grad():
        while pending:
            out = checked_forward(backend, state)
            logp = torch.log_softmax(out.raw_logits, dim=-1)
            if easy_first:
                confidence = {coord: float(logp[state.position(coord)].max()) for coord in pending}
                coord = max(pending, key=confidence.__getitem__)
            else:
                coord = pending[0]
            row = logp[state.position(coord)]
            if sample:
                column = int(torch.multinomial(row.exp().cpu(), 1, generator=generator))
            else:
                scores = row.tolist()
                column = max(range(N_CANONICAL), key=scores.__getitem__)
            residue = CANONICAL_RESIDUES[column]
            step_log_probs.append(float(row[column]))
            chosen_coords.append(coord)
            chosen.append(residue)
            pending.remove(coord)
            state = state.reveal(coord, residue)
    if state.hidden_sites():
        raise AssertionError("completion left masked sites behind")
    return Completion(
        view=state, order=tuple(chosen_coords), residues=tuple(chosen), step_log_probs=tuple(step_log_probs)
    )
