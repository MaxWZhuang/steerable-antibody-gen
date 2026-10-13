"""Tiny software-only backends and cases for the antibody-antigen model tests.

Everything here honours the :class:`ResidueBackend` contract but is a toy: none of it
is, or stands in for, a native model, and no parity claim may rest on it.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from smallAntibodyGen.models.antibody_antigen.denoising import N_CANONICAL, BackendOutput
from smallAntibodyGen.models.antibody_antigen.views import (
    RESIDUE_INDEX,
    ROLES,
    Coord,
    MaskedTargets,
    VisibleView,
    mask_complete,
)

SEQUENCES = {"H": "EVQLVESG", "L": "DIQMTQ", "Ag": "KVFGRCEL"}
OTHER_COLUMNS = 4  # logit columns beyond the 20 canonical ones, so canonical_mass < 1


def make_case(masked: Sequence[Coord], roles: Sequence[str] = ROLES) -> Tuple[VisibleView, MaskedTargets]:
    return mask_complete({role: SEQUENCES[role] for role in roles}, masked)


def targets_with(view: VisibleView, residues: Sequence[str]) -> MaskedTargets:
    """Gold residues for ``view``'s hidden sites, in canonical order."""
    return MaskedTargets.from_mapping(dict(zip(view.hidden_sites(), residues)))


class ToyBackend(torch.nn.Module):
    """Every row sees its own token, role and position plus the mean of all present rows.

    A reveal therefore changes every row's logits (a joint stream), and all parameters
    receive gradient from every query.
    """

    def __init__(self, seed: int = 0, dim: int = 8) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.token = torch.nn.Embedding(N_CANONICAL + 1, dim)  # last row = masked
        self.role = torch.nn.Embedding(len(ROLES), dim)
        self.position = torch.nn.Parameter(0.1 * torch.randn(dim))
        self.mix = torch.nn.Linear(2 * dim, dim)
        self.head = torch.nn.Linear(dim, N_CANONICAL + OTHER_COLUMNS)

    def forward(self, view: VisibleView) -> BackendOutput:
        coords = view.coords()
        ids = torch.tensor([N_CANONICAL if view.token(c) is None else RESIDUE_INDEX[view.token(c)] for c in coords])
        roles = torch.tensor([ROLES.index(c[0]) for c in coords])
        offsets = torch.tensor([float(c[1]) for c in coords])
        own = self.token(ids) + self.role(roles) + offsets[:, None] * self.position
        context = own.mean(dim=0, keepdim=True).expand_as(own)
        hidden = torch.tanh(self.mix(torch.cat([own, context], dim=-1)))
        full = self.head(hidden)
        return BackendOutput(
            coords=coords,
            raw_logits=full[:, :N_CANONICAL],
            canonical_mass=torch.softmax(full, dim=-1)[:, :N_CANONICAL].sum(dim=-1),
            features=hidden,
            mask_ratio=len(view.hidden_sites()) / len(coords),
        )


class FixedLogitsBackend:
    """Logits fixed per coordinate (flat elsewhere), whatever the view contains."""

    def __init__(self, table: Optional[Mapping[Coord, Sequence[float]]] = None, mass: float = 0.9) -> None:
        self.table: Dict[Coord, torch.Tensor] = {
            coord: torch.tensor(list(row), dtype=torch.float32) for coord, row in (table or {}).items()
        }
        self.mass = mass

    def forward(self, view: VisibleView) -> BackendOutput:
        coords = view.coords()
        rows = [self.table.get(c, torch.zeros(N_CANONICAL)) for c in coords]
        return BackendOutput(
            coords=coords,
            raw_logits=torch.stack(rows),
            canonical_mass=torch.full((len(coords),), self.mass),
            features=torch.zeros(len(coords), 1),
            mask_ratio=len(view.hidden_sites()) / len(coords),
        )


class StateDependentBackend:
    """Every row's logits are ``n_hidden * WEIGHTS``: only the number of masks matters."""

    WEIGHTS = [0.05 * i for i in range(N_CANONICAL)]
    MASS = 0.8

    def forward(self, view: VisibleView) -> BackendOutput:
        coords = view.coords()
        row = len(view.hidden_sites()) * torch.tensor(self.WEIGHTS)
        return BackendOutput(
            coords=coords,
            raw_logits=row.expand(len(coords), N_CANONICAL).clone(),
            canonical_mass=torch.full((len(coords),), self.MASS),
            features=torch.zeros(len(coords), 1),
            mask_ratio=len(view.hidden_sites()) / len(coords),
        )


class RecordingBackend:
    """Spy: remembers every view it was asked about, then defers to ``inner``."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.views: List[VisibleView] = []

    def forward(self, view: VisibleView) -> BackendOutput:
        self.views.append(view)
        return self.inner.forward(view)
