"""Immutable views of a partially masked antibody/antigen complex (pure stdlib).

A :class:`VisibleView` holds exactly what a backend may see: for each present role
(``H``, ``L``, ``Ag``, always in that order, any non-empty subset) a tuple of visible
residues, with ``None`` at a masked heavy/light site. It carries no full sequence, no
hidden identity and no record/assay/label field. The gold residues of masked sites live
in a separate :class:`MaskedTargets`, which is never forwarded to a backend.

Coordinates are exact ``(role, index)`` pairs with a 0-based ``index`` into that role's
chain. The antigen is never masked in this increment.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

ROLES: Tuple[str, ...] = ("H", "L", "Ag")
CANONICAL_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_INDEX: Mapping[str, int] = {aa: i for i, aa in enumerate(CANONICAL_RESIDUES)}

Coord = Tuple[str, int]
Token = Optional[str]


def validate_residue(residue: object) -> str:
    if not isinstance(residue, str) or residue not in RESIDUE_INDEX:
        raise ValueError(f"not a canonical uppercase residue: {residue!r}")
    return residue


def validate_coord(coord: object) -> Coord:
    if not isinstance(coord, tuple) or len(coord) != 2:
        raise ValueError(f"a coordinate is a (role, index) tuple, got {coord!r}")
    role, index = coord
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError(f"unknown role in coordinate {coord!r}; roles are {ROLES}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"coordinate index must be a non-negative int, got {coord!r}")
    return (role, index)


def _coord_key(coord: Coord) -> Tuple[int, int]:
    return (ROLES.index(coord[0]), coord[1])


@dataclass(frozen=True)
class VisibleView:
    roles: Tuple[str, ...]
    chains: Tuple[Tuple[Token, ...], ...]

    def __post_init__(self) -> None:
        if type(self.roles) is not tuple or type(self.chains) is not tuple:
            raise TypeError("roles and chains must be tuples")
        if not self.roles:
            raise ValueError("a view needs at least one role")
        if len(self.roles) != len(self.chains):
            raise ValueError("roles and chains must have the same length")
        ranks = []
        for role in self.roles:
            if not isinstance(role, str) or role not in ROLES:
                raise ValueError(f"unknown role {role!r}; roles are {ROLES}")
            ranks.append(ROLES.index(role))
        if any(a >= b for a, b in zip(ranks, ranks[1:])):
            raise ValueError(f"roles must be unique and in canonical order {ROLES}, got {self.roles}")
        for role, chain in zip(self.roles, self.chains):
            if type(chain) is not tuple:
                raise TypeError(f"chain {role} must be a tuple of residue tokens, got {type(chain).__name__}")
            if not chain:
                raise ValueError(f"chain {role} is empty")
            for token in chain:
                if token is None:
                    if role == "Ag":
                        raise ValueError("the antigen is never masked in this increment")
                else:
                    validate_residue(token)

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def chain(self, role: str) -> Tuple[Token, ...]:
        if role not in self.roles:
            raise KeyError(f"role {role!r} is not present in this view {self.roles}")
        return self.chains[self.roles.index(role)]

    def coords(self) -> Tuple[Coord, ...]:
        """Every present residue (antigen included, no specials) in canonical order."""
        return tuple((role, i) for role, chain in zip(self.roles, self.chains) for i in range(len(chain)))

    def position(self, coord: Coord) -> int:
        """Row of ``coord`` within :meth:`coords`."""
        role, index = validate_coord(coord)
        if role not in self.roles:
            raise KeyError(f"role {role!r} is not present in this view {self.roles}")
        if index >= len(self.chain(role)):
            raise IndexError(f"{coord!r} is outside chain {role} of length {len(self.chain(role))}")
        offset = 0
        for present, chain in zip(self.roles, self.chains):
            if present == role:
                return offset + index
            offset += len(chain)
        raise AssertionError("unreachable")

    def token(self, coord: Coord) -> Token:
        self.position(coord)
        return self.chain(coord[0])[coord[1]]

    def hidden_sites(self) -> Tuple[Coord, ...]:
        return tuple(
            (role, i) for role, chain in zip(self.roles, self.chains) for i, token in enumerate(chain) if token is None
        )

    def select_roles(self, roles: Sequence[str]) -> "VisibleView":
        if isinstance(roles, str):
            raise TypeError("pass a sequence of role names, not a string")
        wanted = tuple(roles)
        if not wanted:
            raise ValueError("select at least one role")
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"duplicate roles in {wanted}")
        missing = [role for role in wanted if role not in self.roles]
        if missing:
            raise KeyError(f"roles {missing} are not present in this view {self.roles}")
        keep = tuple(role for role in self.roles if role in wanted)
        return VisibleView(keep, tuple(self.chain(role) for role in keep))

    def without_ag(self) -> "VisibleView":
        return self.select_roles([role for role in self.roles if role != "Ag"])

    def ag_only(self) -> "VisibleView":
        return self.select_roles(["Ag"])

    def reveal(self, coord: Coord, residue: str) -> "VisibleView":
        """A new view with one masked site filled in; this view is left untouched."""
        residue = validate_residue(residue)
        if self.token(coord) is not None:
            raise ValueError(f"{coord!r} is already visible; a known residue is never overwritten")
        role, index = coord
        chains = tuple(
            chain[:index] + (residue,) + chain[index + 1:] if present == role else chain
            for present, chain in zip(self.roles, self.chains)
        )
        return VisibleView(self.roles, chains)

    def content_digest(self) -> str:
        """Deterministic digest of the visible content only (roles, lengths, residues, mask pattern)."""
        parts = [
            f"{role}:{len(chain)}:{''.join('-' if token is None else token for token in chain)}"
            for role, chain in zip(self.roles, self.chains)
        ]
        return hashlib.sha256(("view1|" + "|".join(parts)).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class MaskedTargets:
    """Gold residues of masked sites, in canonical coordinate order. Never given to a backend."""

    items: Tuple[Tuple[Coord, str], ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise TypeError("items must be a tuple of (coord, residue) pairs")
        keys = []
        for pair in self.items:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"expected a (coord, residue) pair, got {pair!r}")
            coord, residue = pair
            keys.append(_coord_key(validate_coord(coord)))
            validate_residue(residue)
        if any(a >= b for a, b in zip(keys, keys[1:])):
            raise ValueError("target coordinates must be unique and in canonical order")

    @classmethod
    def from_mapping(cls, mapping: Mapping[Coord, str]) -> "MaskedTargets":
        coords = [validate_coord(coord) for coord in mapping]
        return cls(tuple((coord, validate_residue(mapping[coord])) for coord in sorted(coords, key=_coord_key)))

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, coord: object) -> bool:
        return any(coord == item[0] for item in self.items)

    def coords(self) -> Tuple[Coord, ...]:
        return tuple(coord for coord, _ in self.items)

    def residue(self, coord: Coord) -> str:
        for item_coord, residue in self.items:
            if item_coord == coord:
                return residue
        raise KeyError(f"no target for {coord!r}")

    def as_dict(self) -> Dict[Coord, str]:
        return dict(self.items)


def mask_complete(
    sequences: Mapping[str, str], masked: Iterable[Coord] = ()
) -> Tuple[VisibleView, MaskedTargets]:
    """Mask a complete input: returns the visible view and, separately, the gold residues."""
    if not sequences:
        raise ValueError("no sequences given")
    for role, sequence in sequences.items():
        if not isinstance(role, str) or role not in ROLES:
            raise ValueError(f"unknown role {role!r}; roles are {ROLES}")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"sequence for {role} must be a non-empty string")
        for residue in sequence:
            validate_residue(residue)
    hidden: Dict[Coord, str] = {}
    for coord in masked:
        role, index = validate_coord(coord)
        if role == "Ag":
            raise ValueError("the antigen is never masked in this increment")
        if role not in sequences:
            raise ValueError(f"cannot mask {coord!r}: role {role} is absent")
        if index >= len(sequences[role]):
            raise ValueError(f"cannot mask {coord!r}: chain {role} has length {len(sequences[role])}")
        if (role, index) in hidden:
            raise ValueError(f"duplicate masked site {coord!r}")
        hidden[(role, index)] = sequences[role][index]
    roles = tuple(role for role in ROLES if role in sequences)
    chains = tuple(
        tuple(None if (role, i) in hidden else residue for i, residue in enumerate(sequences[role])) for role in roles
    )
    return VisibleView(roles, chains), MaskedTargets.from_mapping(hidden)
