"""Nested HER2 endpoint budget selection and restricted post-measurement data.

The selection of post_pool endpoints is outcome-free (SHA256 over budget_seed +
endpoint identity only).  Measured outcome values are only accepted AFTER a budget
is fixed and ONLY for the selected endpoint IDs.

No Torch dependency.  No numeric affinity comparison across assays.
No performance-based expansion.  No all-pairs quadratic enumeration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, asdict, dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    _content_key as _seq_content_key,
)
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES

_CANON_SET: FrozenSet[str] = frozenset(CANONICAL_RESIDUES)


# ---------------------------------------------------------------------------
# Endpoint manifest (outcome-free)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointConstruct:
    """One fixed-scaffold construct: heavy prefix/suffix + canonical 10-residue core,
    fixed light chain, and actual antigen string.

    heavy_prefix: residues before the 10-residue core
    core: exactly 10 canonical residues
    heavy_suffix: residues after the core
    light: fixed light chain
    Ag: antigen amino-acid string
    """
    heavy_prefix: str
    core: str
    heavy_suffix: str
    light: str
    Ag: str

    def __post_init__(self) -> None:
        for name in ("heavy_prefix", "core", "heavy_suffix", "light", "Ag"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"EndpointConstruct.{name} must be a string")
        if len(self.core) != 10:
            raise ValueError(
                f"core must be exactly 10 residues, got {len(self.core)}"
            )
        for name, seq in (("heavy_prefix", self.heavy_prefix),
                          ("core", self.core),
                          ("heavy_suffix", self.heavy_suffix),
                          ("light", self.light),
                          ("Ag", self.Ag)):
            bad = [c for c in seq if c not in _CANON_SET]
            if bad:
                raise ValueError(
                    f"EndpointConstruct.{name} contains non-canonical residues: {bad!r}"
                )
        if not self.Ag:
            raise ValueError("EndpointConstruct.Ag must be nonempty")
        if not self.light:
            raise ValueError("EndpointConstruct.light must be nonempty")

    def full_heavy(self) -> str:
        return self.heavy_prefix + self.core + self.heavy_suffix

    def content_key(self) -> str:
        return _seq_content_key(self.full_heavy(), self.light, self.Ag)


ENDPOINT_ROLES = ("post_pool", "development", "reserved")


@dataclass(frozen=True)
class EndpointEntry:
    """One endpoint in the manifest.

    endpoint_id: stable unique string
    construct: the EndpointConstruct
    role: "post_pool" | "development" | "reserved"
    neighbourhood_group: caller-declared neighbourhood group id (nonempty)
    assay_id: which assay this endpoint belongs to
    endpoint_semantics: human-readable description of what is measured
    """
    endpoint_id: str
    construct: EndpointConstruct
    role: str
    neighbourhood_group: str
    assay_id: str
    endpoint_semantics: str

    def __post_init__(self) -> None:
        for attr in ("endpoint_id", "neighbourhood_group", "assay_id", "endpoint_semantics"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val:
                raise ValueError(f"EndpointEntry.{attr} must be a nonempty string")
        if self.role not in ENDPOINT_ROLES:
            raise ValueError(
                f"EndpointEntry.role must be one of {ENDPOINT_ROLES}, got {self.role!r}"
            )


@dataclass(frozen=True)
class EndpointManifest:
    """Complete outcome-free manifest for the one-scaffold HER2 experiment.

    neighbourhood_groups must never cross roles: each group_id appears in
    exactly one role.  Exact duplicate constructs across roles or IDs are rejected.
    """
    entries: Tuple[EndpointEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("EndpointManifest.entries must be nonempty")

        scaffolds = {(e.construct.heavy_prefix, e.construct.heavy_suffix,
                      e.construct.light, e.construct.Ag) for e in self.entries}
        if len(scaffolds) != 1:
            raise ValueError("EndpointManifest requires one fixed scaffold and antigen")
        semantics = {}
        for entry in self.entries:
            if entry.assay_id in semantics and semantics[entry.assay_id] != entry.endpoint_semantics:
                raise ValueError("inconsistent endpoint semantics within assay")
            semantics[entry.assay_id] = entry.endpoint_semantics
        ids = [e.endpoint_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate endpoint_id in EndpointManifest.entries")

        # Neighbourhood groups must not cross roles
        group_roles: Dict[str, str] = {}
        for e in self.entries:
            if e.neighbourhood_group in group_roles:
                if group_roles[e.neighbourhood_group] != e.role:
                    raise ValueError(
                        f"neighbourhood_group {e.neighbourhood_group!r} appears in "
                        f"roles {group_roles[e.neighbourhood_group]!r} and {e.role!r}; "
                        f"neighbourhood groups must not cross roles"
                    )
            else:
                group_roles[e.neighbourhood_group] = e.role

        # Exact duplicate constructs across different roles/IDs are rejected
        seen_content: Dict[str, str] = {}  # content_key -> endpoint_id
        for e in self.entries:
            ck = e.construct.content_key()
            if ck in seen_content:
                existing_id = seen_content[ck]
                existing_role = next(x.role for x in self.entries if x.endpoint_id == existing_id)
                if existing_id != e.endpoint_id:
                    raise ValueError(
                        f"Duplicate construct content in endpoints {existing_id!r} "
                        f"(role={existing_role!r}) and {e.endpoint_id!r} (role={e.role!r})"
                    )
            else:
                seen_content[ck] = e.endpoint_id

    def post_pool_entries(self) -> Tuple[EndpointEntry, ...]:
        return tuple(e for e in self.entries if e.role == "post_pool")

    def development_entries(self) -> Tuple[EndpointEntry, ...]:
        return tuple(e for e in self.entries if e.role == "development")

    def reserved_entries(self) -> Tuple[EndpointEntry, ...]:
        return tuple(e for e in self.entries if e.role == "reserved")

    def by_id(self, endpoint_id: str) -> EndpointEntry:
        for e in self.entries:
            if e.endpoint_id == endpoint_id:
                return e
        raise KeyError(f"endpoint_id {endpoint_id!r} not found in manifest")

    def manifest_digest(self) -> str:
        return hashlib.sha256(json.dumps(
            [asdict(e) for e in sorted(self.entries, key=lambda e: e.endpoint_id)],
            sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Budget selection (outcome-free)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectedBudget:
    """Fixed selection of post_pool endpoints.

    Selection is deterministic from budget_seed + endpoint identity only.
    No outcome values are present here.
    """
    budget_n: int
    budget_seed: str           # opaque string seed (nonempty)
    selected_ids: Tuple[str, ...]  # sorted by selection SHA256, then endpoint_id for ties
    selection_digest: str      # binds seed, selected IDs and exact construct manifest
    group_manifest: str
    manifest_digest: str


def select_budget(
    manifest: EndpointManifest,
    budget_n: int,
    budget_seed: str,
) -> SelectedBudget:
    """Select N post_pool endpoints outcome-free.

    Selection is a prefix of post_pool endpoints ordered by
    SHA256(budget_seed || endpoint_id), ties broken by endpoint_id lexicographic order.

    Parameters
    ----------
    manifest : EndpointManifest
    budget_n : int — number of unique measured endpoints to select
    budget_seed : str — opaque non-empty seed string (shared across model arms and training seeds)

    Raises
    ------
    ValueError if budget_n > number of post_pool entries or budget_n < 0.
    """
    if not isinstance(budget_seed, str) or not budget_seed:
        raise ValueError("budget_seed must be a nonempty string")
    if isinstance(budget_n, bool) or not isinstance(budget_n, int):
        raise TypeError("budget_n must be an int")

    post_pool = manifest.post_pool_entries()
    if budget_n < 0:
        raise ValueError(f"budget_n must be >= 0, got {budget_n}")
    if budget_n > len(post_pool):
        raise ValueError(
            f"budget_n={budget_n} exceeds post_pool size {len(post_pool)}"
        )

    # Sort by SHA256(budget_seed || endpoint_id), tie-break by endpoint_id
    def sort_key(e: EndpointEntry) -> Tuple[str, str]:
        h = hashlib.sha256(f"{budget_seed}\x00{e.endpoint_id}".encode("utf-8")).hexdigest()
        return (h, e.endpoint_id)

    ordered = sorted(post_pool, key=sort_key)
    selected = tuple(e.endpoint_id for e in ordered[:budget_n])

    selection_digest = hashlib.sha256(
        json.dumps({"budget_seed": budget_seed, "selected": sorted(selected),
                    "manifest_digest": manifest.manifest_digest()},
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    # Group manifest for selected
    group_map: Dict[str, List[str]] = {}
    for eid in selected:
        entry = manifest.by_id(eid)
        group_map.setdefault(entry.neighbourhood_group, []).append(eid)
    for g in group_map:
        group_map[g].sort()
    group_manifest = json.dumps(
        {k: group_map[k] for k in sorted(group_map.keys())},
        sort_keys=True,
    )

    return SelectedBudget(
        budget_n=budget_n,
        budget_seed=budget_seed,
        selected_ids=selected,
        selection_digest=selection_digest,
        group_manifest=group_manifest,
        manifest_digest=manifest.manifest_digest(),
    )


def extend_budget(
    manifest: EndpointManifest,
    existing: SelectedBudget,
    new_budget_n: int,
) -> SelectedBudget:
    """Extend an existing budget to N entries using the same seed and ordering."""
    if existing != select_budget(manifest, existing.budget_n, existing.budget_seed):
        raise ValueError("existing budget does not match manifest")
    if new_budget_n < existing.budget_n:
        raise ValueError(
            f"new_budget_n={new_budget_n} < existing {existing.budget_n}; use select_budget for reduction"
        )
    return select_budget(manifest, new_budget_n, existing.budget_seed)


# ---------------------------------------------------------------------------
# Measured post data (post budget selection)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasuredEndpoint:
    """One measured outcome value for a selected endpoint.

    endpoint_id must be in the SelectedBudget.
    value is a finite float (no bool, no NaN, no inf).
    orientation: "higher_better" | "lower_better" — how values are oriented.
    assay_id: must match the endpoint's declared assay_id.
    """
    endpoint_id: str
    value: float
    orientation: str    # "higher_better" | "lower_better"
    assay_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id:
            raise ValueError("MeasuredEndpoint.endpoint_id must be nonempty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("MeasuredEndpoint.value must be a numeric float")
        v = float(self.value)
        if v != v or v == float("inf") or v == float("-inf"):
            raise ValueError("MeasuredEndpoint.value must be finite")
        if self.orientation not in ("higher_better", "lower_better"):
            raise ValueError(
                f"MeasuredEndpoint.orientation must be 'higher_better' or 'lower_better', "
                f"got {self.orientation!r}"
            )
        if not isinstance(self.assay_id, str) or not self.assay_id:
            raise ValueError("MeasuredEndpoint.assay_id must be nonempty")


@dataclass(frozen=True)
class DevelopmentLabel:
    """A development-selection label for one development endpoint.

    These are counted SEPARATELY from fitting endpoint labels.
    """
    endpoint_id: str
    value: float
    orientation: str
    assay_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id:
            raise ValueError("DevelopmentLabel.endpoint_id must be nonempty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("DevelopmentLabel.value must be a numeric float")
        v = float(self.value)
        if v != v or v == float("inf") or v == float("-inf"):
            raise ValueError("DevelopmentLabel.value must be finite")
        if self.orientation not in ("higher_better", "lower_better"):
            raise ValueError(
                f"DevelopmentLabel.orientation must be 'higher_better' or 'lower_better'"
            )
        if not isinstance(self.assay_id, str) or not self.assay_id:
            raise ValueError("DevelopmentLabel.assay_id must be nonempty")


@dataclass(frozen=True)
class SelectionAllowance:
    """Declares how many development-selection labels are allowed.

    A fixed-last-checkpoint protocol may set allowance=0.
    The actual_development_ids must match exactly allowance entries.
    """
    allowance: int
    actual_development_ids: Tuple[str, ...]  # exact declared development IDs

    def __post_init__(self) -> None:
        object.__setattr__(self, "actual_development_ids", tuple(self.actual_development_ids))
        if isinstance(self.allowance, bool) or not isinstance(self.allowance, int):
            raise TypeError("SelectionAllowance.allowance must be an int")
        if self.allowance < 0:
            raise ValueError("SelectionAllowance.allowance must be >= 0")
        if len(self.actual_development_ids) != self.allowance:
            raise ValueError(
                f"actual_development_ids has {len(self.actual_development_ids)} entries "
                f"but allowance is {self.allowance}"
            )
        for eid in self.actual_development_ids:
            if not isinstance(eid, str) or not eid:
                raise ValueError("all actual_development_ids must be nonempty strings")
        if len(set(self.actual_development_ids)) != len(self.actual_development_ids):
            raise ValueError("Duplicate endpoint_id in actual_development_ids")


@dataclass(frozen=True)
class PairDraw:
    left_id: str
    right_id: str
    better_id: Optional[str]
    worse_id: Optional[str]

    @property
    def informative(self):
        return self.better_id is not None


@dataclass(frozen=True)
class RestrictedPostData:
    """Only selected constructs and their labels survive validation.

    The outcome-free full manifest is an init-only argument; it is not stored.
    Development labels are counted separately and never returned for fitting.
    """
    manifest: InitVar[EndpointManifest]
    budget: SelectedBudget
    measured: Tuple[MeasuredEndpoint, ...]
    fitting_count: int
    development_labels: Tuple[DevelopmentLabel, ...]
    selection_allowance: SelectionAllowance
    inherited_exposure_meta: Optional[Tuple[Tuple[str, str], ...]] = None
    selected_entries: Tuple[EndpointEntry, ...] = field(init=False)

    def __post_init__(self, manifest):
        object.__setattr__(self, "measured", tuple(self.measured))
        object.__setattr__(self, "development_labels", tuple(self.development_labels))
        expected = select_budget(manifest, self.budget.budget_n, self.budget.budget_seed)
        if self.budget != expected:
            raise ValueError("budget does not match manifest")
        selected = set(self.budget.selected_ids)
        measured_ids = [m.endpoint_id for m in self.measured]
        if len(set(measured_ids)) != len(measured_ids):
            raise ValueError("Duplicate measured endpoint ID")
        if set(measured_ids) - selected:
            raise ValueError("measured endpoint not in selected budget")
        if set(measured_ids) != selected:
            raise ValueError("measured endpoints must exactly match selected budget")
        if type(self.fitting_count) is not int or self.fitting_count != len(selected):
            raise ValueError("fitting_count must equal selected budget")
        dev_ids = [d.endpoint_id for d in self.development_labels]
        if len(set(dev_ids)) != len(dev_ids) or set(dev_ids) != set(self.selection_allowance.actual_development_ids):
            raise ValueError("development labels must exactly match declared allowance")
        orientations = {}
        for label in self.measured + self.development_labels:
            entry = manifest.by_id(label.endpoint_id)
            expected_role = "post_pool" if label in self.measured else "development"
            if entry.role != expected_role or label.assay_id != entry.assay_id:
                raise ValueError("label role or assay does not match manifest")
            if label.assay_id in orientations and orientations[label.assay_id] != label.orientation:
                raise ValueError("mixed orientations within assay")
            orientations[label.assay_id] = label.orientation
        object.__setattr__(self, "selected_entries",
                           tuple(manifest.by_id(eid) for eid in self.budget.selected_ids))
        object.__setattr__(self, "measured", tuple(sorted(self.measured, key=lambda m: m.endpoint_id)))

    def constructs(self, manifest=None):
        if manifest is not None and manifest.manifest_digest() != self.budget.manifest_digest:
            raise ValueError("manifest identity changed")
        return tuple((e.endpoint_id, e.construct) for e in self.selected_entries)

    def pair_groups(self):
        groups = {}
        for label in self.measured:
            groups.setdefault(label.assay_id, []).append(label.endpoint_id)
        return tuple(tuple(sorted(ids)) for _, ids in sorted(groups.items()) if len(ids) >= 2)

    def draw_pair(self, rng):
        """Select assay and two IDs before inspecting their values; never replace ties."""
        groups = self.pair_groups()
        if not groups:
            raise ValueError("ordering unsupported: fewer than two endpoints in each assay")
        left, right = rng.sample(rng.choice(groups), 2)
        labels = {m.endpoint_id: m for m in self.measured}
        a, b = labels[left], labels[right]
        if a.value == b.value:
            return PairDraw(left, right, None, None)
        a_better = (a.value > b.value) == (a.orientation == "higher_better")
        return PairDraw(left, right, left if a_better else right, right if a_better else left)

    def ordered_pairs(self, manifest=None, assay_id=None):
        """Deterministic identity-adjacent diagnostic, never used as the training sampler."""
        self.constructs(manifest)
        labels = {m.endpoint_id: m for m in self.measured}
        pairs = []
        for ids in self.pair_groups():
            if assay_id is not None and labels[ids[0]].assay_id != assay_id:
                continue
            for left, right in zip(ids, ids[1:]):
                a, b = labels[left], labels[right]
                if a.value != b.value:
                    better = (a.value > b.value) == (a.orientation == "higher_better")
                    pairs.append((left, right) if better else (right, left))
        return tuple(pairs)

    def data_digest(self):
        payload = {"budget": asdict(self.budget),
                   "measured": [asdict(m) for m in self.measured],
                   "development": sorted((asdict(m) for m in self.development_labels), key=lambda m: m["endpoint_id"]),
                   "allowance": asdict(self.selection_allowance)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8")).hexdigest()
