"""Broad qualification layer for the trastuzumab PoC data contracts.

All raw corpus acquisition is external.  Callers supply explicit normalized input
documents; this module validates, deduplicates, and qualifies them.

No Torch dependency.  No automatic data paths or fallback logic.
No guessed negatives.  Evidence declarations are NOT external biological verification.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Sequence, Tuple

from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES


# ---------------------------------------------------------------------------
# ESM / native tokenizer length limits
# ---------------------------------------------------------------------------
# Released encoder joint layout adds BOS/EOS. Native paired DPLM adds five
# special tokens; heavy-only adds three. Both limits are explicit in declarations.
_ESM_JOINT_LIMIT = 1024  # includes BOS/EOS
_NATIVE_TOKEN_LIMIT = 1026  # released native DPLM max_position_embeddings

# CANONICAL_RESIDUES is defined in views.py as "ACDEFGHIKLMNPQRSTVWY"
_CANON_SET: FrozenSet[str] = frozenset(CANONICAL_RESIDUES)


# ---------------------------------------------------------------------------
# Input record dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputRecord:
    """Caller-supplied normalized record before qualification.

    Fields
    ------
    record_id           : stable caller-assigned ID (unique within the manifest)
    source_id           : which declared source this came from
    source_record_id    : ID within the source (for deduplication reporting)
    H                   : heavy-chain amino-acid string (canonical 20AA, nonempty)
    L                   : light-chain string or None (explicitly heavy-only if None)
    Ag                  : antigen amino-acid string (canonical 20AA, nonempty)
    antibody_format     : "paired" | "heavy_only"
    antibody_family     : caller-declared family string (nonempty)
    target_family       : caller-declared target family string (nonempty)
    partition           : "train" | "val" | "test"
    h_eligible          : eligible H residue coordinate indices (0-based, unique, in-bounds)
    l_eligible          : eligible L residue coordinate indices (0-based, unique, in-bounds) or None
    aux_region          : optional auxiliary region coords as ((role, idx), ...) or None
    evidence_reference  : e.g. PDB id, DOI, accession (nonempty string)
    evidence_assay      : assay type string (nonempty)
    evidence_criterion  : criterion used for positive call (nonempty)
    qualification_rule  : name of the declared qualification rule (nonempty)
    """
    record_id: str
    source_id: str
    source_record_id: str
    H: str
    L: Optional[str]
    Ag: str
    antibody_format: str          # "paired" | "heavy_only"
    antibody_family: str
    target_family: str
    partition: str                # "train" | "val" | "test"
    h_eligible: Tuple[int, ...]
    l_eligible: Optional[Tuple[int, ...]]
    aux_region: Optional[Tuple[Tuple[str, int], ...]]
    evidence_reference: str
    evidence_assay: str
    evidence_criterion: str
    qualification_rule: str

    def __post_init__(self) -> None:
        # -- string fields nonempty --
        for attr in ("record_id", "source_id", "source_record_id",
                     "H", "Ag", "antibody_format", "antibody_family",
                     "target_family", "partition",
                     "evidence_reference", "evidence_assay",
                     "evidence_criterion", "qualification_rule"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val:
                raise ValueError(f"InputRecord.{attr} must be a nonempty string, got {val!r}")

        # -- antibody_format --
        if self.antibody_format not in ("paired", "heavy_only"):
            raise ValueError(
                f"antibody_format must be 'paired' or 'heavy_only', got {self.antibody_format!r}"
            )

        # -- partition --
        if self.partition not in ("train", "val", "test"):
            raise ValueError(
                f"partition must be 'train', 'val', or 'test', got {self.partition!r}"
            )

        # -- sequences: canonical 20AA --
        for attr in ("H", "Ag"):
            seq = getattr(self, attr)
            bad = [c for c in seq if c not in _CANON_SET]
            if bad:
                raise ValueError(
                    f"InputRecord.{attr} contains non-canonical residues: {bad!r}"
                )

        if self.L is not None:
            if not isinstance(self.L, str) or not self.L:
                raise ValueError("InputRecord.L must be a nonempty string or None")
            bad = [c for c in self.L if c not in _CANON_SET]
            if bad:
                raise ValueError(
                    f"InputRecord.L contains non-canonical residues: {bad!r}"
                )

        # -- format/L consistency --
        if self.antibody_format == "paired" and self.L is None:
            raise ValueError("antibody_format='paired' requires L to be present")
        if self.antibody_format == "heavy_only" and self.L is not None:
            raise ValueError("antibody_format='heavy_only' requires L to be None")

        # -- h_eligible --
        if not isinstance(self.h_eligible, tuple):
            raise TypeError("h_eligible must be a tuple")
        if not self.h_eligible:
            raise ValueError("h_eligible must be nonempty")
        _check_coord_indices(self.h_eligible, len(self.H), "h_eligible")

        # -- l_eligible --
        if self.l_eligible is not None:
            if not isinstance(self.l_eligible, tuple):
                raise TypeError("l_eligible must be a tuple or None")
            if not self.l_eligible:
                raise ValueError("l_eligible must be nonempty when present")
            if self.L is None:
                raise ValueError("l_eligible given but L is None")
            _check_coord_indices(self.l_eligible, len(self.L), "l_eligible")

        # -- aux_region: ((role, idx), ...) --
        if self.aux_region is not None:
            if not isinstance(self.aux_region, tuple):
                raise TypeError("aux_region must be a tuple or None")
            seen_aux: Set[Tuple[str, int]] = set()
            for item in self.aux_region:
                if not (isinstance(item, tuple) and len(item) == 2):
                    raise ValueError(f"aux_region entry must be (role, idx), got {item!r}")
                role, idx = item
                if role not in ("H", "L"):
                    raise ValueError(f"aux_region role must be H or L, got {role!r}")
                chain_len = len(self.H) if role == "H" else (len(self.L) if self.L else 0)
                if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= chain_len:
                    raise ValueError(
                        f"aux_region coord ({role!r}, {idx!r}) out of bounds (chain len {chain_len})"
                    )
                if (role, idx) in seen_aux:
                    raise ValueError(f"duplicate aux_region coord {(role, idx)!r}")
                seen_aux.add((role, idx))
            eligible = {("H", i) for i in self.h_eligible} | {("L", i) for i in self.l_eligible or ()}
            if not seen_aux or not seen_aux <= eligible:
                raise ValueError("aux_region must be a nonempty subset of eligible coordinates")
        object.__setattr__(self, "h_eligible", tuple(sorted(self.h_eligible)))
        if self.l_eligible is not None:
            object.__setattr__(self, "l_eligible", tuple(sorted(self.l_eligible)))
        if self.aux_region is not None:
            object.__setattr__(self, "aux_region", tuple(sorted(self.aux_region)))


def _check_coord_indices(indices: Tuple[int, ...], chain_len: int, name: str) -> None:
    seen: Set[int] = set()
    for idx in indices:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"{name} indices must be ints, got {type(idx).__name__!r}")
        if idx < 0 or idx >= chain_len:
            raise ValueError(
                f"{name} index {idx} out of bounds for chain length {chain_len}"
            )
        if idx in seen:
            raise ValueError(f"duplicate index {idx} in {name}")
        seen.add(idx)


# ---------------------------------------------------------------------------
# Source inventory and qualification rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceDeclaration:
    source_id: str
    description: str     # human label; nonempty
    priority: int        # lower int = higher priority for duplicate ownership

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("SourceDeclaration.source_id must be nonempty")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("SourceDeclaration.description must be nonempty")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("SourceDeclaration.priority must be an int")


@dataclass(frozen=True)
class QualificationRule:
    rule_name: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_name, str) or not self.rule_name:
            raise ValueError("QualificationRule.rule_name must be nonempty")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("QualificationRule.description must be nonempty")


@dataclass(frozen=True)
class GroupingDeclaration:
    """Caller declares how to group/separate records.

    separation_axes: which family fields enforce train/val/test separation.
        Allowed values: "antibody_family", "target_family", or both.
    family_id_map: reserved, unsupported. Resolve family identities in the explicit
        normalized records before qualification; non-null overrides are rejected.
    linked_groups: optional list of frozensets of record_ids that must be placed in the
        same partition (transitive closure is enforced).
    """
    separation_axes: Tuple[str, ...]
    family_id_map: Optional[Tuple[Tuple[str, Tuple[str, str]], ...]] = None
    linked_groups: Optional[Tuple[FrozenSet[str], ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "separation_axes", tuple(self.separation_axes))
        if self.linked_groups is not None:
            object.__setattr__(self, "linked_groups", tuple(frozenset(g) for g in self.linked_groups))
        allowed = {"antibody_family", "target_family"}
        for axis in self.separation_axes:
            if axis not in allowed:
                raise ValueError(
                    f"separation_axes entry {axis!r} not in {allowed}"
                )
        if not self.separation_axes:
            raise ValueError("separation_axes must be nonempty")
        if len(set(self.separation_axes)) != len(self.separation_axes):
            raise ValueError("duplicate separation axis")
        if self.family_id_map is not None:
            raise ValueError("family_id_map overrides are unsupported; reconcile declared record families before qualification")


# ---------------------------------------------------------------------------
# Qualified record (result of qualification)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualifiedRecord:
    """A record that has passed all qualification checks."""
    record_id: str
    owning_source_id: str       # deterministic by declared source priority
    supporting_refs: Tuple[str, ...]   # all source_record_ids claiming this content
    H: str
    L: Optional[str]
    Ag: str
    antibody_format: str
    antibody_family: str
    target_family: str
    partition: str
    h_eligible: Tuple[int, ...]
    l_eligible: Optional[Tuple[int, ...]]
    aux_region: Optional[Tuple[Tuple[str, int], ...]]
    evidence_reference: str
    evidence_assay: str
    evidence_criterion: str
    qualification_rule: str

    # Stable content key for deduplication (H + L + Ag, ignoring provenance)
    content_key: str  # hex digest

    def eligible_h_coords(self) -> Tuple[Tuple[str, int], ...]:
        return tuple(("H", i) for i in self.h_eligible)

    def eligible_l_coords(self) -> Tuple[Tuple[str, int], ...]:
        if self.l_eligible is None:
            return ()
        return tuple(("L", i) for i in self.l_eligible)

    def eligible_union(self) -> Tuple[Tuple[str, int], ...]:
        return self.eligible_h_coords() + self.eligible_l_coords()


def _content_key(H: str, L: Optional[str], Ag: str) -> str:
    payload = f"hlag1|{H}|{L or ''}|{Ag}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Exclusion reasons
# ---------------------------------------------------------------------------

EXCL_EXPLICIT_RECORD = "explicit_record_exclusion"
EXCL_CONTENT = "explicit_content_exclusion"
EXCL_ANTIBODY_FAMILY = "explicit_antibody_family_exclusion"
EXCL_TARGET_FAMILY = "explicit_target_family_exclusion"
EXCL_OVERLENGTH_JOINT = "overlength_joint_H_L_Ag"
EXCL_OVERLENGTH_PAIRED = "overlength_paired_arm"
EXCL_OVERLENGTH_ABONLY = "overlength_abonly_arm"
EXCL_MALFORMED = "malformed_record"
EXCL_CONFLICT = "conflicting_record"


@dataclass(frozen=True)
class ExcludedRecord:
    record_id: str
    reason: str
    detail: str


# ---------------------------------------------------------------------------
# Manifest declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestDeclaration:
    """Everything a caller must declare before qualification runs.

    sources: recognized source declarations (nonempty, unique source_ids).
    rules: recognized qualification rules (nonempty, unique rule_names).
    source_priority_order: source_ids in priority order (lowest index = highest priority).
        Must list every declared source_id exactly once.
    grouping: how to separate and group records.
    excluded_record_ids: record_ids to exclude before any processing.
    excluded_content_keys: content keys (H/L/Ag digests) to exclude.
    excluded_antibody_families: antibody_family values to exclude.
    excluded_target_families: target_family values to exclude.
    """
    sources: Tuple[SourceDeclaration, ...]
    rules: Tuple[QualificationRule, ...]
    source_priority_order: Tuple[str, ...]
    grouping: GroupingDeclaration
    excluded_record_ids: Tuple[str, ...] = ()
    excluded_content_keys: Tuple[str, ...] = ()
    excluded_antibody_families: Tuple[str, ...] = ()
    excluded_target_families: Tuple[str, ...] = ()
    encoder_max_tokens: int = _ESM_JOINT_LIMIT
    native_max_tokens: int = _NATIVE_TOKEN_LIMIT

    def __post_init__(self) -> None:
        for name in ("sources", "rules", "source_priority_order", "excluded_record_ids", "excluded_content_keys", "excluded_antibody_families", "excluded_target_families"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for value in (self.encoder_max_tokens, self.native_max_tokens):
            if type(value) is not int or value < 4:
                raise ValueError("token limits must be integers >= 4")
        if not self.sources:
            raise ValueError("ManifestDeclaration.sources must be nonempty")
        source_ids = [s.source_id for s in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Duplicate source_ids in ManifestDeclaration.sources")

        if not self.rules:
            raise ValueError("ManifestDeclaration.rules must be nonempty")
        rule_names = [r.rule_name for r in self.rules]
        if len(set(rule_names)) != len(rule_names):
            raise ValueError("Duplicate rule_names in ManifestDeclaration.rules")

        if set(self.source_priority_order) != set(source_ids):
            raise ValueError(
                "source_priority_order must list every declared source_id exactly once"
            )
        if len(self.source_priority_order) != len(source_ids):
            raise ValueError("source_priority_order has duplicates")

    @property
    def source_id_set(self) -> FrozenSet[str]:
        return frozenset(s.source_id for s in self.sources)

    @property
    def rule_name_set(self) -> FrozenSet[str]:
        return frozenset(r.rule_name for r in self.rules)

    def source_priority(self, source_id: str) -> int:
        return self.source_priority_order.index(source_id)


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualificationCoverage:
    input_rows: int
    unique_pairs: int          # unique (H, L, Ag) content keys
    unique_ab: int             # unique (H, L) constructs
    unique_ag: int             # unique Ag strings
    sources: Tuple[str, ...]   # source_ids with at least one qualified record
    antibody_families: Tuple[str, ...]
    target_families: Tuple[str, ...]
    partitions: Dict[str, int]    # partition -> count
    excluded_reasons: Dict[str, int]  # reason -> count
    length_distribution: Dict[str, Any]   # min/max/mean for H, L, Ag
    evidence_provenance: Tuple[str, ...]  # unique evidence_reference values


# ---------------------------------------------------------------------------
# Qualified population
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualifiedPopulation:
    """Immutable qualified population.  Training sampler only sees train partition."""
    records: Tuple[QualifiedRecord, ...]
    excluded: Tuple[ExcludedRecord, ...]
    coverage: QualificationCoverage
    manifest_digest: str   # SHA256 of the serialized ManifestDeclaration
    population_digest: str  # SHA256 of sorted qualified record_ids + content_keys

    def train_records(self) -> Tuple[QualifiedRecord, ...]:
        return tuple(r for r in self.records if r.partition == "train")

    def by_partition(self, partition: str) -> Tuple[QualifiedRecord, ...]:
        return tuple(r for r in self.records if r.partition == partition)


# ---------------------------------------------------------------------------
# Core qualification function
# ---------------------------------------------------------------------------

def qualify_manifest(
    declaration: ManifestDeclaration,
    records: Sequence[InputRecord],
) -> QualifiedPopulation:
    """Qualify a caller-supplied record list against a manifest declaration.

    Performs validation, deduplication, exclusion, length filtering, and
    group-separation enforcement.  All separation checks run over the FULL
    supplied manifest before any filtering.

    Raises
    ------
    ValueError
        For ambiguous/conflicting records that must not disappear silently.
    """
    records = tuple(records)
    _validate_record_ids_unique(records)
    _check_group_separation(records, declaration.grouping)
    by_content = defaultdict(list)
    source_records = {}
    antibody_families, target_families = {}, {}
    for rec in records:
        if rec.source_id not in declaration.source_id_set:
            raise ValueError(f"source_id {rec.source_id!r} is not declared")
        if rec.qualification_rule not in declaration.rule_name_set:
            raise ValueError(f"qualification_rule {rec.qualification_rule!r} is not declared")
        ck = _content_key(rec.H, rec.L, rec.Ag)
        source_key = (rec.source_id, rec.source_record_id)
        signature = (ck, rec.partition, rec.antibody_family, rec.target_family,
                     rec.h_eligible, rec.l_eligible, rec.aux_region,
                     rec.evidence_assay, rec.evidence_criterion, rec.qualification_rule)
        if source_key in source_records and source_records[source_key] != signature:
            raise ValueError(f"conflicting source-record identity {source_key}")
        source_records[source_key] = signature
        for mapping, key, value in ((antibody_families, (rec.H, rec.L), rec.antibody_family),
                                     (target_families, rec.Ag, rec.target_family)):
            if key in mapping and mapping[key] != value:
                raise ValueError("identical sequence assigned conflicting family identities")
            mapping[key] = value
        by_content[ck].append(rec)
    for group in by_content.values():
        contracts = {(r.partition, r.antibody_family, r.target_family,
                      r.h_eligible, r.l_eligible, r.aux_region) for r in group}
        if len(contracts) != 1:
            raise ValueError("conflicting duplicate content contracts")

    excluded, qualified = [], []
    for ck, group in sorted(by_content.items()):
        candidates = []
        for rec in group:
            reason = None
            if rec.record_id in declaration.excluded_record_ids:
                reason = EXCL_EXPLICIT_RECORD
            elif ck in declaration.excluded_content_keys:
                reason = EXCL_CONTENT
            elif rec.antibody_family in declaration.excluded_antibody_families:
                reason = EXCL_ANTIBODY_FAMILY
            elif rec.target_family in declaration.excluded_target_families:
                reason = EXCL_TARGET_FAMILY
            elif len(rec.H) + len(rec.L or "") + len(rec.Ag) + 2 > declaration.encoder_max_tokens:
                reason = EXCL_OVERLENGTH_JOINT
            elif len(rec.H) + len(rec.L or "") + (5 if rec.L else 3) > declaration.native_max_tokens:
                reason = EXCL_OVERLENGTH_PAIRED
            if reason:
                excluded.append(ExcludedRecord(rec.record_id, reason, reason))
            else:
                candidates.append(rec)
        if not candidates:
            continue
        owner = min(candidates, key=lambda r: (declaration.source_priority(r.source_id),
                                              r.source_record_id, r.record_id))
        # Distinct positive assays provide provenance; their numbers are never compared here.
        refs = tuple(sorted(json.dumps({"source": r.source_id, "record": r.source_record_id,
                                        "reference": r.evidence_reference, "assay": r.evidence_assay,
                                        "criterion": r.evidence_criterion, "rule": r.qualification_rule},
                                       sort_keys=True) for r in candidates))
        fields = asdict(owner)
        fields.pop("source_id")
        fields.pop("source_record_id")
        qualified.append(QualifiedRecord(**fields, owning_source_id=owner.source_id,
                                         supporting_refs=refs, content_key=ck))
    qualified = tuple(sorted(qualified, key=lambda r: r.record_id))
    excluded = tuple(sorted(excluded, key=lambda r: r.record_id))
    return QualifiedPopulation(qualified, excluded, _compute_coverage(records, qualified, excluded),
                               _digest_manifest(declaration), _digest_population(qualified))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_record_ids_unique(records: Sequence[InputRecord]) -> None:
    seen: Set[str] = set()
    for rec in records:
        if rec.record_id in seen:
            raise ValueError(f"Duplicate record_id {rec.record_id!r} in input")
        seen.add(rec.record_id)


def _check_group_separation(
    records: List[InputRecord],
    grouping: GroupingDeclaration,
) -> None:
    """Enforce that no family group straddles partitions."""
    for axis in grouping.separation_axes:
        families = defaultdict(set)
        for rec in records:
            families[getattr(rec, axis)].add(rec.partition)
        for family, partitions in families.items():
            if len(partitions) > 1:
                raise ValueError(f"Group separation violation: {axis} {family!r} spans partitions")
    if grouping.linked_groups:
        known = {r.record_id for r in records}
        if any(not group or not group <= known for group in grouping.linked_groups):
            raise ValueError("linked groups must reference existing record IDs")

    # Linked groups: transitive closure enforcement
    if grouping.linked_groups:
        # Build map: record_id -> partition
        part_map = {rec.record_id: rec.partition for rec in records}
        # Transitive closure via union-find
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
                x = parent.get(x, x)
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for group_set in grouping.linked_groups:
            ids = list(group_set)
            for i in range(1, len(ids)):
                union(ids[0], ids[i])

        # Check each component has a single partition
        component_partitions: Dict[str, Set[str]] = {}
        for rec_id, part in part_map.items():
            root = find(rec_id)
            component_partitions.setdefault(root, set()).add(part)

        for root, parts in component_partitions.items():
            if len(parts) > 1:
                raise ValueError(
                    f"Linked group rooted at {root!r} spans multiple partitions {parts}"
                )


def _compute_coverage(
    input_records: Sequence[InputRecord],
    qualified: Tuple[QualifiedRecord, ...],
    excluded: Tuple[ExcludedRecord, ...],
) -> QualificationCoverage:
    unique_pairs = len(set(r.content_key for r in qualified))
    unique_ab = len(set((r.H, r.L) for r in qualified))
    unique_ag = len(set(r.Ag for r in qualified))
    sources = tuple(sorted(set(r.owning_source_id for r in qualified)))
    ab_fams = tuple(sorted(set(r.antibody_family for r in qualified)))
    tgt_fams = tuple(sorted(set(r.target_family for r in qualified)))

    partitions: Dict[str, int] = {}
    for r in qualified:
        partitions[r.partition] = partitions.get(r.partition, 0) + 1

    excluded_reasons: Dict[str, int] = {}
    for e in excluded:
        excluded_reasons[e.reason] = excluded_reasons.get(e.reason, 0) + 1

    h_lens = [len(r.H) for r in qualified]
    l_lens = [len(r.L) for r in qualified if r.L is not None]
    ag_lens = [len(r.Ag) for r in qualified]

    def _stats(lens: List[int]) -> Dict[str, Any]:
        if not lens:
            return {"min": None, "max": None, "mean": None, "count": 0}
        return {
            "min": min(lens),
            "max": max(lens),
            "mean": sum(lens) / len(lens),
            "count": len(lens),
        }

    length_distribution = {
        "H": _stats(h_lens),
        "L": _stats(l_lens),
        "Ag": _stats(ag_lens),
    }

    evidence_provenance = tuple(sorted(set(r.evidence_reference for r in qualified)))

    return QualificationCoverage(
        input_rows=len(input_records),
        unique_pairs=unique_pairs,
        unique_ab=unique_ab,
        unique_ag=unique_ag,
        sources=sources,
        antibody_families=ab_fams,
        target_families=tgt_fams,
        partitions=partitions,
        excluded_reasons=excluded_reasons,
        length_distribution=length_distribution,
        evidence_provenance=evidence_provenance,
    )


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _digest_manifest(declaration: ManifestDeclaration) -> str:
    value = asdict(declaration)
    value["sources"] = sorted(value["sources"], key=lambda x: x["source_id"])
    value["rules"] = sorted(value["rules"], key=lambda x: x["rule_name"])
    value["grouping"]["separation_axes"] = sorted(value["grouping"]["separation_axes"])
    value["grouping"]["linked_groups"] = sorted(sorted(g) for g in declaration.grouping.linked_groups or ())
    for name in ("excluded_record_ids", "excluded_content_keys", "excluded_antibody_families", "excluded_target_families"):
        value[name] = sorted(value[name])
    return _json_digest(value)


def _digest_population(qualified: Tuple[QualifiedRecord, ...]) -> str:
    return _json_digest([asdict(r) for r in sorted(qualified, key=lambda r: r.record_id)])
