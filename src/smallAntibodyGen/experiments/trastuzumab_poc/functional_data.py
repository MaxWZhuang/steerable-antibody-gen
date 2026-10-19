"""Outcome-free construct and condition-scope contracts for functional transfer.

A functional-constructs document names the constructs (H, optional L, Ag) and the
measurement groups that will later supply an ordering signal.  It carries NO
measured value.  Measurements live in a separate document that is opened only
after the observation budget is fixed (see ``functional_protocol``).

No Torch dependency, no automatic data paths, no outcome-dependent filtering.

Curator assertions versus checks
--------------------------------
The study contract is a set of explicit curator assertions: that one scalar head can
serve ONE target property, under the listed condition scope, across the allowlisted
endpoint semantics/units/directions.  Nothing here equates expression, affinity,
EC50, enrichment or off-rate; the allowlist is the curator's claim and every entry
carries a justification.  What IS checked is linear and dict-based:

* each group's ``{semantics, units, direction}`` equals an allowlisted entry and its
  ``context_id`` is in the declared scope;
* group members are unique, known, share one partition and carry an antigen that is
  byte-identical to the group's ``antigen_sha256``;
* sequences are canonical, format and light chain agree, identifiers are unambiguous.

NOT checked (and not claimed): transitivity inside a group, rank reversals under
unobserved conditions, and consistency between groups.

Cross-manifest boundary
-----------------------
``qualify_functional`` takes the ORIGINAL outcome-free broad ``InputRecord`` sequence,
not only the ``QualifiedPopulation``.  A population keeps only the record IDs and
reasons of excluded rows, so it alone cannot show that an excluded broad row agrees
with the functional document on family, partition and format.  The records are
re-qualified under the supplied declaration and must reproduce the supplied population
exactly, then every broad row (retained or excluded) takes part in the checks:

* content / family exclusions of the declaration apply to functional constructs
  globally; record-ID exclusions name broad ROWS and do not apply to functional
  construct IDs (different namespaces: a construct whose ID string equals an unrelated
  excluded broad record ID is accepted);
* a construct sharing its ``(H, L, Ag)`` content with any broad row must carry the same
  format, antibody family, target family and partition;
* the same ``(H, L)`` or the same ``Ag`` must carry the same family on both sides;
* no family on a separation axis may span partitions across the union;
* a functional ``source_id`` that is also a declared broad source must have the same
  description, and a shared ``(source_id, source_record_id)`` must name the same
  content.  Other functional source IDs are scoped to this document.

Consistent train/train sequence overlap is ALLOWED and reported in
``shared_with_denoising_train``.  The denoising population is never mutated, expanded
or re-weighted; functional constructs never reach the family-first sampler.

Record-ID exclusions do not propagate by content: an excluded broad row still takes
part in the consistency checks above, but it does not exclude a functional construct
that merely shares its sequence.  ``linked_groups`` reference broad record IDs only, so
functional constructs (which have no broad record ID) are covered by the content and
family checks instead.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    InputRecord, ManifestDeclaration, QualifiedPopulation, _content_key, _digest_manifest,
    qualify_manifest,
)
from smallAntibodyGen.models.antibody_antigen.views import CANONICAL_RESIDUES

CONSTRUCTS_SCHEMA = "trastuzumab-functional-constructs/1"
CONTRACT_SCHEMA = "trastuzumab-functional-contract/1"
QUALIFIED_SCHEMA = "trastuzumab-functional-qualified/1"
DIRECTIONS = ("higher_better", "lower_better")
PARTITIONS = ("train", "val", "test")
FORMATS = ("paired", "heavy_only")

_CANON = frozenset(CANONICAL_RESIDUES)
_HEX = frozenset("0123456789abcdef")


# ---------------------------------------------------------------------------
# Strict transport helpers (byte-compatible with artifacts.canonical_bytes / digest,
# which cannot be imported here because it pulls in Torch)
# ---------------------------------------------------------------------------

def canonical_bytes(value) -> bytes:
    """Canonical JSON bytes; signed zero is normalized and nonfinite values are rejected."""
    def clean(x):
        if isinstance(x, float):
            if not math.isfinite(x):
                raise ValueError("JSON values must be finite")
            return 0.0 if x == 0 else x
        if isinstance(x, dict):
            if any(not isinstance(k, str) for k in x):
                raise TypeError("JSON object keys must be strings")
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if x is None or isinstance(x, (str, int, bool)):
            return x
        raise TypeError(f"unsupported JSON value {type(x).__name__}")
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def json_digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_strict_json(path):
    """Decode JSON, rejecting duplicate keys, nonfinite constants and overflowing numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def refuse(token):
        raise ValueError(f"nonfinite JSON: {token}")

    value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=pairs, parse_constant=refuse)
    canonical_bytes(value)
    return value


def exact_keys(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{where}: expected exactly {sorted(keys)}")
    return value


def identifier(value, where):
    """Unambiguous identifier: no surrounding whitespace and no NUL (used as a hash separator)."""
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{where} must be a nonempty identifier without surrounding "
                         "whitespace or NUL")
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{where} must be nonempty text")
    return value


def _nonempty(value, where):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a nonempty string")
    return value


def sha256_hex(value, where):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{where} must be a lowercase SHA-256 hex digest")
    return value


def _sequence(value, where):
    if not isinstance(value, str) or not value or any(c not in _CANON for c in value):
        raise ValueError(f"{where} must be a nonempty canonical amino-acid string")
    return value


def _unique(values: Iterable, where: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {where}: {value!r}")
        seen.add(value)


def _listed(value, where):
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a JSON list")
    return tuple(value)


def _rows(value, keys, where, build):
    return tuple(build(**exact_keys(row, keys, f"{where} row")) for row in _listed(value, where))


# ---------------------------------------------------------------------------
# Contract value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionalConstruct:
    """One physical construct; ``construct_id`` is scoped to the functional document.

    The ID is metadata only.  Model inputs are built from ``H``, ``L`` and ``Ag`` alone.
    """
    construct_id: str
    source_id: str
    source_record_id: str
    H: str
    L: Optional[str]
    Ag: str
    antibody_format: str
    antibody_family: str
    target_family: str
    partition: str
    evidence_reference: str

    def __post_init__(self) -> None:
        for name in ("construct_id", "source_id", "source_record_id"):
            identifier(getattr(self, name), f"construct.{name}")
        for name in ("antibody_family", "target_family"):
            _nonempty(getattr(self, name), f"construct.{name}")
        _text(self.evidence_reference, "construct.evidence_reference")
        _sequence(self.H, "construct.H")
        _sequence(self.Ag, "construct.Ag")
        if self.L is not None:
            _sequence(self.L, "construct.L")
        if self.antibody_format not in FORMATS:
            raise ValueError(f"antibody_format must be one of {FORMATS}")
        if self.antibody_format == "paired" and self.L is None:
            raise ValueError("antibody_format='paired' requires L")
        if self.antibody_format == "heavy_only" and self.L is not None:
            raise ValueError("antibody_format='heavy_only' requires L to be null")
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")

    @property
    def content_key(self) -> str:
        return _content_key(self.H, self.L, self.Ag)

    @property
    def antigen_sha256(self) -> str:
        return hashlib.sha256(self.Ag.encode("ascii")).hexdigest()

    def document(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Endpoint:
    semantics: str
    units: str
    direction: str

    def __post_init__(self) -> None:
        _text(self.semantics, "endpoint.semantics")
        _text(self.units, "endpoint.units")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"endpoint.direction must be one of {DIRECTIONS}")

    def key(self) -> Tuple[str, str, str]:
        return (self.semantics, self.units, self.direction)

    def document(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibleEndpoint:
    """A curator-asserted allowlist entry for the single scalar head."""
    semantics: str
    units: str
    direction: str
    justification: str

    def __post_init__(self) -> None:
        Endpoint(self.semantics, self.units, self.direction)
        _text(self.justification, "compatible_endpoint.justification")

    def endpoint(self) -> Endpoint:
        return Endpoint(self.semantics, self.units, self.direction)

    def document(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyContract:
    """Curator assertion: one target property, one condition scope, allowlisted endpoints."""
    property_id: str
    property_description: str
    scope_id: str
    scope_description: str
    context_ids: Tuple[str, ...]
    compatible_endpoints: Tuple[CompatibleEndpoint, ...]
    assertion: str

    def __post_init__(self) -> None:
        identifier(self.property_id, "target_property.property_id")
        _text(self.property_description, "target_property.description")
        identifier(self.scope_id, "condition_scope.scope_id")
        _text(self.scope_description, "condition_scope.description")
        _text(self.assertion, "study_contract.assertion")
        contexts = tuple(identifier(c, "condition_scope.context_ids entry") for c in self.context_ids)
        if not contexts:
            raise ValueError("condition_scope.context_ids must be nonempty")
        _unique(contexts, "context_id")
        endpoints = tuple(self.compatible_endpoints)
        if not endpoints or any(not isinstance(e, CompatibleEndpoint) for e in endpoints):
            raise ValueError("compatible_endpoints must be a nonempty list of endpoint entries")
        _unique((e.endpoint().key() for e in endpoints), "compatible endpoint")
        object.__setattr__(self, "context_ids", tuple(sorted(contexts)))
        object.__setattr__(self, "compatible_endpoints",
                           tuple(sorted(endpoints, key=lambda e: e.endpoint().key())))

    def allowed_endpoints(self) -> frozenset:
        return frozenset(e.endpoint().key() for e in self.compatible_endpoints)

    def document(self) -> Dict[str, Any]:
        return {"target_property": {"property_id": self.property_id,
                                    "description": self.property_description},
                "condition_scope": {"scope_id": self.scope_id,
                                    "description": self.scope_description,
                                    "context_ids": list(self.context_ids)},
                "compatible_endpoints": [e.document() for e in self.compatible_endpoints],
                "assertion": self.assertion}


@dataclass(frozen=True)
class FunctionalSource:
    source_id: str
    description: str

    def __post_init__(self) -> None:
        identifier(self.source_id, "source.source_id")
        _text(self.description, "source.description")

    def document(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FunctionalGroup:
    """Measurements in one group share a context and endpoint and are meant to be ordered."""
    group_id: str
    context_id: str
    endpoint: Endpoint
    antigen_sha256: str
    member_construct_ids: Tuple[str, ...]
    evidence_reference: str

    def __post_init__(self) -> None:
        identifier(self.group_id, "group.group_id")
        identifier(self.context_id, "group.context_id")
        if not isinstance(self.endpoint, Endpoint):
            raise ValueError("group.endpoint must be an endpoint object")
        sha256_hex(self.antigen_sha256, "group.antigen_sha256")
        _text(self.evidence_reference, "group.evidence_reference")
        members = tuple(identifier(m, "group member") for m in self.member_construct_ids)
        if not members:
            raise ValueError(f"group {self.group_id!r} needs at least one member")
        _unique(members, f"member of group {self.group_id!r}")
        object.__setattr__(self, "member_construct_ids", tuple(sorted(members)))

    def document(self) -> Dict[str, Any]:
        return {"group_id": self.group_id, "context_id": self.context_id,
                "endpoint": self.endpoint.document(), "antigen_sha256": self.antigen_sha256,
                "member_construct_ids": list(self.member_construct_ids),
                "evidence_reference": self.evidence_reference}


@dataclass(frozen=True)
class FunctionalConstructs:
    """A fully self-consistent functional document (no reference to a broad manifest yet).

    Rows are stored in canonical (sorted) order, so every digest is independent of the
    order in which the curator listed constructs, groups, members, sources or endpoints.
    Every construct must be a member of at least one group; a physical construct has ONE
    ID (repeated measurements are separate groups naming the same construct).
    """
    pairs_declaration_digest: str
    study_contract: StudyContract
    sources: Tuple[FunctionalSource, ...]
    constructs: Tuple[FunctionalConstruct, ...]
    groups: Tuple[FunctionalGroup, ...]
    _by_id: Dict[str, FunctionalConstruct] = field(init=False, repr=False, compare=False)
    _groups_by_id: Dict[str, FunctionalGroup] = field(init=False, repr=False, compare=False)
    _group_partition: Dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sha256_hex(self.pairs_declaration_digest, "pairs_declaration_digest")
        if not isinstance(self.study_contract, StudyContract):
            raise ValueError("study_contract must be a StudyContract")
        collected = {}
        for name, kind in (("sources", FunctionalSource), ("constructs", FunctionalConstruct),
                           ("groups", FunctionalGroup)):
            rows = tuple(getattr(self, name))
            if not rows or any(not isinstance(row, kind) for row in rows):
                raise ValueError(f"{name} must be a nonempty list of {kind.__name__}")
            collected[name] = rows
        sources = tuple(sorted(collected["sources"], key=lambda row: row.source_id))
        constructs = tuple(sorted(collected["constructs"], key=lambda row: row.construct_id))
        groups = tuple(sorted(collected["groups"], key=lambda row: row.group_id))
        _unique((s.source_id for s in sources), "source_id")
        _unique((c.construct_id for c in constructs), "construct_id")
        _unique((g.group_id for g in groups), "group_id")

        source_ids = {s.source_id for s in sources}
        by_id = {c.construct_id: c for c in constructs}
        records, contents, ab_family, ag_family = {}, {}, {}, {}
        for c in constructs:
            if c.source_id not in source_ids:
                raise ValueError(f"construct {c.construct_id!r} names undeclared source {c.source_id!r}")
            record = (c.source_id, c.source_record_id)
            if record in records:
                raise ValueError(f"constructs {records[record]!r} and {c.construct_id!r} claim the "
                                 f"same source record {record!r}")
            records[record] = c.construct_id
            if c.content_key in contents:
                raise ValueError(
                    f"constructs {contents[c.content_key]!r} and {c.construct_id!r} are the same "
                    "physical construct; use one construct ID and list it in several groups")
            contents[c.content_key] = c.construct_id
            for mapping, key, value, what in ((ab_family, (c.H, c.L), c.antibody_family, "antibody"),
                                              (ag_family, c.Ag, c.target_family, "target")):
                if key in mapping and mapping[key] != value:
                    raise ValueError(f"identical sequence carries conflicting {what} families")
                mapping[key] = value

        contract = self.study_contract
        contexts = set(contract.context_ids)
        allowed = contract.allowed_endpoints()
        group_partition, referenced = {}, set()
        for g in groups:
            if g.context_id not in contexts:
                raise ValueError(f"group {g.group_id!r} context {g.context_id!r} is outside the "
                                 "declared condition scope")
            if g.endpoint.key() not in allowed:
                raise ValueError(f"group {g.group_id!r} endpoint is not an allowlisted "
                                 "compatible endpoint")
            partitions = set()
            for member in g.member_construct_ids:
                construct = by_id.get(member)
                if construct is None:
                    raise ValueError(f"group {g.group_id!r} names unknown member {member!r}")
                if construct.antigen_sha256 != g.antigen_sha256:
                    raise ValueError(f"group {g.group_id!r} member {member!r} antigen differs from "
                                     "the group's antigen_sha256")
                partitions.add(construct.partition)
                referenced.add(member)
            if len(partitions) != 1:
                raise ValueError(f"group {g.group_id!r} spans partitions {sorted(partitions)}")
            group_partition[g.group_id] = next(iter(partitions))
        unreferenced = sorted(set(by_id) - referenced)
        if unreferenced:
            raise ValueError(f"constructs not referenced by any group: {unreferenced}")

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "constructs", constructs)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_groups_by_id", {g.group_id: g for g in groups})
        object.__setattr__(self, "_group_partition", group_partition)

    def construct(self, construct_id: str) -> FunctionalConstruct:
        try:
            return self._by_id[construct_id]
        except KeyError:
            raise ValueError(f"unknown construct {construct_id!r}") from None

    def group(self, group_id: str) -> FunctionalGroup:
        try:
            return self._groups_by_id[group_id]
        except KeyError:
            raise ValueError(f"unknown group {group_id!r}") from None

    def group_partition(self, group_id: str) -> str:
        self.group(group_id)
        return self._group_partition[group_id]

    def document(self) -> Dict[str, Any]:
        return {"schema": CONSTRUCTS_SCHEMA,
                "pairs_declaration_digest": self.pairs_declaration_digest,
                "study_contract": self.study_contract.document(),
                "sources": [s.document() for s in self.sources],
                "constructs": [c.document() for c in self.constructs],
                "groups": [g.document() for g in self.groups]}

    def document_digest(self) -> str:
        return json_digest(self.document())

    def contract_digest(self) -> str:
        return json_digest({"schema": CONTRACT_SCHEMA,
                            "study_contract": self.study_contract.document()})


def parse_functional_constructs(document) -> FunctionalConstructs:
    """Strictly build a :class:`FunctionalConstructs` from a decoded JSON object."""
    exact_keys(document, ("schema", "pairs_declaration_digest", "study_contract", "sources",
                          "constructs", "groups"), "functional constructs")
    if document["schema"] != CONSTRUCTS_SCHEMA:
        raise ValueError("unsupported functional constructs schema")
    contract = exact_keys(document["study_contract"],
                          ("target_property", "condition_scope", "compatible_endpoints",
                           "assertion"), "study_contract")
    prop = exact_keys(contract["target_property"], ("property_id", "description"),
                      "target_property")
    scope = exact_keys(contract["condition_scope"], ("scope_id", "description", "context_ids"),
                       "condition_scope")
    study = StudyContract(
        property_id=prop["property_id"], property_description=prop["description"],
        scope_id=scope["scope_id"], scope_description=scope["description"],
        context_ids=_listed(scope["context_ids"], "condition_scope.context_ids"),
        compatible_endpoints=_rows(contract["compatible_endpoints"],
                                   ("semantics", "units", "direction", "justification"),
                                   "compatible_endpoints", CompatibleEndpoint),
        assertion=contract["assertion"])
    sources = _rows(document["sources"], ("source_id", "description"), "sources",
                    FunctionalSource)
    constructs = _rows(document["constructs"],
                       ("construct_id", "source_id", "source_record_id", "H", "L", "Ag",
                        "antibody_format", "antibody_family", "target_family", "partition",
                        "evidence_reference"), "constructs", FunctionalConstruct)
    groups = []
    for row in _listed(document["groups"], "groups"):
        exact_keys(row, ("group_id", "context_id", "endpoint", "antigen_sha256",
                         "member_construct_ids", "evidence_reference"), "groups row")
        endpoint = Endpoint(**exact_keys(row["endpoint"], ("semantics", "units", "direction"),
                                         "group endpoint"))
        groups.append(FunctionalGroup(
            group_id=row["group_id"], context_id=row["context_id"], endpoint=endpoint,
            antigen_sha256=row["antigen_sha256"],
            member_construct_ids=_listed(row["member_construct_ids"], "member_construct_ids"),
            evidence_reference=row["evidence_reference"]))
    return FunctionalConstructs(pairs_declaration_digest=document["pairs_declaration_digest"],
                                study_contract=study, sources=sources, constructs=constructs,
                                groups=tuple(groups))


def load_functional_constructs(path) -> FunctionalConstructs:
    return parse_functional_constructs(read_strict_json(path))


# ---------------------------------------------------------------------------
# Qualification against the broad manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualifiedFunctional:
    """A functional document bound to a declaration, a population and the broad input rows.

    ``digest`` binds the canonical document, the declaration, the population, the digest
    of ALL broad input rows (including excluded ones) and the shared-train list.
    ``contract_digest`` binds only the study contract; it enters run identities and never
    a tensor.
    """
    document: FunctionalConstructs
    declaration_digest: str
    population_digest: str
    broad_input_digest: str
    contract_digest: str
    digest: str
    shared_with_denoising_train: Tuple[str, ...]

    def train_constructs(self) -> Tuple[FunctionalConstruct, ...]:
        return tuple(c for c in self.document.constructs if c.partition == "train")

    def train_groups(self) -> Tuple[FunctionalGroup, ...]:
        return tuple(g for g in self.document.groups
                     if self.document.group_partition(g.group_id) == "train")

    def train_observations(self) -> Tuple[Tuple[str, str], ...]:
        """Actual observations are ``(group_id, construct_id)``, sorted."""
        return tuple(sorted((g.group_id, member) for g in self.train_groups()
                            for member in g.member_construct_ids))

    def construct(self, construct_id: str) -> FunctionalConstruct:
        return self.document.construct(construct_id)

    def group(self, group_id: str) -> FunctionalGroup:
        return self.document.group(group_id)

    def direction(self, group_id: str) -> str:
        return self.document.group(group_id).endpoint.direction

    def report(self) -> Dict[str, Any]:
        """Counts that keep repeats from masquerading as independent antibodies."""
        doc = self.document
        constructs = doc.constructs
        observations = [(g.group_id, m) for g in doc.groups for m in g.member_construct_ids]
        by_partition = {p: {"constructs": 0, "groups": 0, "observations": 0} for p in PARTITIONS}
        for c in constructs:
            by_partition[c.partition]["constructs"] += 1
        for g in doc.groups:
            partition = doc.group_partition(g.group_id)
            by_partition[partition]["groups"] += 1
            by_partition[partition]["observations"] += len(g.member_construct_ids)
        return {
            "schema": "trastuzumab-functional-qualified-report/1",
            "digest": self.digest, "contract_digest": self.contract_digest,
            "by_partition": by_partition,
            "unique_observations": len(set(observations)),
            "unique_construct_ids": len({m for _, m in observations}),
            "unique_physical_constructs": len({doc.construct(m).content_key for _, m in observations}),
            "unique_antibodies": len({(doc.construct(m).H, doc.construct(m).L)
                                      for _, m in observations}),
            "unique_antigens": len({doc.construct(m).Ag for _, m in observations}),
            "shared_with_denoising_train": list(self.shared_with_denoising_train),
        }


def qualify_functional(document, declaration, broad_records, *, population=None) -> QualifiedFunctional:
    """Bind a functional document to the broad declaration and ALL outcome-free broad rows.

    ``broad_records`` is the original ``InputRecord`` sequence (retained and excluded
    rows alike).  It is re-qualified under ``declaration``; if ``population`` is given it
    must equal that result exactly, otherwise the derived population is used.
    """
    if not isinstance(document, FunctionalConstructs):
        raise TypeError("document must be a FunctionalConstructs")
    if not isinstance(declaration, ManifestDeclaration):
        raise TypeError("declaration must be a ManifestDeclaration")
    broad = tuple(broad_records)
    if any(not isinstance(r, InputRecord) for r in broad):
        raise TypeError("broad_records must be InputRecord instances")
    derived = qualify_manifest(declaration, broad)
    if population is not None:
        if not isinstance(population, QualifiedPopulation):
            raise TypeError("population must be a QualifiedPopulation")
        if population != derived:
            raise ValueError("supplied population is not the qualification of the supplied broad "
                             "records under the supplied declaration")
    population = derived
    declaration_digest = _digest_manifest(declaration)
    if population.manifest_digest != declaration_digest:
        raise ValueError("population manifest digest does not match the declaration")
    if document.pairs_declaration_digest != declaration_digest:
        raise ValueError("pairs_declaration_digest does not match the supplied declaration")

    excluded_content = set(declaration.excluded_content_keys)
    excluded_antibody = set(declaration.excluded_antibody_families)
    excluded_target = set(declaration.excluded_target_families)
    for c in document.constructs:
        name = c.construct_id
        if c.content_key in excluded_content:
            raise ValueError(f"functional construct {name!r} has an excluded content key")
        if c.antibody_family in excluded_antibody:
            raise ValueError(f"functional construct {name!r} has an excluded antibody family")
        if c.target_family in excluded_target:
            raise ValueError(f"functional construct {name!r} has an excluded target family")
        if len(c.H) + len(c.L or "") + len(c.Ag) + 2 > declaration.encoder_max_tokens:
            raise ValueError(f"functional construct {name!r} exceeds the joint encoder limit "
                             "(paired view, checked for both arms)")
        if len(c.H) + len(c.L or "") + (5 if c.L else 3) > declaration.native_max_tokens:
            raise ValueError(f"functional construct {name!r} exceeds the native token limit")

    # Every broad row takes part, including rows the declaration excluded.
    broad_content, broad_ab, broad_ag, broad_source_records = {}, {}, {}, {}
    axes = {axis: defaultdict(set) for axis in declaration.grouping.separation_axes}
    for r in broad:
        key = _content_key(r.H, r.L, r.Ag)
        broad_content[key] = (r.antibody_format, r.antibody_family, r.target_family, r.partition)
        broad_ab[(r.H, r.L)] = r.antibody_family
        broad_ag[r.Ag] = r.target_family
        broad_source_records[(r.source_id, r.source_record_id)] = key
        for axis, families in axes.items():
            families[getattr(r, axis)].add(r.partition)

    declared_sources = {s.source_id: s.description for s in declaration.sources}
    for s in document.sources:
        if s.source_id in declared_sources and declared_sources[s.source_id] != s.description:
            raise ValueError(f"source {s.source_id!r} is declared in the broad manifest with a "
                             "different description")
    for c in document.constructs:
        name = c.construct_id
        mine = (c.antibody_format, c.antibody_family, c.target_family, c.partition)
        theirs = broad_content.get(c.content_key)
        if theirs is not None and theirs != mine:
            fields = ("antibody_format", "antibody_family", "target_family", "partition")
            differing = [f for f, a, b in zip(fields, mine, theirs) if a != b]
            raise ValueError(f"functional construct {name!r} disagrees with the broad manifest "
                             f"on shared content: {differing}")
        family = broad_ab.get((c.H, c.L))
        if family is not None and family != c.antibody_family:
            raise ValueError(f"functional construct {name!r} antibody family conflicts with the "
                             "broad manifest for the same (H, L)")
        family = broad_ag.get(c.Ag)
        if family is not None and family != c.target_family:
            raise ValueError(f"functional construct {name!r} target family conflicts with the "
                             "broad manifest for the same antigen")
        if c.source_id in declared_sources:
            prior = broad_source_records.get((c.source_id, c.source_record_id))
            if prior is not None and prior != c.content_key:
                raise ValueError(f"functional construct {name!r} reuses a broad source record "
                                 "identity for different content")
        for axis, families in axes.items():
            families[getattr(c, axis)].add(c.partition)
    for axis, families in axes.items():
        for family, partitions in families.items():
            if len(partitions) > 1:
                raise ValueError(f"separation axis {axis} family {family!r} spans partitions "
                                 "across the broad and functional documents")

    train_keys = {r.content_key for r in population.train_records()}
    shared = tuple(sorted(c.construct_id for c in document.constructs
                          if c.partition == "train" and c.content_key in train_keys))
    broad_digest = json_digest(sorted((asdict(r) for r in broad), key=lambda r: r["record_id"]))
    contract_digest = document.contract_digest()
    digest = json_digest({"schema": QUALIFIED_SCHEMA, "document": document.document_digest(),
                          "declaration": declaration_digest,
                          "population": population.population_digest,
                          "broad_input": broad_digest, "contract": contract_digest,
                          "shared_with_denoising_train": list(shared)})
    return QualifiedFunctional(document=document, declaration_digest=declaration_digest,
                               population_digest=population.population_digest,
                               broad_input_digest=broad_digest, contract_digest=contract_digest,
                               digest=digest, shared_with_denoising_train=shared)
