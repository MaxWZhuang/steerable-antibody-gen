"""Explicit exact-pKD panel: closed record/manifest schemas, split audit and support counts.

Pure stdlib plus ``models.antibody_antigen.views`` (and the sibling ``config`` helpers), so
admission and help never need torch, numpy or scipy.

Everything is declared by the user: the manifest names one records file and its byte hash,
the sources and assays that produced the numbers, and the split regime and basis. Nothing is
scanned, discovered or converted. ``measured`` is a DECLARED flag, not a verified one. A schema
violation anywhere rejects the whole manifest; no row is dropped silently.

Identifiers (``record_id``, groups, ``source``, ``assay``) are audit, split and metric metadata.
They never become features, selectors, conditioning inputs or mask seeds, and the order of rows
in the file never determines anything downstream: the panel is sorted by partition and sequence
fingerprints, then source and assay (not by record id).

Split checks here are exact-string and declared-group checks only. They do NOT establish
homology separation or exclusion from any upstream pretraining set; the report says so.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from smallAntibodyGen.experiments.interaction_qualification.config import (
    PARTITIONS,
    SchemaError,
    decode_utf8,
    expect_id,
    expect_object,
    expect_real,
    expect_text,
    read_file_bytes,
    strict_json_loads,
)
from smallAntibodyGen.models.antibody_antigen.views import ROLES, Coord, validate_coord, validate_residue

RECORD_SCHEMA = "aa-record/1"
MANIFEST_SCHEMA = "aa-manifest/1"
MEASUREMENT_KIND = "exact_pkd"
SPLIT_REGIMES: Tuple[str, ...] = ("antibody_holdout", "target_holdout", "both")
_SHA256 = re.compile(r"[0-9a-f]{64}")

WEIGHTING = (
    "every admitted record carries weight 1; distinct sources/assays that measured the same pair in one "
    "partition are separate records and each counts once"
)
SPECIFICITY_NOTE = (
    "no specificity claim for a partition/source/assay group without a complete 2x2 antibody-by-antigen block"
)


# --- identities -----------------------------------------------------------------------------


def _chain(role: str, sequence: Optional[str]) -> str:
    """Length-framed chain text; an absent chain is spelled out, never an empty string."""
    return f"{role}:absent" if sequence is None else f"{role}:{len(sequence)}:{sequence}"


def _digest(tag: str, *parts: str) -> str:
    return hashlib.sha256("|".join((tag,) + parts).encode("ascii")).hexdigest()


def antibody_fingerprint(heavy: str, light: Optional[str]) -> str:
    """Hash of the framed (H, L) chains of validated canonical sequences; ``light=None`` is absent-L."""
    return _digest("ab1", _chain("H", heavy), _chain("L", light))


def antigen_fingerprint(antigen: str) -> str:
    return _digest("ag1", _chain("Ag", antigen))


def pair_fingerprint(heavy: str, light: Optional[str], antigen: str) -> str:
    return _digest("pair1", _chain("H", heavy), _chain("L", light), _chain("Ag", antigen))


def _sequence(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{where} must be a non-empty string of canonical uppercase residues, got {value!r}")
    for residue in value:
        try:
            validate_residue(residue)
        except ValueError:
            raise SchemaError(
                f"{where} contains {residue!r}, which is not one of the 20 canonical uppercase residues"
            ) from None
    return value


def _check_region(region: object, heavy: str, light: Optional[str]) -> None:
    if region is None:
        return
    if type(region) is not tuple:
        raise SchemaError(f"region must be a tuple of (role, index) coordinates, got {type(region).__name__}")
    if not region:
        raise SchemaError("a supplied region must be non-empty")
    lengths = {"H": len(heavy), "L": 0 if light is None else len(light)}
    keys: List[Tuple[int, int]] = []
    for coord in region:
        try:
            role, index = validate_coord(coord)
        except ValueError as exc:
            raise SchemaError(f"region: {exc}") from None
        if role == "Ag":
            raise SchemaError("region: the antigen is never masked")
        if index >= lengths[role]:
            raise SchemaError(f"region: {coord!r} is outside chain {role} of length {lengths[role]}")
        keys.append((ROLES.index(role), index))
    for before, after in zip(keys, keys[1:]):
        if before == after:
            raise SchemaError("region repeats a coordinate")
        if before > after:
            raise SchemaError("region must be sorted heavy-before-light, ascending within a chain")


@dataclass(frozen=True)
class Record:
    """One declared exact-pKD observation. Validated on construction; immutable.

    ``antibody_fingerprint`` / ``antigen_fingerprint`` / ``pair_fingerprint`` identify sequences
    (never ids); ``content_sha256`` binds every evaluated value including ``record_id``.
    """

    record_id: str
    heavy: str
    light: Optional[str]
    antigen: str
    antibody_group: str
    antigen_group: str
    partition: str
    source: str
    assay: str
    pkd: float
    region: Optional[Tuple[Coord, ...]] = None
    antibody_fingerprint: str = field(init=False, repr=False, compare=False)
    antigen_fingerprint: str = field(init=False, repr=False, compare=False)
    pair_fingerprint: str = field(init=False, repr=False, compare=False)
    content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("record_id", "antibody_group", "antigen_group", "source", "assay"):
            expect_id(getattr(self, name), name)
        _sequence(self.heavy, "heavy")
        if self.light is not None:
            _sequence(self.light, "light")
        _sequence(self.antigen, "antigen")
        if not isinstance(self.partition, str) or self.partition not in PARTITIONS:
            raise SchemaError(f"partition must be one of {PARTITIONS}, got {self.partition!r}")
        object.__setattr__(self, "pkd", expect_real(self.pkd, "pkd"))
        _check_region(self.region, self.heavy, self.light)
        object.__setattr__(self, "antibody_fingerprint", antibody_fingerprint(self.heavy, self.light))
        object.__setattr__(self, "antigen_fingerprint", antigen_fingerprint(self.antigen))
        object.__setattr__(self, "pair_fingerprint", pair_fingerprint(self.heavy, self.light, self.antigen))
        text = json.dumps(self.to_row(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        object.__setattr__(self, "content_sha256", hashlib.sha256(text.encode("ascii")).hexdigest())

    def to_row(self) -> Dict[str, Any]:
        """The record in its on-disk (``aa-record/1``) shape; ``parse_record`` reads it back."""
        return {
            "schema": RECORD_SCHEMA,
            "record_id": self.record_id,
            "heavy": self.heavy,
            "light": self.light,
            "antigen": self.antigen,
            "antibody_group": self.antibody_group,
            "antigen_group": self.antigen_group,
            "partition": self.partition,
            "source": self.source,
            "assay": self.assay,
            "measurement": {"kind": MEASUREMENT_KIND, "value": self.pkd, "measured": True},
            "region": None if self.region is None else [[role, index] for role, index in self.region],
        }


_RECORD_KEYS = (
    "schema",
    "record_id",
    "heavy",
    "light",
    "antigen",
    "antibody_group",
    "antigen_group",
    "partition",
    "source",
    "assay",
    "measurement",
)
_MEASUREMENT_KEYS = ("kind", "value", "measured")


def _parse_region(value: object) -> Optional[Tuple[Coord, ...]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise SchemaError(f"region must be null, absent or a list of [role, index] pairs, got {value!r}")
    coords = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise SchemaError(f"region entries must be [role, index] pairs, got {item!r}")
        coords.append((item[0], item[1]))
    return tuple(coords)


def parse_record(value: object) -> Record:
    """One decoded JSON row -> ``Record``; any deviation from ``aa-record/1`` raises ``SchemaError``."""
    row = expect_object(value, "record", _RECORD_KEYS, optional=("region",))
    if row["schema"] != RECORD_SCHEMA:
        raise SchemaError(f"record schema must be {RECORD_SCHEMA!r}, got {row['schema']!r}")
    measurement = expect_object(row["measurement"], "measurement", _MEASUREMENT_KEYS)
    if measurement["kind"] != MEASUREMENT_KIND:
        raise SchemaError(f"measurement kind must be {MEASUREMENT_KIND!r}, got {measurement['kind']!r}")
    if measurement["measured"] is not True:
        raise SchemaError(f"measurement.measured must be the JSON literal true, got {measurement['measured']!r}")
    return Record(
        record_id=row["record_id"],
        heavy=row["heavy"],
        light=row["light"],
        antigen=row["antigen"],
        antibody_group=row["antibody_group"],
        antigen_group=row["antigen_group"],
        partition=row["partition"],
        source=row["source"],
        assay=row["assay"],
        pkd=expect_real(measurement["value"], "measurement.value"),
        region=_parse_region(row.get("region")),
    )


def parse_records(data: bytes) -> Tuple[Record, ...]:
    """Strict JSONL (one object per ``\\n``-terminated line) in file order. An empty file is an empty panel."""
    lines = decode_utf8(data, "records file").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            if not line.strip():
                raise SchemaError("blank line")
            records.append(parse_record(strict_json_loads(line)))
        except SchemaError as exc:
            raise SchemaError(f"records line {number}: {exc}") from None
    return tuple(records)


# --- manifest -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDeclaration:
    source: str
    citation: str
    assays: Tuple[Tuple[str, str], ...]  # (assay, declared description), sorted by assay


@dataclass(frozen=True)
class Manifest:
    records_file: str
    records_sha256: str
    sources: Tuple[SourceDeclaration, ...]  # sorted by source
    split_regime: str
    split_basis: str

    def declared_assays(self) -> Dict[Tuple[str, str], str]:
        return {(s.source, assay): text for s in self.sources for assay, text in s.assays}


_MANIFEST_KEYS = ("schema", "records_file", "records_sha256", "sources", "split_regime", "split_basis")


def _parse_sources(value: object) -> Tuple[SourceDeclaration, ...]:
    if not isinstance(value, dict):
        raise SchemaError(f"sources must be a JSON object, got {type(value).__name__}")
    out = []
    for source in sorted(value):
        expect_id(source, "source name")
        entry = expect_object(value[source], f"sources[{source!r}]", ("citation", "assays"))
        citation = expect_text(entry["citation"], f"sources[{source!r}].citation")
        assays = entry["assays"]
        if not isinstance(assays, dict) or not assays:
            raise SchemaError(f"sources[{source!r}].assays must be a non-empty JSON object")
        declared = []
        for assay in sorted(assays):
            expect_id(assay, f"sources[{source!r}] assay name")
            declared.append((assay, expect_text(assays[assay], f"sources[{source!r}].assays[{assay!r}]")))
        out.append(SourceDeclaration(source=source, citation=citation, assays=tuple(declared)))
    return tuple(out)


def parse_manifest(text: str) -> Manifest:
    root = expect_object(strict_json_loads(text), "manifest", _MANIFEST_KEYS)
    if root["schema"] != MANIFEST_SCHEMA:
        raise SchemaError(f"manifest schema must be {MANIFEST_SCHEMA!r}, got {root['schema']!r}")
    digest = root["records_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise SchemaError("records_sha256 must be 64 lowercase hex characters")
    regime = root["split_regime"]
    if not isinstance(regime, str) or regime not in SPLIT_REGIMES:
        raise SchemaError(f"split_regime must be one of {SPLIT_REGIMES}, got {regime!r}")
    return Manifest(
        records_file=expect_text(root["records_file"], "records_file"),
        records_sha256=digest,
        sources=_parse_sources(root["sources"]),
        split_regime=regime,
        split_basis=expect_text(root["split_basis"], "split_basis"),
    )


# --- panel ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """Admitted records in deterministic order plus what pinned them.

    ``records_file_sha256`` is the byte hash of the file; ``content_sha256`` is an order-independent
    hash of the normalized records (and the regime), so whitespace or row order cannot change it.
    """

    manifest: Manifest
    records_file_sha256: str
    records: Tuple[Record, ...]
    content_sha256: str

    def partition_records(self, partition: str) -> Tuple[Record, ...]:
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition {partition!r}; partitions are {PARTITIONS}")
        return tuple(r for r in self.records if r.partition == partition)

    def by_id(self) -> Dict[str, Record]:
        return {r.record_id: r for r in self.records}


def _order_key(record: Record) -> Tuple[int, str, str, str, str]:
    return (
        PARTITIONS.index(record.partition),
        record.antibody_fingerprint,
        record.antigen_fingerprint,
        record.source,
        record.assay,
    )


def _check_unique_ids(records: Sequence[Record]) -> None:
    seen: Set[str] = set()
    for record in records:
        if record.record_id in seen:
            raise SchemaError(f"duplicate record_id {record.record_id!r}")
        seen.add(record.record_id)


def _check_no_duplicate_measurements(records: Sequence[Record]) -> None:
    first: Dict[Tuple[str, str, str], str] = {}
    for record in records:
        key = (record.pair_fingerprint, record.source, record.assay)
        if key in first:
            raise SchemaError(
                f"duplicate measurement: records {first[key]!r} and {record.record_id!r} share the same "
                f"(heavy, light, antigen) pair, source and assay"
            )
        first[key] = record.record_id


def _check_provenance(manifest: Manifest, records: Sequence[Record]) -> None:
    declared = manifest.declared_assays()
    missing = sorted({(r.source, r.assay) for r in records if (r.source, r.assay) not in declared})
    if missing:
        raise SchemaError(f"no provenance declaration in the manifest for (source, assay) {missing}")


def _check_alias_consistency(records: Sequence[Record]) -> None:
    axes = (
        ("antibody", lambda r: r.antibody_fingerprint, lambda r: r.antibody_group),
        ("antigen", lambda r: r.antigen_fingerprint, lambda r: r.antigen_group),
    )
    for label, fingerprint_of, group_of in axes:
        first: Dict[str, Tuple[str, str]] = {}
        for record in records:
            group, record_id = group_of(record), record.record_id
            seen_group, seen_id = first.setdefault(fingerprint_of(record), (group, record_id))
            if seen_group != group:
                raise SchemaError(
                    f"the same {label} sequence is declared under groups {seen_group!r} (record {seen_id!r}) "
                    f"and {group!r} (record {record_id!r}); one sequence maps to one group"
                )


def _require_single_partition(
    records: Sequence[Record], key_of: Callable[[Record], str], label: str, *, show_key: bool = False
) -> None:
    where: Dict[str, Dict[str, str]] = {}
    for record in records:
        where.setdefault(key_of(record), {}).setdefault(record.partition, record.record_id)
    leaks = sorted(key for key, partitions in where.items() if len(partitions) > 1)
    if leaks:
        partitions = where[leaks[0]]
        detail = ", ".join(f"{p} (record {partitions[p]!r})" for p in PARTITIONS if p in partitions)
        shown = f" {leaks[0]!r}" if show_key else ""
        raise SchemaError(f"{label}{shown} is shared across partitions: {detail}; {len(leaks)} such value(s)")


def _check_split(regime: str, records: Sequence[Record]) -> None:
    _require_single_partition(records, lambda r: r.pair_fingerprint, "exact pair (heavy, light, antigen)")
    if regime in ("antibody_holdout", "both"):
        _require_single_partition(records, lambda r: r.antibody_fingerprint, "antibody sequence (heavy, light)")
        _require_single_partition(records, lambda r: r.antibody_group, "declared antibody_group", show_key=True)
    if regime in ("target_holdout", "both"):
        _require_single_partition(records, lambda r: r.antigen_fingerprint, "antigen sequence")
        _require_single_partition(records, lambda r: r.antigen_group, "declared antigen_group", show_key=True)


def build_panel(manifest: Manifest, records: Sequence[Record], records_file_sha256: str) -> Panel:
    """Cross-record checks, then a deterministic order. Raises ``SchemaError`` on the first violation."""
    if records_file_sha256 != manifest.records_sha256:
        raise SchemaError("records file hash does not match the manifest records_sha256")
    records = tuple(records)
    for record in records:
        if not isinstance(record, Record):
            raise TypeError(f"expected Record objects, got {type(record).__name__}")
    _check_unique_ids(records)
    _check_no_duplicate_measurements(records)
    _check_provenance(manifest, records)
    _check_alias_consistency(records)
    _check_split(manifest.split_regime, records)
    ordered = tuple(sorted(records, key=_order_key))
    text = "|".join(("aa-panel/1", manifest.split_regime) + tuple(sorted(r.content_sha256 for r in records)))
    return Panel(
        manifest=manifest,
        records_file_sha256=records_file_sha256,
        records=ordered,
        content_sha256=hashlib.sha256(text.encode("ascii")).hexdigest(),
    )


def load_manifest(path: Union[str, Path]) -> Panel:
    """Read one manifest, verify the declared records file's byte hash BEFORE parsing it, build the panel.

    Only ``records_file`` is read: relative to the manifest's directory, or absolute. Nothing is scanned.
    """
    manifest_path = Path(path)
    manifest = parse_manifest(decode_utf8(read_file_bytes(manifest_path, "manifest"), "manifest"))
    records_path = Path(manifest.records_file)
    if not records_path.is_absolute():
        records_path = manifest_path.parent / records_path
    data = read_file_bytes(records_path, "records file")
    actual = hashlib.sha256(data).hexdigest()
    if actual != manifest.records_sha256:
        raise SchemaError(
            f"records file hash {actual} does not match the manifest records_sha256 {manifest.records_sha256}"
        )
    return build_panel(manifest, parse_records(data), actual)


# --- support counts -------------------------------------------------------------------------


def complete_2x2_blocks(cells: Iterable[Tuple[str, str]]) -> int:
    """Number of (antibody pair) x (antigen pair) blocks whose four (antibody, antigen) cells are all present."""
    by_antibody: Dict[str, Set[str]] = {}
    by_antigen: Dict[str, Set[str]] = {}
    for antibody, antigen in cells:
        by_antibody.setdefault(antibody, set()).add(antigen)
        by_antigen.setdefault(antigen, set()).add(antibody)

    def cost(adjacency: Mapping[str, Set[str]]) -> int:
        return sum(len(v) * (len(v) - 1) // 2 for v in adjacency.values())

    adjacency = by_antibody if cost(by_antibody) <= cost(by_antigen) else by_antigen
    shared: Counter = Counter()
    for neighbours in adjacency.values():
        shared.update(itertools.combinations(sorted(neighbours), 2))
    return sum(count * (count - 1) // 2 for count in shared.values())


def _cell_counts(records: Sequence[Record]) -> Dict[str, int]:
    return {
        "n_records": len(records),
        "unique_antibodies": len({r.antibody_fingerprint for r in records}),
        "unique_antigens": len({r.antigen_fingerprint for r in records}),
        "unique_pairs": len({r.pair_fingerprint for r in records}),
        "missing_region": sum(1 for r in records if r.region is None),
    }


def support_counts(records: Sequence[Record]) -> Dict[str, Any]:
    """Counts by partition and by (partition, source, assay), with complete 2x2 blocks per group."""
    groups: Dict[Tuple[str, str, str], List[Record]] = {}
    for record in records:
        groups.setdefault((record.partition, record.source, record.assay), []).append(record)
    entries = []
    for key in sorted(groups, key=lambda k: (PARTITIONS.index(k[0]), k[1], k[2])):
        members = groups[key]
        entry: Dict[str, Any] = {"partition": key[0], "source": key[1], "assay": key[2]}
        entry.update(_cell_counts(members))
        entry["complete_2x2_blocks"] = complete_2x2_blocks(
            (r.antibody_fingerprint, r.antigen_fingerprint) for r in members
        )
        entries.append(entry)
    return {
        "overall": _cell_counts(records),
        "declared_measured": len(records),
        "by_partition": {p: _cell_counts([r for r in records if r.partition == p]) for p in PARTITIONS},
        "groups": entries,
    }


def support_report(panel: Panel) -> Dict[str, Any]:
    """JSON-serializable admission report: what is enforced, what is declared, what is NOT audited."""
    regime = panel.manifest.split_regime
    antibody_axis = "enforced" if regime in ("antibody_holdout", "both") else "not_required_by_regime"
    antigen_axis = "enforced" if regime in ("target_holdout", "both") else "not_required_by_regime"
    used = {(r.source, r.assay) for r in panel.records}
    unused = [
        {"source": source, "assay": assay}
        for (source, assay) in sorted(panel.manifest.declared_assays())
        if (source, assay) not in used
    ]
    return {
        "schema": "aa-support/1",
        "records_file_sha256": panel.records_file_sha256,
        "content_sha256": panel.content_sha256,
        "no_support": len(panel.records) == 0,
        "scientific_qualification": "not_assessed",
        "split_regime": regime,
        "split_basis": panel.manifest.split_basis,
        "split_basis_status": "declared",
        "split_checks": {
            "exact_pair_disjoint_across_partitions": "enforced",
            "antibody_sequence_and_group_disjoint": antibody_axis,
            "antigen_sequence_and_group_disjoint": antigen_axis,
            "homology_separation": "not_audited",
            "upstream_pretraining_exclusion": "not_audited",
        },
        "measured_flag": "declared_not_verified",
        "weighting": WEIGHTING,
        "specificity": SPECIFICITY_NOTE,
        "support": support_counts(panel.records),
        "unused_declarations": unused,
        "records": [
            {
                "record_id": r.record_id,
                "partition": r.partition,
                "source": r.source,
                "assay": r.assay,
                "antibody_fingerprint": r.antibody_fingerprint,
                "antigen_fingerprint": r.antigen_fingerprint,
                "pair_fingerprint": r.pair_fingerprint,
                "content_sha256": r.content_sha256,
                "has_region": r.region is not None,
            }
            for r in panel.records
        ],
    }
