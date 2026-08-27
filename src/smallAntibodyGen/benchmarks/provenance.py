"""Schemas and validators for benchmark provenance (ticket J05a).

Scope
-----
This module is pure stdlib, has no I/O beyond reading local files it is handed,
and never touches the network. It defines *what a benchmark source and one assay
measurement look like* and refuses anything that does not fit. It does not parse
CR9114, AVIDa, or Open AlphaSeq releases, and it holds no evaluator logic.

The four failure modes it exists to make impossible
---------------------------------------------------
1. **An unfilled manifest passing for an approved one.** Owner-supplied values
   (release version, license, retrieval date, source URL, hashes) are carried as
   the literal sentinel ``TODO(owner)`` in the committed templates.
   :func:`validate_source_manifest` raises :class:`UnsuppliedOwnerDecisionError`
   on any sentinel anywhere in the document, so an unapproved manifest can never
   be consumed as data. :func:`parse_manifest_document` is the deliberate
   read-only door: it reports which fields are still unsupplied without pretending
   the manifest is usable.

2. **A UniProt full-length sequence silently standing in for the assayed
   construct.** The construct is a two-case tagged union with *no shared
   sequence field*: :class:`ExactAssayedConstruct` carries a sequence and an
   authoritative source; :class:`RegisteredNullConstruct` carries a reason and
   has no ``sequence`` attribute at all. A reference sequence is a third,
   unrelated type (:class:`ReferenceSequence`) that :func:`assayed_construct_sequence`
   refuses with ``TypeError`` and that :class:`AntigenProvenance` refuses as a
   construct. Substituting a reference is therefore a type error at the seam, not
   a convention someone has to remember.

3. **A nonbinder becoming a weak binder.** :class:`Measurement` is a tagged union
   over ``none / left / right / interval / non_binding``. A non-binding outcome
   carries no value; a censored outcome carries a bound, never a point estimate.
   Coercing "no binding detected" into a fake KD is not expressible.

4. **Direction normalized per consumer.** For some sources a lower number means
   a stronger binder. :func:`normalize_measurement` performs that flip exactly
   once, returning a :class:`NormalizedMeasurement` in a higher-is-stronger frame
   that retains the raw measurement and its original direction -- and that swaps
   the censoring side, because a right-censored KD is a left-censored strength.

Determinism
-----------
No timestamps are generated at runtime. Serialization goes through
:func:`dumps_manifest` (sorted keys, two-space indent, trailing newline), so a
manifest rewritten from its own parsed form is byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1"

#: The literal marker for a value only the scientific owner may supply.
#: Its presence anywhere in a manifest makes that manifest invalid.
TODO_OWNER = "TODO(owner)"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class BenchmarkProvenanceError(ValueError):
    """Base class for every rejection in this module."""


class ManifestValidationError(BenchmarkProvenanceError):
    """A source manifest is structurally invalid."""


class UnsuppliedOwnerDecisionError(ManifestValidationError):
    """A manifest still carries at least one ``TODO(owner)`` sentinel."""


class ConstructProvenanceError(BenchmarkProvenanceError):
    """An assayed-construct or reference-sequence record is invalid."""


class RegisteredNullError(BenchmarkProvenanceError):
    """A sequence was required but the construct is a registered null."""


class EpitopeProvenanceError(BenchmarkProvenanceError):
    """An epitope annotation is uncited, out of range, or wrongly numbered."""


class MeasurementValidationError(BenchmarkProvenanceError):
    """An assay, replicate, or measurement record is invalid."""


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The 20 canonical residues plus the ambiguity/rare codes that appear in
# released construct sequences. Gaps, whitespace, and stop symbols are rejected:
# a construct sequence is the exact assayed protein, not an alignment row.
_RESIDUE_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def _require_str(value: Any, label: str, error: type[BenchmarkProvenanceError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{label} must be a non-empty string, got {value!r}")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementValidationError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise MeasurementValidationError(f"{label} must be finite, got {value!r}")
    return number


def _validate_residues(sequence: Any, label: str) -> str:
    text = _require_str(sequence, label, ConstructProvenanceError)
    if text != text.strip():
        raise ConstructProvenanceError(f"{label} must not carry surrounding whitespace")
    bad = sorted(set(text) - _RESIDUE_ALPHABET)
    if bad:
        raise ConstructProvenanceError(
            f"{label} contains non-residue characters {bad!r}; a construct sequence is "
            "the exact assayed protein, not an aligned or annotated string"
        )
    return text


# ---------------------------------------------------------------------------
# 1. source manifest
# ---------------------------------------------------------------------------

MANIFEST_KEYS = (
    "candidate_source_url",
    "candidate_source_url_verified",
    "dataset_name",
    "files",
    "license",
    "notes",
    "owner_decisions",
    "plan_assertions",
    "release_version",
    "retrieval_date",
    "schema_version",
    "source_url",
)

FILE_ENTRY_KEYS = ("relative_path", "sha256", "size_bytes")

OWNER_DECISION_KEYS = ("key", "question", "status")

PLAN_ASSERTION_KEYS = ("claim", "source", "verified")


@dataclass(frozen=True)
class FileEntry:
    """One prepared file: where it sits under the raw root, how big, and its hash."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ManifestValidationError(
                f"size_bytes must be an int, got {self.size_bytes!r}"
            )
        if self.size_bytes < 0:
            raise ManifestValidationError("size_bytes must not be negative")
        if not isinstance(self.sha256, str) or not _SHA256_RE.match(self.sha256):
            raise ManifestValidationError(
                f"sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            )

    def to_json_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SourceManifest:
    """A validated, owner-approved description of one benchmark source release."""

    dataset_name: str
    release_version: str
    source_url: str
    license: str
    retrieval_date: str
    files: tuple[FileEntry, ...]
    candidate_source_url: str | None = None
    notes: str = ""

    def file_by_path(self, relative_path: str) -> FileEntry | None:
        for entry in self.files:
            if entry.relative_path == relative_path:
                return entry
        return None


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"relative_path must be a non-empty string, got {value!r}")
    if value != value.strip():
        raise ManifestValidationError("relative_path must not carry surrounding whitespace")
    if value.startswith("/") or value.endswith("/"):
        raise ManifestValidationError(f"relative_path must be relative to the raw root: {value!r}")
    if "\\" in value:
        raise ManifestValidationError(f"relative_path must use forward slashes: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ManifestValidationError(f"relative_path must not traverse or repeat: {value!r}")
    return value


def _sentinel_paths(node: Any, prefix: str = "") -> list[str]:
    """Return dotted paths of every ``TODO(owner)`` sentinel below ``node``."""
    found: list[str] = []
    if isinstance(node, str):
        if node == TODO_OWNER:
            found.append(prefix or "<root>")
    elif isinstance(node, Mapping):
        for key in sorted(node):
            child = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_sentinel_paths(node[key], child))
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            found.extend(_sentinel_paths(item, f"{prefix}[{index}]"))
    return found


def _check_key_set(document: Any, expected: Sequence[str], label: str) -> Mapping:
    if not isinstance(document, Mapping):
        raise ManifestValidationError(f"{label} must be a mapping, got {type(document).__name__}")
    present = set(document)
    wanted = set(expected)
    missing = sorted(wanted - present)
    unknown = sorted(present - wanted)
    if missing or unknown:
        raise ManifestValidationError(
            f"{label} key mismatch; missing={missing} unknown={unknown}"
        )
    return document


def _check_manifest_shape(document: Any) -> Mapping:
    """Structural checks that hold for templates and approved manifests alike."""
    doc = _check_key_set(document, MANIFEST_KEYS, "manifest")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ManifestValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {doc['schema_version']!r}"
        )
    if not isinstance(doc["files"], list):
        raise ManifestValidationError("files must be a list")
    for index, entry in enumerate(doc["files"]):
        _check_key_set(entry, FILE_ENTRY_KEYS, f"files[{index}]")
    if not isinstance(doc["owner_decisions"], list):
        raise ManifestValidationError("owner_decisions must be a list")
    for index, decision in enumerate(doc["owner_decisions"]):
        _check_key_set(decision, OWNER_DECISION_KEYS, f"owner_decisions[{index}]")
        if decision["status"] not in ("unsupplied", "supplied"):
            raise ManifestValidationError(
                f"owner_decisions[{index}].status must be 'unsupplied' or 'supplied'"
            )
    if not isinstance(doc["plan_assertions"], list):
        raise ManifestValidationError("plan_assertions must be a list")
    for index, assertion in enumerate(doc["plan_assertions"]):
        _check_key_set(assertion, PLAN_ASSERTION_KEYS, f"plan_assertions[{index}]")
        if assertion["verified"] is not False:
            raise ManifestValidationError(
                f"plan_assertions[{index}].verified must be false; nothing in this "
                "repository has verified a claim copied out of the plan"
            )
    if not isinstance(doc["candidate_source_url_verified"], bool):
        raise ManifestValidationError("candidate_source_url_verified must be a boolean")
    if doc["candidate_source_url_verified"] is not False:
        raise ManifestValidationError(
            "candidate_source_url_verified must be false; a URL the owner has verified "
            "is promoted to source_url rather than marked verified in place"
        )
    if doc["candidate_source_url"] is not None and not isinstance(doc["candidate_source_url"], str):
        raise ManifestValidationError("candidate_source_url must be a string or null")
    if not isinstance(doc["notes"], str):
        raise ManifestValidationError("notes must be a string")
    return doc


@dataclass(frozen=True)
class ManifestDocument:
    """A parsed manifest that may still be waiting on the owner.

    ``is_approved`` is false whenever any ``TODO(owner)`` sentinel survives.
    :meth:`validated` is the only way to obtain a :class:`SourceManifest`, and it
    raises for an unapproved document.
    """

    raw: Mapping
    unsupplied_fields: tuple[str, ...]
    source_path: Path | None = None

    @property
    def dataset_name(self) -> str:
        return self.raw["dataset_name"]

    @property
    def is_approved(self) -> bool:
        return not self.unsupplied_fields

    def validated(self) -> SourceManifest:
        return validate_source_manifest(self.raw)


def parse_manifest_document(document: Any, *, source_path: Path | None = None) -> ManifestDocument:
    """Parse a manifest structurally, reporting -- not repairing -- unfilled fields."""
    doc = _check_manifest_shape(document)
    return ManifestDocument(
        raw=doc,
        unsupplied_fields=tuple(_sentinel_paths(doc)),
        source_path=source_path,
    )


def load_manifest_document(path: Path | str) -> ManifestDocument:
    """Read one tracked manifest JSON file. Local file read only; never a download."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    return parse_manifest_document(document, source_path=path)


def validate_source_manifest(document: Any) -> SourceManifest:
    """Validate strictly and return a :class:`SourceManifest`, or raise.

    Rejects rather than repairs: no trimming, no case folding, no defaulting.
    """
    doc = _check_manifest_shape(document)

    sentinels = _sentinel_paths(doc)
    if sentinels:
        raise UnsuppliedOwnerDecisionError(
            "manifest still carries owner-supplied placeholders: " + ", ".join(sentinels)
        )

    pending = [d["key"] for d in doc["owner_decisions"] if d["status"] == "unsupplied"]
    if pending:
        raise UnsuppliedOwnerDecisionError(
            "manifest lists unsupplied owner decisions: " + ", ".join(sorted(pending))
        )

    for key in ("dataset_name", "release_version", "source_url", "license"):
        _require_str(doc[key], key, ManifestValidationError)
        if doc[key] != doc[key].strip():
            raise ManifestValidationError(f"{key} must not carry surrounding whitespace")

    retrieval_date = doc["retrieval_date"]
    if not isinstance(retrieval_date, str) or not _ISO_DATE_RE.match(retrieval_date):
        raise ManifestValidationError(
            f"retrieval_date must be an ISO YYYY-MM-DD string, got {retrieval_date!r}"
        )
    year, month, day = (int(part) for part in retrieval_date.split("-"))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ManifestValidationError(f"retrieval_date is not a real date: {retrieval_date!r}")

    if not doc["files"]:
        raise ManifestValidationError(
            "an approved manifest must list at least one file; an empty list means "
            "nobody has computed the hashes yet"
        )

    entries = tuple(
        FileEntry(
            relative_path=entry["relative_path"],
            size_bytes=entry["size_bytes"],
            sha256=entry["sha256"],
        )
        for entry in doc["files"]
    )
    paths = [entry.relative_path for entry in entries]
    if len(set(paths)) != len(paths):
        duplicates = sorted({p for p in paths if paths.count(p) > 1})
        raise ManifestValidationError(f"duplicate relative_path entries: {duplicates}")

    return SourceManifest(
        dataset_name=doc["dataset_name"],
        release_version=doc["release_version"],
        source_url=doc["source_url"],
        license=doc["license"],
        retrieval_date=retrieval_date,
        files=entries,
        candidate_source_url=doc["candidate_source_url"],
        notes=doc["notes"],
    )


def dumps_manifest(document: Any) -> str:
    """Canonical JSON text: sorted keys, two-space indent, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# hashing helpers (local files only)
# ---------------------------------------------------------------------------

def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file's SHA-256 in chunks; never read a corpus fully into RAM."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_entry_for(root: Path | str, path: Path | str) -> FileEntry:
    """Build a :class:`FileEntry` for ``path`` expressed relative to ``root``."""
    root = Path(root)
    path = Path(path)
    relative = path.relative_to(root).as_posix()
    return FileEntry(
        relative_path=relative,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def verify_manifest_files(manifest: SourceManifest, root: Path | str) -> tuple[str, ...]:
    """Return one human-readable problem string per mismatched or missing file."""
    root = Path(root)
    problems: list[str] = []
    for entry in manifest.files:
        target = root / entry.relative_path
        if not target.is_file():
            problems.append(f"{entry.relative_path}: missing under {root}")
            continue
        size = target.stat().st_size
        if size != entry.size_bytes:
            problems.append(
                f"{entry.relative_path}: size_bytes {size} != manifest {entry.size_bytes}"
            )
        actual = sha256_file(target)
        if actual != entry.sha256:
            problems.append(f"{entry.relative_path}: sha256 {actual} != manifest {entry.sha256}")
    return tuple(problems)


# ---------------------------------------------------------------------------
# 2. assayed construct provenance
# ---------------------------------------------------------------------------

CONSTRUCT_KIND_EXACT = "exact_assayed_construct"
CONSTRUCT_KIND_REGISTERED_NULL = "registered_null_construct"

CONSTRUCT_SOURCE_RELEASE_SUPPLEMENTARY_FILE = "release_supplementary_file"
CONSTRUCT_SOURCE_RELEASE_DATASET_COLUMN = "release_dataset_column"
CONSTRUCT_SOURCE_RELEASE_GENBANK_RECORD = "release_genbank_record"
CONSTRUCT_SOURCE_PDB_ENTRY = "pdb_entry"

#: Only an artifact of the release itself (or a deposited structure of the assayed
#: construct) may define the exact assayed sequence.
AUTHORITATIVE_CONSTRUCT_SOURCES = frozenset(
    {
        CONSTRUCT_SOURCE_RELEASE_SUPPLEMENTARY_FILE,
        CONSTRUCT_SOURCE_RELEASE_DATASET_COLUMN,
        CONSTRUCT_SOURCE_RELEASE_GENBANK_RECORD,
        CONSTRUCT_SOURCE_PDB_ENTRY,
    }
)

REFERENCE_ROLE_UNIPROT_FULL_LENGTH = "uniprot_full_length"
REFERENCE_ROLE_REFERENCE_PROTEOME = "reference_proteome"
REFERENCE_ROLE_MANUAL_RECONSTRUCTION = "manual_reconstruction"

#: Sequences that describe the antigen but are NOT what was in the assay well.
REFERENCE_SEQUENCE_ROLES = frozenset(
    {
        REFERENCE_ROLE_UNIPROT_FULL_LENGTH,
        REFERENCE_ROLE_REFERENCE_PROTEOME,
        REFERENCE_ROLE_MANUAL_RECONSTRUCTION,
    }
)

NULL_REASON_CONSTRUCT_NOT_RELEASED = "construct_not_released"
NULL_REASON_AMBIGUOUS_RELEASE = "ambiguous_release"
NULL_REASON_LICENSE_RESTRICTED = "license_restricted"
NULL_REASON_OWNER_DECISION_PENDING = "owner_decision_pending"

REGISTERED_NULL_REASONS = frozenset(
    {
        NULL_REASON_CONSTRUCT_NOT_RELEASED,
        NULL_REASON_AMBIGUOUS_RELEASE,
        NULL_REASON_LICENSE_RESTRICTED,
        NULL_REASON_OWNER_DECISION_PENDING,
    }
)

CONDITIONING_EXACT_CONSTRUCT = "exact_construct"
CONDITIONING_REGISTERED_NULL = "registered_null"


@dataclass(frozen=True)
class ExactAssayedConstruct:
    """The exact protein that was in the assay, as released by an authoritative artifact.

    ``source_kind`` is restricted to release artifacts on purpose: a UniProt
    full-length entry cannot be recorded here, so it cannot become the construct
    by a moment's inattention.
    """

    sequence: str
    source_kind: str
    source_locator: str

    kind = CONSTRUCT_KIND_EXACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _validate_residues(self.sequence, "sequence"))
        if self.source_kind not in AUTHORITATIVE_CONSTRUCT_SOURCES:
            extra = ""
            if self.source_kind in REFERENCE_SEQUENCE_ROLES:
                extra = (
                    f" -- {self.source_kind!r} is a reference sequence, not an assayed "
                    "construct; record it as a ReferenceSequence beside a "
                    "RegisteredNullConstruct instead of substituting it"
                )
            raise ConstructProvenanceError(
                f"source_kind must be one of {sorted(AUTHORITATIVE_CONSTRUCT_SOURCES)}, "
                f"got {self.source_kind!r}{extra}"
            )
        _require_str(self.source_locator, "source_locator", ConstructProvenanceError)

    def to_json_dict(self) -> dict:
        return {
            "kind": CONSTRUCT_KIND_EXACT,
            "sequence": self.sequence,
            "source_kind": self.source_kind,
            "source_locator": self.source_locator,
        }


@dataclass(frozen=True)
class RegisteredNullConstruct:
    """A *registered* absence: no construct sequence, and the reason is on the record.

    This type deliberately has no ``sequence`` field. Downstream code that wants a
    sequence must either branch on :attr:`kind` or call
    :func:`require_assayed_construct_sequence` and handle
    :class:`RegisteredNullError`. There is no attribute for a substitute to hide in.
    """

    reason_code: str
    detail: str

    kind = CONSTRUCT_KIND_REGISTERED_NULL

    def __post_init__(self) -> None:
        if self.reason_code not in REGISTERED_NULL_REASONS:
            raise ConstructProvenanceError(
                f"reason_code must be one of {sorted(REGISTERED_NULL_REASONS)}, "
                f"got {self.reason_code!r}"
            )
        _require_str(self.detail, "detail", ConstructProvenanceError)

    def to_json_dict(self) -> dict:
        return {
            "kind": CONSTRUCT_KIND_REGISTERED_NULL,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


AssayedConstruct = ExactAssayedConstruct | RegisteredNullConstruct


@dataclass(frozen=True)
class ReferenceSequence:
    """A related sequence that is explicitly NOT what was assayed.

    A UniProt full-length entry belongs here. It is a different type from
    :class:`ExactAssayedConstruct` and is reachable only under its own name, so no
    consumer asking for the assayed construct can be handed one by accident.
    """

    sequence: str
    role: str
    accession: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _validate_residues(self.sequence, "sequence"))
        if self.role not in REFERENCE_SEQUENCE_ROLES:
            raise ConstructProvenanceError(
                f"role must be one of {sorted(REFERENCE_SEQUENCE_ROLES)}, got {self.role!r}; "
                "a reference sequence may not claim to be an assayed construct"
            )
        _require_str(self.accession, "accession", ConstructProvenanceError)

    def to_json_dict(self) -> dict:
        return {"accession": self.accession, "role": self.role, "sequence": self.sequence}


def assayed_construct_sequence(construct: AssayedConstruct) -> str | None:
    """Return the assayed sequence, or ``None`` for a registered null.

    Raises ``TypeError`` for anything else -- notably a :class:`ReferenceSequence`,
    which is how a silent UniProt substitution becomes a hard failure.
    """
    if isinstance(construct, ExactAssayedConstruct):
        return construct.sequence
    if isinstance(construct, RegisteredNullConstruct):
        return None
    raise TypeError(
        "expected ExactAssayedConstruct or RegisteredNullConstruct, got "
        f"{type(construct).__name__}; a reference sequence is not an assayed construct"
    )


def require_assayed_construct_sequence(construct: AssayedConstruct) -> str:
    """Return the assayed sequence or raise :class:`RegisteredNullError`."""
    sequence = assayed_construct_sequence(construct)
    if sequence is None:
        assert isinstance(construct, RegisteredNullConstruct)
        raise RegisteredNullError(
            f"no assayed construct sequence: {construct.reason_code} ({construct.detail})"
        )
    return sequence


# ---------------------------------------------------------------------------
# 3. epitope annotation (provenance only; never a model input mode)
# ---------------------------------------------------------------------------

EVIDENCE_KIND_EXPERIMENTAL = "experimental"
EVIDENCE_KIND_STRUCTURAL = "structural"
EVIDENCE_KINDS = frozenset({EVIDENCE_KIND_EXPERIMENTAL, EVIDENCE_KIND_STRUCTURAL})

POSITION_BASIS_CONSTRUCT = "construct_1_based"
POSITION_BASIS_UNIPROT = "uniprot_1_based"
POSITION_BASES = frozenset({POSITION_BASIS_CONSTRUCT, POSITION_BASIS_UNIPROT})


@dataclass(frozen=True)
class EvidenceCitation:
    """Where an epitope claim came from. Both fields are mandatory."""

    kind: str
    identifier: str
    description: str

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_KINDS:
            raise EpitopeProvenanceError(
                f"evidence kind must be one of {sorted(EVIDENCE_KINDS)}, got {self.kind!r}"
            )
        _require_str(self.identifier, "identifier", EpitopeProvenanceError)
        _require_str(self.description, "description", EpitopeProvenanceError)

    def to_json_dict(self) -> dict:
        return {
            "description": self.description,
            "identifier": self.identifier,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class EpitopeMask:
    """Cited epitope residues, in a stated numbering basis.

    This carries no encoding, tokenization, or channel field. It is an annotation
    about the antigen, kept structurally separate from whatever the model is fed.
    """

    residue_positions: tuple[int, ...]
    position_basis: str
    evidence: EvidenceCitation | None = None

    def __post_init__(self) -> None:
        if self.evidence is None:
            raise EpitopeProvenanceError(
                "an epitope mask requires a cited experimental or structural source"
            )
        if not isinstance(self.evidence, EvidenceCitation):
            raise EpitopeProvenanceError(
                f"evidence must be an EvidenceCitation, got {type(self.evidence).__name__}"
            )
        if self.position_basis not in POSITION_BASES:
            raise EpitopeProvenanceError(
                f"position_basis must be one of {sorted(POSITION_BASES)}, "
                f"got {self.position_basis!r}"
            )
        positions = tuple(self.residue_positions)
        if not positions:
            raise EpitopeProvenanceError("an epitope mask must name at least one residue")
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int):
                raise EpitopeProvenanceError(
                    f"residue positions must be ints, got {position!r}"
                )
            if position < 1:
                raise EpitopeProvenanceError(
                    f"residue positions are 1-based and must be positive, got {position!r}"
                )
        object.__setattr__(self, "residue_positions", tuple(sorted(set(positions))))

    def to_json_dict(self) -> dict:
        assert self.evidence is not None
        return {
            "evidence": self.evidence.to_json_dict(),
            "position_basis": self.position_basis,
            "residue_positions": list(self.residue_positions),
        }


@dataclass(frozen=True)
class AntigenProvenance:
    """One antigen: its construct (exact or registered null), references, epitopes."""

    antigen_id: str
    construct: AssayedConstruct
    reference_sequences: tuple[ReferenceSequence, ...] = ()
    epitope_masks: tuple[EpitopeMask, ...] = ()

    def __post_init__(self) -> None:
        _require_str(self.antigen_id, "antigen_id", ConstructProvenanceError)
        if not isinstance(self.construct, (ExactAssayedConstruct, RegisteredNullConstruct)):
            raise ConstructProvenanceError(
                "construct must be an ExactAssayedConstruct or a RegisteredNullConstruct, got "
                f"{type(self.construct).__name__}; a reference sequence is never a substitute"
            )
        for reference in self.reference_sequences:
            if not isinstance(reference, ReferenceSequence):
                raise ConstructProvenanceError(
                    f"reference_sequences must hold ReferenceSequence, got "
                    f"{type(reference).__name__}"
                )
        sequence = assayed_construct_sequence(self.construct)
        for mask in self.epitope_masks:
            if not isinstance(mask, EpitopeMask):
                raise EpitopeProvenanceError(
                    f"epitope_masks must hold EpitopeMask, got {type(mask).__name__}"
                )
            if mask.position_basis == POSITION_BASIS_CONSTRUCT:
                if sequence is None:
                    raise EpitopeProvenanceError(
                        f"epitope mask for {self.antigen_id!r} is numbered against a construct "
                        "that is a registered null; use uniprot_1_based numbering or register "
                        "the construct first"
                    )
                overflow = [p for p in mask.residue_positions if p > len(sequence)]
                if overflow:
                    raise EpitopeProvenanceError(
                        f"epitope mask positions {overflow} exceed the {len(sequence)}-residue "
                        f"construct for {self.antigen_id!r}"
                    )

    @property
    def has_assayed_construct(self) -> bool:
        return isinstance(self.construct, ExactAssayedConstruct)

    def assayed_sequence_or_none(self) -> str | None:
        return assayed_construct_sequence(self.construct)

    def registered_null_reason(self) -> str | None:
        if isinstance(self.construct, RegisteredNullConstruct):
            return self.construct.reason_code
        return None

    def to_json_dict(self) -> dict:
        return {
            "antigen_id": self.antigen_id,
            "construct": self.construct.to_json_dict(),
            "epitope_masks": [mask.to_json_dict() for mask in self.epitope_masks],
            "reference_sequences": [ref.to_json_dict() for ref in self.reference_sequences],
        }


# ---------------------------------------------------------------------------
# 4. measurements: censoring, replicates, direction
# ---------------------------------------------------------------------------

CENSORING_NONE = "none"
CENSORING_LEFT = "left"
CENSORING_RIGHT = "right"
CENSORING_INTERVAL = "interval"
CENSORING_NON_BINDING = "non_binding"

CENSORING_MODES = frozenset(
    {CENSORING_NONE, CENSORING_LEFT, CENSORING_RIGHT, CENSORING_INTERVAL, CENSORING_NON_BINDING}
)

DIRECTION_HIGHER_IS_STRONGER = "higher_is_stronger"
DIRECTION_LOWER_IS_STRONGER = "lower_is_stronger"
ASSAY_DIRECTIONS = frozenset({DIRECTION_HIGHER_IS_STRONGER, DIRECTION_LOWER_IS_STRONGER})


@dataclass(frozen=True)
class Measurement:
    """One measured outcome in the source's own units and direction.

    Exactly one of five shapes:

    ``none``
        a point value; bounds absent.
    ``left``
        the true value lies below ``upper_bound`` (a detection limit). No point value.
    ``right``
        the true value lies above ``lower_bound``. No point value.
    ``interval``
        the true value lies in ``[lower_bound, upper_bound]``. No point value.
    ``non_binding``
        no binding was detected. No value, no bounds; an optional
        ``detection_limit`` records how hard the assay looked. This shape is what
        keeps a nonbinder from being written down as a weak KD.
    """

    censoring: str
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    detection_limit: float | None = None

    def __post_init__(self) -> None:
        if self.censoring not in CENSORING_MODES:
            raise MeasurementValidationError(
                f"censoring must be one of {sorted(CENSORING_MODES)}, got {self.censoring!r}"
            )
        for name in ("value", "lower_bound", "upper_bound", "detection_limit"):
            raw = getattr(self, name)
            if raw is not None:
                object.__setattr__(self, name, _require_finite(raw, name))

        if self.censoring != CENSORING_NON_BINDING and self.detection_limit is not None:
            raise MeasurementValidationError(
                "detection_limit is only meaningful for a non_binding outcome; a censored "
                "measurement states its bound directly"
            )

        if self.censoring == CENSORING_NONE:
            if self.value is None:
                raise MeasurementValidationError("an uncensored measurement needs a value")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise MeasurementValidationError(
                    "an uncensored measurement must not carry bounds"
                )
        elif self.censoring == CENSORING_LEFT:
            _require_no_value(self)
            if self.upper_bound is None:
                raise MeasurementValidationError("left-censored requires an upper_bound")
            if self.lower_bound is not None:
                raise MeasurementValidationError("left-censored must not carry a lower_bound")
        elif self.censoring == CENSORING_RIGHT:
            _require_no_value(self)
            if self.lower_bound is None:
                raise MeasurementValidationError("right-censored requires a lower_bound")
            if self.upper_bound is not None:
                raise MeasurementValidationError("right-censored must not carry an upper_bound")
        elif self.censoring == CENSORING_INTERVAL:
            _require_no_value(self)
            if self.lower_bound is None or self.upper_bound is None:
                raise MeasurementValidationError("an interval requires both bounds")
            if self.lower_bound > self.upper_bound:
                raise MeasurementValidationError(
                    f"interval bounds are out of order: {self.lower_bound} > {self.upper_bound}"
                )
        else:  # non_binding
            _require_no_value(self)
            if self.lower_bound is not None or self.upper_bound is not None:
                raise MeasurementValidationError(
                    "a non-binding outcome carries no bounds; use detection_limit"
                )

    # constructors -------------------------------------------------------
    @classmethod
    def point(cls, value: float) -> "Measurement":
        return cls(censoring=CENSORING_NONE, value=value)

    @classmethod
    def left_censored(cls, *, upper_bound: float) -> "Measurement":
        return cls(censoring=CENSORING_LEFT, upper_bound=upper_bound)

    @classmethod
    def right_censored(cls, *, lower_bound: float) -> "Measurement":
        return cls(censoring=CENSORING_RIGHT, lower_bound=lower_bound)

    @classmethod
    def interval(cls, lower_bound: float, upper_bound: float) -> "Measurement":
        return cls(
            censoring=CENSORING_INTERVAL, lower_bound=lower_bound, upper_bound=upper_bound
        )

    @classmethod
    def non_binding(cls, *, detection_limit: float | None = None) -> "Measurement":
        return cls(censoring=CENSORING_NON_BINDING, detection_limit=detection_limit)

    # views --------------------------------------------------------------
    @property
    def is_censored(self) -> bool:
        return self.censoring != CENSORING_NONE

    @property
    def is_non_binding(self) -> bool:
        return self.censoring == CENSORING_NON_BINDING

    def to_json_dict(self) -> dict:
        return {
            "censoring": self.censoring,
            "detection_limit": self.detection_limit,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "value": self.value,
        }

    @classmethod
    def from_json_dict(cls, document: Mapping) -> "Measurement":
        _check_key_set(
            document,
            ("censoring", "detection_limit", "lower_bound", "upper_bound", "value"),
            "measurement",
        )
        return cls(
            censoring=document["censoring"],
            value=document["value"],
            lower_bound=document["lower_bound"],
            upper_bound=document["upper_bound"],
            detection_limit=document["detection_limit"],
        )


def _require_no_value(measurement: Measurement) -> None:
    if measurement.value is not None:
        raise MeasurementValidationError(
            f"a {measurement.censoring!r} outcome must not carry a point value "
            f"({measurement.value!r}); that is how a nonbinder becomes a weak binder"
        )


@dataclass(frozen=True)
class NormalizedMeasurement:
    """A measurement moved into a single higher-is-stronger frame, once.

    The raw measurement and its original direction ride along, so a consumer can
    always recover exactly what the source reported.
    """

    strength: float | None
    censoring: str
    lower_bound: float | None
    upper_bound: float | None
    raw: Measurement
    raw_direction: str

    @property
    def is_non_binding(self) -> bool:
        return self.censoring == CENSORING_NON_BINDING

    @property
    def is_censored(self) -> bool:
        return self.censoring != CENSORING_NONE


def normalize_measurement(measurement: Measurement, direction: str) -> NormalizedMeasurement:
    """Flip a raw measurement into the higher-is-stronger frame exactly once.

    Flipping a value flips the censoring side too: ``KD > 1000 nM`` (weak, right
    censored) becomes ``strength < -1000`` (weak, left censored). Doing this in one
    place is the point -- each consumer re-deriving it is how a weak binder and a
    strong binder swap places.
    """
    if not isinstance(measurement, Measurement):
        raise MeasurementValidationError(
            f"normalize_measurement expects a raw Measurement, got "
            f"{type(measurement).__name__}; direction is normalized once, not repeatedly"
        )
    if direction not in ASSAY_DIRECTIONS:
        raise MeasurementValidationError(
            f"direction must be one of {sorted(ASSAY_DIRECTIONS)}, got {direction!r}"
        )

    if direction == DIRECTION_HIGHER_IS_STRONGER or measurement.is_non_binding:
        return NormalizedMeasurement(
            strength=measurement.value,
            censoring=measurement.censoring,
            lower_bound=measurement.lower_bound,
            upper_bound=measurement.upper_bound,
            raw=measurement,
            raw_direction=direction,
        )

    if measurement.censoring == CENSORING_NONE:
        assert measurement.value is not None
        return NormalizedMeasurement(
            strength=-measurement.value,
            censoring=CENSORING_NONE,
            lower_bound=None,
            upper_bound=None,
            raw=measurement,
            raw_direction=direction,
        )
    if measurement.censoring == CENSORING_LEFT:
        assert measurement.upper_bound is not None
        return NormalizedMeasurement(
            strength=None,
            censoring=CENSORING_RIGHT,
            lower_bound=-measurement.upper_bound,
            upper_bound=None,
            raw=measurement,
            raw_direction=direction,
        )
    if measurement.censoring == CENSORING_RIGHT:
        assert measurement.lower_bound is not None
        return NormalizedMeasurement(
            strength=None,
            censoring=CENSORING_LEFT,
            lower_bound=None,
            upper_bound=-measurement.lower_bound,
            raw=measurement,
            raw_direction=direction,
        )
    assert measurement.lower_bound is not None and measurement.upper_bound is not None
    return NormalizedMeasurement(
        strength=None,
        censoring=CENSORING_INTERVAL,
        lower_bound=-measurement.upper_bound,
        upper_bound=-measurement.lower_bound,
        raw=measurement,
        raw_direction=direction,
    )


@dataclass(frozen=True)
class AssaySpec:
    """What was measured, in what unit, and which way stronger points."""

    assay_name: str
    quantity: str
    unit: str
    direction: str

    def __post_init__(self) -> None:
        for name in ("assay_name", "quantity", "unit"):
            _require_str(getattr(self, name), name, MeasurementValidationError)
        if self.direction not in ASSAY_DIRECTIONS:
            raise MeasurementValidationError(
                f"direction must be one of {sorted(ASSAY_DIRECTIONS)}, got {self.direction!r}"
            )

    def to_json_dict(self) -> dict:
        return {
            "assay_name": self.assay_name,
            "direction": self.direction,
            "quantity": self.quantity,
            "unit": self.unit,
        }

    @classmethod
    def from_json_dict(cls, document: Mapping) -> "AssaySpec":
        _check_key_set(document, ("assay_name", "direction", "quantity", "unit"), "assay")
        return cls(
            assay_name=document["assay_name"],
            quantity=document["quantity"],
            unit=document["unit"],
            direction=document["direction"],
        )


@dataclass(frozen=True)
class Replicate:
    """One replicate measurement, kept individually.

    ``source_aggregate`` is the escape hatch for a release that published only a
    summary statistic: it must name its ``aggregation_method`` and, per
    :class:`AssayMeasurement`, it cannot sit beside individual replicates.
    """

    replicate_id: str
    measurement: Measurement
    source_aggregate: bool = False
    aggregation_method: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.replicate_id, "replicate_id", MeasurementValidationError)
        if not isinstance(self.measurement, Measurement):
            raise MeasurementValidationError(
                f"measurement must be a Measurement, got {type(self.measurement).__name__}"
            )
        if not isinstance(self.source_aggregate, bool):
            raise MeasurementValidationError("source_aggregate must be a boolean")
        if self.source_aggregate:
            _require_str(self.aggregation_method, "aggregation_method", MeasurementValidationError)
        elif self.aggregation_method is not None:
            raise MeasurementValidationError(
                "aggregation_method is only meaningful when source_aggregate is true"
            )

    def to_json_dict(self) -> dict:
        return {
            "aggregation_method": self.aggregation_method,
            "measurement": self.measurement.to_json_dict(),
            "replicate_id": self.replicate_id,
            "source_aggregate": self.source_aggregate,
        }

    @classmethod
    def from_json_dict(cls, document: Mapping) -> "Replicate":
        _check_key_set(
            document,
            ("aggregation_method", "measurement", "replicate_id", "source_aggregate"),
            "replicate",
        )
        return cls(
            replicate_id=document["replicate_id"],
            measurement=Measurement.from_json_dict(document["measurement"]),
            source_aggregate=document["source_aggregate"],
            aggregation_method=document["aggregation_method"],
        )


@dataclass(frozen=True)
class AssayMeasurement:
    """One assay applied to one binder/antigen pair, with every replicate retained.

    There is no aggregate field and no aggregation is performed here. A consumer
    that needs a summary must state its own convention; the schema refuses to pick
    one on its behalf.
    """

    assay: AssaySpec
    replicates: tuple[Replicate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assay, AssaySpec):
            raise MeasurementValidationError(
                f"assay must be an AssaySpec, got {type(self.assay).__name__}"
            )
        replicates = tuple(self.replicates)
        if not replicates:
            raise MeasurementValidationError("an assay measurement needs at least one replicate")
        for replicate in replicates:
            if not isinstance(replicate, Replicate):
                raise MeasurementValidationError(
                    f"replicates must hold Replicate, got {type(replicate).__name__}"
                )
        ids = [r.replicate_id for r in replicates]
        if len(set(ids)) != len(ids):
            raise MeasurementValidationError(f"replicate ids must be unique, got {ids}")
        aggregates = [r for r in replicates if r.source_aggregate]
        if aggregates and len(replicates) > 1:
            raise MeasurementValidationError(
                "a source-published aggregate stands alone; it may not sit beside individual "
                "replicates, because the reader could not tell which is which"
            )
        object.__setattr__(self, "replicates", replicates)

    @property
    def has_censored_replicate(self) -> bool:
        return any(r.measurement.is_censored for r in self.replicates)

    @property
    def is_source_aggregate_only(self) -> bool:
        return len(self.replicates) == 1 and self.replicates[0].source_aggregate

    def normalized(self) -> tuple[NormalizedMeasurement, ...]:
        """Every replicate in the higher-is-stronger frame, in replicate order."""
        return tuple(
            normalize_measurement(r.measurement, self.assay.direction) for r in self.replicates
        )

    def to_json_dict(self) -> dict:
        return {
            "assay": self.assay.to_json_dict(),
            "replicates": [r.to_json_dict() for r in self.replicates],
        }

    @classmethod
    def from_json_dict(cls, document: Mapping) -> "AssayMeasurement":
        _check_key_set(document, ("assay", "replicates"), "assay_measurement")
        return cls(
            assay=AssaySpec.from_json_dict(document["assay"]),
            replicates=tuple(Replicate.from_json_dict(r) for r in document["replicates"]),
        )


# ---------------------------------------------------------------------------
# 5. the assay provenance record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssayProvenanceRecord:
    """One binder measured against one antigen, with full provenance attached."""

    record_id: str
    dataset_name: str
    binder_id: str
    binder_sequence: str
    antigen: AntigenProvenance
    measurements: tuple[AssayMeasurement, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("record_id", "dataset_name", "binder_id"):
            _require_str(getattr(self, name), name, BenchmarkProvenanceError)
        _validate_residues(self.binder_sequence, "binder_sequence")
        if not isinstance(self.antigen, AntigenProvenance):
            raise ConstructProvenanceError(
                f"antigen must be an AntigenProvenance, got {type(self.antigen).__name__}; "
                "a bare sequence carries no construct provenance"
            )
        measurements = tuple(self.measurements)
        if not measurements:
            raise MeasurementValidationError("a record needs at least one assay measurement")
        for measurement in measurements:
            if not isinstance(measurement, AssayMeasurement):
                raise MeasurementValidationError(
                    f"measurements must hold AssayMeasurement, got {type(measurement).__name__}"
                )
        object.__setattr__(self, "measurements", measurements)

    @property
    def conditioning_status(self) -> str:
        """``exact_construct`` or ``registered_null`` -- never ambiguous."""
        return (
            CONDITIONING_EXACT_CONSTRUCT
            if self.antigen.has_assayed_construct
            else CONDITIONING_REGISTERED_NULL
        )


def record_to_json_dict(record: AssayProvenanceRecord) -> dict:
    return {
        "antigen": record.antigen.to_json_dict(),
        "binder_id": record.binder_id,
        "binder_sequence": record.binder_sequence,
        "dataset_name": record.dataset_name,
        "measurements": [m.to_json_dict() for m in record.measurements],
        "notes": record.notes,
        "record_id": record.record_id,
    }


def record_from_json_dict(document: Mapping) -> AssayProvenanceRecord:
    _check_key_set(
        document,
        (
            "antigen",
            "binder_id",
            "binder_sequence",
            "dataset_name",
            "measurements",
            "notes",
            "record_id",
        ),
        "record",
    )
    antigen_doc = _check_key_set(
        document["antigen"],
        ("antigen_id", "construct", "epitope_masks", "reference_sequences"),
        "antigen",
    )
    construct_doc = antigen_doc["construct"]
    if not isinstance(construct_doc, Mapping) or "kind" not in construct_doc:
        raise ConstructProvenanceError("construct must be a mapping carrying a 'kind' discriminator")
    if construct_doc["kind"] == CONSTRUCT_KIND_EXACT:
        _check_key_set(
            construct_doc, ("kind", "sequence", "source_kind", "source_locator"), "construct"
        )
        construct: AssayedConstruct = ExactAssayedConstruct(
            sequence=construct_doc["sequence"],
            source_kind=construct_doc["source_kind"],
            source_locator=construct_doc["source_locator"],
        )
    elif construct_doc["kind"] == CONSTRUCT_KIND_REGISTERED_NULL:
        _check_key_set(construct_doc, ("detail", "kind", "reason_code"), "construct")
        construct = RegisteredNullConstruct(
            reason_code=construct_doc["reason_code"], detail=construct_doc["detail"]
        )
    else:
        raise ConstructProvenanceError(
            f"unknown construct kind {construct_doc['kind']!r}; there are exactly two"
        )

    masks = []
    for mask_doc in antigen_doc["epitope_masks"]:
        _check_key_set(
            mask_doc, ("evidence", "position_basis", "residue_positions"), "epitope_mask"
        )
        evidence_doc = _check_key_set(
            mask_doc["evidence"], ("description", "identifier", "kind"), "evidence"
        )
        masks.append(
            EpitopeMask(
                residue_positions=tuple(mask_doc["residue_positions"]),
                position_basis=mask_doc["position_basis"],
                evidence=EvidenceCitation(
                    kind=evidence_doc["kind"],
                    identifier=evidence_doc["identifier"],
                    description=evidence_doc["description"],
                ),
            )
        )

    references = []
    for reference_doc in antigen_doc["reference_sequences"]:
        _check_key_set(reference_doc, ("accession", "role", "sequence"), "reference_sequence")
        references.append(
            ReferenceSequence(
                sequence=reference_doc["sequence"],
                role=reference_doc["role"],
                accession=reference_doc["accession"],
            )
        )

    return AssayProvenanceRecord(
        record_id=document["record_id"],
        dataset_name=document["dataset_name"],
        binder_id=document["binder_id"],
        binder_sequence=document["binder_sequence"],
        antigen=AntigenProvenance(
            antigen_id=antigen_doc["antigen_id"],
            construct=construct,
            reference_sequences=tuple(references),
            epitope_masks=tuple(masks),
        ),
        measurements=tuple(
            AssayMeasurement.from_json_dict(m) for m in document["measurements"]
        ),
        notes=document["notes"],
    )
