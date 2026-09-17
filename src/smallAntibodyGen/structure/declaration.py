"""The declared structural input: manifest schema, strict validation, digests.

Pure stdlib. This module defines *what a caller must state* before a local
PDB/mmCIF file may be turned into ESM-IF1 encoder inputs, and refuses anything
that does not fit. It reads no structure, imports no parser, and never fetches
anything: `prepare.py` does the reading, and it is handed only a manifest that
already passed through here.

One document, one hash
----------------------
The residue correspondence and the edit-space sites live **inside** the manifest
rather than in sibling files, so a single ``manifest_sha256`` pins the source
selection, the mapping, and the sites together. A site declaration cannot drift
away from the mapping whose indices it names.

What the schema refuses to leave implicit
-----------------------------------------
Every field below is load-bearing, and every enum is closed:

- ``source.format`` is **declared, never sniffed from the file extension**.
- ``source.model_ordinal`` is a 1-based position in file order. It is *not* a
  PDB ``MODEL`` serial number: both biotite readers slice by ordinal
  (``pdbx/convert.py:825-833`` takes a slice between first-occurrence indices of
  ``pdbx_PDB_model_num``; ``pdb/file.py:1164-1182`` slices between ``MODEL``
  line indices). A file whose models are numbered 5 and 7 is addressed as
  ordinals 1 and 2.
- ``source.residue_numbering`` must be ``"author"``. ``pdbx.get_structure``
  defaults ``use_author_fields=True`` (``convert.py:252``), which silently
  chooses between two different numbering systems; declaring it makes
  ``res_id`` mean one thing. ``"label"`` is rejected rather than supported.
- ``conventions.altloc`` must be ``"reject_any"`` and
  ``conventions.missing_backbone_atom`` must be ``"reject"``. v1 refuses these
  records instead of representing them, because
  ``specs/esmif1_policy.md`` names the missing-coordinate convention as an open
  gate and picking one here would silently close it.
- ``structural_relationship.kind`` states whether the backbone is the same
  construct as the edit-space context or a template for a different one.
  ``same_construct`` forbids every declared sequence mismatch.
- ``edit_space.sites`` keeps the **caller's order**.
  `ConstrainedEditSpace` sorts its sites by ascending position
  (``esmif1_policy.py:368``) and allele vectors follow that sorted order, so the
  caller's order has to be carried separately. :attr:`StructureManifest.declared_to_policy`
  and :attr:`StructureManifest.policy_to_declared` are that carrier.

Mismatches are checked in both directions: a mismatch that was not declared is
rejected, and a declared mismatch whose residues actually agree is rejected too.
A decorative field is worse than an absent one.

The validated manifest owns its own copy
----------------------------------------
:class:`StructureManifest` stores one authoritative representation: the
**canonical JSON text** of the document as it validated. The caller's dict is
never retained, :attr:`StructureManifest.document` hands back a fresh parse of
that text on every read, and every nested mapping the parsed sections expose is
a read-only view. So neither the object a caller passed in nor the document it
gets back can be edited into relabelling a manifest that already validated --
freezing a dataclass does not freeze the dictionaries inside it.

``dataclasses.replace`` can still build a *new* manifest whose parsed fields say
one thing and whose snapshot -- and digest -- say another, so
:func:`revalidate_manifest` re-parses the snapshot and requires the two to agree.
Every public boundary that acts on a manifest calls it.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


__all__ = [
    "ATTRIBUTION_KEYS",
    "CANONICAL_RESIDUES",
    "MANIFEST_KEYS",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_KIND",
    "RELATIONSHIP_KINDS",
    "STRUCTURE_FORMATS",
    "THREE_TO_ONE",
    "WINDOWS_FORBIDDEN_CHARACTERS",
    "WINDOWS_RESERVED_NAMES",
    "ChainDeclaration",
    "Conventions",
    "CorrespondenceRow",
    "ManifestValidationError",
    "SiteDeclaration",
    "SourceDeclaration",
    "StructureAdapterError",
    "StructureManifest",
    "StructuralRelationship",
    "canonical_json",
    "document_digest",
    "load_manifest",
    "loads_strict_json",
    "revalidate_manifest",
    "sha256_file",
    "validate_manifest",
]


MANIFEST_SCHEMA_VERSION = "esmif1-structure-manifest/1"
MANIFEST_KIND = "esmif1_structure_manifest"

#: The three-letter residue names v1 accepts in a selected chain, and the
#: one-letter code each maps to. Deliberately only the canonical 20: a modified
#: or unknown residue in a selected chain is rejected by name rather than
#: silently filtered out (see `prepare.py`).
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

#: The 20 canonical residues in alphabetical order. Must equal
#: `smallAntibodyGen.models.esmif1_policy.CANONICAL_RESIDUES`; replicated here so
#: that this module stays stdlib-only and importing it never pulls in torch.
#: `test_esmif1_structure.py` pins the two against each other.
CANONICAL_RESIDUES = "".join(sorted(THREE_TO_ONE.values()))

STRUCTURE_FORMATS = ("pdb", "mmcif")
RESIDUE_NUMBERINGS = ("author",)
ALTLOC_POLICIES = ("reject_any",)
MISSING_BACKBONE_POLICIES = ("reject",)
RELATIONSHIP_KINDS = (
    "same_construct",
    "template_for_different_construct",
    "synthetic_example",
)

MANIFEST_KEYS = (
    "chains",
    "conventions",
    "correspondence",
    "decoded_sequence",
    "edit_space",
    "kind",
    "notes",
    "schema_version",
    "source",
    "structural_relationship",
)
SOURCE_KEYS = (
    "attribution",
    "format",
    "model_ordinal",
    "relative_path",
    "residue_numbering",
    "sha256",
    "size_bytes",
)
ATTRIBUTION_KEYS = ("detail", "license", "source")
CHAINS_KEYS = ("confidence", "decoded_chain", "inter_chain_pad_length", "order")
CONVENTIONS_KEYS = ("altloc", "missing_backbone_atom")
RELATIONSHIP_KEYS = ("description", "kind")
CORRESPONDENCE_ROW_KEYS = (
    "chain_id",
    "expected_res_name",
    "ins_code",
    "mismatch",
    "res_id",
    "sequence_index",
    "sequence_residue",
)
MISMATCH_KEYS = ("reason",)
EDIT_SPACE_KEYS = ("sites",)
SITE_KEYS = ("allowed_residues", "attribution", "sequence_index", "site_id")
SITE_ATTRIBUTION_KEYS = ("detail", "source")

#: Basenames Windows resolves to a device whatever extension follows them, so
#: ``CON.prepared.json`` opens the console rather than creating a file. Refused
#: on every platform, in a declared path and in an output name alike. The
#: superscript digits are included because Windows resolves ``COM¹`` to the same
#: device as ``COM1``: the Win32 name parser folds the superscript forms.
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
    | {f"COM{index}" for index in "¹²³"}
    | {f"LPT{index}" for index in "¹²³"}
)

#: Characters Win32 refuses in a filename outright. ``:`` and the separators are
#: reported separately because they mean something specific; the rest are refused
#: as a group so a name that cannot exist on Windows cannot be declared here.
WINDOWS_FORBIDDEN_CHARACTERS = '<>:"|?*\\/'

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHAIN_ID_RE = re.compile(r"[A-Za-z0-9_]{1,4}")
_INS_CODE_RE = re.compile(r"[A-Za-z0-9]?")
_RES_NAME_RE = re.compile(r"[A-Z0-9]{1,3}")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class StructureAdapterError(ValueError):
    """Base class for every rejection in the `structure` package."""


class ManifestValidationError(StructureAdapterError):
    """A structural-input manifest is invalid, unsupported, or ambiguous."""


# ---------------------------------------------------------------------------
# strict JSON
# ---------------------------------------------------------------------------

def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict:
    """Refuse a JSON object that names the same key twice.

    `json.loads` keeps the *last* occurrence, so a document carrying two
    ``"sha256"`` keys validates against whichever one happens to come second.
    Canonicalizing that away would make the digest describe a document nobody
    wrote.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for key, _ in pairs:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ManifestValidationError(
            "JSON object names the same key more than once: "
            f"{sorted(set(duplicates))}; only the last value would survive."
        )
    return dict(pairs)


def _reject_json_constant(token: str) -> Any:
    """Refuse the ``NaN``/``Infinity`` literals Python's JSON accepts by default."""
    raise ManifestValidationError(
        f"JSON document contains the non-finite literal {token!r}; a coordinate, "
        "confidence, or index must be a finite number."
    )


def loads_strict_json(text: str, *, label: str = "document") -> Any:
    """Parse JSON, rejecting duplicate keys and non-finite numeric literals.

    Args:
        text: The document text.
        label: Used in the error message.

    Returns:
        The parsed document.

    Raises:
        ManifestValidationError: On malformed JSON, a duplicate key, or a
            ``NaN``/``Infinity``/``-Infinity`` literal.
    """
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ManifestValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ManifestValidationError(f"{label} is not valid JSON: {error}") from error


def canonical_json(document: Any) -> str:
    """Canonical text for a document: sorted keys, two-space indent, final newline.

    The one serializer used for digests and for every file this package writes,
    so a document rewritten from its own parsed form is byte-identical.
    """
    return json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"


def document_digest(document: Any) -> str:
    """SHA-256 over :func:`canonical_json` of ``document``."""
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Stream a local file's SHA-256. Local read only; never a download."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# small shared validators
# ---------------------------------------------------------------------------

def _check_keys(document: Any, expected: Sequence[str], label: str) -> Mapping:
    if not isinstance(document, Mapping):
        raise ManifestValidationError(
            f"{label} must be an object, got {type(document).__name__}."
        )
    present, wanted = set(document), set(expected)
    missing, unknown = sorted(wanted - present), sorted(present - wanted)
    if missing or unknown:
        raise ManifestValidationError(
            f"{label} key mismatch; missing={missing} unknown={unknown}. Every key "
            "is required, and an unknown key is refused rather than ignored."
        )
    return document


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{label} must be a non-empty string, got {value!r}.")
    if value != value.strip():
        raise ManifestValidationError(f"{label} must not carry surrounding whitespace.")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{label} must be an integer, got {value!r}.")
    return operator.index(value)


def _unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{label} must be a real number, got {value!r}.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        # An int with no float image (10**400) is a declared value this layer
        # refuses by name, not an OverflowError escaping to the caller.
        raise ManifestValidationError(
            f"{label} is not representable as a real number: {error}"
        ) from error
    if not math.isfinite(number):
        raise ManifestValidationError(f"{label} must be finite, got {value!r}.")
    if not 0.0 <= number <= 1.0:
        raise ManifestValidationError(f"{label} must lie in [0, 1], got {number!r}.")
    return number


def _enum(value: Any, allowed: Sequence[str], label: str) -> str:
    if value not in allowed:
        raise ManifestValidationError(
            f"{label} must be one of {list(allowed)}, got {value!r}."
        )
    return value


def _frozen(mapping: Mapping) -> Mapping:
    """A read-only view over a private copy, for a mapping a dataclass exposes."""
    return MappingProxyType(dict(mapping))


def _attribution(document: Any, keys: Sequence[str], label: str) -> Mapping[str, str]:
    doc = _check_keys(document, keys, label)
    return _frozen({key: _text(doc[key], f"{label}.{key}") for key in keys})


def _check_portable_name(part: str, label: str) -> str:
    """Refuse a path component that does not mean one file on every platform.

    Checked on every OS, not only Windows: a manifest that resolves to a
    different file -- or to a device -- when it is carried to another machine is
    not a pinned declaration.

    - ``:`` separates a drive letter or an NTFS alternate data stream, so
      ``toy.pdb:stream`` addresses a hidden stream of ``toy.pdb`` on Windows and
      an ordinary file elsewhere;
    - ``<>"|?*`` are forbidden in a Win32 filename outright, and ``?``/``*`` are
      wildcards rather than characters, so ``bad?.pdb`` names no file there and
      one file here;
    - a trailing ``.`` or space is stripped by the Windows API, so ``toy.pdb.``
      and ``toy.pdb`` are the same file there and two files here;
    - ``CON``, ``NUL``, ``COM1``, ``COM¹`` ... stay device names whatever
      extension follows;
    - a control character is not a filename.
    """
    if any(character < " " or character == "\x7f" for character in part):
        raise ManifestValidationError(
            f"{label} component {part!r} contains a control character."
        )
    if ":" in part:
        raise ManifestValidationError(
            f"{label} component {part!r} contains ':', which names a drive or an "
            "alternate data stream rather than a plain file."
        )
    forbidden = sorted({c for c in part if c in WINDOWS_FORBIDDEN_CHARACTERS})
    if forbidden:
        raise ManifestValidationError(
            f"{label} component {part!r} contains {forbidden}, which Windows refuses in "
            "a filename (and treats '?' and '*' as wildcards); the name would address no "
            "file there and one file here."
        )
    if part != part.rstrip(". "):
        raise ManifestValidationError(
            f"{label} component {part!r} ends in a dot or a space; Windows strips "
            "those, so the name would address a different file there."
        )
    if part.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        raise ManifestValidationError(
            f"{label} component {part!r} is a reserved Windows device name "
            f"({sorted(WINDOWS_RESERVED_NAMES)}); the extension does not make it a file."
        )
    return part


def _relative_path(value: Any) -> str:
    path = _text(value, "source.relative_path")
    if path.startswith("/") or path.endswith("/") or "\\" in path:
        raise ManifestValidationError(
            f"source.relative_path must be a forward-slash relative path, got {path!r}."
        )
    windows = PureWindowsPath(path)
    if windows.drive or windows.root:
        raise ManifestValidationError(
            f"source.relative_path {path!r} is drive-qualified, rooted or a UNC share; "
            "it must be relative to the caller's structure root on every platform."
        )
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ManifestValidationError(
            f"source.relative_path must not traverse or repeat separators: {path!r}."
        )
    for part in parts:
        _check_portable_name(part, "source.relative_path")
    return path


# ---------------------------------------------------------------------------
# declared pieces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceDeclaration:
    """The pinned local structure file and how to read it.

    Attributes:
        relative_path: Forward-slash path below the caller's structure root.
        sha256: Expected content digest of that file, 64 lowercase hex.
        size_bytes: Expected size, checked alongside the digest so a truncated
            read is named as a size mismatch rather than an opaque hash failure.
        format: ``"pdb"`` or ``"mmcif"``. Declared, never inferred from the
            extension.
        model_ordinal: 1-based position of the model **in file order**, not a
            ``MODEL`` serial (see the module docstring).
        residue_numbering: ``"author"``. Fixes what ``res_id`` means.
        attribution: ``source`` / ``detail`` / ``license``, all required.
            Read-only, like every mapping this module exposes.
    """

    relative_path: str
    sha256: str
    size_bytes: int
    format: str
    model_ordinal: int
    residue_numbering: str
    attribution: Mapping[str, str]


@dataclass(frozen=True)
class ChainDeclaration:
    """Which chains are encoded, in which order, and how they are padded apart.

    Attributes:
        order: Chain ids in packing order. ``order[0]`` is the decoded chain,
            because upstream always concatenates the target chain first
            (``esm/inverse_folding/multichain_util.py:68-77``).
        decoded_chain: Must equal ``order[0]``; declared anyway so a manifest
            that disagrees with itself is rejected instead of being reconciled.
        confidence: Per-chain confidence in ``[0, 1]``, one entry per chain.
        inter_chain_pad_length: Rows of ``NaN`` inserted between chains.
            Upstream's default is 10.
    """

    order: tuple[str, ...]
    decoded_chain: str
    confidence: Mapping[str, float]
    inter_chain_pad_length: int

    @property
    def context_chains(self) -> tuple[str, ...]:
        """Chains after the decoded one, in packing order."""
        return self.order[1:]


@dataclass(frozen=True)
class Conventions:
    """The two coordinate conventions v1 declares, both of them refusals."""

    altloc: str
    missing_backbone_atom: str


@dataclass(frozen=True)
class StructuralRelationship:
    """How the backbone relates to the sequence being edited."""

    kind: str
    description: str


@dataclass(frozen=True)
class CorrespondenceRow:
    """One decoded sequence index tied to one full structural residue identity.

    Attributes:
        sequence_index: 0-based index into ``decoded_sequence``.
        chain_id: Always the decoded chain.
        res_id: Author residue number.
        ins_code: Author insertion code, ``""`` when absent. Part of the
            identity: 52, 52A and 52B are three different residues.
        expected_res_name: The three-letter name the structure must hold there.
        sequence_residue: ``decoded_sequence[sequence_index]``, restated so the
            row is readable on its own and checkable against the sequence.
        mismatch: ``None`` when the structural residue and the sequence residue
            agree; otherwise ``{"reason": ...}`` declaring the disagreement.
    """

    sequence_index: int
    chain_id: str
    res_id: int
    ins_code: str
    expected_res_name: str
    sequence_residue: str
    mismatch: Mapping[str, str] | None

    @property
    def residue_key(self) -> tuple[str, int, str]:
        """``(chain_id, res_id, ins_code)`` -- the full structural identity."""
        return (self.chain_id, self.res_id, self.ins_code)


@dataclass(frozen=True)
class SiteDeclaration:
    """One editable site, in the caller's declared order.

    Attributes:
        site_id: Stable identifier. Survives the policy's positional sort.
        sequence_index: 0-based index into ``decoded_sequence``.
        allowed_residues: Exactly two distinct canonical residues.
            ``allowed_residues[0]`` becomes allele index 0, faithfully:
            `EditableSite` takes allele order from the caller's tuple
            (``esmif1_policy.py:257-261``).
        attribution: ``source`` / ``detail``, both required.
    """

    site_id: str
    sequence_index: int
    allowed_residues: tuple[str, str]
    attribution: Mapping[str, str]


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructureManifest:
    """A validated structural-input declaration.

    Holds the parsed sections *and* a canonical text snapshot of the document
    they came from, so the digest that pins this manifest is over exactly what
    the caller wrote and stays over exactly that. ``canonical_text`` is the one
    authoritative copy: the caller's dict is not retained, :attr:`document`
    re-parses the snapshot on every read, and the parsed sections expose
    read-only mappings.
    """

    source: SourceDeclaration
    chains: ChainDeclaration
    conventions: Conventions
    relationship: StructuralRelationship
    decoded_sequence: str
    correspondence: tuple[CorrespondenceRow, ...]
    sites: tuple[SiteDeclaration, ...]
    notes: str
    canonical_text: str

    @property
    def document(self) -> dict:
        """A fresh copy of the declared document, parsed from the snapshot.

        A new object every call, on purpose: whatever a caller does to what it
        gets back cannot reach this manifest.
        """
        return json.loads(self.canonical_text)

    @property
    def digest(self) -> str:
        """SHA-256 of the canonical form of the declared document."""
        return hashlib.sha256(self.canonical_text.encode("utf-8")).hexdigest()

    @property
    def decoded_chain(self) -> str:
        return self.chains.decoded_chain

    @property
    def num_decoded_residues(self) -> int:
        return len(self.decoded_sequence)

    @property
    def declared_site_ids(self) -> tuple[str, ...]:
        """Site ids in the caller's declared order."""
        return tuple(site.site_id for site in self.sites)

    @property
    def policy_sites(self) -> tuple[SiteDeclaration, ...]:
        """Sites in the order `ConstrainedEditSpace` will hold them.

        The policy sorts by ascending position (``esmif1_policy.py:368``) and
        every per-site vector it returns follows that order.
        """
        return tuple(sorted(self.sites, key=lambda site: site.sequence_index))

    @property
    def policy_site_ids(self) -> tuple[str, ...]:
        """Site ids in allele-vector order."""
        return tuple(site.site_id for site in self.policy_sites)

    @property
    def declared_to_policy(self) -> tuple[int, ...]:
        """``declared_to_policy[i]`` is where declared site ``i`` sits in policy order."""
        policy_index = {site_id: i for i, site_id in enumerate(self.policy_site_ids)}
        return tuple(policy_index[site_id] for site_id in self.declared_site_ids)

    @property
    def policy_to_declared(self) -> tuple[int, ...]:
        """``policy_to_declared[j]`` is where policy site ``j`` sits in declared order."""
        declared_index = {site_id: i for i, site_id in enumerate(self.declared_site_ids)}
        return tuple(declared_index[site_id] for site_id in self.policy_site_ids)

    def site_by_id(self, site_id: str) -> SiteDeclaration:
        """The declared site named ``site_id``.

        Raises:
            KeyError: If no site carries that id.
        """
        for site in self.sites:
            if site.site_id == site_id:
                return site
        raise KeyError(site_id)


def validate_manifest(document: Any) -> StructureManifest:
    """Validate a manifest document strictly and return it parsed.

    Rejects rather than repairs: nothing is trimmed, defaulted, case-folded,
    sorted, or reconciled. Every cross-check that could otherwise pass silently
    is listed in the module docstring.

    Args:
        document: A parsed JSON object.

    Returns:
        A :class:`StructureManifest`.

    Raises:
        ManifestValidationError: On any violation, naming the field.
    """
    doc = _check_keys(document, MANIFEST_KEYS, "manifest")
    _enum(doc["schema_version"], (MANIFEST_SCHEMA_VERSION,), "manifest.schema_version")
    _enum(doc["kind"], (MANIFEST_KIND,), "manifest.kind")
    if not isinstance(doc["notes"], str):
        raise ManifestValidationError("manifest.notes must be a string.")

    source = _validate_source(doc["source"])
    chains = _validate_chains(doc["chains"])
    conventions = _validate_conventions(doc["conventions"])
    relationship = _validate_relationship(doc["structural_relationship"])
    decoded_sequence = _validate_decoded_sequence(doc["decoded_sequence"])
    correspondence = _validate_correspondence(
        doc["correspondence"], decoded_sequence, chains, relationship
    )
    sites = _validate_edit_space(doc["edit_space"], decoded_sequence)

    # Snapshot last, once every field has been checked: the text is what the
    # digest, the artifact and every later comparison are taken over, and it is
    # taken by value so a caller still holding `document` cannot edit it.
    try:
        canonical_text = canonical_json(doc)
    except (TypeError, ValueError) as error:
        raise ManifestValidationError(
            f"manifest holds a value that is not JSON-serializable: {error}"
        ) from error
    # The digest, the artifact and every file this package writes are UTF-8 bytes
    # of this text. A lone surrogate is a legal `str` and a legal JSON string, so
    # it is refused here rather than at the first `.encode()` somewhere downstream.
    try:
        canonical_text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ManifestValidationError(
            f"manifest holds text that is not encodable as UTF-8: {error}. Every digest "
            "and every file this package writes is UTF-8, so such a document could be "
            "validated but never hashed or written."
        ) from error

    return StructureManifest(
        source=source,
        chains=chains,
        conventions=conventions,
        relationship=relationship,
        decoded_sequence=decoded_sequence,
        correspondence=correspondence,
        sites=sites,
        notes=doc["notes"],
        canonical_text=canonical_text,
    )


def revalidate_manifest(manifest: StructureManifest) -> StructureManifest:
    """Re-check a `StructureManifest`'s parsed fields against its own snapshot.

    ``canonical_text`` is what the digest, the artifact and every comparison are
    taken over, but the parsed fields beside it are what a caller *acts* on, and
    `dataclasses.replace` builds a new manifest from arbitrary parts while
    carrying the old text -- and therefore the old digest -- along. A manifest
    whose declared support says ``G/T`` while its snapshot says ``G/S`` would edit
    one space and be pinned as another.

    So every public boundary that acts on a manifest re-parses the snapshot and
    requires the result to equal the object in hand, field for field, the
    canonical text included. One JSON parse; no hashing of arrays, no base64.

    Returns:
        The manifest rebuilt from its own snapshot -- equal to ``manifest``.

    Raises:
        ManifestValidationError: If the snapshot is unparseable or invalid, or if
            any parsed field disagrees with it.
    """
    if not isinstance(manifest, StructureManifest):
        raise ManifestValidationError(
            f"expected a validated StructureManifest, got {type(manifest).__name__}."
        )
    rebuilt = validate_manifest(
        loads_strict_json(manifest.canonical_text, label="manifest snapshot")
    )
    if rebuilt != manifest:
        raise ManifestValidationError(
            "the manifest's parsed fields disagree with the canonical snapshot its "
            "digest is taken over. The snapshot is authoritative: a manifest that was "
            "rebuilt field by field (dataclasses.replace) keeps the old text and the "
            "old digest, so acting on the parsed fields would use a declaration nobody "
            "hashed."
        )
    return rebuilt


def load_manifest(path: Path | str) -> StructureManifest:
    """Read and validate one local manifest JSON file.

    Local file read only. The structure file it names is **not** opened here.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestValidationError(f"cannot read manifest {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise ManifestValidationError(
            f"manifest {path} is not valid UTF-8: {error}"
        ) from error
    return validate_manifest(loads_strict_json(text, label=f"manifest {path}"))


# ---------------------------------------------------------------------------
# section validators
# ---------------------------------------------------------------------------

def _validate_source(document: Any) -> SourceDeclaration:
    doc = _check_keys(document, SOURCE_KEYS, "manifest.source")
    sha256 = doc["sha256"]
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ManifestValidationError(
            f"manifest.source.sha256 must be 64 lowercase hex characters, got {sha256!r}."
        )
    size_bytes = _integer(doc["size_bytes"], "manifest.source.size_bytes")
    if size_bytes <= 0:
        raise ManifestValidationError(
            f"manifest.source.size_bytes must be positive, got {size_bytes}."
        )
    model_ordinal = _integer(doc["model_ordinal"], "manifest.source.model_ordinal")
    if model_ordinal < 1:
        raise ManifestValidationError(
            f"manifest.source.model_ordinal is 1-based and must be >= 1, got "
            f"{model_ordinal}. It selects the model's position in file order, not a "
            "PDB MODEL serial number."
        )
    return SourceDeclaration(
        relative_path=_relative_path(doc["relative_path"]),
        sha256=sha256,
        size_bytes=size_bytes,
        format=_enum(doc["format"], STRUCTURE_FORMATS, "manifest.source.format"),
        model_ordinal=model_ordinal,
        residue_numbering=_enum(
            doc["residue_numbering"], RESIDUE_NUMBERINGS, "manifest.source.residue_numbering"
        ),
        attribution=_attribution(
            doc["attribution"], ATTRIBUTION_KEYS, "manifest.source.attribution"
        ),
    )


def _validate_chains(document: Any) -> ChainDeclaration:
    doc = _check_keys(document, CHAINS_KEYS, "manifest.chains")
    raw_order = doc["order"]
    if not isinstance(raw_order, list) or not raw_order:
        raise ManifestValidationError(
            "manifest.chains.order must be a non-empty list of chain ids."
        )
    order: list[str] = []
    for index, chain_id in enumerate(raw_order):
        text = _text(chain_id, f"manifest.chains.order[{index}]")
        if not _CHAIN_ID_RE.fullmatch(text):
            raise ManifestValidationError(
                f"manifest.chains.order[{index}] is not a plausible chain id: {text!r}."
            )
        if text in order:
            raise ManifestValidationError(
                f"manifest.chains.order repeats chain {text!r}; each chain is packed once."
            )
        order.append(text)

    decoded_chain = _text(doc["decoded_chain"], "manifest.chains.decoded_chain")
    if decoded_chain != order[0]:
        raise ManifestValidationError(
            f"manifest.chains.decoded_chain is {decoded_chain!r} but chains.order[0] is "
            f"{order[0]!r}. Upstream concatenates the decoded chain first "
            "(esm/inverse_folding/multichain_util.py:68-77), so the two cannot disagree."
        )

    confidence_doc = doc["confidence"]
    if not isinstance(confidence_doc, Mapping):
        raise ManifestValidationError("manifest.chains.confidence must be an object.")
    if set(confidence_doc) != set(order):
        raise ManifestValidationError(
            "manifest.chains.confidence must name exactly the declared chains; "
            f"missing={sorted(set(order) - set(confidence_doc))} "
            f"unknown={sorted(set(confidence_doc) - set(order))}."
        )
    confidence = _frozen(
        {
            chain_id: _unit_interval(
                confidence_doc[chain_id], f"manifest.chains.confidence[{chain_id!r}]"
            )
            for chain_id in order
        }
    )

    pad_length = _integer(
        doc["inter_chain_pad_length"], "manifest.chains.inter_chain_pad_length"
    )
    if pad_length < 1:
        raise ManifestValidationError(
            f"manifest.chains.inter_chain_pad_length must be positive, got {pad_length}."
        )
    return ChainDeclaration(
        order=tuple(order),
        decoded_chain=decoded_chain,
        confidence=confidence,
        inter_chain_pad_length=pad_length,
    )


def _validate_conventions(document: Any) -> Conventions:
    doc = _check_keys(document, CONVENTIONS_KEYS, "manifest.conventions")
    return Conventions(
        altloc=_enum(doc["altloc"], ALTLOC_POLICIES, "manifest.conventions.altloc"),
        missing_backbone_atom=_enum(
            doc["missing_backbone_atom"],
            MISSING_BACKBONE_POLICIES,
            "manifest.conventions.missing_backbone_atom",
        ),
    )


def _validate_relationship(document: Any) -> StructuralRelationship:
    doc = _check_keys(document, RELATIONSHIP_KEYS, "manifest.structural_relationship")
    return StructuralRelationship(
        kind=_enum(
            doc["kind"], RELATIONSHIP_KINDS, "manifest.structural_relationship.kind"
        ),
        description=_text(
            doc["description"], "manifest.structural_relationship.description"
        ),
    )


def _validate_decoded_sequence(value: Any) -> str:
    sequence = _text(value, "manifest.decoded_sequence")
    bad = sorted({char for char in sequence if char not in CANONICAL_RESIDUES})
    if bad:
        raise ManifestValidationError(
            f"manifest.decoded_sequence contains {bad!r}, which is outside the canonical "
            f"residues {CANONICAL_RESIDUES!r}. v1 accepts only canonical residues in a "
            "selected chain, so a decoded sequence cannot contain anything else."
        )
    return sequence


def _validate_correspondence(
    document: Any,
    decoded_sequence: str,
    chains: ChainDeclaration,
    relationship: StructuralRelationship,
) -> tuple[CorrespondenceRow, ...]:
    if not isinstance(document, list):
        raise ManifestValidationError("manifest.correspondence must be a list.")
    if len(document) != len(decoded_sequence):
        raise ManifestValidationError(
            f"manifest.correspondence has {len(document)} rows but decoded_sequence is "
            f"{len(decoded_sequence)} residues long; every decoded index must be covered "
            "exactly once."
        )

    rows: list[CorrespondenceRow] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for index, row_doc in enumerate(document):
        label = f"manifest.correspondence[{index}]"
        row = _check_keys(row_doc, CORRESPONDENCE_ROW_KEYS, label)

        sequence_index = _integer(row["sequence_index"], f"{label}.sequence_index")
        if sequence_index != index:
            raise ManifestValidationError(
                f"{label}.sequence_index is {sequence_index} but the row sits at position "
                f"{index}; rows must cover 0..{len(decoded_sequence) - 1} in ascending "
                "order, with no holes and no duplicates."
            )

        chain_id = _text(row["chain_id"], f"{label}.chain_id")
        if chain_id != chains.decoded_chain:
            raise ManifestValidationError(
                f"{label}.chain_id is {chain_id!r} but the decoded chain is "
                f"{chains.decoded_chain!r}; the correspondence maps the decoded chain only."
            )

        res_id = _integer(row["res_id"], f"{label}.res_id")
        ins_code = row["ins_code"]
        if not isinstance(ins_code, str) or not _INS_CODE_RE.fullmatch(ins_code):
            raise ManifestValidationError(
                f"{label}.ins_code must be one alphanumeric character or \"\", got "
                f"{ins_code!r}."
            )

        expected_res_name = row["expected_res_name"]
        if not isinstance(expected_res_name, str) or not _RES_NAME_RE.fullmatch(
            expected_res_name
        ):
            raise ManifestValidationError(
                f"{label}.expected_res_name must be an uppercase residue name, got "
                f"{expected_res_name!r}."
            )
        if expected_res_name not in THREE_TO_ONE:
            raise ManifestValidationError(
                f"{label}.expected_res_name is {expected_res_name!r}, which is not one of "
                f"the canonical residues v1 supports ({sorted(THREE_TO_ONE)}). A modified "
                "or unknown residue in a selected chain is rejected, never filtered."
            )

        sequence_residue = row["sequence_residue"]
        if sequence_residue != decoded_sequence[index]:
            raise ManifestValidationError(
                f"{label}.sequence_residue is {sequence_residue!r} but "
                f"decoded_sequence[{index}] is {decoded_sequence[index]!r}."
            )

        key = (chain_id, res_id, ins_code)
        if key in seen_keys:
            raise ManifestValidationError(
                f"{label} maps structural residue {key} a second time; one structural "
                "residue cannot stand at two decoded indices."
            )
        seen_keys.add(key)

        mismatch = _validate_mismatch(
            row["mismatch"], expected_res_name, sequence_residue, relationship, label
        )
        rows.append(
            CorrespondenceRow(
                sequence_index=sequence_index,
                chain_id=chain_id,
                res_id=res_id,
                ins_code=ins_code,
                expected_res_name=expected_res_name,
                sequence_residue=sequence_residue,
                mismatch=mismatch,
            )
        )
    return tuple(rows)


def _validate_mismatch(
    document: Any,
    expected_res_name: str,
    sequence_residue: str,
    relationship: StructuralRelationship,
    label: str,
) -> Mapping[str, str] | None:
    """Check a declared mismatch in both directions.

    An undeclared mismatch would let a template quietly stand in for a different
    construct; a declared mismatch where the residues agree is a field that says
    something untrue about the data.
    """
    structural_residue = THREE_TO_ONE[expected_res_name]
    residues_agree = structural_residue == sequence_residue

    if document is None:
        if not residues_agree:
            raise ManifestValidationError(
                f"{label} maps {expected_res_name} ({structural_residue}) onto sequence "
                f"residue {sequence_residue!r} but declares no mismatch. A structural "
                "template may differ from the edited construct, but the difference has to "
                "be stated."
            )
        return None

    if residues_agree:
        raise ManifestValidationError(
            f"{label} declares a mismatch, but {expected_res_name} and "
            f"{sequence_residue!r} are the same residue."
        )
    if relationship.kind == "same_construct":
        raise ManifestValidationError(
            f"{label} declares a mismatch, but structural_relationship.kind is "
            "'same_construct'. Either the relationship is a template relationship or the "
            "mismatch is an error; both readings cannot hold."
        )
    doc = _check_keys(document, MISMATCH_KEYS, f"{label}.mismatch")
    return _frozen({"reason": _text(doc["reason"], f"{label}.mismatch.reason")})


def _validate_edit_space(
    document: Any, decoded_sequence: str
) -> tuple[SiteDeclaration, ...]:
    doc = _check_keys(document, EDIT_SPACE_KEYS, "manifest.edit_space")
    raw_sites = doc["sites"]
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ManifestValidationError(
            "manifest.edit_space.sites must be a non-empty list; a space with no sites "
            "defines a point distribution, not an editing policy."
        )

    sites: list[SiteDeclaration] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for index, site_doc in enumerate(raw_sites):
        label = f"manifest.edit_space.sites[{index}]"
        site = _check_keys(site_doc, SITE_KEYS, label)

        site_id = _text(site["site_id"], f"{label}.site_id")
        if site_id in seen_ids:
            raise ManifestValidationError(
                f"{label}.site_id {site_id!r} is declared twice; site ids are the stable "
                "handles that survive the policy's positional sort, so they must be unique."
            )
        seen_ids.add(site_id)

        sequence_index = _integer(site["sequence_index"], f"{label}.sequence_index")
        if not 0 <= sequence_index < len(decoded_sequence):
            raise ManifestValidationError(
                f"{label}.sequence_index {sequence_index} is outside the decoded sequence "
                f"(0..{len(decoded_sequence) - 1})."
            )
        if sequence_index in seen_indices:
            raise ManifestValidationError(
                f"{label}.sequence_index {sequence_index} is declared twice."
            )
        seen_indices.add(sequence_index)

        allowed = site["allowed_residues"]
        if not isinstance(allowed, list) or len(allowed) != 2:
            raise ManifestValidationError(
                f"{label}.allowed_residues needs exactly two residues, got {allowed!r}. "
                "`EditableSite` refuses any other arity (esmif1_policy.py:284-289); the "
                "declared support is binary."
            )
        for position, residue in enumerate(allowed):
            if (
                not isinstance(residue, str)
                or len(residue) != 1
                or residue not in CANONICAL_RESIDUES
            ):
                raise ManifestValidationError(
                    f"{label}.allowed_residues[{position}] is {residue!r}; each allele must "
                    f"be exactly one canonical residue from {CANONICAL_RESIDUES!r}."
                )
        if allowed[0] == allowed[1]:
            raise ManifestValidationError(
                f"{label}.allowed_residues are both {allowed[0]!r}; the support would be a "
                "single residue with probability one."
            )
        if decoded_sequence[sequence_index] not in allowed:
            raise ManifestValidationError(
                f"{label}: decoded_sequence[{sequence_index}] is "
                f"{decoded_sequence[sequence_index]!r}, outside this site's support "
                f"{allowed!r}. The context must itself be a member of the edit space."
            )

        sites.append(
            SiteDeclaration(
                site_id=site_id,
                sequence_index=sequence_index,
                allowed_residues=(allowed[0], allowed[1]),
                attribution=_attribution(
                    site["attribution"], SITE_ATTRIBUTION_KEYS, f"{label}.attribution"
                ),
            )
        )
    return tuple(sites)
