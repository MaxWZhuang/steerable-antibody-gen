"""Turn a validated declaration plus a local structure file into encoder inputs.

biotite is imported **lazily**, inside the functions that parse: importing this
module never requires the optional `esm-if1` extra, and
`test_esmif1_structure.py` asserts that in a subprocess.

What this produces
------------------
`ConstrainedEditPolicy.encode_geometry` wants an ``(L, 3, 3)`` N/CA/C array and a
confidence vector. That is all this module builds, packed in upstream's own
multichain convention: the decoded chain first, then, for each further chain, a
block of ``NaN`` rows followed by that chain
(``esm/inverse_folding/multichain_util.py:68-77``). ``NaN`` therefore appears in
exactly one place -- the inter-chain pad -- which is upstream's own meaning for
it. `FixedGeometry` is never serialized: the artifact holds the *inputs* to the
encoder, because the encoding itself is device- and dtype-bound
(``esmif1_policy.py:523-541``).

Why biotite is called directly rather than through `esm.inverse_folding.util`
-----------------------------------------------------------------------------
`util.load_structure` defaults ``altloc="first"`` -- a silent choice among
alternate conformations -- and `util.get_atom_coords_residuewise` writes ``NaN``
for an absent backbone atom, which would turn a missing coordinate into padding
without saying so. Both are refusals in this layer's declared conventions, so it
parses with ``altloc="all"`` (nothing is dropped, and every alternate location is
*visible*) and rejects. The packing convention is validated against the real
``multichain_util._concatenate_coords`` by an optional test rather than by
routing through it.

The v1 rejection rules, all of them restrictive on purpose
----------------------------------------------------------
Within the selected model:

- any record whose author chain identifier is absent, ``.`` or ``?`` is rejected,
  for every chain. Chain selection would otherwise be ambiguous.
- a record on a **selected** chain is rejected if it is a hetero record, if its
  residue name is outside the canonical 20, if its author residue number, residue
  name or atom name is unresolved, or if it carries an alternate-location
  identifier. This means a solvent or ligand record that shares a selected
  chain's author id causes the whole preparation to fail. It is not filtered out,
  because filtering is how a ligand quietly becomes "not there". A declared
  allowance is the future extension; v1 has no silent non-polymer exclusion path.
- every mapped residue must carry exactly one ``N``, one ``CA`` and one ``C`` with
  finite coordinates, matched by exact atom-name identity. Zero is a missing
  atom, more than one is ambiguous; both are named and refused. No residue is
  ever dropped, and no atom is ever chosen by distance or residue number.
- all selected-chain atoms must have finite coordinates and finite occupancy
  in ``(0, 1]``, including atoms omitted from the encoder's backbone array.

The correspondence must cover the **entire observed selected decoded chain** in
file residue order, with no omissions and no reordering: v1 has no cropping, and
a partial mapping would silently encode residues nobody declared.

Numbering gaps are reported as a neutral observation. A gap in author numbering
does **not** establish a spatial break or a missing residue -- author numbering is
frequently discontinuous by convention -- so nothing is inferred from one and
nothing is repaired.

Integrity versus provenance
---------------------------
`load_prepared_structure` re-validates the embedded manifest, the array shapes
and dtypes, the span partition, the ``NaN`` padding, the finiteness of real rows,
the site permutations, and every recorded digest. That is **embedded-content
validation**: it proves the artifact is internally consistent and unmodified
since it was written. It is *not* evidence that the coordinates came from any
particular file. `verify_against_source` is the separate, explicit re-read that
goes back to the source bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np

from .declaration import (
    THREE_TO_ONE,
    ManifestValidationError,
    StructureAdapterError,
    StructureManifest,
    canonical_json,
    document_digest,
    load_manifest,
    loads_strict_json,
    revalidate_manifest,
    sha256_file,
    validate_manifest,
)
# One implementation of the "missing=... unknown=..." key-set message, shared
# with the manifest validator so an artifact and a manifest fail the same way,
# and one implementation of the portable-name rules.
from .declaration import _SHA256_RE, _check_keys, _check_portable_name, _integer, _text


__all__ = [
    "ARTIFACT_KIND",
    "ChainSpan",
    "PACKING_CONVENTION",
    "PAD_CONFIDENCE",
    "PREPARED_SCHEMA_VERSION",
    "PreparationFindings",
    "PreparedArtifactError",
    "PreparedStructure",
    "REPORT_SCHEMA_VERSION",
    "StructurePreparationError",
    "build_report",
    "check_prepared_state",
    "document_to_prepared",
    "load_prepared_structure",
    "prepare_structure",
    "prepared_to_document",
    "structure_path_for",
    "verify_against_source",
    "write_outputs",
]


PREPARED_SCHEMA_VERSION = "esmif1-prepared-structure/1"
ARTIFACT_KIND = "esmif1_prepared_structure"
REPORT_SCHEMA_VERSION = "esmif1-structure-report/1"

#: Names upstream's packing so a future change to it is a schema change here.
PACKING_CONVENTION = "esm_multichain_concatenate_v1"

#: The confidence written on an inter-chain pad row: upstream's
#: ``confidence=None`` default. The converter's padding mask overwrites it with
#: ``-1`` regardless (``esmif1_policy.py:930-941``), so the value is never read --
#: which is exactly why it has to be pinned rather than left free.
PAD_CONFIDENCE = 1.0

#: Columns the mmCIF ``atom_site`` category must carry. The four ``auth_*`` names
#: are required because `pdbx.get_structure(use_author_fields=True)` silently
#: falls back to the ``label_*`` column when an author column is absent: the
#: fallback only emits ``warnings.warn`` (``pdbx/convert.py:455-473``), which is
#: filterable and is therefore not a gate. Declaring author numbering is not
#: enough; the columns have to be there.
REQUIRED_MMCIF_COLUMNS = (
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "auth_asym_id",
    "auth_atom_id",
    "auth_comp_id",
    "auth_seq_id",
    "group_PDB",
    "label_alt_id",
    "occupancy",
    "pdbx_PDB_ins_code",
    "pdbx_PDB_model_num",
    "type_symbol",
)

#: The three backbone atoms ESM-IF1 consumes, in the order it expects them.
BACKBONE_ATOMS = ("N", "CA", "C")

#: Values that mean "this record carries no alternate-location identifier".
#: ``" "`` is what the PDB reader stores for a blank column 17
#: (``pdb/file.py:427`` does not strip); ``"."`` is an inapplicable mmCIF value.
#: ``"?"`` is deliberately absent: an *unknown* altloc is unresolved, not absent.
_NO_ALTLOC = ("", " ", ".")

#: Values that mean an identifier could not be resolved.
_UNRESOLVED = ("", ".", "?")

#: The only two record kinds an ``atom_site`` row may declare. biotite derives
#: ``hetero`` as ``group_PDB == "HETATM"`` (``pdbx/convert.py``), so any *other*
#: value -- a typo, a truncated column, an mmCIF category that is not atom_site
#: content -- would be coerced into an ordinary ATOM record.
_MMCIF_RECORD_KINDS = ("ATOM", "HETATM")

#: A model number is an integer written out in full. Parsed from the raw text
#: rather than through ``as_array(np.int32)``, which wraps 2**31 silently.
_INTEGER_RE = re.compile(r"[+-]?[0-9]+")

ARTIFACT_KEYS = (
    "arrays",
    "findings",
    "identity",
    "kind",
    "manifest",
    "packing",
    "schema_version",
    "site_permutations",
)
IDENTITY_KEYS = (
    "confidence_sha256",
    "content_sha256",
    "coordinates_sha256",
    "manifest_sha256",
    "structure_sha256",
)
PACKING_KEYS = (
    "chain_spans",
    "convention",
    "inter_chain_pad_length",
    "pad_spans",
    "total_rows",
)
CHAIN_SPAN_KEYS = ("chain_id", "end_row", "num_residues", "role", "start_row")
ARRAYS_KEYS = ("confidence", "coordinates")
ARRAY_KEYS = ("base64", "byte_order", "dtype", "order", "shape")
PERMUTATION_KEYS = (
    "declared_site_ids",
    "declared_to_policy",
    "policy_site_ids",
    "policy_to_declared",
)
FINDINGS_KEYS = (
    "context_chain_residue_counts",
    "declared_sequence_mismatches",
    "decoded_chain_residue_count",
    "numbering_gaps",
)
NUMBERING_GAP_KEYS = (
    "res_id_after",
    "res_id_before",
    "sequence_index_after",
    "sequence_index_before",
)
MISMATCH_FINDING_KEYS = (
    "expected_res_name",
    "ins_code",
    "reason",
    "res_id",
    "sequence_index",
    "sequence_residue",
)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class StructurePreparationError(StructureAdapterError):
    """A structure file could not be prepared under its declaration."""


class PreparedArtifactError(StructureAdapterError):
    """A prepared-structure artifact is invalid, inconsistent, or tampered with."""


# ---------------------------------------------------------------------------
# prepared value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainSpan:
    """Where one chain's residues sit in the packed coordinate array.

    Attributes:
        chain_id: Author chain identifier.
        role: ``"decoded"`` for the chain the policy decodes, ``"context"``
            otherwise.
        start_row: First row, inclusive.
        end_row: Last row, exclusive.
        num_residues: ``end_row - start_row``, stated so a truncated span is a
            contradiction rather than a shorter array.
    """

    chain_id: str
    role: str
    start_row: int
    end_row: int
    num_residues: int


@dataclass(frozen=True)
class PreparationFindings:
    """Observations that are reported, never repaired.

    Attributes:
        numbering_gaps: Consecutive decoded residues whose author numbers are not
            consecutive. **No spatial or completeness claim is attached**: author
            numbering is routinely discontinuous, so a gap here is evidence about
            the numbering and nothing else.
        declared_sequence_mismatches: The rows whose structural residue
            deliberately differs from the edited sequence.
        decoded_chain_residue_count: Residues observed in the decoded chain.
        context_chain_residue_counts: Residues observed in each context chain.
    """

    numbering_gaps: tuple[dict[str, int], ...]
    declared_sequence_mismatches: tuple[dict[str, Any], ...]
    decoded_chain_residue_count: int
    context_chain_residue_counts: dict[str, int]

    def to_document(self) -> dict:
        return {
            "context_chain_residue_counts": dict(self.context_chain_residue_counts),
            "declared_sequence_mismatches": [
                dict(row) for row in self.declared_sequence_mismatches
            ],
            "decoded_chain_residue_count": self.decoded_chain_residue_count,
            "numbering_gaps": [dict(gap) for gap in self.numbering_gaps],
        }


@dataclass(frozen=True)
class PreparedStructure:
    """Validated encoder inputs for one declared structure.

    Attributes:
        manifest: The declaration these inputs were built from.
        structure_sha256: Digest of the source file that was read.
        coordinates: ``(rows, 3, 3)`` float32 N/CA/C, packed in upstream's
            multichain order. ``NaN`` appears only in inter-chain pad rows.
        confidence: ``(rows,)`` float32 in ``[0, 1]``. Pad rows carry ``1.0``,
            which is upstream's ``confidence=None`` default; the padding mask
            forces them to ``-1`` inside the converter regardless
            (``esmif1_policy.py:930-941``), so the value is never read.
        chain_spans: One span per declared chain, decoded chain first.
        pad_spans: ``(start, end)`` of each inter-chain ``NaN`` block.
        findings: Observations, not repairs.
    """

    manifest: StructureManifest
    structure_sha256: str
    coordinates: np.ndarray
    confidence: np.ndarray
    chain_spans: tuple[ChainSpan, ...]
    pad_spans: tuple[tuple[int, int], ...]
    findings: PreparationFindings

    @property
    def num_rows(self) -> int:
        """Total packed rows, pad included."""
        return int(self.coordinates.shape[0])

    @property
    def decoded_span(self) -> ChainSpan:
        """The decoded chain's span. Always the first, always starting at row 0."""
        return self.chain_spans[0]

    @property
    def decoded_sequence(self) -> str:
        return self.manifest.decoded_sequence

    @property
    def decoded_rows(self) -> tuple[int, ...]:
        """Packed row index of each decoded sequence index, in order."""
        span = self.decoded_span
        return tuple(range(span.start_row, span.end_row))

    @property
    def coordinates_digest(self) -> str:
        return _array_digest(self.coordinates)

    @property
    def confidence_digest(self) -> str:
        return _array_digest(self.confidence)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParsedModel:
    """One model's atom records, in file order, with nothing filtered out."""

    chain_id: np.ndarray
    res_id: np.ndarray
    ins_code: np.ndarray
    res_name: np.ndarray
    atom_name: np.ndarray
    hetero: np.ndarray
    altloc_id: np.ndarray
    coord: np.ndarray
    occupancy: np.ndarray
    model_count: int


@dataclass(frozen=True)
class _ObservedResidue:
    chain_id: str
    res_id: int
    ins_code: str
    res_name: str
    coordinates: np.ndarray  # (3, 3) float32, N/CA/C

    @property
    def label(self) -> str:
        suffix = self.ins_code or ""
        return f"{self.chain_id}/{self.res_id}{suffix} {self.res_name}"


def _require_biotite() -> None:
    """Raise an actionable error when the optional parser is absent."""
    try:
        import biotite  # noqa: F401
    except ModuleNotFoundError as error:
        raise StructurePreparationError(
            "reading a structure file needs biotite, which ships in the optional "
            "'esm-if1' extra. Install it with: pip install -e \".[esm-if1]\""
        ) from error


def structure_path_for(manifest: StructureManifest, structure_root: Path | str) -> Path:
    """Resolve the declared ``relative_path`` to a real path below ``structure_root``.

    The manifest validator has already refused a drive-qualified, rooted, UNC or
    traversing ``relative_path``, so what is left is the indirection a string
    check cannot see: a symlink. Both sides are therefore fully resolved and the
    result is required to lie inside the resolved root.

    Returns:
        The resolved path -- the one that was checked, never the unresolved
        spelling, so nothing downstream can report an escaped path as being
        below the root.

    Raises:
        StructurePreparationError: If the declared path resolves outside
            ``structure_root``.
    """
    _revalidated(manifest)
    try:
        root = Path(structure_root).resolve()
        resolved = (root / manifest.source.relative_path).resolve()
    except OSError as error:
        raise StructurePreparationError(
            f"cannot resolve {manifest.source.relative_path!r} below {structure_root}: "
            f"{error}"
        ) from error
    if not resolved.is_relative_to(root):
        raise StructurePreparationError(
            f"declared structure {manifest.source.relative_path!r} resolves to "
            f"{resolved}, which is outside the structure root {root}. The declaration "
            "addresses one file below the root; a symlink out of it is not that file."
        )
    return resolved


def _revalidated(manifest: StructureManifest) -> StructureManifest:
    """`revalidate_manifest`, reported as a preparation failure."""
    try:
        return revalidate_manifest(manifest)
    except ManifestValidationError as error:
        raise StructurePreparationError(
            f"the manifest handed to this call is not a validated declaration: {error}"
        ) from error


def _check_source_bytes(manifest: StructureManifest, path: Path) -> str:
    """Hash the local file and refuse anything but the declared bytes."""
    try:
        if not path.is_file():
            raise StructurePreparationError(
                f"declared structure {manifest.source.relative_path!r} is not a file at "
                f"{path}."
            )
        size = path.stat().st_size
    except OSError as error:
        raise StructurePreparationError(
            f"cannot read the declared structure at {path}: {error}"
        ) from error
    if size != manifest.source.size_bytes:
        raise StructurePreparationError(
            f"{path.name} is {size} bytes but the manifest declares "
            f"{manifest.source.size_bytes}."
        )
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise StructurePreparationError(
            f"cannot read the declared structure at {path}: {error}"
        ) from error
    if digest != manifest.source.sha256:
        raise StructurePreparationError(
            f"{path.name} hashes to {digest} but the manifest declares "
            f"{manifest.source.sha256}. The declaration pins exact bytes; a different "
            "file is a different structure."
        )
    return digest


def _parse_mmcif(path: Path, manifest: StructureManifest) -> _ParsedModel:
    """Preflight the raw ``atom_site`` category, then parse the selected model."""
    from biotite.structure.io import pdbx

    try:
        cif = pdbx.CIFFile.read(str(path))
    except Exception as error:  # biotite raises several unrelated types here
        raise StructurePreparationError(
            f"cannot read mmCIF {path.name}: {error}"
        ) from error

    block_names = list(cif.keys())
    if len(block_names) != 1:
        raise StructurePreparationError(
            f"mmCIF {path.name} holds {len(block_names)} data blocks {block_names}. v1 "
            "reads single-block files only: with no block named, biotite silently "
            "returns the first one (pdbx/convert.py:444-449), which would make the "
            "selection invisible."
        )
    try:
        block = cif[block_names[0]]
        if "atom_site" not in block:
            raise StructurePreparationError(
                f"mmCIF {path.name} has no 'atom_site' category."
            )
        atom_site = block["atom_site"]
    except StructurePreparationError:
        raise
    except Exception as error:  # biotite defers parsing until a block is indexed
        raise StructurePreparationError(
            f"cannot read the 'atom_site' category of mmCIF {path.name}: {error}"
        ) from error

    missing = [name for name in REQUIRED_MMCIF_COLUMNS if name not in atom_site]
    if missing:
        raise StructurePreparationError(
            f"mmCIF {path.name} 'atom_site' is missing {missing}. The manifest declares "
            "author residue numbering, and biotite falls back to the matching 'label_*' "
            "column when an author column is absent, warning only "
            "(pdbx/convert.py:455-473). The columns are required so the fallback cannot "
            "be taken."
        )

    # Mirror `_filter_model` exactly (pdbx/convert.py:825-833): models are a
    # contiguous slice between the first occurrences of each distinct value of
    # `pdbx_PDB_model_num`, in file order. This is an ordinal, not a serial.
    model_column = atom_site["pdbx_PDB_model_num"]
    if model_column.mask is not None:
        raise StructurePreparationError(
            f"mmCIF {path.name} has unresolved 'pdbx_PDB_model_num' values; the model "
            "ordinal could not be applied."
        )
    starts = _model_run_starts(model_column.as_array(str).tolist(), path.name)
    model_count = len(starts)
    ordinal = manifest.source.model_ordinal
    if ordinal > model_count:
        raise StructurePreparationError(
            f"mmCIF {path.name} holds {model_count} model(s) but the manifest selects "
            f"ordinal {ordinal}."
        )
    bounds = starts + [int(atom_site.row_count)]
    row_slice = slice(bounds[ordinal - 1], bounds[ordinal])

    # Before the parser coerces anything: a row whose record kind is neither ATOM
    # nor HETATM would become an ordinary atom, because `hetero` is derived as
    # `group_PDB == "HETATM"`.
    group_raw = atom_site["group_PDB"].as_array(str)[row_slice].tolist()

    # The author identifier preflight runs on the raw columns, before the parser
    # can substitute anything, and on exactly the rows the parser will keep.
    chain_raw = atom_site["auth_asym_id"].as_array(str)[row_slice]
    unresolved = sorted({value for value in chain_raw.tolist() if value in _UNRESOLVED})
    if unresolved:
        raise StructurePreparationError(
            f"mmCIF {path.name} model ordinal {ordinal} has records with an unresolved "
            f"author chain identifier ({unresolved}); chain selection would be ambiguous."
        )
    # Built in Python rather than with `np.isin`, so a chain id longer than the
    # column's fixed string width cannot be truncated into a false match.
    declared_chains = set(manifest.chains.order)
    selected = np.array(
        [value in declared_chains for value in chain_raw.tolist()], dtype=bool
    )
    bad_kinds = sorted(
        {
            value
            for value, keep in zip(group_raw, selected.tolist())
            if keep and value not in _MMCIF_RECORD_KINDS
        }
    )
    if bad_kinds:
        raise StructurePreparationError(
            f"mmCIF {path.name} model ordinal {ordinal} has selected-chain records whose "
            f"group_PDB is {bad_kinds}, not one of {list(_MMCIF_RECORD_KINDS)}. biotite "
            "derives 'hetero' as group_PDB == 'HETATM', so any other value would be "
            "silently read as an ordinary atom record."
        )
    for column in ("auth_seq_id", "auth_comp_id", "auth_atom_id", "occupancy",
                   "Cartn_x", "Cartn_y", "Cartn_z"):
        values = atom_site[column].as_array(str)[row_slice][selected]
        bad = sorted({value for value in values.tolist() if value in _UNRESOLVED})
        if bad:
            raise StructurePreparationError(
                f"mmCIF {path.name} model ordinal {ordinal} has selected-chain records "
                f"with an unresolved {column} ({bad}). Unrelated chains may carry "
                "unresolved author identifiers; a selected chain may not."
            )

    try:
        array = pdbx.get_structure(
            cif, model=ordinal, altloc="all", use_author_fields=True,
            extra_fields=["occupancy"],
        )
    except Exception as error:
        raise StructurePreparationError(
            f"cannot parse mmCIF {path.name} model ordinal {ordinal}: {error}"
        ) from error
    return _parsed_model_from_atom_array(array, model_count)


def _model_run_starts(raw_serials: Sequence[str], file_name: str) -> list[int]:
    """First row of each model, requiring one contiguous run per model serial.

    biotite slices a model as the rows between the first occurrences of two
    distinct ``pdbx_PDB_model_num`` values (``pdbx/convert.py:825-833``). If a
    serial occurs in two separate runs -- model 1, then model 2, then model 1
    again -- that slice holds rows from both models, and an ordinal would select
    a mixture. It is refused rather than reordered.

    The serials are parsed from the raw text rather than through
    ``as_array(np.int32)``, which cannot represent a serial past ``2**31 - 1`` as
    itself: whatever it does with one is not the number the file wrote.
    """
    serials: list[int] = []
    for index, raw in enumerate(raw_serials):
        text = raw.strip()
        if not _INTEGER_RE.fullmatch(text):
            raise StructurePreparationError(
                f"mmCIF {file_name} row {index} has pdbx_PDB_model_num {raw!r}, which is "
                "not an integer model number; the model ordinal could not be applied."
            )
        serials.append(int(text))

    starts: list[int] = []
    first_row: dict[int, int] = {}
    for index, serial in enumerate(serials):
        if index and serial == serials[index - 1]:
            continue
        if serial in first_row:
            raise StructurePreparationError(
                f"mmCIF {file_name} model {serial} occupies two separate runs of rows "
                f"(starting at row {first_row[serial]} and row {index}). A model is one "
                "contiguous block of rows here and in biotite's own slice "
                "(pdbx/convert.py:825-833), so an ordinal over interleaved models would "
                "select coordinates from more than one of them."
            )
        first_row[serial] = index
        starts.append(index)
    if not starts:
        raise StructurePreparationError(
            f"mmCIF {file_name} 'atom_site' holds no rows."
        )
    return starts


def _check_pdb_segments(
    path: Path, manifest: StructureManifest, ordinal: int
) -> None:
    """Refuse a residue whose records sit either side of a ``TER`` or another chain.

    biotite's PDB reader keeps ``ATOM``/``HETATM`` lines only, so a ``TER`` -- the
    end of one polymer segment -- leaves no trace in the parsed arrays, and the
    records on both sides of it become contiguous rows of one residue. The raw
    lines are therefore walked here, in the selected model, and a residue that
    appears in two separate runs is named and refused rather than silently joined.
    """
    try:
        text = path.read_bytes().decode("ascii", errors="replace")
    except OSError as error:
        raise StructurePreparationError(
            f"cannot read PDB {path.name}: {error}"
        ) from error

    declared = set(manifest.chains.order)
    models_seen = 0
    current = 1  # a file with no MODEL record holds exactly one model
    segment = 0
    started: set[tuple[str, str, str]] = set()
    previous: tuple[Any, ...] | None = None
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            models_seen += 1
            current = models_seen
            segment += 1
            previous = None
            continue
        if record == "TER":
            segment += 1
            previous = None
            continue
        if record not in ("ATOM", "HETATM") or current != ordinal:
            continue
        key = (line[21:22], line[22:26].strip(), line[26:27].strip())
        state = (key, segment)
        if state == previous:
            continue
        # A new run: the residue changed, another chain's records intervened, or a
        # TER ended the segment. Every record is walked, so that an intervening
        # chain breaks a run, but only declared chains are held to the rule --
        # records on unselected chains are never examined elsewhere either.
        previous = state
        if key[0] not in declared:
            continue
        if key in started:
            chain, res_id, ins_code = key
            raise StructurePreparationError(
                f"PDB {path.name} model ordinal {ordinal}: residue "
                f"{chain}/{res_id}{ins_code} has records in two separate runs, split by a "
                "TER record or by another chain's records. Those are two segments, and "
                "v1 refuses rather than joining them into one residue."
            )
        started.add(key)


def _parse_pdb(path: Path, manifest: StructureManifest) -> _ParsedModel:
    """Parse the selected PDB model.

    A PDB ``ATOM``/``HETATM`` record carries only author identifiers, so the
    author-versus-label fallback that mmCIF has cannot arise here and the
    identifier checks run on the parsed records directly.
    """
    from biotite.structure.io import pdb

    try:
        pdb_file = pdb.PDBFile.read(str(path))
        model_count = int(pdb_file.get_model_count())
    except Exception as error:
        raise StructurePreparationError(
            f"cannot read PDB {path.name}: {error}"
        ) from error

    ordinal = manifest.source.model_ordinal
    if ordinal > model_count:
        raise StructurePreparationError(
            f"PDB {path.name} holds {model_count} model(s) but the manifest selects "
            f"ordinal {ordinal}."
        )
    _check_pdb_segments(path, manifest, ordinal)
    try:
        array = pdb_file.get_structure(model=ordinal, altloc="all", extra_fields=["occupancy"])
    except Exception as error:
        raise StructurePreparationError(
            f"cannot parse PDB {path.name} model ordinal {ordinal}: {error}"
        ) from error

    parsed = _parsed_model_from_atom_array(array, model_count)
    unresolved = sorted(
        {value for value in parsed.chain_id.tolist() if value in _UNRESOLVED}
    )
    if unresolved:
        raise StructurePreparationError(
            f"PDB {path.name} model ordinal {ordinal} has records with a blank chain "
            "identifier; chain selection would be ambiguous."
        )
    return parsed


def _parsed_model_from_atom_array(array: Any, model_count: int) -> _ParsedModel:
    """Copy biotite's annotation arrays out, without re-casting the string ones.

    Their dtypes are already fixed-width unicode; asking numpy for ``dtype=str``
    means "a unicode dtype of unspecified width", which is exactly the kind of
    conversion that could silently truncate a four-character chain id.
    """
    return _ParsedModel(
        chain_id=np.asarray(array.chain_id),
        res_id=np.asarray(array.res_id, dtype=np.int64),
        ins_code=np.asarray(array.ins_code),
        res_name=np.asarray(array.res_name),
        atom_name=np.asarray(array.atom_name),
        hetero=np.asarray(array.hetero, dtype=bool),
        altloc_id=np.asarray(array.altloc_id),
        coord=np.asarray(array.coord, dtype=np.float32),
        occupancy=np.asarray(array.occupancy, dtype=np.float64),
        model_count=model_count,
    )


def _parse_model(path: Path, manifest: StructureManifest) -> _ParsedModel:
    _require_biotite()
    if manifest.source.format == "mmcif":
        return _parse_mmcif(path, manifest)
    return _parse_pdb(path, manifest)


# ---------------------------------------------------------------------------
# residue extraction
# ---------------------------------------------------------------------------

def _observed_residues(
    parsed: _ParsedModel, manifest: StructureManifest, source_label: str
) -> dict[str, tuple[_ObservedResidue, ...]]:
    """Group each selected chain's records into residues, rejecting the rest.

    Every rejection names the record it refuses. Nothing is filtered, deduplicated
    or reconstructed: see the module docstring for the full v1 rule set.
    """
    residues: dict[str, tuple[_ObservedResidue, ...]] = {}
    for chain_id in manifest.chains.order:
        rows = np.flatnonzero(parsed.chain_id == chain_id)
        if rows.size == 0:
            present = sorted(set(parsed.chain_id.tolist()))
            raise StructurePreparationError(
                f"{source_label}: declared chain {chain_id!r} has no records in the "
                f"selected model; the model holds chains {present}."
            )
        _check_selected_chain_records(parsed, rows, chain_id, source_label)
        residues[chain_id] = _group_residues(parsed, rows, chain_id, source_label)
    return residues


def _check_selected_chain_records(
    parsed: _ParsedModel,
    rows: np.ndarray,
    chain_id: str,
    source_label: str,
) -> None:
    for row in rows.tolist():
        res_name = str(parsed.res_name[row])
        res_id = int(parsed.res_id[row])
        ins_code = str(parsed.ins_code[row])
        atom_name = str(parsed.atom_name[row])
        where = f"{chain_id}/{res_id}{ins_code} {res_name!r} atom {atom_name!r}"

        occupancy = float(parsed.occupancy[row])
        if not math.isfinite(occupancy) or not 0.0 < occupancy <= 1.0:
            raise StructurePreparationError(
                f"{source_label}: {where} has occupancy {occupancy!r}; selected "
                "atoms must have finite occupancy in (0, 1]. Zero-occupancy "
                "coordinates are not observed geometry and are never filled in."
            )
        if not np.all(np.isfinite(parsed.coord[row])):
            raise StructurePreparationError(
                f"{source_label}: {where} has a non-finite coordinate. Every "
                "selected-chain atom is checked, including atoms not encoded."
            )

        altloc = str(parsed.altloc_id[row])
        if altloc not in _NO_ALTLOC:
            raise StructurePreparationError(
                f"{source_label}: {where} carries alternate-location identifier "
                f"{altloc!r}. The manifest declares altloc='reject_any': v1 refuses "
                "alternate conformations rather than choosing one, because choosing "
                "silently is how a structure becomes a different structure."
            )
        if bool(parsed.hetero[row]):
            raise StructurePreparationError(
                f"{source_label}: {where} is a HETATM record on selected chain "
                f"{chain_id!r}. v1 rejects non-polymer content in a selected chain "
                "rather than filtering it out, so a solvent or ligand record sharing a "
                "selected chain's author id fails the preparation. Select a chain "
                "without such records, or extend the declaration deliberately."
            )
        if res_name not in THREE_TO_ONE:
            raise StructurePreparationError(
                f"{source_label}: {where} has residue name {res_name!r}, which is not "
                f"one of the canonical residues v1 supports ({sorted(THREE_TO_ONE)}). "
                "Modified and unknown residues are named and refused, never dropped."
            )
        if atom_name in _UNRESOLVED:
            raise StructurePreparationError(
                f"{source_label}: a record on chain {chain_id!r} residue {res_id} has "
                "an unresolved atom name."
            )


def _group_residues(
    parsed: _ParsedModel,
    rows: np.ndarray,
    chain_id: str,
    source_label: str,
) -> tuple[_ObservedResidue, ...]:
    """Split a chain's records into residue runs, in **file** order.

    A run breaks when the residue identity changes *or* when the rows stop being
    consecutive in the model: ``rows`` is already filtered to this chain, so two
    blocks of one residue separated by another chain's records would otherwise be
    adjacent here and merge into a single residue. The row indices are what still
    carry that separation.
    """
    runs: list[list[int]] = []
    previous_key: tuple[int, str] | None = None
    previous_row: int | None = None
    for row in rows.tolist():
        key = (int(parsed.res_id[row]), str(parsed.ins_code[row]))
        if key != previous_key or previous_row is None or row != previous_row + 1:
            runs.append([])
        previous_key, previous_row = key, row
        runs[-1].append(row)

    seen: dict[tuple[int, str], int] = {}
    residues: list[_ObservedResidue] = []
    for position, run in enumerate(runs):
        res_id = int(parsed.res_id[run[0]])
        ins_code = str(parsed.ins_code[run[0]])
        key = (res_id, ins_code)
        if key in seen:
            raise StructurePreparationError(
                f"{source_label}: residue {chain_id}/{res_id}{ins_code} appears in two "
                f"separate blocks of records (chain positions {seen[key]} and "
                f"{position}); its identity is ambiguous, so it is refused rather than "
                "merged or deduplicated."
            )
        seen[key] = position

        res_names = {str(parsed.res_name[row]) for row in run}
        if len(res_names) != 1:
            raise StructurePreparationError(
                f"{source_label}: residue {chain_id}/{res_id}{ins_code} carries more "
                f"than one residue name {sorted(res_names)}."
            )
        res_name = res_names.pop()
        residues.append(
            _ObservedResidue(
                chain_id=chain_id,
                res_id=res_id,
                ins_code=ins_code,
                res_name=res_name,
                coordinates=_backbone_coordinates(
                    parsed, run, chain_id, res_id, ins_code, res_name, source_label
                ),
            )
        )
    return tuple(residues)


def _backbone_coordinates(
    parsed: _ParsedModel,
    run: Sequence[int],
    chain_id: str,
    res_id: int,
    ins_code: str,
    res_name: str,
    source_label: str,
) -> np.ndarray:
    """Exactly one N, one CA and one C, by atom-name identity, all finite."""
    where = f"{chain_id}/{res_id}{ins_code} {res_name}"
    coordinates = np.empty((3, 3), dtype=np.float32)
    for index, atom in enumerate(BACKBONE_ATOMS):
        matches = [row for row in run if str(parsed.atom_name[row]) == atom]
        if not matches:
            raise StructurePreparationError(
                f"{source_label}: residue {where} has no {atom} atom. The manifest "
                "declares missing_backbone_atom='reject', so the residue is named and "
                "refused; it is never dropped and its coordinates are never filled in."
            )
        if len(matches) > 1:
            raise StructurePreparationError(
                f"{source_label}: residue {where} has {len(matches)} atoms named {atom}. "
                "Duplicate backbone atoms are ambiguous, and v1 refuses rather than "
                "choosing between them."
            )
        values = parsed.coord[matches[0]]
        if not np.all(np.isfinite(values)):
            raise StructurePreparationError(
                f"{source_label}: residue {where} atom {atom} has a non-finite "
                f"coordinate {values.tolist()}. NaN is reserved for inter-chain padding "
                "in this packing, so it cannot also mean a missing coordinate."
            )
        coordinates[index] = values
    return coordinates


def _check_decoded_correspondence(
    manifest: StructureManifest,
    observed: Sequence[_ObservedResidue],
    source_label: str,
) -> None:
    """Full-cover bijection onto the observed decoded chain, in file order."""
    chain_id = manifest.decoded_chain
    rows = manifest.correspondence
    if len(observed) != len(rows):
        raise StructurePreparationError(
            f"{source_label}: chain {chain_id!r} holds {len(observed)} residues but the "
            f"correspondence declares {len(rows)}. v1 has no cropping: the correspondence "
            "must cover the entire observed decoded chain."
        )

    positions = {
        (residue.res_id, residue.ins_code): index
        for index, residue in enumerate(observed)
    }
    for index, (row, residue) in enumerate(zip(rows, observed)):
        declared_key = (row.res_id, row.ins_code)
        observed_key = (residue.res_id, residue.ins_code)
        if declared_key != observed_key:
            if declared_key not in positions:
                raise StructurePreparationError(
                    f"{source_label}: correspondence[{index}] declares residue "
                    f"{chain_id}/{row.res_id}{row.ins_code}, which is not present in the "
                    "observed decoded chain."
                )
            raise StructurePreparationError(
                f"{source_label}: correspondence[{index}] declares residue "
                f"{chain_id}/{row.res_id}{row.ins_code}, but chain position {index} holds "
                f"{chain_id}/{residue.res_id}{residue.ins_code}. That residue is observed "
                f"at chain position {positions[declared_key]}; v1 supports neither "
                "reordering nor an internal crop."
            )
        if residue.res_name != row.expected_res_name:
            raise StructurePreparationError(
                f"{source_label}: correspondence[{index}] expects "
                f"{row.expected_res_name} at {chain_id}/{row.res_id}{row.ins_code}, but "
                f"the structure holds {residue.res_name}."
            )


def _numbering_gaps(
    residue_keys: Sequence[tuple[int, str]],
) -> tuple[dict[str, int], ...]:
    """Consecutive decoded residues whose author numbers are not consecutive.

    A neutral observation. Author numbering is discontinuous by convention in many
    deposited structures, so a gap here establishes nothing about the backbone.

    Takes ``(res_id, ins_code)`` pairs rather than parsed residues because the two
    sources of those pairs are required to be identical: preparation refuses
    unless every correspondence row names the residue observed at that chain
    position (`_check_decoded_correspondence`). So the gaps are a function of the
    manifest as much as of the file, and a reader with no file can re-derive them
    exactly instead of taking the artifact's word for them.
    """
    gaps: list[dict[str, int]] = []
    for index in range(len(residue_keys) - 1):
        (res_id_before, ins_before) = residue_keys[index]
        (res_id_after, ins_after) = residue_keys[index + 1]
        if ins_before or ins_after:
            continue
        if res_id_after != res_id_before + 1:
            gaps.append(
                {
                    "res_id_after": res_id_after,
                    "res_id_before": res_id_before,
                    "sequence_index_after": index + 1,
                    "sequence_index_before": index,
                }
            )
    return tuple(gaps)


def _decoded_residue_keys(manifest: StructureManifest) -> tuple[tuple[int, str], ...]:
    """``(res_id, ins_code)`` per decoded index, from the validated manifest."""
    return tuple((row.res_id, row.ins_code) for row in manifest.correspondence)


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------

def _pack(
    manifest: StructureManifest,
    residues: Mapping[str, Sequence[_ObservedResidue]],
) -> tuple[np.ndarray, np.ndarray, tuple[ChainSpan, ...], tuple[tuple[int, int], ...]]:
    """Reproduce `multichain_util._concatenate_coords` from the declared order."""
    pad_length = manifest.chains.inter_chain_pad_length
    blocks: list[np.ndarray] = []
    chain_spans: list[ChainSpan] = []
    pad_spans: list[tuple[int, int]] = []
    row = 0
    for position, chain_id in enumerate(manifest.chains.order):
        if position > 0:
            blocks.append(np.full((pad_length, 3, 3), np.nan, dtype=np.float32))
            pad_spans.append((row, row + pad_length))
            row += pad_length
        chain = residues[chain_id]
        blocks.append(np.stack([residue.coordinates for residue in chain], axis=0))
        chain_spans.append(
            ChainSpan(
                chain_id=chain_id,
                role="decoded" if position == 0 else "context",
                start_row=row,
                end_row=row + len(chain),
                num_residues=len(chain),
            )
        )
        row += len(chain)

    coordinates = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    confidence = np.full(row, PAD_CONFIDENCE, dtype=np.float32)
    for span in chain_spans:
        confidence[span.start_row:span.end_row] = np.float32(
            manifest.chains.confidence[span.chain_id]
        )
    return coordinates, confidence, tuple(chain_spans), tuple(pad_spans)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def prepare_structure(
    manifest: StructureManifest, structure_path: Path | str
) -> PreparedStructure:
    """Read one declared structure file and build its encoder inputs.

    Args:
        manifest: A manifest that already passed `validate_manifest`.
        structure_path: The local file the manifest's ``relative_path`` resolves
            to. Nothing is fetched and no other file is opened.

    Returns:
        A :class:`PreparedStructure`.

    Raises:
        StructurePreparationError: On a byte mismatch against the declaration, an
            unsupported or ambiguous file, or any of the v1 rejection rules in the
            module docstring. Every message names the record it refuses.
    """
    # The declaration is re-derived from its own snapshot first: a manifest built
    # by `dataclasses.replace` carries the old digest beside new parsed fields, and
    # everything below reads the parsed fields.
    manifest = _revalidated(manifest)
    path = Path(structure_path)
    structure_sha256 = _check_source_bytes(manifest, path)
    source_label = f"{path.name} model ordinal {manifest.source.model_ordinal}"

    parsed = _parse_model(path, manifest)
    residues = _observed_residues(parsed, manifest, source_label)
    decoded = residues[manifest.decoded_chain]
    _check_decoded_correspondence(manifest, decoded, source_label)

    coordinates, confidence, chain_spans, pad_spans = _pack(manifest, residues)
    findings = PreparationFindings(
        # From the observed residues, which `_check_decoded_correspondence` has
        # just pinned to the declared ones residue by residue.
        numbering_gaps=_numbering_gaps(
            tuple((residue.res_id, residue.ins_code) for residue in decoded)
        ),
        declared_sequence_mismatches=_declared_mismatches(manifest),
        decoded_chain_residue_count=len(decoded),
        context_chain_residue_counts={
            chain_id: len(residues[chain_id])
            for chain_id in manifest.chains.context_chains
        },
    )
    return PreparedStructure(
        manifest=manifest,
        structure_sha256=structure_sha256,
        coordinates=coordinates,
        confidence=confidence,
        chain_spans=chain_spans,
        pad_spans=pad_spans,
        findings=findings,
    )


def _declared_mismatches(manifest: StructureManifest) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "expected_res_name": row.expected_res_name,
            "ins_code": row.ins_code,
            "reason": row.mismatch["reason"],
            "res_id": row.res_id,
            "sequence_index": row.sequence_index,
            "sequence_residue": row.sequence_residue,
        }
        for row in manifest.correspondence
        if row.mismatch is not None
    )


# ---------------------------------------------------------------------------
# the portable artifact
# ---------------------------------------------------------------------------

def _array_bytes(array: np.ndarray) -> bytes:
    """Little-endian, C-order float32 bytes -- the same on any host."""
    return np.ascontiguousarray(array, dtype=np.float32).astype("<f4", copy=False).tobytes(
        order="C"
    )


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(_array_bytes(array)).hexdigest()


def _encode_array(array: np.ndarray) -> dict:
    return {
        "base64": base64.b64encode(_array_bytes(array)).decode("ascii"),
        "byte_order": "little",
        "dtype": "float32",
        "order": "C",
        "shape": [int(size) for size in array.shape],
    }


def _decode_array(document: Any, label: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    doc = _check_keys_artifact(document, ARRAY_KEYS, label)
    for key, expected in (("dtype", "float32"), ("byte_order", "little"), ("order", "C")):
        if doc[key] != expected:
            raise PreparedArtifactError(
                f"{label}.{key} is {doc[key]!r}; this schema stores {expected!r} only."
            )
    shape = doc["shape"]
    if not isinstance(shape, list) or [type(size) for size in shape] != [int] * len(shape):
        raise PreparedArtifactError(f"{label}.shape must be a list of integers.")
    if tuple(shape) != tuple(expected_shape):
        raise PreparedArtifactError(
            f"{label}.shape is {tuple(shape)} but the packing implies "
            f"{tuple(expected_shape)}."
        )
    if not isinstance(doc["base64"], str):
        raise PreparedArtifactError(f"{label}.base64 must be a string.")
    try:
        raw = base64.b64decode(doc["base64"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise PreparedArtifactError(f"{label}.base64 is not valid base64: {error}") from error
    expected_bytes = 4 * math.prod(shape) if shape else 0
    if len(raw) != expected_bytes:
        raise PreparedArtifactError(
            f"{label} decodes to {len(raw)} bytes but shape {tuple(shape)} needs "
            f"{expected_bytes}."
        )
    return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(tuple(shape))


def _check_keys_artifact(document: Any, expected: Sequence[str], label: str) -> Mapping:
    """`_check_keys` with the artifact error type."""
    try:
        return _check_keys(document, expected, label)
    except ManifestValidationError as error:
        raise PreparedArtifactError(str(error)) from error


def prepared_to_document(prepared: PreparedStructure) -> dict:
    """The portable artifact for ``prepared``: a plain JSON-serializable dict.

    Holds the validated manifest **once**, the packed arrays, the spans, the
    caller-to-policy site permutations, and digests over each of those. No
    per-row echo of the correspondence: the rows live in the embedded manifest,
    so there is nothing for a second copy to drift away from.

    The embedded manifest is a fresh copy of the validated snapshot, so editing
    the returned document cannot reach back into ``prepared``.

    Raises:
        PreparedArtifactError: If the manifest's parsed fields disagree with its
            canonical snapshot. The artifact embeds the snapshot and pins it by
            ``manifest_sha256``, while the permutations and the findings written
            beside it are derived from the parsed fields, so the two have to be
            the same declaration.
    """
    manifest = _revalidated_artifact(prepared.manifest)
    document = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "identity": {
            "confidence_sha256": prepared.confidence_digest,
            "coordinates_sha256": prepared.coordinates_digest,
            "manifest_sha256": manifest.digest,
            "structure_sha256": prepared.structure_sha256,
        },
        "manifest": manifest.document,
        "packing": {
            "chain_spans": [
                {
                    "chain_id": span.chain_id,
                    "end_row": span.end_row,
                    "num_residues": span.num_residues,
                    "role": span.role,
                    "start_row": span.start_row,
                }
                for span in prepared.chain_spans
            ],
            "convention": PACKING_CONVENTION,
            "inter_chain_pad_length": manifest.chains.inter_chain_pad_length,
            "pad_spans": [[start, end] for start, end in prepared.pad_spans],
            "total_rows": prepared.num_rows,
        },
        "arrays": {
            "confidence": _encode_array(prepared.confidence),
            "coordinates": _encode_array(prepared.coordinates),
        },
        "site_permutations": {
            "declared_site_ids": list(manifest.declared_site_ids),
            "declared_to_policy": list(manifest.declared_to_policy),
            "policy_site_ids": list(manifest.policy_site_ids),
            "policy_to_declared": list(manifest.policy_to_declared),
        },
        "findings": prepared.findings.to_document(),
    }
    document["identity"]["content_sha256"] = document_digest(document)
    return document


def document_to_prepared(document: Any) -> PreparedStructure:
    """Validate an artifact document's **embedded content** and rebuild it.

    This proves the artifact is internally consistent and unmodified since it was
    written. It is **not** evidence that the coordinates were derived from any
    particular structure file: nothing here reads the source bytes, and a
    recomputed digest of an embedded value can only agree with the embedded value
    it was computed from. Use `verify_against_source` for the source claim.

    Checks performed: strict key sets and schema version; the embedded manifest
    through the same validator a fresh manifest passes; the packing convention;
    the spans **walked in packing order**, so a permuted chain layout that still
    tiles the array is refused; array dtype, byte order, C ordering, shapes and
    lengths; ``NaN`` in every pad row and nowhere else; confidence finite, in
    ``[0, 1]``, equal to the declared per-chain value on chain rows and to the
    pad convention on pad rows; the site permutations and **every finding,
    numbering gaps included**, re-derived from the manifest; ``structure_sha256``
    required to be lowercase hex and to equal the digest the manifest pins; and
    every recorded digest recomputed.

    Raises:
        PreparedArtifactError: On any inconsistency, naming the field.
    """
    doc = _check_keys_artifact(document, ARTIFACT_KEYS, "artifact")
    if doc["schema_version"] != PREPARED_SCHEMA_VERSION:
        raise PreparedArtifactError(
            f"artifact.schema_version must be {PREPARED_SCHEMA_VERSION!r}, got "
            f"{doc['schema_version']!r}."
        )
    if doc["kind"] != ARTIFACT_KIND:
        raise PreparedArtifactError(
            f"artifact.kind must be {ARTIFACT_KIND!r}, got {doc['kind']!r}."
        )

    identity = _check_keys_artifact(doc["identity"], IDENTITY_KEYS, "artifact.identity")
    try:
        manifest = validate_manifest(doc["manifest"])
    except ManifestValidationError as error:
        raise PreparedArtifactError(
            f"the manifest embedded in the artifact is invalid: {error}"
        ) from error
    if manifest.digest != identity["manifest_sha256"]:
        raise PreparedArtifactError(
            "artifact.identity.manifest_sha256 does not match the embedded manifest."
        )

    packing = _check_keys_artifact(doc["packing"], PACKING_KEYS, "artifact.packing")
    if packing["convention"] != PACKING_CONVENTION:
        raise PreparedArtifactError(
            f"artifact.packing.convention is {packing['convention']!r}; this reader "
            f"understands {PACKING_CONVENTION!r} only."
        )
    pad_length = _artifact_int(
        packing["inter_chain_pad_length"], "artifact.packing.inter_chain_pad_length"
    )
    if pad_length != manifest.chains.inter_chain_pad_length:
        raise PreparedArtifactError(
            "artifact.packing.inter_chain_pad_length disagrees with the embedded "
            "manifest."
        )
    total_rows = _artifact_int(packing["total_rows"], "artifact.packing.total_rows")

    chain_spans = _decode_chain_spans(packing["chain_spans"], manifest)
    pad_spans = _decode_pad_spans(packing["pad_spans"])
    _check_span_partition(chain_spans, pad_spans, total_rows, pad_length)

    arrays = _check_keys_artifact(doc["arrays"], ARRAYS_KEYS, "artifact.arrays")
    coordinates = _decode_array(
        arrays["coordinates"], "artifact.arrays.coordinates", (total_rows, 3, 3)
    )
    confidence = _decode_array(
        arrays["confidence"], "artifact.arrays.confidence", (total_rows,)
    )
    _check_array_values(coordinates, confidence, chain_spans, pad_spans, manifest)

    _check_permutations(doc["site_permutations"], manifest)
    findings = _decode_findings(doc["findings"], manifest, chain_spans)

    if _array_digest(coordinates) != identity["coordinates_sha256"]:
        raise PreparedArtifactError(
            "artifact.identity.coordinates_sha256 does not match the stored coordinates."
        )
    if _array_digest(confidence) != identity["confidence_sha256"]:
        raise PreparedArtifactError(
            "artifact.identity.confidence_sha256 does not match the stored confidence."
        )
    recomputed = dict(doc)
    recomputed["identity"] = {
        key: value for key, value in identity.items() if key != "content_sha256"
    }
    if document_digest(recomputed) != identity["content_sha256"]:
        raise PreparedArtifactError(
            "artifact.identity.content_sha256 does not match the document; the artifact "
            "has been modified since it was written."
        )

    structure_sha256 = identity["structure_sha256"]
    if not isinstance(structure_sha256, str) or not _SHA256_RE.fullmatch(structure_sha256):
        raise PreparedArtifactError(
            "artifact.identity.structure_sha256 must be 64 lowercase hex characters, got "
            f"{structure_sha256!r}."
        )
    if structure_sha256 != manifest.source.sha256:
        raise PreparedArtifactError(
            f"artifact.identity.structure_sha256 is {structure_sha256} but the embedded "
            f"manifest pins {manifest.source.sha256}. Preparation only ever records the "
            "digest it has just checked against the declaration, so the two cannot "
            "differ; an artifact where they do describes a file the manifest does not "
            "declare. (This is still an embedded-consistency check: it compares two "
            "recorded values, not the original bytes.)"
        )
    return PreparedStructure(
        manifest=manifest,
        structure_sha256=structure_sha256,
        coordinates=coordinates,
        confidence=confidence,
        chain_spans=chain_spans,
        pad_spans=pad_spans,
        findings=findings,
    )


def _artifact_int(value: Any, label: str) -> int:
    try:
        return _integer(value, label)
    except ManifestValidationError as error:
        raise PreparedArtifactError(str(error)) from error


def _artifact_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PreparedArtifactError(f"{label} must be a string, got {value!r}.")
    return value


def _revalidated_artifact(manifest: StructureManifest) -> StructureManifest:
    """`revalidate_manifest`, reported as an artifact/state failure."""
    try:
        return revalidate_manifest(manifest)
    except ManifestValidationError as error:
        raise PreparedArtifactError(
            f"the prepared manifest is not a validated declaration: {error}"
        ) from error


def _decode_chain_spans(
    document: Any, manifest: StructureManifest
) -> tuple[ChainSpan, ...]:
    if not isinstance(document, list):
        raise PreparedArtifactError("artifact.packing.chain_spans must be a list.")
    if len(document) != len(manifest.chains.order):
        raise PreparedArtifactError(
            f"artifact.packing.chain_spans has {len(document)} entries but the manifest "
            f"declares {len(manifest.chains.order)} chains."
        )
    spans: list[ChainSpan] = []
    for index, span_doc in enumerate(document):
        label = f"artifact.packing.chain_spans[{index}]"
        span = _check_keys_artifact(span_doc, CHAIN_SPAN_KEYS, label)
        chain_id = span["chain_id"]
        if chain_id != manifest.chains.order[index]:
            raise PreparedArtifactError(
                f"{label}.chain_id is {chain_id!r} but the manifest packs "
                f"{manifest.chains.order[index]!r} at that position."
            )
        expected_role = "decoded" if index == 0 else "context"
        if span["role"] != expected_role:
            raise PreparedArtifactError(
                f"{label}.role is {span['role']!r}; position {index} is {expected_role!r}."
            )
        start_row = _artifact_int(span["start_row"], f"{label}.start_row")
        end_row = _artifact_int(span["end_row"], f"{label}.end_row")
        num_residues = _artifact_int(span["num_residues"], f"{label}.num_residues")
        if start_row < 0 or end_row <= start_row:
            raise PreparedArtifactError(
                f"{label} spans rows [{start_row}, {end_row}), which is empty or negative."
            )
        if num_residues != end_row - start_row:
            raise PreparedArtifactError(
                f"{label}.num_residues is {num_residues} but the span covers "
                f"{end_row - start_row} rows."
            )
        spans.append(
            ChainSpan(
                chain_id=chain_id,
                role=expected_role,
                start_row=start_row,
                end_row=end_row,
                num_residues=num_residues,
            )
        )
    if spans[0].start_row != 0:
        raise PreparedArtifactError(
            "the decoded chain must start at row 0; the decoder's position p is the "
            "decoded chain's residue p by upstream convention "
            "(esm/inverse_folding/multichain_util.py:126-133)."
        )
    if spans[0].num_residues != manifest.num_decoded_residues:
        raise PreparedArtifactError(
            f"the decoded span covers {spans[0].num_residues} residues but the manifest's "
            f"decoded sequence is {manifest.num_decoded_residues} long."
        )
    return tuple(spans)


def _decode_pad_spans(document: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(document, list):
        raise PreparedArtifactError("artifact.packing.pad_spans must be a list.")
    spans: list[tuple[int, int]] = []
    for index, entry in enumerate(document):
        label = f"artifact.packing.pad_spans[{index}]"
        if not isinstance(entry, list) or len(entry) != 2:
            raise PreparedArtifactError(f"{label} must be a [start, end] pair.")
        start = _artifact_int(entry[0], f"{label}[0]")
        end = _artifact_int(entry[1], f"{label}[1]")
        if start < 0 or end <= start:
            raise PreparedArtifactError(f"{label} spans [{start}, {end}), which is empty.")
        spans.append((start, end))
    return tuple(spans)


def _check_span_partition(
    chain_spans: Sequence[ChainSpan],
    pad_spans: Sequence[tuple[int, int]],
    total_rows: int,
    pad_length: int,
) -> None:
    """The spans must tile ``[0, total_rows)`` **in the declared packing order**.

    Walked, never sorted. Sorting the union only asks whether the intervals tile
    the array, which any permutation of the chains does: decoded, pad, context
    and decoded, context, pad tile identically, and so do two context chains with
    their row ranges exchanged. Walking asks the question the artifact actually
    claims -- that chain *i* of ``chains.order`` occupies the rows recorded for
    it, with one pad block before each further chain
    (``esm/inverse_folding/multichain_util.py:68-77``).
    """
    if len(pad_spans) != len(chain_spans) - 1:
        raise PreparedArtifactError(
            f"{len(pad_spans)} pad spans for {len(chain_spans)} chains; upstream inserts "
            "exactly one pad block between consecutive chains."
        )
    for start, end in pad_spans:
        if end - start != pad_length:
            raise PreparedArtifactError(
                f"pad span [{start}, {end}) is {end - start} rows but the declared "
                f"inter-chain pad length is {pad_length}."
            )

    cursor = 0
    for position, span in enumerate(chain_spans):
        if position > 0:
            start, end = pad_spans[position - 1]
            if start != cursor:
                raise PreparedArtifactError(
                    f"the spans do not tile the array in packing order: the pad block "
                    f"before chain {span.chain_id!r} starts at row {start}, but the "
                    f"previous chain ends at row {cursor}."
                )
            cursor = end
        if span.start_row != cursor:
            raise PreparedArtifactError(
                f"the spans do not tile the array in packing order: chain "
                f"{span.chain_id!r} is packed at position {position} and must start at "
                f"row {cursor}, but its span starts at row {span.start_row}."
            )
        cursor = span.end_row
    if cursor != total_rows:
        raise PreparedArtifactError(
            f"the spans cover {cursor} rows but total_rows is {total_rows}."
        )


def _check_array_values(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    chain_spans: Sequence[ChainSpan],
    pad_spans: Sequence[tuple[int, int]],
    manifest: StructureManifest,
) -> None:
    for span in chain_spans:
        block = coordinates[span.start_row:span.end_row]
        if not np.all(np.isfinite(block)):
            raise PreparedArtifactError(
                f"chain {span.chain_id!r} has non-finite coordinates; in this packing NaN "
                "means inter-chain padding and nothing else."
            )
        values = confidence[span.start_row:span.end_row]
        declared = np.float32(manifest.chains.confidence[span.chain_id])
        if not np.all(values == declared):
            raise PreparedArtifactError(
                f"chain {span.chain_id!r} confidence does not equal the declared "
                f"{float(declared)}."
            )
    for start, end in pad_spans:
        if not np.all(np.isnan(coordinates[start:end])):
            raise PreparedArtifactError(
                f"pad span [{start}, {end}) is not entirely NaN; upstream derives the "
                "padding mask from the N atom's x coordinate, so a finite pad row would "
                "enter attention as a real residue."
            )
        if not np.all(confidence[start:end] == np.float32(PAD_CONFIDENCE)):
            raise PreparedArtifactError(
                f"pad span [{start}, {end}) does not carry the declared pad confidence "
                f"{PAD_CONFIDENCE}; that is upstream's confidence=None default, and a "
                "different value here would describe a packing this reader did not write."
            )
    if not np.all(np.isfinite(confidence)):
        raise PreparedArtifactError("confidence contains a non-finite value.")
    if float(confidence.min()) < 0.0 or float(confidence.max()) > 1.0:
        raise PreparedArtifactError("confidence values must lie in [0, 1].")


def _check_permutations(document: Any, manifest: StructureManifest) -> None:
    doc = _check_keys_artifact(document, PERMUTATION_KEYS, "artifact.site_permutations")
    expected = {
        "declared_site_ids": list(manifest.declared_site_ids),
        "declared_to_policy": list(manifest.declared_to_policy),
        "policy_site_ids": list(manifest.policy_site_ids),
        "policy_to_declared": list(manifest.policy_to_declared),
    }
    for key in PERMUTATION_KEYS:
        if not isinstance(doc[key], list):
            raise PreparedArtifactError(
                f"artifact.site_permutations.{key} must be a list."
            )
    # An index is an integer, checked as one: `[True, False] == [1, 0]`, so a
    # permutation written with booleans would compare equal to the real one.
    for key in ("declared_to_policy", "policy_to_declared"):
        for index, value in enumerate(doc[key]):
            _artifact_int(value, f"artifact.site_permutations.{key}[{index}]")
    for key in ("declared_site_ids", "policy_site_ids"):
        for index, value in enumerate(doc[key]):
            _artifact_str(value, f"artifact.site_permutations.{key}[{index}]")
    for key, value in expected.items():
        if doc[key] != value:
            raise PreparedArtifactError(
                f"artifact.site_permutations.{key} is {doc[key]!r} but the embedded "
                f"manifest implies {value!r}. The permutation is what keeps a caller's "
                "site order attached to the policy's positional order, so a stale copy "
                "would silently relabel every allele."
            )


def _decode_findings(
    document: Any, manifest: StructureManifest, chain_spans: Sequence[ChainSpan]
) -> PreparationFindings:
    doc = _check_keys_artifact(document, FINDINGS_KEYS, "artifact.findings")
    for key in ("declared_sequence_mismatches", "numbering_gaps"):
        if not isinstance(doc[key], list):
            raise PreparedArtifactError(f"artifact.findings.{key} must be a list.")
    # Types before values: `True == 1`, so a count or an index of the wrong type
    # would otherwise compare equal to the derived one.
    _check_finding_integers(doc, "artifact.findings")

    expected_mismatches = [dict(row) for row in _declared_mismatches(manifest)]
    if doc["declared_sequence_mismatches"] != expected_mismatches:
        raise PreparedArtifactError(
            "artifact.findings.declared_sequence_mismatches disagrees with the embedded "
            "manifest."
        )

    if doc["decoded_chain_residue_count"] != chain_spans[0].num_residues:
        raise PreparedArtifactError(
            "artifact.findings.decoded_chain_residue_count disagrees with the decoded span."
        )
    expected_counts = {
        span.chain_id: span.num_residues for span in chain_spans if span.role == "context"
    }
    if doc["context_chain_residue_counts"] != expected_counts:
        raise PreparedArtifactError(
            "artifact.findings.context_chain_residue_counts disagrees with the chain spans."
        )

    # Numbering gaps are re-derived, not merely shape-checked: the correspondence
    # carries every decoded residue's (res_id, ins_code), and preparation refuses
    # unless those equal the observed ones, so the embedded manifest determines
    # the gap list exactly. A fabricated or omitted gap is therefore a detectable
    # disagreement rather than a claim about a file the reader cannot see.
    recorded: list[dict[str, int]] = [
        {key: gap[key] for key in NUMBERING_GAP_KEYS} for gap in doc["numbering_gaps"]
    ]
    expected_gaps = [dict(gap) for gap in _numbering_gaps(_decoded_residue_keys(manifest))]
    if recorded != expected_gaps:
        raise PreparedArtifactError(
            f"artifact.findings.numbering_gaps is {recorded!r} but the embedded "
            f"correspondence implies {expected_gaps!r}. A gap is a statement about the "
            "author numbering the manifest itself declares, so it cannot be invented or "
            "dropped in the artifact."
        )

    return PreparationFindings(
        numbering_gaps=tuple(expected_gaps),
        declared_sequence_mismatches=tuple(expected_mismatches),
        decoded_chain_residue_count=chain_spans[0].num_residues,
        context_chain_residue_counts=expected_counts,
    )


def load_prepared_structure(path: Path | str) -> PreparedStructure:
    """Read an artifact file and validate its embedded content.

    See `document_to_prepared` for exactly what is and is not established.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PreparedArtifactError(f"cannot read artifact {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise PreparedArtifactError(
            f"artifact {path} is not valid UTF-8: {error}"
        ) from error
    try:
        document = loads_strict_json(text, label=f"artifact {path}")
    except ManifestValidationError as error:
        raise PreparedArtifactError(str(error)) from error
    return document_to_prepared(document)


def check_prepared_state(prepared: PreparedStructure) -> None:
    """Re-check one `PreparedStructure`'s own invariants, as it stands right now.

    `prepare_structure` and `document_to_prepared` both establish these on the way
    out, but a `PreparedStructure` is an ordinary object: `dataclasses.replace`
    builds a new one from arbitrary parts, and a numpy array is writeable. So
    anything about to *encode* one or *report* on one re-checks it here instead of
    trusting where it came from.

    The checks are the ones an artifact reload performs, on the live object: the
    manifest against its own canonical snapshot, the recorded source digest
    against the one the manifest pins, the spans as real non-negative integers,
    the arrays, and every finding re-derived from the manifest. A state that would
    be refused on reload is refused here too.

    Raises:
        PreparedArtifactError: Naming the first invariant that fails.
    """
    manifest = _revalidated_artifact(prepared.manifest)
    structure_sha256 = prepared.structure_sha256
    if not isinstance(structure_sha256, str) or not _SHA256_RE.fullmatch(structure_sha256):
        raise PreparedArtifactError(
            "structure_sha256 must be 64 lowercase hex characters, got "
            f"{structure_sha256!r}."
        )
    if structure_sha256 != manifest.source.sha256:
        raise PreparedArtifactError(
            f"structure_sha256 is {structure_sha256} but the manifest pins "
            f"{manifest.source.sha256}. Preparation only ever records the digest it has "
            "just checked against the declaration, so the two cannot differ."
        )
    coordinates, confidence = prepared.coordinates, prepared.confidence
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
        raise PreparedArtifactError(
            f"coordinates have shape {tuple(coordinates.shape)}; the encoder takes "
            "(rows, 3, 3) N/CA/C."
        )
    if coordinates.dtype != np.float32 or confidence.dtype != np.float32:
        raise PreparedArtifactError(
            f"coordinates and confidence must both be float32, got "
            f"{coordinates.dtype} and {confidence.dtype}."
        )
    if confidence.shape != (prepared.num_rows,):
        raise PreparedArtifactError(
            f"confidence has shape {tuple(confidence.shape)} but there are "
            f"{prepared.num_rows} packed rows."
        )

    order = manifest.chains.order
    if tuple(span.chain_id for span in prepared.chain_spans) != order:
        raise PreparedArtifactError(
            f"the chain spans are {[span.chain_id for span in prepared.chain_spans]} but "
            f"the manifest packs {list(order)}."
        )
    expected_roles = ("decoded",) + ("context",) * (len(order) - 1)
    if tuple(span.role for span in prepared.chain_spans) != expected_roles:
        raise PreparedArtifactError(
            f"the chain span roles are {[span.role for span in prepared.chain_spans]}; "
            f"the packing implies {list(expected_roles)}."
        )
    for span in prepared.chain_spans:
        label = f"chain {span.chain_id!r}"
        start_row = _artifact_int(span.start_row, f"{label} start_row")
        end_row = _artifact_int(span.end_row, f"{label} end_row")
        num_residues = _artifact_int(span.num_residues, f"{label} num_residues")
        if start_row < 0 or end_row <= start_row:
            raise PreparedArtifactError(
                f"{label} spans rows [{start_row}, {end_row}), which is empty or negative."
            )
        if num_residues != end_row - start_row:
            raise PreparedArtifactError(
                f"chain {span.chain_id!r} claims {num_residues} residues but its "
                f"span covers {end_row - start_row} rows."
            )
    for index, entry in enumerate(prepared.pad_spans):
        label = f"pad span {index}"
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise PreparedArtifactError(f"{label} must be a (start, end) pair.")
        start = _artifact_int(entry[0], f"{label}[0]")
        end = _artifact_int(entry[1], f"{label}[1]")
        if start < 0 or end <= start:
            raise PreparedArtifactError(f"{label} spans [{start}, {end}), which is empty.")
    if prepared.decoded_span.num_residues != manifest.num_decoded_residues:
        raise PreparedArtifactError(
            f"the decoded span covers {prepared.decoded_span.num_residues} residues but "
            f"the decoded sequence is {manifest.num_decoded_residues} long."
        )

    _check_span_partition(
        prepared.chain_spans,
        prepared.pad_spans,
        prepared.num_rows,
        manifest.chains.inter_chain_pad_length,
    )
    _check_array_values(
        coordinates, confidence, prepared.chain_spans, prepared.pad_spans, manifest
    )
    _check_findings(prepared.findings, manifest, prepared.chain_spans)


def _check_finding_integers(document: Mapping, label: str) -> None:
    """Every count and index in a findings document is an integer, never a bool.

    ``True == 1`` and ``1.0 == 1``, so a findings block whose counts are booleans
    or floats compares equal to the derived one. The types are checked here, and
    the values are compared by the caller.
    """
    _artifact_int(
        document["decoded_chain_residue_count"], f"{label}.decoded_chain_residue_count"
    )
    counts = document["context_chain_residue_counts"]
    if not isinstance(counts, Mapping):
        raise PreparedArtifactError(f"{label}.context_chain_residue_counts must be an object.")
    for chain_id, count in counts.items():
        _artifact_str(chain_id, f"{label}.context_chain_residue_counts key")
        _artifact_int(count, f"{label}.context_chain_residue_counts[{chain_id!r}]")
    for index, row in enumerate(document["declared_sequence_mismatches"]):
        row_label = f"{label}.declared_sequence_mismatches[{index}]"
        checked = _check_keys_artifact(row, MISMATCH_FINDING_KEYS, row_label)
        _artifact_int(checked["res_id"], f"{row_label}.res_id")
        _artifact_int(checked["sequence_index"], f"{row_label}.sequence_index")
        for key in ("expected_res_name", "ins_code", "reason", "sequence_residue"):
            _artifact_str(checked[key], f"{row_label}.{key}")
    for index, gap in enumerate(document["numbering_gaps"]):
        gap_label = f"{label}.numbering_gaps[{index}]"
        checked_gap = _check_keys_artifact(gap, NUMBERING_GAP_KEYS, gap_label)
        for key in NUMBERING_GAP_KEYS:
            _artifact_int(checked_gap[key], f"{gap_label}.{key}")


def _check_findings(
    findings: Any, manifest: StructureManifest, chain_spans: Sequence[ChainSpan]
) -> None:
    """The findings of a live object, re-derived exactly as a reload re-derives them."""
    if not isinstance(findings, PreparationFindings):
        raise PreparedArtifactError(
            f"findings must be a PreparationFindings, got {type(findings).__name__}."
        )
    try:
        document = findings.to_document()
    except (AttributeError, TypeError, ValueError) as error:
        raise PreparedArtifactError(f"the findings cannot be read: {error}") from error
    # Counts and indices are compared by value below, and `True == 1`, so the types
    # are checked rather than left to the comparison.
    _check_finding_integers(document, "findings")
    expected = PreparationFindings(
        numbering_gaps=_numbering_gaps(_decoded_residue_keys(manifest)),
        declared_sequence_mismatches=_declared_mismatches(manifest),
        decoded_chain_residue_count=chain_spans[0].num_residues,
        context_chain_residue_counts={
            span.chain_id: span.num_residues
            for span in chain_spans
            if span.role == "context"
        },
    ).to_document()
    if document != expected:
        raise PreparedArtifactError(
            f"the findings are {document!r} but the manifest and the chain spans imply "
            f"{expected!r}. A finding is re-derivable from the declaration, so it cannot "
            "be invented, dropped or left stale on the object."
        )


def verify_against_source(
    prepared: PreparedStructure, structure_path: Path | str
) -> tuple[str, ...]:
    """Re-read the source bytes and re-derive the arrays, explicitly.

    This is the check `load_prepared_structure` deliberately does **not** perform:
    it opens the structure file again, re-runs the whole preparation, and compares
    the source digest, both array digests, the chain and pad spans, and the
    findings. Only this establishes that the artifact describes that file.

    Returns:
        One human-readable problem per disagreement; empty when they agree.
    """
    problems: list[str] = []
    try:
        # The re-preparation below reads the manifest's parsed fields, so those are
        # required to be the snapshot the artifact is pinned by first; otherwise
        # this would re-derive one declaration and compare it against another.
        revalidate_manifest(prepared.manifest)
    except ManifestValidationError as error:
        return (f"the prepared manifest disagrees with its canonical snapshot: {error}",)
    try:
        rebuilt = prepare_structure(prepared.manifest, structure_path)
    except StructureAdapterError as error:
        return (f"re-preparing from {Path(structure_path).name} failed: {error}",)
    if rebuilt.structure_sha256 != prepared.structure_sha256:
        problems.append(
            f"source sha256 {rebuilt.structure_sha256} != artifact "
            f"{prepared.structure_sha256}"
        )
    if rebuilt.coordinates_digest != prepared.coordinates_digest:
        problems.append("re-derived coordinates differ from the artifact's")
    if rebuilt.confidence_digest != prepared.confidence_digest:
        problems.append("re-derived confidence differs from the artifact's")
    if rebuilt.chain_spans != prepared.chain_spans:
        problems.append("re-derived chain spans differ from the artifact's")
    if rebuilt.pad_spans != prepared.pad_spans:
        problems.append("re-derived pad spans differ from the artifact's")
    if rebuilt.findings != prepared.findings:
        problems.append("re-derived findings differ from the artifact's")
    return tuple(problems)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

#: Structural checks, in the order they run, each with the formats it applies to.
#: `build_report` re-verifies all of them against the named files before it will
#: emit one, so the report enumerates what was checked *at report time* rather
#: than what some earlier call is assumed to have checked. The format filter
#: keeps a PDB report from claiming an mmCIF-only check passed.
_ALL_FORMATS = ("pdb", "mmcif")
STRUCTURAL_CHECKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "source_bytes_match_declaration",
        "file size and SHA-256 equal the declared values",
        _ALL_FORMATS,
    ),
    (
        "declared_format_read",
        "parsed as the declared format; the extension was not consulted",
        _ALL_FORMATS,
    ),
    (
        "author_columns_present",
        "atom_site carries every required auth_* column, so the parser's silent label_* "
        "fallback cannot be taken",
        ("mmcif",),
    ),
    (
        "single_data_block",
        "the file holds exactly one data block, so block selection is not implicit",
        ("mmcif",),
    ),
    (
        "model_ordinal_selected",
        "the model was selected by 1-based position in file order, not by MODEL serial",
        _ALL_FORMATS,
    ),
    (
        "author_identifiers_resolved",
        "no record in the selected model has an unresolved author chain identifier, and "
        "no selected-chain record has an unresolved residue number, residue name or atom name",
        _ALL_FORMATS,
    ),
    (
        "selected_chain_content_supported",
        "every selected-chain record is a non-hetero canonical amino acid with no "
        "alternate-location identifier, finite coordinates, and occupancy in (0, 1]; "
        "unsupported content is rejected, never filtered",
        _ALL_FORMATS,
    ),
    (
        "backbone_atoms_exact",
        "every mapped residue has exactly one N, one CA and one C by atom-name identity, "
        "all finite",
        _ALL_FORMATS,
    ),
    (
        "correspondence_covers_decoded_chain",
        "the correspondence maps the entire observed decoded chain in file residue order, "
        "with no omission, reordering or crop",
        _ALL_FORMATS,
    ),
    (
        "declared_residue_identities_match",
        "each mapped residue's name equals the declared expected_res_name",
        _ALL_FORMATS,
    ),
    (
        "declared_mismatches_consistent",
        "every sequence mismatch is declared and every declared mismatch is real",
        _ALL_FORMATS,
    ),
    (
        "edit_space_within_decoded_sequence",
        "each site's index lies in the decoded sequence and the context residue is inside "
        "that site's two-residue support",
        _ALL_FORMATS,
    ),
    (
        "packing_follows_upstream_convention",
        "decoded chain first, then a NaN pad block before each further chain "
        "(esm/inverse_folding/multichain_util.py:68-77)",
        _ALL_FORMATS,
    ),
)

#: What a structural preparation cannot say anything about. Named individually so
#: "not run" is a list of specific absent measurements rather than a disclaimer.
MODEL_INTEGRATION_CHECKS: tuple[str, ...] = (
    "cached_geometry_matches_uncached_native_forward",
    "sample_and_rescore_agree",
    "decoder_gradient_with_frozen_encoder",
    "released_checkpoint_parity",
)


def build_report(
    prepared: PreparedStructure,
    *,
    manifest_path: Path | str,
    structure_path: Path | str,
    artifact_filename: str,
    artifact_content_sha256: str,
) -> dict:
    """A deterministic JSON report for one preparation, re-verified as it is built.

    A report's only content is its claims, so this function does not take
    ``prepared``'s word for any of them. Before it returns anything it:

    1. re-checks ``prepared``'s own invariants (`check_prepared_state`), so a
       hand-built or mutated object cannot be reported on;
    2. **reads the manifest file at** ``manifest_path`` and requires it to be this
       manifest -- the report names that file, so the file has to be that one;
    3. **re-reads** ``structure_path`` **and re-runs the whole preparation**
       (`verify_against_source`), so every source-side check in
       `STRUCTURAL_CHECKS` is one that has just been executed against the file the
       report names -- recomputed digests inside a mutated object cannot stand in
       for it;
    4. recomputes the artifact's ``content_sha256`` and requires
       ``artifact_content_sha256`` to equal it.

    Any disagreement raises instead of producing a report that says ``pass``.
    The cost is one extra parse of the structure file per report, which is the
    price of the report meaning what it says.

    No timestamps are generated (the house rule from
    `benchmarks/provenance.py`). Input paths are recorded **as given**, because
    they are inputs; the artifact is recorded by basename and digest, so two runs
    with the same arguments into different output directories compare
    byte-identical.

    The report separates what was checked here from what was not run at all: this
    tool supplies no model and loads no weights, so every model-integration check
    is ``not_run`` with a reason, never ``pass``.

    Raises:
        StructureAdapterError: If the prepared state is inconsistent, the named
            manifest file is unreadable or is a different manifest, the named
            structure file does not reproduce this preparation, the artifact
            filename is not a plain portable basename, or the artifact digest
            does not match.
    """
    check_prepared_state(prepared)

    filename = _text_or_error(artifact_filename, "artifact filename")
    if Path(filename).name != filename:
        raise StructurePreparationError(
            f"artifact filename {artifact_filename!r} is not a plain basename; the "
            "report records the artifact by name and digest, not by path."
        )
    try:
        _check_portable_name(filename, "artifact filename")
    except ManifestValidationError as error:
        raise StructurePreparationError(str(error)) from error

    declared = load_manifest(manifest_path)
    if declared.digest != prepared.manifest.digest:
        raise StructurePreparationError(
            f"the manifest at {manifest_path} hashes to {declared.digest} but these "
            f"inputs were prepared from {prepared.manifest.digest}. The report would "
            "name a file that does not declare what it describes."
        )

    problems = verify_against_source(prepared, structure_path)
    if problems:
        raise StructurePreparationError(
            "the prepared inputs do not match the structure file this report would "
            f"name ({structure_path}): " + "; ".join(problems) + ". Every source-side "
            "check in this report is re-run here, so a report cannot claim one passed "
            "for inputs the file does not reproduce."
        )

    expected_content = prepared_to_document(prepared)["identity"]["content_sha256"]
    if artifact_content_sha256 != expected_content:
        raise StructurePreparationError(
            f"artifact_content_sha256 is {artifact_content_sha256!r} but this "
            f"preparation serializes to {expected_content}. The report identifies the "
            "artifact by that digest, so it is recomputed rather than accepted."
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "preparation_status": "pass",
        "inputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": prepared.manifest.digest,
            "structure_path": str(structure_path),
            "structure_sha256": prepared.structure_sha256,
        },
        "artifact": {
            "content_sha256": artifact_content_sha256,
            "filename": filename,
        },
        "structural_checks_method": (
            "re-verified while this report was built: the manifest file was re-read and "
            "re-hashed, the structure file was re-read and the whole preparation re-run, "
            "and both had to reproduce these inputs exactly. No report is written "
            "otherwise, so a 'pass' below is a check that ran against the files named in "
            "'inputs'."
        ),
        "declaration": {
            "decoded_chain": prepared.manifest.decoded_chain,
            "decoded_residues": prepared.manifest.num_decoded_residues,
            "chain_order": list(prepared.manifest.chains.order),
            "declared_site_ids": list(prepared.manifest.declared_site_ids),
            "policy_site_ids": list(prepared.manifest.policy_site_ids),
            "structural_relationship": prepared.manifest.relationship.kind,
            "total_packed_rows": prepared.num_rows,
        },
        "structural_checks": [
            {"check": name, "detail": detail, "status": "pass"}
            for name, detail, formats in STRUCTURAL_CHECKS
            if prepared.manifest.source.format in formats
        ],
        "findings": prepared.findings.to_document(),
        "model_integration_checks": {
            "checks": [
                {"check": name, "status": "not_run"} for name in MODEL_INTEGRATION_CHECKS
            ],
            "reason": (
                "this tool prepares encoder inputs only: it supplies no model, loads no "
                "weights, and runs no forward or backward pass. Nothing here is evidence "
                "about a structure's biology or about a released checkpoint."
            ),
            "status": "not_run",
        },
        "environment": _environment(),
    }


def _environment() -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import biotite

        versions["biotite"] = biotite.__version__
    except ModuleNotFoundError:  # pragma: no cover - prepare would have failed first
        versions["biotite"] = "absent"
    return versions


# ---------------------------------------------------------------------------
# writing, without overwriting anything
# ---------------------------------------------------------------------------

def _validate_output_name(name: str) -> str:
    """A plain, portable file stem: one name, one file, on every platform."""
    text = _text_or_error(name, "output name")
    if "/" in text or "\\" in text or text in (".", "..") or ".." in Path(text).parts:
        raise StructurePreparationError(
            f"output name {name!r} must be a plain file stem: no path separators, no "
            "traversal. The output directory is chosen with its own argument."
        )
    if PureWindowsPath(text).drive or PureWindowsPath(text).root:
        raise StructurePreparationError(
            f"output name {name!r} is drive-qualified or rooted; it must be a plain "
            "file stem."
        )
    if Path(text).name != text:
        raise StructurePreparationError(f"output name {name!r} is not a plain file stem.")
    # The same portability rules the manifest's relative_path is held to: ':'
    # would address an alternate data stream, and `CON.prepared.json` is the
    # console device rather than a file.
    try:
        _check_portable_name(text, "output name")
    except ManifestValidationError as error:
        raise StructurePreparationError(str(error)) from error
    return text


def _utf8_bytes(text: str, label: str) -> bytes:
    """The UTF-8 bytes of a canonical document, or a named refusal.

    A lone surrogate is a legal `str` and survives `json.dumps`; it fails only at
    the encode. Doing that here means the failure happens before any path exists.
    """
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise StructurePreparationError(
            f"the {label} document is not encodable as UTF-8: {error}. Nothing was "
            "created: every file this tool writes is UTF-8, so a document that cannot "
            "be encoded is refused before the output directory is made."
        ) from error


def _text_or_error(value: Any, label: str) -> str:
    try:
        return _text(value, label)
    except ManifestValidationError as error:
        raise StructurePreparationError(str(error)) from error


def write_outputs(
    artifact_document: Mapping,
    report_document: Mapping,
    *,
    output_dir: Path | str,
    name: str,
    input_paths: Sequence[Path | str] = (),
) -> tuple[Path, Path]:
    """Write the artifact and the report, never overwriting anything.

    Two sequential writes cannot make "a failure writes nothing" an absolute, so
    the contract is the achievable one: **inputs are never touched, an existing
    output is never replaced, and any file this call created is removed again if a
    later step fails.**

    - the name must be a plain, portable file stem; separators, traversal,
      drive/stream syntax and Windows device names are refused;
    - **both documents are serialized and encoded to UTF-8 bytes before any file
      is created**, so the one failure that is not an OS error -- a document that
      cannot be encoded -- happens while nothing exists yet, the output directory
      included. Serializing to text is not enough: a lone surrogate survives
      ``json.dumps`` and fails at the encode, which used to be *inside* the write;
    - both outputs are resolved and compared against every input path, so an
      output can never alias an input;
    - both outputs are preflighted for non-existence, then created with ``"xb"``,
      which is atomic and refuses a dangling symlink as well as a real file;
    - only a path this call **created** is ever unlinked, and only after an OS
      error while writing;
    - there is no ``--force`` and no overwrite path at all.

    Returns:
        ``(artifact_path, report_path)``.

    Raises:
        StructurePreparationError: On a bad name, a document that is not UTF-8
            encodable, an aliased or existing output, or an OS error while
            writing.
        TypeError | ValueError: Raised **unwrapped** by the JSON serialization
            step, before anything is created, so a caller sees the object that
            could not be encoded rather than a path error it did not cause.
    """
    stem = _validate_output_name(name)
    # Before anything is created, and before the directory is even made: neither
    # an unserializable document nor one that is not UTF-8 encodable may leave a
    # file behind.
    artifact_bytes = _utf8_bytes(canonical_json(artifact_document), "artifact")
    report_bytes = _utf8_bytes(canonical_json(report_document), "report")

    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StructurePreparationError(
            f"cannot create output directory {directory}: {error}"
        ) from error

    artifact_path = directory / f"{stem}.prepared.json"
    report_path = directory / f"{stem}.report.json"

    resolved_inputs = {Path(path).resolve(): Path(path) for path in input_paths}
    for output in (artifact_path, report_path):
        resolved = output.resolve()
        if resolved in resolved_inputs:
            raise StructurePreparationError(
                f"output {output} resolves to the input {resolved_inputs[resolved]}; "
                "this tool never writes over its own inputs."
            )
    if artifact_path.resolve() == report_path.resolve():
        raise StructurePreparationError(
            f"the artifact and the report would be the same file ({artifact_path})."
        )
    for output in (artifact_path, report_path):
        if output.exists() or output.is_symlink():
            raise StructurePreparationError(
                f"{output} already exists. There is no overwrite option: choose another "
                "--name or another --output-dir, and remove the old file deliberately if "
                "you meant to replace it."
            )

    created: list[Path] = []
    try:
        for output, payload in (
            (artifact_path, artifact_bytes),
            (report_path, report_bytes),
        ):
            # Written as bytes that were encoded before anything existed: the
            # encode cannot fail here, and "xb" performs no newline translation,
            # so the file is the canonical text exactly.
            with open(output, "xb") as handle:
                # Appended only once "x" has succeeded, so `created` holds exactly
                # the files this call brought into existence and a pre-existing
                # file can never be unlinked here.
                created.append(output)
                handle.write(payload)
    except OSError as error:
        for path in created:
            try:
                path.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        raise StructurePreparationError(f"cannot write outputs: {error}") from error
    return artifact_path, report_path
