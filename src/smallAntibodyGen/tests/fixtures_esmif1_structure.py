"""Generated synthetic structures and manifests for the ESM-IF1 structural adapter.

WHAT THIS IS
------------
A writer, not a corpus. Every file the adapter tests read is produced here from
an explicit residue list, so the bad variants (an insertion code, an alternate
location, a missing backbone atom, a water record sharing a selected chain, a
second data block, an mmCIF with no author columns, a second model) are one
keyword argument apart from the good one. Nothing is committed as a static file
and nothing is derived from a deposited structure.

THESE ARE NOT PROTEINS
----------------------
The coordinates come from a straight line: residue ``i`` has its CA at
``(3.8 * i, y, 0)`` with fixed N and C offsets. 3.8 A is roughly a real CA-CA
spacing and that is the entire extent of the realism -- there is no secondary
structure, no side chain, no packing, and no chemistry. Nothing measured against
these files is evidence about any biological structure. They exist to exercise
parsing, identity, rejection and packing.

WHY THE COORDINATES ARE WRITTEN AS FORMATTED DECIMALS
-----------------------------------------------------
Both writers emit ``f"{value:.3f}"``. A PDB ``ATOM`` record has three decimals by
format, so writing the mmCIF the same way makes the two files describe *exactly*
the same float32 values. ``test_pdb_and_mmcif_agree_on_the_packed_coordinates``
depends on that: if the mmCIF carried full precision the digests would differ for
a reason that has nothing to do with the code under test.

The expected coordinates are re-derived independently in the test module from
literal constants, not by calling anything here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

#: One-letter to three-letter residue names. Written out rather than inverted
#: from `smallAntibodyGen.structure.declaration.THREE_TO_ONE`, so the tests can
#: check the two tables against each other instead of checking a table against
#: itself.
ONE_TO_THREE: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

#: The decoded chain: ten canonical residues, the same string the policy tests
#: use as their context, with forced positions before, between and after sites.
TOY_DECODED_SEQUENCE = "ACDEFGHIKL"
TOY_DECODED_CHAIN = "H"

#: A short context chain, packed after the decoded one.
TOY_CONTEXT_SEQUENCE = "MNPQ"
TOY_CONTEXT_CHAIN = "A"

#: Sites declared **out of positional order on purpose**: `ConstrainedEditSpace`
#: sorts by position, so a caller reading allele 0 as ``site_b`` would be reading
#: ``site_a`` unless the permutation is carried. ``decoded_sequence[5] == "G"``
#: and ``decoded_sequence[2] == "D"``, so the context is a member of the space.
TOY_SITES: tuple[dict, ...] = (
    {
        "site_id": "site_b",
        "sequence_index": 5,
        "allowed_residues": ["G", "S"],
        "attribution": {
            "source": "synthetic fixture",
            "detail": "declared first although it sits later in the sequence",
        },
    },
    {
        "site_id": "site_a",
        "sequence_index": 2,
        "allowed_residues": ["D", "N"],
        "attribution": {
            "source": "synthetic fixture",
            "detail": "declared second although it sits earlier in the sequence",
        },
    },
)

_CA_SPACING = 3.8
_N_OFFSET = (-1.2, 0.4, 0.0)
_CA_OFFSET = (0.0, 0.0, 0.0)
_C_OFFSET = (1.3, 0.5, 0.0)
_O_OFFSET = (2.1, 1.1, 0.0)

#: Context chain rows sit 20 A off the decoded chain's axis, so the two are
#: obviously separate objects rather than an interleaved mess.
CONTEXT_Y_OFFSET = 20.0


@dataclass(frozen=True)
class Atom:
    """One atom record.

    Attributes:
        name: Atom name, matched by exact identity downstream (``N``/``CA``/``C``
            are the backbone; anything else is ignored by the adapter).
        element: Element symbol for the record's element column.
        x, y, z: Coordinates, written with three decimals.
        altloc: Alternate-location identifier, ``""`` for none.
    """

    name: str
    element: str
    x: float
    y: float
    z: float
    altloc: str = ""


@dataclass(frozen=True)
class Residue:
    """One residue's records, in file order."""

    chain_id: str
    res_id: int
    ins_code: str
    res_name: str
    atoms: tuple[Atom, ...]
    hetero: bool = False


# ---------------------------------------------------------------------------
# residue construction
# ---------------------------------------------------------------------------

def backbone_atoms(index: int, *, y_offset: float = 0.0, with_oxygen: bool = True) -> tuple[Atom, ...]:
    """N, CA, C (and O) for residue ``index`` on a straight line."""
    centre = _CA_SPACING * index
    atoms = [
        Atom("N", "N", centre + _N_OFFSET[0], y_offset + _N_OFFSET[1], _N_OFFSET[2]),
        Atom("CA", "C", centre + _CA_OFFSET[0], y_offset + _CA_OFFSET[1], _CA_OFFSET[2]),
        Atom("C", "C", centre + _C_OFFSET[0], y_offset + _C_OFFSET[1], _C_OFFSET[2]),
    ]
    if with_oxygen:
        # A fourth atom the adapter must ignore rather than trip over.
        atoms.append(
            Atom("O", "O", centre + _O_OFFSET[0], y_offset + _O_OFFSET[1], _O_OFFSET[2])
        )
    return tuple(atoms)


def _numbering(scheme: str, length: int) -> tuple[tuple[int, str], ...]:
    """``(res_id, ins_code)`` for each position under a named numbering scheme."""
    if scheme == "plain":
        return tuple((index + 1, "") for index in range(length))
    if scheme == "insertion_codes":
        if length != 10:
            raise ValueError("the insertion-code scheme is written for ten residues")
        # 5, 5A, 5B are three different residues sharing a number: the case that
        # an ins-code-blind mapping silently collapses.
        return (
            (1, ""), (2, ""), (3, ""), (4, ""), (5, ""),
            (5, "A"), (5, "B"), (6, ""), (7, ""), (8, ""),
        )
    if scheme == "numbering_gap":
        if length != 10:
            raise ValueError("the numbering-gap scheme is written for ten residues")
        # One jump, 5 -> 10. Reported as a finding and nothing more: an author
        # numbering gap is not evidence of a spatial break or a missing residue.
        return (
            (1, ""), (2, ""), (3, ""), (4, ""), (5, ""),
            (10, ""), (11, ""), (12, ""), (13, ""), (14, ""),
        )
    raise ValueError(f"unknown numbering scheme {scheme!r}")


def chain_residues(
    sequence: str,
    *,
    chain_id: str,
    numbering: str = "plain",
    y_offset: float = 0.0,
    res_name_overrides: Mapping[int, str] | None = None,
    with_oxygen: bool = True,
) -> tuple[Residue, ...]:
    """Build one chain's residues from a one-letter sequence.

    Args:
        sequence: One-letter residues.
        chain_id: Author chain identifier.
        numbering: ``"plain"``, ``"insertion_codes"`` or ``"numbering_gap"``.
        y_offset: Shifts the whole chain off the decoded chain's axis.
        res_name_overrides: ``{index: three_letter_name}``. Used to build a
            structural template whose residue deliberately differs from the
            edited sequence, which the manifest must then declare as a mismatch.
        with_oxygen: Include a non-backbone O atom per residue.
    """
    overrides = dict(res_name_overrides or {})
    numbers = _numbering(numbering, len(sequence))
    residues = []
    for index, residue in enumerate(sequence):
        res_id, ins_code = numbers[index]
        residues.append(
            Residue(
                chain_id=chain_id,
                res_id=res_id,
                ins_code=ins_code,
                res_name=overrides.get(index, ONE_TO_THREE[residue]),
                atoms=backbone_atoms(index, y_offset=y_offset, with_oxygen=with_oxygen),
            )
        )
    return tuple(residues)


def toy_decoded_residues(**kwargs) -> tuple[Residue, ...]:
    """The decoded chain of the toy complex."""
    kwargs.setdefault("chain_id", TOY_DECODED_CHAIN)
    return chain_residues(TOY_DECODED_SEQUENCE, **kwargs)


def toy_context_residues(**kwargs) -> tuple[Residue, ...]:
    """The context chain of the toy complex."""
    kwargs.setdefault("chain_id", TOY_CONTEXT_CHAIN)
    kwargs.setdefault("y_offset", CONTEXT_Y_OFFSET)
    return chain_residues(TOY_CONTEXT_SEQUENCE, **kwargs)


def water_residue(chain_id: str, res_id: int = 900) -> Residue:
    """A HETATM water record, for the solvent-in-a-selected-chain rejection."""
    return Residue(
        chain_id=chain_id,
        res_id=res_id,
        ins_code="",
        res_name="HOH",
        atoms=(Atom("O", "O", 50.0, 50.0, 50.0),),
        hetero=True,
    )


def with_altloc(residues: Sequence[Residue], index: int, altloc: str = "A") -> tuple[Residue, ...]:
    """Stamp an alternate-location identifier on one residue's backbone atoms."""
    out = list(residues)
    target = out[index]
    out[index] = replace(
        target, atoms=tuple(replace(atom, altloc=altloc) for atom in target.atoms)
    )
    return tuple(out)


def without_atom(residues: Sequence[Residue], index: int, atom_name: str) -> tuple[Residue, ...]:
    """Drop one named atom from one residue, for the missing-backbone rejection."""
    out = list(residues)
    target = out[index]
    kept = tuple(atom for atom in target.atoms if atom.name != atom_name)
    if len(kept) == len(target.atoms):
        raise ValueError(f"residue {index} has no atom named {atom_name!r}")
    out[index] = replace(target, atoms=kept)
    return tuple(out)


def with_duplicate_atom(residues: Sequence[Residue], index: int, atom_name: str) -> tuple[Residue, ...]:
    """Repeat one named atom in one residue, for the ambiguity rejection."""
    out = list(residues)
    target = out[index]
    duplicate = next(atom for atom in target.atoms if atom.name == atom_name)
    out[index] = replace(
        target, atoms=target.atoms + (replace(duplicate, x=duplicate.x + 0.2),)
    )
    return tuple(out)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def atom_row_count(residues: Sequence[Residue], *, num_models: int = 1) -> int:
    """How many ``atom_site`` rows / ``ATOM`` lines these residues produce.

    Lets a test build a full-length column override without restating the
    fixture's atom counts.
    """
    return num_models * sum(len(residue.atoms) for residue in residues)


def split_residue_atoms(
    residues: Sequence[Residue], index: int, first_atoms: int
) -> tuple[Residue, Residue]:
    """One residue as two records blocks: the first ``first_atoms`` atoms, then the rest.

    The two blocks carry the same ``(chain_id, res_id, ins_code)``, so writing
    something between them -- another chain's residues, or a ``TER`` -- produces a
    file where one residue's atoms are in two separate segments.
    """
    target = residues[index]
    if not 0 < first_atoms < len(target.atoms):
        raise ValueError(
            f"cannot split {len(target.atoms)} atoms after {first_atoms}"
        )
    return (
        replace(target, atoms=target.atoms[:first_atoms]),
        replace(target, atoms=target.atoms[first_atoms:]),
    )


def write_pdb(
    path: Path | str,
    residues: Sequence[Residue],
    *,
    num_models: int = 1,
    ter_after_atoms: int | None = None,
) -> Path:
    """Write fixed-column PDB ``ATOM``/``HETATM`` records.

    Every line is padded to the full 80 columns: biotite's reader slices the
    charge field at ``line[78:80]`` and indexes into it unconditionally
    (``pdb/file.py:430``), so a short line raises there rather than parsing.

    ``num_models`` repeats the same records inside ``MODEL``/``ENDMDL`` blocks,
    shifted by 100 A per model so the models are distinguishable. Model selection
    downstream is by ordinal, which is exactly what biotite's reader does
    (``pdb/file.py:1164-1182`` slices between ``MODEL`` line indices).

    ``ter_after_atoms`` writes a ``TER`` record after that many ``ATOM`` lines in
    every model. biotite keeps ``ATOM``/``HETATM`` lines only, so a ``TER`` leaves
    no trace in the parsed arrays: the records either side of it become adjacent
    rows. That is the case the adapter has to catch from the raw lines.
    """
    path = Path(path)
    lines: list[str] = []
    serial = 1
    for model in range(1, num_models + 1):
        if num_models > 1:
            lines.append(f"MODEL     {model:>4}")
        shift = 100.0 * (model - 1)
        written = 0
        for residue in residues:
            for atom in residue.atoms:
                lines.append(_pdb_line(serial, atom, residue, shift))
                serial += 1
                written += 1
                if written == ter_after_atoms:
                    lines.append(f"{'TER':<6}{serial:>5}{'':69}")
                    serial += 1
        if num_models > 1:
            lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _pdb_line(serial: int, atom: Atom, residue: Residue, shift: float) -> str:
    if len(residue.chain_id) != 1:
        raise ValueError(
            f"a PDB chain identifier is one column wide; got {residue.chain_id!r}"
        )
    name_field = atom.name if len(atom.name) >= 4 else f" {atom.name:<3}"
    line = (
        f"{'HETATM' if residue.hetero else 'ATOM':<6}"
        f"{serial:>5} "
        f"{name_field}"
        f"{atom.altloc or ' ':1}"
        f"{residue.res_name:>3} "
        f"{residue.chain_id:1}"
        f"{residue.res_id:>4}"
        f"{residue.ins_code or ' ':1}   "
        f"{atom.x + shift:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
        f"{1.0:>6.2f}{0.0:>6.2f}"
        f"{'':10}"
        f"{atom.element:>2}"
        f"{'':2}"
    )
    assert len(line) == 80, f"malformed PDB line of {len(line)} columns: {line!r}"
    return line


def write_cif(
    path: Path | str,
    residues: Sequence[Residue],
    *,
    num_models: int = 1,
    include_auth: bool = True,
    num_blocks: int = 1,
    column_overrides: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    """Write an mmCIF ``atom_site`` loop through biotite's own serializer.

    Built column by column rather than through `set_structure` so a fixture can
    omit the ``auth_*`` columns (``include_auth=False``) or carry a second data
    block (``num_blocks=2``) -- the two cases the adapter has to refuse, and the
    two a round trip through an `AtomArray` cannot express.

    ``column_overrides`` replaces whole columns by raw text, which is how a
    fixture writes what no `AtomArray` can hold: a ``group_PDB`` that is neither
    ``ATOM`` nor ``HETATM``, or model numbers that do not run in one block per
    model. Each replacement must be exactly as long as the column it replaces
    (see `atom_row_count`).

    Requires biotite; tests that call it are skipped without the optional extra.
    """
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

    path = Path(path)
    columns = _cif_columns(residues, num_models=num_models, include_auth=include_auth)
    for name, values in dict(column_overrides or {}).items():
        if name not in columns:
            raise ValueError(f"no column named {name!r} to override")
        if len(values) != len(columns[name]):
            raise ValueError(
                f"column {name!r} has {len(columns[name])} rows, override has "
                f"{len(values)}"
            )
        columns[name] = [str(value) for value in values]
    blocks = {}
    for index in range(num_blocks):
        name = "synthetic" if index == 0 else f"synthetic_{index}"
        blocks[name] = CIFBlock(
            {"atom_site": CIFCategory(dict(columns), name="atom_site")}, name=name
        )
    CIFFile(blocks).write(str(path))
    return path


def _cif_columns(
    residues: Sequence[Residue], *, num_models: int, include_auth: bool
) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {
        "group_PDB": [],
        "id": [],
        "type_symbol": [],
        "label_atom_id": [],
        "label_alt_id": [],
        "label_comp_id": [],
        "label_asym_id": [],
        "label_entity_id": [],
        "label_seq_id": [],
        "pdbx_PDB_ins_code": [],
        "Cartn_x": [],
        "Cartn_y": [],
        "Cartn_z": [],
        "occupancy": [],
        "B_iso_or_equiv": [],
        "pdbx_PDB_model_num": [],
    }
    if include_auth:
        for name in ("auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id"):
            columns[name] = []

    serial = 1
    for model in range(1, num_models + 1):
        shift = 100.0 * (model - 1)
        for residue in residues:
            for atom in residue.atoms:
                columns["group_PDB"].append("HETATM" if residue.hetero else "ATOM")
                columns["id"].append(str(serial))
                columns["type_symbol"].append(atom.element)
                columns["label_atom_id"].append(atom.name)
                columns["label_alt_id"].append(atom.altloc or ".")
                columns["label_comp_id"].append(residue.res_name)
                columns["label_asym_id"].append(residue.chain_id)
                columns["label_entity_id"].append("1")
                columns["label_seq_id"].append(str(residue.res_id))
                columns["pdbx_PDB_ins_code"].append(residue.ins_code or ".")
                columns["Cartn_x"].append(f"{atom.x + shift:.3f}")
                columns["Cartn_y"].append(f"{atom.y:.3f}")
                columns["Cartn_z"].append(f"{atom.z:.3f}")
                columns["occupancy"].append("1.00")
                columns["B_iso_or_equiv"].append("0.00")
                columns["pdbx_PDB_model_num"].append(str(model))
                if include_auth:
                    columns["auth_seq_id"].append(str(residue.res_id))
                    columns["auth_comp_id"].append(residue.res_name)
                    columns["auth_asym_id"].append(residue.chain_id)
                    columns["auth_atom_id"].append(atom.name)
                serial += 1
    return columns


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------

def build_manifest_document(
    *,
    relative_path: str,
    structure_path: Path | str,
    structure_format: str,
    decoded_residues: Sequence[Residue],
    decoded_sequence: str = TOY_DECODED_SEQUENCE,
    chain_order: Sequence[str] = (TOY_DECODED_CHAIN, TOY_CONTEXT_CHAIN),
    confidence: Mapping[str, float] | None = None,
    model_ordinal: int = 1,
    relationship_kind: str = "synthetic_example",
    relationship_description: str = (
        "A generated straight-line backbone. It is not a deposited structure and "
        "carries no biological meaning."
    ),
    sites: Sequence[Mapping] = TOY_SITES,
    mismatch_reasons: Mapping[int, str] | None = None,
    inter_chain_pad_length: int = 10,
    residue_numbering: str = "author",
    notes: str = "",
) -> dict:
    """Assemble a manifest document for a written structure file.

    The correspondence is derived from ``decoded_residues`` -- the residues the
    file actually holds -- so a manifest built here agrees with its structure by
    construction, and a test creates a disagreement by changing one of the two
    deliberately.
    """
    from smallAntibodyGen.structure.declaration import sha256_file

    structure_path = Path(structure_path)
    reasons = dict(mismatch_reasons or {})
    if len(decoded_residues) != len(decoded_sequence):
        raise ValueError(
            f"{len(decoded_residues)} decoded residues for a {len(decoded_sequence)}-residue "
            "sequence"
        )

    correspondence = []
    for index, residue in enumerate(decoded_residues):
        reason = reasons.get(index)
        correspondence.append(
            {
                "sequence_index": index,
                "chain_id": residue.chain_id,
                "res_id": residue.res_id,
                "ins_code": residue.ins_code,
                "expected_res_name": residue.res_name,
                "sequence_residue": decoded_sequence[index],
                "mismatch": {"reason": reason} if reason is not None else None,
            }
        )

    order = list(chain_order)
    return {
        "schema_version": "esmif1-structure-manifest/1",
        "kind": "esmif1_structure_manifest",
        "source": {
            "relative_path": relative_path,
            "sha256": sha256_file(structure_path),
            "size_bytes": structure_path.stat().st_size,
            "format": structure_format,
            "model_ordinal": model_ordinal,
            "residue_numbering": residue_numbering,
            "attribution": {
                "source": "generated by fixtures_esmif1_structure.py",
                "detail": "synthetic straight-line backbone; not a deposited structure",
                "license": "not applicable (generated in-repo)",
            },
        },
        "chains": {
            "order": order,
            "decoded_chain": order[0],
            "confidence": dict(confidence) if confidence else {c: 1.0 for c in order},
            "inter_chain_pad_length": inter_chain_pad_length,
        },
        "conventions": {"altloc": "reject_any", "missing_backbone_atom": "reject"},
        "structural_relationship": {
            "kind": relationship_kind,
            "description": relationship_description,
        },
        "decoded_sequence": decoded_sequence,
        "correspondence": correspondence,
        "edit_space": {"sites": [dict(site) for site in sites]},
        "notes": notes,
    }


def write_manifest(path: Path | str, document: Mapping) -> Path:
    """Write a manifest document in the package's canonical JSON form."""
    from smallAntibodyGen.structure.declaration import canonical_json

    path = Path(path)
    path.write_text(canonical_json(document), encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# the one-call generator
# ---------------------------------------------------------------------------

def write_toy_example(
    directory: Path | str,
    *,
    name: str = "toy",
    structure_format: str = "pdb",
    numbering: str = "plain",
) -> dict[str, str]:
    """Write a complete, clearly synthetic manifest + structure pair.

    This is the entry point for producing a runnable CLI example without a test
    harness::

        import fixtures_esmif1_structure as fx
        paths = fx.write_toy_example("outputs/scratch/toy")

    Returns:
        ``{"directory", "structure", "manifest"}`` as strings.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    decoded = toy_decoded_residues(numbering=numbering)
    context = toy_context_residues()
    residues = decoded + context

    suffix = "pdb" if structure_format == "pdb" else "cif"
    structure_path = directory / f"{name}_complex.{suffix}"
    if structure_format == "pdb":
        write_pdb(structure_path, residues)
    else:
        write_cif(structure_path, residues)

    document = build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format=structure_format,
        decoded_residues=decoded,
    )
    manifest_path = write_manifest(directory / f"{name}.manifest.json", document)
    return {
        "directory": str(directory),
        "structure": str(structure_path),
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":  # pragma: no cover - convenience for a manual run
    import argparse
    import sys

    # `src/` on the path, so this runs from a checkout with no editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description=write_toy_example.__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--name", default="toy")
    parser.add_argument("--format", dest="structure_format", default="pdb",
                        choices=("pdb", "mmcif"))
    arguments = parser.parse_args()
    print(json.dumps(
        write_toy_example(
            arguments.directory,
            name=arguments.name,
            structure_format=arguments.structure_format,
        ),
        indent=2,
    ))
