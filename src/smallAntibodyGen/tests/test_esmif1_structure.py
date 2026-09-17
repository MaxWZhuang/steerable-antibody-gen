"""Pins the declared structural-input adapter for the ESM-IF1 editing policy.

The failure modes these tests are built around, rather than coverage:

- **a silent parser choice.** ``pdbx.get_structure`` falls back from an absent
  ``auth_*`` column to the matching ``label_*`` column with only a
  ``warnings.warn`` (``pdbx/convert.py:455-473``), returns the *first* data block
  when none is named (``:444-449``), and defaults ``altloc="first"`` (``:250``).
  Each one changes what a residue number means without raising.
- **a mapping that is quietly partial.** A correspondence covering nine of ten
  observed residues encodes a tenth residue nobody declared; every shape stays
  valid.
- **the site-order swap.** `ConstrainedEditSpace` sorts sites by position
  (``esmif1_policy.py:368``), so a caller who declared ``site_b`` first and then
  reads allele 0 as ``site_b`` reads ``site_a`` instead. Both are legal indices.
- **an artifact that validates against itself.** A recomputed hash agrees with
  the value it was computed from; it is not evidence about the original file. Nor
  is a set of spans that tiles the array evidence that the chains are where the
  artifact says they are -- any permutation of them tiles it too.
- **a mutable declaration.** ``@dataclass(frozen=True)`` does not freeze the dict
  inside it, so a caller holding the document it passed in could edit a manifest
  that had already validated.
- **a path that is one file here and another on Windows.** ``C:toy.pdb``,
  ``toy.pdb:stream`` and ``CON`` all survive a leading-slash check.

Everything up to the preparation section runs on stdlib plus numpy. The
preparation and CLI sections need biotite (the optional ``esm-if1`` extra) and
skip without it. The final section builds a **real** ~32-dimension
`GVPTransformerModel` with random weights: it needs the full extra but never a
checkpoint, downloads nothing, and verifies wiring rather than any biological
result.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from smallAntibodyGen.models.esmif1_policy import (
    CANONICAL_RESIDUES as POLICY_CANONICAL_RESIDUES,
    ESMIF1_TOKENS,
)
from smallAntibodyGen.structure import (
    ManifestValidationError,
    PreparedArtifactError,
    StructurePreparationError,
    build_report,
    canonical_json,
    document_digest,
    document_to_prepared,
    load_manifest,
    prepare_structure,
    prepared_to_document,
    validate_manifest,
    verify_against_source,
    write_outputs,
)
from smallAntibodyGen.structure import declaration as decl
from smallAntibodyGen.structure import prepare as prep
from smallAntibodyGen.structure.policy_adapter import (
    PolicyBindingError,
    bind_policy,
    build_edit_space,
)


def _load_sibling(name: str):
    """Import a module sitting next to this test file, by path.

    ``src/smallAntibodyGen/tests`` deliberately has no ``__init__.py``, so a
    relative import is unavailable and a bare ``import fixtures_esmif1_structure``
    would claim a generic top-level name. Loading by path is what the rest of this
    suite already does (see ``test_target_identity_acceptance.py:107-122``).
    """
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"smallAntibodyGen_tests_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_sibling("fixtures_esmif1_structure")

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
CLI = PROJECT_ROOT / "scripts" / "prepare_esmif1_structure.py"

BIOTITE_AVAILABLE = importlib.util.find_spec("biotite") is not None
PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None
ESM_AVAILABLE = importlib.util.find_spec("esm") is not None
ESM_IF1_STACK = BIOTITE_AVAILABLE and PYG_AVAILABLE and ESM_AVAILABLE

needs_biotite = pytest.mark.skipif(
    not BIOTITE_AVAILABLE, reason="optional 'esm-if1' extra not installed (biotite)"
)

# Independently written literals, not re-derived from the fixture module: residue
# i has its CA at (3.8 * i, y, 0) with N at (-1.2, +0.4) and C at (+1.3, +0.5).
FIRST_DECODED_RESIDUE = [[-1.200, 0.400, 0.0], [0.000, 0.000, 0.0], [1.300, 0.500, 0.0]]
LAST_DECODED_RESIDUE = [[33.000, 0.400, 0.0], [34.200, 0.000, 0.0], [35.500, 0.500, 0.0]]
FIRST_CONTEXT_RESIDUE = [[-1.200, 20.400, 0.0], [0.000, 20.000, 0.0], [1.300, 20.500, 0.0]]
LAST_CONTEXT_RESIDUE = [[10.200, 20.400, 0.0], [11.400, 20.000, 0.0], [12.700, 20.500, 0.0]]

DECODED_ROWS = 10
CONTEXT_ROWS = 4
PAD_ROWS = 10
TOTAL_ROWS = DECODED_ROWS + PAD_ROWS + CONTEXT_ROWS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _write_structure(
    directory: Path,
    residues,
    *,
    structure_format: str = "pdb",
    num_models: int = 1,
    include_auth: bool = True,
    num_blocks: int = 1,
    name: str = "toy",
) -> Path:
    suffix = "pdb" if structure_format == "pdb" else "cif"
    path = directory / f"{name}_complex.{suffix}"
    if structure_format == "pdb":
        fx.write_pdb(path, residues, num_models=num_models)
    else:
        fx.write_cif(
            path,
            residues,
            num_models=num_models,
            include_auth=include_auth,
            num_blocks=num_blocks,
        )
    return path


def _toy(directory: Path, *, structure_format: str = "pdb", numbering: str = "plain", **manifest_kwargs):
    """A good toy complex: written structure, written manifest, validated manifest."""
    decoded = manifest_kwargs.pop("decoded_residues", None)
    if decoded is None:
        decoded = fx.toy_decoded_residues(numbering=numbering)
    context = manifest_kwargs.pop("context_residues", None)
    if context is None:
        context = fx.toy_context_residues()
    extra = tuple(manifest_kwargs.pop("extra_residues", ()))
    file_residues = manifest_kwargs.pop("file_residues", None)
    if file_residues is None:
        file_residues = tuple(decoded) + extra + tuple(context)

    structure_path = _write_structure(
        directory,
        file_residues,
        structure_format=structure_format,
        num_models=manifest_kwargs.pop("num_models", 1),
        include_auth=manifest_kwargs.pop("include_auth", True),
        num_blocks=manifest_kwargs.pop("num_blocks", 1),
    )
    document = fx.build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format="pdb" if structure_format == "pdb" else "mmcif",
        decoded_residues=decoded,
        **manifest_kwargs,
    )
    manifest_path = fx.write_manifest(directory / "toy.manifest.json", document)
    return structure_path, manifest_path, document


def _prepared(directory: Path, **kwargs):
    structure_path, manifest_path, _ = _toy(directory, **kwargs)
    return prepare_structure(load_manifest(manifest_path), structure_path), structure_path


def _report_for(prepared, manifest_path, structure_path, **overrides) -> dict:
    """A report for real files, with the artifact digest computed rather than faked.

    `build_report` re-reads both named files and re-runs the preparation before it
    will emit a report, so there is no such thing as a report for a path that does
    not exist or for a digest nobody computed.
    """
    arguments = {
        "manifest_path": manifest_path,
        "structure_path": structure_path,
        "artifact_filename": "toy.prepared.json",
        "artifact_content_sha256": prepared_to_document(prepared)["identity"][
            "content_sha256"
        ],
    }
    arguments.update(overrides)
    return build_report(prepared, **arguments)


def _reseal(document: dict) -> dict:
    """Recompute ``identity.content_sha256`` so a tamper test targets one check."""
    doc = copy.deepcopy(document)
    doc["identity"].pop("content_sha256", None)
    doc["identity"]["content_sha256"] = document_digest(doc)
    return doc


# --------------------------------------------------------------------------
# The declaration: stdlib only
# --------------------------------------------------------------------------

def test_the_residue_tables_agree_with_the_policy_and_with_each_other():
    """The two replicated tables are the one place this layer can drift."""
    assert decl.CANONICAL_RESIDUES == POLICY_CANONICAL_RESIDUES
    assert fx.ONE_TO_THREE == {one: three for three, one in decl.THREE_TO_ONE.items()}
    assert len(decl.THREE_TO_ONE) == 20
    # Every canonical residue must also be a native token, or the context would
    # encode as <unk> inside the policy.
    assert set(decl.CANONICAL_RESIDUES) <= set(ESMIF1_TOKENS)


def test_a_valid_manifest_round_trips_byte_identically(tmp_path):
    _, manifest_path, document = _toy(tmp_path)
    manifest = load_manifest(manifest_path)

    assert manifest.decoded_chain == "H"
    assert manifest.decoded_sequence == fx.TOY_DECODED_SEQUENCE
    assert len(manifest.correspondence) == DECODED_ROWS
    # The digest is over the canonical form, so rewriting a parsed manifest is a
    # no-op rather than a new document.
    assert manifest_path.read_text(encoding="utf-8") == canonical_json(document)
    assert manifest.digest == document_digest(document)


def test_site_order_is_carried_across_the_policys_positional_sort(tmp_path):
    """The single most likely silent bug in the whole integration.

    ``site_b`` is declared first but sits at position 5; ``site_a`` is declared
    second and sits at position 2. `ConstrainedEditSpace` will hold them the other
    way round, and every allele vector follows *its* order.
    """
    _, manifest_path, _ = _toy(tmp_path)
    manifest = load_manifest(manifest_path)

    assert manifest.declared_site_ids == ("site_b", "site_a")
    assert manifest.policy_site_ids == ("site_a", "site_b")
    assert manifest.declared_to_policy == (1, 0)
    assert manifest.policy_to_declared == (1, 0)

    space = build_edit_space(
        prep.PreparedStructure(
            manifest=manifest,
            structure_sha256="0" * 64,
            coordinates=np.zeros((DECODED_ROWS, 3, 3), dtype=np.float32),
            confidence=np.ones(DECODED_ROWS, dtype=np.float32),
            chain_spans=(prep.ChainSpan("H", "decoded", 0, DECODED_ROWS, DECODED_ROWS),),
            pad_spans=(),
            findings=prep.PreparationFindings((), (), DECODED_ROWS, {}),
        )
    )
    assert space.positions == (2, 5)
    # Allele order *within* a site is the caller's, faithfully.
    assert space.sites[0].alleles == ("D", "N")
    assert space.sites[1].alleles == ("G", "S")


def test_editing_the_caller_document_cannot_relabel_a_validated_manifest(tmp_path):
    """Ingress: `@dataclass(frozen=True)` does not freeze the dict inside it.

    The caller keeps a reference to the document it passed in. If the manifest
    held that object, editing it afterwards would change what the manifest says
    *and* the digest that is supposed to pin it.
    """
    _, _, document = _toy(tmp_path)
    manifest = validate_manifest(document)
    digest = manifest.digest

    document["decoded_sequence"] = "GC"
    document["chains"]["confidence"]["H"] = 0.1
    document["edit_space"]["sites"][0]["site_id"] = "renamed"
    document["correspondence"][0]["res_id"] = 999

    assert manifest.decoded_sequence == fx.TOY_DECODED_SEQUENCE
    assert manifest.document["decoded_sequence"] == fx.TOY_DECODED_SEQUENCE
    assert manifest.chains.confidence["H"] == 1.0
    assert manifest.declared_site_ids == ("site_b", "site_a")
    assert manifest.correspondence[0].res_id == 1
    assert manifest.digest == digest


def test_the_returned_document_is_a_copy_and_the_nested_mappings_are_read_only(tmp_path):
    """Egress: what a caller gets back is a copy, and the parsed views refuse edits."""
    _, _, document = _toy(tmp_path)
    manifest = validate_manifest(document)

    first, second = manifest.document, manifest.document
    assert first == second and first is not second
    first["decoded_sequence"] = "GC"
    first["chains"]["confidence"]["H"] = 0.1
    assert manifest.document["decoded_sequence"] == fx.TOY_DECODED_SEQUENCE
    assert manifest.digest == document_digest(document)

    for mapping, key in (
        (manifest.chains.confidence, "H"),
        (manifest.source.attribution, "source"),
        (manifest.sites[0].attribution, "source"),
    ):
        with pytest.raises(TypeError):
            mapping[key] = "edited"


def test_duplicate_json_keys_are_rejected():
    text = '{"schema_version": "a", "schema_version": "b"}'
    with pytest.raises(ManifestValidationError, match="same key more than once"):
        decl.loads_strict_json(text)


def test_non_finite_json_literals_are_rejected():
    with pytest.raises(ManifestValidationError, match="non-finite literal"):
        decl.loads_strict_json('{"confidence": NaN}')


def test_an_unknown_manifest_key_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["extra"] = 1
    with pytest.raises(ManifestValidationError, match="unknown=\\['extra'\\]"):
        validate_manifest(document)


def test_label_residue_numbering_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["source"]["residue_numbering"] = "label"
    with pytest.raises(ManifestValidationError, match="residue_numbering"):
        validate_manifest(document)


def test_a_zero_model_ordinal_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["source"]["model_ordinal"] = 0
    with pytest.raises(ManifestValidationError, match="1-based"):
        validate_manifest(document)


@pytest.mark.parametrize(
    "relative_path",
    [
        "C:/elsewhere/toy.pdb",   # drive-qualified and rooted
        "C:toy.pdb",              # drive-relative: a *different* directory on Windows
        "//server/share/toy.pdb", # UNC share
        "toy.pdb:stream",         # NTFS alternate data stream
        "sub/../toy.pdb",         # traversal
        "/etc/toy.pdb",           # rooted
        "toy.pdb.",               # Windows strips the trailing dot
        "CON.pdb",                # a device, whatever the extension
        "toy\x01.pdb",            # control character
        "sub\\toy.pdb",           # backslash separator
    ],
)
def test_a_relative_path_that_is_not_one_file_everywhere_is_rejected(tmp_path, relative_path):
    """`relative_path` must name one file below the root on every platform.

    A string check that only looks for a leading ``/`` passes every entry above;
    the first three then resolve *outside* the structure root when joined.
    """
    _, _, document = _toy(tmp_path)
    document["source"]["relative_path"] = relative_path
    with pytest.raises(ManifestValidationError, match="relative_path"):
        validate_manifest(document)


def test_the_decoded_chain_must_be_packed_first(tmp_path):
    _, _, document = _toy(tmp_path)
    document["chains"]["order"] = ["A", "H"]
    with pytest.raises(ManifestValidationError, match="decoded_chain"):
        validate_manifest(document)


def test_a_missing_correspondence_row_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    del document["correspondence"][4]
    with pytest.raises(ManifestValidationError, match="every decoded index"):
        validate_manifest(document)


def test_an_out_of_order_sequence_index_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["correspondence"][3]["sequence_index"] = 4
    document["correspondence"][4]["sequence_index"] = 3
    with pytest.raises(ManifestValidationError, match="ascending order"):
        validate_manifest(document)


def test_two_rows_may_not_claim_the_same_structural_residue(tmp_path):
    _, _, document = _toy(tmp_path)
    document["correspondence"][4]["res_id"] = document["correspondence"][3]["res_id"]
    with pytest.raises(ManifestValidationError, match="a second time"):
        validate_manifest(document)


def test_an_undeclared_sequence_mismatch_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["correspondence"][3]["expected_res_name"] = "SER"
    with pytest.raises(ManifestValidationError, match="declares no mismatch"):
        validate_manifest(document)


def test_a_mismatch_declared_where_the_residues_agree_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["structural_relationship"]["kind"] = "template_for_different_construct"
    document["correspondence"][3]["mismatch"] = {"reason": "not actually different"}
    with pytest.raises(ManifestValidationError, match="the same residue"):
        validate_manifest(document)


def test_a_mismatch_is_rejected_under_same_construct(tmp_path):
    decoded = fx.toy_decoded_residues(res_name_overrides={3: "SER"})
    _, _, document = _toy(
        tmp_path,
        decoded_residues=decoded,
        mismatch_reasons={3: "template carries serine"},
        relationship_kind="template_for_different_construct",
    )
    document["structural_relationship"]["kind"] = "same_construct"
    with pytest.raises(ManifestValidationError, match="same_construct"):
        validate_manifest(document)


def test_three_alleles_are_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["edit_space"]["sites"][0]["allowed_residues"] = ["G", "S", "A"]
    with pytest.raises(ManifestValidationError, match="exactly two residues"):
        validate_manifest(document)


def test_a_context_residue_outside_a_sites_support_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["edit_space"]["sites"][0]["allowed_residues"] = ["W", "Y"]
    with pytest.raises(ManifestValidationError, match="member of the edit space"):
        validate_manifest(document)


def test_a_site_outside_the_decoded_sequence_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["edit_space"]["sites"][0]["sequence_index"] = 99
    with pytest.raises(ManifestValidationError, match="outside the decoded sequence"):
        validate_manifest(document)


def test_duplicate_site_ids_are_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["edit_space"]["sites"][1]["site_id"] = "site_b"
    with pytest.raises(ManifestValidationError, match="declared twice"):
        validate_manifest(document)


def test_a_confidence_outside_the_unit_interval_is_rejected(tmp_path):
    _, _, document = _toy(tmp_path)
    document["chains"]["confidence"]["A"] = 1.5
    with pytest.raises(ManifestValidationError, match=r"\[0, 1\]"):
        validate_manifest(document)


def test_structure_path_for_returns_the_resolved_path_below_the_root(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path)
    manifest = load_manifest(manifest_path)

    resolved = prep.structure_path_for(manifest, tmp_path)
    assert resolved == structure_path.resolve()
    assert resolved.is_absolute()
    assert resolved.is_relative_to(tmp_path.resolve())


def test_structure_path_for_refuses_a_symlink_that_leaves_the_root(tmp_path):
    """The escape a string check cannot see, and the one it must not describe as inside."""
    structure_path, manifest_path, _ = _toy(tmp_path)
    manifest = load_manifest(manifest_path)
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / structure_path.name).symlink_to(structure_path)
    except (OSError, NotImplementedError):  # pragma: no cover - needs privileges
        pytest.skip("creating a symlink is not permitted on this platform")

    with pytest.raises(StructurePreparationError, match="outside the structure root"):
        prep.structure_path_for(manifest, root)


def test_importing_the_package_pulls_in_neither_biotite_nor_torch():
    """biotite is lazy by design, and the torch bridge is a separate module."""
    code = (
        "import sys;"
        "import smallAntibodyGen.structure;"
        "print('biotite' in sys.modules, 'torch' in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


# --------------------------------------------------------------------------
# Preparation: needs biotite
# --------------------------------------------------------------------------

@needs_biotite
def test_a_good_pdb_prepares_the_expected_coordinates(tmp_path):
    prepared, _ = _prepared(tmp_path)

    assert prepared.num_rows == TOTAL_ROWS
    assert prepared.coordinates.shape == (TOTAL_ROWS, 3, 3)
    assert prepared.coordinates.dtype == np.float32
    assert prepared.decoded_rows == tuple(range(DECODED_ROWS))

    assert np.allclose(prepared.coordinates[0], np.array(FIRST_DECODED_RESIDUE), atol=1e-4)
    assert np.allclose(prepared.coordinates[9], np.array(LAST_DECODED_RESIDUE), atol=1e-4)
    assert np.allclose(prepared.coordinates[20], np.array(FIRST_CONTEXT_RESIDUE), atol=1e-4)
    assert np.allclose(prepared.coordinates[23], np.array(LAST_CONTEXT_RESIDUE), atol=1e-4)

    # NaN means inter-chain padding and nothing else.
    assert np.all(np.isnan(prepared.coordinates[10:20]))
    assert np.all(np.isfinite(prepared.coordinates[:10]))
    assert np.all(np.isfinite(prepared.coordinates[20:]))
    assert prepared.confidence.shape == (TOTAL_ROWS,)
    assert np.all(prepared.confidence == 1.0)


@needs_biotite
def test_multichain_packing_follows_the_declared_order(tmp_path):
    prepared, _ = _prepared(tmp_path)

    assert [span.chain_id for span in prepared.chain_spans] == ["H", "A"]
    assert [span.role for span in prepared.chain_spans] == ["decoded", "context"]
    assert prepared.chain_spans[0].start_row == 0
    assert prepared.chain_spans[0].num_residues == DECODED_ROWS
    assert prepared.chain_spans[1].start_row == DECODED_ROWS + PAD_ROWS
    assert prepared.chain_spans[1].num_residues == CONTEXT_ROWS
    assert prepared.pad_spans == ((DECODED_ROWS, DECODED_ROWS + PAD_ROWS),)


@needs_biotite
def test_pdb_and_mmcif_of_the_same_structure_agree_exactly(tmp_path):
    """Both writers emit three decimals, so any digest difference is the code."""
    pdb_dir, cif_dir = tmp_path / "pdb", tmp_path / "cif"
    pdb_dir.mkdir()
    cif_dir.mkdir()
    from_pdb, _ = _prepared(pdb_dir)
    from_cif, _ = _prepared(cif_dir, structure_format="mmcif")

    assert from_pdb.coordinates_digest == from_cif.coordinates_digest
    assert from_pdb.confidence_digest == from_cif.confidence_digest
    assert from_pdb.chain_spans == from_cif.chain_spans


@needs_biotite
def test_a_hash_mismatch_is_rejected(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path)
    manifest = load_manifest(manifest_path)
    structure_path.write_text(
        structure_path.read_text(encoding="utf-8").replace("  1.00", "  0.99", 1),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(StructurePreparationError, match="hashes to"):
        prepare_structure(manifest, structure_path)


@needs_biotite
def test_a_size_mismatch_is_named_before_the_hash(tmp_path):
    structure_path, manifest_path, document = _toy(tmp_path)
    document["source"]["size_bytes"] += 1
    with pytest.raises(StructurePreparationError, match="bytes but the manifest declares"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_insertion_codes_are_part_of_the_residue_identity(tmp_path):
    """5, 5A and 5B are three residues, and the mapping must name all three."""
    prepared, _ = _prepared(tmp_path, numbering="insertion_codes")
    rows = prepared.manifest.correspondence
    assert [(row.res_id, row.ins_code) for row in rows[4:7]] == [(5, ""), (5, "A"), (5, "B")]
    assert prepared.findings.numbering_gaps == ()
    assert np.allclose(prepared.coordinates[5], np.array(
        [[17.800, 0.400, 0.0], [19.000, 0.000, 0.0], [20.300, 0.500, 0.0]]
    ), atol=1e-4)


def test_dropping_the_insertion_code_makes_the_mapping_ambiguous(tmp_path):
    """Without the code, 5A collides with 5 -- refused by identity, not resolved."""
    decoded = fx.toy_decoded_residues(numbering="insertion_codes")
    _, _, document = _toy(tmp_path, decoded_residues=decoded)
    document["correspondence"][5]["ins_code"] = ""
    with pytest.raises(ManifestValidationError, match="a second time"):
        validate_manifest(document)


@needs_biotite
def test_a_missing_backbone_atom_names_the_residue(tmp_path):
    complete = fx.toy_decoded_residues()
    structure_path, manifest_path, _ = _toy(
        tmp_path,
        decoded_residues=complete,
        file_residues=fx.without_atom(complete, 3, "CA") + fx.toy_context_residues(),
    )
    with pytest.raises(StructurePreparationError, match=r"H/4 GLU has no CA atom"):
        prepare_structure(load_manifest(manifest_path), structure_path)


@needs_biotite
def test_a_duplicate_backbone_atom_is_refused_rather_than_chosen_between(tmp_path):
    good = fx.toy_decoded_residues()
    structure_path, manifest_path, _ = _toy(
        tmp_path,
        decoded_residues=good,
        file_residues=fx.with_duplicate_atom(good, 2, "N") + fx.toy_context_residues(),
    )
    with pytest.raises(StructurePreparationError, match="2 atoms named N"):
        prepare_structure(load_manifest(manifest_path), structure_path)


@needs_biotite
def test_an_alternate_location_is_rejected(tmp_path):
    good = fx.toy_decoded_residues()
    structure_path, manifest_path, _ = _toy(
        tmp_path,
        decoded_residues=good,
        file_residues=fx.with_altloc(good, 6) + fx.toy_context_residues(),
    )
    with pytest.raises(StructurePreparationError, match="alternate-location identifier"):
        prepare_structure(load_manifest(manifest_path), structure_path)


@needs_biotite
@pytest.mark.parametrize("structure_format", ["pdb", "mmcif"])
@pytest.mark.parametrize("occupancy", ["0.00", "-0.10", "1.10", "nan"])
def test_unusable_atom_occupancy_is_rejected(tmp_path, structure_format, occupancy):
    decoded = fx.toy_decoded_residues()
    residues = decoded + fx.toy_context_residues()
    path = _write_structure(tmp_path, residues, structure_format=structure_format)
    if structure_format == "pdb":
        lines = path.read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("ATOM  "))
        lines[index] = lines[index][:54] + f"{occupancy:>6}" + lines[index][60:]
        path.write_text("\n".join(lines) + "\n")
    else:
        values = ["1.00"] * fx.atom_row_count(residues)
        values[0] = occupancy
        fx.write_cif(path, residues, column_overrides={"occupancy": values})
    document = fx.build_manifest_document(
        relative_path=path.name, structure_path=path,
        structure_format=structure_format, decoded_residues=decoded,
    )
    with pytest.raises(StructurePreparationError, match="occupancy"):
        prepare_structure(validate_manifest(document), path)


@needs_biotite
@pytest.mark.parametrize("structure_format", ["pdb", "mmcif"])
def test_nonfinite_discarded_atom_is_still_rejected(tmp_path, structure_format):
    decoded = fx.toy_decoded_residues()
    # The encoder only keeps N/CA/C. An invalid O record must not disappear
    # before the selected chain is validated.
    bad = replace(decoded[0], atoms=decoded[0].atoms +
                  (fx.Atom("O", "O", float("nan"), 0.0, 0.0),))
    path, _, document = _toy(
        tmp_path, structure_format=structure_format, decoded_residues=decoded,
        file_residues=(bad,) + decoded[1:] + fx.toy_context_residues(),
    )
    with pytest.raises(StructurePreparationError, match="non-finite coordinate"):
        prepare_structure(validate_manifest(document), path)


@needs_biotite
def test_mmcif_occupancy_is_not_silently_defaulted(tmp_path):
    from biotite.structure.io.pdbx import CIFFile

    path, _, document = _toy(tmp_path, structure_format="mmcif")
    cif = CIFFile.read(path)
    del cif.block["atom_site"]["occupancy"]
    cif.write(path)
    document["source"]["sha256"] = decl.sha256_file(path)
    document["source"]["size_bytes"] = path.stat().st_size
    with pytest.raises(StructurePreparationError, match="missing.*occupancy"):
        prepare_structure(validate_manifest(document), path)


@needs_biotite
def test_a_water_sharing_a_selected_chain_is_rejected_by_name(tmp_path):
    """The documented restrictive v1 rule: solvent is refused, never filtered."""
    good = fx.toy_decoded_residues()
    structure_path, manifest_path, _ = _toy(
        tmp_path,
        decoded_residues=good,
        file_residues=good + (fx.water_residue("H"),) + fx.toy_context_residues(),
    )
    with pytest.raises(StructurePreparationError, match="HETATM record on selected chain"):
        prepare_structure(load_manifest(manifest_path), structure_path)


@needs_biotite
def test_an_internal_crop_is_rejected(tmp_path):
    """Nine declared rows against ten observed residues encodes an undeclared one."""
    decoded = fx.toy_decoded_residues()
    cropped = decoded[:5] + decoded[6:]
    sequence = fx.TOY_DECODED_SEQUENCE[:5] + fx.TOY_DECODED_SEQUENCE[6:]
    structure_path, manifest_path, _ = _toy(
        tmp_path,
        decoded_residues=cropped,
        file_residues=decoded + fx.toy_context_residues(),
        decoded_sequence=sequence,
        sites=[fx.TOY_SITES[1]],
    )
    with pytest.raises(StructurePreparationError, match="v1 has no cropping"):
        prepare_structure(load_manifest(manifest_path), structure_path)


@needs_biotite
def test_a_reordered_correspondence_is_rejected(tmp_path):
    structure_path, _, document = _toy(tmp_path)
    document["correspondence"][3]["res_id"], document["correspondence"][4]["res_id"] = (
        document["correspondence"][4]["res_id"],
        document["correspondence"][3]["res_id"],
    )
    with pytest.raises(StructurePreparationError, match="neither reordering nor"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_a_declared_residue_absent_from_the_structure_is_named(tmp_path):
    structure_path, _, document = _toy(tmp_path)
    document["correspondence"][3]["res_id"] = 99
    with pytest.raises(StructurePreparationError, match="not present in the observed"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_a_wrong_expected_residue_name_is_rejected(tmp_path):
    decoded = fx.toy_decoded_residues()
    structure_path, _, document = _toy(
        tmp_path,
        decoded_residues=decoded,
        relationship_kind="template_for_different_construct",
        mismatch_reasons={3: "declared, but the structure disagrees with the declaration"},
    )
    document["correspondence"][3]["expected_res_name"] = "SER"
    with pytest.raises(StructurePreparationError, match="but the structure holds GLU"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_a_declared_mismatch_prepares_and_is_carried_into_the_findings(tmp_path):
    decoded = fx.toy_decoded_residues(res_name_overrides={3: "SER"})
    prepared, _ = _prepared(
        tmp_path,
        decoded_residues=decoded,
        relationship_kind="template_for_different_construct",
        mismatch_reasons={3: "the template construct carries serine at this position"},
    )
    assert prepared.findings.declared_sequence_mismatches == (
        {
            "expected_res_name": "SER",
            "ins_code": "",
            "reason": "the template construct carries serine at this position",
            "res_id": 4,
            "sequence_index": 3,
            "sequence_residue": "E",
        },
    )


@needs_biotite
def test_a_numbering_gap_is_reported_with_no_spatial_claim(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path, numbering="numbering_gap")
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)

    assert prepared.findings.numbering_gaps == (
        {
            "res_id_after": 10,
            "res_id_before": 5,
            "sequence_index_after": 5,
            "sequence_index_before": 4,
        },
    )
    # A gap is an observation about the numbering; it does not fail the run and
    # nothing repairs it. The report is built from the real files and the real
    # artifact digest, because `build_report` re-verifies both.
    report = _report_for(prepared, manifest_path, structure_path)
    assert report["preparation_status"] == "pass"
    assert len(prepared.coordinates) == TOTAL_ROWS


@needs_biotite
def test_a_missing_declared_chain_is_named(tmp_path):
    structure_path, _, document = _toy(tmp_path)
    document["chains"]["order"] = ["H", "Z"]
    document["chains"]["confidence"] = {"H": 1.0, "Z": 1.0}
    with pytest.raises(StructurePreparationError, match="declared chain 'Z' has no records"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_model_ordinal_two_selects_the_second_model(tmp_path):
    """Ordinal, not MODEL serial: biotite slices between MODEL line indices."""
    structure_path, _, document = _toy(tmp_path, num_models=2)
    document["source"]["model_ordinal"] = 2
    prepared = prepare_structure(validate_manifest(document), structure_path)
    # The fixture shifts each model by 100 A along x.
    assert np.allclose(prepared.coordinates[0, 1, 0], 100.0, atol=1e-3)

    document["source"]["model_ordinal"] = 3
    with pytest.raises(StructurePreparationError, match="selects ordinal 3"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_an_mmcif_without_author_columns_is_rejected(tmp_path):
    structure_path, manifest_path, _ = _toy(
        tmp_path, structure_format="mmcif", include_auth=False
    )
    with pytest.raises(StructurePreparationError, match="auth_asym_id"):
        prepare_structure(load_manifest(manifest_path), structure_path)


@needs_biotite
def test_an_mmcif_with_two_data_blocks_is_rejected(tmp_path):
    structure_path, manifest_path, _ = _toy(
        tmp_path, structure_format="mmcif", num_blocks=2
    )
    with pytest.raises(StructurePreparationError, match="data blocks"):
        prepare_structure(load_manifest(manifest_path), structure_path)


# --------------------------------------------------------------------------
# The artifact and the report
# --------------------------------------------------------------------------

@needs_biotite
def test_the_artifact_round_trips_through_its_own_reader(tmp_path):
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    restored = document_to_prepared(json.loads(canonical_json(document)))

    assert restored.manifest.digest == prepared.manifest.digest
    assert np.array_equal(restored.coordinates, prepared.coordinates, equal_nan=True)
    assert np.array_equal(restored.confidence, prepared.confidence)
    assert restored.chain_spans == prepared.chain_spans
    assert restored.pad_spans == prepared.pad_spans
    assert restored.findings == prepared.findings
    assert restored.structure_sha256 == prepared.structure_sha256


@needs_biotite
def test_two_runs_with_the_same_arguments_write_identical_bytes(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path)
    manifest = load_manifest(manifest_path)

    written = []
    for run in ("one", "two"):
        prepared = prepare_structure(manifest, structure_path)
        artifact = prepared_to_document(prepared)
        report = build_report(
            prepared,
            manifest_path=manifest_path,
            structure_path=structure_path,
            artifact_filename="toy.prepared.json",
            artifact_content_sha256=artifact["identity"]["content_sha256"],
        )
        written.append(
            write_outputs(
                artifact,
                report,
                output_dir=tmp_path / run,
                name="toy",
                input_paths=(manifest_path, structure_path),
            )
        )

    for first, second in zip(written[0], written[1]):
        assert first.read_bytes() == second.read_bytes()


@needs_biotite
def test_the_report_separates_structural_checks_from_model_integration(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    report = _report_for(prepared, manifest_path, structure_path)
    assert {check["status"] for check in report["structural_checks"]} == {"pass"}
    assert "re-verified" in report["structural_checks_method"]
    integration = report["model_integration_checks"]
    assert integration["status"] == "not_run"
    assert {check["status"] for check in integration["checks"]} == {"not_run"}
    assert "loads no weights" in integration["reason"]
    # A PDB report must not claim an mmCIF-only check passed.
    names = {check["check"] for check in report["structural_checks"]}
    assert "single_data_block" not in names
    assert "author_columns_present" not in names


@needs_biotite
def test_a_tampered_coordinate_digest_is_detected(tmp_path):
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    document["identity"]["coordinates_sha256"] = "a" * 64
    with pytest.raises(PreparedArtifactError, match="coordinates_sha256"):
        document_to_prepared(_reseal(document))


@needs_biotite
def test_an_edited_embedded_manifest_is_detected(tmp_path):
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    document["manifest"]["notes"] = "edited after writing"
    with pytest.raises(PreparedArtifactError, match="manifest_sha256"):
        document_to_prepared(document)


@needs_biotite
def test_an_edit_that_leaves_every_other_check_intact_fails_the_content_digest(tmp_path):
    """The content digest is the backstop for a field no other check re-derives."""
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    document["identity"]["structure_sha256"] = "b" * 64
    with pytest.raises(PreparedArtifactError, match="content_sha256"):
        document_to_prepared(document)


@needs_biotite
def test_an_inconsistent_array_shape_is_detected(tmp_path):
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    document["arrays"]["coordinates"]["shape"] = [TOTAL_ROWS - 1, 3, 3]
    with pytest.raises(PreparedArtifactError, match="shape"):
        document_to_prepared(_reseal(document))


@needs_biotite
def test_a_span_that_does_not_tile_the_array_is_detected(tmp_path):
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    document["packing"]["pad_spans"] = [[DECODED_ROWS + 1, DECODED_ROWS + PAD_ROWS + 1]]
    with pytest.raises(PreparedArtifactError, match="do not tile"):
        document_to_prepared(_reseal(document))


@needs_biotite
def test_a_finite_pad_row_is_detected(tmp_path):
    """Digests alone cannot catch this: they are recomputed consistently."""
    prepared, _ = _prepared(tmp_path)
    coordinates = prepared.coordinates.copy()
    coordinates[DECODED_ROWS + 2] = 0.0
    document = prepared_to_document(replace(prepared, coordinates=coordinates))
    with pytest.raises(PreparedArtifactError, match="not entirely NaN"):
        document_to_prepared(document)


@needs_biotite
def test_a_stale_site_permutation_is_detected(tmp_path):
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    document["site_permutations"]["policy_site_ids"] = ["site_b", "site_a"]
    with pytest.raises(PreparedArtifactError, match="site_permutations.policy_site_ids"):
        document_to_prepared(_reseal(document))


@needs_biotite
def test_a_resigned_structure_digest_must_still_be_the_declared_one(tmp_path):
    """Re-signing hides an edit from the content digest, not from the manifest.

    Preparation only ever records the digest it has just checked against the
    declaration, so `identity.structure_sha256` and `manifest.source.sha256`
    cannot legitimately differ. Both a malformed digest and a *valid* digest of
    some other file are refused.
    """
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)

    document["identity"]["structure_sha256"] = "z" * 64
    with pytest.raises(PreparedArtifactError, match="lowercase hex"):
        document_to_prepared(_reseal(document))

    other_file = hashlib.sha256(b"a different file").hexdigest()
    document["identity"]["structure_sha256"] = other_file
    with pytest.raises(PreparedArtifactError, match="the embedded manifest pins"):
        document_to_prepared(_reseal(document))


@needs_biotite
def test_a_resigned_artifact_may_not_permute_the_physical_chain_order(tmp_path):
    """Three chains, context spans exchanged: every span still tiles the array.

    The sorted union of the spans is identical before and after the exchange, so
    a partition check that sorts cannot see it. What the artifact now claims is
    that chain ``A``'s coordinates are ``B``'s rows -- a relabelling of which
    antigen chain is where, with every digest recomputed to match.
    """
    decoded = fx.toy_decoded_residues()
    context = fx.toy_context_residues()
    third = fx.chain_residues("VWY", chain_id="B", y_offset=2 * fx.CONTEXT_Y_OFFSET)
    prepared, _ = _prepared(
        tmp_path,
        decoded_residues=decoded,
        file_residues=decoded + context + third,
        chain_order=["H", "A", "B"],
    )
    assert [span.chain_id for span in prepared.chain_spans] == ["H", "A", "B"]
    assert [(span.start_row, span.end_row) for span in prepared.chain_spans] == [
        (0, 10), (20, 24), (34, 37)
    ]
    # The honest artifact reloads.
    document = prepared_to_document(prepared)
    assert document_to_prepared(json.loads(canonical_json(document))).num_rows == 37

    chain_a, chain_b = document["packing"]["chain_spans"][1:]
    chain_a.update(start_row=34, end_row=37, num_residues=3)
    chain_b.update(start_row=20, end_row=24, num_residues=4)
    document["findings"]["context_chain_residue_counts"] = {"A": 3, "B": 4}
    with pytest.raises(PreparedArtifactError, match="packing order"):
        document_to_prepared(_reseal(document))


@needs_biotite
def test_a_fabricated_or_dropped_numbering_gap_is_detected(tmp_path):
    """Gaps are re-derived from the correspondence, not taken on trust.

    Every decoded residue's `(res_id, ins_code)` is declared in the manifest, and
    preparation refuses unless those equal the observed ones, so the gap list is a
    function of the manifest alone.
    """
    prepared, _ = _prepared(tmp_path)
    assert prepared.findings.numbering_gaps == ()
    document = prepared_to_document(prepared)
    document["findings"]["numbering_gaps"] = [
        {
            "res_id_after": 10,
            "res_id_before": 5,
            "sequence_index_after": 5,
            "sequence_index_before": 4,
        }
    ]
    with pytest.raises(PreparedArtifactError, match="numbering_gaps"):
        document_to_prepared(_reseal(document))

    gap_directory = tmp_path / "gap"
    gap_directory.mkdir()
    with_gap, _ = _prepared(gap_directory, numbering="numbering_gap")
    dropped = prepared_to_document(with_gap)
    assert dropped["findings"]["numbering_gaps"]
    dropped["findings"]["numbering_gaps"] = []
    with pytest.raises(PreparedArtifactError, match="numbering_gaps"):
        document_to_prepared(_reseal(dropped))


@needs_biotite
def test_a_pad_row_carrying_a_different_confidence_is_detected(tmp_path):
    prepared, _ = _prepared(tmp_path)
    confidence = prepared.confidence.copy()
    confidence[DECODED_ROWS + 3] = 0.25
    document = prepared_to_document(replace(prepared, confidence=confidence))
    with pytest.raises(PreparedArtifactError, match="pad confidence"):
        document_to_prepared(document)


@needs_biotite
def test_editing_a_prepared_artifact_document_cannot_relabel_the_prepared_inputs(tmp_path):
    """Egress at the artifact boundary: the embedded manifest is a copy."""
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    manifest_digest = prepared.manifest.digest

    document["manifest"]["decoded_sequence"] = "AAAAAAAAAA"
    document["manifest"]["edit_space"]["sites"][0]["allowed_residues"] = ["G", "W"]
    document["manifest"]["chains"]["confidence"]["A"] = 0.0

    assert prepared.manifest.decoded_sequence == fx.TOY_DECODED_SEQUENCE
    assert prepared.manifest.sites[0].allowed_residues == ("G", "S")
    assert prepared.manifest.digest == manifest_digest
    assert prepared_to_document(prepared)["identity"]["manifest_sha256"] == manifest_digest


@needs_biotite
def test_verify_against_source_agrees_and_then_detects_a_changed_file(tmp_path):
    prepared, structure_path = _prepared(tmp_path)
    assert verify_against_source(prepared, structure_path) == ()

    structure_path.write_text(
        structure_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
        newline="\n",
    )
    problems = verify_against_source(prepared, structure_path)
    assert problems and "failed" in problems[0]


@needs_biotite
def test_a_report_cannot_claim_a_source_pass_for_coordinates_the_file_does_not_hold(
    tmp_path,
):
    """The report's own failure mode: every digest recomputed, nothing re-read.

    The coordinates are moved and the artifact re-serialized, so every embedded
    digest agrees with the edited arrays. The source file is untouched, so the
    only thing that can catch this is re-reading it -- which is what makes the
    report's `structural_checks` mean anything.
    """
    structure_path, manifest_path, _ = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    coordinates = prepared.coordinates.copy()
    coordinates[0, 1, 0] += 5.0
    mutated = replace(prepared, coordinates=coordinates)
    before = structure_path.read_bytes()

    with pytest.raises(StructurePreparationError, match="do not match the structure file"):
        _report_for(mutated, manifest_path, structure_path)
    assert structure_path.read_bytes() == before


@needs_biotite
def test_a_report_must_name_the_manifest_and_the_artifact_it_describes(tmp_path):
    structure_path, manifest_path, document = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)

    # A manifest file that does not exist is not a manifest this report can name.
    with pytest.raises(ManifestValidationError, match="cannot read manifest"):
        _report_for(prepared, tmp_path / "absent.manifest.json", structure_path)

    other = dict(document, notes="a different declaration of the same file")
    other_path = fx.write_manifest(tmp_path / "other.manifest.json", other)
    with pytest.raises(StructurePreparationError, match="these inputs were prepared from"):
        _report_for(prepared, other_path, structure_path)

    with pytest.raises(StructurePreparationError, match="artifact_content_sha256"):
        _report_for(
            prepared, manifest_path, structure_path, artifact_content_sha256="0" * 64
        )
    with pytest.raises(StructurePreparationError, match="plain basename"):
        _report_for(
            prepared,
            manifest_path,
            structure_path,
            artifact_filename="out/toy.prepared.json",
        )


# --------------------------------------------------------------------------
# Writing, without overwriting
# --------------------------------------------------------------------------

@needs_biotite
def test_an_existing_output_is_refused_and_every_input_survives(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    artifact = prepared_to_document(prepared)
    report = build_report(
        prepared,
        manifest_path=manifest_path,
        structure_path=structure_path,
        artifact_filename="toy.prepared.json",
        artifact_content_sha256=artifact["identity"]["content_sha256"],
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "toy.report.json").write_text("existing", encoding="utf-8")
    before = manifest_path.read_bytes(), structure_path.read_bytes()

    with pytest.raises(StructurePreparationError, match="no overwrite option"):
        write_outputs(
            artifact, report, output_dir=output_dir, name="toy",
            input_paths=(manifest_path, structure_path),
        )

    assert (output_dir / "toy.report.json").read_text(encoding="utf-8") == "existing"
    assert not (output_dir / "toy.prepared.json").exists()
    assert (manifest_path.read_bytes(), structure_path.read_bytes()) == before


@needs_biotite
def test_a_name_with_a_path_separator_is_refused(tmp_path):
    prepared, _ = _prepared(tmp_path)
    artifact = prepared_to_document(prepared)
    for name in ("sub/toy", "..", "../toy", "sub\\toy"):
        with pytest.raises(StructurePreparationError, match="plain file stem"):
            write_outputs(artifact, {}, output_dir=tmp_path / "out", name=name)


@needs_biotite
def test_an_output_may_not_alias_an_input(tmp_path):
    structure_path, manifest_path, _ = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    artifact = prepared_to_document(prepared)
    # The manifest is written as `<dir>/toy.manifest.json`; asking for the stem
    # `toy.manifest` in the same directory would land on it.
    aliased = tmp_path / "alias"
    aliased.mkdir()
    target = aliased / "toy.prepared.json"
    target.write_text("input", encoding="utf-8")
    with pytest.raises(StructurePreparationError, match="never writes over its own inputs"):
        write_outputs(
            artifact, {}, output_dir=aliased, name="toy", input_paths=(target,)
        )
    assert target.read_text(encoding="utf-8") == "input"


@needs_biotite
def test_a_failed_write_removes_only_what_it_created(tmp_path):
    """A document that cannot be encoded must not leave the other file behind."""
    structure_path, manifest_path, _ = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    artifact = prepared_to_document(prepared)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    bystander = output_dir / "unrelated.json"
    bystander.write_text("kept", encoding="utf-8")

    # A set is not JSON-serializable. Both documents are serialized before either
    # file is created, so this fails with nothing on disk -- and it fails as the
    # TypeError the caller's object caused, not as a wrapped path error.
    with pytest.raises(TypeError):
        write_outputs(
            artifact,
            {"unserializable": {1, 2}},
            output_dir=output_dir,
            name="toy",
            input_paths=(manifest_path, structure_path),
        )

    assert not (output_dir / "toy.prepared.json").exists()
    assert not (output_dir / "toy.report.json").exists()
    assert bystander.read_text(encoding="utf-8") == "kept"
    assert manifest_path.exists() and structure_path.exists()


@needs_biotite
def test_an_os_error_on_the_second_write_removes_only_the_first_file(
    tmp_path, monkeypatch
):
    """The failure serialization cannot cover: the disk refusing the second file.

    Injected rather than simulated, so the cleanup path runs for real: the first
    output is created, the second `open` raises, and the invocation removes
    exactly the file it created and nothing else.
    """
    structure_path, manifest_path, _ = _toy(tmp_path)
    prepared = prepare_structure(load_manifest(manifest_path), structure_path)
    artifact = prepared_to_document(prepared)
    report = _report_for(prepared, manifest_path, structure_path)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    bystander = output_dir / "unrelated.json"
    bystander.write_text("kept", encoding="utf-8")

    real_open = builtins.open

    def failing_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("toy.report.json"):
            raise OSError(28, "No space left on device")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(StructurePreparationError, match="cannot write outputs"):
        write_outputs(
            artifact,
            report,
            output_dir=output_dir,
            name="toy",
            input_paths=(manifest_path, structure_path),
        )
    monkeypatch.undo()

    assert not (output_dir / "toy.prepared.json").exists()
    assert not (output_dir / "toy.report.json").exists()
    assert bystander.read_text(encoding="utf-8") == "kept"
    assert manifest_path.exists() and structure_path.exists()


@needs_biotite
def test_a_non_portable_output_name_is_refused(tmp_path):
    """Names that are one file here and something else -- or a device -- on Windows."""
    prepared, _ = _prepared(tmp_path)
    artifact = prepared_to_document(prepared)
    output_dir = tmp_path / "out"
    for name in ("toy:stream", "CON", "nul", "LPT1", "toy.", "toy ", "C:toy", "toy\x01"):
        with pytest.raises(StructurePreparationError):
            write_outputs(artifact, {}, output_dir=output_dir, name=name)
    # The name is checked before anything is created, directory included.
    assert not output_dir.exists()


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def _run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


@needs_biotite
def test_the_cli_writes_both_outputs_and_marks_integration_not_run(tmp_path):
    paths = fx.write_toy_example(tmp_path / "toy")
    output_dir = tmp_path / "prepared"

    result = _run_cli(
        "--manifest", paths["manifest"],
        "--output-dir", str(output_dir),
        "--name", "toy",
    )
    assert result.returncode == 0, result.stderr
    assert "NOT RUN" in result.stdout

    artifact = json.loads((output_dir / "toy.prepared.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "toy.report.json").read_text(encoding="utf-8"))
    assert artifact["kind"] == "esmif1_prepared_structure"
    assert report["model_integration_checks"]["status"] == "not_run"
    assert report["artifact"]["filename"] == "toy.prepared.json"
    assert report["artifact"]["content_sha256"] == artifact["identity"]["content_sha256"]
    # And the written artifact reloads through the strict reader.
    assert prep.load_prepared_structure(output_dir / "toy.prepared.json").num_rows == TOTAL_ROWS


@needs_biotite
def test_the_cli_refuses_a_second_run_into_the_same_directory(tmp_path):
    paths = fx.write_toy_example(tmp_path / "toy")
    output_dir = tmp_path / "prepared"
    assert _run_cli(
        "--manifest", paths["manifest"], "--output-dir", str(output_dir), "--name", "toy"
    ).returncode == 0

    first = (output_dir / "toy.prepared.json").read_bytes()
    result = _run_cli(
        "--manifest", paths["manifest"], "--output-dir", str(output_dir), "--name", "toy"
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr
    assert (output_dir / "toy.prepared.json").read_bytes() == first


@needs_biotite
def test_the_cli_reports_a_hash_mismatch_in_one_line(tmp_path):
    paths = fx.write_toy_example(tmp_path / "toy")
    structure = Path(paths["structure"])
    structure.write_text(
        structure.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n"
    )
    result = _run_cli(
        "--manifest", paths["manifest"],
        "--output-dir", str(tmp_path / "prepared"),
        "--name", "toy",
    )
    assert result.returncode == 2
    reported = [line for line in result.stderr.splitlines() if line.startswith("error: ")]
    assert len(reported) == 1
    assert "manifest declares" in reported[0]
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "prepared" / "toy.prepared.json").exists()


# --------------------------------------------------------------------------
# The policy bridge, on a toy backbone
# --------------------------------------------------------------------------

class _ToyEncoder(nn.Module):
    """Geometry-only, like the real encoder: it never sees the sequence."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, channels + 1, dtype=torch.float32))

    def forward(self, coords, padding_mask, confidence, return_all_hiddens=False):
        finite = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        summary = finite.sum(dim=(-1, -2))  # (B, T)
        states = summary.unsqueeze(-1) * self.scale  # (B, T, C)
        return {
            "encoder_out": [states.transpose(0, 1)],
            "encoder_padding_mask": [padding_mask],
            "encoder_embedding": [],
            "encoder_states": [],
        }


class _ToyDecoder(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(ESMIF1_TOKENS), channels)
        self.project = nn.Linear(channels, len(ESMIF1_TOKENS))
        self.dictionary = None

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        hidden = self.embed(prev_output_tokens)
        context = encoder_out["encoder_out"][0].mean(dim=0)
        logits = self.project(hidden + context.unsqueeze(1))
        return logits.transpose(1, 2), {}


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.encoder = _ToyEncoder()
        self.decoder = _ToyDecoder()


@needs_biotite
def test_binding_names_every_allele_by_its_declared_site_id(tmp_path):
    prepared, _ = _prepared(tmp_path)
    bound = bind_policy(prepared, _ToyModel())

    assert bound.declared_site_ids == ("site_b", "site_a")
    assert bound.policy_site_ids == ("site_a", "site_b")
    assert bound.geometry.num_residues == TOTAL_ROWS

    # The parent sequence: site_a holds "D" (allele 0), site_b holds "G" (allele 0).
    parent = fx.TOY_DECODED_SEQUENCE
    assert bound.alleles_by_site_id(bound.space.alleles_for(parent)) == {
        "site_a": "D",
        "site_b": "G",
    }
    # A variant at site_b only. Read positionally, allele 1 would look like site_a.
    variant = parent[:5] + "S" + parent[6:]
    assert bound.space.alleles_for(variant) == (0, 1)
    assert bound.alleles_by_site_id((0, 1)) == {"site_a": "D", "site_b": "S"}
    assert bound.to_declared_order((0, 1)) == (1, 0)

    scores = bound.policy.log_prob([parent, variant], bound.geometry)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


#: Three sites whose declared order is a 3-cycle away from policy order:
#: declared (site_c@8, site_a@2, site_b@5) against policy (site_a, site_b, site_c).
#: A two-site swap is its own inverse, so it cannot tell `declared_to_policy`
#: from `policy_to_declared`; this one can -- they are (2, 0, 1) and (1, 2, 0).
def _site(site_id: str, index: int, alleles: list[str], detail: str) -> dict:
    return {
        "site_id": site_id,
        "sequence_index": index,
        "allowed_residues": alleles,
        "attribution": {"source": "synthetic fixture", "detail": detail},
    }


THREE_SITES = (
    _site("site_c", 8, ["K", "R"], "declared first, sits last"),
    _site("site_a", 2, ["D", "N"], "declared second, sits first"),
    _site("site_b", 5, ["G", "S"], "declared last, sits middle"),
)


@needs_biotite
def test_three_sites_distinguish_the_permutation_from_its_inverse(tmp_path):
    prepared, _ = _prepared(tmp_path, sites=THREE_SITES)
    manifest = prepared.manifest

    assert manifest.declared_site_ids == ("site_c", "site_a", "site_b")
    assert manifest.policy_site_ids == ("site_a", "site_b", "site_c")
    assert manifest.declared_to_policy == (2, 0, 1)
    assert manifest.policy_to_declared == (1, 2, 0)
    assert manifest.declared_to_policy != manifest.policy_to_declared

    bound = bind_policy(prepared, _ToyModel())
    parent = fx.TOY_DECODED_SEQUENCE
    variant = parent[:8] + "R" + parent[9:]      # site_c only, at position 8
    alleles = bound.space.alleles_for(variant)   # policy order

    assert alleles == (0, 0, 1)
    assert bound.alleles_by_site_id(alleles) == {
        "site_a": "D", "site_b": "G", "site_c": "R",
    }
    # Declared order is (site_c, site_a, site_b), so the edited site comes first.
    assert bound.to_declared_order(alleles) == (1, 0, 0)
    assert bound.to_policy_order(bound.to_declared_order(alleles)) == alleles


@needs_biotite
def test_binding_refuses_a_decoded_chain_that_does_not_start_at_row_zero(tmp_path):
    """Closes the policy spec's 'no residue correspondence is checked' limitation."""
    prepared, _ = _prepared(tmp_path)
    shifted = replace(
        prepared,
        chain_spans=(
            replace(prepared.chain_spans[0], start_row=1, end_row=DECODED_ROWS + 1),
            prepared.chain_spans[1],
        ),
    )
    with pytest.raises(PolicyBindingError, match="must occupy rows 0"):
        bind_policy(shifted, _ToyModel())


@needs_biotite
def test_binding_refuses_prepared_inputs_whose_arrays_were_mutated(tmp_path):
    """The encoder would take these: NaN in a chain row reads as padding upstream."""
    prepared, _ = _prepared(tmp_path)

    coordinates = prepared.coordinates.copy()
    coordinates[3] = np.nan
    with pytest.raises(PolicyBindingError, match="not internally consistent"):
        bind_policy(replace(prepared, coordinates=coordinates), _ToyModel())

    finite_pad = prepared.coordinates.copy()
    finite_pad[DECODED_ROWS + 1] = 0.0
    with pytest.raises(PolicyBindingError, match="not internally consistent"):
        bind_policy(replace(prepared, coordinates=finite_pad), _ToyModel())


# --------------------------------------------------------------------------
# Optional: the real upstream stack, built from scratch, no download
# --------------------------------------------------------------------------

@pytest.fixture
def upstream_stack():
    """Install the ESM compatibility layer, then undo its process-global effects.

    A copy of the fixture in ``test_esmif1_policy.py:1005-1042``; see that
    docstring for why the restoration matters (another file asserts on exactly
    the state `install()` leaves behind).
    """
    from smallAntibodyGen.esmif1_compat import install

    had_scatter = "torch_scatter" in sys.modules
    previous_scatter = sys.modules.get("torch_scatter")
    saved: list[tuple[object, str, object]] = []
    if BIOTITE_AVAILABLE:
        import biotite.structure as bs
        from biotite.structure.io import pdbx

        for owner, name in ((bs, "filter_backbone"), (pdbx, "PDBxFile")):
            saved.append((owner, name, getattr(owner, name, None)))

    install()
    try:
        yield
    finally:
        if had_scatter:
            sys.modules["torch_scatter"] = previous_scatter
        else:
            sys.modules.pop("torch_scatter", None)
        for owner, name, value in saved:
            if value is None:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, value)


def _upstream_model():
    """A ~32-dim `GVPTransformerModel` with random weights. No network access."""
    import argparse

    from esm.data import Alphabet
    from esm.inverse_folding.gvp_transformer import GVPTransformerModel

    args = argparse.Namespace(
        dropout=0.1,
        attention_dropout=0.0,
        encoder_embed_dim=32,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_embed_dim=64,
        decoder_embed_dim=32,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_embed_dim=64,
        gvp_top_k_neighbors=6,
        gvp_num_encoder_layers=1,
        gvp_dropout=0.0,
        gvp_node_hidden_dim_scalar=16,
        gvp_node_hidden_dim_vector=8,
        gvp_edge_hidden_dim_scalar=8,
        gvp_edge_hidden_dim_vector=2,
    )
    alphabet = Alphabet.from_architecture("invariant_gvp")
    torch.manual_seed(0)
    return GVPTransformerModel(args, alphabet).eval(), alphabet


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_the_packing_matches_upstream_concatenate_coords(upstream_stack, tmp_path):
    """Validated against the real function rather than against a comment."""
    from esm.inverse_folding.multichain_util import _concatenate_coords

    prepared, _ = _prepared(tmp_path)
    decoded_span, context_span = prepared.chain_spans
    chains = {
        "H": prepared.coordinates[decoded_span.start_row:decoded_span.end_row],
        "A": prepared.coordinates[context_span.start_row:context_span.end_row],
    }
    upstream = _concatenate_coords(chains, "H", padding_length=PAD_ROWS)

    assert upstream.shape == prepared.coordinates.shape
    assert np.array_equal(upstream, prepared.coordinates, equal_nan=True)


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_a_bound_upstream_policy_scores_samples_and_backpropagates(
    upstream_stack, tmp_path
):
    """Wiring only: the weights are random, so nothing here is a biological result."""
    prepared, _ = _prepared(tmp_path)
    model, alphabet = _upstream_model()
    bound = bind_policy(prepared, model, alphabet=alphabet)

    parent = fx.TOY_DECODED_SEQUENCE
    scores = bound.policy.log_prob([parent], bound.geometry)
    assert torch.isfinite(scores).all()

    generator = torch.Generator().manual_seed(7)
    sample = bound.policy.sample(bound.geometry, num_samples=4, generator=generator)
    rescored = bound.policy.log_prob(sample.sequences, bound.geometry)
    assert torch.allclose(sample.log_probability, rescored, atol=1e-5)
    for sequence, alleles in zip(sample.sequences, sample.alleles):
        named = bound.alleles_by_site_id(alleles)
        assert named["site_a"] == sequence[2]
        assert named["site_b"] == sequence[5]

    scores.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in bound.policy.model.decoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in bound.policy.model.encoder.parameters()
    )


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_the_cached_geometry_equals_an_uncached_native_forward(upstream_stack, tmp_path):
    """The parity claim at the *prepared multichain* boundary.

    `test_esmif1_policy.py` pins the same equivalence for a single chain built by
    hand. What is new here is the input: packed multichain coordinates with NaN
    pad rows and a **per-chain** confidence vector, both produced by
    `prepare_structure` and handed to the encoder by `bind_policy`. A confidence
    vector dropped on either side of that handoff -- or a decoded chain read at
    the wrong offset in the packing -- changes these logits and nothing else in
    this file would notice.

    Compared with `assert_close`, not bit-for-bit: the float32 tolerance covers
    reassociation between the cached and uncached paths, nothing about the
    contract.
    """
    from esm.inverse_folding.util import CoordBatchConverter

    prepared, _ = _prepared(tmp_path, confidence={"H": 0.65, "A": 0.9})
    # The point of the non-default values: an ignored vector would still pass a
    # comparison run at the all-ones default.
    assert not np.all(prepared.confidence == 1.0)

    model, alphabet = _upstream_model()
    bound = bind_policy(prepared, model, alphabet=alphabet)
    parent = fx.TOY_DECODED_SEQUENCE

    got = bound.policy.native_logits([parent], bound.geometry)

    coords, confidence, _, tokens, padding_mask = CoordBatchConverter(alphabet)(
        [(prepared.coordinates, prepared.confidence, parent)]
    )
    with torch.no_grad():
        # `tokens[:, :-1]` is upstream's teacher-forcing shift; the second slice
        # keeps the decoded chain's columns, which are the only decisions the
        # policy makes.
        expected, _ = model.forward(
            coords, padding_mask, confidence, tokens[:, :-1][:, :len(parent)]
        )

    assert got.shape == expected.shape
    assert got.shape[-1] == len(parent)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not ESM_IF1_STACK, reason="optional 'esm-if1' extra not installed")
def test_binding_encodes_the_geometry_once_and_reuses_it(upstream_stack, tmp_path):
    """One encoding per bind: the encoder never sees the sequence, so reuse is exact.

    Counted at the encoder module itself, with the hook installed *before*
    `bind_policy`: a stable digest across scoring calls is consistent with a
    re-encode that lands on the same value, so it is not on its own evidence that
    the encoder ran once.
    """
    prepared, _ = _prepared(tmp_path)
    model, alphabet = _upstream_model()

    encoder_calls: list[object] = []
    handle = model.encoder.register_forward_hook(
        lambda module, inputs, output: encoder_calls.append(module)
    )
    try:
        bound = bind_policy(prepared, model, alphabet=alphabet)

        before = bound.geometry.digest
        bound.policy.log_prob([fx.TOY_DECODED_SEQUENCE], bound.geometry)
        bound.policy.native_logits([fx.TOY_DECODED_SEQUENCE], bound.geometry)
        bound.policy.log_prob(bound.space.enumerate_sequences()[:2], bound.geometry)
    finally:
        handle.remove()

    assert len(encoder_calls) == 1
    assert bound.geometry.digest == before
    assert bound.geometry.num_residues == TOTAL_ROWS
    # Upstream pads one flank at each end of the coordinates.
    assert bound.geometry.encoder_out.shape[0] == TOTAL_ROWS + 2
    assert not bound.geometry.encoder_out.requires_grad


# --------------------------------------------------------------------------
# Regressions: BR-01..08, reproduced before they were fixed
# --------------------------------------------------------------------------

def _relabelled(manifest):
    """A manifest whose parsed support reads ``G/T`` while its snapshot reads ``G/S``.

    What `dataclasses.replace` makes available: new parsed fields, the *old*
    canonical text, and therefore the old digest.
    """
    site = manifest.sites[0]
    return replace(
        manifest,
        sites=(replace(site, allowed_residues=("G", "T")),) + manifest.sites[1:],
    )


def test_a_manifest_rebuilt_field_by_field_is_not_the_one_its_digest_pins(tmp_path):
    """BR-01. The snapshot is authoritative; the parsed fields are what callers act on."""
    _, manifest_path, _ = _toy(tmp_path)
    manifest = load_manifest(manifest_path)
    assert decl.revalidate_manifest(manifest) == manifest

    relabelled = _relabelled(manifest)
    assert relabelled.sites[0].allowed_residues == ("G", "T")
    assert relabelled.digest == manifest.digest  # the text, and the hash, are untouched
    with pytest.raises(ManifestValidationError, match="canonical snapshot"):
        decl.revalidate_manifest(relabelled)

    # A snapshot that is valid but not canonical is refused too: the digest would
    # be over text this package never writes.
    document = manifest.document
    with pytest.raises(ManifestValidationError, match="canonical snapshot"):
        decl.revalidate_manifest(
            replace(manifest, canonical_text=json.dumps(document) + "\n")
        )


@needs_biotite
def test_every_boundary_refuses_a_prepared_object_carrying_a_relabelled_manifest(tmp_path):
    """BR-01. The live edit space said G/T while the artifact reloaded G/S."""
    prepared, structure_path = _prepared(tmp_path)
    manifest_path = tmp_path / "toy.manifest.json"
    tampered = replace(prepared, manifest=_relabelled(prepared.manifest))

    with pytest.raises(PolicyBindingError, match="digest pins"):
        build_edit_space(tampered)
    with pytest.raises(PolicyBindingError):
        bind_policy(tampered, _ToyModel())
    with pytest.raises(PreparedArtifactError, match="validated declaration"):
        prep.check_prepared_state(tampered)
    with pytest.raises(PreparedArtifactError, match="validated declaration"):
        prepared_to_document(tampered)
    with pytest.raises(StructurePreparationError, match="validated declaration"):
        prepare_structure(tampered.manifest, structure_path)
    assert verify_against_source(tampered, structure_path)
    with pytest.raises(prep.StructureAdapterError):
        _report_for(tampered, manifest_path, structure_path)


@needs_biotite
def test_the_live_prepared_state_refuses_what_an_artifact_reload_refuses(tmp_path):
    """BR-01. `check_prepared_state` was blind to three things a reload catches."""
    prepared, _ = _prepared(tmp_path)
    prep.check_prepared_state(prepared)  # the honest object passes

    other = hashlib.sha256(b"a different file").hexdigest()
    with pytest.raises(PreparedArtifactError, match="the manifest pins"):
        prep.check_prepared_state(replace(prepared, structure_sha256=other))
    with pytest.raises(PreparedArtifactError, match="lowercase hex"):
        prep.check_prepared_state(
            replace(prepared, structure_sha256=prepared.structure_sha256 + "\n")
        )

    context = prepared.chain_spans[1]
    with pytest.raises(PreparedArtifactError, match="empty or negative"):
        prep.check_prepared_state(
            replace(
                prepared,
                chain_spans=(
                    prepared.chain_spans[0],
                    replace(context, end_row=context.start_row, num_residues=0),
                ),
            )
        )

    invented = prep.PreparationFindings(
        numbering_gaps=(
            {
                "res_id_after": 10,
                "res_id_before": 5,
                "sequence_index_after": 5,
                "sequence_index_before": 4,
            },
        ),
        declared_sequence_mismatches=(),
        decoded_chain_residue_count=DECODED_ROWS,
        context_chain_residue_counts={fx.TOY_CONTEXT_CHAIN: CONTEXT_ROWS},
    )
    with pytest.raises(PreparedArtifactError, match="findings"):
        prep.check_prepared_state(replace(prepared, findings=invented))


@needs_biotite
def test_mmcif_models_must_each_occupy_one_contiguous_run(tmp_path):
    """BR-02. Interleaved serials put two models' coordinates in one ordinal."""
    decoded = fx.toy_decoded_residues()
    context = fx.toy_context_residues()
    residues = decoded + context
    rows = fx.atom_row_count(residues, num_models=2)
    half = rows // 2
    # model 1, model 2, model 1 again: biotite's slice between first occurrences
    # would hand ordinal 2 the rows of both.
    serials = ["1"] * half + ["2"] * (half - 16) + ["1"] * 16
    structure_path = tmp_path / "toy_complex.cif"
    fx.write_cif(
        structure_path,
        residues,
        num_models=2,
        column_overrides={"pdbx_PDB_model_num": serials},
    )
    document = fx.build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format="mmcif",
        decoded_residues=decoded,
        model_ordinal=2,
    )
    manifest = validate_manifest(document)
    with pytest.raises(StructurePreparationError, match="two separate runs"):
        prepare_structure(manifest, structure_path)


@needs_biotite
def test_a_model_number_that_is_not_an_integer_fails_by_name(tmp_path):
    """BR-02/BR-06. `as_array(np.int32)` wrapped 2**31 and raised on 'first'."""
    decoded = fx.toy_decoded_residues()
    context = fx.toy_context_residues()
    residues = decoded + context
    rows = fx.atom_row_count(residues)
    structure_path = tmp_path / "toy_complex.cif"
    fx.write_cif(
        structure_path,
        residues,
        column_overrides={"pdbx_PDB_model_num": ["first"] * rows},
    )
    document = fx.build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format="mmcif",
        decoded_residues=decoded,
    )
    with pytest.raises(StructurePreparationError, match="not an integer model number"):
        prepare_structure(validate_manifest(document), structure_path)

    # A serial past int32 is parsed exactly rather than coerced: the file still
    # holds two models, so ordinal 3 is refused as being past the end.
    wrapped = tmp_path / "wrapped.cif"
    fx.write_cif(
        wrapped,
        residues,
        column_overrides={
            "pdbx_PDB_model_num": ["1"] * (rows // 2) + [str(2**31 + 1)] * (rows - rows // 2)
        },
    )
    wrapped_document = fx.build_manifest_document(
        relative_path=wrapped.name,
        structure_path=wrapped,
        structure_format="mmcif",
        decoded_residues=decoded,
        model_ordinal=3,
    )
    with pytest.raises(StructurePreparationError, match="2 model"):
        prepare_structure(validate_manifest(wrapped_document), wrapped)


@needs_biotite
def test_an_mmcif_record_kind_outside_atom_and_hetatm_is_rejected(tmp_path):
    """BR-03. `hetero` is `group_PDB == 'HETATM'`, so anything else read as ATOM."""
    decoded = fx.toy_decoded_residues()
    context = fx.toy_context_residues()
    residues = decoded + context
    structure_path = tmp_path / "toy_complex.cif"
    fx.write_cif(
        structure_path,
        residues,
        column_overrides={
            "group_PDB": ["INVALID"] * fx.atom_row_count(residues)
        },
    )
    document = fx.build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format="mmcif",
        decoded_residues=decoded,
    )
    with pytest.raises(StructurePreparationError, match="group_PDB"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
@pytest.mark.parametrize("structure_format", ("pdb", "mmcif"))
def test_a_residue_split_by_another_chains_records_is_refused(tmp_path, structure_format):
    """BR-04. Filtering to the chain first made two blocks adjacent, and one residue."""
    decoded = fx.toy_decoded_residues()
    context = fx.toy_context_residues()
    head, tail = fx.split_residue_atoms(decoded, 0, 1)
    file_residues = (head,) + tuple(context) + (tail,) + tuple(decoded[1:])

    structure_path = _write_structure(
        tmp_path, file_residues, structure_format=structure_format
    )
    document = fx.build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format="pdb" if structure_format == "pdb" else "mmcif",
        decoded_residues=decoded,
    )
    # Splitting the run may expose its incomplete backbone before the repeated
    # residue key is reached. Both diagnostics must reject the input; neither
    # may merge the fragments back into a complete residue.
    with pytest.raises(StructurePreparationError, match="two separate|has no CA atom"):
        prepare_structure(validate_manifest(document), structure_path)


@needs_biotite
def test_a_residue_split_by_a_ter_record_is_refused(tmp_path):
    """BR-04. biotite drops TER lines, so the two segments became adjacent rows."""
    decoded = fx.toy_decoded_residues()
    context = fx.toy_context_residues()
    residues = tuple(decoded) + tuple(context)
    structure_path = tmp_path / "toy_complex.pdb"
    # After the first residue's N: its CA and C are then in the next segment.
    fx.write_pdb(structure_path, residues, ter_after_atoms=1)
    document = fx.build_manifest_document(
        relative_path=structure_path.name,
        structure_path=structure_path,
        structure_format="pdb",
        decoded_residues=decoded,
    )
    with pytest.raises(StructurePreparationError, match="TER"):
        prepare_structure(validate_manifest(document), structure_path)


def test_a_manifest_that_cannot_be_encoded_as_utf8_is_refused(tmp_path):
    """BR-05. A lone surrogate validated, then raised UnicodeEncodeError at the digest."""
    _, _, document = _toy(tmp_path)
    document["notes"] = "\ud800"
    with pytest.raises(ManifestValidationError, match="UTF-8"):
        validate_manifest(document)


@needs_biotite
def test_a_report_that_cannot_be_encoded_leaves_no_partial_output(tmp_path):
    """BR-05. The encode used to happen inside the write, after both files existed."""
    prepared, structure_path = _prepared(tmp_path)
    artifact = prepared_to_document(prepared)
    output_dir = tmp_path / "out"

    with pytest.raises(StructurePreparationError, match="UTF-8"):
        write_outputs(
            artifact,
            {"note": "\ud800"},
            output_dir=output_dir,
            name="toy",
            input_paths=(structure_path,),
        )
    assert not output_dir.exists()


def test_a_manifest_or_artifact_that_is_not_utf8_is_a_domain_error(tmp_path):
    """BR-06. `read_text` raises UnicodeDecodeError, which is not an OSError."""
    bad = tmp_path / "toy.manifest.json"
    bad.write_bytes(b'{"notes": "\xff\xfe"}')
    with pytest.raises(ManifestValidationError, match="UTF-8"):
        load_manifest(bad)
    with pytest.raises(PreparedArtifactError, match="UTF-8"):
        prep.load_prepared_structure(bad)


def test_a_confidence_with_no_float_image_is_refused_by_name(tmp_path):
    """BR-06. `float(10**400)` raises OverflowError, which is not a ValueError."""
    _, _, document = _toy(tmp_path)
    document["chains"]["confidence"][fx.TOY_DECODED_CHAIN] = 10 ** 400
    with pytest.raises(ManifestValidationError, match="confidence"):
        validate_manifest(document)


@needs_biotite
def test_the_cli_exits_two_without_a_traceback_on_a_malformed_manifest(tmp_path):
    """BR-06. Every expected input failure is one line on stderr and exit 2."""
    paths = fx.write_toy_example(tmp_path / "toy")
    Path(paths["manifest"]).write_bytes(b'{"notes": "\xff\xfe"}')
    result = _run_cli(
        "--manifest", paths["manifest"],
        "--output-dir", str(tmp_path / "prepared"),
        "--name", "toy",
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr


@needs_biotite
def test_an_allele_index_must_be_an_actual_integer(tmp_path):
    """BR-07. `True in (0, 1)` is True, and a float indexed the allele tuple raw."""
    prepared, _ = _prepared(tmp_path)
    bound = bind_policy(prepared, _ToyModel())

    assert bound.alleles_by_site_id((0, 1)) == {"site_a": "D", "site_b": "S"}
    assert bound.alleles_by_site_id(np.array([0, 1])) == {"site_a": "D", "site_b": "S"}
    assert bound.alleles_by_site_id(torch.tensor([0, 1])) == {
        "site_a": "D",
        "site_b": "S",
    }
    with pytest.raises(PolicyBindingError, match="boolean"):
        bound.alleles_by_site_id((True, False))
    with pytest.raises(PolicyBindingError, match="not an integer"):
        bound.alleles_by_site_id((0.0, 1.0))
    with pytest.raises(PolicyBindingError, match="outside"):
        bound.alleles_by_site_id((0, 2))


@pytest.mark.parametrize(
    "relative_path",
    ("bad?.pdb", "bad*.pdb", 'bad".pdb', "bad<.pdb", "bad|.pdb", "COM¹", "LPT³.pdb"),
)
def test_a_name_windows_refuses_is_refused_here(tmp_path, relative_path):
    """BR-08. Wildcards, forbidden characters and the superscript device names."""
    _, _, document = _toy(tmp_path)
    document["source"]["relative_path"] = relative_path
    with pytest.raises(ManifestValidationError, match="relative_path"):
        validate_manifest(document)


def test_the_strict_validators_are_anchored(tmp_path):
    """BR-08. `$` matches before a trailing newline, so `match` accepted one."""
    _, _, document = _toy(tmp_path)
    trailing_newline = copy.deepcopy(document)
    trailing_newline["source"]["sha256"] += "\n"
    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_manifest(trailing_newline)

    insertion_code = copy.deepcopy(document)
    insertion_code["correspondence"][0]["ins_code"] = "\n"
    with pytest.raises(ManifestValidationError, match="ins_code"):
        validate_manifest(insertion_code)


@needs_biotite
def test_an_artifact_index_written_as_a_boolean_is_detected(tmp_path):
    """BR-08. `[True, False] == [1, 0]`, so the comparison alone could not see it."""
    prepared, _ = _prepared(tmp_path)
    document = prepared_to_document(prepared)
    assert document["site_permutations"]["declared_to_policy"] == [1, 0]

    document["site_permutations"]["declared_to_policy"] = [True, False]
    with pytest.raises(PreparedArtifactError, match="declared_to_policy"):
        document_to_prepared(_reseal(document))

    count = prepared_to_document(prepared)
    count["findings"]["context_chain_residue_counts"][fx.TOY_CONTEXT_CHAIN] = float(
        CONTEXT_ROWS
    )
    with pytest.raises(PreparedArtifactError, match="context_chain_residue_counts"):
        document_to_prepared(_reseal(count))
