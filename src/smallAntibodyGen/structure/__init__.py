"""Declared structural input for the ESM-IF1 constrained editing policy.

The layer between "a local PDB/mmCIF file someone downloaded" and the
``(L, 3, 3)`` N/CA/C array `ConstrainedEditPolicy.encode_geometry` wants. It
pins the source by hash, makes every choice the parsers would otherwise make
silently into a declared field, and fails closed on anything unsupported or
ambiguous.

- :mod:`~smallAntibodyGen.structure.declaration` -- the manifest schema and its
  validator. **Pure stdlib.**
- :mod:`~smallAntibodyGen.structure.prepare` -- reading, packing, the portable
  artifact and the preparation report. numpy plus **lazily imported** biotite.
- :mod:`~smallAntibodyGen.structure.policy_adapter` -- the bridge to
  `ConstrainedEditPolicy`. Imports torch, so it is **not** re-exported here:
  import it by name when you need it.

Importing this package pulls in neither biotite nor torch, which
`test_esmif1_structure.py` checks in a subprocess. `specs/esmif1_structure.md`
owns the schema, the exact supported and rejected cases, and the limitations.

Nothing here selects a benchmark structure, reconstructs a site map, fetches
anything, or loads any weights.
"""

from __future__ import annotations

from .declaration import (
    CANONICAL_RESIDUES,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    THREE_TO_ONE,
    ManifestValidationError,
    StructureAdapterError,
    StructureManifest,
    canonical_json,
    document_digest,
    load_manifest,
    revalidate_manifest,
    sha256_file,
    validate_manifest,
)
from .prepare import (
    PACKING_CONVENTION,
    PAD_CONFIDENCE,
    PREPARED_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    ChainSpan,
    PreparationFindings,
    PreparedArtifactError,
    PreparedStructure,
    StructurePreparationError,
    build_report,
    check_prepared_state,
    document_to_prepared,
    load_prepared_structure,
    prepare_structure,
    prepared_to_document,
    structure_path_for,
    verify_against_source,
    write_outputs,
)

__all__ = [
    "CANONICAL_RESIDUES",
    "MANIFEST_KIND",
    "MANIFEST_SCHEMA_VERSION",
    "PACKING_CONVENTION",
    "PAD_CONFIDENCE",
    "PREPARED_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "THREE_TO_ONE",
    "ChainSpan",
    "ManifestValidationError",
    "PreparationFindings",
    "PreparedArtifactError",
    "PreparedStructure",
    "StructureAdapterError",
    "StructureManifest",
    "StructurePreparationError",
    "build_report",
    "canonical_json",
    "check_prepared_state",
    "document_digest",
    "document_to_prepared",
    "load_manifest",
    "load_prepared_structure",
    "prepare_structure",
    "prepared_to_document",
    "revalidate_manifest",
    "sha256_file",
    "structure_path_for",
    "validate_manifest",
    "verify_against_source",
    "write_outputs",
]
