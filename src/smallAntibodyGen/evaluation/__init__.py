"""Frozen-input evaluation: judge what training CHANGES, not what a run reports.

Every number in this package is produced by scoring a checkpoint against a
**frozen artifact** -- corrupted inputs, labels, masks and donor assignments
materialized once and saved to disk. Scoring never regenerates a mask. Two
checkpoints compared through this package are therefore compared on byte
identical inputs, and a comparison stays reproducible across later changes to
the collator, which re-seeding alone would not survive.

See `frozen_inputs` for the artifact and its invariants.
"""
from smallAntibodyGen.evaluation.frozen_inputs import (
    ABSENT_CHAIN_REMOVED,
    ABSENT_SYNTHETIC_FILLER,
    CONDITIONS,
    DIRECTIONS,
    FROZEN_SCHEMA,
    DonorPolicy,
    EmptyProbeError,
    ExclusionLedger,
    FrozenCase,
    FrozenInputsError,
    PositionDriftError,
    build_frozen_benchmark,
    derive_artifact_id,
    load_frozen,
    save_frozen,
    semantic_digest,
    tensor_digest,
    verify_frozen,
)

__all__ = [
    "ABSENT_CHAIN_REMOVED",
    "ABSENT_SYNTHETIC_FILLER",
    "CONDITIONS",
    "DIRECTIONS",
    "FROZEN_SCHEMA",
    "DonorPolicy",
    "EmptyProbeError",
    "ExclusionLedger",
    "FrozenCase",
    "FrozenInputsError",
    "PositionDriftError",
    "build_frozen_benchmark",
    "derive_artifact_id",
    "load_frozen",
    "save_frozen",
    "semantic_digest",
    "tensor_digest",
    "verify_frozen",
]
