"""Acquired-only frozen-feature comparison matrix: a CPU, NumPy-only, opt-in runner.

What this is. Several FROZEN feature representations of one candidate roster (each shipped as a
versioned cache: a strict JSON header plus a numeric NPZ) are crossed with fixed readout maps, label
budgets and seeds. Every cell is fitted by the accepted baseline (``functional_baseline``): ridge
reward, linear-energy DPO, exact tilt and a finite SIR, frozen into a hashed artifact, and only THEN
scored against the held-out outcomes. What it is not: a native extractor (nothing here imports,
discovers or loads a model), a claim that any representation or readout wins, or evidence about
biology. No winning cell, ranking of cells or headline number is selected.

Feature cache (``functional-feature-cache/1``). ``write_feature_cache`` / ``load_feature_cache``::

    header JSON (strict: duplicate keys, NaN and unknown keys are refused)
      schema, representation_id (safe id), identity {kind "complete_sequence", alphabet, length,
      construct_description}, rows {ids, sequences, groups, splits}  (one entry per cache row, any
      order), feature_space (a FeatureSpace document: kind, names, schema, history, provenance =
      the feature SupervisionHistory), storage {file, file_sha256, array "features", dtype
      float32|float64, shape [rows, dimension], content_sha256}, extractor {kind, checkpoint_sha256,
      source_sha256, description}, activation {location, pooling, token_map}, conditioning
      {candidate_masking "none", answer_free true, family, assay, quality}, cache_sha256
    NPZ: exactly one numeric array "features" (rows, dimension); loaded with allow_pickle=False.

* Identity is the COMPLETE fixed-length physical sequence, canonical (stripped, upper-case), over a
  declared alphabet. It is verified on BOTH sides: the cache and the candidate CSV must agree on
  alphabet, length, and each id's full sequence, group and split. An edited window is not an
  identity, so two constructs that share an edited window can never be joined by it. A structured
  heavy/light identity is NOT supported and is refused explicitly (declare a paired construct as
  one complete sequence with a delimiter in the alphabet and describe it in construct_description).
* The roster join is exact: no missing, extra or duplicate candidate, and no convenient
  intersection. A cache carries no prior; the reference prior is authoritative once, in the
  candidate CSV with the prior provenance of the matrix manifest.
* Conditioning is recorded per row for family, assay and quality. Each field is ``{"recorded":
  false}`` (ABSENT) or ``{"recorded": true, "values": [...]}`` where a value may be an explicit
  null (the model was given "no conditioning"), which is a different statement from absence.
  Prediction-time quality must be recorded and must be explicit null for every row: an
  outcome-derived high/low bin is ineligible and absence cannot be verified answer-free.
* ``cache_sha256`` covers the canonical (sorted-id) rows, contracts, feature space and feature
  bytes, so it is row-order invariant; ``storage.content_sha256`` / ``file_sha256`` bind the stored
  array and file. A hash is integrity, not proof the declared extractor/checkpoint/conditioning
  statements are true.
* The header names its NPZ by a plain file name in its own directory (a path, separator, drive or
  ``..`` is refused), or the caller passes the NPZ path explicitly.

Matrix manifest (``functional-matrix-manifest/1``): target, assay, utility {units, scale,
higher_is_better true, useful_threshold}, grouping_rationale, candidates {identity_kind
"complete_sequence", alphabet, length}, prior {kind, semantics, provenance}, optional notes.

Matrix plan (``functional-matrix-plan/1``), frozen BEFORE anything is fitted: representations (ids),
maps, label_budgets (ints or null = all), seeds, baseline (BaselineConfig options minus seed and
label budget; ``tau`` required, everything else fixed, nothing tuned), reference {representation,
map} and bootstrap {replicates, seed, confidence, min_groups}. A map is ``{"id", "kind": "linear" |
"rbf", "center", "scale": "none" | "std", "trace_normalize", ["bandwidth": {"rule":
"median_heuristic", "multiplier"} | {"rule": "fixed", "value"}]}``.

Readout maps (mathematics). The acquired indices come from ``acquire_labels(len(offered), budget,
seed)`` BEFORE any fitting. From the acquired feature rows ONLY: the column mean and (optionally)
population standard deviation, the RBF centers (the distinct acquired rows in standardised units)
and the bandwidth ``sigma = multiplier * median positive pairwise center distance`` (or a fixed
positive input); no label is read. The map is then applied to EVERY offered and held-out row::

    linear   z = (x - mean) / scale
    rbf      phi_k(x) = exp(-||z(x) - c_k||^2 / (2 sigma^2)),  one feature per distinct center
    both     times one scalar g = 1 / sqrt(mean squared norm of the centred acquired design)

The scalar ``g`` (trace normalisation) puts every readout on one overall scale; it does NOT
equalise regularisation geometry, because spectra differ and one ridge alpha shrinks different
directions differently. The cell therefore reports the output dimension, the design rank and the
effective ridge degrees of freedom ``sum s^2 / (s^2 + alpha)`` (``s`` the singular values of the
centred acquired design), plus deterministic operation counts (no wall-clock time). Rank and df are
taken from the ACTUAL transformed acquired rows, after the multiplication by ``g`` and its rounding,
because those are the rows that are stored, fitted and re-verified; the realized centred variance of
those rows is recorded beside ``g``. It is 1 up to rounding for a well-conditioned design and is not
promised to be exactly 1 for an ill-conditioned one (a few-ulp variation on a large offset makes
``g`` ~ 1e15, and the product rounds).

Constant columns (an EXACT policy, no tolerance). A column is constant only if its acquired values
are bit-identical. A tolerance would silently discard real variation that is merely small in its
units: ``[0, 1e-13]`` is a genuine direction (standardisation maps it to ``[-1, +1]``), so it is
mapped and fitted like any other. A constant column is kept with scale 1 and listed; with
``center=True`` its mean is that exact value, so its acquired coordinate is exactly zero and ridge
and DPO give it weight exactly zero (with ``center=False`` it keeps its raw value, a constant offset
the ridge intercept absorbs). A column that is NOT constant but whose range is within
``NEAR_CONSTANT_TOLERANCE`` (relative to ``max(1, |value|)``) is kept as real variation and only
listed under ``near_constant_columns``, so a units-driven tiny range stays visible instead of being
dropped. Centering of columns and of the mapped design is taken relative to an anchor (the first
acquired row): an exactly constant column or an all-identical design therefore centres to EXACTLY
zero (rank 0, df 0, degenerate, factor 1) instead of to a rounding residue (the mean of three copies
of 0.1 is not 0.1) that trace normalisation would amplify by 1e16. Exact duplicate acquired rows give
ONE RBF center and are counted; all-identical acquired rows leave a degenerate (zero) design, and the
median heuristic is then undefined, so that cell is recorded ``unsupported`` with the reason. The
baseline then fits ridge on the mapped acquired rows and the DPO policy is linear in the MAPPED
basis, not necessarily in the raw features. Unlabelled and unacquired covariates are not used by
the fit.

Order of operations and outputs (``run_matrix``). Alias checks and parsing first; then every cell
is built in memory; then ``cells/cell_*.json`` (one per cell, including ``unsupported`` and
``failed`` ones with an explicit reason), ``matrix_inputs.json`` (raw file digests of the
non-evaluation inputs) and ``matrix_freeze_index.json`` (every cell, hashed together with the
common support/prior/utility/plan identity) are written; they are RELOADED and VERIFIED
(``verify_freeze``); and only then is the evaluation CSV opened. ``verify_freeze`` also binds each
cell's artifact to ITS OWN claims: the artifact's representation, readout map, canonical cache
digest, cache feature-space fingerprint, extractor/activation/conditioning, acquisition and
capacity must equal what the cell key, the plan and the index record, so a valid artifact of another
representation or map (or of another cache) cannot stand in for a cell even if every hash above it is
recomputed. A hash is integrity, not authenticity: someone who rewrites the WHOLE chain consistently
is not detected. The public ``evaluate_matrix`` accepts no caller-held state: it verifies the
current directory on disk itself, refuses an evaluation file that aliases any frozen file or the
report (or its ``.tmp``) before opening it, refuses a pre-existing report ``.tmp`` even with
``overwrite=True``, and replaces an existing report only with
``overwrite=True``; a missing evaluation file after ``run_matrix`` therefore leaves a complete
verified freeze that ``evaluate_matrix`` (CLI: ``evaluate``) can score later. ``matrix_evaluation_report.json``
holds each successful cell's full ``evaluate_frozen`` report and a paired per-group bootstrap of
the readout MSE difference against the declared reference cell at the SAME budget and seed (the
acquisition is identical). Repeated seeds share candidates and are not independent samples: each
seed is its own entry and nothing is pooled; there is no multiplicity adjustment. Changing the
evaluation outcomes leaves every pre-evaluation file byte-identical.

Usage::

    python scripts/functional_matrix.py selftest --output-dir OUT
    python scripts/functional_matrix.py run --candidates C.csv --train-outcomes T.csv \\
        --eval-outcomes E.csv --manifest M.json --plan P.json --cache rep_a=A.cache.json \\
        --cache rep_b=B.cache.json --output-dir OUT
    python scripts/functional_matrix.py evaluate --output-dir OUT --eval-outcomes E.csv

Scope limits (documented, not hidden). The baseline's ridge Gram matrix and DPO Hessian are dense
d x d systems: ``functional_baseline.MAX_DENSE_SYSTEM_ELEMENTS`` (4096 x 4096 by default) bounds d
and ``MAX_PAIR_MATRIX_ELEMENTS`` bounds pairs x d, both checked before anything is allocated. They
bound one system each, not peak memory. The readout checks the same d bound before it maps
anything (the RBF output dimension is its number of distinct acquired centers); a cell above it is
recorded ``failed`` with the reason. The mapped design (offered + held-out rows) x output dimension
is NOT separately bounded. RBF centers are all distinct acquired rows (no subsampling). Cache
writing needs the caller's features; native extraction and shared adapters are separate modules.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import functional_baseline as fb
from .her2_runtime import require, sha256 as file_sha256

CACHE_SCHEMA = "functional-feature-cache/1"
MANIFEST_SCHEMA = "functional-matrix-manifest/1"
PLAN_SCHEMA = "functional-matrix-plan/1"
CELL_SCHEMA = "functional-matrix-cell/1"
INDEX_SCHEMA = "functional-matrix-freeze-index/1"
INPUTS_SCHEMA = "functional-matrix-inputs/1"
RESULTS_SCHEMA = "functional-matrix-evaluation/1"
SELFTEST_SCHEMA = "functional-matrix-selftest/1"

CELLS_DIR = "cells"
INDEX_NAME = "matrix_freeze_index.json"
INPUTS_NAME = "matrix_inputs.json"
RESULTS_NAME = "matrix_evaluation_report.json"
SELFTEST_REPORT_NAME = "matrix_selftest_report.json"

IDENTITY_COMPLETE_SEQUENCE = "complete_sequence"
EXTRACTOR_KINDS = ("native_checkpoint", "synthetic", "external_features")
STORED_DTYPES = ("float32", "float64")
CACHE_ARRAY = "features"
CONDITIONING_FIELDS = ("family", "assay", "quality")

MAP_LINEAR, MAP_RBF = "linear", "rbf"
MAP_KINDS = (MAP_LINEAR, MAP_RBF)
SCALE_MODES = ("none", "std")
BANDWIDTH_MEDIAN, BANDWIDTH_FIXED = "median_heuristic", "fixed"

STATUS_OK, STATUS_UNSUPPORTED, STATUS_FAILED = "ok", "unsupported", "failed"
STATUSES = (STATUS_OK, STATUS_UNSUPPORTED, STATUS_FAILED)

#: DIAGNOSTIC ONLY. A column that is not exactly constant but whose acquired range is within this
#: tolerance of zero (relative to max(1, |value|)) is listed as ``near_constant``. It is never
#: treated as constant: constancy is exact, so real variation is not discarded for its units.
NEAR_CONSTANT_TOLERANCE = 1e-12
#: The keys of ``FeatureCache.summary()`` that an index records per representation.
REPRESENTATION_SUMMARY_KEYS = (
    "representation_id", "cache_sha256", "rows", "feature_dimension", "dtype", "feature_kind",
    "feature_space_sha256", "feature_supervision", "identity", "extractor", "activation",
    "conditioning")
#: Element budget of one block of the pairwise-difference tensor behind the squared distances.
DISTANCE_BLOCK_ELEMENTS = 2_000_000

BASELINE_OPTIONS = ("tau", "ridge_alpha", "declared_validation_labels", "dpo_beta", "dpo_l2",
                    "dpo_max_pairs", "dpo_max_iterations", "dpo_tolerance", "dpo_pressures",
                    "sir_proposal", "sir_proposal_draws", "final_draws")

_SAFE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,30}[A-Za-z0-9])?")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_NPZ_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}\.npz")
_LITTLE_ENDIAN = {"float32": "<f4", "float64": "<f8"}
_PLACEHOLDER_HASH = "0" * 64

CLAIMS = {
    "scope": ("A comparison of declared frozen feature representations through fixed readout maps "
              "on a finite candidate support, fitted by the finite-support baseline. It is not "
              "native generation, not a statement of biological improvement, and it selects no "
              "winning cell."),
    "no_winner": ("No winning cell, ranking of cells or headline number is selected. Cells are "
                  "reported side by side; a null or noise representation is not required to lose "
                  "and a learned one is not required to win."),
    "acquired_only": ("Centering, scaling, RBF centers and the bandwidth use the feature rows of "
                      "the ACQUIRED labelled candidates only. Unlabelled training covariates and "
                      "unacquired offered rows are not used by any fit; they are still mapped and "
                      "kept (offered and held-out counts are unchanged)."),
    "regularisation": ("Trace normalisation fixes the overall scale of each mapped design; it does "
                       "NOT equalise regularisation geometry. The spectra differ across "
                       "representations and maps, so one ridge alpha shrinks them differently: "
                       "compare the reported output dimension and effective ridge df, and do not "
                       "read equal N or equal alpha as equal information or capacity."),
    "dpo_basis": ("DPO is linear in the MAPPED basis (the linear-energy policy over the readout "
                  "features), not necessarily in the raw cached features. Its implicit energy is "
                  "not calibrated utility."),
    "provenance": ("Cache, prior and label provenance is DECLARED, not verified. Hashes prove the "
                   "files were not edited after writing; they do not prove the extractor, "
                   "checkpoint, conditioning or supervision statements are true."),
    "uncertainty": ("The MSE-difference interval is a percentile bootstrap that resamples SUPPORT "
                    "GROUPS with replacement, paired with the reference cell at the same budget "
                    "and seed (same acquisition, same draws). Seeds share candidates and are not "
                    "independent: each seed is a separate entry, nothing is pooled across seeds, "
                    "and no multiplicity adjustment is made. It is nullable when a panel has too "
                    "few groups."),
    "conditioning": ("Prediction-time features must come from the unmasked complete candidate "
                     "with answer-free conditioning; quality is recorded as explicit null. A "
                     "cache that conditions on an outcome-derived bin is refused."),
}


class CellUnsupported(ValueError):
    """A cell the plan names but the data or the map cannot support: recorded, never dropped."""


# ---------------------------------------------------------------------------
# small validators and hashing
# ---------------------------------------------------------------------------

def safe_id(value, name):
    """An id usable inside a file name: 1-32 characters from ``[A-Za-z0-9._-]``, starting and
    ending with a letter or digit. No separator, drive, whitespace or ``..`` prefix can occur."""
    require(isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None,
            f"{name} must be 1-32 characters from [A-Za-z0-9._-] that start and end with a letter "
            f"or digit, got {value!r}")
    return value


def _unique_casefold(ids, name):
    require(len({item.lower() for item in ids}) == len(ids),
            f"{name} must be unique ignoring letter case (file systems may fold case): {list(ids)}")


def _keys(document, required, optional=(), *, label):
    require(isinstance(document, dict), f"{label} must be a JSON object")
    missing = [key for key in required if key not in document]
    require(not missing, f"{label} is missing {missing}")
    unknown = sorted(set(document) - set(required) - set(optional))
    require(not unknown, f"{label} has unknown key(s) {unknown}")
    return document


def _bool(value, name):
    require(isinstance(value, bool), f"{name} must be true or false, got {value!r}")
    return value


def _hex64(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    require(isinstance(value, str) and _HEX64.fullmatch(value) is not None,
            f"{name} must be a 64-character lower-case hex SHA-256" + (" or null" if nullable else "")
            + f", got {value!r}")
    return value


def _content_sha256(array, dtype_name):
    """Digest of the dtype name, shape and little-endian bytes of a stored float array."""
    little = np.ascontiguousarray(array, dtype=_LITTLE_ENDIAN[dtype_name])
    return hashlib.sha256(f"{dtype_name}|{little.shape}|".encode("ascii")
                          + little.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# the feature cache: header parsing
# ---------------------------------------------------------------------------

def _parse_identity(document):
    _keys(document, ("kind", "alphabet", "length", "construct_description"),
          label="cache identity")
    kind = document["kind"]
    if kind != IDENTITY_COMPLETE_SEQUENCE:
        raise ValueError(
            f"unsupported identity kind {kind!r}: this build joins on ONE complete fixed-length "
            f"physical sequence (kind {IDENTITY_COMPLETE_SEQUENCE!r}). A structured heavy/light "
            "identity is refused rather than approximated by an edited window; declare a paired "
            "construct as one complete sequence (a delimiter in the alphabet) and describe it in "
            "construct_description")
    alphabet = document["alphabet"]
    require(isinstance(alphabet, str) and alphabet and alphabet == alphabet.upper()
            and len(set(alphabet)) == len(alphabet) and not any(c.isspace() for c in alphabet),
            "cache identity.alphabet must be a string of distinct upper-case non-space characters")
    return {"kind": kind, "alphabet": alphabet,
            "length": fb._integer(document["length"], "cache identity.length", minimum=1),
            "construct_description": fb._text(document["construct_description"],
                                             "cache identity.construct_description")}


def _validate_rows(identity, ids, sequences, groups, splits):
    """Row-level rules shared by writing and loading. Everything is checked on the full sequence."""
    for name, rows in (("ids", ids), ("sequences", sequences), ("groups", groups),
                       ("splits", splits)):
        require(isinstance(rows, list), f"cache rows.{name} must be a list")
    count = len(ids)
    require(count >= 1, "a cache needs at least one row")
    require(len(sequences) == count and len(groups) == count and len(splits) == count,
            f"cache rows are ragged: {count} ids, {len(sequences)} sequences, {len(groups)} "
            f"groups, {len(splits)} splits")
    seen_ids = set()
    for ident in ids:
        require(isinstance(ident, str) and ident and ident == ident.strip(),
                f"cache candidate ids must be non-empty, already-stripped strings, got {ident!r}")
        require(ident not in seen_ids, f"the cache repeats a candidate id: {ident!r}")
        seen_ids.add(ident)
    alphabet, length = identity["alphabet"], identity["length"]
    first_owner = {}
    for ident, sequence in zip(ids, sequences):
        require(isinstance(sequence, str) and sequence == fb.canonical_sequence(sequence),
                f"cache sequence of {ident!r} is not canonical (stripped, upper-case): "
                f"{sequence!r}")
        require(len(sequence) == length,
                f"cache sequence of {ident!r} has length {len(sequence)}, not the declared "
                f"full-construct length {length}; a cache of edited windows is refused")
        bad = sorted({residue for residue in sequence if residue not in alphabet})
        require(not bad, f"cache sequence of {ident!r} has residue(s) {bad} outside the alphabet "
                         f"{alphabet!r}")
        require(sequence not in first_owner,
                f"cache rows {first_owner.get(sequence)!r} and {ident!r} are the same physical "
                f"sequence {sequence!r}")
        first_owner[sequence] = ident
    for ident, group in zip(ids, groups):
        require(isinstance(group, str) and group and group == group.strip(),
                f"cache group of {ident!r} must be a non-empty, stripped string, got {group!r}")
    for ident, split in zip(ids, splits):
        require(split in (fb.SPLIT_TRAIN, fb.SPLIT_EVAL),
                f"cache split of {ident!r} must be {fb.SPLIT_TRAIN!r} or {fb.SPLIT_EVAL!r}, got "
                f"{split!r}")
    sides = {}
    for group, split in zip(groups, splits):
        sides.setdefault(group, set()).add(split)
    crossing = sorted(group for group, kinds in sides.items() if len(kinds) > 1)
    require(not crossing,
            f"biological group(s) {crossing[:3]} appear in both the training and the evaluation "
            "split of the cache; the grouping rationale is violated")


def _parse_space(document):
    _keys(document, ("kind", "names", "schema", "history", "provenance"),
          label="cache feature_space")
    names, history = document["names"], document["history"]
    require(isinstance(names, list) and names and all(isinstance(name, str) for name in names),
            "cache feature_space.names must be a non-empty list of strings")
    require(isinstance(history, list) and history and all(isinstance(step, str) for step in history),
            "cache feature_space.history must be a non-empty list of strings")
    require(isinstance(document["schema"], dict), "cache feature_space.schema must be an object")
    supervision = fb.SupervisionHistory.from_document(document["provenance"],
                                                      label="cache feature_space")
    return fb.FeatureSpace(kind=document["kind"], names=tuple(names), schema=document["schema"],
                           history=tuple(history), supervision=supervision)


def _safe_npz_name(name):
    require(isinstance(name, str) and _NPZ_NAME.fullmatch(name) is not None,
            f"cache storage.file {name!r} must be a plain file name ending in .npz (letters, "
            "digits, '.', '_', '-') in the header's own directory; a path, separator, drive "
            "letter or '..' is refused")
    return name


def _parse_storage(document, rows, dimension):
    _keys(document, ("file", "file_sha256", "array", "dtype", "shape", "content_sha256"),
          label="cache storage")
    require(document["array"] == CACHE_ARRAY, f"cache storage.array must be {CACHE_ARRAY!r}")
    dtype = document["dtype"]
    require(dtype in STORED_DTYPES,
            f"cache storage.dtype must be one of {STORED_DTYPES}, got {dtype!r}")
    require(document["shape"] == [rows, dimension],
            f"cache storage.shape {document['shape']!r} disagrees with the header: {rows} rows x "
            f"{dimension} feature names")
    return {"file": _safe_npz_name(document["file"]),
            "file_sha256": _hex64(document["file_sha256"], "cache storage.file_sha256"),
            "array": CACHE_ARRAY, "dtype": dtype, "shape": [rows, dimension],
            "content_sha256": _hex64(document["content_sha256"], "cache storage.content_sha256")}


def _parse_extractor(document):
    _keys(document, ("kind", "checkpoint_sha256", "source_sha256", "description"),
          label="cache extractor")
    kind = document["kind"]
    require(kind in EXTRACTOR_KINDS,
            f"cache extractor.kind must be one of {EXTRACTOR_KINDS}, got {kind!r}")
    parsed = {"kind": kind,
              "checkpoint_sha256": _hex64(document["checkpoint_sha256"],
                                          "cache extractor.checkpoint_sha256", nullable=True),
              "source_sha256": _hex64(document["source_sha256"], "cache extractor.source_sha256",
                                      nullable=True),
              "description": fb._text(document["description"], "cache extractor.description")}
    if kind == "native_checkpoint":
        require(parsed["checkpoint_sha256"] is not None and parsed["source_sha256"] is not None,
                "a native_checkpoint cache must bind BOTH the checkpoint and the extraction "
                "source by SHA-256")
    return parsed


def _parse_activation(document):
    _keys(document, ("location", "pooling", "token_map"), label="cache activation")
    return {key: fb._text(document[key], f"cache activation.{key}")
            for key in ("location", "pooling", "token_map")}


def _parse_conditioning(document, rows):
    _keys(document, ("candidate_masking", "answer_free") + CONDITIONING_FIELDS,
          label="cache conditioning")
    require(document["candidate_masking"] == "none",
            "cache conditioning.candidate_masking must be 'none': prediction features come from "
            "the UNMASKED complete candidate")
    require(document["answer_free"] is True,
            "cache conditioning.answer_free must be true: the conditioning may not contain "
            "anything derived from outcomes")
    fields = {}
    for name in CONDITIONING_FIELDS:
        entry = _keys(document[name], ("recorded",), ("values",), label=f"cache conditioning.{name}")
        _bool(entry["recorded"], f"cache conditioning.{name}.recorded")
        if not entry["recorded"]:
            require("values" not in entry,
                    f"cache conditioning.{name} is declared not recorded but lists values")
            fields[name] = None
            continue
        values = entry.get("values")
        require(isinstance(values, list) and len(values) == rows,
                f"cache conditioning.{name}.values must list one entry per row ({rows})")
        for value in values:
            require(value is None or (isinstance(value, str) and value and value == value.strip()),
                    f"cache conditioning.{name} values must be stripped non-empty strings or "
                    f"explicit null, got {value!r}")
        fields[name] = list(values)
    quality = fields["quality"]
    require(quality is not None,
            "prediction-time quality conditioning is not recorded; it must be recorded as an "
            "EXPLICIT null for every row (absence cannot be verified answer-free)")
    require(all(value is None for value in quality),
            "cache conditioning.quality holds non-null values: an outcome-derived or any other "
            "quality bin is ineligible for prediction features; record explicit null")
    return {"candidate_masking": "none", "answer_free": True, "fields": fields}


@dataclass(frozen=True, eq=False)
class _Header:
    representation_id: str
    identity: dict
    ids: tuple
    sequences: tuple
    groups: tuple
    splits: tuple
    space: object
    storage: dict
    extractor: dict
    activation: dict
    conditioning: dict
    cache_sha256: str


def _parse_header(document):
    require(isinstance(document, dict), "a feature cache header must be a JSON object")
    if any(key in document for key in ("prior", "prior_log_weight", "prior_log_weights")):
        raise ValueError("a feature cache cannot carry a prior: the reference prior is "
                         "authoritative once, in the candidate table with the prior provenance of "
                         "the matrix manifest")
    _keys(document, ("schema", "representation_id", "identity", "rows", "feature_space", "storage",
                     "extractor", "activation", "conditioning", "cache_sha256"),
          label="feature cache header")
    require(document["schema"] == CACHE_SCHEMA, f"not a {CACHE_SCHEMA} header")
    identity = _parse_identity(document["identity"])
    rows = _keys(document["rows"], ("ids", "sequences", "groups", "splits"), label="cache rows")
    _validate_rows(identity, rows["ids"], rows["sequences"], rows["groups"], rows["splits"])
    count = len(rows["ids"])
    space = _parse_space(document["feature_space"])
    return _Header(
        representation_id=safe_id(document["representation_id"], "cache representation_id"),
        identity=identity, ids=tuple(rows["ids"]), sequences=tuple(rows["sequences"]),
        groups=tuple(rows["groups"]), splits=tuple(rows["splits"]), space=space,
        storage=_parse_storage(document["storage"], count, space.dimension),
        extractor=_parse_extractor(document["extractor"]),
        activation=_parse_activation(document["activation"]),
        conditioning=_parse_conditioning(document["conditioning"], count),
        cache_sha256=_hex64(document["cache_sha256"], "cache cache_sha256"))


def _conditioning_document(fields, order):
    out = {"candidate_masking": "none", "answer_free": True}
    for name in CONDITIONING_FIELDS:
        values = fields[name]
        out[name] = ({"recorded": False} if values is None
                     else {"recorded": True, "values": [values[i] for i in order]})
    return out


def _conditioning_summary(fields):
    """``not_recorded`` (absent) versus ``explicit_null`` versus ``values``, per field."""
    summary = {}
    for name in CONDITIONING_FIELDS:
        values = fields[name]
        if values is None:
            summary[name] = {"state": "not_recorded"}
        elif all(value is None for value in values):
            summary[name] = {"state": "explicit_null", "rows": len(values)}
        else:
            summary[name] = {"state": "values", "rows": len(values),
                             "null_rows": sum(1 for value in values if value is None),
                             "distinct_values": len({value for value in values if value is not None})}
    return summary


def _canonical_cache_document(header, order, canonical_features):
    """The row-order-invariant document ``cache_sha256`` is the digest of."""
    return {"schema": CACHE_SCHEMA, "representation_id": header.representation_id,
            "identity": header.identity,
            "rows": {"ids": [header.ids[i] for i in order],
                     "sequences": [header.sequences[i] for i in order],
                     "groups": [header.groups[i] for i in order],
                     "splits": [header.splits[i] for i in order]},
            "feature_space": header.space.document(),
            "dtype": header.storage["dtype"], "shape": list(header.storage["shape"]),
            "features_content_sha256": _content_sha256(canonical_features,
                                                       header.storage["dtype"]),
            "extractor": header.extractor, "activation": header.activation,
            "conditioning": _conditioning_document(header.conditioning["fields"], order)}


# ---------------------------------------------------------------------------
# the feature cache: writing, loading, joining
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FeatureCache:
    """A validated cache in CANONICAL (sorted candidate id) row order.

    ``features`` is a read-only float64 matrix (promoted exactly from the stored dtype).
    ``cache_sha256`` is row-order invariant; ``raw`` holds the stored-order file digests, which are
    recorded in the input record but are not part of any frozen identity.
    """

    representation_id: str
    identity: dict
    ids: tuple
    sequences: tuple
    groups: tuple
    splits: tuple
    features: object
    space: object
    dtype: str
    extractor: dict
    activation: dict
    conditioning: dict
    conditioning_summary: dict
    cache_sha256: str
    raw: dict

    def summary(self):
        """The cache facts a freeze index records (no arrays, nothing order-dependent)."""
        return {"representation_id": self.representation_id, "cache_sha256": self.cache_sha256,
                "rows": len(self.ids), "feature_dimension": self.space.dimension,
                "dtype": self.dtype, "feature_kind": self.space.kind,
                "feature_space_sha256": self.space.fingerprint(),
                "feature_supervision": self.space.supervision.document(),
                "identity": copy.deepcopy(self.identity), "extractor": copy.deepcopy(self.extractor),
                "activation": copy.deepcopy(self.activation),
                "conditioning": copy.deepcopy(self.conditioning_summary)}


def _refuse_writer_aliases(header_path, npz_path):
    """Refuse, before anything is written, any two of the writer's four files being one file.

    The four are the header, the NPZ and the ``.tmp`` file each is staged through. They are
    compared as PHYSICAL files, pairwise: resolved paths with letter case folded where the file
    system folds it (``A.npz`` and ``a.npz`` are one file on Windows, which a case-sensitive string
    comparison misses), and by identity for files that already exist (hard links, junctions). A
    collision would otherwise let the header replace the NPZ, or the NPZ the header.
    """
    files = {"header": header_path, "NPZ": npz_path,
             "header staging file": fb._temporary_path(header_path),
             "NPZ staging file": fb._temporary_path(npz_path)}
    for (left_label, left), (right_label, right) in itertools.combinations(files.items(), 2):
        require(not fb._same_file(left, right),
                f"the cache {left_label} {left} and the cache {right_label} {right} are the same "
                "physical file (names that differ only by letter case, a resolved alias or a hard "
                "link count); writing would corrupt the cache. Nothing was written")


def write_feature_cache(header_path, npz_path, *, representation_id, ids, sequences, groups, splits,
                        features, space, identity, extractor, activation, conditioning,
                        dtype="float32", overwrite=False):
    """Write a cache (NPZ then header). Everything is validated BEFORE anything is written.

    ``features`` are rounded to ``dtype`` and the STORED values are what the hashes and the loader
    see. The NPZ must sit in the header's directory (the header names it by plain file name). The
    header, the NPZ and their two staging files must be four DIFFERENT physical files (checked
    case-insensitively where the file system is, through resolved paths and for hard links) before
    anything is written. Existing files, staging files included, are replaced only with
    ``overwrite=True``; the inputs are never modified.
    """
    header_path, npz_path = Path(header_path), Path(npz_path)
    require(isinstance(space, fb.FeatureSpace), "write_feature_cache needs a FeatureSpace")
    require(dtype in STORED_DTYPES, f"dtype must be one of {STORED_DTYPES}")
    _safe_npz_name(npz_path.name)
    same_directory = (os.path.normcase(os.path.realpath(header_path.parent))
                      == os.path.normcase(os.path.realpath(npz_path.parent)))
    require(same_directory, "the NPZ must be written in the header's directory (the header names "
                            "it by plain file name)")
    _refuse_writer_aliases(header_path, npz_path)
    for target in (header_path, npz_path, fb._temporary_path(header_path),
                   fb._temporary_path(npz_path)):
        require(overwrite or not target.exists(),
                f"{target} already exists; pass overwrite=True to replace it")
    try:
        array = np.array(features, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cache features must be a numeric matrix ({error})") from error
    require(array.ndim == 2 and array.shape == (len(ids), space.dimension),
            f"cache features must be ({len(ids)}, {space.dimension}), got {array.shape}")
    require(bool(np.isfinite(array).all()), "cache features must be finite (no NaN or infinity)")
    stored = array.astype(np.dtype(dtype))
    require(bool(np.isfinite(stored).all()), f"casting the features to {dtype} overflowed")
    document = {
        "schema": CACHE_SCHEMA, "representation_id": representation_id,
        "identity": identity,
        "rows": {"ids": list(ids), "sequences": list(sequences), "groups": list(groups),
                 "splits": list(splits)},
        "feature_space": space.document(),
        "storage": {"file": npz_path.name, "file_sha256": _PLACEHOLDER_HASH,
                    "array": CACHE_ARRAY, "dtype": dtype, "shape": [int(stored.shape[0]),
                                                                    int(stored.shape[1])],
                    "content_sha256": _content_sha256(stored, dtype)},
        "extractor": extractor, "activation": activation, "conditioning": conditioning,
        "cache_sha256": _PLACEHOLDER_HASH}
    document = json.loads(fb.canonical_json(document))     # plain JSON types, NaN refused
    parsed = _parse_header(document)
    order = sorted(range(len(parsed.ids)), key=parsed.ids.__getitem__)
    document["cache_sha256"] = fb.digest(_canonical_cache_document(parsed, order, stored[order]))
    buffer = io.BytesIO()
    np.savez(buffer, **{CACHE_ARRAY: stored})
    payload = buffer.getvalue()
    document["storage"]["file_sha256"] = hashlib.sha256(payload).hexdigest()
    _parse_header(document)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    staging = fb._temporary_path(npz_path)
    staging.write_bytes(payload)
    staging.replace(npz_path)
    fb.write_json(header_path, document)
    return {"header": str(header_path), "npz": str(npz_path),
            "cache_sha256": document["cache_sha256"],
            "file_sha256": document["storage"]["file_sha256"]}


def resolve_npz_path(header_path, declared_name, explicit=None):
    """The NPZ a header refers to: the caller's explicit path, or the declared plain file name
    next to the header (never a path, and never a symlink that leaves the header's directory)."""
    header_path = Path(header_path)
    if explicit is not None:
        explicit = Path(explicit)
        require(explicit.is_file(), f"the explicit cache NPZ {explicit} does not exist")
        return explicit
    candidate = header_path.parent / _safe_npz_name(declared_name)
    require(candidate.is_file(), f"the cache NPZ {candidate} named by the header does not exist; "
                                 "pass the file explicitly to use another one")
    real = Path(os.path.realpath(candidate))
    home = os.path.realpath(header_path.parent)
    require(os.path.normcase(str(real.parent)) == os.path.normcase(home),
            f"{candidate} resolves outside the header's directory ({real}); refusing an ambiguous "
            "reference - pass the NPZ explicitly")
    return candidate


def _read_npz_features(npz, header):
    try:
        archive = np.load(npz, allow_pickle=False)
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"{npz} cannot be read as a numeric NPZ ({error})") from error
    except ValueError as error:
        raise ValueError(f"{npz} cannot be loaded without pickle ({error}); only plain numeric "
                         "arrays are accepted") from error
    try:
        members = getattr(archive, "files", None)
        require(members is not None, f"{npz} is not an NPZ archive")
        require(list(members) == [CACHE_ARRAY],
                f"{npz} must hold exactly one array named {CACHE_ARRAY!r}, found {list(members)}")
        try:
            array = archive[CACHE_ARRAY]
        except (OSError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"{npz} array cannot be read ({error})") from error
        except ValueError as error:
            raise ValueError(f"{npz} cannot be loaded without pickle ({error}); object arrays "
                             "are refused") from error
    finally:
        close = getattr(archive, "close", None)
        if close is not None:
            close()
    dtype = header.storage["dtype"]
    require(array.dtype == np.dtype(dtype),
            f"{npz} holds dtype {array.dtype}, but the header declares {dtype}")
    require(array.shape == tuple(header.storage["shape"]),
            f"{npz} has shape {array.shape}, but the header declares {tuple(header.storage['shape'])}")
    require(bool(np.isfinite(array).all()), f"{npz} contains NaN or infinity")
    require(_content_sha256(array, dtype) == header.storage["content_sha256"],
            f"{npz} does not match the header's content_sha256 (the arrays were edited)")
    return array


def load_feature_cache(header_path, *, npz_path=None):
    """Load, verify and canonicalise a cache. Every integrity layer is checked or it is refused:
    strict header, file digest (a replaced NPZ is STALE), numeric dtype/shape/finite content,
    stored-order content digest, and the canonical ``cache_sha256`` (a header edited after writing
    is stale too). Rows are returned sorted by candidate id."""
    header_path = Path(header_path)
    document = fb.read_json(header_path)
    header = _parse_header(document)
    npz = resolve_npz_path(header_path, header.storage["file"], npz_path)
    actual = file_sha256(npz)
    require(actual == header.storage["file_sha256"],
            f"{npz} does not match the header's file_sha256: the cache is STALE (the NPZ was "
            "replaced or edited after the header was written)")
    array = _read_npz_features(npz, header)
    order = sorted(range(len(header.ids)), key=header.ids.__getitem__)
    canonical = np.ascontiguousarray(array[order])
    require(fb.digest(_canonical_cache_document(header, order, canonical)) == header.cache_sha256,
            "the header's cache_sha256 does not match its own rows, contracts and features: the "
            "cache is STALE (the header or the arrays changed after it was written)")
    fields = header.conditioning["fields"]
    return FeatureCache(
        representation_id=header.representation_id, identity=copy.deepcopy(header.identity),
        ids=tuple(header.ids[i] for i in order),
        sequences=tuple(header.sequences[i] for i in order),
        groups=tuple(header.groups[i] for i in order),
        splits=tuple(header.splits[i] for i in order), features=fb._frozen(canonical),
        space=header.space, dtype=header.storage["dtype"],
        extractor=copy.deepcopy(header.extractor), activation=copy.deepcopy(header.activation),
        conditioning=_conditioning_document(fields, order),
        conditioning_summary=_conditioning_summary(fields), cache_sha256=header.cache_sha256,
        raw={"header_sha256": fb.digest(document), "file_sha256": actual,
             "storage_content_sha256": header.storage["content_sha256"]})


def check_cache_against_roster(cache, table, manifest):
    """Join by identity and refuse anything but an exact match: the same alphabet and length, the
    same candidate ids, and for EVERY id the same full sequence, group and split."""
    require(isinstance(cache, FeatureCache), "check_cache_against_roster needs a FeatureCache")
    rep = cache.representation_id
    require(cache.identity["alphabet"] == manifest.alphabet
            and cache.identity["length"] == manifest.length,
            f"cache {rep!r} declares alphabet {cache.identity['alphabet']!r} / length "
            f"{cache.identity['length']}, but the manifest declares {manifest.alphabet!r} / "
            f"{manifest.length}")
    cache_ids, roster_ids = set(cache.ids), set(table.ids)
    missing, extra = sorted(roster_ids - cache_ids), sorted(cache_ids - roster_ids)
    require(not missing, f"cache {rep!r} lacks {len(missing)} of the {len(roster_ids)} roster "
                         f"candidate(s), e.g. {missing[:3]}; there is no convenient intersection")
    require(not extra, f"cache {rep!r} holds {len(extra)} candidate(s) outside the roster, "
                       f"e.g. {extra[:3]}")
    for label, mine, theirs in (("physical sequence", cache.sequences, table.sequences),
                                ("group", cache.groups, table.groups),
                                ("split", cache.splits, table.splits)):
        bad = [cache.ids[i] for i in range(len(cache.ids)) if mine[i] != theirs[i]]
        require(not bad, f"cache {rep!r} binds {len(bad)} candidate(s) to a different {label} "
                         f"than the candidate table, e.g. {bad[:3]}")
    return True


# ---------------------------------------------------------------------------
# the matrix manifest, the frozen plan and the cell keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatrixManifest:
    target: str
    assay: str
    utility: object
    grouping_rationale: str
    alphabet: str
    length: int
    prior: object
    notes: str = ""

    def document(self):
        return {"schema": MANIFEST_SCHEMA, "target": self.target, "assay": self.assay,
                "utility": self.utility.document(), "grouping_rationale": self.grouping_rationale,
                "candidates": {"identity_kind": IDENTITY_COMPLETE_SEQUENCE,
                               "alphabet": self.alphabet, "length": self.length},
                "prior": self.prior.document(), "notes": self.notes}


def parse_matrix_manifest(document):
    document = copy.deepcopy(document)
    _keys(document, ("schema", "target", "assay", "utility", "grouping_rationale", "candidates",
                     "prior"), ("notes",), label="matrix manifest")
    require(document["schema"] == MANIFEST_SCHEMA, f"not a {MANIFEST_SCHEMA} manifest")
    utility = _keys(document["utility"],
                    ("units", "scale", "higher_is_better", "useful_threshold"), ("name",),
                    label="manifest.utility")
    candidates = _keys(document["candidates"], ("identity_kind", "alphabet", "length"),
                       label="manifest.candidates")
    if candidates["identity_kind"] != IDENTITY_COMPLETE_SEQUENCE:
        raise ValueError(f"unsupported identity kind {candidates['identity_kind']!r}: the matrix "
                         f"joins on one complete fixed-length sequence "
                         f"({IDENTITY_COMPLETE_SEQUENCE!r}); a structured heavy/light identity is "
                         "refused")
    alphabet = fb._text(candidates["alphabet"], "manifest.candidates.alphabet").upper()
    require(len(set(alphabet)) == len(alphabet) and not any(c.isspace() for c in alphabet),
            "manifest.candidates.alphabet must be distinct non-space characters")
    prior = _keys(document["prior"], ("kind", "semantics", "provenance"), ("description",),
                  label="manifest.prior")
    return MatrixManifest(
        target=fb._text(document["target"], "manifest.target"),
        assay=fb._text(document["assay"], "manifest.assay"),
        utility=fb.UtilitySpec(units=utility["units"], scale=utility["scale"],
                               higher_is_better=utility["higher_is_better"],
                               useful_threshold=utility["useful_threshold"],
                               name=utility.get("name", "utility")),
        grouping_rationale=fb._text(document["grouping_rationale"], "manifest.grouping_rationale"),
        alphabet=alphabet, length=fb._integer(candidates["length"], "manifest.candidates.length",
                                              minimum=1),
        prior=fb.PriorSpec(
            kind=prior["kind"], semantics=prior["semantics"],
            description=prior.get("description", ""),
            supervision=fb.SupervisionHistory.from_document(prior["provenance"],
                                                            label="manifest.prior")),
        notes=str(document.get("notes", "")))


def load_matrix_manifest(path):
    return parse_matrix_manifest(fb.read_json(path))


def _parse_bandwidth(document):
    require(isinstance(document, dict) and "rule" in document,
            "an rbf map needs a bandwidth object with a 'rule'")
    rule = document["rule"]
    if rule == BANDWIDTH_MEDIAN:
        _keys(document, ("rule", "multiplier"), label="map bandwidth")
        return {"rule": rule, "multiplier": fb._real(document["multiplier"],
                                                     "bandwidth multiplier", positive=True)}
    if rule == BANDWIDTH_FIXED:
        _keys(document, ("rule", "value"), label="map bandwidth")
        return {"rule": rule, "value": fb._real(document["value"], "bandwidth value",
                                                positive=True)}
    raise ValueError(f"bandwidth rule must be {BANDWIDTH_MEDIAN!r} or {BANDWIDTH_FIXED!r} (a "
                     f"covariate-only formula or a fixed positive input), got {rule!r}")


@dataclass(frozen=True, eq=False)
class MapSpec:
    """One fixed readout map. ``bandwidth`` is covariate-only (a rule on the acquired centers) or a
    fixed positive input; nothing is tuned against labels."""

    id: str
    kind: str
    center: bool = True
    scale: str = "std"
    trace_normalize: bool = True
    bandwidth: object = None

    def __post_init__(self):
        safe_id(self.id, "map id")
        require(self.kind in MAP_KINDS, f"map kind must be one of {MAP_KINDS}, got {self.kind!r}")
        _bool(self.center, "map center")
        _bool(self.trace_normalize, "map trace_normalize")
        require(self.scale in SCALE_MODES,
                f"map scale must be one of {SCALE_MODES}, got {self.scale!r}")
        if self.kind == MAP_RBF:
            object.__setattr__(self, "bandwidth", _parse_bandwidth(self.bandwidth))
        else:
            require(self.bandwidth is None, "a linear map has no bandwidth")

    @classmethod
    def from_document(cls, document):
        _keys(document, ("id", "kind", "center", "scale", "trace_normalize"), ("bandwidth",),
              label="plan map")
        return cls(**document)

    def document(self):
        out = {"id": self.id, "kind": self.kind, "center": self.center, "scale": self.scale,
               "trace_normalize": self.trace_normalize}
        if self.kind == MAP_RBF:
            out["bandwidth"] = copy.deepcopy(self.bandwidth)
        return out


def _parse_baseline(document):
    _keys(document, ("tau",), BASELINE_OPTIONS[1:], label="plan.baseline")
    options = dict(document)
    if "dpo_pressures" in options:
        require(isinstance(options["dpo_pressures"], list),
                "plan.baseline.dpo_pressures must be a list")
        options["dpo_pressures"] = tuple(options["dpo_pressures"])
    config = fb.BaselineConfig(**options)
    require(config.sir_proposal != fb.SIR_DECLARED,
            "a 'declared' SIR proposal needs explicit proposal weights, which a matrix plan "
            "cannot carry; use 'prior' or 'uniform_over_target_support'")
    resolved = {name: getattr(config, name) for name in BASELINE_OPTIONS}
    resolved["dpo_pressures"] = list(config.dpo_pressures)
    return resolved


def _budget_order(budget):
    return (budget is None, 0 if budget is None else budget)


@dataclass(frozen=True, eq=False)
class MatrixPlan:
    representations: tuple
    maps: tuple
    label_budgets: tuple
    seeds: tuple
    baseline: dict
    reference: dict
    bootstrap: dict
    notes: str = ""

    def document(self):
        return {"schema": PLAN_SCHEMA, "representations": list(self.representations),
                "maps": [spec.document() for spec in self.maps],
                "label_budgets": list(self.label_budgets), "seeds": list(self.seeds),
                "baseline": copy.deepcopy(self.baseline), "reference": dict(self.reference),
                "bootstrap": dict(self.bootstrap), "notes": self.notes}

    def fingerprint(self):
        return fb.digest(self.document())

    def spec(self, map_id):
        for spec in self.maps:
            if spec.id == map_id:
                return spec
        raise KeyError(map_id)

    def config_for(self, key):
        options = dict(self.baseline)
        options["dpo_pressures"] = tuple(options["dpo_pressures"])
        return fb.BaselineConfig(**options, seed=key.seed, label_budget=key.label_budget)


def parse_plan(document):
    document = copy.deepcopy(document)
    _keys(document, ("schema", "representations", "maps", "label_budgets", "seeds", "baseline",
                     "reference", "bootstrap"), ("notes",), label="matrix plan")
    require(document["schema"] == PLAN_SCHEMA, f"not a {PLAN_SCHEMA} plan")
    reps = document["representations"]
    require(isinstance(reps, list) and reps, "plan.representations must be a non-empty list")
    reps = [safe_id(item, "representation id") for item in reps]
    require(len(set(reps)) == len(reps), "plan.representations repeats an id")
    _unique_casefold(reps, "representation ids")
    maps = document["maps"]
    require(isinstance(maps, list) and maps, "plan.maps must be a non-empty list")
    specs = sorted((MapSpec.from_document(entry) for entry in maps), key=lambda spec: spec.id)
    ids = [spec.id for spec in specs]
    require(len(set(ids)) == len(ids), "plan.maps repeats an id")
    _unique_casefold(ids, "map ids")
    budgets = document["label_budgets"]
    require(isinstance(budgets, list) and budgets, "plan.label_budgets must be a non-empty list")
    budgets = [None if item is None else fb._integer(item, "a label budget", minimum=1)
               for item in budgets]
    require(len(set(budgets)) == len(budgets), "plan.label_budgets repeats a budget")
    seeds = document["seeds"]
    require(isinstance(seeds, list) and seeds, "plan.seeds must be a non-empty list")
    seeds = [fb._integer(item, "a seed", minimum=0) for item in seeds]
    require(len(set(seeds)) == len(seeds), "plan.seeds repeats a seed")
    reference = _keys(document["reference"], ("representation", "map"), label="plan.reference")
    require(reference["representation"] in reps,
            f"plan.reference.representation {reference['representation']!r} is not a planned "
            "representation")
    require(reference["map"] in ids, f"plan.reference.map {reference['map']!r} is not a planned map")
    bootstrap = _keys(document["bootstrap"], ("replicates", "seed", "confidence", "min_groups"),
                      label="plan.bootstrap")
    confidence = fb._real(bootstrap["confidence"], "bootstrap confidence")
    require(0.0 < confidence < 1.0, "bootstrap confidence must lie strictly between 0 and 1")
    return MatrixPlan(
        representations=tuple(sorted(reps)), maps=tuple(specs),
        label_budgets=tuple(sorted(budgets, key=_budget_order)), seeds=tuple(sorted(seeds)),
        baseline=_parse_baseline(document["baseline"]),
        reference={"representation": reference["representation"], "map": reference["map"]},
        bootstrap={"replicates": fb._integer(bootstrap["replicates"], "bootstrap replicates",
                                             minimum=10),
                   "seed": fb._integer(bootstrap["seed"], "bootstrap seed", minimum=0),
                   "confidence": confidence,
                   "min_groups": fb._integer(bootstrap["min_groups"], "bootstrap min_groups",
                                             minimum=2)},
        notes=str(document.get("notes", "")))


def load_plan(path):
    return parse_plan(fb.read_json(path))


@dataclass(frozen=True)
class CellKey:
    representation: str
    map_id: str
    label_budget: object
    seed: int

    @property
    def budget_token(self):
        return "all" if self.label_budget is None else str(self.label_budget)

    @property
    def cell_id(self):
        return f"{self.representation}|{self.map_id}|b{self.budget_token}|s{self.seed}"

    def document(self):
        return {"representation": self.representation, "map": self.map_id,
                "label_budget": self.label_budget, "seed": self.seed}

    def filename(self):
        """Built only from validated ids plus a hash of the key: never an arbitrary fragment."""
        return (f"cell_{self.representation}_{self.map_id}_b{self.budget_token}_s{self.seed}_"
                f"{fb.digest(self.document())[:12]}.json")


def enumerate_cells(plan):
    return tuple(CellKey(rep, spec.id, budget, seed)
                 for rep in plan.representations for spec in plan.maps
                 for budget in plan.label_budgets for seed in plan.seeds)


# ---------------------------------------------------------------------------
# the roster: candidate table + training outcomes, representation independent
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Roster:
    """The labelled offered rows, the unlabelled training rows and the held-out support, by
    position in the (sorted-id) candidate table. Nothing here depends on a representation."""

    manifest: object
    table: object
    labelled_ids: tuple
    labelled_rows: object
    utilities: object
    support_ids: tuple
    support_rows: object
    unlabelled_ids: tuple

    def common_document(self):
        """What EVERY cell shares: support, prior, utility and the offered roster (digests, plus
        the support ids and groups so the evaluation can be checked for completeness)."""
        table, rows = self.table, self.support_rows
        return {
            "manifest": self.manifest.document(),
            "manifest_sha256": fb.digest(self.manifest.document()),
            "prior_spec_sha256": self.manifest.prior.fingerprint(),
            "utility_sha256": fb.digest(self.manifest.utility.document()),
            "roster": table.roster(),
            "offered": {"count": len(self.labelled_ids),
                        "ids_sha256": fb.digest(list(self.labelled_ids)),
                        "utilities_sha256": fb._array_sha256(self.utilities)},
            "unlabelled_train": {"count": len(self.unlabelled_ids),
                                 "ids_sha256": fb.digest(list(self.unlabelled_ids))},
            "support": {"count": len(self.support_ids), "ids": list(self.support_ids),
                        "ids_sha256": fb.digest(list(self.support_ids)),
                        "groups": [table.groups[i] for i in rows.tolist()],
                        "sequences_sha256": fb.digest([table.sequences[i] for i in rows.tolist()]),
                        "prior_log_weight_sha256": fb._array_sha256(table.prior_log_weight[rows])}}


def make_roster(manifest, table, train_outcomes):
    """Join a matrix manifest, a candidate table and TRAINING outcomes (no evaluation outcome).

    The same label contract as the baseline CSV pipeline: outcomes name known training-split
    candidates only, the declared prior is validated over the WHOLE pool (unlabelled rows too),
    and a held-out support must exist.
    """
    require(isinstance(manifest, MatrixManifest) and isinstance(table, fb.CandidateTable),
            "make_roster needs a MatrixManifest and a CandidateTable")
    index = {ident: position for position, ident in enumerate(table.ids)}
    unknown = sorted(set(train_outcomes) - set(index))
    require(not unknown, f"training outcomes name unknown candidate(s) {unknown[:3]}")
    on_eval = sorted(ident for ident in train_outcomes
                     if table.splits[index[ident]] == fb.SPLIT_EVAL)
    require(not on_eval, f"the training outcomes include {len(on_eval)} evaluation-split "
                         f"candidate(s), e.g. {on_eval[:3]}: train/evaluation overlap in ids")
    require(train_outcomes, "the training outcomes file has no rows; there is nothing to learn from")
    fb.validate_prior_log_weights(manifest.prior.kind, table.prior_log_weight)
    labelled = tuple(sorted(train_outcomes))
    support = tuple(ident for ident in table.ids if table.splits[index[ident]] == fb.SPLIT_EVAL)
    require(support, "the candidates file has no evaluation-split candidates")
    unlabelled = tuple(ident for ident in table.ids
                       if table.splits[index[ident]] == fb.SPLIT_TRAIN
                       and ident not in train_outcomes)
    return Roster(
        manifest=manifest, table=table, labelled_ids=labelled,
        labelled_rows=np.array([index[ident] for ident in labelled], dtype=np.int64),
        utilities=fb._frozen([train_outcomes[ident] for ident in labelled]),
        support_ids=support,
        support_rows=np.array([index[ident] for ident in support], dtype=np.int64),
        unlabelled_ids=unlabelled)


# ---------------------------------------------------------------------------
# the readout maps (fit on the ACQUIRED rows only)
# ---------------------------------------------------------------------------

def _squared_distances(rows, centers):
    """``||row - center||^2`` by direct differences in bounded blocks (no cancellation)."""
    count, width = rows.shape[0], rows.shape[1]
    out = np.empty((count, centers.shape[0]))
    step = max(1, DISTANCE_BLOCK_ELEMENTS // max(1, centers.shape[0] * width))
    for start in range(0, count, step):
        difference = rows[start:start + step, None, :] - centers[None, :, :]
        out[start:start + step] = (difference * difference).sum(axis=2)
    return out


def _anchored_centre(matrix):
    """Column-centre ``matrix`` relative to its first row, so a constant column centres to EXACTLY 0.

    ``matrix - matrix.mean(axis=0)`` does not: the mean of three copies of 0.1 is
    0.10000000000000002, which leaves a ~1e-17 residue in a column that has no variation, and trace
    normalisation then amplifies that residue by ~1e16 into a spurious direction. Subtracting the
    first row first is exact for bit-identical entries (an exact zero deviation, whose mean is an
    exact zero), and is the usual shifted-data form of the mean for everything else.
    """
    deviation = matrix - matrix[0]
    return deviation - deviation.mean(axis=0)


def _apply_map(spec, z, centers, sigma):
    if spec.kind == MAP_LINEAR:
        return z
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-_squared_distances(z, centers) / (2.0 * sigma * sigma))


def _bandwidth_record(rule, centers):
    count = centers.shape[0]
    median, pairs = None, 0
    if rule["rule"] == BANDWIDTH_MEDIAN:
        if count >= 2:
            upper = np.sqrt(_squared_distances(centers, centers)[np.triu_indices(count, 1)])
            positive = upper[upper > 0.0]
            pairs = int(positive.shape[0])
            median = float(np.median(positive)) if pairs else None
        if median is None:
            raise CellUnsupported(
                "the median-heuristic bandwidth is undefined: the acquired rows give "
                f"{count} distinct RBF center(s) and no positive pairwise distance (all acquired "
                "feature rows are identical); supply a fixed bandwidth to map such data")
        sigma = rule["multiplier"] * median
    else:
        sigma = rule["value"]
    require(math.isfinite(sigma) and sigma > 0.0,
            f"the RBF bandwidth came out as {sigma}; it must be finite and positive")
    return {**rule, "median_pairwise_distance": median, "positive_pairs": pairs,
            "sigma": float(sigma), "units": "standardised input units",
            "label_free": "computed from acquired feature rows only; no outcome is read"}


@dataclass(frozen=True, eq=False)
class FittedReadout:
    """A frozen map: its state was computed from the acquired feature rows only."""

    spec: object
    input_dimension: int
    output_dimension: int
    acquired_rows: int
    mean: object
    scale: object
    constant_columns: tuple
    near_constant_columns: tuple
    centers: object
    center_positions: tuple
    duplicate_acquired_rows: int
    bandwidth: object
    trace_factor: float
    degenerate_design: bool
    centered_variance: float
    realized_variance: float
    capacity: dict

    def transform(self, features):
        """Map ANY rows (acquired, unacquired offered, unlabelled, held-out) with the frozen state."""
        rows = np.array(features, dtype=np.float64, copy=True)
        require(rows.ndim == 2 and rows.shape[1] == self.input_dimension,
                f"readout input must be (rows, {self.input_dimension}), got {rows.shape}")
        require(bool(np.isfinite(rows).all()), "readout input must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            z = (rows - self.mean) / self.scale
        sigma = None if self.bandwidth is None else self.bandwidth["sigma"]
        out = _apply_map(self.spec, z, self.centers, sigma) * self.trace_factor
        require(bool(np.isfinite(out).all()), "the readout map produced non-finite values")
        return out

    def state_document(self):
        """The transform state. Computed from acquired features only, so it is unchanged when any
        unacquired or held-out feature changes; it carries no digest of the whole cache."""
        centers = None
        if self.centers is not None:
            centers = {"count": int(self.centers.shape[0]),
                       "acquired_positions": list(self.center_positions),
                       "duplicate_acquired_rows": self.duplicate_acquired_rows,
                       "centers_sha256": fb._array_sha256(self.centers),
                       "space": "standardised acquired rows"}
        return {
            "map": self.spec.document(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "constant_columns": list(self.constant_columns),
            "near_constant_columns": list(self.near_constant_columns),
            "constant_handling": (
                "constancy is EXACT: a column is constant only if its acquired values are "
                "bit-identical. A constant column is kept with scale 1; with center=True its mean "
                "is that exact value, so its acquired coordinate is exactly zero and ridge/DPO "
                "give it weight exactly zero (with center=False it keeps its raw value, a "
                "constant offset the ridge intercept absorbs). A column that varies at all is NOT "
                "constant and is never discarded for its units: one whose range is within "
                f"{NEAR_CONSTANT_TOLERANCE} (relative to max(1, |value|)) is only listed under "
                "near_constant_columns. Centering is taken relative to the first acquired row, so "
                "constant columns and all-identical designs centre to exactly zero"),
            "centers": centers, "bandwidth": copy.deepcopy(self.bandwidth),
            "trace_normalization": {
                "enabled": self.spec.trace_normalize, "factor": self.trace_factor,
                "centered_acquired_variance_before": self.centered_variance,
                "realized_centered_acquired_variance_after": self.realized_variance,
                "degenerate_acquired_design": self.degenerate_design,
                "definition": ("g = 1 / sqrt(mean over acquired rows of the squared norm of the "
                               "centred mapped row); 1 when disabled or the design is degenerate"),
                "realized_variance_note": (
                    "the mean squared norm of the centred acquired rows AS ACTUALLY PRODUCED by the "
                    "transform, after the multiplication by g. It is 1 up to rounding for a "
                    "well-conditioned design and is not promised to be exactly 1 otherwise: when g "
                    "is enormous (a variation of a few ulps on a large offset) the product rounds "
                    "and the realized value differs. The recorded rank and df describe these "
                    "rounded rows, the ones that are frozen and fitted"),
                "note": CLAIMS["regularisation"]}}

    def preprocessing_cost(self, transformed_rows):
        """Deterministic operation counts (never wall-clock time), for the fit and the transform."""
        d, out_dim, m = self.input_dimension, self.output_dimension, self.acquired_rows
        centers = 0 if self.centers is None else int(self.centers.shape[0])
        median = (self.spec.kind == MAP_RBF and self.spec.bandwidth["rule"] == BANDWIDTH_MEDIAN)
        per_row = d if self.spec.kind == MAP_LINEAR else d + d * centers
        return {"unit": "operation counts, not wall-clock time (deterministic by design)",
                "fit": {"acquired_rows": m, "column_statistic_entries": m * d,
                        "pairwise_center_distances": centers * (centers - 1) // 2 if median else 0,
                        "design_svd_shape": [m, out_dim]},
                "transform": {"rows": int(transformed_rows),
                              "kernel_evaluations": int(transformed_rows * centers),
                              "multiply_add_estimate": int(transformed_rows * per_row)}}


def fit_readout(spec, acquired_features, *, ridge_alpha):
    """Fit a readout map from the ACQUIRED feature rows only (no label, no other row).

    Raises ``CellUnsupported`` when the map is undefined for the data (the median heuristic over
    all-identical rows). Records the output dimension, the design rank and the effective ridge
    degrees of freedom of the centred, trace-normalised acquired design.
    """
    require(isinstance(spec, MapSpec), "fit_readout needs a MapSpec")
    alpha = fb._real(ridge_alpha, "ridge_alpha", positive=True)
    acquired = np.array(acquired_features, dtype=np.float64, copy=True)
    require(acquired.ndim == 2 and acquired.shape[0] >= 1 and acquired.shape[1] >= 1,
            f"acquired features must be a non-empty (rows, dimension) matrix, got {acquired.shape}")
    require(bool(np.isfinite(acquired).all()), "acquired features must be finite")
    rows, width = acquired.shape
    if spec.kind == MAP_LINEAR:
        # the linear output dimension is the input dimension: refuse a dense system the baseline
        # would refuse, BEFORE any row is mapped (an RBF's dimension is known after its centers)
        fb._require_dense_system(width, "ridge Gram system of the linear readout")
    anchor = acquired[0]
    constant = np.all(acquired == anchor, axis=0)          # EXACT: bit-identical acquired values
    with np.errstate(over="ignore", invalid="ignore"):
        deviation = acquired - anchor                      # exactly 0 in a constant column
        shift = deviation.mean(axis=0)
        spread = np.sqrt(np.mean((deviation - shift) ** 2, axis=0))
        span = acquired.max(axis=0) - acquired.min(axis=0)
        near = (~constant) & (span <= NEAR_CONSTANT_TOLERANCE
                              * np.maximum(1.0, np.abs(acquired).max(axis=0)))
    require(bool(np.isfinite(shift).all() and np.isfinite(spread).all()),
            "a column mean or standard deviation overflowed float64; rescale the features")
    mean = anchor + shift if spec.center else np.zeros(width)
    scale = np.where(constant, 1.0, spread) if spec.scale == "std" else np.ones(width)
    require(bool((scale > 0.0).all()),
            "a column that is not constant has a standard deviation that underflows to zero in "
            "float64; rescale the features or use scale 'none'")
    with np.errstate(over="ignore", invalid="ignore"):
        z = (acquired - mean) / scale
    require(bool(np.isfinite(z).all()),
            "the standardised acquired rows overflowed float64; rescale the features")
    centers, positions, duplicates, bandwidth, sigma = None, (), 0, None, None
    if spec.kind == MAP_RBF:
        _, first = np.unique(z, axis=0, return_index=True)
        positions = tuple(int(i) for i in np.sort(first))
        duplicates = rows - len(positions)
        # one RBF column per distinct center: the baseline's Gram system is that wide
        fb._require_dense_system(len(positions), "ridge Gram system of the RBF readout (one "
                                                 "column per distinct acquired center)")
        centers = z[list(positions)]
        bandwidth = _bandwidth_record(spec.bandwidth, centers)
        sigma = bandwidth["sigma"]
    design = _apply_map(spec, z, centers, sigma)
    centred = _anchored_centre(design)
    variance = float((centred ** 2).sum() / rows)
    degenerate = variance == 0.0
    factor = 1.0 if (not spec.trace_normalize or degenerate) else 1.0 / math.sqrt(variance)
    readout = FittedReadout(
        spec=spec, input_dimension=int(width), output_dimension=int(design.shape[1]),
        acquired_rows=int(rows), mean=fb._frozen(mean), scale=fb._frozen(scale),
        constant_columns=tuple(int(i) for i in np.flatnonzero(constant)),
        near_constant_columns=tuple(int(i) for i in np.flatnonzero(near)),
        centers=None if centers is None else fb._frozen(centers), center_positions=positions,
        duplicate_acquired_rows=int(duplicates), bandwidth=bandwidth, trace_factor=float(factor),
        degenerate_design=bool(degenerate), centered_variance=variance, realized_variance=0.0,
        capacity={})
    # Rank and df are a statement about the rows that are FROZEN and FITTED: the acquired rows as
    # ``transform`` actually produces them, i.e. AFTER the multiplication by the trace factor.
    # ``centre(design) * factor`` is a different object whenever the product rounds: an acquired
    # variation of 2^-50 on an offset of 1 gives g ~ 2.4e15, the stored rows [A, A + 2, A] have df
    # 8/11, and the pre-multiplication design would have claimed 3/4 (and then failed its own
    # verification against the stored rows).
    final = _anchored_centre(readout.transform(acquired))
    realized = float((final ** 2).sum() / rows)
    singular = np.linalg.svd(final, compute_uv=False)
    tolerance = max(final.shape) * np.finfo(np.float64).eps * (
        float(singular.max()) if singular.size else 0.0)
    capacity = {
        "input_dimension": int(width), "output_dimension": int(design.shape[1]),
        "acquired_rows": int(rows), "design_rank": int((singular > tolerance).sum()),
        "effective_ridge_df": float(np.sum(singular ** 2 / (singular ** 2 + alpha))),
        "ridge_alpha": alpha,
        "exactly_constant_columns": int(constant.sum()),
        "near_constant_columns": int(near.sum()),
        "df_definition": ("sum s^2 / (s^2 + alpha) over the singular values s of the centred ACTUAL "
                          "transformed acquired rows (after the trace-factor multiplication and its "
                          "rounding): the rows the ridge fit sees and the artifact stores"),
        "note": CLAIMS["regularisation"]}
    return dataclasses.replace(readout, capacity=capacity, realized_variance=realized)


# ---------------------------------------------------------------------------
# building and running one cell
# ---------------------------------------------------------------------------

def _derived_space(cache, spec, readout, *, acquired, acquired_ids, config, roster):
    """The FeatureSpace a cell's problem is built on: the mapped basis, with the whole transform
    record (state, acquired identities, raw digests, capacity, cost) in its hashed schema."""
    state = readout.state_document()
    if spec.kind == MAP_LINEAR:
        names, center_ids = tuple(f"{spec.id}:{name}" for name in cache.space.names), None
    else:
        center_ids = [acquired_ids[p] for p in readout.center_positions]
        names = tuple(f"{spec.id}@{ident}" for ident in center_ids)
    offered, held_out = len(roster.labelled_ids), len(roster.support_ids)
    schema = {
        "construction": "acquired-only frozen-feature readout map",
        "representation_id": cache.representation_id, "map": spec.document(),
        "transform": state, "transform_sha256": fb.digest(state), "center_ids": center_ids,
        "acquired": {
            "budget": config.label_budget, "seed": config.seed, "offered_labelled_count": offered,
            "acquired_count": int(acquired.shape[0]), "indices": acquired.tolist(),
            "ids": list(acquired_ids), "ids_sha256": fb.digest(list(acquired_ids)),
            "rule": ("fb.acquire_labels(offered, budget, seed), derived BEFORE any fitting; the "
                     "transform saw the feature rows of exactly these candidates")},
        "applied_to": {"offered_rows": offered, "held_out_rows": held_out,
                       "note": ("every original offered and held-out row is mapped; none is "
                                "dropped, reselected or renormalised onto a subset")},
        "provenance": {
            "canonical_cache_sha256": cache.cache_sha256,
            "cache_feature_space_sha256": cache.space.fingerprint(),
            "candidate_roster": roster.table.roster(),
            "offered_ids_sha256": fb.digest(list(roster.labelled_ids)),
            "extractor": copy.deepcopy(cache.extractor),
            "activation": copy.deepcopy(cache.activation),
            "conditioning": copy.deepcopy(cache.conditioning_summary),
            "unlabelled_covariates_used": False, "unacquired_labelled_covariates_used": False,
            "declared_not_verified": True},
        "capacity": copy.deepcopy(readout.capacity),
        "preprocessing_cost": readout.preprocessing_cost(offered + held_out),
        "dpo_basis": CLAIMS["dpo_basis"]}
    history = (
        f"frozen features of representation {cache.representation_id!r} (canonical cache digest "
        f"{cache.cache_sha256[:16]})",
        f"{spec.kind} readout map {spec.id!r} fitted on the {acquired.shape[0]} acquired feature "
        f"rows only (center={spec.center}, scale={spec.scale}), applied to all {offered} offered "
        f"and {held_out} held-out rows",
        f"trace normalisation enabled={spec.trace_normalize}; it does not equalise "
        "regularisation geometry")
    space = fb.FeatureSpace(kind="frozen_readout_map", names=names, schema=schema,
                            history=history, supervision=cache.space.supervision)
    return space


def _build_problem(cache, spec, roster, config):
    """Derive the acquired indices FIRST, fit the map on those feature rows, map everything."""
    offered = len(roster.labelled_ids)
    if config.label_budget is not None and config.label_budget > offered:
        raise CellUnsupported(
            f"label_budget={config.label_budget} exceeds the {offered} labels offered; the "
            "baseline will not invent labels")
    acquired = fb.acquire_labels(offered, config.label_budget, config.seed)
    raw_offered = cache.features[roster.labelled_rows]
    readout = fit_readout(spec, raw_offered[acquired], ridge_alpha=config.ridge_alpha)
    mapped_offered = readout.transform(raw_offered)
    mapped_support = readout.transform(cache.features[roster.support_rows])
    acquired_ids = [roster.labelled_ids[i] for i in acquired.tolist()]
    space = _derived_space(cache, spec, readout, acquired=acquired, acquired_ids=acquired_ids,
                           config=config, roster=roster)
    table, manifest = roster.table, roster.manifest
    lab, sup = roster.labelled_rows.tolist(), roster.support_rows.tolist()
    problem = fb.FiniteSupportProblem(
        target=manifest.target, assay=manifest.assay, utility=manifest.utility,
        grouping_rationale=manifest.grouping_rationale, space=space, prior=manifest.prior,
        train_ids=roster.labelled_ids, train_features=mapped_offered,
        train_utilities=roster.utilities, train_log_prior=table.prior_log_weight[roster.labelled_rows],
        train_groups=tuple(table.groups[i] for i in lab),
        support_ids=roster.support_ids, support_features=mapped_support,
        support_log_prior=table.prior_log_weight[roster.support_rows],
        support_groups=tuple(table.groups[i] for i in sup),
        train_sequences=tuple(table.sequences[i] for i in lab),
        support_sequences=tuple(table.sequences[i] for i in sup),
        unlabelled_train_candidates=len(roster.unlabelled_ids),
        unlabelled_train_ids=roster.unlabelled_ids, candidate_roster=table.roster())
    return problem, readout, acquired


def run_cell(key, cache, roster, plan, common_sha256):
    """One cell, as a JSON envelope. Never raises for a cell-level refusal: an unsupported or failed
    cell keeps its explicit reason and stays in the index."""
    spec, config = plan.spec(key.map_id), plan.config_for(key)
    envelope = {"schema": CELL_SCHEMA, "cell_id": key.cell_id, "key": key.document(),
                "common_sha256": common_sha256, "map": spec.document(), "status": None,
                "reason": None, "capacity": None, "artifact": None}
    try:
        problem, readout, acquired = _build_problem(cache, spec, roster, config)
        fit = fb.fit_stage(problem, config)
        require(np.array_equal(fit.acquired, acquired),
                "internal error: fit_stage acquired different labels than the ones the readout "
                "map was fitted on")
        allocation = fb.allocate(problem.support_ids, problem.support_log_prior, fit.predictions,
                                 config, dpo_energy=fit.dpo_energy)
        artifact = fb.freeze_artifact(problem, config, fit, allocation)
    except CellUnsupported as error:
        envelope.update(status=STATUS_UNSUPPORTED, reason=str(error))
    except (ValueError, ArithmeticError) as error:
        envelope.update(status=STATUS_FAILED, reason=f"{type(error).__name__}: {error}")
    else:
        envelope.update(status=STATUS_OK, artifact=artifact, capacity={
            **copy.deepcopy(readout.capacity),
            "preprocessing_cost": readout.preprocessing_cost(len(roster.labelled_ids)
                                                             + len(roster.support_ids))})
    return fb._builtin(envelope)


# ---------------------------------------------------------------------------
# outputs: paths, aliasing, the freeze index and its verification
# ---------------------------------------------------------------------------

def _output_paths(out, keys):
    paths = {f"cell file {key.cell_id}": out / CELLS_DIR / key.filename() for key in keys}
    paths["freeze index"] = out / INDEX_NAME
    paths["input record"] = out / INPUTS_NAME
    paths["evaluation report"] = out / RESULTS_NAME
    return paths


def _with_staging(paths):
    return {**paths, **{f"{label} temporary file": fb._temporary_path(path)
                        for label, path in paths.items()}}


def _check_output_state(out, final, everything, overwrite):
    cells_dir = out / CELLS_DIR
    require(not cells_dir.exists() or cells_dir.is_dir(), f"{cells_dir} exists and is not a directory")
    if cells_dir.is_dir():
        expected = {path.name for path in everything.values() if path.parent == cells_dir}
        stray = sorted(item.name for item in cells_dir.iterdir() if item.name not in expected)
        require(not stray, f"{cells_dir} holds {len(stray)} file(s) this plan does not produce, "
                           f"e.g. {stray[:3]}; stale cells are never cleaned automatically - use "
                           "a fresh output directory")
    for label, target in final.items():
        require(overwrite or not target.exists(),
                f"{target} ({label}) already exists; pass overwrite=True/--overwrite to replace it")
    # The evaluator refuses a pre-existing report staging file even with overwrite (it is not known
    # to be owned); refusing it here as well keeps a doomed run from freezing first and failing last.
    report_staging = fb._temporary_path(final["evaluation report"])
    require(not os.path.lexists(report_staging),
            f"{report_staging} already exists. It is not known to belong to this run (it may be "
            "another file linked under the report's staging name), so it is never truncated or "
            "deleted, even with overwrite=True. Nothing was written: inspect it, remove it "
            "yourself if it is stale, and run again")


def _artifact_binding_check(artifact, envelope, key, plan, index):
    """The cell's artifact must describe THIS cell: the common support/prior/utility/offered
    roster, the planned configuration, an acquisition that is the seeded selection, and (the part a
    rehashed substitution would otherwise slip through) its OWN representation, readout map, cache,
    extractor/activation/conditioning record and capacity, each equal to what the cell key, the plan
    and the index's representation summary claim. Hashes are integrity, not authenticity: a forger
    who rewrites the whole chain consistently still verifies; an internally INCONSISTENT claim
    (another representation's valid artifact under this key, a wrong cache, a doctored capacity)
    does not."""
    common = index["common"]
    support = common["support"]
    manifest = artifact["manifest"]
    require(manifest == {name: common["manifest"][name]
                         for name in ("target", "assay", "utility", "grouping_rationale")},
            f"{key.cell_id}: the artifact manifest differs from the common manifest")
    require(artifact["support"]["ids"] == support["ids"]
            and artifact["support"]["identity_sha256"] == support["ids_sha256"]
            and artifact["support"]["groups"] == support["groups"]
            and artifact["support"]["sequences_sha256"] == support["sequences_sha256"],
            f"{key.cell_id}: the artifact support differs from the common support")
    require(fb._array_sha256(fb.decode_log(artifact["prior"]["log_weight"]))
            == support["prior_log_weight_sha256"]
            and fb.digest(artifact["prior"]["spec"]) == common["prior_spec_sha256"],
            f"{key.cell_id}: the artifact prior differs from the common prior")
    components = artifact["provenance"]["input_binding"]["components"]
    require(components["prior_spec_sha256"] == common["prior_spec_sha256"]
            and components["candidate_roster"] == common["roster"]
            and components["train"]["ids_sha256"] == common["offered"]["ids_sha256"]
            and components["train"]["utilities_sha256"] == common["offered"]["utilities_sha256"]
            and components["unlabelled_train"]["ids_sha256"] == common["unlabelled_train"]["ids_sha256"],
            f"{key.cell_id}: the artifact input binding differs from the common identity")
    expected_config = fb._builtin({**plan.config_for(key).document(), "schema": fb.ARTIFACT_SCHEMA})
    require(artifact["config"] == expected_config,
            f"{key.cell_id}: the artifact configuration differs from the planned one")
    offered = common["offered"]["count"]
    if key.label_budget is None or key.label_budget <= offered:
        expected = fb.acquire_labels(offered, key.label_budget, key.seed).tolist()
        require(artifact["training"]["acquired_indices"] == expected,
                f"{key.cell_id}: the acquired indices are not the seeded selection")
    space = artifact["features"]["space"]
    schema = space["schema"]
    require(schema["acquired"]["ids"] == artifact["training"]["ids"]
            and schema["acquired"]["indices"] == artifact["training"]["acquired_indices"]
            and schema["transform_sha256"] == fb.digest(schema["transform"]),
            f"{key.cell_id}: the readout record does not describe the stored acquisition")
    # ---- the artifact's own representation / map / cache claims against the key, plan and index
    summary, spec = index["representations"][key.representation], plan.spec(key.map_id)
    expected_map = spec.document()
    require(space["kind"] == "frozen_readout_map"
            and schema["representation_id"] == key.representation == summary["representation_id"]
            and schema["map"] == expected_map and schema["transform"]["map"] == expected_map,
            f"{key.cell_id}: the artifact was built for another representation or readout map "
            "than this cell's key, the plan and the index record")
    provenance = schema["provenance"]
    require(provenance["canonical_cache_sha256"] == summary["cache_sha256"]
            and provenance["cache_feature_space_sha256"] == summary["feature_space_sha256"],
            f"{key.cell_id}: the artifact was fitted on a different cache (canonical digest or "
            f"feature space) than the index records for representation {key.representation!r}")
    require(provenance["extractor"] == summary["extractor"]
            and provenance["activation"] == summary["activation"]
            and provenance["conditioning"] == summary["conditioning"]
            and space["provenance"] == summary["feature_supervision"],
            f"{key.cell_id}: the artifact's extractor, activation, conditioning or feature "
            f"supervision record differs from the index's record of {key.representation!r}")
    require(provenance["candidate_roster"] == common["roster"]
            and provenance["offered_ids_sha256"] == common["offered"]["ids_sha256"],
            f"{key.cell_id}: the artifact's candidate roster differs from the common roster")
    acquisition, applied = schema["acquired"], schema["applied_to"]
    require(acquisition["budget"] == key.label_budget and acquisition["seed"] == key.seed
            and acquisition["offered_labelled_count"] == offered
            and acquisition["acquired_count"] == len(artifact["training"]["ids"])
            and applied["offered_rows"] == offered
            and applied["held_out_rows"] == support["count"],
            f"{key.cell_id}: the readout's acquisition or applied-row counts differ from the key "
            "and the common roster")
    capacity, names, centers = schema["capacity"], space["names"], schema["transform"]["centers"]
    if spec.kind == MAP_LINEAR:
        width_ok = (centers is None and schema["center_ids"] is None
                    and len(names) == summary["feature_dimension"])
    else:
        positions = None if centers is None else centers["acquired_positions"]
        width_ok = (centers is not None and isinstance(positions, list)
                    and positions == sorted(set(positions))
                    and all(isinstance(p, int) and 0 <= p < len(acquisition["ids"])
                            for p in positions)
                    and centers["count"] == len(names) == len(positions)
                    and schema["center_ids"] == [acquisition["ids"][p] for p in positions])
    require(width_ok and len(schema["transform"]["mean"]) == summary["feature_dimension"]
            == capacity["input_dimension"] and capacity["output_dimension"] == len(names)
            and capacity["acquired_rows"] == len(artifact["training"]["ids"])
            and capacity["ridge_alpha"] == artifact["config"]["ridge_alpha"],
            f"{key.cell_id}: the readout's dimensions (input, centers, output, acquired rows, "
            "ridge alpha) disagree with the cache summary and the stored configuration")
    require(envelope["capacity"] == {**capacity, "preprocessing_cost": schema["preprocessing_cost"]},
            f"{key.cell_id}: the cell's reported capacity differs from the capacity inside its "
            "frozen artifact")
    train = np.array(artifact["training"]["features"], dtype=np.float64)
    require(train.shape == (capacity["acquired_rows"], capacity["output_dimension"]),
            f"{key.cell_id}: the stored acquired design does not have the reported shape")
    squares = np.linalg.svd(_anchored_centre(train), compute_uv=False) ** 2
    df = float(np.sum(squares / (squares + capacity["ridge_alpha"])))
    # The df is recomputed from the STORED (already trace-scaled) rows, which differ from the fit's
    # own design by rounding only; the tolerance catches a doctored number, not a last-digit change.
    # The rank is only bounded: its numerical threshold is not reproducible to the last singular
    # value, and refusing a valid freeze over it would be worse than missing a forged rank.
    require(math.isclose(df, capacity["effective_ridge_df"], rel_tol=1e-4, abs_tol=1e-8)
            and 0 <= capacity["design_rank"] <= min(train.shape),
            f"{key.cell_id}: the reported effective ridge df / rank is not that of the stored "
            f"acquired design (recomputed df {df:.12g})")


def verify_freeze(output_dir):
    """Reload and verify the whole freeze from disk; raise unless every part is consistent.

    The index hash, the plan and common digests, the per-representation cache summaries (exactly the
    planned representations), the exact cell set (no missing or extra cell), each cell file's
    digest, its key/status/reason against the index, and for each successful cell the baseline's own
    ``verify_artifact`` plus the binding to the common identity, the plan and the cell's OWN
    representation, map, cache and capacity claims (see ``_artifact_binding_check``). Cell paths are
    re-derived from validated keys, never read from the index. Returns the parsed pieces plus the
    ``directory`` they were read from; it opens no evaluation outcome. The returned dict is a
    snapshot, not a capability: nothing downstream trusts it for another directory or a later disk
    state (``evaluate_matrix`` re-verifies).
    """
    out = Path(output_dir)
    index = fb.read_json(out / INDEX_NAME)
    _keys(index, ("schema", "plan", "plan_sha256", "common", "common_sha256", "representations",
                  "cells", "counts", "claims", "index_sha256"), label="freeze index")
    require(index["schema"] == INDEX_SCHEMA, f"not a {INDEX_SCHEMA} index")
    body = {name: value for name, value in index.items() if name != "index_sha256"}
    require(fb.digest(body) == index["index_sha256"],
            "the freeze index was modified after it was written (index_sha256 no longer matches)")
    plan = parse_plan(index["plan"])
    require(plan.document() == index["plan"] and plan.fingerprint() == index["plan_sha256"],
            "the freeze index's plan is not canonical or no longer matches plan_sha256")
    common = index["common"]
    require(fb.digest(common) == index["common_sha256"],
            "the freeze index's common identity no longer matches common_sha256")
    summaries = index["representations"]
    require(isinstance(summaries, dict) and sorted(summaries) == list(plan.representations),
            "the freeze index's representation summaries are not exactly the planned "
            f"representations {list(plan.representations)}")
    for rep, summary in summaries.items():
        _keys(summary, REPRESENTATION_SUMMARY_KEYS, label=f"index representation {rep}")
        require(summary["representation_id"] == rep,
                f"the index's summary under {rep!r} describes representation "
                f"{summary['representation_id']!r}")
    keys = enumerate_cells(plan)
    entries = index["cells"]
    require(isinstance(entries, list) and len(entries) == len(keys)
            and [entry.get("cell_id") if isinstance(entry, dict) else None for entry in entries]
            == [key.cell_id for key in keys],
            "the freeze index does not list exactly the planned cells (a cell is missing, extra "
            "or out of order)")
    cells_dir = out / CELLS_DIR
    require(cells_dir.is_dir(), f"the cells directory {cells_dir} is missing")
    listed = {key.filename() for key in keys}
    stray = sorted(item.name for item in cells_dir.iterdir() if item.name not in listed)
    require(not stray, f"{cells_dir} holds {len(stray)} file(s) the freeze index does not list, "
                       f"e.g. {stray[:3]}")
    envelopes = {}
    for key, entry in zip(keys, entries):
        _keys(entry, ("cell_id", "file", "status", "reason", "cell_sha256", "artifact_sha256"),
              label=f"index entry {key.cell_id}")
        path = out / CELLS_DIR / key.filename()
        require(entry["file"] == f"{CELLS_DIR}/{key.filename()}",
                f"index entry {key.cell_id} names an unexpected file {entry['file']!r}")
        require(path.is_file(), f"cell file {path} is missing")
        envelope = fb.read_json(path)
        require(fb.digest(envelope) == entry["cell_sha256"],
                f"cell file {path.name} was modified after the freeze (cell_sha256 mismatch)")
        _keys(envelope, ("schema", "cell_id", "key", "common_sha256", "map", "status", "reason",
                         "capacity", "artifact"), label=f"cell {key.cell_id}")
        require(envelope["schema"] == CELL_SCHEMA and envelope["cell_id"] == key.cell_id
                and envelope["key"] == key.document()
                and envelope["common_sha256"] == index["common_sha256"]
                and envelope["map"] == plan.spec(key.map_id).document(),
                f"cell {key.cell_id} does not match its key, the plan map or the common identity")
        require(envelope["status"] in STATUSES and envelope["status"] == entry["status"]
                and envelope["reason"] == entry["reason"],
                f"cell {key.cell_id}: status/reason disagree with the index")
        if envelope["status"] == STATUS_OK:
            artifact = envelope["artifact"]
            fb.verify_artifact(artifact)
            require(artifact["hashes"]["artifact_sha256"] == entry["artifact_sha256"],
                    f"cell {key.cell_id}: artifact hash disagrees with the index")
            with fb._malformed_artifact_as_value_error():
                _artifact_binding_check(artifact, envelope, key, plan, index)
        else:
            require(envelope["artifact"] is None and entry["artifact_sha256"] is None
                    and isinstance(envelope["reason"], str) and envelope["reason"],
                    f"cell {key.cell_id}: a {envelope['status']} cell must carry a reason and no "
                    "artifact")
        envelopes[key.cell_id] = envelope
    counts = {status: sum(1 for item in envelopes.values() if item["status"] == status)
              for status in STATUSES}
    require(index["counts"] == {"cells": len(keys), **counts}, "the index counts are wrong")
    return {"index": index, "plan": plan, "envelopes": envelopes, "keys": keys, "directory": out}


# ---------------------------------------------------------------------------
# evaluation: opened only after the freeze is on disk and verified
# ---------------------------------------------------------------------------

def paired_group_bootstrap(cell_errors, reference_errors, groups, *, replicates, seed,
                           confidence, min_groups, stream=()):
    """Paired cluster bootstrap of the MSE difference ``cell - reference`` over support groups.

    ``cell_errors`` / ``reference_errors`` are per-candidate squared errors on the SAME candidates;
    the unit resampled is the group (with replacement), and each replicate uses the same resampled
    groups for both cells, so the statistic is the pooled mean of the per-candidate differences.
    Percentile interval. ``interval`` is ``None`` (status ``insufficient_groups``) when there are
    fewer than ``min_groups`` groups. A negative difference means the cell has the lower MSE.
    """
    cell = np.asarray(cell_errors, dtype=np.float64)
    reference = np.asarray(reference_errors, dtype=np.float64)
    require(cell.ndim == 1 and cell.shape == reference.shape and cell.shape[0] == len(groups),
            "the paired errors and the group labels must align")
    difference = cell - reference
    labels = sorted(set(groups))
    position = {label: i for i, label in enumerate(labels)}
    member = np.array([position[label] for label in groups], dtype=np.int64)
    record = {"n_candidates": int(cell.shape[0]), "n_groups": len(labels),
              "mse_cell": float(cell.mean()), "mse_reference": float(reference.mean()),
              "mse_difference": float(difference.mean()), "confidence": confidence,
              "replicates": replicates, "min_groups": min_groups,
              "interval_method": "percentile",
              "resampling_unit": ("support group, drawn with replacement; the same draws are used "
                                  "for the cell and the reference"),
              "sign": "cell minus reference; negative means the cell has the lower MSE"}
    if len(labels) < min_groups:
        return {**record, "status": "insufficient_groups", "interval": None,
                "reason": f"{len(labels)} support group(s) < min_groups={min_groups}"}
    sums = np.zeros(len(labels))
    sizes = np.zeros(len(labels))
    np.add.at(sums, member, difference)
    np.add.at(sizes, member, 1.0)
    rng = np.random.default_rng([int(seed), *[int(item) for item in stream]])
    draws = rng.integers(0, len(labels), size=(replicates, len(labels)))
    statistic = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(statistic, [tail, 1.0 - tail])
    return {**record, "status": "ok", "interval": [float(low), float(high)], "reason": None}


def evaluate_matrix(output_dir, eval_outcomes, *, overwrite=False):
    """Verify the freeze in ``output_dir`` AS IT IS ON DISK NOW, then open the evaluation outcomes
    and score every successful cell.

    No caller-held state is accepted: a ``verify_freeze`` result from earlier (or from another
    directory) is not a credential, because the files may have changed since. Before the outcomes
    are opened the evaluation file is refused if it resolves to the report, the report's ``.tmp``
    staging file or any frozen file (cells, index, input record), and an existing report is replaced
    only with ``overwrite=True``. A report staging file that already exists is refused even with
    ``overwrite=True`` (it is not known to be this evaluator's and may be linked to a recorded input);
    the report is staged through a file this call creates exclusively. Writes
    ``matrix_evaluation_report.json`` into ``output_dir`` and
    returns the report. Used on its own, it scores a freeze that ``run_matrix`` left unscored (for
    example because the evaluation file was missing).
    """
    return _evaluate_verified(verify_freeze(output_dir), eval_outcomes, overwrite=overwrite)


def _evaluate_verified(verified, eval_outcomes, *, overwrite):
    """Score a freeze that the CALLER JUST verified from disk (``evaluate_matrix`` and
    ``run_matrix``). Private on purpose: there is no public way to hand it unchecked state."""
    out = Path(verified["directory"])
    index, plan, envelopes, keys = (verified["index"], verified["plan"], verified["envelopes"],
                                    verified["keys"])
    report_path = out / RESULTS_NAME
    staging_path = fb._temporary_path(report_path)
    frozen = {label: path for label, path in _output_paths(out, keys).items()
              if path != report_path}
    # The evaluation file is the one input here. It is compared with EVERY frozen file (cells, index,
    # input record) before it is opened: a frozen file is not an outcome table, and handing one over
    # must be refused by name rather than discovered when the CSV parse happens to fail.
    for label, frozen_path in frozen.items():
        require(not fb._same_file(eval_outcomes, frozen_path),
                f"the evaluation outcomes file {eval_outcomes} resolves to the same file as the "
                f"{label} {frozen_path}; a frozen file is never read as outcomes. Nothing was "
                "opened or written: pass the held-out outcome table")
    fb._refuse_path_aliases(
        {"evaluation outcomes file": eval_outcomes},
        {"evaluation report": report_path, "evaluation report temporary file": staging_path})
    require(not report_path.is_dir(), f"{report_path} exists and is a directory")
    require(overwrite or not report_path.exists(),
            f"{report_path} already exists; pass overwrite=True/--overwrite to replace it")
    # A report staging file that already exists is not known to be this evaluator's: it may be a
    # hard link or a junction onto a file the freeze recorded (a cache header, say), and the write
    # below would truncate it. ``overwrite`` licenses replacing the previous REPORT, never an
    # unknown staging path, so it is refused even then, before anything is opened or written.
    require(not os.path.lexists(staging_path),
            f"{staging_path} already exists. It is not known to belong to this evaluation (it may "
            "be another file linked under the report's staging name), so it is never truncated or "
            "deleted, even with overwrite=True. Nothing was opened or written: inspect it, remove "
            "it yourself if it is stale, and run the evaluation again")
    # ---- the first moment an evaluation outcome is opened ----
    outcomes = fb.read_outcomes(eval_outcomes, label="evaluation outcomes")
    support = index["common"]["support"]
    truth = fb._outcome_vector(support["ids"], outcomes)
    reports, errors = {}, {}
    for key in keys:
        envelope = envelopes[key.cell_id]
        if envelope["status"] != STATUS_OK:
            continue
        artifact = envelope["artifact"]
        reports[key.cell_id] = fb.evaluate_frozen(artifact, outcomes)
        predicted = np.array(artifact["predictions"]["utility"], dtype=np.float64)
        errors[key.cell_id] = (predicted - truth) ** 2
    reference = plan.reference
    cells = []
    for key in keys:
        envelope = envelopes[key.cell_id]
        item = {"cell_id": key.cell_id, "key": key.document(), "status": envelope["status"],
                "reason": envelope["reason"], "map": envelope["map"],
                "capacity": envelope["capacity"], "evaluator_report": None, "readout": None,
                "comparison_to_reference": None}
        if envelope["status"] == STATUS_OK:
            mse = float(errors[key.cell_id].mean())
            item["evaluator_report"] = reports[key.cell_id]
            item["readout"] = {"mse": mse, "rmse": math.sqrt(mse),
                               "acquired_ids_sha256": envelope["artifact"]["provenance"][
                                   "label_accounting"]["acquired_ids_sha256"]}
        ref_key = CellKey(reference["representation"], reference["map"], key.label_budget,
                          key.seed)
        if (key.representation, key.map_id) == (reference["representation"], reference["map"]):
            item["comparison_to_reference"] = {"status": "is_reference",
                                               "reference_cell": ref_key.cell_id}
        elif envelope["status"] != STATUS_OK:
            item["comparison_to_reference"] = {"status": "cell_unavailable",
                                               "reference_cell": ref_key.cell_id,
                                               "reason": envelope["reason"]}
        elif envelopes[ref_key.cell_id]["status"] != STATUS_OK:
            ref_envelope = envelopes[ref_key.cell_id]
            item["comparison_to_reference"] = {
                "status": "reference_unavailable", "reference_cell": ref_key.cell_id,
                "reference_status": ref_envelope["status"], "reference_reason": ref_envelope["reason"]}
        else:
            ref_hash = envelopes[ref_key.cell_id]["artifact"]["provenance"]["label_accounting"][
                "acquired_ids_sha256"]
            require(ref_hash == item["readout"]["acquired_ids_sha256"],
                    f"{key.cell_id} and its reference acquired different labels; the comparison "
                    "would not be paired")
            comparison = paired_group_bootstrap(
                errors[key.cell_id], errors[ref_key.cell_id], support["groups"],
                replicates=plan.bootstrap["replicates"], seed=plan.bootstrap["seed"],
                confidence=plan.bootstrap["confidence"], min_groups=plan.bootstrap["min_groups"],
                stream=(7, key.seed, 0 if key.label_budget is None else key.label_budget))
            item["comparison_to_reference"] = {
                **comparison, "reference_cell": ref_key.cell_id, "paired_acquisition": True,
                "seed_note": ("this seed only; nothing is pooled across seeds, which share the "
                              "candidates and are not independent")}
        cells.append(item)
    counts = {status: sum(1 for item in cells if item["status"] == status) for status in STATUSES}
    results = {
        "schema": RESULTS_SCHEMA, "index_sha256": index["index_sha256"],
        "plan_sha256": index["plan_sha256"], "claims": dict(CLAIMS),
        "no_winner_selected": CLAIMS["no_winner"],
        "evaluation_outcomes": {
            "required": len(support["ids"]), "provided": len(support["ids"]), "complete": True,
            "canonical_sha256": fb.digest({"ids": list(support["ids"]),
                                           "utilities": truth.tolist()}),
            "note": "digest of the outcomes in sorted-id order, so it does not depend on file row order",
            "rule": "every held-out candidate needs an outcome; none is imputed"},
        "reference": dict(reference), "status_counts": {"cells": len(cells), **counts},
        "cells": cells}
    _write_report(report_path, results)
    return results


def _write_report(report_path, results):
    """Strict JSON through a staging file this call CREATES, then a replace.

    ``fb.write_json`` opens its staging path for writing, which truncates whatever is there. The
    report's staging name is created exclusively instead: a file that appeared between the
    preflight and this write makes the call fail rather than be overwritten, and only a file this
    call created is ever removed (when the write itself fails), so a stale staging file cannot be
    left behind by this function and an unknown one is never touched.
    """
    staging = fb._temporary_path(report_path)
    text = json.dumps(fb._builtin(results), indent=2, sort_keys=True, allow_nan=False) + "\n"
    created = False
    try:
        with staging.open("x", encoding="utf-8") as handle:
            created = True
            handle.write(text)
        staging.replace(report_path)
    except BaseException:
        if created and os.path.lexists(staging):
            try:
                staging.unlink()
            except OSError:
                pass
        raise


def summarize_evaluation(output_dir, results):
    """The compact run summary (paths, digests, status counts, per-cell status/reason)."""
    out = Path(output_dir)
    return {"output_dir": str(out), "freeze_index": str(out / INDEX_NAME),
            "evaluation_report": str(out / RESULTS_NAME), "index_sha256": results["index_sha256"],
            "plan_sha256": results["plan_sha256"], "status_counts": results["status_counts"],
            "cells": [{"cell_id": item["cell_id"], "status": item["status"],
                       "reason": item["reason"]} for item in results["cells"]]}


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def _path_mapping(value, name):
    require(isinstance(value, dict), f"{name} must map a representation id to a path")
    return {safe_id(key, f"{name} key"): Path(path) for key, path in value.items()}


def _file_record(path):
    path = Path(path)
    return {"name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def run_matrix(*, candidates, train_outcomes, eval_outcomes, manifest, plan, caches, output_dir,
               cache_npz=None, overwrite=False):
    """Freeze every cell, verify the freeze from disk, and only then open the evaluation outcomes.

    ``caches`` maps each planned representation id to its header path (exactly the planned set);
    ``cache_npz`` optionally maps an id to an explicit NPZ path. Order: refuse any input/output
    alias (resolved paths, staging files included, even with ``overwrite``) -> parse the manifest,
    plan and caches -> build every cell in memory -> write the cells, the input record and the
    freeze index -> ``verify_freeze`` -> read the evaluation outcomes -> evaluate. A missing or
    incomplete evaluation file therefore leaves a verified freeze that never saw it.
    """
    out = Path(output_dir)
    require(not out.exists() or out.is_dir(),
            f"the output directory {out} exists and is not a directory")
    parsed_manifest, parsed_plan = load_matrix_manifest(manifest), load_plan(plan)
    headers = _path_mapping(caches, "caches")
    explicit = _path_mapping(cache_npz or {}, "cache_npz")
    require(set(headers) == set(parsed_plan.representations),
            f"the caches {sorted(headers)} must be exactly the planned representations "
            f"{list(parsed_plan.representations)}")
    require(set(explicit) <= set(headers), "cache_npz names a representation without a cache")
    npz = {rep: resolve_npz_path(headers[rep],
                                 _parse_header(fb.read_json(headers[rep])).storage["file"],
                                 explicit.get(rep)) for rep in headers}
    keys = enumerate_cells(parsed_plan)
    final = _output_paths(out, keys)
    everything = _with_staging(final)
    inputs = {"candidates file": candidates, "training outcomes file": train_outcomes,
              "evaluation outcomes file": eval_outcomes, "manifest file": manifest,
              "plan file": plan}
    for rep in sorted(headers):
        inputs[f"cache header {rep}"] = headers[rep]
        inputs[f"cache NPZ {rep}"] = npz[rep]
    fb._refuse_path_aliases(inputs, everything)
    _check_output_state(out, final, everything, overwrite)
    table = fb.read_candidates(candidates, alphabet=parsed_manifest.alphabet,
                               length=parsed_manifest.length)
    roster = make_roster(parsed_manifest, table,
                         fb.read_outcomes(train_outcomes, label="training outcomes"))
    loaded = {}
    for rep in sorted(headers):
        cache = load_feature_cache(headers[rep], npz_path=npz[rep])
        require(cache.representation_id == rep,
                f"the cache at {headers[rep]} is representation {cache.representation_id!r}, "
                f"not the planned {rep!r}")
        check_cache_against_roster(cache, table, parsed_manifest)
        loaded[rep] = cache
    common = roster.common_document()
    common_sha256 = fb.digest(common)
    envelopes = [run_cell(key, loaded[key.representation], roster, parsed_plan, common_sha256)
                 for key in keys]
    entries = []
    for key, envelope in zip(keys, envelopes):
        artifact = envelope["artifact"]
        entries.append({
            "cell_id": key.cell_id, "file": f"{CELLS_DIR}/{key.filename()}",
            "status": envelope["status"], "reason": envelope["reason"],
            "cell_sha256": fb.digest(envelope),
            "artifact_sha256": None if artifact is None else artifact["hashes"]["artifact_sha256"]})
    counts = {status: sum(1 for item in envelopes if item["status"] == status)
              for status in STATUSES}
    index = {"schema": INDEX_SCHEMA, "plan": parsed_plan.document(),
             "plan_sha256": parsed_plan.fingerprint(), "common": common,
             "common_sha256": common_sha256,
             "representations": {rep: loaded[rep].summary() for rep in sorted(loaded)},
             "cells": entries, "counts": {"cells": len(keys), **counts}, "claims": dict(CLAIMS)}
    index["index_sha256"] = fb.digest(index)
    input_record = {
        "schema": INPUTS_SCHEMA,
        "files": {"candidates": _file_record(candidates),
                  "training_outcomes": _file_record(train_outcomes),
                  "manifest": _file_record(manifest), "plan": _file_record(plan)},
        "caches": {rep: {"header": _file_record(headers[rep]), "npz": _file_record(npz[rep]),
                         "raw": loaded[rep].raw, "canonical_cache_sha256": loaded[rep].cache_sha256}
                   for rep in sorted(loaded)},
        "evaluation_outcomes": "not opened and not hashed before the freeze was verified",
        "note": ("raw, stored-order file digests (file NAMES only, no paths). They depend on row "
                 "order and are not part of any frozen identity, which uses canonical digests.")}
    for key, envelope in zip(keys, envelopes):
        fb.write_json(out / CELLS_DIR / key.filename(), envelope)
    fb.write_json(out / INPUTS_NAME, input_record)
    fb.write_json(out / INDEX_NAME, index)
    verified = verify_freeze(out)
    # ---- the frozen boundary: nothing above opened an evaluation outcome ----
    results = _evaluate_verified(verified, eval_outcomes, overwrite=overwrite)
    return summarize_evaluation(out, results)


# ---------------------------------------------------------------------------
# selftest: synthetic complete sequences, features and caches, and a real matrix
# ---------------------------------------------------------------------------

SELFTEST_REPRESENTATIONS = ("noise", "onehot", "randproj")


def _explicit_space(prefix, width, description):
    names = tuple(f"{prefix}{i}" for i in range(width))
    return fb.FeatureSpace(
        kind="explicit", names=names, schema={"columns": list(names), "construction": description},
        history=(description,),
        supervision=fb.SupervisionHistory(status=fb.HISTORY_LABEL_FREE,
                                          description="synthetic features; no outcomes used",
                                          exposure_status="none"))


def write_synthetic_world(directory, seed=0, *, permute=False):
    """Small synthetic complete sequences, candidate/outcome CSVs, a matrix manifest and plan, and
    three caches: ``onehot`` (the informative planted-truth encoding), ``randproj`` (a fixed random
    projection of it) and ``noise`` (independent Gaussian noise, a null control). With ``permute``
    the CSV rows and the cache rows are shuffled, which must change no frozen byte. Returns the
    explicit paths. (The baseline's own ``manifest.json`` is also written there and is unused.)"""
    directory = Path(directory)
    base = fb.write_synthetic_inputs(directory, seed)
    table = fb.read_candidates(base["candidates"], alphabet=fb.SELFTEST_ALPHABET,
                               length=fb.SELFTEST_LENGTH)
    count = len(table.ids)
    rng = np.random.default_rng([seed, 300])
    if permute:
        for name in ("candidates", "train_outcomes", "eval_outcomes"):
            lines = base[name].read_text(encoding="utf-8").splitlines()
            body = lines[1:]
            order = np.random.default_rng([seed, 301]).permutation(len(body))
            base[name].write_text("\n".join([lines[0]] + [body[int(i)] for i in order]) + "\n",
                                  encoding="utf-8")
    onehot_space = fb.onehot_feature_space(fb.SELFTEST_ALPHABET, fb.SELFTEST_LENGTH,
                                           fb.SupervisionHistory(status=fb.HISTORY_LABEL_FREE,
                                                                 exposure_status="none"))
    onehot = fb.onehot_encode(table.sequences, onehot_space)
    projection = rng.normal(size=(onehot.shape[1], 6)) / 4.0
    noise = rng.normal(size=(count, 5))
    representations = {
        "onehot": (onehot_space, onehot, "fixed one-hot encoding of the complete sequence"),
        "randproj": (_explicit_space("rp", 6, "fixed random Gaussian projection of the one-hot "
                                              "encoding"), onehot @ projection, "random projection"),
        "noise": (_explicit_space("nz", 5, "independent Gaussian noise (a null control)"), noise,
                  "noise null control")}
    order = (np.random.default_rng([seed, 302]).permutation(count).tolist() if permute
             else list(range(count)))
    cache_dir = directory / "caches"
    caches, cache_npz = {}, {}
    for rep, (space, features, description) in representations.items():
        header, npz_path = cache_dir / f"{rep}.cache.json", cache_dir / f"{rep}.cache.npz"
        write_feature_cache(
            header, npz_path, representation_id=rep, ids=[table.ids[i] for i in order],
            sequences=[table.sequences[i] for i in order],
            groups=[table.groups[i] for i in order], splits=[table.splits[i] for i in order],
            features=features[order], space=space,
            identity={"kind": IDENTITY_COMPLETE_SEQUENCE, "alphabet": fb.SELFTEST_ALPHABET,
                      "length": fb.SELFTEST_LENGTH,
                      "construct_description": "one chain of four residues; the whole sequence"},
            extractor={"kind": "synthetic", "checkpoint_sha256": None, "source_sha256": None,
                       "description": f"synthetic: {description}"},
            activation={"location": "synthetic closed form", "pooling": "none (one vector per "
                                                                         "candidate)",
                        "token_map": "residue letters of the complete sequence"},
            conditioning={"candidate_masking": "none", "answer_free": True,
                          "family": {"recorded": False},
                          "assay": {"recorded": True,
                                    "values": ["synthetic-closed-form"] * count},
                          "quality": {"recorded": True, "values": [None] * count}},
            dtype="float64", overwrite=True)
        caches[rep], cache_npz[rep] = header, npz_path
    manifest = directory / "matrix_manifest.json"
    fb.write_json(manifest, {
        "schema": MANIFEST_SCHEMA, "target": "synthetic-additive-target",
        "assay": "synthetic-closed-form",
        "utility": {"name": "planted additive utility", "units": "arbitrary", "scale": "linear",
                    "higher_is_better": True, "useful_threshold": 1.5},
        "grouping_rationale": ("groups are the first two residues; whole groups are held out so "
                               "no group straddles the split. This is a shared-prefix grouping "
                               "only: it does not enforce a minimum sequence distance"),
        "candidates": {"identity_kind": IDENTITY_COMPLETE_SEQUENCE,
                       "alphabet": fb.SELFTEST_ALPHABET, "length": fb.SELFTEST_LENGTH},
        "prior": {"kind": fb.PRIOR_JOINT,
                  "semantics": ("log probability of the whole sequence under an independent "
                                "per-position categorical; sequences ending in AA are forbidden "
                                "(structural zero)"),
                  "provenance": {"status": "label_free",
                                 "description": "synthetic categorical, no outcomes used",
                                 "sequence_only_exposure": {"status": "none"}}},
        "notes": "synthetic selftest inputs"})
    plan = directory / "matrix_plan.json"
    fb.write_json(plan, {
        "schema": PLAN_SCHEMA, "representations": sorted(representations),
        "maps": [{"id": "linear", "kind": "linear", "center": True, "scale": "std",
                  "trace_normalize": True},
                 {"id": "rbf_median", "kind": "rbf", "center": True, "scale": "std",
                  "trace_normalize": True,
                  "bandwidth": {"rule": BANDWIDTH_MEDIAN, "multiplier": 1.0}}],
        "label_budgets": [16, 32], "seeds": [0, 1],
        "baseline": {"tau": 1.0, "ridge_alpha": 1.0, "sir_proposal_draws": 256, "final_draws": 32,
                     "dpo_pressures": [0.5, 1.0, 2.0]},
        "reference": {"representation": "onehot", "map": "linear"},
        "bootstrap": {"replicates": 200, "seed": 0, "confidence": 0.9, "min_groups": 3},
        "notes": "synthetic selftest plan; every setting is fixed, nothing is tuned"})
    return {"candidates": base["candidates"], "train_outcomes": base["train_outcomes"],
            "eval_outcomes": base["eval_outcomes"], "manifest": manifest, "plan": plan,
            "caches": caches, "cache_npz": cache_npz}


def _files_below(root, names):
    root = Path(root)
    return {name: (root / name).read_bytes() for name in names}


def _pre_evaluation_names(directory, *, with_inputs):
    directory = Path(directory)
    names = sorted(f"{CELLS_DIR}/{item.name}" for item in (directory / CELLS_DIR).iterdir())
    names.append(INDEX_NAME)
    if with_inputs:
        names.append(INPUTS_NAME)
    return names


def selftest(output_dir, seed=0):
    """Write a synthetic world, run a real 24-cell matrix and check what must hold. Returns the
    report (also written as ``matrix_selftest_report.json``); ``passed`` only if every check holds.
    It qualifies the machinery on a known synthetic truth, not any representation or biology."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, holds, detail=None):
        checks.append({"name": name, "holds": bool(holds), "detail": detail})
    world = write_synthetic_world(out / "synthetic", seed)
    summary = run_matrix(**world, output_dir=out / "matrix", overwrite=True)
    shuffled = out / "synthetic" / "eval_outcomes_shuffled.csv"
    fb._shuffled_outcomes(world["eval_outcomes"], shuffled, seed)
    second = run_matrix(**{**world, "eval_outcomes": shuffled},
                        output_dir=out / "matrix_shuffled_evaluation", overwrite=True)
    permuted_world = write_synthetic_world(out / "synthetic_permuted", seed, permute=True)
    third = run_matrix(**permuted_world, output_dir=out / "matrix_permuted", overwrite=True)
    verified = verify_freeze(out / "matrix")
    results = fb.read_json(out / "matrix" / RESULTS_NAME)
    shuffled_results = fb.read_json(out / "matrix_shuffled_evaluation" / RESULTS_NAME)
    names = _pre_evaluation_names(out / "matrix", with_inputs=True)
    check("matrix::every_planned_cell_is_ok_and_listed",
          summary["status_counts"] == {"cells": 24, STATUS_OK: 24, STATUS_UNSUPPORTED: 0,
                                       STATUS_FAILED: 0}, summary["status_counts"])
    check("matrix::the_freeze_reloads_and_verifies_from_disk",
          verified["index"]["index_sha256"] == summary["index_sha256"])
    check("matrix::evaluation_outcomes_cannot_change_any_pre_evaluation_file",
          names == _pre_evaluation_names(out / "matrix_shuffled_evaluation", with_inputs=True)
          and _files_below(out / "matrix", names)
          == _files_below(out / "matrix_shuffled_evaluation", names)
          and summary["index_sha256"] == second["index_sha256"])
    check("matrix::the_evaluation_report_does_change_with_the_outcomes",
          results["cells"] != shuffled_results["cells"])
    permuted_names = _pre_evaluation_names(out / "matrix_permuted", with_inputs=False)
    check("matrix::row_order_permutation_changes_no_cell_index_or_result_byte",
          permuted_names == _pre_evaluation_names(out / "matrix", with_inputs=False)
          and _files_below(out / "matrix", permuted_names + [RESULTS_NAME])
          == _files_below(out / "matrix_permuted", permuted_names + [RESULTS_NAME])
          and summary["index_sha256"] == third["index_sha256"])
    acquired = {}
    for key in verified["keys"]:
        artifact = verified["envelopes"][key.cell_id]["artifact"]
        acquired.setdefault((key.label_budget, key.seed), set()).add(
            tuple(artifact["training"]["ids"]))
    check("matrix::acquisitions_are_paired_across_representations_and_maps",
          all(len(variants) == 1 for variants in acquired.values()),
          {f"b{b}/s{s}": len(v) for (b, s), v in acquired.items()})
    check("matrix::budgets_are_nested_for_every_seed",
          all(set(next(iter(acquired[(16, s)]))) <= set(next(iter(acquired[(32, s)])))
              for s in (0, 1)))
    check("matrix::every_regret_and_oracle_identity_holds",
          all(item["evaluator_report"]["identities"]["all_hold"] for item in results["cells"]))
    comparisons = [item["comparison_to_reference"] for item in results["cells"]]
    check("matrix::comparisons_are_paired_per_seed_with_an_interval_or_an_explicit_reason",
          all(item["status"] in ("is_reference", "ok") for item in comparisons)
          and all(item["interval"] is not None for item in comparisons if item["status"] == "ok")
          and sum(1 for item in comparisons if item["status"] == "is_reference") == 4)
    check("matrix::label_free_synthetic_histories_are_clean_low_n_eligible",
          all(item["evaluator_report"]["clean_low_n"]["status"] == "eligible"
              for item in results["cells"]))
    # A fixture directory this call creates and owns (``mkdtemp`` never reuses a path): nothing that
    # already exists under ``out`` is deleted, replaced or reused, and nothing is cleaned up
    # afterwards - the tampered copy is left for inspection and the caller may remove it.
    fixture = Path(tempfile.mkdtemp(prefix="matrix_tampered_", dir=out))
    tampered = fixture / "matrix"
    shutil.copytree(out / "matrix", tampered)
    victim = tampered / names[0]
    victim.write_text(victim.read_text(encoding="utf-8").replace('"ok"', '"failed"', 1),
                      encoding="utf-8")
    try:
        verify_freeze(tampered)
        refused = False
    except ValueError:
        refused = True
    check("matrix::a_tampered_cell_is_refused", refused)
    report = {
        "schema": SELFTEST_SCHEMA, "seed": seed, "passed": bool(all(c["holds"] for c in checks)),
        "checks": checks, "summary": summary, "claims": dict(CLAIMS),
        "inputs": {key: (str(value) if not isinstance(value, dict)
                         else {rep: str(path) for rep, path in value.items()})
                   for key, value in world.items()},
        "tampered_fixture": str(fixture),
        "note": ("synthetic qualification only: it shows the machinery computes what it says on a "
                 "known planted truth. The noise representation is a null control that is neither "
                 "required to lose nor to win, and no cell is selected.")}
    fb.write_json(out / SELFTEST_REPORT_NAME, report)
    return report
