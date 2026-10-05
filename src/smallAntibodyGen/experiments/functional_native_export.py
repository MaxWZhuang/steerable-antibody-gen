"""Native DPLM token-feature export, a ragged token cache and an exact pooled bridge to the matrix.

What this is. An opt-in, caller-driven path from an EXPLICITLY SUPPLIED native ``ConditionedDPLM``
instance to the frozen-feature matrix (``functional_matrix``): an export plan names the complete
candidates, the export runs the real native forward once over the UNMASKED ``H|L`` constructs and
stores every real native token's activation in a ragged, hash-bound cache, and the cache is then
pooled (twice, differently, with no second forward) into ordinary ``functional-feature-cache/1``
files the matrix already knows how to join, fit and freeze. What it is not: a weights loader (the
caller owns the model; nothing here discovers, downloads or deserialises a checkpoint), a claim that
any activation, pooling or readout wins, or evidence about biology. A tiny random CPU run qualifies
the machinery; it is not a pretrained or biological validation.

Usage (Python; the API is the real export path, the CLI only wraps plan/cache/pool handling)::

    from smallAntibodyGen.experiments import functional_native_export as fne

    model = ...     # YOUR ConditionedDPLM; build it however you load checkpoints (caller-owned)
    plan = fne.read_export_plan("plan.json")                      # strict, canonical, retained path
    fne.export_dplm_tokens(model, plan, "tok.json", "tok.npz", batch_size=8,
                           protected_input_paths=["candidates.csv"])
    cache = fne.load_token_cache("tok.json")                      # strict, hash-verified, frozen
    fne.write_pooled_matrix_cache(cache, table, manifest, "all.json", "all.npz",
                                  representation_id="fh_all", pooling="all_nonpad_mean",
                                  protected_input_paths=["candidates.csv", "manifest.json"])

Plan (``functional-native-export/1``): representation_id, conditioning_mode ``none|explicit``,
activation ``{kind: final_hidden|hook, name: null|exact tap}``, feature_supervision (a
``SupervisionHistory`` document), extractor ``{kind, checkpoint_sha256, source_sha256,
description}`` and rows ``{candidate_id, heavy, light, group_id, split, condition}``. The full
physical identity is ``heavy + "|" + light`` (a heavy-only record keeps its trailing delimiter), so
the chain boundary is part of identity and two constructs that share an edited window but differ in
framework are different identities. ``none`` passes NO condition vector (``condition_indices=None``);
zero indices would be a real "first family / low bin / first assay" request, so ``none`` is not
that. ``explicit`` encodes ``Condition(family, bin=null, assay)`` through the model's own
``condition_spec`` (an unknown assay is refused, never given an untrained row; JSON ``null`` is the
trained dropped-assay slot) and records both the logical condition and its indices. Quality is an
explicit null on every row: an outcome-derived bin is not an answer-free prediction feature.

Token cache (``functional-native-token-cache/1``): a strict JSON header plus an NPZ with EXACTLY the
members ``features`` (float32, sum(T_i) x hidden) and ``offsets`` (int64, N + 1). Only real native
tokens are stored, specials included; no batch padding row exists. The header carries the canonical
rows (full chains, groups, splits, conditions and their indices, per-row token maps), the feature
space (with the plan's supervision history), the extractor, the activation, the RUNTIME fingerprint
(see below) and its digest, array dtypes/shapes/content hashes, the NPZ file hash and a canonical
``cache_sha256``. The loader refuses anything it cannot recompute: members, dtypes, finiteness,
offsets against chain lengths, token maps against chains, condition indices against the recorded
``condition_spec``, every digest, and the recorded runtime against the rest of the header (hidden
size against the feature width in BOTH recorded configs, a hook's block against the recorded layer
count, every row's token count against the recorded positions, explicit conditioning against a
recorded ``use_conditions``). That last group compares a cache with ITSELF, never with the current
host or model, so a cache read on another machine is valid. Hashes are integrity, never
authenticity: a cache whose whole chain is rewritten consistently AND without contradicting itself
is not detected. ``TokenCache.space`` is derived from the hash-bound header on every access, so
editing the returned object changes a detached copy, never the provenance the pooled bridge uses.

Runtime fingerprint. A digest of the ACTUAL instance: the full ``asdict`` of its config(s), every
named parameter and buffer (tied names listed separately, non-persistent buffers such as the rotary
frequencies and the id tables included) by dtype, shape and raw bytes (bfloat16 included), module
types and topology, every simple module attribute (LayerNorm eps, dropout p, attention heads,
``lora_enabled``/``lora_scaling`` ...), the condition spec and ``use_conditions``, LoRA names and
settings, the tokenizer tables, the source digests of the modules involved, the declared
diffusion/activation-checkpointing provenance and the execution regime (eval, no_grad, autocast
off) with the torch version, device and backend, and the NUMERICAL BACKEND SETTINGS the native
linear/attention math reads (float32 matmul precision and the per-backend ``fp32_precision`` /
TF32 / reduced-precision-reduction flags, oneDNN and cuDNN enable/deterministic/benchmark flags,
the SDPA math/flash/memory-efficient/cuDNN switches, the deterministic-algorithms flags and the CPU
capability). A flag this torch does not have is recorded as ``"unsupported"`` and one that cannot be
read as ``"unreadable (<error type>)"``; nothing is guessed, and the reads are pure context getters
that touch no device. It is computed before and after the export, and the backend settings are
compared after every batch; a change refuses publication. A caller-asserted ``checkpoint_sha256``/``source_sha256`` stays
provenance, not an authenticated fact, and a parameter-state hash is never relabelled as one.

Safety. The plan is re-parsed from its own document at the export boundary
(``snapshot_export_plan``): a parsed plan holds mutable nested containers, so an edited or
hand-built ``ExportPlan`` is validated again and exported from a detached copy, or refused before
any compute or write; an invalid condition is refused, never normalised away. All input/output
paths (plan, protected inputs, header, NPZ and both staging files) are validated BEFORE any model
compute or write; aliases are compared as physical files (resolved paths, letter case where the
file system folds it, hard links). Only the paths the caller supplied
are protected: a ``CandidateTable`` or ``MatrixManifest`` does not retain its file name, so pass
those files in ``protected_input_paths``. Staging files are created exclusively (never truncated),
a stale one is refused even with ``overwrite=True``, unknown files are never deleted, and the
header is written last, so a half-published cache is refused by the loader. Existing forward and
forward-pre hooks, module-local or global, are REFUSED before the exporter installs its own
(capture-only) hook, and only the exporter's own handle is removed.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.modules.module as torch_module_lib
from torch import nn

from ..models import dplm_policy as dplm
from ..models import her2_routes_hooks as hooks_lib
from . import functional_baseline as fb
from . import functional_matrix as fm
from . import her2_routes_conditions as conditions_lib
from . import her2_routes_interp as interp
from . import her2_routes_serial as serial
from .her2_data import CANONICAL
from .her2_runtime import require, sha256 as file_sha256

PLAN_SCHEMA = "functional-native-export/1"
TOKEN_CACHE_SCHEMA = "functional-native-token-cache/1"
RUNTIME_SCHEMA = "functional-native-runtime/1"
SELFTEST_SCHEMA = "functional-native-export-selftest/1"
SELFTEST_REPORT_NAME = "native_export_selftest_report.json"

CONDITIONING_NONE, CONDITIONING_EXPLICIT = "none", "explicit"
CONDITIONING_MODES = (CONDITIONING_NONE, CONDITIONING_EXPLICIT)
ACTIVATION_FINAL, ACTIVATION_HOOK = "final_hidden", "hook"
ACTIVATION_KINDS = (ACTIVATION_FINAL, ACTIVATION_HOOK)
POOL_ALL, POOL_RESIDUE = "all_nonpad_mean", "residue_mean"
POOLINGS = (POOL_ALL, POOL_RESIDUE)
PLAN_EXTRACTOR_KINDS = ("native_checkpoint", "synthetic")

HOOK_ROUTE = "dplm"
DELIMITER = "|"
FEATURES_ARRAY, OFFSETS_ARRAY = "features", "offsets"
STORED_DTYPE = "float32"
OFFSET_DTYPE = "int64"
TOKEN_SPACE_KIND = "native_dplm_tokens"
POOLED_SPACE_KIND = "native_dplm_pooled"
UNSUPPORTED = "unsupported"

PLAN_KEYS = ("schema", "representation_id", "conditioning_mode", "activation",
             "feature_supervision", "extractor", "rows")
PLAN_ROW_KEYS = ("candidate_id", "heavy", "light", "group_id", "split", "condition")
HEADER_KEYS = ("schema", "representation_id", "conditioning_mode", "activation", "rows",
               "feature_space", "extractor", "runtime", "storage", "token_map_sha256",
               "cache_sha256")
HEADER_ROW_KEYS = ("ids", "heavy", "light", "groups", "splits", "identities", "conditions",
                   "condition_indices", "token_maps")
ACTIVATION_RECORD_KEYS = ("kind", "name", "location", "layout", "position_axis", "stored_dtype")
TOKEN_ROW_KEYS = ("index", "role", "chain", "residue_index", "native_id", "token", "residue")

ROLE_START, ROLE_END = hooks_lib.ROLE_START, hooks_lib.ROLE_END
ROLE_SEPARATOR, ROLE_RESIDUE = hooks_lib.ROLE_SEPARATOR, hooks_lib.ROLE_RESIDUE
ROLES = (ROLE_START, ROLE_END, ROLE_SEPARATOR, ROLE_RESIDUE)
_ROLE_CODE = {ROLE_START: "S", ROLE_END: "E", ROLE_SEPARATOR: "P"}
_CODE_ROLE = {code: role for role, code in _ROLE_CODE.items()}
_LITTLE_ENDIAN = {STORED_DTYPE: "<f4", OFFSET_DTYPE: "<i8"}

_TAP = re.compile(r"route/" + HOOK_ROUTE + r"/block(0|[1-9][0-9]*)/("
                  + "|".join(hooks_lib.SITES) + r")")

CLAIMS = {
    "scope": ("A native token-feature export and an exact pooled bridge into the frozen-feature "
              "matrix. It is not a weights loader, not a statement of biological improvement, and "
              "it selects no best activation, layer, pooling or readout."),
    "qualification": ("A tiny randomly initialised CPU model qualifies the machinery. It is not a "
                      "pretrained model and says nothing about biology or about any production "
                      "weights."),
    "provenance": ("The checkpoint and source hashes in a plan are caller assertions. The runtime "
                   "digest describes the instance that actually ran; it is not a checkpoint file "
                   "hash and is never labelled as one. Hashes are integrity, not authenticity."),
    "numerics": ("Batch size, batch order and right-padding amount can change float32 rounding; "
                 "parity is asserted within a float32 tolerance, never as bit identity."),
    "layers": ("A scientific baseline should use source-development-selected or pre-declared "
               "layers, not the best-looking test layer. This module selects none."),
}


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------

def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _exact_keys(document, required, *, label):
    return fm._keys(document, required, label=label)


def _plain_text(value, name):
    require(isinstance(value, str) and value and value == value.strip(),
            f"{name} must be a non-empty string without surrounding whitespace, got {value!r}")
    return value


def _chain(value, name, *, allow_empty):
    require(isinstance(value, str), f"{name} must be a string, got {value!r}")
    require(allow_empty or value, f"{name} must not be empty")
    bad = sorted({character for character in value if character not in CANONICAL})
    require(not bad,
            f"{name} has non-canonical residue(s) {bad}: chains are complete upper-case sequences "
            f"over {CANONICAL!r}; a gap, an X, a lower-case letter or whitespace is a different "
            "sequence and is refused, never sanitised")
    return value


def identity_of(heavy, light):
    """The full physical identity: ``heavy + '|' + light``. A heavy-only record keeps its trailing
    delimiter, so the chain boundary is part of identity (``AC|D`` is not ``A|CD``)."""
    return f"{heavy}{DELIMITER}{light}"


def _check_roster_lists(ids, identities, groups, splits):
    require(len(set(ids)) == len(ids), "a candidate id is repeated")
    first = {}
    for ident, identity in zip(ids, identities):
        require(identity not in first,
                f"rows {first.get(identity)!r} and {ident!r} are the same full physical identity "
                f"{identity!r}")
        first[identity] = ident
    for ident, split in zip(ids, splits):
        require(split in (fb.SPLIT_TRAIN, fb.SPLIT_EVAL),
                f"the split of {ident!r} must be {fb.SPLIT_TRAIN!r} or {fb.SPLIT_EVAL!r}, got "
                f"{split!r}")
    sides = {}
    for group, split in zip(groups, splits):
        sides.setdefault(group, set()).add(split)
    crossing = sorted(group for group, kinds in sides.items() if len(kinds) > 1)
    require(not crossing,
            f"biological group(s) {crossing[:3]} appear in both the training and the evaluation "
            "split; the grouping rationale is violated")


# ---------------------------------------------------------------------------
# the export plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanRow:
    candidate_id: str
    heavy: str
    light: str
    group_id: str
    split: str
    condition: object

    @property
    def identity(self):
        return identity_of(self.heavy, self.light)

    def document(self):
        return {"candidate_id": self.candidate_id, "heavy": self.heavy, "light": self.light,
                "group_id": self.group_id, "split": self.split,
                "condition": copy.deepcopy(self.condition)}


@dataclass(frozen=True, eq=False)
class ExportPlan:
    """A parsed plan. ``rows`` are in canonical (sorted candidate id) order; ``path`` is the file it
    was read from (``None`` for a plan parsed from a document), kept so the export can refuse an
    output that aliases it. The dataclass is frozen but its nested containers (a row's condition,
    ``activation``, ``extractor``, the supervision entries) are plain mutable objects, so
    ``export_dplm_tokens`` re-validates a snapshot of the plan (:func:`snapshot_export_plan`) instead
    of trusting that it is still the plan that was parsed."""

    representation_id: str
    conditioning_mode: str
    activation: dict
    supervision: object
    extractor: dict
    rows: tuple
    path: object = None

    def document(self):
        return {"schema": PLAN_SCHEMA, "representation_id": self.representation_id,
                "conditioning_mode": self.conditioning_mode,
                "activation": dict(self.activation),
                "feature_supervision": self.supervision.document(),
                "extractor": dict(self.extractor),
                "rows": [row.document() for row in self.rows]}

    def fingerprint(self):
        return fb.digest(self.document())


def _parse_activation_choice(document):
    _exact_keys(document, ("kind", "name"), label="plan activation")
    kind, name = document["kind"], document["name"]
    require(kind in ACTIVATION_KINDS,
            f"plan activation.kind must be one of {ACTIVATION_KINDS}, got {kind!r}")
    if kind == ACTIVATION_FINAL:
        require(name is None,
                "activation kind 'final_hidden' names no tap: name must be null (the final hidden "
                "state is the forward result AFTER the final LayerNorm, not a hook tap)")
    else:
        require(isinstance(name, str) and _TAP.fullmatch(name) is not None,
                f"activation kind 'hook' needs the exact name of a registered tap such as "
                f"'route/{HOOK_ROUTE}/block1/resid_post' (sites {hooks_lib.SITES}), got {name!r}")
    return {"kind": kind, "name": name}


def _parse_plan_extractor(document):
    extractor = fm._parse_extractor(document)
    require(extractor["kind"] in PLAN_EXTRACTOR_KINDS,
            f"a native export extractor.kind must be one of {PLAN_EXTRACTOR_KINDS}, got "
            f"{extractor['kind']!r}")
    if extractor["kind"] == "synthetic":
        require(extractor["checkpoint_sha256"] is None,
                "a synthetic (tiny random) extractor has no checkpoint file: checkpoint_sha256 "
                "must be null. A parameter-state hash is never labelled as a checkpoint hash")
    return extractor


def _parse_condition(value, mode, label):
    if mode == CONDITIONING_NONE:
        require(value is None,
                f"{label}: conditioning_mode 'none' passes NO condition vector, so condition must "
                "be null (zero indices would be a real first-family/low-bin request, not 'none')")
        return None
    require(isinstance(value, dict),
            f"{label}: conditioning_mode 'explicit' needs a condition object "
            "{family, assay, quality} on every row")
    _exact_keys(value, ("family", "assay", "quality"), label=f"{label} condition")
    family = value["family"]
    require(family in conditions_lib.FAMILIES,
            f"{label}: condition family must be one of {conditions_lib.FAMILIES}, got {family!r}")
    assay = value["assay"]
    require(assay is None or (isinstance(assay, str) and assay and assay == assay.strip()
                              and assay != conditions_lib.ASSAY_NULL),
            f"{label}: condition assay must be an enrolled assay name or JSON null (the trained "
            f"dropped-assay slot); the string {conditions_lib.ASSAY_NULL!r} is reserved and "
            f"ambiguous, got {assay!r}")
    require(value["quality"] is None,
            f"{label}: condition quality must be an explicit null (present, not omitted): an "
            "outcome-derived quality bin is not an answer-free prediction feature")
    return {"family": family, "assay": assay, "quality": None}


def parse_export_plan(document, *, path=None):
    """Validate a plan mapping strictly. Exact keys, canonical chains, no duplicate id or full
    identity, no group on both sides of the split; rows come back sorted by candidate id."""
    document = copy.deepcopy(document)
    _exact_keys(document, PLAN_KEYS, label="native export plan")
    require(document["schema"] == PLAN_SCHEMA, f"not a {PLAN_SCHEMA} plan")
    representation_id = fm.safe_id(document["representation_id"], "plan representation_id")
    mode = document["conditioning_mode"]
    require(mode in CONDITIONING_MODES,
            f"plan conditioning_mode must be one of {CONDITIONING_MODES}, got {mode!r}")
    activation = _parse_activation_choice(document["activation"])
    supervision = fb.SupervisionHistory.from_document(document["feature_supervision"],
                                                      label="plan feature_supervision")
    extractor = _parse_plan_extractor(document["extractor"])
    raw_rows = document["rows"]
    require(isinstance(raw_rows, list) and raw_rows, "plan rows must be a non-empty list")
    rows = []
    for position, raw in enumerate(raw_rows):
        label = f"plan row {position}"
        _exact_keys(raw, PLAN_ROW_KEYS, label=label)
        split = raw["split"]
        require(split in (fb.SPLIT_TRAIN, fb.SPLIT_EVAL),
                f"{label}: split must be {fb.SPLIT_TRAIN!r} or {fb.SPLIT_EVAL!r}, got {split!r}")
        rows.append(PlanRow(
            candidate_id=_plain_text(raw["candidate_id"], f"{label} candidate_id"),
            heavy=_chain(raw["heavy"], f"{label} heavy", allow_empty=False),
            light=_chain(raw["light"], f"{label} light", allow_empty=True),
            group_id=_plain_text(raw["group_id"], f"{label} group_id"), split=split,
            condition=_parse_condition(raw["condition"], mode, label)))
    _check_roster_lists([row.candidate_id for row in rows], [row.identity for row in rows],
                        [row.group_id for row in rows], [row.split for row in rows])
    rows.sort(key=lambda row: row.candidate_id)
    return ExportPlan(representation_id=representation_id, conditioning_mode=mode,
                      activation=activation, supervision=supervision, extractor=extractor,
                      rows=tuple(rows), path=None if path is None else Path(path))


def read_export_plan(path):
    """Strict JSON (duplicate keys, NaN and unknown keys refused); the parsed plan keeps ``path``."""
    return parse_export_plan(fb.read_json(path), path=path)


def snapshot_export_plan(plan):
    """A validated, DETACHED copy of ``plan`` exactly as it is now, or a ``ValueError``.

    A parsed plan is only a snapshot of the document it came from: its nested conditions,
    activation, extractor and supervision entries are mutable, and the constructor can be called
    with anything. The plan is therefore serialised and parsed again (the same strict parser, so an
    invalid condition, an unknown key or an edited supervision entry is refused, never normalised
    away), canonicalised and detached from the caller's objects. The retained ``path`` is carried
    over unchanged so alias protection still sees the original file. Nothing of the caller's is
    modified, whether the plan is accepted or refused."""
    require(isinstance(plan, ExportPlan), "export_dplm_tokens needs an ExportPlan")
    require(plan.path is None or isinstance(plan.path, (str, os.PathLike)),
            f"plan.path must be None or a path, got {type(plan.path).__name__}")
    require(isinstance(plan.activation, dict) and isinstance(plan.extractor, dict),
            "plan.activation and plan.extractor must be mappings")
    require(isinstance(plan.supervision, fb.SupervisionHistory),
            "plan.supervision must be a SupervisionHistory")
    require(isinstance(plan.rows, (tuple, list)) and all(isinstance(row, PlanRow)
                                                        for row in plan.rows),
            "plan.rows must hold PlanRow objects")
    try:
        return parse_export_plan(plan.document(), path=plan.path)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(f"the export plan is not coherent ({type(error).__name__}: {error}); it "
                         "was built or edited into a state the parser cannot read") from error


# ---------------------------------------------------------------------------
# native tokens and token maps
# ---------------------------------------------------------------------------

def _construct(heavy, light, record_id=""):
    return serial.Construct(heavy=heavy, light=light, sites=(), source="functional_native_export",
                            record_id=record_id)


def native_length(heavy, light):
    """``<cls> H <eos> <eos> [<cls> L <eos>]``: ``len(H) + 3`` heavy-only, else ``len(H) + len(L) + 5``."""
    return len(heavy) + 3 if not light else len(heavy) + len(light) + 5


def native_token_ids(heavy, light):
    """Direct native ids of the UNMASKED construct: ``dplm_token_ids`` with no masked site."""
    return tuple(int(item) for item in dplm.dplm_token_ids(_construct(heavy, light),
                                                           masked_sites=()))


def independent_native_ids(heavy, light):
    """The same ids through the PUBLIC route: the AbLang serialisation with nothing masked and no
    query (its default masks the edit sites, so it is not used), then the boundary translation. Used
    only to check the direct construction; the export never translates native ids again."""
    serialization = serial.ablang_serialization(_construct(heavy, light), masked_sites=(),
                                                order=())
    return tuple(int(item) for item in dplm.ablang_to_dplm(list(serialization.tokens)).tolist())


def _expected_encoding(heavy, light):
    """The compact per-row token map, built by string arithmetic from the vocabulary table alone."""
    roles = "S" + "H" * len(heavy) + "EP"
    ids = ([dplm.DPLM_CLS_ID] + [dplm.DPLM_RESIDUE_IDS[residue] for residue in heavy]
           + [dplm.DPLM_EOS_ID, dplm.DPLM_EOS_ID])
    if light:
        roles += "S" + "L" * len(light) + "E"
        ids += ([dplm.DPLM_CLS_ID] + [dplm.DPLM_RESIDUE_IDS[residue] for residue in light]
                + [dplm.DPLM_EOS_ID])
    return {"roles": roles, "native_ids": ids}


def native_token_map(heavy, light):
    """The complete per-token map of one construct, specials included.

    Residue rows carry ``chain`` (``H``/``L``), the zero-based ``residue_index`` inside that chain,
    the residue letter and its native id. Special rows (``start`` = ``<cls>``, ``end`` and
    ``separator`` = ``<eos>``) carry a role and a native id only. Heavy residue ``i`` sits at
    ``1 + i``; light residue ``j`` at ``len(H) + 4 + j``.
    """
    ids = native_token_ids(heavy, light)
    rows = []

    def add(role, chain=None, residue_index=None):
        index = len(rows)
        native_id = ids[index]
        token = dplm.DPLM_VOCAB[native_id]
        rows.append({"index": index, "role": role, "chain": chain,
                     "residue_index": residue_index, "native_id": native_id, "token": token,
                     "residue": token if role == ROLE_RESIDUE else None})
    add(ROLE_START)
    for i in range(len(heavy)):
        add(ROLE_RESIDUE, serial.HEAVY, i)
    add(ROLE_END)
    add(ROLE_SEPARATOR)
    if light:
        add(ROLE_START)
        for j in range(len(light)):
            add(ROLE_RESIDUE, serial.LIGHT, j)
        add(ROLE_END)
    require(len(rows) == len(ids), "internal error: the token map and the native ids disagree")
    return rows


def validate_token_map(rows, heavy, light):
    """Refuse any token map that is not exactly the native layout of ``heavy``/``light``.

    Checked one fact at a time, so the refusal names the defect: every index is the row position;
    the role is known; the native id is an integer (not a bool) inside the vocabulary and agrees
    with the token; a residue row names a known chain, a non-negative integer (not bool)
    ``residue_index`` inside that chain, no (chain, index) twice, a letter equal to the chain's
    residue there and the native id of that letter; a special row names no chain and carries
    ``<cls>`` (start) or ``<eos>`` (end/separator); every residue of both chains appears; the length
    is the layout's. Finally the whole list must equal :func:`native_token_map`.
    """
    require(isinstance(rows, (list, tuple)), "a token map must be a list of rows")
    chains = {serial.HEAVY: heavy, serial.LIGHT: light}
    expected = native_length(heavy, light)
    require(len(rows) == expected,
            f"the token map has {len(rows)} row(s) for a {expected}-token layout: a truncated or "
            "padded map is refused (nothing is trimmed or extended)")
    seen = set()
    for position, row in enumerate(rows):
        label = f"token map row {position}"
        _exact_keys(row, TOKEN_ROW_KEYS, label=label)
        require(_is_int(row["index"]) and row["index"] == position,
                f"{label}: index {row['index']!r} is not its position")
        role = row["role"]
        require(role in ROLES, f"{label}: unknown role {role!r}")
        native_id = row["native_id"]
        require(_is_int(native_id) and 0 <= native_id < dplm.DPLM_VOCAB_SIZE,
                f"{label}: native id {native_id!r} is not an integer inside the vocabulary")
        require(row["token"] == dplm.DPLM_VOCAB[native_id],
                f"{label}: token {row['token']!r} is not vocabulary entry {native_id}")
        if role == ROLE_RESIDUE:
            chain, index = row["chain"], row["residue_index"]
            require(isinstance(chain, str) and chain in chains,
                    f"{label}: unknown chain {chain!r}")
            require(_is_int(index) and index >= 0,
                    f"{label}: residue_index {index!r} is not a non-negative integer")
            require(index < len(chains[chain]),
                    f"{label}: residue_index {index} is outside the {len(chains[chain])}-residue "
                    f"{chain} chain")
            require((chain, int(index)) not in seen,
                    f"{label}: residue ({chain}, {index}) appears twice")
            seen.add((chain, int(index)))
            letter = chains[chain][index]
            require(row["residue"] == letter,
                    f"{label}: residue {row['residue']!r} is not {letter!r}, the {chain} chain's "
                    f"residue {index}")
            require(native_id == dplm.DPLM_RESIDUE_IDS[letter],
                    f"{label}: native id {native_id} is not the id of residue {letter!r}")
        else:
            require(row["chain"] is None and row["residue_index"] is None
                    and row["residue"] is None,
                    f"{label}: a {role} token names no chain, residue index or residue")
            want = dplm.DPLM_CLS_ID if role == ROLE_START else dplm.DPLM_EOS_ID
            require(native_id == want,
                    f"{label}: a {role} token must carry native id {want}, got {native_id}")
    complete = ({(serial.HEAVY, i) for i in range(len(heavy))}
                | {(serial.LIGHT, j) for j in range(len(light))})
    require(seen == complete,
            f"the token map is incomplete: {len(complete - seen)} residue(s) are missing")
    require(list(rows) == native_token_map(heavy, light),
            "the token map is not the native <cls> H <eos> <eos> [<cls> L <eos>] layout")
    return True


def encode_token_map(rows):
    """The compact, lossless form stored in a cache header: one role character per token
    (``S`` start, ``H``/``L`` a heavy/light residue, ``E`` end, ``P`` separator) and the native ids."""
    return {"roles": "".join(row["chain"] if row["role"] == ROLE_RESIDUE
                             else _ROLE_CODE[row["role"]] for row in rows),
            "native_ids": [int(row["native_id"]) for row in rows]}


def decode_token_map(encoded, heavy, light):
    """Expand and VALIDATE a stored encoding (see :func:`validate_token_map`)."""
    _exact_keys(encoded, ("roles", "native_ids"), label="encoded token map")
    roles, ids = encoded["roles"], encoded["native_ids"]
    require(isinstance(roles, str) and isinstance(ids, list) and len(roles) == len(ids),
            "an encoded token map needs one native id per role character")
    rows, counters = [], {serial.HEAVY: 0, serial.LIGHT: 0}
    for index, (code, native_id) in enumerate(zip(roles, ids)):
        require(_is_int(native_id) and 0 <= native_id < dplm.DPLM_VOCAB_SIZE,
                f"encoded token {index}: native id {native_id!r} is not inside the vocabulary")
        token = dplm.DPLM_VOCAB[native_id]
        if code in counters:
            rows.append({"index": index, "role": ROLE_RESIDUE, "chain": code,
                         "residue_index": counters[code], "native_id": int(native_id),
                         "token": token, "residue": token})
            counters[code] += 1
        else:
            require(code in _CODE_ROLE, f"encoded token {index}: unknown role code {code!r}")
            rows.append({"index": index, "role": _CODE_ROLE[code], "chain": None,
                         "residue_index": None, "native_id": int(native_id), "token": token,
                         "residue": None})
    validate_token_map(rows, heavy, light)
    return rows


# ---------------------------------------------------------------------------
# model support, hooks, the state inventory and the runtime fingerprint
# ---------------------------------------------------------------------------

_SUPPORTED_MODULES = frozenset({
    dplm.ConditionedDPLM, dplm.DPLMEncoder, dplm._Esm, dplm._Embeddings, dplm._Encoder,
    dplm._Layer, dplm._Attention, dplm._SelfAttention, dplm._SelfOutput, dplm._Intermediate,
    dplm._Output, dplm._LMHead, dplm.DPLMLinear, dplm.DPLMLayerNorm, nn.ModuleList, nn.Embedding,
    nn.Dropout})

_LOCAL_HOOK_FIELDS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                      "_backward_pre_hooks")
_GLOBAL_HOOK_FIELDS = ("_global_forward_hooks", "_global_forward_pre_hooks",
                       "_global_backward_hooks", "_global_backward_pre_hooks")
_REFUSED_LOCAL = ("_forward_hooks", "_forward_pre_hooks")
_REFUSED_GLOBAL = ("_global_forward_hooks", "_global_forward_pre_hooks")


def require_supported_model(model):
    """Only an unmodified ``ConditionedDPLM`` over the native encoder: a subclass, a foreign module
    type or an instance-level callable (a patched ``forward``) can change what a forward computes
    in ways no fingerprint of tensors would show, so each is refused."""
    require(type(model) is dplm.ConditionedDPLM,
            f"only an unmodified ConditionedDPLM is supported, got {type(model).__qualname__}")
    require(type(model.native) is dplm.DPLMEncoder,
            f"the native trunk must be a DPLMEncoder, got {type(model.native).__qualname__}")
    require(isinstance(model.condition_spec, conditions_lib.ConditionSpec),
            "the model's condition_spec is not a ConditionSpec")
    require(isinstance(model.config, dplm.DPLMConfig) and isinstance(model.native.config,
                                                                    dplm.DPLMConfig),
            "the model's config is not a DPLMConfig")
    for name, module in model.named_modules(remove_duplicate=False):
        label = name or "<root>"
        require(type(module) in _SUPPORTED_MODULES,
                f"module {label} is a {type(module).__qualname__}, which is not a supported native "
                "DPLM module type; a custom module is refused rather than fingerprinted blindly")
        patched = sorted(key for key, value in vars(module).items() if callable(value))
        require(not patched,
                f"module {label} carries instance-level callable(s) {patched}: a patched forward "
                "or method would change the computation behind the fingerprint")


def hook_inventory(model):
    """Per-module and global hook counts (forward, forward-pre, backward, backward-pre)."""
    local = {name: {field: len(getattr(module, field, ())) for field in _LOCAL_HOOK_FIELDS}
             for name, module in model.named_modules(remove_duplicate=False)}
    registered = {}
    for field in _GLOBAL_HOOK_FIELDS:
        registry = getattr(torch_module_lib, field, None)
        require(registry is not None,
                f"this torch has no {field}; global hooks cannot be inspected, so the export "
                "cannot prove none is active")
        registered[field] = len(registry)
    return {"local": local, "global": registered}


def refuse_existing_forward_hooks(model):
    """Refuse any forward or forward-pre hook that is already installed, on a module OR globally.

    The exporter installs its own capture-only hook; a pre-existing one could rewrite activations
    or inputs, and the runtime fingerprint cannot describe a closure. Refusing is cheaper and safer
    than reproducing arbitrary hook semantics."""
    inventory = hook_inventory(model)
    for name, counts in inventory["local"].items():
        for field in _REFUSED_LOCAL:
            require(counts[field] == 0,
                    f"module {name or '<root>'} already has {counts[field]} {field.strip('_')}: a "
                    "pre-existing forward/pre-forward hook can change what the export captures. "
                    "Remove it (the exporter will not) and export again")
    for field in _REFUSED_GLOBAL:
        require(inventory["global"][field] == 0,
                f"{inventory['global'][field]} global {field.replace('_global_', '')} "
                "(torch.nn.modules.module.register_module_*) is active; it applies to every "
                "module, so the export refuses to run under it")
    return inventory


def _tensor_sha256(tensor):
    """Digest of the dtype, shape and the RAW bytes, so bfloat16 (no NumPy dtype) is covered."""
    detached = tensor.detach()
    if detached.device.type != "cpu":
        detached = detached.to("cpu")
    flat = detached.contiguous().reshape(-1)
    head = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii")
    if flat.numel() == 0:
        return hashlib.sha256(head).hexdigest()
    return hashlib.sha256(head + flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def _simple(value, where):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Size):
        return [int(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _simple(dataclasses.asdict(value), where)
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{where} has a non-string key")
        return {key: _simple(item, f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_simple(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{where} holds a {type(value).__name__}, which the runtime fingerprint "
                     "cannot represent; unsupported custom model state is refused")


def _module_record(name, module):
    label = name or "<root>"
    attributes = {}
    for key in sorted(vars(module)):
        if key.startswith("_") or key == "training":
            continue
        attributes[key] = _simple(vars(module)[key], f"{label}.{key}")
    return {"type": f"{type(module).__module__}.{type(module).__qualname__}",
            "children": sorted(module._modules), "attributes": attributes}


def _tensor_entries(model):
    """Every named parameter and buffer (duplicates by NAME kept, so tied weights are visible),
    including the non-persistent buffers, plus the empty slots (``lora_A`` before an adapter)."""
    parameters, buffers, absent, owners = [], [], [], {}
    digests = {}

    def digest_of(tensor):
        key = id(tensor)
        if key not in digests:
            digests[key] = _tensor_sha256(tensor)
        return digests[key]
    for module_name, module in model.named_modules(remove_duplicate=False):
        for key, parameter in module._parameters.items():
            full = f"{module_name}.{key}" if module_name else key
            if parameter is None:
                absent.append(full)
                continue
            parameters.append({"name": full, "dtype": str(parameter.dtype),
                               "shape": [int(i) for i in parameter.shape],
                               "requires_grad": bool(parameter.requires_grad),
                               "sha256": digest_of(parameter)})
            owners.setdefault((parameter.data_ptr(), tuple(parameter.shape), str(parameter.dtype),
                               tuple(parameter.stride())), []).append(full)
        for key, buffer in module._buffers.items():
            full = f"{module_name}.{key}" if module_name else key
            if buffer is None:
                absent.append(full)
                continue
            buffers.append({"name": full, "persistent": key not in module._non_persistent_buffers_set,
                            "dtype": str(buffer.dtype), "shape": [int(i) for i in buffer.shape],
                            "sha256": digest_of(buffer)})
    tied = sorted(sorted(names) for names in owners.values() if len(names) > 1)
    return (sorted(parameters, key=lambda item: item["name"]),
            sorted(buffers, key=lambda item: item["name"]), sorted(absent), tied)


def _tokenizer_record():
    document = {
        "canonical_residues": CANONICAL, "dplm_vocab": list(dplm.DPLM_VOCAB),
        "dplm_vocab_size": dplm.DPLM_VOCAB_SIZE,
        "dplm_special_ids": {"cls": dplm.DPLM_CLS_ID, "pad": dplm.DPLM_PAD_ID,
                             "eos": dplm.DPLM_EOS_ID, "unk": dplm.DPLM_UNK_ID,
                             "x": dplm.DPLM_X_ID, "mask": dplm.DPLM_MASK_ID},
        "dplm_canonical_ids": list(dplm.DPLM_CANONICAL_IDS),
        "public_to_native": list(dplm.PUBLIC_TO_NATIVE),
        "native_to_canonical": list(dplm.NATIVE_TO_CANONICAL),
        "ablang": {"start": serial.ABLANG_START_ID, "pad": serial.ABLANG_PAD_ID,
                   "end": serial.ABLANG_END_ID, "mask": serial.ABLANG_MASK_ID,
                   "unknown": serial.ABLANG_UNKNOWN_ID, "separator": serial.ABLANG_SEP_ID,
                   "vocab_size": serial.ABLANG_VOCAB_SIZE,
                   "residue_ids": dict(sorted(serial.ABLANG_RESIDUE_IDS.items()))},
        "token_dropout_mask_ratio_train": dplm.TOKEN_DROPOUT_MASK_RATIO_TRAIN}
    return {"document": document, "sha256": fb.digest(document)}


def _source_digests():
    modules = {"dplm_policy": dplm.__file__, "her2_routes_hooks": hooks_lib.__file__,
               "her2_routes_serial": serial.__file__, "her2_routes_conditions": conditions_lib.__file__,
               "her2_routes_interp": interp.__file__, "functional_native_export": __file__}
    out = {}
    for label, path in sorted(modules.items()):
        require(path and Path(path).is_file(),
                f"the source of {label} cannot be read, so its digest would be a guess")
        out[label] = file_sha256(path)
    return out


def _read_setting(read):
    """One backend setting as plain JSON data: its value, ``UNSUPPORTED`` when this torch has no such
    flag, or ``"unreadable (<error type>)"`` when its getter raises. Never a guess. Deprecation
    chatter from a legacy getter is silenced rather than allowed to become an error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = read()
    except AttributeError:
        return UNSUPPORTED
    except (RuntimeError, TypeError, ValueError) as error:
        return f"unreadable ({type(error).__name__})"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return repr(value)


def _backend_readers():
    """Name -> zero-argument reader for every global setting that changes the native linear or
    attention arithmetic. Each reader resolves its attribute when CALLED, so a torch without the flag
    reports ``UNSUPPORTED`` instead of failing at import. All are pure context getters that touch no
    device (the tests and the selftest assert that CUDA stays uninitialised)."""
    cuda, cudnn, mkldnn = torch.backends.cuda, torch.backends.cudnn, torch.backends.mkldnn
    return {
        "float32_matmul_precision": lambda: torch.get_float32_matmul_precision(),
        "fp32_precision_generic": lambda: torch.backends.fp32_precision,
        "cuda_matmul_fp32_precision": lambda: cuda.matmul.fp32_precision,
        "cuda_matmul_allow_tf32": lambda: cuda.matmul.allow_tf32,
        "cuda_matmul_fp16_reduced_precision_reduction":
            lambda: cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cuda_matmul_bf16_reduced_precision_reduction":
            lambda: cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cuda_matmul_fp16_accumulation": lambda: cuda.matmul.allow_fp16_accumulation,
        "cudnn_enabled": lambda: cudnn.enabled,
        "cudnn_deterministic": lambda: cudnn.deterministic,
        "cudnn_benchmark": lambda: cudnn.benchmark,
        "cudnn_allow_tf32": lambda: cudnn.allow_tf32,
        "mkldnn_enabled": lambda: mkldnn.enabled,
        "mkldnn_deterministic": lambda: mkldnn.deterministic,
        "mkldnn_allow_tf32": lambda: mkldnn.allow_tf32,
        "mkldnn_fp32_precision": lambda: mkldnn.fp32_precision,
        "mkldnn_matmul_fp32_precision": lambda: mkldnn.matmul.fp32_precision,
        "sdp_math_enabled": lambda: cuda.math_sdp_enabled(),
        "sdp_flash_enabled": lambda: cuda.flash_sdp_enabled(),
        "sdp_mem_efficient_enabled": lambda: cuda.mem_efficient_sdp_enabled(),
        "sdp_cudnn_enabled": lambda: cuda.cudnn_sdp_enabled(),
        "sdp_math_fp16_bf16_reduction": lambda: cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "deterministic_algorithms": lambda: torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only":
            lambda: torch.is_deterministic_algorithms_warn_only_enabled(),
        "cpu_capability": lambda: torch.backends.cpu.get_cpu_capability()}


def backend_settings():
    """The current value of every numerical backend setting in :func:`_backend_readers` (see
    :func:`_read_setting` for the two explicit non-value markers). Read-only; touches no device.
    ``torch.set_flush_denormal`` has no getter, so that one setting cannot be recorded."""
    return {name: _read_setting(read) for name, read in sorted(_backend_readers().items())}


def _environment_record(model):
    devices = sorted({str(tensor.device) for tensor in itertools.chain(model.parameters(),
                                                                      model.buffers())})
    return {"torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
            "devices": devices, "cpu_threads": int(torch.get_num_threads()),
            "default_dtype": str(torch.get_default_dtype()),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "backend_settings": backend_settings(),
            "numpy_version": str(np.__version__), "byteorder": sys.byteorder,
            "python": sys.version.split()[0]}


def runtime_fingerprint(model):
    """``{"document", "sha256"}`` describing the ACTUAL model instance (see the module docstring).

    The digest changes with any parameter, buffer (non-persistent ones included), module attribute,
    adapter switch, config value, tokenizer table, source file, numerical backend setting
    (:func:`backend_settings`) or environment fact listed there. It does not depend on train/eval
    flags (those do not move an eval-mode forward) or on object identity: a deep copy has the same
    digest.
    """
    require_supported_model(model)
    parameters, buffers, absent, tied = _tensor_entries(model)
    lora = {"settings": None if model.lora_settings is None else model.lora_settings.document(),
            "names": [str(name) for name in model.lora_names],
            "modules": {name: {"enabled": bool(module.lora_enabled),
                               "scaling": float(module.lora_scaling),
                               "dropout_p": (None if module.lora_dropout is None
                                             else float(module.lora_dropout.p)),
                               "rank": int(module.lora_A.shape[0])}
                        for name, module in sorted(model.native.lora_modules().items())}}
    document = {
        "schema": RUNTIME_SCHEMA,
        "model": {
            "config": _simple(model.config, "config"),
            "native_config": _simple(model.native.config, "native_config"),
            "config_is_native_config": bool(model.config is model.native.config),
            "modules": {name or "<root>": _module_record(name, module)
                        for name, module in model.named_modules(remove_duplicate=False)},
            "parameters": parameters, "buffers": buffers, "absent_slots": absent,
            "tied_parameter_names": tied,
            "condition_spec": model.condition_spec.document(),
            "use_conditions": bool(model.use_conditions), "lora": lora,
            "dtype_policy": _simple(model.dtype_policy(), "dtype_policy"),
            "declared_identity": {"base_identity": _simple(model.base_identity, "base_identity"),
                                  "load_record": _simple(model.load_record, "load_record"),
                                  "note": "declared by whoever built the model; not verified here"},
            "provenance": {"diffusion": model.diffusion.document(),
                           "activation_checkpointing_flag": bool(model.activation_checkpointing)}},
        "tokenizer": _tokenizer_record(),
        "environment": _environment_record(model),
        "sources": _source_digests(),
        "execution": {"module_mode": "eval", "gradients": "disabled", "autocast": "disabled",
                      "activation_checkpointing_active": False,
                      "rng": "CPU and initialised-CUDA generators forked and restored"}}
    return {"document": document, "sha256": fb.digest(document)}


def state_inventory(model):
    """What an export must leave exactly as it found it: every module's training flag, each
    parameter's ``requires_grad`` and any pre-existing ``.grad`` (by content), the hook counts and
    the CPU and (if already initialised) CUDA generator states."""
    grads, requires = [], []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        requires.append((name, bool(parameter.requires_grad)))
        grads.append((name, None if parameter.grad is None else _tensor_sha256(parameter.grad)))
    cuda = None
    if torch.cuda.is_initialized():
        cuda = [hashlib.sha256(state.numpy().tobytes()).hexdigest()
                for state in torch.cuda.get_rng_state_all()]
    return {"training": [(name, bool(module.training))
                         for name, module in model.named_modules(remove_duplicate=False)],
            "requires_grad": requires, "grads": grads, "hooks": hook_inventory(model),
            "cpu_rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda_rng": cuda}


# ---------------------------------------------------------------------------
# paths, staging and publication
# ---------------------------------------------------------------------------

def _protected_mapping(paths):
    require(not isinstance(paths, (str, bytes, os.PathLike)),
            "protected_input_paths must be a list of paths, not a single path")
    mapping = {}
    for number, path in enumerate(paths):
        mapping[f"protected input #{number} ({path})"] = Path(path)
    return mapping


def _validate_cache_outputs(header_path, npz_path, inputs, overwrite):
    """Everything about the output paths that can be refused BEFORE any compute or write.

    The NPZ is named by a plain file name in the header's directory; the header, the NPZ and their
    two staging files must be four different PHYSICAL files, none of them an input the caller named
    (compared by resolved path, letter case where the file system folds it, and by identity for
    files that exist: hard links, junctions). Existing finals need ``overwrite``; a pre-existing
    staging file is refused even then, because it is not known to be ours.
    """
    fm._safe_npz_name(npz_path.name)
    same_directory = (os.path.normcase(os.path.realpath(header_path.parent))
                      == os.path.normcase(os.path.realpath(npz_path.parent)))
    require(same_directory,
            "the NPZ must be written in the header's directory: the header names it by a plain "
            "file name (a loader may be given another location explicitly)")
    outputs = {"cache header": header_path, "cache NPZ": npz_path,
               "cache header staging file": fb._temporary_path(header_path),
               "cache NPZ staging file": fb._temporary_path(npz_path)}
    for (left_label, left), (right_label, right) in itertools.combinations(outputs.items(), 2):
        require(not fb._same_file(left, right),
                f"the {left_label} {left} and the {right_label} {right} are the same physical file "
                "(a resolved alias, a name that differs only by letter case, or a hard link); "
                "writing would corrupt the cache. Nothing was written")
    fb._refuse_path_aliases(inputs, outputs)
    for label, target in (("cache header", header_path), ("cache NPZ", npz_path)):
        require(not target.is_dir(), f"the {label} {target} is a directory")
        require(overwrite or not os.path.lexists(target),
                f"the {label} {target} already exists; pass overwrite=True to replace it. Nothing "
                "was written")
    for label in ("cache header staging file", "cache NPZ staging file"):
        staging = outputs[label]
        require(not os.path.lexists(staging),
                f"{staging} already exists. It is not known to belong to this export (it may be "
                "another file linked under the staging name), so it is never truncated or deleted, "
                "even with overwrite=True. Nothing was written: inspect it, remove it yourself if "
                "it is stale, and run again")
    return outputs


def _publish(header_path, npz_path, header_text, npz_payload):
    """Stage BOTH files through names this call creates exclusively, then replace: NPZ first, header
    last (the header commits the cache). A failure removes only the staging files this call made; a
    final NPZ without its header, or a new NPZ under an old header, is refused by the loader."""
    header_path.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for final, data in ((npz_path, npz_payload), (header_path, header_text.encode("utf-8"))):
            staging = fb._temporary_path(final)
            try:
                handle = open(staging, "xb")
            except FileExistsError as error:
                raise ValueError(f"{staging} appeared while the cache was being written; it is "
                                 "not ours and was not touched") from error
            staged.append(staging)
            with handle:
                handle.write(data)
        fb._temporary_path(npz_path).replace(npz_path)
        fb._temporary_path(header_path).replace(header_path)
    except BaseException:
        for staging in staged:
            if os.path.lexists(staging):
                try:
                    staging.unlink()
                except OSError:
                    pass
        raise


# ---------------------------------------------------------------------------
# the token-cache document: hashing, sealing, parsing
# ---------------------------------------------------------------------------

def _content_sha256(array, dtype_name):
    """Digest of the dtype name, shape and little-endian bytes (the matrix cache's own format)."""
    little = np.ascontiguousarray(array, dtype=_LITTLE_ENDIAN[dtype_name])
    return hashlib.sha256(f"{dtype_name}|{little.shape}|".encode("ascii")
                          + little.tobytes()).hexdigest()


def _canonical_document(document, content_hashes):
    """What ``cache_sha256`` digests: the header without the cache digest and the NPZ FILE hash (the
    file's bytes carry zip timestamps), with each array's content hash recomputed from the arrays."""
    out = copy.deepcopy(document)
    out.pop("cache_sha256", None)
    storage = out["storage"]
    storage.pop("file_sha256", None)
    for name, entry in storage["arrays"].items():
        entry["content_sha256"] = content_hashes[name]
    return out


def _seal(document, features, offsets):
    """``(sealed header document, NPZ bytes)``: array facts, file hash and cache digest filled in."""
    features = np.ascontiguousarray(features, dtype=_LITTLE_ENDIAN[STORED_DTYPE])
    offsets = np.ascontiguousarray(offsets, dtype=_LITTLE_ENDIAN[OFFSET_DTYPE])
    hashes = {FEATURES_ARRAY: _content_sha256(features, STORED_DTYPE),
              OFFSETS_ARRAY: _content_sha256(offsets, OFFSET_DTYPE)}
    sealed = copy.deepcopy(document)
    sealed["storage"]["arrays"] = {
        FEATURES_ARRAY: {"dtype": STORED_DTYPE, "shape": [int(i) for i in features.shape],
                         "content_sha256": hashes[FEATURES_ARRAY]},
        OFFSETS_ARRAY: {"dtype": OFFSET_DTYPE, "shape": [int(i) for i in offsets.shape],
                        "content_sha256": hashes[OFFSETS_ARRAY]}}
    buffer = io.BytesIO()
    np.savez(buffer, **{FEATURES_ARRAY: features, OFFSETS_ARRAY: offsets})
    payload = buffer.getvalue()
    sealed["storage"]["file_sha256"] = hashlib.sha256(payload).hexdigest()
    sealed["cache_sha256"] = fb.digest(_canonical_document(sealed, hashes))
    return sealed, payload


def _activation_location(kind, name):
    if kind == ACTIVATION_FINAL:
        return ("final_hidden: the ConditionedDPLM native forward's last_hidden_state, taken AFTER "
                "emb_layer_norm_after (it is not the last layer's resid_post), tokens_first "
                "(batch, tokens, hidden)")
    return (f"hook {name}: capture-only HookRegistry tap on the native DPLM layer stack, "
            "tokens_first (batch, tokens, hidden), position axis 1; no patch or ablation")


def _activation_record(choice):
    return {"kind": choice["kind"], "name": choice["name"],
            "location": _activation_location(choice["kind"], choice["name"]),
            "layout": hooks_lib.LAYOUT_TOKENS_FIRST, "position_axis": 1,
            "stored_dtype": STORED_DTYPE}


def _parse_header_activation(document):
    _exact_keys(document, ACTIVATION_RECORD_KEYS, label="token cache activation")
    choice = _parse_activation_choice({"kind": document["kind"], "name": document["name"]})
    require(document == _activation_record(choice),
            "the cache's activation record (location, layout, position axis, stored dtype) is not "
            "the one this activation implies")
    return choice


def _token_feature_space(supervision, hidden, choice, mode, runtime_sha256, token_map_sha256):
    names = tuple(f"h{index}" for index in range(hidden))
    return fb.FeatureSpace(
        kind=TOKEN_SPACE_KIND, names=names,
        schema={"layout": "ragged native tokens: features[offsets[i]:offsets[i + 1]] is row i",
                "hidden_size": int(hidden), "stored_dtype": STORED_DTYPE,
                "activation": {"kind": choice["kind"], "name": choice["name"]},
                "conditioning_mode": mode, "runtime_sha256": runtime_sha256,
                "token_map_sha256": token_map_sha256, "candidate_masking": "none"},
        history=("native DPLM forward over the UNMASKED complete H|L candidate, eval/no_grad, "
                 "autocast off, right-padded batches whose padded activations are discarded",
                 "per-token activations of the real native tokens (specials included) are stored "
                 "raggedly; no batch padding row exists"),
        supervision=supervision)


def _dig(document, path, label):
    current = document
    for key in path:
        require(isinstance(current, dict) and key in current,
                f"{label} is malformed: missing {'.'.join(path)}")
        current = current[key]
    return current


def _parse_row_conditions(mode, logical, indices, spec, count):
    require(len(logical) == count and len(indices) == count,
            "the cache's condition lists do not cover every row")
    if mode == CONDITIONING_NONE:
        require(all(item is None for item in logical) and all(item is None for item in indices),
                "conditioning_mode 'none' passes no condition vector, so every condition and "
                "condition index must be null")
        return
    families, bins = spec["families"], spec["bins"]
    assays = spec["assays"]
    for position, (condition, triple) in enumerate(zip(logical, indices)):
        label = f"cache row {position} condition"
        parsed = _parse_condition(condition, CONDITIONING_EXPLICIT, label)
        require(isinstance(triple, list) and len(triple) == 3 and all(_is_int(i) for i in triple),
                f"{label} indices must be three integers")
        want = [families.index(parsed["family"]), bins.index(conditions_lib.BIN_NULL),
                assays.index(conditions_lib.ASSAY_NULL if parsed["assay"] is None
                             else parsed["assay"])] \
            if parsed["family"] in families and (parsed["assay"] is None
                                                  or parsed["assay"] in assays) else None
        require(want is not None,
                f"{label} names a family or assay outside the recorded condition_spec")
        require(triple == want,
                f"{label} indices {triple} are not the recorded condition_spec's encoding {want}")


def _positive_int(value, label):
    require(_is_int(value) and value >= 1,
            f"{label} must be a positive integer (not a bool), got {value!r}")
    return int(value)


def _check_runtime_contract(runtime_document, *, hidden, choice, mode, lengths):
    """Refuse a header whose RECORDED runtime contradicts the rest of the same header.

    A resealed hash chain proves nothing about authenticity, but it cannot make a cache agree with
    itself: the feature width must be the hidden size of BOTH recorded configs (the wrapper's and the
    native trunk's), a hook's block must exist in the recorded layer count (``final_hidden`` is not
    a block tap and has no such constraint), every row's native token count must fit the recorded
    positions, and explicit conditioning needs a runtime that recorded ``use_conditions`` as true.
    Only the recorded runtime is read; the current host and model are never consulted."""
    longest = max(lengths)
    for key in ("config", "native_config"):
        config = _dig(runtime_document, ("model", key), "the runtime document")
        label = f"runtime model.{key}"
        require(isinstance(config, dict), f"the recorded {label} must be an object")
        size = _positive_int(config.get("hidden_size"), f"{label}.hidden_size")
        layers = _positive_int(config.get("num_hidden_layers"), f"{label}.num_hidden_layers")
        positions = _positive_int(config.get("max_position_embeddings"),
                                  f"{label}.max_position_embeddings")
        require(size == hidden,
                f"the cache stores {hidden}-wide features but its recorded {label}.hidden_size is "
                f"{size}")
        require(longest <= positions,
                f"a row has {longest} native tokens but the recorded {label} has only {positions} "
                "positions: the model could not have produced it")
        if choice["kind"] == ACTIVATION_HOOK:
            block = int(_TAP.fullmatch(choice["name"]).group(1))
            require(block < layers,
                    f"the activation {choice['name']!r} taps block {block} but the recorded "
                    f"{label} has {layers} layer(s)")
    if mode == CONDITIONING_EXPLICIT:
        enabled = _dig(runtime_document, ("model", "use_conditions"), "the runtime document")
        require(enabled is True,
                "conditioning_mode 'explicit' needs a runtime that recorded use_conditions=true, "
                f"found {enabled!r}: the conditions could not have reached the forward")


def _check_offsets(offsets, total, lengths):
    require(offsets.ndim == 1 and offsets.shape[0] == len(lengths) + 1,
            f"offsets must hold {len(lengths) + 1} entries, found shape {offsets.shape}")
    require(int(offsets[0]) == 0, f"offsets must start at 0, found {int(offsets[0])}")
    require(bool(np.all(np.diff(offsets) > 0)), "offsets must be strictly increasing")
    require(int(offsets[-1]) == int(total),
            f"offsets end at {int(offsets[-1])} but the features hold {int(total)} token rows")
    require(np.array_equal(np.diff(offsets), np.asarray(lengths, dtype=np.int64)),
            "the offsets do not match the token counts the chains imply (heavy-only: len(H) + 3; "
            "paired: len(H) + len(L) + 5)")


def _parse_token_header(document):
    """Strictly validate a header and RECOMPUTE every semantic binding it can: identities, token
    maps, condition indices, the feature space, the runtime digest, the activation record. Returns
    the validated pieces; array contents are checked by the caller."""
    _exact_keys(document, HEADER_KEYS, label="token cache header")
    require(document["schema"] == TOKEN_CACHE_SCHEMA, f"not a {TOKEN_CACHE_SCHEMA} header")
    representation_id = fm.safe_id(document["representation_id"], "token cache representation_id")
    mode = document["conditioning_mode"]
    require(mode in CONDITIONING_MODES,
            f"token cache conditioning_mode must be one of {CONDITIONING_MODES}, got {mode!r}")
    choice = _parse_header_activation(document["activation"])
    runtime = _exact_keys(document["runtime"], ("sha256", "document"), label="token cache runtime")
    fm._hex64(runtime["sha256"], "token cache runtime.sha256")
    require(isinstance(runtime["document"], dict)
            and runtime["document"].get("schema") == RUNTIME_SCHEMA,
            f"the embedded runtime document is not a {RUNTIME_SCHEMA} document")
    require(fb.digest(runtime["document"]) == runtime["sha256"],
            "the embedded runtime document does not match its digest")
    spec = _dig(runtime["document"], ("model", "condition_spec"), "the runtime document")
    rows = _exact_keys(document["rows"], HEADER_ROW_KEYS, label="token cache rows")
    for name in HEADER_ROW_KEYS:
        require(isinstance(rows[name], list), f"token cache rows.{name} must be a list")
    ids = rows["ids"]
    count = len(ids)
    require(count >= 1, "a token cache needs at least one row")
    require(all(len(rows[name]) == count for name in HEADER_ROW_KEYS),
            "token cache rows are ragged: every row list must have one entry per id")
    for ident in ids:
        _plain_text(ident, "token cache candidate id")
    require(ids == sorted(ids) and len(set(ids)) == count,
            "token cache rows must be in strictly increasing candidate-id order")
    heavy = [_chain(item, "token cache heavy chain", allow_empty=False) for item in rows["heavy"]]
    light = [_chain(item, "token cache light chain", allow_empty=True) for item in rows["light"]]
    identities = [identity_of(h, l) for h, l in zip(heavy, light)]
    require(rows["identities"] == identities,
            "the cache's identities are not heavy|light of its own chains")
    for item in itertools.chain(rows["groups"]):
        _plain_text(item, "token cache group id")
    _check_roster_lists(ids, identities, rows["groups"], rows["splits"])
    _parse_row_conditions(mode, rows["conditions"], rows["condition_indices"], spec, count)
    lengths = []
    for position in range(count):
        encoded = rows["token_maps"][position]
        decode_token_map(encoded, heavy[position], light[position])
        require(encoded == _expected_encoding(heavy[position], light[position]),
                f"token map of row {ids[position]!r} is not the native layout of its chains")
        lengths.append(len(encoded["native_ids"]))
    token_map_sha256 = fm._hex64(document["token_map_sha256"], "token cache token_map_sha256")
    require(fb.digest(rows["token_maps"]) == token_map_sha256,
            "the token maps do not match token_map_sha256")
    storage = _exact_keys(document["storage"], ("file", "file_sha256", "arrays"),
                          label="token cache storage")
    fm._safe_npz_name(storage["file"])
    fm._hex64(storage["file_sha256"], "token cache storage.file_sha256")
    arrays = _exact_keys(storage["arrays"], (FEATURES_ARRAY, OFFSETS_ARRAY),
                         label="token cache storage.arrays")
    features = _exact_keys(arrays[FEATURES_ARRAY], ("dtype", "shape", "content_sha256"),
                           label="token cache features record")
    offsets = _exact_keys(arrays[OFFSETS_ARRAY], ("dtype", "shape", "content_sha256"),
                          label="token cache offsets record")
    require(features["dtype"] == STORED_DTYPE and offsets["dtype"] == OFFSET_DTYPE,
            f"the cache must store {STORED_DTYPE} features and {OFFSET_DTYPE} offsets")
    require(isinstance(features["shape"], list) and len(features["shape"]) == 2
            and all(_is_int(i) for i in features["shape"]),
            "token cache features.shape must be [token rows, hidden]")
    total, hidden = features["shape"]
    require(total == sum(lengths) and hidden >= 1,
            f"features.shape {features['shape']} disagrees with the {sum(lengths)} real tokens "
            "the chains imply")
    _check_runtime_contract(runtime["document"], hidden=hidden, choice=choice, mode=mode,
                            lengths=lengths)
    require(offsets["shape"] == [count + 1],
            f"offsets.shape {offsets['shape']} must be [{count + 1}]")
    fm._hex64(features["content_sha256"], "token cache features.content_sha256")
    fm._hex64(offsets["content_sha256"], "token cache offsets.content_sha256")
    space = fm._parse_space(document["feature_space"])
    expected_space = _token_feature_space(space.supervision, hidden, choice, mode,
                                          runtime["sha256"], token_map_sha256)
    require(space.document() == expected_space.document(),
            "the cache's feature space is not the one its activation, conditioning, runtime and "
            "token maps imply")
    extractor = _parse_plan_extractor(document["extractor"])
    fm._hex64(document["cache_sha256"], "token cache cache_sha256")
    return {"representation_id": representation_id, "mode": mode, "choice": choice, "ids": ids,
            "heavy": heavy, "light": light, "identities": identities, "lengths": lengths,
            "hidden": hidden, "total": total, "space": space, "extractor": extractor,
            "storage": storage, "cache_sha256": document["cache_sha256"],
            "runtime_sha256": runtime["sha256"], "token_map_sha256": token_map_sha256}


def _read_token_arrays(npz, parsed):
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
        require(sorted(members) == sorted([FEATURES_ARRAY, OFFSETS_ARRAY]) and len(members) == 2,
                f"{npz} must hold exactly the members {FEATURES_ARRAY!r} and {OFFSETS_ARRAY!r}, "
                f"found {list(members)}")
        loaded = {}
        for name in (FEATURES_ARRAY, OFFSETS_ARRAY):
            try:
                loaded[name] = archive[name]
            except (OSError, EOFError, zipfile.BadZipFile) as error:
                raise ValueError(f"{npz} member {name!r} cannot be read ({error})") from error
            except ValueError as error:
                raise ValueError(f"{npz} member {name!r} cannot be loaded without pickle "
                                 f"({error}); object arrays are refused") from error
    finally:
        close = getattr(archive, "close", None)
        if close is not None:
            close()
    features, offsets = loaded[FEATURES_ARRAY], loaded[OFFSETS_ARRAY]
    require(features.dtype == np.dtype(STORED_DTYPE) and features.ndim == 2
            and features.shape == (parsed["total"], parsed["hidden"]),
            f"{npz} features have dtype {features.dtype} shape {features.shape}; the header "
            f"declares {STORED_DTYPE} {(parsed['total'], parsed['hidden'])}")
    require(offsets.dtype == np.dtype(OFFSET_DTYPE),
            f"{npz} offsets have dtype {offsets.dtype}; the header declares {OFFSET_DTYPE}")
    require(bool(np.isfinite(features).all()), f"{npz} features contain NaN or infinity")
    _check_offsets(offsets, features.shape[0], parsed["lengths"])
    return (np.ascontiguousarray(features, dtype=_LITTLE_ENDIAN[STORED_DTYPE]),
            np.ascontiguousarray(offsets, dtype=_LITTLE_ENDIAN[OFFSET_DTYPE]))


def _frozen_copy(array):
    owned = np.array(array, copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, eq=False)
class TokenCache:
    """A verified ragged token cache, owned by the caller and frozen.

    ``features`` (sum(T_i) x hidden, float32) and ``offsets`` are private, read-only copies; row
    ``i``'s tokens are ``features[offsets[i]:offsets[i + 1]]``. Everything else is derived from the
    private validated header on each access (nothing handed out can edit it), the feature space and
    its supervision history included: :attr:`space` is a fresh object parsed from the hash-bound
    header every time, so editing what it returns changes only that detached copy. Because a caller
    can still re-enable writing on an array, :meth:`verify` re-derives every digest from the CURRENT
    arrays and header; ``pool_token_features`` and the pooled writer call it first.
    """

    header_path: Path
    npz_path: Path
    _document: dict
    features: object
    offsets: object
    file_sha256: str

    @property
    def space(self):
        """The cache's ``FeatureSpace`` (columns, schema, history and supervision), re-parsed from the
        verified header on every access. It is a detached copy: mutating its nested schema or
        supervision entries cannot reach the provenance the pooled bridge writes."""
        return fm._parse_space(copy.deepcopy(self._document["feature_space"]))

    @property
    def representation_id(self):
        return self._document["representation_id"]

    @property
    def conditioning_mode(self):
        return self._document["conditioning_mode"]

    @property
    def ids(self):
        return tuple(self._document["rows"]["ids"])

    @property
    def heavy(self):
        return tuple(self._document["rows"]["heavy"])

    @property
    def light(self):
        return tuple(self._document["rows"]["light"])

    @property
    def groups(self):
        return tuple(self._document["rows"]["groups"])

    @property
    def splits(self):
        return tuple(self._document["rows"]["splits"])

    @property
    def identities(self):
        return tuple(self._document["rows"]["identities"])

    @property
    def conditions(self):
        return tuple(copy.deepcopy(self._document["rows"]["conditions"]))

    @property
    def condition_indices(self):
        return tuple(copy.deepcopy(self._document["rows"]["condition_indices"]))

    @property
    def token_maps(self):
        return tuple(copy.deepcopy(self._document["rows"]["token_maps"]))

    @property
    def activation(self):
        return copy.deepcopy(self._document["activation"])

    @property
    def extractor(self):
        return copy.deepcopy(self._document["extractor"])

    @property
    def runtime_sha256(self):
        return self._document["runtime"]["sha256"]

    @property
    def runtime_document(self):
        return copy.deepcopy(self._document["runtime"]["document"])

    @property
    def token_map_sha256(self):
        return self._document["token_map_sha256"]

    @property
    def cache_sha256(self):
        return self._document["cache_sha256"]

    @property
    def hidden_size(self):
        return int(self.features.shape[1])

    def lengths(self):
        return [int(item) for item in np.diff(self.offsets)]

    def row_features(self, candidate_id):
        """A writable copy of one row's (T_i, hidden) token features."""
        position = self.ids.index(candidate_id)
        return np.array(self.features[self.offsets[position]:self.offsets[position + 1]])

    def token_map(self, candidate_id):
        position = self.ids.index(candidate_id)
        return decode_token_map(self._document["rows"]["token_maps"][position],
                                self.heavy[position], self.light[position])

    def snapshot(self):
        """Fresh read-only copies of the arrays, independent of this object's."""
        return _frozen_copy(self.features), _frozen_copy(self.offsets)

    def verify(self):
        """Re-validate the header and re-derive every digest from the arrays as they are NOW."""
        document = copy.deepcopy(self._document)
        parsed = _parse_token_header(document)
        features, offsets = self.features, self.offsets
        require(features.dtype == np.dtype(STORED_DTYPE) and features.shape
                == (parsed["total"], parsed["hidden"]),
                "the held features no longer have the declared dtype and shape")
        require(offsets.dtype == np.dtype(OFFSET_DTYPE), "the held offsets changed dtype")
        require(bool(np.isfinite(features).all()), "the held features contain NaN or infinity")
        _check_offsets(offsets, features.shape[0], parsed["lengths"])
        hashes = {FEATURES_ARRAY: _content_sha256(features, STORED_DTYPE),
                  OFFSETS_ARRAY: _content_sha256(offsets, OFFSET_DTYPE)}
        arrays = document["storage"]["arrays"]
        require(hashes[FEATURES_ARRAY] == arrays[FEATURES_ARRAY]["content_sha256"]
                and hashes[OFFSETS_ARRAY] == arrays[OFFSETS_ARRAY]["content_sha256"],
                "the held arrays no longer match the header's content hashes: they were edited "
                "after loading")
        require(fb.digest(_canonical_document(document, hashes)) == document["cache_sha256"],
                "the held cache no longer matches its canonical cache_sha256")
        return True

    def summary(self):
        lengths = self.lengths()
        return {"representation_id": self.representation_id,
                "conditioning_mode": self.conditioning_mode, "rows": len(self.ids),
                "hidden_size": self.hidden_size, "token_rows": int(self.features.shape[0]),
                "tokens_per_row": {"min": min(lengths), "max": max(lengths)},
                "heavy_only_rows": sum(1 for item in self.light if not item),
                "activation": self.activation, "extractor": self.extractor,
                "feature_supervision": self.space.supervision.document(),
                "runtime_sha256": self.runtime_sha256, "token_map_sha256": self.token_map_sha256,
                "cache_sha256": self.cache_sha256, "file_sha256": self.file_sha256,
                "header": str(self.header_path), "npz": str(self.npz_path)}


def load_token_cache(header_path, *, npz_path=None):
    """Load, verify and freeze a token cache. Every integrity layer is checked or it is refused: the
    strict header and every semantic binding it implies, the NPZ file digest (a replaced NPZ is
    STALE), the exact members, dtypes, finiteness, offsets against the chains, the content hashes
    and the canonical ``cache_sha256``. Never unpickles. Hashes are integrity, not authenticity."""
    header_path = Path(header_path)
    document = fb.read_json(header_path)
    parsed = _parse_token_header(document)
    npz = fm.resolve_npz_path(header_path, parsed["storage"]["file"], npz_path)
    actual = file_sha256(npz)
    require(actual == parsed["storage"]["file_sha256"],
            f"{npz} does not match the header's file_sha256: the cache is STALE (the NPZ was "
            "replaced or edited after the header was written)")
    features, offsets = _read_token_arrays(npz, parsed)
    hashes = {FEATURES_ARRAY: _content_sha256(features, STORED_DTYPE),
              OFFSETS_ARRAY: _content_sha256(offsets, OFFSET_DTYPE)}
    arrays = document["storage"]["arrays"]
    require(hashes[FEATURES_ARRAY] == arrays[FEATURES_ARRAY]["content_sha256"]
            and hashes[OFFSETS_ARRAY] == arrays[OFFSETS_ARRAY]["content_sha256"],
            f"{npz} does not match the header's content hashes (the arrays were edited)")
    require(fb.digest(_canonical_document(document, hashes)) == parsed["cache_sha256"],
            "the header's cache_sha256 does not match its own rows, contracts and arrays: the "
            "cache is STALE (the header or the arrays changed after it was written)")
    return TokenCache(header_path=header_path, npz_path=Path(npz),
                      _document=copy.deepcopy(document), features=_frozen_copy(features),
                      offsets=_frozen_copy(offsets), file_sha256=actual)


# ---------------------------------------------------------------------------
# the export itself
# ---------------------------------------------------------------------------

def _resolve_conditions(model, plan):
    count = len(plan.rows)
    if plan.conditioning_mode == CONDITIONING_NONE:
        return [None] * count, [None] * count
    require(bool(model.use_conditions),
            "explicit conditioning was requested but this model has conditioning disabled; "
            "silently ignoring it would report an unconditioned export as a conditioned one")
    spec = model.condition_spec
    logical, indices = [], []
    for row in plan.rows:
        family = row.condition["family"]
        require(family in spec.families,
                f"candidate {row.candidate_id!r}: family {family!r} is not one of the model's "
                f"families {spec.families}")
        assay = (conditions_lib.ASSAY_NULL if row.condition["assay"] is None
                 else row.condition["assay"])
        condition = conditions_lib.Condition(family=family, bin=conditions_lib.BIN_NULL,
                                             assay=assay)
        indices.append([int(item) for item in spec.encode(condition)])
        logical.append({"family": family, "assay": row.condition["assay"], "quality": None})
    return logical, indices


def _resolve_tap(model, plan):
    if plan.activation["kind"] == ACTIVATION_FINAL:
        return None
    name = plan.activation["name"]
    names = hooks_lib.bind_conditioned_dplm(model, route=HOOK_ROUTE).names()
    require(name in names,
            f"the tap {name!r} is not a tap of this model, which has "
            f"{len(model.native.esm.encoder.layer)} layer(s) and {len(names)} taps")
    return name


def _prepare_tokens(model, plan):
    """Native ids per row, built directly AND through the public route AND from the vocabulary table;
    all three must agree, and nothing may exceed the model's positions (nothing is truncated)."""
    limit = int(model.config.max_position_embeddings)
    tokens = []
    for row in plan.rows:
        direct = native_token_ids(row.heavy, row.light)
        require(list(direct) == _expected_encoding(row.heavy, row.light)["native_ids"]
                == list(independent_native_ids(row.heavy, row.light)),
                f"candidate {row.candidate_id!r}: the direct native ids, the vocabulary table and "
                "the public-route translation disagree")
        require(len(direct) <= limit,
                f"candidate {row.candidate_id!r} needs {len(direct)} native tokens but the model "
                f"has {limit} positions; nothing is truncated (a dropped light chain would make "
                "the paired-context claim false)")
        require(dplm.DPLM_PAD_ID not in direct and dplm.DPLM_MASK_ID not in direct,
                f"candidate {row.candidate_id!r}: an unmasked export may not contain pad or mask")
        tokens.append(direct)
    return tokens


def _batch_tensor(tokens, members):
    width = max(len(tokens[i]) for i in members)
    batch = torch.full((len(members), width), dplm.DPLM_PAD_ID, dtype=torch.long)
    for position, member in enumerate(members):
        batch[position, :len(tokens[member])] = torch.tensor(tokens[member], dtype=torch.long)
    return batch


@contextlib.contextmanager
def _frozen_execution(model):
    """Eval mode, no_grad, the CPU and initialised-CUDA generators forked, every module's own
    training flag restored (``controlled_evaluation``) and autocast OFF for the actual device. CUDA
    is never initialised by this function."""
    device_type = next(model.parameters()).device.type
    require(device_type in ("cpu", "cuda"),
            f"unsupported device type {device_type!r}: the native export runs on cpu or cuda")
    with interp.controlled_evaluation(model):
        with torch.autocast(device_type=device_type, enabled=False):
            yield


def _run_batch(model, ids, condition_indices, *, registry, tap):
    """One native forward. Returns the (B, T, hidden) float32 CPU activation INCLUDING the padded
    positions (the caller keeps only each row's real tokens). ``registry`` is None for the final
    hidden state; otherwise the capture-only tap ``tap`` is enabled around the forward and its
    captured tensor is checked against the forward's own shape and the declared layout."""
    with _frozen_execution(model):
        if registry is not None:
            registry.clear().clear_plans()
            require(not registry.interventions, "the exporter's registry carries an intervention")
            registry.enable([tap])
        try:
            output = model.forward_native_ids(ids, condition_indices=condition_indices)
        finally:
            if registry is not None:
                registry.disable()
        final = output["last_hidden_state"]
        if registry is None:
            activation = final
        else:
            exported = registry.export(tap)
            require(exported["layout"] == hooks_lib.LAYOUT_TOKENS_FIRST
                    and exported["position_axis"] == 1,
                    f"{tap}: the capture is not (batch, tokens, hidden) with positions on axis 1")
            require(exported["shape"] == [int(i) for i in final.shape],
                    f"{tap}: captured shape {exported['shape']} differs from the forward's "
                    f"{list(final.shape)}")
            activation = exported["tensor"]
        return activation.detach().to("cpu", dtype=torch.float32).clone()


def _header_document(plan, logical, indices, runtime, token_maps, hidden):
    choice = dict(plan.activation)
    token_map_sha256 = fb.digest(token_maps)
    space = _token_feature_space(plan.supervision, hidden, choice, plan.conditioning_mode,
                                 runtime["sha256"], token_map_sha256)
    rows = plan.rows
    return {
        "schema": TOKEN_CACHE_SCHEMA, "representation_id": plan.representation_id,
        "conditioning_mode": plan.conditioning_mode, "activation": _activation_record(choice),
        "rows": {"ids": [row.candidate_id for row in rows], "heavy": [row.heavy for row in rows],
                 "light": [row.light for row in rows], "groups": [row.group_id for row in rows],
                 "splits": [row.split for row in rows],
                 "identities": [row.identity for row in rows], "conditions": logical,
                 "condition_indices": indices, "token_maps": token_maps},
        "feature_space": space.document(), "extractor": dict(plan.extractor),
        "runtime": {"sha256": runtime["sha256"], "document": runtime["document"]},
        "storage": {"file": None, "file_sha256": None, "arrays": {}},
        "token_map_sha256": token_map_sha256, "cache_sha256": None}


def export_dplm_tokens(model, plan, header_path, npz_path, *, batch_size,
                       protected_input_paths=(), overwrite=False):
    """Run the native forward over every plan row and publish a ragged token cache.

    ``model`` is YOUR ``ConditionedDPLM`` (nothing is loaded or discovered here). Order: re-validate
    the plan as a detached snapshot (:func:`snapshot_export_plan`; an edited or hand-built plan is
    refused, never normalised, and the caller's object is untouched); validate every path and every
    plan/model fact BEFORE any compute or write (aliases, hooks, supported types, token agreement,
    overlength, conditions, tap); fingerprint and inventory the model; run the batches (right-padded,
    discarded padding, controlled evaluation), checking the numerical backend settings after each;
    fingerprint again and REFUSE to publish if anything moved; seal, stage and publish; reload and
    verify. Returns a summary mapping, whose ``plan_sha256`` is the snapshot that was exported.
    ``protected_input_paths`` are the files the caller supplied (the plan's own path is added
    automatically); only those are protected from aliasing.
    """
    plan = snapshot_export_plan(plan)
    batch_size = fb._integer(batch_size, "batch_size", minimum=1)
    header_path, npz_path = Path(header_path), Path(npz_path)
    inputs = _protected_mapping(protected_input_paths)
    if plan.path is not None:
        inputs["export plan file"] = Path(plan.path)
    _validate_cache_outputs(header_path, npz_path, inputs, overwrite)
    require_supported_model(model)
    refuse_existing_forward_hooks(model)
    tokens = _prepare_tokens(model, plan)
    logical, indices = _resolve_conditions(model, plan)
    tap = _resolve_tap(model, plan)
    hidden = int(model.config.hidden_size)
    before = runtime_fingerprint(model)
    backend_before = before["document"]["environment"]["backend_settings"]
    state_before = state_inventory(model)
    registry = None if tap is None else hooks_lib.bind_conditioned_dplm(model, route=HOOK_ROUTE)
    pieces = []
    try:
        for start in range(0, len(plan.rows), batch_size):
            members = list(range(start, min(len(plan.rows), start + batch_size)))
            ids = _batch_tensor(tokens, members)
            condition = (None if plan.conditioning_mode == CONDITIONING_NONE
                         else torch.tensor([indices[i] for i in members], dtype=torch.long))
            activation = _run_batch(model, ids, condition, registry=registry, tap=tap)
            moved = sorted(name for name, value in backend_settings().items()
                           if value != backend_before[name])
            require(not moved,
                    f"numerical backend setting(s) {moved} changed during the export (seen after "
                    f"the batch starting at row {start}); a cache computed under moving precision "
                    "or kernel settings would be unreproducible, so nothing was published")
            require(tuple(activation.shape) == (len(members), int(ids.shape[1]), hidden),
                    f"the native forward returned shape {tuple(activation.shape)}, expected "
                    f"{(len(members), int(ids.shape[1]), hidden)}")
            for position, member in enumerate(members):
                pieces.append(np.array(activation[position, :len(tokens[member])].numpy(),
                                       dtype=np.float32, copy=True))
    finally:
        if registry is not None:
            registry.disable()
    after = runtime_fingerprint(model)
    require(after["sha256"] == before["sha256"],
            "the model changed during the export (its runtime fingerprint moved); a cache of "
            "features from a moving model would be unreproducible, so nothing was published")
    require(state_inventory(model) == state_before,
            "the export disturbed the model's training flags, gradients, hooks or RNG state; "
            "nothing was published")
    features = np.concatenate(pieces, axis=0)
    require(bool(np.isfinite(features).all()),
            "the native forward produced NaN or infinity; nothing was published")
    offsets = np.concatenate([[0], np.cumsum([len(item) for item in tokens])]).astype(np.int64)
    token_maps = [_expected_encoding(row.heavy, row.light) for row in plan.rows]
    document = _header_document(plan, logical, indices, before, token_maps, hidden)
    document["storage"]["file"] = npz_path.name
    sealed, payload = _seal(document, features, offsets)
    _parse_token_header(sealed)
    header_text = json.dumps(fb._builtin(sealed), sort_keys=True, separators=(",", ":"),
                             allow_nan=False) + "\n"
    _validate_cache_outputs(header_path, npz_path, inputs, overwrite)
    _publish(header_path, npz_path, header_text, payload)
    reloaded = load_token_cache(header_path, npz_path=npz_path)
    require(reloaded.cache_sha256 == sealed["cache_sha256"],
            "the reloaded cache does not match what was written")
    return {"header": str(header_path), "npz": str(npz_path),
            "cache_sha256": reloaded.cache_sha256, "file_sha256": reloaded.file_sha256,
            "runtime_sha256": reloaded.runtime_sha256,
            "token_map_sha256": reloaded.token_map_sha256, "rows": len(plan.rows),
            "token_rows": int(features.shape[0]), "hidden_size": hidden,
            "batch_size": batch_size, "plan_sha256": plan.fingerprint(),
            "protected_input_paths": sorted(str(path) for path in inputs.values())}


# ---------------------------------------------------------------------------
# pooling and the matrix bridge
# ---------------------------------------------------------------------------

def _pooling_text(pooling):
    if pooling == POOL_ALL:
        return ("all_nonpad_mean: the mean over ALL real (non-pad) native tokens of a candidate, "
                "<cls>/<eos> specials included - the native ConditionedDPLM.representation "
                "semantics")
    return ("residue_mean: the mean over the residue tokens only (heavy and light residues; the "
            "<cls>/<eos> specials are excluded) - a different named representation")


def pool_token_features(cache, pooling):
    """``(rows, hidden)`` float64 pooled features, accumulated in float64 from the stored float32.
    Both poolings derive from the SAME token cache; no model forward is run."""
    require(isinstance(cache, TokenCache), "pool_token_features needs a TokenCache")
    require(pooling in POOLINGS, f"pooling must be one of {POOLINGS}, got {pooling!r}")
    cache.verify()
    roles = [encoded["roles"] for encoded in cache._document["rows"]["token_maps"]]
    pooled = np.empty((len(roles), cache.hidden_size), dtype=np.float64)
    for position, row_roles in enumerate(roles):
        block = cache.features[cache.offsets[position]:cache.offsets[position + 1]]
        if pooling == POOL_RESIDUE:
            block = block[[index for index, code in enumerate(row_roles) if code in "HL"]]
        pooled[position] = block.astype(np.float64).mean(axis=0)
    return pooled


def _join_roster(cache, table, manifest):
    """The exact full-roster join: ONE fixed full-identity length, every residue and the delimiter
    admitted by the manifest alphabet, the same ids, and for every id the same full ``H|L`` identity,
    group and split. Nothing is trimmed, padded, dropped or intersected."""
    lengths = sorted({len(identity) for identity in cache.identities})
    require(len(lengths) == 1,
            f"the token cache has {len(lengths)} different full-identity lengths {lengths}: the "
            "matrix joins ONE fixed full-identity length, and the bridge never trims, pads or "
            "drops a sequence to make rows fit")
    require(lengths[0] == manifest.length,
            f"the cache's full identities have length {lengths[0]} but the matrix manifest "
            f"declares {manifest.length}")
    unadmitted = sorted(set("".join(cache.identities)) - set(manifest.alphabet))
    require(not unadmitted,
            f"the manifest alphabet {manifest.alphabet!r} does not admit {unadmitted}: every "
            f"residue and the {DELIMITER!r} chain delimiter must be in it")
    cache_ids, roster_ids = set(cache.ids), set(table.ids)
    missing, extra = sorted(roster_ids - cache_ids), sorted(cache_ids - roster_ids)
    require(not missing, f"the token cache lacks {len(missing)} of the {len(roster_ids)} roster "
                         f"candidate(s), e.g. {missing[:3]}; there is no convenient intersection")
    require(not extra, f"the token cache holds {len(extra)} candidate(s) outside the roster, "
                       f"e.g. {extra[:3]}")
    index = {ident: position for position, ident in enumerate(table.ids)}
    for label, mine, theirs in (("full H|L identity", cache.identities, table.sequences),
                                ("group", cache.groups, table.groups),
                                ("split", cache.splits, table.splits)):
        bad = [ident for ident, value in zip(cache.ids, mine) if value != theirs[index[ident]]]
        require(not bad, f"{len(bad)} candidate(s) have a different {label} in the token cache "
                         f"than in the candidate table, e.g. {bad[:3]}")


def _pooled_space(cache, pooling):
    activation = cache.activation
    names = tuple(f"h{index}" for index in range(cache.hidden_size))
    return fb.FeatureSpace(
        kind=POOLED_SPACE_KIND, names=names,
        schema={"construction": "pooled native DPLM token features", "pooling": pooling,
                "pooling_description": _pooling_text(pooling),
                "activation": {"kind": activation["kind"], "name": activation["name"],
                               "location": activation["location"]},
                "conditioning_mode": cache.conditioning_mode,
                "runtime_sha256": cache.runtime_sha256, "token_cache_sha256": cache.cache_sha256,
                "token_map_sha256": cache.token_map_sha256, "hidden_size": cache.hidden_size,
                "identity": "complete heavy|light sequence with the chain delimiter",
                "candidate_masking": "none"},
        history=(f"native DPLM token features (token cache {cache.cache_sha256[:16]}, runtime "
                 f"{cache.runtime_sha256[:16]}) of the UNMASKED complete candidates",
                 f"{pooling} pooled in float64 from the stored float32 tokens, stored float32; no "
                 "second forward was run"),
        supervision=cache.space.supervision)


def _pooled_cache_files(npz_name, *, representation_id, ids, sequences, groups, splits, features,
                        space, identity, extractor, activation, conditioning):
    """``(header text, NPZ bytes)`` of a ``functional-feature-cache/1``, built with the matrix's OWN
    format helpers (``_parse_header``, ``_canonical_cache_document``, ``_content_sha256``) so the
    matrix loader accepts it unchanged, but published by this module's exclusive-staging writer
    (``fm.write_feature_cache`` stages through a non-exclusive ``write_bytes``). The caller reloads
    the result through the real matrix loader, so any drift between the two is caught, not assumed."""
    array = np.array(features, dtype=np.float64)
    require(array.ndim == 2 and array.shape == (len(ids), space.dimension),
            f"pooled features must be ({len(ids)}, {space.dimension}), got {array.shape}")
    require(bool(np.isfinite(array).all()), "pooled features must be finite")
    stored = array.astype(np.dtype(STORED_DTYPE))
    require(bool(np.isfinite(stored).all()), f"casting the pooled features to {STORED_DTYPE} "
                                              "overflowed")
    document = {
        "schema": fm.CACHE_SCHEMA, "representation_id": representation_id, "identity": identity,
        "rows": {"ids": list(ids), "sequences": list(sequences), "groups": list(groups),
                 "splits": list(splits)},
        "feature_space": space.document(),
        "storage": {"file": npz_name, "file_sha256": fm._PLACEHOLDER_HASH,
                    "array": fm.CACHE_ARRAY, "dtype": STORED_DTYPE,
                    "shape": [int(stored.shape[0]), int(stored.shape[1])],
                    "content_sha256": fm._content_sha256(stored, STORED_DTYPE)},
        "extractor": extractor, "activation": activation, "conditioning": conditioning,
        "cache_sha256": fm._PLACEHOLDER_HASH}
    document = json.loads(fb.canonical_json(document))
    parsed = fm._parse_header(document)
    order = sorted(range(len(parsed.ids)), key=parsed.ids.__getitem__)
    document["cache_sha256"] = fb.digest(fm._canonical_cache_document(parsed, order, stored[order]))
    buffer = io.BytesIO()
    np.savez(buffer, **{fm.CACHE_ARRAY: stored})
    payload = buffer.getvalue()
    document["storage"]["file_sha256"] = hashlib.sha256(payload).hexdigest()
    fm._parse_header(document)
    return json.dumps(fb._builtin(document), indent=2, sort_keys=True, allow_nan=False) + "\n", payload


def write_pooled_matrix_cache(cache, candidate_table, matrix_manifest, header_path, npz_path, *,
                              representation_id, pooling, protected_input_paths=()):
    """Pool a token cache into a ``functional-feature-cache/1`` the matrix loads unchanged.

    The roster join is exact (see :func:`_join_roster`); the cache carries the plan's feature
    supervision, the extractor, the activation, the pooling, the runtime and token-map digests and
    the conditioning as per-row values (null in ``none`` mode; quality an explicit null always). The
    reference prior is NOT here: it stays solely in the candidate CSV. Existing outputs are refused
    (no overwrite is offered). ``protected_input_paths`` must list the candidate CSV, the manifest
    file and any other input you want protected from aliasing: a ``CandidateTable`` and a
    ``MatrixManifest`` do not retain their file names, and only the token cache's own header/NPZ
    paths are known here. The files are staged exclusively and published NPZ-first, header-last (as
    the token cache is), then reloaded through the matrix loader and joined again.
    """
    require(isinstance(cache, TokenCache), "write_pooled_matrix_cache needs a TokenCache")
    require(isinstance(candidate_table, fb.CandidateTable),
            "write_pooled_matrix_cache needs a CandidateTable")
    require(isinstance(matrix_manifest, fm.MatrixManifest),
            "write_pooled_matrix_cache needs a MatrixManifest")
    representation_id = fm.safe_id(representation_id, "pooled representation_id")
    require(pooling in POOLINGS, f"pooling must be one of {POOLINGS}, got {pooling!r}")
    header_path, npz_path = Path(header_path), Path(npz_path)
    inputs = {"token cache header": Path(cache.header_path), "token cache NPZ": Path(cache.npz_path)}
    inputs.update(_protected_mapping(protected_input_paths))
    _validate_cache_outputs(header_path, npz_path, inputs, False)
    cache.verify()
    _join_roster(cache, candidate_table, matrix_manifest)
    pooled = pool_token_features(cache, pooling)
    logical, count = cache.conditions, len(cache.ids)
    conditioning = {
        "candidate_masking": "none", "answer_free": True,
        "family": {"recorded": True,
                   "values": [None if item is None else item["family"] for item in logical]},
        "assay": {"recorded": True,
                  "values": [None if item is None else item["assay"] for item in logical]},
        "quality": {"recorded": True, "values": [None] * count}}
    activation = cache.activation
    activation_record = {
        "location": activation["location"], "pooling": _pooling_text(pooling),
        "token_map": (f"native layout <cls> H <eos> <eos> [<cls> L <eos>] (heavy residue i at 1+i, "
                      f"light residue j at len(H)+4+j; heavy-only has no light block); residue "
                      f"positions follow the cache's token map; token_map_sha256="
                      f"{cache.token_map_sha256}")}
    identity = {"kind": fm.IDENTITY_COMPLETE_SEQUENCE, "alphabet": matrix_manifest.alphabet,
                "length": matrix_manifest.length,
                "construct_description": (f"the complete heavy chain, the {DELIMITER!r} chain "
                                          "delimiter and the complete light chain (a heavy-only "
                                          "record keeps its trailing delimiter)")}
    space = _pooled_space(cache, pooling)
    header_text, payload = _pooled_cache_files(
        npz_path.name, representation_id=representation_id, ids=list(cache.ids),
        sequences=list(cache.identities), groups=list(cache.groups), splits=list(cache.splits),
        features=pooled, space=space, identity=identity, extractor=cache.extractor,
        activation=activation_record, conditioning=conditioning)
    _validate_cache_outputs(header_path, npz_path, inputs, False)
    _publish(header_path, npz_path, header_text, payload)
    loaded = fm.load_feature_cache(header_path, npz_path=npz_path)
    fm.check_cache_against_roster(loaded, candidate_table, matrix_manifest)
    expected = pooled.astype(np.float32).astype(np.float64)
    require(np.array_equal(np.asarray(loaded.features), expected),
            "the reloaded pooled features differ from the pooled values that were written")
    return {"header": str(header_path), "npz": str(npz_path), "representation_id": representation_id,
            "pooling": pooling, "rows": count, "dimension": cache.hidden_size,
            "cache_sha256": loaded.cache_sha256, "file_sha256": loaded.raw["file_sha256"],
            "token_cache_sha256": cache.cache_sha256, "runtime_sha256": cache.runtime_sha256,
            "token_map_sha256": cache.token_map_sha256,
            "protected_input_paths": sorted(str(path) for path in inputs.values())}


# ---------------------------------------------------------------------------
# a tiny random CPU model and the end-to-end selftest
# ---------------------------------------------------------------------------

def tiny_model(seed=0, *, lora=None, table_std=0.1):
    """A real, tiny, RANDOMLY initialised CPU ``ConditionedDPLM`` (hidden 32, 2 layers, 4 heads,
    FFN 64, 64 positions, no dropout). The condition tables start at zero in the model; here they
    are randomised (``table_std``; 0 keeps them zero) so conditioned and unconditioned forwards
    differ. Not pretrained: it qualifies code paths, nothing else. The global RNG is untouched."""
    config = dplm.DPLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                             intermediate_size=64, max_position_embeddings=64,
                             hidden_dropout_prob=0.0)
    native = dplm.NativeDPLM.random_init(config, seed=seed, std=0.3)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        model = dplm.ConditionedDPLM(native, conditions_lib.ConditionSpec(assays=("a1", "a2")),
                                     lora=lora)
    if table_std:
        generator = torch.Generator().manual_seed(int(seed) + 5)
        with torch.no_grad():
            for table in model._tables():
                table.weight.copy_(torch.randn(table.weight.shape, generator=generator)
                                   * float(table_std))
    return model


_SELFTEST_WINDOW = "ACDE"
_SELFTEST_LIGHTS = ("DIQMTQ", "EIVLTQ")


def _selftest_rows(seed):
    """Forty fixed-length complete ``H|L`` candidates: an 8-residue heavy chain with a 4-residue
    variable window and one of two light chains (full identity length 15), groups = the first two
    window residues, whole groups held out, an additive planted utility over the window."""
    rng = np.random.default_rng([seed, 700])
    windows = ["".join(item) for item in itertools.product(_SELFTEST_WINDOW, repeat=4)]
    chosen = sorted(int(i) for i in rng.choice(len(windows), size=40, replace=False))
    windows = [windows[i] for i in chosen]
    groups = sorted({window[:2] for window in windows})
    held_out = {str(item) for item in rng.choice(groups, size=4, replace=False)}
    weights = rng.normal(size=(4, len(_SELFTEST_WINDOW)))
    rows = []
    for number, window in enumerate(windows):
        utility = float(sum(weights[position, _SELFTEST_WINDOW.index(residue)]
                            for position, residue in enumerate(window))
                        + rng.normal(scale=0.1))
        rows.append({"candidate_id": f"c{number:03d}", "heavy": "EV" + window + "GG",
                     "light": _SELFTEST_LIGHTS[int(rng.integers(0, 2))],
                     "group_id": f"g-{window[:2]}",
                     "split": fb.SPLIT_EVAL if window[:2] in held_out else fb.SPLIT_TRAIN,
                     "utility": utility})
    return rows


def _selftest_plan_document(rows, *, representation_id, mode, activation):
    explicit = [(family, assay) for family in conditions_lib.FAMILIES for assay in (None, "a1", "a2")]
    plan_rows = []
    for number, row in enumerate(rows):
        condition = None
        if mode == CONDITIONING_EXPLICIT:
            family, assay = explicit[number % len(explicit)]
            condition = {"family": family, "assay": assay, "quality": None}
        plan_rows.append({"candidate_id": row["candidate_id"], "heavy": row["heavy"],
                          "light": row["light"], "group_id": row["group_id"],
                          "split": row["split"], "condition": condition})
    return {"schema": PLAN_SCHEMA, "representation_id": representation_id,
            "conditioning_mode": mode, "activation": activation,
            "feature_supervision": {
                "status": fb.HISTORY_LABEL_FREE,
                "description": "tiny random-initialised CPU DPLM; no labels of any kind",
                "sequence_only_exposure": {"status": "none"}},
            "extractor": {"kind": "synthetic", "checkpoint_sha256": None, "source_sha256": None,
                          "description": ("tiny random-init NativeDPLM on CPU; not pretrained and "
                                          "not a checkpoint")},
            "rows": plan_rows}


def _flip_one_byte(path):
    data = bytearray(Path(path).read_bytes())
    data[len(data) // 2] ^= 0x01
    Path(path).write_bytes(bytes(data))


def _refused(call):
    try:
        call()
    except ValueError:
        return True
    return False


def _selftest_in(work, seed):
    checks = []

    def check(name, holds, detail=None):
        checks.append({"name": name, "holds": bool(holds), "detail": detail})
    rows = _selftest_rows(seed)
    model = tiny_model(seed)
    plans_dir, token_dir, matrix_dir = work / "plans", work / "token_caches", work / "matrix_inputs"
    for directory in (plans_dir, token_dir, matrix_dir):
        directory.mkdir()
    final_choice = {"kind": ACTIVATION_FINAL, "name": None}
    hook_choice = {"kind": ACTIVATION_HOOK, "name": f"route/{HOOK_ROUTE}/block1/resid_post"}
    variants = {"fh_none": (final_choice, CONDITIONING_NONE), "hk_none": (hook_choice, CONDITIONING_NONE),
                "fh_explicit": (final_choice, CONDITIONING_EXPLICIT)}
    state_before = state_inventory(model)
    fingerprint_before = runtime_fingerprint(model)["sha256"]
    plans, exported = {}, {}
    for name, (choice, mode) in variants.items():
        plan_path = plans_dir / f"{name}.plan.json"
        fb.write_json(plan_path, _selftest_plan_document(rows, representation_id=name, mode=mode,
                                                          activation=choice))
        plans[name] = read_export_plan(plan_path)
        exported[name] = export_dplm_tokens(model, plans[name], token_dir / f"{name}.token.json",
                                            token_dir / f"{name}.token.npz", batch_size=7)
    caches = {name: load_token_cache(token_dir / f"{name}.token.json") for name in variants}
    fh, hk, fe = caches["fh_none"], caches["hk_none"], caches["fh_explicit"]
    check("native::three_exports_ran_on_the_real_tiny_model_and_reload_verified",
          all(cache.verify() for cache in caches.values()) and len(fh.ids) == len(rows),
          {"rows": len(rows), "token_rows": int(fh.features.shape[0])})
    check("native::model_fingerprint_modes_rng_and_grads_unchanged_by_exports",
          runtime_fingerprint(model)["sha256"] == fingerprint_before
          and state_inventory(model) == state_before
          and fh.runtime_sha256 == hk.runtime_sha256 == fingerprint_before,
          {"runtime_sha256": fingerprint_before})
    worst_forward, worst_public, worst_mean, worst_residue = 0.0, 0, 0.0, 0.0
    pooled_all, pooled_residue = pool_token_features(fh, POOL_ALL), pool_token_features(fh, POOL_RESIDUE)
    all_roles = [encoded["roles"] for encoded in fh.token_maps]
    with interp.controlled_evaluation(model):
        for position, row in enumerate(plans["fh_none"].rows):
            direct = native_token_ids(row.heavy, row.light)
            worst_public += int(direct != independent_native_ids(row.heavy, row.light))
            reference = model.native(torch.tensor([list(direct)]))["last_hidden_state"][0].numpy()
            stored = fh.features[fh.offsets[position]:fh.offsets[position + 1]]
            worst_forward = max(worst_forward, float(np.abs(stored - reference).max()))
            if position < 8:
                public = serial.ablang_serialization(_construct(row.heavy, row.light),
                                                     masked_sites=(), order=()).tokens
                native_mean = model.representation(torch.tensor([list(public)]))[0].numpy()
                worst_mean = max(worst_mean, float(np.abs(pooled_all[position] - native_mean).max()))
            kept = [index for index, code in enumerate(all_roles[position]) if code in "HL"]
            worst_residue = max(worst_residue, float(np.abs(
                pooled_residue[position] - reference[kept].astype(np.float64).mean(axis=0)).max()))
    check("native::final_hidden_equals_an_independent_unpadded_forward_per_row",
          worst_forward < 1e-5, {"max_abs_difference": worst_forward})
    check("native::direct_native_ids_equal_the_public_translation_for_every_row", worst_public == 0)
    check("native::all_nonpad_mean_equals_the_native_representation_semantics",
          worst_mean < 1e-5, {"max_abs_difference": worst_mean})
    check("native::residue_mean_is_the_residue_only_mean_and_differs_from_all_nonpad",
          worst_residue < 1e-5 and float(np.abs(pooled_all - pooled_residue).max()) > 1e-4,
          {"max_abs_difference_to_hand_mean": worst_residue})
    with torch.no_grad():
        through_ln = model.native.esm.encoder.emb_layer_norm_after(
            torch.from_numpy(np.array(hk.features))).numpy()
    ln_gap = float(np.abs(through_ln - fh.features).max())
    raw_gap = float(np.abs(hk.features - fh.features).max())
    check("native::hook_resid_post_through_the_final_layer_norm_is_final_hidden_and_is_not_it_raw",
          ln_gap < 1e-5 and raw_gap > 1e-3, {"layer_norm_gap": ln_gap, "raw_gap": raw_gap})
    gaps = {}
    for size in (1, len(rows)):
        variant_dir = work / f"batch_{size}"
        variant_dir.mkdir()
        export_dplm_tokens(model, plans["fh_none"], variant_dir / "t.json", variant_dir / "t.npz",
                           batch_size=size)
        other = load_token_cache(variant_dir / "t.json")
        gaps[f"batch_size_{size}"] = float(np.abs(other.features - fh.features).max())
    check("native::batch_size_and_composition_do_not_change_features_beyond_float32_rounding",
          all(gap < 1e-5 for gap in gaps.values()), gaps)
    check("native::explicit_null_conditioning_differs_from_none_when_the_tables_are_nonzero",
          float(np.abs(fe.features - fh.features).max()) > 1e-4
          and all(item is not None for item in fe.conditions)
          and all(item is None for item in fh.conditions),
          {"max_abs_difference": float(np.abs(fe.features - fh.features).max())})
    source_header = (token_dir / "fh_none.token.json").read_text(encoding="utf-8")
    source_npz = (token_dir / "fh_none.token.npz").read_bytes()
    flipped_dir, edited_dir = work / "tamper_flipped", work / "tamper_edited"
    flipped_dir.mkdir()
    edited_dir.mkdir()
    (flipped_dir / "fh_none.token.json").write_text(source_header, encoding="utf-8")
    (flipped_dir / "fh_none.token.npz").write_bytes(source_npz)
    _flip_one_byte(flipped_dir / "fh_none.token.npz")
    marker = '"representation_id":"fh_none"'
    (edited_dir / "fh_none.token.json").write_text(
        source_header.replace(marker, '"representation_id":"fh_nope"'), encoding="utf-8")
    (edited_dir / "fh_none.token.npz").write_bytes(source_npz)
    check("native::a_flipped_npz_byte_and_an_edited_header_are_each_refused",
          marker in source_header
          and _refused(lambda: load_token_cache(flipped_dir / "fh_none.token.json"))
          and _refused(lambda: load_token_cache(edited_dir / "fh_none.token.json")))
    # ---- pooled caches, built with NO further forward ----
    table_path, manifest_path = matrix_dir / "candidates.csv", matrix_dir / "manifest.json"
    identities = {row["candidate_id"]: identity_of(row["heavy"], row["light"]) for row in rows}
    table_path.write_text("candidate_id,sequence,group_id,split,prior_log_weight\n" + "\n".join(
        f"{row['candidate_id']},{identities[row['candidate_id']]},{row['group_id']},{row['split']},0.0"
        for row in rows) + "\n", encoding="utf-8")
    alphabet = "".join(sorted(set("".join(identities.values()))))
    fb.write_json(manifest_path, {
        "schema": fm.MANIFEST_SCHEMA, "target": "synthetic-native-export",
        "assay": "synthetic-closed-form",
        "utility": {"name": "planted additive utility over the variable window",
                    "units": "arbitrary", "scale": "linear", "higher_is_better": True,
                    "useful_threshold": 1.0},
        "grouping_rationale": ("groups are the first two window residues; whole groups are held "
                               "out. This is a shared-prefix grouping only and enforces no minimum "
                               "sequence distance"),
        "candidates": {"identity_kind": fm.IDENTITY_COMPLETE_SEQUENCE, "alphabet": alphabet,
                       "length": len(next(iter(identities.values())))},
        "prior": {"kind": fb.PRIOR_UNIFORM,
                  "semantics": "every candidate in the pool has the same declared reference weight",
                  "provenance": {"status": "label_free",
                                 "description": "declared uniform; no outcomes used",
                                 "sequence_only_exposure": {"status": "none"}}},
        "notes": "synthetic native-export selftest roster"})
    manifest = fm.load_matrix_manifest(manifest_path)
    table = fb.read_candidates(table_path, alphabet=manifest.alphabet, length=manifest.length)
    train_lines = [f"{row['candidate_id']},{row['utility']!r}" for row in rows
                   if row["split"] == fb.SPLIT_TRAIN]
    eval_lines = [f"{row['candidate_id']},{row['utility']!r}" for row in rows
                  if row["split"] == fb.SPLIT_EVAL]
    (matrix_dir / "train_outcomes.csv").write_text("candidate_id,utility\n" + "\n".join(train_lines)
                                                   + "\n", encoding="utf-8")
    forwards = []
    probe = model.native.register_forward_hook(lambda *_: forwards.append(1))
    try:
        pooled_files = {}
        for rep, (source, pooling) in {"fh_all": ("fh_none", POOL_ALL), "fh_res": ("fh_none", POOL_RESIDUE),
                                       "hk_all": ("hk_none", POOL_ALL), "hk_res": ("hk_none", POOL_RESIDUE)}.items():
            pooled_files[rep] = write_pooled_matrix_cache(
                caches[source], table, manifest, matrix_dir / f"{rep}.cache.json",
                matrix_dir / f"{rep}.cache.npz", representation_id=rep, pooling=pooling,
                protected_input_paths=[table_path, manifest_path])
    finally:
        probe.remove()
    check("native::both_poolings_reuse_the_token_cache_with_no_second_forward", not forwards,
          {"forwards_during_pooling": len(forwards)})
    reloaded = {rep: fm.load_feature_cache(matrix_dir / f"{rep}.cache.json") for rep in pooled_files}
    for rep, cache in reloaded.items():
        fm.check_cache_against_roster(cache, table, manifest)
    check("matrix::pooled_caches_reload_through_the_matrix_loader_and_join_the_roster_exactly",
          sorted(reloaded) == sorted(pooled_files) and all(
              cache.space.schema["token_cache_sha256"] == caches[source].cache_sha256
              and cache.conditioning_summary["quality"]["state"] == "explicit_null"
              for cache, source in ((reloaded["fh_all"], "fh_none"), (reloaded["hk_res"], "hk_none"))))
    # A roster whose identities differ in length is exported (the ragged cache supports it) but the
    # fixed-length matrix bridge must refuse it, and write nothing.
    variable_dir = work / "variable"
    variable_dir.mkdir()
    variable_rows = [dict(rows[0]), dict(rows[1]), dict(rows[2], light="")]
    fb.write_json(variable_dir / "plan.json", _selftest_plan_document(
        variable_rows, representation_id="variable", mode=CONDITIONING_NONE,
        activation=final_choice))
    export_dplm_tokens(model, read_export_plan(variable_dir / "plan.json"),
                       variable_dir / "t.json", variable_dir / "t.npz", batch_size=2)
    variable = load_token_cache(variable_dir / "t.json")
    check("native::a_heavy_only_row_has_no_light_block_and_len_H_plus_3_tokens",
          variable.lengths() == [len(rows[0]["heavy"]) + len(rows[0]["light"]) + 5,
                                 len(rows[1]["heavy"]) + len(rows[1]["light"]) + 5,
                                 len(rows[2]["heavy"]) + 3]
          and variable.token_maps[2]["roles"] == "S" + "H" * len(rows[2]["heavy"]) + "EP")
    check("matrix::a_variable_length_roster_is_refused_by_the_bridge_and_nothing_is_written",
          _refused(lambda: write_pooled_matrix_cache(
              variable, table, manifest, variable_dir / "p.json", variable_dir / "p.npz",
              representation_id="variable", pooling=POOL_ALL))
          and not (variable_dir / "p.json").exists() and not (variable_dir / "p.npz").exists())
    plan_path = matrix_dir / "matrix_plan.json"
    fb.write_json(plan_path, {
        "schema": fm.PLAN_SCHEMA, "representations": sorted(pooled_files),
        "maps": [{"id": "linear", "kind": "linear", "center": True, "scale": "std",
                  "trace_normalize": True},
                 {"id": "rbf_median", "kind": "rbf", "center": True, "scale": "std",
                  "trace_normalize": True, "bandwidth": {"rule": fm.BANDWIDTH_MEDIAN,
                                                         "multiplier": 1.0}}],
        "label_budgets": [8, 16], "seeds": [0, 1],
        "baseline": {"tau": 1.0, "ridge_alpha": 1.0, "sir_proposal_draws": 64, "final_draws": 16,
                     "dpo_pressures": [0.5, 1.0]},
        "reference": {"representation": "fh_all", "map": "linear"},
        "bootstrap": {"replicates": 100, "seed": 0, "confidence": 0.9, "min_groups": 3},
        "notes": ("fixed readout, budgets and seeds; the reference cell anchors paired comparisons "
                  "and is not a winner")})
    out = work / "matrix"
    eval_path = matrix_dir / "eval_outcomes.csv"
    try:
        fm.run_matrix(candidates=table_path, train_outcomes=matrix_dir / "train_outcomes.csv",
                      eval_outcomes=eval_path, manifest=manifest_path, plan=plan_path,
                      caches={rep: matrix_dir / f"{rep}.cache.json" for rep in pooled_files},
                      output_dir=out)
        opened_before_freeze = True
    except ValueError as error:
        opened_before_freeze = "does not exist" not in str(error)
    verified = fm.verify_freeze(out)
    check("matrix::the_freeze_was_built_and_verified_before_any_evaluation_outcome_existed",
          not opened_before_freeze and not eval_path.exists())
    eval_path.write_text("candidate_id,utility\n" + "\n".join(eval_lines) + "\n", encoding="utf-8")
    results = fm.evaluate_matrix(out, eval_path)
    counts = results["status_counts"]
    check("matrix::every_planned_cell_fitted_and_froze_with_no_failure",
          counts[fm.STATUS_FAILED] == 0 and counts[fm.STATUS_OK] == counts["cells"],
          counts)
    check("matrix::every_regret_and_oracle_identity_holds",
          all(item["evaluator_report"]["identities"]["all_hold"]
              for item in results["cells"] if item["status"] == fm.STATUS_OK))
    cell_results = [{"cell_id": item["cell_id"], "representation": item["key"]["representation"],
                     "map": item["key"]["map"], "label_budget": item["key"]["label_budget"],
                     "seed": item["key"]["seed"], "status": item["status"],
                     "readout_mse": None if item["readout"] is None else item["readout"]["mse"]}
                    for item in results["cells"]]
    check("selftest::no_cuda_was_initialised_and_threads_were_one",
          not torch.cuda.is_initialized() and torch.get_num_threads() == 1)
    return {"checks": checks, "cell_results": cell_results, "matrix_index_sha256": verified["index"]["index_sha256"],
            "exports": exported, "pooled_caches": pooled_files}


def selftest(output_dir, seed=0):
    """Export from a real tiny random CPU DPLM, pool, and run ONE small matrix; check what must hold.

    Everything is written below a directory this call creates (``mkdtemp`` under ``output_dir``,
    never an existing one); nothing existing is deleted or reused. CPU threads are pinned to one
    and restored; CUDA is never initialised. The report lists every check and each matrix cell's
    actual readout MSE side by side: no cell, representation, layer, pooling or readout is
    selected or ranked, and no win is asserted. Returns the report (also written beside the data);
    ``passed`` is true only if every named check holds. It qualifies the machinery, not biology."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="native_export_selftest_", dir=out))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        body = _selftest_in(work, seed)
    finally:
        torch.set_num_threads(previous)
    report = {
        "schema": SELFTEST_SCHEMA, "seed": seed, "work_directory": str(work),
        "passed": bool(all(item["holds"] for item in body["checks"])),
        "checks": body["checks"], "cell_results": body["cell_results"],
        "matrix_index_sha256": body["matrix_index_sha256"], "exports": body["exports"],
        "pooled_caches": body["pooled_caches"], "claims": dict(CLAIMS),
        "note": ("synthetic qualification of the extraction/cache/pooling machinery on a tiny "
                 "RANDOMLY initialised CPU model. The per-cell MSEs are listed side by side; no "
                 "cell is selected, ranked or declared a winner, and nothing here is evidence "
                 "about a pretrained model or biology.")}
    fb.write_json(work / SELFTEST_REPORT_NAME, report)
    return report
