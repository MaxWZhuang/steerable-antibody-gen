"""Build D: a shared-parameter functional-transfer experiment on frozen native token caches.

What this is. A real learner, not a specification. A small student (one learned projection of the
FROZEN native token features plus two attention blocks) is trained on measured source function and,
in some arms, on exact computational answers about the same sequences (executable SET/SWAP programs,
``functional_composition``). The source-adapted FUNCTION-query features are then frozen and handed to
the existing frozen-feature matrix (``functional_matrix``) beside the unadapted native control, where
fresh readouts are fitted at identical nested label budgets BEFORE any target evaluation outcome is
opened. The research question is whether verified compositional supervision improves data efficiency
on independently evaluated function. A negative or inconclusive result is a valid result; nothing in
the code or the tests requires the composed arm to win. What it is not: a pretrained-model loader (the
caller supplies the native token cache), a biological claim (the pilot's functions are computational),
or evidence that auxiliary accuracy implies functional transfer.

Five arms (same initial functional pathway, labels, batch tape, update count and loss definition):
  F  functional only
  C  shared functional + composed auxiliary (ONE joint Adam step per macrostep on
     functional_mean + lambda * auxiliary_mean; no extra sequential update)
  S  separate functional and composed adapters (independent auxiliary optimizer, clipping and RNG; its
     functional pathway and optimizer match F EXACTLY after every macrostep)
  Q  shared functional + auxiliary targets shuffled ONCE within (output type, template, dependency)
     strata and frozen in the tape (changed VALUES are measured per stratum; a stratum whose shuffle
     would only rename classes is reported unsupported, never counted as corrupted)
  R  shared functional + direct-read/direct-relation supervision: the teacher resolves each composed
     query to original-coordinate operands and/or literals; the student receives those simpler
     instructions with the SAME features, teacher answers, output types and answer counts as C
An optional continued-F control (``continued_f_extra_steps``) discloses its extra functional
presentations; without it no compute superiority is claimed.

Protocol. The accepted v1 protocol is dropout 0 and evaluation-mode forwards WITH gradients, for the
training step and the diagnostic probe alike; a nonzero dropout is refused at the configuration
boundary (stochastic replay is not implemented).

What the student sees. ONLY :class:`StudentBatch`: original native feature tensors, valid-token masks,
token role/chain/residue-index metadata and the typed program/query tensors. Never a sequence string,
a native residue id, a candidate/group/split id, a teacher trace, a transformed sequence, a teacher
record, or (except for R's own simpler instruction) a resolved origin. Teacher records
(``functional_composition.TeacherEvent``) are a different type and never cross that boundary. Source
and auxiliary rows may have different chain lengths: the coordinate space is sized over the declared
input union, every event is validated against its own row's layout.

What a run consumes is owned by the run. ``ArmRun`` verifies the context's arrays against the digests
recorded when it was built and the seed's initial state against its digest (before any model is cloned
from it), then trains from a verified independent snapshot of every small array and of its seed's tape,
so a later write to the shared context cannot reach it. A copied numpy array can have its write flag
re-enabled, so EVERY update first re-verifies the tape, the run-owned arrays and the token-table rows of
exactly that step's batches (per-row digests of the offsets and of the dtype, shape and bytes of every
consumed slice, so an array reinterpreted in place is refused too; the whole large table is not hashed
per forward), and the whole shared table is re-verified at every resume, checkpoint and adapter
publication boundary. An input that no longer matches is refused BEFORE any model, optimizer, cursor or
accounting changes. The features stage verifies the whole in-memory context (token table, small arrays,
event table, tapes) on entry, before it writes or adopts anything: the token roles/chains/indices the
adapted model reads are derived in memory and are not covered by the native cache's own verifier.
The ordinary dicts that select rows and roles are bound too. ``TokenTable.position`` is an owned read-only
map that must equal its ids' exact row order, and ``Cohorts.roles``/``position`` (plain dicts this module
cannot freeze) must still equal the validated manifest, its recorded ``cohort_sha256`` and the table's row
order. Every public stage's ``preflight`` checks this first, in memory and without hashing the features, so
a completed stage is not adopted and a new one not published over a map that no longer matches.

Stage order (file-based, resumable; ``TransferRun``):
  prepare   validate the cohorts against the cache; freeze the event table and per-seed tapes
  train     every (arm, seed): checkpointed at optimizer-update boundaries; every adapter is FROZEN
            (one hashed file each) and scored on SOURCE-DEV before anything else happens
  features  extract FUNCTION-query features of the EXACT sorted target_train + target_eval roster from
            every frozen adapter; pool the native control; write ordinary feature caches
  freeze    ONE combined matrix over native + every arm x source seed (all acquisition seeds, budgets,
            priors); freeze, verify; the target-evaluation outcomes are NOT opened
  evaluate  only now are the evaluation outcomes opened (``fm.evaluate_matrix`` on the verified freeze)
  panels    auxiliary panels (seen/new sequence x seen/held-out dependency; operator deletion/order
            controls; RES vs primary/secondary EQ) after all freezes, with no tuning
  report    one JSON report; negative and inconclusive results are retained
Target outcomes never enter adapter fitting, tape construction, shuffle selection or initialisation.
The pre-declared comparisons are the DIAGONAL (source_seed == acquisition_seed); every off-diagonal
cell is descriptive and none of the nine source/acquisition combinations, the rows or the bootstrap
replicates is an independent population replication.

One bound outcome version per stage. When a stage first reads an outcome table (source-dev in
``train``, target-train in ``freeze``, target-eval in ``evaluate``) its canonical identity (sorted ids,
the numeric values, the declared endpoint) is bound in ``stage_state.json`` BEFORE any dependent artifact
is committed, and every continuation, adoption and re-run is compared with it AND with the historical
record the frozen artifacts carry (the matrix index records the target-training identity, the matrix
report the evaluation identity). The binding is SEMANTIC: the same ids and values in another row order
are the same version; any other change is refused and the existing artifacts are never relabelled.
Output that lacks such a binding cannot be continued: use a fresh output directory.

Owned publication. Every file this module writes is staged through a uniquely named sibling that this
call creates exclusively and then replaces; only a staging file this call created is ever removed, an
existing unknown file is never truncated or deleted, and every predictable destination (final names and
the legacy ``.tmp`` names) is checked against the known input paths (resolved path, hard link,
symlink) before anything is written.

CLI: ``scripts/functional_shared_transfer.py`` (inspect, run, synthetic, selftest, pilot).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import time
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import functional_baseline as fb
from . import functional_composition as fc
from . import functional_matrix as fm
from . import functional_native_export as fne
from .her2_runtime import require, sha256 as file_sha256

CONFIG_SCHEMA = "functional-shared-transfer-config/1"
CHECKPOINT_SCHEMA = "functional-shared-transfer-checkpoint/1"
ADAPTER_SCHEMA = "functional-shared-transfer-adapter/1"
STATE_SCHEMA = "functional-shared-transfer-state/1"
REPORT_SCHEMA = "functional-shared-transfer-report/1"
SELFTEST_SCHEMA = "functional-shared-transfer-selftest/1"
REPORT_NAME = "functional_shared_transfer_report.json"
SELFTEST_REPORT_NAME = "shared_transfer_selftest_report.json"

ARMS = ("F", "C", "S", "Q", "R")
ARM_CONTINUED = "FX"
SHARED_ARMS = ("C", "Q", "R")
AUX_ARMS = ("C", "S", "Q", "R")
NATIVE_REPRESENTATION = "native"
LINEAR_MAP, RBF_MAP = "linear", "rbf_median"
STAGES = ("prepare", "train", "features", "freeze", "evaluate", "panels", "report")
HEAD_PREFIXES = ("func_head", "res_head", "eq_head")
N_ROLES = 4
COMPUTATIONAL_ASSAY = "computational/SET-SWAP-RES-EQ/v1 exact teacher answers"
COMPUTATIONAL_TARGET = "composition teacher over source auxiliary sequences (not a measured outcome)"
FRESH_OUTPUT = "use a fresh output directory (nothing here is relabelled or overwritten)"

CLAIMS = {
    "scope": ("A shared-parameter transfer experiment on frozen native token caches. It is not a "
              "pretrained-model loader, not a biological claim and not evidence that composition "
              "helps; a negative or inconclusive result is a valid result and no test requires the "
              "composed arm to win."),
    "units": ("Functional loss is the mean squared error in squared source-assay units; the "
              "auxiliary loss is (sum of residue cross-entropy + sum of equality BCE) / number of "
              "auxiliary rows, in nats. Both are weighted into the shared arms as "
              "functional_mean + lambda * auxiliary_mean with a fixed lambda and endpoint. No "
              "cross-assay z-scoring and no pooled 'functional' outcome without a shared endpoint."),
    "provenance": ("A source-adapted representation is NOT label-free. Measured source labels, "
                   "derived exact answers, unique auxiliary sequences and repeated presentations are "
                   "different units and are counted separately. The functional bias is initialised "
                   "from the mean of ALL source-training labels, so every measured source label is "
                   "consumed by initialisation before the first batch; that is counted separately "
                   "from direct functional presentations and the union is the total source "
                   "supervision. Native history that is unknown stays unknown; the known new costs "
                   "are recorded in the derived feature schema."),
    "interpretation": [
        "auxiliary accuracy alone does not support functional transfer",
        "beating the shuffled-target arm (Q) alone does not show positive transfer",
        "matching the direct-supervision arm (R) removes composition-specific attribution",
        "matching continued functional-only training (FX) removes compute-efficiency superiority; "
        "without FX no compute superiority is claimed",
        "shared gradients establish wiring, not a shared algorithm",
        "source preservation is SOURCE-DEV functional error versus F after source training; "
        "unchanged frozen weights after target fitting are only an integrity check",
        "negative transfer and inconclusive results are retained"],
    "panel": ("Four target evaluation groups and three paired source seeds are a small finite-panel "
              "screen. The pre-declared comparisons are source_seed == acquisition_seed; all "
              "off-diagonal cells are descriptive. Rows, the nine source/acquisition combinations "
              "and bootstrap replicates are not independent population replications."),
    "compute": ("Equal functional presentation counts are not equal total compute: shared and "
                "separate arms also run auxiliary forwards, and S trains a second adapter."),
    "qualification": ("A CPU result on the synthetic pilot qualifies the implementation. It is not a "
                      "biological result and not an affinity claim."),
}


# ---------------------------------------------------------------------------
# state hashing (plain numeric content, never pickled objects)
# ---------------------------------------------------------------------------

def _feed(hasher, value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        hasher.update(f"T|{array.dtype}|{tuple(array.shape)}|".encode("ascii"))
        hasher.update(array.numpy().tobytes())
    elif isinstance(value, dict):
        hasher.update(b"D{")
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            hasher.update(f"K|{type(key).__name__}|{key!s}|".encode("utf-8"))
            _feed(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, (list, tuple)):
        hasher.update(b"L[")
        for item in value:
            _feed(hasher, item)
        hasher.update(b"]")
    elif value is None:
        hasher.update(b"N")
    elif isinstance(value, (bool, int, float, str)):
        hasher.update(f"{type(value).__name__}|{value!r}|".encode("utf-8"))
    else:
        raise ValueError(f"a state entry of type {type(value).__name__} cannot be hashed; only "
                         "tensors and plain containers are supported")


def tensor_state_digest(value):
    """SHA-256 of the numeric content (dtype, shape, raw bytes) of nested dicts/lists of tensors."""
    hasher = hashlib.sha256()
    _feed(hasher, value)
    return hasher.hexdigest()


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value


def _array_digest(array):
    """SHA-256 of a numpy array's dtype, shape and bytes."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode("ascii")
                          + array.tobytes()).hexdigest()


def answer_label_id(event):
    """A compact, unique identity of one computational answer: candidate + a digest of the CANONICAL
    (program, query) id (recomputed, never the stored string, so an alias cannot mint a second answer).
    It is NOT the candidate id reused for every derived answer."""
    canonical = fc.event_id(event.candidate_id, event.program, event.query)
    return f"{event.candidate_id}#{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:10]}"


def _code_sha256():
    return fb.digest({"functional_composition": file_sha256(Path(fc.__file__)),
                      "functional_shared_transfer": file_sha256(Path(__file__))})


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = (
    "steps", "batch", "aux_batch", "instances_per_sequence", "lam", "lr", "clip_norm", "width",
    "heads", "ffn", "blocks", "dropout", "arms", "source_seeds", "acquisition_seeds",
    "label_budgets", "tape_seed", "continued_f_extra_steps", "checkpoint_every", "probe_steps",
    "panel_instances", "panel_sequences", "include_rbf", "ridge_alpha", "tau",
    "bootstrap_replicates", "extraction_batch")


def _unique_ints(values, name, *, minimum):
    require(isinstance(values, (list, tuple)) and values, f"config.{name} must be a non-empty list")
    out = tuple(fb._integer(item, f"config.{name} entry", minimum=minimum) for item in values)
    require(len(set(out)) == len(out), f"config.{name} repeats a value")
    return out


@dataclass(frozen=True)
class TransferConfig:
    """Every fixed choice. Defaults are the bounded pilot. Nothing is selected from target data:
    ``lam`` and the endpoint are fixed in advance (lambda 0.1 weights the auxiliary mean)."""

    steps: int = 256
    batch: int = 32
    aux_batch: int = 0                      # 0 = same as batch
    instances_per_sequence: int = 4         # x 4 events = 16 auxiliary answers per sequence
    lam: float = 0.1
    lr: float = 2e-3
    clip_norm: float = 1.0
    width: int = 64
    heads: int = 4
    ffn: int = 128
    blocks: int = 2
    dropout: float = 0.0
    arms: tuple = ARMS
    source_seeds: tuple = (0, 1, 2)
    acquisition_seeds: tuple = (0, 1, 2)
    label_budgets: tuple = (16, 64)
    tape_seed: int = 0
    continued_f_extra_steps: int = 0
    checkpoint_every: int = 64
    probe_steps: tuple = ()
    panel_instances: int = 2
    panel_sequences: int = 64
    include_rbf: bool = False
    ridge_alpha: float = 1.0
    tau: float = 1.0
    bootstrap_replicates: int = 200
    extraction_batch: int = 64

    def __post_init__(self):
        def put(name, value):
            object.__setattr__(self, name, value)
        for name, minimum in (("steps", 1), ("batch", 1), ("aux_batch", 0),
                              ("instances_per_sequence", 1), ("width", 1), ("heads", 1),
                              ("ffn", 1), ("blocks", 1), ("tape_seed", 0),
                              ("continued_f_extra_steps", 0), ("checkpoint_every", 1),
                              ("panel_instances", 1), ("panel_sequences", 1),
                              ("bootstrap_replicates", 10), ("extraction_batch", 1)):
            put(name, fb._integer(getattr(self, name), f"config.{name}", minimum=minimum))
        for name in ("lam", "lr", "clip_norm", "ridge_alpha", "tau"):
            put(name, fb._real(getattr(self, name), f"config.{name}", positive=True))
        put("dropout", fb._real(self.dropout, "config.dropout"))
        require(0.0 <= self.dropout < 1.0, "config.dropout must be in [0, 1)")
        require(self.dropout == 0.0,
                "config.dropout must be 0: the accepted v1 protocol is dropout 0 with evaluation-mode "
                "forwards WITH gradients for training and diagnostics alike; stochastic dropout "
                "replay is not implemented")
        require(isinstance(self.include_rbf, bool), "config.include_rbf must be true or false")
        require(self.width % self.heads == 0, "config.width must be divisible by config.heads")
        arms = tuple(self.arms)
        require(arms and all(arm in ARMS for arm in arms) and len(set(arms)) == len(arms),
                f"config.arms must be distinct members of {ARMS}")
        require("F" in arms, "config.arms must include F: it is the paired reference of every "
                             "comparison")
        put("arms", tuple(arm for arm in ARMS if arm in arms))
        put("source_seeds", _unique_ints(self.source_seeds, "source_seeds", minimum=0))
        put("acquisition_seeds", _unique_ints(self.acquisition_seeds, "acquisition_seeds",
                                              minimum=0))
        put("label_budgets", tuple(sorted(_unique_ints(self.label_budgets, "label_budgets",
                                                       minimum=1))))
        probes = tuple(fb._integer(item, "config.probe_steps entry", minimum=0)
                       for item in self.probe_steps)
        require(all(item < self.steps + self.continued_f_extra_steps for item in probes),
                "config.probe_steps must be macrosteps below the total step count")
        put("probe_steps", tuple(sorted(set(probes))))

    @classmethod
    def from_document(cls, document):
        fm._keys(document, (), _CONFIG_FIELDS, label="transfer config")
        values = dict(document)
        for name in ("arms", "source_seeds", "acquisition_seeds", "label_budgets", "probe_steps"):
            if name in values:
                require(isinstance(values[name], (list, tuple)), f"config.{name} must be a list")
                values[name] = tuple(values[name])
        return cls(**values)

    @property
    def aux_batch_size(self):
        return self.aux_batch or self.batch

    def resolved_probe_steps(self):
        if self.probe_steps:
            return tuple(self.probe_steps)
        return tuple(sorted({0, self.steps // 2, self.steps - 1}))

    def arm_list(self):
        """Every trained arm, the optional continued-F control last."""
        return self.arms + ((ARM_CONTINUED,) if self.continued_f_extra_steps else ())

    def total_steps(self, arm):
        return self.steps + (self.continued_f_extra_steps if arm == ARM_CONTINUED else 0)

    def document(self):
        return {name: (list(getattr(self, name)) if isinstance(getattr(self, name), tuple)
                       else getattr(self, name)) for name in _CONFIG_FIELDS}

    def training_document(self):
        """What a training run's result depends on (not the arm list, seeds or analysis settings)."""
        return {"steps": self.steps, "batch": self.batch, "aux_batch": self.aux_batch_size,
                "instances_per_sequence": self.instances_per_sequence, "lam": self.lam,
                "lr": self.lr, "clip_norm": self.clip_norm, "width": self.width,
                "heads": self.heads, "ffn": self.ffn, "blocks": self.blocks,
                "dropout": self.dropout, "tape_seed": self.tape_seed,
                "continued_f_extra_steps": self.continued_f_extra_steps,
                "probe_steps": list(self.resolved_probe_steps())}

    def identity_document(self):
        out = self.document()
        out.pop("checkpoint_every")
        return out


def rep_id(arm, seed):
    return f"{arm}_s{int(seed)}"


# ---------------------------------------------------------------------------
# owned publication: unique exclusive staging, protected inputs, never delete an unknown file
# ---------------------------------------------------------------------------

def _protected_items(protected):
    if not protected:
        return {}
    if isinstance(protected, dict):
        return {str(label): Path(path) for label, path in protected.items()}
    return {f"protected input #{number} ({path})": Path(path)
            for number, path in enumerate(protected)}


def refuse_protected(protected, outputs):
    """Refuse, BEFORE anything is written, an output that resolves to a protected input.

    ``protected`` maps a label to a known input path (or is a list of paths); ``outputs`` maps a label
    to a destination. Both the final name and the legacy ``<name>.tmp`` staging name of every
    destination are checked, by resolved path (symlinks, ``..``, letter case) and by identity for
    existing files (hard links, junctions)."""
    items = _protected_items(protected)
    if not items:
        return
    targets = {}
    for label, path in outputs.items():
        targets[label] = Path(path)
        targets[f"{label} (staging name)"] = fb._temporary_path(path)
    fb._refuse_path_aliases(items, targets)


def _open_staging(final, *, text):
    """Create a UNIQUE sibling of ``final`` exclusively (a name this call invents and an existing file
    can never be truncated through). Returns ``(path, handle)``."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    require(not final.is_dir(), f"{final} is a directory")
    for _ in range(16):
        staging = final.with_name(f".{final.name}.{secrets.token_hex(6)}.staging")
        try:
            handle = (open(staging, "x", encoding="utf-8") if text else open(staging, "xb"))
        except FileExistsError:
            continue
        return staging, handle
    raise ValueError(f"could not create a unique staging file next to {final}")


def _publish(final, write, *, text):
    """Stage through an exclusively created unique file, then replace ``final``. Only the staging file
    THIS call created is removed on failure; ``final`` is replaced atomically or left as it was."""
    final = Path(final)
    staging, handle = _open_staging(final, text=text)
    try:
        with handle:
            write(handle)
        os.replace(staging, final)
    except BaseException:
        if os.path.lexists(staging):
            try:
                staging.unlink()
            except OSError:
                pass
        raise


def _json_text(document):
    return json.dumps(fb._builtin(document), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json_owned(path, document, *, protected=()):
    """Strict JSON through owned staging, after the destination and its legacy staging name were checked
    against the protected inputs. Replaces an existing ``path`` (callers decide whether that is allowed)."""
    path = Path(path)
    refuse_protected(protected, {f"output {path.name}": path})
    text = _json_text(document)
    _publish(path, lambda handle: handle.write(text), text=True)
    return path


def write_text_owned(path, text, *, protected=()):
    path = Path(path)
    refuse_protected(protected, {f"output {path.name}": path})
    _publish(path, lambda handle: handle.write(text), text=True)
    return path


def _load_payload_digest(path):
    """The tensor-content digest of a weights file, or ``None`` if it cannot be read as one."""
    try:
        return tensor_state_digest(torch.load(path, map_location="cpu", weights_only=True))
    except Exception:
        return None


_WEIGHTS_NAME = re.compile(r"^ckpt_\d{6}\.pt$")


# ---------------------------------------------------------------------------
# student-visible token data (no residues) and the typed student batch
# ---------------------------------------------------------------------------

_ROLE_OF_CODE = {"S": 0, "E": 1, "P": 2, "H": 3, "L": 3}
_CHAIN_OF_CODE = {"S": 0, "E": 0, "P": 0, "H": 1, "L": 2}


def _readonly(array):
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class TokenTable:
    """What the student may know about a cached row: its frozen native features and, per token, a
    role (start/end/separator/residue), a chain and a residue index. There is NO residue identity,
    native id or sequence string here, and no label. ``ids``/``position`` are loader-side only.

    ``ids`` fixes the exact original row order and ``position`` is its id-to-row map. The table owns
    ``position`` as a read-only copy (a plain dict handed in is copied, never kept) and refuses a map that
    is not exactly ``{id: row}`` for ``ids``: a swapped entry would emit one candidate's tokens under
    another candidate's id, and so under its label. :meth:`verify_map` re-checks that cheaply (no
    feature bytes are read)."""

    ids: tuple
    position: object
    offsets: object
    features: object
    role: object
    chain: object
    index: object
    hidden: int
    max_tokens: int
    max_heavy: int
    max_light: int
    cache_sha256: str

    def __post_init__(self):
        ids = tuple(self.ids)
        require(len(set(ids)) == len(ids), "a token table's ids must be unique")
        canonical = {cid: row for row, cid in enumerate(ids)}
        require(isinstance(self.position, Mapping) and dict(self.position) == canonical,
                "a token table's id-to-row map must be exactly the position of each id in its ids")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "position", types.MappingProxyType(canonical))

    def verify_map(self):
        """Refuse an id-to-row map that is no longer exactly the position of each id in ``ids``."""
        position, ids = self.position, self.ids
        require(isinstance(position, Mapping) and len(position) == len(ids)
                and all(position.get(cid) == row for row, cid in enumerate(ids)),
                "the token table's id-to-row map no longer matches its ids: the map that selects "
                "which cached row each candidate id reads was modified after the table was built, "
                "so an id could be joined to another candidate's tokens. Nothing was trained or "
                "written. Rebuild the context")
        return True

    def length(self, row):
        return int(self.offsets[row + 1] - self.offsets[row])


def verify_cohort_maps(cohorts, table):
    """Refuse cohorts whose role or row maps no longer agree with what validated them. The manifest must
    still hash to the ``cohort_sha256`` it was validated with, the resolved ``roles`` must equal that
    manifest's sorted roles exactly, and the cohort row map must equal the token table's, which is the
    verified cache order. ``Cohorts.roles`` and ``position`` are ordinary dicts that this module cannot
    make immutable, so they are bound here, at every boundary that consumes them. Cost: one hash of the
    ids, never of the feature bytes."""
    manifest = cohorts.manifest
    try:
        current = manifest.sha256()
    except (AttributeError, KeyError, TypeError, ValueError):
        current = None
    require(current is not None and current == cohorts.cohort_sha256,
            "the cohort manifest no longer matches the cohort_sha256 the cohorts were validated "
            "with: a role or declaration was modified after validation. Nothing was trained or "
            "written. Reopen the pipeline from the cohort manifest file")
    roles = cohorts.roles
    resolved = {role: tuple(sorted(manifest.roles[role])) for role in fc.ROLES}
    require(isinstance(roles, Mapping) and set(roles) == set(fc.ROLES)
            and all(tuple(roles[role]) == resolved[role] for role in fc.ROLES),
            "the cohort roles no longer agree with the validated manifest: a member was added, "
            "removed or replaced after validation, so a stage would publish or score a different "
            "roster under the recorded cohort identity. Nothing was trained or written")
    require(isinstance(cohorts.position, Mapping) and dict(cohorts.position) == dict(table.position),
            "the cohort row map no longer agrees with the token cache order: it selects the rows "
            "every role reads. Nothing was trained or written")
    return True


def build_token_table(cache):
    """A student-visible view of a verified token cache. Roles, chains and residue indices come from the
    cache's role strings (``S``/``E``/``P`` specials, ``H``/``L`` residues); the native ids in the token
    maps are never read."""
    require(isinstance(cache, fne.TokenCache), "build_token_table needs a TokenCache")
    cache.verify()
    total = int(cache.features.shape[0])
    role = np.zeros(total, dtype=np.int64)
    chain = np.zeros(total, dtype=np.int64)
    index = np.full(total, -1, dtype=np.int64)
    heavy_lengths, light_lengths = [], []
    for row, encoded in enumerate(cache.token_maps):
        start, end = int(cache.offsets[row]), int(cache.offsets[row + 1])
        codes = encoded["roles"]
        require(end - start == len(codes), "the token map and the offsets disagree")
        counters = {"H": 0, "L": 0}
        for position, code in enumerate(codes):
            role[start + position] = _ROLE_OF_CODE[code]
            chain[start + position] = _CHAIN_OF_CODE[code]
            if code in counters:
                index[start + position] = counters[code]
                counters[code] += 1
        heavy_lengths.append(counters["H"])
        light_lengths.append(counters["L"])
    return TokenTable(
        ids=tuple(cache.ids), position={cid: i for i, cid in enumerate(cache.ids)},
        offsets=_readonly(cache.offsets), features=cache.features, role=_readonly(role),
        chain=_readonly(chain), index=_readonly(index), hidden=int(cache.hidden_size),
        max_tokens=int(max(cache.lengths())), max_heavy=int(max(heavy_lengths)),
        max_light=int(max(light_lengths)), cache_sha256=cache.cache_sha256)


@dataclass(frozen=True)
class StudentBatch:
    """The ONLY thing the student's forward receives. Tensors only: native features, valid-token
    masks, token role/chain/residue-index metadata and typed instruction tensors
    ``(kind, coord_a, coord_b, symbol_s, symbol_t)`` with an instruction mask. Padding is derived from
    real lengths, never from feature values."""

    features: torch.Tensor
    token_mask: torch.Tensor
    token_role: torch.Tensor
    token_chain: torch.Tensor
    token_index: torch.Tensor
    instr_kind: torch.Tensor
    instr_a: torch.Tensor
    instr_b: torch.Tensor
    instr_s: torch.Tensor
    instr_t: torch.Tensor
    instr_mask: torch.Tensor

    def validate(self, hidden=None):
        for field in dataclasses.fields(self):
            require(isinstance(getattr(self, field.name), torch.Tensor),
                    f"StudentBatch.{field.name} must be a tensor")
        count, tokens = self.token_mask.shape
        require(self.features.dtype == torch.float32 and self.features.shape[:2] == (count, tokens)
                and (hidden is None or self.features.shape[2] == hidden),
                "StudentBatch.features must be float32 (batch, tokens, hidden)")
        require(self.token_mask.dtype == torch.bool and self.instr_mask.dtype == torch.bool,
                "StudentBatch masks must be boolean")
        for name in ("token_role", "token_chain", "token_index"):
            tensor = getattr(self, name)
            require(tensor.dtype == torch.long and tensor.shape == (count, tokens),
                    f"StudentBatch.{name} must be long (batch, tokens)")
        instructions = self.instr_mask.shape[1]
        for name in ("instr_kind", "instr_a", "instr_b", "instr_s", "instr_t"):
            tensor = getattr(self, name)
            require(tensor.dtype == torch.long and tensor.shape == (count, instructions),
                    f"StudentBatch.{name} must be long (batch, instructions)")
        require(bool(self.instr_mask.any(dim=1).all()) and bool(self.token_mask.any(dim=1).all()),
                "every row needs at least one instruction token and one native token")
        return True


STUDENT_FIELDS = tuple(field.name for field in dataclasses.fields(StudentBatch))


def pad_instruction(rows):
    """``((MAX_INSTRUCTIONS, 5) int array, length)`` for a tuple of instruction rows."""
    require(1 <= len(rows) <= fc.MAX_INSTRUCTIONS, "an instruction has 1 to 3 tokens")
    out = np.full((fc.MAX_INSTRUCTIONS, len(fc.INSTRUCTION_FIELDS)), fc.NONE_FIELD, dtype=np.int64)
    out[:, 0] = fc.KIND_PAD
    for position, row in enumerate(rows):
        out[position] = row
    return out, len(rows)


FUNCTION_ARRAY, FUNCTION_LENGTH = pad_instruction(fc.FUNCTION_INSTRUCTION)


def make_student_batch(table, rows, instructions, lengths, *, token_pad=None):
    """Pad ``rows`` (table row positions) to ``token_pad`` native tokens (default: the table's longest
    row, the same for every arm) and attach ``instructions`` ``(B, 3, 5)`` with real ``lengths``."""
    rows = np.asarray(rows, dtype=np.int64)
    count = int(rows.shape[0])
    pad = int(token_pad or table.max_tokens)
    features = np.zeros((count, pad, table.hidden), dtype=np.float32)
    token_mask = np.zeros((count, pad), dtype=bool)
    role = np.zeros((count, pad), dtype=np.int64)
    chain = np.zeros((count, pad), dtype=np.int64)
    index = np.full((count, pad), -1, dtype=np.int64)
    for slot, row in enumerate(rows.tolist()):
        start, end = int(table.offsets[row]), int(table.offsets[row + 1])
        size = end - start
        require(size <= pad, f"a row has {size} native tokens but the padded width is {pad}; "
                             "nothing is truncated")
        features[slot, :size] = table.features[start:end]
        token_mask[slot, :size] = True
        role[slot, :size] = table.role[start:end]
        chain[slot, :size] = table.chain[start:end]
        index[slot, :size] = table.index[start:end]
    instructions = np.asarray(instructions, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    require(instructions.shape == (count, fc.MAX_INSTRUCTIONS, len(fc.INSTRUCTION_FIELDS))
            and lengths.shape == (count,) and bool((lengths >= 1).all())
            and bool((lengths <= fc.MAX_INSTRUCTIONS).all()),
            "instructions must be (batch, 3, 5) with lengths in 1..3")
    mask = np.arange(fc.MAX_INSTRUCTIONS)[None, :] < lengths[:, None]
    require(bool((instructions[~mask][:, 0] == fc.KIND_PAD).all()),
            "padded instruction slots must carry the PAD kind")

    def tensor(array):
        return torch.from_numpy(np.ascontiguousarray(array))
    return StudentBatch(
        features=tensor(features), token_mask=tensor(token_mask), token_role=tensor(role),
        token_chain=tensor(chain), token_index=tensor(index),
        instr_kind=tensor(instructions[:, :, 0]), instr_a=tensor(instructions[:, :, 1]),
        instr_b=tensor(instructions[:, :, 2]), instr_s=tensor(instructions[:, :, 3]),
        instr_t=tensor(instructions[:, :, 4]), instr_mask=tensor(mask))


def function_batch_for(table, rows, *, token_pad=None):
    """FUNCTION-query batch for table rows (no label, no answer)."""
    count = len(rows)
    instructions = np.broadcast_to(FUNCTION_ARRAY, (count,) + FUNCTION_ARRAY.shape)
    return make_student_batch(table, rows, instructions, np.full(count, FUNCTION_LENGTH),
                              token_pad=token_pad)


# ---------------------------------------------------------------------------
# the student
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudentSpec:
    hidden_in: int
    width: int
    heads: int
    ffn: int
    blocks: int
    dropout: float
    n_symbols: int
    max_heavy: int
    max_light: int

    def document(self):
        return dataclasses.asdict(self)


class _Block(nn.Module):
    """Pre-LayerNorm self-attention + feed-forward. Learned routing: no teacher algorithm."""

    def __init__(self, width, heads, ffn, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, ffn)
        self.down = nn.Linear(ffn, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask):
        batch, length, width = x.shape
        head_dim = width // self.heads
        h = self.ln1(x)

        def split(tensor):
            return tensor.view(batch, length, self.heads, head_dim).transpose(1, 2)
        q, k, v = split(self.query(h)), split(self.key(h)), split(self.value(h))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.drop(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.drop(self.out(context))
        return x + self.drop(self.down(F.gelu(self.up(self.ln2(x)))))


class StudentNet(nn.Module):
    """One learned projection of the native token features + two attention blocks, a typed
    FUNCTION/edit query encoding and three heads (functional numeric, residue categorical, equality
    binary). Coordinate embeddings identify chain/position in BOTH the original tokens and the
    instructions. The readout is the representation at the LAST valid instruction token (the query).
    The FUNCTION pathway never touches the auxiliary heads: ``functional_forward`` is
    ``functional_head(encode(batch))``."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        width, none_coord = spec.width, spec.max_heavy + spec.max_light
        self.proj = nn.Linear(spec.hidden_in, width)
        self.role_emb = nn.Embedding(N_ROLES, width)
        self.coord_emb = nn.Embedding(none_coord + 1, width, padding_idx=none_coord)
        self.kind_emb = nn.Embedding(fc.N_KINDS, width, padding_idx=fc.KIND_PAD)
        self.ipos_emb = nn.Embedding(fc.MAX_INSTRUCTIONS, width)
        self.sym_emb = nn.Embedding(spec.n_symbols + 1, width, padding_idx=spec.n_symbols)
        self.blocks = nn.ModuleList(_Block(width, spec.heads, spec.ffn, spec.dropout)
                                    for _ in range(spec.blocks))
        self.ln_f = nn.LayerNorm(width)
        self.func_head = nn.Linear(width, 1)
        self.res_head = nn.Linear(width, spec.n_symbols)
        self.eq_head = nn.Linear(width, 1)
        for embedding in (self.role_emb, self.coord_emb, self.kind_emb, self.ipos_emb,
                          self.sym_emb):
            nn.init.normal_(embedding.weight, std=0.5)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()

    def _coordinates(self, chain, index):
        heavy, none = self.spec.max_heavy, self.spec.max_heavy + self.spec.max_light
        return torch.where(chain == 1, index,
                           torch.where(chain == 2, index + heavy, torch.full_like(index, none)))

    def encode(self, batch):
        """``(B, width)``: the query token's representation after the final LayerNorm."""
        require(isinstance(batch, StudentBatch), "the student accepts only a StudentBatch")
        batch.validate(self.spec.hidden_in)
        none_coord = self.spec.max_heavy + self.spec.max_light
        none_symbol = self.spec.n_symbols
        token = (self.proj(batch.features) + self.role_emb(batch.token_role)
                 + self.coord_emb(self._coordinates(batch.token_chain, batch.token_index)))
        coord_a = torch.where(batch.instr_a < 0, torch.full_like(batch.instr_a, none_coord),
                              batch.instr_a)
        coord_b = torch.where(batch.instr_b < 0, torch.full_like(batch.instr_b, none_coord),
                              batch.instr_b)
        sym_s = torch.where(batch.instr_s < 0, torch.full_like(batch.instr_s, none_symbol),
                            batch.instr_s)
        sym_t = torch.where(batch.instr_t < 0, torch.full_like(batch.instr_t, none_symbol),
                            batch.instr_t)
        count, slots = batch.instr_kind.shape
        position = torch.arange(slots, device=batch.instr_kind.device)[None, :].expand(count, slots)
        instruction = (self.kind_emb(batch.instr_kind) + self.ipos_emb(position)
                       + self.coord_emb(coord_a) + self.coord_emb(coord_b)
                       + self.sym_emb(sym_s) + self.sym_emb(sym_t))
        hidden = torch.cat([instruction, token], dim=1)
        mask = torch.cat([batch.instr_mask, batch.token_mask], dim=1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.ln_f(hidden)
        last = batch.instr_mask.long().sum(dim=1) - 1
        return hidden[torch.arange(count, device=hidden.device), last]

    def functional_forward(self, batch):
        return self.func_head(self.encode(batch)).squeeze(-1)

    def auxiliary_forward(self, batch):
        features = self.encode(batch)
        return self.res_head(features), self.eq_head(features).squeeze(-1)

    def shared_parameters(self):
        """The encoder tensors both losses reach (every parameter that is not a head)."""
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith(HEAD_PREFIXES)]

    def remove_auxiliary_heads(self):
        self.res_head = None
        self.eq_head = None

    def parameter_counts(self):
        shared = sum(p.numel() for p in self.shared_parameters())
        heads = {name: sum(p.numel() for p in getattr(self, name).parameters())
                 for name in HEAD_PREFIXES if getattr(self, name, None) is not None}
        return {"total": int(sum(p.numel() for p in self.parameters())), "shared_encoder": int(shared),
                **{f"{name}": int(count) for name, count in heads.items()}}


def build_student(spec, *, seed=0):
    """A student built under a forked, seeded generator: the global RNG is neither used nor moved."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        return StudentNet(spec)


def initial_student_state(spec, seed, source_mean):
    """The initial parameters for one source seed (identical across arms): seeded default init, with
    the functional bias set to the SOURCE-train mean (a source-only quantity that consumes EVERY source
    training label; the cost report counts that separately from direct presentations)."""
    net = build_student(spec, seed=fc.stable_int("student-init", seed) % (2 ** 62))
    with torch.no_grad():
        net.func_head.bias.fill_(float(source_mean))
    return {name: value.detach().clone() for name, value in net.state_dict().items()}


def student_from_state(spec, state):
    net = build_student(spec)
    net.load_state_dict(state)
    return net


@dataclass(frozen=True)
class AuxTargets:
    """Labels of an auxiliary batch (kept apart from the student's inputs)."""

    is_equality: torch.Tensor
    codes: torch.Tensor


def functional_loss(prediction, labels):
    """Mean squared error, in squared source-assay units."""
    require(prediction.shape == labels.shape, "prediction and labels must align")
    return F.mse_loss(prediction, labels)


def auxiliary_loss(residue_logits, equality_logit, targets):
    """(sum of residue cross-entropy over RES rows + sum of BCE-with-logits over EQ rows) / rows."""
    count = int(targets.codes.shape[0])
    total = equality_logit.new_zeros(())
    is_eq = targets.is_equality
    if bool((~is_eq).any()):
        total = total + F.cross_entropy(residue_logits[~is_eq], targets.codes[~is_eq],
                                        reduction="sum")
    if bool(is_eq.any()):
        total = total + F.binary_cross_entropy_with_logits(
            equality_logit[is_eq], targets.codes[is_eq].to(equality_logit.dtype), reduction="sum")
    return total / count


@contextlib.contextmanager
def _forked_seed(seed, role, step):
    """A per-(seed, role, step) generator state, restored afterwards: forwards are step-deterministic
    and the functional and auxiliary streams are independent of each other."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(fc.stable_int("forward", seed, role, step) % (2 ** 62))
        yield


@contextlib.contextmanager
def evaluation_with_gradients(*nets):
    """Eval mode WITH gradients (the accepted v1 protocol: dropout is 0 and the training step and the
    diagnostic probe use the SAME mode), modes restored regardless, RNG forked. ``controlled_evaluation``
    disables gradients and is unsuitable here."""
    unique = list({id(net): net for net in nets if net is not None}.values())
    saved = [(module, module.training) for net in unique for module in net.modules()]
    for net in unique:
        net.eval()
    try:
        with torch.enable_grad(), torch.random.fork_rng(devices=[]):
            yield
    finally:
        for module, flag in saved:
            module.training = flag


def _flat(gradients, parameters):
    """One vector from per-parameter gradients; an unused parameter contributes zeros of ITS size, so
    two such vectors over the same parameters always align."""
    parts = [(torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
             for grad, parameter in zip(gradients, parameters)]
    return torch.cat(parts)


def _cosine(left, right):
    left_norm, right_norm = float(left.norm()), float(right.norm())
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(left @ right) / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# the source side: events, tapes and batches
# ---------------------------------------------------------------------------

_CONTEXT_ARRAYS = ("event_rows", "event_codes", "event_is_equality", "event_composed",
                   "event_composed_len", "event_direct", "event_direct_len", "functional_rows",
                   "functional_labels")
_TABLE_ARRAYS = ("offsets", "features", "role", "chain", "index")


def _row_digest(table, row):
    """SHA-256 of ONE cache row as the student reads it: its offsets and the dtype, shape and bytes of its
    feature, role, chain and residue-index slices. The dtype and shape are of the slice actually
    consumed, so an array reinterpreted in place (same bytes, other dtype) is a different row. Cheap
    enough to re-derive for the rows of a batch (the whole table is not)."""
    start, end = int(table.offsets[row]), int(table.offsets[row + 1])
    hasher = hashlib.sha256()
    hasher.update(f"{start}|{end}|".encode("ascii"))
    for array in (table.features, table.role, table.chain, table.index):
        hasher.update(_array_digest(array[start:end]).encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True, eq=False)
class SourceContext:
    """Everything training replays, built once from the verified inputs. Teacher data (events,
    answers) and student data (token table, instruction arrays) are kept apart.

    ``layouts`` is the auxiliary layout source (a ``Layout`` or a ``LayoutMap``) and ``layout`` the
    single Layout when every auxiliary row has one shape (``None`` otherwise). ``live_sha256`` holds the
    digests of every array training consumes, recorded when the context was built;
    :meth:`verify_live` re-derives them from the arrays AS THEY ARE NOW. ``token_row_sha256`` is one digest
    per cache row, so :meth:`verify_rows` can check exactly the rows an update is about to read without
    hashing the whole (large, shared) table per forward."""

    config: object
    cache_sha256: str
    cohorts: object
    table: object
    space: object
    layout: object
    spec: object
    events: object
    event_rows: object
    event_codes: object
    event_is_equality: object
    event_composed: object
    event_composed_len: object
    event_direct: object
    event_direct_len: object
    answer_ids: tuple
    functional_ids: tuple
    functional_rows: object
    functional_labels: object
    source_mean: float
    seed_tapes: dict
    init_states: dict
    init_sha256: dict
    coverage: dict
    balance: dict
    training_sha256: str
    inputs_sha256: str
    code_sha256: str
    layouts: object = None
    live_sha256: object = None
    token_row_sha256: object = None

    def __post_init__(self):
        if self.layouts is None:
            object.__setattr__(self, "layouts", self.layout)
        if self.live_sha256 is None:
            object.__setattr__(self, "live_sha256", self.live_digests())
        if self.token_row_sha256 is None:
            object.__setattr__(self, "token_row_sha256",
                               tuple(_row_digest(self.table, row)
                                     for row in range(len(self.table.ids))))

    def live_digests(self, *, tokens=True):
        """Digests of the arrays as they are now (``tokens=False`` skips the large shared token table)."""
        out = {name: _array_digest(getattr(self, name)) for name in _CONTEXT_ARRAYS}
        if tokens:
            out["token_table"] = fb.digest({name: _array_digest(getattr(self.table, name))
                                            for name in _TABLE_ARRAYS})
        return out

    def verify_live(self, *, tokens=True):
        """Refuse when an array training consumes no longer matches what was built (a re-enabled write
        is not a read-only flag's problem: the content is compared, not the flag)."""
        current = self.live_digests(tokens=tokens)
        changed = sorted(name for name, value in current.items()
                         if value != self.live_sha256.get(name))
        require(not changed,
                f"the source context's {changed} no longer match what the context was built from: "
                "an array that training consumes was modified after the context was created. "
                "Nothing was trained or written. Rebuild the context (its tapes then get a new "
                "identity and cannot resume an old checkpoint)")
        return True

    def verify_rows(self, rows):
        """Refuse when a token-table row that is about to be consumed no longer matches the digest
        recorded when the context was built. The large shared table is not re-hashed per forward; the
        rows a batch actually reads are, so a write to the shared features is caught BEFORE the update
        that would have trained on it."""
        wanted = np.unique(np.asarray(rows, dtype=np.int64))
        changed = [int(row) for row in wanted
                   if _row_digest(self.table, int(row)) != self.token_row_sha256[int(row)]]
        require(not changed,
                f"the shared token table's row(s) {changed[:5]} no longer match what the context was "
                "built from: native features or token metadata that this update reads were modified "
                "after the context was created. Nothing was trained or written. Rebuild the context")
        return True

    def verify_maps(self):
        """The loader-side maps that select rows and roles: the table's id-to-row map against its ids and
        the cohorts against their validated manifest and the table's row order. Cheap (no feature bytes
        are hashed), so it can guard every stage, including ones that never touch the feature rows."""
        self.table.verify_map()
        verify_cohort_maps(self.cohorts, self.table)
        return True

    def verify_all(self):
        """Every consumed array, the event table and every seed's tape, against their digests, and the
        id/role maps (:meth:`verify_maps`)."""
        self.verify_maps()
        self.verify_live()
        self.events.verify()
        for tape in self.seed_tapes.values():
            tape.verify()
        return True

    def snapshot(self, seed):
        """A run-owned copy: independent, read-only copies of every small array training consumes, a
        VERIFIED copy of one seed's tape (:meth:`functional_composition.SeedTape.snapshot` raises if the
        tape's arrays changed) and a VERIFIED private copy of that seed's initial state (checked against
        its recorded digest before the copy is handed out, so a modified state is refused before any
        model is built from it). The large shared token table is referenced, not copied; the run
        re-verifies the rows each update reads (:meth:`verify_rows`)."""
        def own(array):
            out = np.array(array, copy=True)
            out.setflags(write=False)
            return out
        state = _clone(self.init_states[seed])
        require(tensor_state_digest(state) == self.init_sha256[seed],
                f"the initial state of source seed {seed} no longer matches its recorded digest: it "
                "was modified after the context was built. No model was built from it")
        return dataclasses.replace(
            self, **{name: own(getattr(self, name)) for name in _CONTEXT_ARRAYS},
            seed_tapes={int(seed): self.seed_tapes[seed].snapshot()},
            init_states={int(seed): state}, init_sha256={int(seed): self.init_sha256[seed]})


def _stack_instructions(rows_list):
    arrays, lengths = zip(*(pad_instruction(rows) for rows in rows_list))
    return np.stack(arrays), np.array(lengths, dtype=np.int64)


def build_context(cache, cohorts, config, source_train_outcomes):
    """Validate-then-build: the source-only event table (coverage-checked and re-verified against the
    ORIGINAL sequences by the teacher), the instruction arrays, the frozen per-seed tapes and the
    shared initial states. Target outcomes are not an argument and are never read here.

    Auxiliary rows may have different chain lengths: each event is validated and compiled against its
    OWN row's layout, while the student's coordinate space is sized over the whole declared input
    union (light coordinates are offset by the space's heavy size, never by the row's)."""
    require(isinstance(config, TransferConfig), "build_context needs a TransferConfig")
    table = build_token_table(cache)
    verify_cohort_maps(cohorts, table)       # BEFORE any role or row map is consumed below
    layouts = cohorts.layouts if getattr(cohorts, "layouts", None) is not None else cohorts.layout
    space = fc.CoordSpace(table.max_heavy, table.max_light)
    spec = StudentSpec(hidden_in=table.hidden, width=config.width, heads=config.heads,
                       ffn=config.ffn, blocks=config.blocks, dropout=config.dropout,
                       n_symbols=len(layouts.alphabet), max_heavy=table.max_heavy,
                       max_light=table.max_light)
    aux_rows = cohorts.rows(cache, "source_aux")
    events = fc.generate_events(layouts, aux_rows, universe=fc.UNIVERSE_TRAIN,
                                instances_per_sequence=config.instances_per_sequence,
                                seed=config.tape_seed)
    coverage = fc.coverage_report(events)
    require(coverage["complete"],
            "the auxiliary event table does not cover every training requirement for layout "
            f"shape(s) {fc.incomplete_coverage(coverage)}; supply more auxiliary sequences of that "
            "shape, raise instances_per_sequence, or remove those rows from source_aux")
    require(not any(event.held_out for event in events.events),
            "the training universe contains a held-out chained two-swap program")
    fc.verify_events_against(events, aux_rows)
    answer_ids = tuple(answer_label_id(event) for event in events.events)
    require(len(set(answer_ids)) == len(answer_ids), "computational answer ids are not unique")
    composed, composed_len = _stack_instructions(
        [fc.compile_instruction(space, events.layout_for(event.candidate_id), event.program,
                                event.query) for event in events.events])
    direct, direct_len = _stack_instructions(
        [fc.compile_direct(space, events.layout_for(event.candidate_id), event.resolution)
         for event in events.events])
    event_rows = np.array([table.position[event.candidate_id] for event in events.events],
                          dtype=np.int64)
    functional_ids = tuple(cohorts.roles["source_train"])
    labels = np.array([source_train_outcomes[cid] for cid in functional_ids], dtype=np.float64)
    require(bool(np.isfinite(labels).all()), "source training outcomes must be finite numbers")
    used = sorted(set(functional_ids) | set(events.candidate_ids()))
    layouts_sha256 = fc.layouts_sha256(layouts)
    mask_binding = {"padded_tokens": table.max_tokens, "padded_instructions": fc.MAX_INSTRUCTIONS,
                    "coordinate_space": {"heavy": table.max_heavy, "light": table.max_light},
                    "layout_sha256": layouts_sha256,
                    "token_lengths": {cid: table.length(table.position[cid]) for cid in used}}
    seed_tapes = {
        seed: fc.build_seed_tape(
            seed=seed, steps=config.steps,
            functional_steps=config.steps + config.continued_f_extra_steps, batch=config.batch,
            aux_batch=config.aux_batch_size, functional_ids=functional_ids,
            functional_labels=labels, table=events, mask_binding=mask_binding)
        for seed in config.source_seeds}
    source_mean = float(labels.mean())
    init_states = {seed: initial_student_state(spec, seed, source_mean)
                   for seed in config.source_seeds}
    training_sha256 = fb.digest(config.training_document())
    code_sha256 = _code_sha256()
    inputs_sha256 = fb.digest({
        "cache_sha256": cache.cache_sha256, "cohort_sha256": cohorts.cohort_sha256,
        "layouts_sha256": layouts_sha256,
        "source_train_outcomes_sha256": fb.digest({"ids": list(functional_ids),
                                                   "labels": labels.tolist()}),
        "training_config_sha256": training_sha256, "event_table_sha256": events.sha256,
        "student_spec": spec.document(), "code_sha256": code_sha256})
    return SourceContext(
        config=config, cache_sha256=cache.cache_sha256, cohorts=cohorts, table=table, space=space,
        layout=layouts if isinstance(layouts, fc.Layout) else None, spec=spec, events=events,
        event_rows=_readonly(event_rows),
        event_codes=events.codes(), event_is_equality=events.is_equality(),
        event_composed=_readonly(composed), event_composed_len=_readonly(composed_len),
        event_direct=_readonly(direct), event_direct_len=_readonly(direct_len),
        answer_ids=answer_ids, functional_ids=functional_ids,
        functional_rows=_readonly([table.position[cid] for cid in functional_ids]),
        functional_labels=_readonly(labels), source_mean=source_mean, seed_tapes=seed_tapes,
        init_states=init_states,
        init_sha256={seed: tensor_state_digest(state) for seed, state in init_states.items()},
        coverage=coverage, balance=fc.balance_report(events), training_sha256=training_sha256,
        inputs_sha256=inputs_sha256, code_sha256=code_sha256, layouts=layouts)


def functional_batch(ctx, tape, step):
    """The FUNCTION-query batch of one macrostep and its labels (labels are NOT in the batch)."""
    members = tape.functional_batches[step]
    rows = ctx.functional_rows[members]
    batch = function_batch_for(ctx.table, rows)
    labels = torch.from_numpy(tape.functional_labels[members].astype(np.float32))
    return batch, labels


def auxiliary_batch(ctx, tape, arm, step):
    """The auxiliary batch of one macrostep: composed instructions (C, S, Q) or the teacher's resolved
    direct instruction (R); true answers, or Q's frozen shuffled answers."""
    members = tape.aux_batches[step]
    if arm == "R":
        instructions, lengths = ctx.event_direct[members], ctx.event_direct_len[members]
    else:
        instructions, lengths = ctx.event_composed[members], ctx.event_composed_len[members]
    batch = make_student_batch(ctx.table, ctx.event_rows[members], instructions, lengths)
    codes = tape.q_targets[members] if arm == "Q" else ctx.event_codes[members]
    return batch, AuxTargets(is_equality=torch.from_numpy(np.array(ctx.event_is_equality[members])),
                             codes=torch.from_numpy(np.array(codes, dtype=np.int64)))


# ---------------------------------------------------------------------------
# one (arm, seed) training run
# ---------------------------------------------------------------------------

def run_bindings(ctx, arm, seed):
    tape = ctx.seed_tapes[seed]
    return {"arm": arm, "seed": int(seed), "init_sha256": ctx.init_sha256[seed],
            "tape_sha256": tape.sha256, "training_config_sha256": ctx.training_sha256,
            "inputs_sha256": ctx.inputs_sha256, "cache_sha256": ctx.cache_sha256,
            "event_table_sha256": ctx.events.sha256, "code_sha256": ctx.code_sha256}


def _fresh_accounting():
    return {"macrosteps": 0, "forward_calls": {"functional": 0, "auxiliary": 0},
            "backward_calls": 0, "optimizer_steps": 0,
            "rows_forwarded": {"functional": 0, "auxiliary": 0},
            "functional_presentations": 0, "auxiliary_presentations": 0,
            "scalar_targets_presented": {"functional": 0, "auxiliary": 0},
            "probe": {"forward_calls": 0, "gradient_calls": 0}}


def _make_adam(parameters, config):
    return torch.optim.Adam(list(parameters), lr=config.lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0, foreach=False)


def _head_of(name):
    return next((head for head in HEAD_PREFIXES if name.startswith(head)), None)


class ArmRun:
    """One arm trained for one source seed, one macrostep at a time.

    Every arm clones the SAME initial state (separate parameter objects). The functional batch, labels
    and update count are identical across arms. C/Q/R take one joint Adam step on
    ``functional_mean + lam * auxiliary_mean``; S runs the F update unchanged and then a fully
    independent auxiliary update on its own adapter, optimizer, clipping and RNG stream; F (and the
    optional continued-F control) has no auxiliary answers. Forward/backward calls, rows,
    presentations, optimizer steps and CPU time are counted separately.

    The run owns what it consumes: construction verifies the context's arrays, the seed's tape and the
    seed's initial state (before any model is built from it) and then trains from a verified snapshot
    (``self.ctx`` / ``self.tape``), so a later write to the shared context cannot reach it. A copied
    numpy array can have its write flag re-enabled, so every update ALSO re-verifies, before it changes
    a parameter, the optimizer, the cursor or any counter, the tape, the run-owned small arrays and the
    token-table rows of exactly that step's batches (:meth:`_verify_consumed`); ``verify_inputs`` re-checks
    the snapshot and the whole shared token table at the resume, checkpoint and publication boundaries.
    Forwards run in evaluation mode WITH gradients, the v1 protocol (dropout 0), exactly like the
    diagnostic probe.
    """

    def __init__(self, ctx, arm, seed):
        require(arm in ARMS + (ARM_CONTINUED,), f"unknown arm {arm!r}")
        require(seed in ctx.seed_tapes, f"no tape for source seed {seed}")
        ctx.verify_live()                       # refuse BEFORE any model, optimizer or cursor exists
        self.source_ctx = ctx
        self.ctx = ctx.snapshot(seed)           # verified, run-owned copies of what training consumes
        self.arm, self.seed = arm, int(seed)
        self.tape = self.ctx.seed_tapes[seed]
        self.total_steps = self.ctx.config.total_steps(arm)
        self.probe_steps = set(self.ctx.config.resolved_probe_steps())
        init = self.ctx.init_states[seed]
        self.func_net = student_from_state(self.ctx.spec, init)
        self.aux_net = None
        if arm in SHARED_ARMS:
            self.aux_net = self.func_net
        elif arm == "S":
            self.aux_net = student_from_state(self.ctx.spec, init)
        self.func_opt = _make_adam(self.func_net.parameters(), self.ctx.config)
        self.aux_opt = (_make_adam(self.aux_net.parameters(), self.ctx.config)
                        if arm == "S" else None)
        self.cursor = 0
        self.accounting = _fresh_accounting()
        self.log, self.diagnostics = [], []
        self.cpu_seconds, self.probe_seconds = 0.0, 0.0
        self.bindings = run_bindings(self.ctx, arm, seed)

    # --- state ---------------------------------------------------------------------------
    @property
    def run_id(self):
        return rep_id(self.arm, self.seed)

    def verify_inputs(self):
        """Refuse when anything this run consumes changed: the run's own tape snapshot, its array
        snapshot and the shared token table (re-hashed here, at boundaries, never per forward) and the
        recorded initial state. Raises before the caller mutates or publishes anything."""
        self.tape.verify()
        self.ctx.verify_live()
        require(tensor_state_digest(self.source_ctx.init_states[self.seed])
                == self.source_ctx.init_sha256[self.seed],
                f"the initial state of source seed {self.seed} no longer matches its recorded "
                "digest: it was modified after the context was built")
        return True

    def _verify_consumed(self, step, *, functional, auxiliary):
        """Refuse BEFORE an update when an input it is about to consume no longer matches what was
        recorded: the tape, every run-owned small array, and the token-table rows of exactly this step's
        functional and auxiliary batches. The numpy write flag of a copied array can be re-enabled, so
        the CONTENT is compared, and it is compared first: nothing below (parameters, optimizer, cursor,
        accounting, log) has changed yet when this raises."""
        self.tape.verify()
        self.ctx.verify_live(tokens=False)
        rows = []
        if functional:
            rows.append(self.ctx.functional_rows[self.tape.functional_batches[step]])
        if auxiliary:
            rows.append(self.ctx.event_rows[self.tape.aux_batches[step]])
        self.ctx.verify_rows(np.concatenate(rows))

    def state_payload(self):
        nets = {"func": _clone(dict(self.func_net.state_dict()))}
        optimizers = {"func": _clone(self.func_opt.state_dict())}
        if self.arm == "S":
            nets["aux"] = _clone(dict(self.aux_net.state_dict()))
            optimizers["aux"] = _clone(self.aux_opt.state_dict())
        return {"format": CHECKPOINT_SCHEMA, "nets": nets, "optimizers": optimizers}

    def state_sha256(self):
        return tensor_state_digest(self.state_payload())

    def functional_pathway_digest(self):
        """The functional net and its optimizer (the part S must share with F exactly)."""
        return tensor_state_digest({"net": _clone(dict(self.func_net.state_dict())),
                                    "optimizer": _clone(self.func_opt.state_dict())})

    def expected_adam_steps(self, cursor=None):
        """Per-parameter Adam step counts the frozen tape implies after ``cursor`` macrosteps, as
        ``{"func": {name: count}, "aux": {...}}`` (``aux`` only for S). A parameter whose loss never
        reached it has count 0 and NO optimizer entry: the residue head moves only in macrosteps whose
        auxiliary batch has a RES row, the equality head only with an EQ row, the auxiliary heads never
        in F, and S's functional pathway never sees an auxiliary loss."""
        return expected_update_counts(self, self.cursor if cursor is None else cursor)

    # --- one optimizer update ------------------------------------------------------------
    def _snapshot(self):
        return {"func": [p.detach().clone() for p in self.func_net.parameters()],
                "aux": ([p.detach().clone() for p in self.aux_net.parameters()]
                        if self.arm == "S" else None)}

    def _step(self, net, optimizer, loss):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = self.ctx.config.clip_norm
        total = float(torch.nn.utils.clip_grad_norm_(list(net.parameters()), clip, foreach=False))
        require(math.isfinite(total), "a non-finite gradient norm: refusing the update")
        coefficient = min(1.0, clip / (total + 1e-6))
        optimizer.step()
        self.accounting["backward_calls"] += 1
        self.accounting["optimizer_steps"] += 1
        return total, coefficient

    def _probe(self, fbatch, labels, abatch, atargets):
        """Gradient norms/cosines on the SHARED tensors for the batches of this macrostep, at the
        parameters BEFORE the update. ``torch.autograd.grad`` is used so no ``.grad`` is touched;
        modes and RNG are restored; the optimizer is untouched. The probe's forwards are the training
        step's forwards (evaluation mode with gradients), so its losses are the step's losses."""
        lam = self.ctx.config.lam
        nets = (self.func_net, self.aux_net)
        out = {}
        with evaluation_with_gradients(*nets):
            shared = self.func_net.shared_parameters()
            f_loss = functional_loss(self.func_net.functional_forward(fbatch), labels)
            f_grads = torch.autograd.grad(f_loss, shared, allow_unused=True)
            f_flat = _flat(f_grads, shared)
            self.accounting["probe"]["forward_calls"] += 1
            self.accounting["probe"]["gradient_calls"] += 1
            out.update(functional_loss=float(f_loss),
                       functional_grad_norm_shared=float(f_flat.norm()))
            if self.arm in AUX_ARMS:
                aux_shared = self.aux_net.shared_parameters()
                res, eq = self.aux_net.auxiliary_forward(abatch)
                a_loss = auxiliary_loss(res, eq, atargets)
                a_grads = torch.autograd.grad(a_loss, aux_shared, allow_unused=True)
                a_flat = _flat(a_grads, aux_shared)
                self.accounting["probe"]["forward_calls"] += 1
                self.accounting["probe"]["gradient_calls"] += 1
                out.update(auxiliary_loss=float(a_loss),
                           auxiliary_grad_norm_shared=float(a_flat.norm()),
                           auxiliary_weighted_grad_norm_shared=lam * float(a_flat.norm()))
                if self.arm in SHARED_ARMS:
                    out["cosine_functional_auxiliary_shared"] = _cosine(f_flat, a_flat)
                    out["shared_tensors"] = len(shared)
                else:
                    out["cosine_functional_auxiliary_shared"] = None
                    out["cosine_note"] = ("separate adapters: the functional and auxiliary losses "
                                          "reach DIFFERENT parameter tensors, so no cosine exists")
        return out

    def macrostep(self):
        """One macrostep: the functional batch (and the auxiliary batch for C/S/Q/R) at this cursor."""
        ctx, cfg, step = self.ctx, self.ctx.config, self.cursor
        require(step < self.total_steps, "every macrostep of this run is already done")
        self._verify_consumed(step, functional=True, auxiliary=self.arm in AUX_ARMS)
        started = time.process_time()
        fbatch, labels = functional_batch(ctx, self.tape, step)
        abatch = atargets = None
        if self.arm in AUX_ARMS:
            abatch, atargets = auxiliary_batch(ctx, self.tape, self.arm, step)
        probing = step in self.probe_steps
        diagnostic, before, probe_cost = {"step": step}, None, 0.0
        if probing:
            tick = time.process_time()
            diagnostic.update(self._probe(fbatch, labels, abatch, atargets))
            before = self._snapshot()
            probe_cost = time.process_time() - tick
        entry = {"step": step}
        with evaluation_with_gradients(self.func_net, self.aux_net), \
                _forked_seed(self.seed, "functional", step):
            prediction = self.func_net.functional_forward(fbatch)
        self.accounting["forward_calls"]["functional"] += 1
        self.accounting["rows_forwarded"]["functional"] += int(labels.shape[0])
        f_loss = functional_loss(prediction, labels)
        entry["functional_loss"] = float(f_loss)
        if self.arm in SHARED_ARMS:
            with evaluation_with_gradients(self.func_net), \
                    _forked_seed(self.seed, "auxiliary", step):
                res, eq = self.func_net.auxiliary_forward(abatch)
            a_loss = auxiliary_loss(res, eq, atargets)
            entry["auxiliary_loss"] = float(a_loss)
            norm, coefficient = self._step(self.func_net, self.func_opt, f_loss + cfg.lam * a_loss)
            entry.update(grad_norm=norm, clip_coefficient=coefficient)
        else:
            norm, coefficient = self._step(self.func_net, self.func_opt, f_loss)
            entry.update(grad_norm=norm, clip_coefficient=coefficient)
            if self.arm == "S":
                with evaluation_with_gradients(self.aux_net), \
                        _forked_seed(self.seed, "auxiliary", step):
                    res, eq = self.aux_net.auxiliary_forward(abatch)
                a_loss = auxiliary_loss(res, eq, atargets)
                entry["auxiliary_loss"] = float(a_loss)
                a_norm, a_coefficient = self._step(self.aux_net, self.aux_opt, cfg.lam * a_loss)
                entry.update(auxiliary_grad_norm=a_norm, auxiliary_clip_coefficient=a_coefficient)
        if self.arm in AUX_ARMS:
            self.accounting["forward_calls"]["auxiliary"] += 1
            self.accounting["rows_forwarded"]["auxiliary"] += int(atargets.codes.shape[0])
            self.accounting["auxiliary_presentations"] += int(atargets.codes.shape[0])
            self.accounting["scalar_targets_presented"]["auxiliary"] += int(atargets.codes.shape[0])
        self.accounting["functional_presentations"] += int(labels.shape[0])
        self.accounting["scalar_targets_presented"]["functional"] += int(labels.shape[0])
        if probing:
            diagnostic.update(grad_norm_before_clip=entry["grad_norm"],
                              clip_coefficient=entry["clip_coefficient"],
                              clipped=entry["clip_coefficient"] < 1.0)
            if self.arm == "S":
                diagnostic.update(auxiliary_grad_norm_before_clip=entry["auxiliary_grad_norm"],
                                  auxiliary_clip_coefficient=entry["auxiliary_clip_coefficient"])
            diagnostic.update(self._update_norms(before))
            self.diagnostics.append(diagnostic)
            self.probe_seconds += probe_cost
        self.cursor += 1
        self.accounting["macrosteps"] += 1
        self.log.append(entry)
        self.cpu_seconds += (time.process_time() - started) - probe_cost
        return entry

    def _update_norms(self, before):
        def moved(parameters, snapshot):
            return math.sqrt(sum(float(((p.detach() - old) ** 2).sum())
                                 for p, old in zip(parameters, snapshot)))
        shared_names = {id(p) for p in self.func_net.shared_parameters()}
        shared_sq = math.sqrt(sum(float(((p.detach() - old) ** 2).sum())
                                  for p, old in zip(self.func_net.parameters(), before["func"])
                                  if id(p) in shared_names))
        out = {"update_norm_shared": shared_sq,
               "update_norm_functional_net": moved(self.func_net.parameters(), before["func"])}
        if self.arm == "S":
            out["update_norm_auxiliary_net"] = moved(self.aux_net.parameters(), before["aux"])
        return out

    def auxiliary_only_step(self):
        """One update on the auxiliary loss ONLY (a wiring probe for tests): in C it moves the shared
        encoder and therefore the FUNCTION features; in S it moves only the separate auxiliary
        adapter. It advances nothing else (cursor, tape and counters are untouched), so a run that has
        taken one no longer matches its tape and its checkpoints are refused on resume."""
        require(self.arm in AUX_ARMS, f"arm {self.arm} has no auxiliary answers")
        self._verify_consumed(0, functional=False, auxiliary=True)
        abatch, atargets = auxiliary_batch(self.ctx, self.tape, self.arm, 0)
        optimizer = self.aux_opt if self.arm == "S" else self.func_opt
        res, eq = self.aux_net.auxiliary_forward(abatch)
        loss = auxiliary_loss(res, eq, atargets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss)

    def function_features(self, rows, *, token_pad=None):
        return extract_function_features(self.ctx.table, self.func_net, rows, token_pad=token_pad)

    # --- reports ---------------------------------------------------------------------------
    def parameter_report(self):
        def updated(net, optimizer):
            return int(sum(p.numel() for p in net.parameters() if p in optimizer.state))
        report = {"functional_net": self.func_net.parameter_counts(),
                  "functional_net_updated": updated(self.func_net, self.func_opt)}
        total = report["functional_net"]["total"]
        trained = report["functional_net_updated"]
        if self.arm == "S":
            report["auxiliary_net"] = self.aux_net.parameter_counts()
            report["auxiliary_net_updated"] = updated(self.aux_net, self.aux_opt)
            total += report["auxiliary_net"]["total"]
            trained += report["auxiliary_net_updated"]
        report.update(total_parameters=int(total), updated_parameters=int(trained))
        return report

    def cost_report(self):
        ctx, steps = self.ctx, self.cursor
        members = self.tape.functional_batches[:steps]
        accounting = copy.deepcopy(self.accounting)
        # The functional bias is the mean of ALL source-training labels, so initialisation consumes
        # every one of them before the first batch; direct presentations are counted separately and
        # the union is the total measured source supervision (also at cursor 0).
        consumed_by_initialization = set(ctx.functional_ids)
        presented = {ctx.functional_ids[int(i)] for i in np.unique(members)}
        union = consumed_by_initialization | presented
        out = {**accounting, "arm": self.arm, "seed": self.seed, "cursor": steps,
               "unique_measured_source_outcomes": len(union),
               "unique_measured_outcomes_consumed_by_initialization": len(consumed_by_initialization),
               "unique_measured_outcomes_directly_presented": len(presented),
               "measured_outcome_consumption": {
                   "initialization": {
                       "count": len(consumed_by_initialization),
                       "how": ("the functional head's bias is initialised to the mean of ALL "
                               "source-training labels, an aggregate over every one of them, "
                               "consumed before the first batch")},
                   "direct_functional_presentations": {
                       "unique_outcomes": len(presented),
                       "presentations": accounting["functional_presentations"]},
                   "union_total": {"count": len(union),
                                   "rule": "initialisation and direct presentations together"}},
               "functional_passes": accounting["functional_presentations"]
               / max(1, len(ctx.functional_ids)),
               "unique_auxiliary_sequences": 0, "unique_computational_answers": 0,
               "auxiliary_passes": 0.0,
               "elapsed_cpu_seconds": self.cpu_seconds, "probe_cpu_seconds": self.probe_seconds,
               "parameters": self.parameter_report(),
               "total_rows_forwarded": accounting["rows_forwarded"]["functional"]
               + accounting["rows_forwarded"]["auxiliary"],
               "total_forward_calls": accounting["forward_calls"]["functional"]
               + accounting["forward_calls"]["auxiliary"]}
        if self.arm in AUX_ARMS:
            events = np.unique(self.tape.aux_batches[:steps])
            out["unique_computational_answers"] = int(events.size)
            out["unique_auxiliary_sequences"] = int(np.unique(ctx.event_rows[events]).size)
            out["auxiliary_passes"] = accounting["auxiliary_presentations"] / max(1, len(ctx.events))
        out["units"] = ("measured source outcomes (initialisation and direct presentations "
                        "separately, and their union), unique auxiliary sequences, unique "
                        "computational answers, repeated presentations, scalar targets, rows and "
                        "forward CALLS are different units and are never added together")
        if self.arm == ARM_CONTINUED:
            extra = ctx.config.continued_f_extra_steps * ctx.config.batch
            out["continued_f_disclosure"] = (
                f"this control trains {ctx.config.continued_f_extra_steps} extra macrosteps, i.e. "
                f"{extra} extra functional presentations beyond the other arms: it equalises "
                "compute, not presentations. Without it no compute superiority is claimed.")
        return out

    def deterministic_result(self):
        """Everything that must be identical between an uninterrupted and a resumed run (no timing)."""
        return {"arm": self.arm, "seed": self.seed, "cursor": self.cursor,
                "bindings": self.bindings, "log": copy.deepcopy(self.log),
                "diagnostics": copy.deepcopy(self.diagnostics),
                "accounting": copy.deepcopy(self.accounting), "state_sha256": self.state_sha256()}


def expected_update_counts(run, cursor):
    """The Adam step count of every parameter after ``cursor`` macrosteps, derived from the frozen
    auxiliary-type tape (see :meth:`ArmRun.expected_adam_steps`). ``{"func": {name: n}[, "aux": ...]}``.

    Derivation, not a convention: Adam skips a parameter whose ``.grad`` is ``None`` and creates no state
    for it, so a parameter's step count is the number of macrosteps in which its loss reached it. The
    encoder and the functional head are reached by the functional loss every step; in the shared arms the
    residue head by every step whose auxiliary batch contains a RES row and the equality head by every
    step with an EQ row; F and S's functional net have no auxiliary heads in the loss at all; S's
    auxiliary net has no functional head in the loss. Zero gradients are never manufactured to advance
    an unused head: that would change Adam's momentum."""
    n_res = n_eq = 0
    if run.arm in AUX_ARMS and cursor:
        is_eq = np.asarray(run.ctx.event_is_equality)[run.tape.aux_batches[:cursor]]
        n_eq, n_res = int(is_eq.any(axis=1).sum()), int((~is_eq).any(axis=1).sum())

    def counts(net, *, functional, residue, equality):
        out = {}
        for name, _ in net.named_parameters():
            head = _head_of(name)
            if head == "func_head":
                out[name] = cursor if functional else 0
            elif head == "res_head":
                out[name] = n_res if residue else 0
            elif head == "eq_head":
                out[name] = n_eq if equality else 0
            else:
                out[name] = cursor
        return out
    shared = run.arm in SHARED_ARMS
    out = {"func": counts(run.func_net, functional=True, residue=shared, equality=shared)}
    if run.arm == "S":
        out["aux"] = counts(run.aux_net, functional=False, residue=True, equality=True)
    return out


# ---------------------------------------------------------------------------
# checkpoints at optimizer-update boundaries (plain numeric format, weights_only load)
# ---------------------------------------------------------------------------

CHECKPOINT_NAME = "checkpoint.json"
_CHECKPOINT_KEYS = ("schema", "run_id", "arm", "seed", "cursor", "total_steps", "bindings",
                    "state_sha256", "weights_file", "weights_file_sha256", "accounting",
                    "cpu_seconds", "probe_seconds", "log", "diagnostics", "header_sha256")


def _owned_predecessor(directory, run):
    """The weights file the CURRENT header publishes (the one tracked, owned predecessor), or ``None``
    when the directory has no header. Nothing else in the directory is ever claimed.

    An existing header is owned only after it is PROVEN to be this run's: a well-formed, self-hashed
    header of this arm, seed, run id and total steps whose bindings equal the run's current bindings, and
    whose weights file exists and hashes to the SHA-256 the header records. A header that is malformed,
    foreign, bound to other inputs, or whose weights bytes were altered is neither owned nor reusable:
    this raises ``ValueError`` and the caller has written nothing yet (the header's own filename and
    self-hash were not enough: a valid header of another arm with identical cursor-0 weights was
    replaced, and a tracked file with altered bytes was overwritten)."""
    directory = Path(directory)
    path = directory / CHECKPOINT_NAME
    if not os.path.lexists(path):
        return None
    header = fb.read_json(path)
    fm._keys(header, _CHECKPOINT_KEYS, label="the existing checkpoint header")
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    require(fb.digest(body) == header["header_sha256"],
            f"the existing checkpoint header {path} was modified after it was written "
            f"(header_sha256 mismatch); it is never replaced. {FRESH_OUTPUT}")
    require(header["schema"] == CHECKPOINT_SCHEMA,
            f"the existing {path} is not a {CHECKPOINT_SCHEMA} checkpoint; it is never replaced")
    require(header["arm"] == run.arm and header["seed"] == run.seed
            and header["run_id"] == run.run_id and header["total_steps"] == run.total_steps,
            f"the existing checkpoint header {path} belongs to a different run "
            f"({header['run_id']!r}, not {run.run_id!r}); it is never replaced. {FRESH_OUTPUT}")
    require(header["bindings"] == run.bindings,
            f"the existing checkpoint header {path} is bound to different inputs (initial state, "
            "tape, training config, inputs, native cache, event table or code) than this run's; it is "
            f"never replaced. {FRESH_OUTPUT}")
    require(fb._integer(header["cursor"], "existing checkpoint cursor", minimum=0) <= run.total_steps,
            f"the existing checkpoint header {path} has a cursor beyond this run's steps")
    name = header["weights_file"]
    require(isinstance(name, str) and _WEIGHTS_NAME.match(name),
            f"the existing checkpoint header {path} names an unrecognised weights file {name!r}")
    weights = directory / name
    require(weights.is_file() and not weights.is_symlink(),
            f"the weights file {weights} that the existing checkpoint header publishes is missing; "
            f"nothing is written over it. {FRESH_OUTPUT}")
    require(file_sha256(weights) == header["weights_file_sha256"],
            f"the weights file {weights} does not match the SHA-256 its checkpoint header records: it "
            f"was modified, and is never overwritten. {FRESH_OUTPUT}")
    return name


def _is_protected(protected, path):
    """True when ``path`` names a protected input (resolved path, letter case, hard link, junction)."""
    return any(fb._same_file(source, path) for source in _protected_items(protected).values())


def _publish_weights(final, payload, *, tracked):
    """Publish ``payload`` as ``final`` through an exclusively created unique staging file.

    An existing ``final`` is never blindly overwritten: identical tensor content is reused (an
    interrupted earlier publication of the same state), a different one is replaced only when it is the
    file the current valid header tracks, and anything else is refused untouched."""
    final = Path(final)
    if os.path.lexists(final):
        require(final.is_file() and not final.is_symlink(),
                f"{final} exists and is not a regular file; it is never replaced")
        if _load_payload_digest(final) == tensor_state_digest(payload):
            return
        require(tracked,
                f"{final} already exists with different content and is not the checkpoint the "
                "current header publishes: it is an earlier interrupted attempt or a foreign file "
                f"and is never overwritten. Inspect it, or {FRESH_OUTPUT.lower()}")
    _publish(final, lambda handle: torch.save(payload, handle), text=False)


def save_checkpoint(run, directory, *, protected=()):
    """Publish the run's state at the CURRENT cursor (an optimizer-update boundary): weights first,
    the header last (it commits the checkpoint). A half-published checkpoint is refused by
    :func:`restore_checkpoint`; the previous header stays valid until the new one replaces it.

    Staging is exclusive and uniquely named (an existing ``ckpt_*.pt.tmp`` or any unknown file is never
    truncated or deleted), the destinations are checked against ``protected`` input paths first, the
    run's consumed inputs are verified, and an existing header is PROVEN to be this run's
    (:func:`_owned_predecessor`: bindings and the weights file's SHA-256) before anything is written.
    Afterwards only the weights file that previous, proven header tracked is removed, and not even that
    one when it is a protected input: an older checkpoint is retained rather than deleted."""
    run.verify_inputs()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = run.state_payload()
    weights_name = f"ckpt_{run.cursor:06d}.pt"
    final, header_path = directory / weights_name, directory / CHECKPOINT_NAME
    refuse_protected(protected, {"checkpoint weights": final, "checkpoint header": header_path})
    previous = _owned_predecessor(directory, run)
    _publish_weights(final, payload, tracked=(previous == weights_name))
    body = {"schema": CHECKPOINT_SCHEMA, "run_id": run.run_id, "arm": run.arm, "seed": run.seed,
            "cursor": run.cursor, "total_steps": run.total_steps, "bindings": run.bindings,
            "state_sha256": tensor_state_digest(payload), "weights_file": weights_name,
            "weights_file_sha256": file_sha256(final),
            "accounting": copy.deepcopy(run.accounting), "cpu_seconds": run.cpu_seconds,
            "probe_seconds": run.probe_seconds, "log": copy.deepcopy(run.log),
            "diagnostics": copy.deepcopy(run.diagnostics)}
    header = {**body, "header_sha256": fb.digest(body)}
    write_json_owned(header_path, header, protected=protected)
    if previous and previous != weights_name:
        stale = directory / previous
        if (stale.is_file() and not stale.is_symlink()
                and not _is_protected(protected, stale)):
            try:
                stale.unlink()
            except OSError:
                pass
    return header


def _expected_accounting(run, cursor):
    cfg = run.ctx.config
    aux = run.arm in AUX_ARMS
    updates = cursor * (2 if run.arm == "S" else 1)
    aux_rows = cursor * cfg.aux_batch_size if aux else 0
    return {"macrosteps": cursor, "optimizer_steps": updates, "backward_calls": updates,
            "forward_calls": {"functional": cursor, "auxiliary": cursor if aux else 0},
            "rows_forwarded": {"functional": cursor * cfg.batch, "auxiliary": aux_rows},
            "functional_presentations": cursor * cfg.batch, "auxiliary_presentations": aux_rows,
            "scalar_targets_presented": {"functional": cursor * cfg.batch, "auxiliary": aux_rows}}


def _validate_accounting(run, header):
    cursor = header["cursor"]
    accounting, log, diagnostics = header["accounting"], header["log"], header["diagnostics"]
    require(isinstance(accounting, dict) and isinstance(log, list) and isinstance(diagnostics, list)
            and accounting.get("macrosteps") == cursor and len(log) == cursor,
            "the checkpoint cursor disagrees with its own accounting and log")
    expected = _expected_accounting(run, cursor)
    require(all(accounting.get(key) == value for key, value in expected.items()),
            "the checkpoint cursor disagrees with its own accounting and log: the call, row, "
            "presentation and optimizer-step counters are not those of that many macrosteps")
    probe = accounting.get("probe")
    require(isinstance(probe, dict) and all(
        fb._integer(probe.get(key), f"checkpoint accounting probe {key}", minimum=0) is not None
        for key in ("forward_calls", "gradient_calls")),
            "the checkpoint's probe accounting is malformed")
    require(all(isinstance(entry, dict) and entry.get("step") == position
                for position, entry in enumerate(log)),
            "the checkpoint's step log is not one entry per macrostep")
    require([entry.get("step") if isinstance(entry, dict) else None for entry in diagnostics]
            == sorted(step for step in run.probe_steps if step < cursor),
            "the checkpoint's diagnostics are not those of the declared probe steps below the cursor")
    for name in ("cpu_seconds", "probe_seconds"):
        value = header[name]
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and value >= 0,
                f"the checkpoint's {name} is not a finite non-negative number")


def _validate_net_state(state, reference, label):
    require(isinstance(state, dict) and set(state) == set(reference),
            f"the checkpoint's {label} network does not hold exactly the expected parameter set")
    for key, expected in reference.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.shape == expected.shape
                and value.dtype == expected.dtype,
                f"the checkpoint's {label} tensor {key!r} has shape/dtype "
                f"{getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}, expected "
                f"{tuple(expected.shape)}/{expected.dtype}")
        require(bool(torch.isfinite(value).all()),
                f"the checkpoint's {label} tensor {key!r} holds non-finite values")


def _validate_optimizer_state(state, net, steps, config, label):
    """Check an optimizer payload against the ACTUAL expected optimizer: its parameter groups, exactly
    the parameters the tape says were updated (no more, no fewer), each with the implied step count and
    moments of its parameter's shape, dtype and finite, non-negative second moments."""
    require(isinstance(state, dict) and set(state) == {"state", "param_groups"},
            f"the checkpoint's {label} optimizer payload is malformed")
    reference = _make_adam(net.parameters(), config).state_dict()["param_groups"]
    require(tensor_state_digest(state["param_groups"]) == tensor_state_digest(reference),
            f"the checkpoint's {label} optimizer parameter groups (learning rate, betas, eps, weight "
            "decay, parameter list) are not those of the expected optimizer")
    named = list(net.named_parameters())
    entries = state["state"]
    require(isinstance(entries, dict) and all(isinstance(key, int) and not isinstance(key, bool)
                                              for key in entries),
            f"the checkpoint's {label} optimizer state is not indexed by parameter position")
    expected_indices = {index for index, (name, _) in enumerate(named) if steps[name] > 0}
    require(set(entries) == expected_indices,
            f"the checkpoint's {label} optimizer holds state for the wrong parameters: the tape "
            "implies exactly the parameters whose loss reached them; a parameter that was never "
            "updated has no entry")
    for index in sorted(expected_indices):
        name, parameter = named[index]
        entry = entries[index]
        require(isinstance(entry, dict) and set(entry) == {"step", "exp_avg", "exp_avg_sq"},
                f"the checkpoint's {label} optimizer entry for {name!r} is malformed")
        step = entry["step"]
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and bool(torch.isfinite(step))
                and float(step) == float(steps[name]),
                f"the checkpoint's {label} optimizer has Adam step "
                f"{float(step) if isinstance(step, torch.Tensor) and step.numel() == 1 else step!r} "
                f"for {name!r}; the frozen tape implies {steps[name]} updates")
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor = entry[moment]
            require(isinstance(tensor, torch.Tensor) and tensor.shape == parameter.shape
                    and tensor.dtype == parameter.dtype and bool(torch.isfinite(tensor).all()),
                    f"the checkpoint's {label} optimizer {moment} for {name!r} does not have the "
                    "parameter's shape and dtype, or is not finite")
        require(bool((entry["exp_avg_sq"] >= 0).all()),
                f"the checkpoint's {label} optimizer second moment for {name!r} is negative")


def _validate_payload(run, cursor, payload):
    """Validate a loaded payload against a DISPOSABLE shadow network and optimizer. Nothing live is
    touched; a payload that passes loads into the live objects without failing midway."""
    expected = {"func"} | ({"aux"} if run.arm == "S" else set())
    require(isinstance(payload.get("nets"), dict) and isinstance(payload.get("optimizers"), dict)
            and set(payload["nets"]) == expected and set(payload["optimizers"]) == expected,
            "the checkpoint holds the wrong set of networks for this arm")
    steps = expected_update_counts(run, cursor)
    config = run.ctx.config
    for name in sorted(expected):
        shadow = build_student(run.ctx.spec)
        _validate_net_state(payload["nets"][name], dict(shadow.state_dict()), name)
        _validate_optimizer_state(payload["optimizers"][name], shadow, steps[name], config, name)
        try:
            shadow.load_state_dict(payload["nets"][name])
            _make_adam(shadow.parameters(), config).load_state_dict(
                _clone(payload["optimizers"][name]))
        except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
            raise ValueError(f"the checkpoint's {name} state cannot be loaded into a network and "
                             f"optimizer of this run: {error}") from error


def _verified_checkpoint(directory, run):
    """Read and verify a checkpoint against ``run`` WITHOUT touching the run. Returns ``(header,
    payload)`` or ``None`` when there is no checkpoint. Any tamper or mismatch raises ``ValueError``."""
    directory = Path(directory)
    path = directory / CHECKPOINT_NAME
    if not path.exists():
        return None
    header = fb.read_json(path)
    fm._keys(header, _CHECKPOINT_KEYS, label="checkpoint header")
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    require(fb.digest(body) == header["header_sha256"],
            "the checkpoint header was modified after it was written (header_sha256 mismatch)")
    require(header["schema"] == CHECKPOINT_SCHEMA, f"not a {CHECKPOINT_SCHEMA} checkpoint")
    require(header["arm"] == run.arm and header["seed"] == run.seed
            and header["run_id"] == run.run_id and header["total_steps"] == run.total_steps,
            "the checkpoint belongs to a different run")
    require(header["bindings"] == run.bindings,
            "the checkpoint's bindings (initial state, tape, training config, inputs, native cache, "
            "event table, code) differ from this run's: the labels, programs, masks, cache identity "
            "or tape membership changed since it was written")
    cursor = header["cursor"]
    require(fb._integer(cursor, "checkpoint cursor", minimum=0) <= run.total_steps,
            "the checkpoint cursor exceeds the run's steps")
    weights = directory / header["weights_file"]
    require(weights.is_file() and weights.name == header["weights_file"],
            "the checkpoint's weights file is missing")
    require(file_sha256(weights) == header["weights_file_sha256"],
            "the checkpoint weights file does not match its recorded SHA-256")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and payload.get("format") == CHECKPOINT_SCHEMA,
            "the checkpoint weights are not a recognised payload")
    require(tensor_state_digest(payload) == header["state_sha256"],
            "the checkpoint tensors do not match their recorded content digest")
    _validate_accounting(run, header)
    _validate_payload(run, cursor, payload)
    return header, payload


def _load_payload(run, payload):
    run.func_net.load_state_dict(payload["nets"]["func"])
    run.func_opt.load_state_dict(_clone(payload["optimizers"]["func"]))
    if run.arm == "S":
        run.aux_net.load_state_dict(payload["nets"]["aux"])
        run.aux_opt.load_state_dict(_clone(payload["optimizers"]["aux"]))


def restore_checkpoint(run, directory, *, protected=()):
    """Verify, THEN load, all-or-nothing. The header, bindings, weights SHA-256, tensor digest,
    accounting and the payload (shapes, dtypes, finiteness, optimizer groups, exactly the parameters
    and per-parameter Adam step counts the frozen tape implies) are all checked against a disposable
    shadow network and optimizer, and the run's consumed inputs are re-verified, before any live object
    is touched. Should loading still fail, the previous live state (networks, optimizers, cursor,
    accounting, log) is restored. Returns True if a checkpoint was applied, False if there was none."""
    checked = _verified_checkpoint(directory, run)
    if checked is None:
        return False
    header, payload = checked
    run.verify_inputs()
    saved = {"payload": run.state_payload(), "cursor": run.cursor,
             "accounting": copy.deepcopy(run.accounting), "log": copy.deepcopy(run.log),
             "diagnostics": copy.deepcopy(run.diagnostics),
             "seconds": (run.cpu_seconds, run.probe_seconds)}
    try:
        _load_payload(run, payload)
        run.cursor = header["cursor"]
        run.accounting = copy.deepcopy(header["accounting"])
        run.log, run.diagnostics = copy.deepcopy(header["log"]), copy.deepcopy(header["diagnostics"])
        run.cpu_seconds, run.probe_seconds = header["cpu_seconds"], header["probe_seconds"]
    except BaseException:
        _load_payload(run, saved["payload"])
        run.cursor, run.accounting = saved["cursor"], saved["accounting"]
        run.log, run.diagnostics = saved["log"], saved["diagnostics"]
        run.cpu_seconds, run.probe_seconds = saved["seconds"]
        raise
    return True


def train_arm(ctx, arm, seed, directory, *, stop_after=None, protected=()):
    """Train (or resume) one run, checkpointing every ``checkpoint_every`` macrosteps, at the end and at
    ``stop_after`` (an interruption point for tests). Returns the run."""
    run = ArmRun(ctx, arm, seed)
    restore_checkpoint(run, directory)
    every = ctx.config.checkpoint_every
    while run.cursor < run.total_steps and (stop_after is None or run.cursor < stop_after):
        run.macrostep()
        if (run.cursor % every == 0 or run.cursor == run.total_steps
                or (stop_after is not None and run.cursor == stop_after)):
            save_checkpoint(run, directory, protected=protected)
    return run


# ---------------------------------------------------------------------------
# frozen adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Adapter:
    run_id: str
    arm: str
    seed: int
    func_net: object
    aux_net: object
    meta: dict
    file_sha256: str
    state_sha256: str


def _freeze_net(net):
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def freeze_adapter(run, directory, source_report=None, *, protected=()):
    """Write the frozen adapter (functional net, plus the auxiliary net when it is separate or shared)
    as one hashed numeric file and a sidecar header. Requires a finished run whose consumed inputs
    still verify; written once, through exclusive staging, never over an existing file."""
    require(run.cursor == run.total_steps, "only a finished run can be frozen")
    run.verify_inputs()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / f"{run.run_id}.adapter.pt"
    header_path = directory / f"{run.run_id}.adapter.json"
    refuse_protected(protected, {"adapter weights": weights, "adapter header": header_path})
    require(not os.path.lexists(weights), f"{weights} already exists; adapters are written once")
    require(not os.path.lexists(header_path),
            f"{header_path} already exists; adapters are written once")
    payload = {"format": ADAPTER_SCHEMA,
               "func": _clone(dict(run.func_net.state_dict())),
               "aux": (_clone(dict(run.aux_net.state_dict())) if run.arm == "S" else None)}
    _publish(weights, lambda handle: torch.save(payload, handle), text=False)
    body = {"schema": ADAPTER_SCHEMA, "run_id": run.run_id, "arm": run.arm, "seed": run.seed,
            "weights_file": weights.name, "weights_file_sha256": file_sha256(weights),
            "adapter_state_sha256": tensor_state_digest(payload),
            "state_sha256_note": ("a hash of the parameter TENSORS' content; it is NOT a checkpoint "
                                  "file hash"),
            "bindings": run.bindings, "spec": run.ctx.spec.document(),
            "cost": run.cost_report(), "source_report": source_report,
            "auxiliary_net": ("separate" if run.arm == "S" else
                              ("shared" if run.arm in SHARED_ARMS else "none"))}
    header = {**body, "header_sha256": fb.digest(body)}
    write_json_owned(header_path, header, protected=protected)
    return header


def load_adapter(directory, run_id, ctx):
    """Load and VERIFY a frozen adapter: header digest, bindings against ``ctx``, file hash, tensor
    digest. ``weights_only=True``; the nets come back frozen and in eval mode."""
    directory = Path(directory)
    header = fb.read_json(directory / f"{run_id}.adapter.json")
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    require(fb.digest(body) == header["header_sha256"],
            f"adapter {run_id}: the header was modified after it was written")
    arm, seed = header["arm"], header["seed"]
    require(header["run_id"] == run_id and header["bindings"] == run_bindings(ctx, arm, seed),
            f"adapter {run_id}: its bindings differ from the current inputs, tape or configuration")
    weights = directory / header["weights_file"]
    require(file_sha256(weights) == header["weights_file_sha256"],
            f"adapter {run_id}: the weights file was modified (SHA-256 mismatch)")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    require(tensor_state_digest(payload) == header["adapter_state_sha256"],
            f"adapter {run_id}: the tensors do not match their recorded digest")
    func = _freeze_net(student_from_state(ctx.spec, payload["func"]))
    if arm == "S":
        aux = _freeze_net(student_from_state(ctx.spec, payload["aux"]))
    elif arm in SHARED_ARMS:
        aux = func
    else:
        aux = None
    return Adapter(run_id=run_id, arm=arm, seed=int(seed), func_net=func, aux_net=aux, meta=header,
                   file_sha256=header["weights_file_sha256"],
                   state_sha256=header["adapter_state_sha256"])


# ---------------------------------------------------------------------------
# extraction and evaluation
# ---------------------------------------------------------------------------

def extract_function_features(table, net, rows, *, batch_size=64, token_pad=None):
    """FUNCTION-query features ``(rows, width)`` float64. Eval mode, no gradients; training flags
    restored. Batch order, batch size and padding amount change float32 rounding only."""
    rows = np.asarray(rows, dtype=np.int64)
    chunks = []
    saved = [(module, module.training) for module in net.modules()]
    net.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = function_batch_for(table, rows[start:start + batch_size],
                                           token_pad=token_pad)
                chunks.append(net.encode(batch).double().numpy())
    finally:
        for module, flag in saved:
            module.training = flag
    return np.concatenate(chunks, axis=0)


def predict_function(table, net, rows, *, batch_size=64, token_pad=None):
    """The functional head's predictions ``(rows,)`` float64 (never touches the auxiliary heads)."""
    features = extract_function_features(table, net, rows, batch_size=batch_size,
                                         token_pad=token_pad)
    with torch.no_grad():
        return net.func_head(torch.from_numpy(features.astype(np.float32))).squeeze(-1) \
            .double().numpy()


def source_report(ctx, adapter, outcomes_train, outcomes_dev, cache, *, bindings=None):
    """Source-train and SOURCE-DEV functional error of a frozen adapter (the source-preservation
    measurement), with the constant source-train-mean predictor for scale. ``bindings`` (the bound
    canonical outcome identities) is recorded so the stored error says WHICH outcome version it scored."""
    units = f"({ctx.cohorts.manifest.source.units})^2"
    report = {"units_of_mse": units, "adapter": adapter.run_id,
              "source_dev_use": "scoring only: these outcomes never enter training"}
    if bindings is not None:
        report["outcome_bindings"] = bindings
    cache_groups = cache.groups
    train_ids, dev_ids = ctx.cohorts.roles["source_train"], ctx.cohorts.roles["source_dev"]
    for name, ids, outcomes in (("source_train", train_ids, outcomes_train),
                                ("source_dev", dev_ids, outcomes_dev)):
        rows = [ctx.table.position[cid] for cid in ids]
        truth = np.array([outcomes[cid] for cid in ids], dtype=np.float64)
        predicted = predict_function(ctx.table, adapter.func_net, rows,
                                     batch_size=ctx.config.extraction_batch)
        squared = (predicted - truth) ** 2
        groups = [cache_groups[ctx.table.position[cid]] for cid in ids]
        entry = {"n": len(ids), "mse": float(squared.mean()), "rmse": float(math.sqrt(squared.mean())),
                 "constant_source_train_mean_mse": float(((truth - ctx.source_mean) ** 2).mean()),
                 "per_group_mse": {group: float(np.mean([sq for sq, g in zip(squared, groups)
                                                         if g == group]))
                                   for group in sorted(set(groups))}}
        report[name] = entry
    return report


# ---------------------------------------------------------------------------
# provenance of derived features
# ---------------------------------------------------------------------------

def derived_supervision(native, *, source, measured_ids, computational_ids, arm):
    """``(SupervisionHistory, new_costs)`` for an adapted FUNCTION feature space.

    A source-adapted representation is not label-free. Native entries and sequence-only exposure are
    preserved. If the native history is UNKNOWN it stays unknown (``SupervisionHistory`` forbids
    entries then) and the known new costs are returned for the schema instead of pretending the
    complete history is known. Otherwise the new costs are appended as documented entries: the
    measured source outcomes (their candidate ids) and, separately, the unique computational answers
    (their own compact answer ids, never the candidate id reused for every derived answer).

    ``measured_ids`` is every measured source label the adapter consumed. That includes ALL
    source-training labels even before the first batch, because the functional bias is initialised from
    their mean; the entry therefore holds for a run at any cursor."""
    require(isinstance(native, fb.SupervisionHistory), "native must be a SupervisionHistory")
    measured = {"assay": source.assay, "target": source.target,
                "label_count": len(measured_ids), "label_ids": list(measured_ids)}
    entries = [measured]
    computational = None
    if computational_ids is not None:
        label = COMPUTATIONAL_ASSAY + (" [SHUFFLED targets: a corruption control]" if arm == "Q"
                                       else " [direct-read instructions]" if arm == "R" else "")
        computational = {"assay": label, "target": COMPUTATIONAL_TARGET,
                         "label_count": len(computational_ids),
                         "label_ids": list(computational_ids)}
        entries.append(computational)
    costs = {"measured_source": {"assay": measured["assay"], "target": measured["target"],
                                 "count": measured["label_count"],
                                 "ids_sha256": fb.digest(list(measured_ids))},
             "computational": None if computational is None else {
                 "assay": computational["assay"], "target": computational["target"],
                 "count": computational["label_count"],
                 "ids_sha256": fb.digest(list(computational_ids))},
             "native_history_status": native.status}
    if native.status == fb.HISTORY_UNKNOWN:
        costs["recorded_in_schema_only"] = ("the native supervision history is UNKNOWN, so the "
                                            "derived history stays unknown; these are the KNOWN new "
                                            "costs, not the complete history")
        costs["measured_source"]["label_ids"] = list(measured_ids)
        if computational is not None:
            costs["computational"]["label_ids"] = list(computational_ids)
        return native.snapshot(), costs
    exposure = native.exposure_status if native.exposure_status != "none" else "declared"
    exposure_text = (native.exposure_description if native.exposure_status != "none"
                     else "") + ("; " if native.exposure_status != "none" else "") + (
        "the adapter was trained on the source auxiliary sequences (sequence-level exposure)")
    history = fb.SupervisionHistory(
        status=fb.HISTORY_DOCUMENTED,
        description=(f"{native.description}; a source-adapted shared student was then fitted on "
                     "measured source outcomes" + (" and exact computational answers"
                                                   if computational is not None else "")),
        documented=tuple(native.documented) + tuple(entries), exposure_status=exposure,
        exposure_description=exposure_text)
    return history, costs


# ---------------------------------------------------------------------------
# auxiliary panels (seen/new sequences x seen/held-out dependency)
# ---------------------------------------------------------------------------

class PanelUnsupported(ValueError):
    """A panel that cannot be built for a stated reason (recorded, never silently dropped)."""


@dataclass(frozen=True, eq=False)
class Panel:
    name: str
    table: object
    rows: object
    composed: object
    composed_len: object
    direct: object
    direct_len: object
    codes: object
    is_equality: object
    counterfactual: object          # {name: (instr, lens, codes_cf, event_positions)} or None
    note: str = ""
    excluded: tuple = ()            # candidates below the task's minimum coordinates (reported)


def make_panel(ctx, name, rows, *, universe, instances, seed, exclude_ids=()):
    """A panel over ``rows`` (``(candidate_id, heavy, light)``): a source-only event table over each
    row's OWN layout (rows with different chain lengths are all kept; only a row below the task's
    minimum coordinates is excluded and listed), teacher-verified, with composed and direct instruction
    arrays and (held-out only) the operator deletion/order counterfactuals. A row that cannot supply a
    complete instance is skipped and recorded under the table's ``unsupported`` notes."""
    space = ctx.space
    supported = [(cid, heavy, light) for cid, heavy, light in rows
                 if len(heavy) >= 1 and len(heavy) + len(light) >= fc.MIN_COORDINATES]
    excluded = tuple(sorted(cid for cid, heavy, light in rows
                            if len(heavy) < 1 or len(heavy) + len(light) < fc.MIN_COORDINATES))
    if not supported:
        raise PanelUnsupported(f"panel {name}: no row has {fc.MIN_COORDINATES} or more coordinates "
                               f"({len(excluded)} excluded); no composition panel is defined")
    layouts = fc.layouts_for_rows(supported, ctx.layouts.alphabet)
    table = fc.generate_events(layouts, supported, universe=universe, instances_per_sequence=instances,
                               seed=seed, exclude_ids=exclude_ids, strict=False)
    if not len(table):
        raise PanelUnsupported(f"panel {name}: no row could supply a complete {universe} instance "
                               f"({len(table.unsupported)} skipped); no composition panel is defined")
    fc.verify_events_against(table, supported)
    sequences = {cid: (heavy, light) for cid, heavy, light in supported}
    composed, composed_len = _stack_instructions(
        [fc.compile_instruction(space, table.layout_for(e.candidate_id), e.program, e.query)
         for e in table.events])
    direct, direct_len = _stack_instructions(
        [fc.compile_direct(space, table.layout_for(e.candidate_id), e.resolution)
         for e in table.events])
    positions = [i for i, e in enumerate(table.events) if len(e.program.ops) == 2]
    counterfactual = None
    if universe == fc.UNIVERSE_HELD_OUT and positions:
        counterfactual = {}
        for variant in ("delete_first", "delete_second", "reversed"):
            rows_cf, codes_cf = [], []
            for i in positions:
                event = table.events[i]
                own = table.layout_for(event.candidate_id)
                program = fc.counterfactual_programs(event.program)[variant]
                rows_cf.append(fc.compile_instruction(space, own, program, event.query))
                heavy, light = sequences[event.candidate_id]
                value = fc.answer(own, heavy, light, program, event.query)
                codes_cf.append(fc.answer_code(own, value, event.query))
            instr, lens = _stack_instructions(rows_cf)
            counterfactual[variant] = (instr, lens, np.array(codes_cf, dtype=np.int64),
                                       np.array(positions, dtype=np.int64))
    return Panel(
        name=name, table=table,
        rows=np.array([ctx.table.position[e.candidate_id] for e in table.events], dtype=np.int64),
        composed=composed, composed_len=composed_len, direct=direct, direct_len=direct_len,
        codes=table.codes(), is_equality=table.is_equality(), counterfactual=counterfactual,
        excluded=excluded)


def predict_codes(ctx, net, rows, instructions, lengths, is_equality, *, batch_size=128):
    """Predicted integer answers: argmax of the residue head for RES rows, ``logit > 0`` for EQ rows."""
    out = np.zeros(len(rows), dtype=np.int64)
    saved = [(module, module.training) for module in net.modules()]
    net.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                stop = min(len(rows), start + batch_size)
                batch = make_student_batch(ctx.table, rows[start:stop], instructions[start:stop],
                                           lengths[start:stop])
                res, eq = net.auxiliary_forward(batch)
                chosen = torch.where(torch.from_numpy(np.array(is_equality[start:stop])),
                                     (eq > 0).long(), res.argmax(dim=1))
                out[start:stop] = chosen.numpy()
    finally:
        for module, flag in saved:
            module.training = flag
    return out


def _rate(correct):
    return {"events": int(len(correct)), "accuracy": float(np.mean(correct)) if len(correct) else None}


def _majority(codes):
    if not len(codes):
        return None
    _, counts = np.unique(codes, return_counts=True)
    return float(counts.max() / len(codes))


def panel_note_summary(notes):
    """Tell the event table's ``unsupported`` notes apart. Most are unsupported-control or FALLBACK notes (a
    secondary stratum fell back to the primary one, a literal-literal control is one-sided): the instance
    still exists and its events are in the panel. Only a note that carries ``rejections`` records an
    instance that was actually skipped. Counting every note as a skipped instance would misreport a panel
    that holds all of its events."""
    by_reason = {}
    for note in notes:
        reason = str(note.get("reason", ""))
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"unsupported_control_notes": len(notes),
            "skipped_instance_notes": sum(1 for note in notes if "rejections" in note),
            "unsupported_control_notes_by_reason": dict(sorted(by_reason.items()))}


def summarize_panel(panel, predicted, *, base_predicted=None):
    """Accuracy overall and by output type, stratum (RES vs primary/secondary EQ), dependency tag and,
    for held-out EQ, by whether the counterfactuals change the ANSWER; plus ``majority_rate``, the
    accuracy of always guessing the group's own commonest observed answer. That is an empirical
    best-constant baseline, NOT a chance rate: a group that mixes RES (many-class residue) and EQ
    (binary) outputs has no single meaningful chance rate."""
    events = panel.table.events
    correct = predicted == panel.codes

    def group(key):
        out = {}
        for value in sorted({key(e) for e in events if key(e) is not None}):
            members = np.array([i for i, e in enumerate(events) if key(e) == value], dtype=np.int64)
            out[value] = {**_rate(correct[members]), "majority_rate": _majority(panel.codes[members])}
        return out
    summary = {"events": len(events), "overall": {**_rate(correct),
                                                  "majority_rate": _majority(panel.codes)},
               "by_output_type": group(lambda e: e.output_type),
               "by_stratum": group(lambda e: e.stratum),
               "by_dependency": group(lambda e: e.dependency),
               "by_stratum_and_dependency": group(lambda e: f"{e.stratum}/{e.dependency}")}
    held = [e for e in events if e.flags is not None and e.stratum == fc.STRATUM_EQ_PRIMARY
            and e.dependency == fc.DEP_BOTH]
    if held:
        summary["held_out_primary_eq_by_answer_change"] = group(
            lambda e: (None if e.flags is None or e.stratum != fc.STRATUM_EQ_PRIMARY
                       or e.dependency != fc.DEP_BOTH
                       else ("answer_changes_under_a_counterfactual" if any(e.flags.values())
                             else "answer_unchanged_under_every_counterfactual")))
    if panel.counterfactual is not None and base_predicted is not None:
        summary["operator_controls"] = _counterfactual_summary(panel, predicted, base_predicted)
    return summary


def _counterfactual_summary(panel, predicted, base_predicted):
    out = {}
    for variant, (_, _, codes_cf, positions) in panel.counterfactual.items():
        base_truth = panel.codes[positions]
        cf_pred = base_predicted[variant]
        changed = codes_cf != base_truth
        entry = {"events": int(len(positions)),
                 "accuracy_on_counterfactual_program": _rate(cf_pred == codes_cf)["accuracy"],
                 "answer_changed_events": int(changed.sum()),
                 "accuracy_where_teacher_answer_changed": (
                     float(np.mean(cf_pred[changed] == codes_cf[changed])) if changed.any() else None),
                 "accuracy_where_teacher_answer_unchanged": (
                     float(np.mean(cf_pred[~changed] == codes_cf[~changed]))
                     if (~changed).any() else None),
                 "prediction_unchanged_from_full_program_where_teacher_answer_changed": (
                     float(np.mean(cf_pred[changed] == predicted[positions][changed]))
                     if changed.any() else None)}
        out[variant] = entry
    return out


def evaluate_panel(ctx, adapter, panel, mode):
    """Score one adapter's auxiliary head on one panel in ``composed`` or ``direct`` instruction mode.
    The held-out panel also scores the operator deletion/order counterfactual programs."""
    net = adapter.aux_net
    instr, lens = ((panel.composed, panel.composed_len) if mode == "composed"
                   else (panel.direct, panel.direct_len))
    predicted = predict_codes(ctx, net, panel.rows, instr, lens, panel.is_equality)
    base = None
    if panel.counterfactual is not None and mode == "composed":
        base = {}
        for variant, (instr_cf, lens_cf, _, positions) in panel.counterfactual.items():
            base[variant] = predict_codes(ctx, net, panel.rows[positions], instr_cf, lens_cf,
                                          panel.is_equality[positions])
    return summarize_panel(panel, predicted, base_predicted=base)


# ---------------------------------------------------------------------------
# bound outcome versions (one per stage that reads them)
# ---------------------------------------------------------------------------

def outcome_binding(role, outcomes, ids, assay):
    """The canonical identity of one outcome table: sorted ids, the numeric values in that order and
    the declared endpoint. SEMANTIC: the same ids and values in another file row order are the same
    version. ``ids_sha256`` and ``utilities_sha256`` use the same definitions as the matrix freeze
    index's ``common.offered`` record, so the two can be compared directly."""
    ordered = sorted(ids)
    values = [float(outcomes[cid]) for cid in ordered]
    endpoint = assay.document()
    return {"role": role, "count": len(ordered), "ids_sha256": fb.digest(ordered),
            "utilities_sha256": fb._array_sha256(values), "endpoint_sha256": fb.digest(endpoint),
            "canonical_sha256": fb.digest({"role": role, "ids": ordered, "utilities": values,
                                           "endpoint": endpoint})}


# ---------------------------------------------------------------------------
# paths and the stage-by-stage pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferInputs:
    """Every input file, explicit. The target-evaluation outcomes are opened only by ``evaluate``."""

    token_header: object
    cohorts: object
    source_train_outcomes: object
    source_dev_outcomes: object
    target_train_outcomes: object
    target_eval_outcomes: object
    token_npz: object = None


class TransferRun:
    """The resumable, file-based pipeline for ONE verified native token cache (see the module
    docstring for the stage order). ``open`` validates everything that can be validated before any
    compute; each stage method is idempotent and refuses inconsistent, partial or foreign state."""

    def __init__(self, inputs, output_dir, config, cache, cohorts, ctx):
        self.inputs, self.config, self.cache, self.cohorts, self.ctx = (inputs, config, cache,
                                                                        cohorts, ctx)
        self.root = Path(output_dir)
        self.tapes = self.root / "tapes"
        self.runs = self.root / "runs"
        self.adapters = self.root / "adapters"
        self.features_dir = self.root / "features"
        self.matrix_inputs = self.root / "matrix_inputs"
        self.freeze_dir = self.root / "matrix_freeze"
        self.analysis_dir = self.root / "analysis"
        self.state_path = self.root / "stage_state.json"
        self.input_paths = {
            "token cache header": Path(inputs.token_header), "token cache NPZ": Path(cache.npz_path),
            "cohort manifest": Path(inputs.cohorts),
            "source training outcomes": Path(inputs.source_train_outcomes),
            "source development outcomes": Path(inputs.source_dev_outcomes),
            "target training outcomes": Path(inputs.target_train_outcomes),
            "target evaluation outcomes": Path(inputs.target_eval_outcomes)}
        self.config_sha256 = fb.digest(config.identity_document())
        self._dev_outcomes = None
        self._dev_binding = None
        self.state = self._load_state()

    # --- opening ---------------------------------------------------------------------------
    @classmethod
    def open(cls, inputs, output_dir, config):
        """Load the verified cache, validate the cohorts, read ONLY the source-train outcomes and build
        the context. No target outcome and no source-dev outcome is opened."""
        require(isinstance(inputs, TransferInputs), "open needs TransferInputs")
        require(isinstance(config, TransferConfig), "open needs a TransferConfig")
        cache = fne.load_token_cache(inputs.token_header, npz_path=inputs.token_npz)
        manifest = fc.read_cohort_manifest(inputs.cohorts)
        cohorts = fc.validate_cohorts(cache, manifest)
        outcomes = fc.read_role_outcomes(inputs.source_train_outcomes,
                                         cohorts.roles["source_train"],
                                         label="source training outcomes")
        ctx = build_context(cache, cohorts, config, outcomes)
        return cls(inputs, output_dir, config, cache, cohorts, ctx)

    def _load_state(self):
        if self.state_path.exists():
            state = fb.read_json(self.state_path)
            require(state.get("schema") == STATE_SCHEMA, f"{self.state_path} is not a stage state")
            require(state["inputs_sha256"] == self.ctx.inputs_sha256
                    and state["config_sha256"] == self.config_sha256,
                    f"{self.root} holds the state of DIFFERENT inputs or configuration; use a fresh "
                    "output directory (nothing is overwritten or reused)")
            state.setdefault("outcomes", {})
            return state
        return {"schema": STATE_SCHEMA, "inputs_sha256": self.ctx.inputs_sha256,
                "config_sha256": self.config_sha256, "stages": {}, "outcomes": {}}

    # --- owned writes ------------------------------------------------------------------------
    def _save_state(self):
        self._write_json(self.state_path, self.state)

    def _write_json(self, path, document):
        write_json_owned(path, document, protected=self.input_paths)

    def _guard(self, path):
        refuse_protected(self.input_paths, {f"output {Path(path).name}": Path(path)})

    def _checkpoint_cursors(self, arm):
        total, every = self.config.total_steps(arm), self.config.checkpoint_every
        return sorted(set(range(every, total + 1, every)) | {total})

    def predictable_outputs(self):
        """Every destination this pipeline will write whose name is known up front (final names; the
        legacy ``.tmp`` names are added by the guard), for all stages."""
        out = {"stage state": self.state_path, "event table": self.tapes / "event_table.json",
               "input record": self.root / "inputs.json", "report": self.root / REPORT_NAME,
               "target analysis": self.analysis_dir / "target_analysis.json",
               "auxiliary panels": self.analysis_dir / "auxiliary_panels.json"}
        for seed in self.config.source_seeds:
            out[f"tape of seed {seed}"] = self.tapes / f"seed_{seed}.json"
        for seed in self.config.source_seeds:
            for arm in self.config.arm_list():
                run_id = rep_id(arm, seed)
                out[f"adapter weights {run_id}"] = self.adapters / f"{run_id}.adapter.pt"
                out[f"adapter header {run_id}"] = self.adapters / f"{run_id}.adapter.json"
                out[f"checkpoint header {run_id}"] = self.runs / run_id / CHECKPOINT_NAME
                for cursor in self._checkpoint_cursors(arm):
                    out[f"checkpoint {run_id} at {cursor}"] = (self.runs / run_id
                                                               / f"ckpt_{cursor:06d}.pt")
        for name in self.representations():
            out[f"feature cache header {name}"] = self.features_dir / f"{name}.cache.json"
            out[f"feature cache NPZ {name}"] = self.features_dir / f"{name}.cache.npz"
        for name in ("candidates.csv", "manifest.json", "plan.json", "target_train_outcomes.csv"):
            out[f"matrix input {name}"] = self.matrix_inputs / name
        plan = fm.parse_plan(self._plan_document())
        for label, path in fm._output_paths(self.freeze_dir, fm.enumerate_cells(plan)).items():
            out[f"matrix freeze {label}"] = path
        return out

    def verify_maps(self):
        """The loader-side id and role maps this pipeline consumes, against the identities that bound
        them: the context's (table row map, cohorts) and, when ``cohorts`` is a separate object, this
        run's own, plus the table's ids against the verified cache's row order. A swapped row entry or a
        changed roster is refused here, in memory, before any file or stage state can change. The feature
        arrays are not re-hashed (:meth:`SourceContext.verify_all` does that where it matters)."""
        ctx = self.ctx
        ctx.verify_maps()
        if self.cohorts is not ctx.cohorts:
            verify_cohort_maps(self.cohorts, ctx.table)
            require(self.cohorts.cohort_sha256 == ctx.cohorts.cohort_sha256,
                    "the run's cohorts and the context's were validated from different manifests")
        require(tuple(self.cache.ids) == ctx.table.ids,
                "the token table's ids no longer match the verified token cache's row order")
        return True

    def preflight(self):
        """Refuse, before anything is written, a changed id/role map (:meth:`verify_maps`) and any known
        input that is (or is linked to) a destination or the legacy staging name of one. Every public
        stage calls this first, so a completed stage is not adopted, nor a new one published, over a map
        that no longer matches. A known collision therefore never destroys an input and never commits
        partial stage state."""
        self.verify_maps()
        refuse_protected(self.input_paths, self.predictable_outputs())

    def _complete(self, stage):
        return bool(self.state["stages"].get(stage, {}).get("complete"))

    def _require(self, stage):
        require(self._complete(stage), f"stage {stage!r} is not complete; run the stages in order "
                                       f"{STAGES}")

    def _write_json_new_or_equal(self, path, document):
        """Write ``document`` if ``path`` is absent; if it exists it must hold exactly this document
        (verified, never blindly overwritten)."""
        path = Path(path)
        document = fb._builtin(document)
        if os.path.lexists(path):
            require(path.is_file() and fb.digest(fb.read_json(path)) == fb.digest(document),
                    f"{path} differs from what the inputs imply: a tape, event table or input "
                    "record was changed after it was frozen; nothing was trained on it")
            return False
        self._write_json(path, document)
        return True

    def _write_text_new_or_equal(self, path, text):
        path = Path(path)
        if os.path.lexists(path):
            require(path.is_file() and path.read_text(encoding="utf-8") == text,
                    f"{path} exists with content other than what this run binds; it is never "
                    f"overwritten. {FRESH_OUTPUT}")
            return False
        write_text_owned(path, text, protected=self.input_paths)
        return True

    # --- bound outcome versions ----------------------------------------------------------------
    def _bind(self, role, outcomes, ids, assay):
        """Bind the canonical outcome version when a stage first reads it, persisting it BEFORE any
        dependent artifact is committed, and refuse any later read that differs from it. Returns
        ``(binding, newly_recorded)``."""
        current = outcome_binding(role, outcomes, ids, assay)
        recorded = self.state["outcomes"].get(role)
        if recorded is None:
            self.state["outcomes"][role] = current
            self._save_state()
            return current, True
        require(recorded == current,
                f"the {role} outcomes differ from the version bound when they were first read "
                f"(bound {recorded['canonical_sha256'][:16]}..., now "
                f"{current['canonical_sha256'][:16]}...): artifacts built from the old version are "
                f"never relabelled and a pipeline never mixes two versions. Restore the original "
                f"outcomes (a different row order is fine) or {FRESH_OUTPUT}")
        return current, False

    # --- prepare ---------------------------------------------------------------------------
    def prepare(self):
        """Freeze the event table, the per-seed tapes and the input record; verify any that exist."""
        ctx = self.ctx
        self.preflight()
        ctx.verify_all()
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_json_new_or_equal(self.tapes / "event_table.json", ctx.events.document())
        for seed, tape in ctx.seed_tapes.items():
            self._write_json_new_or_equal(self.tapes / f"seed_{seed}.json", tape.document())
        source_train = outcome_binding(
            "source_train", dict(zip(ctx.functional_ids, ctx.functional_labels.tolist())),
            ctx.functional_ids, self.cohorts.manifest.source)
        record = {"schema": "functional-shared-transfer-inputs/1",
                  "inputs_sha256": ctx.inputs_sha256, "config": self.config.identity_document(),
                  "cohorts": ctx.cohorts.summary(self.cache),
                  "native_cache": {"cache_sha256": self.cache.cache_sha256,
                                  "file_sha256": self.cache.file_sha256,
                                  "runtime_sha256": self.cache.runtime_sha256,
                                  "token_map_sha256": self.cache.token_map_sha256,
                                  "extractor": self.cache.extractor,
                                  "activation": self.cache.activation},
                  "source_training_labels": {"count": len(ctx.functional_ids),
                                            "mean": ctx.source_mean},
                  "event_table": {"events": len(ctx.events), "sha256": ctx.events.sha256,
                                  "unique_sequences": ctx.events.unique_sequences(),
                                  "layout_shapes": ctx.events.config["layout_shapes"],
                                  "replacement": ctx.events.config["replacement"],
                                  "instance_shapes": ctx.events.config["instance_shapes"],
                                  "coverage": ctx.coverage, "balance": ctx.balance,
                                  "unsupported_notes": len(ctx.events.unsupported)},
                  "tapes": {str(seed): {"sha256": tape.sha256, "q_report": tape.q_report}
                            for seed, tape in ctx.seed_tapes.items()},
                  "student": ctx.spec.document(), "code_sha256": ctx.code_sha256,
                  "loss_definitions": {
                      "functional": (f"mean squared error of the functional head, in "
                                     f"({self.cohorts.manifest.source.units})^2 (raw source units; "
                                     "no z-scoring)"),
                      "auxiliary": ("(sum of residue cross-entropy over RES rows + sum of "
                                    "equality BCE-with-logits over EQ rows) / auxiliary rows, in "
                                    "nats"),
                      "shared_arms": "one Adam step on functional_mean + lam * auxiliary_mean",
                      "lam": self.config.lam, "clip_norm": self.config.clip_norm,
                      "separate_arm": ("independent auxiliary optimizer, clipping and RNG; the "
                                       "auxiliary loss is lam * auxiliary_mean")},
                  "outcome_bindings": {
                      "policy": ("semantic: sorted ids, numeric values and the declared endpoint; "
                                 "the same data in another row order is the same version"),
                      "source_training": source_train},
                  "target_outcomes": ("not opened by prepare, train or features; target training "
                                      "outcomes are read by freeze and target evaluation outcomes "
                                      "only by evaluate")}
        self._write_json_new_or_equal(self.root / "inputs.json", record)
        recorded = self.state["outcomes"].get("source_train")
        require(recorded is None or recorded == source_train,
                f"the source training outcomes differ from the version this output was prepared "
                f"with. {FRESH_OUTPUT}")
        self.state["outcomes"]["source_train"] = source_train
        self.state["stages"]["prepare"] = {
            "complete": True, "event_table_sha256": ctx.events.sha256,
            "tape_sha256": {str(seed): tape.sha256 for seed, tape in ctx.seed_tapes.items()}}
        self._save_state()
        return self.state["stages"]["prepare"]

    # --- train -----------------------------------------------------------------------------
    def _dev_labels(self):
        """The source-development outcomes, bound on the first read (persisted before any adapter that
        was scored on them is written) and verified against that binding on every later read."""
        if self._dev_outcomes is None:
            outcomes = fc.read_role_outcomes(
                self.inputs.source_dev_outcomes, self.cohorts.roles["source_dev"],
                label="source development outcomes")
            self._dev_binding, _ = self._bind("source_dev", outcomes,
                                              self.cohorts.roles["source_dev"],
                                              self.cohorts.manifest.source)
            self._dev_outcomes = outcomes
        return self._dev_outcomes

    def run_ids(self):
        return [rep_id(arm, seed) for seed in self.config.source_seeds
                for arm in self.config.arm_list()]

    def _check_adapter_outcomes(self, adapter):
        """An adapter's stored source error must say it was scored on the bound outcome versions."""
        report = adapter.meta.get("source_report") or {}
        bound = report.get("outcome_bindings")
        recorded = self.state["outcomes"]
        require(isinstance(bound, dict) and recorded.get("source_dev") is not None
                and recorded.get("source_train") is not None,
                f"adapter {adapter.run_id} carries no recorded source outcome binding (an older or "
                f"foreign output), so it cannot be continued or compared. {FRESH_OUTPUT}")
        require(bound.get("source_dev") == recorded["source_dev"]
                and bound.get("source_train") == recorded["source_train"],
                f"adapter {adapter.run_id} was scored on a different version of the source outcomes "
                f"than the one this output is bound to; mixing versions would make the paired "
                f"source-preservation comparison meaningless. {FRESH_OUTPUT}")

    def train(self, *, stop_after=None):
        """Train every (arm, seed), freezing each adapter and scoring it on SOURCE-DEV. Returns True when
        every run is finished and frozen; ``stop_after`` interrupts every unfinished run at that
        macrostep (a checkpoint is written first)."""
        self._require("prepare")
        self.preflight()
        self.ctx.verify_all()
        self._write_json_new_or_equal(self.tapes / "event_table.json", self.ctx.events.document())
        for seed, tape in self.ctx.seed_tapes.items():
            self._write_json_new_or_equal(self.tapes / f"seed_{seed}.json", tape.document())
        if self.state["outcomes"].get("source_dev") is not None:
            self._dev_labels()           # continuation: a changed dev version is refused BEFORE training
        runs = self.state["stages"].setdefault("train", {"complete": False, "runs": {}})["runs"]
        finished = True
        train_labels = dict(zip(self.ctx.functional_ids, self.ctx.functional_labels.tolist()))
        for seed in self.config.source_seeds:
            for arm in self.config.arm_list():
                run_id = rep_id(arm, seed)
                header_path = self.adapters / f"{run_id}.adapter.json"
                if run_id in runs and header_path.exists():
                    adapter = load_adapter(self.adapters, run_id, self.ctx)   # re-verified every time
                    self._check_adapter_outcomes(adapter)
                    continue
                require(not header_path.exists(),
                        f"{header_path} exists but the stage state does not record {run_id} as "
                        "finished: a partial or foreign adapter is never adopted")
                run = train_arm(self.ctx, arm, seed, self.runs / run_id, stop_after=stop_after,
                                protected=self.input_paths)
                if run.cursor < run.total_steps:
                    finished = False
                    continue
                self.adapters.mkdir(parents=True, exist_ok=True)
                dev_labels = self._dev_labels()          # binds + persists before the adapter exists
                provisional = Adapter(run_id=run_id, arm=arm, seed=seed, func_net=run.func_net,
                                      aux_net=run.aux_net, meta={}, file_sha256="", state_sha256="")
                report = source_report(
                    self.ctx, provisional, train_labels, dev_labels, self.cache,
                    bindings={"source_train": self.state["outcomes"]["source_train"],
                              "source_dev": self._dev_binding})
                header = freeze_adapter(run, self.adapters, source_report=report,
                                        protected=self.input_paths)
                runs[run_id] = {"adapter_file_sha256": header["weights_file_sha256"],
                                "adapter_state_sha256": header["adapter_state_sha256"],
                                "header_sha256": header["header_sha256"]}
                self._save_state()
        self.state["stages"]["train"]["complete"] = finished
        self._save_state()
        return finished

    def _adapters(self):
        self._require("train")
        adapters = {run_id: load_adapter(self.adapters, run_id, self.ctx)
                    for run_id in self.run_ids()}
        for adapter in adapters.values():
            self._check_adapter_outcomes(adapter)
        return adapters

    # --- features --------------------------------------------------------------------------
    def target_ids(self):
        return tuple(sorted(set(self.cohorts.roles["target_train"])
                            | set(self.cohorts.roles["target_eval"])))

    def _matrix_manifest_document(self):
        target = self.cohorts.manifest.target
        alphabet = self.cohorts.manifest.alphabet + fne.DELIMITER
        return {
            "schema": fm.MANIFEST_SCHEMA, "target": target.target, "assay": target.assay,
            "utility": {"name": "target_function", "units": target.units, "scale": target.scale,
                        "higher_is_better": True, "useful_threshold": target.useful_threshold},
            "grouping_rationale": self.cohorts.manifest.grouping_rationale,
            "candidates": {"identity_kind": fm.IDENTITY_COMPLETE_SEQUENCE, "alphabet": alphabet,
                           "length": self.cohorts.target_identity_length},
            "prior": {"kind": fb.PRIOR_UNIFORM,
                      "semantics": "every target candidate carries the same declared reference weight",
                      "provenance": {"status": "label_free",
                                     "description": "declared uniform; no outcomes used",
                                     "sequence_only_exposure": {"status": "none"}}},
            "notes": "Build D target roster: exact sorted target_train + target_eval union"}

    def _candidates_csv_text(self):
        identities, groups, splits = self.cache.identities, self.cache.groups, self.cache.splits
        lines = ["candidate_id,sequence,group_id,split,prior_log_weight"]
        for cid in self.target_ids():
            row = self.ctx.table.position[cid]
            lines.append(f"{cid},{identities[row]},{groups[row]},{splits[row]},0.0")
        return "\n".join(lines) + "\n"

    def _write_target_cache(self, ids, features, name, *, space, extractor, activation_record):
        header_path = self.features_dir / f"{name}.cache.json"
        npz_path = self.features_dir / f"{name}.cache.npz"
        cache = self.cache
        inputs = {**{label: path for label, path in self.input_paths.items()},
                  "candidate table": self.matrix_inputs / "candidates.csv",
                  "matrix manifest": self.matrix_inputs / "manifest.json"}
        fne._validate_cache_outputs(header_path, npz_path, inputs, False)
        positions = [self.ctx.table.position[cid] for cid in ids]
        logical = cache.conditions
        conditioning = {
            "candidate_masking": "none", "answer_free": True,
            "family": {"recorded": True,
                       "values": [None if logical[p] is None else logical[p]["family"]
                                  for p in positions]},
            "assay": {"recorded": True,
                      "values": [None if logical[p] is None else logical[p]["assay"]
                                 for p in positions]},
            "quality": {"recorded": True, "values": [None] * len(positions)}}
        identity = {"kind": fm.IDENTITY_COMPLETE_SEQUENCE,
                    "alphabet": self.cohorts.manifest.alphabet + fne.DELIMITER,
                    "length": self.cohorts.target_identity_length,
                    "construct_description": ("the complete heavy chain, the '|' chain delimiter "
                                              "and the complete light chain")}
        identities, groups, splits = cache.identities, cache.groups, cache.splits
        header_text, payload = fne._pooled_cache_files(
            npz_path.name, representation_id=name, ids=list(ids),
            sequences=[identities[p] for p in positions], groups=[groups[p] for p in positions],
            splits=[splits[p] for p in positions], features=features, space=space,
            identity=identity, extractor=extractor, activation=activation_record,
            conditioning=conditioning)
        fne._validate_cache_outputs(header_path, npz_path, inputs, False)
        fne._publish(header_path, npz_path, header_text, payload)
        loaded = fm.load_feature_cache(header_path, npz_path=npz_path)
        manifest = fm.load_matrix_manifest(self.matrix_inputs / "manifest.json")
        table = fb.read_candidates(self.matrix_inputs / "candidates.csv",
                                   alphabet=manifest.alphabet, length=manifest.length)
        fm.check_cache_against_roster(loaded, table, manifest)
        return {"representation": name, "header": str(header_path), "npz": str(npz_path),
                "cache_sha256": loaded.cache_sha256, "file_sha256": loaded.raw["file_sha256"]}

    def _native_space(self, ids):
        base = fne._pooled_space(self.cache, fne.POOL_ALL)
        schema = {**base.schema,
                  "selection": ("exact sorted target_train + target_eval union selected BY ID from "
                                "the verified FULL token cache; the full cache was pooled once and "
                                "its other rows (source cohorts) are not part of this roster"),
                  "parent_token_cache_sha256": self.cache.cache_sha256,
                  "parent_rows": len(self.cache.ids), "selected_rows": len(ids),
                  "selected_roster_sha256": fb.digest(list(ids)),
                  "unadapted": True, "source_adapted": False}
        return fb.FeatureSpace(
            kind=base.kind, names=base.names, schema=schema,
            history=base.history + ("exact target roster selected by id from the pooled full cache",),
            supervision=base.supervision)

    def _adapted_space(self, adapter, ids):
        ctx, cache = self.ctx, self.cache
        events = ctx.events
        computational = None
        if adapter.arm in AUX_ARMS and adapter.arm != "S":
            computational = ctx.answer_ids
        history, costs = derived_supervision(
            cache.space.supervision, source=self.cohorts.manifest.source,
            measured_ids=ctx.functional_ids, computational_ids=computational, arm=adapter.arm)
        meta = adapter.meta
        schema = {
            "construction": ("frozen shared-adapter FUNCTION-query features over the frozen native "
                             "token features of the unmasked complete candidates"),
            "source_adapted": True, "label_free": False, "unadapted": False,
            "representation": {
                "arm": adapter.arm, "source_seed": adapter.seed, "instruction": "FUNCTION",
                "readout": ("final LayerNorm output at the FUNCTION instruction token"),
                "width": ctx.spec.width, "student": ctx.spec.document(),
                "extraction": "eval mode, no_grad, padded to the table's longest row",
                "auxiliary_net_in_this_representation": (
                    "no: this is the functional pathway only" if adapter.arm == "S" else
                    ("shared with the auxiliary pathway" if adapter.arm in SHARED_ARMS else
                     "none"))},
            "parent": {"runtime_sha256": cache.runtime_sha256,
                       "token_cache_sha256": cache.cache_sha256,
                       "token_map_sha256": cache.token_map_sha256,
                       "activation": {"kind": cache.activation["kind"],
                                      "name": cache.activation["name"]},
                       "conditioning_mode": cache.conditioning_mode,
                       "parent_feature_supervision": cache.space.supervision.document()},
            "adapter": {"artifact_file_sha256": adapter.file_sha256,
                        "adapter_state_sha256": adapter.state_sha256,
                        "adapter_state_sha256_note": meta["state_sha256_note"],
                        "init_sha256": meta["bindings"]["init_sha256"],
                        "tape_sha256": meta["bindings"]["tape_sha256"],
                        "training_config_sha256": meta["bindings"]["training_config_sha256"],
                        "inputs_sha256": meta["bindings"]["inputs_sha256"],
                        "event_table_sha256": events.sha256,
                        "layouts_sha256": fc.layouts_sha256(ctx.layouts),
                        "cohort_sha256": self.cohorts.cohort_sha256,
                        "code_sha256": ctx.code_sha256},
            "selected_roster": {"ids_sha256": fb.digest(list(ids)), "rows": len(ids),
                                "rule": "exact sorted target_train + target_eval union by id"},
            "new_source_supervision": costs,
            "history_rule": ("a source-adapted representation is not label-free; native entries are "
                             "preserved and unknown stays unknown")}
        names = tuple(f"a{i}" for i in range(ctx.spec.width))
        text = ("frozen source-adapted student (FUNCTION query) over frozen native token features",
                f"adapter {adapter.run_id}; file {adapter.file_sha256[:16]}; tape "
                f"{meta['bindings']['tape_sha256'][:16]}")
        return fb.FeatureSpace(kind="functional_shared_adapter_function_query", names=names,
                               schema=schema, history=text, supervision=history)

    def features(self):
        """Extract target FUNCTION features from every frozen adapter and pool the native control, for
        the EXACT sorted target roster; write ordinary feature caches. No target outcome is opened. The
        in-memory context is verified first, before anything is written or a completed stage is adopted."""
        self._require("train")
        self.preflight()
        self.ctx.verify_all()        # the in-memory token metadata the adapters read, BEFORE any output
        if self._complete("features"):
            records = self.state["stages"]["features"]["caches"]
            for rep, record in records.items():
                loaded = fm.load_feature_cache(record["header"], npz_path=record["npz"])
                require(loaded.cache_sha256 == record["cache_sha256"],
                        f"feature cache {rep} no longer matches the stage state")
            return records
        partial = [str(path) for path in (self.features_dir, self.matrix_inputs)
                   if path.exists() and any(path.iterdir())]
        require(not partial,
                f"{partial} hold files from an interrupted features stage; partial caches are never "
                "adopted. Remove them and run the stage again")
        adapters = self._adapters()
        ids = self.target_ids()
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.matrix_inputs.mkdir(parents=True, exist_ok=True)
        self._write_text_new_or_equal(self.matrix_inputs / "candidates.csv",
                                      self._candidates_csv_text())
        self._write_json(self.matrix_inputs / "manifest.json", self._matrix_manifest_document())
        rows = [self.ctx.table.position[cid] for cid in ids]
        records = {}
        pooled = fne.pool_token_features(self.cache, fne.POOL_ALL)
        records[NATIVE_REPRESENTATION] = self._write_target_cache(
            ids, pooled[rows], NATIVE_REPRESENTATION, space=self._native_space(ids),
            extractor=self.cache.extractor,
            activation_record={
                "location": self.cache.activation["location"],
                "pooling": fne._pooling_text(fne.POOL_ALL),
                "token_map": (f"native layout; token_map_sha256={self.cache.token_map_sha256}")})
        source_hash = self.ctx.code_sha256
        for run_id in self.run_ids():
            adapter = adapters[run_id]
            matrix = extract_function_features(self.ctx.table, adapter.func_net, rows,
                                               batch_size=self.config.extraction_batch)
            extractor = {"kind": "external_features", "checkpoint_sha256": adapter.file_sha256,
                         "source_sha256": source_hash,
                         "description": (f"frozen shared-adapter student {run_id}: checkpoint_sha256 "
                                         "is the adapter FILE hash; source_sha256 digests the "
                                         "extraction code; the parent native runtime and token "
                                         "cache are bound in the feature schema")}
            records[run_id] = self._write_target_cache(
                ids, matrix, run_id, space=self._adapted_space(adapter, ids),
                extractor=extractor,
                activation_record={
                    "location": ("functional_shared_transfer student FUNCTION-query readout over "
                                 "frozen native token features"),
                    "pooling": "none: one FUNCTION query token attends over all valid native tokens",
                    "token_map": (f"token roles/chains/indices from the native token map; "
                                  f"token_map_sha256={self.cache.token_map_sha256}")})
        self.state["stages"]["features"] = {"complete": True, "roster_ids_sha256": fb.digest(list(ids)),
                                            "caches": records}
        self._save_state()
        return records

    # --- freeze / evaluate -------------------------------------------------------------------
    def representations(self):
        return [NATIVE_REPRESENTATION] + self.run_ids()

    def _plan_document(self):
        maps = [{"id": LINEAR_MAP, "kind": "linear", "center": True, "scale": "std",
                 "trace_normalize": True}]
        if self.config.include_rbf:
            maps.append({"id": RBF_MAP, "kind": "rbf", "center": True, "scale": "std",
                         "trace_normalize": True,
                         "bandwidth": {"rule": fm.BANDWIDTH_MEDIAN, "multiplier": 1.0}})
        return {"schema": fm.PLAN_SCHEMA, "representations": sorted(self.representations()),
                "maps": maps, "label_budgets": list(self.config.label_budgets),
                "seeds": list(self.config.acquisition_seeds),
                "baseline": {"tau": self.config.tau, "ridge_alpha": self.config.ridge_alpha,
                             "sir_proposal_draws": 64, "final_draws": 16,
                             "dpo_pressures": [0.5, 1.0]},
                "reference": {"representation": NATIVE_REPRESENTATION, "map": LINEAR_MAP},
                "bootstrap": {"replicates": self.config.bootstrap_replicates, "seed": 0,
                              "confidence": 0.9, "min_groups": 3},
                "notes": ("ONE combined matrix over native + every arm x source seed; fixed "
                          "readouts, budgets and acquisition seeds; the reference cell anchors paired "
                          "comparisons and is not a winner. Primary: source_seed == acquisition_seed.")}

    def _read_target_train(self):
        outcomes = fc.read_role_outcomes(self.inputs.target_train_outcomes,
                                         self.cohorts.roles["target_train"],
                                         label="target training outcomes")
        binding = outcome_binding("target_train", outcomes, self.cohorts.roles["target_train"],
                                  self.cohorts.manifest.target)
        return outcomes, binding

    @staticmethod
    def _check_freeze_outcomes(index, binding):
        """Compare the CURRENT target-training outcomes with the historical record the freeze index
        carries (what the cells were actually fitted on): never with the current file alone."""
        offered = index["common"]["offered"]
        require(offered["count"] == binding["count"]
                and offered["ids_sha256"] == binding["ids_sha256"]
                and offered["utilities_sha256"] == binding["utilities_sha256"],
                "the target-training outcomes now supplied are not the ones the matrix freeze was "
                "fitted on (the freeze index records their ids and values): the fitted cells are "
                f"never relabelled with a different version. {FRESH_OUTPUT}")

    def freeze(self):
        """Fit and freeze EVERY cell of the combined matrix, verify the freeze from disk, and leave the
        target-evaluation outcomes unopened. A complete verified freeze is adopted, never refitted; a
        partial or foreign directory is refused. The target-training outcomes are bound when first read
        and compared with the freeze index's historical record on every continuation and adoption."""
        self._require("features")
        self.preflight()
        plan_path = self.matrix_inputs / "plan.json"
        train_csv = self.matrix_inputs / "target_train_outcomes.csv"
        if self._complete("freeze"):
            verified = fm.verify_freeze(self.freeze_dir)
            require(verified["index"]["index_sha256"] == self.state["stages"]["freeze"]["index_sha256"],
                    "the freeze on disk is not the one the stage state recorded")
            outcomes, binding = self._read_target_train()
            self._check_freeze_outcomes(verified["index"], binding)    # the freeze's own record
            self._bind("target_train", outcomes, self.cohorts.roles["target_train"],
                       self.cohorts.manifest.target)                   # and the version first bound
            return self.state["stages"]["freeze"]
        plan_document = self._plan_document()
        expected_plan = fm.parse_plan(plan_document).fingerprint()
        records = self.state["stages"]["features"]["caches"]
        if self.freeze_dir.exists() and any(self.freeze_dir.iterdir()):
            verified = fm.verify_freeze(self.freeze_dir)        # raises unless a COMPLETE freeze
            index = verified["index"]
            require(index["plan_sha256"] == expected_plan
                    and all(index["representations"][rep]["cache_sha256"]
                            == records[rep]["cache_sha256"] for rep in self.representations()),
                    f"{self.freeze_dir} holds a verified freeze of a different plan or different "
                    "feature caches; it is not adopted")
            outcomes, binding = self._read_target_train()
            self._check_freeze_outcomes(index, binding)         # the HISTORICAL record, not the file
            self._bind("target_train", outcomes, self.cohorts.roles["target_train"],
                       self.cohorts.manifest.target)
            adopted = True
        else:
            outcomes, _ = self._read_target_train()
            binding, _ = self._bind("target_train", outcomes, self.cohorts.roles["target_train"],
                                    self.cohorts.manifest.target)   # persisted before anything is fitted
            self._write_text_new_or_equal(train_csv, "candidate_id,utility\n" + "".join(
                f"{cid},{outcomes[cid]!r}\n" for cid in sorted(outcomes)))
            self._write_json_new_or_equal(plan_path, plan_document)
            pending = self.matrix_inputs / "target_eval_outcomes.pending"
            require(not pending.exists(), f"{pending} must not exist: it stands in for the "
                                          "unopened evaluation file")
            try:
                fm.run_matrix(
                    candidates=self.matrix_inputs / "candidates.csv", train_outcomes=train_csv,
                    eval_outcomes=pending, manifest=self.matrix_inputs / "manifest.json",
                    plan=plan_path,
                    caches={rep: records[rep]["header"] for rep in self.representations()},
                    output_dir=self.freeze_dir)
                raise ValueError("internal error: the matrix evaluated although the evaluation "
                                 "outcomes were withheld")
            except ValueError as error:
                if not ("evaluation outcomes" in str(error) and "does not exist" in str(error)):
                    raise
            verified = fm.verify_freeze(self.freeze_dir)
            self._check_freeze_outcomes(verified["index"], binding)
            adopted = False
        fitted = fb.read_json(self.freeze_dir / fm.INPUTS_NAME)["files"]["training_outcomes"]
        counts = verified["index"]["counts"]
        self.state["stages"]["freeze"] = {
            "complete": True, "adopted_existing_freeze": adopted,
            "index_sha256": verified["index"]["index_sha256"], "plan_sha256": expected_plan,
            "counts": counts, "evaluation": "not opened",
            "target_training_outcomes": self.state["outcomes"]["target_train"],
            "target_training_outcomes_sha256": self.state["outcomes"]["target_train"][
                "canonical_sha256"],
            "fitted_training_outcomes_csv_sha256": fitted["sha256"],
            "binding_policy": ("semantic, verified against the freeze index's own record of the "
                               "offered ids and utilities, never stamped from the current file")}
        self._save_state()
        return self.state["stages"]["freeze"]

    def evaluate(self):
        """Open the target-evaluation outcomes for the first time, on the verified freeze. The
        evaluation version is bound before the matrix report is written and compared with the report's
        own recorded digest on every re-run: an old report is never paired with analysis computed
        against different labels."""
        self._require("freeze")
        self.preflight()
        verified = fm.verify_freeze(self.freeze_dir)
        require(verified["index"]["index_sha256"] == self.state["stages"]["freeze"]["index_sha256"],
                "the freeze on disk is not the one the stage state recorded")
        eval_path = Path(self.inputs.target_eval_outcomes)
        require(eval_path.is_file(), f"the target evaluation outcomes {eval_path} do not exist yet")
        outcomes = fc.read_role_outcomes(eval_path, self.cohorts.roles["target_eval"],
                                         label="target evaluation outcomes")
        support_ids = verified["index"]["common"]["support"]["ids"]
        matrix_identity = fb.digest({"ids": list(support_ids),
                                     "utilities": [float(outcomes[cid]) for cid in support_ids]})
        report_path = self.freeze_dir / fm.RESULTS_NAME
        if report_path.exists():
            results = fb.read_json(report_path)
            require(results.get("index_sha256") == verified["index"]["index_sha256"],
                    "an evaluation report of another freeze is on disk; it is not adopted")
            require(results["evaluation_outcomes"]["canonical_sha256"] == matrix_identity,
                    "the matrix evaluation report on disk was scored against different "
                    "target-evaluation outcomes than the ones now supplied: it is never re-paired "
                    f"with analysis of another version. {FRESH_OUTPUT}")
            binding, _ = self._bind("target_eval", outcomes, self.cohorts.roles["target_eval"],
                                    self.cohorts.manifest.target)
        else:
            binding, _ = self._bind("target_eval", outcomes, self.cohorts.roles["target_eval"],
                                    self.cohorts.manifest.target)    # persisted before the report
            results = fm.evaluate_matrix(self.freeze_dir, eval_path)
        analysis = analyse_matrix(self.config, self.cohorts, verified, outcomes, self.run_ids())
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_new_or_equal(self.analysis_dir / "target_analysis.json", analysis)
        recorded = self.state["stages"].get("evaluate")
        record = {"complete": True, "matrix_evaluation_sha256": fb.digest(results),
                  "analysis_sha256": fb.digest(analysis),
                  "target_evaluation_outcomes": binding,
                  "target_evaluation_outcomes_sha256": binding["canonical_sha256"]}
        if recorded and recorded.get("complete"):
            require(recorded["matrix_evaluation_sha256"] == record["matrix_evaluation_sha256"]
                    and recorded["analysis_sha256"] == record["analysis_sha256"],
                    "the evaluation on disk is not the one the stage state recorded")
            return analysis
        self.state["stages"]["evaluate"] = record
        self._save_state()
        return analysis

    # --- panels and the report -----------------------------------------------------------------
    def _panel_rows(self, role, limit, salt):
        rows = self.cohorts.rows(self.cache, role)
        rng = np.random.default_rng([self.config.tape_seed, 77, salt])
        if len(rows) > limit:
            chosen = sorted(int(i) for i in rng.choice(len(rows), size=limit, replace=False))
            rows = [rows[i] for i in chosen]
        return rows

    def build_panels(self, *, include_target):
        cfg = self.config
        training_ids = set(self.ctx.events.ids())
        panels, unsupported = {}, {}
        for label, role, salt in (("seen_sequences", "source_aux", 1), ("new_sequences", "source_dev", 2)
                                  ) + ((("target_sequences", "target_eval", 3),) if include_target
                                       else ()):
            rows = self._panel_rows(role, cfg.panel_sequences, salt)
            for dependency, universe in (("seen_dependency", fc.UNIVERSE_TRAIN),
                                         ("held_out_dependency", fc.UNIVERSE_HELD_OUT)):
                name = f"{label}__{dependency}"
                try:
                    panels[name] = make_panel(
                        self.ctx, name, rows, universe=universe, instances=cfg.panel_instances,
                        seed=cfg.tape_seed + 1000 + salt,
                        exclude_ids=training_ids if universe == fc.UNIVERSE_TRAIN else ())
                except PanelUnsupported as error:
                    unsupported[name] = str(error)
        return panels, unsupported

    def panels(self):
        """Score every auxiliary head on the panels AFTER all freezes; nothing is tuned."""
        self._require("freeze")
        self.preflight()
        adapters = self._adapters()
        panels, unsupported = self.build_panels(include_target=True)
        results = {}
        for name, panel in panels.items():
            results[name] = {"events": len(panel.table), "event_table_sha256": panel.table.sha256,
                             "coverage": fc.coverage_report(panel.table),
                             "balance": fc.balance_report(panel.table),
                             "layout_shapes": panel.table.config["layout_shapes"],
                             "excluded_rows_below_minimum_coordinates": list(panel.excluded),
                             **panel_note_summary(panel.table.unsupported), "by_run": {}}
            for run_id in self.run_ids():
                adapter = adapters[run_id]
                if adapter.aux_net is None:
                    continue
                modes = ("composed", "direct") if adapter.arm == "R" else ("composed",)
                results[name]["by_run"][run_id] = {mode: evaluate_panel(self.ctx, adapter, panel, mode)
                                                   for mode in modes}
        out = {"panels": results, "unsupported_panels": unsupported,
               "note": ("F and FX have no auxiliary supervision and are not scored. R is scored on "
                        "composed instructions (which it never saw) and on its own direct "
                        "instructions. majority_rate is each group's empirical best-constant baseline "
                        "(the accuracy of always guessing that group's own commonest observed answer); "
                        "it is not a chance rate, and a group that mixes RES (many-class residue) and "
                        "EQ (binary) outputs has no single meaningful chance rate, so read an output "
                        "type against its own baseline and the primary EQ stratum by output. Rows "
                        "below the task's minimum coordinates are excluded and listed. "
                        "unsupported_control_notes counts unsupported-control and fallback NOTES (the "
                        "instance still exists and its events are in the panel); skipped_instance_notes "
                        "counts instances actually skipped for want of a complete instance; each "
                        "panel's events is what it holds. Target panels are scored only after every "
                        "freeze; nothing here selects a method.")}
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_new_or_equal(self.analysis_dir / "auxiliary_panels.json", out)
        recorded = self.state["stages"].get("panels")
        if recorded and recorded.get("complete"):
            require(recorded["panels_sha256"] == fb.digest(out),
                    "the auxiliary panels on disk are not the ones the stage state recorded")
            return out
        self.state["stages"]["panels"] = {"complete": True, "panels_sha256": fb.digest(out)}
        self._save_state()
        return out

    def report(self):
        self._require("evaluate")
        self._require("panels")
        self.preflight()
        adapters = self._adapters()
        integrity = {run_id: (file_sha256(self.adapters / adapters[run_id].meta["weights_file"])
                              == self.state["stages"]["train"]["runs"][run_id][
                                  "adapter_file_sha256"]) for run_id in self.run_ids()}
        runs = {}
        for run_id in self.run_ids():
            meta = adapters[run_id].meta
            runs[run_id] = {"cost": meta["cost"], "source_report": meta["source_report"],
                            "adapter_file_sha256": meta["weights_file_sha256"],
                            "adapter_state_sha256": meta["adapter_state_sha256"]}
        runtime = {run_id: self._run_curve(run_id) for run_id in self.run_ids()}
        report = {
            "schema": REPORT_SCHEMA, "claims": CLAIMS, "executed": True,
            "inputs": fb.read_json(self.root / "inputs.json"),
            "stage_state": {name: value for name, value in self.state["stages"].items()
                            if name != "report"},
            "outcome_bindings": self.state["outcomes"],
            "runs": runs, "training_curves": runtime,
            "source_preservation": source_preservation(self.config, runs),
            "target": fb.read_json(self.analysis_dir / "target_analysis.json"),
            "auxiliary_panels": fb.read_json(self.analysis_dir / "auxiliary_panels.json"),
            "integrity": {"adapters_unchanged_after_target_fitting": integrity,
                          "note": ("an integrity check only: unchanged frozen weights after target "
                                   "fitting are not evidence against negative transfer")}}
        self._write_json_new_or_equal(self.root / REPORT_NAME, report)
        recorded = self.state["stages"].get("report")
        if recorded and recorded.get("complete"):
            require(recorded["report_sha256"] == fb.digest(report),
                    "the report on disk is not the one the stage state recorded")
            return report
        self.state["stages"]["report"] = {"complete": True, "report_sha256": fb.digest(report)}
        self._save_state()
        return report

    def _run_curve(self, run_id):
        header = fb.read_json(self.runs / run_id / CHECKPOINT_NAME)
        return {"functional_loss": [entry["functional_loss"] for entry in header["log"]],
                "auxiliary_loss": [entry.get("auxiliary_loss") for entry in header["log"]],
                "diagnostics": header["diagnostics"]}

    def run(self, stages=STAGES):
        out = {}
        for stage in stages:
            require(stage in STAGES, f"unknown stage {stage!r}; the stages are {STAGES}")
            out[stage] = getattr(self, stage)()
        return out


# ---------------------------------------------------------------------------
# analysis of the frozen matrix
# ---------------------------------------------------------------------------

def source_preservation(config, runs):
    """SOURCE-DEV functional error of every arm versus F after source training, per seed (paired)."""
    table = {}
    for seed in config.source_seeds:
        reference = runs[rep_id("F", seed)]["source_report"]["source_dev"]["mse"]
        for arm in config.arm_list():
            report = runs[rep_id(arm, seed)]["source_report"]
            entry = table.setdefault(arm, {"per_seed": {}})
            entry["per_seed"][str(seed)] = {
                "source_dev_mse": report["source_dev"]["mse"],
                "difference_from_F": report["source_dev"]["mse"] - reference,
                "source_train_mse": report["source_train"]["mse"],
                "constant_mean_source_dev_mse": report["source_dev"][
                    "constant_source_train_mean_mse"]}
    for arm, entry in table.items():
        diffs = [item["difference_from_F"] for item in entry["per_seed"].values()]
        entry["mean_difference_from_F"] = float(np.mean(diffs))
        entry["seeds_worse_than_F"] = int(sum(1 for d in diffs if d > 0))
        entry["seeds"] = len(diffs)
    return {"measurement": ("SOURCE-DEV functional MSE after source training versus F, paired by "
                            "source seed. A positive difference means worse than F (source "
                            "forgetting or interference); it is reported, never hidden. Every error "
                            "was scored on ONE bound version of the source-development outcomes."),
            "arms": table}


def analyse_matrix(config, cohorts, verified, outcomes, run_ids):
    """Per-cell MSE/rank/per-group error from the VERIFIED frozen artifacts, the diagonal paired
    differences (source_seed == acquisition_seed), learning curves and the descriptive off-diagonal
    table. No cell is selected; no p-value is computed."""
    index = verified["index"]
    support = index["common"]["support"]
    ids, groups = support["ids"], support["groups"]
    truth = np.array([outcomes[cid] for cid in ids], dtype=np.float64)
    group_members = {g: [i for i, value in enumerate(groups) if value == g] for g in sorted(set(groups))}
    cells = {}
    for key in verified["keys"]:
        envelope = verified["envelopes"][key.cell_id]
        if envelope["status"] != fm.STATUS_OK:
            cells[key.cell_id] = {"status": envelope["status"], "reason": envelope["reason"]}
            continue
        predicted = np.array(envelope["artifact"]["predictions"]["utility"], dtype=np.float64)
        squared = (predicted - truth) ** 2
        rank = fb.spearman(predicted, truth)
        cells[key.cell_id] = {
            "status": "ok", "representation": key.representation, "map": key.map_id,
            "label_budget": key.label_budget, "acquisition_seed": key.seed,
            "mse": float(squared.mean()), "rmse": float(math.sqrt(squared.mean())),
            "spearman": rank["value"], "spearman_reason": rank["reason"],
            "per_group_mse": {g: float(squared[members].mean()) for g, members in group_members.items()},
            "capacity": envelope["capacity"]}

    def cell(rep, map_id, budget, seed):
        return cells.get(fm.CellKey(rep, map_id, budget, seed).cell_id)
    diagonal = sorted(set(config.source_seeds) & set(config.acquisition_seeds))
    map_ids = [spec["id"] for spec in verified["index"]["plan"]["maps"]]
    comparisons = {}

    def paired(left_arm, right_arm, map_id, budget):
        per_seed, per_group_values = [], {g: [] for g in group_members}
        for seed in diagonal:
            left = cell(right_arm_rep(left_arm, seed), map_id, budget, seed)
            right = cell(right_arm_rep(right_arm, seed), map_id, budget, seed)
            if not left or not right or left["status"] != "ok" or right["status"] != "ok":
                return None
            per_seed.append({"seed": seed, "mse_difference": left["mse"] - right["mse"],
                             "per_group": {g: left["per_group_mse"][g] - right["per_group_mse"][g]
                                           for g in group_members}})
            for g in group_members:
                per_group_values[g].append(per_seed[-1]["per_group"][g])
        diffs = [item["mse_difference"] for item in per_seed]
        return {"paired_seeds": diagonal, "per_seed": per_seed,
                "mean_mse_difference": float(np.mean(diffs)),
                "seeds_with_lower_mse": int(sum(1 for d in diffs if d < 0)),
                "per_group_mean_difference": {g: float(np.mean(v)) for g, v in per_group_values.items()},
                "groups_with_lower_mse": int(sum(1 for v in per_group_values.values()
                                                 if np.mean(v) < 0)),
                "sign": "left minus right; negative means the left arm has the lower target MSE",
                "n_groups": len(group_members), "n_seeds": len(diagonal)}

    def right_arm_rep(arm, seed):
        return NATIVE_REPRESENTATION if arm == NATIVE_REPRESENTATION else rep_id(arm, seed)
    arms = list(config.arm_list())
    pairs = [(arm, "F") for arm in arms if arm != "F"]
    pairs += [(left, right) for left, right in (("C", "Q"), ("C", "R"))
              if left in arms and right in arms]
    pairs += [(arm, NATIVE_REPRESENTATION) for arm in arms]
    for map_id in map_ids:
        for budget in config.label_budgets:
            for left, right in pairs:
                result = paired(left, right, map_id, budget)
                comparisons[f"{left}-minus-{right}|{map_id}|b{budget}"] = result
    curves = {}
    for map_id in map_ids:
        for arm in [NATIVE_REPRESENTATION] + arms:
            entry = {}
            for budget in config.label_budgets:
                values = []
                for seed in (diagonal if arm != NATIVE_REPRESENTATION else config.acquisition_seeds):
                    item = cell(right_arm_rep(arm, seed), map_id, budget, seed)
                    if item and item["status"] == "ok":
                        values.append({"seed": seed, "mse": item["mse"], "spearman": item["spearman"]})
                entry[f"b{budget}"] = {"per_seed": values,
                                       "mean_mse": float(np.mean([v["mse"] for v in values]))
                                       if values else None}
            curves[f"{arm}|{map_id}"] = entry
    off_diagonal = [item for item in cells.values() if item.get("status") == "ok"
                    and item["representation"] != NATIVE_REPRESENTATION]
    for item in off_diagonal:
        arm_seed = item["representation"].rsplit("_s", 1)
        item["source_seed"] = int(arm_seed[1])
        item["arm"] = arm_seed[0]
        item["diagonal"] = item["source_seed"] == item["acquisition_seed"]
    return {
        "map_primary": "linear", "maps": map_ids,
        "primary_rule": ("diagonal pairs only (source_seed == acquisition_seed); the primary "
                         "readout is the train-only centred/scaled linear ridge. An RBF map, if "
                         "present, is a labelled secondary analysis. No winner is selected."),
        "diagonal_seeds": diagonal, "target_evaluation_groups": sorted(group_members),
        "comparisons": comparisons, "learning_curves": curves,
        "descriptive_all_cells": [item for item in cells.values()],
        "off_diagonal_note": ("every off-diagonal cell is descriptive; the nine source/acquisition "
                              "combinations share candidates and are not independent replications"),
        "evaluation_outcomes": {"candidates": len(ids)}}


def run_transfer(inputs, output_dir, config, *, stages=STAGES):
    """Python entry point: open (validate) and run the requested stages in order."""
    return TransferRun.open(inputs, output_dir, config).run(stages)


# ---------------------------------------------------------------------------
# selftest and bounded synthetic pilot
# ---------------------------------------------------------------------------

def selftest_config():
    """Much smaller than the pilot, but genuine training for all five arms, two paired seeds, two
    nested budgets, resume, probes and one combined matrix."""
    return TransferConfig(
        steps=6, batch=8, aux_batch=8, instances_per_sequence=2, lam=0.1, lr=3e-3, clip_norm=1.0,
        width=32, heads=2, ffn=64, blocks=2, source_seeds=(0, 1), acquisition_seeds=(0, 1),
        label_budgets=(4, 8), checkpoint_every=2, probe_steps=(0, 3, 5), panel_instances=1,
        panel_sequences=8, bootstrap_replicates=50, extraction_batch=16)


def pilot_config():
    """The bounded pilot: 256 macrosteps of 32, three paired source seeds, nested budgets 16/64."""
    return TransferConfig()


@contextlib.contextmanager
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _inputs_from(paths):
    return TransferInputs(
        token_header=paths["token_header"], token_npz=paths["token_npz"], cohorts=paths["cohorts"],
        source_train_outcomes=paths["source_train_outcomes"],
        source_dev_outcomes=paths["source_dev_outcomes"],
        target_train_outcomes=paths["target_train_outcomes"],
        target_eval_outcomes=paths["target_eval_outcomes"])


def _refused(call):
    try:
        call()
    except ValueError:
        return True
    return False


def _selftest_in(work, seed):
    from . import _functional_transfer_synthetic as synth
    checks = []

    def check(name, holds, detail=None):
        checks.append({"name": name, "holds": bool(holds), "detail": detail})
    spec = dataclasses.replace(synth.SELFTEST_SPEC, seed=seed)
    world = synth.build_world(spec)
    paths = synth.write_inputs(world, work / "inputs", model_seed=seed, withhold_eval=True)
    inputs = _inputs_from(paths)
    config = selftest_config()
    out = work / "run"
    run = TransferRun.open(inputs, out, config)
    ctx = run.ctx
    check("data::real_tiny_random_dplm_token_cache_loaded_and_cohorts_validated",
          ctx.cohorts.layout.document() == {"heavy": 4, "light": 4, "alphabet": "ACDE"}
          and len(run.cache.ids) == len(world.rows),
          {"rows": len(world.rows), "hidden": ctx.table.hidden})
    check("tapes::event_table_covers_training_and_excludes_every_held_out_two_swap",
          ctx.coverage["complete"] and not any(e.held_out for e in ctx.events.events),
          ctx.coverage)
    q = ctx.seed_tapes[0].q_report
    check("tapes::shuffled_targets_change_real_values_in_supported_strata",
          q["changed_values"] > 0 and q["changed_fraction"] < 1.0,
          {"changed_fraction": q["changed_fraction"], "unsupported": len(q["unsupported_strata"])})
    run.prepare()
    # --- exact F/S functional parity at every macrostep, and shared-gradient wiring ---
    f_run, s_run = ArmRun(ctx, "F", 0), ArmRun(ctx, "S", 0)
    parity = True
    for _ in range(config.steps):
        f_run.macrostep()
        s_run.macrostep()
        parity = parity and f_run.functional_pathway_digest() == s_run.functional_pathway_digest()
    check("parity::S_functional_pathway_and_optimizer_equal_F_after_every_macrostep", parity)
    c_run, s_aux = ArmRun(ctx, "C", 0), ArmRun(ctx, "S", 0)
    rows = [ctx.table.position[cid] for cid in ctx.functional_ids[:4]]
    c_before = c_run.function_features(rows)
    s_before = s_aux.function_features(rows)
    c_run.auxiliary_only_step()
    s_aux.auxiliary_only_step()
    check("wiring::auxiliary_only_step_moves_C_function_features_but_not_S",
          float(np.abs(c_run.function_features(rows) - c_before).max()) > 0.0
          and bool(np.array_equal(s_aux.function_features(rows), s_before)))
    probe = ArmRun(ctx, "C", 0)
    probe.macrostep()
    diag = probe.diagnostics[0]
    check("wiring::shared_gradient_norms_are_positive_and_the_probe_recorded_clipping_and_update",
          diag["functional_grad_norm_shared"] > 0 and diag["auxiliary_grad_norm_shared"] > 0
          and diag["update_norm_shared"] > 0 and "clip_coefficient" in diag, diag)
    # --- resume parity ---
    straight = train_arm(ctx, "C", 1, work / "straight")
    train_arm(ctx, "C", 1, work / "resumed", stop_after=3)
    resumed = train_arm(ctx, "C", 1, work / "resumed")
    check("resume::interrupted_and_resumed_run_equals_the_uninterrupted_run_exactly",
          straight.deterministic_result() == resumed.deterministic_result())
    # --- the real pipeline: all arms/seeds, one combined freeze, only then the evaluation outcomes ---
    run.train()
    check("train::every_arm_and_seed_trained_and_frozen",
          run._complete("train") and len(run.run_ids()) == 10)
    run.features()
    freeze_state = run.freeze()
    eval_present_at_freeze = Path(paths["target_eval_outcomes"]).exists()
    check("matrix::all_representations_frozen_before_any_target_evaluation_outcome_existed",
          freeze_state["complete"] and not eval_present_at_freeze, freeze_state["counts"])
    shutil.copy(paths["held_eval_outcomes"], paths["target_eval_outcomes"])
    analysis = run.evaluate()
    check("matrix::every_planned_cell_fitted_with_no_failure",
          freeze_state["counts"]["failed"] == 0 and freeze_state["counts"]["unsupported"] == 0,
          freeze_state["counts"])
    run.panels()
    report = run.report()
    check("report::source_preservation_compares_dev_error_with_F",
          set(report["source_preservation"]["arms"]) == set(config.arm_list()))
    check("report::counts_keep_measured_outcomes_answers_and_presentations_separate",
          all(run_cost["cost"]["unique_measured_source_outcomes"] <= len(ctx.functional_ids)
              for run_cost in report["runs"].values())
          and report["runs"]["C_s0"]["cost"]["unique_computational_answers"] <= len(ctx.events))
    check("integrity::frozen_adapters_unchanged_after_target_fitting",
          all(report["integrity"]["adapters_unchanged_after_target_fitting"].values()))
    check("report::no_winner_is_selected_and_diagonal_comparisons_exist",
          "descriptive_all_cells" in analysis and len(analysis["diagonal_seeds"]) == 2)
    check("selftest::no_cuda_was_initialised", not torch.cuda.is_initialized())
    return {"checks": checks, "report_path": str(out / REPORT_NAME)}


def selftest(output_dir, seed=0):
    """Real tiny random CPU DPLM -> real token cache -> every arm trained -> frozen -> one combined
    matrix. Everything is written below a directory this call creates. Qualifies the machinery only."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="shared_transfer_selftest_", dir=out))
    with single_thread():
        body = _selftest_in(work, seed)
    report = {"schema": SELFTEST_SCHEMA, "seed": seed, "work_directory": str(work),
              "passed": bool(all(item["holds"] for item in body["checks"])),
              "checks": body["checks"], "pipeline_report": body["report_path"],
              "claims": CLAIMS,
              "note": ("synthetic qualification of the shared learner on a tiny RANDOM CPU model. It "
                       "asserts mechanics, never a scientific win.")}
    fb.write_json(work / SELFTEST_REPORT_NAME, report)
    return report


def pilot(output_dir, seed=0, *, config=None):
    """The bounded synthetic pilot (128 computed synthetic source labels - computational functions, not
    measured outcomes - 512 auxiliary sequences, 64 source-dev, 128 + 128 target; 256 macrosteps of
    32; three paired seeds; budgets 16/64). It is
    NOT started by anything here: call it explicitly. A negative result is a valid result."""
    from . import _functional_transfer_synthetic as synth
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="shared_transfer_pilot_", dir=out))
    config = config or pilot_config()
    world = synth.build_world(dataclasses.replace(synth.PILOT_SPEC, seed=seed))
    paths = synth.write_inputs(world, work / "inputs", model_seed=seed)
    with single_thread():
        transfer = TransferRun.open(_inputs_from(paths), work / "run", config)
        transfer.run()
    return {"work_directory": str(work), "report": str(work / "run" / REPORT_NAME),
            "stage_state": transfer.state["stages"]}
