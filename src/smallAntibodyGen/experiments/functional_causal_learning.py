"""Build B: learned evidence, composition and interchange-intervention pilot on a finite synthetic world.

What this is. A real learner, not a specification. A small DeepSets student reads RAW observation events
(fact index, bit, numeric reliability) of four hidden Boolean facts, learns a four-block hidden state with a
masked SUM over events, and a learned program encoder and prediction head answer typed programs
(READ / NOT / AND / XOR) with a Bernoulli logit. The exact finite-world teacher
(``functional_evidence``) supplies soft-label targets; it never enters the learned forward. The question is
whether hidden interchange-intervention training (IIT) teaches shared internal computations that improve
held-out composition and controllability, against factual-only training, block-marginal multitask training,
input-level counterfactual augmentation and a corrupted-supervision IIT control. The new method is NOT
required to win; a negative or inconclusive result is a valid result and passing software tests never forces
IIT to win. All tasks are synthetic; nothing here is a statement about biology.

Arms (one initialisation per seed cloned into every arm; one objective / update rule / fixed endpoint):
  F   factual only                        loss = BCE(base prediction, p_base)
  M   factual + block-marginal multitask   + lam_marginal * BCE(head_k(block_k), P(X_k = 1 | evidence))
  A   factual + input counterfactual        + lam_cf * BCE(prediction on the RAW-evidence hybrid, p_cf)
  I   factual + hidden interchange          + lam_cf * BCE(prediction with the DONOR's hidden block of the
                                              site substituted into the base, p_cf)
  Ic  factual + corrupted-supervision IIT    I, but p_cf is replaced by a per-example corrupted target
A and I take the same base loss and the same counterfactual targets and both run TWO evidence encodes and
TWO prediction-head calls (one shared program encode) at a common padded event width, so dense compute is
matched; the useful (unpadded) event rows differ and are reported. F and M differ from A/I in compute and in
label information and are reported as such, not as matched. Perfect posterior-probability labels carry more
information than noisy measured outcomes.

One frozen SOURCE-only tape per seed (``SourceTape``) holds every base, donor, site, query and exact teacher
target; every arm trains on the same tape, with no resampling after an arm diverges. Every base, donor and
resulting one-factor hybrid belongs to the SOURCE support. The corrupted Ic targets are built once from that
tape by a seeded per-example random permutation of the values within (function-orbit, site) strata, so the
histograms are exact and query and site inputs are unchanged; the real value changes (changed and unchanged
rates, absolute difference, KL, unsupported strata) are measured and reported, never assumed. A fixed
site permutation, Boolean complement or deterministic value relabelling is not accepted as a control: it is
a rule a model could learn, and the maximum possible change is a descriptive bound, not a target.

Stage order (``CausalRun``): prepare (verify the frozen world, build the tapes) -> train (every arm and seed,
checkpointed at update boundaries) -> freeze (every source state is published and hashed) -> evaluate (the
held-out panels open only now) -> transfer (adapt each frozen arm and a scratch control on nested new-
composition budgets, freeze every adapter, only then open the transfer-evaluation outcomes) -> export (the
learned query-conditioned penultimate activation goes to the Build A matrix) -> report. Resume is verified
at every boundary; a tampered target, program, site or evidence row refuses BEFORE any optimizer mutation.

Claim boundaries. The q7 split holds out unseen LABELLED evidence contexts, not evidence permutation orbits.
Program splits hold out exact truth tables and S4 orbits. Excess log loss / KL, excess Brier and calibration
are measured outcomes of a learned model, not properties guaranteed by construction. A patched prediction is
compared with BOTH the exact counterfactual oracle and the ordinary prediction on the raw-evidence hybrid:
two wrong predictions can agree perfectly. The 21,168 primary intervention cases come from only 14 base
contexts, so exact finite-panel means and paired seed differences are reported, never row-bootstrap
population evidence. The Build A policy utility is the posterior truth probability of a statement: it
selects high-probability statements and is NOT a reasoning-accuracy measure.
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
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import functional_baseline as fb
from . import functional_evidence as fe
from . import functional_matrix as fm
from . import functional_shared_transfer as fst
from .her2_runtime import require, sha256 as file_sha256

CONFIG_SCHEMA = "functional-causal-config/1"
TAPE_SCHEMA = "functional-causal-tape/1"
INIT_SCHEMA = "functional-causal-init/1"
CHECKPOINT_SCHEMA = "functional-causal-checkpoint/1"
FROZEN_SCHEMA = "functional-causal-frozen/1"
ADAPTER_SCHEMA = "functional-causal-adapter/1"
STATE_SCHEMA = "functional-causal-state/1"
REPORT_SCHEMA = "functional-causal-report/1"
SELFTEST_SCHEMA = "functional-causal-selftest/1"
REPORT_NAME = "functional_causal_report.json"
SELFTEST_REPORT_NAME = "causal_selftest_report.json"

ARMS = ("F", "M", "A", "I", "Ic")
CF_ARMS = ("A", "I", "Ic")
SCRATCH = "scratch"
STAGES = ("prepare", "train", "freeze", "evaluate", "transfer", "export", "report")
N_BLOCKS = fe.N_FACTS
EVENT_FEATURES = fe.N_FACTS + 2 + 1
N_SOURCE_FUNCTIONS = 48
CHANGED_THRESHOLD = 0.1
UNCHANGED_TOLERANCE = 1e-12
#: the fewest events a (function orbit, site) stratum needs for a per-example randomization to mean
#: anything (the same floor as the Build D shuffled-target control)
MIN_STRATUM = 4
FRESH_OUTPUT = "use a fresh output directory (nothing here is relabelled or overwritten)"

PANEL_REUSE = (
    "The export follows a completed transfer stage, so the transfer-evaluation panel (the 14 transfer-eval "
    "contexts crossed with the 12 transfer functions) was ALREADY opened and scored there. The bridge reuses "
    "that same panel for a secondary analysis whose settings (feature kind, readout map, label budgets, "
    "acquisition seeds, bootstrap) are fixed in advance; it is not a fresh confirmation, and a new local "
    "evaluation file written after the matrix freeze does not make the outcomes globally unseen. The "
    "first-run freeze barriers of the source and transfer stages are preserved and each stage's opened-split "
    "audit is kept as recorded, never erased to look fresh.")

CLAIMS = {
    "scope": ("A learned DeepSets evidence encoder, program encoder and prediction head on a finite "
              "synthetic world with an exact teacher. IIT is compared with factual-only, multitask, "
              "input-augmentation and corrupted-supervision controls. It is not required to win; a "
              "negative or inconclusive result is a valid result. Nothing here is a statement about "
              "biology or a molecular mechanism."),
    "split": ("The q7 split holds out unseen LABELLED evidence contexts, not evidence permutation "
              "orbits: signatures stay ordered by fact identity. Program splits hold out exact truth "
              "tables and their S4 variable-renaming orbits."),
    "matching": ("A and I share the base loss, the counterfactual targets, the tape and the padded "
                 "dense compute (two encodes, two head calls at one event width); their useful event "
                 "rows differ and are reported. F and M differ from A/I in compute and label "
                 "information and are not called matched."),
    "information": ("Exact posterior-probability labels carry more information than noisy measured "
                    "outcomes; counterfactual and marginal labels are derived teacher answers and are "
                    "counted separately from factual labels."),
    "fidelity": ("A patched prediction is compared with the exact counterfactual oracle AND with the "
                 "ordinary prediction on the raw-evidence hybrid. Agreement alone is not causal "
                 "fidelity: two wrong predictions can agree."),
    "panel": ("The 21168 primary intervention cases come from 14 base contexts; they are correlated, so "
              "exact finite-panel means and paired seed differences are reported and no row-bootstrap "
              "population claim is made."),
    "alignment": ("A block declaration is an alignment hypothesis being trained, not a correct "
                  "representation by construction; the corrupted-supervision control is not proof of a "
                  "uniquely correct causal alignment."),
    "bridge": ("The exported feature is the learned query-conditioned penultimate activation. The Build "
               "A policy utility is a posterior truth probability: it selects high-probability "
               "statements and is not a reasoning-accuracy measure. The bridge metric is held-out "
               "readout prediction error on context + program pairs."),
    "panel_reuse": PANEL_REUSE,
}


# ---------------------------------------------------------------------------
# small helpers: digests, read-only arrays, metrics
# ---------------------------------------------------------------------------

def _array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode("ascii")
                          + array.tobytes()).hexdigest()


def _digest_arrays(named):
    """SHA-256 over ``(name, dtype, shape, bytes)`` of each array, in the given order."""
    hasher = hashlib.sha256()
    for name, array in named:
        array = np.ascontiguousarray(array)
        hasher.update(f"{name}|{array.dtype.str}|{array.shape}|".encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _readonly(array):
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def _owned(arrays):
    """An immutable name -> read-only-copy mapping (a dict handed in is copied, never kept)."""
    return MappingProxyType({name: _readonly(array) for name, array in arrays.items()})


def code_sha256():
    return fb.digest({"functional_evidence": file_sha256(Path(fe.__file__)),
                      "functional_causal_learning": file_sha256(Path(__file__))})


def sigmoid64(logits):
    """A numerically stable float64 sigmoid."""
    x = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def _clip_probability(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)


def bernoulli_kl(p, q):
    """``KL(Bernoulli(p) || Bernoulli(q))`` elementwise (excess log loss of predicting ``q`` when the
    exact answer is ``p``), with ``0 log 0 = 0``."""
    p = np.asarray(p, dtype=np.float64)
    q = _clip_probability(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)
        second = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1.0)) - np.log(1 - q)), 0.0)
    return first + second


def bernoulli_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
                 + np.where(p < 1, (1 - p) * np.log(np.where(p < 1, 1 - p, 1.0)), 0.0))


def soft_bce(logits, targets):
    """Per-row stable soft-label Bernoulli cross-entropy: ``softplus(z) - t z`` (nats)."""
    return F.softplus(logits) - targets * logits


def _load_plain(path):
    """Plain numeric tensors and primitive containers only (``weights_only=True``). Any failure, a
    pickled object in place of tensors included, is a ``ValueError`` with the file named."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"{path} cannot be loaded as plain numeric tensors with weights_only=True "
                         f"({type(error).__name__}: {error})") from error


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = (
    "steps", "batch", "lr", "clip_norm", "lam_cf", "lam_marginal", "event_hidden", "block_dim",
    "program_embedding", "program_hidden", "head_hidden", "seeds", "arms", "tape_seed", "variants",
    "cancel_max", "null_max", "eval_variant", "checkpoint_every", "transfer_budgets", "adapt_steps",
    "adapt_lr", "adapter_hidden", "export_label_budgets", "export_acquisition_seeds",
    "export_bootstrap_replicates", "export_ridge_alpha", "export_tau")
_TRAINING_FIELDS = (
    "steps", "batch", "lr", "clip_norm", "lam_cf", "lam_marginal", "event_hidden", "block_dim",
    "program_embedding", "program_hidden", "head_hidden", "tape_seed", "variants", "cancel_max",
    "null_max")


def _unique_ints(values, name, *, minimum, maximum=None):
    require(isinstance(values, (list, tuple)) and values, f"config.{name} must be a non-empty list")
    out = tuple(fb._integer(item, f"config.{name} entry", minimum=minimum) for item in values)
    require(len(set(out)) == len(out), f"config.{name} repeats a value")
    require(maximum is None or max(out) <= maximum, f"config.{name} entries must be <= {maximum}")
    return out


@dataclass(frozen=True)
class CausalConfig:
    """Every fixed choice. Defaults are the bounded pilot: 3 paired seeds, 1000 source updates, batch
    64. Nothing is selected from a held-out label: the endpoint and hyper-parameters are fixed in
    advance."""

    steps: int = 1000
    batch: int = 64
    lr: float = 3e-3
    clip_norm: float = 5.0
    lam_cf: float = 1.0
    lam_marginal: float = 1.0
    event_hidden: int = 32
    block_dim: int = 8
    program_embedding: int = 16
    program_hidden: int = 32
    head_hidden: int = 64
    seeds: tuple = (0, 1, 2)
    arms: tuple = ARMS
    tape_seed: int = 0
    variants: int = 4
    cancel_max: int = 1
    null_max: int = 2
    eval_variant: int = 1
    checkpoint_every: int = 100
    transfer_budgets: tuple = (0, 16, 64)
    adapt_steps: int = 300
    adapt_lr: float = 1e-2
    adapter_hidden: int = 32
    export_label_budgets: tuple = (16, 64)
    export_acquisition_seeds: tuple = (0, 1, 2)
    export_bootstrap_replicates: int = 50
    export_ridge_alpha: float = 1.0
    export_tau: float = 1.0

    def __post_init__(self):
        def put(name, value):
            object.__setattr__(self, name, value)
        for name, minimum in (("steps", 1), ("batch", 1), ("event_hidden", 1), ("program_embedding", 1),
                              ("program_hidden", 1), ("head_hidden", 1), ("tape_seed", 0),
                              ("variants", 1), ("checkpoint_every", 1), ("adapt_steps", 1),
                              ("adapter_hidden", 1), ("export_bootstrap_replicates", 10)):
            put(name, fb._integer(getattr(self, name), f"config.{name}", minimum=minimum))
        for name in ("lr", "clip_norm", "lam_cf", "lam_marginal", "adapt_lr", "export_ridge_alpha",
                     "export_tau"):
            put(name, fb._real(getattr(self, name), f"config.{name}", positive=True))
        block = fb._integer(self.block_dim, "config.block_dim", minimum=1)
        require(block in (8, 16), "config.block_dim must be 8 or 16 (the declared block width)")
        put("block_dim", block)
        for name in ("cancel_max", "null_max"):
            value = fb._integer(getattr(self, name), f"config.{name}", minimum=0)
            require(value <= 4, f"config.{name} must be at most 4")
            put(name, value)
        require(fb._integer(self.eval_variant, "config.eval_variant", minimum=0) < self.variants,
                "config.eval_variant must be one of the raw variants")
        arms = tuple(self.arms)
        require(arms and all(arm in ARMS for arm in arms) and len(set(arms)) == len(arms),
                f"config.arms must be distinct members of {ARMS}")
        require("F" in arms, "config.arms must include F: it is the paired reference")
        put("arms", tuple(arm for arm in ARMS if arm in arms))
        put("seeds", _unique_ints(self.seeds, "seeds", minimum=0))
        put("export_acquisition_seeds", _unique_ints(self.export_acquisition_seeds,
                                                    "export_acquisition_seeds", minimum=0))
        put("export_label_budgets", tuple(sorted(_unique_ints(self.export_label_budgets,
                                                               "export_label_budgets", minimum=1,
                                                               maximum=168))))
        budgets = tuple(sorted(_unique_ints(self.transfer_budgets, "transfer_budgets", minimum=0,
                                            maximum=168)))
        require(budgets[0] == 0, "config.transfer_budgets must include 0: N0 is the unchanged frozen "
                                 "source, reported without a fabricated fit")
        put("transfer_budgets", budgets)

    @classmethod
    def from_document(cls, document):
        fm._keys(document, (), _CONFIG_FIELDS, label="causal config")
        values = dict(document)
        for name in ("seeds", "arms", "transfer_budgets", "export_label_budgets",
                     "export_acquisition_seeds"):
            if name in values:
                require(isinstance(values[name], (list, tuple)), f"config.{name} must be a list")
                values[name] = tuple(values[name])
        return cls(**values)

    @property
    def event_width(self):
        return fe.max_events(self.cancel_max, self.null_max)

    def model_spec(self):
        return ModelSpec(self.event_hidden, self.block_dim, self.program_embedding,
                         self.program_hidden, self.head_hidden)

    def document(self):
        return {name: (list(getattr(self, name)) if isinstance(getattr(self, name), tuple)
                       else getattr(self, name)) for name in _CONFIG_FIELDS}

    def training_document(self):
        """What a training run's numbers depend on (not the arm list, seeds or analysis settings)."""
        return {name: getattr(self, name) for name in _TRAINING_FIELDS}

    def identity_document(self):
        out = self.document()
        out.pop("checkpoint_every")
        return out


def run_id(arm, seed):
    return f"{arm}_s{int(seed)}"


# ---------------------------------------------------------------------------
# the student: batches, network, losses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class EventBatch:
    """What the student may see of the evidence: fact index, bit, numeric reliability and a validity
    mask. No event identity, no context identity, no teacher quantity."""

    fact: torch.Tensor
    bit: torch.Tensor
    reliability: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, eq=False)
class ProgramBatch:
    """The typed program as prefix token ids (0 pads) and each program's true length."""

    tokens: torch.Tensor
    length: torch.Tensor


@dataclass(frozen=True, eq=False)
class StudentBatch:
    events: EventBatch
    program: ProgramBatch


def event_batch(fact, bit, reliability, mask):
    """An :class:`EventBatch` from numpy arrays; every tensor is a fresh copy (no aliasing)."""
    return EventBatch(
        fact=torch.tensor(np.asarray(fact), dtype=torch.int64),
        bit=torch.tensor(np.asarray(bit), dtype=torch.int64),
        reliability=torch.tensor(np.asarray(reliability), dtype=torch.float32),
        mask=torch.tensor(np.asarray(mask), dtype=torch.bool))


def program_batch(tokens, length):
    return ProgramBatch(tokens=torch.tensor(np.asarray(tokens), dtype=torch.int64),
                        length=torch.tensor(np.asarray(length), dtype=torch.int64))


@lru_cache(maxsize=None)
def function_arrays():
    """Read-only arrays of every registry function (the student sees only ``tokens`` and ``length``;
    ``table`` and ``orbit_rank`` are teacher-side grouping data)."""
    registry = fe.function_registry()
    count, width = len(registry.functions), registry.program_width
    tokens = np.zeros((count, width), dtype=np.int8)
    length = np.zeros(count, dtype=np.int8)
    table = np.zeros(count, dtype=np.int32)
    orbits = sorted({item.orbit for item in registry.functions})
    orbit_rank = np.zeros(count, dtype=np.int16)
    for item in registry.functions:
        tokens[item.index, :len(item.tokens)] = item.tokens
        length[item.index] = len(item.tokens)
        table[item.index] = item.table
        orbit_rank[item.index] = orbits.index(item.orbit)
    source = registry.role(fe.ROLE_SOURCE)
    require([item.index for item in source] == list(range(N_SOURCE_FUNCTIONS)),
            "the source functions must be the registry's first 48")
    return {name: _readonly(array) for name, array in
            (("tokens", tokens), ("length", length), ("table", table), ("orbit_rank", orbit_rank))}


@dataclass(frozen=True)
class ModelSpec:
    event_hidden: int
    block_dim: int
    program_embedding: int
    program_hidden: int
    head_hidden: int

    def document(self):
        return dataclasses.asdict(self)


class CausalNet(nn.Module):
    """Event encoder (event MLP, masked SUM, learned mixing into four declared blocks that ALL see ALL
    evidence), program encoder (embedding + GRU) and prediction head. No analytic posterior, truth table
    or execution trace is available to ``forward``: it takes a :class:`StudentBatch` and nothing else.

    ``marginal_heads`` exist in every arm's network (one common initialisation) but only arm M's loss
    reaches them; ordinary prediction never touches them."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        hidden, block = spec.event_hidden, spec.block_dim
        self.event_mlp = nn.Sequential(nn.Linear(EVENT_FEATURES, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_BLOCKS * block))
        self.program_embedding = nn.Embedding(fe.N_TOKENS, spec.program_embedding,
                                              padding_idx=fe.TOKEN_PAD)
        self.program_gru = nn.GRU(spec.program_embedding, spec.program_hidden, batch_first=True)
        self.head_in = nn.Linear(N_BLOCKS * block + spec.program_hidden, spec.head_hidden)
        self.head_mid = nn.Linear(spec.head_hidden, spec.head_hidden)
        self.head_out = nn.Linear(spec.head_hidden, 1)
        self.marginal_heads = nn.ModuleList([nn.Linear(block, 1) for _ in range(N_BLOCKS)])

    def encode(self, events):
        """``[B, 4, block]`` hidden blocks: masked SUM over event embeddings, then learned mixing."""
        require(isinstance(events, EventBatch), "encode takes an EventBatch")
        fact = F.one_hot(events.fact.clamp(0, fe.N_FACTS - 1), fe.N_FACTS).to(torch.float32)
        bit = F.one_hot(events.bit.clamp(0, 1), 2).to(torch.float32)
        features = torch.cat([fact, bit, events.reliability.to(torch.float32).unsqueeze(-1)], dim=-1)
        hidden = self.event_mlp(features) * events.mask.unsqueeze(-1).to(torch.float32)
        pooled = hidden.sum(dim=1)
        return self.mix(pooled).reshape(pooled.shape[0], N_BLOCKS, self.spec.block_dim)

    def encode_program(self, program):
        """``[B, program_hidden]``: the GRU state at each program's last real token."""
        require(isinstance(program, ProgramBatch), "encode_program takes a ProgramBatch")
        output, _ = self.program_gru(self.program_embedding(program.tokens))
        index = (program.length - 1).clamp(min=0)
        return output[torch.arange(output.shape[0]), index]

    def fuse(self, blocks, program_state):
        """The query-conditioned penultimate activation (before the Bernoulli projection)."""
        joint = torch.cat([blocks.reshape(blocks.shape[0], -1), program_state], dim=-1)
        return F.gelu(self.head_mid(F.gelu(self.head_in(joint))))

    def read_out(self, penultimate):
        return self.head_out(penultimate).squeeze(-1)

    def predict(self, blocks, program_state):
        penultimate = self.fuse(blocks, program_state)
        return self.read_out(penultimate), penultimate

    def forward(self, batch):
        require(isinstance(batch, StudentBatch), "forward takes a StudentBatch")
        logit, _ = self.predict(self.encode(batch.events), self.encode_program(batch.program))
        return logit

    def marginal_logits(self, blocks):
        return torch.stack([head(blocks[:, k]).squeeze(-1)
                            for k, head in enumerate(self.marginal_heads)], dim=1)

    @staticmethod
    def patch_blocks(base_blocks, donor_blocks, site):
        """The base state with the DONOR's block at ``site`` substituted. Neither input is detached: the
        donor block stays differentiable and the replaced base block is bypassed."""
        site = site.to(torch.int64)
        choose = F.one_hot(site, N_BLOCKS).to(torch.bool).unsqueeze(-1)
        return torch.where(choose, donor_blocks, base_blocks)


def make_net(spec, seed):
    """A freshly initialised network from a private RNG fork (the global RNG and CUDA are untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return CausalNet(spec)


def init_seed(seed):
    return fe.stable_seed("causal-init", seed) % (2 ** 31)


def initial_state(spec, seed):
    net = make_net(spec, init_seed(seed))
    return {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}


def net_from_state(spec, state):
    """A network holding a CLONE of ``state`` after the state's keys, shapes, dtypes and finiteness were
    validated against a fresh network (validated before cloning)."""
    net = CausalNet(spec)
    fst._validate_net_state(state, dict(net.state_dict()), "initial")
    net.load_state_dict(fst._clone(state))
    return net


# ---------------------------------------------------------------------------
# raw-event arrays and the frozen event bank
# ---------------------------------------------------------------------------

EVENT_FIELDS = ("fact", "bit", "reliability", "mask")


def events_to_arrays(rows, width):
    """Padded arrays ``{fact, bit, reliability, mask}`` of shape ``[n, width]`` from raw-event tuples."""
    n = len(rows)
    fact, bit = np.zeros((n, width), dtype=np.int8), np.zeros((n, width), dtype=np.int8)
    reliability = np.zeros((n, width), dtype=np.float32)
    mask = np.zeros((n, width), dtype=bool)
    for i, events in enumerate(rows):
        events = fe.canonical_events(events)
        require(len(events) <= width, f"row {i} has {len(events)} events, above the padded width {width}")
        for j, event in enumerate(events):
            fact[i, j], bit[i, j], reliability[i, j], mask[i, j] = (event.fact, event.bit,
                                                                    event.reliability, True)
    return {"fact": fact, "bit": bit, "reliability": reliability, "mask": mask}


def arrays_to_events(arrays, row, prefix):
    """Reconstruct raw events of one array row with fresh distinct ids (for teacher cross-checks)."""
    out = []
    for j in range(arrays["mask"].shape[1]):
        if arrays["mask"][row, j]:
            out.append(fe.Observation(f"{prefix}.{j}", int(arrays["fact"][row, j]),
                                      int(arrays["bit"][row, j]), float(arrays["reliability"][row, j])))
    return tuple(out)


def compose_hybrid(base, donor, site):
    """Vectorized raw-evidence hybrids: per row, the base events of every fact but ``site`` followed by
    the donor events of ``site``, packed to the left at the same padded width."""
    width = base["mask"].shape[1]
    site = np.asarray(site)
    keep_base = base["mask"] & (base["fact"] != site[:, None])
    keep_donor = donor["mask"] & (donor["fact"] == site[:, None])
    keep = np.concatenate([keep_base, keep_donor], axis=1)
    require(int(keep.sum(axis=1).max(initial=0)) <= width,
            "a hybrid needs more events than the padded width")
    order = np.argsort(np.logical_not(keep), axis=1, kind="stable")[:, :width]
    mask = np.take_along_axis(keep, order, axis=1)
    out = {"mask": mask}
    for name in ("fact", "bit", "reliability"):
        both = np.concatenate([base[name], donor[name]], axis=1)
        out[name] = np.where(mask, np.take_along_axis(both, order, axis=1), 0).astype(base[name].dtype)
    return out


@dataclass(frozen=True, eq=False)
class EventBank:
    """The padded raw-event arrays of every (source signature, variant): the ACTUAL events the student
    is shown. ``row(signature_index, variant)`` indexes it."""

    arrays: object
    variants: int
    width: int
    sha256: str

    def row(self, signature_index, variant):
        return np.asarray(signature_index, dtype=np.int64) * self.variants + variant

    def take(self, rows):
        return {name: self.arrays[name][rows] for name in EVENT_FIELDS}

    def live_sha256(self):
        return _digest_arrays([(name, self.arrays[name]) for name in EVENT_FIELDS])


def build_bank(config):
    """``(bank, events)``: the frozen padded bank and the matching raw-event tuples (teacher side)."""
    world = fe.q7_world()
    events = []
    for signature in world.source:
        for variant in range(config.variants):
            events.append(fe.raw_events(signature, variant, cancel_max=config.cancel_max,
                                        null_max=config.null_max))
    arrays = events_to_arrays(events, config.event_width)
    bank = EventBank(arrays=_owned(arrays), variants=config.variants, width=config.event_width,
                     sha256="")
    object.__setattr__(bank, "sha256", bank.live_sha256())
    return bank, tuple(events)


@lru_cache(maxsize=None)
def donor_candidates():
    """``[1176, 4, 672]`` source indices: for each source base and site, every source signature whose
    level at the site is one of the four legal replacement levels (4 levels x 168)."""
    world = fe.q7_world()
    table = np.zeros((len(world.source), fe.N_FACTS, 672), dtype=np.int16)
    for base, signature in enumerate(world.source):
        for site in range(fe.N_FACTS):
            members = [i for level in world.legal_levels(signature, site)
                       for i in world.source_by_level[(site, level)]]
            require(len(members) == 672, f"{signature} site {site} has {len(members)} donors, not 672")
            table[base, site] = members
    table.setflags(write=False)
    return table


# ---------------------------------------------------------------------------
# the corrupted-supervision control
# ---------------------------------------------------------------------------

def _classes_split(inverse, new_keys):
    """How many ORIGINAL value classes received two or more distinct new values. ``0`` means the new
    column is a function of the old value (a fixed relabelling, the identity included): exactly the
    learnable rule a corrupted control must not be."""
    order = np.lexsort((new_keys, inverse))
    classes, values = inverse[order], new_keys[order]
    differs = (classes[1:] == classes[:-1]) & (values[1:] != values[:-1])
    return int(np.unique(classes[1:][differs]).shape[0])


def _randomize_stratum(values, rng):
    """Seeded per-example randomization of ``values`` (histogram kept exactly).

    Returns ``(new_values, record)``. The new column is a uniformly random permutation of the old one,
    so each example's target is drawn from the stratum's own value distribution independently of what
    its inputs implied. It is NOT a fixed rule: a deterministic rotation of the value-sorted order, or a
    full derangement of a balanced two-value stratum, is a value relabelling (Boolean complementation in
    the two-value case) and is learnable. A draw that happens to be a relabelling (the identity
    included) is repaired by ONE seeded swap that splits a repeated value across two new values.

    A stratum supports this only when it has at least ``MIN_STRATUM`` events, at least two distinct values
    and at least one value that repeats: with a single value nothing can change, a tiny stratum cannot be
    randomized meaningfully, and when every value occurs once every rearrangement is a relabelling of
    distinct values. Such a stratum is reported ``unsupported`` (with its kind) and keeps its true values.
    The maximum number of changes (``n`` when no value passes one half, else ``2 (n - m)``) is a
    DESCRIPTIVE bound only: a random draw is not asked to reach it, because forcing it would force a
    deterministic relabelling on balanced strata. The actual changed count is measured."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.shape[0])
    keys = np.round(values, 12)
    classes, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    distinct = int(classes.shape[0])
    m = int(counts.max()) if n else 0
    supported = distinct >= 2 and n >= MIN_STRATUM and m >= 2
    record = {"n": n, "classes": distinct, "max_class_count": m,
              "dominant_fraction": (m / n) if n else 0.0,
              "descriptive_max_changed": ((n if 2 * m <= n else 2 * (n - m)) if supported else 0)}
    if not supported:
        kind, reason = (
            ("unsupported_constant", "a single value (or no event): nothing can change") if distinct < 2
            else ("unsupported_small", f"fewer than {MIN_STRATUM} events: too few to randomize "
                                       "meaningfully") if n < MIN_STRATUM
            else ("unsupported_singleton_classes",
                  "every value occurs once: any rearrangement is a fixed relabelling of distinct "
                  "values, a learnable rule rather than per-example corruption"))
        record.update(status="unsupported", kind=kind, reason=reason, changed=0, changed_fraction=0.0,
                      classes_split=0, repaired_from_relabelling=False)
        return values.copy(), record
    out = values[rng.permutation(n)]
    split, repaired = _classes_split(inverse, np.round(out, 12)), False
    if split == 0:
        repeated = np.nonzero(counts[inverse] >= 2)[0]
        first = int(repeated[rng.integers(0, repeated.shape[0])])
        partners = np.nonzero(np.round(out, 12) != np.round(out[first], 12))[0]
        second = int(partners[rng.integers(0, partners.shape[0])])
        out[first], out[second] = out[second], out[first]
        split, repaired = _classes_split(inverse, np.round(out, 12)), True
    new_keys = np.round(out, 12)
    require(sorted(new_keys.tolist()) == sorted(keys.tolist()),
            "the randomized stratum does not preserve the target histogram")
    require(split >= 1, "internal error: a supported stratum was left as a fixed value relabelling")
    changed = int((new_keys != keys).sum())
    record.update(status="supported", reason=None, changed=changed, changed_fraction=changed / n,
                  classes_split=split, repaired_from_relabelling=repaired)
    return out, record


def corrupt_targets(p_cf, strata, orbit_of_stratum, rng):
    """Per-example corrupted counterfactual targets: values randomly permuted within each stratum.

    Returns ``(corrupted, report)``. The report MEASURES the change (count and fraction of examples whose
    value changed, mean absolute difference, mean KL, share of changes of at least 0.1), the supported
    and unsupported strata and why, how many original value classes were split across several new
    values in each stratum (``classes_split >= 1`` is what separates per-example randomization from a
    fixed relabelling) and the actual changed rate; the maximum possible change is descriptive only.
    Row-id changes with the same value are not counted as corruption."""
    p_cf = np.asarray(p_cf, dtype=np.float64)
    out = p_cf.copy()
    strata = np.asarray(strata)
    records, unsupported, repaired = [], [], 0
    for stratum in np.unique(strata):
        index = np.nonzero(strata == stratum)[0]
        new, record = _randomize_stratum(p_cf[index], rng)
        out[index] = new
        record["stratum"] = int(stratum)
        record["orbit_rank"] = int(orbit_of_stratum[int(stratum)])
        record["site"] = int(stratum) % fe.N_FACTS
        if record["status"] == "unsupported":
            unsupported.append(record)
        else:
            repaired += int(record["repaired_from_relabelling"])
        records.append(record)
    changed = np.round(out, 12) != np.round(p_cf, 12)
    difference = np.abs(out - p_cf)
    require(sorted(np.round(out, 12).tolist()) == sorted(np.round(p_cf, 12).tolist()),
            "the corrupted targets do not preserve the overall target histogram")
    total = int(p_cf.shape[0])
    in_unsupported = int(sum(item["n"] for item in unsupported))
    supported = total - in_unsupported
    return out, {
        "events": total, "strata": len(records), "supported_strata": len(records) - len(unsupported),
        "unsupported_strata": unsupported, "events_in_supported_strata": supported,
        "events_in_unsupported_strata": in_unsupported,
        "changed_values": int(changed.sum()),
        "changed_fraction": float(changed.mean()) if total else 0.0,
        "unchanged_fraction": float(1.0 - changed.mean()) if total else 0.0,
        "changed_fraction_of_supported_events": (float(changed.sum()) / supported) if supported else 0.0,
        "mean_abs_difference": float(difference.mean()) if total else 0.0,
        "mean_kl_true_to_corrupted": float(bernoulli_kl(p_cf, out).mean()) if total else 0.0,
        "fraction_changed_by_at_least_0.1": float((difference >= CHANGED_THRESHOLD).mean())
        if total else 0.0,
        "descriptive_max_changed": int(sum(item["descriptive_max_changed"] for item in records)),
        "classes_split": int(sum(item["classes_split"] for item in records)),
        "strata_repaired_from_a_fixed_relabelling": int(repaired),
        "histogram_preserved": True, "stratum_records": records,
        "definition": ("strata are (function S4 orbit, site); within a stratum the targets are a seeded "
                       "uniformly random permutation of the stratum's own values, with the query and "
                       "site inputs untouched, so the histogram is exact and each example's target is "
                       "randomized rather than a function of its true value. A fixed value relabelling, "
                       "a Boolean complement or a deterministic rotation is NOT used: it is a learnable "
                       "rule. The maximum possible change is a descriptive bound, never a target: the "
                       "actual changed and unchanged rates are measured. A stratum with a single value, "
                       f"fewer than {MIN_STRATUM} events, or in which every value occurs once cannot "
                       "be randomized meaningfully; it is reported unsupported and keeps its true "
                       "values")}


# ---------------------------------------------------------------------------
# the frozen source tape
# ---------------------------------------------------------------------------

TAPE_ARRAYS = ("base_sig", "base_var", "donor_sig", "donor_var", "site", "function", "hybrid_sig",
               "p_base", "p_cf", "p_cf_corrupt", "marg_base", "useful")
UPDATE_KEYS = ("base_sig", "base_var", "donor_sig", "donor_var", "site", "function", "hybrid_sig",
               "p_base", "p_cf", "p_cf_corrupt", "marg_base", "program_tokens", "program_length")
_STREAMS = ("base", "donor", "hybrid")


def update_arrays(bank, functions, tape_arrays, lo, hi):
    """Materialize EXACTLY what update ``[lo, hi)`` consumes: the index and label slices, the padded raw
    events of the base, the donor and the raw-evidence hybrid, and the program token rows."""
    base_rows = bank.row(tape_arrays["base_sig"][lo:hi], tape_arrays["base_var"][lo:hi])
    donor_rows = bank.row(tape_arrays["donor_sig"][lo:hi], tape_arrays["donor_var"][lo:hi])
    site = tape_arrays["site"][lo:hi]
    base, donor = bank.take(base_rows), bank.take(donor_rows)
    out = {name: tape_arrays[name][lo:hi] for name in TAPE_ARRAYS if name != "useful"}
    for stream, arrays in (("base", base), ("donor", donor),
                           ("hybrid", compose_hybrid(base, donor, site))):
        for name in EVENT_FIELDS:
            out[f"{stream}_{name}"] = arrays[name]
    function = tape_arrays["function"][lo:hi].astype(np.int64)
    out["program_tokens"] = functions["tokens"][function]
    out["program_length"] = functions["length"][function]
    return out


def update_sha256(arrays):
    named = [(name, arrays[name]) for name in UPDATE_KEYS]
    named += [(f"{stream}_{name}", arrays[f"{stream}_{name}"]) for stream in _STREAMS
              for name in EVENT_FIELDS]
    return _digest_arrays(named)


def _tape_digest(arrays, update_digests, report, seed, config_sha256, bank_sha256):
    return fb.digest({
        "schema": TAPE_SCHEMA, "seed": int(seed), "config_sha256": config_sha256,
        "bank_sha256": bank_sha256,
        "arrays": {name: _array_sha(arrays[name]) for name in TAPE_ARRAYS},
        "update_sha256": list(update_digests), "report": report})


@dataclass(frozen=True, eq=False)
class SourceTape:
    """One frozen SOURCE-only tape: every base, donor, site, query and exact teacher target, plus the
    corrupted Ic targets and the per-update digest of everything an update consumes. Arrays are owned
    read-only copies and the digest recomputes from them (:meth:`verify`), so an in-place edit, even
    after a write flag is re-enabled, is detected."""

    seed: int
    config_sha256: str
    bank_sha256: str
    arrays: object
    update_sha256: tuple
    report: dict
    sha256: str

    @property
    def updates(self):
        return len(self.update_sha256)

    def live_sha256(self):
        return _tape_digest(self.arrays, self.update_sha256, self.report, self.seed,
                            self.config_sha256, self.bank_sha256)

    def verify(self):
        require(self.live_sha256() == self.sha256,
                f"the tape of seed {self.seed} no longer matches its recorded digest: a base, donor, "
                "site, program, label or event index was modified after it was frozen")
        return True

    def copy_owned(self):
        """An independent, verified copy (separate buffers and a deep-copied report)."""
        self.verify()
        owned = SourceTape(seed=self.seed, config_sha256=self.config_sha256,
                           bank_sha256=self.bank_sha256, arrays=_owned(self.arrays),
                           update_sha256=tuple(self.update_sha256),
                           report=copy.deepcopy(self.report), sha256=self.sha256)
        owned.verify()
        return owned


def make_tape(seed, config, arrays, update_digests, report, bank_sha256):
    arrays = _owned(arrays)
    sha = _tape_digest(arrays, update_digests, report, seed, fb.digest(config.training_document()),
                       bank_sha256)
    return SourceTape(seed=int(seed), config_sha256=fb.digest(config.training_document()),
                      bank_sha256=bank_sha256, arrays=arrays, update_sha256=tuple(update_digests),
                      report=copy.deepcopy(report), sha256=sha)


def build_tape(config, seed, bank, bank_events):
    """Build the frozen SOURCE-only tape of one seed. Only source signatures are ever labelled."""
    world, registry = fe.q7_world(), fe.function_registry()
    functions = function_arrays()
    rows = config.steps * config.batch
    rng = np.random.default_rng([config.tape_seed, int(seed), 1001])
    base_sig = rng.integers(0, len(world.source), rows)
    base_var = rng.integers(0, config.variants, rows)
    function = rng.integers(0, N_SOURCE_FUNCTIONS, rows)
    site = rng.integers(0, fe.N_FACTS, rows)
    pick = rng.integers(0, 672, rows)
    donor_var = rng.integers(0, config.variants, rows)
    donor_sig = donor_candidates()[base_sig, site, pick].astype(np.int64)
    tables = [registry.functions[i].table for i in range(N_SOURCE_FUNCTIONS)]
    # every counter below is incremented where the call is made, so it is a count of calls and never a
    # formula over rows: the base posterior and its marginals are computed once per DISTINCT base bank
    # row (a cache miss) while the hybrid posterior and both probabilities are computed for every row
    cost = {"base_posterior_computations": 0, "marginal_computations": 0, "marginal_values": 0,
            "hybrid_event_materializations": 0, "hybrid_posterior_computations": 0,
            "factual_probability_evaluations": 0, "counterfactual_probability_evaluations": 0,
            "probability_evaluations": 0, "tape_rows_materialized": 0,
            "marginal_target_values_materialized": 0}
    base_cache = {}

    def base_record(row):
        if row not in base_cache:
            posterior = fe.world_posterior(bank_events[row])
            cost["base_posterior_computations"] += 1
            values = [float(m) for m in fe.marginals(posterior)]
            cost["marginal_computations"] += 1
            cost["marginal_values"] += fe.N_FACTS
            base_cache[row] = (posterior, values)
        return base_cache[row]
    p_base = np.zeros(rows, dtype=np.float64)
    p_cf = np.zeros(rows, dtype=np.float64)
    marg = np.zeros((rows, fe.N_FACTS), dtype=np.float64)
    hybrid_sig = np.zeros(rows, dtype=np.int64)
    for j in range(rows):
        brow = int(base_sig[j]) * config.variants + int(base_var[j])
        drow = int(donor_sig[j]) * config.variants + int(donor_var[j])
        posterior, marginals_ = base_record(brow)
        table = tables[int(function[j])]
        hybrid = fe.hybrid_events(bank_events[brow], bank_events[drow], int(site[j]))
        cost["hybrid_event_materializations"] += 1
        hybrid_posterior = fe.world_posterior(hybrid)
        cost["hybrid_posterior_computations"] += 1
        p_base[j] = float(fe.table_probability(table, posterior))
        cost["factual_probability_evaluations"] += 1
        p_cf[j] = float(fe.table_probability(table, hybrid_posterior))
        cost["counterfactual_probability_evaluations"] += 1
        marg[j] = marginals_
        cost["tape_rows_materialized"] += 1
        cost["marginal_target_values_materialized"] += fe.N_FACTS
        signature = fe.hybrid_signature(world.source[int(base_sig[j])],
                                        world.source[int(donor_sig[j])], int(site[j]))
        require(signature in world.source_index,
                f"row {j}: the hybrid {signature} left the source support")
        require(fe.signature_of_events(hybrid) == signature,
                f"row {j}: the raw-evidence hybrid does not belong to its labelled signature")
        hybrid_sig[j] = world.source_index[signature]
    cost["probability_evaluations"] = (cost["factual_probability_evaluations"]
                                       + cost["counterfactual_probability_evaluations"])
    strata = functions["orbit_rank"][function].astype(np.int64) * fe.N_FACTS + site
    orbit_of_stratum = {int(s): int(s) // fe.N_FACTS for s in np.unique(strata)}
    corrupted, corruption = corrupt_targets(
        p_cf, strata, orbit_of_stratum, np.random.default_rng([config.tape_seed, int(seed), 777]))
    arrays = {"base_sig": base_sig.astype(np.int16), "base_var": base_var.astype(np.int8),
              "donor_sig": donor_sig.astype(np.int16), "donor_var": donor_var.astype(np.int8),
              "site": site.astype(np.int8), "function": function.astype(np.int16),
              "hybrid_sig": hybrid_sig.astype(np.int16), "p_base": p_base, "p_cf": p_cf,
              "p_cf_corrupt": corrupted, "marg_base": marg,
              "useful": np.zeros((config.steps, 3), dtype=np.int32)}
    digests, useful = [], np.zeros((config.steps, 3), dtype=np.int32)
    for u in range(config.steps):
        update = update_arrays(bank, functions, arrays, u * config.batch, (u + 1) * config.batch)
        digests.append(update_sha256(update))
        useful[u] = [int(update[f"{stream}_mask"].sum()) for stream in _STREAMS]
    arrays["useful"] = useful
    # the digests above were taken before ``useful`` was filled; ``useful`` is derived from the very
    # arrays they cover and is bound separately by the tape digest.
    report = {"seed": int(seed), "rows": rows, "updates": config.steps, "batch": config.batch,
              "teacher_cost": cost, "corruption": corruption,
              "support": {"every_base_donor_and_hybrid_in_source": True,
                          "source_signatures": len(world.source)}}
    return make_tape(seed, config, arrays, digests, report, bank.sha256)


def verify_tape(tape, bank, bank_events, config, *, deep=False):
    """Check a tape. Shallow: the digest. Deep: recompute EVERY update digest from the arrays and the
    bank, and recompute every teacher label from the raw events the arrays describe (exact equality)."""
    require(isinstance(tape, SourceTape), "verify_tape needs a SourceTape")
    tape.verify()
    require(tape.bank_sha256 == bank.sha256, "the tape was built over a different event bank")
    if not deep:
        return True
    functions = function_arrays()
    world, registry = fe.q7_world(), fe.function_registry()
    for u in range(tape.updates):
        lo, hi = u * config.batch, (u + 1) * config.batch
        update = update_arrays(bank, functions, tape.arrays, lo, hi)
        require(update_sha256(update) == tape.update_sha256[u], f"update {u} digest mismatch")
        for r in range(hi - lo):
            j = lo + r
            table = registry.functions[int(tape.arrays["function"][j])].table
            base_events = arrays_to_events({n: update[f"base_{n}"] for n in EVENT_FIELDS}, r, f"b{j}")
            hybrid_events_ = arrays_to_events({n: update[f"hybrid_{n}"] for n in EVENT_FIELDS}, r,
                                              f"h{j}")
            require(float(fe.table_probability(table, fe.world_posterior(base_events)))
                    == float(tape.arrays["p_base"][j]), f"row {j}: p_base is not the teacher's")
            require(float(fe.table_probability(table, fe.world_posterior(hybrid_events_)))
                    == float(tape.arrays["p_cf"][j]), f"row {j}: p_cf is not the teacher's")
            require(world.source[int(tape.arrays["hybrid_sig"][j])]
                    == fe.signature_of_events(hybrid_events_),
                    f"row {j}: the hybrid events are not the recorded hybrid signature")
    return True


# ---------------------------------------------------------------------------
# tape / initial-state files
# ---------------------------------------------------------------------------

def _tape_npz_arrays(tape):
    return {name: np.asarray(tape.arrays[name]) for name in TAPE_ARRAYS}


def _npz_differs(path, expected):
    """``None`` when the existing file holds EXACTLY the expected arrays (the same member names and, for
    each, the same dtype, shape and bytes), otherwise the reason it does not. Reads the file only: a
    foreign, truncated, pickled or merely different archive is described, never touched."""
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        return f"cannot be read as a plain numeric NPZ ({type(error).__name__}: {error})"
    try:
        members = getattr(archive, "files", None)
        if members is None:
            return "is not an NPZ archive"
        if sorted(members) != sorted(expected):
            return f"holds the members {sorted(members)}, not {sorted(expected)}"
        for name, wanted in expected.items():
            try:
                found = archive[name]
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
                return f"has an unreadable member {name!r} ({type(error).__name__}: {error})"
            if _array_sha(found) != _array_sha(wanted):
                return f"member {name!r} differs from this tape in dtype, shape or content"
        return None
    finally:
        close = getattr(archive, "close", None)
        if close is not None:
            close()


def preflight_tape(tape, directory, *, protected=()):
    """Everything :func:`save_tape` would refuse, checked WITHOUT writing or creating anything: a
    protected alias, an existing header that is not this tape's, and an existing arrays file that is not
    EXACTLY this tape's arrays (a leftover archive of another run must never receive a header that
    claims it). Returns ``(npz_path, header_path, header)``."""
    directory = Path(directory)
    tape.verify()
    npz, header_path = directory / f"seed_{tape.seed}.npz", directory / f"seed_{tape.seed}.json"
    fst.refuse_protected(protected, {"tape arrays": npz, "tape header": header_path})
    header = {"schema": TAPE_SCHEMA, "seed": tape.seed, "sha256": tape.sha256,
              "config_sha256": tape.config_sha256, "bank_sha256": tape.bank_sha256,
              "update_sha256": list(tape.update_sha256), "report": tape.report,
              "arrays": {name: {"dtype": tape.arrays[name].dtype.str,
                                "shape": list(tape.arrays[name].shape)} for name in TAPE_ARRAYS}}
    if os.path.lexists(header_path):
        require(fb.read_json(header_path) == json.loads(fb.canonical_json(header)),
                f"{header_path} exists and is not this tape; it is never replaced. {FRESH_OUTPUT}")
    if os.path.lexists(npz):
        reason = _npz_differs(npz, _tape_npz_arrays(tape))
        require(reason is None,
                f"{npz} exists and {reason}; it is never replaced and no header is published for it. "
                f"{FRESH_OUTPUT}")
    return npz, header_path, header


def save_tape(tape, directory, *, protected=()):
    """Publish ``seed_<s>.npz`` and ``seed_<s>.json`` through owned staging.

    An existing arrays file is ADOPTED only after its members, dtypes, shapes and bytes were checked
    against this tape (it is then left untouched, so its actual file hash is preserved); a different or
    malformed one refuses before ANY header is published. An identical existing header is adopted too.
    Returns the header plus the actual file hash of the published arrays."""
    directory = Path(directory)
    npz, header_path, header = preflight_tape(tape, directory, protected=protected)
    directory.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(npz):
        fst._publish(npz, lambda handle: np.savez(handle, **_tape_npz_arrays(tape)), text=False)
        reason = _npz_differs(npz, _tape_npz_arrays(tape))
        require(reason is None, f"the published {npz} does not hold this tape: it {reason}")
    if not os.path.lexists(header_path):
        fst.write_json_owned(header_path, header, protected=protected)
    return {**header, "npz_file_sha256": file_sha256(npz)}


def load_tape(directory, seed, config, *, bank_sha256):
    """Load a tape file with strict checks: schema, bindings, member names, dtypes, shapes, digest."""
    directory = Path(directory)
    header = fb.read_json(directory / f"seed_{seed}.json")
    require(header.get("schema") == TAPE_SCHEMA and header.get("seed") == seed,
            f"{directory}/seed_{seed}.json is not a tape header of seed {seed}")
    require(header["config_sha256"] == fb.digest(config.training_document()),
            "the tape was built with a different training configuration")
    require(header["bank_sha256"] == bank_sha256, "the tape was built over a different event bank")
    try:
        archive = np.load(directory / f"seed_{seed}.npz", allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"{directory}/seed_{seed}.npz cannot be read as a plain numeric NPZ "
                         f"({type(error).__name__}: {error})") from error
    try:
        require(sorted(getattr(archive, "files", ())) == sorted(TAPE_ARRAYS),
                "the tape archive holds the wrong arrays")
        arrays = {}
        for name in TAPE_ARRAYS:
            try:
                array = archive[name]
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
                raise ValueError(f"tape array {name!r} cannot be read ({error})") from error
            spec = header["arrays"][name]
            require(array.dtype.str == spec["dtype"] and list(array.shape) == spec["shape"],
                    f"tape array {name!r} has the wrong dtype or shape")
            arrays[name] = array
    finally:
        close = getattr(archive, "close", None)
        if close is not None:
            close()
    tape = SourceTape(seed=int(seed), config_sha256=header["config_sha256"],
                      bank_sha256=header["bank_sha256"], arrays=_owned(arrays),
                      update_sha256=tuple(header["update_sha256"]), report=header["report"],
                      sha256=header["sha256"])
    tape.verify()
    return tape


def preflight_init(state, seed, directory, spec, *, protected=()):
    """Everything :func:`save_init` would refuse, checked WITHOUT writing: a protected alias, existing
    weights that are not this state, and an existing header that is not this state's header (a header
    without its weights file included). Returns ``(weights_path, header_path, digest)``."""
    directory = Path(directory)
    weights, header_path = directory / f"seed_{seed}.pt", directory / f"seed_{seed}.json"
    fst.refuse_protected(protected, {"initial weights": weights, "initial header": header_path})
    digest = fst.tensor_state_digest(state)
    if os.path.lexists(weights):
        require(fst._load_payload_digest(weights) == digest,
                f"{weights} exists and is not this initial state; it is never replaced. {FRESH_OUTPUT}")
    if os.path.lexists(header_path):
        existing = fb.read_json(header_path)
        require(os.path.lexists(weights)
                and existing == {"schema": INIT_SCHEMA, "seed": int(seed), "state_sha256": digest,
                                 "spec": spec.document(),
                                 "weights_file_sha256": file_sha256(weights)},
                f"{header_path} exists and is not this initial state's header; it is never replaced. "
                f"{FRESH_OUTPUT}")
    return weights, header_path, digest


def save_init(state, seed, directory, spec, *, protected=()):
    """Publish the shared initial state of one seed and a header that binds its tensor digest. Existing
    identical weights and header are adopted untouched; different ones refuse before any write."""
    directory = Path(directory)
    weights, header_path, digest = preflight_init(state, seed, directory, spec, protected=protected)
    directory.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(weights):
        fst._publish(weights, lambda handle: torch.save(state, handle), text=False)
    header = {"schema": INIT_SCHEMA, "seed": int(seed), "state_sha256": digest,
              "spec": spec.document(), "weights_file_sha256": file_sha256(weights)}
    if not os.path.lexists(header_path):
        fst.write_json_owned(header_path, header, protected=protected)
    return header


def load_init(directory, seed, spec):
    directory = Path(directory)
    header = fb.read_json(directory / f"seed_{seed}.json")
    require(header.get("schema") == INIT_SCHEMA and header["spec"] == spec.document(),
            "the initial-state header is not this model's")
    weights = directory / f"seed_{seed}.pt"
    require(weights.is_file(), f"the initial weights file {weights} is missing")
    require(file_sha256(weights) == header["weights_file_sha256"],
            "the initial weights file does not match its recorded SHA-256")
    state = _load_plain(weights)
    require(fst.tensor_state_digest(state) == header["state_sha256"],
            "the initial tensors do not match their recorded digest")
    return state


# ---------------------------------------------------------------------------
# the training context and one arm's run
# ---------------------------------------------------------------------------

def _functions_sha256():
    functions = function_arrays()
    return _digest_arrays([(name, functions[name]) for name in sorted(functions)])


class TrainContext:
    """What training consumes: the frozen bank, the per-seed tapes, the shared initial states and the
    bindings that every checkpoint and frozen state carries. Built (:meth:`build`) or loaded once."""

    def __init__(self, config, bank, bank_events, tapes, init_states):
        require(isinstance(config, CausalConfig), "TrainContext needs a CausalConfig")
        self.config, self.spec = config, config.model_spec()
        self.bank, self.bank_events = bank, bank_events
        self.tapes = dict(tapes)
        self.init_states = {seed: fst._clone(state) for seed, state in init_states.items()}
        self.init_sha256 = {seed: fst.tensor_state_digest(state)
                            for seed, state in self.init_states.items()}
        self.functions_sha256 = _functions_sha256()
        self.world_sha256 = fe.world_sha256()
        self.code_sha256 = code_sha256()
        self.training_config_sha256 = fb.digest(config.training_document())
        for seed, tape in self.tapes.items():
            verify_tape(tape, bank, bank_events, config)
            require(seed in self.init_states, f"no initial state for seed {seed}")

    @classmethod
    def build(cls, config, *, seeds=None):
        bank, events = build_bank(config)
        seeds = tuple(config.seeds if seeds is None else seeds)
        tapes = {seed: build_tape(config, seed, bank, events) for seed in seeds}
        states = {seed: initial_state(config.model_spec(), seed) for seed in seeds}
        return cls(config, bank, events, tapes, states)

    def bindings(self, seed):
        return {"init_sha256": self.init_sha256[seed], "tape_sha256": self.tapes[seed].sha256,
                "training_config_sha256": self.training_config_sha256,
                "world_sha256": self.world_sha256, "bank_sha256": self.bank.sha256,
                "functions_sha256": self.functions_sha256, "code_sha256": self.code_sha256}

    def verify_live(self, seed):
        """Refuse when anything shared that a run is about to copy no longer matches its record."""
        require(self.bank.live_sha256() == self.bank.sha256,
                "the event bank no longer matches its recorded digest: a raw event was modified")
        require(_functions_sha256() == self.functions_sha256,
                "the program token table no longer matches its recorded digest")
        require(seed in self.tapes, f"no tape for seed {seed}")
        self.tapes[seed].verify()
        require(fst.tensor_state_digest(self.init_states[seed]) == self.init_sha256[seed],
                f"the initial state of seed {seed} no longer matches its recorded digest")
        return True


def _fresh_accounting():
    return {"updates": 0, "optimizer_steps": 0, "backward_calls": 0,
            "forward_calls": {"evidence_encoder": 0, "program_encoder": 0, "prediction_head": 0,
                              "marginal_heads": 0},
            "rows": {"evidence_encoder_rows": 0, "event_rows_padded": 0, "event_rows_useful": 0,
                     "prediction_rows": 0},
            "presentations": {"factual": 0, "counterfactual": 0, "marginal_targets": 0}}


def expected_accounting(arm, cursor, config, tape):
    """The call/row/presentation counters ``cursor`` updates imply, derived independently from the tape
    (the checkpoint validator compares the run's own incremental counters with this)."""
    batch, width = config.batch, config.event_width
    counterfactual = arm in CF_ARMS
    encodes = cursor * (2 if counterfactual else 1)
    useful = np.asarray(tape.arrays["useful"][:cursor]).sum(axis=0)
    encoded_useful = int(useful[0])
    if arm == "A":
        encoded_useful += int(useful[2])
    elif counterfactual:
        encoded_useful += int(useful[1])
    return {"updates": cursor, "optimizer_steps": cursor, "backward_calls": cursor,
            "forward_calls": {"evidence_encoder": encodes, "program_encoder": cursor,
                              "prediction_head": encodes, "marginal_heads": cursor if arm == "M" else 0},
            "rows": {"evidence_encoder_rows": encodes * batch, "event_rows_padded": encodes * batch * width,
                     "event_rows_useful": encoded_useful, "prediction_rows": encodes * batch},
            "presentations": {"factual": cursor * batch,
                              "counterfactual": cursor * batch if counterfactual else 0,
                              "marginal_targets": cursor * batch * N_BLOCKS if arm == "M" else 0}}


def _head_group(name):
    return "marginal" if name.startswith("marginal_heads") else "main"


def expected_update_counts(arm, cursor, net):
    """The Adam step count of every parameter after ``cursor`` updates. Adam creates no state for a
    parameter whose loss never reached it, so the four marginal heads move only in arm M; everywhere else
    they have NO optimizer entry (an unused auxiliary head legitimately has no state)."""
    return {name: (cursor if (_head_group(name) == "main" or arm == "M") else 0)
            for name, _ in net.named_parameters()}


def _stack(update, stream):
    return event_batch(update[f"{stream}_fact"], update[f"{stream}_bit"],
                       update[f"{stream}_reliability"], update[f"{stream}_mask"])


def raw_multisets(arrays, levels):
    """``[n, 8 * len(levels)]`` counts of every (fact, bit, reliability) category of each padded raw record.

    Event identities do not exist in the arrays and the ORDER of events is discarded, but fact, bit and
    reliability multiplicity is kept (a cancelling pair is two events, not none). Two rows are therefore
    equal exactly when they are the same raw student input up to renaming and reordering of events."""
    mask = np.asarray(arrays["mask"], dtype=bool)
    count, categories = int(mask.shape[0]), 2 * fe.N_FACTS * int(levels.shape[0])
    if count == 0:
        return np.zeros((0, categories), dtype=np.int64)
    reliability = np.asarray(arrays["reliability"])
    position = np.clip(np.searchsorted(levels, reliability), 0, int(levels.shape[0]) - 1)
    require(bool((levels[position] == reliability)[mask].all()),
            "a raw record holds a reliability that is not one of the bank's reliability levels")
    category = ((np.asarray(arrays["fact"], dtype=np.int64) * 2 + np.asarray(arrays["bit"], dtype=np.int64))
                * int(levels.shape[0]) + position)
    row = np.broadcast_to(np.arange(count, dtype=np.int64)[:, None], mask.shape)
    return np.bincount(row[mask] * categories + category[mask],
                       minlength=count * categories).reshape(count, categories)


def _class_ids(*matrices):
    """Integer class ids of the rows of several count matrices in ONE shared id space: equal rows get one
    id even when they come from different matrices."""
    sizes = [int(matrix.shape[0]) for matrix in matrices]
    if sum(sizes) == 0:
        return [np.zeros(0, dtype=np.int64) for _ in matrices]
    _, inverse = np.unique(np.concatenate(matrices, axis=0), axis=0, return_inverse=True)
    inverse, edges = inverse.reshape(-1).astype(np.int64), np.cumsum([0] + sizes)
    return [inverse[edges[i]:edges[i + 1]] for i in range(len(sizes))]


def _distinct(*columns):
    """How many distinct rows the 1-D integer columns form together (0 for no rows)."""
    if not columns or int(np.asarray(columns[0]).shape[0]) == 0:
        return 0
    stacked = np.stack([np.asarray(column, dtype=np.int64) for column in columns], axis=1)
    return int(np.unique(stacked, axis=0).shape[0])


def _distinct_union(*columns):
    """How many distinct values occur across several 1-D integer columns taken together."""
    return int(np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in columns])).shape[0])


class ArmRun:
    """One arm trained for one seed, one update at a time.

    Every arm clones the SAME initial state (separate parameter objects) and consumes the same frozen
    tape, loss definition, optimizer recipe and update count. The run owns what it consumes: construction
    verifies the shared bank, program table, tape and initial state and copies them; EVERY update then
    re-materializes exactly the events, labels and program rows it is about to use and compares their
    digest with the one recorded at tape time BEFORE any parameter, optimizer, cursor or counter changes.
    """

    def __init__(self, ctx, arm, seed):
        require(arm in ARMS, f"unknown arm {arm!r}")
        ctx.verify_live(seed)
        self.source_ctx = ctx
        self.arm, self.seed = arm, int(seed)
        self.config, self.spec = ctx.config, ctx.spec
        self.bank = EventBank(arrays=_owned(ctx.bank.arrays), variants=ctx.bank.variants,
                              width=ctx.bank.width, sha256=ctx.bank.sha256)
        self.functions = {name: _readonly(array) for name, array in function_arrays().items()}
        self.tape = ctx.tapes[seed].copy_owned()
        require(self.bank.live_sha256() == self.bank.sha256
                and _digest_arrays([(n, self.functions[n]) for n in sorted(self.functions)])
                == ctx.functions_sha256, "the run-owned copies do not match the verified originals")
        self.net = net_from_state(self.spec, ctx.init_states[seed])
        self.optimizer = fst._make_adam(self.net.parameters(), self.config)
        self.total_steps = self.config.steps
        self.cursor = 0
        self.accounting = _fresh_accounting()
        self.log = []
        self.cpu_seconds = 0.0
        self.bindings = ctx.bindings(seed)

    @property
    def run_id(self):
        return run_id(self.arm, self.seed)

    def verify_inputs(self):
        """Boundary check (resume, checkpoint, freeze): the run's own copies, the shared originals."""
        self.tape.verify()
        require(self.bank.live_sha256() == self.bank.sha256,
                "the run's event bank no longer matches its recorded digest")
        require(_digest_arrays([(n, self.functions[n]) for n in sorted(self.functions)])
                == self.source_ctx.functions_sha256,
                "the run's program token table no longer matches its recorded digest")
        require(fst.tensor_state_digest(self.source_ctx.init_states[self.seed])
                == self.source_ctx.init_sha256[self.seed],
                f"the initial state of seed {self.seed} no longer matches its recorded digest")
        return True

    def consumed(self, update_index):
        """The exact arrays update ``update_index`` would consume (and their digest)."""
        lo = update_index * self.config.batch
        arrays = update_arrays(self.bank, self.functions, self.tape.arrays, lo,
                               lo + self.config.batch)
        return arrays, update_sha256(arrays)

    def state_payload(self):
        return {"format": CHECKPOINT_SCHEMA, "net": fst._clone(dict(self.net.state_dict())),
                "optimizer": fst._clone(self.optimizer.state_dict())}

    def state_sha256(self):
        return fst.tensor_state_digest(self.state_payload())

    def loss_terms(self, update):
        """The differentiable loss pieces of one update (no optimizer, no counters). Returns
        ``(loss, parts, calls)`` where ``calls`` counts the encodes and head calls performed."""
        cfg, arm, net = self.config, self.arm, self.net
        program = program_batch(update["program_tokens"], update["program_length"])
        p_base = torch.tensor(update["p_base"], dtype=torch.float32)
        blocks_base = net.encode(_stack(update, "base"))
        program_state = net.encode_program(program)
        logit_base, _ = net.predict(blocks_base, program_state)
        factual = soft_bce(logit_base, p_base).mean()
        loss, parts = factual, {"factual": float(factual)}
        calls = {"encodes": 1, "heads": 1, "marginal": 0, "useful": int(update["base_mask"].sum())}
        if arm == "M":
            marg = torch.tensor(update["marg_base"], dtype=torch.float32)
            marginal = soft_bce(net.marginal_logits(blocks_base), marg).mean()
            loss = loss + cfg.lam_marginal * marginal
            parts["marginal"] = float(marginal)
            calls["marginal"] = 1
        elif arm in CF_ARMS:
            if arm == "A":
                other = net.encode(_stack(update, "hybrid"))
                logit_cf, _ = net.predict(other, program_state)
                target_name, stream = "p_cf", "hybrid"
            else:
                donor = net.encode(_stack(update, "donor"))
                site = torch.tensor(update["site"], dtype=torch.int64)
                logit_cf, _ = net.predict(net.patch_blocks(blocks_base, donor, site), program_state)
                target_name, stream = ("p_cf" if arm == "I" else "p_cf_corrupt"), "donor"
            target = torch.tensor(update[target_name], dtype=torch.float32)
            counterfactual = soft_bce(logit_cf, target).mean()
            loss = loss + cfg.lam_cf * counterfactual
            parts["counterfactual"] = float(counterfactual)
            calls.update(encodes=2, heads=2, useful=calls["useful"] + int(update[f"{stream}_mask"].sum()))
        return loss, parts, calls

    def step(self):
        """One optimizer update. The consumed inputs are verified FIRST: a refused update leaves the
        parameters, optimizer, cursor, counters, log and timers exactly as they were."""
        require(self.cursor < self.total_steps, "this run has already taken every update")
        update, digest = self.consumed(self.cursor)
        require(digest == self.tape.update_sha256[self.cursor],
                f"update {self.cursor} would consume a base, donor, site, program, label or event "
                "that no longer matches what the tape recorded; refused before any update")
        started = time.process_time()
        loss, parts, calls = self.loss_terms(update)
        require(bool(torch.isfinite(loss)), f"the loss is not finite at update {self.cursor}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        reached = [p for p in self.net.parameters() if p.grad is not None]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(reached, self.config.clip_norm))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        acc, batch, width = self.accounting, self.config.batch, self.config.event_width
        acc["updates"] += 1
        acc["optimizer_steps"] += 1
        acc["backward_calls"] += 1
        acc["forward_calls"]["evidence_encoder"] += calls["encodes"]
        acc["forward_calls"]["program_encoder"] += 1
        acc["forward_calls"]["prediction_head"] += calls["heads"]
        acc["forward_calls"]["marginal_heads"] += calls["marginal"]
        acc["rows"]["evidence_encoder_rows"] += calls["encodes"] * batch
        acc["rows"]["event_rows_padded"] += calls["encodes"] * batch * width
        acc["rows"]["event_rows_useful"] += calls["useful"]
        acc["rows"]["prediction_rows"] += calls["heads"] * batch
        acc["presentations"]["factual"] += batch
        if self.arm in CF_ARMS:
            acc["presentations"]["counterfactual"] += batch
        if self.arm == "M":
            acc["presentations"]["marginal_targets"] += batch * N_BLOCKS
        self.log.append({"step": self.cursor, "loss": float(loss), **parts, "grad_norm": grad_norm,
                         "update_sha256": digest})
        self.cursor += 1
        self.cpu_seconds += time.process_time() - started

    def cost_report(self):
        """Every cost, in separate units that are never added together.

        The tape rows this run consumed are counted in four different ways. ``canonical_answers`` are the
        teacher's answers: factual ones are keyed by (base signature, program), counterfactual ones by
        (HYBRID signature, program), so a different raw variant or a donor that differs only at facts the
        hybrid does not take is not a new answer. ``raw_inputs`` are the student's actual inputs with
        event ids and event order ignored and fact/bit/reliability multiplicity kept.
        ``intervention_tuples`` are the (base, donor, site, program) questions, canonical and raw.
        ``presentations_and_reuse`` counts rows shown and what was shown beyond each distinct count. The
        raw contexts each arm actually ENCODED are reported separately (base and hybrid for A, base and
        donor for I and Ic)."""
        t, arm = self.tape.arrays, self.arm
        n = self.cursor * self.config.batch
        base_sig = t["base_sig"][:n].astype(np.int64)
        donor_sig = t["donor_sig"][:n].astype(np.int64)
        hybrid_sig = t["hybrid_sig"][:n].astype(np.int64)
        function = t["function"][:n].astype(np.int64)
        site = t["site"][:n].astype(np.int64)
        if n:
            base_events = self.bank.take(self.bank.row(base_sig, t["base_var"][:n]))
            donor_events = self.bank.take(self.bank.row(donor_sig, t["donor_var"][:n]))
            levels = np.unique(self.bank.arrays["reliability"][self.bank.arrays["mask"]])
            base_raw, donor_raw, hybrid_raw = _class_ids(
                raw_multisets(base_events, levels), raw_multisets(donor_events, levels),
                raw_multisets(compose_hybrid(base_events, donor_events, site), levels))
        else:
            base_raw = donor_raw = hybrid_raw = np.zeros(0, dtype=np.int64)
        counterfactual, presented = arm in CF_ARMS, self.accounting["presentations"]
        canonical = {"factual": _distinct(base_sig, function),
                     "counterfactual": _distinct(hybrid_sig, function) if counterfactual else 0,
                     "marginal": N_BLOCKS * _distinct(base_sig) if arm == "M" else 0}
        raw = {"factual": _distinct(base_raw, function),
               "counterfactual": (_distinct(hybrid_raw, function) if arm == "A" else
                                  _distinct(base_raw, donor_raw, site, function) if counterfactual
                                  else 0),
               "marginal": N_BLOCKS * _distinct(base_raw) if arm == "M" else 0}
        interventions = {
            "canonical": _distinct(base_sig, donor_sig, site, function) if counterfactual else 0,
            "raw": _distinct(base_raw, donor_raw, site, function) if counterfactual else 0}
        second = hybrid_raw if arm == "A" else donor_raw if counterfactual else None
        encoded = {"base": _distinct(base_raw),
                   "hybrid": _distinct(hybrid_raw) if arm == "A" else 0,
                   "donor": _distinct(donor_raw) if counterfactual and arm != "A" else 0,
                   "union": (_distinct_union(base_raw, second) if (second is not None and n)
                             else _distinct(base_raw))}

        def reuse(kind, shown):
            return {"presented": shown, "canonical_answers": canonical[kind],
                    "raw_inputs": raw[kind], "reused_beyond_canonical": shown - canonical[kind],
                    "reused_beyond_raw": shown - raw[kind]}
        counts = expected_update_counts(arm, 1, self.net)
        out = {
            "arm": arm, "seed": self.seed, "updates": self.cursor,
            "canonical_answers": canonical, "raw_inputs": raw, "intervention_tuples": interventions,
            "distinct_raw_contexts_encoded": encoded,
            "corrupted_counterfactual_target_rows": int(
                (np.round(t["p_cf_corrupt"][:n], 12) != np.round(t["p_cf"][:n], 12)).sum())
            if arm == "Ic" else 0,
            "unique_programs": _distinct(function),
            "presentations_and_reuse": {
                "factual": reuse("factual", presented["factual"]),
                "counterfactual": reuse("counterfactual", presented["counterfactual"]),
                "marginal": reuse("marginal", presented["marginal_targets"])},
            "raw_observation_rows": {"useful": self.accounting["rows"]["event_rows_useful"],
                                     "padded_dense": self.accounting["rows"]["event_rows_padded"]},
            "counts": copy.deepcopy(self.accounting),
            "oracle_computations_shared_by_the_tape": copy.deepcopy(self.tape.report["teacher_cost"]),
            "cpu_seconds": self.cpu_seconds,
            "parameters": {"total": int(sum(p.numel() for p in self.net.parameters())),
                           "reached_by_a_loss": int(sum(p.numel() for name, p in
                                                        self.net.named_parameters()
                                                        if counts[name] > 0))},
            "unit_definitions": {
                "canonical_answers": ("distinct teacher answers: factual (base signature, program); "
                                      "counterfactual (hybrid signature, program); marginal (base "
                                      "signature, fact). Raw variants of one signature and donors that "
                                      "give one hybrid are NOT new answers"),
                "raw_inputs": ("distinct student inputs, event ids and event order ignored, "
                               "fact/bit/reliability multiplicity kept: factual (raw base record, "
                               "program); counterfactual (raw hybrid record, program) for A and (raw "
                               "base, raw donor, site, program) for I and Ic; marginal (raw base "
                               "record, fact)"),
                "intervention_tuples": "distinct (base, donor, site, program) questions, canonical "
                                       "(signatures) and raw (records)",
                "distinct_raw_contexts_encoded": ("raw records actually passed through the evidence "
                                                  "encoder: base and hybrid for A, base and donor for "
                                                  "I and Ic, base for F and M; union is the distinct "
                                                  "total")},
            "note": ("F and M differ from A/I in compute and label information; A and I match in "
                     "dense padded compute, not in useful event rows. Elapsed CPU time is reported "
                     "and never asserted. Canonical answers, raw inputs, intervention tuples, "
                     "encoded contexts and presentations are separate units and are never added; "
                     "factual, counterfactual and marginal counts are never added either.")}
        return out

    def deterministic_result(self):
        """Everything that must be identical between an uninterrupted and a resumed run (no timing)."""
        return {"arm": self.arm, "seed": self.seed, "cursor": self.cursor, "bindings": self.bindings,
                "log": copy.deepcopy(self.log), "accounting": copy.deepcopy(self.accounting),
                "state_sha256": self.state_sha256()}


# ---------------------------------------------------------------------------
# checkpoints at update boundaries (plain numeric format, weights_only load)
# ---------------------------------------------------------------------------

CHECKPOINT_NAME = "checkpoint.json"
_CHECKPOINT_KEYS = ("schema", "run_id", "arm", "seed", "cursor", "total_steps", "bindings",
                    "state_sha256", "weights_file", "weights_file_sha256", "accounting",
                    "cpu_seconds", "log", "header_sha256")


def _owned_predecessor(directory, run):
    """The weights file the CURRENT header publishes, or ``None`` without a header. An existing header
    is owned only after it is PROVEN this run's: self-hashed, the same arm, seed, run id and total steps,
    bindings equal to the run's, and a weights file that exists and hashes to the recorded SHA-256."""
    directory = Path(directory)
    path = directory / CHECKPOINT_NAME
    if not os.path.lexists(path):
        return None
    header = fb.read_json(path)
    fm._keys(header, _CHECKPOINT_KEYS, label="the existing checkpoint header")
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    require(fb.digest(body) == header["header_sha256"],
            f"the existing checkpoint header {path} was modified after it was written; it is never "
            f"replaced. {FRESH_OUTPUT}")
    require(header["schema"] == CHECKPOINT_SCHEMA, f"the existing {path} is not a {CHECKPOINT_SCHEMA}")
    require(header["arm"] == run.arm and header["seed"] == run.seed
            and header["run_id"] == run.run_id and header["total_steps"] == run.total_steps,
            f"the existing checkpoint header {path} belongs to a different run; it is never replaced. "
            f"{FRESH_OUTPUT}")
    require(header["bindings"] == run.bindings,
            f"the existing checkpoint header {path} is bound to different inputs; it is never "
            f"replaced. {FRESH_OUTPUT}")
    name = header["weights_file"]
    require(isinstance(name, str) and fst._WEIGHTS_NAME.match(name),
            f"the existing checkpoint header {path} names an unrecognised weights file {name!r}")
    weights = directory / name
    require(weights.is_file() and not weights.is_symlink(),
            f"the weights file {weights} that the existing header publishes is missing; nothing is "
            f"written over it. {FRESH_OUTPUT}")
    require(file_sha256(weights) == header["weights_file_sha256"],
            f"the weights file {weights} does not match its header's SHA-256; it is never "
            f"overwritten. {FRESH_OUTPUT}")
    return name


def save_checkpoint(run, directory, *, protected=()):
    """Publish the run's state at the CURRENT cursor: weights first, the header last (it commits the
    checkpoint). Staging is exclusive and uniquely named, destinations are checked against
    ``protected`` inputs first, the run's consumed inputs are verified, an existing header is PROVEN this
    run's before anything is written, and only the previous owned weights file is removed afterwards
    (never a protected input and never an unknown file)."""
    run.verify_inputs()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = run.state_payload()
    weights_name = f"ckpt_{run.cursor:06d}.pt"
    final, header_path = directory / weights_name, directory / CHECKPOINT_NAME
    fst.refuse_protected(protected, {"checkpoint weights": final, "checkpoint header": header_path})
    previous = _owned_predecessor(directory, run)
    fst._publish_weights(final, payload, tracked=(previous == weights_name))
    body = {"schema": CHECKPOINT_SCHEMA, "run_id": run.run_id, "arm": run.arm, "seed": run.seed,
            "cursor": run.cursor, "total_steps": run.total_steps, "bindings": run.bindings,
            "state_sha256": fst.tensor_state_digest(payload), "weights_file": weights_name,
            "weights_file_sha256": file_sha256(final), "accounting": copy.deepcopy(run.accounting),
            "cpu_seconds": run.cpu_seconds, "log": copy.deepcopy(run.log)}
    header = {**body, "header_sha256": fb.digest(body)}
    fst.write_json_owned(header_path, header, protected=protected)
    if previous and previous != weights_name:
        stale = directory / previous
        if stale.is_file() and not stale.is_symlink() and not fst._is_protected(protected, stale):
            try:
                stale.unlink()
            except OSError:
                pass
    return header


def _validate_checkpoint_accounting(run, header):
    cursor = header["cursor"]
    log, accounting = header["log"], header["accounting"]
    require(isinstance(accounting, dict) and isinstance(log, list) and len(log) == cursor,
            "the checkpoint cursor disagrees with its own log")
    require(accounting == expected_accounting(run.arm, cursor, run.config, run.tape),
            "the checkpoint cursor disagrees with its own accounting: the call, row and presentation "
            "counters are not those of that many updates")
    require(all(isinstance(entry, dict) and entry.get("step") == position
                and entry.get("update_sha256") == run.tape.update_sha256[position]
                for position, entry in enumerate(log)),
            "the checkpoint's step log is not one entry per update of this tape")
    value = header["cpu_seconds"]
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and value >= 0,
            "the checkpoint's cpu_seconds is not a finite non-negative number")


def _validate_checkpoint_payload(run, cursor, payload):
    """Validate against a DISPOSABLE shadow network and optimizer; nothing live is touched."""
    require(isinstance(payload, dict) and set(payload) == {"format", "net", "optimizer"},
            "the checkpoint holds the wrong set of entries")
    shadow = CausalNet(run.spec)
    steps = expected_update_counts(run.arm, cursor, shadow)
    fst._validate_net_state(payload["net"], dict(shadow.state_dict()), "net")
    fst._validate_optimizer_state(payload["optimizer"], shadow, steps, run.config, "net")
    try:
        shadow.load_state_dict(payload["net"])
        fst._make_adam(shadow.parameters(), run.config).load_state_dict(
            fst._clone(payload["optimizer"]))
    except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f"the checkpoint state cannot be loaded into a network and optimizer of "
                         f"this run: {error}") from error


def _verified_checkpoint(directory, run):
    """Read and verify a checkpoint against ``run`` WITHOUT touching it. ``(header, payload)`` or
    ``None`` when there is no checkpoint; any tamper or mismatch raises ``ValueError``."""
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
            "the checkpoint's bindings (initial state, tape, training config, world, bank, program "
            "table, code) differ from this run's: the labels, programs, sites, evidence or tape "
            "membership changed since it was written")
    cursor = header["cursor"]
    require(fb._integer(cursor, "checkpoint cursor", minimum=0) <= run.total_steps,
            "the checkpoint cursor exceeds the run's steps")
    weights = directory / header["weights_file"]
    require(weights.is_file() and weights.name == header["weights_file"],
            "the checkpoint's weights file is missing")
    require(file_sha256(weights) == header["weights_file_sha256"],
            "the checkpoint weights file does not match its recorded SHA-256")
    payload = _load_plain(weights)
    require(isinstance(payload, dict) and payload.get("format") == CHECKPOINT_SCHEMA,
            "the checkpoint weights are not a recognised payload")
    require(fst.tensor_state_digest(payload) == header["state_sha256"],
            "the checkpoint tensors do not match their recorded content digest")
    _validate_checkpoint_accounting(run, header)
    _validate_checkpoint_payload(run, cursor, payload)
    return header, payload


def restore_checkpoint(run, directory):
    """Verify, THEN load, all-or-nothing. Returns True when a checkpoint was applied."""
    checked = _verified_checkpoint(directory, run)
    if checked is None:
        return False
    header, payload = checked
    run.verify_inputs()
    saved = {"payload": run.state_payload(), "cursor": run.cursor,
             "accounting": copy.deepcopy(run.accounting), "log": copy.deepcopy(run.log),
             "seconds": run.cpu_seconds}
    try:
        run.net.load_state_dict(payload["net"])
        run.optimizer.load_state_dict(fst._clone(payload["optimizer"]))
        run.cursor = header["cursor"]
        run.accounting = copy.deepcopy(header["accounting"])
        run.log = copy.deepcopy(header["log"])
        run.cpu_seconds = header["cpu_seconds"]
    except BaseException:
        run.net.load_state_dict(saved["payload"]["net"])
        run.optimizer.load_state_dict(fst._clone(saved["payload"]["optimizer"]))
        run.cursor, run.accounting = saved["cursor"], saved["accounting"]
        run.log, run.cpu_seconds = saved["log"], saved["seconds"]
        raise
    return True


def train_arm(ctx, arm, seed, directory, *, stop_after=None, protected=()):
    """Train (or resume) one run, checkpointing every ``checkpoint_every`` updates, at the end and at
    ``stop_after`` (an interruption point for tests). Returns the run."""
    run = ArmRun(ctx, arm, seed)
    restore_checkpoint(run, directory)
    every = ctx.config.checkpoint_every
    while run.cursor < run.total_steps and (stop_after is None or run.cursor < stop_after):
        run.step()
        if (run.cursor % every == 0 or run.cursor == run.total_steps
                or (stop_after is not None and run.cursor == stop_after)):
            save_checkpoint(run, directory, protected=protected)
    return run


# ---------------------------------------------------------------------------
# frozen source states
# ---------------------------------------------------------------------------

FROZEN_KEYS = ("schema", "run_id", "arm", "seed", "updates", "bindings", "state_sha256",
               "weights_file", "weights_file_sha256", "accounting", "cost", "final_step",
               "header_sha256")


def publish_frozen(directory, *, rid, arm, seed, state, bindings, updates, accounting, cost,
                   final_step, protected=()):
    """Publish one frozen network (plain tensors, weights first, header last). An identical existing
    publication is adopted; a different one is refused, never overwritten."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = fst._clone(dict(state))
    weights, header_path = directory / f"{rid}.pt", directory / f"{rid}.frozen.json"
    fst.refuse_protected(protected, {"frozen weights": weights, "frozen header": header_path})
    digest = fst.tensor_state_digest(state)
    if os.path.lexists(weights):
        require(fst._load_payload_digest(weights) == digest,
                f"{weights} already exists with different content; it is never overwritten. "
                f"{FRESH_OUTPUT}")
    else:
        fst._publish(weights, lambda handle: torch.save(state, handle), text=False)
    body = {"schema": FROZEN_SCHEMA, "run_id": rid, "arm": arm, "seed": int(seed),
            "updates": int(updates), "bindings": bindings, "state_sha256": digest,
            "weights_file": weights.name, "weights_file_sha256": file_sha256(weights),
            "accounting": accounting, "cost": cost, "final_step": final_step}
    header = {**body, "header_sha256": fb.digest(body)}
    if os.path.lexists(header_path):
        existing = fb.read_json(header_path)
        require(existing["state_sha256"] == digest and existing["bindings"] == bindings
                and existing["run_id"] == rid,
                f"{header_path} already freezes a different state; it is never replaced. {FRESH_OUTPUT}")
        return existing
    fst.write_json_owned(header_path, header, protected=protected)
    return header


def freeze_run(run, directory, *, protected=()):
    require(run.cursor == run.total_steps,
            f"{run.run_id} has taken {run.cursor} of {run.total_steps} updates: only a complete run "
            "is frozen")
    run.verify_inputs()
    return publish_frozen(
        directory, rid=run.run_id, arm=run.arm, seed=run.seed, state=run.net.state_dict(),
        bindings=run.bindings, updates=run.cursor, accounting=copy.deepcopy(run.accounting),
        cost=run.cost_report(), final_step=copy.deepcopy(run.log[-1]) if run.log else None,
        protected=protected)


def freeze_scratch(ctx, seed, directory, *, protected=()):
    """The scratch control: the seed's shared initial state, never trained."""
    ctx.verify_live(seed)
    return publish_frozen(
        directory, rid=run_id(SCRATCH, seed), arm=SCRATCH, seed=seed, state=ctx.init_states[seed],
        bindings=ctx.bindings(seed), updates=0, accounting=_fresh_accounting(), cost={},
        final_step=None, protected=protected)


@dataclass(frozen=True, eq=False)
class FrozenModel:
    """A published source state loaded read-only: evaluation mode, no gradients."""

    run_id: str
    arm: str
    seed: int
    net: object
    header: dict
    state_sha256: str

    def verify_unchanged(self):
        require(fst.tensor_state_digest(dict(self.net.state_dict())) == self.state_sha256,
                f"{self.run_id}'s weights changed after they were frozen")
        return True


def load_frozen(directory, rid, spec, *, bindings=None):
    directory = Path(directory)
    header = fb.read_json(directory / f"{rid}.frozen.json")
    fm._keys(header, FROZEN_KEYS, label="frozen header")
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    require(fb.digest(body) == header["header_sha256"],
            f"the frozen header of {rid} was modified after it was written")
    require(header["schema"] == FROZEN_SCHEMA and header["run_id"] == rid, f"not the frozen state of {rid}")
    require(bindings is None or header["bindings"] == bindings,
            f"the frozen state of {rid} is bound to other inputs than this run's")
    weights = directory / header["weights_file"]
    require(weights.name == f"{rid}.pt" and weights.is_file(),
            f"the frozen weights file of {rid} is missing")
    require(file_sha256(weights) == header["weights_file_sha256"],
            f"the frozen weights of {rid} do not match their recorded SHA-256")
    state = _load_plain(weights)
    require(fst.tensor_state_digest(state) == header["state_sha256"],
            f"the frozen tensors of {rid} do not match their recorded digest")
    net = net_from_state(spec, state)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return FrozenModel(run_id=rid, arm=header["arm"], seed=header["seed"], net=net, header=header,
                       state_sha256=header["state_sha256"])


# ---------------------------------------------------------------------------
# inference helpers (student side) and the held-out oracle (teacher side)
# ---------------------------------------------------------------------------

def encode_arrays(net, arrays, *, chunk=4096):
    """Hidden blocks ``[n, 4, block]`` of padded event arrays (no gradients)."""
    count = int(arrays["mask"].shape[0])
    parts = []
    with torch.no_grad():
        for lo in range(0, count, chunk):
            parts.append(net.encode(event_batch(*(arrays[name][lo:lo + chunk]
                                                 for name in EVENT_FIELDS))))
    if not parts:
        return torch.zeros((0, N_BLOCKS, net.spec.block_dim))
    return torch.cat(parts)


def program_states(net, function_indices):
    """Program-encoder states ``[f, program_hidden]`` of registry functions (no gradients)."""
    functions = function_arrays()
    index = np.asarray(list(function_indices), dtype=np.int64)
    with torch.no_grad():
        return net.encode_program(program_batch(functions["tokens"][index],
                                                functions["length"][index]))


def cross_predict(net, blocks, states, *, penultimate=False, chunk_rows=32768):
    """Logits ``[n, f]`` (float64 of the float32 logits) of every (context, program) pair, and, with
    ``penultimate``, the query-conditioned penultimate activations ``[n, f, head_hidden]``."""
    count, programs = int(blocks.shape[0]), int(states.shape[0])
    per = max(1, chunk_rows // max(1, programs))
    logits, hidden = [], []
    with torch.no_grad():
        for lo in range(0, count, per):
            sub = blocks[lo:lo + per]
            m = int(sub.shape[0])
            joint_blocks = sub.unsqueeze(1).expand(m, programs, N_BLOCKS, sub.shape[2]).reshape(
                m * programs, N_BLOCKS, sub.shape[2])
            joint_states = states.unsqueeze(0).expand(m, programs, states.shape[1]).reshape(
                m * programs, states.shape[1])
            logit, penult = net.predict(joint_blocks, joint_states)
            logits.append(logit.reshape(m, programs))
            if penultimate:
                hidden.append(penult.reshape(m, programs, penult.shape[-1]))
    all_logits = (torch.cat(logits) if logits else torch.zeros((0, programs))).double().numpy()
    if penultimate:
        return all_logits, (torch.cat(hidden) if hidden else torch.zeros((0, programs, 0))).numpy()
    return all_logits


@dataclass(frozen=True, eq=False)
class ContextSet:
    """Raw-evidence contexts of one split at the declared evaluation variant and the padded arrays the
    student reads. ``posteriors`` are the teacher's exact answers (teacher side): ``None`` on a RAW set,
    which holds the evidence only and is what feature extraction may use."""

    split: str
    signatures: tuple
    events: tuple
    arrays: object
    posteriors: object
    labels: tuple


class PanelOracle:
    """Teacher-side labels for the held-out panels, built lazily and RECORDED: a split's contexts are
    labelled only when first requested (``opened`` lists them in order), so a test can show that no
    held-out label was computed before the freeze. The evidence itself (``raw_context_set``) is a separate
    step that never computes a posterior and is not recorded as opening a split: extracting a feature
    needs the raw events only."""

    def __init__(self, config):
        self.config = config
        self.opened = []
        self._sets, self._raw, self._cache = {}, {}, {}

    def raw_context_set(self, split):
        """The raw evidence of a split: signatures, raw events and padded arrays, with NO teacher
        quantity (no posterior is computed and nothing is added to ``opened``)."""
        if split not in self._raw:
            world = fe.q7_world()
            if split == "source":
                signatures = world.source
            else:
                require(split in fe.CENTER_SPLITS, f"unknown context split {split!r}")
                signatures = world.centers_by_split[split]
            cfg = self.config
            events = tuple(fe.raw_events(s, cfg.eval_variant, cancel_max=cfg.cancel_max,
                                         null_max=cfg.null_max) for s in signatures)
            self._raw[split] = ContextSet(
                split=split, signatures=tuple(signatures), events=events,
                arrays=_owned(events_to_arrays(events, cfg.event_width)), posteriors=None,
                labels=tuple(fe.context_text(s) for s in signatures))
        return self._raw[split]

    def context_set(self, split):
        """The contexts of a split WITH the teacher's exact posteriors: this is what opens the split."""
        if split not in self._sets:
            raw = self.raw_context_set(split)
            self.opened.append(split)
            self._sets[split] = ContextSet(
                split=split, signatures=raw.signatures, events=raw.events, arrays=raw.arrays,
                posteriors=tuple(fe.world_posterior(e) for e in raw.events), labels=raw.labels)
        return self._sets[split]

    @staticmethod
    def probabilities(posteriors, function_indices):
        """Exact probabilities ``[n, f]`` (a Fraction sum of world weights, then a float)."""
        registry = fe.function_registry()
        tables = [registry.functions[i].table for i in function_indices]
        out = np.zeros((len(posteriors), len(tables)), dtype=np.float64)
        for i, posterior in enumerate(posteriors):
            for j, table in enumerate(tables):
                out[i, j] = float(fe.table_probability(table, posterior))
        return out

    def matrix(self, split, function_indices):
        key = ("matrix", split, tuple(function_indices))
        if key not in self._cache:
            self._cache[key] = self.probabilities(self.context_set(split).posteriors,
                                                  function_indices)
        return self._cache[key]

    def hybrids(self, split):
        """Every (base, donor, site) raw-evidence hybrid of a split, base-major: the index triples, the
        padded arrays and the exact hybrid posteriors."""
        key = ("hybrids", split)
        if key not in self._cache:
            cs = self.context_set(split)
            count = len(cs.signatures)
            index, events = [], []
            for b in range(count):
                for d in range(count):
                    for k in range(fe.N_FACTS):
                        hybrid = fe.hybrid_events(cs.events[b], cs.events[d], k)
                        require(fe.signature_of_events(hybrid)
                                == fe.hybrid_signature(cs.signatures[b], cs.signatures[d], k),
                                "a raw-evidence hybrid left its labelled signature")
                        index.append((b, d, k))
                        events.append(hybrid)
            self._cache[key] = {
                "index": np.array(index, dtype=np.int64),
                "arrays": _owned(events_to_arrays(events, self.config.event_width)),
                "posteriors": tuple(fe.world_posterior(e) for e in events)}
        return self._cache[key]

    def restricted(self, split):
        """Posteriors of each context with ONE fact unobserved (``without``) and with ONLY one fact
        observed (``only``), and of no evidence at all: the oracles of ablation and rescue."""
        key = ("restricted", split)
        if key not in self._cache:
            cs = self.context_set(split)
            without = [[fe.world_posterior(fe.restrict_events(
                events, [f for f in range(fe.N_FACTS) if f != k])) for k in range(fe.N_FACTS)]
                for events in cs.events]
            only = [[fe.world_posterior(fe.restrict_events(events, [k]))
                     for k in range(fe.N_FACTS)] for events in cs.events]
            self._cache[key] = {"without": without, "only": only,
                                "empty": fe.world_posterior(())}
        return self._cache[key]


# ---------------------------------------------------------------------------
# metrics and summaries
# ---------------------------------------------------------------------------

def case_metrics(oracle, prediction):
    """Per-case excess log loss (KL of the exact answer from the prediction), excess Brier and error."""
    return {"kl": bernoulli_kl(oracle, prediction), "brier_excess": (prediction - oracle) ** 2,
            "abs_error": np.abs(prediction - oracle)}


def _group_means(values, mask, groups):
    return {int(g): float(values[mask & (groups == g)].mean()) for g in np.unique(groups[mask])}


def summarize(metrics, masks, groups, group_labels):
    """Exact finite-panel means, overall and by group, for every mask. ``groups`` maps a group name to
    an integer array and ``group_labels`` to the label of each integer. Each case is one correlated
    panel cell, not an independent sample: only means (never intervals) are reported."""
    out = {}
    for mask_name, mask in masks.items():
        count = int(mask.sum())
        entry = {"n": count, "metrics": {}}
        if count:
            for name, values in metrics.items():
                item = {"mean": float(values[mask].mean())}
                for group, ids in groups.items():
                    means = _group_means(values, mask, ids)
                    item[f"by_{group}"] = {str(group_labels[group][g]): m for g, m in means.items()}
                    item[f"mean_of_{group}_means"] = float(np.mean(list(means.values())))
                entry["metrics"][name] = item
        out[mask_name] = entry
    return out


def calibration(prediction, oracle, bins=10):
    """Equal-width calibration of predictions against the exact answer probabilities."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    oracle = np.asarray(oracle, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(prediction, edges[1:-1]), 0, bins - 1)
    rows, error = [], 0.0
    for b in range(bins):
        sel = index == b
        if sel.any():
            gap = abs(float(prediction[sel].mean()) - float(oracle[sel].mean()))
            error += sel.mean() * gap
            rows.append({"bin": b, "n": int(sel.sum()), "mean_prediction": float(prediction[sel].mean()),
                         "mean_oracle": float(oracle[sel].mean())})
    return {"bins": rows, "expected_calibration_error": float(error),
            "mean_predicted_entropy": float(bernoulli_entropy(prediction).mean()),
            "mean_oracle_entropy": float(bernoulli_entropy(oracle).mean()),
            "definition": ("the 'outcome' of a bin is the mean exact answer probability, not a "
                           "realised 0/1 draw; realised entropy of one contradictory record can "
                           "exceed the prior's, so entropy is descriptive")}


def ordinary_panel(net, oracle, split, function_indices, *, calibrate=True):
    """Ordinary predictions of every (context of ``split``) x (function) pair against the exact oracle."""
    cs = oracle.context_set(split)
    registry = fe.function_registry()
    functions = list(function_indices)
    blocks = encode_arrays(net, cs.arrays)
    logits = cross_predict(net, blocks, program_states(net, functions))
    prediction, exact = sigmoid64(logits), oracle.matrix(split, functions)
    count, width = prediction.shape
    families = sorted({registry.functions[i].family for i in functions})
    family_of = np.array([families.index(registry.functions[i].family) for i in functions])
    metrics = case_metrics(exact.reshape(-1), prediction.reshape(-1))
    metrics["predicted_entropy"] = bernoulli_entropy(prediction.reshape(-1))
    metrics["oracle_entropy"] = bernoulli_entropy(exact.reshape(-1))
    groups = {"function_family": np.tile(family_of, count),
              "context": np.repeat(np.arange(count), width)}
    labels = {"function_family": families, "context": list(cs.labels)}
    out = {"cases": int(count * width), "contexts": int(count), "functions": int(width),
           "summary": summarize(metrics, {"all": np.ones(count * width, dtype=bool)}, groups, labels)}
    if calibrate:
        out["calibration"] = calibration(prediction, exact)
    return out


def _depends(function_indices):
    registry = fe.function_registry()
    return np.array([[fe.depends_on(registry.functions[i].table, k) for i in function_indices]
                     for k in range(fe.N_FACTS)], dtype=bool)


def intervention_panel(net, oracle, config):
    """The strong primary intervention panel: base AND donor both from the 14 unseen primary centers,
    crossed with 4 sites and the 27 primary queries (21168 cases from 14 base contexts).

    For every case it reports the patched prediction's error against the exact counterfactual oracle,
    the ordinary prediction on the raw-evidence hybrid (and its oracle error), AND the agreement of the
    two predictions: two wrong predictions can agree perfectly, so agreement is never reported alone."""
    registry = fe.function_registry()
    split = fe.SPLIT_PRIMARY
    cs = oracle.context_set(split)
    functions = [f.index for f in registry.role(fe.ROLE_PRIMARY)]
    hybrid = oracle.hybrids(split)
    b_idx, d_idx, k_idx = hybrid["index"][:, 0], hybrid["index"][:, 1], hybrid["index"][:, 2]
    nf, cases = len(functions), int(b_idx.shape[0]) * len(functions)
    base_blocks = encode_arrays(net, cs.arrays)
    hybrid_blocks = encode_arrays(net, hybrid["arrays"])
    states = program_states(net, functions)
    base_logits = cross_predict(net, base_blocks, states)
    hybrid_logits = cross_predict(net, hybrid_blocks, states)
    site = torch.tensor(k_idx, dtype=torch.int64)
    b_rows, d_rows = torch.tensor(b_idx, dtype=torch.int64), torch.tensor(d_idx, dtype=torch.int64)
    patched = CausalNet.patch_blocks(base_blocks[b_rows], base_blocks[d_rows], site)
    patched_logits = cross_predict(net, patched, states)
    count = int(b_idx.shape[0])
    wrong_donor = base_blocks[d_rows][torch.arange(count), (site + 1) % N_BLOCKS]
    mismatched = base_blocks[b_rows].clone()
    mismatched[torch.arange(count), site] = wrong_donor
    mismatch_logits = cross_predict(net, mismatched, states)
    identity = 0.0
    for k in range(N_BLOCKS):
        same = CausalNet.patch_blocks(base_blocks, base_blocks,
                                      torch.full((base_blocks.shape[0],), k, dtype=torch.int64))
        identity = max(identity, float(np.abs(cross_predict(net, same, states) - base_logits).max()))
    o_cf = oracle.probabilities(hybrid["posteriors"], functions)
    o_base = oracle.matrix(split, functions)[b_idx]
    p_patched, p_hybrid = sigmoid64(patched_logits), sigmoid64(hybrid_logits)
    p_base, p_mismatch = sigmoid64(base_logits)[b_idx], sigmoid64(mismatch_logits)

    def flat(array):
        return np.asarray(array).reshape(-1)
    metrics = {}
    for name, prediction in (("patched", p_patched), ("hybrid", p_hybrid), ("mismatched", p_mismatch)):
        for key, values in case_metrics(flat(o_cf), flat(prediction)).items():
            metrics[f"{name}_{key}"] = values
    metrics["agreement_abs"] = flat(np.abs(p_patched - p_hybrid))
    metrics["agreement_kl"] = bernoulli_kl(flat(p_hybrid), flat(p_patched))
    metrics["patched_response"] = flat(np.abs(p_patched - p_base))
    metrics["hybrid_response"] = flat(np.abs(p_hybrid - p_base))
    metrics["oracle_response"] = flat(np.abs(o_cf - o_base))
    metrics["base_ordinary_kl"] = bernoulli_kl(flat(o_base), flat(p_base))
    delta = metrics["oracle_response"]
    changed, unchanged = delta >= CHANGED_THRESHOLD, delta <= UNCHANGED_TOLERANCE
    k_case = np.repeat(k_idx, nf)
    f_case = np.tile(np.arange(nf), count)
    relevant = _depends(functions)[k_case, f_case]
    same_value = np.repeat(np.array([cs.signatures[d][k] == cs.signatures[b][k]
                                     for b, d, k in hybrid["index"]]), nf)
    masks = {"all": np.ones(cases, dtype=bool), "changed": changed, "weak": ~changed & ~unchanged,
             "unchanged": unchanged, "relevant": relevant, "irrelevant": ~relevant,
             "changed_relevant": changed & relevant, "unchanged_relevant": unchanged & relevant,
             "unchanged_irrelevant": unchanged & ~relevant, "same_value_donor": same_value,
             "different_value_donor": ~same_value, "identity_donor": np.repeat(b_idx == d_idx, nf)}
    families = sorted({registry.functions[i].family for i in functions})
    family_of = np.array([families.index(registry.functions[i].family) for i in functions])
    groups = {"function_family": family_of[f_case], "base_context": np.repeat(b_idx, nf),
              "site": k_case}
    labels = {"function_family": families, "base_context": list(cs.labels),
              "site": [f"site{k}" for k in range(N_BLOCKS)]}
    return {
        "cases": cases, "base_contexts": int(base_blocks.shape[0]), "donor_contexts": int(
            base_blocks.shape[0]), "sites": N_BLOCKS, "queries": nf,
        "changed_threshold": CHANGED_THRESHOLD,
        "strata_definition": ("changed: oracle |p_cf - p_base| >= 0.1; unchanged: exactly zero; weak: "
                              "between. Site relevance alone is not a changed-case definition: a "
                              "relevant variable can produce no probability change"),
        "identity_insertion_max_abs_logit_difference": identity,
        "summary": summarize(metrics, masks, groups, labels),
        "note": ("exact finite-panel means over correlated cases from 14 base contexts; paired seed "
                 "differences, not row-bootstrap population evidence")}


def source_train_mean_blocks(net, bank):
    """The frozen SOURCE-TRAIN mean of every hidden block: computed ONCE from the frozen model's states
    on every training (signature, variant) context, never from an evaluation batch."""
    blocks = encode_arrays(net, {name: bank.arrays[name] for name in EVENT_FIELDS})
    return blocks.mean(dim=0)


def rescue_gain(full_prediction, rescue_prediction, rescue_oracle):
    """The improvement in absolute error from rescuing one block, with BOTH predictions scored against
    the SAME target: the exact answer with only that fact observed. Positive means the rescued block moved
    the prediction toward that answer. (Scoring the full ablation against the empty-evidence answer and
    the rescue against the only-k answer and subtracting would compare errors to different targets: with
    empty 0.25, only-k 0.375, full prediction 0.25 and rescue prediction 0.30 it would report -0.075 while
    the error to 0.375 shrinks from 0.125 to 0.075, a gain of +0.05.)"""
    rescue_oracle = np.asarray(rescue_oracle, dtype=np.float64)
    return (np.abs(np.asarray(full_prediction, dtype=np.float64) - rescue_oracle)
            - np.abs(np.asarray(rescue_prediction, dtype=np.float64) - rescue_oracle))


def ablation_panel(net, oracle, config, mean_blocks):
    """Frozen SOURCE-TRAIN mean ablation of one block and rescue of one block (primary contexts, the 27
    primary queries, 4 sites). ``ablate_k`` replaces block k by the training mean and is scored against
    the exact answer with fact k UNOBSERVED; ``rescue_k`` ablates every block but k and is scored
    against the exact answer with ONLY fact k observed. ``rescue_gain_abs`` scores the full ablation and
    the rescued prediction against that SAME only-k answer (:func:`rescue_gain`), while the full
    ablation's fidelity to the EMPTY-evidence answer is its own diagnostic
    (``full_ablation_empty_abs_error``). Descriptive: a mean state need not equal the 'no evidence'
    state."""
    cases = ablation_cases(net, oracle, config, mean_blocks)
    return {"cases": cases["count"], "reference": "mean of the frozen model's states on every "
            "training (signature, variant) context, computed once",
            "summary": summarize(cases["metrics"], cases["masks"], cases["groups"], cases["labels"])}


def ablation_cases(net, oracle, config, mean_blocks):
    """The per-case arrays behind :func:`ablation_panel`: ``metrics``, ``masks``, ``groups``, ``labels``
    for ``summarize`` and the underlying ``predictions`` and ``oracles`` (case order: context, site,
    query)."""
    registry = fe.function_registry()
    split = fe.SPLIT_PRIMARY
    cs = oracle.context_set(split)
    functions = [f.index for f in registry.role(fe.ROLE_PRIMARY)]
    nf, nb = len(functions), len(cs.signatures)
    restricted = oracle.restricted(split)
    base_blocks = encode_arrays(net, cs.arrays)
    states = program_states(net, functions)
    base_p = sigmoid64(cross_predict(net, base_blocks, states))
    exact_base = oracle.matrix(split, functions)
    ablate, rescue = [], []
    for b in range(nb):
        for k in range(N_BLOCKS):
            a = base_blocks[b].clone()
            a[k] = mean_blocks[k]
            ablate.append(a)
            r = mean_blocks.clone()
            r[k] = base_blocks[b][k]
            rescue.append(r)
    p_ablate = sigmoid64(cross_predict(net, torch.stack(ablate), states))
    p_rescue = sigmoid64(cross_predict(net, torch.stack(rescue), states))
    p_full = sigmoid64(cross_predict(net, mean_blocks.unsqueeze(0), states))[0]
    o_full = oracle.probabilities([restricted["empty"]], functions)[0]
    o_ablate = oracle.probabilities([restricted["without"][b][k] for b in range(nb)
                                     for k in range(N_BLOCKS)], functions)
    o_rescue = oracle.probabilities([restricted["only"][b][k] for b in range(nb)
                                     for k in range(N_BLOCKS)], functions)
    b_case = np.repeat(np.arange(nb), N_BLOCKS * nf)
    k_case = np.tile(np.repeat(np.arange(N_BLOCKS), nf), nb)
    f_case = np.tile(np.arange(nf), nb * N_BLOCKS)
    p_base_case = base_p[b_case, f_case]
    metrics = {}
    for key, values in case_metrics(o_ablate.reshape(-1), p_ablate.reshape(-1)).items():
        metrics[f"ablate_{key}"] = values
    for key, values in case_metrics(o_rescue.reshape(-1), p_rescue.reshape(-1)).items():
        metrics[f"rescue_{key}"] = values
    # the full ablation (every block at its training mean) is one prediction per query; it is scored
    # against the empty-evidence answer as a diagnostic of its own, and against the SAME only-k answer
    # as the rescued prediction so that the gain compares like with like
    metrics["full_ablation_empty_abs_error"] = np.abs(p_full[f_case] - o_full[f_case])
    metrics["full_ablation_rescue_target_abs_error"] = np.abs(p_full[f_case] - o_rescue.reshape(-1))
    metrics["rescue_gain_abs"] = rescue_gain(p_full[f_case], p_rescue.reshape(-1), o_rescue.reshape(-1))
    metrics["ablation_effect"] = np.abs(p_ablate.reshape(-1) - p_base_case)
    metrics["oracle_ablation_effect"] = np.abs(o_ablate.reshape(-1) - exact_base[b_case, f_case])
    relevant = _depends(functions)[k_case, f_case]
    delta = metrics["oracle_ablation_effect"]
    masks = {"all": np.ones(b_case.shape[0], dtype=bool), "relevant": relevant,
             "irrelevant": ~relevant, "changed": delta >= CHANGED_THRESHOLD,
             "unchanged": delta <= UNCHANGED_TOLERANCE}
    families = sorted({registry.functions[i].family for i in functions})
    family_of = np.array([families.index(registry.functions[i].family) for i in functions])
    groups = {"function_family": family_of[f_case], "base_context": b_case, "site": k_case}
    labels = {"function_family": families, "base_context": list(cs.labels),
              "site": [f"site{k}" for k in range(N_BLOCKS)]}
    return {"count": int(b_case.shape[0]), "metrics": metrics, "masks": masks, "groups": groups,
            "labels": labels,
            "predictions": {"full": p_full[f_case], "ablate": p_ablate.reshape(-1),
                            "rescue": p_rescue.reshape(-1), "base": p_base_case},
            "oracles": {"empty": o_full[f_case], "without": o_ablate.reshape(-1),
                        "only": o_rescue.reshape(-1)}}


def invariance_checks(net, oracle, config, *, rows=48, tolerance=1e-4):
    """Evidence-order, right-padding and batch-composition invariance of the encoders (the set encoder
    makes them hold by construction; this MEASURES them on the frozen model) and right-padding /
    batch-composition invariance of the program encoder."""
    cs = oracle.context_set("source")
    arrays = {name: np.asarray(cs.arrays[name][:rows]) for name in EVENT_FIELDS}
    reference = encode_arrays(net, arrays)
    rng = np.random.default_rng([config.tape_seed, 5150])
    keys = rng.random(arrays["mask"].shape) + np.logical_not(arrays["mask"]) * 10.0
    order = np.argsort(keys, axis=1)
    permuted = {name: np.take_along_axis(arrays[name], order, axis=1) for name in EVENT_FIELDS}
    padded = {name: np.concatenate([arrays[name], np.zeros((rows, 5), dtype=arrays[name].dtype)],
                                   axis=1) for name in EVENT_FIELDS}
    singles = torch.cat([encode_arrays(net, {name: arrays[name][i:i + 1] for name in EVENT_FIELDS})
                         for i in range(rows)])
    functions = function_arrays()
    count = int(functions["tokens"].shape[0])
    tokens, length = np.asarray(functions["tokens"]), np.asarray(functions["length"])
    with torch.no_grad():
        base_state = net.encode_program(program_batch(tokens, length))
        wide_state = net.encode_program(program_batch(
            np.concatenate([tokens, np.zeros((count, 3), dtype=tokens.dtype)], axis=1), length))
        reversed_state = net.encode_program(program_batch(tokens[::-1].copy(), length[::-1].copy()))[
            torch.arange(count - 1, -1, -1)]

    def worst(a, b):
        return float((a - b).abs().max())
    diffs = {"event_permutation_max_abs_block_difference": worst(reference, encode_arrays(net, permuted)),
             "right_padding_max_abs_block_difference": worst(reference, encode_arrays(net, padded)),
             "batch_composition_max_abs_block_difference": worst(reference, singles),
             "program_right_padding_max_abs_difference": worst(base_state, wide_state),
             "program_batch_composition_max_abs_difference": worst(base_state, reversed_state)}
    return {**diffs, "tolerance": tolerance,
            "all_within_tolerance": bool(all(v <= tolerance for v in diffs.values())),
            "note": ("a measured property of the learned model: the masked sum and the causal GRU make "
                     "it hold up to float rounding by construction of the architecture; it says "
                     "nothing about whether the network learned deduplication (duplicate-id "
                     "semantics belong to the teacher's validator)")}


def evaluate_model(model, oracle, config, bank):
    """Every held-out measurement of one FROZEN model. Opens the held-out panels through ``oracle``."""
    net = model.net
    registry = fe.function_registry()
    source_fns = [f.index for f in registry.role(fe.ROLE_SOURCE)]
    primary_fns = [f.index for f in registry.role(fe.ROLE_PRIMARY)]
    validation_fns = [f.index for f in registry.role(fe.ROLE_VALIDATION)]
    mean_blocks = source_train_mean_blocks(net, bank)
    panels = {
        "source_fit": ordinary_panel(net, oracle, "source", source_fns),
        "primary_composition": ordinary_panel(net, oracle, fe.SPLIT_PRIMARY, primary_fns),
        "primary_source_preservation": ordinary_panel(net, oracle, fe.SPLIT_PRIMARY, source_fns),
        "validation_new_function": ordinary_panel(net, oracle, fe.SPLIT_VALIDATION, validation_fns),
        "validation_source_function": ordinary_panel(net, oracle, fe.SPLIT_VALIDATION, source_fns)}
    interventions = intervention_panel(net, oracle, config)
    ablation = ablation_panel(net, oracle, config, mean_blocks)
    invariance = invariance_checks(net, oracle, config)

    def mean_of(panel, key="kl"):
        return panel["summary"]["all"]["metrics"][key]["mean"]

    def intervention(mask, key):
        entry = interventions["summary"][mask]
        return entry["metrics"][key]["mean"] if entry["n"] else None

    def ablation_value(mask, key):
        entry = ablation["summary"][mask]
        return entry["metrics"][key]["mean"] if entry["n"] else None
    headline = {
        "source_fit_kl": mean_of(panels["source_fit"]),
        "primary_composition_kl": mean_of(panels["primary_composition"]),
        "primary_composition_brier_excess": mean_of(panels["primary_composition"], "brier_excess"),
        "primary_source_preservation_kl": mean_of(panels["primary_source_preservation"]),
        "validation_new_function_kl": mean_of(panels["validation_new_function"]),
        "validation_source_function_kl": mean_of(panels["validation_source_function"]),
        "intervention_all_patched_kl": intervention("all", "patched_kl"),
        "intervention_all_hybrid_kl": intervention("all", "hybrid_kl"),
        "intervention_changed_patched_kl": intervention("changed", "patched_kl"),
        "intervention_changed_hybrid_kl": intervention("changed", "hybrid_kl"),
        "intervention_changed_agreement_abs": intervention("changed", "agreement_abs"),
        "intervention_changed_patched_abs": intervention("changed", "patched_abs_error"),
        "intervention_changed_mismatched_abs": intervention("changed", "mismatched_abs_error"),
        "intervention_unchanged_patched_response": intervention("unchanged", "patched_response"),
        "intervention_irrelevant_patched_response": intervention("irrelevant", "patched_response"),
        "intervention_same_value_patched_response": intervention("same_value_donor",
                                                                 "patched_response"),
        "intervention_irrelevant_hybrid_response": intervention("irrelevant", "hybrid_response"),
        "ablation_relevant_effect": ablation_value("relevant", "ablation_effect"),
        "ablation_irrelevant_effect": ablation_value("irrelevant", "ablation_effect"),
        "ablation_all_abs_error": ablation_value("all", "ablate_abs_error"),
        "rescue_gain_abs": ablation_value("all", "rescue_gain_abs"),
        "identity_insertion_max_abs_logit_difference":
            interventions["identity_insertion_max_abs_logit_difference"]}
    return {"run_id": model.run_id, "arm": model.arm, "seed": model.seed,
            "state_sha256": model.state_sha256, "panels": panels, "interventions": interventions,
            "ablation_and_rescue": ablation, "invariance": invariance,
            "mean_reference_sha256": _array_sha(mean_blocks.numpy()), "headline": headline}


# ---------------------------------------------------------------------------
# the small transfer branch
# ---------------------------------------------------------------------------

TRANSFER_STREAM = 3030


def transfer_selection(config):
    """The train-only, nested, round-robin selection of (transfer-fit context index, function index)
    pairs from the 168 transfer-fitting candidates (14 contexts x 12 functions). Round r takes every
    context's r-th function (each context has its own seeded order), so the first 16 cover all 14
    contexts and the first 64 give every context 4 or 5. No label is read to choose them."""
    world, registry = fe.q7_world(), fe.function_registry()
    contexts = len(world.centers_by_split[fe.SPLIT_TRANSFER_FIT])
    functions = [f.index for f in registry.role(fe.ROLE_TRANSFER)]
    orders = []
    for c in range(contexts):
        rng = np.random.default_rng([config.tape_seed, TRANSFER_STREAM, c])
        orders.append([functions[int(i)] for i in rng.permutation(len(functions))])
    return tuple((c, orders[c][r]) for r in range(len(functions)) for c in range(contexts))


def selection_report(sequence, budget):
    """Contexts and derived labels of one budget, counted SEPARATELY."""
    chosen = sequence[:budget]
    per = {}
    for context, _ in chosen:
        per[context] = per.get(context, 0) + 1
    return {"budget": int(budget), "derived_labels": len(chosen), "contexts_covered": len(per),
            "labels_per_context_min": min(per.values()) if per else 0,
            "labels_per_context_max": max(per.values()) if per else 0,
            "distinct_functions": len({f for _, f in chosen}),
            "nested": "every budget is a prefix of every larger budget",
            "contexts_are_independent_units": ("the 14 transfer-fit contexts, not the labels, are the "
                                               "independent units")}


class ResidualAdapter(nn.Module):
    """A small residual correction ``logit = source_logit + adapter(penultimate)`` with a zero-initialised
    output layer, so an untrained adapter reproduces the frozen source exactly and removing it restores
    the source prediction."""

    def __init__(self, input_dim, hidden):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, penultimate):
        return self.out(F.gelu(self.hidden(penultimate))).squeeze(-1)


def make_adapter(config, seed, input_dim):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fe.stable_seed("adapter", seed) % (2 ** 31))
        return ResidualAdapter(input_dim, config.adapter_hidden)


def adapt_adapter(model, oracle, config, sequence, budget):
    """Fit one adapter on the first ``budget`` selected pairs with a FIXED recipe (full-batch Adam, fixed
    steps and learning rate, no held-out selection). Only the chosen pairs' teacher labels are read. The
    frozen source network is never modified. Returns ``(adapter, record)``."""
    require(budget >= 1, "adapt_adapter needs a positive budget (N0 is the unchanged frozen source)")
    net, registry = model.net, fe.function_registry()
    cs = oracle.context_set(fe.SPLIT_TRANSFER_FIT)
    chosen = sequence[:budget]
    contexts = torch.tensor([c for c, _ in chosen], dtype=torch.int64)
    functions = [f for _, f in chosen]
    labels = np.array([float(fe.table_probability(registry.functions[f].table, cs.posteriors[c]))
                       for c, f in chosen], dtype=np.float64)
    blocks = encode_arrays(net, cs.arrays)[contexts]
    states = program_states(net, functions)
    with torch.no_grad():
        source_logit, penultimate = net.predict(blocks, states)
    adapter = make_adapter(config, model.seed, int(penultimate.shape[1]))
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config.adapt_lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, foreach=False)
    target = torch.tensor(labels, dtype=torch.float32)
    first = last = None
    for _ in range(config.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = soft_bce(source_logit + adapter(penultimate), target).mean()
        loss.backward()
        optimizer.step()
        last = float(loss)
        first = last if first is None else first
    return adapter, {"budget": int(budget), "steps": config.adapt_steps, "lr": config.adapt_lr,
                     "first_loss": first, "last_loss": last,
                     "labels_sha256": fb.digest([float(x) for x in labels]),
                     "selection": selection_report(sequence, budget)}


def publish_adapter(directory, rid, budget, adapter, *, bindings, record, protected=()):
    """Freeze one adapter (plain tensors, weights first, header last). Identical existing: adopted."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = fst._clone(dict(adapter.state_dict()))
    stem = f"{rid}_n{int(budget)}"
    weights, header_path = directory / f"{stem}.pt", directory / f"{stem}.adapter.json"
    fst.refuse_protected(protected, {"adapter weights": weights, "adapter header": header_path})
    digest = fst.tensor_state_digest(state)
    if os.path.lexists(weights):
        require(fst._load_payload_digest(weights) == digest,
                f"{weights} already exists with different content; it is never overwritten. "
                f"{FRESH_OUTPUT}")
    else:
        fst._publish(weights, lambda handle: torch.save(state, handle), text=False)
    body = {"schema": ADAPTER_SCHEMA, "run_id": rid, "budget": int(budget), "bindings": bindings,
            "state_sha256": digest, "weights_file": weights.name,
            "weights_file_sha256": file_sha256(weights), "record": record}
    header = {**body, "header_sha256": fb.digest(body)}
    if os.path.lexists(header_path):
        existing = fb.read_json(header_path)
        require(existing["state_sha256"] == digest and existing["bindings"] == bindings,
                f"{header_path} already freezes a different adapter; it is never replaced. "
                f"{FRESH_OUTPUT}")
        return existing
    fst.write_json_owned(header_path, header, protected=protected)
    return header


def load_adapter(directory, rid, budget, config, input_dim, *, bindings=None):
    directory = Path(directory)
    header = fb.read_json(directory / f"{rid}_n{int(budget)}.adapter.json")
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    require(header.get("schema") == ADAPTER_SCHEMA and fb.digest(body) == header["header_sha256"],
            f"the adapter header of {rid} N{budget} is not intact")
    require(bindings is None or header["bindings"] == bindings,
            f"the adapter of {rid} N{budget} is bound to other inputs than this run's")
    weights = directory / header["weights_file"]
    require(weights.name == f"{rid}_n{int(budget)}.pt" and weights.is_file(),
            f"the adapter weights file of {rid} N{budget} is missing")
    require(file_sha256(weights) == header["weights_file_sha256"],
            f"the adapter weights of {rid} N{budget} do not match their recorded SHA-256")
    state = _load_plain(weights)
    require(fst.tensor_state_digest(state) == header["state_sha256"],
            f"the adapter tensors of {rid} N{budget} do not match their recorded digest")
    adapter = make_adapter(config, header.get("record", {}).get("seed", 0), input_dim)
    fst._validate_net_state(state, dict(adapter.state_dict()), "adapter")
    adapter.load_state_dict(fst._clone(state))
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def transfer_eval(model, adapter, oracle, config):
    """Mean errors on the transfer-EVALUATION contexts: the new composition and, as the source-function
    degradation panel, the 48 source functions on the same contexts. ``adapter=None`` is N0, the
    unchanged frozen source: there is no fabricated fit with no data."""
    net, registry = model.net, fe.function_registry()
    split = fe.SPLIT_TRANSFER_EVAL
    cs = oracle.context_set(split)
    blocks = encode_arrays(net, cs.arrays)
    out = {}
    for panel, role in (("new_composition", fe.ROLE_TRANSFER), ("source_function", fe.ROLE_SOURCE)):
        functions = [f.index for f in registry.role(role)]
        logits, penultimate = cross_predict(net, blocks, program_states(net, functions),
                                            penultimate=True)
        if adapter is not None:
            with torch.no_grad():
                logits = logits + adapter(torch.from_numpy(penultimate)).double().numpy()
        exact = oracle.matrix(split, functions)
        metrics = case_metrics(exact.reshape(-1), sigmoid64(logits).reshape(-1))
        out[panel] = {name: float(values.mean()) for name, values in metrics.items()}
        out[panel]["cases"] = int(exact.size)
    return out


# ---------------------------------------------------------------------------
# the Build A bridge: learned query-conditioned penultimate activations as a feature cache
# ---------------------------------------------------------------------------

BRIDGE_ASSAY = "synthetic-exact-bayesian-teacher"
FEATURE_FUSED, FEATURE_CONCAT = "fused_penultimate", "concat_diagnostic"


def bridge_rows(oracle, *, open_eval_labels=False):
    """The Build A roster: every transfer-fit context (split ``train``) and transfer-eval context
    (split ``eval``) crossed with the 12 transfer functions. Each candidate id and complete sequence bind
    BOTH the labelled evidence context and the canonical program; the group is the evidence context, so
    the 12 queries of a context share it. Teacher labels are attached to train rows; evaluation labels
    stay ``None`` until ``open_eval_labels``."""
    registry = fe.function_registry()
    functions = list(registry.role(fe.ROLE_TRANSFER))
    rows = []
    for split_name, bridge_split in ((fe.SPLIT_TRANSFER_FIT, fb.SPLIT_TRAIN),
                                     (fe.SPLIT_TRANSFER_EVAL, fb.SPLIT_EVAL)):
        cs = oracle.context_set(split_name) if (bridge_split == fb.SPLIT_TRAIN
                                                or open_eval_labels) else None
        centers = fe.q7_world().centers_by_split[split_name]
        labels = (oracle.matrix(split_name, [f.index for f in functions])
                  if cs is not None else None)
        for c, signature in enumerate(centers):
            for position, function in enumerate(functions):
                rows.append({
                    "id": fe.candidate_id(signature, function),
                    "sequence": fe.identity_sequence(signature, function.program),
                    "group": fe.group_id(signature), "split": bridge_split,
                    "split_name": split_name, "context": c, "function": function.index,
                    "position": position, "signature": tuple(signature),
                    "label": None if labels is None else float(labels[c, position])})
    rows.sort(key=lambda row: row["id"])
    require(len({row["id"] for row in rows}) == len(rows)
            and len({row["sequence"] for row in rows}) == len(rows),
            "candidate ids and complete identities must be unique")
    return rows


def bridge_event_rows(rows, oracle):
    """The raw padded event arrays (``fact``, ``bit``, ``reliability``, ``mask``) of every roster row, in
    roster order: what the learned model is actually shown for that candidate. RAW EVIDENCE ONLY: no
    posterior, answer or label is computed, so a held-out split is not opened by asking for them. The
    row-to-context association is checked against each row's labelled signature."""
    raw = {name: oracle.raw_context_set(name) for name in (fe.SPLIT_TRANSFER_FIT, fe.SPLIT_TRANSFER_EVAL)}
    for row in rows:
        require(tuple(raw[row["split_name"]].signatures[row["context"]]) == tuple(row["signature"]),
                f"roster row {row['id']} does not name the context its index points at")
    return {name: np.stack([np.asarray(raw[row["split_name"]].arrays[name][row["context"]])
                            for row in rows]) for name in EVENT_FIELDS}


def bridge_event_binding(rows, events, config):
    """What a cache records about the raw evidence its vectors were computed from: the selected variant,
    the actual dtype, shape and content digest of each roster-ordered event array, a digest of all of
    them together and the row association (candidate ids and the (split, context) each row reads). The
    training bindings do not contain ``eval_variant``; this does."""
    return {
        "variant": config.eval_variant, "cancel_max": config.cancel_max, "null_max": config.null_max,
        "event_width": config.event_width, "rows": len(rows),
        "row_ids_sha256": fb.digest([row["id"] for row in rows]),
        "row_context_sha256": fb.digest([[row["split_name"], int(row["context"])] for row in rows]),
        "event_arrays": {name: {"dtype": events[name].dtype.str, "shape": list(events[name].shape),
                                "sha256": _array_sha(events[name])} for name in EVENT_FIELDS},
        "events_sha256": _digest_arrays([(name, events[name]) for name in EVENT_FIELDS]),
        "teacher_quantities": "none: raw events only; no posterior or answer was computed to build them"}


def bridge_features(model, rows, oracle, *, kind=FEATURE_FUSED):
    """Learned features ``[n_rows, dim]`` of the roster. ``fused_penultimate`` is the actual learned
    query-conditioned penultimate activation (after the final GELU, before the Bernoulli projection):
    it can carry interactions the joint head learned. ``concat_diagnostic`` (evidence blocks next to the
    program state) makes a linear readout additive in those inputs and is a DIAGNOSTIC ONLY, never an
    arm. Only the RAW evidence of the roster's contexts is read (``raw_context_set``): extracting a
    feature never materializes the teacher's posterior of any split, held-out ones included."""
    require(kind in (FEATURE_FUSED, FEATURE_CONCAT), f"unknown bridge feature kind {kind!r}")
    net, registry = model.net, fe.function_registry()
    functions = [f.index for f in registry.role(fe.ROLE_TRANSFER)]
    states = program_states(net, functions)
    blocks_by_split, fused_by_split = {}, {}
    for split_name in (fe.SPLIT_TRANSFER_FIT, fe.SPLIT_TRANSFER_EVAL):
        blocks = encode_arrays(net, oracle.raw_context_set(split_name).arrays)
        blocks_by_split[split_name] = blocks
        if kind == FEATURE_FUSED:
            fused_by_split[split_name] = cross_predict(net, blocks, states, penultimate=True)[1]
    out = []
    for row in rows:
        split_name, c, p = row["split_name"], row["context"], row["position"]
        if kind == FEATURE_FUSED:
            out.append(fused_by_split[split_name][c, p].astype(np.float64))
        else:
            out.append(np.concatenate([blocks_by_split[split_name][c].reshape(-1).numpy(),
                                       states[p].numpy()]).astype(np.float64))
    return np.stack(out)


def bridge_supervision(header):
    """The documented supervision history of a learned representation (what the extractor was trained
    on). A scratch control is label-free: a random initialisation that never saw a label."""
    if header["arm"] == SCRATCH:
        return fb.SupervisionHistory(status=fb.HISTORY_LABEL_FREE,
                                     description="random initialisation, never trained",
                                     exposure_status="none")
    cost, arm, target = header["cost"], header["arm"], "functional_causal_learning synthetic world"
    answers = cost["canonical_answers"]
    # ``label_count`` is the number of DISTINCT canonical answers of that kind (see ``cost_report``): a
    # raw variant of one context or a donor that gives one hybrid is not a second label
    entries = [{"assay": "synthetic exact Bayesian teacher posterior probability of a program "
                         "(source q7 contexts x 48 source functions); distinct (context signature, "
                         "program) answers",
                "target": target, "label_count": int(answers["factual"]), "label_ids": None}]
    if arm == "M":
        entries.append({"assay": "synthetic exact per-fact posterior marginals of the base context "
                                 "(soft block-marginal targets); distinct (context signature, fact) "
                                 "answers", "target": target,
                        "label_count": int(answers["marginal"]), "label_ids": None})
    elif arm in ("A", "I"):
        entries.append({"assay": "synthetic exact counterfactual teacher answers on raw-evidence "
                                 "hybrids; distinct (hybrid signature, program) answers",
                        "target": target, "label_count": int(answers["counterfactual"]),
                        "label_ids": None})
    elif arm == "Ic":
        entries.append({"assay": "CORRUPTED counterfactual targets (values randomly permuted within "
                                 "function-orbit x site strata) on distinct (hybrid signature, "
                                 "program) questions: the targets are not those questions' answers",
                        "target": target, "label_count": int(answers["counterfactual"]),
                        "label_ids": None})
    return fb.SupervisionHistory(
        status=fb.HISTORY_DOCUMENTED, documented=tuple(entries), exposure_status="none",
        description=("source-trained on synthetic exact teacher targets; the held-out contexts and the "
                     "transfer functions were never labelled for training"),
        exposure_description="no pretraining; random initialisation then source training")


def _bridge_cache_args(rows, features, model_header, rid, kind, config, events_binding):
    count, dim = len(rows), int(features.shape[1])
    names = ([f"z{i}" for i in range(dim)] if kind == FEATURE_FUSED
             else [f"block{k}_{j}" for k in range(N_BLOCKS) for j in range(config.block_dim)]
             + [f"program{i}" for i in range(config.program_hidden)])
    activation_location = (
        "learned query-conditioned penultimate activation of the prediction head (after the final "
        "GELU, before the Bernoulli projection)" if kind == FEATURE_FUSED else
        "DIAGNOSTIC ONLY: the four hidden evidence blocks concatenated with the program-encoder "
        "state (a linear readout is additive in these inputs)")
    space = fb.FeatureSpace(
        kind=f"learned_{kind}", names=tuple(names),
        schema={"construction": activation_location, "frozen_state_sha256": model_header["state_sha256"],
                "arm": model_header["arm"], "source_seed": model_header["seed"],
                "bindings": model_header["bindings"],
                "raw_observation_variant": {
                    **copy.deepcopy(events_binding),
                    "note": ("every candidate's evidence is the raw-event record of this ONE selected "
                             "variant of its labelled context; other variants are other records of "
                             "the same signature. The dtype, shape and content digests are those of "
                             "the actual roster-ordered event arrays the vectors were computed from, "
                             "and the row association names which candidates they belong to")},
                "identity_convention": fe.IDENTITY_CONVENTION,
                "teacher_quantities_in_features": ("none: no posterior, truth table, trace or logit is "
                                                   "a feature column; the posterior is the label")},
        history=(f"frozen {model_header['arm']} source model, seed {model_header['seed']}",
                 activation_location),
        supervision=bridge_supervision(model_header))
    extractor = {"kind": "synthetic",
                 "checkpoint_sha256": model_header["weights_file_sha256"],
                 "source_sha256": code_sha256(),
                 "description": (f"{rid}: a frozen functional_causal_learning source model trained "
                                 "in a synthetic finite world (not a native checkpoint); "
                                 "checkpoint_sha256 is the frozen weights FILE hash and "
                                 "source_sha256 digests the code")}
    activation = {"location": activation_location,
                  "pooling": "none (one vector per candidate = evidence context + program)",
                  "token_map": ("raw observation events (fact, bit, reliability) and prefix program "
                                "tokens; the candidate identity string is NOT an input")}
    conditioning = {"candidate_masking": "none", "answer_free": True, "family": {"recorded": False},
                    "assay": {"recorded": True, "values": [BRIDGE_ASSAY] * count},
                    "quality": {"recorded": True, "values": [None] * count}}
    identity = {"kind": fm.IDENTITY_COMPLETE_SEQUENCE, "alphabet": fe.IDENTITY_ALPHABET,
                "length": fe.identity_length(), "construct_description": fe.IDENTITY_CONVENTION}
    return dict(representation_id=rid, ids=[r["id"] for r in rows],
                sequences=[r["sequence"] for r in rows], groups=[r["group"] for r in rows],
                splits=[r["split"] for r in rows], features=features, space=space,
                identity=identity, extractor=extractor, activation=activation,
                conditioning=conditioning, dtype="float64")


def expected_bridge_cache_sha256(rows, features, model_header, rid, kind, config, events_binding):
    """The canonical (row-order-invariant) ``cache_sha256`` the cache writer produces for exactly these
    arguments: the digest of the roster rows, identity, feature space (schema, history and supervision),
    extractor, activation, conditioning and the stored features together. It is obtained by letting the
    writer build the cache in a scratch directory this call creates and removes, so the comparison with an
    existing cache is the writer's own full contract and nothing is written near the output."""
    with tempfile.TemporaryDirectory(prefix="bridge_cache_contract_") as scratch:
        written = fm.write_feature_cache(
            Path(scratch) / f"{rid}.cache.json", Path(scratch) / f"{rid}.cache.npz",
            **_bridge_cache_args(rows, features, model_header, rid, kind, config, events_binding))
    return written["cache_sha256"]


def check_bridge_cache(directory, rid, expected_sha256):
    """``None`` when no cache of ``rid`` exists, else the existing cache's facts after proving it is
    EXACTLY the expected one: the pair loads (file digest, header, features all intact) and its canonical
    digest equals the expected contract's, so a numerically equal cache with other rows, another
    representation, learned state, activation, conditioning, supervision history or raw evidence refuses.
    Reads only."""
    directory = Path(directory)
    header, npz = directory / f"{rid}.cache.json", directory / f"{rid}.cache.npz"
    if not (os.path.lexists(header) or os.path.lexists(npz)):
        return None
    cache = fm.load_feature_cache(header, npz_path=npz)
    require(cache.representation_id == rid and cache.cache_sha256 == expected_sha256,
            f"{header} exists and is not the cache this export computes: its rows, representation, "
            "learned state, activation, conditioning, supervision history, raw evidence or features "
            f"differ (canonical digest {cache.cache_sha256} versus the expected {expected_sha256}); it "
            f"is never overwritten. {FRESH_OUTPUT}")
    return {"header": str(header), "npz": str(npz), "cache_sha256": cache.cache_sha256,
            "file_sha256": cache.raw["file_sha256"], "adopted": True}


def write_bridge_cache(directory, rows, features, model_header, rid, kind, config, events_binding, *,
                       protected=()):
    """Write one feature cache, or ADOPT the existing one untouched when its full semantic contract equals
    the expected one (:func:`check_bridge_cache`); anything else is refused and never overwritten.
    Destinations are checked against ``protected`` inputs first."""
    directory = Path(directory)
    header, npz = directory / f"{rid}.cache.json", directory / f"{rid}.cache.npz"
    fst.refuse_protected(protected, {f"cache header {rid}": header, f"cache NPZ {rid}": npz})
    existing = check_bridge_cache(directory, rid, expected_bridge_cache_sha256(
        rows, features, model_header, rid, kind, config, events_binding))
    if existing is not None:
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    written = fm.write_feature_cache(
        header, npz, **_bridge_cache_args(rows, features, model_header, rid, kind, config,
                                          events_binding))
    return {**written, "adopted": False}


def bridge_manifest(config):
    return {
        "schema": fm.MANIFEST_SCHEMA, "target": "synthetic four-fact Boolean world "
                                                "(functional_causal_learning)",
        "assay": "exact Bayesian posterior probability that a typed program is true (synthetic teacher)",
        "utility": {"name": "posterior truth probability", "units": "probability",
                    "scale": "linear", "higher_is_better": True, "useful_threshold": 0.5},
        "grouping_rationale": ("groups are labelled evidence contexts: the 12 queries of one context "
                               "share a group, whole contexts are held out (q7 center strata "
                               "transfer-fit versus transfer-eval), and this holds out labelled "
                               "contexts, not evidence permutation orbits"),
        "candidates": {"identity_kind": fm.IDENTITY_COMPLETE_SEQUENCE,
                       "alphabet": fe.IDENTITY_ALPHABET, "length": fe.identity_length()},
        "prior": {"kind": fb.PRIOR_UNIFORM,
                  "semantics": "every roster candidate has the same weight",
                  "provenance": {"status": "label_free",
                                 "description": "uniform over the synthetic roster",
                                 "sequence_only_exposure": {"status": "none"}}},
        "notes": ("synthetic; the utility is a statement's posterior truth probability, so a policy "
                  "over it selects high-probability statements and is NOT a reasoning-accuracy "
                  "measure. The bridge metric is held-out readout prediction error on context + "
                  "program pairs.")}


def bridge_plan(config, representations, reference):
    return {
        "schema": fm.PLAN_SCHEMA, "representations": sorted(representations),
        "maps": [{"id": "linear", "kind": "linear", "center": True, "scale": "std",
                  "trace_normalize": True}],
        "label_budgets": list(config.export_label_budgets),
        "seeds": list(config.export_acquisition_seeds),
        "baseline": {"tau": config.export_tau, "ridge_alpha": config.export_ridge_alpha,
                     "sir_proposal_draws": 256, "final_draws": 32, "dpo_pressures": [0.5, 1.0, 2.0]},
        "reference": {"representation": reference, "map": "linear"},
        "bootstrap": {"replicates": config.export_bootstrap_replicates, "seed": 0,
                      "confidence": 0.9, "min_groups": 3},
        "notes": "synthetic bridge plan; every setting is fixed in advance and nothing is tuned"}


BRIDGE_INPUT_FILES = {"candidates": "candidates.csv", "train_outcomes": "train_outcomes.csv",
                      "manifest": "matrix_manifest.json", "plan": "matrix_plan.json"}
EVAL_OUTCOMES_NAME, EVAL_PENDING_NAME = "eval_outcomes.csv", "eval_outcomes.pending"


def bridge_input_texts(rows, config, representations, reference):
    """The exact text of the four bridge input files (candidate table, TRAINING outcomes, manifest, plan),
    by role. The evaluation outcomes are not among them."""
    require(all(r["label"] is not None for r in rows if r["split"] == fb.SPLIT_TRAIN),
            "every training candidate needs its teacher label")
    return {
        "candidates": "candidate_id,sequence,group_id,split,prior_log_weight\n" + "".join(
            f"{r['id']},{r['sequence']},{r['group']},{r['split']},0.0\n" for r in rows),
        "train_outcomes": "candidate_id,utility\n" + "".join(
            f"{r['id']},{r['label']!r}\n" for r in rows if r["split"] == fb.SPLIT_TRAIN),
        "manifest": fst._json_text(bridge_manifest(config)),
        "plan": fst._json_text(bridge_plan(config, representations, reference))}


def _check_exact_text(path, text):
    """Refuse, WITHOUT writing, an existing file whose text is not exactly ``text``; ``True`` when it is
    (it is then adopted untouched) and ``False`` when nothing is there yet."""
    path = Path(path)
    if not os.path.lexists(path):
        return False
    require(path.is_file() and path.read_text(encoding="utf-8") == text,
            f"{path} exists and differs from what this stage computed; it is never replaced. "
            f"{FRESH_OUTPUT}")
    return True


def _publish_text_exact(path, text, *, protected=()):
    """Publish ``text`` at ``path`` only when nothing is there: an existing file with exactly this text is
    adopted untouched, a different one (or a protected path) refuses and is never replaced. Returns whether
    the file was written."""
    path = Path(path)
    fst.refuse_protected(protected, {f"output {path.name}": path})
    if _check_exact_text(path, text):
        return False
    fst.write_text_owned(path, text, protected=protected)
    return True


def write_bridge_inputs(directory, rows, config, representations, reference, *, protected=()):
    """The candidate table, the TRAINING outcomes (teacher probabilities of transfer-fit candidates), the
    manifest and the plan. The evaluation outcomes are NOT written here. An existing file with exactly the
    expected text is adopted untouched; a different one is refused, never replaced, and a destination that
    resolves to a protected input (a direct path, a case alias or a hard link) is refused first, before any
    of the four is written."""
    directory = Path(directory)
    texts = bridge_input_texts(rows, config, representations, reference)
    paths = {role: directory / name for role, name in BRIDGE_INPUT_FILES.items()}
    fst.refuse_protected(protected, {f"bridge input {role}": path for role, path in paths.items()})
    for role, path in paths.items():
        _check_exact_text(path, texts[role])
    directory.mkdir(parents=True, exist_ok=True)
    for role, path in paths.items():
        _publish_text_exact(path, texts[role], protected=protected)
    return paths


def eval_outcome_values(rows, oracle):
    """``{candidate id: teacher probability}`` of the transfer-EVAL candidates, in roster order. This
    materializes the held-out teacher answers (it opens the split): call it only after the matrix freeze
    is verified."""
    labels = oracle.matrix(fe.SPLIT_TRANSFER_EVAL,
                           [f.index for f in fe.function_registry().role(fe.ROLE_TRANSFER)])
    return {r["id"]: float(labels[r["context"], r["position"]])
            for r in rows if r["split"] == fb.SPLIT_EVAL}


def eval_outcomes_text(rows, oracle):
    """The evaluation outcomes CSV text: the teacher probabilities of the transfer-EVAL candidates. Built
    only after the matrix freeze is verified."""
    values = eval_outcome_values(rows, oracle)
    return "candidate_id,utility\n" + "".join(f"{ident},{value!r}\n" for ident, value in values.items())


def expected_common_sha256(texts):
    """The ``common_sha256`` a matrix freeze built from exactly these input texts records: the manifest,
    the candidate roster (ids, sequences, groups, splits, prior), the offered TRAINING labels (ids and
    utilities) and the held-out support. The texts are parsed by the Build A readers from a scratch
    directory this call creates and removes."""
    with tempfile.TemporaryDirectory(prefix="bridge_inputs_contract_") as scratch:
        paths = {}
        for role, name in BRIDGE_INPUT_FILES.items():
            paths[role] = Path(scratch) / name
            paths[role].write_text(texts[role], encoding="utf-8")
        manifest = fm.load_matrix_manifest(paths["manifest"])
        table = fb.read_candidates(paths["candidates"], alphabet=manifest.alphabet,
                                   length=manifest.length)
        roster = fm.make_roster(manifest, table,
                                fb.read_outcomes(paths["train_outcomes"], label="training outcomes"))
        return fb.digest(roster.common_document())


def verify_freeze_binding(verified, *, plan_sha256, cache_sha256, common_sha256):
    """An existing matrix freeze (already verified from disk by ``fm.verify_freeze``) is adopted only if
    its OWN recorded plan, representations (canonical cache digests) and common identity equal what the
    CURRENT inputs produce. The same plan over different learned features, another roster or other
    training labels is a different experiment and refuses."""
    index = verified["index"]
    require(index["plan_sha256"] == plan_sha256,
            "the matrix freeze on disk is a freeze of a different plan; it is not adopted")
    recorded = {rep: summary["cache_sha256"] for rep, summary in index["representations"].items()}
    require(recorded == dict(cache_sha256),
            "the matrix freeze on disk was fitted on other feature caches than the ones this export "
            "computes (its recorded canonical cache digests differ); a freeze of the same plan over "
            f"different learned features is not adopted. {FRESH_OUTPUT}")
    require(index["common_sha256"] == common_sha256,
            "the matrix freeze on disk records another candidate roster, manifest, offered training "
            f"labels or held-out support than the current inputs; it is not adopted. {FRESH_OUTPUT}")
    return True


def verify_matrix_report(results, verified, truth):
    """A stored matrix evaluation report is adopted only if it belongs to THIS freeze and was scored under
    THESE evaluation outcomes: its freeze and plan digests, the canonical digest of the outcomes it was
    scored against equal to the CURRENT offered outcomes (``truth`` maps candidate id to utility), its
    status counts and cell list equal to the freeze's, and, for every successful cell, its own recorded MSE
    equal to the MSE that cell's frozen predictions give under those outcomes. Nothing is stamped onto an
    old result: a report under other truth values on the same freeze refuses."""
    index = verified["index"]
    support = index["common"]["support"]["ids"]
    vector = fb._outcome_vector(support, truth)
    require(results.get("index_sha256") == index["index_sha256"]
            and results.get("plan_sha256") == index["plan_sha256"],
            "the matrix evaluation report on disk is not for this freeze")
    require(results["evaluation_outcomes"]["canonical_sha256"]
            == fb.digest({"ids": list(support), "utilities": vector.tolist()}),
            "the matrix evaluation report on disk was scored under other evaluation outcomes than the "
            f"current offered ones; it is not relabelled. {FRESH_OUTPUT}")
    cells = results["cells"]
    require([cell["cell_id"] for cell in cells] == [key.cell_id for key in verified["keys"]]
            and results["status_counts"] == index["counts"],
            "the matrix evaluation report's cells or status counts differ from the freeze's")
    for cell in cells:
        envelope = verified["envelopes"][cell["cell_id"]]
        require(cell["status"] == envelope["status"],
                f"the report's status of cell {cell['cell_id']} differs from the freeze's")
        if envelope["status"] == fm.STATUS_OK:
            predicted = np.array(envelope["artifact"]["predictions"]["utility"], dtype=np.float64)
            mse = float(((predicted - np.asarray(vector, dtype=np.float64)) ** 2).mean())
            require(math.isclose(float(cell["readout"]["mse"]), mse, rel_tol=1e-9, abs_tol=1e-15),
                    f"the report's recorded MSE of cell {cell['cell_id']} is not the one its frozen "
                    "predictions give under the current evaluation outcomes")
    return True


@dataclass(frozen=True, eq=False)
class ExportExpectation:
    """Everything an export is expected to hold, computed in memory from the frozen models and the
    teacher's TRAINING-side answers only (the roster's transfer-fit labels): the roster, the raw event
    binding, every representation's features and canonical cache digest, the exact input texts, the
    common-identity and plan digests and the reference. ``oracle`` is the one that opened the transfer-fit
    contexts; the held-out answers are opened later, by the caller, and only after a freeze is verified."""

    oracle: object
    rows: list
    events: dict
    binding: dict
    features: dict
    cache_sha256: dict
    texts: dict
    common_sha256: str
    plan: object
    plan_sha256: str
    reference: str
    rids: tuple


def export_expectation(models, config):
    """Build the :class:`ExportExpectation` of a set of frozen models. No held-out teacher quantity is
    materialized: the features read raw events only and the roster labels only the transfer-fit rows."""
    oracle = PanelOracle(config)
    rows = bridge_rows(oracle)
    events = bridge_event_rows(rows, oracle)
    binding = bridge_event_binding(rows, events, config)
    rids = tuple(sorted(models))
    reference = run_id("F", config.seeds[0])
    features = {rid: bridge_features(models[rid], rows, oracle) for rid in rids}
    cache_sha256 = {rid: expected_bridge_cache_sha256(rows, features[rid], models[rid].header, rid,
                                                      FEATURE_FUSED, config, binding) for rid in rids}
    texts = bridge_input_texts(rows, config, list(rids), reference)
    plan = fm.parse_plan(json.loads(fb.canonical_json(bridge_plan(config, list(rids), reference))))
    return ExportExpectation(
        oracle=oracle, rows=rows, events=events, binding=binding, features=features,
        cache_sha256=cache_sha256, texts=texts, common_sha256=expected_common_sha256(texts), plan=plan,
        plan_sha256=plan.fingerprint(), reference=reference, rids=rids)


def bridge_summary(results):
    """Held-out readout prediction error per (arm, source seed, budget, acquisition seed), the bridge
    metric, from a Build A evaluation report."""
    table = {}
    for cell in results["cells"]:
        if cell["status"] != fm.STATUS_OK:
            continue
        rep = cell["key"]["representation"]
        arm, seed = rep.rsplit("_s", 1)
        table[(arm, int(seed), cell["key"]["label_budget"], cell["key"]["seed"])] = (
            cell["readout"]["mse"])
    rows = [{"arm": arm, "source_seed": seed, "label_budget": budget, "acquisition_seed": acq,
             "readout_mse": mse} for (arm, seed, budget, acq), mse in sorted(table.items())]
    return {"cells": rows, "status_counts": results["status_counts"],
            "metric": ("held-out readout MSE on context + program pairs (the bridge metric); the "
                       "policy utility is a posterior truth probability and selects high-probability "
                       "statements, not correct reasoning")}


# ---------------------------------------------------------------------------
# the staged pipeline
# ---------------------------------------------------------------------------

COMPARISONS = (("I", "F"), ("I", "A"), ("I", "Ic"), ("I", "M"), ("A", "F"), ("M", "F"), ("Ic", "F"),
               ("A", "Ic"))
STATE_NAME = "stage_state.json"


def paired_differences(headlines, arms, seeds):
    """Paired SEED differences ``arm_a - arm_b`` of every headline scalar (per seed, then their mean).
    No p-value is computed and nothing is selected: three paired seeds are a finite-panel screen."""
    out = {}
    for left, right in COMPARISONS:
        if left not in arms or right not in arms:
            continue
        entry = {}
        for metric in sorted(next(iter(headlines.values()))):
            per_seed = []
            for seed in seeds:
                a = headlines[run_id(left, seed)][metric]
                b = headlines[run_id(right, seed)][metric]
                if a is None or b is None:
                    per_seed = None
                    break
                per_seed.append(float(a) - float(b))
            entry[metric] = (None if per_seed is None else
                             {"per_seed": per_seed, "mean": float(np.mean(per_seed))})
        out[f"{left}-{right}"] = entry
    return out


class CausalRun:
    """The staged, resumable Build B pipeline in one output directory.

    Stages (see the module docstring): ``prepare``, ``train``, ``freeze``, ``evaluate``, ``transfer``,
    ``export``, ``report``. A stage that is already complete is VERIFIED against its recorded digests
    and adopted, never recomputed or overwritten; a partial or foreign state is refused. The held-out
    labels are opened only by ``evaluate`` and ``transfer`` (after every source state is frozen) and the
    transfer-evaluation labels only after every adapter is frozen. Existing files are never deleted."""

    def __init__(self, output_dir, config, *, protected=()):
        require(isinstance(config, CausalConfig), "CausalRun needs a CausalConfig")
        self.out, self.config, self.protected = Path(output_dir), config, protected
        self._context = None
        self.state = None

    # --- opening and state -----------------------------------------------------------------
    @classmethod
    def open(cls, output_dir, config, *, protected=()):
        run = cls(output_dir, config, protected=protected)
        run.out.mkdir(parents=True, exist_ok=True)
        path = run.out / STATE_NAME
        identity = {"config_sha256": fb.digest(config.identity_document()),
                    "world_sha256": fe.world_sha256(), "code_sha256": code_sha256()}
        if path.exists():
            state = fb.read_json(path)
            require(state.get("schema") == STATE_SCHEMA, f"{path} is not a {STATE_SCHEMA} file")
            for key, value in identity.items():
                require(state.get(key) == value,
                        f"{path} was written for a different {key.split('_')[0]} ({key} differs): "
                        f"{FRESH_OUTPUT}")
            run.state = state
        else:
            run.state = {"schema": STATE_SCHEMA, **identity, "stages": {}}
            run._save()
        return run

    def _save(self):
        fst.write_json_owned(self.out / STATE_NAME, self.state, protected=self.protected)

    def _complete(self, stage):
        return bool(self.state["stages"].get(stage, {}).get("complete"))

    def _require(self, stage):
        require(self._complete(stage), f"the {stage!r} stage is not complete: run it first")

    def _mark(self, stage, record):
        self.state["stages"][stage] = {"complete": True, **record}
        self._save()
        return self.state["stages"][stage]

    def _check_artifact(self, relative, document):
        """Refuse, WITHOUT writing anything, a stage artifact destination that is protected or that holds
        a different document. ``True`` when an identical one exists (it is adopted, never rewritten)."""
        path = self.out / relative
        fst.refuse_protected(self.protected, {f"artifact {relative}": path})
        if not os.path.lexists(path):
            return False
        require(fb.read_json(path) == json.loads(fb.canonical_json(document)),
                f"{path} exists and differs from what this stage computed; it is never replaced. "
                f"{FRESH_OUTPUT}")
        return True

    def _write_artifact(self, relative, document):
        """Publish a stage artifact; an identical existing one is adopted, a different one refused."""
        path = self.out / relative
        if self._check_artifact(relative, document):
            return path
        return fst.write_json_owned(path, document, protected=self.protected)

    def _read_artifact(self, relative, expected_sha256):
        document = fb.read_json(self.out / relative)
        require(fb.digest(document) == expected_sha256,
                f"{relative} does not match the digest the stage state recorded")
        return document

    # --- the training context ---------------------------------------------------------------
    def context(self):
        if self._context is None:
            self._require("prepare")
            bank, events = build_bank(self.config)
            record = self.state["stages"]["prepare"]
            tapes = {seed: load_tape(self.out / "tapes", seed, self.config, bank_sha256=bank.sha256)
                     for seed in self.config.seeds}
            inits = {seed: load_init(self.out / "init", seed, self.config.model_spec())
                     for seed in self.config.seeds}
            for seed in self.config.seeds:
                require(tapes[seed].sha256 == record["tape_sha256"][str(seed)]
                        and fst.tensor_state_digest(inits[seed]) == record["init_sha256"][str(seed)],
                        f"the stored tape or initial state of seed {seed} is not the one the "
                        "prepare stage recorded")
                require(file_sha256(self.out / "tapes" / f"seed_{seed}.npz")
                        == record["tape_file_sha256"][str(seed)],
                        f"the stored tape arrays of seed {seed} are not the bytes the prepare stage "
                        "published")
            self._context = TrainContext(self.config, bank, events, tapes, inits)
        return self._context

    def preflight(self):
        """Verify, in memory, every shared input a stage is about to consume."""
        ctx = self.context()
        for seed in self.config.seeds:
            ctx.verify_live(seed)
        return ctx

    # --- stages -----------------------------------------------------------------------------
    def prepare(self, *, deep_verify=False):
        """Verify the frozen world (q7 counts, every hybrid, the function families), build or load the
        per-seed tapes and shared initial states. Only SOURCE signatures are ever labelled here."""
        world_summary = fe.verify_world()
        registry = fe.function_registry()
        bank, events = build_bank(self.config)
        tapes, inits = {}, {}
        for seed in self.config.seeds:
            if (self.out / "tapes" / f"seed_{seed}.json").exists():
                tapes[seed] = load_tape(self.out / "tapes", seed, self.config,
                                        bank_sha256=bank.sha256)
            else:
                tapes[seed] = build_tape(self.config, seed, bank, events)
            if (self.out / "init" / f"seed_{seed}.json").exists():
                inits[seed] = load_init(self.out / "init", seed, self.config.model_spec())
            else:
                inits[seed] = initial_state(self.config.model_spec(), seed)
        context = TrainContext(self.config, bank, events, tapes, inits)
        if deep_verify:
            for seed, tape in tapes.items():
                verify_tape(tape, bank, events, self.config, deep=True)
        # ---- every refusal is decided BEFORE the first byte is published: a leftover archive, header,
        # weights file or artifact of another run is neither replaced nor given a header that claims it,
        # whichever seed it belongs to
        for seed in self.config.seeds:
            preflight_tape(tapes[seed], self.out / "tapes", protected=self.protected)
            preflight_init(inits[seed], seed, self.out / "init", self.config.model_spec(),
                           protected=self.protected)
        self._check_artifact("config.json", self.config.document())
        self._check_artifact("world.json", fe.world_document())
        published = {}
        for seed in self.config.seeds:
            published[seed] = save_tape(tapes[seed], self.out / "tapes", protected=self.protected)
            save_init(inits[seed], seed, self.out / "init", self.config.model_spec(),
                      protected=self.protected)
        self._write_artifact("config.json", self.config.document())
        self._write_artifact("world.json", fe.world_document())
        self._context = context
        return self._mark("prepare", {
            "world": world_summary,
            "functions": {role: len(registry.role(role)) for role in fe.FUNCTION_ROLES},
            "orbits": {role: len(registry.orbits(role)) for role in fe.FUNCTION_ROLES},
            "bank_sha256": bank.sha256,
            "tape_sha256": {str(s): tapes[s].sha256 for s in self.config.seeds},
            "tape_file_sha256": {str(s): published[s]["npz_file_sha256"] for s in self.config.seeds},
            "init_sha256": {str(s): context.init_sha256[s] for s in self.config.seeds},
            "corruption": {str(s): {k: v for k, v in tapes[s].report["corruption"].items()
                                    if k != "stratum_records"} for s in self.config.seeds},
            "teacher_cost": {str(s): tapes[s].report["teacher_cost"] for s in self.config.seeds},
            "deep_verified": bool(deep_verify)})

    def train(self, *, stop_after=None):
        """Train every (arm, seed) with checkpoints at update boundaries; resumable."""
        self._require("prepare")
        ctx = self.preflight()
        runs = {}
        for seed in self.config.seeds:
            for arm in self.config.arms:
                runs[run_id(arm, seed)] = train_arm(ctx, arm, seed,
                                                    self.out / "runs" / run_id(arm, seed),
                                                    stop_after=stop_after, protected=self.protected)
        done = all(run.cursor == run.total_steps for run in runs.values())
        record = {rid: {"cursor": run.cursor, "state_sha256": run.state_sha256()}
                  for rid, run in runs.items()}
        if done:
            self._mark("train", {"runs": record})
        else:
            self.state["stages"]["train"] = {"complete": False, "runs": record}
            self._save()
        return runs

    def freeze(self):
        """Publish and hash every source state (and the untrained scratch control) BEFORE any held-out
        panel is opened."""
        self._require("train")
        ctx = self.preflight()
        directory = self.out / "frozen"
        frozen = {}
        for seed in self.config.seeds:
            for arm in self.config.arms:
                run = train_arm(ctx, arm, seed, self.out / "runs" / run_id(arm, seed),
                                protected=self.protected)
                require(run.state_sha256() == self.state["stages"]["train"]["runs"][
                    run.run_id]["state_sha256"],
                        f"{run.run_id}'s final state is not the one the train stage recorded")
                frozen[run.run_id] = freeze_run(run, directory, protected=self.protected)
            frozen[run_id(SCRATCH, seed)] = freeze_scratch(ctx, seed, directory,
                                                           protected=self.protected)
        return self._mark("freeze", {"frozen": {
            rid: {"state_sha256": h["state_sha256"], "header_sha256": h["header_sha256"],
                  "weights_file_sha256": h["weights_file_sha256"]} for rid, h in frozen.items()}})

    def frozen_models(self):
        """Every frozen source model, loaded from its published files and VERIFIED against the freeze
        record (a state that changed after freezing refuses)."""
        self._require("freeze")
        ctx = self.preflight()
        models = {}
        for rid, recorded in self.state["stages"]["freeze"]["frozen"].items():
            seed = int(rid.rsplit("_s", 1)[1])
            model = load_frozen(self.out / "frozen", rid, self.config.model_spec(),
                                bindings=ctx.bindings(seed))
            require(model.state_sha256 == recorded["state_sha256"]
                    and model.header["header_sha256"] == recorded["header_sha256"],
                    f"the frozen state of {rid} is not the one the freeze stage recorded")
            models[rid] = model
        return models

    def evaluate(self):
        """Open the held-out panels (primary, validation) for the first time, on frozen models."""
        self._require("freeze")
        models = self.frozen_models()
        ctx = self.context()
        if self._complete("evaluate"):
            record = self.state["stages"]["evaluate"]
            return self._read_artifact("evaluation/evaluation.json", record["evaluation_sha256"])
        oracle = PanelOracle(self.config)
        evaluations = {rid: evaluate_model(model, oracle, self.config, ctx.bank)
                       for rid, model in models.items()}
        for model in models.values():
            model.verify_unchanged()
        document = {
            "schema": REPORT_SCHEMA + "/evaluation", "claims": CLAIMS,
            "config_sha256": self.state["config_sha256"],
            "frozen_before_evaluation": {rid: m.state_sha256 for rid, m in models.items()},
            "opened_splits": list(oracle.opened), "models": evaluations,
            "headlines": {rid: ev["headline"] for rid, ev in evaluations.items()},
            "paired_seed_differences": paired_differences(
                {rid: ev["headline"] for rid, ev in evaluations.items()
                 if ev["arm"] != SCRATCH}, self.config.arms, self.config.seeds)}
        self._write_artifact("evaluation/evaluation.json", document)
        self._mark("evaluate", {"evaluation_sha256": fb.digest(document),
                                "opened_splits": list(oracle.opened)})
        return document

    def transfer(self):
        """Adapt each frozen arm and the scratch control on the SAME nested new-composition budgets,
        FREEZE every adapter, and only then open the transfer-evaluation outcomes."""
        self._require("freeze")
        models = self.frozen_models()
        if self._complete("transfer"):
            document = self._read_artifact("transfer/transfer.json",
                                           self.state["stages"]["transfer"]["transfer_sha256"])
            self._verify_transfer_artifacts(models, document)
            return document
        cfg, directory = self.config, self.out / "transfer" / "adapters"
        oracle = PanelOracle(cfg)
        sequence = transfer_selection(cfg)
        budgets = [n for n in cfg.transfer_budgets if n > 0]
        selection_sha = fb.digest([[int(c), int(f)] for c, f in sequence])
        records, frozen = {}, {}
        for rid, model in models.items():
            for budget in budgets:
                adapter, record = adapt_adapter(model, oracle, cfg, sequence, budget)
                bindings = {"source_state_sha256": model.state_sha256, "selection_sha256": selection_sha,
                            "config_sha256": self.state["config_sha256"],
                            "adapter": {"hidden": cfg.adapter_hidden, "steps": cfg.adapt_steps,
                                        "lr": cfg.adapt_lr}}
                header = publish_adapter(directory, rid, budget, adapter, bindings=bindings,
                                         record={**record, "seed": model.seed}, protected=self.protected)
                frozen[f"{rid}|{budget}"] = {"state_sha256": header["state_sha256"],
                                             "header_sha256": header["header_sha256"]}
                records[f"{rid}|{budget}"] = {**record, "bindings": bindings}
        opened_before = list(oracle.opened)
        self.state["stages"]["transfer"] = {"complete": False, "adapters_frozen": frozen,
                                            "opened_before_adapter_freeze": opened_before}
        self._save()
        # ---- the frozen boundary: every adapter is published and recorded; only now are the
        # transfer-EVALUATION outcomes opened, and the adapters are re-read from their files ----
        results, input_dim = {}, cfg.head_hidden
        for rid, model in models.items():
            entry = {"N0": transfer_eval(model, None, oracle, cfg)}
            for budget in budgets:
                bindings = records[f"{rid}|{budget}"]["bindings"]
                adapter = load_adapter(directory, rid, budget, cfg, input_dim, bindings=bindings)
                require(fb.read_json(directory / f"{rid}_n{budget}.adapter.json")["header_sha256"]
                        == frozen[f"{rid}|{budget}"]["header_sha256"],
                        f"the adapter of {rid} N{budget} changed after it was frozen")
                entry[f"N{budget}"] = transfer_eval(model, adapter, oracle, cfg)
            results[rid] = entry
        for model in models.values():
            model.verify_unchanged()
        removal_restores = all(transfer_eval(model, None, oracle, cfg) == results[rid]["N0"]
                               for rid, model in models.items())
        derived = self._transfer_derived(results, budgets)
        document = {
            "schema": REPORT_SCHEMA + "/transfer", "claims": CLAIMS,
            "budgets": [0] + budgets, "selection": {f"N{n}": selection_report(sequence, n)
                                                     for n in [0] + budgets},
            "selection_sha256": selection_sha, "adapters": records,
            "opened_before_adapter_freeze": opened_before,
            "opened_after_adapter_freeze": list(oracle.opened),
            "results": results, "derived": derived,
            "integrity": {"frozen_source_states_unchanged_after_adaptation": True,
                          "adapter_removal_restores_N0": bool(removal_restores),
                          "note": ("adapter removal and frozen-state equality are INTEGRITY checks, "
                                   "not source-preservation accuracy; source-function degradation "
                                   "is the measured change in source-function error")},
            "scratch_control": ("scratch is the untrained common initialisation under the identical "
                                "adapter protocol; its N0 is the untrained network, not a fabricated "
                                "ridge fit"),
            "note": ("14 transfer-fit contexts, 168 candidates; budgets are nested; adaptation uses a "
                     "declared fixed recipe with no held-out selection; all results are preserved, "
                     "negative transfer included")}
        self._write_artifact("transfer/transfer.json", document)
        self.state["stages"]["transfer"] = {
            "complete": True, "adapters_frozen": frozen,
            "opened_before_adapter_freeze": opened_before, "transfer_sha256": fb.digest(document)}
        self._save()
        return document

    def _verify_transfer_artifacts(self, models, document):
        """Re-verify, on EVERY entry into a completed transfer stage (a same-object call included), the
        artifacts the stored summary refers to: exactly the expected adapters are tracked, each adapter's
        header is intact and equals the frozen record, its weights file exists and matches its recorded
        SHA-256 and tensor digest, and its bindings name the frozen source state, the nested selection and
        this configuration. A deleted or corrupted adapter refuses before anything is accepted."""
        cfg, directory = self.config, self.out / "transfer" / "adapters"
        budgets = [n for n in cfg.transfer_budgets if n > 0]
        expected = {f"{rid}|{n}" for rid in models for n in budgets}
        tracked = self.state["stages"]["transfer"]["adapters_frozen"]
        require(set(tracked) == expected and set(document["adapters"]) == expected,
                "the tracked adapters of the transfer stage are not exactly one per frozen model and "
                "budget")
        selection_sha = fb.digest([[int(c), int(f)] for c, f in transfer_selection(cfg)])
        require(document["selection_sha256"] == selection_sha,
                "the transfer summary records another adapter selection than this configuration's")
        for key in sorted(tracked):
            rid, budget = key.split("|")[0], int(key.split("|")[1])
            bindings = document["adapters"][key]["bindings"]
            require(bindings["source_state_sha256"] == models[rid].state_sha256
                    and bindings["selection_sha256"] == selection_sha
                    and bindings["config_sha256"] == self.state["config_sha256"],
                    f"the adapter of {rid} N{budget} is bound to another source state, selection or "
                    "configuration than the frozen run")
            load_adapter(directory, rid, budget, cfg, cfg.head_hidden, bindings=bindings)
            header = fb.read_json(directory / f"{rid}_n{budget}.adapter.json")
            require(header["header_sha256"] == tracked[key]["header_sha256"]
                    and header["state_sha256"] == tracked[key]["state_sha256"],
                    f"the adapter of {rid} N{budget} is not the one the transfer stage froze")
        return True

    @staticmethod
    def _transfer_derived(results, budgets):
        derived = {}
        for rid, entry in results.items():
            base = entry["N0"]
            derived[rid] = {f"N{n}": {
                "new_composition_kl_improvement": base["new_composition"]["kl"]
                - entry[f"N{n}"]["new_composition"]["kl"],
                "source_function_kl_degradation": entry[f"N{n}"]["source_function"]["kl"]
                - base["source_function"]["kl"]} for n in budgets}
        paired = {}
        for rid, entry in results.items():
            arm, seed = rid.rsplit("_s", 1)
            for n in budgets:
                for reference in (SCRATCH, "F"):
                    other = f"{reference}_s{seed}"
                    if arm == reference or other not in results:
                        continue
                    paired.setdefault(f"{arm}-{reference}", {}).setdefault(f"N{n}", []).append(
                        {"source_seed": int(seed),
                         "new_composition_kl_difference": entry[f"N{n}"]["new_composition"]["kl"]
                         - results[other][f"N{n}"]["new_composition"]["kl"],
                         "source_function_kl_difference": entry[f"N{n}"]["source_function"]["kl"]
                         - results[other][f"N{n}"]["source_function"]["kl"]})
        return {"per_run": derived, "paired_vs_scratch_and_F": paired,
                "sign": "differences are arm minus reference error; negative means lower error"}

    def export(self):
        """Hand the learned query-conditioned penultimate activations to Build A: one feature cache per
        frozen model, a roster join on the full (context, program) identity, a matrix frozen and
        verified BEFORE the transfer-evaluation outcomes are written, then scored."""
        self._require("transfer")
        models = self.frozen_models()
        cfg, base = self.config, self.out / "export"
        expected = export_expectation(models, cfg)
        if self._complete("export"):
            return self._verify_export_artifacts(expected)
        verified = self._export_preflight(expected)
        caches = {rid: write_bridge_cache(base / "caches", expected.rows, expected.features[rid],
                                          models[rid].header, rid, FEATURE_FUSED, cfg,
                                          expected.binding, protected=self.protected)
                  for rid in expected.rids}
        paths = write_bridge_inputs(base / "inputs", expected.rows, cfg, list(expected.rids),
                                    expected.reference, protected=self.protected)
        freeze_dir = base / "matrix"
        if verified is None:
            pending = base / "inputs" / EVAL_PENDING_NAME
            try:
                fm.run_matrix(candidates=paths["candidates"], train_outcomes=paths["train_outcomes"],
                              eval_outcomes=pending, manifest=paths["manifest"], plan=paths["plan"],
                              caches={rid: Path(info["header"]) for rid, info in caches.items()},
                              output_dir=freeze_dir)
                raise ValueError("internal error: the matrix evaluated although the evaluation "
                                 "outcomes were withheld")
            except ValueError as error:
                if not ("evaluation outcomes" in str(error) and "does not exist" in str(error)):
                    raise
            verified = self._check_freeze(freeze_dir, expected)
        freeze_record = {"index_sha256": verified["index"]["index_sha256"],
                         "counts": verified["index"]["counts"]}
        opened_before = list(expected.oracle.opened)
        record = {"complete": False, "freeze": freeze_record,
                  "opened_before_matrix_freeze": opened_before}
        if self.state["stages"].get("export") != record:
            self.state["stages"]["export"] = record
            self._save()
        # ---- the frozen boundary: only now are the transfer-evaluation answers materialized ----
        truth = eval_outcome_values(expected.rows, expected.oracle)
        eval_path = base / "inputs" / EVAL_OUTCOMES_NAME
        _publish_text_exact(eval_path, eval_outcomes_text(expected.rows, expected.oracle),
                            protected=self.protected)
        report_path = freeze_dir / fm.RESULTS_NAME
        results = (fb.read_json(report_path) if os.path.lexists(report_path)
                   else fm.evaluate_matrix(freeze_dir, eval_path))
        verify_matrix_report(results, verified, truth)
        summary = self._export_summary(expected, caches, freeze_record, results)
        self._write_artifact("export/bridge_summary.json", summary)
        self.state["stages"]["export"] = {"complete": True, "freeze": freeze_record,
                                          "opened_before_matrix_freeze": opened_before,
                                          "summary_sha256": fb.digest(summary)}
        self._save()
        return summary

    def _export_destinations(self, expected):
        """Every path an export can write, by label: the caches, the input files, the evaluation file and
        the withheld-file placeholder, every cell, the index, the input record and the report of the
        matrix, and the summary. (The ``.tmp`` staging name of each is checked with it.)"""
        base, freeze = self.out / "export", self.out / "export" / "matrix"
        out = {}
        for rid in expected.rids:
            out[f"cache header {rid}"] = base / "caches" / f"{rid}.cache.json"
            out[f"cache NPZ {rid}"] = base / "caches" / f"{rid}.cache.npz"
        for role, name in BRIDGE_INPUT_FILES.items():
            out[f"bridge input {role}"] = base / "inputs" / name
        out["evaluation outcomes"] = base / "inputs" / EVAL_OUTCOMES_NAME
        out["evaluation outcomes placeholder"] = base / "inputs" / EVAL_PENDING_NAME
        for key in fm.enumerate_cells(expected.plan):
            out[f"matrix cell {key.cell_id}"] = freeze / fm.CELLS_DIR / key.filename()
        out["matrix freeze index"] = freeze / fm.INDEX_NAME
        out["matrix input record"] = freeze / fm.INPUTS_NAME
        out["matrix evaluation report"] = freeze / fm.RESULTS_NAME
        out["bridge summary"] = base / "bridge_summary.json"
        return out

    @staticmethod
    def _check_freeze(freeze_dir, expected):
        """Verify the matrix freeze from disk and bind it to the CURRENT inputs: its plan, canonical cache
        digests and common identity (manifest, roster, offered training labels) must equal what this
        export computes."""
        verified = fm.verify_freeze(freeze_dir)
        verify_freeze_binding(verified, plan_sha256=expected.plan_sha256,
                              cache_sha256=expected.cache_sha256,
                              common_sha256=expected.common_sha256)
        return verified

    def _export_preflight(self, expected):
        """Decide every refusal BEFORE anything is published: protected aliases for ALL predictable
        destinations, existing caches and input files that are not exactly the expected ones, the withheld
        placeholder, and an existing matrix freeze that does not match the current inputs. Returns the
        verified existing freeze, or ``None`` when there is none. Writes nothing."""
        base, freeze = self.out / "export", self.out / "export" / "matrix"
        fst.refuse_protected(self.protected, self._export_destinations(expected))
        for rid in expected.rids:
            check_bridge_cache(base / "caches", rid, expected.cache_sha256[rid])
        for role, name in BRIDGE_INPUT_FILES.items():
            _check_exact_text(base / "inputs" / name, expected.texts[role])
        pending = base / "inputs" / EVAL_PENDING_NAME
        require(not os.path.lexists(pending),
                f"{pending} must not exist: it stands in for the unopened evaluation file")
        require(not freeze.exists() or freeze.is_dir(), f"{freeze} exists and is not a directory")
        if freeze.is_dir() and any(freeze.iterdir()):
            return self._check_freeze(freeze, expected)
        return None

    def _export_summary(self, expected, caches, freeze_record, results):
        rows = expected.rows
        return {
            "schema": REPORT_SCHEMA + "/bridge", "claims": CLAIMS,
            "roster": {"candidates": len(rows),
                       "train": sum(1 for r in rows if r["split"] == fb.SPLIT_TRAIN),
                       "eval": sum(1 for r in rows if r["split"] == fb.SPLIT_EVAL),
                       "contexts": len({r["group"] for r in rows}),
                       "queries_per_context": len({r["function"] for r in rows}),
                       "identity": {"alphabet": fe.IDENTITY_ALPHABET, "length": fe.identity_length()}},
            "feature": FEATURE_FUSED,
            "representations": {rid: {"cache_sha256": info["cache_sha256"]}
                                for rid, info in caches.items()},
            "raw_evidence": {"variant": expected.binding["variant"],
                             "events_sha256": expected.binding["events_sha256"],
                             "row_ids_sha256": expected.binding["row_ids_sha256"],
                             "note": ("the features were computed from raw events only; the digest "
                                      "binds the actual roster-ordered event arrays")},
            "reference": expected.reference, "matrix_freeze": freeze_record,
            "panel_reuse": {
                "statement": PANEL_REUSE,
                "transfer_evaluation_already_scored_by_the_transfer_stage": True,
                "opened_by_this_stage": list(expected.oracle.opened),
                "opened_before_matrix_freeze": copy.deepcopy(
                    self.state["stages"]["export"]["opened_before_matrix_freeze"])},
            "bridge": bridge_summary(results)}

    def _verify_export_artifacts(self, expected):
        """Re-verify, on EVERY entry into a completed export stage (a same-object call included), what the
        stored summary refers to: every feature cache equals the one the frozen model and the raw
        evidence give now, the input files are exactly the expected ones, the matrix freeze verifies from
        disk and binds to the current inputs, the evaluation outcomes file and the matrix report belong to
        the current offered outcomes, and the summary is the one the report gives. A deleted or corrupted
        artifact refuses before any state or report is accepted."""
        base, record = self.out / "export", self.state["stages"]["export"]
        summary = self._read_artifact("export/bridge_summary.json", record["summary_sha256"])
        for rid in expected.rids:
            require(check_bridge_cache(base / "caches", rid, expected.cache_sha256[rid]) is not None,
                    f"the exported feature cache of {rid} is missing")
        for role, name in BRIDGE_INPUT_FILES.items():
            require(_check_exact_text(base / "inputs" / name, expected.texts[role]),
                    f"the exported input file {name} is missing")
        verified = self._check_freeze(base / "matrix", expected)
        require(verified["index"]["index_sha256"] == record["freeze"]["index_sha256"],
                "the matrix freeze on disk is not the one the export stage recorded")
        truth = eval_outcome_values(expected.rows, expected.oracle)
        require(_check_exact_text(base / "inputs" / EVAL_OUTCOMES_NAME,
                                  eval_outcomes_text(expected.rows, expected.oracle)),
                "the exported evaluation outcomes file is missing")
        report_path = base / "matrix" / fm.RESULTS_NAME
        require(report_path.is_file(), "the matrix evaluation report is missing")
        results = fb.read_json(report_path)
        verify_matrix_report(results, verified, truth)
        require(summary["representations"] == {rid: {"cache_sha256": expected.cache_sha256[rid]}
                                               for rid in expected.rids}
                and summary["bridge"] == json.loads(fb.canonical_json(bridge_summary(results))),
                "the stored bridge summary is not the one the verified caches and report give")
        return summary

    def report(self):
        """One JSON report assembled from the verified stage artifacts. Negative and inconclusive
        results are retained; no winner is selected."""
        for stage in ("prepare", "train", "freeze", "evaluate", "transfer", "export"):
            self._require(stage)
        evaluation = self.evaluate()
        transfer = self.transfer()
        bridge = self.export()
        models = self.frozen_models()
        prepare = self.state["stages"]["prepare"]
        document = {
            "schema": REPORT_SCHEMA, "claims": CLAIMS, "config": self.config.document(),
            "world": {"q7": prepare["world"], "functions": prepare["functions"],
                      "orbits": prepare["orbits"], "split_claim": CLAIMS["split"]},
            "tapes": {"corruption": prepare["corruption"], "teacher_cost": prepare["teacher_cost"]},
            "costs": {rid: model.header["cost"] for rid, model in models.items()
                      if model.arm != SCRATCH},
            "evaluation": {"headlines": evaluation["headlines"],
                           "paired_seed_differences": evaluation["paired_seed_differences"],
                           "opened_splits": evaluation["opened_splits"]},
            "transfer": {"derived": transfer["derived"], "selection": transfer["selection"],
                         "integrity": transfer["integrity"]},
            "bridge": bridge["bridge"], "bridge_panel_reuse": bridge["panel_reuse"],
            "integrity": {"frozen_states_unchanged": all(m.verify_unchanged() for m in models.values()),
                          "stages": {k: bool(v.get("complete")) for k, v in
                                     self.state["stages"].items()}},
            "interpretation": [
                "A software pass never implies that IIT wins; every arm's result is preserved",
                "patched-versus-oracle error and patched-versus-hybrid agreement are separate",
                "the corrupted-supervision control is not proof of a uniquely correct alignment",
                "exact finite-panel means and paired seed differences, not population evidence",
                "the bridge reuses the transfer-evaluation panel the transfer stage already scored: a "
                "secondary analysis with settings fixed in advance, not a fresh confirmation",
                "synthetic worlds with an exact teacher: no biological or mechanistic claim"],
            "stage_state_sha256": fb.digest(self.state)}
        self._write_artifact(REPORT_NAME, document)
        return document

    def run(self, stages=STAGES):
        stages = list(stages)
        require(all(stage in STAGES for stage in stages), f"stages must be members of {STAGES}")
        results = {}
        for stage in stages:
            results[stage] = getattr(self, stage)()
        return results


def run_causal(output_dir, config, *, stages=STAGES, protected=()):
    """Python entry point: open (validate) the run and execute the requested stages in order."""
    return CausalRun.open(output_dir, config, protected=protected).run(stages)


# ---------------------------------------------------------------------------
# selftest and bounded pilot
# ---------------------------------------------------------------------------

def selftest_config():
    """Much smaller than the pilot but GENUINE: every arm trains on a real tape for two paired seeds,
    hidden interventions are evaluated on the real primary panel, adapters are fitted and frozen, and a
    real Build A matrix is frozen and scored. The batch is 32 so that the 384 tape rows give most
    (function orbit, site) strata the ``MIN_STRATUM`` events the corrupted control needs; strata too small
    to randomize are still reported unsupported, never counted as corrupted."""
    return CausalConfig(
        steps=12, batch=32, lr=3e-3, event_hidden=16, block_dim=8, program_embedding=8,
        program_hidden=16, head_hidden=32, seeds=(0, 1), variants=3, eval_variant=1,
        checkpoint_every=4, adapt_steps=30, adapter_hidden=8, export_label_budgets=(16, 32),
        export_acquisition_seeds=(0, 1), export_bootstrap_replicates=20)


def pilot_config():
    """The bounded pilot: 3 paired seeds, 1000 source updates, batch 64."""
    return CausalConfig()


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
    config = dataclasses.replace(selftest_config(), tape_seed=seed)
    check("world::q7_counts_and_function_families_verified",
          fe.verify_q7()["source"] == 1176 and len(fe.function_registry().functions) == 93)
    half = fe.world_posterior([fe.Observation("a", 0, 1, 0.75)])
    check("teacher::one_reliable_positive_is_three_quarters_and_no_evidence_is_one_half",
          fe.marginals(half)[0] == fe.POSTERIOR_LEVELS[4]
          and fe.program_probability(fe.Read(0), ()) == fe.POSTERIOR_LEVELS[3])
    check("teacher::correlated_prior_is_refused_by_the_factor_teacher",
          _refused(lambda: fe.interchange_probability((), (), 0, fe.Read(0),
                                                     fe.correlated_prior_x0_equals_x1())))
    run = CausalRun.open(work / "run", config)
    run.prepare(deep_verify=True)
    ctx = run.context()
    report = ctx.tapes[0].report["corruption"]
    check("tape::corrupted_targets_change_real_values_and_unsupported_strata_are_reported",
          report["changed_values"] > 0 and "unsupported_strata" in report,
          {k: report[k] for k in ("changed_fraction", "unchanged_fraction", "mean_abs_difference")})
    check("tape::every_supported_corruption_stratum_splits_a_repeated_value_so_it_is_no_relabelling",
          report["supported_strata"] > 0 and all(
              item["classes_split"] >= 1 for item in report["stratum_records"]
              if item["status"] == "supported"),
          {"supported": report["supported_strata"], "unsupported": len(report["unsupported_strata"])})
    runs = run.train()
    digests = {arm: [e["update_sha256"] for e in runs[run_id(arm, 0)].log] for arm in config.arms}
    check("arms::every_arm_consumed_the_identical_events_labels_and_sites",
          all(digests[arm] == digests["F"] for arm in config.arms))
    acc = {arm: runs[run_id(arm, 0)].accounting for arm in config.arms}
    check("arms::A_and_I_match_in_encodes_head_calls_and_padded_rows",
          acc["A"]["forward_calls"] == acc["I"]["forward_calls"]
          and acc["A"]["rows"]["event_rows_padded"] == acc["I"]["rows"]["event_rows_padded"],
          {"A": acc["A"]["rows"], "I": acc["I"]["rows"]})
    check("arms::M_has_marginal_heads_reached_only_in_M",
          acc["M"]["presentations"]["marginal_targets"] == config.steps * config.batch * N_BLOCKS
          and all(acc[a]["presentations"]["marginal_targets"] == 0 for a in config.arms if a != "M"))
    run.freeze()
    run.evaluate()
    evaluation = fb.read_json(work / "run" / "evaluation" / "evaluation.json")
    check("stages::held_out_panels_open_only_after_every_state_is_frozen",
          run._complete("freeze") and "primary" in evaluation["opened_splits"]
          and "transfer_eval" not in evaluation["opened_splits"])
    check("evaluation::identity_insertion_is_exactly_ordinary_prediction",
          all(m["headline"]["identity_insertion_max_abs_logit_difference"] == 0.0
              for m in evaluation["models"].values()))
    check("evaluation::encoder_invariances_hold_on_every_frozen_model",
          all(m["invariance"]["all_within_tolerance"] for m in evaluation["models"].values()))
    first = evaluation["models"]["I_s0"]["interventions"]
    check("evaluation::strong_primary_panel_has_21168_cases_and_both_comparisons",
          first["cases"] == 21168
          and "agreement_abs" in first["summary"]["changed"]["metrics"]
          and "patched_kl" in first["summary"]["changed"]["metrics"])
    transfer = run.transfer()
    check("transfer::adapters_frozen_before_transfer_eval_opened",
          "transfer_eval" not in transfer["opened_before_adapter_freeze"]
          and "transfer_eval" in transfer["opened_after_adapter_freeze"])
    check("transfer::N16_covers_all_14_contexts_and_N64_gives_4_or_5_each",
          transfer["selection"]["N16"]["contexts_covered"] == 14
          and transfer["selection"]["N64"]["labels_per_context_min"] == 4
          and transfer["selection"]["N64"]["labels_per_context_max"] == 5
          and transfer["selection"]["N16"]["derived_labels"] == 16)
    bridge = run.export()
    check("bridge::every_cell_of_the_real_build_a_matrix_ran_and_context_queries_share_a_group",
          bridge["bridge"]["status_counts"]["failed"] == 0
          and bridge["roster"]["queries_per_context"] == 12 and bridge["roster"]["contexts"] == 28,
          bridge["bridge"]["status_counts"])
    check("bridge::the_export_discloses_that_it_reuses_the_scored_transfer_panel_and_binds_raw_events",
          "ALREADY" in bridge["panel_reuse"]["statement"]
          and bridge["panel_reuse"]["opened_before_matrix_freeze"] == ["transfer_fit"]
          and len(bridge["raw_evidence"]["events_sha256"]) == 64)
    final = run.report()
    check("report::written_with_every_stage_complete",
          all(final["integrity"]["stages"].values()))
    again_transfer, again_export = run.transfer(), run.export()
    check("stages::completed_transfer_and_export_reverify_their_artifacts_on_reentry",
          again_transfer["selection_sha256"] == transfer["selection_sha256"]
          and again_export == json.loads(fb.canonical_json(bridge)))
    # resume parity: an interrupted run equals an uninterrupted one
    straight = train_arm(ctx, "I", 1, work / "straight")
    train_arm(ctx, "I", 1, work / "resumed", stop_after=5)
    resumed = train_arm(ctx, "I", 1, work / "resumed")
    check("resume::interrupted_and_resumed_run_equals_the_uninterrupted_run_exactly",
          straight.deterministic_result() == resumed.deterministic_result())
    probe = ArmRun(ctx, "I", 1)
    before = probe.state_sha256()
    probe.tape.arrays["p_cf"].setflags(write=True)
    probe.tape.arrays["p_cf"][0] += 0.25
    refused = _refused(probe.step)
    check("tamper::an_edited_counterfactual_target_refuses_before_any_optimizer_mutation",
          refused and probe.cursor == 0 and probe.state_sha256() == before)
    check("selftest::no_cuda_was_initialised", not torch.cuda.is_initialized())
    return {"checks": checks, "report_path": str(work / "run" / REPORT_NAME)}


def selftest(output_dir, seed=0):
    """Genuine small end-to-end run: real tapes, every arm trained, frozen, hidden interventions
    evaluated, adapters frozen, a real Build A matrix. Qualifies the machinery only."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="causal_selftest_", dir=out))
    started = time.process_time()
    with fst.single_thread():
        body = _selftest_in(work, seed)
    report = {"schema": SELFTEST_SCHEMA, "seed": seed, "work_directory": str(work),
              "passed": bool(all(item["holds"] for item in body["checks"])),
              "checks": body["checks"], "pipeline_report": body["report_path"],
              "cpu_seconds": time.process_time() - started, "claims": CLAIMS,
              "note": ("synthetic qualification of the learner and its interfaces. It asserts "
                       "mechanics, never a scientific win for any arm.")}
    fb.write_json(work / SELFTEST_REPORT_NAME, report)
    return report


def pilot(output_dir, *, config=None, deep_verify=False):
    """The bounded pilot (3 paired seeds, 1000 source updates, batch 64) in a directory this call
    creates. NOT started by anything here: call it explicitly after the selftest has been timed. A
    negative result is a valid result."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="causal_pilot_", dir=out))
    config = config or pilot_config()
    with fst.single_thread():
        run = CausalRun.open(work / "run", config)
        run.prepare(deep_verify=deep_verify)
        run.run([stage for stage in STAGES if stage != "prepare"])
    return {"work_directory": str(work), "report": str(work / "run" / REPORT_NAME),
            "stage_state": run.state["stages"]}
