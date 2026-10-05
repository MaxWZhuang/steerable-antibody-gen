"""Stable activation names, capture, and causal intervention for all three routes.

The campaign's stated purpose after the bake-off is mechanistic interpretation, so
the hook surface is part of the deliverable rather than something bolted on later.
Two design choices make the difference between an export and an experiment:

* **Names are stable across routes and independent of the upstream module tree.**
  ``route/<id>/block<i>/resid_pre`` means the same thing for the native AbLang2
  encoder block, for the GPTNeoX layer and for the energy stack's convolutional
  block. A name that encoded ``AbRep.encoder_blocks.3.multihead_attention`` would
  break the moment either backbone was re-released, and every saved shard would
  become unreadable.
* **The registry intervenes, not only observes.** Capture answers "what is
  there"; patch, zero-ablate, mean-ablate and cross-condition patch answer "does
  it matter". A registry that only captured would make every downstream claim
  correlational, and the controls in :mod:`her2_routes_controls` need the causal
  half.

Every captured tensor is accompanied by a ``token_map``: for each tensor index,
the ``(chain, residue index, serialisation role, edit-map slot)`` it corresponds
to. Without it an activation shard is a matrix of numbers whose rows nobody can
attribute to a residue, which is how a "position 104 feature" turns out to be the
separator.

Four things the earlier revisions got wrong, fixed here because each silently produces a
plausible-looking intervention result:

* **Mean ablation uses an explicit, frozen donor mean -- never the evaluation batch.**
  The first revision replaced an activation by the mean over *the batch being evaluated*.
  That changes with the batch: a row evaluated alone is replaced by itself (a no-op), the
  same row in a batch of eight is replaced by something else, and a held-out row
  contributes to its own control. :class:`ReferenceMean` is built once from a named donor
  set (:class:`ReferenceMeanBuilder`), cloned and detached, stored with one value per
  position, hashed, and applied identically to every batch. A plan without one raises.
* **A patch source carries its layout.** A bare tensor captured on a ``tokens_first`` tap
  and patched into a ``tokens_last`` tap with square dimensions has the right shape and the
  wrong axes; it would transpose silently. :class:`PatchSource` records the layout and
  :meth:`HookRegistry.plan` compares it with the tap's.
* **Export never guesses a layout.** A capture without a recorded layout raises; it is not
  labelled ``tokens_first``.
* **A capture is a copy taken at tap time.** ``tensor.detach()`` shares storage with the
  tensor the model goes on to use, so an identity tap followed by ``y.add_(5)`` rewrote the
  "captured" activation after the fact and the export showed what the model computed LATER,
  not what the tap saw. :meth:`HookRegistry._apply` clones.

Two further things the first revision of this module got wrong, fixed in the earlier
round because both silently produce a plausible-looking intervention result:

* **A tap declares its tensor layout.** The two transformer trunks emit
  ``(batch, tokens, channels)``; the energy stack's convolutions emit
  ``(batch, channels, positions)``. Indexing ``dim=1`` for "position 1" on the
  second layout zeroes *channel* 1 at every residue. Each tap therefore carries a
  :data:`LAYOUTS` entry and :meth:`HookRegistry.position_axis` resolves it.
* **The autoregressive map separates the token that is THERE from the coordinate
  that is being ASKED FOR.** At emission slot ``context_length + t`` the input
  token is the residue emitted for ``order[t-1]`` (or the separator at ``t = 0``)
  and the next-token logits are the conditional for ``order[t]``. Recording one
  and calling it the other misattributes every query in the window by one step.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from ..experiments.her2_routes_serial import (
    ABLANG_END_ID,
    ABLANG_MASK_ID,
    ABLANG_SEP_ID,
    ABLANG_START_ID,
    HEAVY,
    LIGHT,
    PIGGEN_END_ID,
    PIGGEN_MASK_ID,
    PIGGEN_SEP_ID,
    PIGGEN_START_ID,
)
from ..experiments.her2_runtime import require

#: The four points inside a block this registry can address.
SITES = ("resid_pre", "attn_out", "mlp_out", "resid_post")
#: The interventions available at each of them.
INTERVENTIONS = ("capture", "patch", "zero_ablate", "mean_ablate", "cross_condition_patch")

#: Tensor layouts a tap may declare, and which axis carries sequence position.
#: ``tokens_last`` is the convolutional layout the energy stack emits; addressing
#: it as if it were ``tokens_first`` ablates a channel at every residue instead of
#: a residue at every channel, and the resulting figure looks entirely normal.
LAYOUTS = {"tokens_first": 1, "tokens_last": 2}
LAYOUT_TOKENS_FIRST = "tokens_first"
LAYOUT_TOKENS_LAST = "tokens_last"

ROLE_START = "start"
ROLE_END = "end"
ROLE_SEPARATOR = "separator"
ROLE_MASK = "mask"
ROLE_RESIDUE = "residue"
ROLE_QUERY = "query"
ROLE_PAD = "pad"


def activation_name(route, block, site):
    require(site in SITES, f"Unknown activation site {site!r}; expected one of {SITES}")
    require(isinstance(block, int) and block >= 0, "Block index must be a non-negative int")
    return f"route/{route}/block{int(block)}/{site}"


# ---------------------------------------------------------------------------
# token maps
# ---------------------------------------------------------------------------

def causal_token_map(construct, serialization):
    """Per tensor index: what token is THERE and what coordinate it QUERIES.

    The two are different things at every position of the emission block, and
    conflating them shifts the whole window by one step. Layout, with
    ``C = context_length``::

        0            [START]
        1 .. C-2     VH then VL, masked at the edit coordinates
        C-1          [END]
        C            [SEP]              -> next-token logits are order[0]
        C+1          residue of order[0] -> next-token logits are order[1]
        ...
        C+k          residue of order[k-1] -> queries nothing that is scored

    Each row therefore carries an *input* identity (``chain``/``residue_index``
    for a context residue, ``input_edit_slot`` for an emitted answer) and a
    *query* identity (``query_site``/``query_edit_slot``/``query_pointer``). The
    final trailing answer token has an input and no query, and says so.
    """
    slots = {site: position for position, site in enumerate(serialization.order)}
    order = list(serialization.order)
    pointers = list(serialization.pointers)
    context_length = int(serialization.context_length)
    query_slots = set(int(value) for value in serialization.query_slots)
    rows = []
    for index, token in enumerate(serialization.tokens):
        entry = {"index": index, "chain": None, "residue_index": None, "role": ROLE_RESIDUE,
                 "edit_slot": None, "input_edit_slot": None, "query_site": None,
                 "query_edit_slot": None, "query_pointer": None, "is_scored_query": False}
        if index == 0 and token == PIGGEN_START_ID:
            entry["role"] = ROLE_START
        elif index == context_length - 1 and token == PIGGEN_END_ID:
            entry["role"] = ROLE_END
        elif index < context_length - 1:
            offset = index - 1
            if offset < len(construct.heavy):
                entry.update(chain=HEAVY, residue_index=offset)
            else:
                entry.update(chain=LIGHT, residue_index=offset - len(construct.heavy))
            site = (entry["chain"], entry["residue_index"])
            if token == PIGGEN_MASK_ID:
                entry["role"] = ROLE_MASK
                entry["edit_slot"] = slots.get(site)
        elif token == PIGGEN_SEP_ID and index == context_length:
            entry["role"] = ROLE_SEPARATOR
        else:
            # An emitted answer: its INPUT is the residue of the previous step.
            step = index - context_length - 1
            if 0 <= step < len(order):
                entry.update(input_edit_slot=step, chain=order[step][0],
                             residue_index=order[step][1])
        # The QUERY is the coordinate whose conditional this position's logits are.
        query_step = index - context_length
        if 0 <= query_step < len(order):
            entry.update(query_site=[order[query_step][0], int(order[query_step][1])],
                         query_edit_slot=query_step,
                         query_pointer=int(pointers[query_step]),
                         role=ROLE_QUERY if entry["role"] == ROLE_RESIDUE else entry["role"],
                         is_scored_query=index in query_slots)
        rows.append(entry)
    return rows


def require_causal_query_alignment(rows, serialization, *, label="causal token map"):
    """Every declared query slot maps to its own step, site and pointer. Asserted.

    This is the check that would have caught the off-by-one the map used to carry:
    slot ``context_length`` was recorded as a separator with no query, and slot
    ``context_length + 1`` was recorded as querying ``order[0]`` when its logits
    are the conditional for ``order[1]``.
    """
    by_index = {row["index"]: row for row in rows}
    for step, slot in enumerate(serialization.query_slots):
        row = by_index.get(int(slot))
        require(row is not None, f"{label}: query slot {slot} has no mapped row")
        require(row["query_edit_slot"] == step,
                f"{label}: query slot {slot} is step {step} of the emission order but the map "
                f"records step {row['query_edit_slot']}")
        site = serialization.order[step]
        require(row["query_site"] == [site[0], int(site[1])],
                f"{label}: query slot {slot} should query {site}, map says {row['query_site']}")
        require(row["query_pointer"] == int(serialization.pointers[step]),
                f"{label}: query slot {slot} carries pointer {row['query_pointer']}, the "
                f"serialisation says {serialization.pointers[step]}")
        require(row["is_scored_query"], f"{label}: query slot {slot} is not marked as scored")
    trailing = [row for row in rows if row["input_edit_slot"] is not None
                and row["query_site"] is None]
    return {"query_slots": len(serialization.query_slots),
            "trailing_unqueried_answers": len(trailing),
            "rule": ("the token at a slot and the coordinate that slot predicts are recorded "
                     "separately. The final emitted answer has an input and no scored query.")}


def masked_token_map(construct, serialization):
    """The same mapping for the native ``<VH>|<VL>`` layout."""
    slots = {site: position for position, site in enumerate(serialization.order)}
    heavy_length = len(construct.heavy)
    rows = []
    for index, token in enumerate(serialization.tokens):
        entry = {"index": index, "chain": None, "residue_index": None, "role": ROLE_RESIDUE,
                 "edit_slot": None}
        if token == ABLANG_START_ID:
            entry["role"] = ROLE_START
        elif token == ABLANG_END_ID:
            entry["role"] = ROLE_END
        elif token == ABLANG_SEP_ID:
            entry["role"] = ROLE_SEPARATOR
        else:
            if 1 <= index <= heavy_length:
                entry.update(chain=HEAVY, residue_index=index - 1)
            else:
                entry.update(chain=LIGHT, residue_index=index - (heavy_length + 4))
            site = (entry["chain"], entry["residue_index"])
            if token == ABLANG_MASK_ID:
                entry["role"] = ROLE_MASK
                entry["edit_slot"] = slots.get(site)
        rows.append(entry)
    return rows


def require_token_map(rows, tensor_length, *, label="token map"):
    """Every tensor index must be attributed. A partial map is refused."""
    require(len(rows) == int(tensor_length),
            f"{label}: {len(rows)} mapped indices against a tensor of length {tensor_length}. An "
            "activation row nobody can attribute to a residue is how a 'position 104 feature' "
            "turns out to be the separator.")
    indices = [row["index"] for row in rows]
    require(indices == list(range(int(tensor_length))), f"{label}: indices are not 0..n-1")
    return True


# ---------------------------------------------------------------------------
# patch sources and frozen donor means
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchSource:
    """An activation to patch in, together with the layout it was captured in.

    Shape agreement is not layout agreement: a ``(2, 4, 4)`` tensor is a valid patch for a
    ``tokens_first`` and a ``tokens_last`` tap alike, and patching one into the other
    transposes it without a single shape error. The layout is therefore part of the source.
    """

    tensor: object = field(repr=False, compare=False)
    layout: str = ""
    name: str = ""

    def __post_init__(self):
        require(isinstance(self.tensor, torch.Tensor),
                "A patch source must be a torch tensor carrying a declared layout")
        require(self.layout in LAYOUTS,
                f"A patch source needs a declared layout, one of {sorted(LAYOUTS)}; got "
                f"{self.layout!r}. Without it a square activation patched across layouts is "
                "silently transposed.")
        require(self.tensor.dim() > LAYOUTS[self.layout],
                f"A {self.layout} patch source needs more than {LAYOUTS[self.layout]} "
                f"dimensions; got shape {tuple(self.tensor.shape)}")


def as_patch_source(source, *, label="patch source"):
    """Accept a :class:`PatchSource` or a :meth:`HookRegistry.export` record; refuse a bare tensor."""
    if isinstance(source, PatchSource):
        return source
    if isinstance(source, dict) and "tensor" in source and "layout" in source:
        if "position_axis" in source:
            require(LAYOUTS.get(source["layout"]) == source["position_axis"],
                    f"{label}: position_axis {source['position_axis']} contradicts layout "
                    f"{source['layout']!r}")
        return PatchSource(tensor=source["tensor"], layout=source["layout"],
                           name=str(source.get("name", "")))
    raise ValueError(f"{label} must carry its layout: pass registry.export(name) or a "
                     "PatchSource, not a bare tensor. A bare tensor cannot be checked against "
                     "the target tap's layout, and on a square activation a layout mismatch "
                     "transposes silently.")


#: Keys :meth:`ReferenceMean.document` writes from the record's own fields. Caller-supplied
#: provenance may not use them: the first revision splatted provenance AFTER these, so
#: ``provenance={"sha256": "overridden", "donor_ids": ["eval"]}`` rewrote the document of a
#: reference whose :meth:`ReferenceMean.verify` still passed.
RESERVED_DOCUMENT_KEYS = frozenset({
    "name", "layout", "position_axis", "shape", "sha256", "donor_rows", "donor_ids_named",
    "donor_ids", "min_rows_per_position", "max_rows_per_position"})

#: Keys :meth:`ReferenceMeanBuilder.build` writes into the provenance itself.
RESERVED_BUILDER_KEYS = frozenset({"padding_masked", "ids_supplied"})


def _freeze_provenance(value, path="provenance"):
    """A deep, immutable copy of plain JSON data: dicts -> read-only mappings, lists -> tuples.

    The caller's own containers are never kept, so mutating them afterwards (including a
    nested list inside a dict) cannot change the frozen provenance. Anything that is not
    plain JSON data is refused with its path rather than stringified, because a provenance
    that cannot be canonicalised cannot be hashed.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        require(math.isfinite(value), f"{path} is not finite; provenance must be plain JSON data")
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_provenance(item, f"{path}[{index}]")
                     for index, item in enumerate(value))
    if isinstance(value, Mapping):
        for key in value:
            require(isinstance(key, str),
                    f"{path} has the non-string key {key!r}; provenance keys must be strings")
        return MappingProxyType({key: _freeze_provenance(value[key], f"{path}.{key}")
                                 for key in sorted(value)})
    raise ValueError(f"{path} holds a {type(value).__name__}; provenance must be plain JSON data "
                     "(str, int, float, bool, None, lists and string-keyed dicts of those). "
                     "Convert it explicitly before freezing it into a reference.")


def _thaw_provenance(value):
    """A fresh, mutable copy of frozen provenance, for JSON documents."""
    if isinstance(value, Mapping):
        return {key: _thaw_provenance(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_provenance(item) for item in value]
    return value


def _reference_digest(name, layout, mean, counts, donor_ids, donor_rows, provenance):
    """SHA-256 over the tensors AND the identity that explains them.

    The header carries the donor ids, the actual donor-row count and the canonical
    provenance, so a reference whose provenance was swapped after freezing fails
    :meth:`ReferenceMean.verify` exactly as one whose tensor bytes changed does.
    """
    header = json.dumps(
        {"name": str(name), "layout": str(layout), "shape": list(mean.shape),
         "dtype": str(mean.dtype), "counts_shape": list(counts.shape),
         "counts_dtype": str(counts.dtype), "donor_rows": int(donor_rows),
         "donor_ids": [str(item) for item in donor_ids],
         "provenance": _thaw_provenance(provenance)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    digest = hashlib.sha256()
    digest.update(header.encode("utf-8"))
    digest.update(mean.contiguous().numpy().tobytes())
    digest.update(counts.contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceMean:
    """A frozen per-position donor mean for mean ablation.

    ``mean`` has shape ``(1, ...)``: one reference value per position (and per channel),
    independent of the evaluation batch. It was cloned and detached to CPU float32 when
    built, ``counts`` records how many donor rows contributed at each position (padding is
    excluded by mask), and ``sha256`` lets :meth:`verify` catch an in-place edit after it was
    frozen.

    Two different numbers describe the donors and they are not interchangeable.
    ``donor_rows`` is how many rows actually went into the mean. ``donor_ids`` is the sorted
    tuple of the identities the caller *named* for them -- empty when none were named, and
    then ``donor_rows`` still says how many rows contributed. Identities are kept so a
    held-out row can be refused as a donor and so provenance travels with every result.

    The whole record is immutable, not only the tensor bytes. ``donor_ids`` is a tuple,
    ``provenance`` is a deep read-only copy of the caller's data (later edits to the dict
    the caller passed cannot reach it), the digest covers the donor ids, the row count and
    the provenance, and provenance may not use any key the record writes itself
    (:data:`RESERVED_DOCUMENT_KEYS`).
    """

    name: str
    layout: str
    mean: object = field(repr=False, compare=False)
    counts: object = field(repr=False, compare=False)
    donor_ids: tuple
    donor_rows: int
    sha256: str
    provenance: object = field(default_factory=dict, compare=False)

    def __post_init__(self):
        require(self.layout in LAYOUTS,
                f"Unknown tensor layout {self.layout!r}; expected one of {sorted(LAYOUTS)}")
        require(isinstance(self.mean, torch.Tensor) and isinstance(self.counts, torch.Tensor),
                f"{self.name}: a reference mean and its counts must be tensors")
        require(isinstance(self.donor_ids, (list, tuple)),
                f"{self.name}: donor_ids must be a list or tuple of identities, not a "
                f"{type(self.donor_ids).__name__}; a bare string would be read one character "
                "at a time")
        ids = tuple(sorted(str(item) for item in self.donor_ids))
        require(len(set(ids)) == len(ids),
                f"{self.name}: a donor id appears more than once; a repeated row would count "
                "twice in the mean")
        rows = self.donor_rows
        require(isinstance(rows, int) and not isinstance(rows, bool) and rows >= 1,
                f"{self.name}: donor_rows must be a positive integer, got {rows!r}")
        require(not ids or len(ids) == rows,
                f"{self.name}: {len(ids)} named donor ids for {rows} donor rows; ids are named "
                "for every donor row or for none")
        most = int(self.counts.max()) if self.counts.numel() else 0
        require(most <= rows,
                f"{self.name}: a position reports {most} contributing rows from only {rows} "
                "donor rows")
        require(isinstance(self.provenance, Mapping),
                f"{self.name}: provenance must be a mapping of plain JSON data, not a "
                f"{type(self.provenance).__name__}")
        frozen = _freeze_provenance(self.provenance, f"{self.name} provenance")
        clash = sorted(set(frozen) & RESERVED_DOCUMENT_KEYS)
        require(not clash,
                f"{self.name}: provenance uses reserved key(s) {clash}, which the reference "
                "writes from its own fields. Provenance describes where the donors came from; "
                "it cannot restate the record.")
        object.__setattr__(self, "donor_ids", ids)
        object.__setattr__(self, "provenance", frozen)

    @property
    def position_axis(self):
        return LAYOUTS[self.layout]

    def digest(self):
        return _reference_digest(self.name, self.layout, self.mean, self.counts, self.donor_ids,
                                 self.donor_rows, self.provenance)

    def frozen_copy(self):
        """An independent copy: ``mean`` and ``counts`` are cloned, everything else is carried over.

        :meth:`HookRegistry.plan` installs this copy and not the caller's object. The tensors of a
        frozen dataclass are still mutable, so ``reference.mean.add_(5)`` after planning would
        otherwise rewrite what the intervention writes while the plan's recorded digest still
        named the original bytes. The digest is carried over unchanged: the copy is bit-identical,
        so it verifies exactly as the original did when it was cloned.
        """
        return ReferenceMean(
            name=self.name, layout=self.layout, mean=self.mean.detach().clone(),
            counts=self.counts.detach().clone(), donor_ids=self.donor_ids,
            donor_rows=self.donor_rows, sha256=self.sha256, provenance=self.provenance)

    def verify(self):
        """Raise if the mean, its counts, its donors or its provenance changed after freezing."""
        require(self.digest() == self.sha256,
                f"The reference mean for {self.name} no longer matches its frozen digest; it was "
                "modified after it was built. A reference is built once from its donors and "
                "never edited.")
        return True

    def value_for(self, activation, *, name=None):
        """The reference broadcast to ``activation``'s batch, as a fresh tensor.

        Raises if any dimension beyond the batch differs: a reference built on 14-token
        donors is not silently stretched, truncated or padded onto 16-token rows. The result
        is a copy on ``activation``'s device and dtype, so nothing downstream can write
        through to the frozen mean.
        """
        label = name or self.name
        require(activation.dim() == self.mean.dim(),
                f"{label}: the activation has {activation.dim()} dimensions and the reference "
                f"mean {self.mean.dim()}")
        require(tuple(activation.shape[1:]) == tuple(self.mean.shape[1:]),
                f"{label}: the reference mean has non-batch shape {tuple(self.mean.shape[1:])} "
                f"and the activation {tuple(activation.shape[1:])}. Pad or build the reference "
                "on the same layout; it is never stretched or truncated.")
        value = self.mean.to(device=activation.device, dtype=activation.dtype)
        return value.expand_as(activation).clone()

    def document(self):
        """JSON-able provenance, without the tensors. Every call returns fresh containers.

        The record's own fields are written LAST, so no provenance entry can shadow them (the
        constructor already refuses a clash; this is the second guard). ``donor_rows`` is the
        number of rows that actually went into the mean and ``donor_ids_named`` how many of
        them the caller gave an identity; the two differ whenever no ids were supplied.
        """
        record = {"name": self.name, "layout": self.layout, "position_axis": self.position_axis,
                  "shape": list(self.mean.shape), "sha256": self.sha256,
                  "donor_rows": int(self.donor_rows), "donor_ids_named": len(self.donor_ids),
                  "donor_ids": list(self.donor_ids),
                  "min_rows_per_position": int(self.counts.min()),
                  "max_rows_per_position": int(self.counts.max())}
        return {**_thaw_provenance(self.provenance), **record}


class ReferenceMeanBuilder:
    """Accumulate donor activations into one frozen per-position :class:`ReferenceMean`.

    ``add`` may be called for several donor batches of the same non-batch shape;
    ``mask`` (``(rows, positions)``, true for real tokens) excludes padding so a short donor
    does not drag the late positions toward its pad value. ``build`` refuses a position that
    no donor row reaches.
    """

    def __init__(self, *, name, layout):
        require(layout in LAYOUTS,
                f"Unknown tensor layout {layout!r}; expected one of {sorted(LAYOUTS)}")
        self.name, self.layout = str(name), layout
        self._sum = None
        self._counts = None
        self._shape = None
        self._rows = 0
        self._ids = []
        self._with_ids = None
        self._masked = False

    def add(self, activations, *, mask=None, donor_ids=None):
        axis = LAYOUTS[self.layout]
        require(isinstance(activations, torch.Tensor),
                f"{self.name}: donor activations must be a tensor")
        self._masked = self._masked or mask is not None
        require(activations.dim() > axis and activations.shape[0] >= 1,
                f"{self.name}: a {self.layout} donor needs a batch and more than {axis} "
                f"dimensions; got shape {tuple(activations.shape)}")
        values = activations.detach().to("cpu", dtype=torch.float64).clone()
        rows, length = int(values.shape[0]), int(values.shape[axis])
        if mask is None:
            keep = torch.ones(rows, length, dtype=torch.bool)
        else:
            keep = torch.as_tensor(mask).detach().to("cpu").to(torch.bool).clone()
            require(tuple(keep.shape) == (rows, length),
                    f"{self.name}: the padding mask is {tuple(keep.shape)}; it must be "
                    f"(rows, positions) = {(rows, length)}")
        broadcast = [rows] + [1] * (values.dim() - 1)
        broadcast[axis] = length
        selected = keep.reshape(broadcast).expand_as(values)
        require(bool(torch.isfinite(values[selected]).all()),
                f"{self.name}: a donor activation is not finite")
        values = torch.where(selected, values, torch.zeros_like(values))
        total, counts = values.sum(dim=0, keepdim=True), keep.sum(dim=0)
        if self._sum is None:
            self._sum, self._counts, self._shape = total, counts, tuple(values.shape[1:])
        else:
            require(tuple(values.shape[1:]) == self._shape,
                    f"{self.name}: this donor batch has non-batch shape {tuple(values.shape[1:])} "
                    f"but earlier donors had {self._shape}")
            self._sum = self._sum + total
            self._counts = self._counts + counts
        self._rows += rows
        if donor_ids is not None:
            ids = [str(item) for item in donor_ids]
            require(len(ids) == rows, f"{len(ids)} donor ids for {rows} donor rows")
            require(self._with_ids in (None, True),
                    "Donor ids must be supplied for every donor batch or for none")
            self._with_ids = True
            self._ids.extend(ids)
        else:
            require(self._with_ids in (None, False),
                    "Donor ids must be supplied for every donor batch or for none")
            self._with_ids = False
        return self

    def build(self, *, evaluation_ids=None, provenance=None):
        """Freeze the accumulated donors. ``provenance`` is deep-copied and may not restate the record.

        It must be plain JSON data and may not use a key the reference or this builder writes
        (:data:`RESERVED_DOCUMENT_KEYS`, :data:`RESERVED_BUILDER_KEYS`): the donor ids, the
        donor-row count and the digest come from the donors, never from the caller's notes.
        """
        require(self._sum is not None, f"{self.name}: no donor activations were added")
        require(provenance is None or isinstance(provenance, Mapping),
                f"{self.name}: provenance must be a mapping of plain JSON data, not a "
                f"{type(provenance).__name__}")
        supplied = dict(provenance or {})
        clash = sorted(str(key) for key in supplied
                       if key in RESERVED_DOCUMENT_KEYS | RESERVED_BUILDER_KEYS)
        require(not clash,
                f"{self.name}: provenance may not set {clash}. Those keys are written from the "
                "donors themselves (identity, row count, digest, padding and id flags); a "
                "caller-supplied value would override the record while verify() still passed.")
        axis = LAYOUTS[self.layout]
        empty = [int(index) for index in torch.nonzero(self._counts == 0).flatten()]
        require(not empty,
                f"{self.name}: position(s) {empty[:8]} have no real donor row. A reference mean "
                "needs at least one unmasked donor at every position it will be applied to.")
        shape = [1] * self._sum.dim()
        shape[axis] = int(self._counts.shape[0])
        mean = (self._sum / self._counts.to(torch.float64).reshape(shape)).to(torch.float32).clone()
        ids = tuple(sorted(self._ids))
        require(len(set(ids)) == len(ids),
                f"{self.name}: a donor id appears more than once; a repeated row would count "
                "twice in the mean")
        if evaluation_ids is not None:
            require(self._with_ids,
                    "evaluation_ids were supplied but the donors carry no ids, so disjointness "
                    "cannot be checked")
            overlap = sorted(set(ids) & {str(item) for item in evaluation_ids})
            require(not overlap,
                    f"{len(overlap)} held-out row(s) are also donors, e.g. {overlap[:3]}. A "
                    "reference mean built from the rows it is applied to is a self-mean.")
        counts = self._counts.clone()
        details = {**supplied, "padding_masked": bool(self._masked),
                   "ids_supplied": bool(self._with_ids)}
        frozen = _freeze_provenance(details, f"{self.name} provenance")
        return ReferenceMean(
            name=self.name, layout=self.layout, mean=mean, counts=counts, donor_ids=ids,
            donor_rows=int(self._rows),
            sha256=_reference_digest(self.name, self.layout, mean, counts, ids, self._rows,
                                     frozen),
            provenance=frozen)


def reference_mean(activations, *, name, layout, mask=None, donor_ids=None,
                   evaluation_ids=None, provenance=None):
    """One-shot :class:`ReferenceMeanBuilder` for a single donor batch."""
    return (ReferenceMeanBuilder(name=name, layout=layout)
            .add(activations, mask=mask, donor_ids=donor_ids)
            .build(evaluation_ids=evaluation_ids, provenance=provenance))


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

@dataclass
class HookRegistry:
    """Named taps on one model, with capture and causal intervention at each.

    ``bind_*`` maps the stable names onto whichever modules the route actually
    has. Nothing here reaches into a backbone by string path: a re-release that
    renamed a submodule breaks the bind function loudly, in one place.
    """

    route: str
    taps: dict = field(default_factory=dict)
    captured: dict = field(default_factory=dict)
    captured_layout: dict = field(default_factory=dict)
    handles: list = field(default_factory=list)
    interventions: dict = field(default_factory=dict)

    def register(self, name, module, *, site, layout=LAYOUT_TOKENS_FIRST):
        require(name not in self.taps, f"{name} is already registered")
        require(layout in LAYOUTS,
                f"Unknown tensor layout {layout!r}; expected one of {sorted(LAYOUTS)}. A tap "
                "without a declared layout cannot be position-addressed, because dim 1 is "
                "tokens on a transformer block and channels on a convolution.")
        self.taps[name] = {"module": module, "site": site, "layout": layout}
        return name

    def names(self):
        return sorted(self.taps)

    def layout(self, name):
        require(name in self.taps, f"{name} is not a registered tap")
        return self.taps[name]["layout"]

    def position_axis(self, name):
        """Which tensor axis carries sequence position for this tap."""
        return LAYOUTS[self.layout(name)]

    # -- capture -----------------------------------------------------------
    def _output_tensor(self, output):
        """Blocks return either a tensor or ``(tensor, attention)``; take the state."""
        if isinstance(output, tuple):
            return output[0]
        return output

    def enable(self, names=None):
        """Attach hooks for ``names`` (default: all). Idempotent per call."""
        self.disable()
        for name in (sorted(self.taps) if names is None else list(names)):
            require(name in self.taps, f"{name} is not a registered tap")
            entry = self.taps[name]
            site = entry["site"]
            if site == "resid_pre":
                self.handles.append(entry["module"].register_forward_pre_hook(
                    self._make_pre_hook(name)))
            else:
                self.handles.append(entry["module"].register_forward_hook(
                    self._make_hook(name)))
        return len(self.handles)

    def disable(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return True

    def clear(self):
        self.captured = {}
        self.captured_layout = {}
        return self

    def export(self, name):
        """A captured activation with the layout it was captured in. Never bare.

        An exported shard whose axis order nobody recorded is how a channel gets
        analysed as a residue; the layout travels with the tensor. A capture whose layout
        was not recorded raises: the layout is read from the tap that produced the tensor,
        and it is not assumed to be ``tokens_first`` when it is missing.
        """
        require(name in self.captured, f"{name} has not been captured")
        require(name in self.captured_layout,
                f"{name} was captured without a recorded layout; refusing to guess one. "
                "Re-capture it through this registry.")
        layout = self.captured_layout[name]
        require(layout in LAYOUTS, f"{name} carries the unknown layout {layout!r}")
        # An independent copy: the exported artifact is the caller's to mutate (shards, patch
        # sources), and an in-place edit of it must not rewrite what the registry captured.
        tensor = self.captured[name].detach().clone()
        return {"name": name, "tensor": tensor, "layout": layout,
                "position_axis": LAYOUTS[layout], "shape": list(tensor.shape)}

    def export_all(self, names=None):
        """``{name: export(name)}`` for every captured tap (or the named ones)."""
        selected = sorted(self.captured) if names is None else list(names)
        return {name: self.export(name) for name in selected}

    def _apply(self, name, tensor):
        plan = self.interventions.get(name)
        # A copy, taken now: the model keeps using ``tensor`` and may rewrite it in place, and
        # the capture is evidence of what the tap saw, not of what the tensor later held.
        self.captured[name] = tensor.detach().clone()
        self.captured_layout[name] = self.layout(name)
        if plan is None:
            return tensor
        kind = plan["kind"]
        index = plan.get("positions")
        if kind == "capture":
            return tensor
        modified = tensor.clone()
        if kind == "zero_ablate":
            value = torch.zeros_like(tensor)
        elif kind == "mean_ablate":
            # Replace with the FROZEN donor mean at each position: removes the row-specific
            # part of the activation and leaves a population reference that was fixed before
            # this batch existed. It is never the mean of the batch being evaluated, which
            # changes with the batch and equals the row itself when the batch is one row.
            reference = plan.get("reference")
            require(reference is not None,
                    f"{name}: a mean ablation reached the hook with no frozen reference mean; "
                    "there is no batch-mean fallback.")
            value = reference.value_for(tensor, name=name)
        elif kind in ("patch", "cross_condition_patch"):
            value = plan["source"].tensor.to(tensor.device, dtype=tensor.dtype)
            require(value.shape[1:] == tensor.shape[1:],
                    f"{name}: patch source {tuple(value.shape)} does not match the activation "
                    f"{tuple(tensor.shape)} beyond the batch dimension")
            require(value.shape[0] in (1, tensor.shape[0]),
                    f"{name}: patch source has batch {value.shape[0]} and the activation "
                    f"{tensor.shape[0]}; a source is one row to broadcast or one row per batch row")
            if value.shape[0] == 1:
                value = value.expand_as(tensor)
        else:
            raise ValueError(f"Unknown intervention {kind!r}")
        if index is None:
            # A copy, so nothing downstream can write through to the plan's source/reference.
            return value.clone()
        axis = self.position_axis(name)
        require(axis < tensor.dim(),
                f"{name}: the declared layout puts position on axis {axis} and the activation "
                f"has {tensor.dim()} dimensions")
        positions = torch.as_tensor(list(index), dtype=torch.long, device=tensor.device)
        require(bool((positions < tensor.shape[axis]).all()),
                f"{name}: position {max(int(p) for p in positions)} is outside the "
                f"{tensor.shape[axis]}-long sequence axis of this activation")
        modified.index_copy_(axis, positions, value.index_select(axis, positions))
        return modified

    def _make_hook(self, name):
        def hook(_module, _args, output):
            tensor = self._output_tensor(output)
            replaced = self._apply(name, tensor)
            if replaced is tensor:
                return None
            if isinstance(output, tuple):
                return (replaced, *output[1:])
            return replaced
        return hook

    def _make_pre_hook(self, name):
        def hook(_module, args):
            tensor = args[0]
            replaced = self._apply(name, tensor)
            if replaced is tensor:
                return None
            return (replaced, *args[1:])
        return hook

    # -- interventions -----------------------------------------------------
    def plan(self, name, *, kind, positions=None, source=None, reference=None):
        """Declare the intervention at one tap.

        ``source`` (patch kinds) must carry its layout -- pass :meth:`export` or a
        :class:`PatchSource` -- and the layout must equal the tap's. ``reference``
        (``mean_ablate``) must be a :class:`ReferenceMean` built for this tap and layout and
        still matching its digest. Either argument supplied to a kind that does not use it is
        refused rather than ignored.
        """
        require(name in self.taps, f"{name} is not a registered tap")
        require(kind in INTERVENTIONS, f"Unknown intervention {kind!r}")
        is_patch = kind in ("patch", "cross_condition_patch")
        require(is_patch or source is None, f"{kind} takes no source activation")
        require(kind == "mean_ablate" or reference is None,
                f"{kind} takes no reference mean")
        resolved_source = resolved_reference = None
        if is_patch:
            require(source is not None, f"{kind} needs a source activation")
            candidate = as_patch_source(source, label=f"{name} {kind} source")
            require(candidate.layout == self.layout(name),
                    f"{name}: the patch source was captured as {candidate.layout!r} and this tap "
                    f"is {self.layout(name)!r}. Patching across layouts transposes the activation "
                    "even when both are square.")
            resolved_source = PatchSource(tensor=candidate.tensor.detach().clone(),
                                          layout=candidate.layout, name=candidate.name)
        if kind == "mean_ablate":
            require(isinstance(reference, ReferenceMean),
                    f"{name}: mean ablation needs an explicit frozen donor/reference mean "
                    "(a ReferenceMean built from a named donor set). There is no fallback to the "
                    "evaluation batch's own mean: that changes with the batch and is the row "
                    "itself when the batch is one row.")
            require(reference.name == name,
                    f"{name}: the reference mean was built for {reference.name!r}")
            require(reference.layout == self.layout(name),
                    f"{name}: the reference mean is {reference.layout!r} and this tap is "
                    f"{self.layout(name)!r}")
            reference.verify()
            # The plan owns its own storage. Verified once, here, against the digest it was
            # frozen under; the clone is private to the plan, so no forward needs to re-hash it.
            resolved_reference = reference.frozen_copy()
        self.interventions[name] = {"kind": kind, "positions": positions,
                                    "source": resolved_source, "reference": resolved_reference}
        return self

    def clear_plans(self):
        self.interventions = {}
        return self

    def document(self):
        return {"route": self.route, "taps": self.names(),
                "sites": list(SITES), "interventions": list(INTERVENTIONS),
                "layouts": {name: entry["layout"] for name, entry in sorted(self.taps.items())},
                "position_axes": {name: LAYOUTS[entry["layout"]]
                                  for name, entry in sorted(self.taps.items())},
                "active_plans": {name: plan["kind"]
                                 for name, plan in sorted(self.interventions.items())},
                "naming_rule": ("route/<id>/block<i>/<site>. Stable across routes and independent "
                                "of the upstream module tree, so a re-release does not orphan "
                                "every saved shard."),
                "layout_rule": ("each tap declares whether its tensor is (batch, tokens, "
                                "channels) or (batch, channels, positions). A positional "
                                "intervention addresses the declared sequence axis; without the "
                                "declaration it would ablate a channel and look identical."),
                "mean_ablation_rule": ("replaces the activation with an explicit frozen donor "
                                       "mean (one reference per position, independent of the "
                                       "evaluation batch). A plan without one raises; the "
                                       "evaluation batch's own mean is never used."),
                "active_references": {name: plan["reference"].sha256
                                      for name, plan in sorted(self.interventions.items())
                                      if plan.get("reference") is not None}}


def bind_masked_editor(editor, *, route="ablang2"):
    """Taps on the native AbLang2 encoder blocks. All ``(batch, tokens, channels)``."""
    registry = HookRegistry(route=route)
    for index, block in enumerate(editor.native.AbRep.encoder_blocks):
        registry.register(activation_name(route, index, "resid_pre"), block, site="resid_pre",
                          layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "attn_out"),
                          block.multihead_attention, site="attn_out",
                          layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "mlp_out"),
                          block.intermediate_layer, site="mlp_out", layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "resid_post"), block, site="resid_post",
                          layout=LAYOUT_TOKENS_FIRST)
    return registry


def bind_causal_infiller(infiller, *, route="piggen"):
    """Taps on the GPTNeoX layers of the AR route. All ``(batch, tokens, channels)``."""
    registry = HookRegistry(route=route)
    for index, layer in enumerate(infiller.model.gpt_neox.layers):
        registry.register(activation_name(route, index, "resid_pre"), layer, site="resid_pre",
                          layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "attn_out"), layer.attention,
                          site="attn_out", layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "mlp_out"), layer.mlp, site="mlp_out",
                          layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "resid_post"), layer, site="resid_post",
                          layout=LAYOUT_TOKENS_FIRST)
    return registry


def bind_conditioned_dplm(model, *, route="dplm"):
    """Taps on the native DPLM encoder layers of a :class:`ConditionedDPLM`.

    Every tap is ``(batch, tokens, hidden)`` float32 (``tokens_first``). Sites, for layer ``i`` of
    ``model.native.esm.encoder.layer``:

    * ``resid_pre`` / ``resid_post`` -- the input and output of the ``_Layer`` module. The pre-hook
      replaces ``args[0]`` only; the rotary ``cos``/``sin`` and the key mask pass through untouched.
    * ``attn_out`` -- the dropout module after the attention output projection
      (``layer.attention.output.dropout``): the attention branch's contribution, before the add.
    * ``mlp_out`` -- the dropout module after the FFN output projection (``layer.output.dropout``):
      the MLP branch's contribution, before the add.

    ``_Attention`` / ``_SelfOutput`` / ``_Intermediate`` / ``_Output`` are containers with no
    ``forward``, so a hook on them never fires; the dropout modules are the only call sites of the
    two branch outputs. For each layer, ``resid_post = resid_pre + attn_out + mlp_out`` and
    ``resid_post_i`` is ``resid_pre_(i+1)``. The model's ``last_hidden_state`` is taken AFTER the
    separate ``emb_layer_norm_after``, so it is NOT the last layer's ``resid_post``; the logits are
    that layer-norm and the LM head applied to it.

    What a tap does not say:

    * **Positions are encoded-token coordinates**, including ``<cls>``/``<eos>`` specials. Heavy
      residue ``i`` is at ``1 + i`` but light residue ``j`` is at ``len(heavy) + 4 + j``, so the same
      absolute position names a different biological residue once heavy lengths differ
      (``dplm_policy.dplm_positions``). Any residue-level claim needs that mapping.
    * **A capture holds the latest firing only.** Controlled evaluation (eval mode, no grad, no
      activation checkpointing) fires each tap once per forward. Training/backward with activation
      checkpointing recomputes layers, and iterative decoding runs many forwards; both fire a tap
      repeatedly and the registry keeps the last.
    * **A condition vector is added at the embedding output** and tables start at zero, so a
      donor that differs only in a zero-table condition is identical to the recipient; use donors
      with different sequences unless a nonuniform condition vector is planted on purpose.
    * Hook behaviour here is established on synthetic, randomly initialised CPU models only. It is
      not a statement about pretrained weights, biology, or the memory of a full native run.
    """
    registry = HookRegistry(route=route)
    for index, layer in enumerate(model.native.esm.encoder.layer):
        registry.register(activation_name(route, index, "resid_pre"), layer, site="resid_pre",
                          layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "attn_out"), layer.attention.output.dropout,
                          site="attn_out", layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "mlp_out"), layer.output.dropout,
                          site="mlp_out", layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(route, index, "resid_post"), layer, site="resid_post",
                          layout=LAYOUT_TOKENS_FIRST)
    return registry


def bind_energy_model(model, *, route="dwjs"):
    """Taps on the energy stack and its context tower.

    The convolutional blocks emit ``(batch, channels, positions)``; the context
    tower's transformer layers emit ``(batch, tokens, channels)``. Both are
    registered with their real layout, so "ablate residue 1" means the same thing
    at both. ``attn_out`` is absent from the convolutional stack and is not faked.
    """
    registry = HookRegistry(route=route)
    for index, block in enumerate(model.energy.blocks):
        registry.register(activation_name(route, index, "resid_pre"), block, site="resid_pre",
                          layout=LAYOUT_TOKENS_LAST)
        registry.register(activation_name(route, index, "mlp_out"), block, site="mlp_out",
                          layout=LAYOUT_TOKENS_LAST)
    for index, block in enumerate(getattr(model.context_encoder, "blocks", [])):
        registry.register(activation_name(f"{route}_context", index, "resid_pre"), block,
                          site="resid_pre", layout=LAYOUT_TOKENS_FIRST)
        registry.register(activation_name(f"{route}_context", index, "resid_post"), block,
                          site="resid_post", layout=LAYOUT_TOKENS_FIRST)
    return registry
