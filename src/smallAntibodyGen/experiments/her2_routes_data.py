"""Task streams, batching, and an exposure ledger that does not flatter itself.

The claim this module refuses to let anyone make is "20,000 updates at effective
batch 64 is 1.28 million exposures". That arithmetic is right for exactly one
counter -- row presentations -- and wrong for every other thing the word
"exposure" is used for:

* a preference pair consumes **two** student forwards, so 64 pairs is 128 forwards;
* a local contrast masks the union of the changed positions and scores **both**
  measured alternatives against that one background, so it is two forwards as well;
* a two-site coupled contrast is two *sequential* conditionals, so its forward
  count is two per alternative;
* a construct reused in reconstruction, in a local contrast and in a whole-window
  task accumulates presentations in all three while remaining **one** unique
  measured construct;
* additional masks of the same row create additional supervised targets and **no**
  additional binding measurements.

:class:`ExposureLedger` therefore keeps six counters, reports all six, and derives
none from another. Whether the recorded numbers are *right* is a property of the
call sites, not of this class: the ledger counts what it is told.

The four task streams are the plan's, with their shares as declared *sweep
centres* rather than optima -- and the amendment path is a measured learning curve,
not a borrowed percentage.

Two structural things live here because the objective cannot be correct without
them:

* :class:`MeasurementPayload` carries the measured outcome of every row in its
  own type -- point, interval, censored bound, ordinal class, enrichment -- so
  the objective has something other than a reconstruction loss to apply. A
  training view whose measurement never reaches the batch cannot train a head
  however the coefficients are set.
* :func:`build_local_contrasts` is background-anchored and bounded. An all-pairs
  scan over a mutational library is quadratic in its *output*, not only in its
  runtime, and does not finish at the sizes this campaign enrols.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .her2_routes_conditions import BIN_NULL, Condition
from .her2_routes_serial import (
    HEAVY,
    LIGHT,
    PIGGEN_END_ID,
    PIGGEN_ID_OF_RESIDUE,
    PIGGEN_START_ID,
    ablang_serialization,
    ablang_token_ids,
    causal_serialization,
    emission_order,
    require_bound_vocabulary,
)
from .her2_routes_sources import (
    CENSOR_LEFT,
    CENSOR_NONE,
    CENSOR_RIGHT,
    KIND_ENRICHMENT,
    KIND_INTERVAL,
    KIND_ORDINAL,
    KIND_POINT,
)
from .her2_runtime import require

TASK_RECONSTRUCTION = "paired_reconstruction"
TASK_LOCAL = "measured_local_effects"
TASK_WHOLE = "whole_window_generation"
TASK_HEADS = "observation_heads"
TASKS = (TASK_RECONSTRUCTION, TASK_LOCAL, TASK_WHOLE, TASK_HEADS)

#: Declared initial shares. Sweep centres, adjusted from measured development
#: learning curves and recorded gradient fractions -- never from a borrowed number.
DEFAULT_SHARES = {TASK_RECONSTRUCTION: 0.25, TASK_LOCAL: 0.28125, TASK_WHOLE: 0.28125,
                  TASK_HEADS: 0.1875}

#: The six counters. Their names are the claim; none is computed from another.
COUNTERS = ("optimizer_updates", "microbatch_rows", "student_forwards", "unique_constructs",
            "measurement_presentations", "supervised_targets")

#: What a per-task, per-source or per-family bucket carries. ``optimizer_updates`` is absent on
#: purpose: one update draws many microbatches from many tasks and sources, so no bucket owns an
#: update and a per-bucket figure would not sum to the global one. The distinct counts are
#: per bucket and are NOT additive either -- an identity presented under two sources counts in both.
BUCKET_COUNTERS = ("microbatch_rows", "student_forwards", "supervised_targets",
                   "measurement_presentations", "unique_constructs", "unique_measured_constructs")
#: The bucket counters whose sum over a bucket kind must equal the global counter of that name.
ADDITIVE_BUCKET_COUNTERS = ("microbatch_rows", "student_forwards", "supervised_targets",
                            "measurement_presentations")
#: The bucket counters that need the bucket's identity set. A pre-v2 state cannot supply them.
IDENTITY_BUCKET_COUNTERS = ("measurement_presentations", "unique_constructs",
                            "unique_measured_constructs")
BUCKET_KINDS = ("task", "source", "family")
#: Version 1 kept global identity sets and a per-(task, identity) count, so a per-source or
#: per-family identity set could not be rebuilt from it. Version 2 persists every bucket's
#: identity map (identity -> presentations, measured presentations).
LEDGER_STATE_VERSION = 2
#: Where a batch-level total goes when it cannot be split across the batch's sources.
UNATTRIBUTED_SOURCE = "(batch total; not attributable to a source)"


def _count(value, name):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            f"Exposure count {name} must be an integer, got {value!r}")
    require(int(value) >= 0, f"Exposure counts cannot be negative: {name} is {int(value)}")
    return int(value)


@dataclass(frozen=True)
class ExposureContribution:
    """One source's share of one microbatch: rows, student forwards, targets and identities.

    ``identities`` lists one entry per presentation, so ``[a, a, b]`` is three presentations of
    two distinct constructs. A batch that draws rows from several sources is several
    contributions, one per source -- it is never attributed wholesale to its first row's source.
    ``measured`` says whether those presentations are measured-identity presentations.
    """

    source: str
    rows: int = 0
    forwards: int = 0
    targets: int = 0
    identities: tuple = ()
    measured: bool = False
    family: object = None

    def __post_init__(self):
        for name in ("rows", "forwards", "targets"):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "identities", tuple(str(item) for item in self.identities))
        object.__setattr__(self, "measured", bool(self.measured))
        if self.family is not None:
            object.__setattr__(self, "family", str(self.family))

    def measured_presentations(self):
        return len(self.identities) if self.measured else 0

    def document(self):
        return {"source": self.source, "family": self.family, "rows": self.rows,
                "forwards": self.forwards, "targets": self.targets,
                "identity_presentations": len(self.identities),
                "distinct_identities": len(set(self.identities)),
                "measured": self.measured,
                "measured_presentations": self.measured_presentations()}


def _per_row(values, rows, name):
    """One validated count per row. A scalar and EACH element of a sequence pass :func:`_count`
    before anything is aggregated, so ``[.5, .5]`` and ``[-1, 2]`` are refused rather than
    truncated or netted into a plausible total."""
    if isinstance(values, np.ndarray) and values.ndim == 0:
        values = values[()]
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        return [_count(values, name)] * rows
    values = [_count(value, f"{name}[{index}]") for index, value in enumerate(values)]
    require(len(values) == rows, f"{name}: {len(values)} per-row counts for {rows} rows")
    return values


def contributions_by_source(*, sources, identities, forwards, targets, measured, family=None):
    """Group aligned per-row counts into one :class:`ExposureContribution` per source.

    ``sources``/``identities`` are one entry per row. ``forwards`` and ``targets`` are either
    per-row sequences or one integer applied to every row. Sources keep their first-appearance
    order and every row keeps its OWN source: a mixed batch is attributed row by row.
    """
    sources, identities = list(sources), list(identities)
    rows = len(sources)
    require(len(identities) == rows,
            f"{len(identities)} identities for {rows} rows; they are aligned one per row")
    forwards, targets = _per_row(forwards, rows, "forwards"), _per_row(targets, rows, "targets")
    order, blocks = [], {}
    for index in range(rows):
        key = str(sources[index])
        if key not in blocks:
            blocks[key] = {"rows": 0, "forwards": 0, "targets": 0, "identities": []}
            order.append(key)
        block = blocks[key]
        block["rows"] += 1
        block["forwards"] += forwards[index]
        block["targets"] += targets[index]
        block["identities"].append(identities[index])
    return tuple(ExposureContribution(
        source=key, rows=blocks[key]["rows"], forwards=blocks[key]["forwards"],
        targets=blocks[key]["targets"], identities=tuple(blocks[key]["identities"]),
        measured=measured, family=family) for key in order)


def with_extra_forwards(contributions, extras):
    """Add forwards that belong to a source but to no extra row (a quartet's joint, say).

    ``extras`` is ``[(source, forwards), ...]``. The forwards land on that source's contribution;
    a source with no rows in the batch gets a rows-free contribution of its own, so the batch
    total still equals the sum of its parts.
    """
    out = list(contributions)
    for source, count in extras:
        count = _count(count, "forwards")
        if not count:
            continue
        for position, entry in enumerate(out):
            if entry.source == str(source):
                out[position] = ExposureContribution(
                    source=entry.source, rows=entry.rows, forwards=entry.forwards + count,
                    targets=entry.targets, identities=entry.identities,
                    measured=entry.measured, family=entry.family)
                break
        else:
            family = out[0].family if out else None
            out.append(ExposureContribution(source=str(source), forwards=count, family=family))
    return tuple(out)


class ExposureLedger:
    """Six independent counters plus the per-task, per-source and per-family breakdowns.

    The six globals are recorded once per microbatch. The breakdowns are recorded from the same
    per-source contributions, so each additive bucket counter sums to its global (see
    :meth:`additivity`). Every bucket also holds the identities it presented, with their
    multiplicity, which is what gives it measured presentations and distinct counts of its own.

    A ledger restored from a version-1 state keeps its globals but cannot know which source or
    family each identity was presented under; its identity-derived bucket figures are reported as
    ``None`` (unavailable), never as an invented zero.
    """

    def __init__(self):
        self.totals = {name: 0 for name in COUNTERS}
        self.by_task = {}
        self.by_source = {}
        self.by_family = {}
        self.constructs = set()
        self.measured_constructs = set()
        #: kind -> bucket name -> {identity: [presentations, measured presentations]}
        self._identities = {kind: {} for kind in BUCKET_KINDS}
        #: Per bucket kind: is every identity-derived figure exact? False after a v1 restore.
        self.breakdown_complete = {kind: True for kind in BUCKET_KINDS}
        self.restored_from = None
        self.family_unattributed_rows = 0

    def _counters(self, kind):
        return {"task": self.by_task, "source": self.by_source, "family": self.by_family}[kind]

    @property
    def presentations(self):
        """``{(task, identity): presentations}``, derived from the per-task identity maps."""
        return {(task, identity): int(counts[0])
                for task, identities in self._identities["task"].items()
                for identity, counts in identities.items()}

    def record(self, *, task, source=None, rows=None, forwards=None, targets=None, identities=(),
               measured=False, update=False, family=None, contributions=None):
        """Record one microbatch, either as per-source ``contributions`` or as one source.

        The single-source form (``source``, ``rows``, ``forwards``, ``targets``, ``identities``)
        is one contribution. ``family`` is the default for a contribution that names none.
        Global counters are incremented once per contribution, from the same numbers the
        breakdowns receive, so the two cannot disagree.
        """
        require(task in TASKS, f"Unknown task {task!r}")
        if contributions is None:
            require(source is not None and rows is not None and forwards is not None
                    and targets is not None,
                    "record() needs either contributions or source, rows, forwards and targets")
            contributions = (ExposureContribution(
                source=source, rows=rows, forwards=forwards, targets=targets,
                identities=tuple(identities), measured=measured, family=family),)
        else:
            require(source is None and rows is None and forwards is None and targets is None
                    and not tuple(identities) and not measured,
                    "record() takes contributions OR the single-source arguments, not both")
            contributions = tuple(contributions)
            require(all(isinstance(entry, ExposureContribution) for entry in contributions),
                    "contributions must be ExposureContribution records")
        if update:
            self.totals["optimizer_updates"] += 1
        for entry in contributions:
            self._add(task, entry, default_family=family)
        self.totals["unique_constructs"] = len(self.constructs)
        return self

    def record_outcome(self, *, task, outcome, update=False):
        """Record a :class:`TaskOutcome` through its own per-source contributions."""
        return self.record(task=task, contributions=outcome.exposure(), update=update)

    def _add(self, task, entry, *, default_family=None):
        family = entry.family if entry.family is not None else default_family
        self.totals["microbatch_rows"] += entry.rows
        self.totals["student_forwards"] += entry.forwards
        self.totals["supervised_targets"] += entry.targets
        self.totals["measurement_presentations"] += entry.measured_presentations()
        buckets = [("task", str(task)), ("source", entry.source)]
        if family is not None:
            buckets.append(("family", str(family)))
        else:
            self.family_unattributed_rows += entry.rows
        for kind, name in buckets:
            block = self._counters(kind).setdefault(name, {key: 0 for key in BUCKET_COUNTERS})
            block["microbatch_rows"] += entry.rows
            block["student_forwards"] += entry.forwards
            block["supervised_targets"] += entry.targets
            block["measurement_presentations"] += entry.measured_presentations()
            seen = self._identities[kind].setdefault(name, {})
            for identity in entry.identities:
                counts = seen.get(identity)
                if counts is None:
                    counts = seen[identity] = [0, 0]
                    block["unique_constructs"] += 1
                counts[0] += 1
                if entry.measured:
                    if counts[1] == 0:
                        block["unique_measured_constructs"] += 1
                    counts[1] += 1
        for identity in entry.identities:
            self.constructs.add(identity)
            if entry.measured:
                self.measured_constructs.add(identity)

    def additivity(self):
        """Do the additive bucket counters sum to their globals? Reported, never assumed.

        ``additive`` is ``None`` where the question has no answer: an identity-derived counter of
        a kind restored from a version-1 state, and the family kind when some record named no family.
        """
        out = {}
        for kind in BUCKET_KINDS:
            blocks = list(self._counters(kind).values())
            entry = {}
            for key in ADDITIVE_BUCKET_COUNTERS:
                summed, total = sum(int(block[key]) for block in blocks), int(self.totals[key])
                answerable = (self.breakdown_complete[kind]
                              or key not in IDENTITY_BUCKET_COUNTERS) and not (
                    kind == "family" and self.family_unattributed_rows)
                entry[key] = {"sum_of_buckets": summed, "total": total,
                              "additive": (summed == total) if answerable else None}
            out[kind] = entry
        return out

    def require_additive(self):
        """Raise if a bucket kind that can be checked does not sum to the global counters."""
        broken = [f"{kind}/{key}: buckets sum to {block['sum_of_buckets']}, total "
                  f"{block['total']}"
                  for kind, entry in self.additivity().items()
                  for key, block in entry.items() if block["additive"] is False]
        require(not broken, "The exposure ledger's breakdowns do not sum to its totals: "
                + "; ".join(broken))
        return True

    def state(self):
        """A JSON-able, versioned snapshot a resume restores exactly.

        Identities are interned once in ``identity_table`` and every bucket refers to them by
        position, so a per-source and a per-family identity map do not each repeat the sequence
        strings. Sets become sorted lists; the maps keep their presentation multiplicities.
        """
        table = sorted(self.constructs)
        position = {identity: index for index, identity in enumerate(table)}
        identities = {}
        for kind in BUCKET_KINDS:
            identities[kind] = {
                name: [[position[identity], int(counts[0]), int(counts[1])]
                       for identity, counts in sorted(seen.items(),
                                                      key=lambda item: position[item[0]])]
                for name, seen in sorted(self._identities[kind].items())}
        return {"version": LEDGER_STATE_VERSION, "totals": dict(self.totals),
                "by_task": {key: dict(value) for key, value in sorted(self.by_task.items())},
                "by_source": {key: dict(value) for key, value in sorted(self.by_source.items())},
                "by_family": {key: dict(value) for key, value in sorted(self.by_family.items())},
                "identity_table": table,
                "measured_constructs": sorted(position[item] for item in self.measured_constructs),
                "identities": identities,
                "breakdown_complete": dict(self.breakdown_complete),
                "family_unattributed_rows": int(self.family_unattributed_rows),
                "restored_from": None if self.restored_from is None else dict(self.restored_from)}

    @classmethod
    def restore(cls, state):
        """Rebuild a ledger from :meth:`state`. A resume that skipped this restarts
        every counter at zero, and the matched-exposure check between control arms
        then compares a resumed arm's partial count against an uninterrupted arm's
        full one and refuses both.

        A version-1 state (no ``version`` key) restores its globals and the per-task row, forward
        and target counts, and marks every breakdown incomplete: its identity sets were global,
        so which source or family presented which identity is not recoverable.
        """
        block = dict(state or {})
        version = int(block.get("version", 1))
        require(version in (1, LEDGER_STATE_VERSION),
                f"Unknown exposure-ledger state version {version}; this code reads 1 and "
                f"{LEDGER_STATE_VERSION}")
        ledger = cls()
        ledger.totals = {name: int(dict(block.get("totals") or {}).get(name, 0))
                         for name in COUNTERS}
        if version == 1:
            return ledger._restore_v1(block)
        table = [str(item) for item in block.get("identity_table") or ()]
        ledger.constructs = set(table)
        ledger.measured_constructs = {table[int(index)]
                                      for index in block.get("measured_constructs") or ()}
        ledger.breakdown_complete = {
            kind: bool(dict(block.get("breakdown_complete") or {}).get(kind, True))
            for kind in BUCKET_KINDS}
        ledger.family_unattributed_rows = int(block.get("family_unattributed_rows", 0))
        ledger.restored_from = (None if block.get("restored_from") is None
                                else dict(block["restored_from"]))
        stored = dict(block.get("identities") or {})
        for kind in BUCKET_KINDS:
            counters = ledger._counters(kind)
            for name, values in dict(block.get(f"by_{kind}") or {}).items():
                counters[str(name)] = {key: int(dict(values).get(key, 0))
                                       for key in BUCKET_COUNTERS}
            for name, entries in dict(stored.get(kind) or {}).items():
                ledger._identities[kind][str(name)] = {
                    table[int(index)]: [int(presented), int(measured)]
                    for index, presented, measured in entries}
            if ledger.breakdown_complete[kind]:
                for name, values in counters.items():
                    seen = ledger._identities[kind].get(name, {})
                    require(len(seen) == values["unique_constructs"],
                            f"The restored {kind} bucket {name!r} counts "
                            f"{values['unique_constructs']} distinct constructs but holds "
                            f"{len(seen)}; the ledger state is inconsistent and is not resumed.")
        return ledger

    def _restore_v1(self, block):
        for kind in BUCKET_KINDS:
            for name, values in dict(block.get(f"by_{kind}") or {}).items():
                entry = {key: 0 for key in BUCKET_COUNTERS}
                for key in ("microbatch_rows", "student_forwards", "supervised_targets"):
                    entry[key] = int(dict(values).get(key, 0))
                self._counters(kind)[str(name)] = entry
        self.constructs = {str(item) for item in block.get("constructs") or ()}
        self.measured_constructs = {str(item) for item in block.get("measured_constructs") or ()}
        for task, identity, count in block.get("presentations") or ():
            seen = self._identities["task"].setdefault(str(task), {})
            seen[str(identity)] = [int(count), 0]
            self.constructs.add(str(identity))
        for task, seen in self._identities["task"].items():
            self.by_task.setdefault(task, {key: 0 for key in BUCKET_COUNTERS})
            self.by_task[task]["unique_constructs"] = len(seen)
        self.breakdown_complete = {kind: False for kind in BUCKET_KINDS}
        # A version-1 record could name no family, so a valid old ledger's family buckets may sum
        # to less than its rows. The rows no family owns are the difference -- inferred, never
        # invented per family -- and a family buckets-above-total state cannot have been written.
        family_blocks = list(self.by_family.values())
        for key in ("microbatch_rows", "student_forwards", "supervised_targets"):
            summed = sum(int(entry[key]) for entry in family_blocks)
            require(summed <= int(self.totals[key]),
                    f"The legacy ledger's by_family buckets hold {summed} {key} but its totals "
                    f"record only {self.totals[key]}; family buckets cannot exceed the total, so "
                    "the state is inconsistent and is not resumed.")
        self.family_unattributed_rows = (
            int(self.totals["microbatch_rows"])
            - sum(int(entry["microbatch_rows"]) for entry in family_blocks))
        note = ("restored from a pre-v2 ledger. Globals and the per-task, per-source and "
                "per-family row, forward and target counts carry over; the identity-derived "
                "bucket figures are unavailable because version 1 kept only global identity "
                "sets. Its per-source rows were attributed to each batch's first row's "
                "source, so they are not row-exact either.")
        if self.family_unattributed_rows:
            note += (f" {self.family_unattributed_rows} of its rows were recorded with no family; "
                     "that count is inferred as total rows minus the family buckets' rows and "
                     "is not split across families, so family additivity is unanswerable "
                     "(None), not a broken sum.")
        self.restored_from = {"version": 1, "note": note,
                              "family_unattributed_rows": self.family_unattributed_rows}
        return self

    def family_shares(self):
        """The REALISED per-family row share, against which the declared mix is checked."""
        total = sum(int(block["microbatch_rows"]) for block in self.by_family.values())
        if total == 0:
            return {}
        return {name: int(block["microbatch_rows"]) / total
                for name, block in sorted(self.by_family.items())}

    def reused_constructs(self):
        """Constructs presented under more than one task -- the double-count risk."""
        tasks_per = {}
        for task, identities in self._identities["task"].items():
            for identity in identities:
                tasks_per.setdefault(identity, set()).add(task)
        return {identity: sorted(tasks) for identity, tasks in tasks_per.items()
                if len(tasks) > 1}

    def _bucket_document(self, kind):
        """The buckets of one kind. An identity-derived figure of an incomplete kind is ``None``."""
        complete = self.breakdown_complete[kind]
        out = {}
        for name, block in sorted(self._counters(kind).items()):
            entry = dict(block)
            if not complete:
                for key in IDENTITY_BUCKET_COUNTERS:
                    entry[key] = None
            entry["breakdown_complete"] = complete
            out[name] = entry
        return out

    def document(self):
        reused = self.reused_constructs()
        rows = self.totals["microbatch_rows"]
        forwards = self.totals["student_forwards"]
        return {"state_version": LEDGER_STATE_VERSION, "totals": dict(self.totals),
                "by_task": self._bucket_document("task"),
                "by_source": self._bucket_document("source"),
                "by_family": self._bucket_document("family"),
                "breakdown_complete": dict(self.breakdown_complete),
                "additivity": self.additivity(),
                "restored_from": None if self.restored_from is None else dict(self.restored_from),
                "bucket_rule": (
                    "each bucket is recorded from the same per-source contributions as the "
                    "totals, so rows, forwards, targets and measured presentations sum to them. "
                    "Distinct-construct counts are per bucket and are not additive: an identity "
                    "presented under two sources counts in both. A figure that a restored "
                    "pre-v2 ledger cannot know is None, not zero. optimizer_updates is a "
                    "global only; no bucket owns an update."),
                "realised_family_shares": self.family_shares(),
                "unique_measured_constructs": len(self.measured_constructs),
                "constructs_reused_across_tasks": len(reused),
                "forwards_per_row": None if rows == 0 else forwards / rows,
                "accounting_rule": (
                    "six counters, none derived from another. optimizer_updates x effective batch "
                    "equals microbatch_rows and NOTHING else: a preference pair and a local "
                    "contrast each consume two student forwards, so student_forwards exceeds "
                    "microbatch_rows whenever those tasks are scheduled. Additional masks of one "
                    "row create additional supervised_targets and no additional measurements."),
                "forwards_per_row_note": (
                    "1.0 would mean every scheduled task was single-forward. Any figure above 1 "
                    "is the measured cost of the two-forward tasks, and it is why an 'exposures' "
                    "number quoted as updates x batch understates the compute.")}


# ---------------------------------------------------------------------------
# training views
# ---------------------------------------------------------------------------

@dataclass
class CausalBatch:
    """A padded batch for the AR route. Every field is an aligned array."""

    tokens: object
    query_slots: object
    pointers: object
    targets: object
    condition_indices: object
    attention_mask: object = None
    target_mask: object = None
    identities: tuple = ()
    sources: tuple = ()

    def document(self):
        return {"rows": int(np.asarray(self.tokens).shape[0]),
                "length": int(np.asarray(self.tokens).shape[1]),
                "queries": int(np.asarray(self.query_slots).shape[1]),
                "sources": sorted(set(self.sources))}


@dataclass
class MaskedBatch:
    """A padded batch for the masked-editor route."""

    tokens: object
    query_slots: object
    targets: object
    condition_indices: object
    attention_mask: object = None
    target_mask: object = None
    identities: tuple = ()
    sources: tuple = ()


def pad_rows(rows, *, value, width=None):
    """Right-pad ragged integer rows to a rectangle, returning ``(array, mask)``."""
    lengths = [len(row) for row in rows]
    require(lengths, "Nothing to pad")
    width = int(width) if width is not None else max(lengths)
    require(width >= max(lengths),
            f"A row of length {max(lengths)} does not fit a declared width of {width}; truncating "
            "here would drop a light chain silently")
    out = np.full((len(rows), width), int(value), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for position, row in enumerate(rows):
        out[position, :len(row)] = np.asarray(row, dtype=np.int64)
        mask[position, :len(row)] = 1
    return out, mask


def collate_causal(views, *, pad_id, condition_spec, query_width=None):
    """Build a :class:`CausalBatch` from ``(construct, serialization, condition)`` views.

    Padding is right-side and the attention mask marks the real tokens, so a
    prediction agrees alone and in a padded batch (I8). Query slots are padded by
    **repeating the last real slot** and masked out in the loss: an out-of-range
    slot would index into the padding and produce a finite, wrong number.
    """
    tokens, slots, pointers, targets, conditions, identities, sources = [], [], [], [], [], [], []
    for construct, serialization, condition in views:
        tokens.append(list(serialization.tokens))
        slots.append(list(serialization.query_slots))
        pointers.append(list(serialization.pointers))
        targets.append(list(serialization.targets))
        conditions.append(list(condition_spec.encode(condition)))
        identities.append(construct.identity())
        sources.append(construct.source)
    token_array, attention = pad_rows(tokens, value=int(pad_id))
    width = int(query_width) if query_width is not None else max(len(row) for row in slots)
    slot_array = np.zeros((len(slots), width), dtype=np.int64)
    pointer_array = np.zeros_like(slot_array)
    target_array = np.zeros_like(slot_array)
    target_mask = np.zeros((len(slots), width), dtype=np.int64)
    for row, (slot_row, pointer_row, target_row) in enumerate(zip(slots, pointers, targets)):
        require(len(slot_row) == len(target_row) or not target_row,
                "Query slots and targets disagree in length")
        count = len(slot_row)
        require(count > 0, "A training view with no query has nothing to learn from (I6)")
        slot_array[row, :count] = slot_row
        slot_array[row, count:] = slot_row[-1]
        pointer_array[row, :count] = pointer_row[:count]
        pointer_array[row, count:] = pointer_row[count - 1] if pointer_row else 0
        if target_row:
            target_array[row, :count] = target_row
            target_mask[row, :count] = 1
    return CausalBatch(tokens=token_array, query_slots=slot_array, pointers=pointer_array,
                       targets=target_array, condition_indices=np.asarray(conditions,
                                                                         dtype=np.int64),
                       attention_mask=attention, target_mask=target_mask,
                       identities=tuple(identities), sources=tuple(sources))


def collate_masked(views, *, pad_id, condition_spec, query_width=None):
    """The same for the masked editor. Padding uses the native pad token id 21."""
    tokens, slots, targets, conditions, identities, sources = [], [], [], [], [], []
    for construct, serialization, condition in views:
        tokens.append(list(serialization.tokens))
        slots.append(list(serialization.query_slots))
        targets.append(list(serialization.targets))
        conditions.append(list(condition_spec.encode(condition)))
        identities.append(construct.identity())
        sources.append(construct.source)
    token_array, attention = pad_rows(tokens, value=int(pad_id))
    width = int(query_width) if query_width is not None else max(len(row) for row in slots)
    slot_array = np.zeros((len(slots), width), dtype=np.int64)
    target_array = np.zeros_like(slot_array)
    target_mask = np.zeros((len(slots), width), dtype=np.int64)
    for row, (slot_row, target_row) in enumerate(zip(slots, targets)):
        count = len(slot_row)
        require(count > 0, "A masked training view with no masked query (I6)")
        slot_array[row, :count] = slot_row
        slot_array[row, count:] = slot_row[-1]
        target_array[row, :count] = target_row[:count]
        target_mask[row, :count] = 1
    return MaskedBatch(tokens=token_array, query_slots=slot_array, targets=target_array,
                       condition_indices=np.asarray(conditions, dtype=np.int64),
                       attention_mask=attention, target_mask=target_mask,
                       identities=tuple(identities), sources=tuple(sources))


# ---------------------------------------------------------------------------
# the four task streams
# ---------------------------------------------------------------------------

HEAVY_CHAIN = HEAVY
LIGHT_CHAIN = LIGHT

#: Contrast kinds. Each names a different measured quantity, and they are never
#: pooled into "a pair": a single-site effect against a background, two
#: alternatives at one position, and the quartet that yields an interaction are
#: three different things to learn.
CONTRAST_SINGLE = "single_vs_background"
CONTRAST_ALTERNATIVES = "same_position_alternatives"
CONTRAST_COUPLED = "coupled_quartet"
CONTRAST_KINDS = (CONTRAST_SINGLE, CONTRAST_ALTERNATIVES, CONTRAST_COUPLED)


CENSOR_MIXED = "mixed"

#: The declared partitions a source may assign a row to. ``train`` is the only
#: one any adaptation consumer may read; ``protected`` is the conservative label
#: for a row whose held-out status cannot be separated from an eligibility reject.
PARTITION_TRAIN = "train"
PARTITION_DEVELOPMENT = "development"
PARTITION_PROTECTED = "protected"
PARTITIONS = (PARTITION_TRAIN, PARTITION_DEVELOPMENT, PARTITION_PROTECTED)


@dataclass(frozen=True)
class ReplicateMeasurement:
    """One measured replicate, with its **own** censoring and its own uncertainty.

    The field that makes this class necessary is ``censoring``. An earlier
    revision summarised a construct by taking the *first* replicate's censoring
    and averaging the finite values, so a construct with one floor replicate and
    two point replicates was reported as an exact point at the mean of the two --
    a number the assay never produced, with the floor replicate's information
    discarded in the one direction that flatters affinity. CR9114's H3 and
    influenza-B partitions and Adams' doubly censored Tite-Seq window both
    produce exactly that mixture, so the head likelihood consumes the replicates
    individually and this record is what carries them there.
    """

    value: object
    lower: object
    upper: object
    censoring: str
    uncertainty: object = None
    replicate: object = None
    measurement_id: str = ""

    @property
    def is_point(self):
        return self.censoring == CENSOR_NONE and self.value is not None

    @property
    def exactness(self):
        """Why this replicate is, or is not, an exact finite point.

        ``"exact"`` needs an uncensored finite value whose bounds -- declared, or
        implied by :meth:`bounds` when the source supplied none -- both equal it.
        An absent explicit bound is not a reason to refuse a finite uncensored
        point: most point sources publish no interval at all. A genuinely
        non-degenerate interval is, because a value with room around it is a
        statement about where the measurement fell, not a number the assay produced.
        """
        if self.censoring != CENSOR_NONE:
            return "censored"
        try:
            value = float(self.value)
        except (TypeError, ValueError):
            return "nonfinite"
        if not math.isfinite(value):
            return "nonfinite"
        low, high = self.bounds()
        try:
            low, high = float(low), float(high)
        except (TypeError, ValueError):
            return "nonfinite"
        if not (math.isfinite(low) and math.isfinite(high)):
            return "nonfinite"
        if abs(low - value) > 1e-9 or abs(high - value) > 1e-9:
            return "nondegenerate_bounds"
        return "exact"

    @property
    def is_exact_point(self):
        return self.exactness == "exact"

    def bounds(self):
        """``(lower, upper)`` with ``None`` for an open side. One path, four kinds."""
        if self.censoring == CENSOR_LEFT:
            return None, (self.upper if self.upper is not None else self.value)
        if self.censoring == CENSOR_RIGHT:
            return (self.lower if self.lower is not None else self.value), None
        low = self.lower if self.lower is not None else self.value
        high = self.upper if self.upper is not None else self.value
        return low, high

    def document(self):
        low, high = self.bounds()
        return {"value": None if self.value is None else float(self.value),
                "lower": None if low is None else float(low),
                "upper": None if high is None else float(high),
                "censoring": self.censoring, "replicate": self.replicate,
                "uncertainty": None if self.uncertainty is None else float(self.uncertainty)}


@dataclass
class ConstructSummary:
    """One measured construct in one assay, with every replicate kept.

    Contrast enumeration runs over constructs, not over replicate rows: a scan
    with three replicates otherwise produces nine copies of every pair and the
    reported "number of local contrasts" becomes a statement about the replicate
    count. What the summary does **not** do is collapse the measurements -- they
    are held in :attr:`measurements` with their individual censoring, and the
    aggregate properties below are derived from them and labelled.

    ``region`` is the **declared common support**: the positions that vary across
    this construct's whole (source, parent) group. It is deliberately not the
    construct's own mutated positions, because a whole-window task scored on a
    mutation-dependent region has a different support for every row and its log
    probabilities are not comparable. ``mutated_sites`` carries the row's own
    edits for the local contrasts that need them.
    """

    construct: object
    study: str
    assay: str
    target: str
    parent: str
    family: str
    kind: str
    measurements: tuple
    region: tuple = ()
    mutated_sites: tuple = ()
    partition: str = PARTITION_TRAIN
    scale: str = ""
    source: str = ""
    ordinal_class: object = None
    raw: dict = field(default_factory=dict)

    # -- derived aggregates ------------------------------------------------
    @property
    def values(self):
        return tuple(row.value for row in self.measurements)

    @property
    def replicates(self):
        return tuple(row.replicate for row in self.measurements)

    @property
    def point_values(self):
        return tuple(float(row.value) for row in self.measurements if row.is_point)

    @property
    def value(self):
        """The qualified point estimate: the mean of the **uncensored** replicates.

        ``None`` when every replicate is censored. A censored replicate has no
        point value to average, and inventing one is the defect this property
        exists to refuse.
        """
        points = self.point_values
        return float(np.mean(points)) if points else None

    @property
    def exact_status(self):
        """Whether the construct has an exact value, and if not, the counted reason.

        ``"exact"`` only when **every** replicate is an exact finite point. The other
        answers are ``empty``, ``ordinal`` (a class has no cardinal difference),
        ``censored`` (nothing but censored replicates), ``mixed_censoring`` (some
        exact, some censored), ``nonfinite`` and ``nondegenerate_bounds``.
        """
        if not self.measurements:
            return "empty"
        if self.kind == KIND_ORDINAL:
            return "ordinal"
        statuses = {row.exactness for row in self.measurements}
        problems = statuses - {"exact"}
        if not problems:
            return "exact"
        if "nonfinite" in problems:
            return "nonfinite"
        if "nondegenerate_bounds" in problems:
            return "nondegenerate_bounds"
        return "mixed_censoring" if "exact" in statuses else "censored"

    @property
    def exact_value(self):
        """The mean of the replicates **when all of them are exact points**, else ``None``.

        This is what a local numeric margin or a quartet target may difference.
        :attr:`value` is the qualified point estimate -- the mean of the *uncensored*
        replicates -- which for a construct with one floor replicate and two points
        is a number the assay never produced. Subtracting two of those and fitting a
        decoder's log odds to the result trains on a margin nobody measured, so the
        local-contrast builders read this property instead.

        The number is in the row's own ``scale``. Nothing here asserts pKD units: an
        enrichment ratio is exact in enrichment units and its consumers branch on
        ``kind``/``scale``; a pKD-only consumer checks the scale itself
        (``her2_routes_pairs``). An ordinal construct has no exact value at all.
        Observation likelihoods read :attr:`measurements`, not this.
        """
        if self.exact_status != "exact":
            return None
        return float(np.mean([float(row.value) for row in self.measurements]))

    @property
    def support(self):
        return len(self.point_values)

    @property
    def censored_support(self):
        return len([row for row in self.measurements if not row.is_point])

    @property
    def censoring(self):
        kinds = {row.censoring for row in self.measurements}
        if len(kinds) == 1:
            return next(iter(kinds))
        return CENSOR_MIXED

    @property
    def interval(self):
        """The pooled bound, taken as the **union** of the replicate intervals.

        The union, not the intersection: it is the widest statement the
        replicates jointly support, so a construct whose replicates straddle a
        bin cut is reported as unbinnable instead of being placed by whichever
        replicate was read first.
        """
        lowers, uppers = [], []
        open_low = open_high = False
        for row in self.measurements:
            low, high = row.bounds()
            if low is None:
                open_low = True
            else:
                lowers.append(float(low))
            if high is None:
                open_high = True
            else:
                uppers.append(float(high))
        return (None if open_low or not lowers else min(lowers),
                None if open_high or not uppers else max(uppers))

    @property
    def uncertainty(self):
        """The mean REPORTED uncertainty, or ``None``. Never the replicate spread."""
        reported = [float(row.uncertainty) for row in self.measurements
                    if row.uncertainty is not None]
        return float(np.mean(reported)) if reported else None

    @property
    def replicate_spread(self):
        """Sample standard deviation of the point replicates. A different quantity."""
        points = self.point_values
        return float(np.std(points, ddof=1)) if len(points) > 1 else None

    @property
    def observation_scale(self):
        """The noise the head should assume: reported SE and replicate spread combined.

        Returns ``None`` when neither is available, which is the signal for the
        head to fall back on its learned per-assay scale alone.
        """
        parts = [value for value in (self.uncertainty, self.replicate_spread)
                 if value is not None and float(value) > 0]
        if not parts:
            return None
        return float(math.sqrt(sum(float(value) ** 2 for value in parts)))

    @property
    def has_edit_map(self):
        return bool(self.region)

    @property
    def is_training(self):
        return self.partition == PARTITION_TRAIN

    @property
    def population_key(self):
        """The same key ``Observation.population_key`` carries. Bins attach to it."""
        return (self.construct.identity(), self.study, self.assay, self.target)

    @property
    def identity(self):
        return self.construct.identity()

    def document(self):
        low, high = self.interval
        return {"identity": self.construct.identity()[:32], "assay": self.assay,
                "target": self.target, "parent": self.parent, "kind": self.kind,
                "value": self.value, "replicate_support": self.support,
                "censored_support": self.censored_support,
                "censoring": self.censoring, "partition": self.partition,
                "interval": [low, high],
                "reported_uncertainty": self.uncertainty,
                "replicate_spread": self.replicate_spread,
                "region_sites": len(self.region), "mutated_sites": len(self.mutated_sites),
                "replicates": [row.document() for row in self.measurements[:6]]}


def summarise_observations(observations):
    """Collapse replicate observations into one summary per (construct, assay, target).

    Every replicate survives as a :class:`ReplicateMeasurement` carrying its own
    censoring, bounds and uncertainty. What does not survive is the replicate
    *row*, because every pairwise enumeration downstream would multiply by it.
    """
    grouped = {}
    for record in observations:
        key = (record.identity, record.study, record.assay_key, record.target)
        grouped.setdefault(key, []).append(record)
    out = []
    for (_identity, study, assay, target), rows in sorted(grouped.items()):
        first = rows[0]
        measurements = []
        for row in rows:
            compact = row.raw.get("replicates") or ()
            if compact and all(isinstance(rep, dict) and "log10_kd_nm" in rep
                               for rep in compact):
                # Li stores one compact row per construct/assay. Its aggregate
                # uncertainty is sample SD, not an SE attached to each replicate.
                for rep in compact:
                    value = 9.0 - float(rep["log10_kd_nm"])
                    measurements.append(ReplicateMeasurement(
                        value=value, lower=value, upper=value, censoring=CENSOR_NONE,
                        replicate=rep["replicate"], uncertainty=None,
                        measurement_id=f"{row.raw.get('measurement_id', '')}:{rep['replicate']}"))
            else:
                measurements.append(ReplicateMeasurement(
                    value=row.value,
                    lower=(row.interval[0] if row.interval else None),
                    upper=(row.interval[1] if row.interval else None),
                    censoring=row.censoring, uncertainty=row.uncertainty,
                    replicate=row.replicate,
                    measurement_id=str(row.raw.get("measurement_id", ""))))
        raw = dict(first.raw)
        out.append(ConstructSummary(
            construct=first.construct, study=study, assay=assay, target=target,
            parent=first.parent, family=first.family, kind=first.kind,
            measurements=tuple(measurements),
            region=tuple(first.construct.sites),
            mutated_sites=tuple(tuple(site) for site in (raw.get("mutated_sites") or ())),
            partition=str(raw.get("partition") or PARTITION_TRAIN),
            scale=first.scale, source=first.source,
            ordinal_class=first.ordinal_class,
            raw={key: raw[key] for key in ("withheld_as", "mutation_map_known")
                 if key in raw}))
    return tuple(out)


@dataclass
class LocalContrast:
    """Measured alternatives sharing one masked background, with their numbers.

    ``sites`` is the **union** of the positions that differ, so every alternative
    sees the identical visible background. Without the union the model could tell
    the alternatives apart from their inputs, and the contrast would measure
    nothing about its own preference.

    ``margin`` is **always** ``left_value - right_value`` on the declared scale.
    That one rule is worth stating because an earlier revision had three: a
    single-site contrast carried ``right - left``, a same-position pair carried
    ``left - right``, and a quartet carried the double's effect against the
    background. Three meanings under one field name is a sign error waiting for a
    consumer, and the numeric decoder objective consumes this field directly.
    The quartet's own quantities are separate named fields
    (:attr:`double_effect`, :attr:`interaction`).
    """

    left: object
    right: object
    sites: tuple
    left_value: object
    right_value: object
    assay: str
    target: str
    same_assay: bool
    coupled: bool
    kind: str = CONTRAST_SINGLE
    background: object = None
    double: object = None
    margin: object = None
    interaction: object = None
    double_effect: object = None
    effect_left: object = None
    effect_right: object = None
    substitution: tuple = ()
    scale: str = ""

    def alternatives(self):
        """Every construct this contrast presents, in a fixed order.

        ``(background?, left, right, double?)``. The order is what
        ``contrast_payload``'s row spans index into, so it is fixed here and not
        recomputed by any consumer.
        """
        rows = [self.left, self.right]
        if self.background is not None:
            rows.insert(0, self.background)
        if self.double is not None:
            rows.append(self.double)
        return tuple(rows)

    def pair_rows(self):
        """``(left offset, right offset)`` inside :meth:`alternatives`."""
        offset = 1 if self.background is not None else 0
        return offset, offset + 1

    def document(self):
        return {"kind": self.kind, "sites": [[c, int(i)] for c, i in self.sites],
                "assay": self.assay, "target": self.target,
                "same_assay": bool(self.same_assay), "coupled": bool(self.coupled),
                "left_value": None if self.left_value is None else float(self.left_value),
                "right_value": None if self.right_value is None else float(self.right_value),
                "margin": None if self.margin is None else float(self.margin),
                "margin_definition": "left_value - right_value, for every kind",
                "interaction": None if self.interaction is None else float(self.interaction),
                "double_effect": (None if self.double_effect is None
                                  else float(self.double_effect)),
                "substitution": list(self.substitution),
                "alternatives": len(self.alternatives()), "scale": self.scale}


def _mutation_set(construct, parent_heavy, parent_light):
    """``((chain, index, residue), ...)`` of a construct against its background."""
    if len(construct.heavy) != len(parent_heavy) or len(construct.light) != len(parent_light):
        return None
    out = [(HEAVY_CHAIN, i, a) for i, (a, b) in enumerate(zip(construct.heavy, parent_heavy))
           if a != b]
    out += [(LIGHT_CHAIN, j, a) for j, (a, b) in enumerate(zip(construct.light, parent_light))
            if a != b]
    return tuple(out)


def _packed_group(rows):
    """``(codes, boundary)``: a ``(N, L)`` uint8 array of heavy+light for one group."""
    heavy = len(rows[0].construct.heavy)
    joined = "".join(row.construct.heavy + row.construct.light for row in rows)
    width = heavy + len(rows[0].construct.light)
    return (np.frombuffer(joined.encode("ascii"), dtype=np.uint8).reshape(len(rows), width),
            heavy)


def _site_of(column, boundary):
    return (HEAVY_CHAIN, int(column)) if int(column) < int(boundary) else \
        (LIGHT_CHAIN, int(column) - int(boundary))


class _Reservoir:
    """Uniform reservoir sample of a stream whose length is not known in advance.

    The point is memory, not elegance. CR9114's 50,540 eligible genotypes over a
    16-site binary space have on the order of 400,000 single-substitution
    neighbour pairs; materialising them all and then sampling would allocate the
    whole table to keep a few hundred rows of it. The reservoir keeps ``capacity``
    entries and counts the rest exactly, so both the cap and the available total
    are real numbers.
    """

    def __init__(self, capacity, *, rng):
        self.capacity = max(0, int(capacity))
        self.rng = rng
        self.kept = []
        self.seen = 0

    def offer(self, item):
        self.seen += 1
        if len(self.kept) < self.capacity:
            self.kept.append(item)
            return True
        position = int(self.rng.integers(0, self.seen))
        if position < self.capacity:
            self.kept[position] = item
            return True
        return False

    @property
    def dropped(self):
        return max(0, self.seen - len(self.kept))


def build_local_contrasts(summaries, *, max_sites=2, max_per_background=64, seed=20260927,
                          require_same_assay=True, max_per_context=32,
                          max_quartets_per_background=8, parents=None,
                          training_only=True):
    """Enumerate single-substitution neighbours across **every** measured background.

    Two earlier implementations were wrong in opposite directions and the shape of
    this one is the correction of both.

    The first compared every measured row with every other row inside an assay. At
    the Li release's 104,972 variants that is 5.5 billion comparisons for one
    group, and it is not merely slow: in a single-parent mutational scan almost
    every pair of single mutants *is* at distance two, so the complete enumeration
    is quadratic in its **output** and no implementation of it finishes.

    The second anchored everything on one modal wild type and discarded every
    background more than two substitutions from it. That is cheap and it deletes
    the measurement the campaign exists to learn: the effect of a substitution
    *in a different background* is precisely the contextual signal, and a
    single-parent view of it is the position-only rule the controls are supposed
    to beat. It also skipped whole families outright when no exact wild type was
    measured.

    What this one does instead: index the group by masked context. For every
    varying column ``j``, bucket the constructs by their residues at all the
    **other** varying columns; each bucket is one masked background and every pair
    inside it differs at ``j`` alone. That is the exact single-substitution
    neighbour graph, it costs one bucketing pass per varying column, and it finds
    neighbours on every background rather than on one.

    Each neighbour pair is classified by what the substitution does relative to
    the group's reference residue at that coordinate:

    * ``single_vs_background`` -- one endpoint carries the reference residue, so
      the pair is a measured single-site effect **in that background**. ``left``
      is the variant and ``right`` is the background, so ``margin`` is the effect.
    * ``same_position_alternatives`` -- neither endpoint carries the reference
      residue: two candidate substitutions competing at one coordinate. ``left``
      is the one whose residue sorts first, so the orientation is canonical.

    ``coupled_quartet`` is then a four-cycle in that graph anchored at a
    background that carries the reference residue at both coordinates, which is
    the only configuration an interaction is computable from. Quartets are
    enumerated per background with a declared cap, because the parent of a full
    scan has thousands of neighbours and ``C(n, 2)`` of them is the quadratic
    table this function refuses to build.

    Every cap is a reservoir sample with the dropped count reported, so a capped
    group is visible rather than silently truncated.
    """
    rng = np.random.default_rng(int(seed))
    parents = dict(parents or {})
    by_group = {}
    skipped = {"no_value": 0, "not_exact_value": 0, "unalignable_length": 0,
               "no_varying_position": 0, "not_training_partition": 0, "no_edit_map": 0}
    not_exact_by_status = {}
    for row in summaries:
        if training_only and not row.is_training:
            skipped["not_training_partition"] += 1
            continue
        if row.exact_value is None:
            # A margin is a difference of two measurements, so it needs an EXACT value
            # at each end. A purely censored or quarantined row has none (``no_value``).
            # A row with some exact and some censored replicates has a qualified
            # mean of its uncensored replicates -- not a number the assay produced --
            # and differencing two of those would hand the decoder objective a margin
            # nobody measured (``not_exact_value``). Both are counted, not absent.
            if row.value is None:
                skipped["no_value"] += 1
            else:
                skipped["not_exact_value"] += 1
            status = row.exact_status
            not_exact_by_status[status] = not_exact_by_status.get(status, 0) + 1
            continue
        key = (row.study, row.assay, row.target,
               len(row.construct.heavy), len(row.construct.light))
        by_group.setdefault(key, []).append(row)
    contrasts = []
    pairs_available = quartets_available = 0
    pairs_dropped = quartets_dropped = 0
    contexts = 0
    adjacency_budget = 64 * max(1, int(max_per_background))
    for key, group in sorted(by_group.items()):
        _study, assay, target, _heavy_length, _light_length = key
        if len(group) < 2:
            continue
        codes, boundary = _packed_group(group)
        varying = np.flatnonzero((codes != codes[0]).any(axis=0))
        if varying.size == 0:
            skipped["no_varying_position"] += len(group)
            continue
        block = codes[:, varying]
        reference = _reference_residues(group, block, varying, width=codes.shape[1],
                                        parents=parents)
        pair_reservoir = _Reservoir(max_per_background, rng=rng)
        # The adjacency list is only consulted to find quartets, and it is bounded:
        # a 16-site binary landscape of 50,000 genotypes has on the order of 400,000
        # distance-one edges, and holding all of them to find a few hundred four-cycles
        # is the allocation the reservoir exists to avoid one level up.
        edge_reservoir = _Reservoir(adjacency_budget, rng=rng)
        for column in range(varying.size):
            others = np.delete(block, column, axis=1)
            # Bucketed by the residues at every OTHER varying column: one bucket is
            # one masked background. Keyed on the row's bytes rather than through
            # ``np.unique(axis=0)``, whose inverse-array shape changed between numpy
            # 1.x and 2.x -- a silent reshape here would mis-bucket every row.
            buckets = {}
            for row_index in range(others.shape[0]):
                buckets.setdefault(others[row_index].tobytes(), []).append(row_index)
            site = _site_of(int(varying[column]), boundary)
            for key in sorted(buckets):
                bucket = np.asarray(buckets[key], dtype=np.int64)
                if bucket.size < 2:
                    continue
                contexts += 1
                # Deterministic within a bucket: sort by the residue at this
                # column, then by sequence identity. A bucket is one masked
                # background, so it holds at most one member per residue value
                # unless two sources published the same sequence, and the tie-break
                # keeps the enumeration independent of file order either way.
                members = sorted((int(row) for row in bucket),
                                 key=lambda row: (int(block[row, column]),
                                                  group[row].construct.identity()))
                context_pairs = []
                for left_position in range(len(members)):
                    for right_position in range(left_position + 1, len(members)):
                        left_row, right_row = members[left_position], members[right_position]
                        if int(block[left_row, column]) == int(block[right_row, column]):
                            continue
                        context_pairs.append((left_row, right_row))
                pairs_available += len(context_pairs)
                if len(context_pairs) > int(max_per_context):
                    selected = rng.choice(len(context_pairs), int(max_per_context), replace=False)
                    pairs_dropped += len(context_pairs) - len(selected)
                    context_pairs = [context_pairs[int(i)] for i in selected]
                for left_row, right_row in context_pairs:
                        pair_reservoir.offer((left_row, right_row, site,
                                             int(varying[column])))
                        if int(max_sites) >= 2:
                            edge_reservoir.offer((left_row, right_row, int(varying[column])))
        pairs_dropped += pair_reservoir.dropped
        for left_row, right_row, site, column in pair_reservoir.kept:
            contrasts.append(_pair_contrast(
                group[left_row], group[right_row], site=site,
                reference=reference.get(column), assay=assay, target=target,
                same_assay=bool(require_same_assay)))
        if int(max_sites) >= 2:
            adjacency = {}
            for left_row, right_row, column in edge_reservoir.kept:
                adjacency.setdefault(left_row, []).append((column, right_row))
                adjacency.setdefault(right_row, []).append((column, left_row))
            quartet_reservoir = _Reservoir(max_quartets_per_background, rng=rng)
            sequence_rows = {codes[position].tobytes(): position
                             for position in range(len(group))}
            for anchor, neighbours in sorted(adjacency.items()):
                by_column = {}
                for column, other in neighbours:
                    if reference.get(column) is None:
                        continue
                    if int(codes[anchor, column]) != int(reference[column]):
                        continue          # the anchor is not the background here
                    if int(codes[other, column]) == int(reference[column]):
                        continue          # the neighbour is not an edit
                    by_column.setdefault(column, []).append(other)
                columns = sorted(by_column)
                rng.shuffle(columns)
                for neighbours_at_site in by_column.values():
                    rng.shuffle(neighbours_at_site)
                # The probe budget is what keeps a dense background from becoming
                # the quadratic table this function exists to avoid: the parent of a
                # saturated scan has hundreds of neighbours and C(n, 2) pairs of
                # them. Probes are counted and the budget is reported.
                probes, budget = 0, min(512, 64 * max(1, int(max_quartets_per_background)))
                for first in range(len(columns)):
                    if probes >= budget:
                        break
                    for second in range(first + 1, len(columns)):
                        if probes >= budget:
                            break
                        for left_row in by_column[columns[first]]:
                            if probes >= budget:
                                break
                            for right_row in by_column[columns[second]]:
                                if probes >= budget:
                                    break
                                probes += 1
                                candidate = codes[anchor].copy()
                                candidate[columns[first]] = codes[left_row, columns[first]]
                                candidate[columns[second]] = codes[right_row, columns[second]]
                                double_row = sequence_rows.get(candidate.tobytes())
                                if double_row is None:
                                    continue
                                quartet_reservoir.offer(
                                    (anchor, left_row, right_row, double_row,
                                     columns[first], columns[second]))
            quartets_available += quartet_reservoir.seen
            quartets_dropped += quartet_reservoir.dropped
            for anchor, left_row, right_row, double_row, first, second in quartet_reservoir.kept:
                contrasts.append(_quartet_contrast(
                    background=group[anchor], left=group[left_row], right=group[right_row],
                    double=group[double_row],
                    sites=(_site_of(int(first), boundary), _site_of(int(second), boundary)),
                    assay=assay, target=target, same_assay=bool(require_same_assay)))
    by_kind = {kind: sum(1 for c in contrasts if c.kind == kind) for kind in CONTRAST_KINDS}
    available = pairs_available + quartets_available
    return {"contrasts": tuple(contrasts), "skipped": dict(skipped),
            "not_exact_by_status": dict(sorted(not_exact_by_status.items())),
            "groups": len(by_group), "by_kind": by_kind,
            "masked_contexts": contexts,
            "available_before_cap": available,
            "dropped_by_cap": pairs_dropped + quartets_dropped,
            "pairs_available": pairs_available, "pairs_dropped": pairs_dropped,
            "quartets_available": quartets_available, "quartets_dropped": quartets_dropped,
            "max_per_background": int(max_per_background),
            "max_per_context": int(max_per_context),
            "max_quartets_per_background": int(max_quartets_per_background),
            "adjacency_budget": int(adjacency_budget),
            "seed": int(seed),
            "training_only": bool(training_only),
            "rule": ("masked-context neighbour indexing inside one (study, assay, target, chain "
                     "lengths) group: for every varying column, constructs are bucketed by their "
                     "residues at the other varying columns, and every pair in a bucket differs "
                     "at that column alone. That is the exact single-substitution neighbour graph "
                     "over EVERY measured background, not a scan around one modal wild type, and "
                     "no all-pairs scan exists -- in a single-parent scan almost every pair of "
                     "singles is already at distance two, so the complete pair table is quadratic "
                     "in its output. Quartets are four-cycles of that graph anchored at a "
                     "reference background, capped per background with a reservoir sample.")}


def _reference_residues(group, block, varying, *, width, parents):
    """``{original column: reference byte}`` -- the residue an edit is measured against.

    Taken from the declared parent sequence when one is supplied for the group's
    parent name **and its length matches the group's**, and from the per-column
    mode otherwise. The mode is a *labelling* device only: it decides whether a
    neighbour pair is called a single-site effect or a pair of competing
    alternatives, and it never gates enumeration. The earlier implementation used
    the mode to *require* a measured wild type and dropped whole families that did
    not have one.
    """
    parent = parents.get(group[0].parent)
    if parent is not None:
        joined = f"{parent[0]}{parent[1]}"
        if len(joined) == int(width):
            row = np.frombuffer(joined.encode("ascii"), dtype=np.uint8)
            return {int(column): int(row[int(column)]) for column in varying}
    out = {}
    for position, column in enumerate(varying):
        counts = np.bincount(block[:, position], minlength=256)
        out[int(column)] = int(counts.argmax())
    return out


def _pair_contrast(left, right, *, site, reference, assay, target, same_assay):
    """One single-substitution neighbour pair, oriented canonically."""
    left_residue = left.construct.residue_at(site)
    right_residue = right.construct.residue_at(site)
    reference_residue = None if reference is None else chr(int(reference))
    if reference_residue is not None and right_residue == reference_residue:
        variant, background = left, right
    elif reference_residue is not None and left_residue == reference_residue:
        variant, background = right, left
    else:
        # Two competing substitutions: the canonical orientation is by residue, so
        # the same unordered pair always yields the same signed margin.
        ordered = sorted(((left_residue, left), (right_residue, right)), key=lambda e: e[0])
        variant, background = ordered[0][1], ordered[1][1]
        return LocalContrast(
            left=variant, right=background, sites=(site,), left_value=variant.exact_value,
            right_value=background.exact_value, assay=assay, target=target,
            same_assay=same_assay, coupled=False, kind=CONTRAST_ALTERNATIVES,
            margin=variant.exact_value - background.exact_value,
            substitution=(site[0], int(site[1]), background.construct.residue_at(site),
                          variant.construct.residue_at(site)),
            scale=_scale_of(variant))
    return LocalContrast(
        left=variant, right=background, sites=(site,), left_value=variant.exact_value,
        right_value=background.exact_value, assay=assay, target=target,
        same_assay=same_assay, coupled=False, kind=CONTRAST_SINGLE,
        margin=variant.exact_value - background.exact_value,
        substitution=(site[0], int(site[1]), background.construct.residue_at(site),
                      variant.construct.residue_at(site)),
        scale=_scale_of(variant))


def _quartet_contrast(*, background, left, right, double, sites, assay, target, same_assay):
    # Every member passed the exact-value gate in build_local_contrasts, so these are
    # means of exact replicates and never the uncensored-only mean of a mixed row.
    effect_left = left.exact_value - background.exact_value
    effect_right = right.exact_value - background.exact_value
    effect_double = double.exact_value - background.exact_value
    return LocalContrast(
        left=left, right=right, sites=tuple(sites), left_value=left.exact_value,
        right_value=right.exact_value, assay=assay, target=target, same_assay=same_assay,
        coupled=True, kind=CONTRAST_COUPLED, background=background, double=double,
        margin=left.exact_value - right.exact_value,
        effect_left=effect_left, effect_right=effect_right, double_effect=effect_double,
        interaction=effect_double - (effect_left + effect_right),
        scale=_scale_of(double))


def _scale_of(summary):
    return getattr(summary, "scale", "") or ""


def _endpoint_names(contrast):
    """The role of each entry of :meth:`LocalContrast.alternatives`, in the same order."""
    names = ["left", "right"]
    if contrast.background is not None:
        names.insert(0, "background")
    if contrast.double is not None:
        names.append("double")
    return tuple(names)


def scalar_contrast_problems(contrast):
    """Why this contrast's endpoints are not an exact, compatible scalar target. ``[]`` if they are.

    ``build_local_contrasts`` filters every endpoint by :attr:`ConstructSummary.exact_value`; a
    contrast built by hand (a test, a permutation, an import) reaches the objectives without that
    filter, so the same rule is applied here from the endpoints themselves:

    * every endpoint must be an exact scalar -- ``exact_status == "exact"``: all replicates
      uncensored, finite and degenerate-interval points. A ``KIND_INTERVAL`` row whose replicates
      are degenerate points qualifies; an ordinal class, a censored or mixed-censored row, a
      non-finite value and a row with room around its value do not;
    * every endpoint must name the contrast's own assay and target, one family and one scale,
      the contrast's. A difference across assays or scales is a number on no scale.

    The contrast's ``margin`` is deliberately NOT required to equal the difference of its
    endpoints' values: ``background_permuted`` moves effect bundles between backgrounds on
    purpose, and its contrasts must stay valid.
    """
    problems = []
    endpoints = tuple(contrast.alternatives())
    names = _endpoint_names(contrast)
    for name, endpoint in zip(names, endpoints):
        status = endpoint.exact_status
        if status != "exact":
            problems.append(f"{name} endpoint is {status}, not an exact scalar")
        if endpoint.assay != contrast.assay:
            problems.append(f"{name} endpoint is assay {endpoint.assay!r}, the contrast "
                            f"{contrast.assay!r}")
        if endpoint.target != contrast.target:
            problems.append(f"{name} endpoint is target {endpoint.target!r}, the contrast "
                            f"{contrast.target!r}")
    families = {endpoint.family for endpoint in endpoints}
    if len(families) > 1:
        problems.append(f"endpoints span families {sorted(str(item) for item in families)}")
    scales = {_scale_of(endpoint) for endpoint in endpoints}
    if len(scales) > 1:
        problems.append(f"endpoints span scales {sorted(scales)}")
    elif scales and next(iter(scales)) != (contrast.scale or ""):
        problems.append(f"endpoints are on scale {next(iter(scales))!r}, the contrast declares "
                        f"{contrast.scale!r}")
    return problems


def require_exact_scalar_contrast(contrast, *, label="local contrast"):
    """Raise before any forward when the contrast is not an exact, compatible scalar target.

    See :func:`scalar_contrast_problems`. Called by ``local_outcome`` for every contrast of a
    microbatch and by ``quartet_joint_outcome`` for its own, so a mixed-censored double endpoint
    cannot reach a quartet through a hand-built contrast.
    """
    problems = scalar_contrast_problems(contrast)
    require(not problems,
            f"{label} ({contrast.kind}) is not an exact scalar target: {'; '.join(problems)}. "
            "A margin, effect or interaction differenced from these endpoints is a number "
            "nobody measured.")
    return True


def _modal_sequence(sequences, *, sample=4096):
    """The per-position modal residue of a set of equal-length sequences.

    Retained because the source loaders use it to derive a group's consensus
    background for their own edit maps. It is no longer what
    :func:`build_local_contrasts` anchors on.
    """
    rows = [text for text in sequences if text]
    if not rows:
        return ""
    length = len(rows[0])
    if any(len(text) != length for text in rows):
        return rows[0]
    if len(rows) > int(sample):
        step = len(rows) / float(sample)
        rows = [rows[int(index * step)] for index in range(int(sample))]
    block = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8).reshape(
        len(rows), length)
    out = bytearray()
    for position in range(length):
        counts = np.bincount(block[:, position], minlength=256)
        out.append(int(counts.argmax()))
    return out.decode("ascii")


# ---------------------------------------------------------------------------
# typed measurement payloads
# ---------------------------------------------------------------------------

@dataclass
class MeasurementPayload:
    """The measured outcome of every row of a batch, kept typed.

    The four kinds have four likelihoods and this is what carries the difference
    into the objective: a censored bound is not a point, an ordinal class is not
    a real number, and an enrichment ratio has no scale to be censored on. Every
    array in the first block is row-aligned with the batch, and ``kind_mask``
    selects the rows each head is entitled to.

    The second block is **per replicate**, not per row. ``replicate_row`` points
    back at the batch row each measurement belongs to and ``replicate_weight``
    sums to one within a construct, so the interval likelihood is evaluated on
    every replicate with its own censoring and each construct still contributes
    once. Evaluating the row aggregate instead is how a construct with one floor
    replicate and two point replicates gets trained as an exact point at the mean
    of the two.
    """

    kind: object
    value: object
    lower: object
    upper: object
    finite_lower: object
    finite_upper: object
    ordinal_class: object
    assay_index: object
    uncertainty: object
    support: object
    observation_scale: object = None
    replicate_row: object = None
    replicate_lower: object = None
    replicate_upper: object = None
    replicate_finite_lower: object = None
    replicate_finite_upper: object = None
    replicate_weight: object = None
    replicate_censored: object = None
    replicate_uncertainty: object = None

    def kind_mask(self, kind):
        return (np.asarray(self.kind) == str(kind)).astype(np.int64)

    def replicate_kind_mask(self, kind):
        """``(R,)`` mask over the per-replicate rows whose construct is ``kind``."""
        rows = self.kind_mask(kind)
        return rows[np.asarray(self.replicate_row, dtype=np.int64)]

    def rows(self):
        return int(np.asarray(self.kind).shape[0])

    def replicates(self):
        return 0 if self.replicate_row is None else int(
            np.asarray(self.replicate_row).shape[0])

    def document(self):
        kinds = np.asarray(self.kind)
        return {"rows": self.rows(), "replicate_rows": self.replicates(),
                "by_kind": {str(name): int((kinds == name).sum())
                            for name in sorted(set(kinds.tolist()))},
                "with_point_value": int(np.isfinite(np.asarray(self.value, dtype=float)).sum()),
                "left_censored": int((~np.asarray(self.finite_lower, dtype=bool)).sum()),
                "right_censored": int((~np.asarray(self.finite_upper, dtype=bool)).sum()),
                "censored_replicates": (0 if self.replicate_censored is None else
                                        int(np.asarray(self.replicate_censored).sum())),
                "rule": ("continuous, censored, ordinal and enrichment measurements stay "
                         "distinguished in the payload, so the objective cannot give one kind "
                         "another kind's likelihood. The per-replicate block keeps each "
                         "replicate's OWN censoring, weighted so a construct contributes once.")}


def measurement_payload(summaries, *, assay_index, ordinal_of=None):
    """Build a :class:`MeasurementPayload` from construct summaries.

    ``assay_index`` maps an assay identity to the row of the per-assay observation
    scale. An identity the map does not know is refused rather than defaulted to
    zero: silently sharing assay 0's learned scale is how two sources acquire one
    calibration nobody chose.
    """
    rows = list(summaries)
    require(rows, "A measurement payload needs at least one summary")
    kinds, values, lowers, uppers = [], [], [], []
    finite_low, finite_high, ordinals, assays, spread, support = [], [], [], [], [], []
    scales = []
    replicate_row, replicate_low, replicate_high = [], [], []
    replicate_finite_low, replicate_finite_high, replicate_weight = [], [], []
    replicate_censored, replicate_uncertainty = [], []
    for position, row in enumerate(rows):
        require(row.assay in assay_index,
                f"Assay {row.assay!r} is not in the frozen assay index {sorted(assay_index)[:6]}"
                "... An unknown assay would silently borrow another source's observation scale.")
        kinds.append(str(row.kind))
        # The scalar column is the EXACT value -- the mean of replicates that are all exact
        # finite points -- or NaN. ``row.value`` is the mean of the uncensored replicates, which
        # for one floor replicate and two points is a number the assay never produced, and the
        # enrichment ranking orders rows by this column. The likelihoods do not read it: they read
        # the per-replicate block below, which keeps every replicate's own interval.
        point = row.exact_value
        low, high = row.interval if row.interval else (None, None)
        if low is None and high is None and point is not None:
            low = high = point
        values.append(float("nan") if point is None else float(point))
        finite_low.append(low is not None)
        finite_high.append(high is not None)
        lowers.append(float(low) if low is not None else float("-inf"))
        uppers.append(float(high) if high is not None else float("inf"))
        ordinal = row.ordinal_class
        if ordinal_of is not None:
            ordinal = ordinal_of.get(row.construct.identity(), ordinal)
        ordinals.append(-1 if ordinal is None else int(ordinal))
        assays.append(int(assay_index[row.assay]))
        spread.append(float("nan") if row.uncertainty is None else float(row.uncertainty))
        support.append(int(row.support))
        observed = row.observation_scale
        scales.append(float("nan") if observed is None else float(observed))
        usable = [entry for entry in row.measurements
                  if entry.value is not None or entry.censoring != CENSOR_NONE]
        weight = 0.0 if not usable else 1.0 / len(usable)
        for entry in usable:
            bound_low, bound_high = entry.bounds()
            if bound_low is None and bound_high is None:
                continue
            replicate_row.append(int(position))
            replicate_finite_low.append(bound_low is not None)
            replicate_finite_high.append(bound_high is not None)
            replicate_low.append(float(bound_low) if bound_low is not None else float("-inf"))
            replicate_high.append(float(bound_high) if bound_high is not None else float("inf"))
            replicate_weight.append(float(weight))
            replicate_censored.append(entry.censoring != CENSOR_NONE)
            replicate_uncertainty.append(float("nan") if entry.uncertainty is None
                                         else float(entry.uncertainty))
    return MeasurementPayload(
        kind=np.asarray(kinds, dtype=object), value=np.asarray(values, dtype=np.float64),
        lower=np.asarray(lowers, dtype=np.float64), upper=np.asarray(uppers, dtype=np.float64),
        finite_lower=np.asarray(finite_low, dtype=bool),
        finite_upper=np.asarray(finite_high, dtype=bool),
        ordinal_class=np.asarray(ordinals, dtype=np.int64),
        assay_index=np.asarray(assays, dtype=np.int64),
        uncertainty=np.asarray(spread, dtype=np.float64),
        support=np.asarray(support, dtype=np.int64),
        observation_scale=np.asarray(scales, dtype=np.float64),
        replicate_row=np.asarray(replicate_row, dtype=np.int64),
        replicate_lower=np.asarray(replicate_low, dtype=np.float64),
        replicate_upper=np.asarray(replicate_high, dtype=np.float64),
        replicate_finite_lower=np.asarray(replicate_finite_low, dtype=bool),
        replicate_finite_upper=np.asarray(replicate_finite_high, dtype=bool),
        replicate_weight=np.asarray(replicate_weight, dtype=np.float64),
        replicate_censored=np.asarray(replicate_censored, dtype=bool),
        replicate_uncertainty=np.asarray(replicate_uncertainty, dtype=np.float64))


@dataclass
class HeadBatch:
    """Full, unmasked variant sequences plus their typed measurements.

    The observation heads read the **whole construct**, not a masked window: they
    predict the measured outcome of the molecule that was measured. The condition
    carries ``bin=null`` on every row, so the head cannot read the answer it is
    being asked for off its own conditioning input.
    """

    tokens: object
    attention_mask: object
    condition_indices: object
    measurement: object
    identities: tuple = ()
    sources: tuple = ()
    assays: tuple = ()

    def document(self):
        return {"rows": int(np.asarray(self.tokens).shape[0]),
                "length": int(np.asarray(self.tokens).shape[1]),
                "sources": sorted(set(self.sources)),
                "measurement": self.measurement.document(),
                "input": ("the full variant construct, unmasked, at bin=null. A head that saw "
                          "the quality bin it predicts would report the calibration of the "
                          "condition embedding rather than of the representation.")}


def reconstruction_views(constructs, *, rng, mask_fraction=0.15, condition_spec, family,
                         route="causal", max_positions=400, minimum_masked=1):
    """Prior-retention views: random positions masked, null quality requested.

    The condition is always ``bin=null`` here. Reconstruction is not a quality
    request, and giving it one would teach the model that ``high`` means "this is a
    real antibody".
    """
    views = []
    for construct in constructs:
        total = len(construct.heavy) + len(construct.light)
        count = max(int(minimum_masked), int(round(float(mask_fraction) * total)))
        positions = rng.choice(total, size=min(count, total), replace=False)
        sites = []
        for position in sorted(int(value) for value in positions):
            if position < len(construct.heavy):
                sites.append((HEAVY_CHAIN, position))
            else:
                sites.append((LIGHT_CHAIN, position - len(construct.heavy)))
        masked = construct.__class__(heavy=construct.heavy, light=construct.light,
                                    sites=tuple(sites), source=construct.source,
                                    record_id=construct.record_id)
        condition = Condition(family=family, bin=BIN_NULL, assay="null")
        order = emission_order(masked.sites, order="declared")
        serialization = (causal_serialization(masked, order=order,
                                             residues=masked.residues(order),
                                             max_positions=max_positions)
                         if route == "causal" else
                         ablang_serialization(masked, order=order))
        views.append((masked, serialization, condition))
    return tuple(views)


def whole_window_views(observations, *, condition_spec, bins, route="causal", rng=None,
                       order="declared", max_positions=400, answer_free=False,
                       tile=None, sample_tile=False, progressive_probability=0.0):
    """Whole-window views: the declared common region masked, quality requested.

    This is the task that constrains the FIRST decision when nothing is visible.
    A correct local editor can still miss a useful family fifty-fold -- the plan's
    AA/VV arithmetic -- which is why this is its own task and not a by-product.

    The masked region is :attr:`ConstructSummary.region`, the positions that vary
    across the construct's whole group, **not** the construct's own mutated
    positions. A mutation-dependent region gives every row a different support, so
    its window log probability is not comparable with any other row's and the
    high-versus-null preference the objective trains on would be a comparison of
    two different quantities.

    ``answer_free=True`` emits the same views at ``bin=null``. The pair is what
    makes the preference term a measurement rather than an assertion: the same
    background, the same targets, one quality request and one absence of it.

    ``tile`` bounds the number of queried coordinates per view. A region wider than
    the route's query budget is trained as declared tiles with the full construct
    visible in every one of them, which is what keeps a 60-position AlphaSeq region
    trainable on a 400-position autoregressive budget without truncating a chain.
    """
    views = []
    for record in observations:
        region = tuple(getattr(record, "region", None) or record.construct.sites)
        if not region:
            continue
        bin_name = BIN_NULL if answer_free else bins.get(record.population_key, BIN_NULL)
        condition = Condition(family=record.family, bin=bin_name, assay=record.assay)
        if rng is not None:
            condition = condition_spec.apply_dropout(condition, rng)
        ordered = emission_order(region, order=order, rng=rng)
        chunks = _tiles(ordered, tile)
        if sample_tile and rng is not None:
            chunks = (chunks[int(rng.integers(len(chunks)))],)
        for chunk in chunks:
            construct = with_sites(record.construct, chunk)
            query = chunk
            if route != "causal" and rng is not None and len(chunk) > 1 and \
                    float(rng.random()) < float(progressive_probability):
                # Reveal a prefix from this observed sequence; all future sites
                # remain masked. Retain full-mask examples as a separate mixture.
                query = chunk[int(rng.integers(1, len(chunk))):]
            serialization = (causal_serialization(construct, order=chunk,
                                                  residues=construct.residues(chunk),
                                                  max_positions=max_positions)
                             if route == "causal" else
                             ablang_serialization(construct, masked_sites=query, order=query))
            views.append((construct, serialization, condition))
    return tuple(views)


def with_sites(construct, sites):
    """The same molecule with a different declared edit map. The scaffold is untouched."""
    sites = tuple(sites)
    if tuple(construct.sites) == sites:
        return construct
    return construct.__class__(heavy=construct.heavy, light=construct.light, sites=sites,
                               source=construct.source, record_id=construct.record_id)


def _tiles(sites, tile):
    """Split a region into declared tiles, or return it whole."""
    sites = tuple(sites)
    if tile is None or int(tile) <= 0 or len(sites) <= int(tile):
        return (sites,)
    width = int(tile)
    return tuple(sites[start:start + width] for start in range(0, len(sites), width))


def head_views(summaries, *, condition_spec, route="causal", max_positions=400, rng=None):
    """Full-sequence, answer-free views for the observation heads.

    Two properties, both asserted rather than conventional:

    * **the sequence is the whole measured variant, unmasked.** The head predicts
      the outcome of the molecule that was measured; masking the edit window
      would ask it to predict an outcome of a molecule that does not exist, and
      the earlier implementation did exactly that by reusing the whole-window
      serialisation.
    * **the quality bin is null.** A head that saw the bin it is predicting
      reports the calibration of the condition embedding, not of the
      representation. :func:`require_answer_free` is called by the objective too.
    """
    views = []
    for row in summaries:
        condition = condition_spec.answer_free(
            Condition(family=row.family, bin=BIN_NULL, assay=row.assay))
        if rng is not None:
            condition = condition_spec.apply_dropout(condition, rng)
        construct = row.construct
        if route == "causal":
            require_bound_vocabulary()
            tokens = [PIGGEN_START_ID]
            tokens += [PIGGEN_ID_OF_RESIDUE[residue]
                       for residue in construct.heavy + construct.light]
            tokens.append(PIGGEN_END_ID)
        else:
            tokens = list(ablang_token_ids(construct, masked_sites=()))
        require(len(tokens) <= int(max_positions),
                f"A head view needs {len(tokens)} positions against a budget of {max_positions}; "
                "nothing is truncated, because a dropped light chain would leave the head "
                "predicting an affinity from half a molecule")
        views.append((construct, tuple(tokens), condition, row))
    return tuple(views)


def collate_heads(views, *, pad_id, condition_spec, assay_index, ordinal_of=None):
    """Build a :class:`HeadBatch`. Every condition is checked answer-free."""
    require(views, "A head batch with no rows has nothing to supervise")
    tokens = [list(view[1]) for view in views]
    conditions = [view[2] for view in views]
    require_answer_free(conditions)
    summaries = [view[3] for view in views]
    token_array, attention = pad_rows(tokens, value=int(pad_id))
    encoded = np.asarray([list(condition_spec.encode(condition)) for condition in conditions],
                         dtype=np.int64)
    return HeadBatch(tokens=token_array, attention_mask=attention, condition_indices=encoded,
                     measurement=measurement_payload(summaries, assay_index=assay_index,
                                                     ordinal_of=ordinal_of),
                     identities=tuple(view[0].identity() for view in views),
                     sources=tuple(view[0].source for view in views),
                     assays=tuple(row.assay for row in summaries))


def require_answer_free(conditions, *, label="observation heads"):
    """Every condition in a head batch must carry ``bin=null``."""
    offending = [condition.document() for condition in conditions
                 if condition.bin != BIN_NULL]
    require(not offending,
            f"{label}: {len(offending)} condition(s) carry a quality bin, e.g. {offending[:3]}. "
            "The outcome heads train on an answer-free stream; a head that reads the bin it is "
            "predicting measures the condition embedding, not the representation.")
    return True


# ---------------------------------------------------------------------------
# microbatch shapes
# ---------------------------------------------------------------------------

def positive_integer(value, name):
    """``value`` as an ``int``, or raise. A bool, a float and a string are not sizes."""
    require(isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            f"{name} must be a positive integer, got {value!r}")
    require(int(value) >= 1, f"{name} must be a positive integer, got {int(value)}")
    return int(value)


def local_shape_recipe(curriculum):
    """What the DECLARED curriculum asks of a local microbatch. Derived from the recipe alone.

    A local microbatch is built from whole contrasts: a pair is two endpoint rows and a quartet
    four. So a microbatch size that cannot be tiled by them silently changes the recipe -- a
    requested 1 emits 2 rows, and a requested 3 emits 2 and never forms the quartet the
    recipe asks for. The requirement therefore comes from what the recipe declares, never from
    whether a random draw happened to find a quartet:

    * ``interactions`` -- ``max_contrast_sites >= 2`` and the recipe draws quartets
      (``local_quartet_probability > 0``) or fits their joint (``interaction_weight > 0``): the
      microbatch must be EVEN and at least 4;
    * ``pairs`` -- an explicitly pair-only local recipe: EVEN and at least 2;
    * ``undeclared`` / ``not_scheduled`` -- no local recipe is declared, or the local task has no
      share: nothing beyond a positive size is required here (the stream refuses an
      undeclared recipe itself).
    """
    block = dict(curriculum or {})
    shares = dict(block.get("shares") or {})
    scheduled = float(shares.get(TASK_LOCAL, DEFAULT_SHARES[TASK_LOCAL])) > 0
    sites = block.get("max_contrast_sites")
    if not scheduled:
        return {"scheduled": False, "mode": "not_scheduled", "even": False, "minimum": 1,
                "reason": "the local task has no share, so no local microbatch is drawn"}
    if sites is None:
        return {"scheduled": True, "mode": "undeclared", "even": False, "minimum": 1,
                "reason": "the curriculum declares no max_contrast_sites"}
    probability = float(block.get("local_quartet_probability", 0.25))
    weight = float(block.get("interaction_weight", 0.5))
    if int(sites) >= 2 and (probability > 0 or weight > 0):
        return {"scheduled": True, "mode": "interactions", "even": True, "minimum": 4,
                "reason": (f"max_contrast_sites={int(sites)} with local_quartet_probability="
                           f"{probability:g} and interaction_weight={weight:g} requests local "
                           "interactions, and a quartet is four endpoint rows")}
    return {"scheduled": True, "mode": "pairs", "even": True, "minimum": 2,
            "reason": (f"max_contrast_sites={int(sites)} with local_quartet_probability="
                       f"{probability:g} and interaction_weight={weight:g} is a pair-only local "
                       "recipe, and a pair is two endpoint rows")}


def joint_executable(contrast):
    """Would ``tasks.local_outcome`` actually run this contrast's sequential joint?

    A quartet's joint is the expensive local computation (``1 + residues`` scored rows), and it
    runs only for a coupled contrast with a numeric interaction on a numeric scale: an
    enrichment-scale quartet is ordered, never fitted, and a pair has no joint at all. This is the
    same guard ``local_outcome`` applies (``kind == coupled``, ``left.kind != enrichment``,
    ``interaction`` present, a finite margin) plus the early returns of ``quartet_joint_outcome``
    (a background, a double and two distinct sites). The profile picks its representative
    quartets with this predicate so it times the work production does; the evidence it records
    afterwards is the work that actually ran (``quartets_trained``), which is the arbiter.
    """
    if contrast.kind != CONTRAST_COUPLED or contrast.left.kind == KIND_ENRICHMENT:
        return False
    if contrast.background is None or contrast.double is None:
        return False
    if len(contrast.sites) != 2 or contrast.sites[0] == contrast.sites[1]:
        return False
    return all(value is not None and math.isfinite(float(value))
               for value in (contrast.interaction, contrast.margin))


def local_joint_requirement(curriculum):
    """What EXECUTED local work the declared recipe needs a profile to have measured.

    Derived from the curriculum alone, with the defaults ``FlightServices.task_settings`` applies
    (``interaction_weight`` 0.5, ``max_joint_per_update`` 1):

    * ``coupled_rows_required`` -- the recipe requests quartets (:func:`local_shape_recipe`
      mode ``interactions``), so the profiled local batch must hold coupled contrasts;
    * ``joint_required`` -- additionally the joint term runs (a positive ``interaction_weight``
      and at least one joint per update), so the profile must have trained at least one quartet's
      joint. A probability-only coupled recipe with weight zero declares the cheaper computation:
      its quartet rows are scored and no joint is, and the evidence says so;
    * a pair-only, undeclared or unscheduled local recipe requires neither.
    """
    recipe = local_shape_recipe(curriculum)
    block = dict(curriculum or {})
    weight = float(block.get("interaction_weight", 0.5))
    max_joint = int(block.get("max_joint_per_update", 1))
    coupled = recipe["mode"] == "interactions"
    return {"mode": recipe["mode"], "interaction_weight": weight, "max_joint_per_update": max_joint,
            "coupled_rows_required": coupled,
            "joint_required": bool(coupled and weight > 0 and max_joint >= 1)}


def validate_microbatch(microbatch, *, effective_batch=None, accumulation=None, curriculum=None,
                        label="microbatch"):
    """A microbatch the declared shape and recipe can actually run, or a ``ValueError``.

    Positive integer; when ``effective_batch`` is given it must be divisible by it EXACTLY (a
    floor division would silently train a smaller effective batch than declared); when
    ``accumulation`` is given too, ``microbatch * accumulation`` must equal it; and the local
    recipe's evenness / minimum (:func:`local_shape_recipe`) must hold. Returns the validated
    numbers and the recipe it was judged against.
    """
    size = positive_integer(microbatch, label)
    total = None if effective_batch is None else positive_integer(effective_batch,
                                                                   "effective_batch")
    steps = None
    if total is not None:
        require(total % size == 0,
                f"{label} {size} does not divide the effective batch {total} exactly "
                f"({total} = {total // size} x {size} + {total % size}). Rounding the "
                "accumulation would train an effective batch other than the declared one.")
        steps = total // size
        if accumulation is not None:
            require(positive_integer(accumulation, "accumulation") == steps,
                    f"{label} {size} x accumulation {accumulation} = {size * int(accumulation)}, "
                    f"not the declared effective batch {total}")
    recipe = local_shape_recipe(curriculum)
    if recipe["even"]:
        require(size % 2 == 0 and size >= recipe["minimum"],
                f"{label} {size} cannot carry the declared local recipe: {recipe['reason']}. "
                f"It needs an even microbatch of at least {recipe['minimum']}; a smaller or odd "
                "size is silently rounded to whole contrasts and the requested objective is "
                "not the one that trains.")
    return {"microbatch": size, "accumulation": steps, "effective_batch": total,
            "local_recipe": recipe}


# ---------------------------------------------------------------------------
# the mixture
# ---------------------------------------------------------------------------

#: The three source families the mixture allocates measured work across. They are
#: a property of the enrolled data, not of the HER2 target: ``her2`` is the target
#: population, ``other_functional`` is every other measured landscape, and
#: ``paired_replay`` is the unlabelled paired repertoire.
FAMILY_HER2 = "her2"
FAMILY_OTHER = "other_functional"
FAMILY_REPLAY = "paired_replay"
MIX_FAMILIES = (FAMILY_HER2, FAMILY_OTHER, FAMILY_REPLAY)


@dataclass
class TaskMixture:
    """Which task each update draws from, and which family each microbatch draws from.

    Both tables now reach the stream. ``family_mix`` was declared in the config,
    documented in the artifacts and consumed by nothing: every measured task drew
    uniformly from the pooled eligible summaries, so the effective mixture was
    whatever the row counts happened to be -- and the row counts are dominated by
    whichever source published the most variants. :meth:`family_schedule` makes the
    declared share the one that runs, and the exposure ledger's per-source
    breakdown is what checks it afterwards.
    """

    shares: dict = field(default_factory=lambda: dict(DEFAULT_SHARES))
    family_mix: dict = field(default_factory=lambda: {FAMILY_HER2: 0.50, FAMILY_OTHER: 0.25,
                                                      FAMILY_REPLAY: 0.25})

    def __post_init__(self):
        for name, table in (("task shares", self.shares), ("family mix", self.family_mix)):
            total = sum(float(value) for value in table.values())
            require(abs(total - 1.0) < 1e-9, f"The {name} sum to {total}, not 1")
            require(all(float(value) >= 0 for value in table.values()),
                    f"A negative {name} entry")
        require(set(self.shares) == set(TASKS),
                f"The task shares must name exactly {TASKS}, got {sorted(self.shares)}")
        require(set(self.family_mix) == set(MIX_FAMILIES),
                f"The family mix must name exactly {MIX_FAMILIES}, got {sorted(self.family_mix)}")

    def schedule(self, updates, *, seed):
        """A fixed, seeded task assignment for ``updates`` optimizer steps.

        Fixed in advance so an early stop still leaves every task represented in
        proportion, and so no scheduling decision can follow a result.
        """
        rng = np.random.default_rng(int(seed))
        names = list(TASKS)
        weights = np.asarray([float(self.shares[name]) for name in names])
        draws = rng.choice(len(names), size=int(updates), p=weights / weights.sum())
        return tuple(names[int(value)] for value in draws)

    def family_schedule(self, updates, *, seed, microbatches=1):
        """A fixed, seeded family assignment per **microbatch**, not per update.

        Per microbatch because gradient accumulation is where the mixture actually
        lands: with sixteen accumulation steps an update-level assignment would
        give one family the whole effective batch and the realised mixture would be
        the update-level one at sixteen times the variance.
        """
        rng = np.random.default_rng([int(seed), 7])
        names = list(MIX_FAMILIES)
        weights = np.asarray([float(self.family_mix[name]) for name in names])
        total = int(updates) * max(1, int(microbatches))
        draws = rng.choice(len(names), size=total, p=weights / weights.sum())
        return tuple(names[int(value)] for value in draws)

    def joint_schedule(self, updates, *, seed, microbatches=1):
        """One feasible task/family draw per microbatch, with both declared marginals.

        Unlabelled replay supports reconstruction only. Pairing independent draws
        with an OR fallback overweights replay and changes both marginals.
        Functional reconstruction is allowed only for an explicitly larger
        reconstruction share; a smaller share than replay is impossible.
        """
        replay = float(self.family_mix[FAMILY_REPLAY])
        reconstruction = float(self.shares[TASK_RECONSTRUCTION])
        require(reconstruction + 1e-12 >= replay,
                "Reconstruction share cannot be smaller than the unlabelled replay share")
        cells, probabilities = [], []
        if replay:
            cells.append((TASK_RECONSTRUCTION, FAMILY_REPLAY))
            probabilities.append(replay)
        functional = 1.0 - replay
        if functional > 0:
            for task in TASKS:
                share = float(self.shares[task]) - (replay if task == TASK_RECONSTRUCTION else 0)
                for family in (FAMILY_HER2, FAMILY_OTHER):
                    probability = max(0.0, share) * float(self.family_mix[family]) / functional
                    if probability:
                        cells.append((task, family))
                        probabilities.append(probability)
        require(abs(sum(probabilities) - 1.0) < 1e-9, "Joint curriculum is not normalized")
        rng = np.random.default_rng([int(seed), 19])
        indices = rng.choice(len(cells), size=int(updates) * int(microbatches), p=probabilities)
        return tuple(cells[int(i)] for i in indices)

    def document(self):
        return {"task_shares": dict(self.shares), "family_mix": dict(self.family_mix),
                "family_mix_is_consumed_by": ("family_schedule, one assignment per microbatch. "
                                              "The exposure ledger's per-source breakdown is the "
                                              "check that the realised mixture matches it."),
                "status": ("sweep centres, not literature-derived optima. They are amended from "
                           "measured development learning curves and from the recorded per-task "
                           "gradient fractions, never from a borrowed token percentage.")}


# ---------------------------------------------------------------------------
# the ordinal display pool
# ---------------------------------------------------------------------------

@dataclass
class OrdinalPool:
    """The whole sorted-bin display library, held as arrays and materialised lazily.

    367,042 training rows is the population the ordinal head exists for, and it is
    also 367,042 ``Construct`` objects if it is enrolled the way every other source
    is -- roughly 220 MB of Python objects for ten residues of information each.
    So the cores and their bin indices are held as arrays and
    :meth:`materialise` builds summaries for one drawn microbatch.

    This is the source the plan's positives-only comparison is *against*: a
    positives-only objective sees 120,504 rows and cannot learn the ordering
    between ``low`` and ``mid`` at all.
    """

    cores: tuple
    classes: object
    scaffold: object
    study: str
    assay: str
    target: str
    family: str
    scale: str
    ordinal_classes: int = 3
    source: str = "her2_display"
    split: str = "train"

    def __len__(self):
        return len(self.cores)

    def materialise(self, indices, *, region=None):
        """``ConstructSummary`` rows for the drawn indices. No outcome value, a bin class."""
        from .her2_routes_serial import her2_construct
        out = []
        for index in indices:
            position = int(index)
            core = self.cores[position]
            construct = her2_construct(self.scaffold, core, record_id=f"display{position}")
            if region is not None:
                construct = with_sites(construct, tuple(region))
            out.append(ConstructSummary(
                construct=construct, study=self.study, assay=self.assay, target=self.target,
                parent=f"{self.source}::wild_type", family=self.family, kind=KIND_ORDINAL,
                measurements=(ReplicateMeasurement(
                    value=None, lower=None, upper=None, censoring=CENSOR_NONE,
                    replicate=None, measurement_id=core),),
                region=tuple(construct.sites), mutated_sites=(),
                partition=PARTITION_TRAIN, scale=self.scale, source=self.source,
                ordinal_class=int(self.classes[position])))
        return tuple(out)

    def document(self):
        counts = np.bincount(np.asarray(self.classes, dtype=np.int64),
                             minlength=int(self.ordinal_classes))
        return {"rows": len(self.cores), "split": self.split, "source": self.source,
                "assay": self.assay, "ordinal_classes": int(self.ordinal_classes),
                "by_class": {str(index): int(value) for index, value in enumerate(counts)},
                "held_as": "arrays; summaries are materialised per drawn microbatch",
                "positives_only_would_see": int(counts[-1]) if counts.size else 0,
                "rule": ("the whole sorted-bin population, not the positives. A positives-only "
                         "objective cannot learn the low-versus-mid ordering, and the matched "
                         "positives-only arm is a separate control rather than a footnote.")}


def build_display_pool(root, *, scaffold, split="train", closure=None, positives_only=False,
                       classes=None):
    """The sorted-bin HER2 display library as an :class:`OrdinalPool`.

    The test split is not readable here at all: ``her2_data.load_split`` refuses it
    without a verified selection unlock, and this campaign never mints one. The
    train split's 367,042 rows are the population the ordinal head exists for.

    ``closure`` is applied to the cores up front and the refusals are counted. It
    is expected to refuse nothing: every reserved core sits at least three edits
    from the whole old display library by construction of the split, so nothing in
    the library can be within the radius-2 purge of one -- and asserting that here
    turns the argument into a measurement.

    ``positives_only=True`` is the matched control arm, not an option: it keeps the
    120,504 ``high`` rows and is what the full ordinal objective is compared
    against.
    """
    from . import her2_data
    from .her2_routes_sources import CANONICAL_STUDY, FAMILY_DISPLAY, SCALE_ORDINAL
    order = tuple(classes or her2_data.CLASS_ORDER)
    frame = her2_data.load_split(root, str(split))
    index = {name: position for position, name in enumerate(order)}
    cores = [str(value) for value in frame.seq.tolist()]
    labels = [int(index[str(value)]) for value in frame["class"].tolist()]
    refused = 0
    if closure is not None:
        kept_cores, kept_labels = [], []
        for core, label in zip(cores, labels):
            if closure.violates(window=core) is not None:
                refused += 1
                continue
            kept_cores.append(core)
            kept_labels.append(label)
        cores, labels = kept_cores, kept_labels
    if positives_only:
        positive = int(index[her2_data.POSITIVE_CLASS])
        pairs = [(core, label) for core, label in zip(cores, labels) if label == positive]
        cores = [core for core, _ in pairs]
        labels = [label for _, label in pairs]
    require(cores, f"The {split} display split produced no core after the closure")
    pool = OrdinalPool(
        cores=tuple(cores), classes=np.asarray(labels, dtype=np.int8), scaffold=scaffold,
        study=CANONICAL_STUDY.get("her2_display", "her2_display"),
        assay=f"{CANONICAL_STUDY.get('her2_display', 'her2_display')}:{FAMILY_DISPLAY}:HER2",
        target="HER2", family=FAMILY_DISPLAY, scale=SCALE_ORDINAL,
        ordinal_classes=len(order), split=str(split))
    return {"pool": pool, "rows_in_split": int(len(frame)), "refused_by_closure": refused,
            "positives_only": bool(positives_only),
            "closure_finding": (
                "the closure refused nothing, which is the expected measurement: every reserved "
                "core is at least the declared novelty separation from the whole old display "
                "library, so no library core can sit inside the radius purge around one."
                if refused == 0 else
                f"the closure refused {refused} display core(s). That contradicts the split's own "
                "novelty separation and is a finding, not a filter."),
            "test_split": ("never read. her2_data.load_split refuses it without a verified "
                           "selection unlock and this campaign mints none.")}


def task_gradient_fractions(gradients):
    """Each task's share of the total gradient norm. Reported beside its update share.

    The plan's own instruction: normalise losses over active targets and record the
    *gradients* as well as the update fractions. A task with 30% of the updates and
    3% of the gradient is not contributing 30%.
    """
    norms = {name: float(value) for name, value in dict(gradients).items()}
    total = sum(norms.values())
    return {"gradient_norms": norms,
            "fractions": ({name: value / total for name, value in sorted(norms.items())}
                          if total > 0 else {name: None for name in sorted(norms)}),
            "total": total,
            "reading": ("compare each fraction to its share of updates. A large update share with "
                        "a small gradient share means the task is not teaching, and that is an "
                        "amendment signal rather than a number to report and move past.")}


def measurement_kinds(observations):
    """Counts by measurement kind, so the four never silently collapse into one."""
    counts = {KIND_ORDINAL: 0, KIND_INTERVAL: 0, KIND_POINT: 0, KIND_ENRICHMENT: 0}
    censored = {"left": 0, "right": 0, "none": 0}
    for record in observations:
        counts[record.kind] = counts.get(record.kind, 0) + 1
        censored[record.censoring] = censored.get(record.censoring, 0) + 1
    return {"by_kind": counts, "by_censoring": censored,
            "rule": ("continuous, censored, ordinal and expression/enrichment measurements stay "
                     "distinguished in the likelihood, not only in a label")}
