"""Typed SET/SWAP programs, an independent provenance teacher, frozen event tapes and cohort contracts.

What this is. The data side of Build D (``functional_shared_transfer``): a tiny executable task over
the ORIGINAL residues of an antibody (heavy ``H`` and light ``L`` chains) whose answers are exact and
computed offline by a teacher, plus the contracts that keep the student honest. What it is not: a
biological model, a source of affinity labels, or a claim that learning the task helps anything. The
answers are computational facts about sequences, not measurements.

The task. A program has zero, one or two fixed-length operations over coordinates ``(chain, index)``:
``SET(coord, symbol)`` overwrites a position with a literal and ``SWAP(a, b)`` exchanges two positions.
A query reads the program's final state: ``RES(coord)`` (a residue) or ``EQ(a, b)`` (are two positions
equal). Coordinates are validated as integers (never bools), chains as ``H``/``L``, bounds against a
:class:`Layout`, symbols against its alphabet, arity per operation.

Layouts. A sequence has its own chain lengths. A :class:`Layout` is one ``(heavy, light, alphabet)``
shape; a :class:`LayoutMap` is the complete canonical candidate -> layout map of a roster whose rows have
different shapes. Every event is validated against ITS OWN sequence's layout: the student's coordinate
space (``CoordSpace(max heavy, max light)`` over the declared input union) never licenses a coordinate
that is out of range on a shorter row, and light coordinates are offset by the space's heavy size, never
by the row's. The layout map is part of the event table's hash, and therefore of every tape and run.

The teacher executes VALUES AND PROVENANCE. Every position holds a :class:`Cell`: ``Orig(coord)`` (the
original token that now sits there) or ``Literal(symbol)``. ``resolve`` returns the cells a query reads;
``evaluate_resolution`` turns them into a value using the original sequence. The independent value
oracle (a plain mutable list) lives in the tests, deliberately not here: an implementation is never
validated against itself.

Held-out dependency. ALL two-SWAP programs whose coordinate sets intersect in exactly one position
("chained" swaps, e.g. ``SWAP(a,b); SWAP(b,c)``) are excluded from auxiliary training. Training covers
every primitive, every coordinate and symbol, zero/one-operation programs, two DISJOINT swaps and
SET/SWAP mixtures. Two swaps on the same pair (a no-op) are valid programs but are never sampled. The
claim is "a new operational dependency", NOT a globally unseen Boolean or extensional function.
Template classes are canonicalised under coordinate renaming, SWAP operand symmetry and EQ operand
symmetry (the SET symbol is abstracted).

The PRIMARY held-out panel needs BOTH swaps: the query must be an EQ of two DISTINCT original origins
and deleting either swap, or reversing their order, must each change the resolved unordered origin
pair. Events record whether those counterfactuals change the ANSWER too, so panels can report
answer-changing cases and unchanged-answer controls separately. Literal-derived and constant events are
secondary strata and always labelled as such.

Generation. ``generate_events`` is a deterministic, source-only generator. Every accepted instance is
COMPLETE: four events, an EQ with answer true and an EQ with answer false that share one full canonical
program+query template (so the pair differs only in the data), and two RES queries. An instance that
cannot be completed (for example every untouched residue is equal, so no primary EQ is false) is
replaced by a deterministic, bounded redraw of the same class; the number and reasons of replacements are
recorded. A redraw must DIFFER from the draw it replaces: what a rejected draw consumed from the
shuffled decks (SET coordinates and symbols, swap coordinates) is owed to a later cycle, not handed
back, because handing it back repeats an identical program whenever the pending queue pins it. Coverage
is a property of the ACCEPTED events and is checked there (:func:`coverage_report`), never assumed.
A sequence that cannot supply a complete instance at all is REFUSED, never shortened.
The one exception to "a matched true/false pair of the goal stratum" is the literal-literal control: a
program of at most two SETs and distinct EQ coordinates reads at most ONE literal-literal pair, so no
true/false pair of it can exist. Its instance keeps a matched PRIMARY pair, the single literal-literal
control (one-sided, balance reported unsupported) and ONE RES query, still four unique answers.
Event identity is canonical: ``candidate|program|query`` is the only id an event may carry.

Tapes. The resulting :class:`EventTable` and the per-seed :class:`SeedTape` (functional batches,
auxiliary batches, the shuffled control targets, mask-binding lengths) are canonical, read-only and
hashed. The student never sees a teacher record: see ``functional_shared_transfer`` for the typed
tensors it does receive.

Cohorts. :func:`parse_cohort_manifest` / :func:`validate_cohorts` bind ONE verified native token cache
to five declared roles (source_train, source_aux, source_dev, target_train, target_eval), refuse rows
outside the declared union, overlapping physical sequences or groups, wrong cache splits and target
rosters of mixed full-identity length (never truncated). Source and auxiliary rows may have different
layouts; a row with fewer than ``MIN_COORDINATES`` coordinates cannot host the task and is refused.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass

import numpy as np

from . import functional_baseline as fb
from . import functional_matrix as fm
from . import functional_native_export as fne
from .her2_runtime import require

COHORT_SCHEMA = "functional-cohort-manifest/1"
EVENT_TABLE_SCHEMA = "functional-composition-events/1"
SEED_TAPE_SCHEMA = "functional-composition-tape/1"

OP_SET, OP_SWAP = "SET", "SWAP"
QUERY_RES, QUERY_EQ = "RES", "EQ"
CHAIN_HEAVY, CHAIN_LIGHT = "H", "L"
CHAINS = (CHAIN_HEAVY, CHAIN_LIGHT)
MAX_OPERATIONS = 2
MIN_COORDINATES = 6

OUTPUT_RESIDUE, OUTPUT_EQUALITY = "residue", "equality"

STRATUM_RES_ORIGINAL = "res_original"
STRATUM_RES_LITERAL = "res_literal"
STRATUM_EQ_PRIMARY = "eq_primary"
STRATUM_EQ_ORIGINAL_LITERAL = "eq_original_literal"
STRATUM_EQ_LITERAL_LITERAL = "eq_literal_literal"
STRATUM_EQ_SAME_ORIGIN = "eq_same_origin"
STRATA = (STRATUM_RES_ORIGINAL, STRATUM_RES_LITERAL, STRATUM_EQ_PRIMARY,
          STRATUM_EQ_ORIGINAL_LITERAL, STRATUM_EQ_LITERAL_LITERAL, STRATUM_EQ_SAME_ORIGIN)

DEP_NONE, DEP_IGNORES, DEP_ONE = "no_program", "ignores_program", "op_required"
DEP_BOTH, DEP_PARTIAL = "both_required", "partial"
DEPENDENCIES = (DEP_NONE, DEP_IGNORES, DEP_ONE, DEP_BOTH, DEP_PARTIAL)

CLASS_ZERO, CLASS_SET, CLASS_SWAP = "zero", "set", "swap"
CLASS_SET_SET, CLASS_SET_SWAP, CLASS_SWAP_SET = "set_set", "set_swap", "swap_set"
CLASS_SWAP_DISJOINT, CLASS_SWAP_CHAINED, CLASS_SWAP_SAME = (
    "swap_swap_disjoint", "swap_swap_chained", "swap_swap_same")
TRAIN_CLASSES = (CLASS_ZERO, CLASS_SET, CLASS_SWAP, CLASS_SET_SET, CLASS_SET_SWAP, CLASS_SWAP_SET,
                 CLASS_SWAP_DISJOINT)

UNIVERSE_TRAIN, UNIVERSE_HELD_OUT = "train", "held_out"
UNIVERSES = (UNIVERSE_TRAIN, UNIVERSE_HELD_OUT)
EVENTS_PER_INSTANCE = 4
HELD_OUT_MIN_EVENTS = 2

# student instruction vocabulary (integers only; see ``compile_instruction``)
KIND_PAD, KIND_FUNCTION, KIND_SET, KIND_SWAP, KIND_RES, KIND_EQ = 0, 1, 2, 3, 4, 5
KIND_READ_ORIGINAL, KIND_READ_LITERAL = 6, 7
KIND_REL_ORIGINAL_ORIGINAL, KIND_REL_ORIGINAL_LITERAL, KIND_REL_LITERAL_LITERAL = 8, 9, 10
N_KINDS = 11
NONE_FIELD = -1
MAX_INSTRUCTIONS = 3
INSTRUCTION_FIELDS = ("kind", "a", "b", "s", "t")
FUNCTION_INSTRUCTION = ((KIND_FUNCTION, NONE_FIELD, NONE_FIELD, NONE_FIELD, NONE_FIELD),)

STREAM_FUNCTIONAL, STREAM_AUX, STREAM_Q = 11, 12, 13
STREAM_UNIVERSE = {UNIVERSE_TRAIN: 21, UNIVERSE_HELD_OUT: 22}

ROLES = ("source_train", "source_aux", "source_dev", "target_train", "target_eval")
EXPECTED_SPLIT = {"source_train": fb.SPLIT_TRAIN, "source_aux": fb.SPLIT_TRAIN,
                  "source_dev": fb.SPLIT_EVAL, "target_train": fb.SPLIT_TRAIN,
                  "target_eval": fb.SPLIT_EVAL}

MIN_STRATUM = 4
SHUFFLE_ATTEMPTS = 128
GENERATION_ATTEMPTS = 200


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def stable_int(*parts):
    """A process-independent non-negative integer from text parts (``hash`` is salted per process)."""
    text = json.dumps([str(part) for part in parts], separators=(",", ":"))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


# ---------------------------------------------------------------------------
# coordinates, layout, operations, queries
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Coord:
    """A position ``(chain, index)``: chain ``H`` or ``L``, a non-negative integer (never a bool)."""

    chain: str
    index: int

    def __post_init__(self):
        require(isinstance(self.chain, str) and self.chain in CHAINS,
                f"a coordinate's chain must be one of {CHAINS}, got {self.chain!r}")
        require(_is_int(self.index) and int(self.index) >= 0,
                f"a coordinate's index must be a non-negative integer (not a bool), got "
                f"{self.index!r}")
        object.__setattr__(self, "index", int(self.index))

    @property
    def label(self):
        return f"{self.chain}{self.index}"

    def document(self):
        return [self.chain, self.index]


def as_coord(value):
    """A :class:`Coord` from a Coord or a ``[chain, index]`` pair; anything else is refused."""
    if isinstance(value, Coord):
        return value
    require(isinstance(value, (list, tuple)) and len(value) == 2,
            f"a coordinate is a [chain, index] pair, got {value!r}")
    return Coord(value[0], value[1])


@dataclass(frozen=True)
class CoordSpace:
    """The coordinate-id space the student's embeddings use: heavy ids first, then light ids.

    The space is sized over the DECLARED INPUT UNION (the longest heavy and light chain of any row), so
    a light coordinate's id is ``heavy + index`` for every row, never an offset by that row's own heavy
    length. It is NOT a bound check for a row: each event is validated against its own :class:`Layout`.
    """

    heavy: int
    light: int

    def __post_init__(self):
        require(_is_int(self.heavy) and self.heavy >= 1 and _is_int(self.light) and self.light >= 0,
                f"a coordinate space needs heavy >= 1 and light >= 0, got {self.heavy!r}/"
                f"{self.light!r}")

    @property
    def n_coords(self):
        return int(self.heavy) + int(self.light)

    def coord_id(self, coord):
        coord = as_coord(coord)
        limit = self.heavy if coord.chain == CHAIN_HEAVY else self.light
        require(coord.index < limit,
                f"coordinate {coord.label} is outside the {limit}-position {coord.chain} chain of "
                "this coordinate space")
        return coord.index if coord.chain == CHAIN_HEAVY else int(self.heavy) + coord.index


@dataclass(frozen=True)
class Layout:
    """Chain lengths and the residue alphabet the task is defined over (one shape)."""

    heavy: int
    light: int
    alphabet: str

    def __post_init__(self):
        require(_is_int(self.heavy) and self.heavy >= 1, f"layout heavy must be >= 1, got "
                                                          f"{self.heavy!r}")
        require(_is_int(self.light) and self.light >= 0, f"layout light must be >= 0, got "
                                                          f"{self.light!r}")
        require(isinstance(self.alphabet, str) and self.alphabet and self.alphabet.isalpha()
                and self.alphabet == self.alphabet.upper()
                and len(set(self.alphabet)) == len(self.alphabet),
                f"layout alphabet must be distinct upper-case letters, got {self.alphabet!r}")
        object.__setattr__(self, "heavy", int(self.heavy))
        object.__setattr__(self, "light", int(self.light))

    @property
    def n_coords(self):
        return self.heavy + self.light

    @property
    def n_symbols(self):
        return len(self.alphabet)

    @property
    def label(self):
        return f"H{self.heavy}L{self.light}"

    def coords(self):
        return (tuple(Coord(CHAIN_HEAVY, i) for i in range(self.heavy))
                + tuple(Coord(CHAIN_LIGHT, j) for j in range(self.light)))

    def layout_for(self, candidate_id):
        """A fixed layout applies to every candidate (a :class:`LayoutMap` differs per candidate)."""
        return self

    def classes(self):
        """The distinct ``(heavy, light)`` shapes this layout source covers."""
        return ((self.heavy, self.light),)

    def check_coord(self, value):
        coord = as_coord(value)
        limit = self.heavy if coord.chain == CHAIN_HEAVY else self.light
        require(coord.index < limit,
                f"coordinate {coord.label} is outside the {limit}-position {coord.chain} chain")
        return coord

    def check_symbol(self, symbol):
        require(isinstance(symbol, str) and len(symbol) == 1 and symbol in self.alphabet,
                f"symbol {symbol!r} is not in the alphabet {self.alphabet!r}")
        return symbol

    def symbol_index(self, symbol):
        return self.alphabet.index(self.check_symbol(symbol))

    def check_program(self, program):
        require(isinstance(program, Program), f"expected a Program, got {type(program).__name__}")
        for op in program.ops:
            if isinstance(op, SetOp):
                self.check_coord(op.coord)
                self.check_symbol(op.symbol)
            else:
                self.check_coord(op.a)
                self.check_coord(op.b)
        return program

    def check_query(self, query):
        require(isinstance(query, (ResQuery, EqQuery)),
                f"expected a RES or EQ query, got {type(query).__name__}")
        for coord in query.coords():
            self.check_coord(coord)
        return query

    def space(self):
        return CoordSpace(self.heavy, self.light)

    def document(self):
        return {"heavy": self.heavy, "light": self.light, "alphabet": self.alphabet}


@dataclass(frozen=True, eq=False)
class LayoutMap:
    """The complete canonical candidate -> :class:`Layout` map of a roster with different shapes.

    ``shapes`` is a sorted tuple of ``(candidate_id, heavy, light)``. The whole map is part of the
    identity of an event table (and so of every tape, checkpoint and adapter bound to it): changing one
    candidate's chain lengths changes the hash. It offers the same ``layout_for``/``classes``/
    ``alphabet``/``document`` surface as :class:`Layout`, so code that only needs a candidate's layout
    treats the two alike.
    """

    alphabet: str
    shapes: tuple

    def __post_init__(self):
        entries = []
        for item in self.shapes:
            require(isinstance(item, (list, tuple)) and len(item) == 3,
                    f"a layout map entry is (candidate_id, heavy, light), got {item!r}")
            cid, heavy, light = item
            require(isinstance(cid, str) and cid and _is_int(heavy) and _is_int(light),
                    f"a layout map entry needs a candidate id and integer chain lengths, got "
                    f"{item!r}")
            entries.append((cid, int(heavy), int(light)))
        entries.sort()
        require(entries, "a layout map names at least one candidate")
        require(len({cid for cid, _, _ in entries}) == len(entries),
                "a layout map repeats a candidate id")
        by_shape = {}
        for _, heavy, light in entries:
            if (heavy, light) not in by_shape:
                by_shape[(heavy, light)] = Layout(heavy, light, self.alphabet)
        object.__setattr__(self, "shapes", tuple(entries))
        object.__setattr__(self, "_by_shape", by_shape)
        object.__setattr__(self, "_by_candidate",
                           {cid: by_shape[(heavy, light)] for cid, heavy, light in entries})

    def layout_for(self, candidate_id):
        require(candidate_id in self._by_candidate,
                f"candidate {candidate_id!r} has no layout in this layout map")
        return self._by_candidate[candidate_id]

    def classes(self):
        return tuple(sorted(self._by_shape))

    def counts(self):
        """Candidates per ``H<h>L<l>`` shape label."""
        out = {}
        for _, heavy, light in self.shapes:
            label = self._by_shape[(heavy, light)].label
            out[label] = out.get(label, 0) + 1
        return out

    def candidate_ids(self):
        return tuple(cid for cid, _, _ in self.shapes)

    def document(self):
        return {"alphabet": self.alphabet,
                "candidates": [[cid, heavy, light] for cid, heavy, light in self.shapes]}


def layouts_sha256(layouts):
    """The identity of a Layout or LayoutMap."""
    return fb.digest(layouts.document())


def layouts_for_rows(rows, alphabet):
    """The layout source of ``(candidate_id, heavy, light)`` rows: one :class:`Layout` when every row
    has the same shape (the fixed-length case), otherwise a :class:`LayoutMap`. A row with fewer than
    ``MIN_COORDINATES`` coordinates (or no heavy residue) cannot host the SET/SWAP task and is refused
    by name rather than dropped."""
    rows = [(str(cid), heavy, light) for cid, heavy, light in rows]
    require(rows, "a layout source needs at least one row")
    small = [(cid, len(heavy), len(light)) for cid, heavy, light in rows
             if len(heavy) < 1 or len(heavy) + len(light) < MIN_COORDINATES]
    require(not small,
            f"{len(small)} candidate(s) have fewer than {MIN_COORDINATES} coordinates (or no heavy "
            f"residue), e.g. {[f'{cid} (H{h}/L{l})' for cid, h, l in small[:3]]}: the SET/SWAP task "
            "cannot be defined over them. Remove them from the cohort that needs the task")
    shapes = {(len(heavy), len(light)) for _, heavy, light in rows}
    if len(shapes) == 1:
        heavy, light = next(iter(shapes))
        return Layout(heavy, light, alphabet)
    return LayoutMap(alphabet, tuple((cid, len(heavy), len(light)) for cid, heavy, light in rows))


@dataclass(frozen=True)
class SetOp:
    """``SET(coord, symbol)``: the position now holds the literal ``symbol``."""

    coord: Coord
    symbol: str
    kind = OP_SET

    def __post_init__(self):
        require(isinstance(self.coord, Coord), "SET needs a Coord")
        require(isinstance(self.symbol, str) and len(self.symbol) == 1 and self.symbol.isalpha()
                and self.symbol == self.symbol.upper(),
                f"SET needs one upper-case residue letter, got {self.symbol!r}")

    def coords(self):
        return (self.coord,)


@dataclass(frozen=True)
class SwapOp:
    """``SWAP(a, b)``: two DISTINCT positions exchange contents. Symmetric, so stored sorted."""

    a: Coord
    b: Coord
    kind = OP_SWAP

    def __post_init__(self):
        require(isinstance(self.a, Coord) and isinstance(self.b, Coord), "SWAP needs two Coords")
        require(self.a != self.b, f"SWAP needs two distinct coordinates, got {self.a.label} twice")
        if self.b < self.a:
            first, second = self.b, self.a
            object.__setattr__(self, "a", first)
            object.__setattr__(self, "b", second)

    def coords(self):
        return (self.a, self.b)


@dataclass(frozen=True)
class Program:
    """Zero, one or two operations, applied in order."""

    ops: tuple = ()

    def __post_init__(self):
        ops = tuple(self.ops)
        require(len(ops) <= MAX_OPERATIONS,
                f"a program has at most {MAX_OPERATIONS} operations, got {len(ops)}")
        require(all(isinstance(op, (SetOp, SwapOp)) for op in ops),
                "a program holds only SET and SWAP operations")
        object.__setattr__(self, "ops", ops)


@dataclass(frozen=True)
class ResQuery:
    """``RES(coord)``: the residue at ``coord`` after the program."""

    coord: Coord
    kind = QUERY_RES

    def __post_init__(self):
        require(isinstance(self.coord, Coord), "RES needs a Coord")

    def coords(self):
        return (self.coord,)


@dataclass(frozen=True)
class EqQuery:
    """``EQ(a, b)``: are two DISTINCT positions equal after the program. Symmetric, stored sorted."""

    a: Coord
    b: Coord
    kind = QUERY_EQ

    def __post_init__(self):
        require(isinstance(self.a, Coord) and isinstance(self.b, Coord), "EQ needs two Coords")
        require(self.a != self.b, f"EQ needs two distinct coordinates, got {self.a.label} twice")
        if self.b < self.a:
            first, second = self.b, self.a
            object.__setattr__(self, "a", first)
            object.__setattr__(self, "b", second)

    def coords(self):
        return (self.a, self.b)


def output_type(query):
    return OUTPUT_RESIDUE if isinstance(query, ResQuery) else OUTPUT_EQUALITY


def op_document(op):
    if isinstance(op, SetOp):
        return [OP_SET, op.coord.document(), op.symbol]
    return [OP_SWAP, op.a.document(), op.b.document()]


def parse_op(document):
    require(isinstance(document, (list, tuple)) and document,
            f"an operation is [name, ...], got {document!r}")
    name = document[0]
    if name == OP_SET:
        require(len(document) == 3, f"SET takes (coord, symbol); got {len(document) - 1} argument(s)")
        return SetOp(as_coord(document[1]), document[2])
    if name == OP_SWAP:
        require(len(document) == 3, f"SWAP takes (a, b); got {len(document) - 1} argument(s)")
        return SwapOp(as_coord(document[1]), as_coord(document[2]))
    raise ValueError(f"unknown operation {name!r}: the operations are {OP_SET} and {OP_SWAP}")


def program_document(program):
    return [op_document(op) for op in program.ops]


def parse_program(document, layout=None):
    require(isinstance(document, (list, tuple)), f"a program is a list of operations, got {document!r}")
    program = Program(tuple(parse_op(item) for item in document))
    return program if layout is None else layout.check_program(program)


def query_document(query):
    if isinstance(query, ResQuery):
        return [QUERY_RES, query.coord.document()]
    return [QUERY_EQ, query.a.document(), query.b.document()]


def parse_query(document, layout=None):
    require(isinstance(document, (list, tuple)) and document,
            f"a query is [name, ...], got {document!r}")
    name = document[0]
    if name == QUERY_RES:
        require(len(document) == 2, f"RES takes one coordinate; got {len(document) - 1} argument(s)")
        query = ResQuery(as_coord(document[1]))
    elif name == QUERY_EQ:
        require(len(document) == 3, f"EQ takes two coordinates; got {len(document) - 1} argument(s)")
        query = EqQuery(as_coord(document[1]), as_coord(document[2]))
    else:
        raise ValueError(f"unknown query {name!r}: the queries are {QUERY_RES} and {QUERY_EQ}")
    return query if layout is None else layout.check_query(query)


def op_key(op):
    if isinstance(op, SetOp):
        return f"SET({op.coord.label}={op.symbol})"
    return f"SWAP({op.a.label},{op.b.label})"


def program_key(program):
    return ";".join(op_key(op) for op in program.ops) if program.ops else "NOP"


def query_key(query):
    return f"{query.kind}({','.join(coord.label for coord in query.coords())})"


def event_id(candidate_id, program, query):
    """The identity of one computational answer: (candidate, canonical program, canonical query)."""
    require(isinstance(candidate_id, str) and candidate_id and "|" not in candidate_id,
            f"a candidate id used in an event id must be non-empty text without '|', got "
            f"{candidate_id!r}")
    return f"{candidate_id}|{program_key(program)}|{query_key(query)}"


# ---------------------------------------------------------------------------
# the provenance teacher
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """What now sits at a position: ``Orig(coord)`` (an original token) or ``Literal(symbol)``."""

    coord: object = None
    symbol: object = None

    def __post_init__(self):
        require((self.coord is None) != (self.symbol is None),
                "a cell is exactly one of Orig(coord) or Literal(symbol)")

    @staticmethod
    def original(coord):
        return Cell(coord=as_coord(coord))

    @staticmethod
    def literal(symbol):
        return Cell(symbol=symbol)

    @property
    def is_literal(self):
        return self.symbol is not None

    @property
    def key(self):
        """Sortable identity: originals first (by coordinate), then literals (by symbol)."""
        if self.is_literal:
            return (1, self.symbol, -1)
        return (0, self.coord.chain, self.coord.index)

    @property
    def label(self):
        return f"Literal({self.symbol})" if self.is_literal else f"Orig({self.coord.label})"

    def value(self, heavy, light):
        return self.symbol if self.is_literal else read_residue(heavy, light, self.coord)

    def document(self):
        return ["lit", self.symbol] if self.is_literal else ["orig", *self.coord.document()]


def read_residue(heavy, light, coord):
    chain = heavy if coord.chain == CHAIN_HEAVY else light
    require(coord.index < len(chain),
            f"coordinate {coord.label} is outside the {len(chain)}-residue {coord.chain} chain")
    return chain[coord.index]


def provenance_state(layout, program):
    """Execute ``program`` over provenance: coordinate -> Cell after the last operation."""
    layout.check_program(program)
    state = {coord: Cell.original(coord) for coord in layout.coords()}
    for op in program.ops:
        if isinstance(op, SetOp):
            state[op.coord] = Cell.literal(op.symbol)
        else:
            state[op.a], state[op.b] = state[op.b], state[op.a]
    return state


@dataclass(frozen=True)
class Resolution:
    """The cells a query reads after a program: one for RES, two for EQ."""

    query: object
    operands: tuple

    @property
    def stratum(self):
        return classify(self)

    def document(self):
        return [cell.document() for cell in self.operands]


def _cells(state, query):
    return tuple(state[coord] for coord in query.coords())


def resolve(layout, program, query):
    layout.check_query(query)
    return Resolution(query=query, operands=_cells(provenance_state(layout, program), query))


def classify(resolution):
    """The event's dependency stratum. A primary EQ compares two DISTINCT original origins."""
    cells = resolution.operands
    if len(cells) == 1:
        return STRATUM_RES_LITERAL if cells[0].is_literal else STRATUM_RES_ORIGINAL
    first, second = cells
    if first.is_literal and second.is_literal:
        return STRATUM_EQ_LITERAL_LITERAL
    if first.is_literal or second.is_literal:
        return STRATUM_EQ_ORIGINAL_LITERAL
    if first.coord == second.coord:
        return STRATUM_EQ_SAME_ORIGIN
    return STRATUM_EQ_PRIMARY


def evaluate_resolution(resolution, heavy, light):
    values = [cell.value(heavy, light) for cell in resolution.operands]
    return values[0] if len(values) == 1 else values[0] == values[1]


def answer(layout, heavy, light, program, query):
    """The exact answer: a residue symbol for RES, a bool for EQ."""
    return evaluate_resolution(resolve(layout, program, query), heavy, light)


def answer_code(layout, value, query):
    """Integer target: the symbol index for RES, 0/1 for EQ."""
    if isinstance(query, ResQuery):
        return layout.symbol_index(value)
    require(isinstance(value, (bool, np.bool_)), f"an EQ answer is a bool, got {value!r}")
    return int(bool(value))


def _signature(cells):
    """Order-free identity of what a query reads."""
    return tuple(sorted(cell.key for cell in cells))


def counterfactual_programs(program):
    """Delete the first op, delete the second op, reverse the order (two-operation programs only)."""
    require(len(program.ops) == 2, "counterfactual programs need a two-operation program")
    first, second = program.ops
    return {"delete_first": Program((second,)), "delete_second": Program((first,)),
            "reversed": Program((second, first))}


def depends_on_both(layout, program, query):
    """True when the query reads one (RES) or two DISTINCT (EQ) ORIGINAL origins and deleting either
    operation, or reversing the order, changes the resolved unordered origin pair."""
    base = resolve(layout, program, query)
    if classify(base) not in (STRATUM_EQ_PRIMARY, STRATUM_RES_ORIGINAL):
        return False
    signature = _signature(base.operands)
    return all(_signature(resolve(layout, variant, query).operands) != signature
               for variant in counterfactual_programs(program).values())


def dependency_tag(layout, program, query):
    """How much of the program the query depends on (by resolved origins, not by the answer)."""
    if not program.ops:
        return DEP_NONE
    signature = _signature(_cells(provenance_state(layout, program), query))
    if _signature(_cells(provenance_state(layout, Program(())), query)) == signature:
        return DEP_IGNORES
    if len(program.ops) == 1:
        return DEP_ONE
    changed = [_signature(_cells(provenance_state(layout, variant), query)) != signature
               for variant in counterfactual_programs(program).values()]
    return DEP_BOTH if all(changed) else DEP_PARTIAL


def counterfactual_answer_changes(layout, heavy, light, program, query):
    """For a two-operation program: does each counterfactual change the ANSWER (not just the origins)?"""
    base = answer(layout, heavy, light, program, query)
    return {name: answer(layout, heavy, light, variant, query) != base
            for name, variant in counterfactual_programs(program).items()}


# ---------------------------------------------------------------------------
# template classes and the held-out dependency
# ---------------------------------------------------------------------------

def _template_form(items, flips):
    labels, out, flip_iter = {}, [], iter(flips)
    for name, coords, symmetric in items:
        use = tuple(reversed(coords)) if symmetric and next(flip_iter) else coords
        numbered = []
        for coord in use:
            if coord not in labels:
                labels[coord] = len(labels)
            numbered.append(labels[coord])
        out.append((name, tuple(numbered)))
    return tuple(out)


def template_id(program, query=None):
    """Canonical template of a program (and optional query) under coordinate renaming and SWAP/EQ
    operand symmetry. The SET symbol is abstracted (``*``): the claim is about structure, not literals."""
    items = [(op.kind, op.coords(), isinstance(op, SwapOp)) for op in program.ops]
    if query is not None:
        items.append((query.kind, query.coords(), isinstance(query, EqQuery)))
    symmetric = sum(1 for item in items if item[2])
    best = min(_template_form(items, flips)
               for flips in itertools.product((False, True), repeat=symmetric))
    parts = []
    for name, numbered in best[:len(program.ops)]:
        inner = ",".join(f"v{i}" for i in numbered) + ("=*" if name == OP_SET else "")
        parts.append(f"{name}({inner})")
    text = ";".join(parts) if parts else "NOP"
    if query is not None:
        name, numbered = best[-1]
        text += f"|{name}({','.join(f'v{i}' for i in numbered)})"
    return text


def _swap_overlap(program):
    if len(program.ops) != 2 or not all(isinstance(op, SwapOp) for op in program.ops):
        return None
    first, second = (set(op.coords()) for op in program.ops)
    return len(first & second)


def is_chained_swap_pair(program):
    """The HELD-OUT class: two SWAPs whose coordinate sets intersect in exactly one position."""
    return _swap_overlap(program) == 1


def program_class(program):
    kinds = tuple(op.kind for op in program.ops)
    if kinds == ():
        return CLASS_ZERO
    if kinds == (OP_SET,):
        return CLASS_SET
    if kinds == (OP_SWAP,):
        return CLASS_SWAP
    if kinds == (OP_SET, OP_SET):
        return CLASS_SET_SET
    if kinds == (OP_SET, OP_SWAP):
        return CLASS_SET_SWAP
    if kinds == (OP_SWAP, OP_SET):
        return CLASS_SWAP_SET
    return {0: CLASS_SWAP_DISJOINT, 1: CLASS_SWAP_CHAINED, 2: CLASS_SWAP_SAME}[_swap_overlap(program)]


# ---------------------------------------------------------------------------
# the typed instruction rows the student is compiled from
# ---------------------------------------------------------------------------

def _row(kind, a=NONE_FIELD, b=NONE_FIELD, s=NONE_FIELD, t=NONE_FIELD):
    return (int(kind), int(a), int(b), int(s), int(t))


def compile_instruction(space, layout, program, query):
    """The student's COMPOSED instruction: one token per operation, then the query. Integers only:
    ``(kind, coord_a, coord_b, symbol_s, symbol_t)`` with ``-1`` for an unused field. It is compiled
    from the ORIGINAL program: no answer, trace, transformed sequence or origin.

    ``layout`` is the layout of the ROW the program is asked about: every coordinate is checked against
    it, so the (larger) shared ``space`` can never license a coordinate that is out of range there."""
    layout.check_program(program)
    layout.check_query(query)
    require(space.heavy >= layout.heavy and space.light >= layout.light,
            "the coordinate space is smaller than the layout")
    rows = []
    for op in program.ops:
        if isinstance(op, SetOp):
            rows.append(_row(KIND_SET, space.coord_id(op.coord), s=layout.symbol_index(op.symbol)))
        else:
            rows.append(_row(KIND_SWAP, space.coord_id(op.a), space.coord_id(op.b)))
    if isinstance(query, ResQuery):
        rows.append(_row(KIND_RES, space.coord_id(query.coord)))
    else:
        rows.append(_row(KIND_EQ, space.coord_id(query.a), space.coord_id(query.b)))
    return tuple(rows)


def compile_direct(space, layout, resolution):
    """The R-control instruction: the teacher's RESOLVED operands (original coordinates and/or
    literals) as ONE simpler token. It carries no residue of the candidate's own sequence, only
    coordinates and any SET literal."""
    cells = sorted(resolution.operands, key=lambda cell: cell.key)
    if len(cells) == 1:
        cell = cells[0]
        if cell.is_literal:
            return (_row(KIND_READ_LITERAL, s=layout.symbol_index(cell.symbol)),)
        layout.check_coord(cell.coord)
        return (_row(KIND_READ_ORIGINAL, space.coord_id(cell.coord)),)
    first, second = cells
    if first.is_literal:
        return (_row(KIND_REL_LITERAL_LITERAL, s=layout.symbol_index(first.symbol),
                     t=layout.symbol_index(second.symbol)),)
    if second.is_literal:
        layout.check_coord(first.coord)
        return (_row(KIND_REL_ORIGINAL_LITERAL, space.coord_id(first.coord),
                     s=layout.symbol_index(second.symbol)),)
    layout.check_coord(first.coord)
    layout.check_coord(second.coord)
    return (_row(KIND_REL_ORIGINAL_ORIGINAL, space.coord_id(first.coord),
                 space.coord_id(second.coord)),)


# ---------------------------------------------------------------------------
# teacher events and their deterministic generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class TeacherEvent:
    """One exact computational question and its answer. A TEACHER record: it holds the answer, the
    resolved origins and counterfactual flags, and is never given to the student.

    ``event_id`` must be the canonical id of (candidate, program, query). An alias would let one
    question be stored and counted twice, so it is refused at construction, before any table, count,
    serialisation or parse."""

    event_id: str
    candidate_id: str
    program: Program
    query: object
    answer: object
    answer_code: int
    output_type: str
    template_id: str
    stratum: str
    dependency: str
    held_out: bool
    resolution: Resolution
    flags: object = None

    def __post_init__(self):
        canonical = event_id(self.candidate_id, self.program, self.query)
        require(self.event_id == canonical,
                f"event id {self.event_id!r} is not the canonical id {canonical!r} of its "
                "candidate, program and query: an alias would count one question twice")

    def document(self):
        return {"id": self.event_id, "candidate": self.candidate_id,
                "program": program_document(self.program), "query": query_document(self.query),
                "answer": self.answer, "code": self.answer_code, "output": self.output_type,
                "template": self.template_id, "stratum": self.stratum,
                "dependency": self.dependency, "held_out": self.held_out,
                "resolved": self.resolution.document(),
                "flags": None if self.flags is None else dict(self.flags)}


def _make_event(layout, candidate_id, program, query, resolution, value):
    flags = None
    return TeacherEvent(
        event_id=event_id(candidate_id, program, query), candidate_id=candidate_id,
        program=program, query=query, answer=value, answer_code=answer_code(layout, value, query),
        output_type=output_type(query), template_id=template_id(program, query),
        stratum=classify(resolution), dependency=dependency_tag(layout, program, query),
        held_out=is_chained_swap_pair(program), resolution=resolution, flags=flags)


def _with_flags(layout, heavy, light, event):
    if len(event.program.ops) != 2:
        return event
    flags = counterfactual_answer_changes(layout, heavy, light, event.program, event.query)
    return TeacherEvent(**{**event.__dict__, "flags": flags})


class _Deck:
    """Shuffled-deck sampler: every item is drawn once per cycle.

    A draw that ends in a REJECTED instance is consumed, not handed back. Handing it back (restoring
    ``queue``) made the next attempt identical whenever the pending queue pinned it: one pending SET
    coordinate is the first SET of every retry, and two pending coordinates and symbols are the whole
    program, so advancing the RNG changes nothing. What a rejected attempt consumed is instead OWED
    (:meth:`defer`): the next cycle is built from the items that are not owed and the owed ones are queued
    LAST, so a rejected coordinate or symbol is presented again later while the retry sees a different one
    now. A cycle is therefore still each item once. Whether the ACCEPTED events cover every coordinate and
    symbol is checked by :func:`coverage_report`; nothing here promises it."""

    def __init__(self, items, rng):
        self.items, self.rng, self.queue = list(items), rng, []
        self.drawn, self.owed = [], []

    def draw(self):
        if not self.queue:
            fresh = [int(i) for i in self.rng.permutation(len(self.items))]
            owed = set(self.owed)
            self.queue = list(reversed(self.owed)) + [i for i in fresh if i not in owed]
            self.owed = []
        index = self.queue.pop()
        self.drawn.append(index)
        return self.items[index]

    def begin(self):
        """Start recording what the next attempt consumes."""
        self.drawn = []

    def defer(self):
        """The attempt was rejected: what it consumed is owed to a later cycle."""
        for index in self.drawn:
            if index not in self.owed:
                self.owed.append(index)
        self.drawn = []


def _random_other(layout, rng, avoid):
    pool = [coord for coord in layout.coords() if coord not in avoid]
    return pool[int(rng.integers(len(pool)))]


def _draw_train_program(klass, layout, decks, rng):
    def make_set():
        return SetOp(decks["set_coord"].draw(), decks["set_symbol"].draw())

    def make_swap():
        a = decks["swap_coord"].draw()
        return SwapOp(a, _random_other(layout, rng, (a,)))
    if klass == CLASS_ZERO:
        return Program(())
    if klass == CLASS_SET:
        return Program((make_set(),))
    if klass == CLASS_SWAP:
        return Program((make_swap(),))
    if klass == CLASS_SET_SET:
        return Program((make_set(), make_set()))
    if klass == CLASS_SET_SWAP:
        return Program((make_set(), make_swap()))
    if klass == CLASS_SWAP_SET:
        return Program((make_swap(), make_set()))
    first = make_swap()
    used = (first.a, first.b)
    a = _random_other(layout, rng, used)
    b = _random_other(layout, rng, used + (a,))
    return Program((first, SwapOp(a, b)))


def _draw_chain_program(layout, decks, rng):
    a = decks["chain_coord"].draw()
    b = _random_other(layout, rng, (a,))
    c = _random_other(layout, rng, (a, b))
    ops = (SwapOp(a, b), SwapOp(b, c))
    return Program(ops if int(rng.integers(2)) == 0 else ops[::-1])


def _universe(layout, program, heavy, light):
    """Every RES and EQ query over the program's final state: ``(Resolution, answer)`` pairs."""
    state = provenance_state(layout, program)
    entries = []
    for coord in layout.coords():
        resolution = Resolution(ResQuery(coord), (state[coord],))
        entries.append((resolution, evaluate_resolution(resolution, heavy, light)))
    for a, b in itertools.combinations(layout.coords(), 2):
        resolution = Resolution(EqQuery(a, b), (state[a], state[b]))
        entries.append((resolution, evaluate_resolution(resolution, heavy, light)))
    return state, entries


def _pick(pool, rng):
    return pool[int(rng.integers(len(pool)))]


def _matched_pair(program, entries, rng):
    """One EQ with answer true and one with answer false that share ONE full canonical program+query
    template (``template_id(program, query)``), or ``None`` when no template has both answers.

    The template, not the program, is the unit: ``SWAP(H0,H1)`` with ``EQ(H0,L0)`` and with ``EQ(H0,H2)``
    is one template, while ``EQ(H2,L0)`` (positions the program never touches) is another. A matched
    pair therefore differs only in the data, never in what the query asks."""
    groups = {}
    for resolution, value in entries:
        slot = groups.setdefault(template_id(program, resolution.query), ([], []))
        slot[0 if value else 1].append((resolution, value))
    feasible = sorted(key for key, (yes, no) in groups.items() if yes and no)
    if not feasible:
        return None
    yes, no = groups[feasible[int(rng.integers(len(feasible)))]]
    return _pick(yes, rng), _pick(no, rng)


def _train_instance(layout, candidate_id, heavy, light, program, goal, rng, notes):
    """The events of one training instance, or ``[]`` when it cannot be completed.

    A complete instance is four events: a matched true/false EQ pair (same full program+query
    template) of the ``goal`` stratum - or, when the goal is a secondary stratum that has no matched
    pair, of the primary stratum, which is recorded - and two distinct RES queries.

    The literal-literal goal differs. A program of at most two SETs and distinct EQ coordinates reads at
    most ONE literal-literal pair (``SET(a,s); SET(b,t)`` with ``a != b``), so no true/false pair of it
    exists, and that single query is a valid control whose answer depends only on the two SET symbols
    (``SET(H0,A);SET(H1,A);EQ(H0,H1)`` is true, ``SET(H0,A);SET(H1,C);EQ(H0,H1)`` false). The instance
    therefore keeps a matched PRIMARY pair, that one control (balance unsupported, recorded) and ONE RES
    query: still four unique answers. When the program has no literal-literal query (both SETs hit one
    coordinate) the primary stratum is used, as for any secondary goal without a pair."""
    state, entries = _universe(layout, program, heavy, light)
    eq = [(r, v) for r, v in entries if isinstance(r.query, EqQuery)]
    res = [(r, v) for r, v in entries if isinstance(r.query, ResQuery)]

    def note(reason):
        notes.append({"candidate": candidate_id, "program": program_key(program),
                      "template": template_id(program), "goal": goal, "reason": reason})
    literal_goal = goal == STRATUM_EQ_LITERAL_LITERAL
    pair_stratum = STRATUM_EQ_PRIMARY if literal_goal else goal
    pair = _matched_pair(program, [(r, v) for r, v in eq if r.stratum == pair_stratum], rng)
    if pair is None and pair_stratum != STRATUM_EQ_PRIMARY:
        note(f"no {goal} EQ pair with both answers shares a full program+query template; used the "
             "primary stratum")
        pair = _matched_pair(program, [(r, v) for r, v in eq if r.stratum == STRATUM_EQ_PRIMARY], rng)
    if pair is None:
        note("no primary EQ pair with both answers shares a full program+query template")
        return []
    chosen = list(pair)
    if literal_goal:
        controls = [(r, v) for r, v in eq if r.stratum == STRATUM_EQ_LITERAL_LITERAL]
        if controls:
            chosen.append(_pick(controls, rng))
            note("literal-literal control kept ONE-SIDED: a program of at most two SETs and distinct EQ "
                 "coordinates reads at most one literal-literal pair, so no true/false pair of it "
                 "exists and its balance is unsupported; the instance holds a primary pair with both "
                 "answers, this control and one RES query")
        else:
            note("no literal-literal EQ exists in this program (its SETs share a coordinate); used the "
                 "primary stratum")
    moved = [(r, v) for r, v in res if state[r.query.coord] != Cell.original(r.query.coord)]
    first = _pick(moved or res, rng)
    chosen.append(first)
    if len(chosen) < EVENTS_PER_INSTANCE:
        rest = [item for item in res if item[0].query != first[0].query]
        chosen.append(_pick(rest, rng))
    return [_make_event(layout, candidate_id, program, r.query, r, v) for r, v in chosen]


def _held_out_instance(layout, candidate_id, heavy, light, program, rng, notes):
    """The events of one held-out instance, or ``[]`` when no primary EQ needing BOTH swaps has a
    matched true/false pair. Besides that pair: one EQ that ignores the program (a labelled control)
    and one RES that needs both swaps, when they exist."""
    state, entries = _universe(layout, program, heavy, light)
    variants = {name: provenance_state(layout, variant)
                for name, variant in counterfactual_programs(program).items()}
    base_empty = provenance_state(layout, Program(()))

    def both(resolution):
        query = resolution.query
        if classify(resolution) not in (STRATUM_EQ_PRIMARY, STRATUM_RES_ORIGINAL):
            return False
        signature = _signature(resolution.operands)
        return all(_signature(_cells(variant, query)) != signature for variant in variants.values())

    def ignores(resolution):
        return _signature(_cells(base_empty, resolution.query)) == _signature(resolution.operands)
    eq = [(r, v) for r, v in entries if isinstance(r.query, EqQuery)]
    res = [(r, v) for r, v in entries if isinstance(r.query, ResQuery)]
    pair = _matched_pair(program, [(r, v) for r, v in eq if both(r)], rng)
    if pair is None:
        notes.append({"candidate": candidate_id, "program": program_key(program),
                      "template": template_id(program), "goal": DEP_BOTH,
                      "reason": "no primary EQ needing both swaps has a matched true/false pair "
                                "(same full program+query template)"})
        return []
    chosen = list(pair)
    ignored = [(r, v) for r, v in eq if ignores(r)]
    if ignored:
        chosen.append(_pick(ignored, rng))
    needing = [(r, v) for r, v in res if both(r)]
    if needing:
        chosen.append(_pick(needing, rng))
    return [_make_event(layout, candidate_id, program, r.query, r, v) for r, v in chosen]


@dataclass(frozen=True, eq=False)
class EventTable:
    """An ordered, frozen set of teacher events with unique canonical ids, plus how it was built.

    ``layout`` is a :class:`Layout` (every candidate has that shape) or a :class:`LayoutMap` (each
    candidate has its own); either way the WHOLE layout document is part of the table's hash, and every
    event is validated against its own candidate's layout."""

    layout: object
    events: tuple
    universe: str
    config: dict
    unsupported: tuple
    sha256: str = ""

    def __post_init__(self):
        require(isinstance(self.layout, (Layout, LayoutMap)),
                "an event table needs a Layout or a LayoutMap")
        events = tuple(self.events)
        for event in events:
            canonical = event_id(event.candidate_id, event.program, event.query)
            require(event.event_id == canonical,
                    f"event id {event.event_id!r} is not the canonical id {canonical!r} of its "
                    "candidate, program and query")
            own = self.layout.layout_for(event.candidate_id)
            own.check_program(event.program)
            own.check_query(event.query)
        ids = [event.event_id for event in events]
        require(len(set(ids)) == len(ids), "an event table holds a duplicate event id")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "config", copy.deepcopy(self.config))
        object.__setattr__(self, "unsupported", tuple(copy.deepcopy(list(self.unsupported))))
        digest = fb.digest(self._body())
        if self.sha256:
            require(self.sha256 == digest, "the event table does not match its recorded digest")
        object.__setattr__(self, "sha256", digest)

    def _body(self):
        return {"schema": EVENT_TABLE_SCHEMA, "layout": self.layout.document(),
                "universe": self.universe, "config": self.config,
                "events": [event.document() for event in self.events],
                "unsupported": list(self.unsupported)}

    def document(self):
        return {**self._body(), "event_table_sha256": self.sha256}

    def verify(self):
        require(fb.digest(self._body()) == self.sha256,
                "the event table no longer matches its digest")
        return True

    def layout_for(self, candidate_id):
        """The layout of one candidate's sequence (every event is checked against it)."""
        return self.layout.layout_for(candidate_id)

    def __len__(self):
        return len(self.events)

    def ids(self):
        return tuple(event.event_id for event in self.events)

    def candidate_ids(self):
        return tuple(event.candidate_id for event in self.events)

    def codes(self):
        out = np.array([event.answer_code for event in self.events], dtype=np.int64)
        out.setflags(write=False)
        return out

    def is_equality(self):
        out = np.array([event.output_type == OUTPUT_EQUALITY for event in self.events])
        out.setflags(write=False)
        return out

    def unique_sequences(self):
        return len(set(self.candidate_ids()))


def parse_event_document(document, layout):
    """Rebuild an :class:`EventTable` from its document, recomputing every derived field from the
    program, query and (for the answer) the stored code, and refusing any disagreement. Needs the
    sequences only to recompute flags, which are therefore taken as stored and re-derived by callers
    that hold sequences (``verify_events_against``). Each event is parsed against ITS candidate's
    layout and must carry the canonical id (an alias is refused before anything is built)."""
    require(isinstance(document, dict) and document.get("schema") == EVENT_TABLE_SCHEMA,
            f"not a {EVENT_TABLE_SCHEMA} document")
    require(document["layout"] == layout.document(), "the event table's layout differs")
    events = []
    for item in document["events"]:
        own = layout.layout_for(item["candidate"])
        program = parse_program(item["program"], own)
        query = parse_query(item["query"], own)
        require(item["id"] == event_id(item["candidate"], program, query),
                f"event id {item['id']!r} is not the canonical id of its candidate, program and "
                "query")
        resolution = resolve(own, program, query)
        require(item["resolved"] == resolution.document() and item["stratum"] == classify(resolution)
                and item["template"] == template_id(program, query)
                and item["dependency"] == dependency_tag(own, program, query)
                and item["held_out"] == is_chained_swap_pair(program)
                and item["output"] == output_type(query),
                f"event {item['id']!r} disagrees with its own program and query")
        events.append(TeacherEvent(
            event_id=item["id"], candidate_id=item["candidate"], program=program, query=query,
            answer=item["answer"], answer_code=item["code"], output_type=item["output"],
            template_id=item["template"], stratum=item["stratum"], dependency=item["dependency"],
            held_out=item["held_out"], resolution=resolution, flags=item["flags"]))
    return EventTable(layout=layout, events=tuple(events), universe=document["universe"],
                      config=document["config"], unsupported=tuple(document["unsupported"]),
                      sha256=document["event_table_sha256"])


def verify_events_against(table, rows):
    """Recompute every answer (and flags) from the ORIGINAL sequences with the teacher. ``rows`` are
    ``(candidate_id, heavy, light)``. Refuses an event whose stored answer is not the teacher's, and a
    sequence whose chain lengths are not its candidate's layout."""
    sequences = {cid: (heavy, light) for cid, heavy, light in rows}
    for event in table.events:
        require(event.candidate_id in sequences,
                f"event {event.event_id!r} names a candidate outside the supplied rows")
        heavy, light = sequences[event.candidate_id]
        layout = table.layout_for(event.candidate_id)
        require(len(heavy) == layout.heavy and len(light) == layout.light,
                f"candidate {event.candidate_id!r} has H{len(heavy)}/L{len(light)} but its layout is "
                f"{layout.label}")
        truth = answer(layout, heavy, light, event.program, event.query)
        require(truth == event.answer and event.answer_code == answer_code(
            layout, truth, event.query),
                f"event {event.event_id!r}: the stored answer is not the teacher's")
        if len(event.program.ops) == 2:
            require(event.flags == counterfactual_answer_changes(
                layout, heavy, light, event.program, event.query),
                    f"event {event.event_id!r}: the stored counterfactual flags are wrong")
    return True


def _bind_layouts(layout, rows):
    """Validate ``layout`` (a Layout or a LayoutMap) against every ``(candidate_id, heavy, light)`` row
    and return it. The minimum-coordinate requirement is checked first, per layout shape."""
    require(isinstance(layout, (Layout, LayoutMap)), "generate_events needs a Layout or a LayoutMap")
    for heavy, light in layout.classes():
        require(heavy + light >= MIN_COORDINATES,
                f"event generation needs at least {MIN_COORDINATES} coordinates per layout, got "
                f"{heavy + light} (H{heavy}/L{light}): the SET/SWAP task cannot be defined over a "
                "shape this small; remove those candidates from the cohort that needs the task")
    return layout


def _draw_instance(universe, layout, candidate_id, heavy, light, counter, decks, rng, taken, notes):
    """One ACCEPTED instance, by bounded deterministic replacement.

    A drawn instance is rejected when it is incomplete (no matched true/false primary pair) or repeats
    an event id already taken. The class is unchanged but the next attempt must not repeat the draw: what
    the rejected attempt consumed from every deck is owed to a later cycle (:meth:`_Deck.defer`), so the
    retry sees other coordinates and symbols now and the rejected ones come back later. Handing them back
    at once repeated an identical program on every one of the attempts whenever a pending queue pinned
    it. Coverage of the accepted events is checked afterwards, not assumed here. Returns
    ``(events_or_None, rejected_attempts, {reason: count})``."""
    klass = TRAIN_CLASSES[counter % len(TRAIN_CLASSES)] if universe == UNIVERSE_TRAIN \
        else CLASS_SWAP_CHAINED
    needed = EVENTS_PER_INSTANCE if universe == UNIVERSE_TRAIN else HELD_OUT_MIN_EVENTS
    reasons = {}
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        for deck in decks.values():
            deck.begin()
        attempt_notes = []
        if universe == UNIVERSE_TRAIN:
            program = _draw_train_program(klass, layout, decks, rng)
            goal = STRATUM_EQ_PRIMARY
            if counter % EVENTS_PER_INSTANCE == EVENTS_PER_INSTANCE - 1 and any(
                    isinstance(op, SetOp) for op in program.ops):
                goal = (STRATUM_EQ_LITERAL_LITERAL if klass == CLASS_SET_SET
                        else STRATUM_EQ_ORIGINAL_LITERAL)
            made = _train_instance(layout, candidate_id, heavy, light, program, goal, rng,
                                   attempt_notes)
        else:
            program = _draw_chain_program(layout, decks, rng)
            made = _held_out_instance(layout, candidate_id, heavy, light, program, rng,
                                      attempt_notes)
        if len(made) < needed:
            reason = (attempt_notes[-1]["reason"] if attempt_notes
                      else "the instance is incomplete")
        elif any(event.event_id in taken for event in made):
            reason = "the instance repeats an event id that is already taken"
        else:
            notes.extend(attempt_notes)
            return made, attempt - 1, reasons
        reasons[reason] = reasons.get(reason, 0) + 1
        for deck in decks.values():
            deck.defer()
    return None, GENERATION_ATTEMPTS, reasons


def generate_events(layout, rows, *, universe, instances_per_sequence, seed, exclude_ids=(),
                    strict=True):
    """Deterministic, SOURCE-ONLY event generation (no outcome of any kind is read).

    ``rows`` are ``(candidate_id, heavy, light)``. ``layout`` is a :class:`Layout` (every sequence must
    have exactly its chain lengths and alphabet: the original fixed-length contract) or a
    :class:`LayoutMap` naming exactly these candidates (each sequence has its own shape). Generation is
    per layout shape: each shape has its own seeded stream, shuffled decks and class schedule, so one
    shape's events do not depend on another's.

    Each sequence receives ``instances_per_sequence`` COMPLETE instances of four events: an EQ with
    answer true and an EQ with answer false that share one full canonical program+query template (an
    infeasible secondary-stratum request falls back to the primary stratum and says so under
    ``unsupported``) and two RES queries. ``train`` instances cycle through :data:`TRAIN_CLASSES` (the
    held-out chained two-swap class and the degenerate same-pair class are never drawn) and every fourth
    instance asks for a secondary EQ stratum when the program has a SET. ``held_out`` instances are
    chained two-swap programs whose EQ events need BOTH swaps (plus one ignores-the-program control and
    one RES needing both swaps).

    The literal-literal request is the one that cannot be a pair (at most one literal-literal query exists
    in a program of at most two SETs and distinct EQ coordinates): that instance holds a matched primary
    pair, the one-sided literal-literal control and ONE RES query, still four unique answers. The actual
    mix is reported under ``config['instance_shapes']`` (``"2RES+2EQ"``, ``"1RES+3EQ"``) and the control
    under :func:`balance_report`; every arm trains on the same table.

    An instance that cannot be completed is replaced by a deterministic bounded redraw of the same
    class (the count and reasons are recorded under ``config['replacement']``); the retry is a DIFFERENT
    draw, because what a rejected attempt consumed from the shuffled decks is owed to a later cycle rather
    than handed back. If a sequence cannot supply any complete instance, ``strict`` (the default, used
    for training) REFUSES with the reason rather than shorten the roster; ``strict=False`` (auxiliary
    panels) skips the instance and records it under ``unsupported``. Whether the ACCEPTED events cover
    every coordinate, symbol and class is not guaranteed by the decks: :func:`coverage_report` checks it
    per layout shape and training refuses when it is incomplete.
    """
    require(universe in UNIVERSES, f"universe must be one of {UNIVERSES}, got {universe!r}")
    layout = _bind_layouts(layout, rows)
    require(_is_int(instances_per_sequence) and instances_per_sequence >= 1,
            "instances_per_sequence must be a positive integer")
    rows = sorted((str(cid), heavy, light) for cid, heavy, light in rows)
    require(rows, "event generation needs at least one sequence")
    require(len({cid for cid, _, _ in rows}) == len(rows), "a candidate id is repeated")
    if isinstance(layout, LayoutMap):
        require(set(layout.candidate_ids()) == {cid for cid, _, _ in rows},
                "the layout map must name exactly the generated candidates")
    alphabet = layout.alphabet
    for cid, heavy, light in rows:
        own = layout.layout_for(cid)
        require(len(heavy) == own.heavy and len(light) == own.light
                and set(heavy + light) <= set(alphabet),
                f"candidate {cid!r} does not fit the layout {own.document()}: every sequence is "
                "validated against its own layout and the shared alphabet")
    by_shape = {}
    for row in rows:
        by_shape.setdefault((len(row[1]), len(row[2])), []).append(row)
    taken, per_candidate, notes = set(exclude_ids), {}, []
    log = {"instances": 0, "replaced_instances": 0, "rejected_attempts": 0, "rejection_reasons": {}}
    instance_shapes = {}
    for shape in sorted(by_shape):
        shape_layout = Layout(shape[0], shape[1], alphabet)
        rng = np.random.default_rng([int(seed), STREAM_UNIVERSE[universe], shape[0], shape[1]])
        decks = {"set_coord": _Deck(shape_layout.coords(), rng),
                 "swap_coord": _Deck(shape_layout.coords(), rng),
                 "chain_coord": _Deck(shape_layout.coords(), rng),
                 "set_symbol": _Deck(tuple(alphabet), rng)}
        counter = 0
        for candidate_id, heavy, light in by_shape[shape]:
            made_events = per_candidate.setdefault(candidate_id, [])
            for _ in range(instances_per_sequence):
                made, rejected, reasons = _draw_instance(
                    universe, shape_layout, candidate_id, heavy, light, counter, decks, rng, taken,
                    notes)
                counter += 1
                log["instances"] += 1
                log["rejected_attempts"] += rejected
                log["replaced_instances"] += 1 if (made is not None and rejected) else 0
                for reason, count in reasons.items():
                    log["rejection_reasons"][reason] = log["rejection_reasons"].get(reason, 0) + count
                if made is None:
                    summary = "; ".join(f"{count}x {reason}" for reason, count in sorted(reasons.items()))
                    require(not strict,
                            f"candidate {candidate_id!r} ({shape_layout.label}: heavy {heavy!r}, light "
                            f"{light!r}) cannot supply a complete {universe} instance after "
                            f"{GENERATION_ATTEMPTS} bounded replacement attempts ({summary}). The "
                            "declared roster is never shortened: remove the candidate from the "
                            "cohort that needs the task")
                    notes.append({"candidate": candidate_id, "layout": shape_layout.label,
                                  "universe": universe, "reason": "no complete instance could be "
                                  "drawn; the instance is skipped", "rejections": dict(reasons)})
                    continue
                kind = (f"{sum(e.output_type == OUTPUT_RESIDUE for e in made)}RES+"
                        f"{sum(e.output_type == OUTPUT_EQUALITY for e in made)}EQ")
                instance_shapes[kind] = instance_shapes.get(kind, 0) + 1
                for event in made:
                    taken.add(event.event_id)
                    made_events.append(_with_flags(shape_layout, heavy, light, event))
    events = [event for cid, _, _ in rows for event in per_candidate.get(cid, [])]
    config = {"universe": universe, "instances_per_sequence": int(instances_per_sequence),
              "events_per_instance": EVENTS_PER_INSTANCE, "seed": int(seed),
              "sequences": len(rows),
              "nominal_events": len(rows) * int(instances_per_sequence) * EVENTS_PER_INSTANCE,
              "layout_shapes": {Layout(h, l, alphabet).label: len(members)
                                for (h, l), members in sorted(by_shape.items())},
              "instance_shapes": dict(sorted(instance_shapes.items())),
              "replacement": {**log, "attempt_bound": GENERATION_ATTEMPTS,
                              "rule": ("an incomplete or repeated instance is redrawn in the same "
                                       "class; what the rejected attempt consumed from the shuffled "
                                       "decks is owed to a later cycle (never re-drawn identically at "
                                       "once); coverage is checked on the accepted events; nothing is "
                                       "shortened")},
              "rule": ("source-only: sequences and the teacher only; no outcome of any kind is read; "
                       "the chained two-swap class and same-pair two-swap programs are never drawn "
                       "in the train universe")}
    return EventTable(layout=layout, events=tuple(events), universe=universe, config=config,
                      unsupported=tuple(notes))


def _class_coverage(layout, universe, events):
    set_coords, set_symbols, swap_coords, classes, lengths = set(), set(), set(), set(), set()
    for event in events:
        lengths.add(len(event.program.ops))
        classes.add(program_class(event.program))
        for op in event.program.ops:
            if isinstance(op, SetOp):
                set_coords.add(op.coord)
                set_symbols.add(op.symbol)
            else:
                swap_coords.update((op.a, op.b))
    eq_values = {event.answer for event in events if event.output_type == OUTPUT_EQUALITY
                 and event.stratum == STRATUM_EQ_PRIMARY}
    report = {"RES_and_EQ_queries": {event.output_type for event in events}
              == {OUTPUT_RESIDUE, OUTPUT_EQUALITY},
              "primary_EQ_true_and_false": eq_values == {True, False}}
    if universe == UNIVERSE_TRAIN:
        report.update({
            "every_coordinate_in_a_SET": set_coords == set(layout.coords()),
            "every_symbol_in_a_SET": set_symbols == set(layout.alphabet),
            "every_coordinate_in_a_SWAP": swap_coords == set(layout.coords()),
            "zero_one_and_two_operation_programs": lengths == {0, 1, 2},
            "every_training_class": classes >= set(TRAIN_CLASSES),
            "no_held_out_or_degenerate_class": not (classes & {CLASS_SWAP_CHAINED,
                                                               CLASS_SWAP_SAME})})
    else:
        report["only_chained_swap_programs"] = classes == {CLASS_SWAP_CHAINED}
    report["complete"] = all(report.values())
    return report


def coverage_report(table):
    """Which training coverage requirements hold, PER LAYOUT SHAPE (``by_layout``) and overall.

    Every requirement is evaluated against the shape's own coordinates, so a shape with too few
    sequences or instances to cover them reports ``False`` for exactly those requirements. The
    top-level flags are the conjunction over shapes and ``complete`` is true only when every flag of
    every shape holds."""
    layouts, grouped = table.layout, {shape: [] for shape in table.layout.classes()}
    for event in table.events:
        own = table.layout_for(event.candidate_id)
        grouped[(own.heavy, own.light)].append(event)
    by_layout = {}
    for (heavy, light), events in sorted(grouped.items()):
        shape_layout = Layout(heavy, light, layouts.alphabet)
        by_layout[shape_layout.label] = _class_coverage(shape_layout, table.universe, events)
    names = sorted({name for entry in by_layout.values() for name in entry if name != "complete"})
    report = {name: all(entry.get(name, False) for entry in by_layout.values()) for name in names}
    report["by_layout"] = by_layout
    report["complete"] = all(report[name] for name in names)
    return report


def incomplete_coverage(report):
    """``{layout label: [failed requirement, ...]}`` of a :func:`coverage_report` (empty if complete)."""
    return {label: sorted(name for name, holds in entry.items() if name != "complete" and not holds)
            for label, entry in report["by_layout"].items() if not entry["complete"]}


def balance_report(table):
    """True/false balance of PRIMARY EQ answers per (candidate, full canonical program+query template)
    and per template, with support counts; infeasible matches are listed, never hidden.

    The primary scope is every ``eq_primary`` event in the train universe and, in the held-out
    universe, only the ``eq_primary`` events that need BOTH swaps. Held-out events that ignore the
    program are reported separately under ``program_ignoring_and_other_controls`` (they are controls,
    not primary evidence), and every EQ stratum is counted under ``eq_by_stratum``.

    ``events_by_output`` is the ACTUAL number of RES and EQ events, and ``secondary_eq_balance`` says for
    every non-primary EQ stratum whether both answers occur. The literal-literal stratum can never be a
    true/false pair (a program of at most two SETs reads at most one literal-literal pair), so it is
    reported ``one_sided_unsupported`` when only one answer occurs: a valid control, not a balanced one."""
    held = table.universe == UNIVERSE_HELD_OUT
    per_unit, per_template, template_candidates, strata, controls = {}, {}, {}, {}, {}
    by_output = {OUTPUT_RESIDUE: 0, OUTPUT_EQUALITY: 0}
    for event in table.events:
        by_output[event.output_type] = by_output.get(event.output_type, 0) + 1
        if event.output_type != OUTPUT_EQUALITY:
            continue
        side = 0 if event.answer else 1
        strata.setdefault(event.stratum, [0, 0])[side] += 1
        if event.stratum != STRATUM_EQ_PRIMARY:
            continue
        if held and event.dependency != DEP_BOTH:
            controls.setdefault(event.dependency, [0, 0])[side] += 1
            continue
        per_unit.setdefault((event.candidate_id, event.template_id), [0, 0])[side] += 1
        per_template.setdefault(event.template_id, [0, 0])[side] += 1
        template_candidates.setdefault(event.template_id, set()).add(event.candidate_id)
    unbalanced = sorted(f"{cid}|{tpl}" for (cid, tpl), (t, f) in per_unit.items() if t != f)
    one_sided = sorted(tpl for tpl, (t, f) in per_template.items() if t == 0 or f == 0)
    infeasible = [note for note in table.unsupported if "reason" in note
                  and ("matched" in note["reason"] or "shares a full" in note["reason"]
                       or "no complete instance" in note["reason"])]
    return {"primary_eq_units": len(per_unit),
            "units_with_equal_true_and_false": len(per_unit) - len(unbalanced),
            "unbalanced_units": unbalanced[:20], "unbalanced_unit_count": len(unbalanced),
            "unit": "(candidate, full canonical program+query template)",
            "primary_scope": ("held-out: eq_primary events that need BOTH swaps; train: every "
                              "eq_primary event"),
            "templates": {tpl: {"true": t, "false": f, "candidates": len(template_candidates[tpl])}
                          for tpl, (t, f) in sorted(per_template.items())},
            "templates_with_only_one_answer": one_sided,
            "eq_by_stratum": {name: {"true": t, "false": f} for name, (t, f) in sorted(strata.items())},
            "events_by_output": {name: int(count) for name, count in sorted(by_output.items())},
            "secondary_eq_balance": {
                name: {"true": t, "false": f,
                       "balance": ("both_answers_present" if t and f else "one_sided_unsupported")}
                for name, (t, f) in sorted(strata.items()) if name != STRATUM_EQ_PRIMARY},
            "program_ignoring_and_other_controls": {
                name: {"true": t, "false": f, "events": t + f}
                for name, (t, f) in sorted(controls.items())},
            "infeasible_matches": {"count": len(infeasible), "examples": infeasible[:5]},
            "unsupported_notes": len(table.unsupported)}


# ---------------------------------------------------------------------------
# the shuffled-target control (Q)
# ---------------------------------------------------------------------------

def _unsupported_status(n, distinct, counts):
    """Why a stratum cannot supply a corruption control, or ``None`` when it can."""
    if distinct.shape[0] < 2:
        return "unsupported_constant"
    if n < MIN_STRATUM:
        return "unsupported_small"
    if int(counts.max()) < 2:
        return "unsupported_singleton_classes"
    return None


def _split_classes(values, candidate):
    """The original classes that occur at least twice and are sent to at least two different targets."""
    split = []
    for value in np.unique(values):
        mask = values == value
        if int(mask.sum()) >= 2 and np.unique(candidate[mask]).shape[0] >= 2:
            split.append(int(value))
    return split


def shuffle_targets(table, seed):
    """Shuffle answer codes ONCE within (output type, template, dependency stratum) and report the
    values that ACTUALLY changed per stratum (not row ids).

    The control must corrupt the RELATION between question and answer, not rename classes. A stratum
    is shuffled only if it has at least ``MIN_STRATUM`` events, at least two distinct values and at
    least one value that occurs twice or more; a constant, tiny or all-singleton-class stratum is
    reported ``unsupported_*`` and left unshuffled, never silently counted as corrupted (four events
    with four different labels can only be permuted into a learnable renaming). An accepted
    permutation keeps the stratum's histogram, changes at least one and leaves at least one value, AND
    sends at least one repeated original class to more than one new target, so no deterministic
    original-class -> shuffled-class mapping exists (a fixed inversion of a Boolean stratum, or any
    class permutation, is therefore never accepted). Permutations are redrawn (deterministic
    source-only rng; the attempt number is recorded) until the changed share is within [0.5, 1.5] x the
    share a random shuffle changes in expectation; the best usable one is kept otherwise. No held-out
    label is read; the table is the source-only training table.
    """
    codes = table.codes().copy()
    strata = {}
    for position, event in enumerate(table.events):
        strata.setdefault((event.output_type, event.template_id, event.stratum), []).append(position)
    shuffled, entries = codes.copy(), []
    for key in sorted(strata):
        members = np.array(strata[key], dtype=np.int64)
        values = codes[members]
        distinct, counts = np.unique(values, return_counts=True)
        n = int(members.shape[0])
        entry = {"output": key[0], "template": key[1], "stratum": key[2], "events": n,
                 "distinct_values": int(distinct.shape[0]),
                 "value_counts": {str(int(v)): int(c) for v, c in zip(distinct, counts)}}
        status = _unsupported_status(n, distinct, counts)
        if status is not None:
            entry.update(status=status, changed_values=0, attempts=0, changed_fraction=0.0)
            entries.append(entry)
            continue
        expected = float(1.0 - np.sum((counts / n) ** 2))
        rng = np.random.default_rng([int(seed), STREAM_Q, stable_int(*key)])
        best, accepted = None, False
        for attempt in range(1, SHUFFLE_ATTEMPTS + 1):
            candidate = values[rng.permutation(n)]
            changed = int(np.sum(candidate != values))
            share = changed / n
            split = _split_classes(values, candidate) if 1 <= changed <= n - 1 else []
            usable = bool(split)
            within = 0.5 * expected <= share <= min(1.0 - 1.0 / n, 1.5 * expected)
            score = abs(share - expected)
            if usable and (best is None or score < best[0]):
                best = (score, candidate, changed, attempt, split)
            if usable and within:
                accepted = True
                best = (score, candidate, changed, attempt, split)
                break
        if best is None:
            entry.update(status="unsupported_degenerate", changed_values=0,
                         attempts=SHUFFLE_ATTEMPTS, changed_fraction=0.0)
        else:
            shuffled[members] = best[1]
            entry.update(status="shuffled" if accepted else "shuffled_best_effort",
                         changed_values=best[2], attempts=best[3],
                         changed_fraction=best[2] / n, expected_changed_fraction=expected,
                         split_original_classes=len(best[4]), deterministic_class_mapping=False)
        entries.append(entry)
    changed_total = int(np.sum(shuffled != codes))
    by_output = {}
    for entry in entries:
        slot = by_output.setdefault(entry["output"], {"events": 0, "changed_values": 0,
                                                       "unshuffled_events": 0})
        slot["events"] += entry["events"]
        slot["changed_values"] += entry["changed_values"]
        if entry["status"].startswith("unsupported"):
            slot["unshuffled_events"] += entry["events"]
    report = {"seed": int(seed), "events": int(codes.shape[0]), "changed_values": changed_total,
              "changed_fraction": changed_total / max(1, int(codes.shape[0])),
              "by_output_type": by_output,
              "unsupported_strata": [entry for entry in entries
                                     if entry["status"].startswith("unsupported")],
              "strata": entries,
              "rule": ("labels shuffled ONCE within (output type, template, dependency stratum) from "
                       "the source-only event table, then frozen in the tape; a stratum that is "
                       "constant, too small or has no repeated class is reported unsupported and NOT "
                       "shuffled; an accepted shuffle keeps the histogram, changes and keeps values, "
                       "and splits at least one repeated class (no deterministic class renaming); "
                       "the changed counts are values that differ, not row ids that moved")}
    out = shuffled.astype(np.int64)
    out.setflags(write=False)
    return out, report


# ---------------------------------------------------------------------------
# the frozen per-seed tape
# ---------------------------------------------------------------------------

def stream_batches(n_items, steps, batch, seed, stream):
    """``(steps, batch)`` positions from a seeded stream of epoch permutations. The first ``k`` rows do
    not depend on ``steps``: a longer tape extends a shorter one (a batch may span an epoch boundary)."""
    require(_is_int(n_items) and n_items >= 1 and _is_int(steps) and steps >= 1
            and _is_int(batch) and batch >= 1,
            "stream_batches needs positive integer sizes")
    rng = np.random.default_rng([int(seed), int(stream)])
    total, pieces, have = int(steps) * int(batch), [], 0
    while have < total:
        pieces.append(rng.permutation(int(n_items)))
        have += int(n_items)
    return np.concatenate(pieces)[:total].astype(np.int64).reshape(int(steps), int(batch))


_TAPE_ARRAYS = ("functional_labels", "functional_batches", "aux_batches", "q_targets")


@dataclass(frozen=True, eq=False)
class SeedTape:
    """Everything one source seed's training replays, frozen and hashed: the functional batch
    membership and labels, the auxiliary batch membership over the event table, the shuffled-control
    targets (with their per-stratum report), and the mask binding (the real token length of every row
    in use, from which every padded mask is derived; it also carries the layout-map digest). No fresh
    sample or corruption happens on replay.

    The arrays are copied and marked read-only at construction, but a read-only flag can be re-enabled:
    :meth:`verify` re-derives the digest from the arrays AS THEY ARE NOW, and :meth:`snapshot` returns
    an independent verified copy for a consumer that must not see a later mutation."""

    seed: int
    steps: int
    functional_steps: int
    batch: int
    aux_batch: int
    functional_ids: tuple
    functional_labels: object
    functional_batches: object
    aux_batches: object
    q_targets: object
    q_report: dict
    event_table_sha256: str
    mask_binding: dict
    sha256: str = ""

    def __post_init__(self):
        for name in _TAPE_ARRAYS:
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "functional_ids", tuple(self.functional_ids))
        object.__setattr__(self, "q_report", copy.deepcopy(self.q_report))
        object.__setattr__(self, "mask_binding", copy.deepcopy(self.mask_binding))
        require(self.functional_batches.shape == (self.functional_steps, self.batch)
                and self.aux_batches.shape == (self.steps, self.aux_batch)
                and self.functional_steps >= self.steps,
                "the tape's batch arrays do not have the declared shapes")
        require(self.functional_labels.shape == (len(self.functional_ids),),
                "the tape needs one functional label per functional candidate")
        digest = fb.digest(self._body())
        if self.sha256:
            require(self.sha256 == digest, "the tape does not match its recorded digest")
        object.__setattr__(self, "sha256", digest)

    def _body(self):
        return {"schema": SEED_TAPE_SCHEMA, "seed": int(self.seed), "steps": int(self.steps),
                "functional_steps": int(self.functional_steps), "batch": int(self.batch),
                "aux_batch": int(self.aux_batch), "functional_ids": list(self.functional_ids),
                "functional_labels": self.functional_labels.tolist(),
                "functional_batches": self.functional_batches.tolist(),
                "aux_batches": self.aux_batches.tolist(), "q_targets": self.q_targets.tolist(),
                "q_report": self.q_report, "event_table_sha256": self.event_table_sha256,
                "mask_binding": self.mask_binding}

    def document(self):
        return {**self._body(), "tape_sha256": self.sha256}

    def verify(self):
        """Re-derive the digest from the arrays AS THEY ARE NOW (a re-enabled write is caught)."""
        require(fb.digest(self._body()) == self.sha256, "the tape no longer matches its digest")
        return True

    def snapshot(self):
        """An independent copy that is verified against the recorded digest while it is made: it
        raises if the arrays changed since the tape was built, and the copy cannot be changed through
        this tape's arrays afterwards."""
        return SeedTape(**{field: getattr(self, field) for field in (
            "seed", "steps", "functional_steps", "batch", "aux_batch", "functional_ids",
            "functional_labels", "functional_batches", "aux_batches", "q_targets", "q_report",
            "event_table_sha256", "mask_binding", "sha256")})

    @classmethod
    def from_document(cls, document):
        keys = ("schema", "seed", "steps", "functional_steps", "batch", "aux_batch",
                "functional_ids", "functional_labels", "functional_batches", "aux_batches",
                "q_targets", "q_report", "event_table_sha256", "mask_binding", "tape_sha256")
        fm._keys(document, keys, label="seed tape")
        require(document["schema"] == SEED_TAPE_SCHEMA, f"not a {SEED_TAPE_SCHEMA} document")
        return cls(seed=document["seed"], steps=document["steps"],
                   functional_steps=document["functional_steps"], batch=document["batch"],
                   aux_batch=document["aux_batch"], functional_ids=tuple(document["functional_ids"]),
                   functional_labels=np.array(document["functional_labels"], dtype=np.float64),
                   functional_batches=np.array(document["functional_batches"], dtype=np.int64),
                   aux_batches=np.array(document["aux_batches"], dtype=np.int64),
                   q_targets=np.array(document["q_targets"], dtype=np.int64),
                   q_report=document["q_report"], event_table_sha256=document["event_table_sha256"],
                   mask_binding=document["mask_binding"], sha256=document["tape_sha256"])


def build_seed_tape(*, seed, steps, functional_steps, batch, aux_batch, functional_ids,
                    functional_labels, table, mask_binding):
    """Freeze one seed's tape. The functional stream and the auxiliary stream are independent seeded
    permutation streams; the shuffled targets are drawn once from the SOURCE-ONLY event table."""
    functional_ids = tuple(functional_ids)
    labels = np.array(functional_labels, dtype=np.float64)
    require(labels.shape == (len(functional_ids),) and bool(np.isfinite(labels).all()),
            "functional labels must be one finite number per functional candidate")
    shuffled, report = shuffle_targets(table, seed)
    return SeedTape(
        seed=int(seed), steps=int(steps), functional_steps=int(functional_steps), batch=int(batch),
        aux_batch=int(aux_batch), functional_ids=functional_ids, functional_labels=labels,
        functional_batches=stream_batches(len(functional_ids), functional_steps, batch, seed,
                                          STREAM_FUNCTIONAL),
        aux_batches=stream_batches(len(table), steps, aux_batch, seed, STREAM_AUX),
        q_targets=shuffled, q_report=report, event_table_sha256=table.sha256,
        mask_binding=mask_binding)


# ---------------------------------------------------------------------------
# cohort manifest and the data boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssayDeclaration:
    """One declared target/assay/units/scale. v1 supports exactly one source and one target."""

    target: str
    assay: str
    units: str
    scale: str
    useful_threshold: object = None
    description: str = ""

    def document(self):
        out = {"target": self.target, "assay": self.assay, "units": self.units,
               "scale": self.scale, "description": self.description}
        if self.useful_threshold is not None:
            out["useful_threshold"] = self.useful_threshold
        return out


def _parse_assay(document, *, label, with_threshold):
    required = ("target", "assay", "units", "scale") + (("useful_threshold",) if with_threshold
                                                        else ())
    fm._keys(document, required, ("description",), label=label)
    threshold = fb._real(document["useful_threshold"], f"{label}.useful_threshold") \
        if with_threshold else None
    return AssayDeclaration(
        target=fb._text(document["target"], f"{label}.target"),
        assay=fb._text(document["assay"], f"{label}.assay"),
        units=fb._text(document["units"], f"{label}.units"),
        scale=fb._text(document["scale"], f"{label}.scale"), useful_threshold=threshold,
        description=str(document.get("description", "")))


@dataclass(frozen=True)
class CohortManifest:
    alphabet: str
    source: AssayDeclaration
    target: AssayDeclaration
    roles: dict
    aux_defaulted: bool
    grouping_rationale: str
    notes: str = ""

    def document(self):
        cohorts = {role: list(self.roles[role]) for role in ROLES
                   if not (role == "source_aux" and self.aux_defaulted)}
        return {"schema": COHORT_SCHEMA, "alphabet": self.alphabet,
                "source": self.source.document(), "target": self.target.document(),
                "cohorts": cohorts, "grouping_rationale": self.grouping_rationale,
                "notes": self.notes}

    def resolved_document(self):
        """With ``source_aux`` spelled out, so a default and an explicit copy hash identically."""
        out = self.document()
        out["cohorts"] = {role: list(self.roles[role]) for role in ROLES}
        return out

    def sha256(self):
        return fb.digest(self.resolved_document())


def parse_cohort_manifest(document):
    """Strict parse. Unknown or missing keys, repeated ids and empty roles are refused. ``source_aux``
    may be omitted or null: it then defaults to ``source_train``."""
    document = copy.deepcopy(document)
    fm._keys(document, ("schema", "alphabet", "source", "target", "cohorts",
                        "grouping_rationale"), ("notes",), label="cohort manifest")
    require(document["schema"] == COHORT_SCHEMA, f"not a {COHORT_SCHEMA} manifest")
    alphabet = document["alphabet"]
    require(isinstance(alphabet, str) and alphabet and alphabet == alphabet.upper()
            and alphabet.isalpha() and len(set(alphabet)) == len(alphabet),
            "manifest alphabet must be distinct upper-case letters")
    cohorts = fm._keys(document["cohorts"], ("source_train", "source_dev", "target_train",
                                             "target_eval"), ("source_aux",), label="cohorts")
    roles = {}
    for role in ROLES:
        members = cohorts.get(role)
        if role == "source_aux" and members is None:
            continue
        require(isinstance(members, list) and members
                and all(isinstance(item, str) and item and item == item.strip() for item in members),
                f"cohorts.{role} must be a non-empty list of stripped candidate ids")
        require(len(set(members)) == len(members), f"cohorts.{role} repeats a candidate id")
        roles[role] = tuple(sorted(members))
    aux_defaulted = "source_aux" not in roles
    if aux_defaulted:
        roles["source_aux"] = roles["source_train"]
    return CohortManifest(
        alphabet=alphabet,
        source=_parse_assay(document["source"], label="manifest.source", with_threshold=False),
        target=_parse_assay(document["target"], label="manifest.target", with_threshold=True),
        roles=roles, aux_defaulted=aux_defaulted,
        grouping_rationale=fb._text(document["grouping_rationale"], "manifest.grouping_rationale"),
        notes=str(document.get("notes", "")))


def read_cohort_manifest(path):
    return parse_cohort_manifest(fb.read_json(path))


@dataclass(frozen=True, eq=False)
class Cohorts:
    """Validated cohorts over ONE token cache: sorted ids per role, the cache row of every id and the
    auxiliary layout source. ``layouts`` is a :class:`Layout` when every auxiliary row has one shape
    and a :class:`LayoutMap` otherwise; ``layout`` is the single Layout in the first case and ``None``
    in the second. ``summary`` is what a report records."""

    manifest: CohortManifest
    roles: dict
    position: dict
    layout: object
    target_identity_length: int
    cache_sha256: str
    cohort_sha256: str
    layouts: object = None

    def rows(self, cache, role):
        """``(candidate_id, heavy, light)`` of a role, from the verified cache (teacher side only)."""
        heavy, light = cache.heavy, cache.light          # each access builds a tuple: take them once
        return [(cid, heavy[self.position[cid]], light[self.position[cid]])
                for cid in self.roles[role]]

    def summary(self, cache):
        groups = cache.groups
        layouts = self.layouts if self.layouts is not None else self.layout
        shapes = ({layouts.label: len(self.roles["source_aux"])} if isinstance(layouts, Layout)
                  else layouts.counts())
        return {"cohort_sha256": self.cohort_sha256, "cache_sha256": self.cache_sha256,
                "aux_defaulted_to_source_train": self.manifest.aux_defaulted,
                "layout": None if self.layout is None else self.layout.document(),
                "auxiliary_layouts": {"sha256": layouts_sha256(layouts), "shapes": shapes,
                                      "variable": not isinstance(layouts, Layout)},
                "target_identity_length": self.target_identity_length,
                "roles": {role: {"candidates": len(self.roles[role]),
                                 "groups": sorted({groups[self.position[cid]]
                                                   for cid in self.roles[role]})}
                          for role in ROLES}}


def validate_cohorts(cache, manifest):
    """Refuse everything the data boundary forbids, BEFORE any compute (see module docstring).

    ``source_train`` may be a subset of ``source_aux`` (its default); otherwise cohorts have disjoint
    ids, complete physical sequences AND groups. A PARTIAL overlap of ``source_train`` and
    ``source_aux`` is refused as ambiguous. Source aux may not include any source-dev or target row or
    group. Cache rows outside the declared union and declared ids missing from the cache are refused,
    and every role's rows must carry the cache split that role expects. The target roster must have
    ONE full-identity length (the matrix cannot join variable lengths; nothing is truncated). Source and
    auxiliary rows may have DIFFERENT chain lengths: each keeps its own layout (``Cohorts.layouts``) and
    only a row below the task's minimum coordinates is refused. H4/L4 and H5/L3 target rows can coexist
    because their full ``H|L`` identities have the same length.
    """
    require(isinstance(cache, fne.TokenCache), "validate_cohorts needs a TokenCache")
    require(isinstance(manifest, CohortManifest), "validate_cohorts needs a CohortManifest")
    cache.verify()
    all_ids, all_identities, all_groups, all_splits = (cache.ids, cache.identities, cache.groups,
                                                       cache.splits)
    all_heavy, all_light = cache.heavy, cache.light     # each property access builds a new tuple
    position = {cid: i for i, cid in enumerate(all_ids)}
    roles = {role: tuple(sorted(manifest.roles[role])) for role in ROLES}
    union = set().union(*(set(members) for members in roles.values()))
    for role, members in roles.items():
        missing = [cid for cid in members if cid not in position]
        require(not missing, f"cohort {role} names {len(missing)} candidate(s) the token cache does "
                             f"not hold, e.g. {missing[:3]}")
    extra = sorted(set(all_ids) - union)
    require(not extra, f"the token cache holds {len(extra)} row(s) outside the declared cohort "
                       f"union, e.g. {extra[:3]}; rows outside the union are refused, never "
                       "ignored")
    sets = {role: set(members) for role, members in roles.items()}
    train, aux = sets["source_train"], sets["source_aux"]
    overlap = train & aux
    require(not overlap or overlap == train,
            "source_train overlaps source_aux only partially: source_train may be a SUBSET of "
            "source_aux or disjoint from it")
    subset = bool(overlap)
    exclusive = ("source_dev", "target_train", "target_eval")
    for first, second in itertools.combinations(("source_train", "source_aux") + exclusive, 2):
        if {first, second} == {"source_train", "source_aux"} and subset:
            continue
        shared = sets[first] & sets[second]
        require(not shared, f"cohorts {first} and {second} share {len(shared)} candidate(s), e.g. "
                            f"{sorted(shared)[:3]}")
    identities = {role: {all_identities[position[cid]] for cid in members}
                  for role, members in roles.items()}
    groups = {role: {all_groups[position[cid]] for cid in members}
              for role, members in roles.items()}
    for first, second in itertools.combinations(("source_train", "source_aux") + exclusive, 2):
        if {first, second} == {"source_train", "source_aux"} and subset:
            continue
        same_sequence = identities[first] & identities[second]
        require(not same_sequence,
                f"cohorts {first} and {second} share {len(same_sequence)} complete physical "
                "sequence(s)")
        same_group = groups[first] & groups[second]
        require(not same_group, f"cohorts {first} and {second} share group(s) "
                                f"{sorted(same_group)[:3]}: whole groups belong to one cohort")
    for role, members in roles.items():
        bad = [cid for cid in members if all_splits[position[cid]] != EXPECTED_SPLIT[role]]
        require(not bad, f"cohort {role} rows must carry cache split {EXPECTED_SPLIT[role]!r}; "
                         f"{len(bad)} do not, e.g. {bad[:3]}")
    stray = sorted({residue for identity in all_identities for residue in identity
                    if residue != fne.DELIMITER and residue not in manifest.alphabet})
    require(not stray, f"the token cache holds residue(s) {stray} outside the manifest alphabet "
                       f"{manifest.alphabet!r}")
    bar = [cid for cid in roles["source_aux"] if "|" in cid]
    require(not bar, f"auxiliary candidate ids may not contain '|' (they build event ids): {bar[:3]}")
    layouts = layouts_for_rows(
        [(cid, all_heavy[position[cid]], all_light[position[cid]]) for cid in roles["source_aux"]],
        manifest.alphabet)
    target_lengths = {len(all_identities[position[cid]])
                      for role in ("target_train", "target_eval") for cid in roles[role]}
    require(len(target_lengths) == 1,
            f"the target roster has {len(target_lengths)} different full-identity lengths "
            f"{sorted(target_lengths)}: the frozen-feature matrix joins ONE fixed full-identity "
            "length, and nothing is truncated or padded to make rows fit; supply a compatible "
            "target roster")
    return Cohorts(manifest=manifest, roles=roles, position=position,
                   layout=layouts if isinstance(layouts, Layout) else None,
                   target_identity_length=int(next(iter(target_lengths))),
                   cache_sha256=cache.cache_sha256, cohort_sha256=manifest.sha256(),
                   layouts=layouts)


def read_role_outcomes(path, ids, *, label):
    """A ``candidate_id, utility`` CSV whose ids are EXACTLY ``ids``: finite numbers, no repeats, no
    missing or extra candidate."""
    outcomes = fb.read_outcomes(path, label=label)
    wanted = set(ids)
    missing, extra = sorted(wanted - set(outcomes)), sorted(set(outcomes) - wanted)
    require(not missing, f"{label} lacks {len(missing)} of the {len(wanted)} candidate(s), e.g. "
                         f"{missing[:3]}")
    require(not extra, f"{label} names {len(extra)} candidate(s) outside its cohort, e.g. "
                       f"{extra[:3]}")
    return outcomes
