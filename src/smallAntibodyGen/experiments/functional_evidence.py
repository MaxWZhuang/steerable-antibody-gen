"""Build B contracts: the exact finite-world teacher, the frozen q7 split and the Boolean function families.

What this is. The TEACHER side of a synthetic learned-evidence experiment: four labelled hidden Boolean
facts ``X0..X3``, an exact enumeration of the 16 worlds, raw observation events (fact, bit, reliability,
event identity), typed programs (READ / NOT / AND / XOR) and the exact posterior probability of a program.
It also freezes the q7 evidence split and the function families that the learner
(``functional_causal_learning``) is trained and evaluated on. What it is not: a learner, a hand-supplied
posterior feature, or a claim about biology. Every task here is a synthetic finite world.

The teacher is SEPARATE from the student. Truth tables, exact posteriors, execution traces, context
identities and the raw hidden world are teacher-side objects: nothing in this module is a tensor and
nothing here is ever passed to the learned forward. Identities (``context``, ``function_id``, event ids)
are grouping / validation metadata only.

Observation semantics. An event ``(fact i, bit b, reliability r)`` is a noisy report of the hidden bit
``X_i``: ``P(report = b | X_i = v) = r`` if ``v == b`` else ``1 - r``. Distinct event ids are independent
observations. An id repeated with IDENTICAL content counts once (it is one measurement shown twice); an id
repeated with different content is a contradiction in the record and is refused. Reliability 1/2 is
uninformative, reliability 0 or 1 is hard evidence, and evidence that rules out every world is refused
(never a division by zero and never a default). All arithmetic is exact (``fractions.Fraction``; a float
reliability converts exactly), so the oracle identities (1/2, 3/4, 9/10, ...) are exact equalities.
Realized entropy can INCREASE after contradictory evidence; only the expected information gain is
non-negative, and nothing here asserts otherwise.

Priors. The default is the uniform independent prior. An explicit normalized 16-world prior is allowed
for stress cases. The independent-factor interchange teacher (``interchange_probability``) substitutes a
donor posterior FACTOR into the base and keeps the base query; that is valid only when the world
distribution factorizes. It checks that exactly and REFUSES a correlated prior: with ``X0 == X1`` every
marginal is 1/2 but ``AND(X0, X1)`` is 1/2, not the product of marginals 1/4.

The q7 split. Evidence for a fact is a level 0..6 meaning the signed reliable-observation count
``m = level - 3`` at reliability 3/4 (posterior odds ``3**m``, levels 1/28, 1/10, 1/4, 1/2, 3/4, 9/10,
27/28). A context is the ORDERED 4-tuple of levels by fact identity. The modular code
``C = {(a, b, a+b, a+2b) mod 7}`` has 49 centers at minimum Hamming distance 3; its radius-one
neighbourhoods exclude 49 * 25 = 1225 of the 2401 signatures and leave 1176 SOURCE signatures, 168 at each
level of each coordinate, with exactly four legal single-fact replacement levels (the current one
included) at every source context and site. Centers are partitioned by ``t = (b - 2a) mod 7``: primary
``{0, 1}`` (14 centers), validation ``{2}`` (7), transfer-fit ``{3, 4}`` (14), transfer-eval ``{5, 6}``
(14). CLAIM BOUNDARY: signatures remain ordered by fact identity, so the split holds out unseen LABELLED
evidence contexts and NOT evidence permutation orbits (center ``(1,2,3,5)`` has the permutation
``(1,2,5,3)`` in the source set). Program splits, in contrast, hold out exact 16-world truth tables AND
their whole variable-renaming (S4) orbits; input complementation and output negation are NOT identified
(``NOT XOR`` is a different function and a different orbit from ``XOR``).

Functions. 48 source / 27 primary / 6 validation / 12 transfer functions from 10 / 3 / 1 / 1 S4 orbits
(families below). Every primitive appears in the source set. Syntax aliases, associations and
commutations of every family are enumerated and must map into that family's own truth tables only.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import numbers
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType

import numpy as np

from . import functional_baseline as fb
from .her2_runtime import require

N_FACTS = 4
N_WORLDS = 16
N_LEVELS = 7
LEVEL_MIDDLE = 3
LEVEL_RELIABILITY = 0.75
NULL_RELIABILITY = 0.5
POSTERIOR_LEVELS = (Fraction(1, 28), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2),
                    Fraction(3, 4), Fraction(9, 10), Fraction(27, 28))

ROLE_SOURCE, ROLE_PRIMARY, ROLE_VALIDATION, ROLE_TRANSFER = "source", "primary", "validation", "transfer"
FUNCTION_ROLES = (ROLE_SOURCE, ROLE_PRIMARY, ROLE_VALIDATION, ROLE_TRANSFER)
SPLIT_PRIMARY, SPLIT_VALIDATION = "primary", "validation"
SPLIT_TRANSFER_FIT, SPLIT_TRANSFER_EVAL = "transfer_fit", "transfer_eval"
CENTER_SPLITS = (SPLIT_PRIMARY, SPLIT_VALIDATION, SPLIT_TRANSFER_FIT, SPLIT_TRANSFER_EVAL)
_SPLIT_OF_T = {0: SPLIT_PRIMARY, 1: SPLIT_PRIMARY, 2: SPLIT_VALIDATION, 3: SPLIT_TRANSFER_FIT,
               4: SPLIT_TRANSFER_FIT, 5: SPLIT_TRANSFER_EVAL, 6: SPLIT_TRANSFER_EVAL}

#: Program token ids the STUDENT may see (a prefix serialization): 0 pads, 1..4 read X0..X3.
TOKEN_PAD, TOKEN_READ0, TOKEN_NOT, TOKEN_AND, TOKEN_XOR = 0, 1, 5, 6, 7
N_TOKENS = 8

#: The declared fixed-width synthetic identity of a (context, program) candidate for Build A.
LEVEL_LETTERS = "ABCDEFG"
TOKEN_LETTERS = {1: "W", 2: "X", 3: "Y", 4: "Z", 5: "N", 6: "T", 7: "O"}
PAD_LETTER = "Q"
IDENTITY_ALPHABET = LEVEL_LETTERS + "WXYZNTO" + PAD_LETTER
IDENTITY_CONVENTION = (
    "A candidate is ONE fixed-width uppercase string over the alphabet " + IDENTITY_ALPHABET + ": four "
    "evidence-level letters (A..G = level 0..6, fact order X0..X3) followed by the canonical program in "
    "PREFIX order (W X Y Z read X0..X3, N = NOT, T = AND, O = XOR) right-padded with the pad letter Q to the "
    "longest canonical program. Prefix order is self-delimiting, so the padding is unambiguous. The "
    "context alone repeats across its queries and is NOT an identity; the group is the context.")


# ---------------------------------------------------------------------------
# strict scalars
# ---------------------------------------------------------------------------

def _strict_int(value, name, *, low, high):
    require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)),
            f"{name} must be an integer and not a bool, got {value!r}")
    number = int(value)
    require(low <= number <= high, f"{name} must lie in [{low}, {high}], got {number}")
    return number


def _real(value, name):
    require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
            f"{name} must be a real number and not a bool, got {value!r}")
    number = float(value)
    require(math.isfinite(number), f"{name} must be finite, got {value!r}")
    return number


def _reliability(value, name):
    number = _real(value, name)
    require(0.0 <= number <= 1.0, f"{name} must lie in [0, 1], got {number}")
    return number


# ---------------------------------------------------------------------------
# raw observation events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One raw observation event: a noisy report ``bit`` of hidden fact ``fact`` with a known
    ``reliability`` and an independent event identity. Bits and indices are strict integers (never
    bools); reliability is finite and in [0, 1]."""

    event_id: str
    fact: int
    bit: int
    reliability: float

    def __post_init__(self):
        require(isinstance(self.event_id, str) and self.event_id
                and self.event_id == self.event_id.strip(),
                f"an event id must be a non-empty, stripped string, got {self.event_id!r}")
        object.__setattr__(self, "fact", _strict_int(self.fact, "observation fact", low=0,
                                                     high=N_FACTS - 1))
        object.__setattr__(self, "bit", _strict_int(self.bit, "observation bit", low=0, high=1))
        object.__setattr__(self, "reliability", _reliability(self.reliability,
                                                             "observation reliability"))


def canonical_events(events):
    """The validated, de-duplicated events in first-occurrence order.

    An id repeated with identical content is one measurement and counts ONCE; an id repeated with
    conflicting content refuses. Distinct ids are independent observations."""
    unique = {}
    for event in events:
        require(isinstance(event, Observation),
                f"events must be Observation objects, got {type(event).__name__}")
        seen = unique.get(event.event_id)
        if seen is None:
            unique[event.event_id] = event
        else:
            require(seen == event,
                    f"event id {event.event_id!r} is repeated with conflicting content ({seen} versus "
                    f"{event}): an identical repeat counts once, a conflicting duplicate refuses")
    return tuple(unique.values())


# ---------------------------------------------------------------------------
# world priors, exact posteriors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldPrior:
    """A normalized distribution over the 16 worlds (world ``w`` has ``X_i = (w >> i) & 1``)."""

    weights: tuple
    name: str = "explicit"

    def __post_init__(self):
        weights = tuple(Fraction(item) for item in self.weights)
        require(len(weights) == N_WORLDS and all(item >= 0 for item in weights),
                f"a world prior needs {N_WORLDS} non-negative weights")
        require(sum(weights, Fraction(0)) == 1, "a world prior must be normalized exactly (sum 1)")
        object.__setattr__(self, "weights", weights)


def explicit_prior(weights, name="explicit"):
    """A prior from 16 finite non-negative numbers that sum to 1 within 1e-9, renormalized exactly."""
    weights = list(weights)
    require(len(weights) == N_WORLDS, f"an explicit prior needs {N_WORLDS} weights, got {len(weights)}")
    exact = []
    for position, item in enumerate(weights):
        number = _real(item, f"prior weight {position}")
        require(number >= 0.0, f"prior weight {position} is negative: {number}")
        exact.append(Fraction(item) if isinstance(item, (Fraction, numbers.Integral))
                     else Fraction(number))
    total = sum(exact, Fraction(0))
    require(abs(float(total) - 1.0) <= 1e-9,
            f"an explicit prior must be normalized (sum within 1e-9 of 1), got {float(total)!r}")
    return WorldPrior(tuple(item / total for item in exact), name)


UNIFORM_PRIOR = WorldPrior(tuple(Fraction(1, N_WORLDS) for _ in range(N_WORLDS)),
                           "uniform_independent")


def correlated_prior_x0_equals_x1():
    """The stress prior: ``X0 == X1`` with the other facts uniform. Every marginal is 1/2, yet
    ``P(AND(X0, X1)) = 1/2``, not the product of marginals 1/4."""
    return WorldPrior(tuple(Fraction(1, 8) if (w & 1) == ((w >> 1) & 1) else Fraction(0)
                            for w in range(N_WORLDS)), "correlated_x0_equals_x1")


def marginals(distribution):
    """``P(X_i = 1)`` for the four facts under a 16-world distribution."""
    return tuple(sum((distribution[w] for w in range(N_WORLDS) if (w >> i) & 1), Fraction(0))
                 for i in range(N_FACTS))


def is_product_distribution(distribution):
    """Exactly: does the distribution equal the product of its own marginals?"""
    given = marginals(distribution)
    for w in range(N_WORLDS):
        expected = Fraction(1)
        for i in range(N_FACTS):
            expected *= given[i] if (w >> i) & 1 else 1 - given[i]
        if distribution[w] != expected:
            return False
    return True


@lru_cache(maxsize=None)
def _fact_likelihood(reports):
    """``(L(X=0), L(X=1))`` of one fact's reports ``((bit, reliability), ...)`` (exact)."""
    zero, one = Fraction(1), Fraction(1)
    for bit, reliability in reports:
        r = Fraction(reliability)
        if bit == 1:
            one *= r
            zero *= 1 - r
        else:
            zero *= r
            one *= 1 - r
    return zero, one


def world_posterior(events, prior=UNIFORM_PRIOR):
    """The exact posterior over the 16 worlds given raw events (duplicates by id counted once).

    Refuses evidence that rules out every world (inconsistent hard evidence): never a default."""
    require(isinstance(prior, WorldPrior), "prior must be a WorldPrior")
    per_fact = [[] for _ in range(N_FACTS)]
    for event in canonical_events(events):
        per_fact[event.fact].append((event.bit, event.reliability))
    likelihood = [_fact_likelihood(tuple(sorted(reports))) for reports in per_fact]
    weights = []
    for w in range(N_WORLDS):
        weight = prior.weights[w]
        for i in range(N_FACTS):
            weight = weight * likelihood[i][(w >> i) & 1]
        weights.append(weight)
    total = sum(weights, Fraction(0))
    require(total > 0,
            "the evidence is inconsistent: every world has zero posterior mass (hard evidence that "
            "contradicts itself or the prior); the teacher refuses instead of returning a default")
    return tuple(weight / total for weight in weights)


# ---------------------------------------------------------------------------
# typed programs
# ---------------------------------------------------------------------------

class Program:
    """Marker base class of the typed AST (READ, NOT, AND, XOR)."""


def _is_program(value):
    return isinstance(value, Program)


@dataclass(frozen=True)
class Read(Program):
    index: int

    def __post_init__(self):
        object.__setattr__(self, "index", _strict_int(self.index, "READ index", low=0,
                                                      high=N_FACTS - 1))


@dataclass(frozen=True)
class Not(Program):
    child: Program

    def __post_init__(self):
        require(_is_program(self.child), f"NOT needs one program, got {type(self.child).__name__}")


@dataclass(frozen=True)
class And(Program):
    left: Program
    right: Program

    def __post_init__(self):
        require(_is_program(self.left) and _is_program(self.right),
                "AND needs exactly two programs")


@dataclass(frozen=True)
class Xor(Program):
    left: Program
    right: Program

    def __post_init__(self):
        require(_is_program(self.left) and _is_program(self.right),
                "XOR needs exactly two programs")


def evaluate(program, world):
    """The 0/1 value of a program in one world (a plain recursive executor, no probabilities)."""
    world = _strict_int(world, "world", low=0, high=N_WORLDS - 1)
    if isinstance(program, Read):
        return (world >> program.index) & 1
    if isinstance(program, Not):
        return 1 - evaluate(program.child, world)
    if isinstance(program, And):
        return evaluate(program.left, world) & evaluate(program.right, world)
    if isinstance(program, Xor):
        return evaluate(program.left, world) ^ evaluate(program.right, world)
    raise ValueError(f"not a program: {program!r}")


def truth_table(program):
    """The exact 16-world truth table as a 16-bit integer (bit ``w`` = value in world ``w``)."""
    return sum(evaluate(program, w) << w for w in range(N_WORLDS))


def read_variables(program):
    """The facts a program syntactically reads."""
    if isinstance(program, Read):
        return frozenset((program.index,))
    if isinstance(program, Not):
        return read_variables(program.child)
    if isinstance(program, (And, Xor)):
        return read_variables(program.left) | read_variables(program.right)
    raise ValueError(f"not a program: {program!r}")


def depends_on(table, fact):
    """Semantic dependence: does flipping ``X_fact`` change the truth table anywhere?"""
    fact = _strict_int(fact, "fact", low=0, high=N_FACTS - 1)
    return any(((table >> w) & 1) != ((table >> (w ^ (1 << fact))) & 1) for w in range(N_WORLDS))


def program_text(program):
    if isinstance(program, Read):
        return f"X{program.index}"
    if isinstance(program, Not):
        return f"NOT({program_text(program.child)})"
    if isinstance(program, And):
        return f"AND({program_text(program.left)},{program_text(program.right)})"
    if isinstance(program, Xor):
        return f"XOR({program_text(program.left)},{program_text(program.right)})"
    raise ValueError(f"not a program: {program!r}")


def token_ids(program):
    """The PREFIX serialization of a program as student token ids (no padding)."""
    if isinstance(program, Read):
        return (TOKEN_READ0 + program.index,)
    if isinstance(program, Not):
        return (TOKEN_NOT,) + token_ids(program.child)
    if isinstance(program, And):
        return (TOKEN_AND,) + token_ids(program.left) + token_ids(program.right)
    if isinstance(program, Xor):
        return (TOKEN_XOR,) + token_ids(program.left) + token_ids(program.right)
    raise ValueError(f"not a program: {program!r}")


def _parse_tokens(ids, position):
    require(position < len(ids), "the token sequence ended before the program was complete (arity)")
    token = _strict_int(ids[position], "program token", low=0, high=N_TOKENS - 1)
    if TOKEN_READ0 <= token < TOKEN_READ0 + N_FACTS:
        return Read(token - TOKEN_READ0), position + 1
    if token == TOKEN_NOT:
        child, position = _parse_tokens(ids, position + 1)
        return Not(child), position
    if token in (TOKEN_AND, TOKEN_XOR):
        left, position = _parse_tokens(ids, position + 1)
        right, position = _parse_tokens(ids, position)
        return (And if token == TOKEN_AND else Xor)(left, right), position
    raise ValueError(f"token {token} cannot start a program")


def program_from_tokens(ids):
    """The inverse of :func:`token_ids` with strict arity: a missing operand or a trailing token
    refuses."""
    ids = tuple(ids)
    program, used = _parse_tokens(ids, 0)
    require(used == len(ids), f"{len(ids) - used} trailing token(s) after a complete program")
    return program


def program_from_document(document):
    """A program from ``{"op": "READ", "index": i}`` / ``{"op": "NOT", "args": [p]}`` /
    ``{"op": "AND"|"XOR", "args": [p, q]}`` with strict arity and index validation."""
    require(isinstance(document, dict) and "op" in document, f"not a program document: {document!r}")
    op = document["op"]
    if op == "READ":
        require(set(document) == {"op", "index"}, "READ takes exactly one index")
        return Read(document["index"])
    require(op in ("NOT", "AND", "XOR"), f"unknown program operator {op!r}")
    require(set(document) == {"op", "args"} and isinstance(document["args"], list),
            f"{op} takes exactly an 'args' list")
    args = [program_from_document(item) for item in document["args"]]
    arity = 1 if op == "NOT" else 2
    require(len(args) == arity, f"{op} takes exactly {arity} argument(s), got {len(args)}")
    return {"NOT": Not, "AND": And, "XOR": Xor}[op](*args)


# ---------------------------------------------------------------------------
# the teacher
# ---------------------------------------------------------------------------

def table_probability(table, posterior):
    """The probability a truth table is true under a 16-world distribution (sum of world weights)."""
    return sum((posterior[w] for w in range(N_WORLDS) if (table >> w) & 1), Fraction(0))


def program_probability(program, events, prior=UNIFORM_PRIOR):
    """The exact posterior probability that ``program`` is true given raw ``events``."""
    require(isinstance(program, Program), "program_probability needs a typed program")
    return table_probability(truth_table(program), world_posterior(events, prior))


def require_independent_factor_regime(prior):
    """Refuse any world prior that is not the product of its own marginals."""
    require(isinstance(prior, WorldPrior), "prior must be a WorldPrior")
    require(is_product_distribution(prior.weights),
            f"the prior {prior.name!r} is not a product of its marginals (the facts are "
            "correlated): the independent-factor interchange teacher substitutes a donor posterior "
            "FACTOR into the base and multiplies marginals, which is wrong here (with X0 == X1 every "
            "marginal is 1/2 but AND(X0, X1) is 1/2, not 1/4). Use the full-world teacher or a "
            "richer representation; the product-of-marginals target is never applied silently")


def interchange_probability(base_events, donor_events, site, program, prior=UNIFORM_PRIOR):
    """The independent-factor interchange target: the base query probability after replacing the base
    posterior factor of ``site`` by the donor's, with every other factor kept. Valid only for a product
    prior (checked exactly, refused otherwise); it equals the full teacher on the raw-evidence hybrid
    there, which the tests check."""
    require_independent_factor_regime(prior)
    site = _strict_int(site, "site", low=0, high=N_FACTS - 1)
    chosen = list(marginals(world_posterior(base_events, prior)))
    chosen[site] = marginals(world_posterior(donor_events, prior))[site]
    table = truth_table(program)
    total = Fraction(0)
    for w in range(N_WORLDS):
        if (table >> w) & 1:
            weight = Fraction(1)
            for i in range(N_FACTS):
                weight *= chosen[i] if (w >> i) & 1 else 1 - chosen[i]
            total += weight
    return total


# ---------------------------------------------------------------------------
# truth-table symmetry
# ---------------------------------------------------------------------------

PERMUTATIONS = tuple(itertools.permutations(range(N_FACTS)))


def rename_table(table, permutation):
    """The table of the function with its variables renamed by ``permutation``
    (``g(x) = f(y)`` where ``y_i = x_permutation[i]``)."""
    out = 0
    for w in range(N_WORLDS):
        source = 0
        for i in range(N_FACTS):
            source |= ((w >> permutation[i]) & 1) << i
        out |= ((table >> source) & 1) << w
    return out


def orbit_key(table):
    """The S4 variable-renaming orbit of a truth table (its least member). Complementing the inputs or
    negating the output is NOT an equivalence here."""
    return min(rename_table(table, permutation) for permutation in PERMUTATIONS)


# ---------------------------------------------------------------------------
# function families
# ---------------------------------------------------------------------------

def _r(index):
    return Read(index)


def _nary(op, leaves):
    """Every binary-tree arrangement (association and commutation) of ``op`` over ``leaves``."""
    leaves = list(leaves)
    if len(leaves) == 1:
        return [leaves[0]]
    count, out = len(leaves), []
    for mask in range(1, (1 << count) - 1):
        left = [leaves[i] for i in range(count) if (mask >> i) & 1]
        right = [leaves[i] for i in range(count) if not (mask >> i) & 1]
        for lhs in _nary(op, left):
            for rhs in _nary(op, right):
                out.append(op(lhs, rhs))
    return out


def _both_orders(op, xs, ys):
    return [op(x, y) for x in xs for y in ys] + [op(y, x) for x in xs for y in ys]


def _b_read(v):
    return [_r(v[0])]


def _b_not_read(v):
    return [Not(_r(v[0]))]


def _b_and2(v):
    return _nary(And, [_r(v[0]), _r(v[1])])


def _b_xor2(v):
    return _nary(Xor, [_r(v[0]), _r(v[1])])


def _b_not_xor2(v):
    return [Not(x) for x in _nary(Xor, [_r(v[0]), _r(v[1])])]


def _b_and3(v):
    return _nary(And, [_r(i) for i in v])


def _b_xor3(v):
    return _nary(Xor, [_r(i) for i in v])


def _b_and4(v):
    return _nary(And, [_r(i) for i in v])


def _b_xor4(v):
    return _nary(Xor, [_r(i) for i in v])


def _b_and_andnot(v):
    return _nary(And, [_r(v[0]), _r(v[1]), Not(_r(v[2]))])


def _b_and_xor_read(v):
    return _both_orders(And, _nary(Xor, [_r(v[0]), _r(v[1])]), [_r(v[2])])


def _b_and_xor_xor(v):
    return _both_orders(And, _nary(Xor, [_r(v[0]), _r(v[1])]), _nary(Xor, [_r(v[2]), _r(v[3])]))


def _b_and_xor_not(v):
    return _both_orders(And, _nary(Xor, [_r(v[0]), _r(v[1])]), [Not(_r(v[2]))])


def _b_not_and2(v):
    return [Not(x) for x in _nary(And, [_r(v[0]), _r(v[1])])]


def _b_xor_and_read(v):
    return _both_orders(Xor, _nary(And, [_r(v[0]), _r(v[1])]), [_r(v[2])])


@dataclass(frozen=True)
class FamilyDef:
    name: str
    role: str
    arity: int
    functions: int
    builder: object


FAMILY_DEFS = (
    FamilyDef("READ", ROLE_SOURCE, 1, 4, _b_read),
    FamilyDef("NOT_READ", ROLE_SOURCE, 1, 4, _b_not_read),
    FamilyDef("AND2", ROLE_SOURCE, 2, 6, _b_and2),
    FamilyDef("XOR2", ROLE_SOURCE, 2, 6, _b_xor2),
    FamilyDef("NOT_XOR2", ROLE_SOURCE, 2, 6, _b_not_xor2),
    FamilyDef("AND3", ROLE_SOURCE, 3, 4, _b_and3),
    FamilyDef("XOR3", ROLE_SOURCE, 3, 4, _b_xor3),
    FamilyDef("AND4", ROLE_SOURCE, 4, 1, _b_and4),
    FamilyDef("XOR4", ROLE_SOURCE, 4, 1, _b_xor4),
    FamilyDef("AND_ANDNOT", ROLE_SOURCE, 3, 12, _b_and_andnot),
    FamilyDef("AND_XOR_READ", ROLE_PRIMARY, 3, 12, _b_and_xor_read),
    FamilyDef("AND_XOR_XOR", ROLE_PRIMARY, 4, 3, _b_and_xor_xor),
    FamilyDef("AND_XOR_NOT", ROLE_PRIMARY, 3, 12, _b_and_xor_not),
    FamilyDef("NOT_AND2", ROLE_VALIDATION, 2, 6, _b_not_and2),
    FamilyDef("XOR_AND_READ", ROLE_TRANSFER, 3, 12, _b_xor_and_read),
)
EXPECTED_FUNCTIONS = {ROLE_SOURCE: 48, ROLE_PRIMARY: 27, ROLE_VALIDATION: 6, ROLE_TRANSFER: 12}
EXPECTED_ORBITS = {ROLE_SOURCE: 10, ROLE_PRIMARY: 3, ROLE_VALIDATION: 1, ROLE_TRANSFER: 1}


@dataclass(frozen=True, eq=False)
class FunctionSpec:
    """One distinct Boolean function: its canonical program, exact truth table and S4 orbit."""

    index: int
    function_id: str
    family: str
    role: str
    program: Program
    table: int
    orbit: int
    tokens: tuple


@dataclass(frozen=True, eq=False)
class FunctionRegistry:
    functions: tuple
    by_table: object
    alias_tables: object
    alias_programs: object
    program_width: int

    def role(self, role):
        require(role in FUNCTION_ROLES, f"role must be one of {FUNCTION_ROLES}")
        return tuple(item for item in self.functions if item.role == role)

    def of_family(self, family):
        return tuple(item for item in self.functions if item.family == family)

    def tables(self, role):
        return frozenset(item.table for item in self.role(role))

    def orbits(self, role):
        return frozenset(item.orbit for item in self.role(role))


def _build_registry():
    alias_tables, alias_programs, canonical = {}, {}, {}
    owner = {}
    for family in FAMILY_DEFS:
        programs_by_table = {}
        for assignment in itertools.permutations(range(N_FACTS), family.arity):
            for program in family.builder(assignment):
                programs_by_table.setdefault(truth_table(program), []).append(program)
        tables = frozenset(programs_by_table)
        require(len(tables) == family.functions,
                f"family {family.name} has {len(tables)} distinct truth tables, expected "
                f"{family.functions}")
        for table in sorted(tables):
            # the message may only name an owner that exists, so it is built from ``get`` and never
            # from ``owner[table]``, which does not exist yet for a table seen for the first time
            previous = owner.get(table)
            require(previous is None,
                    f"truth table {table:#06x} belongs to both {previous} and {family.name}: "
                    "an alias leaks across families")
            owner[table] = family.name
        orbits = {orbit_key(table) for table in tables}
        require(len(orbits) == 1,
                f"family {family.name} is not one S4 orbit ({len(orbits)} orbits)")
        alias_tables[family.name] = tables
        alias_programs[family.name] = tuple(
            program for table in sorted(programs_by_table) for program in programs_by_table[table])
        for table, programs in programs_by_table.items():
            canonical[table] = min(programs, key=lambda item: (len(token_ids(item)),
                                                               token_ids(item)))
    functions = []
    for role in FUNCTION_ROLES:
        for family in FAMILY_DEFS:
            if family.role != role:
                continue
            for table in sorted(alias_tables[family.name]):
                program = canonical[table]
                functions.append(FunctionSpec(
                    index=len(functions), function_id=f"{family.name}-{table:04x}",
                    family=family.name, role=role, program=program, table=table,
                    orbit=orbit_key(table), tokens=token_ids(program)))
    registry = FunctionRegistry(
        functions=tuple(functions),
        by_table=MappingProxyType({item.table: item for item in functions}),
        alias_tables=MappingProxyType(dict(alias_tables)),
        alias_programs=MappingProxyType(dict(alias_programs)),
        program_width=max(len(item.tokens) for item in functions))
    for role in FUNCTION_ROLES:
        require(len(registry.role(role)) == EXPECTED_FUNCTIONS[role],
                f"{role} has {len(registry.role(role))} functions, expected "
                f"{EXPECTED_FUNCTIONS[role]}")
        require(len(registry.orbits(role)) == EXPECTED_ORBITS[role],
                f"{role} has {len(registry.orbits(role))} S4 orbits, expected "
                f"{EXPECTED_ORBITS[role]}")
    for left, right in itertools.combinations(FUNCTION_ROLES, 2):
        require(not registry.tables(left) & registry.tables(right),
                f"{left} and {right} share an exact truth table")
        require(not registry.orbits(left) & registry.orbits(right),
                f"{left} and {right} share an S4 variable-renaming orbit")
    source_primitives = {family.name for family in FAMILY_DEFS if family.role == ROLE_SOURCE}
    require({"READ", "NOT_READ", "AND2", "XOR2"} <= source_primitives,
            "every primitive must occur in the source set")
    return registry


@lru_cache(maxsize=None)
def function_registry():
    """The frozen registry of the 93 distinct functions (built and cross-checked once)."""
    return _build_registry()


def program_width():
    return function_registry().program_width


# ---------------------------------------------------------------------------
# the q7 split
# ---------------------------------------------------------------------------

def hamming(left, right):
    return sum(1 for a, b in zip(left, right) if a != b)


def _signature(value):
    require(isinstance(value, (tuple, list)) and len(value) == N_FACTS,
            f"a signature is {N_FACTS} levels, got {value!r}")
    return tuple(_strict_int(item, "evidence level", low=0, high=N_LEVELS - 1) for item in value)


def level_m(level):
    """The signed reliable-observation count of an evidence level."""
    return _strict_int(level, "evidence level", low=0, high=N_LEVELS - 1) - LEVEL_MIDDLE


def level_posterior(level):
    return POSTERIOR_LEVELS[_strict_int(level, "evidence level", low=0, high=N_LEVELS - 1)]


def center_t(center):
    a, b = center[0], center[1]
    return (b - 2 * a) % 7


@dataclass(frozen=True, eq=False)
class Q7World:
    centers: tuple
    all_signatures: tuple
    excluded: frozenset
    source: tuple
    source_index: object
    split_of_center: object
    centers_by_split: object
    source_by_level: object

    def legal_levels(self, signature, site):
        """The levels ``l`` for which replacing ``signature[site]`` by ``l`` stays in the source set."""
        signature = _signature(signature)
        site = _strict_int(site, "site", low=0, high=N_FACTS - 1)
        return tuple(level for level in range(N_LEVELS)
                     if signature[:site] + (level,) + signature[site + 1:] in self.source_index)


@lru_cache(maxsize=None)
def q7_world():
    centers = tuple((a, b, (a + b) % 7, (a + 2 * b) % 7) for a in range(7) for b in range(7))
    everything = tuple(itertools.product(range(N_LEVELS), repeat=N_FACTS))
    excluded = frozenset(s for s in everything if min(hamming(s, c) for c in centers) <= 1)
    source = tuple(s for s in everything if s not in excluded)
    by_split = {name: [] for name in CENTER_SPLITS}
    for center in centers:
        by_split[_SPLIT_OF_T[center_t(center)]].append(center)
    by_level = {(site, level): [] for site in range(N_FACTS) for level in range(N_LEVELS)}
    for position, signature in enumerate(source):
        for site in range(N_FACTS):
            by_level[(site, signature[site])].append(position)
    return Q7World(
        centers=centers, all_signatures=everything, excluded=excluded, source=source,
        source_index=MappingProxyType({s: i for i, s in enumerate(source)}),
        split_of_center=MappingProxyType({c: _SPLIT_OF_T[center_t(c)] for c in centers}),
        centers_by_split=MappingProxyType({k: tuple(v) for k, v in by_split.items()}),
        source_by_level=MappingProxyType({k: tuple(v) for k, v in by_level.items()}))


def hybrid_signature(base, donor, site):
    """``base`` with the level of ``site`` taken from ``donor`` (the one-factor hybrid)."""
    base, donor = _signature(base), _signature(donor)
    site = _strict_int(site, "site", low=0, high=N_FACTS - 1)
    return base[:site] + (donor[site],) + base[site + 1:]


def verify_q7():
    """Independently re-derive and CHECK every frozen q7 count (raises ``ValueError`` on any miss).

    Returns the counts. Includes the exact code, its minimum distance, the radius-one exclusion, the
    per-coordinate/per-level source balance, the four legal replacements at every source context and
    site (every hybrid is checked), the stratum sizes and level balance, and that primary hybrids leave
    the source support."""
    world = q7_world()
    require(len(world.centers) == 49 and len(set(world.centers)) == 49, "the code must have 49 centers")
    require(min(hamming(a, b) for a, b in itertools.combinations(world.centers, 2)) == 3,
            "the code's minimum Hamming distance must be 3")
    require(len(world.all_signatures) == 2401, "there must be 2401 signatures")
    require(len(world.excluded) == 49 * 25 == 1225,
            f"the radius-one neighbourhoods must exclude 1225 signatures, got {len(world.excluded)}")
    require(len(world.source) == 1176, f"expected 1176 source signatures, got {len(world.source)}")
    for site in range(N_FACTS):
        for level in range(N_LEVELS):
            require(len(world.source_by_level[(site, level)]) == 168,
                    f"coordinate {site} level {level} has "
                    f"{len(world.source_by_level[(site, level)])} source signatures, expected 168")
    for signature in world.source:
        for site in range(N_FACTS):
            legal = world.legal_levels(signature, site)
            require(len(legal) == 4 and signature[site] in legal,
                    f"source context {signature} site {site} has legal levels {legal}, expected "
                    "exactly four including the current level")
            for level in range(N_LEVELS):
                hybrid = signature[:site] + (level,) + signature[site + 1:]
                require((hybrid in world.source_index) == (level in legal),
                        f"hybrid {hybrid} disagrees with the legal-level set {legal}")
    sizes = {name: len(world.centers_by_split[name]) for name in CENTER_SPLITS}
    require(sizes == {SPLIT_PRIMARY: 14, SPLIT_VALIDATION: 7, SPLIT_TRANSFER_FIT: 14,
                      SPLIT_TRANSFER_EVAL: 14}, f"stratum sizes are {sizes}")
    for name, members in world.centers_by_split.items():
        per_level = len(members) // N_LEVELS
        for site in range(N_FACTS):
            counts = [sum(1 for c in members if c[site] == level) for level in range(N_LEVELS)]
            require(counts == [per_level] * N_LEVELS,
                    f"stratum {name} coordinate {site} is not level-balanced: {counts}")
    for base in world.centers_by_split[SPLIT_PRIMARY]:
        for donor in world.centers_by_split[SPLIT_PRIMARY]:
            for site in range(N_FACTS):
                hybrid = hybrid_signature(base, donor, site)
                require(hybrid in world.excluded,
                        f"primary hybrid {hybrid} must leave the source support")
    return {"centers": 49, "excluded": 1225, "source": 1176, "per_level": 168, "legal_levels": 4,
            "strata": sizes}


EXPECTED_PANELS = {
    "source_context_query_pairs": 56448, "validation_new_function": 42,
    "validation_source_function": 336, "primary_composition": 378, "primary_source_preservation": 672,
    "transfer_fit_candidates": 168, "transfer_eval_candidates": 168,
    "strong_primary_interventions": 21168}


def panel_counts():
    """The ACTUAL case counts of every declared panel, derived from the frozen split and the function
    registry (not copied from the declaration)."""
    world, registry = q7_world(), function_registry()
    n = {role: len(registry.role(role)) for role in FUNCTION_ROLES}
    c = {name: len(members) for name, members in world.centers_by_split.items()}
    return {
        "source_context_query_pairs": len(world.source) * n[ROLE_SOURCE],
        "validation_new_function": c[SPLIT_VALIDATION] * n[ROLE_VALIDATION],
        "validation_source_function": c[SPLIT_VALIDATION] * n[ROLE_SOURCE],
        "primary_composition": c[SPLIT_PRIMARY] * n[ROLE_PRIMARY],
        "primary_source_preservation": c[SPLIT_PRIMARY] * n[ROLE_SOURCE],
        "transfer_fit_candidates": c[SPLIT_TRANSFER_FIT] * n[ROLE_TRANSFER],
        "transfer_eval_candidates": c[SPLIT_TRANSFER_EVAL] * n[ROLE_TRANSFER],
        "strong_primary_interventions": c[SPLIT_PRIMARY] ** 2 * N_FACTS * n[ROLE_PRIMARY]}


def verify_world():
    """``verify_q7`` plus the actual panel counts: raises ``ValueError`` unless every declared count
    holds. Run BEFORE any training."""
    summary = verify_q7()
    counts = panel_counts()
    require(counts == EXPECTED_PANELS, f"the panel counts are {counts}, expected {EXPECTED_PANELS}")
    return {**summary, "panels": counts}


# ---------------------------------------------------------------------------
# raw evidence for a signature
# ---------------------------------------------------------------------------

def max_events(cancel_max, null_max):
    """The most raw events a variant can hold: the fixed padded event width of the learner."""
    return N_FACTS * (3 + 2 * cancel_max + null_max)


def signature_posteriors(signature):
    """Exact per-fact posteriors of a signature (odds 3**m under the uniform prior)."""
    return tuple(level_posterior(level) for level in _signature(signature))


def raw_events(signature, variant, *, cancel_max=1, null_max=2, prefix=None):
    """Raw observation events for a labelled signature. Deterministic in ``(signature, variant)``.

    Variant 0 is the plain minimal record (``|m|`` reliable events per fact). A variant above 0 adds, per
    fact, up to ``cancel_max`` independently identified cancelling pairs (one positive and one negative
    reliable event) and up to ``null_max`` reliability-1/2 events, and shuffles the order, so total event
    count is not a sufficient statistic. Every variant has exactly the posterior of its signature."""
    signature = _signature(signature)
    variant = _strict_int(variant, "variant", low=0, high=10 ** 6)
    cancel_max = _strict_int(cancel_max, "cancel_max", low=0, high=4)
    null_max = _strict_int(null_max, "null_max", low=0, high=4)
    prefix = prefix or ("s" + "".join(str(level) for level in signature) + f"v{variant}")
    rng = np.random.default_rng([variant, 9100, *signature])
    drawn = []
    for fact in range(N_FACTS):
        m = level_m(signature[fact])
        reports = [(1 if m > 0 else 0, LEVEL_RELIABILITY)] * abs(m)
        if variant > 0:
            cancels = int(rng.integers(0, cancel_max + 1))
            nulls = int(rng.integers(0, null_max + 1))
            for _ in range(cancels):
                reports += [(1, LEVEL_RELIABILITY), (0, LEVEL_RELIABILITY)]
            for _ in range(nulls):
                reports.append((int(rng.integers(0, 2)), NULL_RELIABILITY))
        drawn += [(fact, bit, reliability) for bit, reliability in reports]
    if variant > 0:
        drawn = [drawn[int(i)] for i in rng.permutation(len(drawn))]
    counters = [0] * N_FACTS
    events = []
    for fact, bit, reliability in drawn:
        events.append(Observation(f"{prefix}.f{fact}.{counters[fact]}", fact, bit, reliability))
        counters[fact] += 1
    return tuple(events)


def signature_of_events(events):
    """The labelled signature a canonical raw record belongs to (refuses a non-q7 record)."""
    totals = [0] * N_FACTS
    for event in canonical_events(events):
        if event.reliability == LEVEL_RELIABILITY:
            totals[event.fact] += 1 if event.bit == 1 else -1
        else:
            require(event.reliability == NULL_RELIABILITY,
                    f"event {event.event_id!r} has reliability {event.reliability}: only "
                    f"{LEVEL_RELIABILITY} and {NULL_RELIABILITY} belong to the q7 evidence")
    for fact, total in enumerate(totals):
        require(-3 <= total <= 3, f"fact {fact} has net reliable count {total}, outside -3..3")
    return tuple(total + LEVEL_MIDDLE for total in totals)


def hybrid_events(base_events, donor_events, site):
    """The raw-evidence one-factor hybrid: the base events of every other fact plus the DONOR's events
    of ``site``. Event ids are kept, so distinct sources stay independent observations."""
    site = _strict_int(site, "site", low=0, high=N_FACTS - 1)
    return (tuple(e for e in base_events if e.fact != site)
            + tuple(e for e in donor_events if e.fact == site))


def restrict_events(events, facts):
    """The events that report one of ``facts`` (the others are not observed)."""
    facts = frozenset(_strict_int(f, "fact", low=0, high=N_FACTS - 1) for f in facts)
    return tuple(e for e in events if e.fact in facts)


# ---------------------------------------------------------------------------
# the Build A candidate identity
# ---------------------------------------------------------------------------

def identity_length():
    return N_FACTS + program_width()


def identity_sequence(signature, program):
    """The fixed-width synthetic identity of a (context, canonical program) candidate."""
    signature = _signature(signature)
    require(isinstance(program, Program), "identity_sequence needs a typed program")
    letters = "".join(LEVEL_LETTERS[level] for level in signature)
    body = "".join(TOKEN_LETTERS[token] for token in token_ids(program))
    width = program_width()
    require(len(body) <= width, f"program {program_text(program)} exceeds the identity width {width}")
    return letters + body + PAD_LETTER * (width - len(body))


def parse_identity_sequence(sequence):
    """``(signature, program)`` from an identity string (strict: the padding must be the pad letter)."""
    require(isinstance(sequence, str) and len(sequence) == identity_length()
            and all(c in IDENTITY_ALPHABET for c in sequence),
            f"{sequence!r} is not a fixed-width identity of length {identity_length()}")
    signature = tuple(LEVEL_LETTERS.index(c) for c in sequence[:N_FACTS])
    body = sequence[N_FACTS:]
    stop = body.find(PAD_LETTER)
    stop = len(body) if stop < 0 else stop
    require(set(body[stop:]) <= {PAD_LETTER} and stop > 0, f"{sequence!r} has a malformed program body")
    letter_token = {letter: token for token, letter in TOKEN_LETTERS.items()}
    require(all(c in letter_token for c in body[:stop]),
            f"{sequence!r} has a non-program letter inside its program body")
    return signature, program_from_tokens([letter_token[c] for c in body[:stop]])


def context_text(signature):
    return "".join(LEVEL_LETTERS[level] for level in _signature(signature))


def candidate_id(signature, function):
    return f"{context_text(signature)}-{function.function_id}"


def group_id(signature):
    """The group of a candidate is its evidence context (every query of a context shares it)."""
    return f"ctx-{context_text(signature)}"


# ---------------------------------------------------------------------------
# the frozen world document
# ---------------------------------------------------------------------------

def world_document():
    """A plain JSON description of the frozen world (centers, split, functions, orbits)."""
    world, registry = q7_world(), function_registry()
    return {
        "schema": "functional-evidence-world/1",
        "levels": {"count": N_LEVELS, "reliability": LEVEL_RELIABILITY,
                   "posteriors": [f"{p.numerator}/{p.denominator}" for p in POSTERIOR_LEVELS]},
        "q7": {"centers": [list(c) for c in world.centers],
               "split_of_center": {context_text(c): world.split_of_center[c] for c in world.centers},
               "source_sha256": fb.digest([list(s) for s in world.source]),
               "excluded_sha256": fb.digest(sorted(list(s) for s in world.excluded)),
               "counts": {"centers": len(world.centers), "excluded": len(world.excluded),
                          "source": len(world.source)}},
        "functions": [{"index": f.index, "id": f.function_id, "family": f.family, "role": f.role,
                       "table": f.table, "orbit": f.orbit, "tokens": list(f.tokens)}
                      for f in registry.functions],
        "program_width": registry.program_width,
        "identity": {"alphabet": IDENTITY_ALPHABET, "length": identity_length(),
                     "convention": IDENTITY_CONVENTION},
        "claim_boundary": ("signatures remain ORDERED by fact identity: the split holds out unseen "
                           "labelled evidence contexts, not evidence permutation orbits; programs hold "
                           "out exact truth tables and their whole S4 renaming orbits"),
    }


def world_sha256():
    return fb.digest(world_document())


def stable_seed(*parts):
    """A deterministic 63-bit integer from text/number parts (never Python's salted ``hash``)."""
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") >> 1
