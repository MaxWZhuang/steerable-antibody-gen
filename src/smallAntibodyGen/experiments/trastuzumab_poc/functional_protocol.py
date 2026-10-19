"""Outcome-free functional budget, pair schedule and restricted measurements.

The observation budget and the pair schedule are functions of identity only: the
qualified functional document, a budget seed, and ``(update, example)`` keys.  Measured
values are accepted only AFTER a budget is fixed, only for exactly the selected
observations, and only inside :class:`RestrictedFunctionalData`.

An observation is a ``(group_id, construct_id)`` pair.  One physical construct measured
in two groups is two observations but ONE construct, so every report carries both counts.

Order of operations (enforced by the signatures)
  1. ``qualify_functional`` (functional_data) -> ``QualifiedFunctional``
  2. ``select_functional_budget`` -> ``FunctionalBudget`` (identity only)
  3. ``functional_exposure_plan`` / ``replay_schedule`` / ``scheduled_coverage``
     (data-free, no labels)
  4. ``load_restricted_measurements`` (label-dependent) -> ``RestrictedFunctionalData``

The measurements file is necessarily decoded by the JSON parser before its
``selection_digest`` is compared; the guarantee is that no value is interpreted,
validated or returned unless that digest matches the fixed budget.

No Torch dependency.  No redraw: a tied or unresolved pair still counts as one
opportunity.  No all-pairs enumeration: pair counts are arithmetic and replay streams
counters and sets whose size is bounded by the number of selected observations.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from smallAntibodyGen.experiments.trastuzumab_poc.functional_data import (
    DIRECTIONS, QualifiedFunctional, exact_keys, identifier, json_digest, read_strict_json,
    sha256_hex,
)

BUDGET_SCHEMA = "trastuzumab-functional-budget/1"
SELECTION_SCHEMA = "trastuzumab-functional-selection/1"
MEASUREMENTS_SCHEMA = "trastuzumab-functional-measurements/1"
EXPOSURE_SCHEMA = "trastuzumab-functional-exposure/1"
# Must equal randomness.RNG_ALGORITHM (that module imports Torch, this one must not);
# a test pins the equivalence of _keyed_seed and randomness.keyed_seed.
RNG_ALGORITHM = "sha256-purpose-update/1"
PAIR_PURPOSE = "functional-pair"

A_BETTER = "a_better"
B_BETTER = "b_better"
TIE = "tie"
UNRESOLVED = "unresolved"

MEASUREMENT_KINDS = ("point", "interval", "lower_bound", "upper_bound")
_KIND_FIELDS = {"point": ("value",), "interval": ("lower", "upper"),
                "lower_bound": ("lower",), "upper_bound": ("upper",)}

Observation = Tuple[str, str]


# ---------------------------------------------------------------------------
# Small validators and the keyed seed
# ---------------------------------------------------------------------------

def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


def _keyed_seed(seed, update, example, purpose):
    """Same value as ``randomness.keyed_seed`` without importing Torch."""
    _seed(seed)
    if any(type(x) is not int or x < 0 for x in (update, example)):
        raise ValueError("update and example must be non-negative integers")
    return int(json_digest({"algorithm": RNG_ALGORITHM, "seed": seed, "update": update,
                            "example": example, "purpose": purpose})[:16], 16) % (2**63)


def _budget_seed(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("budget_seed must be a nonempty string without NUL")
    return value


# ---------------------------------------------------------------------------
# Measurements and interval comparison
# ---------------------------------------------------------------------------

def _finite(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{where} is not representable as a finite float") from None
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return 0.0 if number == 0 else number


@dataclass(frozen=True)
class Measurement:
    """An exact point, a closed interval, or a one-sided censored bound.

    Censoring is an explicit ``kind`` with the missing bound simply absent; JSON
    infinity is never used.  Bounds are closed, so touching bounds are unresolved.

      point        {"kind": "point", "value": v}
      interval     {"kind": "interval", "lower": a, "upper": b}   (a < b)
      lower_bound  {"kind": "lower_bound", "lower": a}            true value >= a
      upper_bound  {"kind": "upper_bound", "upper": b}            true value <= b
    """
    kind: str
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in MEASUREMENT_KINDS:
            raise ValueError(f"measurement kind must be one of {MEASUREMENT_KINDS}")
        required = _KIND_FIELDS[self.kind]
        present = tuple(name for name in ("value", "lower", "upper")
                        if getattr(self, name) is not None)
        if set(present) != set(required):
            raise ValueError(f"{self.kind} measurement requires exactly {list(required)}")
        for name in required:
            object.__setattr__(self, name, _finite(getattr(self, name), f"measurement {name}"))
        if self.kind == "interval" and not self.lower < self.upper:
            raise ValueError("interval measurement requires lower < upper; use a point otherwise")

    def bounds(self) -> Tuple[float, float]:
        """Closed raw-value bounds; infinity is internal and never serialized."""
        if self.kind == "point":
            return (self.value, self.value)
        if self.kind == "interval":
            return (self.lower, self.upper)
        if self.kind == "lower_bound":
            return (self.lower, math.inf)
        return (-math.inf, self.upper)

    def document(self) -> Dict[str, Any]:
        return {"kind": self.kind, **{name: getattr(self, name) for name in _KIND_FIELDS[self.kind]}}

    @classmethod
    def from_document(cls, document) -> "Measurement":
        if not isinstance(document, dict) or document.get("kind") not in MEASUREMENT_KINDS:
            raise ValueError(f"measurement must be an object with a kind in {MEASUREMENT_KINDS}")
        kind = document["kind"]
        exact_keys(document, ("kind", *_KIND_FIELDS[kind]), f"{kind} measurement")
        return cls(kind=kind, **{name: document[name] for name in _KIND_FIELDS[kind]})


def compare(a: Measurement, b: Measurement, direction: str) -> str:
    """Oriented order of two measurements: a_better | b_better | tie | unresolved.

    Two equal points tie.  Strictly disjoint closed intervals resolve the order.  Every
    other case (overlap, touching bounds, censoring that cannot separate the values,
    equal non-point intervals) is unresolved; nothing is guessed.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if not isinstance(a, Measurement) or not isinstance(b, Measurement):
        raise TypeError("compare needs two Measurement objects")
    if a.kind == "point" and b.kind == "point" and a.value == b.value:
        return TIE
    a_low, a_high = a.bounds()
    b_low, b_high = b.bounds()
    if a_high < b_low:
        a_larger = False
    elif b_high < a_low:
        a_larger = True
    else:
        return UNRESOLVED
    return A_BETTER if a_larger == (direction == "higher_better") else B_BETTER


# ---------------------------------------------------------------------------
# Budget (identity only)
# ---------------------------------------------------------------------------

def _selection_key(seed: str, group_id: str, construct_id: str) -> str:
    return hashlib.sha256(f"{seed}\x00{group_id}\x00{construct_id}".encode("utf-8")).hexdigest()


def _selection_digest(qualified_digest, contract_digest, seed, n, selected) -> str:
    return json_digest({"schema": SELECTION_SCHEMA, "qualified": qualified_digest,
                        "contract": contract_digest, "budget_seed": seed, "budget_n": n,
                        "selected": [[g, c] for g, c in selected]})


@dataclass(frozen=True)
class FunctionalBudget:
    """A fixed, nested prefix of the seeded observation order.

    Observations are ordered by ``sha256(seed NUL group_id NUL construct_id)`` (ties by
    identity), so budgets with the same seed nest as prefixes.  Construction re-derives
    only what is checkable without the qualified document: count, uniqueness, hash order
    and the selection digest.  It cannot know which candidates were left out, so a
    self-consistent budget that is NOT the prefix of the seeded order (for example
    ``full.selected[1:3]`` with its digest recomputed) can be constructed.  Every consumer
    that holds the qualified document goes through ``_check_budget_matches``, which
    recomputes ``select_functional_budget`` and requires exact equality.
    """
    qualified_digest: str
    contract_digest: str
    budget_seed: str
    budget_n: int
    selected: Tuple[Observation, ...]
    selection_digest: str
    _groups: Dict[str, Tuple[str, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("qualified_digest", "contract_digest", "selection_digest"):
            sha256_hex(getattr(self, name), f"budget.{name}")
        _budget_seed(self.budget_seed)
        _positive(self.budget_n, "budget_n")
        selected = []
        for item in self.selected:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("selected observations must be [group_id, construct_id] pairs")
            selected.append((identifier(item[0], "selected group_id"),
                             identifier(item[1], "selected construct_id")))
        selected = tuple(selected)
        if len(selected) != self.budget_n:
            raise ValueError("selected observation count differs from budget_n")
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate selected observation")
        order = [(_selection_key(self.budget_seed, g, c), g, c) for g, c in selected]
        if any(left >= right for left, right in zip(order, order[1:])):
            raise ValueError("selected observations are not in the seeded selection order")
        expected = _selection_digest(self.qualified_digest, self.contract_digest,
                                     self.budget_seed, self.budget_n, selected)
        if self.selection_digest != expected:
            raise ValueError("selection digest does not bind the seed, size, order and "
                             "qualified document")
        groups: Dict[str, list] = {}
        for g, c in selected:
            groups.setdefault(g, []).append(c)
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "_groups", {g: tuple(sorted(cs)) for g, cs in groups.items()})

    @property
    def selected_set(self) -> FrozenSet[Observation]:
        return frozenset(self.selected)

    def group_members(self) -> Dict[str, Tuple[str, ...]]:
        """Selected construct IDs per group (sorted), including singleton groups."""
        return dict(self._groups)

    def members_of(self, group_id: str) -> Tuple[str, ...]:
        return self._groups.get(group_id, ())

    def pair_groups(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Groups with two or more selected members, sorted by group ID."""
        return tuple((g, members) for g, members in sorted(self._groups.items())
                     if len(members) >= 2)


def select_functional_budget(qualified: QualifiedFunctional, n: int,
                             budget_seed: str) -> FunctionalBudget:
    """Select ``n`` train observations by seeded identity hash; no outcome is involved."""
    if not isinstance(qualified, QualifiedFunctional):
        raise TypeError("qualified must be a QualifiedFunctional")
    _budget_seed(budget_seed)
    observations = qualified.train_observations()
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if n > len(observations):
        raise ValueError(f"n={n} exceeds the {len(observations)} train observations")
    ordered = sorted(observations, key=lambda o: (_selection_key(budget_seed, *o), o))
    selected = tuple(ordered[:n])
    return FunctionalBudget(
        qualified_digest=qualified.digest, contract_digest=qualified.contract_digest,
        budget_seed=budget_seed, budget_n=n, selected=selected,
        selection_digest=_selection_digest(qualified.digest, qualified.contract_digest,
                                           budget_seed, n, selected))


def functional_budget_identity(budget: FunctionalBudget) -> Dict[str, Any]:
    """Plain-JSON identity; values are never part of it."""
    if not isinstance(budget, FunctionalBudget):
        raise TypeError("budget must be a FunctionalBudget")
    return {"schema": BUDGET_SCHEMA, "qualified_digest": budget.qualified_digest,
            "contract_digest": budget.contract_digest, "budget_seed": budget.budget_seed,
            "budget_n": budget.budget_n, "selected": [[g, c] for g, c in budget.selected],
            "selection_digest": budget.selection_digest}


def validate_functional_budget_identity(identity) -> FunctionalBudget:
    """Data-free validation of a recorded budget identity (no document, no labels).

    This proves the identity is well formed and self-consistent, not that it is the seeded
    prefix of any particular document; that needs the qualified document.
    """
    exact_keys(identity, ("schema", "qualified_digest", "contract_digest", "budget_seed",
                          "budget_n", "selected", "selection_digest"), "functional budget")
    if identity["schema"] != BUDGET_SCHEMA:
        raise ValueError("unsupported functional budget schema")
    if not isinstance(identity["selected"], list):
        raise ValueError("selected must be a JSON list")
    return FunctionalBudget(
        qualified_digest=identity["qualified_digest"], contract_digest=identity["contract_digest"],
        budget_seed=identity["budget_seed"], budget_n=identity["budget_n"],
        selected=tuple(identity["selected"]), selection_digest=identity["selection_digest"])


def _check_budget_matches(qualified: QualifiedFunctional, budget: FunctionalBudget) -> None:
    if not isinstance(qualified, QualifiedFunctional):
        raise TypeError("qualified must be a QualifiedFunctional")
    if not isinstance(budget, FunctionalBudget):
        raise TypeError("budget must be a FunctionalBudget")
    if (budget.qualified_digest != qualified.digest
            or budget.contract_digest != qualified.contract_digest):
        raise ValueError("budget was not selected from this qualified functional document")
    if not budget.selected_set <= set(qualified.train_observations()):
        raise ValueError("budget selects observations outside the qualified train observations")
    # Digests, membership and hash order all pass for a self-consistent non-prefix such as
    # selected[1:3]; only the full candidate list shows what was left out.
    if select_functional_budget(qualified, budget.budget_n, budget.budget_seed) != budget:
        raise ValueError("budget is not the seeded prefix of the qualified train observations")


# ---------------------------------------------------------------------------
# Outcome-free pair schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairDraw:
    """One scheduled comparison; it names identities only."""
    update: int
    index: int
    group_id: str
    left_id: str
    right_id: str

    @property
    def observations(self) -> Tuple[Observation, Observation]:
        return ((self.group_id, self.left_id), (self.group_id, self.right_id))


class FunctionalSchedule:
    """Uniform pair-eligible group, then two distinct selected members of that group.

    A draw is a pure function of ``(budget, seed, update, k)``.  It never sees a value and
    is never replaced, so ties and unresolved pairs still consume their opportunity.
    """

    def __init__(self, budget: FunctionalBudget) -> None:
        if not isinstance(budget, FunctionalBudget):
            raise TypeError("budget must be a FunctionalBudget")
        self._groups = budget.pair_groups()
        if not self._groups:
            raise ValueError("ordering unsupported: no group has two or more selected members")

    @property
    def groups(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        return self._groups

    def draw(self, seed: int, update: int, k: int) -> PairDraw:
        rng = random.Random(_keyed_seed(seed, update, k, PAIR_PURPOSE))
        group_id, members = self._groups[rng.randrange(len(self._groups))]
        count = len(members)
        left = rng.randrange(count)
        right = rng.randrange(count - 1)
        if right >= left:
            right += 1
        return PairDraw(update, k, group_id, members[left], members[right])


@dataclass(frozen=True)
class ScheduleReplay:
    """Data-free replay of the first ``updates`` (possibly zero) updates of the schedule."""
    seed: int
    pairs_per_update: int
    updates: int
    pair_opportunities: int
    exposures: Tuple[Tuple[Observation, int], ...]
    group_draws: Tuple[Tuple[str, int], ...]

    @property
    def compared(self) -> FrozenSet[Observation]:
        return frozenset(key for key, count in self.exposures if count)

    @property
    def compared_fraction(self) -> Fraction:
        return Fraction(len(self.compared), len(self.exposures))


def replay_schedule(budget: FunctionalBudget, seed: int, pairs_per_update: int,
                    updates: int) -> ScheduleReplay:
    """Stream the schedule into per-observation counters; nothing else is retained.

    ``updates=0`` is a valid horizon (nothing drawn, every exposure zero); the budget must
    still be schedulable.
    """
    _seed(seed)
    _positive(pairs_per_update, "pairs_per_update")
    _non_negative(updates, "updates")
    schedule = FunctionalSchedule(budget)
    exposures = dict.fromkeys(budget.selected, 0)
    group_draws = dict.fromkeys((g for g, _ in schedule.groups), 0)
    for update in range(updates):
        for k in range(pairs_per_update):
            draw = schedule.draw(seed, update, k)
            exposures[(draw.group_id, draw.left_id)] += 1
            exposures[(draw.group_id, draw.right_id)] += 1
            group_draws[draw.group_id] += 1
    return ScheduleReplay(seed=seed, pairs_per_update=pairs_per_update, updates=updates,
                          pair_opportunities=updates * pairs_per_update,
                          exposures=tuple(sorted(exposures.items())),
                          group_draws=tuple(sorted(group_draws.items())))


def scheduled_coverage(budget: FunctionalBudget, seed: int, pairs_per_update: int,
                       total_updates: int) -> Fraction:
    """Exact fraction of ALL selected observations compared at least once over the horizon.

    Singleton-group observations can never be compared, so they stay in the denominator.
    """
    return replay_schedule(budget, seed, pairs_per_update, total_updates).compared_fraction


def coverage_requirement_met(coverage: Fraction, minimum) -> bool:
    """Exact check of a separately declared coverage requirement.

    ``minimum`` is a decimal threshold in [0, 1] (``0.9`` means 9/10, not the nearest
    binary float), so a coverage of exactly 9/10 meets it.  This is a requirement that
    the caller declares; it is never replaced by requiring every selected observation to
    be observed.
    """
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise ValueError("minimum coverage must be a number")
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("minimum coverage must be between zero and one")
    if not isinstance(coverage, Fraction):
        raise TypeError("coverage must be an exact Fraction")
    return coverage >= Fraction(repr(float(minimum)))


def functional_exposure_plan(qualified: QualifiedFunctional, budget: FunctionalBudget, *,
                             seed: Optional[int] = None, pairs_per_update: Optional[int] = None,
                             total_updates: Optional[int] = None) -> Dict[str, Any]:
    """Outcome-free plan: structure of the selection and, optionally, horizon exposure.

    Reports observations AND unique constructs/antibodies/antigens, singleton groups
    (never comparable) and, when a schedule is given, every unseen observation.  Pair
    counts are arithmetic (``m choose 2`` per group), never enumerated.
    """
    _check_budget_matches(qualified, budget)
    given = [value is not None for value in (seed, pairs_per_update, total_updates)]
    if any(given) and not all(given):
        raise ValueError("seed, pairs_per_update and total_updates must be given together")
    eligible = budget.pair_groups()
    if not eligible:
        raise ValueError("ordering unsupported: no group has two or more selected members")
    members = budget.group_members()
    construct_ids = [c for _, c in budget.selected]
    constructs = {c: qualified.construct(c) for c in set(construct_ids)}
    pairs = {g: len(m) * (len(m) - 1) // 2 for g, m in members.items()}
    plan: Dict[str, Any] = {
        "schema": EXPOSURE_SCHEMA, "qualified_digest": budget.qualified_digest,
        "contract_digest": budget.contract_digest, "selection_digest": budget.selection_digest,
        "budget_seed": budget.budget_seed, "budget_n": budget.budget_n,
        "selected_observations": len(budget.selected),
        "unique_selected_construct_ids": len(constructs),
        "unique_physical_constructs": len({c.content_key for c in constructs.values()}),
        "unique_antibodies": len({(c.H, c.L) for c in constructs.values()}),
        "unique_antigens": len({c.Ag for c in constructs.values()}),
        "repeated_construct_ids": sorted(c for c, n in Counter(construct_ids).items() if n > 1),
        "groups": {g: {"selected_members": len(m), "pair_eligible": len(m) >= 2,
                       "possible_pairs": pairs[g]} for g, m in sorted(members.items())},
        "pair_eligible_group_ids": [g for g, _ in eligible],
        "singleton_group_ids": sorted(g for g, m in members.items() if len(m) == 1),
        "pair_eligible_observations": sum(len(m) for _, m in eligible),
        "pair_ineligible_observations": sum(len(m) for m in members.values()
                                            if len(m) == 1),
        "possible_pair_count": sum(pairs.values()),
        "scope": "outcome-free; identities only",
        "measurement_limit": "exposure counts comparisons, not informative labels or nonzero "
                             "gradients; outcomes are not read",
    }
    if all(given):
        replay = replay_schedule(budget, seed, pairs_per_update, total_updates)
        counts = [count for _, count in replay.exposures]
        fraction = replay.compared_fraction
        plan["schedule"] = {
            "rng": RNG_ALGORITHM, "purpose": PAIR_PURPOSE, "seed": seed,
            "pairs_per_update": pairs_per_update, "total_updates": total_updates,
            "pair_opportunities": replay.pair_opportunities,
            "per_observation_counts": [[g, c, n] for (g, c), n in replay.exposures],
            "seen_observation_count": len(replay.compared),
            "unseen_observation_count": counts.count(0),
            "unseen_observations": [[g, c] for (g, c), n in replay.exposures if n == 0],
            "compared_fraction": {"numerator": fraction.numerator,
                                  "denominator": fraction.denominator, "value": float(fraction)},
            "minimum_count_all_selected": min(counts), "maximum_count_all_selected": max(counts),
            "group_draws": {g: n for g, n in replay.group_draws},
        }
    return plan


# ---------------------------------------------------------------------------
# Restricted measurements (label-dependent)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairResolution:
    draw: PairDraw
    outcome: str                      # "ordered" | "tie" | "unresolved"
    better_id: Optional[str] = None
    worse_id: Optional[str] = None

    @property
    def informative(self) -> bool:
        return self.outcome == "ordered"


@dataclass(frozen=True)
class OutcomeReplay:
    """Label-dependent replay: the schedule plus how each scheduled pair resolved.

    ``ordered_updates`` counts outer updates containing at least one ordered pair,
    whether or not any hinge was active.  A head optimizer step is taken once per such
    update, so head steps equal ``ordered_updates``, never ``ordered_pairs`` or active pairs.
    """
    seed: int
    pairs_per_update: int
    updates: int
    pair_opportunities: int
    ordered_pairs: int
    tie_pairs: int
    unresolved_pairs: int
    ordered_updates: int
    compared: FrozenSet[Observation]
    ordered_observations: FrozenSet[Observation]


@dataclass(frozen=True, eq=False)
class RestrictedFunctionalData:
    """Exactly the selected observations' measurements, bound to one fixed budget.

    Only already-authorized callers (training, resume) hold one.  Inference, export,
    parent loading and the study validators never receive it.

    ``directions`` and ``values`` are read-only snapshots (copied at construction, then
    exposed as mapping proxies), so neither the caller's original dicts nor the public
    fields can change supervision after the identity was captured.  Mapping proxies do not
    deep-copy; rebuild from the measurements document instead.  Direct construction is
    allowed, so a holder that did not come from ``from_document`` should call
    :meth:`verify_against` before trusting the budget or the directions.
    """
    budget: FunctionalBudget
    directions: Mapping[str, str]
    values: Mapping[Observation, Measurement]

    def __post_init__(self) -> None:
        if not isinstance(self.budget, FunctionalBudget):
            raise TypeError("budget must be a FunctionalBudget")
        directions, values = dict(self.directions), dict(self.values)
        if set(directions) != set(self.budget.group_members()) or any(
                d not in DIRECTIONS for d in directions.values()):
            raise ValueError("directions must cover exactly the selected groups")
        if set(values) != set(self.budget.selected) or any(
                not isinstance(v, Measurement) for v in values.values()):
            raise ValueError("measurements must cover exactly the selected observations")
        object.__setattr__(self, "directions", MappingProxyType(directions))
        object.__setattr__(self, "values", MappingProxyType(values))

    def verify_against(self, qualified: QualifiedFunctional) -> None:
        """Bind this object to the qualified document it claims to come from.

        Requires the budget to be the exact seeded prefix of ``qualified`` and every
        group's direction to be the qualified endpoint direction.  Label values are not
        checked here (the qualified document holds none); qualification is not repeated.
        """
        _check_budget_matches(qualified, self.budget)
        expected = {g: qualified.direction(g) for g in self.budget.group_members()}
        if dict(self.directions) != expected:
            raise ValueError("directions differ from the qualified endpoint directions")

    @classmethod
    def from_document(cls, document, qualified: QualifiedFunctional,
                      budget: FunctionalBudget) -> "RestrictedFunctionalData":
        _check_budget_matches(qualified, budget)
        exact_keys(document, ("schema", "selection_digest", "measurements"),
                   "functional measurements")
        if document["schema"] != MEASUREMENTS_SCHEMA:
            raise ValueError("unsupported functional measurements schema")
        # Compared before any row is interpreted.
        if document["selection_digest"] != budget.selection_digest:
            raise ValueError("measurement selection digest does not match the fixed budget")
        rows = document["measurements"]
        if not isinstance(rows, list):
            raise ValueError("measurements must be a JSON list")
        selected = budget.selected_set
        values: Dict[Observation, Measurement] = {}
        for row in rows:
            exact_keys(row, ("group_id", "construct_id", "measurement"), "measurement row")
            key = (identifier(row["group_id"], "measurement group_id"),
                   identifier(row["construct_id"], "measurement construct_id"))
            if key in values:
                raise ValueError(f"duplicate measurement for observation {key!r}")
            if key not in selected:
                raise ValueError(f"measurement for observation {key!r} outside the fixed budget")
            values[key] = Measurement.from_document(row["measurement"])
        missing = sorted(selected - set(values))
        if missing:
            raise ValueError(f"missing measurements for selected observations: {missing}")
        directions = {g: qualified.direction(g) for g in budget.group_members()}
        return cls(budget=budget, directions=directions, values=values)

    def measurement(self, group_id: str, construct_id: str) -> Measurement:
        return self.values[(group_id, construct_id)]

    def data_digest(self) -> str:
        return json_digest({"schema": MEASUREMENTS_SCHEMA,
                            "selection_digest": self.budget.selection_digest,
                            "directions": sorted(self.directions.items()),
                            "measurements": [[g, c, m.document()]
                                             for (g, c), m in sorted(self.values.items())]})

    def classify(self, draw: PairDraw) -> PairResolution:
        if not isinstance(draw, PairDraw):
            raise TypeError("draw must be a PairDraw")
        # ``values`` holds exactly the selected observations, so membership is a hash lookup.
        left, right = (draw.group_id, draw.left_id), (draw.group_id, draw.right_id)
        if draw.left_id == draw.right_id or left not in self.values or right not in self.values:
            raise ValueError("draw names observations outside the fixed budget")
        result = compare(self.values[left], self.values[right], self.directions[draw.group_id])
        if result == A_BETTER:
            return PairResolution(draw, "ordered", draw.left_id, draw.right_id)
        if result == B_BETTER:
            return PairResolution(draw, "ordered", draw.right_id, draw.left_id)
        return PairResolution(draw, "tie" if result == TIE else "unresolved")

    def replay_outcomes(self, seed: int, pairs_per_update: int, updates: int) -> OutcomeReplay:
        """Replay the first ``updates`` updates with labels, streaming counters and sets.

        ``updates=0`` replays nothing: zero counters and empty sets.
        """
        _seed(seed)
        _positive(pairs_per_update, "pairs_per_update")
        _non_negative(updates, "updates")
        schedule = FunctionalSchedule(self.budget)
        counts = Counter()
        compared, ordered_seen = set(), set()
        ordered_updates = 0
        for update in range(updates):
            ordered_in_update = False
            for k in range(pairs_per_update):
                resolution = self.classify(schedule.draw(seed, update, k))
                counts[resolution.outcome] += 1
                compared.update(resolution.draw.observations)
                if resolution.informative:
                    ordered_in_update = True
                    ordered_seen.update(resolution.draw.observations)
            ordered_updates += ordered_in_update
        return OutcomeReplay(seed=seed, pairs_per_update=pairs_per_update, updates=updates,
                             pair_opportunities=updates * pairs_per_update,
                             ordered_pairs=counts["ordered"], tie_pairs=counts["tie"],
                             unresolved_pairs=counts["unresolved"],
                             ordered_updates=ordered_updates,
                             compared=frozenset(compared),
                             ordered_observations=frozenset(ordered_seen))


def load_restricted_measurements(path, qualified: QualifiedFunctional,
                                 budget: FunctionalBudget) -> RestrictedFunctionalData:
    """Open the measurements file for a budget that is already fixed.

    Qualification and selection are complete before the file is touched (both are
    arguments).  See the module docstring for what the digest check does and does not
    guarantee about the JSON decoder.
    """
    _check_budget_matches(qualified, budget)
    return RestrictedFunctionalData.from_document(read_strict_json(path), qualified, budget)


# ---------------------------------------------------------------------------
# Counter invariants
# ---------------------------------------------------------------------------

def _observation_set(value, where) -> FrozenSet[Observation]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{where} must be a list of [group_id, construct_id] pairs")
    result = set()
    for item in value:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or any(not isinstance(part, str) for part in item)):
            raise ValueError(f"{where} entries must be [group_id, construct_id] pairs")
        result.add((item[0], item[1]))
    return frozenset(result)


def check_seen_within_budget(seen: Iterable, budget: FunctionalBudget) -> FrozenSet[Observation]:
    """Seen observations must be a SUBSET of the budget; equality is never required.

    A completed run need not have observed every available fitting observation.  Coverage
    is a separately declared requirement (``scheduled_coverage``), not an equality.
    """
    observed = _observation_set(list(seen), "seen observations")
    outside = sorted(observed - budget.selected_set)
    if outside:
        raise ValueError(f"seen observations outside the fixed budget: {outside}")
    return observed


def check_counter_invariants(counters, seen, replay, pairs_per_update: int, updates: int) -> None:
    """Verify recorded counters against the replayed schedule.

    Data-free part (a :class:`ScheduleReplay` suffices): without redraw
    ``pair_opportunities == updates * pairs_per_update`` exactly, and ``seen["compared"]``
    equals the replayed compared set.  Outcome-dependent counters (``ordered_pairs``,
    ``tie_pairs``, ``unresolved_pairs``, optional ``active_pairs`` and ``ordered_updates``;
    ``seen["ordered"]``) are held only to partition and bound checks, unless the replay is
    an :class:`OutcomeReplay`, which additionally pins them exactly (``active_pairs`` is
    optimizer-side and is never pinned).

    Bounds: every ordered pair names two distinct observations, so a positive
    ``ordered_pairs`` needs at least two distinct ordered observations and allows at most
    ``2 * ordered_pairs``; an available observation that no ordered pair reached is
    legitimately absent.  When ``ordered_updates`` is given it must lie between
    ``ceil(ordered_pairs / pairs_per_update)`` and ``min(ordered_pairs, updates)``.
    ``updates == 0`` is a valid horizon with every counter zero.
    """
    _positive(pairs_per_update, "pairs_per_update")
    _non_negative(updates, "updates")
    if not isinstance(counters, dict) or not isinstance(seen, dict):
        raise ValueError("counters and seen must be objects")
    if not isinstance(replay, (ScheduleReplay, OutcomeReplay)):
        raise TypeError("replay must be a ScheduleReplay or OutcomeReplay")
    opportunities = pairs_per_update * updates
    if (replay.pairs_per_update, replay.updates, replay.pair_opportunities) != (
            pairs_per_update, updates, opportunities):
        raise ValueError("replay was computed for a different horizon")
    recorded = counters.get("pair_opportunities")
    if type(recorded) is not int or recorded != opportunities:
        raise ValueError("pair_opportunities must equal updates x pairs_per_update "
                         "(no redraw)")
    if "compared" not in seen:
        raise ValueError("seen.compared is required")
    if _observation_set(seen["compared"], "seen.compared") != replay.compared:
        raise ValueError("compared observations differ from the replayed outcome-free schedule")
    names = ("ordered_pairs", "tie_pairs", "unresolved_pairs")
    present = [name in counters for name in names]
    if not any(present):
        if isinstance(replay, OutcomeReplay):
            raise ValueError("outcome counters are required when replaying outcomes")
        if "ordered" in seen or "active_pairs" in counters or "ordered_updates" in counters:
            raise ValueError("ordered/active records need the ordered, tie and unresolved counters")
        return
    if not all(present):
        raise ValueError("ordered_pairs, tie_pairs and unresolved_pairs must be given together")
    values = [counters[name] for name in names]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("outcome counters must be non-negative integers")
    if sum(values) != opportunities:
        raise ValueError("ordered + tie + unresolved pairs must partition the pair opportunities")
    ordered_pairs = counters["ordered_pairs"]
    active = counters.get("active_pairs", 0)
    if type(active) is not int or not 0 <= active <= ordered_pairs:
        raise ValueError("active_pairs must be an integer between zero and ordered_pairs")
    ordered_seen = _observation_set(seen.get("ordered", []), "seen.ordered")
    compared = replay.compared
    if not ordered_seen <= compared:
        raise ValueError("ordered observations must be a subset of the compared observations")
    if len(ordered_seen) > 2 * ordered_pairs or (ordered_pairs == 0 and ordered_seen):
        raise ValueError("ordered observations exceed what the ordered pair count allows")
    if ordered_pairs > 0 and len(ordered_seen) < 2:
        raise ValueError("an ordered pair needs two distinct ordered observations")
    has_updates = "ordered_updates" in counters
    if has_updates:
        recorded_updates = counters["ordered_updates"]
        fewest = -(-ordered_pairs // pairs_per_update)       # ceil: one update holds P pairs
        if type(recorded_updates) is not int or not fewest <= recorded_updates <= min(
                ordered_pairs, updates):
            raise ValueError("ordered_updates must be an integer between ceil(ordered_pairs / "
                             "pairs_per_update) and min(ordered_pairs, updates)")
    if isinstance(replay, OutcomeReplay):
        if (values != [replay.ordered_pairs, replay.tie_pairs, replay.unresolved_pairs]
                or ordered_seen != replay.ordered_observations
                or (has_updates and recorded_updates != replay.ordered_updates)):
            raise ValueError("outcome counters differ from the label-dependent replay")
