"""Visible-only masked diagnostics: an immutable plan, an exact forward budget, then no-grad scoring.

Planning is pure Python (hashlib, heapq, the sibling ``records``/``config`` helpers and the existing
``her2_multiscale.masks`` canonical-region helper); it never imports torch. Execution imports torch and the
``denoising`` scorers inside the functions that run a backend, so admission and ``--help`` stay torch-free.

What is selected, and how:

* Every priority, mask position and reveal order is a SHA-256 of CONTENT (a seed, role-framed sequences, a
  canonical region, a masked visible view). Python ``hash``, record ids, source/assay names, measured values
  and row indices never enter. Metadata only decides eligibility (partition, same source/assay) and how
  results are grouped in a report.
* Singleton tasks are unique ``(H, L, Ag)`` content x canonical region. The same content measured in several
  assays is ONE task that lists all its record ids. ``max_examples`` keeps the globally lowest content
  priorities.
* Pair tasks are unique canonically oriented ``(endpoint A, endpoint B)`` content x canonical region at
  Hamming distance 1..3, every change inside the declared region, found by screening each
  ``(partition, source, assay, antigen)`` group. The screen is QUADRATIC within a group: this is bounded pilot
  qualification, not a scalable training pipeline. Candidates are kept in a bounded max-heap while the whole
  scan completes (not the first ones met), so no list of all pairs is ever built. One task gathers every
  same-source/same-assay contrast of its two contents; a contrast across assays or sources is never built.
* The forward budget is exact and computed before any backend exists: see :func:`singleton_cell_cost` and
  :func:`pair_task_cost`. :class:`CountingBackend` refuses an overrun and confirms the planned count on success.

A scalar probe may use full unmasked features. Nothing here does: a masked scorer sees only a
:class:`VisibleView` and never a pooled-feature memo, a head or a measured value.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from smallAntibodyGen.experiments.her2_multiscale.masks import MODE_FULL, MODE_PARTIAL, MaskDraw, canonical_region
from smallAntibodyGen.experiments.interaction_qualification.config import (
    FULL_MASK,
    GEOMETRIES,
    PARTITIONS,
    DiagnosticsConfig,
)
from smallAntibodyGen.experiments.interaction_qualification.records import Record
from smallAntibodyGen.models.antibody_antigen.views import ROLES, Coord, MaskedTargets, VisibleView, mask_complete

if TYPE_CHECKING:  # torch is imported at run time only inside the functions that run a backend
    from smallAntibodyGen.models.antibody_antigen.denoising import ResidueBackend

DIAGNOSTICS_SCHEMA = "aa-diagnostics/1"
FULL_GEOMETRY = "full"
MAX_DISTANCE = 3
ARMS: Tuple[str, ...] = ("joint", "antibody_only")
ORDER_NAMES: Tuple[str, ...] = ("forward", "reverse")
PHASE_FEATURES = "features"
PHASE_DIAGNOSTICS = "diagnostics"
_GEOMETRY_RANK = {"suffix": 0, "subset": 1, FULL_GEOMETRY: 2}

INDEPENDENCE_NOTE = (
    "several mask counts, geometries and reveal orders of one molecule are repeated views of it, not independent "
    "molecules; each (partition, count, geometry, arm) group is reported separately and no statistic pools "
    "counts as if they were independent"
)
NATIVE_CONFOUND_NOTE = (
    "the native model's eval-time token-dropout rescaling depends on the observed mask ratio, which changes when "
    "unmasked antigen residues are added to the visible view; a joint-minus-antibody-only difference therefore "
    "mixes antigen information with that rescaling. The observed mask ratios and their ratio are reported and the "
    "native behaviour is left unchanged. No antigen-specificity claim is made"
)
PAIR_NOTE = (
    "each gap is a sum over the changed sites (not divided by the distance) of the sequentially revealed log "
    "probability of endpoint B minus endpoint A; target = kappa * (pKD_B - pKD_A) for the measured contrast. "
    "Two reveal orders are compared; nothing here identifies epistasis"
)


class BudgetExceeded(ValueError):
    """The planned forward count is larger than ``max_forwards``; raised before any backend exists."""


class BudgetOverrun(RuntimeError):
    """A backend call outside a budgeted phase, or beyond the planned count of its phase."""


class BudgetMismatch(RuntimeError):
    """The calls actually made differ from the plan."""


# --- small deterministic helpers ----------------------------------------------------------------


def _digest(tag: str, *parts: str) -> str:
    return hashlib.sha256("|".join((tag,) + parts).encode("ascii")).hexdigest()


def coord_text(coord: Coord) -> str:
    return f"{coord[0]}:{coord[1]}"


def coords_text(coords: Iterable[Coord]) -> List[str]:
    return [coord_text(coord) for coord in coords]


def region_text(region: Sequence[Coord]) -> str:
    return ",".join(coord_text(coord) for coord in region)


def _coord_key(coord: Coord) -> Tuple[int, int]:
    return (ROLES.index(coord[0]), coord[1])


def chain_sequences(heavy: str, light: Optional[str], antigen: str) -> Dict[str, str]:
    sequences = {"H": heavy, "Ag": antigen}
    if light is not None:
        sequences["L"] = light
    return sequences


def order_count(sites: int) -> int:
    """Reveal orders scored for ``sites`` masked sites: one for a single site, else forward and literal reverse."""
    return 1 if sites == 1 else 2


def singleton_cell_cost(count: int) -> int:
    """Forwards for one mask case, BOTH arms: per arm one masked NLL, ``q`` teacher-forced sequences of ``k``
    forwards each, and one greedy completion of ``k`` forwards -- ``2 * (1 + (q + 1) * k)``."""
    return 2 * (1 + (order_count(count) + 1) * count)


def pair_task_cost(distance: int) -> int:
    """Forwards for one pair task, both arms and all orders: ``4 * q * d`` (each gap scores two endpoints)."""
    return 4 * order_count(distance) * distance


# --- fixed-count masks and reveal orders ---------------------------------------------------------


def fixed_count_mask(
    region: Sequence[Coord],
    count: int,
    geometry: str,
    *,
    seed: int,
    content_key: str,
    heavy_length: int,
    light_length: int,
) -> MaskDraw:
    """Hide exactly ``count`` sites of ``region`` (canonical order): the whole region, the suffix, or a subset.

    ``count == len(region)`` is the full mask whatever geometry was asked for. A partial ``"suffix"`` is the
    last ``count`` sites in canonical coordinates; a partial ``"subset"`` is a uniform draw without replacement
    that takes the ``count`` positions with the lowest SHA-256 of (seed, content key, count, position), so it
    depends on content and seed alone. The existing ``draw_mask`` is not used: it chooses ``k`` itself.
    """
    sites = canonical_region(region, heavy_length=heavy_length, light_length=light_length)
    size = len(sites)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= size:
        raise ValueError(f"mask count must be an integer in 1..{size}, got {count!r}")
    if count == size:
        return MaskDraw(sites=sites, count=size, size=size, mode=MODE_FULL, geometry=FULL_GEOMETRY)
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    if geometry == "suffix":
        chosen = sites[size - count:]
    else:
        ranked = sorted(range(size), key=lambda i: (_digest("subset", str(seed), content_key, str(count), str(i)), i))
        chosen = tuple(sites[i] for i in sorted(ranked[:count]))
    return MaskDraw(sites=chosen, count=count, size=size, mode=MODE_PARTIAL, geometry=geometry)


def explicit_order(sites: Sequence[Coord], view_digest: str, seed: int, tag: str = "order") -> Tuple[Coord, ...]:
    """A reveal order from the seed, the shared masked VISIBLE content and the coordinate alone."""
    return tuple(
        sorted(sites, key=lambda c: (_digest(tag, str(seed), view_digest, coord_text(c)), _coord_key(c)))
    )


def feasible_cases(
    size: int, mask_counts: Sequence[Any], geometries: Sequence[str]
) -> Tuple[List[Tuple[int, str]], List[int]]:
    """``(cases, skipped)`` for a region of ``size`` sites.

    ``"full"`` maps to ``size``; an integer above ``size`` is skipped (never truncated to the region) and
    returned in ``skipped``; counts are de-duplicated. The full mask is ONE case under geometry ``"full"``
    however many geometries were requested; each partial count is one case per requested geometry.
    """
    counts = set()
    skipped = set()
    for entry in mask_counts:
        count = size if entry == FULL_MASK else entry
        if count > size:
            skipped.add(count)
        else:
            counts.add(count)
    geometry_list = [g for g in GEOMETRIES if g in geometries]
    cases: List[Tuple[int, str]] = []
    for count in sorted(counts):
        if count == size:
            cases.append((count, FULL_GEOMETRY))
        else:
            cases.extend((count, geometry) for geometry in geometry_list)
    return cases, sorted(skipped)


# --- plan data ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskCase:
    count: int
    geometry: str
    mode: str
    sites: Tuple[Coord, ...]  # canonical order
    order: Tuple[Coord, ...]  # the explicit forward reveal order, shared by both arms
    reverse_order: Optional[Tuple[Coord, ...]]  # the literal reverse; None for a single site
    view_digest: str  # the joint masked visible view (no truth in it)
    cost: int


@dataclass(frozen=True)
class SingletonTask:
    task_id: str
    priority: str
    pair_fingerprint: str
    partition: str
    heavy: str
    light: Optional[str]
    antigen: str
    region: Tuple[Coord, ...]
    record_ids: Tuple[str, ...]  # audit only
    sources_assays: Tuple[Tuple[str, str], ...]  # audit only
    skipped_counts: Tuple[int, ...]
    cases: Tuple[MaskCase, ...]

    @property
    def cost(self) -> int:
        return sum(case.cost for case in self.cases)


@dataclass(frozen=True)
class Endpoint:
    heavy: str
    light: Optional[str]


@dataclass(frozen=True)
class Contrast:
    record_id_a: str
    record_id_b: str
    source: str
    assay: str


@dataclass(frozen=True)
class PairTask:
    task_id: str
    priority: str
    partition: str
    antigen: str
    region: Tuple[Coord, ...]
    a: Endpoint
    b: Endpoint
    a_fingerprint: str
    b_fingerprint: str
    changed: Tuple[Coord, ...]
    order: Tuple[Coord, ...]
    reverse_order: Optional[Tuple[Coord, ...]]
    view_digest: str
    cost: int
    contrasts: Tuple[Contrast, ...]  # record ids only; measured values are joined at report time

    @property
    def distance(self) -> int:
        return len(self.changed)


@dataclass(frozen=True)
class ForwardBudget:
    feature: int
    singleton: int
    pair: int
    max_forwards: int

    @property
    def diagnostics(self) -> int:
        return self.singleton + self.pair

    @property
    def total(self) -> int:
        return self.feature + self.singleton + self.pair

    @property
    def within_budget(self) -> bool:
        return self.total <= self.max_forwards

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_forwards": self.feature,
            "singleton_forwards": self.singleton,
            "pair_forwards": self.pair,
            "diagnostic_forwards": self.diagnostics,
            "total_forwards": self.total,
            "max_forwards": self.max_forwards,
            "within_budget": self.within_budget,
            "formula": {
                "feature": "distinct unmasked joint / antibody-only / antigen-only views of the active eligible records",
                "singleton_cell": "2 * (1 + (q + 1) * k), q = 1 if k == 1 else 2",
                "pair_task": "4 * q * d, q = 1 if d == 1 else 2",
                "cache": "none: no diagnostic cache and no uncounted native call",
            },
        }


def require_within_budget(budget: ForwardBudget) -> None:
    if not budget.within_budget:
        raise BudgetExceeded(
            f"planned {budget.total} backend forwards (features {budget.feature}, singleton {budget.singleton}, "
            f"pair {budget.pair}) exceed max_forwards {budget.max_forwards}"
        )


@dataclass(frozen=True)
class DiagnosticPlan:
    seed: int
    kappa: float
    partitions: Tuple[str, ...]
    singleton_tasks: Tuple[SingletonTask, ...]
    pair_tasks: Tuple[PairTask, ...]
    budget: ForwardBudget
    singleton_stats: Mapping[str, Any]
    pair_stats: Mapping[str, Any]

    @property
    def n_cells(self) -> int:
        return sum(len(task.cases) for task in self.singleton_tasks)


# --- singleton planning -----------------------------------------------------------------------------


def _record_region(record: Record) -> Tuple[Coord, ...]:
    assert record.region is not None
    return canonical_region(
        record.region, heavy_length=len(record.heavy), light_length=0 if record.light is None else len(record.light)
    )


def _build_case(task_key: str, record: Record, region: Tuple[Coord, ...], count: int, geometry: str, seed: int) -> MaskCase:
    draw = fixed_count_mask(
        region,
        count,
        geometry,
        seed=seed,
        content_key=task_key,
        heavy_length=len(record.heavy),
        light_length=0 if record.light is None else len(record.light),
    )
    view, _ = mask_complete(chain_sequences(record.heavy, record.light, record.antigen), draw.sites)
    digest = view.content_digest()
    order = explicit_order(draw.sites, digest, seed)
    reverse = None if len(order) == 1 else tuple(reversed(order))
    return MaskCase(
        count=draw.count,
        geometry=draw.geometry,
        mode=draw.mode,
        sites=draw.sites,
        order=order,
        reverse_order=reverse,
        view_digest=digest,
        cost=singleton_cell_cost(draw.count),
    )


def plan_singletons(records: Sequence[Record], config: DiagnosticsConfig) -> Tuple[Tuple[SingletonTask, ...], Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Record]] = {}
    missing_region = 0
    for record in records:
        if record.region is None:
            missing_region += 1
            continue
        groups.setdefault((record.pair_fingerprint, region_text(_record_region(record))), []).append(record)
    candidates: List[Tuple[str, str, str]] = []
    without_cases = 0
    for (pair_fp, rtext), members in groups.items():
        cases, _ = feasible_cases(len(_record_region(members[0])), config.mask_counts, config.geometries)
        if not cases:
            without_cases += 1
            continue
        candidates.append((_digest("prio-singleton", str(config.seed), pair_fp, rtext), pair_fp, rtext))
    chosen = heapq.nsmallest(config.max_examples, candidates)
    tasks: List[SingletonTask] = []
    for priority, pair_fp, rtext in chosen:
        members = sorted(groups[(pair_fp, rtext)], key=lambda r: (r.source, r.assay, r.record_id))
        first = members[0]
        if len({r.partition for r in members}) != 1:
            raise ValueError("one sequence content appears in several partitions; the split audit should prevent this")
        region = _record_region(first)
        task_id = _digest("task1", pair_fp, rtext)
        specs, skipped = feasible_cases(len(region), config.mask_counts, config.geometries)
        cases = tuple(_build_case(task_id, first, region, count, geometry, config.seed) for count, geometry in specs)
        tasks.append(
            SingletonTask(
                task_id=task_id,
                priority=priority,
                pair_fingerprint=pair_fp,
                partition=first.partition,
                heavy=first.heavy,
                light=first.light,
                antigen=first.antigen,
                region=region,
                record_ids=tuple(r.record_id for r in members),
                sources_assays=tuple(sorted({(r.source, r.assay) for r in members})),
                skipped_counts=tuple(skipped),
                cases=cases,
            )
        )
    stats = {
        "records_considered": len(records),
        "records_missing_region": missing_region,
        "content_tasks_with_region": len(groups),
        "content_tasks_without_feasible_case": without_cases,
        "content_tasks_selected": len(tasks),
        "max_examples": config.max_examples,
        "cap_binding": len(candidates) > config.max_examples,
        "cells": sum(len(task.cases) for task in tasks),
        "skipped_count_entries": sum(len(task.skipped_counts) for task in tasks),
        "selection": "globally lowest SHA-256 content priority; labels, ids and metadata are not inputs",
    }
    return tuple(tasks), stats


# --- pair planning ---------------------------------------------------------------------------------


@dataclass
class _PairCandidate:
    lo: Record
    hi: Record
    region: Tuple[Coord, ...]
    changed: Tuple[Coord, ...]


class _Worst:
    """Heap item ordered so ``heapq`` keeps the LARGEST sort key at the root (a bounded max-heap)."""

    __slots__ = ("key",)

    def __init__(self, key: Tuple[str, str, str, str]) -> None:
        self.key = key

    def __lt__(self, other: "_Worst") -> bool:
        return self.key > other.key


def _content_key(record: Record) -> Tuple[str, str, str]:
    return (record.heavy, "" if record.light is None else record.light, record.antigen)


def _compare(
    a: Record, b: Record, region_a: Tuple[Coord, ...], region_b: Tuple[Coord, ...]
) -> Tuple[Optional[Tuple[Coord, ...]], Optional[str]]:
    if region_a != region_b:
        return None, "region_mismatch"
    if (
        (a.light is None) != (b.light is None)
        or len(a.heavy) != len(b.heavy)
        or (a.light is not None and b.light is not None and len(a.light) != len(b.light))
    ):
        return None, "chain_or_length_mismatch"
    changed: List[Coord] = []
    for role, left, right in (("H", a.heavy, b.heavy), ("L", a.light, b.light)):
        if left is None or right is None:
            continue
        for index, (x, y) in enumerate(zip(left, right)):
            if x != y:
                changed.append((role, index))
                if len(changed) > MAX_DISTANCE:
                    return None, "distance_above_3"
    if not changed:
        return None, "identical_sequences"
    allowed = set(region_a)
    if any(coord not in allowed for coord in changed):
        return None, "edits_outside_region"
    return tuple(changed), None


def plan_pairs(records: Sequence[Record], config: DiagnosticsConfig) -> Tuple[Tuple[PairTask, ...], Dict[str, Any]]:
    gap = config.local_gap
    note = "quadratic within (partition, source, assay, antigen) groups: bounded pilot qualification, not scalable"
    if not gap.enabled:
        return (), {"enabled": False, "selected_tasks": 0, "max_pairs": gap.max_pairs, "screening": note}
    groups: Dict[Tuple[int, str, str, str], List[Record]] = {}
    content_index: Dict[Tuple[str, str], List[Record]] = {}
    for record in records:
        if record.region is None:
            continue
        rtext = region_text(_record_region(record))
        groups.setdefault(
            (PARTITIONS.index(record.partition), record.source, record.assay, record.antigen_fingerprint), []
        ).append(record)
        content_index.setdefault((record.pair_fingerprint, rtext), []).append(record)
    selected: Dict[Tuple[str, str, str], _PairCandidate] = {}
    heap: List[_Worst] = []
    screened = 0
    eligible_contrasts = 0
    cap_binding = False
    refused: Dict[str, int] = {}
    for group_key in sorted(groups):
        members = sorted(groups[group_key], key=lambda r: r.pair_fingerprint)
        regions = [_record_region(r) for r in members]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                screened += 1
                changed, refusal = _compare(members[i], members[j], regions[i], regions[j])
                if refusal is not None:
                    refused[refusal] = refused.get(refusal, 0) + 1
                    continue
                assert changed is not None
                eligible_contrasts += 1
                lo, hi = sorted((members[i], members[j]), key=_content_key)
                rtext = region_text(regions[i])
                key = (lo.pair_fingerprint, hi.pair_fingerprint, rtext)
                if key in selected:
                    continue
                sort_key = (_digest("prio-pair", str(config.seed), *key),) + key
                if len(selected) < gap.max_pairs:
                    heapq.heappush(heap, _Worst(sort_key))
                elif sort_key < heap[0].key:
                    evicted = heapq.heapreplace(heap, _Worst(sort_key))
                    del selected[evicted.key[1:]]
                    cap_binding = True
                else:
                    cap_binding = True
                    continue
                selected[key] = _PairCandidate(lo=lo, hi=hi, region=regions[i], changed=changed)
    tasks: List[PairTask] = []
    for key in sorted(selected, key=lambda k: (_digest("prio-pair", str(config.seed), *k),) + k):
        candidate = selected[key]
        tasks.append(_build_pair_task(key, candidate, content_index, config.seed))
    stats = {
        "enabled": True,
        "groups_screened": len(groups),
        "screened_pairs": screened,
        "refused": dict(sorted(refused.items())),
        "eligible_contrasts_scanned": eligible_contrasts,
        "selected_tasks": len(tasks),
        "selected_contrasts": sum(len(task.contrasts) for task in tasks),
        "max_pairs": gap.max_pairs,
        "cap_binding": cap_binding,
        "screening": note,
        "selection": "globally lowest SHA-256 content priority over unique oriented content pairs",
    }
    return tuple(tasks), stats


def _build_pair_task(
    key: Tuple[str, str, str],
    candidate: _PairCandidate,
    content_index: Mapping[Tuple[str, str], Sequence[Record]],
    seed: int,
) -> PairTask:
    lo, hi, changed = candidate.lo, candidate.hi, candidate.changed
    view_a, targets_a = mask_complete(chain_sequences(lo.heavy, lo.light, lo.antigen), changed)
    view_b, targets_b = mask_complete(chain_sequences(hi.heavy, hi.light, hi.antigen), changed)
    if view_a != view_b:
        raise ValueError("the endpoints' masked visible views differ; the unchanged background is not identical")
    if targets_a.coords() != tuple(changed) or targets_b.coords() != tuple(changed):
        raise ValueError("an endpoint does not cover exactly the changed coordinates")
    digest = view_a.content_digest()
    order = explicit_order(changed, digest, seed, tag="pair-order")
    reverse = None if len(order) == 1 else tuple(reversed(order))
    contrasts = sorted(
        (
            Contrast(record_id_a=ra.record_id, record_id_b=rb.record_id, source=ra.source, assay=ra.assay)
            for ra in content_index[(key[0], key[2])]
            for rb in content_index[(key[1], key[2])]
            if (ra.source, ra.assay) == (rb.source, rb.assay)
        ),
        key=lambda c: (c.source, c.assay),
    )
    if not contrasts:
        raise ValueError("a selected pair task has no same-source, same-assay contrast")
    return PairTask(
        task_id=_digest("pairtask1", *key),
        priority=_digest("prio-pair", str(seed), *key),
        partition=lo.partition,
        antigen=lo.antigen,
        region=candidate.region,
        a=Endpoint(lo.heavy, lo.light),
        b=Endpoint(hi.heavy, hi.light),
        a_fingerprint=lo.pair_fingerprint,
        b_fingerprint=hi.pair_fingerprint,
        changed=tuple(changed),
        order=order,
        reverse_order=reverse,
        view_digest=digest,
        cost=pair_task_cost(len(changed)),
        contrasts=tuple(contrasts),
    )


# --- the plan ---------------------------------------------------------------------------------------


def plan_diagnostics(
    records: Sequence[Record], config: DiagnosticsConfig, feature_forwards: int, *, allow_test: bool = False
) -> DiagnosticPlan:
    """Plan every masked diagnostic from the active eligible ``records`` and price it, with no backend.

    ``records`` must already exclude the sealed test partition unless it is unsealed; ``allow_test=False``
    refuses a test record outright rather than trusting the caller.
    """
    if isinstance(feature_forwards, bool) or not isinstance(feature_forwards, int) or feature_forwards < 0:
        raise ValueError(f"feature_forwards must be a non-negative integer, got {feature_forwards!r}")
    if not allow_test and any(record.partition == "test" for record in records):
        raise ValueError("a sealed test record reached diagnostic planning")
    chosen = [record for record in records if record.partition in config.partitions]
    singleton_tasks, singleton_stats = plan_singletons(chosen, config)
    pair_tasks, pair_stats = plan_pairs(chosen, config)
    budget = ForwardBudget(
        feature=feature_forwards,
        singleton=sum(task.cost for task in singleton_tasks),
        pair=sum(task.cost for task in pair_tasks),
        max_forwards=config.max_forwards,
    )
    return DiagnosticPlan(
        seed=config.seed,
        kappa=config.local_gap.kappa,
        partitions=tuple(config.partitions),
        singleton_tasks=singleton_tasks,
        pair_tasks=pair_tasks,
        budget=budget,
        singleton_stats=MappingProxyType(dict(singleton_stats)),
        pair_stats=MappingProxyType(dict(pair_stats)),
    )


def plan_document(plan: DiagnosticPlan) -> Dict[str, Any]:
    """The plan as finite JSON. It names coordinates, orders and ids; it holds no residue or measured value."""
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "seed": plan.seed,
        "kappa": plan.kappa,
        "partitions": list(plan.partitions),
        "budget": plan.budget.to_dict(),
        "singleton_stats": dict(plan.singleton_stats),
        "pair_stats": dict(plan.pair_stats),
        "singleton_tasks": [
            {
                "task_id": task.task_id,
                "priority": task.priority,
                "pair_fingerprint": task.pair_fingerprint,
                "partition": task.partition,
                "region": coords_text(task.region),
                "record_ids": list(task.record_ids),
                "sources_assays": [list(pair) for pair in task.sources_assays],
                "skipped_counts": list(task.skipped_counts),
                "cost": task.cost,
                "cases": [
                    {
                        "count": case.count,
                        "geometry": case.geometry,
                        "mode": case.mode,
                        "sites": coords_text(case.sites),
                        "order": coords_text(case.order),
                        "reverse_order": None if case.reverse_order is None else coords_text(case.reverse_order),
                        "view_digest": case.view_digest,
                        "cost": case.cost,
                    }
                    for case in task.cases
                ],
            }
            for task in plan.singleton_tasks
        ],
        "pair_tasks": [
            {
                "task_id": task.task_id,
                "priority": task.priority,
                "partition": task.partition,
                "endpoint_a_pair_fingerprint": task.a_fingerprint,
                "endpoint_b_pair_fingerprint": task.b_fingerprint,
                "region": coords_text(task.region),
                "changed": coords_text(task.changed),
                "distance": task.distance,
                "order": coords_text(task.order),
                "reverse_order": None if task.reverse_order is None else coords_text(task.reverse_order),
                "view_digest": task.view_digest,
                "cost": task.cost,
                "contrasts": [
                    {
                        "record_id_a": c.record_id_a,
                        "record_id_b": c.record_id_b,
                        "source": c.source,
                        "assay": c.assay,
                    }
                    for c in task.contrasts
                ],
            }
            for task in plan.pair_tasks
        ],
    }


# --- counting backend -------------------------------------------------------------------------------


class CountingBackend:
    """Counts every forward by phase, refuses one beyond the planned count and confirms the total on success.

    The wrapper adds nothing else: no cache, no batching, no change to the view it passes on.
    """

    def __init__(self, inner: Optional["ResidueBackend"], limits: Mapping[str, int]) -> None:
        self._inner = inner
        self._limits = dict(limits)
        for name, limit in self._limits.items():
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError(f"limit for phase {name!r} must be a non-negative integer, got {limit!r}")
        self.calls: Dict[str, int] = {name: 0 for name in self._limits}
        self._phase: Optional[str] = None

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())

    @contextmanager
    def phase(self, name: str) -> Iterator["CountingBackend"]:
        if name not in self._limits:
            raise KeyError(f"unknown phase {name!r}; phases are {sorted(self._limits)}")
        if self._phase is not None:
            raise RuntimeError("budget phases do not nest")
        self._phase = name
        try:
            yield self
        finally:
            self._phase = None

    def forward(self, view: VisibleView) -> Any:
        phase = self._phase
        if phase is None:
            raise BudgetOverrun("a backend forward was requested outside a budgeted phase")
        if self.calls[phase] >= self._limits[phase]:
            raise BudgetOverrun(f"phase {phase!r} would exceed its planned {self._limits[phase]} forwards")
        self.calls[phase] += 1
        if self._inner is None:
            raise BudgetOverrun("no backend was constructed because no forward was planned")
        return self._inner.forward(view)

    def describe(self) -> Dict[str, Any]:
        describe = getattr(self._inner, "describe", None)
        return dict(describe()) if callable(describe) else {"backend": type(self._inner).__name__}

    def verify(self) -> None:
        wrong = {name: (self.calls[name], limit) for name, limit in self._limits.items() if self.calls[name] != limit}
        if wrong:
            raise BudgetMismatch(f"actual calls differ from the plan (phase: (actual, planned)): {wrong}")


# --- execution (torch, imported lazily) ----------------------------------------------------------------


def _floats(tensor: Any) -> List[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _sequential_entry(result: Any) -> Dict[str, Any]:
    total = float(result.log_prob.detach())
    return {
        "order": coords_text(result.order),
        "sum": total,
        "mean": total / len(result.order),
        "step_log_probs": _floats(result.step_log_probs),
        "step_canonical_mass": _floats(result.step_mass),
        "step_mask_ratio": [float(ratio) for ratio in result.step_mask_ratio],
    }


def _score_cell(
    backend: Any, task: SingletonTask, case: MaskCase, arm: str, view: VisibleView, targets: MaskedTargets
) -> Dict[str, Any]:
    import torch

    from smallAntibodyGen.models.antibody_antigen.denoising import complete, masked_nll, sequential_log_prob

    with torch.no_grad():
        nll = masked_nll(backend, view, targets)
        forward = sequential_log_prob(backend, view, targets, case.order)
        reverse = (
            None if case.reverse_order is None else sequential_log_prob(backend, view, targets, case.reverse_order)
        )
    # The completion is given the view and the shared first order only; it is scored against truth below.
    completion = complete(backend, view, order=case.order)
    if set(completion.order) != set(case.sites) or completion.view.hidden_sites():
        raise AssertionError("completion did not fill exactly the hidden sites")
    for coord in view.coords():
        if view.token(coord) is not None and completion.view.token(coord) != view.token(coord):
            raise AssertionError(f"completion changed the known residue at {coord!r}")
    correct = sum(1 for coord, residue in zip(completion.order, completion.residues) if targets.residue(coord) == residue)
    site_nll = _floats(nll.site_nll)
    nll_sum = float(nll.total.detach())
    forward_entry = _sequential_entry(forward)
    reverse_entry = None if reverse is None else _sequential_entry(reverse)
    sensitivity: Optional[Dict[str, Any]] = None
    if reverse is not None and reverse_entry is not None:
        f_steps = dict(zip(forward.order, _floats(forward.step_log_probs)))
        r_steps = dict(zip(reverse.order, _floats(reverse.step_log_probs)))
        per_site = [f_steps[coord] - r_steps[coord] for coord in case.sites]
        sensitivity = {
            "coords": coords_text(case.sites),
            "per_site_forward_minus_reverse": per_site,
            "sum_difference": forward_entry["sum"] - reverse_entry["sum"],
            "max_abs_per_site": max(abs(value) for value in per_site),
        }
    return {
        "task_id": task.task_id,
        "pair_fingerprint": task.pair_fingerprint,
        "partition": task.partition,
        "count": case.count,
        "geometry": case.geometry,
        "arm": arm,
        "coords": coords_text(case.sites),
        "nll": {
            "sum": nll_sum,
            "token_mean": nll_sum / case.count,
            "site": site_nll,
            "canonical_mass": _floats(nll.canonical_mass),
            "mask_ratio": float(nll.mask_ratio),
        },
        "sequential": {"forward": forward_entry, "reverse": reverse_entry},
        "order_sensitivity": sensitivity,
        "completion": {
            "order": coords_text(completion.order),
            "residues": "".join(completion.residues),
            "step_log_probs": [float(value) for value in completion.step_log_probs],
            "correct": correct,
            "total": len(completion.order),
            "exact": correct == len(completion.order),
            "known_residues_preserved": True,
        },
    }


def run_singleton_tasks(backend: Any, tasks: Sequence[SingletonTask]) -> List[Dict[str, Any]]:
    """Score every planned mask case with BOTH arms; each masked view is built once and the antibody-only one derived."""
    cells: List[Dict[str, Any]] = []
    for task in tasks:
        sequences = chain_sequences(task.heavy, task.light, task.antigen)
        for case in task.cases:
            joint, targets = mask_complete(sequences, case.sites)
            if joint.content_digest() != case.view_digest:
                raise AssertionError("the masked view differs from the plan")
            for arm, view in (("joint", joint), ("antibody_only", joint.without_ag())):
                cells.append(_score_cell(backend, task, case, arm, view, targets))
    return cells


def _gap_entry(gap: Any) -> Dict[str, Any]:
    return {
        "gap": float(gap.gap.detach()),
        "j_a": float(gap.a.log_prob.detach()),
        "j_b": float(gap.b.log_prob.detach()),
        "a_step_log_probs": _floats(gap.a.step_log_probs),
        "b_step_log_probs": _floats(gap.b.step_log_probs),
        "a_step_canonical_mass": _floats(gap.a.step_mass),
        "b_step_canonical_mass": _floats(gap.b.step_mass),
        "step_mask_ratio": [float(ratio) for ratio in gap.a.step_mask_ratio],
    }


def run_pair_tasks(backend: Any, tasks: Sequence[PairTask]) -> List[Dict[str, Any]]:
    """``J(B) - J(A)`` for every planned pair task, per arm and order; each task is scored once per arm/order."""
    import torch

    from smallAntibodyGen.models.antibody_antigen.denoising import sequential_gap

    results: List[Dict[str, Any]] = []
    for task in tasks:
        view, targets_a = mask_complete(chain_sequences(task.a.heavy, task.a.light, task.antigen), task.changed)
        view_b, targets_b = mask_complete(chain_sequences(task.b.heavy, task.b.light, task.antigen), task.changed)
        if view != view_b or view.content_digest() != task.view_digest:
            raise AssertionError("the shared masked view differs from the plan")
        arms: Dict[str, Any] = {}
        for arm, arm_view in (("joint", view), ("antibody_only", view.without_ag())):
            entry: Dict[str, Any] = {}
            for name, order in (("forward", task.order), ("reverse", task.reverse_order)):
                if order is None:
                    continue
                with torch.no_grad():
                    entry[name] = _gap_entry(sequential_gap(backend, arm_view, order, targets_a, targets_b))
            entry["order_gap_difference"] = (
                entry["forward"]["gap"] - entry["reverse"]["gap"] if "reverse" in entry else None
            )
            arms[arm] = entry
        results.append(
            {
                "task_id": task.task_id,
                "partition": task.partition,
                "distance": task.distance,
                "changed": coords_text(task.changed),
                "order": coords_text(task.order),
                "reverse_order": None if task.reverse_order is None else coords_text(task.reverse_order),
                "arms": arms,
            }
        )
    return results


# --- aggregation ------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    value = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)
    return min(1.0, max(-1.0, value))


def count_label(cell: Mapping[str, Any]) -> Any:
    """The report's mask-count label: the integer ``k`` of a partial mask, ``"full"`` for the whole region."""
    return FULL_GEOMETRY if cell["geometry"] == FULL_GEOMETRY else cell["count"]


def _group_sort_key(key: Tuple[Any, ...]) -> Tuple[int, ...]:
    """Order ``(partition, count label, geometry[, arm])`` groups: partition, partial counts then full, geometry, arm."""
    partition, label, geometry, *arm = key
    size = label if isinstance(label, int) else 1 << 30
    return (PARTITIONS.index(partition), size, _GEOMETRY_RANK[geometry], *(ARMS.index(name) for name in arm))


def _singleton_entry(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sites = sum(cell["count"] for cell in cells)
    with_reverse = [cell for cell in cells if cell["sequential"]["reverse"] is not None]
    sensitive = [cell["order_sensitivity"] for cell in cells if cell["order_sensitivity"] is not None]
    entry: Dict[str, Any] = {
        "n_views": len(cells),
        "n_distinct_sequence_pairs": len({cell["pair_fingerprint"] for cell in cells}),
        "n_distinct_content_tasks": len({cell["task_id"] for cell in cells}),
        "n_masked_sites": sites,
        "mask_sizes": sorted({cell["count"] for cell in cells}),
        "nll": {
            "macro_case_mean": _mean([cell["nll"]["token_mean"] for cell in cells]),
            "token_weighted_mean": math.fsum(cell["nll"]["sum"] for cell in cells) / sites,
        },
        "sequential_forward": {
            "macro_case_mean": _mean([cell["sequential"]["forward"]["mean"] for cell in cells]),
            "token_weighted_mean": math.fsum(cell["sequential"]["forward"]["sum"] for cell in cells) / sites,
        },
        "sequential_reverse": None,
        "order_sensitivity": None,
        "recovery": {
            "macro_case_mean_rate": _mean([cell["completion"]["correct"] / cell["completion"]["total"] for cell in cells]),
            "token_weighted_rate": math.fsum(cell["completion"]["correct"] for cell in cells)
            / math.fsum(cell["completion"]["total"] for cell in cells),
            "exact_fraction": _mean([1.0 if cell["completion"]["exact"] else 0.0 for cell in cells]),
        },
        "canonical_mass": {
            "nll_mean_over_sites": math.fsum(m for cell in cells for m in cell["nll"]["canonical_mass"]) / sites
        },
        "mask_ratio": {"initial_mean": _mean([cell["nll"]["mask_ratio"] for cell in cells])},
    }
    if with_reverse:
        reverse_sites = sum(cell["count"] for cell in with_reverse)
        entry["sequential_reverse"] = {
            "n_views": len(with_reverse),
            "macro_case_mean": _mean([cell["sequential"]["reverse"]["mean"] for cell in with_reverse]),
            "token_weighted_mean": math.fsum(cell["sequential"]["reverse"]["sum"] for cell in with_reverse) / reverse_sites,
        }
    if sensitive:
        entry["order_sensitivity"] = {
            "n_views": len(sensitive),
            "mean_abs_sum_difference": _mean([abs(item["sum_difference"]) for item in sensitive]),
            "mean_max_abs_per_site": _mean([item["max_abs_per_site"] for item in sensitive]),
        }
    return entry


def _paired_entry(pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
    sites = sum(joint["count"] for joint, _ in pairs)
    ratios = [j["nll"]["mask_ratio"] / a["nll"]["mask_ratio"] for j, a in pairs if a["nll"]["mask_ratio"] > 0.0]
    return {
        "n_views": len(pairs),
        "n_distinct_sequence_pairs": len({joint["pair_fingerprint"] for joint, _ in pairs}),
        "n_distinct_content_tasks": len({joint["task_id"] for joint, _ in pairs}),
        "n_masked_sites": sites,
        "mask_sizes": sorted({joint["count"] for joint, _ in pairs}),
        "nll_difference": {
            "macro_case_mean": _mean([j["nll"]["token_mean"] - a["nll"]["token_mean"] for j, a in pairs]),
            "token_weighted_mean": math.fsum(j["nll"]["sum"] - a["nll"]["sum"] for j, a in pairs) / sites,
        },
        "sequential_forward_difference": {
            "macro_case_mean": _mean(
                [j["sequential"]["forward"]["mean"] - a["sequential"]["forward"]["mean"] for j, a in pairs]
            ),
            "token_weighted_mean": math.fsum(
                j["sequential"]["forward"]["sum"] - a["sequential"]["forward"]["sum"] for j, a in pairs
            )
            / sites,
        },
        "native_mask_ratio": {
            "joint_mean": _mean([j["nll"]["mask_ratio"] for j, _ in pairs]),
            "antibody_only_mean": _mean([a["nll"]["mask_ratio"] for _, a in pairs]),
            "joint_over_antibody_only_mean": _mean(ratios) if ratios else None,
        },
    }


def summarize_singletons(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per (partition, count label, geometry, arm) groups, then paired joint-minus-antibody-only differences.

    The label is ``k`` for a partial mask and ``"full"`` for the whole-region mask, whose region size can vary
    between tasks; that is where the macro-case mean and the token-weighted mean differ.
    """
    grouped: Dict[Tuple[str, Any, str, str], List[Mapping[str, Any]]] = {}
    by_case: Dict[Tuple[str, Any, str, str, int], Dict[str, Mapping[str, Any]]] = {}
    for cell in cells:
        label = count_label(cell)
        grouped.setdefault((cell["partition"], label, cell["geometry"], cell["arm"]), []).append(cell)
        by_case.setdefault((cell["partition"], label, cell["geometry"], cell["task_id"], cell["count"]), {})[
            cell["arm"]
        ] = cell
    paired: Dict[Tuple[str, Any, str], List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for (partition, label, geometry, _, _), arms in by_case.items():
        if "joint" in arms and "antibody_only" in arms:
            paired.setdefault((partition, label, geometry), []).append((arms["joint"], arms["antibody_only"]))
    return {
        "groups": [
            dict(zip(("partition", "count_label", "geometry", "arm"), key), **_singleton_entry(grouped[key]))
            for key in sorted(grouped, key=_group_sort_key)
        ],
        "paired_joint_minus_antibody_only": [
            dict(zip(("partition", "count_label", "geometry"), key), **_paired_entry(paired[key]))
            for key in sorted(paired, key=_group_sort_key)
        ],
        "independence_note": INDEPENDENCE_NOTE,
        "native_confound_note": NATIVE_CONFOUND_NOTE,
        "antigen_specificity_of_joint_gap": "unestablished",
    }


def _contrast_rows(
    results: Sequence[Mapping[str, Any]], tasks: Sequence[PairTask], records_by_id: Mapping[str, Record], kappa: float
) -> List[Dict[str, Any]]:
    by_id = {task.task_id: task for task in tasks}
    rows: List[Dict[str, Any]] = []
    for result in results:
        task = by_id[result["task_id"]]
        for contrast in task.contrasts:
            delta = records_by_id[contrast.record_id_b].pkd - records_by_id[contrast.record_id_a].pkd
            target = kappa * delta
            arms: Dict[str, Any] = {}
            for arm in ARMS:
                arms[arm] = {
                    name: {"gap": result["arms"][arm][name]["gap"], "residual": result["arms"][arm][name]["gap"] - target}
                    for name in ORDER_NAMES
                    if name in result["arms"][arm]
                }
            rows.append(
                {
                    "task_id": task.task_id,
                    "partition": task.partition,
                    "distance": task.distance,
                    "source": contrast.source,
                    "assay": contrast.assay,
                    "record_id_a": contrast.record_id_a,
                    "record_id_b": contrast.record_id_b,
                    "delta_pkd": delta,
                    "target": target,
                    "arms": arms,
                }
            )
    return rows


def _gap_group(rows: Sequence[Mapping[str, Any]], arm: str, order: str) -> Dict[str, Any]:
    gaps = [row["arms"][arm][order]["gap"] for row in rows]
    deltas = [row["delta_pkd"] for row in rows]
    targets = [row["target"] for row in rows]
    residuals = [row["arms"][arm][order]["residual"] for row in rows]
    nonzero = [(g, d) for g, d in zip(gaps, deltas) if g != 0.0 and d != 0.0]
    return {
        "n_pairs": len({row["task_id"] for row in rows}),
        "n_contrasts": len(rows),
        "mean_gap": _mean(gaps),
        "mean_delta_pkd": _mean(deltas),
        "mean_target": _mean(targets),
        "mean_residual": _mean(residuals),
        "mean_abs_residual": _mean([abs(r) for r in residuals]),
        "rmse_residual": math.sqrt(math.fsum(r * r for r in residuals) / len(residuals)),
        "sign_agreement": {"n_nonzero": len(nonzero), "n_agree": sum(1 for g, d in nonzero if (g > 0) == (d > 0))},
        "pearson_gap_vs_delta_pkd": _pearson(gaps, deltas),
    }


def summarize_pairs(
    results: Sequence[Mapping[str, Any]],
    tasks: Sequence[PairTask],
    records_by_id: Mapping[str, Record],
    kappa: float,
) -> Dict[str, Any]:
    """Primary groups are (partition, source, assay, distance); a task counts once inside each group."""
    rows = _contrast_rows(results, tasks, records_by_id, kappa)
    grouped: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["partition"], row["source"], row["assay"], row["distance"]), []).append(row)
    names = ("partition", "source", "assay", "distance")
    groups: List[Dict[str, Any]] = []
    joint_minus: List[Dict[str, Any]] = []
    order_diff: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=lambda k: (PARTITIONS.index(k[0]), k[1], k[2], k[3])):
        members = grouped[key]
        for arm in ARMS:
            for order in ORDER_NAMES:
                if order in members[0]["arms"][arm]:
                    groups.append(dict(zip(names, key), arm=arm, order=order, **_gap_group(members, arm, order)))
        for order in ORDER_NAMES:
            if order in members[0]["arms"]["joint"]:
                joint_minus.append(
                    dict(
                        zip(names, key),
                        order=order,
                        n_contrasts=len(members),
                        mean_gap_difference=_mean(
                            [m["arms"]["joint"][order]["gap"] - m["arms"]["antibody_only"][order]["gap"] for m in members]
                        ),
                        mean_abs_residual_difference=_mean(
                            [
                                abs(m["arms"]["joint"][order]["residual"]) - abs(m["arms"]["antibody_only"][order]["residual"])
                                for m in members
                            ]
                        ),
                    )
                )
        if "reverse" in members[0]["arms"]["joint"]:
            for arm in ARMS:
                diffs = [m["arms"][arm]["forward"]["gap"] - m["arms"][arm]["reverse"]["gap"] for m in members]
                order_diff.append(
                    dict(
                        zip(names, key),
                        arm=arm,
                        n_contrasts=len(members),
                        mean_forward_minus_reverse=_mean(diffs),
                        mean_abs_forward_minus_reverse=_mean([abs(d) for d in diffs]),
                    )
                )
    return {
        "unique_pair_tasks": len(results),
        "measurement_contrasts": len(rows),
        "groups": groups,
        "paired_joint_minus_antibody_only": joint_minus,
        "order_reversal": order_diff,
        "contrasts": rows,
        "note": PAIR_NOTE,
        "kappa": kappa,
        "antigen_specificity_of_joint_gap": "unestablished",
    }
