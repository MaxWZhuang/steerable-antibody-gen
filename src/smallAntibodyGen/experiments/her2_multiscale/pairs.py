"""Measured endpoint pairs at Hamming distance 1, 2 or 3, from construct summaries.

A pair is two **training** constructs that share one study, family, assay, target, scale, parent
and declared common region, have equal chain lengths, differ at between one and three
coordinates, every difference lying inside that region, and each carry an exact finite pKD.
"Exact" is :func:`~..her2_routes_pairs.quantitative_status` -- read off the replicates, never off
``summary.kind``: a degenerate uncensored interval is exact, a censored, mixed-censoring, ordinal
or non-finite summary is not. Two endpoints with equal numeric values are a valid pair with
``delta == 0``. Identities are the full VH+VL sequences, never the core alone, so two constructs
that share a core but differ elsewhere are different endpoints and a difference outside the
region disqualifies the pair.

**Input and refusals.** :func:`build_pair_pool` takes every summary it is given. A summary is one
observed *population*: ``summary.population_key`` = ``(full identity, study, assay, target)``.
Before anything is filtered, summaries naming the same population are deduplicated if their
evidence (group fields, region, partition, provenance, every replicate) is identical and *refused
with an error* if it is not -- a population that is ``train`` in one record and ``development`` in
another, or that carries two parents, families, scales or regions, is a conflict and is never split
into two groups. A malformed region raises too. Only then are summaries filtered, and each one
dropped is counted by reason: ``not_train_partition`` (the summary's own declared partition must be
``"train"``; nothing is inferred), ``not_training_population`` (its population key must be in the
caller's read-only ``training_population_keys``, the authority a mutable flag is not),
``closure_refused`` (the :class:`~..her2_routes_split.ExclusionClosure` is applied to the identity
always and to the window ``window_of(construct)`` returns when it returns one), the
:func:`quantitative_status` reasons, and ``no_region``.

**The closure window is the caller's, not a guess.** The closure's radius rule is defined for one
edit space (the fixed HER2 scaffold's core). ``window_of`` -- production passes
``FlightServices.closure_window`` -- returns that window for a construct in that space and ``None``
for any other, and only a returned window is tested. A region string of an unrelated target that
happens to spell a held-out core is therefore not refused.

**Region coordinates.** A summary's ``region`` may list its coordinates in any order; it is
canonicalised (:func:`.masks.canonical_region`, heavy-then-light ascending) before it is compared,
hashed or stored. Two summaries whose regions are the same set are in the same group.

**The pool.** For every group and every configured distance the pool keeps at most ``capacity``
pairs: the ``capacity`` with the smallest SHA-256 retention key, a function of only the seed, the
group, the distance and the two identities. The retained set is therefore the same for every input
order, nested as capacity grows, and independent of every measurement. It bounds *storage*; it does
not bound or equalise how many retained pairs an endpoint is in, and some endpoints may have none.
:meth:`PairPool.report` states the degree and the sampling exposure the retained graph really has.
No claim of uniform total exposure or of coverage of the construct graph is made.

**Cost.** Constructs in one group are bucketed by their full masked background (every chain residue
outside the region), because only constructs with equal backgrounds can differ solely inside the
region. Within a bucket of ``m`` constructs and a region of ``n`` coordinates the scan compares all
``m * (m - 1) / 2`` windows, vectorised row against rest: ``O(m^2 * n)`` time. Memory is
``O(N)`` summaries plus at most ``capacity`` retained keys per (group, distance); pairs are never
materialised, so there is no quadratic memory. The number of pairs this skips because their
backgrounds differ is reported as ``cross_background_pairs``.

**Two digests.** ``structural_digest`` names the pair -- group, region, the two identities and the
changed coordinates -- and does not move when a measurement moves. ``supervision_digest`` also
binds each endpoint's exact replicates, metadata, partition and provenance. A pair carries
references to its two summaries, not copies of their sequences.

**Staleness.** Summaries are mutable and a pool outlives the moment it was built.
:func:`require_pair_current` re-reads the live summaries and the closure, ``window_of`` and
``training_population_keys`` the consumer holds *now*, and refuses a pair whose endpoints left the
training partition or the training population, entered the closure, stopped being exact, changed
support or identity, changed value, or changed anything the supervision digest binds.

**Anchors are populations.** An endpoint is a population (the unit :meth:`PairPool.report` counts as
``eligible_endpoints``), so the anchor table, the uniform anchor draw and the coverage figures are all
keyed by ``population_key``. One molecule measured in two assays is two endpoints.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from numbers import Integral

import numpy as np

from .. import her2_routes_data as data_lib
from ..her2_routes_pairs import masked_context, quantitative_status
from ..her2_routes_split import ExclusionClosure
from ..her2_runtime import require
from .masks import canonical_region

#: Bumped when pair eligibility or the meaning of a pair record changes.
PAIRS_VERSION = 1

#: The largest Hamming distance a pair may have. The objective needs one sequential reveal per
#: changed coordinate, and three is the pilot's declared limit.
MAX_DISTANCE = 3
DEFAULT_DISTANCES = (1, 2, 3)

#: Retained pairs per (group, distance). Bounds storage, not an endpoint's degree.
DEFAULT_CAPACITY = 256

#: The summary fields two endpoints must share (the declared region is the seventh).
GROUP_FIELDS = ("study", "family", "assay", "target", "scale", "parent")


def _digest(document):
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict_int(value, name, *, minimum, maximum=None):
    require(isinstance(value, Integral) and not isinstance(value, bool),
            f"{name} must be an integer, got {value!r}")
    value = int(value)
    allowed = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
    require(value >= minimum and (maximum is None or value <= maximum),
            f"{name} must be {allowed}, got {value}")
    return value


def validated_distances(distances):
    """A sorted tuple of distinct integers in ``1..MAX_DISTANCE``; anything else is refused."""
    require(isinstance(distances, (tuple, list)) and len(distances) > 0,
            f"distances must be a non-empty tuple or list of integers, got {distances!r}")
    cleaned = tuple(_strict_int(item, "A distance", minimum=1, maximum=MAX_DISTANCE)
                    for item in distances)
    require(len(set(cleaned)) == len(cleaned), f"distances repeats a value: {list(distances)}")
    return tuple(sorted(cleaned))


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def _coordinates(sites):
    return [[chain, int(index)] for chain, index in sites]


def _number(value):
    """A JSON-safe rendering of a measurement field: ``None`` stays, a non-finite float is text."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else str(number)


def _measurement_record(item):
    low, high = item.bounds()
    return [str(item.measurement_id), None if item.replicate is None else str(item.replicate),
            _number(item.value), _number(low), _number(high), str(item.censoring),
            _number(item.uncertainty)]


def _group(summary):
    return tuple(str(getattr(summary, name)) for name in GROUP_FIELDS)


def _region_of(summary):
    """The summary's canonical declared region, or ``()`` when it declares none."""
    if not summary.region:
        return ()
    return canonical_region(summary.region, heavy_length=len(summary.construct.heavy),
                            light_length=len(summary.construct.light))


def _evidence(summary, region):
    """Everything supervision could depend on for one summary, as plain data."""
    records = sorted((_measurement_record(item) for item in summary.measurements),
                     key=lambda record: json.dumps(record, sort_keys=True))
    return {"identity": summary.identity, "group": list(_group(summary)),
            "region": _coordinates(region), "kind": str(summary.kind),
            "partition": str(summary.partition), "source": str(summary.source),
            "construct_source": str(summary.construct.source),
            "record_id": str(summary.construct.record_id), "measurements": records}


@dataclass(frozen=True)
class _Endpoint:
    summary: object
    region: tuple
    group: tuple
    evidence: str


def _conflict_message(known, evidence):
    before, after = json.loads(known.evidence), json.loads(evidence)
    fields = sorted(name for name in before if before[name] != after[name])
    return (f"Conflicting evidence for construct {known.summary.identity[:24]}... in study "
            f"{known.summary.study!r}, assay {known.summary.assay!r}, target "
            f"{known.summary.target!r}: two summaries name the same observed population and "
            f"disagree in {fields}. One of them is wrong, and keeping whichever was listed first "
            "would train on it.")


def validated_population_keys(keys):
    """``keys`` as a frozenset of ``(identity, study, assay, target)`` string tuples.

    The training population is an explicit, read-only set the caller derives from its own
    authority (the restored split membership), never from a flag on the rows being tested. It must
    be a non-empty ``set`` or ``frozenset``: a generator would be consumed by the first check, and
    a string or a list is not a set of keys.
    """
    require(isinstance(keys, (set, frozenset)) and len(keys) > 0,
            "training_population_keys must be a non-empty set or frozenset of "
            "(identity, study, assay, target) keys")
    for key in keys:
        require(isinstance(key, tuple) and len(key) == 4 and all(isinstance(part, str)
                                                                for part in key),
                f"A training population key is a (identity, study, assay, target) tuple of "
                f"strings, got {key!r}")
    return frozenset(keys)


def population_keys_digest(keys):
    """SHA-256 over the sorted keys, streamed. Names the set a pool was built against."""
    digest = hashlib.sha256()
    for key in sorted(validated_population_keys(keys)):
        digest.update(json.dumps(list(key), separators=(",", ":"), ensure_ascii=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _require_window_of(window_of):
    require(callable(window_of),
            "window_of(construct) is required: the closure's radius rule is defined for the HER2 "
            "edit window only, so the caller says which constructs live in that space (production: "
            "FlightServices.closure_window) and a window is never guessed from a region")
    return window_of


def _unique_endpoints(summaries):
    """``(endpoints, duplicates)``: identical evidence deduplicated, conflicting evidence refused.

    The unit is the observed population, ``summary.population_key`` = ``(identity, study, assay,
    target)``. Runs over every summary before any filter and compares the group fields, region,
    partition and every other piece of evidence, so a population that is ``train`` in one record and
    ``development`` in another -- or carries two regions, parents, families or scales -- is a
    conflict, not a silent drop of the second and not two groups.
    """
    kept, duplicates = {}, 0
    for summary in summaries:
        require(isinstance(summary, data_lib.ConstructSummary),
                f"build_pair_pool takes ConstructSummary objects, got {type(summary).__name__}")
        region = _region_of(summary)
        evidence = json.dumps(_evidence(summary, region), sort_keys=True, separators=(",", ":"))
        key = summary.population_key
        known = kept.get(key)
        if known is None:
            kept[key] = _Endpoint(summary=summary, region=region, group=_group(summary),
                                  evidence=evidence)
        elif known.evidence == evidence:
            duplicates += 1
        else:
            raise ValueError(_conflict_message(known, evidence))
    ordered = sorted(kept.values(),
                     key=lambda item: (item.group, item.region, item.summary.identity))
    return ordered, duplicates


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------

def _closure_reason(summary, closure, window_of):
    """The closure's reason to refuse ``summary``, or ``None``.

    The full identity is always checked. The radius rule sees only the window ``window_of`` returns
    for this construct, and nothing when it returns ``None``.
    """
    return closure.violates(window=window_of(summary.construct), identity=summary.identity)


def eligibility_reason(summary, region, *, closure, window_of, training_population_keys):
    """``None`` when ``summary`` may be a pair endpoint, else the counted reason it may not.

    Three independent gates all say "training": the summary's own declared partition, membership of
    its population key in the caller's ``training_population_keys``, and the closure.
    """
    if summary.partition != data_lib.PARTITION_TRAIN:
        return "not_train_partition"
    if summary.population_key not in training_population_keys:
        return "not_training_population"
    if _closure_reason(summary, closure, window_of) is not None:
        return "closure_refused"
    eligible, reason = quantitative_status(summary)
    if not eligible:
        return reason
    if not region:
        return "no_region"
    return None


# ---------------------------------------------------------------------------
# one pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasuredPair:
    """Two endpoints, the coordinates they differ at, and the numbers measured for them.

    ``first_value``/``second_value`` are the exact pKD each endpoint had when the pair was made;
    :func:`require_pair_current` compares them with the live summaries. ``delta`` is
    ``second - first``. Orientation is a presentation choice: both digests treat the pair as
    unordered, and :meth:`oriented` swaps it.
    """

    group: tuple
    region: tuple
    sites: tuple
    distance: int
    first_value: float
    second_value: float
    structural_digest: str
    supervision_digest: str
    first: object = field(compare=False, repr=False)
    second: object = field(compare=False, repr=False)

    @property
    def delta(self):
        return self.second_value - self.first_value

    @property
    def identities(self):
        return (self.first.identity, self.second.identity)

    @property
    def population_keys(self):
        return (self.first.population_key, self.second.population_key)

    def oriented(self, anchor_identity):
        """This pair with the endpoint named ``anchor_identity`` first."""
        if self.first.identity == anchor_identity:
            return self
        require(self.second.identity == anchor_identity,
                "The anchor is not an endpoint of this pair")
        return replace(self, first=self.second, second=self.first,
                       first_value=self.second_value, second_value=self.first_value)

    def document(self):
        return {"distance": int(self.distance), "sites": _coordinates(self.sites),
                "group": dict(zip(GROUP_FIELDS, self.group)), "delta": float(self.delta),
                "structural_digest": self.structural_digest,
                "supervision_digest": self.supervision_digest}


def _differences(left, right):
    """Every coordinate at which two constructs differ, heavy first then light, ascending."""
    changed = []
    for chain, before, after in (("H", left.heavy, right.heavy), ("L", left.light, right.light)):
        require(len(before) == len(after),
                f"The {chain} chains have different length ({len(before)} and {len(after)}), so "
                "no Hamming distance is defined")
        changed.extend((chain, index) for index, (one, other) in enumerate(zip(before, after))
                       if one != other)
    return tuple(changed)


def pair_between(first, second):
    """The :class:`MeasuredPair` of two summaries, computed from the summaries alone.

    The full-chain comparison here is the ground truth the pool scan is checked against. It
    raises, naming the reason, if either summary is not an exact finite pKD measurement, if they
    do not share study/family/assay/target/scale/parent and region, if they are the same
    construct, if the chains differ in length, if they differ at more than
    :data:`MAX_DISTANCE` coordinates, or if any difference lies outside the region.
    """
    regions = []
    for name, summary in (("first", first), ("second", second)):
        require(isinstance(summary, data_lib.ConstructSummary),
                f"The {name} endpoint is not a ConstructSummary")
        eligible, reason = quantitative_status(summary)
        require(eligible, f"The {name} endpoint is not an exact finite pKD measurement ({reason})")
        region = _region_of(summary)
        require(region, f"The {name} endpoint declares no common region")
        regions.append(region)
    first_region, second_region = regions
    require(_group(first) == _group(second),
            "The endpoints do not share study, family, assay, target, scale and parent")
    require(first_region == second_region, "The endpoints declare different common regions")
    require(first.identity != second.identity, "The endpoints are the same construct")
    changed = _differences(first.construct, second.construct)
    require(1 <= len(changed) <= MAX_DISTANCE,
            f"The endpoints differ at {len(changed)} coordinates; a pair is at distance "
            f"1..{MAX_DISTANCE}")
    require(set(changed) <= set(first_region),
            "The endpoints differ outside the declared common region")
    group = _group(first)
    structural = _digest({"version": PAIRS_VERSION, "group": list(group),
                          "region": _coordinates(first_region),
                          "identities": sorted([first.identity, second.identity]),
                          "sites": _coordinates(changed)})
    supervision = _digest({"structural": structural,
                           "endpoints": sorted([_evidence(first, first_region),
                                                _evidence(second, second_region)],
                                               key=lambda item: item["identity"])})
    return MeasuredPair(group=group, region=first_region, sites=changed, distance=len(changed),
                        first_value=float(first.exact_value), second_value=float(second.exact_value),
                        structural_digest=structural, supervision_digest=supervision,
                        first=first, second=second)


def require_pair_current(pair, *, closure, window_of, training_population_keys):
    """Re-validate ``pair`` against its live summaries and the consumer's boundary; raise if stale.

    Returns ``True``. Every check reads the summaries as they are now: the training partition, the
    ``training_population_keys`` and ``closure`` the *consumer* passes (with its ``window_of``),
    exactness, and then a full recomputation of the pair compared with what the pair claims (support
    and identity, numeric values, and the supervision digest).
    """
    require(isinstance(pair, MeasuredPair), f"Expected a MeasuredPair, got {type(pair).__name__}")
    require(isinstance(closure, ExclusionClosure), "A pair is consumed against an ExclusionClosure")
    _require_window_of(window_of)
    keys = validated_population_keys(training_population_keys)
    label = pair.structural_digest[:12]
    for name in ("first", "second"):
        summary = getattr(pair, name)
        reason = eligibility_reason(summary, _region_of(summary), closure=closure,
                                    window_of=window_of, training_population_keys=keys)
        require(reason is None,
                f"Pair {label} is stale: its {name} endpoint is no longer an eligible training "
                f"endpoint ({reason})")
    try:
        fresh = pair_between(pair.first, pair.second)
    except ValueError as error:
        raise ValueError(f"Pair {label} is stale: {error}") from error
    require(fresh.structural_digest == pair.structural_digest and fresh.group == pair.group
            and fresh.region == pair.region and fresh.sites == pair.sites
            and fresh.distance == pair.distance,
            f"Pair {label} is stale: its support or identity changed")
    require(fresh.first_value == pair.first_value and fresh.second_value == pair.second_value,
            f"Pair {label} is stale: a numeric value changed")
    require(fresh.supervision_digest == pair.supervision_digest,
            f"Pair {label} is stale: measurements, metadata or provenance changed")
    return True


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------

def _retention_key(seed, label, distance, low, high):
    text = f"{seed}|{label}|{distance}|{low}|{high}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _offer(heap, capacity, key, low, high):
    """Keep the ``capacity`` smallest keys: a max-heap on the key, stored negated."""
    entry = (-key, low, high)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _scan(eligible, distances, capacity, seed):
    """``(heaps, seen, cross)`` over every pair of eligible endpoints that share a background.

    ``heaps[(group_key, distance)]`` holds the retained ``(-key, low, high)`` entries,
    ``seen[...]`` counts every qualified pair, ``cross`` counts the same-group pairs skipped
    because their backgrounds differ outside the region. ``eligible`` is sorted by group, region
    and identity, so ``low`` always names the smaller identity.
    """
    wanted = np.asarray(distances)
    digests = [hashlib.sha256(item.summary.identity.encode("utf-8")).hexdigest()
               for item in eligible]
    members = {}
    for position, item in enumerate(eligible):
        members.setdefault((item.group, item.region), []).append(position)
    heaps, seen, cross = {}, {}, 0
    for group_key, positions in members.items():
        label = _digest({"group": list(group_key[0]), "region": _coordinates(group_key[1])})
        backgrounds = {}
        for position in positions:
            item = eligible[position]
            backgrounds.setdefault(masked_context(item.summary.construct, item.region),
                                   []).append(position)
        pairs_in_group = len(positions) * (len(positions) - 1) // 2
        cross += pairs_in_group - sum(len(bucket) * (len(bucket) - 1) // 2
                                      for bucket in backgrounds.values())
        for bucket in backgrounds.values():
            windows = np.asarray([[ord(letter) for letter in
                                   eligible[position].summary.construct.residues(
                                       eligible[position].region)]
                                  for position in bucket], dtype=np.int16)
            for row in range(len(bucket) - 1):
                gaps = (windows[row + 1:] != windows[row]).sum(axis=1)
                for offset in np.flatnonzero(np.isin(gaps, wanted)):
                    low, high = bucket[row], bucket[row + 1 + int(offset)]
                    key = (group_key, int(gaps[offset]))
                    seen[key] = seen.get(key, 0) + 1
                    _offer(heaps.setdefault(key, []), capacity,
                           _retention_key(seed, label, key[1], digests[low], digests[high]),
                           low, high)
    return heaps, seen, cross


@dataclass(frozen=True)
class PairPool:
    """The retained pairs, the counts that say what was seen and refused, and two digests."""

    pairs: tuple
    distances: tuple
    capacity: int
    seed: int
    structural_digest: str
    supervision_digest: str
    population_digest: str
    counts: dict = field(compare=False, repr=False)

    @cached_property
    def _anchors(self):
        """``{distance: ((population_key, pairs), ...)}``: each endpoint population with its pairs.

        The key is the population (``identity, study, assay, target``), the unit
        ``counts["eligible_endpoints"]`` counts. Both endpoints of a pair share study, assay and
        target, so the two keys of a pair differ in the identity component only.
        """
        table = {distance: {} for distance in self.distances}
        for pair in self.pairs:
            for key in pair.population_keys:
                table[pair.distance].setdefault(key, []).append(pair)
        return {distance: tuple((key, tuple(table[distance][key]))
                                for key in sorted(table[distance]))
                for distance in table}

    def at_distance(self, distance):
        return tuple(pair for pair in self.pairs if pair.distance == distance)

    def draw(self, distance, rng):
        """One pair at ``distance``: a uniform anchor population, then a uniform retained partner.

        Each endpoint population with at least one retained pair at this distance is equally likely
        to be the anchor, whatever its degree; the partner is uniform among that anchor's retained
        pairs. The returned pair has the anchor first (oriented by the key's identity component).
        This equalises anchors, not total exposure -- see :meth:`report`. Two integers are drawn
        from ``rng`` per call.
        """
        distance = _strict_int(distance, "distance", minimum=1, maximum=MAX_DISTANCE)
        require(distance in self.distances,
                f"distance {distance} is not one of this pool's configured {self.distances}")
        anchors = self._anchors[distance]
        key, options = anchors[int(rng.integers(len(anchors)))]
        return options[int(rng.integers(len(options)))].oriented(key[0])

    def report(self):
        """The degree, coverage and sampling exposure the retained graph actually has, per distance.

        Every figure counts populations: an endpoint is one ``(identity, study, assay, target)``,
        the same unit as ``eligible_endpoints``. ``coverage`` is the share of those populations that
        are an endpoint of at least one retained pair at this distance. Exposure is the exact
        probability that a population appears (as anchor or as partner) in one :meth:`draw` at that
        distance. It is reported, not equalised: endpoints with few retained pairs are anchors as
        often as hubs, and hubs are partners far more often.
        """
        eligible_endpoints = self.counts["eligible_endpoints"]
        per_distance = {}
        for distance in self.distances:
            anchors = self._anchors[distance]
            total = len(anchors)
            degrees = [len(options) for _, options in anchors]
            exposure = {key: 1.0 / total for key, _ in anchors}
            for key, options in anchors:
                for pair in options:
                    partner = (pair.second.population_key
                               if pair.first.population_key == key else pair.first.population_key)
                    exposure[partner] += 1.0 / (total * len(options))
            levels = list(exposure.values())
            per_distance[str(distance)] = {
                "unit": "population (identity, study, assay, target)",
                "anchors": total, "retained_pairs": sum(degrees) // 2,
                "qualified_seen": self.counts["qualified_seen"][str(distance)],
                "population_endpoints": eligible_endpoints,
                "coverage": total / eligible_endpoints,
                "endpoints_without_a_retained_pair": eligible_endpoints - total,
                "degree": {"min": min(degrees), "median": float(np.median(degrees)),
                           "max": max(degrees)},
                "exposure": {"min": min(levels), "mean": float(np.mean(levels)),
                             "max": max(levels)}}
        return per_distance

    def document(self):
        return {"version": PAIRS_VERSION, "distances": list(self.distances),
                "capacity": int(self.capacity), "seed": int(self.seed),
                "structural_digest": self.structural_digest,
                "supervision_digest": self.supervision_digest,
                "training_population_digest": self.population_digest,
                "counts": self.counts, "report": self.report(),
                "reading": ("capacity bounds storage, not an endpoint's degree; exposure is "
                            "reported as it is and is not equalised")}


def build_pair_pool(summaries, *, closure, window_of, training_population_keys,
                    distances=DEFAULT_DISTANCES, capacity=DEFAULT_CAPACITY, seed=0):
    """Build the :class:`PairPool` of every qualified pair among ``summaries``.

    ``closure`` is the :class:`~..her2_routes_split.ExclusionClosure` every endpoint must clear,
    seen through ``window_of(construct)`` (``None`` for a construct outside the closure's edit
    space). ``training_population_keys`` is the REQUIRED read-only set of population keys that may
    be trained on; an endpoint outside it is refused whatever its ``partition`` says. ``distances``
    and ``capacity`` are strict integers. Raises if two summaries conflict, if any region is
    malformed, or if a configured distance has no qualified pair at all -- a stratified draw at that
    distance could not run, and a pool that quietly lacked it would train on fewer scales than the
    configuration says.
    """
    distances = validated_distances(distances)
    capacity = _strict_int(capacity, "capacity", minimum=1)
    seed = _strict_int(seed, "seed", minimum=0)
    require(isinstance(closure, ExclusionClosure), "build_pair_pool needs an ExclusionClosure")
    _require_window_of(window_of)
    keys = validated_population_keys(training_population_keys)
    summaries = list(summaries)
    endpoints, duplicates = _unique_endpoints(summaries)
    refused, eligible = {}, []
    for item in endpoints:
        reason = eligibility_reason(item.summary, item.region, closure=closure,
                                    window_of=window_of, training_population_keys=keys)
        if reason is None:
            eligible.append(item)
        else:
            refused[reason] = refused.get(reason, 0) + 1
    heaps, seen, cross = _scan(eligible, distances, capacity, seed)
    pairs = []
    for (_group_key, distance), heap in heaps.items():
        for _negated_key, low, high in heap:
            pair = pair_between(eligible[low].summary, eligible[high].summary)
            require(pair.distance == distance,
                    f"The scan placed a pair at distance {distance} but the full-chain comparison "
                    f"says {pair.distance}")
            pairs.append(pair)
    pairs.sort(key=lambda pair: (pair.distance, pair.structural_digest))
    seen_by_distance = {str(d): sum(n for (_, dist), n in seen.items() if dist == d)
                        for d in distances}
    kept_by_distance = {str(d): sum(1 for pair in pairs if pair.distance == d) for d in distances}
    for distance in distances:
        require(kept_by_distance[str(distance)] > 0,
                f"No qualified pair at distance {distance} among {len(eligible)} eligible "
                f"endpoints (configured distances {list(distances)}; refused "
                f"{dict(sorted(refused.items()))}). A stratified draw at that distance could "
                "not run.")
    counts = {"summaries": len(summaries), "duplicates": duplicates,
              "unique_endpoints": len(endpoints), "eligible_endpoints": len(eligible),
              "training_population_keys": len(keys),
              "refused": dict(sorted(refused.items())),
              "cross_background_pairs": cross,
              "qualified_seen": seen_by_distance, "retained": kept_by_distance,
              "groups": [{"group": dict(zip(GROUP_FIELDS, key[0][0])),
                          "region": _coordinates(key[0][1]), "distance": key[1],
                          "qualified_seen": seen[key], "retained": len(heaps[key])}
                         for key in sorted(seen)]}
    return PairPool(pairs=tuple(pairs), distances=distances, capacity=capacity, seed=seed,
                    structural_digest=_digest(sorted(pair.structural_digest for pair in pairs)),
                    supervision_digest=_digest(sorted(pair.supervision_digest for pair in pairs)),
                    population_digest=population_keys_digest(keys), counts=counts)
