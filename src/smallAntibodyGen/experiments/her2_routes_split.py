"""The sequence-only hard split, its freeze, the exclusion closure and the power arithmetic.

Everything here runs **before any outcome is opened**, which is the only thing that
makes the reserved panel reserved. The inputs are sorted sequences and a seed; the
outputs are a frozen assignment and a certificate. ``outcomes_opened`` in the
acquisition manifest is checked, and :func:`freeze_split` refuses to run if it is
true.

Five decisions, each with its reason:

* **The panel is the plan's: exactly 100 reserved test cores and exactly 100
  development cores, in four seeded batches of fifty, test chosen first.** An
  earlier revision enlarged the reserved set to "every remaining eligible core" on
  a power argument that was arithmetically wrong and that never counted what the
  enlargement costs. The arithmetic: Spearman's asymptotic ``SE = (1 - rho^2) /
  sqrt(n - 1)`` is 0.0754 at ``n = 100`` and 0.0419 at ``n = 322``; the 95%
  **half-width** is 1.96 times that, i.e. **0.148** and **0.082**. The plan's
  +0.05 ambition is therefore unresolvable for a single Spearman at *either* size,
  so enlarging buys nothing it was claimed to buy -- while the radius-2 purge
  around a larger held-out set removes novel adaptation cores that the audited
  100/100 design retains (1,934 after the radius exclusion at the audit seed).
  The ambition is resolvable, if at all, only on the **paired per-core
  difference**, whose width is measured by bootstrap rather than by this formula.
  :func:`spearman_half_width` and :func:`power_note` report the single-correlation
  arithmetic honestly, including where the answer is "no".
* **Every member of a batch is at least ``batch_separation`` edits from every
  member of every earlier batch** -- not merely from the earlier batch *seeds*.
  Separating the seeds while taking unfiltered nearest neighbours lets batch three
  sit one substitution from batch one's rim.
* **Batches are not families.** The algorithm has no radius cap and audited batch
  diameters were 6-10 edits. :func:`batch_diameters` measures them and every
  batched result is reported beside its measured diameter.
* **Novelty is measured against the FULL old display library**, not only its
  training split. A candidate that is one substitution from an old *validation*
  window is not novel to a parent that was selected on that validation split.
* **The exclusion closure covers every consumer and carries a radius.** A reserved
  core must not appear as a reconstruction target, a preference endpoint, a
  teacher query, an enumerated counterfactual, a synthetic candidate, a replay row
  or a post-training refresh candidate -- and neither may anything within
  ``radius`` edits of one. Shared *partial* masked contexts, including a fully
  masked window, stay allowed: that is a context and not an identity.

The distance work is vectorised. A candidate-versus-old-library check written as a
nested Python loop is 3,619 x 367,042 = 1.3 billion comparisons;
:func:`min_distance_to_library` uses the same exact deletion-mask search over
packed codes that ``her2_data.nearest_training_labels`` uses, so the cost is
``sum_k C(L, k)`` sorted lookups instead.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .her2_runtime import canonical_json, require

SPLIT_SCHEMA = "her2-routes-split/2"

TRAIN = "train"
DEVELOPMENT = "development"
RESERVED = "reserved"
EXCLUDED = "excluded"
ASSIGNMENTS = (TRAIN, DEVELOPMENT, RESERVED, EXCLUDED)

#: Every consumer the closure must cover. Named so a new one cannot be added
#: without either appearing here or failing :func:`closure_report`.
CONSUMERS = ("reconstruction_input", "reconstruction_target", "local_contrast_endpoint",
             "whole_window_target", "preference_chosen", "preference_rejected",
             "teacher_query", "enumerated_counterfactual", "synthetic_candidate",
             "replay_row", "posttrain_refresh", "observation_head_row")

#: Bits per symbol in the packed code. Five bits hold the twenty canonical
#: residues; ``length * BITS <= 64`` keeps the packing lossless in one uint64, so
#: equality of codes is equality of the kept positions and there is no collision.
CODE_BITS = 5


def window_residues(construct, sites):
    return "".join(construct.residue_at(site) for site in sites)


def window_hamming(left, right):
    require(len(left) == len(right),
            f"Window lengths {len(left)} and {len(right)} are not comparable")
    return sum(1 for a, b in zip(left, right) if a != b)


def stable_key(seed, text):
    """A seeded, order-independent tie-break key. Ties never depend on file order."""
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# vectorised distance
# ---------------------------------------------------------------------------

def encode_windows(windows, *, alphabet=None):
    """``(N, L)`` symbol indices plus the alphabet they were encoded against.

    Length and alphabet are validated per row rather than after joining: a 9-mer
    and an 11-mer cancel out in a total length check and the reshape then splits
    residues across a row boundary, producing windows that appear in neither input.
    """
    rows = [str(window) for window in windows]
    require(rows, "No windows to encode")
    length = len(rows[0])
    for position, window in enumerate(rows):
        require(len(window) == length,
                f"Window {position} has length {len(window)}, expected {length}")
        require(window.isascii(), f"Window {position} carries a non-ASCII character")
    symbols = alphabet if alphabet is not None else sorted({c for row in rows for c in row})
    require(len(symbols) <= 2 ** CODE_BITS,
            f"{len(symbols)} distinct symbols do not fit in {CODE_BITS} bits per position")
    require(length * CODE_BITS <= 64,
            f"A {length}-symbol window at {CODE_BITS} bits does not pack into one uint64")
    table = np.full(256, -1, dtype=np.int16)
    for index, symbol in enumerate(symbols):
        table[ord(symbol)] = index
    raw = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8).reshape(len(rows), length)
    encoded = table[raw]
    bad = np.flatnonzero((encoded < 0).any(axis=1))
    require(bad.size == 0,
            f"Window(s) {bad[:5].tolist()} carry a symbol outside the declared alphabet")
    return encoded.astype(np.int64), tuple(symbols)


def packed_codes(index, drop_positions=()):
    """Pack a window into one uint64, ``CODE_BITS`` per position, deleting some.

    Deleted positions contribute zero, which is exactly the "ignore this position"
    semantics an exact neighbour search needs.
    """
    values = np.asarray(index, dtype=np.int64)
    require(values.ndim == 2, "Expected (N, L) window indices")
    weights = (np.uint64(1) << (np.uint64(CODE_BITS)
                                * np.arange(values.shape[1], dtype=np.uint64))).copy()
    for position in drop_positions:
        weights[int(position)] = np.uint64(0)
    return (values.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def min_distance_to_library(queries, library, *, max_distance=2):
    """Exact minimum Hamming distance up to ``max_distance``; ``-1`` beyond it.

    The same deletion-mask argument ``her2_data.nearest_training_labels`` relies
    on: masks are applied in increasing size and a query leaves the pending set as
    soon as it matches, so a hit at size ``k`` is a pair at distance exactly ``k``.
    ``-1`` means "further than ``max_distance``", which is the only fact the
    novelty and purge rules need.
    """
    query_rows = [str(value) for value in queries]
    library_rows = [str(value) for value in library]
    if not library_rows:
        return np.full(len(query_rows), -1, dtype=np.int64)
    alphabet = sorted({c for row in query_rows + library_rows for c in row})
    encoded_queries, _ = encode_windows(query_rows, alphabet=alphabet)
    encoded_library, _ = encode_windows(library_rows, alphabet=alphabet)
    require(encoded_queries.shape[1] == encoded_library.shape[1],
            "Query and library windows have different lengths and are not comparable")
    length = encoded_queries.shape[1]
    require(0 <= int(max_distance) <= length, "Unsupported max_distance")
    out = np.full(encoded_queries.shape[0], -1, dtype=np.int64)
    pending = np.arange(encoded_queries.shape[0])
    for size in range(int(max_distance) + 1):
        if pending.size == 0:
            break
        found = np.zeros(pending.size, dtype=bool)
        for positions in itertools.combinations(range(length), size):
            keys = np.sort(packed_codes(encoded_library, positions))
            probe = packed_codes(encoded_queries[pending], positions)
            left = np.searchsorted(keys, probe, side="left")
            right = np.searchsorted(keys, probe, side="right")
            found |= right > left
        out[pending[found]] = size
        pending = pending[~found]
    return out


def deletion_keys(window, radius):
    """Every key obtained by blanking up to ``radius`` positions of ``window``.

    Two windows are within Hamming distance ``r`` **iff** they share a key with at
    most ``r`` blanks. That is the same argument :func:`min_distance_to_library`
    rests on, in a form a dictionary can answer, and it is what turns the greedy
    clustering of 100,000 generated windows from ``draws x medoids`` Hamming
    computations into ``sum_k C(L, k)`` lookups per window.
    """
    text = str(window)
    out = []
    for size in range(int(radius) + 1):
        for positions in itertools.combinations(range(len(text)), size):
            buffer = list(text)
            for position in positions:
                buffer[position] = "*"
            out.append("".join(buffer))
    return tuple(out)


def greedy_medoids(windows, *, radius=1, existing=()):
    """Greedy cover at ``radius`` over lexicographically sorted unique windows.

    Returns ``(new medoids, pre-existing medoids)``. Exact for the declared radius
    and linear in the number of windows at fixed length, which is what makes the
    frozen clustering computable at the plan's maximum draw count.
    """
    covered, existing_medoids = set(), []
    for window in sorted(set(str(value) for value in existing)):
        existing_medoids.append(window)
        covered.update(deletion_keys(window, radius))
    fresh = []
    for window in sorted(set(str(value) for value in windows)):
        keys = deletion_keys(window, radius)
        if any(key in covered for key in keys):
            continue
        fresh.append(window)
        covered.update(keys)
    return tuple(fresh), tuple(existing_medoids)


class HammingIndex:
    """Membership and nearest-distance queries up to a declared radius. Hash-based.

    ``nearest`` returns ``-1`` beyond the radius, the same convention
    :func:`min_distance_to_library` uses, so the two cannot disagree about what
    ``-1`` means.
    """

    def __init__(self, windows, *, radius=2):
        self.radius = int(radius)
        self.exact = set(str(window) for window in windows)
        self.levels = []
        for size in range(1, self.radius + 1):
            table = set()
            for window in self.exact:
                for positions in itertools.combinations(range(len(window)), size):
                    buffer = list(window)
                    for position in positions:
                        buffer[position] = "*"
                    table.add("".join(buffer))
            self.levels.append(table)

    def __contains__(self, window):
        return str(window) in self.exact

    def nearest(self, window):
        text = str(window)
        if text in self.exact:
            return 0
        for size, table in enumerate(self.levels, start=1):
            for positions in itertools.combinations(range(len(text)), size):
                buffer = list(text)
                for position in positions:
                    buffer[position] = "*"
                if "".join(buffer) in table:
                    return size
        return -1

    def document(self):
        return {"bank": len(self.exact), "radius": self.radius,
                "keys_per_level": [len(table) for table in self.levels],
                "convention": "-1 means further than the radius, as in min_distance_to_library"}


def assign_to_medoids(windows, medoids, *, radius=1):
    """Occupancy of a medoid set, nearest first with a lexicographic tie-break."""
    medoids = tuple(str(value) for value in medoids)
    tables = []
    for size in range(int(radius) + 1):
        table = {}
        for index, medoid in enumerate(medoids):
            for positions in itertools.combinations(range(len(medoid)), size):
                buffer = list(medoid)
                for position in positions:
                    buffer[position] = "*"
                key = "".join(buffer)
                if key not in table or medoids[table[key]] > medoid:
                    table[key] = index
        tables.append(table)
    counts, unassigned = {}, 0
    for window in windows:
        text = str(window)
        found = None
        for size, table in enumerate(tables):
            best = None
            for positions in itertools.combinations(range(len(text)), size):
                buffer = list(text)
                for position in positions:
                    buffer[position] = "*"
                index = table.get("".join(buffer))
                if index is not None and (best is None or medoids[index] < medoids[best]):
                    best = index
            if best is not None:
                found = best
                break
        if found is None:
            unassigned += 1
            continue
        counts[found] = counts.get(found, 0) + 1
    return {"clusters_occupied": len(counts), "clusters_total": len(medoids),
            "unassigned": unassigned,
            "counts": {str(key): int(value) for key, value in sorted(counts.items())}}


def distance_matrix(windows, *, chunk=512):
    """``(N, N)`` exact Hamming distances, computed in row chunks.

    Chunked because the naive broadcast allocates ``N * N * L`` booleans at once:
    at the campaign's 3,619 candidates that is 131 MB of temporary for a 13 MB
    answer, and it grows quadratically with any later pool.
    """
    encoded, _ = encode_windows(windows)
    rows = encoded.shape[0]
    out = np.empty((rows, rows), dtype=np.int16)
    for start in range(0, rows, int(chunk)):
        stop = min(rows, start + int(chunk))
        block = encoded[start:stop, None, :] != encoded[None, :, :]
        out[start:stop] = block.sum(axis=-1).astype(np.int16)
    return out


# ---------------------------------------------------------------------------
# power arithmetic
# ---------------------------------------------------------------------------

def spearman_half_width(rho, n, *, confidence=1.96):
    """Asymptotic 95% half-width of a single Spearman correlation.

    ``SE = (1 - rho**2) / sqrt(n - 1)``; the half-width is ``confidence * SE``.
    The two are reported separately because confusing them is how a panel gets
    resized: at ``rho = 0.5`` and ``n = 322`` the SE is 0.0419 and the half-width
    is 0.0821, and only the second is the quantity a +0.05 ambition is compared
    against.
    """
    rho, n = float(rho), int(n)
    require(n > 1, "A correlation interval needs at least two observations")
    standard_error = (1.0 - rho ** 2) / math.sqrt(n - 1)
    return {"rho": rho, "n": n, "standard_error": standard_error,
            "half_width": float(confidence) * standard_error,
            "confidence_multiplier": float(confidence),
            "formula": "SE = (1 - rho^2) / sqrt(n - 1); half_width = 1.96 * SE"}


def power_note(*, target, rho, sizes):
    """The resolvability table this campaign publishes beside its ambition target."""
    rows = [dict(spearman_half_width(rho, size), target=float(target),
                 resolvable=bool(float(target) > spearman_half_width(rho, size)["half_width"]))
            for size in sorted(set(int(size) for size in sizes))]
    return {"target_absolute_spearman": float(target), "assumed_rho": float(rho),
            "rows": rows,
            "any_size_resolvable": any(row["resolvable"] for row in rows),
            "finding": ("an ambition threshold smaller than the half-width at a panel's size is "
                        "not resolvable AS A SINGLE CORRELATION on that panel. At rho ~ 0.5 the "
                        "half-width is 0.148 at n=100 and 0.082 at n=322, so +0.05 is out of "
                        "reach at both: enlarging the panel does not buy resolvability, and the "
                        "earlier amendment that enlarged it on this arithmetic was wrong."),
            "primary_endpoint": ("the PAIRED per-core difference against the frozen control, "
                                 "bootstrapped over cores. The arms see identical cores, so the "
                                 "between-arm variance is far smaller than either arm's own "
                                 "sampling variance and the formula above does NOT bound it. Its "
                                 "width is measured, not assumed."),
            "retention": ("the reserved panel also costs training data through the radius-2 "
                          "purge around it. A design that enlarges the held-out set without "
                          "counting that cost is not a free improvement.")}


# ---------------------------------------------------------------------------
# the batched selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """The seeded batched selection, with its measured geometry."""

    selected: tuple
    batches: tuple
    diameters: tuple
    exhausted: bool
    eligible: int
    roles: tuple = ()

    def document(self):
        return {"selected": len(self.selected), "batches": len(self.batches),
                "batch_sizes": [len(batch) for batch in self.batches],
                "batch_roles": list(self.roles),
                "batch_diameters": [int(value) for value in self.diameters],
                "max_batch_diameter": int(max(self.diameters)) if self.diameters else None,
                "exhausted_before_target": bool(self.exhausted),
                "eligible_pool": int(self.eligible),
                "separation_rule": ("every member of a batch is at least the declared separation "
                                    "from every member of every EARLIER batch, not only from "
                                    "their seeds"),
                "diameter_caveat": ("the algorithm has no radius cap, so a batch is a seeded "
                                    "neighbourhood and not a biological family. Audited diameters "
                                    "were 6-10 edits. No family-level claim rests on a batch.")}


def batch_diameters(batches):
    """Max pairwise Hamming inside each batch. The number that refutes 'family'."""
    out = []
    for batch in batches:
        if len(batch) < 2:
            out.append(0)
            continue
        matrix = distance_matrix(list(batch))
        out.append(int(matrix.max()))
    return tuple(out)


def select_batches(candidates, *, batches, batch_size, seed, separation, roles=None,
                   previous=()):
    """``batches`` seeded neighbourhoods of ``batch_size``, separated from each other.

    Pick a seed core in seeded order; take its nearest eligible neighbours to fill
    the batch. Eligibility for **every** member of batch ``i`` is: at least
    ``separation`` edits from every member already selected in batches ``< i`` (and
    from ``previous``, which carries a prior stage's selections). Ties are broken by
    :func:`stable_key`, so the result depends on the seed and the *set* of
    candidates, never on the order they arrived in.
    """
    require(int(batches) > 0 and int(batch_size) > 0, "Batch count and size must be positive")
    pool = sorted(set(str(core) for core in candidates))
    require(pool, "No candidate cores to select from")
    order = sorted(range(len(pool)), key=lambda i: stable_key(seed, pool[i]))
    matrix = distance_matrix(pool)
    prior = [str(core) for core in previous]
    # Distance from every candidate to the already-selected set, maintained
    # incrementally so the eligibility test is a lookup and not a rescan.
    if prior:
        nearest = np.asarray(min_distance_to_library(pool, prior,
                                                     max_distance=int(separation)))
        nearest = np.where(nearest < 0, int(separation), nearest)
    else:
        nearest = np.full(len(pool), int(separation), dtype=np.int64)
    taken_rows, out_batches, role_names = [], [], []
    names = list(roles) if roles is not None else [f"batch{i}" for i in range(int(batches))]
    require(len(names) == int(batches), f"{len(names)} role names for {batches} batches")
    selected_rows = set()
    for batch_index in range(int(batches)):
        allowed = np.flatnonzero((nearest >= int(separation)))
        allowed = np.asarray([row for row in allowed if row not in selected_rows])
        if allowed.size == 0:
            break
        allowed_set = set(int(row) for row in allowed)
        centre = next((row for row in order if row in allowed_set), None)
        if centre is None:
            break
        neighbours = sorted((row for row in allowed_set if row != centre),
                            key=lambda row: (int(matrix[centre, row]), stable_key(seed, pool[row])))
        batch_rows = [centre] + neighbours[:int(batch_size) - 1]
        if len(batch_rows) < int(batch_size):
            # Report the shortfall rather than padding from an ineligible core.
            out_batches.append(tuple(pool[row] for row in batch_rows))
            role_names.append(names[batch_index])
            taken_rows.extend(batch_rows)
            break
        out_batches.append(tuple(pool[row] for row in batch_rows))
        role_names.append(names[batch_index])
        taken_rows.extend(batch_rows)
        selected_rows.update(batch_rows)
        # Every later batch must clear the separation from ALL of these members.
        nearest = np.minimum(nearest, matrix[batch_rows, :].min(axis=0))
    selected = tuple(pool[row] for row in taken_rows)
    return SelectionResult(selected=selected, batches=tuple(out_batches),
                           diameters=batch_diameters(out_batches),
                           exhausted=len(selected) < int(batches) * int(batch_size),
                           eligible=len(pool), roles=tuple(role_names))


def select_batched(candidates, *, target, batch_size, seed, separation, previous=()):
    """``target`` cores in ``ceil(target / batch_size)`` batches. Thin wrapper."""
    require(int(target) > 0, "Target must be positive")
    count = int(math.ceil(int(target) / int(batch_size)))
    result = select_batches(candidates, batches=count, batch_size=int(batch_size), seed=seed,
                            separation=separation, previous=previous)
    selected = result.selected[:int(target)]
    return SelectionResult(selected=selected, batches=result.batches,
                           diameters=result.diameters,
                           exhausted=len(selected) < int(target), eligible=result.eligible,
                           roles=result.roles)


# ---------------------------------------------------------------------------
# the hard split
# ---------------------------------------------------------------------------

@dataclass
class HardSplit:
    """A sequence-only assignment of candidate windows, plus every exclusion reason."""

    assignment: dict
    development: tuple
    reserved: tuple
    excluded: dict
    selection: SelectionResult
    settings: dict
    counts: dict = field(default_factory=dict)

    def document(self):
        return {"schema_version": SPLIT_SCHEMA, "counts": dict(self.counts),
                "development": len(self.development), "reserved": len(self.reserved),
                "excluded": {reason: len(cores) for reason, cores in
                             sorted(self.excluded.items())},
                "selection": self.selection.document(), "settings": dict(self.settings)}


def build_hard_split(*, candidates, old_library, settings, old_training=None):
    """Freeze the development and reserved populations from sequences alone.

    The pipeline, in the order the plan specifies:

    1. **panel eligibility** -- a *panel candidate* must sit at least
       ``novel_separation`` edits from every window in the **full old display
       library**. ``old_training`` alone is not enough: a parent selected on the
       old validation split has seen those windows too. This rule makes the PANEL
       hard; it does not remove anything from training, and conflating the two is
       how a design loses most of its adaptation data for no benefit.
    2. **selection** -- ``test_batches + development_batches`` seeded batches of
       ``batch_size``, **test first**, every member of a batch at least
       ``batch_separation`` edits from every member of every earlier batch.
    3. **radius exclusion** -- every remaining candidate within ``purge_radius``
       of any selected core is excluded from training, not moved into a panel.
       This applies to the whole candidate population, not only to the
       panel-eligible part of it.
    4. **the rest stay trainable**, including candidates that were too near the
       old library to be panel candidates. The retained count is reported: it is
       the thing a larger held-out panel spends, and it is measured.
    """
    required = ("novel_separation", "purge_radius", "test_target", "development_target",
                "batch_size", "seed", "batch_separation", "minimum_reserved")
    for key in required:
        require(key in settings, f"The split settings are missing {key!r}")
    novel = int(settings["novel_separation"])
    radius = int(settings["purge_radius"])
    test_target = int(settings["test_target"])
    development_target = int(settings["development_target"])
    batch_size = int(settings["batch_size"])
    require(test_target % batch_size == 0 and development_target % batch_size == 0,
            f"{test_target} test and {development_target} development cores do not divide into "
            f"batches of {batch_size}; a partial batch is a different algorithm")
    pool = sorted(set(str(core) for core in candidates))
    library = sorted(set(str(core) for core in (old_library if old_library is not None
                                                else old_training or ())))
    require(pool, "No candidate windows")

    excluded = {}
    nearest_old = min_distance_to_library(pool, library, max_distance=max(0, novel - 1))
    panel_eligible, not_panel_eligible = [], []
    for core, distance in zip(pool, nearest_old):
        (not_panel_eligible if distance >= 0 else panel_eligible).append(core)
    require(panel_eligible,
            f"No candidate sits at least {novel} edits from the {len(library)} old library "
            "windows. The hard panel does not exist under these settings, and relaxing the "
            "separation after seeing that would be choosing the difficulty from the result.")

    test_batches = test_target // batch_size
    development_batches = development_target // batch_size
    selection = select_batches(
        panel_eligible, batches=test_batches + development_batches, batch_size=batch_size,
        seed=int(settings["seed"]), separation=int(settings["batch_separation"]),
        roles=[RESERVED] * test_batches + [DEVELOPMENT] * development_batches)
    reserved, development = [], []
    for role, batch in zip(selection.roles, selection.batches):
        (reserved if role == RESERVED else development).extend(batch)
    reserved, development = tuple(reserved), tuple(development)
    held_out = list(reserved) + list(development)
    require(held_out, "The batched selection produced no held-out cores")

    # The purge runs over the WHOLE candidate population, not only the
    # panel-eligible part: a core that was too near the old library to be a panel
    # candidate is still a leak if it sits within the radius of a held-out core.
    remaining = [core for core in pool if core not in set(held_out)]
    purged, retained = [], []
    if remaining:
        distances = min_distance_to_library(remaining, held_out, max_distance=radius)
        for core, distance in zip(remaining, distances):
            (purged if distance >= 0 else retained).append(core)
    if purged:
        excluded[f"within_{radius}_edits_of_a_held_out_core"] = sorted(purged)

    assignment = {core: TRAIN for core in sorted(retained)}
    assignment.update({core: EXCLUDED for cores in excluded.values() for core in cores})
    assignment.update({core: DEVELOPMENT for core in development})
    assignment.update({core: RESERVED for core in reserved})
    purged_set = set(purged)
    counts = {"candidates": len(pool), "old_library": len(library),
              "panel_eligible": len(panel_eligible),
              "not_panel_eligible": len(not_panel_eligible),
              "not_panel_eligible_retained_for_training": sum(
                  1 for core in not_panel_eligible if core not in purged_set),
              "development": len(development), "reserved": len(reserved),
              "retained_for_training": len(retained),
              "purged_within_radius": len(purged),
              "excluded": sum(len(cores) for cores in excluded.values()),
              "minimum_reserved": int(settings["minimum_reserved"]),
              "reserved_meets_minimum": len(reserved) >= int(settings["minimum_reserved"]),
              "selection_complete": (len(reserved) == test_target
                                     and len(development) == development_target),
              "eligibility_rule": (f"panel candidates are >= {novel} edits from the full old "
                                   "library; training retention is the whole candidate pool "
                                   f"minus the held-out cores and everything within {radius} "
                                   "edits of one")}
    return HardSplit(assignment=assignment, development=development,
                     reserved=reserved, excluded=excluded, selection=selection,
                     settings=dict(settings), counts=counts)


def split_certificate(split, *, rho=0.5, target=0.05):
    """Zero-violation certificate plus the measured power of the reserved endpoint."""
    violations = []
    for core in split.development:
        if split.assignment.get(core) != DEVELOPMENT:
            violations.append({"core": core, "problem": "development core is not assigned"})
    separation = int(split.settings["batch_separation"])
    if split.reserved and split.development:
        crossing = min_distance_to_library(list(split.reserved), list(split.development),
                                           max_distance=separation - 1)
        for core, distance in zip(split.reserved, crossing):
            if distance >= 0:
                violations.append({"core": core, "distance": int(distance),
                                   "problem": "reserved core is too near a development core"})
    overlap = sorted(set(split.development) & set(split.reserved))
    if overlap:
        violations.append({"problem": "a core is both development and reserved",
                           "cores": overlap[:10]})
    if not split.counts.get("selection_complete", True):
        violations.append({"problem": ("the batched selection could not fill the declared panel "
                                       "sizes; the shortfall is reported rather than padded"),
                           "counts": {"reserved": len(split.reserved),
                                      "development": len(split.development)}})
    # A panel of fewer than two cores has no correlation interval at all. Reporting
    # that as an omitted size is the answer; asking for one would raise here and
    # hide the shortfall the violations above are about.
    sizes = [size for size in (len(split.development), len(split.reserved), 152) if size > 1]
    power = dict(power_note(target=target, rho=rho, sizes=sizes or [152]),
                 omitted_sizes=[size for size in (len(split.development), len(split.reserved))
                                if size <= 1])
    return {"schema_version": SPLIT_SCHEMA, "record_kind": "split_certificate",
            "zero_violations": not violations, "violations": violations,
            "counts": dict(split.counts), "power": power,
            "reserved_status": ("at the declared size"
                                if split.counts["reserved_meets_minimum"]
                                else f"UNDERPOWERED: {len(split.reserved)} reserved cores against "
                                     f"a declared minimum of {split.settings['minimum_reserved']}. "
                                     "The endpoint is reported with its actual n and this "
                                     "arithmetic, not relaxed."),
            "reserved_means": ("unopened BY US. It is public antibody sequence data on GitHub, and "
                               "upstream pretraining exposure for p-IgGen and AbLang2 is "
                               "unresolved (docs/EVALUATION_INDEPENDENCE.md). Both facts stay in "
                               "the report.")}


def freeze_split(split, *, sequence_digest, algorithm_digest, acquisition_manifest,
                 previous=None):
    """The pre-outcome freeze record. Refuses to run once outcomes have been opened.

    ``previous`` is an already-written freeze record. A second freeze that
    disagrees with it is refused rather than overwritten: a re-freeze is how a
    panel quietly becomes the one that suited the result.
    """
    opened = bool(dict(acquisition_manifest).get("outcomes_opened"))
    require(not opened,
            "The acquisition manifest records outcomes_opened=true. A split frozen after outcomes "
            "were read is not a reserved population, and re-freezing would launder that.")
    record = {"schema_version": SPLIT_SCHEMA, "record_kind": "hard_split_freeze",
              "settings": dict(split.settings), "counts": dict(split.counts),
              "sequence_sha256": str(sequence_digest),
              "algorithm_sha256": str(algorithm_digest),
              "development": list(split.development), "reserved": list(split.reserved),
              "excluded": {reason: sorted(cores)
                           for reason, cores in sorted(split.excluded.items())},
              "selection": split.selection.document(),
              "acquisition": {"schema": dict(acquisition_manifest).get("schema"),
                              "retrieved_utc": dict(acquisition_manifest).get("retrieved_utc"),
                              "outcomes_opened": opened,
                              "pins": dict(dict(acquisition_manifest).get("pins") or {})},
              "order": ("sorted sequence inputs, their digest, the algorithm digest and the seed "
                        "are fixed here. No outcome column has been materialised at this point.")}
    if previous is not None:
        differing = sorted(key for key in ("sequence_sha256", "algorithm_sha256", "development",
                                           "reserved")
                           if dict(previous).get(key) != record[key])
        require(not differing,
                f"A frozen split already exists and this one differs in {differing}. The frozen "
                "panel is the panel; re-freezing silently is how a held-out set becomes the one "
                "that suited the answer. Declare a new revision instead.")
        record["reconfirmed"] = True
    return record


def algorithm_digest(settings, *, functions=("select_batches", "build_hard_split")):
    """A digest over the settings and the names of the functions that consumed them."""
    return hashlib.sha256(canonical_json({"settings": dict(settings),
                                          "functions": list(functions),
                                          "schema": SPLIT_SCHEMA}).encode("utf-8")).hexdigest()


def sequence_digest(cores):
    """A digest over the sorted candidate windows. Order-independent by construction."""
    joined = "\n".join(sorted(set(str(core) for core in cores)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the exclusion closure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExclusionClosure:
    """The identities no consumer may materialise, and the reason for each.

    ``radius`` is the load-bearing addition. Refusing only the exact held-out
    identities leaves a one-substitution neighbour of a reserved core available to
    every adaptation consumer, which is the leak the split's own purge step exists
    to prevent. ``radius=0`` is the identity-only closure and is still available
    for the places that genuinely want it.
    """

    windows: frozenset
    identities: frozenset
    reasons: dict
    radius: int = 0
    guarded: tuple = ()

    @classmethod
    def build(cls, *, reserved_windows=(), development_windows=(), reserved_identities=(),
              radius=0, extra=None):
        reasons = {}
        for window in reserved_windows:
            reasons[str(window)] = "reserved test window"
        for window in development_windows:
            reasons[str(window)] = "frozen development window"
        for identity in reserved_identities:
            reasons[str(identity)] = "reserved construct identity"
        for key, reason in dict(extra or {}).items():
            reasons[str(key)] = str(reason)
        guarded = tuple(sorted({str(w) for w in reserved_windows}
                               | {str(w) for w in development_windows}))
        return cls(windows=frozenset(guarded),
                   identities=frozenset(str(i) for i in reserved_identities),
                   reasons=reasons, radius=int(radius), guarded=guarded)

    @cached_property
    def _window_index(self):
        return HammingIndex(self.guarded, radius=self.radius)

    def nearest_held_out(self, window):
        """Exact distance to the nearest held-out window, or ``-1`` beyond ``radius``.

        A window of a different length is **not comparable** and returns ``-1``. The
        guarded set is the HER2 edit window; the campaign also enrols a 16-site CR9114
        genotype, a 20-site Adams window and a 68-site AlphaSeq region, and a Hamming
        distance between two windows of different lengths is not defined. Asking for
        one raised inside ``encode_windows``, which would have made every non-HER2 row
        a crash rather than an allowed one -- the radius rule protects the HER2 panel
        and cannot say anything about a different edit space.
        """
        if not self.guarded or self.radius <= 0:
            return -1
        text = str(window)
        if len(text) != len(self.guarded[0]):
            return -1
        return self._window_index.nearest(text)

    def violates(self, *, window=None, identity=None):
        if window is not None and str(window) in self.windows:
            return f"window is {self.reasons.get(str(window), 'reserved')}"
        if identity is not None and str(identity) in self.identities:
            return f"identity is {self.reasons.get(str(identity), 'reserved')}"
        if window is not None and self.radius > 0:
            distance = self.nearest_held_out(window)
            if distance >= 0:
                return (f"window is within {distance} edit(s) of a held-out core, inside the "
                        f"declared radius {self.radius}")
        return None

    def require_allowed(self, *, consumer, window=None, identity=None):
        require(consumer in CONSUMERS,
                f"{consumer!r} is not a declared consumer. Add it to CONSUMERS so the closure "
                "covers it; an unnamed consumer is one the audit cannot check.")
        reason = self.violates(window=window, identity=identity)
        require(reason is None,
                f"{consumer}: refused because the {reason}. The exclusion applies to every "
                "consumer -- reconstruction targets, pair endpoints, teacher queries, enumerated "
                "counterfactuals, synthetic candidates, replay and post-training refresh -- not "
                "only to the training rows.")
        return True

    def document(self):
        return {"reserved_windows": len(self.windows), "reserved_identities": len(self.identities),
                "radius": int(self.radius), "consumers": list(CONSUMERS),
                "guarded_window_length": (len(self.guarded[0]) if self.guarded else None),
                "radius_rule": ("a window within the declared radius of ANY held-out core is "
                                "refused for every consumer. Exact-identity exclusion alone "
                                "leaves a one-substitution neighbour available to the very "
                                "adaptation paths the purge exists to protect."),
                "length_rule": ("the radius rule applies to windows of the GUARDED length only. "
                                "A 16-site CR9114 genotype and a 68-site AlphaSeq region are "
                                "different edit spaces, and a Hamming distance between windows of "
                                "different lengths is not defined; those rows are protected by "
                                "their own source's declared partition instead."),
                "allowed": ("a shared PARTIAL masked context, including a fully masked edit "
                            "window, is allowed: that is a context, not an identity. Inference "
                            "restores the full canonical support.")}


def closure_report(closure, batches):
    """Audit a set of ``(consumer, window, identity)`` triples against the closure."""
    checked, violations = 0, []
    for consumer, window, identity in batches:
        checked += 1
        reason = closure.violates(window=window, identity=identity)
        if reason is not None:
            violations.append({"consumer": consumer, "reason": reason,
                               "window": window, "identity": None if identity is None
                               else str(identity)[:32]})
    covered = sorted({consumer for consumer, _, _ in batches})
    return {"schema_version": SPLIT_SCHEMA, "record_kind": "closure_report",
            "checked": checked, "violations": violations, "passed": not violations,
            "radius": int(closure.radius),
            "consumers_covered": covered,
            "consumers_unaudited": sorted(set(CONSUMERS) - set(covered)),
            "unaudited_meaning": ("a declared consumer with no audited triples is reported here. "
                                 "An empty audit is not a passing audit.")}


# ---------------------------------------------------------------------------
# CR9114 reserved genotypes, derived without reading H1
# ---------------------------------------------------------------------------

#: Columns the eligibility whitelist may be read for. ``split`` says train or
#: development; the H1 replicate columns that defined eligibility are not here and
#: are never opened.
CR9114_WHITELIST_COLUMNS = ("genotype", "split")


def cr9114_eligibility_whitelist(path, *, expected_total=None, expected_sha256=None):
    """Read the pinned eligibility whitelist. Two columns, no outcome.

    The whitelist is an explicit identity list, which is the only way to protect
    the historical reserved genotypes: their eligibility was itself defined by H1
    replicate support, so it cannot be recomputed without reading H1. The
    selection bias that introduces is disclosed in the record rather than
    described as neutral.
    """
    import csv
    from pathlib import Path
    from .her2_runtime import sha256
    target = Path(path)
    require(target.is_file(), f"The CR9114 eligibility whitelist {target} is missing")
    digest = sha256(target)
    if expected_sha256 is not None:
        require(digest == str(expected_sha256),
                f"{target.name} hashes to {digest}, the pinned whitelist is {expected_sha256}")
    eligible, by_split = {}, {}
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = [name for name in CR9114_WHITELIST_COLUMNS if name not in fields]
        require(not missing, f"{target.name} has no {missing} column(s); it has {fields}")
        forbidden = sorted(name for name in fields if name.lower().startswith("h1"))
    # DictReader would materialise every outcome cell before picking the two
    # identity columns. Restrict parsing itself, as for the raw assay release.
    import pandas as pd
    frame = pd.read_csv(target, usecols=list(CR9114_WHITELIST_COLUMNS),
                        dtype={"genotype": "string", "split": "string"})
    for genotype, split in zip(frame.genotype, frame.split):
        genotype, split = str(genotype).strip(), str(split).strip()
        require(genotype not in eligible, f"Duplicate whitelist genotype {genotype}")
        eligible[genotype] = split
        by_split[split] = by_split.get(split, 0) + 1
    if expected_total is not None:
        require(len(eligible) == int(expected_total),
                f"{target.name} lists {len(eligible)} eligible genotypes, the pin says "
                f"{expected_total}")
    return {"eligible": eligible, "count": len(eligible), "by_split": dict(sorted(by_split.items())),
            "sha256": digest, "columns_read": list(CR9114_WHITELIST_COLUMNS),
            "h1_columns_present_but_unread": forbidden,
            "selection_bias": ("this whitelist was historically selected on H1 assay eligibility. "
                               "Training on its complement's complement therefore inherits that "
                               "selection, and any transfer claim from this population carries "
                               "the bias. Disclosed, not corrected.")}


def cr9114_protected_genotypes(whitelist, genotypes, *, expected_protected=None,
                               expected_sha256=None):
    """The protected complement: every genotype the whitelist does not list.

    Conservative by construction. The complement contains the historical reserved
    block *and* every genotype that failed eligibility, and the two cannot be
    separated without reading H1. Protecting the union is the direction that
    cannot leak.
    """
    eligible = dict(dict(whitelist)["eligible"])
    protected = sorted(str(value) for value in genotypes if str(value) not in eligible)
    digest = hashlib.sha256(("\n".join(protected) + "\n").encode("utf-8")).hexdigest()
    if expected_protected is not None:
        require(len(protected) == int(expected_protected),
                f"{len(protected)} protected genotypes against a pinned {expected_protected}")
    if expected_sha256 is not None:
        require(digest == str(expected_sha256),
                f"The protected complement hashes to {digest}, the pin says {expected_sha256}")
    return {"protected": tuple(protected), "count": len(protected),
            "sorted_newline_sha256": digest,
            "eligible_count": len(eligible),
            "rule": ("protect the COMPLEMENT of the eligibility whitelist, not a re-derived "
                     "reserved block. The complement is the union of the historical reserved "
                     "genotypes and the eligibility rejects, and separating them would require "
                     "the H1 columns this campaign never reads.")}


def cr9114_reserved_genotypes(genotypes, *, seed, loci_count, train_blocks, development_blocks):
    """Re-derive the pilot's sequence-only block assignment. A SECONDARY record.

    Kept because it documents how the historical split was produced and consumes
    no label. It is **not** the operative protection: eligibility itself was
    defined by H1 replicate support, so only the explicit whitelist complement
    (:func:`cr9114_protected_genotypes`) is conservative.
    """
    rng = np.random.default_rng(int(seed))
    loci = sorted(rng.choice(16, size=int(loci_count), replace=False).tolist())
    blocks = rng.permutation(2 ** len(loci)).tolist()
    ntrain, ndev = int(train_blocks), int(development_blocks)
    require(ntrain > 0 and ndev > 0 and ntrain + ndev < len(blocks),
            "Invalid block counts for the CR9114 sequence-only split")
    names = {block: (TRAIN if position < ntrain else
                     DEVELOPMENT if position < ntrain + ndev else RESERVED)
             for position, block in enumerate(blocks)}
    assignment = {}
    for genotype in genotypes:
        text = str(genotype)
        require(len(text) == 16 and set(text) <= {"0", "1"},
                f"{text!r} is not a 16-bit CR9114 genotype")
        assignment[text] = names[int("".join(text[i] for i in loci), 2)]
    reserved = sorted(g for g, name in assignment.items() if name == RESERVED)
    return {"assignment": assignment, "reserved": tuple(reserved),
            "loci_0based": loci, "blocks": {str(k): v for k, v in sorted(names.items())},
            "seed": int(seed),
            "counts": {name: sum(1 for v in assignment.values() if v == name)
                       for name in (TRAIN, DEVELOPMENT, RESERVED)},
            "derivation": ("reproduced from the seed and the block counts alone. No H1 column is "
                           "read."),
            "status": ("SECONDARY. The operative protection is the complement of the pinned "
                       "eligibility whitelist; this reproduction cannot see the eligibility "
                       "rejects, which is why it is not used on its own."),
            "complement_of_eligible": {
                "resolved": True,
                "resolved_by": "cr9114_protected_genotypes against the pinned whitelist",
                "reason": ("eligibility is defined by H1 replicate support, which this campaign "
                           "cannot read. The explicit whitelist supplies the identities instead, "
                           "and its complement is protected."),
                "policy": "exclude the reserved blocks AND any genotype not positively eligible"}}
