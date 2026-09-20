"""Pre-resolved exposure streams, the replay order, the check cadence and the seeds.

Every arm at one parent seed consumes the **same ordered chosen stream**, so the
comparison between lambda values differs in the loss and in nothing else. That is
a property of a list of row IDs resolved before any fitting, not of two RNGs
agreeing at run time, so the list is built here, hashed, and written into the
input manifest the freeze binds.

Four things are fixed here rather than left to convention:

* **Batches are filled across cycle boundaries.** A cycle is one pass over the
  120,477 eligible chosen rows; 240,000 exposures is two of them with 954 rows to
  spare, and the boundary lands *inside* update 1,883. Dropping that partial tail
  and restarting at a cycle edge would make the declared 64-per-update exposure
  accounting false for one update in every cycle.
* **The task stream and the replay stream have independent, purpose-specific
  seeds.** The pairing seeds are inherited from the completed campaign
  (``20260924 + parent seed`` and ``20260925``); the replay and monitoring seeds
  are new literals chosen outside the band the historical draw seeds occupy, and
  :func:`require_disjoint_seeds` refuses a config that lands inside it.
* **The cadence is a function of the update number.** ``MonitorSchedule`` in
  :mod:`her2_guard` measures its interval *from the last check*, so
  ``first_update=1, update_interval=25`` produces 1, 26, 51, ... and it also
  requires a positive GPU-second trigger. This screen declares checks at 1, 25,
  50, ... matched on updates so replay cost cannot change how many checks happen,
  which is an amendment to the cadence and to nothing else: the statistic, the
  population and the 1.0 nat/sequence threshold are unchanged. The historical
  module is imported, never edited.
* **Nothing here uses ``hash()``.** Process-randomized hashing cannot appear
  anywhere near a seed: the same configuration would consume different rows on two
  runs and no artifact could be re-derived.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .her2_runtime import require
from .her2_support_paths import array_digest, digest_document

#: The band the historical guarded draw seeds occupy: ``20260921 +
#: sha256(name)[:8] % 100000`` (``scripts/posttrain_her2_guarded.py``). A "new"
#: seed like 20260930 lands inside it, would coincide with some historical draw
#: seed for some checkpoint name, and would make two banks that are described as
#: independent share their sampling stream.
HISTORICAL_SEED_BAND = (20260921, 20360920)


def require_disjoint_seeds(seeds, *, band=HISTORICAL_SEED_BAND, label="replay seeds"):
    """Refuse any declared seed that falls inside the historical draw-seed band."""
    low, high = int(band[0]), int(band[1])
    collisions = sorted({f"{name}={int(value)}" for name, value in dict(seeds).items()
                         if low <= int(value) <= high})
    require(not collisions,
            f"{label}: {collisions} fall inside the historical draw-seed band [{low}, {high}]. "
            "The historical guarded seeds are 20260921 + sha256(checkpoint name)[:8] % 100000, so "
            "a seed in that range can coincide with a bank this campaign describes as "
            "independently sampled. Choose literals outside the band.")
    duplicates = sorted({str(value) for value in dict(seeds).values()
                         if list(dict(seeds).values()).count(value) > 1})
    require(not duplicates,
            f"{label}: seed values {duplicates} are used for more than one purpose. Purpose-"
            "specific seeds are what make two streams independent; sharing one silently couples "
            "them.")
    return {"band": [low, high], "seeds": {str(k): int(v) for k, v in sorted(dict(seeds).items())},
            "rule": "declared literals, verified disjoint from the historical band"}


# ---------------------------------------------------------------------------
# the common ordered task stream
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskStream:
    """One parent seed's resolved chosen/rejected row order for the whole screen."""

    seed: int
    chosen_rows: np.ndarray
    rejected_rows: np.ndarray
    cycle_of_position: np.ndarray
    batch_rows: int
    chosen_population: int
    rejected_population: int
    pairing_seed: int

    @property
    def exposures(self):
        return int(self.chosen_rows.size)

    @property
    def updates(self):
        return self.exposures // int(self.batch_rows)

    def batch(self, update):
        """``(chosen_rows, rejected_rows)`` for a 1-based update number."""
        require(isinstance(update, (int, np.integer)) and 1 <= int(update) <= self.updates,
                f"Update {update!r} is outside 1..{self.updates} for this resolved stream")
        start = (int(update) - 1) * int(self.batch_rows)
        stop = start + int(self.batch_rows)
        return self.chosen_rows[start:stop], self.rejected_rows[start:stop]

    def cycle_boundary_updates(self):
        """The updates whose batch spans two cycles. Filled, never dropped."""
        spans = []
        for update in range(1, self.updates + 1):
            start = (update - 1) * int(self.batch_rows)
            block = self.cycle_of_position[start:start + int(self.batch_rows)]
            if block.size and int(block[0]) != int(block[-1]):
                spans.append(int(update))
        return spans

    def document(self):
        cycles, counts = np.unique(self.cycle_of_position, return_counts=True)
        return {"seed": int(self.seed), "pairing_seed": int(self.pairing_seed),
                "exposures": self.exposures, "updates": self.updates,
                "batch_rows": int(self.batch_rows),
                "chosen_population": int(self.chosen_population),
                "rejected_population": int(self.rejected_population),
                "cycles": {str(int(c)): int(n) for c, n in zip(cycles, counts)},
                "cycle_boundary_updates": self.cycle_boundary_updates(),
                "chosen_rows_sha256": array_digest(self.chosen_rows),
                "rejected_rows_sha256": array_digest(self.rejected_rows),
                "cycle_of_position_sha256": array_digest(self.cycle_of_position),
                "distinct_chosen_rows": int(np.unique(self.chosen_rows).size),
                "distinct_rejected_rows": int(np.unique(self.rejected_rows).size),
                "order": "position 0..N-1; update u consumes positions [(u-1)*batch, u*batch)",
                "note": ("every arm at this seed consumes exactly this order. IPO additionally "
                         "consumes the paired rejected row at the same position; continued SFT "
                         "consumes the chosen column only and performs no rejected inference.")}


def resolve_task_stream(pairing, *, seed, exposures, batch_rows, pairing_seed):
    """Materialize the ordered chosen/rejected rows for ``exposures`` examples.

    Built from the inherited :class:`her2_preferences.PreferencePairing`: its
    ``cycle_rows`` is the same seeded shuffle and the same distance-matched partner
    rotation the completed campaign used, so this screen inherits the eligibility
    and pairing rules rather than filtering a new positive population.

    ``exposures`` must divide by ``batch_rows``: a stream that ended mid-update
    would make the last update's declared 64 chosen examples untrue, and the
    endpoints are defined by exact exposure counts.
    """
    exposures, batch_rows = int(exposures), int(batch_rows)
    require(exposures > 0 and batch_rows > 0, "Exposures and batch rows must be positive")
    require(exposures % batch_rows == 0,
            f"{exposures} exposures do not divide into batches of {batch_rows}; the declared "
            "endpoints are exact exposure counts and a partial final update would not be one")
    chosen_parts, rejected_parts, cycle_parts = [], [], []
    filled, cycle = 0, 0
    while filled < exposures:
        order, partners = pairing.cycle_rows(cycle)
        take = min(int(order.size), exposures - filled)
        chosen_parts.append(np.asarray(order[:take], dtype=np.int64))
        rejected_parts.append(np.asarray(partners[:take], dtype=np.int64))
        cycle_parts.append(np.full(take, cycle, dtype=np.int32))
        filled += take
        cycle += 1
    chosen = np.concatenate(chosen_parts)
    rejected = np.concatenate(rejected_parts)
    cycles = np.concatenate(cycle_parts)
    require(chosen.size == exposures and rejected.size == exposures,
            "The resolved stream does not hold the declared number of exposures")
    population = pairing.population
    require(bool((population.chosen_distance[chosen]
                  == population.rejected_distance[rejected]).all()),
            "A resolved stream position pairs rows from different wild-type distance groups; the "
            "inherited matching is what removes the distance shortcut and is not relaxed here")
    return TaskStream(seed=int(seed), chosen_rows=chosen, rejected_rows=rejected,
                      cycle_of_position=cycles, batch_rows=batch_rows,
                      chosen_population=int(population.chosen_index.shape[0]),
                      rejected_population=int(population.rejected_index.shape[0]),
                      pairing_seed=int(pairing_seed))


def exposure_endpoints(updates, *, batch_rows):
    """``{updates: chosen exposures}`` for the declared endpoint update counts."""
    return {int(u): int(u) * int(batch_rows) for u in sorted(int(v) for v in updates)}


# ---------------------------------------------------------------------------
# the replay order
# ---------------------------------------------------------------------------

def replay_order(*, bank_rows, exposures, seed, parent_seed):
    """Row indices into one seed's replay bank, cycling through reshuffled full banks.

    Every lambda arm at one parent seed reads this same order, so two coefficients
    differ in their weight and not in which parent draws they saw. The stream is
    decoupled from the task stream: it has its own seed and its own cycle counter,
    and the task batches are identical whether or not replay is running.
    """
    bank_rows, exposures = int(bank_rows), int(exposures)
    require(bank_rows > 0 and exposures > 0, "Replay order needs positive counts")
    parts, filled, cycle = [], 0, 0
    while filled < exposures:
        permutation = np.random.default_rng([int(seed), int(parent_seed), cycle]).permutation(
            bank_rows)
        take = min(bank_rows, exposures - filled)
        parts.append(np.asarray(permutation[:take], dtype=np.int64))
        filled += take
        cycle += 1
    order = np.concatenate(parts)
    require(order.size == exposures, "The replay order does not hold the declared exposures")
    return order


def replay_order_document(order, *, bank_rows, seed, parent_seed, batch_rows):
    values = np.asarray(order)
    return {"rows": int(values.size), "bank_rows": int(bank_rows),
            "seed": int(seed), "parent_seed": int(parent_seed), "batch_rows": int(batch_rows),
            "cycles": int(np.ceil(values.size / float(bank_rows))),
            "distinct_rows": int(np.unique(values).size),
            "order_sha256": array_digest(values),
            "rule": ("independently shuffled full banks, concatenated and truncated. Duplicates "
                     "within a cycle are impossible by construction; a row appearing in two "
                     "cycles is the sampling law, not a defect.")}


def replay_batch_rows(order, update, *, batch_rows):
    """The replay rows one 1-based update consumes."""
    start = (int(update) - 1) * int(batch_rows)
    stop = start + int(batch_rows)
    require(stop <= int(np.asarray(order).size),
            f"Update {update} would read replay rows beyond the resolved order")
    return np.asarray(order)[start:stop]


# ---------------------------------------------------------------------------
# the check cadence
# ---------------------------------------------------------------------------

class UpdateCadence:
    """Checks at completed updates 1, 25, 50, ... plus every endpoint and the final state.

    A pure function of the update number, which is what makes it de-duplicating:
    an update that is both a multiple of the interval and a declared endpoint is
    one check with one reason, not two. The historical
    :class:`her2_guard.MonitorSchedule` measures its interval from the previous
    check and also requires a GPU-second trigger, so neither this cadence nor the
    removal of that trigger is expressible in it. It stays untouched.
    """

    def __init__(self, *, first_update=1, interval=25, endpoints=()):
        require(isinstance(first_update, int) and first_update >= 1,
                "The first check is at a positive update")
        require(isinstance(interval, int) and interval > 0, "The interval is a positive integer")
        self.first_update = int(first_update)
        self.interval = int(interval)
        self.endpoints = tuple(sorted(int(value) for value in endpoints))
        require(all(value > 0 for value in self.endpoints), "Endpoints are positive update counts")

    def due(self, update, *, final=False):
        """``(due, reason)`` for one completed update. At most one reason per update."""
        update = int(update)
        if update in self.endpoints:
            return True, "exposure_endpoint"
        if update == self.first_update:
            return True, "first_update"
        if update % self.interval == 0:
            return True, "update_interval"
        if final:
            return True, "final_state"
        return False, None

    def scheduled(self, total_updates):
        """Every update at which a check is due, in order, for a full-length run."""
        total = int(total_updates)
        return [update for update in range(1, total + 1)
                if self.due(update, final=(update == total))[0]]

    def document(self):
        return {"first_update": self.first_update, "update_interval": self.interval,
                "endpoints": list(self.endpoints),
                "also": ["every declared exposure endpoint", "the final completed update"],
                "duplicate_rule": ("the cadence is a function of the update number, so an update "
                                   "that is both an interval multiple and an endpoint produces "
                                   "exactly one check"),
                "amendment": ("this replaces the historical additional five-GPU-second trigger. "
                              "Matching the cadence on updates keeps the number of scheduled "
                              "checks independent of replay cost. The statistic, the population "
                              "and the threshold are unchanged."),
                "inherited_schedule_not_used": ("her2_guard.MonitorSchedule measures its interval "
                                                "from the last check, so first_update=1 with "
                                                "interval 25 yields 1, 26, 51, ...; it also "
                                                "requires gpu_second_interval > 0")}


# ---------------------------------------------------------------------------
# the resolved input manifest block
# ---------------------------------------------------------------------------

def stream_manifest(streams, *, endpoints, batch_rows, cadence, replay_orders=None):
    """The reviewable block the freeze binds: every stream identity, no arrays.

    Arrays live in the run directory as shards; what travels into the tracked
    manifest is their digests, their lengths and the rules that produced them.
    """
    document = {
        "schema_version": "her2-parent-replay/1",
        "record_kind": "resolved_streams",
        "batch_rows": int(batch_rows),
        "exposure_endpoints": {str(k): v for k, v in
                               exposure_endpoints(endpoints, batch_rows=batch_rows).items()},
        "cadence": cadence.document(),
        "task_streams": {str(int(seed)): stream.document()
                         for seed, stream in sorted(streams.items())},
        "replay_orders": {str(int(seed)): dict(block)
                          for seed, block in sorted((replay_orders or {}).items())},
        "note": ("resolved before any fit. The replay and monitoring banks are generated after "
                 "the training freeze, under the sampling rule, seeds and counts the freeze "
                 "binds; their completed draw-order digests are bound to the freeze afterwards.")}
    document["identity_sha256"] = digest_document(
        {key: value for key, value in document.items() if key != "identity_sha256"})
    return document
