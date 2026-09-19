"""The parent-relative likelihood gate: its reference, its threshold and its cadence.

The question this answers is the one the original campaign could not: *is the
continuation still producing the parent's chosen sequences?* DPO's aggregate
validation AP went up while its chosen NLL went from ~1.49 to ~3.2 nats/residue,
and nothing stopped it. So before any update, the selected SFT parent scores the
**ordered fixed validation pairs**, chosen *and* rejected, and those vectors are
frozen, hashed and bound to the parent's identity. Thereafter

``D = mean(parent_chosen - current_chosen)``

is measured in nats **per sequence** (ten positions); the threshold is 1.0
nat/sequence, which is 0.1 nats/residue, and both numbers are printed everywhere
so a per-sequence D is never compared against a per-residue bound.

Fixed here rather than left to convention:

* **One gate, one identity, the full pair set, every check.** There is no monitor
  subset, no prefix, no tier and no population knob: the same 25,722 ordered pairs
  are scored on both sides at every check, and the rows handed to a check are
  re-hashed against the reference identity *before* they are scored
  (:func:`verify_pairs_against_identity`), so a reordered or re-sliced pair set is
  refused rather than scored and stamped with the old identity. Scoring cost is
  *measured* -- GPU and wall seconds per check -- and reported under
  ``monitor_cost``; it is never charged to the training clock and never netted
  against it.
* **Strict ``>`` stops.** Exact equality passes. A non-finite score stops
  regardless of D, and records JSON-safe metadata (``None`` plus counts) rather
  than writing a NaN into an artifact that ``allow_nan=False`` would then refuse.
* **Checks run between updates.** The caller runs them after ``optimizer.step()``
  and ``scheduler.step()`` complete and before the next update begins, outside
  every charged segment.
* **No threshold ladder, no per-run adjustment.** A run stopped at 1.0 cannot
  observe what training past 1.0 would have done, and this module exposes no path
  that pretends otherwise. What it does keep is the full observed D distribution
  per check: mean, seven quantiles, and the per-pair vectors persisted with their
  identity, which supports honest retrospective description and nothing
  counterfactual.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .her2_data import CORE_LENGTH
from .her2_objectives import CORE_POSITIONS
from .her2_preferences import (PROBABILITY_CONVENTION, array_digest, core_digest, score_sequences)
from .her2_runtime import load_json, require, save_json

PARENT_REFERENCE_SCHEMA = "her2-parent-validation-reference/1"
#: The population marker. It is part of the identity so that the *training*
#: reference -- same parent weights, different rows -- cannot be substituted for
#: the validation one without the mismatch being named.
GATE_POPULATION = "fixed_validation_pairs"

#: nats per SEQUENCE over the ten core positions. 1.0 / 10 = 0.1 nats/residue.
DEFAULT_THRESHOLD_NATS_PER_SEQUENCE = 1.0
#: Cadence: after update 1, then whichever of these comes first, plus every
#: nominal-budget crossing and the final state.
DEFAULT_UPDATE_INTERVAL = 25
DEFAULT_GPU_SECOND_INTERVAL = 5.0

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)

#: How far a recomputed D may sit from the journalled one before the evidence is
#: refused. Both are float64 means over the same two vectors and JSON round-trips
#: a float exactly, so this is a rounding tolerance and nothing more.
GATE_RECOMPUTATION_ATOL = 1e-9


def json_number(value):
    """A float that ``canonical_json`` will accept, or ``None`` if it would not.

    ``save_json`` runs with ``allow_nan=False`` on purpose: a NaN in an artifact
    is read later as a number. So every measured quantity that *can* come out
    non-finite passes through here, and the artifact says ``null`` with a reason
    beside it instead of failing to be written at all.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def quantile_summary(values, *, quantiles=QUANTILES):
    """Mean plus the seven declared quantiles of a vector, JSON-safe.

    Non-finite entries are counted rather than silently propagated into a NaN
    mean, because "the mean is NaN" and "three of 25,722 pairs are NaN" are
    different facts and only the second one is actionable.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(array)
    document = {"count": int(array.size), "nonfinite": int((~finite).sum())}
    if not finite.any():
        document.update(mean=None, quantiles={str(q): None for q in quantiles})
        return document
    usable = array[finite]
    document["mean"] = json_number(usable.mean())
    document["quantiles"] = {str(q): json_number(np.quantile(usable, q)) for q in quantiles}
    return document


# ---------------------------------------------------------------------------
# the frozen parent scores on the fixed validation pairs
# ---------------------------------------------------------------------------

def parent_reference_identity(*, parent_checkpoint_sha256, parent_state_sha256, config_sha256,
                              scaffold_prefix, chosen_index, rejected_index,
                              convention=PROBABILITY_CONVENTION):
    """Everything that would change the parent's numbers, in one comparable block.

    Both core orders are hashed separately: a re-sorted or re-sliced pair set is a
    different measurement even when the two multisets agree, because D is a mean
    of *paired* differences taken row by row.
    """
    chosen = np.asarray(chosen_index)
    rejected = np.asarray(rejected_index)
    require(chosen.shape == rejected.shape,
            f"The fixed validation pairs must be aligned: {chosen.shape} vs {rejected.shape}")
    require(chosen.ndim == 2 and chosen.shape[1] == CORE_LENGTH, "Expected (N, 10) cores")
    return {"schema_version": PARENT_REFERENCE_SCHEMA,
            "population": GATE_POPULATION,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_state_sha256": parent_state_sha256,
            "config_sha256": config_sha256,
            "scaffold_prefix_sha256": hashlib.sha256(scaffold_prefix.encode()).hexdigest(),
            "probability_convention": convention,
            "chosen_core_order_sha256": core_digest(chosen),
            "rejected_core_order_sha256": core_digest(rejected),
            "pairs": int(chosen.shape[0])}


def verify_pairs_against_identity(pairs, identity, *, where):
    """Re-derive both core orders from the ACTUAL rows and compare to the identity.

    An identity block is a claim about which rows were scored. Carrying it beside
    a vector proves nothing unless the rows handed in at scoring time are hashed
    again and found to agree -- otherwise a reordered, re-sliced or substituted
    pair set is scored and then labelled with the old identity, and D silently
    becomes a mean of differences between rows that are not partners.

    This runs at parent-reference construction and at **every** monitor check. It
    costs two SHA-256 passes over the core blocks; the alternative is an artifact
    whose identity field is decoration.
    """
    chosen = np.asarray(pairs["chosen_index"])
    rejected = np.asarray(pairs["rejected_index"])
    require(chosen.shape == rejected.shape,
            f"{where}: the pair set is not aligned ({chosen.shape} vs {rejected.shape})")
    require(chosen.ndim == 2 and chosen.shape[1] == CORE_LENGTH,
            f"{where}: expected (N, {CORE_LENGTH}) cores, got {chosen.shape}")
    observed = {"population": GATE_POPULATION,
                "pairs": int(chosen.shape[0]),
                "chosen_core_order_sha256": core_digest(chosen),
                "rejected_core_order_sha256": core_digest(rejected)}
    declared = {key: identity.get(key) for key in observed}
    differing = sorted(key for key in observed if observed[key] != declared[key])
    require(not differing,
            f"{where}: the rows being scored are not the rows this reference was built on. "
            f"Differing: {differing}. Observed {observed}, identity declares {declared}. D is a "
            "mean of PAIRED differences taken row by row, so a reordered or re-sliced pair set is "
            "a different measurement, not the same one under a new order.")
    declared_count = pairs.get("pairs")
    if declared_count is not None:
        require(int(declared_count) == observed["pairs"],
                f"{where}: the pair block says {declared_count} pairs but carries "
                f"{observed['pairs']} rows")
    return observed


@dataclass(frozen=True)
class ParentValidationReference:
    """The parent's chosen/rejected sum log probabilities, bound to their identity."""

    identity: dict
    chosen: np.ndarray
    rejected: np.ndarray
    gpu_seconds: float
    wall_seconds: float
    reused: bool = False

    @property
    def pairs(self):
        return int(self.chosen.size)

    def document(self):
        return {"identity": self.identity, "pairs": self.pairs,
                "mean_parent_chosen": json_number(self.chosen.mean()),
                "mean_parent_rejected": json_number(self.rejected.mean()),
                "parent_chosen_nll_per_residue": json_number(-self.chosen.mean() / CORE_POSITIONS),
                "gpu_seconds": json_number(self.gpu_seconds),
                "wall_seconds": json_number(self.wall_seconds),
                "reused_from_disk": bool(self.reused),
                "cost_note": ("scoring the parent is an evaluation cost: it is measured on the "
                              "monitor clock and is never charged to the training budget")}


def build_parent_reference(policy, pairs, identity, *, clock=None, batch_size=256):
    """Score the parent on both sides of every fixed validation pair, exactly once.

    ``clock`` is the **monitor** clock, not the training one. Passing the training
    clock here would make the gate's own reference eat the first budget, which is
    precisely the accounting error this design is arranged to prevent.

    The rows are verified against ``identity`` before they are scored, so the
    identity the reference is bound to describes the rows that actually produced
    its numbers.
    """
    started = time.perf_counter()
    verify_pairs_against_identity(pairs, identity, where="parent reference construction")
    if clock is None:
        chosen = score_sequences(policy, pairs["chosen_index"], batch_size=batch_size,
                                 progress_every=0)
        rejected = score_sequences(policy, pairs["rejected_index"], batch_size=batch_size,
                                   progress_every=0)
        gpu_seconds = 0.0
    else:
        with clock.segment() as segment:
            chosen = score_sequences(policy, pairs["chosen_index"], batch_size=batch_size,
                                     progress_every=0)
            rejected = score_sequences(policy, pairs["rejected_index"], batch_size=batch_size,
                                       progress_every=0)
        gpu_seconds = float(segment.seconds)
    require(bool(np.isfinite(chosen).all()) and bool(np.isfinite(rejected).all()),
            "The parent produced a nonfinite validation score; the gate would have no reference")
    bound = dict(identity, chosen_values_sha256=array_digest(chosen),
                 rejected_values_sha256=array_digest(rejected))
    chosen = np.array(chosen, dtype=np.float64)
    rejected = np.array(rejected, dtype=np.float64)
    chosen.setflags(write=False)
    rejected.setflags(write=False)
    return ParentValidationReference(identity=bound, chosen=chosen, rejected=rejected,
                                     gpu_seconds=gpu_seconds,
                                     wall_seconds=time.perf_counter() - started)


def save_parent_reference(path, reference):
    """Write the parent vectors beside their identity. An existing file is kept."""
    path = Path(path)
    require(path.suffix == ".npz", "The parent reference array path must end in .npz")
    require(not path.is_file(), f"A parent validation reference already exists at {path}; it is "
                                "immutable, and a second one would silently redefine D")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, chosen=reference.chosen, rejected=reference.rejected)
    save_json(path.with_suffix(".json"), {"schema_version": PARENT_REFERENCE_SCHEMA,
                                          "identity": reference.identity,
                                          "gpu_seconds": json_number(reference.gpu_seconds),
                                          "wall_seconds": json_number(reference.wall_seconds)})
    return path


def load_parent_reference(path, expected_identity):
    """Reload and re-verify. Every identity field is compared; none is optional."""
    started = time.perf_counter()
    path = Path(path)
    document = load_json(path.with_suffix(".json"))
    require(document["schema_version"] == PARENT_REFERENCE_SCHEMA,
            "Unsupported parent-reference schema")
    stored = document["identity"]
    with np.load(path) as archive:
        chosen = np.array(archive["chosen"], dtype=np.float64)
        rejected = np.array(archive["rejected"], dtype=np.float64)
    require(array_digest(chosen) == stored.get("chosen_values_sha256")
            and array_digest(rejected) == stored.get("rejected_values_sha256"),
            "Parent reference contents do not match the digests recorded with them")
    expected = dict(expected_identity,
                    chosen_values_sha256=stored.get("chosen_values_sha256"),
                    rejected_values_sha256=stored.get("rejected_values_sha256"))
    differing = sorted(key for key in set(stored) | set(expected)
                       if stored.get(key) != expected.get(key))
    require(not differing, f"Parent reference identity mismatch. Differing keys: {differing}")
    chosen.setflags(write=False)
    rejected.setflags(write=False)
    return ParentValidationReference(identity=stored, chosen=chosen, rejected=rejected,
                                     gpu_seconds=float(document["gpu_seconds"] or 0.0),
                                     wall_seconds=time.perf_counter() - started, reused=True)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LikelihoodGate:
    """The threshold, its units, and the verdict on one set of current scores."""

    threshold_nats_per_sequence: float = DEFAULT_THRESHOLD_NATS_PER_SEQUENCE
    core_positions: int = CORE_POSITIONS

    def __post_init__(self):
        require(math.isfinite(self.threshold_nats_per_sequence)
                and self.threshold_nats_per_sequence > 0,
                "The gate threshold must be finite and positive")
        require(isinstance(self.core_positions, int) and self.core_positions > 0,
                "The core width must be a positive integer")

    @property
    def threshold_nats_per_residue(self):
        return self.threshold_nats_per_sequence / self.core_positions

    def document(self):
        return {"threshold_nats_per_sequence": float(self.threshold_nats_per_sequence),
                "threshold_nats_per_residue": float(self.threshold_nats_per_residue),
                "statistic": "D = mean(parent_chosen - current_chosen) over the fixed validation "
                             "pairs, in nats per sequence (10 positions)",
                "rule": "stop on strict D > threshold (exact equality passes) or any nonfinite "
                        "score; the gate applies to every arm including continued_sft",
                "population": GATE_POPULATION,
                "pairs": "all of them, chosen and rejected, on every check"}

    def evaluate(self, reference, current_chosen, current_rejected):
        """The verdict plus the whole observed distribution, as a JSON-safe record."""
        chosen = np.asarray(current_chosen, dtype=np.float64)
        rejected = np.asarray(current_rejected, dtype=np.float64)
        require(chosen.shape == reference.chosen.shape and rejected.shape
                == reference.rejected.shape,
                f"The gate scored {chosen.size}/{rejected.size} rows against a reference of "
                f"{reference.chosen.size}/{reference.rejected.size}; the pair set is fixed")
        nonfinite = int((~np.isfinite(chosen)).sum() + (~np.isfinite(rejected)).sum())
        drop = reference.chosen - chosen
        record = {"pairs": int(chosen.size),
                  "nonfinite_scores": nonfinite,
                  "threshold_nats_per_sequence": float(self.threshold_nats_per_sequence),
                  "threshold_nats_per_residue": float(self.threshold_nats_per_residue),
                  "chosen_drop": quantile_summary(drop),
                  "rejected_drop": quantile_summary(reference.rejected - rejected),
                  "current_chosen": quantile_summary(chosen),
                  "mean_current_chosen_nll_per_residue": None,
                  "pair_accuracy": None,
                  "mean_implicit_margin": None}
        if nonfinite:
            record.update(D=None, D_per_residue=None, passed=False,
                          stop_reason="nonfinite_validation_score",
                          note=("a nonfinite score is not a small D: the policy has left the "
                                "domain where the comparison means anything, so the trajectory "
                                "stops and the numbers are reported as null, not as zeros"))
            return record
        value = float(drop.mean())
        margin = (chosen - rejected) - (reference.chosen - reference.rejected)
        record.update(D=json_number(value),
                      D_per_residue=json_number(value / self.core_positions),
                      mean_current_chosen_nll_per_residue=json_number(
                          -chosen.mean() / self.core_positions),
                      pair_accuracy=json_number((chosen > rejected).mean()),
                      mean_implicit_margin=json_number(margin.mean()),
                      passed=bool(value <= self.threshold_nats_per_sequence))
        if not record["passed"]:
            record["stop_reason"] = "parent_relative_likelihood_breach"
        return record


class MonitorSchedule:
    """When a check is due: after update 1, then every 25 updates OR 5 GPU seconds.

    "Whichever comes first" is measured from the *last check*, not from the start,
    so a fast stretch is checked on the update count and a slow one on the clock.
    Budget crossings and the final state are forced by the caller and also reset
    the counters -- otherwise a crossing check would be followed by a redundant
    one an update later.
    """

    def __init__(self, *, first_update=1, update_interval=DEFAULT_UPDATE_INTERVAL,
                 gpu_second_interval=DEFAULT_GPU_SECOND_INTERVAL):
        require(isinstance(first_update, int) and first_update >= 1, "The first check is update 1")
        require(isinstance(update_interval, int) and update_interval > 0,
                "The update interval must be a positive integer")
        require(gpu_second_interval > 0, "The GPU-second interval must be positive")
        self.first_update = first_update
        self.update_interval = update_interval
        self.gpu_second_interval = float(gpu_second_interval)
        self.last_update = None
        self.last_gpu_seconds = 0.0

    def due(self, update, gpu_seconds):
        """``(due, reason)`` for an ordinary between-updates check."""
        if self.last_update is None:
            return (update >= self.first_update, "first_update")
        if update - self.last_update >= self.update_interval:
            return True, "update_interval"
        if gpu_seconds - self.last_gpu_seconds >= self.gpu_second_interval:
            return True, "gpu_second_interval"
        return False, None

    def record(self, update, gpu_seconds):
        self.last_update = int(update)
        self.last_gpu_seconds = float(gpu_seconds)

    def document(self):
        return {"first_update": self.first_update, "update_interval": self.update_interval,
                "gpu_second_interval": self.gpu_second_interval,
                "also": ["every nominal-budget crossing", "the final state"],
                "note": "checks run between updates and are never charged to the training clock"}


class GateMonitor:
    """One trajectory's gate: scores the policy, judges it, and persists the evidence.

    The three costs are kept apart because they answer different questions:
    ``monitor_gpu_seconds`` is device time spent scoring, ``monitor_wall_seconds``
    is everything the check took including the persistence, and neither touches
    ``clock`` -- the training budget -- at all.
    """

    def __init__(self, policy, reference, *, gate=None, schedule=None, directory,
                 batch_size=256, score_sequences_fn=score_sequences, monitor_clock=None):
        self.policy = policy
        self.reference = reference
        self.gate = gate or LikelihoodGate()
        self.schedule = schedule or MonitorSchedule()
        self.directory = Path(directory)
        self.batch_size = batch_size
        self._score = score_sequences_fn
        self.monitor_clock = monitor_clock
        self.gpu_seconds = 0.0
        self.wall_seconds = 0.0
        self.checks = 0

    def _score_both(self, pairs):
        if self.monitor_clock is None:
            chosen = self._score(self.policy, pairs["chosen_index"], batch_size=self.batch_size,
                                 progress_every=0)
            rejected = self._score(self.policy, pairs["rejected_index"],
                                   batch_size=self.batch_size, progress_every=0)
            return chosen, rejected, 0.0
        with self.monitor_clock.segment() as segment:
            chosen = self._score(self.policy, pairs["chosen_index"], batch_size=self.batch_size,
                                 progress_every=0)
            rejected = self._score(self.policy, pairs["rejected_index"],
                                   batch_size=self.batch_size, progress_every=0)
        return chosen, rejected, float(segment.seconds)

    def check(self, pairs, *, update, gpu_seconds, reason):
        """Score every fixed validation pair on both sides and judge the result.

        The rows are re-hashed against the reference identity first: the check is
        a comparison against *those* rows in *that* order, and a check that scored
        something else and then stamped the old identity on it would be worse than
        no check. ``monitor_wall_seconds`` is taken last, after the persistence and
        every digest, because those are part of what a check costs.

        The schedule is advanced here for a monitor used on its own; the guarded
        loop advances the schedule it was given as well, so the cadence never
        depends on this side effect having happened.
        """
        started = time.perf_counter()
        verified = verify_pairs_against_identity(pairs, self.reference.identity,
                                                 where=f"monitor check at update {update}")
        was_training = getattr(self.policy, "model", None) is not None and self.policy.model.training
        chosen, rejected, measured = self._score_both(pairs)
        record = self.gate.evaluate(self.reference, chosen, rejected)
        stored = self.directory / "monitor_scores" / f"check_{self.checks:05d}_update{update}.npz"
        stored.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stored, chosen=np.asarray(chosen, dtype=np.float64),
                 rejected=np.asarray(rejected, dtype=np.float64))
        record.update(check=self.checks, update=int(update),
                      training_gpu_seconds=json_number(gpu_seconds), reason=reason,
                      monitor_gpu_seconds=json_number(measured),
                      scores={"path": str(stored),
                              "sha256": _file_digest(stored),
                              "chosen_values_sha256": array_digest(chosen),
                              "rejected_values_sha256": array_digest(rejected),
                              "note": ("the per-pair D vector is exactly "
                                       "parent_chosen - current_chosen over these rows")},
                      identity=dict(self.reference.identity),
                      scored_rows=verified,
                      policy_was_training=bool(was_training))
        record["monitor_wall_seconds"] = json_number(time.perf_counter() - started)
        record["monitor_wall_note"] = ("the whole check: row verification, scoring, the verdict, "
                                       "writing the score vectors and hashing them")
        self.checks += 1
        self.gpu_seconds += measured
        self.wall_seconds += record["monitor_wall_seconds"] or 0.0
        self.schedule.record(update, gpu_seconds)
        return record

    def document(self):
        return {"checks": self.checks,
                "monitor_gpu_seconds": json_number(self.gpu_seconds),
                "monitor_wall_seconds": json_number(self.wall_seconds),
                "gate": self.gate.document(), "cadence": self.schedule.document(),
                "reference": self.reference.document(),
                "accounting_note": ("monitoring GPU and wall time are measured, reported and "
                                    "additional; they are not subtracted from the training budget "
                                    "and the budget arithmetic does not cap them")}


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# re-deriving a journalled verdict from the vectors it was measured on
# ---------------------------------------------------------------------------

def monitor_score_arrays(directory, verdict):
    """The retained per-check score vectors, found beside the run and re-hashed.

    The path a check recorded is the absolute path it wrote to on the machine that
    ran it, so the file is looked up by name under this run's own
    ``monitor_scores`` directory: a copied or moved run directory still resolves,
    and an array that is simply gone is refused rather than inferred around.
    """
    scores = dict(verdict.get("scores") or {})
    recorded = scores.get("path")
    require(recorded,
            f"A gate verdict in {directory} names no retained score vectors. D is a mean over two "
            "vectors; without them the number in the journal cannot be checked and is a claim, "
            "not a measurement.")
    path = Path(directory) / "monitor_scores" / Path(str(recorded)).name
    require(path.is_file(),
            f"{path} is missing. The vectors this verdict was measured from are not beside the "
            "run, so its D cannot be recomputed and the verdict is not evidence about anything.")
    digest = _file_digest(path)
    require(digest == scores.get("sha256"),
            f"{path} changed since the check wrote it ({digest} != {scores.get('sha256')}). These "
            "are not the numbers the verdict was measured on.")
    with np.load(path) as archive:
        chosen = np.array(archive["chosen"], dtype=np.float64)
        rejected = np.array(archive["rejected"], dtype=np.float64)
    require(array_digest(chosen) == scores.get("chosen_values_sha256")
            and array_digest(rejected) == scores.get("rejected_values_sha256"),
            f"{path}: the stored vectors do not match the value digests the verdict recorded")
    return chosen, rejected, {"path": str(path), "sha256": digest}


def recompute_gate_verdict(directory, verdict, reference, *, gate, atol=GATE_RECOMPUTATION_ATOL):
    """Recompute D from the retained vectors and hold the journalled verdict to it.

    A journal line is a claim about a measurement. This is the measurement again:
    the arrays that check wrote are re-hashed, the verified parent reference
    supplies the other side, and ``gate.evaluate`` -- the same declared statistic,
    the same strict ``>`` -- produces D afresh. A line whose arrays are gone, whose
    identity names other rows, whose threshold is not the declared one, or whose
    ``D``/``passed`` are not what those vectors produce is refused. That includes
    the shape this check exists for: ``passed: true`` beside a D no comparison of
    these two vectors could have returned.
    """
    chosen, rejected, stored = monitor_score_arrays(directory, verdict)
    require(chosen.shape == reference.chosen.shape and rejected.shape == reference.rejected.shape,
            f"{stored['path']}: {chosen.size}/{rejected.size} retained scores against a parent "
            f"reference of {reference.chosen.size}/{reference.rejected.size}; the pair set is "
            "fixed and these are different populations")
    require(verdict.get("identity") == reference.identity,
            f"{stored['path']}: this verdict was measured against a different parent reference "
            "than the verified one frozen in this run directory")
    require(int(verdict.get("pairs", -1)) == reference.pairs,
            f"{stored['path']}: the verdict claims {verdict.get('pairs')} pairs and the verified "
            f"reference carries {reference.pairs}")
    declared = verdict.get("threshold_nats_per_sequence")
    require(declared is not None
            and abs(float(declared) - float(gate.threshold_nats_per_sequence)) <= 1e-12,
            f"{stored['path']}: the verdict was judged against {declared} nats/sequence and this "
            f"campaign declares {gate.threshold_nats_per_sequence}. A verdict measured under "
            "another threshold is not a verdict under this protocol.")
    recomputed = gate.evaluate(reference, chosen, rejected)
    require(recomputed["D"] is not None,
            f"{stored['path']}: recomputing D over the retained vectors gives no finite value "
            f"({recomputed.get('stop_reason')}); nothing may rest on it")
    journalled = verdict.get("D")
    require(journalled is not None,
            f"{stored['path']}: the verdict records no measured D, so there is nothing to check "
            "the recomputation against")
    difference = abs(float(journalled) - float(recomputed["D"]))
    require(difference <= atol + atol * abs(float(recomputed["D"])),
            f"{stored['path']}: the journal records D={journalled} and these vectors give "
            f"D={recomputed['D']} (difference {difference}). The recorded number is not the "
            "measurement.")
    require(bool(verdict.get("passed")) is bool(recomputed["passed"]),
            f"{stored['path']}: the journal records passed={verdict.get('passed')} and the "
            f"declared gate applied to these vectors gives passed={recomputed['passed']} at "
            f"D={recomputed['D']} against {gate.threshold_nats_per_sequence} nats/sequence")
    return {"D": recomputed["D"], "D_per_residue": recomputed["D_per_residue"],
            "passed": bool(recomputed["passed"]),
            "pairs": recomputed["pairs"],
            "threshold_nats_per_sequence": recomputed["threshold_nats_per_sequence"],
            "chosen_drop_mean": (recomputed["chosen_drop"] or {}).get("mean"),
            "scores": stored,
            "parent_reference_identity": dict(reference.identity),
            "note": ("D was recomputed from the retained per-check vectors against the verified "
                     "parent reference and agrees with the journalled verdict; it is not read "
                     "from the journal")}
