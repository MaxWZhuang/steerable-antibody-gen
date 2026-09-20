"""The exposure-matched replay campaign: one writer, truthful statuses, exact exposures.

This module owns what happens between the training freeze and the report: the
queue, the exclusive lock, one trajectory's update loop, the gate, the
preservation diagnostics, the endpoint evaluations and the terminal status of
every arm-seed in the grid.

Six properties are enforced here rather than left to convention:

* **There is no resume.** A trajectory that began and did not finish is
  ``incomplete`` forever and is never restarted under the same identity. Resuming
  would require saving and restoring the model, the optimizer, the scheduler,
  every RNG state, both stream positions and the exposure counters atomically, and
  then *proving* interrupted and uninterrupted runs agree. The historical guarded
  loop does not provide that contract, this screen does not implement it, and a
  fit that silently continued from a partial state would be a different
  trajectory wearing a finished one's name.
* **Liveness is decided by the operating system, and observing it changes
  nothing.** A forced kill runs no handler, so no in-process flag can be trusted.
  The campaign holds an exclusive OS lock for its whole writing life; the kernel
  releases it when the process dies. A reader answers the question with
  :func:`lock_state`, which tests and immediately drops the lock without creating
  a missing file and without writing an owner record -- a status page must not
  rewrite the provenance of the run it is looking at. PID and process-start
  identity are recorded as evidence, never used as the liveness test: PIDs are
  reused.
* **A gate breach stops one trajectory; an unexpected exception stops the
  campaign.** The first is a declared scientific outcome and the queue continues to
  the next arm. The second means something is wrong with the machinery, and
  running 30 more trajectories under it would produce 30 more unusable results.
* **A reached endpoint is created last.** The gate must have passed at that exact
  update, the checkpoint bytes must have been written *and re-read*, and the
  endpoint evaluation must have been saved. Only then does an endpoint record
  exist. A failed gate never produces a selectable nominal endpoint, and a
  diversity-ineligible endpoint is a real observed point with a flag, which is a
  different thing from a likelihood breach and from a winner.
* **Exposures are counted, not assumed.** Every update consumes exactly the
  declared rows from the resolved stream, the accumulator refuses an update that
  consumed a different number, and the endpoints are exact exposure counts.
* **The teacher never enters the fit as a model.** Its targets are materialized
  once into cached probabilities and logs, so the trajectory holds one model, one
  optimizer state and no second copy of the backbone on a 4 GB device.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import her2_replay as replay_lib
from . import her2_replay_streams as streams_lib
from . import her2_support_paths as paths
from . import her2_support_scoring as scoring
from .her2_runtime import require

CAMPAIGN_SCHEMA = "her2-parent-replay/1"
#: New schemas. The guarded saver stamps ``her2-guarded-trajectory/1`` on whatever
#: it writes, so reusing it would label this campaign's diagnostics as artifacts of
#: the completed one.
TRAJECTORY_SCHEMA = "her2-parent-replay-trajectory/1"
ENDPOINT_SCHEMA = "her2-parent-replay-endpoint/1"

#: Terminal statuses. ``stopped_by_gate`` is a declared scientific outcome;
#: ``incomplete`` is a trajectory that began and did not finish; ``failed`` is the
#: machinery, not the science. They are never collapsed into one another.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped_by_gate"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_STOPPED, STATUS_INCOMPLETE, STATUS_FAILED)

#: Files inside one trajectory directory.
IDENTITY_JSON = "identity.json"
STATUS_JSON = "status.json"
PROGRESS_JSON = "trajectory_progress.json"
UPDATES_JSONL = "updates.jsonl"
MONITOR_JSONL = "monitor.jsonl"
#: Append-only journal of the endpoint **records**, written as each one is created.
#: A reached endpoint has to survive a forced kill as a record with its identity,
#: its checkpoint digest and its evaluation -- a ``.pt`` on disk is not an endpoint.
ENDPOINTS_JSONL = "endpoints.jsonl"
LAST_PASSING = "last_passing.pt"
FAILED_STATE = "failed_state.pt"

#: The only ``ValueError`` from strict scoring that this campaign is allowed to
#: record as a diagnostic observation rather than fail on. A schema, ordering or
#: programming error is not an observation about the model.
RECOGNIZED_SCORING_OBSERVATIONS = ("nonfinite",)


def arm_id(task, replay_lambda):
    """The stable identifier one arm carries through every artifact and table."""
    text = repr(float(replay_lambda))
    if text.endswith(".0"):
        text = text[:-2]
    return f"{task}_lambda{text.replace('.', 'p').replace('-', 'm')}"


def trajectory_id(task, replay_lambda, seed):
    return f"{arm_id(task, replay_lambda)}_seed{int(seed)}"


def build_queue(config):
    """The full 36-trajectory grid in one fixed, result-independent order.

    Controls first within each parent seed, then ascending lambda, then the tasks
    in their declared order. Declared before fitting so that an early stop still
    leaves the matched ``lambda = 0`` control available for every arm that was
    reached, and so no ordering decision can be made after seeing a result.
    """
    screen = config["screen"]
    lambdas = sorted(float(value) for value in screen["replay_lambdas"])
    require(lambdas and lambdas[0] == 0.0,
            "The grid must contain the lambda = 0 control; a screen without its matched control "
            "has nothing to report a difference against")
    rows = []
    for seed in [int(value) for value in screen["parent_seeds"]]:
        for value in lambdas:
            for task in list(screen["tasks"]):
                rows.append({"trajectory": trajectory_id(task, value, seed),
                             "arm_id": arm_id(task, value), "task": task,
                             "replay_lambda": float(value), "seed": int(seed),
                             "is_control": value == 0.0})
    require(len(rows) == len(screen["tasks"]) * len(lambdas) * len(screen["parent_seeds"]),
            "The queue is not the declared grid")
    return rows


# ---------------------------------------------------------------------------
# one writer at a time
# ---------------------------------------------------------------------------

def owner_identity():
    """Evidence about this process. Never the liveness test; PIDs are reused."""
    return {"pid": int(os.getpid()),
            "session": uuid.uuid4().hex,
            "started_at": paths.utc_now(),
            "monotonic_at_start": time.monotonic(),
            "python": platform.python_version(),
            "host_fingerprint": hashlib.sha256(
                platform.node().encode("utf-8")).hexdigest()[:16],
            "note": ("recorded so a stale heartbeat can be attributed. Liveness is decided by the "
                     "OS lock, which the kernel releases on a forced kill, not by this block.")}


def _take_lock(handle):
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _drop_lock(handle):
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


#: What an observer can learn about a lock without claiming it.
LOCK_MISSING = "missing"
LOCK_HELD = "held"
LOCK_RELEASED = "released"
LOCK_INACCESSIBLE = "inaccessible"


def lock_state(path):
    """Observe a lock **without** creating it, writing to it or claiming ownership.

    The previous implementation answered this question by calling
    :meth:`CampaignLock.acquire`, which creates a missing lock file and overwrites
    ``.owner.json`` with the *observer's* session. A status page or a dashboard
    asking "is anybody writing?" would therefore rewrite the recorded provenance of
    the run it was only supposed to be looking at. Observation is now read-only:
    the file is opened for update and the lock is tested and immediately dropped,
    no bytes are written, and a missing file is reported as missing rather than
    created.

    Four outcomes, because they mean different things: ``missing`` (no campaign has
    ever taken this lock), ``held`` (a live owner has it), ``released`` (a lock file
    exists and nobody holds it -- the owner finished or was killed) and
    ``inaccessible`` (the question could not be answered, which is never reported as
    "nobody is there").
    """
    path = Path(path)
    record = path.with_suffix(".owner.json")
    owner = paths.read_json(record) if record.is_file() else None
    block = {"path": path.name, "recorded_owner": owner, "observed_at": paths.utc_now(),
             "observation": ("read-only: the lock is tested and released, no file is created and "
                             "no owner record is written")}
    if not path.is_file():
        return dict(block, state=LOCK_MISSING,
                    note="no lock file exists; no campaign process has taken this run directory")
    try:
        handle = path.open("r+b")
    except OSError as error:
        return dict(block, state=LOCK_INACCESSIBLE, reason=f"{type(error).__name__}: {error}",
                    note=("the lock could not be opened, so liveness is unknown. Unknown is never "
                          "reported as 'the owner is gone'."))
    try:
        _take_lock(handle)
    except OSError as error:
        handle.close()
        return dict(block, state=LOCK_HELD, reason=f"{type(error).__name__}: {error}",
                    note="a live process holds the exclusive write lock")
    try:
        _drop_lock(handle)
    finally:
        handle.close()
    return dict(block, state=LOCK_RELEASED,
                note=("the lock file exists and nobody holds it. The kernel releases it on a "
                      "forced kill, so this covers both a clean exit and a killed owner."))


class CampaignLock:
    """An exclusive OS lock held for the entire writing life of the campaign.

    Held across bank generation *and* fitting, because both write into the run
    directory and two writers there would interleave shard records, checkpoints and
    the queue. Acquisition is non-blocking and a failure names the recorded owner:
    two campaigns on one box is an operator error worth reporting, not a race to
    resolve silently.

    Taking the lock replaces ``.owner.json`` with this process's identity, so the
    record it replaces is appended to an append-only owners journal first. Who held
    this directory before, and when, is evidence about an interrupted campaign; it
    is not something the next campaign gets to overwrite.
    """

    def __init__(self, path, *, owner=None):
        self.path = Path(path)
        self.owner = dict(owner or owner_identity())
        self.handle = None
        self.acquired_at = None

    def _lock(self, handle):
        _take_lock(handle)

    def _unlock(self, handle):
        _drop_lock(handle)

    def owner_path(self):
        return self.path.with_suffix(".owner.json")

    def owners_journal_path(self):
        return self.path.with_suffix(".owners.jsonl")

    def recorded_owner(self):
        record = self.owner_path()
        return paths.read_json(record) if record.is_file() else None

    def previous_owners(self):
        """Every owner record this lock has displaced, oldest first."""
        journal = self.owners_journal_path()
        if not journal.is_file():
            return []
        out = []
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"record_kind": "unreadable_owner_record", "raw_bytes": len(line)})
        return out

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            self._lock(handle)
        except OSError as error:
            handle.close()
            previous = self.recorded_owner() or {}
            raise ValueError(
                "Another campaign process holds the write lock on this run directory "
                f"({type(error).__name__}). The recorded owner is pid {previous.get('pid')}, "
                f"session {previous.get('session')}, started {previous.get('started_at')}. One "
                "writer at a time: a second one would interleave checkpoints, shard records and "
                "the queue.") from error
        self.handle = handle
        self.acquired_at = paths.utc_now()
        displaced = self.recorded_owner()
        if displaced is not None:
            with self.owners_journal_path().open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(
                    dict(displaced, record_kind="displaced_campaign_lock_owner",
                         displaced_at=self.acquired_at,
                         displaced_by=self.owner.get("session")),
                    sort_keys=True) + "\n")
        paths.write_json(self.owner_path(),
                         dict(self.owner, acquired_at=self.acquired_at,
                              record_kind="campaign_lock_owner",
                              schema_version=CAMPAIGN_SCHEMA,
                              previous_owners=self.owners_journal_path().name))
        return self

    def release(self):
        if self.handle is None:
            return False
        try:
            self._unlock(self.handle)
        finally:
            self.handle.close()
            self.handle = None
        return True

    @contextmanager
    def held(self):
        self.acquire()
        try:
            yield self
        finally:
            self.release()


def owner_is_gone(path):
    """``True`` when nobody holds the lock at ``path``. A read-only observation.

    The only safe liveness question available after a forced kill, and it is now
    asked without mutating anything: see :func:`lock_state`. An ``inaccessible``
    lock answers ``False`` -- "I could not tell" must never become "the owner is
    gone", because that is the answer that marks a live campaign's trajectories
    incomplete.
    """
    return lock_state(path)["state"] in (LOCK_MISSING, LOCK_RELEASED)


class Heartbeat:
    """The latest liveness evidence: wall time, monotonic time and the session token.

    One atomically replaced file, not an append-only journal: this answers "how
    long since the writer last said anything", and the history of that question is
    the per-update journal. Its age is a *progress* signal and never the liveness
    test -- a heartbeat can be stale because the owner is inside a long evaluation,
    and it can be young because a supervisor kept writing while the work stopped.
    Liveness is :func:`lock_state`.

    Wall and monotonic are both recorded because they fail differently: a clock
    change moves the first and not the second, and a suspended machine moves
    neither in a way that means the process was working.
    """

    def __init__(self, path, *, owner, every=15.0):
        self.path = Path(path)
        self.owner = dict(owner)
        self.every = float(every)
        self.last = None

    def beat(self, **fields):
        now = time.monotonic()
        if self.last is not None and now - self.last < self.every and not fields.get("force"):
            return None
        self.last = now
        record = dict(self.owner, schema_version=CAMPAIGN_SCHEMA, record_kind="campaign_heartbeat",
                      heartbeat_at=paths.utc_now(), monotonic=now,
                      **{k: v for k, v in fields.items() if k != "force"})
        paths.write_json(self.path, record)
        return record


# ---------------------------------------------------------------------------
# state files
# ---------------------------------------------------------------------------

def state_dict_digest(state):
    """Re-digest restored tensors by the same rule :func:`her2_policy.state_digest` uses.

    The point of having it here is that it runs on a *loaded* ``state_dict`` rather
    than on a live module: it is what lets a save be checked against the bytes that
    actually landed on disk instead of against the string that was written beside
    them.
    """
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(dict(state).items()):
        digest.update(str(name).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_schema(state):
    """``{name: (dtype, shape)}`` -- the structural claim a checkpoint has to satisfy."""
    return {str(name): [str(tensor.dtype), list(tensor.shape)]
            for name, tensor in dict(state).items()}


#: Metadata keys that say *when* a save happened rather than *what* it is.
#: ``paths.scientific_projection`` already drops the project's shared operational
#: keys; the trajectory identity adds ``started_at``, which is the clock at the
#: beginning of the fit and not part of the checkpoint's identity.
CHECKPOINT_METADATA_OPERATIONAL = frozenset({"started_at"})


def checkpoint_identity(metadata):
    """The scientific identity a checkpoint's metadata claims.

    Which trajectory, which parent, which freeze. A payload whose tensors are
    perfect but whose metadata says it belongs to another arm is not this update's
    checkpoint, and a comparison that skipped metadata could not tell: the earlier
    version of :meth:`ReplayStateSaver._inspect` excluded it, and a save that kept
    the tensors while rewriting ``trajectory`` and ``freeze`` was published.
    """
    node = paths.scientific_projection(dict(metadata or {}))
    return {key: value for key, value in node.items()
            if key not in CHECKPOINT_METADATA_OPERATIONAL}


class ReplayStateSaver:
    """Saves validated **before** they replace anything, and recorded after.

    ``last_passing`` and ``failed_state`` are different file names and there is no
    code path that points the failure save at the passing path: writing failing
    weights into a file whose name says "passing" is the one lie this design exists
    to prevent. An endpoint gets its own name per update, so a later endpoint can
    never overwrite an earlier one.

    Three properties, each of which was a real hole:

    * **The tensors are re-digested, and the identity metadata is compared.**
      Comparing the stored ``state_sha256`` string to the expected string proves
      only that the string was copied. A save that serialized corrupted tensors
      while keeping the digest field passed that check; recomputing the digest from
      the restored tensors does not. The metadata that says which trajectory,
      parent and freeze the weights belong to is compared as well -- by
      :func:`checkpoint_identity`, so an operational timestamp is not an identity
      change -- because tensors alone do not say whose they are.
    * **Validation happens on the temporary file.** The previous code
      ``os.replace``d first and validated afterwards, so a failed readback had
      already destroyed the last passing checkpoint. Nothing is published until the
      temporary has been loaded, structurally checked and re-digested; on any
      failure the temporary is removed and the file that was already there is
      untouched.
    * **Endpoints and failure snapshots are write-once.** Re-saving one is allowed
      only when the bytes on disk already record exactly this state, which makes a
      retry idempotent without making an overwrite possible. ``last_passing`` is
      explicitly the one rolling artifact, and says so in its own record.

    ``register`` is offered the write-once snapshots only. The rolling state is
    replaced at every passing check, and a write-once ledger would refuse its
    second write; that boundary lives here, in the one place that knows which save
    is rolling, rather than in each caller's callback.
    """

    def __init__(self, directory, model, *, metadata=None, register=None):
        self.directory = Path(directory)
        self.model = model
        self.metadata = dict(metadata or {})
        self.register = register
        self.saves = 0
        self.wall_seconds = 0.0
        self.bytes_written = 0

    def _inspect(self, target, *, expected_schema=None, expected_digest=None, payload=None):
        """Load a checkpoint and check every claim it makes about itself."""
        import torch
        restored = torch.load(target, map_location="cpu", weights_only=True)
        require(isinstance(restored, dict),
                f"{target}: a checkpoint is a dict payload, not {type(restored).__name__}")
        state = restored.get("state")
        require(isinstance(state, dict) and state,
                f"{target}: the payload carries no state dict, so there are no weights in it")
        if expected_schema is not None:
            observed = state_schema(state)
            missing = sorted(set(expected_schema) - set(observed))
            extra = sorted(set(observed) - set(expected_schema))
            require(not missing and not extra,
                    f"{target}: the restored state is missing {missing} and carries unexpected "
                    f"{extra}. A checkpoint with different keys is a different model.")
            wrong = sorted(name for name in expected_schema
                           if observed[name] != expected_schema[name])
            require(not wrong,
                    f"{target}: {wrong} came back with a different dtype or shape than the model "
                    "produced. A silently re-typed or truncated tensor is not this update's "
                    "weights.")
        recomputed = state_dict_digest(state)
        require(recomputed == restored.get("state_sha256"),
                f"{target}: the tensors on disk digest to {recomputed} and the payload records "
                f"{restored.get('state_sha256')}. The digest is recomputed from the restored "
                "tensors, so a payload that kept its digest field while its weights changed fails "
                "here rather than being published.")
        if expected_digest is not None:
            require(recomputed == expected_digest,
                    f"{target}: the restored tensors digest to {recomputed} and this update "
                    f"produced {expected_digest}. The bytes are not the weights this update "
                    "produced and nothing is published.")
        if payload is not None:
            differing = sorted(key for key in payload
                               if key != "metadata" and restored.get(key) != payload[key])
            require(not differing,
                    f"{target}: the reloaded payload disagrees about {differing}")
            # Metadata is compared too, and by its scientific identity rather than
            # byte-for-byte: it carries the trajectory, the parent and the freeze
            # this checkpoint belongs to. Excluding it -- as this did -- let a save
            # that kept every tensor while rewriting those fields be published.
            expected_identity = checkpoint_identity(payload.get("metadata"))
            observed_identity = checkpoint_identity(restored.get("metadata"))
            wrong_identity = sorted(key for key in set(expected_identity) | set(observed_identity)
                                    if expected_identity.get(key) != observed_identity.get(key))
            require(not wrong_identity,
                    f"{target}: the reloaded payload claims a different identity in "
                    f"{wrong_identity}. A checkpoint whose tensors are intact and whose metadata "
                    "names another trajectory, parent or freeze is not this update's checkpoint.")
        return {"state_sha256": recomputed, "keys": len(state),
                "schema_version": restored.get("schema_version"),
                "kind": restored.get("kind"), "update": restored.get("update")}

    def _write(self, name, payload, *, rolling=False):
        import torch
        from .her2_policy import state_digest
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        state = self.model.state_dict()
        digest = state_digest(self.model)
        schema = state_schema(state)
        if path.is_file() and not rolling:
            # Write-once. Identical content is accepted so a retry is idempotent;
            # different content is refused rather than replacing a reached
            # endpoint's or a recorded failure's bytes.
            existing = self._inspect(path, expected_schema=schema, expected_digest=digest,
                                     payload=payload)
            return self._record(name, path, payload, digest=digest, elapsed=0.0,
                                validated=("already on disk with exactly this state; a write-once "
                                           "snapshot is verified, never replaced"),
                                keys=existing["keys"], counted=False)
        temporary = path.with_name(path.name + ".tmp")
        previous = paths.sha256_file(path) if path.is_file() else None
        try:
            torch.save(dict(payload, state=state, state_sha256=digest), temporary)
            inspected = self._inspect(temporary, expected_schema=schema, expected_digest=digest,
                                      payload=payload)
        except BaseException as error:
            if temporary.is_file():
                temporary.unlink()
            raise ValueError(
                f"{path}: the checkpoint did not validate before publication "
                f"({type(error).__name__}: {error}). The temporary file was removed and the file "
                f"already at this path was not touched (sha256 {previous}). A save that cannot be "
                "read back is a failed save, not a reason to lose the last one that worked."
            ) from error
        os.replace(temporary, path)
        elapsed = time.perf_counter() - started
        return self._record(name, path, payload, digest=digest, elapsed=elapsed,
                            validated=("written to a temporary, re-read, structurally checked and "
                                       "re-digested from the restored tensors, and only then "
                                       "published over the destination"),
                            keys=inspected["keys"], counted=True, rolling=rolling)

    def _record(self, name, path, payload, *, digest, elapsed, validated, keys, counted,
                rolling=False):
        size = int(path.stat().st_size)
        if counted:
            self.saves += 1
            self.wall_seconds += elapsed
            self.bytes_written += size
        record = {"file": name, "sha256": paths.sha256_file(path), "state_sha256": digest,
                  "bytes": size, "wall_seconds": elapsed, "tensors": int(keys),
                  "validated": validated}
        # A rolling state is never offered to the ledger: it is replaced at the next
        # passing check, and binding it write-once would refuse that write.
        if self.register is not None and not rolling:
            self.register(name=name, path=path, kind=str(payload.get("kind")),
                          scientific_digest=digest)
        return record

    def save_last_passing(self, *, update, exposures, check):
        record = self._write(LAST_PASSING, {
            "schema_version": TRAJECTORY_SCHEMA, "kind": "last_passing",
            "update": int(update), "exposures": dict(exposures), "check": int(check),
            "metadata": self.metadata}, rolling=True)
        record.update(kind="last_passing", update=int(update), check=int(check), rolling=True,
                      note=("a rolling latest-passing state: this is the one artifact here that a "
                            "later check replaces. The digest recorded at an earlier check "
                            "described the bytes at this path then; it does not resolve there now "
                            "and no claim is made that it does."))
        return record

    def save_failed(self, *, update, exposures, stop_reason):
        record = self._write(FAILED_STATE, {
            "schema_version": TRAJECTORY_SCHEMA, "kind": "failed_state",
            "update": int(update), "exposures": dict(exposures), "stop_reason": stop_reason,
            "metadata": self.metadata})
        record.update(kind="failed_state", update=int(update), stop_reason=stop_reason,
                      rolling=False,
                      note=("diagnostic weights at the recorded stop reason. They are kept for "
                            "diagnosis and are never a last-passing state or a reached endpoint."))
        return record

    def save_endpoint(self, *, update, exposures):
        name = f"endpoint_update{int(update)}.pt"
        record = self._write(name, {
            "schema_version": ENDPOINT_SCHEMA, "kind": "exposure_endpoint",
            "update": int(update), "exposures": dict(exposures), "metadata": self.metadata})
        record.update(kind="exposure_endpoint", update=int(update), rolling=False)
        return record


# ---------------------------------------------------------------------------
# preservation diagnostics on the fresh monitoring bank
# ---------------------------------------------------------------------------

@dataclass
class PreservationMonitor:
    """Forward KL, sequence-drop tails and conditional KL on the fresh monitor bank.

    Three quantities, reported apart and never merged. The sequence drop
    ``log p0(y) - log ptheta(y)`` and the conditional ``K(y)`` have the same
    population mean and different distributions, so their tails describe different
    things; the forward-KL estimate is the mean of the first. All three are
    **recorded diagnostics** in this screen: no preservation stopping rule was
    declared before fitting, so none is applied.
    """

    index: np.ndarray
    parent_log_probability: np.ndarray
    teacher_probabilities: object
    teacher_log_probabilities: object
    batch_size: int = 256
    thresholds: dict = field(default_factory=lambda: dict(scoring.DEFAULT_THRESHOLDS))

    def measure(self, policy, *, conditional_batch=64):
        import torch
        block = {"rows": int(self.index.shape[0]),
                 "bank": "fresh independent monitoring draws, never trained on",
                 "estimands": {
                     "forward_kl": "mean(log p_parent(y) - log p_policy(y)) over parent draws",
                     "conditional_kl": ("mean over rows of K(y) = sum_t sum_a p0 (log p0 - log q); "
                                        "the same population mean as the sequence drop and a "
                                        "different random variable")}}
        try:
            scored = scoring.strict_sequence_log_probabilities(
                policy, self.index, batch_size=int(self.batch_size),
                label="preservation monitoring bank")
        except ValueError as error:
            # Only a recognized nonfinite-score observation follows the declared
            # diagnostic policy. A shape, ordering or programming failure is not an
            # observation about the model, and swallowing it would let a broken
            # monitor report "preservation unavailable" for 36 trajectories and let
            # the campaign call itself complete.
            text = str(error).lower()
            require(any(marker in text for marker in RECOGNIZED_SCORING_OBSERVATIONS),
                    "The preservation monitor failed with an error that is not a recognized "
                    f"nonfinite-score observation: {type(error).__name__}: {error}. The declared "
                    "diagnostic policy covers nonfinite scores on the monitoring bank and nothing "
                    "else; a schema, ordering or programming failure stops the campaign instead "
                    "of being recorded as a missing diagnostic.")
            block.update(available=False, forward_kl=None, tails=None, quantiles=None,
                         conditional_kl=None,
                         reason=f"{type(error).__name__}: {error}",
                         recognized_as="nonfinite score on the monitoring bank",
                         consequence=("recorded as a diagnostic failure. Preservation metrics are "
                                      "diagnostics in this screen and introduce no stopping rule; "
                                      "the declared likelihood gate is what stops a trajectory."))
            return block
        drop = scoring.signed_drop(self.parent_log_probability, scored["sum_log_probability"],
                                   label="preservation monitoring bank")
        was_training = bool(getattr(policy.model, "training", False))
        policy.model.eval()
        totals = []
        try:
            with torch.inference_mode():
                for start in range(0, int(self.index.shape[0]), int(conditional_batch)):
                    rows = slice(start, start + int(conditional_batch))
                    student = replay_lib.student_log_probabilities(policy, self.index[rows])
                    totals.append(replay_lib.sequence_conditional_kl(
                        self.teacher_probabilities[rows].to(student.device),
                        self.teacher_log_probabilities[rows].to(student.device),
                        student).double().cpu().numpy())
        finally:
            if was_training:
                policy.model.train()
        conditional = np.concatenate(totals)
        block.update(
            available=True,
            forward_kl={"mean": float(drop.mean()),
                        "standard_error": float(drop.std(ddof=1) / np.sqrt(drop.size)),
                        "rows": int(drop.size),
                        "estimator": "monte_carlo_from_independent_parent_draws"},
            quantiles=scoring.quantile_block(drop),
            tails=scoring.tail_block(drop, self.thresholds),
            policy_log_probability=scoring.log_probability_block(
                scored["sum_log_probability"], label="policy on the monitoring bank"),
            conditional_kl={"mean": float(conditional.mean()),
                            "standard_error": float(conditional.std(ddof=1)
                                                    / np.sqrt(conditional.size)),
                            "rows": int(conditional.size),
                            "quantiles": scoring.quantile_block(conditional),
                            "distinct_from_drop": ("these quantiles are of K(y), not of the "
                                                   "sequence drop; the two share a mean and are "
                                                   "not interchangeable")},
            logit_checks=scored["checks"])
        return block


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------

class CostLedger:
    """Separately measured seconds per category inside one wall clock.

    GPU seconds are not a matching variable and not an endpoint in this screen, so
    these are device-synchronized wall seconds by category rather than CUDA-event
    budgets. Their sum is not the wall time: model loading, imports and glue sit in
    the remainder and are reported there rather than folded into "optimizer".
    """

    CATEGORIES = ("optimizer", "gate", "preservation", "generation", "evaluation", "io",
                  "teacher_cache")

    def __init__(self, *, synchronize=None):
        self.started = time.perf_counter()
        self.seconds = {name: 0.0 for name in self.CATEGORIES}
        self.counts = {name: 0 for name in self.CATEGORIES}
        self._synchronize = synchronize

    def _sync(self):
        if self._synchronize is not None:
            self._synchronize()

    @contextmanager
    def segment(self, category):
        require(category in self.seconds, f"Unknown cost category {category!r}")
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[category] += time.perf_counter() - start
            self.counts[category] += 1

    @property
    def wall_seconds(self):
        return time.perf_counter() - self.started

    def document(self):
        wall = self.wall_seconds
        measured = sum(self.seconds.values())
        return {"seconds": {name: self.seconds[name] for name in self.CATEGORIES},
                "segments": {name: self.counts[name] for name in self.CATEGORIES},
                "wall_seconds": wall,
                "unattributed_seconds": max(0.0, wall - measured),
                "basis": ("device-synchronized wall seconds per category. GPU seconds are neither "
                          "a matching variable nor an endpoint here; replay cost is measured, not "
                          "assumed to be twice the baseline.")}


# ---------------------------------------------------------------------------
# what the append-only journals prove happened
# ---------------------------------------------------------------------------

def read_journal(path, *, record_kind=None):
    """Every **complete** record in an append-only journal, and an honest tail note.

    A killed process can leave a partial final line: the bytes of one record were
    being written when the kernel took the process away. That line is not a
    completed anything, so it is dropped -- and the fact that it was there is
    reported rather than silently smoothed over, because "the last update may not
    have finished" is exactly the sort of thing an interrupted run has to say.

    A final line that parses as a whole JSON object but carries no newline after it
    is a *different* state, and the two are reported separately. That record is
    complete, so it is kept, and ``unterminated_final_line`` says the writer was
    killed between the object and its terminator. Reporting a kept record as
    dropped was a reporting error, not a counting one.
    """
    path = Path(path)
    if not path.is_file():
        return {"records": [], "lines": 0, "truncated_trailing_line": False,
                "unterminated_final_line": False, "present": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    records, truncated = [], False
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            # Only the final line may be partial; a broken line in the middle is
            # corruption and is reported as such rather than skipped quietly.
            truncated = position == len(lines) - 1
            require(truncated,
                    f"{path}: line {position + 1} of {len(lines)} is not valid JSON and it is not "
                    "the final line. A journal damaged in the middle is corruption, not an "
                    "interrupted write.")
            continue
        if record_kind is None or record.get("record_kind") == record_kind:
            records.append(record)
    unterminated = bool(lines) and not text.endswith("\n") and not truncated
    return {"records": records, "lines": len(lines), "truncated_trailing_line": truncated,
            "unterminated_final_line": unterminated, "present": True}


def durable_progress(directory):
    """Completed updates, exposures, checks and reached endpoint RECORDS, off disk.

    The authority for "what actually finished" is the append-only journals, not the
    progress file: progress is written at checks, so after a kill between two checks
    it is behind by up to one whole interval. Recovering from the journals is what
    makes an interrupted trajectory report the 1,024 updates it completed instead of
    the 1,000 the last check happened to record.

    A reached endpoint is recovered as its **record** -- identity, checkpoint digest,
    evaluation -- from the endpoint journal. A ``.pt`` file on disk is not consulted
    and would not be accepted: an interrupted save leaves one behind, and this must
    not turn that into a reached endpoint.
    """
    directory = Path(directory)
    updates = read_journal(directory / UPDATES_JSONL, record_kind="update")
    monitor = read_journal(directory / MONITOR_JSONL, record_kind="gate_verdict")
    endpoints = read_journal(directory / ENDPOINTS_JSONL, record_kind="exposure_endpoint")
    last = updates["records"][-1] if updates["records"] else None
    reached = {}
    for record in endpoints["records"]:
        key = str(record.get("update"))
        if key != "None":
            reached[key] = record
    return {
        "updates": int((last or {}).get("update") or 0),
        "exposures": dict((last or {}).get("exposures") or {}),
        "checks": len(monitor["records"]),
        "journalled_update_records": len(updates["records"]),
        "endpoints_reached": reached,
        "endpoints_reached_updates": sorted(reached, key=lambda value: int(value)),
        "truncated_trailing_update_line": bool(updates["truncated_trailing_line"]),
        "truncated_trailing_endpoint_line": bool(endpoints["truncated_trailing_line"]),
        "unterminated_final_update_line": bool(updates["unterminated_final_line"]),
        "unterminated_final_endpoint_line": bool(endpoints["unterminated_final_line"]),
        "basis": ("the trajectory's own append-only journals. An update is counted only after its "
                  "completed record was journalled, so an attempted update is never reported as a "
                  "completed one. A final line that does not parse is incomplete and is dropped "
                  "and reported; a final line that parses whole but lost its newline is a complete "
                  "record, is kept, and is reported separately as unterminated."),
        "forced_kill_note": ("a process killed between completing an update and journalling it "
                             "loses that update from this count. The count is therefore a lower "
                             "bound on what happened, which is the safe direction.")}


# ---------------------------------------------------------------------------
# one trajectory
# ---------------------------------------------------------------------------

def weighted_diagnostic(blocks, key):
    """A row-weighted mean of one detached per-pair diagnostic across microbatches.

    Row-weighted for the same reason the loss is: a short final microbatch
    contributes its own count, not an equal share. ``None`` when the arm does not
    produce that diagnostic at all -- ``continued_sft`` has no margin, and saying
    so is different from reporting a zero.
    """
    import torch
    total, rows = 0.0, 0
    for count, block in blocks:
        value = (block or {}).get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            value = float(value.detach().double().mean())
        total += float(value) * int(count)
        rows += int(count)
    return None if rows == 0 else total / rows


def run_trajectory(*, row, directory, policy, optimizer, scheduler, stream, replay_order,
                   cadence, endpoints, batch_rows, microbatch_rows, task_batch, replay_batch,
                   gate_check, preservation, on_endpoint, gradient_clip, identity, heartbeat=None,
                   max_updates=None, ledger=None, journal_factory=None,
                   register_checkpoint=None, register_status=None):
    """Fit one arm at one seed and return its truthful terminal document.

    ``task_batch(rows_chosen, rows_rejected)`` and ``replay_batch(rows)`` return
    ``(mean, diagnostics)`` for one microbatch; ``gate_check(update, reason)``
    returns the gate record; ``on_endpoint(update, exposures)`` performs and
    persists the endpoint evaluation and returns its record. The loop owns the
    exposure accounting, the cadence, the statuses and the saves, and owns nothing
    about the science of any of those callbacks.

    An unexpected exception is **not** returned as a document. The trajectory's
    terminal status is written as ``failed`` -- with its counters taken from the
    journals, so an attempted update is not reported as a completed one -- and the
    exception is re-raised, which fail-stops the campaign.
    """
    from .her2_guarded_trajectory import JsonlJournal
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ledger = ledger or CostLedger()
    factory = journal_factory or JsonlJournal
    updates_journal = factory(directory / UPDATES_JSONL)
    monitor_journal = factory(directory / MONITOR_JSONL)
    endpoints_journal = factory(directory / ENDPOINTS_JSONL)
    saver = ReplayStateSaver(directory, policy.model, metadata=dict(identity),
                             register=register_checkpoint)
    total_updates = int(max_updates if max_updates is not None else max(endpoints))
    replay_lambda = float(row["replay_lambda"])
    started = time.perf_counter()

    exposures = {"chosen": 0, "rejected": 0, "replay": 0}
    reached, checks, clipped = {}, 0, 0
    status, stop_reason, last_passing = STATUS_RUNNING, None, None
    endpoints_declared = sorted(int(value) for value in endpoints)
    update = 0

    def persist(reason, *, completed=None):
        paths.write_json(directory / PROGRESS_JSON, {
            "schema_version": TRAJECTORY_SCHEMA, "record_kind": "trajectory_progress",
            "trajectory": row["trajectory"], "reason": reason, "status": status,
            "stop_reason": stop_reason,
            "update": update if completed is None else int(completed),
            "attempted_update": update, "checks": checks,
            "exposures": dict(exposures), "clipped_updates": clipped,
            "endpoints_reached": sorted(reached), "last_passing": last_passing,
            "declared_endpoints": endpoints_declared,
            "cost": ledger.document(), "heartbeat_at": paths.utc_now(),
            "note": ("written at checks, so it can lag the journals by up to one interval. The "
                     "append-only updates/monitor/endpoint journals beside it are the authority "
                     "for what completed; see durable_progress.")})

    persist("started")
    fatal = None
    try:
        while update < total_updates:
            update += 1
            chosen_rows, rejected_rows = stream.batch(update)
            require(int(chosen_rows.size) == int(batch_rows),
                    f"Update {update} drew {chosen_rows.size} chosen rows, not the declared "
                    f"{batch_rows}")
            accumulator = replay_lib.MicrobatchAccumulator(
                {"task": int(batch_rows), **({"replay": int(batch_rows)} if replay_lambda else {})},
                coefficients={"task": 1.0, **({"replay": replay_lambda} if replay_lambda else {})},
                label=f"{row['trajectory']} update {update}")
            replay_rows = (streams_lib.replay_batch_rows(replay_order, update,
                                                         batch_rows=int(batch_rows))
                           if replay_lambda else None)
            task_diagnostics, replay_diagnostics = [], []
            with ledger.segment("optimizer"):
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, int(batch_rows), int(microbatch_rows)):
                    stop = min(start + int(microbatch_rows), int(batch_rows))
                    counts, means = {}, {}
                    task_mean, task_block = task_batch(chosen_rows[start:stop],
                                                       rejected_rows[start:stop])
                    counts["task"], means["task"] = stop - start, task_mean
                    task_diagnostics.append((stop - start, task_block))
                    if replay_lambda:
                        replay_mean, replay_block = replay_batch(replay_rows[start:stop])
                        counts["replay"], means["replay"] = stop - start, replay_mean
                        replay_diagnostics.append((stop - start, replay_block))
                    accumulator.add(counts, means, backward=lambda tensor: tensor.backward())
                components = accumulator.finish()
                step = replay_lib.clip_and_step(policy.model, optimizer, scheduler,
                                                gradient_clip=gradient_clip)
            clipped += int(bool(step["clipped"]))
            exposures["chosen"] += int(batch_rows)
            exposures["rejected"] += int(batch_rows) if row["task"] == "ipo" else 0
            exposures["replay"] += int(batch_rows) if replay_lambda else 0
            updates_journal.append({
                "record_kind": "update", "update": update,
                "cycle": int(stream.cycle_of_position[(update - 1) * int(batch_rows)]),
                "spans_cycle_boundary": bool(
                    stream.cycle_of_position[(update - 1) * int(batch_rows)]
                    != stream.cycle_of_position[update * int(batch_rows) - 1]),
                "exposures": dict(exposures),
                "task_loss": components["component_means"]["task"],
                "replay_loss": components["component_means"].get("replay"),
                "weighted_total": components["weighted_total"],
                "replay_lambda": replay_lambda,
                "microbatches": components["microbatches"],
                "margin_mean": weighted_diagnostic(task_diagnostics, "margin"),
                "sign_accuracy": weighted_diagnostic(task_diagnostics, "sign_correct"),
                "chosen_nll_per_residue": weighted_diagnostic(task_diagnostics,
                                                              "chosen_nll_per_residue"),
                "conditional_kl_mean": weighted_diagnostic(replay_diagnostics,
                                                           "mean_conditional_kl"),
                **step})
            if heartbeat is not None:
                heartbeat.beat(trajectory=row["trajectory"], update=update,
                               exposures=dict(exposures), checks=checks)

            due, reason = cadence.due(update, final=(update == total_updates))
            if not due:
                continue
            with ledger.segment("gate"):
                verdict = gate_check(update=update, reason=reason, exposures=dict(exposures))
            checks += 1
            with ledger.segment("preservation"):
                preservation_block = preservation(update=update, reason=reason)
            monitor_journal.append(dict(verdict, record_kind="gate_verdict",
                                        preservation=preservation_block,
                                        exposures=dict(exposures), update=update))
            if not verdict.get("passed"):
                stop_reason = verdict.get("stop_reason") or "parent_relative_likelihood_breach"
                with ledger.segment("io"):
                    failure = saver.save_failed(update=update, exposures=dict(exposures),
                                                stop_reason=stop_reason)
                monitor_journal.append({"record_kind": "snapshot", "update": update,
                                        "passed": False, "failed_state": failure,
                                        "consequence": ("the trajectory stops here. This state is "
                                                        "a diagnostic and is never a reached "
                                                        "endpoint; the queue continues with the "
                                                        "next arm.")})
                status = STATUS_STOPPED
                persist("gate_stop")
                break
            with ledger.segment("io"):
                last_passing = saver.save_last_passing(update=update, exposures=dict(exposures),
                                                       check=checks)
            monitor_journal.append({"record_kind": "snapshot", "update": update, "passed": True,
                                    "last_passing": last_passing})
            if update in endpoints_declared:
                # Order matters: the gate has passed at this exact update and the
                # weights are on disk and re-read before anything is evaluated, and
                # the endpoint record is created only after the evaluation is saved.
                with ledger.segment("io"):
                    checkpoint = saver.save_endpoint(update=update, exposures=dict(exposures))
                record = on_endpoint(update=update, exposures=dict(exposures),
                                     checkpoint=checkpoint, gate=verdict,
                                     preservation=preservation_block, ledger=ledger)
                # Journalled as a record, immediately. A reached endpoint that only
                # existed in memory would disappear from an interrupted run, and its
                # checkpoint file alone would not be allowed to stand in for it.
                endpoints_journal.append(dict(record, record_kind="exposure_endpoint",
                                              update=int(update)))
                reached[str(update)] = record
                persist(f"endpoint_{update}")
            persist(f"check_{update}")
        else:
            status = STATUS_COMPLETED
    except BaseException as error:                              # noqa: BLE001 - recorded, re-raised
        # Machinery, not science. The trajectory gets its own terminal status --
        # `failed` is not `stopped_by_gate` and not `incomplete` -- and the
        # exception is re-raised below so the campaign fail-stops rather than
        # running thirty more arms under whatever is wrong.
        status, fatal, raised = STATUS_FAILED, f"{type(error).__name__}: {error}", error
        stop_reason = stop_reason or fatal
        # The journals are flushed and fsynced per record, so they are readable
        # here, and they are what says which updates completed. ``update`` at this
        # point counts an attempt.
        persist("failed", completed=durable_progress(directory)["updates"])
    finally:
        updates_journal.close()
        monitor_journal.close()
        endpoints_journal.close()

    # The journals, not the in-memory counters: after a failure mid-update the
    # loop's `update` counts an attempt, and an attempted update is never reported
    # as a completed one. There is no fallback to the in-flight counters when the
    # journals are empty -- an injected failure on the very first append produced
    # zero journalled updates beside four in-memory exposures, and pairing those
    # under "counters from the journals" is exactly the claim this must not make.
    # Zero is a truthful answer; the attempted counts are kept in their own fields.
    durable = durable_progress(directory)
    completed_updates = durable["updates"] if fatal else update
    counted_exposures = dict(durable["exposures"]) if fatal else dict(exposures)
    for key, value in reached.items():
        durable["endpoints_reached"].setdefault(key, value)
    reached = durable["endpoints_reached"] if fatal else reached
    unreached = {str(value): {"reason": ("the trajectory stopped before this exposure endpoint; "
                                         "no checkpoint exists and none is substituted")}
                 for value in endpoints_declared if str(value) not in reached}
    if status == STATUS_COMPLETED and unreached:
        status = STATUS_STOPPED if stop_reason else STATUS_INCOMPLETE
    document = {
        "schema_version": TRAJECTORY_SCHEMA, "record_kind": "trajectory_status",
        "trajectory": row["trajectory"], "arm_id": row["arm_id"], "task": row["task"],
        "replay_lambda": replay_lambda, "seed": int(row["seed"]),
        "status": status, "stop_reason": stop_reason,
        "updates": completed_updates, "declared_updates": total_updates,
        "attempted_updates": update, "checks": checks,
        "clipped_updates": clipped,
        "exposures": counted_exposures,
        "attempted_exposures": dict(exposures),
        "endpoints_reached": {key: dict(value) for key, value in sorted(reached.items())},
        "endpoints_not_reached": unreached,
        "last_passing": last_passing,
        "identity": dict(identity),
        "cost": ledger.document(),
        "durable": {key: value for key, value in durable.items() if key != "endpoints_reached"},
        "wall_seconds": time.perf_counter() - started,
        "resume_policy": ("none. An interrupted or failed trajectory is never restarted under "
                          "this identity and no partial state is adopted."),
        "completed_at": paths.utc_now()}
    if fatal:
        document["failure"] = {
            "error": fatal,
            "consequence": ("an unexpected exception. The trajectory is `failed`, which is the "
                            "machinery and not a declared scientific outcome, and the campaign "
                            "stops rather than continuing the queue under it."),
            "counters_from": ("the append-only journals, not the in-flight loop counters. "
                              "`updates` and `exposures` are the journalled ones and are zero "
                              "when nothing was journalled; `attempted_updates` and "
                              "`attempted_exposures` are what the loop had counted when it "
                              "failed, and are never presented as completed work.")}
        write_terminal_status(directory, document, register=register_status)
        persist("terminal", completed=completed_updates)
        # Re-raised rather than returned: a returned failure document would let the
        # caller write a status and move to the next arm, which is the opposite of
        # fail-stop. The terminal `failed` status is already on disk.
        raise raised
    persist("terminal")
    return document


def write_terminal_status(directory, document, *, register=None):
    """Write a trajectory's terminal status once. A completed one is never rewritten.

    ``register(path=..., document=...)`` binds the finished record into the run's
    completion authority. A terminal status is the trajectory's own statement of
    what it reached, so a later verification has to be able to hold it to its
    bytes; a status that nothing recorded could be edited or deleted and no
    coverage check would notice. Running trajectories write ``trajectory_progress``
    instead, which is mutable by design and is not bound here.
    """
    directory = Path(directory)
    path = directory / STATUS_JSON
    require(document.get("status") in TERMINAL_STATUSES,
            f"{path}: {document.get('status')!r} is not a terminal status")
    if path.is_file():
        existing = paths.read_json(path)
        require(paths.scientific_projection(existing) == paths.scientific_projection(document),
                f"{path} already records a different terminal outcome "
                f"({existing.get('status')} vs {document.get('status')}). A completed trajectory "
                "is not rewritten; a different result belongs in a new run directory.")
        digest = paths.sha256_file(path)
    else:
        digest = paths.write_json(path, document)
    if register is not None:
        register(path=path, document=document)
    return digest


def read_terminal_status(directory):
    path = Path(directory) / STATUS_JSON
    return paths.read_json(path) if path.is_file() else None


def mark_begun_incomplete(directory, *, reason, owner=None, register=None):
    """A trajectory that began and has no terminal status becomes ``incomplete``.

    Called by the *next* campaign process, after it has taken the lock and
    therefore knows the previous owner is gone. Nothing is adopted, nothing is
    resumed, and the scientific artifacts the dead run did finish -- its journals,
    its saved states, its reached endpoints -- are left exactly as they are.

    Every counter it reports comes from the journals, including zero. The progress
    file's counters are carried in their own ``progress_file_*`` fields and are
    never promoted into the reported ones when the journals are empty: "the
    journals say nothing completed" and "the last check said 1,000" are different
    statements and only the first is durable evidence.
    """
    directory = Path(directory)
    identity_path = directory / IDENTITY_JSON
    if not identity_path.is_file() or (directory / STATUS_JSON).is_file():
        return None
    identity = paths.read_json(identity_path)
    progress_path = directory / PROGRESS_JSON
    progress = paths.read_json(progress_path) if progress_path.is_file() else {}
    # The journals are the authority, not the progress file. Progress is written at
    # checks, so a kill between two checks leaves it behind by up to a whole
    # interval -- and the endpoints it lists are bare update numbers, which is not
    # enough to be an endpoint. The completed endpoint RECORDS are recovered with
    # their identities, checkpoint digests and evaluations, or not at all.
    durable = durable_progress(directory)
    declared = list(progress.get("declared_endpoints") or [])
    reached = dict(durable["endpoints_reached"])
    dropped = sorted(set(str(value) for value in (progress.get("endpoints_reached") or []))
                     - set(reached))
    document = {
        "schema_version": TRAJECTORY_SCHEMA, "record_kind": "trajectory_status",
        "trajectory": identity.get("trajectory"), "arm_id": identity.get("arm_id"),
        "task": identity.get("task"), "replay_lambda": identity.get("replay_lambda"),
        "seed": identity.get("seed"),
        "status": STATUS_INCOMPLETE, "stop_reason": None,
        "updates": max(int(durable["updates"]), 0),
        "progress_file_updates": progress.get("update", 0),
        "checks": int(durable["checks"]),
        "progress_file_checks": progress.get("checks", 0),
        "declared_updates": (declared or [None])[-1],
        "exposures": dict(durable["exposures"]),
        "progress_file_exposures": progress.get("exposures") or {},
        "endpoints_reached": {key: dict(value) for key, value in sorted(reached.items())},
        "endpoints_not_reached": {
            str(value): {"reason": ("the interrupted run had not reached this exposure endpoint; "
                                    "no endpoint record exists and none is substituted")}
            for value in declared if str(value) not in reached},
        "endpoint_records_not_recovered": dropped,
        "last_passing": progress.get("last_passing"),
        "identity": identity,
        "cost": progress.get("cost"),
        "durable": {key: value for key, value in durable.items() if key != "endpoints_reached"},
        "incomplete_reason": reason,
        "owner_evidence": dict(owner or {}),
        "recovery": ("completed updates, exposures, checks and reached endpoint records are "
                     "recovered from the trajectory's own append-only journals, including when "
                     "they recovered nothing. The progress file's counters are kept beside them "
                     "in their own progress_file_* fields for comparison and never stand in for "
                     "them; a checkpoint file on disk is never read as a reached endpoint."),
        "resume_policy": ("none. This trajectory began and did not finish; it is not restarted "
                          "under this identity and no partial state is adopted. It is never "
                          "restarted. Its saved artifacts are left untouched as evidence."),
        "completed_at": paths.utc_now()}
    write_terminal_status(directory, document, register=register)
    return document


def campaign_state(run_root, queue, *, lock_path=None):
    """What every trajectory's own artifacts say, with nothing inferred from a process.

    A queued trajectory is one with no directory. A running one is one whose
    directory exists, whose terminal status is absent and whose lock is *held*. The
    same directory with an acquirable lock is an interrupted run, and it is
    reported as such rather than as progress.
    """
    run_root = Path(run_root)
    observation = None if lock_path is None else lock_state(lock_path)
    state = None if observation is None else observation["state"]
    live = None if state is None else state == LOCK_HELD
    known = state in (LOCK_MISSING, LOCK_RELEASED, LOCK_HELD)
    rows, counts = [], {}
    for entry in queue:
        directory = run_root / "trajectories" / entry["trajectory"]
        terminal = read_terminal_status(directory)
        progress_path = directory / PROGRESS_JSON
        progress = paths.read_json(progress_path) if progress_path.is_file() else {}
        durable = durable_progress(directory) if directory.is_dir() else None
        if terminal is not None:
            status = terminal["status"]
        elif not directory.is_dir():
            status = STATUS_QUEUED
        elif live:
            status = STATUS_RUNNING
        elif known:
            status = STATUS_INCOMPLETE
        else:
            # The lock was not asked about, or could not be read. A begun
            # trajectory is not declared abandoned on an unanswered question; it is
            # reported as running with the observation attached, and the next
            # campaign process -- which must actually take the lock -- settles it.
            status = STATUS_RUNNING
        counts[status] = counts.get(status, 0) + 1
        source = terminal if terminal is not None else progress
        reached = sorted((terminal or {}).get("endpoints_reached")
                         or (durable or {}).get("endpoints_reached") or [],
                         key=lambda value: int(value))
        counters = terminal if terminal is not None else (durable or {})
        rows.append({**entry, "status": status,
                     "updates": int(counters.get("updates") or 0),
                     "checks": int(counters.get("checks") or 0),
                     "exposures": counters.get("exposures") or {},
                     "stop_reason": source.get("stop_reason"),
                     "endpoints_reached": reached,
                     "counters_from": ("its terminal status" if terminal is not None
                                       else "its append-only journals"),
                     "liveness": state or "not_observed",
                     "cost": source.get("cost"),
                     "terminal": terminal is not None})
    return {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "campaign_state",
            "generated_at": paths.utc_now(),
            "writer_alive": live,
            "lock": observation,
            "trajectories": rows, "counts": counts, "total": len(rows),
            "status_note": ("queued = no directory; running = a directory with a live lock owner; "
                            "incomplete = a directory whose owner is gone and which is never "
                            "resumed; stopped_by_gate = a declared scientific outcome. A lock that "
                            "could not be read leaves a begun trajectory reported as running, "
                            "never as abandoned."),
            "counter_note": ("a non-terminal trajectory's updates, exposures, checks and reached "
                             "endpoints come from its append-only journals, not from the progress "
                             "file, which is only written at checks")}


def first_update_and_check(run_root, queue):
    """Evidence that a launch actually started working: one real update, one passed check.

    "Healthy" is not reportable from a process that started. It needs a journalled
    completed optimizer update and a gate check that was measured and passed, and
    both are read back off disk here rather than remembered.

    What this establishes is **historical**: the first update and the first check of
    the first trajectory that has both. It stays true after the campaign later dies
    or fails, so it is never the current-health statement on its own --
    :func:`launch_health` combines it with what the campaign is doing now.
    """
    run_root = Path(run_root)
    for entry in queue:
        directory = run_root / "trajectories" / entry["trajectory"]
        updates = directory / UPDATES_JSONL
        monitor = directory / MONITOR_JSONL
        if not updates.is_file() or not monitor.is_file():
            continue
        first_update, first_check = None, None
        for line in updates.read_text(encoding="utf-8").splitlines():
            if line.strip():
                first_update = json.loads(line)
                break
        for line in monitor.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_kind") == "gate_verdict":
                first_check = record
                break
        if first_update is not None and first_check is not None:
            return {"trajectory": entry["trajectory"],
                    "first_update_completed": int(first_update.get("update", 0)),
                    "first_update_loss": first_update.get("weighted_total"),
                    "first_check_update": int(first_check.get("update", 0)),
                    "first_check_passed": bool(first_check.get("passed")),
                    "first_check_D": first_check.get("D"),
                    "healthy": bool(first_update.get("update") and first_check.get("passed")),
                    "basis": "read back from the trajectory's own journals"}
    return {"trajectory": None, "first_update_completed": None, "first_check_update": None,
            "first_check_passed": None, "healthy": False,
            "basis": ("no trajectory has both a journalled completed update and a measured gate "
                      "check yet; a started process is not a healthy launch")}


#: What a health tile is allowed to say. ``launch_verified`` is a statement about
#: the past; only ``healthy`` claims anything about now.
HEALTH_NOT_ESTABLISHED = "not_established"
HEALTH_LAUNCH_VERIFIED = "launch_verified"
HEALTH_RUNNING = "healthy"
HEALTH_STOPPED_AFTER_LAUNCH = "launch_verified_but_not_running"
HEALTH_FAILED_AFTER_LAUNCH = "launch_verified_but_campaign_failed"


def launch_health(evidence, *, writer_state, recorded_status=None, failed_trajectories=0):
    """Combine the historical launch evidence with what the campaign is doing now.

    The first update and the first passed check stay true forever once they happen.
    A tile that printed "Healthy" off that evidence alone would go on printing it
    beside a dead writer and a failed campaign, which is precisely the reassurance
    this screen is not allowed to give. So the evidence is kept -- it is real and it
    is what a launch review needs -- and the *label* is qualified by the current
    writer state and by whether any trajectory failed.
    """
    established = bool((evidence or {}).get("healthy"))
    if not established:
        status = HEALTH_NOT_ESTABLISHED
        detail = (evidence or {}).get("basis")
    elif recorded_status == "failed" or int(failed_trajectories or 0) > 0:
        status = HEALTH_FAILED_AFTER_LAUNCH
        detail = ("the first update and the first gate check are real and are kept, but a later "
                  "failure means this is not a statement about the campaign now")
    elif writer_state == LOCK_HELD:
        status = HEALTH_RUNNING
        detail = "a live writer holds the lock and the launch evidence is on disk"
    elif writer_state in (LOCK_MISSING, LOCK_RELEASED):
        status = HEALTH_STOPPED_AFTER_LAUNCH
        detail = ("nobody holds the write lock. The launch evidence is historical; whether the "
                  "campaign finished or was interrupted is in the trajectory statuses")
    else:
        status = HEALTH_LAUNCH_VERIFIED
        detail = ("the launch evidence is on disk and the writer's state could not be observed, "
                  "so no claim is made about now")
    return dict(evidence or {}, status=status, detail=detail,
                writer_state=writer_state, established=established,
                scope=("launch evidence is historical: the first journalled update and the first "
                       "measured check. It is never displayed as current health on its own."))
