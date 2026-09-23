"""One trajectory: the loop, its durable state, and a resume that is actually exact.

The inherited ``her2_replay_campaign.run_trajectory`` declares
``resume_policy: "none"`` and persists model tensors only. A 63,000-update
queue cannot be run that way on a box that may be interrupted, so this loop adds
real resume -- and resume is the part that is easy to get plausibly wrong:

* **the optimizer and the scheduler are state, not derived.** AdamW's ``step``
  counter and ``LambdaLR``'s ``last_epoch`` both live in their ``state_dict``.
  Reconstructing them from the update number shifts the inherited ``step + 1``
  warmup by one, which at update 1 is a factor of two in the learning rate and
  is invisible in a loss curve.
* **there is no GradScaler**, and its absence is recorded with the reason rather
  than left as a missing key: the forward is float32 and the reductions are
  float64, so there is no autocast region to scale. A resume that silently
  invented one would change the update.
* **RNG is captured for all four generators** -- Python, NumPy, torch CPU and
  every CUDA device -- and restored before the first continued update.
* **the journals are the authority for what happened.** The state is written at
  every FULL check, so resume replays at most one ``full_interval`` of updates
  between the last saved state and the crash. Those updates are deterministic, so
  they reproduce the same weights, but they would append duplicate journal
  records; :func:`durable_progress` therefore de-duplicates by update number and
  takes the last record for each, so a resumed run cannot double-count updates,
  exposures or reached endpoints. A journal whose final line was cut short is
  repaired by :func:`repair_truncated_journal` *before* the writer opens it in
  append mode, because appending onto a half-written line is how an interrupted
  tail becomes permanent interior corruption.
* **an interrupted endpoint is finished, not skipped.** An endpoint is recorded
  as ``pending`` in the state that carries its weights and cleared only after its
  checkpoint and journal record exist. A resume that finds a pending endpoint
  publishes it from those exact weights before running another update.

Stopping science and crashing are different outcomes and get different statuses.
A gate breach is ``stopped_by_gate``, an outcome. An unexpected exception is
``failed``, is journalled with counters taken from the journals rather than from
the in-flight loop, and is re-raised: there is no broad ``except`` here that
swallows an error and continues with scientifically corrupt state.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import her2_replay as replay_lib
from . import her2_support_paths as paths
from .her2_nf_contract import NF_SCHEMA
from .her2_replay_campaign import read_journal, state_schema
from .her2_nf_cost import BudgetExhausted, CostLedger
from .her2_nf_storage import feed_tensor, load_cpu, state_dict_digest
from .her2_runtime import require

TRAJECTORY_SCHEMA = "her2-next-flight-trajectory/1"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped_by_gate"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_STOPPED, STATUS_INCOMPLETE, STATUS_FAILED)

UPDATES_JSONL = "updates.jsonl"
MONITOR_JSONL = "monitor.jsonl"
ENDPOINTS_JSONL = "endpoints.jsonl"
RESUME_JSONL = "resume.jsonl"
STATUS_JSON = "status.json"
PROGRESS_JSON = "trajectory_progress.json"
RESUME_STATE = "resume_state.pt"
FAILED_STATE = "failed_state.pt"

#: Why there is no scaler, recorded in every saved state. The TRAINING forward,
#: loss and reductions are float32 throughout -- that is the inherited protocol
#: and it is not changed here. float64 appears only in diagnostic accumulation
#: (tails, drift, total correlation) after the optimizer step, which is not an
#: autocast region either.
NO_SCALER_REASON = ("this protocol trains entirely in float32, with no autocast region anywhere "
                    "in the update, so torch.amp.GradScaler was never constructed. float64 is "
                    "used only for post-hoc diagnostic accumulation and is not part of any "
                    "training reduction. The field is present and null with this reason so a "
                    "resume cannot silently invent one.")


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

def capture_rng_state():
    """Every generator that can affect a continued run, in one serializable block."""
    import torch
    return {"python": random.getstate(),
            "numpy": np.random.get_state(legacy=True),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
            "cuda_devices": (torch.cuda.device_count() if torch.cuda.is_available() else 0)}


def restore_rng_state(state):
    """Restore all four generators; a CUDA-count mismatch is refused, not ignored."""
    import torch
    random.setstate(_as_tuple(state["python"]))
    np.random.set_state(_as_tuple(state["numpy"]))
    torch.set_rng_state(_as_cpu_byte_tensor(state["torch_cpu"]))
    saved = list(state.get("torch_cuda") or [])
    if torch.cuda.is_available() and saved:
        require(len(saved) == torch.cuda.device_count(),
                f"the saved state carries {len(saved)} CUDA generator states and this host has "
                f"{torch.cuda.device_count()}. Restoring a mismatched set would leave one device's "
                "stream at an undeclared position.")
        torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(value) for value in saved])
    return True


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def _as_cpu_byte_tensor(value):
    import torch
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().to("cpu").to(torch.uint8)


#: Payload keys the whole-state digest covers. ``state`` is excluded because its
#: own ``state_sha256`` already covers it; ``payload_sha256`` is excluded because
#: it is the digest.
_DIGESTED_KEYS = ("identity", "state_sha256", "state_schema", "optimizer", "scheduler",
                  "scaler", "rng", "progress", "stream_digests")


def _payload_digest(payload):
    """A digest over everything a resume restores EXCEPT the model tensors.

    Optimizer moments, the scheduler's ``last_epoch``, all four RNG streams and
    the cadence/exposure cursors are hashed together. The model tensors are
    covered separately by ``state_sha256``; hashing them twice would double the
    cost of every save for nothing.
    """
    import hashlib
    digest = hashlib.sha256()
    for key in _DIGESTED_KEYS:
        digest.update(key.encode("utf-8"))
        _feed_digest(digest, payload.get(key))
    return digest.hexdigest()


def _feed_digest(digest, node):
    import torch
    if isinstance(node, torch.Tensor):
        digest.update(b"T")
        digest.update(str(tuple(node.shape)).encode("utf-8"))
        feed_tensor(digest, node)
        return
    if isinstance(node, np.ndarray):
        digest.update(b"A")
        digest.update(str(node.shape).encode("utf-8"))
        digest.update(memoryview(np.ascontiguousarray(node)).cast("B"))
        return
    if isinstance(node, dict):
        digest.update(b"D")
        for key in sorted(node, key=repr):
            digest.update(repr(key).encode("utf-8"))
            _feed_digest(digest, node[key])
        return
    if isinstance(node, (list, tuple)):
        digest.update(b"L")
        for item in node:
            _feed_digest(digest, item)
        return
    digest.update(b"S")
    digest.update(repr(node).encode("utf-8"))


def _optimizer_step_count(state):
    """AdamW's own per-parameter ``step`` counter, for an equivalence comparison."""
    values = []
    for entry in (dict(state or {}).get("state") or {}).values():
        step = dict(entry).get("step")
        if step is None:
            continue
        try:
            values.append(int(float(step)))
        except (TypeError, ValueError):                         # pragma: no cover - exotic states
            continue
    return max(values) if values else None


# ---------------------------------------------------------------------------
# what a resumable trajectory knows about itself
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryState:
    """The bookkeeping half of a trajectory. The tensors live beside it."""

    update: int = 0
    checks: int = 0
    sentinels: int = 0
    clipped: int = 0
    status: str = STATUS_RUNNING
    stop_reason: object = None
    exposures: dict = field(default_factory=lambda: {"chosen": 0, "rejected": 0,
                                                     "replay": 0, "tail": 0})
    endpoints_reached: list = field(default_factory=list)
    checkpoints_saved: list = field(default_factory=list)
    previous_sentinel_D: object = None
    stream_position: int = 0
    replay_position: int = 0
    resumes: int = 0
    #: The endpoint whose weights are in this saved state but whose checkpoint and
    #: journal record were not published yet. Set BEFORE the save that carries
    #: those weights, cleared after the publication; a resume that finds one
    #: completes it rather than stepping past it.
    pending_endpoint: object = None
    pending_checkpoint: object = None

    def document(self):
        return {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                for k, v in asdict(self).items()}


def repair_truncated_journal(path):
    """Make an append-only journal safe to append to after a kill.

    ``JsonlJournal`` opens in append mode, so a final line whose bytes were cut
    short by the kernel becomes the PREFIX of the next record: a corrupt interior
    line, which :func:`read_journal` refuses outright and which no later repair
    can separate. Two distinct states are fixed here, both before the writer
    opens the file:

    * a final line that does not parse -- truncated at the last complete record;
    * a final line that parses but has no terminating newline -- the newline is
      added, so the next record starts on its own line.

    Nothing already-complete is ever dropped, and a damaged line anywhere but at
    the end is left alone for :func:`read_journal` to refuse.
    """
    import json as json_lib

    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return {"path": str(target), "repaired": False, "reason": "absent or empty"}
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return {"path": str(target), "repaired": False, "reason": "no lines"}
    final = lines[-1]
    try:
        json_lib.loads(final)
        parses = True
    except ValueError:
        parses = False
    if parses and text.endswith("\n"):
        return {"path": str(target), "repaired": False, "reason": "already terminated"}
    kept = lines[:-1] if not parses else lines
    body = "".join(line + "\n" for line in kept)
    temporary = target.with_name(target.name + ".repair")
    temporary.write_text(body, encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return {"path": str(target), "repaired": True,
            "dropped_truncated_final_line": not parses,
            "added_missing_newline": parses,
            "lines_before": len(lines), "lines_after": len(kept),
            "reason": ("the final line was cut short by a kill" if not parses else
                       "the final record was complete but its newline was not written")}


def exposures_at(update, *, batch_rows, uses_rejected, replay_rows, tail_rows):
    """Exposures as a pure function of the update number.

    Deterministic rather than accumulated, so replayed updates after a resume
    cannot inflate a counter, and a journal that lost its last line cannot
    deflate one below what its update count implies.
    """
    update, batch_rows = int(update), int(batch_rows)
    return {"chosen": update * batch_rows,
            "rejected": update * batch_rows if uses_rejected else 0,
            "replay": update * int(replay_rows),
            "tail": update * int(tail_rows)}


class TrajectoryStateStore:
    """Atomic save and verified load of the full resumable state.

    Validated on the temporary file before it replaces anything, exactly as the
    inherited checkpoint saver does: a state that cannot be read back is a failed
    save, not a reason to lose the one that worked.
    """

    def __init__(self, directory, *, identity):
        self.directory = Path(directory)
        self.identity = dict(identity)
        self.saves = 0
        self.wall_seconds = 0.0

    def path(self):
        return self.directory / RESUME_STATE

    def exists(self):
        return self.path().is_file()

    def save(self, *, policy, optimizer, scheduler, state, stream_digests):
        import torch
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        tensors = policy.model.state_dict()
        digest = state_dict_digest(tensors)
        payload = {"schema_version": TRAJECTORY_SCHEMA, "kind": "resume_state",
                   "identity": dict(self.identity),
                   "state": tensors, "state_sha256": digest,
                   "state_schema": state_schema(tensors),
                   "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(),
                   "scaler": None, "scaler_reason": NO_SCALER_REASON,
                   "rng": capture_rng_state(),
                   "progress": state.document(),
                   "stream_digests": dict(stream_digests),
                   "saved_at": paths.utc_now()}
        # Every non-tensor half of the payload is covered too. A model digest
        # alone certifies the weights and says nothing about the AdamW moments,
        # the LambdaLR step counter, the four RNG streams or the cadence
        # counters -- and a resume that restores correct weights beside a
        # silently damaged optimizer produces a plausible, wrong trajectory.
        payload["payload_sha256"] = _payload_digest(payload)
        target = self.path()
        temporary = target.with_name(target.name + ".tmp")
        previous = paths.sha256_file(target) if target.is_file() else None
        try:
            torch.save(payload, temporary)
            restored = load_cpu(temporary)
            recomputed = state_dict_digest(restored["state"])
            require(recomputed == digest,
                    f"the resume state on disk digests to {recomputed} and this update produced "
                    f"{digest}")
            require(_payload_digest(restored) == payload["payload_sha256"],
                    "the reloaded resume state does not reproduce the optimizer/scheduler/RNG/"
                    "cursor digest that was written with it")
            require(restored["progress"]["update"] == state.update,
                    "the reloaded resume state disagrees about its update number")
            require(restored["identity"] == self.identity,
                    "the reloaded resume state names a different trajectory identity")
            del restored
        except BaseException as error:
            if temporary.is_file():
                temporary.unlink()
            raise ValueError(
                f"{target}: the resume state did not validate before publication "
                f"({type(error).__name__}: {error}). The temporary was removed and the state "
                f"already at this path was not touched (sha256 {previous})." ) from error
        os.replace(temporary, target)
        elapsed = time.perf_counter() - started
        self.saves += 1
        self.wall_seconds += elapsed
        return {"file": RESUME_STATE, "sha256": paths.sha256_file(target),
                "state_sha256": digest, "payload_sha256": payload["payload_sha256"],
                "update": int(state.update),
                "pending_endpoint": (state.pending_endpoint or {}).get("update")
                if isinstance(state.pending_endpoint, dict) else state.pending_endpoint,
                "bytes": int(target.stat().st_size), "wall_seconds": elapsed,
                "covers": ["model", "optimizer", "scheduler", "rng", "progress",
                           "stream_digests", "pending_endpoint"],
                "validated": ("written to a temporary, reloaded, re-digested over the WHOLE "
                              "payload, then published")}

    def load(self, *, policy, optimizer, scheduler, expect_stream_digests=None):
        import torch
        target = self.path()
        require(target.is_file(), f"{target} does not exist; there is nothing to resume from")
        payload = load_cpu(target)
        require(payload.get("schema_version") == TRAJECTORY_SCHEMA,
                "unsupported resume-state schema")
        observed = dict(payload.get("identity") or {})
        differing = sorted(key for key in set(observed) | set(self.identity)
                           if observed.get(key) != self.identity.get(key))
        require(not differing,
                f"{target}: the saved state names a different trajectory in {differing}. Resuming "
                "another arm's or another parent's weights under this identity would attribute "
                "its updates to this trajectory.")
        recomputed = state_dict_digest(payload["state"])
        require(recomputed == payload.get("state_sha256"),
                f"{target}: the tensors digest to {recomputed} and the payload records "
                f"{payload.get('state_sha256')}. A corrupted state is not resumed from.")
        require(payload.get("payload_sha256") is not None,
                f"{target}: the saved state carries no whole-payload digest, so its optimizer, "
                "scheduler, RNG and cursors cannot be checked. It is not resumed from.")
        require(_payload_digest(payload) == payload["payload_sha256"],
                f"{target}: the optimizer/scheduler/RNG/cursor half of the payload does not "
                "reproduce its recorded digest. Correct weights beside a damaged optimizer "
                "produce a plausible, wrong trajectory, so this is refused.")
        if expect_stream_digests is not None:
            saved = dict(payload.get("stream_digests") or {})
            wrong = sorted(key for key in set(saved) | set(expect_stream_digests)
                           if saved.get(key) != expect_stream_digests.get(key))
            require(not wrong,
                    f"{target}: the saved state was produced against different data streams "
                    f"({wrong}). Continuing would change which rows this trajectory consumed.")
        device = next(policy.model.parameters()).device
        policy.model.load_state_dict(payload["state"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        require(payload.get("scaler") is None,
                "a scaler state was found; this protocol declares none and would not know what to "
                "do with it")
        restore_rng_state(payload["rng"])
        require(state_dict_digest(policy.model.state_dict()) == payload["state_sha256"],
                "Strict loading did not restore the checkpoint model exactly")
        actual = dict(payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                      rng=capture_rng_state())
        require(_payload_digest(actual) == payload["payload_sha256"],
                "Optimizer, scheduler or RNG state changed while restoring the checkpoint")
        del actual
        progress = dict(payload["progress"])
        state = TrajectoryState(**{key: progress[key] for key in progress
                                   if key in TrajectoryState.__dataclass_fields__})
        state.resumes = int(state.resumes) + 1
        return {"state": state, "state_sha256": recomputed,
                "payload_sha256": payload.get("payload_sha256"),
                "saved_at": payload.get("saved_at"),
                "optimizer_steps": _optimizer_step_count(payload.get("optimizer")),
                "scheduler_last_epoch": (payload.get("scheduler") or {}).get("last_epoch"),
                "scaler_reason": payload.get("scaler_reason")}


# ---------------------------------------------------------------------------
# durable progress, de-duplicated
# ---------------------------------------------------------------------------

def durable_progress(directory):
    """What the append-only journals prove finished, immune to a replayed resume.

    Records are keyed by update number and the last one for each key wins, so the
    deterministic replay of the few updates between the last saved state and a
    crash contributes no extra counts and no extra endpoints.
    """
    directory = Path(directory)
    updates = read_journal(directory / UPDATES_JSONL, record_kind="update")
    monitor = read_journal(directory / MONITOR_JSONL, record_kind="gate_verdict")
    sentinels = read_journal(directory / MONITOR_JSONL, record_kind="sentinel_check")
    endpoints = read_journal(directory / ENDPOINTS_JSONL, record_kind="exposure_endpoint")
    resumes = read_journal(directory / RESUME_JSONL, record_kind="resume_boundary")
    by_update = {}
    for record in updates["records"]:
        key = record.get("update")
        if key is not None:
            by_update[int(key)] = record
    checks = {int(record["update"]): record for record in monitor["records"]
              if record.get("update") is not None}
    sentinel_by_update = {int(record["update"]): record for record in sentinels["records"]
                          if record.get("update") is not None}
    reached = {}
    for record in endpoints["records"]:
        key = record.get("update")
        if key is not None:
            reached[str(int(key))] = record
    last = by_update[max(by_update)] if by_update else None
    return {"updates": int(max(by_update)) if by_update else 0,
            "distinct_update_records": len(by_update),
            "journalled_lines": int(updates["lines"]),
            "replayed_update_records": int(updates["lines"]) - len(by_update)
            if updates["lines"] else 0,
            "exposures": dict((last or {}).get("exposures") or {}),
            "checks": len(checks), "sentinels": len(sentinel_by_update),
            "endpoints_reached": reached,
            "endpoints_reached_updates": sorted(reached, key=lambda value: int(value)),
            "resume_boundaries": resumes["records"],
            "truncated_trailing_update_line": bool(updates["truncated_trailing_line"]),
            "unterminated_final_update_line": bool(updates["unterminated_final_line"]),
            "basis": ("append-only journals, de-duplicated by update number with the last record "
                      "for each key. An attempted update is never counted as a completed one, and "
                      "a deterministic replay after a resume is not counted twice."),
            "forced_kill_note": ("an update completed but not journalled before a kill is lost "
                                 "from this count, which is a lower bound -- the safe direction.")}


# ---------------------------------------------------------------------------
# per-term gradient diagnostics
# ---------------------------------------------------------------------------

def gradient_term_diagnostics(model, task_closure, preservation_closure):
    """Task and preservation gradient norms and their cosine, on one microbatch.

    Two extra backward passes on a single microbatch, at a modest cadence. The
    task gradient is held on the CPU between the passes so the device never holds
    two full gradient copies, and the optimizer's gradients are zeroed afterwards
    so this diagnostic cannot leak into the update that follows it.
    """
    import torch
    model.zero_grad(set_to_none=True)
    task_closure().backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    stored = [p.grad.detach().to("cpu", copy=True) for p in parameters]
    task_norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in stored)))
    model.zero_grad(set_to_none=True)
    preservation_closure().backward()
    keep_sq, dot = 0.0, 0.0
    for parameter, saved in zip(parameters, stored):
        if parameter.grad is None:
            continue
        current = parameter.grad.detach().double()
        keep_sq += float((current ** 2).sum())
        dot += float((current * saved.to(current.device).double()).sum())
    model.zero_grad(set_to_none=True)
    keep_norm = float(np.sqrt(keep_sq))
    return {"task_gradient_norm": task_norm, "preservation_gradient_norm": keep_norm,
            "cosine": (dot / (task_norm * keep_norm)) if task_norm > 0 and keep_norm > 0 else None,
            "ratio": (keep_norm / task_norm) if task_norm > 0 else None,
            "basis": "one microbatch, two extra backward passes; the update itself is unchanged"}


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def run_trajectory(*, row, directory, policy, optimizer, scheduler, stream, replay_order,
                   plan, endpoints, batch_rows, microbatch_rows, task_batch, preservation_batch,
                   full_check, sentinel_check, on_endpoint, gradient_clip, identity,
                   uses_rejected, preservation_family, preservation_lambda, max_updates=None,
                   heartbeat=None, ledger=None, journal_factory=None, resume=True,
                   register_status=None, gradient_diagnostic_interval=100,
                   gradient_diagnostic=None, stream_digests=None, restore_sentinel=None,
                   on_checkpoint=None, checkpoints=(), budget=None):
    """Fit one arm at one seed, resumably, and return its truthful terminal document."""
    from .her2_guarded_trajectory import JsonlJournal

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    import torch
    synchronize = torch.cuda.synchronize if next(policy.model.parameters()).is_cuda else None
    ledger = ledger or CostLedger(path=directory / "cost.json", synchronize=synchronize,
                                  budget=budget)
    factory = journal_factory or JsonlJournal
    # Repair before opening in append mode: a kill can leave the final line half
    # written, and appending onto it would produce a corrupt interior record that
    # read_journal refuses for the whole rest of the run.
    journal_repairs = [repair_truncated_journal(directory / name)
                       for name in (UPDATES_JSONL, MONITOR_JSONL, ENDPOINTS_JSONL, RESUME_JSONL)]
    updates_journal = factory(directory / UPDATES_JSONL)
    monitor_journal = factory(directory / MONITOR_JSONL)
    endpoints_journal = factory(directory / ENDPOINTS_JSONL)
    resume_journal = factory(directory / RESUME_JSONL)
    store = TrajectoryStateStore(directory, identity=identity)
    digests = dict(stream_digests or {})
    total_updates = int(max_updates if max_updates is not None else max(endpoints))
    requested_updates = total_updates
    failures = [record for record in read_journal(directory / MONITOR_JSONL,
                                                 record_kind="gate_verdict")["records"]
                if not record.get("passed")]
    authoritative_stop = min(failures, key=lambda record: int(record["update"])) if failures else None
    if authoritative_stop is not None:
        total_updates = min(total_updates, int(authoritative_stop["update"]))
    endpoints_declared = sorted(int(value) for value in endpoints)
    checkpoints_declared = sorted(int(value) for value in checkpoints)
    replay_rows = int(batch_rows) if preservation_family == "fkl" else 0
    tail_rows = int(batch_rows) if preservation_family == "tail" else 0
    started = time.perf_counter()

    state = TrajectoryState()
    resumed = None
    reconciled = None
    if resume and store.exists():
        with ledger.segment("io"):
            resumed = store.load(policy=policy, optimizer=optimizer, scheduler=scheduler,
                                 expect_stream_digests=digests or None)
        state = resumed["state"]
        durable = durable_progress(directory)
        require(durable["updates"] >= state.update,
                f"the saved state claims update {state.update} and the journals only prove "
                f"{durable['updates']}. A state ahead of its own journal is not resumed from: its "
                "weights would be attributed to updates that were never recorded.")
        resume_journal.append({
            "record_kind": "resume_boundary", "resumed_at": paths.utc_now(),
            "from_update": int(state.update), "journalled_updates": int(durable["updates"]),
            "state_sha256": resumed["state_sha256"],
            "payload_sha256": resumed.get("payload_sha256"),
            "optimizer_steps": resumed.get("optimizer_steps"),
            "whole_state_restore_verified": True,
            "scheduler_last_epoch": resumed.get("scheduler_last_epoch"),
            "saved_at": resumed["saved_at"],
            "replayed_updates": int(durable["updates"]) - int(state.update),
            "journal_repairs": [entry for entry in journal_repairs if entry["repaired"]],
            "pending_endpoint": (state.pending_endpoint or {}).get("update")
            if isinstance(state.pending_endpoint, dict) else None,
            "note": ("updates between the saved state and the last journalled update are replayed "
                     "deterministically. Their journal records are de-duplicated by update number, "
                     "so the replay adds no counts and no endpoints. The bound on that replay is "
                     "the FULL-check interval, which is where the state is saved.")})
        state.endpoints_reached = sorted(set(int(value) for value
                                             in durable["endpoints_reached_updates"]))
        if restore_sentinel is not None:
            restore_sentinel(state.previous_sentinel_D)

    if authoritative_stop is not None and state.update >= int(authoritative_stop["update"]):
        state.status = STATUS_STOPPED
        state.stop_reason = authoritative_stop.get("stop_reason") or "recorded_gate_stop"
    status, stop_reason = STATUS_RUNNING, state.stop_reason
    update = int(state.update)
    fatal = None
    last_state_record = None

    def publish_endpoint(pending):
        """Run the endpoint side effects for weights already in the policy.

        Called both in the loop and, on resume, for an endpoint whose weights
        were saved but whose checkpoint and journal record were never written.
        Without this an injected failure between the save and the publication
        skipped the endpoint permanently: the resume started at ``update + 1``,
        the endpoint number was already behind it, and the trajectory finished
        reporting a checkpoint that does not exist.
        """
        value = int(pending["update"])
        with ledger.segment("io"):
            record = on_endpoint(update=value, exposures=dict(pending["exposures"]),
                                 gate=dict(pending.get("gate") or {}),
                                 preservation=pending.get("preservation"), ledger=ledger)
        endpoints_journal.append(dict(_journal_safe(record), record_kind="exposure_endpoint",
                                      update=value))
        state.endpoints_reached = sorted(set(state.endpoints_reached) | {value})
        state.pending_endpoint = None
        return record

    def publish_checkpoint(pending):
        require(on_checkpoint is not None, "A pending checkpoint needs its publication callback")
        value = int(pending["update"])
        with ledger.segment("io"):
            record = on_checkpoint(update=value, exposures=dict(pending["exposures"]),
                                   gate=dict(pending.get("gate") or {}))
        monitor_journal.append(dict(_journal_safe(record), record_kind="checkpoint", update=value))
        state.checkpoints_saved = sorted(set(state.checkpoints_saved) | {value})
        state.pending_checkpoint = None
        return record

    if resumed is not None and isinstance(state.pending_endpoint, dict):
        pending = dict(state.pending_endpoint)
        if int(pending["update"]) not in set(state.endpoints_reached):
            reconciled = publish_endpoint(pending)
            resume_journal.append({
                "record_kind": "resume_reconciliation", "at": paths.utc_now(),
                "published_endpoint": int(pending["update"]),
                "basis": ("the saved state carried this endpoint's weights and recorded it as "
                          "pending. Its checkpoint and journal record are written now, from "
                          "those exact weights, before any further update runs.")})
            with ledger.segment("io"):
                last_state_record = store.save(policy=policy, optimizer=optimizer,
                                               scheduler=scheduler, state=state,
                                               stream_digests=digests)
        else:
            state.pending_endpoint = None

    if resumed is not None and isinstance(state.pending_checkpoint, dict):
        pending = dict(state.pending_checkpoint)
        publish_checkpoint(pending)
        resume_journal.append({"record_kind": "resume_reconciliation", "at": paths.utc_now(),
                               "published_checkpoint": int(pending["update"])})
        with ledger.segment("io"):
            last_state_record = store.save(policy=policy, optimizer=optimizer,
                                           scheduler=scheduler, state=state,
                                           stream_digests=digests)

    def persist(reason, *, completed=None):
        paths.write_json(directory / PROGRESS_JSON, {
            "schema_version": TRAJECTORY_SCHEMA, "record_kind": "trajectory_progress",
            "trajectory": row["trajectory"], "reason": reason, "status": status,
            "stop_reason": stop_reason,
            "update": update if completed is None else int(completed),
            "attempted_update": update, "checks": state.checks, "sentinels": state.sentinels,
            "exposures": dict(state.exposures), "clipped_updates": state.clipped,
            "endpoints_reached": sorted(state.endpoints_reached),
            "declared_endpoints": endpoints_declared, "resumes": int(state.resumes),
            "resume_state": last_state_record, "cost": ledger.document(),
            "heartbeat_at": paths.utc_now(),
            "note": ("written at checks, so it can lag the journals by up to one interval. The "
                     "append-only journals are the authority for what completed.")})

    persist("started" if resumed is None else "resumed")
    try:
        while update < total_updates and state.status != STATUS_STOPPED:
            update += 1
            chosen_rows, rejected_rows = stream.batch(update)
            require(int(chosen_rows.size) == int(batch_rows),
                    f"Update {update} drew {chosen_rows.size} chosen rows, not {batch_rows}")
            preservation_rows = None
            if preservation_family != "none":
                preservation_rows = replay_lib_batch(replay_order, update, batch_rows)
            totals = {"task": int(batch_rows)}
            coefficients = {"task": 1.0}
            if preservation_family != "none":
                totals[preservation_family] = int(batch_rows)
                coefficients[preservation_family] = float(preservation_lambda)
            accumulator = replay_lib.MicrobatchAccumulator(
                totals, coefficients=coefficients,
                label=f"{row['trajectory']} update {update}")
            task_blocks, preservation_blocks = [], []
            term_diagnostics = None
            if (gradient_diagnostic is not None and preservation_family != "none"
                    and update % int(gradient_diagnostic_interval) == 0):
                with ledger.segment("optimizer"):
                    term_diagnostics = gradient_diagnostic(
                        update=update, chosen_rows=chosen_rows[:int(microbatch_rows)],
                        rejected_rows=rejected_rows[:int(microbatch_rows)],
                        preservation_rows=(preservation_rows[:int(microbatch_rows)]
                                           if preservation_rows is not None else None))
            with ledger.segment("optimizer"):
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, int(batch_rows), int(microbatch_rows)):
                    stop = min(start + int(microbatch_rows), int(batch_rows))
                    counts, means = {}, {}
                    task_mean, task_block = task_batch(chosen_rows[start:stop],
                                                       rejected_rows[start:stop])
                    counts["task"], means["task"] = stop - start, task_mean
                    task_blocks.append((stop - start, task_block))
                    if preservation_family != "none":
                        keep_mean, keep_block = preservation_batch(preservation_rows[start:stop])
                        counts[preservation_family] = stop - start
                        means[preservation_family] = keep_mean
                        preservation_blocks.append((stop - start, keep_block))
                    accumulator.add(counts, means, backward=lambda tensor: tensor.backward())
                components = accumulator.finish()
                step = replay_lib.clip_and_step(policy.model, optimizer, scheduler,
                                                gradient_clip=gradient_clip)
            state.clipped += int(bool(step["clipped"]))
            state.exposures = exposures_at(update, batch_rows=batch_rows,
                                           uses_rejected=uses_rejected, replay_rows=replay_rows,
                                           tail_rows=tail_rows)
            # Cursor positions, so a resumed run can be checked against the
            # streams it is about to keep consuming rather than against its own
            # update counter. A replay arm has advanced the replay order by the
            # same number of rows; a no-preservation control has not moved it at
            # all, which is what makes its update identical to one produced by
            # code with no replay term.
            state.stream_position = int(update) * int(batch_rows)
            state.replay_position = (int(update) * int(batch_rows)
                                     if preservation_family != "none" else 0)
            updates_journal.append({
                "record_kind": "update", "update": int(update),
                "cycle": int(stream.cycle_of_position[(update - 1) * int(batch_rows)]),
                "spans_cycle_boundary": bool(
                    stream.cycle_of_position[(update - 1) * int(batch_rows)]
                    != stream.cycle_of_position[update * int(batch_rows) - 1]),
                "exposures": dict(state.exposures),
                "task_loss": components["component_means"]["task"],
                "preservation_loss": components["component_means"].get(preservation_family),
                "preservation_family": preservation_family,
                "preservation_lambda": float(preservation_lambda),
                "weighted_total": components["weighted_total"],
                "microbatches": components["microbatches"],
                "raw_task_loss": _weighted(task_blocks, "raw_task_loss"),
                "scaled_task_loss": _weighted(task_blocks, "scaled_task_loss"),
                "margin_mean": _weighted(task_blocks, "margin"),
                "sign_accuracy": _weighted(task_blocks, "sign_correct"),
                "chosen_nll_per_residue": _weighted(task_blocks, "chosen_nll_per_residue"),
                "preservation_mean": _weighted(preservation_blocks, "mean_conditional_kl")
                if preservation_family == "fkl" else _weighted(preservation_blocks,
                                                               "mean_penalty"),
                "tail_active_fraction": _weighted(preservation_blocks, "active_fraction"),
                # The MAXIMUM across the whole update, not the mean of the
                # microbatch maxima: averaging four maxima reports a number no
                # replay row produced and understates the worst drop, which is
                # the only thing this field is for.
                "tail_max_drop": _maximum(preservation_blocks, "max_drop_nats"),
                "stream_position": int(state.stream_position),
                "replay_position": int(state.replay_position),
                "gradient_terms": term_diagnostics,
                **step})
            if heartbeat is not None:
                heartbeat.beat(trajectory=row["trajectory"], update=update,
                               exposures=dict(state.exposures), checks=state.checks)

            kind, reason = plan.due(update, final=(update == total_updates))
            if kind == "sentinel":
                rng_before = capture_rng_state()
                with ledger.segment("gate"):
                    sentinel = sentinel_check(update=update, reason=reason)
                restore_rng_state(rng_before)
                state.sentinels += 1
                state.previous_sentinel_D = sentinel.get("D")
                monitor_journal.append(dict(sentinel, update=int(update),
                                            exposures=dict(state.exposures)))
                if not sentinel.get("request_full_check"):
                    continue
                kind, reason = "full", "sentinel_request"
            if kind != "full":
                continue

            rng_before = capture_rng_state()
            with ledger.segment("gate"):
                verdict = full_check(update=update, reason=reason,
                                     exposures=dict(state.exposures))
            with ledger.segment("preservation"):
                preservation_record = verdict.pop("preservation", None)
            restore_rng_state(rng_before)
            state.checks += 1
            if authoritative_stop is not None and update == int(authoritative_stop["update"]):
                verdict["replayed_gate_verdict"] = dict(verdict)
                verdict["passed"] = False
                verdict["stop_reason"] = authoritative_stop.get("stop_reason") or "recorded_gate_stop"
            monitor_journal.append(dict(_journal_safe(verdict), record_kind="gate_verdict",
                                        preservation=_journal_safe(preservation_record),
                                        exposures=dict(state.exposures), update=int(update),
                                        evaluation_rng="restored; a check never advances the "
                                                       "training generators"))
            if not verdict.get("passed"):
                stop_reason = verdict.get("stop_reason") or "parent_relative_likelihood_breach"
                with ledger.segment("io"):
                    failure = _save_failed(directory, policy, update=update,
                                           exposures=dict(state.exposures),
                                           stop_reason=stop_reason, identity=identity)
                monitor_journal.append({"record_kind": "snapshot", "update": int(update),
                                        "passed": False, "failed_state": failure,
                                        "consequence": ("the trajectory stops here. A stopped arm "
                                                        "is an OUTCOME, not a failure and not an "
                                                        "omitted row; the queue continues.")})
                status = STATUS_STOPPED
                state.status, state.stop_reason, state.update = status, stop_reason, update
                with ledger.segment("io"):
                    last_state_record = store.save(policy=policy, optimizer=optimizer,
                                                   scheduler=scheduler, state=state,
                                                   stream_digests=digests)
                persist("gate_stop")
                break
            state.update = update
            state.status = STATUS_RUNNING
            # The endpoint is declared PENDING before the save that carries its
            # weights, so an interruption anywhere between here and the journal
            # commit leaves a state that knows what it still owes. Publishing
            # first and saving afterwards would have the mirror-image hole.
            due_endpoint = (int(update) in endpoints_declared
                            and int(update) not in set(state.endpoints_reached))
            state.pending_endpoint = ({"update": int(update),
                                       "exposures": dict(state.exposures),
                                       "gate": _journal_safe(verdict),
                                       "preservation": _journal_safe(preservation_record)}
                                      if due_endpoint else None)
            due_checkpoint = (on_checkpoint is not None and int(update) in checkpoints_declared
                              and int(update) not in endpoints_declared
                              and int(update) not in set(state.checkpoints_saved))
            state.pending_checkpoint = ({"update": int(update),
                                         "exposures": dict(state.exposures),
                                         "gate": _journal_safe(verdict)}
                                        if due_checkpoint else None)
            with ledger.segment("io"):
                last_state_record = store.save(policy=policy, optimizer=optimizer,
                                               scheduler=scheduler, state=state,
                                               stream_digests=digests)
            monitor_journal.append({"record_kind": "snapshot", "update": int(update),
                                    "passed": True, "resume_state": last_state_record})
            if due_endpoint:
                publish_endpoint(dict(state.pending_endpoint,
                                      gate=verdict, preservation=preservation_record))
                with ledger.segment("io"):
                    last_state_record = store.save(policy=policy, optimizer=optimizer,
                                                   scheduler=scheduler, state=state,
                                                   stream_digests=digests)
                persist(f"endpoint_{update}")
            # A declared checkpoint that is ALSO an endpoint is already on disk
            # under its endpoint name; writing a second 88 MB copy of the same
            # tensors would double the storage for nothing.
            if due_checkpoint:
                publish_checkpoint(state.pending_checkpoint)
                with ledger.segment("io"):
                    last_state_record = store.save(policy=policy, optimizer=optimizer,
                                                   scheduler=scheduler, state=state,
                                                   stream_digests=digests)
            persist(f"check_{update}")
        else:
            status = STATUS_STOPPED if state.status == STATUS_STOPPED else STATUS_COMPLETED
    except BudgetExhausted as error:
        status, stop_reason = STATUS_INCOMPLETE, str(error)
        observed = durable_progress(directory)
        update = int(observed["updates"])
        state.exposures = dict(observed["exposures"] or state.exposures)
        # The last authoritative full-check state stays resumable; budget
        # exhaustion cannot turn a partially checked endpoint into a success.
        persist("budget_exhausted", completed=durable_progress(directory)["updates"])
    except BaseException as error:                              # noqa: BLE001 - recorded, re-raised
        status, fatal, raised = STATUS_FAILED, f"{type(error).__name__}: {error}", error
        stop_reason = stop_reason or fatal
        persist("failed", completed=durable_progress(directory)["updates"])
    finally:
        updates_journal.close()
        monitor_journal.close()
        endpoints_journal.close()
        resume_journal.close()

    durable = durable_progress(directory)
    completed_updates = durable["updates"] if fatal else update
    counted_exposures = dict(durable["exposures"]) if fatal else dict(state.exposures)
    reached = dict(durable["endpoints_reached"])
    unreached = {str(value): {"reason": ("the trajectory stopped before this exposure endpoint; "
                                         "no checkpoint exists and none is substituted")}
                 for value in endpoints_declared if str(value) not in reached}
    if status == STATUS_COMPLETED and unreached:
        status = STATUS_STOPPED if stop_reason else STATUS_INCOMPLETE
    document = {
        "schema_version": TRAJECTORY_SCHEMA, "record_kind": "trajectory_status",
        "trajectory": row["trajectory"], "arm_id": row.get("arm_id"),
        "block": row.get("block"), "regime": row.get("regime"),
        "task": row.get("task"), "preservation": preservation_family,
        "preservation_lambda": float(preservation_lambda), "seed": int(row["seed"]),
        "status": status, "stop_reason": stop_reason,
        "updates": completed_updates, "declared_updates": requested_updates,
        "attempted_updates": update, "checks": state.checks, "sentinels": state.sentinels,
        "clipped_updates": state.clipped, "resumes": int(state.resumes),
        "exposures": counted_exposures, "attempted_exposures": dict(state.exposures),
        "endpoints_reached": {key: dict(value) for key, value in sorted(reached.items())},
        "endpoints_not_reached": unreached,
        "checkpoints_saved": sorted(int(value) for value in state.checkpoints_saved),
        "declared_checkpoints": checkpoints_declared,
        "identity": dict(identity), "cost": ledger.document(),
        "durable": {key: value for key, value in durable.items() if key != "endpoints_reached"},
        "stream_position": int(state.stream_position),
        "replay_position": int(state.replay_position),
        "previous_sentinel_D": state.previous_sentinel_D,
        "journal_repairs": [entry for entry in journal_repairs if entry["repaired"]],
        "resume_reconciliation": (None if reconciled is None else
                                  {"published_endpoint": int(reconciled["update"]),
                                   "basis": "an endpoint whose weights were saved but whose "
                                            "publication was interrupted"}),
        "wall_seconds": time.perf_counter() - started,
        "resume_policy": ("exact. Model, optimizer, scheduler, all four RNG streams, the stream "
                          "and replay cursors, the sentinel's previous D, the cadence counters "
                          "and any pending endpoint publication are restored, and the whole "
                          "non-tensor payload carries its own digest. There is no scaler and its "
                          "absence is recorded with the reason. The state is written at every "
                          "FULL check, so a kill replays at most one full-check interval of "
                          "deterministic updates."),
        "completed_at": paths.utc_now()}
    if fatal:
        document["failure"] = {
            "error": fatal,
            "consequence": ("an unexpected exception. The trajectory is `failed`, which is "
                            "machinery and NOT a declared scientific outcome; the exception is "
                            "re-raised so the campaign fail-stops rather than continuing under "
                            "whatever is wrong."),
            "counters_from": ("the de-duplicated append-only journals, never the in-flight loop "
                              "counters. `attempted_*` fields carry those and are never presented "
                              "as completed work.")}
        write_terminal_status(directory, document, register=register_status)
        persist("terminal", completed=completed_updates)
        raise raised
    previous_terminal = read_terminal_status(directory)
    if (status == STATUS_STOPPED and previous_terminal is not None
            and previous_terminal.get("status") == STATUS_STOPPED
            and int(previous_terminal["updates"]) == int(completed_updates)):
        return previous_terminal
    write_terminal_status(directory, document, register=register_status)
    persist("terminal")
    return document


def replay_lib_batch(order, update, batch_rows):
    from . import her2_replay_streams as streams_lib
    return streams_lib.replay_batch_rows(order, update, batch_rows=int(batch_rows))


def _weighted(blocks, key):
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


def _maximum(blocks, key):
    """The maximum of a per-microbatch maximum, over the whole update.

    A row-count weighted mean of four microbatch maxima is not a maximum of
    anything: it is below the worst drop the update actually saw, which is the
    direction that hides a tail event.
    """
    import torch
    best = None
    for _, block in blocks:
        value = (block or {}).get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            value = float(value.detach().double().max())
        value = float(value)
        best = value if best is None else max(best, value)
    return best


#: Journal keys whose values are raw per-row vectors. They are persisted as
#: hashed arrays beside the journal and would make a JSONL line unreadable.
_VECTOR_KEYS = frozenset({"scores", "current_chosen_scores", "conditional_kl_per_row",
                          "conditional_kl_per_position", "log_marginals", "marginals"})


def _journal_safe(record):
    """A JSON-writable projection: no arrays, no numpy scalars, no NaN surprises.

    ``JsonlJournal`` writes with ``allow_nan=False`` and plain ``json.dumps``, so
    a ``numpy.float64`` that reached a record would raise mid-trajectory. The
    coercion happens here, once, rather than at each of the sites that build a
    verdict.
    """
    if record is None:
        return None
    if isinstance(record, dict):
        return {key: _journal_safe(value) for key, value in record.items()
                if key not in _VECTOR_KEYS and not isinstance(value, np.ndarray)}
    if isinstance(record, (list, tuple)):
        return [_journal_safe(value) for value in record
                if not isinstance(value, np.ndarray)]
    if isinstance(record, np.generic):
        return record.item()
    return record


def _save_failed(directory, policy, *, update, exposures, stop_reason, identity):
    import torch
    from .her2_nf_storage import state_digest
    target = Path(directory) / FAILED_STATE
    tensors = policy.model.state_dict()
    digest = state_digest(policy.model)
    payload = {"schema_version": TRAJECTORY_SCHEMA, "kind": "failed_state",
               "update": int(update), "exposures": dict(exposures), "stop_reason": stop_reason,
               "identity": dict(identity), "state": tensors, "state_sha256": digest}
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    restored = load_cpu(temporary)
    require(state_dict_digest(restored["state"]) == digest,
            "the diagnostic stop snapshot did not read back as the weights that produced it")
    del restored
    os.replace(temporary, target)
    return {"file": FAILED_STATE, "sha256": paths.sha256_file(target), "state_sha256": digest,
            "update": int(update), "stop_reason": stop_reason,
            "note": ("diagnostic weights at the recorded stop reason. Never a last-passing state "
                     "and never a reached endpoint.")}


def write_terminal_status(directory, document, *, register=None):
    """Write a trajectory's terminal status once. A completed one is never rewritten."""
    directory = Path(directory)
    target = directory / STATUS_JSON
    if target.is_file():
        existing = paths.read_json(target)
        if paths.scientific_projection(existing) == paths.scientific_projection(document):
            return {"path": str(target), "rewritten": False, "identical": True}
        require(existing.get("status") not in TERMINAL_STATUSES
                or existing.get("status") == STATUS_FAILED,
                f"{target} already records terminal status {existing.get('status')!r}. A terminal "
                "status is the trajectory's own statement of what it reached and is not rewritten.")
    digest = paths.write_json(target, document)
    if register is not None:
        register(path=target, document=document)
    return {"path": str(target), "sha256": digest, "rewritten": True}


def read_terminal_status(directory):
    target = Path(directory) / STATUS_JSON
    return paths.read_json(target) if target.is_file() else None


def archive_terminal_status(directory):
    """Move a finished status aside under its own update count, for a continuation.

    A 500-update calibration pilot legitimately continues to 1,000 in the same
    directory, on the same stream, from the same state. Its terminal status is a
    true statement about what it reached and is neither deleted nor overwritten:
    it is renamed ``status_u500.json`` and stays there beside the continuation's
    own record. If that name is taken the archive is refused rather than
    clobbered.
    """
    directory = Path(directory)
    target = directory / STATUS_JSON
    if not target.is_file():
        return None
    document = paths.read_json(target)
    archived = directory / f"status_u{int(document.get('updates') or 0)}.json"
    require(not archived.is_file(),
            f"{archived} already exists; a finished status is never clobbered, so this "
            "continuation is refused rather than overwriting a record of what was reached.")
    os.replace(target, archived)
    return {"archived_to": archived.name, "status": document.get("status"),
            "updates": document.get("updates"),
            "reason": ("a continuation of this setting is about to run in the same directory; "
                       "the finished record is retained under its own update count.")}
