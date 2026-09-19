"""One guarded trajectory: incremental journals, a rolling last-passing state, a real stop.

This is a **new** loop rather than an edit to
:func:`~.her2_preferences.run_budgeted_trajectory`. Two reasons, in order of
weight: the original file's hash is part of the initial-SFT freeze's identity, so
editing it would make the parents unreadable under their own provenance; and the
gate interleaves with the loop in a way that would turn the original into a
different function with three optional arguments.

What this loop guarantees, by construction rather than by docstring:

* **Nothing is charged to the training clock except updates.** Journalling,
  quantile reductions, exposure counting, checkpoint I/O and every gate check run
  outside ``clock.segment()``. Their cost is measured on its own timers and
  reported beside the budget, never netted against it.
* **A budget checkpoint exists only if its gate check passed.** The order at a
  crossing is check, then checkpoint. A failed check therefore cannot leave a
  "600 s checkpoint" behind: the budget is recorded ``reached: false`` with the
  update and the consumed seconds, and no file is written.
* **The verdict is durable before any snapshot is written.** At each check the
  gate record goes to ``monitor.jsonl`` first, as its own ``gate_verdict`` line;
  the checkpoint that follows is reported on a separate ``snapshot`` line. A save
  that fails therefore loses bytes, not the measured D, the score-vector hashes
  or the stop itself.
* **The last passing weights are actually saved.** At *every* passing check the
  resident weights go to a rolling ``last_passing.pt`` through a temporary file
  and ``os.replace``; the snapshot line records the update, the cumulative
  training seconds and the sha of the bytes written. Update 0 records the
  verified parent by path and freeze sha rather than by copying it. A rolling
  save that fails **stops the trajectory** with status ``failed``: the verdict and
  the previous passing bytes are kept, no further update runs and no nominal
  endpoint is written on top of a stale rolling artifact. That is an artifact/IO
  failure, not a gate stop, and it does not advance a stage.
* **A budget is recorded reached only after its endpoint was durably written.**
  The budget leaves the outstanding list after ``on_budget`` returns, so a failed
  checkpoint write leaves it under ``budgets_not_reached`` instead of deleting it
  from both lists at once.
* **Failing weights are never written as last-passing.** On a breach the resident
  state goes to ``failed_state.pt``, beside the optimizer and scheduler states,
  as diagnostics. That save is best effort: if it fails the gate stop and the
  journalled verdict still stand. There is no resume feature.
* **A stop or an exception still leaves complete evidence.** Journals are
  appended, flushed and fsynced per line, ``budgets.jsonl`` and
  ``trajectory_progress.json`` record each crossing as it happens rather than at
  the end, and ``stop.json`` / the returned document are written in a ``finally``.

The rolling checkpoint is a *latest-state* artifact. A sha recorded for an
earlier check describes bytes that were at that path at that moment; it does not
resolve there afterwards, and the artifacts say so in those words.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .her2_data import CORE_LENGTH
from .her2_guard import json_number, quantile_summary
from .her2_runtime import canonical_json, require, save_json, sha256

TRAJECTORY_SCHEMA = "her2-guarded-trajectory/1"

#: Per-pair vectors that get the full seven-quantile treatment: these are the
#: distributions a mean would misrepresent, which is the whole reason the
#: original histories are not enough.
DISTRIBUTION_KEYS = ("margin", "coefficient", "hinge", "chosen_change", "rejected_change",
                     "modified_margin", "modified_coefficient", "residual")
#: Loss components. Their means are what reconcile against the reported loss.
COMPONENT_KEYS = ("loss", "preference_component", "hinge_component", "nll_component",
                  "nll_per_residue", "chosen_log_probability", "rejected_log_probability",
                  "chosen_nll_per_residue", "rejected_nll_per_residue", "target_margin")

#: Per-pair vectors that are 0/1 indicators. A fraction carries everything their
#: quantiles would, under the name of the thing the fraction actually is.
INDICATOR_KEYS = {"hinge_active": "hinge_active_fraction", "sign_correct": "sign_accuracy"}

#: Scalar fields the loop writes per update. A per-pair summary may not land on
#: any of them: ``loss`` in a diagnostics dict is the per-pair vector, and letting
#: its summary dict overwrite the scalar batch loss is how a number that every
#: later table reads becomes a different type.
RESERVED_UPDATE_FIELDS = frozenset({
    "update", "loss", "gradient_norm", "learning_rate_used", "learning_rate_next",
    "update_gpu_seconds", "cumulative_gpu_seconds", "exposures", "core_token_exposures",
    "distinct_exposures", "within_pair_hamming"})

ROLLING_NOTE = ("last_passing.pt is a rolling latest-state artifact. The sha recorded at an "
                "earlier check described the bytes at that path at that time; it does not resolve "
                "there now, and no claim is made that it does.")


class JsonlJournal:
    """Append-only JSONL that survives a kill: one line, flushed and fsynced.

    A buffered writer loses the last few thousand updates of a trajectory that
    was stopped by the gate -- exactly the records the stop is evidence about --
    so every line pays for an fsync. The cost is measured by the caller and
    reported under the diagnostic clock.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines = 0
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")

    def append(self, record):
        self._stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.lines += 1
        return record

    def close(self):
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()

    def document(self):
        return {"path": str(self.path), "lines": self.lines,
                "durability": "one JSON object per line, flushed and fsynced as it is written"}


class StateSaver:
    """Atomic rolling saves for the passing state, and a separate slot for a breach.

    ``last_passing`` and ``failed_state`` are different file names on purpose.
    Writing the failing weights into the file whose name says "passing" would be
    the one lie this whole design exists to avoid, so there is no code path that
    can do it: :meth:`save_failed` cannot be pointed at the passing path.
    """

    def __init__(self, directory, model, *, optimizer=None, scheduler=None, metadata=None):
        self.directory = Path(directory)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metadata = dict(metadata or {})
        self.wall_seconds = 0.0
        self.saves = 0

    def _write(self, name, payload):
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        digest = sha256(path)
        self.saves += 1
        elapsed = time.perf_counter() - started
        self.wall_seconds += elapsed
        return {"path": str(path), "sha256": digest, "wall_seconds": json_number(elapsed)}

    def save_last_passing(self, *, update, gpu_seconds, check):
        record = self._write("last_passing.pt", {
            "schema_version": TRAJECTORY_SCHEMA, "kind": "last_passing",
            "state": self.model.state_dict(), "update": int(update),
            "training_gpu_seconds": float(gpu_seconds), "check": int(check),
            "metadata": self.metadata})
        record.update(update=int(update), training_gpu_seconds=json_number(gpu_seconds),
                      check=int(check), note=ROLLING_NOTE)
        return record

    def save_failed(self, *, update, gpu_seconds, stop_reason):
        """The breach state and its optimizer/scheduler, as diagnostics only."""
        record = self._write("failed_state.pt", {
            "schema_version": TRAJECTORY_SCHEMA, "kind": "failed_state",
            "state": self.model.state_dict(), "update": int(update),
            "training_gpu_seconds": float(gpu_seconds), "stop_reason": stop_reason,
            "metadata": self.metadata})
        record.update(update=int(update), training_gpu_seconds=json_number(gpu_seconds),
                      stop_reason=stop_reason,
                      note=("these are diagnostic weights at the recorded stop reason. They are kept for "
                            "diagnosis and are never the last-passing state."))
        if self.optimizer is not None:
            record["optimizer_state"] = self._write(
                "optimizer_state.pt", {"schema_version": TRAJECTORY_SCHEMA,
                                       "kind": "optimizer_state",
                                       "state": self.optimizer.state_dict(),
                                       "update": int(update)})
        if self.scheduler is not None:
            record["scheduler_state"] = self._write(
                "scheduler_state.pt", {"schema_version": TRAJECTORY_SCHEMA,
                                       "kind": "scheduler_state",
                                       "state": self.scheduler.state_dict(),
                                       "update": int(update)})
        return record


def summarize_per_pair(vectors):
    """Quantiles for the distributions, means for the components. Detached input only.

    Runs outside every charged segment. Indicator vectors become fractions rather
    than distributions, because a 0/1 vector's quantiles carry no information the
    fraction does not: ``hinge_active`` becomes ``hinge_active_fraction`` and
    ``sign_correct`` becomes ``sign_accuracy``.

    The per-pair ``loss`` vector is summarized under ``per_pair_loss``. It is a
    distribution over pairs and the update's ``loss`` is the scalar the optimizer
    actually stepped on; giving them the same name once let the summary dict
    replace the scalar, which turned a number every later table reads into a
    dict.
    """
    document = {}
    for name, values in sorted((vectors or {}).items()):
        array = _to_numpy(values)
        if name in INDICATOR_KEYS:
            document[INDICATOR_KEYS[name]] = json_number(array.mean()) if array.size else None
        elif name in DISTRIBUTION_KEYS:
            document[name] = quantile_summary(array)
        else:
            key = "per_pair_loss" if name == "loss" else name
            document[key] = {"mean": json_number(np.nanmean(array)) if array.size else None}
    return document


def within_pair_hamming(chosen_index, rejected_index):
    """Hamming distance between a chosen core and ITS OWN rejected partner.

    The pairing matches on distance to the wild-type core, not on distance to each
    other, so this is new information rather than a restatement of the design. It
    is descriptive: nothing here establishes that within-pair distance caused the
    original decline.
    """
    chosen = np.asarray(chosen_index)
    rejected = np.asarray(rejected_index)
    require(chosen.shape == rejected.shape, "Within-pair distance needs aligned pair cores")
    distance = (chosen != rejected).sum(axis=1).astype(np.float64)
    return {"mean": json_number(distance.mean()), "min": json_number(distance.min()),
            "max": json_number(distance.max()), "pairs": int(distance.size),
            "note": "chosen vs its own rejected partner; the pairing matches on WT distance"}


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().double().cpu().numpy().ravel()
    return np.asarray(values, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def run_guarded_trajectory(*, step, budgets, clock, optimizer, scheduler, model, gradient_clip,
                           monitor_check, directory, parent_state, saver=None, on_budget=None,
                           exposure_fields=(), cumulative_exposures=None, max_updates=None,
                           progress=None, schedule=None, identity=None):
    """Train one continuous trajectory under a gate that can stop it between updates.

    ``step(update)`` returns ``(loss, statistics)``. ``statistics`` may carry
    ``per_pair`` (a dict of **detached** per-pair vectors) and ``pair_cores``
    (the chosen/rejected index blocks); both are removed before journalling and
    reduced outside the charged segment.

    ``monitor_check(update, gpu_seconds, reason)`` performs the gate check and
    returns its record; it must contain ``passed``. ``schedule`` is consulted for
    the ordinary cadence and **advanced here**, by this loop, after every check:
    the cadence is the controller's business and must not depend on the callback
    happening to have a schedule of its own to tick.

    Three counters, not one. ``updates`` is completed optimizer updates;
    ``attempted_updates`` includes an attempt that raised part-way; and the GPU
    seconds a failed attempt really spent are charged and reported rather than
    forgiven. An update is counted only after ``optimizer.step()`` and
    ``scheduler.step()`` have both returned, so "2 updates" in a document always
    means two updates whose journal lines exist.
    """
    remaining = sorted(float(b) for b in budgets)
    require(remaining, "A trajectory needs at least one budget")
    precharged = float(clock.elapsed_seconds)
    require(precharged < remaining[0],
            f"{precharged:.1f} GPU seconds were already charged to this trajectory (reference "
            f"cache creation) before the first update, which is at or beyond the smallest budget "
            f"of {remaining[0]:.1f} s. The budgeted trajectory cannot be reported honestly under "
            "these budgets; raise them or reduce the reference population rather than recording a "
            "checkpoint whose 'one update of overshoot' is a fiction.")
    directory = Path(directory)
    saver = saver or StateSaver(directory, model, optimizer=optimizer, scheduler=scheduler,
                                metadata={"identity": dict(identity or {})})
    schedule = schedule if schedule is not None else getattr(monitor_check, "schedule", None)
    require(schedule is not None, "The monitor cadence must be supplied")

    updates_journal = JsonlJournal(directory / "updates.jsonl")
    monitor_journal = JsonlJournal(directory / "monitor.jsonl")
    budget_journal = JsonlJournal(directory / "budgets.jsonl")
    started = time.perf_counter()
    totals = {name: 0 for name in exposure_fields}
    reached, unreached, history_tail = {}, {}, []
    diagnostic_wall, evaluation_wall = 0.0, 0.0
    update, checks, stopped, stop_reason, failure = 0, 0, False, None, None
    attempted, failed_attempts, failed_work_seconds, capped = 0, 0, 0.0, False
    snapshot_errors, artifact_failure = [], None
    # Update 0: the verified parent IS the last passing state, recorded by path and
    # freeze sha. Copying it would produce a second set of bytes claiming to be the
    # parent, which is a worse artifact than a reference to the one that exists.
    last_passing = {"update": 0, "training_gpu_seconds": json_number(precharged),
                    "kind": "verified_parent",
                    "checkpoint": parent_state.get("checkpoint"),
                    "sha256": parent_state.get("sha256"),
                    "note": "the parent named by the initial-SFT freeze, not a copy of it"}
    stop_record = None
    attempt_state = {"optimizer_stepped": False, "scheduler_stepped": False, "attempt": 0}

    def persist_progress(reason):
        """Write the reached-budget mapping and the counters now, not in ``finally``.

        A budget that was crossed at update 1,200 is a fact about this trajectory
        the moment it happens. Keeping it in memory until the end means a kill at
        update 1,201 leaves a directory with a 180 s checkpoint in it and nothing
        on disk saying which budget that checkpoint belongs to.
        """
        save_json(directory / "trajectory_progress.json", {
            "schema_version": TRAJECTORY_SCHEMA, "record_kind": "progress", "reason": reason,
            "identity": dict(identity or {}),
            "updates": update, "attempted_updates": attempted,
            "failed_attempts": failed_attempts, "checks": checks,
            "budgets": dict(reached),
            "budgets_outstanding": [float(budget) for budget in remaining],
            "training_gpu_seconds": json_number(clock.elapsed_seconds),
            "precharged_gpu_seconds": json_number(precharged),
            "failed_work_gpu_seconds": json_number(failed_work_seconds),
            "exposures": dict(totals),
            "last_passing": dict(last_passing),
            "stopped": bool(stopped), "stop_reason": stop_reason,
            "note": ("rewritten at every gate check and every budget crossing; budgets.jsonl is "
                     "the append-only record of the same crossings")})

    try:
        while remaining and not stopped:
            if max_updates is not None and update >= max_updates:
                capped = True
                break
            attempted += 1
            attempt = update + 1
            attempt_state.update(attempt=attempt, optimizer_stepped=False,
                                 scheduler_stepped=False)
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            charged_before = float(clock.elapsed_seconds)
            try:
                with clock.segment() as segment:
                    optimizer.zero_grad(set_to_none=True)
                    loss, statistics = step(attempt)
                    require(bool(torch.isfinite(loss)),
                            f"Nonfinite loss at update attempt {attempt}")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                                          error_if_nonfinite=True)
                    optimizer.step()
                    attempt_state["optimizer_stepped"] = True
                    scheduler.step()
                    attempt_state["scheduler_stepped"] = True
            except BaseException:
                # The device time this attempt spent was really spent. It is charged
                # and reported as failed work; what it is NOT is a completed update.
                failed_attempts += 1
                failed_work_seconds += float(clock.elapsed_seconds) - charged_before
                raise
            update += 1
            elapsed = clock.elapsed_seconds

            diagnostics_started = time.perf_counter()
            statistics = dict(statistics)
            per_pair = statistics.pop("per_pair", None)
            pair_cores = statistics.pop("pair_cores", None)
            for name in exposure_fields:
                totals[name] += int(statistics.get(name, 0))
            entry = {"update": update, "loss": json_number(loss.detach()),
                     "gradient_norm": json_number(norm),
                     "learning_rate_used": json_number(learning_rate_used),
                     "learning_rate_next": json_number(scheduler.get_last_lr()[0]),
                     "update_gpu_seconds": json_number(segment.seconds),
                     "cumulative_gpu_seconds": json_number(elapsed),
                     "exposures": dict(totals),
                     "core_token_exposures": int(totals.get("sequences", 0)) * CORE_LENGTH,
                     **statistics}
            if per_pair:
                summary = summarize_per_pair(per_pair)
                collisions = sorted(set(summary) & RESERVED_UPDATE_FIELDS)
                require(not collisions,
                        f"The per-pair summary would overwrite the update's own scalar fields "
                        f"{collisions}. Those are different quantities and must keep different "
                        "names; see summarize_per_pair.")
                entry.update(summary)
            if pair_cores is not None:
                entry["within_pair_hamming"] = within_pair_hamming(*pair_cores)
            if cumulative_exposures is not None:
                entry["distinct_exposures"] = dict(cumulative_exposures())
            updates_journal.append(entry)
            history_tail.append(entry)
            if len(history_tail) > 200:
                history_tail.pop(0)
            if progress is not None:
                progress.update(update, loss=entry["loss"], gpu_seconds=elapsed)
            diagnostic_wall += time.perf_counter() - diagnostics_started

            crossing = [budget for budget in remaining if elapsed >= budget]
            capped = bool(max_updates is not None and update >= max_updates)
            final = bool(len(crossing) == len(remaining) or capped)
            due, reason = schedule.due(update, elapsed)
            if crossing:
                due, reason = True, "budget_crossing"
            elif final:
                due, reason = True, "final_state"

            if due:
                record = monitor_check(update=update, gpu_seconds=elapsed, reason=reason)
                checks += 1
                require("passed" in record, "A monitor check must report a pass/fail verdict")
                # The controller owns the cadence. Advancing it here means the next
                # check is due 25 updates later whether or not the callback ticked a
                # schedule of its own.
                schedule.record(update, elapsed)
                # The verdict is journalled BEFORE any snapshot I/O. A failed save is
                # then a separate, later fact about bytes; it cannot take the measured
                # D and the score-vector hashes down with it.
                monitor_journal.append(dict(record, record_kind="gate_verdict",
                                            run=dict(identity or {})))
                snapshot = {"record_kind": "snapshot", "check": checks - 1, "update": update,
                            "passed": bool(record["passed"]),
                            "training_gpu_seconds": json_number(elapsed),
                            "run": dict(identity or {}), "D": record.get("D"),
                            "scores_sha256": (record.get("scores") or {}).get("sha256"),
                            "verdict_journal_line": monitor_journal.lines - 1}
                if record["passed"]:
                    try:
                        saved = saver.save_last_passing(update=update, gpu_seconds=elapsed,
                                                        check=checks - 1)
                    except Exception as error:                # noqa: BLE001 - recorded, not raised
                        snapshot["save_failed"] = f"{type(error).__name__}: {error}"
                        snapshot["consequence"] = (
                            "no rolling last-passing bytes were written at this check. The verdict "
                            "above stands and the previous last-passing state is untouched, and "
                            "the trajectory stops here: another update would move the weights past "
                            "a passing state that was never saved, and a nominal endpoint written "
                            "after it would rest on a rolling artifact that is already stale.")
                        snapshot_errors.append(dict(snapshot))
                        stopped = True
                        stop_reason = "last_passing_snapshot_failure"
                        artifact_failure = {
                            "type": type(error).__name__, "message": str(error),
                            "where": "save_last_passing", "update": update, "check": checks - 1,
                            "training_gpu_seconds": json_number(elapsed),
                            "gate_verdict": {
                                "passed": True, "D": record.get("D"),
                                "scores_sha256": (record.get("scores") or {}).get("sha256")},
                            "note": ("an artifact/IO failure, not a gate stop and not a "
                                     "completion. The measured verdict above is kept, the "
                                     "previous passing bytes are kept, no further update was "
                                     "performed, no nominal endpoint was written, and this "
                                     "trajectory does not advance a stage.")}
                    else:
                        record["last_passing"] = saved
                        snapshot["last_passing"] = saved
                        last_passing = {"update": update,
                                        "training_gpu_seconds": json_number(elapsed),
                                        "kind": "rolling_file", "checkpoint": saved["path"],
                                        "sha256": saved["sha256"], "check": checks - 1,
                                        "note": ROLLING_NOTE}
                else:
                    stop_reason = record.get("stop_reason", "gate_stop")
                    stopped = True
                    try:
                        failed_state = saver.save_failed(update=update, gpu_seconds=elapsed,
                                                         stop_reason=stop_reason)
                    except Exception as error:                # noqa: BLE001 - recorded, not raised
                        snapshot["save_failed"] = f"{type(error).__name__}: {error}"
                        snapshot["consequence"] = (
                            "the breaching weights could not be written. The stop stands on the "
                            "journalled verdict: the measured D and the persisted score vectors "
                            "are the evidence, not the diagnostic snapshot.")
                        snapshot_errors.append(dict(snapshot))
                    else:
                        record["failed_state"] = failed_state
                        snapshot["failed_state"] = failed_state
                monitor_journal.append(snapshot)
                persist_progress("gate_check")
                if stopped:
                    stop_record = _stop_document(record, last_passing, update, elapsed,
                                                 stop_reason, identity=identity,
                                                 snapshot=snapshot)
                    break

            for budget in crossing:
                overshoot = elapsed - budget
                budget_record = {
                    "target_gpu_seconds": budget, "reached": True,
                    "actual_gpu_seconds": json_number(elapsed),
                    "overshoot_seconds": json_number(overshoot),
                    "last_update_gpu_seconds": entry["update_gpu_seconds"],
                    "overshoot_within_one_update":
                        bool(overshoot <= (entry["update_gpu_seconds"] or 0.0) + 1e-9),
                    "precharged_gpu_seconds": json_number(precharged),
                    "updates": update, "exposures": dict(totals),
                    "core_token_exposures": int(totals.get("sequences", 0)) * CORE_LENGTH,
                    "last_loss": entry["loss"],
                    "gate_check": checks - 1,
                    "gate_note": "this checkpoint exists because the gate check above it passed",
                    "trajectory_wall_seconds": json_number(time.perf_counter() - started)}
                if cumulative_exposures is not None:
                    budget_record["distinct_exposures"] = dict(cumulative_exposures())
                if not budget_record["overshoot_within_one_update"]:
                    budget_record["overshoot_note"] = (
                        "this budget and an earlier one were both passed inside the same update, "
                        "so the overshoot is bounded by that update only in aggregate")
                evaluation_started = time.perf_counter()
                try:
                    if on_budget is not None:
                        budget_record.update(on_budget(budget, budget_record) or {})
                except BaseException as error:
                    # The budget is removed from `remaining` only after its endpoint
                    # was durably written. Removing it first and then failing deletes
                    # it from the reached AND the not-reached list at once, and a
                    # declared budget that appears in neither is unaccounted for.
                    evaluation_wall += time.perf_counter() - evaluation_started
                    budget_journal.append({
                        "record_kind": "budget_checkpoint_failed",
                        "target_gpu_seconds": budget, "reached": False, "updates": update,
                        "actual_gpu_seconds": json_number(elapsed),
                        "error": f"{type(error).__name__}: {error}",
                        "run": dict(identity or {}),
                        "note": ("the endpoint write failed, so this budget was NOT reached: it "
                                 "stays outstanding and is reported under budgets_not_reached "
                                 "rather than disappearing from both lists")})
                    persist_progress("budget_checkpoint_failed")
                    raise
                budget_record["budget_evaluation_wall_seconds"] = json_number(
                    time.perf_counter() - evaluation_started)
                evaluation_wall += time.perf_counter() - evaluation_started
                remaining.remove(budget)
                reached[str(budget)] = budget_record
                budget_journal.append(dict(budget_record, record_kind="budget_reached",
                                           run=dict(identity or {})))
                persist_progress("budget_crossing")
            if final:
                break
    except BaseException as error:                        # noqa: BLE001 - re-raised below
        failure = {"type": type(error).__name__, "message": str(error),
                   "updates": update, "attempted_update": attempt_state["attempt"],
                   "attempted_updates": attempted, "failed_attempts": failed_attempts,
                   "optimizer_stepped": bool(attempt_state["optimizer_stepped"]),
                   "scheduler_stepped": bool(attempt_state["scheduler_stepped"]),
                   "failed_work_gpu_seconds": json_number(failed_work_seconds),
                   "training_gpu_seconds": json_number(clock.elapsed_seconds),
                   "counting_note": (
                       "`updates` is completed updates, each with a journal line. The failing "
                       "attempt is counted under attempted_updates only; its device time is "
                       "charged and reported as failed work, and its exposures are NOT added to "
                       "the totals because the step never returned its statistics.")}
        if attempt_state["optimizer_stepped"] and not attempt_state["scheduler_stepped"]:
            failure["parameter_note"] = (
                "optimizer.step() completed and scheduler.step() did not, so the resident weights "
                "include this attempt while the schedule does not. The attempt is still not a "
                "completed update; failed_state.pt carries those weights.")
        stop_reason = stop_reason or "exception"
        try:
            failure["failed_state"] = saver.save_failed(
                update=update, gpu_seconds=clock.elapsed_seconds, stop_reason="exception")
        except Exception as nested:                       # pragma: no cover - best effort
            failure["failed_state_error"] = f"{type(nested).__name__}: {nested}"
        stop_record = _stop_document(failure, last_passing, update, clock.elapsed_seconds,
                                     stop_reason, identity=identity)
        raise
    finally:
        for budget in remaining:
            unreached[str(budget)] = {
                "target_gpu_seconds": budget, "reached": False,
                "stopped_at_update": update,
                "consumed_gpu_seconds": json_number(clock.elapsed_seconds),
                "stop_reason": stop_reason,
                "note": ("no checkpoint exists at this budget and none was fabricated; the "
                         "last-passing state below is diagnostic and is NOT this budget's "
                         "checkpoint")}
        if failure is not None or artifact_failure is not None:
            # An artifact/IO failure is not a gate stop. It ends the trajectory the
            # same way an exception does, and it may not be read as a completion or
            # as a prespecified likelihood stop.
            status = "failed"
        elif stopped:
            status = "stopped"
        elif unreached:
            # Every declared budget must be reached or the trajectory did not finish
            # the thing it was asked to do. A max_updates cap is an artificial end,
            # not a completion, and calling it one would let a truncated run be read
            # as a full-budget result.
            status = "incomplete"
        else:
            status = "completed"
        document = {
            "schema_version": TRAJECTORY_SCHEMA,
            "identity": dict(identity or {}),
            "status": status,
            "stop_reason": stop_reason,
            "updates": update, "checks": checks,
            "attempted_updates": attempted, "failed_attempts": failed_attempts,
            "failed_work_gpu_seconds": json_number(failed_work_seconds),
            "update_cap": max_updates, "update_cap_reached": bool(capped),
            "counting_note": ("`updates` counts completed optimizer updates, one journal line "
                              "each. An attempt that raised before optimizer.step() returned is "
                              "counted under attempted_updates and its device seconds under "
                              "failed_work_gpu_seconds; it is never a completed update."),
            "status_note": ("`completed` means every declared budget was reached under a passing "
                            "gate. A run capped by update_cap with budgets outstanding is "
                            "`incomplete`, never `completed`, and a run that ended because an "
                            "artifact could not be written is `failed`, never `stopped`: only a "
                            "declared gate condition produces `stopped`."),
            "budgets": dict(reached), "budgets_not_reached": dict(unreached),
            "exposures": dict(totals),
            "core_token_exposures": int(totals.get("sequences", 0)) * CORE_LENGTH,
            "last_passing": last_passing,
            "last_passing_is_selectable": False,
            "selection_note": ("a nominal budget that was not reached has no checkpoint. The "
                               "rolling last-passing state is diagnostic evidence about where the "
                               "trajectory was still parent-compatible; it is never relabelled as "
                               "a budget it did not reach, and it is not selectable even when "
                               "every budget WAS reached -- selection reads the budget "
                               "checkpoints, which carry their own diversity verdicts."),
            "selectable_endpoints": sorted(reached),
            "snapshot_errors": list(snapshot_errors),
            "cost": {
                "training_gpu_seconds": json_number(clock.elapsed_seconds),
                "precharged_gpu_seconds": json_number(precharged),
                "failed_work_gpu_seconds": json_number(failed_work_seconds),
                "diagnostic_wall_seconds": json_number(diagnostic_wall),
                "checkpoint_wall_seconds": json_number(saver.wall_seconds),
                "budget_evaluation_wall_seconds": json_number(evaluation_wall),
                "trajectory_wall_seconds": json_number(time.perf_counter() - started),
                "note": ("only updates and charged reference scoring appear in "
                         "training_gpu_seconds. Monitoring cost is measured by the monitor and "
                         "reported beside this block; nothing here is subtracted from anything.")},
            "journals": {"updates": updates_journal.document(),
                         "monitor": monitor_journal.document(),
                         "budgets": budget_journal.document()},
            "recent_updates": list(history_tail[-25:]),
            "rolling_checkpoint_note": ROLLING_NOTE}
        if cumulative_exposures is not None:
            document["distinct_exposures"] = dict(cumulative_exposures())
        if stop_record is not None:
            document["stop"] = stop_record
            save_json(directory / "stop.json", stop_record)
        if failure is not None:
            document["failure"] = failure
        if artifact_failure is not None:
            document["failure"] = dict(artifact_failure)
            document["artifact_failure"] = dict(artifact_failure)
        updates_journal.close()
        monitor_journal.close()
        budget_journal.close()
        save_json(directory / "trajectory.json", document)
        persist_progress("final")
    return document


def _stop_document(record, last_passing, update, gpu_seconds, stop_reason, *, identity=None,
                   snapshot=None):
    """What stopped, where, and which two states were actually observed.

    The check that *detected* the breach is not the moment it happened, and the
    interval between the last passing check and this one is **not** a bracket on
    the first crossing either: D is observed only at checks, so an excursion that
    crossed and recovered before the last passing check would have left exactly
    these two observations. What is recorded here is therefore the two observed
    states and the distance between them, and no first-crossing time at all.
    """
    previous_update = int(last_passing.get("update") or 0)
    previous_seconds = float(last_passing.get("training_gpu_seconds") or 0.0)
    document = {"stop_reason": stop_reason,
                "run": dict(identity or {}),
                "stop_check": {"update": int(update),
                               "training_gpu_seconds": json_number(gpu_seconds)},
                "last_passing_check": {"update": previous_update,
                                       "training_gpu_seconds": json_number(previous_seconds)},
                "interval_updates": int(update) - previous_update,
                "interval_gpu_seconds": json_number(float(gpu_seconds) - previous_seconds),
                "bracket_note": ("last_passing_check identifies the retained passing state; "
                                 "stop_check identifies the observed stop reason. An artifact or "
                                 "optimizer failure does not imply a likelihood breach. The "
                                 "monitor did not observe the first crossing, "
                                 "and the interval between those two checks is not a bracket on "
                                 "it: an earlier transient excursion that had recovered by the "
                                 "last passing check would look identical from here. No onset is "
                                 "claimed."),
                "detail": record,
                "last_passing": dict(last_passing)}
    if snapshot is not None:
        document["snapshot"] = dict(snapshot)
    return document


def trajectory_digest(document):
    """A content hash of a trajectory document, for the stage freeze to name."""
    import hashlib
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
