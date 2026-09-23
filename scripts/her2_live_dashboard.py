"""Serve a read-only, localhost dashboard for an existing guarded campaign.

Uses only the standard library; never imports the trainer or opens model weights.
Run from the repository root: python scripts/her2_live_dashboard.py
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("her2_dashboard")

#: How long a trajectory's own artifacts may go unwritten before the page stops
#: calling it running. A live supervisor says nothing about a live trainer: the
#: campaign heartbeat keeps ticking while a child hangs, so progress has to be
#: judged from the trajectory's own files. The bound has to clear the longest
#: legitimate silence, which is a gate check (about 14 s on the recorded
#: campaign) plus a checkpoint write, and the pre-first-update parent scoring
#: (about 70 s). Three minutes leaves room above both.
STALL_SECONDS = 180.0


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def number(value, default=0.0):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else default


def mean(row, key):
    value = row.get(key)
    return value.get("mean") if isinstance(value, dict) else value


def gate_point(row):
    """A completed gate check: the verdict, what it cost, and the distribution."""
    keys = ("D", "check", "update", "passed", "training_gpu_seconds", "pairs",
            "monitor_gpu_seconds", "monitor_wall_seconds", "reason",
            "mean_current_chosen_nll_per_residue", "mean_implicit_margin", "pair_accuracy")
    result = {k: row.get(k) for k in keys}
    result["record_kind"] = "gate_verdict"
    result["quantiles"] = (row.get("chosen_drop") or {}).get("quantiles", {})
    # A mean D of 0.34 with 1.4% of the population down more than five nats is a
    # different state from a uniform 0.34 shift, and the mean cannot tell them
    # apart. Checks written before the gate journalled these simply have none.
    fractions = row.get("chosen_drop_fractions") or {}
    result["fractions"] = {name: fractions.get(name) for name in
                           ("fraction_below_parent", "fraction_drop_gt1",
                            "fraction_drop_gt5", "fraction_below_uniform")
                           } if fractions else None
    return result


def snapshot_point(row):
    """What was written -- or could not be written -- because of a verdict.

    The trainer is careful that a declared gate stop and an artifact/IO failure
    are different outcomes: one advances a stage, the other does not. Dropping
    these lines shows both as "Stopped" with nothing on the page saying which
    happened, so they are kept and rendered apart.
    """
    keys = ("check", "update", "passed", "training_gpu_seconds", "D",
            "save_failed", "consequence", "scores_sha256", "verdict_journal_line")
    result = {k: row.get(k) for k in keys if k in row}
    result["record_kind"] = "snapshot"
    for name in ("last_passing", "failed_state"):
        written = row.get(name)
        if isinstance(written, dict):
            result[name] = {k: written.get(k)
                            for k in ("path", "sha256", "update", "wall_seconds")}
    return result


def monitor_point(row):
    """Both monitor record kinds, tagged. Anything else is not a monitor record."""
    kind = row.get("record_kind")
    if kind == "gate_verdict":
        return gate_point(row)
    if kind == "snapshot":
        return snapshot_point(row)
    return None


def verdicts(rows):
    """Gate checks only. Costs and check counts are sums over THESE rows.

    A snapshot line repeats its verdict's ``D`` and sits next to it in the
    journal; counting or summing across both kinds double-counts every check.
    """
    return [row for row in rows if row.get("record_kind") == "gate_verdict"]


def snapshots(rows):
    return [row for row in rows if row.get("record_kind") == "snapshot"]


def update_point(row):
    keys = ("update", "cumulative_gpu_seconds", "loss", "gradient_norm",
            "learning_rate_used", "cycle", "position", "sign_accuracy",
            "hinge_active_fraction", "exposures")
    result = {k: row.get(k) for k in keys}
    for k in ("margin", "coefficient", "modified_margin", "modified_coefficient",
              "chosen_nll_per_residue", "rejected_nll_per_residue",
              "preference_component", "hinge_component", "nll_component"):
        result[k] = mean(row, k)
    return result


class Journal:
    """Incremental reader. An unfinished final line is retried on the next poll.

    Resuming from a byte offset is sound only while the file is *the same file*,
    appended to. A journal that was replaced rather than appended -- a rerun into
    a reused directory, a restored copy -- can come back at or above the old
    offset, and an offset-only reader then serves the previous run's lines as this
    run's history. Shrinking is therefore not the only invalidation: the reader
    also matches the file's identity (device, inode) and re-checks the bytes it
    has already consumed. ``os.replace`` preserves the source mtime, so a
    replacement can reproduce size and mtime exactly; the consumed-bytes check is
    what catches an in-place rewrite that lands on the same inode.

    The consumed-bytes check stands on its own, which matters on the Windows
    training box: ``st_ino`` is the NTFS file index there, but it is 0 on volumes
    that do not supply one, and an identity check alone would then never fire.
    """

    #: Enough of the file head to anchor the consumed prefix. Journal lines here
    #: are whole JSON objects, so the first few hundred bytes differ between two
    #: different runs' first records.
    HEAD_BYTES = 512

    def __init__(self, path, project):
        self.path, self.project = Path(path), project
        self.offset = 0
        self.rows = []
        self.bad_lines = 0
        self.resets = 0
        self.stamp = None
        self.head = b""

    def _discard(self):
        """Forget everything read so far. The retained rows describe other bytes."""
        self.offset, self.rows, self.bad_lines, self.head = 0, [], 0, b""
        self.resets += 1

    def read(self):
        try:
            stat = self.path.stat()
            stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if stamp == self.stamp:
                return self.rows
            if self.stamp is not None and stamp[:2] != self.stamp[:2]:
                self._discard()                      # a different file at the same path
            elif stat.st_size < self.offset:
                self._discard()                      # truncated
            with self.path.open("rb") as handle:
                head = handle.read(self.HEAD_BYTES)
                # ``self.head`` is only ever the bytes already consumed, which an
                # append cannot change. A mismatch means they were rewritten.
                if self.head and head[:len(self.head)] != self.head:
                    self._discard()
                handle.seek(self.offset)
                while True:
                    line = handle.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offset = handle.tell()
                    try:
                        row = self.project(json.loads(line))
                    except (ValueError, TypeError, AttributeError):
                        self.bad_lines += 1
                        continue
                    if row is not None:
                        self.rows.append(row)
                self.head = head[:min(self.offset, self.HEAD_BYTES)]
            self.stamp = stamp
        except FileNotFoundError:
            # A removed artifact must not keep masquerading as current evidence.
            self.offset, self.rows, self.stamp, self.head = 0, [], None, b""
        return self.rows


def tail(path, limit=50, byte_limit=65536):
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, 2)
            start = max(0, handle.tell() - byte_limit)
            handle.seek(start)
            data = handle.read()
        lines = data.splitlines(keepends=True)
        if start and lines:
            lines = lines[1:]
        return [s.decode("utf-8", errors="replace").rstrip("\r\n")
                for s in lines if s.endswith(b"\n")][-limit:]
    except FileNotFoundError:
        return []


def latest_update(path):
    for line in reversed(tail(path, limit=6)):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {}


class Campaign:
    def __init__(self, output, control, *, stall_seconds=STALL_SECONDS):
        self.output, self.control = Path(output), Path(control)
        self.stall_seconds = float(stall_seconds)
        self.cache, self.journals = {}, {}
        self.update_journal = None
        self.lock = threading.RLock()
        self.warnings = []
        self.gpu_time, self.gpu = 0, None

    def document(self, path):
        path = Path(path)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if path in self.cache and self.cache[path][0] == stamp:
                return self.cache[path][1]
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            self.cache[path] = stamp, value
            return value
        except FileNotFoundError:
            self.cache.pop(path, None)
            return {}
        except (OSError, ValueError) as exc:
            self.warnings.append(f"Could not read {path.name}: {type(exc).__name__}; retrying.")
            return {}

    def monitors(self, directory):
        path = directory / "monitor.jsonl"
        if path not in self.journals:
            self.journals[path] = Journal(path, monitor_point)
        journal = self.journals[path]
        rows = journal.read()
        if journal.bad_lines:
            self.warnings.append(f"{directory.name}: {journal.bad_lines} unreadable monitoring lines.")
        if journal.resets:
            self.warnings.append(
                f"{directory.name}: monitor.jsonl was replaced, not appended to; the displayed "
                f"history was reloaded from the file now on disk.")
        return rows

    def gpu_status(self):
        if time.monotonic() - self.gpu_time < 15:
            return self.gpu
        self.gpu_time = time.monotonic()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            columns = result.stdout.splitlines()[0].split(",")
            self.gpu = dict(name=columns[0].strip(), utilization=float(columns[1]),
                            memory_used_mib=float(columns[2]), memory_total_mib=float(columns[3]))
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            self.gpu = None
        return self.gpu

    def run(self, arm, seed, campaign_status):
        run_id = f"stage{arm['stage']}/{arm['arm_id']}_seed{seed}"
        directory = self.output / run_id
        summary = self.document(directory / "summary.json")
        trajectory = summary or self.document(directory / "trajectory.json")
        progress = trajectory or self.document(directory / "trajectory_progress.json")
        ledger = self.document(directory / "run.json")
        rows = self.monitors(directory)
        checks = verdicts(rows)
        written = snapshots(rows)
        gate = checks[-1] if checks else None
        latest = latest_update(directory / "updates.jsonl")
        # Progress is judged from the trajectory's OWN artifacts. The campaign
        # heartbeat says the supervisor is alive, which is a different claim.
        updated_paths = [directory / n for n in ("updates.jsonl", "monitor.jsonl", "run.json")]
        modified = max((p.stat().st_mtime for p in updated_paths if p.exists()), default=None)
        idle = (time.time() - modified) if modified is not None else None
        state = trajectory.get("status")
        if not state:
            if progress.get("stopped") or (gate and gate.get("passed") is False):
                state = "stopped"
            elif ledger.get("status") == "failed":
                state = "failed"
            elif directory.exists():
                if campaign_status != "running":
                    state = "interrupted"
                elif idle is not None and idle > self.stall_seconds:
                    # A live supervisor can watch a hung child indefinitely. Saying
                    # "running" here is the one claim this page cannot support.
                    state = "stalled"
                else:
                    state = "running"
            else:
                state = "queued"
        cost = trajectory.get("cost", {})
        recorded_costs = [latest.get("cumulative_gpu_seconds"),
                          progress.get("training_gpu_seconds"), cost.get("training_gpu_seconds")]
        training = max((number(v) for v in recorded_costs), default=0)
        budget_records = dict(progress.get("budgets") or {})
        for line in tail(directory / "budgets.jsonl", limit=10, byte_limit=262144):
            try:
                row = json.loads(line)
                if row.get("reached") is True:
                    budget_records[str(row["target_gpu_seconds"])] = row
            except (ValueError, KeyError):
                continue
        reached = sorted(float(k) for k, v in budget_records.items() if v.get("reached") is True)
        reference = self.document(directory / "parent_validation_reference.json")
        # Verdict lines only. A snapshot line repeats its verdict's cost fields and
        # sits next to it, so summing across both kinds double-counts every check.
        monitoring = sum(number(r.get("monitor_gpu_seconds")) for r in checks)
        parent_gpu = number(reference.get("gpu_seconds"))
        excluded = summary.get("excluded_costs") or {}
        if excluded.get("parent_validation_reference_reused_from_disk"):
            parent_gpu = 0.0
        # Rolling last-passing saves are real work that no clock on this page used
        # to show: one full model serialization plus a sha256 read per passing
        # check. The recorded campaign did 3,987 of them.
        saves = [r for r in written if r.get("last_passing")]
        checkpoint_wall = sum(number((r.get("last_passing") or {}).get("wall_seconds"))
                              for r in saves)
        # End-of-run I/O split, when the trajectory got far enough to write one.
        io_wall = dict((summary.get("cost_summary") or {}).get("io_wall_seconds") or {})
        io_wall.pop("note", None)
        failure = trajectory.get("failure") or trajectory.get("artifact_failure")
        artifact_errors = [r for r in written if r.get("save_failed")]
        return dict(id=run_id, stage=arm["stage"], arm=arm["arm_id"],
                    objective=arm["objective"], coefficients=arm.get("coefficients", {}), seed=seed,
                    status=state, updates=max(number(progress.get("updates")), number(latest.get("update"))),
                    training_gpu_seconds=training, monitor_gpu_seconds=monitoring + parent_gpu,
                    monitor_checks_gpu_seconds=monitoring, parent_reference_gpu_seconds=parent_gpu,
                    checks=len(checks), gate=gate, budgets_reached=reached,
                    exposures=latest.get("exposures") or progress.get("exposures"),
                    stop_reason=progress.get("stop_reason") or trajectory.get("status_note"),
                    last_passing=progress.get("last_passing"), latest_update=update_point(latest),
                    updated_at=modified, idle_seconds=idle,
                    rolling_checkpoint_saves=len(saves),
                    rolling_checkpoint_wall_seconds=checkpoint_wall,
                    io_wall_seconds=io_wall or None,
                    attempted_updates=progress.get("attempted_updates"),
                    failed_attempts=progress.get("failed_attempts"),
                    failed_work_gpu_seconds=progress.get("failed_work_gpu_seconds"),
                    precharged_gpu_seconds=progress.get("precharged_gpu_seconds"),
                    failure=failure or None,
                    artifact_errors=artifact_errors,
                    snapshot_errors=trajectory.get("snapshot_errors") or [],
                    whole_run_wall_seconds=summary.get("whole_run_wall_seconds"))

    def snapshot(self):
        with self.lock:
            self.warnings = []
            plan = self.document(self.output / "plan.json")
            status = self.document(self.control / "campaign_status.json")
            now = time.time()
            try:
                heartbeat_age = now - datetime.fromisoformat(status["updated_at"]).timestamp()
            except (KeyError, ValueError):
                heartbeat_age = None
            effective_status = status.get("status", "unknown")
            if effective_status == "running" and (heartbeat_age is None or heartbeat_age > 30):
                effective_status = "stale"
            runs = [self.run(arm, seed, effective_status)
                    for arm in plan.get("arms", []) if arm.get("reused_from_stage") is None
                    for seed in plan.get("seeds", [])]
            counts = dict(Counter(r["status"] for r in runs))
            active = next((r["id"] for r in runs if r["status"] == "running"), None)
            for stalled in (r for r in runs if r["status"] == "stalled"):
                self.warnings.append(
                    f"{stalled['id']} has written nothing for "
                    f"{stalled['idle_seconds']:.0f}s while the campaign heartbeat is current. "
                    f"The supervisor is alive; this trajectory's progress is not confirmed.")
            for broken in (r for r in runs if r["failure"] or r["artifact_errors"]):
                self.warnings.append(
                    f"{broken['id']}: {(broken['failure'] or {}).get('type') or 'artifact write'} "
                    f"failure recorded. This is not a gate stop and does not advance a stage.")
            phases = [{k: p.get(k) for k in ("stage", "phase", "started_at", "ended_at",
                                            "wall_seconds", "exit_code")}
                      for p in status.get("phases", [])]
            log = []
            active_phase = status.get("active_phase", "")
            if active_phase and all(c.isalnum() or c == "_" for c in active_phase):
                log = tail(self.output / f"driver_{active_phase}.log", 45)
            stage_rows = []
            for stage in (1, 2, 3):
                stage_runs = [r for r in runs if r["stage"] == stage]
                stage_rows.append(dict(stage=stage, total=len(stage_runs),
                    counts=dict(Counter(r["status"] for r in stage_runs)),
                    frozen=(self.output / f"stage{stage}_complete.json").exists()))
            return clean(dict(
                generated_at=datetime.now(timezone.utc).isoformat(), status=effective_status,
                recorded_status=status.get("status"), heartbeat_age_seconds=heartbeat_age,
                active_phase=active_phase, active_run=active,
                started_at=status.get("started_at"),
                elapsed_wall_seconds=status.get("elapsed_wall_seconds"),
                training_allocation_gpu_seconds=status.get("training_allocation_gpu_seconds"),
                recorded_training_gpu_seconds=sum(r["training_gpu_seconds"] for r in runs),
                recorded_monitor_gpu_seconds=sum(r["monitor_gpu_seconds"] for r in runs),
                recorded_checkpoint_wall_seconds=sum(r["rolling_checkpoint_wall_seconds"]
                                                     for r in runs),
                recorded_checkpoint_saves=sum(r["rolling_checkpoint_saves"] for r in runs),
                stall_seconds=self.stall_seconds,
                counts=counts, total_runs=len(runs), gate=plan.get("gate", {}),
                budgets=plan.get("budgets_gpu_seconds", []), runs=runs, stages=stage_rows,
                phases=phases, gpu=self.gpu_status(), log=log,
                warnings=self.warnings, error=status.get("error"), output=str(self.output)))

    def detail(self, run_id):
        with self.lock:
            plan = self.document(self.output / "plan.json")
            allowed = {f"stage{a['stage']}/{a['arm_id']}_seed{s}"
                       for a in plan.get("arms", []) if a.get("reused_from_stage") is None
                       for s in plan.get("seeds", [])}
            if run_id not in allowed:
                raise KeyError(run_id)
            directory = self.output / run_id
            path = directory / "updates.jsonl"
            if self.update_journal is None or self.update_journal.path != path:
                self.update_journal = Journal(path, update_point)
            updates = self.update_journal.read()
            # Bound transfer size without smoothing the recorded values. Retain the
            # first 100 updates to show onset, then evenly sample, and keep the last.
            stride = max(1, math.ceil(max(0, len(updates) - 100) / 1200))
            plotted = updates[:100] + updates[100::stride]
            if updates and (not plotted or plotted[-1] != updates[-1]):
                plotted.append(updates[-1])
            rows = self.monitors(directory)
            return clean(dict(id=run_id, gates=verdicts(rows), snapshots=snapshots(rows),
                              updates=plotted, update_count=len(updates), plot_stride=stride,
                              unreadable_update_lines=self.update_journal.bad_lines,
                              update_journal_resets=self.update_journal.resets))


class AuditRun:
    """Read-only view of a HER2 support-audit run directory.

    Additive: it shares the server and the security posture with the campaign
    view and imports nothing from the audit package -- no model, no torch, no
    scientific module -- so a broken or half-written run directory can never do
    more than make this panel say so.

    Three things it refuses to do, because each of them turns an unfinished audit
    into a finished-looking one:

    * invent a denominator. A stage records ``total: null`` when it cannot know
      one, and this reports a count with ``fraction: null`` rather than a bar.
    * treat a ``running`` heartbeat as progress. A stage whose progress file has
      not been touched for longer than the stall bound is reported ``stalled``.
    * call the audit complete because the process exited. Completion is read from
      the ``audit_complete.json`` the CLI writes only after its own requirement
      checks pass, and the unmet requirements are shown when it is absent.
    """

    STAGES = ("inventory", "prepare", "preflight", "freeze", "score", "ches", "decide", "report")
    #: Documents this view may read, by logical name. Nothing else is opened, so a
    #: crafted name cannot walk out of the run directory.
    DOCUMENTS = {
        "inventory": "inventory.json",
        "coverage": "summaries/coverage.json",
        "checkpoints": "summaries/checkpoints.json",
        "ches": "summaries/ches.json",
        "decision": "decision.json",
        "freeze": "audit_spec_frozen.json",
        "verification": "verification.json",
        "complete": "audit_complete.json",
    }

    def __init__(self, root, *, stall_seconds=STALL_SECONDS):
        self.root = Path(root)
        self.stall_seconds = float(stall_seconds)
        self.lock = threading.Lock()
        self.errors = []

    # -- safe, bounded reads ----------------------------------------------
    def path(self, relative):
        """Join one forward-slash relative name under the run root, or refuse it.

        Absolute names, drive letters, empty and traversing components are all
        rejected before the join: ``os.path.join(base, "", "etc", "passwd")``
        happily produces a path under ``base``, so a containment check alone is
        not enough to refuse ``/etc/passwd``.
        """
        parts = str(relative).replace("\\", "/").split("/")
        if any(part in ("", ".", "..") or ":" in part for part in parts):
            raise KeyError(relative)
        base = Path(os.path.normpath(os.path.abspath(str(self.root))))
        target = Path(os.path.normpath(os.path.join(str(base), *parts)))
        if base != target and base not in target.parents:
            raise KeyError(relative)
        return target

    def document(self, relative):
        try:
            target = self.path(relative)
        except KeyError:
            return None
        if not target.is_file():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.errors.append(f"{relative}: {type(error).__name__}: {error}")
            return None

    def age(self, relative):
        try:
            target = self.path(relative)
        except KeyError:
            return None
        if not target.is_file():
            return None
        return max(0.0, datetime.now(timezone.utc).timestamp() - target.stat().st_mtime)

    # -- the panel ---------------------------------------------------------
    def stages(self):
        rows = []
        for name in self.STAGES:
            relative = f"progress/{name}.json"
            record = self.document(relative)
            if name == "freeze" and record is None:
                marker = self.document(self.DOCUMENTS["freeze"])
                if marker and marker.get("record_kind") == "audit_spec_frozen" \
                        and (marker.get("git") or {}).get("commit"):
                    record = {"status": "completed", "completed": 1, "total": 1,
                              "current": None, "error": None,
                              "started_at": marker.get("frozen_at")}
            if record is None:
                rows.append({"stage": name, "status": "not_started", "completed": None,
                             "total": None, "fraction": None, "current": None, "error": None,
                             "stale": False, "age_seconds": None})
                continue
            age = self.age(relative)
            status = record.get("status")
            stale = bool(status == "running" and age is not None and age > self.stall_seconds)
            total = record.get("total")
            completed = record.get("completed")
            rows.append({
                "stage": name,
                # A stage that stopped writing is not running, whatever its file says.
                "status": "stalled" if stale else status,
                "recorded_status": status,
                "completed": completed, "total": total,
                "fraction": (completed / total if isinstance(total, (int, float)) and total
                             and isinstance(completed, (int, float)) else None),
                "total_note": record.get("total_note"),
                "current": record.get("current"), "error": record.get("error"),
                "stale": stale, "age_seconds": age,
                "started_at": record.get("started_at"),
                "elapsed_wall_seconds": record.get("elapsed_wall_seconds")})
        return rows

    def snapshot(self):
        with self.lock:
            self.errors = []
            if not self.root.is_dir():
                return clean({"present": False, "root": self.root.name,
                              "generated_at": datetime.now(timezone.utc).isoformat(),
                              "stages": [], "errors": [], "complete": False,
                              "unmet_requirements": [], "decision": None, "frozen": None,
                              "counts": {}, "coverage": {}, "results": {}, "verification": {},
                              "note": "no audit run directory yet"})
            inventory = self.document(self.DOCUMENTS["inventory"]) or {}
            coverage_document = self.document(self.DOCUMENTS["coverage"]) or {}
            checkpoints = self.document(self.DOCUMENTS["checkpoints"]) or {}
            ches = self.document(self.DOCUMENTS["ches"]) or {}
            decision = self.document(self.DOCUMENTS["decision"]) or {}
            freeze = self.document(self.DOCUMENTS["freeze"])
            verification = self.document(self.DOCUMENTS["verification"]) or {}
            complete = self.document(self.DOCUMENTS["complete"])
            coverage = inventory.get("coverage") or coverage_document.get("inventory_coverage") or {}
            stages = self.stages()
            current = next((row for row in stages if row["status"] == "running"), None)
            return clean({
                "present": True, "root": self.root.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "audit_id": inventory.get("audit_id") or decision.get("audit_id"),
                "protocol": inventory.get("protocol"),
                "stages": stages,
                "active_stage": current["stage"] if current else None,
                "checkpoint": (current or {}).get("current"),
                "frozen": None if not freeze else {
                    "commit": (freeze.get("git") or {}).get("commit"),
                    "frozen_at": freeze.get("frozen_at"),
                    "sources": len(freeze.get("source_sha256") or {}),
                    "inputs": freeze.get("input_count"),
                    "evidence": len(freeze.get("evidence_sha256") or {})},
                "counts": {
                    "states_enumerated": coverage.get("total"),
                    "states_verified": coverage.get("verified_total"),
                    "states_expected": (coverage.get("expected") or {}).get("total"),
                    "distinct_computations":
                        (inventory.get("deduplication") or {}).get("distinct_computations"),
                    "scored": (coverage_document.get("scored_count")
                               if coverage_document.get("scored_count") is not None else
                               next((row.get("completed") for row in stages
                                     if row["stage"] == "score"), None)),
                    "parent_banks": len(inventory.get("parent_banks") or {}),
                    "ches_parent_blocks": len(ches.get("parent") or {}),
                    "ches_endpoint_blocks": len(ches.get("endpoints") or {}),
                    "increments": len(ches.get("increments") or {})},
                "coverage": {
                    "complete": coverage.get("complete"),
                    "shortfalls": coverage.get("shortfalls") or [],
                    "unverified": sorted((coverage_document.get("unverified") or {}))[:20]},
                "results": {
                    "self_controls": {name: block.get("within_tolerance")
                                      for name, block in
                                      sorted((checkpoints.get("self_controls") or {}).items())},
                    "paired_comparisons":
                        (checkpoints.get("paired_method_differences") or {}).get("count"),
                    "ches_increment_gaps": len(ches.get("increment_gaps") or []),
                    "endpoints": self.endpoint_rows(checkpoints, inventory)},
                "decision": None if not decision else {
                    "outcome": decision.get("outcome"),
                    "blocking": decision.get("blocking") or [],
                    "coverage_complete": decision.get("coverage_complete"),
                    "audit_complete": decision.get("audit_complete"),
                    "methods": {name: {"outcome": block.get("outcome"),
                                       "seeds_usable": block.get("seeds_usable"),
                                       "seeds_crossing": block.get("seeds_crossing"),
                                       "seeds_declared": block.get("seeds_declared")}
                                for name, block in sorted((decision.get("methods") or {}).items())}},
                "verification": {"immutable": verification.get("immutable"),
                                 "shards_checked": verification.get("shards_checked"),
                                 "problems": (verification.get("problems") or [])[:20]},
                # Completion is the marker the CLI writes after its own checks, never
                # "the last stage's process exited".
                "complete": bool(complete) and verification.get("immutable") is not False
                            and not self.errors,
                "completed_at": (complete or {}).get("completed_at"),
                "unmet_requirements": ((complete or {}).get("requirements") or
                                       (decision.get("completion") or {})).get("unmet") or [],
                "errors": list(self.errors)})

    def endpoint_rows(self, checkpoints, inventory, limit=200):
        """The scored 600 s endpoints, enough to show the decision inputs honestly."""
        records = {record.get("id"): record for record in inventory.get("records") or []}
        rows = []
        for identifier, block in sorted((checkpoints.get("checkpoints") or {}).items()):
            record = records.get(identifier) or {}
            tails = ((block.get("tails") or {}).get("counts") or {})
            rows.append({
                "id": identifier, "role": record.get("role"), "arm_id": record.get("arm_id"),
                "seed": record.get("seed"),
                "budget_gpu_seconds": record.get("nominal_budget_gpu_seconds"),
                "forward_kl": (block.get("forward_kl") or {}).get("mean"),
                "ci_low": (block.get("forward_kl") or {}).get("ci_low"),
                "ci_high": (block.get("forward_kl") or {}).get("ci_high"),
                "tenfold_fraction": (tails.get("tenfold") or {}).get("fraction"),
                "tenfold_wilson_lower": (tails.get("tenfold") or {}).get("lower")})
            if len(rows) >= limit:
                break
        return rows


class ReplayRun:
    """Read-only view of a HER2 parent-replay campaign directory.

    Additive in the same way the audit panel is: it shares the server and the
    security posture, imports nothing from the experiment package, and opens no
    model weights. A half-written or absent run directory can do no more than make
    this panel say so.

    Five things it refuses to do, because each turns an unstarted, interrupted or
    dead campaign into a healthy-looking one:

    * call a launch healthy because a process exists. "Healthy" needs a journalled
      completed optimizer update **and** a gate check that was measured and passed,
      and both are read back off the trajectory's own files. That evidence is also
      *historical*: it stays true after the campaign dies, so it is qualified by the
      writer's current state and never displayed as an unqualified tile beside a
      failed or interrupted campaign.
    * decide liveness from a recorded status or from a heartbeat's age. The OS lock
      is what a forced kill releases, so the lock is the liveness test -- observed
      read-only, without creating it and without writing an owner record. A missing
      heartbeat does not make a writer alive and an old one does not make it dead:
      a live owner can be thirty seconds into scoring 25,722 validation pairs. The
      heartbeat's age is reported separately, as what it is -- a progress and stall
      signal.
    * invent a queue. Before the queue document exists the panel says the campaign
      has not been launched, rather than showing 36 empty rows as though they were
      waiting.
    * read a checkpoint file as a reached endpoint. Endpoints come from the
      trajectory's own terminal status or from its append-only endpoint journal,
      both of which are written after the bytes were re-read and the evaluation was
      saved.
    * report counters from the progress file for a trajectory that is still moving.
      Progress is written at checks; the journals are written at every update, and
      they are what this panel counts.
    """

    #: Documents this view may read, by logical name. Nothing else is opened, so a
    #: crafted name cannot walk out of the run directory.
    DOCUMENTS = {
        "status": "status.json",
        "campaign": "campaign_status.json",
        "queue": "queue.json",
        "freeze": "training_spec_frozen.json",
        "banks": "banks_manifest.json",
        "preflight": "preflight.json",
        "verification": "verification.json",
        "tables": "summaries/tables.json",
        "heartbeat": "campaign_heartbeat.json",
    }
    #: Statuses a trajectory's own artifacts can report.
    TERMINAL = ("completed", "stopped_by_gate", "incomplete", "failed")
    #: The campaign's exclusive write lock, observed and never taken.
    LOCK_FILE = "campaign.lock"

    def __init__(self, root, *, stall_seconds=STALL_SECONDS):
        self.root = Path(root)
        self.stall_seconds = float(stall_seconds)
        self.lock = threading.Lock()
        self.errors = []

    def path(self, relative):
        parts = str(relative).replace("\\", "/").split("/")
        if any(part in ("", ".", "..") or ":" in part for part in parts):
            raise KeyError(relative)
        base = Path(os.path.normpath(os.path.abspath(str(self.root))))
        target = Path(os.path.normpath(os.path.join(str(base), *parts)))
        if base != target and base not in target.parents:
            raise KeyError(relative)
        return target

    def document(self, relative):
        try:
            target = self.path(relative)
        except KeyError:
            return None
        if not target.is_file():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.errors.append(f"{relative}: {type(error).__name__}: {error}")
            return None

    def age(self, relative):
        try:
            target = self.path(relative)
        except KeyError:
            return None
        if not target.is_file():
            return None
        return max(0.0, datetime.now(timezone.utc).timestamp() - target.stat().st_mtime)

    def lock_observation(self):
        """Who holds the campaign write lock, observed without mutating anything.

        The same question :func:`her2_replay_campaign.lock_state` answers, asked
        again here rather than imported: this server deliberately imports nothing
        from the experiment package. It opens the existing lock for update, tests
        it, drops it immediately and writes nothing -- no owner record, and no lock
        file is created where there was none. ``missing``, ``held``, ``released``
        and ``inaccessible`` are four different answers and "I could not tell" is
        never reported as "nobody is there".
        """
        try:
            target = self.path(self.LOCK_FILE)
        except KeyError:
            return {"state": "inaccessible", "reason": "the lock path is not inside the run root"}
        if not target.is_file():
            return {"state": "missing",
                    "reason": "no lock file exists; no campaign process has taken this directory"}
        try:
            handle = target.open("r+b")
        except OSError as error:
            return {"state": "inaccessible", "reason": f"{type(error).__name__}: {error}"}
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            return {"state": "held", "reason": f"{type(error).__name__}: {error}"}
        finally:
            handle.close()
        return {"state": "released",
                "reason": ("a lock file exists and nobody holds it. The kernel releases it on a "
                           "forced kill, so this covers a clean exit and a killed owner alike")}

    def journal_counters(self, trajectory):
        """Completed updates, exposures and reached endpoints, from the journals.

        The progress file is written at checks, so between two checks it is behind
        by up to a whole interval; the update journal gets a record as each update
        completes. ``checks`` still comes from the progress file, because a check is
        exactly when that file is written.
        """
        record, endpoints = None, []
        try:
            updates = self.path(f"trajectories/{trajectory}/updates.jsonl")
            journal = self.path(f"trajectories/{trajectory}/endpoints.jsonl")
        except KeyError:
            return {"updates": None, "exposures": {}, "endpoints_reached": []}
        for line in reversed(tail(updates, limit=4)):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if candidate.get("record_kind") == "update":
                record = candidate
                break
        for line in tail(journal, limit=12):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if candidate.get("record_kind") == "exposure_endpoint":
                endpoints.append(str(candidate.get("update")))
        return {"updates": (record or {}).get("update"),
                "exposures": (record or {}).get("exposures") or {},
                "endpoints_reached": sorted(endpoints, key=lambda value: int(value))}

    def last_gate(self, trajectory):
        """The most recent measured gate verdict for one trajectory, or ``None``."""
        try:
            target = self.path(f"trajectories/{trajectory}/monitor.jsonl")
        except KeyError:
            return None
        for line in reversed(tail(target, limit=40)):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("record_kind") == "gate_verdict":
                preservation = row.get("preservation") or {}
                forward = preservation.get("forward_kl") or {}
                tails = ((preservation.get("tails") or {}).get("counts") or {})
                return {"update": row.get("update"), "D": row.get("D"),
                        "passed": row.get("passed"), "reason": row.get("reason"),
                        "stop_reason": row.get("stop_reason"),
                        "forward_kl": forward.get("mean"),
                        "tenfold_fraction": (tails.get("tenfold") or {}).get("fraction"),
                        "conditional_kl": (preservation.get("conditional_kl") or {}).get("mean"),
                        "preservation_available": preservation.get("available")}
        return None

    def first_evidence(self, trajectories):
        """The first journalled update and the first measured gate check, if they exist."""
        for name in trajectories:
            try:
                updates = self.path(f"trajectories/{name}/updates.jsonl")
                monitor = self.path(f"trajectories/{name}/monitor.jsonl")
            except KeyError:
                continue
            if not updates.is_file() or not monitor.is_file():
                continue
            first_update, first_check = None, None
            # The FIRST journalled update, read from the head of the file. The last
            # line would answer a different question: that a process is still
            # writing, which is not what "the first update completed" claims.
            try:
                with updates.open("rb") as handle:
                    head = handle.readline()
                first_update = json.loads(head) if head.strip() else None
            except (OSError, ValueError):
                first_update = None
            try:
                with monitor.open("rb") as handle:
                    for _ in range(40):
                        line = handle.readline()
                        if not line:
                            break
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row.get("record_kind") == "gate_verdict":
                            first_check = row
                            break
            except OSError:
                first_check = None
            if first_update and first_check:
                return {"trajectory": name,
                        "first_update_completed": first_update.get("update"),
                        "first_check_update": first_check.get("update"),
                        "first_check_passed": bool(first_check.get("passed")),
                        "first_check_D": first_check.get("D"),
                        "healthy": bool(first_update.get("update")
                                        and first_check.get("passed")),
                        "basis": "read back from the trajectory's own journals"}
        return {"trajectory": None, "first_update_completed": None, "first_check_update": None,
                "first_check_passed": None, "healthy": False,
                "basis": ("no trajectory has both a journalled completed update and a measured "
                          "gate check yet; a started process is not a healthy launch")}

    def trajectory(self, entry, *, writer_state):
        name = entry.get("trajectory")
        terminal = self.document(f"trajectories/{name}/status.json")
        progress = self.document(f"trajectories/{name}/trajectory_progress.json")
        try:
            exists = self.path(f"trajectories/{name}").is_dir()
        except KeyError:
            exists = False
        if terminal:
            status = terminal.get("status")
        elif not exists:
            status = "queued"
        elif writer_state == "alive":
            status = "running"
        elif writer_state == "gone":
            status = "interrupted"
        else:
            status = "unknown"
        source = terminal or progress or {}
        journal = {} if terminal else self.journal_counters(name)
        # No fallback to the progress file. Its counters are written at checks and
        # are carried in their own fields; letting them stand in here would pair
        # them with a row that says the counters came from the journals.
        exposures = (source.get("exposures") if terminal else journal.get("exposures")) or {}
        reached = ((terminal or {}).get("endpoints_reached") if terminal
                   else journal.get("endpoints_reached")) or []
        return {"trajectory": name, "arm_id": entry.get("arm_id"), "task": entry.get("task"),
                "replay_lambda": entry.get("replay_lambda"), "seed": entry.get("seed"),
                "is_control": entry.get("is_control"),
                "status": status, "terminal": bool(terminal),
                "stop_reason": source.get("stop_reason"),
                "updates": (source.get("updates") if terminal else journal.get("updates")),
                "checks": source.get("checks"),
                "progress_file_updates": None if terminal else (progress or {}).get("update"),
                "progress_file_exposures": (None if terminal
                                            else ((progress or {}).get("exposures") or None)),
                "counters_from": ("its terminal status" if terminal
                                  else "its append-only journals"),
                "chosen_exposures": exposures.get("chosen"),
                "replay_exposures": exposures.get("replay"),
                "endpoints_reached": sorted(reached, key=lambda value: int(value)),
                "gate": self.last_gate(name),
                "cost": (source.get("cost") or {}).get("seconds")}

    def snapshot(self):
        with self.lock:
            self.errors = []
            now = datetime.now(timezone.utc).isoformat()
            if not self.root.is_dir():
                return clean({"present": False, "root": self.root.name, "generated_at": now,
                              "phase": "no_run_directory", "trajectories": [], "counts": {},
                              "queue_present": False, "health": None, "frozen": None,
                              "banks": None, "errors": [],
                              "note": "no replay run directory yet"})
            campaign = self.document(self.DOCUMENTS["campaign"]) or {}
            queue_document = self.document(self.DOCUMENTS["queue"]) or {}
            queue = queue_document.get("queue") or []
            freeze = self.document(self.DOCUMENTS["freeze"])
            banks = self.document(self.DOCUMENTS["banks"])
            heartbeat = self.document(self.DOCUMENTS["heartbeat"]) or {}
            heartbeat_age = self.age(self.DOCUMENTS["heartbeat"])
            recorded = campaign.get("status")
            lock = self.lock_observation()
            stalled = (heartbeat_age is not None and heartbeat_age > self.stall_seconds)
            # The lock decides liveness; the recorded status and the heartbeat
            # describe what the writer last said and how long ago it said it.
            if lock["state"] == "held":
                writer_state = "alive"
                writer_note = (
                    f"a live process holds the write lock and the heartbeat is {heartbeat_age:.0f}s"
                    " old. A stale heartbeat under a held lock usually means a long evaluation, "
                    "not a dead writer." if stalled else None)
            elif lock["state"] == "inaccessible":
                writer_state = "unknown"
                writer_note = (f"the write lock could not be read ({lock['reason']}), so liveness "
                               "is unknown. Unknown is not reported as finished.")
            elif recorded == "failed":
                writer_state, writer_note = "failed", campaign.get("error")
            elif recorded == "running":
                writer_state = "gone"
                writer_note = (
                    "the campaign status still says running and nobody holds the write lock, so "
                    "the writer is gone. The next campaign process takes the lock and marks any "
                    "begun-and-unfinished trajectory incomplete. It is never resumed.")
            elif recorded == "finished":
                writer_state, writer_note = "finished", None
            else:
                writer_state, writer_note = "not_started", None
            rows = [self.trajectory(entry, writer_state=writer_state) for entry in queue]
            counts = dict(Counter(row["status"] for row in rows))
            phase = ("fitting" if queue and recorded else
                     "preparing_banks" if freeze and not banks else
                     "banks_ready" if freeze and banks else
                     "before_freeze")
            active = next((row["trajectory"] for row in rows if row["status"] == "running"), None)
            evidence = self.first_evidence([row["trajectory"] for row in rows])
            failed = sum(1 for row in rows if row["status"] == "failed")
            # The launch evidence is historical and stays true; the label is not.
            if not evidence["healthy"]:
                health_status = "not_established"
                health_detail = evidence["basis"]
            elif writer_state == "failed" or failed:
                health_status = "launch_verified_but_campaign_failed"
                health_detail = ("the first update and the first passed gate check are real and "
                                 "are kept. A later failure means this is not a statement about "
                                 "the campaign now.")
            elif writer_state == "alive":
                health_status = "healthy"
                health_detail = ("a live writer holds the lock and the launch evidence is on "
                                 "disk")
            elif writer_state == "gone":
                health_status = "launch_verified_but_interrupted"
                health_detail = ("nobody holds the write lock while the campaign status says "
                                 "running. The launch evidence is historical.")
            elif writer_state == "finished":
                health_status = "launch_verified_and_session_finished"
                health_detail = "the writing session ended; the trajectory statuses say what ran"
            else:
                health_status = "launch_verified"
                health_detail = ("the launch evidence is on disk and no claim is made about the "
                                 "campaign now")
            health = dict(evidence, status=health_status, detail=health_detail,
                          writer_state=writer_state,
                          scope=("launch evidence is historical: the first journalled update and "
                                 "the first measured check. It is never shown as current health "
                                 "on its own."))
            tables = self.document(self.DOCUMENTS["tables"]) or {}
            coverage = tables.get("coverage") or {}
            deltas = tables.get("matched_control_deltas") or {}
            verification = self.document(self.DOCUMENTS["verification"]) or {}
            warnings = []
            if not queue:
                warnings.append("No queue document yet: the campaign has not been launched. "
                                "Nothing here is waiting to run.")
            if writer_note:
                warnings.append(writer_note)
            if stalled and writer_state == "alive":
                warnings.append(
                    f"The campaign heartbeat is {heartbeat_age:.0f}s old while a live process "
                    "still holds the write lock. That is a progress signal, not a liveness one: a "
                    "gate check scores every fixed validation pair and can take a while.")
            if failed:
                warnings.append(f"{failed} trajectory status(es) record `failed`, which is the "
                                "machinery rather than a declared scientific outcome. The campaign "
                                "fail-stops on one.")
            if freeze is None:
                warnings.append("No training_spec_frozen marker: fitting refuses to start without "
                                "one, and the banks are generated only after it exists.")
            elif banks is None:
                warnings.append("The specification is frozen and the banks have not been "
                                "generated yet.")
            return clean({
                "present": True, "root": self.root.name, "generated_at": now,
                "campaign_id": campaign.get("campaign_id") or queue_document.get("campaign_id"),
                "phase": phase, "writer_state": writer_state,
                "recorded_status": recorded,
                "lock": lock,
                "heartbeat_stalled": bool(stalled),
                "heartbeat_age_seconds": heartbeat_age,
                "heartbeat": {"update": heartbeat.get("update"),
                              "trajectory": heartbeat.get("trajectory"),
                              "checks": heartbeat.get("checks"),
                              "exposures": heartbeat.get("exposures")},
                "queue_present": bool(queue), "total": len(rows),
                "counts": counts, "active": active, "trajectories": rows,
                "health": health,
                "frozen": None if not freeze else {
                    "commit": (freeze.get("git") or {}).get("commit"),
                    "frozen_at": freeze.get("frozen_at"),
                    "sources": len((freeze.get("source") or {}).get("sha256") or {}),
                    "inputs": freeze.get("input_count"),
                    "audit_decision": (freeze.get("audit") or {}).get("decision_outcome")},
                "banks": None if not banks else {
                    "count": banks.get("bank_count"),
                    "freeze_commit": (banks.get("freeze") or {}).get("commit"),
                    "overlap": {seed: block.get("monitor_draws_also_in_replay")
                                for seed, block in sorted((banks.get("overlap") or {}).items())}},
                "results": {
                    "reached_endpoint_rows": coverage.get("reached_endpoint_rows"),
                    "declared_endpoint_rows": coverage.get("declared_endpoint_rows"),
                    "matched_deltas": len(deltas.get("matched") or []),
                    "unavailable_deltas": len(deltas.get("unavailable") or [])},
                "verification": {"immutable": verification.get("immutable"),
                                 "artifacts_checked": verification.get("artifacts_checked")},
                "warnings": warnings, "errors": list(self.errors),
                "error": campaign.get("error")})


#: How long one next-flight job may leave its OWN artifacts unwritten before the
#: page stops calling it active. Wider than ``STALL_SECONDS`` on purpose: between
#: two calibration pilots the flight scores a full validation split and a parent
#: bank, and that work writes nothing inside any job directory while it runs. Ten
#: minutes clears that window, and the wording is "no recent writes" rather than
#: "stalled" because a quiet inter-pilot score is normal, not a hang.
QUIET_SECONDS = 600.0


def compact(value, depth=3, width=24):
    """A bounded copy of a small metadata block.

    The flight's identity, coefficient and category blocks are small by
    construction, but this panel refreshes every five seconds and must not turn
    into an unbounded transfer because a writer grew one of them. Truncation is
    reported in place rather than silently dropping the rest.
    """
    if isinstance(value, dict):
        if depth <= 0:
            return {"...": f"{len(value)} keys not expanded"}
        return {str(k): compact(v, depth - 1, width) for k, v in list(value.items())[:width]}
    if isinstance(value, (list, tuple)):
        if depth <= 0:
            return [f"{len(value)} items not expanded"]
        rows = [compact(v, depth - 1, width) for v in list(value)[:width]]
        if len(value) > width:
            rows.append(f"... {len(value) - width} more not expanded")
        return rows
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


def iso_instant(stamp):
    """One ISO-8601 instant as a POSIX timestamp, or ``None`` if it is not one."""
    try:
        moment = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def failure_is_current(failure, campaign):
    """Is a recorded supervisor failure the LATEST word, or a superseded one?

    The supervisor writes ``supervisor_failure.json`` on a crash and never
    removes it on a later success, so a resumed-and-finished flight kept
    reporting "Supervisor failed" forever once its lock was released. A failure
    older than the published campaign status has been superseded by that status;
    it stays on disk and stays visible as historical evidence, but it is no
    longer the current state. An unstamped or unparseable pair is treated as
    current, because "I cannot tell which came last" is not evidence that the
    crash was recovered.
    """
    if not failure:
        return False
    published = iso_instant((campaign or {}).get("finished_at"))
    crashed = iso_instant(failure.get("at"))
    if published is None or crashed is None:
        return True
    return crashed >= published


def iso_age(stamp):
    """Seconds since an ISO-8601 instant, or ``None`` if it is not one."""
    try:
        moment = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, datetime.now(timezone.utc).timestamp() - moment.timestamp())


def endpoint_names(value):
    """Reached endpoints as sorted strings, from a list, a mapping or nothing."""
    if isinstance(value, dict):
        items = list(value)
    elif isinstance(value, (list, tuple)):
        items = [v.get("update") if isinstance(v, dict) else v for v in value]
    else:
        items = []
    names = [str(v) for v in items if v is not None]
    try:
        return sorted(names, key=int)
    except ValueError:
        return sorted(names)


def identity_key(value):
    """A stable key for one work-budget or ledger identity, or ``None``."""
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return None
    return None if value is None else str(value)


def is_full_gate(row):
    """A full gate check, as opposed to a sentinel.

    A sentinel is a requested look that never stops a trajectory, and counting it
    as a gate would put a verdict on the page that nothing acts on. A record that
    does not say which it is is treated as full: dropping a real verdict is the
    worse error, and the recorded ``kind`` is carried through either way.
    """
    return row.get("kind") in (None, "full")


def nf_gate_point(row):
    """One recorded gate verdict: the decision and the threshold it was made against."""
    preservation = row.get("preservation") or {}
    events = (((preservation.get("drop") or {}).get("tails") or {}).get("events") or {})
    return {"record_kind": "gate_verdict", "kind": row.get("kind"), "update": row.get("update"),
            "D": row.get("D"), "passed": row.get("passed"), "reason": row.get("reason"),
            "stop_reason": row.get("stop_reason"),
            # The threshold comes from the verdict itself. A threshold read from
            # anywhere else can disagree with the one the decision was made under.
            "threshold_nats_per_sequence": row.get("threshold_nats_per_sequence"),
            "tenfold_rate": (events.get("tenfold") or {}).get("rate"),
            "hundredfold_rate": (events.get("hundredfold") or {}).get("rate")}


def nf_monitor_point(row):
    """Every monitor record kind, tagged and kept apart.

    The line after a full gate verdict is a *snapshot of the same check*. Merging
    the two doubles every check on the page and puts two points on the chart where
    one measurement happened, so snapshots are projected separately and never
    counted as verdicts.
    """
    kind = row.get("record_kind")
    if kind == "gate_verdict":
        return nf_gate_point(row)
    if isinstance(kind, str) and kind:
        return {"record_kind": kind, "kind": row.get("kind"), "update": row.get("update"),
                "passed": row.get("passed"), "D": row.get("D"), "reason": row.get("reason")}
    return None


def nf_update_point(row):
    """One completed optimizer update. Anything else in the journal is not one."""
    if row.get("record_kind") not in (None, "update"):
        return None
    result = {k: row.get(k) for k in
              ("update", "weighted_total", "task_loss", "preservation_loss", "raw_task_loss",
               "margin_mean", "gradient_norm", "learning_rate_used")}
    result["exposures"] = row.get("exposures") or None
    return result


def observe_lock(target):
    """Who holds ``target``, observed without creating it and without writing.

    The four answers :meth:`ReplayRun.lock_observation` gives -- ``missing``,
    ``held``, ``released``, ``inaccessible`` -- asked of a flight root. The lock is
    opened for update, tested and dropped immediately: no owner record is written
    and no lock file is created where there was none. "I could not tell" is never
    reported as "nobody is there".
    """
    if not target.is_file():
        return {"state": "missing",
                "reason": "no lock file exists; no flight process has taken this directory"}
    try:
        handle = target.open("r+b")
    except OSError as error:
        return {"state": "inaccessible", "reason": f"{type(error).__name__}: {error}"}
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        return {"state": "held", "reason": f"{type(error).__name__}: {error}"}
    finally:
        handle.close()
    return {"state": "released",
            "reason": ("a lock file exists and nobody holds it. The kernel releases it on a "
                       "forced kill, so this covers a clean exit and a killed owner alike")}


class NextFlightRun:
    """Read-only view of a next-flight campaign directory.

    Additive in the same way the audit and replay panels are: it shares the server
    and the security posture, imports nothing from the experiment package, opens no
    weights and no ``.pt`` file at all, and writes nothing in the run. A half-written
    or absent flight directory can do no more than make this panel say so.

    What it refuses to do, because each turns an unfinished flight into a finished-
    looking one:

    * **Infer a stage from silence.** The twelve stages are shown in their declared
      order and a missing record reads ``not_recorded`` -- never ``completed``.
    * **Call 500 updates a qualification.** A pilot that completed its declared
      updates is *measured*. Qualification is frozen in ``calibration_outcome.json``
      and nowhere else, so the coefficient-freeze card is absent until that document
      exists, however many pilots have finished.
    * **Call a missing queue an unlaunched flight.** ``queue.json`` and
      ``campaign_status.json`` are published on terminal supervisor status, so for
      the whole flight neither exists. Production is enumerated from the job
      directories actually on disk and labelled as recorded jobs, not as a full
      queued count.
    * **Read liveness from a PID or a heartbeat.** The OS lock is the liveness test,
      observed read-only. The heartbeat's background tick advances on a hung flight,
      so its file age and its ``last_work_callback_at`` are reported as two different
      facts, and work evidence comes from each job's own journals.
    * **Call the flight complete on a report alone.** Completion is the conjunction
      of ``report.json`` completion, ``verification.json`` passing, and the ``verify``
      stage recorded completed.
    """

    STAGES = ("recover", "preflight", "geometry", "mixtures", "profile", "calibrate",
              "freeze", "production", "audit", "couple", "report", "verify")
    #: Job groups that may be enumerated and selected. Nothing else is listed.
    GROUPS = ("calibration", "parents", "trajectories")
    #: ``calibration/budgets`` holds the per-pilot work-budget records, not a pilot.
    NOT_A_JOB = frozenset({"budgets"})
    #: A job's own artifacts. Freshness here -- not the campaign heartbeat -- is
    #: what this panel will accept as evidence that the job is being worked on.
    WORK_FILES = ("updates.jsonl", "monitor.jsonl", "endpoints.jsonl",
                  "trajectory_progress.json")
    BUDGETS = "calibration/budgets"
    #: Documents this view may read, by logical name. Nothing else is opened, so a
    #: crafted name cannot walk out of the run directory.
    DOCUMENTS = {
        "heartbeat": "heartbeat.json",
        "source": "source_snapshot.json",
        "forecast": "runtime_forecast.json",
        "ledger": "calibration_ledger.json",
        "outcome": "calibration_outcome.json",
        "protocol": "resolved_protocol.json",
        "queue": "queue.json",
        "campaign": "campaign_status.json",
        "supervisor_failure": "supervisor_failure.json",
        "report": "report.json",
        "verification": "verification.json",
    }
    LOCK_FILE = "campaign.lock"

    def __init__(self, root, *, quiet_seconds=QUIET_SECONDS):
        # Never resolved: the production root is a junction, and the containment
        # check below stays correct across a reparse point only if it is not.
        self.root = Path(root)
        self.quiet_seconds = float(quiet_seconds)
        self.lock = threading.Lock()
        self.errors = []
        self.selection = None
        self.update_journal = None
        self.monitor_journal = None

    # -- safe, bounded reads ----------------------------------------------
    def path(self, relative):
        parts = str(relative).replace("\\", "/").split("/")
        if any(part in ("", ".", "..") or ":" in part for part in parts):
            raise KeyError(relative)
        base = Path(os.path.normpath(os.path.abspath(str(self.root))))
        target = Path(os.path.normpath(os.path.join(str(base), *parts)))
        if base != target and base not in target.parents:
            raise KeyError(relative)
        return target

    def document(self, relative):
        try:
            target = self.path(relative)
        except KeyError:
            return None
        if not target.is_file():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            self.errors.append(f"{relative}: {type(error).__name__}: {error}")
            return None

    def record(self, relative):
        """A document this panel can read as a record, or ``None``.

        A JSON array where a record was expected is a half-written or foreign file,
        not a record. It reads as absent rather than raising on the first ``.get``.
        """
        value = self.document(relative)
        return value if isinstance(value, dict) else None

    def age(self, relative):
        try:
            target = self.path(relative)
        except KeyError:
            return None
        if not target.is_file():
            return None
        return max(0.0, datetime.now(timezone.utc).timestamp() - target.stat().st_mtime)

    def entries(self, group):
        """The job directories actually on disk under one group, sorted."""
        if group not in self.GROUPS:
            return []
        try:
            base = self.path(group)
        except KeyError:
            return []
        try:
            names = [item.name for item in base.iterdir()
                     if item.is_dir() and item.name not in self.NOT_A_JOB]
        except OSError:
            return []
        return sorted(names)

    def work_age(self, base, filenames=None):
        """Seconds since this job last wrote one of its OWN artifacts."""
        ages = [self.age(f"{base}/{name}") for name in
                (self.WORK_FILES if filenames is None else filenames)]
        ages = [value for value in ages if value is not None]
        return min(ages) if ages else None

    def lock_observation(self):
        try:
            target = self.path(self.LOCK_FILE)
        except KeyError:
            return {"state": "inaccessible", "reason": "the lock path is not inside the run root"}
        return observe_lock(target)

    # -- the panel ---------------------------------------------------------
    def stage_rows(self):
        rows = []
        for order, name in enumerate(self.STAGES):
            record = self.record(f"stages/{name}.json")
            if record is None:
                rows.append({"stage": name, "order": order, "status": "not_recorded",
                             "recorded_status": None, "reason": None, "identity": None,
                             "recorded_at": None, "wall_seconds": None,
                             "note": "no stage record has been written yet; this is not a success"})
                continue
            rows.append({"stage": name, "order": order,
                         "status": record.get("status") or "unknown",
                         "recorded_status": record.get("status"),
                         "reason": record.get("reason"),
                         "identity": compact(record.get("identity")),
                         "recorded_at": (record.get("recorded_at") or record.get("completed_at")
                                         or record.get("ended_at")),
                         "wall_seconds": record.get("wall_seconds"), "note": None})
        return rows

    def journal_counters(self, base):
        """Counters from the job's own journals, never from its progress file.

        ``trajectory_progress.json`` is rewritten at checks, so between two checks
        it is behind by a whole interval. The update journal gets a record as each
        update completes, and :func:`tail` returns only newline-terminated lines, so
        a half-written final record cannot be read as the latest state.
        """
        record, endpoints = None, []
        try:
            updates = self.path(f"{base}/updates.jsonl")
            reached = self.path(f"{base}/endpoints.jsonl")
        except KeyError:
            return {"updates": None, "exposures": None, "latest": None, "endpoints_reached": []}
        for line in reversed(tail(updates, limit=6)):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict) and candidate.get("record_kind") in (None, "update") \
                    and candidate.get("update") is not None:
                record = candidate
                break
        for line in tail(reached, limit=24):
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict) and candidate.get("record_kind") == "exposure_endpoint":
                endpoints.append(candidate.get("update"))
        return {"updates": (record or {}).get("update"),
                "exposures": (record or {}).get("exposures") or None,
                "latest": nf_update_point(record) if record else None,
                "endpoints_reached": endpoint_names(endpoints)}

    def last_full_gate(self, base):
        """The most recent full gate verdict for one job, or ``None``."""
        try:
            target = self.path(f"{base}/monitor.jsonl")
        except KeyError:
            return None
        for line in reversed(tail(target, limit=40)):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("record_kind") == "gate_verdict" \
                    and is_full_gate(row):
                return nf_gate_point(row)
        return None

    def job(self, group, name, *, writer_state, budgets=None):
        """One calibration pilot or production trajectory, read from its own files."""
        base = f"{group}/{name}"
        terminal = self.record(f"{base}/status.json")
        previous = self.record(f"{base}/status_u500.json")
        progress = self.record(f"{base}/trajectory_progress.json") or {}
        idle = self.work_age(base)
        counters = self.journal_counters(base)
        if terminal:
            status, note = terminal.get("status") or "unknown", None
        elif previous:
            # A continuation renames the finished leg's record and writes the same
            # directory again. That is work in flight, not a stop and not a failure.
            status = "continuation_in_progress"
            note = "Continuing the same pilot beyond its first 500 updates."
        elif idle is not None:
            status, note = "in_progress", None
        else:
            status, note = "not_started", "the directory exists with no journalled work yet"
        if terminal:
            activity = "finished"
        elif writer_state == "alive" and idle is None:
            activity = "preparing"
        elif writer_state == "alive" and idle is not None and idle <= self.quiet_seconds:
            activity = "writing"
        elif writer_state == "alive":
            # Not "stalled": the flight writes nothing in any job directory while it
            # scores a validation split between pilots.
            activity = "no_recent_writes"
        elif writer_state == "unknown":
            activity = "unknown"
        else:
            activity = "writer_absent"
        identity = (terminal or previous or {}).get("identity") or {}
        if group == "calibration" and not terminal:
            # The first leg has no status/identity document until it finishes.
            # Its budget records the coefficients before work begins. Extensions
            # have a separate budget but continue in the original pilot directory.
            matches = []
            for budget in self.budget_rows() if budgets is None else budgets:
                candidate = budget.get("identity") or {}
                entry_id = str(candidate.get("entry_id") or "")
                if entry_id.removeprefix("EXT:").replace(":", "_") == name:
                    matches.append((entry_id.startswith("EXT:"), candidate))
            if matches:
                identity = max(matches, key=lambda item: item[0])[1]
        return {"id": base, "group": group, "name": name, "status": status, "status_note": note,
                "terminal": bool(terminal), "activity": activity, "idle_seconds": idle,
                "stop_reason": (terminal or {}).get("stop_reason"),
                "declared_updates": (terminal or previous or {}).get("declared_updates"),
                "updates": (terminal.get("updates") if terminal else counters.get("updates")),
                "counters_from": ("its terminal status" if terminal
                                  else "its append-only journals"),
                "progress_file_update": None if terminal else progress.get("update"),
                "exposures": ((terminal or {}).get("exposures") if terminal
                              else counters.get("exposures")),
                "latest_update": counters.get("latest"),
                "endpoints_reached": endpoint_names(
                    (terminal or {}).get("endpoints_reached") if terminal
                    else counters.get("endpoints_reached")),
                "coefficients": compact(identity.get("coefficients")),
                "identity": compact(identity),
                "task": (terminal or {}).get("task"),
                "preservation": compact((terminal or {}).get("preservation")),
                "cost": compact((terminal or {}).get("cost")),
                "gate": self.last_full_gate(base),
                # 500 completed updates is a measurement, not a qualification.
                "qualified": None,
                "previous_leg": None if not previous else {
                    "status": previous.get("status"), "updates": previous.get("updates"),
                    "stop_reason": previous.get("stop_reason")}}

    def parent_rows(self):
        """Fresh-parent progress, at the granularity the writer actually records.

        ``stage1_history.json`` gets a record once per epoch; finer progress lives
        only inside ``stage1_state.pt``, which this reader does not open. The row
        says so rather than implying an update-level number it cannot have.
        """
        rows = []
        for name in self.entries("parents"):
            base = f"parents/{name}"
            history = self.document(f"{base}/stage1_history.json")
            records = history if isinstance(history, list) else (
                (history.get("epochs") or history.get("history") or [])
                if isinstance(history, dict) else [])
            records = [row for row in records if isinstance(row, dict)]
            selection = self.record(f"{base}/selection.json")
            progress = self.record(f"{base}/stage1_progress.json") or {}
            checkpoints = []
            try:
                checkpoints = sorted(item.name for item in self.path(base).iterdir()
                                     if item.is_file() and item.suffix == ".pt")[:20]
            except (KeyError, OSError):
                checkpoints = []
            rows.append({"parent": name, "epochs_recorded": len(records),
                         "last_epoch": compact(records[-1]) if records else None,
                         "recorded_epoch": progress.get("epoch"),
                         "selected": bool(selection),
                         "selection": compact(selection) if selection else None,
                         # Observed by filename. No checkpoint is opened.
                         "checkpoint_files": checkpoints,
                         "idle_seconds": self.work_age(base, (
                             "stage1_history.json", "stage1_progress.json", "selection.json",
                             "stage1_state.pt")),
                         "granularity": "Progress is recorded after each completed epoch."})
        return rows

    def budget_rows(self, limit=60):
        """The live per-pilot work-budget records. Small metadata only."""
        rows = []
        try:
            base = self.path(self.BUDGETS)
        except KeyError:
            return rows
        try:
            names = sorted(item.name for item in base.iterdir()
                           if item.is_file() and item.suffix == ".json")
        except OSError:
            return rows
        for name in names[:limit]:
            record = self.record(f"{self.BUDGETS}/{name}") or {}
            rows.append({"file": name, "identity": compact(record.get("identity")),
                         "key": identity_key(record.get("identity")),
                         "stem": name[:-len(".json")],
                         "measured_seconds": record.get("measured_seconds"),
                         "uncertainty_debit_seconds": record.get("uncertainty_debit_seconds"),
                         "pending": compact(record.get("pending")),
                         "limit_seconds": record.get("limit_seconds"),
                         "categories": compact(record.get("categories"))})
        return rows

    def calibration_cost(self, ledger, budgets):
        """What calibration has actually cost, without charging a pilot twice.

        The ledger is re-flushed after each pilot, so it lags the pilot in flight by
        a whole pilot. The live work-budget records carry that charge -- but they
        stay on disk after their pilot lands in the ledger, so adding all of them to
        the ledger total double-counts every completed pilot. Only budgets whose
        identity is not already a ledger entry are added, and the two totals are
        reported separately so the lag is visible rather than smoothed away.
        """
        entries = (ledger or {}).get("entries") or []
        recorded = {identity_key(((ledger or {}).get("overhead") or {}).get("identity"))}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for field in ("identity", "entry_id", "name", "pilot"):
                key = identity_key(entry.get(field))
                if key:
                    recorded.add(key)
            work = (entry.get("cost") or {}).get("gpu_work_units") or {}
            recorded.add(identity_key(work.get("identity")))
        recorded.discard(None)
        # Match the writer's identity, including its colon-delimited entry_id.
        # Budget filenames are slugged; they are not the ledger's entry IDs.
        live = [row for row in budgets
                if not (recorded & {key for key in (
                    row["key"], row["stem"],
                    identity_key((row.get("identity") or {}).get("entry_id"))) if key})]
        matched = len(budgets) - len(live)
        live_seconds = sum(number(row["measured_seconds"]) for row in live)
        live_debit = sum(number(row["uncertainty_debit_seconds"]) for row in live)
        persisted = (ledger or {}).get("measured_gpu_seconds")
        persisted_debit = (ledger or {}).get("uncertainty_debit_seconds")
        cap_hours = (ledger or {}).get("gpu_hour_cap")
        combined = (number(persisted) + live_seconds) if (ledger or budgets) else None
        combined_debit = number(persisted_debit) + live_debit
        remaining = (float(cap_hours) * 3600 - combined - combined_debit) if (
            isinstance(cap_hours, (int, float)) and combined is not None) else None
        return {"present": bool(ledger) or bool(budgets),
                "cap_gpu_hours": cap_hours,
                "ledger_measured_seconds": persisted,
                "ledger_uncertainty_debit_seconds": persisted_debit,
                "ledger_entries": len(entries),
                "overhead": compact((ledger or {}).get("overhead")),
                "live_budget_records": len(live), "budgets_already_in_ledger": matched,
                "live_measured_seconds": live_seconds if live else 0.0,
                "live_uncertainty_debit_seconds": live_debit if live else 0.0,
                "combined_measured_seconds": combined,
                "combined_uncertainty_debit_seconds": combined_debit,
                "remaining_seconds_under_cap": remaining,
                "basis": ("the ledger's persisted total plus the live per-pilot budget records "
                          "whose identity is not already a ledger entry. The ledger is flushed "
                          "after each pilot and lags the pilot in flight."),
                "budgets": budgets}

    def liveness(self):
        lock = self.lock_observation()
        heartbeat = self.record(self.DOCUMENTS["heartbeat"]) or {}
        campaign = self.record(self.DOCUMENTS["campaign"]) or {}
        failure = self.record(self.DOCUMENTS["supervisor_failure"])
        if lock["state"] == "held":
            writer_state = "alive"
        elif lock["state"] == "inaccessible":
            writer_state = "unknown"
        elif failure_is_current(failure, campaign) or campaign.get("status") == "failed":
            writer_state = "failed"
        elif campaign.get("status") in ("finished", "completed", "stopped", "incomplete"):
            writer_state = "finished"
        elif lock["state"] == "missing":
            writer_state = "not_launched"
        else:
            writer_state = "gone"
        return {"lock": lock, "writer_state": writer_state,
                "recorded_status": campaign.get("status"),
                "heartbeat": {"stage": heartbeat.get("stage"), "phase": heartbeat.get("phase"),
                              "pid": heartbeat.get("pid"),
                              "heartbeat_at": heartbeat.get("heartbeat_at"),
                              "liveness_tick": heartbeat.get("liveness_tick"),
                              "completed": compact(heartbeat.get("completed")),
                              "last_work_callback_at": heartbeat.get("last_work_callback_at")},
                "heartbeat_age_seconds": self.age(self.DOCUMENTS["heartbeat"]),
                # Two different facts. A background thread ticks the heartbeat on a
                # schedule, so its file keeps advancing on a hung flight; the work
                # callback is the only stamp that moves when work advances.
                "work_callback_age_seconds": iso_age(heartbeat.get("last_work_callback_at")),
                "heartbeat_note": ("the heartbeat is a background tick and advances whether or not "
                                   "work advances; it is a liveness hint, never work evidence"),
                "supervisor_failure": compact(failure) if failure else None,
                # Kept on display either way: a superseded crash is still part of
                # this run's history, and hiding it would lose the evidence that
                # the flight was restarted at all.
                "supervisor_failure_is_current": failure_is_current(failure, campaign)}

    def snapshot(self, select=None):
        with self.lock:
            self.errors = []
            now = datetime.now(timezone.utc).isoformat()
            if not self.root.is_dir():
                return clean({"present": False, "root": self.root.name, "generated_at": now,
                              "state": "no_run_directory", "stages": [], "calibration_jobs": [],
                              "parents": [], "production": {"jobs": []}, "warnings": [],
                              "errors": [], "complete": False, "detail": None,
                              "note": "no next-flight run directory yet"})
            live = self.liveness()
            writer_state = live["writer_state"]
            stages = self.stage_rows()
            by_stage = {row["stage"]: row for row in stages}
            source = self.record(self.DOCUMENTS["source"])
            forecast = self.record(self.DOCUMENTS["forecast"])
            outcome = self.record(self.DOCUMENTS["outcome"])
            ledger = self.record(self.DOCUMENTS["ledger"])
            report = self.record(self.DOCUMENTS["report"]) or {}
            verification = self.record(self.DOCUMENTS["verification"]) or {}
            queue = self.document(self.DOCUMENTS["queue"])
            budgets = self.budget_rows()
            pilots = [self.job("calibration", name, writer_state=writer_state, budgets=budgets)
                      for name in self.entries("calibration")]
            jobs = [self.job("trajectories", name, writer_state=writer_state)
                    for name in self.entries("trajectories")]
            parents = self.parent_rows()
            cost = self.calibration_cost(ledger, budgets)
            frozen = (outcome or {}).get("frozen")
            families = None
            if isinstance(frozen, dict):
                families = sorted(str(name) for name, value in frozen.items()
                                  if isinstance(value, dict) and value.get("entry_id")
                                  and isinstance(value.get("coefficients"), dict))
            elif isinstance(frozen, list):
                names = [row.get("family") or row.get("name") if isinstance(row, dict) else row
                         for row in frozen]
                families = sorted(str(name) for name in names if isinstance(name, (str, int)))
            total = forecast.get("total_hours") if forecast else None
            reserved = forecast.get("with_reserve_hours") if forecast else None
            active_rows = [row for row in pilots + jobs if row["activity"] == "writing"]
            active = min(active_rows, key=lambda row: row["idle_seconds"])["id"] if active_rows else None
            if active is None and writer_state == "alive":
                active = next((f"parents/{row['parent']}" for row in parents
                               if not row["selected"] and row["idle_seconds"] is not None
                               and row["idle_seconds"] <= self.quiet_seconds), None)
            # Completion is the conjunction. A report on disk is not a finished
            # flight, and neither is a verify stage that ran without passing.
            report_complete = (report.get("completion") or {}).get("complete")
            verified = verification.get("passed")
            verify_stage = by_stage.get("verify", {}).get("status")
            complete = bool(report_complete is True and verified is True
                            and verify_stage == "completed")
            warnings = []
            if writer_state == "not_launched":
                warnings.append("No lock file exists: no flight process has taken this directory.")
            elif writer_state == "gone":
                warnings.append(
                    "Nobody holds the write lock and no terminal status was published. The writer "
                    "is gone; what is displayed is the last state its artifacts recorded.")
            elif writer_state == "unknown":
                warnings.append(f"The write lock could not be read ({live['lock']['reason']}), so "
                                "liveness is unknown. Unknown is not reported as finished.")
            elif writer_state == "failed":
                warnings.append("A supervisor failure is recorded. This is the machinery, not a "
                                "declared scientific outcome.")
            for quiet in (row for row in pilots + jobs if row["activity"] == "no_recent_writes"):
                warnings.append(
                    f"{quiet['id']} has written none of its own artifacts for "
                    f"{quiet['idle_seconds']:.0f}s while a live process holds the lock. Scoring "
                    "between pilots is quiet by design; this is not confirmed progress.")
            for broken in (row for row in pilots + jobs if row["status"] == "failed"):
                warnings.append(f"{broken['id']} records `failed`, which is the machinery rather "
                                "than a declared scientific outcome.")
            detail = None if not select else self.detail(select, writer_state)
            return clean({
                "present": True, "root": self.root.name, "generated_at": now,
                "state": writer_state, "lock": live["lock"],
                "liveness": live, "quiet_seconds": self.quiet_seconds,
                "stages": stages, "stage_order": list(self.STAGES),
                "stages_completed": sum(1 for row in stages if row["status"] == "completed"),
                "stages_total": len(self.STAGES),
                "active_stage": live["heartbeat"]["stage"],
                "active_job": active,
                "source_freeze": None if not source else {
                    "snapshot_sha256": source.get("snapshot_sha256"),
                    "file_count": (source.get("file_count")
                                   if source.get("file_count") is not None
                                   else (len(source.get("files") or []) or None)),
                    "frozen_at": source.get("frozen_at"),
                    "git_state": compact(source.get("git_state"))},
                "protocol_frozen": self.document(self.DOCUMENTS["protocol"]) is not None,
                # Two documents, opposite cadences. The ledger is re-flushed after
                # every pilot; the outcome is written once, after the whole ladder.
                # The freeze is never inferred from the ledger.
                "coefficient_freeze": {
                    "present": outcome is not None, "families": families,
                    "frozen": compact(frozen), "has_ledger": bool((outcome or {}).get("ledger")),
                    "note": ("qualification is frozen only in calibration_outcome.json. A pilot "
                             "that completed its declared updates is measured, not qualified.")},
                "forecast": None if not forecast else {
                    "total_hours": total, "with_reserve_hours": reserved,
                    "reserve_hours": (reserved - total if isinstance(total, (int, float))
                                      and isinstance(reserved, (int, float)) else None),
                    "limitations": compact(forecast.get("limitations")),
                    "basis": "measured, with the declared reserve; the reserve is not a guarantee"},
                "calibration": dict(cost, jobs=pilots,
                                    counts=dict(Counter(row["status"] for row in pilots)),
                                    stage=by_stage.get("calibrate", {}).get("status")),
                "calibration_jobs": pilots,
                "parents": parents,
                "production": {
                    "jobs": jobs, "counts": dict(Counter(row["status"] for row in jobs)),
                    "recorded_jobs": len(jobs),
                    "queue_document_present": queue is not None,
                    "stage": by_stage.get("production", {}).get("status"),
                    # queue.json is published on terminal supervisor status, so its
                    # absence says nothing about whether the flight is running.
                    "note": ("these are the job directories recorded on disk, not a queued count. "
                             "queue.json is published only with a terminal supervisor status, and "
                             "no queue before the calibration freeze is normal.")},
                "terminal": {"campaign_status": live["recorded_status"],
                             "supervisor_failure": live["supervisor_failure"],
                             "report_completion": report_complete,
                             "verification_passed": verified,
                             "verify_stage": verify_stage},
                "complete": complete,
                "completion_basis": ("report.json completion, verification.json passed and a "
                                     "completed verify stage, all three"),
                "detail": detail,
                "warnings": warnings, "errors": list(self.errors)})

    def detail(self, select, writer_state=None):
        """Histories for ONE selected job, whitelisted against the directories on disk."""
        group, _, name = str(select).partition("/")
        if group not in self.GROUPS or not name or name not in self.entries(group):
            raise KeyError(select)
        base = f"{group}/{name}"
        self.path(base)
        if group == "parents":
            row = next((item for item in self.parent_rows() if item["parent"] == name), None)
            return dict(row or {"parent": name}, id=base, group=group,
                        updates_trace=[], gates=[], sentinels=[],
                        note="a fresh parent records one history entry per epoch")
        # One selection at a time: switching jobs drops the previous readers rather
        # than keeping every journal this session ever looked at in memory.
        if self.selection != base:
            self.selection = base
            self.update_journal = Journal(self.path(f"{base}/updates.jsonl"), nf_update_point)
            self.monitor_journal = Journal(self.path(f"{base}/monitor.jsonl"), nf_monitor_point)
        updates = list(self.update_journal.read())
        rows = list(self.monitor_journal.read())
        stride = max(1, math.ceil(max(0, len(updates) - 100) / 1200))
        plotted = updates[:100] + updates[100::stride]
        if updates and (not plotted or plotted[-1] != updates[-1]):
            plotted.append(updates[-1])
        gates = [row for row in rows
                 if row["record_kind"] == "gate_verdict" and is_full_gate(row)]
        sentinels = [row for row in rows
                     if row["record_kind"] == "sentinel_check" or (
                         row["record_kind"] == "gate_verdict" and not is_full_gate(row))]
        others = [row for row in rows
                  if row["record_kind"] not in ("gate_verdict", "sentinel_check")]
        if writer_state is None:
            writer_state = self.liveness()["writer_state"]
        job = self.job(group, name, writer_state=writer_state)
        return dict(job, id=base, group=group, updates_trace=plotted, update_count=len(updates),
                    plot_stride=stride, gates=gates, sentinels=sentinels,
                    other_monitor_records=dict(Counter(row["record_kind"] for row in others)),
                    threshold_nats_per_sequence=(gates[-1]["threshold_nats_per_sequence"]
                                                 if gates else None),
                    unreadable_update_lines=self.update_journal.bad_lines,
                    unreadable_monitor_lines=self.monitor_journal.bad_lines,
                    journal_resets=self.update_journal.resets + self.monitor_journal.resets,
                    # A complete final line without its newline is not consumed yet;
                    # it is retried on the next poll rather than half-read now.
                    pending_update_bytes=self.pending(f"{base}/updates.jsonl",
                                                      self.update_journal),
                    pending_monitor_bytes=self.pending(f"{base}/monitor.jsonl",
                                                       self.monitor_journal))

    def pending(self, relative, journal):
        try:
            target = self.path(relative)
            return max(0, target.stat().st_size - journal.offset)
        except (KeyError, OSError):
            return 0


def make_handler(campaign, audit=None, replay=None, flight=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in allowed or (origin and origin != f"http://{host}"):
                self.send_error(403)
                return
            url = urlsplit(self.path)
            try:
                if url.path == "/api/status":
                    body = json.dumps(campaign.snapshot(), allow_nan=False).encode()
                    mime = "application/json"
                elif url.path == "/api/run":
                    key = parse_qs(url.query).get("id", [""])[0]
                    body = json.dumps(campaign.detail(key), allow_nan=False).encode()
                    mime = "application/json"
                elif url.path == "/api/audit":
                    if audit is None:
                        self.send_error(404)
                        return
                    body = json.dumps(audit.snapshot(), allow_nan=False).encode()
                    mime = "application/json"
                elif url.path == "/api/replay":
                    if replay is None:
                        self.send_error(404)
                        return
                    body = json.dumps(replay.snapshot(), allow_nan=False).encode()
                    mime = "application/json"
                elif url.path == "/api/next-flight":
                    if flight is None:
                        self.send_error(404)
                        return
                    # A selection that is not one of the directories on disk raises
                    # KeyError below and is answered 404, so a crafted name can
                    # neither be read nor reported as an empty job.
                    select = parse_qs(url.query).get("select", [""])[0]
                    body = json.dumps(flight.snapshot(select or None),
                                      allow_nan=False).encode()
                    mime = "application/json"
                elif url.path in ("/", "/app.js", "/style.css"):
                    file = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[url.path]
                    body = (ASSETS / file).read_bytes()
                    mime = {"/": "text/html", "/app.js": "text/javascript", "/style.css": "text/css"}[url.path]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except KeyError:
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as exc:
                print(f"dashboard read failed: {type(exc).__name__}: {exc}", flush=True)
                self.send_error(500, "Snapshot unavailable; retrying on next refresh")

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/her2_guarded_20260918")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/her2_guarded_launch_20260918")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--audit", type=Path, default=None,
                        help="support-audit run directory to serve at /api/audit (read only); "
                             "omit to serve the campaign view alone")
    parser.add_argument("--replay", type=Path, default=None,
                        help="parent-replay campaign directory to serve at /api/replay (read "
                             "only); omit to leave the Replay tab empty")
    parser.add_argument("--stall-seconds", type=float, default=STALL_SECONDS,
                        help="mark a trajectory stalled after this long with no artifact write")
    parser.add_argument("--next-flight", type=Path, default=None,
                        help="next-flight campaign directory to serve at /api/next-flight (read "
                             "only); omit to leave the Next flight tab empty")
    parser.add_argument("--next-flight-quiet-seconds", type=float, default=QUIET_SECONDS,
                        help="report a next-flight job as having no recent writes after this "
                             "long without one of its own artifacts changing")
    args = parser.parse_args()
    campaign = Campaign(args.output, args.control, stall_seconds=args.stall_seconds)
    audit = None if args.audit is None else AuditRun(args.audit,
                                                     stall_seconds=args.stall_seconds)
    replay = None if args.replay is None else ReplayRun(args.replay,
                                                        stall_seconds=args.stall_seconds)
    flight = None if args.next_flight is None else NextFlightRun(
        args.next_flight, quiet_seconds=args.next_flight_quiet_seconds)
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(campaign, audit, replay, flight))
    print(f"HER2 dashboard: http://127.0.0.1:{args.port} (read only, PID {os.getpid()})"
          + (f" · audit {args.audit}" if audit is not None else "")
          + (f" · replay {args.replay}" if replay is not None else "")
          + (f" · next flight {args.next_flight}" if flight is not None else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
