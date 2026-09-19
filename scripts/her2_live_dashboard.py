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
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("her2_dashboard")


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
    if row.get("record_kind") != "gate_verdict":
        return None
    keys = ("D", "check", "update", "passed", "training_gpu_seconds", "pairs",
            "monitor_gpu_seconds", "monitor_wall_seconds", "reason",
            "mean_current_chosen_nll_per_residue", "mean_implicit_margin", "pair_accuracy")
    result = {k: row.get(k) for k in keys}
    result["quantiles"] = (row.get("chosen_drop") or {}).get("quantiles", {})
    return result


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
    """Incremental reader. An unfinished final line is retried on the next poll."""
    def __init__(self, path, project):
        self.path, self.project = Path(path), project
        self.offset = 0
        self.rows = []
        self.bad_lines = 0
        self.stamp = None

    def read(self):
        try:
            stat = self.path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            if stamp == self.stamp:
                return self.rows
            if stat.st_size < self.offset:
                self.offset, self.rows, self.bad_lines = 0, [], 0
            with self.path.open("rb") as handle:
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
            self.stamp = stamp
        except FileNotFoundError:
            # A removed artifact must not keep masquerading as current evidence.
            self.offset, self.rows, self.stamp = 0, [], None
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
    def __init__(self, output, control):
        self.output, self.control = Path(output), Path(control)
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
            self.journals[path] = Journal(path, gate_point)
        journal = self.journals[path]
        rows = journal.read()
        if journal.bad_lines:
            self.warnings.append(f"{directory.name}: {journal.bad_lines} unreadable monitoring lines.")
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
        gate = rows[-1] if rows else None
        latest = latest_update(directory / "updates.jsonl")
        state = trajectory.get("status")
        if not state:
            if progress.get("stopped") or (gate and gate.get("passed") is False):
                state = "stopped"
            elif ledger.get("status") == "failed":
                state = "failed"
            elif directory.exists():
                state = "running" if campaign_status == "running" else "interrupted"
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
        # The journals contain checks only; parent reference scoring is a separate cost.
        monitoring = sum(number(r.get("monitor_gpu_seconds")) for r in rows)
        parent_gpu = number(reference.get("gpu_seconds"))
        excluded = summary.get("excluded_costs") or {}
        if excluded.get("parent_validation_reference_reused_from_disk"):
            parent_gpu = 0.0
        updated_paths = [directory / n for n in ("updates.jsonl", "monitor.jsonl", "run.json")]
        modified = max((p.stat().st_mtime for p in updated_paths if p.exists()), default=None)
        return dict(id=run_id, stage=arm["stage"], arm=arm["arm_id"],
                    objective=arm["objective"], coefficients=arm.get("coefficients", {}), seed=seed,
                    status=state, updates=max(number(progress.get("updates")), number(latest.get("update"))),
                    training_gpu_seconds=training, monitor_gpu_seconds=monitoring + parent_gpu,
                    monitor_checks_gpu_seconds=monitoring, parent_reference_gpu_seconds=parent_gpu,
                    checks=len(rows), gate=gate, budgets_reached=reached,
                    exposures=latest.get("exposures") or progress.get("exposures"),
                    stop_reason=progress.get("stop_reason") or trajectory.get("status_note"),
                    last_passing=progress.get("last_passing"), latest_update=update_point(latest),
                    updated_at=modified,
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
            return clean(dict(id=run_id, gates=self.monitors(directory), updates=plotted,
                              update_count=len(updates), plot_stride=stride,
                              unreadable_update_lines=self.update_journal.bad_lines))


def make_handler(campaign):
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
    args = parser.parse_args()
    campaign = Campaign(args.output, args.control)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(campaign))
    print(f"HER2 dashboard: http://127.0.0.1:{args.port} (read only, PID {__import__('os').getpid()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
