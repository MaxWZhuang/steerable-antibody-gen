"""Watch a guarded campaign from outside it, and say something when it matters.

This is a **separate process** on purpose. The trainer cannot alert on its own
death, and neither can the supervisor: the case that most needs a message is
exactly the one where the thing that would send it is gone. So nothing here is
imported by the trainer, nothing here can stop or steer a campaign, and the only
files it writes are its own journal, its own state and its own heartbeat.

What it alerts on, and what it does not:

* **Urgent, immediately** -- a trajectory that stopped writing while the
  supervisor's heartbeat is still current; a trajectory that ``failed``; an
  artifact write that failed; the supervisor heartbeat going stale; the campaign
  status file disappearing after the campaign had been seen running; an error
  recorded by the supervisor. These are all "something is wrong or nothing is
  happening", and every one of them can waste hours of GPU time unobserved.
* **Digested, on an interval** -- a trajectory that stopped on the gate, a
  trajectory that completed, a stage that froze. A gate stop is a **declared
  scientific outcome**, not a fault: the recorded 2026-09-19 campaign produced 30
  of them, and paging 30 times for the protocol working as written is how an
  operator learns to ignore the channel. They are collected and sent as one
  message.

Alerting is **edge triggered on state, not on the event stream.** Each poll
computes the set of conditions currently true; a condition that is new fires
once, a condition that is unchanged fires nothing, and a condition that can
recover (a stall, a stale heartbeat) fires a matching all-clear and re-arms. That
is what keeps a five-second poll from producing a five-second alert.

Delivery is durable and at-least-once. Every event is appended, flushed and
fsynced to ``watchdog.jsonl`` **before** any send is attempted, so the local
record survives a transport that is down, misconfigured or never configured at
all. Remote sends that fail stay pending in ``watchdog_state.json`` and are
retried on later polls; an event is marked delivered only after a transport
returned success, so a watchdog killed mid-send re-sends rather than silently
dropping. After ``--max-attempts`` tries an event is abandoned with its last
error recorded, so one bad URL cannot grow the queue without bound.

**Nothing leaves the machine unless you pass ``--webhook``.** With no transport
configured the watchdog is a local journal and a console, which is already the
whole of the detection value; the webhook is how you find out from another
machine. Payloads carry run ids, statuses, GPU seconds and D values.

Who watches this? Nothing does, and pretending otherwise would be the same
mistake this file exists to fix. Two honest answers: it writes
``watchdog_heartbeat.json`` every poll, so anything else can check its age; and
``--once`` performs a single poll and exits, which lets Task Scheduler or cron
supervise it instead of a loop that can itself hang.

From the repository root::

    .venv/Scripts/python.exe scripts/her2_watchdog.py --webhook https://ntfy.sh/your-topic
    .venv/Scripts/python.exe scripts/her2_watchdog.py --once --dry-run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_SCHEMA = "her2-watchdog/1"

#: Fires now. Something is wrong, or nothing is happening.
URGENT = "urgent"
#: Collected and sent together. A declared outcome, not a fault.
DIGEST = "digest"

#: Conditions that can stop being true. Everything else is terminal: once a
#: trajectory has failed it does not un-fail, and re-arming it would re-alert on
#: the same fact for the rest of the campaign.
RECOVERABLE = ("heartbeat_dead", "campaign_status_missing", "campaign_error", "stalled:")


def load_dashboard():
    """Reuse the page's reader so "stalled" means one thing in this repository.

    A watchdog with its own idea of what counts as a running trajectory is a
    second definition to keep in step with the first, and the two would drift on
    the first change to either. This is the same import the dashboard's tests
    use; the module is a script rather than a package module.
    """
    path = Path(__file__).with_name("her2_live_dashboard.py")
    spec = importlib.util.spec_from_file_location("her2_live_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def duration(seconds):
    if not isinstance(seconds, (int, float)):
        return "unknown"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"


def recoverable(key):
    return any(key == name or key.startswith(name) for name in RECOVERABLE)


def conditions(snapshot, *, seen_running):
    """Every condition currently true, keyed so it dedupes across polls.

    Returns ``{key: (severity, title, body)}``. The key is what makes an alert
    fire once rather than once per poll, so it must name the condition and its
    subject and nothing that changes between polls -- no timestamp, no elapsed
    second, no metric value.
    """
    found = {}
    status = snapshot.get("status")
    if status == "stale":
        found["heartbeat_dead"] = (
            URGENT, "Supervisor heartbeat is stale",
            f"No campaign heartbeat for {duration(snapshot.get('heartbeat_age_seconds'))}. "
            f"The last saved records are still on disk; continued execution is unconfirmed.")
    elif status == "unknown" and seen_running:
        found["campaign_status_missing"] = (
            URGENT, "Campaign status file is gone",
            "This campaign was running and its status file can no longer be read.")
    if snapshot.get("error"):
        error = snapshot["error"]
        found["campaign_error"] = (
            URGENT, "Supervisor recorded an error",
            error if isinstance(error, str) else json.dumps(error))
    recorded = snapshot.get("recorded_status")
    if recorded in ("failed", "aborted"):
        found[f"campaign_{recorded}"] = (
            URGENT, f"Campaign {recorded}", f"The supervisor recorded status {recorded!r}.")
    elif recorded == "completed":
        found["campaign_completed"] = (
            DIGEST, "Campaign completed",
            f"{snapshot.get('counts', {}).get('completed', 0)} completed, "
            f"{snapshot.get('counts', {}).get('stopped', 0)} stopped, "
            f"of {snapshot.get('total_runs')} trajectories.")

    for run in snapshot.get("runs", []):
        run_id, state = run["id"], run["status"]
        where = (f"update {run.get('updates')}, "
                 f"{(run.get('training_gpu_seconds') or 0):.1f} training GPU s")
        if state == "stalled":
            found[f"stalled:{run_id}"] = (
                URGENT, f"No progress: {run_id}",
                f"Nothing written for {duration(run.get('idle_seconds'))} at {where}, while the "
                f"campaign heartbeat is current. The supervisor is alive; this trajectory's "
                f"progress is not confirmed.")
        elif state == "failed":
            failure = run.get("failure") or {}
            found[f"failed:{run_id}"] = (
                URGENT, f"Failed: {run_id}",
                f"{failure.get('type') or 'Failure'} at {where}. "
                f"{failure.get('message') or ''}\n"
                f"This is not a gate stop and does not advance a stage.")
        elif state == "stopped":
            gate = run.get("gate") or {}
            # Hoisted out of the f-string: a nested f-string reusing the same
            # quote character only parses on 3.12+ (PEP 701), and this repo runs
            # 3.11, where it is a SyntaxError that aborts pytest collection for
            # the whole suite rather than just this module.
            gate_d = gate.get("D")
            gate_detail = f", D={gate_d:.3f}" if isinstance(gate_d, (int, float)) else ""
            found[f"stopped:{run_id}"] = (
                DIGEST, f"Gate stop: {run_id}",
                f"{run.get('stop_reason') or 'gate stop'} at {where}{gate_detail}.")
        elif state == "completed":
            found[f"completed:{run_id}"] = (
                DIGEST, f"Completed: {run_id}",
                f"Reached {run.get('budgets_reached')} at {where}.")
        # Reported apart from the run's status: a save can fail at a check whose
        # verdict passed, and the trajectory's own status does not say so.
        for error in run.get("artifact_errors") or []:
            found[f"artifact:{run_id}:{error.get('check')}"] = (
                URGENT, f"Artifact write failed: {run_id}",
                f"Check {error.get('check')}: {error.get('save_failed')}\n"
                f"{error.get('consequence') or ''}")

    for stage in snapshot.get("stages", []):
        if stage.get("frozen"):
            found[f"stage_frozen:{stage['stage']}"] = (
                DIGEST, f"Stage {stage['stage']} selection frozen",
                f"{stage.get('counts', {}).get('completed', 0)} completed, "
                f"{stage.get('counts', {}).get('stopped', 0)} stopped, of {stage.get('total')}.")
    return found


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------

def webhook_body(event, fmt):
    """The bytes and content type for one event, per service convention."""
    text = f"{event['title']}\n{event['body']}".strip()
    if fmt == "text":                       # ntfy and anything that takes a plain POST
        return text.encode("utf-8"), "text/plain; charset=utf-8"
    if fmt == "slack":
        return json.dumps({"text": text}).encode("utf-8"), "application/json"
    if fmt == "discord":
        return json.dumps({"content": text[:1900]}).encode("utf-8"), "application/json"
    return json.dumps(event).encode("utf-8"), "application/json"


class Webhook:
    """A POST, with the stdlib. Failure is returned, never raised."""

    def __init__(self, url, fmt="text", timeout=10.0):
        self.url, self.fmt, self.timeout = url, fmt, float(timeout)

    def send(self, event):
        data, content_type = webhook_body(event, self.fmt)
        headers = {"Content-Type": content_type}
        if self.fmt == "text":              # ntfy reads these; other services ignore them
            headers["Title"] = event["title"][:120]
            headers["Priority"] = "high" if event["severity"] == URGENT else "default"
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if 200 <= response.status < 300:
                    return True, None
                return False, f"HTTP {response.status}"
        except (urllib.error.URLError, OSError, ValueError) as error:
            return False, f"{type(error).__name__}: {error}"


class Apprise:
    """Optional. One dependency, about a hundred services; absent is not an error."""

    def __init__(self, targets):
        import apprise                                   # noqa: PLC0415 - optional at runtime
        self.client = apprise.Apprise()
        for target in targets:
            if not self.client.add(target):
                raise ValueError(f"apprise rejected the target {target!r}")

    def send(self, event):
        try:
            ok = self.client.notify(title=event["title"], body=event["body"])
            return bool(ok), None if ok else "apprise reported no successful delivery"
        except Exception as error:                        # noqa: BLE001 - reported, not raised
            return False, f"{type(error).__name__}: {error}"


class Console:
    """Always on. Running this in a terminal has to be useful with no setup."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout

    def send(self, event):
        mark = "!!" if event["severity"] == URGENT else "--"
        print(f"{mark} [{event['at']}] {event['title']}\n   "
              + event["body"].replace("\n", "\n   "), file=self.stream, flush=True)
        return True, None


# ---------------------------------------------------------------------------
# durable state
# ---------------------------------------------------------------------------

def save_json(path, value):
    """Atomic, like every other artifact here: a torn state file is a silent reset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fresh_state():
    return {"schema_version": WATCHDOG_SCHEMA, "active": {}, "pending": [], "digest": [],
            "seen_running": False, "digest_opened_at": None, "last_digest_at": None,
            "events": 0, "abandoned": []}


class Watchdog:
    """Edge-triggered conditions, a durable outbox, and a digest for the expected.

    ``active`` is the set of conditions that were true at the last poll; it is
    what makes a repeating condition fire once. ``pending`` is the outbox: an
    event leaves it only when a transport reported success, so the failure mode
    of a dead webhook is a retry, not a lost alert.
    """

    def __init__(self, directory, *, transports=(), digest_seconds=1800.0,
                 max_attempts=10, digest_max=25):
        self.directory = Path(directory)
        self.state_path = self.directory / "watchdog_state.json"
        self.journal_path = self.directory / "watchdog.jsonl"
        self.heartbeat_path = self.directory / "watchdog_heartbeat.json"
        self.transports = list(transports)
        self.digest_seconds = float(digest_seconds)
        self.max_attempts = int(max_attempts)
        self.digest_max = int(digest_max)
        self.state = self._load()

    def _load(self):
        try:
            stored = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return fresh_state()
        if stored.get("schema_version") != WATCHDOG_SCHEMA:
            return fresh_state()
        state = fresh_state()
        state.update(stored)
        return state

    def journal(self, event):
        """The durable record, written before any send is attempted.

        A transport that is down, wrong or never configured must not be able to
        lose the fact that the condition occurred.
        """
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def raise_events(self, snapshot):
        """Compare the conditions now against the conditions last seen."""
        if snapshot.get("status") == "running":
            self.state["seen_running"] = True
        current = conditions(snapshot, seen_running=self.state["seen_running"])
        active, events = self.state["active"], []
        for key, (severity, title, body) in sorted(current.items()):
            if key in active:
                continue
            active[key] = {"since": now_iso(), "severity": severity}
            events.append(self.event(key, severity, title, body, "raised"))
        for key in sorted(active):
            if key in current or not recoverable(key):
                continue
            # It can be true again later, so it is removed rather than remembered:
            # a second stall is a second thing worth being told about.
            record = active.pop(key)
            # An all-clear matches the severity of the alarm it cancels: if being
            # woken for this was urgent, being told to stand down is too.
            events.append(self.event(key, record["severity"], f"Recovered: {key}",
                                     f"This condition is no longer true. It was raised at "
                                     f"{record['since']}.", "resolved"))
        return events

    def event(self, key, severity, title, body, transition):
        self.state["events"] += 1
        return {"schema_version": WATCHDOG_SCHEMA, "key": key, "severity": severity,
                "transition": transition, "title": title, "body": body, "at": now_iso(),
                "campaign": str(self.directory), "sequence": self.state["events"],
                "attempts": 0}

    def deliver(self, events, *, force_digest=False):
        """Journal everything, send the urgent, collect the expected.

        Ordering matters: the journal write happens first and unconditionally, so
        the local evidence of a condition does not depend on a network.
        """
        for event in events:
            self.journal(event)
        self.state["pending"].extend(e for e in events if e["severity"] == URGENT)
        queued = [e for e in events if e["severity"] == DIGEST]
        if queued and not self.state["digest"]:
            # The interval runs from when the queue OPENED, not from the last send.
            # Timing it from the last send makes the first digest of a campaign due
            # immediately, which batches nothing at all.
            self.state["digest_opened_at"] = now_iso()
        self.state["digest"].extend(queued)
        if self.digest_due(force=force_digest):
            summary = self.build_digest()
            if summary is not None:
                self.journal(summary)
                self.state["pending"].append(summary)
        self.flush()
        return events

    def digest_due(self, *, force=False):
        queued = self.state["digest"]
        if not queued:
            return False
        if force or len(queued) >= self.digest_max:
            return True
        opened = self.state.get("digest_opened_at")
        if opened is None:
            return False
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(opened)).total_seconds()
        except ValueError:
            return True
        return age >= self.digest_seconds

    def build_digest(self):
        """One message for everything expected, grouped by what kind of thing it is."""
        queued = self.state["digest"]
        if not queued:
            return None
        groups = {}
        for event in queued:
            groups.setdefault(event["key"].split(":", 1)[0], []).append(event)
        lines = []
        for kind in sorted(groups):
            items = groups[kind]
            lines.append(f"{len(items)} x {kind.replace('_', ' ')}")
            lines.extend(f"  {e['title']} - {e['body'].splitlines()[0]}" for e in items)
        self.state["digest"] = []
        self.state["digest_opened_at"] = None
        self.state["last_digest_at"] = now_iso()
        return self.event("digest", DIGEST, f"Campaign digest ({len(queued)} events)",
                          "\n".join(lines), "digest")

    def flush(self):
        """Retry the outbox. Delivered means a transport said so, not that we tried."""
        remaining = []
        for event in self.state["pending"]:
            event["attempts"] = event.get("attempts", 0) + 1
            errors = []
            for transport in self.transports:
                ok, error = transport.send(event)
                if not ok:
                    errors.append(f"{type(transport).__name__}: {error}")
            if not errors:
                continue
            if event["attempts"] >= self.max_attempts:
                # One unreachable URL must not grow the queue forever. The event
                # stays in the journal; only the retrying stops, and it is recorded.
                self.state["abandoned"].append(
                    dict(event, abandoned_at=now_iso(), last_errors=errors))
                self.journal(dict(event, transition="abandoned", last_errors=errors))
                continue
            event["last_errors"] = errors
            remaining.append(event)
        self.state["pending"] = remaining

    def tick(self, snapshot, *, force_digest=False):
        events = self.raise_events(snapshot)
        self.deliver(events, force_digest=force_digest)
        save_json(self.state_path, self.state)
        save_json(self.heartbeat_path, {
            "schema_version": WATCHDOG_SCHEMA, "updated_at": now_iso(), "pid": os.getpid(),
            "campaign_status": snapshot.get("status"),
            "active_conditions": sorted(self.state["active"]),
            "pending_deliveries": len(self.state["pending"]),
            "abandoned_deliveries": len(self.state["abandoned"]),
            "queued_for_digest": len(self.state["digest"]),
            "note": ("nothing watches this watchdog. Check this file's age to know it is "
                     "alive, or run it under --once from a scheduler.")})
        return events


def build_transports(args):
    transports = []
    if not args.quiet:
        transports.append(Console())
    if args.webhook and not args.dry_run:
        transports.append(Webhook(args.webhook, fmt=args.webhook_format, timeout=args.timeout))
    if args.apprise and not args.dry_run:
        transports.append(Apprise(args.apprise))
    return transports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/her2_guarded_20260918")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/her2_guarded_launch_20260918")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="where the journal, state and heartbeat go (default: --output)")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true",
                        help="poll once and exit, for a scheduler to supervise instead of a loop")
    parser.add_argument("--stall-seconds", type=float, default=None,
                        help="override the dashboard's stall bound (default 180)")
    parser.add_argument("--digest-seconds", type=float, default=1800.0,
                        help="how often declared outcomes are sent as one message")
    parser.add_argument("--digest-max", type=int, default=25,
                        help="send the digest early once this many events are queued")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--webhook", default=None,
                        help="POST alerts here. Without it nothing leaves this machine.")
    parser.add_argument("--webhook-format", default="text",
                        choices=("text", "slack", "discord", "json"))
    parser.add_argument("--apprise", action="append", default=None,
                        help="apprise target URL, repeatable; needs the optional apprise package")
    parser.add_argument("--dry-run", action="store_true",
                        help="detect and journal, but send to no remote transport")
    parser.add_argument("--quiet", action="store_true", help="no console transport")
    args = parser.parse_args(argv)

    dashboard = load_dashboard()
    stall = args.stall_seconds if args.stall_seconds is not None else dashboard.STALL_SECONDS
    campaign = dashboard.Campaign(args.output, args.control, stall_seconds=stall)
    # The watchdog alerts on artifacts, not on telemetry, and a subprocess per
    # poll on the training box buys nothing here.
    campaign.gpu_status = lambda: None
    watchdog = Watchdog(args.state_dir or args.output, transports=build_transports(args),
                        digest_seconds=args.digest_seconds, max_attempts=args.max_attempts,
                        digest_max=args.digest_max)
    print(f"HER2 watchdog: watching {args.output} (read only, PID {os.getpid()})", flush=True)
    if not args.webhook and not args.apprise:
        print("  no remote transport configured; journalling locally only", flush=True)
    try:
        while True:
            try:
                snapshot = campaign.snapshot()
            except Exception as error:                    # noqa: BLE001 - a poll may fail
                print(f"  snapshot failed: {type(error).__name__}: {error}", flush=True)
            else:
                terminal = snapshot.get("recorded_status") in ("completed", "failed", "aborted")
                watchdog.tick(snapshot, force_digest=args.once or terminal)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
