# HER2 live dashboard

From the repository root:

```powershell
.venv/Scripts/python.exe scripts/her2_live_dashboard.py
```

Open **http://127.0.0.1:8765** on the training machine. The server uses the Python
standard library and binds only to loopback. It has no training controls and does
not import the model, load checkpoints, or write campaign artifacts. Closing the
browser does not stop either the dashboard server or the separate trainer.

The default campaign and supervisor directories are the guarded 20260918 run.
Use `--output`, `--control`, and `--port` to watch another compatible campaign.
If the computer or server restarts, run the command again. Starting the dashboard
does not start or resume training.

The page refreshes every five seconds. Select a trajectory, click a ledger row,
or keep **Follow active run** enabled. The metric selector switches the training
trace. **Show queued** reveals every declared arm; the stage filter narrows the
ledger. Hover chart points for exact recorded values.

Interpretation:

- Training GPU seconds include the runner's charged reference scoring. Monitoring
  GPU time is the sum of recorded checks and separately recorded parent scoring.
  Wall time comes from the supervisor and also includes evaluation and I/O.
  Work still in progress can appear only after its measurement is persisted.
- A completed trajectory and a likelihood-stopped trajectory are counted separately.
  Reached endpoint chips come from saved budget records, never from elapsed time.
  Reused SFT controls are not counted or charged again in later stages.
- Fixed-validation chosen-drop quantiles and mean are separate from training-batch
  traces. Pair-ranking accuracy is not AP, affinity, or generation eligibility.
  Passing the likelihood gate does not establish diversity or improvement.
- Every likelihood check is plotted. Long training traces show the first 100
  updates and an explicitly labelled regular subsample thereafter, without
  smoothing; the full history remains in the runner's journals.
- A supervisor heartbeat older than 30 seconds is marked stale. Connection errors
  leave the last displayed data visible with an explicit warning.
- A live heartbeat says the *supervisor* is alive and nothing more. A trajectory
  whose own artifacts have gone unwritten for longer than `--stall-seconds`
  (default 180) reads **No progress**, with its idle time, and is not offered as
  the active run. The bound clears a gate check plus a checkpoint write and the
  pre-first-update parent scoring; raise it if a slower device makes it fire.
- A **gate stop and an artifact failure are different outcomes** and are shown
  apart. A run that ended because a save failed carries a red panel naming the
  exception, where it happened and its recorded consequence, and its gate verdict
  is displayed unchanged beside it -- a passing check whose save failed reads as
  exactly that, not as a likelihood breach. Neither advances a stage.
- Rolling checkpoint I/O is reported: one full model serialization and hash per
  passing check. The completed 2026-09-19 campaign did 3,987 of them.
- The chosen-drop **fractions** sit beside the quantiles: how much of the
  population is below the parent, has lost more than one or five nats, and is
  below a uniform draw. The gate stops on the mean, and a mean cannot separate a
  small uniform shift from a small tail collapse. They are reported and decide
  nothing. Checks written before the gate journalled `chosen_drop_fractions`
  show "Not recorded for this check" rather than a zero.

The reader tolerates an unfinished last JSONL line and reads appended monitoring
records incrementally. It keeps at most one training-history reader, so switching
runs does not load every trajectory's update history into memory.

A journal that was **replaced rather than appended to** is detected and its
retained rows are discarded, with a warning: resuming from a byte offset is sound
only for the same file, and a replacement can come back at or above the old
offset. The reader matches device, inode, size and mtime *and* re-checks the bytes
it already consumed, so an in-place rewrite that lands on the same inode is caught
too. The one case it cannot see is a replacement that reproduces all four
identity fields exactly; nothing short of re-reading the whole file each poll
would.

## Next flight

```powershell
.venv/Scripts/python.exe scripts/her2_live_dashboard.py --port 8766 `
  --audit outputs/her2_support_audit_20260919_r2 `
  --replay outputs/her2_parent_replay_20260920 `
  --next-flight outputs/her2_next_flight_20260922/production
```

Without `--next-flight` the tab says **Not configured** and the other three views are
unchanged. The reader uses the standard library only, imports nothing from the
experiment package, opens no `.pt` file — not even a parent checkpoint, which is
observed by filename — and writes nothing in the run. The four tabs are deep
linkable: `#campaign`, `#audit`, `#replay`, `#next-flight`. A URL without a
fragment still opens the campaign view.

What the tab claims, and what it refuses to:

- **Twelve stages in their declared order** (`recover` → `verify`). A stage with no
  record reads **Not recorded**, never *completed*. `partial`, `not_run`,
  `blocked_readiness` and `blocked_feasibility` are four different answers and are
  shown as themselves.
- **Liveness is the OS write lock**, observed read-only: `held`, `released`,
  `missing` and `inaccessible` are four different answers, and *unknown* is never
  rendered as finished or as dead. The heartbeat is reported beside it as what it
  is — a background tick that keeps advancing whether or not work does — with
  `last_work_callback_at` as a separate, weaker-to-fake fact.
- **Work evidence is each job's own journals.** A job that has not written one of
  its own artifacts for longer than `--next-flight-quiet-seconds` (default 600)
  reads **No recent writes**, with the idle time. That is deliberately not called a
  stall: between two pilots the flight scores a full validation split and a bank,
  which writes nothing in any job directory, and 600 s clears that quiet window.
- **A missing queue is normal.** `queue.json` and `campaign_status.json` are
  published only with a terminal supervisor status, so for the whole flight neither
  exists. The production table is enumerated from the job directories on disk and
  is labelled *recorded jobs* — never a queued count, and never "unlaunched".
- **500 updates is not a qualification.** A pilot that finished its declared
  updates is *measured*. Qualification is frozen once, in `calibration_outcome.json`,
  and the coefficient-freeze card stays **Not frozen** until that document exists
  however many pilots have landed. It is a separate card from the source freeze,
  which says which code closure is running.
- **Five pilot states stay apart:** `completed`, `stopped_by_gate` (a declared
  scientific outcome), `incomplete`, `failed` (the machinery), and a directory
  holding `status_u500.json` with no current `status.json` — a **continuation in
  flight**, reusing the same directory.
- **Calibration cost is the ledger plus what the ledger does not have yet.** The
  ledger is flushed after each pilot and lags the pilot in flight; the live
  per-pilot budget records carry that charge. Budget identities are matched to
  ledger entry IDs and nested work-unit identities, including parent-baseline
  overhead. Already charged work is excluded, and uncertainty debits also reduce
  the remaining cap. An extension has its own budget identity.
- **Selected-job detail** plots the journalled `weighted_total` / `task_loss` /
  `preservation_loss` trace against the recorded update, and full-gate `D` against
  the update with the verdict's **own** recorded threshold. The snapshot line that
  follows a check is that same check and is never drawn twice; sentinel looks are
  counted apart, because nothing stops on them. A missing value is `null` and
  renders "—", never a zero.
- **Complete needs all three:** the report's completion flag, an independent
  verification that passed, and a `verify` stage recorded completed. A live PID, a
  fresh heartbeat and a table of finished pilots are none of them.

Only the selected job's journals are read, incrementally and one selection at a
time, with the same replaced-journal detection the campaign view uses. A selection
that is not one of the directories on disk is refused (404) rather than shown as
an empty job.

The inspector follows the active job by default. Selecting an earlier job pauses
following; **Follow active** resumes it. A selection made during a refresh takes
precedence over the earlier response. The `#next-flight` link survives reloads.

## Being told, rather than looking

The page shows you what is happening while you are looking at it. `scripts/her2_watchdog.py`
is the companion that tells you when you are not:

```powershell
.venv/Scripts/python.exe scripts/her2_watchdog.py --webhook https://ntfy.sh/your-topic
.venv/Scripts/python.exe scripts/her2_watchdog.py --once        # for Task Scheduler
```

It is a **separate process** because the case that most needs a message is the one
where the trainer and its supervisor are both gone. It imports the page's reader,
so "stalled" means the same thing in both; it writes only its own journal, state
and heartbeat; and it cannot stop or steer a campaign.

- **Paged immediately:** a trajectory that stopped writing while the heartbeat is
  current, a `failed` trajectory, a failed artifact write, a stale heartbeat, a
  status file that disappeared, a supervisor error.
- **Digested on an interval** (`--digest-seconds`, default 30 min): gate stops,
  completions, stage freezes. A gate stop is the protocol working as written --
  the recorded campaign produced 30 -- and paging for each is how an operator
  learns to ignore the channel.
- **Edge triggered.** A standing condition alerts once. One that can recover
  sends a matching all-clear and re-arms, so a second stall is a second alert.
- **At-least-once delivery.** Every event is fsynced to `watchdog.jsonl` *before*
  any send. A send that fails stays pending in `watchdog_state.json` and retries;
  an event is delivered only when a transport said so, so a watchdog killed
  mid-send re-sends rather than dropping. After `--max-attempts` it is abandoned
  with its error recorded, so one bad URL cannot grow the queue without bound.
- **Nothing leaves the machine without `--webhook`** (or `--apprise`, which needs
  that optional package). With neither, it is a local journal and a console.
  Payloads carry run ids, statuses, GPU seconds and D values.

Nothing watches the watchdog. It writes `watchdog_heartbeat.json` every poll so
its age can be checked, and `--once` polls once and exits so a scheduler
supervises it instead of a loop that can itself hang. The dashboard does not yet
display that heartbeat.

Targeted verification:

```powershell
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_dashboard.py -q
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_support_dashboard.py -q
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_replay_dashboard.py -q
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_nf_dashboard.py -q
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_guard.py -q
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_watchdog.py -q
node --check scripts/her2_dashboard/app.js
```
