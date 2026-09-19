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

The reader tolerates an unfinished last JSONL line and reads appended monitoring
records incrementally. It keeps at most one training-history reader, so switching
runs does not load every trajectory's update history into memory.

Targeted verification:

```powershell
.venv/Scripts/python.exe -m pytest src/smallAntibodyGen/tests/test_her2_dashboard.py -q
node --check scripts/her2_dashboard/app.js
```
