# J11 — SwiGLU width selection (680 vs 1024)

**Status:** budget frozen and **pipeline preflight PASSED**. Evidence runs cleared to launch; see §6.
**Predeclared:** 2026-08-28, before any quality metric was computed.

## 1. Question

The modern block (RoPE, pre-RMSNorm, SwiGLU, no attention/FFN biases) is adopted as the
canonical engineering baseline by owner decision. J11 no longer asks whether the legacy block
deserves to survive — the legacy block keeps only a smoke/regression role.

> Does roughly **33.6% more capacity** produce a non-marginal benefit?

| Architecture | parameters | vs legacy |
|---|---|---|
| legacy | 4,822,530 | — |
| modern, SwiGLU 680 | **4,719,106** | −2.1% |
| modern, SwiGLU 1024 | **6,304,258** | +30.7% |

Confirmed **by construction**, not by formula. The two arms differ by exactly
`3 × 256 × (1024 − 680) × 6 = 1,585,152` parameters, all of it feed-forward width. The 680 arm
lands *below* legacy because RoPE removes the learned position table (289 × 256 = 73,984; the
+1 is the reserved pad row).

Parameter parity is **not** required. Accidental capacity is bad, which is why
`swiglu_hidden_dim` is explicit and never inferred from `d_ff`; deliberate capacity is the
thing being measured.

## 2. Pairing — why three seeds is evidence

The arms differ only in the SwiGLU projections, but the wider arm's feed-forward consumes more
initialization RNG, so **every parameter constructed after it lands on a different value** —
attention, norms, embeddings, the head. Three seeds of a comparison whose arms start from
different weights is not three paired observations; it is six unrelated runs, and 1.0
percentage point of HCDR3 recovery sits well inside that noise.

`experiments.init_parity.reinitialize_by_module_name` re-derives every parameter from a seed
hashed on its module **name**, so values do not depend on construction order and therefore not
on the width. `reset_training_rng` then puts both arms on one data-order and dropout stream.

Measured at the canonical shape: **40 same-shape parameters identical, 0 differing, 18
shape-mismatched** — and the 18 are exactly the `gate_proj` / `up_proj` / `down_proj` tensors,
the axis under test.

Two details that would silently break it, both pinned by test: residual-depth scaling is a
post-init multiply and must be re-applied after re-initialization, and the pass must restore
the global RNG so it does not itself perturb data order.

## 3. Frozen budget

| | |
|---|---|
| Seeds | 42, 31415, 271828 |
| Batch size | 16 |
| Schedule | identical examples, masks, optimizer steps, and warmup — **not** equal wall-clock |
| Schedule per run | **51,000** optimizer updates = 1,000 warmup + 50,000 post-warmup |
| Total evidence budget | **50 GPU-hours** across all six runs (amended from 36 on 2026-08-28) |
| Minimum | no selection from fewer than **50,000** post-warmup updates per arm |

The budget was amended **while blind** — no evaluation metric had been computed or viewed, so
the increase could not have been influenced by which arm was ahead. That is what separates a
budget amendment from a moved goalpost.

Wall-clock and throughput are **reported costs**, not levers on how much data each arm
receives. Deriving the common step count from the slower arm is deliberate: giving the faster
arm more steps would answer a different question and flatter the narrow arm.

## 4. Timing calibration (measured 2026-08-28)

Exact forward/backward/update path, GTX 1650 SUPER, batch 16, AMP on. **No validation metric
was computed or retained**, so the budget was frozen without anyone seeing which arm wins.

| width | params | median step | p90 | allocated | reserved |
|---|---|---|---|---|---|
| 680 | 4,719,106 | **379.81 ms** | 382.97 ms | 515.2 MiB | 576.0 MiB |
| 1024 | 6,304,258 | **446.89 ms** | 448.72 ms | 608.9 MiB | 684.0 MiB |

**Width 1024 is 17.7% slower** — comfortably inside the 35% promotion threshold.

One complete corpus epoch (2,970,227 rows → 185,640 steps) across all six runs would cost
**127.89 GPU-hours** against a 36-hour budget, so one epoch is out. The derived common step
count is **52,256**, which clears the 50,000 floor by **+4.5%**.

### 4.1 The compute-only projection was pessimistic, and here is why

The calibration padded every batch to the full `max_length` of 288 and projected 52,256
affordable steps against the old 36-hour budget — clearing the 50,000 floor by 4.5%, which any
dataloading overhead would have erased. That reasoning contained an error worth recording,
because it pointed the wrong way.

**Full occupancy is not what training sees.** The unpaired OAS corpus is single-chain, so its
sequences tokenize to ~120 tokens; `max_length: 288` is a *cap*, and the length-bucketed batch
sampler packs each batch to its bucket rather than to the cap. Measured over the real loader,
batch sequence lengths are **111–135 tokens, median 119**. So the synthetic figure was a
worst-case **upper** bound on step time, not the lower bound the earlier draft of this section
claimed. Dataloading does add overhead; bucketing removes far more.

### 4.2 Pipeline-inclusive preflight (measured 2026-08-28)

Production loaders, real corpus, including dataset construction, the bucketed sampler,
collation, dynamic MLM masking, forward/backward/update, checkpoint serialization, and a
validation traversal. **No numerical metric was retained**: validation batches were traversed
and their per-batch counts computed — that Python-level work is a real cost — then discarded
without aggregation, so nothing that could reveal which width is ahead was ever formed.

| width | median step | checkpoint save | validation | peak allocated |
|---|---|---|---|---|
| 680 | **151.6 ms** | 0.07 s (54 MiB) | 73.6 ms/batch | 282 MiB |
| 1024 | **179.9 ms** | 0.08 s (72 MiB) | 84.8 ms/batch | 335 MiB |

Real-loop slowdown of the wider arm is **+18.7%**, against +17.7% from the synthetic
calibration — the absolute step time did not transfer, but the *ratio* did, which is what the
35% promotion criterion turns on.

Projected for the frozen schedule (51,000 updates x 3 seeds x 2 widths, with validation and
checkpointing):

| | per run | x3 seeds |
|---|---|---|
| width 680 | 2.17 h | 6.51 h |
| width 1024 | 2.58 h | 7.74 h |
| **total** | | **14.25 GPU-hours** |

**Against a 50-hour budget, with 35.75 hours to spare.** J11 may launch.

That headroom is large enough to be worth stating plainly rather than banking: it exists
because stage-1 sequences are short, and it would not survive a move to the paired corpus
(~236 tokens) or the antibody-antigen corpus (~232). It is not a general licence to run longer
experiments on this card.

## 5. Promotion rule

Promote **width 1024** only when **all** hold:

1. Full 288/1024 dual-stream at batch 16 retains **≥25% physical-memory headroom** with no
   driver spill.
2. Stage-1 median step time is **≤35% slower** than 680.
3. Mean fixed-mask HCDR3 token recovery improves by **≥1.0 absolute percentage point**.
4. **All three paired seeds** favor 1024 on HCDR3 token recovery.
5. Mean validation MLM loss does not regress by more than **0.01 nats**.
6. HCDR3 span-exact recovery does not regress by more than **0.25 percentage points**.
7. Neither arm shows NaNs, unexplained AMP skips, or instability.

Otherwise promote **680**. It is the tie and inconclusive fallback: the extra 33.6% capacity
must earn its place with a practical CDR3 benefit.

### 5.1 Criteria already satisfied

Criteria 1 and 2 are measurable before any quality metric and both pass.

**Memory**, full 288/1024 dual-stream, batch 16, AMP on (allocated quoted for comparison per
the standing convention; reserved checked for spill):

| arm | allocated | reserved | reserved headroom |
|---|---|---|---|
| 680 | 2101.7 MiB | 2392.0 MiB | **41.6%** |
| 1024 | 2431.8 MiB | 2808.0 MiB | **31.4%** |

Both clear 25% with no spill. (Batch 32 does **not** fit either modern arm — 108.6% and 126.1%
of the card — so batch 16 is not merely the frozen choice, it is the only one available.)

**Speed:** 17.7% ≤ 35%. ✓

## 5.2 Objective: canonical 0.4, not plain MLM

The arm configs use `hcdr3_span_probability: 0.4`, matching
`configs/pretrain_oas_small.yaml`. An earlier draft carried `0.0`, which would have made J11 a
plain-MLM pilot wearing a stage-1 name.

That is not only a consistency point. With no HCDR3 span deliberately masked, `hcdr3_valid_spans`
is ~0 and span-exact recovery is NaN — so promotion criteria 3 and 6, the two the whole
experiment turns on, would have been unmeasurable. The objective mismatch would have surfaced as
empty metrics after six runs had already been spent.

## 5.3 Execution harness

The frozen design has to be what actually runs. These enforce it:

| Requirement | Mechanism |
|---|---|
| Stop at exactly update 51,000 | `max_updates` in the trainer, counting UPDATES the optimizer applied — not batches, so AMP skips cannot shorten one arm |
| 1,000-update warmup | `warmup_steps: 1000`; validation rejects `warmup_steps >= max_updates` |
| Paired init before the first update | `paired_init_seed`, applied straight after construction and before any warm start |
| Clean worktree, recorded commit | `run_j11_experiment.py launch` refuses a dirty tree, and refuses a non-`main` HEAD without `--allow-branch` |
| Final-step metrics only | `read_final_metrics` takes the LAST validation record; best-checkpoint selection is a maximum over noise |
| No early stopping | `TrainConfig.validate` rejects `max_updates` together with `early_stopping_patience` |
| Refuse incomplete evidence | `collect` surveys all six runs before reading any, and reports every gap at once |
| One axis, both directions | widths at fixed seed may differ only in `swiglu_hidden_dim`/`output_dir`; seeds at fixed width only in `seed`/`paired_init_seed`/`output_dir` |

Six explicit tracked configs, `configs/experiments/swiglu_width/arm_w{680,1024}_s{42,31415,271828}.yaml`,
written out rather than generated at launch so the executed experiment is a diffable artifact.

## 6. Launch status

Both preconditions are cleared:

1. **Budget.** The pipeline preflight projects 14.25 GPU-hours against 50, and the frozen
   51,000-update schedule delivers exactly the 50,000 post-warmup updates the evidence floor
   requires. If a rerun ever projects beyond the budget, J11 goes back to blocked — the floor
   is not lowered and no selection is made from partial runs.
2. **Durable remote.** `fix/cross-platform-artifacts` is pushed to `origin`; CI runs on push.
3. **Harness (§5.3).** In place, and the launcher refuses anything that is not the frozen design.

**Remaining before launch: merge this PR, then train a clean checkout of the resulting
`origin/main` SHA** and record that SHA with every run. The launcher enforces this — it refuses
a non-`main` HEAD unless explicitly overridden — because evidence pinned to a feature-branch
commit is pinned to a revision that may never exist in the public history.

One consequence of the local-only `docs/` policy, recorded because it is easy to be surprised
by: `docs/ARCHITECTURE.md` does not travel to the public repository, so a clone has the
dual-stream model without its integration diagram. This tracked contract does travel.

## 7. Artifacts

| Artifact | Path |
|---|---|
| Timing calibration (compute only) | `scripts/calibrate_j11_timing.py`, `outputs/j11-timing-calibration.json` |
| Pipeline preflight | `scripts/preflight_j11_pipeline.py`, `outputs/j11-pipeline-preflight.json` |
| Pairing mechanism | `src/smallAntibodyGen/experiments/init_parity.py` |
| Paired arm configs | `configs/experiments/swiglu_width/arm_{680,1024}.yaml` |
| Tests | `src/smallAntibodyGen/tests/test_j11_pairing.py` (9) |

## 8. Verdict — width 680 selected (2026-08-28)

**Emitted by `python scripts/run_j11_experiment.py compare`**, not by hand.
Report: **`specs/evidence/j11-comparison.json`** (`schema_version: j11-comparison/1`), bound
to all six runs' fingerprint hashes, their shared training commit `561312b`, and a hash of
the exact final metrics record each number came from.

The report is tracked, deliberately. It was first written to `outputs/`, which
`.gitignore` excludes — so the six fingerprint and metrics hashes, the entire point of
emitting them, vanished in a fresh clone. It carries no absolute paths: evidence sources are
serialized repo-relative POSIX, so two machines comparing the same evidence produce the same
bytes and no home directory leaks into a tracked file.

Everything above this section is **byte-identical** to the version hashed into all six run
fingerprints as `specs/experiments/j11_swiglu_width.md` →
`ee760b4a57d4f3838abaadc8eae1dbbe5a80d1142c15f41e31b452573b3128fe`. That hash is the
predeclaration proof; it is stale by construction now that §8 and §9 are appended below it.

Verify it exactly — **the recorded hash is of the CRLF form, 12,006 bytes**, which is what the
fingerprint read off the Windows working tree. `git show` yields the LF form and hashes to
`13ee222d…` instead, so hashing the git blob directly will look like a mismatch when nothing
is wrong:

```python
import hashlib, subprocess
orig = subprocess.run(["git", "show", "561312b:specs/experiments/j11_swiglu_width.md"],
                      capture_output=True).stdout
orig = orig.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")        # 12,006 bytes
assert hashlib.sha256(orig).hexdigest().startswith("ee760b4a")     # the recorded hash
assert open("specs/experiments/j11_swiglu_width.md", "rb").read().startswith(orig)
```

The second assertion is the one that matters: the predeclared text is a **prefix** of this
file, so §8 and §9 could only be appended, never edited into it. (Do not hash "everything
above §8" — that is 12,008 bytes, two more than the original, because the append added a
blank separator line.)

### 8.1 Paired results, final step 51,000

Validation masks are identical within a seed across widths (`hcdr3_target_tokens` matches
exactly: 772,352 / 771,937 / 771,724), so the within-seed comparison is exact. Masks differ
*between* seeds, so the seed-to-seed spread bundles evaluation-mask variation with
initialization and data-order variation — do not read it as pure model noise.

| seed | tok-acc 680 | tok-acc 1024 | Δ pp | MLM loss Δ nats | span-exact Δ pp |
|---|---|---|---|---|---|
| 42 | 46.222% | 46.582% | **+0.360** | −0.00414 | −0.403 |
| 31415 | 46.063% | 46.445% | **+0.382** | −0.00301 | +0.193 |
| 271828 | 46.305% | 46.481% | **+0.176** | −0.00109 | −0.399 |
| **mean** | **46.197%** | **46.503%** | **+0.306** | −0.00275 | −0.203 |

### 8.2 The rule, applied

| # | Criterion | Verdict | Measured |
|---|---|---|---|
| 1 | ≥25% dual-stream reserved headroom | **not_auditable** | no retained probe — see §8.3 |
| 2 | ≤35% slower | pass | **+18.70%** (preflight medians 0.15158 s / 0.17993 s) |
| 3 | ≥+1.0 pp HCDR3 token recovery | **FAIL** | **+0.306 pp** |
| 4 | all three paired seeds favour 1024 | pass | 3/3 |
| 5 | MLM loss regression ≤0.01 nats | pass | −0.00275 (1024 better) |
| 6 | span-exact regression ≤0.25 pp | pass | 0.203 pp — but see §9.2 |
| 7 | no NaNs / AMP skips / instability | **not_auditable** | metrics finite; skip count never persisted |

**Selected width: 680.** Promotion of 1024 requires **every** criterion to pass; criterion 3
fails by a factor of 3.3, and two further criteria are unauditable. 680 is the predeclared
tie-and-inconclusive fallback.

### 8.3 Two criteria have no retained measurement

Recorded because the protocol asserted both as satisfied and neither can be checked from
what the runs left behind. Neither changes the selection — criterion 3 fails on its own —
but both would have mattered had the quality result been close.

- **Criterion 1.** §5.1 quotes 2101.7/2392.0 MiB (41.6% headroom) and 2431.8/2808.0 MiB
  (31.4%) at the 288/1024 dual-stream shape. Those figures appear in no file under
  `outputs/`: `gpu-memory-probe-stage2.json` is the legacy antibody-only model
  (4,797,954 params), `-esm.json` is the ESM encoder, `-longcontext.json` is the AB-07
  sweep at `max_length: 192`. The comparator reads
  `outputs/j11-dual-stream-memory.json` and downgrades the criterion when it is absent.
  Producing that probe at the canonical shape is what makes criterion 1 auditable; the
  comparator rejects a probe taken at any other shape rather than accepting a flattering
  one, because on this box the absence of an OOM is not evidence a config fits.
- **Criterion 7.** `UPDATE_COUNTER["amp_skips"]` (`scripts/mlm_train.py`) is counted
  in-process and never written to `metrics.jsonl`, the checkpoint payload, or any retained
  log. The NaN half of the criterion **is** checkable and passes: every retained metric in
  all six final records is finite. The skip half is not. Persisting `amp_skips` into the
  metrics record makes this criterion auditable for future runs.

`not_auditable` is a third verdict, never rewritten to `pass`. "We measured this and it
lost" and "we never measured this" are different facts about an experiment, and a report
that collapses them claims more verification than was performed.

### 8.4 What this licenses

> **Claim.** Width 1024 showed a consistent but practically insufficient early-training
> gain: +0.306 pp mean HCDR3 token recovery across 3/3 paired seeds, against a +1.0 pp bar,
> for 18.7% more step time and 33.6% more parameters.

> **Claim limit.** Measured at the frozen 51,000-update schedule — **30.5% of one training
> epoch** (§9.1). This does **not** establish asymptotic equivalence after full training. It
> is a practical negative on paying for extra width at this budget, **not** a scientific
> negative on capacity.

The 3/3 sign consistency is directional evidence that 1024 genuinely learns faster early,
and it is not established as statistically significant. The naive per-arm binomial standard
error on token recovery (0.057 pp over ~772k residues) treats residues as independent; they
are correlated within antibodies, within repeated biological families, and within a shared
mask. The true experimental unit is closer to the paired seed, of which there are three.

A supporting practical argument, beyond the rule: 680 retains ~10 pp more reserved headroom
at the dual-stream shape — the stage where this card actually binds — and headroom is the
safety margin given the measured driver-spill behaviour.

### 8.5 The harness that produced this verdict was hardened after it first ran

Recorded because the sequence matters. The verdict in §8.2 was first emitted by a comparator
whose evidence binding **failed open** in four ways, found in review on 2026-08-28. The
selection did not change — it was re-verified under every check below, and criterion 3 fails
by a factor of 3.3 regardless — but a report that advertises provenance it did not verify is
the most expensive shape of wrong, because the artifact asserts the very property it lost.

| Failed open | Now |
|---|---|
| A missing `run_fingerprint.json` was recorded as `null` | Refused — a null hash is a hole shaped like a binding |
| Six empty commit strings passed the "one shared commit" check, because they are all equal | Refused — each commit must be a 40-char hex SHA before equality is tested |
| Provenance `width`/`seed`/schedule were never compared to the run they sat beside | Refused on any disagreement with the directory or with the frozen 51,000/1,000 |
| The fingerprint's own `manifests.source.commit` was never cross-checked | Refused on disagreement with `j11_provenance.json`; a dirty `worktree_dirty` is refused too |

Two further gaps closed at the same time:

- **Paired masks are enforced, not quoted.** The report copied the 680 arm's
  `hcdr3_target_tokens` and `hcdr3_valid_spans` into the table without ever comparing them to
  the 1024 arm. An unpaired comparison would have produced a confident verdict under a
  paired-looking table. Unequal masks at any seed now refuse. (They are equal here:
  772,352 / 771,937 / 771,724, verified rather than assumed — `paired_masks_verified: true`.)
- **Criterion 1's probe must prove what it measured.** Shape alone is insufficient: a
  288/1024 batch-16 probe of the single-stream model, of the legacy block, or without AMP
  measures a different thing. A probe is now read only when every arm records
  `model_kind: antibody_antigen`, `ffn_type: swiglu`, `norm_type: rmsnorm`,
  `position_encoding: rope`, `use_amp: true`, the full 288/1024/batch-16 shape, a positive
  `total_parameters` that is larger for the 1024 arm, and one shared `device_total_mib`; and
  only when the two rows differ off-axis in nothing. Anything else is `not_auditable`, never
  a pass. Driver spill is a `fail`, not a downgrade — on this box CUDA falls back to system
  RAM instead of raising, so a spilling config that "ran fine" is precisely what this
  criterion exists to catch.

Criterion 7 stays `not_auditable` for these six runs permanently. Persisting `amp_skips`
fixes future experiments; it cannot retroactively instrument a completed one, and re-running
the six arms to recover it is not warranted when criterion 3 already determines the result.

## 9. Errata against §1–§7

Recorded rather than edited: §1–§7 are the predeclared text (§8), and the six arm configs
and `outputs/j11-timing-calibration.json` are frozen artifacts whose hashes are bound into
the run fingerprints. Correcting a number *inside* them would falsify that provenance chain,
so the corrections live here.

### 9.1 One epoch is ~166,987 updates, not 185,640

§4 states "One complete corpus epoch (2,970,227 rows → 185,640 steps)". That divides the
**whole corpus** by the batch size. A training epoch traverses the **training split**:

| | rows |
|---|---|
| corpus (`data/processed/oas_unpaired_3m/stats.json`, `records_kept`) | 2,970,227 |
| `kept_by_split.train` | 2,672,808 |
| minus the 1,024-row `row_random` probe, removed from training | **2,671,784** |

2,671,784 / 16 = **166,987** updates per epoch (rounded up). The frozen 51,000-update
schedule is therefore **30.54%** of one training epoch, not the 27.5% the corpus denominator
implies. (The `known_target` probe is sampled from *retained* rows and subtracts nothing.)

The figure does not change any conclusion — 51,000 was frozen as an absolute update count,
never as a fraction — but it makes the run less under-trained than the stale number
suggested, which matters for §8.4's claim limit.

The stale value survives deliberately in: the six `configs/experiments/swiglu_width/*.yaml`
headers, `outputs/j11-timing-calibration.json` (`steps_per_epoch`), and §4 above. It is
corrected in `scripts/mlm_train.py` and `src/smallAntibodyGen/tests/test_j11_harness.py`,
which are live code that would otherwise mislead future work.

### 9.2 Criterion 6's threshold is narrower than its own noise

Span-exact recovery is measured on ~505 valid spans per validation pass. The unpaired
binomial standard error at the observed rate is ≈**0.51 pp**, against a 0.25 pp promotion
margin. The criterion passed at 0.203 pp, but it could not have discriminated a real
regression from sampling noise at that scale.

This is under-instrumentation, not impossibility: because the masks are paired within a
seed, the variance of the *paired* difference can be well below the per-arm binomial error.
The retained artifacts store only aggregate rates, not per-span paired outcomes, so that
analysis cannot be run after the fact.

> The threshold was smaller than the unpaired sampling noise, and the stored aggregates
> cannot support the appropriate paired analysis.

Any future protocol reusing this clause should either retain per-span outcomes or set the
margin from a measured noise band rather than from intuition.
