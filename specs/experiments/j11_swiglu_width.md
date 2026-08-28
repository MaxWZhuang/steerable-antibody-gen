# J11 — SwiGLU width selection (680 vs 1024)

**Status:** budget frozen from timing-only calibration. **Evidence runs not started**; see §6.
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
| Total evidence budget | 36 GPU-hours across all six runs |
| Minimum | no selection from fewer than **50,000** post-warmup updates per arm |

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

### 4.1 That margin does not survive contact with the real loop

The calibration measures **compute only**. It executes no dataloading, no collation, no
masking, no checkpoint writes, and no validation, so its step time is a *lower bound*.

| validation reserve \ overhead | 0% | 10% | 20% | 30% |
|---|---|---|---|---|
| 0 h | **52,256** | 47,505 ✗ | 43,547 ✗ | 40,197 ✗ |
| 1 h | **50,804** | 46,186 ✗ | 42,337 ✗ | 39,080 ✗ |
| 2 h | 49,353 ✗ | 44,866 ✗ | 41,127 ✗ | 37,964 ✗ |
| 4 h | 46,450 ✗ | 42,227 ✗ | 38,708 ✗ | 35,730 ✗ |

✗ = below the 50,000-update floor, where the frozen budget forbids a selection.

**The 36-hour budget clears the floor only if validation reserve ≤ 1 hour and real-loop
overhead is essentially zero.** Ten percent overhead — an ordinary figure for dataloading and
collation — puts every cell under the floor. This is an owner decision before the six runs
start, not something to discover at hour 30. The options are: raise the budget, lower the
floor, shrink the per-step cost, or accept that J11 cannot conclude.

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

## 6. What blocks the evidence runs

1. **The budget question in §4.1.** Running six arms that cannot legally produce a selection
   would spend 36 GPU-hours to learn nothing.
2. **A durable remote reference.** The owner requires the branch pushed or backed up, CI run,
   and the training commit pinned before the evidence runs. Timing calibration was explicitly
   permitted to remain local; the evidence runs are not.

## 7. Artifacts

| Artifact | Path |
|---|---|
| Timing calibration | `scripts/calibrate_j11_timing.py`, `outputs/j11-timing-calibration.json` |
| Pairing mechanism | `src/smallAntibodyGen/experiments/init_parity.py` |
| Paired arm configs | `configs/experiments/swiglu_width/arm_{680,1024}.yaml` |
| Tests | `src/smallAntibodyGen/tests/test_j11_pairing.py` (9) |
