# HER2 p-IgGen post-training: run in progress

Snapshot: 2026-09-18, after the complete initial fitting stage. Implementation
commit `08d57ae`; continuation fitting is in progress and final evaluation remains pending.

p-IgGen is the generator. The CNN is an auxiliary classifier added by Codex to
compare ranking of measured sequences. It does not generate sequences, supply
rewards or preference pairs, or choose checkpoints. DPO preferences use measured
high-versus-low bins at matched wild-type mutation distance.

The [fixed protocol](../specs/her2_hcdr3_benchmark.md) compares continued SFT and
DPO at 3/6/10 additional measured GPU minutes per seed, with DPO also continuing
to 20/30 minutes. There are three seeds. Reference scoring is charged to DPO;
the common initial SFT cost is reported separately.

## Initial pretrained SFT: all three seeds

Batch 128, all 120,504 training high-bin rows per pass, five passes per seed.
Each run took about 12.2 minutes including validation and checkpoint checks.
Native cached/full scoring and all 52 parameter-gradient checks passed after
training in each run; no run triggered the suspected memory-spill flag.

| Seed | Pass | Validation high-bin NLL/residue | Validation AP | Validation AUROC |
|---|---|---:|---:|---:|
| 20260918 | 1 | 1.565302 | 0.942168 | 0.961068 |
| 20260918 | 3 | 1.491964 | 0.965874 | 0.978198 |
| 20260918 | 5 | 1.502257 | 0.969287 | 0.980280 |
| 20260919 | 1 | 1.564050 | 0.938045 | 0.959678 |
| 20260919 | 3 | 1.488833 | 0.967717 | 0.979037 |
| 20260919 | 5 | 1.501824 | 0.969823 | 0.980580 |
| 20260920 | 1 | 1.558214 | 0.940381 | 0.961579 |
| 20260920 | 3 | 1.486741 | 0.966903 | 0.978529 |
| 20260920 | 5 | 1.500826 | 0.970103 | 0.980810 |

The declared initial checkpoint rule uses validation high-bin NLL, so pass 3 is
best for every pretrained seed. In all three seeds, NLL worsened after pass 3 while training loss continued to
fall, which is consistent with the onset of overfitting. The slightly better AP
at pass 5 does not change the selection rule. These three pass-3 parents are now
frozen for both continuation methods.

The [initial SFT evidence](evidence/her2-initial-sft-2026-09-18.json) retains the
pretrained per-seed training reports and checkpoint hashes. Continuations,
generation and final evaluation remain pending. These are
provisional validation observations, not a final benchmark result.
The validation nearest-neighbor baseline already reaches approximately 0.981 AP,
and 90.2% of validation cores are one mutation from a training core. Independent
assay evaluation and generated-sequence diversity have not yet been measured for
the fitted policies. No measured affinity is assigned to unassayed new sequences.

## Matched random-initialization controls

The same architecture, batch size, five-pass schedule and validation-NLL rule
select pass 5 for each random-initialization control. Each run took about 12.2
minutes. Cached/full scoring and all 52 parameter-gradient checks passed;
none triggered the suspected memory-spill flag.

| Seed | Selected pass | Validation high-bin NLL/residue | Validation AP |
|---|---:|---:|---:|
| 20260918 | 5 | 1.541481 | 0.945652 |
| 20260919 | 5 | 1.538224 | 0.948217 |
| 20260920 | 5 | 1.542707 | 0.946272 |

Pretraining improves validation ranking under this shared schedule. This does
not establish the best performance attainable by a separately tuned scratch
model, and the nearest-neighbor baseline still has higher validation AP than
either generator arm.

All auxiliary classifier fits also finished. Validation three-class cross-entropy
selected pass 10 for each CNN seed (0.129181, 0.132055, 0.137546); the additive
model achieved 0.666580. These classification losses are not directly comparable
with generator sequence NLL. The [complete initial-stage evidence](evidence/her2-initial-fits-2026-09-18.json)
includes all fits and the intermediate selection record. Every selected
checkpoint hash and scientific source-code hash was verified before continuation.

<!-- continuation-progress:start -->
## Continuation fitting progress

2 of 6 trajectories completed. Only completed trajectories appear below.

| method        |     seed |   GPU minutes |   updates |   validation high NLL |   validation pair accuracy |
|:--------------|---------:|--------------:|----------:|----------------------:|---------------------------:|
| continued_sft | 20260918 |       3.00000 |      1335 |               1.49891 |                    0.97154 |
| continued_sft | 20260918 |       6.00000 |      2663 |               1.50421 |                    0.97123 |
| continued_sft | 20260918 |      10.00000 |      4431 |               1.52686 |                    0.97104 |
| dpo           | 20260918 |       3.00000 |       741 |               3.19986 |                    0.99230 |
| dpo           | 20260918 |       6.00000 |      1941 |               3.62332 |                    0.99452 |
| dpo           | 20260918 |      10.00000 |      3540 |               3.85419 |                    0.99545 |
| dpo           | 20260918 |      20.00000 |      7538 |               4.20228 |                    0.99584 |
| dpo           | 20260918 |      30.00000 |     11535 |               4.62396 |                    0.99635 |

These are validation diagnostics, not final affinity or diversity results. No continuation checkpoint is selected until full validation ranking and generation diagnostics are complete. [Progress evidence](evidence/her2-continuation-progress-2026-09-18.json) retains actual GPU time, unique and repeated exposures, reference costs and checkpoint digests.
<!-- continuation-progress:end -->

For the first seed, longer DPO fitting improves held-out preference ordering while
sharply worsening the likelihood of measured high-bin sequences. At 30 GPU minutes,
pair accuracy is 0.996346 but high-bin NLL is 4.623964 per residue; continued SFT
at its 10-minute endpoint has pair accuracy 0.971036 and NLL 1.526862. The parent
NLL was 1.491964. This is a substantial density shift, not evidence that the
generated antibodies bind better. Full ranking, actual sampling, the remaining
seeds and independent assay evaluation are still needed.

## Verification and artifacts

Before fitting: 1,984 repository tests passed, 3 skipped; a native 22M-parameter GPU
scoring/gradient check passed; and a tiny synthetic run completed both training
methods, checkpointing, generation, validation and the final evaluation handoff.

Full local artifacts are in `outputs/her2_posttrain_20260918/`. The initial
integrity audit read aggregate counts and two example rows from every published
split, including test; model-based final test and SPR evaluation remain pending.
p-IgGen's HER2 pretraining exposure remains unresolved.
