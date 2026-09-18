# HER2 p-IgGen post-training: run in progress

Snapshot: 2026-09-18, after all initial and continuation fits. Implementation
commit `08d57ae`; validation and sampling are in progress; final evaluation remains pending.

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
pretrained per-seed training reports and checkpoint hashes. Generation and final
evaluation remain pending. These are
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

## Initial policy sampling

Each policy produced 10,000 native temperature-1 draws, with duplicates retained. All seven initial policies pass the declared distributional diversity checks.

| policy                      |   validation AP |   joint entropy (nats) |   unique fraction |   not exactly in train |   mean core Hamming | eligible   |
|:----------------------------|----------------:|-----------------------:|------------------:|-----------------------:|--------------------:|:-----------|
| piggen_zeroshot             |         0.46958 |               23.71456 |           0.99920 |                1.00000 |             9.15365 | True       |
| policy_scratch_seed20260918 |         0.94565 |               15.41947 |           0.97840 |                0.78170 |             7.55492 | True       |
| policy_scratch_seed20260919 |         0.94822 |               15.39218 |           0.98110 |                0.77330 |             7.54397 | True       |
| policy_scratch_seed20260920 |         0.94627 |               15.45066 |           0.98150 |                0.78730 |             7.50908 | True       |
| policy_sft_seed20260918     |         0.96587 |               14.28494 |           0.97630 |                0.67760 |             7.50636 | True       |
| policy_sft_seed20260919     |         0.96772 |               14.46298 |           0.97910 |                0.68580 |             7.55122 | True       |
| policy_sft_seed20260920     |         0.96690 |               14.55890 |           0.98100 |                0.70670 |             7.47170 | True       |

The pretrained SFT parents improve validation ranking while retaining broad sequence diversity: 97.63-98.10% unique draws, and 67.76-70.67% not exactly present in training. These are sequence-distribution diagnostics, not experimental affinity measurements of the novel draws. Continuation sampling and independent outcome evaluation remain pending.

[Initial sampling evidence](evidence/her2-initial-generation-2026-09-18.json) includes the persisted draw hashes and detailed diversity diagnostics.

<!-- continuation-progress:start -->
## Continuation fitting progress

6 of 6 trajectories completed. Only completed trajectories appear below.

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
| continued_sft | 20260919 |       3.00000 |      1327 |               1.49859 |                    0.97220 |
| continued_sft | 20260919 |       6.00000 |      2654 |               1.50530 |                    0.97228 |
| continued_sft | 20260919 |      10.00000 |      4423 |               1.52515 |                    0.97076 |
| dpo           | 20260919 |       3.00000 |       742 |               3.16644 |                    0.99347 |
| dpo           | 20260919 |       6.00000 |      1941 |               3.50013 |                    0.99549 |
| dpo           | 20260919 |      10.00000 |      3540 |               3.80018 |                    0.99557 |
| dpo           | 20260919 |      20.00000 |      7539 |               4.34547 |                    0.99580 |
| dpo           | 20260919 |      30.00000 |     11539 |               4.55968 |                    0.99654 |
| continued_sft | 20260920 |       3.00000 |      1326 |               1.49653 |                    0.97368 |
| continued_sft | 20260920 |       6.00000 |      2652 |               1.50536 |                    0.97255 |
| continued_sft | 20260920 |      10.00000 |      4420 |               1.52640 |                    0.97185 |
| dpo           | 20260920 |       3.00000 |       741 |               3.24992 |                    0.99226 |
| dpo           | 20260920 |       6.00000 |      1940 |               3.53793 |                    0.99522 |
| dpo           | 20260920 |      10.00000 |      3539 |               3.69654 |                    0.99561 |
| dpo           | 20260920 |      20.00000 |      7537 |               4.16278 |                    0.99627 |
| dpo           | 20260920 |      30.00000 |     11535 |               4.72645 |                    0.99627 |

These are validation diagnostics, not final affinity or diversity results. No continuation checkpoint is selected until full validation ranking and generation diagnostics are complete. [Progress evidence](evidence/her2-continuation-progress-2026-09-18.json) retains actual GPU time, unique and repeated exposures, reference costs and checkpoint digests.
<!-- continuation-progress:end -->

All three continued-SFT trajectories are complete. Their 10-minute high-bin NLL
is 1.525151-1.526862, with held-out preference accuracy 0.970764-0.971853. The
increase in validation NLL relative to the selected parents repeats across seeds.

All three DPO seeds improve held-out preference ordering while sharply worsening
the likelihood of measured high-bin sequences. At 30 GPU minutes, pair accuracy
is 0.996268-0.996540 and high-bin NLL is 4.559677-4.726449 per residue, compared
with initial parent NLL 1.486741-1.491964. Preference-accuracy gains from 10 to
30 minutes are 0.066-0.097 percentage points; the third seed has no further pair
accuracy gain from 20 to 30 minutes while its high-bin NLL keeps increasing.
This is not evidence that generated antibodies bind better, and likelihood
change alone does not establish collapse or DPO overfitting. Full ranking,
actual sampling and independent assay evaluation are still needed.

All 24 budget checkpoint hashes and six parent states verified against their
recorded identities. Every budget exceeded its nominal target by less than one
update; total charged training/reference time was 7,200.399 GPU seconds. Source
code and configuration hashes remained unchanged. The
[integrity record](evidence/her2-continuation-integrity-2026-09-18.json) also retains
the separately measured whole-run wall time and update counts.

<!-- validation-progress:start -->
## Continuation sampling progress

21/31 policy checkpoints have completed validation scoring and native sampling. Reference-KL diagnostics and the final freeze follow the complete per-policy pass.

continued_sft: 9 of 9 sampled checkpoints eligible; dpo: 0 of 5 sampled checkpoints eligible.

| method        |     seed |   minutes |   val AP |   entropy |   unique |   largest mode |   not in train | eligible   |
|:--------------|---------:|----------:|---------:|----------:|---------:|---------------:|---------------:|:-----------|
| continued_sft | 20260918 |   3.00000 |  0.96863 |  13.72458 |  0.97650 |        0.00030 |        0.59670 | True       |
| continued_sft | 20260918 |   6.00000 |  0.96857 |  13.72838 |  0.97960 |        0.00040 |        0.57900 | True       |
| continued_sft | 20260918 |  10.00000 |  0.96793 |  13.46387 |  0.97760 |        0.00030 |        0.53070 | True       |
| continued_sft | 20260919 |   3.00000 |  0.96924 |  13.63752 |  0.97180 |        0.00030 |        0.57280 | True       |
| continued_sft | 20260919 |   6.00000 |  0.96908 |  13.68453 |  0.97710 |        0.00030 |        0.56990 | True       |
| continued_sft | 20260919 |  10.00000 |  0.96782 |  13.45174 |  0.97900 |        0.00030 |        0.52910 | True       |
| continued_sft | 20260920 |   3.00000 |  0.96918 |  13.71337 |  0.97800 |        0.00030 |        0.59840 | True       |
| continued_sft | 20260920 |   6.00000 |  0.96874 |  13.67380 |  0.97980 |        0.00050 |        0.57250 | True       |
| continued_sft | 20260920 |  10.00000 |  0.96810 |  13.37872 |  0.97390 |        0.00030 |        0.51240 | True       |
| dpo           | 20260918 |   3.00000 |  0.97362 |   8.52300 |  0.42030 |        0.00690 |        0.43590 | False      |
| dpo           | 20260918 |   6.00000 |  0.97237 |   8.16098 |  0.37200 |        0.01520 |        0.44090 | False      |
| dpo           | 20260918 |  10.00000 |  0.97348 |   7.72669 |  0.30840 |        0.01260 |        0.45150 | False      |
| dpo           | 20260918 |  20.00000 |  0.97341 |   6.93898 |  0.23670 |        0.04340 |        0.46530 | False      |
| dpo           | 20260918 |  30.00000 |  0.97133 |   6.19388 |  0.18920 |        0.11750 |        0.46880 | False      |

For DPO seed 20260918, all five budgets fail the diversity requirements. Unique draws fall from 42.03% at 3 minutes to 18.92% at 30 minutes; the largest single sequence accounts for 11.75% of draws at 30 minutes. Joint entropy falls from the SFT parent's 14.28 nats to 8.52 at 3 minutes and 6.19 at 30 minutes. This is direct sampling evidence of mode collapse, despite improved held-out high-versus-low pair ordering. Full validation AP falls from 0.97362 at 3 minutes to 0.97133 at 30 minutes.

These are validation observations. [Sampling progress evidence](evidence/her2-validation-progress-2026-09-18.json) retains individual diagnostics and draw/checkpoint hashes. Independent SPR and final test evaluation remain pending.
<!-- validation-progress:end -->

## Verification and artifacts

Validation paused after 21/31 policies because one FP32 sampler/scorer comparison
exceeded its original tolerance. A read-only FP64 diagnosis supports rounding,
not a cache-logic discrepancy: full teacher forcing and autoregression agreed to
8.53e-14 nats on the worst rows. Native math SDPA passed the unchanged FP32
tolerance, with identical 10,000 draws on the failing checkpoint; the ten remaining
DPO checkpoints also passed. All 31 policies are being revalidated uniformly
under this backend before final selection. The original partial tables above
remain preliminary automatic-SDPA evidence. See the
[numerical amendment](../specs/her2_hcdr3_benchmark.md#9-numerical-evaluation-amendment--2026-09-18)
and [audit evidence](evidence/her2-numerical-audit-2026-09-18.json).

The added evaluation driver passed 37 driver/runner checks. It preserves the
original training identities and binds its own hash and backend separately in
the final freeze; it does not alter weights or relax the scoring tolerance.

Before fitting: 1,984 repository tests passed, 3 skipped; a native 22M-parameter GPU
scoring/gradient check passed; and a tiny synthetic run completed both training
methods, checkpointing, generation, validation and the final evaluation handoff.

Full local artifacts are in `outputs/her2_posttrain_20260918/`. The initial
integrity audit read aggregate counts and two example rows from every published
split, including test; model-based final test and SPR evaluation remain pending.
p-IgGen's HER2 pretraining exposure remains unresolved.
