# CR9114 direct-DPO and SFT-to-DPO pilots

**2026-09-16: both bounded pilots completed and checkpoint reloads verified.**
The [subsequent overfitting and diversity investigation](cr9114-dpo-diagnostics.md)
replicates the selection tradeoff on 2,048 additional development candidates and
finds substantial distributional concentration in both DPO arms.
The [fixed protocol](../specs/cr9114_dpo.md) and
[run configuration](../configs/experiments/cr9114_dpo_pilot.json) were committed
before launch at `a06134e`. The [run evidence](evidence/cr9114-dpo-pilot-2026-09-16.json)
now contains both completed arms. The earlier direct-only evidence was committed
separately at `4534683` while the second branch was still running.

The local run is `outputs/cr9114_dpo_pilot_20260916/`. Both arms use the same
512-pair schedule, drawn from the 114,395 declared training pairs. It contains
1,007 unique variants across all ten training blocks. Each reference cache covers
all 38,066 represented training variants, allowing later larger training runs
to use the same fixed reference. No test labels are used.

## Four-model development comparison

All models use the same 1,399 development pairs and 512-candidate pool. The
additive ridge predictor remains a practical baseline.

| Model | Balanced pair accuracy | Spearman correlation | Top-16 mean measured H1 | Top-32 mean measured H1 |
|---|---:|---:|---:|---:|
| Parent | 66.16% | 0.289 | 9.4579 | 9.4718 |
| SFT | 89.21% | 0.808 | 9.5603 | 9.5528 |
| Direct DPO | 94.96% | 0.794 | 9.5266 | 9.5195 |
| SFT-to-DPO | 96.47% | 0.794 | 9.5118 | 9.5171 |
| Additive ridge | 97.57% | 0.814 | 9.4873 | 9.5021 |

SFT-to-DPO improves reliable within-block pair accuracy over SFT by **7.26
percentage points**, but the top-16 and top-32 measured means decrease and global
rank correlation slips. Direct DPO similarly improves pair ordering relative
to SFT without matching its top-candidate selection. Neither DPO arm beats the
additive baseline on pair accuracy. These outcomes do not support claiming a
general DPO improvement or promoting the DPO checkpoint for strongest-binder
selection. They support a mechanistically working DPO implementation and a
development-side tradeoff worth investigating.

The 7.26-point estimate is not a confirmation of the previously proposed
five-point minimum useful effect: uncertainty, training-seed variation and
untouched-test evaluation remain unresolved. SFT-to-DPO accuracies by block are
92.10%, 99.24%, and 98.08%; the leave-one-block-out range is 95.09%-98.66%,
which is a sensitivity summary rather than a confidence interval.

## Runtime and validation

The released parent initializes both policy and reference. All 38,066 reference
scores were cached in 385.4 seconds; the 256 DPO updates took 36.8 seconds including
checkpoint writes. SFT-reference caching took another 385.4 seconds and its
256 DPO updates took 36.6 seconds. Peak PyTorch CUDA allocation during either
arm was approximately 1,365 MiB;
this excludes desktop/driver memory and the earlier structural encoding pass.

| Development metric | Parent | Direct DPO |
|---|---:|---:|
| Variant/block-balanced pair accuracy | 66.16% | 94.96% |
| Spearman correlation on 512 candidates | 0.289 | 0.794 |
| Mean measured H1 among selected top 16 | 9.4579 | 9.5266 |
| Mean measured H1 among selected top 32 | 9.4718 | 9.5195 |

Both initial losses were log(2), confirming policy/reference agreement. Every update
had a finite loss and nonzero finite decoder gradients. The encoder state stayed
identical and reference-cache bytes stayed unchanged. Both final checkpoints load
with `weights_only=True`, restore decoder tensors exactly, and reproduce saved
development scores within 0.0000153 absolute log probability. Independent
post-run checks reconstructed both pair accuracies from saved candidate scores,
verified implicit rewards, and confirmed that the two references share genotype
order and encoder state while using distinct decoder states and scores.

The full code suite passed before launch: **1,721 passed, three skipped**.

## Artifacts and limits

Each arm directory contains `reference_cache.json`, `reference_checks.json`,
`history.json`, `development_scores.csv`, `development_pair_scores.csv`, four
decoder/optimizer checkpoints and `result.json`. The root stores the exact
sampled pair schedule, configuration/source identity and completed-arm results.
Raw scores are constrained `log q`; implicit rewards are saved separately.

Final checkpoints are `direct_dpo/decoder_step_0256.pt` and
`sft_dpo/decoder_step_0256.pt`, relative to the run directory. Source and file
hashes are recorded in the compact evidence; cache and checkpoint files remain
local and are not committed.

This is a bounded single-seed development comparison, not final-test evidence.
The same 512 scored development candidates and only three genotype blocks are
used. Pair dependencies and block variation preclude treating the pair count as
an independent sample size. No checkpoint is promoted, and no biological
improvement is claimed. SFT-to-DPO includes its earlier SFT compute in addition
to its DPO updates, so total compute differs from direct DPO.
