# CR9114 direct-DPO and SFT-to-DPO pilots

**2026-09-16: direct DPO completed; SFT-to-DPO is running.**
The [fixed protocol](../specs/cr9114_dpo.md) and
[run configuration](../configs/experiments/cr9114_dpo_pilot.json) were committed
before launch at `a06134e`. The [run evidence](evidence/cr9114-dpo-pilot-2026-09-16.json)
reports completed arms only and distinguishes partial from complete status.

The local run is `outputs/cr9114_dpo_pilot_20260916/`. Both arms use the same
512-pair schedule, drawn from the 114,395 declared training pairs. It contains
1,007 unique variants across all ten training blocks. Each reference cache covers
all 38,066 represented training variants, allowing later larger training runs
to use the same fixed reference. No test labels are used.

## Direct-DPO result

The released parent initializes both policy and reference. All 38,066 reference
scores were cached in 385.4 seconds; the 256 DPO updates took 36.8 seconds including
checkpoint writes. Peak PyTorch CUDA allocation during the arm was 1,364 MiB;
this excludes desktop/driver memory and the earlier structural encoding pass.

| Development metric | Parent | Direct DPO |
|---|---:|---:|
| Variant/block-balanced pair accuracy | 66.16% | 94.96% |
| Spearman correlation on 512 candidates | 0.289 | 0.794 |
| Mean measured H1 among selected top 16 | 9.4579 | 9.5266 |
| Mean measured H1 among selected top 32 | 9.4718 | 9.5195 |

For context, the earlier SFT pilot scored 89.21% pair accuracy and selected a
top-16 group with mean 9.5603; the additive baseline scored 97.57% and selected a
top-16 mean of 9.4873. Direct DPO improves over the parent and improves pair
ordering relative to this SFT checkpoint, but does not beat SFT's top-candidate
selection. Pair ordering and measured top-K remain separate endpoints.

The initial loss was log(2), confirming policy/reference agreement. Every update
had a finite loss and nonzero finite decoder gradients. The encoder state stayed
identical and reference-cache bytes stayed unchanged. The final checkpoint loads
with `weights_only=True`, restores decoder tensors exactly, and reproduces saved
development scores within 0.0000153 absolute log probability.

## Artifacts and limits

Each arm directory contains `reference_cache.json`, `reference_checks.json`,
`history.json`, `development_scores.csv`, `development_pair_scores.csv`, four
decoder/optimizer checkpoints and `result.json`. The root stores the exact
sampled pair schedule, configuration/source identity and completed-arm results.
Raw scores are constrained `log q`; implicit rewards are saved separately.

This is a bounded single-seed development comparison, not final-test evidence.
The same 512 scored development candidates and only three genotype blocks are
used. Pair dependencies and block variation preclude treating the pair count as
an independent sample size. No checkpoint is promoted, and no biological
improvement is claimed. SFT-to-DPO includes its earlier SFT compute in addition
to its DPO updates, so total compute differs from direct DPO.
