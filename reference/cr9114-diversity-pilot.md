# Explicit diversity control: CR9114 development pilot

**2026-09-16: all six planned arms completed and independently audited.**
The [fixed protocol](../specs/cr9114_diversity.md) adds a separate on-policy
sequence-entropy term to SFT-to-DPO. It borrows ProteinZero's explicit diversity
control principle, rather than reproducing its embedding loss or reward pipeline.
The [complete evidence](evidence/cr9114-diversity-pilot-2026-09-16.json) includes
all arms, numerical replay, measurement checks and artifact hashes.

All arms share one SFT initialization and reference. Three entropy coefficients
(0, 0.03, 0.1) and two DPO schedule seeds are fixed in advance. A new development
pool contains 2,048 candidates and 5,796 pairs. No test labels are evaluated.

## Completed measurements

Unique counts and mean Hamming distances use 1,024 fresh temperature-one draws.
Affinity means refer to model-ranked candidates in the fixed development pool,
not to the unconditional generated sample population.

| Model | DPO seed | Pair accuracy | Top-16 mean H1 | Top-32 mean H1 | Unique samples | Mean Hamming | Joint screen |
|---|---:|---:|---:|---:|---:|---:|---|
| SFT | shared | 89.27% | 9.5606 | 9.5649 | 996 | 7.31 | baseline |
| DPO, entropy 0 | 20260916 | 96.20% | 9.4887 | 9.5047 | 248 | 3.13 | fail: both |
| DPO, entropy 0.03 | 20260916 | 96.48% | 9.5319 | 9.5255 | 916 | 6.40 | fail: affinity |
| DPO, entropy 0.1 | 20260916 | 95.95% | 9.4944 | 9.5011 | 954 | 6.63 | fail: affinity |
| DPO, entropy 0 | 20260918 | 96.25% | 9.5033 | 9.4917 | 563 | 4.62 | fail: both |
| DPO, entropy 0.03 | 20260918 | 96.40% | 9.5170 | 9.5118 | 963 | 6.68 | fail: affinity |
| DPO, entropy 0.1 | 20260918 | 96.61% | 9.5009 | 9.4967 | 974 | 6.72 | fail: affinity |

Every regularized arm meets both diversity-retention gates. Entropy 0.03 also
improves both top-candidate means over its matched plain-DPO control in both seeds.
However, **none of the six arms matches SFT on either affinity budget**, so none
passes the predeclared joint screen. SFT remains the candidate-selection baseline;
no checkpoint is promoted. These are descriptive comparisons, without claims
that the affinity differences exceed assay uncertainty.

The result separates two issues: an explicit entropy term substantially reduces
distributional concentration, while the gap between preference ordering and
top-candidate affinity persists. Removing much of the diversity loss is not enough
to justify a claim of improved binding selection. A stronger entropy coefficient
does not monotonically improve selection quality.

The screen requires top-16 and top-32 means at least as high as SFT, plus at least
80% of SFT's unique-sample count and mean Hamming distance. These correspond to
at least 797 unique draws and mean Hamming 5.8447 for this fixed sample seed.
All coefficients fail the two-seed joint screen. No thresholds were relaxed.

The six training runs took about 587 seconds in total, excluding initialization,
reference scoring and evaluation. Peak PyTorch CUDA allocation was approximately
2,083 MiB for regularized training, excluding driver/desktop memory. The regularized
arms use additional sample-generation and scoring compute; matched preference
exposures do not imply matched computation.

## Validation and limits

The entropy estimator's gradient is checked against exact differentiation on
enumerable categorical distributions. Targeted tests: **45 passed**; the full
repository suite passed **1,741 tests with three skips**. Every arm passed
sample/rescore agreement, frozen-encoder checks and strict checkpoint reload.
The final run uses deterministic PyTorch operations and repeated geometry encoding
agrees byte-for-byte. An [independent replay](../scripts/replay_cr9114_diversity_control.py)
in a separate process reproduced the first control's final decoder **bit-for-bit**.
The runtime source is `66875f8`; local artifacts
are under `outputs/cr9114_diversity_pilot_20260916_v3/`.

Earlier attempts exposed an unstable cross-process encoding byte hash and
historical same-seed training drift. They are retained locally; the protocol
records the corrections. New reference scores are bound to the current encoding.
Historical nondeterministic training is not assumed to replay bit-for-bit: its
scores differ from the final deterministic control by up to 4.60 log units,
despite score-rank correlation 0.9996. The successful deterministic replay is a
separate check, not a claim that the older historical run was reproduced exactly.

The independent audit verified all output and checkpoint hashes, cohort exclusion,
pair accuracies, both top-K means, entropy, Hamming distance, KL estimates,
training histories, regularization sample counts, and each pass/fail gate directly
from saved artifacts. All 256-step histories have finite losses and gradient norms.

Fresh measurements and sample-based diversity metrics evaluate behavior outside
the optimized loss, but they remain within the same assay and three development
blocks. Both DPO seeds share the same SFT model. Unlabelled on-policy training
samples did coincide with 14-23 fresh-cohort identities per regularized arm; their
assay labels were never used in training. Each regularized arm used 512 unlabelled
draws across 64 entropy updates, preserving duplicates. The pass/fail gates are
descriptive engineering screens, not confidence intervals or biological validation.

This experiment does not evaluate the unconditional generated population's
affinity, independent targets, additional SFT seeds or a separate biological
study. The reserved test labels remain untouched. The next useful experiment
should address affinity-selection alignment while retaining the demonstrated
diversity control, with another declared development protocol before fitting.
