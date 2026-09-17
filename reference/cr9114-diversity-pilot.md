# Explicit diversity control: CR9114 development pilot

**2026-09-16: in progress; two of six planned arms completed.**
The [fixed protocol](../specs/cr9114_diversity.md) adds a separate on-policy
sequence-entropy term to SFT-to-DPO. It borrows ProteinZero's explicit diversity
control principle, rather than reproducing its embedding loss or reward pipeline.
The [evidence snapshot](evidence/cr9114-diversity-pilot-2026-09-16.json) is partial.

All arms share one SFT initialization and reference. Three entropy coefficients
(0, 0.03, 0.1) and two DPO schedule seeds are fixed in advance. A new development
pool contains 2,048 candidates and 5,796 pairs. No test labels are evaluated.

## Interim measurements

Unique counts and mean Hamming distances use 1,024 fresh temperature-one draws.
Affinity means refer to model-ranked candidates in the fixed development pool,
not to the unconditional generated sample population.

| Model | DPO seed | Pair accuracy | Top-16 mean H1 | Top-32 mean H1 | Unique samples | Mean Hamming | Joint screen |
|---|---:|---:|---:|---:|---:|---:|---|
| SFT | shared | 89.27% | 9.5606 | 9.5649 | 996 | 7.31 | baseline |
| DPO, entropy 0 | 20260916 | 96.20% | 9.4887 | 9.5047 | 248 | 3.13 | fail |
| DPO, entropy 0.03 | 20260916 | 96.48% | 9.5319 | 9.5255 | 916 | 6.40 | fail: affinity |

The first regularized arm meets both diversity-retention gates and improves
top-candidate means over its matched DPO control. It still falls below SFT on
both affinity budgets, so it does not meet the predeclared joint screen.
The stronger coefficient and second seed remain pending. No checkpoint is promoted.

## Validation and limits

The entropy estimator's gradient is checked against exact differentiation on
enumerable categorical distributions. Targeted tests: **45 passed**. The completed
arms passed sample/rescore agreement, frozen-encoder checks and strict checkpoint
reload. The final run uses deterministic PyTorch operations and repeated geometry
encoding agrees byte-for-byte. The runtime source is `66875f8`; local artifacts
are under `outputs/cr9114_diversity_pilot_20260916_v3/`.

Earlier attempts exposed an unstable cross-process encoding byte hash and
historical same-seed training drift. They are retained locally; the protocol
records the corrections. New reference scores are bound to the current encoding.
Historical nondeterministic training is not assumed to replay bit-for-bit.

Fresh measurements and sample-based diversity metrics evaluate behavior outside
the optimized loss, but they remain within the same assay and three development
blocks. Both DPO seeds share the same SFT model. Unlabelled on-policy training
samples may coincide with evaluation identities; their assay labels are never
used in training. The pass/fail gates are descriptive engineering screens, not
confidence intervals or biological validation.
