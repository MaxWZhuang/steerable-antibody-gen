# Explicit sequence diversity and independent measurement evaluation

This fixed development experiment adapts the separate diversity-control principle
from [ProteinZero](https://arxiv.org/html/2506.07459v4#S3). It does not reproduce
that paper's embedding loss, online reward models or monomer-folding benchmark.
Our constrained 16-site antibody policy permits direct sampling-based sequence
entropy optimization without changing the structural model or scoring contract.

## Training objective

Initialize every arm from the identical verified SFT checkpoint and keep that
checkpoint as DPO reference. The target is `L_DPO - lambda * H(q)` with beta 0.1
and the existing fixed geometry. Estimate the negative-entropy gradient from
eight fresh, independent on-policy samples every fourth update. Preserve duplicate
draws. Re-score immediately with gradients, before any optimizer update. Use the
score-function gradient with a detached leave-one-out baseline, whose expectation
equals the full constrained sequence-entropy gradient. The gradient carrier's
numerical value is not an entropy estimate. Report mean negative sampled log q.

Multiply the entropy coefficient by four on those updates so its nominal average
weight is lambda. This periodic schedule is an approximation to simultaneous
optimization, not an assertion of identical SGD trajectories. Sampling variance,
rare-mode coverage and gradient clipping can affect the result. Entropy applies
over all 65,536 genotypes; no generated sample is joined to an assay label during
training. Consequently, a generated identity can coincide with a development or
test identity. Such draws are unlabelled; we make no never-seen-sequence claim.

The [configuration](../configs/experiments/cr9114_diversity_pilot.json) fixes
lambda at 0, 0.03 and 0.1 and DPO schedule seeds 20260916 and 20260918. All six
arms receive 256 updates and the same pair schedule within each seed. Lambda zero
is a matched DPO control. Regularized arms incur additional generation/scoring
compute; equal pair exposures do not imply equal computation. No coefficients
or step counts may change after viewing evaluation results. Two seeds characterize
DPO sampling variation only: the preceding SFT model is shared.

## Evaluation outside the optimized loss

Before fitting, select another 2,048 eligible development genotypes by sequence
hash, excluding the original 512 and the 2,048 diagnostic candidates. All models
use this identical fixed pool. Retain the training-derived preference rule and
variance floor. Score measured affinity top-16/top-32 means, rank correlation,
variant/block-balanced pair accuracy, and per-block results. Compare with SFT
and the train-only additive predictor. Evaluation does not use a learned reward
model to declare improvement.

Independently of the entropy gradient's training samples, draw 1,024 new
temperature-one samples per final policy. Report actual unique counts, collisions,
Hamming distance, allele frequencies, entropy MC estimates, and KL to the frozen
SFT reference. Rescore samples to verify the autoregressive probability contract.
Record any overlap between unlabelled training draws and development identities.

Predeclared screening rule, separately for each seed: both top-16 and top-32
measured means must be at least SFT's means; unique count and mean pairwise
Hamming must each retain at least 80% of SFT's corresponding statistics. A setting
passes the two-seed screen only if both seeds pass. These are descriptive
engineering gates, not confidence bounds, biological validation or automatic
checkpoint promotion. Report all arms even when none passes.

Only three development blocks from the same assay/lineage remain available.
This checks different measurements and metrics, not an independent biological
study, distant-antibody generalization or independent experimental replication.
Do not open reserved test labels. More permissive post-hoc gates require a new
explicitly labeled experiment rather than rewriting this result.

## Verification

Enumerate small categorical distributions to check the entropy estimator's
expected gradient against exact entropy differentiation. Verify reference cache
identity and contents, initial log(2) DPO loss, unchanged encoder/reference files,
finite gradients, strict checkpoint reload and score parity. Commit code and
protocol before fitting. Keep large checkpoints and sample CSVs local; commit
compact evidence and an outcome report afterward.
