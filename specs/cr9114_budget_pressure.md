# Affinity-only training-budget stress test

2026-09-17. One diagnostic continuation, specified before fitting. The question
is how far the policy moves with substantially more labelled training and what
happens to measured affinity and coverage when it does. This is not a new
regularizer comparison, a noise-level significance contest, or a model promotion.

## Intervention

Extend the existing seed-20260925 affinity-only control from 256 to 16,384
updates: 65,536 labelled sequence exposures, 64 times its earlier budget and
about 6.76 exposures per member of the 9,698-positive training population.
Start from the same original SFT checkpoint. Keep the existing training-only
affinity weights, batch size 4, NLL/16, AdamW 1e-5, weight decay .01 and gradient
clip 1. Encoder frozen, decoder trainable, dropout off, deterministic float32.

There is no reference-KL, entropy, embedding, or reference-likelihood penalty.
The earlier affinity-only control already had these penalties disabled; **budget
is the only intervention**. Ordinary optimizer weight decay and gradient clipping
are retained, so "leash off" does not mean removing all numerical/optimization
controls. The objective still fits a broad bounded-weight training population;
it does not maximize an unconstrained affinity reward. A small eventual change
could reflect that objective, not a failed implementation.

Population weights enter only through which identities the schedule draws. The
minibatch loss is an unweighted mean; weights are never applied a second time.
One AdamW object serves every update and is never rebuilt, so the optimizer
state carries across checkpoints and evaluations.

Use the identical seeded sampling stream, and verify its first 256 rows against
the prior arm's saved schedule file and pinned hash before fitting. Require the
256-update decoder to reproduce the prior affinity-only arm's **decoder state
digest** `1619d9fe8ba40ba7bcd525922d0563fc7c16445a05fa3658d650d793e1140477`.
The checkpoint *files* cannot match byte for byte, because they embed different
configuration and metadata; the state digest is the reproduction claim. If that
gate fails, stop, investigate and report it. There is no pre-authorized variant
run that relaxes it.

Checkpoints are fixed at 256, 2,048, 8,192 and 16,384 updates. Log every update;
no early stopping based on affinity, diversity, or development results. Stop only
for a failed integrity check or nonfinite numerical values. Report all
checkpoints; do not select the best one or change the budget/learning rate after
inspection. The run's stopping condition is successful completion plus a passing
independent audit and test suite, not any measured result.

## Measurement

Score all 65,536 legal identities under SFT and the final decoder, without any
affinity lookup. Verify total probability is within 2e-5 of one, then normalize
only for floating-point summation. Report exact joint entropy, effective support,
collision probability, KL in both directions, total variation, maximum atom
probability and mass by train/development/test identity split. This is a
distributional movement check, not a parameter-norm or top-K-overlap proxy.
The normalization check bounds float accumulation error at the two endpoints
only. It says nothing about the intermediate checkpoints, which are never
exhaustively scored.

Split membership is a deterministic function of four of the sixteen editable
sites (0-based loci 2, 5, 8, 10). Training on train-block positives can shift
those four marginals, and therefore mass between splits, without any affinity
learning. So report all sixteen per-site allele-1 marginals, mass for all
sixteen blocks, and conditional development affinity per development block as
well as overall. Otherwise a mass shift reads as a generalization gain.

At every checkpoint retain the prior 2,048-member fixed development ranking
evaluation, ordinary top-16/top-32, overlap/swaps, shortlist Hamming, and a
continuous probability-weighted affinity conditional on that pool. Pool scores
are constrained joint log probabilities over exactly this 65,536-point support,
so `sum(exp(log q))` on any subset is that subset's true unconditional mass.
Pass those raw weights through; never renormalize inside the pool first and
never call the result a distribution summary.

Quantify affinity-difference assay uncertainty with shared identities cancelled;
this assumes independent errors across genotypes and is a heuristic based on the
existing effective SEM, not a calibrated noise floor or significance test. The
shared pooled floor in that SEM is a common estimated scale parameter, not
evidence of correlated genotype errors, and it does not license a sensitivity or
power claim. Report the effective number of weighted genotypes, `1/sum(w^2)` on
the conditional weights, beside every conditional mean, because a concentrated
policy can rest a mean on very few identities.

Also evaluate 1,024 actual unconditional samples at each checkpoint, seed
20260928. SFT and the bitwise-matching 256-step arm reuse their hash-verified
saved draws from the earlier study: the pinned artifact is the contract, checked
by hash and by a teacher-forced rescore under the live model. Do not regenerate
those draws and then ignore a mismatch. Later checkpoints produce new draws
through the native sampler. Freeze/save identities before any assay joins.
Retain duplicates when evaluating sampling frequencies. Verify sample and
teacher-forced score agreement below 1e-4 and look up reference scores in the
exhaustive SFT cache. Evaluation receives no optimizer, runs under no-grad with
its own generator, and must leave decoder state, optimizer state and the global
RNG unchanged.

Sampled KL to SFT and sampled entropy are Monte Carlo estimates with sampling
error at every checkpoint. At the exhaustively scored endpoints they estimate the
exact value; they do not equal it. Report their standard errors and never
present them as exact.

The report shows the generated draws' own conditional train and development
affinity at every checkpoint, with measured draw counts, unscored draw counts
and sampling standard error, alongside the fixed-pool continuous mean, mass and
paired delta. A main trajectory that shows only fixed-pool top-K affinity hides
what the sampled distribution actually did, so neither may be omitted in favour
of the other. Per-block mass is displayed, not merely archived. State that the
prepared 5CJQ context is a different construct from the assayed one, an
engineered stem trimer standing in for the assayed ectodomain and Fv geometry
standing in for the assayed scFv, and link the existing context report for the
full substitution list.

Use only training measurements and the union of previously evaluated development
cohorts (8,704 eligible identities) for generated-affinity lookup. Keep the other
3,046 eligible development measurements and all reserved test measurements
unused. Report counts/probability mass with and without available measurements,
training-label exposure overlap, and affinity separately for training and
development identities. Training affinity is not independent generalization.

Missing outcomes are reported in six membership-defined categories: measured
train, measured development, train-ineligible, development-ineligible,
eligible-development-withheld and test. Measured train and measured development
stay separate columns, so the table has six entries and not five with those two
merged; the prose count must match the table. Eligibility mixes assay-floor
censoring with under-replication, and the safe eligible table records survivors
only, so the reason for any one exclusion is not recoverable in scope. State
that the conditional mean may be biased in unknown direction and unquantified
magnitude; do not assert a sign, invent a missing value, or read a raw landscape
or held-out measurement to manufacture categories.

The original SFT run drew its own labelled batches. Reconstruct those exposures
against `training_positives.csv` in its own row order, prove that order against
the pinned `split.csv` restricted to the verified positive training identities,
and check the replayed generator state against the state stored in the portable
SFT checkpoint. Indexing the sorted training population instead produces a
plausible but wrong exposure set. Publish unique counts and overlap for the
original SFT draws, this continuation's schedule prefix at each checkpoint, and
their union, and save the exact exposure identities.

At the two exhaustive endpoints compute exact probability-weighted affinity
conditional on each eligible evaluated population, alongside that population's
probability mass and joint probability of exceeding the training-only upper
quartile threshold. Never present a conditional mean as unconditional generated
affinity or silently discard missing/censored/test outcomes. Exact distribution
calculations remove sampling noise, not assay uncertainty. This is still a
finite, previously assayed 16-site landscape; it is not evidence for novel
sequences beyond this edit space or for another antibody lineage/antigen.

The weighted training target's mean affinity, 9.540962930314794, is a benchmark
and not a bound: it is the mean of the target distribution, so a policy
concentrated on a strong measured identity exceeds it. It is not a convergence
guarantee or a gate.

The best measured training identity is a different kind of quantity. It *is* an
upper bound, but only on a conditional average taken over those same measured
training identities, because such an average is a weighted mean of measured
values and cannot exceed the largest one. It does not bound conditional
development affinity, affinity outside the measured set, or what the policy
could reach at an unassayed identity. Report the two distinctly; do not label
both "not a bound".

## Validation and reporting

Before fitting, audit the prior study's set overlap, shared-identity assay-error
propagation and policy movement. Unit-test distribution/measurement calculations,
duplicate handling and held-out-label isolation. Independently recompute saved
metrics, schedules, hashes and endpoint distributions after completion, using the
auditor's own formulas over saved artifacts rather than the runner's helpers.
Recompute the sampled standard errors, Hamming, split counts and exposure counts
too, re-gate endpoint normalization, and compare the locally saved development
pool against the pinned cohort and the safe source labels rather than trusting
the run's own copy. Claim only what is actually recomputed: recorded metadata
and the live strict-reload, encoder-unchanged and RNG-invariance assertions are
read back, not re-derived, and training is not replayed. Report wall clock
separately from optimizer-loop time; a training-seconds field that silently
includes intermediate sampling and exhaustive scoring is not a training time.
Read score CSVs with round-trip float parsing on both sides, so a saved score
compares exactly against its array source; identity, hash and decoder-state
checks stay exact equality rather than acquiring a tolerance.
Prove label isolation positively: perturbing the 3,046 withheld development
measurements must change no reported number, and a synthetic test-labelled row
must be rejected, not silently filtered away. Test-label sentinels use synthetic
fixtures; no reserved measurement is read. Verify safe strict checkpoint reloads,
unchanged encoder state, finite parameters, correct step counters, and the
256-update prefix agreement with the prior independently completed arm. Commit
code/protocol before the run and compact evidence afterward; large local
artifacts remain ignored. No checkpoint promotion.
