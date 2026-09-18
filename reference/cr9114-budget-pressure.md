# Affinity-only training-budget stress test

2026-09-17. One predeclared continuation of 16,384 updates completed; no checkpoint was selected or promoted.

After 16,384 updates and 65,536 labelled exposures, the policy moved 0.4678 in total variation from SFT over the 65,536 legal identities, with exact entropy 9.5037 nats (effective support 13408.9) against 10.2889 nats at SFT. Probability assigned to development blocks changed from 19.1808% to 0.0421%. The final 1,024 unconditional draws contained 0 development identities and 979 distinct identities overall.

Ordinary top-16 affinity changed from 9.572041 to 9.498454; top-32 changed from 9.559854 to 9.510253. Conditional measured affinity on the 8,704 evaluated development identities moved 0.039067 (24.4x its heuristic paired assay-SEM proxy), on 0.000313 of the total probability mass with 1470.8 effective genotypes. These are descriptive quantities from one seeded continuation, not a significance test and not a promotion decision.

## Policy movement by checkpoint

The Monte Carlo columns come from 1,024 actual draws and carry sampling error. The
two affinity columns are a top-K ranking of a fixed 2,048-identity development pool,
not a property of the generated distribution; the continuous pool quantities and the
affinity of the actual draws follow in the next two subsections. Exact distribution
statistics exist only at the two exhaustively scored endpoints, further below.

| Updates | Labelled exposures | KL to SFT (MC, nats) | MC SE | Entropy (MC, nats) | Unique / 1,024 | Top-16 affinity | Top-32 affinity | Top-16 shared with SFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | -0.0000 | 0.0000 | 10.2792 | 1004 | 9.572041 | 9.559854 | 16/16 |
| 256 | 1,024 | 0.0703 | 0.0132 | 10.2436 | 998 | 9.584705 | 9.577684 | 7/16 |
| 2,048 | 8,192 | 0.2033 | 0.0215 | 10.0699 | 990 | 9.577388 | 9.573616 | 3/16 |
| 8,192 | 32,768 | 0.7322 | 0.0273 | 9.5608 | 969 | 9.577532 | 9.571880 | 6/16 |
| 16,384 | 65,536 | 0.7500 | 0.0246 | 9.4830 | 979 | 9.498454 | 9.510253 | 0/16 |

### Fixed-pool continuous affinity and mass

Continuous quantities over that same fixed pool. Pool scores are constrained joint
log probabilities over the full 65,536-point support, so the mass column is the true
unconditional mass the pool's measured identities carry; the mean is conditional on
them. Both are available at every checkpoint, not only at the endpoints.

| Updates | Measured mass on the pool | Conditional mean | Effective genotypes | Delta vs SFT | Paired SEM proxy | Mass above threshold |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.033056693 | 9.487130 | 823.8 | 0.000000 | 0.000000 | 0.023719895 |
| 256 | 0.036205951 | 9.488512 | 651.9 | 0.001382 | 0.001085 | 0.027604315 |
| 2,048 | 0.030888136 | 9.520402 | 621.1 | 0.033272 | 0.001589 | 0.025364596 |
| 8,192 | 0.000367970 | 9.517493 | 448.4 | 0.030363 | 0.002528 | 0.000288639 |
| 16,384 | 0.000069210 | 9.529644 | 354.1 | 0.042514 | 0.003139 | 0.000056957 |

### Conditional affinity of the actual generated draws

1,024 native draws per checkpoint, duplicates retained. Only draws that land on an
identity carrying a usable measurement inside the evaluated whitelist enter a mean;
the rest are counted as unscored rather than dropped. The sampling SE is the spread
of the measured draws themselves and does not include assay uncertainty, which is
the separate proxy column. A mean over very few draws is a noisy quantity, not a
small effect.

| Updates | Population | Measured draws | Conditional mean | Sampling SE | Assay-SEM proxy | Distinct genotypes | Unscored draws |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | train | 704/1,024 | 9.471282 | 0.006615 | 0.002883 | 688 | 197/1,024 |
| 0 | development | 123/1,024 | 9.487385 | 0.013444 | 0.006943 | 120 | 197/1,024 |
| 256 | train | 685/1,024 | 9.471422 | 0.007902 | 0.003044 | 664 | 197/1,024 |
| 256 | development | 142/1,024 | 9.502615 | 0.011399 | 0.006465 | 140 | 197/1,024 |
| 2,048 | train | 705/1,024 | 9.500818 | 0.005217 | 0.002947 | 677 | 192/1,024 |
| 2,048 | development | 127/1,024 | 9.533172 | 0.005478 | 0.006806 | 124 | 192/1,024 |
| 8,192 | train | 1019/1,024 | 9.516493 | 0.003178 | 0.002463 | 964 | 4/1,024 |
| 8,192 | development | 1/1,024 | 9.594360 | n/a | 0.072421 | 1 | 4/1,024 |
| 16,384 | train | 1023/1,024 | 9.520538 | 0.002356 | 0.002433 | 978 | 1/1,024 |
| 16,384 | development | 0/1,024 | n/a | n/a | n/a | 0 | 1/1,024 |

## Exact distribution at the two exhaustively scored endpoints

Every legal identity was scored without any affinity lookup. Mass by split is
partly mechanical: split membership is a deterministic function of four of the
sixteen editable sites, so movement there can reflect block drift rather than
affinity learning. Per-site marginals and per-block mass are reported for that reason.

| Endpoint | Support mass | Entropy, nats | Effective support | Collision | Max atom | KL to SFT | KL from SFT | Total variation | Train mass | Development mass | Test mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.999999940 | 10.2889 | 29403.6 | 4.563e-05 | 1.741e-04 | 0.0000 | 0.0000 | 0.0000 | 0.6776 | 0.1918 | 0.1306 |
| 16,384 | 0.999999904 | 9.5037 | 13408.9 | 9.038e-05 | 2.632e-04 | 0.7672 | 2.5302 | 0.4678 | 0.9993 | 0.0004 | 0.0003 |

Allele-1 marginal per editable site. The four split-defining sites are marked.

| Site | Split-defining | 0 | 16,384 |
|---|---|---:|---:|
| 0 |  | 0.6321 | 0.6756 |
| 1 |  | 0.5472 | 0.5756 |
| 2 | yes | 0.5663 | 0.5298 |
| 3 |  | 0.5661 | 0.6384 |
| 4 |  | 0.5390 | 0.6338 |
| 5 | yes | 0.6236 | 0.6548 |
| 6 |  | 0.4948 | 0.5166 |
| 7 |  | 0.4839 | 0.5279 |
| 8 | yes | 0.7563 | 0.7282 |
| 9 |  | 0.9593 | 0.9632 |
| 10 | yes | 0.4365 | 0.3686 |
| 11 |  | 0.5614 | 0.4710 |
| 12 |  | 0.5804 | 0.4886 |
| 13 |  | 0.5034 | 0.4872 |
| 14 |  | 0.5022 | 0.5320 |
| 15 |  | 0.5449 | 0.4570 |

Exact mass per split-defining block. The block is the four split-defining digits,
so this is the finest partition the train/development/test split is a function of;
the three blocks the evaluated development pool draws from are marked.

| Block | Development block | 0 | 16,384 |
|---|---|---:|---:|
| 0 |  | 0.025269 | 0.025330 |
| 1 |  | 0.014899 | 0.035910 |
| 2 |  | 0.066379 | 0.000047 |
| 3 |  | 0.057703 | 0.147372 |
| 4 |  | 0.041287 | 0.084112 |
| 5 | yes | 0.024062 | 0.000023 |
| 6 |  | 0.109654 | 0.177009 |
| 7 | yes | 0.094458 | 0.000372 |
| 8 |  | 0.033106 | 0.000045 |
| 9 |  | 0.018949 | 0.034431 |
| 10 |  | 0.086820 | 0.102065 |
| 11 | yes | 0.073288 | 0.000027 |
| 12 |  | 0.055023 | 0.091724 |
| 13 |  | 0.031139 | 0.000225 |
| 14 |  | 0.145927 | 0.151097 |
| 15 |  | 0.122037 | 0.150210 |

## Conditional measured affinity

These means are conditional on identities that carry a usable measurement and are
inside the evaluated whitelist. They are not unconditional generated affinity. The
assay-SEM proxy assumes independent genotype errors and rests on a shared pooled
floor, so it is a heuristic scale, not a calibrated noise floor or a significance
test. Shared identities cancel in the paired delta, which is the headline column.
Training affinity is exposure, not generalization.

| Endpoint | Population | Measured mass | Conditional mean | Effective genotypes | Distinct genotypes | Assay-SEM proxy | Delta vs SFT | Paired SEM proxy | Mass above threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | train | 0.676175 | 9.462811 | 13258.2 | 38790 | 0.000641 | 0.000000 | 0.000000 | 0.415215 |
| 0 | development | 0.142020 | 9.488510 | 3520.2 | 8704 | 0.001248 | 0.000000 | 0.000000 | 0.103041 |
| 0 | development_block_5 | 0.017959 | 9.374410 | 1299.7 | 2678 | 0.002120 | 0.000000 | 0.000000 | 0.008492 |
| 0 | development_block_7 | 0.069390 | 9.518295 | 1420.5 | 3007 | 0.001956 | 0.000000 | 0.000000 | 0.060859 |
| 0 | development_block_11 | 0.054671 | 9.488188 | 1428.8 | 3019 | 0.001967 | 0.000000 | 0.000000 | 0.033691 |
| 16,384 | train | 0.998777 | 9.520988 | 11040.4 | 38790 | 0.000704 | 0.058176 | 0.000419 | 0.787887 |
| 16,384 | development | 0.000313 | 9.527577 | 1470.8 | 8704 | 0.001914 | 0.039067 | 0.001600 | 0.000252 |
| 16,384 | development_block_5 | 0.000018 | 9.519488 | 706.5 | 2678 | 0.002782 | 0.145078 | 0.001901 | 0.000014 |
| 16,384 | development_block_7 | 0.000275 | 9.529686 | 1154.3 | 3007 | 0.002161 | 0.011391 | 0.001693 | 0.000223 |
| 16,384 | development_block_11 | 0.000020 | 9.505376 | 946.3 | 3019 | 0.002374 | 0.017188 | 0.001445 | 0.000015 |

Probability mass by measurement availability, in six categories that partition the
support, plus their total. Measured train and measured development stay separate.
No missing outcome is imputed and no bias direction is asserted: eligibility mixes
assay-floor censoring with under-replication, and the eligible table records
survivors only. The conditional mean may therefore be biased in a direction and by a
magnitude this study does not establish.

| Endpoint | measured train | measured development | train ineligible | development ineligible | eligible development withheld | test | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.676175 | 0.142020 | 0.001392 | 0.000294 | 0.049493 | 0.130625 | 1.000000 |
| 16,384 | 0.998777 | 0.000313 | 0.000485 | 0.000000 | 0.000109 | 0.000317 | 1.000000 |

## Exposure overlap

The original SFT run drew its own labelled batches. Overlap is reported against
those identities, against this continuation's schedule prefix, and against the union.

| Updates | SFT-exposed identities | Continuation-exposed identities | Union | Draws hitting the union |
|---|---:|---:|---:|---:|
| 0 | 971 | 0 | 971 | 43/1,024 |
| 256 | 971 | 966 | 1826 | 88/1,024 |
| 2,048 | 971 | 5451 | 5877 | 330/1,024 |
| 8,192 | 971 | 9259 | 9310 | 767/1,024 |
| 16,384 | 971 | 9672 | 9674 | 813/1,024 |

## What was fixed before fitting

The [protocol](../specs/cr9114_budget_pressure.md) and code were committed at `7493364c3702` before any update. Budget is the only intervention: the
objective is the same unweighted NLL/16 over the same affinity-weighted training
population, with no reference KL, entropy or embedding penalty, the same frozen
encoder, the same seeded schedule, learning rate, weight decay and gradient clip.
One AdamW object serves all updates; evaluation runs between steps without an
optimizer and is asserted not to change decoder state, optimizer state or RNG.
The 256-update decoder reproduced the earlier affinity-only arm's state digest
`1619d9fe8ba4` exactly, which is the
reproduction claim; the checkpoint files differ because their metadata differs.
Checkpoints are fixed at 256, 2,048, 8,192 and 16,384 updates and all are reported.

The weighted training target has mean affinity 9.540963. That is a benchmark, not a
bound: it is the mean of the target distribution, and a policy concentrated on a
strong measured identity exceeds it. It is not a convergence criterion or a gate.

The best measured training identity is 9.793633. That value *is* an upper
bound, but only on a conditional average taken over these measured training
identities, since such an average is a weighted mean of measured values and cannot
exceed the largest of them. It is not a bound on conditional development affinity,
on affinity outside the measured set, or on what the model could reach at an
identity nobody has assayed.

## Validation and limits

The [independent audit](evidence/cr9114-budget-pressure-2026-09-17.json) re-derived the
reported statistics with its own formulas over 40 saved artifacts: the schedule and
the continuation exposures, the training population, the original SFT exposure
replay and its generator state, the development whitelist and pool, the exact
endpoint distributions and their normalization gate, the label-blind selections and
portfolios, and the sampled diversity, KL, Hamming, conditional-affinity, split and
exposure statistics with their standard errors. It does not re-derive recorded
metadata: library versions and the strict-reload, optimizer-carried-forward,
encoder-unchanged and RNG-invariance assertions were checked live during the run and
are read back here; timing fields are only checked for internal consistency; and
training itself is not replayed. Perturbing the 3,046 withheld development
measurements changed no reported number, and a synthetic test-labelled row is
rejected rather than filtered away.
The full test suite passed: 1826 passed, 3 skipped.

Monte Carlo KL and entropy remain estimates with sampling error at every
checkpoint, including the exhaustively scored endpoints, where they estimate the
exact value rather than equal it. The endpoint normalization check bounds float
accumulation at the endpoints; it does not bound error at the intermediate
checkpoints, which were never exhaustively scored. Intermediate pool statistics
are true unconditional mass on a 2,048-identity subset, not a distribution summary.

Only training measurements and the 8,704 previously evaluated eligible development
identities entered evaluation. The remaining 3,046 eligible development
measurements and every reserved test measurement stayed unused. This is a finite,
previously assayed 16-site edit space around one antibody lineage on one antigen;
nothing here is evidence about sequences outside that space, another lineage, or a
new binding assay. No checkpoint was promoted.

The model conditions on the [prepared 5CJQ context](cr9114-5cjq-context.md), which is
a different construct from the one the affinities were measured on: an engineered
stem trimer standing in for the assayed H1 ectodomain, and Fv geometry standing in
for the assayed scFv, with two declared VH template differences and fragment
separators where antigen coordinates are missing. No missing geometry was filled.
Affinities remain the original experimental measurements. The construct mismatch
limits conclusions about how well the structural conditioning transfers to the
assay construct.

[Compact evidence](evidence/cr9114-budget-pressure-2026-09-17.json) carries the full
result document, the independent audit and every artifact hash.
