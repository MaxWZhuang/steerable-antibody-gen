# Matched reference-KL and diversity comparison

2026-09-17. All twelve predeclared runs completed; no checkpoint was promoted.

No regularizer retained or improved both ordinary top-16 and top-32 affinity against its direct matched control in all three seeds. This descriptive comparison does not establish a best regularizer.

## Measured results

All arms rank the same 2,048 development candidates. Within each portfolio,
affinity and diversity are computed on the same selected identities.
Higher measured H1 affinity is better; Hamming counts differing editable sites.

| Seed | Arm | Ordinary top-16 affinity | Ordinary top-32 affinity | Top-32 Hamming | Diverse top-32 affinity | Diverse top-32 Hamming |
|---|---|---:|---:|---:|---:|---:|
| - | SFT | 9.572041 | 9.559854 | 4.2964 | 9.537314 | 5.6996 |
| 20260925 | affinity | 9.584705 | 9.577684 | 4.0403 | 9.555556 | 5.4375 |
| 20260925 | kl | 9.585300 | 9.577616 | 4.0081 | 9.553767 | 5.3851 |
| 20260925 | kl_entropy | 9.587622 | 9.580566 | 4.0141 | 9.559769 | 5.3609 |
| 20260925 | kl_embedding | 9.585037 | 9.577616 | 4.0081 | 9.554786 | 5.3609 |
| 20260926 | affinity | 9.574855 | 9.573260 | 4.0040 | 9.567738 | 5.4194 |
| 20260926 | kl | 9.579211 | 9.572980 | 4.0423 | 9.555272 | 5.2238 |
| 20260926 | kl_entropy | 9.571845 | 9.574210 | 4.0968 | 9.552639 | 5.2379 |
| 20260926 | kl_embedding | 9.576503 | 9.574029 | 4.0544 | 9.565262 | 5.2621 |
| 20260927 | affinity | 9.574168 | 9.579290 | 3.9677 | 9.546773 | 5.2782 |
| 20260927 | kl | 9.572653 | 9.578611 | 3.9859 | 9.554763 | 5.1915 |
| 20260927 | kl_entropy | 9.572295 | 9.576439 | 3.9919 | 9.551456 | 5.2661 |
| 20260927 | kl_embedding | 9.572653 | 9.578611 | 3.9859 | 9.554763 | 5.1915 |

For the effect of adding a diversity term, compare against the same-seed KL-only
control. The following are descriptive paired differences, not significance tests.

| Seed | Added term | Top-16 affinity delta vs KL | Top-32 affinity delta vs KL | Top-32 Hamming delta vs KL |
|---|---|---:|---:|---:|
| 20260925 | entropy | +0.002322 | +0.002950 | +0.0060 |
| 20260925 | embedding | -0.000263 | +0.000000 | +0.0000 |
| 20260926 | entropy | -0.007366 | +0.001230 | +0.0544 |
| 20260926 | embedding | -0.002707 | +0.001049 | +0.0121 |
| 20260927 | entropy | -0.000358 | -0.002173 | +0.0060 |
| 20260927 | embedding | +0.000000 | +0.000000 | +0.0000 |

## Sampling and representation diagnostics

These concern unconditional policy samples, not just the selected high-score sets.
Entropy and KL are Monte Carlo estimates; the evidence includes standard errors.
A learned cosine improvement needs corroboration from fixed features and sequences.

| Seed | Arm | Unique / 1,024 | Entropy, nats | Hamming | KL to SFT, nats | Frozen cosine | Live cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| - | SFT | 1004 | 10.2792 | 7.3146 | 0.0000 | 0.999426 | 0.999426 |
| 20260925 | affinity | 998 | 10.2436 | 7.2653 | 0.0703 | 0.999423 | 0.999402 |
| 20260925 | kl | 998 | 10.2720 | 7.2910 | 0.0557 | 0.999419 | 0.999398 |
| 20260925 | kl_entropy | 1004 | 10.3855 | 7.3883 | 0.0596 | 0.999415 | 0.999403 |
| 20260925 | kl_embedding | 997 | 10.2723 | 7.2915 | 0.0560 | 0.999420 | 0.999393 |
| 20260926 | affinity | 1008 | 10.2209 | 7.2768 | 0.1777 | 0.999414 | 0.999349 |
| 20260926 | kl | 1005 | 10.2432 | 7.3023 | 0.1554 | 0.999412 | 0.999353 |
| 20260926 | kl_entropy | 1007 | 10.3921 | 7.4143 | 0.1472 | 0.999403 | 0.999362 |
| 20260926 | kl_embedding | 1005 | 10.2430 | 7.3025 | 0.1537 | 0.999412 | 0.999344 |
| 20260927 | affinity | 994 | 10.2033 | 7.2507 | 0.1145 | 0.999423 | 0.999393 |
| 20260927 | kl | 997 | 10.2167 | 7.2643 | 0.1043 | 0.999419 | 0.999388 |
| 20260927 | kl_entropy | 999 | 10.2993 | 7.3214 | 0.0998 | 0.999413 | 0.999395 |
| 20260927 | kl_embedding | 997 | 10.2157 | 7.2630 | 0.1034 | 0.999418 | 0.999375 |

A post-hoc gradient-scale probe at the first final embedding checkpoint gave
pre-clipping norms NLL/site=0.658683, scheduled KL=0.110819,
and scheduled embedding=0.00124008. The regularizer norms
include their coefficients and four-update multiplier. This uses one fixed
eight-sample batch and one labelled training minibatch, changes no parameters,
and is not a typical-gradient estimate or a coefficient-tuning procedure.
The small embedding contribution in this probe limits a negative result:
this does not rule out differently scaled or differently pooled embedding
variants. Gradient norms also do not directly measure Adam update magnitudes.

## How the comparison isolates the mechanism

The [protocol](../specs/cr9114_shortlist.md) was committed before the new shortlist
development result. Each arm starts from the same SFT checkpoint, with a frozen
encoder and fresh AdamW optimizer. Within each seed, the four arms receive the
same 1,024 labelled sequence exposures over 256 updates. Only the regularizer
changes. All use the existing training-only affinity-weighted likelihood.

Comparisons against SFT include 256 additional optimization steps. Only the
within-seed matched contrasts isolate the regularizer; this study does not
re-establish the benefit of affinity weighting over uniform continuation.

The objective is NLL/16, plus 0.1*KL(q||SFT)/16 when enabled. The entropy arm
also subtracts 0.1*H(q)/16. The embedding arm instead adds 0.1 times mean
off-diagonal cosine of normalized mean-pooled last decoder features. Thus the
entropy coefficient here is 0.00625 per joint-sequence nat, smaller than the
previous pilot's 0.03. These are fixed starting coefficients, not an optimized
or equal-gradient-strength comparison.

Eight fresh samples every four updates supply KL and entropy score-function
gradients; leave-one-out baselines and duplicate samples are retained. The
gradients are multiplied by four to account for update frequency. A separate
frozen SFT decoder supplies reference probabilities. The embedding penalty
differentiates the trainable representation of detached generated identities;
it is not a frozen-feature reward and does not include a distribution-gradient
term through the discrete sample. All 121 valid causal prediction contexts
are pooled, including fixed scaffold positions.

This is a [ProteinZero-inspired](https://arxiv.org/html/2506.07459v4) adaptation,
not a reproduction of its full online reward/GRPO system. The affinity objective
remains weighted SFT; it is not [ProteinDPO's](https://www.nature.com/articles/s41592-026-03137-3)
scalar-label reference-relative DPO objective.

## Validation and limits

The independent audit verified 80 output artifacts and all twelve
checkpoint identities, independently recomputed schedules, selections, affinity,
sequence metrics, reference KL and both embedding metrics. A separate-process
replay of the first KL+embedding arm reproduced the final decoder bit for bit.
The full test suite passed: 1786 passed, 3 skipped (13 warnings).
Native-model tests check feature-path score/gradient parity, and an evaluation
test confirms that changing development labels cannot change selected identities.

This cohort was reused from the completed shortlist study, whose combined
quality/diversity screen failed at K=32. The admission rule remains locked to
the top 4*K model scores. It was calibrated on SFT only; applying that fixed
window to adapted policies is a transfer test, not fresh affinity calibration.
Test measurements remain reserved, and no additional
fresh development measurements were consumed by training comparison. No
checkpoint, seed or coefficient was selected from these results. The three
continuation seeds share one SFT initialization, one antibody lineage, and
three development blocks; this is not evidence against all forms of overfitting.
A post-hoc audit decomposes shortlist Hamming into the four split-defining
sites and the other twelve sites, and reports within-block pair distances.
This checks whether block mixing alone explains diversity; it changes no gate.
Frozen SFT features provide a fixed numerical diagnostic, not an independent
biological assay. Uniqueness is near its sample-size ceiling and connected
sequence clusters are not validated biological modes.

Affinity was evaluated by ranking a fixed measured pool; unconditional sample
affinity was not evaluated and no new binding assay was performed. The 5CJQ
engineered stem context remains a structural proxy for the measured landscape.

[Compact evidence](evidence/cr9114-regularization-2026-09-17.json) includes
per-seed results, selected identities, paired comparisons, hashes and audits.
