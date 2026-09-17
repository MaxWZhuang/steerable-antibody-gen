# CR9114 overfitting and diversity diagnostics

**Completed 2026-09-16.** Protocol committed before running at `da2a5d3`.
This is a post-hoc investigation of
the [DPO pilot tradeoff](cr9114-dpo-pilot.md), not a new confirmatory experiment.
No additional training, checkpoint selection or reserved-test evaluation.

The [diagnostic runner](../scripts/diagnose_cr9114_dpo.py) performed these checks:

- Evaluate each DPO checkpoint at steps 64, 128, 192 and 256 against its initial
  reference. Report raw-policy pair accuracy and reference-relative DPO loss on
  the fixed 512-pair training schedule and the existing development pairs. Also
  distinguish training pairs already exposed at each checkpoint from future
  entries in that fixed schedule. Report development correlation and top-K.
- Choose 2,048 additional eligible development genotypes by a deterministic
  sequence hash (seed 20260917), excluding all 512 original candidates. Construct
  pairs with the unchanged training-derived variance floor and preference rule.
  Evaluate parent, SFT and both final DPO models, plus the existing train-only
  additive ridge baseline. These are new candidates in the same three blocks;
  they are not an independent target or final test set.
- Draw 1,024 independent temperature-one samples per final model using its
  actual autoregressive policy. Report unique counts, duplicate rate, unbiased
  collision and mean Hamming estimates, marginal allele frequencies, and
  Monte Carlo entropy from mean negative log probability. Never substitute a
  softmax over the development pool for the full policy distribution. Estimate
  KL to the actual reference by rescoring adapted-policy samples. Sampling uses
  the entire 65,536-genotype space but never joins samples to test assay labels.
- Verify saved-score parity and sample/teacher-forced scoring parity. Record
  checkpoint and output hashes. Inspect distance to training identities to
  distinguish absence of duplicates from distance-separated generalization.

Monte Carlo standard errors concern sampling from these fixed models only.
They do not quantify training-seed variation or experimental uncertainty. Pair
dependencies and only three development blocks prohibit treating pair counts
as independent sample sizes. A larger gap between training and development
does not by itself establish overfitting when those populations differ.

## Results and interpretation

The [complete evidence](evidence/cr9114-dpo-diagnostics-2026-09-16.json) records
checkpoint curves, per-block metrics, sample statistics, hashes and an independent
recalculation of fresh-cohort accuracy, top-16 means, entropy and KL estimates.
Local CSVs are in `outputs/cr9114_dpo_diagnostics_20260916/`.

**The preference-ordering gains replicate, but DPO sharply contracts diversity
and still selects weaker top candidates than SFT.** These checks do not establish
classic training-set overfitting, nor do they rule it out beyond this narrow
development distribution. The results are consistent with objective mismatch or
overoptimization relative to the strongest-candidate selection goal. Those are
interpretations, not an identified causal mechanism.

Fresh evaluation uses 2,048 additional candidates and 5,543 pairs involving
1,970 of those candidates. Top-K selection uses all 2,048 candidates.

| Model | Fresh balanced pair accuracy | Fresh Spearman | Fresh top-16 mean H1 | Fresh top-32 mean H1 |
|---|---:|---:|---:|---:|
| Parent | 64.60% | 0.277 | 9.5068 | 9.4861 |
| SFT | 89.08% | 0.796 | 9.5584 | 9.5535 |
| Direct DPO | 93.85% | 0.775 | 9.5032 | 9.4930 |
| SFT-to-DPO | 96.13% | 0.778 | 9.4771 | 9.4961 |
| Additive ridge | 97.35% | 0.817 | 9.4259 | 9.4391 |

Top-K means are descriptive; there is no claim that their differences exceed
experimental uncertainty. As before, the additive baseline is strongest on
pair accuracy while SFT is strongest on top-candidate selection. The preference
screen omits close affinity comparisons, and pairs are constructed within blocks
while top-K selection spans blocks. Either distinction can contribute to the
metric mismatch; neither is isolated by this diagnostic.

All sample statistics below use **1,024 temperature-one draws per model**.
Entropy-equivalent support is exp(E[-log q]), estimated from genuine on-policy
samples. It is neither a count of all possible sequences nor a count of modes.

| Model | Unique draws | Duplicate fraction | Entropy (nats, MC SE) | Entropy-equivalent support | Mean pairwise Hamming |
|---|---:|---:|---:|---:|---:|
| Parent | 626 | 38.87% | 7.322 (0.053) | 1,513 | 4.94 |
| SFT | 1,007 | 1.66% | 10.282 (0.028) | 29,191 | 7.31 |
| Direct DPO | 215 | 79.00% | 4.764 (0.055) | 117 | 3.09 |
| SFT-to-DPO | 231 | 77.44% | 4.723 (0.064) | 112 | 2.95 |

SFT-to-DPO reduces entropy-equivalent support approximately 260-fold relative
to SFT. Its most frequent sampled genotype accounts for 11.13% of draws. Direct
DPO reduces support approximately 13-fold relative to its parent reference.
This is clear distributional concentration, or partial collapse-like behavior;
neither model has collapsed to a single sequence. Concentration alone does not
prove pathological mode collapse: preference optimization is expected to move
probability toward favored variants, and generated-sample biological quality
was not evaluated here. Nevertheless, it is a material loss if diverse designs
are part of the objective.

Monte Carlo KL(q || reference), in nats, is 4.537 (SE 0.074) for direct DPO
against parent and 4.903 (SE 0.061) for SFT-to-DPO against SFT. SFT itself is
8.295 (SE 0.201) from parent. KL magnitude alone does not diagnose collapse:
SFT becomes substantially more diverse despite moving further from parent.

The fixed training-schedule pool and original development DPO losses both fall
through all saved checkpoints. Each arm starts at log(2) = 0.6931 against its
own reference. Training entries are equally weighted schedule draws from the
declared weighted sampler; development retains variant/block weights.

| Arm | Step | Training-pool loss | Development loss | Development top-16 mean H1 |
|---|---:|---:|---:|---:|
| Direct DPO | 64 | 0.3875 | 0.3943 | 9.5329 |
| Direct DPO | 128 | 0.1803 | 0.1520 | 9.5187 |
| Direct DPO | 192 | 0.1489 | 0.1250 | 9.5270 |
| Direct DPO | 256 | 0.1370 | 0.1048 | 9.5266 |
| SFT-to-DPO | 64 | 0.3062 | 0.3163 | 9.5424 |
| SFT-to-DPO | 128 | 0.1804 | 0.1510 | 9.5071 |
| SFT-to-DPO | 192 | 0.1596 | 0.1253 | 9.5154 |
| SFT-to-DPO | 256 | 0.1389 | 0.0988 | 9.5118 |

Thus, there is no observed training-loss/development-loss divergence across
these checkpoints. Development selection quality does not track that loss.
These fixed-pool training summaries include future scheduled exposures at early
checkpoints; the evidence also reports already-exposed pairs separately.

The identity audit found zero overlap between development and the 1,007 exposed
DPO training variants. Of the fresh variants, 184 have a nearest exposed DPO
training neighbor at Hamming distance one, 1,143 at two, 717 at three, and four
at four. Against the entire eligible training pool, 2,046 are distance one and
two are distance two. The latter pool includes variants that were not sampled
for optimization. This is absence of exact-identity leakage, not evidence of
distance-separated or cross-lineage generalization.

## Consequences and remaining work

Retain SFT as the current candidate-selection baseline; do not promote either
DPO checkpoint based only on pair accuracy. Before a larger DPO run, declare
joint acceptance criteria for top-candidate quality and diversity, then examine
more conservative updates and reference regularization in a bounded development
comparison. Repeat training across seeds. Generalization beyond close sequence
neighbors requires a separately planned distance-separated or independent-target
evaluation. These were proposed next steps at the time of this diagnostic.

Follow-up on 2026-09-16: the [two-seed entropy-control comparison](cr9114-diversity-pilot.md)
has now completed. It restores much of the lost diversity but does not pass the
joint affinity/diversity screen. Independent-target and distance-separated
evaluations remain outstanding; the two DPO seeds share one SFT initialization.

The targeted diagnostic, DPO-runner and objective tests passed: **33 tests**.
Saved-score and sample/teacher-forced parity checks passed for all four models.
No new training, checkpoint promotion or reserved-test label evaluation occurred.

## Research motivating the checks

[Rafailov et al., NeurIPS 2024](https://arxiv.org/abs/2406.02900) demonstrate
overoptimization in direct alignment, including deterioration before a complete
training epoch. Consequently, few update steps cannot rule it out. Their study
also motivates measuring actual policy divergence and task quality rather than
relying on the preference objective alone.

[Kirk et al., ICLR 2024](https://arxiv.org/abs/2310.06452) find a generalization
and diversity tradeoff in RLHF: generalization can improve while output diversity
falls. Their experiments concern language models and RLHF; they do not diagnose
our antibody model or establish that DPO necessarily collapses.

[Yang et al., 2024](https://arxiv.org/abs/2409.14836) investigate DPO failure
modes including reduced generation diversity. These papers justify separate
checks for generalization, selection quality and distributional concentration.
They are not biological validation of this experiment.
