# CR9114 overfitting and diversity diagnostics

Protocol fixed before running on 2026-09-16. This is a post-hoc investigation of
the [DPO pilot tradeoff](cr9114-dpo-pilot.md), not a new confirmatory experiment.
No additional training, checkpoint selection or reserved-test evaluation.

The [diagnostic runner](../scripts/diagnose_cr9114_dpo.py) will:

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
