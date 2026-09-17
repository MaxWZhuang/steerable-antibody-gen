# Affinity-weighted likelihood with explicit diversity control

Protocol fixed before fitting, 2026-09-16. This is a development experiment on
the protective CR9114 antibody's existing H1 measurements, using the verified
5CJQ proxy context and the same SFT initialization. No new assay, target, or
pathogen sequence modification is involved. Reserved test labels are never read.

## Question and objective

DPO improved broad pair ordering but failed to match SFT's top-16/top-32 affinity.
We now test emphasizing absolute high measured affinity within the existing SFT
positive population. This is a surrogate for selection quality, not a direct
differentiable top-K objective or a guarantee of better binding.

Only eligible TRAIN records enter the population builder. Select mean affinity
at least the training 75th percentile, as in the original SFT. Recompute pooled
within-genotype replicate variance v using degrees-of-freedom weights. For each
positive i use u_i = mean_i - sqrt(max(sample_variance_i, v)/replicate_count_i).
The subtraction is a heuristic uncertainty penalty, not a confidence guarantee.

Set T = sqrt(v), w_i = exp(max((u_i - max(u))/T, -log(20))), then
p_i = 0.75*w_i/sum(w) + 0.25/N. This bounds the raw weight ratio at 20 and keeps
a uniform component. The temperature and threshold use training data only.
No block balancing is added: the control is uniform over the original SFT pool,
and weighted block masses are reported. No affinity predictor is fitted.

Sample with replacement from p, then minimize **unweighted** mean -log q(x)/16.
Do not apply sampling weights again in the loss. The divisor is the 16 variable
sites, so the likelihood term is in nats per site. With entropy control the full
objective is E_p[-log q(x)/16] - 0.03 H(q). Entropy is in nats per full sequence;
the coefficient is not assumed equivalent in effect to the old DPO coefficient.
Use the already gradient-tested on-policy leave-one-out entropy estimator every
four updates, eight fresh samples each time, multiplying its gradient by four
to preserve the declared average coefficient. Do not join these samples to labels.

This borrows the weighted-likelihood principle from reward-weighted methods
([AWR, Peng et al.](https://arxiv.org/abs/1910.00177)); it does not implement AWR's
value learning, a full RL algorithm, or ProteinZero's embedding penalty.

## Fixed comparison

The JSON config fixes three arms and two seeds, all starting from the identical
SFT decoder with a fresh AdamW optimizer and frozen structure encoder:

1. Continued SFT: uniform positive sampling + entropy 0.03.
2. Affinity-weighted SFT: p sampling + entropy 0.03.
3. Affinity-weighted SFT: p sampling, entropy 0 (ablation).

Each arm gets 256 updates of four labelled examples, learning rate 1e-5,
weight decay 0.01, clipping 1.0, float32, dropout off. Save schedules before
training. Common random uniforms couple the control and weighted schedules;
weighted arms share exactly the same labelled examples. Entropy arms have equal
extra sampling budgets (512 unlabelled draws); the plain arm uses less compute.
The two seeds share one SFT model and are not independent pretraining replicates.
Deterministic CUDA/PyTorch operations, repeated encoding digest checks, strict
checkpoint reload, frozen-encoder checks and sample/rescore checks are required.

## Evaluation and decision rule

Before fitting, select 2,048 eligible development genotypes by a fixed identity
hash, excluding all 4,608 previously evaluated development identities. The same
cohort evaluates the initial SFT and every final-step arm. No intermediate
checkpoint selection or coefficient sweep. Test labels stay untouched; unlabelled
entropy draws may contain any identity in the constrained sequence space.

Primary quality metrics: measured top-16 and top-32 mean affinity after ranking
the fixed cohort by native log q, with genotype as deterministic tie-breaker.
Report per-block and leave-one-block-out sensitivity, selected-candidate overlap,
and uncertainty-adjusted selected means. These are descriptive summaries, not
independent-pair confidence intervals. Three development blocks from one lineage
do not establish broader generalization or exclude overfitting.

Diversity: 1,024 independent temperature-one policy draws per checkpoint, fixed
evaluation seed shared across models. Report unique count, unbiased mean Hamming
distance, Monte Carlo entropy and KL to the initial SFT. No assay join is performed
on generated samples; top-K quality concerns the held-out labelled candidate pool.

Keep the previous joint screen: both top-K means must at least match initial SFT,
and both unique count and mean Hamming must retain at least 80% of SFT. In addition,
the affinity+entropy arm must at least match its seed-matched continued-SFT+entropy
control on both top-K means and strictly exceed it on at least one. Require all
these conditions in BOTH seeds to claim this pilot supports the proposed change.
The plain ablation is reported but not eligible as the diversity-controlled method.
No automatic production checkpoint promotion, even if the descriptive screen passes.

An independent artifact audit recomputes primary metrics and screens from saved
CSV files and verifies hashes, cohort exclusion, schedules, and label isolation.
All six arms are reported, including negative results. No post-hoc threshold changes.
