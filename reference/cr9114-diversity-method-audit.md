# Diversity-method audit: ProteinDPO, ProteinZero and our pilot

2026-09-17. **The current entropy control is a valid anti-concentration baseline,
but it is not established as the best method for a diverse, high-affinity
shortlist.** Do not increase its coefficient or substitute an embedding loss
without a matched comparison. The main gap is alignment between the population
being diversified and the candidates being selected.

This audit reviews the implementation and existing results; it performs no new
training or assay-label evaluation. [Reproducible calculations](evidence/cr9114-diversity-strategy-audit-2026-09-17.json)
come from the saved, independently audited six-arm experiment. The
[audit script](../scripts/audit_cr9114_diversity_strategy.py) also checks analytic
counterexamples to interpreting distance as coverage.

## What the papers actually contribute

| Method | Relevant mechanism | What that establishes for this project |
|---|---|---|
| [ProteinDPO, published 2026 methods](https://www.nature.com/articles/s41592-026-03137-3) | Reference-relative paired, ranked, and scalar-label weighted DPO. Algorithms 2-4 contain no separate diversity loss. | A fitness-alignment comparator, not evidence that DPO guarantees diverse outputs. Its weighted DPO is different from our weighted SFT. |
| [ProteinZero v4, sections 3.1-3.2 and Table 3](https://arxiv.org/html/2506.07459v4) | Online reward optimization, explicit reference KL, and a separate cosine penalty on normalized, pooled decoder embeddings. Evaluation uses sequence Hamming distance. | A candidate method to test. Its embedding-versus-diversity-reward ablations do not compare against our exact sequence-entropy estimator. Its folding/stability objectives differ from affinity. |

For ProteinDPO, the distinction between scalar-label weighted DPO and weighted
SFT is substantive: the former compares a candidate-set distribution of
reference-relative scores to numerical-label targets; our implementation changes
the sampling distribution for ordinary likelihood training. Neither should be
described as a reproduction of the other.

## Findings from our code and measurements

1. **High priority: the diversity objective and affinity evaluation act on
   different populations.** The entropy loss covers the entire constrained
   policy. Affinity is measured on the top-ranked members of a fixed development
   pool. Higher global entropy can reward probability mass on poor candidates;
   it does not directly preserve variety among strong selected candidates.
   This is an objective limitation, not a coding error in entropy estimation.

2. **High priority: more entropy is not supported by the latest ablation.** Both
   no-entropy weighted-SFT runs already pass the two diversity gates. Adding
   entropy increases sampled mean Hamming by 0.2474 and 0.1455, but lowers top-16
   mean affinity by 0.0112 and 0.0138 relative to those same-seed plain controls.
   It improves top-32 means by 0.0011 and 0.0028. Two seeds cannot establish a
   statistical trade-off, but they do not justify claiming a free quality gain.
   The earlier DPO-collapse result should not be generalized to every objective.

3. **High priority: shortlist diversity is not uniformly preserved relative to
   SFT.** Weighted-plus-entropy top-32 shortlist Hamming is 4.1714 and 4.3327,
   below SFT's 4.4435, despite higher global sampled Hamming. Top-16 shortlist
   Hamming improves. This post-hoc diagnostic does not change the completed
   experiment's gates; it identifies a missing requirement for the next study.

4. **Medium priority: reference KL is monitored, not optimized.** The affinity
   runner's training loss contains likelihood and entropy; `kl_to_sft_mc` is
   computed after training. A measured KL is not a training constraint. The
   labelled likelihood target supplies an anchor of its own, but there is no
   explicit penalty to retain the SFT policy. Compare a reference-KL-only control
   before attributing gains to a more elaborate diversity term. KL also does not
   guarantee broad mode coverage by itself.

5. **Medium priority: coefficient units changed with the objective.** The nominal
   loss is `E[-log q / 16] - 0.03 H(q)`. Multiplying the whole objective by 16
   gives `E[-log q] - 0.48 H(q)`. Thus 0.03 is not a portable measure of strength
   across the old DPO loss and this per-site likelihood loss. This normalization
   was declared correctly; it is not an undisclosed implementation bug. Scaling
   a loss can also interact with Adam, clipping and numerical tolerances, so this
   algebra is not a claim of identical optimizer trajectories.

6. **Medium priority: uniqueness is close to its sample-size ceiling.** A uniform
   policy over 65,536 genotypes produces about 1,016.05 unique identities in 1,024
   draws on average. Counts of 1,001 versus 1,005 offer limited resolution and
   have sampling uncertainty. Our diagnostics already include joint entropy and
   collisions; the decision gate currently uses only uniqueness and mean Hamming.
   Retain the richer diagnostics rather than interpreting a few extra unique
   draws as a proven improvement.

7. **Implementation check passed.** The estimator re-scores fresh independent
   samples, retains duplicates, and uses a leave-one-out baseline. The sign and
   expected gradient are verified against exact categorical entropy; all nine
   entropy tests passed again during this audit. It estimates the gradient of
   joint sequence entropy, not a sum of independent marginal entropies or a
   softmax over the development candidates. There is no evidence here of an
   entropy-gradient bug.

Code locations: [entropy estimator](../src/smallAntibodyGen/experiments/diversity.py),
[affinity training and post-training KL](../scripts/run_cr9114_affinity_pilot.py),
[joint screening rule](../scripts/run_cr9114_diversity_pilot.py), and
[completed results](cr9114-affinity-pilot.md).

## Why changing the distance function alone is insufficient

The following are our own mathematical checks, not additional protein experiments.

For 16 binary variables, a distribution assigning equal probability to two
opposite strings has expected Hamming distance 8, exactly the same as the uniform
distribution over all 65,536 strings. Their entropies are log(2) and 16 log(2),
and their collision probabilities are 1/2 and 1/65,536. Hamming is informative
about marginal variability but cannot establish broad joint support on its own.

Similarly, two opposite unit vectors and four equally weighted unit vectors
around a circle both have zero expected cosine similarity between independent
draws: `E[z dot z'] = ||E[z]||^2`. Average cosine repulsion alone does not determine
the number of occupied modes. This does not negate empirical benefits of a
combined objective; it limits what one metric certifies.

A learned embedding distance is also a moving measurement when the same model
optimizes it. Our concern is a possible shortcut through representation changes,
not evidence that the published system exhibits that failure. For our fixed
scaffold, whole-chain pooling could dilute the effects of the 16 variable sites;
this needs a sensitivity check rather than an assumption. Use an independent,
frozen representation for validation. A frozen-feature training variant needs a
valid discrete-policy gradient; ordinary backpropagation through sampled residue
identities does not provide one.

## Recommended next comparison

**First define the desired portfolio: maintain a prespecified level of diversity
among acceptable candidates while improving affinity.** Treat excess randomness
as neither inherently useful nor a substitute for quality.

Before another training change, assess a quality-constrained shortlist selector
on a fixed candidate pool. Admission/ranking must use model scores or a predictor
fitted strictly on training data; held-out affinity labels are for evaluation
only. Compare ordinary top-K selection with diversity-aware selection inside
that admitted pool. This can test the actual portfolio requirement without
changing the generator. A predictor or selector still needs its own calibration
and validation; the current generator likelihood is not calibrated affinity.

For training, preregister matched controls: the chosen affinity objective alone,
reference KL alone, KL plus entropy, and KL plus an embedding-diversity term.
Distinguish a faithful trainable-embedding reproduction from a frozen-feature
adaptation. If affinity alignment is changed to scalar-label DPO, hold that
change fixed across diversity arms; do not change both mechanisms and then
attribute the outcome only to diversity. Normalize loss units consistently.

Evaluate top-K affinity and shortlist diversity on the **same selected sets**,
alongside collision rates, near-duplicate clusters, joint entropy and global
coverage. Declare an affinity-loss tolerance and diversity requirement before
fitting, report the trade-off, and use additional independent training seeds.
Our current two continuation seeds share one SFT checkpoint and one lineage.

The small 16-bit support also permits an optional exact policy audit over all
65,536 legal identities, without accessing any assay labels. This would remove
Monte Carlo uncertainty from entropy and collision estimates, at additional
scoring cost. No such exhaustive scoring was performed in this audit.

**Recommendation:** keep entropy as an anti-concentration baseline; prioritize
quality-conditioned shortlist evaluation and the missing reference-KL control.
Test embedding diversity as an alternative, not an already-proven replacement.
The present evidence cannot identify a universally best diversity method.
