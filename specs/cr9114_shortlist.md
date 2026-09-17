# Locked shortlist comparison and matched training follow-up

2026-09-17. This protocol is committed before new development evaluation.
The fixed CR9114/5CJQ SFT checkpoint supplies ranking scores. Its likelihood is
not calibrated affinity. No antigen sequence is optimized.

## Shortlist comparison

Use four disjoint 2,048-member training calibration cohorts, selected by one
identity hash ordering (seed 20260923). Score all with the unchanged SFT model.
Compare ordinary top-16/top-32 with greedy max-min Hamming selection inside the
top m*K scores for m in {1,2,4,8}. Start with the highest-scoring candidate;
ties use higher score then lexicographic identity. m=1 is ordinary top-K.

Choose one m across both budgets: it must lose no more than 0.02 H1 assay units
relative to ordinary selection in **every** training-cohort/budget cell, then
maximize mean Hamming gain (tie: smaller m). The 0.02 tolerance is a declared
engineering trade-off, not a validated biological noninferiority margin.
Training calibration is not independent validation of the trained model.

Save and hash the choice before scoring or evaluating 2,048 fresh development
identities (seed 20260924), excluding all four previously used development
cohorts. Only identity and model score may enter the selector. Held-out assay
labels are joined only after selections are frozen. Evaluate affinity, affinity
minus effective SEM (a heuristic), mean/minimum Hamming, collision fraction,
Hamming<=1 connected components, and the same metrics among selected candidates
above the training-only upper-quartile affinity threshold. Components are not
biological modes. Scores and all selected identities remain auditable.

A screening pass requires affinity loss <=0.02 and mean Hamming gain >=0.25 at
both K, with no increase in the fraction of pairs at Hamming<=1. Report all
measurements, including failures. This one-cohort screen is descriptive; no
statistical or biological noninferiority claim and no checkpoint promotion.
Never adjust the choice after reading development results. Test labels stay
reserved. The chosen selector will also be reported for all matched training
arms below, using this same cohort; that follow-up is **reused development**.

## Matched training comparison (declared before the shortlist result)

Keep the existing affinity-weighted likelihood population and weights unchanged.
From the same SFT checkpoint, use seeds 20260925, 20260926, 20260927 and four arms:
affinity alone; affinity+reference KL; affinity+KL+entropy; affinity+KL+learned
embedding cosine. Each has 256 steps, four labelled sequences/step, AdamW 1e-5,
weight decay .01, gradient clipping 1, dropout off, deterministic float32.
Matched labelled schedules within each seed; encoder frozen; new optimizer.

Loss units: NLL/16 + 0.1*KL(q||SFT)/16 - 0.1*H(q)/16, including only terms
present in the arm. The embedding arm substitutes +0.1*mean off-diagonal cosine
for the entropy term. Coefficients are fixed engineering starting values,
**not** imported paper hyperparameters or claimed gradient-strength matches.
Fresh eight-sample on-policy batches every four updates, with regularization
gradients multiplied by four. KL and entropy use score-function estimators
with leave-one-out baselines, keeping duplicates.

The embedding arm is a ProteinZero-inspired adaptation: normalized mean-pooled
last decoder features, direct gradients through the trainable representation of
detached generated sequences. It is not a frozen-feature reward or a faithful
reproduction of ProteinZero's full reward/GRPO system. Include all 121 valid
decoder prediction positions, no padding. Native ESM-IF1 uses shifted causal
prefixes, so these represent prediction contexts, not bidirectional residue
embeddings. Evaluate sequence distances independently and compare frozen-SFT
embeddings to detect improvements confined to the trainable representation.

Use the same locked shortlist rule and ordinary top-K for all arms. Report
1,024 unconditional samples (fixed seed 20260928), sequence entropy and
collision diagnostics, MC reference KL, and frozen-representation similarity.
Quality/diversity must be checked on the same selected sets. Report each seed;
do not select a winning seed. This fixed comparison identifies evidence at
these coefficients, not an optimal regularizer or proof against overfitting.
