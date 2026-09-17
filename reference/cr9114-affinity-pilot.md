# Affinity-weighted likelihood: CR9114 development pilot

The [fixed protocol](../specs/cr9114_affinity.md) tests whether emphasizing strong
measured binders improves the generator's top-candidate selection while retaining
explicit sequence-entropy control. The six declared arms are currently running;
no checkpoint has been promoted. Training source: `d092137`.

## How the change works

All arms start from the existing SFT checkpoint. The structure encoder stays
frozen; only the ESM-IF1 decoder trains. The measured training population contains
38,790 eligible records. The existing SFT top-quartile rule retains 9,698 of them.

For weighted training, each retained example receives a utility equal to its
measured mean H1 affinity minus its effective standard error. The effective
standard error uses the larger of the example's replicate variance and pooled
training replicate variance. This reduces emphasis on uncertain measurements;
it is not a calibrated confidence bound.

The sampling probabilities combine 75% normalized exponential utility weights
with 25% uniform sampling. Temperature is the training replicate SD, 0.12544;
raw weights have a maximum ratio of 20. The resulting distribution has effective
sampling population 9,085.4, probability ratio 11.24, and expected measured affinity
9.5410, compared with 9.5320 under uniform positive sampling. It is a modest shift.

Training minimizes unweighted negative log probability per variable site on
these sampled examples. The sampling probabilities already implement weighting:
the loss does not apply the weights again. The entropy-controlled arms also
maximize full-sequence entropy, using fresh unlabelled samples from the current
policy and the previously gradient-tested estimator. They do not use generated
samples' assay labels, a learned affinity predictor, or a structure-scoring reward.

Each seed compares uniform continued SFT plus entropy, weighted SFT plus entropy,
and weighted SFT without entropy. Both entropy arms have the same update and
sampling budgets. Each run has 256 updates of four labelled examples; every
fourth update uses eight unlabelled entropy samples. Weighted arms share the same
labelled-example schedule. The plain arm uses less sampling compute.

## How improvement is assessed

The initial SFT and every final checkpoint rank the same 2,048 previously
unevaluated development candidates by native generator probability. Their
measured top-16 and top-32 affinity means assess selection quality. This is not
an affinity prediction head, and these means do not measure the unconditional
generated population's affinity. Diversity uses 1,024 fresh temperature-one
policy draws per checkpoint, without looking up assay labels.

The proposed weighted-plus-entropy method must match or exceed initial SFT on
both affinity budgets, retain at least 80% of its unique-sample count and mean
Hamming distance, and improve over seed-matched continued SFT on at least one
budget without reducing the other. Both seeds must pass. Thresholds and training
settings stay fixed, and all outcomes will be reported.

The new cohort excludes all 4,608 previously evaluated development identities.
It still comes from the same antibody lineage and three development blocks;
fresh identities do not establish broad generalization or rule out overfitting.
Unlabelled entropy samples may coincide with held-out identities, but their
measurements are never used in training. Reserved test labels remain untouched.

## Reproduction and audit

Local input artifacts and ESM-IF1 weights must already be available as described
in the preceding experiment reports. The runner verifies their recorded hashes
and requires committed code and a fresh output directory.

```powershell
.\.venv\Scripts\python.exe -u scripts/run_cr9114_affinity_pilot.py --output outputs/cr9114_affinity_repeat
.\.venv\Scripts\python.exe scripts/audit_cr9114_affinity_pilot.py --run-dir outputs/cr9114_affinity_repeat
.\.venv\Scripts\python.exe -u scripts/replay_cr9114_affinity_pilot.py --run-dir outputs/cr9114_affinity_repeat
```

The independent artifact audit recomputes weights, schedules, candidate means,
selection sensitivity, diversity and pass/fail decisions from saved files. The
separate-process replay retrains the first weighted-plus-entropy arm and requires
bitwise equality of the final decoder, including its on-policy entropy updates.
