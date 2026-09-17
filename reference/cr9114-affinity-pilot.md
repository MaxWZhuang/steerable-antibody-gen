# Affinity-weighted likelihood: CR9114 development pilot

**All six declared arms completed on 2026-09-17.** Affinity weighting with entropy
improved top-32 mean affinity in both seeds and retained sampling diversity, but
top-16 affinity regressed in the second seed. The required two-seed screen fails.
**SFT remains the baseline; no checkpoint is promoted.** Training source: `d092137`.

The [fixed protocol](../specs/cr9114_affinity.md) was committed before fitting.
It tests a surrogate for better top-candidate selection, not a guarantee of higher
binding affinity. The [complete evidence](evidence/cr9114-affinity-pilot-2026-09-17.json)
contains every arm, artifact audit, numerical replay and validation record.
Local artifacts are under `outputs/cr9114_affinity_pilot_20260916/`;
the directory date records when the experiment started.

## Completed measurements

Affinity columns are measured H1 means among model-ranked candidates in the same
fresh development pool. Unique counts and Hamming distances use 1,024 unlabelled
policy draws, not those ranked shortlists.

| Model | Seed | Top-16 mean H1 | Top-32 mean H1 | Unique samples | Mean Hamming | SFT joint screen |
|---|---:|---:|---:|---:|---:|---|
| Initial SFT | shared | 9.5784 | 9.5589 | 1001 | 7.3406 | baseline |
| Continued SFT + entropy | 20260920 | 9.5584 | 9.5583 | 1013 | 7.5936 | fail: affinity |
| Affinity-weighted + entropy | 20260920 | 9.5793 | 9.5744 | 1005 | 7.5560 | pass |
| Affinity-weighted, no entropy | 20260920 | 9.5905 | 9.5734 | 1001 | 7.3087 | pass |
| Continued SFT + entropy | 20260921 | 9.5671 | 9.5549 | 1005 | 7.4306 | fail: affinity |
| Affinity-weighted + entropy | 20260921 | 9.5477 | 9.5615 | 1005 | 7.5442 | fail: top 16 |
| Affinity-weighted, no entropy | 20260921 | 9.5615 | 9.5587 | 1004 | 7.3987 | fail: affinity |

The first weighted-plus-entropy seed also beats its matched continued-SFT control
on both budgets. The second improves top 32 but reduces top 16 relative to that
control and the original SFT. Its top-16 reduction versus SFT is 0.0307, while the
first seed's increase is only 0.0009. Selecting the successful seed would conceal
the declared repeatability failure. These differences are descriptive; no assay
significance or generalization claim follows from them.

All six arms pass both sampling-diversity thresholds. There is no recurrence of
the earlier DPO concentration on these metrics. Within weighted training, entropy
increases Hamming distance in both seeds, but the no-entropy arm has higher top-16
affinity in both. The plain arm also fails in the second seed, so removing entropy
does not solve the top-candidate instability. This experiment does not establish
that an entropy term is necessary for this likelihood objective.

A separately labelled **post-hoc** diagnostic checks diversity of the selected
shortlists. Weighted-plus-entropy top-16 mean Hamming distances are 3.9417 and
4.0583, versus SFT's 3.4667. Its top-32 distances are 4.1714 and 4.3327, versus
SFT's 4.4435. Sampling diversity and shortlist diversity are different quantities;
the latter is not uniformly improved and was not added to the decision rule.

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
settings stayed fixed, and all six outcomes are reported above.

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

The artifact audit passed: all 25 recorded CSV/schedule hashes and six checkpoint
hashes match, and the primary metrics, sensitivity summaries, sample counts and
decision gates reproduce. Training histories are finite; all runs passed strict
checkpoint reload, sample/rescore parity and frozen-encoder checks. Maximum
evaluation sample/rescore error is 4.77e-6; checkpoint reload error is zero.
The first weighted-plus-entropy arm reproduced its final decoder **bit-for-bit**
in a separate process, including deterministic geometry encoding and all 64
on-policy entropy updates.

Targeted checks passed **33 tests**. The isolated full suite passed **1,762 tests,
with three skips and 13 warnings**. An earlier concurrent full-suite attempt
aborted with a native Windows access violation; the isolated rerun passed without
a code change. The abort's root cause is not established.

The six training arms took about 585 seconds in total, excluding model loading,
evaluation and the additional replay. Peak PyTorch CUDA allocation was about
2,083 MiB, excluding desktop/driver memory. The entropy arms use more compute
than the plain ablation. Their unlabelled training draws overlapped with 9-17
fresh-development identities per arm, but no corresponding measurements were
used in training.
