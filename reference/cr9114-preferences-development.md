# CR9114 preference pairs and development evaluation

Completed 2026-09-16 using the existing parent and 256-step SFT pilot scores.
No additional model training or test-label evaluation was performed.

**Subsequent work:** both [bounded DPO pilots](cr9114-dpo-pilot.md) have now
completed using these frozen pairs. The pre-DPO checklist below records the
handoff from this construction/evaluation stage; multi-seed and confirmatory
evaluation remain outstanding.

- [Explicit pair-selection configuration](../configs/experiments/cr9114_preferences.json)
- [Reproducible builder and evaluator](../scripts/prepare_cr9114_preferences.py)
- [Source hashes, audits and complete results](evidence/cr9114-preferences-development-2026-09-16.json)

## What was necessary

The existing genotype split, raw H1 replicate measurements, fixed structure/site
mapping, and saved development scores were sufficient. Pair construction adds
an explicit uncertainty rule and bounded variant reuse. Development evaluation
compares identical pairs and accounts for their shared variants in the metric;
it does not treat thousands of overlapping pairs as independent experiments.

Only eligible training measurements fit the variance regularizer. The original
SFT positive-only cohort is not reused as the preference universe: comparisons
use all 38,790 eligible training variants, including weaker binders.

## Fixed construction rule

Keep the pilot's requirement of at least two finite replicates, all observed
replicates above the 7.0 censoring floor. Recompute means and sample variances.
The pooled within-genotype variance fitted on eligible training data is
0.0157342 (SD 0.125436). For each variant, use
`effective_SEM = sqrt(max(sample_variance, pooled_training_variance) / n)`.
Zero-spread replicates therefore do not imply certain measurements.

Accept an ordering only when the higher mean exceeds the lower by both **0.1**
log10 units and **2 times the combined effective SEM**, combined in quadrature.
These settings were written before calculating pair accuracies. This is a
heuristic uncertainty screen; it is not a calibrated 95% hypothesis test or a
simultaneous confidence guarantee. It does not model covariance between assays.

Pairs stay within the same split **and the same four-locus genotype block**.
The latter means this first pair task measures ordering while the four split
loci are fixed; it does not evaluate rankings between blocks. No pair crosses
training/development, and reserved test rows are removed before aggregation.

Seeded random matchings propose candidates, independently of model scores.
Accept at most eight distinct pairs per variant over 64 rounds. Some variants
have no accepted partner; that can reflect uncertainty, available gaps or the
bounded proposal budget, and does not prove no reliable partner exists.

Pair weights average per-variant ordering accuracy within each represented
block, then give each block equal weight. A pair receives
`(1/degree_chosen + 1/degree_rejected) / (represented_variants_in_block * block_count)`.
Weights sum to one separately for training and development. Prediction ties
within 1e-6 receive half credit.

## Constructed data

| Split | Pairs | Represented variants | Candidate variants | Blocks |
|---|---:|---:|---:|---:|
| Training | 114,395 | 38,066 | 38,790 | 10 |
| Development | 1,399 | 498 | 512 previously scored | 3 |

The verified handoff is `outputs/cr9114_preferences_verified_20260916/`:

- `training_pairs.csv`: chosen/rejected genotypes, measurements, uncertainty,
  block, endpoint degrees, and normalized pair weights.
- `development_pairs.csv`: separate frozen evaluation comparisons.
- `development_pair_scores.csv`: parent, SFT and additive model correctness on each pair.
- `eligible_non_test_records.csv`: audit data, containing training and development;
  **not a training-loader input**.
- `manifest.json`: configuration, input/output hashes, coverage and evaluation.

Read genotype fields as strings to preserve leading zeros. Recover antibody
sequences through the prepared artifact's validated 16-site edit space.
Pair files contain antibody genotype decisions, not antigen sequence designs.

## Development results

| Model | Variant/block-balanced pair accuracy | Mean measured H1 value among top 16 | Among top 32 |
|---|---:|---:|---:|
| Parent | 66.16% | 9.4579 | 9.4718 |
| SFT pilot | 89.21% | 9.5603 | 9.5528 |
| Additive ridge | 97.57% | 9.4873 | 9.5021 |

Higher measured H1 values indicate stronger binding. Top-K uses the same 512
development candidates across blocks, whereas pair ordering is within blocks;
these are different endpoints. SFT improves both relative to its parent. The
additive baseline wins reliable pair ordering, while SFT selects a stronger
measured top group on this development pool. Neither result is a final-test claim.

SFT's pair accuracy improves by **23.05 percentage points** over the parent and
trails the additive baseline by **8.36 points**. Its mathematical distance from
perfect pair accuracy is **10.79 points**. That leaves room for a DPO pilot;
it does not predict that DPO will realize the gain or establish statistical power.

SFT per-block accuracies are 82.11%, 92.25% and 93.28%. Leaving one of the three
blocks out gives 87.18%-92.77%; this range is a sensitivity summary, **not a
confidence interval**. Three blocks and one training seed are insufficient for
a strong confirmatory uncertainty assessment. No independent-pair binomial
interval is reported.

## What remains before DPO training

1. Implement and test the DPO objective using this policy's constrained `log q`,
   including preference direction and frozen-reference gradients.
2. Cache reference scores for the unique **training-pair** genotypes. Use the
   parent reference for direct DPO and the SFT reference for SFT-to-DPO. Tie each
   cache to the checkpoint, geometry, genotype order and probability convention.
3. Consume the declared pair weights exactly once: weighted objective or sampling
   proportional to the weights. Uniform unweighted training changes the objective.
4. Measure a bounded DPO step, choose development-only settings and budgets, and
   compare pair ordering plus measured top-K before scaling to multiple seeds.
5. Keep the final test reserved and finalize a confirmatory uncertainty protocol
   before making held-out improvement claims.

## Verification and reproduction

47 tests passed across preference construction, pilot data rules and existing
regressions. A second execution reproduced all four CSV artifacts byte-for-byte.
An independent reread checked split membership, unordered-pair uniqueness,
endpoint caps, uncertainty thresholds and normalized weights. Source replicate
summaries and the training-only additive baseline were independently reproduced.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_cr9114_preferences.py --output-dir outputs/cr9114_preferences_reproduction
```

Use a new output directory. The script reads local pinned inputs and loads no
model weights. Its source and configuration hashes are recorded even when run
from an uncommitted worktree; the evidence explicitly records that state.
