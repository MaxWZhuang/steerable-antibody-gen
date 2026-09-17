# CR9114 / 5CJQ development pilot

The user authorized model scoring and training on 2026-09-16. The bounded first
run uses [this explicit configuration](../configs/experiments/cr9114_5cjq_pilot.json)
and [this runner](../scripts/run_cr9114_esmif1_pilot.py). It is a supervised
development pilot, not the confirmatory four-arm SFT/DPO study.

The pilot independently pins the already audited landscape, released model
weights, prepared artifact, configuration and clean Git revision. It does not
mark the older general benchmark manifest's unresolved decisions as approved:
multi-target objectives, evaluator custody, query budgets and confirmatory
evaluation remain outside this run. The fixed structural proxy is explicit;
no exact assayed antigen sequence is inferred or reconstructed.

Data: Phillips et al., [eLife 71393](https://pmc.ncbi.nlm.nih.gov/articles/PMC8476123/),
Figure 1 source data 1 v3, distributed with the CC BY article. Only measured H1
replicates enter training. Require at least two observed replicates and exclude
any genotype with an observed replicate at or below the censoring floor. Recompute
mean and sample SEM; do not use the released zero SEMs as precision weights.

Seed 20260916 selects four loci independently of labels. Their 16 combinations
are assigned to 10 training, three development and three reserved test blocks.
Whole blocks stay together, but neighboring genotypes can cross splits: this is
compositional generalization, not distance-separated antibody transfer. Reserved
test rows are removed before label aggregation, threshold selection or fitting.
This is a local separation of data use, not an independently administered evaluator.

Training samples uniformly from eligible training genotypes at or above their
training-only 75th percentile. The native decoder updates for 256 steps, batch
four, learning rate 1e-5, float32, AdamW, gradient clipping at 1. The structural
encoder stays frozen and its outputs are cached. Temperature is 1, dropout is
off, and the objective is mean negative constrained log probability over the
16 editable sites. Other residues are fixed. No preference training runs.

Before updating weights, native uncached logits must agree with the cached
policy on three actual benchmark candidates, and incremental sampling must
agree with teacher-forced rescoring. Training rejects nonfinite losses or
gradients and any encoder gradient. After training, complete encoder and decoder
state hashes establish unchanged encoder and changed decoder weights.

Parent and final scores cover the same 512 development genotypes selected by
label-independent hashes. A fixed additive ridge baseline fits eligible training
measurements only. The final step is reported without development-based checkpoint
selection; there is no final-test evaluation or checkpoint promotion. A one-seed
pilot cannot establish reproducible generalization or biological improvement.

```powershell
.\.venv\Scripts\python.exe -u scripts/run_cr9114_esmif1_pilot.py --output-dir outputs/cr9114_5cjq_pilot_20260916
```

The runner requires a clean worktree and a new output directory. It reads cached,
hash-verified weights without downloading. It writes source/config identities,
split assignments, cohort statistics, scoring checks, parent/final candidate
scores, progress, four decoder/optimizer checkpoints, and a final result report.
The checkpoint format is specific to this pilot; automatic resume is not implemented.

Code validation before the first launch: 131 tests passed across the new pilot
data rules, existing regression suite, and constrained-policy tests. Runtime
outcomes belong in the generated report and must not be inferred from those tests.
