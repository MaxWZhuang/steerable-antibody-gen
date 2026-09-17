# CR9114 constrained-policy DPO pilot

The [preference construction](../reference/cr9114-preferences-development.md)
fixes the pairs, weighting, development pool and uncertainty convention.
Use the same 5CJQ geometry and binary 16-site support for policy and reference.

## Objective and reference contract

The loss follows [Rafailov et al., equation 7](https://arxiv.org/abs/2305.18290):

`softplus(-beta * ((log_q_chosen - log_q_rejected) - (log_ref_chosen - log_ref_rejected)))`.

All scores are summed constrained sequence log probabilities at temperature 1,
with dropout off. They are not average residue likelihoods or affinity predictions.
`smallAntibodyGen.experiments.dpo.dpo_per_pair_loss` returns one loss per pair,
rejects broadcasting, nonfinite inputs and trainable reference tensors, and leaves
weighting to the caller. Weight-proportional pair sampling uses an unweighted
minibatch mean so weights are applied exactly once.

Direct DPO initializes from the released parent and fixes that parent as reference.
SFT-to-DPO initializes from the saved 256-step SFT decoder and fixes that decoder
as reference. Both keep the structural encoder frozen. Checkpoint-specific
reference scores can be computed without gradients once and reused; an adapted
policy must never silently replace its own reference.

Reference caches pin the source checkpoint, actual decoder/encoder state hashes,
prepared input, geometry encoding, scorer implementation and probability convention.
They contain exact genotype order and one finite nonpositive log probability per
genotype, plus a content digest. Loading refuses a changed identity or order;
returned arrays are read-only. Existing cache files cannot be overwritten.

## Validation

At policy/reference equality the loss must equal log(2). Analytical gradient
tests on an enumerable four-sequence distribution verify preference direction,
normalization cancellation and pair weighting. Additional tests cover extreme
margins, malformed shapes, trainable references, invalid beta, cache corruption,
checkpoint/geometry/contract mismatches and genotype reordering.

Runtime validation must verify cached versus fresh scores, unchanged reference
files, absence of encoder gradients, unchanged encoder state, changed decoder
state, finite training, and safe strict checkpoint reload. Development evaluation
uses raw constrained `log q`; the implicit reward `beta * (log q - log q_ref)`
is a separate quantity. Reserved test labels remain excluded.

## Bounded run protocol

[The configuration](../configs/experiments/cr9114_dpo_pilot.json) fixes two arms,
seed 20260916, 256 updates, two pairs per update, beta 0.1, learning rate 1e-5,
AdamW, gradient clipping at 1, float32, and a frozen cached encoder. The pair
schedule is sampled once proportional to the normalized training weights and
shared by both arms. Each arm sees 512 pair exposures / 1,024 sequence exposures.
SFT-to-DPO additionally includes the preceding 256-step SFT run; equal DPO budgets
do not mean equal total adaptation compute.

The [runner](../scripts/run_cr9114_dpo_pilot.py) requires committed source and a
fresh output directory. It verifies the pinned preference manifest, CSVs,
development scores, original split membership, structural source, and both
initial checkpoints. It caches all 38,066 training-pair variants for each
reference and verifies a subset against fresh scoring. All 512 initial
development scores must also reproduce the earlier pilot.

The initial DPO loss must be log(2). Checkpoints are saved every 64 updates and
only the predeclared final step is evaluated. The runtime checks include strict
`weights_only=True` reload, exact restored decoder tensors, rescoring parity,
unchanged encoder state and an unchanged reference-cache file. Checkpoints and
caches are local artifacts, not Git content. Automatic resume is not implemented.

```powershell
.\.venv\Scripts\python.exe -u scripts/run_cr9114_dpo_pilot.py --output-dir outputs/cr9114_dpo_pilot_20260916
```

Both arms are evaluated against their immediate initialization on the same
1,399 development pairs and 512-candidate top-K pool. Results use the established
variant/block weighting and retain the three-block uncertainty limitation.
This pilot is not the multi-seed confirmatory experiment and does not promote
a checkpoint or claim biological improvement.

Pre-launch validation on 2026-09-16: the full repository suite passed with
1,721 tests and three skips. A separate input preflight verified all 114,395
training pairs, 38,066 reference genotypes, 1,399 development pairs and the
256-by-2 sampled schedule against the pinned artifacts.
