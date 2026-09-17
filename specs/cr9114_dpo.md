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
