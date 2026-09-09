# Decision 0003: pretrained initialization for an antigen-conditioned policy

**Date:** 2026-09-09

**Status:** direction accepted; implementation pending; backbone not selected.

## Decision

Make the central experiment a test of how antigen information changes HCDR3
generation, how post-training changes that response, and how a separate guide
interacts with the resulting policy. Initialize the generative backbone from
pretrained weights and adapt it to the antibody task.

The first milestone is a policy whose fixed-length HCDR3 predictions respond to
held-out antigen changes in agreement with measurements. Keep the heavy-chain
framework and light chain fixed. A compatibility classifier, an antigen-sensitive
embedding, or an unmeasured antigen-swap response alone does not meet that milestone.

## Consequences

- Reuse data preparation, target identity, leakage controls, frozen evaluation,
  checkpoint provenance, and the existing fusion and guidance infrastructure.
- Pursue conditional-data acquisition and actual training-GPU measurement alongside
  mechanical integration. Backbone selection needs hardware evidence; conditional
  training needs a verified measured cohort. A CPU-only local baseline does not
  establish the remote training budget.
- Add an explicit pretrained antibody-backbone boundary. The existing ESM option
  replaces only the antigen encoder; it does not implement this decision.
- Adapt VH/VL behavior where measured deficits justify it. Preserve antibody and
  pairing performance during antigen adaptation.
- Feed antigen information into residue predictions. Keep the policy, guide,
  preference reference, and final evaluator as distinct roles.
- Use masked diffusion as the planned generative path. Continue an MLM under a
  specified diffusion objective or adapt an existing diffusion checkpoint under
  its objective. Diffusion is not a universal prerequisite for post-training.
- Establish a matched partial-state baseline and recheck conditioning after any
  objective change. Exact probability semantics and estimator tests precede
  diffusion preference training.
- Collect controlled activation measurements before post-training. Causal
  interventions begin with a reproducible antigen response; SAE discovery comes
  later and cannot supply its own biological ground truth.

## Work removed from the critical path

Scratch OAS pretraining, custom Transformer architecture sweeps, the mandatory
four-stage custom checkpoint chain, variable-length design, structure co-design,
and a broad sweep of preference algorithms are no longer prerequisites. Existing
code, configs, and experiment evidence remain usable as controls and historical
reference. This decision does not establish that an existing checkpoint is bad.

ESM-2, VESM, and DPLM are candidates, not promoted models. Exact checkpoint,
revision, supported usage, tokenizer, objective, and measured compute requirements
must be recorded before selecting an implementation. VESM's variant-effect
results are not evidence of antibody generation quality.

## Authority

[The migration specification](../pretrained_conditioned_policy.md) defines the
work order, implementation boundaries, and acceptance evidence. This decision
supersedes earlier roadmap priorities and the requirement to finish J11/J24 or
train the custom v5 chain before pursuing the main experiment. Historical results
and data-integrity contracts remain in force.

The future `specs/masked_diffusion.md` and `specs/diffusion_dpo.md` must separately
specify the generative process and preference estimator. Neither file exists at
the time of this decision; this document is not a substitute for either contract.
