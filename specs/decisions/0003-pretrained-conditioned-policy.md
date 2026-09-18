# Decision 0003: pretrained initialization for an antigen-conditioned policy

**Date:** 2026-09-09

**Status, 2026-09-18:** historical architecture decision. ESM-IF1 integration and
CR9114 pilots are complete; the subsequent p-IgGen/HER2 campaign is complete.
The active scope is [mechanistic interpretation](../../reference/research-design.md).
The optional conditioned-policy extension remains separate. The dated decisions
below record earlier choices and impose no performance or approval prerequisite
on current interpretation work.

## Scope clarification — 2026-09-14

The first experiment followed the narrower offline preference scope (its working
plan is now retired) and the user's stated priority: establish a measured
fixed-target task with an existing pretrained generator, then study post-training
and interpretation.
The user selected **ESM-IF1** (`esm_if1_gvp4_t16_142M_UR50`) after considering
future antigen conditioning. The first experiment uses one declared fixed
structural context. Cross-antigen response is a later extension. The original
decision below describes that longer-term program; its antigen-response milestone,
custom fusion, and diffusion path are not prerequisites for this first experiment.

Use ESM-IF1's existing autoregressive decoder and geometric encoder. Pin the exact
weights, source revision, alphabet, structural input, and residue mapping; verify
probability semantics, exposure, and measured compute before checkpoint promotion.
This is structural conditioning, not raw-antigen-sequence conditioning. The first
pilot proposes a frozen encoder and decoder adaptation. Fixed-length editing may
use the measured CR9114 VH sites outside HCDR3, keeping the remaining residues and
light chain fixed. The previously evaluated p-IgGen candidate is retained as an
alternative; its release and exposure checks do not audit ESM-IF1.

The [research recommendation](../../reference/fixed-target-posttraining-recommendation.md)
records the CR9114/H1 rationale and limits of the intended claim.
The [local readiness audit](../../reference/evidence/esm-if1-readiness-2026-09-14.json)
recorded that model loading, device/gradient checks, and compute measurement were
pending on that date. Those checks and the pilots subsequently completed; see
the [integration record](../esmif1_policy.md#integration-status). The dated
clarification itself is not evidence of a training or biological result.

## Original broader decision — later program

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
optional conditioning work and its implementation boundaries. This decision
supersedes earlier roadmap priorities and the requirement to finish J11/J24 or
train the custom v5 chain before pursuing the main experiment. Historical results
and data-integrity contracts remain in force.

The future `specs/masked_diffusion.md` and `specs/diffusion_dpo.md` must separately
specify the generative process and preference estimator. Neither file exists at
the time of this decision; this document is not a substitute for either contract.
