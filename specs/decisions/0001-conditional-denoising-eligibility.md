# 0001. Restrict conditional denoising by a per-stage eligibility policy

| | |
|---|---|
| **Status** | Accepted, 2026-08-27 |
| **Contract** | [../conditional_denoising_eligibility.md](../conditional_denoising_eligibility.md) |

## Context

`AntibodyAntigenRealLabelCollator` overrides only `_build_antibody_antigen_batch`, which
derives compatibility labels from `binder_label`. The inherited `__call__` then masks tokens
for every row in the batch. A measured nonbinder therefore becomes a positive reconstruction
target for the policy under the antigen it is labeled *not* to bind.

This contradicts the intended semantics of a policy whose conditional distribution is
supposed to represent binders. It is a behavioral defect independent of any planned work on
replay, null conditioning, or matched controls.

The obvious fix — make eligibility `binder_label == 1` inside the collator — is correct for
Stage 3 and destructive for Stage 4, because the two stages construct the same collator over
different dataset filters.

*Verified against commit `2dde1af`.*

**Stage 3 (`antigen_real_label_refine`)** filters rows to those with `binder_label in (0, 1)`,
optionally widened by `include_strength_rows`. Every row reaching the collator already
carries a binary label, so `binder_label == 1` selects exactly the binders.

**Stage 4 (`is_hcdr3_infill_record`)** filters on `is_strong_binder`, which the training
script already documents as deliberately broader than `binder_label`: the label is populated
only for rows whose `affinity_type` is literally `"bool"`, so KD, -log KD, and fuzzy strong
binders all carry `binder_label is None`. The script's own comment states that restricting to
`binder_label == 1` "would silently drop the large majority of strong binders."

The failure would be invisible at runtime. MLM loss over an all-ignored batch returns a
differentiable zero rather than NaN, so Stage 4 would train on almost nothing while reporting
a plausible loss curve.

## Decision

Eligibility is a named, serializable per-stage policy rather than a single predicate.

Stage 3 uses `binary_binders_only` (`binder_label == 1`). Stage 4 uses `all_filtered_rows`,
under which every row admitted by the stage's dataset filter is eligible. No
`is_strong_binder` check is added inside any collator.

Zero-eligible handling differs by policy rather than being a uniform hard failure: an
all-nonbinder batch is legitimate under `binary_binders_only` and must still contribute
compatibility supervision, whereas under `all_filtered_rows` it can only mean incorrect
wiring and raises immediately.

The change ships independently of replay work. A reduced MLM budget is preferable to training
the conditional policy toward known nonbinders.

Alternatives neglected: a single `binder_label == 1` predicate for both stages, which
silently destroys Stage 4's population; extending eligibility to `is_strong_binder` inside
the collator, which changes which population Stage 4 denoises and is a scientific change
rather than a defect fix; and holding the fix until an antibody-only replay path exists.

## Consequences

Stage-3 nonbinder rows lose **all** MLM signal, not just their antigen-conditioned share,
because the dual-stream model has no antibody-only MLM path — MLM logits are read from the
fused antibody stream. Until an antibody-only forward and replay exist, those sequences teach
the model nothing about plausibility. The config-estimated binder share is roughly 54%, so
the expected reduction is about 46% of rows; the actual eligible-token fraction should be
recorded when the corpus is inventoried.

Stage-3 MLM loss measures a different population before and after this change and is not
comparable across it. Existing Stage-3 baselines must be re-measured rather than compared,
and any regression budget defined against a pre-change number is void.

Resume against a populated Stage-3 output directory is unsafe, because the semantic change is
invisible to the current resume path. A fresh output directory or an explicit no-resume flag
is required until fingerprinting can reject the mismatch.

`all_filtered_rows` makes a collator's behavior depend on a filter enforced elsewhere. That
coupling is deliberate — it avoids duplicating the population definition — but it means a
change to a stage's dataset filter silently changes what that stage denoises. The immediate
error on a zero-eligible batch is the guard against the worst version of this.
