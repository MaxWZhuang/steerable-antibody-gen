# Conditional-denoising eligibility

| | |
|---|---|
| **State** | `implemented` — J22c, 2026-08-27. See "Implementation status" at the end. |
| **Applies to** | `AntibodyAntigenCollator` and `AntibodyAntigenRealLabelCollator` |
| **Rationale** | [decisions/0001-conditional-denoising-eligibility.md](decisions/0001-conditional-denoising-eligibility.md) |

A row's **conditional-denoising eligibility** decides whether that row contributes
antigen-conditioned MLM targets. It is independent of whether the row contributes
compatibility or strength supervision, and it never changes the corrupted input.

The policy is a named, serializable value — an enum or string, never a callable — so it can
be recorded in a config, saved in a checkpoint, and hashed into a run fingerprint.

## Policies

| Policy | Eligible rows | Used by |
|---|---|---|
| `binary_binders_only` | `binder_label == 1` | Stage 3, `antigen_real_label_refine` |
| `all_filtered_rows` | every row the stage's dataset filter admitted | Stage 4, `is_hcdr3_infill_record` |

`all_filtered_rows` defers to the dataset filter rather than re-deriving a predicate. The
name states that it does not broaden a stage's source population. No `is_strong_binder`
check belongs inside a collator.

Under `binary_binders_only`, Stage-3 graded-strength rows admitted through
`include_strength_rows` are ineligible for conditional MLM unless they also carry
`binder_label == 1`. Their strength targets are unaffected.

The base `AntibodyAntigenCollator` defaults to `all_filtered_rows` for backward
compatibility, but every production construction site passes the policy explicitly. A
default reaching production through omission is a defect.

## Mechanics

Eligibility is applied **after** `_mask_tokens` and **before** `_build_hcdr3_metadata`.

For an ineligible row:

- `antibody_labels` are set to the ignore index.
- `antibody_target_mask` is set false.
- The corrupted input is left unchanged, so compatibility still trains on noisy states.
- Compatibility and strength labels and masks are left intact.

The batch carries an explicit row-level eligibility mask for observability.

Two constraints that are easy to violate and hard to detect:

**RNG.** `_mask_tokens` draws randomness per selected position. Applying eligibility after it
leaves the stream untouched, which is what makes eligible-row output byte-identical under a
fixed seed. Filtering rows earlier shifts every subsequent draw.

**Metadata coupling.** `_build_hcdr3_metadata` derives `hcdr3_target_mask` from
`antibody_target_mask`. Clearing labels without clearing the target mask leaves HCDR3 target
counts and mask-fraction bins reporting positions that contribute no loss, which silently
corrupts the calibration bins guide training depends on.

## Zero-eligible batches

An all-nonbinder batch is legitimate under `binary_binders_only` and must still contribute
compatibility supervision, so this is not a uniform hard failure.

| Policy | Zero eligible rows in a nonempty batch |
|---|---|
| `all_filtered_rows` | immediate error — nothing but incorrect wiring can produce it |
| `binary_binders_only` | counted and logged; fails only when preflight or a whole epoch yields zero eligible rows or tokens |

MLM loss over an all-ignored batch is a finite zero, not NaN.

## Fingerprint

The policy is declared as a `TrainConfig` field, so it flows into `train_config` in every
saved checkpoint and into the run's config JSON without further work, and joins the run
fingerprint when fingerprinting exists.

Consequence for resume: the two policies produce different training populations from
otherwise identical-looking configs, so a fingerprint check must reject a policy change
against a populated output directory rather than silently reinterpreting it.

## Required test coverage

Under `binary_binders_only`:

- A nonbinder row has ignore-index labels and a false target mask, retains its compatibility
  label and mask, and keeps its corrupted input unchanged.
- A binder row is byte-identical to pre-change behavior under the same seed.
- An unlabeled row receives neither objective.
- A graded-strength row with `binder_label is None` is ineligible for conditional MLM and
  retains its strength target.
- An all-nonbinder batch yields a finite zero MLM loss, is counted and logged, and does not
  raise.
- Zero eligible rows at preflight or across a whole epoch fails.

Under `all_filtered_rows`:

- Rows with `binder_label is None` arising from KD, -log KD, and fuzzy assay types remain
  fully eligible. This requires its own fixture; it must not be an incidental consequence of
  another test.
- Zero eligible rows in a nonempty batch raises immediately.
- The base `AntibodyAntigenCollator` is byte-identical to pre-change behavior.

Wiring, both policies:

- Train and evaluation loader builders each pass the policy explicitly, for both collator
  classes.
- The policy round-trips through config serialization and checkpoint save/load.

Shared:

- HCDR3 target counts and mask-fraction metadata agree with the post-eligibility target mask.

## Out of scope

Synthetic shuffled-antigen rows in the base collator still receive conditional MLM targets.
That is a separate defect on a different config path.

Ineligible rows contribute no MLM signal at all, because the dual-stream model has no
antibody-only MLM path — MLM logits are read from the fused antibody stream. Restoring that
signal requires an antibody-only forward and replay, which are separate work.

## Implementation status

Shipped as J22c on 2026-08-27 against commit `2dde1af`.

| Surface | Where |
|---|---|
| Policy names | `CONDITIONAL_DENOISING_ELIGIBILITY_POLICIES` in `src/smallAntibodyGen/data/MLMCollator.py` |
| Predicate and per-batch mask | `AntibodyAntigenCollator._is_conditional_denoising_eligible` / `._conditional_denoising_eligibility_mask` |
| Application point | `AntibodyAntigenCollator.__call__`, between `_mask_tokens` and `_build_hcdr3_metadata` |
| Observability key | `conditional_denoising_eligible`, bool `[B]`, in every dual-stream batch |
| Config field | `TrainConfig.conditional_denoising_eligibility`, resolved per stage in `__post_init__` |
| Preflight | `summarize_conditional_denoising_eligibility` plus the fatal check in `main` |
| Per-epoch census and guards | `train_one_epoch`; reported without a guard in `evaluate` |
| Tests | `src/smallAntibodyGen/tests/test_conditional_denoising_eligibility.py` (37) |

### Deferred, with the reason

**Resume does not yet reject a policy change.** The Fingerprint section requires
that a fingerprint check reject a policy change against a populated output
directory. No fingerprinting mechanism exists in the repository yet; `main`
resumes from `last.pt` without reading its `train_config` at all, so a policy
change against a populated directory is silently reinterpreted and
`best_val_loss` is carried across the discontinuity. That work is **J03** in
`docs/PLAN-steering-prerequisites.md` (untracked local plan). Until it lands, a
Stage-3 run whose policy changed requires a fresh output directory or
`--no-resume-from-last`; the per-epoch census keys are what make the switch
visible after the fact.

### Two findings from implementation, recorded because they are not obvious

**The "finite zero, not NaN" claim held only in fp32.** The all-ignored guard in
`models/mlm.py` returned `logits.sum() * 0.0`. Under AMP the logits are fp16,
which saturates at 65504; a Stage-3 batch is 16 x 192 x 35 = 107,520 logits, so a
mean logit magnitude above ~0.61 overflows the sum to `inf` and `inf * 0.0` is
NaN. `binary_binders_only` is what makes that branch routine on Stage 3 — before
it, every row carried targets and the branch was unreachable there. The guard now
promotes to fp32 before summing.

**Zero eligible *tokens* is reachable only under `partial_span`.**
`_select_target_positions` floors the budget at `max(1, ...)` for `sampled_span`
and `full_span`, so those modes always produce at least one target per row.
`partial_span` draws `k ~ uniform{0..L}` and `k == 0` is legitimate, which is why
the tokens guard is separate from the rows guard.
