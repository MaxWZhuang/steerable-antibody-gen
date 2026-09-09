# Pretrained antigen-conditioned policy: migration and experiment specification

**Date:** 2026-09-09

**Status:** direction accepted in [Decision 0003](decisions/0003-pretrained-conditioned-policy.md).
The migration is not implemented. All M01-M08 deliverables below are pending.

## Research question and first milestone

How does antigen context causally influence HCDR3 choices, and how do post-training
and external guidance change that influence?

The first milestone is a pretrained, fused policy that changes its HCDR3
predictions in agreement with held-out measurements when antigen context changes.
Start with fixed-length HCDR3 editing: framework, light chain, and edit boundaries
are fixed within a comparison. A heavy-only cohort is reported separately and
cannot establish VH/VL conditioning.

Three levels of evidence must remain distinct:

1. **Pathway capacity:** a synthetic task demonstrates that antigen information
   can reach the residue head. The existing pathway probe tests this.
2. **Policy sensitivity:** matched antigen substitutions change token probabilities
   or variant rankings. Unmeasured substitutions have no binding correctness label.
3. **Measured specificity:** those changes agree with held-out measurements for
   the corresponding antibody-antigen combinations and exceed shortcut controls.

Only level 3 meets the first scientific milestone. Code and fixtures can establish
the mechanics while real evaluation assets are prepared; they cannot promote a model.

## Repository inventory and required changes

This table describes code capabilities, not evidence that a trained model passes.

| Existing surface | What it already provides | Required extension |
|---|---|---|
| `models/mlm.py` | Custom antibody encoder, bidirectional cross-attention, fused residue logits, compatibility/strength/length heads | Pretrained antibody backbone with its native residue head; explicit policy interface and fusion observations |
| `models/esm_antigen_encoder.py`, `antigen_tokenization.py` | Frozen/LoRA ESM antigen encoder, projection, separate antigen vocabulary and length budget | Reuse behind the new policy; pin upstream revisions and preprocessing |
| `tokenizer.py`, `data/MLMCollator.py` | Custom chain tokens, HCDR3 spans, partial masking, binder eligibility | Map biological coordinates into each native tokenizer; preserve chain identity and frozen context |
| `scripts/mlm_train.py` | Custom-model construction, warm-start chain, MLM and auxiliary losses | Pretrained initialization and explicit adaptation tasks without requiring scratch-stage parents |
| `infill/hcdr3.py`, `scripts/hcdr3_infill.py` | Single-pass and iterative infilling; external-guide hook | Separate policy/guide tokenization and antigen inputs; one controlled sampler for guide-off/on comparisons; replayable traces |
| `evaluation/frozen_inputs.py`, `evaluation/contrasts.py`, `evaluation/contrast_scoring.py` | Frozen inputs, measured variant rankings, unmeasured substitution diagnostics | Semantic cases reusable across tokenizers; measured antigen-variant evaluator; explicit score types |
| `scripts/probe_antigen_pathway.py`, `scripts/probe_steering_reachability.py` | Synthetic capacity and local guidance diagnostics | Adapt to the policy interface; retain their limited claim scope |
| `experiment.py` | Config, tokenizer, source, contract and data fingerprints; checkpoint lineage | External model/weight revisions, adapter lineage, role, objective, sampler and activation-site metadata |
| `benchmarks/provenance.py`, target identity and split audits | Provenance validation and leakage protections | Reuse; materialize an eligible conditional benchmark without inventing missing manifest values |
| `infill/evidence.py` | Evaluation-time order-mixture evidence estimates | Keep separate from diffusion likelihood estimators and post-training objectives |

Paths beginning with `models/`, `data/`, `evaluation/`, `infill/`, or a bare Python
filename in this table are relative to `src/smallAntibodyGen/`.

## Model boundaries

The names below describe proposed interfaces, not callable APIs in this checkout.
Implement the smallest boundary needed by the first selected backbone; do not
build a general model registry before one integration works.

### Policy and biological coordinates

A `ConditionedPolicy` consumes a partially masked antibody, chain identities,
editable residue coordinates, and antigen context. It returns residue logits
aligned to those coordinates and, on request, named intermediate activations.
Sampling and compatibility scoring are separate operations.

The backbone wrapper owns its native tokenizer, special tokens, context limits,
residue-to-token mapping, and pretrained output head. Reusing the custom
`[CLS] [IGH] ... [SEP] [IGK] ...` integer IDs with a different vocabulary is invalid.
Chain identity must have an explicit supported representation; added tokens or
embeddings are new parameters and must be recorded. Reject examples whose complete
fixed antibody context cannot fit rather than silently dropping a chain or CDR.

With fusion disabled, a supported single-chain input must reproduce the upstream
model's residue logits within a declared numerical tolerance. Preserve the native
head, including its transforms and weight tying; a random linear replacement is
not a pretrained generative policy. Record whether scores normalize over the full
native vocabulary or only the canonical amino acids.

Start with a pretrained antigen encoder and trainable projection/cross-attention
into the antibody residue path. Reuse the existing fusion implementation where its
shape and forward semantics fit. A compatibility head may provide auxiliary
supervision, but success of that head does not prove policy conditioning. Begin
with limited backbone adaptation and expand it only after a measured deficit.

Named observation sites should include antibody states before fusion, antigen
states presented to fusion, the cross-attention update, antibody states after
fusion, and residue logits. Each site needs residue coordinates and checkpoint
identity. A disabled observer must leave logits unchanged; a read-only observation
must not consume the sampler's random stream.

### Guide, reference, and evaluator

A `PartialStateGuide` scores candidate next biological states with its own antigen
input, tokenizer, limits, and frozen checkpoint. Passing policy token IDs directly
to a guide with another tokenizer is forbidden. The external-guidance seam already
separates antigen tokenizers, but still shares the custom antibody-token layout.

The new experimental path requires an explicit guide when guidance is enabled.
Keep the historical internal-head fallback available only in the legacy path or
an explicitly named ablation. A separately loaded checkpoint is not automatically
an independent judge.

The preference reference is the frozen pre-post-training policy, including its
fusion parameters and preprocessing. The final evaluator uses withheld measurements
or an independently validated evaluation source; neither guide scores nor SAE
features become ground truth by being stored in another file.

### Sampling and traces

Use one sampler implementation for the guide-off and guide-on arms. Record the
position/order rule, schedule, temperature, truncation, RNG state, and actual
normalized action probabilities. A logged action probability is not a final
sequence marginal likelihood.

**Current confound:** the CLI calls single-pass `infill` at guidance strength zero
and iterative `guided_infill` above zero. Comparing those runs changes both the
sampler and guidance. The method `guided_infill(..., guidance_strength=0)` already
supports iterative unguided decoding; the new experiment runner must use the same
sampler on both sides. Fixed replay cases/orders isolate local guidance effects;
free-running adaptive trajectories are reported separately.

Trace each step's biological state, mask positions, policy and guide antigen IDs,
policy logits, guide contributions, chosen action, and selected activation sites.
Store traces under local `outputs/`, with semantic-case and tensor digests. Full
activation dumps are opt-in and limited to a fixed diagnostic cohort.

## Data and evaluation contract

Keep target identity resolution, ancestor/exposure audits, study metadata, assay
direction, censoring, and conditional-denoising eligibility. Measured nonbinders
can supervise compatibility; they must not silently become positive conditional
generation targets. Retention data must respect the evaluation leakage firewall.

Build cases in residue space first, then materialize and hash native tensors for
each model. Equal random seeds or equal token IDs across incompatible tokenizers
do not establish equal biological inputs. Within a checkpoint comparison, hold
the corrupted antibody, observed residues, masks, positions, and time level fixed.
Use the same antigen residues/crop across arms and record missing or truncated
assayed constructs. Do not infer binding labels for synthetic antigen swaps.

Separate development data for fitting/tuning from sealed evaluation labels.
Report uncertainty with the relevant independent unit (target, antibody family,
or study), avoiding per-residue or repeated-variant pseudoreplication. Specify
split scope, primary metric, null/permutation controls, retention margins, seeds,
and compute budget before inspecting selection results. Missing evidence remains
missing; it is neither zero nor a pass.

The first biological report must include:

- Correct-antigen versus matched-alternative token/variant response with guide off,
  including high-mask states and a fully masked HCDR3.
- Agreement with measured antigen-variant effects, with antibody-only, antigen-only
  or prevalence, and source-study controls where applicable.
- Framework/VL preservation, antibody quality, measured VH/VL retention, diversity,
  and exposure/novelty strata.
- Separate labels for sensitivity-only, measured specificity, and generated
  candidates without experimental outcomes.

The existing HCDR3 contrast machinery is reusable for native measured rankings.
Its `substituted_unmeasured` cases cannot establish the direction of binding changes.
The [benchmark manifests](benchmarks/README.md) remain templates until their real
values and files are verified. Antigen-mutant generalization on one target does
not establish generalization to unseen target families.

## Diffusion and post-training boundaries

Two initialization routes are eligible: adapt an existing diffusion checkpoint
under its stated objective, or establish conditioning in an MLM and then continue
under a specified masked-diffusion objective. The latter has precedent in
[DPLM](https://arxiv.org/html/2402.18567v2#A2). [VESM](https://github.com/ntranoslab/vesm)
is a candidate initialization based on variant-effect work, not a proven
antibody-conditioned generator. Model choice follows feasibility and task evidence.

For an MLM route, compare objectives from the same conditioned parent without
changing the backbone or fusion design. An ESM-versus-DPLM checkpoint comparison
selects a package and does not isolate the causal effect of diffusion training.
For a DPLM route, no additional general-protein diffusion pretraining is required
by this plan, but adaptation still needs a specified corruption process and sampler.

`mask_rate_schedule: uniform`, `partial_span`, and `mask_only` are useful existing
mechanisms; enabling them does not establish a diffusion objective. Before training
the new diffusion arm, write `specs/masked_diffusion.md` with the editable-region
forward process, time distribution, loss weighting/reduction, zero-mask/end-point
handling, fixed-context behavior, reverse transitions, and enumerable toy checks.
Recheck antigen conditioning and retention after continuation.

Diffusion defines a generation process; exact final-sequence probabilities can
remain intractable. DPO based on sampled ELBOs has both a variational approximation
gap and finite-sample nonlinear bias. [VRPO](https://arxiv.org/abs/2505.19223) is a
methodological reference, not evidence that a particular estimator works here.
Before preference trainer code, write `specs/diffusion_dpo.md` specifying the
policy/reference terms, same-sequence corruption reuse, winner/loser coupling,
Monte Carlo budget, reductions, reference behavior, and toy tests that distinguish
those error sources. Neither future contract is supplied by this migration document.

Use supervised continuation on measured desirable examples as the first weight-update
baseline. Evaluate DPO/VRPO with trustworthy pairs sharing antigen, framework, VL,
edit length, and comparable assay context. GRPO requires a reward that can assess
new rollouts and a sampler-compatible estimator; defer it until both exist. An
IRPO-style iterative loop likewise needs a defensible way to judge new candidates.
No general post-training method is made valid merely by naming the model diffusion.

## Main experiment and causal interpretation

Freeze one conditioned parent, one guide, the evaluation cases, and the sampler
protocol. Run this matrix with the same guide checkpoint in both rows:

| Policy | Guide off | Guide on |
|---|---|---|
| Before post-training | A: learned conditioning | B: external guidance effect |
| After post-training | C: learned policy update | D: combined intervention |

`C - A` measures the policy update without guidance; `B - A` and `D - C` measure
the guidance effect at each checkpoint. Report their difference on a declared
metric scale with uncertainty. Include matched-budget sampling plus reranking and
supervised-continuation controls. Report selection budget and compute separately.

Cross policy and guide antigen inputs independently: correct/correct,
correct/swapped, swapped/correct, and swapped/swapped. These are diagnostic
interventions; an unmeasured swapped pair acquires no experimental label. Include
appropriate matched substitutions and null controls without calling an artificial
null sequence in-distribution.

At fixed inputs, a frozen deterministic policy has the same activations whether an
external guide is enabled or disabled. Guided trajectories subsequently change
inputs, so trajectory activation differences alone do not demonstrate learned
fusion. Maintain two distinct analyses:

- **Fixed-state analysis:** compare antigen conditions or policy checkpoints on
  identical corrupted antibodies, masks, positions, and time levels.
- **Trajectory analysis:** characterize the states each sampler visits; separate
  those distribution changes from changes to the policy's response at a fixed state.

Begin causal intervention within each checkpoint: ablate antigen access or patch
aligned fusion activations from one matched antigen condition into another. Measure
whether the predicted preference changes or is restored. Include no-op, self-patch,
and matched unrelated-donor controls, verify untouched inputs, and report off-manifold
limitations. Do not directly patch between checkpoint coordinate systems without
establishing comparability. Repeat the within-checkpoint intervention before and
after post-training. Probe accuracy and attention maps are descriptive evidence.
SAEs follow a reproducible causal response and require independent feature validation.

## Implementation order and completion evidence

All rows are **pending**. Existing components named above are starting points,
not completed migration deliverables.

| Work item | Scope | Done when |
|---|---|---|
| M01: freeze the pilot | Choose one measured conditional cohort; pin exact candidate model assets and actual hardware budget; establish semantic cases and splits | Verified data/model provenance, recorded limits/metrics/margins, and honest sensitivity-only versus measured-specificity coverage |
| M02: integrate one backbone | Native tokenizer/head wrapper, policy construction/loading, new lineage, opt-in activation sites | Upstream-logit parity, correct residue/chain mapping, intended gradient flow, save/load parity, incompatible resumes rejected, real memory/time smoke report |
| M03: adapt and test fusion | VH/VL adaptation as needed, antigen fusion into residue logits, retention and conditional evaluator | Synthetic capacity controls pass and a held-out report establishes measured specificity beyond shortcuts; otherwise no policy promotion |
| M04: implement the diffusion contract | Adopt native diffusion for a diffusion backbone, or continue the conditioned MLM with an objective control | Enumerated process/sampler tests, honest probability labels, conditioning/retention report; the native contract precedes any M03 diffusion training |
| M05: establish the guide | Partial-state fitting/calibration, separate tokenization and antigen inputs, controlled sampler and traces | Mask-bin evaluation, frozen guide provenance, guide-off sampler parity, independent antigen-swap controls, deterministic trace replay |
| M06: establish causal baselines | Fixed-state activation collection and within-checkpoint antigen-path interventions | No-op/self-patch parity, controlled transfer/ablation report, separation from trajectory effects; capture before post-training |
| M07: compare weight updates | Supervised continuation first, then the specified diffusion preference method | Estimator/toy checks and measured holdout gains with frozen reference, conditioning, retention and diversity evidence |
| M08: run the interaction study | A/B/C/D matrix, crossed antigen inputs, repeat causal interventions | Reproducible report distinguishing policy change, guidance effect, interaction and biological claim limits |

M01's data preparation and M02's mechanical integration can progress independently.
M05/M06 follow an eligible M03 policy; M07 needs the generative contract and causal
baseline. Guide-derived training arms also need M05, while independently measured
preference training need not wait for a guide. M08 requires both branches. No task
requires training the old v5 chain first.

Start M01 data work and the hardware measurement alongside M02's mechanical
preparation. Do not defer asset acquisition until an adapter is finished. Select
one backbone only after the actual training GPU/VRAM is recorded and a feasible
pilot configuration is measured. M02 can establish interfaces and coordinate
fixtures before that, but cannot claim a selected training package. M03 requires
both M01's verified conditional data/splits and M02's integration evidence.

### M01 readiness record, 2026-09-09

| Workstream | Observed state | Next evidence required |
|---|---|---|
| Conditional data | `data/raw/` absent; local processed directories contain OAS data, no antibody-antigen corpus | Retrieve one authoritative measured conditional release, record actual file hashes and assay semantics, and establish development/evaluation splits |
| Benchmark provenance | All three manifests contain four `TODO(owner)` values each, empty `files`, and unresolved owner decisions | Verify values and decisions for the selected cohort; validate its completed manifest rather than treating all three as prerequisites |
| Training hardware | Exact training device unrecorded; the recorded local baseline is CPU-only macOS | Capture `nvidia-smi --query-gpu=name,memory.total --format=csv` on the training box, then measure the actual adaptation configuration |
| Backbone | No selection; 4 GB is an earlier planning assumption | Match a pinned checkpoint to the measured hardware and pilot requirements, retaining native prediction parity |

AVIDa-hIL6 remains a candidate for antigen-mutant evidence, not an acquired or
approved pilot. Its VHH setting cannot establish VH/VL pairing; pairing evidence
needs an appropriate separate cohort. Do not replace missing data or a missing GPU
measurement with a completed adapter milestone. No multi-model architecture sweep
or preference-trainer sweep is scheduled before the first conditional pilot.

The feasibility report measures actual sequence lengths, padding, batch size,
precision, trainable parameters, and peak memory/time for the intended adaptation.
Budget the policy, antigen encoder, guide, preference reference, and activations
at the stages where they are needed. LoRA reduces trainable state; it does not
make activation memory or grouped sampling free.

## Documentation and compatibility

The [README](../README.md) states the research direction and current capabilities;
[Decision 0003](decisions/0003-pretrained-conditioned-policy.md) records the decision;
this file owns migration priorities. Older scratch-model experiment specs are
historical/reference experiments and do not gate this path.

Keep existing commands/config meanings and checkpoint loaders working for the
custom-model baseline. Add an explicitly named pretrained path with a versioned
manifest; do not reinterpret old checkpoints as pretrained or diffusion policies.
Do not add runnable-looking YAML for an unimplemented backend.

`experiment.py` records all six component hashes in `run_hash`; `specs/` remains
visible in both source and contract provenance. Its separate `resume_hash` covers
architecture, effective config, tokenizer, and data. Those four must match before
state restoration. Source/contract differences warn without refusing a compatible
resume, and full prior segment fingerprints survive in checkpoint/sidecar
`resume_history`. Schema-1 compatibility is derived from existing components;
fingerprintless or incomplete compatibility evidence still refuses resume.

Warm-start rules and explicit clean-worktree requirements are unchanged. A new
source revision is visible evidence, not proof that the objective is unchanged:
intentional changes to loss or evaluation semantics need a new experiment identity.
Rejected resumes preserve existing sidecars, and successful resumes retain their
actual warm-start parent rather than attributing weights to a subsequently edited
initialization file.

Internal working notes remain under ignored `docs/`; datasets, weights, and run
artifacts remain local. This specification does not change the local-only guardrail.
