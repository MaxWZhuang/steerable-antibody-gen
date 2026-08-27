## Steerable Antibody Generation: Antigen Conditioning, Guidance, and Preference Post-Training

This project explores how to make antibody generation more interpretable and more controllable, with an eventual focus on antigen-conditioned antibody design and targeted HCDR3 editing.

**The control claim:** inference-time guidance can only re-rank what the base model already puts mass on, and `scripts/probe_steering_reachability.py` measures that ceiling directly. Preference post-training may move that ceiling by changing the policy itself, but it is promoted only after the base policy is demonstrably antigen-conditioned and the objective has passed independent gates.

Sparse feature methods such as SAEs remain a later interpretability direction. They may eventually help name or diagnose what moved, but they are not a prerequisite for guidance or preference optimization and are deferred until the antigen-conditioned policy is validated.

The core modeling sequence is: learn strong antibody representations, fuse them with antigen context, verify that the antigen changes the policy in experimentally meaningful ways, and only then optimize or interpret that policy.

Rather than jumping directly to a large autoregressive generator, the project starts with masked language modeling (MLM) on antibody sequences from OAS (Observed Antibody Space). That first stage is meant to teach antibody sequence grammar, chain-specific structure, and local residue constraints well enough to support later antigen-aware refinement.

---

## Project Goals

The long-term goal is not just to generate antibody sequences, but to build a system that can:

- learn antibody sequence biology and chain-specific grammar,
- incorporate antigen and assay context,
- predict useful downstream properties,
- expose biologically meaningful latent concepts,
- steer generation toward independently validated properties or concepts, both at sampling time and by post-training on preference data, and
- support controlled optimization of promising binders.

---

## Where Steering Lives

Three different places in the stack can carry a steering intervention, and they
fail in different ways. Keeping them distinct is the main architectural
commitment of the project; conflating them is how a project ends up unable to
say whether a result came from the model, the judge, or the sampler.

| Surface | Where the intervention lives | Touches weights? | Status |
|---|---|---|---|
| **1. Inference-time guidance** | the per-position sampling distribution, one candidate at a time | no | **implemented** — `guided_infill`, see [Guided generation](#guided-generation-proteinguide-style-opt-in) |
| **2. Preference post-training** | the generator's own parameters | yes | **planned** — Phase 5b |
| **3. Concept discovery and activation steering** | the residual stream, through SAE features | no for discovery; optional at decode | **deferred** until the antigen-conditioned policy is validated |

How they relate:

- **Surface 1 is bounded by the base model.** Guidance reweights
  `log p_MLM(a | x)` by `gamma * log p(binder | ...)`. If the base model assigns
  a residue negligible mass, no finite `gamma` recovers it, and the binder term's
  *spread* over candidate residues is a hard ceiling on what any `gamma` can do.
  `scripts/probe_steering_reachability.py` exists specifically to measure that
  ceiling before spending compute on a `gamma` sweep.
- **Surface 2 moves the ceiling.** Preference post-training changes the base
  distribution itself, so it can put mass where guidance had nothing to amplify.
  The cost is that it is a permanent, global edit: it can degrade antibody
  plausibility everywhere, whereas a bad `gamma` only ruins one sampling run.
- **Surface 3 may later supply a vocabulary.** A validated SAE could help turn
  "the score went up" into "these concepts moved" or offer an alternative target
  to the compatibility head. Until its features have independent biological
  validation, it is a diagnostic research direction rather than preference truth.

A later composition may form a loop: discovery names a validated concept,
post-training changes the weights, guidance tunes behavior per run, and discovery
is rerun as a diagnostic. The current build does not depend on that loop.

---

## High-Level Roadmap

### Phase 1 - Antibody-Only Pretraining on OAS

Pretrain an MLM in PyTorch on antibody sequences from OAS to learn antibody-specific sequence structure before introducing any binding context.

#### Why start here?

- MLM is a practical first objective for learning contextual residue relationships.
- It supports local infilling behavior, which fits the eventual HCDR3 editing goal.
- It lets the model learn antibody regularities from a much larger corpus before moving to smaller antigen-conditioned datasets.

#### Current implementation

- OAS preprocessing exists in `scripts/prepare_oas.py`.
- MLM training exists in `scripts/mlm_train.py`.
- HCDR3-focused masking is already supported through `hcdr3_span_probability`.

---

### Phase 2a - Paired VH/VL Refinement on OAS

After antibody-only pretraining, refine the encoder on paired OAS examples so it learns heavy/light compatibility rather than only single-chain syntax.

#### Goal of this stage

- move from single-chain plausibility to multi-chain antibody coherence,
- preserve the antibody prior learned during MLM pretraining, and
- prepare the encoder for later antibody-antigen modeling.

#### Current implementation

- Paired OAS preprocessing is handled by `scripts/prepare_oas.py`.
- The paired refinement stage is trained through `scripts/mlm_train.py`.
- The current auxiliary pairing task is native-vs-shuffled heavy/light compatibility, not yet antibody-antigen binding.

---

### Phase 2b - Antibody-Antigen Dataset Construction from ASD

Build a clean antigen-aware training dataset from the ASD: Antigen-Specific Antibody Database.

This repository now includes `scripts/prepare_antibody_antigen.py`, which preprocesses parquet shards from ASD into a cleaned JSONL dataset for later antigen-conditioned training.

#### Why ASD matters here

- ASD provides explicit antibody-antigen examples rather than antibody-only repertoire data.
- The dataset includes heavy/light sequences, antigen sequences, confidence annotations, and nested numbering metadata.
- The nested heavy-chain numbering metadata includes CDR annotations, which makes ASD especially useful for the eventual HCDR3-conditioning objective.

#### Important modeling note

ASD is heterogeneous. It mixes:

- paired antibodies and heavy-only / nanobody examples,
- multiple affinity or assay types,
- binary-style binding labels and continuous measurements,
- and examples from multiple source studies.

Because of that, the first antigen-conditioned stage should stay conservative about supervision rather than forcing all measurements into one regression target too early.

---

### Phase 3 - Antigen-Conditioned Compatibility Modeling

Once the ASD-derived dataset is prepared, train an antibody-antigen model that conditions antibody context on antigen sequence.

#### Planned architecture

- start from the antibody encoder refined on OAS,
- encode the antigen separately,
- fuse antibody and antigen representations through cross-attention or a similar interaction mechanism,
- produce a joint antibody-antigen representation,
- and train conservative early tasks such as compatibility or binder-vs-non-binder prediction.

#### Why this stage is separate from paired refinement

Paired VH/VL refinement teaches internal antibody consistency. Antigen-conditioned modeling teaches whether an antibody context is compatible with a target. Those are related, but not the same problem, so they should remain distinct stages.

---

### Phase 4 - Richer Binding Representation and Supervised Heads

Once a fused antibody-antigen representation is stable, attach smaller supervised heads to predict downstream properties that matter for screening and optimization.

#### Potential target tasks

- binder vs non-binder classification,
- pKd / delta-G regression where labels are sufficiently standardized,
- mutation effect prediction,
- and other assay-aware readouts where the metadata supports them.

The goal here is not only predictive performance, but also to encourage the shared representation to organize around biophysically meaningful factors.

---

### Phase 5a - SAE-Based Concept Discovery

**Status: deferred.** Train an SAE only after the base policy has passed the antigen-conditioning gates. The purpose is to test whether sparse, reusable latent features provide useful diagnostics or independently validated control targets; preference post-training does not wait for this phase.

#### Intended workflow

- collect intermediate activations from the trained model,
- fit a sparse dictionary / SAE,
- identify sparse latent features,
- annotate those features biologically where possible,
- and use them to analyze and eventually steer model behavior.

#### What discovery is for, concretely

A named feature is useful in three distinct ways, and it is worth being explicit about which one a given experiment is claiming:

1. **As a diagnostic.** Run the SAE before and after preference post-training and report which features moved. This can detect representational change, but it does not by itself establish that the change improved binding.
2. **As a steering target.** A feature may become an alternative to the compatibility head only after an independent biological evaluation shows that it represents the intended property.
3. **As a pair-proposal tool.** Feature-contrasting candidates may be useful examples for an assay or independent judge. An SAE score alone cannot create preference truth or evaluate training performed from its pairs.

This stage is where representation learning and interpretability meet most directly in the project.

---

### Phase 5b - Preference Post-Training (DPO and Relatives)

**Status: planned, not implemented.** No code in `src/` or `scripts/` trains on preferences today. This section exists because it constrains what the earlier phases must produce, not because it is close to running.

This README explains the rationale, evidence hierarchy, and failure modes. It intentionally does not specify the stochastic estimator. Once scientifically approved, the sole implementation contract will be the tracked `specs/diffusion_dpo.md`; that file does not exist yet, so preference training is currently blocked by design.

This phase asks whether the generator can be made to prefer independently validated outcomes by changing its weights rather than only reweighting samples at decode time. It is the second steering surface from [Where Steering Lives](#where-steering-lives) and does not depend on the deferred SAE work.

#### Terms used below

- **DPO** (Direct Preference Optimization) — training on pairs where one output is marked better than the other, without fitting a separate reward model.
- **Preferred / dispreferred** — the two members of such a pair.
- **Policy** — the model being trained. **Reference** — a frozen copy of it from before training started, which the loss compares against so the policy cannot drift arbitrarily.
- **P0 contract** — the future reviewed `specs/diffusion_dpo.md` specification that fixes the objective, corruption coupling, Monte Carlo budget, reductions, reference behavior, and exact toy tests before implementation.

#### Why consider this phase

Guided generation cannot exceed what the base model already considers possible. Each sampled residue is scored as `unguided + gamma * binder`, where `gamma` is the guidance strength and `binder` is the compatibility signal. Both terms depend only on the current state, so the *spread* of the binder term across candidate residues is a hard ceiling on what any `gamma` can achieve — and `scripts/probe_steering_reachability.py` measures that ceiling directly. Once the probe reports a decision as unreachable, no sweep recovers it. Changing the base distribution is the remaining class of intervention; simpler supervised continuation and rejection methods must be compared before DPO.

#### Where preference pairs come from

Three candidate sources, listed from most trustworthy and scarcest to least trustworthy and most plentiful:

- **Direct, co-measured experimental orderings.** Use pairs from the same antigen, assay context, and editable framework only when uncertainty or censoring leaves a clear order. `affinity_strength_quantile` can organize eligible measurements within an assay family, but it does not make heterogeneous assays comparable.
- **Measured binary labels.** Binder versus non-binder rows can form a coarse ordering only when target and sequence context are matched and both labels are experimental.
- **Model- or feature-ranked generations.** `decision_score`, guide scores, and SAE features can propose candidates for an assay or independent judge. They cannot supply canonical preference truth and then evaluate the policy trained from it.

#### Objective authority

DPO ordinarily consumes policy and reference sequence log-probabilities. A masked denoiser does not expose the same tractable left-to-right likelihood, so preference training requires a reviewed masked-diffusion surrogate. `E-hat` from `smallAntibodyGen.infill.evidence` remains an evaluation-time order-mixture lower bound; it is not the training objective, and `content_seed` is not a training coupling mechanism.

The distinction remains load-bearing: a difference of lower bounds is not a bound on a likelihood difference, and passing Monte Carlo estimates through the nonlinear preference loss introduces bias. The P0 contract must therefore define and test the absorbing process, timestep weighting, preferred/dispreferred sampling relationship, policy/reference reuse, draw count, length reduction, and zero-mask behavior. This README deliberately chooses none of them.

Before trainer code exists, an exactly enumerable toy must separate the variational-surrogate gap from finite-sample nonlinear bias. The implementation must reproduce that specification and its symmetry, frozen-reference, and variance tests.

#### Prerequisites outside this phase

Preference post-training begins only after all of the following are true:

1. Data manifests, split firewalls, checkpoint lineage, numerical compute budgets, and blind evaluators are reproducible.
2. The additive antibody/paired/antigen chain passes frozen retention budgets, including conditional denoising only on eligible binder rows.
3. The antigen changes policy outputs in the correct direction on measured held-out antigen variants, beyond identity and source-study baselines.
4. A named masked-diffusion loss and matching reverse sampler have beaten or justified replacing the partial-state MLM control without erasing antigen dependence.
5. Preference pairs have auditable experimental or independent-judge provenance; the guide that proposes a pair is not its final judge.
6. `specs/masked_diffusion.md` and `specs/diffusion_dpo.md` are approved, tracked, fingerprinted, and covered by enumerable toy tests.

#### Remaining engineering costs

- **A frozen reference model doubles resident parameters**, on top of an antigen encoder that may already be ESM-2.
- **Scores cover the HCDR3 span only.** The preference gradient shapes what goes inside the edited region and says nothing about the rest of the antibody. That is mostly the intended scope, but it means degradation elsewhere needs its own metric — the preference loss will not report it.

#### Cheaper methods to try before DPO

Preference optimization is a family, and DPO is its most expensive member. In increasing order of cost:

1. **Rejection-sampling fine-tuning** (also called best-of-`n`, or RAFT). Generate candidates, keep the top-scoring ones, and continue ordinary MLM training on them. No reference model, no likelihood ratio, no new loss — it reuses the existing objective in `scripts/mlm_train.py`. This is the natural first experiment, because it establishes whether the scorer is even good enough to be worth optimizing against.
2. **Reward-weighted MLM loss.** Weight the existing masked-token loss by `affinity_strength_quantile`. Uses real measurements only, and still needs no reference model.
3. **Pairwise preference optimization**, using only the approved P0 estimator and frozen reference.

#### The failure mode to design against

If the compatibility head labels the pairs, and the same head also steers generation, then post-training optimizes one model's errors and guidance amplifies them. Prefer direct measurements over model-ranked pairs and keep an evaluation judge that never saw the pairs. `--guidance-checkpoint` separates the policy from an external guide, which is necessary but insufficient: it does not make that guide an independent final judge.

The reachability probe remains a useful diagnostic: a score that moves while candidate-local reachability does not is evidence that the policy may have learned the judge. It is not the graduation criterion. Promotion requires improvement on evaluator-hidden experimental preferences or measurements over the base policy and cheaper baselines, while antigen dependence, sequence quality, and every frozen retention budget continue to pass.

---

### Phase 6 - Antigen-Conditioned HCDR3 Infilling and Lead Optimization

The generation setting is intentionally narrow at first.

Rather than unconstrained de novo generation, the early focus is:

- start from an existing antibody context,
- condition on antigen information,
- mask and infill the heavy-chain CDR3 span,
- preserve broader antibody plausibility,
- and gradually expand toward more controllable optimization.

This makes the generation problem more realistic for lead refinement and better aligned with the project goal of steerable design.

---

## Data Pipeline

Basic data pipeline detailed below:

- `scripts/prepare_oas.py`
  Cleans raw OAS data into processed antibody-only or paired heavy/light JSONL files.

- `scripts/prepare_antibody_antigen.py`
  Cleans ASD parquet shards into processed antibody-antigen JSONL files, keeps heavy/light plus antigen context, preserves nested numbering metadata, computes HCDR3 spans when possible, and assigns leakage-aware splits.

  KD values arrive either in molar (`1e-9`) or already in nanomolar (`1.0`), so units are
  inferred per row by magnitude. Pass `--strict-units` to make a suspected unit mislabel a
  hard error instead of a warning: without it, a dataset whose units are misread yields **zero
  strong binders**, and the HCDR3 infill stage then trains on an empty population without
  complaining. Use it whenever you ingest a new ASD export.

- `scripts/mlm_train.py`
  Trains the antibody MLM, paired VH/VL refinement, antigen-conditioned compatibility refinement (synthetic-negative and real-label), and the fixed-length antigen-conditioned HCDR3 infill stage.

- `scripts/hcdr3_infill.py`
  Generates fixed-length or empirical-length HCDR3 candidates from a trained antigen-conditioned infill checkpoint, with optional compatibility scoring.

---

## Training options (all stages)

`scripts/mlm_train.py` shares the knobs below across every stage. Each is
**opt-in and defaults to the historical behavior**, so existing configs are
byte-for-byte unchanged unless you set them. Every knob also has a matching CLI
flag that overrides the config value.

- **LR schedule** — `lr_schedule: constant` (default) keeps warmup-then-flat LR.
  `lr_schedule: cosine` decays from the peak LR down to `min_lr_ratio × peak`
  (default `0.0`) over the full run after `warmup_steps`. Flags: `--lr-schedule`,
  `--min-lr-ratio`.
- **Early stopping** — `early_stopping_patience: N` (default `0` = off) stops when
  validation loss has not improved for `N` consecutive epochs;
  `early_stopping_min_delta` (default `0.0`) is the minimum improvement that
  counts. `best.pt` remains the val-loss-optimal checkpoint, so you can set a
  generous `epochs` ceiling and let the run self-truncate. Flags:
  `--early-stopping-patience`, `--early-stopping-min-delta`.
- **Intra-epoch checkpointing** — `checkpoint_every_steps: N` (default `0` = off)
  rewrites `last.pt` every `N` batches. Resume is epoch-granular: after a crash a
  resumed run re-enters the in-progress epoch from its first batch with the saved
  weights, so no weight progress is lost. Checkpoint writes are **atomic**
  (temp file + `os.replace`), so a crash mid-write cannot corrupt `last.pt`. Flag:
  `--checkpoint-every-steps`.
- **TensorBoard** — `tensorboard: true` (default `false`) logs train/val loss, val
  MLM accuracy, and LR to `<output_dir>/tb`. Needs the optional `tb` extra:

  ```bash
  pip install -e ".[tb]"
  python scripts/mlm_train.py --config <config>.yaml   # config sets tensorboard: true
  tensorboard --logdir checkpoints                      # overlays all stages' runs
  ```

  Flag: `--tensorboard`.
- **Norm placement** — `norm_first: true` (default) is pre-LN,
  `x + sublayer(LayerNorm(x))`, which keeps an unnormalized identity path from
  input to output so gradients reach early layers without being rescaled at each
  depth. `norm_first: false` is post-LN, `LayerNorm(x + sublayer(x))`: the
  original Transformer arrangement and the PyTorch default. It governs both
  encoder stacks **and** the cross-attention fusion block, so the two cannot
  drift apart. Accepted at the top level or inside the nested `model:` section.
  Flags: `--norm-first`, `--no-norm-first`.

  > The default is pre-LN because every checked-in config sets it. A `false`
  > default only ever fired for a hand-written config that omitted the key, and it
  > handed that run the post-LN stack silently.

  > **Changing this is a from-scratch retrain of the whole chain, not a
  > migration.** Post-LN and pre-LN produce *identical* parameter names and
  > shapes in the single-stream model, so a checkpoint trained under one setting
  > loads into the other under `strict=True` reporting "All keys matched
  > successfully" and then computes something different. The init-compat check
  > in `scripts/mlm_train.py` is what makes the mismatch fatal, and it treats a
  > checkpoint with no `norm_first` key as post-LN (every pre-knob checkpoint
  > was). Warm-starting a pre-LN stage from a post-LN checkpoint is an error, by
  > design.
- **Compatibility readout** — `compat_readout: cls` (default) is the historical
  CLS-concat: the two fused streams are collapsed to their index-0 summaries
  before `fusion_mlp`. `compat_readout: mean` mask-aware mean-pools each fused
  stream over its non-pad positions instead. The knob exists because the
  CLS-concat readout is only weakly sensitive to a single-residue substitution —
  and that logit is exactly the term guided decoding steers with. No parameters
  change in either mode, so a checkpoint loads under both; the init-compat check
  therefore validates `compat_readout` the way it validates `norm_first`.
  Flag: `--compat-readout {cls,mean}`.
- **Weight initialization** — `initializer_range: 0.02` (default) applies
  `N(0, 0.02)` to every embedding table, `Linear` weight, and cross-attention
  projection, zeroes biases, sets `LayerNorm` to `(1, 0)`, and re-zeroes
  `padding_idx` rows. `scale_residual_init: true` (default) additionally scales the
  residual-branch *output* projections (`attn.out_proj`, `ffn.linear2`) by
  `1/sqrt(2 * n_layers)`.

  > Before this existed the models used PyTorch's per-module defaults, which put
  > `nn.Embedding` at `N(0, 1)`. With `tie_weights: true` that unit-variance table
  > **is** the output projection, so a fresh `d_model: 256` model produced logits
  > of std ~29 and an untrained MLM loss of **~149 nats** on the real OAS corpus
  > against an ideal `ln(vocab) ≈ 3.56`. The opening phase of every from-scratch
  > run went into shrinking the embedding norm, and `grad_clip_norm: 1.0` clipped
  > exactly the gradients that would have done it. Now a fresh model starts at
  > ~3.5.
  >
  > Unlike `norm_first`, this is **not** a cross-generation boundary: names,
  > shapes, and the forward are unchanged, so warm-starting and resuming from any
  > existing checkpoint still work (the loaded weights overwrite the init). Only
  > the from-scratch stage is affected, and it is deliberately not an init-compat
  > key.
- **Loss weights** — `mlm_loss_weight` (default `1.0`) scales the MLM term on the
  antigen stages; `0.0` lets the compatibility term alone drive training. The
  reported `mlm_loss` stays unweighted so curves remain comparable. Flag:
  `--mlm-loss-weight`.
- **New-module LR** — `new_module_lr_multiplier` (default `1.0`) gives the
  modules the antigen warm-start leaves randomly initialized (cross-attention,
  fusion norms, `fusion_mlp`, `compatibility_head`) their own learning rate, so a
  warm-started trunk is not dragged by a head still finding its scale. `1.0`
  keeps the historical two-group optimizer exactly. Flag:
  `--new-module-lr-multiplier`.
- **Checkpoint selection metric** — `best_checkpoint_metric: val_loss` (default)
  or `val_compat_loss`. On the antigen stages the combined loss is dominated by
  the MLM term, so "best" can advance on an epoch where the compatibility head
  got *worse* — and that head is what scoring and guidance actually read. The
  knob routes `best.pt`, early stopping, and the checkpoint's tracked `val_loss`
  through one selection value, so resume stays coherent. Flag:
  `--best-checkpoint-metric`.
- **Masking-rate schedule** — `mask_rate_schedule: fixed` (default) keeps the
  per-row target budget from `mask_probability` and consumes zero extra collator
  RNG draws, so existing runs are byte-identical. `uniform` draws a per-row rate
  `t ~ U(0, 1]` and ignores `mask_probability` — the schedule-covering corruption
  a masked-diffusion denoiser needs, so one model is trained across the whole
  ladder rather than only at 15%. Inert in `full_span` HCDR3 mode.
  `eval_mask_rate_schedule` (default `""` = inherit) sets the eval-side schedule
  independently, so arms differing in train schedule can share one
  arm-independent eval protocol. `report_masked_fraction_bins: true` (default
  `false`) additionally emits per-epoch MLM accuracy binned by each row's
  realized masked fraction (`mlm_acc_frac_0_20` … `mlm_acc_frac_80_100`, plus
  token counts). Flags: `--mask-rate-schedule`, `--eval-mask-rate-schedule`,
  `--report-masked-fraction-bins` / `--no-report-masked-fraction-bins`.
- **Graded-affinity supervision (experimental)** — `strength_loss_weight`
  (default `0.0`) adds a scalar regression head on the joint representation,
  trained against per-`(dataset, affinity_type)` strength quantiles produced by
  `scripts/annotate_affinity_targets.py`. Weight `0.0` builds **no** head at all
  (zero extra init RNG), so every existing run is byte-identical.
  `include_strength_rows` (default `false`) separately widens the stage-3 row
  filter to admit rows carrying a quantile but no binary label — kept separate on
  purpose, because flipping both at once confounds "the graded head helps" with
  "the larger population helps". Reports a pooled tie-aware
  `val_strength_spearman`. Flags: `--strength-loss-weight`,
  `--include-strength-rows` / `--no-include-strength-rows`.

  ```bash
  python scripts/annotate_affinity_targets.py \
    --input  data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
    --output data/processed/antibody_antigen/antibody_antigen_quantiled.jsonl.gz
  ```

  The annotator fits its CDFs on the **train split only**, negates the score
  where lower means stronger (raw KD) so `1.0` is always the strongest, uses
  mid-ranks so ties are interchangeable, excludes groups with fewer than
  `--min-group-size` train rows, and refuses to write in place. Stripping the
  added key reproduces the input byte-for-byte.
- **Learned length posterior (experimental)** — `length_loss_weight` (default
  `0.0`) adds a categorical head over `1..length_head_max` predicting the HCDR3
  length from `(scaffold, antigen)`. It is queried on a **separate forward** over
  the *collapsed-span* encoding (the whole HCDR3 replaced by exactly one
  `[MASK]`), because on the ordinary encoding the number of mask tokens *is* the
  answer. Out-of-range lengths are **masked out**, never clamped. Reports
  `length_acc` and `length_nll`. Flags: `--length-loss-weight`,
  `--length-head-max`.

  Choose `length_head_max` from a census rather than by eye:

  ```bash
  python scripts/length_census.py \
    --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz
  ```

  It reports the length distribution per split and, separately, for the
  strong-binder population the infill stage actually trains on, plus the fraction
  of rows each candidate `length_head_max` would exclude.

---

## Antigen-Conditioned HCDR3 Infilling (implemented)

The first generation task from Phase 6 is implemented as a fixed-length,
antigen-conditioned HCDR3 infiller on top of the dual-stream antibody/antigen
model.

### Training stage: `antigen_hcdr3_infill_refine`

- Fine-tunes the dual-stream antibody/antigen model on strong-binder rows only.
  Positives are gated on `is_strong_binder`, which covers explicit boolean
  positives **and** the KD / -log KD / fuzzy strong binders. Gating on
  `binder_label == 1` (set only for `affinity_type == "bool"` rows) would silently
  drop the large majority of strong binders, so the broader flag keeps the
  training population representative of observed binders.
- Keeps the antibody framework, optional light chain, and antigen visible.
- Masks the entire known heavy-chain CDR3 span (`hcdr3_mask_mode=full_span`,
  `mask_replacement_strategy=always_mask`) and trains the MLM head to reconstruct
  those residues.
- Sets compatibility loss to `0.0`: the goal is residue infilling for strong
  binders, not binder-vs-non-binder classification, which stays a separate
  scoring step.
- Heavy-only / nanobody records are supported and encode with their real heavy
  chain token (`[IGH]`) at both training and generation, so the masked-input
  distribution the model learns matches the one it is asked to infill.
- Reports HCDR3-specific metrics: token accuracy, full-span exact match,
  target-token count, and valid-span count.

This is *fixed-length* infilling: the number of `[MASK]` tokens equals the HCDR3
length, so the model learns which residues belong in a known-size hole.
Unknown-length design is handled as a separate proposal step — choose candidate
lengths first, then run the same fixed-length infiller for each proposed length.

### Generation infrastructure (`src/smallAntibodyGen/infill/hcdr3.py`)

- `FixedLengthHCDR3Infiller` builds the masked antibody/antigen input once and
  samples HCDR3 residues from the shared MLM logits (one forward per record,
  regardless of the number of samples).
- `LengthProposalStrategy` is the interface for swappable length predictors.
- `EmpiricalHCDR3LengthPrior` samples lengths from strong-binder HCDR3s — the same
  `is_strong_binder` population the infiller trains on.
- `AntigenCompatibilityScorer` ranks generated candidates with the real-label
  compatibility head.
- `guided_infill` is the opt-in, ProteinGuide-style guided sampler (see
  [Guided generation](#guided-generation-proteinguide-style-opt-in) below):
  iterative easy-first unmasking that steers each residue toward the binder
  class. The single-pass `infill` remains the default.

### Running infill refinement

```bash
python scripts/mlm_train.py --config configs/refine_antigen_hcdr3_infill.yaml
```

### Antigen encoder: from-scratch or pretrained ESM-2 (hybrid)

The antigen stream can be encoded two ways, selected by `antigen_encoder_type`:

- `scratch` (default) — the in-repo from-scratch transformer. No extra dependencies.
- `esm` — a pretrained ESM-2 encoder projected to the model width (Direction 1: hybrid
  antigen encoder). Requires the optional `esm` extra and swaps the antigen stream only;
  the antibody stream, cross-attention fusion, and heads are unchanged.

```bash
pip install -e ".[esm]"    # transformers + peft; ESM-2 8M weights download on first use
python scripts/mlm_train.py --config configs/refine_antigen_hcdr3_infill_esm.yaml
```

The ESM config warm-starts the antibody encoder + fusion + heads from the real-label
checkpoint and keeps the ESM backbone at its pretrained weights (`finetune: frozen` trains
only the projection/fusion/heads; `finetune: lora` adds LoRA adapters on the ESM backbone).
Compare its HCDR3 metrics and compatibility AUROC against the scratch baseline
(`refine_antigen_hcdr3_infill.yaml`) on the same split before committing to it. See
`docs/antigen-encoder-hybrid-implementation.md` for the full rollout and rationale.

### Unknown-length design

`--length-mode` selects how many `[MASK]` tokens to place:

- `fixed` (default) — the record's known HCDR3 length.
- `empirical` — sample from a context-free histogram of training-set HCDR3
  lengths (`EmpiricalHCDR3LengthPrior`, fitted on the strong-binder population).
- `learned` — sample from the model's *conditional* posterior
  `p(L | scaffold, antigen)` via `LearnedLengthProposal`. This requires a
  checkpoint trained with `length_loss_weight > 0`; the checkpoint's own
  `length_head_max` is authoritative, and asking for a larger one is a
  construction-time error rather than an `IndexError` deep inside proposal. The
  posterior is filtered to lengths whose masked encoding actually fits
  `max_length` and renormalized, so no probability mass is spent on lengths that
  could never generate. `--learned-length-mode top_k` makes it deterministic.

### Generating candidates

Fixed-length (uses each target record's known HCDR3 length):

```bash
python scripts/hcdr3_infill.py \
  --checkpoint checkpoints/mlm_antigen_hcdr3_infill_v3/best.pt \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --split val \
  --num-records 20 \
  --num-samples 16 \
  --length-mode fixed \
  --score-checkpoint checkpoints/mlm_antigen_real_label_v3/best.pt \
  --output-path outputs/hcdr3_fixed_candidates.jsonl
```

Empirical unknown-length (samples proposed lengths from the strong-binder
training distribution, then infills each proposed length):

```bash
python scripts/hcdr3_infill.py \
  --checkpoint checkpoints/mlm_antigen_hcdr3_infill_v3/best.pt \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --split val \
  --num-records 20 \
  --num-samples 16 \
  --length-mode empirical \
  --score-checkpoint checkpoints/mlm_antigen_real_label_v3/best.pt \
  --output-path outputs/hcdr3_empirical_length_candidates.jsonl
```

Each JSONL row includes the record identity, target metadata, the true HCDR3 (if
known), proposed length, generated HCDR3, generated heavy-chain sequence, MLM log
probability, length-normalized mean log probability, optional compatibility
score, and the guidance provenance (`guidance_strength`, `guidance_order`).
Because raw `log_probability` grows with length, rank across different proposed
lengths by `mean_log_probability`, not the raw sum.

### Guided generation (ProteinGuide-style, opt-in)

This is **steering surface 1** of the three in
[Where Steering Lives](#where-steering-lives): it changes what gets sampled and
never changes a weight. Its limits are what motivate surface 2, preference
post-training ([Phase 5b](#phase-5b---preference-post-training-dpo-and-relatives)).

`guided_infill` turns the binder signal from a *post-hoc ranker* into an
*in-sampling guide*. It follows ProteinGuide (Xiong et al., 2025,
[arXiv:2505.04823](https://arxiv.org/abs/2505.04823)), whose key observation is
that a masked language model is equivalent to a discrete (masked) diffusion
model — so classifier guidance can steer generation at inference time with **no
retraining of the generative model**.

Instead of one forward pass with independent per-position sampling, guided
generation unmasks the HCDR3 **iteratively, one position per step**:

1. Pick the next position to fill (default `confidence`: the lowest-entropy,
   most-certain remaining position — MaskGIT-style easy-first decoding).
2. Reweight that position's residue distribution by the binder signal. For each
   canonical residue `a`,

   ```
   score(a) = log p_MLM(a | x) + gamma * log p(binder | x with this position = a)
   ```

   The binder term is computed by **exact enumeration** — one batched forward
   over all ~20 candidate residues — which is tractable precisely because the
   amino-acid vocabulary is tiny.
3. Sample the residue, commit it, and repeat so later positions condition on it.

`gamma` is `--guidance-strength` (`0` disables guidance and restores the default
single-pass `infill`; larger values steer harder). The order is
`--guidance-order` (`confidence`, `random`, or `left_to_right`).

```bash
python scripts/hcdr3_infill.py \
  --checkpoint checkpoints/mlm_antigen_hcdr3_infill_v3/best.pt \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --split val \
  --num-records 20 \
  --num-samples 16 \
  --length-mode fixed \
  --guidance-strength 5.0 \
  --guidance-order confidence \
  --output-path outputs/hcdr3_guided_candidates.jsonl
```

Things to keep in mind:

- **By default the guidance predictor is the generation model's own
  compatibility head.** `--score-checkpoint` is unrelated: it only attaches a
  post-hoc compatibility score for reporting and never influences sampling.
- **`--guidance-checkpoint` attaches an EXTERNAL classifier** to supply the
  binder term instead, so the steerer and the judge can be different heads. Only
  the binder term switches: position selection and the reported unguided
  marginals still come from the generation model. It requires
  `--guidance-strength > 0` (at γ = 0 no classifier is consulted, so accepting it
  would label sweep rows with a checkpoint that touched nothing), and each output
  row records `guidance_checkpoint` for provenance.
- **The head is trained on a different state distribution than it is queried
  on.** The compatibility head is trained on fully HCDR3-masked inputs, whereas
  guidance queries it on *partially* filled intermediate states, so the signal is
  noisiest at the earliest steps. Two levers now exist for this:
  `hcdr3_mask_mode: partial_span` trains on a uniformly random `k`-subset of the
  span (`k ~ U{0..L}`, so both the fully-visible and fully-masked endpoints are
  reachable), which is the state distribution decoding actually visits; and
  `--guidance-checkpoint` lets a separately trained classifier do the steering.
- **Before funding a γ sweep, measure whether γ can move anything.**

  ```bash
  python scripts/probe_steering_reachability.py \
    --checkpoint checkpoints/mlm_antigen_hcdr3_infill_v3/best.pt \
    --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
    --split val --num-records 20
  ```

  Both terms of `guided = unguided + γ · binder` are functions of the state, not
  of γ, so **two** forwards give the exact guided distribution for every γ on a
  grid. The probe reports the flip fraction, total variation, and the *spread* of
  the binder term — the ceiling on what any γ can do. Exact only for a fixed
  context (in a real run the committed residues are themselves γ-dependent), so
  it answers "can γ move this decision", not "what does a full sweep generate".
- **Score reporting.** `log_probability` / `mean_log_probability` are always
  accumulated from the model's *unguided* marginals, so guidance changes which
  residues are drawn without inflating the reported likelihood. They are **not**
  the same quantity as `infill`'s score, though: `infill` sums independent
  per-position marginals from one fully-masked forward, while `guided_infill`
  sums the unguided conditionals along the iterative unmasking path. The two
  summations differ **even at γ = 0**, so do not pool guided and single-pass
  candidates into one ranking by these fields.
- **To rank across samplers, use evidence instead.**
  `smallAntibodyGen.infill.evidence` computes `E-hat`, the order-averaged path
  log-likelihood over `K` random unmasking orders — sampler-independent by
  construction, with a Monte-Carlo standard error so a rank gap smaller than the
  error is visibly not a gap. `scripts/score_candidates.py` attaches it plus an
  explicit weighted decision score:

  ```bash
  python scripts/score_candidates.py \
    --candidates outputs/hcdr3_guided_candidates.jsonl \
    --checkpoint checkpoints/mlm_antigen_hcdr3_infill_v3/best.pt \
    --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
    --split val --num-orders 8 --w-match 1.0 --w-evidence 0.5 \
    --output outputs/hcdr3_candidates_scored.jsonl --report-demotion 5
  ```

  The weights are required, not defaulted — there is no calibrated default to
  pretend to. A row missing its compatibility score gets `decision_score: null`
  and `decision_score_omitted: ["compatibility"]` rather than a partial score
  that looks complete. `--report-demotion k` measures the real tension: evidence
  rewards *typicality*, so ranking by it can demote exactly the unusual
  candidates steering exists to find.

---

## Leakage-Aware Splitting

For antibody-antigen data, random row splitting is often too optimistic.

Examples from the same target, the same study, or the same antibody family can easily appear in both train and validation if the split is done naively. In that case, validation may measure memorization of repeated biological problems or source-specific conventions rather than true antigen-conditioned generalization.

The ASD preprocessing path therefore uses target-aware split assignment, preferring stable target identifiers such as:

- UniProt when available,
- then PDB identifiers,
- then normalized target names,
- and finally an antigen-sequence hash as a fallback.

This is not a perfect solution, but it is a much better default than row-wise random splitting and helps keep future validation results more honest.

### Paired OAS: split key and length budget

Two corpus-level defects were found and fixed in `scripts/prepare_oas.py`. **Both
require regenerating the paired corpus** — the fixes are in the producer, so any
already-written `oas_paired.jsonl.gz` still has them baked in.

**1. The paired split is now keyed on the HEAVY chain.** It used to be keyed on the
full `(heavy, light)` pair, so one heavy chain observed with several cognate light
chains was scattered across both splits. On the shipped corpus, 1,406 heavy
sequences appeared in both splits and 6.3% of val rows had a byte-identical heavy
chain in train — so stage-2 val loss, which selects `best.pt`, was partly measuring
memorization of the exact HCDR3 target. The unpaired path never had this defect (it
keys on `f"{locus}:{variable_aa}"`); the paired path now matches it. Dedup is
unchanged and still keys on the full pair, which is correct there — a distinct
`(heavy, light)` pair is a distinct example.

**2. `max_length` silently truncates almost the whole paired corpus.** `prepare_oas.py`
bounds heavy and light independently (`--max-heavy` 180 + `--max-light` 160 + 5
specials = up to 345 tokens) and writes `token_length` unclamped; nothing ties the
corpus to any encoder budget. At the shipped `max_length: 192`:

| | paired corpus |
|---|---|
| rows | 306,760 |
| exceed `max_length=192` | 306,666 (**99.97%**) |
| lose their **light CDR3 entirely** | 306,057 (**99.77%**) |

The paired stage exists to learn heavy/light compatibility, and CDR-L3 is the region
that most determines pairing — so the shuffled-negative task was asking the model to
tell cognate from non-cognate light chains with CDR-L3 deleted from both. This was
invisible because Python deduplicates the tokenizer's truncation `UserWarning`.

`scripts/mlm_train.py` now runs a **preflight** at startup that reports, per split,
how many rows overflow and how many lose their heavy or light CDR3. It warns rather
than refuses, because the remedy is a research decision:

- raise `max_length` (a retrain, and more compute per step — attention is quadratic),
- tighten `--max-heavy` / `--max-light` in `prepare_oas.py`, or
- filter rows whose `token_length` exceeds `max_length`.

Note that `ChainLengthBucketBatchSampler` buckets on the stored `token_length`, so
while the corpus overflows, it is bucketing on a length no row actually encodes to.

---

## Model Sketch

```text
OAS antibody sequences
        |
        v
Antibody MLM pretraining
        |
        v
Paired VH/VL encoder refinement
        |
        v
ASD antibody-antigen preprocessing
        |
        +--------------------+
        |                    |
        v                    v
Antibody representation   Antigen encoder
        |                    |
        +---- cross-attention+
                 fusion
                   |
                   v
         Joint antibody-antigen embedding
                   |
        +----------+-----------+
        |                      |
        v                      v
  Supervised property heads    SAE on activations
   (compat / strength /          |  (deferred until conditioning passes)
    length)                      v
        |              Independently validated features
        |                      |
        +----------+-----------+
                   |
                   v
      HCDR3 infilling and constrained lead optimization
                   |
                   |  <-- surface 1: guidance reweights sampling (no weight change)
                   v
      Candidates scored by compatibility + evidence (E-hat)
                   |
                   v
      Preference pairs  (experimental | independently judged)
                   |
                   |  <-- surface 2: approved masked-diffusion post-training
                   v
          Updated generator weights
                   |
                   +----> back to HCDR3 infilling (control loop)
                   |
                   +----> optional later SAE diagnostic (did representations move?)
```

The two `<--` annotations are the point of the diagram: surface 1 is a read-only
detour around the generator, while surface 2 is a gated cycle back through its
weights. The SAE branch is not on the critical path.

---

## Why This Approach?

### Why MLM first?

Starting with an MLM is a sensible first objective because it:

- is more data-efficient than immediately scaling a generative model,
- teaches contextual residue relationships,
- supports local infilling behavior,
- and matches the short-term goal of targeted HCDR3 editing.

### Why chain tokens?

Heavy and light chains do not obey identical sequence statistics. Explicit chain tokens let the model condition on chain identity instead of forcing it to infer that structure implicitly every time.

### Why separate paired refinement from antigen conditioning?

Heavy/light compatibility and antibody-antigen compatibility are different relational problems. Separating them lets the model first learn antibody coherence and only then learn binding context.

### Why use ASD for the antigen-conditioned stage?

ASD provides explicit antibody-antigen examples, including structured metadata and frequent heavy-chain CDR annotations. That makes it a strong bridge between antibody representation learning and antigen-conditioned HCDR3 modeling.

### Why SAEs?

SAEs offer a path toward sparse, more interpretable internal features. The hope is that these features can be:

- inspected,
- labeled,
- linked to meaningful biology,
- and used to steer sequence optimization more deliberately.

They are a deferred *discovery* method, not preference supervision or a graduation metric. An SAE by itself changes no behavior, and a sparse feature is not biologically meaningful until an independent evaluation establishes that relationship.

### Why preference post-training as well, rather than guidance alone?

Because guidance and post-training are limited by different things, and only one of those limits is fixable by turning a knob.

Guidance is cheap, reversible, and tunable per run — `gamma` is a command-line flag and a bad value costs one sampling run. But it can only redistribute mass the base model already assigned, and `scripts/probe_steering_reachability.py` will tell you when there is nothing to redistribute. At that point the sweep is finished before it starts.

Preference post-training attacks the other end: it changes what the base model puts mass on. It is expensive, it needs pairs, it is a global edit that can degrade antibody plausibility far from the region of interest, and it is the surface most vulnerable to optimizing a flawed judge. In exchange it is the only mechanism in the design that can raise the ceiling instead of pressing against it.

The two also compose in the obvious direction: post-train to make a behavior reachable, then guide to dial it per target.

### Why DPO is not a drop-in here

DPO ordinarily needs each sequence's log-probability. A masked language model does not supply a tractable normalized joint by construction. Under this repository's implied any-order mixture, different reveal orders give different factorizations; `E-hat` from `smallAntibodyGen.infill.evidence` is a lower bound on that mixture likelihood, not the likelihood itself, as its module docstring is careful to state.

That leaves two jobs for two different quantities, and conflating them is the mistake to avoid:

- **Evaluation uses `E-hat`.** It was built to make candidate rankings independent of which sampler produced them, and that is what it should keep doing.
- **Training must not.** Computing `E-hat` costs one forward pass per hidden position per order and does not define the approved masked-diffusion preference estimator.

The future tracked `specs/diffusion_dpo.md` is the sole authority for corruption coupling, draw count, weighting, and reduction. Until that P0 contract and its enumerable toy tests exist, there is no canonical DPO implementation. See [Phase 5b](#phase-5b---preference-post-training-dpo-and-relatives) for the rationale and prerequisites.

---

## Long-Term Vision

The broader aim is to develop a system that can move beyond black-box scoring and toward interpretable antibody design:

- understand what the model has learned,
- validate antigen-conditioned behavior against independent measurements,
- use validated signals to steer generation in a controlled way at decode time and in the weights, and
- later test whether sparse internal features provide additional interpretable diagnostics.

---

## Notes

This project is research-oriented and iterative by design. Architectural choices, supervision targets, split policies, and representation formats may evolve as the data and experiments mature.
