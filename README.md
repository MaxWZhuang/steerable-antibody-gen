## Introducing Steerability to Antibody Generation via SAE-Derived Concept Labeling

This project explores how to make antibody generation more interpretable and more controllable, with an eventual focus on antigen-conditioned antibody design and targeted HCDR3 editing.

The core idea is to first learn strong antibody sequence representations, then fuse those representations with antigen context, and finally use sparse feature methods such as SAEs to expose more interpretable internal concepts that could support steering.

Rather than jumping directly to a large autoregressive generator, the project starts with masked language modeling (MLM) on antibody sequences from OAS (Observed Antibody Space). That first stage is meant to teach antibody sequence grammar, chain-specific structure, and local residue constraints well enough to support later antigen-aware refinement.

---

## Project Goals

The long-term goal is not just to generate antibody sequences, but to build a system that can:

- learn antibody sequence biology and chain-specific grammar,
- incorporate antigen and assay context,
- predict useful downstream properties,
- expose biologically meaningful latent concepts, and
- support controlled optimization of promising binders.

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

#### Current implementation status

- `scripts/mlm_train.py` includes the original `antigen_refine` stage, which trains a synthetic native-vs-shuffled antigen compatibility objective.
- `scripts/mlm_train.py` also includes `antigen_real_label_refine`, which trains the same dual-stream cross-attention model using only experimental `binder_label` rows where the label is exactly `0` or `1`.
- `antigen_real_label_refine` filters out unlabeled / non-binary ASD rows for the compatibility objective and uses measured binders/non-binders instead of shuffled strong-binder negatives.
- Compatibility accuracy is aggregated over all labeled compatibility rows in the epoch, not as a simple average of per-batch accuracies.
- Antigen-stage metrics now include accuracy, balanced accuracy, precision, recall, specificity, MCC, AUROC, AUPRC, positive rate, and labeled-example count.
- Per-epoch metrics are written to `metrics.jsonl` in the output directory for easier analysis after training.

#### Compatibility task variants

- `antigen_refine` remains useful as a synthetic diagnostic, but it is shortcut-prone. Positives are strong-binder rows, and negatives are created by shuffling antigens across strong-binder rows while loosely matching format and antigen length.
- Because of that, high `antigen_refine` compatibility metrics should be interpreted as success on a synthetic cognate-vs-shuffled discrimination task, not as proof of real binder-vs-nonbinder generalization.
- `antigen_real_label_refine` is the preferred next training stage for real binder classification. It treats `binder_label=1` as positive and `binder_label=0` as negative, and it does not create shuffled antigen negatives.
- This is still not a final biological benchmark: the labels remain assay-heterogeneous and target-held-out validation may still differ from true prospective antibody-antigen generalization.

#### Running real-label antigen refinement

The repository includes `configs/refine_antigen_real_label.yaml`, which initializes from the paired OAS refinement checkpoint:

```bash
python scripts/mlm_train.py --config configs/refine_antigen_real_label.yaml
```

For a quick implementation check:

```bash
python scripts/mlm_train.py --config configs/refine_antigen_real_label.yaml --smoke-test-only
```

The stage is designed to start from `checkpoints/mlm_3m_paired_refine_hcdr3_01/best.pt` rather than from the older shuffled-antigen checkpoint, so the compatibility head does not inherit the synthetic shortcut objective.

If YAML support is not installed, the equivalent explicit CLI command is:

```bash
python scripts/mlm_train.py \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --training-stage antigen_real_label_refine \
  --init-checkpoint checkpoints/mlm_3m_paired_refine_hcdr3_01/best.pt \
  --output-dir checkpoints/mlm_antigen_real_label_refine \
  --resume-from-last \
  --max-length 192 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --train-num-workers 0 \
  --eval-num-workers 0 \
  --bucket-width 8 \
  --mask-probability 0.10 \
  --hcdr3-span-probability 0.0 \
  --hcdr3-span-min 3 \
  --hcdr3-span-max 8 \
  --shuffle-pair-probability 0.0 \
  --shuffle-antigen-probability 0.0 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 6 \
  --d-ff 1024 \
  --dropout 0.1 \
  --learning-rate 0.00005 \
  --weight-decay 0.01 \
  --grad-clip-norm 1.0 \
  --pair-loss-weight 0.0 \
  --compatibility-loss-weight 1.0 \
  --epochs 12 \
  --seed 42 \
  --use-amp \
  --device cuda
```

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

### Phase 5 - SAE-Based Concept Discovery

Train an SAE on activations from the fused antibody-antigen model to identify sparse, reusable latent features.

#### Intended workflow

- collect intermediate activations from the trained model,
- fit a sparse dictionary / SAE,
- identify sparse latent features,
- annotate those features biologically where possible,
- and use them to analyze and eventually steer model behavior.

This stage is where representation learning and interpretability meet most directly in the project.

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

#### Current implementation status

The repository now includes a fixed-length antigen-conditioned HCDR3 infilling stage:

- `antigen_hcdr3_infill_refine` fine-tunes the dual-stream antibody/antigen model on positive binder rows only.
- It keeps the antibody framework, optional light chain, and antigen visible.
- It masks the entire known heavy-chain CDR3 span and trains the MLM head to reconstruct those residues.
- Compatibility loss is set to `0.0` in this stage because the goal is residue infilling for positive binders, not binder-vs-non-binder classification.
- HCDR3-specific metrics include token accuracy, full-span exact match, target-token count, and valid-span count.

This is fixed-length infilling because the number of `[MASK]` tokens equals the HCDR3 length. That makes the first design task intentionally narrower: the model learns what residues belong in a known-size HCDR3 hole. Unknown-length generation is handled as a separate proposal step: choose one or more candidate lengths first, then run the same fixed-length infiller for each proposed length.

The initial unknown-length infrastructure lives in `src/smallAntibodyGen/infill/hcdr3.py`:

- `FixedLengthHCDR3Infiller` builds masked antibody/antigen inputs and samples HCDR3 residues.
- `LengthProposalStrategy` defines the interface for future length predictors.
- `EmpiricalHCDR3LengthPrior` is the first usable length proposer, sampling lengths from positive-binder HCDR3s.
- `AntigenCompatibilityScorer` can rank generated candidates with the real-label compatibility head.

#### Running HCDR3 infill refinement

The checked-in YAML config is:

```bash
python scripts/mlm_train.py --config configs/refine_antigen_hcdr3_infill.yaml
```

Equivalent explicit CLI command:

```bash
python scripts/mlm_train.py \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --training-stage antigen_hcdr3_infill_refine \
  --init-checkpoint checkpoints/mlm_antigen_real_label_refine/best.pt \
  --output-dir checkpoints/mlm_antigen_hcdr3_infill_refine \
  --resume-from-last \
  --max-length 192 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --train-num-workers 0 \
  --eval-num-workers 0 \
  --bucket-width 8 \
  --mask-probability 0.10 \
  --hcdr3-span-probability 1.0 \
  --hcdr3-span-min 3 \
  --hcdr3-span-max 8 \
  --hcdr3-mask-mode full_span \
  --mask-replacement-strategy always_mask \
  --shuffle-pair-probability 0.0 \
  --shuffle-antigen-probability 0.0 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 6 \
  --d-ff 1024 \
  --dropout 0.1 \
  --learning-rate 0.00003 \
  --weight-decay 0.01 \
  --grad-clip-norm 1.0 \
  --pair-loss-weight 0.0 \
  --compatibility-loss-weight 0.0 \
  --epochs 8 \
  --seed 42 \
  --use-amp \
  --device cuda
```

For a CPU smoke test without YAML:

```bash
python scripts/mlm_train.py \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --training-stage antigen_hcdr3_infill_refine \
  --init-checkpoint checkpoints/mlm_antigen_real_label_refine/best.pt \
  --output-dir checkpoints/.tmp_hcdr3_infill_smoke \
  --no-resume-from-last \
  --smoke-test-only \
  --max-length 192 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --hcdr3-mask-mode full_span \
  --mask-replacement-strategy always_mask \
  --compatibility-loss-weight 0.0 \
  --device cpu \
  --no-progress
```

#### Generating candidates

Fixed-length candidate generation uses the known HCDR3 length from each target record:

```bash
python scripts/hcdr3_infill.py \
  --checkpoint checkpoints/mlm_antigen_hcdr3_infill_refine/best.pt \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --split val \
  --num-records 5 \
  --num-samples 16 \
  --length-mode fixed \
  --temperature 1.0 \
  --top-k 10 \
  --device cuda \
  --output-path outputs/hcdr3_fixed_candidates.jsonl
```

Empirical unknown-length candidate generation samples proposed HCDR3 lengths from the positive-binder training distribution, then infills each proposed length:

```bash
python scripts/hcdr3_infill.py \
  --checkpoint checkpoints/mlm_antigen_hcdr3_infill_refine/best.pt \
  --data-path data/processed/antibody_antigen/antibody_antigen.jsonl.gz \
  --split val \
  --num-records 5 \
  --num-samples 16 \
  --length-mode empirical \
  --temperature 1.0 \
  --top-k 10 \
  --score-checkpoint checkpoints/mlm_antigen_real_label_refine/best.pt \
  --device cuda \
  --output-path outputs/hcdr3_empirical_length_candidates.jsonl
```

Each JSONL row includes the record identity, target metadata, true HCDR3 if known, proposed length, generated HCDR3, generated heavy-chain sequence, MLM log probability, and optional compatibility score.

---

## Data Pipeline

The repository now has a clearer staged preprocessing story:

- `scripts/prepare_oas.py`
  Cleans raw OAS data into processed antibody-only or paired heavy/light JSONL files.

- `scripts/prepare_antibody_antigen.py`
  Cleans ASD parquet shards into processed antibody-antigen JSONL files, keeps heavy/light plus antigen context, preserves nested numbering metadata, computes HCDR3 spans when possible, and assigns leakage-aware splits.

- `scripts/mlm_train.py`
  Trains antibody MLM, paired-refinement, synthetic antigen refinement, real-label antibody-antigen compatibility refinement, and fixed-length antigen-conditioned HCDR3 infill refinement stages.

- `scripts/hcdr3_infill.py`
  Generates fixed-length or empirical-length HCDR3 candidates from a trained antigen-conditioned infill checkpoint, with optional compatibility scoring.

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
        |                      |
        +----------+-----------+
                   |
                   v
      HCDR3 infilling and constrained lead optimization
```

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

---

## Current Repository Status

This repository is no longer only a high-level roadmap. It now includes working implementations for:

- antibody-only OAS preprocessing,
- paired VH/VL OAS preprocessing,
- antibody MLM training,
- paired VH/VL refinement,
- ASD-based antibody-antigen parquet preprocessing,
- synthetic shuffled-antigen compatibility refinement,
- real-label antibody-antigen compatibility refinement from ASD binary binder labels,
- and fixed-length antigen-conditioned HCDR3 infill refinement on positive binder rows.

The antigen-conditioned model now exists as a dual-stream antibody/antigen cross-attention model. The main open research question is no longer whether the stage is wired, but how well different compatibility objectives avoid shortcut learning and generalize across targets, assay families, and antibody families.

For the original `antigen_refine` stage, treat compatibility metrics as diagnostics for the synthetic shuffled-antigen task. For `antigen_real_label_refine`, treat metrics as real binary-label diagnostics, but still audit them against source, target, assay, and split effects before interpreting them biologically.

For `antigen_hcdr3_infill_refine`, treat HCDR3 reconstruction metrics as evidence about antigen-conditioned residue infilling given a known span length. They should not be read as proof of fully unconstrained binder design, because fixed-length infilling supplies the HCDR3 length through the number of mask tokens.

---

## Long-Term Vision

The broader aim is to develop a system that can move beyond black-box scoring and toward interpretable antibody design:

- understand what the model has learned,
- map internal features to biological concepts,
- and use those concepts to steer generation in a controlled way.

That would make the model useful not only for prediction, but also for hypothesis generation, lead refinement, and mechanistically informed protein engineering.

---

## Notes

This project is research-oriented and iterative by design. Architectural choices, supervision targets, split policies, and representation formats may evolve as the data and experiments mature.
