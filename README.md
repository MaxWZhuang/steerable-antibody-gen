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

---

## Data Pipeline

The repository now has a clearer staged preprocessing story:

- `scripts/prepare_oas.py`
  Cleans raw OAS data into processed antibody-only or paired heavy/light JSONL files.

- `scripts/prepare_antibody_antigen.py`
  Cleans ASD parquet shards into processed antibody-antigen JSONL files, keeps heavy/light plus antigen context, preserves nested numbering metadata, computes HCDR3 spans when possible, and assigns leakage-aware splits.

- `scripts/mlm_train.py`
  Trains the current antibody MLM and paired-refinement stages from the processed OAS datasets.

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
- and ASD-based antibody-antigen parquet preprocessing.

The antigen-conditioned model itself is still the next major build step, but the preprocessing foundation for that stage now exists.

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
