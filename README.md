
## Introducing Steerability to Antibody Generation via SAE-derived concept labelling

With active discussion in the intersection of machine learning and biology in order to discern the true ability of ML to derive biophysically-relevant ideas (vs. just memorizing surface-level ideas), this project aims to explore and introduce steerability into the antigen-conditional generation of an antibody (staring with the HCDR3 region). 

The core idea is to first learn strong antibody sequence representations, fusing those representations with antigen context, and then deriving concepts from a SAE to make a more interpretable and steerable generation.

Rather than jumping directly to a large autoregressive generator, this project starts with a masked language modeling (MLM) objective on antibody sequences from OAS (Observed Antibody Space). That choice is motivated by both dataset scale and the immediate modeling goal: learning antibody sequence regularities well enough to support focused edits—especially in HCDR3, which plays a major role in binding, specificity, and structural compatibility.

---

## Project goals

The long-term goal is not just to generate antibody sequences, but to build a model that can:

- learn antibody sequence biology and chain-specific grammar,
- incorporate antigen and assay context,
- predict useful downstream properties,
- expose biologically meaningful latent concepts, and
- support **controlled optimization** of promising binders.

## High-level roadmap

### Step 1 — Antibody-only pretraining on OAS

Pretrain a MLM in PyTorch, using antibody sequence data from the Observed Antibody Space (OAS) to learn antibody-specific sequence structure.

#### Why start here?

- MLM  the ultimate goal of learning contextual sequence contraints while supporting the constrained goal of HCDR3 generation.
- This stage is meant to capture the statistical and biological regularities of antibody sequences before introducing antigen information.

#### Design choices

- Train on antibody sequences using a masked language modeling objective.
- Tag chain type explicitly with special tokens:
  - heavy vs light,
  - and potentially finer-grained tokens such as IGH / IGK / IGL later.
- Create config argument of "hcdr3-span-probability", to randomly encourage the masking tokens to focus on hcdr3, using its alloted tokens to mask  around that area, then use the rest of the mask quotas on other chain tokens
- Focus is on learning:
  - framework and CDR regularities,
  - chain-specific sequence grammar,
  - antibody syntax and plausible residue patterns.

---

### Phase 2 — Pair VH/VL and introduce antigen context

After antibody-only pretraining, move to a smaller, paired subset of OAS to model the interaction between chains and, eventually, between antibody and antigen.

#### Planned architecture

- Concatenate VH / VL representations.
- Retrain or adapt the antibody encoder on paired data.
- Add an antigen encoder using antigen sequence inputs.
- Fuse antibody and antigen information using cross-attention.
- Produce a pooled joint embedding that represents the interaction context.

This is the stage where the model begins to move from “learning antibody sequence biology” to “learning binding-relevant structure in context.”

---

### Phase 3 — Build a richer binding representation

The joint model is intended to represent binding as a function of multiple interacting factors, not sequence alone.

#### Planned input components

- Antibody representation
- Antigen / epitope representation
- Assay context

Depending on available data, the antigen side may later expand beyond sequence to include:

- epitope annotations,
- structural features,
- or other experimentally relevant metadata.

---

### Phase 4 — Add multitask supervised heads

Once a fused antibody–antigen representation exists, attach smaller supervised heads to predict downstream properties that matter for screening and optimization.

#### Initial target tasks

- Binder vs non-binder classification
- pKd / ΔG regression
- Mutation propensity / mutation effect prediction

The goal here is not just performance, but to encourage the model to be more focused and shared representation to organize around biophysically meaningful factors.

---

### Phase 5 — Train an SAE for interpretable concept discovery

In order to support the major goal of the project (steerability, not just prediction), the plan is to train a SAE on activations from the fused antibody-antigen model.

#### Intended workflow

- collect intermediate activations from the trained model,
- fit a sparse dictionary / SAE,
- identify sparse latent features,
- annotate those features with biological interpretations where possible,
- use the resulting concept layer to understand and eventually steer model behavior.

This workflow is my preliminary methodology to bridge representation learning and mechanistic interpretability.

---

### Phase 6 — Constrained generation and lead optimization

The eventual generation setting is intentionally narrow at first.

Rather than unconstrained de novo sequence generation, the early focus is:

- start from an existing binder,
- make constrained edits, particularly doing HCDR3 span-infilling,
- preserve broader antibody plausibility,
- and gradually expand the scope of controllable optimization.

This makes the generation problem more realistic and more actionable for lead optimization.

---

## Model sketch

```text
OAS antibody sequences
        |
        v
Antibody MLM pretraining
        |
        v
Paired VH/VL encoder adaptation
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
      Interpretable, constrained lead optimization
```

---

## Why this approach?

### Why MLM first?

Starting with an MLM provides a sensible first objective because it:

- is more data-efficient than immediately scaling up a generative system,
- teaches contextual residue relationships,
- supports local infilling behavior,
- and matches the short-term goal of **targeted HCDR3 editing**.

### Why chain tokens?

Heavy and light chains do not obey identical sequence statistics. Explicit chain tokens let the model condition on chain identity instead of forcing it to infer that structure implicitly every time.

### Why fuse antibody and antigen representations?

Binding is relational. A strong antibody representation alone is not enough; affinity and specificity depend on the context of the target antigen, and often on assay conditions as well.

### Why SAEs?

SAEs offer a path toward sparse, more interpretable internal features. The hope is that these features can be:

- inspected,
- labeled,
- linked to meaningful biology,
- and used to steer sequence optimization more deliberately.

---

## Near-term milestones

- [ ] Build and validate the OAS preprocessing pipeline
- [ ] Train a baseline antibody MLM in PyTorch
- [ ] Add chain-specific special tokens (heavy/light, later IGH/IGK/IGL)
- [ ] Create reproducible train/val/test splits
- [ ] Evaluate masked reconstruction with emphasis on HCDR3 behavior
- [ ] Fine-tune or retrain on paired VH/VL data
- [ ] Add antigen encoder + cross-attention fusion
- [ ] Attach multitask binding / biophysical heads
- [ ] Train an SAE on fused activations
- [ ] Test constrained HCDR3 optimization around known binders

---

## Repository status

This repository is currently an active research roadmap and implementation project. The README describes the intended modeling direction and staged build-out of the system.

The near-term emphasis is on:

1. strong antibody-only pretraining,
2. clean representation learning for paired chains,
3. principled antibody–antigen fusion,
4. interpretable latent structure,
5. and constrained, biologically grounded sequence optimization.

---

## Long-term vision

The broader aim is to develop a system that can move beyond black-box scoring and toward interpretable antibody design:

- understand what the model has learned,
- map internal features to biological concepts,
- and use those concepts to steer generation in a controlled way.

That would make the model useful not only for prediction, but for hypothesis generation, lead refinement, and mechanistically informed protein engineering.

---

## Notes

This project is research-oriented and iterative by design. Architectural choices, supervision targets, and representation formats may evolve as the data and experiments mature.
