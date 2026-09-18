# Research scope and implementation status

[Project overview](../README.md)

**Updated, 2026-09-18:** the active research studies the causal computations of
the standalone p-IgGen policy before and after HER2 post-training. The training
and evaluation campaign is [complete](her2-posttrain.md). This document replaces
the older implementation roadmap, which described completed work as pending and
made antigen conditioning a prerequisite for interpretation.

The detailed local build queue is
`docs/BUILD-PLAN-her2-mechanistic-interpretability.md`, kept under the existing
local-only documentation policy. It specifies work packages, artifacts, and tests.

## What is already built

| Capability | Implementation and evidence |
|---|---|
| Native p-IgGen policy | `experiments/her2_policy.py`: pinned GPTNeoX load, native vocabulary, canonical-residue normalization, full/cached scoring, sampling, checkpoint loading |
| HER2 data and comparators | `experiments/her2_data.py`, `her2_baselines.py`: provenance, scaffold coordinates, labels, proximity analysis, additive model and CNN |
| Supervised training | `scripts/train_her2.py`: pretrained and scratch-initialized arms, validation selection, parity checks |
| Preference and continued supervised training | `scripts/posttrain_her2.py`: frozen reference scoring, DPO, continued SFT, measured GPU budgets, generation diagnostics |
| Evaluation | `scripts/evaluate_her2.py`, `evaluate_her2_math.py`, `experiments/her2_eval.py`: ranking, uncertainty, native probability checks, diversity, independent assay reporting |
| Completed HER2 evidence | [Campaign report](her2-posttrain.md) and [final integrity record](evidence/her2-final-integrity-2026-09-18.json) |
| Earlier ESM-IF1 integration | [Policy specification](../specs/esmif1_policy.md), [prepared context](cr9114-5cjq-context.md), [SFT pilot](cr9114-5cjq-pilot.md), [DPO pilot](cr9114-dpo-pilot.md) |
| Shared infrastructure | Provenance, run fingerprints/resume compatibility, split audits, target identity, frozen evaluation inputs |
| Custom-model research | MLM/fusion/guide interfaces and a synthetic antigen-pathway probe; distinct from the HER2 policy |

Python package paths in the table are relative to `src/smallAntibodyGen/`.
Recorded training results do not imply that checkpoint bytes are present in every
checkout. Analysis should inventory actual artifact locations and reuse them.

## Mechanistic work remaining

Build replayable matched-prefix cases and a native GPTNeoX observation/patching
adapter. Its graph must reflect causal masking, token offsets, rotary positions,
normalization, head output projections, and the loaded residual-block ordering.
The current HER2 policy has four layers and eight heads per layer, and consumes
only a fixed VH prefix plus previously chosen core residues. It sees no antigen,
light chain, or FR4 context.

Measure complete local single/double-mutant effects across backgrounds, with
directed conditional residue contrasts. An additive classifier's fixed class-logit
contrast supplies a zero-interaction control; transformed probabilities need not
be additive. Model-score interactions and experimentally measured epistasis are
reported separately.

Test repeat completion, position preferences, amino-acid bias, and biochemical
grouping as candidate explanations. Establish both attention routing and the
information transmitted through values/output projections. Synthetic repeats
diagnose capacity; original-model interventions on native antibody cases establish
whether those components mediate the behavior under study.

Include head pairs, MLPs, functional groups, conditional ablations, and restoration
from the start. Recompute descendants when measuring compensation. Report
fidelity against component count and collateral effects across contexts rather
than assuming a small, unique circuit exists.

Compare per-layer and cross-layer transcoders as sparse MLP replacements. Separate
original-to-replacement error, pruning error, and agreement between predicted and
actual interventions in the original model. Record frozen-attention and recursive
replacement modes separately. Validate proposed intermediate steps, since accurate
replacement outputs alone do not establish faithful computation.

Apply the same cases and interventions across base, SFT, continued SFT, DPO, and
scratch-trained checkpoints. Include all available budgets and seeds in the
inventory, preserving historical selection outcomes as metadata. Structure and
measured mutation effects support biological comparison where their mapping and
coverage are known; they do not substitute for model interventions.

## How results govern claims

Predictive superiority, a DPO gain, preserved diversity, successful antigen
conditioning, and a particular sparse-recovery percentage are not entry conditions
for this research. Additive behavior, copying, collapse, null effects, and broad
causal dependence are legitimate subjects. State what an intervention supports
without requiring the optimizer or explanatory method to succeed.

Keep checkpoint/probability integrity, explicit donor choices, appropriate
controls, and held-out analysis cases. A failure of those checks limits the
affected result; it does not suspend unrelated implementation or analysis. The
completed benchmark's test results have already been inspected. A new analysis
partition is not a newly blinded biological evaluation.

The [HER2 protocol](../specs/her2_hcdr3_benchmark.md) retains the original run's
selection rules and measured-GPU-time budgets. Those rules do not exclude failed
or unselected checkpoints from interpretation. New exposure-matched training,
additional measurements, or prospective antibody validation are separate studies.

## Optional extensions and retained documentation

The [antigen-conditioned policy specification](../specs/pretrained_conditioned_policy.md)
describes a separate extension involving fusion, external guidance, and a possible
diffusion objective. Those components remain available research directions, with
their own data requirements; they are not the active HER2 build sequence.

The [custom-model workflow](custom-model-workflow.md) documents working commands.
Earlier experiment specifications and reports remain implementation contracts and
historical evidence. Completed build instructions have been retired from the local
planning queue; numerical checks and historical outcomes remain intact.
