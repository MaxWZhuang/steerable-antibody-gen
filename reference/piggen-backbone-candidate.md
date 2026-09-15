# p-IgGen: previously evaluated backbone candidate

**Status, 2026-09-14:** retained alternative. The user selected ESM-IF1 for the first experiment. This earlier candidate audit does not establish ESM-IF1 readiness or exposure. See the [current recommendation](fixed-target-posttraining-recommendation.md).

## Backbone: a concrete candidate, with a promotion gate

**Candidate:** `ollieturnbull/p-IgGen`  
**Observed revision:** `0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023`  
**Release:** [checkpoint commit](https://huggingface.co/ollieturnbull/p-IgGen/commit/0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023), [model card](https://huggingface.co/ollieturnbull/p-IgGen), [official code](https://github.com/OliverT1/p-IgGen).

Use the base release. Its optional developability adaptation adds another prior that is unnecessary for the first attribution experiment.

The released model has **22,097,408 parameters**, four layers and eight heads. The paper describes a 17.35M, three-layer, twelve-head model; replacing four layers with three in the released architecture does not recover the paper's stated count. This proposal uses the released artifact and does not claim exact reproduction of the paper architecture. Fingerprint architecture, weights and activation sites before reusing interpretation artifacts; the discrepancy alone neither establishes compatibility nor proves incompatibility. The model card states BSD-3-Clause licensing.

The [official implementation](https://raw.githubusercontent.com/OliverT1/p-IgGen/main/src/piggen/model.py) uses `1 + VH + VL + 2` and supports VH generation from the prompt `2 + reverse(VL)`. It detects MPS. Its convenience likelihood routine uses disabled gradients and mean cross-entropy, so a training scorer must operate on native logits instead.

Promotion requires loading weights and tokenizer from the same revision, hashing artifacts, checking residue mapping and upstream logits, verifying gradients, and measuring forward/backward memory and throughput on the actual device. Weight-loading, gradient and timing checks remain unperformed. MPS availability or small parameter count alone does not prove feasible training throughput. The [architecture decision addendum](../specs/decisions/0003-pretrained-conditioned-policy.md#scope-clarification--2026-09-14) explicitly admits this candidate while keeping it unpromoted.

### Required token and exposure checks

The pinned metadata has an actual generation defect: **configured EOS 2 maps to arginine `R`; configured BOS 0 maps to `<PAD>`**. The real sentinels are literal `1` (ID 23) and `2` (ID 24); padding is ID 0. The [official wrapper](https://github.com/OliverT1/p-IgGen/blob/a4030ba7da81729c64b3198d331811ce1c91911e/src/piggen/model.py) overrides stop tokens according to generation direction. Generic generation can stop at a generated arginine. Complete teacher-forced logits are not intrinsically corrupted by the generation config.

Before promotion, test internal-arginine continuation, forward/reverse stop behavior, padded-batch versus single-example score parity, and exclusion of forced/special tokens from the editable-decision loss. Set semantic tokens explicitly rather than trusting config defaults. A metadata audit is not a passed generation test.

The [published sequence release](https://zenodo.org/records/13880874) contains three paired splits. We verified their checksums and scanned all three after alignment-character removal:

| Paired file | Parsed pairs |
|---|---:|
| Train | 1,632,916 |
| Validation | 43,661 |
| Test | 69,858 |
| **Total released** | **1,746,435** |

No exact mature CR9114 VH, fixed VL, mature pair, or either library HCDR3 allele occurred in these searches, including reverse-orientation substring checks. Exact archive hashes, queries, independent row counts, and results are in the [audit](evidence/piggen-release-audit-2026-09-14.json). The [paper's Section 2.3](https://academic.oup.com/bioinformatics/article/40/11/btae659/7888884) reports **1,800,545 pairs in the filtered dataset**, not explicitly in the training CSV alone. Validation and test account for 113,519 of the apparent 167,629 train-file gap; **54,110 pairs (3.01%) remain unreconciled**. The supplement describes deduplication and splitting but does not establish the cause of this residual difference. Do not label it deduplication without evidence.

This only clears those searches in the released paired files. The 3.276 GB unpaired-heavy and 3.114 GB unpaired-light training archives and near-neighbor exposure remain unchecked. The unresolved count difference and archive-to-shipped-checkpoint lineage prevent a complete pretraining-exposure claim. Exact sequence exposure, family-level exposure, and experimental-label leakage are separate questions; an exact-match search cannot settle all three. Use the precise term **held out from post-training**, with pretraining exposure reported separately. Related germline sequences in repertoire pretraining are expected and do not by themselves imply leakage of affinity labels.

Why this candidate: native autoregression permits efficient differentiable probability scoring; paired-chain generation is already supported; the model is small; antibody interpretability precedent exists. Its selection is based on practical fit, not demonstrated superiority at affinity prediction.

Use an **AbLang2 frozen-embedding predictor** as a strong biological representation comparator. IgLM remains an alternative for contiguous HCDR3 infilling, but is unpaired and its advantage is smaller for sparse VH editing. ESM2 is a useful established comparator; neither VESM's variant-effect results nor DPLM's generation results establish superiority for this task. A diffusion conversion would add another research uncertainty.

