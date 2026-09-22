# Steerable antibody generation

`smallAntibodyGen` is research code for antibody sequence modelling: a small
antibody masked language model, an optional antigen-conditioned dual-stream
extension, fixed-length HCDR3 infilling, and the post-training machinery
(supervised fine-tuning, preference optimization, parent replay) used to steer a
policy toward a fixed target.

This repository is the **implementation**. Experiment write-ups, results,
figures and evidence are kept outside it.

## Installation

Python 3.10+. The package uses a src layout; an editable install is the clean
setup, and the test suite adds `src/` to `sys.path` via `conftest.py` so it also
runs without one.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Optional extras, each imported lazily so the base install never needs them:

| Extra | Pulls in | Needed for |
|---|---|---|
| `dev` | pytest | the test suite |
| `esm` | transformers, peft | `antigen_encoder_type="esm"` (PLM antigen encoder) |
| `her2` | transformers, safetensors | the pinned p-IgGen backbone used by HER2 post-training |
| `tb` | tensorboard | optional run logging |
| `esm-if1` | fair-esm and friends | the ESM-IF1 inverse-folding integration |

`esm-if1` deliberately omits `torch-scatter`; `smallAntibodyGen.esmif1_compat`
substitutes the single function ESM calls from it, which avoids a native
extension build on Windows. Call `esmif1_compat.install()` before the first
`esm.inverse_folding` import.

## Layout

```
src/smallAntibodyGen/
  tokenizer.py          AminoAcidTokenizer (residues + special/chain tokens)
  data/                 datasets, samplers, collators (MLM and HCDR3-span masking)
  models/               the MLM and the dual-stream antibody/antigen model
  infill/               fixed-length HCDR3 infilling, length priors, scorers
  experiments/          post-training: SFT, DPO/IPO, replay, audits, campaigns
  evaluation/           metrics and evaluators
  structure/            hash-pinned PDB/mmCIF input for the ESM-IF1 policy
  benchmarks/           benchmark loading and provenance
  tests/                pytest suite
scripts/                CLI entry points (data prep, training, infill, audits)
configs/                YAML training configs and JSON experiment configs
specs/benchmarks/       benchmark definitions loaded at runtime
```

## Running the pipeline

All commands are run from the repository root.

```bash
# Data preparation
python scripts/prepare_oas.py                  # OAS -> antibody-only / paired JSONL
python scripts/prepare_antibody_antigen.py     # ASD parquet shards -> antibody-antigen JSONL

# Training; the stage is selected inside the YAML
python scripts/mlm_train.py --config configs/pretrain_oas_small.yaml
python scripts/mlm_train.py --config configs/refine_oas_paired.yaml
python scripts/mlm_train.py --config configs/refine_antigen_real_label.yaml
python scripts/mlm_train.py --config configs/refine_antigen_hcdr3_infill.yaml

# HCDR3 candidate generation
python scripts/hcdr3_infill.py --checkpoint <ckpt> --data-path <jsonl.gz> --split val

# Tests
python -m pytest src/smallAntibodyGen/tests -q
```

Post-training campaigns are staged CLIs driven by a JSON config under
`configs/experiments/`. Each runs its stages in order and refuses to skip one:

```bash
python scripts/posttrain_her2_replay.py prepare   --config configs/experiments/her2_parent_replay.json
python scripts/posttrain_her2_replay.py preflight --config configs/experiments/her2_parent_replay.json
python scripts/posttrain_her2_replay.py freeze    --config configs/experiments/her2_parent_replay.json
python scripts/posttrain_her2_replay.py banks     --config configs/experiments/her2_parent_replay.json
python scripts/posttrain_her2_replay.py fit       --config configs/experiments/her2_parent_replay.json
python scripts/posttrain_her2_replay.py report    --config configs/experiments/her2_parent_replay.json
python scripts/posttrain_her2_replay.py verify    --config configs/experiments/her2_parent_replay.json
```

`freeze` records a hash of the source closure, the config and the inputs;
later stages re-hash them and refuse to run if anything moved. Checkpoint
loading is strict — a parameter/checkpoint mismatch is an error, never a silent
coercion.

## Implemented components

| Area | Implementation |
|---|---|
| Antibody policy | Custom antibody MLM; paired VH/VL refinement |
| Antigen fusion | Cross-attention into antibody residue logits; optional frozen/LoRA ESM antigen encoder |
| Sampling and guidance | Single-pass and iterative fixed-length HCDR3 infill; optional external guide |
| Generative objective | MLM, partial-state masking, mask-rate schedules |
| Data and evaluation | OAS/ASD preparation, target-identity and leakage audits, HCDR3 contrast scoring |
| Post-training | Supervised fine-tuning, DPO/IPO, token-level parent replay, exposure-matched campaign runners with freeze/verify gates |
| Interpretability | Synthetic antigen-pathway probe for the custom model |
| ESM-IF1 compatibility | `esmif1_compat` makes the archived `fair-esm` stack importable on this repo's torch/numpy |
| ESM-IF1 editing policy | `models.esmif1_policy` scores and samples a fixed-geometry, two-alleles-per-site constrained edit space through the native decoder |
| ESM-IF1 structural input | `structure` turns a hash-pinned local PDB/mmCIF file plus an explicit residue correspondence into encoder inputs, failing closed on anything ambiguous |

Two things that are easy to conflate. A pretrained antibody backbone is used by
the HER2 post-training path, but it is **not** integrated into the
antigen-conditioned architecture, which has no pretrained backbone and no
masked-diffusion objective. A preference trainer exists and is used, but not for
the antigen-conditioned policy — the ESM option there replaces only the antigen
encoder.

## Conventions

- **Strong-binder gating** uses `is_strong_binder`, which covers boolean
  positives and KD / -log KD / fuzzy strong binders. Gating on `binder_label == 1`
  silently drops most strong binders.
- **Ranking infill candidates** across different proposed HCDR3 lengths uses
  `mean_log_probability`; the raw `log_probability` sum grows with length.
- **Fixed-length infill** means the number of `[MASK]` tokens equals the HCDR3
  length. Unknown-length design is a separate length-proposal step followed by
  fixed-length infill per proposed length.
- **Heavy-only / nanobody records** encode with the real `[IGH]` token at both
  training and generation time, so the masked-input distribution matches.
- **Splits are leakage-aware**: antibody-antigen splitting is target-aware
  (UniProt → PDB → normalized name → antigen-sequence hash), not row-wise random.
