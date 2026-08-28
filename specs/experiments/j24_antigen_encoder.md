# J24 — antigen-encoder selection

**Status:** scoped, **not run**. Execution is blocked; see §7.
**Predeclared:** 2026-08-28, before any arm has been trained.

## 1. Question

> Given the same pretrained antibody model, data, supervision, initialization, and
> 1024-token antigen crop, does frozen ESM-2 produce stronger antigen-dependent policy
> behavior than the scratch antigen encoder?

## 2. What J24 is and is not

J24 chooses the antigen **sensor**. It is **not** the target-specific claim.

The later, separate experiment tests the real claim: whether the chosen sensor helps generate
better held-out HCDR3 choices for a specific target and controlled contrast antigens. Reading
a J24 result as evidence for that claim is the main way this experiment can be misused, so the
report carries the distinction in its `scope` field and the comparison command prints it.

**What the result licenses, precisely.** The scratch antigen encoder trains; the ESM backbone
is frozen. The arms therefore differ in **training regime** as well as in representation, so a
win says *this encoder package works better under the intended training regime*. It does **not**
isolate the effect of pretraining — that would require a trainable-ESM or frozen-scratch arm,
which J24 deliberately does not run. The report carries this as `claim_limit`, and both
`total_parameters` and `trainable_parameters` are required per arm so the asymmetry is visible
in the numbers rather than only in the prose.

## 3. Design constraints

Each is a property some mechanism enforces, not an intention.

| Constraint | Enforced by |
|---|---|
| Both arms start from the same promoted stage-2 checkpoint | `validate_paired_configs` (identical `init_checkpoint`, non-empty); the warm-start gate rejects a dual-stream parent |
| Same antigen residues despite different special-token counts | `experiments/antigen_residues.py`; both arms crop to the shared residue budget |
| Projection, cross-attention, fusion, heads freshly initialized and **bit-identical between arms at step zero** | `experiments/init_parity.reinitialize_shared_modules` |
| No stage-3 scratch fusion or head weights enter the ESM arm | stage-2 rooting, plus the warm-start gate |
| Data rows, ordering, masks, replay, optimizer steps, supervision identical | paired configs differing in exactly `antigen_encoder` and `output_dir` |
| Scratch antigen weights train; ESM stays frozen | `finetune: frozen`, rejected otherwise |
| At least three seeds | `decide()` refuses fewer |
| Cache only frozen ESM outputs, cached/uncached parity proven | `experiments/antigen_cache.py` + `test_cached_and_uncached_encodings_agree` |

### 3.1 The residue asymmetry, measured

The two tokenizers spend different budgets on special tokens:

```
scratch:  [CLS] [OTHER_CHAIN] ...residues... [EOS]     3 special tokens
ESM-2:    <cls>               ...residues... <eos>     2 special tokens
```

At `antigen_max_length: 1024` that is **1021 residues for scratch against 1022 for ESM** — one
extra residue, in the same arm, on every antigen. Small, systematic, invisible in a loss curve,
and enough to make the comparison two-axis. Both arms therefore crop to **1021 residues** before
encoding. The ESM arm's sacrifice (1 residue) is reported rather than hidden.

### 3.2 Why initialization needs its own mechanism

Seeding both runs identically does **not** give them identical fusion weights. Module
construction and every `normal_` draw from the global RNG in order, and the two arms build
different antigen encoders, so every parameter created afterwards — all the shared ones — lands
on different values. A win could then be a lucky fusion draw. `reinitialize_shared_modules`
re-derives each shared module from a seed hashed per module **name**, so the values depend on
neither construction order nor which encoder was built.

## 4. Promotion rule — lexicographic, never blended

Applied in this order. A blended score would let a large throughput win pay for an absent
policy response, which is precisely the trade J24 must not make.

1. **Gate 1 — antigen dependence.** Positive correct-antigen versus matched-swap HCDR3 policy
   response, **beyond the null/permutation noise band**. "Beyond the band", not "greater than
   zero": the permutation control is what measures zero here.
2. **Gate 2 — no regression.** Compatibility calibration and held-out AUPRC must not regress
   beyond the predeclared margins.
3. **Rank among passing arms**, in order: AVIDa inner-development mutant response → seed
   variance → throughput → cache cost.
4. **Veto.** A classifier-only improvement cannot win when token-policy response is absent. J24
   selects a sensor for *generation*.
5. **Ties.** Neither passing → **promote neither**. Indistinguishable within measured noise →
   **keep scratch**, the simpler dependency.
6. **Untouched.** Final target-specific and AVIDa evaluation splits are not read while
   selecting the encoder.

Margins are **inputs**, not defaults. `compare_antigen_encoder_arms.py` refuses to score
results without them, because a threshold chosen after seeing results is not a threshold.

## 5. Budget

Frozen after a **timing-only calibration** of both arms — wall-clock and throughput measured,
no evaluation metric read. Memory is not a constraint: at 288/1024 batch 16 the frozen ESM arm
measures **816 MiB** against the scratch arm's **1842 MiB** of 4095.7, because a frozen backbone
stores no gradients, optimizer state, or backward activations.

## 6. Report schema

`schema_version: j24-antigen-encoder-comparison/1`, emitted by
`scripts/compare_antigen_encoder_arms.py`. Per arm:

| Field | Meaning |
|---|---|
| `policy_response_correct_vs_swap` | HCDR3 policy response, correct antigen vs matched swap |
| `policy_response_noise_band` | Null/permutation band for that response |
| `compatibility_auprc` | Held-out AUPRC (primary; binders are rare) |
| `compatibility_calibration_error` | Calibration error |
| `avida_inner_dev_mutant_response` | AVIDa **inner-development** mutant response |
| `seed_variance` | Across ≥3 seeds |
| `throughput_sequences_per_second` | From the timing-only calibration |
| `cache_build_seconds` | Frozen-antigen cache build cost |
| `seeds` | Seed count; <3 is refused |
| `total_parameters` | All parameters, trainable or not |
| `trainable_parameters` | Parameters receiving gradients; the arms differ here by design |

The report also records the margins it was judged under, the two config paths, and the keys
that differed between them — so a later reader can check the comparison was one-axis without
re-deriving it.

## 7. What blocks execution

1. **A promoted v5 stage-2 checkpoint.** `checkpoints/mlm_paired_refine_v5/best.pt` does not
   exist; the v5 chain has not been trained.
2. **Approved inner-development evaluation assets** — the AVIDa inner development-mutant split
   is J05c, `not_started`. Gate 1's noise band and the AVIDa ranking criterion both depend on
   it.

The configs, mechanisms, tests, comparison command, and this document exist now so that when
those land, the experiment is a run rather than a design exercise — and so the design was fixed
before any number was visible.

## 8. Artifacts

| Artifact | Path |
|---|---|
| Paired pilot configs | `configs/experiments/antigen_encoder/arm_{scratch,esm}.yaml` |
| Residue parity | `src/smallAntibodyGen/experiments/antigen_residues.py` |
| Initialization parity | `src/smallAntibodyGen/experiments/init_parity.py` |
| Frozen-antigen cache | `src/smallAntibodyGen/experiments/antigen_cache.py` |
| Comparison + promotion rule | `scripts/compare_antigen_encoder_arms.py` |
| Contract tests | `src/smallAntibodyGen/tests/test_j24_antigen_encoder.py` (25), `test_j24_comparison.py` (18) |
