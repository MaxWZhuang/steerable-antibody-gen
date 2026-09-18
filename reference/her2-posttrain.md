# HER2 p-IgGen post-training: run in progress

Snapshot: 2026-09-18, after all three initial pretrained SFT seeds. Implementation commit
`08d57ae`; model fitting and the remaining evaluation are still running.

p-IgGen is the generator. The CNN is an auxiliary classifier added by Codex to
compare ranking of measured sequences. It does not generate sequences, supply
rewards or preference pairs, or choose checkpoints. DPO preferences use measured
high-versus-low bins at matched wild-type mutation distance.

The [fixed protocol](../specs/her2_hcdr3_benchmark.md) compares continued SFT and
DPO at 3/6/10 additional measured GPU minutes per seed, with DPO also continuing
to 20/30 minutes. There are three seeds. Reference scoring is charged to DPO;
the common initial SFT cost is reported separately.

## Initial pretrained SFT: all three seeds

Batch 128, all 120,504 training high-bin rows per pass, five passes per seed.
Each run took about 12.2 minutes including validation and checkpoint checks.
Native cached/full scoring and all 52 parameter-gradient checks passed after
training in each run; no run triggered the suspected memory-spill flag.

| Seed | Pass | Validation high-bin NLL/residue | Validation AP | Validation AUROC |
|---|---|---:|---:|---:|
| 20260918 | 1 | 1.565302 | 0.942168 | 0.961068 |
| 20260918 | 3 | 1.491964 | 0.965874 | 0.978198 |
| 20260918 | 5 | 1.502257 | 0.969287 | 0.980280 |
| 20260919 | 1 | 1.564050 | 0.938045 | 0.959678 |
| 20260919 | 3 | 1.488833 | 0.967717 | 0.979037 |
| 20260919 | 5 | 1.501824 | 0.969823 | 0.980580 |
| 20260920 | 1 | 1.558214 | 0.940381 | 0.961579 |
| 20260920 | 3 | 1.486741 | 0.966903 | 0.978529 |
| 20260920 | 5 | 1.500826 | 0.970103 | 0.980810 |

The declared initial checkpoint rule uses validation high-bin NLL, so pass 3 is
best for every pretrained seed. In all three seeds, NLL worsened after pass 3 while training loss continued to
fall, which is consistent with the onset of overfitting. The slightly better AP
at pass 5 does not change the selection rule. Final parent selection is frozen
after all initial runs finish.

The [initial SFT evidence](evidence/her2-initial-sft-2026-09-18.json) retains the
per-seed training reports and checkpoint hashes. Random-initialization controls,
continuations, generation and final evaluation remain pending. These are
provisional validation observations, not a final benchmark result.
The validation nearest-neighbor baseline already reaches approximately 0.981 AP,
and 90.2% of validation cores are one mutation from a training core. Independent
assay evaluation and generated-sequence diversity have not yet been measured for
the fitted policies. No measured affinity is assigned to unassayed new sequences.

## Verification and artifacts

Before fitting: 1,984 repository tests passed, 3 skipped; a native 22M-parameter GPU
scoring/gradient check passed; and a tiny synthetic run completed both training
methods, checkpointing, generation, validation and the final evaluation handoff.

Full local artifacts are in `outputs/her2_posttrain_20260918/`. The initial
integrity audit read aggregate counts and two example rows from every published
split, including test; model-based final test and SPR evaluation remain pending.
p-IgGen's HER2 pretraining exposure remains unresolved.
