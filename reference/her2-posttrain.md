# HER2 p-IgGen post-training: run in progress

Snapshot: 2026-09-18, after the first initial SFT seed. Implementation commit
`08d57ae`; model fitting and the remaining evaluation are still running.

p-IgGen is the generator. The CNN is an auxiliary classifier added by Codex to
compare ranking of measured sequences. It does not generate sequences, supply
rewards or preference pairs, or choose checkpoints. DPO preferences use measured
high-versus-low bins at matched wild-type mutation distance.

The [fixed protocol](../specs/her2_hcdr3_benchmark.md) compares continued SFT and
DPO at 3/6/10 additional measured GPU minutes per seed, with DPO also continuing
to 20/30 minutes. There are three seeds. Reference scoring is charged to DPO;
the common initial SFT cost is reported separately.

## Initial SFT: first seed only

Seed20260918, batch128, all120504training high-bin rows per pass. Five passes
took735.6wall seconds. Native cached/full scoring and all52parameter-gradient
checks passed after training; the spill monitor reported no suspected spill.

| Pass | Validation high-bin NLL/residue | Validation AP | Validation AUROC |
|---|---:|---:|---:|
| 1 | 1.565302 | 0.942168 | 0.961068 |
| 3 | 1.491964 | 0.965874 | 0.978198 |
| 5 | 1.502257 | 0.969287 | 0.980280 |

The declared initial checkpoint rule uses validation high-bin NLL, so pass3 is
best for this seed. NLL worsened after pass3 while training loss continued to
fall, which is consistent with the onset of overfitting. The slightly better AP
at pass5 does not change the selection rule. Final parent selection is frozen
after all initial runs finish.

These are provisional validation observations, not a final benchmark result.
The validation nearest-neighbor baseline already reaches approximately0.981AP,
and90.2%of validation cores are one mutation from a training core. Independent
assay evaluation and generated-sequence diversity have not yet been measured for
the fitted policies. No measured affinity is assigned to unassayed new sequences.

## Verification and artifacts

Before fitting:1984repository tests passed,3skipped; a native22M-parameter GPU
scoring/gradient check passed; and a tiny synthetic run completed both training
methods, checkpointing, generation, validation and the final evaluation handoff.

Full local artifacts are in `outputs/her2_posttrain_20260918/`. The initial
integrity audit read aggregate counts and two example rows from every published
split, including test; model-based final test and SPR evaluation remain pending.
p-IgGen's HER2 pretraining exposure remains unresolved.
