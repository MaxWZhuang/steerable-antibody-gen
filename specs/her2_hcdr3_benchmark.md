# HER2 HCDR3 core benchmark and post-training protocol

**Protocol fixed before fitting, 2026-09-18.** Settings below were declared before
HER2 fitting, checkpoint selection, model-based test evaluation or assay-outcome
evaluation began. Counts come from the published release. The initial audit read
aggregate labels and printed two example rows from each split, including test
(§7.2). Execution status and results are recorded in the
[run report](../reference/her2-posttrain.md).

"Fixed in advance" means fixed here, in this repository, before any fit — it is not
a claim of external preregistration.

Data sources are hash-pinned in `specs/benchmarks/{buzz_her2_affinity,
absci_denovo_her2,piggen_backbone}.json`. The affinity library and SPR workbook are
oxpig/`Tz_her2_affinity_and_beyond` (BSD-3-Clause); the AbSci release used for the
support-mismatch diagnostic in §7.3 carries a license clause requiring that any
reference to or publication of those data be **attributed to Absci Corporation
(2023)**.

## 1. The task

One antibody lineage (trastuzumab), one target (HER2), a fixed heavy framework and
a fixed light chain, and **ten editable HCDR3 positions**. The heavy chain reads

```
...YYC | SR | <10-residue core> | Y | WGQGTLVTVSS
         ^96        ^98                 (FR4)
```

All twenty canonical residues occur at all ten positions in the training data. The
published labels are sorted binding **bins** — `high`, `mid`, `low` — not KD
values, and nothing in this pipeline invents one.

This is a local library benchmark, and the reason held-out ranking is
interpolation is **proximity to the training set**, not proximity to wild type.
The library is dense over ten sites and mostly far from trastuzumab: 76.5% of
training cores sit 7 or more mutations from the wild-type core and only 442 of
367,042 sit at 1 or 2. But 90.2% of validation rows are within Hamming **1** of a
training core and 98.4% within 2, so a nearest-training-neighbour label lookup
reaches 0.981 average precision on validation with no model at all. That number is
the bar; the proximity strata in the audit are the disclosure rather than a fix.

(Measured on validation only, before any fit:
`outputs/claude_codex_her2_migration_20260918/extra_probes_round2.json`.)

| Population | Rows | high | mid | low |
|---|---|---|---|---|
| train | 367,042 | 120,504 | 133,008 | 113,530 |
| val | 78,652 | 25,822 | 28,502 | 24,328 |
| test | 78,652 | 25,823 | 28,501 | 24,328 |

Splits are the published `random_split/0.7_0.15_0.15` and are disjoint by core;
the union is 524,346 sequences.

## 2. The policy

The pinned p-IgGen release (`GPTNeoXForCausalLM`, 4 layers, 8 heads, hidden 768,
22,097,408 parameters), loaded with its native class and `strict=True`. Float32,
full parameters, no adapters, no frozen blocks.

* **Context:** a fixed 99-token prefix — the start token plus `VH[:98]`, ending
  `...YYCSR`. No FR4, no light chain, no antigen encoder. The prefix offsets are
  asserted constants, not a substring search.
* **Distribution:** the 20-way renormalized categorical over canonical residues,
  identical in training loss, scoring and sampling. The shipped `eos_token_id: 2`
  is residue `R`, so generation never consults it and always runs exactly ten steps.
* **Shared prefix cache:** built inside every call, used once, never detached. A
  parity probe compares it against ordinary full teacher forcing on logits, loss
  and *all* parameter gradients, before training and again after the last update.
* **Ranking** uses mean log probability per residue; **entropy and KL** use the sum.
* **Two tolerances, kept apart.** Logit and gradient parity is gated at 2e-5
  (measured on the pinned weights: 6.20e-6 and 3.43e-6). Comparisons of *summed*
  log probabilities produced by two fp32 routes — sampler vs scorer, cached vs
  full, batch 16 vs 256 — are gated at atol 5e-5 / rtol 2e-6, because the measured
  native worst cases are 2.38e-5, 3.34e-5 and 2.07e-5 over 10,000 draws and no
  amount of float64 accumulation removes rounding that happened in fp32 kernels.
  One shared helper (`her2_policy.compare_sum_log_probabilities`) performs every
  such comparison, records the measured error, and rejects nonfinite values
  outright.

## 3. Stage A — initial SFT (`scripts/train_her2.py`)

Two arms, `sft` (pinned weights) and `scratch` (same architecture, random
initialization), at seeds 20260918/19/20. Five full passes over all 120,504
training high-bin rows, checkpoints after passes 1, 3 and 5. Batch 128 (declared
fallback 64), AdamW lr 1e-4, betas (0.9, 0.999), weight decay 0.01, gradient clip
1.0, 5% linear warmup then cosine to 0.1 of peak, `drop_last=False`.

The batch size is resolved **once** from the driver's free-VRAM reading and frozen
for the whole campaign; a later, larger free-VRAM reading is not a reason to change
it. Selection is by **validation positive NLL per residue** and nothing else. The
result is `base_selection.json`, stage `initial_sft_selection`, which starts
post-training and deliberately **cannot** unlock reserved labels.

The scratch arm is a budget-matched contrast whose *initialization* cannot have
seen this benchmark. It is not a tuned from-scratch baseline and it does not make
either arm's supervised exposure clean.

Baselines fitted in the same stage: class prior, negative distance to the
wild-type core, nearest-training-neighbour label mean (all ties at the minimum
distance ≤ 2, else the prior), an additive multinomial logistic model (603
parameters, float64 LBFGS, ridge 1e-4), and a three-class CNN at three seeds.

**The CNN is an auxiliary comparison.** It is not the generator, not a reward
model, not a preference source, and not a selection criterion. p-IgGen is the
generator. The CNN predicts assay bins and exists to answer whether a plain
supervised classifier ranks held-out variants — it is the Buzz authors' own
baseline architecture, included because the published benchmark ships it, not
because a second backbone was wanted.

Scope clarification (round 3): the CNN and the linear model score only **measured**
populations — the reserved test split and the assay cohort. Scoring *generated*
draws with them is optional and is deliberately **not** produced: a proxy
probability on an unassayed design is not a measurement. The generation tables
therefore carry counts, distances, entropies and measured catalogue lookups, and
no classifier column.

## 4. Stage B — continuation under matched GPU budgets (`scripts/posttrain_her2.py`)

For each seed, the selected initial-SFT checkpoint is cloned twice and continued
two ways from that same parent. The two methods differ in what they train on, not
only in the loss:

| Method | Trains on | Reference | Budgets (measured GPU seconds, per seed) |
|---|---|---|---|
| continued SFT | the eligible **high-bin positives** (120,477 rows), same objective as initial SFT | none | 180, 360, 600 |
| DPO | distance-matched **high-vs-low pairs** drawn from those positives and the 113,530 low rows | frozen cache of the parent over the 234,007-row union | 180, 360, 600, 1200, 1800 |

Both use full parameters, AdamW lr 1e-5, betas (0.9, 0.999), weight decay 0.01,
clip 1.0, and a **fixed 100-update linear warmup then a constant rate**. The
schedule is deliberately horizon-independent: a budget-stopped run has no known
total step count, and a cosine would make the 1800 s trajectory something other
than a continuation of the 180 s one.

Each method runs as **one continuous trajectory** that drops a checkpoint as it
passes each budget. Nothing is restarted per budget.

### Preferences

Chosen = a `high` row; rejected = a `low` row at the **same** Hamming distance to
the wild-type core, so that "closer to trastuzumab" is constant within a pair.
`mid` is kept for the classifier and is never treated as a non-binder. Eligible
training chosen rows: 120,477 (27 distance-1 high rows have no low partner and are
excluded); training low rows: 113,530; reference population 234,007. Independent
fixed validation pairs come from the 25,722 eligible validation high rows (100
excluded).

One cycle visits every chosen row once under a seeded shuffle. Within a distance
group, the k-th chosen row takes the k-th entry of a fixed permutation of that
group's low rows at an offset that advances by the group's size each cycle, so a
long run sweeps the entire low set rather than resampling it. Every incomplete
batch is trained on. No proxy or adversarial negatives, and no test rows.

At batch 128 sequences: 64 DPO pairs per update; continued SFT uses 128 chosen
positives per update over the same eligible population. (Fallback 64 → 32 pairs.)

### DPO

`beta = 0.1`, loss from `experiments.dpo.dpo_per_pair_loss`, policy term = **sum**
of the ten core log probabilities. The reference is the frozen selected parent,
scored **once** over the 234,007-row union without gradients and cached. The cache
is bound to the checkpoint hash, config hash, scaffold prefix, probability
convention, core order and a content digest; it is immutable, and a fresh parity
re-score at sampled indices must agree or the run fails. A scoring callback on the
current weights is not a substitute and is not used.

### Budget accounting

Budgets are **measured device-elapsed seconds of training updates plus DPO reference scoring**, taken with
CUDA events at 4 intra-op CPU threads (the setting the probe timings were taken
at, declared in `configs/experiments/her2_posttrain.json` under `runtime`).

Recorded at **every** budget, for both methods:

* nominal target and actual GPU seconds, the last update's duration, and whether
  the overshoot is within that one update — checked, not asserted;
* cumulative **distinct** chosen and rejected row ids reached so far, with the
  population sizes they are a fraction of. A per-batch unique count answers a
  different question and is kept under `batch_unique_*`;
* sequence, pair and core-token exposures (core tokens = sequences × 10);
* total trajectory wall time, and separately the excluded checkpoint/validation
  wall time.

Further rules:

* The budget is checked between updates, so the actual time may exceed the target
  by at most one update. The actual value is what is recorded, and selection
  compares the **nominal** target — comparing the measured time would exclude a
  checkpoint from the budget that defines it.
* The reference cache creation is charged once, inside DPO's own budget. A reused
  cache still reports the original creation seconds as the cold-start cost, and
  the warm reuse wall time is measured around the completed reload and reported
  separately. If that pre-charge alone reaches the smallest budget, the run
  **fails** rather than reporting a first checkpoint whose one-update overshoot
  bound is fiction.
* Validation, generation and checkpoint I/O are excluded from the training budget
  and reported separately — excluded, not subtracted.
* The shared initial-SFT cost is reported once, in **wall seconds on the recorded
  device**, and is charged to neither continuation: both start from the same
  parent. It is not comparable to the continuation budgets and is labelled so.
* These are device timings on one GTX 1650 Super. They are **not** FLOP or energy
  measurements, and **no equal-update or equal-epoch fairness is claimed.**

## 5. Stage C — validation, generation and the final freeze

Every raw budget checkpoint — all 3 × (3 + 5) = 24, plus the 6 initial policies and
the zero-shot model — is streamed one at a time onto the 4 GB card and measured:

* full validation AP / AUROC / P@32 / P@100 / P@1000, positive NLL, and held-out
  preference-pair metrics (pair accuracy, implicit-reward accuracy, margins);
* 10,000 draws at temperature 1 with duplicates retained: unique fraction, maximum
  single-core frequency, summed per-site entropy, mean pairwise Hamming, exact
  training novelty, distance to the nearest training core, distance to wild type,
  Monte Carlo entropy from the sum log probability, and MC KL to both the zero-shot
  model and the policy's own SFT parent with standard errors. The KL estimate may
  be negative near zero; that is sampling error and it is not clamped.

Draws, their metrics, the per-policy validation score vector and their hashes are
persisted here, before any reserved label is read, and the evaluation re-reads
them rather than re-sampling.

Each validation record carries an **identity** — config hash, code hashes, source
hashes, parent, scaffold prefix, validation-pair digest, draw seed, draw count and
temperature. A record is reused only when that identity matches and the files it
names still hash to what it recorded; otherwise it is recomputed. A Monte Carlo KL
is reused only when it was measured under the same reference checkpoint hash and
the same draw file.

### Reference-relative ranking (diagnostic)

Alongside the raw density, validation reports each policy's ranking under
`log pi - log pi_parent` (DPO's implicit reward, whose `beta > 0` is a rescaling
that cannot reorder anything) and under `log pi - log pi_zero-shot`. The same
three rules are reported on the test split and the assay cohort. They are
**diagnostics**: selection is by raw-density validation average precision, and
choosing whichever rule ranked best would be exactly the outcome-driven selection
this protocol is arranged to avoid.

### Preregistered diversity eligibility

Fixed in advance, and heuristics for distributional collapse rather than a
guarantee of anything functional. A checkpoint is eligible when:

* mean pairwise Hamming **and** summed site entropy are ≥ 0.90 of **both** the
  exact full training-high reference and the policy's own SFT parent (drawn under
  the same rule and draw count);
* unique fraction ≥ 0.90;
* maximum single-core frequency ≤ 0.01;
* ≥ 0.50 of draws are not exact training cores.

### Selection

For each method × seed × budget: the **best eligible checkpoint at or below that
budget** by validation average precision, ties broken toward the earlier budget. A
budget where nothing is eligible is recorded as `null` with the reason; the
zero-budget parent is never quietly promoted into a successful slot. Initial-SFT
selection remains validation positive NLL. Thresholds are never revisited after
looking at results, and no "best seed" is chosen.

`selection_frozen.json` (stage `final`) names every artifact, including the failed
ones, plus the selection mapping, config/source/code hashes, the **exact** declared
method/seed/budget name sets, and generation hashes. It is the only file that
unlocks reserved labels. This is a workflow and schema guarantee, not a
cryptographic one.

### The stage contract

Every stage verifies the previous stage's evidence **before** it writes or reads
anything:

* the continuation reads `base_selection.json` checking stage, config hash, code
  hashes, source hashes, the exact expected artifact names and every checkpoint
  hash — and the base-stage unlock it obtains is refused by every reserved loader,
  which re-checks the stage at the point of access rather than trusting the token's
  type;
* `continuation_results.json` and `training_results.json` are identity-checked
  before being rewritten, so a validate-only or freeze-only invocation cannot
  re-attribute completed runs to the current code or commit. The commit that first
  claimed a run directory survives a restart; the restart's own commit is recorded
  separately;
* the freeze refuses stale or missing validation records, including incomplete KL
  and relative-ranking diagnostics; record identity also binds the selected parent;
* the evaluator verifies the freeze against the **current** code and source hashes, the
  declared name sets, the validation-records file, every draw file and every
  validation score vector before it opens the reserved split or the workbook.

## 6. Stage D — evaluation (`scripts/evaluate_her2.py`)

Runs once, on a fresh output directory, and refuses to overwrite a completed run.

* **Reserved test split (78,652 rows):** every scorer and every raw budget
  checkpoint through one metric function, so the scaling curve and the selected
  points are the same code. Strata by distance to wild type and to the training
  set. Paired bootstrap on the declared matched comparisons only (DPO vs its own
  continued-SFT control at the same budget; DPO vs its parent; the original arm
  comparisons) — the quadratic all-pairs table is deliberately not computed. The
  observed difference is the point difference on the original sample; the bootstrap
  mean is reported separately. Seed spread is reported apart from the intervals.
* **Generation:** the persisted draws are re-read by hash and joined against the
  now-unlocked test labels, giving exact catalogue hit counts and conditional high
  rates **per split**, aggregated separately over the held-out splits. The pooled
  number is kept only under a name that says it is pooled. These are lookups of
  sequences that already carry a published bin; no new assay is conducted on any
  generated sequence.
* **Independent assay cohort:** the Buzz SPR workbook, restricted to the same
  scaffold and a canonical 10-mer core, one of the five design methods, and no
  exact overlap with any published split — expected 695 unique primary designs.
  Controls are reported separately. Spearman against −log10(KD) on rows with a
  finite KD, with a **bootstrap** confidence interval (a permutation distribution
  is a null and is reported, if at all, as a p-value under its own name). `I.C.`
  rows are binding-observed positives in the binary endpoint with no KD, never a
  fabricated number and never a negative. Workbook metadata declares Carterra CMDP
  SPR; the repository README abstract says BLI, and the discrepancy is preserved.
  Within-method strata reduce cross-method generative bias and do **not** make the
  cohort selection-bias free.

## 7. Known limitations

1. **p-IgGen's HER2 exposure is unscanned and unresolved**, for both the paired
   fine-tuning corpus and the unpaired heavy archive. An earlier paired-corpus scan
   in this repository queried CR9114, not HER2, and says nothing here. No
   clean-exposure claim is made.
2. Labels were read for aggregate integrity checks, and two example rows per split
   were printed, during the 2026-09-18 audit. That is disclosed; the splits were
   not sealed from the start. What is checkable is narrower: no outcome-based model
   selection and no final evaluation have occurred.
3. The AbSci release is a **support-mismatch diagnostic only**. In-support means
   jointly: HCDR3 length 13, the `SR`/`Y` anchors, a canonical 10-mer core, and the
   fixed trastuzumab H1/H2. Designs outside that support are not cropped into this
   benchmark and compared.
4. The classifiers are independent of the policy in parameters, architecture and
   objective, but **not in data**: they are fit on the same library.
5. Ranking held-out members of this library is interpolation. Nothing here measures
   de novo design.

## 8. Running it

```bash
python scripts/audit_her2.py
python scripts/train_her2.py  --stages policies,classifiers,linear,select
python scripts/posttrain_her2.py --stages continue
python scripts/posttrain_her2.py --stages validate,freeze
python scripts/evaluate_her2.py
```

Code, config and this protocol are committed before any fitting begins.
