# Bounded functional midtraining — configurations and runbook

This directory holds the **templates** for the low-resource trastuzumab transfer experiment and the
exact workflow that uses them. The software is built and independently reviewed; this document makes
it usable. It is not a dataset, and it reports no measurement.

**What this document does not claim.** It reports no biological gain from any stage, no GPU
throughput, memory fit or cost, and no test result (record the suite you ran, against the frozen
source, in your own run notes). Every number in a template — 62,500 updates, 16 examples per
update, a 189-trial study — is an *illustrative schema value*, not a recommended or endorsed budget.
Choose budgets after profiling the real workload on the target GPU (see [Compute](#compute-qualification-before-any-paid-run)).

Contents: [Objectives](#1-what-each-objective-estimates) ·
[Contrasts](#2-the-scientific-contrasts) · [Data contracts](#3-data-contracts-and-order-of-operations) ·
[Functional arm](#4-the-functional-arm-in-detail) · [Configs](#5-the-files-in-this-directory) ·
[Workflow](#6-workflow) · [Retention](#7-retention) · [Study](#8-the-preregistered-study) ·
[Checkpoints](#9-boundaries-checkpoints-and-the-frozen-source) · [Trust](#10-what-is-and-is-not-established) ·
[Compute](#compute-qualification-before-any-paid-run) · [Source map](#source-map)

---

## 1. What each objective estimates

All objectives act on a DPLM backbone with a trainable final-hidden residual fusion; the interaction
encoder is frozen in every stage and receives no gradient. Model inputs are residues only: the
antibody heavy chain `H`, the optional light chain `L`, and (in the *paired* context arm) the
antigen `Ag`. Record IDs, source IDs, assay IDs, groups, families and measured outcomes are never
tensors.

| Objective | Where | What it estimates | What it is not |
|---|---|---|---|
| **Masked denoising** | every midtraining and post run | The mean masked-site reconstruction NLL: draw an eligible-site mask (size uniform over `1..n`, then a uniform subset), average cross-entropy over the masked sites, then over the examples of an update. This estimates the mean sequential reconstruction NLL averaged over random reveal orders. | The exact likelihood of the iterative generation sampler. Not a binding measure. |
| **Functional observation head** (optional, midtraining only) | `functional` section of a mid config | A margin surrogate for *within-group ordering*: whether a scalar read-out of the antibody's pooled representation separates the measurably better construct from the measurably worse one. Per pair: `weight / pairs_per_update × relu(margin − (s_better − s_worse))²`, where `s = head(mean of the final hidden state over the antibody's H/L residue slots of the complete, unmasked view)`. Tie and unresolved opportunities contribute zero. | Not an affinity predictor and not calibrated: the head is a bias-free `Linear(H, 1)` trained against ordinal pairs only, and it is **discarded at parent export**. Not a universal ordering — only pairs inside one curated group are ever compared. |
| **Post-training** | `post_650m.json` | Reconstruction of the selected target cores plus a whole-core ordering hinge between two selected endpoints of the same assay (teacher-forced under one reveal order, 20 states per informative pair). | Not a KD target; the margin is dimensionless. |
| **Retention read-out** | `plan-retention` / `evaluate-retention` | The same mean masked-site NLL as the denoising objective, on frozen held-out masks, with no update. It shows how much broad masked reconstruction a checkpoint kept or lost. | **Reconstruction, not binding.** Not evidence of biological success. |
| **Study read-out** | `plan-study` / `score` / `select-study` | Ordinal concordance of whole-core sequence scores with measured development values: over unordered pairs with different oriented values, the fraction whose score order agrees (equal scores count one half; measured ties are excluded). `0.5` is the expectation of an uninformative score. | Not affinity, not a correlation, not a calibrated probability. |

Two properties of the functional objective follow from the schedule and are worth keeping in mind:

* The weight is per **group**, not per observation: a pair-eligible group is drawn uniformly, then two
  distinct selected members of it. A small group is drawn as often as a large one, and a construct
  measured in two groups is two observations but one physical construct.
* Pair comparison is **within-comparable-assay ordering only**. Curators assert that the listed
  endpoints, units and directions inside a group measure one property under one condition scope;
  the software checks the assertion's *shape* (below), not its truth.

## 2. The scientific contrasts

Three axes vary independently. Keep them separate when you read a result.

| Axis | Values | Set by |
|---|---|---|
| **Broad stage** | direct release (no midtraining) · denoising only · + functional head **detached** · + functional head **shared** | presence and `gradient_mode` of the `functional` section |
| **Context arm** (midtraining) | `paired` (antigen visible to the encoder) · `abonly` (antibody only; the same frozen encoder on the visible antibody) | `training.arm` |
| **Update mode** (midtraining) | `lora` (adapters; fusion still trains) · `full` (whole DPLM plus fusion) | `fusion.update_mode` |

The intended midtraining contrast is **denoising only vs detached head vs shared head**, holding the
denoising data, draws and compute contract matched, and budgeting the auxiliary labels separately:

* *Detached* — the head trains; the policy receives nothing from the functional pairs (features are
  computed without a graph). It isolates "the head can read the representation".
* *Shared* — the same pairs also send gradient into the policy, accumulated with the denoising
  gradient before the **single** policy clip. It tests "ordering supervision changes the representation".

Paired/abonly and LoRA/full are **controls on the whole midtraining stage**. The context arm applies to
the functional features too: in an `abonly` run the complete view used for pooling omits the antigen,
and both complete views of every functional train construct are preflighted so the admitted set does
not depend on the arm. Copy a mid config and change `training.arm` or `fusion.update_mode` for the
matched control. Note that
the templates ship different learning rates for the two modes (`1e-4` LoRA, `2e-5` full): a LoRA/full
comparison therefore mixes mode and learning rate unless you equalize or sweep them deliberately.

The **study's post recipe is fixed**: paired context, a fresh LoRA, one shared recipe for every parent.
It accepts parent origins whose midtraining was LoRA or full, and does not support a full-fine-tune
post stage. Which mid recipes, data budgets and update counts are compared across arms is a
**preregistered design obligation**; the selector verifies the declared post contract and each parent's
identity and lineage, not every possible confound between arms.

## 3. Data contracts and order of operations

Inputs are strict JSON (unknown fields, duplicate keys and non-finite numbers are rejected). Nothing is
discovered automatically. The order is fixed by the code so that outcome-bearing files cannot influence
what is selected:

1. **Outcome-free broad document** (`trastuzumab-pairs/1`) and **outcome-free functional-construct
   document** (`trastuzumab-functional-constructs/1`) are opened. *No measured value exists in either.*
2. **Qualification** of *all* raw broad rows (retained and excluded alike) and *all* functional inputs
   against each other: shared content must agree on format, family and partition; no family on a
   separation axis may span partitions; the transfer guard checks every functional **train** construct
   — whether or not the budget later selects it — against the downstream endpoint manifest.
3. **Budget**: a nested, seeded identity-hash prefix of the train observations, then the exact pair
   schedule is replayed to report coverage. Still no measurement.
4. **Authorized loading** of the selected measurements only, and only if their `selection_digest`
   matches the fixed budget.

Curated groups and families *assert* a compatible scalar property and assay context. The software does
not discover comparability or homology; it checks shape (below). Metadata and hash checks establish that
supplied records agree with each other — not that they are true, not hidden history, and not the
chronology of a preregistration.

Schematic documents (angle brackets are what you must supply; none of this is data):

<details><summary>Functional constructs — outcome-free</summary>

```json
{
  "schema": "trastuzumab-functional-constructs/1",
  "pairs_declaration_digest": "<input.declaration printed by `plan` on a mid config with functional: null>",
  "study_contract": {
    "target_property": {"property_id": "<id>", "description": "<text>"},
    "condition_scope": {"scope_id": "<id>", "description": "<text>", "context_ids": ["<context id>"]},
    "compatible_endpoints": [
      {"semantics": "<text>", "units": "<text>", "direction": "higher_better", "justification": "<why one head may serve this>"}
    ],
    "assertion": "<the curator's claim that these endpoints order one property>"
  },
  "sources": [{"source_id": "<id>", "description": "<text>"}],
  "constructs": [
    {"construct_id": "<id>", "source_id": "<id>", "source_record_id": "<id>",
     "H": "<residues>", "L": "<residues or null>", "Ag": "<residues>",
     "antibody_format": "paired", "antibody_family": "<id>", "target_family": "<id>",
     "partition": "train", "evidence_reference": "<text>"}
  ],
  "groups": [
    {"group_id": "<id>", "context_id": "<context id>",
     "endpoint": {"semantics": "<text>", "units": "<text>", "direction": "higher_better"},
     "antigen_sha256": "<sha256 of the group's antigen sequence>",
     "member_construct_ids": ["<id>", "<id>"], "evidence_reference": "<text>"}
  ]
}
```
Checked: each group's endpoint equals an allowlisted entry and its context is in scope; members are
unique, known, in one partition, and carry the group's antigen byte-for-byte; every construct is in at
least one group. **Not checked:** transitivity inside a group, rank reversals under unobserved
conditions, consistency between groups.
</details>

<details><summary>Functional measurements — restricted; opened only after the budget is fixed</summary>

```json
{
  "schema": "trastuzumab-functional-measurements/1",
  "selection_digest": "<functional.budget.selection_digest printed by `plan`>",
  "measurements": [
    {"group_id": "<id>", "construct_id": "<id>", "measurement": {"kind": "point", "value": 1.0}},
    {"group_id": "<id>", "construct_id": "<id>", "measurement": {"kind": "interval", "lower": 0.5, "upper": 2.0}},
    {"group_id": "<id>", "construct_id": "<id>", "measurement": {"kind": "lower_bound", "lower": 10.0}},
    {"group_id": "<id>", "construct_id": "<id>", "measurement": {"kind": "upper_bound", "upper": 0.1}}
  ]
}
```
Exactly the selected `(group_id, construct_id)` observations — a missing, extra or duplicate row is
rejected. Censoring is an explicit `kind`; infinity is never written.
</details>

<details><summary>Endpoint manifest, selected labels, development labels</summary>

```json
{"schema": "trastuzumab-endpoints/1", "entries": [
  {"endpoint_id": "<id>", "role": "<post_pool | development | reserved>", "neighbourhood_group": "<id>",
   "assay_id": "<id>", "endpoint_semantics": "<text>",
   "construct": {"heavy_prefix": "<residues>", "core": "<10 residues>", "heavy_suffix": "<residues>",
                 "light": "<residues>", "Ag": "<residues>"}}
]}

{"schema": "trastuzumab-selected-labels/1", "selection_digest": "<from select-budget>",
 "measurements": [{"endpoint_id": "<id>", "value": 0.0, "orientation": "higher_better", "assay_id": "<id>"}]}

{"schema": "trastuzumab-study-development-labels/1", "plan_sha256": "<plan-study plan_sha256>",
 "assay_id": "<plan development.assay_id>", "orientation": "<plan development.orientation>",
 "labels": [{"endpoint_id": "<id>", "value": 0.0, "orientation": "<same>", "assay_id": "<same>"}]}
```
The selected-labels file must contain **exactly** the selected endpoints; the development-labels file
exactly the planned development endpoints.
</details>

## 4. The functional arm in detail

**Observations and the budget.** An observation is a `(group_id, construct_id)` pair. The budget of
size `N` is the first `N` train observations ordered by `sha256(seed ␀ group ␀ construct)`, so budgets
sharing a seed **nest as prefixes**. Singleton groups can never be compared; they stay in the coverage
denominator. `minimum_compared_fraction` (default `1.0`) refuses a run whose replayed schedule would
compare fewer than that fraction of the selected observations at least once. This gate is
outcome-free: it counts *comparisons*, not informative labels or non-zero gradients.

**Comparing intervals.** After orienting by the group's direction, strictly disjoint closed intervals
give an **ordered** pair; two equal points give a **tie**; every other case — overlap, touching
bounds, censoring that cannot separate the values — is **unresolved**. Nothing is guessed. Only
ordered pairs produce a loss.

**Sampling without redraw.** A draw is a pure function of `(budget, seed, update, k)`: a uniform
pair-eligible group, then two distinct selected members. The pair is read from the schedule *before*
its values are consulted and is never replaced. A tie or unresolved pair still consumes its
opportunity, so there is no search for an informative label.

**Isolation from the denoising stream.** The denoising batches, masks and model RNG are identical with
and without the functional arm. The functional forwards and backwards run under their own keyed Torch
stream (`functional-model`), the head is initialised `Normal(0, H^-½)` on CPU under a purpose-keyed
isolated RNG, and the head trains with its **own** AdamW and its **own** gradient clip
(`head_gradient_clip`) — never the policy's. The head takes one step per update that backpropagated at
least one ordered pair (even if every hinge is inactive); its learning rate is the training schedule at
the *global* update times `head_lr_scale`. In `shared` mode the policy gradient joins the denoising
gradient before the single policy clip; in `detached` mode the policy gets nothing.

**Exact resume.** Draws are pure functions of `(seed, update, purpose)`, so a checkpoint stores
`next_update` rather than generator state. Resume requires a pristine trainer and validates the
identity, **both** optimizer states (policy and head), and a label-replay of the functional counters
before changing any live tensor. The draws of a resumed run are identical by construction; no claim is
made that floating-point results are bit-identical across different hardware.

**Parent handoff.** `export-parent` merges LoRA into an FP32 backbone and keeps the learned fusion. The
**head weights and both optimizer states are discarded**; the head's description stays only as
supervision provenance. The post stage loads that parent, attaches fresh zero-output adapters and a
fresh optimizer, and fits the target budget from scratch.

## 5. The files in this directory

| File | Purpose |
|---|---|
| `mid_650m_lora.json`, `mid_650m_full.json` | Denoising-only midtraining (`functional: null`), LoRA / full. Includes the required `transfer_guard`. |
| `functional_{detached,shared}_650m_{lora,full}.json` | Same midtraining plus a `functional` section; `gradient_mode` and `fusion.update_mode` are in the file name. |
| `post_650m.json` | Fixed post recipe: fresh LoRA from a parent (`initialization.kind: "parent"`), selected-label budget, `functional: null`. For the direct-release baseline set `initialization` to `{"kind": "release"}`. |
| `retention/heldout_spec.json` | Specification for `plan-retention`. |
| `study/study_spec.json` | Specification for `plan-study`: recipe, budgets, identity, arms, origins. |

Retention and study specifications live in sub-directories on purpose: the example-configuration test
globs `*.json` in this directory and expects exactly the seven run configs.

Every `REPLACE…` value and every `000…a001`-style hash is an **inert placeholder**. Copy the examples,
replace *all* of them with real paths and the hashes the tools print, and never invent a hash. Path
rules: a **run config's** paths resolve relative to the config file; **spec** paths resolve relative
to the spec file; **CLI flag** paths are relative to the working directory; a results index's
`export_path` is relative to the index file.

`run.json` fields, in one place: `schema` (`trastuzumab-run/2`), `model`, `fusion`, `training`,
`data`, `initialization`, `device`, and `functional` (`null`, or `config` + `constructs_path` +
`measurements_path` + `budget_n` + `budget_seed`). `functional` is rejected unless `training.stage` is
`mid`. A released midtraining run requires `data.transfer_guard`. The functional `config` keys are
`gradient_mode`, `weight`, `pairs_per_update`, `margin`, `head_lr_scale`, `head_weight_decay`,
`head_gradient_clip`, `minimum_compared_fraction` (defined in
[`functional_train.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/functional_train.py)).

## 6. Workflow

Run from the repository root, inside the project virtual environment. Every `--output` must not exist
yet; for JSON outputs the parent directory must exist (a `train` output directory is created for you).
`python scripts/trastuzumab_train.py <command> --help` lists every flag.

| Command | Reads | Writes |
|---|---|---|
| `qualify` | broad pairs | coverage report |
| `select-budget` | endpoint manifest | post budget identity |
| `plan` | run config; broad pairs; (mid) functional constructs and endpoint manifest — **no weights, no measurements** | input plan (`trastuzumab-input-plan/2`) |
| `profile` | run config; opens the model and the selected measurements | timing / memory report |
| `train` | run config; selected measurements | run directory |
| `export-parent` | run config + a mid checkpoint | merged `parent.pt` |
| `score`, `generate` | run config + checkpoint + a query panel | export directory |
| `plan-retention`, `evaluate-retention` | spec; plan + checkpoint | retention plan; result |
| `plan-study`, `select-study` | spec; plan + exports + (last) development labels | study plan; selection |

### 6.0 Freeze the source first

Checkpoint, parent, score and resume identities include a hash of
`experiments/trastuzumab_poc/*.py`, `models/antibody_antigen/*.py` and `models/dplm_policy.py`. A
checkpoint written under one source tree **does not load** under another (`checkpoint source identity
mismatch`). Commit the exact source you will use *before* the first real run and run every later
command from that tree. Updated identity contracts reject earlier checkpoints rather than migrating
them.

### 6.1 Outcome-free planning

```powershell
python scripts/trastuzumab_train.py plan --config run.json --output input-plan.json
```

For a functional mid config this qualifies the broad rows and the functional constructs, selects the
budget, replays the schedule, applies the transfer guard (over **all** functional train constructs)
and reports `coverage_requirement_met`. No measurement file is opened. Read the report before
anything else: unseen observations, singleton groups, repeated constructs, and unique
antibodies/antigens (not just observation counts).

### 6.2 Profile on the target GPU

```powershell
python scripts/trastuzumab_train.py profile --config run.json --warmup-updates 3 --measured-updates 10 --output profile.json
```

`profile` performs **real updates** on a freshly built model (never use it as a read-only probe of a
live run). It counts *denoising* examples per second via `examples_per_update`, but the timing also
includes the auxiliary functional pairs, so read the rate as "denoising examples per second *while
carrying* the auxiliary load". `warmup + measured` must not exceed `total_updates`.

### 6.3 Train, stop at a boundary, resume

`--stop-after-updates` is an **absolute** boundary inside the original horizon; the declared
`total_updates` (and any cosine schedule) never changes. `checkpoint.pt` is overwritten in place, so use
a **fresh output directory per boundary / resume segment**.

```powershell
python scripts/trastuzumab_train.py train --config run.json --output run-u50 --stop-after-updates 50 --checkpoint-every 50
$sha50 = (Get-Content -Raw run-u50/result.json | ConvertFrom-Json).checkpoint.sha256
python scripts/trastuzumab_train.py train --config run.json --output run-u100 --resume run-u50/checkpoint.pt --resume-sha256 $sha50 --stop-after-updates 100 --checkpoint-every 50
```

A run directory holds `config.json`, `identity.json`, `metrics.jsonl`, `checkpoint.pt` and
`result.json` (`status` is `complete` or `stopped_at_boundary`), plus `exposure-plan.json` (post) or
`functional-exposure-plan.json` (functional mid). Resume keeps the **original initialization**.
Always pass `--resume-sha256`.

### 6.4 Mid → parent → post

```powershell
$midSha = (Get-Content -Raw mid-final/result.json | ConvertFrom-Json).checkpoint.sha256
python scripts/trastuzumab_train.py export-parent --config mid.json --checkpoint mid-final/checkpoint.pt --checkpoint-sha256 $midSha --output parent.pt
```

`export-parent` prints `{"sha256": …}`; put that value in the post config's `initialization.sha256` and
the parent path in `initialization.path`. The post config's `data` names the endpoint manifest, the
selected-labels file, `budget_n` and `budget_seed`. For the direct-release baseline use
`{"kind": "release"}` instead.

## 7. Retention

Retention scores a fixed set of qualified **val/test** broad records, each with fixed literal masks, by
the production reconstruction NLL — at saved mid/post training boundaries, on one plan. The
documented CLI workflow evaluates saved mid/post checkpoints. `plan-retention` needs no checkpoint.
`evaluate-retention` requires one; the training CLI accepts only positive stopping boundaries, so
this workflow does not provide a bare-release or zero-update evaluation. A direct-release comparison
here means a post checkpoint initialized from release weights.

```powershell
python scripts/trastuzumab_train.py plan-retention --spec retention-spec.json --output retention-plan.json
python scripts/trastuzumab_train.py evaluate-retention --config run.json --plan retention-plan.json --checkpoint run-u50/checkpoint.pt --checkpoint-sha256 $sha50 --arm paired --output retention-u50.json
```

* **Plan.** Opens only the files the spec names, and no model. Each selected record is checked against
  *all* raw denoising training rows, *all* functional train constructs (selected or not), and *all*
  protected endpoint entries on the exact `(H, L)`, the exact antigen and the declared family axes.
  Set `functional_constructs_path` to the **full** functional constructs document for any functional
  comparison; the same plan then also works for no-aux and direct-release post checkpoints. `record_ids` and
  `limit {count, seed}` are mutually exclusive.
* **Aggregation.** Mean over masked sites → mean over a record's masks → mean over records. Sites and
  masks are never pooled.
* **Failure.** A numerical failure of one mask (non-finite NLL or CUDA out-of-memory) is recorded and
  the result is `incomplete` with **no aggregate at all** — never an aggregate over the successful
  subset. Evaluation preserves module modes and the Python/Torch generators.
* It measures reconstruction, **not** binding, and cannot rule out that the frozen encoder saw the
  held-out records in its own pretraining.

## 8. The preregistered study

Question: after different broad stages, how fast does a fixed low-resource post recipe adapt?
Every arm shares one post recipe (only learning rate and post seed vary), one nested budget family,
one horizon with declared boundary updates, one development panel and one read-out.

**Plan** (specification and endpoint manifest only; no label, model or checkpoint):

```powershell
python scripts/trastuzumab_train.py plan-study --spec study-spec.json --output study-plan.json
```

The plan seals, under `plan_sha256`, the full development panel (every development entry), the literal
reveal orders, the nested budgets, the pins below, and the **full trial × boundary inventory**.
**Pin the `plan_sha256` that `plan-study` wrote, and freeze the plan before any target post outcome.**

| Pinned in the specification | Source |
|---|---|
| source | `source_identity()["sha256"]`, or `source.sha256` in a run's `identity.json` |
| runtime | `execution` (hardware, software, parameter counts) from a **post** run's `identity.json` → `runtime` |
| encoder | `encoder.identity` / `encoder.runtime` from that run's `model.encoder` / `encoder_runtime` |
| release base | `release_base_sha256`: `artifacts.digest(identity["model"]["base"])` of a release-initialized run's `identity.json` |
| each mid origin | `parent_sha256` (`export-parent`), `parent_checkpoint_sha256` (the mid checkpoint), `parent_provenance_sha256` (below) |
| each budget | `selected_data_sha256`, `selected_labels_sha256` — **opaque fitting-label pins** |

The two label pins are **supplied, never derived by the planner**. Obtain them with the existing
loader, outside planning:

```python
from smallAntibodyGen.experiments.trastuzumab_poc.inputs import load_post
restricted, identity = load_post(manifest_path, labels_path, budget_n=n, budget_seed=seed)
pin = {"budget_n": n, "selected_data_sha256": identity["selected_data"],
       "selected_labels_sha256": identity["selected_labels_sha256"]}
```

This deliberately **opens the authorized fitting labels**; it is label preparation, not outcome-free
hashing. The `budget_seed` must equal the spec's `budgets.budget_seed`. The parent provenance digest:

```python
import torch
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest
print(digest(torch.load("parent.pt", map_location="cpu", weights_only=True, mmap=True)["provenance"]))
```

**Run the trials.** For every planned trial (arm × origin × budget × learning rate × post seed) train
through each declared boundary (§6.3), and `score` at each boundary with the flags the spec's `query`
section gives. Score takes the plan's `queries` section — **not the whole plan**:

```powershell
python -I -c "import json,sys; json.dump(json.load(open(sys.argv[1], encoding='utf-8'))['queries'], open(sys.argv[2], 'x', encoding='utf-8'))" study-plan.json development-queries.json
$postSha50 = (Get-Content -Raw post-u50/result.json | ConvertFrom-Json).checkpoint.sha256
python scripts/trastuzumab_train.py score --config post-trial.json --checkpoint post-u50/checkpoint.pt --checkpoint-sha256 $postSha50 --queries development-queries.json --arm paired --scorer-id whole-core --order-seed 0 --orders 4 --output scores-u50
```

`post-trial.json` is a post run config whose `training` equals the spec's `recipe.training` plus that
trial's `learning_rate` and `seed`, whose `fusion` equals `recipe.fusion`, and whose `initialization`
and `data` name the trial's parent (or `release`) and budget. A score export is a directory
whose `manifest.json` is written last.

**Results index.** One record per (trial, boundary) — an export or an explicit failure. Export paths
are relative to the index file:

```json
{"schema": "trastuzumab-study-results/1", "plan_sha256": "<plan_sha256>", "records": [
  {"trial_id": "<id>", "boundary": 50, "status": "exported", "export_path": "scores-u50", "manifest_sha256": "<sha256 of scores-u50/manifest.json>"},
  {"trial_id": "<id>", "boundary": 100, "status": "failed", "reason": "<why>"}
]}
```

**Select.**

```powershell
python scripts/trastuzumab_train.py select-study --plan study-plan.json --manifest endpoints.json --results results-index.json --labels development-labels.json --output selection.json
```

The selector validates the whole inventory and **every** export (byte hashes, the full panel, exact case
IDs, the checkpoint's training/progress identity, source, fusion, encoder, runtime, origin lineage, the
available fitting budget and both label pins) **before** the development labels are opened. A missing,
extra, duplicated or inconsistent record blocks selection; nothing is written first.

* **Rule.** A learning rate is chosen per (arm, budget) at the **final horizon** only, by the exact
  mean concordance over **all** planned (origin, post seed) trials — never from one checkpoint, never
  from an intermediate boundary. It is eligible only if every planned trial has a valid final export
  (one failed query invalidates the candidate). An exact tie chooses the smallest rate.
* **Winners are all seeds, not the best seed:** every origin × seed checkpoint of the winning rate.
* **Two curves, kept apart.** The *prospective* fixed-rate curve (the rate declared in the plan before
  any development label) is the one for adaptation-speed comparisons. The *retrospective* selected-rate
  curve is optimistic, because the final metric chose the rate.
* **Label accounting.** Unique target labels = the largest nested fitting budget ∪ the fixed, disjoint
  development IDs. Repeated trials and boundary scorings are reported as **opportunities**, separately.
  Broad functional supervision is a third, separate count, `unavailable` (unknown, not zero) where no
  verified export exists.
* The development set is small and reused by every scoring: the selected rate is optimistic for those
  labels and says nothing about other endpoints.

The shipped spec lists 7 origins (release plus two origins in each of three mid arms) × 3 budgets ×
3 learning rates × 3 seeds = **189 post trials**, and 4 boundaries = **756 exports**. This is an
illustration of the schema's size, not a budget commitment.

## 9. Boundaries, checkpoints and the frozen source

* Each boundary of each trial is its own checkpoint, in its own output directory. **Nothing here
  removes a checkpoint automatically** — not training, scoring, nor selection. Keep every boundary
  artifact; a checkpoint hash may not repeat across records, and a post checkpoint is never a parent.
* Always pass the expected checkpoint SHA-256 (`--checkpoint-sha256`, `--resume-sha256`). A mismatch
  fails before any load.
* Checkpoints are atomic replacements of a single latest file; a failed update cannot be saved.
* Resume is not "continue from anywhere": the identity (config, model, data, source, initialization,
  optimizer, precision, runtime, RNG) must match exactly, or training must restart.

## 10. What is and is not established

Established by the code: that inputs, pins and records are internally consistent and agree with each
other; the order of operations in §3; the schedule properties in §4; that failures are reported rather
than dropped.

**Not** established: the truth of curator assertions (compatible endpoints, family assignments,
partitions, evidence); homology or leakage beyond exact sequence equality and the declared family
axes; unreported history or the chronology of a preregistration (an author who recomputes every digest
can write a self-consistent set of records); that two arms differ only in what you intended; the
encoder's own pretraining exposure; and any biological effect. Sequence scores are neither measured
affinity nor proof of binding. Manifests and metadata prove supplied consistency only.

## Compute qualification before any paid run

No H100 fit, throughput or duration is established by synthetic or CPU checks, and none is claimed
here. Before committing any budget, profile the **actual worst-case workload on the intended GPU**
(single 48–80 GB class card is the target): the longest admitted joint lengths, the *shared* functional
arm (which retains a graph), the post ordering path, optimizer updates, checkpoint save/resume, and
repeated deterministic scoring. Size the study from measured per-update and per-scoring cost, not from
the template counts. These commands use an already provisioned machine; they do not rent hardware or
submit cloud jobs.

## Source map

| Concern | File (under `src/smallAntibodyGen/experiments/trastuzumab_poc/`) |
|---|---|
| CLI and run-config schema | [`cli.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/cli.py) |
| Head, pooling, pair loss | [`functional_train.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/functional_train.py) |
| Constructs document, qualification | [`functional_data.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/functional_data.py) |
| Budget, schedule, intervals, measurements | [`functional_protocol.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/functional_protocol.py) |
| Functional identity, progress, coverage gate | [`functional_checkpoint.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/functional_checkpoint.py) |
| Trainer step and update order | [`runner.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/runner.py) |
| Checkpoint and resume | [`checkpoint.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/checkpoint.py) |
| Parent export / load | [`parent.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/parent.py) |
| Transfer guard | [`transfer_guard.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/transfer_guard.py) |
| Retention | [`retention.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/retention.py) |
| Study plan and selection | [`study.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/study.py) |
| Profiling | [`profiling.py`](../../src/smallAntibodyGen/experiments/trastuzumab_poc/profiling.py) |
