# Fixed-target post-training and interpretability: research recommendation

**Date:** 2026-09-14  
**Status:** ESM-IF1 selected by the user for integration; no training run or checkpoint promotion.  
**Objective:** use established antibody modeling machinery and a learnable measured task, concentrating research novelty in post-training and causal interpretation.

**Follow-up audit:** the data corrections remain applicable. ESM-IF1 is now the selected backbone; its [readiness record](evidence/esm-if1-readiness-2026-09-14.json) is separate from the retained [p-IgGen model/exposure audit](evidence/piggen-release-audit-2026-09-14.json). See also the [data audit](evidence/fixed-target-data-audit-2026-09-14.json).

## Recommendation

Start with **CR9114 binding to H1**, allowing its **16 published variable heavy-chain sites**, and the **pretrained ESM-IF1 backbone**. Use one declared fixed structural context, with the antigen target, light chain, and all other residues fixed. Establish supervised baselines, run the existing four-arm preference experiment, and investigate the learned changes with controlled interventions.

This is a recommendation about reducing experimental and engineering risk. There is no published comparison establishing this exact task/checkpoint combination as the best affinity optimizer. The model must pass a bounded development pilot before promotion. A useful supervised-adaptation result can support the interpretability project even if DPO adds no benefit.

The most consequential scope change is permitting the measured variable sites outside HCDR3. The previous HCDR3-only restriction discards useful data and is unnecessary for the user's current research question. This remains local sequence optimization of one antibody lineage against one target.

## What the literature supports

| Primary evidence | Result relevant here | Limit on the inference |
|---|---|---|
| [Hie et al., Nature Biotechnology, online 2023](https://doi.org/10.1038/s41587-023-01763-2) | Sequence-only recommendations improved seven antibodies using at most 20 measured variants each over two rounds. | Used ESM-1b/1v consensus and laboratory selection; does not validate p-IgGen or arbitrary library likelihood ranking. |
| [Li et al., Nature Communications, 2023](https://www.nature.com/articles/s41467-023-39022-2) | Fixed-target supervised affinity modeling and Bayesian search produced experimentally improved antibody libraries. | Supports conventional supervised prediction/search; not a generative SFT or DPO comparison. |
| [ProteinDPO, Nature Methods, 2026](https://pubmed.ncbi.nlm.nih.gov/42601461/) | DPO outperformed the parent and positive-example SFT for stability scoring, with prospective stabilization experiments. | Same backbone family, but a large stability dataset; transfer to this antibody affinity task remains unproven. A released turnkey trainer has not been verified. |
| [g-DPO, 2025 preprint, v2](https://arxiv.org/html/2510.19474v2) | ESM2-650M preference training included experimental antibody expression and enzyme stability tests. | Approximate union-mask likelihood; no positive-example SFT arm; generated candidates underwent predictor-based downselection. |
| [p-IgGen, Bioinformatics, 2024](https://academic.oup.com/bioinformatics/article/40/11/btae659/7888884) | Released small paired autoregressive antibody generation, including generation of one chain given the other. | Generated-library validation is computational; not prospective affinity validation. |
| [InterPLM, Nature Methods, 2025](https://doi.org/10.1038/s41592-025-02836-7) | Protein-language-model features can be associated with biological annotations and used to alter token predictions. | Does not establish an antibody-affinity mechanism or measured improvement from steering. |
| [Antibody SAE study, v3, May 2026](https://arxiv.org/html/2512.05794v3) | Strong multivariate concept probes coexist with failed TopK steering; Ordered features have reported steering successes. | The broader TopK failure is partly unshown; this does not explain the effect of affinity post-training. |

Direct antibody evidence supports both sequence-based supervision and established structure-informed modeling. ESM-IF1 adds structural context while retaining a conventional autoregressive objective. Preference training is a credible research variable. The evidence does not justify making a DPO victory a prerequisite for project success.

## Task choice: prioritize useful measured variation

### First benchmark: CR9114 / H1

The [primary landscape paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8476123/) describes 16 binary heavy-chain sites and biological triplicates. We independently parsed the [pinned primary CSV](https://cdn.elifesciences.org/articles/71393/elife-71393-fig1-data1-v3.csv), verifying SHA-256 `ee31f8fc26fce2730d2ae772e7a095aa334b4c6c5d8e7fbfe4b4a5a1c9a794fe`:

| Antigen | Above-floor means | Floor-censored means | Missing means |
|---|---:|---:|---:|
| H1 | 63,419 | 1,675 | 442 |
| H3 | 7,174 | 58,361 | 1 |
| Influenza B | 198 | 65,336 | 2 |

H1-specific pooled within-genotype sample SD is **0.13237 log10 units**, computed from the raw measurements for 63,942 genotypes with at least two finite replicates. This is comparable to the TIGIT paired-disagreement estimator below under its assumptions: the untrimmed TIGIT estimate is about **3.9 times** larger. Comparing single-observation noise to an SEM of averaged measurements would exaggerate the ratio.

Recompute uncertainty from the raw replicate columns **before eligibility filtering, preference construction, or weighting**. In this release, 1,152 H1 rows have only one observed replicate and reported SEM zero; **526 are above the detection floor**. Using those zeros as variances would give undefined/infinite inverse-variance weights and falsely certain orderings. Sample SEM is undefined for fewer than two observations. Preserve these rows and the original fields, but require adequate replication for the initial uncertainty-gated comparison or declare a separate pooled/shrunk variance model before using them.

The SEM discrepancy is deterministic but depends on replicate count:

| Finite H1 replicates | Rows | Rows with positive sample SEM | Released SEM / conventional sample SEM |
|---|---:|---:|---:|
| 3 | 62,224 | 61,788 | `1/sqrt(2)` |
| 2 | 1,718 | 1,105 | `1/sqrt(3)` |
| 1 | 1,152 | Undefined | Released zero |

These ratios hold to floating-point precision where the denominator is positive. Thus multiplying every released SEM by `sqrt(2)` is not a general correction. Compute `std(finite_replicates, ddof=1) / sqrt(n)` for `n >= 2` under an explicitly declared convention; zero sample spread still does not resolve censoring or establish error-free measurement. The upstream derivation remains untraced, so this is a verified discrepancy from conventional sample SEM, not a demonstrated cause in upstream code.

Use H1 for the principal quantitative comparison. H3/B can support explicitly censored or classification-oriented analyses; a floor-tied ordinary regression metric does not measure ordering within the censored block. They are not universally unusable, but answer a different question.

Advantages for this project:

- Many measured combinations of a small, declared set of substitutions.
- A finite sequence space permits exact probability audits and retrospective evaluation against measurements.
- Mutation interactions provide concrete interpretation questions.
- Fixed target and parental lineage let us reuse established structural conditioning without making cross-antigen response a first-milestone requirement.

Limitations: only one lineage and two allowed residues at each site; assay bounds and missing measurements persist. All 65,536 genotypes are listed, but the complete policy support does **not** have quantitative fitness labels: H1 has 442 missing means and 1,675 censored means. Report probability mass on these categories and evaluate measured-pool selection explicitly. This is a retrospective benchmark of known library sequences; it cannot establish discovery of a new binder or a physical binding mechanism. Generalization within this landscape is not transfer to arbitrary antibodies. Previously published landscape analyses are benchmark-design prior knowledge, so disclose their use and keep model/intervention selection away from final test labels.

The earlier audit rejected this task because 15 of the 16 sites are outside HCDR3. That objection applies to the earlier editing specification; it does not make this a poor post-training benchmark.

### Second benchmark: broader PP489 / human TIGIT

A fresh read-only computation on the pinned local YM_0693 release gives:

| Filter | Usable human-TIGIT records | Unique full sequences |
|---|---:|---:|
| HCDR3-only, single target | 718 | 716 |
| All non-parental same-length PP489 variants | 13,041 | 13,015 |

For comparison, the existing paired-antigen pilot has 588 records / 587 unique HCDR3 sequences. Dropping cross-antigen completeness increases the count modestly; broadening the editing region supplies approximately 18 times as many usable human-target sequences. Changing from a cross-antigen differential to a single-target endpoint is a separate signal-quality decision and should not be conflated with this volume comparison.

The fixed-context boundary check is now closed: every usable human-target difference lies in **zero-based positions 31–113 inclusive (one-based 32–114)**. All residues outside this 83-position region, including linker and the complete light chain, equal the parent. These remain quality-filter counts rather than a finalized preference dataset. Deduplicate experimental records appropriately. Do not require strong binding for every loser: sufficiently informative weak/censored measurements may supply ordering, while missing observations alone cannot be invented as negative labels.

There are **26 exact-sequence duplicate groups, each with two distinct candidate IDs**; all 26 are double mutants. Under independent equal-variance measurement errors, `sqrt(mean(pair_difference**2)/2)` estimates single-observation error SD: **0.51859 log10 Kd**. Removing the two largest absolute differences gives **0.41422**; this is a post-hoc sensitivity analysis, not a justified exclusion. Without independence assumptions, call these scaled disagreement statistics. Distinct IDs alone do not prove independent biological replication. These groups cover about 0.2% of the usable unique sequences and are not a representative error panel. There is no direct duplicate-based error estimate for singles or triples; a triple holdout would require an explicit extrapolation of this error model.

The user's epistasis analysis is sensitive to the error estimate; it was not independently reproduced in this pass. Treat TIGIT as a later performance-replication task. It currently lacks adequate evidence for a confident claim about learned biological epistasis. That is not evidence that epistasis is absent.

Use release intervals numerically, preserving their original fields: the names describe stronger/weaker bounds and are reversed relative to numerical log-Kd ordering. Counting intervals do not capture all biological noise. YM_0988 may supply a later campaign test after auditing selection and overlap; its model-selected library is not an unbiased sample of sequence space.

The [Li et al. Ab-14 dataset](https://doi.org/10.5281/zenodo.7783546) is another credible future replication task, with published local affinity-optimization success. Its training and designed libraries have different assay dynamic ranges, so do not pool their measurements without a justified calibration.

### Correction to the noise claim

The 1.61 spread/noise ratio concerns a cross-antigen difference and noise estimated from repeated parental constructs. It is not a validated ceiling for library-wide correlation, and especially not a ceiling for within-human-TIGIT ranking. The [reproduction script](../scripts/reproduce_ym0693_cohort.py) and [manifest](../docs/research-followup/cohort/ym0693-manifest.json) now also correct their false assertion that the release has no repeated non-parental sequences. A representative replicated-variant panel and an appropriate error model would be needed to support a library-wide ceiling claim.

The proposed H1/TIGIT reliability table does not yet supply that model. H1's approximately **0.554** spread is the SD of above-floor **genotype means**, whereas approximately **0.133** is pooled error for **individual replicate observations**. The error variance of an independent replicate mean is `sigma^2/n`; a construct-shared error component does not shrink this way. Both landscape spreads are observed label spread, not known biological signal spread. The assays also sample different levels of replication, and the sparse TIGIT duplicates do not identify noise throughout its library.

Under the classical model `Y = F + E` with uncorrelated signal and error, `R = 1 - Var(E)/Var(Y)` and `sqrt(R)` describes an oracle's Pearson correlation with noisy labels. It is not a general ceiling for rank correlation or preference accuracy. Treat the suggested **0.94/0.49 reliabilities and 0.97/0.70 correlations as conditional sensitivity calculations**, not measured task ceilings. Likewise, the approximately **0.37** hidden-construction-noise threshold assumes reallocating a fixed observed H1 variance between signal and noise; it is not an empirical bound on construction variance. H1 remains the better-supported first task from its quantitative coverage and replication, without relying on these ceiling claims.

## Backbone: ESM-IF1 selected for integration

**User decision, 2026-09-14:** use **ESM-IF1** for the first experiment, keeping the target fixed.  
**Official loader:** `esm_if1_gvp4_t16_142M_UR50`.  
**Implementation status:** not integrated; exact source/weight hashes, structural input, and measured training device remain unpinned. Model-family selection is not checkpoint promotion.

The [official implementation](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) provides an autoregressive sequence decoder conditioned on backbone geometry, with multichain scoring and sampling. It provides an existing route to structural antigen context without developing a new antigen encoder. The first experiment measures post-training on one fixed context; changing antigen context and testing specificity remain later work.

[Shanker et al., Science 2024](https://www.science.org/doi/10.1126/science.adk8946) provide experimental antibody-optimization evidence and a retrospective CR9114/H1 analysis. Their CR9114 structural input was an H5 complex, so their H1 result must not be described as conditioning on a matched H1 structure. A template used here must be identified and justified explicitly before model evaluation.

### Additional work compared with p-IgGen

- **Structural records:** pin the input structure, chain selection, residue mapping, missing-coordinate convention, and relation between the structural antigen and the measured H1 target. Start with one declared fixed geometry across variants; avoid per-variant structure generation in this initial benchmark. A fixed geometry is an approximation, not evidence that every variant adopts that conformation.
- **Isolated environment:** the upstream repository is archived, and its reference setup uses older PyTorch/PyG dependencies. These need a compatible pinned environment. CPU scoring is supported in the source; MPS operation and useful training speed require measurement. ESMFold/OpenFold are not required for inverse folding.
- **Differentiable scoring:** train against native decoder logits. The public convenience scoring helper detaches results and reports an average loss, so it is not itself a training loss. Align sampling and scoring on the same declared process before DPO.
- **Interpretation and retention:** use ESM-IF1 activation sites and sequence/structure evaluation pairs. The p-IgGen SAE study supplies methodological precedent, not transferable feature identities. The previous [p-IgGen audit](piggen-backbone-candidate.md) does not clear ESM-IF1 pretraining exposure.

The initial pilot should freeze the structure encoder and adapt the decoder in both supervised and preference arms. With fixed geometry and deterministic encoder evaluation, its representation can be cached after validating parity with the full forward pass. This is a proposed compute simplification and limits the first mechanistic claims to decoder adaptation; whole-model adaptation is not assumed.

### Readiness and promotion

The [local readiness record](evidence/esm-if1-readiness-2026-09-14.json) records package metadata and an actual failure of the existing Python environment to import PyTorch: duplicate OpenMP runtimes, exit 134. This occurred before any ESM-IF1 installation/import and is not a demonstrated ESM-IF1 failure. No weights have been loaded, model gradients tested, or training throughput measured. An isolated environment is the next engineering step.

Promotion requires pinned model/alphabet/code, a declared structural input, upstream score parity, padding and missing-coordinate checks, a successful backward pass, and measured memory/throughput on the chosen training device. Preserve the enumerable toy probability/gradient check and development-side SFT headroom gate. No custom antigen-fusion or diffusion implementation is required for this first experiment.

Use conventional supervised predictors and an AbLang2 frozen-embedding predictor as comparisons. Keep p-IgGen as an evaluated alternative; do not silently switch back if environment work is needed.

### Exact probability contract for the finite benchmark

Use ESM-IF1's native decoding order with the declared fixed geometry C; force every immutable residue. At each of the 16 editable sites, normalize the native logits over that site's two allowed residues. For a genotype y:

```text
log q_theta(y | C) = sum over editable decisions t:
    log softmax(logits_theta(prefix_t, C)[allowed_alleles_t])[chosen_allele_t]
```

All earlier forced and chosen tokens appear in the prefix. Forced residues contribute probability one in this process, not their original language-model probability. This defines a normalized constrained editing policy; it is not the original model conditioned probabilistically on every fixed residue. The complete teacher-forced sequence supports causal scoring per candidate, with fixed structural encoding. The encoder receives backbone coordinates, including selected context chains; the native multichain route does not directly encode their amino-acid identities. Keeping VL sequence fixed in the dataset is not a claim of explicit VL-sequence conditioning.

Use identical supports, direction, temperature 1, and dropout-off scoring for policy and reference. Disable nucleus sampling and post-generation rejection for the initial probability contract. Verify normalization, enumerated versus teacher-forced probabilities, and preference gradients on a tiny enumerable example. Restrict retrospective biological evaluation to measured candidates and report missing-label coverage; never assign oracle rewards to unmeasured sequences.

## Experiment order and success criteria

1. **Choose the endpoint and allowed sites; construct biological records.** Dataset work and hardware/backbone feasibility can run concurrently. The current [offline scope](../docs/OFFLINE-DPO-SCOPE.md) explicitly permits records independent of tokenization.
2. **Establish learnability on development data.** Compare mutation count, additive mutation regression, a regularized interaction model, and frozen-embedding regression. Include a conventional supervised predictor as the practical selection baseline. Evaluate pair ordering and measured top-K selection on the intended split. High measurement reliability does not guarantee a low-order interaction model generalizes, and a strong supervised fitness predictor does not establish that positive-example generative SFT is saturated. A weak zero-shot likelihood correlation alone is not a reason to reject an adaptable generator.
3. **Complete the model and toy checks, then test development-side headroom.** Run a bounded SFT pilot to establish stable learning and feasible compute. Measure the actual generative SFT baseline, its remaining pair-accuracy headroom, and uncertainty accounting for shared sequences/blocks before committing to the four-arm training budget. A +5-point gain cannot occur above 95% baseline accuracy, and sufficient numerical headroom alone does not establish statistical power. Avoid a broad backbone tournament. If ESM-IF1 fails this gate, distinguish environment/device failures from model or objective failures before reconsidering the backbone.
4. **Run the four checkpoints required by the current scope:**

   | Arm | Initialization | Frozen preference reference |
   |---|---|---|
   | Parent | P0 | — |
   | SFT | P0 | — |
   | Direct DPO | P0 | P0 |
   | SFT then DPO | SFT | SFT |

   Select SFT examples and weights using training data. Count each sequence independently of how often it occurs in pairs. Compare each DPO branch with its immediate parent, report total compute, and retain the supervised ranker/reranker as a practical baseline. Reward-weighted SFT is a useful later control if a claimed DPO advantage needs stronger testing.
5. **Freeze a confirmatory protocol before final evaluation.** Suggested primary contrast: SFT-then-DPO minus SFT on reliable held-out pair accuracy, with sequence weighting fixed in advance. Use `log q` for primary policy ordering; report the DPO implicit reward `beta * log(q / q_reference)` separately. Top-K measured affinity/enrichment within the final held-out measured candidate pool, at a fixed selection budget, is the practical secondary endpoint. Also report held-out rank correlation, absolute candidate probabilities, diversity, and probability mass on measured support. Any full-library summary that includes training sequences is descriptive only.
6. **Interpret changes at matched inputs**, then test targeted interventions on held-out examples.

A candidate engineering minimum useful effect is **+5 percentage points of pair accuracy**, with a paired uncertainty interval required to clear that floor for a confirmatory improvement claim. This is a proposal, not a literature-derived biological threshold or a preregistration already in force. Use development-side power and headroom checks to determine whether it is feasible; finalize the numerical rule before final-test inspection. If it is infeasible, record that finding and revise or stop the performance experiment before confirmatory training; do not lower the floor after inspecting final results. Strong conventional predictors make limited DPO headroom a credible risk, not an established expectation of a null result. Report training-seed variability separately and use at least three training seeds.

Split sequences before constructing pairs. For CR9114, ordinary whole-sequence identity clustering is ill-suited to a single-lineage combinatorial landscape; connecting all distance-one neighbors makes the full hypercube one component. Use a declared genotype-combination holdout, with explicit proximity and mutation-count audits. For example, reserve combinations at a sequence-only selected subset of loci, with enough held-out blocks to estimate uncertainty. This tests compositional generalization and can still have close training neighbors; do not describe it as distance-separated scaffold transfer. Account for shared variants and blocks in intervals instead of treating all preference pairs as independent.

The public dataset is already extensively studied. A sealed local split reduces adaptive model selection; it cannot make published biological knowledge unseen. A second campaign/lineage is needed for stronger replication.

## Where to put the novelty

Engage directly with the [current p-IgGen SAE study](https://arxiv.org/html/2512.05794v3). Its near-perfect multivariate probes are not the individually steered latents. TopK steering failures are plotted for selected features; the broader cross-layer/gene claim is partly unshown. Ordered features have positive dose-response evidence, and v3 adds IgLM/ProGen2-OAS experiments. Training budgets and dictionary sizes differ, so ordering alone is not isolated. Repeating the probe-versus-steering dissociation would not establish a new contribution.

Begin with a more specific question: **does successful post-training change the model's use of mutation combinations, and which model components selectively contribute to that improvement?** This is a candidate research contribution, not an asserted first in the literature. Compare against germline distance, mutation count, additive mutation effects, and parent-model likelihood. Separate two intervention claims before running them:

1. **Primary: selective necessity under a specified ablation.** Within an adapted checkpoint, targeted ablation should remove a predeclared portion of an independently established task improvement while preserving sequence-modeling performance within a separate noninferiority margin. Compare with no-op/self-patch, matched random components and intervention magnitudes, generic-damage controls, and corresponding interventions in P0. Measure retention using native per-residue conditional likelihood on both independent antibody sequence/structure pairs and held-out local genotypes under their declared structural context; forced immutable residues in the constrained policy are not evidence of retained modeling ability. Specify the task-effect floor, retention margin, and baseline for both before final testing. Require uncertainty bounds for both effects: a nonsignificant retention loss does not establish preservation. Even with these controls, necessity is conditional on the tested behavior and intervention, not proof of biological specificity.
2. **Secondary: sufficiency under injection into the parent.** Test whether inserting the proposed change into P0 reproduces the gain while meeting its own retention criterion. This is a stronger, higher-risk claim. A null injection does not erase a valid selective-ablation result; ablation alone cannot support sufficiency. Establish activation compatibility before transferring components across checkpoints.

- Compare model probability changes with additive and interaction baselines on held-out genotypes.
- Match mutation burden and inspect where the effect of one substitution changes across backgrounds.
- Compare activations at identical sequences and decision positions before and after adaptation.
- Within each checkpoint, ablate or patch candidate components with no-op, matched-donor, random-component, and collateral-damage controls.
- Use a development set to select interventions and a separate test set to confirm selective behavioral effects.

This can establish a causal contribution to the model's predictions. It does not by itself establish a physical mechanism of binding. SAEs become useful if simpler analyses cannot isolate the change; shared feature indices or cross-checkpoint patches require demonstrated representational correspondence.

A strong final result could be a repeatable supervised adaptation plus a causal account of its learned interaction behavior, with DPO characterized honestly as beneficial, equivalent within a declared margin, harmful, or inconclusive. Failure to detect a gain alone does not establish equivalence. Predeclare a development-only checkpoint-selection rule: if SFT improves over P0 but DPO adds no reliable gain, the primary mechanism study may examine SFT versus P0. Keep that claim distinct from a mechanism of DPO improvement. If no adaptation improves, do not claim to explain an improvement. Confirm the selected adaptation and intervention on untouched test data. That keeps the research objective productive without assuming an optimizer must win.

Antigen retrieval, fusion comparisons, diffusion continuation, and online reward optimization contribute little to this first fixed-target milestone. The cheapest relevant next computation is **single-target learnability with ordinary predictors**, alongside the small backbone correctness/compute pilot.

## Reproducing the fresh TIGIT counts

Input: `data/raw/open_alphaseq/YM_0693.parquet`  
SHA-256: `6b82f9bb39c835ec160ca192ef0e1ba165a0d6651c520791081e800c485c907b`.

The exploratory read used the pinned local parquet, excluded the parental amino-acid sequence and length changes, and used `affinity.notna() & above_background & sufficient_replicate_observations` separately for each target. HCDR3-only additionally required all residues outside zero-based slice `[96:114]` to equal the parent. Unique counts use the full `mata_sequence`, not pair combinations.

```python
import pandas as pd

df = pd.read_parquet("data/raw/open_alphaseq/YM_0693.parquet")
b = df[["mata_description", "mata_sequence"]].dropna().drop_duplicates("mata_description")
parents = b.loc[b.mata_description.str.startswith("wt_"), "mata_sequence"].unique()
assert len(parents) == 1
parent = parents[0]
assert len(parent) == 247 and parent[96:114] == "ARSTYYYDSSGYDYYFDP"
same = b[b.mata_sequence.str.len().eq(247) & b.mata_sequence.ne(parent)]
h3 = same[same.mata_sequence.map(lambda s: s[:96] == parent[:96] and s[114:] == parent[114:])]
for name, cohort in [("HCDR3-only", h3), ("All same-length", same)]:
    rows = df[df.mata_description.isin(cohort.mata_description)
              & df.matalpha_description.eq("TIGIT_22-137_POI-AGA2")]
    usable = rows[rows.alphaseq_affinity.notna()
                  & rows.above_background.fillna(False)
                  & rows.sufficient_replicate_observations.fillna(False)]
    print(name, len(usable), usable.mata_sequence.nunique())
# HCDR3-only 718 716
# All same-length 13041 13015
```

This research pass did not rerun the full test suite or train a model. The follow-up audit clarified Decision 0003, corrected provenance wording in the paired-cohort generator/manifest and historical memo, and reproduced the manifest with unchanged cohort counts and variant digest.
