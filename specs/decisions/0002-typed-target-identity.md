# ADR 0002 — Typed target identity and the Level-1 leakage-safe evaluation path

**Date:** 2026-09-01
**Status:** Accepted, opt-in. The producer's default is unchanged.
**Supersedes:** nothing. **Superseded by:** nothing.
**Implements:** `docs/ENTITY-RESOLUTION-AND-LEAKAGE-SAFE-EVALUATION.md`, Level 1 of
its conformance ladder, plus steps 2, 3, 4, 5, 6 and 9 of its §13.

---

## 1. Context

`docs/ENTITY-RESOLUTION-AND-LEAKAGE-SAFE-EVALUATION.md` is a subject-stripped
design for entity resolution and leakage-safe evaluation. Its domain-specific
half already existed in this repository as an **executable acceptance contract
with the implementation deliberately absent** (`e4a9b5e`): sixteen real antigen
records, six pinned outcomes, 14 passing fixture-power and guard tests, and 12
`xfail(strict=True)` specifications. Two previous implementations were written
against that contract and both were reverted.

This ADR records what was built, what was deliberately not built, and the two
findings that forced a change to the contract itself.

## 2. Decision

Build the **Level-1 reference path** and the typed-identity engine underneath it.
Do **not** change what the producer does by default.

### 2.1 What was built

| Module | Job |
|---|---|
| `entity_resolution/alignment.py` | Local affine Smith-Waterman, BLOSUM62, open 11 extend 1, vectorised, with a naive oracle beside it |
| `entity_resolution/blocking.py` | Candidate generation with a *derived* recall guarantee, measured against exhaustive all-pairs |
| `entity_resolution/clustering.py` | Bounded (complete-linkage) clustering, plus the single-linkage it replaces, exported for fault injection |
| `entity_resolution/synthetic.py` | An independently implemented planted corpus — the calibration population |
| `entity_resolution/conformance.py` | The claim plane, the four Level-1 artifacts, the sealed claim reducer, and the validator |
| `target_identity.py` | The three relations and the evidence-to-action table |
| `target_identity_labels.py` | Curated ground truth for the audit population |
| `scripts/resolve_target_identity.py` | The shadow build: run over the corpus, emit all artifacts, validate |
| `scripts/scan_anti_percolation_triples.py` | Find real chains that are open at a given operating point |

### 2.2 The three relations

Deriving all three from one graph is the failure the contract exists to prevent.

- **`construct_id`** — the same concrete construct. Identity ≥ 0.99, minimum
  coverage ≥ 0.95, overlap ≥ 30, clustered with a bounded criterion.
- **`biological_target_id`** — the same target. Shared UniProt accession, an
  approved name, or identity ≥ 0.90 / coverage ≥ 0.80 / overlap ≥ 30.
- **`quarantine_partners`** — must share a side without being the same thing:
  local containment, a shared PDB entry, an unapproved name. Direct partners
  only; the transitive closure is `split_group_id`'s job.

### 2.3 The operating point, and why these numbers

The construct threshold is set by a real constraint, not a preference. The 564-aa
HER2 ectodomain is a **byte-exact sub-region** of the 607-aa one — local identity
1.0000 — and they must be different constructs. So the separator has to be
coverage (0.9292 for that pair), not identity. Conversely the 287/289-aa Omicron
NTD pair, which differs by a two-residue interior insertion, must be one
construct: identity 0.9931, coverage 1.0000.

The family threshold is set by the influenza-B haemagglutinins, which are one
target across three UniProt accessions at 0.9193–0.9797 identity, and by IL23A vs
IL12B at 0.3462, which are two.

**Predeclared error asymmetry:** tolerate false splits over false merges. A false
merge destroys target diversity, corrupts every per-target number, and is
invisible to a leakage audit because a merged pair never straddles anything. A
false split costs leakage, which an audit does detect. This is recorded on every
error report so it cannot be chosen after the numbers are known.

### 2.4 The percolation cut

A previous attempt percolated to a **77% component on the relation the split
actually keys on**. Nothing here PREVENTS percolation absolutely, and claiming
otherwise would be the same mistake one level up: a chain of individually legal
constraints can still close into one large group. Two mechanisms make it much
less likely and a third makes it impossible to miss:

1. Similarity relations use complete-linkage, so cluster diameter is bounded by
   the threshold rather than by the longest chain through the data.
2. A **concatenated construct** (a poly-glycine linker run) is marked
   test-ineligible and its quarantine edges are recorded but not closed through.
   The 1057-aa Fab-plus-HER2 fusion contains both the 107-aa IGKC domain and the
   607-aa HER2 ectodomain; closing through it would weld two unrelated families.
3. A container spanning more than `max_container_span` (default 8) families is a
   high-degree bridge: it marks its members test-ineligible instead of linking
   them, and the refusal is published.
4. **Detection, which is the one that actually guarantees something.** The claim
   manifest predeclares `max_split_group_row_share` before any number exists,
   and `validate_level_1` FAILS the level when the largest split group exceeds
   it. Measured in ROWS, not records: one target carries 63.4% of this corpus's
   rows while being one antigen among 9,574, so a share counted in antigens is
   blind to exactly this failure. This is what the design document means by
   "if a huge component remains after defensible preprocessing, the intended
   held-out claim may genuinely be unsupported".

Each of 1, 2 and 3 has a fault-injection test that fails when the mechanism is
removed. An adversarial review pass found that an earlier version of 2 and 3 had
no such test -- the whole suite stayed green with both cuts deleted -- and that
a refused bridge was neither linked NOR flagged, leaving its members with no
constraint at all. Both are fixed and both now have the injection.

## 3. Two findings that changed the contract

### 3.1 The contract as committed was unsatisfiable as a whole

`test_fixture_subset_reproduces_the_corpus_verdict` pinned, for every fixture
record, the `canonical_target_id` the committed rule gives it. For outcome 3 that
is `uniprot:p29460` for **both** IL23A and IL12B. And
`test_outcome_3_4grw_entities_remain_distinct_targets` required those two ids to
**differ**. Both ran against the same `committed` fixture. They cannot both hold.

The same contradiction exists for outcome 5 (`wt_rbd` and `wt_s2` both pinned to
`pdb:7ch5` and required to differ) and outcome 6 (`rac1` pinned to
`uniprot:p63000` and required not to be it).

So the question was never whether to edit the contract, but which half to keep.
**Kept:** the corpus-fidelity test, because what the rule that built the shipped
corpus did to these sixteen records stays true forever, and it is the evidence
behind every "VIOLATED TODAY" verdict in the fixture module. **Moved:** the five
specifications, to the engine — which is what the file's own instruction to
"re-point them" always meant. The three regression guards still run against the
committed rule, untouched, and `test_target_identity_engine.py` carries
engine-side twins of them.

### 3.2 The outcome-4 fixture really was powerless, confirmed by measurement

The contract shipped outcome 4 with a warning that it passed with the
anti-percolation criterion deleted. That was re-measured in the engine's own
metric before the repair was designed, and confirmed:

```
influenza-B haemagglutinins at construct (0.99 / 0.95 / 30)
    A~B  identity 0.9797  min-coverage 0.9583   no edge
    B~C  identity 0.9309  min-coverage 1.0000   no edge
    A~C  identity 0.9193  min-coverage 0.9586   no edge
    complete-linkage and single-linkage agree; merges_refused == 0
```

The repair followed the warning's own instruction: **scan, do not invent, and do
not move the operating point until the old fixture works.**
`scripts/scan_anti_percolation_triples.py` aligned every candidate pair among all
9,574 distinct antigen sequences and found **6,412** open triples. The one chosen
is three SARS-CoV-2 RBD constructs with different purification tags:

```
A  199 aa, PDB 7E3O                  B  205 aa, PDB 7KMG/7KMH/7KMI/7MMO
C  209 aa, name sars_cov2_rbd, 67,058 shard rows

A~B  identity 1.000000  min-coverage 0.960976  ADMITTED
B~C  identity 1.000000  min-coverage 0.952153  ADMITTED
A~C  identity 1.000000  min-coverage 0.913876  REFUSED
```

Every identity is exactly 1.0, so **no identity threshold could ever separate
these three**. The chain is closed in identity and open only in coverage, over
(0.913876, 0.952153] — 3.83 points, with the shipped 0.95 inside it. Single
linkage over the two admitted edges merges all three; the bounded criterion
refuses, and records `merges_refused == 1`. Exact containment fails on all three
pairs because the tags differ, the same mechanism that defeated a previous
attempt on the Omicron pair.

The old fixture is left exactly where it is — it is real and its
Needleman-Wunsch relations are still true of its sequences.

## 4. A recorded claim I could not reproduce, and one I confirmed

The reinstated xfail markers said the reverted attempt's "alignment primitive was
not a correct affine Smith-Waterman". A survey pass reported it could not
reproduce a discrepancy. Running the recovered bytecode against this
implementation and against its own naive oracle, **the claim is confirmed --
but on evidence that is not reproducible from this repository.** The recovered
engine exists only as an untracked `.pyc` in a local `__pycache__`, with no
source; anyone checking this out will not have it, and a `git clean` removes it.
Treat what follows as an observation made once, recorded because it is the only
corroboration the recorded claim will ever get: on
`wt_ntd ~ wt_s2` the recovered engine reports 88 alignment columns at identity
0.3068, while both the vectorised and the naive implementations here agree on 24
columns at identity 0.4583 with a strictly higher score. The recovered engine was
not reporting the optimal-score local alignment on that pair.

The other recorded claim — 77% percolation — is **not re-measured here**, because
the code that produced it no longer exists. It is treated as a risk to guard
against rather than as a number to reproduce.

## 5. What was deliberately NOT built

- **Producer adoption.** `scripts/prepare_antibody_antigen.py` is UNEDITED. No
  flag was added to it and no code path through it reaches the engine, so the
  shipped corpus, every stored `canonical_target_id`, and every existing
  checkpoint remain valid. The cost of adoption is concrete and worth stating
  before anyone pays it: `scripts/size_asd_cohorts.py` binds the SHA-256 of the
  producer into its `provenance_key`, so **any** byte change to that file makes
  all **3,176** sizing checkpoints under `outputs/asd-sizing-checkpoints/`
  unresumable, and adopting the typed split means a new corpus generation plus
  retraining stages 3 and 4. The design document's own §13 puts a shadow build
  (step 9) before sealing a publishable version (step 10);
  `scripts/resolve_target_identity.py` is that shadow build.
- **The antibody axis.** This split groups on the antigen. The same antibody may
  appear on both sides, and measured on the shipped corpus it usually does —
  78.6% of stage-4 validation rows have their HCDR3 in train. That is stated in
  the claim's `permitted_exposure_relationships` and in the leakage line rather
  than being quietly absent. Holding out antibodies needs a different
  `unit_of_generalisation` and is a separate decision.
- **Levels 2 and 3.** Tamper-*evident* sealing is present; signed or externally
  timestamped seals, feedback budgets, disclosure logs, reusable-holdout
  mechanisms, model-side canaries and an independent steward are not. Most of
  Level 3 governs a *published, adaptively reused* benchmark with an external
  audience, which this is not. Adding the vocabulary without the audience would
  be ceremony.
- **Semantic, structural and aggregate-mosaic relations.** Declared unassessed in
  the operationalisation, not silently omitted. Two antigens can share an epitope
  surface and no residues, and nothing here would see it.

## 6. Consequences

- The full suite goes from `1206 passed, 12 xfailed` to all-passing with no
  xfails, plus the new tests. No previously passing test was weakened.
- Every anti-percolation assertion is paired with a fault injection that proves
  it fails without the criterion, including the two that an adversarial pass
  showed were previously unguarded. `single_linkage` is exported for exactly
  this.
- The calibration population is synthetic and shares no code with the resolver;
  the audit population is the five curated real families, and the two are
  disjoint by construction. **This does not mean the audit families set no
  thresholds.** They did: §2.3 above derives the construct coverage threshold
  from the HER2 truncation pair and the family threshold from the influenza-B
  haemagglutinins, both of which are audit families. So the audit report is a
  consistency check on records the thresholds were fitted to, and the only
  population that satisfies requirement A in full is the synthetic one. An
  honest reading is: the operating point is *calibrated* on planted data and
  *pinned* by the contract's six outcomes, and a genuinely held-out real
  population does not exist here. §8 of
  `specs/entity-resolution-conformance.md` lists that as a gap rather than
  hiding it.
- Two component-health numbers are reported rather than one:
  `criterion_min_pairwise_identity`, where the bounded criterion decided and its
  guarantee holds, and `component_min_pairwise_identity` over all components,
  which includes families joined by a curator's accession and therefore carries
  no diameter guarantee. Blending them would either understate or overstate the
  guarantee.

## 6b. What an adversarial review found, and what it changed

The engine was put through a five-lens adversarial review with every finding sent
to an independent refuter. 51 verification passes ran; most findings were
refuted. These survived, and every one of them changed the code or the words:

**Correctness, in the engine**

1. **E7 name approval was iterative.** `force_union` ran inside the name loop, so
   each name was judged against a partition the previous name had already
   changed — and the coherence check skips pairs already in one group, so an
   earlier merge could excuse a later name's worst pair. Renaming a curator label
   from `a_acd` to `z_acd`, with no other change, produced a different split.
   Every verdict is now taken against a partition frozen before the loop.
2. **Freezing the verdicts was not enough.** Union is transitive, so nine
   individually correct name merges still percolated a ten-rung ladder into one
   family whose ends shared 22.67% identity. Approved names now go through
   `agglomerate_complete_linkage` like every other similarity relation, and the
   name bridge gained a coverage floor — without one, a 120-residue shared motif
   at 25% reciprocal coverage bought a merge the family relation explicitly
   refused at 0.80.
3. **A refused high-degree bridge was silently dropped.** It was neither linked
   nor flagged, so its members ended up with no constraint of any kind — the
   opposite of the conservative behaviour the refusal exists for, and contrary to
   what three documents said it did.
4. **`max_container_span` cannot bound the closure.** It bounds one container's
   degree; a chain of containers each spanning two families is legal at any span
   bound and closes into one group. An over-large closure now keeps its
   constraints and marks every construct in it test-ineligible, which is what
   percolation actually costs: evaluation diversity, not correctness.
5. **The length band rejected pairs at exactly the floor.** `0.9 * 0.8` is
   `0.7200000000000001`, one ulp above the derived 0.72, and the derivation is an
   inequality with equality allowed.
6. **A vacuous k-mer floor dropped pairs instead of admitting them,** and sub-k
   sequences were enumerated in one direction only, so whether a pair was
   proposed depended on the sort order of its keys.
7. **`row_identity_view` could not read a processed record.** The producer
   renames the field to `sequence_antigen` on the way out, so every processed
   record would have resolved to an empty antigen.
8. **`identity` and `cov_*` were properties of the traceback tie-break, not of
   the pair** — measured at 24 of 400 real corpus pairs. The orientation is now
   canonicalised in both the implementation and its oracle.

**Tests that could not fail**

9. The label-blindness test wrote ONE constant onto every row, so a leaked field
   carried no information and could not change any partition. It now varies per
   row and per pass, and checks the reports as well as the ids.
10. **E7 and E5 were both dead under test**: deleting either force left the suite
    green. Both now have populations where removing them changes the answer.
11. Both percolation cuts could be deleted with the suite green. Both now have
    fault injections.
12. `assert report.false_merges is not None` on a non-`Optional[int]` is a
    tautology — it passed with 99,999 errors in either column.
13. `conftest.py` inserted `src/smallAntibodyGen/src` on `sys.path`, which does
    not exist, so the suite silently resolved through the editable install.
14. The symmetry test used only mutated copies of one sequence, where ties are
    rare; it now spends half its pairs on unrelated sequences, where they live.
15. The ambiguity-code test covered only `J`, `O` and `U` — the three that cannot
    fail. `B`, `Z` and `*` do score as self-matches, and now say so.

**Claims that were wrong**

16. `--target-identity typed`, cited in four places as the opt-in mechanism, does
    not exist. The producer is unedited and the engine is reached through
    `scripts/resolve_target_identity.py`.
17. The alignment docstring's cost model was 10x optimistic and cited a test file
    that does not exist.
18. The outcome-4 fixture's stated selection rationale was false on the scan's
    own metrics: the scan ranks by identity window, and the triple was chosen on
    the coverage window. Both the fixture and this ADR now say so.
19. The 199-mer carries an 8x His tag, not 7x — and that number IS the fixture's
    arithmetic, since 199 − 8 = 191 is where the 0.9139 closing coverage comes
    from.
20. §6's claim that no family set a threshold and then confirmed it was wrong;
    §2.3 derives two thresholds from audit families. Corrected above.

## 7. Cost, and the one measurement still outstanding

The shadow build is expensive, and the number belongs here rather than in
somebody's surprised console. At the shipped operating point the similarity pass
proposes **450,381** candidate pairs over the 9,574 distinct antigens; a typical
pair (~420 × 430) costs about **25 ms**, so a full run is **3–4 hours** of CPU.
The cost is per-row numpy call overhead, not cells — 7.4M DP cells/second.

Two cuts were taken and one was deliberately not:

- **Taken.** The overlap floor is now a third derived blocking filter. Three
  antigens in the corpus are shorter than the k-mer width, so they carry no k-mer
  filter, and in a containment search they carry no length band either — between
  them they were contributing 28,719 candidates that provably cannot meet a
  30-residue overlap floor.
- **Taken.** Containment is measured only across families. It proposed 450,381
  pairs of its own, nearly all of them two constructs of one family that
  genuinely contain each other — a constraint the family relation already
  imposes, since the split closes over families. The count skipped for that
  reason is reported, not dropped.
- **Not taken.** Aligning lazily inside the agglomeration would avoid a large
  fraction of the remaining work, because most within-cluster edges turn out
  redundant. It changes the clustering loop, and changing the clustering loop
  under time pressure is how the previous two attempts failed. It is the right
  next optimisation and it needs its own tests.

### 7.1 The percolation measurement, on the shipped code

A deterministic 3,000-antigen prefix (1,189,709 shard rows), run against the code
as landed, 3,516 s, 108,702 alignments:

```
constructs                     2,925      families                     2,638
split groups                   2,496      test-ineligible constructs      24
construct merges refused          22      family merges refused          334
accession conflicts               13      high-degree bridges refused      1

largest split group      1.73% of antigens, 31 constructs, 44.27% of ROWS
largest family                                            44.13% of ROWS
group size distribution   p50 1, p90 1, max 52
achieved val fraction     9.77% against a 10% target
conformance               Level 1, no failures
```

**The 77% failure does not reproduce.** The number that matters is the gap
between the last two rows of the block above: the largest *family* already holds
44.13% of rows because one antigen family genuinely dominates this corpus, and
the quarantine closure on top of it adds **0.146 percentage points**. The
relation the split keys on is doing almost nothing to concentrate it. The median
split group holds one construct and the largest holds 31.

The bounded criterion's guarantee holds where it decided:
`criterion_min_pairwise_identity` is 0.9022 over 188 criterion-built components,
against a 0.90 family threshold. Eleven components were curator-merged and carry
no such guarantee, which is why the two numbers are reported separately.

Three caveats belong with these figures rather than in a footnote:

- A subsample UNDER-estimates percolation. A dirty result here would have been
  conclusive; a clean one is only suggestive. The full 9,574-antigen run is 3–4
  hours and remains outstanding.
- The blocking-recall audit on this run found **4** qualifying pairs in a
  200-sequence exhaustive sample. Recall was 1.0 over them, and 1.0 over four
  pairs is thin evidence. The validator refuses zero, which is the floor, not a
  standard; the audit sample should grow until it holds a useful number of
  positives.
- The audit population on a 3,000-antigen prefix contains only part of the
  curated label set — 43 scored pairs, 1 positive. The 16-record run in the test
  suite is the one with real audit power (116 pairs, 7 positive, 0 errors).

## 8. How to check this

```bash
python -m pytest src/smallAntibodyGen/tests/test_target_identity_acceptance.py \
                 src/smallAntibodyGen/tests/test_target_identity_engine.py \
                 src/smallAntibodyGen/tests/test_entity_resolution_primitives.py \
                 src/smallAntibodyGen/tests/test_entity_resolution_conformance.py -q

python scripts/resolve_target_identity.py --name asd-typed-3k --max-antigens 3000
python scripts/scan_anti_percolation_triples.py --identity 0.99 --coverage 0.95 \
       --output outputs/anti-percolation-triples-construct.json
```

The second prints the standard leakage line and the component-health figures, and
writes the five artifacts. It reads the raw shards and writes no corpus.
