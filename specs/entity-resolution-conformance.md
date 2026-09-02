# Entity resolution and leakage-safe evaluation — normative specification

**Status:** Level 1 normative and implemented. Levels 2 and 3 are stated as
requirements but NOT claimed by this repository; §8 says exactly which of their
clauses are met anyway and which are not.

**Implements:** `docs/ENTITY-RESOLUTION-AND-LEAKAGE-SAFE-EVALUATION.md` §13 step 1
(local-only: `docs/` is git-ignored, so that design document is **not** in the
published tree — the citation records provenance, and this file is the tracked
normative statement it asks for).
**Realised by:** `src/smallAntibodyGen/entity_resolution/`,
`src/smallAntibodyGen/target_identity.py`,
`specs/decisions/0002-typed-target-identity.md`.

Requirement levels are RFC-2119: **MUST**, **MUST NOT**, **SHOULD**, **MAY**.
Every requirement carries the artifact or test that makes it checkable. A
requirement with no checkable evidence is a wish, and this document does not
contain any.

---

## 1. Scope

This specification governs any train/validation split, any grouping of records
into entities, and any performance number quoted from such a split in this
repository. It governs the antibody-antigen corpus today and is written to govern
the OAS corpora and any future benchmark without amendment.

It does NOT govern model architecture, training schedules, or metric choice.

---

## 2. The claim plane

**2.1** Every split **MUST** declare a claim before it is evaluated against, and
that claim **MUST** use a class from the controlled vocabulary in
`entity_resolution.conformance.CLAIM_CLASSES`. Free-form leakage statements are
too easy to weaken invisibly.
*Checked by:* `Claim.__post_init__`; `test_claim_class_must_come_from_the_controlled_vocabulary`.

**2.2** A claim **MUST** declare its `unit_of_generalisation`. Two claims with the
same class and different units are different claims, and a number produced under
one **MUST NOT** be reported under the other.
*Checked by:* `validate_level_1`.

**2.3** A claim **MUST** separately enumerate `prohibited_exposure_relationships`
and `permitted_exposure_relationships`. Naming what is permitted is not a
loophole; it is what stops a permitted relationship being discovered later and
mistaken for a defect — and what stops a prohibited one hiding in the gap.

**2.4** The claim **MUST** be authoritative and the thresholds **MUST** be
declared approximations of it. When a dependence is found that the claim
prohibits and the operationalisation missed, that is a **benchmark defect**, and
it **MUST NOT** be accepted as a model behaviour on the grounds that no measure
named it.

**2.5** An operationalisation **MUST** account for every relation class in
`RELATION_CLASSES` as either assessed or unassessed. Listing only what was
checked reads as if everything was checked.
*Checked by:* `Operationalisation.__post_init__`;
`test_every_relation_class_must_be_declared_assessed_or_not`.

**2.6** A `monitor` policy row **MUST** carry an owner, a metric, a threshold, a
deadline and an escalation. Without them it is an undocumented acceptance of
leakage wearing a policy's name.
*Checked by:* `test_a_monitor_rule_without_an_escalation_is_rejected`.

**2.7** When a dependence axis is discovered that the sealed claim does not
clearly cover, the affected result **MUST** be marked uninterpretable, and the
decision about that axis **MUST** be applied only to a subsequent version. The
triggering result **MUST NOT** be rescued retroactively.

---

## 3. The construction plane

**3.1 Label blindness.** Grouping, masking, quarantine and eligibility **MUST**
consume a whitelisted record view that cannot reach outcomes, scores,
confidences, supervision, or an existing split assignment. Blindness **SHOULD**
be structural — a whitelist — rather than a convention.
*Checked by:* `row_identity_view`; `test_guard_twin_identity_is_blind_to_supervision`;
`test_row_identity_view_reads_nothing_but_the_whitelist`.

**3.2 Three relations.** A resolver **MUST** expose strict identity, broader
grouping, and quarantine as three separately computed relations. It **MUST NOT**
derive all three from one partition.
*Checked by:* `test_the_three_relations_stay_separate`.

**3.3 Containment is not identity.** Containment is asymmetric and identity is
symmetric. Local containment **MUST NOT** create an identity edge. It **SHOULD**
create a quarantine constraint.
*Checked by:* `test_outcome_2_fusion_overlap_is_a_quarantine_edge_not_an_identity`;
`test_fault_injection_promoting_containment_to_identity_fuses_igkc_into_her2`.

**3.4 Containers are not their contents.** Co-occurrence in a container — a
structure entry, an order, an archive — **MUST NOT** create an identity edge. It
**SHOULD** create a quarantine constraint.
*Checked by:* `test_shared_pdb_quarantines_without_merging`.

**3.5 Bounded clustering.** Any relation built from a similarity threshold
**MUST** use a cluster-level criterion — complete linkage, a maximum diameter, or
a representative every member must match. Connected components over thresholded
edges is single-linkage and **MUST NOT** be used for an identity relation.
*Checked by:* `agglomerate_complete_linkage`;
`test_bounded_criterion_refuses_the_closing_edge` paired with
`test_fault_injection_single_linkage_percolates_the_same_chain`.

**3.6 Component reporting.** Every group **MUST** be able to report its members,
a representative, its minimum pairwise similarity, its minimum pairwise coverage,
and its maximum diameter. A component that cannot state its own diameter **MUST**
be assumed to have percolated.
*Checked by:* `ClusterReport`; `test_component_reports_its_own_minimum_identity_and_coverage`.

**3.7 Two kinds of minimum.** Where hard curated evidence (an accession, an
approved name) merges a group, the similarity criterion's diameter guarantee does
not apply, and the report **MUST** distinguish the two rather than publishing one
blended number.
*Checked by:* `criterion_min_pairwise_identity` vs
`component_min_pairwise_identity` in `stats()`.

**3.8 Identity and coverage are separate.** A similarity relation **MUST** be
expressed as identity AND coverage AND an absolute overlap floor, never as
identity alone. Coverage **MUST** be span-over-length; defining it as
matched-over-length multiplies the two together and makes a truncation
indistinguishable from a divergence.
*Checked by:* `PairAlignment`;
`test_local_alignment_reads_a_truncation_as_full_identity_over_partial_coverage`.

**3.9 Names.** A name that spans a coherent set of groups **SHOULD** merge them.
A name that spans an incoherent set **MUST NOT** merge them and **SHOULD**
quarantine them. The decision **MUST** be computed once against a frozen
partition and **MUST NOT** be iterated to a fixed point. Every name considered
**MUST** produce a recorded decision with a reason.
*Checked by:* `NameDecision`; `test_name_approval_is_decided_once_and_recorded`;
`test_name_approval_is_a_decision_with_a_reason`.

**3.10 Conflicting identifiers.** When sources disagree about an entity's
identifier, the resolver **MUST** report the disagreement and **MUST NOT** settle
it. The group **MUST** receive a neutral deterministic id rather than whichever
identifier sorts first, ranks highest, or carries more rows — all three are the
same failure.
*Checked by:* `AccessionConflict`;
`test_outcome_6_conflicting_accessions_are_not_silently_resolved`.

**3.11 Quarantine closure.** Hard co-assignment constraints **MUST** be
transitively closed, because a constraint that is not closed does not constrain.
Closure **MUST NOT** be routed through a container or a high-degree bridge;
those **SHOULD** be marked test-ineligible instead. Aggregate or container
evidence **MUST** produce test-ineligibility, never a merge.
*Checked by:* `_close_quarantine`;
`test_fault_injection_a_composite_that_bridges_welds_two_families`.

**3.11.1** Every constraint the previous clause declines to enforce **MUST** be
counted and reportable as residual exposure. Refusing to close through a
container is the right trade and it is not free: the constraint stops
constraining, so the split may separate a construct from the container that
contains it byte-for-byte. A producer resolves each one by EXCLUDING the
container, which the policy table allows and which costs only that container's
own rows.
*Checked by:* `unclosed_constraints()`;
`test_refusing_to_close_through_a_container_is_counted_as_residual_exposure`.

**3.11.2** The closure itself **MUST** be bounded, separately from any single
container's degree. A chain of containers each spanning two families is legal at
any span bound and closes into one group. An over-large closure **MUST** keep its
constraints and **MUST** stop being scored, because percolation there costs
evaluation diversity rather than correctness.
*Checked by:* `test_a_percolated_quarantine_closure_stops_being_scored`.

**3.9.1** An approved name **MUST** merge through the same bounded criterion as
any other similarity relation. Freezing the verdicts is necessary and not
sufficient: union is transitive, so a ladder of individually correct name merges
percolates even when every verdict is right.
*Checked by:* `test_a_ladder_of_individually_valid_names_does_not_percolate`.

**3.12 Blocking recall is correctness.** Candidate generation **MUST** be treated
as part of the resolver's correctness contract. Its filters **SHOULD** be derived
necessary conditions rather than heuristics, and its recall **MUST** be measured
against an independent route, not asserted. The measurement **MUST** be made at
the configuration production runs, and a conformance validator **MUST** read it:
a derivation nobody checked is not evidence.
*Checked by:* `blocking_recall_report`;
`test_blocking_recall_is_total_against_exhaustive_all_pairs`;
`test_a_missing_blocking_recall_measurement_fails_the_level`;
`test_a_lost_qualifying_pair_fails_the_level`.

**3.15 A derived filter's boundary is inclusive.** A filter derived as an
inequality with equality allowed **MUST NOT** be implemented as a strict
comparison against a floating-point product. The family band floor evaluates to
0.7200000000000001 rather than 0.72, and rejecting a pair at exactly the floor is
silent leakage introduced by arithmetic.
*Checked by:* the exact-floor case in
`test_blocking_recall_is_total_against_exhaustive_all_pairs`'s population.

**3.16 A vacuous filter admits, it does not drop.** When a derived filter's bound
is non-positive the filter is vacuous, and a vacuous filter **MUST** admit every
candidate the other filters allow. Enumerating only the partners that happen to
share evidence with this one is a silent drop wearing a disabled filter's name,
and enumeration **MUST NOT** depend on the sort order of the two keys.

**3.13 No silent caps.** A comparison that was not made **MUST** be counted and
reported. Refusing an oversized pair is acceptable; skipping one silently is not.
*Checked by:* `AlignmentTooLarge`, `target_alignments_refused_too_large`;
`test_oversized_pairs_are_refused_rather_than_skipped`.

**3.14 Determinism.** Group ids **MUST** be a function of the population, not of
arrival order. Tie-breaks **MUST** be pinned on immutable keys.
*Checked by:* `test_resolution_does_not_depend_on_row_order`;
`test_merge_order_is_pinned_under_ties`.

---

## 4. Calibration and audit

**4.1** Thresholds **MUST NOT** be validated on the population they were selected
on. Calibration and audit populations **MUST** be disjoint.
*Checked by:* `test_threshold_selection_is_audited_on_families_it_never_saw`.

**4.2** The split between them **MUST** be by family, not by pair, so that no
family can contribute a pair to selection and another to confirmation.

**4.3** An error report **MUST** state false merges and false splits separately,
because they cost different things: a false merge destroys diversity and is
invisible to a leakage audit, a false split costs leakage and is not.
*Checked by:* `ErrorReport`;
`test_both_error_kinds_are_reported_against_the_predeclared_asymmetry`.

**4.4** The tolerated error asymmetry **MUST** be declared before any measurement
and carried on every report.

**4.5** An error report over zero pairs, or with no positive pairs, or with no
negative pairs, **MUST NOT** be quoted: one of its two counts is free.
*Checked by:* `validate_level_1`; `test_an_audit_over_zero_pairs_is_not_evidence`.

**4.6** A synthetic calibration generator **MUST NOT** reuse the resolver's
helpers. An oracle that shares the implementation it scores measures nothing.
*Checked by:* `test_synthetic_generator_is_seeded_and_independent`.

---

## 5. Testing

**5.1** Every fixture **MUST** prove its own premises. An anti-percolation test
**MUST** positively assert that its chain edges are admitted at the operating
thresholds before asserting that the endpoints stay apart.
*Checked by:* `test_outcome_4_fixture_has_power_at_the_shipped_operating_point`.

**5.2** Every criterion **MUST** have a fault injection proving the corresponding
test fails when the criterion is removed. A test that survives the fault is
testing nothing. This applies to the percolation cuts as much as to the cluster
criterion: an adversarial pass found the whole suite green with BOTH cuts
deleted, which meant they were asserted and not tested.
*Checked by:* the `test_fault_injection_*` family, including
`test_fault_injection_closing_quarantine_through_a_container_welds_two_families`
and `test_fault_injection_disabling_the_container_span_guard_changes_the_answer`.

**5.5** A rule stated as "decided once against a frozen partition" **MUST** be
tested against the thing that breaks it, which is ORDER. Judging names in sorted
order while merging as you go makes the verdict depend on how the names are
spelled.
*Checked by:* `test_name_verdicts_do_not_depend_on_what_other_names_are_called`.

**5.3** A fixture's recorded measurements **MUST** be expressed in the metric the
operating point uses. Quoting a number measured with one aligner against a
threshold expressed in another is how a fixture becomes powerless without anyone
noticing.

**5.4** Three kinds of evidence **SHOULD** all be present: real fixtures for
concrete semantic outcomes, a synthetic planted corpus for algorithm behaviour,
and an independent natural population for real-world error rates. This repository
has the first two; the third is **not** met and is named as a gap in §8.

---

## 6. The release plane

**6.1** Every parameter that can influence eligibility, preprocessing, assignment
or interpretation is a benchmark parameter and **MUST** be sealed before the
first adaptive evaluation.

**6.2** The exact record-to-side assignment **MUST** be sealed, not merely enough
parameters to reconstruct it. Reconstruction requires the code and the corpus to
stay byte-identical forever; recording does not.
*Checked by:* `test_the_split_manifest_seals_the_assignment_itself`.

**6.3** A manifest **MUST** bind its provenance. A hash over the body alone lets
the inputs change while the manifest still verifies.
*Checked by:* `test_a_claim_manifest_without_provenance_seals_nothing`.

**6.4** A split manifest **MUST** name the claim manifest it serves. A split that
does not name its claim can be quoted under any claim.
*Checked by:* `test_a_split_not_bound_to_its_claim_is_rejected`.

**6.5** The achieved side ratio AND the largest group's share **MUST** both be
reported. Hitting a target ratio while one group holds most of the corpus is not
what the ratio suggests.

**6.5.1** The concentration share **MUST** be measured in SOURCE ROWS, not in
distinct records. One target carries 63.4% of this corpus's rows while being one
antigen among 9,574, so a share counted in records is blind to the failure the
measure exists for.

**6.5.2** A bound on it **MUST** be predeclared in the claim manifest before the
numbers exist, and exceeding it **MUST** fail the level rather than produce a
note. A concentration figure nobody can fail is a statistic, not a gate.
*Checked by:* `test_a_dominant_group_fails_the_level_even_when_the_ratio_is_hit`;
`test_concentration_without_a_predeclared_bound_fails`;
`test_concentration_is_gated_in_rows_not_records`.

**6.6** A published test snapshot is closed. New records **MUST NOT** enlarge an
existing test version. A new record linking to test only **MUST** be excluded from
train; one linking to both train and test **MUST** open an incident.
*Status:* stated normatively; no admission pipeline exists here yet (§8).

---

## 7. Results and validity

**7.1** A result's identity is at least the test version, the training snapshot
and the exposure state. Two results over the same test with different train or
exposure states **MUST NOT** be presented as a controlled comparison.

**7.2** Every number **MUST** carry a standard leakage line generated from
controlled machine-readable fields, never written by hand.
*Checked by:* `LeakageLine`; `test_the_leakage_line_renders_every_controlled_field`.

**7.3** The mapping from a missing conditioning channel to a weaker standard
claim **MUST** be sealed in advance. A channel with no sealed mapping makes the
result uninterpretable; it **MUST NOT** be resolved by argument after the numbers
are known.
*Checked by:* `CLAIM_REDUCTION`, `reduce_claim`;
`test_the_reducer_moves_only_downward_and_never_invents_a_rescue`.

**7.4** A leakage line **MUST NOT** claim more than the sealed reducer supports.
*Checked by:* `test_a_leakage_line_stronger_than_the_reducer_allows_is_rejected`.

**7.5** Status **MUST NOT** be collapsed into one word. Execution integrity,
domain membership, evidentiary status, currency, disclosure and conformance are
orthogonal. Unknown **MUST NOT** silently become inside.
*Status:* partially met — `ConformanceAttestation` separates level, supported
claim, uninterpretable channels and failures, but there is no append-only
validity ledger (§8).

---

## 8. What this repository does and does not meet

### Met — Level 1
Every requirement in §2 through §5, and §6.1–§6.5, §7.2–§7.4.

### Deliberately not met, with reasons

| Requirement | Why not |
|---|---|
| §5.4 independent natural confirmation | Needs a second, independently curated antigen corpus. None is available; the audit population is 16 sequences and cannot estimate a real-world error rate. |
| §6.6 closed-snapshot admission | No test version has been sealed, so there is nothing yet to admit against. Required before any published version. |
| §7.1 exposure state in result identity | The antigen encoder's upstream pretraining is opaque; the strongest honest claim is "no detected exposure across declared static channels", which the reducer already emits. |
| §7.5 validity ledger | Level 3. Meaningful only with an external audience and an adaptive reuse budget, neither of which exists here. |
| Signed or timestamped seals | Level 2. The seals here are tamper-evident against accidental drift, which is what a single-operator repository needs. |
| Feedback budgets, disclosure logs, canaries, an independent steward | Level 3, and all presuppose a published benchmark with participants. Adding the vocabulary without the audience would be ceremony. |

### The gap that matters most

The shipped antibody-antigen split groups on the **antigen**. It does not hold
out the **antibody**, and measured on the corpus 78.6% of stage-4 validation rows
have their HCDR3 in train. That is a `permitted_exposure_relationship` under the
declared claim, so it is not a violation of this specification — and it is
exactly the kind of thing §2.3 exists to force into the open rather than let sit
in a gap. Any claim about generalisation to unseen antibodies needs a different
`unit_of_generalisation` and a different split, and **MUST NOT** be supported by
numbers from this one.
