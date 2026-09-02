"""
Engine-side tests for `smallAntibodyGen.target_identity`.

`test_target_identity_acceptance.py` is the contract: six pinned outcomes over
sixteen real records, and it says what the engine must achieve. This file says
how the engine must behave -- the properties the contract does not pin because
they are about mechanism rather than outcome.

Four kinds of test live here:

1. **Engine-side twins of the acceptance regression guards.** Those guards run
   against the committed rule, which this change leaves alone. Without twins,
   the engine could regress on exactly the behaviours the guards protect and
   nothing would notice.
2. **Fault injections.** For every anti-percolation claim, the corresponding
   fault, proving the claim's test would fail without the criterion.
3. **Determinism and blindness.** Row order, annotation order, and the
   structural unreachability of supervision.
4. **Non-vacuity.** The calibration and audit populations are checked for having
   any content at all, because an error report over zero pairs is a clean
   scorecard that measured nothing.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from smallAntibodyGen import target_identity as ti
from smallAntibodyGen import target_identity_labels as labels
from smallAntibodyGen.entity_resolution import synthetic
from smallAntibodyGen.entity_resolution.clustering import single_linkage


def _load_sibling(name: str):
    """Import a fixture module sitting next to this file, by path."""
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"engine_tests_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_sibling("fixtures_target_identity")
ap = _load_sibling("fixtures_anti_percolation")


@pytest.fixture(scope="module")
def resolved():
    """The engine over all sixteen acceptance fixture records."""
    return ti.resolve_target_identity(
        [row for record in fx.ALL_RECORDS for row in record.rows()]
    )


@pytest.fixture(scope="module")
def chain():
    """The engine over the re-sourced anti-percolation chain."""
    return ti.resolve_target_identity(
        [row for record in ap.CHAIN_RECORDS for row in record.rows()]
    )


# =========================================================================== #
# 1. Engine-side twins of the acceptance regression guards
# =========================================================================== #

def test_guard_twin_igkc_is_not_her2(resolved):
    """The guard the previous attempt broke, restated against the engine.

    That attempt added an exact-substring containment edge and fused the human
    IGKC constant domain (P01834) with HER2, because one shard record is a
    concatenated Fab-plus-HER2 construct and the 107-aa IGKC antigen sits inside
    it. The committed rule keeps them apart only by having no containment
    relation at all. The engine has one, and must still keep them apart.
    """
    case = fx.CASES_BY_OUTCOME[2]
    igkc, her2 = case.record("igkc"), case.record("her2_ecd")
    assert resolved.construct_id(igkc) != resolved.construct_id(her2)
    assert resolved.biological_target_id(igkc) != resolved.biological_target_id(her2)
    assert resolved.biological_target_id(igkc) == "uniprot:p01834"
    assert resolved.split_group_id(igkc) != resolved.split_group_id(her2), (
        "the fusion is a container and must not weld its two components into "
        "one split group -- that is the percolation this design refuses"
    )


def test_guard_twin_omicron_pair_shares_one_split(resolved):
    """The Omicron pair must stay together, and now for the right reason.

    Under the committed rule they share a split only because a curator wrote one
    name on both; delete the name and the family dissolves. Under the engine they
    are held by a 99.31%-identity sequence relation, which is why
    `test_outcome_1_family_survives_deleting_the_name` can strip the name and
    still get the same answer.
    """
    case = fx.CASES_BY_OUTCOME[1]
    short, long = case.record("omicron_ntd_287"), case.record("omicron_ntd_289")
    assert resolved.construct_id(short) == resolved.construct_id(long)
    assert resolved.split_group_id(short) == resolved.split_group_id(long)


def test_guard_twin_identity_is_blind_to_supervision():
    """Contract requirement C against the engine, enforced structurally.

    Every supervision field the producer knows about is attached to every fixture
    row with values chosen to disagree violently between two copies, and the
    resolution must be byte-identical. The engine passes because
    `row_identity_view` whitelists three metadata fields; the point of pinning it
    is that a future version consulting an affinity or a confidence to break a
    tie fails here immediately.
    """
    plain = [row for record in fx.ALL_RECORDS for row in record.rows()]
    baseline = ti.resolve_target_identity(plain)

    # The values must VARY BETWEEN ROWS, and vary differently in each pass. An
    # earlier version wrote one constant onto every row, which made a leaked
    # field carry no information at all: a resolver reading it would have grouped
    # every row together on a value they all shared, and the partition would have
    # come out identical anyway. A test that cannot detect the leak it is named
    # for is worse than no test, because it is quoted as evidence.
    rng = random.Random(2718)
    for pass_index in range(4):
        contaminated = []
        for position, row in enumerate(plain):
            # Deliberately correlated with the answer: a resolver that read any
            # of these would find a field that agrees with the digest it should
            # not know about, which is the strongest possible temptation.
            spiked = {
                field: f"{field}:{pass_index}:{position}:{rng.random()}"
                for field in ti.FORBIDDEN_LABEL_FIELDS
            }
            spiked["binder_label"] = position % 2
            spiked["is_strong_binder"] = position % 3 == 0
            spiked["affinity"] = 1e-12 * (position + 1)
            spiked["split"] = "val" if position % 4 else "train"
            metadata = dict(row["metadata"])
            metadata.update(spiked)
            polluted = {
                "metadata": metadata,
                "antigen_sequence": row["antigen_sequence"],
            }
            polluted.update(spiked)
            contaminated.append(polluted)
        result = ti.resolve_target_identity(contaminated)
        for record in fx.ALL_RECORDS:
            assert result.construct_id(record) == baseline.construct_id(record)
            assert (
                result.biological_target_id(record)
                == baseline.biological_target_id(record)
            )
            assert result.split_group_id(record) == baseline.split_group_id(record)
        # Not just the ids: the whole report, so a supervision leak into a
        # counter or a name verdict is caught too.
        assert result.stats()["edge_counts"] == baseline.stats()["edge_counts"]
        assert [d.approved for d in result.name_decisions()] == [
            d.approved for d in baseline.name_decisions()
        ]


def test_row_identity_view_reads_nothing_but_the_whitelist():
    """The whitelist is the mechanism, so it is tested directly too."""
    fields, sequence = ti.row_identity_view({
        "antigen_sequence": "acdefg",
        "target_name": "Spike Protein",
        "target_pdb": "6XYZ_A",
        "target_uniprot": "P12345-2",
        "is_strong_binder": True,
        "affinity": 1e-9,
        "split": "val",
    })
    assert set(fields) == set(ti.IDENTITY_METADATA_FIELDS)
    assert sequence == "ACDEFG"
    assert ti.normalize_target_name(fields["target_name"]) == "spike_protein"
    assert ti.canonicalize_accession(fields["target_pdb"]) == "6xyz"
    assert ti.canonicalize_accession(fields["target_uniprot"]) == "p12345"


# =========================================================================== #
# 2. Fault injections
# =========================================================================== #

def test_fault_injection_single_linkage_would_merge_the_rbd_chain(chain):
    """The acceptance test's anti-percolation claim fails without the criterion.

    This is the proof that
    `test_outcome_4_cluster_criterion_refuses_the_closing_edge` is testing the
    criterion rather than the data. The same three records, the same admitted
    edges, the wrong clustering algorithm: single-linkage welds all three into
    one construct, so the acceptance assertion ``construct(A) != construct(C)``
    would fail. If this test ever showed the two algorithms agreeing, the
    acceptance test would have quietly become powerless -- which is exactly what
    happened to the fixture this one replaced.
    """
    point = ti.DEFAULT_OPERATING_POINT
    digests = {
        key: ti.antigen_digest(ap.record(key).antigen_sequence)
        for key in ("rbd_199", "rbd_205", "rbd_209")
    }
    admitted = []
    for edge in ap.CHAIN_EDGES:
        if edge.admitted_at_construct_point:
            admitted.append((digests[edge.left], digests[edge.right]))
    assert len(admitted) == 2, "the chain must offer exactly two admitted edges"

    percolated = single_linkage(sorted(digests.values()), admitted)
    assert len(percolated.components()) == 1, (
        "single-linkage must merge the chain, or the fault is not a fault"
    )
    # And the shipped criterion does not.
    assert chain.construct_id(ap.record("rbd_199")) != chain.construct_id(
        ap.record("rbd_209")
    )
    assert chain.stats()["target_construct_merges_refused"] >= 1


def test_fault_injection_promoting_containment_to_identity_fuses_igkc_into_her2(resolved):
    """The containment demotion is load-bearing, demonstrated rather than asserted.

    If a containment relation were allowed to merge, the 107-aa IGKC domain and
    the 607-aa HER2 ectodomain would become one target through the 1057-aa fusion
    that contains both. The engine records both overlaps and merges neither; this
    shows what the alternative costs.
    """
    case = fx.CASES_BY_OUTCOME[2]
    igkc, her2, fusion = (
        case.record(k) for k in ("igkc", "her2_ecd", "fusion_1n8z")
    )
    # Both containment edges exist -- the premise.
    assert resolved.construct_id(fusion) in resolved.quarantine_partners(igkc)
    assert resolved.construct_id(fusion) in resolved.quarantine_partners(her2)
    # Promoting them to identity would connect IGKC to HER2 through the fusion.
    promoted = single_linkage(
        [resolved.construct_id(r) for r in (igkc, her2, fusion)],
        [
            (resolved.construct_id(igkc), resolved.construct_id(fusion)),
            (resolved.construct_id(her2), resolved.construct_id(fusion)),
        ],
    )
    assert len(promoted.components()) == 1
    # The engine does not.
    assert resolved.biological_target_id(igkc) != resolved.biological_target_id(her2)


def test_fault_injection_a_composite_that_bridges_welds_two_families(resolved):
    """The test-ineligibility cut on containers is what stops the split percolating.

    The fusion is marked test-ineligible and its quarantine edges are recorded
    but not closed through. Closing them would put IGKC's family and HER2's
    family in one split group -- the shape of the 77% component a previous
    attempt reached. This asserts the container is flagged, and that the two
    groups are in fact separate today.
    """
    case = fx.CASES_BY_OUTCOME[2]
    igkc, her2, fusion = (
        case.record(k) for k in ("igkc", "her2_ecd", "fusion_1n8z")
    )
    assert resolved.is_composite(fusion)
    assert resolved.test_ineligible(fusion)
    assert not resolved.test_ineligible(igkc)
    assert not resolved.test_ineligible(her2)
    assert resolved.split_group_id(igkc) != resolved.split_group_id(her2)
    # Closing through the container is what the cut prevents.
    welded = single_linkage(
        [resolved.split_group_id(r) for r in (igkc, her2, fusion)],
        [
            (resolved.split_group_id(igkc), resolved.split_group_id(fusion)),
            (resolved.split_group_id(her2), resolved.split_group_id(fusion)),
        ],
    )
    assert len(welded.components()) == 1


def _linker_population():
    """Two unrelated families joined only through a concatenated construct.

    Built rather than borrowed, because the acceptance fixtures contain exactly
    one composite and the guards below need a population where deleting each
    guard changes the answer. An adversarial reviewer showed that the whole suite
    stayed green with BOTH percolation cuts removed, which meant the guards were
    asserted and not tested. These two tests are the repair.
    """
    rng = random.Random(4242)
    residues = "ACDEFGHIKLMNPQRSTVWY"

    def protein(length):
        return "".join(rng.choice(residues) for _ in range(length))

    left = protein(180)
    right = protein(400)
    fusion = left + "G" * 12 + right
    rows = []
    for sequence, name, accession in (
        (left, "left domain", "l00001"),
        (right, "right domain", "r00001"),
        (fusion, "fusion construct", ""),
    ):
        rows.append({
            "antigen_sequence": sequence,
            "metadata": {"target_name": name, "target_pdb": "",
                         "target_uniprot": accession},
        })
    return rows, left, right, fusion


def test_fault_injection_closing_quarantine_through_a_container_welds_two_families():
    """Guard 2, tested by removing it rather than by asserting it.

    The container is test-ineligible and its quarantine edges are recorded but
    not closed through. This builds a population where that is the ONLY thing
    keeping two unrelated families in separate split groups, checks the guard
    holds, and then closes through the container anyway to show the guard is
    load-bearing. If both branches agreed, the guard would be decoration.
    """
    rows, left, right, fusion = _linker_population()
    resolution = ti.resolve_target_identity(rows)

    left_digest = ti.antigen_digest(left)
    right_digest = ti.antigen_digest(right)
    fusion_digest = ti.antigen_digest(fusion)

    # Premise: the container really does contain both, and really is flagged.
    assert resolution.is_composite(fusion_digest)
    assert resolution.test_ineligible(fusion_digest)
    partners = resolution.quarantine_partners(left_digest)
    assert resolution.construct_id(fusion_digest) in partners, (
        "the containment edge must exist, or the guard has nothing to refuse"
    )
    assert resolution.construct_id(right_digest) not in partners

    # The guard: the two components are NOT welded into one split group.
    assert resolution.split_group_id(left_digest) != resolution.split_group_id(
        right_digest
    )

    # The fault: close through the container anyway.
    welded = single_linkage(
        sorted({resolution.split_group_id(d)
                for d in (left_digest, right_digest, fusion_digest)}),
        [
            (resolution.split_group_id(left_digest),
             resolution.split_group_id(fusion_digest)),
            (resolution.split_group_id(right_digest),
             resolution.split_group_id(fusion_digest)),
        ],
    )
    assert len(welded.components()) == 1, (
        "without the cut the two families share one split group; if this ever "
        "showed otherwise, the guard above would be testing nothing"
    )


def test_fault_injection_disabling_the_container_span_guard_changes_the_answer():
    """Guard 3, tested by disabling it.

    A container spanning more families than `max_container_span` marks its
    members test-ineligible instead of linking them. Setting the span high enough
    to admit it must change the partition; if it does not, the guard never fires
    on this population and any assertion about it is vacuous.
    """
    rng = random.Random(99)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for index in range(6):
        sequence = "".join(rng.choice(residues) for _ in range(200 + index * 13))
        rows.append({
            "antigen_sequence": sequence,
            "metadata": {"target_name": "", "target_pdb": "9hub9",
                         "target_uniprot": f"h{index:05d}"},
        })
    digests = [ti.antigen_digest(row["antigen_sequence"]) for row in rows]

    guarded = ti.resolve_target_identity(rows, max_container_span=3)
    permissive = ti.resolve_target_identity(rows, max_container_span=99)

    # With the guard on, the hub is refused and its members are test-ineligible.
    assert all(guarded.test_ineligible(d) for d in digests)
    assert guarded.stats()["edge_counts"]["high_degree_bridges_refused"] == 1
    guarded_groups = {guarded.split_group_id(d) for d in digests}

    # With it off, the hub links all six into one split group and nothing is
    # flagged. The two answers must differ, or the guard is inert here.
    assert not any(permissive.test_ineligible(d) for d in digests)
    permissive_groups = {permissive.split_group_id(d) for d in digests}
    assert len(permissive_groups) == 1
    assert guarded_groups != permissive_groups, (
        "the span guard must change the partition on this population"
    )


def test_a_refused_bridge_is_flagged_rather_than_silently_dropped():
    """Refusing to link a hub must not leave its members unconstrained.

    An earlier version returned without linking AND without flagging, so the
    members of a refused container got no constraint of any kind -- the opposite
    of the conservative behaviour the refusal exists for, and contrary to what
    three separate documents said it did.
    """
    rng = random.Random(7)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    rows = [
        {
            "antigen_sequence": "".join(
                rng.choice(residues) for _ in range(150 + index * 11)
            ),
            "metadata": {"target_name": "", "target_pdb": "9big9",
                         "target_uniprot": f"b{index:05d}"},
        }
        for index in range(5)
    ]
    resolution = ti.resolve_target_identity(rows, max_container_span=2)
    digests = [ti.antigen_digest(row["antigen_sequence"]) for row in rows]
    assert resolution.stats()["edge_counts"]["high_degree_bridges_refused"] == 1
    assert all(resolution.test_ineligible(d) for d in digests), (
        "a refused container's members must be test-ineligible, not merely "
        "un-linked"
    )
    assert resolution.stats()["target_test_ineligible_constructs"] == len(digests)


# =========================================================================== #
# 3. Determinism, and the demotions
# =========================================================================== #

def test_resolution_does_not_depend_on_row_order():
    """Ids are a function of the population, not of arrival order."""
    rows = [row for record in fx.ALL_RECORDS for row in record.rows()]
    forward = ti.resolve_target_identity(rows)
    shuffled = list(rows)
    random.Random(1234).shuffle(shuffled)
    reversed_order = ti.resolve_target_identity(shuffled)
    for record in fx.ALL_RECORDS:
        assert forward.construct_id(record) == reversed_order.construct_id(record)
        assert (
            forward.biological_target_id(record)
            == reversed_order.biological_target_id(record)
        )
        assert forward.split_group_id(record) == reversed_order.split_group_id(record)


def test_shared_pdb_quarantines_without_merging(resolved):
    """E6: a structural container is co-occurrence, not identity.

    IL23A and IL12B share PDB entry 4GRW, share 24.78% of their residues, and
    share no 8-mer. They must stay two targets and still share a split group --
    the combination a single partition cannot express.
    """
    case = fx.CASES_BY_OUTCOME[3]
    il23a, il12b = case.record("il23a"), case.record("il12b")
    assert resolved.biological_target_id(il23a) == "uniprot:q9npf7"
    assert resolved.biological_target_id(il12b) == "uniprot:p29460"
    assert resolved.split_group_id(il23a) == resolved.split_group_id(il12b)
    assert resolved.construct_id(il12b) in resolved.quarantine_partners(il23a)
    kinds = {
        edge.kind
        for edge in resolved.quarantine_edge_list()
        if {edge.left, edge.right} == {
            resolved.construct_id(il23a), resolved.construct_id(il12b)
        }
    }
    assert kinds == {"shared_container"}


def test_name_approval_is_a_decision_with_a_reason(resolved):
    """E7 and E8: a coherent name merges, an incoherent one quarantines.

    A previous attempt deleted names as a merging force and measured the result:
    the partition changed by exactly zero components, because names had become
    inert labels on already-frozen components. Approving coherent names is what
    stops that; refusing incoherent ones is what stops outcome 5.
    """
    decisions = {decision.name: decision for decision in resolved.name_decisions()}

    incoherent = decisions["sars_cov2_wt"]
    assert incoherent.approved is False
    assert incoherent.attached_to is None
    assert len(incoherent.spanned_groups) >= 2
    assert incoherent.worst_identity is not None
    assert incoherent.reason

    coherent = decisions["sars_cov2_omicron"]
    assert coherent.approved is True
    assert coherent.attached_to is not None
    assert coherent.reason

    assert any(d.approved for d in resolved.name_decisions()), (
        "if no name is ever approved, names have been deleted rather than judged "
        "-- the failure this design was rewritten to avoid"
    )


def test_an_approved_name_actually_merges_something():
    """E7 is a force, not a label. Recorded failure mode (d), pinned.

    The design's central lesson is that deleting names as a merging force is not
    the fix: a previous attempt froze the sequence components first and appended
    names afterwards, changing the partition by exactly ZERO components. An
    adversarial pass showed that refusing every multi-group name left the whole
    suite green -- E7 fired zero times in every tested population, so the force
    was dead code that nothing would have noticed losing.

    This builds the population E7 exists for: two constructs that are similar
    enough to be the same target but NOT similar enough for the family relation,
    carrying one curator name. Approving the name must merge them, and the merge
    must be counted.
    """
    rng = random.Random(8675309)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    base = "".join(rng.choice(residues) for _ in range(300))
    # ~18% of positions changed: comfortably below the 0.90 family identity
    # threshold and comfortably above the 0.75 name bridge.
    cousin = list(base)
    for position in rng.sample(range(len(cousin)), 54):
        current = cousin[position]
        cousin[position] = rng.choice([r for r in residues if r != current])
    cousin = "".join(cousin)

    from smallAntibodyGen.entity_resolution.alignment import align_pair

    measured = align_pair(base, cousin)
    point = ti.DEFAULT_OPERATING_POINT
    assert measured.identity < point.family_identity, (
        "the premise: the family relation must REFUSE this pair, or the name has "
        "nothing to bridge"
    )
    assert measured.identity >= point.name_bridge_identity
    assert measured.overlap >= point.name_bridge_overlap

    rows = [
        {"antigen_sequence": sequence,
         "metadata": {"target_name": "shared curator name", "target_pdb": "",
                      "target_uniprot": accession}}
        for sequence, accession in ((base, "e70001"), (cousin, "e70002"))
    ]
    resolution = ti.resolve_target_identity(rows)
    decisions = {d.name: d for d in resolution.name_decisions()}
    decision = decisions["shared_curator_name"]
    assert decision.approved is True, decision.reason
    assert len(decision.spanned_groups) == 2, "it must have bridged two groups"
    assert resolution.stats()["edge_counts"]["E7_approved_name_merges"] >= 1, (
        "an approved name that merges nothing is a label, and labels were the "
        "failure this design replaced"
    )
    assert (
        resolution.biological_target_id(ti.antigen_digest(base))
        == resolution.biological_target_id(ti.antigen_digest(cousin))
    )


def test_a_shared_accession_actually_merges_something():
    """E5 is a force too, and nothing pinned it either.

    Deleting the whole accession loop left the suite green: E5 fired zero times
    on the acceptance fixtures. A curator writing one accession over two
    constructs is the commonest identity evidence in this corpus, so it needs a
    population where removing it changes the answer.
    """
    rng = random.Random(5150)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    # Two genuinely unrelated sequences, joined ONLY by a shared accession, so
    # no similarity relation can be doing the work.
    first = "".join(rng.choice(residues) for _ in range(240))
    second = "".join(rng.choice(residues) for _ in range(310))
    rows = [
        {"antigen_sequence": sequence,
         "metadata": {"target_name": "", "target_pdb": "", "target_uniprot": acc}}
        for sequence, acc in ((first, "p11111"), (second, "p11111"))
    ]
    resolution = ti.resolve_target_identity(rows)
    assert resolution.stats()["edge_counts"]["E5_shared_accession_merges"] == 1
    assert (
        resolution.biological_target_id(ti.antigen_digest(first))
        == resolution.biological_target_id(ti.antigen_digest(second))
        == "uniprot:p11111"
    )
    # And they are still two CONSTRUCTS: a shared accession says "same target",
    # never "same construct".
    assert resolution.construct_id(ti.antigen_digest(first)) != resolution.construct_id(
        ti.antigen_digest(second)
    )
    # The curator merge carries no diameter guarantee, and the report says so.
    stats = resolution.stats()
    assert stats["component_curator_merged"] == 1
    assert stats["criterion_component_count"] == 0


def test_a_ladder_of_individually_valid_names_does_not_percolate():
    """Freezing the verdicts is necessary and not sufficient. Union is transitive.

    Found by an adversarial reviewer after the freeze was in place, which is
    exactly why the freeze alone was not enough: ten antigens in a ladder, nine
    curator names, each name written on one adjacent pair that genuinely shares a
    long identical block. Every verdict is individually CORRECT -- each name does
    bridge two groups whose cross pair clears the bridge -- and an unconditional
    union of nine correct merges produces one family whose two ends share about
    22% identity and no 8-mer.

    That is the A~B admitted, B~C admitted, A~C refused shape the cluster
    criterion exists to refuse, arriving through names instead of sequences. So
    approved names go through `agglomerate_complete_linkage` like every other
    similarity relation, and this is the population that proves it.
    """
    rng = random.Random(1010)
    residues = "ACDEFGHIKLMNPQRSTVWY"

    def protein(length):
        return "".join(rng.choice(residues) for _ in range(length))

    # A sliding window over one long sequence: ten 500-residue constructs, each
    # offset 100 from the last. Neighbours share 400 residues (coverage 0.80,
    # comfortably over the bridge), rungs three apart share 200 (0.40, under it),
    # and the ends share nothing. This is the shape real construct series take,
    # and it is open exactly where a chain has to be open to percolate.
    genome = protein(1400)
    rungs = [genome[offset:offset + 500] for offset in range(0, 1000, 100)]
    assert len(rungs) == 10 and all(len(r) == 500 for r in rungs)

    rows = []
    for index in range(9):
        for member in (rungs[index], rungs[index + 1]):
            rows.append({
                "antigen_sequence": member,
                "metadata": {"target_name": f"ladder rung {index}",
                             "target_pdb": "", "target_uniprot": ""},
            })
    resolution = ti.resolve_target_identity(rows)
    digests = [ti.antigen_digest(r) for r in rungs]

    families = {resolution.biological_target_id(d) for d in digests}
    assert len(families) > 1, (
        "nine locally valid name merges must not collapse a ten-rung ladder "
        "into one biological target"
    )
    assert resolution.stats()["largest_family_row_share"] < 1.0

    groups = {resolution.split_group_id(d) for d in digests}
    assert len(groups) > 1, "the split groups must not collapse either"

    # Premise: the ladder really is open. The ends share almost nothing, and the
    # criterion really did refuse merges rather than there being none to make.
    from smallAntibodyGen.entity_resolution.alignment import align_pair

    ends = align_pair(rungs[0], rungs[-1])
    assert ends.identity < ti.DEFAULT_OPERATING_POINT.name_bridge_identity or (
        ends.min_coverage < ti.DEFAULT_OPERATING_POINT.name_bridge_coverage
    ), "the ladder ends must not themselves clear the bridge"
    assert resolution.stats()["edge_counts"]["E7_name_merges_refused"] >= 1, (
        "the criterion must be recorded as having refused a name merge, or this "
        "population is not exercising it"
    )

    # And the fault: an unconditional union of the same nine approved merges.
    percolated = single_linkage(
        sorted(digests),
        [(digests[i], digests[i + 1]) for i in range(9)],
    )
    assert len(percolated.components()) == 1, (
        "the fault must produce one component, or the assertion above is about "
        "the data rather than about the criterion"
    )


def test_refusing_to_close_through_a_container_is_counted_as_residual_exposure(resolved):
    """The price of the percolation cut, made visible instead of assumed away.

    Refusing to close through the fusion is what keeps IGKC's family and HER2's
    family apart. It also means the containment constraint between the fusion and
    each of them is NOT enforced, so the split may separate them -- and an
    adversarial reviewer demonstrated exactly that: at a different validation
    fraction the 607-aa HER2 ectodomain lands in scored validation while the
    1057-aa fusion containing it byte-for-byte lands in train.

    The refusal is still right; closing would weld two unrelated families. What
    would be wrong is calling it free. Each unenforced constraint is counted, so
    a producer can see the bill and pay it by excluding the container.
    """
    unclosed = resolved.unclosed_constraints()
    assert len(unclosed) >= 2, (
        "the fusion's two containment constraints are exactly the ones that go "
        "unenforced, and they must be counted rather than disappear"
    )
    assert resolved.stats()["unclosed_constraints"] == len(unclosed)
    kinds = {entry["kind"] for entry in unclosed}
    assert "containment" in kinds

    case = fx.CASES_BY_OUTCOME[2]
    fusion_construct = resolved.construct_id(case.record("fusion_1n8z"))
    assert any(
        fusion_construct in (entry["left"], entry["right"]) for entry in unclosed
    ), "the container is the thing whose constraints went unenforced"


def test_a_percolated_quarantine_closure_stops_being_scored():
    """The closure bound, which `max_container_span` cannot provide.

    `max_container_span` bounds ONE container's degree. It does not bound the
    CLOSURE and it cannot: a chain of containers each spanning exactly two
    families is legal at any span bound and closes into one group. Dropping those
    constraints is not an option -- they are the leakage constraints -- so the
    group stays closed and every construct in it stops being scored, which is
    what an over-large closure actually costs.
    """
    rng = random.Random(606)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    # Ten unrelated proteins, chained by nine PDB entries each holding exactly
    # two of them. Each container spans two families, so the span guard never
    # fires; the closure still swallows all ten.
    sequences = ["".join(rng.choice(residues) for _ in range(200 + i * 7))
                 for i in range(10)]
    rows = []
    for index in range(9):
        for member in (sequences[index], sequences[index + 1]):
            rows.append({
                "antigen_sequence": member,
                "metadata": {"target_name": "", "target_pdb": f"9c{index:02d}",
                             "target_uniprot": f"c{index:05d}{member[:1]}"},
            })
    digests = [ti.antigen_digest(s) for s in sequences]

    bounded = ti.resolve_target_identity(rows, max_split_group_families=4)
    assert bounded.stats()["edge_counts"]["high_degree_bridges_refused"] == 0, (
        "no single container may exceed the span bound, or this population is "
        "testing the wrong guard"
    )
    assert len({bounded.split_group_id(d) for d in digests}) == 1, (
        "the constraints are honoured, not dropped"
    )
    percolated = bounded.stats()["percolated_split_groups"]
    assert len(percolated) == 1 and percolated[0]["families_spanned"] > 4
    assert all(bounded.test_ineligible(d) for d in digests)

    # Above the bound nothing is flagged, so the flag is a property of the
    # closure rather than of the population.
    permissive = ti.resolve_target_identity(rows, max_split_group_families=99)
    assert permissive.stats()["percolated_split_groups"] == []
    assert not any(permissive.test_ineligible(d) for d in digests)


def test_name_verdicts_do_not_depend_on_what_other_names_are_called():
    """Requirement 2, tested where it actually breaks: alphabetical order.

    Names are judged in sorted order. An earlier version merged each approved
    name into the partition as it went, so the next name was judged against a
    partition the previous one had changed -- and the coherence check skips pairs
    already in one group, so an earlier merge could silently excuse a later
    name's worst pair. That makes the verdict depend on how the names happen to
    be spelled, which is iterative name bridging wearing a different hat.

    Same records, same relationships, names renamed so their sort order reverses.
    Every verdict must be identical.
    """
    rng = random.Random(31337)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    families = ["".join(rng.choice(residues) for _ in range(220)) for _ in range(3)]

    def rows_with(first: str, second: str):
        return [
            {"antigen_sequence": families[0],
             "metadata": {"target_name": first, "target_pdb": "",
                          "target_uniprot": "f00001"}},
            {"antigen_sequence": families[1],
             "metadata": {"target_name": first, "target_pdb": "",
                          "target_uniprot": "f00002"}},
            {"antigen_sequence": families[1],
             "metadata": {"target_name": second, "target_pdb": "",
                          "target_uniprot": "f00002"}},
            {"antigen_sequence": families[2],
             "metadata": {"target_name": second, "target_pdb": "",
                          "target_uniprot": "f00003"}},
        ]

    forward = ti.resolve_target_identity(rows_with("aaa bridge", "zzz bridge"))
    reverse = ti.resolve_target_identity(rows_with("zzz bridge", "aaa bridge"))

    def verdicts(resolution, mapping):
        return {
            mapping[d.name]: (d.approved, len(d.spanned_groups))
            for d in resolution.name_decisions()
        }

    forward_map = {"aaa_bridge": "first", "zzz_bridge": "second"}
    reverse_map = {"zzz_bridge": "first", "aaa_bridge": "second"}
    assert verdicts(forward, forward_map) == verdicts(reverse, reverse_map)
    # And the premise: both names really do span two groups, so there is a
    # verdict to get wrong.
    assert all(len(d.spanned_groups) == 2 for d in forward.name_decisions())


def test_a_container_never_bridges_a_name(resolved):
    """A concatenated construct is excluded from the evidence a name is judged on.

    ``human_her2`` is written on the fusion and on the HER2 ectodomain. The
    fusion resembles the ectodomain because it contains it, and treating that as
    evidence the name is coherent would let any container validate any name
    written on it.
    """
    decisions = {decision.name: decision for decision in resolved.name_decisions()}
    her2_name = decisions["human_her2"]
    fusion_digest = ti.antigen_digest(
        fx.CASES_BY_OUTCOME[2].record("fusion_1n8z").antigen_sequence
    )
    assert fusion_digest in her2_name.composite_members


def test_containment_is_measured_only_where_the_family_relation_did_not_answer(resolved):
    """A deliberate scoping choice, pinned so it cannot drift into a silent gap.

    Containment candidates are not length-banded -- they cannot be -- so on the
    corpus the pass proposed 450,381 pairs, an order of magnitude more than the
    length-banded similarity pass.
    Nearly all of the excess is two constructs of ONE biological family that
    genuinely contain each other, and a quarantine edge between them constrains
    nothing the family has not already constrained, because the split closes over
    families.

    So containment is measured only across families. This test pins both halves:
    the case the relation exists for is still measured, and the count of pairs
    skipped for already being one family is REPORTED rather than silently
    dropped.
    """
    case = fx.CASES_BY_OUTCOME[2]
    igkc, fusion, her2 = (
        case.record(k) for k in ("igkc", "fusion_1n8z", "her2_ecd")
    )
    # Premise: these really are different families, so the pass must look at them.
    assert resolved.biological_target_id(igkc) != resolved.biological_target_id(fusion)
    assert resolved.biological_target_id(her2) != resolved.biological_target_id(fusion)
    assert resolved.construct_id(fusion) in resolved.quarantine_partners(igkc)
    assert resolved.construct_id(fusion) in resolved.quarantine_partners(her2)

    containment = [
        entry for entry in resolved.stats()["blocking"]
        if entry.get("relation") == "containment"
    ]
    assert len(containment) == 1
    assert "skipped_already_one_family" in containment[0], (
        "a pass that quietly narrows its own scope is a silent cap"
    )
    assert "measured" in containment[0]


def test_unknown_claims_are_refused_rather_than_defaulted(resolved):
    """Claims are a controlled vocabulary."""
    record = fx.ALL_RECORDS[0]
    assert resolved.split_group_id(record, "generic")
    assert resolved.split_group_id(record, "unseen_mutant")
    with pytest.raises(ValueError, match="unknown claim"):
        resolved.split_group_id(record, "unseen_antibody")


def test_record_fields_carry_every_relation(resolved):
    """What a producer writes onto a row is all three relations, plus eligibility."""
    fields = resolved.record_fields(fx.CASES_BY_OUTCOME[2].record("fusion_1n8z"))
    assert set(fields) == {
        "antigen_sha256", "construct_id", "biological_target_id",
        "split_group_id", "unseen_mutant_group_id", "quarantine_partner_count",
        "test_ineligible", "antigen_is_composite",
    }
    assert fields["test_ineligible"] is True
    assert fields["antigen_is_composite"] is True
    assert fields["quarantine_partner_count"] >= 2


# =========================================================================== #
# 4. Non-vacuity of the calibration and audit populations
# =========================================================================== #

def test_curated_labels_match_the_committed_fixtures():
    """Every label's digest is reproduced from the sequence it claims to describe.

    A label whose digest drifted would score a different record than the one it
    names, silently and with a clean-looking report.
    """
    by_digest = {
        ti.antigen_digest(record.antigen_sequence): record
        for record in fx.ALL_RECORDS
    }
    assert set(labels.CURATED_LABELS) == set(by_digest), (
        "the label set and the fixture set must describe the same records"
    )
    for digest, record in by_digest.items():
        assert digest == record.antigen_sha256_32
        assert labels.CURATED_LABELS[digest].note


def test_audit_population_is_not_empty_and_scores_clean(resolved):
    """The audit is over real pairs, and the engine gets them right.

    An `ErrorReport` with zero pairs would report zero false merges and zero
    false splits and mean nothing at all, so the counts are asserted before the
    verdict is.
    """
    report = resolved.audit_report()
    assert set(report.families) == set(labels.PINNED_AUDIT_FAMILIES)
    assert report.pairs > 0
    assert report.positive_pairs > 0, (
        "with no positive pairs a false-split count of zero is free"
    )
    assert report.negative_pairs > 0, (
        "with no negative pairs a false-merge count of zero is free"
    )
    assert report.unadjudicated == len(labels.UNADJUDICATED_PAIRS)
    assert report.false_merges == 0, report.false_merge_examples
    assert report.false_splits == 0, report.false_split_examples
    assert report.tolerated_error == labels.PREDECLARED_ERROR_ASYMMETRY


def test_calibration_population_is_disjoint_from_the_audit_and_not_empty(resolved):
    """Requirement A: the thresholds are confirmed on families they never saw."""
    calibration = resolved.calibration_report()
    audit = resolved.audit_report()
    assert set(calibration.families) & set(audit.families) == set()
    assert len(calibration.families) >= 10, (
        "a calibration set small enough to be an accident is not a calibration set"
    )
    assert calibration.positive_pairs > 0 and calibration.negative_pairs > 0
    assert calibration.false_merges == 0, calibration.false_merge_examples
    assert calibration.false_splits == 0, calibration.false_split_examples


def test_synthetic_generator_plants_every_pathology_it_claims():
    """The calibration population is only worth what it actually contains."""
    population = synthetic.build_population()
    assert set(population.pathologies) == {
        "similarity_chain", "truncation_series", "container", "shared_container",
        "accession_conflict", "high_degree_name", "tie", "interior_indel",
        "unrelated_singleton",
    }
    assert all(count > 0 for count in population.pathologies.values())
    assert len(population.truth) > 50
    assert len(population.rows) > len(population.truth), (
        "several rows per sequence, or the annotation channel is untested"
    )


def test_synthetic_generator_is_seeded_and_independent():
    """Two builds agree; the generator imports none of the resolver's machinery."""
    first = synthetic.build_population(synthetic.CALIBRATION_SEED)
    second = synthetic.build_population(synthetic.CALIBRATION_SEED)
    assert first.truth == second.truth
    assert synthetic.build_population(1).truth != first.truth

    # Checked against the parsed import graph rather than the file text, so that
    # naming the resolver in a docstring stays allowed and importing it does not.
    import ast

    tree = ast.parse(Path(synthetic.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported <= {"functools", "random", "dataclasses", "typing",
                        "__future__"}, (
        f"the generator imports {sorted(imported)}; an oracle that shares the "
        f"resolver's helpers measures nothing"
    )


def test_conflicting_accessions_are_reported_with_both_sides(resolved):
    """E9: the disagreement survives into the report, with a neutral id in its place."""
    conflicts = resolved.accession_conflicts()
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.accessions == ("p63000", "p63001")
    assert conflict.names == ("rac1_human", "rac1_mouse")
    assert conflict.resolved_id.startswith("family:")
    assert conflict.resolved_id not in ("uniprot:p63000", "uniprot:p63001")
