"""
Tests for the three primitives the typed-identity engine is built on.

Every claim these modules make in prose is checked here against something that
does not share its implementation:

- the vectorised aligner against a naive rewrite of the same recurrences,
- the blocker's derived recall guarantee against exhaustive all-pairs,
- the bounded cluster criterion against the single-linkage it replaces.

The last one is the important pattern. A test that the criterion produces the
right partition is only worth something if substituting the WRONG algorithm makes
it fail, so every anti-percolation assertion here is paired with a fault
injection that proves it.
"""
from __future__ import annotations

import random

import pytest

from smallAntibodyGen.entity_resolution import blocking, clustering
from smallAntibodyGen.entity_resolution.alignment import (
    EMPTY_ALIGNMENT,
    AlignmentTooLarge,
    align_pair,
    reference_align_pair,
)

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def _protein(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(RESIDUES) for _ in range(length))


def _mutate(rng: random.Random, sequence: str, edits: int) -> str:
    residues = list(sequence)
    for _ in range(edits):
        position = rng.randrange(len(residues))
        roll = rng.random()
        if roll < 0.6:
            residues[position] = rng.choice(RESIDUES)
        elif roll < 0.8:
            residues.insert(position, rng.choice(RESIDUES))
        elif len(residues) > 1:
            residues.pop(position)
    return "".join(residues)


# =========================================================================== #
# Alignment
# =========================================================================== #

def test_vectorised_aligner_matches_its_naive_oracle():
    """The prefix-maximum unrolling in `_fill` is exact, not approximate.

    `align_pair` replaces the serial horizontal-gap recurrence with a running
    maximum, which is an algebraic identity rather than a heuristic. This is what
    makes that a checked claim: `reference_align_pair` implements the same
    recurrences one cell at a time with three full matrices and shares no code
    path with the fast version, and the two must agree exactly -- score, matched
    residues, columns and both coverages -- over randomised inputs spanning
    identical, near-identical, and unrelated pairs.
    """
    rng = random.Random(20260901)
    for _ in range(120):
        left = _protein(rng, rng.randint(1, 48))
        if rng.random() < 0.6:
            right = _mutate(rng, left, rng.randint(0, 10)) or "A"
        else:
            right = _protein(rng, rng.randint(1, 48))
        assert align_pair(left, right) == reference_align_pair(left, right), (
            left, right
        )


def test_affine_gaps_prefer_one_long_gap_to_several_short_ones():
    """The gap model is affine, and observably so.

    A linear penalty charges the same for one gap of length six as for six gaps
    of length one, so it has no reason to prefer the biologically right answer.
    Affine charges an opening once. This pins the difference where it shows: a
    single interior deletion must align as ONE gap, giving one contiguous
    alignment rather than a scatter of short matches.
    """
    left = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
    right = left[:12] + left[18:]
    result = align_pair(left, right)
    assert result.matched == len(right)
    assert result.columns == len(left)
    # Six gap columns, opened once. Under BLAST's defaults that costs 11 + 6*1;
    # six separately opened gaps would cost 6 * 12 and the DP would prefer almost
    # any other path.
    assert result.columns - result.matched == 6


def test_local_alignment_reads_a_truncation_as_full_identity_over_partial_coverage():
    """Identity and coverage stay orthogonal, which is the whole reason for local.

    A global aligner charges for the overhang and reports a truncation as a
    divergence. That single conflation is what makes "same protein, different
    construct boundaries" impossible to say, and every relation in the engine
    depends on being able to say it.
    """
    rng = random.Random(7)
    whole = _protein(rng, 400)
    part = whole[40:360]
    result = align_pair(part, whole)
    assert result.identity == 1.0
    assert result.cov_left == 1.0
    assert result.cov_right == pytest.approx(320 / 400)


def test_alignment_is_symmetric_and_reflexive():
    """Reversing the arguments mirrors the result; a sequence covers itself fully.

    This needs UNRELATED pairs to have any power. Identity and coverage are read
    off one co-optimal path, and transposing the arguments swaps the two gap
    states, so ties resolve the other way -- but near-identical pairs have almost
    no ties. An earlier version of this test used only mutated copies of one
    sequence and passed while the property was false on 24 of 400 real corpus
    pairs. Half the pairs here are unrelated, which is where the ties live.
    """
    rng = random.Random(11)
    unrelated_pairs = 0
    for index in range(80):
        left = _protein(rng, rng.randint(20, 160))
        if index % 2:
            right = _mutate(rng, left, rng.randint(0, 12)) or "A"
        else:
            right = _protein(rng, rng.randint(20, 160))
            unrelated_pairs += 1
        assert align_pair(left, right) == align_pair(right, left).flipped(), (
            left, right
        )
        identical = align_pair(left, left)
        assert identical.identity == 1.0
        assert identical.cov_left == identical.cov_right == 1.0
    assert unrelated_pairs >= 30, "the tie-prone half must actually be present"


def test_ambiguity_codes_that_do_score_against_themselves_are_pinned_too():
    """Not every unscored-looking residue is unscored, and the difference matters.

    `J`, `O` and `U` fold onto `X`, which scores -1 against itself, so a run of
    them produces no alignment at all. But `B` (Asp/Asn) and `Z` (Glu/Gln) have
    real BLOSUM62 rows scoring +4 against themselves, and `*` scores +1 -- so a
    run of THOSE aligns perfectly and would fuse two targets that share nothing
    but a stretch of ambiguity. Testing only the three codes that cannot fail is
    how that would go unnoticed.
    """
    for residue in "JOU":
        assert align_pair(residue * 40, residue * 40) is EMPTY_ALIGNMENT

    for residue in "BZ*":
        result = align_pair(residue * 40, residue * 40)
        assert result is not EMPTY_ALIGNMENT, (
            f"{residue!r} scores positively against itself and this test exists "
            f"to record that, not to wish it away"
        )
        assert result.identity == 1.0
    # The consequence, stated where a reader will see it: an antigen that is a
    # long run of B or Z would look identical to any other such run. The corpus
    # has none -- if one appears, the operating point's overlap floor is the
    # thing standing between it and a false merge, so the floor is not optional.
    assert align_pair("B" * 40, "Z" * 40).identity < 1.0


def test_empty_and_unrelated_inputs_do_not_invent_a_relation():
    """An empty side, or no positive-scoring segment, returns the empty alignment."""
    assert align_pair("", "ACDEF") is EMPTY_ALIGNMENT
    assert align_pair("ACDEF", "") is EMPTY_ALIGNMENT


def test_exotic_residues_fold_onto_the_ambiguity_code():
    """`J`, `O` and `U` survive scoring and buy no similarity.

    `clean_aa_sequence` can emit all three and BLOSUM62 has no row for any of
    them, so the aligner folds them onto ``X``. Two consequences are worth
    pinning. First, they never crash. Second -- and this is the one that matters
    for leakage -- a stretch of unknown residues scores NEGATIVELY against
    itself, so two sequences of nothing but ambiguity codes produce no alignment
    at all rather than looking identical. An implementation that scored them as
    self-matches would fuse every poorly sequenced antigen into one target.
    """
    for residue in "JOU":
        assert align_pair(residue * 40, residue * 40) is EMPTY_ALIGNMENT, (
            "ambiguity codes must not manufacture a relation out of nothing"
        )

    # Folded, not dropped: a single exotic residue inside a real sequence costs
    # one mismatch and leaves the surrounding alignment intact.
    rng = random.Random(5)
    clean = _protein(rng, 60)
    for residue in "JOUX":
        contaminated = clean[:30] + residue + clean[31:]
        result = align_pair(clean, contaminated)
        assert result.columns == 60
        assert result.matched == 59, residue


def test_oversized_pairs_are_refused_rather_than_skipped():
    """A comparison that cannot be made must be countable.

    Silently skipping an expensive pair is an unmeasured leakage risk: the pair
    might have been a near-duplicate straddling the split, and nothing would say
    so.
    """
    with pytest.raises(AlignmentTooLarge):
        align_pair("A" * 400, "C" * 400, max_cells=1000)


# =========================================================================== #
# Blocking
# =========================================================================== #

def test_length_band_floor_is_the_product_of_the_two_thresholds():
    """The derived band, checked against the derivation in the module docstring."""
    assert blocking.length_band_floor(0.99, 0.95) == pytest.approx(0.9405)
    assert blocking.length_band_floor(0.90, 0.80) == pytest.approx(0.72)
    with pytest.raises(ValueError):
        blocking.length_band_floor(0.0, 0.9)
    with pytest.raises(ValueError):
        blocking.length_band_floor(0.9, 1.5)


def test_kmer_floor_tightens_as_the_thresholds_rise():
    """A stricter relation demands more shared k-mers, and never a negative count."""
    strict = blocking.guaranteed_shared_kmers(400, 0.99, 0.95)
    loose = blocking.guaranteed_shared_kmers(400, 0.90, 0.80)
    assert strict > loose > 0
    assert blocking.guaranteed_shared_kmers(9, 0.90, 0.80) == 0, (
        "a sequence too short for the bound to be positive must disable the "
        "filter rather than be dropped by it"
    )


@pytest.mark.parametrize(
    "identity,coverage", [(0.99, 0.95), (0.90, 0.80), (0.70, 0.50)]
)
def test_blocking_recall_is_total_against_exhaustive_all_pairs(identity, coverage):
    """A pair never proposed cannot be recovered downstream, so recall is correctness.

    The population deliberately contains everything the filters could get wrong:
    a mutation ladder, nested truncations, byte-identical duplicates, sequences
    shorter than the k-mer width, and unrelated padding. Every pair that actually
    meets the thresholds under exhaustive alignment must have been proposed.
    """
    rng = random.Random(4242)
    population = {}
    for family in range(5):
        base = _protein(rng, rng.randint(80, 220))
        population[f"f{family}"] = base
        population[f"f{family}_dup"] = base
        for step, edits in enumerate((2, 6, 18, 40), start=1):
            population[f"f{family}_m{step}"] = _mutate(rng, base, edits) or "A"
        population[f"f{family}_trunc"] = base[: int(len(base) * 0.7)]
        population[f"f{family}_inner"] = base[10 : len(base) - 10]
    population["tiny_a"] = "ACDE"
    population["tiny_b"] = "ACDE"
    population["tiny_c"] = "WYWY"

    report = blocking.blocking_recall_report(population, identity, coverage)
    assert report.missed == (), f"blocker lost qualifying pairs: {report.missed}"
    assert report.recall == 1.0
    assert report.qualifying_pairs > 0, (
        "a recall of 1.0 over zero qualifying pairs would prove nothing"
    )


def test_candidate_generation_is_order_independent():
    """The same population in a different insertion order gives the same candidates."""
    rng = random.Random(9)
    population = {f"s{i}": _protein(rng, rng.randint(60, 160)) for i in range(20)}
    forward = blocking.CandidateIndex(population).candidate_pairs(0.90, 0.80)[0]
    shuffled = dict(sorted(population.items(), key=lambda kv: kv[1]))
    reverse = blocking.CandidateIndex(shuffled).candidate_pairs(0.90, 0.80)[0]
    assert forward == reverse


def test_the_overlap_floor_prunes_pairs_that_cannot_meet_it():
    """Filter 3, and it matters most exactly where filter 1 is switched off.

    A sequence shorter than the relation's matched-residue floor can never meet
    it, because matched residues cannot exceed either sequence's length. Without
    this, a sequence shorter than the k-mer width carries no k-mer filter AND no
    length band during a containment search, and pairs with the entire
    population: the shipped corpus has three such antigens, which between them
    would contribute 28,719 candidates that provably cannot qualify.
    """
    rng = random.Random(77)
    population = {f"s{i}": _protein(rng, rng.randint(100, 300)) for i in range(30)}
    population["runt"] = "ACDEFG"

    index = blocking.CandidateIndex(population)
    unfiltered, _ = index.candidate_pairs(0.95, 0.95, apply_length_band=False)
    filtered, _ = index.candidate_pairs(
        0.95, 0.95, apply_length_band=False, min_overlap=30
    )
    runt_pairs = [p for p in unfiltered if "runt" in p]
    assert len(runt_pairs) == 30, "the premise: the runt pairs with everything"
    assert not any("runt" in p for p in filtered)
    assert set(filtered) <= set(unfiltered), "the floor prunes, it never adds"


def test_containment_search_must_drop_the_length_band():
    """A domain inside a fusion is outside any length band, and must still be found."""
    rng = random.Random(31)
    domain = _protein(rng, 107)
    fusion = _protein(rng, 400) + domain + _protein(rng, 550)
    population = {"domain": domain, "fusion": fusion}
    index = blocking.CandidateIndex(population)
    banded, _ = index.candidate_pairs(0.95, 0.95)
    unbanded, _ = index.candidate_pairs(0.95, 0.95, apply_length_band=False)
    assert banded == (), "the length band correctly excludes a 0.10 ratio pair"
    assert unbanded == (("domain", "fusion"),)


# =========================================================================== #
# Clustering
# =========================================================================== #

def _chain_population():
    """Three members forming an open chain: A~B and B~C admitted, A~C refused."""
    admitted = {("a", "b"), ("b", "c")}

    def qualifies(left, right):
        return (left, right) in admitted or (right, left) in admitted

    return ["a", "b", "c"], sorted(admitted), qualifies


def test_bounded_criterion_refuses_the_closing_edge():
    """Complete-linkage will not merge a component whose cross pair fails."""
    members, edges, qualifies = _chain_population()
    partition = clustering.agglomerate_complete_linkage(members, edges, qualifies)
    components = {frozenset(m) for _, m in partition.components()}
    assert components == {frozenset({"a", "b"}), frozenset({"c"})}
    assert partition.merges_refused == 1, (
        "the criterion must record that it actually blocked a merge"
    )


def test_fault_injection_single_linkage_percolates_the_same_chain():
    """The fault that makes the test above fail, proving it was testing something.

    Same members, same admitted edges, wrong algorithm. If this produced the same
    partition, the assertion above would be about the data rather than about the
    criterion.
    """
    members, edges, _ = _chain_population()
    percolated = clustering.single_linkage(members, edges)
    components = {frozenset(m) for _, m in percolated.components()}
    assert components == {frozenset({"a", "b", "c"})}
    assert components != {frozenset({"a", "b"}), frozenset({"c"})}


def test_merge_order_is_pinned_under_ties():
    """Every permutation of a tied edge set gives one partition.

    Complete-linkage agglomeration is order-sensitive when edges tie, so the
    order is fixed rather than left to whatever the caller happened to build.
    """
    import itertools

    members = ["m1", "m2", "m3", "m4"]
    admitted = {("m1", "m2"), ("m2", "m3"), ("m3", "m4"), ("m1", "m3")}

    def qualifies(left, right):
        return (left, right) in admitted or (right, left) in admitted

    results = set()
    for permutation in itertools.permutations(sorted(admitted)):
        partition = clustering.agglomerate_complete_linkage(
            members, list(permutation), qualifies
        )
        results.add(tuple(sorted(frozenset(m) for _, m in partition.components())))
    assert len(results) == 1, f"merge order leaked into the partition: {results}"


def test_hard_merges_constrain_later_clustering():
    """A pre-merged component is a constraint, not a suggestion.

    Byte-identical sequences and shared accessions merge before any threshold
    applies. Passing that partition in means the completeness check runs over the
    whole pre-merged component, so a later similarity edge cannot attach to one
    member while failing against another.
    """
    members = ["x", "y", "z"]
    seeded = clustering.Partition.singletons(members)
    seeded.force_union("x", "y")

    def qualifies(left, right):
        # z is close to y and far from x.
        return {left, right} == {"y", "z"}

    partition = clustering.agglomerate_complete_linkage(
        members, [("y", "z")], qualifies, partition=seeded
    )
    assert partition.members("z") == ("z",)
    assert partition.merges_refused == 1


def test_cluster_report_measures_rather_than_assumes():
    """The report's minima come from the measured pairs, and a singleton is honest."""
    from smallAntibodyGen.entity_resolution.alignment import PairAlignment

    def measure(left, right):
        table = {("a", "b"): 0.98, ("a", "c"): 0.91, ("b", "c"): 0.93}
        identity = table[tuple(sorted((left, right)))]
        return PairAlignment(
            identity=identity, cov_left=identity, cov_right=identity,
            overlap=100, columns=100, matched=int(identity * 100),
            span_left=100, span_right=100, score=0,
        )

    report = clustering.cluster_report("cid", ["a", "b", "c"], measure)
    assert report.min_pairwise_identity == pytest.approx(0.91)
    assert report.max_diameter == pytest.approx(0.09)
    assert report.representative == "b", (
        "the representative is the member with the best worst-case identity"
    )
    assert report.exhaustive and report.pairs_measured == 3

    singleton = clustering.cluster_report("s", ["only"], measure)
    assert singleton.size == 1
    assert singleton.min_pairwise_identity == 1.0
    assert singleton.max_diameter == 0.0


def test_oversized_clusters_report_that_they_were_sampled():
    """A partial minimum is never quoted as a total one."""
    from smallAntibodyGen.entity_resolution.alignment import PairAlignment

    flat = PairAlignment(
        identity=1.0, cov_left=1.0, cov_right=1.0, overlap=10, columns=10,
        matched=10, span_left=10, span_right=10, score=0,
    )
    members = [f"m{i:04d}" for i in range(300)]
    report = clustering.cluster_report(
        "big", members, lambda a, b: flat, max_report_pairs=100
    )
    assert report.exhaustive is False
    assert report.pairs_measured < 300 * 299 // 2
