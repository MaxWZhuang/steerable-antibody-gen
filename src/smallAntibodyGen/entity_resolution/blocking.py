"""
Candidate generation, with a recall guarantee rather than a recall hope.

WHY BLOCKING IS CORRECTNESS AND NOT SPEED
-----------------------------------------
The design this package implements is explicit about it: "A pair never generated
as a candidate cannot be recovered downstream. Candidate generation is therefore
part of the resolver's correctness contract." A blocker that quietly drops a
near-duplicate pair produces a split that *looks* group-disjoint and is not, and
nothing downstream can tell.

The corpus makes blocking unavoidable. Measured on the shipped shards: 9,574
distinct antigen sequences, median length 401, 99th percentile 3,391, maximum
34,350 -- 45.8 million pairs and about 1.5e13 dynamic-programming cells. Aligning
all of them is not an option at any constant factor.

So the filters here are chosen to be *provable* rather than merely effective. Each
one is a necessary condition for a pair to meet a threshold, derived below, so a
rejected pair is rejected because it cannot qualify, not because it looked
unpromising.

FILTER 1 -- THE LENGTH BAND
---------------------------
Let the pair meet ``identity >= t`` and ``min_coverage >= c``. Write ``S`` for the
shorter sequence and ``L`` for the longer, and let the alignment consume
``span_S`` and ``span_L`` residues over ``columns`` columns.

Coverage of the long side gives ``span_L >= c * len(L)``. Identity bounds the gaps
at ``(1 - t) * columns``, and since the short side loses at most those columns,
``span_S >= t * columns >= t * span_L``. Finally ``span_S <= len(S)``. Chaining::

    c * len(L) <= span_L <= span_S / t <= len(S) / t
    =>  len(S) / len(L)  >=  c * t

so the length ratio floor is exactly ``coverage_threshold * identity_threshold``.
At the construct operating point (0.99, 0.95) that is 0.9405; at the family point
(0.90, 0.80) it is 0.72. Measured on the corpus, a 0.90 band alone cuts 45.8
million pairs to 3.6 million.

FILTER 2 -- THE SHARED-K-MER FLOOR
----------------------------------
An alignment with ``columns`` columns and ``e`` non-matching columns has its
matched positions in at most ``e + 1`` runs totalling ``columns - e`` residues. A
run of length r contributes ``max(0, r - k + 1)`` exact k-mers, so the number of
k-mer positions matched in BOTH sequences is at least::

    (columns - e) - (e + 1) * (k - 1),   with e <= (1 - t) * columns

and ``columns >= c * len(S)`` because the alignment covers at least that much of
the shorter side. Every one of those k-mers occurs in both sequences, so

    sum over k-mers of min(count_in_a, count_in_b)  >=  that floor

is a *necessary* condition, counted with multiplicity so that repeats cannot
break it. Measured on the corpus at (0.93, 0.93), it takes the 513,700 pairs that
survive the length band and a shared 8-mer down to 20,012 -- a 25-fold cut with
no pair lost that could have qualified.

WHAT THIS DOES NOT COVER
------------------------
Both filters are necessary conditions for *local sequence similarity above a
threshold*. They say nothing about pairs related by structure, by function, or by
paraphrase in some other modality, and they are not a defence against a
relationship the operating point does not name. `blocking_recall_report` measures
what the blocker actually recovers against exhaustive all-pairs on a sample, and
that measurement -- not this derivation -- is what the conformance artifact
quotes.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Set, Tuple

#: k-mer width. Eight is the same width the acceptance fixtures use for their
#: exact non-homology witness (`shares_no_kmer`), so a pair this blocker refuses
#: on the k-mer floor and a pair the fixtures call non-homologous are talking
#: about the same evidence.
DEFAULT_K = 8


def length_band_floor(identity_threshold: float, coverage_threshold: float) -> float:
    """Smallest ``len(short) / len(long)`` a qualifying pair can have.

    Derived in the module docstring: ``coverage * identity``. A pair outside this
    band cannot meet both thresholds, whatever its residues are.

    Args:
        identity_threshold: Minimum matched-over-columns the relation requires.
        coverage_threshold: Minimum of the two coverages the relation requires.

    Returns:
        The length-ratio floor, in ``(0, 1]``.

    Raises:
        ValueError: When either threshold is outside ``(0, 1]``.
    """
    if not 0.0 < identity_threshold <= 1.0:
        raise ValueError(f"identity_threshold out of range: {identity_threshold}")
    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError(f"coverage_threshold out of range: {coverage_threshold}")
    return identity_threshold * coverage_threshold


def guaranteed_shared_kmers(
    shorter_length: int,
    identity_threshold: float,
    coverage_threshold: float,
    k: int = DEFAULT_K,
) -> int:
    """Fewest k-mer positions a qualifying pair must share, counted with multiplicity.

    Derived in the module docstring. Returns 0 when the shorter sequence is too
    short for the bound to be positive, which correctly disables the filter
    rather than silently dropping short pairs -- those are cheap to align anyway.

    Args:
        shorter_length: Length of the shorter sequence in the pair.
        identity_threshold: Minimum matched-over-columns the relation requires.
        coverage_threshold: Minimum of the two coverages the relation requires.
        k: k-mer width.

    Returns:
        A non-negative floor on shared k-mer occurrences.
    """
    columns = coverage_threshold * shorter_length
    errors = (1.0 - identity_threshold) * columns
    floor = (columns - errors) - (errors + 1.0) * (k - 1)
    if floor <= 0:
        return 0
    return int(floor)


def kmer_counts(sequence: str, k: int = DEFAULT_K) -> Mapping[str, int]:
    """Count every k-mer of a sequence.

    Args:
        sequence: Residue string.
        k: k-mer width.

    Returns:
        A mapping from k-mer to occurrence count. Empty when the sequence is
        shorter than ``k``.
    """
    if len(sequence) < k:
        return {}
    counts: Dict[str, int] = collections.defaultdict(int)
    for position in range(len(sequence) - k + 1):
        counts[sequence[position:position + k]] += 1
    return counts


@dataclass(frozen=True)
class BlockingStats:
    """What one candidate-generation pass did, for the guard report.

    Attributes:
        sequences: Sequences indexed.
        distinct_kmers: Distinct k-mers in the index.
        length_band_pairs: Pairs that cleared the length band.
        kmer_survivors: Pairs that also cleared the shared-k-mer floor.
        short_sequences: Sequences below ``k`` residues, compared exhaustively
            against every length-compatible partner because no k-mer filter
            applies to them.
        identity_threshold: The threshold the filters were derived from.
        coverage_threshold: The threshold the filters were derived from.
        k: k-mer width.
    """

    sequences: int
    distinct_kmers: int
    length_band_pairs: int
    kmer_survivors: int
    short_sequences: int
    identity_threshold: float
    coverage_threshold: float
    k: int

    def as_dict(self) -> Dict[str, object]:
        """Render for a JSON guard report."""
        return {
            "sequences": self.sequences,
            "distinct_kmers": self.distinct_kmers,
            "length_band_pairs": self.length_band_pairs,
            "kmer_survivors": self.kmer_survivors,
            "short_sequences": self.short_sequences,
            "identity_threshold": self.identity_threshold,
            "coverage_threshold": self.coverage_threshold,
            "k": self.k,
            "length_band_floor": length_band_floor(
                self.identity_threshold, self.coverage_threshold
            ),
        }


class CandidateIndex:
    """A k-mer inverted index over a fixed, ordered sequence population.

    The population is ordered once by the caller and never reordered, so the
    candidate pairs this yields -- and therefore every edge, cluster and id
    derived from them -- do not depend on the order rows arrived in.

    Attributes:
        keys: The population's keys, in the order given.
    """

    def __init__(self, sequences: Mapping[str, str], k: int = DEFAULT_K) -> None:
        """Index a population.

        Args:
            sequences: Key to residue string. Iterated in sorted key order, so
                two callers with the same population build the same index.
            k: k-mer width.
        """
        self.k = k
        self.keys: Tuple[str, ...] = tuple(sorted(sequences))
        self._sequence = {key: sequences[key] for key in self.keys}
        self._length = {key: len(sequences[key]) for key in self.keys}
        self._counts = {key: kmer_counts(sequences[key], k) for key in self.keys}
        self._postings: Dict[str, List[str]] = collections.defaultdict(list)
        for key in self.keys:
            for kmer in self._counts[key]:
                self._postings[kmer].append(key)
        # Keys short enough to carry no k-mer at all. They bypass filter 2, and
        # are counted so the guard report can say how many pairs were compared
        # exhaustively rather than filtered.
        self._short = tuple(key for key in self.keys if self._length[key] < k)

    def sequence(self, key: str) -> str:
        """The residue string indexed under ``key``."""
        return self._sequence[key]

    def length(self, key: str) -> int:
        """The length of the sequence indexed under ``key``."""
        return self._length[key]

    def candidate_pairs(
        self,
        identity_threshold: float,
        coverage_threshold: float,
        *,
        apply_length_band: bool = True,
        min_overlap: int = 0,
    ) -> Tuple[Tuple[Tuple[str, str], ...], BlockingStats]:
        """Every pair that could possibly meet the thresholds.

        Args:
            identity_threshold: Minimum matched-over-columns.
            coverage_threshold: Minimum of the two coverages.
            apply_length_band: Whether filter 1 applies. It must be switched OFF
                for a containment search, where the thresholds bind only the
                SHORTER side and the length ratio is unbounded by design -- the
                107-aa IGKC domain sits inside a 1057-aa fusion at a ratio of
                0.10. Filter 2 still applies, derived from the shorter length, so
                the search stays bounded.
            min_overlap: The relation's absolute matched-residue floor, if it has
                one. This is filter 3 and it is derived like the others: matched
                residues cannot exceed either sequence's length, so a pair whose
                SHORTER side is below the floor can never meet it. It matters most
                exactly where the length band is switched off -- the corpus holds
                three antigens shorter than the k-mer width, which carry no k-mer
                filter at all and would otherwise pair with all 9,573 others for
                28,719 candidates that provably cannot qualify.

        Returns:
            ``(pairs, stats)``. Pairs are ``(a, b)`` with ``a < b`` in the
            population order, and the tuple itself is sorted, so the result is a
            pure function of the population and the thresholds.
        """
        floor_ratio = (
            length_band_floor(identity_threshold, coverage_threshold)
            if apply_length_band
            else 0.0
        )
        pairs: Set[Tuple[str, str]] = set()
        band_pairs = 0
        # The band floor is a product of two doubles, so at the shipped family
        # point it evaluates to 0.7200000000000001 -- one ulp ABOVE the derived
        # 0.72. Comparing lengths against it directly rejects a pair whose length
        # ratio is exactly the floor, which the derivation admits. The
        # derivation is an inequality with equality allowed, so the comparison
        # has to allow it too.
        slack = 1e-9

        for index, key in enumerate(self.keys):
            own_length = self._length[key]
            own_counts = self._counts[key]
            if own_length < min_overlap:
                continue
            low = max(own_length * floor_ratio, float(min_overlap)) - slack
            high = (own_length / floor_ratio if floor_ratio > 0 else float("inf")) + slack

            def band_ok(other_key: str) -> bool:
                return low <= self._length[other_key] <= high

            # The k-mer filter is only usable when this sequence HAS k-mers and
            # the derived floor is positive. When either fails the filter is
            # vacuous, and a vacuous filter must admit every band-compatible
            # partner rather than admit only the partners that happen to share a
            # k-mer -- a pair sharing none would otherwise never be enumerated at
            # all, which is a silent drop rather than a disabled filter. The scan
            # runs over the whole population rather than the successors, because
            # the partner may sort either side of this key and the result must
            # not depend on that.
            floor_for_own = guaranteed_shared_kmers(
                own_length, identity_threshold, coverage_threshold, self.k
            )
            if own_length < self.k or floor_for_own <= 0:
                for other in self.keys:
                    if other == key or not band_ok(other):
                        continue
                    band_pairs += 1
                    pairs.add((key, other) if key < other else (other, key))
                continue

            shared: Dict[str, int] = collections.defaultdict(int)
            for kmer, own_count in own_counts.items():
                for other in self._postings[kmer]:
                    if other <= key or not band_ok(other):
                        continue
                    shared[other] += min(own_count, self._counts[other][kmer])

            for other, observed in shared.items():
                band_pairs += 1
                other_length = self._length[other]
                needed = guaranteed_shared_kmers(
                    min(own_length, other_length),
                    identity_threshold,
                    coverage_threshold,
                    self.k,
                )
                # The partner's own floor may be vacuous even when this one's is
                # not -- it is derived from the SHORTER length -- and in that
                # case the partner's own sweep above has already enumerated the
                # pair. Admitting it here too is harmless and keeps the two
                # directions in agreement.
                if observed >= needed:
                    pairs.add((key, other))

        stats = BlockingStats(
            sequences=len(self.keys),
            distinct_kmers=len(self._postings),
            length_band_pairs=band_pairs,
            kmer_survivors=len(pairs),
            short_sequences=len(self._short),
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            k=self.k,
        )
        return tuple(sorted(pairs)), stats


@dataclass(frozen=True)
class BlockingRecallReport:
    """Measured recall of the blocker against exhaustive all-pairs on a sample.

    The derivation in this module's docstring proves recall is 1.0. That proof is
    worth exactly as much as its implementation, which is why this exists: it
    re-runs the question empirically on a population small enough to compare
    against every pair, and the conformance artifact quotes the measurement.

    Attributes:
        population: Sequences in the audited sample.
        exhaustive_pairs: Pairs compared exhaustively.
        qualifying_pairs: Pairs that actually met the thresholds.
        recovered_pairs: Qualifying pairs the blocker also proposed.
        missed: The qualifying pairs the blocker did not propose. Must be empty.
        candidate_pairs: Pairs the blocker proposed, qualifying or not.
        identity_threshold: The threshold audited.
        coverage_threshold: The threshold audited.
    """

    population: int
    exhaustive_pairs: int
    qualifying_pairs: int
    recovered_pairs: int
    missed: Tuple[Tuple[str, str], ...]
    candidate_pairs: int
    identity_threshold: float
    coverage_threshold: float

    @property
    def recall(self) -> float:
        """Qualifying pairs recovered, over qualifying pairs. 1.0 when none exist."""
        if self.qualifying_pairs == 0:
            return 1.0
        return self.recovered_pairs / self.qualifying_pairs

    def as_dict(self) -> Dict[str, object]:
        """Render for a JSON guard report."""
        return {
            "population": self.population,
            "exhaustive_pairs": self.exhaustive_pairs,
            "qualifying_pairs": self.qualifying_pairs,
            "recovered_pairs": self.recovered_pairs,
            "missed": [list(pair) for pair in self.missed],
            "candidate_pairs": self.candidate_pairs,
            "identity_threshold": self.identity_threshold,
            "coverage_threshold": self.coverage_threshold,
            "recall": self.recall,
        }


def blocking_recall_report(
    sequences: Mapping[str, str],
    identity_threshold: float,
    coverage_threshold: float,
    *,
    k: int = DEFAULT_K,
    align=None,
    apply_length_band: bool = True,
    min_overlap: int = 0,
) -> BlockingRecallReport:
    """Audit the blocker against exhaustive all-pairs alignment.

    Only usable on a population small enough to align exhaustively; that is the
    point. `missed` non-empty is a correctness failure of this module, not a
    tuning problem.

    Args:
        sequences: Key to residue string.
        identity_threshold: Minimum matched-over-columns.
        coverage_threshold: Minimum of the two coverages.
        k: k-mer width.
        align: Pair aligner, defaulting to `alignment.align_pair`. Injectable so
            the audit can run against an independent implementation.
        apply_length_band: Must match the configuration being audited.
        min_overlap: Must match the configuration being audited, and is applied
            to the qualifying criterion too. An audit that runs a blocker
            configuration production never runs, or scores it against a
            criterion production never uses, measures a different system.

    Returns:
        A `BlockingRecallReport`.
    """
    if align is None:
        from .alignment import align_pair as align

    index = CandidateIndex(sequences, k=k)
    proposed, _ = index.candidate_pairs(
        identity_threshold, coverage_threshold,
        apply_length_band=apply_length_band, min_overlap=min_overlap,
    )
    proposed_set = set(proposed)

    keys = index.keys
    qualifying: List[Tuple[str, str]] = []
    exhaustive = 0
    for left_position, left in enumerate(keys):
        for right in keys[left_position + 1:]:
            exhaustive += 1
            result = align(sequences[left], sequences[right])
            if (
                result.identity >= identity_threshold
                and result.min_coverage >= coverage_threshold
                and result.overlap >= min_overlap
            ):
                qualifying.append((left, right))

    missed = tuple(pair for pair in qualifying if pair not in proposed_set)
    return BlockingRecallReport(
        population=len(keys),
        exhaustive_pairs=exhaustive,
        qualifying_pairs=len(qualifying),
        recovered_pairs=len(qualifying) - len(missed),
        missed=missed,
        candidate_pairs=len(proposed),
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
    )
