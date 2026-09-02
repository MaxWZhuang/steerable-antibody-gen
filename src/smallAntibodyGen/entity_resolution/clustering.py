"""
Bounded clustering: components whose diameter is a property, not a hope.

THE FAILURE THIS PREVENTS
-------------------------
Thresholded similarity plus union-find is single-linkage clustering, and
single-linkage percolates. Given ``A ~ B`` and ``B ~ C`` admitted while ``A ~ C``
is rejected, connected components merge all three anyway: a threshold constrains
adjacent edges, never component diameter. One noisy edge contaminates everything
it touches, and from outside the partition there is no way to tell a tight
component from a percolated one.

This repository has both halves of that on record. The shipped corpus contains a
component holding 12 distinct antigen sequences, 7 variant names and 11 PDB codes
whose two most distant members share 20.41% identity -- reached through name
equality rather than a similarity threshold, but the same mechanism. And a
previous redesign attempt replaced the name bridge with a similarity relation and
percolated to a 77% component on the relation the split actually keys on.

WHAT REPLACES IT
----------------
`agglomerate_complete_linkage` merges two components only when *every* cross pair
between them clears the operating thresholds. The consequence is a guarantee
rather than a tendency: every pair inside a returned cluster clears the
thresholds, so ``max_diameter <= 1 - identity_threshold`` by construction, and
`ClusterReport` publishes the realised minimum so a caller can check rather than
trust.

`single_linkage` is exported too, and it is the wrong algorithm. It exists so
that fault-injection tests can substitute it and prove the anti-percolation tests
actually fail without the criterion -- the design this package implements is
blunt about that: "A test is trustworthy only if the corresponding fault makes it
fail." Nothing in the production path calls it.

DETERMINISM
-----------
Complete-linkage agglomeration is order-sensitive under ties, so the order is
pinned rather than left to chance. Edges are processed strongest-first by
``(-identity, -min_coverage, -overlap, left_key, right_key)``, every key
comparison is on immutable strings, and no floating-point value is compared for
equality anywhere in the merge decision. Components only ever grow, so a refused
merge can never become admissible later and a single pass is complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClusterReport:
    """What one cluster is, measured rather than assumed.

    The design this package implements requires every group to report its
    members, its representative, its minimum pairwise similarity, its minimum
    pairwise coverage and its maximum diameter -- precisely so that a percolated
    component is visible from outside. A partition that cannot answer these is
    one nobody can audit.

    Attributes:
        cluster_id: The cluster's stable identifier.
        members: Member keys, sorted.
        representative: The member with the best worst-case identity to the rest;
            ``None`` only for an empty cluster.
        min_pairwise_identity: Lowest identity over all member pairs. 1.0 for a
            singleton, which is honest: a singleton has no internal distance.
        min_pairwise_coverage: Lowest ``min_coverage`` over all member pairs.
        min_pairwise_overlap: Lowest matched-residue count over all member pairs.
        max_diameter: ``1 - min_pairwise_identity``. Named separately because the
            criterion is stated as a diameter bound and a reader should not have
            to do the subtraction to check it.
        size: Number of members.
        pairs_measured: Member pairs actually aligned to produce these numbers.
        exhaustive: Whether every member pair was measured. False when the
            cluster exceeded ``max_report_pairs``, in which case the minima are
            upper bounds and must be reported as such rather than quoted.
    """

    cluster_id: str
    members: Tuple[str, ...]
    representative: Optional[str]
    min_pairwise_identity: float
    min_pairwise_coverage: float
    min_pairwise_overlap: int
    max_diameter: float
    size: int
    pairs_measured: int
    exhaustive: bool

    def as_dict(self) -> Dict[str, object]:
        """Render for a JSON guard report."""
        return {
            "cluster_id": self.cluster_id,
            "members": list(self.members),
            "representative": self.representative,
            "min_pairwise_identity": self.min_pairwise_identity,
            "min_pairwise_coverage": self.min_pairwise_coverage,
            "min_pairwise_overlap": self.min_pairwise_overlap,
            "max_diameter": self.max_diameter,
            "size": self.size,
            "pairs_measured": self.pairs_measured,
            "exhaustive": self.exhaustive,
        }


@dataclass
class Partition:
    """A disjoint partition of member keys, with the evidence for each merge.

    Attributes:
        root_of: Member key to component root key.
        members_of: Component root key to sorted member keys.
        merges_applied: Merges the criterion admitted.
        merges_refused: Merges the criterion refused because a cross pair failed.
            This is the anti-percolation criterion's activity meter: a fixture
            that leaves it at zero has not exercised the criterion, whatever else
            its assertions say.
        merges_redundant: Edges whose endpoints were already together.
    """

    root_of: Dict[str, str] = field(default_factory=dict)
    members_of: Dict[str, List[str]] = field(default_factory=dict)
    merges_applied: int = 0
    merges_refused: int = 0
    merges_redundant: int = 0

    @classmethod
    def singletons(cls, members: Iterable[str]) -> "Partition":
        """Every member in its own component.

        Args:
            members: Member keys.

        Returns:
            A `Partition` with one component per member.
        """
        partition = cls()
        for member in sorted(members):
            partition.root_of[member] = member
            partition.members_of[member] = [member]
        return partition

    def find(self, member: str) -> str:
        """The component root of ``member``."""
        return self.root_of[member]

    def members(self, member: str) -> Tuple[str, ...]:
        """Every member sharing a component with ``member``, sorted."""
        return tuple(self.members_of[self.root_of[member]])

    def components(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Every component as ``(root, members)``, sorted by root."""
        return tuple(
            (root, tuple(members))
            for root, members in sorted(self.members_of.items())
        )

    def _union(self, left_root: str, right_root: str) -> None:
        """Merge two components, keeping the lexicographically smaller root."""
        keep, drop = sorted((left_root, right_root))
        moved = self.members_of.pop(drop)
        self.members_of[keep].extend(moved)
        self.members_of[keep].sort()
        for member in moved:
            self.root_of[member] = keep

    def force_union(self, left: str, right: str) -> bool:
        """Merge two members unconditionally, for evidence that is not a threshold.

        Byte-identical sequences and shared curated accessions are hard evidence:
        there is no similarity criterion to apply to them and no cluster diameter
        to bound, because the relation is an equivalence to begin with. Keeping
        this separate from `agglomerate_complete_linkage` is what stops a hard
        merge from being mistaken for a clustered one when the counters are read.

        Args:
            left: A member key.
            right: A member key.

        Returns:
            True when a merge happened, False when the two were already together.
        """
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            self.merges_redundant += 1
            return False
        self._union(left_root, right_root)
        self.merges_applied += 1
        return True


def single_linkage(
    members: Iterable[str], edges: Iterable[Tuple[str, str]]
) -> Partition:
    """Connected components. THE WRONG ALGORITHM -- for fault injection only.

    Present so that tests can substitute it for `agglomerate_complete_linkage`
    and demonstrate that the anti-percolation assertions fail without the
    criterion. An assertion that survives this substitution is testing nothing.
    Nothing in the production path calls this.

    Args:
        members: Member keys.
        edges: Admitted ``(left, right)`` pairs.

    Returns:
        A `Partition` of connected components.
    """
    partition = Partition.singletons(members)
    for left, right in sorted(edges):
        left_root, right_root = partition.find(left), partition.find(right)
        if left_root == right_root:
            partition.merges_redundant += 1
            continue
        partition._union(left_root, right_root)
        partition.merges_applied += 1
    return partition


def agglomerate_complete_linkage(
    members: Iterable[str],
    edges: Sequence[Tuple[str, str]],
    qualifies: Callable[[str, str], bool],
    rank: Optional[Callable[[str, str], tuple]] = None,
    partition: Optional[Partition] = None,
) -> Partition:
    """Merge components only when every cross pair clears the operating point.

    The resulting guarantee is what single-linkage cannot offer: for any two
    members of a returned cluster, ``qualifies`` is true. Diameter is therefore
    bounded by the threshold rather than by the length of the longest chain
    through the data.

    Args:
        members: Member keys.
        edges: Admitted ``(left, right)`` pairs. Order is irrelevant; they are
            sorted by ``rank`` before use.
        qualifies: ``(left, right) -> bool``, true when that pair clears the
            operating point. Called for cross pairs that may never have been
            candidate edges, so it must answer for any pair -- returning False
            for a pair the blocker never proposed is correct, because a pair that
            could have qualified would have been proposed.
        rank: Sort key for an edge, strongest first. Defaults to sorting by the
            pair keys alone, which is deterministic but ignores edge strength;
            callers with alignment evidence should pass a real ranking.
        partition: Optional starting partition, for the case where hard evidence
            has already merged some members before any threshold is applied. The
            completeness check then runs over the pre-merged components, which is
            what makes a hard merge a constraint on later clustering rather than
            something clustering can quietly undo.

    Returns:
        A `Partition` whose ``merges_refused`` counts the times the criterion
        actually blocked a merge -- the number a fixture must be able to show is
        non-zero before claiming it exercised anti-percolation.
    """
    if partition is None:
        partition = Partition.singletons(members)
    ordered = sorted(edges, key=rank) if rank is not None else sorted(edges)

    for left, right in ordered:
        left_root, right_root = partition.find(left), partition.find(right)
        if left_root == right_root:
            partition.merges_redundant += 1
            continue
        left_members = partition.members_of[left_root]
        right_members = partition.members_of[right_root]
        if all(
            qualifies(one, other) for one in left_members for other in right_members
        ):
            partition._union(left_root, right_root)
            partition.merges_applied += 1
        else:
            partition.merges_refused += 1
    return partition


def cluster_report(
    cluster_id: str,
    members: Sequence[str],
    measure: Callable[[str, str], Optional[object]],
    *,
    max_report_pairs: int = 20_000,
    tie_break: Optional[Callable[[str], tuple]] = None,
) -> ClusterReport:
    """Measure a cluster's internal geometry.

    Args:
        cluster_id: The cluster's identifier.
        members: Member keys.
        measure: ``(left, right) -> alignment or None``. ``None`` means the pair
            has no measurable relation, which drives the minima to zero -- the
            honest answer for a cluster held together by something other than
            sequence.
        max_report_pairs: Above this many member pairs, measure a deterministic
            subset and mark the report non-exhaustive rather than either hanging
            or silently reporting a partial minimum as if it were total.
        tie_break: Optional sort key used to pick the representative among
            equally good candidates. Defaults to the member key.

    Returns:
        A `ClusterReport`. A singleton reports identity 1.0 and diameter 0.0.
    """
    ordered = tuple(sorted(members))
    size = len(ordered)
    if size == 0:
        return ClusterReport(
            cluster_id=cluster_id, members=(), representative=None,
            min_pairwise_identity=1.0, min_pairwise_coverage=1.0,
            min_pairwise_overlap=0, max_diameter=0.0, size=0,
            pairs_measured=0, exhaustive=True,
        )
    if size == 1:
        return ClusterReport(
            cluster_id=cluster_id, members=ordered, representative=ordered[0],
            min_pairwise_identity=1.0, min_pairwise_coverage=1.0,
            min_pairwise_overlap=0,
            max_diameter=0.0, size=1, pairs_measured=0, exhaustive=True,
        )

    total_pairs = size * (size - 1) // 2
    exhaustive = total_pairs <= max_report_pairs
    # When the cluster is too large to measure exhaustively, take a deterministic
    # stride through the pair list rather than a random sample, so the number in
    # the report is reproducible.
    stride = 1 if exhaustive else max(1, total_pairs // max_report_pairs)

    worst_identity = 1.0
    worst_coverage = 1.0
    worst_overlap = None
    worst_to: Dict[str, float] = {member: 1.0 for member in ordered}
    measured = 0
    counter = 0
    for left_position, left in enumerate(ordered):
        for right in ordered[left_position + 1:]:
            counter += 1
            if counter % stride:
                continue
            measured += 1
            result = measure(left, right)
            identity = 0.0 if result is None else result.identity
            coverage = 0.0 if result is None else result.min_coverage
            overlap = 0 if result is None else result.overlap
            worst_identity = min(worst_identity, identity)
            worst_coverage = min(worst_coverage, coverage)
            worst_overlap = overlap if worst_overlap is None else min(worst_overlap, overlap)
            worst_to[left] = min(worst_to[left], identity)
            worst_to[right] = min(worst_to[right], identity)

    key = tie_break or (lambda member: (member,))
    representative = min(ordered, key=lambda member: (-worst_to[member],) + key(member))

    return ClusterReport(
        cluster_id=cluster_id,
        members=ordered,
        representative=representative,
        min_pairwise_identity=worst_identity,
        min_pairwise_coverage=worst_coverage,
        min_pairwise_overlap=0 if worst_overlap is None else worst_overlap,
        max_diameter=1.0 - worst_identity,
        size=size,
        pairs_measured=measured,
        exhaustive=exhaustive,
    )


def transitive_closure(
    members: Iterable[str], edges: Iterable[Tuple[str, str]]
) -> Partition:
    """Close a quarantine relation transitively.

    Hard co-assignment constraints MUST percolate: if A must share a side with B
    and B with C, all three must share a side. So single-linkage has not
    disappeared from this package, it has moved to the one relation where it is
    correct -- and it is correct there only because the evidence-to-constraint
    policy upstream has already decided which observations deserve hard status.
    That is why this is a separate function with its own name rather than a call
    to `single_linkage`: the two are the same algorithm doing opposite jobs, and
    conflating them is how quarantine evidence turns into identity.

    Args:
        members: Member keys.
        edges: Hard co-assignment constraints.

    Returns:
        A `Partition` of transitively closed constraint components.
    """
    return single_linkage(members, edges)
