"""
Typed target identity: three relations, computed separately, never collapsed.

WHAT THIS REPLACES
------------------
The producer's committed rule (`scripts/prepare_antibody_antigen.py`) treats
target identity as one union-find over namespaced string nodes: a UniProt
accession, a PDB code, a normalised target name and a SHA-256 of the antigen
sequence, joined whenever they co-occur on a row. One partition, one id, one
split. Measured on the shipped corpus that rule fuses IL23A with IL12B because
both appear in PDB entry 4GRW, fuses four non-homologous SARS-CoV-2 spike
constructs because a curator wrote ``sars-cov2_wt`` on all of them, and holds the
287-aa and 289-aa Omicron NTDs together only through a name -- delete the name and
the family dissolves, because that rule has no similarity relation at all.

The replacement is three relations with three different jobs:

===================  ==========================================================
`construct_id`       The same concrete construct. Near-exact sequence identity
                     with near-full reciprocal coverage.
`biological_target_id`
                     The same biological target. Shared accession, an approved
                     name, or family-level sequence similarity.
`quarantine_partners`
                     Records that must not straddle a split, without being
                     claimed to be the same thing: local containment, a shared
                     structural container, an unapproved name.
===================  ==========================================================

Deriving all three from one graph is the failure this module exists to prevent.
Local containment has to be able to quarantine the 107-aa IGKC constant domain
against the 1057-aa Fab-HER2 fusion it sits inside WITHOUT ever calling them one
target; a shared UniProt accession has to be able to place two constructs in one
family WITHOUT calling them one construct. One partition cannot say both.

THE EVIDENCE-TO-ACTION TABLE
----------------------------
Raw resemblance is not automatically a hard edge. Every observation lands in
exactly one row of this table, and the row decides what it may do:

===  ==========================================  ===========  ======  ==========
ID   Evidence                                    Construct    Family  Quarantine
===  ==========================================  ===========  ======  ==========
E1   byte-identical antigen sequence             merge        merge   --
E2   near-identical construct                    merge        merge   --
E3   family-level similarity                     --           merge   --
E4   local containment / partial overlap         --           --      link
E5   same UniProt accession                      --           merge   --
E6   same PDB entry                              --           --      link
E7   same normalised name, APPROVED              --           merge   --
E8   same normalised name, NOT approved          --           --      link
E9   one sequence, conflicting accessions        --           --      reported
===  ==========================================  ===========  ======  ==========

E6 is demoted deliberately. A PDB entry is a complex containing several distinct
polymer chains, so sharing one is co-occurrence, not identity -- IL23A and IL12B
share 4GRW and share 24.78% of their residues and no 8-mer at all. E4 is demoted
for the same reason in the other direction: containment is asymmetric and
identity is symmetric, so "A occurs inside B" can never be "A is B".

E8 is the interesting one. Deleting names as a merging force is not the fix
either -- a previous attempt did that and changed the partition by exactly zero
components, because it froze the sequence components first and then appended
names as inert labels. Here a name that spans a *coherent* set of constructs
genuinely merges them (E7), a name that spans an incoherent set quarantines them
instead (E8), and the decision is computed ONCE against the frozen
sequence/accession groups and never iterated to a fixed point, because iterative
name attachment recreates exactly the transitive chaining the bounded clustering
is there to stop.

ANTI-PERCOLATION
----------------
Every similarity relation here is clustered with `agglomerate_complete_linkage`,
not with connected components. See `entity_resolution.clustering` for why, and
for the fault-injection hook that proves the tests notice when it is removed.
The quarantine relation IS transitively closed -- co-assignment constraints must
percolate or they do not constrain -- which is why it is a separate graph reached
only after the evidence table above has decided what deserves hard status.

LABEL BLINDNESS
---------------
`row_identity_view` reads three metadata fields and the antigen sequence, and
nothing else reaches the resolver. Supervision cannot influence which rows are
grouped and therefore cannot influence the split. `FORBIDDEN_LABEL_FIELDS` names
what is deliberately unreachable; the acceptance suite attaches every one of them
to every fixture row with violently disagreeing values and requires byte-identical
output.

DETERMINISM
-----------
Ids are functions of sorted digests, never of arrival order. Clustering processes
edges in a pinned strongest-first order. Blocking iterates a sorted population.
Two runs over the same rows in any order produce the same partition, the same
ids, and the same reports.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional,
    Sequence, Set, Tuple,
)

from .entity_resolution.alignment import (
    AlignmentTooLarge, PairAlignment, align_pair,
)
from .entity_resolution.blocking import CandidateIndex
from .entity_resolution.clustering import (
    Partition, agglomerate_complete_linkage, cluster_report, transitive_closure,
)

#: The only metadata fields identity may read.
IDENTITY_METADATA_FIELDS: Tuple[str, ...] = (
    "target_name", "target_pdb", "target_uniprot",
)

#: Fields that must never influence identity. Not merely unused -- unreachable:
#: `row_identity_view` whitelists, so adding a field here without adding it to
#: `IDENTITY_METADATA_FIELDS` keeps it out by construction.
FORBIDDEN_LABEL_FIELDS: FrozenSet[str] = frozenset({
    "affinity", "affinity_raw", "affinity_type", "binder_label", "confidence",
    "is_strong_binder", "processed_measurement", "processed_measurement_float",
    "processed_measurement_raw", "split",
})

#: A run of at least eight glycines is the linker signature of a concatenated
#: construct. 28 of the corpus's 9,574 distinct antigens carry one, and the
#: 1057-aa entry that fuses a Fab to the HER2 ectodomain is the reason this
#: matters: a composite contains several unrelated things, so its similarity to
#: any one of them is evidence about that part and not about the whole.
COMPOSITE_LINKER = re.compile(r"G{8,}")

#: Namespaces that may NAME a biological family, strongest first, and the ONLY
#: ones. `pdb` is absent on purpose: it is quarantine evidence under E6, and a
#: namespace that is not allowed to merge must not be allowed to name either --
#: the committed rule calls a four-construct SARS-CoV-2 component ``pdb:7ch5``
#: for a population whose rows mostly carry no PDB code at all.
#:
#: A component whose strongest present namespace holds more than one candidate
#: gets a neutral ``family:<hash>`` instead, because choosing a winner where the
#: sources disagree is the defect `AccessionConflict` exists to report.
FAMILY_NAMESPACE_RANK: Tuple[str, ...] = ("uniprot", "name", "seq")


@dataclass(frozen=True)
class OperatingPoint:
    """Every threshold that can move a record from one side of a split to the other.

    Sealed as a unit and reported with every artifact, because a threshold quoted
    without the population and the metric it was measured in is not a number
    anybody can check. All identities and coverages are in the local affine
    Smith-Waterman metric of `entity_resolution.alignment`; a number measured
    with a different aligner is not comparable to these and must not be
    substituted for one.

    Attributes:
        construct_identity: Minimum identity for E2, "the same construct".
        construct_coverage: Minimum of the two coverages for E2. The 564-aa HER2
            ECD is a byte-exact sub-region of the 607-aa one -- identity 1.0 --
            and they are still different constructs, so it is coverage (0.9292)
            and not identity that has to separate them.
        construct_overlap: Minimum matched residues for E2.
        family_identity: Minimum identity for E3, "the same target".
        family_coverage: Minimum of the two coverages for E3.
        family_overlap: Minimum matched residues for E3.
        containment_identity: Minimum identity for E4.
        containment_coverage: Minimum coverage of the CONTAINED side for E4.
        containment_max_coverage: Coverage of the containing side must stay
            BELOW this, or the pair is a family relation rather than a
            containment one.
        containment_overlap: Minimum matched residues for E4.
        name_bridge_identity: Minimum identity for a name to be allowed to merge
            two otherwise-separate groups (E7 rather than E8). Looser than the
            family threshold on purpose -- a curator's name is weaker evidence
            than a sequence, so it must clear a bar, but demanding the family bar
            would make every name inert and reproduce the failure this design
            replaced.
        name_bridge_coverage: Minimum of the two coverages for that bridge.
            Without it, two constructs sharing one 120-residue motif at 25%
            reciprocal coverage clear the bridge while the family relation
            refuses them at 0.80 -- so a curator's name would buy a merge that
            the sequence evidence explicitly denies.
        name_bridge_overlap: Minimum matched residues for that bridge.
    """

    construct_identity: float = 0.99
    construct_coverage: float = 0.95
    construct_overlap: int = 30
    family_identity: float = 0.90
    family_coverage: float = 0.80
    family_overlap: int = 30
    containment_identity: float = 0.95
    containment_coverage: float = 0.95
    containment_max_coverage: float = 0.80
    containment_overlap: int = 30
    name_bridge_identity: float = 0.75
    name_bridge_coverage: float = 0.50
    name_bridge_overlap: int = 30

    def as_dict(self) -> Dict[str, object]:
        """Render for a sealed manifest."""
        return {
            "metric": "local affine Smith-Waterman, BLOSUM62, open 11 extend 1",
            "construct": {
                "identity": self.construct_identity,
                "coverage": self.construct_coverage,
                "overlap": self.construct_overlap,
            },
            "family": {
                "identity": self.family_identity,
                "coverage": self.family_coverage,
                "overlap": self.family_overlap,
            },
            "containment": {
                "identity": self.containment_identity,
                "coverage": self.containment_coverage,
                "max_coverage": self.containment_max_coverage,
                "overlap": self.containment_overlap,
            },
            "name_bridge": {
                "identity": self.name_bridge_identity,
                "coverage": self.name_bridge_coverage,
                "overlap": self.name_bridge_overlap,
            },
        }


DEFAULT_OPERATING_POINT = OperatingPoint()


@dataclass(frozen=True)
class NameDecision:
    """Whether one normalised name was allowed to merge, and why.

    A name that merges silently and a name that is ignored silently are equally
    unauditable. Every name the resolver considered produces one of these,
    approved or not.

    Attributes:
        name: The normalised name.
        approved: Whether it merged (E7) or quarantined (E8).
        attached_to: The family id it attached to when approved, else ``None``.
        reason: Why, in words, referencing the measurement that decided it.
        spanned_groups: The pre-name family groups the name was written across.
        worst_identity: Weakest cross-group identity the name would have bridged;
            ``None`` when it spanned fewer than two groups.
        worst_overlap: Matched residues of that weakest bridge.
        composite_members: Members excluded from the bridge evidence for being
            concatenated constructs.
    """

    name: str
    approved: bool
    attached_to: Optional[str]
    reason: str
    spanned_groups: Tuple[str, ...] = ()
    worst_identity: Optional[float] = None
    worst_overlap: Optional[int] = None
    composite_members: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AccessionConflict:
    """One antigen sequence carrying more than one accession, with no winner chosen.

    Two accessions written over one byte-identical sequence is a disagreement in
    the source data. The resolver's job is to report it, not to settle it: the
    committed rule takes ``min`` over the node strings, so ``uniprot:p63000``
    beats ``uniprot:p63001`` on lexicographic order alone and the loser vanishes
    with nothing anywhere counting the loss.

    Attributes:
        antigen_sha256: Digest of the disputed sequence.
        accessions: Every accession written over it, sorted.
        names: Every name written over it, sorted.
        rows: Source rows behind the disagreement.
        resolved_id: The neutral id the component received instead.
    """

    antigen_sha256: str
    accessions: Tuple[str, ...]
    names: Tuple[str, ...]
    rows: int
    resolved_id: str


@dataclass(frozen=True)
class QuarantineEdge:
    """One co-assignment constraint, with the evidence class that created it.

    Attributes:
        left: Construct id, sorted first.
        right: Construct id, sorted second.
        kind: ``containment``, ``shared_container`` or ``ambiguous_name``.
        evidence: Human-readable evidence for the constraint.
    """

    left: str
    right: str
    kind: str
    evidence: str


@dataclass(frozen=True)
class ErrorReport:
    """False merges and false splits over a labelled population.

    Both kinds are reported because they cost different things. A false merge
    destroys target diversity, corrupts every per-target number computed
    downstream, and is invisible to a leakage audit -- a merged pair never
    straddles anything. A false split costs leakage, which a leakage audit does
    detect. `tolerated_error` records which direction was declared acceptable
    BEFORE any of this was measured, so nobody can choose it afterwards.

    Attributes:
        name: Which population this is -- ``calibration`` or ``audit``.
        families: The labelled families in this population, sorted.
        pairs: Labelled pairs scored.
        positive_pairs: Pairs whose ground truth is "same target".
        negative_pairs: Pairs whose ground truth is "different target".
        false_merges: Negative pairs the resolver put in one family.
        false_splits: Positive pairs the resolver put in different families.
        tolerated_error: The predeclared asymmetry.
        unadjudicated: Pairs excluded from scoring because the right answer was
            not settled in advance. Counted, never silently dropped.
        false_merge_examples: Up to ten witnesses.
        false_split_examples: Up to ten witnesses.
    """

    name: str
    families: Tuple[str, ...]
    pairs: int
    positive_pairs: int
    negative_pairs: int
    false_merges: int
    false_splits: int
    tolerated_error: str
    unadjudicated: int = 0
    false_merge_examples: Tuple[Tuple[str, str], ...] = ()
    false_split_examples: Tuple[Tuple[str, str], ...] = ()

    def as_dict(self) -> Dict[str, object]:
        """Render for a JSON guard report."""
        return {
            "name": self.name,
            "families": list(self.families),
            "pairs": self.pairs,
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "false_merges": self.false_merges,
            "false_splits": self.false_splits,
            "tolerated_error": self.tolerated_error,
            "unadjudicated": self.unadjudicated,
            "false_merge_examples": [list(p) for p in self.false_merge_examples],
            "false_split_examples": [list(p) for p in self.false_split_examples],
        }


# --------------------------------------------------------------------------- #
# Normalisation. Deliberately identical to the producer's, so that switching
# engines changes which records group together and never what an identifier
# means.
# --------------------------------------------------------------------------- #

def clean_text(value: object) -> str:
    """Trim a possibly-missing text field to a string."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def normalize_target_name(text: object) -> str:
    """Lowercase and punctuation-normalise a target name.

    Args:
        text: Raw name.

    Returns:
        The name with runs of non-alphanumerics collapsed to ``_``, stripped.
        ``sars-cov2_wt`` becomes ``sars_cov2_wt``.
    """
    lowered = clean_text(text).lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def canonicalize_accession(value: object) -> str:
    """Normalise a UniProt or PDB accession for grouping.

    Drops a chain or assembly suffix introduced by a separator (``6XYZ_A`` and
    ``6xyz.A`` both become ``6xyz``) and a trailing UniProt isoform or version
    suffix (``P12345-2`` becomes ``p12345``). Neither namespace contains an
    internal ``_`` or ``.``, so this is safe for both.

    Args:
        value: Raw accession.

    Returns:
        The canonical accession, lowercase, or ``""``.
    """
    text = clean_text(value).lower()
    if not text:
        return ""
    text = re.split(r"[\s_.]", text, maxsplit=1)[0]
    return re.sub(r"-\d+$", "", text)


def clean_antigen_sequence(value: object) -> str:
    """Upper-case a residue string and drop everything that is not a residue."""
    text = clean_text(value).upper()
    return re.sub(r"[^A-Z*]", "", text)


def antigen_digest(sequence: str) -> str:
    """The stable key for one antigen sequence.

    Args:
        sequence: Cleaned residue string.

    Returns:
        The first 32 hex characters of its SHA-256. Truncated SHA-1 is
        deliberately not used: this key participates in merging, so an accidental
        collision would fuse two real targets.
    """
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:32]


def row_identity_view(row: Mapping[str, object]) -> Tuple[Dict[str, str], str]:
    """Reduce a source row to the only things identity may see.

    Accepts both shapes the repository produces: a raw shard row with the target
    fields nested under ``metadata`` and its antigen under ``antigen_sequence``,
    and a PROCESSED record with the target fields at the top level and its
    antigen under ``sequence_antigen`` -- the producer renames the field on the
    way out (`prepare_antibody_antigen.py`, ``build_processed_record``). Reading
    only the raw spelling made this silently return an empty sequence for every
    processed record, which would have put every one of them in its own
    identity-less group. Anything else in the row -- affinities, labels,
    confidences, an existing split -- is not read, and cannot be, because this
    whitelists.

    Args:
        row: A mapping carrying the antigen under either spelling, plus the
            target fields.

    Returns:
        ``(fields, antigen_sequence)`` where ``fields`` has exactly the keys in
        `IDENTITY_METADATA_FIELDS`.
    """
    nested = row.get("metadata")
    source: Mapping[str, object] = nested if isinstance(nested, Mapping) else row
    fields = {name: clean_text(source.get(name)) for name in IDENTITY_METADATA_FIELDS}
    antigen = row.get("antigen_sequence")
    if antigen is None:
        antigen = row.get("sequence_antigen")
    return fields, clean_antigen_sequence(antigen)


# --------------------------------------------------------------------------- #
# The resolution
# --------------------------------------------------------------------------- #

class TargetIdentityResolution:
    """The three relations over one population of rows, with their evidence.

    Built by `resolve_target_identity`; not constructed directly. Every accessor
    takes a "key or row": an antigen digest, a mapping shaped like a source row,
    or any object exposing ``antigen_sequence``, so callers do not have to know
    which of those they are holding.
    """

    def __init__(
        self,
        *,
        operating_point: OperatingPoint,
        sequences: Mapping[str, str],
        rows_of: Mapping[str, int],
        names_of: Mapping[str, Set[str]],
        pdbs_of: Mapping[str, Set[str]],
        accessions_of: Mapping[str, Set[str]],
        composites: Set[str],
        alignments: Mapping[Tuple[str, str], PairAlignment],
        construct_partition: Partition,
        family_partition: Partition,
        name_decisions: Sequence[NameDecision],
        accession_conflicts: Sequence[AccessionConflict],
        quarantine_edges: Sequence[QuarantineEdge],
        edge_counts: Mapping[str, int],
        rows_seen: int,
        rows_without_antigen: int,
        alignments_refused: int,
        blocking_stats: Sequence[Mapping[str, object]],
        curated_labels: Mapping[str, "CuratedLabel"],
        audit_families: Sequence[str],
        unadjudicated_pairs: FrozenSet[Tuple[str, str]],
        tolerated_error: str,
        max_cells: Optional[int] = None,
        max_container_span: int = 8,
        max_split_group_families: int = 12,
        high_degree_bridges: Sequence[Mapping[str, object]] = (),
        curator_merged: FrozenSet[str] = frozenset(),
        bridge_members: FrozenSet[str] = frozenset(),
    ) -> None:
        self.operating_point = operating_point
        self._sequences = dict(sequences)
        self._rows_of = dict(rows_of)
        self._names_of = {k: set(v) for k, v in names_of.items()}
        self._pdbs_of = {k: set(v) for k, v in pdbs_of.items()}
        self._accessions_of = {k: set(v) for k, v in accessions_of.items()}
        self._composites = set(composites)
        self._alignments = dict(alignments)
        self._construct = construct_partition
        self._family = family_partition
        self._name_decisions = tuple(name_decisions)
        self._accession_conflicts = tuple(accession_conflicts)
        self._quarantine_edges = tuple(quarantine_edges)
        self._edge_counts = dict(edge_counts)
        self._rows_seen = rows_seen
        self._rows_without_antigen = rows_without_antigen
        self._alignments_refused = alignments_refused
        self._blocking_stats = tuple(blocking_stats)
        self._curated_labels = dict(curated_labels)
        self._audit_families = tuple(audit_families)
        self._unadjudicated = unadjudicated_pairs
        self._tolerated_error = tolerated_error
        self._max_cells = max_cells
        self._max_container_span = max_container_span
        self._max_split_group_families = max_split_group_families
        self._high_degree_bridges = tuple(dict(b) for b in high_degree_bridges)
        #: Digests whose family membership rests on a curated identifier -- a
        #: shared accession or an approved name -- rather than on the similarity
        #: criterion. Their components have no diameter guarantee, because a
        #: curator asserting that two constructs are one target outranks what
        #: their residues say, and that has to be visible in the reports rather
        #: than quietly widening a number the criterion is supposed to bound.
        self._curator_merged = frozenset(curator_merged)
        #: Digests belonging to a container the span guard refused to link. They
        #: are test-ineligible rather than unconstrained -- see `add_container`.
        self._bridge_members = frozenset(bridge_members)

        self._construct_id_of = _component_ids(self._construct, "construct")
        self._family_id_of = self._name_families()
        self._partners = self._direct_partners()
        self._ineligible = self._test_ineligible_constructs()
        self._split_roots = self._close_quarantine()
        self._percolated_groups = self._flag_percolated_closures()
        self._calibration: Optional[ErrorReport] = None

    # ---------------------------------------------------------------- lookup

    def digest_for(self, key_or_row: Any) -> str:
        """Resolve a digest, a row mapping, or a record object to an antigen digest.

        Args:
            key_or_row: A 32-character digest, a mapping carrying
                ``antigen_sequence``, or any object exposing that attribute.

        Returns:
            The antigen digest.

        Raises:
            KeyError: When the resulting digest is not in this resolution.
        """
        if isinstance(key_or_row, str):
            digest = key_or_row
        elif isinstance(key_or_row, Mapping):
            digest = antigen_digest(row_identity_view(key_or_row)[1])
        else:
            sequence = getattr(key_or_row, "antigen_sequence", None)
            if sequence is None:
                raise KeyError(f"cannot derive an antigen digest from {key_or_row!r}")
            digest = antigen_digest(clean_antigen_sequence(sequence))
        if digest not in self._sequences:
            raise KeyError(f"digest {digest!r} was not observed by this resolution")
        return digest

    def antigen_digests(self) -> Tuple[str, ...]:
        """Every distinct antigen sequence observed, as sorted digests."""
        return tuple(sorted(self._sequences))

    def antigen_sequence(self, key_or_row: Any) -> str:
        """The residue string behind a key."""
        return self._sequences[self.digest_for(key_or_row)]

    def rows_for(self, key_or_row: Any) -> int:
        """Source rows behind one antigen.

        Concentration has to be measurable in rows and not only in distinct
        antigens: one target carries 63% of this corpus's rows while being one
        antigen among 9,574, so a share counted in antigens is blind to exactly
        the imbalance that matters.
        """
        return self._rows_of[self.digest_for(key_or_row)]

    def is_composite(self, key_or_row: Any) -> bool:
        """Whether this antigen carries a poly-glycine linker run.

        A composite is a container, not a target. It may quarantine with each of
        its parts and it may not bridge them.
        """
        return self.digest_for(key_or_row) in self._composites

    # ------------------------------------------------------------- relations

    def construct_id(self, key_or_row: Any) -> str:
        """The id of the near-exact construct cluster this record belongs to."""
        return self._construct_id_of[self.digest_for(key_or_row)]

    def biological_target_id(self, key_or_row: Any) -> str:
        """The id of the biological target this record belongs to.

        Named after the component's strongest unambiguous identifier: a UniProt
        accession, else an approved name, else the antigen sequence itself. A
        component whose strongest namespace holds more than one candidate gets a
        neutral ``family:`` id instead of whichever candidate sorts first --
        choosing a winner where the sources disagree is the failure
        `accession_conflicts` exists to surface.
        """
        return self._family_id_of[self._family.find(self.digest_for(key_or_row))]

    def quarantine_partners(self, key_or_row: Any) -> FrozenSet[str]:
        """Construct ids that must share a side with this one, without being it.

        DIRECT partners only. Quarantine is between an overlap and its container,
        not transitive between two things that merely share a container: the
        107-aa IGKC domain and the 607-aa HER2 ectodomain both sit inside the
        same 1057-aa fusion and are not thereby related to each other. The
        transitive closure that a split needs is `split_group_id`'s job.
        """
        return self._partners.get(self.construct_id(key_or_row), frozenset())

    def split_group_id(self, key_or_row: Any, claim: str = "generic") -> str:
        """The indivisible unit for a split, under a named claim.

        A generic held-out-target split and a predeclared unseen-mutant benchmark
        are different claims about what "unseen" means, and one grouping cannot
        serve both. For a generic split, an antibody repeated across two
        near-identical antigens is contamination and the two must stay together;
        for an unseen-mutant benchmark that repetition is the intended
        counterfactual and they must be separable. Rows produced under the two
        claims must not quietly coexist in one validation metric.

        Args:
            key_or_row: The record.
            claim: ``"generic"`` -- the quarantine-closed family component; or
                ``"unseen_mutant"`` -- the exact antigen sequence, because a
                point mutant is the axis that benchmark is about and closing over
                it would erase the thing being measured.

        Returns:
            The group id.

        Raises:
            ValueError: On an unknown claim. Claims are a controlled vocabulary;
                accepting an unrecognised one would silently give whatever
                grouping the default happens to be and call it a benchmark.
        """
        digest = self.digest_for(key_or_row)
        if claim == "unseen_mutant":
            return f"mutant:{digest}"
        if claim == "generic":
            return self._split_roots[self.construct_id(digest)]
        raise ValueError(
            f"unknown claim {claim!r}; expected 'generic' or 'unseen_mutant'"
        )

    def test_ineligible(self, key_or_row: Any) -> bool:
        """Whether this record may be used for training but not scored in test.

        Aggregate evidence produces test-ineligibility, never a merge. A
        concatenated construct is reachable from several unrelated targets at
        once, and making it a hard link between them would merge those targets
        and recreate exactly the percolation the bounded clustering prevents. The
        conservative alternative is to keep it out of scored evaluation and leave
        every other component intact.
        """
        return self.construct_id(key_or_row) in self._ineligible

    # --------------------------------------------------------------- reports

    def construct_cluster_report(self, construct_id: str):
        """Measure one construct cluster's internal geometry."""
        return self._report(construct_id, self._construct, self._construct_id_of)

    def family_cluster_report(self, family_id: str):
        """Measure one biological family's internal geometry."""
        return self._report(family_id, self._family, self._family_id_of, by_root=True)

    def name_decisions(self) -> Tuple[NameDecision, ...]:
        """Every normalised name considered, approved or not, with its reason."""
        return self._name_decisions

    def accession_conflicts(self) -> Tuple[AccessionConflict, ...]:
        """Every antigen sequence carrying accessions that disagree."""
        return self._accession_conflicts

    def unclosed_constraints(self) -> Tuple[Mapping[str, object], ...]:
        """Co-assignment constraints that were recorded and NOT enforced.

        Each one is a residual exposure: two constructs the evidence says must
        share a side, which the split may nonetheless separate because closing
        the constraint would have routed it through a container and welded two
        unrelated families. Refusing to close is the right trade; pretending it
        is free is not. A producer adopting this engine resolves each of these by
        EXCLUDING the container, which the design's policy table allows and which
        costs only the container's own rows.
        """
        return self._unclosed_constraints

    def quarantine_edge_list(self) -> Tuple[QuarantineEdge, ...]:
        """Every co-assignment constraint, with the evidence class behind it."""
        return self._quarantine_edges

    def audit_report(self) -> ErrorReport:
        """Error rates on the curated real families, which set no thresholds.

        The operating point is calibrated on an independently generated synthetic
        population (`calibration_report`) and audited here, on families it never
        saw. An operating point chosen on a set and then validated on the same set
        reports its own selection back as an achieved error rate.
        """
        labelled = {
            digest: label
            for digest, label in self._curated_labels.items()
            if digest in self._sequences and label.family in self._audit_families
        }
        return self._score("audit", labelled, self.biological_target_id)

    def calibration_report(self) -> ErrorReport:
        """Error rates on the synthetic population the operating point was chosen on.

        Generated by `entity_resolution.synthetic`, which shares no code with this
        module and plants its own ground truth, so the families here cannot
        overlap the curated real ones and the two reports are quotable
        separately. Computed lazily and cached, because most callers want the
        partition and not the calibration.
        """
        if self._calibration is None:
            from .entity_resolution.synthetic import calibration_population

            population = calibration_population()
            resolved = resolve_target_identity(
                population.rows,
                operating_point=self.operating_point,
                curated_labels={},
                audit_families=(),
                unadjudicated_pairs=frozenset(),
                tolerated_error=self._tolerated_error,
                # The calibration must describe the run that publishes it. An
                # earlier version let these fall back to their defaults while the
                # caller had set them and sealed them into the manifest, so the
                # error rates were measured under a configuration nobody ran.
                max_cells=self._max_cells,
                max_container_span=self._max_container_span,
                max_split_group_families=self._max_split_group_families,
            )
            labelled = {
                antigen_digest(sequence): CuratedLabel(
                    family=label.family, target=label.target, note=label.note
                )
                for sequence, label in population.truth.items()
            }
            self._calibration = resolved._score(
                "calibration", labelled, resolved.biological_target_id
            )
        return self._calibration

    def stats(self) -> Dict[str, object]:
        """Everything a guard report needs to decide whether this partition is sane.

        Includes the two numbers a thresholded union-find cannot produce at all --
        the worst component's minimum pairwise identity and coverage. Without
        them a percolated component is indistinguishable from a tight one from
        outside, which is how a component holding 12 antigens at 20.41% minimum
        identity survived in the shipped corpus with nothing anywhere saying so.
        """
        construct_reports = [
            self.construct_cluster_report(cluster_id)
            for cluster_id in sorted(set(self._construct_id_of.values()))
        ]
        family_reports = [
            self._report(family_id, self._family, self._family_id_of, by_root=True)
            for family_id in sorted(set(self._family_id_of.values()))
        ]
        multi = [report for report in family_reports if report.size > 1]
        # The criterion's guarantee applies only where the criterion decided.
        # A component containing a curator-merged member was joined by an
        # accession or an approved name, which outranks sequence similarity by
        # design and therefore cannot be bounded by a similarity threshold.
        # Reporting one number over both kinds would either understate the
        # guarantee or overstate it, so both are reported.
        criterion_only = [
            report for report in multi
            if not (set(report.members) & self._curator_merged)
        ]
        rows_total = max(1, sum(self._rows_of.values()))

        group_rows: Dict[str, int] = {}
        for digest in self._sequences:
            group = self.split_group_id(digest)
            group_rows[group] = group_rows.get(group, 0) + self._rows_of[digest]
        largest_group = max(group_rows.values()) if group_rows else 0

        family_rows: Dict[str, int] = {}
        for digest in self._sequences:
            family = self.biological_target_id(digest)
            family_rows[family] = family_rows.get(family, 0) + self._rows_of[digest]
        largest_family = max(family_rows.values()) if family_rows else 0

        return {
            "target_rows_seen": self._rows_seen,
            "target_rows_without_antigen": self._rows_without_antigen,
            "target_distinct_antigens": len(self._sequences),
            "target_constructs": len(set(self._construct_id_of.values())),
            "target_families": len(set(self._family_id_of.values())),
            "target_split_groups": len(group_rows),
            "target_composite_antigens": len(self._composites),
            "target_test_ineligible_constructs": len(self._ineligible),
            "target_accession_conflicts": len(self._accession_conflicts),
            "target_names_considered": len(self._name_decisions),
            "target_names_approved": sum(1 for d in self._name_decisions if d.approved),
            "target_names_quarantined": sum(
                1 for d in self._name_decisions if not d.approved
            ),
            "target_quarantine_edges": len(self._quarantine_edges),
            "target_construct_merges_refused": self._construct.merges_refused,
            "target_family_merges_refused": self._family.merges_refused,
            "target_alignments_measured": len(self._alignments),
            "target_alignments_refused_too_large": self._alignments_refused,
            "component_min_pairwise_identity": (
                min(r.min_pairwise_identity for r in multi) if multi else 1.0
            ),
            "component_min_pairwise_coverage": (
                min(r.min_pairwise_coverage for r in multi) if multi else 1.0
            ),
            "component_max_diameter": (
                max(r.max_diameter for r in multi) if multi else 0.0
            ),
            "component_curator_merged": len(multi) - len(criterion_only),
            "criterion_component_count": len(criterion_only),
            "criterion_min_pairwise_identity": (
                min(r.min_pairwise_identity for r in criterion_only)
                if criterion_only else 1.0
            ),
            "criterion_min_pairwise_coverage": (
                min(r.min_pairwise_coverage for r in criterion_only)
                if criterion_only else 1.0
            ),
            "criterion_max_diameter": (
                max(r.max_diameter for r in criterion_only) if criterion_only else 0.0
            ),
            "largest_family_row_share": largest_family / rows_total,
            "largest_split_group_row_share": largest_group / rows_total,
            "largest_split_group_constructs": _largest_bucket(self._split_roots),
            "edge_counts": dict(self._edge_counts),
            "high_degree_bridges": [dict(b) for b in self._high_degree_bridges],
            "percolated_split_groups": [dict(g) for g in self._percolated_groups],
            "unclosed_constraints": len(self._unclosed_constraints),
            "unclosed_constraint_examples": [
                dict(c) for c in self._unclosed_constraints[:10]
            ],
            "blocking": [dict(s) for s in self._blocking_stats],
            "operating_point": self.operating_point.as_dict(),
            "tolerated_error": self._tolerated_error,
        }

    def record_fields(self, key_or_row: Any) -> Dict[str, object]:
        """The identity fields a producer should write onto one record."""
        digest = self.digest_for(key_or_row)
        return {
            "antigen_sha256": digest,
            "construct_id": self.construct_id(digest),
            "biological_target_id": self.biological_target_id(digest),
            "split_group_id": self.split_group_id(digest, "generic"),
            "unseen_mutant_group_id": self.split_group_id(digest, "unseen_mutant"),
            "quarantine_partner_count": len(self.quarantine_partners(digest)),
            "test_ineligible": self.test_ineligible(digest),
            "antigen_is_composite": self.is_composite(digest),
        }

    # ------------------------------------------------------------- internals

    def alignment(self, left: str, right: str) -> Optional[PairAlignment]:
        """The measured relation between two digests, oriented left-to-right.

        Returns ``None`` when the pair was never a candidate. That is the correct
        answer rather than a gap: the blocker's filters are necessary conditions,
        so a pair it never proposed could not have met any threshold that uses
        them, and `blocking_recall_report` is what keeps that claim checked.
        """
        if left == right:
            sequence = self._sequences[left]
            return PairAlignment(
                identity=1.0, cov_left=1.0, cov_right=1.0,
                overlap=len(sequence), columns=len(sequence),
                matched=len(sequence), span_left=len(sequence),
                span_right=len(sequence), score=0,
            )
        key = (left, right) if left < right else (right, left)
        result = self._alignments.get(key)
        if result is None:
            return None
        return result if key[0] == left else result.flipped()

    def _members_by_id(self, partition, id_of, by_root: bool):
        """Invert an id map to member lists, once per partition rather than per call.

        Built lazily and cached. Scanning every digest per cluster made `stats()`
        quadratic, which is invisible on sixteen fixture records and costs
        minutes on the corpus's 9,574.
        """
        cache_key = ("root" if by_root else "member", id(partition))
        cached = getattr(self, "_members_cache", None)
        if cached is None:
            cached = self._members_cache = {}
        if cache_key not in cached:
            grouped: Dict[str, List[str]] = {}
            for digest in sorted(self._sequences):
                key = id_of[partition.find(digest)] if by_root else id_of[digest]
                grouped.setdefault(key, []).append(digest)
            cached[cache_key] = grouped
        return cached[cache_key]

    def _report(self, cluster_id, partition, id_of, *, by_root: bool = False):
        """Build a `ClusterReport` for one cluster of either partition."""
        members = self._members_by_id(partition, id_of, by_root).get(cluster_id, [])
        return cluster_report(
            cluster_id,
            members,
            self.alignment,
            tie_break=lambda digest: (-len(self._sequences[digest]), digest),
        )

    def _name_families(self) -> Dict[str, str]:
        """Name every family component after its strongest unambiguous identifier."""
        naming: Dict[str, str] = {}
        for root, members in self._family.components():
            accessions = sorted({a for m in members for a in self._accessions_of[m]})
            approved = sorted({
                decision.name for decision in self._name_decisions
                if decision.approved and decision.attached_to == root
            })
            candidates = {
                "uniprot": accessions,
                "name": approved,
                "seq": members if len(members) == 1 else [],
            }
            naming[root] = ""
            for namespace in FAMILY_NAMESPACE_RANK:
                available = candidates[namespace]
                if not available:
                    continue
                # Exactly one candidate in the strongest present namespace names
                # the component; more than one is a disagreement, and a
                # disagreement resolved by sort order is the defect, not the fix.
                if len(available) == 1:
                    naming[root] = f"{namespace}:{available[0]}"
                break
            if not naming[root]:
                blob = "|".join(sorted(members)).encode("utf-8")
                naming[root] = f"family:{hashlib.sha256(blob).hexdigest()[:16]}"
        return naming

    def _direct_partners(self) -> Dict[str, FrozenSet[str]]:
        """Invert the quarantine edge list into direct partner sets."""
        partners: Dict[str, Set[str]] = {}
        for edge in self._quarantine_edges:
            partners.setdefault(edge.left, set()).add(edge.right)
            partners.setdefault(edge.right, set()).add(edge.left)
        return {key: frozenset(value) for key, value in partners.items()}

    def _test_ineligible_constructs(self) -> FrozenSet[str]:
        """Constructs that may train but may not be scored, and may not bridge.

        A concatenated construct is reachable from several unrelated targets, so
        letting it carry a co-assignment constraint through would merge those
        targets into one split group. Marking it test-ineligible keeps the leak
        it represents out of scored evaluation while leaving every other
        component's diversity intact.
        """
        ineligible = set()
        for digest in self._composites:
            ineligible.add(self._construct_id_of[digest])
        # A member of a container the span guard refused to link is in the same
        # position as a composite: reachable from several unrelated targets at
        # once, and unsafe to score even though it is safe to train on.
        for digest in self._bridge_members:
            ineligible.add(self._construct_id_of[digest])
        return frozenset(ineligible)

    def _close_quarantine(self) -> Dict[str, str]:
        """Transitively close quarantine over families, refusing to route through bridges.

        Co-assignment constraints MUST be closed or they do not constrain. But
        closing them through a container merges everything the container touches,
        which is how a previous attempt reached a 77% split group. Edges incident
        to a test-ineligible construct are therefore recorded and NOT closed: the
        container is placed with one side and kept out of scored evaluation,
        rather than being allowed to weld two families together.
        """
        family_of_construct = {
            self._construct_id_of[digest]: self.biological_target_id(digest)
            for digest in sorted(self._sequences)
        }
        families = sorted(set(family_of_construct.values()))
        edges = []
        for edge in self._quarantine_edges:
            if edge.left in self._ineligible or edge.right in self._ineligible:
                continue
            left = family_of_construct.get(edge.left)
            right = family_of_construct.get(edge.right)
            if left and right and left != right:
                edges.append((left, right))
        # Every constraint refused above is a constraint NOT enforced, and the
        # honest name for that is residual exposure rather than a design
        # feature. Concretely: the 107-aa IGKC domain and the 607-aa HER2
        # ectodomain both sit byte-for-byte inside the 1057-aa fusion, the fusion
        # is test-ineligible so nothing closes through it, and nothing then stops
        # the ectodomain being SCORED in validation while the fusion that
        # contains it sits in train. Refusing to close is still the right trade
        # -- closing welds two unrelated families -- but the price is real and is
        # counted here so a caller can see it, decide whether to exclude the
        # container outright, and never mistake the refusal for a solution.
        self._unclosed_constraints = tuple(
            {
                "left": edge.left, "right": edge.right, "kind": edge.kind,
                "evidence": edge.evidence,
            }
            for edge in self._quarantine_edges
            if (edge.left in self._ineligible or edge.right in self._ineligible)
            and family_of_construct.get(edge.left)
            != family_of_construct.get(edge.right)
        )
        closed = transitive_closure(families, edges)
        group_of_family = {
            family: f"group:{hashlib.sha256('|'.join(sorted(closed.members(family))).encode('utf-8')).hexdigest()[:16]}"
            for family in families
        }
        return {
            construct: group_of_family[family]
            for construct, family in family_of_construct.items()
        }

    def _flag_percolated_closures(self):
        """Mark every construct in an over-large closed group test-ineligible.

        `max_container_span` bounds ONE container's degree. It does not bound
        the CLOSURE, and it cannot: a chain of containers each spanning two
        families is individually legal at any span bound and closes into one
        group. That was demonstrated on a ten-rung ladder where nine curator
        names each linked one adjacent pair.

        Quarantine percolation is intentional -- co-assignment constraints must
        be transitively closed or they do not constrain -- so the answer is not
        to drop the constraints, which would create the leak they exist to
        prevent. It is to keep them and stop SCORING inside the result: a
        percolated closure costs evaluation diversity, not correctness. The
        design document says the same thing in the other direction -- if a huge
        component remains after defensible preprocessing, the intended held-out
        claim may genuinely be unsupported -- and `validate_level_1` fails the
        level on the row share, so this is visible twice.

        Returns:
            One record per group that exceeded the bound.
        """
        families_of_group: Dict[str, Set[str]] = {}
        constructs_of_group: Dict[str, Set[str]] = {}
        for digest in sorted(self._sequences):
            group = self._split_roots[self._construct_id_of[digest]]
            families_of_group.setdefault(group, set()).add(
                self.biological_target_id(digest)
            )
            constructs_of_group.setdefault(group, set()).add(
                self._construct_id_of[digest]
            )
        percolated = []
        newly_ineligible = set(self._ineligible)
        for group in sorted(families_of_group):
            spanned = len(families_of_group[group])
            if spanned <= self._max_split_group_families:
                continue
            percolated.append({
                "group": group,
                "families_spanned": spanned,
                "constructs": len(constructs_of_group[group]),
            })
            newly_ineligible.update(constructs_of_group[group])
        self._ineligible = frozenset(newly_ineligible)
        return tuple(percolated)

    def _score(self, name, labelled, id_of) -> ErrorReport:
        """Score one labelled population for false merges and false splits."""
        digests = sorted(labelled)
        families = tuple(sorted({labelled[d].family for d in digests}))
        pairs = positives = negatives = 0
        false_merges: List[Tuple[str, str]] = []
        false_splits: List[Tuple[str, str]] = []
        unadjudicated = 0
        for position, left in enumerate(digests):
            for right in digests[position + 1:]:
                if (left, right) in self._unadjudicated or (right, left) in self._unadjudicated:
                    unadjudicated += 1
                    continue
                pairs += 1
                same_truth = labelled[left].target == labelled[right].target
                same_result = id_of(left) == id_of(right)
                if same_truth:
                    positives += 1
                    if not same_result:
                        false_splits.append((left, right))
                else:
                    negatives += 1
                    if same_result:
                        false_merges.append((left, right))
        return ErrorReport(
            name=name,
            families=families,
            pairs=pairs,
            positive_pairs=positives,
            negative_pairs=negatives,
            false_merges=len(false_merges),
            false_splits=len(false_splits),
            tolerated_error=self._tolerated_error,
            unadjudicated=unadjudicated,
            false_merge_examples=tuple(false_merges[:10]),
            false_split_examples=tuple(false_splits[:10]),
        )


def _largest_bucket(assignment: Mapping[str, str]) -> int:
    """Size of the biggest bucket in a key-to-bucket map, in one pass.

    Written as a counted pass rather than a max over a per-bucket scan: the
    latter is O(buckets x keys), which is invisible on sixteen fixture records
    and costs minutes on the corpus's nine thousand.

    Args:
        assignment: Key to bucket id.

    Returns:
        The largest bucket's size, or 0 when the map is empty.
    """
    counts: Dict[str, int] = {}
    for bucket in assignment.values():
        counts[bucket] = counts.get(bucket, 0) + 1
    return max(counts.values()) if counts else 0


def _component_ids(partition: Partition, prefix: str) -> Dict[str, str]:
    """Give every component a neutral id derived from its sorted members."""
    ids: Dict[str, str] = {}
    for root, members in partition.components():
        blob = "|".join(members).encode("utf-8")
        component_id = f"{prefix}:{hashlib.sha256(blob).hexdigest()[:16]}"
        for member in members:
            ids[member] = component_id
    return ids


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CuratedLabel:
    """Ground truth for one antigen sequence, written by hand before measurement.

    Attributes:
        family: The labelled family. Families are the unit the calibration and
            audit populations are split by, so that no family can both choose a
            threshold and confirm it.
        target: The labelled target within that family. Two records share a
            target when a correct resolver must place them in one biological
            family, and share only a family when they are related but distinct.
        note: What the record is, in words.
    """

    family: str
    target: str
    note: str


def _ingest(rows: Iterable[Mapping[str, object]]):
    """Reduce source rows to the per-antigen identity view.

    Every row is observed, including rows a downstream filter will drop: a
    curator writing an accession on a row is evidence about the target whether or
    not that particular antibody survives filtering, and dropping the evidence
    would make identity depend on filter thresholds.

    Args:
        rows: Source rows.

    Returns:
        ``(sequences, rows_of, names_of, pdbs_of, accessions_of, rows_seen,
        rows_without_antigen)``.
    """
    sequences: Dict[str, str] = {}
    rows_of: Dict[str, int] = {}
    names_of: Dict[str, Set[str]] = {}
    pdbs_of: Dict[str, Set[str]] = {}
    accessions_of: Dict[str, Set[str]] = {}
    rows_seen = 0
    rows_without_antigen = 0

    for row in rows:
        rows_seen += 1
        fields, sequence = row_identity_view(row)
        if not sequence:
            rows_without_antigen += 1
            continue
        digest = antigen_digest(sequence)
        sequences.setdefault(digest, sequence)
        rows_of[digest] = rows_of.get(digest, 0) + 1
        names_of.setdefault(digest, set())
        pdbs_of.setdefault(digest, set())
        accessions_of.setdefault(digest, set())
        name = normalize_target_name(fields["target_name"])
        if name:
            names_of[digest].add(name)
        pdb = canonicalize_accession(fields["target_pdb"])
        if pdb:
            pdbs_of[digest].add(pdb)
        accession = canonicalize_accession(fields["target_uniprot"])
        if accession:
            accessions_of[digest].add(accession)

    return (sequences, rows_of, names_of, pdbs_of, accessions_of,
            rows_seen, rows_without_antigen)


def _align_all(sequences, pairs, alignments, max_cells, progress, label):
    """Align a candidate list into ``alignments``, counting refusals.

    Args:
        sequences: Digest to residue string.
        pairs: Candidate pairs, already sorted.
        alignments: Destination, mutated in place. Pairs already present are
            skipped, so the two relations share one measurement.
        max_cells: Largest DP matrix to attempt.
        progress: Optional ``(done, total, label) -> None`` callback.
        label: Which pass this is, so a long run's log says which of the two
            candidate sets it is working through.

    Returns:
        The number of pairs refused for exceeding ``max_cells``.
    """
    refused = 0
    for position, (left, right) in enumerate(pairs):
        if progress is not None and position % 1000 == 0:
            progress(position, len(pairs), label)
        if (left, right) in alignments:
            continue
        try:
            alignments[(left, right)] = align_pair(
                sequences[left], sequences[right], max_cells=max_cells
            )
        except AlignmentTooLarge:
            refused += 1
    return refused


def _containment_candidates(index, point, family_of):
    """Containment candidates worth measuring: the ones that cross a family.

    Two blocking passes are needed because the two relations bind different
    sides. Similarity binds both coverages, so the length band applies and cuts
    hard. Containment binds only the shorter side, so it cannot -- a domain
    inside a fusion sits at a length ratio of 0.10 -- and the resulting candidate
    set is large: on the shipped corpus the containment pass proposed 450,381
    pairs, an order of magnitude more than the length-banded similarity pass.

    Almost all of that excess is REDUNDANT rather than informative. The corpus
    holds thousands of SARS-CoV-2 spike constructs that genuinely contain one
    another, and every one of those pairs is already in a single biological
    family. A quarantine edge between two constructs of one family constrains
    nothing the family has not already constrained, because the split closes over
    families. So containment is measured only where the family relation has not
    already answered -- which is exactly where the 107-aa IGKC domain sits inside
    the 1057-aa HER2 fusion, the case the relation exists for.

    This is an ordering choice, not a threshold: the family partition is built
    from similarity, accessions and names, none of which consult containment, so
    there is no circularity.

    Args:
        index: The built `CandidateIndex`.
        point: The operating point.
        family_of: Digest to family root, from the finished family partition.

    Returns:
        ``(pairs, stats, skipped_same_family)``.
    """
    candidates, stats = index.candidate_pairs(
        point.containment_identity,
        point.containment_coverage,
        apply_length_band=False,
        min_overlap=point.containment_overlap,
    )
    kept = tuple(
        pair for pair in candidates
        if family_of[pair[0]] != family_of[pair[1]]
    )
    return kept, stats, len(candidates) - len(kept)


def resolve_target_identity(
    rows: Iterable[Mapping[str, object]],
    *,
    operating_point: OperatingPoint = DEFAULT_OPERATING_POINT,
    curated_labels: Optional[Mapping[str, CuratedLabel]] = None,
    audit_families: Optional[Sequence[str]] = None,
    unadjudicated_pairs: Optional[FrozenSet[Tuple[str, str]]] = None,
    tolerated_error: Optional[str] = None,
    max_cells: Optional[int] = 50_000_000,
    max_container_span: int = 8,
    max_split_group_families: int = 12,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> TargetIdentityResolution:
    """Resolve target identity over a population of rows.

    The order of construction is not arbitrary. Identity is settled first, from
    sequences and accessions alone; names are judged ONCE against that frozen
    partition; and only then does the quarantine relation, which is the one
    relation allowed to percolate, get built on top. Judging names before the
    sequence partition exists would let a name bridge decide what the name is
    then measured against, and closing quarantine before the evidence table has
    run would turn every container into an identity claim.

    Args:
        rows: Source rows. Each must carry ``antigen_sequence`` and the three
            target fields, either nested under ``metadata`` or at the top level.
            Nothing else is read.
        operating_point: The sealed thresholds.
        curated_labels: Ground truth by antigen digest, for `audit_report`.
            Defaults to the labels shipped in `target_identity_labels`.
        audit_families: Families reserved for the audit population, which the
            operating point may not be tuned on. Defaults to the pinned set.
        unadjudicated_pairs: Pairs whose correct answer was never settled, excluded
            from scoring and counted rather than silently dropped.
        tolerated_error: The predeclared error asymmetry, recorded on every
            report so it cannot be chosen after the numbers are known.
        max_cells: Largest alignment matrix to attempt; larger pairs are refused
            and counted.
        max_container_span: A shared container -- a PDB entry, or a name that was
            not approved -- spanning more than this many distinct families is
            treated as a high-degree bridge: it marks its members test-ineligible
            instead of linking them, because closing a constraint through a hub
            merges everything the hub touches. Set high to disable, and expect
            percolation if you do.
        progress: Optional ``(done, total, label) -> None`` alignment progress
            callback. ``label`` is ``"similarity"`` or ``"containment"``; the two
            passes have different totals, and a log that does not say which is
            which reads like the total changed.

    Returns:
        A `TargetIdentityResolution`.
    """
    from . import target_identity_labels as labels

    if curated_labels is None:
        curated_labels = labels.CURATED_LABELS
    if audit_families is None:
        audit_families = labels.PINNED_AUDIT_FAMILIES
    if unadjudicated_pairs is None:
        unadjudicated_pairs = labels.UNADJUDICATED_PAIRS
    if tolerated_error is None:
        tolerated_error = labels.PREDECLARED_ERROR_ASYMMETRY

    (sequences, rows_of, names_of, pdbs_of, accessions_of,
     rows_seen, rows_without_antigen) = _ingest(rows)
    digests = sorted(sequences)
    composites = {d for d in digests if COMPOSITE_LINKER.search(sequences[d])}

    index = CandidateIndex(sequences)
    similarity, similarity_stats = index.candidate_pairs(
        operating_point.family_identity,
        operating_point.family_coverage,
        min_overlap=operating_point.family_overlap,
    )
    alignments: Dict[Tuple[str, str], PairAlignment] = {}
    refused = _align_all(
        sequences, similarity, alignments, max_cells, progress, "similarity"
    )

    def relation(left: str, right: str) -> Optional[PairAlignment]:
        key = (left, right) if left < right else (right, left)
        return alignments.get(key)

    def meets(left, right, identity, coverage, overlap) -> bool:
        result = relation(left, right)
        return bool(
            result
            and result.identity >= identity
            and result.min_coverage >= coverage
            and result.overlap >= overlap
        )

    point = operating_point
    edge_counts: Dict[str, int] = {}

    def construct_pair(left, right) -> bool:
        return meets(left, right, point.construct_identity,
                     point.construct_coverage, point.construct_overlap)

    def family_pair(left, right) -> bool:
        return meets(left, right, point.family_identity,
                     point.family_coverage, point.family_overlap)

    def name_bridge_pair(left, right) -> bool:
        return meets(left, right, point.name_bridge_identity,
                     point.name_bridge_coverage, point.name_bridge_overlap)

    def strength(pair) -> tuple:
        result = relation(*pair)
        if result is None:
            return (0.0, 0.0, 0, pair[0], pair[1])
        return (-result.identity, -result.min_coverage, -result.overlap,
                pair[0], pair[1])

    # E1/E2 -- the construct relation. Byte-identical sequences share a digest
    # and are therefore already one member, so E1 needs no edge of its own.
    construct_edges = [
        pair for pair in sorted(alignments) if construct_pair(*pair)
    ]
    edge_counts["E2_near_identical_construct"] = len(construct_edges)
    construct_partition = agglomerate_complete_linkage(
        digests, construct_edges, construct_pair, rank=strength
    )

    # The family relation. Constructs merge first as hard evidence, then shared
    # accessions, then bounded clustering over family-level similarity.
    family_partition = Partition.singletons(digests)
    for _, members in construct_partition.components():
        for member in members[1:]:
            family_partition.force_union(members[0], member)

    by_accession: Dict[str, List[str]] = {}
    for digest in digests:
        for accession in sorted(accessions_of[digest]):
            by_accession.setdefault(accession, []).append(digest)
    accession_merges = 0
    #: Digests whose family membership rests on a curator's identifier rather
    #: than on the similarity criterion. Tracked so that the guard report can
    #: state the criterion's diameter guarantee over the components the
    #: criterion actually decided, and state the other components separately
    #: instead of blending the two into one misleading minimum.
    curator_merged: Set[str] = set()
    for accession in sorted(by_accession):
        members = by_accession[accession]
        for member in members[1:]:
            if family_partition.force_union(members[0], member):
                accession_merges += 1
                curator_merged.update((members[0], member))
    edge_counts["E5_shared_accession_merges"] = accession_merges

    family_edges = [pair for pair in sorted(alignments) if family_pair(*pair)]
    edge_counts["E3_family_similarity"] = len(family_edges)
    family_partition = agglomerate_complete_linkage(
        digests, family_edges, family_pair, rank=strength,
        partition=family_partition,
    )

    # E7/E8 -- names, decided ONCE against the partition as it now stands.
    name_members: Dict[str, List[str]] = {}
    for digest in digests:
        for name in sorted(names_of[digest]):
            name_members.setdefault(name, []).append(digest)

    # FROZEN before any name is judged, and never re-read inside the loop. An
    # earlier version of this code called `force_union` as each name was
    # approved, which meant the next name was judged against a partition the
    # previous name had already changed -- and a coherence check that skips
    # pairs already in one group then skips pairs an earlier name put there.
    # That is iterative name bridging by another route: names chain transitively
    # and the verdict depends on their alphabetical order. Requirement 2 says the
    # decision is computed ONCE against a frozen partition, so it is.
    frozen_family = {digest: family_partition.find(digest) for digest in digests}

    # The bridge threshold is LOOSER than the family threshold -- it has to be,
    # because a name that spans two family groups is by definition spanning
    # groups the family relation refused to join. So the family pass's candidate
    # list cannot answer it: any pair between the bridge threshold and the family
    # threshold is invisible to a blocker run at the family thresholds, and
    # judging the bridge on that list would refuse every name for lack of
    # evidence rather than on evidence.
    #
    # No third blocking pass is needed, because the pairs a name decision depends
    # on are enumerable exactly: they are the cross-group pairs among the
    # non-composite members of each name that spans more than one group. On the
    # corpus that is a small, targeted set, so they are aligned directly.
    bridging = set()
    for name, members in sorted(name_members.items()):
        usable = sorted(m for m in members if m not in composites)
        if len({frozen_family[m] for m in usable}) >= 2:
            bridging.update(usable)
    bridging_sorted = sorted(bridging)
    bridge_pairs = set()
    for position, left in enumerate(bridging_sorted):
        for right in bridging_sorted[position + 1:]:
            bridge_pairs.add((left, right))
    refused += _align_all(
        sequences, tuple(sorted(bridge_pairs)), alignments, max_cells, progress,
        "name-bridge",
    )
    edge_counts["name_bridge_pairs_measured"] = len(bridge_pairs)

    decisions: List[NameDecision] = []
    anchors: Dict[str, str] = {}
    approved: List[List[str]] = []
    for name in sorted(name_members):
        members = name_members[name]
        composite_members = tuple(m for m in members if m in composites)
        usable = [m for m in members if m not in composites]
        if not usable:
            decisions.append(NameDecision(
                name=name, approved=False, attached_to=None,
                reason=("every record carrying this name is a concatenated "
                        "construct; a container names nothing"),
                composite_members=composite_members,
            ))
            continue
        groups = sorted({frozen_family[m] for m in usable})
        if len(groups) == 1:
            anchors[name] = usable[0]
            decisions.append(NameDecision(
                name=name, approved=True, attached_to=None,
                reason=("spans one family group, so approving it merges nothing "
                        "and contradicts nothing"),
                spanned_groups=tuple(groups),
                composite_members=composite_members,
            ))
            continue
        worst_identity: Optional[float] = None
        worst_overlap: Optional[int] = None
        unmeasured = 0
        coherent = True
        for position, left in enumerate(usable):
            for right in usable[position + 1:]:
                if frozen_family[left] == frozen_family[right]:
                    continue
                result = relation(left, right)
                if result is None:
                    # The blocker is run at the family thresholds, which are
                    # STRICTER than the name bridge, so a pair it never proposed
                    # is a pair whose bridge strength is unknown rather than
                    # zero. Refusing it is the conservative direction -- the name
                    # quarantines instead of merging -- and the reason says
                    # "unmeasured" rather than reporting a 0.0000 nobody measured.
                    unmeasured += 1
                    coherent = False
                    continue
                worst_identity = (
                    result.identity if worst_identity is None
                    else min(worst_identity, result.identity)
                )
                worst_overlap = (
                    result.overlap if worst_overlap is None
                    else min(worst_overlap, result.overlap)
                )
                if (result.identity < point.name_bridge_identity
                        or result.min_coverage < point.name_bridge_coverage
                        or result.overlap < point.name_bridge_overlap):
                    coherent = False
        if coherent:
            anchors[name] = usable[0]
            approved.append(usable)
            decisions.append(NameDecision(
                name=name, approved=True, attached_to=None,
                reason=(f"bridges {len(groups)} family groups and every cross "
                        f"pair clears the name bridge "
                        f"(worst identity {worst_identity:.4f}, "
                        f"worst overlap {worst_overlap})"),
                spanned_groups=tuple(groups), worst_identity=worst_identity,
                worst_overlap=worst_overlap, composite_members=composite_members,
            ))
        else:
            measured = (
                f"whose weakest measured cross pair is identity "
                f"{worst_identity:.4f} over {worst_overlap} matched residues"
                if worst_identity is not None
                else "with no measured cross pair at all"
            )
            unmeasured_note = (
                f", and {unmeasured} cross pairs the blocker never proposed at "
                f"the family thresholds, whose bridge strength is therefore "
                f"unknown rather than zero"
                if unmeasured else ""
            )
            decisions.append(NameDecision(
                name=name, approved=False, attached_to=None,
                reason=(f"spans {len(groups)} family groups {measured}"
                        f"{unmeasured_note}; a name that spans two families "
                        f"names neither, so it quarantines instead of merging"),
                spanned_groups=tuple(groups), worst_identity=worst_identity,
                worst_overlap=worst_overlap, composite_members=composite_members,
            ))

    # Applied only now that every verdict is in, so no name's decision can have
    # been influenced by another name's merge -- AND applied through the same
    # bounded criterion E3 uses, not through an unconditional union.
    #
    # Freezing the verdicts is necessary and not sufficient. Union is transitive,
    # so a chain of individually valid names percolates even when every single
    # verdict is correct: nine names each bridging one adjacent pair of a
    # ten-member ladder are each locally right, and their union is one family
    # whose ends share 22.67% identity. That is the A~B admitted, B~C admitted,
    # A~C refused shape the cluster criterion exists to refuse, arriving through
    # names instead of sequences, and an unconditional `force_union` here would
    # let it through the one door the design leaves open.
    before = {digest: family_partition.find(digest) for digest in digests}
    name_edges = sorted({
        (left, right) if left < right else (right, left)
        for usable in approved
        for left in usable
        for right in usable
        if left != right
    })
    family_partition = agglomerate_complete_linkage(
        digests, name_edges, name_bridge_pair, rank=strength,
        partition=family_partition,
    )
    approved_merges = 0
    for digest in digests:
        if family_partition.find(digest) != before[digest]:
            approved_merges += 1
            curator_merged.add(digest)
    edge_counts["E7_approved_name_merges"] = approved_merges
    edge_counts["E7_name_merges_refused"] = family_partition.merges_refused
    decisions = [
        NameDecision(
            name=d.name,
            approved=d.approved,
            attached_to=(family_partition.find(anchors[d.name])
                         if d.approved and d.name in anchors else None),
            reason=d.reason,
            spanned_groups=d.spanned_groups,
            worst_identity=d.worst_identity,
            worst_overlap=d.worst_overlap,
            composite_members=d.composite_members,
        )
        for d in decisions
    ]

    construct_id_of = _component_ids(construct_partition, "construct")

    # The containment pass runs HERE, after the family partition is settled, so
    # it can be restricted to the pairs that still carry information. See
    # `_containment_candidates`.
    containment_pairs, containment_stats, same_family = _containment_candidates(
        index, point, {d: family_partition.find(d) for d in digests}
    )
    refused += _align_all(
        sequences, containment_pairs, alignments, max_cells, progress, "containment"
    )
    blocking_stats = (
        {"relation": "similarity", **similarity_stats.as_dict()},
        {"relation": "containment", **containment_stats.as_dict(),
         "skipped_already_one_family": same_family,
         "measured": len(containment_pairs)},
        {"relation": "union", "pairs": len(alignments)},
    )

    # E9 -- accession disagreements. Reported, never resolved.
    conflicts = tuple(
        AccessionConflict(
            antigen_sha256=digest,
            accessions=tuple(sorted(accessions_of[digest])),
            names=tuple(sorted(names_of[digest])),
            rows=rows_of[digest],
            resolved_id="",
        )
        for digest in digests
        if len(accessions_of[digest]) > 1
    )

    # E4/E6/E8 -- the quarantine relation.
    quarantine: Dict[Tuple[str, str], QuarantineEdge] = {}
    bridges: List[Dict[str, object]] = []

    def add_edge(left_digest, right_digest, kind, evidence):
        left, right = construct_id_of[left_digest], construct_id_of[right_digest]
        if left == right:
            return
        key = (left, right) if left < right else (right, left)
        quarantine.setdefault(
            key, QuarantineEdge(left=key[0], right=key[1], kind=kind,
                                evidence=evidence)
        )

    containment_edges = 0
    for (left, right), result in sorted(alignments.items()):
        if (result.identity >= point.containment_identity
                and result.max_coverage >= point.containment_coverage
                and result.min_coverage < point.containment_max_coverage
                and result.overlap >= point.containment_overlap):
            containment_edges += 1
            add_edge(left, right, "containment",
                     f"identity {result.identity:.4f}, coverage "
                     f"{result.cov_left:.4f}/{result.cov_right:.4f}, "
                     f"{result.overlap} matched residues")
    edge_counts["E4_containment"] = containment_edges

    bridge_members: Set[str] = set()

    def add_container(members, kind, label):
        """Link every member of a container, unless the container is a hub.

        A hub is NOT silently dropped. Refusing to link it and then walking away
        would leave its members with no constraint of any kind, which is the
        opposite of the conservative behaviour the refusal exists for: every
        document describing this says a high-degree bridge marks its members
        test-ineligible INSTEAD of linking them, and this is where that happens.
        """
        families = {family_partition.find(m) for m in members}
        if len(families) > max_container_span:
            bridges.append({
                "kind": kind, "label": label, "members": len(members),
                "families_spanned": len(families),
            })
            bridge_members.update(members)
            return
        for position, left in enumerate(sorted(members)):
            for right in sorted(members)[position + 1:]:
                add_edge(left, right, kind, label)

    by_pdb: Dict[str, List[str]] = {}
    for digest in digests:
        for pdb in sorted(pdbs_of[digest]):
            by_pdb.setdefault(pdb, []).append(digest)
    for pdb in sorted(by_pdb):
        if len(by_pdb[pdb]) > 1:
            add_container(by_pdb[pdb], "shared_container",
                          f"pdb:{pdb} holds several polymer entities")
    edge_counts["E6_shared_container"] = sum(
        1 for e in quarantine.values() if e.kind == "shared_container"
    )

    unapproved = {d.name for d in decisions if not d.approved}
    for name in sorted(unapproved):
        members = name_members[name]
        if len(members) > 1:
            add_container(members, "ambiguous_name",
                          f"name:{name} spans constructs that are not one family")
    edge_counts["E8_ambiguous_name"] = sum(
        1 for e in quarantine.values() if e.kind == "ambiguous_name"
    )
    edge_counts["high_degree_bridges_refused"] = len(bridges)

    resolution = TargetIdentityResolution(
        operating_point=point,
        sequences=sequences,
        rows_of=rows_of,
        names_of=names_of,
        pdbs_of=pdbs_of,
        accessions_of=accessions_of,
        composites=composites,
        alignments=alignments,
        construct_partition=construct_partition,
        family_partition=family_partition,
        name_decisions=tuple(decisions),
        accession_conflicts=conflicts,
        quarantine_edges=tuple(quarantine[key] for key in sorted(quarantine)),
        edge_counts=edge_counts,
        rows_seen=rows_seen,
        rows_without_antigen=rows_without_antigen,
        alignments_refused=refused,
        blocking_stats=blocking_stats,
        curated_labels=curated_labels,
        audit_families=audit_families,
        unadjudicated_pairs=unadjudicated_pairs,
        tolerated_error=tolerated_error,
        max_cells=max_cells,
        max_container_span=max_container_span,
        max_split_group_families=max_split_group_families,
        high_degree_bridges=tuple(bridges),
        curator_merged=frozenset(curator_merged),
        bridge_members=frozenset(bridge_members),
    )
    # The neutral id a disputed component actually received, filled in now that
    # the naming pass has run, so a caller reading a conflict can see what
    # replaced the accession that would have won on sort order alone.
    resolution._accession_conflicts = tuple(
        AccessionConflict(
            antigen_sha256=conflict.antigen_sha256,
            accessions=conflict.accessions,
            names=conflict.names,
            rows=conflict.rows,
            resolved_id=resolution.biological_target_id(conflict.antigen_sha256),
        )
        for conflict in conflicts
    )
    return resolution
