"""
The acceptance contract for antigen target identity, as real data.

WHAT THIS IS
------------
Six pinned outcomes decide whether a target-identity design is acceptable. Each
one is a real record -- or a real set of records -- lifted out of the producer's
own input, ``data/raw/asd-antibody-antigen/*.parquet`` (20 shards, 1,227,083
rows, 9,574 distinct antigen sequences), and out of the corpus those shards
produced, ``data/processed/antibody_antigen_v2/antibody_antigen.jsonl.gz``
(831,869 rows). Distilling them here is what lets the acceptance tests in
``test_target_identity_acceptance.py`` run in a few seconds, and keeps them
runnable on a checkout with no ``data/`` directory at all.

Nothing here is invented. Every sequence is byte-for-byte what the shards carry
after ``prepare_antibody_antigen.clean_aa_sequence``; every accession, PDB code
and target name is what the curator wrote; every ``raw_rows`` count is the
number of shard rows behind that exact
(sequence, name, pdb, uniprot, dataset) tuple. ``SYNTHESIS_LEDGER`` records
which parts of the contract had to be synthesised: it is empty.

LABEL BLINDNESS (contract requirement C)
----------------------------------------
Identity must be computable from sequences and identifiers alone. This module
enforces that structurally rather than by promise: ``AnnotationRow`` and
``AntigenRecord`` have no field for a label, an affinity, a measurement or a
confidence, so an acceptance test written against these fixtures *cannot* read
one even by accident. ``FORBIDDEN_LABEL_FIELDS`` names what was deliberately
left out, and the acceptance suite asserts the dataclasses still carry none of
it. ``OMICRON_SHARED_HEAVY`` is the sole antibody-side datum and is not an
identity input: it exists only to price what leaks when the Omicron pair is
separated.

THE REFERENCE ALIGNER IS NOT THE CONTRACT
-----------------------------------------
``reference_aligned_identity`` is a plain Needleman-Wunsch (match +1, mismatch
-1, linear gap -1) that carries matched-residue and alignment-column counters
along the best path instead of storing a traceback. It has exactly one job: to
prove that the identity and coverage constants recorded below still describe the
sequences recorded below, so a later edit cannot drift a number away from its
data unnoticed. An implementation may use a different scoring scheme, a
different library, or a different definition of coverage -- it has to satisfy
the pinned OUTCOMES, not reproduce these constants. The constants were measured
with this function, and independently with a numpy implementation of the same
recurrence, which agreed to six decimal places.

HOW THE NUMBERS WERE MEASURED
-----------------------------
Identity is matched residues over alignment columns, gap columns included.
Coverage of a sequence is matched residues over that sequence's own length, so a
short sequence buried inside a long one scores full coverage on one side and
poor coverage on the other -- which is the whole point of contract requirement
1. The ``committed_rule`` block on each case was read two ways that agree: the
``canonical_target_id`` stored in the v2 corpus, and a rebuild of the partition
with the production ``TargetIdentityIndex`` and ``extract_target_nodes`` over
every distinct annotation tuple in the shards. That rebuild reports 25,597
identity nodes in 8,931 components, and its canonical id matches the stored one
for every record in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple


#: Parts of the six pinned outcomes that had to be synthesised because no real
#: record could be found. Deliberately empty: all six were sourced from
#: ``data/raw/asd-antibody-antigen``. If a future edit adds a synthetic case it
#: belongs here, named, so nobody quotes it as evidence about the corpus.
SYNTHESIS_LEDGER: Tuple[str, ...] = ()

#: Field names deliberately absent from every dataclass below. Supervision must
#: not be reachable from an identity fixture, so the guarantee is enforced by
#: what the classes do not have rather than by a comment asking nicely.
FORBIDDEN_LABEL_FIELDS = frozenset({
    "affinity",
    "affinity_raw",
    "affinity_type",
    "binder_label",
    "confidence",
    "is_strong_binder",
    "processed_measurement",
    "processed_measurement_float",
    "processed_measurement_raw",
})


@dataclass(frozen=True)
class AnnotationRow:
    """One curator annotation observed over one antigen sequence.

    These are the identifier fields ``extract_target_fields`` reads, plus the
    shard dataset the rows came from and how many shard rows carried exactly
    this combination. Nothing else -- see the label-blindness note above.
    """

    target_name: str
    target_pdb: str
    target_uniprot: str
    dataset: str
    raw_rows: int


@dataclass(frozen=True)
class AntigenRecord:
    """One distinct antigen sequence plus every annotation written over it."""

    key: str
    antigen_sequence: str
    antigen_sha256_32: str
    annotations: Tuple[AnnotationRow, ...]

    @property
    def length(self) -> int:
        return len(self.antigen_sequence)

    @property
    def raw_rows(self) -> int:
        """Shard rows carrying this antigen sequence, over all annotations."""
        return sum(a.raw_rows for a in self.annotations)

    def metadata(self, annotation_index: int = 0) -> Dict[str, str]:
        """The nested-metadata shape ``extract_target_nodes`` expects.

        Args:
            annotation_index:
                Which of this record's annotations to present. A record with
                more than one annotation is a record whose identity depends on
                which curator you ask, so the caller has to choose rather than
                get a silent default that hides the disagreement.

        Returns:
            A ``{target_name, target_pdb, target_uniprot}`` mapping.
        """
        a = self.annotations[annotation_index]
        return {
            "target_name": a.target_name,
            "target_pdb": a.target_pdb,
            "target_uniprot": a.target_uniprot,
        }

    def merged_metadata(self) -> Dict[str, str]:
        """The union of the identifiers any curator wrote over this sequence.

        Several fixtures carry a name on one shard row and a PDB code on
        another. The identity graph sees both, because it observes every row, so
        a helper that showed only the first annotation would understate what the
        production index is actually given.

        Returns:
            A ``{target_name, target_pdb, target_uniprot}`` mapping taking the
            first non-empty value for each field.
        """
        return {
            "target_name": next((a.target_name for a in self.annotations if a.target_name), ""),
            "target_pdb": next((a.target_pdb for a in self.annotations if a.target_pdb), ""),
            "target_uniprot": next((a.target_uniprot for a in self.annotations if a.target_uniprot), ""),
        }

    def rows(self) -> Tuple[Dict[str, object], ...]:
        """This record as raw-row dicts, one per annotation.

        Args:
            None.

        Returns:
            One dict per annotation, each shaped like a parquet row reduced to
            its identity fields: ``metadata`` plus ``antigen_sequence``.
        """
        return tuple(
            {"metadata": self.metadata(i), "antigen_sequence": self.antigen_sequence}
            for i in range(len(self.annotations))
        )


@dataclass(frozen=True)
class PairEvidence:
    """A measured relation between two fixture sequences.

    ``identity`` and the two coverages come from ``reference_aligned_identity``.
    ``exact_substring`` is the containment test the reverted attempt relied on;
    it is recorded because three of the six outcomes turn on where containment
    and alignment disagree.
    """

    left: str
    right: str
    identity: float
    coverage_left: float
    coverage_right: float
    exact_substring: bool
    note: str = ""


@dataclass(frozen=True)
class PinnedCase:
    """One of the six acceptance outcomes, with its data and its evidence."""

    outcome: int
    title: str
    required: str
    records: Tuple[AntigenRecord, ...]
    evidence: Tuple[PairEvidence, ...] = ()
    committed_rule: Mapping[str, object] = field(default_factory=dict)
    provenance: str = ""

    def record(self, key: str) -> AntigenRecord:
        """Look up one record by its fixture key."""
        for r in self.records:
            if r.key == key:
                return r
        raise KeyError(f"{key!r} not in outcome {self.outcome}")

    def relation(self, left: str, right: str) -> PairEvidence:
        """Look up the measured relation between two fixture keys, either order."""
        for e in self.evidence:
            if {e.left, e.right} == {left, right}:
                return e
        raise KeyError(f"({left!r}, {right!r}) not measured for outcome {self.outcome}")


# --------------------------------------------------------------------------- #
# Reference aligner -- a fixture-integrity device, NOT the contract
# --------------------------------------------------------------------------- #

def reference_aligned_identity(a: str, b: str) -> Tuple[float, float, float]:
    """Global alignment identity and per-sequence coverage.

    Needleman-Wunsch, match +1 / mismatch -1 / linear gap -1, carrying matched
    residue and alignment-column counters forward along the best path so no
    traceback matrix is needed. Two rows of state, so memory is O(len(b)) and
    the whole fixture suite costs well under a second.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        ``(identity, coverage_a, coverage_b)``. ``identity`` is matched residues
        over alignment columns, gap columns included; each coverage is matched
        residues over that sequence's own length.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0
    gap = -1
    prev_score = [j * gap for j in range(m + 1)]
    prev_match = [0] * (m + 1)
    prev_cols = list(range(m + 1))
    for i in range(1, n + 1):
        ai = a[i - 1]
        cur_score = [i * gap] + [0] * m
        cur_match = [0] * (m + 1)
        cur_cols = [i] + [0] * m
        for j in range(1, m + 1):
            hit = ai == b[j - 1]
            diagonal = prev_score[j - 1] + (1 if hit else -1)
            up = prev_score[j] + gap
            left = cur_score[j - 1] + gap
            if diagonal >= up and diagonal >= left:
                cur_score[j] = diagonal
                cur_match[j] = prev_match[j - 1] + (1 if hit else 0)
                cur_cols[j] = prev_cols[j - 1] + 1
            elif up >= left:
                cur_score[j] = up
                cur_match[j] = prev_match[j]
                cur_cols[j] = prev_cols[j] + 1
            else:
                cur_score[j] = left
                cur_match[j] = cur_match[j - 1]
                cur_cols[j] = cur_cols[j - 1] + 1
        prev_score, prev_match, prev_cols = cur_score, cur_match, cur_cols
    matched = prev_match[m]
    return matched / prev_cols[m], matched / n, matched / m


def shares_no_kmer(a: str, b: str, k: int = 8) -> bool:
    """True when two sequences share no k-mer at all.

    A cheap, exact non-homology witness used where a full alignment would be
    wasted: two sequences with no shared 8-mer cannot have a meaningful local
    alignment, and the check is O(len(a) + len(b)).

    Args:
        a: First sequence.
        b: Second sequence.
        k: k-mer width.

    Returns:
        True when the k-mer sets are disjoint.
    """
    left = {a[i:i + k] for i in range(len(a) - k + 1)}
    right = {b[i:i + k] for i in range(len(b) - k + 1)}
    return not (left & right)


# --------------------------------------------------------------------------- #
# Outcome 1 -- the Omicron 2-residue indel
#
# Two SARS-CoV-2 spike NTD constructs from `covid-19` shard rows. Half the rows
# for each carry the name `sars-cov2_omicron`; the other half carry no
# identifier at all, so the seq node is the only thing holding them together.
# The 289-mer is the 287-mer with `HV` inserted after position 53. Every one of
# the 287-mer's residues is matched, so the pair is 99.31% identical with full
# coverage of the shorter side -- and `287 in 289` is still False, because the
# insertion is interior. That one fact is what let the reverted attempt's
# exact-substring containment test tear this family in half.
# --------------------------------------------------------------------------- #
# 287-aa Omicron spike NTD construct.
OMICRON_NTD_287 = AntigenRecord(
    key='omicron_ntd_287',
    antigen_sequence=(
        "SQCVNLITRTQSYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAISGTNGTKRFDN"
        "PVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDVYY"
        "HKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTP"
        "INLGRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPR"
        "TFLLKYNENGTITDAVDCALDPLSETKCTLK"
    ),
    antigen_sha256_32='3a917f8d4893e8791048ede5f795f0b1',
    annotations=(
        AnnotationRow(target_name='', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=13),
        AnnotationRow(target_name='sars-cov2_omicron', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=13),
    ),
)
assert OMICRON_NTD_287.length == 287
assert OMICRON_NTD_287.raw_rows == 26

# 289-aa Omicron spike NTD construct: the 287-mer plus an interior `HV`.
OMICRON_NTD_289 = AntigenRecord(
    key='omicron_ntd_289',
    antigen_sequence=(
        "SQCVNLITRTQSYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRF"
        "DNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDV"
        "YYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKH"
        "TPINLGRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQ"
        "PRTFLLKYNENGTITDAVDCALDPLSETKCTLK"
    ),
    antigen_sha256_32='af4a498a5a5050210a542511ee691e81',
    annotations=(
        AnnotationRow(target_name='', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=10),
        AnnotationRow(target_name='sars-cov2_omicron', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=10),
    ),
)
assert OMICRON_NTD_289.length == 289
assert OMICRON_NTD_289.raw_rows == 20

#: The residues the 289-mer has and the 287-mer does not, and where they sit.
#: Pinning the offset rather than only the identity score means the acceptance
#: suite can prove the relation is a clean 2-residue insertion in O(n), with no
#: alignment and no threshold involved.
OMICRON_INSERTION = "HV"
OMICRON_INSERTION_OFFSET = 53

#: Heavy variable domains seen against BOTH Omicron constructs. These are
#: heavy-only records -- the light chain is empty on every one of these shard
#: rows -- so the antibody identity is the VH alone. The 287-mer carries 12
#: distinct VH sequences and the 289-mer 10; 10 are shared, i.e. EVERY antibody
#: seen against the 289-mer is also seen against the 287-mer. Splitting the two
#: constructs across a train/val boundary therefore leaks 100% of the val side's
#: antibodies -- which is what the reverted attempt measured in the large
#: (335 val rows, all 325 distinct antibodies also present in train).
#:
#: NOT an identity input. See the label-blindness note in the module docstring.
OMICRON_SHARED_HEAVY: Tuple[str, ...] = (
    (
        "EVQLVESGGGLVQPGGSLRLACVASGFTFSIYEMNWVRQAPGKGLEWVSYITTSGHARYNADSV"
        "KGRFTISRDNSKNSFYLQMNSLRAEDTAIYYCARPQYHYYDTSTYHSYGFDIWGQGTMVTVSS"
    ),
    (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANINQDGSEKYYVDSV"
        "KGRFTISRDNAKNSLYLQVNSLRAEDTAVYYCARDWDYDILTGSWFGAFDIWGQGTTVTVSS"
    ),
    (
        "QITLKESGPTLVKPTQTLTLTCKLSGFSVNTGGVGVGWIRQPPGKALEWLALIYWNDDKLYSPS"
        "LKSRLTVTKDTSKNQVVLTMTNMDPVDTATYYCAHVLVWFGEVLPDAFDVWGQGTMVTVSS"
    ),
    (
        "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPS"
        "LKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARCRPEYYFGSGSYLDFDYWGQGTLVTVSS"
    ),
    (
        "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKHYADSV"
        "KGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDSGYNYGYSWFDPWGQGTLVTVSS"
    ),
    (
        "QVQLVQSGAEVKKAGSSVKVSCKASGGTFSSHTITWVRQAPGQGLEWMGRIIPILGIANYAQKF"
        "QGRVTITADKSTSTAYMELSSLRSEDTAVYYCASLQTVDTAIEKYYGMDVWGQGTTVTVSS"
    ),
    (
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGVINPSGGSTSYAEKF"
        "RGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARDREPHSDSSGYWDSLKYYYYYALDVWGQGT"
        "TVTVSS"
    ),
    (
        "QVQLVQSGAEVKKPGASVKVSCKVSGYTLIELSMHWVRQAPGKGLEWMGGFDPEDAETIYAQKF"
        "QGRVTMTEDTSTDTAYMELSSLRSEDTAVYYCATGIAVIGPPPSTYYYYGMDVWGQGTTVTVSS"
    ),
    (
        "QVQLVQSGAEVKKPGSSVKVSCKTSGGTFNSFAINWVRQAPGQGPEWMGRVIPVLEIANYAQKF"
        "QGRITITADKSTSTAYMELSSLTSEDTAIYYCARHHIAVAQPYFDYWGQGTLVTVSS"
    ),
    (
        "VQLVQSGAEVKKPGASVKVSCKVSGYTLTELSMHWVRQAPGKGLEWMGGFDPEDGETIYAQKFQ"
        "GRVTMTEDTSTDTAYMELSSLRSEDTAVYYCATGPAVRRGSWFDPWGQGTLVTVSS"
    ),
)
assert len(OMICRON_SHARED_HEAVY) == 10
assert len(set(OMICRON_SHARED_HEAVY)) == 10

OMICRON_INDEL = PinnedCase(
    outcome=1,
    title="Omicron 2-residue indel -> SAME family, cannot straddle generic validation",
    required=(
        "The 287-mer and the 289-mer must land in one biological_target_id and "
        "one generic split_group_id, on SEQUENCE evidence. Not because a curator "
        "happened to write the same name on both: the name is a single-linkage "
        "bridge (see outcome 5, where the same mechanism fuses non-homologous "
        "constructs), so a family that survives only while the name survives is "
        "not a family."
    ),
    records=(OMICRON_NTD_287, OMICRON_NTD_289),
    evidence=(
    PairEvidence(
        left='omicron_ntd_287', right='omicron_ntd_289',
        identity=0.993080, coverage_left=1.000000, coverage_right=0.993080,
        exact_substring=False,
        note="2-residue insertion 'HV' after offset 53; interior, so containment fails",
    ),
    ),
    committed_rule={
        "canonical_target_id": {
            "omicron_ntd_287": "name:sars_cov2_omicron",
            "omicron_ntd_289": "name:sars_cov2_omicron",
        },
        "split": {"omicron_ntd_287": "train", "omicron_ntd_289": "train"},
        "component_distinct_antigens": 20,
        "component_raw_rows": 37974,
        "verdict": (
            "SATISFIED TODAY, BY THE WRONG MECHANISM. Both land in one component "
            "because the seq nodes of their name-bearing sibling rows attach to "
            "name:sars_cov2_omicron. The committed rule has no sequence-similarity "
            "relation at all, so remove the name and the family dissolves."
        ),
    },
    provenance=(
        "data/raw/asd-antibody-antigen, dataset `covid-19`. 26 and 20 shard rows; "
        "12 and 10 raw rows respectively survive keep_record into "
        "data/processed/antibody_antigen_v2, all on the train side."
    ),
)


# --------------------------------------------------------------------------- #
# Outcome 2 -- the 1N8Z fusion construct
#
# One `flab_shanehsazzadeh2023` antigen field holds an entire Fab welded to its
# target: light chain, GGGGGGGG, heavy chain, GGGGGGGG, HER2 ECD. Two other real
# antigens are exact substrings of it -- the 107-aa IGKC constant domain
# (P01834, from `patents`) at offset 107, and the 607-aa HER2 ECD at offset 450.
# Containment alone therefore says IGKC is HER2. Reciprocal coverage says
# otherwise: the IGKC match covers 100% of IGKC and 10.12% of the fusion.
# 28 of the 9,574 distinct antigen sequences in the shards carry a G{8,} run.
# --------------------------------------------------------------------------- #
# The 1057-aa concatenated Fab + linker + HER2 construct.
FUSION_1N8Z = AntigenRecord(
    key='fusion_1n8z',
    antigen_sequence=(
        "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSG"
        "SRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSG"
        "TASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVY"
        "ACEVTHQGLSSPVTKSFNRGECGGGGGGGGEVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYI"
        "HWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR"
        "WGGDGFYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWN"
        "SGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPGGGGGG"
        "GGTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYV"
        "LIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEI"
        "LKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDC"
        "QSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNT"
        "DTFESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCAR"
        "VCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLE"
        "EITGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALI"
        "HHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCS"
        "QFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFC"
        "VARCPSGVKPDLSYMPIWKFPDEEGACQPCPIN"
    ),
    antigen_sha256_32='c6627db2e640c16d92cb55e9745cff6a',
    annotations=(
        AnnotationRow(target_name='human her2', target_pdb='1n8z', target_uniprot='',
                      dataset='flab_shanehsazzadeh2023', raw_rows=446),
    ),
)
assert FUSION_1N8Z.length == 1057
assert FUSION_1N8Z.raw_rows == 446

# The 607-aa HER2 extracellular domain. 42.8% of all shard rows hang off this one antigen.
HER2_ECD = AntigenRecord(
    key='her2_ecd',
    antigen_sequence=(
        "TQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLI"
        "AHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILK"
        "GGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQS"
        "LTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDT"
        "FESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVC"
        "YGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEI"
        "TGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHH"
        "NTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQF"
        "LRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVA"
        "RCPSGVKPDLSYMPIWKFPDEEGACQPCPIN"
    ),
    antigen_sha256_32='ae38d43ea82403ec59e716cdfe9c5d3d',
    annotations=(
        AnnotationRow(target_name='', target_pdb='1n8z', target_uniprot='',
                      dataset='abbd', raw_rows=419),
        AnnotationRow(target_name='', target_pdb='1n8z', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=1),
        AnnotationRow(target_name='human her2', target_pdb='1n8z', target_uniprot='',
                      dataset='buzz', raw_rows=524346),
    ),
)
assert HER2_ECD.length == 607
assert HER2_ECD.raw_rows == 524766

# The 564-aa HER2 ECD variant from `aatp`. 93 shard rows, and the only place the accession q9uk79 enters the corpus at all -- the name `human_her2` then carries it onto 525,413 rows, the fusion included.
HER2_3WSQ = AntigenRecord(
    key='her2_3wsq',
    antigen_sequence=(
        "TQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLI"
        "AHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILK"
        "GGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQS"
        "LTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDT"
        "FESMPNPEGRYTFGASCVTACPYNYLSTDVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVC"
        "YGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEI"
        "TGYLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHH"
        "NTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLACHQLCARGHCWGPGPTQCVNCSQF"
        "LRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVA"
    ),
    antigen_sha256_32='4ee35229afcd289ca9edacc9151bde61',
    annotations=(
        AnnotationRow(target_name='human her2', target_pdb='3wsq', target_uniprot='q9uk79',
                      dataset='aatp', raw_rows=93),
    ),
)
assert HER2_3WSQ.length == 564
assert HER2_3WSQ.raw_rows == 93

# The 107-aa human IGKC constant domain, P01834. A different protein entirely.
IGKC = AntigenRecord(
    key='igkc',
    antigen_sequence=(
        "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDS"
        "TYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    ),
    antigen_sha256_32='d591c34db571eedb43a8703ef3c4eec2',
    annotations=(
        AnnotationRow(target_name='igkc_human', target_pdb='', target_uniprot='p01834',
                      dataset='patents', raw_rows=94),
    ),
)
assert IGKC.length == 107
assert IGKC.raw_rows == 94

#: Offsets of the two ``GGGGGGGG`` linker runs inside the fusion. A composite
#: construct announces itself; a design that segments on the linker sees three
#: parts, a design that does not sees one 1057-aa "antigen".
FUSION_LINKER_OFFSETS = (214, 442)

#: Distinct antigen sequences in the shards carrying a run of >= 8 glycines.
CORPUS_LINKER_ANTIGEN_COUNT = 28

FUSION_1N8Z_OVERLAP = PinnedCase(
    outcome=2,
    title="1N8Z fusion -> overlap/quarantine relation, but IGKC != HER2 identity",
    required=(
        "The fusion overlaps both IGKC and HER2, so both overlaps must produce "
        "QUARANTINE edges -- the fusion must not sit in generic validation while "
        "either component sits in train. Neither overlap may produce identity: "
        "construct_id(IGKC) != construct_id(HER2) and "
        "biological_target_id(IGKC) != biological_target_id(HER2), because P01834 "
        "and the HER2 ECD are different proteins."
    ),
    records=(FUSION_1N8Z, HER2_ECD, HER2_3WSQ, IGKC),
    evidence=(
    PairEvidence(
        left='igkc', right='fusion_1n8z',
        identity=0.101230, coverage_left=1.000000, coverage_right=0.101230,
        exact_substring=True,
        note='exact substring at offset 107; full coverage of IGKC, 10.12% of the fusion',
    ),
    PairEvidence(
        left='her2_ecd', right='fusion_1n8z',
        identity=0.574267, coverage_left=1.000000, coverage_right=0.574267,
        exact_substring=True,
        note='exact substring at offset 450; full coverage of HER2, 57.43% of the fusion',
    ),
    PairEvidence(
        left='igkc', right='her2_ecd',
        identity=0.131148, coverage_left=0.747664, coverage_right=0.131796,
        exact_substring=False,
        note='no relation: the two components of the fusion are unrelated proteins',
    ),
    PairEvidence(
        left='her2_3wsq', right='her2_ecd',
        identity=0.929160, coverage_left=1.000000, coverage_right=0.929160,
        exact_substring=True,
        note='the 564-mer is a truncation of the 607-mer: full coverage of the shorter side',
    ),
    ),
    committed_rule={
        "canonical_target_id": {
            "fusion_1n8z": "uniprot:q9uk79",
            "her2_ecd": "uniprot:q9uk79",
            "her2_3wsq": "uniprot:q9uk79",
            "igkc": "uniprot:p01834",
        },
        "split": {
            "fusion_1n8z": "train", "her2_ecd": "train",
            "her2_3wsq": "train", "igkc": "train",
        },
        "verdict": (
            "HALF SATISFIED. IGKC and HER2 are separate components today, because "
            "the committed rule has no containment or similarity edge -- that is "
            "the half to protect, and the reverted attempt broke it. The other "
            "half does not exist: there is no quarantine relation, so the fusion "
            "and its components could be split apart with nothing to notice. "
            "Note also that the HER2 component is canonically named uniprot:q9uk79 "
            "-- an accession from a single 93-row `aatp` annotation (`human her2 / "
            "3wsq / q9uk79`) that the name `human_her2` bridged onto 525,413 rows."
        ),
        "her2_component_raw_rows": 525413,
        "her2_component_min_pairwise_identity": 0.533586,
    },
    provenance=(
        "Fusion: `flab_shanehsazzadeh2023`, 446 shard rows, 445 in the v2 corpus. "
        "HER2 ECD: `buzz` + `abbd` + `structures-antibodies`, 524,766 shard rows. "
        "IGKC: `patents`, 94 shard rows, 0 in the v2 corpus -- keep_record drops "
        "them, but build_target_identity_index observes them anyway, so they still "
        "shape the partition."
    ),
)


# --------------------------------------------------------------------------- #
# Outcome 3 -- PDB 4GRW is two polymer entities
#
# Both antigens below appear on `structures-nanobodies` rows annotated
# `pdb: 4grw` and nothing else. Other shard rows carry the same two sequences
# with accessions, and they are different genes: Q9NPF7 (IL23A) and P29460
# (IL12B), 24.78% identical, sharing no 8-mer. A bare PDB id is not a target.
# --------------------------------------------------------------------------- #
# IL23A / Q9NPF7, 189 aa. One shard row also mis-labels it `p2ry12_human` under the same accession.
IL23A = AntigenRecord(
    key='il23a',
    antigen_sequence=(
        "MLGSRAVMLLLLLPWTAQGRAVPGGSSPAWTQCQQLSQKLCTLAWSAHPLVGHMDLREEGDEET"
        "TNDVPHIQCGDGCDPQGLRDNSQFCLQRIHQGLIFYEKLLGSDIFTGEPSLLPDSPVGQLHASL"
        "LGLSQLLQPEGHHWETQQIPSLSPSQPWQRLLLRFKILRSLQAFVAVAARVFAHGAATLSP"
    ),
    antigen_sha256_32='18441f075ed6b551c796f246a6bac733',
    annotations=(
        AnnotationRow(target_name='', target_pdb='4grw', target_uniprot='',
                      dataset='structures-nanobodies', raw_rows=2),
        AnnotationRow(target_name='il23a_human', target_pdb='', target_uniprot='q9npf7',
                      dataset='patents', raw_rows=213),
        AnnotationRow(target_name='p2ry12_human', target_pdb='', target_uniprot='q9npf7',
                      dataset='patents', raw_rows=1),
    ),
)
assert IL23A.length == 189
assert IL23A.raw_rows == 216

# IL12B / P29460, 328 aa.
IL12B = AntigenRecord(
    key='il12b',
    antigen_sequence=(
        "MCHQQLVISWFSLVFLASPLVAIWELKKDVYVVELDWYPDAPGEMVVLTCDTPEEDGITWTLDQ"
        "SSEVLGSGKTLTIQVKEFGDAGQYTCHKGGEVLSHSLLLLHKKEDGIWSTDILKDQKEPKNKTF"
        "LRCEAKNYSGRFTCWWLTTISTDLTFSVKSSRGSSDPQGVTCGAATLSAERVRGDNKEYEYSVE"
        "CQEDSACPAAEESLPIEVMVDAVHKLKYENYTSSFFIRDIIKPDPPKNLQLKPLKNSRQVEVSW"
        "EYPDTWSTPHSYFSLTFCVQVQGKSKREKKDRVFTDKTSATVICRKNASISVRAQDRYYSSSWS"
        "EWASVPCS"
    ),
    antigen_sha256_32='997621c8a338440d7db35c06ca18632d',
    annotations=(
        AnnotationRow(target_name='', target_pdb='4grw', target_uniprot='',
                      dataset='structures-nanobodies', raw_rows=4),
        AnnotationRow(target_name='', target_pdb='5mzv', target_uniprot='',
                      dataset='structures-nanobodies', raw_rows=1),
        AnnotationRow(target_name='il12b_human', target_pdb='', target_uniprot='p29460',
                      dataset='patents', raw_rows=417),
    ),
)
assert IL12B.length == 328
assert IL12B.raw_rows == 422

PDB_4GRW_ENTITIES = PinnedCase(
    outcome=3,
    title="4GRW -> distinct polymer entities remain distinct targets",
    required=(
        "construct_id(IL23A) != construct_id(IL12B) and "
        "biological_target_id(IL23A) != biological_target_id(IL12B). Sharing a PDB "
        "entry carries NO identity implication; entity/chain-level information is "
        "what distinguishes them, and where it is missing the PDB code must not "
        "stand in for it."
    ),
    records=(IL23A, IL12B),
    evidence=(
    PairEvidence(
        left='il23a', right='il12b',
        identity=0.247788, coverage_left=0.444444, coverage_right=0.256098,
        exact_substring=False,
        note='different genes; no shared 8-mer',
    ),
    ),
    committed_rule={
        "canonical_target_id": {"il23a": "uniprot:p29460", "il12b": "uniprot:p29460"},
        "split": {"il23a": "val", "il12b": "val"},
        "component_min_pairwise_identity": 0.247788,
        "component_uniprots": ("p29460", "q9npf7"),
        "component_pdbs": ("4grw", "5mzv"),
        "verdict": (
            "VIOLATED TODAY. `pdb:4grw` is a full-strength merging node, so the two "
            "entities are one component whose canonical id is uniprot:p29460 -- the "
            "IL12B accession, silently applied to IL23A rows. Both land in val, so "
            "any per-target report over that val slice mixes two genes."
        ),
    },
    provenance=(
        "data/raw/asd-antibody-antigen: `structures-nanobodies` (pdb-only rows) and "
        "`patents` (accession-bearing rows). 216 and 422 shard rows; 1 and 2 rows "
        "survive into the v2 corpus."
    ),
)


# --------------------------------------------------------------------------- #
# Outcome 4 -- a real A-B-C similarity chain
#
# Three influenza B haemagglutinins from `patents`, each with its own UniProt
# accession. Found by scanning all 9,574 distinct antigen sequences: a 5-mer
# Jaccard prefilter over the 6,727 sequences of length 100..800 produced 12,185
# near-identical candidate pairs, and every open triple among them was confirmed
# by full alignment. This is the widest-separated chain in the corpus that has
# exactly one accession per node, so nothing about it is confounded with
# outcome 6.
#
# A~B and B~C both clear 93% identity at >= 93% reciprocal coverage; A~C is
# 88.43%. Any threshold in (0.8843, 0.9337] admits both edges and rejects the
# closing one: single-linkage merges all three, a cluster-level criterion does
# not. A lacks the 15-residue signal peptide `MKAIIVLLMVVTSNA` that B and C both
# carry, which is why its coverage runs above its identity.
# --------------------------------------------------------------------------- #
# Chain node A: `hema_inbyb` / P18880, 345 aa.
HEMA_A_INBYB = AntigenRecord(
    key='hema_inbyb',
    antigen_sequence=(
        "DRICTGITSSNSPHVVKTATQGEVNVTGVIPLTTTPTKSHFANLKGTKTRGKLCPNCLNCTDLD"
        "VALGRPMCMGTIPSAKASILHEVRPVTSGCFPIMHDRTKIRQLPNLLRGYENIRLSTHNVINAE"
        "RAPGGPYRLGTSGSCPNVTSRNGFFATMAWAVPRDNKTATNPLTVEVPYICTKGEDQITVWGFH"
        "SDNKAQMKNLYGDSNPQKFTSSANGVTTHYVSQIGDFPNQTEDGGLPQSGRIVVDYMVQKPGKT"
        "GTIVYQRGVLLPQKVWCASGRSKVIKGSLPLIGEADCLHEKYGGLNKSKPYYTGEHAKAIGNCP"
        "IWVKTPLKLANGTKYRPPAKLLKER"
    ),
    antigen_sha256_32='0fe84f645dff0dc297d214c34b36214d',
    annotations=(
        AnnotationRow(target_name='hema_inbyb', target_pdb='', target_uniprot='p18880',
                      dataset='patents', raw_rows=2),
    ),
)
assert HEMA_A_INBYB.length == 345
assert HEMA_A_INBYB.raw_rows == 2

# Chain node B: `hema_inbte` / Q67378, 360 aa. The hub.
HEMA_B_INBTE = AntigenRecord(
    key='hema_inbte',
    antigen_sequence=(
        "MKAIIVLLMVVTSNADRICTGITSSNSPHVVKTATQGEVNVTGVIPLTTTPTKSHFANLKGTKT"
        "RGKLCPNCLNCTDLDVALARPMCIGTIPSAKASILHEVRPVTSGCFPIMHDRTKIRQLPNLLRG"
        "YENIRLSTHNVINAERAPGGPYRLGTSGSCPNVTSRSGFFATMAWAVPRDNKTATNPLTVEVPY"
        "ICTKGEDQITVWGFHSDNKIQMNKLYGDSNPQKFTSSANGVTTHYVSQIGGFPNQTEDGGLPQS"
        "GRIVVDYMVQKPGKTGTIVYQRGVLLPQKVWCASGRSKVIKGSLPLIGEADCLHEKYGGLNKSK"
        "PYYTGEHAKAIGNCPIWVKTPLKLANGTKYRPPAKLLKER"
    ),
    antigen_sha256_32='3d9e7e7feb2c7a686afb02da5b52fb21',
    annotations=(
        AnnotationRow(target_name='hema_inbte', target_pdb='', target_uniprot='q67378',
                      dataset='patents', raw_rows=2),
    ),
)
assert HEMA_B_INBTE.length == 360
assert HEMA_B_INBTE.raw_rows == 2

# Chain node C: `hema_inbvm` / Q67381, 362 aa.
HEMA_C_INBVM = AntigenRecord(
    key='hema_inbvm',
    antigen_sequence=(
        "MKAIIVLLMVVTSNADRICTGITSSNSPHVVKTATQGEVNVTGVIPLTTTPTKSHFANLKGTKT"
        "RGKLCPKCLNCTDLDVALARPKCTGTIPSAKASILHEVKPVTFGCFPIMHDRTKIRQLPNLLRG"
        "YEHIRLSTHNVINAEKAPGGPYKIGTSGSCPNVTNGNGFFATMAWAVPKNDNNKTATNSLTVEV"
        "PYICTEGEDQITVWGFHSDNEIQMVKLYGDSKPQKFTSSANGVTTHYVSQIGGFPNQAEDGGLP"
        "QSGRIVVDYMVQKSGKTGTITYQRGILLPQKVWCASGRSKVIKGSLPLIGEADCLHEKYGGLNK"
        "SKPYYTGEHAKAIGNCPIWVKTPLKLANGTKYRPPAKLLKER"
    ),
    antigen_sha256_32='fca047d7a0b2c4bccfb3c27a43f2bed0',
    annotations=(
        AnnotationRow(target_name='hema_inbvm', target_pdb='', target_uniprot='q67381',
                      dataset='patents', raw_rows=2),
    ),
)
assert HEMA_C_INBVM.length == 362
assert HEMA_C_INBVM.raw_rows == 2

SIMILARITY_CHAIN = PinnedCase(
    outcome=4,
    title="A-B-C similarity chain -> A and C may not merge unless the CLUSTER criterion permits",
    required=(
        "A~B and B~C are real edges and must be usable. A and C must NOT end up in "
        "one construct cluster on the strength of those two edges alone. Whatever "
        "admits a cluster -- a representative sequence every member must match, or "
        "a maximum-diameter bound -- has to be evaluated at cluster level, and the "
        "cluster report must carry its minimum pairwise identity AND minimum "
        "pairwise coverage so a percolated component is visible without re-deriving "
        "it."
    ),
    records=(HEMA_A_INBYB, HEMA_B_INBTE, HEMA_C_INBVM),
    evidence=(
    PairEvidence(
        left='hema_inbyb', right='hema_inbte',
        identity=0.939058, coverage_left=0.982609, coverage_right=0.941667,
        exact_substring=False,
        note='edge A~B: admitted at any threshold <= 0.9390',
    ),
    PairEvidence(
        left='hema_inbte', right='hema_inbvm',
        identity=0.933702, coverage_left=0.938889, coverage_right=0.933702,
        exact_substring=False,
        note='edge B~C: admitted at any threshold <= 0.9337',
    ),
    PairEvidence(
        left='hema_inbyb', right='hema_inbvm',
        identity=0.884298, coverage_left=0.930435, coverage_right=0.886740,
        exact_substring=False,
        note='closing edge A~C: rejected at any threshold > 0.8843',
    ),
    ),
    committed_rule={
        "canonical_target_id": {
            "hema_inbyb": "uniprot:p18880",
            "hema_inbte": "uniprot:q67378",
            "hema_inbvm": "uniprot:q67381",
        },
        "split": {"hema_inbyb": "train", "hema_inbte": "val", "hema_inbvm": "train"},
        "verdict": (
            "VACUOUSLY SATISFIED. All three are separate components -- but only "
            "because the committed rule has no sequence-similarity relation of any "
            "kind. The same absence is why outcome 1 depends on a name. Adding "
            "thresholded similarity + union-find would merge all three; that is the "
            "mechanism this outcome exists to forbid, and the reason the outcome "
            "cannot be judged by the split it produces today."
        ),
        "threshold_window_that_percolates": (0.8843, 0.9337),
    },
    provenance=(
        "data/raw/asd-antibody-antigen, dataset `patents`, 2 shard rows each; none "
        "survive keep_record into the v2 corpus, but all three are observed by "
        "build_target_identity_index."
    ),
)


# --------------------------------------------------------------------------- #
# Outcome 5 -- a generic name-only record spanning incompatible sequences
#
# `sars-cov2_wt` is written on shard rows carrying NO accession and NO PDB code.
# It is one of only six normalized names in the whole corpus that both span more
# than one antigen sequence and appear with no identifier at all -- the other
# five are its sibling variant names. It spans four constructs. Two of them are
# sequence-coherent with a third: 99.55% of the 223-aa RBD and 98.97% of the
# 292-aa NTD are matched inside the 539-aa S1 fragment, while covering only
# 41.19% and 53.62% of it. But the RBD and the NTD share no 8-mer with EACH
# OTHER, and the 666-aa S2 region shares no 8-mer with any of the other three.
# So the name spans at least two mutually non-homologous families, and cannot
# name either of them.
# --------------------------------------------------------------------------- #
# 223-aa receptor-binding domain.
WT_RBD_223 = AntigenRecord(
    key='wt_rbd',
    antigen_sequence=(
        "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGV"
        "SPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVG"
        "GNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRV"
        "VVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    ),
    antigen_sha256_32='af5ee94d7b888059635fa1e91fc810f8',
    annotations=(
        AnnotationRow(target_name='', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=6102),
        AnnotationRow(target_name='', target_pdb='7ch5', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=1),
        AnnotationRow(target_name='', target_pdb='7chb', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=1),
        AnnotationRow(target_name='', target_pdb='7chc', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=2),
        AnnotationRow(target_name='', target_pdb='7chf', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=2),
        AnnotationRow(target_name='', target_pdb='7djz', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=1),
        AnnotationRow(target_name='', target_pdb='7dk2', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=4),
        AnnotationRow(target_name='', target_pdb='7e7y', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=2),
        AnnotationRow(target_name='', target_pdb='7ean', target_uniprot='',
                      dataset='structures-antibodies', raw_rows=1),
        AnnotationRow(target_name='sars-cov2_wt', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=6101),
    ),
)
assert WT_RBD_223.length == 223
assert WT_RBD_223.raw_rows == 12217

# 292-aa N-terminal domain. No homology to the RBD.
WT_NTD_292 = AntigenRecord(
    key='wt_ntd',
    antigen_sequence=(
        "SQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGT"
        "KRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPF"
        "LGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIY"
        "SKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVG"
        "YLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLK"
    ),
    antigen_sha256_32='be45fbf1376c9e681f595e2a2da4a21f',
    annotations=(
        AnnotationRow(target_name='', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=481),
        AnnotationRow(target_name='sars-cov2_wt', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=481),
    ),
)
assert WT_NTD_292.length == 292
assert WT_NTD_292.raw_rows == 962

# 539-aa S1 fragment: contains near-copies of both the NTD and the RBD.
WT_S1_539 = AntigenRecord(
    key='wt_s1',
    antigen_sequence=(
        "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTW"
        "FHAISGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKV"
        "CEFQFCNDPFLGVYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFK"
        "NIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWT"
        "AGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPT"
        "ESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKL"
        "NDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNY"
        "LYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSF"
        "ELLHAPATVCGPKKSTNLVKNKCVNFN"
    ),
    antigen_sha256_32='5fff89447258877507d3143480e1c481',
    annotations=(
        AnnotationRow(target_name='', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=82),
        AnnotationRow(target_name='sars-cov2_alpha', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=4),
        AnnotationRow(target_name='sars-cov2_beta', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=4),
        AnnotationRow(target_name='sars-cov2_delta', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=5),
        AnnotationRow(target_name='sars-cov2_gamma', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=1),
        AnnotationRow(target_name='sars-cov2_kappa', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=4),
        AnnotationRow(target_name='sars-cov2_wt', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=61),
    ),
)
assert WT_S1_539.length == 539
assert WT_S1_539.raw_rows == 161

# 666-aa S2 region. No homology to any of the other three.
WT_S2_666 = AntigenRecord(
    key='wt_s2',
    antigen_sequence=(
        "FNGLTGTGVLTESNKKFLPFQQFGRDIDDTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSN"
        "QVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGI"
        "CASYQTQTNSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPINFTISVTTEILPVSMTKT"
        "SVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFG"
        "GFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLP"
        "PLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQ"
        "FNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILARLDKVEAE"
        "VQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQ"
        "SAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTH"
        "NTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKE"
        "IDRLNEVANNLNESLIDLQELGKYEQ"
    ),
    antigen_sha256_32='de0d346e851eb0bd502970f1f24ee52f',
    annotations=(
        AnnotationRow(target_name='', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=212),
        AnnotationRow(target_name='sars-cov2_alpha', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=4),
        AnnotationRow(target_name='sars-cov2_beta', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=4),
        AnnotationRow(target_name='sars-cov2_delta', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=7),
        AnnotationRow(target_name='sars-cov2_gamma', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=6),
        AnnotationRow(target_name='sars-cov2_mu', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=1),
        AnnotationRow(target_name='sars-cov2_wt', target_pdb='', target_uniprot='',
                      dataset='covid-19', raw_rows=171),
    ),
)
assert WT_S2_666.length == 666
assert WT_S2_666.raw_rows == 405

#: Normalized names in the whole corpus that span > 1 antigen sequence AND never
#: carry a UniProt accession or a PDB code on any row. All six are SARS-CoV-2
#: variant labels, so this outcome is measured on the only population that
#: exhibits it.
CORPUS_PURE_NAME_ONLY_MULTISEQ_NAMES = (
    "sars_cov2_alpha", "sars_cov2_beta", "sars_cov2_delta",
    "sars_cov2_gamma", "sars_cov2_omicron", "sars_cov2_wt",
)

NAME_ONLY_GENERIC = PinnedCase(
    outcome=5,
    title="generic name-only record -> attaches to one sequence-coherent family, else quarantine",
    required=(
        "`name:sars_cov2_wt` may attach to at most one sequence-coherent family. "
        "It spans two that share no homology, so it must attach to neither and "
        "quarantine instead. Concretely: construct_id(RBD) != construct_id(S2) and "
        "biological_target_id(RBD) != biological_target_id(S2). Name approval is "
        "computed ONCE against the sequence/accession clusters and never iterated "
        "to a fixed point."
    ),
    records=(WT_RBD_223, WT_NTD_292, WT_S1_539, WT_S2_666),
    evidence=(
    PairEvidence(
        left='wt_rbd', right='wt_ntd',
        identity=0.291536, coverage_left=0.417040, coverage_right=0.318493,
        exact_substring=False,
        note='two disjoint spike domains: no homology',
    ),
    PairEvidence(
        left='wt_rbd', right='wt_s1',
        identity=0.411874, coverage_left=0.995516, coverage_right=0.411874,
        exact_substring=False,
        note='the RBD sits inside S1: full RBD coverage, 41.19% of S1',
    ),
    PairEvidence(
        left='wt_ntd', right='wt_s1',
        identity=0.533210, coverage_left=0.989726, coverage_right=0.536178,
        exact_substring=False,
        note='the NTD sits inside S1: full NTD coverage, 53.62% of S1',
    ),
    PairEvidence(
        left='wt_rbd', right='wt_s2',
        identity=0.204142, coverage_left=0.618834, coverage_right=0.207207,
        exact_substring=False,
        note='no homology',
    ),
    PairEvidence(
        left='wt_ntd', right='wt_s2',
        identity=0.220913, coverage_left=0.513699, coverage_right=0.225225,
        exact_substring=False,
        note='no homology',
    ),
    PairEvidence(
        left='wt_s1', right='wt_s2',
        identity=0.273713, coverage_left=0.374768, coverage_right=0.303303,
        exact_substring=False,
        note='no homology: S1 and S2 are the two halves of spike',
    ),
    ),
    committed_rule={
        "canonical_target_id": {
            "wt_rbd": "pdb:7ch5", "wt_ntd": "pdb:7ch5",
            "wt_s1": "pdb:7ch5", "wt_s2": "pdb:7ch5",
        },
        "split": {"wt_rbd": "train", "wt_ntd": "train", "wt_s1": "train", "wt_s2": "train"},
        "component_distinct_antigens": 12,
        "component_names": (
            "sars_cov2_alpha", "sars_cov2_beta", "sars_cov2_delta", "sars_cov2_gamma",
            "sars_cov2_kappa", "sars_cov2_mu", "sars_cov2_wt",
        ),
        "component_pdbs": (
            "7ch5", "7chb", "7chc", "7chf", "7djz", "7dk2", "7e7y", "7ean",
            "7ek0", "7f6z", "8hrd",
        ),
        "component_raw_rows": 16675,
        "component_min_pairwise_identity": 0.204142,
        "component_min_pairwise_coverage": 0.207207,
        "verdict": (
            "VIOLATED TODAY, AND PERCOLATED. All four constructs share one "
            "canonical_target_id. The component has swallowed 12 antigen sequences, "
            "7 variant names and 11 PDB codes, and its two most distant members "
            "share 20.41% identity. It is named `pdb:7ch5` -- a PDB code, for a "
            "population whose rows mostly carry no PDB code at all."
        ),
    },
    provenance=(
        "data/raw/asd-antibody-antigen, dataset `covid-19` (name-only rows) plus a "
        "handful of `structures-antibodies` rows that contribute the PDB codes. "
        "12,217 / 962 / 161 / 405 shard rows; 6,008 / 479 / 61 / 168 in the v2 "
        "corpus, all on the train side."
    ),
)


# --------------------------------------------------------------------------- #
# Outcome 6 -- one exact sequence, conflicting accessions
#
# A single 192-aa sequence annotated `rac1_human / P63000` on four `patents`
# rows and `rac1_mouse / P63001` on a fifth. The sequences are byte-identical --
# not similar, identical -- so no threshold is involved and no tie-break between
# the accessions is defensible. 30 of the 9,574 distinct antigen sequences in
# the shards carry more than one canonical UniProt accession; the widest carries
# eight.
# --------------------------------------------------------------------------- #
# 192 aa, P63000 and P63001 over one byte-identical sequence.
RAC1_CONFLICT = AntigenRecord(
    key='rac1',
    antigen_sequence=(
        "MQAIKCVVVGDGAVGKTCLLISYTTNAFPGEYIPTVFDNYSANVMVDGKPVNLGLWDTAGQEDY"
        "DRLRPLSYPQTDVFLICFSLVSPASFENVRAKWYPEVRHHCPNTPIILVGTKLDLRDDKDTIEK"
        "LKEKKLTPITYPQGLAMAKEIGAVKYLECSALTQRGLKTVFDEAIRAVLCPPPVKKRKRKCLLL"
    ),
    antigen_sha256_32='48db732bf13b377c21467adf0bac39d4',
    annotations=(
        AnnotationRow(target_name='rac1_human', target_pdb='', target_uniprot='p63000',
                      dataset='patents', raw_rows=4),
        AnnotationRow(target_name='rac1_mouse', target_pdb='', target_uniprot='p63001',
                      dataset='patents', raw_rows=1),
    ),
)
assert RAC1_CONFLICT.length == 192
assert RAC1_CONFLICT.raw_rows == 5

#: Distinct antigen sequences in the shards carrying > 1 canonical UniProt
#: accession, and the largest number of accessions on any single sequence.
CORPUS_ACCESSION_CONFLICT_SEQUENCES = 30
CORPUS_MAX_ACCESSIONS_PER_SEQUENCE = 8

ACCESSION_CONFLICT = PinnedCase(
    outcome=6,
    title="exact sequence, conflicting accessions -> recorded as a CONFLICT",
    required=(
        "The resolver must surface this as a conflict a caller can enumerate and "
        "act on. It must NOT silently name the record after whichever accession "
        "sorts first, or wins on namespace rank, or carries more rows: all three "
        "are the same failure, choosing a winner where the data reports a "
        "disagreement."
    ),
    records=(RAC1_CONFLICT,),
    committed_rule={
        "canonical_target_id": {"rac1": "uniprot:p63000"},
        "split": {"rac1": "train"},
        "conflicting_accessions": ("p63000", "p63001"),
        "stats_keys": (
            "target_identity_node_count", "target_components", "target_alias_merges",
            "target_sequence_merges", "target_rows_without_identifier",
            "target_rows_without_identity",
        ),
        "verdict": (
            "VIOLATED TODAY. The seq node fuses both accessions into one component "
            "and canonical_target_id_from_nodes returns min() by (namespace rank, "
            "node), so `uniprot:p63000` wins on nothing but lexicographic order and "
            "P63001 disappears. TargetIdentityIndex.stats() has no conflict counter, "
            "so the disagreement is not reported anywhere either."
        ),
    },
    provenance="data/raw/asd-antibody-antigen, dataset `patents`, 5 shard rows; 0 in the v2 corpus.",
)


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

ALL_CASES: Tuple[PinnedCase, ...] = (
    OMICRON_INDEL,
    FUSION_1N8Z_OVERLAP,
    PDB_4GRW_ENTITIES,
    SIMILARITY_CHAIN,
    NAME_ONLY_GENERIC,
    ACCESSION_CONFLICT,
)

CASES_BY_OUTCOME: Dict[int, PinnedCase] = {c.outcome: c for c in ALL_CASES}

ALL_RECORDS: Tuple[AntigenRecord, ...] = tuple(
    r for case in ALL_CASES for r in case.records
)

assert len(ALL_CASES) == 6
assert sorted(CASES_BY_OUTCOME) == [1, 2, 3, 4, 5, 6]
assert len({r.key for r in ALL_RECORDS}) == len(ALL_RECORDS)


#: The error asymmetry, declared HERE and BEFORE any measurement, per contract
#: requirement D. A false MERGE is the more expensive error: it destroys target
#: diversity, corrupts every per-target number computed downstream, and is
#: invisible in a leakage audit because a merged pair never straddles anything.
#: A false SPLIT costs leakage, which a leakage audit does detect and report. So
#: the operating point is chosen to tolerate more false splits than false
#: merges. Outcome 1 is the bound in the other direction: it is the false split
#: this corpus is known to punish hardest, and it must not happen.
PREDECLARED_ERROR_ASYMMETRY = "tolerate_false_splits_over_false_merges"

#: Threshold selection must not validate on its own selection set (requirement
#: A). Labelled pairs are partitioned by FAMILY, not by pair, so a family can
#: contribute to the calibration set or the audit set but never both -- Omicron
#: must not pick an operating point and then confirm it. Families that decide
#: their own outcome are named here so an implementation cannot quietly leave
#: them in calibration.
LEAVE_ONE_FAMILY_OUT_AUDIT_FAMILIES = (
    "sars_cov2_spike",
    "her2_igkc_fusion",
    "il12_il23",
    "influenza_b_haemagglutinin",
    "rac1",
)
