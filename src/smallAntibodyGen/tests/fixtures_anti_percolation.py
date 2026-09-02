"""
The anti-percolation fixture, re-sourced so that it has power at the shipped
operating point.

WHY THIS MODULE EXISTS
----------------------
`test_target_identity_acceptance.py` shipped its outcome-4 test with a
prominent warning: the three influenza-B haemagglutinins it used form NO
construct edge at the thresholds an implementation would plausibly choose, so
``construct(A) != construct(C)`` held trivially and the cluster criterion was
never exercised. Two adversarial reviewers found that independently. The warning
also said what the repair had to be -- a fixture that closes a triangle AT THE
OPERATING THRESHOLDS, with the A~B and B~C edges positively asserted.

That was re-measured before anything was written here, in the engine's own
metric rather than the reference aligner's, and the warning was confirmed:

    influenza-B haemagglutinins, at construct (0.99 / 0.95 / 30)
        A~B  identity 0.9797  min-coverage 0.9583   -> no edge (identity)
        B~C  identity 0.9309  min-coverage 1.0000   -> no edge (identity)
        A~C  identity 0.9193  min-coverage 0.9586   -> no edge (identity)
        complete-linkage and single-linkage agree; merges_refused == 0

The original fixture is therefore left exactly where it is -- it is real, and the
Needleman-Wunsch relations it pins are still true of the sequences beside it --
and this module supplies a triple that is genuinely open instead.

HOW THIS TRIPLE WAS FOUND
-------------------------
By scanning, not by inventing, and not by moving the operating point until the
old fixture worked. `scripts/scan_anti_percolation_triples.py` aligned every
candidate pair among all 9,574 distinct antigen sequences in
``data/raw/asd-antibody-antigen`` at the construct operating point and reported
every triple whose two chain edges are admitted and whose closing edge is
refused. It found 6,412, of which the report retains the top 60 under the scan's
own ranking (measured refusals first, then by source rows, then by the identity
window).

This triple was then chosen from those 60 by a criterion the scan does not rank
on, and saying so matters: the scan's `window` field measures separation in
IDENTITY, and for every candidate here that window is zero or negative, because
these are exact sub-ranges of one protein. The selection was by the separation
that actually decides the outcome -- COVERAGE -- and this triple has the widest
coverage window of the 60 (0.0383 against a next-best 0.0287). It is not the
best-separated triple in the corpus on any metric the scan itself sorts by; it
is the best on the metric the operating point uses.

WHAT IT IS
----------
Three SARS-CoV-2 spike receptor-binding-domain constructs. They are nested
sub-ranges of one protein with three different purification tags:

    A  199 aa, PDB 7E3O                     ...APATVCGP + HHHHHHHH  (8x His)
    B  205 aa, PDB 7KMG/7KMH/7KMI/7MMO      ...APATVCGP + HHHHHH    (6x His)
    C  209 aa, name `sars_cov2_rbd`, 67,058 shard rows   no tag

The tag lengths are not decoration, they ARE the arithmetic. Strip the trailing
histidines and A becomes a 191-residue exact substring of C and B a 199-residue
one, which is why A~C coverage is 191/209 = 0.9139 and B~C is 199/209 = 0.9522.

The measured relations are the reason this fixture is worth having:

    A~B  identity 1.000000  min-coverage 0.960976  overlap 197  ADMITTED
    B~C  identity 1.000000  min-coverage 0.952153  overlap 199  ADMITTED
    A~C  identity 1.000000  min-coverage 0.913876  overlap 191  REFUSED

**Every identity is exactly 1.0.** Identity cannot distinguish these three at any
threshold whatsoever, because each is a byte-exact sub-range of the RBD. The
chain is closed in identity and open only in COVERAGE, over the window
(0.913876, 0.952153] -- 3.83 points wide, with the shipped construct coverage
threshold of 0.95 inside it. A criterion that looks only at identity merges all
three; single-linkage over the two admitted edges merges all three; only a
bounded criterion that evaluates the closing pair keeps A and C apart.

Exact containment fails on all three pairs, because the tags differ: ``A in B``
is False even though A's 199 residues are all present in B. That is the same
mechanism that defeated a previous attempt on the Omicron pair, arriving here by
a different route.

C carries 67,058 of the corpus's 1,227,083 shard rows, so this is not an obscure
corner of the data. Splitting A from C at the CONSTRUCT level costs no leakage:
all three remain one biological family (min-coverage 0.9139 clears the family
threshold of 0.80), so they still share a split group. The fixture separates
constructs, not targets.

NOTHING HERE IS SYNTHETIC
-------------------------
Every sequence is byte-for-byte what the shards carry after
``prepare_antibody_antigen.clean_aa_sequence``; every annotation is what the
curator wrote; every row count is the number of shard rows behind that sequence.
`SYNTHESIS_LEDGER` is empty, and the acceptance suite checks it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

#: Empty, like the ledger in `fixtures_target_identity`. If a future edit adds a
#: synthetic case it belongs here, named, so nobody quotes it as evidence about
#: the corpus.
SYNTHESIS_LEDGER: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ChainRecord:
    """One antigen sequence in the chain, with its shard provenance."""

    key: str
    antigen_sequence: str
    antigen_sha256_32: str
    #: ``(target_name, target_pdb, target_uniprot)`` as the producer normalises
    #: them, one entry per distinct annotation observed over this sequence.
    annotations: Tuple[Tuple[str, str, str], ...]
    #: Shard rows carrying this sequence, over all annotations.
    raw_rows: int

    @property
    def length(self) -> int:
        return len(self.antigen_sequence)

    def rows(self) -> Tuple[Dict[str, object], ...]:
        """This record as raw-row dicts, one per annotation."""
        return tuple(
            {
                "metadata": {
                    "target_name": name,
                    "target_pdb": pdb,
                    "target_uniprot": uniprot,
                },
                "antigen_sequence": self.antigen_sequence,
            }
            for name, pdb, uniprot in self.annotations
        )


# 199-aa RBD construct with a seven-histidine tag. PDB 7E3O.
RBD_199 = ChainRecord(
    key="rbd_199",
    antigen_sequence=(
        "PFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSF"
        "VIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPF"
        "ERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPH"
        "HHHHHHH"
    ),
    antigen_sha256_32="74bb6976aeb82e571ce1128300de2ab4",
    annotations=(("", "7e3o", ""),),
    raw_rows=1,
)
assert RBD_199.length == 199

# 205-aa RBD construct with a six-histidine tag, starting six residues earlier.
RBD_205 = ChainRecord(
    key="rbd_205",
    antigen_sequence=(
        "FPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCF"
        "TNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLF"
        "RKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHA"
        "PATVCGPHHHHHH"
    ),
    antigen_sha256_32="7297c26177a53d2d03e1920b59ac0bd6",
    annotations=(("", "7kmg", ""), ("", "7kmh", ""), ("", "7kmi", ""),
                 ("", "7mmo", "")),
    raw_rows=6,
)
assert RBD_205.length == 205

# 209-aa untagged RBD, the corpus's canonical `sars_cov2_rbd` construct.
RBD_209 = ChainRecord(
    key="rbd_209",
    antigen_sequence=(
        "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGV"
        "SPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVG"
        "GNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRV"
        "VVLSFELLHAPATVCGP"
    ),
    antigen_sha256_32="92ebca1a7958c1677797a22da49e0b98",
    annotations=(("sars_cov2_rbd", "", ""),),
    raw_rows=67058,
)
assert RBD_209.length == 209

CHAIN_RECORDS: Tuple[ChainRecord, ...] = (RBD_199, RBD_205, RBD_209)
assert len({r.key for r in CHAIN_RECORDS}) == 3


@dataclass(frozen=True)
class ChainEdge:
    """One measured relation in the engine's own metric.

    Recorded in the metric the operating point is expressed in -- local affine
    Smith-Waterman, BLOSUM62, open 11 extend 1 -- and NOT in the reference
    Needleman-Wunsch of `fixtures_target_identity`. Quoting a number measured
    with one aligner against a threshold expressed in another is how the previous
    fixture came to look like it had power.
    """

    left: str
    right: str
    identity: float
    cov_left: float
    cov_right: float
    overlap: int
    admitted_at_construct_point: bool
    note: str


CHAIN_EDGES: Tuple[ChainEdge, ...] = (
    ChainEdge(
        left="rbd_199", right="rbd_205",
        identity=1.000000, cov_left=0.989950, cov_right=0.960976, overlap=197,
        admitted_at_construct_point=True,
        note="edge A~B: admitted; the 199-mer's 8x His tag is what keeps "
             "coverage off 1.0",
    ),
    ChainEdge(
        left="rbd_205", right="rbd_209",
        identity=1.000000, cov_left=0.970732, cov_right=0.952153, overlap=199,
        admitted_at_construct_point=True,
        note="edge B~C: admitted, and only just -- 0.952153 against a 0.95 "
             "threshold is what makes the window real rather than comfortable",
    ),
    ChainEdge(
        left="rbd_199", right="rbd_209",
        identity=1.000000, cov_left=0.959799, cov_right=0.913876, overlap=191,
        admitted_at_construct_point=False,
        note="closing edge A~C: REFUSED on coverage alone, at identity 1.0",
    ),
)


def edge(left: str, right: str) -> ChainEdge:
    """Look up a measured edge by fixture key, in either order."""
    for candidate in CHAIN_EDGES:
        if {candidate.left, candidate.right} == {left, right}:
            return candidate
    raise KeyError(f"({left!r}, {right!r}) is not a measured edge of this chain")


def record(key: str) -> ChainRecord:
    """Look up a record by fixture key."""
    for candidate in CHAIN_RECORDS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"{key!r} is not in the anti-percolation chain")


#: The open window, in COVERAGE rather than identity, measured rather than
#: chosen: every identity in this chain is exactly 1.0, so identity cannot
#: separate these three at any threshold. ``(refused, admitted]``.
COVERAGE_WINDOW: Tuple[float, float] = (0.913876, 0.952153)

#: What the shipped operating point uses for the construct relation. Pinned here
#: so the fixture-power test can assert the window CONTAINS it, rather than
#: assuming it does.
CONSTRUCT_COVERAGE_THRESHOLD = 0.95

#: Under single-linkage, the two admitted edges weld all three into one
#: construct. Under a bounded criterion they do not. The acceptance test asserts
#: both halves, because an anti-percolation test that cannot show what the wrong
#: algorithm would have done is not testing the criterion.
SINGLE_LINKAGE_WOULD_MERGE = ("rbd_199", "rbd_205", "rbd_209")
BOUNDED_CRITERION_KEEPS = (("rbd_199", "rbd_205"), ("rbd_209",))

#: The scan that produced this fixture, for reproduction.
PROVENANCE = (
    "scripts/scan_anti_percolation_triples.py --identity 0.99 --coverage 0.95 "
    "--overlap 30, over all 9,574 distinct antigen sequences in "
    "data/raw/asd-antibody-antigen (20 shards, 1,227,083 rows). 6,412 open "
    "triples found; this is the best-separated one whose closing edge was "
    "aligned and refused rather than never proposed. Report retained at "
    "outputs/anti-percolation-triples-construct.json."
)
