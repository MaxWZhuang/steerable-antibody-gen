"""
The acceptance contract for antigen target identity, as executable specification.

This file was written BEFORE the implementation existed, so that the next attempt
would be gated rather than judged afterwards. The engine landed on 2026-09-01 and
every specification here now passes; the outcomes are unchanged from the day they
were pinned, and §"WHY THE SPECIFICATIONS MOVED OFF THE COMMITTED RULE" records
the only two edits and why each was forced. Every test names the pinned outcome
it encodes and the real record it came from; the records themselves live in
``fixtures_target_identity`` and were distilled out of
``data/raw/asd-antibody-antigen`` and ``data/processed/antibody_antigen_v2`` so
this suite runs in a few seconds and needs no ``data/`` directory. The two
slowest tests are the ones that earn it: re-aligning every recorded relation,
and re-executing the 1,400-line producer to rebuild its partition.

THREE KINDS OF TEST LIVE HERE
-----------------------------
1. FIXTURE-POWER tests. Unmarked, and they pass today. They prove the fixtures
   actually exercise the conditions the outcomes are about -- that the Omicron
   pair really is a clean 2-residue insertion, that IGKC really is an exact
   substring of the fusion, that the A-B-C chain really is open -- and that the
   16-record subset reproduces, exactly, the canonical id and split the full
   corpus gives those records. Without these, every other test in the file could
   be passing or failing for reasons unrelated to the contract.

2. REGRESSION GUARDS. Unmarked, and they pass today, because the committed rule
   happens to get that part right. They exist because the reverted attempt broke
   two of them. Do not delete them when the new engine lands: re-point them.

3. SPECIFICATIONS. These were all marked ``xfail(strict=True)`` while the
   implementation was absent. `smallAntibodyGen.target_identity` has now landed
   and every one of them passes, so the markers are gone. ``strict=True`` did
   its job: it made satisfying an outcome force an edit here rather than let a
   silent XPASS pass for progress.

WHY THE SPECIFICATIONS MOVED OFF THE COMMITTED RULE (2026-09-01)
----------------------------------------------------------------
Five of the twelve specifications were originally written against
``scripts/prepare_antibody_antigen.py`` through the ``committed`` fixture,
because their outcomes were expressible in the vocabulary that rule already had.
Those five are now written against the engine instead. This was not a
convenience: THE CONTRACT AS COMMITTED WAS UNSATISFIABLE AS A WHOLE, and here is
the proof.

``test_fixture_subset_reproduces_the_corpus_verdict`` pins, for every fixture
record, the exact ``canonical_target_id`` the full 1,227,083-row shard set gives
it under the committed rule. For outcome 3 that pinned value is
``uniprot:p29460`` for BOTH ``il23a`` and ``il12b``. And
``test_outcome_3_4grw_entities_remain_distinct_targets`` required
``committed.canonical_id("il23a") != committed.canonical_id("il12b")``. Those two
cannot both hold of the same object. The same contradiction exists for outcome 5
(``wt_rbd`` and ``wt_s2`` are both pinned to ``pdb:7ch5`` and required to
differ) and for outcome 6 (``rac1`` is pinned to ``uniprot:p63000`` and required
not to be it).

So the choice was never "edit the contract or not". It was which half to keep.
The half kept is the corpus-fidelity test, because it records something that
stays true forever -- what the rule that BUILT the shipped corpus did to these
sixteen records -- and it is the evidence behind every "VIOLATED TODAY" verdict
in the fixture module. The specifications move to the engine, which is what the
file's own instruction to "re-point them" always meant. The committed rule is
literally untouched by this change: `scripts/prepare_antibody_antigen.py` has not
been edited, so the shipped corpus and every stored `canonical_target_id` still
come from it. The engine is reached through
`scripts/resolve_target_identity.py`, which writes no corpus.

The three regression guards still run against the committed rule, unchanged and
passing, and `test_target_identity_engine.py` carries engine-side twins of them
so that neither rule can regress without a test noticing.

CAVEAT ON WHAT THIS SUITE CANNOT SEE
------------------------------------
These are 16 antigen sequences out of 9,574. They are enough to falsify a design
and not remotely enough to validate one: passing every test here says the six
pinned outcomes hold on the six pinned records, and says nothing about the error
rate on the other 9,558. That measurement is the calibration/audit protocol of
contract requirement A, which this suite pins the SHAPE of (outcome-A tests) but
cannot itself perform without the corpus.

The two things this suite specifically cannot see, and where they are measured
instead:

- **Percolation.** The failure that reverted a previous attempt was a 77%
  component on the relation the split keys on, and over 16 records the largest
  split group is a handful. `scripts/resolve_target_identity.py` runs the engine
  over all 9,574 and publishes the component-size distribution, the largest
  group's share, and the high-degree bridges it refused.
- **Error rate.** 116 labelled pairs cannot estimate one.
  `entity_resolution.synthetic` supplies a planted population, disjoint from
  these families by construction, so `calibration_report()` has something to
  score that this file does not contain.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pytest


def _load_sibling(name: str):
    """Import a module sitting next to this test file, by path.

    ``src/smallAntibodyGen/tests`` deliberately has no ``__init__.py`` -- adding
    one would change collection for every other test module here -- so a
    relative import is unavailable and a bare ``import fixtures_target_identity``
    would claim a generic top-level name. Loading by path is what the rest of
    this suite already does for ``scripts/*.py``.
    """
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"smallAntibodyGen_tests_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_sibling("fixtures_target_identity")
#: The re-sourced anti-percolation chain. See that module's docstring for why the
#: original outcome-4 fixture could not be used at the shipped operating point.
ap = _load_sibling("fixtures_anti_percolation")


# --------------------------------------------------------------------------- #
# The committed rule: scripts/prepare_antibody_antigen.py
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _load_script(scripts_dir: str, name: str):
    """Import a ``scripts/*.py`` module by path, once per interpreter.

    Cached because ``project_root`` is function scoped, so a module-scoped
    fixture would raise ScopeMismatch, and re-executing a 1,400-line script for
    every test in the file is pure waste.
    """
    directory = Path(scripts_dir)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paa(project_root: Path):
    """The committed target-identity rule, ``scripts/prepare_antibody_antigen.py``."""
    return _load_script(str(project_root.parents[1] / "scripts"), "prepare_antibody_antigen")


class CommittedPartition:
    """The committed rule's partition, built over a chosen set of fixture records.

    Two-phase like the production index: observe every annotation of every
    record, finalize, then query. Building it over a subset rather than the
    corpus is only defensible because
    ``test_fixture_subset_reproduces_the_corpus_verdict`` proves the subset gives
    the same answer the corpus does for all 16 records.
    """

    def __init__(self, paa_module, records: Sequence[fx.AntigenRecord], drop_names: bool = False):
        self._paa = paa_module
        self._index = paa_module.TargetIdentityIndex()
        self._nodes: Dict[str, List[str]] = {}
        for record in records:
            for row in record.rows():
                metadata = dict(row["metadata"])
                if drop_names:
                    metadata["target_name"] = ""
                self._index.observe(
                    paa_module.extract_target_nodes(metadata, row["antigen_sequence"])
                )
            metadata = dict(record.merged_metadata())
            if drop_names:
                metadata["target_name"] = ""
            self._nodes[record.key] = paa_module.extract_target_nodes(
                metadata, record.antigen_sequence
            )
        self._index.finalize()

    def canonical_id(self, key: str) -> str:
        return self._index.canonical_id(self._nodes[key])

    def split(self, key: str) -> str:
        return self._paa.deterministic_split(self.canonical_id(key))

    def stats(self) -> Dict[str, int]:
        return self._index.stats()


@pytest.fixture
def committed(paa):
    """The committed rule's partition over every fixture record."""
    return CommittedPartition(paa, fx.ALL_RECORDS)


# --------------------------------------------------------------------------- #
# The target rule: the module that does not exist yet
# --------------------------------------------------------------------------- #

CONTRACT = """
The acceptance contract expects ONE module -- importable as
`smallAntibodyGen.target_identity` or as `scripts/target_identity.py` --
exposing at least:

    resolve_target_identity(rows) -> TargetIdentityResolution
        rows: iterable of mappings carrying `metadata` (target_name /
              target_pdb / target_uniprot) and `antigen_sequence`, i.e. exactly
              what `extract_target_nodes` reads today and nothing more.

    TargetIdentityResolution
        .construct_id(key_or_row)         -> str
        .biological_target_id(key_or_row) -> str
        .split_group_id(key_or_row, claim) -> str
              claim in {"generic", "unseen_mutant"}; a generic held-out-target
              split and an AVIDa-style label-switching benchmark are different
              claims and must not share one grouping.
        .quarantine_partners(key_or_row)  -> frozenset[str]
              construct ids that OVERLAP this one without being identical --
              the relation local containment produces. Never an identity.
        .accession_conflicts()            -> tuple of records with
              .antigen_sha256, .accessions, and no chosen winner.
        .construct_cluster_report(construct_id) -> report with
              .members, .representative, .min_pairwise_identity,
              .min_pairwise_coverage, .max_diameter
        .name_decisions()                 -> tuple of records with
              .name, .approved, .attached_to, .reason -- computed ONCE against
              the sequence/accession clusters, never iterated to a fixed point.
        .calibration_report() / .audit_report() -> reports with
              .families, .false_merges, .false_splits, .tolerated_error
              over DISJOINT family sets.

The three ids must come from three separate relations. A design that derives all
three from one union-find partition fails contract requirement B no matter what
these tests report.
"""

_TARGET_MODULE_CANDIDATES = (
    "smallAntibodyGen.target_identity",
    "target_identity",
)

REQUIRED_SURFACE = (
    "resolve_target_identity",
)


class ContractNotImplemented(AssertionError):
    """Raised when the target-identity module or one of its methods is absent."""


def load_target_rule():
    """Import the target-identity module, or explain precisely what is missing.

    Returns:
        The imported module.

    Raises:
        ContractNotImplemented: when no candidate module imports, or when the
            module that does import lacks part of the required surface.
    """
    tried = []
    for name in _TARGET_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            tried.append(f"{name}: {exc}")
            continue
        missing = [n for n in REQUIRED_SURFACE if not hasattr(module, n)]
        if missing:
            raise ContractNotImplemented(
                f"{name} imports but is missing {missing}.\n{CONTRACT}"
            )
        return module
    raise ContractNotImplemented(
        "no target-identity module found; tried " + "; ".join(tried) + "\n" + CONTRACT
    )


def resolve(records: Sequence[fx.AntigenRecord]):
    """Run the target rule over fixture records, label-blind.

    Args:
        records: Fixture records to resolve.

    Returns:
        The resolution object described in ``CONTRACT``.
    """
    module = load_target_rule()
    rows = [row for record in records for row in record.rows()]
    return module.resolve_target_identity(rows)


def requires_method(resolution, name: str):
    """Fetch one method of the resolution, or fail naming the missing surface."""
    method = getattr(resolution, name, None)
    if method is None:
        raise ContractNotImplemented(
            f"TargetIdentityResolution has no {name!r}.\n{CONTRACT}"
        )
    return method


# Two reason strings, so the report can separate "answers the question wrongly"
# from "has no vocabulary for the question".
WRONG_ANSWER = "SPEC: the committed rule answers this, and answers it wrongly"
NO_VOCABULARY = "SPEC: the committed rule has no vocabulary for this question"


# =========================================================================== #
# 1. FIXTURE POWER -- these must pass, or nothing below means anything
# =========================================================================== #

def test_no_pinned_outcome_was_synthesised():
    """All six outcomes come from real records; the synthesis ledger is empty.

    A contract built partly out of invented data would let an implementation
    satisfy an invented case and be quoted as evidence about the corpus.
    """
    assert fx.SYNTHESIS_LEDGER == (), (
        "synthetic cases present: " + ", ".join(fx.SYNTHESIS_LEDGER)
    )
    assert len(fx.ALL_CASES) == 6
    for record in fx.ALL_RECORDS:
        assert record.annotations, f"{record.key} has no curator annotation"
        assert record.raw_rows >= 1, f"{record.key} is backed by no shard row"
        assert record.length == len(record.antigen_sequence)
        assert set(record.antigen_sequence) <= set("ACDEFGHIKLMNPQRSTVWYXBZJUO*"), (
            f"{record.key} carries characters clean_aa_sequence would not emit"
        )


def test_fixtures_expose_no_supervision_field():
    """Contract requirement C, enforced structurally rather than by promise.

    Identity must be computed from sequences and identifiers only. The guarantee
    here is that a test written against these fixtures CANNOT read a label: the
    dataclasses have no field for one, so the mistake is unavailable rather than
    merely discouraged.
    """
    for cls in (fx.AnnotationRow, fx.AntigenRecord, fx.PairEvidence):
        names = {f.name for f in dataclasses.fields(cls)}
        overlap = names & fx.FORBIDDEN_LABEL_FIELDS
        assert not overlap, f"{cls.__name__} exposes supervision fields {overlap}"
    for record in fx.ALL_RECORDS:
        for row in record.rows():
            assert set(row) == {"metadata", "antigen_sequence"}
            assert set(row["metadata"]) == {"target_name", "target_pdb", "target_uniprot"}


def test_recorded_evidence_still_describes_the_recorded_sequences():
    """Every identity/coverage constant is reproduced from the sequences beside it.

    The reference aligner is not the contract -- see the fixture module docstring
    -- it is the device that stops a constant drifting away from its data during
    a later edit.
    """
    checked = 0
    for case in fx.ALL_CASES:
        for evidence in case.evidence:
            left = case.record(evidence.left).antigen_sequence
            right = case.record(evidence.right).antigen_sequence
            identity, cov_left, cov_right = fx.reference_aligned_identity(left, right)
            assert identity == pytest.approx(evidence.identity, abs=5e-7)
            assert cov_left == pytest.approx(evidence.coverage_left, abs=5e-7)
            assert cov_right == pytest.approx(evidence.coverage_right, abs=5e-7)
            assert evidence.exact_substring == (left in right or right in left)
            checked += 1
    assert checked == 15, f"expected 15 measured relations, found {checked}"


def test_fixture_subset_reproduces_the_corpus_verdict(committed):
    """The 16 fixture records give the same committed-rule answer the corpus does.

    Every ``committed_rule`` block in the fixture module records what
    ``canonical_target_id`` the full 1,227,083-row shard set assigns -- read both
    from the stored field in ``data/processed/antibody_antigen_v2`` and from a
    rebuild of the partition with the production classes. This test rebuilds the
    same partition over the 16 fixtures alone and requires the same answer, which
    is what licenses every committed-rule test below to run on the subset.
    """
    for case in fx.ALL_CASES:
        expected_ids = case.committed_rule["canonical_target_id"]
        expected_splits = case.committed_rule["split"]
        for record in case.records:
            assert committed.canonical_id(record.key) == expected_ids[record.key], (
                f"outcome {case.outcome}: {record.key} diverges from the corpus"
            )
            assert committed.split(record.key) == expected_splits[record.key]


def test_outcome_1_fixture_is_a_clean_two_residue_insertion():
    """Outcome 1 fixture power. Real record: the 287/289-aa Omicron spike NTDs.

    Proves the pair is what the outcome claims -- one interior 2-residue
    insertion, every other residue identical -- and that exact-substring
    containment nevertheless fails on it. That failure is the specific mechanism
    that tore this family apart in the reverted attempt, so a fixture that did
    not exhibit it would test nothing.
    """
    case = fx.CASES_BY_OUTCOME[1]
    short = case.record("omicron_ntd_287").antigen_sequence
    long = case.record("omicron_ntd_289").antigen_sequence
    offset = fx.OMICRON_INSERTION_OFFSET
    assert len(long) - len(short) == len(fx.OMICRON_INSERTION) == 2
    assert long[offset:offset + 2] == fx.OMICRON_INSERTION
    assert long[:offset] + long[offset + 2:] == short
    assert short not in long, "an interior insertion must defeat containment"
    assert case.relation("omicron_ntd_287", "omicron_ntd_289").identity > 0.99


def test_outcome_1_every_antibody_on_the_289_side_is_also_on_the_287_side():
    """Outcome 1 fixture power: what a false split of this pair would cost.

    10 of the 12 heavy variable domains seen against the 287-mer are also seen
    against the 289-mer, and that is all 10 of the 289-mer's. These are
    heavy-only records, so the VH is the whole antibody identity. Separating the
    two constructs leaks 100% of the smaller side's antibodies -- the mechanism
    behind the reverted attempt's 335 val rows whose exact VH+VL was in train.
    """
    assert len(fx.OMICRON_SHARED_HEAVY) == 10
    assert len(set(fx.OMICRON_SHARED_HEAVY)) == 10
    for heavy in fx.OMICRON_SHARED_HEAVY:
        assert 100 <= len(heavy) <= 200


def test_outcome_2_fixture_is_a_real_linker_fusion_with_exact_substrings():
    """Outcome 2 fixture power. Real record: the 1057-aa 1N8Z fusion construct.

    Proves the fusion is a composite -- two ``GGGGGGGG`` linker runs at the
    recorded offsets -- and that both IGKC and the HER2 ECD are exact substrings
    of it. Containment therefore genuinely fires here; the outcome is about what
    that firing is allowed to mean.
    """
    case = fx.CASES_BY_OUTCOME[2]
    fusion = case.record("fusion_1n8z").antigen_sequence
    igkc = case.record("igkc").antigen_sequence
    her2 = case.record("her2_ecd").antigen_sequence
    assert fusion.find(igkc) == 107
    assert fusion.find(her2) == 450
    for offset in fx.FUSION_LINKER_OFFSETS:
        assert fusion[offset:offset + 8] == "G" * 8
    assert fusion.count("G" * 8) == len(fx.FUSION_LINKER_OFFSETS) == 2
    igkc_vs_fusion = case.relation("igkc", "fusion_1n8z")
    assert igkc_vs_fusion.exact_substring is True
    assert igkc_vs_fusion.coverage_left == pytest.approx(1.0)
    assert igkc_vs_fusion.coverage_right < 0.11, (
        "reciprocal coverage is what separates a component from its container"
    )


def test_outcome_3_fixture_entities_share_a_pdb_and_nothing_else():
    """Outcome 3 fixture power. Real record: PDB 4GRW's two polymer entities.

    Proves the two sequences really do share the PDB annotation, really do carry
    different accessions elsewhere in the shards, and really are unrelated -- no
    shared 8-mer at all.
    """
    case = fx.CASES_BY_OUTCOME[3]
    il23a, il12b = case.record("il23a"), case.record("il12b")
    pdbs_23 = {a.target_pdb for a in il23a.annotations if a.target_pdb}
    pdbs_12 = {a.target_pdb for a in il12b.annotations if a.target_pdb}
    assert "4grw" in pdbs_23 and "4grw" in pdbs_12
    accessions_23 = {a.target_uniprot for a in il23a.annotations if a.target_uniprot}
    accessions_12 = {a.target_uniprot for a in il12b.annotations if a.target_uniprot}
    assert accessions_23 == {"q9npf7"}
    assert accessions_12 == {"p29460"}
    assert not (accessions_23 & accessions_12)
    assert fx.shares_no_kmer(il23a.antigen_sequence, il12b.antigen_sequence, k=8)
    assert case.relation("il23a", "il12b").identity < 0.30


def test_outcome_4_fixture_chain_is_open_across_a_real_threshold_window():
    """Outcome 4 fixture power. Real records: three influenza B haemagglutinins.

    Proves the A-B-C chain is genuinely open: there is a non-empty window of
    thresholds at which both A~B and B~C are admitted with high reciprocal
    coverage and A~C is rejected. A fixture whose window were empty could not
    distinguish single-linkage from a cluster criterion, which is the only thing
    the outcome is about.
    """
    case = fx.CASES_BY_OUTCOME[4]
    ab = case.relation("hema_inbyb", "hema_inbte")
    bc = case.relation("hema_inbte", "hema_inbvm")
    ac = case.relation("hema_inbyb", "hema_inbvm")
    low, high = case.committed_rule["threshold_window_that_percolates"]
    assert ac.identity == pytest.approx(low, abs=5e-5)
    assert min(ab.identity, bc.identity) == pytest.approx(high, abs=5e-5)
    assert low < high, "the chain must be open at some threshold"
    assert high - low > 0.04, "an open window this narrow would be a knife edge"
    for edge in (ab, bc):
        assert min(edge.coverage_left, edge.coverage_right) >= 0.93, (
            "both admitted edges must clear coverage too, or requirement 1 "
            "already rejects them and the chain proves nothing"
        )
    accessions = {
        a.target_uniprot
        for record in case.records
        for a in record.annotations
        if a.target_uniprot
    }
    assert len(accessions) == 3, "one accession per node, so nothing is confounded"


def test_outcome_5_fixture_name_spans_two_non_homologous_families():
    """Outcome 5 fixture power. Real record: the name-only ``sars-cov2_wt`` rows.

    Proves the name carries no accession and no PDB code on the rows that
    define it, and that the constructs it spans include a pair sharing no 8-mer.
    Contract requirement 4 says ambiguity means biological incoherence, not
    graph fragmentation -- so the fixture has to exhibit incoherence.
    """
    case = fx.CASES_BY_OUTCOME[5]
    name_only = [
        a
        for record in case.records
        for a in record.annotations
        if a.target_name == "sars-cov2_wt"
    ]
    assert name_only, "the fixture must contain the name-only annotation"
    for annotation in name_only:
        assert annotation.target_uniprot == ""
        assert annotation.target_pdb == ""
    assert len({r.key for r in case.records if any(
        a.target_name == "sars-cov2_wt" for a in r.annotations
    )}) == 4, "one name, four constructs"
    rbd = case.record("wt_rbd").antigen_sequence
    ntd = case.record("wt_ntd").antigen_sequence
    s2 = case.record("wt_s2").antigen_sequence
    assert fx.shares_no_kmer(rbd, s2, k=8)
    assert fx.shares_no_kmer(rbd, ntd, k=8)
    assert case.relation("wt_rbd", "wt_s2").identity < 0.25
    assert case.relation("wt_rbd", "wt_ntd").identity < 0.30
    coherent = case.relation("wt_rbd", "wt_s1")
    assert coherent.coverage_left > 0.99, "the RBD really does sit inside S1"
    assert coherent.coverage_right < 0.45, "...covering under half of it"


def test_outcome_6_fixture_is_one_byte_identical_sequence():
    """Outcome 6 fixture power. Real record: RAC1, P63000 and P63001.

    Proves both accessions are written over ONE sequence, byte-identical, so no
    similarity threshold and no alignment choice can be blamed for the merge and
    no tie-break between the accessions has any evidence behind it.
    """
    case = fx.CASES_BY_OUTCOME[6]
    record = case.record("rac1")
    assert len(record.annotations) == 2
    accessions = {a.target_uniprot for a in record.annotations}
    assert accessions == {"p63000", "p63001"}
    assert {a.target_name for a in record.annotations} == {"rac1_human", "rac1_mouse"}
    rows = record.rows()
    assert rows[0]["antigen_sequence"] == rows[1]["antigen_sequence"]
    assert record.length == 192


# =========================================================================== #
# 2. REGRESSION GUARDS -- the committed rule gets these right; keep it that way
# =========================================================================== #

def test_guard_outcome_2_igkc_is_not_her2_today(committed):
    """Outcome 2, protected half. Real record: IGKC (P01834) vs the HER2 ECD.

    The reverted attempt added an exact-substring containment edge and fused
    these two through the 1N8Z fusion construct. The committed rule keeps them
    apart because it has no containment edge at all. This guard is what turns
    that accident into a requirement: whatever relation replaces it, local
    containment must not become identity.
    """
    assert committed.canonical_id("igkc") != committed.canonical_id("her2_ecd")
    assert committed.canonical_id("igkc") == "uniprot:p01834"


def test_guard_outcome_1_omicron_pair_shares_one_split_today(committed):
    """Outcome 1, protected half. Real record: the 287/289-aa Omicron NTDs.

    Today they share a canonical id and therefore a split. The mechanism is
    wrong -- see the specification below -- but the outcome is right, and a
    redesign that separates them has regressed regardless of how principled its
    reason is.
    """
    assert committed.canonical_id("omicron_ntd_287") == committed.canonical_id("omicron_ntd_289")
    assert committed.split("omicron_ntd_287") == committed.split("omicron_ntd_289")


def test_guard_identity_is_blind_to_supervision(paa):
    """Contract requirement C, against the committed rule. All six outcomes.

    Attaches every supervision field the producer knows about to each fixture
    row, with values chosen to disagree violently between the two copies, and
    requires byte-identical identity nodes. The committed rule passes because
    ``extract_target_nodes`` reads only the target fields; the point of pinning
    it is that a future engine which starts consulting a confidence or an
    affinity to break a tie fails here immediately.
    """
    for record in fx.ALL_RECORDS:
        for row in record.rows():
            clean = paa.extract_target_nodes(row["metadata"], row["antigen_sequence"])
            for supervision in (
                {f: 1 for f in fx.FORBIDDEN_LABEL_FIELDS},
                {f: 0 for f in fx.FORBIDDEN_LABEL_FIELDS},
                {f: "very_high" for f in fx.FORBIDDEN_LABEL_FIELDS},
            ):
                metadata = dict(row["metadata"])
                metadata.update(supervision)
                assert paa.extract_target_nodes(metadata, row["antigen_sequence"]) == clean, (
                    f"{record.key}: identity moved when supervision changed"
                )


# =========================================================================== #
# 3. SPECIFICATIONS -- outcomes the committed rule answers, and answers wrongly.
#    Re-pointed at the engine on 2026-09-01; see the module docstring for the
#    proof that they could not stay where they were.
# =========================================================================== #

def test_outcome_1_family_survives_deleting_the_name():
    """Pinned outcome 1. Real records: the 287-aa and 289-aa Omicron spike NTDs.

    The two constructs are 99.31% identical with full coverage of the shorter
    side. That, and not a curator's label, is why they are one family. This test
    deletes ``target_name`` from every row and requires the family to survive.

    The committed rule fails it, and the failure is structural rather than
    incidental: it has no sequence-similarity relation, so stripping the name
    leaves two different ``seq:`` nodes that never touch. The same name that
    saves this pair is what fuses four non-homologous constructs in outcome 5, so
    "keep the name bridge" was never an available fix -- only adding a similarity
    relation is.
    """
    case = fx.CASES_BY_OUTCOME[1]
    nameless = []
    for record in case.records:
        for row in record.rows():
            metadata = dict(row["metadata"])
            metadata["target_name"] = ""
            nameless.append({
                "metadata": metadata,
                "antigen_sequence": row["antigen_sequence"],
            })
    module = load_target_rule()
    resolution = module.resolve_target_identity(nameless)
    short, long = case.record("omicron_ntd_287"), case.record("omicron_ntd_289")
    assert resolution.construct_id(short) == resolution.construct_id(long)
    assert resolution.biological_target_id(short) == resolution.biological_target_id(long)
    assert resolution.split_group_id(short) == resolution.split_group_id(long)


def test_outcome_3_4grw_entities_remain_distinct_targets():
    """Pinned outcome 3. Real records: IL23A (Q9NPF7) and IL12B (P29460), PDB 4GRW.

    A PDB entry contains multiple polymer entities, so sharing one carries no
    identity implication. These two share 4GRW, share 24.78% of their residues,
    and share no 8-mer.

    Under the committed rule ``pdb:4grw`` is a full-strength merging node: the
    two entities become one component named ``uniprot:p29460``, the IL12B
    accession, silently applied to IL23A rows -- and both land in val, so any
    per-target number computed over that slice mixes two genes. The engine
    demotes PDB co-entry to quarantine, so the two stay distinct targets AND
    still share a split group, which is the combination the committed rule could
    not express.
    """
    case = fx.CASES_BY_OUTCOME[3]
    resolution = resolve(case.records)
    il23a, il12b = case.record("il23a"), case.record("il12b")
    assert resolution.biological_target_id(il23a) != resolution.biological_target_id(il12b)
    assert resolution.construct_id(il23a) != resolution.construct_id(il12b)
    assert resolution.split_group_id(il23a) == resolution.split_group_id(il12b), (
        "demoting the container must not also discard the leakage constraint"
    )


def test_outcome_5_name_only_label_does_not_fuse_incompatible_constructs():
    """Pinned outcome 5. Real records: the four ``sars-cov2_wt`` constructs.

    ``sars-cov2_wt`` is written on rows carrying no accession and no PDB code.
    It spans a 223-aa RBD, a 292-aa NTD, a 539-aa S1 fragment and a 666-aa S2
    region. The RBD and the S2 share no 8-mer. A name that spans two
    non-homologous families names neither, so it must quarantine.

    Under the committed rule a normalized name is a full-strength merging node.
    All four constructs share one canonical id, ``pdb:7ch5`` -- and that
    component has swallowed 12 antigen sequences, 7 variant names and 11 PDB
    codes, with a minimum pairwise identity of 20.41%.
    """
    case = fx.CASES_BY_OUTCOME[5]
    resolution = resolve(case.records)
    rbd, s2 = case.record("wt_rbd"), case.record("wt_s2")
    assert resolution.biological_target_id(rbd) != resolution.biological_target_id(s2)
    assert resolution.split_group_id(rbd) == resolution.split_group_id(s2), (
        "an ambiguous name is still evidence of co-occurrence, so refusing the "
        "merge must not also drop the constraint"
    )


def test_outcome_6_conflicting_accessions_are_not_silently_resolved():
    """Pinned outcome 6. Real record: 192 aa, P63000 and P63001, one sequence.

    Two accessions written over one byte-identical sequence is a disagreement in
    the source data, and the resolver's job is to report it, not to settle it.

    The committed rule fails it twice over. ``canonical_target_id_from_nodes``
    takes ``min`` over ``(namespace rank, node)``, so ``uniprot:p63000`` wins on
    lexicographic order alone and P63001 vanishes; and
    ``TargetIdentityIndex.stats()`` has no conflict counter, so the disagreement
    is reported nowhere. This test pins both halves: the id must not be one of
    the disputed accessions, and the stats must carry a conflict count.
    """
    case = fx.CASES_BY_OUTCOME[6]
    resolution = resolve(case.records)
    stats = resolution.stats()
    assert "target_accession_conflicts" in stats, (
        "a conflict that is not counted is a conflict nobody will find"
    )
    assert stats["target_accession_conflicts"] == 1
    rac1 = case.record("rac1")
    assert resolution.biological_target_id(rac1) not in (
        "uniprot:p63000", "uniprot:p63001"
    )
    conflict = resolution.accession_conflicts()[0]
    assert conflict.accessions == ("p63000", "p63001"), (
        "both accessions must survive into the report; dropping the loser is "
        "the failure being tested"
    )


def test_component_reports_its_own_minimum_identity_and_coverage():
    """Contract requirement 5, on real records: the ``pdb:7ch5`` component.

    Thresholded similarity plus union-find is single-linkage clustering, and
    single-linkage components are invisible from the outside: nothing in the
    partition says whether a component is tight or has percolated. Requirement 5
    asks for a representative or maximum-diameter constraint AND for each
    component to report its minimum pairwise identity and coverage.

    The committed rule's ``stats()`` reports six counts, none of them per
    component. Concretely, the component holding all four ``sars-cov2_wt``
    constructs has a minimum pairwise identity of 20.41% and nothing anywhere
    says so.
    """
    resolution = resolve(fx.ALL_RECORDS)
    stats = resolution.stats()
    assert "component_min_pairwise_identity" in stats
    assert "component_min_pairwise_coverage" in stats

    # Not merely present. Two numbers are reported and they mean different
    # things, because two different kinds of evidence build a family.
    #
    # Where the bounded criterion decided, its guarantee holds by construction:
    # complete-linkage admits a merge only when every cross pair clears the
    # operating point, so no member pair can sit below it.
    point = resolution.operating_point
    assert stats["criterion_min_pairwise_identity"] >= point.family_identity
    assert stats["criterion_min_pairwise_coverage"] >= point.family_coverage
    assert stats["criterion_max_diameter"] <= 1.0 - point.family_identity

    # Where a curator's accession or an approved name decided, there is no such
    # guarantee and the report must not pretend otherwise -- a curator asserting
    # that two constructs are one target outranks what their residues say. What
    # the all-components number promises is only that it is REPORTED, which is
    # the whole of requirement 5: the committed rule's ``pdb:7ch5`` component
    # sits at 0.2041 minimum pairwise identity with nothing anywhere saying so.
    assert 0.0 <= stats["component_min_pairwise_identity"] <= 1.0
    assert "component_curator_merged" in stats
    assert stats["component_min_pairwise_identity"] > 0.2041, (
        "the engine must not reproduce the committed rule's percolated component"
    )


# =========================================================================== #
# 4. SPECIFICATIONS -- questions the committed rule has no vocabulary for
# =========================================================================== #

def test_outcome_2_fusion_overlap_is_a_quarantine_edge_not_an_identity():
    """Pinned outcome 2. Real records: the 1N8Z fusion, the HER2 ECD, IGKC.

    The 1057-aa fusion contains the 107-aa IGKC domain at offset 107 and the
    607-aa HER2 ECD at offset 450. Both overlaps are real and both must be
    recorded -- the fusion must not sit in generic validation while either
    component sits in train. Neither overlap may become identity: IGKC covers
    10.12% of the fusion, and P01834 is not HER2.

    This is the one outcome that needs both halves of the typed-relation design
    at once, and the committed rule has neither: overlap is either identity or
    nothing, so there is no way to express "quarantine these two without merging
    them".
    """
    case = fx.CASES_BY_OUTCOME[2]
    resolution = resolve(case.records)
    partners = requires_method(resolution, "quarantine_partners")
    construct = requires_method(resolution, "construct_id")
    biological = requires_method(resolution, "biological_target_id")

    igkc, her2, fusion = (case.record(k) for k in ("igkc", "her2_ecd", "fusion_1n8z"))
    assert construct(igkc) != construct(her2)
    assert biological(igkc) != biological(her2)
    assert construct(fusion) in partners(igkc)
    assert construct(fusion) in partners(her2)
    assert construct(her2) not in partners(igkc), (
        "quarantine is between an overlap and its container, not transitive "
        "between two things that happen to share a container"
    )


def test_the_anti_percolation_fixture_gets_the_same_integrity_guards():
    """The new fixture inherits the old one's guards rather than the old one's word.

    `test_no_pinned_outcome_was_synthesised` reads only ``fx.SYNTHESIS_LEDGER``,
    so when the anti-percolation chain moved into its own module it silently lost
    every integrity check the original fixtures have -- while its docstring
    claimed "the acceptance suite checks it". This is that check.
    """
    assert ap.SYNTHESIS_LEDGER == (), (
        "synthetic cases present: " + ", ".join(ap.SYNTHESIS_LEDGER)
    )
    for record in ap.CHAIN_RECORDS:
        assert record.annotations, f"{record.key} has no curator annotation"
        assert record.raw_rows >= 1, f"{record.key} is backed by no shard row"
        assert record.length == len(record.antigen_sequence)
        assert set(record.antigen_sequence) <= set("ACDEFGHIKLMNPQRSTVWYXBZJUO*"), (
            f"{record.key} carries characters clean_aa_sequence would not emit"
        )
        # The digest is the fixture's link back to the corpus. If it drifted from
        # the sequence beside it, every downstream claim would be about a record
        # nobody can find.
        module = load_target_rule()
        assert module.antigen_digest(record.antigen_sequence) == record.antigen_sha256_32
        for name, pdb, uniprot in record.annotations:
            assert isinstance(name, str) and isinstance(pdb, str)
            assert isinstance(uniprot, str)
        for row in record.rows():
            assert set(row) == {"metadata", "antigen_sequence"}
            assert set(row["metadata"]) == {
                "target_name", "target_pdb", "target_uniprot"
            }


def test_the_anti_percolation_chain_is_a_tag_difference_and_says_so():
    """The chain's arithmetic, checked rather than described.

    Strip the trailing histidines and A is a 191-residue exact substring of C
    while B is a 199-residue one. That is where the coverages come from, and it
    is what makes this a construct-boundary chain rather than a divergence chain.
    A fixture whose stated mechanism does not match its bytes is a fixture nobody
    can reason from.
    """
    a, b, c = (ap.record(k).antigen_sequence for k in
               ("rbd_199", "rbd_205", "rbd_209"))
    assert len(a) - len(a.rstrip("H")) == 8, "the 199-mer carries an 8x His tag"
    assert len(b) - len(b.rstrip("H")) == 6, "the 205-mer carries a 6x His tag"
    assert not c.endswith("H"), "the 209-mer is untagged"

    assert a.rstrip("H") in c and len(a.rstrip("H")) == 191
    assert b.rstrip("H") in c and len(b.rstrip("H")) == 199
    assert ap.edge("rbd_199", "rbd_209").cov_right == pytest.approx(191 / 209, abs=5e-7)
    assert ap.edge("rbd_205", "rbd_209").cov_right == pytest.approx(199 / 209, abs=5e-7)


def test_outcome_4_fixture_has_power_at_the_shipped_operating_point():
    """Outcome 4 fixture power, in the metric the thresholds are expressed in.

    The original outcome-4 fixture -- three influenza B haemagglutinins -- was
    shipped with a warning that it was POWERLESS, found independently by two
    adversarial reviewers. That was re-measured in the engine's own local affine
    Smith-Waterman metric before this repair was written, and confirmed: at the
    construct operating point all three form no edge at all, complete-linkage and
    single-linkage agree, and ``merges_refused`` is zero. The assertion passed
    with the anti-percolation criterion deleted.

    This test is the repair's premise. It proves, positively, that the re-sourced
    chain in ``fixtures_anti_percolation`` really does close a triangle AT the
    shipped thresholds: both chain edges are admitted, the closing edge is
    refused, and the separating quantity is coverage rather than identity --
    every identity in the chain is exactly 1.0, so no identity threshold could
    ever separate these three.
    """
    module = load_target_rule()
    point = module.DEFAULT_OPERATING_POINT
    low, high = ap.COVERAGE_WINDOW
    assert low < point.construct_coverage <= high, (
        "the shipped coverage threshold must sit inside the open window, or the "
        "chain is not open where it is actually used"
    )
    assert high - low > 0.02, "an open window this narrow would be a knife edge"

    for edge in ap.CHAIN_EDGES:
        measured = module.align_pair(
            ap.record(edge.left).antigen_sequence,
            ap.record(edge.right).antigen_sequence,
        )
        assert measured.identity == pytest.approx(edge.identity, abs=5e-7)
        assert measured.cov_left == pytest.approx(edge.cov_left, abs=5e-7)
        assert measured.cov_right == pytest.approx(edge.cov_right, abs=5e-7)
        assert measured.overlap == edge.overlap
        admitted = (
            measured.identity >= point.construct_identity
            and measured.min_coverage >= point.construct_coverage
            and measured.overlap >= point.construct_overlap
        )
        assert admitted == edge.admitted_at_construct_point, (
            f"{edge.left}~{edge.right} is recorded as "
            f"admitted={edge.admitted_at_construct_point} and measures {admitted}"
        )
        assert measured.identity == 1.0, (
            "the chain must be closed in identity, or coverage is not what the "
            "criterion is being tested on"
        )
        # Exact containment fails on every pair despite identity 1.0, because the
        # purification tags differ. Same mechanism as the Omicron indel, reached
        # from the other end.
        left, right = (
            ap.record(edge.left).antigen_sequence,
            ap.record(edge.right).antigen_sequence,
        )
        assert left not in right and right not in left


def test_outcome_4_cluster_criterion_refuses_the_closing_edge():
    """Pinned outcome 4. Real records: three SARS-CoV-2 spike RBD constructs.

    A 199-mer, a 205-mer and a 209-mer, nested sub-ranges of one receptor-binding
    domain carrying three different purification tags. A~B and B~C are admitted
    at the construct operating point; A~C is refused, on coverage, at identity
    1.0. Single-linkage over those two admitted edges welds all three into one
    construct; a bounded criterion does not.

    ``test_outcome_4_fixture_has_power_at_the_shipped_operating_point`` proves
    the premise, and `test_target_identity_engine.py` proves the converse by
    fault injection: substituting single-linkage for the bounded criterion makes
    this test fail. Without both, an implementation could satisfy this by never
    forming an edge at all -- which is exactly how the original fixture came to
    be powerless.
    """
    module = load_target_rule()
    resolution = module.resolve_target_identity(
        [row for record in ap.CHAIN_RECORDS for row in record.rows()]
    )
    construct = requires_method(resolution, "construct_id")
    report = requires_method(resolution, "construct_cluster_report")
    a, b, c = (ap.record(k) for k in ("rbd_199", "rbd_205", "rbd_209"))

    # Positive power: the two chain edges must actually be present, or the
    # refusal below is about nothing.
    assert construct(a) == construct(b), (
        "the A~B edge must be admitted at the operating point, or the closing "
        "edge is never offered to the criterion"
    )
    assert construct(a) != construct(c), (
        "A and C cover only 91.39% of each other; a chain through B may not "
        "merge them"
    )
    assert resolution.stats()["target_construct_merges_refused"] >= 1, (
        "the criterion must be recorded as having actually refused a merge"
    )

    # And all three remain ONE biological target, so refusing the construct
    # merge costs no leakage: they still share a split group.
    assert (
        resolution.biological_target_id(a)
        == resolution.biological_target_id(b)
        == resolution.biological_target_id(c)
    )
    assert resolution.split_group_id(a) == resolution.split_group_id(c)

    for record in (a, b, c):
        cluster = report(construct(record))
        assert cluster.min_pairwise_identity >= 0.90
        assert cluster.min_pairwise_coverage >= 0.90
        assert cluster.representative is not None
        assert cluster.max_diameter <= 1.0 - module.DEFAULT_OPERATING_POINT.construct_identity


def test_the_three_relations_stay_separate():
    """Contract requirement B. Real records: the 1N8Z fusion set and 4GRW.

    construct identity, biological family and leakage quarantine are three
    relations, and collapsing them into one graph is the failure this contract
    was rewritten to prevent. The witness is concrete: local containment must be
    able to quarantine IGKC against the HER2 fusion WITHOUT ever making them one
    target, and sharing a UniProt accession must place two constructs in one
    family WITHOUT making them one construct.

    A design that derives all three ids from one union-find fails this
    requirement whatever the ids happen to be, so this test also requires the
    three to be observably different functions.
    """
    records = fx.CASES_BY_OUTCOME[2].records + fx.CASES_BY_OUTCOME[3].records
    resolution = resolve(records)
    construct = requires_method(resolution, "construct_id")
    biological = requires_method(resolution, "biological_target_id")
    partners = requires_method(resolution, "quarantine_partners")

    case2 = fx.CASES_BY_OUTCOME[2]
    igkc = case2.record("igkc")
    fusion = case2.record("fusion_1n8z")
    her2_607 = case2.record("her2_ecd")
    her2_564 = case2.record("her2_3wsq")

    # Quarantine without identity.
    assert construct(fusion) in partners(igkc)
    assert biological(igkc) != biological(fusion)

    # Family without construct identity: two HER2 ECD truncations, 92.92%
    # identical, one an exact substring of the other.
    assert biological(her2_607) == biological(her2_564)
    assert construct(her2_607) != construct(her2_564)


def test_generic_and_unseen_mutant_claims_get_separate_split_contracts():
    """Claim dependence. Real records: the 287/289-aa Omicron NTD pair.

    For a generic held-out-target split, the same antibody repeated across two
    near-identical antigens is contamination and the pair must stay together.
    For a predeclared unseen-mutant benchmark it is the intended counterfactual
    and the pair must be separable. Those rows must not quietly coexist in one
    validation metric, so one split cannot serve both claims.

    The committed rule has exactly one split, derived from one id, so the
    question cannot be asked of it.
    """
    case = fx.CASES_BY_OUTCOME[1]
    resolution = resolve(case.records)
    split_group = requires_method(resolution, "split_group_id")
    short, long = case.record("omicron_ntd_287"), case.record("omicron_ntd_289")

    assert split_group(short, claim="generic") == split_group(long, claim="generic")
    assert split_group(short, claim="unseen_mutant") != split_group(long, claim="unseen_mutant")


def test_threshold_selection_is_audited_on_families_it_never_saw():
    """Contract requirement A. Real families: all five in the fixture set.

    An operating point chosen on a set and then validated on the same set
    reports its own selection back as an achieved error rate. The split must be
    by FAMILY, not by pair, so that Omicron or the fusion cases cannot dictate a
    threshold and then confirm it, and the two reports must be quotable
    separately.

    Fails today because there is no threshold to calibrate.
    """
    resolution = resolve(fx.ALL_RECORDS)
    calibration = requires_method(resolution, "calibration_report")()
    audit = requires_method(resolution, "audit_report")()
    assert set(calibration.families) & set(audit.families) == set(), (
        "leave-one-family-out means no family appears in both reports"
    )
    assert set(fx.LEAVE_ONE_FAMILY_OUT_AUDIT_FAMILIES) <= set(audit.families)
    for report in (calibration, audit):
        # `is not None` on a non-Optional int is a tautology: `ErrorReport`
        # declares both as `int` on a frozen dataclass, so the original form of
        # this assertion passed with 99,999 errors in either column. What the
        # requirement is about is that BOTH numbers are produced over a
        # population that could have produced either, and that they are actually
        # zero here.
        assert report.pairs > 0, f"{report.name} scored no pairs"
        assert report.positive_pairs > 0, (
            f"{report.name} has no positive pairs, so its false-split count is free"
        )
        assert report.negative_pairs > 0, (
            f"{report.name} has no negative pairs, so its false-merge count is free"
        )
        assert report.false_merges == 0, report.false_merge_examples
        assert report.false_splits == 0, report.false_split_examples


def test_both_error_kinds_are_reported_against_the_predeclared_asymmetry():
    """Contract requirement D. All six outcomes.

    False merges and false splits cost different things -- a false merge
    destroys target diversity and corrupts per-target reporting while remaining
    invisible to a leakage audit; a false split costs leakage, which a leakage
    audit does detect. The asymmetry is declared in the fixture module BEFORE any
    measurement, and the audit report must carry it so nobody can choose it after
    seeing the numbers.
    """
    resolution = resolve(fx.ALL_RECORDS)
    audit = requires_method(resolution, "audit_report")()
    assert audit.tolerated_error == fx.PREDECLARED_ERROR_ASYMMETRY
    # Both kinds, counted separately, over a population that could have shown
    # either. The two counts are `int`, so asserting they are not None asserts
    # nothing; what the requirement needs is that the population has pairs of
    # both signs and that the resolver got them right.
    assert audit.positive_pairs > 0 and audit.negative_pairs > 0
    assert audit.false_merges == 0, audit.false_merge_examples
    assert audit.false_splits == 0, audit.false_split_examples
    # The asymmetry is a direction, and the direction has to be legible: a false
    # merge is the expensive error, so a design tolerating MORE merges than
    # splits would be the wrong trade even at these counts.
    assert fx.PREDECLARED_ERROR_ASYMMETRY == "tolerate_false_splits_over_false_merges"


def test_name_approval_is_decided_once_and_recorded():
    """Contract requirement 2, with outcome 5's record: ``sars-cov2_wt``.

    Coherent names genuinely merge -- appending a name to a frozen component is
    semantically inert, which is how the reverted attempt changed the partition
    by exactly zero components. But approval is computed ONCE against the
    sequence/accession clusters and never iterated to a fixed point, because
    iterative name bridging recreates transitive chaining.

    This test requires the decision to be a recorded, inspectable object: which
    name, approved or not, what it attached to, and why. ``sars-cov2_wt`` must
    come back unapproved, because it spans two families that share no 8-mer.
    """
    case = fx.CASES_BY_OUTCOME[5]
    resolution = resolve(case.records)
    decisions = {d.name: d for d in requires_method(resolution, "name_decisions")()}
    assert "sars_cov2_wt" in decisions, "every name that was considered must be recorded"
    decision = decisions["sars_cov2_wt"]
    assert decision.approved is False
    assert decision.attached_to is None
    assert decision.reason
