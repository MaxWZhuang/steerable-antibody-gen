"""
Canonical target identity (J02): aliases of one biological target must never
straddle the train/val boundary.

``build_target_key`` picks the FIRST available identifier from four mutually
exclusive branches (uniprot -> pdb -> name -> antigen hash) with no cross-branch
reconciliation, so one antigen seen once with a UniProt accession and once with
only a PDB code produced two different keys and two independent split draws.
``canonical_target_id`` replaces it for splitting: identifiers observed together
on one source record are unioned into a connected component, and every row in a
component gets one id and therefore one split.

These tests are fixture-only by construction. There is no ``data/raw/`` and no
ASD parquet in this checkout, so nothing here exercises the real corpus. The
end-to-end CLI tests additionally need ``pyarrow``, which is absent from the
local environment; the identity engine itself is therefore also tested directly
against the imported module so the contract is provable without parquet.
"""

from __future__ import annotations

import gzip
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_script(project_root: Path, name: str):
    """Import a scripts/*.py module by name (scripts/ added to sys.path)."""
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paa(project_root: Path):
    return _load_script(project_root, "prepare_antibody_antigen")


@pytest.fixture
def prepare_args(paa, monkeypatch):
    """Default CLI args, exactly as the script would parse them with no flags."""
    monkeypatch.setattr(sys, "argv", ["prepare_antibody_antigen"])
    return paa.parse_args()


# Distinct, in-range heavy variable domains. The dedupe key is the exact
# (heavy, light, antigen) triple, so rows that must both survive differ here.
HEAVY_A = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRF"
    "TISRDNSKNTLYLQMNSLRAEDTAVYYCAKNDILVGYSAFDYWGQGTLVTVSS"
)
HEAVY_B = (
    "QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRF"
    "TISRDNARNTVYLQMNSLKPEDTAVYYCAKNDILVGYSAFDYWGQGTQVTVSS"
)
HEAVY_C = (
    "EVQLLESGGGLVQPGGSLRLSCAASGFTFSNYAMSWVRQAPGKGLEWVSTISGSGGYTYYADSVKGRF"
    "TISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSGWYFDYWGQGTLVTVSS"
)
HEAVY_D = (
    "EVQLVESGGGLVKPGGSLRLSCAASGFTFSDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRF"
    "TISRDNAKNSLYLQMNSLRAEDTALYYCAKDRGYSSSWYFDYWGQGTLVTVSS"
)

LIGHT = (
    "DIQMTQSPSSLSASVGDRVTITCQASQDINNYLNWYQQKPGKAPKLLIYYTSRLHSGVPSRFSGSGSGTDFT"
    "LTISSLQPEDFATYYCQQYNSYPWTFGQGTKVEIK"
)

ANTIGEN_1 = "MKTIIALSYIFCLVFADYKDDDDK"
ANTIGEN_2 = "ACDEFGHIKLMNPQRSTVWYACDE"
ANTIGEN_3 = "MNNNKQQQQQQQQQQQQQQQWWWW"
ANTIGEN_4 = "MGGSHHHHHHSSGLVPRGSHMASL"
ANTIGEN_5 = "MTEYKLVVVGAGGVGKSALTIQLI"


def make_row(
    *,
    heavy_sequence: str,
    antigen_sequence: str,
    target_uniprot: str = "",
    target_pdb: str = "",
    target_name: str = "",
    light_sequence: str = LIGHT,
    affinity_type: str = "bool",
    affinity: str = "1.0",
    processed_measurement: str = "1.0",
    dataset: str = "asd-test",
) -> dict:
    """One source row with explicitly controlled target identifiers."""
    return {
        "dataset": dataset,
        "heavy_sequence": heavy_sequence,
        "light_sequence": light_sequence,
        "scfv": False,
        "affinity_type": affinity_type,
        "affinity": affinity,
        "antigen_sequence": antigen_sequence,
        "confidence": "very_high",
        "nanobody": False,
        "processed_measurement": processed_measurement,
        "metadata": {
            "target_name": target_name,
            "target_pdb": target_pdb,
            "target_uniprot": target_uniprot,
            "source_url": "https://example.org/asd",
            "heavy_riot_numbering": {
                "sequence_alignment_aa": heavy_sequence,
                "cdr1_aa": "GFTFSSYA",
                "cdr2_aa": "ISGSGGST",
                "cdr3_aa": "AKNDILVGYSAFDY",
            },
            "light_riot_numbering": {
                "sequence_alignment_aa": light_sequence,
                "cdr1_aa": "QDINNYLN",
                "cdr2_aa": "TSRLHSGV",
                "cdr3_aa": "QQYNSYPWT",
            },
        },
    }


def build_all(paa, args, rows: list[dict]) -> list[dict]:
    """Run the real two-pass flow in-process: index every row, then build records."""
    index = paa.TargetIdentityIndex()
    for row in rows:
        index.observe_row(row)
    index.finalize()

    records = []
    for idx, row in enumerate(rows):
        record, reason = paa.build_processed_record(
            row=row,
            shard_name="part-00000.parquet",
            row_idx=idx,
            args=args,
            identity_index=index,
        )
        assert record is not None, f"row {idx} unexpectedly dropped: {reason}"
        records.append(record)
    return records


def by_antigen(records) -> dict[str, dict]:
    return {record["sequence_antigen"]: record for record in records}


# --------------------------------------------------------------------------- #
# Node extraction
# --------------------------------------------------------------------------- #
def test_extract_target_nodes_namespaces_every_available_identifier(paa):
    nodes = paa.extract_target_nodes(
        {"target_uniprot": "P12345-2", "target_pdb": "6XYZ_A", "target_name": "Spike Protein"},
        "ACDEFGHIK",
    )

    assert "uniprot:p12345" in nodes
    assert "pdb:6xyz" in nodes
    assert "name:spike_protein" in nodes
    assert sum(node.startswith("seq:") for node in nodes) == 1


def test_extract_target_nodes_omits_missing_identifiers(paa):
    nodes = paa.extract_target_nodes({"target_pdb": "1abc"}, "ACDEFGHIK")

    assert [node for node in nodes if not node.startswith("seq:")] == ["pdb:1abc"]


def test_extract_target_nodes_ignores_labels_entirely(paa):
    """Component construction must never see supervision."""
    metadata = {"target_uniprot": "P12345"}
    with_labels = dict(
        metadata,
        binder_label=1,
        is_strong_binder=True,
        affinity_type="kd",
        processed_measurement=1e-12,
    )

    assert paa.extract_target_nodes(metadata, "ACDEFGHIK") == paa.extract_target_nodes(
        with_labels, "ACDEFGHIK"
    )


# --------------------------------------------------------------------------- #
# Component construction
# --------------------------------------------------------------------------- #
def test_two_aliases_on_one_row_merge(paa, prepare_args):
    """A row carrying both a UniProt accession and a PDB code links the two."""
    records = build_all(
        paa,
        prepare_args,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70001",
                target_pdb="1AAA",
            ),
            # Same biological target, but this source only knew the PDB code.
            make_row(
                heavy_sequence=HEAVY_B,
                antigen_sequence=ANTIGEN_2,
                target_pdb="1AAA",
            ),
        ],
    )

    indexed = by_antigen(records)
    # Pre-fix these were "uniprot:p70001" and "pdb:1aaa" -- two independent draws.
    assert indexed[ANTIGEN_1]["target_key"] == "uniprot:p70001"
    assert indexed[ANTIGEN_2]["target_key"] == "pdb:1aaa"
    assert indexed[ANTIGEN_1]["canonical_target_id"] == "uniprot:p70001"
    assert indexed[ANTIGEN_2]["canonical_target_id"] == "uniprot:p70001"
    assert indexed[ANTIGEN_1]["split"] == indexed[ANTIGEN_2]["split"]


def test_transitive_aliases_merge_into_one_component(paa, prepare_args):
    """uniprot~pdb on one row and pdb~name on another must reach name~uniprot."""
    records = build_all(
        paa,
        prepare_args,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70002",
                target_pdb="1BBB",
            ),
            make_row(
                heavy_sequence=HEAVY_B,
                antigen_sequence=ANTIGEN_2,
                target_pdb="1BBB",
                target_name="Beta Antigen",
            ),
            make_row(
                heavy_sequence=HEAVY_C,
                antigen_sequence=ANTIGEN_3,
                target_name="beta antigen",
            ),
        ],
    )

    assert {record["canonical_target_id"] for record in records} == {"uniprot:p70002"}
    assert len({record["split"] for record in records}) == 1


def test_identical_antigen_sequences_merge(paa, prepare_args):
    """Two accessions annotating byte-identical antigen sequences are one target."""
    records = build_all(
        paa,
        prepare_args,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70004",
            ),
            make_row(
                heavy_sequence=HEAVY_B,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70003",
            ),
        ],
    )

    # Lowest-ranked node of the merged component, not whichever row came first.
    assert {record["canonical_target_id"] for record in records} == {"uniprot:p70003"}
    assert len({record["split"] for record in records}) == 1


def test_unrelated_identifiers_stay_separate(paa, prepare_args):
    """No fuzzy matching: distinct identifiers over distinct sequences stay distinct."""
    records = build_all(
        paa,
        prepare_args,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70005",
                target_pdb="1CCC",
                target_name="gamma antigen",
            ),
            make_row(
                heavy_sequence=HEAVY_B,
                antigen_sequence=ANTIGEN_2,
                target_uniprot="Q70006",
                target_pdb="2DDD",
                target_name="delta antigen",
            ),
        ],
    )

    indexed = by_antigen(records)
    assert indexed[ANTIGEN_1]["canonical_target_id"] == "uniprot:p70005"
    assert indexed[ANTIGEN_2]["canonical_target_id"] == "uniprot:q70006"


def test_similar_names_are_not_fuzzily_merged(paa, prepare_args):
    """"spike protein" and "spike protein s1" are separate nodes, not one target."""
    records = build_all(
        paa,
        prepare_args,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_name="spike protein",
            ),
            make_row(
                heavy_sequence=HEAVY_B,
                antigen_sequence=ANTIGEN_2,
                target_name="spike protein s1",
            ),
        ],
    )

    assert {record["canonical_target_id"] for record in records} == {
        "name:spike_protein",
        "name:spike_protein_s1",
    }


def test_component_id_survives_a_row_with_no_identifiers_at_all(paa, prepare_args):
    """A row with only an antigen sequence still gets a stable sequence-derived id."""
    records = build_all(
        paa,
        prepare_args,
        [make_row(heavy_sequence=HEAVY_A, antigen_sequence=ANTIGEN_1)],
    )

    canonical = records[0]["canonical_target_id"]
    assert canonical.startswith("seq:")
    assert records[0]["target_key"].startswith("antigen_sha1:")


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def _mixed_corpus() -> list[dict]:
    return [
        make_row(
            heavy_sequence=HEAVY_A,
            antigen_sequence=ANTIGEN_1,
            target_uniprot="P70007",
            target_pdb="1EEE",
        ),
        make_row(heavy_sequence=HEAVY_B, antigen_sequence=ANTIGEN_2, target_pdb="1EEE"),
        make_row(
            heavy_sequence=HEAVY_C,
            antigen_sequence=ANTIGEN_3,
            target_name="Epsilon Antigen",
            target_pdb="1EEE",
        ),
        make_row(
            heavy_sequence=HEAVY_D,
            antigen_sequence=ANTIGEN_4,
            target_uniprot="Q70008",
            target_name="zeta antigen",
        ),
        make_row(
            heavy_sequence=HEAVY_A,
            antigen_sequence=ANTIGEN_5,
            target_name="zeta antigen",
        ),
    ]


def _identity_map(records) -> dict[tuple[str, str, str], tuple[str, str]]:
    return {
        (
            record["sequence_heavy"],
            record["sequence_light"] or "",
            record["sequence_antigen"],
        ): (record["canonical_target_id"], record["split"])
        for record in records
    }


def test_input_order_does_not_change_canonical_ids(paa, prepare_args):
    """
    Ids come from sorted component nodes, never from encounter order.

    Every permutation of a 5-row corpus spanning alias, transitive-alias and
    sequence-linked merges must produce the identical id and split per row.
    """
    corpus = _mixed_corpus()
    baseline = _identity_map(build_all(paa, prepare_args, corpus))

    for permutation in itertools.permutations(range(len(corpus))):
        shuffled = [corpus[i] for i in permutation]
        assert _identity_map(build_all(paa, prepare_args, shuffled)) == baseline, (
            f"permutation {permutation} changed canonical identity"
        )


def test_every_row_in_a_component_gets_one_split(paa, prepare_args):
    """The whole point: one component never straddles train/val."""
    records = build_all(paa, prepare_args, _mixed_corpus())

    splits_by_component: dict[str, set[str]] = {}
    for record in records:
        splits_by_component.setdefault(record["canonical_target_id"], set()).add(record["split"])

    assert splits_by_component
    for component, splits in splits_by_component.items():
        assert len(splits) == 1, f"component {component} straddles {sorted(splits)}"


def test_labels_never_participate_in_component_construction(paa, prepare_args):
    """
    Rows that share a target merge despite disagreeing labels, rows that share a
    label do not merge, and flipping every label leaves identity untouched.
    """
    def corpus(first_measurement: str, second_measurement: str) -> list[dict]:
        return [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70009",
                target_pdb="1FFF",
                affinity=first_measurement,
                processed_measurement=first_measurement,
            ),
            make_row(
                heavy_sequence=HEAVY_B,
                antigen_sequence=ANTIGEN_2,
                target_pdb="1FFF",
                affinity=second_measurement,
                processed_measurement=second_measurement,
            ),
            # Same label as row two, entirely unrelated target.
            make_row(
                heavy_sequence=HEAVY_C,
                antigen_sequence=ANTIGEN_3,
                target_uniprot="Q70010",
                affinity=second_measurement,
                processed_measurement=second_measurement,
            ),
        ]

    positive = by_antigen(build_all(paa, prepare_args, corpus("1.0", "0.0")))
    flipped = by_antigen(build_all(paa, prepare_args, corpus("0.0", "1.0")))

    # The labels really did change, so the comparison below is not vacuous.
    assert positive[ANTIGEN_1]["binder_label"] != flipped[ANTIGEN_1]["binder_label"]

    # Disagreeing labels did not stop the alias merge.
    assert (
        positive[ANTIGEN_1]["canonical_target_id"] == positive[ANTIGEN_2]["canonical_target_id"]
    )
    # An agreeing label did not create one.
    assert (
        positive[ANTIGEN_2]["canonical_target_id"] != positive[ANTIGEN_3]["canonical_target_id"]
    )
    # And flipping every label changed nothing about identity or split.
    for antigen in (ANTIGEN_1, ANTIGEN_2, ANTIGEN_3):
        assert (
            positive[antigen]["canonical_target_id"] == flipped[antigen]["canonical_target_id"]
        )
        assert positive[antigen]["split"] == flipped[antigen]["split"]


# --------------------------------------------------------------------------- #
# Audit surface and counts
# --------------------------------------------------------------------------- #
def test_legacy_target_key_is_retained_but_not_used_for_splitting(paa, prepare_args):
    records = build_all(
        paa,
        prepare_args,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70011",
                target_pdb="1GGG",
            ),
            make_row(heavy_sequence=HEAVY_B, antigen_sequence=ANTIGEN_2, target_pdb="1GGG"),
        ],
    )

    legacy = by_antigen(records)[ANTIGEN_2]
    assert legacy["target_key"] == "pdb:1ggg"
    assert legacy["canonical_target_id"] == "uniprot:p70011"
    # The split follows the canonical id, not the legacy key.
    assert legacy["split"] == paa.deterministic_split(
        "uniprot:p70011", val_percent=prepare_args.val_percent
    )
    # Raw identifiers survive for audit: target_identity_nodes records what THIS
    # row declared (its own normalized identifiers), which is what makes the
    # merge reconstructible -- the component is recovered by grouping rows on
    # canonical_target_id.
    assert legacy["target_pdb"] == "1GGG"
    assert "pdb:1ggg" in legacy["target_identity_nodes"]
    assert "uniprot:p70011" not in legacy["target_identity_nodes"]
    assert "uniprot:p70011" in by_antigen(records)[ANTIGEN_1]["target_identity_nodes"]


def test_identity_index_reports_component_and_merge_counts(paa):
    index = paa.TargetIdentityIndex()
    for row in _mixed_corpus():
        index.observe_row(row)
    # One extra row whose only link to anything is a repeated antigen sequence.
    index.observe_row(
        make_row(
            heavy_sequence=HEAVY_B,
            antigen_sequence=ANTIGEN_1,
            target_uniprot="Q70099",
        )
    )
    index.finalize()
    stats = index.stats()

    # 6 named identifiers (p70007, 1eee, epsilon_antigen, q70008, zeta_antigen,
    # q70099) plus 5 antigen-sequence nodes.
    assert stats["target_identity_node_count"] == 11
    # Two biological targets: the 1EEE cluster and the zeta cluster.
    assert stats["target_components"] == 2
    # p70007~1eee, 1eee~epsilon_antigen, q70008~zeta_antigen.
    assert stats["target_alias_merges"] == 3
    # Only q70099, which shares ANTIGEN_1 with the 1EEE cluster. A row's own
    # antigen node attaching to its own identifiers is not counted.
    assert stats["target_sequence_merges"] == 1
    assert stats["target_rows_without_identity"] == 0
    assert stats["target_rows_without_identifier"] == 0
    # The sequence link really did fuse Q70099 into the P70007/1EEE component.
    nodes = paa.extract_target_nodes({"target_uniprot": "Q70099"}, ANTIGEN_1)
    assert index.canonical_id(nodes) == "uniprot:p70007"


def test_identity_index_counts_rows_with_no_usable_identity(paa):
    index = paa.TargetIdentityIndex()
    # No identifiers AND no antigen sequence: nothing to group on at all.
    index.observe_row(make_row(heavy_sequence=HEAVY_A, antigen_sequence=""))
    # No identifiers, but an antigen sequence still gives a groupable node.
    index.observe_row(make_row(heavy_sequence=HEAVY_B, antigen_sequence=ANTIGEN_1))
    index.observe_row(
        make_row(heavy_sequence=HEAVY_C, antigen_sequence=ANTIGEN_2, target_uniprot="P70013")
    )
    index.finalize()
    stats = index.stats()

    assert stats["target_rows_without_identity"] == 1
    assert stats["target_rows_without_identifier"] == 2


def test_build_processed_record_without_an_index_is_a_singleton(paa, prepare_args):
    """The identity index is optional; a lone row still gets a canonical id."""
    record, reason = paa.build_processed_record(
        row=make_row(
            heavy_sequence=HEAVY_A,
            antigen_sequence=ANTIGEN_1,
            target_uniprot="P70012",
            target_pdb="1HHH",
        ),
        shard_name="part-00000.parquet",
        row_idx=0,
        args=prepare_args,
    )

    assert reason == "kept"
    assert record["canonical_target_id"] == "uniprot:p70012"


# --------------------------------------------------------------------------- #
# End-to-end CLI (needs pyarrow; skipped where parquet support is unavailable)
# --------------------------------------------------------------------------- #
@pytest.fixture
def antigen_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "prepare_antibody_antigen.py"


def load_jsonl_gz(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_prepare_antibody_antigen(
    script_path: Path,
    input_path: Path,
    output_path: Path,
    extra_args: list[str] | None = None,
):
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _run_cli(script_path: Path, tmp_path: Path, rows: list[dict], name: str):
    raw_dir = tmp_path / name / "raw"
    out_path = tmp_path / name / "processed" / "antibody_antigen.jsonl.gz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(raw_dir / "part-00000.parquet", index=False)
    result = run_prepare_antibody_antigen(script_path, raw_dir, out_path)
    return load_jsonl_gz(out_path), result.stdout


def test_main_runs_two_passes_and_merges_aliases(paa, tmp_path: Path, monkeypatch):
    """
    Drive the real ``main()`` with parquet reading stubbed out.

    This is the only local coverage of the two-pass driver itself: ``pyarrow``
    is absent here, so the subprocess CLI tests below are skipped. It also pins
    the pass count -- the identity pass and the write pass must each read every
    shard exactly once.
    """
    shard = tmp_path / "part-00000.parquet"
    frame = pd.DataFrame(
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70014",
                target_pdb="1JJJ",
            ),
            make_row(heavy_sequence=HEAVY_B, antigen_sequence=ANTIGEN_2, target_pdb="1JJJ"),
        ]
    )
    reads: list[Path] = []

    def fake_read_parquet(path):
        reads.append(path)
        return frame.copy()

    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"
    monkeypatch.setattr(paa, "iter_parquet_files", lambda _input: iter([shard]))
    monkeypatch.setattr(paa.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_antibody_antigen",
            "--input",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
    )

    paa.main()

    assert reads == [shard, shard]
    rows = load_jsonl_gz(out_path)
    assert len(rows) == 2
    assert {row["canonical_target_id"] for row in rows} == {"uniprot:p70014"}
    assert len({row["split"] for row in rows}) == 1


def test_dedupe_semantics_are_unchanged_but_identity_sees_dropped_duplicates(
    paa, tmp_path: Path, monkeypatch
):
    """
    Pin the interaction between dedupe and the identity pass.

    Dedupe is untouched: it still keys on the exact (heavy, light, antigen)
    triple and still keeps whichever row was read first. But the identity pass
    runs BEFORE dedupe and over every input row, so a dropped duplicate still
    contributes its identifiers. Here the surviving row was annotated only with
    a PDB code and its canonical id is the UniProt accession that appears
    nowhere except on the row dedupe discarded -- which is the correct grouping,
    and worth being explicit about rather than surprised by.
    """
    shard = tmp_path / "part-00000.parquet"
    frame = pd.DataFrame(
        [
            make_row(heavy_sequence=HEAVY_A, antigen_sequence=ANTIGEN_1, target_pdb="1KKK"),
            # Identical modeling triple, different annotation. Dropped by dedupe.
            make_row(
                heavy_sequence=HEAVY_A, antigen_sequence=ANTIGEN_1, target_uniprot="P70020"
            ),
            make_row(heavy_sequence=HEAVY_B, antigen_sequence=ANTIGEN_2, target_pdb="1KKK"),
        ]
    )
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"
    monkeypatch.setattr(paa, "iter_parquet_files", lambda _input: iter([shard]))
    monkeypatch.setattr(paa.pd, "read_parquet", lambda _path: frame.copy())
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_antibody_antigen", "--input", str(tmp_path), "--output", str(out_path)],
    )

    paa.main()
    rows = load_jsonl_gz(out_path)

    # First-read-wins dedupe, unchanged.
    assert len(rows) == 2
    survivor = by_antigen(rows)[ANTIGEN_1]
    assert survivor["target_pdb"] == "1KKK"
    assert survivor["target_uniprot"] == ""
    # ...yet its component knows about the accession only the dropped row carried.
    assert survivor["canonical_target_id"] == "uniprot:p70020"
    assert {row["canonical_target_id"] for row in rows} == {"uniprot:p70020"}


def test_cli_merges_aliases_and_reports_identity_counts(tmp_path: Path, antigen_script_path: Path):
    pytest.importorskip("pyarrow")

    rows, stdout = _run_cli(
        antigen_script_path,
        tmp_path,
        [
            make_row(
                heavy_sequence=HEAVY_A,
                antigen_sequence=ANTIGEN_1,
                target_uniprot="P70001",
                target_pdb="1AAA",
            ),
            make_row(heavy_sequence=HEAVY_B, antigen_sequence=ANTIGEN_2, target_pdb="1AAA"),
        ],
        "cli_merge",
    )

    assert len(rows) == 2
    assert {row["canonical_target_id"] for row in rows} == {"uniprot:p70001"}
    assert len({row["split"] for row in rows}) == 1
    assert "target_components:" in stdout
    assert "target_alias_merges:" in stdout
    assert "target_sequence_merges:" in stdout
    assert "target_rows_without_identity:" in stdout


def test_cli_canonical_ids_are_independent_of_shard_row_order(
    tmp_path: Path, antigen_script_path: Path
):
    pytest.importorskip("pyarrow")

    corpus = _mixed_corpus()
    forward, _ = _run_cli(antigen_script_path, tmp_path, corpus, "cli_forward")
    backward, _ = _run_cli(antigen_script_path, tmp_path, list(reversed(corpus)), "cli_backward")

    assert _identity_map(forward) == _identity_map(backward)


# --------------------------------------------------------------------------- #
# Runtime consumption (step 8)
# --------------------------------------------------------------------------- #
def _runtime_record(
    *,
    record_id: str,
    heavy_sequence: str,
    antigen_sequence: str,
    target_key: str,
    canonical_target_id: str | None,
) -> dict:
    record = {
        "record_id": record_id,
        "sequence": heavy_sequence,
        "sequence_heavy": heavy_sequence,
        "sequence_light": None,
        "sequence_antigen": antigen_sequence,
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "split": "train",
        "length": len(heavy_sequence),
        "target_key": target_key,
        "target_name": "test_target",
        "target_pdb": "1abc",
        "target_uniprot": "P12345",
        "dataset": "asd-test",
        "confidence": "very_high",
        "affinity_type": "bool",
        "affinity_raw": "1.0",
        "processed_measurement_raw": "1.0",
        "processed_measurement_float": 1.0,
        "binder_label": 1,
        "is_strong_binder": True,
        "is_nanobody": True,
        "scfv": False,
        "cdr3_aa_heavy": "CARDRST",
        "cdr3_start_aa_heavy": 10,
        "cdr3_end_aa_heavy": 17,
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": False,
        "antigen_length": len(antigen_sequence),
    }
    if canonical_target_id is not None:
        record["canonical_target_id"] = canonical_target_id
    return record


def test_oas_record_parses_canonical_target_id(tmp_path: Path, write_processed_jsonl_gz):
    from smallAntibodyGen.data.MLMCollator import OASSequenceDataset

    path = write_processed_jsonl_gz(
        tmp_path / "canonical.jsonl.gz",
        [
            _runtime_record(
                record_id="antigen-1",
                heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
                antigen_sequence=ANTIGEN_1,
                target_key="pdb:1ggg",
                canonical_target_id="uniprot:p70011",
            )
        ],
    )
    ds = OASSequenceDataset(path, split="train")

    assert ds[0].canonical_target_id == "uniprot:p70011"
    assert ds[0].target_key == "pdb:1ggg"


def test_canonical_target_id_falls_back_to_legacy_key_on_older_corpora(
    tmp_path: Path, write_processed_jsonl_gz
):
    """Corpora written before this ticket have no canonical field; keep them usable."""
    from smallAntibodyGen.data.MLMCollator import OASSequenceDataset

    path = write_processed_jsonl_gz(
        tmp_path / "legacy.jsonl.gz",
        [
            _runtime_record(
                record_id="antigen-1",
                heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
                antigen_sequence=ANTIGEN_1,
                target_key="uniprot:p11111",
                canonical_target_id=None,
            )
        ],
    )
    ds = OASSequenceDataset(path, split="train")

    assert ds[0].canonical_target_id == "uniprot:p11111"


def test_antigen_collator_emits_canonical_target_ids(
    tmp_path: Path, tokenizer, write_processed_jsonl_gz
):
    from smallAntibodyGen.data.MLMCollator import AntibodyAntigenCollator, OASSequenceDataset

    path = write_processed_jsonl_gz(
        tmp_path / "collator.jsonl.gz",
        [
            _runtime_record(
                record_id="antigen-1",
                heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
                antigen_sequence=ANTIGEN_1,
                target_key="uniprot:p11111",
                canonical_target_id="uniprot:p11111",
            ),
            _runtime_record(
                record_id="antigen-2",
                heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
                antigen_sequence=ANTIGEN_2,
                target_key="pdb:1ggg",
                canonical_target_id="uniprot:p11111",
            ),
        ],
    )
    ds = OASSequenceDataset(path, split="train")
    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=0.0,
        rng_seed=42,
    )
    batch = collator([ds[0], ds[1]])

    assert "canonical_target_ids" in batch
    assert batch["canonical_target_ids"] == ["uniprot:p11111", "uniprot:p11111"]
    # Legacy metadata stays available for existing consumers.
    assert batch["target_keys"] == ["uniprot:p11111", "pdb:1ggg"]


def test_shuffled_negative_donor_respects_canonical_target_id(
    tmp_path: Path, tokenizer, write_processed_jsonl_gz
):
    """
    Control matching: two aliases of one target must not become each other's
    "non-cognate" antigen. Their legacy target_keys differ, so only the
    canonical id can rule the swap out.
    """
    from smallAntibodyGen.data.MLMCollator import AntibodyAntigenCollator, OASSequenceDataset

    path = write_processed_jsonl_gz(
        tmp_path / "donor.jsonl.gz",
        [
            _runtime_record(
                record_id="antigen-1",
                heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
                antigen_sequence=ANTIGEN_1,
                target_key="uniprot:p70011",
                canonical_target_id="uniprot:p70011",
            ),
            _runtime_record(
                record_id="antigen-2",
                heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVA",
                antigen_sequence=ANTIGEN_2,
                target_key="pdb:1ggg",
                canonical_target_id="uniprot:p70011",
            ),
        ],
    )
    ds = OASSequenceDataset(path, split="train")
    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_antigen_probability=1.0,
        rng_seed=7,
    )
    batch = collator([ds[0], ds[1]])

    # No valid donor exists, so nothing may be relabeled as a negative.
    assert batch["is_shuffled_antigen"].tolist() == [False, False]
    assert batch["compatibility_labels"].tolist() == [1, 1]


# --------------------------------------------------------------------------- #
# Leakage report consumer
# --------------------------------------------------------------------------- #
def test_summarize_target_overlap_uses_canonical_target_id(project_root):
    from dataclasses import dataclass

    module = _load_script(project_root, "mlm_train")

    @dataclass
    class _Rec:
        target_key: str
        canonical_target_id: str

    class _DS:
        def __init__(self, records):
            self.records = records

    # Two aliases of ONE target, one on each side of the split. Grouping on
    # target_key reports zero overlap; grouping on canonical_target_id reports one.
    train = _DS([_Rec(target_key="uniprot:p1", canonical_target_id="uniprot:p1")])
    val = _DS([_Rec(target_key="pdb:1abc", canonical_target_id="uniprot:p1")])

    assert module.summarize_target_overlap(train, val)["overlap"] == 1


# --------------------------------------------------------------------------- #
# Shortcut baselines (Gate 2A reads these)
# --------------------------------------------------------------------------- #
def test_group_majority_baselines_group_on_canonical_target_id(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    """
    The target-family shortcut baseline must group on `canonical_target_id`.

    Gate 2A requires Stage 3 to beat "prevalence, target-family, and source-study
    baselines". Grouping the target-family baseline on the legacy `target_key`
    splits one biological target across several groups whenever it appears under
    different accessions, which leaves fewer labeled rows per group and falls back
    to the global majority more often -- an ARTIFICIALLY WEAK baseline that a
    model clears too easily. Beating a baseline that is weak for a bookkeeping
    reason is not evidence about the model.
    """
    mlm_train = _load_script(project_root, "mlm_train")

    # One biological target under two aliases: same canonical id, different
    # legacy keys. Its rows carry a consistent label, so a baseline that groups
    # them together predicts perfectly and one that splits them does not.
    def row(record_id, legacy_key, canonical, heavy, binder):
        return {
            "record_id": record_id,
            "sequence": heavy,
            "sequence_heavy": heavy,
            "sequence_light": None,
            "sequence_antigen": ANTIGEN_1,
            "locus": "PAIRED_ANTIGEN",
            "chain_group": "paired_antigen",
            "split": "train",
            "length": len(heavy),
            "target_key": legacy_key,
            "canonical_target_id": canonical,
            "dataset": "asd-test",
            "confidence": "very_high",
            "affinity_type": "bool",
            "processed_measurement_raw": str(float(binder)),
            "processed_measurement_float": float(binder),
            "binder_label": binder,
            "heavy_locus": "IGH",
            "is_paired": False,
            "metadata": {},
        }

    records = []
    for i, heavy in enumerate((HEAVY_A, HEAVY_B, HEAVY_C, HEAVY_D)):
        # Alternate the alias so the legacy key fragments the group.
        legacy = "uniprot:p12345" if i % 2 == 0 else "pdb:1abc"
        records.append(row(f"r{i}", legacy, "uniprot:p12345", heavy, 1))

    from smallAntibodyGen.data.MLMCollator import OASSequenceDataset

    data_path = write_processed_jsonl_gz(tmp_path / "aliased.jsonl.gz", records)
    dataset = OASSequenceDataset(str(data_path), split="train")
    assert {r.target_key for r in dataset.records} == {"uniprot:p12345", "pdb:1abc"}
    assert {r.canonical_target_id for r in dataset.records} == {"uniprot:p12345"}

    cfg = mlm_train.TrainConfig(
        data_path=str(data_path),
        training_stage="antigen_real_label_refine",
        init_checkpoint="parent.pt",
        batch_size=2,
        eval_batch_size=2,
        max_length=192,
        hcdr3_span_probability=0.0,
    )
    baselines = mlm_train.fit_group_majority_baselines(dataset, tokenizer, cfg)
    metrics = mlm_train.evaluate_group_majority_baselines(
        dataset, tokenizer, cfg, baselines
    )

    # Both are reported, and the canonical one is the Gate-2A number.
    assert "canonical_target_majority_acc" in metrics
    assert "target_key_majority_acc" in metrics
    assert metrics["canonical_target_majority_acc"] >= metrics["target_key_majority_acc"]

    summary = mlm_train.format_baseline_summary(metrics, prefix="val")
    assert "val_canonical_target_majority_acc=" in summary
