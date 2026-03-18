from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


def load_jsonl_gz(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_prepare_oas(
    script_path: Path,
    input_dir: Path,
    output_dir: Path,
    stats_path: Path,
    extra_args: list[str] | None = None,
):
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--stats-output",
        str(stats_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result


def test_prepare_oas_preserves_metadata_from_csv_quoted_json(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    metadata = {"Run": "SRR_TEST", "Species": "human", "Chain": "Heavy"}
    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(
        raw_file,
        rows=[make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3)],
        metadata=metadata,
        quoted_metadata=True,
    )

    run_prepare_oas(script_path, raw_dir, out_dir, stats_path)

    rows = load_jsonl_gz(out_dir / "oas_all.jsonl.gz")
    assert len(rows) == 1

    rec = rows[0]
    # adjust if your metadata is flattened instead of nested
    assert rec["metadata"]["run"] == "SRR_TEST"
    assert rec["metadata"]["species"] == "human"


def test_prepare_oas_dedupes_within_locus_but_not_across_locus(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    rows = [
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, locus="H"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, locus="H"),   # duplicate same locus
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, locus="K", d_call=""),  # same seq, different locus
    ]
    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(raw_file, rows, metadata={"Chain": "Heavy"}, quoted_metadata=True)

    run_prepare_oas(
        script_path,
        raw_dir,
        out_dir,
        stats_path,
        extra_args=["--min-heavy", "20", "--min-light", "20"],
    )

    stats = json.loads(stats_path.read_text())
    assert stats["records_kept"] == 2
    assert stats["duplicates_dropped"] == 1
    assert stats["kept_by_locus"]["IGH"] == 1
    assert stats["kept_by_locus"]["IGK"] == 1


def test_prepare_oas_computes_valid_cdr3_spans(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(
        raw_file,
        rows=[make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3)],
        metadata={"Chain": "Heavy"},
        quoted_metadata=True,
    )

    run_prepare_oas(script_path, raw_dir, out_dir, stats_path)

    rows = load_jsonl_gz(out_dir / "oas_all.jsonl.gz")
    assert len(rows) == 1

    rec = rows[0]
    start = rec["cdr3_start_aa"]
    end = rec["cdr3_end_aa"]
    assert start is not None
    assert end is not None
    assert rec["sequence"][start:end] == rec["cdr3_aa"]


def test_prepare_oas_rowwise_fallback_to_frcdr_concat(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
):
    """
    Liability test:
    If the script chooses sequence source globally by column presence,
    this test will fail on the second row.

    We want row-wise fallback behavior.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    concat_seq = "EVQL" + "GGGG" + "TTTT" + "YYYY" + "AAAAAKKKK" + "CARDRSTY" + "WGQGTLVTVS"

    row1 = make_oas_row(
        sequence_alignment_aa="QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGGYWGQGTLVTVSS",
        cdr3_aa="ARGGGY",
        locus="H",
    )
    row2 = make_oas_row(
        sequence_alignment_aa="",
        cdr3_aa="CARDRSTY",
        locus="H",
        fwr1_aa="EVQL",
        cdr1_aa="GGGG",
        fwr2_aa="TTTT",
        cdr2_aa="YYYY",
        fwr3_aa="AAAAAKKKK",
        fwr4_aa="WGQGTLVTVS",
    )

    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(raw_file, [row1, row2], metadata={"Chain": "Heavy"}, quoted_metadata=True)

    run_prepare_oas(
        script_path,
        raw_dir,
        out_dir,
        stats_path,
        extra_args=["--min-heavy", "20"],
    )

    rows = load_jsonl_gz(out_dir / "oas_all.jsonl.gz")
    assert len(rows) == 2

    seqs = {r["sequence"] for r in rows}
    assert concat_seq in seqs

    stats = json.loads(stats_path.read_text())
    # adjust if you name this field differently
    assert stats["sequence_source_counts"]["frcdr_concat"] == 1


def test_prepare_oas_redundancy_sum_by_locus_is_correct(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
    light_seq,
    light_cdr3,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    rows = [
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, redundancy=3, locus="H"),
        make_oas_row(sequence_alignment_aa=heavy_seq[:-1] + "A", cdr3_aa=heavy_cdr3[:-1] + "A", redundancy=5, locus="H"),
        make_oas_row(sequence_alignment_aa=light_seq, cdr3_aa=light_cdr3, redundancy=7, locus="K", d_call=""),
    ]
    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(raw_file, rows, metadata={"Chain": "Heavy"}, quoted_metadata=True)

    run_prepare_oas(
        script_path,
        raw_dir,
        out_dir,
        stats_path,
        extra_args=["--min-heavy", "20", "--min-light", "20"],
    )

    stats = json.loads(stats_path.read_text())
    assert stats["redundancy_sum_by_locus"]["IGH"] == 8
    assert stats["redundancy_sum_by_locus"]["IGK"] == 7


def test_prepare_oas_drop_reasons_are_accounted_for(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "processed"
    stats_path = tmp_path / "stats.json"

    rows = [
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, stop_codon="T"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, productive="F"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, vj_in_frame="F"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, locus="OTHER"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3),  # one valid row
    ]
    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(raw_file, rows, metadata={"Chain": "Heavy"}, quoted_metadata=True)

    run_prepare_oas(script_path, raw_dir, out_dir, stats_path)

    stats = json.loads(stats_path.read_text())

    assert stats["records_seen"] == 5
    assert stats["records_kept"] == 1
    assert stats["drop_reasons"]["stop_codon"] == 1
    assert stats["drop_reasons"]["non_productive"] == 1
    assert stats["drop_reasons"]["out_of_frame"] == 1
    assert stats["drop_reasons"]["bad_locus"] == 1


def test_prepare_oas_split_is_stable_across_runs(
    tmp_path: Path,
    script_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    rows = [
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3),
        make_oas_row(sequence_alignment_aa=heavy_seq[:-1] + "A", cdr3_aa=heavy_cdr3[:-1] + "A"),
    ]
    raw_file = raw_dir / "tiny.csv.gz"
    write_oas_data_unit(raw_file, rows, metadata={"Chain": "Heavy"}, quoted_metadata=True)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    stats1 = tmp_path / "stats1.json"
    stats2 = tmp_path / "stats2.json"

    run_prepare_oas(script_path, raw_dir, out1, stats1)
    run_prepare_oas(script_path, raw_dir, out2, stats2)

    rows1 = load_jsonl_gz(out1 / "oas_all.jsonl.gz")
    rows2 = load_jsonl_gz(out2 / "oas_all.jsonl.gz")

    split_map_1 = {(r["locus"], r["sequence"]): r["split"] for r in rows1}
    split_map_2 = {(r["locus"], r["sequence"]): r["split"] for r in rows2}

    assert split_map_1 == split_map_2