from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")


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


@pytest.fixture
def antigen_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "prepare_antibody_antigen.py"


@pytest.fixture
def write_antibody_antigen_parquet():
    def _write(path: Path, rows: list[dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return path

    return _write


@pytest.fixture
def make_antibody_antigen_row(heavy_seq: str, heavy_cdr3: str, light_seq: str, light_cdr3: str):
    def _make(
        *,
        dataset: str = "asd-test",
        heavy_sequence: str = heavy_seq,
        light_sequence: str = light_seq,
        antigen_sequence: str = "MKTIIALSYIFCLVFADYKDDDDK",
        scfv: bool = False,
        affinity_type: str = "bool",
        affinity: str = "1.0",
        confidence: str = "very_high",
        nanobody: bool | None = False,
        processed_measurement: str = "1.0",
        target_name: str = "test_target",
        target_pdb: str = "1abc",
        target_uniprot: str = "P12345",
        source_url: str = "https://example.org/asd",
        heavy_alignment: str | None = None,
        light_alignment: str | None = None,
        heavy_cdr3_override: str | None = None,
        light_cdr3_override: str | None = None,
    ) -> dict:
        return {
            "dataset": dataset,
            "heavy_sequence": heavy_sequence,
            "light_sequence": light_sequence,
            "scfv": scfv,
            "affinity_type": affinity_type,
            "affinity": affinity,
            "antigen_sequence": antigen_sequence,
            "confidence": confidence,
            "nanobody": nanobody,
            "processed_measurement": processed_measurement,
            "metadata": {
                "target_name": target_name,
                "target_pdb": target_pdb,
                "target_uniprot": target_uniprot,
                "source_url": source_url,
                "heavy_riot_numbering": {
                    "sequence_alignment_aa": heavy_alignment or heavy_sequence,
                    "cdr1_aa": "GFTFSSYA",
                    "cdr2_aa": "ISGSGGST",
                    "cdr3_aa": heavy_cdr3_override or heavy_cdr3,
                },
                "light_riot_numbering": {
                    "sequence_alignment_aa": light_alignment or light_sequence,
                    "cdr1_aa": "QDINNYLN",
                    "cdr2_aa": "TSRLHSGV",
                    "cdr3_aa": light_cdr3_override or light_cdr3,
                },
            },
        }

    return _make


def test_prepare_antibody_antigen_prefers_alignment_and_computes_hcdr3_span(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    aligned_heavy = (
        "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRF"
        "TISRDNSKNTLYLQMNSLRAEDTAVYYCAKNDILVGYSAFDYWGQGTLVTVSS"
    )
    row = make_antibody_antigen_row(
        heavy_sequence="MKKLL" + aligned_heavy + "HHHHHH",
        heavy_alignment=aligned_heavy,
    )
    write_antibody_antigen_parquet(raw_dir / "part-00000.parquet", [row])

    run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)
    rows = load_jsonl_gz(out_path)
    assert len(rows) == 1

    rec = rows[0]
    assert rec["sequence_heavy"] == aligned_heavy
    assert rec["heavy_sequence_source"] == "metadata_sequence_alignment_aa"
    assert rec["cdr3_aa_heavy"] == "AKNDILVGYSAFDY"
    assert rec["cdr3_start_aa_heavy"] is not None
    assert rec["cdr3_end_aa_heavy"] is not None
    assert rec["sequence_heavy"][rec["cdr3_start_aa_heavy"]:rec["cdr3_end_aa_heavy"]] == rec["cdr3_aa_heavy"]


def test_prepare_antibody_antigen_supports_heavy_only_nanobody_records(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    row = make_antibody_antigen_row(
        light_sequence="",
        nanobody=True,
        target_uniprot="",
        target_pdb="5ovw",
    )
    row["metadata"]["light_riot_numbering"] = {
        "sequence_alignment_aa": None,
        "cdr1_aa": None,
        "cdr2_aa": None,
        "cdr3_aa": None,
    }
    write_antibody_antigen_parquet(raw_dir / "part-00000.parquet", [row])

    run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)
    rows = load_jsonl_gz(out_path)
    assert len(rows) == 1

    rec = rows[0]
    assert rec["is_paired"] is False
    assert rec["is_nanobody"] is True
    assert rec["sequence_light"] is None
    assert rec["cdr3_aa_light"] is None


def test_prepare_antibody_antigen_dedupes_exact_triples(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    row = make_antibody_antigen_row()
    duplicate = make_antibody_antigen_row()
    distinct = make_antibody_antigen_row(antigen_sequence="ACDEFGHIKLMNPQRSTVWY")
    write_antibody_antigen_parquet(raw_dir / "part-00000.parquet", [row, duplicate, distinct])

    result = run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)
    rows = load_jsonl_gz(out_path)

    assert len(rows) == 2
    assert "duplicates_dropped:  1" in result.stdout


def test_prepare_antibody_antigen_uses_target_aware_split_keys(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    row_a = make_antibody_antigen_row(
        heavy_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKNDILVGYSAFDYWGQGTLVTVSS",
        target_uniprot="P99999",
        target_pdb="",
        target_name="same_target",
    )
    row_b = make_antibody_antigen_row(
        heavy_sequence="QVQLQESGGGLVQAGGSLRLSCAASGFTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNARNTVYLQMNSLKPEDTAVYYCAKNDILVGYSAFDYWGQGTQVTVSS",
        target_uniprot="P99999",
        target_pdb="",
        target_name="same_target",
        antigen_sequence="MNNNKQQQQQQQQQQQQQQQ",
    )
    write_antibody_antigen_parquet(raw_dir / "part-00000.parquet", [row_a, row_b])

    run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)
    rows = load_jsonl_gz(out_path)

    assert len(rows) == 2
    assert rows[0]["target_key"] == "uniprot:p99999"
    assert rows[1]["target_key"] == "uniprot:p99999"
    assert rows[0]["split"] == rows[1]["split"]


def test_prepare_antibody_antigen_only_assigns_binder_label_for_bool_rows(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    bool_row = make_antibody_antigen_row(affinity_type="bool", processed_measurement="0.0")
    kd_row = make_antibody_antigen_row(
        affinity_type="kd",
        processed_measurement="1e-9",
        antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
        target_uniprot="Q11111",
        target_pdb="",
    )
    write_antibody_antigen_parquet(raw_dir / "part-00000.parquet", [bool_row, kd_row])

    run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)
    rows = load_jsonl_gz(out_path)

    assert len(rows) == 2
    by_type = {row["affinity_type"]: row for row in rows}
    assert by_type["bool"]["binder_label"] == 0
    assert by_type["kd"]["binder_label"] is None


def test_prepare_antibody_antigen_filters_by_confidence_and_antigen_length(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    keep_row = make_antibody_antigen_row(confidence="high")
    low_conf_row = make_antibody_antigen_row(
        confidence="medium",
        antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
        target_uniprot="Q22222",
        target_pdb="",
    )
    long_antigen_row = make_antibody_antigen_row(
        confidence="high",
        antigen_sequence="A" * 3000,
        target_uniprot="Q33333",
        target_pdb="",
    )
    write_antibody_antigen_parquet(raw_dir / "part-00000.parquet", [keep_row, low_conf_row, long_antigen_row])

    result = run_prepare_antibody_antigen(
        antigen_script_path,
        raw_dir,
        out_path,
        extra_args=["--max-antigen", "2048", "--allowed-confidence", "high,very_high"],
    )
    rows = load_jsonl_gz(out_path)

    assert len(rows) == 1
    assert rows[0]["confidence"] == "high"
    assert "confidence_filtered" in result.stdout
    assert "antigen_length_out_of_range" in result.stdout
