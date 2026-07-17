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
    assert by_type["bool"]["is_strong_binder"] is False
    assert by_type["kd"]["is_strong_binder"] is True


def test_prepare_antibody_antigen_marks_requested_strong_binder_categories(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    raw_dir = tmp_path / "raw"
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    bool_positive = make_antibody_antigen_row(
        affinity_type="bool",
        processed_measurement="1.0",
    )
    fuzzy_high = make_antibody_antigen_row(
        affinity_type="fuzzy",
        affinity="h",
        processed_measurement="h",
        antigen_sequence="ACDEFGHIKLMNPQRSTVWY",
        target_uniprot="Q11111",
        target_pdb="",
    )
    fuzzy_mid = make_antibody_antigen_row(
        affinity_type="fuzzy",
        affinity="m",
        processed_measurement="m",
        antigen_sequence="MNNNKQQQQQQQQQQQQQQQ",
        target_uniprot="Q22222",
        target_pdb="",
    )
    kd_strong = make_antibody_antigen_row(
        affinity_type="kd",
        processed_measurement="1e-10",
        antigen_sequence="GGGGGGGGGGGGGGGGGGGG",
        target_uniprot="Q33333",
        target_pdb="",
    )
    kd_weak = make_antibody_antigen_row(
        affinity_type="kd",
        processed_measurement="1e-8",
        antigen_sequence="TTTTTTTTTTTTTTTTTTTT",
        target_uniprot="Q44444",
        target_pdb="",
    )
    neg_log_kd_strong = make_antibody_antigen_row(
        affinity_type="-log KD",
        processed_measurement="9.5",
        antigen_sequence="VVVVVVVVVVVVVVVVVVVV",
        target_uniprot="Q55555",
        target_pdb="",
    )
    neg_log_kd_weak = make_antibody_antigen_row(
        affinity_type="-log KD",
        processed_measurement="8.5",
        antigen_sequence="YYYYYYYYYYYYYYYYYYYY",
        target_uniprot="Q66666",
        target_pdb="",
    )
    write_antibody_antigen_parquet(
        raw_dir / "part-00000.parquet",
        [bool_positive, fuzzy_high, fuzzy_mid, kd_strong, kd_weak, neg_log_kd_strong, neg_log_kd_weak],
    )

    run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)
    rows = load_jsonl_gz(out_path)

    assert len(rows) == 7
    by_type_and_measurement = {
        (row["affinity_type"], row["processed_measurement_raw"]): row
        for row in rows
    }
    assert by_type_and_measurement[("bool", "1.0")]["is_strong_binder"] is True
    assert by_type_and_measurement[("fuzzy", "h")]["is_strong_binder"] is True
    assert by_type_and_measurement[("fuzzy", "m")]["is_strong_binder"] is False
    assert by_type_and_measurement[("kd", "1e-10")]["is_strong_binder"] is True
    assert by_type_and_measurement[("kd", "1e-8")]["is_strong_binder"] is False
    assert by_type_and_measurement[("-log KD", "9.5")]["is_strong_binder"] is True
    assert by_type_and_measurement[("-log KD", "8.5")]["is_strong_binder"] is False


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


def _nanomolar_kd_rows(make_row, n: int = 60) -> list[dict]:
    # n DISTINCT kd rows whose measurements are nanomolar (median 50 nM), so no
    # row clears the <=1.0 nM strong bar -> the dataset reports zero strong
    # binders with a median far above the molar/nanomolar boundary, which is
    # exactly the mislabeled-units shape --strict-units exists to catch. Rows must
    # be distinct on the (heavy, light, antigen) dedupe triple to be counted, and
    # the guard only fires above 50 values.
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return [
        make_row(
            affinity_type="kd",
            affinity="50.0",
            processed_measurement="50.0",
            antigen_sequence="MKTIIALSYIFCLVFADYKDD" + aa[i // 20] + aa[i % 20],
        )
        for i in range(n)
    ]


def test_strict_units_refuses_to_commit_the_corpus(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    # AB-03: the units guard raised only AFTER writer.close(), so the poisoned
    # corpus was already sitting at --output when the run failed -- a file
    # indistinguishable from a good one. The guard must prevent the artifact, not
    # merely report on it afterwards.
    raw_dir = tmp_path / "raw"
    write_antibody_antigen_parquet(
        raw_dir / "shard-0.parquet", _nanomolar_kd_rows(make_antibody_antigen_row)
    )
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_prepare_antibody_antigen(
            antigen_script_path, raw_dir, out_path, ["--strict-units"]
        )

    assert "[kd-units]" in excinfo.value.stderr
    assert not out_path.exists()  # pre-fix: the poisoned corpus was already here


def test_without_strict_units_the_same_corpus_is_committed_with_a_warning(
    tmp_path: Path,
    antigen_script_path: Path,
    write_antibody_antigen_parquet,
    make_antibody_antigen_row,
):
    # The staged-commit path must leave the default (warn-only) behavior intact:
    # same corpus, no --strict-units -> warns, and still commits every record.
    raw_dir = tmp_path / "raw"
    write_antibody_antigen_parquet(
        raw_dir / "shard-0.parquet", _nanomolar_kd_rows(make_antibody_antigen_row)
    )
    out_path = tmp_path / "processed" / "antibody_antigen.jsonl.gz"

    result = run_prepare_antibody_antigen(antigen_script_path, raw_dir, out_path)

    assert "WARNING" in result.stdout
    assert out_path.exists()
    assert len(load_jsonl_gz(out_path)) == 60
    # The staging file must not survive a successful run.
    assert not out_path.with_name(out_path.name + ".tmp").exists()
