from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smallAntibodyGen.data.oas import (
    choose_sequence_column,
    clean_aa_sequence,
    infer_chain,
    iter_clean_oas_records,
    normalize_flag,
    parse_possible_json_metadata,
    read_oas_table,
)

def test_parse_possible_json_metadata_accepts_plain_json():
    line = json.dumps({"Run": "TEST", "Species": "human"})
    meta = parse_possible_json_metadata(line)

    assert meta is not None
    assert meta["Run"] == "TEST"
    assert meta["Species"] == "human"


def test_parse_possible_json_metadata_accepts_csv_quoted_json():
    raw = json.dumps({"Run": "TEST", "Species": "human"}).replace('"', '""')
    line = f'"{raw}"'
    meta = parse_possible_json_metadata(line)

    assert meta is not None
    assert meta["Run"] == "TEST"
    assert meta["Species"] == "human"


def test_read_oas_table_handles_comma_delimited_with_metadata(
    tmp_path: Path, write_oas_data_unit, make_oas_row, heavy_seq, heavy_cdr3
):
    file_path = tmp_path / "comma.csv.gz"
    write_oas_data_unit(
        file_path,
        rows=[make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3)],
        metadata={"Run": "TEST_RUN", "Chain": "Heavy"},
        quoted_metadata=True,
        delimiter=",",
    )

    metadata, df = read_oas_table(file_path)

    assert metadata["Run"] == "TEST_RUN"
    assert "sequence_alignment_aa" in df.columns
    assert len(df) == 1


def test_read_oas_table_handles_tab_delimited_with_metadata(
    tmp_path: Path, write_oas_data_unit, make_oas_row, heavy_seq, heavy_cdr3
):
    file_path = tmp_path / "tab.csv.gz"
    write_oas_data_unit(
        file_path,
        rows=[make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3)],
        metadata={"Run": "TEST_RUN", "Chain": "Heavy"},
        quoted_metadata=True,
        delimiter="\t",
    )

    metadata, df = read_oas_table(file_path)

    assert metadata["Run"] == "TEST_RUN"
    assert "sequence_alignment_aa" in df.columns
    assert len(df) == 1


def test_normalize_flag_handles_common_values():
    assert normalize_flag("T") is True
    assert normalize_flag("true") is True
    assert normalize_flag("1") is True
    assert normalize_flag("F") is False
    assert normalize_flag("false") is False
    assert normalize_flag("0") is False
    assert normalize_flag(None) is None
    assert normalize_flag(float("nan")) is None
    assert normalize_flag("weird") is None


def test_infer_chain_maps_heavy_light_and_nano():
    row_h = pd.Series({"locus": "H"})
    row_k = pd.Series({"locus": "IGK"})
    row_l = pd.Series({"locus": "L"})
    row_vhh = pd.Series({"locus": "VHH"})

    assert infer_chain(row_h, {}) == "heavy"
    assert infer_chain(row_k, {}) == "light"
    assert infer_chain(row_l, {}) == "light"
    assert infer_chain(row_vhh, {}) == "nano"


def test_clean_aa_sequence_removes_noise_but_keeps_allowed_symbols():
    cleaned = clean_aa_sequence(" cAr-*D12Xbzou ")
    # adjust this assertion if you intentionally dropped ambiguity symbols
    assert cleaned == "CARDXBZOU"


def test_choose_sequence_column_prefers_full_variable_domain_field():
    df = pd.DataFrame(
        {
            "sequence_alignment_aa": ["AAA"],
            "v_sequence_alignment_aa": ["BBB"],
            "sequence_aa": ["CCC"],
            "sequence": ["DDD"],
        }
    )
    assert choose_sequence_column(df) == "sequence_alignment_aa"


def test_iter_clean_oas_records_filters_bad_rows(
    tmp_path: Path,
    write_oas_data_unit,
    make_oas_row,
    heavy_seq,
    heavy_cdr3,
):
    rows = [
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, productive="T", stop_codon="F", vj_in_frame="T"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, productive="F", stop_codon="F", vj_in_frame="T"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, productive="T", stop_codon="T", vj_in_frame="T"),
        make_oas_row(sequence_alignment_aa=heavy_seq, cdr3_aa=heavy_cdr3, productive="T", stop_codon="F", vj_in_frame="F"),
    ]
    path = tmp_path / "tiny.csv.gz"
    write_oas_data_unit(path, rows, metadata={"Chain": "Heavy"}, quoted_metadata=True)

    records = list(iter_clean_oas_records(path, min_length=70, max_length=200))
    assert len(records) == 1
    assert records[0].sequence == heavy_seq
    assert records[0].chain == "heavy"