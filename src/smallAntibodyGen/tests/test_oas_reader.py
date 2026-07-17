from __future__ import annotations

import json
from pathlib import Path

from smallAntibodyGen.data.oas import (
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


def test_parse_possible_json_metadata_strips_bom():
    # OAS exports can carry a UTF-8 BOM ahead of the header JSON blob.
    # chr(0xFEFF) rather than a literal: an invisible char in source is a trap.
    meta = parse_possible_json_metadata(chr(0xFEFF) + json.dumps({"Run": "TEST"}))

    assert meta is not None
    assert meta["Run"] == "TEST"


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
