"""
Raw OAS table ingestion: turn an OAS CSV/gzip data unit into `(metadata, DataFrame)`.

This module is deliberately narrow. It reads; it does not interpret. Extracting
sequences, loci, flags and regions from a row is the job of `scripts/prepare_oas.py`
(`build_variable_aa`, `normalize_locus`, `clean_aa_sequence`, ...), which is the single
authority on what a row means. A second, parallel interpretation used to live here and
drifted from that authority in several places before being removed; keep this module a
reader so it cannot drift again.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def parse_possible_json_metadata(line: str) -> Dict[str, Any] | None:
    if not line:
        return None

    stripped = line.lstrip("\ufeff").strip()
    if not stripped:
        return None

    if stripped.startswith("#"):
        stripped = stripped[1:].strip()

    candidates: list[str] = [stripped]

    # If the line is a CSV-quoted single field like:
    # "{""Run"": ""SRR3099049"", ...}"
    # this will unwrap it to:
    # {"Run": "SRR3099049", ...}
    try:
        row = next(csv.reader([stripped]))
        if len(row) == 1:
            candidates.append(row[0].strip())
    except Exception:
        pass

    # Additional fallbacks
    expanded: list[str] = []
    for c in candidates:
        expanded.append(c)

        if c.startswith('"') and c.endswith('"'):
            inner = c[1:-1]
            expanded.append(inner)
            expanded.append(inner.replace('""', '"'))

        expanded.append(c.replace('""', '"'))

    seen = set()
    for c in expanded:
        c = c.strip()
        if c in seen:
            continue
        seen.add(c)

        if not c.startswith("{"):
            continue

        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_oas_table(path: Path) -> tuple[Dict[str, Any], pd.DataFrame]:
    with open_text_maybe_gzip(path) as f:
        first_line = f.readline()
        metadata = parse_possible_json_metadata(first_line)
        has_metadata = metadata is not None
        if has_metadata:
            sample = f.read(4096)
        else:
            metadata = {}
            sample = first_line + f.read(4096)

    # OAS files are described as .csv.gz; allow a fallback sniffer to tolerate odd exports.
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    # Reopen the file and stream the table into pandas so large gzip members
    # do not have to be materialized as one giant in-memory string first.
    with open_text_maybe_gzip(path) as f:
        if has_metadata:
            f.readline()
        df = pd.read_csv(f, sep=delimiter, low_memory=False)
    return metadata, df
