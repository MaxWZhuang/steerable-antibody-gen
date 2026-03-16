from __future__ import annotations

import csv
import gzip
import io
import json
import re

from tabulate import tabulate
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pandas as pd

CANONICAL_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class OASSequenceRecord:
    sequence: str
    chain: str
    source_file: str
    metadata: Dict[str, Any]
    extra: Dict[str, Any]


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
        if metadata is not None:
            remaining = f.read()
            buffer = io.StringIO(remaining)
        else:
            metadata = {}
            remaining = first_line + f.read()
            buffer = io.StringIO(remaining)
        # OAS files are described as .csv.gz; allow a fallback sniffer to tolerate odd exports.
        sample = buffer.read(4096)
        buffer.seek(0)
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        df = pd.read_csv(buffer, sep=delimiter, low_memory=False)
    return metadata, df


def normalize_flag(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"t", "true", "1", "yes"}:
        return True
    if text in {"f", "false", "0", "no"}:
        return False
    return None


def infer_chain(row: pd.Series, metadata: Dict[str, Any]) -> str:
    candidates = [
        row.get("locus"),
        row.get("chain"),
        metadata.get("Chain"),
        metadata.get("chain"),
    ]
    for item in candidates:
        if item is None or pd.isna(item):
            continue
        text = str(item).strip().upper()
        if text in {"IGH", "VH", "H", "HEAVY"}:
            return "heavy"
        if text in {"IGK", "IGL", "VL", "L", "LIGHT"}:
            return "light"
        if text in {"VHH", "NANO", "NANOBODY"}:
            return "nano"
    return "heavy"


def clean_aa_sequence(seq: Any) -> str:
    if seq is None or pd.isna(seq):
        return ""
    text = str(seq).upper()
    text = re.sub(r"[^A-Z]", "", text)
    text = "".join(ch for ch in text if ch in CANONICAL_AA_SET)
    return text


def choose_sequence_column(df: pd.DataFrame) -> str:
    candidates = [
        "sequence_alignment_aa",
        "v_sequence_alignment_aa",
        "sequence_aa",
        "sequence",
        "sequence_heavy",
        "sequence_light",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a usable amino-acid sequence column. Available columns: {list(df.columns)[:25]}"
    )


def iter_clean_oas_records(
    path: Path,
    min_length: int = 70,
    max_length: int = 160,
) -> Iterator[OASSequenceRecord]:
    metadata, df = read_oas_table(path)
    seq_col = choose_sequence_column(df)
    for _, row in df.iterrows():
        stop_codon = normalize_flag(row.get("stop_codon"))
        productive = normalize_flag(row.get("productive"))
        vj_in_frame = normalize_flag(row.get("vj_in_frame"))
        if stop_codon is True:
            continue
        if productive is False:
            continue
        if vj_in_frame is False:
            continue

        seq = clean_aa_sequence(row.get(seq_col))
        if not (min_length <= len(seq) <= max_length):
            continue

        chain = infer_chain(row, metadata)
        extra = {
            "v_call": None if pd.isna(row.get("v_call")) else row.get("v_call"),
            "j_call": None if pd.isna(row.get("j_call")) else row.get("j_call"),
            "cdr3_aa": None if pd.isna(row.get("cdr3_aa")) else row.get("cdr3_aa"),
        }
        yield OASSequenceRecord(
            sequence=seq,
            chain=chain,
            source_file=path.name,
            metadata=metadata,
            extra=extra,
        )
