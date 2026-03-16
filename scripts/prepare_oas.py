#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.data.oas import read_oas_table
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZOU")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare raw OAS files into chain-aware, pretokenized JSONL for MLM training."
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stats-output", type=Path, default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--val-percent", type=int, default=10)

    # chain-specific length filters in amino acids
    p.add_argument("--min-heavy", type=int, default=80)
    p.add_argument("--max-heavy", type=int, default=180)
    p.add_argument("--min-light", type=int, default=70)
    p.add_argument("--max-light", type=int, default=160)
    p.add_argument("--min-nano", type=int, default=70)
    p.add_argument("--max-nano", type=int, default=150)

    # tokenization
    p.add_argument("--token-max-length", type=int, default=192)

    # strictness
    p.add_argument("--require-complete-vdj", action="store_true")
    p.add_argument("--drop-ambiguous-cdr3-span", action="store_true")

    return p.parse_args()


def normalize_bool(value: Any) -> Optional[bool]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"t", "true", "1", "yes"}:
        return True
    if text in {"f", "false", "0", "no"}:
        return False
    return None


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(text)
    except ValueError:
        return default


def clean_aa_sequence(seq: Any) -> str:
    if seq is None or pd.isna(seq):
        return ""
    text = str(seq).upper()
    text = "".join(ch for ch in text if ch.isalpha())
    text = "".join(ch for ch in text if ch in VALID_AA)
    return text


def normalize_locus(row: pd.Series, metadata: Dict[str, Any]) -> str:
    """
    Preserve chain identity explicitly:
    H / IGH -> IGH
    K / IGK -> IGK
    L / IGL -> IGL
    VHH / NANO / NANOBODY -> VHH
    """
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
            return "IGH"
        if text in {"IGK", "K", "KAPPA"}:
            return "IGK"
        if text in {"IGL", "L", "LAMBDA"}:
            return "IGL"
        if text in {"VHH", "NANO", "NANOBODY"}:
            return "VHH"
    return "OTHER"


def chain_group_from_locus(locus: str) -> str:
    if locus == "IGH":
        return "heavy"
    if locus in {"IGK", "IGL"}:
        return "light"
    if locus == "VHH":
        return "nano"
    return "other"


def extract_basic_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    return {
        "run": lowered.get("run"),
        "link": lowered.get("link"),
        "author": lowered.get("author"),
        "species": lowered.get("species"),
        "bsource": lowered.get("bsource"),
        "btype": lowered.get("btype"),
        "vaccine": lowered.get("vaccine"),
        "disease": lowered.get("disease"),
        "age": lowered.get("age"),
        "subject": lowered.get("subject"),
        "longitudinal": lowered.get("longitudinal"),
        "declared_chain": lowered.get("chain"),
        "declared_isotype": lowered.get("isotype"),
        "unique_sequences_in_file": lowered.get("unique sequences"),
        "total_sequences_in_file": lowered.get("total sequences"),
    }


def build_variable_aa(row: pd.Series) -> Tuple[str, str]:
    """
    Prefer the full variable-domain AA sequence.
    sequence_alignment_aa is usually the correct primary field.
    """
    seq_alignment_aa = clean_aa_sequence(row.get("sequence_alignment_aa"))
    if seq_alignment_aa:
        return seq_alignment_aa, "sequence_alignment_aa"

    frcdr_cols = [
        "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa",
        "fwr3_aa", "cdr3_aa", "fwr4_aa",
    ]
    frcdr_parts = [clean_aa_sequence(row.get(col)) for col in frcdr_cols]
    if all(frcdr_parts):
        return "".join(frcdr_parts), "frcdr_concat"

    v_seq_aa = clean_aa_sequence(row.get("v_sequence_alignment_aa"))
    if v_seq_aa:
        return v_seq_aa, "v_sequence_alignment_aa"

    seq_aa = clean_aa_sequence(row.get("sequence_aa"))
    if seq_aa:
        return seq_aa, "sequence_aa"

    raw_seq = clean_aa_sequence(row.get("sequence"))
    if raw_seq:
        return raw_seq, "sequence"

    return "", "missing"


def locate_cdr3_span(
    variable_aa: str,
    cdr3_aa: str,
    drop_ambiguous: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return zero-based [start, end) amino-acid coordinates of cdr3_aa inside variable_aa.

    If not found, return (None, None).
    If found multiple times:
        - return (None, None) if drop_ambiguous is True
        - otherwise return the first match
    """
    if not variable_aa or not cdr3_aa:
        return None, None

    starts = []
    start = variable_aa.find(cdr3_aa)
    while start != -1:
        starts.append(start)
        start = variable_aa.find(cdr3_aa, start + 1)

    if len(starts) == 0:
        return None, None
    if len(starts) > 1 and drop_ambiguous:
        return None, None

    s = starts[0]
    e = s + len(cdr3_aa)
    return s, e


def deterministic_split(key: str, val_percent: int = 10) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def keep_record(
    locus: str,
    seq: str,
    productive: Optional[bool],
    vj_in_frame: Optional[bool],
    stop_codon: Optional[bool],
    v_frameshift: Optional[bool],
    complete_vdj: Optional[bool],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if not seq:
        return False, "empty_sequence"

    if productive is False:
        return False, "non_productive"
    if vj_in_frame is False:
        return False, "out_of_frame"
    if stop_codon is True:
        return False, "stop_codon"
    if v_frameshift is True:
        return False, "v_frameshift"

    if args.require_complete_vdj and complete_vdj is not True:
        return False, "incomplete_vdj"

    if locus == "IGH":
        if not (args.min_heavy <= len(seq) <= args.max_heavy):
            return False, "length_out_of_range"
        return True, "kept"

    if locus in {"IGK", "IGL"}:
        if not (args.min_light <= len(seq) <= args.max_light):
            return False, "length_out_of_range"
        return True, "kept"

    if locus == "VHH":
        if not (args.min_nano <= len(seq) <= args.max_nano):
            return False, "length_out_of_range"
        return True, "kept"

    return False, "bad_locus"


class JsonlGzWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "wt", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self.handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.handle.close()


def iter_input_files(input_dir: Path) -> Iterator[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix in {".gz", ".csv"}:
            yield path


def main() -> None:
    args = parse_args()

    tokenizer = AminoAcidTokenizer()

    input_files = list(iter_input_files(args.input_dir))
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    if not input_files:
        raise FileNotFoundError(f"No .csv or .csv.gz files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "all": JsonlGzWriter(args.output_dir / "oas_all.jsonl.gz"),
        "IGH": JsonlGzWriter(args.output_dir / "oas_igh.jsonl.gz"),
        "IGK": JsonlGzWriter(args.output_dir / "oas_igk.jsonl.gz"),
        "IGL": JsonlGzWriter(args.output_dir / "oas_igl.jsonl.gz"),
        "VHH": JsonlGzWriter(args.output_dir / "oas_vhh.jsonl.gz"),
    }

    seen = set()
    stats = {
        "files_seen": 0,
        "records_seen": 0,
        "records_kept": 0,
        "duplicates_dropped": 0,
        "drop_reasons": Counter(),
        "kept_by_locus": Counter(),
        "kept_by_split": Counter(),
        "sequence_source_counts": Counter(),
        "redundancy_sum_by_locus": Counter(),
    }

    try:
        for path in input_files:
            stats["files_seen"] += 1

            metadata, df = read_oas_table(path)
            basic_meta = extract_basic_metadata(metadata)

            for _, row in df.iterrows():
                stats["records_seen"] += 1

                locus = normalize_locus(row, metadata)
                chain_group = chain_group_from_locus(locus)

                productive = normalize_bool(row.get("productive"))
                vj_in_frame = normalize_bool(row.get("vj_in_frame"))
                stop_codon = normalize_bool(row.get("stop_codon"))
                v_frameshift = normalize_bool(row.get("v_frameshift"))
                complete_vdj = normalize_bool(row.get("complete_vdj"))

                variable_aa, sequence_source = build_variable_aa(row)
                cdr3_aa = clean_aa_sequence(row.get("cdr3_aa"))
                cdr3_start_aa, cdr3_end_aa = locate_cdr3_span(
                    variable_aa,
                    cdr3_aa,
                    drop_ambiguous=args.drop_ambiguous_cdr3_span,
                )

                keep, reason = keep_record(
                    locus=locus,
                    seq=variable_aa,
                    productive=productive,
                    vj_in_frame=vj_in_frame,
                    stop_codon=stop_codon,
                    v_frameshift=v_frameshift,
                    complete_vdj=complete_vdj,
                    args=args,
                )
                if not keep:
                    stats["drop_reasons"][reason] += 1
                    continue

                # dedupe by locus + variable AA, not sequence alone
                dedupe_key = (locus, variable_aa)
                if dedupe_key in seen:
                    stats["duplicates_dropped"] += 1
                    continue
                seen.add(dedupe_key)

                split = deterministic_split(f"{locus}:{variable_aa}", val_percent=args.val_percent)
                token_ids = tokenizer.encode_sequence(
                    sequence=variable_aa,
                    locus=locus,
                    max_length=args.token_max_length,
                )

                record = {
                    "sequence": variable_aa,              # keep compatibility
                    "variable_aa": variable_aa,
                    "token_ids": token_ids,
                    "length": len(variable_aa),
                    "token_length": len(token_ids),

                    "locus": locus,
                    "chain_group": chain_group,
                    "split": split,

                    "productive": productive,
                    "vj_in_frame": vj_in_frame,
                    "stop_codon": stop_codon,
                    "v_frameshift": v_frameshift,
                    "complete_vdj": complete_vdj,

                    "sequence_source": sequence_source,

                    "cdr3_aa": cdr3_aa or None,
                    "cdr3_start_aa": cdr3_start_aa,
                    "cdr3_end_aa": cdr3_end_aa,

                    "v_call": None if pd.isna(row.get("v_call")) else row.get("v_call"),
                    "d_call": None if pd.isna(row.get("d_call")) else row.get("d_call"),
                    "j_call": None if pd.isna(row.get("j_call")) else row.get("j_call"),

                    "redundancy": safe_int(row.get("Redundancy"), default=1),

                    "metadata": basic_meta,
                    "source_file": path.name,
                }

                writers["all"].write(record)
                if locus in writers:
                    writers[locus].write(record)

                stats["records_kept"] += 1
                stats["kept_by_locus"][locus] += 1
                stats["kept_by_split"][split] += 1
                stats["sequence_source_counts"][sequence_source] += 1
                stats["redundancy_sum_by_locus"][locus] += record["redundancy"]

                if args.max_records is not None and stats["records_kept"] >= args.max_records:
                    raise StopIteration

    except StopIteration:
        pass
    finally:
        for writer in writers.values():
            writer.close()

    serializable_stats = {
        "files_seen": stats["files_seen"],
        "records_seen": stats["records_seen"],
        "records_kept": stats["records_kept"],
        "duplicates_dropped": stats["duplicates_dropped"],
        "drop_reasons": dict(stats["drop_reasons"]),
        "kept_by_locus": dict(stats["kept_by_locus"]),
        "kept_by_split": dict(stats["kept_by_split"]),
        "sequence_source_counts": dict(stats["sequence_source_counts"]),
        "redundancy_sum_by_locus": dict(stats["redundancy_sum_by_locus"]),
        "outputs": {
            "all": str(args.output_dir / "oas_all.jsonl.gz"),
            "IGH": str(args.output_dir / "oas_igh.jsonl.gz"),
            "IGK": str(args.output_dir / "oas_igk.jsonl.gz"),
            "IGL": str(args.output_dir / "oas_igl.jsonl.gz"),
            "VHH": str(args.output_dir / "oas_vhh.jsonl.gz"),
        },
    }

    print(json.dumps(serializable_stats, indent=2))

    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, indent=2)


if __name__ == "__main__":
    main()