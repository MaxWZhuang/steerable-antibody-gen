from __future__ import annotations #type hints become more clean

import argparse # allows script to be run from command line
import csv # read delimited tables (oas raw files --> structured tables)
import gzip # read + write compressed files like .gz
import hashlib # deterministic train/validation splitting
import json # parsing metadata + writing output recrods
import re # sequence cleaning
from pathlib import Path #filesystem handling easier
from typing import Dict, Iterable, Iterator, Optional, TextIO

from tqdm import tqdm # progress bars while processing files

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZOU") 
# alphabet of allowed amino-acid symbols:
# 20 canonical amino acids 
# X = unknown residue
# B, Z = ambiguity codes
# U, 0 = rare amino acids

TRUTHY = {"T", "TRUE", "1", "YES", "Y"}
FALSY = {"F", "FALSE", "0", "NO", "N"}
AA_ONLY = re.compile(r"[^A-Z]") # cleans data

VARIABLE_REGION_AA_COLUMNS = [
    "fwr1_aa",
    "cdr1_aa",
    "fwr2_aa",
    "cdr2_aa",
    "fwr3_aa",
    "cdr3_aa",
    "fwr4_aa",
]

def open_text(path: Path) -> TextIO: # convenience wrapper
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def parse_metadata_line(line: str) -> Dict[str, object]:
    line = line.strip() # remove leading/trailing whitespace
    if not line:
        return {} # if empty, return null
    if line.startswith("#"):
        line = line[1:].strip() # remove #
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {"raw_metadata": line}


def detect_delimiter(path: Path) -> str:
    # guesses which one the table is (comma-sep, tab-sep, semicolon-sep)
    with open_text(path) as f:
        _ = f.readline() #skip first line, metadata
        sample = "".join(f.readline() for _ in range(5)) # read first 5 lines
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter # infer delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def normalize_bool(value: object) -> Optional[bool]: # cleaning truthy, falsey
    if value is None:
        return None
    value = str(value).strip().upper()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    return None


def clean_aa_sequence(seq: str) -> str:
    seq = (seq or "").upper().replace(" ", "") # remove spaces, error-catching
    seq = AA_ONLY.sub("", seq) # remove anything that is not an uppercase letter
    return "".join(ch for ch in seq if ch in VALID_AA) # list comprehension, joining to valid aa seq


def choose_aa_sequence(row: Dict[str, str]) -> str:
    candidates = [
        row.get("v_sequence_alignment_aa"),
        row.get("sequence_alignment_aa"),
        row.get("sequence_aa"),
        row.get("sequence"), #trying different candidate names
    ]
    for cand in candidates:
        cleaned = clean_aa_sequence(cand or "")
        if cleaned:
            return cleaned
    return ""


def choose_cdr3_aa(row: Dict[str, str]) -> str:
    candidates = [
        row.get("cdr3_aa"),
        row.get("junction_aa"),
    ]
    for cand in candidates:
        cleaned = clean_aa_sequence(cand or "")
        if cleaned:
            return cleaned
    return ""


def extract_basic_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    return {
        "species": lowered.get("species"),
        "author": lowered.get("author"),
        "subject": lowered.get("subject"),
        "disease": lowered.get("disease"),
        "vaccine": lowered.get("vaccine"),
        "bsource": lowered.get("bsource") or lowered.get("b-cell source"),
        "run": lowered.get("run"),
    }


def deterministic_split(sequence: str, n_val_percent: int = 10) -> str:
    h = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100 #hashing
    return "val" if bucket < n_val_percent else "train"


def iter_oas_records(path: Path) -> Iterator[Dict[str, object]]:
    delimiter = detect_delimiter(path)
    with open_text(path) as f:
        metadata = parse_metadata_line(f.readline())
    basic_meta = extract_basic_metadata(metadata)

    with open_text(path) as f:
        _ = f.readline()
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield {
                **basic_meta,
                "source_file": str(path.name),
                "locus": (row.get("locus") or "").strip().upper(),
                "productive": normalize_bool(row.get("productive")),
                "vj_in_frame": normalize_bool(row.get("vj_in_frame")),
                "stop_codon": normalize_bool(row.get("stop_codon")),
                "v_call": row.get("v_call"),
                "d_call": row.get("d_call"),
                "j_call": row.get("j_call"),
                "cdr3_aa": choose_cdr3_aa(row),
                "sequence": choose_aa_sequence(row),
            }


def keep_record(record: Dict[str, object], min_length: int, max_length: int) -> bool:
    seq = str(record.get("sequence") or "")
    if not seq:
        return False
    if len(seq) < min_length or len(seq) > max_length:
        return False
    if record.get("productive") is False:
        return False
    if record.get("vj_in_frame") is False:
        return False
    if record.get("stop_codon") is True:
        return False
    if record.get("locus").upper not in {"H", "K", "L"}:
        return False
    return True


def write_jsonl_gz(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw OAS data-units into JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--min-length", type=int, default=70)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    input_files = sorted(
        [p for p in args.input_dir.rglob("*") if p.is_file() and (p.suffix == ".gz" or p.suffix == ".csv")]
    )
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    if not input_files:
        raise FileNotFoundError(f"No OAS files found under {args.input_dir}")

    seen = set()
    stats = {
        "files_seen": 0,
        "records_seen": 0,
        "records_kept": 0,
        "duplicates_dropped": 0,
        "too_short_or_long": 0,
        "non_productive": 0,
        "out_of_frame": 0,
        "stop_codon": 0,
        "bad_locus": 0,
        "empty_sequence": 0,
    }

    def filtered_records() -> Iterator[Dict[str, object]]:
        for path in tqdm(input_files, desc="Parsing OAS files"):
            stats["files_seen"] += 1
            try:
                for record in iter_oas_records(path):
                    stats["records_seen"] += 1

                    seq = str(record.get("sequence") or "")
                    if not seq:
                        stats["empty_sequence"] += 1
                        continue

                    if len(seq) < args.min_length or len(seq) > args.max_length:
                        stats["too_short_or_long"] += 1
                        continue

                    if record.get("productive") is False:
                        stats["non_productive"] += 1
                        continue

                    if record.get("vj_in_frame") is False:
                        stats["out_of_frame"] += 1
                        continue

                    if record.get("stop_codon") is True:
                        stats["stop_codon"] += 1
                        continue

                    if record.get("locus").upper not in {"H", "K", "L"}:
                        stats["bad_locus"] += 1
                        continue

                    if seq in seen:
                        stats["duplicates_dropped"] += 1
                        continue
                    seen.add(seq)

                    record["length"] = len(seq)
                    record["split"] = deterministic_split(seq)
                    stats["records_kept"] += 1
                    yield record

                    if args.max_records is not None and stats["records_kept"] >= args.max_records:
                        return
            except Exception as exc:
                print(f"[WARN] Failed to parse {path}: {exc}")

    write_jsonl_gz(filtered_records(), args.output)
    print(f"Wrote processed dataset to {args.output}")

    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats to {args.stats_output}")


if __name__ == "__main__":
    main()