from __future__ import annotations #type hints become more clean

import argparse # allows script to be run from command line
import csv # read delimited tables (oas raw files --> structured tables)
import gzip # read + write compressed files like .gz
import hashlib # deterministic train/validation splitting
import json # parsing metadata + writing output recrods
import re # sequence cleaning
from pathlib import Path #filesystem handling easier
from collections import Counter
from typing import Dict, Iterable, Iterator, Optional, TextIO, Tuple

from tqdm import tqdm # progress bars while processing files

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZOU") 

AA_ONLY = re.compile(r"[^A-Z]")

# alphabet of allowed amino-acid symbols:
# 20 canonical amino acids 
# X = unknown residue
# B, Z = ambiguity codes
# U, 0 = rare amino acids

TRUTHY = {"T", "TRUE", "1", "YES", "Y"}
FALSY = {"F", "FALSE", "0", "NO", "N"}

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
    """
    OAS data will place metadeta in the first line. Parser if possible, 
    else preserves the raw line. Tries two different formats (if line is CSV-quoted single field, will unwrap) with potential
    other fallbacks as necessary

    Args:
        line (str): First line

    Returns:
        Dict[str, object]: Returns Dict in format of {"raw_metadeta": line}
    """
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


def detect_delimiter(path: Path) -> str:
    """
    
    First skips the metadeta line, then sniff the delimiter from a few table lines
    Eg: Guesses the kind of table is (comma-sep, tab-sep, semicolon-sep)

    Args:
        path (Path): Path to OAS file

    Returns:
        str: type of delimiter from [, or \t or ;]
    """
    delimiter_options = "m/t;"
    with open_text(path) as f:
        _ = f.readline() #skip first line, metadata
        sample = "".join(f.readline() for _ in range(5)) # read first 5 lines
    try:
        return csv.Sniffer().sniff(sample, delimiters=delimiter_options).delimiter # infer delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def normalize_bool(value: object) -> Optional[bool]: # cleaning truthy, falsey
    """
    Simply boolean normalizer (data cleaning): 
    1. if it's in the Truthy set defined above, return True
    2. If it's in the Falsey set defined above, return False
    3. If it's not in either one of them, return None

    Args:
        value (object): Any kind of object

    Returns:
        Optional[bool]: Potentially returns True/False, can also return None
    """
    if value is None:
        return None
    value = str(value).strip().upper()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    return None

def safe_int(value: object, default: Optional[int] = None) -> Optional[int]:
    """
    Handles conversion to integers safely with error catching.

    Args:
        value (object): Any string attempted to return
        default (Optional[int], optional): Backup value (null imputation). Defaults to None.

    Returns:
        Optional[int]: Integer representation of the string
    """
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try: 
        return int(text)
    except ValueError:
        return default
    
    
def clean_aa_sequence(seq: str) -> str:
    """
    Normalize amino-acid strings:
    - uppercase
    - remove spaces / punctuation / non-letters
    - keep only accepted AA symbols

    Args:
        seq (str): Sequence

    Returns:
        str: Cleaned sequence (as explained above)
    """
    seq = str(seq or "").upper().replace(" ", "") # remove spaces, error-catching
    seq = AA_ONLY.sub("", seq) # remove anything that is not an uppercase letter
    return "".join(ch for ch in seq if ch in VALID_AA) # list comprehension, joining to valid aa seq

def normalize_locus(raw_locus: object) -> str: 
    """
    Map OAS-style single-letter loci onto more explicit immunoglobulin locus names.
    - Maps H/IGH -> IGH
    - Maps K/IGK -> IGK
    - Maps L/IGL -> IGL
    
    Fallback: returns other

    Args:
        raw_locus (object): _description_

    Returns:
        str: _description_
    """
    locus = str(raw_locus or "").strip().upper()
    if locus in {"H", "IGH"}:
        return "IGH"
    if locus in {"K", "IGK"}:
        return "IGK"
    if locus in {"L", "IGL"}:
        return "IGL"
    return "OTHER"

def chain_group_from_locus(locus: str) -> str:
    """
    Simplification of appropriate locus types to type of chain (Heavy or Light)

    Args:
        locus (str): normalized locus

    Returns:
        str: "heavy" or "light" or "other"
    """
    if locus == "IGH":
        return "heavy"
    if locus in {"IGK", "IGL"}:
        return "light"
    return "other"

def deterministic_split(key: str, val_percent: int = 10) -> str:
    """
    Creates train/val/split in dataset stably. Key should include locus so splitting is chain-aware.

    Args:
        sequence (str): Includes locus
        val_percent (int, optional): Percent of training data that goes to validation. Defaults to 10.

    Returns:
        str: "val" or "train"
    """
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100 #hashing
    return "val" if bucket < val_percent else "train"

def extract_basic_metadeta(metadata: Dict[str, object]) -> Dict[str, object]:
    """
    Normalize the file-level metadeta into a smaller, predicted schema. Renaming, essentially

    Args:
        metadata (Dict[str, object]): Original metadata

    Returns:
        Dict[str, object]: New metadata
    """
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

def build_variable_aa(row: Dict[str, str]) -> Tuple[str, str]:
    """
    We prefer sequence_alignment_aa because it contains the FULL variable-domain AA sequence.
    This is as opposed to v_sequence_alignment_aa because that only uses the V-segment 
    portion and can exclude the full heavy-chain CDR3/FWR4.
    If sequence_alignment_aa is missing, we reconstruct from FR/CDR pieces.

    Args:
        row (Dict[str, str]): _description_

    Returns:
        Tuple[str, str]: (full_variable_domain_aa, source_field_used)
    """
    seq_alignment_aa = clean_aa_sequence(row.get("sequence_alignment_aa"))
    if seq_alignment_aa:
        return seq_alignment_aa, "sequence_alignment_aa"

    parts = [clean_aa_sequence(row.get(col)) for col in VARIABLE_REGION_AA_COLUMNS]
    if all(parts):
        return "".join(parts), "frcdr_concat"

    # Fallback only if needed
    v_only = clean_aa_sequence(row.get("v_sequence_alignment_aa"))
    if v_only:
        return v_only, "v_sequence_alignment_aa"

    return "", "missing"

def extract_region_aas(row: Dict[str, str]) -> Dict[str, str]:
    """
    Annotates and retrieves each region's aa sequence.

    Args:
        row (Dict[str, str]): original 

    Returns:
        Dict[str, str]: labelled, stratified data
    """
    out: Dict[str, str] = {}
    for col in VARIABLE_REGION_AA_COLUMNS:
        out[col] = clean_aa_sequence(row.get(col))
    return out

def choose_nt_sequence(row: Dict[str, str]) -> str:
    """
    Preserve the original nucleotide rearrangement sequence if available.
    """
    candidates = [
        row.get("sequence_alignment"),
        row.get("sequence"),
        row.get("junction"),
    ]
    for cand in candidates:
        text = str(cand or "").upper().replace(" ", "")
        text = re.sub(r"[^ACGTN]", "", text)
        if text:
            return text
    return ""

def keep_record(
    record: Dict[str, object],
    min_heavy: int,
    max_heavy: int,
    min_light: int,
    max_light: int,
    require_complete_vdj: bool = False,
) -> Tuple[bool, str]:
    """
    Returns (keep, reason_if_dropped).
    """
    locus = str(record.get("locus", "OTHER"))
    seq = str(record.get("sequence") or "")

    if not seq:
        return False, "empty_sequence"

    if locus == "IGH":
        if len(seq) < min_heavy or len(seq) > max_heavy:
            return False, "length_out_of_range"
    elif locus in {"IGK", "IGL"}:
        if len(seq) < min_light or len(seq) > max_light:
            return False, "length_out_of_range"
    else:
        return False, "bad_locus"

    if record.get("productive") is False:
        return False, "non_productive"

    if record.get("vj_in_frame") is False:
        return False, "out_of_frame"

    if record.get("stop_codon") is True:
        return False, "stop_codon"

    if record.get("v_frameshift") is True:
        return False, "v_frameshift"

    if require_complete_vdj and record.get("complete_vdj") is not True:
        return False, "incomplete_vdj"

    return True, "kept"

def locate_cdr3_span(sequence: str, cdr3_aa: str) -> tuple[Optional[int], Optional[int]]:
    if not sequence or not cdr3_aa:
        return None, None

    starts = []
    start = sequence.find(cdr3_aa)
    while start != -1:
        starts.append(start)
        start = sequence.find(cdr3_aa, start + 1)

    if len(starts) != 1:
        return None, None

    s = starts[0]
    e = s + len(cdr3_aa)
    return s, e

def iter_oas_records(path: Path) -> Iterator[Dict[str, object]]:
    delimiter = detect_delimiter(path)

    with open_text(path) as f:
        metadata = parse_metadata_line(f.readline())
    basic_meta = extract_basic_metadeta(metadata)
    
    with open_text(path) as f:
        _ = f.readline()  # skip metadata line
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            locus = normalize_locus(row.get("locus"))
            chain_group = chain_group_from_locus(locus)
            variable_aa, sequence_source = build_variable_aa(row)
            region_aas = extract_region_aas(row)
            cdr3_aa = clean_aa_sequence(row.get("cdr3_aa"))
            cdr3_start_aa, cdr3_end_aa = locate_cdr3_span(variable_aa, cdr3_aa)

            record = {
                **basic_meta,
                "source_file": path.name,

                # Core sequence fields
                "sequence": variable_aa,         # keep trainer compatibility
                "variable_aa": variable_aa,      # explicit name for clarity
                "sequence_source": sequence_source,
                "sequence_nt": choose_nt_sequence(row),

                # Chain identity
                "locus": locus,
                "chain_group": chain_group,

                # Productivity / QC flags
                "productive": normalize_bool(row.get("productive")),
                "vj_in_frame": normalize_bool(row.get("vj_in_frame")),
                "v_frameshift": normalize_bool(row.get("v_frameshift")),
                "stop_codon": normalize_bool(row.get("stop_codon")),
                "complete_vdj": normalize_bool(row.get("complete_vdj")),
                "rev_comp": normalize_bool(row.get("rev_comp")),

                # Gene calls
                "v_call": row.get("v_call"),
                "d_call": row.get("d_call"),
                "j_call": row.get("j_call"),

                # Region-level AA annotations
                **region_aas,
                "junction_aa": clean_aa_sequence(row.get("junction_aa")),
                "cdr3_aa": cdr3_aa,
                "cdr3_start_aa": cdr3_start_aa,
                "cdr3_end_aa": cdr3_end_aa,

                # Useful numerical metadata
                "redundancy": safe_int(row.get("Redundancy"), default=1),
                "junction_length": safe_int(row.get("junction_length")),
                "junction_aa_length": safe_int(row.get("junction_aa_length")),

                # Identity metrics if present
                "v_identity": row.get("v_identity"),
                "d_identity": row.get("d_identity"),
                "j_identity": row.get("j_identity"),

                # ANARCI
                "anarci_numbering": row.get("ANARCI_numbering"),
                "anarci_status": row.get("ANARCI_status"),
            }

            yield record

class JsonlGzWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "wt", encoding="utf-8")

    def write(self, record: Dict[str, object]) -> None:
        self.handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.handle.close()
        


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schema-aware OAS preprocessor for variable-domain antibody LM pretraining."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw OAS .csv.gz files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write processed outputs")
    parser.add_argument("--stats-output", type=Path, default=None, help="Optional JSON stats file")
    parser.add_argument("--max-files", type=int, default=None, help="Process only the first N files")
    parser.add_argument("--max-records", type=int, default=None, help="Stop after writing N kept records total")
    parser.add_argument("--val-percent", type=int, default=10, help="Validation percent")
    parser.add_argument("--min-heavy", type=int, default=80, help="Minimum full variable-domain AA length for heavy chains")
    parser.add_argument("--max-heavy", type=int, default=180, help="Maximum full variable-domain AA length for heavy chains")
    parser.add_argument("--min-light", type=int, default=70, help="Minimum full variable-domain AA length for light chains")
    parser.add_argument("--max-light", type=int, default=160, help="Maximum full variable-domain AA length for light chains")
    parser.add_argument("--require-complete-vdj", action="store_true", help="Drop rows not explicitly marked complete_vdj")
    args = parser.parse_args()

    input_files = sorted(
        [p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix in {".gz", ".csv"}]
    )
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    if not input_files:
        raise FileNotFoundError(f"No .csv or .csv.gz files found under {args.input_dir}")

    writers = {
        "all": JsonlGzWriter(args.output_dir / "oas_all.jsonl.gz"),
        "IGH": JsonlGzWriter(args.output_dir / "oas_igh.jsonl.gz"),
        "IGK": JsonlGzWriter(args.output_dir / "oas_igk.jsonl.gz"),
        "IGL": JsonlGzWriter(args.output_dir / "oas_igl.jsonl.gz"),
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

            try:
                for record in iter_oas_records(path):
                    stats["records_seen"] += 1

                    keep, reason = keep_record(
                        record,
                        min_heavy=args.min_heavy,
                        max_heavy=args.max_heavy,
                        min_light=args.min_light,
                        max_light=args.max_light,
                        require_complete_vdj=args.require_complete_vdj,
                    )
                    if not keep:
                        stats["drop_reasons"][reason] += 1
                        continue

                    locus = str(record["locus"])
                    seq = str(record["sequence"])

                    # Dedupe within locus, not globally across all chains.
                    dedupe_key = (locus, seq)
                    if dedupe_key in seen:
                        stats["duplicates_dropped"] += 1
                        continue
                    seen.add(dedupe_key)

                    split_key = f"{locus}:{seq}"
                    split = deterministic_split(split_key, val_percent=args.val_percent)

                    record["length"] = len(seq)
                    record["split"] = split

                    writers["all"].write(record)
                    if locus in writers:
                        writers[locus].write(record)

                    stats["records_kept"] += 1
                    stats["kept_by_locus"][locus] += 1
                    stats["kept_by_split"][split] += 1
                    stats["sequence_source_counts"][str(record["sequence_source"])] += 1
                    stats["redundancy_sum_by_locus"][locus] += int(record.get("redundancy") or 1)

                    if args.max_records is not None and stats["records_kept"] >= args.max_records:
                        raise StopIteration

            except StopIteration:
                break
            except Exception as exc:
                print(f"[WARN] Failed while parsing {path}: {exc}")

    finally:
        for writer in writers.values():
            writer.close()

    # Convert Counters to normal dicts for JSON serialization
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
        },
    }

    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, indent=2)

    print(json.dumps(serializable_stats, indent=2))


if __name__ == "__main__":
    main()