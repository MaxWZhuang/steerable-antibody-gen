from __future__ import annotations #type hints become more clean

import argparse # allows script to be run from command line
import csv # read delimited tables (oas raw files --> structured tables)
import gzip # read + write compressed files like .gz
import hashlib # deterministic train/validation splitting
import json # parsing metadata + writing output recrods
import math
import re # sequence cleaning
import sys
from pathlib import Path #filesystem handling easier
from collections import Counter
from typing import Dict, Iterable, Iterator, Optional, TextIO, Tuple
import random # for shuffling files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.data.oas import read_oas_table

import pandas as pd

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

PAIRED_CHAIN_SUFFIXES = ("heavy", "light")

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
    delimiter_options = ",\t;"
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

def extract_basic_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
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


def build_variable_aa_for_suffix(row: Dict[str, str], suffix: str) -> Tuple[str, str]:
    """
    Build one chain's cleaned amino-acid variable domain from a paired OAS row.

    Paired OAS files store heavy and light information in "wide" format with
    suffixed column names such as `sequence_alignment_aa_heavy` and
    `cdr3_aa_light`. This helper mirrors `build_variable_aa()` but resolves the
    chain-specific column names dynamically so paired preprocessing stays
    consistent with the single-chain path.

    Args:
        row:
            Row-like object from pandas exposing `.get(...)`.
        suffix:
            Chain suffix, typically `"heavy"` or `"light"`.

    Returns:
        Tuple `(clean_sequence, source_name)` where `source_name` records which
        input field was used.
    """
    seq_alignment_aa = clean_aa_sequence(row.get(f"sequence_alignment_aa_{suffix}"))
    if seq_alignment_aa:
        return seq_alignment_aa, f"sequence_alignment_aa_{suffix}"

    parts = [clean_aa_sequence(row.get(f"{col}_{suffix}")) for col in VARIABLE_REGION_AA_COLUMNS]
    if all(parts):
        return "".join(parts), f"frcdr_concat_{suffix}"

    v_only = clean_aa_sequence(row.get(f"v_sequence_alignment_aa_{suffix}"))
    if v_only:
        return v_only, f"v_sequence_alignment_aa_{suffix}"

    return "", f"missing_{suffix}"

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


def extract_region_aas_for_suffix(row: Dict[str, str], suffix: str) -> Dict[str, str]:
    """
    Extract cleaned framework/CDR amino-acid segments for one paired chain.

    Args:
        row:
            Row-like object from pandas exposing paired OAS columns.
        suffix:
            Chain suffix, typically `"heavy"` or `"light"`.

    Returns:
        Dictionary keyed by unsuffixed region name for easier downstream use.
    """
    out: Dict[str, str] = {}
    for col in VARIABLE_REGION_AA_COLUMNS:
        out[col] = clean_aa_sequence(row.get(f"{col}_{suffix}"))
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


def choose_nt_sequence_for_suffix(row: Dict[str, str], suffix: str) -> str:
    """
    Choose one chain's nucleotide rearrangement sequence from a paired OAS row.

    Args:
        row:
            Row-like object from pandas exposing paired OAS columns.
        suffix:
            Chain suffix, typically `"heavy"` or `"light"`.

    Returns:
        Cleaned nucleotide string, or an empty string if none is available.
    """
    candidates = [
        row.get(f"sequence_alignment_{suffix}"),
        row.get(f"sequence_{suffix}"),
        row.get(f"junction_{suffix}"),
    ]
    for cand in candidates:
        text = str(cand or "").upper().replace(" ", "")
        text = re.sub(r"[^ACGTN]", "", text)
        if text:
            return text
    return ""


def is_paired_oas_table(metadata: Dict[str, object], df: pd.DataFrame) -> bool:
    """
    Decide whether a raw OAS table uses the wide paired heavy/light schema.

    Args:
        metadata:
            Parsed file-level metadata dictionary.
        df:
            Data table read from the raw OAS file.

    Returns:
        True when the table looks like paired OAS data, otherwise False.
    """
    declared_chain = str(metadata.get("Chain") or metadata.get("chain") or "").strip().lower()
    if declared_chain == "paired":
        return True
    required_columns = {
        "sequence_alignment_aa_heavy",
        "sequence_alignment_aa_light",
        "locus_heavy",
        "locus_light",
    }
    return required_columns.issubset(set(df.columns))


def build_paired_chain_record(
    row: Dict[str, object],
    suffix: str,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, object]], str]:
    """
    Normalize and validate one chain from a paired OAS row.

    This helper centralizes all per-chain extraction for paired data so the
    heavy and light paths stay symmetric and easier to reason about.

    Args:
        row:
            One paired OAS row.
        suffix:
            Chain suffix, typically `"heavy"` or `"light"`.
        args:
            Parsed CLI arguments containing filtering thresholds.

    Returns:
        Tuple `(record, reason)` where `record` is a normalized per-chain
        dictionary when the chain passes filtering, otherwise None, and `reason`
        explains why the chain was dropped.
    """
    raw_locus = row.get(f"locus_{suffix}")
    locus = normalize_locus(raw_locus)

    variable_aa, sequence_source = build_variable_aa_for_suffix(row, suffix)
    cdr3_aa = clean_aa_sequence(row.get(f"cdr3_aa_{suffix}"))
    cdr3_start_aa, cdr3_end_aa = locate_cdr3_span(variable_aa, cdr3_aa)

    productive = normalize_bool(row.get(f"productive_{suffix}"))
    vj_in_frame = normalize_bool(row.get(f"vj_in_frame_{suffix}"))
    stop_codon = normalize_bool(row.get(f"stop_codon_{suffix}"))
    v_frameshift = normalize_bool(row.get(f"v_frameshift_{suffix}"))
    complete_vdj = normalize_bool(row.get(f"complete_vdj_{suffix}"))

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
        return None, reason

    return {
        "sequence": variable_aa,
        "length": len(variable_aa),
        "locus": locus,
        "chain_group": chain_group_from_locus(locus),
        "productive": productive,
        "vj_in_frame": vj_in_frame,
        "stop_codon": stop_codon,
        "v_frameshift": v_frameshift,
        "complete_vdj": complete_vdj,
        "sequence_source": sequence_source,
        "sequence_nt": choose_nt_sequence_for_suffix(row, suffix),
        "cdr3_aa": cdr3_aa or None,
        "cdr3_start_aa": cdr3_start_aa,
        "cdr3_end_aa": cdr3_end_aa,
        "v_call": None if pd.isna(row.get(f"v_call_{suffix}")) else row.get(f"v_call_{suffix}"),
        "d_call": None if pd.isna(row.get(f"d_call_{suffix}")) else row.get(f"d_call_{suffix}"),
        "j_call": None if pd.isna(row.get(f"j_call_{suffix}")) else row.get(f"j_call_{suffix}"),
        "redundancy": safe_int(row.get(f"Redundancy_{suffix}"), default=safe_int(row.get("Redundancy"), default=1)),
        "regions": extract_region_aas_for_suffix(row, suffix),
        "junction_aa": clean_aa_sequence(row.get(f"junction_aa_{suffix}")),
        "junction_length": safe_int(row.get(f"junction_length_{suffix}")),
        "junction_aa_length": safe_int(row.get(f"junction_aa_length_{suffix}")),
        "v_identity": row.get(f"v_identity_{suffix}"),
        "d_identity": row.get(f"d_identity_{suffix}"),
        "j_identity": row.get(f"j_identity_{suffix}"),
        "anarci_numbering": row.get(f"ANARCI_numbering_{suffix}"),
        "anarci_status": row.get(f"ANARCI_status_{suffix}"),
    }, "kept"


def iter_kept_paired_records_for_file(
    path: Path,
    args: argparse.Namespace,
    stats: dict | None = None,
) -> Iterator[Dict[str, object]]:
    """
    Yield native heavy/light paired examples from one paired OAS raw file.

    Each output record represents one cognate heavy/light pairing from the raw
    source. Negatives are intentionally not created here; they are synthesized
    later by the collator so the same stored native pair can participate in many
    different shuffled pairings across epochs.

    Args:
        path:
            Input paired OAS file.
        args:
            Parsed CLI arguments controlling filtering and splitting.
        stats:
            Optional mutable stats dictionary.

    Yields:
        Normalized paired-example dictionaries ready for deduplication/writing.
    """
    if stats is not None:
        stats["files_seen"] += 1

    metadata, df = read_oas_table(path)
    basic_meta = extract_basic_metadata(metadata)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        if stats is not None:
            stats["records_seen"] += 1

        heavy_record, heavy_reason = build_paired_chain_record(row, "heavy", args)
        if heavy_record is None:
            if stats is not None:
                stats["drop_reasons"][f"paired_heavy_{heavy_reason}"] += 1
            continue

        light_record, light_reason = build_paired_chain_record(row, "light", args)
        if light_record is None:
            if stats is not None:
                stats["drop_reasons"][f"paired_light_{light_reason}"] += 1
            continue

        pair_key = (
            f"{heavy_record['locus']}:{heavy_record['sequence']}"
            f"|{light_record['locus']}:{light_record['sequence']}"
        )

        yield {
            "pair_id": f"{path.name}:{row_idx}",
            "is_paired": True,
            "pair_source": "native",
            "sequence": heavy_record["sequence"],
            "variable_aa": heavy_record["sequence"],
            "sequence_heavy": heavy_record["sequence"],
            "sequence_light": light_record["sequence"],
            "heavy_locus": heavy_record["locus"],
            "light_locus": light_record["locus"],
            "length": int(heavy_record["length"]) + int(light_record["length"]),
            "token_length": int(heavy_record["length"]) + int(light_record["length"]) + 5,
            "locus": "PAIRED",
            "chain_group": "paired",
            "split": deterministic_split(pair_key, val_percent=args.val_percent),
            "productive": heavy_record["productive"] and light_record["productive"],
            "vj_in_frame": heavy_record["vj_in_frame"] and light_record["vj_in_frame"],
            "stop_codon": bool(heavy_record["stop_codon"]) or bool(light_record["stop_codon"]),
            "v_frameshift": bool(heavy_record["v_frameshift"]) or bool(light_record["v_frameshift"]),
            "complete_vdj": bool(heavy_record["complete_vdj"]) and bool(light_record["complete_vdj"]),
            "sequence_source": f"{heavy_record['sequence_source']}|{light_record['sequence_source']}",
            "cdr3_aa": heavy_record["cdr3_aa"],
            "cdr3_start_aa": heavy_record["cdr3_start_aa"],
            "cdr3_end_aa": heavy_record["cdr3_end_aa"],
            "cdr3_aa_heavy": heavy_record["cdr3_aa"],
            "cdr3_start_aa_heavy": heavy_record["cdr3_start_aa"],
            "cdr3_end_aa_heavy": heavy_record["cdr3_end_aa"],
            "cdr3_aa_light": light_record["cdr3_aa"],
            "cdr3_start_aa_light": light_record["cdr3_start_aa"],
            "cdr3_end_aa_light": light_record["cdr3_end_aa"],
            "v_call": heavy_record["v_call"],
            "d_call": heavy_record["d_call"],
            "j_call": heavy_record["j_call"],
            "v_call_heavy": heavy_record["v_call"],
            "d_call_heavy": heavy_record["d_call"],
            "j_call_heavy": heavy_record["j_call"],
            "v_call_light": light_record["v_call"],
            "d_call_light": light_record["d_call"],
            "j_call_light": light_record["j_call"],
            "redundancy": max(int(heavy_record["redundancy"] or 1), int(light_record["redundancy"] or 1)),
            "metadata": basic_meta,
            "source_file": path.name,
        }

def keep_record(
    *,
    locus: str,
    seq: str,
    productive: bool | None,
    vj_in_frame: bool | None,
    stop_codon: bool | None,
    v_frameshift: bool | None,
    complete_vdj: bool | None,
    args,
) -> tuple[bool, str]:
    """
    Decide whether one parsed OAS record should be kept.

    Args:
        locus:
            Normalized locus string such as IGH, IGK, IGL, or VHH.
        seq:
            Cleaned amino-acid sequence chosen for modeling.
        productive:
            Whether the rearrangement is marked productive.
        vj_in_frame:
            Whether the rearrangement is in frame.
        stop_codon:
            Whether a stop codon is present.
        v_frameshift:
            Whether a V-region frameshift is present.
        complete_vdj:
            Whether the sequence is explicitly marked as complete_vdj.
        args:
            Parsed CLI arguments containing length thresholds and strictness flags.

    Returns:
        A tuple `(keep, reason)` where:
            - keep is True if the record should be kept
            - reason is a short string explaining why it was kept or dropped
    """
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

    if getattr(args, "require_complete_vdj", False) and complete_vdj is not True:
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
    basic_meta = extract_basic_metadata(metadata)
    
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
        

def iter_kept_records_for_file(
    path: Path,
    args: argparse.Namespace,
    stats: dict = None,
):
    """
    Yield cleaned, kept records from one raw OAS file.

    This helper:
    - parses one file
    - applies the same row-level filtering you already have
    - yields record dicts that are ready to be deduplicated/written

    It does NOT:
    - deduplicate across files
    - write output
    - stop at max_records

    Those remain the responsibility of the outer loop.
    """
    metadata, df = read_oas_table(path)
    if is_paired_oas_table(metadata, df):
        yield from iter_kept_paired_records_for_file(path, args, stats=stats)
        return

    if stats is not None: 
        stats["files_seen"] += 1

    basic_meta = extract_basic_metadata(metadata)

    for _, row in df.iterrows():
        if stats is not None:
            stats["records_seen"] += 1
        
        raw_locus = row.get("locus")
        if raw_locus is None or pd.isna(raw_locus) or str(raw_locus).strip() == "":
            raw_locus = metadata.get("Chain")

        locus = normalize_locus(raw_locus)

        locus = normalize_locus(raw_locus)
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
            cdr3_aa
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
            if stats is not None:
                stats["drop_reasons"][reason] += 1
            continue

        record = {
            "sequence": variable_aa,
            "variable_aa": variable_aa,
            "length": len(variable_aa),
            "locus": locus,
            "chain_group": chain_group,
            "split": deterministic_split(f"{locus}:{variable_aa}", val_percent=args.val_percent),
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

        yield record

def write_record(
    record: dict,
    writers: dict,
    stats: dict,
    seen: set,
) -> bool:
    """
    Deduplicate one record globally, write it if new, and update stats.

    Returns:
        True if the record was written,
        False if it was dropped as a duplicate.
    """
    if record.get("chain_group") == "paired":
        dedupe_key = (
            record.get("heavy_locus"),
            record.get("sequence_heavy"),
            record.get("light_locus"),
            record.get("sequence_light"),
        )
    else:
        dedupe_key = (record["locus"], record["sequence"])
    if dedupe_key in seen:
        stats["duplicates_dropped"] += 1
        return False

    seen.add(dedupe_key)

    writers["all"].write(record)
    if record["locus"] in writers:
        writers[record["locus"]].write(record)

    stats["records_kept"] += 1
    stats["kept_by_locus"][record["locus"]] += 1
    stats["kept_by_split"][record["split"]] += 1
    stats["sequence_source_counts"][record["sequence_source"]] += 1
    stats["redundancy_sum_by_locus"][record["locus"]] += int(record.get("redundancy") or 1)
    stats["kept_by_source_file"][record["source_file"]] += 1

    return True

def stable_seed_from_path(path: Path, base_seed: int = 42) -> int:
    """
    Build deterministic integer seed from a file path. 
    Python's built-in hash() is randomized between interepreter sessions. For reproducible sampling, it's important to have a stable 
    file-dependent seed.

    Args:
        path (Path): Path to one raw OAS file
        base_seed (int, optional): Global run-level seed. Defaults to 42.

    Returns:
        int: Integer seed that is stable for this file.
    """
    
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)

def _sampling_group_for_locus(locus: str) -> str:
    """
    Map a normalized locus to the balancing group used for quota allocation.

    We rebalance unpaired heavy vs light chains while keeping paired records in
    their own group so the same machinery can still process mixed corpora.
    """
    if locus == "IGH":
        return "heavy"
    if locus in {"IGK", "IGL"}:
        return "light"
    if locus == "PAIRED":
        return "paired"
    return "other"


def _allocate_weighted_integer_quotas(
    capacities: Dict[str, int],
    total: int | None,
    alpha: float = 0.0,
) -> Dict[str, int]:
    """
    Allocate integer quotas with a temperature-like power law.

    The target share for each bucket is proportional to:

        capacity ** (1 - alpha)

    so:
        alpha = 0.0 -> preserve natural frequencies
        alpha = 1.0 -> equalize across non-empty buckets
        alpha > 1.0 -> increasingly favor smaller buckets

    Allocation is capacity-aware and redistributes any leftover quota when a
    bucket saturates.
    """
    nonempty = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    if not nonempty:
        return {}

    if total is None:
        return dict(nonempty)

    total = min(int(total), sum(nonempty.values()))
    if total <= 0:
        return {}

    remaining_capacity = dict(nonempty)
    quotas = {key: 0 for key in nonempty}
    remaining_total = total

    while remaining_total > 0 and remaining_capacity:
        weights = {
            key: value ** (1.0 - alpha)
            for key, value in remaining_capacity.items()
            if value > 0
        }
        if not weights:
            break

        weight_sum = sum(weights.values())
        floors: Dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        allocated_this_round = 0

        for key, weight in weights.items():
            ideal = remaining_total * (weight / weight_sum)
            floor_quota = min(remaining_capacity[key], int(math.floor(ideal)))
            floors[key] = floor_quota
            allocated_this_round += floor_quota
            remainders.append((ideal - floor_quota, key))

        for key, amount in floors.items():
            quotas[key] += amount
            remaining_capacity[key] -= amount

        remaining_total -= allocated_this_round
        if remaining_total <= 0:
            break

        progressed = False
        for _, key in sorted(remainders, reverse=True):
            if remaining_total <= 0:
                break
            if remaining_capacity.get(key, 0) <= 0:
                continue
            quotas[key] += 1
            remaining_capacity[key] -= 1
            remaining_total -= 1
            progressed = True

        if progressed:
            remaining_capacity = {key: value for key, value in remaining_capacity.items() if value > 0}
            continue

        # If every remainder tied at zero because the requested total is still
        # larger than the sum of floors, fall back to any remaining capacity.
        for key in sorted(remaining_capacity):
            if remaining_total <= 0:
                break
            spare = remaining_capacity.get(key, 0)
            if spare <= 0:
                continue
            take = min(spare, remaining_total)
            quotas[key] += take
            remaining_capacity[key] -= take
            remaining_total -= take

        remaining_capacity = {key: value for key, value in remaining_capacity.items() if value > 0}

    return {key: value for key, value in quotas.items() if value > 0}


def count_valid_records_per_file_and_locus(input_files, args):
    counts = {}

    for i, path in enumerate(input_files, start=1):
        per_locus = Counter()
        for record in iter_kept_records_for_file(path, args, stats=None):
            per_locus[str(record.get("locus", "OTHER"))] += 1
        counts[path] = dict(per_locus)

        if i % 10 == 0:
            print(f"[count] processed {i}/{len(input_files)} files")

    return counts

def allocate_chain_balanced_quotas(
    counts_by_file_locus: Dict[Path, Dict[str, int]],
    total_records: int,
    chain_balance_alpha: float = 0.0,
) -> tuple[Dict[Path, int], Dict[Path, Dict[str, int]], Dict[str, int]]:
    """
    Allocate quotas per file with optional heavy/light rebalancing.

    Workflow:
        1. Aggregate available counts by balancing group (heavy / light / paired)
        2. Allocate a target quota to each group using the alpha-adjusted power law
        3. Split each group quota into loci
        4. Allocate each locus quota across files proportionally to per-file
           availability for that locus

    Args:
        counts_by_file_locus:
            Mapping file -> mapping of locus -> valid record count.
        total_records:
            Total number of records desired across all files.
        chain_balance_alpha:
            Alpha parameter controlling how strongly large groups are
            down-weighted. 0 preserves the natural distribution, 1 makes the
            heavy/light groups equally weighted when they both have capacity.

    Returns:
        Tuple of:
            - mapping file -> total quota
            - mapping file -> mapping locus -> quota
            - mapping locus -> global target quota
    """
    total_by_locus: Counter[str] = Counter()
    for file_counts in counts_by_file_locus.values():
        total_by_locus.update({locus: int(count) for locus, count in file_counts.items() if int(count) > 0})

    if not total_by_locus:
        return {}, {}, {}

    if total_records is None:
        per_file_locus = {
            path: {locus: int(count) for locus, count in file_counts.items() if int(count) > 0}
            for path, file_counts in counts_by_file_locus.items()
            if sum(file_counts.values()) > 0
        }
        per_file = {path: sum(file_counts.values()) for path, file_counts in per_file_locus.items()}
        return per_file, per_file_locus, dict(total_by_locus)

    capacities_by_group: Counter[str] = Counter()
    loci_by_group: dict[str, list[str]] = {"heavy": [], "light": [], "paired": [], "other": []}
    for locus, count in total_by_locus.items():
        group = _sampling_group_for_locus(locus)
        capacities_by_group[group] += count
        loci_by_group.setdefault(group, []).append(locus)

    target_by_group = _allocate_weighted_integer_quotas(
        capacities=dict(capacities_by_group),
        total=total_records,
        alpha=chain_balance_alpha,
    )

    target_by_locus: dict[str, int] = {}
    for group, group_quota in target_by_group.items():
        locus_capacities = {
            locus: total_by_locus[locus]
            for locus in loci_by_group.get(group, [])
            if total_by_locus[locus] > 0
        }
        locus_alpha = 0.0 if group == "light" else chain_balance_alpha
        target_by_locus.update(
            _allocate_weighted_integer_quotas(
                capacities=locus_capacities,
                total=group_quota,
                alpha=locus_alpha,
            )
        )

    quotas_by_file: Counter[Path] = Counter()
    quotas_by_file_locus: dict[Path, dict[str, int]] = {}
    for locus, locus_quota in target_by_locus.items():
        per_file_capacity = {
            path: int(file_counts.get(locus, 0))
            for path, file_counts in counts_by_file_locus.items()
            if int(file_counts.get(locus, 0)) > 0
        }
        per_file_quota = _allocate_weighted_integer_quotas(
            capacities={str(path): count for path, count in per_file_capacity.items()},
            total=locus_quota,
            alpha=0.0,
        )
        for path in per_file_capacity:
            allocated = int(per_file_quota.get(str(path), 0))
            if allocated <= 0:
                continue
            quotas_by_file[path] += allocated
            quotas_by_file_locus.setdefault(path, {})[locus] = allocated

    return dict(quotas_by_file), quotas_by_file_locus, target_by_locus

def reservoir_sample_file(
    path: Path,
    args,
    quotas_by_locus: Dict[str, int],
    seed: int,
) -> list[dict]:
    """
    Uniformly sample valid records from one file with separate locus reservoirs.

    Why reservoir sampling:
        It lets us sample k records from a stream of unknown size using O(k) memory.

    Args:
        path:
            Raw OAS file.
        args:
            Parsed CLI arguments.
        quotas_by_locus:
            Mapping locus -> number of records to sample from this file.
        seed:
            Deterministic random seed for this file.

    Returns:
        List of sampled record dictionaries. Each locus uses an independent
        reservoir so the per-file heavy/light quota can be enforced while
        preserving O(quota) memory.
    """
    active_loci = {locus: int(quota) for locus, quota in quotas_by_locus.items() if int(quota) > 0}
    if not active_loci:
        return []

    rng = random.Random(seed)
    reservoirs: dict[str, list[dict]] = {locus: [] for locus in active_loci}
    seen_by_locus: Counter[str] = Counter()

    for record in iter_kept_records_for_file(path, args):
        locus = str(record.get("locus", "OTHER"))
        quota = active_loci.get(locus, 0)
        if quota <= 0:
            continue

        seen_by_locus[locus] += 1
        reservoir = reservoirs[locus]

        if len(reservoir) < quota:
            reservoir.append(record)
            continue

        j = rng.randrange(seen_by_locus[locus])
        if j < quota:
            reservoir[j] = record

    sampled_records: list[dict] = []
    for locus in sorted(reservoirs):
        sampled_records.extend(reservoirs[locus])
    rng.shuffle(sampled_records)
    return sampled_records

def sample_with_file_quotas(
    input_files: list[Path],
    args,
    writers: dict,
    stats: dict,
    seen: set,
    base_seed: int = 42,
) -> None:
    """
    Build a balanced subset by assigning a quota to each file and sampling within each file.

    Workflow:
        Pass 1:
            count valid records in each file
        Pass 2:
            allocate equal per-file quotas
        Pass 3:
            reservoir-sample each file independently
        Pass 4:
            globally deduplicate and write sampled records

    Args:
        input_files:
            Raw OAS files selected for this run.
        args:
            Parsed CLI arguments.
        writers:
            Output writers, e.g. all / IGH / IGK / IGL / VHH.
        stats:
            Mutable stats dictionary.
        seen:
            Global dedupe set, typically (locus, sequence).
        base_seed:
            Global seed for deterministic sampling.

    Returns:
        None.
    """
    # --------
    # PASS 1: count valid records in each file and locus
    # --------
    counts_by_file_locus = count_valid_records_per_file_and_locus(input_files, args)
    stats["valid_records_per_file"] = {
        str(path): sum(file_counts.values())
        for path, file_counts in counts_by_file_locus.items()
    }
    stats["valid_records_per_file_locus"] = {
        str(path): dict(file_counts)
        for path, file_counts in counts_by_file_locus.items()
    }

    # --------
    # PASS 2: choose quotas
    # --------
    quotas, quotas_by_file_locus, target_by_locus = allocate_chain_balanced_quotas(
        counts_by_file_locus=counts_by_file_locus,
        total_records=args.max_records,
        chain_balance_alpha=args.chain_balance_alpha,
    )
    stats["allocated_quota_per_file"] = {str(p): q for p, q in quotas.items()}
    stats["allocated_quota_per_file_locus"] = {
        str(path): dict(file_quotas)
        for path, file_quotas in quotas_by_file_locus.items()
    }
    stats["allocated_quota_by_locus"] = dict(target_by_locus)

    # --------
    # PASS 3 + 4: sample each file independently, then write
    # --------
    for path in input_files:
        quota = quotas.get(path, 0)
        if quota <= 0:
            continue

        stats["files_seen"] += 1
        file_seed = stable_seed_from_path(path, base_seed=base_seed)
        quotas_by_locus = quotas_by_file_locus.get(path, counts_by_file_locus.get(path, {}))
        sampled_records = reservoir_sample_file(
            path=path,
            args=args,
            quotas_by_locus=quotas_by_locus,
            seed=file_seed,
        )

        print(f"[sample pass] sampling {quota} from {path.name}")

        for record in sampled_records:
            write_record(record, writers, stats, seen)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schema-aware OAS preprocessor for variable-domain antibody LM pretraining."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw OAS .csv.gz files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write processed outputs")
    parser.add_argument("--stats-output", type=Path, default=None, help="Optional JSON stats file")
    parser.add_argument("--max-files", type=int, default=100, help="Process only the first N files")
    parser.add_argument("--max-records", type=int, default=None, help="Stop after writing N kept records total")
    parser.add_argument("--val-percent", type=int, default=10, help="Validation percent")
    parser.add_argument("--min-heavy", type=int, default=80, help="Minimum full variable-domain AA length for heavy chains")
    parser.add_argument("--max-heavy", type=int, default=180, help="Maximum full variable-domain AA length for heavy chains")
    parser.add_argument("--min-light", type=int, default=70, help="Minimum full variable-domain AA length for light chains")
    parser.add_argument("--max-light", type=int, default=160, help="Maximum full variable-domain AA length for light chains")
    parser.add_argument("--require-complete-vdj", action="store_true", help="Drop rows not explicitly marked complete_vdj")
    parser.add_argument("--file-shuffle-seed", type = int, default = 42, help = "Seed used to randomly shuffle file input order before processing")
    parser.add_argument(
        "--chain-balance-alpha",
        type=float,
        default=0.0,
        help="Soft heavy/light rebalance strength for 'round_robin'. "
             "0.0 keeps the natural chain distribution, 1.0 equalizes heavy vs light when capacity allows.",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["greedy", "round_robin"],
        default="round_robin",
        help="How to sample records across files. "
            "'greedy' fills from file 1, then file 2, etc. "
            "'round_robin' interleaves kept records across files."
    )
    
    args = parser.parse_args()

    if args.chain_balance_alpha < 0:
        raise ValueError("--chain-balance-alpha must be >= 0")

    input_files = sorted(
        [p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix in {".gz", ".csv"}]
    )
    
    rng = random.Random(args.file_shuffle_seed)
    rng.shuffle(input_files)
    
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    if not input_files:
        raise FileNotFoundError(f"No .csv or .csv.gz files found under {args.input_dir}")

    writers = {
        "all": JsonlGzWriter(args.output_dir / "oas_all.jsonl.gz"),
        "IGH": JsonlGzWriter(args.output_dir / "oas_igh.jsonl.gz"),
        "IGK": JsonlGzWriter(args.output_dir / "oas_igk.jsonl.gz"),
        "IGL": JsonlGzWriter(args.output_dir / "oas_igl.jsonl.gz"),
        "PAIRED": JsonlGzWriter(args.output_dir / "oas_paired.jsonl.gz"),
    }
    
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
        "kept_by_source_file": Counter()
    }

    seen = set()

    try:
        if args.sampling_mode == "greedy":
            # Old behavior: fill from file 1, then file 2, etc.
            for path in input_files:
                for record in iter_kept_records_for_file(path, args, stats):
                    write_record(record, writers, stats, seen)

                    if args.max_records is not None and stats["records_kept"] >= args.max_records:
                        raise StopIteration

        elif args.sampling_mode == "round_robin":
            sample_with_file_quotas(
                input_files = input_files,
                args = args,
                writers = writers,
                stats = stats,
                seen = seen,
                base_seed = args.file_shuffle_seed
            )

        else:
            raise ValueError(f"Unknown sampling mode: {args.sampling_mode}")

    except StopIteration:
        pass
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
        "kept_by_source_file": dict(stats["kept_by_source_file"]),
        "outputs": {
            "all": str(args.output_dir / "oas_all.jsonl.gz"),
            "IGH": str(args.output_dir / "oas_igh.jsonl.gz"),
            "IGK": str(args.output_dir / "oas_igk.jsonl.gz"),
            "IGL": str(args.output_dir / "oas_igl.jsonl.gz"),
        }
    }

    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, indent=2)

    print(json.dumps(serializable_stats, indent=2))


if __name__ == "__main__":
    main()
