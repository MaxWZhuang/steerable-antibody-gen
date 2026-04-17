from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent
    def tqdm(iterable, **kwargs):
        return iterable

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZOU")
AA_ONLY = re.compile(r"[^A-Z]")

BOOLEAN_TRUE = {"1", "TRUE", "T", "YES", "Y"}
BOOLEAN_FALSE = {"0", "FALSE", "F", "NO", "N"}


def clean_aa_sequence(seq: object) -> str:
    """
    Normalize amino-acid strings into the compact format used everywhere else
    in this repository.

    This mirrors the OAS preprocessing path:
    - cast missing values safely
    - uppercase
    - remove punctuation / whitespace / non-letters
    - retain only accepted amino-acid symbols

    Args:
        seq:
            Raw sequence-like value from parquet.

    Returns:
        Cleaned amino-acid string. Empty string means unusable / missing.
    """
    seq = str(seq or "").upper().replace(" ", "")
    seq = AA_ONLY.sub("", seq)
    return "".join(ch for ch in seq if ch in VALID_AA)


def clean_text(value: object) -> str:
    """
    Normalize generic metadata text fields without over-interpreting them.

    Args:
        value:
            Raw scalar value from parquet or nested metadata.

    Returns:
        Trimmed string, or empty string when missing.
    """
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_bool(value: object) -> Optional[bool]:
    """
    Conservative boolean parser used for parquet scalar fields.

    Args:
        value:
            Raw scalar.

    Returns:
        True / False when confidently parseable, else None.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in BOOLEAN_TRUE:
        return True
    if text in BOOLEAN_FALSE:
        return False
    return None


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    """
    Parse one numeric field while tolerating noisy strings.

    Args:
        value:
            Raw scalar value.
        default:
            Value returned when parsing fails.

    Returns:
        Parsed float, or default.
    """
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def deterministic_split(key: str, val_percent: int = 10) -> str:
    """
    Deterministically assign a row to train/val.

    The key should reflect the grouping choice we care about. For this dataset,
    we usually want target-aware grouping to reduce leakage, so callers should
    pass a target-derived key rather than a row id when possible.

    Args:
        key:
            Stable grouping string.
        val_percent:
            Fraction of examples routed to validation.

    Returns:
        "train" or "val".
    """
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def locate_cdr3_span(sequence: str, cdr3_aa: str) -> tuple[Optional[int], Optional[int]]:
    """
    Find one unique CDR3 span inside a cleaned variable-domain sequence.

    We intentionally require a unique match. If the substring appears more than
    once, we return None/None rather than silently choosing the wrong span.

    Args:
        sequence:
            Cleaned variable-domain amino-acid sequence.
        cdr3_aa:
            Cleaned CDR3 amino-acid sequence.

    Returns:
        Tuple `(start, end)` using zero-based, end-exclusive indexing.
    """
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


def normalize_confidence(value: object) -> str:
    """
    Normalize confidence into a small stable vocabulary.

    Args:
        value:
            Raw parquet confidence field.

    Returns:
        Lowercase confidence label, or "unknown".
    """
    text = clean_text(value).lower()
    return text if text else "unknown"


def normalize_metadata_dict(value: object) -> Dict[str, object]:
    """
    Ensure the nested parquet metadata field is always a plain dictionary.

    Args:
        value:
            Raw `metadata` parquet field.

    Returns:
        Dictionary suitable for downstream access and JSON serialization.
    """
    if isinstance(value, dict):
        return value
    return {}


def normalize_target_name(text: str) -> str:
    """
    Canonicalize noisy target names for grouping / split assignment.

    Args:
        text:
            Raw target name.

    Returns:
        Lowercased, punctuation-normalized target name.
    """
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def extract_target_fields(metadata: Dict[str, object]) -> Dict[str, str]:
    """
    Pull out the most important target identity fields from nested metadata.

    These fields are central for:
    - leakage-aware split assignment
    - dataset auditing
    - later antigen-family analysis

    Args:
        metadata:
            Normalized nested metadata dict.

    Returns:
        Dictionary with cleaned target descriptors.
    """
    return {
        "target_name": clean_text(metadata.get("target_name")),
        "target_pdb": clean_text(metadata.get("target_pdb")),
        "target_uniprot": clean_text(metadata.get("target_uniprot")),
        "source_url": clean_text(metadata.get("source_url")),
    }


def build_target_key(metadata: Dict[str, object], antigen_sequence: str) -> str:
    """
    Build a stable target grouping key used for leakage-aware splitting.

    Priority:
    1. UniProt when present
    2. PDB code when present
    3. normalized target name when present
    4. fallback hash of antigen sequence

    Args:
        metadata:
            Nested metadata dict.
        antigen_sequence:
            Cleaned antigen sequence.

    Returns:
        Stable target-group key.
    """
    fields = extract_target_fields(metadata)
    if fields["target_uniprot"]:
        return f"uniprot:{fields['target_uniprot'].lower()}"
    if fields["target_pdb"]:
        return f"pdb:{fields['target_pdb'].lower()}"
    normalized_name = normalize_target_name(fields["target_name"])
    if normalized_name:
        return f"name:{normalized_name}"
    antigen_hash = hashlib.sha1(antigen_sequence.encode("utf-8")).hexdigest()[:16]
    return f"antigen_sha1:{antigen_hash}"


def build_chain_features(
    row: Dict[str, object],
    chain_name: str,
) -> Dict[str, object]:
    """
    Derive the modeling-ready representation for one antibody chain.

    The parquet has both full expressed sequences and RIOT-numbered variable
    regions nested inside metadata. For modeling, the variable-domain alignment
    is usually the cleaner choice because it removes signal peptides / tags /
    constant regions when available. We therefore:
    - preserve the cleaned raw input sequence
    - prefer metadata `sequence_alignment_aa` as the modeling sequence
    - carry over CDR annotations and compute CDR3 span coordinates

    Args:
        row:
            One parquet row converted to a dict.
        chain_name:
            `"heavy"` or `"light"`.

    Returns:
        Dictionary of derived chain features.
    """
    metadata = normalize_metadata_dict(row.get("metadata"))
    numbering = metadata.get(f"{chain_name}_riot_numbering")
    if not isinstance(numbering, dict):
        numbering = {}

    raw_sequence = clean_aa_sequence(row.get(f"{chain_name}_sequence"))
    aligned_sequence = clean_aa_sequence(numbering.get("sequence_alignment_aa"))
    model_sequence = aligned_sequence or raw_sequence
    sequence_source = "metadata_sequence_alignment_aa" if aligned_sequence else "raw_sequence"

    cdr1_aa = clean_aa_sequence(numbering.get("cdr1_aa"))
    cdr2_aa = clean_aa_sequence(numbering.get("cdr2_aa"))
    cdr3_aa = clean_aa_sequence(numbering.get("cdr3_aa"))
    cdr3_start_aa, cdr3_end_aa = locate_cdr3_span(model_sequence, cdr3_aa)

    return {
        f"{chain_name}_sequence_raw": raw_sequence,
        f"{chain_name}_variable_aa": model_sequence,
        f"{chain_name}_sequence_source": sequence_source,
        f"cdr1_aa_{chain_name}": cdr1_aa or None,
        f"cdr2_aa_{chain_name}": cdr2_aa or None,
        f"cdr3_aa_{chain_name}": cdr3_aa or None,
        f"cdr3_start_aa_{chain_name}": cdr3_start_aa,
        f"cdr3_end_aa_{chain_name}": cdr3_end_aa,
    }


def parse_binder_label(affinity_type: str, processed_measurement: object) -> Optional[int]:
    """
    Convert explicitly boolean assay rows into a binary binding label.

    We intentionally keep this conservative. Only rows whose `affinity_type`
    says `bool` are converted. Everything else stays unlabeled here and can be
    handled later by task-specific code.

    Args:
        affinity_type:
            Cleaned affinity type.
        processed_measurement:
            Raw processed measurement field.

    Returns:
        1, 0, or None.
    """
    if affinity_type.strip().lower() != "bool":
        return None
    numeric = safe_float(processed_measurement)
    if numeric is None:
        parsed = normalize_bool(processed_measurement)
        if parsed is None:
            return None
        return 1 if parsed else 0
    return 1 if numeric > 0.5 else 0


def keep_record(
    *,
    heavy_variable_aa: str,
    light_variable_aa: str,
    antigen_sequence: str,
    confidence: str,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """
    Centralized row-level filtering policy for parquet preprocessing.

    This function is intentionally narrow and explicit so future experiments can
    relax or tighten individual criteria without rewriting the whole script.

    Args:
        heavy_variable_aa:
            Modeling heavy-chain sequence.
        light_variable_aa:
            Modeling light-chain sequence.
        antigen_sequence:
            Cleaned antigen sequence.
        confidence:
            Normalized confidence label.
        args:
            Parsed CLI args.

    Returns:
        Tuple `(keep, reason)`.
    """
    if not heavy_variable_aa:
        return False, "missing_heavy"
    if len(heavy_variable_aa) < args.min_heavy or len(heavy_variable_aa) > args.max_heavy:
        return False, "heavy_length_out_of_range"

    if light_variable_aa:
        if len(light_variable_aa) < args.min_light or len(light_variable_aa) > args.max_light:
            return False, "light_length_out_of_range"

    if not antigen_sequence:
        return False, "missing_antigen"
    if len(antigen_sequence) < args.min_antigen or len(antigen_sequence) > args.max_antigen:
        return False, "antigen_length_out_of_range"

    allowed_confidence = {item.strip().lower() for item in args.allowed_confidence.split(",") if item.strip()}
    if allowed_confidence and confidence not in allowed_confidence:
        return False, "confidence_filtered"

    return True, "kept"


def build_processed_record(
    row: Dict[str, object],
    shard_name: str,
    row_idx: int,
    args: argparse.Namespace,
) -> tuple[Optional[Dict[str, object]], str]:
    """
    Normalize one parquet row into the processed schema used by later stages.

    Args:
        row:
            One parquet row converted to a dict.
        shard_name:
            Source parquet shard file name.
        row_idx:
            Zero-based row index inside this shard.
        args:
            Parsed CLI args.

    Returns:
        Tuple `(record, reason)`. `record` is None when filtering drops the row.
    """
    metadata = normalize_metadata_dict(row.get("metadata"))
    heavy = build_chain_features(row, "heavy")
    light = build_chain_features(row, "light")

    antigen_sequence = clean_aa_sequence(row.get("antigen_sequence"))
    confidence = normalize_confidence(row.get("confidence"))
    affinity_type = clean_text(row.get("affinity_type"))
    affinity_raw = clean_text(row.get("affinity"))
    processed_measurement_raw = clean_text(row.get("processed_measurement"))
    processed_measurement_float = safe_float(row.get("processed_measurement"))

    keep, reason = keep_record(
        heavy_variable_aa=heavy["heavy_variable_aa"],
        light_variable_aa=light["light_variable_aa"],
        antigen_sequence=antigen_sequence,
        confidence=confidence,
        args=args,
    )
    if not keep:
        return None, reason

    target_fields = extract_target_fields(metadata)
    target_key = build_target_key(metadata, antigen_sequence)
    split = deterministic_split(target_key, val_percent=args.val_percent)

    heavy_variable_aa = str(heavy["heavy_variable_aa"])
    light_variable_aa = str(light["light_variable_aa"])

    return {
        "record_id": f"{shard_name}:{row_idx}",
        "source_file": shard_name,
        "dataset": clean_text(row.get("dataset")),
        "source_url": target_fields["source_url"],
        "target_name": target_fields["target_name"],
        "target_pdb": target_fields["target_pdb"],
        "target_uniprot": target_fields["target_uniprot"],
        "target_key": target_key,
        "split": split,
        "sequence": heavy_variable_aa,
        "locus": "PAIRED_ANTIGEN",
        "chain_group": "paired_antigen",
        "length": len(heavy_variable_aa) + len(light_variable_aa),
        "sequence_heavy": heavy_variable_aa,
        "sequence_light": light_variable_aa or None,
        "sequence_antigen": antigen_sequence,
        "heavy_sequence_raw": heavy["heavy_sequence_raw"] or None,
        "light_sequence_raw": light["light_sequence_raw"] or None,
        "heavy_sequence_source": heavy["heavy_sequence_source"],
        "light_sequence_source": light["light_sequence_source"],
        "heavy_locus": "IGH",
        "light_locus": None,
        "is_paired": bool(heavy_variable_aa and light_variable_aa),
        "is_nanobody": bool(normalize_bool(row.get("nanobody"))),
        "scfv": bool(normalize_bool(row.get("scfv"))),
        "confidence": confidence,
        "affinity_type": affinity_type,
        "affinity_raw": affinity_raw or None,
        "processed_measurement_raw": processed_measurement_raw or None,
        "processed_measurement_float": processed_measurement_float,
        "binder_label": parse_binder_label(affinity_type, row.get("processed_measurement")),
        "cdr1_aa_heavy": heavy["cdr1_aa_heavy"],
        "cdr2_aa_heavy": heavy["cdr2_aa_heavy"],
        "cdr3_aa": heavy["cdr3_aa_heavy"],
        "cdr3_aa_heavy": heavy["cdr3_aa_heavy"],
        "cdr3_start_aa": heavy["cdr3_start_aa_heavy"],
        "cdr3_end_aa": heavy["cdr3_end_aa_heavy"],
        "cdr3_start_aa_heavy": heavy["cdr3_start_aa_heavy"],
        "cdr3_end_aa_heavy": heavy["cdr3_end_aa_heavy"],
        "cdr1_aa_light": light["cdr1_aa_light"],
        "cdr2_aa_light": light["cdr2_aa_light"],
        "cdr3_aa_light": light["cdr3_aa_light"],
        "cdr3_start_aa_light": light["cdr3_start_aa_light"],
        "cdr3_end_aa_light": light["cdr3_end_aa_light"],
        "heavy_variable_aa": heavy_variable_aa,
        "light_variable_aa": light_variable_aa or None,
        "antigen_length": len(antigen_sequence),
        "metadata": metadata,
    }, "kept"


class JsonlGzWriter:
    """
    Minimal gzip JSONL writer matching the repository's existing preprocessing
    pattern.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "wt", encoding="utf-8")

    def write(self, record: Dict[str, object]) -> None:
        self.handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.handle.close()


def iter_parquet_files(input_path: Path) -> Iterator[Path]:
    """
    Yield parquet shards from either a single file path or a directory.

    Args:
        input_path:
            Input file or directory.

    Yields:
        Parquet file paths in sorted order.
    """
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.glob("part-*.parquet")):
        yield path


def write_record(
    record: Dict[str, object],
    writer: JsonlGzWriter,
    seen: set[Tuple[str, str, str]],
    stats: dict,
) -> bool:
    """
    Deduplicate and write one processed record.

    We deduplicate on the exact modeling triple:
    heavy variable region, light variable region, antigen sequence.

    Args:
        record:
            Processed record ready for writing.
        writer:
            Gzip JSONL writer.
        seen:
            Global dedupe set.
        stats:
            Mutable stats dictionary.

    Returns:
        True when written, else False.
    """
    dedupe_key = (
        str(record.get("sequence_heavy") or ""),
        str(record.get("sequence_light") or ""),
        str(record.get("sequence_antigen") or ""),
    )
    if dedupe_key in seen:
        stats["duplicates_dropped"] += 1
        return False

    seen.add(dedupe_key)
    writer.write(record)

    stats["records_kept"] += 1
    stats["kept_by_split"][record["split"]] += 1
    stats["kept_by_dataset"][record["dataset"]] += 1
    stats["kept_by_confidence"][record["confidence"]] += 1
    affinity_type = str(record.get("affinity_type") or "missing")
    stats["kept_by_affinity_type"][affinity_type] += 1
    stats["kept_by_dataset_affinity_type"][(str(record["dataset"]), affinity_type)] += 1
    if record.get("is_paired"):
        stats["paired_records"] += 1
    else:
        stats["heavy_only_records"] += 1
    if record.get("is_nanobody"):
        stats["nanobody_records"] += 1
    if record.get("binder_label") is not None:
        stats["binder_labelable_rows"] += 1
    if record.get("binder_label") == 1:
        stats["binder_positive_records"] += 1
    elif record.get("binder_label") == 0:
        stats["binder_negative_records"] += 1
    if record.get("processed_measurement_float") is not None:
        stats["numeric_processed_measurement_rows"] += 1
    return True


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for parquet preprocessing.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess antibody-antigen parquet shards into a cleaned, "
            "leakage-aware JSONL(.gz) dataset for later antigen-conditioned stages."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "asd-antibody-antigen",
        help="Input parquet directory or one parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "antibody_antigen" / "antibody_antigen.jsonl.gz",
        help="Output processed JSONL(.gz) path.",
    )
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--min-heavy", type=int, default=70)
    parser.add_argument("--max-heavy", type=int, default=180)
    parser.add_argument("--min-light", type=int, default=70)
    parser.add_argument("--max-light", type=int, default=170)
    parser.add_argument("--min-antigen", type=int, default=8)
    parser.add_argument("--max-antigen", type=int, default=2048)
    parser.add_argument(
        "--allowed-confidence",
        type=str,
        default="high,very_high",
        help="Comma-separated confidence labels to keep. Empty string keeps all.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional global cap for quick debugging.",
    )
    return parser.parse_args()


def main() -> None:
    """
    End-to-end parquet preprocessing driver.

    Workflow:
    1. discover parquet shards
    2. read one shard at a time
    3. convert rows into one normalized processed schema
    4. apply conservative filtering
    5. assign deterministic leakage-aware splits
    6. deduplicate
    7. write gzip JSONL
    8. report stats
    """
    args = parse_args()
    if not (0 <= args.val_percent <= 100):
        raise ValueError("--val-percent must be in [0, 100]")

    parquet_files = list(iter_parquet_files(args.input))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {args.input}")

    writer = JsonlGzWriter(args.output)
    seen: set[Tuple[str, str, str]] = set()
    stats = {
        "files_seen": 0,
        "rows_seen": 0,
        "records_kept": 0,
        "duplicates_dropped": 0,
        "paired_records": 0,
        "heavy_only_records": 0,
        "nanobody_records": 0,
        "binder_labelable_rows": 0,
        "binder_positive_records": 0,
        "binder_negative_records": 0,
        "numeric_processed_measurement_rows": 0,
        "drop_reasons": Counter(),
        "kept_by_split": Counter(),
        "kept_by_dataset": Counter(),
        "kept_by_confidence": Counter(),
        "kept_by_affinity_type": Counter(),
        "kept_by_dataset_affinity_type": Counter(),
    }

    try:
        progress = tqdm(parquet_files, desc="parquet_shards")
        for parquet_path in progress:
            stats["files_seen"] += 1
            df = pd.read_parquet(parquet_path)

            for row_idx, row in df.iterrows():
                stats["rows_seen"] += 1
                record, reason = build_processed_record(
                    row=row.to_dict(),
                    shard_name=parquet_path.name,
                    row_idx=int(row_idx),
                    args=args,
                )
                if record is None:
                    stats["drop_reasons"][reason] += 1
                    continue

                write_record(record, writer, seen, stats)

                if args.max_records is not None and stats["records_kept"] >= args.max_records:
                    if hasattr(progress, "close"):
                        progress.close()
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        writer.close()

    print("\n=== PARQUET PREPROCESS SUMMARY ===")
    print(f"input:               {args.input}")
    print(f"output:              {args.output}")
    print(f"files_seen:          {stats['files_seen']}")
    print(f"rows_seen:           {stats['rows_seen']}")
    print(f"records_kept:        {stats['records_kept']}")
    print(f"duplicates_dropped:  {stats['duplicates_dropped']}")
    print(f"paired_records:      {stats['paired_records']}")
    print(f"heavy_only_records:  {stats['heavy_only_records']}")
    print(f"nanobody_records:    {stats['nanobody_records']}")
    print(f"binder_labelable:    {stats['binder_labelable_rows']}")
    print(f"binder_positive:     {stats['binder_positive_records']}")
    print(f"binder_negative:     {stats['binder_negative_records']}")
    print(f"numeric_measurement_rows: {stats['numeric_processed_measurement_rows']}")
    print(f"kept_by_split:       {dict(stats['kept_by_split'])}")
    print(f"kept_by_confidence:  {dict(stats['kept_by_confidence'])}")
    print(f"kept_by_affinity_type: {dict(stats['kept_by_affinity_type'])}")
    print("top_datasets:")
    for key, value in stats["kept_by_dataset"].most_common(10):
        print(f"  {key}: {value}")
    print("top_dataset_affinity_type_pairs:")
    for (dataset, affinity_type), value in stats["kept_by_dataset_affinity_type"].most_common(15):
        print(f"  {dataset} x {affinity_type}: {value}")
    print("drop_reasons:")
    for key, value in stats["drop_reasons"].most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
