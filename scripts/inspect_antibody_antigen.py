#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_jsonl(path: Path):
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize_lengths(records: list[dict], field: str) -> dict[str, float]:
    values = [len(str(record.get(field) or "")) for record in records]
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)

    def quantile(q: float) -> float:
        idx = int(round((len(values_sorted) - 1) * q))
        return float(values_sorted[idx])

    return {
        "count": len(values),
        "nonempty_frac": float(sum(v > 0 for v in values) / len(values)),
        "min": float(values_sorted[0]),
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "max": float(values_sorted[-1]),
    }


def print_counter(title: str, counter: Counter, limit: int = 15) -> None:
    print(title)
    for key, value in counter.most_common(limit):
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect processed antibody-antigen JSONL(.gz) outputs."
    )
    p.add_argument("path", type=Path, help="Processed JSONL(.gz) path.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on records read for faster inspection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records: list[dict] = []
    for idx, record in enumerate(iter_jsonl(args.path)):
        records.append(record)
        if args.limit is not None and idx + 1 >= args.limit:
            break

    if not records:
        print("No records found.")
        return

    target_key_counts = Counter(str(r.get("target_key") or "missing") for r in records)
    dataset_counts = Counter(str(r.get("dataset") or "missing") for r in records)
    affinity_type_counts = Counter(str(r.get("affinity_type") or "missing") for r in records)
    split_counts = Counter(str(r.get("split") or "missing") for r in records)
    binder_label_counts = Counter(str(r.get("binder_label")) for r in records if r.get("binder_label") is not None)
    target_name_counts = Counter(str(r.get("target_name") or "missing") for r in records)
    target_pdb_counts = Counter(str(r.get("target_pdb") or "missing") for r in records)
    target_uniprot_counts = Counter(str(r.get("target_uniprot") or "missing") for r in records)
    dataset_affinity_counts = Counter(
        (str(r.get("dataset") or "missing"), str(r.get("affinity_type") or "missing"))
        for r in records
    )

    paired_frac = sum(bool(r.get("is_paired")) for r in records) / len(records)
    nanobody_frac = sum(bool(r.get("is_nanobody")) for r in records) / len(records)
    hcdr3_frac = sum((r.get("cdr3_aa_heavy") is not None) for r in records) / len(records)
    hcdr3_span_frac = sum(
        (r.get("cdr3_start_aa_heavy") is not None and r.get("cdr3_end_aa_heavy") is not None)
        for r in records
    ) / len(records)

    print("=== ANTIBODY-ANTIGEN INSPECTION ===")
    print(f"path: {args.path}")
    print(f"records_read: {len(records)}")
    print(f"unique_target_keys: {len(target_key_counts)}")
    print(f"paired_frac: {paired_frac:.4f}")
    print(f"nanobody_frac: {nanobody_frac:.4f}")
    print(f"heavy_cdr3_present_frac: {hcdr3_frac:.4f}")
    print(f"heavy_cdr3_span_present_frac: {hcdr3_span_frac:.4f}")
    print()

    print_counter("split_counts:", split_counts)
    print_counter("dataset_counts:", dataset_counts)
    print_counter("affinity_type_counts:", affinity_type_counts)
    print_counter("binder_label_counts:", binder_label_counts)
    print_counter("top_target_keys:", target_key_counts, limit=20)
    print_counter("top_target_names:", target_name_counts, limit=20)
    print_counter("top_target_pdb:", target_pdb_counts, limit=20)
    print_counter("top_target_uniprot:", target_uniprot_counts, limit=20)

    print("top_dataset_affinity_type_pairs:")
    for (dataset, affinity_type), value in dataset_affinity_counts.most_common(20):
        print(f"  {dataset} x {affinity_type}: {value}")

    print("length_summary:")
    print(f"  heavy:   {summarize_lengths(records, 'sequence_heavy')}")
    print(f"  light:   {summarize_lengths(records, 'sequence_light')}")
    print(f"  antigen: {summarize_lengths(records, 'sequence_antigen')}")


if __name__ == "__main__":
    main()
