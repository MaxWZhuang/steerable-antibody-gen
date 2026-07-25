#!/usr/bin/env python
"""Census of HCDR3 lengths in a prepared corpus — the input to `length_head_max`.

Why this exists
---------------
The learned length head is categorical over ``1..length_head_max`` classes.
Choosing that bound by eye is how you end up masking out a tail you did not know
was there: rows longer than the bound are silently excluded from the length loss
(never clamped — a clamp would teach the head a wrong answer), so a bound set too
low trains a head that has never seen the lengths it will be asked about, and
nothing in the training logs says so.

This script answers the three questions that decision needs:

1. What does the length distribution actually look like (per split, per
   population)?
2. What fraction of rows would be EXCLUDED at a candidate ``length_head_max``?
3. Is the strong-binder population — the one stage 4 trains on — shaped
   differently from the corpus as a whole?

It is a read-only census. It fits nothing and writes no model artifact.

Usage::

    python scripts/length_census.py --data-path data/processed/antibody_antigen.jsonl.gz
    python scripts/length_census.py --data-path ... --candidate-max 24 32 40
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.data.MLMCollator import OASSequenceDataset  # noqa: E402
from smallAntibodyGen.infill.hcdr3 import HCDR3Span  # noqa: E402

DEFAULT_CANDIDATES = (16, 20, 24, 28, 32, 40)


def collect_lengths(records: Sequence[Any]) -> tuple[Counter, int]:
    """
    Return ``(length_counter, invalid_span_count)`` for one record population.

    A record with no valid span contributes to ``invalid_span_count`` rather than
    being dropped quietly: "how many rows have no usable HCDR3 at all" is part of
    the answer, not noise to be filtered out.
    """
    counts: Counter = Counter()
    invalid = 0
    for record in records:
        try:
            span = HCDR3Span.from_record(record)
        except ValueError:
            invalid += 1
            continue
        counts[span.length] += 1
    return counts, invalid


def percentile(counts: Counter, q: float) -> int | None:
    """The smallest length at or below which at least ``q`` of the mass lies."""
    total = sum(counts.values())
    if total == 0:
        return None
    target = q * total
    cumulative = 0
    for length in sorted(counts):
        cumulative += counts[length]
        if cumulative >= target:
            return length
    return max(counts)


def coverage_at(counts: Counter, cap: int) -> float:
    """Fraction of rows whose length is within ``1..cap``."""
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    covered = sum(n for length, n in counts.items() if 1 <= length <= cap)
    return covered / total


def summarize(name: str, counts: Counter, invalid: int, candidates: Sequence[int]) -> dict:
    total = sum(counts.values())
    summary = {
        "population": name,
        "rows_with_valid_span": total,
        "rows_without_valid_span": invalid,
        "min_length": min(counts) if counts else None,
        "max_length": max(counts) if counts else None,
        "p50": percentile(counts, 0.50),
        "p95": percentile(counts, 0.95),
        "p99": percentile(counts, 0.99),
        "coverage": {str(cap): coverage_at(counts, cap) for cap in candidates},
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n[{summary['population']}]")
    print(
        f"  valid spans: {summary['rows_with_valid_span']}"
        f"  (no valid span: {summary['rows_without_valid_span']})"
    )
    if summary["rows_with_valid_span"] == 0:
        return
    print(
        f"  length min/p50/p95/p99/max: "
        f"{summary['min_length']}/{summary['p50']}/{summary['p95']}/"
        f"{summary['p99']}/{summary['max_length']}"
    )
    for cap, fraction in summary["coverage"].items():
        excluded = (1.0 - fraction) * summary["rows_with_valid_span"]
        print(
            f"  length_head_max={cap:>4}: covers {fraction:6.2%} "
            f"({excluded:.0f} rows would be MASKED OUT of the length loss)"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to census (default: train val).",
    )
    parser.add_argument(
        "--candidate-max",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATES),
        help="Candidate length_head_max values to report coverage for.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the census as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if any(cap <= 0 for cap in args.candidate_max):
        parser.error("--candidate-max values must be > 0")

    summaries: list[dict] = []
    for split in args.splits:
        dataset = OASSequenceDataset(args.data_path, split=split)
        counts, invalid = collect_lengths(dataset.records)
        summaries.append(summarize(f"{split} (all rows)", counts, invalid, args.candidate_max))

        # The strong-binder population is what the HCDR3-infill stage trains on,
        # so its tail is the one that actually constrains length_head_max. Gating
        # on `is_strong_binder` and NOT on `binder_label == 1` is deliberate: the
        # latter is only set for boolean-assay rows and silently drops most
        # strong binders.
        strong = [r for r in dataset.records if getattr(r, "is_strong_binder", False)]
        if strong:
            s_counts, s_invalid = collect_lengths(strong)
            summaries.append(
                summarize(f"{split} (strong binders)", s_counts, s_invalid, args.candidate_max)
            )

    for summary in summaries:
        print_summary(summary)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        print(f"\n[census] wrote {args.output_json}")


if __name__ == "__main__":
    main()
