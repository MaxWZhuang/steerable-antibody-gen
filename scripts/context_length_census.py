#!/usr/bin/env python
"""Census of ENCODER CONTEXT length in a prepared corpus — the input to `max_length`.

Why this exists
---------------
``scripts/length_census.py`` measures HCDR3 **amino-acid** length, which is the
input to ``length_head_max``. It never touches a tokenizer, so it cannot answer
"what encoder context do we need". This script answers that one, and it answers
it by running the **production tokenizer** rather than by re-deriving the token
layout for a fifth time.

The layout is currently reconstructed arithmetically in three places besides the
tokenizer (see ``data/lengths.py``), and one of them —
``OASSequenceDataset._load``'s ``token_length`` fallback — is the *only* thing
standing between the training preflight and a corpus that never wrote the field.
``prepare_antibody_antigen.py`` never writes it. So this census encodes every row
for real and reports how far the reconstruction is from the truth.

What it reports
---------------
1. Encoded token-length percentiles per split, per record type
   (``single_heavy``, ``single_light``, ``paired``, ``antibody_antigen``).
2. For each candidate ``max_length``: row overflow, worst overflow, heavy-CDR3
   loss and light-CDR3 loss.
3. The ANTIBODY and ANTIGEN streams separately, because the dual-stream model
   encodes them separately — and because on the ``scratch`` antigen path the
   antigen's effective cap is not ``antigen_max_length`` at all. The collator
   computes it as::

       max_length if antigen_encoder_type == "scratch" else (antigen_max_length or max_length)

   so a scratch run silently caps 2048-aa antigens at the antibody ``max_length``.
   Nothing else in the repository counts how often that truncates.
4. How well the arithmetic reconstructions agree with the tokenizer.

It is a read-only census. It streams the JSONL (it never instantiates
``OASSequenceDataset``, which loads an entire split into memory), fits nothing,
and recommends nothing: choosing the canonical limits is an owner decision that
also needs a GPU-memory probe.

Usage::

    python scripts/context_length_census.py \\
        --data-path data/processed/oas_paired_all/oas_paired.jsonl.gz \\
        --candidate-max-length 192 256 384 512 \\
        --output-json outputs/context-census-paired.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.antigen_tokenization import build_antigen_tokenizer  # noqa: E402
from smallAntibodyGen.data.lengths import (  # noqa: E402
    heavy_cdr3_token_end,
    light_cdr3_token_end,
    paired_token_length,
    single_chain_token_length,
)
from smallAntibodyGen.tokenizer import AminoAcidTokenizer  # noqa: E402

DEFAULT_CANDIDATES = (192, 256, 384, 512)

# Record types, in the order they are reported. Fixed rather than discovered so
# two runs over two corpora produce diffable JSON.
RECORD_TYPES = ("single_heavy", "single_light", "paired", "antibody_antigen", "other")

# Percentiles reported for every length distribution, in order.
PERCENTILES = (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999))

# A cap far above any real protein, used to encode a row at its NATURAL length.
# The tokenizers take a mandatory cap; passing one that can never bind is how we
# ask them "how long would this be if nothing truncated it" without tripping the
# truncation warning on every row.
UNCAPPED_MAX_LENGTH = 1 << 20


# --------------------------------------------------------------------------- #
# Parsing one prepared JSONL row.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LengthRecord:
    """The only fields of a prepared row that bear on context length."""

    split: str
    record_type: str
    heavy: str
    light: str
    antigen: str
    heavy_locus: str | None
    light_locus: str | None
    cdr3_end_aa_heavy: int | None
    cdr3_end_aa_light: int | None
    stored_token_length: int | None


def _text(value: Any) -> str:
    return str(value).upper().strip() if isinstance(value, str) else ""


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def classify_record_type(heavy: str, light: str, antigen: str, row: dict) -> str:
    """
    Which population this row belongs to, for reporting.

    ``antibody_antigen`` wins over ``paired`` because the antigen stream is what
    makes the row's context budget different: it is encoded separately and, on the
    scratch path, shares the antibody's ``max_length``.
    """
    if antigen:
        return "antibody_antigen"
    if heavy and light:
        return "paired"
    chain_group = _text(row.get("chain_group")).lower()
    locus = _text(row.get("locus"))
    if chain_group == "heavy" or locus == "IGH":
        return "single_heavy"
    if chain_group == "light" or locus in {"IGK", "IGL"}:
        return "single_light"
    return "other"


def parse_length_record(row: dict) -> LengthRecord:
    """
    Project one decoded JSONL row onto the fields that determine its token length.

    Mirrors how ``OASSequenceDataset._load`` reads the same row: the heavy chain
    falls back to ``sequence`` when ``sequence_heavy`` is absent (the single-chain
    corpora only have ``sequence``), and the heavy CDR3 falls back to the unsuffixed
    ``cdr3_end_aa`` for rows that are not paired.
    """
    heavy = _text(row.get("sequence_heavy")) or _text(row.get("sequence"))
    light = _text(row.get("sequence_light"))
    antigen = _text(row.get("sequence_antigen"))

    cdr3_end_heavy = _int_or_none(row.get("cdr3_end_aa_heavy"))
    if cdr3_end_heavy is None and not (heavy and light):
        cdr3_end_heavy = _int_or_none(row.get("cdr3_end_aa"))

    return LengthRecord(
        split=str(row.get("split", "")),
        record_type=classify_record_type(heavy, light, antigen, row),
        heavy=heavy,
        light=light,
        antigen=antigen,
        heavy_locus=row.get("heavy_locus") or row.get("locus"),
        light_locus=row.get("light_locus"),
        cdr3_end_aa_heavy=cdr3_end_heavy,
        cdr3_end_aa_light=_int_or_none(row.get("cdr3_end_aa_light")),
        stored_token_length=_int_or_none(row.get("token_length")),
    )


def iter_jsonl(data_path: Path) -> Iterator[dict]:
    """Stream a plain or gzipped JSONL file one decoded row at a time."""
    opener = gzip.open if data_path.suffix == ".gz" else open
    with opener(data_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Encoding one record with the PRODUCTION tokenizer.
# --------------------------------------------------------------------------- #
def encode_antibody_length(record: LengthRecord, tokenizer: AminoAcidTokenizer) -> int:
    """
    Natural (untruncated) antibody-stream token length, from the real tokenizer.

    A record with both chains goes through ``encode_paired_sequences``, exactly as
    ``MLMCollator`` routes it; anything else goes through ``encode_sequence``.
    """
    if record.heavy and record.light:
        return len(
            tokenizer.encode_paired_sequences(
                record.heavy,
                record.light,
                heavy_locus=record.heavy_locus,
                light_locus=record.light_locus,
                max_length=None,
            )
        )
    return len(
        tokenizer.encode_sequence(
            record.heavy,
            locus=record.heavy_locus,
            max_length=None,
        )
    )


def reconstructed_antibody_length(record: LengthRecord) -> int:
    """
    What the arithmetic reconstruction claims the same row's token length is.

    This is ``OASSequenceDataset._load``'s ``token_length`` fallback and
    ``prepare_oas.py``'s written value, both expressed through ``data/lengths.py``.
    Reported next to :func:`encode_antibody_length` so the drift is visible instead
    of assumed absent.
    """
    if record.heavy and record.light:
        return paired_token_length(len(record.heavy), len(record.light))
    return single_chain_token_length(len(record.heavy))


# --------------------------------------------------------------------------- #
# Accumulating a population.
# --------------------------------------------------------------------------- #
@dataclass
class PopulationAccumulator:
    """Streaming counters for one (split, record_type) population."""

    rows: int = 0
    antibody_lengths: Counter = field(default_factory=Counter)
    antigen_lengths: Counter = field(default_factory=Counter)
    rows_with_antigen: int = 0
    # Where each CDR3 ends, as an exclusive TOKEN index. Counted rather than
    # stored so a candidate sweep is O(distinct offsets), not O(rows).
    heavy_cdr3_token_ends: Counter = field(default_factory=Counter)
    # (heavy_residue_count -> light CDR3 token end) already resolved to a token
    # index, so the sweep needs only the index.
    light_cdr3_token_ends: Counter = field(default_factory=Counter)
    rows_with_heavy_cdr3: int = 0
    rows_with_light_cdr3: int = 0
    # Reconstruction-vs-tokenizer agreement.
    rows_with_stored_token_length: int = 0
    stored_mismatches: int = 0
    stored_max_abs_delta: int = 0
    reconstruction_mismatches: int = 0
    reconstruction_max_abs_delta: int = 0

    def add(
        self,
        record: LengthRecord,
        antibody_length: int,
        antigen_length: int | None,
    ) -> None:
        self.rows += 1
        self.antibody_lengths[antibody_length] += 1

        if antigen_length is not None:
            self.rows_with_antigen += 1
            self.antigen_lengths[antigen_length] += 1

        if record.cdr3_end_aa_heavy is not None:
            self.rows_with_heavy_cdr3 += 1
            self.heavy_cdr3_token_ends[heavy_cdr3_token_end(record.cdr3_end_aa_heavy)] += 1

        if record.light and record.cdr3_end_aa_light is not None:
            self.rows_with_light_cdr3 += 1
            self.light_cdr3_token_ends[
                light_cdr3_token_end(len(record.heavy), record.cdr3_end_aa_light)
            ] += 1

        reconstructed = reconstructed_antibody_length(record)
        if reconstructed != antibody_length:
            self.reconstruction_mismatches += 1
            self.reconstruction_max_abs_delta = max(
                self.reconstruction_max_abs_delta, abs(reconstructed - antibody_length)
            )

        if record.stored_token_length is not None:
            self.rows_with_stored_token_length += 1
            if record.stored_token_length != antibody_length:
                self.stored_mismatches += 1
                self.stored_max_abs_delta = max(
                    self.stored_max_abs_delta,
                    abs(record.stored_token_length - antibody_length),
                )


# --------------------------------------------------------------------------- #
# Summarizing.
# --------------------------------------------------------------------------- #
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


def length_stats(counts: Counter) -> dict:
    """Min/mean/max plus the standard percentiles for one length distribution."""
    total = sum(counts.values())
    if total == 0:
        return {"rows": 0}
    stats: dict = {
        "rows": total,
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(length * n for length, n in counts.items()) / total, 2),
    }
    for name, q in PERCENTILES:
        stats[name] = percentile(counts, q)
    return stats


def count_above(counts: Counter, threshold: int) -> int:
    """How many rows have a value strictly greater than ``threshold``."""
    return sum(n for value, n in counts.items() if value > threshold)


def candidate_row(
    accumulator: PopulationAccumulator,
    max_length: int,
) -> dict:
    """
    What one candidate ``max_length`` costs this population.

    ``lost_*_cdr3`` uses the same predicate as the training preflight
    (``summarize_length_truncation``): the CDR3 extends past the window, so at
    least one of its residues is dropped outright.

    ``clipped_*_cdr3`` is the stricter count, and the one the tokenizer's actual
    behavior demands: when a row overflows, the tokenizer truncates to
    ``max_length`` and then OVERWRITES the final surviving token with ``[EOS]``.
    A CDR3 ending exactly at token index ``max_length`` therefore also loses its
    last residue, even though the preflight's ``> max_length`` test says it fits.
    """
    rows = accumulator.rows
    overflow = count_above(accumulator.antibody_lengths, max_length)
    lost_heavy = count_above(accumulator.heavy_cdr3_token_ends, max_length)
    lost_light = count_above(accumulator.light_cdr3_token_ends, max_length)
    worst = max(accumulator.antibody_lengths, default=0) - max_length
    return {
        "max_length": max_length,
        "rows": rows,
        "overflow": overflow,
        "overflow_fraction": (overflow / rows) if rows else None,
        "worst_overflow": max(worst, 0),
        "lost_heavy_cdr3": lost_heavy,
        "lost_heavy_cdr3_fraction": (lost_heavy / rows) if rows else None,
        "lost_light_cdr3": lost_light,
        "lost_light_cdr3_fraction": (lost_light / rows) if rows else None,
        "clipped_heavy_cdr3": count_above(accumulator.heavy_cdr3_token_ends, max_length - 1),
        "clipped_light_cdr3": count_above(accumulator.light_cdr3_token_ends, max_length - 1),
    }


def antigen_candidate_row(accumulator: PopulationAccumulator, max_length: int) -> dict:
    """What one antigen cap costs the antigen stream (no CDR3s in an antigen)."""
    rows = accumulator.rows_with_antigen
    overflow = count_above(accumulator.antigen_lengths, max_length)
    worst = max(accumulator.antigen_lengths, default=0) - max_length
    return {
        "max_length": max_length,
        "rows": rows,
        "overflow": overflow,
        "overflow_fraction": (overflow / rows) if rows else None,
        "worst_overflow": max(worst, 0),
    }


def summarize_population(
    split: str,
    record_type: str,
    accumulator: PopulationAccumulator,
    candidates: Sequence[int],
    antigen_candidates: Sequence[int],
) -> dict:
    summary: dict = {
        "split": split,
        "record_type": record_type,
        "rows": accumulator.rows,
        "rows_with_heavy_cdr3": accumulator.rows_with_heavy_cdr3,
        "rows_with_light_cdr3": accumulator.rows_with_light_cdr3,
        "antibody_stream": length_stats(accumulator.antibody_lengths),
        "antibody_candidates": [candidate_row(accumulator, c) for c in candidates],
        "layout_agreement": {
            "rows_with_stored_token_length": accumulator.rows_with_stored_token_length,
            "stored_vs_tokenizer_mismatches": accumulator.stored_mismatches,
            "stored_vs_tokenizer_max_abs_delta": accumulator.stored_max_abs_delta,
            "reconstruction_vs_tokenizer_mismatches": accumulator.reconstruction_mismatches,
            "reconstruction_vs_tokenizer_max_abs_delta": (
                accumulator.reconstruction_max_abs_delta
            ),
        },
    }
    if accumulator.rows_with_antigen:
        summary["antigen_stream"] = length_stats(accumulator.antigen_lengths)
        summary["antigen_candidates"] = [
            antigen_candidate_row(accumulator, c) for c in antigen_candidates
        ]
    else:
        summary["antigen_stream"] = None
        summary["antigen_candidates"] = None
    return summary


# --------------------------------------------------------------------------- #
# The census.
# --------------------------------------------------------------------------- #
def run_census(
    data_path: Path,
    candidates: Sequence[int],
    antigen_candidates: Sequence[int],
    tokenizer: AminoAcidTokenizer,
    antigen_tokenizer: Any,
    splits: Sequence[str] | None = None,
    limit: int | None = None,
) -> dict:
    """
    Stream ``data_path`` and build the census.

    Deliberately does NOT instantiate ``OASSequenceDataset``: that class appends a
    whole split to ``self.records``, so a full-corpus census through it would hold
    the corpus in memory twice over. Everything here is a ``Counter``.
    """
    accumulators: dict[tuple[str, str], PopulationAccumulator] = {}
    rows_read = 0
    rows_skipped_by_split = 0

    for row in iter_jsonl(data_path):
        if limit is not None and rows_read >= limit:
            break
        record = parse_length_record(row)
        if splits is not None and record.split not in splits:
            rows_skipped_by_split += 1
            continue
        rows_read += 1

        antibody_length = encode_antibody_length(record, tokenizer)
        antigen_length = (
            len(antigen_tokenizer.encode(record.antigen, UNCAPPED_MAX_LENGTH))
            if record.antigen
            else None
        )
        key = (record.split, record.record_type)
        accumulators.setdefault(key, PopulationAccumulator()).add(
            record, antibody_length, antigen_length
        )

    populations = [
        summarize_population(split, record_type, accumulators[(split, record_type)],
                             candidates, antigen_candidates)
        # Sorted split-major, then by the fixed RECORD_TYPES order, so two runs
        # over two corpora produce diffable JSON.
        for split in sorted({s for s, _ in accumulators})
        for record_type in RECORD_TYPES
        if (split, record_type) in accumulators
    ]

    return {
        "data_path": str(data_path),
        "rows_censused": rows_read,
        "rows_skipped_by_split_filter": rows_skipped_by_split,
        "candidate_max_lengths": list(candidates),
        "candidate_antigen_max_lengths": list(antigen_candidates),
        "populations": populations,
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def print_census(census: dict) -> None:
    print(f"\n[census] {census['data_path']}")
    print(f"[census] rows censused: {census['rows_censused']}")

    for population in census["populations"]:
        print(f"\n[{population['split']} / {population['record_type']}]  rows={population['rows']}")
        antibody = population["antibody_stream"]
        if antibody.get("rows"):
            print(
                "  antibody tokens  min/p50/p90/p95/p99/p99.9/max: "
                f"{antibody['min']}/{antibody['p50']}/{antibody['p90']}/{antibody['p95']}/"
                f"{antibody['p99']}/{antibody['p99_9']}/{antibody['max']}"
                f"  (mean {antibody['mean']})"
            )
        for row in population["antibody_candidates"]:
            print(
                f"  max_length={row['max_length']:>5}: "
                f"overflow {row['overflow']:>9} ({row['overflow_fraction']:7.2%}) "
                f"worst +{row['worst_overflow']:<5} "
                f"lost H-CDR3 {row['lost_heavy_cdr3']:>9} ({row['lost_heavy_cdr3_fraction']:7.2%}) "
                f"lost L-CDR3 {row['lost_light_cdr3']:>9} ({row['lost_light_cdr3_fraction']:7.2%})"
            )

        agreement = population["layout_agreement"]
        print(
            "  layout agreement: stored token_length present on "
            f"{agreement['rows_with_stored_token_length']} rows, "
            f"{agreement['stored_vs_tokenizer_mismatches']} disagree with the tokenizer "
            f"(max delta {agreement['stored_vs_tokenizer_max_abs_delta']}); "
            f"arithmetic reconstruction disagrees on "
            f"{agreement['reconstruction_vs_tokenizer_mismatches']} rows "
            f"(max delta {agreement['reconstruction_vs_tokenizer_max_abs_delta']})"
        )

        antigen = population["antigen_stream"]
        if antigen:
            print(
                "  antigen tokens   min/p50/p90/p95/p99/p99.9/max: "
                f"{antigen['min']}/{antigen['p50']}/{antigen['p90']}/{antigen['p95']}/"
                f"{antigen['p99']}/{antigen['p99_9']}/{antigen['max']}"
                f"  (mean {antigen['mean']})"
            )
            print(
                "  NOTE: on antigen_encoder_type='scratch' the collator caps the antigen at "
                "the ANTIBODY max_length and ignores antigen_max_length entirely."
            )
            for row in population["antigen_candidates"]:
                print(
                    f"  antigen cap={row['max_length']:>5}: "
                    f"overflow {row['overflow']:>9} ({row['overflow_fraction']:7.2%}) "
                    f"worst +{row['worst_overflow']}"
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Restrict to these splits (default: every split present in the file).",
    )
    parser.add_argument(
        "--candidate-max-length",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATES),
        help="Candidate encoder max_length values to report truncation for.",
    )
    parser.add_argument(
        "--candidate-antigen-max-length",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Candidate antigen caps (default: the same list as "
            "--candidate-max-length, because the scratch path caps the antigen at "
            "the antibody max_length)."
        ),
    )
    parser.add_argument(
        "--antigen-encoder-type",
        choices=["scratch", "esm"],
        default="scratch",
        help="Which antigen tokenizer to census the antigen stream with.",
    )
    parser.add_argument(
        "--esm-model-name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model id, used only with --antigen-encoder-type esm.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Census at most this many rows (for smoke-testing on a large corpus).",
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
    if any(c <= 0 for c in args.candidate_max_length):
        parser.error("--candidate-max-length values must be > 0")
    antigen_candidates = args.candidate_antigen_max_length or args.candidate_max_length
    if any(c <= 0 for c in antigen_candidates):
        parser.error("--candidate-antigen-max-length values must be > 0")

    tokenizer = AminoAcidTokenizer()
    antigen_tokenizer = build_antigen_tokenizer(
        antigen_encoder_type=args.antigen_encoder_type,
        tokenizer=tokenizer,
        esm_model_name=args.esm_model_name,
    )

    census = run_census(
        data_path=Path(args.data_path),
        candidates=sorted(set(args.candidate_max_length)),
        antigen_candidates=sorted(set(antigen_candidates)),
        tokenizer=tokenizer,
        antigen_tokenizer=antigen_tokenizer,
        splits=args.splits,
        limit=args.limit,
    )
    census["antigen_encoder_type"] = args.antigen_encoder_type

    print_census(census)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(census, indent=2, sort_keys=False), encoding="utf-8")
        print(f"\n[census] wrote {output_path}")


if __name__ == "__main__":
    main()
