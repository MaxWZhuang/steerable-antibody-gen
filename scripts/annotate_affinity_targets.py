#!/usr/bin/env python
"""Attach per-(dataset, affinity_type) strength quantiles to a prepared corpus.

Why quantiles and not raw measurements
--------------------------------------
The antibody-antigen corpus mixes assays that are not on a shared scale: KD in
molar, KD in nanomolar, -log KD, ddG, ELISA ratios. Regressing a single head
against raw values would let one dataset's units dominate the loss and would
make "stronger" mean different things in different rows. Ranking WITHIN a
(dataset, affinity_type) group and storing the rank as a quantile in [0, 1]
makes every group's target comparable while preserving the only thing the assays
actually agree on: ordering.

Conventions that matter
-----------------------
- **Train split only.** The CDF is fitted on ``split == "train"`` rows and then
  applied to every split. Fitting on all rows would leak validation ordering into
  the target a validation metric is computed against.
- **1.0 = strongest.** The score is negated before ranking where a LOWER raw
  value means a stronger binder (raw KD), so the direction is uniform.
- **Mid-ranks for ties.** Tied measurements receive the same quantile; ordinal
  ranking would turn input order into signal.
- **Small groups are excluded.** A group with fewer than ``--min-group-size``
  train rows gets no quantile at all (the field stays absent) rather than a
  quantile estimated from a handful of points.
- **Never in place.** The annotator refuses to overwrite its input; it writes a
  NEW file whose only difference is the added ``affinity_strength_quantile``
  field. Every other key, and the row order, is preserved byte-for-byte.

Usage::

    python scripts/annotate_affinity_targets.py \\
        --input data/processed/antibody_antigen.jsonl.gz \\
        --output data/processed/antibody_antigen_quantiled.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.data import affinity as affinity_rules  # noqa: E402

QUANTILE_FIELD = "affinity_strength_quantile"

# Assay types where a LOWER raw measurement means a STRONGER binder, so the score
# is negated before ranking. -log KD is already "higher is stronger".
LOWER_IS_STRONGER = {"kd"}


def open_maybe_gzip(path: Path, mode: str):
    """Open a plain or gzipped text file based on the suffix."""
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def group_key(record: Dict[str, Any]) -> tuple[str, str]:
    """The population a row's quantile is computed within."""
    dataset = str(record.get("dataset_name") or "unknown")
    affinity_type = affinity_rules.normalize_affinity_type(record.get("affinity_type"))
    return dataset, affinity_type


def orientation_signed_score(record: Dict[str, Any]) -> float | None:
    """
    The scalar a row is ranked by, signed so that LARGER always means stronger.

    Returns ``None`` for rows with no usable graded measurement -- binary rows
    included, because a 0/1 label carries no within-group ordering and giving it
    a quantile would fabricate one.
    """
    affinity_type = affinity_rules.normalize_affinity_type(record.get("affinity_type"))
    if affinity_rules.affinity_family_for_type(affinity_type) != "ranking_regression":
        return None
    value = affinity_rules.finite_float(record.get("processed_measurement_float"))
    if value is None:
        return None
    if affinity_type in LOWER_IS_STRONGER:
        if value <= 0:
            return None
        # Normalize nanomolar-encoded values to molar first, for the same reason
        # base_affinity_strength_score does: otherwise the two encodings of one
        # measurement land in different parts of the same group's ordering.
        if value >= affinity_rules.KD_MOLAR_NANOMOLAR_BOUNDARY:
            value = value * 1e-9
        return -value
    return value


def mid_rank_quantiles(values: Sequence[float]) -> list[float]:
    """
    Map values to [0, 1] by mid-rank, so ties share one quantile.

    A single-element group maps to 1.0 (it is both the weakest and the strongest
    row it has); callers exclude such groups via ``min_group_size`` anyway.
    """
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return [rank / n for rank in ranks]


def fit_train_cdfs(
    records: Sequence[Dict[str, Any]],
    *,
    min_group_size: int,
) -> dict[tuple[str, str], dict[float, float]]:
    """
    Fit one value -> quantile lookup per (dataset, affinity_type) group.

    Only ``split == "train"`` rows contribute. Groups with fewer than
    ``min_group_size`` usable train rows are omitted entirely, which is what
    makes them produce no annotation downstream.
    """
    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if str(record.get("split")) != "train":
            continue
        score = orientation_signed_score(record)
        if score is None:
            continue
        by_group[group_key(record)].append(score)

    lookups: dict[tuple[str, str], dict[float, float]] = {}
    for key, values in by_group.items():
        if len(values) < min_group_size:
            continue
        quantiles = mid_rank_quantiles(values)
        # Tied values share a quantile by construction, so a value -> quantile
        # dict is well defined and lets non-train rows be scored by exact match.
        lookups[key] = {value: q for value, q in zip(values, quantiles)}
    return lookups


def nearest_quantile(lookup: dict[float, float], score: float) -> float:
    """
    Quantile for a score, interpolating for values unseen in the train fit.

    Val/test rows can hold measurements no train row had. Rather than dropping
    them, place them at the quantile of the nearest fitted value; outside the
    fitted range this clamps to 0.0/1.0, which is the honest statement that the
    row is at or beyond an endpoint of the train distribution.
    """
    if score in lookup:
        return lookup[score]
    nearest = min(lookup, key=lambda value: abs(value - score))
    return lookup[nearest]


def annotate(
    input_path: Path,
    output_path: Path,
    *,
    min_group_size: int,
) -> dict[str, int]:
    if input_path.resolve() == output_path.resolve():
        # `resolve()` also normalizes case, which is what catches the Windows
        # attack of passing the same path with different capitalization.
        raise ValueError(
            "refusing to annotate in place; --output must differ from --input "
            "(a partial write would corrupt the only copy of the corpus)"
        )

    records = list(iter_jsonl(input_path))
    lookups = fit_train_cdfs(records, min_group_size=min_group_size)

    stats = {
        "rows_total": len(records),
        "rows_annotated": 0,
        "groups_fitted": len(lookups),
        "groups_seen": len({group_key(r) for r in records}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_maybe_gzip(output_path, "wt") as handle:
        for record in records:
            score = orientation_signed_score(record)
            lookup = lookups.get(group_key(record))
            if score is not None and lookup:
                # Additive only: the field is appended, nothing else is touched,
                # so stripping this key reproduces the input byte-for-byte.
                record[QUANTILE_FIELD] = float(nearest_quantile(lookup, score))
                stats["rows_annotated"] += 1
            handle.write(json.dumps(record) + "\n")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum usable TRAIN rows for a (dataset, affinity_type) group to "
        "be fitted at all; smaller groups get no quantile (default: 20).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.min_group_size < 2:
        parser.error("--min-group-size must be >= 2")

    stats = annotate(
        Path(args.input),
        Path(args.output),
        min_group_size=args.min_group_size,
    )
    print(
        f"[annotate] rows={stats['rows_total']} annotated={stats['rows_annotated']} "
        f"groups_fitted={stats['groups_fitted']}/{stats['groups_seen']}"
    )
    if stats["rows_annotated"] == 0:
        # A run that annotates nothing looks like success but leaves the strength
        # head with no supervision at all; say so rather than exiting quietly.
        print(
            "[warn] no rows were annotated: no (dataset, affinity_type) group had "
            f">= {args.min_group_size} usable graded TRAIN rows.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
