#!/usr/bin/env python
"""Attach order-averaged evidence and a weighted decision score to generated rows.

What this is for
----------------
``scripts/hcdr3_infill.py`` emits candidates carrying ``log_probability`` /
``mean_log_probability`` (single-path scores) and an optional
``compatibility_score``. Those two quantities answer different questions and are
not on a common scale, and the path scores are explicitly NOT poolable across the
``infill`` and ``guided_infill`` samplers.

This script adds the two things needed to rank candidates that came from
different samplers:

1. ``evidence`` (``E-hat``), the order-averaged path log-likelihood from
   ``smallAntibodyGen.infill.evidence`` — sampler-independent by construction,
   with its Monte-Carlo standard error, so a rank difference smaller than the
   error is visibly not a real difference.
2. ``decision_score``, an explicit weighted combination
   ``w_M * compatibility + w_E * (E-hat / L)``.

Two deliberate refusals
-----------------------
- **The weights are yours, not the script's.** There is no default that pretends
  to be calibrated. ``--w-match`` and ``--w-evidence`` must be passed, and both
  are recorded in every output row.
- **Omitted terms are MARKED, never silently dropped.** A row whose
  ``compatibility_score`` is null gets ``decision_score: null`` and
  ``decision_score_omitted: ["compatibility"]`` rather than a score computed from
  the remaining term — a partially-computed score that looks complete is worse
  than no score.

The tension worth watching
--------------------------
Evidence rewards TYPICALITY. The compatibility judge rewards predicted binding.
Ranking by evidence can therefore demote exactly the unusual candidates steering
exists to find. ``--report-demotion`` measures that directly (see
``evidence.demotion_rate_for_record``) instead of leaving it assumed.

Usage::

    python scripts/score_candidates.py \\
        --candidates outputs/candidates.jsonl \\
        --checkpoint checkpoints/mlm_antigen_hcdr3_infill/best.pt \\
        --data-path data/processed/antibody_antigen.jsonl.gz \\
        --split val --num-orders 8 --w-match 1.0 --w-evidence 0.5 \\
        --output outputs/candidates_scored.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from smallAntibodyGen.data.MLMCollator import OASSequenceDataset  # noqa: E402
from smallAntibodyGen.infill import evidence as ev  # noqa: E402

from hcdr3_infill import (  # noqa: E402
    build_infiller,
    build_tokenizer,
    choose_device,
    load_dual_stream_model,
)


def content_seed(record_id: str, candidate: str) -> int:
    """
    Deterministic order-sampling seed derived from CONTENT, not list position.

    Two identical candidates must receive identical orders and therefore tie
    exactly on evidence. Seeding by index would inject Monte-Carlo noise into
    precisely the rank comparisons this script exists to report.
    """
    digest = hashlib.sha1(f"{record_id}\x00{candidate}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**31)


def decision_score(
    *,
    compatibility: float | None,
    evidence_value: float,
    length: int,
    w_match: float,
    w_evidence: float,
) -> tuple[float | None, list[str]]:
    """
    Weighted decision score, with omitted terms named rather than dropped.

    Evidence is divided by ``length`` so the term is per-position and therefore
    comparable across candidates of different HCDR3 lengths -- the same reason
    ``mean_log_probability`` exists alongside ``log_probability``.
    """
    omitted: list[str] = []
    if compatibility is None:
        omitted.append("compatibility")
    if length <= 0:
        omitted.append("evidence")
    if omitted:
        return None, omitted
    return (
        w_match * float(compatibility) + w_evidence * (evidence_value / length),
        [],
    )


def load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--num-orders", type=int, default=8)
    parser.add_argument("--w-match", type=float, required=True)
    parser.add_argument("--w-evidence", type=float, required=True)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--report-demotion",
        type=int,
        default=0,
        help="If > 0, also report the top-k demotion rate between the judge and "
        "evidence rankings per record (0 disables).",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.num_orders <= 0:
        parser.error("--num-orders must be > 0")
    if args.report_demotion < 0:
        parser.error("--report-demotion must be >= 0")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        parser.error(f"--candidates does not exist: {candidates_path}")

    device = choose_device(args.device)
    tokenizer = build_tokenizer()
    model, cfg = load_dual_stream_model(
        Path(args.checkpoint), data_path=args.data_path, device=device
    )
    infiller = build_infiller(model, tokenizer, cfg, device)

    dataset = OASSequenceDataset(args.data_path, split=args.split)
    by_id = {r.record_id: r for r in dataset.records}

    rows = load_candidates(candidates_path)
    scored: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        record = by_id.get(row.get("record_id"))
        candidate = row.get("generated_hcdr3") or ""
        if record is None or not candidate:
            skipped += 1
            print(
                f"[warn] skipping row with record_id={row.get('record_id')!r}: "
                "record not found in split or empty candidate",
                file=sys.stderr,
            )
            continue
        estimate = ev.estimate_evidence(
            infiller,
            record,
            candidate,
            num_orders=args.num_orders,
            seed=content_seed(str(row.get("record_id")), candidate),
        )
        score, omitted = decision_score(
            compatibility=row.get("compatibility_score"),
            evidence_value=estimate.evidence,
            length=estimate.length,
            w_match=args.w_match,
            w_evidence=args.w_evidence,
        )
        enriched = dict(row)
        enriched.update(
            {
                "evidence": estimate.evidence,
                "evidence_se": estimate.evidence_se,
                "evidence_orders": estimate.num_orders,
                "evidence_half_delta": estimate.half_delta,
                "evidence_half_within_2se": estimate.half_within_2se,
                "decision_score": score,
                "decision_score_omitted": omitted,
                "w_match": args.w_match,
                "w_evidence": args.w_evidence,
            }
        )
        scored.append(enriched)

    if not scored:
        raise SystemExit(
            f"no candidate rows were scored ({skipped} skipped); check --candidates "
            "and --split"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wt", encoding="utf-8") as handle:
        for row in scored:
            # allow_nan=False is the backstop: a NaN that escaped the sanitizer
            # would otherwise be written as the non-standard literal `NaN`.
            handle.write(json.dumps(ev.to_json_safe(row), allow_nan=False) + "\n")
    print(f"[score] wrote {len(scored)} rows to {out} ({skipped} skipped)")

    if args.report_demotion > 0:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            grouped[str(row.get("record_id"))].append(row)
        fractions = []
        for record_id, group in grouped.items():
            judge = [r.get("compatibility_score") for r in group]
            if any(value is None for value in judge):
                continue
            fraction, judge_ties, evidence_ties = ev.demotion_rate_for_record(
                [float(v) for v in judge],
                [float(r["evidence"]) for r in group],
                args.report_demotion,
            )
            fractions.append(fraction)
            print(
                f"[demotion] {record_id}: top-{args.report_demotion} demotion="
                f"{fraction:.2%} (judge ties at cutoff={judge_ties}, "
                f"evidence ties={evidence_ties})"
            )
        if fractions:
            mean = sum(fractions) / len(fractions)
            print(
                f"[demotion] mean over {len(fractions)} records: {mean:.2%} "
                "-- evidence rewards typicality, so a high number here is the "
                "measured cost of ranking by it."
            )


if __name__ == "__main__":
    main()
