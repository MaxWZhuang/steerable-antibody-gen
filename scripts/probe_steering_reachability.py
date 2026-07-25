#!/usr/bin/env python
"""Exact per-gamma steering reachability — is guidance doing anything at all?

The question
------------
``guided_infill`` combines two vectors at each masked position::

    guided_logits = unguided_logprobs + gamma * binder_logprobs

If ``binder_logprobs`` is nearly flat across the ~20 canonical residues, then no
usable ``gamma`` changes which residue is drawn, and guided generation is an
expensive re-run of unguided generation. That is not a hypothetical: the
compatibility head reads a CLS-concat readout, which collapses each stream to one
position and is therefore only weakly sensitive to a single-residue substitution.

The instrument
--------------
Both vectors are functions of the state alone, NOT of gamma. So for one position
in one held context, TWO forwards (one MLM forward + one batched ~20-row
enumeration) are enough to compute the guided distribution for EVERY gamma on a
grid, exactly, in float64. No extra model calls per gamma.

Per gamma it reports:

- ``flip_fraction``  — how often the argmax residue differs from gamma = 0,
- ``total_variation`` — half the L1 distance between the guided and unguided
  distributions,
- ``delta_p_target``  — change in probability mass on the residue the unguided
  model would have chosen,
- ``binder_spread``   — max-min of ``binder_logprobs``, the ceiling on what any
  gamma can do at this position.

The caveat that keeps this honest
---------------------------------
The two cached vectors are exact only for a FIXED context. In a real
``guided_infill`` run the residues committed at earlier steps depend on gamma, so
from the second position onward the context itself is gamma-dependent and the
cached vectors do NOT extend. This is therefore precisely a
one-position/held-context instrument. It answers "can gamma move this decision",
not "what does a full gamma sweep generate" — which is the right first question,
because if the answer here is no, a full sweep cannot rescue it.

Usage::

    python scripts/probe_steering_reachability.py \\
        --checkpoint checkpoints/mlm_antigen_hcdr3_infill/best.pt \\
        --data-path data/processed/antibody_antigen.jsonl.gz \\
        --split val --num-records 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from smallAntibodyGen.data.MLMCollator import OASSequenceDataset  # noqa: E402
from smallAntibodyGen.infill.hcdr3 import HCDR3Span  # noqa: E402

from hcdr3_infill import (  # noqa: E402
    build_infiller,
    build_tokenizer,
    choose_device,
    load_dual_stream_model,
    select_records,
)

DEFAULT_GAMMAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def reachability_at_position(
    infiller: Any,
    record: Any,
    *,
    position_index: int,
    gammas: Sequence[float],
    guidance_target: int = 1,
) -> dict[str, Any]:
    """
    Exact per-gamma reachability at ONE masked position of a fully-masked span.

    Runs exactly two forwards, then does float64 arithmetic per gamma.

    Args:
        infiller: a ``FixedLengthHCDR3Infiller``.
        record: the antibody-antigen record.
        position_index: which HCDR3 position (0-based) to probe.
        gammas: the guidance strengths to evaluate.
        guidance_target: compatibility class to steer toward (1 = binder).
    """
    span = HCDR3Span.from_record(record)
    base_ids, base_attn, mask_positions, _, _ = (
        infiller._encode_antibody_with_masked_hcdr3(
            record, span, proposed_length=span.length
        )
    )
    if not (0 <= position_index < len(mask_positions)):
        raise ValueError(
            f"position_index {position_index} out of range for a "
            f"{len(mask_positions)}-residue span"
        )
    position = mask_positions[position_index]
    antigen_ids, antigen_attn = infiller._encode_antigen(record)
    infiller.model.eval()

    with torch.no_grad():
        mlm_logits, _ = infiller.model(
            antibody_input_ids=base_ids,
            antibody_attention_mask=base_attn,
            antigen_input_ids=antigen_ids,
            antigen_attention_mask=antigen_attn,
        )
        canonical_logits = mlm_logits[0, position][infiller.canonical_token_ids]
        unguided = torch.log_softmax(canonical_logits.double(), dim=-1)
        binder = infiller._binder_logprobs_by_candidate(
            base_ids,
            base_attn,
            antigen_ids,
            antigen_attn,
            position,
            guidance_target=guidance_target,
        ).double()

    unguided_probs = unguided.exp()
    base_argmax = int(unguided.argmax().item())
    base_p_target = float(unguided_probs[base_argmax].item())

    rows = []
    for gamma in gammas:
        guided = unguided + float(gamma) * binder
        guided_probs = torch.softmax(guided, dim=-1)
        rows.append(
            {
                "gamma": float(gamma),
                "argmax_residue": infiller.tokenizer.id_to_token[
                    infiller.canonical_token_ids[int(guided.argmax().item())]
                ],
                "flipped": int(guided.argmax().item()) != base_argmax,
                "total_variation": float(
                    0.5 * (guided_probs - unguided_probs).abs().sum().item()
                ),
                "delta_p_target": float(
                    guided_probs[base_argmax].item() - base_p_target
                ),
            }
        )

    return {
        "record_id": getattr(record, "record_id", None),
        "target_key": getattr(record, "target_key", None),
        "split": getattr(record, "split", None),
        "hcdr3_length": span.length,
        "position_index": position_index,
        "unguided_argmax_residue": infiller.tokenizer.id_to_token[
            infiller.canonical_token_ids[base_argmax]
        ],
        # The ceiling on what ANY gamma can do here: a flat binder term means no
        # gamma changes the ranking, only the temperature.
        "binder_spread": float((binder.max() - binder.min()).item()),
        "binder_std": float(binder.std(unbiased=False).item()),
        "gammas": rows,
    }


def summarize(results: Sequence[dict[str, Any]], gammas: Sequence[float]) -> dict[str, Any]:
    """Aggregate per-position results into one flip-fraction curve."""
    summary: dict[str, Any] = {
        "positions_probed": len(results),
        "binder_spread_median": (
            statistics.median(r["binder_spread"] for r in results) if results else float("nan")
        ),
        "binder_spread_max": max((r["binder_spread"] for r in results), default=float("nan")),
        "curve": [],
    }
    for i, gamma in enumerate(gammas):
        flips = [r["gammas"][i]["flipped"] for r in results]
        tvs = [r["gammas"][i]["total_variation"] for r in results]
        summary["curve"].append(
            {
                "gamma": float(gamma),
                "flip_fraction": (sum(flips) / len(flips)) if flips else float("nan"),
                "total_variation_median": (
                    statistics.median(tvs) if tvs else float("nan")
                ),
            }
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--num-records", type=int, default=20)
    parser.add_argument(
        "--positions-per-record",
        type=int,
        default=1,
        help="How many HCDR3 positions to probe per record, taken from the "
        "start of the span (default 1).",
    )
    parser.add_argument("--guidance-target", type=int, default=1)
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=list(DEFAULT_GAMMAS),
        help="Guidance strengths to evaluate (all computed from the same two forwards).",
    )
    parser.add_argument(
        "--guidance-checkpoint",
        type=str,
        default=None,
        help="Optional external classifier supplying the binder term, matching "
        "hcdr3_infill.py's --guidance-checkpoint.",
    )
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.num_records <= 0:
        parser.error("--num-records must be > 0")
    if args.positions_per_record <= 0:
        parser.error("--positions-per-record must be > 0")
    if any(g < 0 for g in args.gammas):
        parser.error("--gammas must be >= 0")

    device = choose_device(args.device)
    tokenizer = build_tokenizer()
    model, cfg = load_dual_stream_model(
        Path(args.checkpoint), data_path=args.data_path, device=device
    )
    guidance_model = None
    guidance_cfg = None
    if args.guidance_checkpoint:
        guidance_model, guidance_cfg = load_dual_stream_model(
            Path(args.guidance_checkpoint), data_path=args.data_path, device=device
        )
    infiller = build_infiller(
        model,
        tokenizer,
        cfg,
        device,
        guidance_model=guidance_model,
        guidance_cfg=guidance_cfg,
    )

    dataset = OASSequenceDataset(args.data_path, split=args.split)
    records = select_records(dataset, record_id=None, num_records=args.num_records)

    results: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        span = HCDR3Span.from_record(record)
        for index in range(min(args.positions_per_record, span.length)):
            try:
                results.append(
                    reachability_at_position(
                        infiller,
                        record,
                        position_index=index,
                        gammas=args.gammas,
                        guidance_target=args.guidance_target,
                    )
                )
            except ValueError as exc:
                skipped += 1
                print(f"[warn] skipped {record.record_id}[{index}]: {exc}", file=sys.stderr)

    if not results:
        # A probe that measures nothing must not exit 0 looking like a null result.
        raise SystemExit(
            f"probe produced no measurements ({skipped} skipped); check --data-path "
            "and --split"
        )

    summary = summarize(results, args.gammas)
    print("[reachability] binder-term spread (the ceiling on any gamma):")
    print(
        f"  median={summary['binder_spread_median']:.4f}  "
        f"max={summary['binder_spread_max']:.4f}  over {summary['positions_probed']} positions"
    )
    print("[reachability] gamma -> flip fraction / median total variation:")
    for point in summary["curve"]:
        print(
            f"  gamma={point['gamma']:>6.2f}  flip={point['flip_fraction']:6.2%}  "
            f"TV={point['total_variation_median']:.4f}"
        )
    print(
        "\n[caveat] Exact only for a FIXED context: in a real guided run the "
        "residues committed at earlier steps are themselves gamma-dependent, so "
        "these numbers describe one held-context decision, not a full sweep."
    )

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "guidance_checkpoint": args.guidance_checkpoint,
            "split": args.split,
            "gammas": list(args.gammas),
            "summary": summary,
            "positions": results,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[reachability] wrote {out}")


if __name__ == "__main__":
    main()
