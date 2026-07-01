#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.data.MLMCollator import OASSequenceDataset
from smallAntibodyGen.infill.hcdr3 import (
    AntigenCompatibilityScorer,
    EmpiricalHCDR3LengthPrior,
    FixedLengthHCDR3Infiller,
    HCDR3Span,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention
from mlm_train import TrainConfig, _train_config_defaults, build_model, build_tokenizer, choose_device


DEFAULT_SCORE_CHECKPOINT = Path("checkpoints/mlm_antigen_real_label_refine/best.pt")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for antigen-conditioned HCDR3 generation.

    The CLI works with existing processed antibody-antigen JSONL files. A
    target record supplies the framework, optional light chain, antigen, and
    HCDR3 insertion boundaries. ``--length-mode fixed`` uses the record's known
    HCDR3 length. ``--length-mode empirical`` samples lengths from positive
    binder HCDR3 lengths in the training split and then asks the same fixed-
    length infiller to generate residues for each proposed length.

    ``--guidance-strength`` opts into ProteinGuide-style antigen-binder
    guidance. When it is ``0`` (default) the CLI uses the original single-pass
    ``infill`` sampler and output is unchanged. When it is ``> 0`` the CLI
    switches to the iterative ``guided_infill`` sampler, which steers each
    residue toward the binder class using the *generation model's own*
    compatibility head. This is distinct from ``--score-checkpoint``, which only
    attaches a post-hoc compatibility score for reporting and never influences
    sampling.
    """
    parser = argparse.ArgumentParser(description="Generate antigen-conditioned HCDR3 infill candidates.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--record-id", type=str)
    parser.add_argument("--num-records", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-mode", choices=("fixed", "empirical"), default="fixed")
    # ProteinGuide-style guidance (opt-in; 0.0 keeps the original infill path).
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=0.0,
        help="Binder-guidance factor gamma. 0 disables guidance (uses single-pass infill); "
        ">0 enables iterative guided_infill steering toward the binder class.",
    )
    parser.add_argument(
        "--guidance-order",
        choices=("confidence", "random", "left_to_right"),
        default="confidence",
        help="Unmasking order for guided decoding (only used when --guidance-strength > 0).",
    )
    parser.add_argument(
        "--guidance-target",
        type=int,
        default=1,
        choices=(0, 1),
        help="Compatibility-head class index to steer toward (1 = binder).",
    )
    parser.add_argument("--score-checkpoint", type=str, default=None)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def config_from_checkpoint(checkpoint: dict[str, Any], *, data_path: str, device: str | None = None) -> TrainConfig:
    """
    Reconstruct a TrainConfig from a saved checkpoint.

    Older checkpoints predate the HCDR3 infilling fields, so this helper starts
    from current dataclass defaults and overlays the checkpoint's saved
    ``train_config`` dictionary. ``data_path`` and ``device`` are supplied by
    the generation CLI because the checkpoint may have been trained on a
    different machine or with a different runtime device.
    """
    merged = _train_config_defaults()
    saved = checkpoint.get("train_config")
    if isinstance(saved, dict):
        # Only overlay keys that are still TrainConfig fields, so a checkpoint
        # saved under an older/newer schema (a renamed or removed field) cannot
        # crash reconstruction with an unexpected keyword argument.
        valid_fields = {f.name for f in fields(TrainConfig)}
        merged.update({k: v for k, v in saved.items() if k in valid_fields})
    merged["data_path"] = data_path
    if device is not None:
        merged["device"] = device
    cfg = TrainConfig(**merged)
    cfg.validate()
    return cfg


def load_dual_stream_model(
    checkpoint_path: Path,
    *,
    data_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, TrainConfig]:
    """
    Load an antibody-antigen checkpoint for infilling or scoring.

    The model architecture is reconstructed from the checkpoint's training
    config, then the checkpoint weights are loaded directly. The function
    expects a dual-stream checkpoint because HCDR3 infilling here is
    antigen-conditioned.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_checkpoint(checkpoint, data_path=data_path, device=str(device))
    tokenizer = build_tokenizer()
    model = build_model(tokenizer, cfg, device)
    if not isinstance(model, AntibodyAntigenCrossAttention):
        raise ValueError(
            f"checkpoint {checkpoint_path} reconstructs training_stage = {cfg.training_stage!r},\n which builds {type(model).__name__};\n HCDR3" 
            f"infilling needs a dual-stream AntibodyAntigenCrossAttention checkpoint. "
        )
    # strict=True so a renamed/resized/mismatched checkpoint fails loudly instead
    # of silently leaving submodules at random init and generating garbage.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def select_records(dataset: OASSequenceDataset, *, record_id: str | None, num_records: int) -> list[Any]:
    """
    Select target records from one split for candidate generation.

    Records without valid HCDR3 spans are skipped because the current unknown-
    length infrastructure still needs numbered framework boundaries: it hides
    the length of the replacement loop, not the location of the loop.
    """
    selected: list[Any] = []
    for record in dataset.records:
        if record_id is not None and record.record_id != record_id:
            continue
        try:
            HCDR3Span.from_record(record)
        except ValueError:
            continue
        selected.append(record)
        if record_id is not None or len(selected) >= num_records:
            break
    if record_id is not None and not selected:
        raise ValueError(f"record_id not found with a valid HCDR3 span: {record_id}")
    return selected


def candidate_to_json(
    *,
    record: Any,
    true_span: HCDR3Span,
    length_mode: str,
    candidate: Any,
    guidance_strength: float = 0.0,
    guidance_order: str | None = None,
) -> dict[str, Any]:
    """
    Convert one generated candidate into a JSON-serializable output row.

    ``guidance_strength`` / ``guidance_order`` are recorded for provenance so a
    downstream consumer can tell guided candidates from unguided ones and knows
    which schedule produced them. ``guidance_order`` is reported only when
    guidance was actually active (``guidance_strength > 0``).
    """
    return {
        "record_id": record.record_id,
        "target_key": record.target_key,
        "target_name": record.target_name,
        "split": record.split,
        "length_mode": length_mode,
        "true_hcdr3": true_span.original_hcdr3,
        "true_hcdr3_length": true_span.length,
        "proposed_hcdr3_length": candidate.length,
        "generated_hcdr3": candidate.generated_hcdr3,
        "generated_heavy_sequence": candidate.heavy_sequence,
        "log_probability": candidate.log_probability,
        "mean_log_probability": candidate.mean_log_probability,
        "compatibility_score": candidate.compatibility_score,
        "guidance_strength": guidance_strength,
        "guidance_order": guidance_order if guidance_strength > 0 else None,
    }


def write_jsonl(rows: Sequence[dict[str, Any]], output_path: str | None) -> None:
    """
    Emit generated candidates as JSONL.

    When ``output_path`` is omitted, rows are printed to stdout so the command
    can be used in shell pipelines. When a path is provided, parent directories
    are created and rows are written there.
    """
    if output_path is None:
        for row in rows:
            print(json.dumps(row))
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.num_records <= 0:
        parser.error("--num-records must be > 0")
    if args.num_samples <= 0:
        parser.error("--num-samples must be > 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k is not None and args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 or omitted disables top-k filtering)")
    if args.guidance_strength < 0:
        parser.error("--guidance-strength must be >= 0 (0 disables guidance)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = choose_device(args.device)
    tokenizer = build_tokenizer()
    model, cfg = load_dual_stream_model(Path(args.checkpoint), data_path=args.data_path, device=device)
    infiller = FixedLengthHCDR3Infiller(
        model,
        tokenizer,
        max_length=cfg.max_length,
        device=device,
    )

    scorer = None
    score_checkpoint_arg = args.score_checkpoint
    score_checkpoint = Path(score_checkpoint_arg) if score_checkpoint_arg else DEFAULT_SCORE_CHECKPOINT
    if not args.no_score:
        # An explicitly-provided --score-checkpoint that is missing is a user
        # error: fail loudly rather than silently emitting null scores. Only the
        # implicit DEFAULT_SCORE_CHECKPOINT is allowed to be absent (skip scoring).
        if score_checkpoint_arg is not None and not score_checkpoint.exists():
            parser.error(f"--score-checkpoint path does not exist: {score_checkpoint}")
        if score_checkpoint.exists():
            score_model, score_cfg = load_dual_stream_model(score_checkpoint, data_path=args.data_path, device=device)
            scorer = AntigenCompatibilityScorer(
                score_model,
                tokenizer,
                max_length=score_cfg.max_length,
                device=device,
            )

    target_dataset = OASSequenceDataset(args.data_path, split=args.split)
    target_records = select_records(target_dataset, record_id=args.record_id, num_records=args.num_records)

    length_prior = None
    if args.length_mode == "empirical":
        prior_dataset = OASSequenceDataset(args.data_path, split="train")
        length_prior = EmpiricalHCDR3LengthPrior.fit(prior_dataset.records, positive_only=True)

    rows: list[dict[str, Any]] = []
    for record in target_records:
        true_span = HCDR3Span.from_record(record)
        if args.length_mode == "fixed":
            lengths = [true_span.length] * args.num_samples
        else:
            assert length_prior is not None
            lengths = length_prior.propose_lengths(record, num_lengths=args.num_samples, rng=rng)

        for proposed_length in lengths:
            try:
                if args.guidance_strength > 0:
                    # Opt-in guided path: iterative, binder-steered decoding using
                    # the generation model's own compatibility head.
                    candidates = infiller.guided_infill(
                        record,
                        length=proposed_length,
                        num_samples=1,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        guidance_strength=args.guidance_strength,
                        guidance_target=args.guidance_target,
                        order=args.guidance_order,
                        scorer=scorer,
                        rng=rng,
                    )
                else:
                    # Default path: unchanged single-pass independent sampling.
                    candidates = infiller.infill(
                        record,
                        length=proposed_length,
                        num_samples=1,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        scorer=scorer,
                    )
            except ValueError as exc:
                # e.g. a proposed length that overflows max_length; skip this
                # length rather than aborting the whole generation run.
                print(f"[warn] skipping length {proposed_length} for {record.record_id}: {exc}")
                continue
            for candidate in candidates:
                rows.append(
                    candidate_to_json(
                        record=record,
                        true_span=true_span,
                        length_mode=args.length_mode,
                        candidate=candidate,
                        guidance_strength=args.guidance_strength,
                        guidance_order=args.guidance_order,
                    )
                )

    write_jsonl(rows, args.output_path)


if __name__ == "__main__":
    main()
