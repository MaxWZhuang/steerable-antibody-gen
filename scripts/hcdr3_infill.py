#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
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
        merged.update(saved)
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
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
) -> dict[str, Any]:
    """
    Convert one generated candidate into a JSON-serializable output row.
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
        "compatibility_score": candidate.compatibility_score,
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
    if not args.no_score and score_checkpoint.exists():
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
            candidates = infiller.infill(
                record,
                length=proposed_length,
                num_samples=1,
                temperature=args.temperature,
                top_k=args.top_k,
                scorer=scorer,
            )
            for candidate in candidates:
                rows.append(
                    candidate_to_json(
                        record=record,
                        true_span=true_span,
                        length_mode=args.length_mode,
                        candidate=candidate,
                    )
                )

    write_jsonl(rows, args.output_path)


if __name__ == "__main__":
    main()
