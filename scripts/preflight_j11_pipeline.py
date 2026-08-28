#!/usr/bin/env python
"""
J11 pipeline-inclusive preflight: does the real loop fit 50 GPU-hours?

The timing calibration measured COMPUTE ONLY -- forward, backward, update on
synthetic tensors. Its step time is a lower bound, and the projection built on it
cleared the 50,000-update floor by a margin that ordinary dataloading overhead
would erase. This runs the parts that calibration skipped, on the real corpus,
with the production loaders:

  - dataset construction and the length-bucketed batch sampler
  - collation and dynamic MLM masking
  - forward / backward / optimizer update
  - checkpoint serialization
  - a validation traversal

**No numerical metric is retained.** Validation batches are traversed and their
per-batch metric counts computed -- because that Python-level work is a real cost
-- and then discarded without aggregation. Nothing that could reveal which width
is ahead is ever formed, printed, or written. J11's budget must stay blind, and a
preflight that leaked a val loss would unblind it as surely as a full run.

If the projection exceeds the budget, J11 stays blocked. Do not lower the floor
and do not select from partial runs: an underpowered comparison that produces a
winner is worse than no comparison, because it looks like evidence.

Usage::

    python scripts/preflight_j11_pipeline.py --steps 60 --val-batches 40 \\
        --output-json outputs/j11-pipeline-preflight.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.experiments import init_parity  # noqa: E402

SCHEMA_VERSION = "j11-pipeline-preflight/1"

# Frozen J11 budget (owner, amended 2026-08-28 while blind).
ARM_WIDTHS = (680, 1024)
SEEDS = (42, 31415, 271828)
TOTAL_UPDATES = 51_000
WARMUP_UPDATES = 1_000
MIN_POST_WARMUP_UPDATES = 50_000
TOTAL_GPU_HOURS = 50.0
BYTES_PER_MIB = 1024 * 1024


def load_mlm_train():
    """Import the training script so the preflight uses the PRODUCTION loaders."""
    spec = importlib.util.spec_from_file_location(
        "mlm_train_preflight", PROJECT_ROOT / "scripts" / "mlm_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def time_arm(
    mlm_train,
    config_path: Path,
    *,
    steps: int,
    warmup: int,
    val_batches: int,
    device: torch.device,
    scratch_dir: Path,
) -> dict[str, Any]:
    """Time one arm's real training loop, checkpoint save, and validation traversal."""
    from smallAntibodyGen.tokenizer import AminoAcidTokenizer

    cfg = mlm_train.parse_args(["--config", str(config_path)])
    tokenizer = AminoAcidTokenizer()

    dataset_started = time.perf_counter()
    train_dataset, val_dataset = mlm_train.build_datasets(cfg)
    dataset_seconds = time.perf_counter() - dataset_started

    model = mlm_train.build_model(tokenizer, cfg, device)
    # The same pairing the real runs use, so the preflight times the real thing.
    init_parity.reinitialize_by_module_name(model, mlm_train.build_model_config(tokenizer, cfg), seed=cfg.seed)
    init_parity.reset_training_rng(cfg.seed)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
    train_loader = mlm_train.build_train_loader(train_dataset, tokenizer, cfg, epoch=0, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- training steps, INCLUDING data fetch, collation, and masking --------
    durations: list[float] = []
    seen = 0
    loader_iter = iter(train_loader)
    while seen < warmup + steps:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast(device.type, enabled=cfg.use_amp and device.type == "cuda"):
            out = model(ids, mask)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if seen >= warmup:
            durations.append(time.perf_counter() - started)
        seen += 1
    # `loss` is deliberately never read as a value beyond finiteness.
    loss_finite = bool(torch.isfinite(loss).item())

    # ---- checkpoint serialization -------------------------------------------
    scratch_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = scratch_dir / f"preflight_{cfg.swiglu_hidden_dim}.pt"
    save_started = time.perf_counter()
    mlm_train.save_checkpoint(
        ckpt_path, model, optimizer, cfg, epoch=0, val_loss=float("nan"), scaler=scaler
    )
    checkpoint_seconds = time.perf_counter() - save_started
    checkpoint_mib = ckpt_path.stat().st_size / BYTES_PER_MIB
    ckpt_path.unlink()

    # ---- validation traversal, metrics computed then DISCARDED --------------
    eval_loader = mlm_train.build_eval_loader(val_dataset, tokenizer, cfg, device=device)
    model.eval()
    val_started = time.perf_counter()
    val_seen = 0
    with torch.no_grad():
        for batch in eval_loader:
            if val_seen >= val_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast(device.type, enabled=cfg.use_amp and device.type == "cuda"):
                out = model(ids, mask)
                logits = out[0] if isinstance(out, tuple) else out
            # Per-batch counts ARE computed, because that Python-level work is a
            # real cost this preflight exists to measure -- and then dropped on
            # the floor without aggregation. No value survives this loop.
            _discarded = mlm_train.hcdr3_metric_counts(
                logits,
                labels,
                batch["hcdr3_target_mask"].to(device),
                batch["hcdr3_token_start"].to(device),
                batch["hcdr3_token_end"].to(device),
                batch["hcdr3_valid_mask"].to(device),
            )
            del _discarded
            val_seen += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    val_seconds = time.perf_counter() - val_started

    result: dict[str, Any] = {
        "swiglu_hidden_dim": cfg.swiglu_hidden_dim,
        "config": config_path.name,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "dataset_build_seconds": round(dataset_seconds, 2),
        "timed_steps": len(durations),
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": statistics.fmean(durations),
        "p90_step_seconds": sorted(durations)[int(0.9 * (len(durations) - 1))],
        "checkpoint_save_seconds": round(checkpoint_seconds, 3),
        "checkpoint_mib": round(checkpoint_mib, 1),
        "val_batches_timed": val_seen,
        "val_seconds_per_batch": (val_seconds / val_seen) if val_seen else 0.0,
        "loss_finite": loss_finite,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB
        reserved = torch.cuda.max_memory_reserved(device) / BYTES_PER_MIB
        total = torch.cuda.get_device_properties(device).total_memory / BYTES_PER_MIB
        result.update(
            {
                "peak_allocated_mib": round(allocated, 1),
                "peak_reserved_mib": round(reserved, 1),
                "device_total_mib": round(total, 1),
                "reserved_headroom_fraction": round(1.0 - reserved / total, 4),
                "fits_without_driver_spill": bool(reserved <= total),
            }
        )

    del model, optimizer, train_loader, eval_loader, train_dataset, val_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def project(arms: list[dict[str, Any]], validations_per_run: int) -> dict[str, Any]:
    """Project the six evidence runs against the 50 GPU-hour budget."""
    per_run = []
    for arm in arms:
        train_seconds = TOTAL_UPDATES * arm["median_step_seconds"]
        val_seconds = (
            validations_per_run * arm["val_seconds_per_batch"] * arm["val_batches_timed"]
        )
        ckpt_seconds = validations_per_run * arm["checkpoint_save_seconds"]
        per_run.append(
            {
                "swiglu_hidden_dim": arm["swiglu_hidden_dim"],
                "train_hours": train_seconds / 3600.0,
                "validation_hours": val_seconds / 3600.0,
                "checkpoint_hours": ckpt_seconds / 3600.0,
                "dataset_build_hours": arm["dataset_build_seconds"] / 3600.0,
                "total_hours_per_run": (
                    train_seconds + val_seconds + ckpt_seconds + arm["dataset_build_seconds"]
                )
                / 3600.0,
            }
        )
    total_hours = len(SEEDS) * sum(entry["total_hours_per_run"] for entry in per_run)
    return {
        "total_updates_per_run": TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "post_warmup_updates": TOTAL_UPDATES - WARMUP_UPDATES,
        "meets_evidence_floor": (TOTAL_UPDATES - WARMUP_UPDATES) >= MIN_POST_WARMUP_UPDATES,
        "seeds": list(SEEDS),
        "validations_per_run": validations_per_run,
        "per_run": per_run,
        "projected_total_gpu_hours": round(total_hours, 2),
        "budget_gpu_hours": TOTAL_GPU_HOURS,
        "fits_budget": total_hours <= TOTAL_GPU_HOURS,
        "headroom_hours": round(TOTAL_GPU_HOURS - total_hours, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    default_dir = PROJECT_ROOT / "configs/experiments/swiglu_width"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--configs", type=Path, nargs="+",
                        default=[default_dir / "arm_680.yaml", default_dir / "arm_1024.yaml"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--validations-per-run", type=int, default=10,
                        help="How many validation passes a run performs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scratch-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / ".j11_preflight_scratch")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device)
    mlm_train = load_mlm_train()

    print(f"[j11-preflight] device={device}")
    print("[j11-preflight] PIPELINE-INCLUSIVE. No numerical metric is retained.\n")

    arms = []
    for config_path in args.configs:
        arm = time_arm(
            mlm_train,
            config_path,
            steps=args.steps,
            warmup=args.warmup,
            val_batches=args.val_batches,
            device=device,
            scratch_dir=args.scratch_dir,
        )
        arms.append(arm)
        print(
            f"  width {arm['swiglu_hidden_dim']:>4}  "
            f"step={arm['median_step_seconds']*1000:>7.1f} ms  "
            f"ckpt={arm['checkpoint_save_seconds']:>5.2f} s ({arm['checkpoint_mib']:.0f} MiB)  "
            f"val={arm['val_seconds_per_batch']*1000:>6.1f} ms/batch  "
            f"alloc={arm.get('peak_allocated_mib', 0):.0f} MiB"
        )

    projection = project(arms, args.validations_per_run)
    print(f"\n  schedule: {TOTAL_UPDATES:,} updates ({WARMUP_UPDATES:,} warmup + "
          f"{TOTAL_UPDATES-WARMUP_UPDATES:,} post-warmup)")
    print(f"  evidence floor met by schedule: {projection['meets_evidence_floor']}")
    for entry in projection["per_run"]:
        print(f"    width {entry['swiglu_hidden_dim']:>4}: "
              f"train {entry['train_hours']:.2f} h + val {entry['validation_hours']:.2f} h "
              f"+ ckpt {entry['checkpoint_hours']:.2f} h = "
              f"{entry['total_hours_per_run']:.2f} h/run")
    print(f"\n  PROJECTED TOTAL: {projection['projected_total_gpu_hours']} GPU-hours "
          f"(budget {TOTAL_GPU_HOURS})")
    verdict = "FITS -- J11 may launch" if projection["fits_budget"] else (
        "EXCEEDS BUDGET -- J11 stays blocked; do not lower the floor or select "
        "from partial runs"
    )
    print(f"  VERDICT: {verdict}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Pipeline-inclusive timing. Validation batches were traversed and their "
            "per-batch counts computed then discarded without aggregation; no "
            "numerical metric was retained, so J11 remains blind."
        ),
        "arms": arms,
        "projection": projection,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\n[j11-preflight] wrote {args.output_json}")
    return 0 if projection["fits_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
