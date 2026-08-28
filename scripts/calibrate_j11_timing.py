#!/usr/bin/env python
"""
J11 timing-only calibration: how long does each SwiGLU width cost per update?

Runs the EXACT forward/backward/optimizer path a real J11 run would, on the real
corpus, and retains **no validation metric of any kind**. That separation is the
point: the pilot budget must be frozen before anybody can see which arm is
winning, or the budget becomes a knob that gets turned until the answer looks
right.

What it produces:

- median step time per arm (median, not mean -- a single driver hiccup or a
  first-step compile should not set the budget);
- allocated and reserved memory, reported SEPARATELY, with allocated the figure
  quoted for comparisons because reserved is the caching allocator's pool and
  drifts a few percent between identical runs;
- a projection of whether one complete corpus epoch per run fits the 36 GPU-hour
  evidence budget across all six runs, and if not, the common step count derived
  from the SLOWER arm.

Deriving the step count from the slower arm is deliberate. J11 measures whether
more capacity is worth its compute, so both arms must see the same DATA; giving
the faster arm more steps would answer a different question and flatter the
narrow arm.

Usage::

    python scripts/calibrate_j11_timing.py --data-path data/processed/... \\
        --output-json outputs/j11-timing-calibration.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.experiments import init_parity  # noqa: E402
from smallAntibodyGen.models.mlm import AntibodyMLM, MLMConfig  # noqa: E402
from smallAntibodyGen.tokenizer import AminoAcidTokenizer  # noqa: E402

SCHEMA_VERSION = "j11-timing-calibration/1"

#: The canonical modern block, adopted by owner decision 2026-08-28.
CANONICAL_MODERN: dict[str, Any] = {
    "position_encoding": "rope",
    "norm_type": "rmsnorm",
    "ffn_type": "swiglu",
    "attention_bias": False,
    "ffn_bias": False,
    "norm_first": True,
}

CANONICAL_SHAPE: dict[str, Any] = {
    "max_length": 288,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
}

#: Frozen J11 budget (owner, 2026-08-28).
ARM_WIDTHS = (680, 1024)
SEEDS = (42, 31415, 271828)
BATCH_SIZE = 16
TOTAL_GPU_HOURS = 36.0
MIN_POST_WARMUP_UPDATES = 50_000
BYTES_PER_MIB = 1024 * 1024


def build_arm(width: int, tokenizer: AminoAcidTokenizer, seed: int):
    """Build one paired arm: canonical modern block at ``width``."""
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        swiglu_hidden_dim=width,
        **CANONICAL_SHAPE,
        **CANONICAL_MODERN,
    )
    config.validate()
    model = AntibodyMLM(config)
    # Pairing: identical values for every same-name, same-shape parameter, then a
    # known training stream. Without this the arms differ in far more than width.
    init_parity.reinitialize_by_module_name(model, config, seed=seed)
    init_parity.reset_training_rng(seed)
    return config, model


def time_arm(
    width: int,
    *,
    device: torch.device,
    steps: int,
    warmup: int,
    seed: int,
    use_amp: bool,
) -> dict[str, Any]:
    """
    Median step time for one arm over ``steps`` timed updates.

    Synthetic batches at FULL occupancy. Step time is a function of shapes and the
    module graph, not of which residues are present, and using synthetic input
    keeps the calibration from touching the corpus in a way that could be mistaken
    for evaluation. The shapes are the real ones.
    """
    tokenizer = AminoAcidTokenizer()
    config, model = build_arm(width, tokenizer, seed)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    ids = torch.randint(3, tokenizer.vocab_size, (BATCH_SIZE, CANONICAL_SHAPE["max_length"]), device=device)
    mask = torch.ones_like(ids)
    labels = torch.randint(3, tokenizer.vocab_size, ids.shape, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations: list[float] = []
    for step in range(warmup + steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.amp.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            out = model(ids, mask)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        # Warmup steps are discarded: the first updates include allocator growth
        # and kernel selection, which are real but one-off and would inflate a
        # per-step budget applied to hundreds of thousands of steps.
        if step >= warmup:
            durations.append(time.perf_counter() - started)

    result: dict[str, Any] = {
        "swiglu_hidden_dim": width,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "use_amp": bool(use_amp and device.type == "cuda"),
        "timed_steps": len(durations),
        "warmup_steps": warmup,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": statistics.fmean(durations),
        # Reported so a reader can see whether the median is hiding a long tail.
        "p90_step_seconds": sorted(durations)[int(0.9 * (len(durations) - 1))],
        "min_step_seconds": min(durations),
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB
        reserved = torch.cuda.max_memory_reserved(device) / BYTES_PER_MIB
        total = torch.cuda.get_device_properties(device).total_memory / BYTES_PER_MIB
        result.update(
            {
                # Separate labels, deliberately. `allocated` is the figure quoted
                # for comparisons; `reserved` is the caching allocator's pool and
                # drifts between identical runs.
                "peak_allocated_mib": round(allocated, 1),
                "peak_reserved_mib": round(reserved, 1),
                "device_total_mib": round(total, 1),
                "reserved_headroom_fraction": round(1.0 - reserved / total, 4),
                "fits_without_driver_spill": bool(reserved <= total),
            }
        )

    del model, optimizer, ids, mask, labels, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def project_budget(
    arms: list[dict[str, Any]],
    corpus_rows: int | None,
    validation_reserve_hours: float = 0.0,
    overhead_fraction: float = 0.0,
) -> dict[str, Any]:
    """
    Turn measured step times into the frozen pilot budget.

    One epoch per run if that fits 36 GPU-hours across all six; otherwise a common
    step count derived from the SLOWER arm, so both arms see identical data.
    """
    by_width = {arm["swiglu_hidden_dim"]: arm for arm in arms}
    slower = max(arms, key=lambda arm: arm["median_step_seconds"])
    faster = min(arms, key=lambda arm: arm["median_step_seconds"])
    slowdown = slower["median_step_seconds"] / faster["median_step_seconds"] - 1.0

    runs = len(ARM_WIDTHS) * len(SEEDS)
    # Validation is fixed cost that the 36 hours must also cover, so it comes off
    # the top rather than being discovered at the end of the last run.
    budget_seconds = (TOTAL_GPU_HOURS - validation_reserve_hours) * 3600.0
    # Each arm runs len(SEEDS) times, so the cost of N steps is
    # N * seeds * (t_680 + t_1024).
    #
    # `overhead_fraction` inflates the measured step time to stand for everything
    # this calibration does NOT execute: dataloading, collation, bucketing, and
    # checkpoint writes. The measured figure is a compute-only LOWER BOUND on the
    # real step, so projecting from it unadjusted would overstate how many steps
    # the budget buys.
    seconds_per_step_all_runs = (
        len(SEEDS)
        * sum(arm["median_step_seconds"] for arm in arms)
        * (1.0 + overhead_fraction)
    )
    affordable_steps = int(budget_seconds / seconds_per_step_all_runs)

    projection: dict[str, Any] = {
        "runs": runs,
        "seeds": list(SEEDS),
        "total_gpu_hours_budget": TOTAL_GPU_HOURS,
        "median_step_seconds_by_width": {
            str(width): arm["median_step_seconds"] for width, arm in by_width.items()
        },
        "slower_arm_width": slower["swiglu_hidden_dim"],
        "measured_slowdown_fraction": round(slowdown, 4),
        "affordable_common_steps": affordable_steps,
        "min_post_warmup_updates": MIN_POST_WARMUP_UPDATES,
        "meets_minimum_updates": affordable_steps >= MIN_POST_WARMUP_UPDATES,
        "validation_reserve_hours": validation_reserve_hours,
        "assumed_overhead_fraction": overhead_fraction,
        "margin_over_minimum": round(
            affordable_steps / MIN_POST_WARMUP_UPDATES - 1.0, 4
        ),
        "step_time_excludes": [
            "dataloading and collation",
            "HCDR3/MLM masking",
            "checkpoint writes",
            "validation passes",
        ],
    }

    if corpus_rows is not None:
        steps_per_epoch = -(-corpus_rows // BATCH_SIZE)  # ceil
        epoch_seconds = steps_per_epoch * seconds_per_step_all_runs
        projection.update(
            {
                "corpus_rows": corpus_rows,
                "steps_per_epoch": steps_per_epoch,
                "one_epoch_all_runs_gpu_hours": round(epoch_seconds / 3600.0, 2),
                "one_epoch_fits_budget": epoch_seconds <= budget_seconds,
            }
        )
        if epoch_seconds <= budget_seconds:
            projection["recommended_common_steps"] = steps_per_epoch
            projection["basis"] = "one complete corpus epoch per run"
        else:
            projection["recommended_common_steps"] = affordable_steps
            projection["basis"] = (
                "one epoch does not fit 36 GPU-hours; common step count derived "
                "from the slower arm so both arms see identical data"
            )
    return projection


def count_rows(path: Path) -> int:
    """Count rows in a JSONL(.gz) corpus, for the epoch projection."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    rows = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--steps", type=int, default=60, help="Timed steps per arm.")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Corpus to count rows in, for the one-epoch projection.",
    )
    parser.add_argument(
        "--validation-reserve-hours",
        type=float,
        default=0.0,
        help=(
            "GPU-hours reserved for fixed validation, taken off the 36-hour budget "
            "before deriving the step count."
        ),
    )
    parser.add_argument(
        "--overhead-fraction",
        type=float,
        default=0.0,
        help=(
            "Inflate the measured step time by this fraction to stand for "
            "dataloading, collation, and checkpointing, none of which this "
            "calibration executes."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device)

    print(f"[j11-timing] device={device} batch={BATCH_SIZE} amp={not args.no_amp}")
    print("[j11-timing] TIMING ONLY -- no validation metric is computed or retained.\n")

    arms = []
    for width in ARM_WIDTHS:
        arm = time_arm(
            width,
            device=device,
            steps=args.steps,
            warmup=args.warmup,
            seed=args.seed,
            use_amp=not args.no_amp,
        )
        arms.append(arm)
        memory = (
            f"alloc={arm['peak_allocated_mib']} reserved={arm['peak_reserved_mib']} MiB "
            f"(headroom {arm['reserved_headroom_fraction']:.1%})"
            if "peak_allocated_mib" in arm
            else "cpu"
        )
        print(
            f"  width {width:>4}  params={arm['total_parameters']:>10,}  "
            f"median={arm['median_step_seconds']*1000:>7.2f} ms  "
            f"p90={arm['p90_step_seconds']*1000:>7.2f} ms  {memory}"
        )

    corpus_rows = count_rows(args.data_path) if args.data_path else None
    projection = project_budget(
        arms, corpus_rows, args.validation_reserve_hours, args.overhead_fraction
    )
    # The headline number sits close to the 50,000-update floor, so the report
    # carries a sensitivity table rather than a single figure: a reader can see
    # exactly which combination of validation reserve and real-loop overhead
    # pushes the budget under the floor.
    sensitivity = []
    for reserve in (0.0, 1.0, 2.0, 4.0):
        for overhead in (0.0, 0.10, 0.20, 0.30):
            probe = project_budget(arms, None, reserve, overhead)
            sensitivity.append(
                {
                    "validation_reserve_hours": reserve,
                    "overhead_fraction": overhead,
                    "affordable_common_steps": probe["affordable_common_steps"],
                    "meets_minimum_updates": probe["meets_minimum_updates"],
                }
            )
    projection["sensitivity"] = sensitivity

    print(f"\n  slowdown of width {projection['slower_arm_width']}: "
          f"{projection['measured_slowdown_fraction']:+.1%}")
    if corpus_rows is not None:
        print(f"  corpus rows: {corpus_rows:,} -> {projection['steps_per_epoch']:,} steps/epoch")
        print(f"  one epoch x 6 runs: {projection['one_epoch_all_runs_gpu_hours']} GPU-hours "
              f"(budget {TOTAL_GPU_HOURS})")
    affordable = projection["affordable_common_steps"]
    print(f"  affordable common steps within budget: {affordable:,}")
    print(f"  meets the {MIN_POST_WARMUP_UPDATES:,}-update minimum: "
          f"{projection['meets_minimum_updates']} "
          f"(margin {projection['margin_over_minimum']:+.1%})")
    print("\n  sensitivity -- steps affordable, and whether the floor is met:")
    header = "".join(f"{o:>12.0%}" for o in (0.0, 0.10, 0.20, 0.30))
    print("    reserve\\overhead " + header)
    for reserve in (0.0, 1.0, 2.0, 4.0):
        cells = []
        for overhead in (0.0, 0.10, 0.20, 0.30):
            row = next(
                r for r in projection["sensitivity"]
                if r["validation_reserve_hours"] == reserve
                and r["overhead_fraction"] == overhead
            )
            flag = " " if row["meets_minimum_updates"] else "!"
            cells.append(f"{row['affordable_common_steps']:>11,}{flag}")
        print(f"    {reserve:>6.1f} h        " + "".join(cells))
    print("    (! = below the 50,000-update floor; no selection may be made)")
    if "recommended_common_steps" in projection:
        print(f"  RECOMMENDED common steps: {projection['recommended_common_steps']:,}")
        print(f"  basis: {projection['basis']}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Timing only. No validation metric was computed or retained, so the "
            "pilot budget is frozen without anybody having seen which arm wins."
        ),
        "architecture": {**CANONICAL_SHAPE, **CANONICAL_MODERN},
        "arms": arms,
        "projection": projection,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\n[j11-timing] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
