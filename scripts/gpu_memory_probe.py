#!/usr/bin/env python
"""
Measure peak GPU memory for candidate encoder-context limits (Open Decision 9).

`scripts/context_length_census.py` answers "what context does the DATA need".
This answers the other half of the same decision: "what context does the GPU
fit". The plan requires both before an owner selects `max_length` /
`antigen_max_length` -- a census alone will happily recommend a limit that
cannot be trained on the machine that has to train it.

Synthetic batches on purpose. Peak activation memory is a function of shapes,
dtype, and the module graph, not of which residues are in the tensor, so no
corpus is needed and the probe runs anywhere. What it CANNOT tell you is
anything about data quality; pair it with the census.

Two coupling facts this probe surfaces, both easy to get wrong:

1. The two streams size independently since AB-07: the antigen encoder is built
   with `config.effective_antigen_max_length` and the collator truncates to the
   same number. `--pairing coupled` measures the symmetric configuration
   (antigen budget == antibody budget); `--pairing decoupled` measures the full
   cross product, which is what the owner needs in order to price a long antigen
   context against a short antibody one. Before AB-07 every decoupled pair was
   unbuildable and the sweep reported `StructuralLimit` for all of them.
2. `use_amp` changes peak memory materially and is a per-run config field, so
   the probe reports both settings rather than picking one.

Usage:

    python scripts/gpu_memory_probe.py \\
        --max-length 192 256 288 \\
        --antigen-max-length 192 512 1024 2048 \\
        --batch-size 8 16 \\
        --output-json outputs/gpu-memory-probe.json
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.models.mlm import (  # noqa: E402
    AntibodyAntigenCrossAttention,
    MLMConfig,
)
from smallAntibodyGen.tokenizer import AminoAcidTokenizer  # noqa: E402

# The stage-3/stage-4 architecture as checked in. Kept here as an explicit
# literal rather than parsed from a YAML so the probe measures ONE known shape
# and a config edit cannot silently change what a recorded number refers to.
BASE_ARCHITECTURE: dict[str, Any] = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    "norm_first": True,
}

BYTES_PER_MIB = 1024 * 1024


def describe_device(device: torch.device) -> dict[str, Any]:
    """Record what the numbers were measured on; a MiB figure is meaningless without it."""
    info: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "name": props.name,
                "total_memory_mib": round(props.total_memory / BYTES_PER_MIB, 1),
                "capability": f"{props.major}.{props.minor}",
                "cuda": torch.version.cuda,
            }
        )
    return info


def build_probe_model(
    max_length: int,
    antigen_max_length: int,
    tokenizer: AminoAcidTokenizer,
) -> AntibodyAntigenCrossAttention:
    """Build the dual-stream model at one candidate context pair."""
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=max_length,
        antigen_max_length=antigen_max_length,
        antigen_encoder_type="scratch",
        **BASE_ARCHITECTURE,
    )
    return AntibodyAntigenCrossAttention(config)


def probe_once(
    *,
    max_length: int,
    antigen_max_length: int,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
    tokenizer: AminoAcidTokenizer,
) -> dict[str, Any]:
    """
    One forward + backward at full sequence occupancy, reporting peak memory.

    Every position is a real token (no padding): the probe must measure the
    WORST case the limit permits, not the average the corpus happens to have,
    because the limit is what decides whether a batch can be allocated at all.
    """
    result: dict[str, Any] = {
        "max_length": max_length,
        "antigen_max_length": antigen_max_length,
        "batch_size": batch_size,
        "use_amp": use_amp,
    }

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        model = build_probe_model(max_length, antigen_max_length, tokenizer).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        result["parameters"] = sum(p.numel() for p in model.parameters())

        # Ids in the residue range; the specific ids do not affect memory.
        low, high = 0, tokenizer.vocab_size
        antibody = torch.randint(low, high, (batch_size, max_length), device=device)
        antigen = torch.randint(low, high, (batch_size, antigen_max_length), device=device)
        antibody_mask = torch.ones_like(antibody)
        antigen_mask = torch.ones_like(antigen)
        labels = torch.randint(low, high, (batch_size, max_length), device=device)

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp and device.type == "cuda"
        ):
            logits, compat_logits = model(
                antibody, antibody_mask, antigen, antigen_mask
            )
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        result["ok"] = True
        result["loss_finite"] = bool(torch.isfinite(loss).item())
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB
            peak_reserved = torch.cuda.max_memory_reserved(device) / BYTES_PER_MIB
            total = torch.cuda.get_device_properties(device).total_memory / BYTES_PER_MIB
            result["peak_allocated_mib"] = round(peak_allocated, 1)
            result["peak_reserved_mib"] = round(peak_reserved, 1)
            # `ok: True` is NOT the same as "fits". On Windows the NVIDIA driver
            # (WDDM, r536+) silently spills CUDA allocations into shared system
            # memory over PCIe instead of raising OutOfMemoryError, so a
            # configuration needing 5.6 GB completes on a 4 GB card -- at a speed
            # that makes it useless for training. Measured here: ab=288/ag=2048/
            # bs=32 reported 5628.6 MiB allocated on a 4095.7 MiB device.
            # Without this flag the row reads as a success and someone budgets
            # against it.
            result["device_total_mib"] = round(total, 1)
            result["fits_in_device_memory"] = bool(peak_reserved <= total)
            if not result["fits_in_device_memory"]:
                result["warning"] = (
                    f"peak {peak_reserved:.0f} MiB exceeds device memory "
                    f"{total:.0f} MiB; the driver spilled to shared system RAM, "
                    "so this ran but does not fit"
                )
    except torch.cuda.OutOfMemoryError as exc:
        # An OOM is a RESULT, not a crash: "this limit does not fit" is exactly
        # what the owner needs recorded, so it is captured rather than raised.
        result["ok"] = False
        result["error"] = "OutOfMemoryError"
        result["error_detail"] = str(exc).splitlines()[0]
    except ValueError as exc:
        # A STRUCTURAL rejection, not a memory one, and the distinction matters:
        # it means the requested pair cannot be built at all on this path, so no
        # amount of batch-size reduction or gradient accumulation makes it fit.
        # On the scratch path this fires whenever antigen > max_length, because
        # the antigen encoder is constructed from the SAME config as the antibody
        # encoder and inherits the antibody `max_length`; `antigen_max_length` is
        # read nowhere in models/mlm.py except its own validator. It also fires
        # for antigen_max_length > 1024, which that validator caps.
        result["ok"] = False
        result["error"] = "StructuralLimit"
        result["error_detail"] = str(exc).splitlines()[0]
    except RuntimeError as exc:
        result["ok"] = False
        result["error"] = type(exc).__name__
        result["error_detail"] = str(exc).splitlines()[0]
    finally:
        for name in ("model", "optimizer", "antibody", "antigen", "labels", "logits"):
            if name in locals():
                del locals()[name]
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-length", type=int, nargs="+", default=[192, 256, 288])
    parser.add_argument(
        "--antigen-max-length", type=int, nargs="+", default=[192, 512, 1024, 2048]
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--amp",
        choices=["on", "off", "both"],
        default="both",
        help="use_amp is a per-run config field; 'both' measures each.",
    )
    parser.add_argument(
        "--pairing",
        choices=["coupled", "decoupled", "both"],
        default="both",
        help=(
            "coupled: antigen cap == max_length, which is what the scratch path "
            "actually does today. decoupled: every antigen cap at every "
            "max_length, i.e. what fixing that coupling would cost."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device)
    tokenizer = AminoAcidTokenizer()

    amp_settings = {"on": [True], "off": [False], "both": [False, True]}[args.amp]

    combos: list[tuple[int, int]] = []
    if args.pairing in ("coupled", "both"):
        combos.extend((length, length) for length in args.max_length)
    if args.pairing in ("decoupled", "both"):
        combos.extend(
            (length, antigen)
            for length in args.max_length
            for antigen in args.antigen_max_length
        )
    # Deterministic order, no duplicates.
    combos = sorted(set(combos))

    device_info = describe_device(device)
    print(f"[probe] device: {json.dumps(device_info, sort_keys=True)}")
    if device.type != "cuda":
        print("[probe] WARNING: not a CUDA device; memory figures will be absent.")

    results: list[dict[str, Any]] = []
    for use_amp in amp_settings:
        for max_length, antigen_max_length in combos:
            for batch_size in sorted(args.batch_size):
                row = probe_once(
                    max_length=max_length,
                    antigen_max_length=antigen_max_length,
                    batch_size=batch_size,
                    use_amp=use_amp,
                    device=device,
                    tokenizer=tokenizer,
                )
                results.append(row)
                if not row["ok"]:
                    status = f"FAILED ({row.get('error')})"
                elif row.get("fits_in_device_memory") is False:
                    status = (
                        f"{row['peak_reserved_mib']} MiB reserved  "
                        f"** DOES NOT FIT (spilled to system RAM) **"
                    )
                else:
                    status = f"{row.get('peak_reserved_mib', '-')} MiB reserved"
                print(
                    f"  amp={'on ' if use_amp else 'off'} "
                    f"ab={max_length:>4} ag={antigen_max_length:>4} "
                    f"bs={batch_size:>3}  {status}"
                )

    payload = {
        "schema_version": "gpu-memory-probe/1",
        "device": device_info,
        "architecture": BASE_ARCHITECTURE,
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[probe] wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
