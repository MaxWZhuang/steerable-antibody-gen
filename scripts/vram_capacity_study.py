#!/usr/bin/env python
"""How much GPU memory can a dedicated training process actually sustain here?

This answers a question no passive counter can. On this Windows/WDDM box CUDA
reports an oversubscribable budget rather than free framebuffer (AB-14), the
per-process GPU counters double-count shared allocations, and an over-large
configuration does not raise -- it spills to system RAM and reports success. So
"what is free right now" does not tell you what a training process may claim and
still run at full speed.

The study has two stages, and the distinction between them is the point:

* **sweep** -- allocate an actively exercised working set of increasing size in a
  FRESH CHILD PROCESS per point, read and write the whole allocation every
  iteration, and measure achieved bandwidth. Ballast that is allocated once and
  never touched is not evidence: WDDM will happily page out memory nobody reads.
* **pilot** -- run a real forward/backward/optimizer-step loop at increasing
  microbatch, including optimizer-state initialisation, for minutes rather than
  a handful of steps.

Acceptance is declared BEFORE the boundary is examined, and escalation stops on
the first sustained failure. The last passing point is then re-confirmed in a
fresh process, because a point that passes only after a warm allocator is not a
point you can launch into.

**What this can and cannot conclude.** A sustained pass means "throughput stayed
within the declared tolerance at this working-set size, under the desktop
conditions recorded alongside it". It is NOT a claim that no paging occurred --
demonstrating that needs WDDM tracing (GPUView paging packets), which this script
does not collect. It reports `sustained_throughput_within_tolerance`, never
`paging_free`.

Writes to its own diagnostic root, never into a campaign run root: the profile
CLI writes `gpu_profile.json` and the forecast directly into its resolved run
root, and this repository has already had a manifest damaged by a follow-up
helper aimed at a landed flight.

Usage::

    python scripts/vram_capacity_study.py --out outputs/vram_capacity_<date>
    python scripts/vram_capacity_study.py --stage sweep --max-mib 3600
    python scripts/vram_capacity_study.py --stage pilot --microbatches 1,2,4,8
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments.her2_runtime import (  # noqa: E402
    require, save_json)
from smallAntibodyGen.experiments.her2_vram import vram_reading  # noqa: E402

#: Declared before any boundary is examined, so the stopping rule cannot drift.
DEFAULT_TOLERANCE = 0.75
DEFAULT_SECONDS = 20.0
DEFAULT_WARMUP = 3.0


def desktop_state():
    """Whatever we can record about what else is using the card.

    The envelope is a function of the desktop, so a capacity number without this
    block is not reproducible. Per-process values are recorded but never summed:
    Windows attributes shared allocations to several processes at once.
    """
    state = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
             "utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            name, total, used, free, util, driver = [
                part.strip() for part in out.stdout.strip().splitlines()[0].split(",")]
            state["gpu"] = {"name": name, "total_mib": int(total), "used_mib": int(used),
                            "free_mib": int(free), "utilization_percent": int(util),
                            "driver": driver}
    except Exception as error:                     # pragma: no cover - platform specific
        state["gpu_error"] = f"{type(error).__name__}: {error}"
    state["vram"] = vram_reading()
    return state


# --------------------------------------------------------------------------
# stage 1: actively exercised working set, one child process per point
# --------------------------------------------------------------------------

_CHILD = r"""
import json, sys, time
import torch
mib = int(sys.argv[1]); seconds = float(sys.argv[2]); warmup = float(sys.argv[3])
result = {"requested_mib": mib}
try:
    elements = (mib * 1024 * 1024) // 4
    block = torch.empty(elements, dtype=torch.float32, device="cuda")
    block.fill_(1.0)
    torch.cuda.synchronize()
    # Touch the WHOLE allocation every iteration. Allocate-once ballast proves
    # nothing: WDDM pages out memory nobody reads, and the run still "passes".
    def pass_over():
        block.mul_(1.0000001)
        total = float(block[::4096].sum().item())
        torch.cuda.synchronize()
        return total
    deadline = time.perf_counter() + warmup
    while time.perf_counter() < deadline:
        pass_over()
    passes, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds:
        pass_over(); passes += 1
    elapsed = time.perf_counter() - started
    result.update(
        ok=True, passes=passes, elapsed_seconds=elapsed,
        # Each pass reads and writes the block once, hence 2x.
        gib_per_second=(passes * 2 * mib / 1024.0) / elapsed if elapsed else None,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
except torch.cuda.OutOfMemoryError as error:
    result.update(ok=False, failure=f"OutOfMemoryError: {error}")
except Exception as error:
    result.update(ok=False, failure=f"{type(error).__name__}: {error}")
print("RESULT " + json.dumps(result))
"""


def _run_point(child, size, seconds, warmup):
    """One probe in a fresh process. A timeout is an OBSERVATION, not a crash.

    Severe paging is exactly what makes a child overrun its timeout, and that is
    exactly the boundary this study exists to find. Letting ``TimeoutExpired``
    propagate would abort the whole run before its JSON is written, discarding
    every completed measurement at the most interesting point.
    """
    try:
        completed = subprocess.run([sys.executable, str(child), str(size),
                                    str(seconds), str(warmup)],
                                   capture_output=True, text=True,
                                   timeout=seconds * 6 + 120)
    except subprocess.TimeoutExpired:
        return {"requested_mib": size, "ok": False,
                "failure": f"child exceeded {seconds * 6 + 120:.0f}s",
                "timed_out": True}
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
            payload.setdefault("timed_out", False)
            return payload
    return {"requested_mib": size, "ok": False, "timed_out": False,
            "failure": f"child exited {completed.returncode}: "
                       f"{completed.stderr.strip()[:400]}"}


def _sustained(payload, reference, tolerance):
    """The single acceptance predicate, applied to sweep points AND confirmation."""
    rate = payload.get("gib_per_second")
    ratio = None if not (rate and reference) else rate / reference
    return ratio, bool(payload.get("ok") and ratio is not None and ratio >= tolerance)


def sweep(out_dir, *, start_mib, step_mib, max_mib, seconds, warmup, tolerance):
    """Escalate an exercised working set until throughput collapses."""
    child = out_dir / "_probe_child.py"
    child.write_text(_CHILD, encoding="utf-8")
    rows, reference, last_pass = [], None, None
    size = int(start_mib)
    while size <= int(max_mib):
        before = desktop_state()
        payload = _run_point(child, size, seconds, warmup)
        rate = payload.get("gib_per_second")
        if payload.get("ok") and rate and reference is None:
            reference = rate                        # the smallest point sets the reference
        ratio, sustained = _sustained(payload, reference, tolerance)
        payload.update(throughput_ratio=ratio, sustained=sustained,
                       desktop_before=before, desktop_after=desktop_state())
        rows.append(payload)
        print(f"  {size:>6} MiB  ok={str(payload.get('ok')):<5} "
              f"rate={rate if rate is None else round(rate, 2)} GiB/s  "
              f"ratio={ratio if ratio is None else round(ratio, 3)}  "
              f"sustained={sustained}", flush=True)
        if not sustained:
            break
        last_pass = size
        size += int(step_mib)

    # The confirmation is EVALUATED, not merely stored. A re-run that OOMs, times
    # out or falls below tolerance withdraws the qualified capacity: reporting a
    # size whose own confirmation failed is exactly the overclaim this study is
    # supposed to prevent.
    confirmation, qualified = None, None
    if last_pass is not None:
        print(f"  re-confirming {last_pass} MiB in a fresh process", flush=True)
        confirmation = _run_point(child, last_pass, seconds, warmup)
        ratio, ok = _sustained(confirmation, reference, tolerance)
        confirmation.update(throughput_ratio=ratio, sustained=ok,
                            desktop_at_confirmation=desktop_state())
        qualified = last_pass if ok else None
        print(f"  confirmation {'PASSED' if ok else 'FAILED'} "
              f"-> qualified capacity {qualified}", flush=True)
    child.unlink(missing_ok=True)
    return {"rows": rows, "reference_gib_per_second": reference,
            "largest_sustained_mib": last_pass,
            "qualified_capacity_mib": qualified,
            "confirmation": confirmation,
            "tolerance": tolerance,
            "qualified_rule": ("qualified_capacity_mib is null unless the fresh-process "
                               "confirmation ALSO met the declared tolerance. "
                               "largest_sustained_mib is the unconfirmed escalation "
                               "result and must not be quoted as a capacity."),
            "claim": ("sustained_throughput_within_tolerance under the recorded desktop "
                      "state. NOT a paging-free claim: that needs WDDM tracing this "
                      "script does not collect."),
            "reference_note": ("the smallest measured point sets the reference, so its own "
                               "ratio is 1.0 by construction and it cannot detect a "
                               "collapse that was already happening at that size")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "vram_capacity_study",
                        help="diagnostic root; must NOT be a campaign run root")
    parser.add_argument("--stage", choices=("sweep",), default="sweep",
                        help="pilot stage lands with the corrected profiler; sweep is "
                             "the device-level envelope")
    parser.add_argument("--start-mib", type=int, default=256)
    parser.add_argument("--step-mib", type=int, default=256)
    parser.add_argument("--max-mib", type=int, default=3840)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="declared BEFORE the boundary is examined")
    args = parser.parse_args()

    for marker in ("stages", "campaign.lock", "revision.json", "source_snapshot.json"):
        require(not (args.out / marker).exists(),
                f"{args.out} looks like a campaign run root (found {marker}). "
                "This study writes its own diagnostic root and never into a flight.")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    require(torch.cuda.is_available(), "No CUDA device; this study measures a real card")
    print(f"capacity study -> {args.out}", flush=True)
    print(f"declared tolerance {args.tolerance} of the reference rate", flush=True)

    started = time.time()
    document = {"record_kind": "vram_capacity_study", "schema_version": "vram-capacity/1",
                "started_at": started, "argv": sys.argv[1:],
                "tolerance": args.tolerance, "seconds_per_point": args.seconds,
                "desktop_at_start": desktop_state()}
    document["sweep"] = sweep(args.out, start_mib=args.start_mib, step_mib=args.step_mib,
                              max_mib=args.max_mib, seconds=args.seconds,
                              warmup=args.warmup, tolerance=args.tolerance)
    document["desktop_at_end"] = desktop_state()
    document["wall_seconds"] = time.time() - started
    target = args.out / "vram_capacity_study.json"
    save_json(target, document)
    print(f"\nwritten to {target}", flush=True)
    sweep_doc = document["sweep"]
    largest, qualified = sweep_doc["largest_sustained_mib"], sweep_doc["qualified_capacity_mib"]
    print(f"largest sustained (unconfirmed): {largest} MiB", flush=True)
    print(f"QUALIFIED capacity (confirmed) : {qualified} MiB", flush=True)
    if qualified is None and largest is not None:
        print("  the confirmation did not meet tolerance, so no capacity is qualified.",
              flush=True)
    print("this is a device-level envelope under the recorded desktop state; a "
          "specific model still needs its own full-update pilot.", flush=True)


if __name__ == "__main__":
    main()
