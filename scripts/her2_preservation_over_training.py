#!/usr/bin/env python
"""Preservation as a function of training length, on one audit bank per seed.

The flight audits its two PRIMARY endpoints only -- u1000 and u3750 -- so the
matched arm table has preservation at two points and ranking at five. That makes
"the tail penalty degrades with training while forward-KL improves" an
interpolation between two points rather than a measured curve.

This fills the intermediate updates. Every checkpoint is scored on the SAME
50,000-row ``final_preservation`` bank the flight drew for that parent seed and
shared across all of its Block A arms, so the whole curve is paired within a
seed and comparable with the flight's own audit column.

Grid, set by which checkpoints exist rather than by choice:

* ``IPO_TAIL`` -- u250, u500, u1000, u2000, u3750 (the flight kept
  ``checkpoint_update250/500.pt`` alongside the endpoints)
* ``IPO_0``, ``IPO_FKL``, ``continued_sft`` -- u1000, u2000, u3750

u1000 and u3750 are **reused from the flight's persisted per-row arrays**, not
rescored: they are already on this bank, and recomputing them here under the
same convention is a consistency check rather than new evidence. Only the
intermediate checkpoints need the GPU.

Nothing here selects anything. The endpoints were named before the flight ran
and this adds points between them.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

TENFOLD, HUNDREDFOLD = math.log(10.0), math.log(100.0)
#: any Block A audit file at a seed carries that seed's shared bank
BANK_SOURCE = "A_IPO_0@u1000_seed{seed}.npz"
GRID = {
    "IPO_TAIL": (250, 500, 1000, 2000, 3750),
    "IPO_0": (1000, 2000, 3750),
    "IPO_FKL": (1000, 2000, 3750),
    "continued_sft": (1000, 2000, 3750),
}
#: already audited by the flight on this same bank
PERSISTED = (1000, 3750)


def checkpoint_path(family, update, seed, run_root, historical):
    if family == "IPO_TAIL":
        directory = run_root / "trajectories" / f"A_IPO_TAIL_seed{seed}"
        # the flight names sub-endpoint saves checkpoint_*, endpoints endpoint_*
        for prefix in ("endpoint", "checkpoint"):
            candidate = directory / f"{prefix}_update{update}.pt"
            if candidate.is_file():
                return candidate
        return directory / f"endpoint_update{update}.pt"
    name = {"IPO_0": "ipo_lambda0", "IPO_FKL": "ipo_lambda10",
            "continued_sft": "continued_sft_lambda0p1"}[family]
    return historical / "trajectories" / f"{name}_seed{seed}" / f"endpoint_update{update}.pt"


def persisted_path(family, update, seed, run_root):
    stem = ("continued_sft" if family == "continued_sft" else f"A_{family}")
    return run_root / "preservation" / f"{stem}@u{update}_seed{seed}.npz"


def tail_block(drop):
    ten, hundred = int((drop >= TENFOLD).sum()), int((drop >= HUNDREDFOLD).sum())
    rows = int(drop.size)
    return {"rows": rows, "mean_drop": float(drop.mean()),
            "standard_error": float(drop.std(ddof=1) / math.sqrt(rows)),
            "median": float(np.median(drop)),
            "fraction_below_parent": float((drop > 0).mean()),
            "max_drop": float(drop.max()),
            "tenfold_rate": ten / rows, "tenfold_count": ten,
            "tenfold_wilson": metrics.wilson_interval(ten, rows),
            "hundredfold_rate": hundred / rows, "hundredfold_count": hundred,
            "hundredfold_wilson": metrics.wilson_interval(hundred, rows),
            "convention": "inclusive: drop >= ln(10), drop >= ln(100)"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--historical", default="outputs/her2_parent_replay_20260920")
    parser.add_argument("--output", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_root, historical = ROOT / args.run_root, ROOT / args.historical
    out = ROOT / args.output
    (out / "preservation_curve").mkdir(parents=True, exist_ok=True)

    context = spec.resolve_context(ROOT, config_path=ROOT / args.config, run_root=run_root)
    services = campaign.build_services(context, device=args.device)
    batch = int(context.config["inference"]["score_batch_size"])

    records, missing = [], []
    for seed in seeds:
        source = run_root / "preservation" / BANK_SOURCE.format(seed=seed)
        if not source.is_file():
            missing.append({"seed": seed, "missing": source.name})
            continue
        bank = paths.read_arrays(source)
        index = np.asarray(bank["core_index"])
        parent_scores = np.asarray(bank["parent_sum_log_probability"], dtype=np.float64)
        for family, updates in GRID.items():
            for update in updates:
                origin, drop = None, None
                if update in PERSISTED:
                    persisted = persisted_path(family, update, seed, run_root)
                    if persisted.is_file():
                        drop = np.asarray(paths.read_arrays(persisted)["log_ratio"],
                                          dtype=np.float64)
                        origin = f"flight: {persisted.name}"
                if drop is None:
                    checkpoint = checkpoint_path(family, update, seed, run_root, historical)
                    if not checkpoint.is_file():
                        missing.append({"family": family, "seed": seed, "update": update,
                                        "missing": str(checkpoint.name)})
                        print(f"  MISSING {family} seed{seed} u{update}", flush=True)
                        continue
                    policy = services.policy(checkpoint)
                    scored = np.asarray(
                        policy.score(index, batch_size=batch)["sum_log_probability"],
                        dtype=np.float64)
                    del policy
                    services.release_transient()
                    drop = parent_scores - scored
                    origin = "scored here on the flight's audit bank"
                    np.savez_compressed(
                        out / "preservation_curve" / f"{family}@u{update}_seed{seed}.npz",
                        parent_sum_log_probability=parent_scores,
                        policy_sum_log_probability=scored, log_ratio=drop, core_index=index)
                block = tail_block(drop)
                records.append({"family": family, "seed": seed, "update": update,
                                "origin": origin, "bank": source.name, "drop": block})
                print("  seed%d %-14s u%-5d mean %.4f  tenfold %.5f  hundredfold %.5f  %s" % (
                    seed, family, update, block["mean_drop"], block["tenfold_rate"],
                    block["hundredfold_rate"],
                    "(reused)" if origin.startswith("flight") else ""), flush=True)

    # -- per-family curves, averaged over seeds ------------------------------
    curves = []
    for family, updates in GRID.items():
        points = []
        for update in updates:
            values = [r for r in records if r["family"] == family and r["update"] == update]
            if not values:
                continue
            points.append({
                "update": update, "seeds": len(values),
                "mean_drop": float(np.mean([v["drop"]["mean_drop"] for v in values])),
                "mean_drop_per_seed": [v["drop"]["mean_drop"] for v in values],
                "tenfold_rate": float(np.mean([v["drop"]["tenfold_rate"] for v in values])),
                "hundredfold_rate": float(np.mean([v["drop"]["hundredfold_rate"]
                                                   for v in values]))})
        if len(points) < 2:
            continue
        first, last = points[0], points[-1]
        differences = [b - a for a, b in zip(first["mean_drop_per_seed"],
                                             last["mean_drop_per_seed"])]
        curves.append({"family": family, "points": points,
                       "last_minus_first": metrics.paired_t(differences)
                       if len(differences) >= 2 else None,
                       "span": f"u{first['update']} to u{last['update']}"})

    document = {"record_kind": "preservation_over_training",
                "bank": ("the flight's own 50,000-row final_preservation bank, one per parent "
                         "seed, shared by every Block A arm at that seed"),
                "reuse": ("u1000 and u3750 come from the flight's persisted per-row arrays; "
                          "only the intermediate updates were scored here"),
                "convention": "inclusive tail events, drop >= ln(10) and drop >= ln(100)",
                "interval": "paired t over the three parent seeds, df = 2",
                "grid": {k: list(v) for k, v in GRID.items()},
                "results": records, "curves": curves, "missing": missing,
                "recorded_at": paths.utc_now()}
    paths.write_json(out / "preservation_curve.json", document)

    print("\n%-16s %s" % ("family", "  ".join("%9s" % f"u{u}" for u in (250, 500, 1000, 2000, 3750))))
    for label, field in (("mean drop", "mean_drop"), ("tenfold", "tenfold_rate")):
        print("-- %s" % label)
        for curve in curves:
            by_update = {p["update"]: p[field] for p in curve["points"]}
            print("%-16s %s" % (curve["family"], "  ".join(
                ("%9.4f" % by_update[u]) if u in by_update else "%9s" % "-"
                for u in (250, 500, 1000, 2000, 3750))))
    print("\nlast minus first, mean drop (paired t over seeds, df = 2)")
    for curve in curves:
        block = curve["last_minus_first"]
        if block is None:
            continue
        print("  %-16s %-14s %+.4f [%+.4f, %+.4f] %s" % (
            curve["family"], curve["span"], block["mean"], block["lower"], block["upper"],
            "*" if block["excludes_zero"] else ""))
    print(f"\nwrote {out / 'preservation_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
