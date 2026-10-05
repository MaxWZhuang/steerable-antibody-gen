#!/usr/bin/env python
"""Ranking at the frozen fixed-vs-fresh checkpoints.

The other half of the predeclared primary comparison. Both arms are scored on
the SAME rows -- the original validation population, the same rows every other
ranking number in this flight is computed on -- with the same
``stratified_rank_block`` the campaign uses, so the macro-AP here is comparable
with the flight's own table.

Predeclared tolerance: an arm difference of 0.005 macro-AP or less means the
ranking benefit survived the intervention (P4).

Nothing here selects a checkpoint. u1000 and u3750 were frozen before training.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_proximity as proximity  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

TOLERANCE = 0.005
UPDATES = (1000, 3750)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--experiment", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    experiment = ROOT / args.experiment
    context = spec.resolve_context(ROOT, config_path=ROOT / args.config,
                                   run_root=ROOT / args.run_root)
    services = campaign.build_services(context, device=args.device)

    frame = services.split("val")
    index = services.index("val")
    positive = np.asarray(frame["class"]) == "high"
    strata = services.original_strata()["labels"]
    batch = int(context.config["inference"]["score_batch_size"])

    results = []
    for seed in seeds:
        parent_checkpoint, _ = services.parent_checkpoint(seed)
        jobs = [("parent", 0, parent_checkpoint)]
        for arm in ("fixed", "fresh"):
            for update in UPDATES:
                jobs.append((arm, update, experiment / "trajectories"
                             / f"FB_{arm}_seed{seed}" / f"endpoint_update{update}.pt"))
        for arm, update, checkpoint in jobs:
            if not Path(checkpoint).is_file():
                results.append({"arm": arm, "seed": seed, "update": update,
                                "status": "missing"})
                continue
            policy = services.policy(Path(checkpoint))
            scored = policy.score(index, batch_size=batch)
            ranking = np.asarray(scored["mean_log_probability"], dtype=np.float64)
            block = metrics.stratified_rank_block(ranking, positive, strata,
                                                  categories=proximity.ORIGINAL_STRATA)
            del policy
            services.release_transient()
            results.append({"arm": arm, "seed": seed, "update": update, "status": "scored",
                            "macro_average_precision": block["macro_average_precision"],
                            "pooled_average_precision": block["pooled"]["average_precision"],
                            "pooled_auroc": block["pooled"]["auroc"],
                            "per_stratum": {k: v["average_precision"]
                                            for k, v in block["per_stratum"].items()}})
            print("  seed%d %-6s u%-5d macro-AP %.4f  pooled %.4f" % (
                seed, arm, update, block["macro_average_precision"],
                block["pooled"]["average_precision"]), flush=True)

    scored = {(r["arm"], r["seed"], r["update"]): r["macro_average_precision"]
              for r in results if r["status"] == "scored"}
    contrasts = []
    for update in UPDATES:
        pairs = [(scored[("fresh", s, update)] - scored[("fixed", s, update)])
                 for s in seeds if ("fresh", s, update) in scored
                 and ("fixed", s, update) in scored]
        if len(pairs) < 2:
            continue
        block = metrics.paired_t(pairs)
        contrasts.append({
            "update": update, "contrast": "fresh minus fixed",
            "per_seed": pairs, "mean": block["mean"],
            "lower": block["lower"], "upper": block["upper"],
            "excludes_zero": block["excludes_zero"],
            "within_declared_tolerance": bool(abs(block["mean"]) <= TOLERANCE),
            "tolerance": TOLERANCE,
            "reading": ("the predeclared test is whether the ranking benefit SURVIVES the "
                        "intervention, so a difference inside the tolerance supports the "
                        "fresh stream rather than refuting it"),
        })

    document = {"record_kind": "fixed_vs_fresh_ranking",
                "population": "the original validation split, all 78,652 rows",
                "metric": "macro average precision over the original proximity strata",
                "tolerance": TOLERANCE, "results": results, "contrasts": contrasts,
                "recorded_at": paths.utc_now()}
    paths.write_json(experiment / "ranking.json", document)

    print("\n%-6s %8s %10s %10s %10s" % ("upd", "seed", "fixed", "fresh", "difference"))
    for update in UPDATES:
        for seed in seeds:
            if ("fresh", seed, update) not in scored:
                continue
            print("%-6d %8d %10.4f %10.4f %+10.4f" % (
                update, seed, scored[("fixed", seed, update)],
                scored[("fresh", seed, update)],
                scored[("fresh", seed, update)] - scored[("fixed", seed, update)]))
    print()
    for block in contrasts:
        print("u%-5d fresh-minus-fixed %+.4f [%+.4f, %+.4f]  |difference| <= %.3f : %s" % (
            block["update"], block["mean"], block["lower"], block["upper"],
            TOLERANCE, block["within_declared_tolerance"]))
    parents = {s: scored.get(("parent", s, 0)) for s in seeds}
    print("\nparent macro-AP: " + "  ".join(f"seed{s}: {v:.4f}" for s, v in parents.items()
                                            if v is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
