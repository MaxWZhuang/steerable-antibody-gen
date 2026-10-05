#!/usr/bin/env python
"""Full-sequence G-versus-alternative odds, on frozen backgrounds.

Reads the row set frozen by ``her2_site1_freeze.py`` -- 2,000 complete
nine-residue backgrounds x 20 site-1 residues -- verifies its digest, and scores
every row under the parent and every arm at u3750, three seeds each.

The estimand, per background ``b`` and alternative ``X``:

    D(b, X, model) = log p_model(b with G at site 1) - log p_model(b with X)

and the contrast is ``D(b, X, arm) - D(b, X, that arm's own parent)``, paired
within ``(background, alternative)``.

**Why this and not the marginal.** The +2 pp site-1 frequency shift is a
marginal, and a marginal can move with no change in any conditional: an arm that
merely visits G-friendlier backgrounds more often shifts the marginal while
``p(G | background)`` is identical everywhere. Holding the background fixed by
construction closes that loophole. If the log odds does not move, the frequency
shift was reweighting.

The comparison is also reported split by background source (measured library vs
parent-generated, disjoint by construction) and by the background's own distance
to wild type, because a G effect that only appears near wild type is a
wild-type-proximity effect wearing a residue's name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

SEEDS = (20260918, 20260919, 20260920)
UPDATE = 3750
ARMS = ("continued_sft", "IPO_0", "IPO_FKL", "IPO_TAIL", "FB_fixed", "FB_fresh")
TAIL = {"IPO_TAIL", "FB_fixed", "FB_fresh"}


def checkpoint_for(family, seed, run_root, historical, experiment, repository_root):
    if family == "parent":
        return (repository_root / "outputs" / "her2_posttrain_20260918"
                / f"policy_sft_seed{seed}" / "epoch_3.pt")
    if family in ("FB_fixed", "FB_fresh"):
        arm = family.split("_")[1]
        return experiment / "trajectories" / f"FB_{arm}_seed{seed}" / f"endpoint_update{UPDATE}.pt"
    if family == "IPO_TAIL":
        return run_root / "trajectories" / f"A_IPO_TAIL_seed{seed}" / f"endpoint_update{UPDATE}.pt"
    name = {"IPO_0": "ipo_lambda0", "IPO_FKL": "ipo_lambda10",
            "continued_sft": "continued_sft_lambda0p1"}[family]
    return historical / "trajectories" / f"{name}_seed{seed}" / f"endpoint_update{UPDATE}.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--historical", default="outputs/her2_parent_replay_20260920")
    parser.add_argument("--experiment", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--output", default="outputs/her2_site1_20260923")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    out = ROOT / args.output
    freeze = json.loads((out / "site1_freeze.json").read_text(encoding="utf-8"))
    table = freeze["table"]
    digest = hashlib.sha256("\n".join(f"{r['source']}:{r['background_id']}:{r['core']}"
                                      for r in table).encode()).hexdigest()
    if digest != freeze["rows_sha256"]:
        raise SystemExit("the frozen row set does not reproduce its digest; refusing to score it")
    print("freeze verified: %d rows, %s" % (len(table), digest[:16]), flush=True)

    index = data_lib.encode_cores([r["core"] for r in table])
    source = np.array([r["source"] for r in table])
    background = np.array([r["background_id"] for r in table])
    residue = np.array([r["residue"] for r in table])
    background_distance = np.array([r["background_wt_distance"] for r in table])
    is_target = residue == freeze["target_residue"]

    context = spec.resolve_context(ROOT, config_path=ROOT / args.config,
                                   run_root=ROOT / args.run_root)
    services = campaign.build_services(context, device=args.device)
    batch = int(context.config["inference"]["score_batch_size"])

    # key (source, background_id) -> row positions, so the odds are exact lookups
    target_row = {}
    for position in np.flatnonzero(is_target):
        target_row[(source[position], background[position])] = position

    scores, missing = {}, []
    for family in ("parent",) + ARMS:
        for seed in SEEDS:
            checkpoint = checkpoint_for(family, seed, ROOT / args.run_root,
                                        ROOT / args.historical, ROOT / args.experiment,
                                        context.repository_root)
            if not Path(checkpoint).is_file():
                missing.append({"family": family, "seed": seed, "path": str(checkpoint)})
                print(f"  MISSING {family} seed{seed}", flush=True)
                continue
            policy = services.policy(Path(checkpoint))
            block = policy.score(index, batch_size=batch)
            scores[(family, seed)] = np.asarray(block["sum_log_probability"], dtype=np.float64)
            del policy
            services.release_transient()
            print("  %-14s seed%d scored %d rows" % (family, seed, len(table)), flush=True)

    # -- log odds per (background, alternative) ------------------------------
    alternative = np.flatnonzero(~is_target)
    target_for_alternative = np.array([target_row[(source[p], background[p])]
                                       for p in alternative])

    def log_odds(vector):
        return vector[target_for_alternative] - vector[alternative]

    results, per_arm = [], []
    for family in ARMS:
        per_seed, shares = [], []
        by_source, by_distance = {}, {}
        for seed in SEEDS:
            if (family, seed) not in scores or ("parent", seed) not in scores:
                continue
            arm_odds = log_odds(scores[(family, seed)])
            parent_odds = log_odds(scores[("parent", seed)])
            delta = arm_odds - parent_odds
            per_seed.append(float(delta.mean()))
            shares.append(float((delta > 0).mean()))
            for name in np.unique(source[alternative]):
                mask = source[alternative] == name
                by_source.setdefault(name, []).append(float(delta[mask].mean()))
            for value in np.unique(background_distance[alternative]):
                mask = background_distance[alternative] == value
                if mask.sum() >= 200:
                    by_distance.setdefault(int(value), []).append(float(delta[mask].mean()))
            results.append({"family": family, "seed": seed,
                            "mean_log_odds_shift": float(delta.mean()),
                            "share_positive": float((delta > 0).mean()),
                            "median_log_odds_shift": float(np.median(delta)),
                            "cells": int(delta.size)})
        if len(per_seed) < 2:
            continue
        block = metrics.paired_t(per_seed)
        per_arm.append({
            "family": family, "is_tail_family": family in TAIL,
            "mean_log_odds_shift": block["mean"],
            "lower": block["lower"], "upper": block["upper"],
            "excludes_zero": block["excludes_zero"],
            "per_seed": per_seed,
            "share_positive_mean": float(np.mean(shares)),
            "share_positive_per_seed": shares,
            "by_background_source": {k: {"per_seed": v, "mean": float(np.mean(v))}
                                     for k, v in by_source.items()},
            "by_background_wt_distance": {str(k): {"per_seed": v, "mean": float(np.mean(v))}
                                          for k, v in sorted(by_distance.items())},
        })
        print("%-14s log-odds shift %+.4f [%+.4f, %+.4f] %s   share positive %.3f" % (
            family, block["mean"], block["lower"], block["upper"],
            "*" if block["excludes_zero"] else " ", float(np.mean(shares))), flush=True)

    document = {
        "record_kind": "site1_fixed_background_odds",
        "freeze": {"rows_sha256": freeze["rows_sha256"], "backgrounds": freeze["backgrounds"],
                   "alternatives": freeze["alternatives"]},
        "estimand": freeze["estimand"],
        "contrast": freeze["primary_contrast"],
        "interval": "paired t over the three parent seeds, df = 2",
        "reading": ("a shift in this log odds is a within-background change in the model's "
                    "preference for G, which the marginal frequency shift alone cannot "
                    "establish. No size was predicted, so size is described, not tested."),
        "per_arm": per_arm, "per_seed": results, "missing": missing,
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "site1_odds.json", document)

    print("\nby background source (mean log-odds shift over seeds)")
    for block in per_arm:
        print("  %-14s %s" % (block["family"], "  ".join(
            "%s %+.4f" % (k, v["mean"]) for k, v in sorted(block["by_background_source"].items()))))
    print(f"\nwrote {out / 'site1_odds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
