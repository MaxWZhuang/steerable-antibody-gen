#!/usr/bin/env python
"""Per-arm generated residue frequencies, and whether the shift replicates across seeds.

`site_atlas.csv` carries `model_frequency_status: pending_gpu_regeneration`
because the flight's coupling stage persisted its banks' digests but not their
arrays. `her2_matched_table_fill.py` now saves the 50,000-row sampled indices, so
this closes that column without new GPU time.

**The question it is pointed at.** One-step pair corrections do NOT replicate
across seeds -- `pair_consistency.json` shows the overlap sitting at the
independent-seeds null, so the aggregate ordering gain is a rate shift rather
than a per-pair behavior. A behavior worth measuring independently therefore has
to be defined at the distribution level. The natural candidate is what the arms
generate: if preference training moves a particular site toward a particular
residue, and that move has the same sign and rough size in all three seeds, that
IS a stable behavior, and it is one an ordered-and-assayed panel could test.

So the deliverable is not the frequency table alone. It is, per arm and site and
residue, the shift from that seed's OWN parent bank, and whether it agrees in
sign across all three seeds more than chance allows.

Every bank is 50,000 draws at temperature 1.0 with duplicates retained, so all
arms are size-matched by construction and no subsampling is needed.

Generated frequencies are what the model PRODUCES. They are not measured
functional data and nothing here attaches a binding claim to a residue.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

SEEDS = (20260918, 20260919, 20260920)
ARMS = ("continued_sft", "IPO_0", "IPO_FKL", "IPO_TAIL", "FB_fixed", "FB_fresh")
UPDATE = 3750
#: a shift this small is not worth calling a behavior whatever its interval
MIN_SHIFT_PP = 1.0


def load_bank(directory, family, update, seed):
    path = directory / f"{family}_u{update}_seed{seed}.npz"
    if not path.is_file():
        return None
    return np.asarray(np.load(path)["index"])


def frequencies(index):
    """(10, 20) residue frequency per site, from an integer core index."""
    counts = np.zeros((data_lib.CORE_LENGTH, len(data_lib.CANONICAL)), dtype=np.float64)
    for site in range(data_lib.CORE_LENGTH):
        values, totals = np.unique(index[:, site], return_counts=True)
        counts[site, values] = totals
    return counts / index.shape[0], counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", default="outputs/her2_matched_table_20260923/banks")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    args = parser.parse_args(argv)

    banks, out = ROOT / args.banks, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    parent = {}
    for seed in SEEDS:
        index = load_bank(banks, "parent", 0, seed)
        if index is None:
            print(f"  MISSING parent bank seed{seed}", flush=True)
            continue
        parent[seed] = frequencies(index)
    if not parent:
        raise SystemExit("no parent banks; run scripts/her2_matched_table_fill.py first")

    rows, shifts = [], []
    for family in ARMS:
        per_seed = {}
        for seed in SEEDS:
            index = load_bank(banks, family, UPDATE, seed)
            if index is None or seed not in parent:
                continue
            per_seed[seed] = frequencies(index)
        if not per_seed:
            print(f"  MISSING every bank for {family}", flush=True)
            continue
        for site in range(data_lib.CORE_LENGTH):
            for code, residue in enumerate(data_lib.CANONICAL):
                arm_values = [per_seed[s][0][site, code] for s in sorted(per_seed)]
                parent_values = [parent[s][0][site, code] for s in sorted(per_seed)]
                deltas = [100.0 * (a - p) for a, p in zip(arm_values, parent_values)]
                if len(deltas) < 2:
                    continue
                block = metrics.paired_t(deltas)
                same_sign = (all(d > 0 for d in deltas) or all(d < 0 for d in deltas))
                rows.append({
                    "arm": family, "update": UPDATE, "site": site, "residue": residue,
                    "wild_type": data_lib.WT_CORE[site] == residue,
                    "arm_frequency_mean": float(np.mean(arm_values)),
                    "parent_frequency_mean": float(np.mean(parent_values)),
                    "shift_pp_mean": block["mean"],
                    "shift_pp_lower": block["lower"], "shift_pp_upper": block["upper"],
                    "shift_pp_per_seed": deltas,
                    "seeds": len(deltas),
                    "same_sign_all_seeds": same_sign,
                    "excludes_zero": block["excludes_zero"],
                    "stable": bool(same_sign and block["excludes_zero"]
                                   and abs(block["mean"]) >= MIN_SHIFT_PP),
                })
        stable = [r for r in rows if r["arm"] == family and r["stable"]]
        shifts.append({"arm": family, "cells": 200, "seeds": len(per_seed),
                       "stable_shifts": len(stable),
                       "largest": sorted(stable, key=lambda r: -abs(r["shift_pp_mean"]))[:8]})
        print("%-14s seeds %d   stable site/residue shifts (|>=%.1f pp|, same sign 3/3, "
              "interval excludes 0): %d of 200" % (
                  family, len(per_seed), MIN_SHIFT_PP, len(stable)), flush=True)

    fields = ["arm", "update", "site", "residue", "wild_type", "arm_frequency_mean",
              "parent_frequency_mean", "shift_pp_mean", "shift_pp_lower", "shift_pp_upper",
              "seeds", "same_sign_all_seeds", "excludes_zero", "stable"]
    with (out / "generated_residue_shifts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    document = {
        "record_kind": "generated_residue_frequencies",
        "closes": "site_atlas.csv model_frequency_status = pending_gpu_regeneration",
        "banks": ("50,000 draws per arm per seed at temperature 1.0, duplicates retained, "
                  "drawn by scripts/her2_matched_table_fill.py. Size-matched by construction."),
        "question": ("whether preference training moves a site toward a residue with the SAME "
                     "sign and a comparable size in all three seeds. One-step pair corrections "
                     "do not replicate (see pair_consistency.json), so a distribution-level "
                     "shift is the remaining candidate for a stable behavior."),
        "stable_rule": (f"same sign in all three seeds AND paired-t interval excluding zero AND "
                        f"|mean shift| >= {MIN_SHIFT_PP} percentage points. The size floor is "
                        "there because a 0.1 pp shift with a tight interval is reproducible and "
                        "uninteresting."),
        "interval": "paired t over the three parent seeds, df = 2",
        "claim_limit": ("these are frequencies the model GENERATES. They are not measured "
                        "functional data and no binding claim attaches to a residue here."),
        "per_arm": shifts, "cells": rows, "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "generated_residue_shifts.json", document)

    print("\nlargest stable shifts, by arm")
    for block in shifts:
        if not block["largest"]:
            print("  %-14s none" % block["arm"])
            continue
        print("  %-14s %s" % (block["arm"], "  ".join(
            "site%d %s %+.1fpp" % (r["site"], r["residue"], r["shift_pp_mean"])
            for r in block["largest"][:5])))
    print(f"\nwrote {out / 'generated_residue_shifts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
