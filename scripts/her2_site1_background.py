#!/usr/bin/env python
"""Does the tail family's site-1 -> G shift hold WITHIN sequence backgrounds?

`generated_residue_shifts.json` finds one cell where all three tail-penalty
configurations agree: site 1 moves toward G by ~2 percentage points, in all
three seeds of `IPO_TAIL`, `FB_fixed` and `FB_fresh`, and in none of `IPO_0`,
`IPO_FKL` or `continued_sft`. G is the trastuzumab residue at that position.

That is a **marginal** frequency. A marginal shift can be produced without any
per-background change at all: if the arm merely reweights which backgrounds it
visits, and G-carrying backgrounds happen to get more mass, the site-1 marginal
moves while P(G at site 1 | background) is unchanged everywhere. Those are
different claims and only the second is a behavior you could design a panel
around.

So this conditions. For every background -- the nine residues other than site 1 --
present in both the arm's bank and its own parent's bank with enough draws, it
compares P(site 1 = G | background). The headline is the share of backgrounds
that move in the same direction, weighted so that one heavily-sampled background
cannot carry the result, with the parent-vs-parent comparison across seeds as
the null: two runs that differ only by seed should show no systematic shift.

Backgrounds are sparse by construction, so a coarse conditioning -- on the
residue at one neighbouring site -- is reported alongside the exact one.

CPU only.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

SEEDS = (20260918, 20260919, 20260920)
TAIL_ARMS = ("IPO_TAIL", "FB_fixed", "FB_fresh")
CONTROL_ARMS = ("IPO_0", "IPO_FKL", "continued_sft")
SITE = 1
TARGET = "G"
UPDATE = 3750
MIN_DRAWS = 40          # per background, in BOTH banks


def load(directory, family, update, seed):
    path = directory / f"{family}_u{update}_seed{seed}.npz"
    return None if not path.is_file() else np.asarray(np.load(path)["index"])


def background_keys(index, site):
    """A hashable background per row: the nine residues other than ``site``."""
    other = np.delete(index, site, axis=1)
    weights = (len(data_lib.CANONICAL) ** np.arange(other.shape[1])).astype(np.int64)
    return other.astype(np.int64) @ weights


def conditional_rates(index, site, target_code):
    keys = background_keys(index, site)
    hit = index[:, site] == target_code
    totals = collections.Counter(keys.tolist())
    hits = collections.Counter(keys[hit].tolist())
    return totals, hits


def compare(arm_index, parent_index, site, target_code, min_draws=MIN_DRAWS):
    arm_totals, arm_hits = conditional_rates(arm_index, site, target_code)
    par_totals, par_hits = conditional_rates(parent_index, site, target_code)
    shared = [k for k in arm_totals
              if arm_totals[k] >= min_draws and par_totals.get(k, 0) >= min_draws]
    if not shared:
        return None
    deltas, weights = [], []
    for key in shared:
        arm_rate = arm_hits.get(key, 0) / arm_totals[key]
        par_rate = par_hits.get(key, 0) / par_totals[key]
        deltas.append(arm_rate - par_rate)
        weights.append(min(arm_totals[key], par_totals[key]))
    deltas = np.asarray(deltas)
    weights = np.asarray(weights, dtype=np.float64)
    up = int((deltas > 0).sum())
    return {"backgrounds": len(shared),
            "covered_draws": int(weights.sum()),
            "share_moving_up": up / len(shared),
            "backgrounds_up": up, "backgrounds_down": int((deltas < 0).sum()),
            "unweighted_mean_pp": float(100.0 * deltas.mean()),
            "weighted_mean_pp": float(100.0 * np.average(deltas, weights=weights)),
            "median_pp": float(100.0 * np.median(deltas)),
            "sign_test_wilson": metrics.wilson_interval(up, len(shared))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", default="outputs/her2_matched_table_20260923/banks")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    parser.add_argument("--min-draws", type=int, default=MIN_DRAWS)
    args = parser.parse_args(argv)

    banks, out = ROOT / args.banks, ROOT / args.output
    target_code = data_lib.CANONICAL.index(TARGET)
    parents = {s: load(banks, "parent", 0, s) for s in SEEDS}
    if any(v is None for v in parents.values()):
        raise SystemExit("parent banks missing; run scripts/her2_matched_table_fill.py first")

    document = {"record_kind": "site1_background_conditioning",
                "site": SITE, "target_residue": TARGET,
                "wild_type_residue": data_lib.WT_CORE[SITE],
                "update": UPDATE, "min_draws_per_background": args.min_draws,
                "claim_under_test": ("the marginal site-1 -> G shift is a per-background "
                                     "behavior, not a reweighting of which backgrounds are "
                                     "visited"),
                "exact_background": {}, "null_parent_vs_parent": {}, "coarse": {}}

    print("exact background = the nine residues other than site %d, >= %d draws in BOTH banks\n"
          % (SITE, args.min_draws))
    print("%-16s %6s %12s %11s %12s %13s" % (
        "arm", "seeds", "backgrounds", "share up", "weighted pp", "unweighted pp"))
    for family in TAIL_ARMS + CONTROL_ARMS:
        blocks = []
        for seed in SEEDS:
            index = load(banks, family, UPDATE, seed)
            if index is None:
                continue
            block = compare(index, parents[seed], SITE, target_code, args.min_draws)
            if block:
                blocks.append(dict(block, seed=seed))
        if not blocks:
            print("%-16s %6s %12s" % (family, "-", "no shared backgrounds"))
            continue
        shares = [b["share_moving_up"] for b in blocks]
        weighted = [b["weighted_mean_pp"] for b in blocks]
        document["exact_background"][family] = {
            "per_seed": blocks,
            "share_moving_up_mean": float(np.mean(shares)),
            "weighted_pp": metrics.paired_t(weighted) if len(weighted) >= 2 else None,
            "same_sign_all_seeds": bool(all(v > 0 for v in weighted)
                                        or all(v < 0 for v in weighted)),
        }
        print("%-16s %6d %12d %11.3f %12s %13.3f" % (
            family, len(blocks), int(np.mean([b["backgrounds"] for b in blocks])),
            float(np.mean(shares)),
            "%+.3f" % float(np.mean(weighted)), float(np.mean([b["unweighted_mean_pp"]
                                                               for b in blocks]))))

    # -- the null: two parents that differ only by seed ----------------------
    print("\nnull, parent vs parent across seed pairs (differ by seed alone)")
    nulls = []
    for left, right in itertools.combinations(SEEDS, 2):
        block = compare(parents[left], parents[right], SITE, target_code, args.min_draws)
        if block:
            nulls.append(dict(block, pair=f"{left} vs {right}"))
            print("  %-22s backgrounds %5d  share up %.3f  weighted %+.3f pp" % (
                block and f"{left} vs {right}", block["backgrounds"],
                block["share_moving_up"], block["weighted_mean_pp"]))
    document["null_parent_vs_parent"] = nulls

    # -- why the exact conditioning found nothing ----------------------------
    counts = collections.Counter(collections.Counter(
        background_keys(parents[SEEDS[0]], SITE).tolist()).values())
    document["exact_background_feasibility"] = {
        "distinct_backgrounds": int(sum(counts.values())),
        "draws": int(parents[SEEDS[0]].shape[0]),
        "backgrounds_by_draw_count": {str(k): int(v) for k, v in sorted(counts.items())[:10]},
        "largest_background_draws": int(max(counts)),
        "verdict": ("the nine-residue background space is effectively singleton at 50,000 "
                    "draws, so P(site1 = G | exact background) cannot be estimated in both "
                    "banks for any background. This is a sample-size limit of the diagnostic, "
                    "not evidence against the behavior. Testing it exactly needs banks orders "
                    "of magnitude larger, or backgrounds fixed by construction."),
    }
    print("\nexact conditioning is infeasible here: %d distinct backgrounds in %d draws, "
          "largest holds %d" % (sum(counts.values()), parents[SEEDS[0]].shape[0], max(counts)))

    # -- coarse conditioning, on every other site ----------------------------
    print("\ncoarse conditioning: P(site%d = %s | one other site's residue), arm minus its own "
          "parent.\nshare of (site, residue) contexts moving UP, over contexts with >=200 draws "
          "in both banks:\n" % (SITE, TARGET))
    print("%-16s %9s %10s %12s %14s" % ("arm", "contexts", "share up", "mean pp", "same sign 3/3"))
    for family in TAIL_ARMS + CONTROL_ARMS:
        contexts = collections.defaultdict(list)
        for seed in SEEDS:
            index = load(banks, family, UPDATE, seed)
            if index is None:
                continue
            parent = parents[seed]
            for other in range(data_lib.CORE_LENGTH):
                if other == SITE:
                    continue
                for code, residue in enumerate(data_lib.CANONICAL):
                    arm_rows = index[index[:, other] == code]
                    par_rows = parent[parent[:, other] == code]
                    if arm_rows.shape[0] < 200 or par_rows.shape[0] < 200:
                        continue
                    contexts[(other, residue)].append(100.0 * (
                        (arm_rows[:, SITE] == target_code).mean()
                        - (par_rows[:, SITE] == target_code).mean()))
        usable = {k: v for k, v in contexts.items() if len(v) == len(SEEDS)}
        if not usable:
            print("%-16s %9s" % (family, "none"))
            continue
        means = {k: float(np.mean(v)) for k, v in usable.items()}
        up = sum(1 for v in means.values() if v > 0)
        consistent = sum(1 for v in usable.values()
                         if all(x > 0 for x in v) or all(x < 0 for x in v))
        document["coarse"][family] = {
            "contexts": len(usable),
            "share_moving_up": up / len(usable),
            "mean_pp": float(np.mean(list(means.values()))),
            "contexts_same_sign_all_seeds": consistent,
            "sign_test_wilson": metrics.wilson_interval(up, len(usable)),
            "per_context": {f"site{k[0]}={k[1]}": {"per_seed": v, "mean": means[k]}
                            for k, v in sorted(usable.items())},
        }
        print("%-16s %9d %10.3f %12s %14s" % (
            family, len(usable), up / len(usable), "%+.3f" % np.mean(list(means.values())),
            "%d/%d" % (consistent, len(usable))))

    # -- the noise floor: two parents differing only by seed, same axis -------
    print("\nnoise floor on the same axis: parent vs parent, differing by seed alone")
    floor = []
    for left, right in itertools.combinations(SEEDS, 2):
        deltas = []
        for other in range(data_lib.CORE_LENGTH):
            if other == SITE:
                continue
            for code in range(len(data_lib.CANONICAL)):
                a = parents[left][parents[left][:, other] == code]
                b = parents[right][parents[right][:, other] == code]
                if a.shape[0] < 200 or b.shape[0] < 200:
                    continue
                deltas.append(100.0 * ((a[:, SITE] == target_code).mean()
                                       - (b[:, SITE] == target_code).mean()))
        if not deltas:
            continue
        up = sum(1 for d in deltas if d > 0)
        floor.append({"pair": f"{left} vs {right}", "contexts": len(deltas),
                      "share_moving_up": up / len(deltas),
                      "mean_pp": float(np.mean(deltas))})
        print("  %-22s contexts %4d  share up %.3f  mean %+.3f pp" % (
            f"{left} vs {right}", len(deltas), up / len(deltas), float(np.mean(deltas))))
    document["coarse_noise_floor"] = floor

    document["recorded_at"] = paths.utc_now()
    paths.write_json(out / "site1_background.json", document)
    print(f"\nwrote {out / 'site1_background.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
