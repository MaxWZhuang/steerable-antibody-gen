#!/usr/bin/env python
"""Does the one-step gain live in the same pairs across seeds, or a different set each time?

The aggregate is settled: every IPO arm orders ~3 points more of the 1,108
one-step pairs correctly than its own parent, three seeds out of three, interval
excluding zero. That is a statement about a *rate*. It is compatible with two
very different mechanisms:

* **a behavior** -- the same pairs get corrected every time, so there is
  something stable about those sequence backgrounds that preference training
  learns; or
* **a rate shift** -- each seed corrects a different ~3% of pairs, and only the
  count is reproducible.

Only the second is consistent with "the individual corrections rarely repeat",
and the two have completely different consequences for whether there is anything
here worth measuring independently. A rate shift gives you nothing to order and
assay; a behavior does.

This separates them. For each pair and each arm, a **correction** is a pair the
arm's own parent orders wrongly and the arm orders correctly; a **regression** is
the reverse. The test is whether corrections concentrate in the same pairs across
the three seeds far more than independent seeds at the same per-seed rate would
produce -- the null is explicit and computed per arm from that arm's own observed
rates, not assumed.

Then, for whatever does repeat, it asks whether those pairs share structure:
site, the residue substituted in or out, and the class transition.

CPU only; reads the persisted pair rows.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

SEEDS = (20260918, 20260919, 20260920)
ARMS = {"IPO_0": "A_IPO_0_u3750_seed%d", "IPO_FKL": "A_IPO_FKL_u3750_seed%d",
        "IPO_TAIL": "A_IPO_TAIL_u3750_seed%d"}
PARENT = "parent_seed%d"
NULL_DRAWS = 20_000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="outputs/her2_biology_20260923/substitution_pair_rows.csv")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args(argv)

    out = ROOT / args.output
    rows = list(csv.DictReader((ROOT / args.rows).open(encoding="utf-8")))
    n = len(rows)
    print(f"pairs {n}", flush=True)

    def correct(column):
        return np.array([float(r["delta__" + column]) > 0 for r in rows])

    generator = np.random.default_rng(args.seed)
    summary = {"record_kind": "one_step_pair_consistency",
               "pairs": n, "checkpoint": "u3750", "seeds": list(SEEDS),
               "definitions": {
                   "correction": "the arm's own parent orders the pair wrongly and the arm orders it correctly",
                   "regression": "the parent orders it correctly and the arm does not",
                   "null": ("per arm, three independent Bernoulli seeds at that arm's OWN observed "
                            "per-seed correction rate over the pairs its parent gets wrong. This "
                            "asks whether corrections REPEAT, holding the rate fixed."),
               },
               "arms": {}, "shared_structure": {}}

    for arm, pattern in ARMS.items():
        per_seed_correction, per_seed_regression, eligible = [], [], []
        correction_masks = []
        for seed in SEEDS:
            parent_ok = correct(PARENT % seed)
            arm_ok = correct(pattern % seed)
            correction = (~parent_ok) & arm_ok
            regression = parent_ok & (~arm_ok)
            correction_masks.append(correction)
            per_seed_correction.append(int(correction.sum()))
            per_seed_regression.append(int(regression.sum()))
            eligible.append(int((~parent_ok).sum()))
        stack = np.vstack(correction_masks)
        repeats = stack.sum(axis=0)                      # 0..3 seeds correcting each pair
        observed = collections.Counter(int(v) for v in repeats)
        all_three = int((repeats == 3).sum())
        any_one = int((repeats >= 1).sum())

        # Null: each seed independently corrects a random subset of the pairs its
        # own parent gets wrong, at that seed's observed rate.
        null_all_three = np.empty(NULL_DRAWS, dtype=np.int64)
        parent_wrong = [~correct(PARENT % seed) for seed in SEEDS]
        for draw in range(NULL_DRAWS):
            simulated = np.zeros(n, dtype=np.int64)
            for position, seed in enumerate(SEEDS):
                pool = np.flatnonzero(parent_wrong[position])
                chosen = generator.choice(pool, size=per_seed_correction[position],
                                          replace=False)
                simulated[chosen] += 1
            null_all_three[draw] = int((simulated == 3).sum())

        exceed = float((null_all_three >= all_three).mean())
        summary["arms"][arm] = {
            "eligible_pairs_parent_wrong": eligible,
            "corrections_per_seed": per_seed_correction,
            "regressions_per_seed": per_seed_regression,
            "net_per_seed": [c - r for c, r in zip(per_seed_correction, per_seed_regression)],
            "pairs_corrected_by_n_seeds": {str(k): int(v) for k, v in sorted(observed.items())},
            "corrected_by_all_three": all_three,
            "corrected_by_at_least_one": any_one,
            "null_all_three_mean": float(null_all_three.mean()),
            "null_all_three_p95": float(np.quantile(null_all_three, 0.95)),
            "monte_carlo_p_value": exceed,
            "reading": ("if corrected_by_all_three sits at the null mean, the aggregate gain is "
                        "a rate shift and there is no stable per-pair behavior to measure. If it "
                        "greatly exceeds the null, the same backgrounds are being corrected."),
        }
        print("%-9s corrections/seed %-18s all-three %4d   null mean %6.1f (p95 %5.1f)  p=%.4f"
              % (arm, str(per_seed_correction), all_three, null_all_three.mean(),
                 np.quantile(null_all_three, 0.95), exceed), flush=True)

    # -- does anything repeat across ARMS as well as seeds? -------------------
    everywhere = np.ones(len(rows), dtype=bool)
    for arm, pattern in ARMS.items():
        for seed in SEEDS:
            everywhere &= (~correct(PARENT % seed)) & correct(pattern % seed)
    stable = [rows[i] for i in np.flatnonzero(everywhere)]
    summary["corrected_by_all_nine_checkpoints"] = {
        "count": len(stable),
        "of_pairs": n,
        "clone_pairs": [{"pair_id": r["pair_id"], "site": int(r["site"]),
                         "background": r["background"], "strong_core": r["strong_core"],
                         "weak_core": r["weak_core"], "strong_residue": r["strong_residue"],
                         "weak_residue": r["weak_residue"], "transition": r["transition"]}
                        for r in stable],
        "meaning": ("pairs every IPO arm at every seed corrects and every parent gets wrong. "
                    "This is the candidate set for a behavior that survives seeds AND arms."),
    }
    print("\ncorrected by ALL nine IPO checkpoints and missed by all three parents: %d of %d"
          % (len(stable), n), flush=True)

    if stable:
        for field in ("site", "transition", "strong_residue", "weak_residue"):
            counts = collections.Counter(r[field] for r in stable)
            overall = collections.Counter(r[field] for r in rows)
            summary["shared_structure"][field] = {
                "stable": dict(counts),
                "all_pairs": dict(overall),
                "enrichment": {k: round((counts[k] / len(stable))
                                        / (overall[k] / n), 3) for k in counts},
            }
            print("  %-16s %s" % (field, ", ".join(
                "%s:%d(x%.2f)" % (k, v, (v / len(stable)) / (overall[k] / n))
                for k, v in counts.most_common(6))))
        with (out / "stable_corrections.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stable[0]))
            writer.writeheader()
            writer.writerows(stable)

    summary["recorded_at"] = paths.utc_now()
    paths.write_json(out / "pair_consistency.json", summary)
    print(f"\nwrote {out / 'pair_consistency.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
