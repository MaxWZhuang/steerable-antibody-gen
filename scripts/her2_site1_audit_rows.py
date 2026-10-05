#!/usr/bin/env python
"""Does site-1 G explain severe preservation loss, once difficulty is controlled?

The audit banks are parent draws, so each of the 50,000 rows carries the
parent's own site-1 residue, the parent's log probability, the policy's, and
their difference. That makes it possible to ask whether an arm's severe losses
avoid the rows the arm also likes to generate -- which would tie the generation
behavior to preservation rather than leaving them as two separate observations.

Two confounders have to be removed before the question means anything:

* **parent probability.** A row the parent finds unlikely has more room to lose
  probability and is over-represented in any tail. If G rows are systematically
  higher-probability under the parent, a raw G/non-G tail gap is that, not G.
* **distance to wild type.** G is the wild-type residue at site 1, so G rows are
  one mutation closer to wild type on average, and wild-type-proximate rows
  behave differently throughout this project.

So the raw difference is reported, then the same difference within strata of
(parent-probability decile x wild-type distance), then a logistic fit whose G
coefficient is adjusted for both. Three views because the stratified estimate is
the honest one and the other two say how much the adjustment mattered.

CPU only; reads the persisted per-row audit arrays.
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

SEEDS = (20260918, 20260919, 20260920)
SITE, TARGET = 1, "G"
TENFOLD = math.log(10.0)
DECILES = 10
MIN_STRATUM = 50
ARMS = {
    "IPO_TAIL": "A_IPO_TAIL@u{u}_seed{s}.npz",
    "IPO_0": "A_IPO_0@u{u}_seed{s}.npz",
    "IPO_FKL": "A_IPO_FKL@u{u}_seed{s}.npz",
    "continued_sft": "continued_sft@u{u}_seed{s}.npz",
    "FB_fixed": "FB_fixed@u{u}_seed{s}.npz",
    "FB_fresh": "FB_fresh@u{u}_seed{s}.npz",
}
TAIL = {"IPO_TAIL", "FB_fixed", "FB_fresh"}


def stratified_difference(event, group, strata, min_stratum=MIN_STRATUM):
    """Weighted mean of the within-stratum rate difference, groups G vs non-G.

    Weighting is by the harmonic-style min of the two group sizes so a stratum
    that is almost entirely one group cannot dominate a difference it barely
    informs.
    """
    total_weight, total = 0.0, 0.0
    used, skipped = 0, 0
    for key in np.unique(strata):
        mask = strata == key
        left, right = mask & group, mask & (~group)
        if left.sum() < min_stratum or right.sum() < min_stratum:
            skipped += 1
            continue
        weight = float(min(left.sum(), right.sum()))
        total += weight * (event[left].mean() - event[right].mean())
        total_weight += weight
        used += 1
    if total_weight == 0:
        return None
    return {"difference": total / total_weight, "strata_used": used,
            "strata_skipped": skipped, "weight": total_weight}


def logistic_adjusted(event, group, parent_score, distance):
    """G coefficient in logit(event) ~ G + parent log p + wild-type distance."""
    from sklearn.linear_model import LogisticRegression

    design = np.column_stack([group.astype(np.float64), parent_score, distance])
    design = (design - design.mean(axis=0)) / np.maximum(design.std(axis=0), 1e-12)
    if len(np.unique(event)) < 2:
        return None
    model = LogisticRegression(penalty=None, max_iter=500)
    model.fit(design, event.astype(int))
    return {"g_coefficient_standardized": float(model.coef_[0][0]),
            "parent_score_coefficient": float(model.coef_[0][1]),
            "wt_distance_coefficient": float(model.coef_[0][2]),
            "note": ("coefficients are on standardized columns, so they are comparable "
                     "with each other and are not odds ratios on the raw scale")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--matched", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--output", default="outputs/her2_site1_20260923")
    parser.add_argument("--update", type=int, default=3750)
    args = parser.parse_args(argv)

    run_root, matched = ROOT / args.run_root, ROOT / args.matched
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    target_code = data_lib.CANONICAL.index(TARGET)
    wt_index = data_lib.encode_cores([data_lib.WT_CORE])[0]

    document = {"record_kind": "site1_audit_row_analysis",
                "update": args.update,
                "question": ("whether an arm's severe preservation losses avoid the site-1 G "
                             "rows it also prefers to generate, after removing parent "
                             "probability and wild-type distance"),
                "confounders_removed": ["parent log probability decile", "distance to wild type"],
                "severe_loss": "drop >= ln(10), the inclusive tenfold event",
                "stratum_rule": f"at least {MIN_STRATUM} rows of BOTH groups in a stratum",
                "arms": {}}

    print("%-14s %8s %10s %10s %11s %12s %11s" % (
        "arm", "G rows", "tenfold G", "tenfold ~G", "raw diff", "stratified", "logistic b"))
    for family, pattern in ARMS.items():
        raw, strat, coefficients, g_rates, other_rates, counts = [], [], [], [], [], []
        for seed in SEEDS:
            directory = matched if family.startswith("FB_") else run_root
            sub = "preservation"
            path = directory / sub / pattern.format(u=args.update, s=seed)
            if not path.is_file():
                continue
            arrays = paths.read_arrays(path)
            index = np.asarray(arrays["core_index"])
            drop = np.asarray(arrays["log_ratio"], dtype=np.float64)
            parent_score = np.asarray(arrays["parent_sum_log_probability"], dtype=np.float64)
            group = index[:, SITE] == target_code
            distance = (index != wt_index[None, :]).sum(axis=1).astype(np.float64)
            event = (drop >= TENFOLD)

            decile = np.digitize(parent_score, np.quantile(
                parent_score, np.linspace(0, 1, DECILES + 1)[1:-1]))
            strata = decile * 100 + distance.astype(int)

            g_rates.append(float(event[group].mean()))
            other_rates.append(float(event[~group].mean()))
            raw.append(g_rates[-1] - other_rates[-1])
            counts.append(int(group.sum()))
            block = stratified_difference(event, group, strata)
            if block:
                strat.append(block["difference"])
            fit = logistic_adjusted(event, group, parent_score, distance)
            if fit:
                coefficients.append(fit["g_coefficient_standardized"])
        if not raw:
            continue
        raw_block = metrics.paired_t(raw) if len(raw) >= 2 else None
        strat_block = metrics.paired_t(strat) if len(strat) >= 2 else None
        document["arms"][family] = {
            "is_tail_family": family in TAIL,
            "g_rows_per_seed": counts,
            "tenfold_rate_G": g_rates, "tenfold_rate_nonG": other_rates,
            "raw_difference": raw_block, "stratified_difference": strat_block,
            "logistic_g_coefficient": {
                "per_seed": coefficients,
                "mean": float(np.mean(coefficients)) if coefficients else None},
        }
        print("%-14s %8d %10.5f %10.5f %11s %12s %11s" % (
            family, int(np.mean(counts)), float(np.mean(g_rates)), float(np.mean(other_rates)),
            "%+.5f" % raw_block["mean"] if raw_block else "-",
            ("%+.5f%s" % (strat_block["mean"], "*" if strat_block["excludes_zero"] else ""))
            if strat_block else "-",
            "%+.3f" % np.mean(coefficients) if coefficients else "-"))

    # -- does the G advantage differ between the tail family and the controls?
    tail = [document["arms"][a]["stratified_difference"]["mean"]
            for a in ARMS if a in document["arms"] and a in TAIL
            and document["arms"][a]["stratified_difference"]]
    control = [document["arms"][a]["stratified_difference"]["mean"]
               for a in ARMS if a in document["arms"] and a not in TAIL
               and document["arms"][a]["stratified_difference"]]
    if tail and control:
        document["tail_versus_control"] = {
            "tail_mean": float(np.mean(tail)), "control_mean": float(np.mean(control)),
            "difference": float(np.mean(tail) - np.mean(control)),
            "basis": ("arm-level means of the stratified G-minus-nonG tenfold difference. "
                      "Three arms per side, so this is descriptive: it is not a paired "
                      "seed-level contrast and carries no interval."),
        }
        print("\nstratified G-minus-nonG tenfold difference: tail family %+.5f, controls %+.5f"
              % (np.mean(tail), np.mean(control)))

    document["recorded_at"] = paths.utc_now()
    paths.write_json(out / "site1_audit_rows.json", document)
    print(f"\nwrote {out / 'site1_audit_rows.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
