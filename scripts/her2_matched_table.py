#!/usr/bin/env python
"""One matched comparison of the candidate arms, at common checkpoints.

The results this table replaces sit in four reports at three different
checkpoint sets. This puts them on one page, per seed, at the two endpoints every
candidate actually reached -- **u1000 and u3750**.

**Recomputed, not joined.** Every column that can be is derived here from the
persisted per-row arrays under one convention, rather than lifted from each
report's summary fields:

* macro-AP from ``row_scores/*.npz`` via the same ``stratified_rank_block`` over
  the original proximity strata;
* expected distinct known-high yield from the same arrays, ``Y(N) = sum_x
  [1-(1-q_x)^N]`` over the 25,822 distinct high identities of the validation
  panel, at the flight's primary budget of 10,000 draws;
* independent-audit mean drop and the inclusive tenfold/hundredfold rates from
  ``preservation/*.npz``, with Wilson intervals.

Two columns cannot be recomputed and are joined, with their provenance stated:
off-library fraction (needs the generated banks, drawn by
``her2_matched_table_fill.py``) and cost (per-trajectory, from ``cost.json``).

The recomputed macro-AP is cross-checked against the flight's own
``report.rankings`` for every cell that appears in both. A disagreement beyond
1e-9 means the two are not the same statistic and the table says so rather than
quietly preferring one.

The arms, as requested: parent, continued SFT, IPO without preservation,
forward-KL replay, tail penalty, fixed versus fresh tail, and the parent mixture.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_mixture as mixture_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_proximity as proximity  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

TENFOLD, HUNDREDFOLD = math.log(10.0), math.log(100.0)
YIELD_BUDGET = 10_000
UPDATES = (1000, 3750)
#: display order, and the label each family carries in the report
FAMILIES = [
    ("parent", "parent (initial SFT)"),
    ("continued_sft", "continued SFT (lambda 0.1)"),
    ("IPO_0", "IPO, no preservation"),
    ("IPO_FKL", "IPO + forward-KL replay (lambda 10)"),
    ("IPO_TAIL", "IPO + tail penalty (lambda 1)"),
    ("FB_fixed", "tail, fixed 100k bank"),
    ("FB_fresh", "tail, fresh 240k stream"),
    ("mixture089", "parent mixture, alpha 0.89"),
]


def row_score_path(family, update, seed, run_root, matched):
    if family == "parent":
        return run_root / "row_scores" / f"parent_seed{seed}.npz"
    if family == "mixture089":
        return run_root / "row_scores" / f"mixture_ipo0_u{update}_seed{seed}.npz"
    if family in ("continued_sft",):
        return run_root / "row_scores" / f"continued_sft_u{update}_seed{seed}.npz"
    if family in ("FB_fixed", "FB_fresh"):
        return matched / "row_scores" / f"{family}_u{update}_seed{seed}.npz"
    return run_root / "row_scores" / f"A_{family}_u{update}_seed{seed}.npz"


def preservation_path(family, update, seed, run_root, matched):
    if family == "parent":
        return run_root / "preservation" / f"parent_seed{seed}.npz"
    if family == "continued_sft":
        return run_root / "preservation" / f"continued_sft@u{update}_seed{seed}.npz"
    if family in ("FB_fixed", "FB_fresh"):
        return matched / "preservation" / f"{family}@u{update}_seed{seed}.npz"
    if family == "mixture089":
        return None          # derived from its two components below
    return run_root / "preservation" / f"A_{family}@u{update}_seed{seed}.npz"


def tail_block(drop):
    ten, hundred = int((drop >= TENFOLD).sum()), int((drop >= HUNDREDFOLD).sum())
    rows = int(drop.size)
    return {"rows": rows, "mean_drop": float(drop.mean()),
            "standard_error": float(drop.std(ddof=1) / math.sqrt(rows)),
            "fraction_below_parent": float((drop > 0).mean()),
            "tenfold_rate": ten / rows, "tenfold_count": ten,
            "tenfold_wilson": metrics.wilson_interval(ten, rows),
            "hundredfold_rate": hundred / rows, "hundredfold_count": hundred,
            "hundredfold_wilson": metrics.wilson_interval(hundred, rows),
            "convention": "inclusive: drop >= ln(10), drop >= ln(100)"}


def cost_for(family, seed, run_root, historical, experiment):
    """Wall seconds of the trajectory that produced the checkpoint.

    No per-update timing is journalled, so a u1000 checkpoint taken from a
    3,750-update trajectory cannot be costed separately from its own trajectory.
    The figure is the trajectory's, and the table says so rather than prorating
    by update count, which would invent an attribution the journals do not carry.
    """
    if family == "parent":
        return {"wall_seconds": 0.0, "basis": "already existed; no cost is charged here"}
    if family == "mixture089":
        return {"wall_seconds": None,
                "basis": ("no training of its own -- it is the IPO_0 checkpoint plus the "
                          "parent. Its operating cost is two model files and a component "
                          "draw per sequence, not GPU seconds")}
    if family in ("FB_fixed", "FB_fresh"):
        arm = family.split("_")[1]
        target = experiment / "trajectories" / f"FB_{arm}_seed{seed}" / "cost.json"
    elif family == "IPO_TAIL":
        target = run_root / "trajectories" / f"A_IPO_TAIL_seed{seed}" / "cost.json"
    else:
        name = {"IPO_0": "ipo_lambda0", "IPO_FKL": "ipo_lambda10",
                "continued_sft": "continued_sft_lambda0p1"}[family]
        status = historical / "trajectories" / f"{name}_seed{seed}" / "status.json"
        if not status.is_file():
            return {"wall_seconds": None, "basis": "not recorded"}
        document = json.loads(status.read_text(encoding="utf-8"))
        return {"wall_seconds": document.get("wall_seconds"),
                "basis": "historical campaign trajectory status.json"}
    if not target.is_file():
        return {"wall_seconds": None, "basis": "not recorded"}
    document = json.loads(target.read_text(encoding="utf-8"))
    return {"wall_seconds": document.get("wall_seconds"),
            "seconds": document.get("seconds"),
            "unattributed_seconds": document.get("unattributed_seconds"),
            "basis": "device-synchronized wall seconds, whole trajectory"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--historical", default="outputs/her2_parent_replay_20260920")
    parser.add_argument("--experiment", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--matched", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--output", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_root, historical = ROOT / args.run_root, ROOT / args.historical
    experiment, matched = ROOT / args.experiment, ROOT / args.matched
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    published = json.loads((run_root / "report.json").read_text(encoding="utf-8"))["rankings"]
    fill_path = matched / "fill.json"
    off_library = {}
    if fill_path.is_file():
        for record in json.loads(fill_path.read_text(encoding="utf-8"))["banks"]:
            off_library[(record["family"], record["seed"], record["update"])] = record

    cells, crosschecks, missing = [], [], []
    for family, label in FAMILIES:
        for update in ((0,) if family == "parent" else UPDATES):
            for seed in seeds:
                path = row_score_path(family, update, seed, run_root, matched)
                if not path.is_file():
                    missing.append({"family": family, "seed": seed, "update": update,
                                    "missing": str(path.name)})
                    continue
                arrays = paths.read_arrays(path)
                if "sum_log_probability" in arrays:
                    sum_log = np.asarray(arrays["sum_log_probability"], dtype=np.float64)
                    mean_log = np.asarray(arrays["mean_log_probability"], dtype=np.float64)
                    strata = np.asarray(arrays["original_train_distance_label"]).astype(str)
                else:
                    # The mixture file stores its two components and the mixed
                    # density under different names and carries no strata column;
                    # the rows are the same panel in the same order, so the
                    # parent's labels apply.
                    sum_log = np.asarray(arrays["mixture_sum_log_probability_primary"],
                                         dtype=np.float64)
                    mean_log = sum_log / 10.0
                    strata = np.asarray(paths.read_arrays(
                        run_root / "row_scores" / f"parent_seed{seed}.npz"
                    )["original_train_distance_label"]).astype(str)
                positive = np.asarray(arrays["positive"]).astype(bool)
                block = metrics.stratified_rank_block(mean_log, positive, strata,
                                                      categories=proximity.ORIGINAL_STRATA)
                curve = metrics.yield_curve(sum_log[positive])

                # cross-check against the flight's own published macro-AP
                key = ({"parent": f"parent_seed{seed}",
                        "mixture089": f"mixture089::A_IPO_0@u{update}_seed{seed}",
                        "continued_sft": f"continued_sft@u{update}_seed{seed}"}
                       .get(family, f"A_{family}@u{update}_seed{seed}"))
                if key in published:
                    difference = (block["macro_average_precision"]
                                  - published[key]["macro_average_precision"])
                    crosschecks.append({"cell": key, "recomputed": block["macro_average_precision"],
                                        "published": published[key]["macro_average_precision"],
                                        "difference": difference,
                                        "agrees": bool(abs(difference) < 1e-9)})

                audit = None
                audit_source = None
                preservation = preservation_path(family, update, seed, run_root, matched)
                if preservation is not None and preservation.is_file():
                    arrays = paths.read_arrays(preservation)
                    audit = tail_block(np.asarray(arrays["log_ratio"], dtype=np.float64))
                    audit_source = preservation.name
                elif family == "mixture089":
                    # Exact: the mixture's density on the audit bank is
                    # logaddexp of its two components, both of which are persisted.
                    parent_file = run_root / "preservation" / f"parent_seed{seed}.npz"
                    policy_file = run_root / "preservation" / f"A_IPO_0@u{update}_seed{seed}.npz"
                    if parent_file.is_file() and policy_file.is_file():
                        parent_scores = np.asarray(
                            paths.read_arrays(parent_file)["parent_sum_log_probability"],
                            dtype=np.float64)
                        policy_scores = np.asarray(
                            paths.read_arrays(policy_file)["policy_sum_log_probability"],
                            dtype=np.float64)
                        mixed = mixture_lib.mixture_log_probability(
                            parent_scores, policy_scores, mixture_lib.PRIMARY_ALPHA)
                        audit = tail_block(parent_scores - mixed)
                        audit_source = ("derived exactly from parent and IPO_0 component "
                                        "scores on the same audit bank")

                bank = off_library.get((family, seed, update))
                if bank is None and family == "mixture089":
                    # A mixture draw comes from ONE component, so the expected
                    # off-library RATE is the alpha-weighted mixture of the two
                    # component rates. Exact for a rate; not for uniqueness.
                    alpha = float(mixture_lib.PRIMARY_ALPHA)
                    parent_bank = off_library.get(("parent", seed, 0))
                    policy_bank = off_library.get(("IPO_0", seed, update))
                    if parent_bank and policy_bank:
                        bank = {"off_library_fraction":
                                alpha * policy_bank["off_library_fraction"]
                                + (1 - alpha) * parent_bank["off_library_fraction"],
                                "unique_fraction": None,
                                "derived": ("alpha-weighted component rates; exact for a "
                                            "per-draw rate, not for uniqueness"),
                                "rows": policy_bank["rows"]}

                cells.append({
                    "family": family, "label": label, "seed": seed, "update": update,
                    "macro_average_precision": block["macro_average_precision"],
                    "pooled_average_precision": block["pooled"]["average_precision"],
                    "yield_at_budget": curve["curve"][str(YIELD_BUDGET)],
                    "yield_identities": curve["identities"],
                    "audit": audit, "audit_source": audit_source,
                    "off_library_fraction": (bank or {}).get("off_library_fraction"),
                    "unique_fraction": (bank or {}).get("unique_fraction"),
                    "off_library_source": ("derived" if (bank or {}).get("derived")
                                           else ("bank" if bank else None)),
                    "cost": cost_for(family, seed, run_root, historical, experiment),
                    "row_scores": path.name})

    # -- seed means and contrasts against the parent --------------------------
    def value(cell, field):
        if field == "audit_mean":
            return None if cell["audit"] is None else cell["audit"]["mean_drop"]
        if field in ("tenfold", "hundredfold"):
            return None if cell["audit"] is None else cell["audit"][f"{field}_rate"]
        return cell.get(field)

    index = {(c["family"], c["seed"], c["update"]): c for c in cells}
    fields = ["macro_average_precision", "yield_at_budget", "audit_mean", "tenfold",
              "hundredfold", "off_library_fraction"]
    summary, contrasts = [], []
    for family, label in FAMILIES:
        for update in ((0,) if family == "parent" else UPDATES):
            present = [index[(family, s, update)] for s in seeds
                       if (family, s, update) in index]
            if not present:
                continue
            block = {"family": family, "label": label, "update": update,
                     "seeds": len(present)}
            for field in fields:
                values = [value(c, field) for c in present]
                values = [v for v in values if v is not None]
                block[field] = {"mean": float(np.mean(values)) if values else None,
                                "per_seed": values} if values else None
            costs = [c["cost"].get("wall_seconds") for c in present
                     if c["cost"].get("wall_seconds") is not None]
            block["wall_seconds"] = {"mean": float(np.mean(costs)),
                                     "per_seed": costs} if costs else None
            summary.append(block)
            if family == "parent":
                continue
            entry = {"family": family, "update": update, "against": "parent"}
            for field in fields:
                pairs = []
                for seed in seeds:
                    cell, base = index.get((family, seed, update)), index.get(("parent", seed, 0))
                    if cell is None or base is None:
                        continue
                    left, right = value(cell, field), value(base, field)
                    if left is None or right is None:
                        continue
                    pairs.append(left - right)
                if len(pairs) >= 2:
                    entry[field] = metrics.paired_t(pairs)
            contrasts.append(entry)

    document = {
        "record_kind": "matched_arm_comparison",
        "checkpoints": list(UPDATES),
        "panel": ("the original validation split, 78,652 rows, 25,822 distinct high "
                  "identities. Its exposure to E is disclosed by the flight."),
        "recomputed_from": ("per-row arrays, not from each report's summary fields, so every "
                            "cell shares one convention"),
        "columns": {
            "macro_average_precision": "macro over the original proximity strata (1, 2, >=3)",
            "yield_at_budget": (f"expected distinct known-high identities at {YIELD_BUDGET:,} "
                                "draws, over the 25,822 high identities of this panel"),
            "audit_mean": ("mean drop on the independent 50,000-row final_preservation bank, "
                           "one bank per parent seed shared by every arm at that seed"),
            "tenfold / hundredfold": "INCLUSIVE rates on the same bank, with Wilson intervals",
            "off_library_fraction": ("of 50,000 generated draws, the fraction whose core is in "
                                     "neither the training nor the validation split; duplicates "
                                     "retained, no sealed-test labels read"),
            "wall_seconds": ("the whole trajectory's device-synchronized wall seconds. No "
                             "per-update timing is journalled, so a u1000 checkpoint cannot be "
                             "costed apart from its own 3,750-update trajectory."),
        },
        "interval_convention": ("contrasts are a paired t over the three parent seeds, df = 2. "
                                "Rate intervals are Wilson on one bank. The two are never pooled."),
        "macro_ap_crosscheck": {
            "checked": len(crosschecks),
            "disagreements": [c for c in crosschecks if not c["agrees"]],
            "max_abs_difference": (max(abs(c["difference"]) for c in crosschecks)
                                   if crosschecks else None),
            "meaning": ("the recomputation reproduces the flight's published macro-AP for every "
                        "cell that appears in both. A disagreement would mean the table and the "
                        "report are not computing the same statistic."),
        },
        "cells": cells, "summary": summary, "contrasts": contrasts, "missing": missing,
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "matched_table.json", document)

    header = ["family", "update", "seed", "macro_AP", "yield_10k", "audit_mean",
              "tenfold", "hundredfold", "off_library", "unique", "wall_seconds"]
    with (out / "matched_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for cell in cells:
            writer.writerow([
                cell["family"], cell["update"], cell["seed"],
                "%.4f" % cell["macro_average_precision"],
                "%.1f" % cell["yield_at_budget"],
                "" if cell["audit"] is None else "%.4f" % cell["audit"]["mean_drop"],
                "" if cell["audit"] is None else "%.5f" % cell["audit"]["tenfold_rate"],
                "" if cell["audit"] is None else "%.5f" % cell["audit"]["hundredfold_rate"],
                "" if cell["off_library_fraction"] is None else "%.4f" % cell["off_library_fraction"],
                "" if cell["unique_fraction"] is None else "%.4f" % cell["unique_fraction"],
                "" if cell["cost"].get("wall_seconds") is None else "%.1f" % cell["cost"]["wall_seconds"]])

    bad = [c for c in crosschecks if not c["agrees"]]
    print("macro-AP cross-check against report.rankings: %d cells, %d disagreements%s" % (
        len(crosschecks), len(bad),
        "" if not bad else "  <-- the table and the report disagree"))
    print("\n%-26s %5s %9s %9s %9s %10s %11s %11s %10s" % (
        "arm", "upd", "macro-AP", "Y@10k", "auditmean", "tenfold", "hundredfold",
        "off-library", "wall s"))
    for block in summary:
        def cell(name, fmt):
            entry = block.get(name)
            return (fmt % entry["mean"]) if entry else "-"
        print("%-26s %5d %9s %9s %9s %10s %11s %11s %10s" % (
            block["family"], block["update"],
            cell("macro_average_precision", "%.4f"), cell("yield_at_budget", "%.1f"),
            cell("audit_mean", "%.4f"), cell("tenfold", "%.5f"),
            cell("hundredfold", "%.5f"), cell("off_library_fraction", "%.4f"),
            ("%.0f" % block["wall_seconds"]["mean"]) if block.get("wall_seconds") else "-"))
    if missing:
        print("\nmissing cells: %d" % len(missing))
        for record in missing[:12]:
            print("  %s seed%d u%d -> %s" % (record["family"], record["seed"],
                                             record["update"], record["missing"]))
    print(f"\nwrote {out / 'matched_table.json'}")
    print(f"wrote {out / 'matched_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
