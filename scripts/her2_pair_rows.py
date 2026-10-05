#!/usr/bin/env python
"""Pair-level one-step substitution rows, with the source split and the additive baseline.

``scripts/her2_biology_cpu.py`` builds the 1,108 one-step measured pairs and then
persists only the per-site aggregate, so the two checks that decide whether the
residue-context reading is settled are not computable from the saved output:

* **the source split** -- whether a model's ordering skill is concentrated in
  pairs it could have memorized. Each pair gets both members' original
  proximity stratum, whether each member is in the adaptation training split,
  and the resulting pair-level category.
* **the additive-score baseline** -- whether a scorer with no residue context
  orders the same pairs as well. Three persisted comparators are applied:
  ``additive`` (the per-site linear score), ``main_effects_only`` (logistic,
  190 main-effect features, no pair terms) and ``pairwise`` (logistic, the same
  190 plus 16,245 pair features). If the additive scorers order these pairs as
  well as the policies do, residue context is not what the policies are using.

A pair is two measured validation cores differing at exactly one of the ten
editable positions and carrying DIFFERENT class labels; the member with the
stronger label is ``strong``. "Ordered correctly" means the scorer gives the
strong member the higher score. Ties count as incorrect, which is the
conservative direction.

CPU only, and it loads the comparators on CPU deliberately so it can run beside
a GPU job.
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
from smallAntibodyGen.experiments import her2_nf_coupling as coupling  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_services as services_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

CLASS_RANK = {"low": 0, "mid": 1, "high": 2}
COMPARATORS = ("additive", "main_effects_only", "pairwise")


def build_pairs(cores, classes):
    """Every unordered one-step pair of measured cores with different labels."""
    position_of = {core: i for i, core in enumerate(cores)}
    pairs, seen = [], set()
    for core, i in position_of.items():
        for site in range(data_lib.CORE_LENGTH):
            for residue in data_lib.CANONICAL:
                if residue == core[site]:
                    continue
                j = position_of.get(core[:site] + residue + core[site + 1:])
                if j is None:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                left, right = classes[i], classes[j]
                if CLASS_RANK[left] == CLASS_RANK[right]:
                    continue
                strong, weak = (i, j) if CLASS_RANK[left] > CLASS_RANK[right] else (j, i)
                pairs.append({"site": site, "strong_row": strong, "weak_row": weak,
                              "background": core[:site] + "." + core[site + 1:],
                              "strong_core": cores[strong], "weak_core": cores[weak],
                              "strong_residue": cores[strong][site],
                              "weak_residue": cores[weak][site],
                              "transition": f"{classes[weak]}->{classes[strong]}"})
    return pairs


def clustered(pairs, correct, key):
    """Fraction correct, weighting each distinct ``key`` value once.

    Overlapping neighbours of one background are not independent trials, so the
    naive Wilson interval over pairs is optimistic and is reported beside this.
    """
    groups = collections.defaultdict(list)
    for pair, ok in zip(pairs, correct):
        groups[pair[key]].append(bool(ok))
    means = [float(np.mean(v)) for v in groups.values()]
    flat = [ok for v in groups.values() for ok in v]
    interval = metrics.wilson_interval(int(sum(flat)), len(flat))
    return {"pairs": len(flat), "groups": len(groups),
            "fraction_ordered_correctly": float(np.mean(flat)),
            "naive_lower": interval["lower"], "naive_upper": interval["upper"],
            "clustered_mean": float(np.mean(means)),
            "clustered_se": (float(np.std(means, ddof=1) / np.sqrt(len(means)))
                             if len(means) > 1 else float("nan"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--raw-root", default="data/raw/her2_functional_20260918")
    parser.add_argument("--extra-scores", default="outputs/her2_matched_table_20260923/row_scores",
                        help="a second row_scores directory, e.g. the fixed/fresh fill")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    args = parser.parse_args(argv)

    run_root, out = ROOT / args.run_root, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    raw = ROOT / args.raw_root

    val = data_lib.load_split(raw, "val")
    cores = val.seq.tolist()
    classes = np.asarray(val["class"]).astype(str)
    index = data_lib.encode_cores(cores)
    train_cores = set(data_lib.load_split(raw, "train").seq.tolist())

    pairs = build_pairs(cores, classes)
    print(f"one-step pairs {len(pairs)}   backgrounds "
          f"{len({p['background'] for p in pairs})}", flush=True)

    # -- the source split ---------------------------------------------------
    strata = None
    for candidate in sorted((run_root / "row_scores").glob("parent_seed*.npz")):
        strata = np.asarray(paths.read_arrays(candidate)["original_train_distance_label"]).astype(str)
        break
    for pair in pairs:
        pair["strong_in_train"] = pair["strong_core"] in train_cores
        pair["weak_in_train"] = pair["weak_core"] in train_cores
        pair["members_in_train"] = int(pair["strong_in_train"]) + int(pair["weak_in_train"])
        if strata is not None:
            pair["strong_stratum"] = strata[pair["strong_row"]]
            pair["weak_stratum"] = strata[pair["weak_row"]]
            pair["pair_stratum"] = ("mixed" if pair["strong_stratum"] != pair["weak_stratum"]
                                    else pair["strong_stratum"])
    counts = collections.Counter(p["members_in_train"] for p in pairs)
    print("members in the adaptation training split: " +
          ", ".join(f"{k} -> {v}" for k, v in sorted(counts.items())), flush=True)

    # -- scorers ------------------------------------------------------------
    scorers = {}
    for directory in (run_root / "row_scores", ROOT / args.extra_scores):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.npz")):
            if path.stem.startswith("challenge_"):
                continue
            arrays = paths.read_arrays(path)
            if "mean_log_probability" not in arrays:
                continue
            scorers[path.stem] = np.asarray(arrays["mean_log_probability"], dtype=np.float64)
    print(f"policy scorers {len(scorers)}", flush=True)

    comparator_dir = run_root / "comparators" / "original_split"
    comparator_scores = {}
    additive_path = comparator_dir / "additive.pt"
    if additive_path.is_file():
        import torch

        from smallAntibodyGen.experiments import her2_baselines as baselines
        stored = torch.load(additive_path, map_location="cpu", weights_only=True)
        comparator_scores["additive"] = np.asarray(
            baselines.AdditiveLinear(stored["weight"], stored["bias"]).positive_scores(index),
            dtype=np.float64)
    for name, filename in (("main_effects_only", "main_only.npz"), ("pairwise", "pairwise.npz")):
        fit = services_lib._load_sklearn(comparator_dir / filename)
        if fit is not None:
            comparator_scores[name] = np.asarray(
                coupling.classifier_scores(fit, index)["p_high"], dtype=np.float64)
    print(f"comparator scorers {sorted(comparator_scores)}", flush=True)

    # -- pair-level table ---------------------------------------------------
    all_scores = dict(scorers, **comparator_scores)
    names = sorted(all_scores)
    fields = ["pair_id", "site", "background", "strong_core", "weak_core", "strong_residue",
              "weak_residue", "transition", "strong_in_train", "weak_in_train",
              "members_in_train", "strong_stratum", "weak_stratum", "pair_stratum"]
    with (out / "substitution_pair_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields + [f"delta__{n}" for n in names])
        for position, pair in enumerate(pairs):
            writer.writerow([position] + [pair.get(f, "") for f in fields[1:]]
                            + ["%.6g" % (all_scores[n][pair["strong_row"]]
                                         - all_scores[n][pair["weak_row"]]) for n in names])

    # -- summaries ----------------------------------------------------------
    summary = {"record_kind": "one_step_pair_rows",
               "pairs": len(pairs),
               "backgrounds": len({p["background"] for p in pairs}),
               "definition": ("two measured validation cores differing at exactly one editable "
                              "position with different class labels; the stronger label is "
                              "'strong'. A tie in score counts as incorrect."),
               "source_split_note": ("members_in_train is DEGENERATE BY CONSTRUCTION and is "
                                     "recorded only to make that visible: pairs are built from "
                                     "validation cores, train and validation cores are disjoint, "
                                     "so every pair is 0-in-train. The informative split is "
                                     "by_proximity_stratum -- each member's distance from the "
                                     "original training rows. A scorer that orders the distance-1 "
                                     "pairs far better than the >=3 pairs is leaning on proximity "
                                     "to the adaptation data."),
               "comparator_note": ("additive and main_effects_only carry NO residue-context "
                                   "term. If they match the policies here, residue context is "
                                   "not what separates the policies on this statistic."),
               "overall": {}, "by_source_split": {}, "by_proximity_stratum": {},
               "by_transition": {}}
    for name in names:
        score = all_scores[name]
        correct = [score[p["strong_row"]] > score[p["weak_row"]] for p in pairs]
        summary["overall"][name] = clustered(pairs, correct, "background")
        by_split = {}
        for members in sorted({p["members_in_train"] for p in pairs}):
            subset = [p for p in pairs if p["members_in_train"] == members]
            subset_correct = [score[p["strong_row"]] > score[p["weak_row"]] for p in subset]
            by_split[str(members)] = clustered(subset, subset_correct, "background")
        summary["by_source_split"][name] = by_split
        by_stratum = {}
        for stratum in sorted({p.get("pair_stratum", "") for p in pairs}):
            subset = [p for p in pairs if p.get("pair_stratum", "") == stratum]
            subset_correct = [score[p["strong_row"]] > score[p["weak_row"]] for p in subset]
            by_stratum[stratum or "unlabelled"] = clustered(subset, subset_correct, "background")
        summary["by_proximity_stratum"][name] = by_stratum
        by_transition = {}
        for transition in sorted({p["transition"] for p in pairs}):
            subset = [p for p in pairs if p["transition"] == transition]
            subset_correct = [score[p["strong_row"]] > score[p["weak_row"]] for p in subset]
            by_transition[transition] = clustered(subset, subset_correct, "background")
        summary["by_transition"][name] = by_transition
    summary["recorded_at"] = paths.utc_now()
    paths.write_json(out / "substitution_pair_summary.json", summary)

    print("\n%-34s %8s %10s %10s" % ("scorer", "pairs", "correct", "clustered"))
    for name in names:
        block = summary["overall"][name]
        print("%-34s %8d %10.4f %10.4f" % (name, block["pairs"],
                                           block["fraction_ordered_correctly"],
                                           block["clustered_mean"]))
    strata_keys = sorted(summary["by_proximity_stratum"][names[0]])
    print("\nby proximity stratum (each member's distance from the original training rows)")
    print("%-34s %s" % ("scorer", "  ".join("%10s" % k for k in strata_keys)))
    for name in names:
        row = summary["by_proximity_stratum"][name]
        print("%-34s %s" % (name, "  ".join(
            "%10.4f" % row[k]["fraction_ordered_correctly"] for k in strata_keys)))
    print("\npairs per stratum: " + ", ".join(
        "%s=%d" % (k, summary["by_proximity_stratum"][names[0]][k]["pairs"])
        for k in strata_keys))
    print("members_in_train is degenerate by construction (all pairs are validation cores): " +
          ", ".join(f"{k} -> {v}" for k, v in sorted(counts.items())))
    print(f"\nwrote {out / 'substitution_pair_rows.csv'}")
    print(f"wrote {out / 'substitution_pair_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
