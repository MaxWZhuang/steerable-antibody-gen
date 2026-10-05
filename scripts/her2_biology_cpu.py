#!/usr/bin/env python
"""CPU-only biology interpretation on the completed flight's persisted artifacts.

Three deliverables that need no GPU, written beside the run and outside its
verified root:

* ``site_atlas.csv``   -- the ten editable sites, measured-class residue
  frequencies, and the wild-type residue. **Model-generated frequencies are NOT
  here**: the coupling banks persisted their digests but not their arrays, so
  policy draws have to be regenerated on the GPU.
* ``motif_tests.csv``  -- named joint patterns, counted in the LABELLED LIBRARY
  with their binding-class association, against matched non-carriers.
* ``substitution_pairs.csv`` -- measured sequences differing at exactly one
  editable site, and whether each model's score ordering agrees with the
  measured binding-group ordering.

Position naming: this repository contains no IMGT or Kabat map for the HER2
core, so the atlas carries the verified 0-based core index and the trastuzumab
residue only. The IMGT/Kabat columns are emitted as ``unverified`` rather than
filled from an analyst-derived mapping.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

CLASS_RANK = {"low": 0, "mid": 1, "high": 2}
#: Declared BEFORE ranking sites, as the brief requires: a site is called
#: "shifted" only if the change exceeds this in absolute percentage points.
EFFECT_THRESHOLD_PP = 2.0


def labelled_library(raw):
    """Every measured core with its class, from both splits, de-duplicated."""
    table = {}
    for split in ("train", "val"):
        frame = data_lib.load_split(raw, split)
        for core, klass in zip(frame.seq, frame["class"]):
            table.setdefault(core, {"class": klass, "split": split})
    return table


def site_atlas(library, out):
    """Residue frequency per site, per measured class. Denominator: library rows."""
    by_class = collections.defaultdict(list)
    for core, block in library.items():
        by_class[block["class"]].append(core)
    rows = []
    for position in range(data_lib.CORE_LENGTH):
        record = {
            "core_index": position,
            "imgt_position": "unverified",
            "kabat_position": "unverified",
            "numbering_note": ("no IMGT/Kabat map for this core exists in the repository; "
                               "verify against the trastuzumab heavy chain and 1N8Z before "
                               "naming"),
            "trastuzumab_residue": data_lib.WT_CORE[position],
            "denominator": "labelled library rows, de-duplicated by core",
        }
        for klass in ("high", "mid", "low"):
            cores = by_class[klass]
            counts = collections.Counter(core[position] for core in cores)
            total = len(cores)
            record[f"{klass}_rows"] = total
            top = counts.most_common(4)
            record[f"{klass}_distribution"] = ";".join(
                f"{residue}:{count / total:.4f}" for residue, count in top)
            record[f"{klass}_top_residue"] = top[0][0] if top else ""
            record[f"{klass}_top_fraction"] = top[0][1] / total if top else 0.0
            record[f"{klass}_wt_fraction"] = counts.get(
                data_lib.WT_CORE[position], 0) / total
            # Wilson bound on the wild-type residue's share, so a site that
            # merely has few rows cannot look constrained.
            wilson = metrics.wilson_interval(counts.get(data_lib.WT_CORE[position], 0), total)
            record[f"{klass}_wt_lower"] = wilson["lower"]
            record[f"{klass}_wt_upper"] = wilson["upper"]
        # "High-binding constrained" is a statement about the measured rows only.
        record["high_minus_low_wt_pp"] = 100.0 * (record["high_wt_fraction"]
                                                  - record["low_wt_fraction"])
        record["high_constrained"] = bool(
            record["high_wt_lower"] > record["low_wt_upper"])
        record["model_frequency_status"] = "pending_gpu_regeneration"
        rows.append(record)

    fields = list(rows[0])
    with (out / "site_atlas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def pattern_association(library, positions, residues, label):
    """Binding-class counts for carriers vs residue-matched non-carriers.

    The matching is the point: a carrier of T-H-T is compared with sequences
    carrying each component residue at the same site but NOT the full pattern,
    so the triplet's incremental association is separated from its parts.
    """
    carriers, partial, neither = [], [], []
    for core, block in library.items():
        hits = sum(core[p] == r for p, r in zip(positions, residues))
        if hits == len(positions):
            carriers.append(block["class"])
        elif hits > 0:
            partial.append(block["class"])
        else:
            neither.append(block["class"])

    def block_of(items):
        counts = collections.Counter(items)
        total = max(1, len(items))
        high = counts.get("high", 0)
        return {"rows": len(items), "high": high, "mid": counts.get("mid", 0),
                "low": counts.get("low", 0), "high_fraction": high / total,
                "high_wilson": metrics.wilson_interval(high, total)}

    carrier, part, none = block_of(carriers), block_of(partial), block_of(neither)
    return {
        "pattern": label,
        "positions": "+".join(str(p) for p in positions),
        "residues": "".join(residues),
        "carrier_rows": carrier["rows"], "carrier_high": carrier["high"],
        "carrier_mid": carrier["mid"], "carrier_low": carrier["low"],
        "carrier_high_fraction": carrier["high_fraction"],
        "carrier_high_lower": carrier["high_wilson"]["lower"],
        "carrier_high_upper": carrier["high_wilson"]["upper"],
        "partial_rows": part["rows"], "partial_high_fraction": part["high_fraction"],
        "nonmatch_rows": none["rows"], "nonmatch_high_fraction": none["high_fraction"],
        "incremental_vs_partial_pp": 100.0 * (carrier["high_fraction"]
                                              - part["high_fraction"]),
        "separated_from_partial": bool(
            carrier["high_wilson"]["lower"] > part["high_wilson"]["upper"]
            or carrier["high_wilson"]["upper"] < part["high_wilson"]["lower"]),
        "coverage_note": ("carriers are compared with rows sharing at least one component "
                          "residue at the same site; sparse carrier counts make the interval "
                          "wide rather than the estimate wrong"),
    }


def motif_tests(library, out):
    """The two NAMED leads, plus every consecutive triple as declared context.

    All eight consecutive triples are reported, not the largest. The named lead
    is tested at every consecutive position triple because this repository holds
    no verified IMGT map, so which core indices "H100A/H100B/H101" denote is not
    established here.
    """
    rows = []
    for start in range(data_lib.CORE_LENGTH - 2):
        positions = (start, start + 1, start + 2)
        rows.append({**pattern_association(library, positions, ("T", "H", "T"),
                                           f"T-H-T@core{start}-{start + 2}"),
                     "lead": "named exploratory lead (position naming UNVERIFIED)",
                     "preselected": "declared before inspection; all 8 triples reported"})
    # The H96/H97/H98 cluster is a PAIR lead: report every adjacent pair rather
    # than the best of 45.
    for start in range(data_lib.CORE_LENGTH - 1):
        positions = (start, start + 1)
        wt = (data_lib.WT_CORE[start], data_lib.WT_CORE[start + 1])
        rows.append({**pattern_association(library, positions, wt,
                                           f"WT-pair@core{start}-{start + 1}"),
                     "lead": "adjacent-pair context (wild-type residues)",
                     "preselected": "declared before inspection; all 9 pairs reported"})
    fields = list(rows[0])
    with (out / "motif_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def substitution_pairs(library, run_root, out):
    """One-step measured neighbours, and whether each model orders them correctly.

    Uncertainty is clustered by BACKGROUND -- the nine residues the pair shares --
    because overlapping neighbours of one background are not independent trials.
    """
    val_scores = {}
    score_dir = run_root / "row_scores"
    for path in sorted(score_dir.glob("*.npz")):
        name = path.stem
        if name.startswith("challenge_") or name.startswith("mixture"):
            continue
        arrays = paths.read_arrays(path)
        if "mean_log_probability" not in arrays:
            continue
        val_scores[name] = arrays
    if not val_scores:
        return [], {}
    sample = next(iter(val_scores.values()))
    cores = data_lib.load_split(ROOT / "data/raw/her2_functional_20260918", "val").seq.tolist()
    classes = np.asarray(sample["class"]).astype(str)
    position_of = {core: i for i, core in enumerate(cores)}

    pairs = []
    seen = set()
    for core, i in position_of.items():
        for site in range(data_lib.CORE_LENGTH):
            for residue in data_lib.CANONICAL:
                if residue == core[site]:
                    continue
                other = core[:site] + residue + core[site + 1:]
                j = position_of.get(other)
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
                              "strong_residue": cores[strong][site],
                              "weak_residue": cores[weak][site],
                              "transition": f"{classes[weak]}->{classes[strong]}"})

    rows = []
    for name, arrays in sorted(val_scores.items()):
        score = np.asarray(arrays["mean_log_probability"], dtype=np.float64)
        by_site = collections.defaultdict(lambda: collections.defaultdict(list))
        for pair in pairs:
            correct = bool(score[pair["strong_row"]] > score[pair["weak_row"]])
            by_site[pair["site"]][pair["background"]].append(correct)
        for site, backgrounds in sorted(by_site.items()):
            flat = [ok for values in backgrounds.values() for ok in values]
            # One value per background, then averaged: the clustered estimate.
            clustered = [float(np.mean(values)) for values in backgrounds.values()]
            interval = metrics.wilson_interval(int(sum(flat)), len(flat))
            rows.append({
                "model": name, "site": site,
                "pairs": len(flat), "backgrounds": len(backgrounds),
                "fraction_ordered_correctly": float(np.mean(flat)),
                "naive_lower": interval["lower"], "naive_upper": interval["upper"],
                "clustered_by_background": float(np.mean(clustered)),
                "clustered_se": float(np.std(clustered, ddof=1) / np.sqrt(len(clustered)))
                if len(clustered) > 1 else float("nan"),
                "note": ("naive interval treats pairs as independent and is optimistic; "
                         "the clustered mean weights each background once"),
            })
    fields = list(rows[0]) if rows else []
    if rows:
        with (out / "substitution_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {"one_step_pairs": len(pairs), "models_scored": len(val_scores),
               "distinct_backgrounds": len({p["background"] for p in pairs})}
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/her2_functional_20260918")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    args = parser.parse_args(argv)

    raw, run_root = ROOT / args.raw_root, ROOT / args.run_root
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    library = labelled_library(raw)
    print(f"labelled library: {len(library)} distinct measured cores")

    atlas = site_atlas(library, out)
    print("\nsite atlas (measured-class residue frequencies, library rows):")
    print("%5s %3s %9s %9s %9s %10s %s" % ("index", "WT", "high WT%", "mid WT%", "low WT%",
                                           "hi-lo pp", "constrained"))
    for record in atlas:
        print("%5d %3s %9.3f %9.3f %9.3f %10.1f %s" % (
            record["core_index"], record["trastuzumab_residue"],
            record["high_wt_fraction"], record["mid_wt_fraction"],
            record["low_wt_fraction"], record["high_minus_low_wt_pp"],
            record["high_constrained"]))

    motifs = motif_tests(library, out)
    print("\nmotif tests in the labelled library (carriers vs residue-matched partials):")
    print("%-26s %8s %8s %12s %12s %s" % ("pattern", "carriers", "high", "carrier hi%",
                                          "partial hi%", "separated"))
    for record in motifs:
        if not record["pattern"].startswith("T-H-T"):
            continue
        print("%-26s %8d %8d %12.3f %12.3f %s" % (
            record["pattern"], record["carrier_rows"], record["carrier_high"],
            record["carrier_high_fraction"], record["partial_high_fraction"],
            record["separated_from_partial"]))

    rows, summary = substitution_pairs(library, run_root, out)
    print(f"\none-step measured neighbour pairs with a label change: "
          f"{summary.get('one_step_pairs')} over "
          f"{summary.get('distinct_backgrounds')} backgrounds, "
          f"{summary.get('models_scored')} models scored")

    paths.write_json(out / "biology_cpu_summary.json", {
        "record_kind": "biology_cpu_analysis",
        "library_cores": len(library),
        "effect_threshold_pp": EFFECT_THRESHOLD_PP,
        "substitution": summary,
        "pending_gpu": [
            "per-arm generated residue frequencies (coupling banks persisted digests, "
            "not arrays)",
            "motif counts on size-matched generated banks",
            "affinity cohort scoring",
        ],
        "numbering": ("no IMGT/Kabat map for this core exists in the repository; the atlas "
                      "carries verified 0-based core indices and the trastuzumab residue"),
        "recorded_at": paths.utc_now(),
    })
    print(f"\nwrote {out}/site_atlas.csv, motif_tests.csv, substitution_pairs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
