#!/usr/bin/env python
"""Score the recovered measured-affinity cohort with every trained arm.

``scripts/her2_affinity_cohort.py`` recovered the cohort from the published SPR
plate and built the two model-free baselines. This scores it.

**The question.** Every ranking number in this flight is computed against the
labelled library's three-way class label. That label is the adaptation target,
so a model ranking it well is partly reporting that it learned its training
signal. This cohort is an *independent* readout: 187 clones with a measured KD
from an assay this project never touched. Does a policy's own log probability
order them by measured affinity, and does preference optimization move that
ordering relative to the policy it started from?

**The populations**, nested, declared before any correlation was computed:

===  =========================================================================
187  every in-support row with a finite positive KD
156  the five *design* methods only (blosum, ablang_all, ablang_one, esm_one,
     protein_mpnn); drops the 8 positive controls and the 23 greiff rows, which
     are classifier-audit categories rather than designs
153  156 minus the 3 rows whose core is also in the labelled library (IgG_43,
     IgG_51, IgG_144 -- all blosum, all class ``high``). Those rows were
     available to training and to the flight's evaluation panel, so a
     correlation over them is not independent of the adaptation data
152  **primary.** 153 minus IgG_489 (ablang_all, KD 1.386e-5 M), whose KD sits
     15.6x beyond the next-weakest row at 8.86e-7 M. A single point that far
     outside the rest of the assay's range moves a rank correlation on its own
===  =========================================================================

152 is primary because it is the largest population that is both independent of
the adaptation data and inside the assay's observed range. 153 and 187 are
reported alongside it so the cost of each exclusion is visible; no conclusion
here may rest on which of the three is chosen.

**The comparison.** Each arm is differenced against *its own* parent -- Block A
arms against the initial-SFT parent they started from, Block B arms against the
stage-1 parent of their own corpus regime. Differencing a purge arm against the
SFT parent would confound the objective with the corpus.

**What this cannot say.** ``mean_log_probability`` is a model score, not an
experimental binding value. A correlation here says the policy's likelihood
tracks measured affinity on 152 clones; it does not say the policy predicts
binding, and it never substitutes for an assay.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

DESIGN_METHODS = ("ablang_all", "ablang_one", "blosum", "esm_one", "protein_mpnn")
#: The one row outside the assay's observed range; see the module docstring.
RANGE_OUTLIER = "IgG_489"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923


# -- statistics ----------------------------------------------------------

def spearman(a, b):
    """Rank correlation, average ranks on ties."""
    return float(stats.spearmanr(a, b).statistic)


def _rank_rows(matrix):
    """Average ranks along each row of a 2-D array."""
    return stats.rankdata(matrix, axis=1)


def _row_pearson(a, b):
    """Pearson correlation of each aligned row pair."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(axis=1) / denominator, np.nan)


def bootstrap_rows(affinity, *vectors, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Spearman of each vector against affinity under a shared row resample.

    The SAME resampled rows are used for every vector in one draw, which is what
    makes the difference between two of them a paired quantity. Rows are clones,
    not models: this describes how much the correlation depends on which clones
    the plate happened to contain, and never stands in for seed variability.
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    rows = generator.integers(0, affinity.size, size=(int(draws), affinity.size))
    reference = _rank_rows(affinity[rows])
    return [_row_pearson(_rank_rows(np.asarray(v, dtype=np.float64)[rows]), reference)
            for v in vectors]


def interval(samples, *, confidence=0.95):
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"lower": None, "upper": None, "draws": 0}
    tail = (1.0 - confidence) / 2.0
    return {"lower": float(np.quantile(finite, tail)),
            "upper": float(np.quantile(finite, 1.0 - tail)),
            "draws": int(finite.size)}


# -- the checkpoint inventory --------------------------------------------

def inventory(seed, run_root, historical_root, repository_root):
    """Every checkpoint scored at one seed, each arm tagged with its own parent."""
    parents = {"A": repository_root / "outputs" / "her2_posttrain_20260918"
                    / f"policy_sft_seed{seed}" / "epoch_3.pt"}
    for regime in ("purge", "matched"):
        selection = json.loads((run_root / "parents" / f"{regime}_seed{seed}"
                                / "selection.json").read_text(encoding="utf-8"))
        parents[f"B_{regime}"] = Path(selection["selected"]["path"])
    arms = [
        ("IPO_0", "A", historical_root / "trajectories" / f"ipo_lambda0_seed{seed}", (1000, 3750)),
        ("IPO_FKL", "A", historical_root / "trajectories" / f"ipo_lambda10_seed{seed}", (1000, 3750)),
        ("IPO_TAIL", "A", run_root / "trajectories" / f"A_IPO_TAIL_seed{seed}", (1000, 3750)),
        ("DPO_0", "A", run_root / "trajectories" / f"A_DPO_0_seed{seed}", (1000, 3750)),
        ("purge_IPO_0", "B_purge", run_root / "trajectories" / f"B_purge_IPO_0_seed{seed}", (1000,)),
        ("purge_IPO_FKL", "B_purge", run_root / "trajectories" / f"B_purge_IPO_FKL_seed{seed}", (1000,)),
        ("matched_IPO_0", "B_matched", run_root / "trajectories" / f"B_matched_IPO_0_seed{seed}", (1000,)),
        ("matched_IPO_FKL", "B_matched", run_root / "trajectories" / f"B_matched_IPO_FKL_seed{seed}", (1000,)),
    ]
    jobs = [(f"parent_{family}", family, 0, path) for family, path in sorted(parents.items())]
    for arm, family, trajectory, updates in arms:
        for update in updates:
            jobs.append((arm, family, update, trajectory / f"endpoint_update{update}.pt"))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--cohort", default="outputs/her2_biology_20260923/affinity_152.csv")
    parser.add_argument("--output", default="outputs/her2_biology_20260923")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((ROOT / args.cohort).open(encoding="utf-8")))
    # The file is named for the primary population but holds every in-support
    # row; the populations are cut from it here, not upstream.
    affinity = np.array([float(r["neg_log10_kd"]) for r in rows], dtype=np.float64)
    cores = [r["core"] for r in rows]
    index = data_lib.encode_cores(cores)
    design = np.array([r["source_method"] in DESIGN_METHODS for r in rows])
    exposed = np.array([r["in_library"] == "True" for r in rows])
    out_of_range = np.array([r["clone_id"] == RANGE_OUTLIER for r in rows])
    populations = {
        "primary_152": design & ~exposed & ~out_of_range,
        "sensitivity_153_with_range_outlier": design & ~exposed,
        "sensitivity_187_all_in_support": np.ones(len(rows), dtype=bool),
    }
    for name, mask in populations.items():
        print(f"{name:42s} {int(mask.sum()):4d} rows", flush=True)

    context = spec.resolve_context(ROOT, config_path=ROOT / args.config,
                                   run_root=ROOT / args.run_root)
    services = campaign.build_services(context, device=args.device)
    historical_root = services.historical_root()
    batch = int(context.config["inference"]["score_batch_size"])

    scores, missing = {}, []
    for seed in seeds:
        for arm, family, update, checkpoint in inventory(
                seed, ROOT / args.run_root, historical_root, context.repository_root):
            if not Path(checkpoint).is_file():
                missing.append({"arm": arm, "seed": seed, "update": update,
                                "path": str(checkpoint)})
                print(f"  MISSING {arm} seed{seed} u{update}", flush=True)
                continue
            policy = services.policy(Path(checkpoint))
            block = policy.score(index, batch_size=batch)
            scores[(arm, family, seed, update)] = np.asarray(
                block["mean_log_probability"], dtype=np.float64)
            del policy
            services.release_transient()
            rho = spearman(scores[(arm, family, seed, update)][populations["primary_152"]],
                           affinity[populations["primary_152"]])
            print(f"  seed{seed} {arm:16s} u{update:<5d} rho(152) {rho:+.4f}", flush=True)

    # -- model-free baselines. The GPU pass has to beat these, not beat chance.
    wt_distance = np.array([-float(r["wt_distance"]) for r in rows])
    neighbour = np.array([float(r["nearest_neighbour_score"]) for r in rows])
    baselines = {}
    for name, mask in populations.items():
        samples = bootstrap_rows(affinity[mask], wt_distance[mask], neighbour[mask])
        baselines[name] = {
            "negative_distance_to_wild_type": {
                "spearman": spearman(wt_distance[mask], affinity[mask]),
                **interval(samples[0])},
            "nearest_labelled_neighbour_class": {
                "spearman": spearman(neighbour[mask], affinity[mask]),
                **interval(samples[1])},
            "rows": int(mask.sum()),
        }

    # -- per checkpoint, per population --------------------------------------
    per_checkpoint = []
    for (arm, family, seed, update), vector in scores.items():
        record = {"arm": arm, "parent_family": family, "seed": seed, "update": update,
                  "populations": {}, "by_source_method": {}}
        for name, mask in populations.items():
            samples = bootstrap_rows(affinity[mask], vector[mask])[0]
            record["populations"][name] = {"spearman": spearman(vector[mask], affinity[mask]),
                                           "rows": int(mask.sum()), **interval(samples)}
        primary = populations["primary_152"]
        for method in DESIGN_METHODS:
            mask = primary & np.array([r["source_method"] == method for r in rows])
            if mask.sum() >= 8:
                record["by_source_method"][method] = {
                    "spearman": spearman(vector[mask], affinity[mask]),
                    "rows": int(mask.sum())}
        per_checkpoint.append(record)

    # -- arm minus its own parent -------------------------------------------
    contrasts = []
    arms = sorted({(a, f, u) for (a, f, s, u) in scores if not a.startswith("parent_")})
    for arm, family, update in arms:
        block = {"arm": arm, "update": update, "parent_family": family,
                 "parent": f"parent_{family}", "per_population": {}}
        for name, mask in populations.items():
            per_seed, row_intervals = [], []
            for seed in seeds:
                key, parent_key = (arm, family, seed, update), (f"parent_{family}", family, seed, 0)
                if key not in scores or parent_key not in scores:
                    continue
                arm_rho = spearman(scores[key][mask], affinity[mask])
                parent_rho = spearman(scores[parent_key][mask], affinity[mask])
                per_seed.append(arm_rho - parent_rho)
                arm_samples, parent_samples = bootstrap_rows(
                    affinity[mask], scores[key][mask], scores[parent_key][mask])
                row_intervals.append({"seed": seed, "arm_spearman": arm_rho,
                                      "parent_spearman": parent_rho,
                                      "difference": arm_rho - parent_rho,
                                      **interval(arm_samples - parent_samples)})
            if len(per_seed) < 2:
                continue
            across = metrics.paired_t(per_seed)
            block["per_population"][name] = {
                "rows": int(mask.sum()), "per_seed": row_intervals,
                "across_seed_mean": across["mean"], "across_seed_lower": across["lower"],
                "across_seed_upper": across["upper"],
                "across_seed_excludes_zero": across["excludes_zero"],
                "seeds_agreeing_in_sign": int(max(sum(1 for d in per_seed if d > 0),
                                                  sum(1 for d in per_seed if d < 0))),
                "seeds": len(per_seed)}
        contrasts.append(block)

    document = {
        "record_kind": "affinity_cohort_scoring",
        "cohort": str(Path(args.cohort).as_posix()),
        "measurement": "-log10(KD) from the published SPR plate; higher is stronger",
        "model_score": "mean_log_probability over the ten editable positions",
        "population_definitions": {
            "primary_152": ("the five design methods, excluding the 3 cores present in the "
                            "labelled library and the 1 row outside the assay's observed "
                            f"range ({RANGE_OUTLIER}, KD 1.386e-5 M)"),
            "sensitivity_153_with_range_outlier": "primary plus the out-of-range row",
            "sensitivity_187_all_in_support": ("every in-support row, including the positive "
                                               "controls, the greiff audit categories and the "
                                               "3 library-exposed cores"),
        },
        "population_rows": {k: int(v.sum()) for k, v in populations.items()},
        "row_bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                          "resamples": "clones, with the same rows applied to every vector "
                                       "in a draw so arm-minus-parent is paired",
                          "is_not": "a statement about training-seed variability"},
        "across_seed_interval": "paired t over the three parent seeds, df=2",
        "baselines": baselines,
        "per_checkpoint": per_checkpoint,
        "contrasts": contrasts,
        "missing": missing,
        "claim_limit": ("mean_log_probability is a model score. A correlation here is evidence "
                        "that the policy's likelihood tracks measured affinity on this cohort. "
                        "It is not an experimental binding value and does not replace an assay."),
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "affinity_scoring.json", document)

    fields = ["clone_id", "core", "source_method", "neg_log10_kd", "wt_distance",
              "in_library", "primary_152", "sensitivity_153", "sensitivity_187"]
    keys = sorted(scores)
    with (out / "affinity_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields + [f"{a}_seed{s}_u{u}" for (a, _f, s, u) in keys])
        for position, row in enumerate(rows):
            writer.writerow([row["clone_id"], row["core"], row["source_method"],
                             row["neg_log10_kd"], row["wt_distance"], row["in_library"],
                             bool(populations["primary_152"][position]),
                             bool(populations["sensitivity_153_with_range_outlier"][position]),
                             True]
                            + [f"{scores[k][position]:.6f}" for k in keys])

    print("\nbaselines on the primary 152 rows")
    for name, block in baselines["primary_152"].items():
        if isinstance(block, dict):
            print("  %-34s %+.4f [%+.4f, %+.4f]" % (name, block["spearman"],
                                                    block["lower"], block["upper"]))
    print("\n%-16s %6s %10s %12s %10s %s" % ("arm", "update", "mean diff", "95% CI",
                                             "", "seeds agreeing"))
    for block in contrasts:
        entry = block["per_population"].get("primary_152")
        if entry is None:
            continue
        print("%-16s %6d %+10.4f [%+.4f, %+.4f] %3d/%d" % (
            block["arm"], block["update"], entry["across_seed_mean"],
            entry["across_seed_lower"], entry["across_seed_upper"],
            entry["seeds_agreeing_in_sign"], entry["seeds"]))
    print(f"\nwrote {out / 'affinity_scoring.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
