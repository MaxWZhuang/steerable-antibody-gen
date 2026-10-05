#!/usr/bin/env python
"""Did the FKL arms learn their fixed replay bank?

For every IPO-with-replay checkpoint, score TWO parent-drawn banks with the same
parent and the same policy:

* the **training replay bank** the arm actually replayed against, and
* an **independent fresh bank** drawn from the same parent.

Both are draws from the same distribution P; only one of them entered training.
The reported statistic is the per-sequence drop ``log P_parent(x) - log P_policy(x)``.
A consistently smaller drop on the training bank is what finite-bank
overfitting would look like; indistinguishable distributions mean no gap at this
resolution.

Two things this deliberately does NOT do. It never writes inside the completed
run root -- the artifact manifest there is verified and a new file would break
it -- and its numbers are descriptive. The fresh bank stays an audit of frozen
checkpoints; nothing here may be used to pick a new setting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

TENFOLD = float(np.log(10.0))
HUNDREDFOLD = float(np.log(100.0))
#: The matched-subset draw. Declared here so the subset is reproducible and is
#: not quietly re-rolled until a difference appears.
SUBSET_SEED = 20260923
SUBSET_ROWS = 50_000


def drop_block(parent, policy, *, label, rows):
    """One drop distribution, with the percentiles and rates the question asks for."""
    drop = np.asarray(parent, dtype=np.float64) - np.asarray(policy, dtype=np.float64)
    tenfold = int((drop >= TENFOLD).sum())
    hundredfold = int((drop >= HUNDREDFOLD).sum())
    quantiles = np.percentile(drop, [50, 90, 95, 99])
    return {
        "label": label, "rows": int(drop.size), "bank_rows": int(rows),
        "mean": float(drop.mean()),
        "standard_error": float(drop.std(ddof=1) / np.sqrt(drop.size)),
        "median": float(quantiles[0]), "p90": float(quantiles[1]),
        "p95": float(quantiles[2]), "p99": float(quantiles[3]),
        "max": float(drop.max()),
        "tenfold_count": tenfold, "tenfold_rate": tenfold / drop.size,
        "tenfold_wilson": metrics.wilson_interval(tenfold, int(drop.size)),
        "hundredfold_count": hundredfold, "hundredfold_rate": hundredfold / drop.size,
        "hundredfold_wilson": metrics.wilson_interval(hundredfold, int(drop.size)),
        "inclusive": "drop >= ln(k), the convention this flight uses throughout",
    }


def bootstrap_difference(a, b, *, seed, draws=2000):
    """Bootstrap interval for mean(a) - mean(b) on two INDEPENDENT samples."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        diffs[i] = (rng.choice(a, a.size, replace=True).mean()
                    - rng.choice(b, b.size, replace=True).mean())
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(a.mean() - b.mean()), "lower": float(lower),
            "upper": float(upper), "draws": int(draws), "seed": int(seed),
            "excludes_zero": bool(lower > 0.0 or upper < 0.0),
            "basis": "independent-sample bootstrap of the difference in means"}


def duplicate_block(index, *, label):
    """Duplicate structure of one bank. Duplicates are retained, never filtered."""
    from smallAntibodyGen.experiments.her2_nf_contract import index_hashes

    hashes = index_hashes(np.asarray(index))
    unique, counts = np.unique(hashes, return_counts=True)
    return {"label": label, "rows": int(hashes.size), "unique_cores": int(unique.size),
            "unique_fraction": float(unique.size / hashes.size),
            "max_repeat": int(counts.max()),
            "rows_in_repeated_cores": int(counts[counts > 1].sum()),
            "hashes": hashes}


def overlap(left_hashes, right_hashes):
    """Sequence-identity overlap between two banks, reported not rejected."""
    shared = np.intersect1d(left_hashes, right_hashes)
    return {"shared_identities": int(shared.size),
            "left_rows_in_shared": int(np.isin(left_hashes, shared).sum()),
            "right_rows_in_shared": int(np.isin(right_hashes, shared).sum()),
            "note": ("both banks are draws from the SAME parent, so overlap is expected "
                     "by chance and is a property of the distribution, not a defect")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--historical", default="outputs/her2_parent_replay_20260920")
    parser.add_argument("--output", default="outputs/her2_next_flight_20260922/"
                                            "analysis_replay_memorisation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--block", default="both", choices=("A", "B", "both"))
    #: CONTROL ARMS. IPO_0 and IPO_TAIL never replay against the bank, so any
    #: difference they show between the two banks belongs to the BANKS -- a
    #: population artifact -- and not to memorisation. Without this the FKL
    #: result is uninterpretable, because the two banks were drawn by different
    #: seeds and, for Block A, by a different campaign's code.
    parser.add_argument("--arms", default="IPO_FKL",
                        help="comma-separated arm ids; IPO_0 / IPO_TAIL are the controls")
    args = parser.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    run_root = ROOT / args.run_root
    out_dir = ROOT / args.output
    assert not str(out_dir.resolve()).startswith(str(run_root.resolve())), (
        "refusing to write inside the completed run root: its artifact manifest is verified")
    out_dir.mkdir(parents=True, exist_ok=True)

    context = spec.resolve_context(ROOT, config_path=ROOT / args.config, run_root=run_root)
    services = campaign.build_services(context, device=args.device)
    registry = services.checkpoint_registry()["entries"]

    jobs = []
    if args.block in ("B", "both"):
        for regime in ("purge", "matched"):
            for seed in context.config["block_b"]["parent_seeds"]:
                for arm in arms:
                    jobs.append({
                        "block": "B", "regime": regime, "seed": int(seed), "update": 1000,
                        "arm": arm, "replayed": arm == "IPO_FKL",
                        "model": f"B_{regime}_{arm}@u1000_seed{seed}",
                        "parent": f"B_{regime}_parent_seed{seed}",
                        "replay_bank": run_root / "banks"
                                       / f"replay_{regime}_seed{seed}_100000.npz",
                        "fresh_bank": run_root / "banks"
                                      / f"final_parent__{regime}__seed{seed}.npz",
                        "fresh_scores": run_root / "preservation"
                                        / f"B_{regime}_{arm}@u1000_seed{seed}.npz"})
    if args.block in ("A", "both"):
        for seed in context.config["block_a"]["parent_seeds"]:
            for update in (1000, 2000, 3750):
                for arm in arms:
                    jobs.append({
                        "block": "A", "regime": None, "seed": int(seed), "update": update,
                        "arm": arm, "replayed": arm == "IPO_FKL",
                        "model": f"A_{arm}@u{update}_seed{seed}",
                        "parent": f"parent_seed{seed}",
                        "replay_bank": ROOT / args.historical / "banks"
                                       / f"seed{seed}" / "replay" / "bank.npz",
                        "fresh_bank": run_root / "banks" / f"final_parent__seed{seed}.npz",
                        "fresh_scores": run_root / "preservation"
                                        / f"A_{arm}@u{update}_seed{seed}.npz"})

    results = []
    for job in jobs:
        name = job["model"]
        entry = registry.get(name)
        parent_entry = registry.get(job["parent"])
        if entry is None or entry.get("status") != "present":
            results.append({**{k: v for k, v in job.items() if not isinstance(v, Path)},
                            "status": "skipped",
                            "reason": f"{name} is not present in the checkpoint registry"})
            print(f"skip {name}: not present", flush=True)
            continue
        for key in ("replay_bank", "fresh_bank", "fresh_scores"):
            if not job[key].is_file():
                results.append({**{k: v for k, v in job.items() if not isinstance(v, Path)},
                                "status": "skipped",
                                "reason": f"{job[key]} is not readable"})
                break
        else:
            print(f"scoring {name}", flush=True)
            replay = paths.read_arrays(job["replay_bank"])
            replay_index = np.asarray(replay["core_index"])
            fresh = paths.read_arrays(job["fresh_scores"])
            fresh_parent = np.asarray(fresh["parent_sum_log_probability"], dtype=np.float64)
            fresh_policy = np.asarray(fresh["policy_sum_log_probability"], dtype=np.float64)
            fresh_index = np.asarray(fresh["core_index"])

            parent_policy = services.policy(Path(parent_entry["checkpoint"]))
            scored_parent = np.asarray(parent_policy.score(
                replay_index, batch_size=args.batch_size)["sum_log_probability"],
                dtype=np.float64)
            del parent_policy
            services.release_transient()
            model_policy = services.policy(Path(entry["checkpoint"]))
            scored_policy = np.asarray(model_policy.score(
                replay_index, batch_size=args.batch_size)["sum_log_probability"],
                dtype=np.float64)
            del model_policy
            services.release_transient()

            # The saved parent scores travel with the historical bank; comparing
            # them to this run's rescore is a free identity check on the bank.
            saved_parent = replay.get("parent_sum_log_probability")
            parent_agreement = None
            if saved_parent is not None:
                delta = np.abs(np.asarray(saved_parent, dtype=np.float64) - scored_parent)
                parent_agreement = {"max_abs_difference": float(delta.max()),
                                    "mean_abs_difference": float(delta.mean()),
                                    "rows": int(delta.size)}

            rng = np.random.default_rng([SUBSET_SEED, job["seed"], job["update"]])
            take = min(SUBSET_ROWS, replay_index.shape[0])
            subset = rng.choice(replay_index.shape[0], size=take, replace=False)
            subset.sort()

            dup_replay = duplicate_block(replay_index, label="replay")
            dup_fresh = duplicate_block(fresh_index, label="fresh")

            block = {
                **{k: v for k, v in job.items() if not isinstance(v, Path)},
                "status": "scored",
                "paths": {k: str(job[k].relative_to(ROOT).as_posix())
                          for k in ("replay_bank", "fresh_bank", "fresh_scores")},
                "identity": {
                    "policy_checkpoint_state": entry.get("state_sha256"),
                    "parent_checkpoint_state": parent_entry.get("state_sha256"),
                    "same_parent": True,
                    "parent_scores_agreement": parent_agreement},
                "replay_full": drop_block(scored_parent, scored_policy,
                                          label="replay bank, all rows",
                                          rows=replay_index.shape[0]),
                "replay_matched": drop_block(scored_parent[subset], scored_policy[subset],
                                             label=f"replay bank, {take} matched rows",
                                             rows=take),
                "fresh": drop_block(fresh_parent, fresh_policy,
                                    label="independent fresh bank",
                                    rows=fresh_index.shape[0]),
                "difference_matched": bootstrap_difference(
                    (scored_parent[subset] - scored_policy[subset]),
                    (fresh_parent - fresh_policy),
                    seed=SUBSET_SEED + job["seed"]),
                "duplicates": {"replay": {k: v for k, v in dup_replay.items() if k != "hashes"},
                               "fresh": {k: v for k, v in dup_fresh.items() if k != "hashes"}},
                "overlap": overlap(dup_replay["hashes"], dup_fresh["hashes"]),
                "subset_rule": (f"a seeded draw of {take} rows without replacement from the "
                                f"replay bank, entropy [{SUBSET_SEED}, seed, update]. Declared "
                                "before scoring and not re-rolled."),
            }
            results.append(block)
            print("  replay mean %.4f  fresh mean %.4f  diff %.4f [%.4f, %.4f]" % (
                block["replay_matched"]["mean"], block["fresh"]["mean"],
                block["difference_matched"]["difference"],
                block["difference_matched"]["lower"], block["difference_matched"]["upper"]),
                flush=True)

    document = {
        "record_kind": "replay_memorisation_analysis",
        "schema_version": "her2-next-flight-analysis/1",
        "run_root": args.run_root,
        "historical_root": args.historical,
        "statistic": "log P_parent(x) - log P_policy(x), per sequence",
        "question": ("does an FKL arm assign systematically less probability loss to the "
                     "bank it replayed against than to an independent draw from the same "
                     "parent?"),
        "reading": ("a consistently smaller drop on the training bank indicates finite-bank "
                    "overfitting. Overlapping distributions mean no gap was detected AT THIS "
                    "RESOLUTION, which is not the same as no gap."),
        "constraint": ("the fresh bank remains an audit of frozen checkpoints. Nothing here "
                       "selects a setting, and no result feeds back into training."),
        "results": results,
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out_dir / "replay_memorisation.json", document)
    print(f"\nwrote {out_dir / 'replay_memorisation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
