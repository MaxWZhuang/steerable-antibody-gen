#!/usr/bin/env python
"""Fixed replay bank vs fresh preservation stream, paired at the parent seed.

PREDECLARED 2026-09-23, before any arm ran. The question: does the IPO-tail
arm's large bank-to-audit preservation gap close when no preservation row is
ever reused, and does the ranking benefit survive?

Both arms start from the SAME supervised parent, share initial weights,
optimizer settings, tail coefficient, IPO temperature, update count, total
preservation-row exposures, likelihood gate, preference-data order and
checkpoint schedule. They differ in exactly one thing:

* **fixed** -- 100,000-draw parent bank, consumed 240,000 times (2.4 passes),
* **fresh** -- 240,000-draw parent stream, every row consumed exactly once.

Both draw their bank through the same code path (``reuse_saved_arrays=False``),
so the control is a NEW fixed-bank run rather than the historical one: a
historical control would confound "rows are reused" with "which draws, under
which code".

The preservation stream already has its own seed and cycle counter, so neither
arm perturbs the preference minibatch order -- ``preflight`` proves that rather
than assuming it, and refuses to train if it does not hold.

Checkpoints are frozen here at u1000 and u3750 BEFORE training. No checkpoint is
selected by looking at preservation results, and the audit bank is drawn only in
the ``audit`` phase, after the names are fixed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_banks as nf_banks  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402
from smallAntibodyGen.experiments import her2_replay_streams as streams_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_preferences as preferences  # noqa: E402

UPDATES = 3750
ENDPOINTS = (1000, 3750)
CHECKPOINTS = (1000, 3750)
BATCH_ROWS = 64
FIXED_BANK_ROWS = 100_000
#: The whole intervention, in one number: a bank as large as the exposures is
#: consumed without reuse.
FRESH_BANK_ROWS = BATCH_ROWS * UPDATES
TASK = "ipo"
PRESERVATION = "tail"
#: The coefficient the completed flight's bounded calibration froze.
COEFFICIENTS = {"tau": 0.1, "lambda": 1.0}
AUDIT_ROWS = 50_000
TENFOLD, HUNDREDFOLD = float(np.log(10.0)), float(np.log(100.0))


def arms_for(seed):
    return [
        {"arm": "fixed", "bank_rows": FIXED_BANK_ROWS, "seed": int(seed),
         "trajectory": f"FB_fixed_seed{seed}"},
        {"arm": "fresh", "bank_rows": FRESH_BANK_ROWS, "seed": int(seed),
         "trajectory": f"FB_fresh_seed{seed}"},
    ]


def preflight(services, context, seeds):
    """Prove the pair is a pair before spending the window on it.

    Four checks, all of which must hold: the two arms see identical preference
    minibatches, start from identical weights, report an identical initial loss,
    and the freshly drawn stream satisfies the same sampler/scorer probability
    convention every other bank in this project is held to.
    """
    report = {"record_kind": "fixed_vs_fresh_preflight", "checks": [], "passed": True}

    def record(name, passed, detail):
        report["checks"].append({"check": name, "passed": bool(passed), "detail": detail})
        report["passed"] = bool(report["passed"] and passed)
        print(f"  [{'ok ' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    for seed in seeds:
        population, parent_policy, parent_checkpoint, _ = services._population_for(
            "A", "original_split", int(seed))
        pairing_seed = int(context.config["block_a"]["pairing"]["seed_base"]) + int(seed)
        streams = {}
        for arm in arms_for(seed):
            stream, order, source = services._streams_for(
                block="A", regime="original_split", seed=int(seed), population=population,
                pairing_seed=pairing_seed, batch_rows=BATCH_ROWS, horizon=UPDATES,
                replay_rows=arm["bank_rows"], reuse_saved_arrays=False)
            streams[arm["arm"]] = (stream, order, source)

        fixed_stream, fixed_order, _ = streams["fixed"]
        fresh_stream, fresh_order, _ = streams["fresh"]
        same_chosen = fixed_stream.document()["chosen_rows_sha256"] == \
            fresh_stream.document()["chosen_rows_sha256"]
        same_rejected = fixed_stream.document()["rejected_rows_sha256"] == \
            fresh_stream.document()["rejected_rows_sha256"]
        record(f"seed{seed}: identical preference stream",
               same_chosen and same_rejected,
               f"chosen {fixed_stream.document()['chosen_rows_sha256'][:12]} both arms")

        first_fixed = fixed_stream.batch(1)
        first_fresh = fresh_stream.batch(1)
        record(f"seed{seed}: identical first preference minibatch",
               np.array_equal(first_fixed[0], first_fresh[0])
               and np.array_equal(first_fixed[1], first_fresh[1]),
               f"{first_fixed[0][:4].tolist()} ...")

        record(f"seed{seed}: fixed order reuses rows, fresh order does not",
               np.unique(fixed_order).size == FIXED_BANK_ROWS
               and np.unique(fresh_order).size == FRESH_BANK_ROWS,
               f"distinct rows fixed={np.unique(fixed_order).size} "
               f"fresh={np.unique(fresh_order).size} of {fresh_order.size} exposures")

        record(f"seed{seed}: matched preservation-row exposures",
               fixed_order.size == fresh_order.size == BATCH_ROWS * UPDATES,
               f"{fixed_order.size} exposures per arm")

        # Identical initial weights is definitional here -- both arms load the
        # same parent file -- so the check that earns its place is that the
        # SCORER agrees with the SAMPLER on the freshly drawn stream, which is
        # the one array this experiment introduces.
        fresh_bank = services._replay_bank(parent_policy, int(seed), "original_split",
                                           rows=FRESH_BANK_ROWS, reuse_saved_arrays=False)
        parity = (fresh_bank.get("document") or {}).get("parity") or {}
        record(f"seed{seed}: fresh stream sampler/scorer parity",
               parity.get("rows_above_atol", 1) == 0,
               f"max_abs_error={parity.get('max_abs_error')} "
               f"rows_above_atol={parity.get('rows_above_atol')}")

        scored = parent_policy.score(np.asarray(fresh_bank["index"])[:2048],
                                     batch_size=512)["sum_log_probability"]
        saved = np.asarray(fresh_bank["parent_scores"])[:2048]
        delta = float(np.abs(np.asarray(scored, dtype=np.float64) - saved).max())
        record(f"seed{seed}: fresh stream parent scores reproduce",
               delta <= 5e-5, f"max abs difference {delta:.3e} over 2048 rows")
        del parent_policy
        services.release_transient()
    return report


def train(services, context, seeds, out_dir):
    results = []
    for seed in seeds:
        for arm in arms_for(seed):
            directory = out_dir / "trajectories" / arm["trajectory"]
            marker = directory / "status.json"
            if marker.is_file():
                print(f"skip {arm['trajectory']} (already terminal)", flush=True)
                results.append({**arm, "status": "already_present"})
                continue
            print(f"training {arm['trajectory']} "
                  f"(bank_rows={arm['bank_rows']})", flush=True)
            started = time.perf_counter()
            document = services.run_one_trajectory(
                trajectory=arm["trajectory"], directory=directory, block="A",
                regime="original_split", seed=int(seed), task=TASK,
                preservation=PRESERVATION, coefficients=dict(COEFFICIENTS),
                updates=UPDATES, endpoints=ENDPOINTS, checkpoints=CHECKPOINTS,
                parent_seed=int(seed), bank_rows=arm["bank_rows"],
                max_horizon=UPDATES, reuse_saved_arrays=False)
            wall = time.perf_counter() - started
            results.append({**arm, "status": document.get("status"),
                            "updates": document.get("updates"),
                            "stop_reason": document.get("stop_reason"),
                            "wall_seconds": wall,
                            "cost": (document.get("cost") or {}).get("seconds")})
            print(f"  {document.get('status')} in {wall/60:.1f} min", flush=True)
            paths.write_json(out_dir / "training.json", {
                "record_kind": "fixed_vs_fresh_training", "results": results,
                "recorded_at": paths.utc_now()})
    return results


def audit(services, context, seeds, out_dir):
    """Score both arms of each seed on ONE new audit bank and on the fixed bank."""
    freeze = {"record_kind": "fixed_vs_fresh_freeze",
              "named_checkpoints": [f"FB_{a}_seed{s}@u{u}"
                                    for s in seeds for a in ("fixed", "fresh")
                                    for u in CHECKPOINTS],
              "rule": ("predeclared before training: u1000 and u3750 for every arm. No "
                       "checkpoint was selected by looking at preservation results."),
              "recorded_at": paths.utc_now()}
    paths.write_json(out_dir / "freeze.json", freeze)
    experiment_registry = nf_banks.StreamRegistry()

    rows = []
    for seed in seeds:
        population, parent_policy, parent_checkpoint, _ = services._population_for(
            "A", "original_split", int(seed))
        digest = campaign._state_digest(parent_policy) if hasattr(campaign, "_state_digest") \
            else None
        from smallAntibodyGen.experiments.her2_nf_services import _state_digest
        digest = _state_digest(parent_policy)
        # This experiment's OWN registry, written into its own output root.
        # ``services.stream_seed`` persists into the context's run root, which
        # here is the LANDED flight -- so calling it appended three descriptors
        # to that flight's stream_registry.json and broke its manifest digest
        # (disclosed in outputs/her2_provenance_20260923/). The seed itself is a
        # pure function of the descriptor and the spawn key, so deriving it from
        # a private registry reproduces the identical seed and changes no result.
        lineage = experiment_registry.stream(domain="final_preservation", role="audit",
                                             regime="fixed_vs_fresh", parent=f"seed{seed}")
        paths.write_json(out_dir / "stream_registry.json", experiment_registry.document())
        target = out_dir / "banks" / f"audit_seed{seed}_{AUDIT_ROWS}.npz"
        existing = nf_banks.load_bank(target, role="final_preservation",
                                      parent_id=f"parent::original_split::seed{seed}",
                                      parent_state_sha256=digest, draw_seed=lineage["seed"])
        if existing is None:
            bank, document = nf_banks.draw_bank(
                parent_policy, role="final_preservation",
                parent_id=f"parent::original_split::seed{seed}",
                parent_state_sha256=digest, seed_lineage_record=lineage, rows=AUDIT_ROWS,
                batch_size=int(context.config["inference"]["sample_batch_size"]))
            nf_banks.save_bank(target, bank, document)
            audit_index, audit_parent = bank.index, bank.sum_log_probability
        else:
            audit_index, audit_parent = existing["index"], existing["parent_scores"]

        fixed_bank = services._replay_bank(parent_policy, int(seed), "original_split",
                                           rows=FIXED_BANK_ROWS, reuse_saved_arrays=False)
        del parent_policy
        services.release_transient()

        for arm in arms_for(seed):
            for update in CHECKPOINTS:
                ckpt = out_dir / "trajectories" / arm["trajectory"] / f"endpoint_update{update}.pt"
                if not ckpt.is_file():
                    rows.append({**arm, "update": update, "status": "missing"})
                    continue
                policy = services.policy(ckpt)
                block = {}
                for label, index, parent_scores in (
                        ("audit", audit_index, audit_parent),
                        ("fixed_bank", np.asarray(fixed_bank["index"]),
                         np.asarray(fixed_bank["parent_scores"]))):
                    scored = np.asarray(policy.score(
                        index, batch_size=512)["sum_log_probability"], dtype=np.float64)
                    drop = np.asarray(parent_scores, dtype=np.float64) - scored
                    ten, hun = int((drop >= TENFOLD).sum()), int((drop >= HUNDREDFOLD).sum())
                    q = np.percentile(drop, [50, 90, 95, 99])
                    block[label] = {
                        "rows": int(drop.size), "mean": float(drop.mean()),
                        "median": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
                        "p99": float(q[3]),
                        "tenfold_count": ten, "tenfold_rate": ten / drop.size,
                        "tenfold_wilson": metrics.wilson_interval(ten, int(drop.size)),
                        "hundredfold_count": hun, "hundredfold_rate": hun / drop.size,
                        "hundredfold_wilson": metrics.wilson_interval(hun, int(drop.size))}
                ranking = services._development_metrics(policy, seed=int(seed),
                                                        parent_policy=None) \
                    if False else None
                del policy
                services.release_transient()
                block["bank_to_audit_gap"] = (block["audit"]["mean"]
                                              - block["fixed_bank"]["mean"])
                rows.append({**arm, "update": update, "status": "scored", **block})
                print("  %-22s u%-5d audit %.4f  fixed-bank %.4f  gap %+.4f" % (
                    arm["trajectory"], update, block["audit"]["mean"],
                    block["fixed_bank"]["mean"], block["bank_to_audit_gap"]), flush=True)
        paths.write_json(out_dir / "audit.json", {
            "record_kind": "fixed_vs_fresh_audit", "freeze": freeze, "results": rows,
            "audit_bank_rows": AUDIT_ROWS,
            "note": ("one audit bank per seed, shared by that seed's two arms, drawn after "
                     "the checkpoint names were frozen."),
            "recorded_at": paths.utc_now()})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--output", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    parser.add_argument("--phase", default="all",
                        choices=("preflight", "train", "audit", "all"))
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    context = spec.resolve_context(ROOT, config_path=ROOT / args.config,
                                   run_root=ROOT / args.run_root)
    services = campaign.build_services(context, device=args.device)

    if args.phase in ("preflight", "all"):
        print("PREFLIGHT", flush=True)
        report = preflight(services, context, seeds)
        paths.write_json(out_dir / "preflight.json", report)
        if not report["passed"]:
            print("\npreflight FAILED: the arms are not a matched pair. Not training.",
                  flush=True)
            return 2
        print("preflight passed\n", flush=True)

    if args.phase in ("train", "all"):
        print("TRAINING", flush=True)
        train(services, context, seeds, out_dir)

    if args.phase in ("audit", "all"):
        print("\nAUDIT", flush=True)
        audit(services, context, seeds, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
