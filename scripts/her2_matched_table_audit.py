#!/usr/bin/env python
"""Put the fixed/fresh arms on the flight's own audit bank.

The flight draws ONE 50,000-row `final_preservation` bank per parent seed and
scores every Block A checkpoint on it, so the whole audit column is paired
across arms within a seed. The fixed-versus-fresh experiment drew its own audit
bank per seed -- correctly, because it had to be drawn after its own checkpoints
were named -- which means its audit numbers sit on different rows.

Verified, seed 20260918: the two banks differ (index digests `bf45a529...` vs
`aa65b096...`, parent mean sum log probability -14.2969 vs -14.2728). At 50,000
rows that is Monte Carlo, not a defect, but it is a difference the matched table
would otherwise carry silently into every fixed/fresh-versus-Block-A comparison.

So the six fixed/fresh trajectories are rescored here on the flight's bank. The
experiment's own audit stands unchanged and remains the record for the
fixed-versus-fresh contrast; this is an additional read for the joined table
only. Nothing is selected and no checkpoint is renamed.

Convention, identical to both sources: drop = log p_parent - log p_policy,
tail events counted INCLUSIVELY at `drop >= ln(10)` and `drop >= ln(100)`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_metrics as metrics  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

TENFOLD = math.log(10.0)
HUNDREDFOLD = math.log(100.0)
#: Any Block A audit file at a seed carries that seed's shared bank.
BANK_SOURCE = "A_IPO_0@u1000_seed{seed}.npz"


def tail_block(drop):
    ten, hundred = int((drop >= TENFOLD).sum()), int((drop >= HUNDREDFOLD).sum())
    rows = int(drop.size)
    return {"rows": rows, "mean_drop": float(drop.mean()),
            "standard_error": float(drop.std(ddof=1) / math.sqrt(rows)),
            "median": float(np.median(drop)),
            "fraction_below_parent": float((drop > 0).mean()),
            "max_drop": float(drop.max()),
            "quantiles": {q: float(np.quantile(drop, float(q)))
                          for q in ("0.5", "0.9", "0.95", "0.99")},
            "events": {
                "tenfold": {"count": ten, "rate": ten / rows, "threshold_nats": TENFOLD,
                            "wilson": metrics.wilson_interval(ten, rows)},
                "hundredfold": {"count": hundred, "rate": hundred / rows,
                                "threshold_nats": HUNDREDFOLD,
                                "wilson": metrics.wilson_interval(hundred, rows)}},
            "comparison": ">=",
            "convention_note": ("inclusive tail convention, matching the flight's "
                                "preservation audit and the fixed-versus-fresh audit")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--experiment", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--output", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = ROOT / args.output
    (out / "preservation").mkdir(parents=True, exist_ok=True)
    run_root = ROOT / args.run_root

    context = spec.resolve_context(ROOT, config_path=ROOT / args.config, run_root=run_root)
    services = campaign.build_services(context, device=args.device)
    batch = int(context.config["inference"]["score_batch_size"])

    records = []
    for seed in seeds:
        source = run_root / "preservation" / BANK_SOURCE.format(seed=seed)
        if not source.is_file():
            print(f"  MISSING audit bank for seed {seed}", flush=True)
            continue
        bank = np.load(source)
        index = np.asarray(bank["core_index"])
        parent_scores = np.asarray(bank["parent_sum_log_probability"], dtype=np.float64)
        for arm in ("fixed", "fresh"):
            for update in (1000, 3750):
                checkpoint = (ROOT / args.experiment / "trajectories" / f"FB_{arm}_seed{seed}"
                              / f"endpoint_update{update}.pt")
                if not checkpoint.is_file():
                    print(f"  MISSING FB_{arm} seed{seed} u{update}", flush=True)
                    continue
                policy = services.policy(checkpoint)
                scored = np.asarray(policy.score(index, batch_size=batch)["sum_log_probability"],
                                    dtype=np.float64)
                del policy
                services.release_transient()
                drop = parent_scores - scored
                block = tail_block(drop)
                np.savez_compressed(
                    out / "preservation" / f"FB_{arm}@u{update}_seed{seed}.npz",
                    parent_sum_log_probability=parent_scores,
                    policy_sum_log_probability=scored, log_ratio=drop, core_index=index)
                records.append({"arm": f"FB_{arm}", "seed": seed, "update": update,
                                "bank": source.name, "drop": block})
                print("  seed%d FB_%-6s u%-5d mean %.4f  tenfold %.5f  hundredfold %.5f" % (
                    seed, arm, update, block["mean_drop"],
                    block["events"]["tenfold"]["rate"],
                    block["events"]["hundredfold"]["rate"]), flush=True)

    document = {"record_kind": "matched_table_audit_on_flight_bank",
                "bank": ("the flight's own 50,000-row final_preservation bank, one per parent "
                         "seed, shared by every Block A checkpoint at that seed"),
                "why": ("the fixed-versus-fresh experiment audited on its own bank; rescoring "
                        "here puts the whole joined audit column on one bank per seed"),
                "supersedes": "nothing. The experiment's own audit remains its record.",
                "results": records, "recorded_at": paths.utc_now()}
    paths.write_json(out / "audit_on_flight_bank.json", document)
    print(f"\nwrote {out / 'audit_on_flight_bank.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
