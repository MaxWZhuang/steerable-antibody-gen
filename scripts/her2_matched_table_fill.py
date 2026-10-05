#!/usr/bin/env python
"""Fill the two cells the matched arm table cannot join from saved results.

The flight persisted per-row scores for every Block A cell (`row_scores/*.npz`),
the independent audit for all 45 audited checkpoints, and the yield curves. Two
columns of the requested comparison are genuinely absent:

1. **The fixed/fresh tail arms have no row scores.** `her2_fixed_vs_fresh_ranking`
   computed macro-AP and threw the per-row vector away, so yield and class mass
   are not recoverable for those six trajectories without rescoring.
2. **Off-library fraction exists only at u1000, and only for four entries.** The
   flight's coupling stage drew 50k generation banks for `parent`, `IPO_0`,
   `IPO_FKL` and `IPO_TAIL` at u1000. Nothing at u3750, nothing for
   `continued_sft`, nothing for the fixed/fresh arms.

This fills both under the protocol the flight used, so the filled cells are
comparable with the joined ones rather than merely adjacent to them:

* the same original validation panel, the same `score_batch_size`;
* the same `q_generation_finalist` role at 50,000 draws, temperature 1.0, the
  same sampler-versus-scorer parity check per bank, duplicates retained;
* the same stream descriptor, so the twelve cells the flight already drew are
  **redrawn under the identical seed** and reproduce bitwise. Those twelve are
  the control: if they do not reproduce, nothing else here is comparable either.

Nothing here selects anything. Every checkpoint was named before this ran, and
`q_generation_finalist` is a `may_influence_selection=False` role.

Conditionals are not retained -- this needs the sampled index, not the coupling
tensors -- which is why a bank here costs ~55s rather than ~72s. The index is
drawn before conditionals either way, so the bitwise reproduction is unaffected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_banks as nf_banks  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_campaign as campaign  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_contract as contract  # noqa: E402
from smallAntibodyGen.experiments import her2_nf_spec as spec  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

BANK_ROLE = "q_generation_finalist"
BANK_ROWS = 50_000
#: The four entries the flight already drew at u1000; redrawn here as the control.
FLIGHT_DREW = {("parent", 0), ("IPO_0", 1000), ("IPO_FKL", 1000), ("IPO_TAIL", 1000)}


def inventory(seed, run_root, historical, experiment, repository_root):
    """The seven families of the matched comparison, at the two common endpoints.

    ``name`` reproduces the flight's own entry naming for the cells it drew, so
    the stream descriptor -- and therefore the draw seed -- matches exactly.
    """
    entries = [("parent", 0,
                repository_root / "outputs" / "her2_posttrain_20260918"
                / f"policy_sft_seed{seed}" / "epoch_3.pt")]
    families = [
        ("continued_sft", historical / "trajectories" / f"continued_sft_lambda0p1_seed{seed}"),
        ("IPO_0", historical / "trajectories" / f"ipo_lambda0_seed{seed}"),
        ("IPO_FKL", historical / "trajectories" / f"ipo_lambda10_seed{seed}"),
        ("IPO_TAIL", run_root / "trajectories" / f"A_IPO_TAIL_seed{seed}"),
        ("FB_fixed", experiment / "trajectories" / f"FB_fixed_seed{seed}"),
        ("FB_fresh", experiment / "trajectories" / f"FB_fresh_seed{seed}"),
    ]
    for family, trajectory in families:
        for update in (1000, 3750):
            entries.append((family, update, trajectory / f"endpoint_update{update}.pt"))
    return entries


def flight_entry_name(family, update, seed):
    """The name the flight's coupling stage used, for descriptor parity."""
    if family == "parent":
        return f"parent_seed{seed}"
    prefix = "A_" if family in ("IPO_0", "IPO_FKL", "IPO_TAIL") else ""
    return f"{prefix}{family}@u{update}_seed{seed}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/her2_next_flight.json")
    parser.add_argument("--run-root", default="outputs/her2_next_flight_20260922/production_03")
    parser.add_argument("--experiment", default="outputs/her2_fixed_vs_fresh_20260923")
    parser.add_argument("--output", default="outputs/her2_matched_table_20260923")
    parser.add_argument("--phase", choices=("score", "banks", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = ROOT / args.output
    (out / "row_scores").mkdir(parents=True, exist_ok=True)
    (out / "banks").mkdir(parents=True, exist_ok=True)

    context = spec.resolve_context(ROOT, config_path=ROOT / args.config,
                                   run_root=ROOT / args.run_root)
    services = campaign.build_services(context, device=args.device)
    historical = services.historical_root()
    batch = int(context.config["inference"]["score_batch_size"])
    sample_batch = int(context.config["inference"]["sample_batch_size"])

    frame = services.split("val")
    index = services.index("val")
    classes = np.asarray(frame["class"])
    positive = classes == "high"
    strata = services.original_strata()["labels"]

    # A private registry. Descriptors match the flight's, so a shared descriptor
    # yields a shared seed -- which is the point -- but nothing is written into
    # the landed run root.
    registry = nf_banks.StreamRegistry()

    scored_records, bank_records = [], []
    for seed in seeds:
        for family, update, checkpoint in inventory(
                seed, ROOT / args.run_root, historical, ROOT / args.experiment,
                context.repository_root):
            if not Path(checkpoint).is_file():
                print(f"  MISSING {family} seed{seed} u{update}", flush=True)
                continue
            needs_score = args.phase in ("score", "all") and family.startswith("FB_")
            needs_bank = args.phase in ("banks", "all")
            if not (needs_score or needs_bank):
                continue
            policy = services.policy(Path(checkpoint))

            if needs_score:
                block = policy.score(index, batch_size=batch)
                target = out / "row_scores" / f"{family}_u{update}_seed{seed}.npz"
                paths.write_arrays(target, {
                    "row_id": contract.index_hashes(index).astype("S64"),
                    "source_split": np.asarray("val"),
                    "model": np.asarray(f"{family}@u{update}_seed{seed}"),
                    "class": classes.astype("S4"),
                    "sum_log_probability": np.asarray(block["sum_log_probability"],
                                                      dtype=np.float64),
                    "mean_log_probability": np.asarray(block["mean_log_probability"],
                                                       dtype=np.float64),
                    "positive": positive.astype(np.int8),
                    "original_train_distance_label": np.asarray(strata)})
                scored_records.append({"family": family, "seed": seed, "update": update,
                                       "rows": int(index.shape[0]),
                                       "artifact": target.name})
                print(f"  seed{seed} {family:14s} u{update:<5d} scored {index.shape[0]} rows",
                      flush=True)

            if needs_bank:
                name = flight_entry_name(family, update, seed)
                lineage = registry.stream(domain=BANK_ROLE, role=BANK_ROLE, regime=None,
                                          parent=f"parent::seed{seed}", model=name, repeat=0)
                bank, document = nf_banks.draw_bank(
                    policy, role=BANK_ROLE, parent_id=f"parent::seed{seed}",
                    parent_state_sha256=document_digest(policy),
                    seed_lineage_record=lineage, rows=BANK_ROWS,
                    retain_conditionals=False, batch_size=sample_batch)
                coverage = services._panel_coverage(bank.index)
                cores = data_lib.decode_cores(bank.index)
                unique = len(set(cores))
                record = {"family": family, "seed": seed, "update": update,
                          "flight_entry": name, "draw_seed": int(lineage["seed"]),
                          "rows": int(bank.rows),
                          "redrawn_from_flight": (family, update) in FLIGHT_DREW,
                          "index_sha256": document["index_sha256"],
                          "parity_rows_above_atol": document["parity"]["rows_above_atol"],
                          "known_panel": coverage,
                          "off_library_fraction": coverage["outside_known_panel"] / bank.rows,
                          "unique_cores": unique, "unique_fraction": unique / bank.rows,
                          "mean_sum_log_probability": document["mean_sum_log_probability"],
                          "timings": document["timings"]}
                bank_records.append(record)
                np.savez_compressed(out / "banks" / f"{family}_u{update}_seed{seed}.npz",
                                    index=bank.index,
                                    sum_log_probability=bank.sum_log_probability)
                print("  seed%d %-14s u%-5d off-library %.4f  unique %.4f%s" % (
                    seed, family, update, record["off_library_fraction"],
                    record["unique_fraction"],
                    "   [redraw of a flight bank]" if record["redrawn_from_flight"] else ""),
                    flush=True)

            del policy
            services.release_transient()

    document = {
        "record_kind": "matched_table_fill",
        "purpose": ("the two columns of the matched arm comparison that are not a join of "
                    "saved flight results"),
        "bank_protocol": {"role": BANK_ROLE, "rows": BANK_ROWS, "temperature": 1.0,
                          "duplicates_retained": True,
                          "conditionals_retained": False,
                          "selection_note": ("may_influence_selection is False for this role and "
                                             "nothing here selects; every checkpoint was named "
                                             "before this ran")},
        "off_library_definition": ("draws whose core is in neither the training nor the "
                                   "validation split, over all 50,000 draws including repeats. "
                                   "No sealed-test labels are read."),
        "reproduction_control": ("the twelve cells the flight already drew are redrawn under the "
                                 "identical stream descriptor and seed; index_sha256 must match "
                                 "the flight's coupling.json for those cells"),
        "row_scores": scored_records, "banks": bank_records,
        "stream_registry": registry.document(),
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "fill.json", document)
    print(f"\nwrote {out / 'fill.json'}")
    return 0


def document_digest(policy):
    """The loaded policy's state digest, as the flight records it."""
    from smallAntibodyGen.experiments import her2_nf_services as services_lib
    return services_lib._state_digest(policy)


if __name__ == "__main__":
    raise SystemExit(main())
