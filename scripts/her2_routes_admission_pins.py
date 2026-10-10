#!/usr/bin/env python
"""Build the admission-phase config and its pins from the finished origin flights. Read-only.

    python scripts/her2_routes_admission_pins.py \\
        --base configs/experiments/her2_generative_routes_joint.json \\
        --out  configs/experiments/her2_generative_routes_admitted_gate.json \\
        --origin ablang2_masked_editor F:/.../runs/t1_rev01/flight \\
                 configs/experiments/her2_generative_routes.json development_affinity_spearman_high \\
        --origin piggen_causal_infiller F:/.../runs/t1_rev01/flight \\
                 configs/experiments/her2_generative_routes.json development_affinity_spearman_high \\
        --origin dplm_150m_masked F:/.../runs/dplm_train_rev01/flight_dplm_train \\
                 configs/experiments/her2_generative_routes_dplm_train.json \\
                 hr17_fixed_path_high_minus_null_negative_spearman

Each ``--origin`` is ``ROUTE FLIGHT_ROOT CONFIG SELECTOR``. CONFIG is relative to the origin flight's
own ``source_snapshot`` directory: the frozen copy it actually ran, not whatever the repository's file
of that name says today. The script reads each origin's ``best.pt`` (CPU, memory-mapped) only to take
its step, arm, seed and schedule digest; it hashes the file; and it refuses a pin that would not
verify (a live route, ``her2`` or ``dplm`` block that differs from the origin's, prose excluded).

Nothing is written into an origin flight. The output is a new config file only.

What the written config is, and is not. It enables the four routes of the admission phase, drops the
five-way ``competition`` block and the blocks only the held stages read, and carries the code-held
``admission_phase`` declaration plus the pins. It is not competition admission and does not clear the
joint hold; see ``her2_routes_spec.admission_phase``. The gate numbers are the owner's, unmodified
(0.05 and 0.02 / 0.02 / 0.0) and are refused if they differ.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_routes_admission as admission  # noqa: E402
from smallAntibodyGen.experiments import her2_routes_spec as spec  # noqa: E402

CAMPAIGN_ID = "her2_generative_routes_admitted_gate_20261006"
LEAF = "flight_admitted_gate"
REVISION = "admitted_gate_rev02"
#: rev01 trained dWJS in a fresh process whose first split() call re-stamped the frozen split record
#: ("reconfirmed"), moving the digest every identity is bound to. Fixed in
#: FlightServices.persist_split_freeze; the rev01 run root is retained read-only and never resumed.
SUPERSEDES = "admitted_gate_rev01"
#: Blocks only the held stages read. Left out so they cannot be mistaken for something this runs.
HELD_ONLY_BLOCKS = ("competition", "posttrain", "confirmation", "controls", "priors")


def build_config(base, *, config_logical):
    """The admission-phase config WITHOUT pins, derived from the joint ``base``."""
    config = copy.deepcopy(base)
    for key in HELD_ONLY_BLOCKS:
        config.pop(key, None)
    config["campaign_id"] = CAMPAIGN_ID
    config["config_path"] = config_logical
    config["run_root_leaf"] = LEAF
    config["opt_in_note"] = (
        "ADMISSION-PHASE config: runs route_gate on three routes that FINISHED in other flights "
        "(ablang2_masked_editor and piggen_causal_infiller from t1_rev01, dplm_150m_masked from "
        "dplm_train_rev01), each at its own best.pt, and retrains dwjs_energy under energy contract "
        "v2 because its finished checkpoint predates that contract and is refused. It is NOT "
        "competition admission and does not clear the five-way hold: controls, priors, "
        "post-training, decoding, confirmation, the report and verify are held in code, and a "
        "finished phase is never a complete flight. dplm_650m_lora_masked is tabled. Every number "
        "this produces is selected on the development panel.")
    routes = config["routes"]
    routes["enabled"] = list(spec.ADMISSION_PHASE_ROUTES)
    routes["enabled_note"] = ("the four routes of the admission phase, in the order the code "
                              "requires. The route blocks are copied verbatim from the origin "
                              "flights' configs apart from prose; the admitted routes are "
                              "compared with the origin's by content on every load.")
    routes.pop("dplm_650m_lora_masked", None)
    excluded = dict(routes.get("excluded") or {})
    excluded["dplm_650m_lora_masked"] = (
        "tabled by the owner on 2026-10-05: its training was stopped at update 1,500 of 20,000 "
        "(a truncation record sits beside its run root), so it has no finished checkpoint to admit.")
    routes["excluded"] = excluded
    config["revisions"] = {
        "current": REVISION, "kind": "development", "supersedes": SUPERSEDES, "frozen_after": [],
        "reason": ("the second revision of the admission phase: route_gate on finished routes. "
                   "admitted_gate_rev01 was stopped after a fresh process re-stamped its frozen "
                   "split record (the content was unchanged, the digest moved, so its dWJS "
                   "checkpoints and the admitted routes could not share one split); that is fixed "
                   "in FlightServices.persist_split_freeze and the rev01 root is retained "
                   "read-only and never resumed. Independent of t1_rev01, dplm_train_rev01 and "
                   "joint_rev01: their run roots are only read."),
        "rule": base["revisions"]["rule"]}
    config[spec.ADMISSION_PHASE_KEY] = spec.admission_phase_declaration()
    return config


def derive_pins(config, origins):
    """``route -> pin``, each taken from the origin flight's own files."""
    import torch
    pins = {}
    for route, flight, origin_config, selector in origins:
        checkpoint = Path(flight) / "bakeoff" / route / "best.pt"
        payload = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
        identity = payload["identity"]
        pins[route] = admission.derive_pin(
            flight, route=route, origin_config=origin_config, checkpoint=checkpoint,
            checkpoint_step=int(payload["step"]), arm=identity["arm"], seed=int(identity["seed"]),
            schedule_sha256=identity.get("schedule_sha256"), selector=selector, live_config=config)
        print(f"{route}: step {payload['step']} best_step {payload.get('best_step')} "
              f"sha256 {pins[route]['checkpoint_sha256'][:16]} bytes {pins[route]['checkpoint_bytes']}",
              flush=True)
    return pins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--origin", action="append", nargs=4, required=True,
                        metavar=("ROUTE", "FLIGHT_ROOT", "CONFIG", "SELECTOR"))
    args = parser.parse_args(argv)
    base_path, out_path = Path(args.base), Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; refusing to overwrite a committed config")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    logical = out_path.resolve().relative_to(ROOT).as_posix()
    config = build_config(base, config_logical=logical)
    named = {origin[0] for origin in args.origin}
    if named != set(spec.ADMISSION_ADMITTED_ROUTES):
        raise SystemExit(f"--origin must name exactly {list(spec.ADMISSION_ADMITTED_ROUTES)}, "
                         f"got {sorted(named)}")
    ordered = sorted(args.origin, key=lambda item: spec.ADMISSION_ADMITTED_ROUTES.index(item[0]))
    config[spec.ADMISSION_PINS_KEY] = derive_pins(config, ordered)
    spec.admission_phase(config)  # refuses a config the code would refuse
    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_path} ({len(config[spec.ADMISSION_PINS_KEY])} pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
