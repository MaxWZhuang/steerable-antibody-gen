#!/usr/bin/env python
"""Freeze the backgrounds and alternatives for the fixed-background site-1 test.

Runs BEFORE any model scores a single one of these sequences. The whole point of
the test is that the background is held exactly fixed, so the set of backgrounds
must not be able to respond to what the scoring finds.

**What is frozen here**

* ``backgrounds`` -- complete nine-residue contexts (every core position except
  site 1), drawn from two sources so background provenance can be separated from
  the effect: the MEASURED labelled library, and the PARENT's own generated bank.
  A background that only the policy generates would make the comparison
  circular, so generated backgrounds come from the parent, never from an arm.
* ``alternatives`` -- all 19 non-G canonical residues at site 1. Complete rather
  than selected: picking "the residues the parent likes" would be a choice made
  on model output, and picking "the residues that move most" would be the result.

Each background is stratified by its own distance to wild type over the nine
fixed positions, so the frozen set spans near-wild-type and far contexts rather
than concentrating wherever the sampler happens to be dense.

The test this enables: for a fixed background b, the full-sequence log odds
``log p(b with G at site 1) - log p(b with X at site 1)`` under each model. That
is a within-background, within-alternative quantity, so it cannot be produced by
an arm merely reweighting which backgrounds it visits -- which is exactly the
loophole the marginal +2pp frequency shift leaves open.

Nothing here scores anything. Output: ``site1_freeze.json`` with the row set and
its digest.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

SITE = 1
TARGET = "G"
SELECTION_SEED = 20260924
PER_SOURCE = 1000


def background_of(index, site):
    return np.delete(np.asarray(index), site, axis=1)


def stratified_pick(backgrounds, count, seed):
    """Spread the pick over distance-to-wild-type strata, not over sampler density."""
    wt = np.array([data_lib.CANONICAL.index(r)
                   for i, r in enumerate(data_lib.WT_CORE) if i != SITE])
    distance = (backgrounds != wt[None, :]).sum(axis=1)
    generator = np.random.default_rng(seed)
    chosen, strata = [], {}
    for value in sorted(set(distance.tolist())):
        rows = np.flatnonzero(distance == value)
        strata[int(value)] = int(rows.size)
    available = {k: v for k, v in strata.items() if v > 0}
    # even allocation across strata, remainder to the better-populated ones
    share = max(1, count // max(1, len(available)))
    for value in sorted(available):
        rows = np.flatnonzero(distance == value)
        take = min(share, rows.size)
        chosen.extend(generator.choice(rows, size=take, replace=False).tolist())
    if len(chosen) < count:
        remaining = np.setdiff1d(np.arange(backgrounds.shape[0]), np.array(chosen, dtype=int))
        extra = generator.choice(remaining, size=min(count - len(chosen), remaining.size),
                                 replace=False)
        chosen.extend(extra.tolist())
    chosen = sorted(set(chosen))[:count]
    return np.asarray(chosen, dtype=int), strata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/her2_functional_20260918")
    parser.add_argument("--banks", default="outputs/her2_matched_table_20260923/banks")
    parser.add_argument("--output", default="outputs/her2_site1_20260923")
    parser.add_argument("--per-source", type=int, default=PER_SOURCE)
    args = parser.parse_args(argv)

    raw, banks = ROOT / args.raw_root, ROOT / args.banks
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "site1_freeze.json"
    if existing.is_file():
        raise SystemExit(f"{existing} already exists. A freeze is verified and reused, never "
                         "rewritten: delete it deliberately or point --output elsewhere.")

    # -- source 1: the measured labelled library -----------------------------
    measured_cores = []
    for split in ("train", "val"):
        measured_cores.extend(data_lib.load_split(raw, split).seq.tolist())
    measured_index = data_lib.encode_cores(sorted(set(measured_cores)))
    measured_backgrounds = np.unique(background_of(measured_index, SITE), axis=0)

    # -- source 2: the PARENT's generated bank, never an arm's ---------------
    generated = []
    for seed in (20260918, 20260919, 20260920):
        path = banks / f"parent_u0_seed{seed}.npz"
        if path.is_file():
            generated.append(np.asarray(np.load(path)["index"]))
    if not generated:
        raise SystemExit("no parent generation banks; run scripts/her2_matched_table_fill.py")
    generated_backgrounds = np.unique(
        background_of(np.concatenate(generated, axis=0), SITE), axis=0)

    # backgrounds present in the measured library are dropped from the generated
    # set so the two sources are disjoint and can be compared.
    measured_keys = {row.tobytes() for row in measured_backgrounds}
    generated_backgrounds = np.array(
        [row for row in generated_backgrounds if row.tobytes() not in measured_keys])

    picked, blocks = {}, {}
    for name, pool in (("measured", measured_backgrounds),
                       ("generated_by_parent", generated_backgrounds)):
        rows, strata = stratified_pick(pool, args.per_source,
                                       SELECTION_SEED + (0 if name == "measured" else 1))
        picked[name] = pool[rows]
        blocks[name] = {"pool": int(pool.shape[0]), "selected": int(rows.size),
                        "distance_strata_in_pool": strata}
        print("%-20s pool %7d  selected %5d" % (name, pool.shape[0], rows.size), flush=True)

    alternatives = [r for r in data_lib.CANONICAL if r != TARGET]
    target_code = data_lib.CANONICAL.index(TARGET)

    # -- materialise the row set ---------------------------------------------
    rows = []
    for source, background_set in picked.items():
        for position, background in enumerate(background_set):
            core = np.insert(background, SITE, target_code)
            rows.append({"source": source, "background_id": position,
                         "residue": TARGET, "core": data_lib.decode_cores(core[None, :])[0]})
            for residue in alternatives:
                variant = np.insert(background, SITE, data_lib.CANONICAL.index(residue))
                rows.append({"source": source, "background_id": position,
                             "residue": residue,
                             "core": data_lib.decode_cores(variant[None, :])[0]})
    digest = hashlib.sha256("\n".join(f"{r['source']}:{r['background_id']}:{r['core']}"
                                      for r in rows).encode()).hexdigest()

    wt = data_lib.WT_CORE
    for record in rows:
        record["wt_distance"] = int(sum(a != b for a, b in zip(record["core"], wt)))
        record["background_wt_distance"] = int(
            sum(a != b for i, (a, b) in enumerate(zip(record["core"], wt)) if i != SITE))

    document = {
        "record_kind": "site1_fixed_background_freeze",
        "frozen_before": "any model scored any of these sequences",
        "site": SITE, "target_residue": TARGET,
        "wild_type_residue": data_lib.WT_CORE[SITE],
        "alternatives": alternatives,
        "alternatives_rule": ("all 19 non-G canonical residues. Complete rather than selected: "
                              "choosing the residues the parent favours would be a choice made "
                              "on model output, and choosing the ones that move most would be "
                              "the result."),
        "background_rule": ("complete nine-residue contexts, stratified by their own distance "
                            "to wild type over those nine positions. Generated backgrounds come "
                            "from the PARENT's banks only -- a background only an arm produces "
                            "would make the comparison circular -- and are disjoint from the "
                            "measured ones."),
        "selection_seed": SELECTION_SEED,
        "sources": blocks,
        "backgrounds": sum(b["selected"] for b in blocks.values()),
        "rows": len(rows),
        "rows_sha256": digest,
        "estimand": ("log p(background with G at site 1) - log p(background with X at site 1), "
                     "the full-sequence log odds, under each parent and each policy. Within "
                     "background and within alternative, so an arm cannot produce it by "
                     "reweighting which backgrounds it visits."),
        "primary_contrast": ("policy minus its OWN parent, on that log odds, paired within "
                             "(background, alternative). Paired t over the three parent seeds."),
        "predictions": {
            "if the marginal shift is a broad behavior":
                "the tail arms raise the G-versus-alternative log odds across most backgrounds "
                "and most alternatives, and the non-tail arms do not",
            "if it is background reweighting":
                "the within-background log odds is unchanged and only the marginal moved",
            "tolerance": ("no threshold is declared for the size. The marginal shift is ~2pp "
                          "and does not translate into a log-odds units prediction, so a size "
                          "claim here would be invented rather than predicted."),
        },
        "table": rows,
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(existing, document)
    print("\nbackgrounds %d  alternatives %d  rows %d" % (
        document["backgrounds"], len(alternatives), len(rows)))
    print("rows sha256 %s" % digest)
    print(f"wrote {existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
