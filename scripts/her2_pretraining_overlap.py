#!/usr/bin/env python
"""How much of the HER2 evaluation population is in the pretraining corpus?

Every policy in this project descends from p-IgGen, whose model card states it is
"pretrained on unpaired heavy and light chain sequences from the Observed
Antibody Space (OAS)". So OAS is the pretraining corpus, and any HER2 sequence
that appears in OAS was seen -- in some form -- before this project's adaptation
data existed. A likelihood on such a sequence is not a clean out-of-sample
quantity.

Three checks, in increasing severity:

1. **The wild type.** Is trastuzumab's own HCDR3 in the corpus? If yes, the
   scaffold and the wild-type core are memorised, which bears directly on any
   result phrased as a preference for a wild-type residue -- the site-1 G finding
   in particular.
2. **The library.** How many of the labelled library's ten-residue cores occur
   inside an OAS CDR3? These are the rows every ranking number is computed on.
3. **The published distances.** Absci's zero-shot HER2 binders ship a
   ``Minimum HCDR3 Edit Distance to OAS`` column computed against the FULL OAS,
   which is the number this local check cannot produce.

**What a negative here does and does not mean.** The local corpus is a sample of
OAS, not the copy p-IgGen trained on. A hit is therefore decisive and a miss is
weak: absence from this subset is not absence from the pretraining set. The
published Absci column is the corrective, and it is reported alongside.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_data as data_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths  # noqa: E402

CORE = data_lib.CORE_LENGTH


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oas", default="data/processed/oas_unpaired_3m/oas_igh.jsonl.gz")
    parser.add_argument("--raw-root", default="data/raw/her2_functional_20260918")
    parser.add_argument("--output", default="outputs/her2_overlap_20260923")
    args = parser.parse_args(argv)

    raw, out = ROOT / args.raw_root, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    scaffold = data_lib.load_scaffold(raw)
    wild_type_hcdr3 = data_lib.ANCHOR_LEFT + data_lib.WT_CORE + data_lib.ANCHOR_RIGHT
    print("trastuzumab HCDR3 (this benchmark's convention): %s" % wild_type_hcdr3, flush=True)

    library, by_split = {}, {}
    for split in ("train", "val"):
        frame = data_lib.load_split(raw, split)
        by_split[split] = set(frame.seq.tolist())
        for core, klass in zip(frame.seq, frame["class"]):
            library.setdefault(core, klass)
    cores = set(library)
    print("labelled library cores: %d (train %d, val %d)" % (
        len(cores), len(by_split["train"]), len(by_split["val"])), flush=True)

    # -- stream the corpus ---------------------------------------------------
    records = 0
    cdr3_exact = collections.Counter()
    core_hits = collections.Counter()
    anchored_hits = collections.Counter()
    wild_type_exact = 0
    wild_type_core_containing = 0
    with gzip.open(ROOT / args.oas, "rt") as handle:
        for line in handle:
            records += 1
            cdr3 = json.loads(line).get("cdr3_aa") or ""
            if not cdr3:
                continue
            if cdr3 == wild_type_hcdr3:
                wild_type_exact += 1
            if data_lib.WT_CORE in cdr3:
                wild_type_core_containing += 1
            cdr3_exact[cdr3] += 1
            # every ten-residue window of this CDR3 that is a library core
            for start in range(len(cdr3) - CORE + 1):
                window = cdr3[start:start + CORE]
                if window in cores:
                    core_hits[window] += 1
                    if (start >= len(data_lib.ANCHOR_LEFT)
                            and cdr3[start - len(data_lib.ANCHOR_LEFT):start] == data_lib.ANCHOR_LEFT
                            and cdr3[start + CORE:start + CORE + 1] == data_lib.ANCHOR_RIGHT):
                        anchored_hits[window] += 1
            if records % 500_000 == 0:
                print("  %d records, %d distinct library cores hit so far"
                      % (records, len(core_hits)), flush=True)
    print("corpus records %d, distinct CDR3 %d" % (records, len(cdr3_exact)), flush=True)

    hit_classes = collections.Counter(library[c] for c in core_hits)
    anchored_classes = collections.Counter(library[c] for c in anchored_hits)
    in_val = sum(1 for c in core_hits if c in by_split["val"])

    # -- the published full-OAS distances ------------------------------------
    absci = ROOT / args.raw_root / "absci" / "zero-shot-binders.csv"
    published = None
    if absci.is_file():
        rows = list(csv.DictReader(absci.open(encoding="utf-8")))
        column = "Minimum HCDR3 Edit Distance to OAS"
        values = [int(r[column]) for r in rows if r.get(column, "").strip().lstrip("-").isdigit()]
        published = {
            "source": "data/raw/her2_functional_20260918/absci/zero-shot-binders.csv",
            "population": "422 Absci zero-shot designed HER2 binders with measured KD",
            "column": column,
            "rows": len(values),
            "distance_zero": sum(1 for v in values if v == 0),
            "distance_at_most_one": sum(1 for v in values if v <= 1),
            "distance_at_most_two": sum(1 for v in values if v <= 2),
            "distribution": dict(sorted(collections.Counter(values).items())),
            "why_it_matters": ("this is computed against the FULL OAS, which the local subset "
                               "cannot reproduce. It is the honest upper view of pretraining "
                               "exposure for a measured HER2 binder population."),
        }

    document = {
        "record_kind": "pretraining_overlap",
        "pretraining_corpus": ("OAS. p-IgGen's model card: 'Pretrained on unpaired heavy and "
                               "light chain sequences from the Observed Antibody Space (OAS), "
                               "and finetuned on paired sequences.'"),
        "local_corpus": {"path": args.oas, "records": records,
                         "distinct_cdr3": len(cdr3_exact),
                         "caveat": ("a SAMPLE of OAS, not the copy p-IgGen trained on. A hit is "
                                    "decisive; a miss is weak evidence of absence.")},
        "wild_type": {
            "hcdr3": wild_type_hcdr3,
            "exact_cdr3_matches": wild_type_exact,
            "cdr3_containing_the_core": wild_type_core_containing,
            "core": data_lib.WT_CORE,
        },
        "library": {
            "cores": len(cores),
            "cores_found_in_a_cdr3": len(core_hits),
            "fraction": len(core_hits) / len(cores),
            "occurrences": int(sum(core_hits.values())),
            "by_class": dict(hit_classes),
            "found_cores_in_validation_split": in_val,
            "anchored_matches": {
                "rule": f"the core appears with {data_lib.ANCHOR_LEFT!r} before and "
                        f"{data_lib.ANCHOR_RIGHT!r} after, i.e. the same HCDR3 as this benchmark",
                "cores": len(anchored_hits),
                "occurrences": int(sum(anchored_hits.values())),
                "by_class": dict(anchored_classes),
            },
            "examples": [{"core": c, "class": library[c], "occurrences": n}
                         for c, n in core_hits.most_common(15)],
        },
        "published_full_oas_distances": published,
        "recorded_at": paths.utc_now(),
    }
    paths.write_json(out / "pretraining_overlap.json", document)

    print("\nwild-type HCDR3 %s: %d exact CDR3 matches, %d CDR3 containing the core" % (
        wild_type_hcdr3, wild_type_exact, wild_type_core_containing))
    print("library cores found inside an OAS CDR3: %d of %d (%.4f%%), %d occurrences" % (
        len(core_hits), len(cores), 100.0 * len(core_hits) / len(cores), sum(core_hits.values())))
    print("  by class: %s" % dict(hit_classes))
    print("  with this benchmark's anchors: %d cores, %d occurrences" % (
        len(anchored_hits), sum(anchored_hits.values())))
    if published:
        print("\npublished full-OAS distance, 422 measured Absci binders:")
        print("  exactly in OAS: %d   within 1: %d   within 2: %d" % (
            published["distance_zero"], published["distance_at_most_one"],
            published["distance_at_most_two"]))
    print(f"\nwrote {out / 'pretraining_overlap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
