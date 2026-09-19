#!/usr/bin/env python
"""Audit the pinned Absci release: KD availability, support compatibility, overlap.

This is a **diagnostic**. The Absci designs are variable-length HCDR3s on a
different antibody context; they are not a larger interchangeable version of this
benchmark's cohort, and nothing here crops one into a ten-mer core to pretend
otherwise. The audit reports, from the bytes:

* every KD cell's class -- finite, blank, censored, ``N.B.``, ``I.C.``, ``N/A``,
  zero-width-space contaminated or unsupported -- with counts, converting none of
  them into a number;
* the explicit ``Binder`` column counted true/false/unknown in every reported
  subset. A blank KD is a missing number and is never read as a negative;
* HCDR3 lengths, uniqueness and duplication per file, and -- where HCDR1/HCDR2
  exist -- duplication of the full CDR context, which is a different number;
* CDR-context compatibility with the fixed scaffold, and whether it is verified
  (HCDR1 *and* HCDR2 present and equal to trastuzumab's) or an upper bound. These
  files carry no VH/VL sequence, so this is never proof of the whole scaffold;
* overlap **between** the files by HCDR3, because they describe overlapping
  measurements and their row counts are not additive;
* overlap with the published Buzz splits by core, for the compatible rows only.

The Buzz test split is read for **sequences only** (``data.test_sequences``);
no test label is materialized anywhere in this script.

Any reference to or publication of these data must be attributed to
**Absci Corporation (2023)**, per the fourth clause the supplied Clear BSD text
adds to the SPDX template.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments import her2_absci_audit as absci  # noqa: E402
from smallAntibodyGen.experiments import her2_data as data  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import require, save_json  # noqa: E402


def library_cores(raw_root):
    """The three published split core sets. Test contributes sequences only."""
    train = data.load_split(raw_root, "train")
    val = data.load_split(raw_root, "val")
    return {"train": set(train.seq), "val": set(val.seq),
            "test": set(data.test_sequences(raw_root))}


def run(config_path, output, *, raw_root=None, with_library=True):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw_root = Path(raw_root) if raw_root else ROOT / config["raw_root"]
    manifest = next((relative for relative in config["source_manifests"]
                     if "absci" in relative), None)
    require(manifest is not None,
            f"{config_path} pins no Absci manifest; the audit verifies bytes before reading them")
    cores = library_cores(raw_root) if with_library else None
    document = absci.audit(ROOT, raw_root, manifest_path=manifest, library_cores=cores)
    save_json(output, document)
    print(f"Absci audit written to {output}", flush=True)
    for name, report in sorted(document["files"].items()):
        support = report["support"]
        print(f"  {name}: {report['rows']} rows, "
              f"{report['hcdr3']['unique']} unique HCDR3 "
              f"(CDR-context unique {report['hcdr3']['cdr_context']['unique']}), KD classes "
              f"{report['kd'].get('classes')}, binder {report['binder_flag']}", flush=True)
        print(f"    CDR-context compatible rows {support['compatible_rows']}"
              f"{' (UPPER BOUND: no HCDR1/HCDR2 columns)' if support.get('upper_bound') else ''}"
              f", compatible KD classes {support['compatible_kd_classes']}"
              f", compatible binder flags {support['compatible_binder_flags']}", flush=True)
    print(f"  cross-file: {document['cross_file']['pairwise']}", flush=True)
    print(f"  attribution: {document['attribution']}", flush=True)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiments/her2_guarded_continuation.json")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/claude_codex_her2_guarded_20260918/absci_audit.json")
    parser.add_argument("--skip-library-overlap", action="store_true",
                        help="skip the Buzz split overlap (which loads the published splits)")
    args = parser.parse_args()
    run(args.config, args.output, raw_root=args.raw_root,
        with_library=not args.skip_library_overlap)
