"""Acquire pinned primary records used to qualify the HER2 route sources.

Existing AlphaSeq CSV archives are reused: their decompressed bytes were verified
against the primary release. This command fetches the small provenance files and
Adams replicate tables, without copying the large archives a second time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


RELEASES = (
    ("adams2017", "jbkinney/16_titeseq", "54130c8fc6081bbfa8301c41e1afcfa6a5d79590",
     ("README.md", "data/replicate_1.csv", "data/replicate_2.csv", "data/replicate_3.csv",
      "figure_5_landscape.py", "KD_fit_log_poiss.py")),
    ("petersen2024_parent_export", "matsengrp/dasm-experiments",
     "ce1b71c605f3d584ac24423633e51fda35efeb00",
     ("data/whitehead/petersen/original/AntibodySequences2.csv",
      "data/whitehead/petersen/original/MAGMAseq_anchor_FLAB.csv",
      "data/whitehead/petersen/original/MAGMAseq_anchor_FLABbinary.csv",
      "data/whitehead/petersen/processed/README.md")),
    ("alphaseq", "mit-ll/AlphaSeq_Antibody_Dataset",
     "72218839adc6abfe26aae1fde35d5d7b390a0459",
     ("README.md", "antibody_dataset_1/README.md", "antibody_dataset_1/LICENSE",
      "antibody_dataset_1/SPDX.spdx", "antibody_dataset_2/README.md",
      "antibody_dataset_2/LICENSE", "antibody_dataset_2/SPDX.spdx")),
)


def acquire(root: Path) -> dict:
    records = []
    for name, repository, revision, files in RELEASES:
        for relative in files:
            target = root / "data" / "primary" / name / relative
            url = f"https://raw.githubusercontent.com/{repository}/{revision}/{relative}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                request = urllib.request.Request(url, headers={"User-Agent": "smallAntibodyGen-research"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = response.read()
                temporary = target.with_name(target.name + ".partial")
                temporary.write_bytes(payload)
                temporary.replace(target)
            payload = target.read_bytes()
            record = {"source": name, "repository": repository, "revision": revision,
                      "url": url, "path": str(target), "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()}
            records.append(record)
            print(json.dumps({"source": name, "file": relative, "bytes": len(payload)}), flush=True)
    document = {"schema": "her2-primary-acquisition/1", "files": records,
                "bytes": sum(record["bytes"] for record in records),
                "outcomes": "No HER2 or CR9114 H1 outcomes are acquired by this command."}
    manifest = root / "data" / "primary_review_acquisition_manifest.json"
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    document = acquire(args.artifact_root.resolve())
    print(json.dumps({"files": len(document["files"]), "bytes": document["bytes"]}))
