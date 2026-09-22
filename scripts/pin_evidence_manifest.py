#!/usr/bin/env python
"""Pin a campaign's local evidence by digest so a freeze can authenticate it.

Campaign evidence is local-only research material: it is deliberately not
committed, so a freeze cannot ask "is this tracked, and equal to HEAD?" without
publishing it first. This writes the committed half of that bargain -- a manifest
of SHA-256 digests under ``configs/evidence_manifests/`` -- while the bytes stay
on the machine that produced them.

Re-pinning is a deliberate act. The freeze refuses when the evidence on disk
disagrees with the manifest, which is the whole point: the alternative is a
freeze that certifies whatever happened to be in the directory.

Usage::

    python scripts/pin_evidence_manifest.py --config configs/experiments/her2_parent_replay.json
    python scripts/pin_evidence_manifest.py --config <config> --check   # verify only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_DIR = "configs/evidence_manifests"

#: Both campaign families name their identity differently; neither is renamed
#: here, because the id is what the freeze looks the manifest up by.
ID_KEYS = ("campaign_id", "audit_id")


def campaign_id(config: dict) -> str:
    for key in ID_KEYS:
        if config.get(key):
            return str(config[key])
    raise SystemExit(f"config has none of {ID_KEYS}; cannot name the manifest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_digests(root: Path, evidence_root: str) -> dict[str, str]:
    base = root / evidence_root
    if not base.is_dir():
        raise SystemExit(f"{evidence_root} does not exist; run the prepare stage first")
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"{evidence_root} holds no evidence; run the prepare stage first")
    return {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True,
                        help="campaign config carrying evidence_root and the campaign id")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true",
                        help="verify the existing manifest instead of writing one")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    name = campaign_id(config)
    observed = evidence_digests(args.root, config["evidence_root"])
    target = args.root / MANIFEST_DIR / f"{name}.json"

    if args.check:
        if not target.is_file():
            print(f"absent: {target}", file=sys.stderr)
            return 1
        recorded = (json.loads(target.read_text(encoding="utf-8")).get("sha256") or {})
        if recorded == observed:
            print(f"{target.name}: {len(observed)} files match")
            return 0
        missing = sorted(set(recorded) - set(observed))
        extra = sorted(set(observed) - set(recorded))
        changed = sorted(k for k in set(recorded) & set(observed) if recorded[k] != observed[k])
        print(f"{target.name} does not match the evidence on disk:", file=sys.stderr)
        for label, rows in (("absent here", missing), ("not pinned", extra),
                            ("different bytes", changed)):
            if rows:
                print(f"  {label}: {rows}", file=sys.stderr)
        return 1

    document = {
        "kind": "campaign_evidence_manifest",
        "schema_version": "1",
        "campaign_id": name,
        "evidence_root": config["evidence_root"],
        "note": ("Digests of local-only campaign evidence. The bytes are research material "
                 "and are not committed; these hashes are, so a freeze can authenticate the "
                 "evidence without publishing it. Re-pin deliberately."),
        "file_count": len(observed),
        "sha256": observed,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"pinned {len(observed)} evidence files -> {target.relative_to(args.root)}")
    print("Commit this manifest: an untracked manifest certifies nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
