"""Freeze graded contrasts, audit checkpoint ancestry, and score masked HCDR3s.

Run subcommands in order: build, audit, score. This CLI never trains a model,
changes a training gate, or rewrites the source corpus/splits.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.evaluation.contrasts import (
    build_manifest, file_sha256, iter_jsonl, load_json, save_json, verify_manifest,
)


def implementation_hashes(*relative_paths):
    return {p: file_sha256(ROOT / p) for p in relative_paths}


def write_result(args, result, large_field):
    save_json(args.output, result)
    summary = {k: v for k, v in result.items() if k != large_field}
    if args.summary_output:
        save_json(args.summary_output, summary)
    print(json.dumps({k: summary[k] for k in ("summary", "per_target", "broad_census_without_ha")
                      if k in summary}, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


def build(args):
    # Import the producer, including its normalization and global alias graph.
    from prepare_antibody_antigen import TargetIdentityIndex, extract_target_nodes
    data = Path(args.data_path).resolve()
    source_hash = file_sha256(data)
    protocol = load_json(args.protocol)
    index = TargetIdentityIndex()
    count = 0
    for count, row in enumerate(iter_jsonl(data), 1):
        index.observe(extract_target_nodes(row, row.get("sequence_antigen") or ""))
        if count % 250000 == 0:
            print(f"identity: {count:,} rows", flush=True)
    index.finalize()
    provenance = {
        "corpus": str(data), "corpus_sha256": source_hash, "corpus_rows": count,
        "protocol_sha256": file_sha256(args.protocol),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": implementation_hashes(
            "scripts/prepare_antibody_antigen.py", "src/smallAntibodyGen/target_identity.py",
            "src/smallAntibodyGen/evaluation/contrasts.py", "scripts/hcdr3_contrast_benchmark.py"),
    }
    manifest = build_manifest(
        iter_jsonl(data), protocol=protocol, provenance=provenance,
        canonicalize=lambda row: index.canonical_id(
            extract_target_nodes(row, row.get("sequence_antigen") or "")),
    )
    if file_sha256(data) != source_hash:
        raise RuntimeError("corpus changed while building the manifest")
    write_result(args, manifest, "groups")


def audit(args):
    from smallAntibodyGen.evaluation.contrast_exposure import audit_exposure, checkpoint_chain
    manifest = load_json(args.manifest)
    verify_manifest(manifest)
    override_config = load_json(args.corpus_overrides) if args.corpus_overrides else {"overrides": {}}
    stages = checkpoint_chain(Path(args.checkpoint), ROOT, override_config["overrides"])
    result = audit_exposure(manifest, stages, progress=lambda s: print(s, flush=True))
    result["corpus_overrides"] = override_config
    result["implementation_sha256"] = implementation_hashes(
        "src/smallAntibodyGen/evaluation/contrast_exposure.py", "src/smallAntibodyGen/evaluation/contrasts.py",
        "src/smallAntibodyGen/ancestor_quarantine.py", "src/smallAntibodyGen/data/affinity.py",
        "scripts/mlm_train.py", "scripts/hcdr3_contrast_benchmark.py")
    write_result(args, result, "variants")


def score(args):
    # Required before CUDA initializes. No sampling or stochastic masks are used.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from hcdr3_infill import build_infiller, build_tokenizer, choose_device, load_dual_stream_model
    from smallAntibodyGen.evaluation.contrast_scoring import score_manifest
    manifest, exposure = load_json(args.manifest), load_json(args.exposure)
    verify_manifest(manifest)
    checkpoint_hash = file_sha256(args.checkpoint)
    if (exposure.get("schema") != "hcdr3-contrast-exposure/1"
            or exposure["manifest_sha256"] != manifest["manifest_sha256"]
            or exposure["stages"][-1]["checkpoint_sha256"] != checkpoint_hash):
        raise ValueError("exposure audit does not match this manifest and checkpoint")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = choose_device(args.device)
    model, cfg = load_dual_stream_model(Path(args.checkpoint), data_path=manifest["provenance"]["corpus"],
                                       device=device)
    infiller = build_infiller(model, build_tokenizer(), cfg, device)
    result = score_manifest(manifest, infiller, fold=args.fold, batch_size=args.batch_size,
                            progress=lambda s: print(s, flush=True), scoring_mode=args.scoring_mode)
    result["provenance"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checkpoint_hash,
        "reconstructed_train_config": dataclasses.asdict(cfg),
        "exposure_audit": str(Path(args.exposure).resolve()), "exposure_sha256": file_sha256(args.exposure),
        "torch_version": torch.__version__, "device": str(device),
        "deterministic_algorithms": True,
        "implementation_sha256": implementation_hashes(
            "scripts/hcdr3_infill.py", "scripts/mlm_train.py", "src/smallAntibodyGen/models/mlm.py",
            "src/smallAntibodyGen/infill/hcdr3.py", "src/smallAntibodyGen/antigen_tokenization.py",
            "src/smallAntibodyGen/tokenizer.py", "src/smallAntibodyGen/evaluation/contrasts.py",
            "src/smallAntibodyGen/evaluation/contrast_scoring.py", "scripts/hcdr3_contrast_benchmark.py"),
    }
    if file_sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed while scoring")
    write_result(args, result, "groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build", help="freeze the contrast predicate and prospective background folds")
    p.add_argument("--data-path", required=True)
    p.add_argument("--protocol", default=str(ROOT / "configs/evaluation/hcdr3_contrasts.json"))
    p.set_defaults(run=build)
    p = commands.add_parser("audit", help="separate current assignment, eligibility, and ancestor contact")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus-overrides")
    p.set_defaults(run=audit)
    p = commands.add_parser("score", help="fully masked ranking and unmeasured antigen substitutions")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--exposure", required=True)
    p.add_argument("--fold", choices=("train", "validation", "test", "all"), default="validation")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--scoring-mode", choices=("full_span", "pll"), default="full_span")
    p.add_argument("--device", default=None)
    p.set_defaults(run=score)
    for p in commands.choices.values():
        p.add_argument("--output", required=True, help="new JSON or JSON.gz artifact; different existing bytes refused")
        p.add_argument("--summary-output", help="optional JSON without per-group/per-variant details")
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
