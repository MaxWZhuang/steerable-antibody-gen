#!/usr/bin/env python
"""Evaluation-only numerical amendment for the completed 2026-09-18 HER2 fits.

Use native PyTorch math SDPA uniformly for validation, sampling and final scoring.
The original FP32 precision and score tolerances are unchanged. This driver never
trains: it imports the original, hash-verified fits into a fresh validation folder,
binds this extra driver and backend in a separate manifest, and attaches that
manifest to the final selection freeze before any reserved outcomes are opened.

Run --stage validate, then --stage evaluate. The original automatic-SDPA records
are retained separately and are never reused by this driver.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import evaluate_her2 as final_evaluation
import posttrain_her2 as posttrain
from smallAntibodyGen.experiments.her2_runtime import (
    code_digests, load_json, relative_key, require, save_json, sha256,
)

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SCHEMA = "her2-numerical-amendment/1"


def expected_manifest(config_path, run):
    config = load_json(config_path)
    require(config["policy"]["precision"] == "float32", "This amendment retains FP32")
    return {
        "schema_version": BACKEND_SCHEMA,
        "attention_backend": "torch.nn.attention.SDPBackend.MATH",
        "precision": "float32",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "stages": ["validation", "sampling", "reference_kl", "final_evaluation"],
        "training_updates": 0,
        "driver": {"path": relative_key(Path(__file__), ROOT), "sha256": sha256(__file__)},
        "original_scientific_code_digests": code_digests(ROOT),
        "config_sha256": sha256(config_path),
        "base_selection_sha256": sha256(run / "base_selection.json"),
        "original_continuation_results_sha256": sha256(
            run / "continuation/continuation_results.json"),
        "original_partial_validation_sha256": sha256(
            run / "continuation/validation_records.json"),
        "tolerances": config["tolerances"],
        "reason": (
            "One automatic-SDPA FP32 sampler/scorer check failed on a trained DPO "
            "checkpoint. FP64 teacher forcing and autoregression agreed to 8.53e-14 "
            "nats on the worst rows. Native math SDPA passed the original FP32 "
            "tolerance and reproduced all 10,000 original draws on that checkpoint. "
            "All policies are revalidated uniformly; no original records are reused."
        ),
    }


def manifest_reference(path):
    return {"path": relative_key(path, ROOT), "sha256": sha256(path)}


def check_manifest(path, expected):
    require(path.is_file(), "Missing numerical-backend manifest; validate first")
    require(load_json(path) == expected,
            "Numerical evaluation identity changed; do not reuse this run directory")


def bind_freeze(freeze_path, manifest_path):
    freeze = load_json(freeze_path)
    reference = manifest_reference(manifest_path)
    require(freeze.get("stage") == "final", "Only a final selection can be bound")
    previous = freeze.get("numerical_evaluation")
    require(previous is None or previous == reference,
            "Frozen numerical evaluation points to a different manifest")
    if previous is None:
        freeze["numerical_evaluation"] = reference
        save_json(freeze_path, freeze)


def run_stage(stage, config_path, run):
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                    text=True).strip()
    require(not dirty, "Commit the numerical amendment before running it")
    config_path, run = Path(config_path).resolve(), Path(run).resolve()
    validation = run / "validation_math"
    evaluation = run / "evaluation_math"
    manifest_path = validation / "numerical_backend.json"
    freeze_path = validation / "selection_frozen.json"
    expected = expected_manifest(config_path, run)

    if stage == "validate":
        require(not (evaluation / "results.json").exists(),
                "Final outcome evaluation has started; validation is now fixed")
        if manifest_path.exists():
            check_manifest(manifest_path, expected)
        else:
            require(not validation.exists(),
                    "Refusing a validation directory without numerical provenance")
            validation.mkdir(parents=True)
            save_json(manifest_path, expected)
        imported = validation / "continuation_results.json"
        if not imported.exists():
            shutil.copyfile(run / "continuation/continuation_results.json", imported)
    else:
        check_manifest(manifest_path, expected)
        require(load_json(freeze_path).get("numerical_evaluation")
                == manifest_reference(manifest_path),
                "Selection freeze does not bind this numerical-backend manifest")

    with sdpa_kernel(SDPBackend.MATH):
        require(torch.backends.cuda.math_sdp_enabled()
                and not torch.backends.cuda.flash_sdp_enabled()
                and not torch.backends.cuda.mem_efficient_sdp_enabled()
                and not torch.backends.cuda.cudnn_sdp_enabled(),
                "The evaluation must use math SDPA exclusively")
        if stage == "validate":
            posttrain.run(config_path, validation, stages=("validate", "freeze"),
                          allow_dirty=False, allow_cpu=False, discard_incomplete=False,
                          base_selection=run / "base_selection.json")
            bind_freeze(freeze_path, manifest_path)
        else:
            final_evaluation.run(config_path, evaluation, allow_dirty=False, allow_cpu=False,
                                 training_output=run, freeze_path=freeze_path)
            result_path = evaluation / "results.json"
            result = load_json(result_path)
            require(result["status"] == "completed", "Final evaluation did not complete")
            require(result["selection_freeze_sha256"] == sha256(freeze_path),
                    "Evaluation consumed a different selection freeze")
            result["numerical_evaluation"] = manifest_reference(manifest_path)
            save_json(result_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("validate", "evaluate"))
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiments/her2_posttrain.json")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/her2_posttrain_20260918")
    args = parser.parse_args()
    run_stage(args.stage, args.config, args.run)
