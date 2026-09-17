#!/usr/bin/env python
"""Replay the first affinity+entropy arm in a separate deterministic process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.diversity import negative_entropy_surrogate  # noqa: E402
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest  # noqa: E402


def run(directory):
    record = json.loads((directory / "results.json").read_text())
    require(record["status"] == "completed", "Training must finish first")
    config, identity = record["config"], record["initial_identity"]
    seed = config["seeds"][0]
    name = f"seed_{seed}_affinity_entropy"
    result = record["arms"][name]
    destination = directory / "deterministic_replay.json"
    require(not destination.exists(), "Replay evidence already exists")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    from smallAntibodyGen.esmif1_compat import install
    from smallAntibodyGen.structure import load_prepared_structure, verify_against_source
    from smallAntibodyGen.structure.policy_adapter import bind_policy
    install()
    import esm
    prior = json.loads((ROOT / "configs/experiments/cr9114_dpo_pilot.json").read_text())
    pilot = json.loads((ROOT / prior["pilot_dir"] / "run.json").read_text())
    prepared_path = ROOT / pilot["config"]["prepared_artifact"]
    require(sha256(prepared_path) == identity["prepared_artifact_sha256"], "Structure input changed")
    prepared = load_prepared_structure(prepared_path)
    require(not verify_against_source(prepared, ROOT / pilot["config"]["structure"]), "Structure source changed")
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    require(sha256(weights) == identity["weights_sha256"], "Parent weights changed")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    require(bound.geometry.digest == identity["geometry_digest"], "Deterministic geometry changed across processes")
    sft_path = ROOT / prior["sft_checkpoint"]
    require(sha256(sft_path) == identity["sft_checkpoint_sha256"], "SFT checkpoint changed")
    checkpoint = torch.load(sft_path, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    del checkpoint
    require(state_digest(model.decoder) == identity["decoder_state_sha256"]
            and state_digest(model.encoder) == identity["encoder_state_sha256"], "Initial model mismatch")
    population_path = directory / "training_population.csv"
    require(sha256(population_path) == record["output_sha256"][population_path.name], "Population changed")
    population = pd.read_csv(population_path, dtype={"genotype": "string"})
    require(set(population.split) == {"train"}, "Held-out labels in training population")
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in population.genotype]
    schedule_path = directory / f"schedule_{result['schedule_key']}.npy"
    require(sha256(schedule_path) == record["schedule_sha256"][result["schedule_key"]], "Schedule changed")
    schedule = np.load(schedule_path, allow_pickle=False)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    for step, indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        logq = bound.policy.log_prob([sequences[i] for i in indices], bound.geometry)
        (-logq.mean() / config["likelihood_normalizer"]).backward()
        if step % config["entropy_every"] == 0:
            sample = bound.policy.sample(bound.geometry, num_samples=config["entropy_samples"], generator=generator)
            rescored = bound.policy.log_prob(sample.sequences, bound.geometry)
            require(float((rescored.detach() - sample.log_probability).abs().max()) < 2e-4, "Entropy sampling parity failed")
            (result["arm"]["entropy_coefficient"] * config["entropy_every"] * negative_entropy_surrogate(rescored)).backward()
            del sample, rescored
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step % 32 == 0:
            print(f"Affinity+entropy replay: {step}/{len(schedule)}", flush=True)
    actual, expected = state_digest(model.decoder), result["decoder_state_sha256"]
    evidence = {"arm": name, "replay_script_sha256": sha256(Path(__file__)), "encoding_digest_matches": True,
                "replayed_decoder_sha256": actual, "original_decoder_sha256": expected,
                "decoder_bitwise_identical": actual == expected,
                "encoder_unchanged": state_digest(model.encoder) == identity["encoder_state_sha256"],
                "reserved_test_labels_evaluated": False}
    save_json(destination, evidence)
    require(evidence["decoder_bitwise_identical"] and evidence["encoder_unchanged"], "Deterministic replay failed; evidence retained")
    print("Separate process reproduced affinity+entropy decoder bit-for-bit", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)
