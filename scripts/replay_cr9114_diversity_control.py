#!/usr/bin/env python
"""Independently replay a completed deterministic plain-DPO control."""
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
from smallAntibodyGen.experiments.dpo import dpo_per_pair_loss, load_reference_cache  # noqa: E402
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest  # noqa: E402
from run_cr9114_dpo_pilot import load_inputs  # noqa: E402


def run(directory):
    record = json.loads((directory / "results.json").read_text())
    config, identity = record["config"], record["reference_identity"]
    seed = config["seeds"][0]
    name = f"seed_{seed}_entropy_0"
    require(name in record["arms"], "No completed first-seed control")
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
    pilot, _, pairs, _, _, _ = load_inputs(prior)
    base = pilot["config"]
    prepared_path = ROOT / base["prepared_artifact"]
    require(sha256(prepared_path) == identity["prepared_artifact_sha256"], "Structure input changed")
    prepared = load_prepared_structure(prepared_path)
    require(not verify_against_source(prepared, ROOT / base["structure"]), "Structure source changed")
    weights = Path(torch.hub.get_dir()) / "checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
    require(sha256(weights) == base["weights_sha256"], "Parent weights changed")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    model.eval().cuda()
    bound = bind_policy(prepared, model, alphabet)
    require(bound.geometry.digest == identity["geometry_digest"], "Deterministic encoding changed across processes")
    sft_path = ROOT / prior["sft_checkpoint"]
    require(sha256(sft_path) == identity["source_checkpoint_sha256"], "SFT checkpoint changed")
    checkpoint = torch.load(sft_path, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    del checkpoint
    require(state_digest(model.decoder) == identity["decoder_state_sha256"]
            and state_digest(model.encoder) == identity["encoder_state_sha256"], "Initial model mismatch")
    cache_path = directory / "reference_cache.json"
    require(sha256(cache_path) == record["reference_cache_sha256"], "Reference cache changed")
    cache = json.loads(cache_path.read_text())
    references = pd.Series(load_reference_cache(cache_path, identity, cache["genotypes"]), index=cache["genotypes"])
    schedule_path = directory / f"schedule_{seed}.npy"
    require(sha256(schedule_path) == record["schedule_sha256"][str(seed)], "Schedule changed")
    schedule = np.load(schedule_path, allow_pickle=False)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    for step, indices in enumerate(schedule, 1):
        batch = pairs.iloc[indices]
        gs = batch.chosen_genotype.tolist() + batch.rejected_genotype.tolist()
        optimizer.zero_grad(set_to_none=True)
        logq = bound.policy.log_prob([bound.space.sequence_for(tuple(map(int, g))) for g in gs], bound.geometry)
        ref = torch.tensor(references.loc[gs].to_numpy(), dtype=logq.dtype, device=logq.device)
        size = len(indices)
        loss = dpo_per_pair_loss(logq[:size], logq[size:], ref[:size], ref[size:], beta=config["beta"]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step % 64 == 0:
            print(f"Deterministic control replay: {step}/{len(schedule)}", flush=True)
    actual, expected = state_digest(model.decoder), record["arms"][name]["decoder_state_sha256"]
    result = {"arm": name, "replay_script_sha256": sha256(Path(__file__)),
              "encoding_digest_matches": True, "replayed_decoder_sha256": actual,
              "original_decoder_sha256": expected, "decoder_bitwise_identical": actual == expected,
              "encoder_unchanged": state_digest(model.encoder) == identity["encoder_state_sha256"],
              "reserved_test_labels_evaluated": False}
    save_json(destination, result)
    require(result["decoder_bitwise_identical"] and result["encoder_unchanged"], "Deterministic replay failed; evidence retained")
    print("Independent process reproduces the final decoder bit-for-bit", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)
