#!/usr/bin/env python
"""Independent deterministic replay of the first KL+embedding training arm."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments.regularization import decoder_statistics
from run_cr9114_esmif1_pilot import require, save_json, sha256, state_digest
from run_cr9114_shortlist import load_non_test, load_sft


def replay(directory):
    result = json.loads((directory / "results.json").read_text())
    require(result["status"] == "completed", "Complete primary comparison first")
    config = result["config"]
    seed = config["seeds"][0]
    name = f"seed_{seed}_kl_embedding"
    expected = result["arms"][name]
    prior, pilot, _, _ = load_non_test()
    model, bound, _, identity = load_sft(prior, pilot)
    require(identity == result["initial_identity"], "Initial identity changed")
    reference = copy.deepcopy(model.decoder).eval().requires_grad_(False)
    population_path = directory / "training_population.csv"
    require(sha256(population_path) == result["output_sha256"][population_path.name], "Population changed")
    population = pd.read_csv(population_path, dtype={"genotype": "string"})
    require(set(population.split) == {"train"}, "Training crossed splits")
    sequences = [bound.space.sequence_for(tuple(map(int, g))) for g in population.genotype]
    schedule_path = directory / f"schedule_{seed}.npy"
    require(sha256(schedule_path) == expected["schedule_sha256"], "Schedule changed")
    schedule = np.load(schedule_path, allow_pickle=False)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    started = time.perf_counter()
    max_parity = 0.
    for step, indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        (-bound.policy.log_prob([sequences[i] for i in indices], bound.geometry).mean() / 16).backward()
        if step % 4 == 0:
            samples = bound.policy.sample(bound.geometry, num_samples=8, generator=generator)
            with torch.no_grad():
                refs, _ = decoder_statistics(reference, bound, samples.sequences)
            logq, z = decoder_statistics(model.decoder, bound, samples.sequences)
            max_parity = max(max_parity, float((logq.detach() - samples.log_probability).abs().max()))
            # Spell out the estimator rather than calling the training loss helper.
            ratio = logq.detach() - refs
            baseline = (ratio.sum() - ratio) / 7
            kl_carrier = (logq * (ratio - baseline)).mean()
            cosine = (z.sum(0).square().sum() - z.square().sum()) / 56
            (4 * (.1 * kl_carrier / 16 + .1 * cosine)).backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 32 == 0:
            print(f"Replay {step}/256 ({time.perf_counter() - started:.1f}s)", flush=True)
    actual = state_digest(model.decoder)
    require(actual == expected["decoder_state_sha256"], "Deterministic final decoder did not reproduce")
    require(state_digest(model.encoder) == identity["encoder_state_sha256"] and state_digest(reference) == identity["decoder_state_sha256"], "Frozen state changed")
    document = {"passes": True, "arm": name, "bitwise_final_decoder_match": True,
                "decoder_state_sha256": actual, "sample_rescore_max_error": max_parity,
                "training_seconds": time.perf_counter() - started, "reference_and_encoder_unchanged": True,
                "assay_labels_used_for_training": "training split only", "development_evaluated": False}
    save_json(directory / "deterministic_replay.json", document)
    print(json.dumps(document))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/cr9114_regularization_20260917")
    replay(parser.parse_args().directory)
