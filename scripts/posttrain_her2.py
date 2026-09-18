#!/usr/bin/env python
"""Continue each selected SFT checkpoint under matched GPU budgets: DPO vs more SFT.

Three stages, in this order and no other:

``continue``
    For every seed, clone the initial-SFT checkpoint that ``train_her2.py`` chose
    by validation-positive NLL, and run **two** continuations from that same
    parent: more SFT on the eligible high-bin positives, and DPO on distance-
    matched high-vs-low pairs whose reference is the frozen parent. Each is one
    continuous trajectory that drops a checkpoint as it passes each measured GPU
    budget -- 180/360/600 s for both, plus 1200/1800 s for DPO.

``validate``
    Score and sample every raw checkpoint, one at a time, on the 4 GB card: full
    validation ranking metrics, held-out preference-pair metrics, positive NLL,
    and 10,000 temperature-1 draws with the diversity diagnostics. Draws, the
    validation score vector and their hashes are persisted, so the final
    evaluation re-reads them instead of re-randomizing. Each record carries an
    identity block and is reused only when that identity and those bytes still
    match.

``freeze``
    Verify every validation record against the current identity, apply the
    diversity gates fixed in advance, select the best eligible checkpoint *within*
    each budget by validation average precision, and write
    ``selection_frozen.json`` (stage ``final``) naming every raw artifact,
    including the ones that failed. This is the only file that unlocks reserved
    labels, and it is written here, before any of them are read.

Only test sequences are read for overlap counts; test labels and assay outcomes
are not read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments import her2_data as data  # noqa: E402
from smallAntibodyGen.experiments import her2_eval as evaluation  # noqa: E402
from smallAntibodyGen.experiments import her2_policy as policy_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_preferences as preferences  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import (  # noqa: E402
    GpuBudgetClock, Progress, RunLedger, code_digests, digest_document, load_json, relative_key,
    require, save_json, set_cpu_threads, sha256, torch_runtime)

STAGES = ("continue", "validate", "freeze")
METHODS = ("continued_sft", "dpo")
BASE_SELECTION = "base_selection.json"
#: Bumped when a validation record's meaning changes, so an old record is
#: recomputed rather than reused under the new interpretation.
VALIDATION_RECORD_SCHEMA = "her2-validation-record/3"


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_key(method, seed):
    return f"{method}_seed{seed}"


#: One naming implementation, shared with the evaluation's expectations.
checkpoint_key = evaluation.checkpoint_name


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

def build_context(config, raw_root):
    """Splits, eligible preference populations, pairing and the fixed validation pairs."""
    settings = config["continuation"]
    train = data.load_split(raw_root, "train")
    val = data.load_split(raw_root, "val")
    train_population = preferences.build_population(train, "train")
    val_population = preferences.build_population(val, "val")
    expected = settings["expected_counts"]
    require(int(train_population.chosen_index.shape[0]) == expected["train_chosen"],
            f"Eligible training chosen rows {train_population.chosen_index.shape[0]} != "
            f"{expected['train_chosen']}")
    require(int(train_population.rejected_index.shape[0]) == expected["train_rejected"],
            "Eligible training rejected rows changed")
    require(int(val_population.chosen_index.shape[0]) == expected["val_chosen"],
            "Eligible validation chosen rows changed")
    require(sum(train_population.excluded_chosen_distances.values())
            == expected["excluded_train_chosen"], "Excluded training high rows changed")
    require(sum(val_population.excluded_chosen_distances.values())
            == expected["excluded_val_chosen"], "Excluded validation high rows changed")

    reference_index = np.concatenate([train_population.chosen_index,
                                      train_population.rejected_index], axis=0)
    require(reference_index.shape[0] == expected["reference_rows"],
            f"Reference population is {reference_index.shape[0]} rows, expected "
            f"{expected['reference_rows']}")
    chosen_rows = np.arange(train_population.chosen_index.shape[0])
    rejected_rows = chosen_rows.size + np.arange(train_population.rejected_index.shape[0])

    val_pairing = preferences.PreferencePairing(val_population,
                                                seed=settings["pairing"]["validation_seed"])
    val_pairs = val_pairing.fixed_validation_pairs(settings["pairing"]["validation_pairs"])
    val_index = data.encode_cores(val.seq)
    train_index = data.encode_cores(train.seq)
    train_labels = (train["class"] == data.POSITIVE_CLASS).to_numpy().astype(np.float64)
    val_positive = (val["class"] == data.POSITIVE_CLASS).to_numpy()
    split_of, class_of = data.labelled_lookup({"train": train, "val": val})
    for core in data.test_sequences(raw_root):
        split_of.setdefault(core, "test")
    return {
        "train": train, "val": val,
        "train_index": train_index, "train_labels": train_labels,
        "val_index": val_index, "val_positive": val_positive,
        "val_positive_index": val_index[val_positive],
        "val_cores": val.seq.to_numpy(dtype=str),
        "train_population": train_population, "val_population": val_population,
        "reference_index": reference_index,
        "reference_chosen_rows": chosen_rows, "reference_rejected_rows": rejected_rows,
        "val_pairs": val_pairs,
        "catalogs": {"train": set(train.seq), "val": set(val.seq),
                     "test": set(data.test_sequences(raw_root))},
        "split_of": split_of, "class_of": class_of,
        "training_diversity_reference": evaluation.diversity_reference(
            train_index[train_labels.astype(bool)]),
        "scaffold": data.load_scaffold(raw_root),
        "vocab": policy_lib.load_vocab(raw_root),
    }


def base_selection_names(config):
    """Exactly what the initial-SFT freeze must name, built from the config."""
    return sorted(set(evaluation.initial_policy_names(config))
                  | {f"cnn_3class_seed{seed}" for seed in config["classifier"]["seeds"]}
                  | {"piggen_zeroshot", "linear_3class"})


def read_base_selection(path, config, identity_base):
    """The initial-SFT freeze, verified against the CURRENT identity before use.

    Everything checkable is checked here, before this stage writes anything: the
    schema and stage marker, the config hash, the scientific code hashes, the
    pinned source hashes, the exact expected artifact names, and every named
    checkpoint's content hash. A protocol, a trainer or a data file edited between
    the two stages is a different experiment, not a resumable one -- and the
    previous version checked only the config, so a continuation could clone parents
    produced by code that no longer exists and label the result with today's.

    The unlock this returns is deliberately discarded: it is a base-stage token and
    :func:`her2_data._require_unlock` refuses it at every reserved loader.
    """
    document, _ = data.read_selection_freeze(
        path, root=ROOT, expected_config_sha256=identity_base["config_sha256"],
        expected_stage=data.SELECTION_STAGE_BASE,
        expected_selected=base_selection_names(config),
        expected_code_digests=identity_base["code_digests"])
    stored_sources = document.get("source_digests") or {}
    differing = sorted(key for key in set(stored_sources) | set(identity_base["source_digests"])
                       if stored_sources.get(key) != identity_base["source_digests"].get(key))
    require(not differing,
            f"{path} was produced against different pinned sources; the continuation would be "
            f"reading data the parents were not fitted on. Differing entries: {differing[:5]}")
    parents = {}
    for seed in config["continuation"]["seeds"]:
        key = f"policy_sft_seed{seed}"
        require(key in document["selected"],
                f"{path} does not name an initial SFT checkpoint for seed {seed}")
        parents[seed] = document["selected"][key]
    return document, parents


def load_parent_policy(entry, raw_root, context, device):
    model = policy_lib.architecture_model(raw_root, device=device)
    policy_lib.load_checkpoint(ROOT / entry["checkpoint"], model, device=device)
    return policy_lib.CorePolicy.from_prefix(model, context["scaffold"].prefix, context["vocab"],
                                             device=device)


# ---------------------------------------------------------------------------
# continuation
# ---------------------------------------------------------------------------

def reference_for_seed(policy, context, config, directory, identity, clock):
    """Build or reuse the frozen reference cache for one seed's DPO trajectory."""
    settings = config["continuation"]["reference"]
    identity_block = preferences.reference_identity(
        checkpoint_sha256=identity["parent_sha256"], config_sha256=identity["config_sha256"],
        scaffold_prefix=context["scaffold"].prefix, index=context["reference_index"])
    path = Path(directory) / "reference_cache.npy"
    if path.is_file():
        # The reuse cost is timed inside load_reference_cache, around the completed
        # read/digest/identity check. Timing it here, as an argument, subtracted two
        # readings taken before the load and always reported ~0.
        cache = preferences.load_reference_cache(path, identity_block)
        print(f"  reusing reference cache ({cache.values.size} rows) in "
              f"{cache.warm_reuse_wall_seconds:.2f} wall seconds; charging its original "
              f"{cache.creation_gpu_seconds:.1f} GPU seconds to the reported cold-start budget",
              flush=True)
        clock.charge_reused(cache.creation_gpu_seconds, reason="reused frozen reference cache")
    else:
        print(f"  scoring {context['reference_index'].shape[0]} reference rows once", flush=True)
        cache = preferences.build_reference_cache(
            policy, context["reference_index"], identity_block, clock=clock,
            batch_size=settings["score_batch_size"])
        preferences.save_reference_cache(path, cache)
    probe = np.random.default_rng(settings["parity_probe_seed"]).choice(
        cache.values.size, size=min(settings["parity_probe_rows"], cache.values.size),
        replace=False)
    parity = preferences.verify_reference_parity(
        policy, cache, context["reference_index"], probe,
        atol=config["tolerances"]["sum_log_probability_atol"],
        rtol=config["tolerances"]["sum_log_probability_rtol"])
    return cache, parity


def budget_checkpoint(policy, config, context, directory, method, seed, budget, record, identity,
                      reference=None):
    """Save one raw budget checkpoint and measure it. Runs OUTSIDE the GPU budget clock."""
    started = time.perf_counter()
    path = Path(directory) / f"budget_{int(budget)}.pt"
    digest = policy_lib.save_checkpoint(path, policy, {"method": method, "seed": seed,
                                                       "budget_seconds": budget,
                                                       "identity": identity, "progress": record})
    batch = config["inference"]["score_batch_size"]
    positive = policy.score(context["val_positive_index"], batch_size=batch)
    pair_metrics = preferences.validation_pair_metrics(
        policy, context["val_pairs"], batch_size=batch,
        reference_chosen=None if reference is None else reference["chosen"],
        reference_rejected=None if reference is None else reference["rejected"],
        beta=config["continuation"]["dpo"]["beta"])
    return {"checkpoint": relative_key(path, ROOT), "checkpoint_sha256": sha256(path),
            "state_sha256": digest,
            "val_positive_nll_per_residue": float(-positive["mean_log_probability"].mean()),
            "val_pair_metrics": pair_metrics,
            "evaluation_wall_seconds": time.perf_counter() - started,
            "evaluation_note": ("checkpoint I/O and validation are excluded from the training GPU "
                                "budget and reported here instead of subtracted from anything")}


def run_continuation(method, seed, config, context, output, *, device, batch_sequences, parents,
                     identity_base, discard_incomplete, raw_root, clock_factory):
    settings = config["continuation"]
    directory = Path(output) / run_key(method, seed)
    parent = parents[seed]
    identity = dict(identity_base, method=method, seed=seed,
                    parent_checkpoint=parent["checkpoint"], parent_sha256=parent["sha256"],
                    batch_sequences=batch_sequences,
                    budgets=settings["budgets_gpu_seconds"][method],
                    optimization=settings["optimization"])
    ledger = RunLedger(directory, identity, metadata={"first_claimed_git_commit": git_commit()})
    if ledger.start(discard_incomplete=discard_incomplete) == "completed":
        print(f"skipping completed run {directory.name}", flush=True)
        return load_json(ledger.path)["summary"]

    run_started = time.perf_counter()
    policy = load_parent_policy(parent, raw_root, context, device)
    # The parent's own scores on the fixed validation pairs. Taken BEFORE the first
    # update, so these are the frozen reference values, and computed for both arms so
    # the reported pair metrics are the same quantity in each. This is an evaluation
    # cost: it is measured in wall time and is not charged to the training budget.
    reference_started = time.perf_counter()
    val_reference = {
        "chosen": preferences.score_sequences(policy, context["val_pairs"]["chosen_index"],
                                              batch_size=config["inference"]["score_batch_size"],
                                              progress_every=0),
        "rejected": preferences.score_sequences(policy, context["val_pairs"]["rejected_index"],
                                                batch_size=config["inference"]["score_batch_size"],
                                                progress_every=0)}
    validation_reference_wall = time.perf_counter() - reference_started
    policy.model.train()
    clock = clock_factory()
    population = context["train_population"]
    optimization = settings["optimization"]
    optimizer = preferences.build_optimizer(policy.model, optimization)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: policy_lib.continuation_learning_rate_scale(
            step + 1, warmup_steps=optimization["warmup_updates"]))
    summary = {"method": method, "seed": seed, "identity": identity,
               "parent_state_sha256": policy_lib.state_digest(policy.model),
               "batch_sequences": batch_sequences,
               "validation_reference": {
                   "pairs": int(context["val_pairs"]["pairs"]),
                   "digest": context["val_pairs"]["digest"],
                   "wall_seconds": validation_reference_wall,
                   "note": ("parent scores on the fixed validation pairs, taken before the first "
                            "update; an evaluation cost, reported separately and not charged to "
                            "the training GPU budget")}}

    if method == "dpo":
        cache, parity = reference_for_seed(policy, context, config, directory, identity, clock)
        summary["reference_cache"] = dict(cache.document(), fresh_parity=parity)
        batch_pairs = batch_sequences // 2
        pairing = preferences.PreferencePairing(
            population, seed=settings["pairing"]["seed_base"] + seed)
        stream = pairing.stream(batch_pairs)
        summary["pairing"] = {"batch_pairs": batch_pairs,
                              "population": population.document(),
                              "cycle_batches": int(np.ceil(pairing.chosen_count / batch_pairs))}
        # Distinct row ids seen anywhere in the trajectory. A per-batch unique count
        # answers "were there duplicates inside this batch", which is not the
        # question: what a budget consumed is how much of the population it reached.
        seen = {"chosen": set(), "rejected": set()}
        chosen_total = int(population.chosen_index.shape[0])
        rejected_total = int(population.rejected_index.shape[0])

        def step(update):
            cycle, number, chosen_rows, rejected_rows = next(stream)
            chosen_index, rejected_index = pairing.pair_cores(chosen_rows, rejected_rows)
            reference_chosen = cache.tensor(chosen_rows, device=policy.device)
            reference_rejected = cache.tensor(
                context["reference_rejected_rows"][rejected_rows], device=policy.device)
            loss, statistics = preferences.dpo_batch(
                policy, chosen_index, rejected_index, reference_chosen, reference_rejected,
                beta=settings["dpo"]["beta"])
            seen["chosen"].update(chosen_rows.tolist())
            seen["rejected"].update(rejected_rows.tolist())
            statistics.update(cycle=cycle, batch=number, sequences=2 * len(chosen_rows),
                              batch_unique_chosen=int(np.unique(chosen_rows).size),
                              batch_unique_rejected=int(np.unique(rejected_rows).size))
            return loss, statistics

        def cumulative_exposures():
            return {"distinct_chosen_rows": len(seen["chosen"]),
                    "distinct_rejected_rows": len(seen["rejected"]),
                    "chosen_population": chosen_total, "rejected_population": rejected_total,
                    "distinct_chosen_fraction": len(seen["chosen"]) / max(1, chosen_total),
                    "distinct_rejected_fraction": len(seen["rejected"]) / max(1, rejected_total),
                    "unit": "distinct eligible training rows reached so far in this trajectory"}

        exposure_fields = ("pairs", "sequences")
    else:
        order_state = {"cycle": 0, "position": 0, "order": None}
        chosen_total = int(population.chosen_index.shape[0])
        summary["population"] = population.document()
        seen = {"chosen": set()}

        def step(update):
            if order_state["order"] is None or order_state["position"] >= chosen_total:
                if order_state["order"] is not None:
                    order_state["cycle"] += 1
                order_state["order"] = np.random.default_rng(
                    [settings["pairing"]["seed_base"] + seed, order_state["cycle"]]).permutation(
                        chosen_total)
                order_state["position"] = 0
            rows = order_state["order"][order_state["position"]:
                                        order_state["position"] + batch_sequences]
            order_state["position"] += len(rows)
            loss, statistics = preferences.continued_sft_batch(
                policy, population.chosen_index[rows])
            seen["chosen"].update(rows.tolist())
            statistics.update(cycle=order_state["cycle"],
                              batch_unique_chosen=int(np.unique(rows).size))
            return loss, statistics

        def cumulative_exposures():
            return {"distinct_chosen_rows": len(seen["chosen"]),
                    "distinct_rejected_rows": 0, "chosen_population": chosen_total,
                    "rejected_population": 0,
                    "distinct_chosen_fraction": len(seen["chosen"]) / max(1, chosen_total),
                    "unit": "distinct eligible training rows reached so far in this trajectory",
                    "note": ("continued SFT trains on the chosen positives only: no pairs, no "
                             "rejected rows and no reference cache")}

        exposure_fields = ("sequences",)

    # The update count is unknown in advance: the trajectory is stopped by a GPU
    # budget, not by a step target, so the progress log carries no denominator.
    progress = Progress(directory / "progress.json", None, every=50, label=directory.name)
    result = preferences.run_budgeted_trajectory(
        step=step, budgets=settings["budgets_gpu_seconds"][method], clock=clock,
        optimizer=optimizer, scheduler=scheduler, model=policy.model,
        gradient_clip=optimization["gradient_clip"], progress=progress,
        exposure_fields=exposure_fields, cumulative_exposures=cumulative_exposures,
        on_budget=lambda budget, record: budget_checkpoint(
            policy, config, context, directory, method, seed, budget, record, identity,
            reference=val_reference))
    progress.finish()
    summary.update(result)
    summary["clock"] = clock.document()
    summary["excluded_wall_seconds"] = {
        "parent_validation_reference": validation_reference_wall,
        "budget_checkpoints_and_validation": result["excluded_evaluation_wall_seconds"],
        "note": ("wall-clock costs that are real but are NOT training: they are excluded from the "
                 "measured GPU budget and reported here, not subtracted from anything")}
    summary["final_state_sha256"] = policy_lib.state_digest(policy.model)
    summary["whole_run_wall_seconds"] = time.perf_counter() - run_started
    summary["whole_run_wall_note"] = (
        "includes model loading, reference preparation, updates and checkpoint evaluation; "
        "trajectory_wall_seconds starts after reference preparation")
    if torch.cuda.is_available():
        summary["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        torch.cuda.reset_peak_memory_stats()
    save_json(directory / "summary.json", summary)
    ledger.complete(summary)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# validation and generation for every raw checkpoint
# ---------------------------------------------------------------------------

def policy_artifacts(config, base, continuation):
    """Every artifact the final freeze must carry, with its role. Nothing is dropped."""
    artifacts = {}
    for name, entry in base["selected"].items():
        if entry["kind"] == "pinned_zero_shot":
            artifacts[name] = dict(entry, role="initial", budget_seconds=0.0, method="zero_shot",
                                   arm=None)
        elif entry["kind"] == "policy":
            # Both initial arms are SFT, but they are not the same method: one starts
            # from the pinned p-IgGen weights and one from a matched random
            # initialization. Labelling them both "initial_sft" loses the only thing
            # that distinguishes the contrast.
            arm, _, seed = name[len("policy_"):].partition("_seed")
            artifacts[name] = dict(entry, role="initial", budget_seconds=0.0,
                                   method=f"initial_sft_{arm}", arm=arm,
                                   seed=int(seed) if seed.isdigit() else None)
    for key, summary in continuation.items():
        for budget, record in summary["budgets"].items():
            name = checkpoint_key(summary["method"], summary["seed"], float(budget))
            artifacts[name] = {"checkpoint": record["checkpoint"],
                               "sha256": record["checkpoint_sha256"], "kind": "policy",
                               "role": "raw_budget", "method": summary["method"],
                               "seed": summary["seed"],
                               # The parent is fixed by construction, so it is recorded
                               # here rather than attached later by whichever stage
                               # happened to run: `--stages freeze` alone must not
                               # produce artifacts that lost their lineage.
                               "parent_name": f"policy_sft_seed{summary['seed']}",
                               "budget_seconds": float(record["target_gpu_seconds"]),
                               "actual_gpu_seconds": float(record["actual_gpu_seconds"]),
                               "updates": int(record["updates"]),
                               "exposures": dict(record.get("exposures") or {}),
                               "distinct_exposures": dict(record.get("distinct_exposures") or {}),
                               "core_token_exposures": record.get("core_token_exposures"),
                               "overshoot_seconds": float(record["overshoot_seconds"]),
                               "overshoot_within_one_update":
                                   bool(record.get("overshoot_within_one_update", False))}
    return artifacts


def validation_identity(name, entry, config, identity_base, context):
    """What a validation record means, so reuse can check it instead of assuming it.

    Reusing a record keyed on the checkpoint hash alone was the hole: the same
    weights measured under a different config, different metric code, a different
    draw seed or a different scaffold produce different numbers, and the record
    would have been reused as though they were the same measurement.
    """
    generation = config["generation"]
    return {"schema_version": VALIDATION_RECORD_SCHEMA,
            "name": name, "checkpoint_sha256": entry["sha256"],
            "parent_name": entry.get("parent_name"),
            "base_selection_sha256": identity_base["base_selection_sha256"],
            "config_sha256": identity_base["config_sha256"],
            "code_digests_sha256": digest_document(identity_base["code_digests"]),
            "source_digests_sha256": digest_document(identity_base["source_digests"]),
            "scaffold_prefix_sha256": digest_document({"prefix": context["scaffold"].prefix}),
            "validation_pairs_digest": context["val_pairs"]["digest"],
            "draw_seed": generation["seed_base"] + context["generation_offsets"][name],
            "draws": generation["draws"], "temperature": generation["temperature"]}


def validate_one(name, entry, config, context, output, raw_root, device, parent_draws, identity):
    """Load one policy, measure it, sample it, persist everything, then drop it.

    One policy is resident at a time. Thirty-one of these do not fit on a 4 GB
    card together, and staging them all just to score the same rows would buy
    nothing: the populations are identical, so the arrays are reused on the CPU.
    """
    started = time.perf_counter()
    if entry["kind"] == "pinned_zero_shot":
        model = policy_lib.load_pinned_model(raw_root, device=device)
    else:
        model = policy_lib.architecture_model(raw_root, device=device)
        policy_lib.load_checkpoint(ROOT / entry["checkpoint"], model, device=device)
    model.eval()
    policy = policy_lib.CorePolicy.from_prefix(model, context["scaffold"].prefix, context["vocab"],
                                               device=device)
    batch = config["inference"]["score_batch_size"]
    scored = policy.score(context["val_index"], batch_size=batch)
    metrics = evaluation.rank_metrics(scored["mean_log_probability"], context["val_positive"],
                                      context["val_cores"],
                                      k_values=tuple(config["evaluation"]["k_values"]))
    positive = scored["mean_log_probability"][context["val_positive"]]
    # The validation score vector is persisted and re-read, so the reference-relative
    # ranking diagnostics below are a subtraction of two stored arrays rather than a
    # second fit of anything.
    score_path = Path(output) / "validation_scores" / f"{name}.npy"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_path, scored["mean_log_probability"])
    document = {"name": name, "checkpoint": entry["checkpoint"], "sha256": entry["sha256"],
                "role": entry["role"], "method": entry.get("method"),
                "seed": entry.get("seed"), "budget_seconds": entry.get("budget_seconds", 0.0),
                "actual_gpu_seconds": entry.get("actual_gpu_seconds"),
                "budget_basis": ("budget_seconds is the nominal target; actual_gpu_seconds is the "
                                 "measured device time, which overshoots it by at most one update"),
                "parent_name": entry.get("parent_name"),
                "identity": identity,
                "val_metrics": metrics,
                "val_positive_nll_per_residue": float(-positive.mean()),
                "val_scores": {"path": relative_key(score_path, ROOT),
                               "sha256": sha256(score_path),
                               "rows": int(scored["mean_log_probability"].size),
                               "quantity": "mean log probability per residue, validation rows"},
                "val_pair_metrics": preferences.validation_pair_metrics(
                    policy, context["val_pairs"], batch_size=batch)}

    settings = config["generation"]
    seed = identity["draw_seed"]
    index, sampled = policy.sample(settings["draws"], seed=seed,
                                   temperature=settings["temperature"],
                                   batch_size=config["inference"]["sample_batch_size"])
    rescored = policy.score(index, batch_size=batch)
    parity = policy_lib.compare_sum_log_probabilities(
        rescored["sum_log_probability"], sampled, label=f"{name}: sampler vs scorer",
        atol=config["tolerances"]["sum_log_probability_atol"],
        rtol=config["tolerances"]["sum_log_probability_rtol"])
    diagnostics = evaluation.generation_diagnostics(
        index, rescored["sum_log_probability"], catalogs=context["catalogs"],
        train_index=context["train_index"], train_labels=context["train_labels"],
        split_of=context["split_of"], class_of=context["class_of"])
    diagnostics.update(draw_seed=seed, temperature=settings["temperature"],
                       sampler_scorer_parity=parity)
    samples = pd.DataFrame({"draw_index": np.arange(len(index)),
                            "core": data.decode_cores(index),
                            "sum_log_probability": rescored["sum_log_probability"],
                            "mean_log_probability": rescored["mean_log_probability"]})
    sample_path = Path(output) / "generation" / f"draws_{name}.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(sample_path, index=False)
    document["generation"] = diagnostics
    document["generation_samples"] = {"path": relative_key(sample_path, ROOT),
                                      "sha256": sha256(sample_path),
                                      "draws": int(len(index))}
    parent_reference = parent_draws.get(entry.get("parent_name") or name)
    document["diversity"] = evaluation.diversity_eligibility(
        diagnostics, training_reference=context["training_diversity_reference"],
        parent_reference=parent_reference or evaluation.diversity_reference(index))
    document["diversity"]["parent_reference_name"] = entry.get("parent_name") or name
    document["wall_seconds"] = time.perf_counter() - started
    del policy, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return document, evaluation.diversity_reference(index)


def stored_validation_scores(record):
    """Re-read one persisted validation score vector, refusing a changed file."""
    path = ROOT / record["val_scores"]["path"]
    require(path.is_file(), f"Persisted validation scores are missing: {path}")
    digest = sha256(path)
    require(digest == record["val_scores"]["sha256"],
            f"Persisted validation scores for {record['name']} changed on disk: {digest}")
    values = np.load(path)
    require(values.size == record["val_scores"]["rows"], "Validation score vector changed length")
    return values


def reference_relative_validation(records, artifacts, context, config):
    """Validation ranking of each policy MINUS its own parent, and minus zero-shot.

    The DPO implicit reward is ``beta * (log pi - log pi_ref)``; the parent is the
    reference DPO actually trained against, so this is the ranking the objective
    optimized, while minus-zero-shot is the whole post-training path. Both are
    diagnostics: the frozen selection rule is raw-density validation average
    precision and it is untouched by anything computed here.
    """
    k_values = tuple(config["evaluation"]["k_values"])
    cache = {name: stored_validation_scores(record) for name, record in records.items()
             if "val_scores" in record}
    zero_shot = cache.get("piggen_zeroshot")
    for name, record in records.items():
        if name not in cache:
            continue
        parent = artifacts.get(name, {}).get("parent_name")
        if parent and parent in cache and parent != name:
            record["val_metrics_minus_parent"] = evaluation.implicit_reward_metrics(
                cache[name], cache[parent], context["val_positive"], context["val_cores"],
                k_values=k_values, reference_name=parent)
        if zero_shot is not None and name != "piggen_zeroshot":
            record["val_metrics_minus_zero_shot"] = evaluation.implicit_reward_metrics(
                cache[name], zero_shot, context["val_positive"], context["val_cores"],
                k_values=k_values, reference_name="piggen_zeroshot")
        record["selection_note"] = ("selection uses val_metrics.average_precision (raw density) "
                                    "only; the reference-relative tables are diagnostics")
    return records


def score_draws_under(policy, names, records, config, key):
    """Monte Carlo KL from each policy's own draws to one reference, with its SE.

    The expectation in KL(policy || reference) is under the *policy*, so the
    reference scores the policy's draws -- scoring the reference's own draws would
    estimate a different quantity. One reference model is resident at a time and
    it scores every draw file that needs it, which is why this is a second pass
    rather than 31 simultaneous models.
    """
    batch = config["inference"]["score_batch_size"]
    for name, reference_name, reference_sha in names:
        stored = records[name]["generation_samples"]
        path = ROOT / stored["path"]
        digest = sha256(path)
        require(digest == stored["sha256"],
                f"{name}: the persisted draws changed since they were written ({digest}); the KL "
                "would be measured on different sequences than the record describes")
        frame = pd.read_csv(path, dtype={"core": str})
        index = data.encode_cores(frame.core)
        reference = policy.score(index, batch_size=batch)["sum_log_probability"]
        records[name]["generation"][key] = dict(
            evaluation.monte_carlo_kl(frame.sum_log_probability.to_numpy(dtype=np.float64),
                                      reference),
            reference_name=reference_name, reference_sha256=reference_sha,
            draws_sha256=digest)


def reference_kl_pass(records, artifacts, config, context, output, *, raw_root, device):
    """Fill in KL to the zero-shot model and to each policy's own SFT parent.

    An existing KL is reused only when it was measured under *this* reference
    checkpoint and against the draw file that is on disk now. Keying reuse on the
    presence of the key alone would carry a number computed against a different
    reference through the freeze under the same name.
    """
    families = {}
    for name, entry in artifacts.items():
        parent = entry.get("parent_name")
        if parent and parent in artifacts:
            families.setdefault(parent, []).append(name)
    # The zero-shot model's KL to itself is exactly 0 by definition, and it is
    # written with the reference hash so a rerun skips it instead of scoring the
    # reference against its own draws to rediscover the zero.
    records["piggen_zeroshot"]["generation"]["kl_to_zero_shot"] = {
        "kl_nats": 0.0, "reference_name": "piggen_zeroshot",
        "reference_sha256": artifacts["piggen_zeroshot"]["sha256"],
        "draws_sha256": records["piggen_zeroshot"]["generation_samples"]["sha256"],
        "note": "this is the reference distribution"}
    staged = [("piggen_zeroshot", sorted(records), "kl_to_zero_shot")]
    staged += [(parent, sorted(children), "kl_to_sft_parent")
               for parent, children in sorted(families.items())]
    for reference_name, names, key in staged:
        reference_sha = artifacts[reference_name]["sha256"]
        pending = [(name, reference_name, reference_sha) for name in names
                   if records[name]["generation"].get(key, {}).get("reference_sha256")
                   != reference_sha
                   or records[name]["generation"].get(key, {}).get("draws_sha256")
                   != records[name]["generation_samples"]["sha256"]]
        if not pending:
            continue
        print(f"  scoring {len(pending)} draw sets under {reference_name} for {key}", flush=True)
        entry = artifacts[reference_name]
        if entry["kind"] == "pinned_zero_shot":
            model = policy_lib.load_pinned_model(raw_root, device=device)
        else:
            model = policy_lib.architecture_model(raw_root, device=device)
            policy_lib.load_checkpoint(ROOT / entry["checkpoint"], model, device=device)
        model.eval()
        policy = policy_lib.CorePolicy.from_prefix(model, context["scaffold"].prefix,
                                                   context["vocab"], device=device)
        score_draws_under(policy, pending, records, config, key)
        del policy, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def reusable_record(record, identity):
    """Whether a stored validation record still describes THIS measurement.

    Three things have to hold, and only the first of them used to be checked: the
    record's full identity block matches (config, code, sources, parent, draw seed,
    scaffold, validation pairs), the draw file it names is still on disk with the
    hash it recorded, and the persisted validation score vector is too. Anything
    else and the record is recomputed rather than carried forward under a name that
    no longer describes it.
    """
    if record.get("identity") != identity:
        return False, "identity changed"
    for key in ("generation_samples", "val_scores"):
        stored = record.get(key)
        if not stored:
            return False, f"{key} was not recorded"
        path = ROOT / stored["path"]
        if not path.is_file():
            return False, f"{key} file is missing"
        if sha256(path) != stored["sha256"]:
            return False, f"{key} file changed on disk"
    return True, "identity and artifacts verified"


def generation_offsets(artifacts):
    """Per-policy draw-seed offset: position in the sorted artifact list.

    Deterministic and stage-independent, so the freeze can re-derive the seed a
    record claims to have been drawn under instead of trusting the record.
    """
    return {name: position for position, name in enumerate(sorted(artifacts))}


def run_validate(config, context, output, artifacts, identity_base, *, raw_root, device):
    """Stream every artifact, parents first so their diversity reference exists."""
    context["generation_offsets"] = generation_offsets(artifacts)
    order = ([name for name in sorted(artifacts) if artifacts[name]["role"] == "initial"]
             + [name for name in sorted(artifacts) if artifacts[name]["role"] != "initial"])
    records, parent_draws = {}, {}
    path = Path(output) / "validation_records.json"
    existing = load_json(path) if path.is_file() else {}
    for name in order:
        entry = artifacts[name]
        identity = validation_identity(name, entry, config, identity_base, context)
        if name in existing:
            reusable, reason = reusable_record(existing[name], identity)
            if reusable:
                print(f"  reusing validation record for {name} ({reason})", flush=True)
                records[name] = existing[name]
                if entry["role"] == "initial":
                    parent_draws[name] = {
                        key: existing[name]["generation"][key]
                        for key in ("mean_pairwise_hamming", "sum_site_entropy_nats",
                                    "unique_fraction")}
                continue
            print(f"  re-validating {name}: {reason}", flush=True)
        print(f"  validating {name}", flush=True)
        record, reference = validate_one(name, entry, config, context, output, raw_root, device,
                                         parent_draws, identity)
        records[name] = record
        if entry["role"] == "initial":
            parent_draws[name] = reference
        save_json(path, records)
    reference_kl_pass(records, artifacts, config, context, output, raw_root=raw_root,
                      device=device)
    reference_relative_validation(records, artifacts, context, config)
    save_json(path, records)
    return records


# ---------------------------------------------------------------------------
# final freeze
# ---------------------------------------------------------------------------

def freeze_final(config, context, output, artifacts, records, *, base, identity_base):
    """Verify every validation record against the current identity, THEN freeze.

    The freeze is what a later stage treats as evidence, so a stale or missing
    record has to stop it here -- before the file that unlocks reserved labels
    exists. ``--stages freeze`` on its own used to reach this point having checked
    only that the names lined up.
    """
    settings = config["continuation"]
    metric = settings["selection_metric"]
    # Shape first, then evidence. Exact name sets, not counts: "24 raw checkpoints"
    # is satisfied by a table that ran one seed twice and another not at all.
    required = evaluation.required_artifact_sets(config)
    observed = {"initial_policies": sorted(n for n, e in artifacts.items()
                                           if e["role"] == "initial"),
                "raw_budget_checkpoints": sorted(n for n, e in artifacts.items()
                                                 if e["role"] == "raw_budget")}
    for key in ("initial_policies", "raw_budget_checkpoints"):
        require(observed[key] == required[key],
                f"{key} do not match the declared campaign. Missing: "
                f"{sorted(set(required[key]) - set(observed[key]))}; unexpected: "
                f"{sorted(set(observed[key]) - set(required[key]))}")
    require(set(records) == set(artifacts),
            f"Validation records and artifacts disagree: {sorted(set(artifacts) ^ set(records))}")
    required["validated"] = sorted(records)
    context["generation_offsets"] = generation_offsets(artifacts)
    for name, entry in sorted(artifacts.items()):
        reusable, reason = reusable_record(
            records[name], validation_identity(name, entry, config, identity_base, context))
        require(reusable,
                f"{name}: its validation record cannot be frozen ({reason}). Re-run "
                "--stages validate rather than freezing evidence that no longer describes the "
                "artifacts on disk.")
    evaluation.require_reference_diagnostics(records, artifacts)
    selection = {}
    for method in settings["methods"]:
        for seed in settings["seeds"]:
            table = {}
            for name, record in records.items():
                if record.get("method") == method and record.get("seed") == seed:
                    # budget_seconds is the NOMINAL target; the measured time rides
                    # along so the frozen table carries both, and the rule compares
                    # the nominal one it was declared against.
                    table[name] = {"budget_seconds": float(record["budget_seconds"]),
                                   "actual_gpu_seconds": record.get("actual_gpu_seconds"),
                                   "eligible": bool(record["diversity"]["eligible"]),
                                   "val_average_precision":
                                       record["val_metrics"]["average_precision"]}
            require(table, f"No raw checkpoints recorded for {method} seed {seed}")
            selection[run_key(method, seed)] = evaluation.select_within_budget(
                table, budgets=settings["budgets_gpu_seconds"][method], metric=metric)
    selected = dict(artifacts)
    for name, entry in base["selected"].items():
        selected.setdefault(name, dict(entry, role="baseline",
                                       budget_seconds=0.0, method=entry["kind"]))
    document = {
        "schema_version": data.SELECTION_SCHEMA,
        "stage": data.SELECTION_STAGE_FINAL,
        "config_sha256": identity_base["config_sha256"],
        "config_digest": identity_base["config_digest"],
        "code_digests": identity_base["code_digests"],
        "git_commit": base.get("git_commit"),
        "frozen_at_git_commit": git_commit(),
        "source_digests": identity_base["source_digests"],
        "base_selection_sha256": identity_base["base_selection_sha256"],
        "shared_initial_cost": base.get("shared_initial_cost"),
        "selection_metric": metric,
        "selection_rule": ("best eligible checkpoint at or below each budget by validation "
                           "average precision; ties broken toward the earlier budget. The "
                           "initial-SFT parents were selected earlier by validation positive NLL "
                           "and that rule is unchanged."),
        "diversity_gates": dict(evaluation.DIVERSITY_GATES),
        "selection_within_budget": selection,
        "selected": selected,
        "generation": {name: record["generation_samples"] for name, record in records.items()},
        "validation_records": {
            "path": relative_key(Path(output) / "validation_records.json", ROOT),
            "sha256": sha256(Path(output) / "validation_records.json"),
            "note": ("validation metrics, diversity verdicts and the Monte Carlo KLs for every "
                     "raw checkpoint; produced before any reserved label was read")},
        "required": required,
        "reserved_test_labels_read": False,
        "assay_outcomes_read": False,
        "disclosure": ("frozen before any reserved test label or assay outcome was read in this "
                       "workflow. Aggregate integrity checks and two printed example rows per "
                       "split were performed during the 2026-09-18 audit and are disclosed there; "
                       "this is not a claim that the splits were sealed from the start."),
    }
    require(dict(evaluation.DIVERSITY_GATES) == settings["diversity_gates"],
            "The config's diversity gates disagree with the preregistered constants in code")
    path = Path(output) / "selection_frozen.json"
    if path.is_file():
        existing = load_json(path)
        require(existing["selected"] == document["selected"]
                and existing["selection_within_budget"] == document["selection_within_budget"],
                f"A different selection is already frozen at {path}; move it aside deliberately "
                "rather than overwriting what an evaluation may already have consumed")
        print(f"selection already frozen at {path}", flush=True)
        return existing
    save_json(path, document)
    print(f"final selection frozen at {path}", flush=True)
    return document


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(config_path, output, *, stages, allow_dirty, allow_cpu, discard_incomplete,
        base_selection, clock_factory=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(config["schema_version"] == "her2-posttrain/1", "Unsupported protocol schema")
    settings = config["continuation"]
    require(tuple(settings["methods"]) == METHODS, "Declared continuation methods changed")
    if not allow_dirty:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        text=True).strip()
        require(not dirty, "Commit the protocol, config and code before fitting")
    raw_root = ROOT / config["raw_root"]
    problems = data.verify_all_sources(raw_root, ROOT, config["source_manifests"])
    require(not problems, f"Pinned source files failed verification: {problems}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    threads = set_cpu_threads(config["runtime"]["cpu_threads"])
    print(f"CPU threads: {threads} (declared in the config; the probe timings were taken here)",
          flush=True)
    base_path = Path(base_selection)

    preflight_path = output.parent / "preflight.json" if (output.parent / "preflight.json").is_file() \
        else output / "preflight.json"
    require(preflight_path.is_file(),
            f"No preflight decision at {preflight_path}; the batch size is chosen once, before "
            "any fitting, and continuations must not re-resolve it")
    preflight = load_json(preflight_path)
    device = preflight["device"]
    require(device == "cuda" or allow_cpu, "The declared budget is a GPU budget")
    # The batch size is frozen, but the device is read again now: resuming on a box
    # that can no longer carry it must fail rather than spill into system RAM.
    preflight_check = policy_lib.recheck_preflight(config["preflight"], preflight,
                                                   allow_cpu=allow_cpu)
    batch_sequences = (settings["batch_sequences"] if preflight["batch_size"] == 128
                       else settings["fallback_batch_sequences"])
    require(batch_sequences in (settings["batch_sequences"], settings["fallback_batch_sequences"]),
            "Unresolvable continuation batch size")
    print(f"continuation at {batch_sequences} sequences/update "
          f"({batch_sequences // 2} pairs for DPO), device {device}, "
          f"{preflight_check['current_free_vram_mib']} MiB free now", flush=True)

    # The identity is built BEFORE the base selection is read, because reading it is
    # the first thing that has to be checked against this identity.
    identity_base = {"config_sha256": sha256(config_path), "config_digest": digest_document(config),
                     "code_digests": code_digests(ROOT),
                     "source_digests": data.source_digests(ROOT, raw_root,
                                                           config["source_manifests"]),
                     "base_selection_sha256": sha256(base_path),
                     "preflight_batch_size": preflight["batch_size"], "device": device}
    base, parents = read_base_selection(base_path, config, identity_base)
    context = build_context(config, raw_root)
    results_path = output / "continuation_results.json"
    results = load_json(results_path) if results_path.is_file() else {}
    if results.get("identity") is not None:
        # Verified before the file is rewritten, and in every stage: the previous
        # version overwrote this block unconditionally, so a validate-only or
        # freeze-only invocation could relabel trajectories it never checked.
        differing = sorted(key for key in set(results["identity"]) | set(identity_base)
                           if results["identity"].get(key) != identity_base.get(key))
        require(not differing,
                f"{results_path} was written under a different identity; these runs are not this "
                f"code's runs. Differing keys: {differing}. Choose a fresh output directory rather "
                "than re-attributing completed trajectories.")
    results.setdefault("schema_version", "her2-continuation/1")
    results.setdefault("runs", {})
    results["identity"] = identity_base
    results["preflight_check"] = preflight_check
    results["shared_initial_cost"] = base.get("shared_initial_cost")
    results["runtime"] = torch_runtime()
    results["populations"] = {"train": context["train_population"].document(),
                              "val": context["val_population"].document(),
                              "validation_pairs": {k: v for k, v in context["val_pairs"].items()
                                                   if k in ("pairs", "digest")}}
    save_json(results_path, results)

    if clock_factory is None:
        def clock_factory():
            return GpuBudgetClock(device=device)

    if "continue" in stages:
        for seed in settings["seeds"]:
            for method in settings["methods"]:
                print(f"=== {method} seed {seed} ===", flush=True)
                summary = run_continuation(
                    method, seed, config, context, output, device=device,
                    batch_sequences=batch_sequences, parents=parents, identity_base=identity_base,
                    discard_incomplete=discard_incomplete, raw_root=raw_root,
                    clock_factory=clock_factory)
                results["runs"][run_key(method, seed)] = summary
                save_json(results_path, results)

    if "validate" in stages or "freeze" in stages:
        expected_runs = {run_key(method, seed) for method in settings["methods"]
                         for seed in settings["seeds"]}
        require(set(results["runs"]) == expected_runs,
                f"Every continuation trajectory must finish before validation or the freeze. "
                f"Missing: {sorted(expected_runs - set(results['runs']))}; unexpected: "
                f"{sorted(set(results['runs']) - expected_runs)}")
        artifacts = policy_artifacts(config, base, results["runs"])
    if "validate" in stages:
        print("=== validating every raw checkpoint ===", flush=True)
        run_validate(config, context, output, artifacts, identity_base, raw_root=raw_root,
                     device=device)
    if "freeze" in stages:
        records = load_json(output / "validation_records.json")
        freeze_final(config, context, output, artifacts, records, base=base,
                     identity_base=identity_base)
    print("post-training stage complete", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiments/her2_posttrain.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/her2_posttrain_20260918/continuation")
    parser.add_argument("--base-selection", type=Path,
                        default=ROOT / f"outputs/her2_posttrain_20260918/{BASE_SELECTION}")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--discard-incomplete", action="store_true",
                        help="restart a trajectory left running by a killed process")
    args = parser.parse_args()
    chosen = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = sorted(set(chosen) - set(STAGES))
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    run(args.config, args.output, stages=chosen, allow_dirty=args.allow_dirty,
        allow_cpu=args.allow_cpu, discard_incomplete=args.discard_incomplete,
        base_selection=args.base_selection)
