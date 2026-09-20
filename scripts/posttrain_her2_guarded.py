#!/usr/bin/env python
"""Guarded HER2 continuation: inspect, plan, continue, validate, freeze.

The revision this drives is one change of question. The 2026-09-18 campaign asked
"does more budget rank better?" and got yes while the policy's chosen likelihood
collapsed. This asks "does more budget rank better **while the parent's
distribution survives?**", and enforces the second half with a parent-relative
likelihood gate that stops a trajectory between updates.

Stages, in this order and no other:

``inspect``
    Recover the original six per-update histories, hash every input, summarize
    the declared windows, the per-cycle regime and the next-update learning rate,
    and -- where the frozen reference cache is on disk, with its identity sidecar
    verified against what that run recorded and at the batch size that run
    recorded -- recompute the training batch-mean chosen drop from the
    deterministic pairing. Also re-verifies that the original scientific sources
    still hash as the initial-SFT freeze recorded them. Writes
    ``inspect_report.json``. CPU only; reads nothing reserved.

``plan``
    Recompute the staged allocation from the declared grid and **fail** if it does
    not reconcile with the config's totals. Prints the gate, the arms and the
    costs that are deliberately outside the ceiling. ``--measure-monitor-check``
    optionally performs one real parent-vs-parent gate check and reports its
    measured GPU and wall cost; it performs no update and is never implicit.

``continue --stage N``
    Refuses without verified inspection evidence, and for N > 1 without a verified
    ``stage{N-1}_complete.json`` naming its artifacts by hash. Then runs one
    guarded trajectory per arm and seed from the same parent, the same optimizer,
    the same pairing and the same frozen training reference the original used.

``validate``
    Checks before it measures: the declared arm x seed grid is all present, each
    trajectory carries this campaign's whole identity, each checkpoint still
    hashes to what its trajectory recorded **and** its payload names this arm,
    seed, budget, campaign and progress, the frozen parent reference is verified
    against the identity rebuilt from the selected parent and the actual ordered
    pairs, and the gate verdict that checkpoint exists because of is re-read from
    ``monitor.jsonl`` and **recomputed** from that check's retained score vectors
    against that reference under the declared threshold. Then it measures:
    validation ranking aggregate and per stratum, chosen/rejected NLL on the fixed
    pairs, 10,000 temperature-1 draws with the diversity verdicts, and Monte Carlo
    KL to the SFT parent with its standard error. Score vectors and draws are
    persisted with their hashes, and the whole validation clock -- parent draws and
    KL included -- is reported.

``freeze``
    Re-verifies those bytes, requires the validated document to carry this whole
    campaign identity and **exactly one** endpoint per actually reached
    trajectory-budget -- no duplicates, no foreign checkpoints, no endpoint
    relabelled to a budget that was never reached and none quietly missing --
    enumerates **every** declared trajectory including the
    stopped ones, applies the prespecified selection rule to every declared
    objective (an objective whose runs all stopped is reported "none eligible",
    not omitted), and writes the stage marker the next stage will demand. Stages 2
    and 3 reuse the stage-1 ``continued_sft`` control through that marker under
    hash verification instead of re-fitting or re-measuring it. The document
    carries a schema and stage marker outside ``SELECTION_STAGES``, so it cannot
    unlock a reserved test label.

``continue`` requires CUDA: the budgets are measured GPU seconds, and
``--allow-cpu`` is for the no-fit stages only.

No stage here reads a reserved test label or an assay outcome.
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
from smallAntibodyGen.experiments import her2_guard as guard  # noqa: E402
from smallAntibodyGen.experiments import her2_guarded_eval as selection_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_guarded_trajectory as trajectory_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_history as history_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_lineage as lineage  # noqa: E402
from smallAntibodyGen.experiments import her2_objectives as objectives  # noqa: E402
from smallAntibodyGen.experiments import her2_policy as policy_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_preferences as preferences  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import (  # noqa: E402
    GpuBudgetClock, Progress, load_json, relative_key, require, save_json, set_cpu_threads,
    sha256, torch_runtime)

STAGES = ("inspect", "plan", "continue", "validate", "freeze")
CONFIG_SCHEMA = "her2-guarded-continuation/1"
INSPECT_SCHEMA = "her2-guarded-inspection/1"
PLAN_SCHEMA = "her2-guarded-plan/1"
ENDPOINT_SCHEMA = "her2-guarded-endpoint/1"

#: The six original trajectories this revision inspects but never rewrites.
ORIGINAL_RUNS = ("dpo_seed20260918", "dpo_seed20260919", "dpo_seed20260920",
                 "continued_sft_seed20260918", "continued_sft_seed20260919",
                 "continued_sft_seed20260920")


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_config(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    require(document["schema_version"] == CONFIG_SCHEMA,
            f"Unsupported guarded protocol schema {document.get('schema_version')!r}")
    return document


def trajectory_key(arm, seed):
    return f"{arm['arm_id']}_seed{seed}"


def checkpoint_name(arm, seed, budget):
    """One name per (arm, seed, budget), derived in the one place that defines it."""
    return selection_lib.endpoint_name(arm["arm_id"], seed, budget)


# ---------------------------------------------------------------------------
# identity and context
# ---------------------------------------------------------------------------

def build_identity(config, config_path, *, raw_root, device, batch_sequences):
    """The two-block identity, plus the verified parents bound into the inherited half."""
    inherited = config["inherited"]
    identity = lineage.guarded_identity(
        ROOT, config_path=config_path,
        original_config_path=ROOT / inherited["original_config"],
        base_selection_path=ROOT / inherited["base_selection"],
        source_digests=data.source_digests(ROOT, raw_root, config["source_manifests"]),
        device=device, batch_sequences=batch_sequences)
    lineage.verify_inherited_code(ROOT, identity)
    expected = _base_selection_names(config)
    _, parents = lineage.read_original_freeze(
        ROOT / inherited["base_selection"], root=ROOT, identity=identity,
        expected_selected=expected, seeds=config["seeds"])
    return lineage.bind_parents(identity, parents), parents


def _base_selection_names(config):
    """Exactly what the initial-SFT freeze must name, rebuilt from the ORIGINAL config."""
    original = json.loads((ROOT / config["inherited"]["original_config"]).read_text(
        encoding="utf-8"))
    return sorted(set(evaluation.initial_policy_names(original))
                  | {f"cnn_3class_seed{seed}" for seed in original["classifier"]["seeds"]}
                  | {"piggen_zeroshot", "linear_3class"})


def build_context(config, raw_root):
    """Splits, eligible populations, the pairing and the fixed validation pairs.

    The expected counts are asserted against the config before anything is fitted,
    so a changed release stops here rather than producing a differently sized
    "fixed" validation pair set under the same name.
    """
    train = data.load_split(raw_root, "train")
    val = data.load_split(raw_root, "val")
    train_population = preferences.build_population(train, "train")
    val_population = preferences.build_population(val, "val")
    expected = config["expected_counts"]
    require(int(train_population.chosen_index.shape[0]) == expected["train_chosen"],
            f"Eligible training chosen rows {train_population.chosen_index.shape[0]} != "
            f"{expected['train_chosen']}")
    require(int(train_population.rejected_index.shape[0]) == expected["train_rejected"],
            "Eligible training rejected rows changed")
    require(int(val_population.chosen_index.shape[0]) == expected["val_chosen"],
            "Eligible validation chosen rows changed")

    reference_index = np.concatenate([train_population.chosen_index,
                                      train_population.rejected_index], axis=0)
    require(reference_index.shape[0] == expected["reference_rows"],
            f"Reference population is {reference_index.shape[0]} rows, expected "
            f"{expected['reference_rows']}")
    rejected_rows = (train_population.chosen_index.shape[0]
                     + np.arange(train_population.rejected_index.shape[0]))

    val_pairing = preferences.PreferencePairing(val_population,
                                                seed=config["pairing"]["validation_seed"])
    val_pairs = val_pairing.fixed_validation_pairs(config["pairing"]["validation_pairs"])
    require(int(val_pairs["pairs"]) == expected["val_chosen"],
            f"The gate must see every fixed validation pair: {val_pairs['pairs']} != "
            f"{expected['val_chosen']}. There is no monitor subset in this protocol.")
    val_index = data.encode_cores(val.seq)
    train_index = data.encode_cores(train.seq)
    train_labels = (train["class"] == data.POSITIVE_CLASS).to_numpy().astype(np.float64)
    val_positive = (val["class"] == data.POSITIVE_CLASS).to_numpy()
    split_of, class_of = data.labelled_lookup({"train": train, "val": val})
    for core in data.test_sequences(raw_root):
        split_of.setdefault(core, "test")
    return {
        "train": train, "val": val, "train_index": train_index, "train_labels": train_labels,
        "val_index": val_index, "val_positive": val_positive,
        "val_positive_index": val_index[val_positive],
        "val_cores": val.seq.to_numpy(dtype=str),
        "train_population": train_population, "val_population": val_population,
        "reference_index": reference_index, "reference_rejected_rows": rejected_rows,
        "val_pairs": val_pairs,
        "catalogs": {"train": set(train.seq), "val": set(val.seq),
                     "test": set(data.test_sequences(raw_root))},
        "split_of": split_of, "class_of": class_of,
        "training_diversity_reference": evaluation.diversity_reference(
            train_index[train_labels.astype(bool)]),
        "scaffold": data.load_scaffold(raw_root),
        "vocab": policy_lib.load_vocab(raw_root),
    }


def validation_strata(config, context):
    """Training-distance strata for the validation rows. Data-only, computed once."""
    lookup = data.nearest_training_labels(
        context["val_index"], context["train_index"], context["train_labels"],
        max_distance=int(config["evaluation"]["max_train_distance"]))
    return lookup.strata()


def load_parent_policy(entry, raw_root, context, device):
    model = policy_lib.architecture_model(raw_root, device=device)
    policy_lib.load_checkpoint(ROOT / entry["checkpoint"], model, device=device)
    return policy_lib.CorePolicy.from_prefix(model, context["scaffold"].prefix, context["vocab"],
                                             device=device)


# ---------------------------------------------------------------------------
# stage: inspect
# ---------------------------------------------------------------------------

def original_run_paths(config, *, runs=ORIGINAL_RUNS):
    root = ROOT / config["inherited"]["original_continuation_root"]
    summaries = {name: root / name / "summary.json" for name in runs}
    caches = {name: root / name / "reference_cache.npy" for name in runs}
    return root, summaries, {name: path for name, path in caches.items() if path.is_file()}


def run_inspect(config, output, identity):
    """Recover the original histories, hash them, and record what is missing."""
    root, summaries, caches = original_run_paths(config)
    missing = sorted(name for name, path in summaries.items() if not path.is_file())
    require(not missing,
            f"These original summaries are absent under {root}: {missing}. The guarded "
            "continuation is a revision of that campaign and will not proceed without its "
            "recorded histories.")
    report = history_lib.inspect_original_campaign(
        summaries, root=root,
        reference={"cache_paths": caches,
                   "pairing_seed_base": config["pairing"]["seed_base"],
                   "batch_pairs": int(config["batch_sequences"]) // 2,
                   "chosen_count": int(config["expected_counts"]["train_chosen"])})
    document = {"schema_version": INSPECT_SCHEMA,
                "identity": {"revision": identity["revision"], "inherited": identity["inherited"]},
                "history": report,
                "original_code_digests_verified": True,
                "note": ("the original campaign's artifacts are read-only inputs here. This "
                         "revision writes nothing under their output root.")}
    path = Path(output) / "inspect_report.json"
    save_json(path, document)
    print(f"inspection written to {path}", flush=True)
    for name, entry in sorted(report["runs"].items()):
        drop = entry["batch_chosen_drop"]
        batch = entry.get("batch") or {}
        pairs = (batch.get("batch_pairs") if batch.get("pairs_applicable", True)
                 else "no (chosen only)")
        print(f"  {name}: {entry['updates']} updates, cycles "
              f"{sorted(entry['cycles'], key=int)}, recorded batch "
              f"{batch.get('batch_sequences')} sequences / {pairs} pairs, "
              f"batch-drop reconstruction "
              f"{'available' if drop.get('available') else 'unavailable'}", flush=True)
    return document


def require_inspection(output, identity):
    """Fail closed: the inspection must exist, match this revision, and still hash true.

    "Still hash true" covers every input the inspection used -- the six summaries,
    the frozen reference caches and their identity sidecars -- not only the
    summaries. The reconstruction is a statement about those cache values, so a
    cache that changed since it was read invalidates the evidence this stage is
    standing on.
    """
    path = Path(output) / "inspect_report.json"
    require(path.is_file(),
            f"No inspection evidence at {path}. Run `--stage inspect` first; a staged "
            "continuation may not bypass the recovery of what the original campaign recorded.")
    document = load_json(path)
    require(document.get("schema_version") == INSPECT_SCHEMA, f"{path} is not an inspection report")
    require(document["identity"]["revision"] == identity["revision"],
            f"{path} was produced by different revision code or config; re-run inspect")
    for name, entry in sorted(document["history"]["inputs"].items()):
        source = Path(entry["path"])
        require(source.is_file(), f"{name}: the inspected summary {source} is gone")
        digest = sha256(source)
        require(digest == entry["sha256"],
                f"{name}: {source} changed since it was inspected ({digest} != {entry['sha256']}). "
                "The evidence this stage depends on is not the evidence on disk.")
    return document


# ---------------------------------------------------------------------------
# stage: plan
# ---------------------------------------------------------------------------

def stage_arms(config, stage):
    """The arms fitted at one stage, and the controls it reuses from an earlier one."""
    arms = selection_lib.declared_arms(config)
    fitted = [arm for arm in arms if arm["stage"] == int(stage) and not arm["reused_from_stage"]]
    reused = [arm for arm in arms if arm["stage"] == int(stage) and arm["reused_from_stage"]]
    require(fitted, f"Stage {stage} declares no fitted arms")
    return fitted, reused


def run_plan(config, output, identity, *, measure_monitor_check=False, raw_root=None,
             device="cpu", parents=None, context=None):
    """Recompute the allocation from the grid and refuse to proceed if it disagrees."""
    allocation = selection_lib.require_allocation_reconciles(config)
    arms = selection_lib.declared_arms(config)
    document = {
        "schema_version": PLAN_SCHEMA,
        "identity": {"revision": identity["revision"], "inherited": identity["inherited"]},
        "allocation": allocation,
        "arms": [dict(arm, description=objectives.describe(arm["objective"], arm["coefficients"]))
                 for arm in arms],
        "gate": dict(config["gate"]),
        "selection": dict(config["selection"]),
        "budgets_gpu_seconds": [float(b) for b in config["budgets_gpu_seconds"]],
        "seeds": list(config["seeds"]),
        "monitor_cost": {"measured": False,
                         "note": ("the per-check cost of scoring every fixed validation pair on "
                                  "both sides is measured at run time and reported per "
                                  "trajectory. No estimate is asserted here.")},
    }
    if measure_monitor_check:
        document["monitor_cost"] = measure_monitor_check_cost(
            config, raw_root=raw_root, device=device, parents=parents, context=context)
    path = Path(output) / "plan.json"
    save_json(path, document)
    print(f"plan written to {path}", flush=True)
    print(f"  {allocation['configurations_fitted']} fitted configurations x "
          f"{allocation['seeds']} seeds x {max(document['budgets_gpu_seconds']):.0f} s = "
          f"{allocation['total_charged_training_gpu_seconds']:.0f} charged training GPU seconds",
          flush=True)
    print(f"  NOT {allocation['incorrect_sum_of_budgets']:.0f}: a trajectory drops three "
          "checkpoints, it does not run three times", flush=True)
    for stage, entry in sorted(allocation["stages"].items()):
        print(f"  stage {stage}: {entry['allocation_gpu_seconds']:.0f} s over "
              f"{entry['fitted_configurations']} fitted configurations", flush=True)
    return document


def measure_monitor_check_cost(config, *, raw_root, device, parents, context):
    """One real parent-vs-parent gate check, measured. Performs no update.

    Explicitly opt-in. It exists so the monitor's cost is an observation rather
    than an assumption before the grid is committed, and it is the same code path
    a trajectory uses -- a check against the parent itself, whose D is 0 by
    construction.
    """
    require(context is not None and parents is not None,
            "The measured monitor check needs the data context and the verified parents")
    seed = int(config["seeds"][0])
    policy = load_parent_policy(parents[seed], raw_root, context, device)
    monitor_clock = GpuBudgetClock(device=device) if device == "cuda" else None
    identity_block = guard.parent_reference_identity(
        parent_checkpoint_sha256=parents[seed]["sha256"],
        parent_state_sha256=policy_lib.state_digest(policy.model),
        config_sha256="measurement-only", scaffold_prefix=context["scaffold"].prefix,
        chosen_index=context["val_pairs"]["chosen_index"],
        rejected_index=context["val_pairs"]["rejected_index"])
    started = time.perf_counter()
    reference = guard.build_parent_reference(
        policy, context["val_pairs"], identity_block, clock=monitor_clock,
        batch_size=config["inference"]["monitor_batch_size"])
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        monitor = guard.GateMonitor(policy, reference, directory=scratch,
                                    batch_size=config["inference"]["monitor_batch_size"],
                                    monitor_clock=monitor_clock)
        record = monitor.check(context["val_pairs"], update=0, gpu_seconds=0.0,
                               reason="measurement_only")
    return {"measured": True, "seed": seed, "pairs": record["pairs"],
            "D": record["D"], "D_per_residue": record["D_per_residue"],
            "monitor_gpu_seconds": record["monitor_gpu_seconds"],
            "monitor_wall_seconds": record["monitor_wall_seconds"],
            "parent_reference_gpu_seconds": guard.json_number(reference.gpu_seconds),
            "parent_reference_wall_seconds": guard.json_number(reference.wall_seconds),
            "whole_measurement_wall_seconds": guard.json_number(time.perf_counter() - started),
            "note": ("one real check of the parent against itself, so D is 0 by construction. "
                     "No update was performed and nothing was fitted.")}


# ---------------------------------------------------------------------------
# stage: continue
# ---------------------------------------------------------------------------

def prepare_parent_reference(policy, context, config, directory, parent, *, monitor_clock):
    """Build or reuse the parent's fixed-validation scores. Never on the training clock."""
    identity_block = guard.parent_reference_identity(
        parent_checkpoint_sha256=parent["sha256"],
        parent_state_sha256=policy_lib.state_digest(policy.model),
        config_sha256=parent["config_sha256"], scaffold_prefix=context["scaffold"].prefix,
        chosen_index=context["val_pairs"]["chosen_index"],
        rejected_index=context["val_pairs"]["rejected_index"])
    path = Path(directory) / "parent_validation_reference.npz"
    if path.is_file():
        return guard.load_parent_reference(path, identity_block)
    reference = guard.build_parent_reference(
        policy, context["val_pairs"], identity_block, clock=monitor_clock,
        batch_size=config["inference"]["monitor_batch_size"])
    guard.save_parent_reference(path, reference)
    return reference


def training_reference(policy, context, config, directory, parent, clock):
    """The frozen TRAINING reference: the original cache, reused under full identity.

    Same parent weights as the gate's reference, different rows. The two are named
    apart and never substituted: the gate refuses a vector carrying the training
    population's digest, and this one refuses a cache whose recorded config,
    checkpoint, prefix, convention or core order differs by a byte.

    Reuse is charged: the cache's ORIGINAL creation seconds go on this
    trajectory's training clock as a cold-start cost, exactly as the original
    campaign did, and the warm reuse wall time is reported separately. "Physically
    reused" and "charged as a cold start" are both recorded, because they are
    different facts and either one alone misstates what the run spent.

    The parity probe is charged too, on its own segment. It is real device work
    that the protocol requires before this arm may train -- it is how a cache that
    passed its identity check but came from different weights is caught -- so it
    goes on the training clock as its own line rather than disappearing into the
    wall-time residue.
    """
    settings = config["reference"]
    identity_block = preferences.reference_identity(
        checkpoint_sha256=parent["sha256"], config_sha256=parent["original_config_sha256"],
        scaffold_prefix=context["scaffold"].prefix, index=context["reference_index"])
    path = Path(directory) / "reference_cache.npy"
    original = Path(parent.get("original_reference_cache") or "")
    if not path.is_file() and settings.get("reuse_original_cache") and original.is_file():
        path = original
    warm_reuse_wall, fresh_build_wall = None, None
    if path.is_file():
        reuse_started = time.perf_counter()
        cache = preferences.load_reference_cache(path, identity_block)
        clock.charge_reused(cache.creation_gpu_seconds, reason="reused frozen reference cache")
        # What the reuse physically cost: reading and re-verifying the bytes. It is
        # measured because "charged as a cold start" and "took this long" are
        # different facts, and reporting the first alone overstates the work done.
        warm_reuse_wall = time.perf_counter() - reuse_started
        physically_reused = True
    else:
        build_started = time.perf_counter()
        cache = preferences.build_reference_cache(
            policy, context["reference_index"], identity_block, clock=clock,
            batch_size=settings["score_batch_size"])
        preferences.save_reference_cache(path, cache)
        fresh_build_wall = time.perf_counter() - build_started
        physically_reused = False
    probe = np.random.default_rng(settings["parity_probe_seed"]).choice(
        cache.values.size, size=min(settings["parity_probe_rows"], cache.values.size),
        replace=False)
    parity_started = time.perf_counter()
    with clock.segment() as parity_segment:
        parity = preferences.verify_reference_parity(
            policy, cache, context["reference_index"], probe,
            atol=config["tolerances"]["sum_log_probability_atol"],
            rtol=config["tolerances"]["sum_log_probability_rtol"])
    document = dict(cache.document(), fresh_parity=parity, path=str(path),
                    physically_reused=physically_reused,
                    cold_start_charged_gpu_seconds=guard.json_number(cache.creation_gpu_seconds),
                    warm_reuse_wall_seconds=guard.json_number(warm_reuse_wall),
                    fresh_build_wall_seconds=guard.json_number(fresh_build_wall),
                    parity_probe_rows=int(probe.size),
                    parity_probe_charged_gpu_seconds=guard.json_number(parity_segment.seconds),
                    parity_probe_wall_seconds=guard.json_number(
                        time.perf_counter() - parity_started),
                    charge_note=("the cold-start cost is charged to this trajectory even when the "
                                 "bytes were physically produced once. Both numbers are reported "
                                 "so neither stands in for the other. The parity probe is separate "
                                 "device work and is charged on its own segment, not folded into "
                                 "the cold-start number."))
    return cache, document


def build_step(arm, policy, context, config, cache, batch_sequences, state):
    """The per-update closure for one arm. Continued SFT never touches a rejected row."""
    name = arm["objective"]
    coefficients = arm["coefficients"]
    population = context["train_population"]
    chosen_total = int(population.chosen_index.shape[0])
    if name == "continued_sft":
        rejected_total = 0

        def step(update):
            if state["order"] is None or state["position"] >= chosen_total:
                if state["order"] is not None:
                    state["cycle"] += 1
                state["order"] = np.random.default_rng(
                    [state["pairing_seed"], state["cycle"]]).permutation(chosen_total)
                state["position"] = 0
            rows = state["order"][state["position"]:state["position"] + batch_sequences]
            state["position"] += len(rows)
            chosen_index = population.chosen_index[rows]
            scores = policy.sequence_log_probs(chosen_index)
            loss, diagnostics = objectives.batch_loss("continued_sft", policy_chosen=scores)
            state["seen_chosen"].update(rows.tolist())
            statistics = {"cycle": state["cycle"], "position": int(state["position"]),
                          "sequences": int(len(rows)), "pairs": 0,
                          "batch_unique_chosen": int(np.unique(rows).size),
                          "per_pair": diagnostics}
            return loss, statistics
    else:
        rejected_total = int(population.rejected_index.shape[0])
        batch_pairs = batch_sequences // 2

        def step(update):
            cycle, number, chosen_rows, rejected_rows = next(state["stream"])
            chosen_index, rejected_index = state["pairing"].pair_cores(chosen_rows, rejected_rows)
            reference_chosen = cache.tensor(chosen_rows, device=policy.device)
            reference_rejected = cache.tensor(
                context["reference_rejected_rows"][rejected_rows], device=policy.device)
            objectives.require_frozen_references(reference_chosen, reference_rejected,
                                                 where=arm["arm_id"])
            policy_chosen = policy.sequence_log_probs(chosen_index)
            policy_rejected = policy.sequence_log_probs(rejected_index)
            loss, diagnostics = objectives.batch_loss(
                name, policy_chosen=policy_chosen, policy_rejected=policy_rejected,
                reference_chosen=reference_chosen.to(policy_chosen.dtype),
                reference_rejected=reference_rejected.to(policy_rejected.dtype),
                coefficients=coefficients)
            state["seen_chosen"].update(chosen_rows.tolist())
            state["seen_rejected"].update(rejected_rows.tolist())
            statistics = {"cycle": int(cycle), "batch": int(number),
                          "sequences": int(2 * len(chosen_rows)), "pairs": int(len(chosen_rows)),
                          "batch_unique_chosen": int(np.unique(chosen_rows).size),
                          "batch_unique_rejected": int(np.unique(rejected_rows).size),
                          "per_pair": diagnostics,
                          "pair_cores": (chosen_index, rejected_index)}
            return loss, statistics
        state["batch_pairs"] = batch_pairs

    def cumulative_exposures():
        document = {"distinct_chosen_rows": len(state["seen_chosen"]),
                    "distinct_rejected_rows": len(state["seen_rejected"]),
                    "chosen_population": chosen_total, "rejected_population": rejected_total,
                    "distinct_chosen_fraction": len(state["seen_chosen"]) / max(1, chosen_total),
                    "unit": "distinct eligible training rows reached so far in this trajectory"}
        if rejected_total:
            document["distinct_rejected_fraction"] = (len(state["seen_rejected"])
                                                      / max(1, rejected_total))
        else:
            document["note"] = ("continued SFT trains on the chosen positives only: no pairs, no "
                                "rejected rows and no reference cache")
        return document

    return step, cumulative_exposures


def run_trajectory(arm, seed, config, context, output, *, stage, device, batch_sequences, parents,
                   identity, raw_root, clock_factory, monitor_clock_factory, allow_dirty):
    """One arm at one seed: parent -> guarded continuation -> durable evidence."""
    directory = Path(output) / f"stage{int(stage)}" / trajectory_key(arm, seed)
    parent = dict(parents[seed])
    run_identity = dict(identity, arm=arm, seed=int(seed), stage=int(stage),
                        budgets_gpu_seconds=[float(b) for b in config["budgets_gpu_seconds"]],
                        gate=dict(config["gate"]), optimization=dict(config["optimization"]),
                        pairing=dict(config["pairing"]))
    ledger = lineage.GuardedRunLedger(directory, run_identity,
                                      metadata={"git_commit": None if allow_dirty else git_commit()})
    if ledger.claim() == "completed":
        print(f"skipping completed trajectory {directory.name}", flush=True)
        return load_json(ledger.path)["summary"]

    run_started = time.perf_counter()
    policy = load_parent_policy(parent, raw_root, context, device)
    monitor_clock = monitor_clock_factory()
    reference = prepare_parent_reference(policy, context, config, directory, parent,
                                         monitor_clock=monitor_clock)
    clock = clock_factory()
    cache, cache_document = (None, None)
    if arm["objective"] != "continued_sft":
        cache, cache_document = training_reference(policy, context, config, directory, parent,
                                                   clock)
    policy.model.train()
    optimizer = preferences.build_optimizer(policy.model, config["optimization"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: policy_lib.continuation_learning_rate_scale(
            step + 1, warmup_steps=config["optimization"]["warmup_updates"]))

    state = {"cycle": 0, "position": 0, "order": None, "seen_chosen": set(),
             "seen_rejected": set(), "pairing_seed": config["pairing"]["seed_base"] + int(seed)}
    if arm["objective"] != "continued_sft":
        pairing = preferences.PreferencePairing(context["train_population"],
                                                seed=state["pairing_seed"])
        state["pairing"] = pairing
        state["stream"] = pairing.stream(batch_sequences // 2)
    step, cumulative_exposures = build_step(arm, policy, context, config, cache, batch_sequences,
                                            state)

    gate = guard.LikelihoodGate(
        threshold_nats_per_sequence=float(config["gate"]["threshold_nats_per_sequence"]))
    schedule = guard.MonitorSchedule(
        first_update=int(config["gate"]["first_update"]),
        update_interval=int(config["gate"]["update_interval"]),
        gpu_second_interval=float(config["gate"]["gpu_second_interval"]))
    monitor = guard.GateMonitor(policy, reference, gate=gate, schedule=schedule,
                                directory=directory,
                                batch_size=config["inference"]["monitor_batch_size"],
                                monitor_clock=monitor_clock)

    def monitor_check(*, update, gpu_seconds, reason):
        record = monitor.check(context["val_pairs"], update=update, gpu_seconds=gpu_seconds,
                               reason=reason)
        print(f"  check {record['check']} at update {update}: D={record['D']} "
              f"({record['D_per_residue']} /residue) threshold "
              f"{record['threshold_nats_per_sequence']} -> "
              f"{'pass' if record['passed'] else 'STOP'}", flush=True)
        return record

    def on_budget(budget, record):
        started = time.perf_counter()
        path = directory / f"budget_{int(budget)}.pt"
        digest = policy_lib.save_checkpoint(path, policy, {
            "arm": arm, "seed": int(seed), "budget_seconds": float(budget),
            "identity": run_identity, "progress": record})
        return {"checkpoint": relative_key(path, ROOT), "checkpoint_sha256": sha256(path),
                "state_sha256": digest,
                "checkpoint_wall_seconds": time.perf_counter() - started}

    progress = Progress(directory / "progress.json", None, every=50, label=directory.name)
    summary = {"arm": arm, "seed": int(seed), "stage": int(stage), "identity": run_identity,
               "parent": {"checkpoint": parent["checkpoint"], "sha256": parent["sha256"],
                          "state_sha256": policy_lib.state_digest(policy.model)},
               "batch_sequences": batch_sequences}
    status = "completed"
    try:
        result = trajectory_lib.run_guarded_trajectory(
            step=step, budgets=config["budgets_gpu_seconds"], clock=clock, optimizer=optimizer,
            scheduler=scheduler, model=policy.model,
            gradient_clip=config["optimization"]["gradient_clip"], monitor_check=monitor_check,
            directory=directory, parent_state=parent, on_budget=on_budget,
            exposure_fields=("sequences", "pairs"),
            cumulative_exposures=cumulative_exposures, progress=progress, schedule=schedule,
            identity=run_identity)
        summary.update(result)
        status = result.get("status", "failed")
    except BaseException:
        status = "failed"
        raise
    finally:
        progress.finish()
        if status == "failed":
            # The controller writes its document in its own `finally`; the exception
            # then skipped `summary.update(result)` above. Reading that document back
            # is the difference between a summary that reports what the failed run
            # actually did -- its completed updates, its charged seconds, its reached
            # and outstanding budgets -- and one that reports nothing at all.
            persisted = directory / "trajectory.json"
            if persisted.is_file():
                summary.update(load_json(persisted))
                summary["controller_document"] = {
                    "recovered": True, "path": relative_key(persisted, ROOT),
                    "note": ("this trajectory failed; its counters, costs and budget accounting "
                             "are the controller's own persisted document, not defaults")}
            else:
                summary["controller_document"] = {
                    "recovered": False,
                    "note": ("this trajectory failed before the controller persisted a document, "
                             "so its counters and costs are unavailable. They are reported "
                             "missing rather than as zeros.")}
        summary["monitor"] = monitor.document()
        summary["training_reference"] = cache_document
        summary["clock"] = clock.document()
        summary["monitor_clock"] = monitor_clock.document() if monitor_clock else None
        summary["excluded_costs"] = {
            "parent_validation_reference_gpu_seconds": guard.json_number(reference.gpu_seconds),
            "parent_validation_reference_wall_seconds": guard.json_number(reference.wall_seconds),
            "parent_validation_reference_reused_from_disk": bool(reference.reused),
            "monitor_checks": monitor.checks,
            "monitor_gpu_seconds": guard.json_number(monitor.gpu_seconds),
            "monitor_wall_seconds": guard.json_number(monitor.wall_seconds),
            "note": ("real costs that are NOT training. They are measured and reported here, "
                     "never subtracted from the training budget and never capped by the "
                     "allocation arithmetic.")}
        summary["whole_run_wall_seconds"] = time.perf_counter() - run_started
        trajectory_cost = dict((summary.get("cost") or {}))
        budget_checkpoint_wall = sum(float(record.get("checkpoint_wall_seconds") or 0.0)
                                     for record in (summary.get("budgets") or {}).values())
        reference_costs = {
            "physically_reused": (cache_document or {}).get("physically_reused"),
            "cold_start_charged_gpu_seconds":
                (cache_document or {}).get("cold_start_charged_gpu_seconds"),
            "warm_reuse_wall_seconds": (cache_document or {}).get("warm_reuse_wall_seconds"),
            "fresh_build_wall_seconds": (cache_document or {}).get("fresh_build_wall_seconds"),
            "parity_probe_charged_gpu_seconds":
                (cache_document or {}).get("parity_probe_charged_gpu_seconds"),
            "measured": cache_document is not None,
            "note": ("physical reuse and the charged cold start are different facts and both are "
                     "recorded: warm_reuse_wall_seconds is what reading and re-verifying the "
                     "frozen bytes actually took, and cold_start_charged_gpu_seconds is what the "
                     "arm was charged for them regardless. The arm paid the cold start whether or "
                     "not it re-scored the rows.")}
        if cache_document is None:
            reference_costs["unavailable_reason"] = (
                "continued SFT trains on the chosen positives only: it builds and reuses no "
                "training reference, so there is no cost here rather than a cost of zero")
        summary["cost_summary"] = {
            "charged_training_gpu_seconds": guard.json_number(clock.elapsed_seconds),
            "training_reference": reference_costs,
            "monitoring": {"checks": monitor.checks,
                           "gpu_seconds": guard.json_number(monitor.gpu_seconds),
                           "wall_seconds": guard.json_number(monitor.wall_seconds),
                           "parent_reference_gpu_seconds":
                               guard.json_number(reference.gpu_seconds),
                           "parent_reference_wall_seconds":
                               guard.json_number(reference.wall_seconds),
                           "parent_reference_reused_from_disk": bool(reference.reused),
                           "note": ("the parent's own scoring of every fixed validation pair is "
                                    "part of what monitoring cost, and is counted here rather "
                                    "than left in the residue between the clocks")},
            "io_wall_seconds": {
                "rolling_last_passing_checkpoints":
                    trajectory_cost.get("checkpoint_wall_seconds"),
                "nominal_budget_checkpoints": guard.json_number(budget_checkpoint_wall),
                "diagnostics_and_journals": trajectory_cost.get("diagnostic_wall_seconds"),
                "budget_evaluation": trajectory_cost.get("budget_evaluation_wall_seconds"),
                "note": ("the rolling last-passing saves and the nominal endpoint writes are "
                         "separate work and are reported separately")},
            "work_done": {"updates": summary.get("updates"),
                          "attempted_updates": summary.get("attempted_updates"),
                          "failed_attempts": summary.get("failed_attempts"),
                          "failed_work_gpu_seconds": summary.get("failed_work_gpu_seconds"),
                          "exposures": summary.get("exposures"),
                          "core_token_exposures": summary.get("core_token_exposures"),
                          "distinct_exposures": summary.get("distinct_exposures"),
                          "budgets_reached": sorted(summary.get("budgets") or {}),
                          "budgets_not_reached": sorted(summary.get("budgets_not_reached") or {})},
            "status": summary.get("status", status),
            "whole_run_wall_seconds": summary["whole_run_wall_seconds"],
            "note": ("one place that says what this trajectory actually cost: charged training "
                     "seconds, the reference it reused and what that was charged as, the "
                     "monitoring it paid for outside the budget -- the parent reference "
                     "included -- the I/O split by what wrote it, the updates and exposures it "
                     "really bought and the whole elapsed run. A quantity that was not measured "
                     "is null with a reason, never 0.")}
        summary["runtime"] = torch_runtime()
        if torch.cuda.is_available():
            summary["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
            torch.cuda.reset_peak_memory_stats()
        save_json(directory / "summary.json", summary)
        # The ledger says what happened. A failed run is marked failed rather than
        # left reading "running": either way the directory is never reclaimed, but
        # only one of the two is true.
        ledger.finish(summary, status="completed" if status in ("completed", "stopped")
                      else status)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_continue(config, context, output, identity, parents, *, stage, device, batch_sequences,
                 raw_root, clock_factory, monitor_clock_factory, allow_dirty):
    """Every fitted arm of one stage, at every seed, after the prerequisites verify."""
    require_inspection(output, identity)
    selection_lib.require_previous_stage(output, stage, identity=identity, root=ROOT)
    selection_lib.require_allocation_reconciles(config)
    fitted, reused = stage_arms(config, stage)
    print(f"=== stage {stage}: {len(fitted)} fitted arms, {len(reused)} reused controls ===",
          flush=True)
    summaries = {}
    for arm in fitted:
        for seed in config["seeds"]:
            print(f"=== {arm['arm_id']} seed {seed} ===", flush=True)
            summaries[trajectory_key(arm, seed)] = run_trajectory(
                arm, seed, config, context, output, stage=stage, device=device,
                batch_sequences=batch_sequences, parents=parents, identity=identity,
                raw_root=raw_root, clock_factory=clock_factory,
                monitor_clock_factory=monitor_clock_factory, allow_dirty=allow_dirty)
    expected = selection_lib.expected_trajectories(config, stage)
    missing = sorted(set(expected) - set(summaries))
    require(not missing,
            f"Stage {stage} declares {len(expected)} arm-seed trajectories and did not produce "
            f"{missing}. The grid is the protocol; a stage that ran part of it has not run.")
    return summaries


# ---------------------------------------------------------------------------
# stage: validate
# ---------------------------------------------------------------------------

def load_stage_trajectories(output, *, stage, identity=None, expected=None):
    """Every trajectory document of one stage, verified to belong to this campaign.

    ``expected`` is the declared arm x seed grid. A missing run fails here: a
    trajectory that stopped is evidence and is kept, while a trajectory that never
    ran is an absence of evidence, and a stage that cannot tell those apart can
    report "none eligible" for a grid it never fitted.
    """
    directory = Path(output) / f"stage{int(stage)}"
    trajectories, directories = {}, {}
    for run in sorted(p for p in directory.glob("*/trajectory.json")):
        document = load_json(run)
        name = run.parent.name
        require(document.get("schema_version") == trajectory_lib.TRAJECTORY_SCHEMA,
                f"{run} is not a guarded trajectory document")
        if identity is not None:
            run_identity = document.get("identity") or {}
            require(run_identity.get("revision") == identity["revision"],
                    f"{run} was produced by different revision code or config")
            differing = selection_lib.campaign_identity_differences(run_identity, identity)
            require(not differing,
                    f"{run} ran under a different campaign identity ({differing}); its parents, "
                    "pinned sources, device or batch size are not this campaign's")
            require(int(run_identity.get("stage", stage)) == int(stage),
                    f"{run} records stage {run_identity.get('stage')}, not stage {stage}")
        trajectories[name] = document
        directories[name] = run.parent
    if expected is not None:
        missing = sorted(set(expected) - set(trajectories))
        extra = sorted(set(trajectories) - set(expected))
        require(not missing,
                f"Stage {stage} declares {len(expected)} arm-seed trajectories and {missing} have "
                "no trajectory.json. A declared run that never ran is not an implicit failure; "
                "run it, or fix the grid.")
        require(not extra,
                f"These stage {stage} trajectories are not in the declared grid: {extra}")
    return trajectories, directories


def reached_checkpoints(output, *, stage, identity=None, expected=None):
    """Every budget checkpoint that exists, and every budget that was not reached."""
    trajectories, directories = load_stage_trajectories(output, stage=stage, identity=identity,
                                                        expected=expected)
    endpoints = []
    for name, document in sorted(trajectories.items()):
        for budget, record in sorted((document.get("budgets") or {}).items()):
            endpoints.append({"trajectory": name, "directory": directories[name],
                              "nominal_budget": float(budget), "record": record,
                              "reached": True})
        for budget, record in sorted((document.get("budgets_not_reached") or {}).items()):
            endpoints.append({"trajectory": name, "directory": directories[name],
                              "nominal_budget": float(budget), "record": record,
                              "reached": False})
    return trajectories, endpoints


def expected_parent_reference_identity(parent, context, *, root=ROOT):
    """The identity the gate's frozen reference MUST carry, rebuilt from first inputs.

    Rebuilt, not read: the selected parent named by the initial-SFT freeze (its
    checkpoint bytes re-hashed and its recorded state digest read out of the
    payload), this campaign's config hash, the scaffold prefix, and the **actual**
    ordered fixed validation pairs this run is being validated against. A reference
    whose sidecar claims another parent, another config or another pair order is
    then a mismatch rather than a claim nobody compared to anything.
    """
    path = Path(root) / parent["checkpoint"]
    require(path.is_file(), f"The selected parent checkpoint {path} is missing")
    digest = sha256(path)
    require(digest == parent["sha256"],
            f"{path} no longer hashes as the initial-SFT freeze recorded it ({digest} != "
            f"{parent['sha256']}); this is not the parent the gate's reference was built from")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_sha256")
    require(state,
            f"{path} records no state digest, so the reference identity cannot be bound to the "
            "weights that produced it")
    return guard.parent_reference_identity(
        parent_checkpoint_sha256=parent["sha256"], parent_state_sha256=state,
        config_sha256=parent["config_sha256"], scaffold_prefix=context["scaffold"].prefix,
        chosen_index=context["val_pairs"]["chosen_index"],
        rejected_index=context["val_pairs"]["rejected_index"])


def verified_parent_reference(directory, expected_identity):
    """The frozen parent reference this trajectory's gate used, arrays included.

    Both halves are required. The sidecar's identity must equal the block rebuilt
    from the expected parent and the actual pairs, and the vectors beside it must
    still digest to what that sidecar recorded -- ``load_parent_reference`` checks
    both. Reading the sidecar's claim and stopping there accepts a reference built
    from another parent, another pair order, or with no arrays behind it at all,
    and every gate verdict in the run is a comparison against those arrays.
    """
    path = Path(directory) / "parent_validation_reference.npz"
    sidecar = path.with_suffix(".json")
    require(sidecar.is_file(),
            f"{sidecar} is missing. Every gate verdict in this run was measured against that "
            "reference; without it the verdicts cannot be tied to the rows they scored.")
    require(path.is_file(),
            f"{path} is missing. The sidecar describes vectors that are not on disk, so no D in "
            "this run can be recomputed and none of its verdicts can be verified.")
    return guard.load_parent_reference(path, expected_identity)


def verified_gate_record(directory, budget_record, *, reference, gate):
    """The journalled gate verdict this checkpoint depends on, recomputed from its vectors.

    "A checkpoint exists, therefore its gate passed" is an inference from a file
    system, and a file system is not evidence about a measurement. So the verdict
    is read back from ``monitor.jsonl`` by check number and then **re-measured**:
    the per-check score vectors it wrote are found beside the run, re-hashed, and D
    is recomputed against the verified parent reference under the declared
    threshold (:func:`her2_guard.recompute_gate_verdict`). A verdict whose arrays
    are missing or moved, whose D is not the D those vectors give, whose threshold
    is not this campaign's, or which reads ``passed`` against its own numbers stops
    validation instead of being recertified.
    """
    path = Path(directory) / "monitor.jsonl"
    require(path.is_file(), f"{path} is missing; the gate verdicts are not on disk")
    check = budget_record.get("gate_check")
    require(check is not None,
            f"{path}: this budget record names no gate check, so nothing ties its checkpoint to a "
            "verdict")
    verdicts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_kind", "gate_verdict") != "gate_verdict":
            continue
        if record.get("check") == check:
            verdicts.append(record)
    require(len(verdicts) == 1,
            f"{path}: expected exactly one gate verdict for check {check}, found {len(verdicts)}")
    verdict = verdicts[0]
    require(verdict.get("passed") is True,
            f"{path}: check {check} did not pass, but a checkpoint was written for it")
    require(verdict.get("identity") == reference.identity,
            f"{path}: check {check} was measured against a different parent reference than the one "
            "verified in this run directory")
    require(int(verdict.get("update", -1)) == int(budget_record.get("updates", -1)),
            f"{path}: check {check} was taken at update {verdict.get('update')} and the budget "
            f"record claims {budget_record.get('updates')}")
    recomputed = guard.recompute_gate_verdict(directory, verdict, reference, gate=gate)
    require(recomputed["passed"] is True,
            f"{path}: recomputing check {check} from its retained vectors gives a stop at "
            f"D={recomputed['D']}; the checkpoint written for it is not a gate-passed endpoint")
    return dict(verdict, recomputed=recomputed)


def trajectory_gate_artifacts(directory):
    """Retain and verify the vectors behind every check, including a stopping check."""
    directory = Path(directory)
    reference_path = directory / "parent_validation_reference.npz"
    sidecar = reference_path.with_suffix(".json")
    reference = verified_parent_reference(directory, load_json(sidecar)["identity"])
    paths = {"parent_validation_reference.npz": reference_path,
             "parent_validation_reference.json": sidecar,
             "monitor.jsonl": directory / "monitor.jsonl"}
    checks = 0
    for line in paths["monitor.jsonl"].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        verdict = json.loads(line)
        if verdict.get("record_kind", "gate_verdict") != "gate_verdict":
            continue
        require(verdict.get("identity") == reference.identity,
                f"{directory}: a monitor verdict names a different parent reference")
        _, _, scores = guard.monitor_score_arrays(directory, verdict)
        path = Path(scores["path"])
        paths[f"monitor_scores::{path.name}"] = path
        checks += 1
    require(checks > 0, f"{directory}: no retained gate verdict exists")
    return {key: lineage.artifact_record(path, ROOT) for key, path in paths.items()}


def verify_budget_checkpoint(entry, *, root):
    """Re-hash the checkpoint bytes against what the trajectory recorded for them."""
    record = entry["record"]
    require(record.get("checkpoint") and record.get("checkpoint_sha256"),
            f"{entry['trajectory']} at {entry['nominal_budget']}s records no checkpoint path and "
            "hash; an unnamed file is not an endpoint")
    path = Path(root) / record["checkpoint"]
    require(path.is_file(), f"The checkpoint {path} named by the trajectory is missing")
    digest = sha256(path)
    require(digest == record["checkpoint_sha256"],
            f"{path} changed since the trajectory recorded it ({digest} != "
            f"{record['checkpoint_sha256']}). These are not the weights the gate passed.")
    return path, digest


def verify_checkpoint_payload(payload, *, trajectory, arm, seed, budget, record, identity, where):
    """What the checkpoint says it is, against what the endpoint claims it is.

    The byte hash proves the file has not changed. It does not say which run, which
    arm, which seed or which budget produced it, and a file whose hash matches its
    own new record is exactly what a substituted checkpoint looks like. The payload
    carries all of that, plus the progress record the loop wrote at the crossing,
    so it is checked here rather than trusted.
    """
    require(isinstance(payload, dict) and payload.get("state_sha256"),
            f"{where}: the checkpoint carries no state digest")
    require(payload.get("state_sha256") == record.get("state_sha256"),
            f"{where}: the checkpoint's state digest {payload.get('state_sha256')} is not the "
            f"{record.get('state_sha256')} the trajectory recorded when it wrote it")
    stored_arm = payload.get("arm") or {}
    require(stored_arm.get("arm_id") == arm["arm_id"],
            f"{where}: the checkpoint was written for arm {stored_arm.get('arm_id')!r}, not "
            f"{arm['arm_id']!r}")
    require(stored_arm.get("objective") == arm["objective"]
            and (stored_arm.get("coefficients") or {}) == (arm.get("coefficients") or {}),
            f"{where}: the checkpoint's objective and coefficients are not this arm's")
    require(int(payload.get("seed", -1)) == int(seed),
            f"{where}: the checkpoint was written at seed {payload.get('seed')}, not {seed}")
    require(abs(float(payload.get("budget_seconds", -1.0)) - float(budget)) <= 1e-9,
            f"{where}: the checkpoint was written at budget {payload.get('budget_seconds')} s, "
            f"not {budget} s")
    stored_identity = payload.get("identity") or {}
    differing = selection_lib.campaign_identity_differences(stored_identity, identity)
    require(not differing,
            f"{where}: the checkpoint was written under a different campaign identity "
            f"({differing}); its parents, pinned sources, device or batch size are not this "
            "campaign's")
    require(f"{stored_arm.get('arm_id')}_seed{int(payload.get('seed', -1))}" == trajectory,
            f"{where}: the checkpoint's own coordinates do not name trajectory {trajectory}")
    progress = payload.get("progress") or {}
    require(int(progress.get("updates", -1)) == int(record.get("updates", -2)),
            f"{where}: the checkpoint was written at {progress.get('updates')} completed updates "
            f"and the trajectory records {record.get('updates')} at this budget")
    require(int(progress.get("gate_check", -1)) == int(record.get("gate_check", -2)),
            f"{where}: the checkpoint names gate check {progress.get('gate_check')} and the "
            f"budget record names {record.get('gate_check')}")
    require(abs(float(progress.get("target_gpu_seconds", -1.0)) - float(budget)) <= 1e-9,
            f"{where}: the checkpoint's progress record targets "
            f"{progress.get('target_gpu_seconds')} s, not {budget} s")
    return {"arm_id": stored_arm.get("arm_id"), "seed": int(payload.get("seed")),
            "budget_seconds": float(payload.get("budget_seconds")),
            "state_sha256": payload.get("state_sha256"),
            "updates": int(progress.get("updates")),
            "gate_check": int(progress.get("gate_check")),
            "note": ("the checkpoint payload's own run, arm, seed, budget, state digest and "
                     "progress, checked against the endpoint they are claimed for")}


def validate_checkpoint(name, path, arm, seed, config, context, strata, output, raw_root, device,
                        parent_draws, expected_state_sha256=None):
    """Score, sample and diagnose one raw budget checkpoint; persist the evidence."""
    started = time.perf_counter()
    model = policy_lib.architecture_model(raw_root, device=device)
    restored = policy_lib.load_checkpoint(path, model, device=device)
    require(expected_state_sha256 is None
            or restored.get("state_sha256") == expected_state_sha256,
            f"{name}: the checkpoint loaded for scoring carries state digest "
            f"{restored.get('state_sha256')}, not the {expected_state_sha256} verified from its "
            "payload a moment ago")
    model.eval()
    policy = policy_lib.CorePolicy.from_prefix(model, context["scaffold"].prefix, context["vocab"],
                                              device=device)
    batch = config["inference"]["score_batch_size"]
    scored = policy.score(context["val_index"], batch_size=batch)
    k_values = tuple(config["evaluation"]["k_values"])
    metrics = evaluation.rank_metrics(scored["mean_log_probability"], context["val_positive"],
                                      context["val_cores"], k_values=k_values)
    per_stratum = evaluation.stratified_metrics(
        scored["mean_log_probability"], context["val_positive"], context["val_cores"], strata,
        k_values=(32,), categories=tuple(config["evaluation"]["train_distance_strata"]))
    pair_metrics = preferences.validation_pair_metrics(policy, context["val_pairs"],
                                                       batch_size=batch)
    score_path = Path(output) / "validation" / "validation_scores" / f"{name}.npy"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_path, scored["mean_log_probability"])

    settings = config["generation"]
    draw_seed = settings["seed_base"] + _draw_offset(name)
    index, sampled = policy.sample(settings["draws"], seed=draw_seed,
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
        split_of=context["split_of"], class_of=context["class_of"],
        max_train_distance=int(config["evaluation"]["max_train_distance"]))
    diagnostics.update(draw_seed=draw_seed, temperature=settings["temperature"],
                       sampler_scorer_parity=parity)
    samples = pd.DataFrame({"draw_index": np.arange(len(index)),
                            "core": data.decode_cores(index),
                            "sum_log_probability": rescored["sum_log_probability"],
                            "mean_log_probability": rescored["mean_log_probability"]})
    draw_path = Path(output) / "validation" / "generation" / f"draws_{name}.csv"
    draw_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(draw_path, index=False)
    diversity = evaluation.diversity_eligibility(
        diagnostics, training_reference=context["training_diversity_reference"],
        parent_reference=parent_draws["reference"], gates=config["diversity_gates"])
    record = {
        "schema_version": ENDPOINT_SCHEMA, "name": name, "arm_id": arm["arm_id"],
        "objective": arm["objective"], "coefficients": arm["coefficients"], "seed": int(seed),
        "checkpoint": relative_key(path, ROOT), "checkpoint_sha256": sha256(path),
        "val_metrics": metrics, "val_strata": per_stratum,
        "chosen_nll_per_residue": pair_metrics["chosen_nll_per_residue"],
        "rejected_nll_per_residue": pair_metrics["rejected_nll_per_residue"],
        "val_pair_metrics": pair_metrics,
        "val_scores": {"path": relative_key(score_path, ROOT), "sha256": sha256(score_path),
                       "rows": int(scored["mean_log_probability"].size),
                       "quantity": "mean log probability per residue, validation rows"},
        "generation": diagnostics,
        "generation_samples": {"path": relative_key(draw_path, ROOT), "sha256": sha256(draw_path),
                               "draws": int(len(index))},
        "diversity": diversity, "diversity_eligible": bool(diversity["eligible"]),
        "wall_seconds": time.perf_counter() - started,
    }
    del policy, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _draw_offset(name):
    """A deterministic, stage-independent draw-seed offset for one checkpoint name.

    ``hash()`` is salted per process, so it cannot appear anywhere near a seed:
    the same checkpoint would draw different sequences on two runs and the record
    could not be re-derived. This is a stable digest instead.
    """
    import hashlib
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % 100000


def parent_kl_pass(records, parents, config, context, raw_root, device):
    """Monte Carlo KL from each endpoint's own draws to its SFT parent, with the SE."""
    batch = config["inference"]["score_batch_size"]
    by_seed = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)
    for seed, rows in sorted(by_seed.items()):
        policy = load_parent_policy(parents[seed], raw_root, context, device)
        policy.model.eval()
        for record in rows:
            path = ROOT / record["generation_samples"]["path"]
            digest = sha256(path)
            require(digest == record["generation_samples"]["sha256"],
                    f"{record['name']}: the persisted draws changed since they were written")
            frame = pd.read_csv(path, dtype={"core": str})
            index = data.encode_cores(frame.core)
            reference = policy.score(index, batch_size=batch)["sum_log_probability"]
            record["parent_kl"] = dict(
                evaluation.monte_carlo_kl(frame.sum_log_probability.to_numpy(dtype=np.float64),
                                          reference),
                reference_name=parents[seed]["name"],
                reference_sha256=parents[seed]["sha256"], draws_sha256=digest)
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def require_stage_not_frozen(output, stage):
    """Validation refuses to rewrite an endpoint document that a freeze already hashed.

    The stage freeze records the sha256 of ``stageN_endpoints.json`` under
    ``validation::endpoints``, and every later stage re-hashes it before it will
    start. That document carries the validation's own wall clock -- the whole-run
    seconds, the parent-draw seconds and a per-endpoint ``wall_seconds`` -- so
    writing it a second time changes its hash even when every measured result is
    identical. The later stages would then refuse the freeze and say the artifact
    changed underneath them, which reads like tampering and is really a stopwatch.

    So this is refused here, at the top, rather than discovered afterwards: the
    expensive half of validation (the parent draws and the Monte Carlo KL) has not
    run yet, and the existing frozen evidence is still on disk. Deleting the marker
    is the deliberate way to redo a frozen stage.
    """
    marker = Path(output) / selection_lib.stage_marker_name(int(stage))
    endpoints = Path(output) / "validation" / f"stage{int(stage)}_endpoints.json"
    require(not marker.is_file(),
            f"Stage {stage} is already frozen by {marker}, and that freeze records the sha256 of "
            f"{endpoints}. Re-validating would rewrite that file with fresh timings, changing its "
            "hash without changing a single measured result, and every later stage would then "
            f"reject the freeze. Delete {marker} first if you really mean to re-validate this "
            "stage.")


def run_validate(config, context, output, identity, parents, *, stage, device, raw_root):
    """Every reached budget checkpoint, measured once, with its evidence persisted.

    Nothing is measured before its evidence is checked, and nothing is checked by
    reading a claim: the trajectory belongs to this campaign; the checkpoint bytes
    still hash to what the trajectory recorded **and** its payload names this arm,
    seed, budget, campaign identity, state digest and progress; the frozen parent
    reference matches the identity rebuilt from the selected parent, this config
    and the actual ordered pairs, and its vectors are loaded; and the gate verdict
    the checkpoint exists because of is recomputed from that check's retained score
    vectors against those parent vectors under the declared threshold. Only then is
    the checkpoint loaded and scored. The parent's own draws -- the reference every
    diversity verdict is relative to -- are persisted and hashed rather than
    discarded, and the endpoint set is checked to be exactly the reached one.
    """
    require_inspection(output, identity)
    require_stage_not_frozen(output, stage)
    started = time.perf_counter()
    expected = selection_lib.expected_trajectories(config, stage)
    trajectories, endpoints = reached_checkpoints(output, stage=stage, identity=identity,
                                                  expected=expected)
    require(trajectories, f"No stage {stage} trajectories to validate under {output}")
    marker = selection_lib.require_previous_stage(output, stage, identity=identity, root=ROOT)
    reused_controls, reuse_note = selection_lib.reused_control_endpoints(
        marker, root=ROOT, source_stage=int(stage) - 1)
    strata = validation_strata(config, context)
    arms = {arm["arm_id"]: arm for arm in selection_lib.declared_arms(config)}
    gate = guard.LikelihoodGate(
        threshold_nats_per_sequence=float(config["gate"]["threshold_nats_per_sequence"]))
    parent_draws, parent_identities, draw_wall = {}, {}, 0.0
    records = []
    for entry in endpoints:
        if not entry["reached"]:
            continue
        run = entry["trajectory"]
        arm_id, _, seed = run.rpartition("_seed")
        arm = arms[arm_id]
        seed = int(seed)
        name = checkpoint_name(arm, seed, entry["nominal_budget"])
        path, digest = verify_budget_checkpoint(entry, root=ROOT)
        if seed not in parent_identities:
            parent_identities[seed] = expected_parent_reference_identity(
                parents[seed], context, root=ROOT)
        reference = verified_parent_reference(entry["directory"], parent_identities[seed])
        verdict = verified_gate_record(entry["directory"], entry["record"], reference=reference,
                                       gate=gate)
        binding = verify_checkpoint_payload(
            torch.load(path, map_location="cpu", weights_only=True), trajectory=run, arm=arm,
            seed=seed, budget=float(entry["nominal_budget"]), record=entry["record"],
            identity=identity, where=name)
        if seed not in parent_draws:
            draw_started = time.perf_counter()
            parent_draws[seed] = _parent_draw_reference(parents[seed], config, context, raw_root,
                                                        device, output)
            draw_wall += time.perf_counter() - draw_started
        print(f"  validating {name}", flush=True)
        record = validate_checkpoint(name, path, arm, seed, config, context, strata, output,
                                     raw_root, device, parent_draws[seed],
                                     expected_state_sha256=binding["state_sha256"])
        require(record["checkpoint_sha256"] == digest,
                f"{name}: the checkpoint changed between verification and scoring")
        recomputed = verdict["recomputed"]
        record.update(nominal_budget=float(entry["nominal_budget"]), reached=True,
                      trajectory=run, gate_passed=True,
                      checkpoint_binding=binding,
                      gate_evidence={"check": verdict["check"], "update": verdict["update"],
                                     "D": recomputed["D"],
                                     "D_per_residue": recomputed["D_per_residue"],
                                     "journalled_D": verdict.get("D"),
                                     "threshold_nats_per_sequence":
                                         recomputed["threshold_nats_per_sequence"],
                                     "pairs": recomputed["pairs"],
                                     "scores_sha256": recomputed["scores"]["sha256"],
                                     "parent_reference_identity": dict(reference.identity),
                                     "parent_reference_pairs": reference.pairs,
                                     "recomputation_note": recomputed["note"]},
                      gate_note=("gate_passed is the journalled verdict for the check this "
                                 "checkpoint was written at, recomputed from that check's "
                                 "retained score vectors against the verified parent reference "
                                 "under the declared threshold; it is not inferred from the "
                                 "checkpoint existing and it is not read out of the journal"),
                      parent_draws={"name": parent_draws[seed]["name"],
                                    **parent_draws[seed]["draws"]},
                      actual_gpu_seconds=entry["record"].get("actual_gpu_seconds"),
                      updates=entry["record"].get("updates"),
                      exposures=entry["record"].get("exposures"),
                      distinct_exposures=entry["record"].get("distinct_exposures"))
        records.append(record)
    selection_lib.require_exact_endpoint_coverage(
        records, selection_lib.reached_endpoint_index(trajectories),
        where=f"stage {stage} validation")
    kl_started = time.perf_counter()
    parent_kl_pass(records, parents, config, context, raw_root, device)
    kl_wall = time.perf_counter() - kl_started
    document = {"schema_version": ENDPOINT_SCHEMA, "stage": int(stage),
                "identity": identity,
                "identity_note": ("the whole campaign identity, not the revision block alone: the "
                                  "freeze re-checks the parents, the pinned sources, the device "
                                  "and the batch size this validation ran under"),
                "endpoints": records,
                "parent_draw_references": {str(seed): dict(entry["draws"], name=entry["name"])
                                           for seed, entry in sorted(parent_draws.items())},
                "not_reached": [dict(entry["record"], trajectory=entry["trajectory"])
                                for entry in endpoints if not entry["reached"]],
                "declared_trajectories": expected,
                "reused_controls": reused_controls,
                "reused_control_note": reuse_note,
                "cost": {"validation_wall_seconds": time.perf_counter() - started,
                         "parent_draw_wall_seconds": draw_wall,
                         "parent_kl_wall_seconds": kl_wall,
                         "endpoint_wall_seconds": {record["name"]: record["wall_seconds"]
                                                   for record in records},
                         "endpoints_measured": len(records),
                         "reused_controls_measured": 0,
                         "note": ("the whole validation clock, including the parent draws and the "
                                  "Monte Carlo KL. Reused controls are NOT re-measured here and "
                                  "cost this stage nothing; their metrics come from the stage that "
                                  "fitted them.")},
                "strata": {"selection": list(selection_lib.SELECTION_STRATA),
                           "reported": list(config["evaluation"]["train_distance_strata"])}}
    path = Path(output) / "validation" / f"stage{int(stage)}_endpoints.json"
    save_json(path, document)
    print(f"validation written to {path}", flush=True)
    return document


def _sample_parent_draws(parent, config, context, raw_root, device):
    """Draw from the parent itself. The one expensive half of the diversity floor."""
    policy = load_parent_policy(parent, raw_root, context, device)
    policy.model.eval()
    settings = config["generation"]
    index, _ = policy.sample(settings["draws"], seed=settings["seed_base"] + _draw_offset(
        parent["name"]), temperature=settings["temperature"],
        batch_size=config["inference"]["sample_batch_size"])
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return index


def _parent_draw_reference(parent, config, context, raw_root, device, output):
    """The parent's own draws, for the relative diversity floor. One per seed, persisted.

    Every endpoint's diversity verdict is relative to these draws, so they are
    evidence and not scratch: they are written out, hashed and named in the
    validation document and the stage freeze. Discarding them left a published
    eligibility decision resting on numbers that existed only inside one process.
    """
    index = _sample_parent_draws(parent, config, context, raw_root, device)
    settings = config["generation"]
    draw_path = Path(output) / "validation" / "generation" / f"parent_draws_{parent['name']}.csv"
    draw_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"draw_index": np.arange(len(index)), "core": data.decode_cores(index)})
    frame.to_csv(draw_path, index=False)
    return {"reference": evaluation.diversity_reference(index), "name": parent["name"],
            "draws": {"path": relative_key(draw_path, ROOT), "sha256": sha256(draw_path),
                      "draws": int(len(index)),
                      "draw_seed": settings["seed_base"] + _draw_offset(parent["name"]),
                      "temperature": settings["temperature"],
                      "parent_sha256": parent["sha256"],
                      "note": ("the parent's own temperature-1 draws: the reference every "
                               "endpoint's relative diversity verdict was measured against")}}


# ---------------------------------------------------------------------------
# stage: freeze
# ---------------------------------------------------------------------------

def run_freeze(config, output, identity, *, stage):
    """Re-verify the bytes, select on validation only, and write the stage marker.

    The validated document is held to this whole campaign identity and to an exact
    one-to-one correspondence with what was actually reached: one endpoint per
    reached trajectory-budget, each naming the checkpoint that trajectory recorded,
    with no duplicate, no foreign checkpoint, no endpoint at an unreached budget
    and none missing. The parent draws every diversity verdict was measured against
    are re-hashed and named among the artifacts.

    Every objective the stage declares appears in the selection table, including
    one whose trajectories all stopped: "none eligible" is a result. The matched
    control is whatever the protocol says it is -- fitted here in stage 1, reused
    under hash verification in stages 2 and 3 -- and when no control endpoint
    exists at all, the marker carries the predecessor's own control status forward,
    stopped trajectories and stop reasons intact, rather than rebuilding it from a
    stage that fitted no control and reporting an empty table.
    """
    require_inspection(output, identity)
    endpoints_path = Path(output) / "validation" / f"stage{int(stage)}_endpoints.json"
    require(endpoints_path.is_file(),
            f"No validated endpoints at {endpoints_path}; run `--stage validate` first")
    validated = load_json(endpoints_path)
    require(validated.get("schema_version") == ENDPOINT_SCHEMA,
            f"{endpoints_path} is not a validated endpoint document")
    require(int(validated.get("stage", -1)) == int(stage),
            f"{endpoints_path} carries stage {validated.get('stage')}, not stage {stage}")
    require((validated.get("identity") or {}).get("revision") == identity["revision"],
            f"{endpoints_path} was produced by different revision code or config")
    differing = selection_lib.campaign_identity_differences(validated.get("identity"), identity)
    require(not differing,
            f"{endpoints_path} was produced under a different campaign identity ({differing}). "
            "The revision hashes agree, so this is a change of parents, pinned sources, device or "
            "batch size: these endpoints were not measured for this campaign.")
    expected = selection_lib.expected_trajectories(config, stage)
    trajectories, stage_endpoints = reached_checkpoints(output, stage=stage, identity=identity,
                                                        expected=expected)
    reached_index = selection_lib.reached_endpoint_index(trajectories)
    selection_lib.require_exact_endpoint_coverage(
        validated["endpoints"], reached_index, where=f"stage {stage} freeze ({endpoints_path})")
    for entry in stage_endpoints:
        if entry["reached"]:
            verify_budget_checkpoint(entry, root=ROOT)
    marker = selection_lib.require_previous_stage(output, stage, identity=identity, root=ROOT)
    reused_controls, reuse_note = selection_lib.reused_control_endpoints(
        marker, root=ROOT, source_stage=int(stage) - 1)
    artifacts, control_artifacts = {}, {}
    for record in validated["endpoints"]:
        require(record["trajectory"] in trajectories,
                f"{endpoints_path} carries an endpoint for {record['trajectory']}, which is not a "
                f"stage {stage} trajectory")
        directory = Path(output) / f"stage{int(stage)}" / record["trajectory"]
        evidence = record.get("gate_evidence") or {}
        require(evidence.get("parent_reference_identity"),
                f"{record['name']}: validation retained no parent reference identity")
        reference = verified_parent_reference(directory, evidence["parent_reference_identity"])
        verdict = verified_gate_record(
            directory, reached_index[(record["trajectory"], float(record["nominal_budget"]))],
            reference=reference, gate=guard.LikelihoodGate(
                threshold_nats_per_sequence=config["gate"]["threshold_nats_per_sequence"]))
        require(record.get("gate_passed") is True
                and evidence.get("check") == verdict["check"]
                and evidence.get("update") == verdict["update"]
                and evidence.get("D") is not None
                and abs(float(evidence["D"]) - verdict["recomputed"]["D"])
                    <= guard.GATE_RECOMPUTATION_ATOL,
                f"{record['name']}: validation's gate evidence differs from its retained vectors")
        named = {"checkpoint": {"path": record["checkpoint"],
                                "sha256": record["checkpoint_sha256"]},
                 "val_scores": record["val_scores"],
                 "generation_samples": record["generation_samples"]}
        for key, entry in named.items():
            artifacts[f"{record['name']}::{key}"] = {
                "path": entry["path"], "sha256": entry["sha256"],
                "bytes": int((ROOT / entry["path"]).stat().st_size)}
        if record["objective"] == "continued_sft":
            control_artifacts[record["name"]] = artifacts[f"{record['name']}::checkpoint"]
    artifacts["validation::endpoints"] = lineage.artifact_record(endpoints_path, ROOT)
    for seed, entry in sorted((validated.get("parent_draw_references") or {}).items()):
        draw_path = ROOT / entry["path"]
        require(draw_path.is_file(),
                f"The parent draw reference {draw_path} the validation measured diversity against "
                "is missing; the eligibility verdicts rest on it")
        digest = sha256(draw_path)
        require(digest == entry["sha256"],
                f"{draw_path} changed since validation drew it ({digest} != {entry['sha256']})")
        artifacts[f"parent_draws::seed{seed}"] = {
            "path": entry["path"], "sha256": digest,
            "bytes": int(draw_path.stat().st_size)}
    for name, document in sorted(trajectories.items()):
        path = Path(output) / f"stage{int(stage)}" / name / "trajectory.json"
        artifacts[f"{name}::trajectory"] = lineage.artifact_record(path, ROOT)
        for key, entry in trajectory_gate_artifacts(path.parent).items():
            artifacts[f"{name}::{key}"] = entry
        for journal in ("updates.jsonl", "monitor.jsonl", "budgets.jsonl"):
            journal_path = path.parent / journal
            if journal_path.is_file():
                artifacts[f"{name}::{journal}"] = lineage.artifact_record(journal_path, ROOT)
    if marker is not None:
        # Later stages retain the original control's scores, journals and parent
        # reference as well as its weights, including the all-stopped case.
        for key, entry in marker["artifacts"].items():
            if (key.startswith("inherited_control::") or "continued_sft" in key
                    or key.startswith("parent_draws::") or key == "validation::endpoints"):
                inherited_key = (key if key.startswith("inherited_control::") else
                                 f"inherited_control::stage{marker['stage']}::{key}")
                artifacts[inherited_key] = entry
    rows = _selection_records(validated["endpoints"])
    control_rows = [row for row in rows if row["objective"] == "continued_sft"]
    if reused_controls:
        require(not control_rows,
                f"Stage {stage} both fitted a continued_sft endpoint and inherited one. The "
                "control is fitted once; one of these is a double charge.")
        control_rows = list(reused_controls)
        control_artifacts = dict(marker["control_artifacts"])
        rows = rows + control_rows
    declared = selection_lib.declared_arms(config)
    objectives_declared = sorted({arm["objective"] for arm in declared
                                  if arm["stage"] == int(stage)})
    control_trajectories = {name: entry for name, entry in trajectories.items()
                            if name.startswith("continued_sft_seed")}
    if control_trajectories:
        control_status = selection_lib.control_status_document(
            control_trajectories=control_trajectories, control_endpoints=control_rows,
            source_stage=int(stage))
    elif marker is not None:
        # This stage fitted no control, so there is no local trajectory to describe
        # one from. The predecessor's status travels forward whole -- including the
        # stopped control trajectories and their reasons, which rebuilding from this
        # stage's own control-free grid silently dropped by stage 3.
        control_status = selection_lib.inherited_control_status(
            marker, control_endpoints=control_rows, stage=int(stage))
    else:
        control_status = selection_lib.control_status_document(
            control_trajectories={}, control_endpoints=control_rows, source_stage=int(stage))
    selection = selection_lib.select_all(
        rows, objectives=objectives_declared,
        budgets=[float(b) for b in config["budgets_gpu_seconds"]], seeds=config["seeds"])
    document = selection_lib.guarded_freeze_document(
        stage=stage, identity=identity, config=config, trajectories=trajectories,
        endpoints=rows, selection=selection,
        artifacts=artifacts, control_artifacts=control_artifacts,
        allocation=selection_lib.stage_allocation(config), root=ROOT,
        objectives=objectives_declared, control_status=control_status,
        control_endpoints=control_rows, controls=control_rows,
        git_commit=None)
    document["reused_controls"] = reuse_note
    document["stage_cost"] = stage_cost_summary(output, stage=stage,
                                                trajectories=trajectories,
                                                validated=validated,
                                                reused_controls=bool(reused_controls))
    path = Path(output) / selection_lib.stage_marker_name(stage)
    if path.is_file():
        existing = load_json(path)
        require(existing["selection"] == document["selection"]
                and existing["artifacts"] == document["artifacts"],
                f"A different stage {stage} freeze already exists at {path}; move it aside "
                "deliberately rather than overwriting evidence a later stage may have consumed")
        print(f"stage {stage} already frozen at {path}", flush=True)
        return existing
    save_json(path, document)
    print(f"stage {stage} frozen at {path}", flush=True)
    for budget, table in sorted(selection.items()):
        for objective, result in sorted(table.items()):
            print(f"  {budget}s {objective}: {result['selected'] or result.get('reason')}",
                  flush=True)
    return document


def stage_cost_summary(output, *, stage, trajectories, validated, reused_controls):
    """What this stage actually spent, beside the ceiling it declared.

    The allocation arithmetic is a ceiling on charged training seconds. This is
    the measurement: charged training time, the monitoring that ran on top of it,
    the I/O, the validation clock, and the updates and exposures that were really
    bought. A control fitted in stage 1 and reused here is named as reused and
    contributes nothing to this stage's charge -- counting it twice is exactly the
    arithmetic the allocation basis exists to prevent.
    """
    directory = Path(output) / f"stage{int(stage)}"
    per_run, missing, unavailable = {}, [], {}

    def measured(source, key, *, run, field):
        """A measurement or an explicit absence. Never a silent zero."""
        value = (source or {}).get(key)
        if value is None:
            unavailable.setdefault(field, []).append(run)
            return 0.0
        return value

    for name in sorted(trajectories):
        path = directory / name / "summary.json"
        if not path.is_file():
            missing.append(name)
            continue
        summary = load_json(path)
        cost = summary.get("cost") or {}
        excluded = summary.get("excluded_costs") or {}
        reference = summary.get("training_reference") or {}
        exposures = summary.get("exposures") or {}
        budget_checkpoint_wall = sum(
            float(record.get("checkpoint_wall_seconds") or 0.0)
            for record in (summary.get("budgets") or {}).values())
        row = {
            "charged_training_gpu_seconds": measured(cost, "training_gpu_seconds", run=name,
                                                     field="charged_training_gpu_seconds"),
            "failed_work_gpu_seconds": measured(cost, "failed_work_gpu_seconds", run=name,
                                                field="failed_work_gpu_seconds"),
            "monitor_gpu_seconds": measured(excluded, "monitor_gpu_seconds", run=name,
                                            field="monitor_gpu_seconds"),
            "monitor_wall_seconds": measured(excluded, "monitor_wall_seconds", run=name,
                                             field="monitor_wall_seconds"),
            # The parent's own validation scoring happens once per trajectory, before
            # the first update. It is real monitoring work and belongs in the stage's
            # monitoring total rather than in the residue between the clocks.
            "parent_reference_gpu_seconds": measured(
                excluded, "parent_validation_reference_gpu_seconds", run=name,
                field="parent_reference_gpu_seconds"),
            "parent_reference_wall_seconds": measured(
                excluded, "parent_validation_reference_wall_seconds", run=name,
                field="parent_reference_wall_seconds"),
            "rolling_checkpoint_wall_seconds": measured(cost, "checkpoint_wall_seconds", run=name,
                                                        field="rolling_checkpoint_wall_seconds"),
            "budget_checkpoint_wall_seconds": budget_checkpoint_wall,
            "budget_evaluation_wall_seconds": measured(cost, "budget_evaluation_wall_seconds",
                                                       run=name,
                                                       field="budget_evaluation_wall_seconds"),
            "diagnostic_wall_seconds": measured(cost, "diagnostic_wall_seconds", run=name,
                                                field="diagnostic_wall_seconds"),
            "reference_cold_start_charged_gpu_seconds": float(
                reference.get("cold_start_charged_gpu_seconds") or 0.0),
            "reference_warm_reuse_wall_seconds": float(
                reference.get("warm_reuse_wall_seconds") or 0.0),
            "whole_run_wall_seconds": measured(summary, "whole_run_wall_seconds", run=name,
                                               field="whole_run_wall_seconds"),
            "updates": measured(summary, "updates", run=name, field="updates"),
            "attempted_updates": measured(summary, "attempted_updates", run=name,
                                          field="attempted_updates"),
            "failed_attempts": measured(summary, "failed_attempts", run=name,
                                        field="failed_attempts"),
            "monitor_checks": measured(excluded, "monitor_checks", run=name,
                                       field="monitor_checks"),
            "sequence_exposures": int(exposures.get("sequences") or 0),
            "pair_exposures": int(exposures.get("pairs") or 0),
            "core_token_exposures": int(summary.get("core_token_exposures") or 0),
        }
        row["status"] = summary.get("status")
        row["training_reference_physically_reused"] = reference.get("physically_reused")
        per_run[name] = row
    totals = {}
    for row in per_run.values():
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    totals.setdefault("charged_training_gpu_seconds", 0.0)
    document = {"per_trajectory": per_run, "totals": totals,
                "summaries_missing": missing,
                "unavailable_measurements": {field: sorted(runs)
                                             for field, runs in sorted(unavailable.items())},
                "validation": (validated.get("cost") or {}),
                "control_reused_from_earlier_stage": bool(reused_controls),
                "accounting_note": (
                    "monitoring totals include the once-per-trajectory parent validation "
                    "reference scoring; I/O totals separate the rolling last-passing saves from "
                    "the nominal budget checkpoint writes; exposures are the sequences, pairs and "
                    "core tokens the stage really trained on; and the reference cold start that "
                    "was CHARGED to the training clock is reported apart from the wall time the "
                    "warm reuse physically took."),
                "note": ("measured, not declared. Monitoring, validation, generation and I/O are "
                         "additional to the charged training seconds and are never netted against "
                         "them.")}
    if unavailable:
        document["unavailable_note"] = (
            "these measurements are absent from the trajectory summaries named beside them. They "
            "contribute nothing to the totals and are reported missing rather than counted as "
            "zero, which would read as 'it cost nothing'.")
    if reused_controls:
        document["reuse_note"] = (
            "the continued_sft control was fitted and charged in the earlier stage that produced "
            "it. Its seconds are not added here, and its endpoints were not re-measured.")
    if missing:
        document["missing_note"] = (
            "these trajectories have no summary.json, so their measured cost is unavailable and "
            "is reported missing rather than estimated")
    return document


def _selection_records(endpoints):
    """The exactly-validation view the selector is allowed to see."""
    rows = []
    for record in endpoints:
        rows.append({"arm_id": record["arm_id"], "objective": record["objective"],
                     "coefficients": record["coefficients"], "seed": int(record["seed"]),
                     "nominal_budget": float(record["nominal_budget"]),
                     "reached": bool(record["reached"]),
                     "gate_passed": bool(record["gate_passed"]),
                     "gate_evidence": record.get("gate_evidence"),
                     "diversity_eligible": bool(record["diversity_eligible"]),
                     "val_strata": record["val_strata"],
                     "val_average_precision": record["val_metrics"]["average_precision"],
                     "val_auroc": record["val_metrics"]["auroc"],
                     "chosen_nll_per_residue": record["chosen_nll_per_residue"],
                     "rejected_nll_per_residue": record["rejected_nll_per_residue"],
                     "parent_kl": record.get("parent_kl"),
                     "diversity": record["diversity"]})
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def require_production_device(device, stages):
    """A continuation fits under a measured GPU budget or it does not run.

    ``--allow-cpu`` exists for inspection and planning, which perform no update.
    It must not reach ``continue``: the budget is defined as measured device
    elapsed time, ``GpuBudgetClock`` times with CUDA events whenever a CUDA device
    exists on the host, and pointing it at CPU work therefore produces a number
    that is neither CPU time nor GPU time while every artifact calls it a GPU
    budget. Unit tests inject an explicit clock instead, which is honest because
    the injected clock is not pretending to measure anything.
    """
    if "continue" not in set(stages):
        return device
    require(device == "cuda",
            f"The guarded continuation fits on CUDA only; this run resolved device {device!r}. "
            "The declared budgets are measured GPU seconds, so a CPU run would report a "
            "pseudo-GPU budget. --allow-cpu is for inspect/plan, which perform no update.")
    return device


def resolve_runtime(config, args):
    """Device and batch size, frozen by the campaign that produced the parents."""
    if args.allow_cpu:
        return "cpu", int(config["fallback_batch_sequences"])
    preflight_path = Path(args.preflight)
    require(preflight_path.is_file(),
            f"No preflight decision at {preflight_path}. The batch size was chosen once, before "
            "any fitting, and this continuation must not re-resolve it. Pass --allow-cpu only for "
            "synthetic runs.")
    preflight = load_json(preflight_path)
    original = json.loads((ROOT / config["inherited"]["original_config"]).read_text(
        encoding="utf-8"))
    policy_lib.recheck_preflight(original["preflight"], preflight, allow_cpu=False)
    batch = (int(config["batch_sequences"]) if preflight["batch_size"] == 128
             else int(config["fallback_batch_sequences"]))
    return preflight["device"], batch


def run(config_path, output, *, stages, stage, args):
    config = load_config(config_path)
    raw_root = ROOT / config["raw_root"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    set_cpu_threads(config["runtime"]["cpu_threads"])
    problems = data.verify_all_sources(raw_root, ROOT, config["source_manifests"])
    require(not problems, f"Pinned source files failed verification: {problems}")
    device, batch_sequences = resolve_runtime(config, args)
    require_production_device(device, stages)
    identity, parents = build_identity(config, config_path, raw_root=raw_root, device=device,
                                       batch_sequences=batch_sequences)
    original_config_sha = identity["inherited"]["original_config_sha256"]
    for seed, entry in parents.items():
        entry["config_sha256"] = identity["revision"]["config_sha256"]
        entry["original_config_sha256"] = original_config_sha
        entry["original_reference_cache"] = str(
            ROOT / config["inherited"]["original_continuation_root"]
            / f"dpo_seed{seed}" / "reference_cache.npy")

    needs_data = bool({"continue", "validate"} & set(stages)) or (
        "plan" in stages and args.measure_monitor_check)
    context = build_context(config, raw_root) if needs_data else None

    if "inspect" in stages:
        run_inspect(config, output, identity)
    if "plan" in stages:
        run_plan(config, output, identity, measure_monitor_check=args.measure_monitor_check,
                 raw_root=raw_root, device=device, parents=parents, context=context)
    if "continue" in stages:
        if not args.allow_dirty:
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                            text=True).strip()
            require(not dirty, "Commit the protocol, config and code before fitting")
        require_production_device(device, ("continue",))
        run_continue(config, context, output, identity, parents, stage=stage, device=device,
                     batch_sequences=batch_sequences, raw_root=raw_root,
                     clock_factory=lambda: GpuBudgetClock(device="cuda"),
                     monitor_clock_factory=lambda: GpuBudgetClock(device="cuda"),
                     allow_dirty=args.allow_dirty)
    if "validate" in stages:
        run_validate(config, context, output, identity, parents, stage=stage, device=device,
                     raw_root=raw_root)
    if "freeze" in stages:
        run_freeze(config, output, identity, stage=stage)
    print("guarded continuation stage complete", flush=True)
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiments/her2_guarded_continuation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/claude_codex_her2_guarded_20260918/run")
    parser.add_argument("--preflight", type=Path,
                        default=ROOT / "outputs/her2_posttrain_20260918/preflight.json")
    parser.add_argument("--stages", default="inspect,plan")
    parser.add_argument("--stage", type=int, default=1, help="which declared grid stage to run")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--measure-monitor-check", action="store_true",
                        help="perform ONE real parent-vs-parent gate check and report its measured "
                             "GPU and wall cost; performs no update")
    args = parser.parse_args()
    chosen = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = sorted(set(chosen) - set(STAGES))
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    run(args.config, args.output, stages=chosen, stage=args.stage, args=args)
