#!/usr/bin/env python
"""Fit the HER2 post-training campaign: two policy arms x three seeds, plus baselines.

Order of work, all of it declared in the config before anything is fitted:

1. Verify every pinned source hash, then resolve the batch size ONCE from the
   driver's free-VRAM reading and freeze it for every arm and seed.
2. Six policy runs -- ``sft`` (pinned p-IgGen) and ``scratch`` (same architecture,
   random init) at seeds 20260918/19/20 -- five full passes over all 120,504
   training high-bin rows, checkpoints after passes 1, 3 and 5.
3. Three CNN seeds over all 367,042 labelled rows and one additive linear model.
4. Freeze the initial selection: the checkpoint with the lowest **validation
   positive NLL**, written with every artifact's hash into ``base_selection.json``.

Selection reads validation only. Nothing here opens the test split or the assay
workbook. ``base_selection.json`` is stage-marked ``initial_sft_selection``: it is
what ``scripts/posttrain_her2.py`` clones its parents from, and it deliberately
cannot unlock reserved labels. Only the final freeze, written after every
continuation budget has been evaluated, can do that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smallAntibodyGen.experiments import her2_baselines as baselines  # noqa: E402
from smallAntibodyGen.experiments import her2_data as data  # noqa: E402
from smallAntibodyGen.experiments import her2_eval as evaluation  # noqa: E402
from smallAntibodyGen.experiments import her2_policy as policy_lib  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import (  # noqa: E402
    RunLedger, code_digests, digest_document, load_json, relative_key, require, save_json,
    set_cpu_threads, sha256, torch_runtime)

STAGES = ("policies", "classifiers", "linear", "select")
DECLARED = {"arms": ["sft", "scratch"], "seeds": [20260918, 20260919, 20260920], "epochs": 5,
            "checkpoints": [1, 3, 5], "batch_size": 128, "fallback_batch_size": 64}
#: The intermediate freeze. Named apart from ``selection_frozen.json`` so that no
#: reader can mistake the file that starts post-training for the one that unlocks
#: reserved labels.
BASE_SELECTION = "base_selection.json"


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_digests(config, raw_root):
    """Hashes of everything the fit consumes; part of every run's identity."""
    return data.source_digests(ROOT, raw_root, config["source_manifests"])


def fitting_has_started(output):
    """True once any run directory exists. The batch size freezes at that moment."""
    return any(path.is_file() for path in Path(output).glob("*/run.json"))


def resolve_preflight(config, output, *, allow_cpu):
    """Resolve the batch size once, then RE-CHECK the device on every later call.

    Two arms and three seeds are only comparable at one batch size, so the choice
    is made before the first fit and is not revisited. In particular, a *larger*
    free-VRAM reading on a later day is not a reason to move a campaign from 64 to
    128: that would silently change what the earlier checkpoints are being compared
    against.

    What the earlier version got wrong is the other half: once fitting had started
    it returned the stored decision without looking at the device at all, so a
    resumed run reported another day's free-VRAM reading as if it were current. The
    decision is reused; the reading is taken fresh every time, and falling below
    the floor the frozen batch was allowed to run at is a failure to report rather
    than a parameter to adjust.
    """
    path = Path(output) / "preflight.json"
    if path.is_file():
        previous = load_json(path)
        observation = policy_lib.recheck_preflight(config["preflight"], previous,
                                                   allow_cpu=allow_cpu)
        state = "fitting has already started" if fitting_has_started(output) else "no fit yet"
        print(f"preflight frozen at batch {previous['batch_size']} ({previous['decision']}; "
              f"{state}); device re-checked now: "
              f"{observation['current_free_vram_mib']} MiB free", flush=True)
        return dict(previous, current_check=observation)
    decision = policy_lib.preflight(config["preflight"], allow_cpu=allow_cpu)
    decision["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    decision["runtime"] = torch_runtime()
    save_json(path, decision)
    return decision


def _gradient_snapshot(model, label):
    """Every parameter gradient, checked for existence, finiteness and signal.

    Taken **before** any ``zero_grad``. The earlier version asked whether all
    gradients were finite after clearing them, which is a vacuous ``all()`` over an
    empty generator and answers True no matter what happened. It also checks that
    the gradients are not identically zero: a comparison of two all-zero gradient
    sets agrees perfectly and proves nothing about the two code paths.
    """
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters() if parameter.grad is not None}
    require(gradients, f"{label}: the backward pass produced no gradients at all")
    nonfinite = sorted(name for name, value in gradients.items()
                       if not bool(torch.isfinite(value).all()))
    require(not nonfinite, f"{label}: nonfinite gradient in {nonfinite[:5]}")
    norms = {name: float(value.norm()) for name, value in gradients.items()}
    require(max(norms.values()) > 0.0, f"{label}: every gradient is exactly zero")
    return gradients, norms


def parity_probe(policy, index, tolerance):
    """Cached shared-prefix path vs ordinary full teacher forcing, incl. gradients.

    Run on real weights at run start and again after the last optimizer step, so
    "parity holds at initialization" cannot stand in for "parity holds while
    training".
    """
    model = policy.model
    model.zero_grad(set_to_none=True)
    core_ids = policy.token_ids(index)
    full = policy.full_logits(core_ids)
    target = torch.as_tensor(np.asarray(index), dtype=torch.long, device=policy.device)
    full_loss = torch.nn.functional.cross_entropy(full.reshape(-1, full.shape[-1]),
                                                  target.reshape(-1))
    full_loss.backward()
    reference, reference_norms = _gradient_snapshot(model, "full teacher forcing")
    model.zero_grad(set_to_none=True)
    cached = policy.core_logits(core_ids)
    cached_loss = torch.nn.functional.cross_entropy(cached.reshape(-1, cached.shape[-1]),
                                                    target.reshape(-1))
    cached_loss.backward()
    gradients, cached_norms = _gradient_snapshot(model, "shared-prefix cache")
    require(set(gradients) == set(reference),
            "The two paths produced gradients for different parameter sets: "
            f"{sorted(set(gradients) ^ set(reference))[:5]}")
    errors = {name: float((gradients[name] - reference[name]).abs().max()) for name in gradients}
    report = {"logits_max_abs": float((full - cached).abs().max().detach()),
              "loss_abs": float(abs(full_loss - cached_loss).detach()),
              "gradient_max_abs": max(errors.values()),
              "largest_gradient_error_parameter": max(errors, key=errors.get),
              "parameters_with_gradient": len(errors),
              "parameters_in_model": sum(1 for _ in model.parameters()),
              "min_gradient_norm_full": min(reference_norms.values()),
              "max_gradient_norm_full": max(reference_norms.values()),
              "max_gradient_norm_cached": max(cached_norms.values()),
              # Derived, not asserted: a NaN or Inf anywhere in a gradient makes its
              # norm nonfinite, and `_gradient_snapshot` raises before we get here.
              # Writing a literal True would keep claiming it if that check moved.
              "all_gradients_finite": bool(np.isfinite(list(reference_norms.values())
                                                       + list(cached_norms.values())).all()),
              "all_gradients_nonzero": bool(min(reference_norms.values()) > 0.0
                                            and min(cached_norms.values()) > 0.0),
              "tolerance": tolerance}
    model.zero_grad(set_to_none=True)
    require(report["logits_max_abs"] < tolerance and report["gradient_max_abs"] < tolerance,
            f"Shared-prefix parity failed: {report}")
    return report


def policy_checkpoint_evaluation(policy, config, context, epoch, directory, identity):
    """Save the checkpoint, then measure validation NLL and the disclosed diagnostics."""
    path = Path(directory) / f"epoch_{epoch}.pt"
    digest = policy_lib.save_checkpoint(path, policy, {"epoch": epoch, "identity": identity})
    batch = config["inference"]["score_batch_size"]
    positive = policy.score(context["val_positive_index"], batch_size=batch)
    everything = policy.score(context["val_index"], batch_size=batch)
    metrics = evaluation.rank_metrics(everything["mean_log_probability"], context["val_positive"],
                                      context["val_cores"], k_values=tuple(config["evaluation"]["k_values"]))
    return {"checkpoint": relative_key(path, ROOT), "checkpoint_sha256": sha256(path),
            "state_sha256": digest,
            "val_positive_nll_per_residue": float(-positive["mean_log_probability"].mean()),
            "val_positive_nll_sum": float(-positive["sum_log_probability"].mean()),
            "val_positive_rows": int(len(context["val_positive_index"])),
            "val_ranking_diagnostic": metrics,
            "selection_note": ("selection uses val_positive_nll_per_residue only; the ranking "
                               "metrics above are disclosed diagnostics and are NOT used to "
                               "choose a checkpoint")}


def run_policy(arm, seed, config, context, output, *, device, batch_size, identity_base,
               discard_incomplete):
    directory = Path(output) / f"policy_{arm}_seed{seed}"
    identity = dict(identity_base, arm=arm, seed=seed, batch_size=batch_size,
                    epochs=config["policy"]["epochs"])
    ledger = RunLedger(directory, identity, metadata={"first_claimed_git_commit": git_commit()})
    if ledger.start(discard_incomplete=discard_incomplete) == "completed":
        print(f"skipping completed run {directory.name}", flush=True)
        return load_json(ledger.path)["summary"]
    run_started = time.perf_counter()
    raw_root = ROOT / config["raw_root"]
    if arm == "sft":
        model = policy_lib.load_pinned_model(raw_root, device=device)
    else:
        model = policy_lib.random_init_model(raw_root, seed=seed, device=device)
    policy = policy_lib.CorePolicy.from_prefix(model, context["scaffold"].prefix, context["vocab"],
                                               device=device)
    tolerance = config["tolerances"]["cached_vs_full_parity"]
    probe_rows = context["train_positive_index"][:8]
    summary = {"arm": arm, "seed": seed, "identity": identity,
               "initial_state_sha256": policy_lib.state_digest(model),
               "parity_at_initialization": parity_probe(policy, probe_rows, tolerance),
               "noncanonical_mass_mean": float(policy.noncanonical_mass(probe_rows).mean())}
    plan = policy_lib.TrainingPlan(rows=int(context["train_positive_index"].shape[0]),
                                   batch_size=batch_size, epochs=config["policy"]["epochs"],
                                   checkpoints=tuple(config["policy"]["checkpoints"]))
    summary["plan"] = plan.document()
    save_json(directory / "summary.json", summary)
    # The step-time budget is per batch size, because the probe measured it per batch size.
    spill = dict(config["preflight"],
                 max_median_step_seconds=config["preflight"]["max_median_step_seconds"].get(
                     str(batch_size)))
    started = time.perf_counter()
    result = policy_lib.train_policy(
        policy, context["train_positive_index"], plan, config["policy"]["optimization"],
        seed=seed, directory=directory,
        on_checkpoint=lambda p, epoch, record: policy_checkpoint_evaluation(
            p, config, context, epoch, directory, identity),
        progress_every=config["inference"]["progress_every"], spill=spill)
    summary.update(result)
    summary["parity_after_training"] = parity_probe(policy, probe_rows, tolerance)
    summary["final_state_sha256"] = policy_lib.state_digest(model)
    summary["training_and_final_probe_wall_seconds"] = time.perf_counter() - started
    summary["wall_clock_seconds"] = time.perf_counter() - run_started
    if torch.cuda.is_available():
        summary["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        torch.cuda.reset_peak_memory_stats()
    save_json(directory / "summary.json", summary)
    ledger.complete(summary)
    del model, policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_classifiers(config, context, output, *, device, identity_base, discard_incomplete):
    summaries = {}
    for seed in config["classifier"]["seeds"]:
        directory = Path(output) / f"cnn_seed{seed}"
        identity = dict(identity_base, model="cnn_3class", seed=seed,
                        settings=config["classifier"])
        ledger = RunLedger(directory, identity)
        if ledger.start(discard_incomplete=discard_incomplete) == "completed":
            print(f"skipping completed run {directory.name}", flush=True)
            summaries[str(seed)] = load_json(ledger.path)["summary"]
            continue
        model, report = baselines.train_cnn(
            context["train_index"], context["train_targets"], context["val_index"],
            context["val_targets"], config["classifier"], seed=seed, device=device,
            directory=directory)
        path = directory / "cnn.pt"
        torch.save({"schema_version": "her2-cnn/1", "state": model.state_dict(),
                    "identity": identity, "report": report}, path)
        report["checkpoint"] = relative_key(path, ROOT)
        report["checkpoint_sha256"] = sha256(path)
        summaries[str(seed)] = report
        save_json(directory / "summary.json", report)
        ledger.complete(report)
    return summaries


def run_linear(config, context, output, *, identity_base, discard_incomplete):
    directory = Path(output) / "linear_3class"
    identity = dict(identity_base, model="additive_linear_3class", settings=config["linear"])
    ledger = RunLedger(directory, identity)
    if ledger.start(discard_incomplete=discard_incomplete) == "completed":
        print(f"skipping completed run {directory.name}", flush=True)
        return load_json(ledger.path)["summary"]
    model, report = baselines.fit_additive_linear(
        context["train_index"], context["train_targets"], ridge=config["linear"]["ridge"],
        max_iterations=config["linear"]["max_iterations"])
    with torch.no_grad():
        logits = model.logits(context["val_index"])
        report["val_cross_entropy"] = float(torch.nn.functional.cross_entropy(
            logits, torch.as_tensor(context["val_targets"], dtype=torch.long)))
    path = directory / "linear.pt"
    torch.save({"schema_version": "her2-linear/1", "weight": model.weight, "bias": model.bias,
                "identity": identity, "report": report}, path)
    report["checkpoint"] = relative_key(path, ROOT)
    report["checkpoint_sha256"] = sha256(path)
    save_json(directory / "summary.json", report)
    ledger.complete(report)
    return report


def shared_initial_cost(results):
    """What the initial SFT cost, per policy run, in units that say what they are.

    This cost is paid **once** and is common to both continuations: DPO and
    continued SFT start from the same parent, so it belongs to neither arm's
    budget and is reported here instead of being folded into either. The numbers
    are wall seconds of the whole run on the recorded device, which is a different
    quantity from the continuation budgets (device-elapsed seconds of training
    updates only) and is labelled as such rather than compared with them.
    """
    document = {"unit": "wall seconds of the whole run, not device-elapsed training seconds",
                "device": (results.get("preflight") or {}).get("device"),
                "note": ("paid once per seed before either continuation; common to DPO and "
                         "continued SFT, and therefore charged to neither budget"),
                "runs": {}}
    for key, summary in results.get("policies", {}).items():
        document["runs"][key] = {
            "wall_clock_seconds": summary.get("wall_clock_seconds"),
            "total_steps": summary.get("total_steps"),
            "total_exposures": summary.get("total_exposures"),
            "mean_step_seconds": summary.get("mean_step_seconds"),
            "batch_size": (summary.get("identity") or {}).get("preflight_batch_size")}
    return document


def freeze_selection(config, results, output):
    """Pick by validation positive NLL and write the INTERMEDIATE initial-SFT freeze.

    This file is what starts post-training: it names the parent checkpoint each
    continuation clones, and it is hash-verified for exactly that reason. It is
    stage-marked ``initial_sft_selection`` and cannot unlock reserved labels --
    the final freeze comes after every continuation, and only that one does.
    """
    selected, table = {}, {}
    for key, summary in results["policies"].items():
        epochs = {e: record for e, record in summary["epochs"].items()
                  if "val_positive_nll_per_residue" in record}
        require(epochs, f"{key} recorded no checkpoint evaluation")
        best = min(epochs, key=lambda e: epochs[e]["val_positive_nll_per_residue"])
        table[key] = {e: {k: v for k, v in record.items() if k != "val_ranking_diagnostic"}
                      for e, record in epochs.items()}
        selected[f"policy_{key}"] = {
            "checkpoint": epochs[best]["checkpoint"], "sha256": epochs[best]["checkpoint_sha256"],
            "epoch": int(best),
            "val_positive_nll_per_residue": epochs[best]["val_positive_nll_per_residue"],
            "kind": "policy"}
    weights = Path(config["raw_root"]) / data.PIGGEN_DIR / "model.safetensors"
    selected["piggen_zeroshot"] = {
        "checkpoint": weights.as_posix(), "sha256": sha256(ROOT / weights), "epoch": 0,
        "val_positive_nll_per_residue": None, "kind": "pinned_zero_shot"}
    for seed, report in results["classifiers"].items():
        selected[f"cnn_3class_seed{seed}"] = {
            "checkpoint": report["checkpoint"], "sha256": report["checkpoint_sha256"],
            "epoch": report["selected_epoch"],
            "val_three_class_cross_entropy": report["selected_val_cross_entropy"],
            "kind": "classifier_proxy"}
    selected["linear_3class"] = {
        "checkpoint": results["linear"]["checkpoint"], "sha256": results["linear"]["checkpoint_sha256"],
        "epoch": None, "val_three_class_cross_entropy": results["linear"]["val_cross_entropy"],
        "kind": "classifier_proxy"}
    expected = (set(evaluation.initial_policy_names(config))
                | {f"cnn_3class_seed{seed}" for seed in config["classifier"]["seeds"]}
                | {"piggen_zeroshot", "linear_3class"})
    require(set(selected) == expected,
            f"Initial selection key set mismatch. Missing: {sorted(expected - set(selected))}; "
            f"unexpected: {sorted(set(selected) - expected)}")
    document = {"schema_version": data.SELECTION_SCHEMA, "stage": data.SELECTION_STAGE_BASE,
                "config_sha256": results["config_sha256"],
                "config_digest": results["config_digest"],
                "git_commit": results.get("first_git_commit", results["git_commit"]),
                "frozen_at_git_commit": results["git_commit"],
                "code_digests": results["code_digests"],
                "source_digests": results["source_digests"], "preflight": results["preflight"],
                "selection_metric": config["policy"]["selection_metric"],
                "selection_population": config["policy"]["selection_population"],
                "checkpoint_table": table, "selected": selected,
                "shared_initial_cost": shared_initial_cost(results),
                "expected_keys": sorted(expected),
                "reserved_test_labels_read": False, "assay_outcomes_read": False,
                "note": ("intermediate stage: this freeze starts post-training and deliberately "
                         "does NOT unlock the reserved test labels or the assay outcomes. The "
                         "final freeze is written by scripts/posttrain_her2.py after every "
                         "continuation budget has been evaluated on validation.")}
    path = Path(output) / BASE_SELECTION
    if path.is_file():
        # A frozen selection may already have been consumed by an evaluation run.
        # Silently rewriting it would retroactively change what was evaluated.
        existing = load_json(path)
        require(existing["selected"] == selected,
                f"A different selection is already frozen at {path}; move it aside deliberately "
                "rather than overwriting it")
        print(f"selection already frozen at {path}", flush=True)
        return existing
    save_json(path, document)
    print(f"selection frozen at {path}", flush=True)
    return document


def build_context(config, raw_root):
    train = data.load_split(raw_root, "train")
    val = data.load_split(raw_root, "val")
    expected = config["expected_counts"]
    require(len(train) == expected["train_rows"] and len(val) == expected["val_rows"],
            "Split sizes differ from the declared population")
    train_index = data.encode_cores(train.seq)
    val_index = data.encode_cores(val.seq)
    train_positive = (train["class"] == data.POSITIVE_CLASS).to_numpy()
    val_positive = (val["class"] == data.POSITIVE_CLASS).to_numpy()
    require(int(train_positive.sum()) == expected["train_high"], "Training high-bin count changed")
    require(int(val_positive.sum()) == expected["val_high"], "Validation high-bin count changed")
    return {"train": train, "val": val, "train_index": train_index, "val_index": val_index,
            "train_positive_index": train_index[train_positive],
            "val_positive_index": val_index[val_positive],
            "train_targets": baselines.class_targets(train["class"]),
            "val_targets": baselines.class_targets(val["class"]),
            "val_positive": val_positive, "val_cores": val.seq.to_numpy(dtype=str),
            "train_prior": float(train_positive.mean()),
            "scaffold": data.load_scaffold(raw_root), "vocab": policy_lib.load_vocab(raw_root)}


def run(config_path, output, *, stages, allow_dirty, allow_cpu, discard_incomplete):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(config["schema_version"] == "her2-posttrain/1", "Unsupported protocol schema")
    for key, value in DECLARED.items():
        actual = config["policy"].get(key, config["preflight"].get(key))
        require(actual == value, f"Declared policy budget changed: {key} is {actual!r}")
    require(config["policy"]["kl_coefficient"] == 0.0
            and config["policy"]["reference_leash"] is False
            and config["policy"]["frozen_parameters"] == [],
            "This is full-parameter SFT with no KL penalty and no reference leash")
    if not allow_dirty:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        require(not dirty, "Commit the protocol, config and code before fitting")
    raw_root = ROOT / config["raw_root"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    threads = set_cpu_threads(config["runtime"]["cpu_threads"])
    print(f"CPU threads: {threads} (declared in the config; the probe timings were taken here)",
          flush=True)

    # Both records, at every entry point: the untracked retrieval manifest next to
    # the data AND the tracked manifests under specs/, re-hashed against the files.
    problems = data.verify_all_sources(raw_root, ROOT, config["source_manifests"])
    require(not problems, f"Pinned source files failed verification: {problems}")
    digests = source_digests(config, raw_root)
    preflight = resolve_preflight(config, output, allow_cpu=allow_cpu)
    device = preflight["device"]
    batch_size = preflight["batch_size"]
    print(f"preflight: {preflight['decision']} -> device {device}, batch {batch_size}, "
          f"free {preflight['free_vram_mib']} MiB", flush=True)

    context = build_context(config, raw_root)
    # Scientific code hashes are part of the identity: an edited trainer must not be
    # able to "resume" a completed fit produced by the previous version and inherit
    # its attribution. Documentation commits do not touch these files.
    identity_base = {"config_digest": digest_document(config), "source_digests": digests,
                     "code_digests": code_digests(ROOT),
                     "preflight_batch_size": batch_size, "device": device}
    results = {"schema_version": "her2-training/1", "config_sha256": sha256(config_path),
               "config_digest": identity_base["config_digest"], "git_commit": git_commit(),
               "code_digests": identity_base["code_digests"],
               "source_digests": digests, "preflight": preflight, "runtime": torch_runtime(),
               "training_population": {"policy_rows": int(context["train_positive_index"].shape[0]),
                                       "classifier_rows": int(context["train_index"].shape[0]),
                                       "val_positive_rows": int(context["val_positive_index"].shape[0]),
                                       "train_positive_prior": context["train_prior"]},
               "policies": {}, "classifiers": {}, "linear": {}}
    results_path = output / "training_results.json"
    results["first_git_commit"] = results["git_commit"]
    if results_path.is_file():
        # Every identity check happens BEFORE this file is rewritten. The previous
        # version verified the config alone and then overwrote the code hashes,
        # source hashes and commit attribution of results it had not checked --
        # so a `--stages select` invocation could re-attribute an older fit to the
        # current code without ever loading a checkpoint.
        previous = load_json(results_path)
        require(previous["config_digest"] == results["config_digest"],
                "An earlier run in this directory used a different configuration")
        for key, what in (("code_digests", "scientific code"), ("source_digests", "pinned sources")):
            stored = previous.get(key) or {}
            differing = sorted(name for name in set(stored) | set(results[key])
                               if stored.get(name) != results[key].get(name))
            require(not differing,
                    f"An earlier run in this directory used different {what}. Its results are not "
                    "this code's results; choose a fresh output directory rather than relabelling "
                    f"them. Differing entries: {differing[:5]}")
        results["first_git_commit"] = previous.get("first_git_commit", previous.get("git_commit"))
        results.update({k: previous.get(k, results[k]) for k in ("policies", "classifiers", "linear")})

    if "policies" in stages:
        for arm in config["policy"]["arms"]:
            for seed in config["policy"]["seeds"]:
                print(f"=== policy {arm} seed {seed} ===", flush=True)
                summary = run_policy(arm, seed, config, context, output, device=device,
                                     batch_size=batch_size, identity_base=identity_base,
                                     discard_incomplete=discard_incomplete)
                results["policies"][f"{arm}_seed{seed}"] = summary
                save_json(results_path, results)
    if "classifiers" in stages:
        print("=== classifier proxy (3 seeds) ===", flush=True)
        results["classifiers"] = run_classifiers(config, context, output, device=device,
                                                 identity_base=identity_base,
                                                 discard_incomplete=discard_incomplete)
        save_json(results_path, results)
    if "linear" in stages:
        print("=== additive linear ===", flush=True)
        results["linear"] = run_linear(config, context, output, identity_base=identity_base,
                                       discard_incomplete=discard_incomplete)
        save_json(results_path, results)
    save_json(results_path, results)
    if "select" in stages:
        # Exact sets, not counts: a count check passes when one arm is missing and
        # another is present twice.
        expected_policies = {f"{arm}_seed{seed}" for arm in config["policy"]["arms"]
                             for seed in config["policy"]["seeds"]}
        require(set(results["policies"]) == expected_policies,
                f"Every policy run must finish before the selection is frozen. Missing: "
                f"{sorted(expected_policies - set(results['policies']))}; unexpected: "
                f"{sorted(set(results['policies']) - expected_policies)}")
        expected_classifiers = {str(seed) for seed in config["classifier"]["seeds"]}
        require(set(results["classifiers"]) == expected_classifiers and results["linear"],
                f"Every baseline must finish before the selection is frozen. Classifier seeds "
                f"present: {sorted(results['classifiers'])}, expected {sorted(expected_classifiers)}")
        freeze_selection(config, results, output)
    print("training stage complete", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/her2_posttrain.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/her2_posttrain_20260918")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help=f"comma-separated subset of {STAGES}")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="skip the clean-worktree gate (synthetic runs only)")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="permit a CPU run; the declared budget is a GPU budget")
    parser.add_argument("--discard-incomplete", action="store_true",
                        help="restart runs left in the running state by a killed process")
    args = parser.parse_args()
    chosen = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = sorted(set(chosen) - set(STAGES))
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    run(args.config, args.output, stages=chosen, allow_dirty=args.allow_dirty,
        allow_cpu=args.allow_cpu, discard_incomplete=args.discard_incomplete)
