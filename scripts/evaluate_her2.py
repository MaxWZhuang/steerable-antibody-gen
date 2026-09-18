#!/usr/bin/env python
"""One-shot HER2 evaluation: reserved test split, persisted draws, independent assay.

This script cannot run without a **final**-stage ``selection_frozen.json``, and
before it reads a single reserved label it verifies: every checkpoint hash in that
file, the current scientific code hashes against the ones the freeze was written
under, the exact declared method/seed/budget name sets, the validation-records
file, every persisted draw file and every persisted validation score vector. That
is a data-flow gate, not an approval request: it is what makes "the checkpoints we
evaluated are the ones validation chose" a checkable statement rather than a
promise. It is a workflow guarantee, not a cryptographic one.

The three-class CNN and the additive linear model appear here as **auxiliary
ranking comparators on measured populations** -- the reserved test split and the
assay cohort -- and nowhere else. They did not generate anything, score any
preference, or choose any checkpoint, and no generated draw receives a proxy
probability from them.

It also refuses to overwrite a completed evaluation. Re-running against reserved
labels until a number improves is the failure mode; a fresh output directory is
the deliberate way to run again.

Three populations, kept apart on purpose:

* the 78,652 reserved test rows -- every scorer, and every raw budget checkpoint,
  through one metric function, so the scaling curve and the selected points come
  from the same code;
* the draws already generated and persisted during the validation stage --
  re-read by hash, never re-randomized here, and now joined against the test
  labels as well so the per-split conditional rates are complete;
* the independent assay cohort -- other people's designs, measured by somebody
  else, with every Buzz-library overlap removed.

Policies are streamed one at a time. Thirty-one of them do not fit on a 4 GB card
at once, and there is no reason to hold them: the populations are identical, so
only the resulting score vectors are kept, on the CPU.

No new assay is performed on any generated sequence. Generated cores that exactly
match a catalogue row inherit that row's published bin, which is a lookup, not a
measurement of the design.
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
from smallAntibodyGen.experiments import her2_baselines as baselines  # noqa: E402
from smallAntibodyGen.experiments import her2_data as data  # noqa: E402
from smallAntibodyGen.experiments import her2_eval as evaluation  # noqa: E402
from smallAntibodyGen.experiments import her2_policy as policy_lib  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import (  # noqa: E402
    code_digests, directory_digests, load_json, relative_key, require, save_json, set_cpu_threads,
    sha256, torch_runtime)

POLICY_KINDS = ("policy", "pinned_zero_shot")

#: Exactly which artifacts the final freeze must name. One implementation, shared
#: with the stage that writes the freeze.
expected_selection = evaluation.expected_selection_names


def verify_freeze_evidence(freeze, config, *, root=ROOT):
    """Re-hash everything the freeze cites and check its declared sets -- BEFORE labels.

    Ordering is the whole point. The previous version verified the draw files
    halfway through the run, after the reserved test split and the assay outcomes
    had already been loaded, so a failure would have fired with the labels already
    in the process. Everything checkable about the freeze is therefore checked
    here, at the top, and the reserved loaders run only if this returns.

    What it checks: the exact declared method/seed/budget name sets (not counts),
    the validation-records file by hash, every persisted draw file and every
    persisted validation score vector by hash, and that the generation table and
    the validation records name the same artifacts as the selection.
    """
    required = freeze.get("required") or {}
    expected = evaluation.required_artifact_sets(config)
    for key in ("initial_policies", "raw_budget_checkpoints"):
        require(sorted(required.get(key) or []) == expected[key],
                f"The freeze's {key} do not match the declared campaign. Missing: "
                f"{sorted(set(expected[key]) - set(required.get(key) or []))}; unexpected: "
                f"{sorted(set(required.get(key) or []) - set(expected[key]))}")
    for key in ("methods", "seeds"):
        require(list(required.get(key) or []) == expected[key],
                f"The freeze declares {key} {required.get(key)!r}, not {expected[key]!r}")
    require(dict(required.get("budgets_gpu_seconds") or {}) == expected["budgets_gpu_seconds"],
            "The freeze declares different budgets than the config")

    evidence = {"generation_files": 0, "validation_score_files": 0}
    stored = freeze.get("validation_records")
    require(stored, "The freeze cites no validation records; there is nothing to verify")
    record_path = root / stored["path"]
    require(record_path.is_file(), f"Validation records are missing: {record_path}")
    digest = sha256(record_path)
    require(digest == stored["sha256"],
            f"Validation records changed since the freeze: {digest} != {stored['sha256']}")
    records = load_json(record_path)
    evidence["validation_records_sha256"] = digest
    policies = {name for name, entry in freeze["selected"].items()
                if entry["kind"] in POLICY_KINDS}
    require(set(records) == policies,
            f"The validation records and the frozen policies disagree: "
            f"{sorted(set(records) ^ policies)}")
    require(set(freeze["generation"]) == policies,
            f"The frozen generation table and the frozen policies disagree: "
            f"{sorted(set(freeze['generation']) ^ policies)}")
    for name, entry in sorted(freeze["generation"].items()):
        path = root / entry["path"]
        require(path.is_file(), f"Persisted draws are missing: {path}")
        actual = sha256(path)
        require(actual == entry["sha256"],
                f"Persisted draws for {name} changed on disk: {actual} != {entry['sha256']}")
        evidence["generation_files"] += 1
    for name, record in sorted(records.items()):
        require(record.get("generation_samples") == freeze["generation"][name],
                f"{name}: frozen draw evidence disagrees with the validation record")
        scores = record.get("val_scores")
        require(scores, f"{name}: the validation record carries no persisted score vector")
        path = root / scores["path"]
        require(path.is_file(), f"Persisted validation scores are missing: {path}")
        actual = sha256(path)
        require(actual == scores["sha256"],
                f"Persisted validation scores for {name} changed on disk: {actual}")
        evidence["validation_score_files"] += 1
    evaluation.require_reference_diagnostics(
        records, {name: freeze["selected"][name] for name in policies})
    evidence["required"] = required
    evidence["checked_before_any_reserved_label"] = True
    return evidence, records


def parent_of(freeze):
    """``policy name -> the reference it was continued from``, from the freeze itself."""
    return {name: entry.get("parent_name") for name, entry in freeze["selected"].items()
            if entry["kind"] in POLICY_KINDS and entry.get("parent_name")}


def load_policy_model(entry, raw_root, device):
    if entry["kind"] == "pinned_zero_shot":
        return policy_lib.load_pinned_model(raw_root, device=device)
    model = policy_lib.architecture_model(raw_root, device=device)
    policy_lib.load_checkpoint(ROOT / entry["checkpoint"], model, device=device)
    return model


def stream_policy_scores(freeze, config, raw_root, scaffold, vocab, device, populations):
    """Score every population with every policy, one policy resident at a time.

    Returns ``{policy_name: {population_name: mean log probability array}}``. The
    arrays live on the CPU; the model is released before the next one is built,
    which is what makes 31 policies fit on a 4 GB card.
    """
    batch = config["inference"]["score_batch_size"]
    out, timings, by_digest = {}, {}, {}
    for name in sorted(freeze["selected"]):
        entry = freeze["selected"][name]
        if entry["kind"] not in POLICY_KINDS:
            continue
        if entry["sha256"] in by_digest:
            # Byte-identical weights give byte-identical scores. Two names for one
            # artifact is a bookkeeping fact, not a second measurement.
            source = by_digest[entry["sha256"]]
            out[name] = out[source]
            timings[name] = 0.0
            print(f"  {name} is byte-identical to {source}; reusing its scores", flush=True)
            continue
        started = time.perf_counter()
        model = load_policy_model(entry, raw_root, device)
        model.eval()
        policy = policy_lib.CorePolicy.from_prefix(model, scaffold.prefix, vocab, device=device)
        out[name] = {population: policy.score(index, batch_size=batch)["mean_log_probability"]
                     for population, index in populations.items()}
        del policy, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        by_digest[entry["sha256"]] = name
        timings[name] = time.perf_counter() - started
        print(f"  scored {name} ({timings[name]:.1f}s)", flush=True)
    return out, timings


def assemble_scores(population, index, cores, *, policy_scores, classifiers, linear, neighbour,
                    prior, device, parents=None):
    """Every declared scorer over one population, in one place, on identical rows.

    Three policy scoring rules travel together, under names that say which is
    which: the raw density, the density minus the zero-shot model (the whole
    post-training path), and the density minus the policy's **own SFT parent** --
    the DPO implicit reward, which is the quantity the objective optimized. Beta is
    omitted because a positive scale cannot reorder anything.

    All three are reported. None of them is a selection rule: selection already
    happened, on validation, by raw density.
    """
    scores = {"prior": baselines.prior_scores(len(index), prior),
              "neg_wt_hamming": baselines.negative_wt_distance(index),
              "nn_label": neighbour.score,
              "linear_3class": linear.positive_scores(index)}
    ensemble, per_seed = baselines.ensemble_probabilities(list(classifiers.values()), index,
                                                         device=device)
    high = data.CLASS_ORDER.index("high")
    scores["cnn_3class_ensemble"] = ensemble[:, high]
    for name, probabilities in zip(classifiers, per_seed):
        scores[name] = probabilities[:, high]
    for name, values in policy_scores.items():
        scores[name] = values[population]
    zero_shot = policy_scores["piggen_zeroshot"][population]
    for name in list(policy_scores):
        if name != "piggen_zeroshot":
            scores[f"{name}_minus_zeroshot"] = policy_scores[name][population] - zero_shot
    for name, parent in sorted((parents or {}).items()):
        if name in policy_scores and parent in policy_scores and parent != name:
            scores[f"{name}_minus_parent"] = (policy_scores[name][population]
                                              - policy_scores[parent][population])
    require(all(len(v) == len(index) for v in scores.values()),
            "A scorer returned the wrong length")
    require(len(cores) == len(index), "Cores and scores are misaligned")
    return scores


def load_classifiers(freeze, device):
    models = {}
    for name, entry in freeze["selected"].items():
        if not name.startswith("cnn_3class_seed"):
            continue
        payload = torch.load(ROOT / entry["checkpoint"], map_location="cpu", weights_only=True)
        model = baselines.MasonCNN()
        model.load_state_dict(payload["state"], strict=True)
        models[name] = model.to(device).eval()
    return models


def load_linear(freeze):
    payload = torch.load(ROOT / freeze["selected"]["linear_3class"]["checkpoint"],
                         map_location="cpu", weights_only=True)
    return baselines.AdditiveLinear(payload["weight"], payload["bias"])


def bootstrap_pairs(config, freeze):
    """Matched and extended comparisons only -- never the quadratic all-pairs set.

    With 31 policies the full pairwise table is 465 comparisons on 78,652 rows,
    which is a lot of compute for a table nobody reads. The declared pairs are the
    ones the experiment was designed to answer: each seed's DPO against its own
    continued-SFT control at the same budget, each against its parent, and the
    original arm-level comparisons.
    """
    pairs = []
    for left, right in config["evaluation"]["paired_comparisons"]:
        if "{seed}" in left or "{seed}" in right:
            pairs.extend((left.format(seed=seed), right.format(seed=seed))
                         for seed in config["policy"]["seeds"])
        else:
            pairs.append((left, right))
    continuation = config["continuation"]
    shared = sorted(set(continuation["budgets_gpu_seconds"]["dpo"])
                    & set(continuation["budgets_gpu_seconds"]["continued_sft"]))
    for seed in continuation["seeds"]:
        for budget in shared:
            pairs.append((f"dpo_seed{seed}_budget{int(budget)}",
                          f"continued_sft_seed{seed}_budget{int(budget)}"))
        for budget in continuation["budgets_gpu_seconds"]["dpo"]:
            pairs.append((f"dpo_seed{seed}_budget{int(budget)}", f"policy_sft_seed{seed}"))
    return pairs


def reuse_generation(freeze, config, context, output, validation):
    """Re-read the persisted draws by hash and complete them with the test labels.

    The draws were generated and hashed during validation, before any reserved
    label was read. Re-sampling here would mean the evaluated distribution was
    drawn after the labels were in the process; re-reading the same file keeps the
    ordering honest and makes the artifact the thing under test.

    ``validation`` is the record set :func:`verify_freeze_evidence` already
    verified, so the hashes below are a second, cheap confirmation rather than the
    first look.
    """
    document = {}
    for name, record in sorted(freeze["generation"].items()):
        path = ROOT / record["path"]
        require(path.is_file(), f"Persisted draws are missing: {path}")
        digest = sha256(path)
        require(digest == record["sha256"],
                f"Persisted draws for {name} changed on disk: {digest} != {record['sha256']}")
        frame = pd.read_csv(path, dtype={"core": str})
        require(len(frame) == record["draws"], f"{name}: draw count changed")
        index = data.encode_cores(frame.core)
        diagnostics = evaluation.generation_diagnostics(
            index, frame.sum_log_probability.to_numpy(dtype=np.float64),
            catalogs=context["catalogs"], train_index=context["train_index"],
            train_labels=context["train_labels"], split_of=context["split_of"],
            class_of=context["class_of"])
        diagnostics["source"] = {"path": record["path"], "sha256": digest,
                                 "note": "generated and hashed during validation, not re-sampled here"}
        # The Monte Carlo KLs were measured during validation, against models that
        # are not resident here; they are carried over rather than recomputed.
        for key in ("kl_to_zero_shot", "kl_to_sft_parent"):
            if name in validation and key in validation[name].get("generation", {}):
                diagnostics[key] = validation[name]["generation"][key]
        if name in validation:
            diagnostics["diversity"] = validation[name]["diversity"]
            diagnostics["val_metrics"] = validation[name]["val_metrics"]
            for key in ("val_metrics_minus_parent", "val_metrics_minus_zero_shot"):
                if key in validation[name]:
                    diagnostics[key] = validation[name][key]
        document[name] = diagnostics
    return document


def run(config_path, output, *, allow_dirty, allow_cpu, training_output, freeze_path=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(config["schema_version"] == "her2-posttrain/1", "Unsupported protocol schema")
    output = Path(output)
    results_path = output / "results.json"
    if results_path.is_file():
        previous = load_json(results_path)
        require(previous.get("status") != "completed",
                f"A completed evaluation already exists at {results_path}. Reserved labels are "
                "read once; choose a fresh output directory rather than overwriting it.")
    if not allow_dirty:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        text=True).strip()
        require(not dirty, "Commit the code and protocol before the final evaluation")
    raw_root = ROOT / config["raw_root"]
    problems = data.verify_all_sources(raw_root, ROOT, config["source_manifests"])
    require(not problems, f"Pinned source files failed verification: {problems}")

    freeze_path = Path(freeze_path or Path(training_output) / "continuation"
                       / "selection_frozen.json")
    # Current code hashes, not the freeze's own: copying the freeze's hashes into
    # the report proves only that the file is self-consistent. If the scientific
    # code has moved since the freeze, this evaluation is of something else.
    freeze, unlock = data.read_selection_freeze(
        freeze_path, root=ROOT, expected_config_sha256=sha256(config_path),
        expected_stage=data.SELECTION_STAGE_FINAL,
        expected_selected=expected_selection(config),
        expected_code_digests=code_digests(ROOT),
        expected_source_digests=data.source_digests(ROOT, raw_root, config["source_manifests"]))
    evidence, validation = verify_freeze_evidence(freeze, config)
    print(f"verified {len(unlock.verified)} frozen artifacts, "
          f"{evidence['generation_files']} draw files and "
          f"{evidence['validation_score_files']} validation score vectors from {freeze_path} "
          "-- before any reserved label was read", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    require(device == "cuda" or allow_cpu, "No CUDA device; pass --allow-cpu to accept CPU timings")
    output.mkdir(parents=True, exist_ok=True)
    threads = set_cpu_threads(config["runtime"]["cpu_threads"])
    print(f"CPU threads: {threads} (declared in the config)", flush=True)

    scaffold = data.load_scaffold(raw_root)
    vocab = policy_lib.load_vocab(raw_root)
    train = data.load_split(raw_root, "train")
    val = data.load_split(raw_root, "val")
    test = data.load_split(raw_root, "test", unlock=unlock)
    train_index = data.encode_cores(train.seq)
    test_index = data.encode_cores(test.seq)
    train_labels = (train["class"] == data.POSITIVE_CLASS).to_numpy().astype(np.float64)
    prior = float(train_labels.mean())
    catalogs = {"train": set(train.seq), "val": set(val.seq), "test": set(test.seq)}
    split_of, class_of = data.labelled_lookup({"train": train, "val": val, "test": test})
    context = {"train_index": train_index, "train_labels": train_labels, "catalogs": catalogs,
               "split_of": split_of, "class_of": class_of}

    results = {"schema_version": "her2-evaluation/2", "status": "running",
               "config_sha256": sha256(config_path), "git_commit": freeze["git_commit"],
               "frozen_at_git_commit": freeze.get("frozen_at_git_commit"),
               "evaluated_at_git_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "code_digests": code_digests(ROOT),
               "freeze_code_digests_match_current": True,
               "freeze_evidence": evidence,
               "shared_initial_cost": freeze.get("shared_initial_cost"),
               "selection_freeze_sha256": unlock.freeze_sha256,
               "selection_freeze_path": relative_key(freeze_path, ROOT),
               "selection_within_budget": freeze["selection_within_budget"],
               "selected": freeze["selected"], "runtime": torch_runtime(), "device": device,
               "test_rows": int(len(test)), "train_positive_prior": prior}
    save_json(results_path, results)

    print("building the independent assay cohort", flush=True)
    library = catalogs["train"] | catalogs["val"] | catalogs["test"]
    cohort = data.assay_cohort(raw_root, scaffold, library, unlock=unlock, include_outcome=True)
    primary = cohort["primary"].copy()
    outcomes = [data.classify_outcome(value) for value in primary.kd_molar]
    primary["outcome_class"] = [entry[0] for entry in outcomes]
    primary["kd_molar"] = [entry[1] for entry in outcomes]
    assay_index = data.encode_cores(primary.core)

    print("streaming every policy over the test split and the assay cohort", flush=True)
    policy_scores, policy_seconds = stream_policy_scores(
        freeze, config, raw_root, scaffold, vocab, device,
        {"test": test_index, "assay": assay_index})
    results["policy_scoring_seconds"] = policy_seconds
    classifiers = load_classifiers(freeze, device)
    linear = load_linear(freeze)

    neighbour = data.nearest_training_labels(test_index, train_index, train_labels,
                                             max_distance=2, prior=prior)
    cores = test.seq.to_numpy(dtype=str)
    parents = parent_of(freeze)
    scores = assemble_scores("test", test_index, cores, policy_scores=policy_scores,
                             classifiers=classifiers, linear=linear, neighbour=neighbour,
                             prior=prior, device=device, parents=parents)
    # The score file is written before any label is joined. That is software
    # separation in this process, not proof of independence: the labels were loaded
    # earlier in the same interpreter, so the honest claim is that no label entered
    # any scorer's inputs, which the score computation above makes checkable.
    pd.DataFrame({"seq": test.seq, **scores}).to_csv(output / "test_scores.csv", index=False)
    results["test_score_sha256"] = sha256(output / "test_scores.csv")
    save_json(results_path, results)

    positive = (test["class"] == data.POSITIVE_CLASS).to_numpy()
    k_values = tuple(config["evaluation"]["k_values"])
    results["test_metrics"] = {name: evaluation.rank_metrics(value, positive, cores,
                                                             k_values=k_values)
                               for name, value in scores.items()}
    for name, metrics in results["test_metrics"].items():
        base_name = name.replace("_minus_zeroshot", "").replace("_minus_parent", "")
        entry = freeze["selected"].get(name) or freeze["selected"].get(base_name, {})
        metrics["role"] = entry.get("role")
        metrics["method"] = entry.get("method")
        metrics["seed"] = entry.get("seed")
        # Nominal target and measured device time travel together everywhere; the
        # selection rule compared the nominal one and says so.
        metrics["budget_seconds"] = entry.get("budget_seconds")
        metrics["actual_gpu_seconds"] = entry.get("actual_gpu_seconds")
        metrics["updates"] = entry.get("updates")
        metrics["distinct_exposures"] = entry.get("distinct_exposures")
        if name.endswith("_minus_parent"):
            metrics["scoring_rule"] = "policy minus its own SFT parent (DPO implicit reward)"
        elif name.endswith("_minus_zeroshot"):
            metrics["scoring_rule"] = "policy minus the zero-shot model"
    selected_names = {choice["selected"] for run in freeze["selection_within_budget"].values()
                      for choice in run.values() if choice.get("selected")}
    for name in selected_names:
        if name in results["test_metrics"]:
            results["test_metrics"][name]["selected_within_budget"] = True
    results["scaling_curve_note"] = (
        "every raw budget checkpoint is evaluated, including the ones the diversity gates "
        "rejected and the budgets at which nothing was eligible. `selected_within_budget` marks "
        "the checkpoints the frozen rule chose; the rest are the curve, not failures to hide.")
    results["policy_scoring_rules"] = {
        "<name>": "raw mean log probability per residue -- the declared selection quantity",
        "<name>_minus_zeroshot": "policy minus the pinned zero-shot model: the whole "
                                 "post-training path",
        "<name>_minus_parent": "policy minus its own initial-SFT parent: DPO's implicit reward, "
                               "the quantity the objective optimized. Beta is omitted because a "
                               "positive scale cannot change a ranking.",
        "note": ("all three are reported for every policy. Selection was already made on "
                 "validation by raw density; reporting three rules here is disclosure, not a "
                 "best-of choice.")}
    wt_strata = evaluation.wt_distance_strata(test_index)
    train_strata = neighbour.strata()
    results["test_metrics_by_wt_distance"] = {
        name: evaluation.stratified_metrics(value, positive, cores, wt_strata,
                                            categories=config["evaluation"]["wt_distance_strata"])
        for name, value in scores.items()}
    results["test_metrics_by_train_distance"] = {
        name: evaluation.stratified_metrics(value, positive, cores, train_strata,
                                            categories=config["evaluation"]["train_distance_strata"])
        for name, value in scores.items()}
    sanity = results["test_metrics_by_wt_distance"]["neg_wt_hamming"]
    results["stratifier_sanity_check"] = {
        "claim": "neg_wt_hamming is constant inside a d(WT) stratum, so its stratified AUROC must "
                 "be exactly 0.5 wherever both classes are present",
        "observed": {k: v.get("auroc") for k, v in sanity.items()},
        "passes": all(v.get("auroc") is None or abs(v["auroc"] - 0.5) < 1e-9
                      for v in sanity.values())}
    require(results["stratifier_sanity_check"]["passes"], "Stratified AUROC sanity check failed")
    save_json(results_path, results)

    print("bootstrap paired differences (declared comparisons only)", flush=True)
    pairs = [pair for pair in bootstrap_pairs(config, freeze)
             if pair[0] in scores and pair[1] in scores]
    results["paired_differences"] = evaluation.bootstrap_paired_differences(
        scores, positive, cores, pairs, draws=config["evaluation"]["bootstrap_draws"],
        seed=config["evaluation"]["bootstrap_seed"],
        confidence=config["evaluation"]["confidence"])
    results["paired_difference_scope"] = (
        "matched and extended comparisons declared in the config; the quadratic all-pairs table "
        "is deliberately not computed")
    # Seed spread is reported next to, and separately from, the bootstrap intervals:
    # three seeds disagreeing is a different uncertainty from row-level resampling,
    # and averaging them into one number hides which is doing the work.
    spread = {}
    for method in config["continuation"]["methods"]:
        for budget in config["continuation"]["budgets_gpu_seconds"][method]:
            values = []
            for seed in config["continuation"]["seeds"]:
                name = f"{method}_seed{seed}_budget{int(budget)}"
                if name in results["test_metrics"]:
                    values.append({"seed": seed,
                                   "average_precision":
                                       results["test_metrics"][name]["average_precision"],
                                   "auroc": results["test_metrics"][name]["auroc"]})
            points = [v["average_precision"] for v in values if v["average_precision"] is not None]
            spread[f"{method}_budget{int(budget)}"] = {
                "per_seed": values,
                "average_precision_min": min(points) if points else None,
                "average_precision_max": max(points) if points else None}
    results["seed_spread"] = {
        "by_method_and_budget": spread,
        "note": "seed-to-seed spread, reported apart from the row-resampling bootstrap intervals"}
    save_json(results_path, results)

    print("re-reading the persisted draws and joining the reserved labels", flush=True)
    results["generation"] = reuse_generation(freeze, config, context, output, validation)
    results["generation_note"] = (
        "identical draw count for every policy, temperature 1, duplicates retained, no restriction "
        "to the training catalogue. Draws were produced and hashed during validation and are "
        "re-read here, not re-sampled. Overlap is a count with an exact interval, reported per "
        "split and separately for the held-out splits; an empty intersection is reported as zero.")
    save_json(results_path, results)

    print("independent assay endpoint", flush=True)
    assay_neighbour = data.nearest_training_labels(assay_index, train_index, train_labels,
                                                   max_distance=2, prior=prior)
    assay_scores = assemble_scores("assay", assay_index, primary.core.to_numpy(dtype=str),
                                   policy_scores=policy_scores, classifiers=classifiers,
                                   linear=linear, neighbour=assay_neighbour, prior=prior,
                                   device=device, parents=parents)
    for name, value in assay_scores.items():
        primary[name] = value
    primary.drop(columns=[c for c in ("heavy", "light", "H", "L") if c in primary.columns]).to_csv(
        output / "assay_cohort.csv", index=False)
    results["assay"] = {"cohort": cohort["counts"], "sheet_name": cohort["sheet_name"],
                        "headers": cohort["header"],
                        "unsupported_outcome_rows": int((primary.outcome_class ==
                                                         data.OUTCOME_UNSUPPORTED).sum()),
                        "missing_outcome_rows": int((primary.outcome_class ==
                                                     data.OUTCOME_MISSING).sum()),
                        "endpoints": {name: evaluation.assay_endpoints(
                            primary, name, seed=config["evaluation"]["assay_seed"],
                            draws=config["evaluation"]["spearman_draws"])
                            for name in assay_scores},
                        "assay_modality": ("recorded as SPR on a Carterra CMDP chip, which is what "
                                           "the workbook metadata declares. The repository README "
                                           "abstract says Biolayer Interferometry; the discrepancy "
                                           "is preserved and the workbook is the labelled source."),
                        "boundary": ("this endpoint scores other people's designs. It says nothing "
                                     "about sequences this policy generated, and no new assay was "
                                     "conducted on any generated sequence.")}
    if results["assay"]["unsupported_outcome_rows"]:
        results["assay"]["unsupported_outcome_note"] = (
            "rows whose outcome cell is neither a finite positive KD, N.B. nor I.C. are excluded "
            "from both endpoints and reported here; none is guessed")
    save_json(results_path, results)

    results["artifact_sha256"] = directory_digests(output)
    results["reserved_test_labels_read"] = True
    results["assay_outcomes_read"] = True
    results["disclosure"] = (
        "Test labels and assay outcomes were read only after the final selection_frozen.json "
        "existed and every hash in it verified. Aggregate integrity checks and two printed example "
        "rows per split were performed during the 2026-09-18 audit, before the freeze; that "
        "earlier access is disclosed rather than described as sealed from the start. HER2 "
        "pretraining exposure for p-IgGen remains unscanned and unresolved.")
    results["status"] = "completed"
    save_json(results_path, results)
    print(f"evaluation written to {results_path}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiments/her2_posttrain.json")
    parser.add_argument("--training-output", type=Path,
                        default=ROOT / "outputs/her2_posttrain_20260918")
    parser.add_argument("--freeze", type=Path, default=None,
                        help="path to the final selection_frozen.json (defaults to the "
                             "continuation directory under --training-output)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/her2_posttrain_20260918/evaluation")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, allow_dirty=args.allow_dirty, allow_cpu=args.allow_cpu,
        training_output=args.training_output, freeze_path=args.freeze)
