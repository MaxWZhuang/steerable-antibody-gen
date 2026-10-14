"""Explicit command interface for the low-resource transfer experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    QueryPanel, canonical_bytes, generate_attempts, positive_count, read_json,
    require_keys, score_queries, write_export, seed_value, text_value, validate_generation_request,
)
from smallAntibodyGen.experiments.trastuzumab_poc.checkpoint import (
    load_inference_checkpoint, resume_checkpoint, save_checkpoint,
)
from smallAntibodyGen.experiments.trastuzumab_poc.factory import build_models
from smallAntibodyGen.experiments.trastuzumab_poc.exposure import plan_post_exposure
from smallAntibodyGen.experiments.trastuzumab_poc.inputs import (
    coverage_report, load_broad, load_endpoint_manifest, load_post, record, source_identity,
)
from smallAntibodyGen.experiments.trastuzumab_poc.parent import export_parent
from smallAntibodyGen.experiments.trastuzumab_poc.profiling import profile_updates
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import select_budget
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import qualify_transfer
from smallAntibodyGen.experiments.trastuzumab_poc.runner import StageTrainer, TrainingConfig, check_exposure_plan
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    require_keys(config, ("schema", "model", "fusion", "training", "data", "initialization", "device"), "run config")
    if config["schema"] != "trastuzumab-run/1":
        raise ValueError("unsupported run schema")
    record(FusionConfig, config["fusion"])
    train = record(TrainingConfig, config["training"])
    data = config["data"]
    if not isinstance(config["model"], dict) or not isinstance(config["initialization"], dict):
        raise ValueError("model and initialization must be objects")
    if train.stage == "mid":
        keys = ("pairs_path", "transfer_guard") if "transfer_guard" in data else ("pairs_path",)
        require_keys(data, keys, "data config")
        if "transfer_guard" not in data and config["model"].get("kind") != "tiny":
            raise ValueError("released midtraining requires an outcome-free transfer_guard")
        if "transfer_guard" in data:
            require_keys(data["transfer_guard"], ("endpoint_manifest_path", "excluded_antibody_families",
                                                 "excluded_target_families"), "transfer guard config")
    else:
        require_keys(data, ("endpoint_manifest_path", "selected_labels_path", "budget_n", "budget_seed"),
                     "data config")
    # Resolve only explicitly declared paths relative to the configuration file.
    for section, keys in ((config["model"], ("weights_dir", "pins_path", "encoder_source_root",
                                           "encoder_source_pins_path", "encoder_checkpoint")),
                          (config["initialization"], ("path",)),
                          (data, ("pairs_path", "endpoint_manifest_path", "selected_labels_path")),
                          (data.get("transfer_guard", {}), ("endpoint_manifest_path",))):
        for key in keys:
            if key in section:
                value = section[key]
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{key} must be a nonempty path")
                section[key] = str((path.parent / value).resolve())
    return config


def models(config):
    training = record(TrainingConfig, config["training"])
    return build_models(config["model"], record(FusionConfig, config["fusion"]),
                        device=config["device"], seed=training.seed,
                        initialization=config["initialization"])


def trainer(config):
    training = record(TrainingConfig, config["training"])
    if training.stage == "mid":
        population, data_identity = load_broad(config["data"]["pairs_path"])
        data = {"population": population}
        guard = config["data"].get("transfer_guard")
        if guard is not None:
            data["transfer_certificate"] = qualify_transfer(
                population, load_endpoint_manifest(guard["endpoint_manifest_path"]),
                excluded_antibody_families=guard["excluded_antibody_families"],
                excluded_target_families=guard["excluded_target_families"])
    else:
        spec = config["data"]
        # Inspect the exact schedule before opening labels or constructing models.
        manifest, budget, plan = post_plan(config)
        check_exposure_plan(plan, training)
        restricted, data_identity = load_post(spec["endpoint_manifest_path"], spec["selected_labels_path"],
                                               budget_n=spec["budget_n"], budget_seed=spec["budget_seed"])
        if restricted.budget != budget:
            raise ValueError("endpoint manifest changed after exposure planning")
        data = {"post_data": restricted}
    policy, encoder, initialization = models(config)
    return StageTrainer(policy, encoder, training, **data, data_identity=data_identity,
                        source_identity=source_identity(), initialization_identity=initialization)


def post_plan(config):
    training = record(TrainingConfig, config["training"])
    spec = config["data"]
    manifest = load_endpoint_manifest(spec["endpoint_manifest_path"])
    budget = select_budget(manifest, spec["budget_n"], spec["budget_seed"])
    by_id = {entry.endpoint_id: entry for entry in manifest.entries}
    plan = plan_post_exposure([by_id[eid] for eid in budget.selected_ids], training)
    return manifest, budget, plan


def plan_inputs(config):
    """No model/weight/measurement access; report corpus or schedule qualification."""
    training = record(TrainingConfig, config["training"])
    if training.stage == "mid":
        population, identity = load_broad(config["data"]["pairs_path"])
        report = {"coverage": coverage_report(population), "input": identity, "transfer_certificate": None}
        guard = config["data"].get("transfer_guard")
        if guard is not None:
            report["transfer_certificate"] = qualify_transfer(
                population, load_endpoint_manifest(guard["endpoint_manifest_path"]),
                excluded_antibody_families=guard["excluded_antibody_families"],
                excluded_target_families=guard["excluded_target_families"])
    else:
        _, budget, plan = post_plan(config)
        report = {"budget": asdict(budget), "exposure": plan,
                  "coverage_requirement_met": not training.ordering_weight or
                      plan["pair_labels"]["fraction_of_selected"] >= training.minimum_ordering_endpoint_coverage}
    return {"schema": "trastuzumab-input-plan/1", "training": asdict(training), **report}


def write_new_json(path, document):
    with Path(path).open("xb") as stream:
        stream.write(canonical_bytes(document) + b"\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--pairs", required=True)
    qualify.add_argument("--output", required=True)
    select = commands.add_parser("select-budget")
    select.add_argument("--manifest", required=True)
    select.add_argument("--budget", required=True, type=int)
    select.add_argument("--budget-seed", required=True)
    select.add_argument("--output", required=True)
    for name in ("plan", "train", "profile", "score", "generate", "export-parent"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--resume")
            command.add_argument("--resume-sha256")
            command.add_argument("--stop-after-updates", type=int)
            command.add_argument("--checkpoint-every", type=int, default=100)
        elif name == "profile":
            command.add_argument("--warmup-updates", type=int, default=3)
            command.add_argument("--measured-updates", type=int, default=10)
        elif name != "plan":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--checkpoint-sha256", required=True)
            if name != "export-parent":
                command.add_argument("--queries", required=True)
                command.add_argument("--arm", choices=("paired", "abonly"), required=True)
            if name == "score":
                command.add_argument("--scorer-id", required=True)
                command.add_argument("--orders", type=int, default=4)
                command.add_argument("--order-seed", type=int, default=0)
            if name == "generate":
                command.add_argument("--arm-id", required=True)
                command.add_argument("--seeds", type=int, nargs="+", required=True)
                command.add_argument("--attempts-per-seed", type=int, required=True)
                command.add_argument("--reference-core", required=True)
                command.add_argument("--editable-count", type=int, default=10)
                command.add_argument("--steps", type=int, default=20)
                command.add_argument("--strategy", default="reparam-uncond-deterministic-linear")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if Path(args.output).exists():
        raise ValueError("output already exists; choose a new explicit output path")
    if args.command == "qualify":
        population, _ = load_broad(args.pairs)
        write_new_json(args.output, coverage_report(population))
        return
    if args.command == "select-budget":
        budget = select_budget(load_endpoint_manifest(args.manifest), args.budget, args.budget_seed)
        write_new_json(args.output, {"schema": "trastuzumab-budget/1", **asdict(budget)})
        return
    config = load_config(args.config)
    if args.command == "plan":
        write_new_json(args.output, plan_inputs(config))
        return
    if args.command in ("train", "profile"):
        if args.command == "train":
            positive_count(args.checkpoint_every, "checkpoint_every")
            if args.resume_sha256 and not args.resume:
                raise ValueError("--resume-sha256 requires --resume")
            if args.stop_after_updates is not None:
                positive_count(args.stop_after_updates, "stop_after_updates")
                if args.stop_after_updates > record(TrainingConfig, config["training"]).total_updates:
                    raise ValueError("stop boundary exceeds declared horizon")
        else:
            positive_count(args.warmup_updates, "warmup_updates")
            positive_count(args.measured_updates, "measured_updates")
            if args.warmup_updates + args.measured_updates > record(TrainingConfig, config["training"]).total_updates:
                raise ValueError("profile steps exceed configured horizon; declare enough steps explicitly")
        run = trainer(config)
        if args.command == "profile":
            report = profile_updates(run.step, run.policy, run.encoder,
                                     warmup_updates=args.warmup_updates, measured_updates=args.measured_updates,
                                     examples_per_update=run.config.examples_per_update,
                                     workload_identity=run.identity)
            write_new_json(args.output, report)
            print(json.dumps({"status": report["status"], "output": args.output}))
            return
        if args.resume:
            resume_checkpoint(run, args.resume, expected_sha256=args.resume_sha256)
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=False)
        write_new_json(destination / "config.json", config)
        write_new_json(destination / "identity.json", run.identity)
        if run.exposure_plan is not None:
            write_new_json(destination / "exposure-plan.json", run.exposure_plan)
        with (destination / "metrics.jsonl").open("x", encoding="utf-8") as log:
            def completed(current, metrics):
                log.write(canonical_bytes(metrics).decode("ascii") + "\n")
                log.flush()
                print(json.dumps(metrics), flush=True)
                if current.next_update % args.checkpoint_every == 0:
                    save_checkpoint(current, destination / "checkpoint.pt")
            run.run(stop_after_updates=args.stop_after_updates, on_update=completed)
        artifact = save_checkpoint(run, destination / "checkpoint.pt")
        write_new_json(destination / "result.json", {"schema": "trastuzumab-run-result/1",
                       "status": "complete" if run.next_update == run.config.total_updates else "stopped_at_boundary",
                       "checkpoint": artifact, "progress": run.progress()})
        return
    # Inference and parent export never reopen fitting or reserved outcomes.
    if args.command in ("score", "generate"):
        panel = QueryPanel.from_document(read_json(args.queries))
        if args.command == "score":
            positive_count(args.orders, "orders")
            seed_value(args.order_seed)
            text_value(args.scorer_id, "scorer_id")
        else:
            validate_generation_request(panel, arm_id=args.arm_id, sampling_seeds=args.seeds,
                                        attempts_per_seed=args.attempts_per_seed, steps=args.steps,
                                        strategy=args.strategy, reference_core=args.reference_core,
                                        editable_count=args.editable_count)
    policy, encoder, _ = models(config)
    provenance = load_inference_checkpoint(policy, args.checkpoint,
                                           encoder=encoder,
                                           expected_sha256=args.checkpoint_sha256,
                                           source_identity=source_identity())
    if args.command == "export-parent":
        result = export_parent(policy, args.output, provenance=provenance)
        print(json.dumps(result))
        return
    if args.command == "score":
        payload, metadata = score_queries(policy, encoder, panel, arm=args.arm,
                                            checkpoint_identity=provenance,
                                            order_seed=args.order_seed, orders=args.orders,
                                            scorer_id=args.scorer_id)
        filename = "scores.json"
    else:
        payload, metadata = generate_attempts(policy, encoder, panel, arm=args.arm, arm_id=args.arm_id,
                                               checkpoint_identity=provenance,
                                               sampling_seeds=args.seeds,
                                               attempts_per_seed=args.attempts_per_seed, steps=args.steps,
                                               strategy=args.strategy, reference_core=args.reference_core,
                                               editable_count=args.editable_count)
        filename = "attempts.json"
    write_export(args.output, payload=payload, metadata=metadata, payload_name=filename)
    print(json.dumps({"status": "exported", "output": args.output}))


if __name__ == "__main__":
    main()
