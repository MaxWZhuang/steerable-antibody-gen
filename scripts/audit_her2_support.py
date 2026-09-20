"""HER2 support-preservation audit: inventory, freeze, forward-KL scoring, CHES, decision.

Read-only with respect to every historical artifact. This CLI never commits, never
trains and never samples a model: it reads the persisted parent draw banks and the
checkpoints the two completed campaigns already wrote.

Stages, in protocol order::

    inventory   verify every state, bank and identity; write the run-root inventory
    prepare     write the portable tracked evidence a reviewer commits (no scoring)
    preflight   bounded numerical probes on the new code only, on persisted rows
    freeze      verify the committed sources/config/evidence/inputs; write the marker
    score       signed drops, tails, strata and the labelled historical join
    ches        parent CHES, displacement and the temporal analyses
    decide      apply the declared escalation rule
    report      publish the rebuildable narrative, figures and the coverage tables
    verify      re-check completed outputs for immutable content and timings

``--inventory-only`` and ``--verify-only`` are aliases for the first and last.

Examples::

    python scripts/audit_her2_support.py --help
    python scripts/audit_her2_support.py inventory
    python scripts/audit_her2_support.py --verify-only
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_ches as ches_lib            # noqa: E402
from smallAntibodyGen.experiments import her2_support as support         # noqa: E402
from smallAntibodyGen.experiments import her2_support_inventory as inventory_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths     # noqa: E402
from smallAntibodyGen.experiments import her2_support_scoring as scoring  # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import require            # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_support_audit.json"


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------

def choose_device(requested):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unknown device {requested!r}")
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_policy(context, device):
    """The pinned architecture, the fixed 99-token prefix and the canonical vocabulary.

    The container is built once and every checkpoint is strict-loaded into it, so
    this object is both the verifier and the scorer.
    """
    from smallAntibodyGen.experiments import her2_data as data
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    from smallAntibodyGen.experiments.her2_runtime import set_cpu_threads
    set_cpu_threads(int(context.config["inference"]["cpu_threads"]))
    raw = context.root("raw").local_path
    scaffold = data.load_scaffold(raw)
    vocab = policy_lib.load_vocab(raw)
    model = policy_lib.architecture_model(raw, device=device)
    model.eval()
    return policy_lib.CorePolicy.from_prefix(model, scaffold.prefix, vocab, device=device)


def load_state(context, reader, record):
    """Strict-load one inventory record into the shared container, by its adapter."""
    root = context.root("guarded") if record["root"] == context.root("guarded").logical \
        else context.root("original")
    logical = f"{record['root']}/{record['logical_path']}"
    target = root.path(record["logical_path"])
    if record["payload_schema"] == inventory_lib.POLICY_SCHEMA:
        return reader.read_policy(target, logical=logical,
                                  expected_file_sha256=record["file_sha256"])
    return reader.read_trajectory(target, logical=logical,
                                  expected_file_sha256=record["file_sha256"])


def score_cores(policy, index, *, batch_size, label):
    """The audit's scoring route: eval + inference mode, every logit checked finite."""
    block = scoring.strict_sequence_log_probabilities(
        policy, np.asarray(index), batch_size=int(batch_size), label=label)
    return np.asarray(block["sum_log_probability"], dtype=np.float64), block["checks"]


# ---------------------------------------------------------------------------
# stage: inventory
# ---------------------------------------------------------------------------

def cmd_inventory(context, args):
    device = choose_device(args.device)
    policy = build_policy(context, device)
    reader = inventory_lib.CheckpointReader(lambda: policy.model)
    progress = context.progress("inventory", every=5)
    with progress.guard():
        document, _ = support.build_inventory(context, reader=reader, progress=progress)
        document["device"] = device
        document["environment"] = support.environment_record()
        document["generated_at"] = paths.utc_now()
        support.require_new_or_identical(context.run.path(support.INVENTORY_JSON), document,
                                         what="the inventory", context=context, kind="inventory")
        support.record_timing_block(context, "inventory", document["timings"])
    coverage = document["coverage"]
    print(f"inventory: {coverage['total']} states, "
          f"{coverage['by_status'].get('verified', 0)} verified, "
          f"{document['deduplication']['distinct_computations']} distinct computations",
          flush=True)
    for shortfall in coverage["shortfalls"]:
        print(f"  COVERAGE GAP {shortfall}", flush=True)
    return document


# ---------------------------------------------------------------------------
# stage: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(context, args):
    inventory = paths.read_json(context.run.path(support.INVENTORY_JSON))
    # After a freeze, the tracked evidence is part of the frozen identity. Rerunning
    # prepare is allowed -- it must reproduce -- but it may not quietly rewrite the
    # files the marker pinned.
    frozen = support.freeze_marker_path(context).is_file()
    before = ({logical: paths.sha256_file(context.repository_root / logical)
               for logical in sorted((support.read_freeze_marker(context).get("evidence_sha256")
                                      or {}))} if frozen else {})
    clock = paths.StageClock()
    progress = context.progress("prepare", total=4)
    with progress.guard():
        with clock.segment("analysis"):
            pairs = support.build_pair_populations(context)
        progress.advance("pair populations")
        # Every evidence file is checked against what is already on disk BEFORE any
        # of them is written. Writing first and comparing digests afterwards reports
        # the violation only after the frozen bytes have already been replaced,
        # which is the one moment at which the check still had something to protect.
        with clock.segment("io"):
            pair_texts = support.render_pair_evidence(context, pairs)
            pair_files = support.verify_pair_evidence(context, pair_texts)
        progress.advance("pair evidence")
        with clock.segment("analysis"):
            manifest = support.build_input_manifest(context, inventory, pairs, pair_files)
        progress.advance("input manifest")
        with clock.segment("io"):
            evidence = context.repository_root / context.config["evidence_root"]
            support.require_new_or_identical(evidence / "inventory.json",
                                             support.portable_inventory(inventory),
                                             what="the portable inventory evidence")
            support.require_new_or_identical(
                context.repository_root / context.config["input_manifest"], manifest,
                what="the committed input manifest")
            support.write_pair_evidence(context, pair_texts)
            support.require_new_or_identical(context.run.path(support.AUDIT_CONFIG_JSON), {
                "schema_version": paths.AUDIT_SCHEMA, "record_kind": "resolved_config",
                "config_path": support._relative(context.config_path, context.repository_root),
                "config_sha256": context.config_sha256, "config_digest": context.config_digest,
                "config": context.config,
                "input_manifest": context.config["input_manifest"],
                "note": ("a resolved copy for the run directory. The committed config is the "
                         "authority and this file never feeds a hash back into it.")},
                what="the resolved configuration", context=context, kind="resolved_config")
            paths.write_local_roots(context.run.run_root,
                                    {root.logical: root.local_path
                                     for root in context.roots.values()})
        progress.advance("evidence written")
        changed = sorted(logical for logical, digest in before.items()
                         if paths.sha256_file(context.repository_root / logical) != digest)
        require(not changed,
                f"Rerunning prepare changed frozen evidence {changed}. The marker pins those "
                "bytes; a different result belongs in a new revision directory.")
        support.record_timings(context, "prepare", clock)
    print(f"prepare: {manifest['input_count']} input hashes, "
          f"{len(pair_files)} pair files written under {context.config['evidence_root']}",
          flush=True)
    print("prepare: commit the evidence and config, then run the freeze stage. This CLI does not "
          "commit.", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------

def cmd_preflight(context, args):
    inventory = paths.read_json(context.run.path(support.INVENTORY_JSON))
    device = choose_device(args.device)
    policy = build_policy(context, device)
    reader = inventory_lib.CheckpointReader(lambda: policy.model)
    seed = context.seeds[0]
    parent = _record_for(inventory, f"parent::policy_sft_seed{seed}")
    load_state(context, reader, parent)
    banks = support.verify_banks(context, support.stage_documents(context),
                                 {int(record["seed"]): record["file_sha256"]
                                  for record in inventory["records"]
                                  if record["role"] == inventory_lib.ROLE_PARENT})
    settings = context.config["inference"]
    progress = context.progress("preflight", total=1)
    with progress.guard():
        report = support.run_preflight(
            context, policy=policy, bank=banks[seed], rows=int(settings["preflight_rows"]),
            batch_rows=int(settings["preflight_batch_rows"]),
            batch_sizes=[int(size) for size in settings["preflight_batch_sizes"]])
        progress.advance("probes")
        report["device"] = device
        report["parent"] = {"id": parent["id"], "file_sha256": parent["file_sha256"],
                            "state_digest_audit_computed": parent["state_digest_audit_computed"]}
        # What this probe is evidence about. The freeze re-derives it and refuses a
        # preflight that was run against different sources, config, parents or banks.
        report["binding"] = support.preflight_binding(context)
        support.require_new_or_identical(context.run.path(support.PREFLIGHT_JSON), report,
                                         what="the preflight report", context=context,
                                         kind="preflight")
        support.record_timing_block(context, "preflight", report["timings"])
    print(f"preflight: head reconstruction "
          f"{report['head_reconstruction_full']['max_abs_error']:.3g} / "
          f"{report['head_reconstruction_cached']['max_abs_error']:.3g}, CHES cached-vs-full "
          f"{report['ches_cached_vs_full']['max_abs_error']:.3g}", flush=True)
    return report


def _record_for(inventory, identifier):
    for record in inventory["records"]:
        if record["id"] == identifier:
            return record
    raise ValueError(f"{identifier} is not in the inventory")


# ---------------------------------------------------------------------------
# stage: freeze
# ---------------------------------------------------------------------------

def cmd_freeze(context, args):
    marker = support.run_freeze(context)
    print(f"freeze: commit {marker['git']['commit']}, {marker['input_count']} inputs, "
          f"{len(marker['evidence_sha256'])} evidence files", flush=True)
    return marker


# ---------------------------------------------------------------------------
# stage: score
# ---------------------------------------------------------------------------

def parent_cache(context, *, policy, bank, parent_record, clock, probe_rows):
    """Score the parent once against its own bank, hash-bound, and probe it on reuse."""
    directory = context.run.path(support.shard_directory("parent_banks", parent_record["id"]))
    identity = {
        "parent_id": parent_record["id"],
        "parent_file_sha256": parent_record["file_sha256"],
        "parent_state_digest": parent_record["state_digest_audit_computed"],
        "bank_logical_path": bank.logical_path, "bank_sha256": bank.file_sha256,
        "bank_order_sha256": bank.order_sha256, "rows": bank.rows,
        "temperature": bank.temperature, "draw_seed": bank.draw_seed,
        "probability_convention": inventory_lib.CORE_CONTRACT["probability_convention"],
        "score_batch_size": int(context.config["inference"]["score_batch_size"])}
    batch = int(context.config["inference"]["score_batch_size"])
    reused = support.reusable_shard(context, "parent_banks", parent_record["id"], "parent_scores",
                                    identity=identity)
    if reused is not None:
        arrays, record = reused
        cached = arrays["parent_log_probability"]
        with clock.segment("inference"):
            probe, _ = score_cores(policy, bank.index[probe_rows], batch_size=batch,
                                   label=f"{parent_record['id']} cached parent probe")
        from smallAntibodyGen.experiments.her2_policy import compare_sum_log_probabilities
        record["probe_on_reuse"] = compare_sum_log_probabilities(
            probe, cached[probe_rows], label=f"{parent_record['id']} cached parent probe",
            atol=float(context.config["tolerances"]["sum_log_probability_atol"]),
            rtol=float(context.config["tolerances"]["sum_log_probability_rtol"]))
        return cached, record
    with clock.segment("inference"):
        values, checks = score_cores(policy, bank.index, batch_size=batch,
                                     label=f"{parent_record['id']} parent bank")
    scoring.require_finite(values, label=f"{parent_record['id']} parent bank scores")
    with clock.segment("io"):
        record = paths.write_shard(
            directory, "parent_scores",
            {"parent_log_probability": values,
             "draw_index": np.arange(bank.rows, dtype=np.int64)},
            {"record_kind": "parent_bank_scores", "identity": identity,
             "bank": bank.document(), "logit_checks": checks,
             "summary": scoring.log_probability_block(
                 values, label=f"{parent_record['id']} parent log probability"),
             "timings": clock.document()},
            order="bank draw order 0..N-1",
            logical_prefix=support.shard_directory("parent_banks", parent_record["id"]),
            run_root=context.run.run_root)
    return values, record


def verify_completed_scores(context, args, *, inventory, summary):
    """Verify a completed score stage and re-probe its parent caches on bounded rows.

    Three separate claims, checked separately:

    * every scored computation still has a completion-manifest-bound shard whose
      arrays, dtypes, shapes, order and container hash all reproduce;
    * every state the summary claims was scored is one the inventory says is
      primary and verified, so a reused summary cannot quietly shrink;
    * the same weights and the same route still produce the cached parent numbers,
      on a bounded sample of rows rather than all 10,000 -- enough to catch a
      changed model, a changed bank or a changed scoring path without paying for
      the whole stage again.
    """
    tolerances = context.config["tolerances"]
    expected = {record["id"] for record in inventory["records"]
                if record.get("status") == inventory_lib.STATUS_VERIFIED
                and (record.get("deduplication") or {}).get("is_primary")}
    scored = set(summary.get("checkpoints") or {})
    require(scored == expected,
            f"The completed checkpoint summary scores {len(scored)} computations but the committed "
            f"inventory declares {len(expected)} primary verified states; missing "
            f"{sorted(expected - scored)[:5]}, unexpected {sorted(scored - expected)[:5]}. A "
            "reused summary is not allowed to describe a different population.")
    for identifier in sorted(scored):
        support.reusable_shard(context, "scores", identifier, "sequence_scores")

    device = choose_device(args.device)
    policy = build_policy(context, device)
    reader = inventory_lib.CheckpointReader(lambda: policy.model)
    parents_by_seed = {int(record["seed"]): record["file_sha256"]
                       for record in inventory["records"]
                       if record["role"] == inventory_lib.ROLE_PARENT}
    banks = support.verify_banks(context, support.stage_documents(context), parents_by_seed)
    batch = int(context.config["inference"]["score_batch_size"])
    statistics = context.config["statistics"]
    probes, worst = [], 0.0
    for seed in context.seeds:
        bank = banks[seed]
        parent_record = _record_for(inventory, f"parent::policy_sft_seed{seed}")
        finished = support.reusable_shard(context, "parent_banks", parent_record["id"],
                                          "parent_scores")
        require(finished is not None,
                f"{parent_record['id']} has no completed parent-bank shard, but the checkpoint "
                "summary claims a completed score stage. The stage is not reusable; rerun it "
                "in a new revision directory.")
        cached, _ = finished
        rows = np.sort(np.random.default_rng(
            [int(statistics["bootstrap_seed"]), int(seed), 7]).choice(
                bank.rows, size=min(64, bank.rows), replace=False))
        load_state(context, reader, parent_record)
        values, _ = score_cores(policy, bank.index[rows], batch_size=batch,
                                label=f"{parent_record['id']} reuse probe")
        from smallAntibodyGen.experiments.her2_policy import compare_sum_log_probabilities
        block = compare_sum_log_probabilities(
            values, cached["parent_log_probability"][rows],
            label=f"{parent_record['id']} completed-stage reuse probe",
            atol=float(tolerances["sum_log_probability_atol"]),
            rtol=float(tolerances["sum_log_probability_rtol"]))
        worst = max(worst, float(block["max_abs_error"]))
        probes.append({"seed": int(seed), "rows": int(rows.size), "comparison": block})
    return {"parents_probed": len(probes), "probe_rows": int(min(64, banks[context.seeds[0]].rows)),
            "max_abs_error": worst, "probes": probes,
            "note": ("a completed stage is verified against its saved artifacts and a bounded "
                     "parent probe, not recomputed in full to refresh numbers that are already "
                     "bound to a completion manifest")}


def cmd_score(context, args):
    support.require_frozen_identity(context)
    inventory = paths.read_json(context.run.path(support.INVENTORY_JSON))
    binding = support.require_committed_inventory(context, inventory)
    # A completed score stage is verified and reused. Repeating 159 full-bank
    # forward passes to rewrite the same numbers is not a stronger check than
    # re-reading them against the completion manifest they were bound to; it is the
    # same check plus an opportunity to write something different.
    completed = support.completed_stage(context, {"summary": support.CHECKPOINTS_JSON,
                                                  "coverage": support.COVERAGE_JSON})
    if completed is not None and not args.recompute:
        probe = verify_completed_scores(context, args, inventory=inventory,
                                        summary=completed["summary"])
        print(f"score: reusing {completed['summary']['scored']} completed computations; "
              f"{probe['parents_probed']} parent caches re-probed within tolerance "
              f"(max {probe['max_abs_error']:.3g})", flush=True)
        return completed["summary"]
    statistics = context.config["statistics"]
    batch = int(context.config["inference"]["score_batch_size"])
    device = choose_device(args.device)
    policy = build_policy(context, device)
    reader = inventory_lib.CheckpointReader(lambda: policy.model)
    clock = paths.StageClock()

    from smallAntibodyGen.experiments.her2_data import load_split
    with clock.segment("io"):
        train_frame = load_split(context.root("raw").local_path, "train")
    parents_by_seed = {int(record["seed"]): record["file_sha256"]
                       for record in inventory["records"]
                       if record["role"] == inventory_lib.ROLE_PARENT}
    banks = support.verify_banks(context, support.stage_documents(context), parents_by_seed)

    verified = [record for record in inventory["records"]
                if record.get("status") == inventory_lib.STATUS_VERIFIED]
    primary = [record for record in verified
               if (record.get("deduplication") or {}).get("is_primary")]
    progress = context.progress("score", total=len(primary), every=1)
    checkpoints, controls, matrices, drops = {}, {}, {}, {}
    with progress.guard():
        for seed in context.seeds:
            bank = banks[seed]
            matrix = scoring.bootstrap_index_matrix(
                bank.rows, draws=int(statistics["bootstrap_draws"]),
                seed=int(statistics["bootstrap_seed"]) + int(seed))
            matrices[int(seed)] = matrix
            probe_rows = np.sort(np.random.default_rng(
                [int(statistics["bootstrap_seed"]), int(seed), 7]).choice(
                    bank.rows, size=min(64, bank.rows), replace=False))
            parent_record = _record_for(inventory, f"parent::policy_sft_seed{seed}")
            load_state(context, reader, parent_record)
            parent_lp, parent_record_shard = parent_cache(
                context, policy=policy, bank=bank, parent_record=parent_record, clock=clock,
                probe_rows=probe_rows)
            with clock.segment("analysis"):
                strata = support.strata_for_bank(context, bank, parent_lp, train_frame)
            for record in [item for item in primary if int(item.get("seed", -1)) == int(seed)]:
                identity = {"state_digest": record["state_digest_audit_computed"],
                            "file_sha256": record["file_sha256"],
                            "bank_sha256": bank.file_sha256,
                            "bank_order_sha256": bank.order_sha256,
                            "probability_convention":
                                inventory_lib.CORE_CONTRACT["probability_convention"],
                            "score_batch_size": batch, "device": device}
                # An interrupted stage restarts here: a state whose shard completed
                # is validated and its recorded numbers are reused, and only the
                # states that never finished are scored again.
                finished = support.reusable_shard(context, "scores", record["id"],
                                                  "sequence_scores", identity=identity)
                if finished is not None:
                    arrays, shard = finished
                    drops[record["id"]] = arrays["drop"]
                    block = dict(shard["statistics"])
                    block["shard"] = {"arrays": shard["arrays"], "container": shard["container"],
                                      "reused": "verified against the completion manifest"}
                    checkpoints[record["id"]] = block
                    if record["role"] == inventory_lib.ROLE_PARENT and "self_control" in block:
                        controls[record["id"]] = block["self_control"]
                    progress.advance(record["id"])
                    continue
                payload = load_state(context, reader, record)
                require(payload.state_digest_audit_computed
                        == record["state_digest_audit_computed"],
                        f"{record['id']}: the state digest changed since the inventory")
                with clock.segment("inference"):
                    policy_lp, policy_checks = score_cores(policy, bank.index, batch_size=batch,
                                                           label=record["id"])
                with clock.segment("analysis"):
                    drop = scoring.signed_drop(parent_lp, policy_lp, label=record["id"])
                    drops[record["id"]] = drop
                    block = support.score_one(
                        context, record=record, drop=drop, index_matrix=matrix, strata=strata,
                        historical=support.historical_join(record))
                    block["parent"] = {"parent_id": parent_record["id"],
                                       "bank_sha256": bank.file_sha256,
                                       "bank_order_sha256": bank.order_sha256,
                                       "rows": bank.rows}
                    block["policy_log_probability"] = scoring.log_probability_block(
                        policy_lp, label=f"{record['id']} policy log probability")
                    block["parent_log_probability"] = scoring.log_probability_block(
                        parent_lp, label=f"{parent_record['id']} parent log probability")
                    if record["role"] == inventory_lib.ROLE_PARENT:
                        controls[record["id"]] = scoring.self_control(
                            parent_lp, policy_lp,
                            atol=float(context.config["tolerances"]["self_control_atol"]),
                            rtol=float(context.config["tolerances"]["self_control_rtol"]),
                            label=f"{record['id']} full-bank self control")
                        block["self_control"] = controls[record["id"]]
                logical_prefix = support.shard_directory("scores", record["id"])
                directory = context.run.path(logical_prefix)
                with clock.segment("io"):
                    shard = paths.write_shard(
                        directory, "sequence_scores",
                        {"parent_log_probability": parent_lp,
                         "policy_log_probability": policy_lp, "drop": drop,
                         "draw_index": np.arange(bank.rows, dtype=np.int64)},
                        {"record_kind": "sequence_scores", "checkpoint_id": record["id"],
                         "identity": identity,
                         "logit_checks": policy_checks,
                         "statistics": block, "timings": clock.document()},
                        order="bank draw order 0..N-1", logical_prefix=logical_prefix,
                        run_root=context.run.run_root)
                block["shard"] = {"arrays": shard["arrays"], "container": shard["container"]}
                checkpoints[record["id"]] = block
                progress.advance(record["id"])

        # Same bank, same rows, same bootstrap indices on both sides.
        with clock.segment("analysis"):
            paired = support.paired_method_differences(
                drops, inventory_records=inventory["records"], statistics=statistics,
                matrices=matrices)
        summary = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "checkpoint_summaries",
                   "audit_id": context.config["audit_id"], "device": device,
                   "scored": len(checkpoints), "checkpoints": checkpoints,
                   "self_controls": controls,
                   "inventory_binding": binding,
                   "paired_method_differences": paired,
                   "per_seed_then_seeds": _equal_weight(checkpoints, inventory),
                   "support_versus_diversity": support.support_versus_diversity(
                       checkpoints, inventory["records"]),
                   "timings": clock.document(), "generated_at": paths.utc_now()}
        # Persist the science BEFORE the stage is allowed to report completion: a
        # stage marked completed whose summary was never written is the exact shape
        # of a partial run that later reads as a finished one.
        support.require_new_or_identical(context.run.path(support.CHECKPOINTS_JSON), summary,
                                         what="the checkpoint summary", context=context,
                                         kind="checkpoint_summaries")
        coverage_document = {
            "schema_version": paths.AUDIT_SCHEMA, "record_kind": "coverage",
            "inventory_coverage": inventory["coverage"],
            "scored": sorted(checkpoints),
            "scored_count": len(checkpoints),
            "expected_computations": inventory["deduplication"]["distinct_computations"],
            "scored_all_expected_computations":
                len(checkpoints) == inventory["deduplication"]["distinct_computations"],
            "paired_comparisons": paired["count"],
            "parent_banks": sorted(inventory["parent_banks"]),
            "self_controls": sorted(controls),
            "aliases_not_rescored": {
                record["id"]: (record.get("deduplication") or {}).get("scored_as")
                for record in verified
                if not (record.get("deduplication") or {}).get("is_primary")},
            "unverified": {record["id"]: record.get("status_reason")
                           for record in inventory["records"]
                           if record.get("status") != inventory_lib.STATUS_VERIFIED},
            "complete": bool(inventory["coverage"]["complete"]
                             and len(checkpoints)
                             == inventory["deduplication"]["distinct_computations"]
                             and len(controls) == len(context.seeds))}
        support.require_new_or_identical(context.run.path(support.COVERAGE_JSON),
                                         coverage_document, what="the coverage record",
                                         context=context, kind="coverage")
        support.record_timings(context, "score", clock)
    print(f"score: {len(checkpoints)} distinct computations scored on "
          f"{sum(bank.rows for bank in banks.values())} bank rows", flush=True)
    return summary


def _equal_weight(checkpoints, inventory):
    """Per-arm, per-budget equal-weight summaries over the three training seeds."""
    by_id = {record["id"]: record for record in inventory["records"]}
    groups = {}
    for identifier, block in checkpoints.items():
        record = by_id.get(identifier, {})
        key = f"{record.get('role')}::{record.get('arm_id')}::{record.get('nominal_budget_gpu_seconds')}"
        groups.setdefault(key, []).append((record.get("seed"), block))
    out = {}
    for key, members in sorted(groups.items()):
        out[key] = {
            "seeds": sorted(seed for seed, _ in members if seed is not None),
            "forward_kl": scoring.equal_weight_summary(
                [block["forward_kl"]["mean"] for _, block in members], label="forward KL"),
            "tenfold_fraction": scoring.equal_weight_summary(
                [block["tails"]["counts"]["tenfold"]["fraction"] for _, block in members],
                label="tenfold fraction"),
            "hundredfold_fraction": scoring.equal_weight_summary(
                [block["tails"]["counts"]["hundredfold"]["fraction"] for _, block in members],
                label="hundredfold fraction")}
    return out


# ---------------------------------------------------------------------------
# stage: ches
# ---------------------------------------------------------------------------

def cmd_ches(context, args):
    support.require_frozen_identity(context)
    inventory = paths.read_json(context.run.path(support.INVENTORY_JSON))
    support.require_committed_inventory(context, inventory)
    # As with the score stage: a completed CHES stage is verified against its
    # completion manifest and its shards, not recomputed to produce the same
    # numbers a second time.
    completed = support.completed_stage(context, {"summary": support.CHES_JSON})
    if completed is not None and not args.recompute:
        blocks = sorted(list(completed["summary"].get("parent") or {})
                        + list(completed["summary"].get("endpoints") or {}))
        for key in blocks:
            checkpoint_id, _, population_id = str(key).rpartition("::")
            support.reusable_shard(context, "ches", checkpoint_id, population_id)
        print(f"ches: reusing {len(blocks)} completed blocks, every shard verified against the "
              "completion manifest", flush=True)
        return completed["summary"]
    settings = context.config["ches"]
    batch = int(context.config["inference"]["ches_batch_size"])
    score_batch = int(context.config["inference"]["score_batch_size"])
    device = choose_device(args.device)
    policy = build_policy(context, device)
    reader = inventory_lib.CheckpointReader(lambda: policy.model)
    clock = paths.StageClock()

    with clock.segment("analysis"):
        pairs = support.build_pair_populations(context)
    committed = paths.read_json(
        context.repository_root / context.config["input_manifest"])["pair_populations"]
    require(pairs["identity"]["fixed_validation"] == committed["fixed_validation"],
            "The fixed validation pair identity differs from the frozen input manifest; the pair "
            "IDs and core digests are fixed before scoring, not after")
    for seed, identity in sorted(pairs["identity"]["v1_training"].items()):
        require(identity == committed["v1_training"][str(seed)],
                f"The v1 training pair identity for seed {seed} differs from the frozen manifest")

    v1 = [record for record in inventory["records"]
          if record["role"] == inventory_lib.ROLE_V1_ENDPOINT
          and record.get("status") == inventory_lib.STATUS_VERIFIED]
    require(v1, "No verified v1 endpoints to analyse")
    consumption = {}
    for seed in context.seeds:
        exposures = {record["id"]: _pair_exposures(record)
                     for record in v1 if int(record["seed"]) == int(seed)
                     and record["arm_id"] == "dpo"}
        consumption[str(seed)] = ches_lib.verify_consumption(
            pairs_used=int(context.config["populations"]["v1_training_pairs_per_seed"]),
            cycle=int(context.config["populations"]["v1_training_cycle"]),
            cycle_length=int(pairs["training"][seed]["cycle_length"]),
            endpoint_pair_exposures=exposures, label=f"seed {seed} v1 training pairs")

    # The cached extraction path is an optimization, so it is allowed only by the
    # parity the preflight actually recorded -- not by a config flag on its own.
    use_cached = str(settings.get("extraction_path", "full")) == "cached"
    if use_cached:
        recorded = paths.read_json(context.run.path(support.PREFLIGHT_JSON))
        require((recorded.get("ches_cached_vs_full") or {}).get("within_tolerance") is True,
                "The config asks for the cached CHES extraction path, but the recorded preflight "
                "does not show cached-versus-full parity within the declared tolerance. Run the "
                "preflight, or set extraction_path to 'full'.")

    populations = _ches_populations(context, pairs)
    # One unit of work per (state, population). ``len(populations)`` is the number
    # of SEEDS -- the mapping is keyed by seed -- so using it as the population
    # count made the denominator 81 where the real total is 54.
    per_seed = {seed: len(block) for seed, block in populations.items()}
    total = (sum(per_seed.values())
             + sum(per_seed[int(record["seed"])] for record in v1))
    progress = context.progress("ches", total=total, every=1)
    parent_blocks, endpoint_blocks, associations = {}, {}, {}
    with progress.guard():
        for seed in context.seeds:
            parent_record = _record_for(inventory, f"parent::policy_sft_seed{seed}")
            # An interrupted CHES stage restarts here. The parent state is loaded
            # only if at least one of its populations still has to be computed;
            # reusing every shard for a seed should cost no GPU work at all.
            reusable = {population_id: support.reusable_shard(
                            context, "ches", parent_record["id"], population_id,
                            identity=support.ches_identity(block, parent_record))
                        for population_id, block in sorted(populations[seed].items())}
            if any(value is None for value in reusable.values()):
                load_state(context, reader, parent_record)
            for population_id, block in sorted(populations[seed].items()):
                finished = reusable[population_id]
                if finished is not None:
                    arrays, _ = finished
                    scored = {name: arrays[name] for name in
                              ("ches", "chosen_log_probability", "rejected_log_probability")}
                else:
                    with clock.segment("inference"):
                        scored = ches_lib.ches_scores(
                            policy, block["chosen_index"], block["rejected_index"],
                            batch_size=batch, score_batch_size=score_batch, cached=use_cached,
                            label=f"parent {seed} {population_id}")
                with clock.segment("analysis"):
                    repeats = ches_lib.repeated_core_identities(block["chosen_index"])
                key = f"{parent_record['id']}::{population_id}"
                parent_blocks[key] = {
                    "seed": int(seed), "population_id": population_id,
                    "checkpoint_id": parent_record["id"], "temporal_label": "parent",
                    "identity": block["identity"],
                    "ches_summary": ches_lib.summarize(scored["ches"], label="parent CHES"),
                    "chosen_log_probability": ches_lib.summarize(
                        scored["chosen_log_probability"], label="parent chosen lp"),
                    "repeated_core_identities": {k: v for k, v in repeats.items()
                                                 if k != "per_row_repeat_count"},
                    "reused_completed_shard": finished is not None}
                if finished is None:
                    with clock.segment("io"):
                        _write_ches_shard(context, parent_record, population_id, scored,
                                          block, repeats,
                                          extra={"temporal_label": "parent",
                                                 "timings": clock.document()})
                progress.advance(key)

        parent_scores = {seed: {pid: _require_parent_ches(context, seed, pid)
                                for pid in sorted(populations[seed])}
                         for seed in context.seeds}

        earliest = _earliest_endpoints(v1)
        for record in sorted(v1, key=lambda item: item["id"]):
            seed = int(record["seed"])
            compute_ches = record["id"] in earliest and "earliest_v1_endpoint" in settings[
                "compute_ches_at"]
            reusable = {population_id: support.reusable_shard(
                            context, "ches", record["id"], population_id,
                            identity=support.ches_identity(block, record))
                        for population_id, block in sorted(populations[seed].items())}
            if any(value is None for value in reusable.values()):
                load_state(context, reader, record)
            for population_id, block in sorted(populations[seed].items()):
                finished = reusable[population_id]
                if finished is not None:
                    arrays, _ = finished
                    scored = {"chosen_log_probability": arrays["chosen_log_probability"],
                              "rejected_log_probability": arrays["rejected_log_probability"],
                              "ches": arrays.get("ches")}
                else:
                    with clock.segment("inference"):
                        if compute_ches:
                            scored = ches_lib.ches_scores(
                                policy, block["chosen_index"], block["rejected_index"],
                                batch_size=batch, score_batch_size=score_batch, cached=use_cached,
                                label=f"{record['id']} {population_id}")
                        else:
                            scored = dict(ches_lib.pair_log_probabilities(
                                policy, block["chosen_index"], block["rejected_index"],
                                batch_size=score_batch, label=f"{record['id']} {population_id}"),
                                ches=None)
                reference = parent_scores[seed][population_id]
                with clock.segment("analysis"):
                    displacement = (reference["chosen_log_probability"]
                                    - scored["chosen_log_probability"])
                    analysis = ches_lib.displacement_analysis(
                        parent_ches=reference["ches"], displacement=displacement,
                        chosen_index=block["chosen_index"],
                        rejected_index=block["rejected_index"],
                        parent_chosen_log_probability=reference["chosen_log_probability"],
                        deciles=int(context.config["statistics"]["ches_deciles"]),
                        quartiles=int(context.config["statistics"]["parent_lp_quartiles"]))
                key = f"{record['id']}::{population_id}"
                endpoint_blocks[key] = {
                    "checkpoint_id": record["id"], "seed": seed,
                    "population_id": population_id,
                    "arm_id": record["arm_id"],
                    "nominal_budget_gpu_seconds": record["nominal_budget_gpu_seconds"],
                    "temporal_label": ("earliest_endpoint" if record["id"] in earliest
                                       else "later_endpoint"),
                    "own_ches_computed": bool(compute_ches),
                    "own_ches_reason": (None if compute_ches else
                                        "CHES is computed at the parent and at the earliest "
                                        "endpoint per run; later endpoints supply displacement"),
                    "displacement": ches_lib.summarize(displacement, label="displacement"),
                    "control_availability": _control_status(context, record),
                    "reused_completed_shard": finished is not None}
                associations[key] = analysis
                if finished is None:
                    with clock.segment("io"):
                        _write_ches_shard(context, record, population_id,
                                          dict(scored, displacement=displacement), block, None,
                                          extra={"temporal_label":
                                                 endpoint_blocks[key]["temporal_label"],
                                                 "timings": clock.document()})
                progress.advance(key)

        with clock.segment("analysis"):
            increments, increment_gaps = _increments(context, v1, populations)

        expected_blocks = sum(per_seed[int(record["seed"])] for record in v1)
        summary = {"schema_version": paths.AUDIT_SCHEMA, "record_kind": "ches_summary",
                   "audit_id": context.config["audit_id"], "device": device,
                   "extraction": settings["extraction"],
                   "extraction_path": "cached" if use_cached else "full",
                   "extraction_path_justification": settings.get(
                       "extraction_path_justification"),
                   "positions": list(ches_lib.EXTRACTED_POSITIONS),
                   "pair_populations": pairs["identity"],
                   "consumption": consumption,
                   "parent": parent_blocks, "endpoints": endpoint_blocks,
                   "associations": associations, "increments": increments,
                   "increment_gaps": increment_gaps,
                   "coverage": {
                       "expected_parent_blocks": sum(per_seed.values()),
                       "parent_blocks": len(parent_blocks),
                       "expected_endpoint_blocks": expected_blocks,
                       "endpoint_blocks": len(endpoint_blocks),
                       "increments": len(increments), "gaps": len(increment_gaps),
                       "complete": bool(len(parent_blocks) == sum(per_seed.values())
                                        and len(endpoint_blocks) == expected_blocks
                                        and not increment_gaps)},
                   "missing_controls": dict(settings["missing_controls"]),
                   "spearman": {"ranks": "average", "p_values": None,
                                "p_value_reason": context.config["statistics"][
                                    "spearman_p_value_reason"]},
                   "timings": clock.document(), "generated_at": paths.utc_now()}
        support.require_new_or_identical(context.run.path(support.CHES_JSON), summary,
                                         what="the CHES summary", context=context,
                                         kind="ches_summary")
        support.record_timings(context, "ches", clock)
    print(f"ches: {len(parent_blocks)} parent blocks, {len(endpoint_blocks)} endpoint blocks",
          flush=True)
    return summary


#: Names the v1 exposure block may use for its consumed-pair count, in the order
#: they are trusted. An absent count is refused rather than read as zero: a default
#: of 0 would make the consumption check fail for the wrong reason, and a default of
#: "large enough" would make it pass without evidence.
PAIR_EXPOSURE_KEYS = ("pairs", "preference_pairs", "chosen_pairs", "chosen", "chosen_sequences")


def _pair_exposures(record):
    exposures = record.get("exposures") or {}
    for key in PAIR_EXPOSURE_KEYS:
        if exposures.get(key) is not None:
            return int(exposures[key])
    raise ValueError(
        f"{record['id']} records no consumed-pair count under any of {list(PAIR_EXPOSURE_KEYS)}; "
        f"its exposure block has {sorted(exposures)}. Whether the declared training pairs were "
        "actually consumed before this endpoint cannot be established, and it is not assumed.")


def _ches_populations(context, pairs):
    """``{seed: {population_id: block}}`` for the two declared CHES populations."""
    declared = list(context.config["ches"]["populations"])
    out = {}
    for seed in context.seeds:
        block = {}
        if "fixed_validation_pairs" in declared:
            block["fixed_validation_pairs"] = {
                "chosen_index": pairs["validation"]["chosen_index"],
                "rejected_index": pairs["validation"]["rejected_index"],
                "identity": pairs["identity"]["fixed_validation"]}
        if "v1_training_cycle0_prefix" in declared:
            training = pairs["training"][seed]
            block[f"v1_training_cycle{training['cycle']}_seed{seed}"] = {
                "chosen_index": training["chosen_index"],
                "rejected_index": training["rejected_index"],
                "identity": pairs["identity"]["v1_training"][str(seed)]}
        out[seed] = block
    return out


def _write_ches_shard(context, record, population_id, scored, block, repeats, *, extra):
    checkpoint_id = record["id"]
    logical_prefix = support.shard_directory("ches", checkpoint_id)
    directory = context.run.path(logical_prefix)
    arrays = {"chosen_log_probability": scored["chosen_log_probability"],
              "rejected_log_probability": scored["rejected_log_probability"],
              "pair_index": np.arange(len(scored["chosen_log_probability"]), dtype=np.int64)}
    if scored.get("ches") is not None:
        arrays["ches"] = scored["ches"]
    if scored.get("displacement") is not None:
        arrays["displacement"] = scored["displacement"]
    if repeats is not None:
        arrays["repeat_count"] = repeats["per_row_repeat_count"]
    return paths.write_shard(
        directory, population_id, arrays,
        dict({"record_kind": "ches_scores", "checkpoint_id": checkpoint_id,
              "population_id": population_id,
              "identity": support.ches_identity(block, record)}, **extra),
        order="declared pair order, frozen before scoring",
        logical_prefix=logical_prefix, run_root=context.run.run_root)


def _require_parent_ches(context, seed, population_id):
    """The parent's completed CHES arrays for one population, or a named failure."""
    finished = support.reusable_shard(context, "ches", f"parent::policy_sft_seed{int(seed)}",
                                      population_id)
    require(finished is not None,
            f"The parent CHES shard for seed {seed} / {population_id} is absent or unregistered, "
            "so no displacement can be measured against it. The reference is never replaced by "
            "another seed's or another population's parent.")
    return finished[0]


def _earliest_endpoints(records):
    """The lowest-budget verified endpoint of each v1 run: where early CHES is computed."""
    best = {}
    for record in records:
        key = (record["arm_id"], int(record["seed"]))
        budget = float(record["nominal_budget_gpu_seconds"])
        if key not in best or budget < best[key][0]:
            best[key] = (budget, record["id"])
    return {identifier for _, identifier in best.values()}


def _control_status(context, record):
    """Whether this endpoint has a matched continued-SFT control at the same budget."""
    budget = float(record["nominal_budget_gpu_seconds"])
    missing = context.config["ches"]["missing_controls"]
    key = f"v1_continued_sft_{int(budget)}"
    if key in missing:
        return {"matched": False, "reason": missing[key],
                "label": "uncontrolled; never described as control-matched"}
    return {"matched": True, "reason": None,
            "label": "a matched continued-SFT endpoint exists at this budget"}


def _increments(context, records, populations):
    """Early-checkpoint CHES against subsequent increments, with real controls.

    Three things this does that a "parent CHES versus later displacement" loop
    does not: it uses the CHES computed AT the early checkpoint (the spec's
    temporal claim), it subtracts the matched continued-SFT control's own
    displacement at the same budget where one exists, and it reports the
    combinations it could not form instead of skipping them silently.
    """
    out, gaps = {}, []
    by_run, by_budget = {}, {}
    for record in records:
        by_run.setdefault((record["arm_id"], int(record["seed"])), []).append(record)
        by_budget[(record["arm_id"], int(record["seed"]),
                   float(record["nominal_budget_gpu_seconds"]))] = record
    control_arm = context.config["ches"].get("control_arm", "continued_sft")
    for key, members in sorted(by_run.items()):
        members.sort(key=lambda item: float(item["nominal_budget_gpu_seconds"]))
        earliest = members[0]
        seed = int(earliest["seed"])
        for population_id in sorted(populations[seed]):
            block_pairs = populations[seed][population_id]
            early = _load_displacement(context, earliest["id"], population_id)
            early_ches = _load_ches(context, earliest["id"], population_id)
            parent_ches = _load_parent_ches(context, seed, population_id)
            parent_chosen = _load_parent_chosen(context, seed, population_id)
            if early is None:
                gaps.append({"label": f"{key[0]}_seed{key[1]}::{population_id}",
                             "reason": (f"no completed displacement shard for the earliest "
                                        f"endpoint {earliest['id']}")})
                continue
            for later in members[1:]:
                budget = float(later["nominal_budget_gpu_seconds"])
                label = (f"{key[0]}_seed{key[1]}::{population_id}::"
                         f"{int(float(earliest['nominal_budget_gpu_seconds']))}"
                         f"->{int(budget)}")
                shifted = _load_displacement(context, later["id"], population_id)
                if shifted is None:
                    gaps.append({"label": label,
                                 "reason": f"no completed displacement shard for {later['id']}"})
                    continue
                block = ches_lib.increment_analysis(
                    early, shifted, label=label, early_ches=early_ches,
                    early_checkpoint={"id": earliest["id"],
                                      "file_sha256": earliest["file_sha256"],
                                      "state_digest": earliest.get(
                                          "state_digest_audit_computed"),
                                      "nominal_budget_gpu_seconds":
                                          earliest["nominal_budget_gpu_seconds"]},
                    chosen_index=block_pairs["chosen_index"],
                    rejected_index=block_pairs["rejected_index"],
                    parent_chosen_log_probability=parent_chosen,
                    parent_ches=parent_ches,
                    deciles=int(context.config["statistics"]["ches_deciles"]),
                    quartiles=int(context.config["statistics"]["parent_lp_quartiles"]))
                block["control"] = _matched_control(
                    context, later, control_arm=control_arm, by_budget=by_budget,
                    population_id=population_id, treated=shifted, block_pairs=block_pairs)
                out[label] = block
    return out, gaps


def _matched_control(context, record, *, control_arm, by_budget, population_id, treated,
                     block_pairs):
    """The matched same-seed, same-budget control's *measurements*, or a named absence."""
    seed, budget = int(record["seed"]), float(record["nominal_budget_gpu_seconds"])
    availability = _control_status(context, record)
    control = by_budget.get((control_arm, seed, budget))
    if control is None:
        return ches_lib.unavailable_control(
            label=f"{control_arm}_seed{seed}_budget{int(budget)}", budget=budget,
            reason=(availability.get("reason")
                    or f"no {control_arm} endpoint exists at {budget:g} GPU seconds for seed "
                       f"{seed}"))
    values = _load_displacement(context, control["id"], population_id)
    if values is None:
        return ches_lib.unavailable_control(
            label=control["id"], budget=budget,
            reason=("the matched control exists but has no completed displacement shard for "
                    f"{population_id}"))
    return ches_lib.matched_control_comparison(
        treated, values, label=f"{record['id']}__minus__{control['id']}",
        treated_id=record["id"], control_id=control["id"], budget=budget,
        chosen_index=block_pairs["chosen_index"], rejected_index=block_pairs["rejected_index"])


def _load_array(context, checkpoint_id, population_id, name):
    finished = support.reusable_shard(context, "ches", checkpoint_id, population_id)
    if finished is None:
        return None
    return finished[0].get(name)


def _load_displacement(context, checkpoint_id, population_id):
    return _load_array(context, checkpoint_id, population_id, "displacement")


def _load_ches(context, checkpoint_id, population_id):
    return _load_array(context, checkpoint_id, population_id, "ches")


def _load_parent_ches(context, seed, population_id):
    return _load_array(context, f"parent::policy_sft_seed{int(seed)}", population_id, "ches")


def _load_parent_chosen(context, seed, population_id):
    return _load_array(context, f"parent::policy_sft_seed{int(seed)}", population_id,
                       "chosen_log_probability")


# ---------------------------------------------------------------------------
# stage: decide and report
# ---------------------------------------------------------------------------

def audit_requirements(context, *, inventory, checkpoints_document, coverage, ches_document,
                       verification):
    """Evaluate every completion requirement from artifacts, not from exit codes."""
    seeds = context.seeds
    banks = inventory.get("parent_banks") or {}
    controls = (checkpoints_document or {}).get("self_controls") or {}
    manifest_path = context.repository_root / context.config["input_manifest"]
    manifest = paths.read_json(manifest_path) if manifest_path.is_file() else {}
    pair_populations = manifest.get("pair_populations") or {}
    ches_coverage = (ches_document or {}).get("coverage") or {}
    # Only the stages that produce scientific artifacts. ``decide`` and ``report``
    # are transient while they are running -- the decision record is built inside
    # the decide stage, so it would record its own status as "running" and then
    # disagree with a decision rebuilt after the report completed. A completion
    # requirement is about artifacts, and those two produce none that this record
    # is a function of.
    required_stages = ("inventory", "prepare", "preflight", "score", "ches")
    stages = {stage: (context.stage_status(stage) or {}).get("status")
              for stage in required_stages}
    return inventory_lib.audit_completion({
        "inventory_coverage": {
            "satisfied": bool(inventory["coverage"]["complete"]),
            "detail": inventory["coverage"]["shortfalls"]},
        "state_verification": {
            "satisfied": (inventory["coverage"]["verified_total"]
                          == inventory["coverage"]["expected"].get("total")),
            "detail": {"verified": inventory["coverage"]["verified_total"],
                       "expected": inventory["coverage"]["expected"].get("total")}},
        "parent_banks": {
            "satisfied": len(banks) == len(seeds),
            "detail": {"verified_banks": sorted(banks), "declared_seeds": seeds}},
        "scored_computations": {
            "satisfied": bool((coverage or {}).get("scored_all_expected_computations")),
            "detail": {"scored": (coverage or {}).get("scored_count"),
                       "expected": (coverage or {}).get("expected_computations")}},
        "alias_resolution": {
            "satisfied": not (inventory.get("reused_controls") or {}).get("unmatched"),
            "detail": (inventory.get("reused_controls") or {}).get("unmatched")},
        "pair_populations": {
            "satisfied": bool(pair_populations.get("fixed_validation")
                              and len(pair_populations.get("v1_training") or {}) == len(seeds)),
            "detail": sorted(pair_populations)},
        "ches_populations": {
            "satisfied": bool(ches_coverage.get("complete")),
            "detail": ches_coverage or "the CHES stage produced no coverage block"},
        "numerical_controls": {
            "satisfied": (len(controls) == len(seeds)
                          and all(block.get("within_tolerance") for block in controls.values())),
            "detail": {name: block.get("within_tolerance") for name, block in controls.items()}},
        "decision_inputs": {
            "satisfied": bool(checkpoints_document),
            "detail": {"scored": (checkpoints_document or {}).get("scored")}},
        "stage_status": {
            "satisfied": (all(stages.get(name) == "completed" for name in required_stages)
                          and bool((verification or {}).get("immutable", False))),
            "detail": {"stages": {name: stages.get(name) for name in required_stages},
                       "immutable": (verification or {}).get("immutable")}}})


def build_decision(context, *, verification=None):
    support.require_frozen_identity(context)
    inventory = paths.read_json(context.run.path(support.INVENTORY_JSON))
    support.require_committed_inventory(context, inventory)
    summary_path = context.run.path(support.CHECKPOINTS_JSON)
    require(summary_path.is_file(),
            f"{summary_path} is absent; the decision is a function of the scored artifacts")
    checkpoints_document = paths.read_json(summary_path)
    checkpoints = checkpoints_document["checkpoints"]
    coverage = paths.read_json(context.run.path(support.COVERAGE_JSON))
    ches_path = context.run.path(support.CHES_JSON)
    ches_document, ches_summary = None, None
    if ches_path.is_file():
        ches_document = paths.read_json(ches_path)
        ches_summary = {"parent_blocks": len(ches_document.get("parent") or {}),
                        "endpoint_blocks": len(ches_document.get("endpoints") or {}),
                        "increments": len(ches_document.get("increments") or {}),
                        "increment_gaps": ches_document.get("increment_gaps"),
                        "coverage": ches_document.get("coverage"),
                        "missing_controls": ches_document.get("missing_controls")}
    completion = audit_requirements(
        context, inventory=inventory, checkpoints_document=checkpoints_document,
        coverage=coverage, ches_document=ches_document,
        verification=verification if verification is not None else support.verify_outputs(context))
    inputs = support.decision_inputs(checkpoints, settings=context.config["decision"],
                                     inventory_records=inventory["records"])
    decision = scoring.decision_record(inputs, settings=context.config["decision"],
                                       coverage=coverage, completion=completion,
                                       ches_summary=ches_summary)
    decision.update(schema_version=paths.AUDIT_SCHEMA, record_kind="decision",
                    audit_id=context.config["audit_id"], generated_at=paths.utc_now(),
                    frozen_commit=support.read_freeze_marker(context)["git"]["commit"])
    return decision


def cmd_decide(context, args):
    progress = context.progress("decide", total=1)
    with progress.guard():
        decision = build_decision(context)
        support.require_new_or_identical(context.run.path(support.DECISION_JSON), decision,
                                         what="the decision record", context=context,
                                         kind="decision")
        progress.advance("decision")
    print(f"decide: {decision['outcome']}", flush=True)
    for method, block in sorted(decision["methods"].items()):
        print(f"  {method}: {block['seeds_crossing']}/{block['seeds_usable']} seeds crossing "
              f"-> {block['outcome']}", flush=True)
    for reason in decision.get("blocking") or []:
        print(f"  BLOCKING {reason}", flush=True)
    return decision


def cmd_report(context, args):
    support.require_frozen_identity(context)
    inventory = paths.read_json(context.run.path(support.INVENTORY_JSON))
    support.require_committed_inventory(context, inventory)
    scored = paths.read_json(context.run.path(support.CHECKPOINTS_JSON))
    checkpoints = scored["checkpoints"]
    coverage = paths.read_json(context.run.path(support.COVERAGE_JSON))["inventory_coverage"]
    ches_path = context.run.path(support.CHES_JSON)
    ches_summary = paths.read_json(ches_path) if ches_path.is_file() else None
    decision_path = context.run.path(support.DECISION_JSON)
    if not decision_path.is_file():
        support.require_new_or_identical(decision_path, build_decision(context),
                                         what="the decision record", context=context,
                                         kind="decision")
    decision = paths.read_json(decision_path)
    clock = paths.StageClock()
    progress = context.progress("report", total=2)
    with progress.guard():
        with clock.segment("analysis"):
            figures = (support.render_figures(context, checkpoints=checkpoints,
                                              inventory=inventory, ches_summary=ches_summary)
                       if context.config["reporting"]["figures"] else
                       {"disabled": {"written": False, "name": "figures",
                                     "reason": "disabled in the config"}})
        progress.advance("figures")
        text = support.render_report(context, inventory=inventory, checkpoints=checkpoints,
                                     ches_summary=ches_summary, decision=decision,
                                     coverage=coverage, timings=support.stage_timings(context),
                                     figures=figures,
                                     paired=scored.get("paired_method_differences"))
        with clock.segment("io"):
            # Immutable, and bound into the completion manifest like any other
            # finished artifact: a second report run verifies these bytes instead
            # of overwriting a deliverable somebody may already have cited.
            support.require_new_or_identical_text(context.run.path(support.REPORT_MD), text,
                                                  what="the report", context=context,
                                                  kind="report")
        progress.advance("report")
    support.record_timings(context, "report", clock)
    verification = support.verify_outputs(context)
    paths.write_json(context.run.path("verification.json"), verification)
    if args.publish:
        publication = support.publish_deliverable(
            context,
            render=lambda plan: support.render_report(
                context, inventory=inventory, checkpoints=checkpoints, ches_summary=ches_summary,
                decision=decision, coverage=coverage, timings=support.stage_timings(context),
                figures=figures, paired=scored.get("paired_method_differences"),
                publication=plan),
            decision=decision, coverage=coverage, checkpoints_document=scored,
            ches_document=ches_summary, verification=verification, figures=figures)
        print(f"report: published {publication['file_count']} files under "
              f"{context.config['publication']['root']}", flush=True)
    stages = {stage: (context.stage_status(stage) or {}).get("status")
              for stage in ("inventory", "prepare", "preflight", "score", "ches", "decide",
                            "report")}
    # Completion is a property of the evidence, not of the process: every declared
    # stage completed, every required artifact present and verified, and the
    # immutability check clean. A stage that exited is none of those.
    requirements = audit_requirements(
        context, inventory=inventory,
        checkpoints_document=scored,
        coverage=paths.read_json(context.run.path(support.COVERAGE_JSON)),
        ches_document=ches_summary, verification=verification)
    complete = bool(requirements["complete"]
                    and all(stages[name] == "completed" for name in stages))
    if complete:
        support.require_new_or_identical(context.run.path(support.COMPLETE_JSON), {
            "schema_version": paths.AUDIT_SCHEMA, "record_kind": "audit_complete",
            "audit_id": context.config["audit_id"], "completed_at": paths.utc_now(),
            "frozen_commit": support.read_freeze_marker(context)["git"]["commit"],
            "stages": stages, "requirements": requirements,
            "verification": {"immutable": True,
                             "shards_checked": verification["shards_checked"]},
            "decision_outcome": decision["outcome"],
            "report": {"path": context.run.logical(support.REPORT_MD),
                       "sha256": paths.sha256_text(text)},
            "note": ("written only after every declared stage completed, every declared "
                     "requirement was satisfied and the immutability verification passed. A "
                     "process that merely exited is not a complete audit.")},
            what="the completion marker", context=context, kind="audit_complete")
    else:
        print(f"report: written, but the audit is NOT complete: stages={stages}, "
              f"unmet={requirements['unmet']}, immutable={verification['immutable']}", flush=True)
    print(f"report: {context.run.logical(support.REPORT_MD)}"
          + (" (published)" if args.publish else ""), flush=True)
    return text


def cmd_verify(context, args):
    progress = context.progress("verify", total=1)
    with progress.guard():
        verification = support.verify_outputs(context)
        # Deliberately NOT immutable: a verification record is a fresh observation
        # of the artifacts, and the artifacts it observes grow as later stages
        # complete. What has to be immutable is what it checks, not the checking.
        paths.write_json(context.run.path("verification.json"), verification)
        progress.advance("verification")
    print(f"verify: {verification['shards_checked']} shards, "
          f"immutable={verification['immutable']}", flush=True)
    for problem in verification["problems"]:
        print(f"  PROBLEM {problem}", flush=True)
    require(verification["immutable"],
            "Completed outputs are not immutable; see verification.json. A rerun that would "
            "change content or timings belongs in a new revision directory.")
    return verification


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

COMMANDS = {"inventory": cmd_inventory, "prepare": cmd_prepare, "preflight": cmd_preflight,
            "freeze": cmd_freeze, "score": cmd_score, "ches": cmd_ches, "decide": cmd_decide,
            "report": cmd_report, "verify": cmd_verify}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="audit_her2_support.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", nargs="?", default=None,
                        choices=sorted(set(support.STAGES) | set(support.STAGE_ALIASES)),
                        help="audit stage to run (aliases: "
                             + ", ".join(f"{k}={v}" for k, v in
                                         sorted(support.STAGE_ALIASES.items())) + ")")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="audit configuration (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="run root override; defaults to the config's run_root")
    parser.add_argument("--guarded-root", default=None,
                        help="local directory holding the guarded campaign output")
    parser.add_argument("--original-root", default=None,
                        help="local directory holding the original v1 post-training output")
    parser.add_argument("--raw-root", default=None, help="local directory holding the raw release")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--inventory-only", action="store_true",
                        help="alias for the inventory stage")
    parser.add_argument("--verify-only", action="store_true",
                        help="alias for the verify stage: re-check completed outputs only")
    parser.add_argument("--publish", action="store_true",
                        help="report stage: also write the tracked reference/ report, small "
                             "summaries and figures")
    parser.add_argument("--recompute", action="store_true",
                        help="score/ches stages: recompute instead of verifying and reusing a "
                             "completed stage. The completed artifacts are still immutable, so "
                             "this fails loudly if the numbers differ rather than overwriting "
                             "them")
    return parser


def resolve_stage(args):
    chosen = [name for name, flag in (("inventory", args.inventory_only),
                                      ("verify", args.verify_only)) if flag]
    if args.stage is not None:
        chosen.append(support.STAGE_ALIASES.get(args.stage, args.stage))
    unique = sorted(set(chosen))
    require(len(unique) == 1,
            "Choose exactly one stage: a positional stage, --inventory-only or --verify-only "
            f"(got {unique or 'none'})")
    return unique[0]


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    stage = resolve_stage(args)
    context = support.resolve_context(
        ROOT, config_path=ROOT / args.config, run_root=args.output,
        root_overrides={"guarded": args.guarded_root, "original": args.original_root,
                        "raw": args.raw_root})
    print(f"audit stage {stage} -> {context.run.logical_run_root}", flush=True)
    COMMANDS[stage](context, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
