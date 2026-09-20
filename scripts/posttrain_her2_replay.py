"""HER2 parent replay: prepare, preflight, freeze, banks, fit, status, report, verify, publish.

Phase D of ``specs/her2_support_preservation_plan.md``, conditional on the
completed and published support audit. Stages run in protocol order and each one
refuses to start on evidence the previous one did not produce::

    prepare    resolve every existing input and the common ordered training stream
    preflight  native inference-only parity on the real parents, and a synthetic-
               weight gradient control on the native architecture. No optimizer
               step ever touches a HER2 parent before the freeze.
    freeze     verify the committed sources, config, evidence and inputs; write
               the training_spec_frozen marker naming the real HEAD
    banks      generate the replay and monitoring banks and the frozen teacher
               caches, and bind their digests to that marker
    fit        run the declared 36-trajectory queue under one exclusive writer
    status     what the artifacts say, including what has not happened
    report     the preservation-versus-ranking tables, deltas and figures
    verify     re-check completed outputs for immutable content and timings
    publish    copy the report, the small tables and the figures into reference/

Examples::

    python scripts/posttrain_her2_replay.py --help
    python scripts/posttrain_her2_replay.py prepare
    python scripts/posttrain_her2_replay.py preflight
    python scripts/posttrain_her2_replay.py status

This CLI never commits and never pushes. The freeze verifies a commit somebody
else made.

The shard container envelope written under the run root carries
``her2-support-audit/1``: that is the *artifact format* tag of
``her2_support_paths``, which writes every shard, progress file and completion
manifest in this repository. The scientific schema of these payloads is
``her2-parent-replay/1`` and each record says so in its own fields.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from smallAntibodyGen.experiments import her2_replay as replay_lib            # noqa: E402
from smallAntibodyGen.experiments import her2_replay_banks as banks_lib       # noqa: E402
from smallAntibodyGen.experiments import her2_replay_campaign as campaign_lib  # noqa: E402
from smallAntibodyGen.experiments import her2_replay_report as report_lib     # noqa: E402
from smallAntibodyGen.experiments import her2_replay_spec as spec             # noqa: E402
from smallAntibodyGen.experiments import her2_replay_streams as streams_lib   # noqa: E402
from smallAntibodyGen.experiments import her2_support as support              # noqa: E402
from smallAntibodyGen.experiments import her2_support_paths as paths          # noqa: E402
from smallAntibodyGen.experiments.her2_runtime import require                 # noqa: E402

DEFAULT_CONFIG = "configs/experiments/her2_parent_replay.json"
STAGES = ("prepare", "preflight", "freeze", "banks", "fit", "status", "report", "verify",
          "publish")

LOCK_FILE = "campaign.lock"
HEARTBEAT_FILE = "campaign_heartbeat.json"
REPORT_MD = "report/her2-parent-replay.md"


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------

def draw_offset(name):
    """A stable, stage-independent seed offset for one name. Never ``hash()``.

    ``hash()`` is salted per process, so a seed derived from it would draw
    different sequences on two runs and no record could be re-derived.
    """
    return int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16) % 100000


def choose_device(context, requested):
    """The declared device. ``--device`` may confirm it and may not change it.

    A stage that ran on a different device than the config declares would still
    write its numbers into artifacts that cite the config's native probe and the
    frozen runtime contract. ``device`` is a frozen immutable field, so the flag is
    accepted only when it names the declared device -- confirming, never
    overriding.
    """
    import torch
    declared = str(context.config["inference"]["device"])
    if requested in (None, "auto"):
        requested = declared
    require(requested in ("cpu", "cuda"), f"Unknown device {requested!r}")
    require(requested == declared,
            f"--device {requested} would run this stage on a different device than the config "
            f"declares ({declared}). Device is a frozen immutable field: preflight, banks and the "
            "fit must all run where the recorded numbers say they ran, and a stage that quietly "
            "moved would still cite the declared device's probe. Change the config and re-review, "
            "or run where it says.")
    if requested == "cuda":
        require(torch.cuda.is_available(),
                "This campaign is declared on CUDA and no device is present. The declared screen "
                "is a measured GPU screen; it is not silently moved to CPU.")
    return requested


def build_context_data(context, *, device, need_generation=False):
    """Splits, eligible populations, the fixed validation pairs and the scaffold.

    Exactly the inherited construction: the same eligibility rule, the same
    distance-matched pairing and the same fixed validation pair set the completed
    campaign used. The expected counts are asserted before anything is fitted, so a
    changed release stops here rather than producing a differently sized "fixed"
    population under the same name.
    """
    from smallAntibodyGen.experiments import her2_data as data
    from smallAntibodyGen.experiments import her2_eval as evaluation
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    from smallAntibodyGen.experiments import her2_preferences as preferences
    from smallAntibodyGen.experiments.her2_runtime import set_cpu_threads
    set_cpu_threads(int(context.config["inference"]["cpu_threads"]))
    raw = context.root("raw").local_path
    expected = context.config["expected_counts"]
    train = data.load_split(raw, "train")
    val = data.load_split(raw, "val")
    train_population = preferences.build_population(train, "train")
    val_population = preferences.build_population(val, "val")
    require(int(train_population.chosen_index.shape[0]) == int(expected["train_chosen"]),
            f"Eligible training chosen rows {train_population.chosen_index.shape[0]} != "
            f"{expected['train_chosen']}")
    require(int(train_population.rejected_index.shape[0]) == int(expected["train_rejected"]),
            "Eligible training rejected rows changed")
    require(int(val_population.chosen_index.shape[0]) == int(expected["val_chosen"]),
            "Eligible validation chosen rows changed")
    reference_index = np.concatenate([train_population.chosen_index,
                                      train_population.rejected_index], axis=0)
    require(reference_index.shape[0] == int(expected["reference_rows"]),
            f"The reference population is {reference_index.shape[0]} rows, expected "
            f"{expected['reference_rows']}")
    val_pairing = preferences.PreferencePairing(
        val_population, seed=int(context.config["pairing"]["validation_seed"]))
    val_pairs = val_pairing.fixed_validation_pairs(context.config["pairing"]["validation_pairs"])
    require(int(val_pairs["pairs"]) == int(expected["val_chosen"]),
            f"The gate must see every fixed validation pair: {val_pairs['pairs']} != "
            f"{expected['val_chosen']}. There is no monitor subset in this protocol.")
    block = {"train": train, "val": val,
             "train_population": train_population, "val_population": val_population,
             "reference_index": reference_index,
             "reference_rejected_rows": (train_population.chosen_index.shape[0]
                                         + np.arange(train_population.rejected_index.shape[0])),
             "val_pairs": val_pairs,
             "scaffold": data.load_scaffold(raw), "vocab": policy_lib.load_vocab(raw),
             "raw_root": raw, "device": device}
    if need_generation:
        train_index = data.encode_cores(train.seq)
        train_labels = (train["class"] == data.POSITIVE_CLASS).to_numpy().astype(np.float64)
        split_of, class_of = data.labelled_lookup({"train": train, "val": val})
        # Read once. ``test_sequences`` passes usecols=['seq'], so the reserved
        # split's SEQUENCES cross the wire and its labels do not; the file is pinned
        # by digest in the input manifest as a sequence-only input, because it is a
        # file this screen genuinely opens. It feeds the inherited overlap/novelty
        # diagnostic and nothing else -- no reserved label enters any number here.
        test_cores = data.test_sequences(raw)
        for core in test_cores:
            split_of.setdefault(core, "test")
        block.update(
            train_index=train_index, train_labels=train_labels,
            val_index=data.encode_cores(val.seq),
            val_positive=(val["class"] == data.POSITIVE_CLASS).to_numpy(),
            val_cores=val.seq.to_numpy(dtype=str),
            catalogs={"train": set(train.seq), "val": set(val.seq),
                      "test": set(test_cores)},
            split_of=split_of, class_of=class_of,
            training_diversity_reference=evaluation.diversity_reference(
                train_index[train_labels.astype(bool)]),
            val_strata=data.nearest_training_labels(
                data.encode_cores(val.seq), train_index, train_labels,
                max_distance=int(context.config["evaluation"]["max_train_distance"])).strata(),
            inherited_diversity_reference=inherited_diversity_references(context))
    return block


def inherited_diversity_references(context):
    """The relative diversity floor, from the completed campaign's own parent draws.

    Deliberately not the freshly sampled monitoring bank. The monitoring bank
    supplies preservation diagnostics that this campaign looks at while it runs;
    letting it also define the diversity floor would make an endpoint's eligibility
    depend on a quantity the run had been watching, and would silently change what
    an "eligible" endpoint means relative to the completed campaign.
    """
    from smallAntibodyGen.experiments import her2_eval as evaluation
    out = {}
    for seed, entry in sorted(inherited_bank_records(context).items()):
        index, _ = banks_lib.historical_reference_document(
            context.root("guarded").path(entry["logical_path"]),
            expected_sha256=entry["sha256"],
            expected_rows=int(context.config["banks"]["monitor_rows"]),
            label=entry["logical_path"])
        out[int(seed)] = evaluation.diversity_reference(index)
    return out


def parent_records(context):
    """The three selected initial-SFT parents, by seed, from the published audit manifest."""
    manifest, _ = spec.audit_input_manifest(context.repository_root, context.config)
    found = {}
    for logical, entry in sorted((manifest.get("inputs") or {}).items()):
        if entry.get("kind") != "selected_parent" or entry.get("root") != "original":
            continue
        for seed in context.seeds:
            if f"seed{seed}" in entry["logical_path"]:
                found[int(seed)] = {"logical": logical, "logical_path": entry["logical_path"],
                                    "sha256": entry["sha256"], "seed": int(seed),
                                    "parent_id": f"parent::policy_sft_seed{seed}"}
    missing = sorted(seed for seed in context.seeds if seed not in found)
    require(not missing,
            f"The published audit input manifest names no selected parent for seeds {missing}. "
            "A parent is resolved from that manifest, never from a directory listing.")
    return found


def inherited_bank_records(context):
    """The persisted historical parent draw banks, by seed, from the published manifest."""
    manifest, _ = spec.audit_input_manifest(context.repository_root, context.config)
    found = {}
    for entry in (manifest.get("inputs") or {}).values():
        if entry.get("kind") != "parent_bank" or entry.get("root") != "guarded":
            continue
        for seed in context.seeds:
            if f"seed{seed}" in entry["logical_path"]:
                found[int(seed)] = dict(entry)
    missing = sorted(seed for seed in context.seeds if seed not in found)
    require(not missing,
            f"The published audit input manifest names no persisted parent draw bank for seeds "
            f"{missing}. The inherited relative diversity floor is measured against those exact "
            "files and against nothing else.")
    return found


def load_parent_policy(context, parent, *, device, data_block):
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    target = context.root("original").path(parent["logical_path"])
    observed = paths.sha256_file(target)
    require(observed == parent["sha256"],
            f"{parent['logical']} hashes {observed}; the published audit manifest recorded "
            f"{parent['sha256']}")
    model = policy_lib.architecture_model(data_block["raw_root"], device=device)
    restored = policy_lib.load_checkpoint(target, model, device=device)
    policy = policy_lib.CorePolicy.from_prefix(model, data_block["scaffold"].prefix,
                                               data_block["vocab"], device=device)
    return policy, {"parent_id": parent["parent_id"], "seed": parent["seed"],
                    "file_sha256": observed, "state_sha256": restored["state_sha256"],
                    "logical": parent["logical"]}


def resolve_streams(context, data_block):
    """The common ordered chosen/rejected stream for every arm, per parent seed."""
    from smallAntibodyGen.experiments import her2_preferences as preferences
    screen = context.config["screen"]
    exposures = int(max(screen["endpoint_updates"])) * int(screen["chosen_per_update"])
    out = {}
    for seed in context.seeds:
        pairing_seed = int(context.config["pairing"]["seed_base"]) + int(seed)
        pairing = preferences.PreferencePairing(data_block["train_population"], seed=pairing_seed)
        out[int(seed)] = streams_lib.resolve_task_stream(
            pairing, seed=seed, exposures=exposures,
            batch_rows=int(screen["chosen_per_update"]), pairing_seed=pairing_seed)
    return out


def cadence_for(context):
    return streams_lib.UpdateCadence(
        first_update=int(context.config["gate"]["first_update"]),
        interval=int(context.config["gate"]["update_interval"]),
        endpoints=context.endpoint_updates)


def replay_orders(context):
    screen = context.config["screen"]
    exposures = int(max(screen["endpoint_updates"])) * int(screen["chosen_per_update"])
    return {int(seed): streams_lib.replay_order(
        bank_rows=int(context.config["banks"]["replay_rows"]), exposures=exposures,
        seed=int(context.config["seeds"]["replay_order"]), parent_seed=int(seed))
        for seed in context.seeds}


def declared_seed_block(context):
    seeds = context.config["seeds"]
    flat = {"replay_order": seeds["replay_order"], "generation_base": seeds["generation_base"],
            "cache_probe": seeds["cache_probe"],
            "synthetic_gradient_control": seeds["synthetic_gradient_control"]}
    for purpose in ("replay_draw", "monitor_draw"):
        for seed, value in seeds[purpose].items():
            flat[f"{purpose}_{seed}"] = value
    return streams_lib.require_disjoint_seeds(flat, label="replay campaign seeds")


# ---------------------------------------------------------------------------
# stage: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(context, args):
    """Resolve existing inputs and the common ordered stream. Nothing is sampled here."""
    clock = paths.StageClock()
    progress = context.progress("prepare", total=5)
    screen = context.config["screen"]
    with progress.guard():
        with clock.segment("analysis"):
            data_block = build_context_data(context, device="cpu")
            seed_block = declared_seed_block(context)
        progress.advance("populations")
        with clock.segment("analysis"):
            streams = resolve_streams(context, data_block)
            orders = replay_orders(context)
            cadence = cadence_for(context)
        progress.advance("streams")
        with clock.segment("io"):
            for seed, stream in sorted(streams.items()):
                prefix = f"streams/seed{seed}"
                paths.write_shard(
                    context.run.path(prefix), "task_stream",
                    {"chosen_rows": stream.chosen_rows, "rejected_rows": stream.rejected_rows,
                     "cycle_of_position": stream.cycle_of_position,
                     "replay_order": orders[seed]},
                    {"record_kind": "resolved_task_stream", "campaign_schema": spec.REPLAY_SCHEMA,
                     "campaign_id": context.config["campaign_id"],
                     "identity": stream.document(),
                     "replay_order": streams_lib.replay_order_document(
                         orders[seed], bank_rows=int(context.config["banks"]["replay_rows"]),
                         seed=int(context.config["seeds"]["replay_order"]), parent_seed=seed,
                         batch_rows=int(screen["chosen_per_update"])),
                     "timings": clock.document()},
                    order="stream position 0..N-1", logical_prefix=prefix,
                    run_root=context.run.run_root)
        progress.advance("stream shards")
        with clock.segment("analysis"):
            manifest = build_input_manifest(context, data_block, streams, orders, cadence,
                                            seed_block)
        progress.advance("input manifest")
        with clock.segment("io"):
            evidence = context.repository_root / context.config["evidence_root"]
            support.require_new_or_identical(
                context.repository_root / context.config["input_manifest"], manifest,
                what="the committed input manifest")
            support.require_new_or_identical(
                evidence / "streams.json", manifest["streams"],
                what="the committed stream document")
            support.require_new_or_identical(
                context.run.path(spec.RESOLVED_CONFIG_JSON),
                {"schema_version": spec.REPLAY_SCHEMA, "record_kind": "resolved_config",
                 "config_path": context.relative(context.config_path),
                 "config_sha256": context.config_sha256, "config_digest": context.config_digest,
                 "config": context.config,
                 "input_manifest": context.config["input_manifest"],
                 "note": ("a resolved copy for the run directory. The committed config is the "
                          "authority and this file never feeds a hash back into it.")},
                what="the resolved configuration", context=context, kind="resolved_config")
            paths.write_local_roots(context.run.run_root,
                                    {root.logical: root.local_path
                                     for root in context.roots.values()})
        progress.advance("evidence")
        record_timings(context, "prepare", clock)
    print(f"prepare: {manifest['input_count']} verified inputs, "
          f"{len(streams)} resolved task streams of "
          f"{streams[context.seeds[0]].exposures:,} exposures each", flush=True)
    print("prepare: review and commit the evidence and the config, then run preflight and freeze. "
          "This CLI does not commit.", flush=True)
    return manifest


def build_input_manifest(context, data_block, streams, orders, cadence, seed_block):
    """Every existing input this campaign depends on, plus the resolved stream identity."""
    from smallAntibodyGen.experiments import her2_data as data
    from smallAntibodyGen.experiments import her2_preferences as preferences
    audit_manifest, audit_entry = spec.audit_input_manifest(context.repository_root,
                                                            context.config)
    inputs = {}
    for seed, parent in sorted(parent_records(context).items()):
        target = context.root("original").path(parent["logical_path"])
        observed = paths.sha256_file(target)
        require(observed == parent["sha256"],
                f"{parent['logical']} hashes {observed}, the published audit manifest recorded "
                f"{parent['sha256']}")
        inputs[parent["logical"]] = {"root": "original", "logical_path": parent["logical_path"],
                                     "sha256": observed, "kind": "selected_parent",
                                     "seed": int(seed)}
    inherited = {}
    for seed, entry in sorted(inherited_bank_records(context).items()):
        target = context.root("guarded").path(entry["logical_path"])
        index, record = banks_lib.historical_reference_document(
            target, expected_sha256=entry["sha256"],
            expected_rows=int(context.config["banks"]["monitor_rows"]),
            label=entry["logical_path"])
        logical = f"{context.root('guarded').logical}/{entry['logical_path']}"
        inputs[logical] = {"root": "guarded", "logical_path": entry["logical_path"],
                           "sha256": entry["sha256"],
                           "kind": "inherited_diversity_reference", "seed": int(seed)}
        inherited[str(seed)] = record
    raw_root = context.root("raw")
    for logical_path in sorted(context.config["roots"]["raw"]["probe_files"]):
        target = raw_root.path(logical_path)
        inputs[f"{raw_root.logical}/{logical_path}"] = {
            "root": "raw", "logical_path": logical_path,
            "sha256": paths.sha256_file(target), "kind": "pinned_raw_source"}
    # The reserved split's SEQUENCES, bound rather than read unpinned. It is opened
    # with usecols=['seq'] by her2_data.test_sequences for the inherited generation
    # overlap diagnostic: no bin, no label and no SPR outcome crosses the wire, and
    # nothing in this screen is scored against the reserved split. Pinning it is the
    # honest alternative to a config note claiming the file is never opened.
    published_raw = {entry["logical_path"]: entry["sha256"]
                     for entry in (audit_manifest.get("inputs") or {}).values()
                     if entry.get("root") == "raw"}
    for logical_path in sorted(context.config["roots"]["raw"].get("sequence_only_files") or []):
        target = raw_root.path(logical_path)
        observed = paths.sha256_file(target)
        expected = published_raw.get(logical_path)
        require(expected is None or expected == observed,
                f"{logical_path} hashes {observed} and the published audit input manifest recorded "
                f"{expected}. A sequence-only input is bound to the same published digest as every "
                "other raw file this screen reads.")
        inputs[f"{raw_root.logical}/{logical_path}"] = {
            "root": "raw", "logical_path": logical_path,
            "sha256": observed, "kind": "pinned_raw_sequence_only",
            "columns_read": ["seq"],
            "published_audit_sha256": expected,
            "note": ("read through her2_data.test_sequences, which passes usecols=['seq']; the "
                     "reserved labels are not loaded by this screen at any point, and nothing "
                     "here is scored against the reserved split")}
    for name, entry in sorted(context.audit["published"].items()):
        inputs[entry["logical_path"]] = {"root": None, "logical_path": entry["logical_path"],
                                         "sha256": entry["sha256"],
                                         "kind": f"published_audit::{name}"}
    inputs[audit_entry["logical_path"]] = {"root": None,
                                           "logical_path": audit_entry["logical_path"],
                                           "sha256": audit_entry["sha256"],
                                           "kind": "published_audit::input_manifest"}
    publication_binding = context.audit["publication_manifest_binding"]
    inputs[publication_binding["logical_path"]] = {
        "root": None, "logical_path": publication_binding["logical_path"],
        "sha256": publication_binding["sha256"],
        "kind": "published_audit::publication_manifest",
        "note": ("the authority every published audit digest above was compared to. Binding the "
                 "files without binding the manifest that published them would leave the "
                 "authority itself free to move.")}
    marker = context.config["audit"]["completion_marker"]
    marker_path = context.root("audit_run").path(marker)
    require(marker_path.is_file(),
            f"{marker} is absent under the verified audit run root. This screen is conditional on "
            "a completed audit, and the completion marker is what records that completion.")
    completion = paths.read_json(marker_path)
    completion_binding = spec.require_audit_completion(context.config, completion,
                                                       audit=context.audit)
    inputs[f"{context.root('audit_run').logical}/{marker}"] = {
        "root": "audit_run", "logical_path": marker,
        "sha256": paths.sha256_file(marker_path), "kind": "published_audit::completion",
        "note": ("the immutable completion marker, written once after the audit's own requirement "
                 "checks passed. The run directory's verification.json and progress files are "
                 "operational, are rewritten by every later verify, and are deliberately not "
                 "inputs here.")}

    val_pairs = data_block["val_pairs"]
    pair_identity = {
        "fixed_validation": {
            "pairs": int(val_pairs["pairs"]),
            "digest": val_pairs["digest"],
            "chosen_core_order_sha256": preferences.core_digest(val_pairs["chosen_index"]),
            "rejected_core_order_sha256": preferences.core_digest(val_pairs["rejected_index"]),
            "pairing_seed": int(context.config["pairing"]["validation_seed"]),
            "source": "her2_preferences.PreferencePairing.fixed_validation_pairs, cycle 0"}}
    scaffold_prefix = data_block["scaffold"].prefix
    streams_block = streams_lib.stream_manifest(
        streams, endpoints=context.endpoint_updates,
        batch_rows=int(context.config["screen"]["chosen_per_update"]), cadence=cadence,
        replay_orders={seed: streams_lib.replay_order_document(
            orders[seed], bank_rows=int(context.config["banks"]["replay_rows"]),
            seed=int(context.config["seeds"]["replay_order"]), parent_seed=seed,
            batch_rows=int(context.config["screen"]["chosen_per_update"]))
            for seed in streams})
    return {"schema_version": spec.REPLAY_SCHEMA, "record_kind": "input_manifest",
            "campaign_id": context.config["campaign_id"], "protocol": context.config["protocol"],
            "config": {"path": context.relative(context.config_path),
                       "sha256": context.config_sha256, "digest": context.config_digest},
            "probability_contract": {
                "convention": "sum_log_probability_over_10_core_positions_20way_renormalized",
                "core_length": int(data.CORE_LENGTH), "canonical_residues": len(data.CANONICAL),
                "forward_dtype": "float32", "reduction_dtype": "float64"},
            "scaffold": {"prefix_length": len(scaffold_prefix),
                         "prefix_sha256": paths.sha256_text(scaffold_prefix),
                         "source": "row 0 of the submitted design table via her2_data.load_scaffold",
                         "source_file": f"{raw_root.logical}/{data.SUBMITTED_CSV}",
                         "source_sha256": paths.sha256_file(raw_root.path(data.SUBMITTED_CSV)),
                         "note": ("the submitted design table is pinned above as a raw input "
                                  "because this is a file the screen actually opens, at every "
                                  "stage. No reserved test label and no SPR outcome is read to "
                                  "obtain the prefix: load_scaffold reads the H/L columns of row "
                                  "0 and asserts the fixed offsets.")},
            "inputs": dict(sorted(inputs.items())), "input_count": len(inputs),
            "pair_populations": pair_identity,
            "inherited_diversity_references": inherited,
            "streams": streams_block,
            "seeds": seed_block,
            "audit": {"decision_outcome": context.audit["decision_outcome"],
                      "decision_methods": context.audit["decision_methods"],
                      "source_freeze_commit": context.audit["audit_source_freeze_commit"],
                      "completion_marker_sha256": paths.sha256_file(marker_path),
                      "completion": completion_binding,
                      "publication_manifest": context.audit["publication_manifest_binding"],
                      "published_verification": context.audit["published_verification"],
                      "escalation_detail": context.config["audit"]["escalation_detail"]},
            "banks_note": ("the replay and monitoring banks do not exist yet and are not inputs "
                           "here. The freeze binds their sampling rule, seeds, counts and dtypes; "
                           "they are generated afterwards and bound to it then."),
            "note": ("generated by the prepare stage and committed. The config links to it and "
                     "does not restate it, so neither document hashes itself.")}


def record_timings(context, stage, clock):
    document = dict(clock.document(), stage=stage, recorded_at=paths.utc_now(),
                    schema_version=spec.REPLAY_SCHEMA)
    target = context.run.path(f"timings/{stage}.json")
    if target.is_file():
        document["note"] = ("a rerun of a stage whose original timings are recorded; those are "
                            "preserved and this file records the rerun's own cost")
        return paths.write_json(context.run.path(f"timings/{stage}.rerun.json"), document)
    return paths.write_json(target, document)


def stage_timings(context):
    out = {}
    for stage in STAGES:
        path = context.run.path(f"timings/{stage}.json")
        if path.is_file():
            out[stage] = paths.read_json(path)
    return out


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------

def cmd_preflight(context, args):
    """Native inference-only parity on the real parents, plus a synthetic gradient control.

    Both halves run before the freeze and neither is a fit. The first loads real
    parent weights and performs **no optimizer step**; the second performs optimizer
    steps on a freshly randomly initialized model of the same architecture, with a
    synthetic prefix and synthetic cores, and loads no checkpoint at all.
    """
    import torch
    device = choose_device(context, args.device)
    clock = paths.StageClock()
    progress = context.progress("preflight", total=2)
    with progress.guard():
        with clock.segment("analysis"):
            data_block = build_context_data(context, device=device)
        with clock.segment("inference"):
            native = native_inference_probe(context, data_block, device=device)
        progress.advance("native inference")
        with clock.segment("inference"):
            control = synthetic_gradient_control(context, device=device,
                                                 raw_root=data_block["raw_root"])
        progress.advance("synthetic gradient control")
        closure = spec.source_closure(context.repository_root, spec.entry_points(context))
        report = {"schema_version": spec.REPLAY_SCHEMA, "record_kind": "preflight",
                  "campaign_id": context.config["campaign_id"], "status": "completed",
                  "device": device, "native": native, "gradient_control": control,
                  "tiny_cpu_controls": tiny_cpu_controls(context),
                  "binding": spec.preflight_binding(context, closure),
                  "environment": support.environment_record(),
                  "capacity": spec.capacity_record(
                      context.run.run_root,
                      minimum_bytes=int(context.config["storage"]["min_free_bytes"]),
                      required=False),
                  "vram": vram_record(),
                  "no_fit_claim": ("no optimizer step touched a HER2 parent here. The gradient "
                                   "control ran on randomly initialized weights of the same "
                                   "architecture with synthetic inputs and loaded no checkpoint."),
                  "timings": clock.document(), "generated_at": paths.utc_now()}
        support.require_new_or_identical(context.run.path(spec.PREFLIGHT_JSON), report,
                                         what="the preflight report", context=context,
                                         kind="preflight")
        record_timings(context, "preflight", clock)
    print(f"preflight: cached-vs-full {native['cached_vs_full']['max_abs_error']:.3g}, "
          f"microbatch parity {native['batch_parity']['max_abs_error']:.3g}, "
          f"native gradient {control['direct_vs_accumulated']['max_abs_error']:.3g}, "
          f"post-AdamW {control['post_step_parameters']['max_abs_error']:.3g}", flush=True)
    if torch.cuda.is_available():
        print(f"preflight: max VRAM allocated "
              f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)
    return report


def vram_record():
    import torch
    if not torch.cuda.is_available():
        return {"available": False,
                "note": "no CUDA device; absence of an OOM on this box is never cited as evidence"}
    free, total = torch.cuda.mem_get_info()
    return {"available": True, "device": torch.cuda.get_device_name(0),
            "free_mib": free / 2 ** 20, "total_mib": total / 2 ** 20,
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
            "note": ("recorded because on this box an over-large configuration does not raise, it "
                     "spills into system RAM and reports success. The absence of an OOM is never "
                     "cited as evidence that a configuration fits.")}


def native_inference_probe(context, data_block, *, device):
    """Real parent weights, inference only: route, batch-shape and cache parity.

    No optimizer, no sampling and no new bank. The probe rows come from the
    already-persisted, hash-verified historical parent draws, so this stage creates
    no new draws of any kind before the freeze.
    """
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    from smallAntibodyGen.experiments import her2_support_scoring as scoring
    tolerances = context.config["tolerances"]
    seed = context.seeds[0]
    parent = parent_records(context)[seed]
    entry = inherited_bank_records(context)[seed]
    index, reference = banks_lib.historical_reference_document(
        context.root("guarded").path(entry["logical_path"]), expected_sha256=entry["sha256"],
        expected_rows=int(context.config["banks"]["monitor_rows"]),
        label=entry["logical_path"])
    rows = banks_lib.probe_rows(index.shape[0], count=int(context.config["banks"]["probe_rows"]),
                                seed=int(context.config["seeds"]["cache_probe"]))
    probe_index = index[rows]
    policy, identity = load_parent_policy(context, parent, device=device, data_block=data_block)
    policy.model.eval()
    started = time.perf_counter()
    cached = scoring.strict_sequence_log_probabilities(
        policy, probe_index, batch_size=16, label="preflight cached route")
    full = scoring.strict_sequence_log_probabilities(
        policy, probe_index, batch_size=16, label="preflight full route", cached=False)
    cached_vs_full = policy_lib.compare_sum_log_probabilities(
        cached["sum_log_probability"], full["sum_log_probability"],
        label="preflight: cached prefix versus full teacher forcing",
        atol=float(tolerances["sum_log_probability_atol"]),
        rtol=float(tolerances["sum_log_probability_rtol"]))
    sizes = [16, 64, 256]
    by_size = {str(size): scoring.strict_sequence_log_probabilities(
        policy, probe_index, batch_size=size, label=f"preflight batch {size}")[
            "sum_log_probability"] for size in sizes}
    batch_parity = policy_lib.compare_sum_log_probabilities(
        by_size["16"], by_size["256"], label="preflight: batch 16 versus batch 256",
        atol=float(tolerances["sum_log_probability_atol"]),
        rtol=float(tolerances["sum_log_probability_rtol"]))
    probabilities, logs, cache_block = banks_lib.build_teacher_cache(
        policy, probe_index, batch_size=int(context.config["inference"]["teacher_batch_size"]),
        label="preflight teacher cache")
    cache_validation = banks_lib.validate_teacher_cache(
        probabilities, logs, probe_index,
        sampler_sum_log_probability=cached["sum_log_probability"],
        label="preflight teacher cache",
        atol=float(tolerances["sum_log_probability_atol"]),
        rtol=float(tolerances["sum_log_probability_rtol"]))
    throughput_rows = int(min(index.shape[0], int(context.config["banks"]["monitor_rows"])))
    segment_started = time.perf_counter()
    bulk = scoring.strict_sequence_log_probabilities(
        policy, index[:throughput_rows],
        batch_size=int(context.config["inference"]["score_batch_size"]),
        label="preflight bounded throughput segment")
    bulk_seconds = time.perf_counter() - segment_started
    return {"parent": identity, "probe_rows": rows.tolist(),
            "probe_source": dict(reference),
            "cached_vs_full": cached_vs_full, "batch_parity": batch_parity,
            "batch_sizes": sizes,
            "teacher_cache": cache_block, "teacher_cache_validation": cache_validation,
            "logit_checks": cached["checks"],
            "throughput": {"rows": throughput_rows, "seconds": bulk_seconds,
                           "rows_per_second": throughput_rows / max(bulk_seconds, 1e-9),
                           "batch_size": int(context.config["inference"]["score_batch_size"]),
                           "sum_log_probability_sha256":
                               paths.array_digest(bulk["sum_log_probability"]),
                           "note": ("a bounded measured segment over thousands of rows, not a "
                                    "five-row extrapolation. It is a rough planning number; the "
                                    "campaign reports its actual measured costs separately.")},
            "wall_seconds": time.perf_counter() - started,
            "optimizer_steps": 0,
            "claim": ("inference only on the real parent weights. No optimizer step, no sampling "
                      "and no new bank before the freeze.")}


def synthetic_gradient_control(context, *, device, raw_root):
    """The native architecture, random weights, synthetic prefix and cores, at full LR.

    This is the check a real-parent inference probe cannot make: that gradient
    actually reaches the prefix through :meth:`CorePolicy.core_logits`, that the
    frozen teacher's *cached* targets are what the student distils, and that
    accumulating microbatches reproduces the direct full-batch update at the real
    learning rate. It uses :func:`her2_policy.random_init_model`, which seeds the
    global RNG, rather than ``architecture_model``, which deliberately does not --
    an unseeded initialization would make the routes incomparable across
    invocations. No checkpoint is loaded and no HER2 weight is touched.

    Four things this version does that the first one did not:

    * it runs at the **production effective batch** (64 rows) with the production
      microbatch (16, an exact partition) and an additional partial one (24), rather
      than at 12 rows;
    * it steps at the **full configured learning rate**, not at the 1e-7 first
      warmup rate, where two parameter vectors that barely moved would compare equal
      whatever the routes did;
    * it proves the prefix gradient on an embedding **row** that no core token can
      reach, instead of on a whole embedding tensor the core positions also write
      into;
    * it builds the teacher targets through the real teacher-cache path and feeds
      the cached float32 arrays, which is what the fit reads.
    """
    import torch
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    from smallAntibodyGen.experiments.her2_data import CANONICAL
    settings = context.config["preflight"]
    tolerances = context.config["tolerances"]
    optimization = context.config["optimization"]
    control_seed = int(context.config["seeds"]["synthetic_gradient_control"])
    rows = int(settings["synthetic_control_rows"])
    microbatches = [int(value) for value in settings["synthetic_control_microbatches"]]
    prefix_length = int(settings["synthetic_control_prefix_length"])
    full_lr = float(settings["synthetic_control_full_learning_rate"])
    vocabulary = int(policy_lib.PIGGEN_ARCHITECTURE["vocab_size"])
    canonical = len(CANONICAL)
    require(vocabulary > canonical,
            f"The pinned architecture has {vocabulary} vocabulary entries and {canonical} "
            "canonical residues; there is no non-canonical id to build a prefix-only token from")
    #: A vocabulary id the cores cannot reach: cores index the canonical ids
    #: 0..19, so this embedding row receives gradient through the prefix or not
    #: at all.
    prefix_only_token = canonical
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def fresh_policy(seed):
        model = policy_lib.random_init_model(raw_root, seed=seed, device=device)
        model.train()
        generator = np.random.default_rng([seed, 11])
        ids = generator.integers(canonical, vocabulary, size=prefix_length).tolist()
        ids[0] = prefix_only_token
        prefix_ids = torch.tensor([ids], dtype=torch.long, device=device)
        canonical_ids = torch.arange(canonical, dtype=torch.long, device=device)
        return policy_lib.CorePolicy(model, prefix_ids, canonical_ids)

    def embedding_parameter(model):
        for name, parameter in model.named_parameters():
            if name.endswith("embed_in.weight"):
                return name, parameter
        raise ValueError("The native architecture no longer exposes an input embedding named "
                         "embed_in.weight; the prefix-only gradient proof cannot name its row")

    generator = np.random.default_rng([control_seed, 3])
    cores = generator.integers(0, canonical, size=(rows, 10)).astype(np.int8)
    # A real second instance of the architecture, frozen, whose targets go through
    # the production teacher-cache build. "The teacher is frozen and its cached
    # targets are what the student distils" is only worth checking on the path the
    # fit actually reads.
    teacher_policy = fresh_policy(control_seed + 1)
    teacher_model = teacher_policy.model
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    teacher_before = policy_lib.state_digest(teacher_model)
    cached_probabilities, cached_logs, cache_block = banks_lib.build_teacher_cache(
        teacher_policy, cores, batch_size=int(context.config["inference"]["teacher_batch_size"]),
        label="synthetic gradient control teacher cache")
    teacher_probabilities = torch.from_numpy(cached_probabilities).to(device)
    teacher_logs = torch.from_numpy(cached_logs).to(device)
    replay_lib.require_teacher_targets(teacher_probabilities, teacher_logs, rows=rows,
                                       label="synthetic gradient control teacher cache")

    def component_means(policy, block):
        chosen = policy.sequence_log_probs(cores[block])
        task_mean, _ = replay_lib.task_term("continued_sft", policy_chosen=chosen)
        student = replay_lib.student_log_probabilities(policy, cores[block])
        replay_mean, _ = replay_lib.replay_term(
            teacher_probabilities[block], teacher_logs[block], student,
            label="synthetic gradient control")
        return task_mean, replay_mean

    def gradient_vector(model):
        return np.concatenate([p.grad.detach().double().reshape(-1).cpu().numpy()
                               for _, p in sorted(model.named_parameters())])

    def full_lr_step(policy):
        """One AdamW step at the full configured rate. The warmup is not involved.

        Returns the step record and the gradients **as the optimizer saw them**.
        ``clip_grad_norm_`` rescales ``.grad`` in place, and ``optimizer.step()``
        does not touch it afterwards, so reading the gradients after the step is
        what the first-step oracle has to be computed from: comparing a clipped
        update against an oracle built from unclipped gradients would fail for a
        reason that has nothing to do with the accumulation.
        """
        optimizer = torch.optim.AdamW(
            policy.model.parameters(), lr=full_lr,
            betas=tuple(optimization["betas"]),
            weight_decay=float(optimization["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        record = replay_lib.clip_and_step(policy.model, optimizer, scheduler,
                                          gradient_clip=float(optimization["gradient_clip"]))
        return record, gradient_vector(policy.model)

    lam = 1.0
    direct = fresh_policy(control_seed)
    direct.model.zero_grad(set_to_none=True)
    task_mean, replay_mean = component_means(direct, slice(0, rows))
    replay_lib.combined_loss(task_mean, replay_mean, replay_coefficient=lam).backward()
    direct_gradients = gradient_vector(direct.model)
    embedding_name, embedding = embedding_parameter(direct.model)
    # Cloned: ``clip_grad_norm_`` rescales ``.grad`` in place a few lines below, and
    # a view would quietly become the post-clip values.
    prefix_row = embedding.grad.detach()[prefix_only_token].clone()
    core_row = embedding.grad.detach()[0].clone()
    direct_before = replay_lib.state_vector(direct.model)
    direct_step, direct_stepped_gradients = full_lr_step(direct)
    direct_after = replay_lib.state_vector(direct.model)

    routes, comparisons = {}, {}
    for microbatch in microbatches:
        accumulated = fresh_policy(control_seed)
        accumulated.model.zero_grad(set_to_none=True)
        accumulator = replay_lib.MicrobatchAccumulator(
            {"task": rows, "replay": rows}, coefficients={"task": 1.0, "replay": lam},
            label=f"synthetic gradient control micro {microbatch}")
        partition = []
        for start in range(0, rows, microbatch):
            block = slice(start, min(start + microbatch, rows))
            partition.append(block.stop - block.start)
            task_block, replay_block = component_means(accumulated, block)
            accumulator.add({"task": block.stop - block.start,
                             "replay": block.stop - block.start},
                            {"task": task_block, "replay": replay_block},
                            backward=lambda tensor: tensor.backward())
        components = accumulator.finish()
        gradients = gradient_vector(accumulated.model)
        before = replay_lib.state_vector(accumulated.model)
        step, stepped_gradients = full_lr_step(accumulated)
        after = replay_lib.state_vector(accumulated.model)
        key = f"micro{microbatch}"
        routes[key] = {
            "microbatch_rows": microbatch, "partition": partition,
            "exact_partition": bool(rows % microbatch == 0),
            "components": components, "learning_rate_used": step["learning_rate_used"],
            "gradient_norm": step["gradient_norm"], "clipped": step["clipped"]}
        comparisons[key] = {
            "gradients": replay_lib.compare_vectors(
                gradients, direct_gradients,
                atol=float(tolerances["native_gradient_atol"]),
                rtol=float(tolerances["native_gradient_rtol"]),
                label=f"synthetic control micro {microbatch}: accumulated versus direct gradients"),
            # Each route against ITS OWN gradients through an independent closed-form
            # first AdamW step. This holds whether or not the two routes agree, which
            # is what makes it the load-bearing check.
            "own_adamw_oracle": replay_lib.compare_vectors(
                after, replay_lib.adamw_first_step(
                    before, stepped_gradients, learning_rate=full_lr,
                    weight_decay=float(optimization["weight_decay"]),
                    betas=tuple(optimization["betas"])),
                atol=float(tolerances["post_adamw_oracle_atol"]),
                rtol=float(tolerances["post_adamw_oracle_rtol"]),
                label=f"synthetic control micro {microbatch}: parameters versus its own AdamW "
                      "first-step oracle"),
            "cross_route_parameters": replay_lib.reconcile_post_step(
                actual=after, expected=direct_after,
                gradient_actual=stepped_gradients,
                gradient_expected=direct_stepped_gradients,
                atol=float(tolerances["post_adamw_parameter_atol"]),
                rtol=float(tolerances["post_adamw_parameter_rtol"]),
                learning_rate=full_lr, weight_decay=float(optimization["weight_decay"]),
                parameters_before=before,
                label=f"synthetic control micro {microbatch}: accumulated versus direct "
                      "parameters after one full-rate AdamW step")}

    direct_oracle = replay_lib.compare_vectors(
        direct_after, replay_lib.adamw_first_step(
            direct_before, direct_stepped_gradients, learning_rate=full_lr,
            weight_decay=float(optimization["weight_decay"]),
            betas=tuple(optimization["betas"])),
        atol=float(tolerances["post_adamw_oracle_atol"]),
        rtol=float(tolerances["post_adamw_oracle_rtol"]),
        label="synthetic control direct: parameters versus its own AdamW first-step oracle")

    frozen, teacher_report = replay_lib.teacher_is_frozen(teacher_model)
    teacher_after = policy_lib.state_digest(teacher_model)
    del teacher_policy, teacher_model
    # Reused rather than another instance of the backbone: the student == teacher
    # control is about the loss, not about which weights it runs on, and a 4 GB
    # device does not need another 88 MB of parameters plus its gradients to say so.
    direct.model.zero_grad(set_to_none=True)
    student_logs = replay_lib.student_log_probabilities(direct, cores)
    zero_mean, _ = replay_lib.replay_term(student_logs.exp().detach(), student_logs.detach(),
                                          student_logs, label="student equals teacher")
    zero_mean.backward()
    zero_gradient = max((float(p.grad.detach().abs().max())
                         for _, p in direct.model.named_parameters() if p.grad is not None),
                        default=0.0)
    record = {
        "device": device, "rows": rows, "microbatch_rows": microbatches,
        "seed": control_seed, "replay_lambda": lam,
        "learning_rate": full_lr,
        "learning_rate_note": ("the full configured rate. The production step+1 warmup is "
                               "unchanged and is not exercised here: at its first rate of 1e-7 a "
                               "parameter comparison would accept two routes that did nothing."),
        "weights": "freshly random-initialized native architecture; no checkpoint was loaded",
        "teacher_cache": cache_block,
        "routes": routes,
        "comparisons": comparisons,
        "direct": {"learning_rate_used": direct_step["learning_rate_used"],
                   "gradient_norm": direct_step["gradient_norm"],
                   "clipped": direct_step["clipped"],
                   "own_adamw_oracle": direct_oracle},
        # Kept under their original names so the freeze binding and the printed
        # summary keep pointing at the production microbatch's numbers.
        "direct_vs_accumulated": comparisons[f"micro{microbatches[0]}"]["gradients"],
        "post_step_parameters": comparisons[f"micro{microbatches[0]}"]["cross_route_parameters"],
        "acceptance": {
            "gradients": "direct versus accumulated, within the declared gradient tolerance",
            "own_route_oracle": ("each route's post-step parameters against an independent "
                                 "closed-form first AdamW step from its own gradients"),
            "cross_route": ("reconciled: every coordinate outside the parameter tolerance must be "
                            "one where the two routes' gradients bracket zero within their own "
                            "measured discrepancy, and must stay inside one step's maximum "
                            "movement"),
            "amendment": context.config["tolerances"]["native_post_step_amendment"]},
        "prefix_gradient": {
            "parameter": embedding_name,
            "prefix_only_token": prefix_only_token,
            "row_max_abs": float(prefix_row.abs().max()),
            "core_row_max_abs": float(core_row.abs().max()),
            "nonzero": bool(float(prefix_row.abs().max()) > 0),
            "finite": bool(torch.isfinite(prefix_row).all()),
            "note": ("the gradient of the embedding ROW of a vocabulary id that appears only in "
                     "the prefix and in no core. CorePolicy.core_logits builds its prefix cache "
                     "inside the call, so the prefix computation is part of the student graph; a "
                     "zero here means the prefix was detached and only the ten core positions were "
                     "training. The whole embedding tensor would be nonzero either way, because "
                     "the core positions write into it.")},
        "teacher_frozen": dict(teacher_report, digest_unchanged=bool(
            teacher_after == teacher_before), frozen=frozen,
            produced_the_targets=True,
            targets_via="the production teacher-cache build, not an independently sampled tensor"),
        "student_equals_teacher": {
            "replay_loss": float(zero_mean.detach()),
            "max_abs_gradient": zero_gradient,
            "atol": float(tolerances["native_gradient_atol"]),
            "loss_is_zero": bool(abs(float(zero_mean.detach()))
                                 <= float(tolerances["native_gradient_atol"])),
            "gradient_is_zero": bool(zero_gradient
                                     <= float(tolerances["native_gradient_atol"]))},
        "parameters_moved_max_abs": float(np.abs(direct_after - direct_before).max()),
        "vram": vram_record()}
    require(record["prefix_gradient"]["nonzero"] and record["prefix_gradient"]["finite"],
            "The synthetic control found no finite nonzero gradient on the prefix-only embedding "
            "row. The student's prefix must receive gradient; if it does not, only the ten core "
            "positions are training.")
    require(record["parameters_moved_max_abs"] > float(tolerances["post_adamw_parameter_atol"]),
            f"One full-rate AdamW step moved the parameters by at most "
            f"{record['parameters_moved_max_abs']:.3g}, which is inside the comparison tolerance "
            "itself. A parameter comparison at that scale would accept two routes that had done "
            "nothing, so the control is not evidence and the run stops here.")
    require(record["teacher_frozen"]["frozen"] and record["teacher_frozen"]["digest_unchanged"],
            "The teacher carried gradients or changed parameters across a student step. The "
            "teacher is frozen and detached by construction; this is not a tolerance question.")
    require(record["student_equals_teacher"]["loss_is_zero"]
            and record["student_equals_teacher"]["gradient_is_zero"],
            "At student == teacher the replay loss and its gradient must both be zero within the "
            "declared gradient tolerance; they were not.")
    return record


def tiny_cpu_controls(context):
    """The float64 enumerable control, recorded beside the native numbers, never instead of them.

    A tiny CPU fixture that passes says the mathematics is right. It says nothing
    about the native backbone, the CUDA kernels or the released weights, and it is
    recorded here under a name that keeps the two apart.
    """
    import torch
    tolerances = context.config["tolerances"]
    generator = torch.Generator().manual_seed(20260920)
    teacher_logits = torch.randn(7, 10, 20, generator=generator, dtype=torch.float64)
    student_logits = torch.randn(7, 10, 20, generator=generator, dtype=torch.float64)
    teacher_logs = torch.log_softmax(teacher_logits, dim=-1)
    probabilities = teacher_logs.exp()
    student_logs = torch.log_softmax(student_logits, dim=-1)
    fast = replay_lib.sequence_conditional_kl(probabilities, teacher_logs, student_logs)
    slow = torch.zeros(7, dtype=torch.float64)
    for row in range(7):
        for position in range(10):
            for residue in range(20):
                p = float(probabilities[row, position, residue])
                if p > 0:
                    slow[row] += p * (float(teacher_logs[row, position, residue])
                                      - float(student_logs[row, position, residue]))
    error = float((fast - slow).abs().max())
    require(error <= float(tolerances["tiny_cpu_atol"]),
            f"The vectorized conditional KL disagrees with the direct triple sum by {error:.3g}")
    return {"rows": 7, "max_abs_error": error, "dtype": "float64",
            "atol": float(tolerances["tiny_cpu_atol"]),
            "scope": ("an enumerable float64 control of the formula only. CPU-only fixture "
                      "coverage is not evidence that the native backbone passed.")}


# ---------------------------------------------------------------------------
# stage: freeze
# ---------------------------------------------------------------------------

def cmd_freeze(context, args):
    marker = spec.run_freeze(context)
    print(f"freeze: commit {marker['git']['commit']}, "
          f"{len(marker['source']['sha256'])} source files in the computed closure, "
          f"{marker['input_count']} inputs, {len(marker['evidence_sha256'])} evidence files",
          flush=True)
    print("freeze: the banks do not exist yet. Run the banks stage next; its digests are bound "
          "to this marker.", flush=True)
    return marker


# ---------------------------------------------------------------------------
# stage: banks
# ---------------------------------------------------------------------------

def cmd_banks(context, args):
    """Generate the replay and monitoring banks and the frozen teacher caches."""
    device = choose_device(context, args.device)
    marker = spec.require_frozen_identity(context, device=device, label="banks")
    spec.require_immutable_fields(marker, {"device": None if args.device == "auto"
                                           else args.device})
    lock = campaign_lib.CampaignLock(context.run.path(LOCK_FILE))
    clock = paths.StageClock()
    # The teacher cache is timed apart from the sampling it sits next to. It is the
    # cost every replay arm pays before any of them runs, and folding it into "bank
    # inference" is what produced a cost table with a zero in the teacher-cache row
    # while the work was actually happening.
    ledger = campaign_lib.CostLedger()
    entries, overlap = {}, {}
    with lock.held():
        progress = context.progress("banks", total=len(context.seeds))
        with progress.guard():
            data_block = build_context_data(context, device=device)
            parents = parent_records(context)
            for seed in context.seeds:
                entries.update(build_seed_banks(context, seed=seed, parent=parents[seed],
                                                device=device, data_block=data_block,
                                                clock=clock, marker=marker, ledger=ledger))
                overlap[str(seed)] = paths.read_json(
                    context.run.path(f"banks/seed{seed}/overlap.json"))
                progress.advance(f"seed {seed}")
            manifest = banks_lib.banks_manifest(
                entries, campaign_id=context.config["campaign_id"],
                freeze_commit=marker["git"]["commit"],
                freeze_sha256=paths.sha256_file(spec.freeze_marker_path(context)),
                overlap=overlap,
                timings=dict(clock.document(), categories=ledger.document()))
            support.require_new_or_identical(context.run.path(spec.BANKS_MANIFEST_JSON), manifest,
                                             what="the banks manifest", context=context,
                                             kind="banks_manifest")
            record_timings(context, "banks", clock)
            paths.write_json(context.run.path("timings/banks.categories.json"),
                             dict(ledger.document(), stage="banks",
                                  schema_version=spec.REPLAY_SCHEMA,
                                  record_kind="bank_cost_categories",
                                  recorded_at=paths.utc_now(),
                                  note=("teacher-cache seconds are measured separately from "
                                        "sampling and scoring, because they are what the replay "
                                        "arms pay for before the first update")))
    print(f"banks: {manifest['bank_count']} bank artifacts across {len(context.seeds)} seeds, "
          f"teacher cache {ledger.seconds['teacher_cache']:.1f}s, "
          f"generation {ledger.seconds['generation']:.1f}s", flush=True)
    return manifest


def build_seed_banks(context, *, seed, parent, device, data_block, clock, marker, ledger=None):
    """Everything one parent contributes: references, banks and frozen teacher caches."""
    from smallAntibodyGen.experiments import her2_guard as guard
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    from smallAntibodyGen.experiments import her2_preferences as preferences
    settings = context.config["banks"]
    tolerances = context.config["tolerances"]
    ledger = ledger or campaign_lib.CostLedger()
    directory = context.run.path(f"banks/seed{seed}")
    directory.mkdir(parents=True, exist_ok=True)
    policy, identity = load_parent_policy(context, parent, device=device, data_block=data_block)
    policy.model.eval()
    entries = {}

    def bind(key, relative, *, role, scientific_digest, kind, **fields):
        """Record one produced artifact in the banks manifest and in the run ledger.

        Both, for different questions. The banks manifest is the authority the fit
        re-hashes every artifact against before the first update; the completion
        ledger is what a later ``verify`` walks. An identity sidecar that appears in
        neither can be deleted or edited without either one noticing, and the
        arrays it describes cannot be loaded without it.
        """
        target = context.run.path(relative)
        entries[key] = {"file": relative, "sha256": paths.sha256_file(target), "role": role,
                        **fields}
        paths.record_completion(context.run.run_root, context.run.logical(relative), target,
                                kind=kind, scientific_digest=scientific_digest)
        return entries[key]

    with clock.segment("inference"), ledger.segment("evaluation"):
        reference_identity = guard.parent_reference_identity(
            parent_checkpoint_sha256=identity["file_sha256"],
            parent_state_sha256=policy_lib.state_digest(policy.model),
            config_sha256=context.config_sha256,
            scaffold_prefix=data_block["scaffold"].prefix,
            chosen_index=data_block["val_pairs"]["chosen_index"],
            rejected_index=data_block["val_pairs"]["rejected_index"])
        gate_path = directory / "parent_validation_reference.npz"
        if gate_path.is_file():
            gate_reference = guard.load_parent_reference(gate_path, reference_identity)
        else:
            gate_reference = guard.build_parent_reference(
                policy, data_block["val_pairs"], reference_identity,
                batch_size=int(context.config["inference"]["monitor_batch_size"]))
            guard.save_parent_reference(gate_path, gate_reference)
        bind(f"seed{seed}::parent_validation_reference",
             f"banks/seed{seed}/parent_validation_reference.npz",
             role="gate_reference", pairs=gate_reference.pairs,
             identity=dict(gate_reference.identity),
             kind="parent_validation_reference",
             scientific_digest=paths.digest_document(dict(gate_reference.identity)))
        # The sidecar is not a by-product: load_parent_reference refuses to load
        # without it, and every identity field D is measured against lives in it.
        bind(f"seed{seed}::parent_validation_reference{banks_lib.SIDECAR_SUFFIX}",
             f"banks/seed{seed}/parent_validation_reference.json",
             role="gate_reference_sidecar", describes="parent_validation_reference.npz",
             kind="parent_validation_reference_sidecar",
             scientific_digest=paths.digest_document(dict(gate_reference.identity)))

    if "ipo" in context.config["screen"]["tasks"]:
        with clock.segment("inference"), ledger.segment("evaluation"):
            cache_identity = preferences.reference_identity(
                checkpoint_sha256=identity["file_sha256"], config_sha256=context.config_sha256,
                scaffold_prefix=data_block["scaffold"].prefix,
                index=data_block["reference_index"])
            cache_path = directory / "ipo_reference_cache.npy"
            if cache_path.is_file():
                cache = preferences.load_reference_cache(cache_path, cache_identity)
            else:
                from smallAntibodyGen.experiments.her2_runtime import GpuBudgetClock
                budget = GpuBudgetClock(clock=time.perf_counter)
                cache = preferences.build_reference_cache(
                    policy, data_block["reference_index"], cache_identity, clock=budget,
                    batch_size=int(context.config["inference"]["score_batch_size"]))
                preferences.save_reference_cache(cache_path, cache)
            probe = np.random.default_rng([int(context.config["seeds"]["cache_probe"]),
                                           int(seed)]).choice(
                cache.values.size, size=min(64, cache.values.size), replace=False)
            parity = preferences.verify_reference_parity(
                policy, cache, data_block["reference_index"], probe,
                atol=float(tolerances["sum_log_probability_atol"]),
                rtol=float(tolerances["sum_log_probability_rtol"]))
            bind(f"seed{seed}::ipo_reference_cache",
                 f"banks/seed{seed}/ipo_reference_cache.npy",
                 role="ipo_parent_reference", rows=int(cache.values.size),
                 identity=dict(cache.identity), fresh_parity=parity,
                 kind="ipo_reference_cache",
                 scientific_digest=paths.digest_document(dict(cache.identity)))
            bind(f"seed{seed}::ipo_reference_cache{banks_lib.SIDECAR_SUFFIX}",
                 f"banks/seed{seed}/ipo_reference_cache.json",
                 role="ipo_parent_reference_sidecar", describes="ipo_reference_cache.npy",
                 kind="ipo_reference_cache_sidecar",
                 scientific_digest=paths.digest_document(dict(cache.identity)))

    drawn = {}
    for role, rows, seed_key in (("replay", int(settings["replay_rows"]), "replay_draw"),
                                 ("monitor", int(settings["monitor_rows"]), "monitor_draw")):
        prefix = f"banks/seed{seed}/{role}"
        with clock.segment("inference"), ledger.segment("generation"):
            bank, probes = banks_lib.draw_bank(
                policy, role=role, parent_seed=seed, parent_id=identity["parent_id"],
                parent_state_sha256=identity["state_sha256"], rows=rows,
                draw_seed=int(context.config["seeds"][seed_key][str(seed)]),
                temperature=float(settings["temperature"]),
                batch_size=int(context.config["inference"]["sample_batch_size"]),
                atol=float(tolerances["sum_log_probability_atol"]),
                rtol=float(tolerances["sum_log_probability_rtol"]))
        with clock.segment("inference"), ledger.segment("teacher_cache"):
            probabilities, logs, cache_block = banks_lib.build_teacher_cache(
                policy, bank.index,
                batch_size=int(context.config["inference"]["teacher_batch_size"]),
                label=f"{role} teacher cache seed {seed}")
        with clock.segment("analysis"):
            validation = banks_lib.validate_teacher_cache(
                probabilities, logs, bank.index,
                sampler_sum_log_probability=bank.sampler_sum_log_probability,
                label=f"{role} teacher cache seed {seed}",
                atol=float(tolerances["sum_log_probability_atol"]),
                rtol=float(tolerances["sum_log_probability_rtol"]))
            rows_probed = banks_lib.probe_rows(
                bank.rows, count=int(settings["probe_rows"]),
                seed=int(context.config["seeds"]["cache_probe"]) + int(seed))
        with clock.segment("inference"), ledger.segment("evaluation"):
            live_probe = banks_lib.probe_teacher_cache(
                policy, probabilities, logs, bank.index, rows=rows_probed,
                label=f"{role} teacher cache seed {seed}",
                atol=float(tolerances["teacher_cache_probe_atol"]),
                rtol=float(tolerances["teacher_cache_probe_rtol"]))
        with clock.segment("io"), ledger.segment("io"):
            record = paths.write_shard(
                context.run.path(prefix), "bank",
                {"core_index": bank.index.astype(np.int8),
                 "draw_index": np.arange(bank.rows, dtype=np.int64),
                 "sampler_sum_log_probability": bank.sampler_sum_log_probability,
                 "parent_sum_log_probability": banks_lib.cached_sequence_log_probability(
                     logs, bank.index),
                 "teacher_probabilities": probabilities, "teacher_log_probabilities": logs},
                {"record_kind": "replay_bank", "campaign_schema": spec.REPLAY_SCHEMA,
                 "role": role, "seed": int(seed),
                 "identity": banks_lib.bank_identity(
                     bank, campaign_id=context.config["campaign_id"],
                     freeze_commit=marker["git"]["commit"]),
                 "bank": bank.document(), "teacher_cache": cache_block,
                 "validation": validation, "live_probe": live_probe,
                 "sampler_scorer_parity": probes["sampler_scorer_parity"],
                 "timings": clock.document()},
                order="bank draw order 0..N-1", logical_prefix=prefix,
                run_root=context.run.run_root)
        entries[f"seed{seed}::{role}_bank"] = {
            "file": f"{prefix}/bank.npz", "sha256": record["container"]["sha256"],
            "role": role, "rows": bank.rows,
            "order_sha256": paths.array_digest(bank.index),
            "arrays": record["arrays"]}
        # The shard's own completion record, named in the manifest beside its
        # container. ``write_shard`` already bound both files into the run ledger,
        # so this is a manifest entry only -- re-recording it here would offer the
        # ledger a second, different scientific digest for the same artifact.
        entries[f"seed{seed}::{role}_bank{banks_lib.SHARD_RECORD_SUFFIX}"] = {
            "file": f"{prefix}/bank{paths.SHARD_RECORD}",
            "sha256": paths.sha256_file(context.run.path(f"{prefix}/bank{paths.SHARD_RECORD}")),
            "role": f"{role}_bank_record", "describes": f"{prefix}/bank.npz",
            "note": ("read_shard refuses a container whose completion record is absent or "
                     "disagrees with it, so the record is an artifact of this stage and the fit "
                     "re-hashes it against this manifest like any other")}
        drawn[role] = bank

    overlap = banks_lib.bank_overlap(drawn["replay"].index, drawn["monitor"].index)
    support.require_new_or_identical(
        directory / "overlap.json",
        dict(overlap, schema_version=spec.REPLAY_SCHEMA, record_kind="bank_overlap",
             seed=int(seed)),
        what=f"the bank overlap record for seed {seed}", context=context, kind="bank_overlap")
    del policy
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return entries


# ---------------------------------------------------------------------------
# stage: fit
# ---------------------------------------------------------------------------

def status_registrar(context, trajectory):
    """A callback that binds one trajectory's terminal status into the run ledger.

    A terminal status is written once and is the trajectory's own statement of what
    it reached, so a later verification has to be able to hold it to its bytes.
    Nothing registered it before, which meant a deleted or edited ``status.json``
    left no trace in either the ledger or the coverage list. The digest is the
    scientific projection: the record's timestamps move, its measurements do not.
    """
    def register(*, path, document):
        logical = context.run.logical(f"trajectories/{trajectory}/{campaign_lib.STATUS_JSON}")
        return paths.record_completion(
            context.run.run_root, logical, path, kind="replay_trajectory_status",
            scientific_digest=paths.digest_document(paths.scientific_projection(document)))
    return register


class CadenceAdapter:
    """The declared cadence behind the inherited monitor's schedule interface.

    ``GateMonitor`` records a check against whatever schedule it was given. The
    inherited :class:`her2_guard.MonitorSchedule` measures its interval from the
    last check and requires a GPU-second trigger, so it cannot express this
    screen's cadence; handing the monitor this adapter keeps the inherited gate
    code untouched while the cadence document that travels into every artifact is
    the one this screen actually used.
    """

    def __init__(self, cadence):
        self.cadence = cadence
        self.last_update = None

    def record(self, update, gpu_seconds):
        self.last_update = int(update)

    def document(self):
        return self.cadence.document()


def cmd_fit(context, args):
    """Run the declared queue under one exclusive writer. No resume, ever."""
    import torch
    device = choose_device(context, args.device)
    marker = spec.require_frozen_identity(context, device=device, label="fit")
    spec.require_immutable_fields(marker, {"device": args.device if args.device != "auto"
                                           else None})
    closure = marker["source"]["closure"]
    capacity = spec.capacity_record(context.run.run_root,
                                    minimum_bytes=int(context.config["storage"]["min_free_bytes"]))
    queue = campaign_lib.build_queue(context.config)
    lock = campaign_lib.CampaignLock(context.run.path(LOCK_FILE))
    heartbeat = campaign_lib.Heartbeat(context.run.path(HEARTBEAT_FILE), owner=lock.owner)
    ran, interrupted = [], []
    # Every write into the run directory happens under the lock, the queue document
    # included. Writing it first would mean two campaign processes could each stamp
    # a queue before either discovered the other.
    with lock.held():
        spec.require_runtime_closure(closure, repository_root=context.repository_root)
        paths.write_json(context.run.path(spec.QUEUE_JSON),
                         {"schema_version": spec.REPLAY_SCHEMA, "record_kind": "campaign_queue",
                          "campaign_id": context.config["campaign_id"], "queue": queue,
                          "order": context.config["screen"]["execution_order"],
                          "generated_at": paths.utc_now()})
        banks_manifest = paths.read_json(context.run.path(spec.BANKS_MANIFEST_JSON))
        banks_lib.require_banks_bound(
            banks_manifest, freeze_commit=marker["git"]["commit"],
            freeze_sha256=paths.sha256_file(spec.freeze_marker_path(context)))
        # Once, before the first update: every role, every seed, every bank
        # container and every reference-cache sidecar re-hashed against the
        # completed manifest. A self-validating shard proves only that its sidecar
        # agrees with its arrays, which a coherent rewrite also achieves.
        bank_verification = banks_lib.verify_bank_artifacts(
            banks_manifest, resolve=lambda logical: context.run.path(logical),
            expected_keys=banks_lib.expected_bank_keys(
                context.seeds,
                include_ipo_reference="ipo" in context.config["screen"]["tasks"]),
            label="fit")
        for entry in queue:
            directory = context.run.path(f"trajectories/{entry['trajectory']}")
            marked = campaign_lib.mark_begun_incomplete(
                directory, owner=lock.recorded_owner(),
                register=status_registrar(context, entry["trajectory"]),
                reason=("this trajectory began under a previous campaign process that is no "
                        "longer holding the write lock. It is not resumed: no partial optimizer, "
                        "scheduler, RNG or stream state is adopted."))
            if marked is not None:
                interrupted.append(entry["trajectory"])
        data_block = build_context_data(context, device=device, need_generation=True)
        parents = parent_records(context)
        streams = resolve_streams(context, data_block)
        orders = replay_orders(context)
        cadence = cadence_for(context)
        status = {"schema_version": spec.REPLAY_SCHEMA, "record_kind": "campaign_status",
                  "campaign_id": context.config["campaign_id"], "status": "running",
                  "device": device, "started_at": paths.utc_now(), "owner": lock.owner,
                  "capacity": capacity, "queue_length": len(queue),
                  "banks_verified": bank_verification,
                  "previous_owners": lock.previous_owners(),
                  "marked_incomplete_at_start": interrupted}
        paths.write_json(context.run.path(spec.CAMPAIGN_STATUS_JSON), status)
        try:
            for entry in queue:
                directory = context.run.path(f"trajectories/{entry['trajectory']}")
                if campaign_lib.read_terminal_status(directory) is not None:
                    continue
                if args.max_trajectories is not None and len(ran) >= int(args.max_trajectories):
                    break
                document = fit_one(context, entry=entry, device=device, data_block=data_block,
                                   parent=parents[int(entry["seed"])],
                                   stream=streams[int(entry["seed"])],
                                   replay_order=orders[int(entry["seed"])], cadence=cadence,
                                   marker=marker, heartbeat=heartbeat, lock=lock,
                                   banks_manifest=banks_manifest)
                campaign_lib.write_terminal_status(
                    directory, document, register=status_registrar(context, entry["trajectory"]))
                ran.append({"trajectory": entry["trajectory"], "status": document["status"],
                            "updates": document["updates"]})
                print(f"fit: {entry['trajectory']} -> {document['status']} "
                      f"after {document['updates']} updates", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except BaseException as error:                          # noqa: BLE001 - recorded, re-raised
            paths.write_json(context.run.path(spec.CAMPAIGN_STATUS_JSON),
                             dict(status, status="failed", ended_at=paths.utc_now(),
                                  error=f"{type(error).__name__}: {error}", ran=ran,
                                  fail_stop=("an unexpected exception stops the campaign. A gate "
                                             "breach is a declared outcome and continues the "
                                             "queue; this is not one.")))
            raise
        paths.write_json(context.run.path(spec.CAMPAIGN_STATUS_JSON),
                         dict(status, status="finished", ended_at=paths.utc_now(), ran=ran))
    print(f"fit: {len(ran)} trajectories run this session, {len(interrupted)} earlier ones marked "
          "incomplete", flush=True)
    return ran


def fit_one(context, *, entry, device, data_block, parent, stream, replay_order, cadence, marker,
            heartbeat, lock, banks_manifest=None):
    """One arm at one seed, from the parent to a truthful terminal document."""
    import torch
    from smallAntibodyGen.experiments import her2_guard as guard
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    from smallAntibodyGen.experiments import her2_preferences as preferences
    screen = context.config["screen"]
    optimization = context.config["optimization"]
    directory = context.run.path(f"trajectories/{entry['trajectory']}")
    directory.mkdir(parents=True, exist_ok=True)
    policy, identity = load_parent_policy(context, parent, device=device, data_block=data_block)
    seed = int(entry["seed"])
    # The zero-replay control reads no replay cache at all. Loading and validating
    # 160 MB of teacher targets for an arm that never touches them would contradict
    # the claim that lambda = 0 performs no replay work -- and would charge the
    # control the replay arms' I/O.
    uses_replay = float(entry["replay_lambda"]) != 0.0
    bank = (load_bank(context, seed, "replay", device=device, manifest=banks_manifest)
            if uses_replay else None)
    monitor_bank = load_bank(context, seed, "monitor", device=device, manifest=banks_manifest)
    gate_reference = guard.load_parent_reference(
        context.run.path(f"banks/seed{seed}/parent_validation_reference.npz"),
        guard.parent_reference_identity(
            parent_checkpoint_sha256=identity["file_sha256"],
            parent_state_sha256=identity["state_sha256"],
            config_sha256=context.config_sha256,
            scaffold_prefix=data_block["scaffold"].prefix,
            chosen_index=data_block["val_pairs"]["chosen_index"],
            rejected_index=data_block["val_pairs"]["rejected_index"]))
    reference_cache = None
    if entry["task"] == "ipo":
        reference_cache = preferences.load_reference_cache(
            context.run.path(f"banks/seed{seed}/ipo_reference_cache.npy"),
            preferences.reference_identity(
                checkpoint_sha256=identity["file_sha256"], config_sha256=context.config_sha256,
                scaffold_prefix=data_block["scaffold"].prefix,
                index=data_block["reference_index"]))

    trajectory_identity = {
        "schema_version": spec.REPLAY_SCHEMA, "record_kind": "trajectory_identity",
        "campaign_id": context.config["campaign_id"], **entry,
        "parent": identity, "freeze_commit": marker["git"]["commit"],
        "freeze_marker_sha256": paths.sha256_file(spec.freeze_marker_path(context)),
        "stream": stream.document(), "cadence": cadence.document(),
        "optimization": dict(optimization), "device": device,
        "microbatch_rows": int(screen["microbatch_rows"]),
        "generation_seed": int(context.config["seeds"]["generation_base"])
                           + draw_offset(entry["trajectory"]),
        "started_at": paths.utc_now()}
    support.require_new_or_identical(directory / campaign_lib.IDENTITY_JSON, trajectory_identity,
                                     what=f"the identity of {entry['trajectory']}")

    policy.model.train()
    optimizer = preferences.build_optimizer(policy.model, optimization)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: policy_lib.continuation_learning_rate_scale(
            step + 1, warmup_steps=int(optimization["warmup_updates"])))
    ledger = campaign_lib.CostLedger(
        synchronize=torch.cuda.synchronize if device == "cuda" else None)
    population = data_block["train_population"]
    tau = float(screen["ipo_tau"])

    def task_batch(chosen_rows, rejected_rows):
        chosen_lp = policy.sequence_log_probs(population.chosen_index[chosen_rows])
        if entry["task"] == "continued_sft":
            return replay_lib.task_term("continued_sft", policy_chosen=chosen_lp)
        rejected_lp = policy.sequence_log_probs(population.rejected_index[rejected_rows])
        reference_chosen = reference_cache.tensor(chosen_rows, device=policy.device)
        reference_rejected = reference_cache.tensor(
            data_block["reference_rejected_rows"][rejected_rows], device=policy.device)
        return replay_lib.task_term(
            "ipo", policy_chosen=chosen_lp, policy_rejected=rejected_lp,
            reference_chosen=reference_chosen.to(chosen_lp.dtype),
            reference_rejected=reference_rejected.to(rejected_lp.dtype), tau=tau)

    def replay_batch(rows):
        require(bank is not None,
                f"{entry['trajectory']} is a zero-replay control and asked for a replay batch. "
                "The control's replay bank is never loaded, so this call is a bug rather than a "
                "missing file.")
        student = replay_lib.student_log_probabilities(policy, bank["core_index"][rows])
        return replay_lib.replay_term(
            bank["teacher_probabilities"][rows].to(student.device),
            bank["teacher_log_probabilities"][rows].to(student.device), student,
            label=f"{entry['trajectory']} replay")

    monitor = guard.GateMonitor(
        policy, gate_reference,
        gate=guard.LikelihoodGate(
            threshold_nats_per_sequence=float(context.config["gate"]
                                              ["threshold_nats_per_sequence"])),
        schedule=CadenceAdapter(cadence), directory=directory,
        batch_size=int(context.config["inference"]["monitor_batch_size"]))
    preservation = campaign_lib.PreservationMonitor(
        index=monitor_bank["core_index"],
        parent_log_probability=monitor_bank["parent_sum_log_probability"],
        teacher_probabilities=monitor_bank["teacher_probabilities"],
        teacher_log_probabilities=monitor_bank["teacher_log_probabilities"],
        batch_size=int(context.config["inference"]["monitor_batch_size"]))

    def gate_check(*, update, reason, exposures):
        was_training = policy.model.training
        policy.model.eval()
        try:
            return monitor.check(data_block["val_pairs"], update=update,
                                 gpu_seconds=ledger.seconds["optimizer"], reason=reason)
        finally:
            if was_training:
                policy.model.train()

    def measure_preservation(*, update, reason):
        return preservation.measure(
            policy, conditional_batch=int(context.config["inference"]["conditional_batch_size"]))

    def on_endpoint(*, update, exposures, checkpoint, gate, preservation, ledger):
        return evaluate_endpoint(context, entry=entry, update=update, exposures=exposures,
                                 checkpoint=checkpoint, gate=gate, preservation=preservation,
                                 policy=policy, data_block=data_block, ledger=ledger,
                                 directory=directory,
                                 generation_seed=trajectory_identity["generation_seed"])

    def register_checkpoint(*, name, path, kind, scientific_digest):
        """Bind a written checkpoint into the run's completion manifest.

        Endpoints and failure snapshots are write-once, so they go into the ledger
        and a later verification can prove the bytes are still there. The rolling
        ``last_passing`` must not: the ledger refuses a second, different set of
        bytes under one logical name, which is the right rule for an artifact that
        is written once and the wrong one for an artifact that is replaced at every
        check. Its current digest travels in the trajectory's own records.

        The saver is what applies that boundary -- it knows which write is rolling
        and does not offer it here. This refuses rather than silently skipping, so
        the two halves cannot drift into disagreeing about the contract.
        """
        require(kind != "last_passing",
                f"{name}: the rolling last-passing state was offered to the write-once completion "
                "ledger. It is replaced at every passing check, so the ledger would refuse its "
                "second write; the saver does not offer it and this is the assertion that says so.")
        logical = context.run.logical(f"trajectories/{entry['trajectory']}/{name}")
        return paths.record_completion(context.run.run_root, logical, path,
                                       kind=f"replay_checkpoint::{kind}",
                                       scientific_digest=scientific_digest)


    document = campaign_lib.run_trajectory(
        row=entry, directory=directory, policy=policy, optimizer=optimizer, scheduler=scheduler,
        stream=stream, replay_order=replay_order, cadence=cadence,
        endpoints=context.endpoint_updates, batch_rows=int(screen["chosen_per_update"]),
        microbatch_rows=int(screen["microbatch_rows"]), task_batch=task_batch,
        replay_batch=replay_batch, gate_check=gate_check, preservation=measure_preservation,
        on_endpoint=on_endpoint, gradient_clip=float(optimization["gradient_clip"]),
        identity=trajectory_identity, heartbeat=heartbeat,
        max_updates=int(screen["max_updates"]), ledger=ledger,
        register_checkpoint=register_checkpoint,
        register_status=status_registrar(context, entry["trajectory"]))
    document["monitor"] = monitor.document()
    document["parent"] = identity
    del policy, optimizer, scheduler
    return document


def load_bank(context, seed, role, *, device, manifest=None):
    """One bank and its frozen teacher cache, validated before a single row is used.

    ``read_shard`` proves the container agrees with the sidecar beside it. That is
    not the same claim as "this is the bank the completed manifest recorded": a
    regenerated cache with a regenerated sidecar passes the first and fails the
    second, so when the banks manifest is available its recorded container digest
    and draw order are compared here too.
    """
    import torch
    prefix = f"banks/seed{seed}/{role}"
    arrays, record = paths.read_shard(context.run.path(prefix), "bank")
    if manifest is not None:
        entry = (manifest.get("banks") or {}).get(f"seed{seed}::{role}_bank")
        require(entry is not None,
                f"{prefix}: the completed banks manifest records no {role} bank for seed {seed}")
        container_sha256 = (record.get("container") or {}).get("sha256")
        require(entry["sha256"] == container_sha256,
                f"{prefix}: the container hashes {container_sha256} and the completed banks "
                f"manifest recorded {entry['sha256']}. The manifest is the authority; a shard that "
                "only agrees with its own sidecar has not been checked against it.")
        require(entry["order_sha256"] == paths.array_digest(arrays["core_index"]),
                f"{prefix}: the draw order does not reproduce the digest the banks manifest "
                "recorded; a re-sorted or regenerated bank is a different measurement")
    probabilities = torch.from_numpy(np.ascontiguousarray(arrays["teacher_probabilities"]))
    logs = torch.from_numpy(np.ascontiguousarray(arrays["teacher_log_probabilities"]))
    replay_lib.require_teacher_targets(probabilities, logs, rows=int(arrays["core_index"].shape[0]),
                                       label=f"{role} bank seed {seed}")
    require(bool((arrays["draw_index"] == np.arange(arrays["core_index"].shape[0])).all()),
            f"{prefix}: the bank's draw_index is not the exact 0..N-1 order; a re-sorted bank is a "
            "different measurement")
    return {"core_index": np.asarray(arrays["core_index"]),
            "teacher_probabilities": probabilities, "teacher_log_probabilities": logs,
            "parent_sum_log_probability": np.asarray(arrays["parent_sum_log_probability"],
                                                      dtype=np.float64),
            "record": record}


def evaluate_endpoint(context, *, entry, update, exposures, checkpoint, gate, preservation,
                      policy, data_block, ledger, directory, generation_seed):
    """Validation ranking and generation diversity at one reached exposure endpoint."""
    from smallAntibodyGen.experiments import her2_data as data
    from smallAntibodyGen.experiments import her2_eval as evaluation
    from smallAntibodyGen.experiments import her2_policy as policy_lib
    settings = context.config["evaluation"]
    generation = context.config["generation"]
    was_training = policy.model.training
    policy.model.eval()
    try:
        with ledger.segment("evaluation"):
            scored = policy.score(data_block["val_index"],
                                  batch_size=int(context.config["inference"]["score_batch_size"]))
            ranking = report_lib.endpoint_evaluation(
                scored_mean_log_probability=scored["mean_log_probability"],
                positives=data_block["val_positive"], cores=data_block["val_cores"],
                strata=data_block["val_strata"],
                categories=tuple(settings["train_distance_strata"]),
                k_values=tuple(settings["k_values"]))
        with ledger.segment("generation"):
            index, sampled = policy.sample(
                int(generation["draws"]), seed=int(generation_seed),
                temperature=float(generation["temperature"]),
                batch_size=int(context.config["inference"]["sample_batch_size"]))
            rescored = policy.score(index,
                                    batch_size=int(context.config["inference"]["score_batch_size"]))
            parity = policy_lib.compare_sum_log_probabilities(
                rescored["sum_log_probability"], sampled,
                label=f"{entry['trajectory']} update {update}: sampler versus scorer",
                atol=float(context.config["tolerances"]["sum_log_probability_atol"]),
                rtol=float(context.config["tolerances"]["sum_log_probability_rtol"]))
            diagnostics = evaluation.generation_diagnostics(
                index, rescored["sum_log_probability"], catalogs=data_block["catalogs"],
                train_index=data_block["train_index"], train_labels=data_block["train_labels"],
                split_of=data_block["split_of"], class_of=data_block["class_of"],
                max_train_distance=int(settings["max_train_distance"]))
            diagnostics.update(draw_seed=int(generation_seed),
                               temperature=float(generation["temperature"]),
                               sampler_scorer_parity=parity)
            diversity = evaluation.diversity_eligibility(
                diagnostics,
                training_reference=data_block["training_diversity_reference"],
                parent_reference=data_block["inherited_diversity_reference"][int(entry["seed"])],
                gates={key: value for key, value in context.config["diversity_gates"].items()
                       if isinstance(value, (int, float))})
    finally:
        if was_training:
            policy.model.train()
    with ledger.segment("io"):
        prefix = f"trajectories/{entry['trajectory']}/endpoints/update{int(update)}"
        paths.write_shard(
            context.run.path(prefix), "evaluation",
            {"validation_mean_log_probability": scored["mean_log_probability"],
             "generation_core_index": np.asarray(index, dtype=np.int8),
             "generation_sum_log_probability": rescored["sum_log_probability"]},
            {"record_kind": "endpoint_evaluation", "campaign_schema": spec.REPLAY_SCHEMA,
             "trajectory": entry["trajectory"], "update": int(update),
             "identity": {"checkpoint_sha256": checkpoint["sha256"],
                          "state_sha256": checkpoint["state_sha256"],
                          "draw_seed": int(generation_seed)},
             "generation": diagnostics, "diversity": diversity, "ranking": ranking},
            order="validation row order; generation draw order 0..N-1",
            logical_prefix=prefix, run_root=context.run.run_root)
        record = report_lib.trajectory_endpoint_record(
            row=entry, update=update, exposures=exposures, checkpoint=checkpoint, gate=gate,
            preservation=preservation, evaluation=ranking, diversity=diversity)
        support.require_new_or_identical(
            context.run.path(f"{prefix}/endpoint.json"), record,
            what=f"the endpoint record for {entry['trajectory']} at update {update}",
            context=context, kind="endpoint")
    return record


# ---------------------------------------------------------------------------
# stage: status, report, verify, publish
# ---------------------------------------------------------------------------

def cmd_status(context, args):
    queue = campaign_lib.build_queue(context.config)
    state = campaign_lib.campaign_state(context.run.run_root, queue,
                                        lock_path=context.run.path(LOCK_FILE))
    campaign_path = context.run.path(spec.CAMPAIGN_STATUS_JSON)
    recorded = (paths.read_json(campaign_path).get("status")
                if campaign_path.is_file() else None)
    health = campaign_lib.launch_health(
        campaign_lib.first_update_and_check(context.run.run_root, queue),
        writer_state=(state.get("lock") or {}).get("state"),
        recorded_status=recorded,
        failed_trajectories=int(state["counts"].get(campaign_lib.STATUS_FAILED, 0)))
    frozen = spec.freeze_marker_path(context).is_file()
    banks_path = context.run.path(spec.BANKS_MANIFEST_JSON)
    document = dict(state, health=health,
                    frozen=frozen,
                    banks_ready=banks_path.is_file(),
                    phase=("fitting" if banks_path.is_file() and frozen else
                           "preparing_banks" if frozen else "before_freeze"),
                    campaign_id=context.config["campaign_id"])
    paths.write_json(context.run.path("status.json"), document)
    print(f"status: {document['phase']}, {document['counts']}", flush=True)
    print(f"status: launch={health['status']} ({health['detail']})", flush=True)
    return document


def cmd_report(context, args):
    queue = campaign_lib.build_queue(context.config)
    state = report_lib.attach_endpoint_records(
        context.run.run_root,
        campaign_lib.campaign_state(context.run.run_root, queue,
                                    lock_path=context.run.path(LOCK_FILE)))
    screen = context.config["screen"]
    rows = report_lib.endpoint_rows(state, endpoint_updates=context.endpoint_updates,
                                    batch_rows=int(screen["chosen_per_update"]))
    deltas = report_lib.matched_control_deltas(rows)
    pareto = report_lib.pareto_front(rows)
    coverage = report_lib.coverage_summary(state, rows)
    banks_path = context.run.path(spec.BANKS_MANIFEST_JSON)
    banks_document = paths.read_json(banks_path) if banks_path.is_file() else {}
    banks_block = {
        "replay_rows": int(context.config["banks"]["replay_rows"]),
        "monitor_rows": int(context.config["banks"]["monitor_rows"]),
        "generated": bool(banks_document),
        "overlap_summary": ", ".join(
            f"seed {seed}: {block.get('monitor_draws_also_in_replay')} of "
            f"{block.get('monitor_rows')} monitoring draws"
            for seed, block in sorted((banks_document.get("overlap") or {}).items()))
        or "not generated yet"}
    marker = (paths.read_json(spec.freeze_marker_path(context))
              if spec.freeze_marker_path(context).is_file() else None)
    freeze_block = None if marker is None else {
        "commit": marker["git"]["commit"], "frozen_at": marker["frozen_at"],
        "audit_decision_outcome": (marker.get("audit") or {}).get("decision_outcome"),
        "audit_source_freeze_commit": (marker.get("audit") or {}).get(
            "audit_source_freeze_commit")}
    costs = aggregate_costs(state, banks=banks_document)
    figures = (report_lib.render_figures(
        context.run.path("report/figures"), rows,
        prefix=context.config["publication"]["figure_prefix"])
        if context.config["reporting"]["figures"] else
        {"disabled": {"written": False, "name": "figures", "reason": "disabled in the config"}})
    text = report_lib.render_report(
        config=context.config, state=state, rows=rows, deltas=deltas, pareto=pareto,
        coverage=coverage, banks=banks_block, freeze=freeze_block, costs=costs, figures=figures)
    report_lib.require_no_forbidden_claim(text)
    paths.write_text(context.run.path(REPORT_MD), text)
    tables = report_lib.published_tables(rows=rows, deltas=deltas, pareto=pareto,
                                         coverage=coverage, state=state, banks=banks_block)
    paths.write_json(context.run.path("summaries/tables.json"), tables)
    if args.publish:
        publication = publish_deliverable(context, tables=tables, figures=figures,
                                          render=lambda plan: report_lib.render_report(
                                              config=context.config, state=state, rows=rows,
                                              deltas=deltas, pareto=pareto, coverage=coverage,
                                              banks=banks_block, freeze=freeze_block, costs=costs,
                                              figures=figures, publication=plan))
        print(f"report: published {publication['file_count']} files under "
              f"{context.config['publication']['root']}", flush=True)
    print(f"report: {coverage['reached_endpoint_rows']} of "
          f"{coverage['declared_endpoint_rows']} declared endpoint rows reached; "
          f"{len(deltas['matched'])} matched zero-replay differences", flush=True)
    return text


def aggregate_costs(state, *, banks=None):
    """Measured seconds by category, fitting and bank preparation together.

    Bank preparation is part of what this screen cost. Leaving it out would report
    a replay campaign whose teacher caches appeared to be free, and the teacher
    cache is precisely the cost the replay arms pay before any of them runs.
    """
    totals = {}
    for entry in state["trajectories"]:
        for name, value in ((entry.get("cost") or {}).get("seconds") or {}).items():
            totals[name] = totals.get(name, 0.0) + float(value)
        wall = (entry.get("cost") or {}).get("wall_seconds")
        if wall is not None:
            totals["wall"] = totals.get("wall", 0.0) + float(wall)
    categories = ((banks or {}).get("timings") or {}).get("categories") or {}
    for name, value in (categories.get("seconds") or {}).items():
        if not float(value):
            continue
        totals[f"banks::{name}"] = totals.get(f"banks::{name}", 0.0) + float(value)
    bank_wall = ((banks or {}).get("timings") or {}).get("wall_seconds")
    if bank_wall is not None:
        totals["banks::wall"] = float(bank_wall)
    return totals


def publish_deliverable(context, *, tables, figures, render):
    """Copy the report, the small tables and the figures into the tracked tree.

    Write-once and local only: a published file that already exists must hold
    exactly these bytes. Host paths are scrubbed from everything tracked, because
    they reach published evidence through recorded error strings rather than
    through fields anybody designed.
    """
    settings = context.config["publication"]
    root = context.repository_root / settings["root"]
    names = sorted(entry["name"] for entry in figures.values() if entry.get("written"))
    plan = {"root": settings["root"], "data_directory": settings["data_directory"],
            "figure_directory": settings["figure_directory"],
            "report": support._relative(context.repository_root / context.config[
                "published_report"], context.repository_root / settings["root"]),
            "files": sorted([f"{settings['data_directory']}/tables.json"]
                            + [f"{settings['figure_directory']}/{name}" for name in names])}
    published = {}
    scrubbed = paths.scrub_host_paths(tables)
    leaks = paths.host_path_leaks(scrubbed)
    require(not leaks, f"A published table still carries host paths at {leaks}")
    target = root / f"{settings['data_directory']}/tables.json"
    if target.is_file():
        existing = paths.read_json(target)
        require(paths.scientific_projection(existing) == paths.scientific_projection(scrubbed),
                f"{target} is already published with different scientific content; a changed "
                "result belongs in a new revision directory")
        published[f"{settings['data_directory']}/tables.json"] = paths.sha256_file(target)
    else:
        published[f"{settings['data_directory']}/tables.json"] = paths.write_json(target, scrubbed)
    for entry in sorted(figures.values(), key=lambda item: str(item.get("name"))):
        if not entry.get("written"):
            continue
        logical = f"{settings['figure_directory']}/{entry['name']}"
        destination = root / logical
        payload = context.run.path(f"report/figures/{entry['name']}").read_bytes()
        if destination.is_file():
            require(destination.read_bytes() == payload,
                    f"{destination} is already published with different bytes; a re-rendered "
                    "figure is a new artifact and belongs in a new revision directory")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        published[logical] = paths.sha256_bytes(payload)
    require(sorted(published) == plan["files"],
            f"The publisher wrote {sorted(published)} but the report was rendered against "
            f"{plan['files']}")
    text = render(plan)
    report_lib.require_no_forbidden_claim(text)
    published[plan["report"]] = support.require_new_or_identical_text(
        context.repository_root / context.config["published_report"], text,
        what="the published replay report")
    manifest = {"schema_version": spec.REPLAY_SCHEMA, "record_kind": "publication_manifest",
                "campaign_id": context.config["campaign_id"], "root": settings["root"],
                "files": dict(sorted(published.items())), "file_count": len(published),
                "links": ("every path above is relative to the published report at "
                          f"{context.config['published_report']}"),
                "note": ("local tracked evidence. This CLI performs no network or git action; "
                         "committing these files is a separate, human decision.")}
    manifest_path = root / settings["manifest"]
    if manifest_path.is_file():
        existing = paths.read_json(manifest_path)
        require(paths.scientific_projection(existing) == paths.scientific_projection(manifest),
                f"{manifest_path} already publishes a different set of files")
    else:
        paths.write_json(manifest_path, manifest)
    return manifest


def cmd_publish(context, args):
    args.publish = True
    return cmd_report(context, args)


def expected_run_outputs(context, queue):
    """Every artifact the declared grid and the begun trajectories say must exist.

    Derived from what was *declared* and from what each trajectory's own records
    say it *reached* -- never from a directory listing. A scan can only enumerate
    what is still there, so an artifact whose file and whose ledger entry were both
    removed disappears from the scan and from the problem list with it, which is the
    exact state this check exists to catch. Every requirement below is therefore
    derived from a **record** -- a terminal status, a journalled snapshot, the banks
    manifest -- and never from whether the file happens to be on disk.
    """
    expected, rolling, missing = [], [], []
    begun = [row for row in queue
             if context.run.path(f"trajectories/{row['trajectory']}").is_dir()]
    for name in (spec.RESOLVED_CONFIG_JSON, spec.PREFLIGHT_JSON, spec.BANKS_MANIFEST_JSON):
        if context.run.path(name).is_file():
            expected.append(context.run.logical(name))
        elif begun:
            # Once a trajectory exists, the stage documents that had to precede it
            # are required. Treating them as optional would let a run whose freeze,
            # preflight or banks manifest had been removed still report immutable.
            missing.append({"artifact": name,
                            "problem": ("fitting has begun and this stage document is not in the "
                                        "run directory; the fit cannot have run without it")})
    if begun and not spec.freeze_marker_path(context).is_file():
        missing.append({"artifact": spec.FREEZE_MARKER,
                        "problem": "fitting has begun and the freeze marker is not on disk"})
    banks_path = context.run.path(spec.BANKS_MANIFEST_JSON)
    if banks_path.is_file():
        for entry in sorted((paths.read_json(banks_path).get("banks") or {}).values(),
                            key=lambda value: str(value.get("file"))):
            target = context.run.path(entry["file"])
            if not target.is_file():
                missing.append({"artifact": entry["file"],
                                "problem": ("the completed banks manifest records this artifact "
                                            "and the file is gone from the run directory")})
    for row in queue:
        directory = context.run.path(f"trajectories/{row['trajectory']}")
        if not directory.is_dir():
            continue
        prefix = f"trajectories/{row['trajectory']}"
        terminal = campaign_lib.read_terminal_status(directory)
        durable = campaign_lib.durable_progress(directory)
        for update in sorted((terminal or {}).get("endpoints_reached")
                             or durable["endpoints_reached"], key=lambda value: int(value)):
            # A reached endpoint must still have its checkpoint bytes, its
            # evaluation shard and its endpoint record.
            expected.append(context.run.logical(f"{prefix}/endpoint_update{int(update)}.pt"))
            expected.append(context.run.logical(
                f"{prefix}/endpoints/update{int(update)}/endpoint.json"))
            # Shard keys are run-relative, because that is the ledger convention
            # ``write_shard`` records them under.
            expected.extend(paths.shard_logicals(
                f"{prefix}/endpoints/update{int(update)}", "evaluation"))
        snapshots = campaign_lib.read_journal(directory / campaign_lib.MONITOR_JSONL,
                                              record_kind="snapshot")["records"]
        passed_snapshots = [record for record in snapshots if record.get("passed")]
        failed_snapshots = [record for record in snapshots if record.get("failed_state")]
        # A recorded failure snapshot, or a terminal stop, is what requires the
        # diagnostic state -- not the presence of the file. Deriving the requirement
        # from ``is_file()`` meant deleting the file also deleted the requirement.
        if failed_snapshots or (terminal or {}).get("stop_reason"):
            expected.append(context.run.logical(f"{prefix}/{campaign_lib.FAILED_STATE}"))
        if terminal is not None:
            # The terminal status itself: completed, stopped_by_gate, incomplete or
            # failed. It is written once and registered, so it is verifiable; a
            # trajectory that is still running writes only mutable progress.
            expected.append(context.run.logical(f"{prefix}/{campaign_lib.STATUS_JSON}"))
        for leaf in (campaign_lib.IDENTITY_JSON, campaign_lib.UPDATES_JSONL,
                     campaign_lib.MONITOR_JSONL):
            if (directory / leaf).is_file():
                continue
            missing.append({"artifact": f"{prefix}/{leaf}",
                            "problem": "a begun trajectory is missing one of its own journals"})
        passing = directory / campaign_lib.LAST_PASSING
        latest = passed_snapshots[-1] if passed_snapshots else None
        if latest is not None and not passing.is_file():
            missing.append({"artifact": f"{prefix}/{campaign_lib.LAST_PASSING}",
                            "problem": ("a passing check recorded a last-passing state at update "
                                        f"{latest.get('update')} and the file is not on disk")})
        elif passing.is_file():
            # The one rolling artifact. It is verified as a *latest* state: against
            # the digest the most recent passing snapshot recorded, not against an
            # arbitrary hash of whatever bytes are there now, and not against an
            # older check's digest, which described this path then and does not now.
            recorded = ((latest or {}).get("last_passing") or {})
            observed = paths.sha256_file(passing)
            entry = {"artifact": f"{prefix}/{campaign_lib.LAST_PASSING}",
                     "sha256": observed, "kind": "rolling_latest_state",
                     "recorded_sha256": recorded.get("sha256"),
                     "recorded_state_sha256": recorded.get("state_sha256"),
                     "recorded_at_update": recorded.get("update"),
                     "recorded_at_check": recorded.get("check"),
                     "matches_latest_snapshot": (None if not recorded
                                                 else observed == recorded.get("sha256")),
                     "note": ("replaced at every passing check by design, so it is not bound "
                              "write-once in the completion ledger. It is compared to the most "
                              "recent passing snapshot's recorded digest; every earlier digest "
                              "stays true of the bytes it described then.")}
            rolling.append(entry)
            if entry["matches_latest_snapshot"] is False:
                missing.append({"artifact": entry["artifact"],
                                "problem": ("the rolling last-passing state does not hash to the "
                                            "digest its most recent passing snapshot recorded "
                                            f"({observed} vs {recorded.get('sha256')})")})
            elif latest is None:
                missing.append({"artifact": entry["artifact"],
                                "problem": ("a last-passing state is on disk and no passing check "
                                            "recorded writing one")})
        # Nothing is required from a passing *check count*: a trajectory whose first
        # gate check failed records one check, no passing snapshot and no rolling
        # state, and demanding one there was a false problem on the legitimate
        # first-check stop.
    return {"expected": sorted(set(expected)), "rolling": rolling, "problems": missing}


def cmd_verify(context, args):
    """Re-check the completed artifacts against the saved authorities, not against disk."""
    queue = campaign_lib.build_queue(context.config)
    coverage = expected_run_outputs(context, queue)
    verification = paths.verify_completions(
        context.run.run_root, resolve=lambda logical: _resolve_run_logical(context, logical),
        expected=coverage["expected"])
    problems = list(verification["problems"]) + list(coverage["problems"])
    frozen, source = None, None
    if spec.freeze_marker_path(context).is_file():
        # The frozen source, config and input identity, re-hashed here too: a run
        # whose outputs are all intact but whose sources moved is not verified.
        try:
            marker = spec.require_frozen_identity(context, device=None, label="verify")
            frozen = {"commit": marker["git"]["commit"],
                      "sources": len(marker["source"]["sha256"]),
                      "inputs": marker["input_count"]}
            source = "re-hashed against the freeze marker"
        except ValueError as error:
            problems.append({"artifact": spec.FREEZE_MARKER,
                             "problem": f"{type(error).__name__}: {error}"})
    banks_path = context.run.path(spec.BANKS_MANIFEST_JSON)
    banks_block = None
    if banks_path.is_file():
        try:
            banks_block = banks_lib.verify_bank_artifacts(
                paths.read_json(banks_path),
                resolve=lambda logical: context.run.path(logical),
                expected_keys=banks_lib.expected_bank_keys(
                    context.seeds,
                    include_ipo_reference="ipo" in context.config["screen"]["tasks"]),
                label="verify")
        except ValueError as error:
            problems.append({"artifact": spec.BANKS_MANIFEST_JSON,
                             "problem": f"{type(error).__name__}: {error}"})
    document = dict(verification, schema_version=spec.REPLAY_SCHEMA,
                    record_kind="verification", campaign_id=context.config["campaign_id"],
                    problems=problems,
                    nothing_to_verify=bool(not coverage["expected"]
                                           and not verification["artifacts_checked"]),
                    verified_claim=("this is an immutability statement about the artifacts that "
                                    "exist, not a completeness statement about the grid. An empty "
                                    "ledger is trivially self-consistent and says nothing was "
                                    "produced; the campaign's coverage is in the report."),
                    expected_from=("the declared grid, the banks manifest and each trajectory's "
                                   "own records: its reached endpoints, its journalled passing "
                                   "and failing snapshots, and its terminal status"),
                    expected_output_count=len(coverage["expected"]),
                    rolling_artifacts=coverage["rolling"],
                    frozen_identity=frozen, source_check=source,
                    banks=banks_block,
                    immutable=not problems, generated_at=paths.utc_now())
    paths.write_json(context.run.path("verification.json"), document)
    print(f"verify: {document['artifacts_checked']} ledger artifacts, "
          f"{document['expected_output_count']} expected outputs, "
          f"{len(coverage['rolling'])} rolling states, immutable={document['immutable']}",
          flush=True)
    if document["nothing_to_verify"]:
        print("verify: nothing has been produced yet. An empty ledger is self-consistent and is "
              "not a completed campaign.", flush=True)
    for problem in document["problems"]:
        print(f"  PROBLEM {problem}", flush=True)
    require(document["immutable"],
            "Completed outputs are not immutable; see verification.json. A rerun that would "
            "change content or timings belongs in a new run directory.")
    return document


def _resolve_run_logical(context, logical):
    """Map a completion-manifest key back to a file.

    Two conventions live in one ledger and both are real: documents are recorded
    under the run's logical root (``outputs/<run>/preflight.json``) and shards under
    a run-relative prefix (``banks/seed.../bank.npz``), because that is what
    ``write_shard`` was given. Returning ``None`` for the second kind would report
    every shard in the run as absent, which is a false problem and, worse, a
    distracting one.
    """
    prefix = f"{context.run.logical_run_root}/"
    relative = str(logical)[len(prefix):] if str(logical).startswith(prefix) else str(logical)
    try:
        return context.run.path(relative)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

COMMANDS = {"prepare": cmd_prepare, "preflight": cmd_preflight, "freeze": cmd_freeze,
            "banks": cmd_banks, "fit": cmd_fit, "status": cmd_status, "report": cmd_report,
            "verify": cmd_verify, "publish": cmd_publish}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="posttrain_her2_replay.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES, help="campaign stage to run")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="replay configuration (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="run root override; defaults to the config's run_root")
    parser.add_argument("--original-root", default=None,
                        help="local directory holding the original v1 post-training output")
    parser.add_argument("--guarded-root", default=None,
                        help="local directory holding the guarded campaign output")
    parser.add_argument("--raw-root", default=None, help="local directory holding the raw release")
    parser.add_argument("--audit-root", default=None,
                        help="local directory holding the completed support-audit run")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="rejected at fit time if it would change the frozen device")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="operational bound on how many queued trajectories one fit session "
                             "starts. It does not change the declared grid, and the rest stay "
                             "queued")
    parser.add_argument("--publish", action="store_true",
                        help="report stage: also write the tracked reference/ report, tables and "
                             "figures")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    context = spec.resolve_context(
        ROOT, config_path=ROOT / args.config, run_root=args.output,
        root_overrides={"original": args.original_root, "guarded": args.guarded_root,
                        "raw": args.raw_root, "audit_run": args.audit_root})
    print(f"replay stage {args.stage} -> {context.run.logical_run_root}", flush=True)
    COMMANDS[args.stage](context, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
