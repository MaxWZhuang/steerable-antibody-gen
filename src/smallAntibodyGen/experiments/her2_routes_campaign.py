"""The stage machine, the queue, the measured forecast and the durable supervisor.

Stage order is fixed and deterministic::

    acquire -> preflight -> split -> sources -> profile -> smoke -> freeze
    -> bakeoff -> route_gate -> controls -> priors -> posttrain -> decoding
    -> confirm -> report -> verify

**``split`` precedes ``sources``, and that ordering is the substance of the
reserved panel.** ``sources`` opens outcome columns; the split is a function of
sequences alone. Running them the other way round materialises every HER2
measurement -- including the reserved panel's -- before anything has decided what
the reserved panel is, and no later discipline recovers from that. The split
stage writes the frozen window assignment and the sources stage reads it and
gates the outcome loader on it.

``run-all`` advances through them with no follow-up call. It may stop -- loudly,
with evidence retained -- at a **readiness gate** (a missing input, a changed
snapshot, insufficient disk) or a **feasibility gate** (the hard split is
underpowered, no route clears G1). What it may not do is skip a required stage or
call a partial flight complete. ``verify`` asserts **completeness, not
correctness**: green checks mean every required stage ran, not that the science is
right.

Reused unchanged from the landed harness: ``CampaignLock`` for single-writer
exclusion, ``Heartbeat`` for liveness, ``launch_detached``'s detached-process
pattern (background jobs are killed at the agent turn boundary, which is why a
supervisor is a detached process with its own log), and ``StageProgress`` for
per-stage heartbeats.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from contextlib import contextmanager

from . import her2_routes_lineage as lineage
from . import her2_routes_spec as spec
from . import her2_support_paths as paths
from .her2_replay_campaign import CampaignLock, Heartbeat as _BaseHeartbeat
from .her2_replay_campaign import lock_state, owner_identity
from .her2_runtime import require

CAMPAIGN_SCHEMA = spec.ROUTES_SCHEMA

STAGES = ("acquire", "preflight", "split", "sources", "profile", "smoke", "freeze", "bakeoff",
          "route_gate", "controls", "priors", "posttrain", "decoding", "confirmation_freeze",
          "confirm", "report", "verify")
EXTRA_STAGES = ("status", "run-all", "launch", "health")
REQUIRED_STAGES = STAGES

LOCK_FILE = "campaign.lock"
HEARTBEAT_JSON = "heartbeat.json"
STATUS_JSON = "campaign_status.json"
STAGE_RECORDS = "stages"

#: Which stages a failed stage blocks. The rest still run, which is why a
#: feasibility failure in one route is not a whole-flight failure.
DEPENDENTS = {
    "acquire": tuple(STAGES[1:]),
    "preflight": tuple(STAGES[2:]),
    # The split gates the outcome read, so everything downstream of sources is
    # downstream of it -- including sources itself.
    "split": tuple(STAGES[3:]),
    "sources": tuple(STAGES[4:]),
    "profile": tuple(STAGES[5:]),
    "smoke": tuple(STAGES[6:]),
    "freeze": tuple(STAGES[7:]),
    "bakeoff": tuple(STAGES[8:]),
    "route_gate": tuple(STAGES[9:]),
    "controls": tuple(STAGES[10:]),
    "priors": tuple(STAGES[11:]),
    "posttrain": tuple(STAGES[12:]),
    "decoding": tuple(STAGES[13:]),
    "confirmation_freeze": tuple(STAGES[14:]),
    "confirm": tuple(STAGES[15:]),
    "report": ("verify",),
}

#: Stages that may not run on partial upstream work, and why. ``bakeoff`` may be
#: partial and the independent downstream stages still run on what exists; the
#: confirmation may not, because it freezes a selection that does not exist yet.
STRICT_PREREQUISITES = {
    "confirmation_freeze": ("route_gate", "controls", "priors", "posttrain", "decoding"),
    "confirm": ("confirmation_freeze",),
}


#: The campaign statuses that mean "this worker finished what it was asked to". The training-phase
#: status is deliberately a different word from ``completed``: a finished DPLM training phase is
#: not a complete flight and never reads as one.
STATUS_COMPLETED = "completed"
STATUS_TRAINING_PHASE_COMPLETED = "training_phase_completed"
SUCCESS_STATUSES = (STATUS_COMPLETED, STATUS_TRAINING_PHASE_COMPLETED)


def phase_policy(config):
    """The validated training-phase declaration for ``config``, or ``None`` for a legacy config."""
    return spec.training_phase(config)


def stages_for(config):
    """The stage list a caller that names none runs. The full flight, or the phase's own list."""
    policy = phase_policy(config)
    return STAGES if policy is None else tuple(policy["stages"])


def require_stage_allowed(context, stage):
    """Refuse a stage the declared training phase holds. Nothing has been touched yet.

    Central: ``run_stage`` (every direct stage and every ``run_all`` iteration) and the entry
    points that resolve a stage list all call this, so no path reaches a held stage's effects.
    A legacy config has no phase and every stage is allowed, exactly as before.
    """
    policy = phase_policy(context.config)
    if policy is not None:
        require(str(stage) not in policy["held_stages"] and str(stage) in policy["stages"],
                f"stage {stage!r} is held by the {policy['phase_id']} training phase "
                f"({policy['policy_sha256']}): it is training-only, not competition admission, and "
                "selection, controls, priors, post-training, decoding, confirmation and the report "
                "stay held until an explicit admission protocol is frozen")
    return policy


def require_stages_allowed(context, stages):
    """Validate a whole stage list before the first stage of it runs."""
    for stage in stages:
        require_stage_allowed(context, stage)


def resolve_stages(context, stages=None):
    """``stages`` if given (validated), else the default for this config's phase."""
    chosen = tuple(stages_for(context.config) if stages is None else stages)
    require_stages_allowed(context, chosen)
    return chosen


def select_stages(context, *, from_stage=None, through_stage=None):
    """The stage list for a worker: the default list, narrowed by ``--from/--through-stage``.

    Both bounds must lie inside the default list, so under a phase a held stage cannot be named
    as either end. A legacy config gets exactly the slice it always did.
    """
    stages = tuple(stages_for(context.config))
    for label, bound in (("from-stage", from_stage), ("through-stage", through_stage)):
        require(bound is None or bound in stages,
                f"--{label} {bound!r} is not one of this config's stages {list(stages)}")
    if from_stage:
        stages = stages[stages.index(from_stage):]
    if through_stage:
        require(through_stage in stages, f"--through-stage {through_stage!r} precedes --from-stage")
        stages = stages[:stages.index(through_stage) + 1]
    return resolve_stages(context, stages)


class ReadinessGate(RuntimeError):
    """A prerequisite is missing. The flight pauses; nothing scientific is claimed."""


class FeasibilityGate(RuntimeError):
    """A declared scientific criterion failed. Dependants stop; the rest still run."""


class Heartbeat(_BaseHeartbeat):
    """Liveness during long calls. Journals stay the progress evidence."""

    def __init__(self, path, *, owner, every=15.0):
        super().__init__(path, owner=owner, every=every)
        self._guard = threading.Lock()
        self._fields = {}
        self._failure = None

    def beat(self, *, _tick=False, **fields):
        with self._guard:
            if self._failure is not None:
                raise RuntimeError("The background heartbeat failed") from self._failure
            if not _tick:
                self._fields.update(fields)
                self._fields["last_work_callback_at"] = paths.utc_now()
            return super().beat(**dict(self._fields, force=bool(fields.get("force")),
                                       liveness_tick=bool(_tick)))

    @contextmanager
    def running(self):
        stopped = threading.Event()

        def tick():
            while not stopped.wait(max(0.01, self.every)):
                try:
                    self.beat(_tick=True, force=True)
                except Exception as error:                 # noqa: BLE001 - recorded
                    self._failure = error
                    stopped.set()

        worker = threading.Thread(target=tick, name="her2-routes-heartbeat", daemon=True)
        worker.start()
        try:
            yield self
        finally:
            stopped.set()
            worker.join(timeout=1.0)


def _plain(node):
    """numpy scalars coerced to JSON types, so a stage record cannot fail at the end."""
    import numpy as np
    if isinstance(node, dict):
        return {str(key): _plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(value) for value in node]
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, np.ndarray):
        return node.tolist()
    return node


def stage_record_path(context, stage):
    return context.path(STAGE_RECORDS, f"{stage}.json")


def write_stage_record(context, stage, document):
    record = dict(document, schema_version=CAMPAIGN_SCHEMA, record_kind="stage_record",
                  stage=str(stage), recorded_at=paths.utc_now())
    paths.write_json(stage_record_path(context, stage), record)
    return record


def read_stage_record(context, stage):
    target = stage_record_path(context, stage)
    return paths.read_json(target) if target.is_file() else None


def stage_summary(context, stages=STAGES):
    return {stage: {"ran": (record := read_stage_record(context, stage)) is not None,
                    "status": (record or {}).get("status", "not_run"),
                    "reason": (record or {}).get("reason"),
                    "recorded_at": (record or {}).get("recorded_at")}
            for stage in stages}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_acquire(context, services, **_):
    """Verify every acquisition manifest and every external production source.

    No download happens here: acquisition is a separate, explicit script. This stage
    is the check that what those scripts produced is complete, hashed and describes
    the files on disk -- across **all three** manifests, whose paths are normalised to
    logical names because one of them records absolute ones.

    The closure is derived from the manifests. The earlier hand-written list named
    ``ablang2/__init__.py``; the pinned package is at
    ``upstream/ablang2/ablang2/__init__.py``, so the verification failed against its
    own note that the closure was three paths.
    """
    manifest, record = spec.read_acquisition_manifest(context)
    table, per_manifest = spec.read_all_manifests(context)
    closure = spec.external_closure(
        context, table=table,
        extra=tuple(context.config["artifacts"].get("external_sources_extra") or ()))
    # A DPLM route's weights, config and vocabulary enter the frozen identity only through
    # the DPLM acquisition manifest; a config that names the paths without one is refused.
    dplm = spec.require_dplm_binding(context, table=table, per_manifest=per_manifest,
                                     closure=closure)
    digests = spec.external_source_digests(
        context, manifest, logical_paths=closure["logical_paths"], table=table)
    document = spec.write_external_marker(
        context, manifest_record=record, digests=digests,
        native_report={"manifests": per_manifest, "closure": closure, "dplm": dplm})
    return {"status": "completed", "external_files": document["file_count"],
            "external_sha256": document["external_sha256"],
            "manifests": per_manifest, "dplm": dplm,
            "closure_paths": len(closure["logical_paths"]),
            "outcomes_opened": record["outcomes_opened"], "pins": record["pins"],
            "note": ("digests come from the three acquisition manifests and are re-hashed against "
                     "disk. The closure is derived from those manifests rather than listed; a "
                     "missing or mismatched digest blocks the freeze, and nothing is hard-coded "
                     "here.")}


def stage_preflight(context, services, **_):
    """Environment ordering, disk, native evidence, and the gradient control."""
    from . import her2_replay_spec as replay_spec
    from .her2_routes_env import assert_no_home_cache, home_cache_baseline
    baseline = home_cache_baseline()
    capacity = replay_spec.capacity_record(
        context.artifact_root, minimum_bytes=int(context.config["storage"]["min_free_bytes"]),
        required=False)
    if not capacity["sufficient"]:
        raise ReadinessGate(
            f"the artifact volume has {capacity['free_gib']:.1f} GiB free and this flight declares "
            f"{capacity['required_free_gib']:.1f} GiB. Bank and checkpoint counts are not reduced "
            "to fit; provision the volume and rerun.")
    spec.verify_external_sources(context, label="preflight")
    evidence = {route: services.native_evidence(route, home_cache_baseline=baseline)
                for route in context.config["routes"]["enabled"]}
    paths.write_json(context.path("native_evidence.json"), _plain(evidence))
    return {"status": "completed", "capacity": capacity, "routes": sorted(evidence),
            "home_cache_baseline": _plain(baseline),
            "home_cache": assert_no_home_cache(baseline=baseline),
            "parity": {route: (block.get("parity_at_zero")
                               or block.get("native_token_parity")
                               or block.get("context_permutation"))
                       for route, block in evidence.items()},
            "note": ("parity, parameter counts, attention scaling and gradient reach are measured "
                     "numbers in native_evidence.json. The conditioning and decode parity probes "
                     "are run with PERTURBED tables, because at zero initialisation they pass "
                     "whatever the code does.")}


def stage_sources(context, services, **_):
    """Load, qualify and inventory every enrolled source, behind the frozen split.

    This is where outcome columns are opened. It runs after ``split`` and reads
    that stage's frozen window assignment, so the reserved panel's measurements
    are filtered out by the parser rather than by a later convention.
    """
    spec.verify_external_sources(context, label="sources")
    require(context.path(spec.SPLIT_FREEZE).is_file(),
            "The frozen split record is absent, so there is no train/development window set to "
            "gate the outcome read with. Run the split stage first; loading outcomes without it "
            "materialises the reserved panel.")
    inventory = services.build_sources()
    paths.write_json(context.path("source_inventory.json"), _plain(inventory))
    missing = [row["source"] for row in inventory["sources"]
               if row["enrolled"] and not row["observations"]]
    if missing:
        raise ReadinessGate(
            f"these enrolled sources produced no qualified observation: {missing}. An enrolled "
            "source with nothing in it is a loader or a qualification defect, not an empty table.")
    return {"status": "completed", "sources": len(inventory["sources"]),
            "observations": inventory["totals"]["observations"],
            "unique_constructs": inventory["totals"]["unique_constructs"],
            "unenrolled": [row["source"] for row in inventory["sources"] if not row["enrolled"]]}


def _freeze_pinned_split(context, services, policy):
    """Derive the split WITHOUT persisting, compare it to the pin, and only then write it.

    ``freeze_hard_split`` writes the record itself unless told not to, so a mismatching split
    would already be on disk when the comparison ran. Here a mismatch stops the phase before
    anything is written and long before any production update. A re-run whose record already
    holds the pinned digest keeps it: ``freeze_split`` stamps ``reconfirmed`` on a second freeze,
    which would change the digest every training identity is bound to.
    """
    result = services.freeze_hard_split(persist=False)
    freeze = _plain(result["freeze"])
    canonical = {key: value for key, value in freeze.items() if key != "reconfirmed"}
    observed = paths.digest_document(canonical)
    if observed != policy["expected_split_sha256"] or (
            canonical.get("sequence_sha256") != policy["expected_sequence_sha256"]):
        raise ReadinessGate(
            f"the split this run derives digests to {observed} (sequence "
            f"{canonical.get('sequence_sha256')}); the {policy['phase_id']} phase pins "
            f"{policy['expected_split_sha256']} (sequence {policy['expected_sequence_sha256']}). "
            "Nothing was written and no update ran; the split is not invented or changed here.")
    paths.write_json(context.path("split_certificate.json"), _plain(result["certificate"]))
    target = context.path(spec.SPLIT_FREEZE)
    if not (target.is_file() and paths.digest_document(paths.read_json(target)) == observed):
        paths.write_json(target, canonical)
    spec.verify_training_phase_split(context, policy)
    return result


def stage_split(context, services, **_):
    """Freeze the sequence-only hard split and publish its certificate.

    The certificate is published **whether or not** it passes its power
    precondition, before any panel revision. An underpowered reserved set is
    reported with its actual n and the arithmetic; it is not relaxed.
    """
    policy = phase_policy(context.config)
    if policy is None:
        result = services.freeze_hard_split()
        paths.write_json(context.path("split_certificate.json"), _plain(result["certificate"]))
        paths.write_json(context.path(spec.SPLIT_FREEZE), _plain(result["freeze"]))
    else:
        result = _freeze_pinned_split(context, services, policy)
    certificate = result["certificate"]
    if not certificate["zero_violations"]:
        raise FeasibilityGate(
            f"the hard split certificate records {len(certificate['violations'])} violation(s). "
            "The table is published above before any revision.")
    counts = certificate["counts"]
    return {"status": "completed", "counts": counts,
            "power": certificate["power"], "reserved_status": certificate["reserved_status"],
            "underpowered": not counts["reserved_meets_minimum"],
            "retention": {"retained_for_training": counts["retained_for_training"],
                          "purged_within_radius": counts["purged_within_radius"],
                          "panel_eligible": counts["panel_eligible"],
                          "held_out": counts["reserved"] + counts["development"]},
            "note": ("the panel is the plan's 100 test + 100 development and is not enlarged: "
                     "+0.05 absolute Spearman is below the 95% half-width at every size this "
                     "population supports, so enlarging buys no resolvability while the radius "
                     "purge around a larger panel spends training data. An underpowered reserved "
                     "panel is a reported status, not a reason to relax the separation after "
                     "seeing it.")}


def profile_routes(context, services, *, routes=None, announce=None):
    """Profile ``routes`` (default: every enabled one) exactly as the profile stage does.

    The one place the profile budget, the schedule and the energy route's two-phase update
    count are read. The stage and the standalone profiler both call it, so their arguments
    cannot drift apart again (HR-06: the script had omitted the config's microbatches and
    steps, scheduled the energy route by ``energy_updates`` alone, and passed a ``views``
    argument the profiler never read).
    """
    profiles, schedule = {}, {}
    budget = context.config["profile"]
    for route in list(routes if routes is not None else context.config["routes"]["enabled"]):
        if route == "dwjs_energy":
            profiles[route] = services.profile_energy()
            schedule[route] = {"updates": (int(context.config["routes"][route]
                                               ["denoiser_updates"])
                                           + int(context.config["routes"][route]
                                                 ["energy_updates"]))}
        else:
            profiles[route] = services.profile(
                route, microbatches=tuple(budget.get("microbatches") or (1, 2, 4)),
                steps=int(budget.get("steps", 4)))
            schedule[route] = {"updates": int(context.config["routes"][route]["updates"])}
        if announce is not None:
            announce(route, profiles[route])
    return profiles, schedule


def guard_standalone_profile(context):
    """Refuse, before any CUDA or model work, a root the standalone profiler must not touch.

    Three refusals, and no ``force`` escape: a completed ``profile`` stage record (the
    stage record would still say reusable while ``gpu_profile.json`` changed under it), a
    freeze marker (the frozen protocol already embeds the profile), and a campaign lock that
    is held or inaccessible (a live or unobservable writer owns this root).
    """
    record = read_stage_record(context, "profile")
    require((record or {}).get("status") != "completed",
            "The profile stage already completed for this run root. Re-profiling would rewrite "
            f"{spec.PROFILE_JSON} beneath a stage record that still reads as reusable. Declare a "
            "new revision instead.")
    marker = context.path(spec.FREEZE_MARKER)
    require(not marker.is_file(),
            f"{marker.name} exists: this root is frozen and its protocol embeds the profile it "
            "was frozen with. A profile written now would disagree with it.")
    observed = lock_state(context.path(LOCK_FILE))
    require(observed["state"] in ("missing", "released"),
            f"The campaign lock is {observed['state']!r}"
            + (": a live writer owns this run root." if observed["state"] == "held" else
               ": its state could not be observed, so an unknown writer may own this run root."))
    return {"profile_stage": (record or {}).get("status", "not_run"),
            "freeze_marker": False, "lock": observed["state"]}


def standalone_profile(context, build, *, routes=None, announce=None):
    """Profile under the campaign's REAL lock, refusing first. Returns the profiles and forecast.

    ``build()`` constructs the services and is called only after every guard has passed and
    the lock is held, so a refused root never reaches CUDA or a model. The lock is held for
    the whole profile: a second writer (a supervisor launched meanwhile) fails to acquire it
    instead of interleaving with this one.
    """
    guard_standalone_profile(context)
    lock = CampaignLock(context.path(LOCK_FILE), owner=owner_identity())
    try:
        lock.acquire()
    except OSError as error:
        raise ValueError(f"The campaign lock could not be taken ({type(error).__name__}: "
                         f"{error}); the standalone profiler will not run without it.") from error
    try:
        # Re-check under the lock: the state may have changed between the guard and the take.
        guard_after = read_stage_record(context, "profile")
        require((guard_after or {}).get("status") != "completed"
                and not context.path(spec.FREEZE_MARKER).is_file(),
                "The run root changed while the lock was being taken")
        services = build()
        profiles, schedule = profile_routes(context, services, routes=routes, announce=announce)
        paths.write_json(context.path(spec.PROFILE_JSON), _plain(profiles))
        forecast = services.forecast(profiles, schedule=schedule,
                                     ledger_sample=services.ledger_sample())
        paths.write_json(context.path(spec.FORECAST_JSON), _plain(forecast))
        return profiles, forecast
    finally:
        lock.release()


def describe_run_root(context):
    """What is at this run root, and which sibling revisions exist. Read-only.

    A status page that prints ``0/17`` for a root that simply does not exist hides the
    likelier truth: the flight is under another revision. This names the resolved root, says
    whether it exists and holds records, and lists the sibling revision roots with their
    record counts.
    """
    run_root = context.run_root
    stages = run_root / STAGE_RECORDS

    def count(directory):
        return sum(1 for _ in directory.glob("*.json")) if directory.is_dir() else 0
    leaf = str(context.config.get("run_root_leaf") or run_root.name)
    siblings = []
    runs = context.artifact_root / "runs"
    if runs.is_dir():
        for revision in sorted(child for child in runs.iterdir() if child.is_dir()):
            candidate = revision / leaf
            if candidate == run_root or not candidate.is_dir():
                continue
            siblings.append({"revision": revision.name, "run_root": str(candidate),
                             "stage_records": count(candidate / STAGE_RECORDS),
                             "lock": lock_state(candidate / LOCK_FILE)["state"]})
    records = count(stages)
    return {"revision": context.revision, "run_root": str(run_root),
            "exists": run_root.is_dir(), "stage_records": records,
            "empty": records == 0, "sibling_revisions": siblings,
            "note": (None if records else
                     "no stage record exists at this run root. That means nothing has run under "
                     f"revision {context.revision!r}; it does not mean no flight exists"
                     + (": these other revision roots do exist -- "
                        + ", ".join(f"{row['revision']} ({row['stage_records']} records, lock "
                                    f"{row['lock']})" for row in siblings) if siblings else "."))}


def stage_profile(context, services, **_):
    """Measure the real GPU cost of every enabled route, then forecast from it.

    The profiled computation is the production objective set on real draws of every
    scheduled task, so the accepted microbatch is a statement about the configuration
    that runs and ``selected_shapes`` reads it back.
    """
    profiles, schedule = profile_routes(context, services)
    paths.write_json(context.path(spec.PROFILE_JSON), _plain(profiles))
    unfit = [route for route, block in profiles.items() if block["chosen_microbatch"] is None]
    if unfit:
        raise FeasibilityGate(
            f"no microbatch passed all three spill signals for {unfit}. The route does not fit this "
            "card at the declared shapes, and an absent OOM is not evidence that it does.")
    forecast = services.forecast(profiles, schedule=schedule,
                                 ledger_sample=services.ledger_sample())
    paths.write_json(context.path(spec.FORECAST_JSON), _plain(forecast))
    return {"status": "completed",
            "chosen_microbatch": {route: block["chosen_microbatch"]
                                  for route, block in profiles.items()},
            "total_hours": forecast["total_hours"],
            "with_reserve_hours": forecast["with_reserve_hours"],
            "drop_order": forecast["drop_order"],
            "note": ("a measured forecast. It is reported whether or not it fits, and the six "
                     "exposure counters are multiplied separately rather than derived from one "
                     "another.")}


def stage_smoke(context, services, **_):
    _require_phase_split(context)
    results ={route: services.smoke(route, directory=context.path("smoke", route))
               for route in context.config["routes"]["enabled"]}
    paths.write_json(context.path("smoke.json"), _plain(results))
    failed = sorted(route for route, block in results.items() if not block["passed"])
    if failed:
        return {"status": "failed", "reason": f"smoke failed for {failed}",
                "detail": {route: results[route]["failures"] for route in failed}}
    return {"status": "completed", "routes": sorted(results),
            "checks": {route: block["checks"] for route, block in results.items()}}


def _require_phase_split(context):
    """A phase config must be valid and its split must be the pinned one; else a readiness gate.

    A legacy config has no phase and this returns ``None`` without reading anything.
    """
    try:
        policy = phase_policy(context.config)
        if policy is not None:
            spec.verify_training_phase_split(context, policy)
    except ValueError as error:
        raise ReadinessGate(f"training phase: {error}") from error
    return policy


def stage_freeze(context, services, **_):
    """Snapshot the repository closure, bind the external table, resolve the protocol."""
    _require_phase_split(context)
    snapshot = spec.source_snapshot(context)
    external = spec.verify_external_sources(context, label="freeze")
    revision = spec.write_revision_marker(
        context, kind=context.config["revisions"]["kind"],
        supersedes=context.config["revisions"].get("supersedes"),
        reason=context.config["revisions"].get("reason"),
        frozen_after=context.config["revisions"].get("frozen_after") or ())
    split = paths.read_json(context.path(spec.SPLIT_FREEZE)) \
        if context.path(spec.SPLIT_FREEZE).is_file() else None
    profile = paths.read_json(context.path(spec.PROFILE_JSON)) \
        if context.path(spec.PROFILE_JSON).is_file() else None
    forecast = paths.read_json(context.path(spec.FORECAST_JSON)) \
        if context.path(spec.FORECAST_JSON).is_file() else None
    protocol = spec.resolved_protocol(context, snapshot=snapshot, external=external, split=split,
                                      profile=profile, forecast=forecast, revision=revision)
    if protocol["unresolved"]:
        raise ReadinessGate(
            f"the resolved protocol still has unresolved fields {protocol['unresolved']}. "
            "Production runs against a filled protocol.")
    return {"status": "completed", "identity": spec.combined_identity(context),
            "revision": {"revision": revision["revision"], "kind": revision["kind"]},
            "repository_files": snapshot["file_count"], "external_files": external["files"]}


def _guarded(context, stage, body, *, selection_frozen=False):
    spec.verify_source_snapshot(context, label=stage)
    spec.verify_external_sources(context, label=stage)
    spec.require_resolved(context, label=stage)
    spec.require_revision_discipline(context, stage=stage, selection_frozen=selection_frozen)
    # Under a training phase the pinned split is re-verified before any production update.
    _require_phase_split(context)
    return body()


def stage_bakeoff(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "bakeoff",
                      lambda: services.run_bakeoff(heartbeat=heartbeat))
    paths.write_json(context.path("bakeoff.json"), _plain(result))
    record = {"status": "completed" if result["all_terminal"] else "partial",
              "routes": sorted(result["routes"]), "completed": result["completed"],
              "remaining": result["remaining"],
              "reason": (None if result["all_terminal"] else
                         "not every route reached a terminal status; the flight remains in "
                         "progress")}
    policy = phase_policy(context.config)
    if policy is not None:
        record["training_phase"] = {"phase_id": policy["phase_id"],
                                    "policy_sha256": policy["policy_sha256"],
                                    "admission": policy["admission"],
                                    "selection_metric": policy["selection_metric"]}
    return record


def stage_route_gate(context, services, **_):
    result = _guarded(context, "route_gate", services.evaluate_route_gate)
    paths.write_json(context.path("route_gate.json"), _plain(result))
    if not result["advancing"]:
        raise FeasibilityGate(
            "no route cleared G1 on the predeclared Pareto criterion. The measured table is "
            f"published above. Decisions: {result['decisions']}")
    return {"status": "completed", "advancing": result["advancing"],
            "decisions": result["decisions"], "winner": result["winner"]}


def stage_controls(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "controls",
                      lambda: services.run_controls(heartbeat=heartbeat))
    paths.write_json(context.path("controls.json"), _plain(result))
    if not result["survives"]:
        raise FeasibilityGate(
            "the winner's gain did not survive E17 or E19. That is a reported distance or shortcut "
            f"artefact and it ends the campaign with a finding: {result['failed']}")
    return {"status": "completed", "arms": sorted(result["arms"]),
            "matched_exposure": result["matched_exposure"], "survives": True}


def stage_priors(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "priors",
                      lambda: services.run_priors(heartbeat=heartbeat))
    paths.write_json(context.path("priors.json"), _plain(result))
    distilled = {name: block.get("checkpoint")
                 for name, block in sorted(dict(result.get("distillations") or {}).items())}
    return {"status": "completed", "teachers": sorted(result["teachers"]),
            "skipped": result["skipped"],
            # The per-teacher decisions are in priors.json; this record keeps the names and
            # the distilled checkpoints so a later stage can resolve its lineage from either.
            "distilled_checkpoints": distilled,
            "lineage": {name: block.get("lineage")
                        for name, block in sorted(dict(result.get("distillations") or {}).items())},
            "note": ("a conditional teacher is skipped only with recorded prerequisite evidence, "
                     "never with a placeholder implementation")}


def stage_posttrain(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "posttrain",
                      lambda: services.run_posttrain(heartbeat=heartbeat))
    paths.write_json(context.path("posttrain.json"), _plain(result))
    # HR-02: the pair bank is PER METHOD (each arm draws from its own under the reuse cap);
    # ``run_posttrain`` has no top-level one, and reading ``result["pair_bank"]`` raised.
    return {"status": "completed", "methods": sorted(result["methods"]),
            "reference_sha256": result["reference_sha256"],
            "pair_bank": {name: block["pair_bank"]
                          for name, block in sorted(result["methods"].items())},
            "selected_method": (result.get("selection") or {}).get("best"),
            "checkpoint": result.get("selected_checkpoint"),
            "lineage": result.get("lineage"),
            "note": ("the selected method and its checkpoint are named here and in posttrain.json "
                     "so decoding, the report and the recipe resolve them explicitly")}


def stage_decoding(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "decoding",
                      lambda: services.run_decoding(heartbeat=heartbeat))
    paths.write_json(context.path("decoding.json"), _plain(result))
    return {"status": "completed", "strategies": sorted(result["strategies"]),
            "clustering": result["clustering"], "draw_prefixes": result["draw_prefixes"],
            "lineage": result.get("lineage")}


def confirmation_prerequisites(context):
    """Which selections the confirmation revision must be frozen AFTER. Computed.

    The revision marker's ``frozen_after`` list is what
    ``require_revision_discipline`` checks, and writing it by hand in a config is how
    a confirmation gets frozen before a selection it claims to follow. This derives
    it from the stage records that exist, so a missing selection is a missing entry
    and the discipline check sees it.
    """
    order = ("route_gate", "controls", "priors", "posttrain", "decoding")
    completed = [stage for stage in order
                 if (read_stage_record(context, stage) or {}).get("status") == "completed"]
    return {"frozen_after": completed,
            "missing": [stage for stage in order if stage not in completed],
            "rule": ("architecture, objectives, mix, decoding and checkpoint selection are all "
                     "frozen before the confirmation revision exists. The list is derived from "
                     "the stage records rather than declared.")}


def stage_confirmation_freeze(context, services, **_):
    """Promote the run to the confirmation revision, after every selection is frozen.

    The automatic path the review named as missing. It writes a confirmation revision
    marker whose ``frozen_after`` list is **derived** from the completed selection
    stages, so ``require_revision_discipline`` refuses the confirmation stage until
    every one of them has run rather than trusting a hand-written list.
    """
    prerequisites = confirmation_prerequisites(context)
    if prerequisites["missing"]:
        raise ReadinessGate(
            f"the confirmation revision cannot be frozen: {prerequisites['missing']} have not "
            "completed. A confirmation frozen before a selection is evidence about a "
            "specification that selection had already seen.")
    # The recipe is assembled and checked for replayability BEFORE the revision marker is
    # written. The marker used to come first, so a recipe that could not be assembled (or
    # could not be replayed) left a confirmation revision with nothing frozen under it.
    # A recipe naming a qualified teacher, a post-training method or a decoding strategy is
    # refused (HR-08): replay of those is not implemented and is never reported as executed.
    recipe = services.frozen_recipe()
    try:
        plan = lineage.require_replayable(recipe, where="confirmation_freeze")
    except lineage.RecipeNotReplayable as error:
        raise ReadinessGate(str(error)) from error
    revision = spec.write_revision_marker(
        context, kind=spec.REVISION_CONFIRMATION,
        supersedes=context.config["revisions"].get("current"),
        reason=("every selection stage completed; the recipe is frozen and the reserved panel is "
                "opened once under it"),
        frozen_after=prerequisites["frozen_after"])
    paths.write_json(context.path("frozen_recipe.json"), recipe)
    return {"status": "completed", "revision": revision["revision"],
            "kind": revision["kind"], "frozen_after": prerequisites["frozen_after"],
            "recipe_sha256": recipe["sha256"], "replay_plan": plan,
            "note": ("the frozen_after list is derived from the completed selection stages, not "
                     "declared in the config.")}


def stage_confirm(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "confirm",
                      lambda: services.run_confirmation(heartbeat=heartbeat),
                      selection_frozen=True)
    paths.write_json(context.path("confirmation.json"), _plain(result))
    if not result["complete"]:
        return {"status": "partial", "seeds": result["seeds"],
                "reason": ("the confirmation seeds are still running. The status says RUNNING, "
                           "not complete, and the reserved test stays closed.")}
    return {"status": "completed", "seeds": result["seeds"],
            "reserved_opened": result["reserved_opened"],
            "reserved_n": result["reserved_n"]}


def stage_report(context, services, *, heartbeat=None, **_):
    result = _guarded(context, "report",
                      lambda: services.build_report(heartbeat=heartbeat))
    paths.write_json(context.path("report.json"), _plain(result))
    return {"status": "completed", "panels": result["panels"],
            "unsupported_panels": result["unsupported"],
            "complete_flight": result["completion"]["complete"],
            "lineage": result.get("lineage")}


def stage_verify(context, services, **_):
    document = verify(context, services)
    paths.write_json(context.path("verification.json"), _plain(document))
    record = {"status": "completed" if document["passed"] else "failed",
              "checks": document["checks"], "passed": document["passed"],
              "reason": None if document["passed"] else "one or more verification checks failed"}
    if "training_phase" in document:
        record["training_phase_completed"] = document["training_phase_completed"]
        record["complete_flight"] = False
    return record


STAGE_FUNCTIONS = {
    "acquire": stage_acquire, "preflight": stage_preflight, "sources": stage_sources,
    "split": stage_split, "profile": stage_profile, "smoke": stage_smoke,
    "freeze": stage_freeze, "bakeoff": stage_bakeoff, "route_gate": stage_route_gate,
    "controls": stage_controls, "priors": stage_priors, "posttrain": stage_posttrain,
    "decoding": stage_decoding, "confirmation_freeze": stage_confirmation_freeze,
    "confirm": stage_confirm, "report": stage_report, "verify": stage_verify}


def unmet_prerequisites(context, stage):
    """Which strict prerequisites of ``stage`` have not completed. A list, not a bool."""
    return [name for name in STRICT_PREREQUISITES.get(str(stage), ())
            if (read_stage_record(context, name) or {}).get("status") != "completed"]


# ---------------------------------------------------------------------------
# verification: completeness, not correctness
# ---------------------------------------------------------------------------

#: ``verify`` is the stage doing the verifying, so its own record does not exist yet
#: when it runs. Including it in the required list made the check fail on every first
#: run and pass only on a rerun, which is a property of the check rather than of the
#: flight.
VERIFIED_STAGES = tuple(stage for stage in REQUIRED_STAGES if stage != "verify")


def _identity_differences(recorded, expected):
    recorded, expected = dict(recorded or {}), dict(expected or {})
    return sorted(key for key in set(recorded) | set(expected)
                  if recorded.get(key) != expected.get(key))


def _verify_route_checkpoints(context, services, policy, route, add):
    """Every checkpoint of one route against the identity it MUST carry, loaded one at a time.

    The expected identity is recomputed from the frozen run (native base, route, seed, arm,
    config, source, external table, condition vocabulary, pinned split) rather than read back
    from a checkpoint, so a checkpoint from another run, seed, arm, config or split is refused
    even when it agrees with its neighbours. Only small scalars enter the record; the loaded
    payload is dropped before the next one is opened.
    """
    import math

    import torch
    block = dict(context.config["routes"][route])
    directory = context.path("bakeoff", route)
    updates = int(policy["updates"])
    expected = services.training_identity(route, arm="full", seed=int(block["seed"]))
    add(f"phase::identity_split_pinned::{route}",
        expected.get("split_sha256") == policy["expected_split_sha256"],
        split_sha256=expected.get("split_sha256"))
    run_path = directory / "run.json"
    if run_path.is_file():
        run = paths.read_json(run_path)
        differing = _identity_differences(run.get("identity"), expected)
        add(f"phase::run_record::{route}",
            run.get("updates") == updates and not differing
            and (run.get("training_phase") or {}).get("policy_sha256") == policy["policy_sha256"]
            and (run.get("training_phase") or {}).get("selection_metric")
            == policy["selection_metric"],
            updates=run.get("updates"), identity_differences=differing,
            policy_sha256=(run.get("training_phase") or {}).get("policy_sha256"))
    else:
        add(f"phase::run_record::{route}", False, reason=f"{run_path.name} is absent")
    wanted = [("last.pt", updates), ("best.pt", None),
              *[(f"step{step}.pt", int(step)) for step in policy["milestones"]]]
    for name, step in wanted:
        target = directory / name
        label = f"phase::checkpoint::{route}::{name}"
        if not target.is_file():
            add(label, False, reason="absent")
            continue
        payload = torch.load(str(target), map_location="cpu", weights_only=False)
        try:
            differing = _identity_differences(payload.get("identity"), expected)
            found = int(payload.get("step", -1))
            metric = payload.get("best_metric")
            ok = not differing and (found == step if step is not None else 1 <= found <= updates)
            if name == "best.pt":
                ok = ok and metric is not None and math.isfinite(float(metric))
            add(label, ok, step=found, identity_differences=differing,
                identity_sha256=(payload.get("identity") or {}).get("identity_sha256"),
                bytes=target.stat().st_size)
        finally:
            del payload


#: What the held stages write (see ``stage_*`` above and the services' run directories).
_HELD_STAGE_FILES = ("route_gate.json", "controls.json", "priors.json", "posttrain.json",
                     "decoding.json", "confirmation.json", "report.json",
                     "frozen_recipe.json", "reserved_opening.json")
_HELD_STAGE_DIRECTORIES = ("controls", "priors", "posttrain", "decoding", "confirm", "report")


def _verify_training_phase(context, services, policy, add):
    """The phase-specific completeness checks. Small records only; no tensors in the JSON."""
    try:
        pin = spec.verify_training_phase_split(context, policy)
        add("phase::split_pinned", True, **pin)
    except ValueError as error:
        add("phase::split_pinned", False, reason=str(error))
    # A held stage leaves no record at all: the phase guard refuses before anything is written,
    # so a failed or partial record is as much a violation as a completed one. Evidence found
    # here is reported and left in place.
    for stage in policy["held_stages"]:
        record = read_stage_record(context, stage)
        status = "not_run" if record is None else str((record or {}).get("status", "recorded"))
        add(f"phase::held::{stage}", record is None, status=status)
    for logical in _HELD_STAGE_FILES:
        add(f"phase::absent::{logical}", not context.path(logical).exists())
    for logical in _HELD_STAGE_DIRECTORIES:
        directory = context.path(logical)
        present = directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
        add(f"phase::absent::{logical}/", not present)
    resolved = context.path("resolved_protocol.json")
    block = paths.read_json(resolved).get(spec.TRAINING_PHASE_KEY) if resolved.is_file() else None
    add("phase::resolved_protocol_carries_policy",
        bool(block) and block.get("policy_sha256") == policy["policy_sha256"])
    if services is None:
        add("phase::checkpoints", False,
            reason="checkpoint identities are recomputed by the services, and none were supplied")
        return
    for route in policy["routes"]:
        _verify_route_checkpoints(context, services, policy, route, add)


def verify(context, services=None):
    """Assert every required stage ran and every declared artifact exists.

    This is a **completeness** check. Green checks mean the required stages ran and
    their artifacts are on disk with the recorded digests; they do not mean the
    science is right, and the report says so beside the count.

    ``verify`` itself is excluded from the list it checks: it is running, so its own
    record is not written yet, and requiring it made the first run of a complete
    flight fail and the second one pass.

    Under a declared training phase the checked stage list is the phase's own, the phase
    checks are added, and the document says ``complete_flight: false`` whatever else passed.
    """
    checks, problems = [], []
    phase_problem, policy = None, None
    try:
        policy = phase_policy(context.config)
    except ValueError as error:
        phase_problem = str(error)
    verified = (VERIFIED_STAGES if policy is None else
                tuple(stage for stage in policy["stages"] if stage != "verify"))
    summary = stage_summary(context, verified)
    if phase_problem is not None:
        checks.append({"check": "phase::declaration_valid", "passed": False,
                       "reason": phase_problem})
        problems.append(f"the training phase declaration is invalid: {phase_problem}")
    for stage in verified:
        block = summary[stage]
        ok = block["status"] in ("completed",)
        checks.append({"check": f"stage::{stage}", "passed": ok, "status": block["status"]})
        if not ok:
            problems.append(f"{stage} is {block['status']}")
    checks.append({"check": "verify_excludes_itself", "passed": True,
                   "reason": ("verify is the stage doing the verifying and its record is written "
                              "after it returns. Requiring it would fail every first run.")})
    for logical in ("native_evidence.json", "source_inventory.json", "split_certificate.json",
                    spec.PROFILE_JSON, spec.FORECAST_JSON, spec.EXTERNAL_MARKER,
                    spec.REVISION_MARKER, "resolved_protocol.json"):
        exists = context.path(logical).is_file()
        checks.append({"check": f"artifact::{logical}", "passed": exists})
        if not exists:
            problems.append(f"{logical} is absent")
    try:
        spec.verify_source_snapshot(context, label="verify")
        checks.append({"check": "repository_snapshot_unchanged", "passed": True})
    except ValueError as error:
        checks.append({"check": "repository_snapshot_unchanged", "passed": False,
                       "reason": str(error)})
        problems.append("the repository source snapshot changed")
    try:
        spec.verify_external_sources(context, label="verify")
        checks.append({"check": "external_sources_unchanged", "passed": True})
    except ValueError as error:
        checks.append({"check": "external_sources_unchanged", "passed": False,
                       "reason": str(error)})
        problems.append("the external source table changed")
    # The reserved-opening guard, if it exists, must match the frozen recipe the
    # confirmation stage wrote. A guard bound to a different selection means the
    # panel was opened under a specification that is no longer the one being reported.
    guard = context.path("reserved_opening.json")
    recipe = context.path("frozen_recipe.json")
    if guard.is_file():
        opened = paths.read_json(guard)
        frozen = paths.read_json(recipe) if recipe.is_file() else {}
        matches = bool(frozen and opened.get("frozen_selection_sha256") == frozen.get("sha256"))
        checks.append({"check": "reserved_opening_bound_to_frozen_selection",
                       "passed": matches,
                       "guard_sha256": opened.get("frozen_selection_sha256"),
                       "recipe_sha256": frozen.get("sha256")})
        if not matches:
            problems.append("the reserved opening is not bound to the frozen recipe on disk")
    if policy is not None:
        def add(name, passed, **fields):
            checks.append(dict({"check": name, "passed": bool(passed)}, **fields))
            if not passed:
                problems.append(f"{name} failed" + (f": {fields['reason']}" if "reason" in fields
                                                     else ""))
        _verify_training_phase(context, services, policy, add)
    document = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "verification",
                "checks": len(checks), "results": checks, "problems": problems,
                "passed": not problems,
                "verified_stages": list(verified),
                "reserved_opened": guard.is_file(),
                "asserts": ("COMPLETENESS, not correctness. Green checks mean every required "
                            "stage ran and every declared artifact is present with the recorded "
                            "digests. They say nothing about whether the science is right.")}
    if policy is not None:
        document["training_phase"] = {"phase_id": policy["phase_id"],
                                      "policy_sha256": policy["policy_sha256"],
                                      "admission": policy["admission"],
                                      "held_stages": list(policy["held_stages"])}
        document["training_phase_completed"] = not problems
        document["complete_flight"] = False
        document["asserts"] += (" A completed training phase is NOT a complete flight and NOT "
                                "competition admission: the held stages did not run.")
    elif phase_problem is not None:
        # A phase was declared but is malformed: nothing about it passes, and nothing is complete.
        document["training_phase_completed"] = False
        document["complete_flight"] = False
    return document


# ---------------------------------------------------------------------------
# stage identity, reuse and the runner
# ---------------------------------------------------------------------------

UPSTREAM_DEPENDENT = ("bakeoff", "route_gate", "controls", "priors", "posttrain", "decoding",
                      "confirm", "report", "verify")


def stage_identity(context, stage):
    marker = context.path(spec.EXTERNAL_MARKER)
    snapshot = context.path("routes_source_snapshot.json")
    identity = {"stage": str(stage), "revision": context.revision,
                "config_sha256": paths.sha256_file(context.config_path),
                "external_sha256": (paths.read_json(marker)["external_sha256"]
                                    if marker.is_file() else None),
                "snapshot_sha256": (paths.read_json(snapshot)["snapshot_sha256"]
                                    if snapshot.is_file() else None)}
    if str(stage) in UPSTREAM_DEPENDENT:
        upstream = read_stage_record(context, "bakeoff")
        identity["bakeoff_outcome_sha256"] = (
            None if upstream is None else
            paths.sha256_text(paths.canonical_json(
                {key: upstream.get(key) for key in ("status", "completed", "remaining")})))
    return identity


def stage_is_reusable(context, stage):
    record = read_stage_record(context, stage)
    if record is None:
        return False, "no record"
    if record.get("status") != "completed":
        return False, f"the recorded status is {record.get('status')!r}"
    recorded = record.get("identity")
    if recorded is None:
        return False, "the record carries no identity to compare"
    current = stage_identity(context, stage)
    differing = sorted(key for key in set(recorded) | set(current)
                       if recorded.get(key) != current.get(key))
    if differing:
        return False, f"the stage identity changed in {differing}"
    return True, "completed under the same source, external table and configuration"


def run_stage(context, stage, services, *, reuse=True, **kwargs):
    require(stage in STAGE_FUNCTIONS, f"Unknown stage {stage!r}")
    # Before reuse, and before any effect: a held stage neither runs nor is "reused" as done.
    policy = require_stage_allowed(context, stage)
    # Under a phase, verify inspects checkpoints on disk, so a stale "completed" record is never
    # reused as though it had looked at the files now present.
    if reuse and not (policy is not None and stage == "verify"):
        reusable, reason = stage_is_reusable(context, stage)
        if reusable:
            record = dict(read_stage_record(context, stage))
            record["reused"] = {"reason": reason, "at": paths.utc_now()}
            return record
    started = time.perf_counter()
    try:
        document = _plain(STAGE_FUNCTIONS[stage](context, services, **kwargs))
        document["wall_seconds"] = time.perf_counter() - started
        document["identity"] = stage_identity(context, stage)
        return write_stage_record(context, stage, document)
    except (ReadinessGate, lineage.LineageError) as error:
        # A missing or ambiguous checkpoint (or an unreplayable recipe) is a missing
        # prerequisite, not a crash: the stage stops with the reason recorded.
        return write_stage_record(context, stage, {
            "status": "blocked_readiness", "reason": str(error),
            "wall_seconds": time.perf_counter() - started,
            "consequence": ("a prerequisite is missing. The flight pauses here; no scientific "
                            "claim is made and nothing is marked complete.")})
    except FeasibilityGate as error:
        return write_stage_record(context, stage, {
            "status": "blocked_feasibility", "reason": str(error),
            "wall_seconds": time.perf_counter() - started,
            "consequence": ("a declared scientific criterion failed. Evidence is retained, "
                            "dependent stages stop, and independent stages still run. This is a "
                            "legitimate outcome, not a bug.")})


def run_all(context, services, *, stages=None, heartbeat=None, **kwargs):
    # An omitted list is the phase's own (the full flight for a legacy config). An explicit list
    # naming a held stage is refused whole, before the first stage of it has any effect.
    stages = resolve_stages(context, stages)
    policy = phase_policy(context.config)
    required_stages = STAGES if policy is None else tuple(policy["stages"])
    blocked, records = {}, {}
    for stage in stages:
        if heartbeat is not None:
            heartbeat.beat(force=True, phase="stage", stage=stage,
                           completed=sorted(name for name, record in records.items()
                                            if record["status"] == "completed"))
        if stage in blocked:
            records[stage] = write_stage_record(context, stage, {
                "status": "not_run", "reason": blocked[stage],
                "consequence": "a required stage that did not run; the flight is not complete"})
            continue
        unmet = unmet_prerequisites(context, stage)
        if unmet:
            # A partial bake-off legitimately lets the independent downstream stages
            # run on what exists. A partial SELECTION does not let the confirmation
            # freeze a recipe that no selection produced, so this stage is held rather
            # than launched on absent prerequisites.
            records[stage] = write_stage_record(context, stage, {
                "status": "not_run",
                "reason": f"strict prerequisites have not completed: {unmet}",
                "consequence": ("this stage freezes or consumes a selection. Running it on absent "
                                "prerequisites would record a selection nothing selected.")})
            for dependent in DEPENDENTS.get(stage, ()):
                blocked.setdefault(dependent, f"blocked by {stage}: unmet prerequisites {unmet}")
            continue
        record = run_stage(context, stage, services, heartbeat=heartbeat, **kwargs)
        records[stage] = record
        usable_partial = (stage in ("bakeoff", "confirm") and record["status"] == "partial")
        if usable_partial:
            records[stage] = dict(record, downstream=(
                "some work reached a terminal status, so the independent downstream stages run on "
                "what exists. The queue is incomplete and the completion check says so."))
            continue
        if record["status"].startswith("blocked") or record["status"] in ("failed", "partial"):
            for dependent in DEPENDENTS.get(stage, ()):
                blocked.setdefault(dependent, f"blocked by {stage}: {record.get('reason')}")
    completed = [name for name, record in records.items() if record["status"] == "completed"]
    document = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "run_all",
                "stages": {name: record["status"] for name, record in records.items()},
                "completed_stages": completed, "blocked": blocked,
                "required_stages": list(required_stages),
                "all_required_completed": all(
                    (records.get(name) or read_stage_record(context, name) or {}).get("status")
                    == "completed" for name in required_stages),
                "claim": ("a stage list with statuses. Completion of the FLIGHT is asserted only by "
                          "the verify stage against the required stage list; a launch, a running "
                          "worker and a partial stage list are three different facts."),
                "finished_at": paths.utc_now()}
    if policy is not None:
        # ``all_required_completed`` ranges over the PHASE's stages here. That is a completed
        # training phase, which is a different fact from a complete flight.
        document["scope"] = "training_phase"
        document["training_phase"] = {"phase_id": policy["phase_id"],
                                      "policy_sha256": policy["policy_sha256"],
                                      "admission": policy["admission"],
                                      "held_stages": list(policy["held_stages"])}
        document["training_phase_completed"] = document["all_required_completed"]
        document["complete_flight"] = False
    paths.write_json(context.path("run_all.json"), document)
    return document


def supervise(context, services, *, stages=None, heartbeat_seconds=15.0, owner=None,
              max_restarts=0, **kwargs):
    """Hold the single-writer lock, beat, run the flight, publish a truthful status."""
    # Resolved before the lock is taken, so a refused list leaves no lock, heartbeat or status.
    stages = resolve_stages(context, stages)
    lock = CampaignLock(context.path(LOCK_FILE), owner=owner or owner_identity())
    beat = Heartbeat(context.path(HEARTBEAT_JSON), owner=lock.owner,
                     every=float(heartbeat_seconds))
    started = paths.utc_now()
    with lock.held(), beat.running():
        beat.beat(force=True, phase="starting", stages=list(stages))
        attempts, document, failure = 0, None, None
        while True:
            attempts += 1
            try:
                document = run_all(context, services, stages=stages, heartbeat=beat, **kwargs)
                break
            except BaseException as error:                 # noqa: BLE001 - recorded, re-raised
                failure = f"{type(error).__name__}: {error}"
                paths.write_json(context.path("supervisor_failure.json"), {
                    "schema_version": CAMPAIGN_SCHEMA, "record_kind": "supervisor_failure",
                    "attempt": attempts, "error": failure, "at": paths.utc_now(),
                    "consequence": ("a crash, not a scientific outcome. The status says failed; "
                                    "nothing is reported as complete.")})
                if attempts > int(max_restarts):
                    _publish(context, stages=stages, started=started, owner=lock.owner,
                             status="failed", failure=failure, document=None)
                    raise
                beat.beat(force=True, phase="restarting", attempt=attempts, error=failure)
        status = ((STATUS_TRAINING_PHASE_COMPLETED if document.get("scope") == "training_phase"
                   else STATUS_COMPLETED) if document["all_required_completed"]
                  else "stopped" if document["blocked"] else "incomplete")
        beat.beat(force=True, phase="finished", status=status)
        return _publish(context, stages=stages, started=started, owner=lock.owner,
                        status=status, failure=None, document=document)


def _publish(context, *, stages, started, owner, status, failure, document):
    record = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "campaign_status",
              "campaign_id": context.campaign_id, "revision": context.revision,
              "status": status, "failure": failure, "owner": dict(owner),
              "started_at": started, "finished_at": paths.utc_now(),
              "stages": stage_summary(context, stages), "run_all": document,
              "launch_versus_completion": (
                  "a verified worker and advancing journals establish that this flight is RUNNING. "
                  "Scientific completion is a separate fact, asserted only when every required "
                  "stage has completed."),
              "lock": lock_state(context.path(LOCK_FILE))}
    paths.write_json(context.path(STATUS_JSON), record)
    return record


def launch_detached(context, *, python=None, script="scripts/her2_routes_supervisor.py",
                    arguments=()):
    """Start the supervisor as a detached process so it outlives this session.

    Background jobs started inside an agent turn are killed at the turn boundary --
    nanosecond-identical kills confirmed a group kill -- so this is a detached
    process with its own log rather than a child of the caller.
    """
    executable = str(python or sys.executable)
    target = context.repository_root / script
    require(target.is_file(), f"{script} is not present")
    observed = lock_state(context.path(LOCK_FILE))
    # Only a lock that is observably free authorises a launch. ``inaccessible`` means the
    # question could not be answered, and an unknown state must never permit a second
    # detached writer: refused, with no escape flag.
    require(observed["state"] in ("missing", "released"),
            f"The campaign lock is {observed['state']!r}: "
            + ("a live writer holds it; inspect health before launching another."
               if observed["state"] == "held" else
               "its state could not be observed, so a launch is not authorised. Resolve the "
               "access problem and inspect health first."))
    log = context.path("supervisor.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, str(target), *[str(value) for value in arguments],
               "--config", str(context.config_path), "--artifact-root", str(context.artifact_root),
               "--revision", context.revision]
    options = {}
    if sys.platform == "win32":
        options["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        options["start_new_session"] = True
    with log.open("ab") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   cwd=str(context.repository_root), close_fds=True, **options)
    record = {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "launch",
              "pid": int(process.pid), "command": command, "log": str(log),
              "launched_at": paths.utc_now(),
              "verification_required": ("a PID is not progress. Launch is verified by the lock "
                                        "being held, the heartbeat advancing and the per-stage "
                                        "journals gaining records.")}
    paths.write_json(context.path("launch.json"), record)
    return record


def launch_health(context, *, minimum_steps=1):
    """Is a worker alive, and has anything advanced? Two different questions."""
    launch = context.path("launch.json")
    record = paths.read_json(launch) if launch.is_file() else None
    lock = lock_state(context.path(LOCK_FILE))
    heartbeat_path = context.path(HEARTBEAT_JSON)
    heartbeat = paths.read_json(heartbeat_path) if heartbeat_path.is_file() else None
    steps = 0
    for progress in (sorted(context.run_root.rglob("progress.json"))
                     if context.run_root.is_dir() else ()):
        try:
            steps += int(paths.read_json(progress).get("completed") or 0)
        except Exception:                                  # noqa: BLE001 - a partial file
            continue
    done = sum(1 for block in stage_summary(context).values()
               if block["status"] == "completed")
    # Three answers, not two: held (running), missing/released (not running) and
    # inaccessible (UNKNOWN). ``lock_state`` documents that unknown is never reported as "the
    # owner is gone"; collapsing it to False here reported a live flight as stopped (HR-05).
    unknown = lock["state"] == "inaccessible"
    running = None if unknown else lock["state"] == "held"
    progressed = bool(steps >= int(minimum_steps) or done > 0)
    return {"schema_version": CAMPAIGN_SCHEMA, "record_kind": "launch_health",
            "launch_record": record, "lock": lock, "heartbeat": heartbeat,
            "journalled_steps": steps, "stages_completed": done,
            "worker_running": running,
            "progress_observed": progressed,
            "health": ("unknown_lock_state" if unknown
                       else "running_with_progress" if running and progressed
                       else "running_without_progress_yet" if running else "not_running"),
            "claim": ("scientific completion is never inferred from a successful launch. This "
                      "answers 'is a writer alive' and 'has anything advanced', and nothing more.")}


def build_services(context, *, device="cuda"):
    """The concrete runtime, imported lazily so orchestration is testable without torch."""
    from .her2_routes_services import FlightServices
    return FlightServices(context, device=device)
