"""The pilot run: one arm, one fresh model, one loop, one snapshot, one record.

This is an independent loop. It does not call, subclass or patch ``FlightServices.train`` or its
dispatch; it uses the services' public parts -- ``build_route``, ``build_heads``,
``build_optimizer`` (Adafactor for the 150M route, never substituted), ``route_learning_rate``,
``task_settings``, ``selected_shapes``, ``fresh_state``, ``curriculum_stream`` and
``tasks.dispatch`` -- and owns accumulation, clipping, the finite checks, the warm-up and the
update. A run starts from the pretrained model and ends at ``updates``; it has no resume, no
plateau schedule and no best-checkpoint selection, because there is nothing to select between.

**What an arm is.** ``baseline`` (suffix masks, distance-1 pairs), ``masks`` (subset masks,
distance-1 pairs), ``pairs`` (suffix masks, distances 1-3) and ``combined`` (subset masks, distances
1-3). All four keep the base reconstruction, heads and measured sources, and draw the base stream
first, unchanged, so their source, task and quartet sequences agree. They are comparisons INSIDE this
pilot, not a replay of any earlier flight, and nothing here says the adaptation improves biology.

**What is written** (all under the fresh run root): ``progress.jsonl`` (one line per update),
``snapshot.pt`` (the final model and heads, for evaluation only: it holds no optimizer, scheduler or
generator and cannot resume), ``evaluation.json`` (before and after), ``inputs.json`` (every identity
consumed), ``current_bins.json`` (the complete current quality-bin thresholds and support and the
fingerprint of the population-to-bin mapping, written before any model is built), ``runtime.json``
(the prepared environment and the interpreter and library versions and locations imported at the
START of the run, before the inputs, model and optimizer; :data:`RUNTIME_CAPTURE_STAGE`), ``run.json``
on success (with the libraries imported by its end) or ``failure.json`` on any error. A failed run
is never recorded as a smaller success.

**What the identity covers.** ``code`` is the hash of EVERY file in the static import closure of the
entry script (the pilot package AND the inherited services, tasks, sources, native model and
optimizer it executes), read from the code root. Third-party packages are recorded by version and
location only: the dependency acquisition manifest is a pip install report with two wheel archive
hashes, not a byte pin of the installed tree, and nothing here claims the dependencies are
byte-verified.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from ...models.her2_routes_heads import HeadAccumulator
from .. import her2_replay_spec as replay_spec
from .. import her2_routes_checkpoints as checkpoints
from .. import her2_routes_controls as controls
from .. import her2_routes_data as data_lib
from .. import her2_routes_env as env_lib
from .. import her2_routes_spec as spec
from .. import her2_routes_tasks as tasks_lib
from .. import her2_support_paths as paths
from ..her2_runtime import require
from . import config as config_lib
from . import draws as draws_lib
from . import evaluation as evaluation_lib
from . import inputs as inputs_lib
from . import pairs as pairs_lib
from . import roots as roots_lib

SNAPSHOT_SCHEMA = "her2-multiscale-snapshot/1"
RUN_SCHEMA = "her2-multiscale-run/1"
REVISION = "multiscale_pilot"
SNAPSHOT = "snapshot.pt"
PROGRESS = "progress.jsonl"
#: What ``runtime.json`` is a capture OF. It is taken before the inputs, the model and the optimizer
#: exist; ``run.json['libraries_imported_at_completion']`` is the second capture, after ``execute``.
RUNTIME_CAPTURE_STAGE = {
    "runtime": ("start of the run: after the environment check and the code identity, before the "
                "inputs, the model and the optimizer were built. A library imported later (the "
                "optimizer's, for one) is NOT in this record"),
    "completion": ("run.json['libraries_imported_at_completion'] lists the libraries imported once "
                   "execute() returned"),
    "late_imports": ("the identity's code.external_versions binds the installed version of every "
                     "third-party package the static import closure names, late imports included")}


@dataclass(frozen=True)
class ArmPlan:
    """What one arm does, resolved from the pilot config and the base curriculum."""

    arm: str
    geometry: str
    full_probability: float
    replacement_probability: float
    distance_weights: dict
    updates: int
    seed: int
    diagnostics: dict

    @classmethod
    def from_pilot(cls, pilot, curriculum):
        return cls(arm=pilot.arm, geometry=pilot.geometry,
                   full_probability=draws_lib.full_probability(curriculum),
                   replacement_probability=pilot.replacement_probability,
                   distance_weights=dict(pilot.distance_weights), updates=pilot.updates,
                   seed=pilot.seed, diagnostics=dict(pilot.diagnostics))

    def document(self):
        return {"arm": self.arm, "mask_geometry": self.geometry,
                "full_probability": self.full_probability,
                "local_replacement_probability": self.replacement_probability,
                "distance_weights": {str(key): value for key, value in
                                     sorted(self.distance_weights.items())},
                "updates": self.updates, "seed": self.seed, "diagnostics": self.diagnostics}


# ---------------------------------------------------------------------------
# the snapshot: evaluation only
# ---------------------------------------------------------------------------

def save_snapshot(path, *, model, heads, identity, step):
    """Write the final model and heads. It cannot resume a run, and says so."""
    state, model_format = model.checkpoint_state()
    payload = {"schema_version": SNAPSHOT_SCHEMA, "evaluation_only": True,
               "resume": ("not supported: no optimizer, scheduler or generator state is stored, "
                          "and this loop has no resume"),
               "step": int(step), "model": state, "model_format": model_format,
               "heads": heads.state_dict(), "identity": dict(identity)}
    with paths.atomic_open(path, "wb") as stream:
        torch.save(payload, stream)
    return paths.sha256_file(path)


def require_state_matches(supplied, expected, label):
    """Refuse a state mapping whose keys, shapes or dtypes differ from ``expected``; copies nothing.

    ``load_state_dict`` copies a same-shape tensor of another dtype into the parameter silently, and
    the native checkpoint loader checks keys and shapes but not dtypes, so the snapshot's dtypes are
    compared here, against the CURRENT model's and heads' own state maps, before either is touched.
    """
    require(isinstance(supplied, dict)
            and all(isinstance(value, torch.Tensor) for value in supplied.values()),
            f"the snapshot's {label} state is not a flat mapping of tensors")
    missing = sorted(set(expected) - set(supplied))
    unexpected = sorted(set(supplied) - set(expected))
    require(not missing and not unexpected,
            f"the snapshot's {label} keys differ from the current ones: missing {missing[:6]}, "
            f"unexpected {unexpected[:6]}")
    for key in sorted(expected):
        got, want = supplied[key], expected[key]
        require(tuple(got.shape) == tuple(want.shape),
                f"the snapshot's {label} {key} has shape {tuple(got.shape)}, the current one "
                f"holds {tuple(want.shape)}")
        require(got.dtype == want.dtype,
                f"the snapshot's {label} {key} has dtype {got.dtype}, the current one holds "
                f"{want.dtype}; a snapshot is never cast into a model")


def load_snapshot(path, *, model, heads, identity):
    """Strictly load a snapshot into a freshly built model and heads, or refuse.

    The schema, the evaluation-only marker and the identity must all agree before a tensor is
    copied. Then BOTH the model's and the heads' supplied keys, shapes and dtypes are compared with
    the current expected state maps (:func:`require_state_matches`) before either is loaded, so a
    bad head dtype cannot leave the model already overwritten. The model's own protocol (base
    identity, tied weights, format) validates before it mutates, and the heads load with
    ``strict=True`` last. Nothing is coerced.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    require(payload.get("schema_version") == SNAPSHOT_SCHEMA,
            f"{path} is a {payload.get('schema_version')!r} file, not {SNAPSHOT_SCHEMA}")
    require(payload.get("evaluation_only") is True, f"{path} is not marked evaluation-only")
    absent = [key for key in ("model", "model_format", "heads", "step", "identity")
              if key not in payload]
    require(not absent, f"{path} lacks {absent}")
    differing = sorted(key for key in set(identity) | set(payload["identity"])
                       if payload["identity"].get(key) != dict(identity).get(key))
    require(not differing, f"{path} was written under another identity (differs in {differing})")
    expected_model, _ = model.checkpoint_state()
    require_state_matches(payload["model"], expected_model, "model")
    require_state_matches(payload["heads"], heads.state_dict(), "heads")
    record = model.load_checkpoint_state(payload["model"], payload["model_format"], strict=True)
    heads.load_state_dict(payload["heads"], strict=True)
    return {"path": str(path), "step": int(payload["step"]), "model": record,
            "evaluation_only": True}


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def realized_training(tally):
    """What the loop has actually done so far, read from its own tally (never from the config).

    ``replacement_*`` count the measured-pair objective that replaced a local draw; the
    ``retained_legacy_local`` block counts the local draws that kept the inherited single/pair
    contrast or quartet path, which no replacement distance describes.
    """
    return {"local_draws": int(tally["local_draws"]),
            "replacement_microbatches": int(tally["local_replaced"]),
            "replacement_endpoint_rows": int(tally["pair_endpoint_rows"]),
            "replacement_distances": {str(key): int(value) for key, value in
                                      sorted(tally["replacement_distances"].items())},
            "retained_legacy_local": {
                "her2_pair_only_not_replaced": int(tally["local_pair_eligible"]
                                                   - tally["local_replaced"]),
                "her2_with_quartets_or_without_contrasts": int(tally["local_with_quartets"]),
                "other_family": int(tally["local_other_family"])}}


def execute(services, route, *, plan, stream, pool, closure, window_of, keys, panel, directory,
            identity, log=None):
    """Train ``route`` for ``plan.updates`` updates on ``stream`` and evaluate before and after.

    ``stream`` is ``services.curriculum_stream(route, seed=..., arm='full', updates=...)`` (or a
    fixture with the same call signature); ``pool`` the :class:`~.pairs.PairPool`; ``closure``,
    ``window_of`` and ``keys`` the training boundary every pair is re-validated against on every
    draw; ``panel`` the development panel rows. Returns the run record.
    """
    require(route == config_lib.ROUTE, f"the pilot trains {config_lib.ROUTE} only, not {route!r}")
    directory = Path(directory)
    shapes = services.selected_shapes()
    accumulation = int(shapes["accumulation"])
    optimization = services.config["optimization"]
    settings = services.task_settings()
    coefficients = controls.arm_coefficients(controls.ARM_FULL,
                                             base=services.config["curriculum"]["weights"])
    # Preflight, before a model exists: a weighted distance the pool lacks is an error here and is
    # never renormalised away.
    local = draws_lib.LocalPairs(
        pool, seed=plan.seed, probability=plan.replacement_probability,
        distance_weights=plan.distance_weights, closure=closure, window_of=window_of,
        training_keys=keys, kappa=settings["kappa"],
        reconstruction_weight=settings["local_reconstruction_weight"])
    model = services.build_route(route, seed=plan.seed)
    heads = services.build_heads(route, model)
    parameters = [p for p in list(model.parameters()) + list(heads.parameters())
                  if p.requires_grad]
    optimizer = services.build_optimizer(route, parameters, lr=services.route_learning_rate(route),
                                         betas=tuple(optimization["betas"]),
                                         weight_decay=float(optimization["weight_decay"]))
    state = services.fresh_state(plan.seed)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, checkpoints.warmup_plateau_lambda(
            state, warmup=int(optimization["warmup"]),
            plateau_factor=float(optimization["plateau_factor"])))
    accumulator = HeadAccumulator()
    tally = {"base_draws": 0, "skipped_base_draws": 0, "whole_window_draws": 0,
             "mask_modes": {}, "mask_counts": {}, "local_draws": 0, "local_with_quartets": 0,
             "local_other_family": 0, "local_pair_eligible": 0, "local_replaced": 0,
             "replacement_distances": {}, "pair_endpoint_rows": 0}
    configured = plan.distance_weights if plan.replacement_probability > 0.0 else {}

    def evaluate():
        # What the arm is CONFIGURED to draw and what it has REALIZED so far are reported apart: the
        # first evaluation runs before any update, so its realized counts are empty by construction.
        return evaluation_lib.evaluate(
            services, route, model, heads, panel=panel, pool=pool, closure=closure,
            window_of=window_of, keys=keys, kappa=settings["kappa"], seed=plan.seed,
            diagnostics=plan.diagnostics, configured_distances=configured,
            realized=realized_training(tally))

    before = evaluate()
    started = time.time()
    progress = directory / PROGRESS
    model.train()
    heads.train()
    with progress.open("a", encoding="utf-8", newline="\n") as journal:
        while state.step < plan.updates:
            state.step += 1
            step = state.step
            optimizer.zero_grad(set_to_none=True)
            total, terms, drawn, counted = 0.0, {}, [], False
            for micro in range(accumulation):
                draw = stream(step, micro, state.sampler_rng)
                state.sampler_position += 1
                tally["base_draws"] += 1
                if draw is None or draw.rows() == 0:
                    tally["skipped_base_draws"] += 1
                    drawn.append({"micro": micro, "skipped": True})
                    continue
                entry = {"micro": micro, "task": draw.task, "family": draw.family,
                         "skipped": False}
                if draw.task == data_lib.TASK_WHOLE:
                    draw, mask_documents = draws_lib.replace_masks(
                        draw, seed=plan.seed, step=step, micro=micro, geometry=plan.geometry,
                        probability=plan.full_probability)
                    summary = draws_lib.summarise_masks(mask_documents)
                    entry["masks"] = summary
                    tally["whole_window_draws"] += 1
                    for name, number in summary["modes"].items():
                        tally["mask_modes"][name] = tally["mask_modes"].get(name, 0) + number
                    for name, number in summary["counts"].items():
                        tally["mask_counts"][name] = tally["mask_counts"].get(name, 0) + number
                if draw.task == data_lib.TASK_LOCAL:
                    tally["local_draws"] += 1
                    if draw.family != data_lib.FAMILY_HER2:
                        tally["local_other_family"] += 1
                    elif not local.eligible(draw):
                        tally["local_with_quartets"] += 1
                    else:
                        tally["local_pair_eligible"] += 1
                if local.wants(draw, step=step, micro=micro):
                    outcome, replacement = local.outcome(services, route, model, draw,
                                                         step=step, micro=micro)
                    outcome.scheduled_task = draw.task
                    entry["replacement"] = replacement
                    tally["local_replaced"] += 1
                    distance = str(replacement["distance"])
                    tally["replacement_distances"][distance] = (
                        tally["replacement_distances"].get(distance, 0) + 1)
                    tally["pair_endpoint_rows"] += int(outcome.rows)
                else:
                    outcome = tasks_lib.dispatch(
                        services, route, model, heads, draw, arm=controls.ARM_FULL,
                        settings=settings, rng=state.mask_rng, accumulator=accumulator)
                    if draw.task == data_lib.TASK_LOCAL and draw.contrasts:
                        entry["kept_base_contrasts"] = {
                            kind: sum(1 for c in draw.contrasts if c.kind == kind)
                            for kind in data_lib.CONTRAST_KINDS}
                require(bool(torch.isfinite(outcome.loss).all()),
                        f"the {draw.task} loss is not finite at update {step}, microbatch {micro}")
                weight = float(coefficients.get(outcome.term, 0.0))
                if weight > 0.0:
                    scaled = weight * outcome.loss / accumulation
                    scaled.backward()
                    total += float(scaled.detach())
                for name, value in outcome.terms.items():
                    terms[name] = terms.get(name, 0.0) + float(value)
                state.ledger.record_outcome(task=draw.task, outcome=outcome, update=not counted)
                counted = True
                entry.update(term=outcome.term, coefficient=weight, rows=int(outcome.rows),
                             forwards=int(outcome.forwards), targets=int(outcome.targets),
                             source=outcome.source,
                             contributions=[item.document() for item in outcome.exposure()])
                drawn.append(entry)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(optimization["gradient_clip"]), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            record = {"step": step, "loss": total, "terms": terms,
                      "gradient_norm": float(norm),
                      "learning_rate": float(scheduler.get_last_lr()[0]), "microbatches": drawn}
            journal.write(json.dumps(inputs_lib.plain(record), sort_keys=True, allow_nan=False)
                          + "\n")
            journal.flush()
            if log is not None and (step == 1 or step % 25 == 0 or step == plan.updates):
                log(f"update {step}/{plan.updates} loss {total:.5f} norm {float(norm):.3f}")
    state.ledger.require_additive()
    after = evaluate()
    digest = save_snapshot(directory / SNAPSHOT, model=model, heads=heads, identity=identity,
                           step=state.step)
    evaluation = {"schema_version": RUN_SCHEMA, "arm": plan.arm, "before": before, "after": after}
    paths.write_json(directory / "evaluation.json", inputs_lib.plain(evaluation))
    return inputs_lib.plain({
        "schema_version": RUN_SCHEMA, "status": "completed", "arm": plan.arm,
        "plan": plan.document(), "updates": int(state.step), "shapes": dict(shapes),
        "task_settings": dict(settings), "coefficients": coefficients,
        "optimizer": type(optimizer).__name__, "tally": tally,
        "realized_training": realized_training(tally),
        "exposure": state.ledger.document(), "head_accumulator": accumulator.document(),
        "pair_pool": pool.document(),
        "snapshot": {"path": str(directory / SNAPSHOT), "sha256": digest,
                     "evaluation_only": True, "step": int(state.step)},
        "identity": dict(identity), "wall_seconds": round(time.time() - started, 3),
        "claim": ("a within-pilot comparison arm: no biological-improvement claim, no resume, no "
                  "policy or guided sampler, no changed assay semantics")})


# ---------------------------------------------------------------------------
# the whole pilot
# ---------------------------------------------------------------------------

#: The static closure is seeded from the command line, so it names every module the pilot can
#: execute: the pilot package and, through it, the inherited services, tasks, sources, native model
#: and optimizer (including imports written inside functions).
CLOSURE_ENTRY = "scripts/her2_multiscale_pilot.py"
PACKAGE_PREFIX = "src/smallAntibodyGen/experiments/her2_multiscale/"


def code_identity(code_root):
    """SHA-256 of EVERY file in the static import closure of the entry script, from ``code_root``.

    ``her2_replay_spec.source_closure`` walks the syntax (function-level imports included) from
    :data:`CLOSURE_ENTRY`; each file it names is hashed from the code root given -- the worktree the
    run executes, never the data repository. A missing closure file, an empty closure or one that
    lacks the entry script or the pilot package is an error, not an empty identity. The scope is
    static: a module reached through ``importlib`` or a plugin is not in it, and third-party
    packages are recorded by version only (``external_versions``), not by bytes.
    """
    root = Path(code_root)
    closure = replay_spec.source_closure(root, [CLOSURE_ENTRY])
    names = sorted(closure["files"])
    require(CLOSURE_ENTRY in names and any(name.startswith(PACKAGE_PREFIX) for name in names),
            f"the source closure under {root} does not contain the pilot ({len(names)} file(s)); "
            "an empty or partial code identity is never accepted")
    files = {name: paths.sha256_file(root / name) for name in names}
    return {"scope": ("static import closure of the entry script, hashed from the code root; "
                      "modules reached by a path the syntax does not reveal are not in it, and "
                      "third-party packages are recorded by version, not by bytes"),
            "entry_points": list(closure["entry_points"]), "file_count": len(files),
            "files": files,
            "closure_sha256": paths.sha256_text("\n".join(f"{name} {files[name]}"
                                                          for name in sorted(files))),
            "external": list(closure["external"]),
            "external_versions": closure["external_versions"]}


def runtime_provenance(external_versions):
    """Interpreter, platform and the third-party libraries imported SO FAR, with locations.

    Read from ``sys.modules`` for the packages the closure names, at the moment of the call: nothing
    is scanned, no model is loaded and no CUDA call is made. ``run_pilot`` calls it twice, at the
    start (``runtime.json``) and after ``execute`` (``libraries_imported_at_completion``), and a
    library imported in between appears only in the second. ``version`` is the installed
    distribution's, ``module_version`` the imported module's own ``__version__`` (they can differ),
    ``location`` where it was imported from. This is a record of what was imported, not a byte pin
    of the installed dependencies.
    """
    libraries = {}
    for name, info in sorted(external_versions.items()):
        module = sys.modules.get(name)
        if module is None or info.get("kind") == "standard_library":
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            origin = next(iter(getattr(module, "__path__", None) or ()), None)
        libraries[name] = {
            "version": info.get("version"),
            "module_version": None if getattr(module, "__version__", None) is None
            else str(module.__version__),
            "location": None if origin is None else str(Path(origin).resolve().parent)}
    return {"python": {"version": sys.version, "implementation": platform.python_implementation(),
                       "executable": sys.executable, "prefix": sys.prefix},
            "platform": platform.platform(), "torch_cuda_build": torch.version.cuda,
            "libraries": libraries}


def runtime_identity(provenance):
    """The stable half of the runtime: versions only (no paths), so identities compare across moves.

    It is taken from the start-of-run capture; the versions of libraries imported later are bound by
    ``code.external_versions`` in the same identity.
    """
    python = ".".join(str(part) for part in sys.version_info[:3])
    return {"python": python, "implementation": provenance["python"]["implementation"],
            "libraries": {name: info["version"] or info["module_version"]
                          for name, info in sorted(provenance["libraries"].items())}}


def run_identity(pilot, loaded, services, route, pool, code, runtime):
    return {"schema": RUN_SCHEMA, "pilot_sha256": pilot.sha256,
            "base_config_sha256": pilot.base_sha256, "arm": pilot.arm, "seed": int(pilot.seed),
            "updates": int(pilot.updates), "route": route,
            "native_sha256": services.native_identity(route),
            "split_sha256": loaded.membership.split_sha256,
            "sequence_sha256": loaded.membership.sequence_sha256,
            "vocabulary_file_sha256": loaded.frozen["vocabulary"]["file_sha256"],
            "inventory_file_sha256": loaded.frozen["inventory"]["file_sha256"],
            "bin_policy": loaded.bin_identity,
            "pair_structural_digest": pool.structural_digest,
            "pair_supervision_digest": pool.supervision_digest,
            "training_population_digest": pool.population_digest,
            "code": dict(code), "runtime": dict(runtime)}


def _write_failure(directory, error):
    record = {"schema_version": RUN_SCHEMA, "status": "failed",
              "error_type": type(error).__name__, "message": str(error),
              "traceback": traceback.format_exc().splitlines()[-40:]}
    try:
        paths.write_json(Path(directory) / "failure.json", record)
    except Exception:                      # noqa: BLE001 - the original error is the one to raise
        pass


def run_pilot(pilot, pilot_roots, *, device, environment, log=None):
    """Run one pilot arm end to end over a reserved run root. Returns the completion record.

    ``environment`` is the record ``prepare_environment`` returned (the command line passes it on
    rather than discarding it); it must show the cache variables were assigned before the scientific
    imports, and it is persisted in ``runtime.json`` beside the interpreter and library provenance
    captured at that point (before the inputs, model and optimizer; see
    :data:`RUNTIME_CAPTURE_STAGE`). The code identity is computed first, so an incomplete source
    closure stops the run before any input is opened.

    Any error -- a refused input, a non-finite loss, a failed write -- is recorded in
    ``failure.json`` and re-raised; nothing is caught into a success.
    """
    directory = pilot_roots.run_root
    try:
        require(isinstance(environment, dict) and environment,
                "run_pilot needs the record prepare_environment returned")
        env_lib.require_environment_ordering(environment, label="multiscale pilot")
        code = code_identity(pilot_roots.code_root)
        provenance = runtime_provenance(code["external_versions"])
        paths.write_json(directory / "runtime.json", inputs_lib.plain({
            "schema_version": RUN_SCHEMA, "environment": environment, "runtime": provenance,
            "capture_stage": RUNTIME_CAPTURE_STAGE,
            "code": {"file_count": code["file_count"], "closure_sha256": code["closure_sha256"],
                     "scope": code["scope"]},
            "verification": ("the code closure is hashed from the code root; interpreter and "
                             "library versions and locations are RECORDED, not byte-verified. The "
                             "dependency acquisition manifest is a pip install report with two "
                             "wheel archive hashes, not a pin of the installed tree"),
            "caches": "every cache and temporary lives under this run root (runtime/)"}))
        base = spec.load_config(pilot.base_path)
        policy = spec.training_phase(base)
        config = inputs_lib.narrowed_config(base)
        context = roots_lib.PilotContext(config=config, config_path=pilot.base_path,
                                         roots=pilot_roots, revision=REVISION,
                                         campaign_id=pilot.pilot_id)
        paths.write_json(directory / "pilot_config.json",
                         inputs_lib.plain(dict(pilot.document_record(), roots=pilot_roots.document())))
        loaded = inputs_lib.load_inputs(pilot, context, policy=policy,
                                        frozen_dir=pilot_roots.frozen_inputs, device=device)
        # The bins every arm uses, as numbers, before any model exists (execute builds the model).
        inputs_lib.write_current_bins(loaded, directory)
        services = loaded.services
        pool = pairs_lib.build_pair_pool(
            loaded.pool, closure=loaded.closure, window_of=loaded.window_of,
            training_population_keys=loaded.training_keys, distances=config_lib.DISTANCES,
            capacity=pilot.reservoir_capacity, seed=pilot.seed)
        plan = ArmPlan.from_pilot(pilot, config["curriculum"])
        identity = run_identity(pilot, loaded, services, config_lib.ROUTE, pool, code,
                                runtime_identity(provenance))
        paths.write_json(directory / "inputs.json", inputs_lib.plain(
            dict(loaded.document(), pair_pool=pool.document(), identity=identity)))
        stream = services.curriculum_stream(config_lib.ROUTE, seed=pilot.seed, arm=controls.ARM_FULL,
                                            updates=pilot.updates)
        record = execute(services, config_lib.ROUTE, plan=plan, stream=stream, pool=pool,
                         closure=loaded.closure, window_of=loaded.window_of,
                         keys=loaded.training_keys, panel=loaded.panel, directory=directory,
                         identity=identity, log=log)
        # The second capture (the first is runtime.json, taken before the model and optimizer).
        record["libraries_imported_at_completion"] = inputs_lib.plain(
            runtime_provenance(code["external_versions"])["libraries"])
        paths.write_json(directory / "run.json", record)
        return record
    except BaseException as error:         # noqa: BLE001 - recorded, then re-raised
        _write_failure(directory, error)
        raise
