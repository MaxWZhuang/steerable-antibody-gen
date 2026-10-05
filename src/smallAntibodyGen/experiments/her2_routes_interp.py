"""Activation shards, token maps, and the causal-intervention experiments.

Every shard carries the source identity, the revision, the checkpoint digest, the
``token_map`` binding each tensor index to ``(chain, residue index, serialisation
role, edit-map slot)``, the condition vector, the ablation-control label **and the layout
and position axis of every tensor**. Written through ``her2_support_paths.write_shard``,
so per-array content digests and a completion record come for free and a partial write
cannot be read back as complete.

The interventions, and exactly what each one licenses you to say:

* **zero ablation** -- set the activation to zero. Removes the feature and the
  position's contribution together, so on its own it over-attributes.
* **mean ablation** -- replace it with an explicit **frozen donor mean**
  (:class:`~smallAntibodyGen.models.her2_routes_hooks.ReferenceMean`): one reference value
  per position, built once from a named donor set and applied identically to every
  evaluation batch. It is never the mean of the batch being evaluated.
* **cross-condition patch** -- run with condition A, capture, then run with condition B
  while patching A's activation in.

What none of them is: a proof that a *pathway* was isolated. Each reports the effect of
replacing one site's activation with one named value. A patch that moves the output and an
ablation that does not is *consistent with* the site carrying the condition's information;
it does not exclude a parallel route through other sites, a residual-stream bypass, or an
effect of feeding the model an activation it never produces. :func:`cross_condition_experiment`
therefore reports the three effects side by side and claims no more than that.

Every helper that runs a forward does so under :func:`controlled_evaluation`: eval mode,
no gradients, every module's training flag restored afterwards, and the caller's RNG state
untouched. CUDA is never initialised on a CPU run.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from ..models.her2_routes_hooks import (
    LAYOUTS,
    ReferenceMean,
    ReferenceMeanBuilder,
    require_token_map,
)
from . import her2_support_paths as paths
from .her2_runtime import require

INTERP_SCHEMA = "her2-routes-interp/1"


# ---------------------------------------------------------------------------
# controlled evaluation
# ---------------------------------------------------------------------------

def _initialised_cuda_devices():
    """The CUDA devices whose RNG may be saved: none unless CUDA is already initialised.

    ``torch.cuda.get_rng_state`` lazily initialises the CUDA runtime, which creates a context
    and, on a CPU run, either fails or allocates memory on a card the run never used. If
    CUDA is not already up, nothing the caller is doing can have advanced a CUDA generator.
    """
    if not torch.cuda.is_initialized():
        return []
    return list(range(torch.cuda.device_count()))


@contextlib.contextmanager
def controlled_evaluation(model=None, *, seed=None):
    """Eval mode, ``no_grad``, and no side effect on the caller's RNG or training flags.

    * Each submodule's ``training`` flag is saved and restored individually, not only the
      top-level one, so a module deliberately left in another mode comes back unchanged.
    * The CPU generator and every *initialised* CUDA generator are saved and restored.
      ``seed`` (optional) seeds inside the fork, so a stochastic forward is reproducible
      without disturbing the surrounding stream.
    * CUDA is not initialised by this function: on a CPU run no CUDA state is touched.
    """
    flags = [] if model is None else [(module, module.training) for module in model.modules()]
    devices = _initialised_cuda_devices()
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            # ``torch.manual_seed`` would also queue a CUDA seeding call that fires when CUDA
            # is first initialised later; seed the CPU generator directly and touch CUDA only
            # if it is already up.
            torch.default_generator.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed_all(int(seed))
        if model is not None:
            model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            for module, flag in flags:
                module.training = flag


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------

def shard_document(*, route, revision, checkpoint_digest, source_digest, condition, token_map,
                   control_label, arm, stage, extra=None):
    """The provenance block every activation shard carries."""
    document = {"interp_schema": INTERP_SCHEMA, "route": str(route), "revision": str(revision),
                "checkpoint_sha256": str(checkpoint_digest),
                "source_snapshot_sha256": str(source_digest),
                "condition": dict(condition), "control_label": str(control_label),
                "arm": str(arm), "stage": str(stage),
                "token_map": [dict(row) for row in token_map],
                "token_map_rule": ("each tensor index is bound to (chain, residue index, "
                                   "serialisation role, edit-map slot). Without it an activation "
                                   "row cannot be attributed to a residue.")}
    if extra:
        document.update(dict(extra))
    return document


def _numpy_float32(tensor):
    """A float32 CPU array from a torch tensor (any device, any dtype) or an array-like."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().to("cpu", dtype=torch.float32).numpy()
    return np.asarray(tensor, dtype=np.float32)


def _resolve_export(key, value, layouts):
    """``(tensor, layout)`` for one activation. A missing or unknown layout raises."""
    if isinstance(value, dict) and "tensor" in value:
        layout = value.get("layout")
        tensor = value["tensor"]
        if "position_axis" in value:
            require(LAYOUTS.get(layout) == value["position_axis"],
                    f"{key}: the record says position_axis {value['position_axis']} but layout "
                    f"{layout!r} puts position on axis {LAYOUTS.get(layout)}")
    else:
        tensor = value
        layout = None if layouts is None else dict(layouts).get(key)
    require(layout is not None,
            f"{key}: no layout was supplied. A bare tensor is not written: pass "
            "registry.export(name) or a layouts mapping. Guessing tokens_first would analyse a "
            "channel as a residue on a convolutional tap.")
    require(layout in LAYOUTS,
            f"{key}: unknown layout {layout!r}; expected one of {sorted(LAYOUTS)}")
    return tensor, layout


def write_activations(directory, name, activations, document, *, run_root, logical_prefix,
                      token_map, layouts=None, order="capture order, one row per batch row"):
    """Write one activation shard after checking the token map against every tensor.

    Each activation is a :meth:`HookRegistry.export` record (or a tensor named in
    ``layouts``). The token map is checked against the length of the tensor's **declared
    position axis**, not against ``shape[1]``: on a ``tokens_last`` (convolutional) tensor
    ``shape[1]`` is the channel count, and validating the map against it would accept a
    channel axis as a residue axis. Tensors are detached, moved to CPU and stored as float32 in
    the layout they were captured in; the per-tensor layout, position axis and shape are
    written into the shard record so a reader can attribute every row.
    """
    arrays, described = {}, {}
    for key, value in dict(activations).items():
        tensor, layout = _resolve_export(key, value, layouts)
        values = _numpy_float32(tensor)
        axis = LAYOUTS[layout]
        require(values.ndim > axis,
                f"{key}: a {layout} activation needs more than {axis} dimensions to carry "
                f"positions on axis {axis}; got shape {values.shape}")
        require_token_map(token_map, values.shape[axis], label=f"{key} token map")
        arrays[key] = values
        described[key] = {"layout": layout, "position_axis": axis,
                          "shape": [int(size) for size in values.shape],
                          "stored_dtype": "float32"}
    require(arrays, "No activations to write")
    record = dict(document)
    record["activation_layouts"] = described
    record["layout_rule"] = ("each array is stored in the layout it was captured in; "
                             "activation_layouts names that layout and the axis that carries "
                             "sequence position, and token_map is indexed along that axis.")
    return paths.write_shard(directory, name, arrays, record, order=order,
                             logical_prefix=logical_prefix, run_root=run_root)


def read_activations(directory, name):
    """``({key: {tensor, layout, position_axis, shape}}, record)`` from a shard, validated.

    Re-checks what the writer promised: the recorded layout is known, the recorded shape is
    the array's shape, and the token map covers the declared position axis. A shard with no
    recorded layouts is refused, not read with a guessed one.
    """
    arrays, record = paths.read_shard(directory, name)
    described = dict(record.get("activation_layouts") or {})
    require(sorted(described) == sorted(arrays),
            f"{name}: the shard describes layouts for {sorted(described)} but holds "
            f"{sorted(arrays)}. A shard written before layouts were recorded cannot be read "
            "with a guessed one.")
    out = {}
    for key, values in sorted(arrays.items()):
        entry = described[key]
        require(entry.get("layout") in LAYOUTS, f"{name}/{key}: unknown layout {entry.get('layout')!r}")
        axis = LAYOUTS[entry["layout"]]
        require(entry.get("position_axis") == axis,
                f"{name}/{key}: recorded position_axis {entry.get('position_axis')} contradicts "
                f"layout {entry['layout']!r}")
        require(list(values.shape) == list(entry.get("shape") or []),
                f"{name}/{key}: shape {list(values.shape)} differs from the recorded "
                f"{entry.get('shape')}")
        require_token_map(record.get("token_map") or [], values.shape[axis],
                          label=f"{name}/{key} token map")
        out[key] = {"tensor": values, "layout": entry["layout"], "position_axis": axis,
                    "shape": list(values.shape)}
    return out, record


# ---------------------------------------------------------------------------
# capture and intervention
# ---------------------------------------------------------------------------

def capture(model, registry, *, batch, names=None, forward):
    """Run one forward with taps enabled and return the captured tensors with their layouts.

    ``activations`` is kept for callers that want the bare tensors; ``exports`` carries each
    one with its layout and position axis, which is what a patch or a shard needs. Both are
    caller-owned copies: ``dict(registry.captured)`` would hand out the registry's own tensors,
    so an in-place edit of a returned activation would rewrite what the registry holds and what
    every later export of it reports.
    """
    registry.clear().clear_plans()
    registry.enable(names)
    try:
        with controlled_evaluation(model):
            output = forward(model, batch)
    finally:
        registry.disable()
    return {"output": output,
            "activations": {key: value.detach().clone()
                            for key, value in registry.captured.items()},
            "exports": registry.export_all(), "names": sorted(registry.captured)}


def intervene(model, registry, *, batch, name, kind, forward, positions=None, source=None,
              reference=None):
    """Run one forward with a single intervention active at ``name``.

    ``source`` (patch kinds) carries its layout; ``reference`` (``mean_ablate``) is the frozen
    donor mean. Both are validated by :meth:`HookRegistry.plan` before any forward runs.
    """
    registry.clear().clear_plans()
    registry.plan(name, kind=kind, positions=positions, source=source, reference=reference)
    registry.enable([name])
    try:
        with controlled_evaluation(model):
            output = forward(model, batch)
    finally:
        registry.disable()
        registry.clear_plans()
    return {"output": output, "name": name, "kind": kind,
            "positions": None if positions is None else [int(p) for p in positions],
            "reference_sha256": None if reference is None else reference.sha256}


def build_reference_mean(model, registry, *, donor_batches, name, forward, ids_of=None,
                         mask_of=None, evaluation_ids=None, provenance=None):
    """Capture ``name`` on every donor batch and freeze the per-position mean.

    ``ids_of(batch)`` returns the donor row identities (required whenever
    ``evaluation_ids`` is given: a reference whose donors are unnamed cannot be checked
    against the rows it will be applied to). ``mask_of(batch)`` returns the
    ``(rows, positions)`` real-token mask so padding is excluded from the mean.
    ``evaluation_ids`` are refused as donors. The donor set is frozen into the result's
    provenance. Nothing is left registered or enabled on the registry.

    ``provenance`` describes where the donors came from. It may not set ``donor_batches``: that
    key is the number of donor batches actually captured here, and a caller-supplied value
    would replace it in the stored record while :meth:`ReferenceMean.verify` still passed.
    """
    layout = registry.layout(name)
    builder = ReferenceMeanBuilder(name=name, layout=layout)
    batches = list(donor_batches)
    require(batches, "A reference mean needs at least one donor batch")
    supplied = dict(provenance or {})
    require("donor_batches" not in supplied,
            f"{name}: provenance may not set 'donor_batches'. It is written from the "
            f"{len(batches)} donor batch(es) actually captured; a caller-supplied value would "
            "override the record.")
    for batch in batches:
        grabbed = capture(model, registry, batch=batch, names=[name], forward=forward)
        builder.add(grabbed["exports"][name]["tensor"],
                    mask=None if mask_of is None else mask_of(batch),
                    donor_ids=None if ids_of is None else ids_of(batch))
    # The actual count is written LAST, so even a future change to the check above could not let
    # a caller's note shadow it.
    return builder.build(evaluation_ids=evaluation_ids,
                         provenance={**supplied, "donor_batches": len(batches)})


def intervention_effect(baseline, intervened, *, metric):
    """The measured change a single intervention produced, with its control alongside."""
    left, right = float(metric(baseline)), float(metric(intervened))
    return {"baseline": left, "intervened": right, "difference": right - left,
            "relative": None if left == 0 else (right - left) / abs(left)}


def cross_condition_experiment(model, registry, *, batch_a, batch_b, name, forward, metric,
                               mean_reference, positions=None, evaluation_ids=None):
    """Patch condition A's activation into a condition-B forward, beside two ablations.

    Reports three effects at the same site: the patch of A's activation, a zero ablation, and
    a replacement by ``mean_reference`` -- the explicit frozen donor mean, which has no
    default. Missing it is an error, and so is a reference built for another tap or layout.
    ``evaluation_ids``, when supplied, must be disjoint from the reference's donors.

    The three numbers answer three narrow questions: what happens when this site's
    activation is replaced by condition A's, by zero, and by the donor-population mean.
    They do **not** isolate a pathway: another route through the network, or the fact that a
    patched or zeroed activation is off the model's data distribution, can produce or hide
    any of them.
    """
    require(isinstance(mean_reference, ReferenceMean),
            "cross_condition_experiment needs mean_reference, an explicit frozen ReferenceMean "
            "(see build_reference_mean). There is no fallback to a batch mean.")
    require(mean_reference.name == name,
            f"mean_reference was built for {mean_reference.name!r}, not {name!r}")
    if evaluation_ids is not None:
        require(mean_reference.donor_ids,
                "evaluation_ids were supplied but the reference mean records no donor ids, so "
                "disjointness cannot be verified")
        overlap = sorted(set(mean_reference.donor_ids) & {str(item) for item in evaluation_ids})
        require(not overlap,
                f"{len(overlap)} evaluated row(s) are also donors of the reference mean, e.g. "
                f"{overlap[:3]}. A held-out row must not contribute to its own control.")
    captured = capture(model, registry, batch=batch_a, names=[name], forward=forward)
    source = captured["exports"][name]
    with controlled_evaluation(model):
        baseline = forward(model, batch_b)
    patched = intervene(model, registry, batch=batch_b, name=name,
                        kind="cross_condition_patch", forward=forward, source=source,
                        positions=positions)
    zeroed = intervene(model, registry, batch=batch_b, name=name, kind="zero_ablate",
                       forward=forward, positions=positions)
    meaned = intervene(model, registry, batch=batch_b, name=name, kind="mean_ablate",
                       forward=forward, positions=positions, reference=mean_reference)
    return {"name": name,
            "patch": intervention_effect(baseline, patched["output"], metric=metric),
            "zero_ablate": intervention_effect(baseline, zeroed["output"], metric=metric),
            "mean_ablate": intervention_effect(baseline, meaned["output"], metric=metric),
            "mean_reference": mean_reference.document(),
            "positions": None if positions is None else [int(p) for p in positions],
            "reading": ("three effects of replacing one site's activation: with condition A's, "
                        "with zero, and with the frozen donor mean. A patch effect beside a "
                        "small ablation effect is CONSISTENT with the site carrying condition "
                        "information; it does not isolate a pathway, because parallel sites, a "
                        "residual bypass and off-distribution activations are not excluded.")}


def checkpoint_state(*, model, optimizer=None, scheduler=None, sampler_state=None,
                     mask_rng=None, energy_buffer=None, exposure=None, revision=None,
                     source_digest=None, external_digests=None, step=0):
    """Everything a resumed run must restore. An omission here is a silent divergence.

    The mask RNG and the sampler state are in the list because a resume that
    restored the weights and re-seeded the masking would produce a different
    trajectory from the uninterrupted run while passing every weight check.
    """
    state = {"interp_schema": INTERP_SCHEMA, "step": int(step), "revision": revision,
             "source_snapshot_sha256": source_digest,
             "external_source_sha256": dict(external_digests or {}),
             "model": model.state_dict(),
             "optimizer": None if optimizer is None else optimizer.state_dict(),
             "scheduler": None if scheduler is None else scheduler.state_dict(),
             "sampler_state": sampler_state, "mask_rng": mask_rng,
             "energy_buffer": energy_buffer,
             "exposure": dict(exposure or {}),
             "restores": ["model", "optimizer", "scheduler", "sampler_state", "mask_rng",
                          "energy_buffer", "exposure"],
             "torch_rng": torch.get_rng_state(),
             # initialized-only: asking an uninitialized CUDA runtime for its RNG state creates the context
             "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)}
    return state


def milestone_policy(*, best=True, last=True, milestones=(), disk_budget_bytes=None,
                     bytes_per_checkpoint=None):
    """Which checkpoints are kept, and the measured disk cost of keeping them.

    Best and last plus declared sparse milestones. Hundreds of full optimizer copies
    is how an artifact volume fills at trajectory 30 and loses the trajectory, so the
    forecast is computed here rather than discovered.
    """
    count = int(bool(best)) + int(bool(last)) + len(tuple(milestones))
    record = {"keep_best": bool(best), "keep_last": bool(last),
              "milestones": [int(value) for value in milestones],
              "checkpoints_retained": count}
    if bytes_per_checkpoint is not None:
        total = int(bytes_per_checkpoint) * count
        record.update(bytes_per_checkpoint=int(bytes_per_checkpoint), total_bytes=total,
                      total_gib=total / 1024 ** 3)
        if disk_budget_bytes is not None:
            record.update(disk_budget_gib=int(disk_budget_bytes) / 1024 ** 3,
                          fits=total <= int(disk_budget_bytes))
            require(total <= int(disk_budget_bytes),
                    f"The retention policy needs {total / 1024 ** 3:.1f} GiB and the declared "
                    f"budget is {int(disk_budget_bytes) / 1024 ** 3:.1f} GiB. Reduce the "
                    "milestones deliberately rather than discovering it at the last trajectory.")
    return record
