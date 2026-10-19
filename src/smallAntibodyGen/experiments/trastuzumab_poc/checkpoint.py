"""Strict update-boundary checkpoints, distinct from standalone stage parents.

Schema 2 carries the functional arm explicitly: ``identity.functional``,
``progress.functional`` and ``payload.functional`` are each ``null`` for a run without it
(never absent).  There is no migration of schema 1 checkpoints; retrain instead.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    canonical_bytes, file_digest, require_keys,
)
from smallAntibodyGen.experiments.trastuzumab_poc.functional_checkpoint import (
    SEEN_KEYS, validate_functional_training,
)
from smallAntibodyGen.experiments.trastuzumab_poc.post_budget import validate_post_budget_identity
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import RNG_ALGORITHM
from smallAntibodyGen.experiments.trastuzumab_poc.runner import TrainingConfig, learning_rate
from smallAntibodyGen.experiments.trastuzumab_poc.transfer_guard import TRAINING_SCHEMA

CHECKPOINT_SCHEMA = "trastuzumab-checkpoint/2"


def atomic_save(payload, path):
    """Replace one latest checkpoint after a complete fsynced save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(trainer, path):
    if (trainer.failed or any(p.grad is not None for _, p in trainer.parameters)
            or (trainer.head is not None and trainer.head.weight.grad is not None)):
        raise ValueError("checkpoint requires a successful optimizer boundary")
    # Synchronous save: the caller must not train concurrently. Detached references
    # avoid duplicating an entire FP32 650M backbone in host memory.
    payload = {
        "schema": CHECKPOINT_SCHEMA, "identity": copy.deepcopy(trainer.identity),
        "weights": {name: p.detach() for name, p in trainer.parameters},
        "optimizer": trainer.optimizer.state_dict(), "progress": trainer.progress(),
        "rng": {"algorithm": RNG_ALGORITHM, "seed": trainer.config.seed,
                "next_update": trainer.next_update},
        "functional": None if trainer.head is None else {
            "weights": {"weight": trainer.head.weight.detach()},
            "optimizer": trainer.head_optimizer.state_dict()},
    }
    _validate(payload, trainer.policy, expected_identity=trainer.identity, head=trainer.head,
              restricted=None if trainer.functional is None else trainer.functional.data)
    atomic_save(payload, path)
    return {"path": str(Path(path).resolve()), "sha256": file_digest(path),
            "next_update": trainer.next_update}


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _tensor(value, target, name, *, nonnegative=False):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.shape != target.shape or value.dtype != target.dtype
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"invalid tensor: {name}")
    if nonnegative and bool((value < 0).any()):
        raise ValueError(f"negative second moment: {name}")


def _adamw(actual, parameters, *, lr, weight_decay, group_lr, step, label):
    """Validate one AdamW state against the recipe; returns a staged optimizer over ``parameters``.

    Exactly one group with the recorded options and parameter order, ``group_lr`` as its
    learning rate, and either an empty state (``step`` zero) or one entry per parameter, keyed by
    the exact integers ``0..n-1``, with a strided FP32 scalar step count equal to ``step`` and
    finite, shape- and dtype-exact moments (the second moment nonnegative).  The policy and the
    head share this one definition.
    """
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=lr, weight_decay=weight_decay,
                                  foreach=False, fused=False, capturable=False)
    expected = optimizer.state_dict()
    require_keys(actual, ("state", "param_groups"), f"{label} optimizer")
    if not isinstance(actual["param_groups"], list) or len(actual["param_groups"]) != 1:
        raise ValueError(f"{label} optimizer must contain exactly one group")
    group = actual["param_groups"][0]
    template = expected["param_groups"][0]
    template["lr"] = group_lr
    if canonical_bytes(group) != canonical_bytes(template):
        raise ValueError(f"{label} optimizer parameter order or options mismatch")
    state = actual["state"]
    # Indices are exactly ``int``: False and 0.0 hash and compare equal to 0, so a set
    # comparison alone would accept them as the first parameter.
    if (not isinstance(state, dict) or any(type(key) is not int for key in state)
            or set(state) != (set(range(len(parameters))) if step else set())):
        raise ValueError(f"{label} optimizer state inventory mismatch")
    for index, (_, parameter) in enumerate(parameters):
        if not step:
            break
        entry = state[index]
        require_keys(entry, ("step", "exp_avg", "exp_avg_sq"), "AdamW state")
        count = entry["step"]
        # The layout is checked before any backend operation: isfinite is not implemented for
        # every layout (a sparse scalar raises NotImplementedError, not a validation error).
        if (not isinstance(count, torch.Tensor) or count.layout != torch.strided
                or count.shape != torch.Size([]) or count.dtype != torch.float32
                or not bool(torch.isfinite(count)) or float(count) != step):
            raise ValueError("invalid AdamW step")
        _tensor(entry["exp_avg"], parameter, "exp_avg")
        _tensor(entry["exp_avg_sq"], parameter, "exp_avg_sq", nonnegative=True)
    return optimizer


def _validate(payload, policy, *, expected_identity=None, head=None, restricted=None):
    """Validate everything and return the staged ``(policy optimizer, head optimizer or None)``.

    Nothing live is touched.  ``head`` (the live head, resume and save) supplies the exact
    parameter the staged optimizer is built over; without it a CPU template of the recorded
    shape is used, so inference validation needs no trainer.  ``restricted`` (resume and save)
    upgrades the data-free functional replay to a label replay.
    """
    require_keys(payload, ("schema", "identity", "weights", "optimizer", "progress", "rng", "functional"),
                 "checkpoint")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    identity = payload["identity"]
    require_keys(identity, ("schema", "config", "model", "encoder_runtime", "data", "input", "source",
                            "initialization", "rng", "runtime", "optimizer", "selection",
                            "parameter_names", "transfer_certificate", "ordering_readout", "functional"),
                 "training identity")
    canonical_bytes(identity)
    if (identity["schema"] != TRAINING_SCHEMA or identity["selection"] != "fixed-last"
            or identity["ordering_readout"] != "dropout-disabled-with-checkpointing/1"
            or identity["optimizer"] != {"kind": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                                         "foreach": False, "fused": False, "capturable": False}):
        raise ValueError("unsupported training schema, selection or optimizer recipe")
    if expected_identity is not None and canonical_bytes(identity) != canonical_bytes(expected_identity):
        raise ValueError("checkpoint training identity mismatch")
    if identity["model"] != policy.identity() or identity["rng"] != RNG_ALGORITHM:
        raise ValueError("checkpoint model or random algorithm mismatch")
    config = TrainingConfig(**identity["config"])
    recorded_input = identity["input"]
    if not isinstance(recorded_input, dict) or not {"budget", "functional"} <= set(recorded_input):
        raise ValueError("training input identity lacks the derived budget and functional records")
    if config.stage == "mid":
        if recorded_input["budget"] is not None:
            raise ValueError("midtraining records no post fitting budget")
        budget_ids = ()
    else:
        budget_ids = validate_post_budget_identity(recorded_input["budget"])
    parameters = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if identity["parameter_names"] != [name for name, _ in parameters]:
        raise ValueError("checkpoint ordered parameter names differ")
    require_keys(payload["weights"], [name for name, _ in parameters], "weights")
    for name, parameter in parameters:
        _tensor(payload["weights"][name], parameter, name)
    progress = payload["progress"]
    require_keys(progress, ("next_update", "counters", "seen", "unique", "fitting_endpoints_available",
                            "selection_labels", "inherited_exposure", "functional"), "progress")
    update = _nonnegative(progress["next_update"], "next_update")
    if update > config.total_updates:
        raise ValueError("checkpoint exceeds declared horizon")
    if payload["rng"] != {"algorithm": RNG_ALGORITHM, "seed": config.seed, "next_update": update}:
        raise ValueError("checkpoint RNG boundary mismatch")
    counters = progress["counters"]
    keys = ("examples", "masked_residues", "teacher_forced_states", "pair_draws",
            "informative_pairs", "ties", "antigen_examples")
    require_keys(counters, keys, "counters")
    for key, value in counters.items():
        _nonnegative(value, key)
    examples = update * config.examples_per_update
    draws = examples if config.ordering_weight else 0
    if (counters["examples"] != examples or counters["masked_residues"] < examples
            or counters["antigen_examples"] != (examples if config.arm == "paired" else 0)
            or counters["pair_draws"] != draws
            or counters["informative_pairs"] + counters["ties"] != draws
            or counters["teacher_forced_states"] != 20 * counters["informative_pairs"]
            or (update == 0 and any(counters.values()))):
        raise ValueError("inconsistent training counters")
    endpoint_keys = ("endpoints", "reconstruction_endpoints", "ordering_endpoints",
                     "informative_ordering_endpoints")
    seen_keys = ("records", "antibody_families", "target_families", "sources", *endpoint_keys)
    require_keys(progress["seen"], seen_keys, "seen IDs")
    require_keys(progress["unique"], seen_keys, "unique counts")
    for key in seen_keys:
        values = progress["seen"][key]
        if (not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values)
                or values != sorted(set(values)) or type(progress["unique"][key]) is not int
                or progress["unique"][key] != len(values)
                or len(values) > examples + 2 * draws):
            raise ValueError("invalid observed-ID counters")
    available = _nonnegative(progress["fitting_endpoints_available"], "fitting endpoints")
    # Seen endpoints must be a SUBSET of the available budget; completion never requires equality.
    if (progress["selection_labels"] != 0 or type(progress["selection_labels"]) is not int
            or (config.stage == "mid" and (available or any(progress["seen"][key] for key in endpoint_keys)))
            or (config.stage == "post" and (not available or
                any(progress["seen"][key] for key in seen_keys if key not in endpoint_keys)))
            or available != len(budget_ids)
            or len(progress["seen"]["endpoints"]) > available
            or set(progress["seen"]["endpoints"]) - set(budget_ids)):
        raise ValueError("invalid endpoint exposure counters")
    seen = {key: set(progress["seen"][key]) for key in endpoint_keys}
    if (seen["endpoints"] != seen["reconstruction_endpoints"] | seen["ordering_endpoints"]
            or not seen["informative_ordering_endpoints"] <= seen["ordering_endpoints"]
            or len(seen["reconstruction_endpoints"]) > examples
            or len(seen["ordering_endpoints"]) > 2 * draws
            or len(seen["informative_ordering_endpoints"]) > 2 * counters["informative_pairs"]):
        raise ValueError("inconsistent reconstruction and measurement exposure")
    canonical_bytes(progress)
    optimizer = _adamw(payload["optimizer"], parameters, lr=config.learning_rate,
                       weight_decay=config.weight_decay,
                       group_lr=learning_rate(config, max(0, update - 1)), step=update, label="policy")

    functional = validate_functional_training(identity, progress, restricted=restricted)
    section = payload["functional"]
    if functional is None:
        if section is not None:
            raise ValueError("checkpoint carries a head that its identity does not declare")
        return optimizer, None
    settings, hidden = functional["config"], functional["hidden_size"]
    if hidden != policy.backbone.config.hidden_size:
        raise ValueError("head width differs from the policy hidden size")
    require_keys(section, ("weights", "optimizer"), "functional payload")
    require_keys(section["weights"], ("weight",), "head weights")
    parameter = head.weight if head is not None else torch.nn.Parameter(
        torch.empty(1, hidden, dtype=torch.float32))
    if tuple(parameter.shape) != (1, hidden) or parameter.dtype != torch.float32:
        raise ValueError("head parameter differs from the recorded architecture")
    _tensor(section["weights"]["weight"], parameter, "head weight")
    # The head moves only with a head step, so without one it is still the initial head
    # (a resume starts from a pristine trainer, i.e. the deterministic initialization).
    # Save hands over the live (possibly CUDA) tensor and resume a CPU one: compare on one device.
    if (head is not None and functional["ordered_updates"] == 0
            and not torch.equal(section["weights"]["weight"].detach().cpu(), parameter.detach().cpu())):
        raise ValueError("head weights changed although no head step was taken")
    # The head learning rate tracks the last GLOBAL update, also when it did not step.
    head_optimizer = _adamw(
        section["optimizer"], [("weight", parameter)], lr=config.learning_rate * settings.head_lr_scale,
        weight_decay=settings.head_weight_decay,
        group_lr=learning_rate(config, max(0, update - 1)) * settings.head_lr_scale,
        step=functional["ordered_updates"], label="head")
    return optimizer, head_optimizer


def read_checkpoint(path, *, expected_sha256=None):
    if expected_sha256 is not None and file_digest(path) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    # mmap avoids eagerly allocating full backbone + Adam state on the host.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def resume_checkpoint(trainer, path, *, expected_sha256=None):
    """Resume into a pristine trainer; all semantic checks precede live mutation.

    Validation failures leave the trainer unchanged: every metadata, tensor, BOTH optimizers
    and the label replay of the functional counters are checked first, and both staged
    optimizer states are allocated (CPU mmap aliases cloned) before any live tensor changes.
    Hardware failure during the final tensor copies is fatal; reload into a new trainer
    rather than continuing.
    """
    if trainer.next_update or trainer.optimizer.state or trainer.failed:
        raise ValueError("resume requires a pristine trainer")
    if trainer.head is not None and (trainer.head_optimizer.state or trainer.head.weight.grad is not None):
        raise ValueError("resume requires a pristine trainer")
    payload = read_checkpoint(path, expected_sha256=expected_sha256)
    staged, staged_head = _validate(
        payload, trainer.policy, expected_identity=trainer.identity, head=trainer.head,
        restricted=None if trainer.functional is None else trainer.functional.data)
    progress = payload["progress"]
    if progress["inherited_exposure"] != trainer.progress()["inherited_exposure"]:
        raise ValueError("checkpoint inherited exposure mismatch")
    if trainer.post_data:
        if (progress["fitting_endpoints_available"] != trainer.post_data.fitting_count
                or set(progress["seen"]["endpoints"]) - set(trainer.post_data.budget.selected_ids)
                or progress["inherited_exposure"] != trainer.progress()["inherited_exposure"]):
            raise ValueError("checkpoint endpoint exposure mismatch")
    if trainer.sampler:
        records = trainer.sampler._record_by_id
        allowed = {"records": set(records),
                   "antibody_families": {r.antibody_family for r in records.values()},
                   "target_families": {r.target_family for r in records.values()},
                   "sources": {r.owning_source_id for r in records.values()}}
        if any(set(progress["seen"][key]) - values for key, values in allowed.items()):
            raise ValueError("checkpoint observed identities not in training population")
    # Allocate optimizer state before changing model tensors; detach all mmap
    # aliases, including Adam's CPU step scalars (important on Windows replace).
    staged.load_state_dict(payload["optimizer"])
    states = [staged]
    if staged_head is not None:
        staged_head.load_state_dict(payload["functional"]["optimizer"])
        states.append(staged_head)
    for optimizer in states:
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    # CUDA moments already copied on load; CPU tensors still alias mmap.
                    if value.device.type == "cpu":
                        entry[key] = value.clone()
    try:
        trainer.policy.load_trainable_state(payload["weights"])
        if staged_head is not None:
            with torch.no_grad():
                trainer.head.weight.copy_(payload["functional"]["weights"]["weight"])
    except BaseException:
        trainer.failed = True
        raise
    trainer.optimizer = staged
    trainer.head_optimizer = staged_head
    trainer.next_update = progress["next_update"]
    trainer.counters = dict(progress["counters"])
    trainer.seen = {key: set(values) for key, values in progress["seen"].items()}
    if progress["functional"] is not None:
        trainer.functional_counters = dict(progress["functional"]["counters"])
        trainer.functional_seen = {key: {tuple(item) for item in progress["functional"]["seen"][key]}
                                   for key in SEEN_KEYS}
    return trainer.progress()


def load_inference_checkpoint(policy, path, *, encoder, expected_sha256, source_identity):
    """Load inference weights without reopening any training measurement file.

    The head and both optimizer states are validated like any other part of the file, then
    dropped: only the policy weights are loaded.
    """
    payload = read_checkpoint(path, expected_sha256=expected_sha256)
    _validate(payload, policy)
    if payload["identity"]["source"] != source_identity:
        raise ValueError("checkpoint source identity mismatch")
    if payload["identity"]["encoder_runtime"] != encoder.runtime_identity():
        raise ValueError("checkpoint encoder runtime mismatch")
    policy.load_trainable_state(payload["weights"])
    return {"checkpoint_sha256": expected_sha256, "training": payload["identity"],
            "progress": payload["progress"]}
