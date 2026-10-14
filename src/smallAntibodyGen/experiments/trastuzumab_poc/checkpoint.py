"""Strict update-boundary checkpoints, distinct from standalone stage parents."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    canonical_bytes, file_digest, require_keys,
)
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import RNG_ALGORITHM
from smallAntibodyGen.experiments.trastuzumab_poc.runner import TrainingConfig, learning_rate


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
    if trainer.failed or any(p.grad is not None for _, p in trainer.parameters):
        raise ValueError("checkpoint requires a successful optimizer boundary")
    # Synchronous save: the caller must not train concurrently. Detached references
    # avoid duplicating an entire FP32 650M backbone in host memory.
    payload = {
        "schema": "trastuzumab-checkpoint/1", "identity": copy.deepcopy(trainer.identity),
        "weights": {name: p.detach() for name, p in trainer.parameters},
        "optimizer": trainer.optimizer.state_dict(), "progress": trainer.progress(),
        "rng": {"algorithm": RNG_ALGORITHM, "seed": trainer.config.seed,
                "next_update": trainer.next_update},
    }
    _validate(payload, trainer.policy, expected_identity=trainer.identity)
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


def _validate(payload, policy, *, expected_identity=None):
    require_keys(payload, ("schema", "identity", "weights", "optimizer", "progress", "rng"), "checkpoint")
    if payload["schema"] != "trastuzumab-checkpoint/1":
        raise ValueError("unsupported checkpoint schema")
    identity = payload["identity"]
    require_keys(identity, ("schema", "config", "model", "encoder_runtime", "data", "input", "source",
                            "initialization", "rng", "runtime", "optimizer", "selection",
                            "parameter_names"), "training identity")
    canonical_bytes(identity)
    if (identity["schema"] != "trastuzumab-training/1" or identity["selection"] != "fixed-last"
            or identity["optimizer"] != {"kind": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                                         "foreach": False, "fused": False, "capturable": False}):
        raise ValueError("unsupported training schema, selection or optimizer recipe")
    if expected_identity is not None and canonical_bytes(identity) != canonical_bytes(expected_identity):
        raise ValueError("checkpoint training identity mismatch")
    if identity["model"] != policy.identity() or identity["rng"] != RNG_ALGORITHM:
        raise ValueError("checkpoint model or random algorithm mismatch")
    config = TrainingConfig(**identity["config"])
    parameters = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if identity["parameter_names"] != [name for name, _ in parameters]:
        raise ValueError("checkpoint ordered parameter names differ")
    require_keys(payload["weights"], [name for name, _ in parameters], "weights")
    for name, parameter in parameters:
        _tensor(payload["weights"][name], parameter, name)
    progress = payload["progress"]
    require_keys(progress, ("next_update", "counters", "seen", "unique", "fitting_endpoints_available",
                            "selection_labels", "inherited_exposure"), "progress")
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
    seen_keys = ("records", "antibody_families", "target_families", "sources", "endpoints")
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
    if (progress["selection_labels"] != 0 or type(progress["selection_labels"]) is not int
            or (config.stage == "mid" and (available or progress["seen"]["endpoints"]))
            or (config.stage == "post" and (not available or
                any(progress["seen"][key] for key in seen_keys if key != "endpoints")))
            or len(progress["seen"]["endpoints"]) > available):
        raise ValueError("invalid endpoint exposure counters")
    canonical_bytes(progress)
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=config.learning_rate,
                                   weight_decay=config.weight_decay, foreach=False, fused=False, capturable=False)
    expected = optimizer.state_dict()
    actual = payload["optimizer"]
    require_keys(actual, ("state", "param_groups"), "optimizer")
    if not isinstance(actual["param_groups"], list) or len(actual["param_groups"]) != 1:
        raise ValueError("optimizer must contain exactly one group")
    group = actual["param_groups"][0]
    template = expected["param_groups"][0]
    template["lr"] = learning_rate(config, max(0, update - 1))
    if canonical_bytes(group) != canonical_bytes(template):
        raise ValueError("optimizer parameter order or options mismatch")
    state = actual["state"]
    if not isinstance(state, dict) or set(state) != (set(range(len(parameters))) if update else set()):
        raise ValueError("optimizer state inventory mismatch")
    for index, (_, parameter) in enumerate(parameters):
        if not update:
            break
        entry = state[index]
        require_keys(entry, ("step", "exp_avg", "exp_avg_sq"), "AdamW state")
        step = entry["step"]
        if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                or step.dtype != torch.float32 or not bool(torch.isfinite(step))
                or float(step) != update):
            raise ValueError("invalid AdamW step")
        _tensor(entry["exp_avg"], parameter, "exp_avg")
        _tensor(entry["exp_avg_sq"], parameter, "exp_avg_sq", nonnegative=True)
    return optimizer


def read_checkpoint(path, *, expected_sha256=None):
    if expected_sha256 is not None and file_digest(path) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    # mmap avoids eagerly allocating full backbone + Adam state on the host.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def resume_checkpoint(trainer, path, *, expected_sha256=None):
    """Resume into a pristine trainer; all semantic checks precede live mutation.

    Validation failures leave the trainer unchanged. Hardware failure during the
    final tensor copies is fatal; reload into a new trainer rather than continuing.
    """
    if trainer.next_update or trainer.optimizer.state or trainer.failed:
        raise ValueError("resume requires a pristine trainer")
    payload = read_checkpoint(path, expected_sha256=expected_sha256)
    staged = _validate(payload, trainer.policy, expected_identity=trainer.identity)
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
    for entry in staged.state.values():
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                # CUDA moments already copied on load; CPU tensors still alias mmap.
                if value.device.type == "cpu":
                    entry[key] = value.clone()
    try:
        trainer.policy.load_trainable_state(payload["weights"])
    except BaseException:
        trainer.failed = True
        raise
    trainer.optimizer = staged
    trainer.next_update = progress["next_update"]
    trainer.counters = dict(progress["counters"])
    trainer.seen = {key: set(values) for key, values in progress["seen"].items()}
    return trainer.progress()


def load_inference_checkpoint(policy, path, *, expected_sha256, source_identity):
    """Load inference weights without reopening any training measurement file."""
    payload = read_checkpoint(path, expected_sha256=expected_sha256)
    _validate(payload, policy)
    if payload["identity"]["source"] != source_identity:
        raise ValueError("checkpoint source identity mismatch")
    policy.load_trainable_state(payload["weights"])
    return {"checkpoint_sha256": expected_sha256, "training": payload["identity"],
            "progress": payload["progress"]}
