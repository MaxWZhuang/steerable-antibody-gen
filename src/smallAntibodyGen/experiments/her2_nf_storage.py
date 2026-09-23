"""Bounded-memory, verified publication for this flight's checkpoints."""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

from .her2_runtime import require


def feed_tensor(digest, tensor, *, chunk_elements=262144):
    """Feed the historical contiguous byte order without a full tensor byte copy."""
    flat = tensor.detach().contiguous().view(-1)
    for piece in flat.split(chunk_elements):
        array = piece.cpu().numpy()
        digest.update(memoryview(array).cast("B"))


def state_dict_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(str(name).encode())
        feed_tensor(digest, tensor)
    return digest.hexdigest()


def state_digest(model):
    return state_dict_digest(model.state_dict())


def load_cpu(path, *, weights_only=False):
    import torch
    # Checkpoints use PyTorch's zip format. Mapping avoids an additional complete
    # model + Adam moments allocation while verifying bytes already on disk.
    return torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)


def atomic_torch_save(payload, target, *, verify):
    import torch
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    restored = None
    try:
        torch.save(payload, temporary)
        restored = load_cpu(temporary)
        verify(restored)
        del restored
        restored = None
        gc.collect()  # Release Windows mappings before rename, including cycles.
        os.replace(temporary, target)
    finally:
        del restored
        if temporary.is_file():
            temporary.unlink()


def save_checkpoint(path, policy, document):
    from .her2_policy import POLICY_SCHEMA
    digest = state_digest(policy.model)
    tensors = policy.model.state_dict()
    payload = {"schema_version": POLICY_SCHEMA, "state": tensors,
               "state_sha256": digest, **document}

    def verify(restored):
        require(restored.get("schema_version") == POLICY_SCHEMA,
                "Checkpoint schema changed during publication")
        require(restored.get("state_sha256") == digest
                and state_dict_digest(restored["state"]) == digest,
                "Checkpoint tensors do not reproduce their recorded digest")
        require(set(restored["state"]) == set(tensors), "Checkpoint parameter set changed")
        for key, value in tensors.items():
            actual = restored["state"][key]
            require(actual.dtype == value.dtype and actual.shape == value.shape,
                    f"Checkpoint tensor contract changed: {key}")
        require(all(restored.get(key) == value for key, value in document.items()),
                "Checkpoint identity or metadata changed")

    target = Path(path)
    if target.is_file():
        restored = load_cpu(target, weights_only=True)
        try:
            verify(restored)
        finally:
            del restored
        return digest
    atomic_torch_save(payload, target, verify=verify)
    return digest


def collect_unused():
    """Release exception cycles before starting the next independent model."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
