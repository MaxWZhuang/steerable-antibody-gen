"""Versioned, purpose-separated random streams with no outer RNG mutation."""
from contextlib import contextmanager
import random

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import digest, seed_value

RNG_ALGORITHM = "sha256-purpose-update/1"


def keyed_seed(seed, update, example, purpose):
    seed_value(seed)
    if any(type(x) is not int or x < 0 for x in (update, example)):
        raise ValueError("update and example must be non-negative integers")
    return int(digest({"algorithm": RNG_ALGORITHM, "seed": seed, "update": update,
                       "example": example, "purpose": purpose})[:16], 16) % (2**63)


def keyed_random(seed, update, example, purpose):
    return random.Random(keyed_seed(seed, update, example, purpose))


@contextmanager
def isolated_torch_rng(seed, device="cpu"):
    seed_value(seed)
    device = torch.device(device)
    devices = [torch.cuda.current_device() if device.index is None else device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield
