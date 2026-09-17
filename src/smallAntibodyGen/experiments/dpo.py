"""DPO on declared sequence log probabilities and identity-bound reference caches.

Loss: Rafailov et al., arXiv:2305.18290, equation 7. Callers either sample
according to pair weights or weight the loss, never both. References are fixed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def dpo_per_pair_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta):
    """Return one stable loss per pair, without changing the caller's weighting."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    tensors = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(not isinstance(t, torch.Tensor) or not t.is_floating_point() for t in tensors):
        raise ValueError("DPO requires floating-point tensors")
    if policy_chosen.ndim != 1 or policy_chosen.numel() == 0:
        raise ValueError("DPO expects nonempty vectors, one value per pair")
    if any(t.shape != policy_chosen.shape or t.device != policy_chosen.device
           or t.dtype != policy_chosen.dtype for t in tensors):
        raise ValueError("All DPO tensors must share shape, dtype and device")
    if reference_chosen.requires_grad or reference_rejected.requires_grad:
        raise ValueError("Reference log probabilities must be frozen")
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("Nonfinite DPO log probability")
    margin = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    if not bool(torch.isfinite(margin).all()):
        raise ValueError("Nonfinite DPO margin")
    return F.softplus(-margin)


def _canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def genotype_digest(genotypes):
    return hashlib.sha256(_canonical(list(genotypes))).hexdigest()


def _check_genotypes(genotypes):
    if (not genotypes or len(set(genotypes)) != len(genotypes)
            or any(not isinstance(g, str) or len(g) != 16 or set(g) - {"0", "1"} for g in genotypes)):
        raise ValueError("Cache genotypes must be unique 16-character binary strings")


def write_reference_cache(path, identity, genotypes, log_probabilities):
    """Write a new cache with a content digest; never overwrite an earlier reference."""
    genotypes = list(genotypes)
    _check_genotypes(genotypes)
    values = np.asarray(log_probabilities, dtype=float)
    if values.shape != (len(genotypes),) or not np.isfinite(values).all() or (values > 1e-6).any():
        raise ValueError("Cache needs one finite, nonpositive sequence log probability per genotype")
    body = {"schema_version": "fixed-reference-cache/1", "identity": identity,
            "genotypes": genotypes, "log_probabilities": values.tolist()}
    document = {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    encoded = json.dumps(document, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def load_reference_cache(path, expected_identity, expected_genotypes):
    """Reject a different checkpoint, geometry, scoring contract or genotype order."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(document) != {"schema_version", "identity", "genotypes", "log_probabilities", "content_sha256"}:
        raise ValueError("Unexpected reference-cache fields")
    digest = document.pop("content_sha256")
    if hashlib.sha256(_canonical(document)).hexdigest() != digest:
        raise ValueError("Reference-cache content digest mismatch")
    if document["schema_version"] != "fixed-reference-cache/1" or document["identity"] != expected_identity:
        raise ValueError("Reference-cache identity mismatch")
    _check_genotypes(document["genotypes"])
    if document["genotypes"] != list(expected_genotypes):
        raise ValueError("Reference-cache genotype order mismatch")
    values = np.asarray(document["log_probabilities"], dtype=float)
    if values.shape != (len(expected_genotypes),) or not np.isfinite(values).all() or (values > 1e-6).any():
        raise ValueError("Invalid reference-cache probabilities")
    values.setflags(write=False)
    return values
