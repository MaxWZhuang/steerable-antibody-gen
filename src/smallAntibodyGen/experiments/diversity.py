"""Score-function entropy gradients for freshly sampled discrete policies."""
from __future__ import annotations

import torch


def negative_entropy_surrogate(log_q):
    """Unbiased gradient of -H(q) when observations are independent fresh q draws.

    Re-score the sampled sequences with gradients enabled. Sampling itself is
    detached. E[(log q + 1) grad log q] = grad(-H); the constant 1 has zero
    expected contribution. A leave-one-out baseline reduces variance without
    depending on the current sample. Do not use teacher-forced dataset rows,
    deduplicated samples, stale samples, or a within-batch softmax here.

    The returned scalar is a gradient carrier, NOT an entropy estimate or a
    meaningful loss value. Report -mean(detached log_q) as Monte Carlo entropy.
    """
    if (not isinstance(log_q, torch.Tensor) or not log_q.is_floating_point()
            or log_q.ndim != 1 or log_q.numel() < 2):
        raise ValueError("Entropy requires at least two floating-point sequence log probabilities")
    if not bool(torch.isfinite(log_q).all()) or bool((log_q > 1e-6).any()):
        raise ValueError("Entropy requires finite nonpositive log probabilities")
    detached = log_q.detach()
    baseline = (detached.sum() - detached) / (len(detached) - 1)
    return (log_q * (detached - baseline)).mean()
