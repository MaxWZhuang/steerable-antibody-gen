"""Reference-policy and representation controls for discrete-policy experiments."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reverse_kl_surrogate(log_q, reference_log_q):
    """Unbiased score-function gradient of KL(q || fixed reference).

    Inputs are re-scores of independent fresh q samples, with duplicates kept.
    The reference must be frozen. The detached log-ratio gets a leave-one-out
    baseline; the constant +1 has zero expected score-function contribution.
    This scalar carries gradients, not an estimated KL value.
    """
    if (not isinstance(log_q, torch.Tensor) or not log_q.is_floating_point()
            or log_q.ndim != 1 or log_q.numel() < 2
            or not isinstance(reference_log_q, torch.Tensor)
            or reference_log_q.shape != log_q.shape or reference_log_q.device != log_q.device
            or not reference_log_q.is_floating_point() or reference_log_q.requires_grad):
        raise ValueError("KL requires matching floating scores and a frozen reference")
    if (not bool(torch.isfinite(log_q).all()) or not bool(torch.isfinite(reference_log_q).all())
            or bool((log_q > 1e-6).any()) or bool((reference_log_q > 1e-6).any())):
        raise ValueError("KL requires finite nonpositive log probabilities")
    ratio = log_q.detach() - reference_log_q
    baseline = (ratio.sum() - ratio) / (len(ratio) - 1)
    return (log_q * (ratio - baseline)).mean()


def pooled_embeddings(features, valid):
    """B,T,D features -> normalized means over valid decoder contexts."""
    if (features.ndim != 3 or not features.is_floating_point() or valid.shape != features.shape[:2]
            or valid.dtype != torch.bool or valid.device != features.device
            or not bool(valid.any(dim=1).all())):
        raise ValueError("Invalid features or valid-position mask")
    # Mask before summing: padded NaNs must not contaminate valid means.
    means = features.masked_fill(~valid.unsqueeze(-1), 0).sum(1) / valid.sum(1, keepdim=True)
    if not bool(torch.isfinite(means).all()) or bool((means.norm(dim=-1) < 1e-8).any()):
        raise ValueError("Nonfinite or zero pooled embedding")
    return F.normalize(means, dim=-1)


def off_diagonal_cosine(embeddings):
    """Average cosine over distinct batch indices, retaining duplicate identities.

    Direct representation gradient only; no claim of differentiating the
    distribution of discrete sampled sequences. Evaluation must also use an
    independent fixed representation or sequence metric.
    """
    if (embeddings.ndim != 2 or len(embeddings) < 2 or not embeddings.is_floating_point()
            or not bool(torch.isfinite(embeddings).all())
            or not bool(torch.allclose(embeddings.norm(dim=-1), torch.ones_like(embeddings[:, 0]), atol=1e-5, rtol=1e-5))):
        raise ValueError("At least two finite unit embeddings required")
    n = len(embeddings)
    return (embeddings.sum(0).square().sum() - embeddings.square().sum()) / (n * (n - 1))


def decoder_statistics(decoder, bound, sequences):
    """Native ESM-IF1 log q plus normalized last-feature means in one forward.

    Experiment-specific: all constrained sequences have the same 121 valid
    prediction positions. The decoder must already be in eval mode. Structural
    context is the same cached frozen encoding for live and reference decoders.
    """
    if decoder.training:
        raise ValueError("Experiment decoder requires dropout off")
    tokens = bound.policy.prefix_tokens(sequences)
    features, _ = decoder(tokens, encoder_out=bound.geometry.decoder_encoder_out(len(sequences)),
                          incremental_state=None, features_only=True)
    features = features.transpose(1, 2)  # upstream returns B,D,T
    logits = decoder.output_layer(features).transpose(1, 2)
    alleles = torch.tensor([bound.space.alleles_for(seq) for seq in sequences], device=tokens.device)
    log_q = bound.policy.log_prob_from_logits(logits, alleles)
    valid = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
    return log_q, pooled_embeddings(features, valid)
