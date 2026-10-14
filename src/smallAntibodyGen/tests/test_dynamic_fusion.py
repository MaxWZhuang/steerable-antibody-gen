"""Dynamic context, update_mode, compute_dtype and BF16 contracts."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from smallAntibodyGen.models.dplm_policy import DPLMConfig, NativeDPLM, DPLM_MASK_ID
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.antibody_antigen.esm_ppi import FrozenESMPPI
from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.precision import native_bf16_supported
from smallAntibodyGen.models.antibody_antigen.views import mask_complete, VisibleView
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import (
    whole_core_scores, MaskedTargets,
)

IDENTITY = {
    "source_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
    "extraction_layer": "layer_2", "precision": "float32",
    "tokenizer": "ESM-1b", "eval_mode": "true",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeAlphabet:
    cls_idx, eos_idx, mask_idx = 0, 2, 32
    def get_idx(self, residue):
        return 4 + "ACDEFGHIKLMNPQRSTVWY".index(residue)


class SpyEncoder(nn.Module):
    """Records every extract() call: tokens seen and grad-enabled state."""
    def __init__(self, feature_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(33, feature_dim)
        self.num_layers = 2
        self.alphabet = FakeAlphabet()
        self.calls: list = []

    def forward(self, tokens, **kwargs):
        self.calls.append({
            "tokens": tokens.clone(),
            "training": self.training,
            "grad_enabled": torch.is_grad_enabled(),
        })
        x = self.embedding(tokens)
        return {"representations": {2: x}}


def make_spy_encoder(feature_dim=8) -> FrozenESMPPI:
    native = SpyEncoder(feature_dim)
    return FrozenESMPPI(native, IDENTITY)


def make_policy(*, feature_dim=8, update_mode="lora", compute_dtype="float32",
                checkpointing=False):
    torch.manual_seed(42)
    cfg = DPLMConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
    )
    native = NativeDPLM.random_init(cfg, seed=7, std=0.1)
    config = FusionConfig(
        context_dim=feature_dim, bottleneck=6, lora_rank=2, lora_alpha=4,
        activation_checkpointing=checkpointing,
        update_mode=update_mode,
        compute_dtype=compute_dtype,
    )
    return FusedDPLM(native, IDENTITY, config)


def make_static_context(view, feature_dim=8):
    return StaticContext(view, torch.randn(len(view.coords()), feature_dim), IDENTITY)


# ---------------------------------------------------------------------------
# FusionConfig validation
# ---------------------------------------------------------------------------

def test_fusion_config_defaults_preserve_lora_fp32():
    cfg = FusionConfig(context_dim=8, bottleneck=4, lora_rank=2)
    assert cfg.update_mode == "lora"
    assert cfg.compute_dtype == "float32"


def test_fusion_config_rejects_unknown_update_mode():
    with pytest.raises(ValueError, match="update_mode"):
        FusionConfig(context_dim=8, bottleneck=4, lora_rank=2, update_mode="half")


def test_fusion_config_rejects_unknown_compute_dtype():
    with pytest.raises(ValueError, match="compute_dtype"):
        FusionConfig(context_dim=8, bottleneck=4, lora_rank=2, compute_dtype="float16")


# ---------------------------------------------------------------------------
# update_mode="full" parameter inventory
# ---------------------------------------------------------------------------

def test_full_mode_all_backbone_params_require_grad():
    policy = make_policy(update_mode="full")
    backbone_params = list(policy.backbone.parameters())
    assert backbone_params, "backbone has no parameters"
    assert all(p.requires_grad for p in backbone_params), (
        "update_mode='full' must leave all backbone params trainable"
    )


def test_lora_mode_base_backbone_frozen_adapters_trainable():
    policy = make_policy(update_mode="lora")
    for name, p in policy.backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad, f"{name} should require grad in lora mode"
        else:
            assert not p.requires_grad, f"{name} should be frozen in lora mode"


def test_full_mode_base_grads_flow():
    """Backward must update base backbone weights in full mode."""
    policy = make_policy(update_mode="full")
    policy.train()
    view, targets = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    ctx = make_static_context(view)
    loss = policy.reconstruction_nll([view], [ctx], [targets])
    loss.backward()
    backbone_grads = {n: p.grad for n, p in policy.backbone.named_parameters()
                      if p.grad is not None}
    assert backbone_grads, "no backbone gradients in full mode"


def test_full_mode_rejects_half_storage():
    cfg = DPLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    native = NativeDPLM.random_init(cfg, seed=0, storage_dtype=torch.float16)
    with pytest.raises(ValueError, match="float32"):
        FusedDPLM(native, IDENTITY, FusionConfig(context_dim=8, bottleneck=4,
                                                  lora_rank=2, update_mode="full"))


def test_full_and_lora_same_init_same_forward():
    """At zero-residual init both modes produce identical logits (fusion residual is zero)."""
    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    ctx = make_static_context(view)

    def make(mode):
        torch.manual_seed(0)
        cfg = DPLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64)
        native = NativeDPLM.random_init(cfg, seed=5, std=0.1)
        p = FusedDPLM(native, IDENTITY,
                      FusionConfig(context_dim=8, bottleneck=6, lora_rank=2, update_mode=mode))
        p.eval()
        return p

    lora_p = make("lora")
    full_p = make("full")
    with torch.no_grad():
        a = lora_p.bind(ctx).forward(view).raw_logits
        b = full_p.bind(ctx).forward(view).raw_logits
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# identity() includes update_mode and compute_dtype
# ---------------------------------------------------------------------------

def test_identity_includes_update_mode_and_compute_dtype():
    policy = make_policy(update_mode="full", compute_dtype="float32")
    ident = policy.identity()
    assert ident["update_mode"] == "full"
    assert ident["compute_dtype"] == "float32"


# ---------------------------------------------------------------------------
# BF16 compute_dtype
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not native_bf16_supported("cuda"),
    reason="no native BF16 CUDA",
)
def test_bf16_forward_produces_finite_fp32_logits():
    policy = make_policy(compute_dtype="bfloat16")
    policy = policy.cuda()
    policy.eval()
    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    feats_cpu = torch.randn(len(view.coords()), 8)
    ctx = StaticContext(view, feats_cpu, IDENTITY)
    with torch.no_grad():
        logits, _, _ = policy.forward_views([view], [ctx])
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    not native_bf16_supported("cuda"),
    reason="no native BF16 CUDA",
)
def test_bf16_reconstruction_finite_and_close_to_fp32():
    """FP32 and BF16 (same weights, eval mode) must produce close but not exact results.

    The backbone runs under bfloat16 autocast, so intermediate activations differ by
    rounding; a bounded tolerance is correct.  Exact equality is NOT expected because
    the backbone hidden states differ between the two compute paths.
    """
    view, targets = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])

    def make_cuda(dtype_str):
        torch.manual_seed(0)
        cfg = DPLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64)
        native = NativeDPLM.random_init(cfg, seed=3, std=0.1)
        p = FusedDPLM(native, IDENTITY, FusionConfig(
            context_dim=8, bottleneck=6, lora_rank=2, compute_dtype=dtype_str))
        return p.cuda().eval()  # eval: no dropout; consistent across calls

    fp32_p = make_cuda("float32")
    bf16_p = make_cuda("bfloat16")
    feats = torch.randn(len(view.coords()), 8)
    ctx_cpu = StaticContext(view, feats, IDENTITY)

    with torch.no_grad():
        fp32_loss = fp32_p.reconstruction_nll([view], [ctx_cpu], [targets])
        bf16_loss = bf16_p.reconstruction_nll([view], [ctx_cpu], [targets])
    assert torch.isfinite(fp32_loss) and torch.isfinite(bf16_loss)
    # BF16 backbone rounds intermediate activations; 5% relative tolerance accepted.
    torch.testing.assert_close(fp32_loss, bf16_loss, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not native_bf16_supported("cuda"),
    reason="no native BF16 CUDA",
)
def test_bf16_backward_finite():
    """BF16 compute mode must produce finite gradients for all trainable parameters."""
    view, targets = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    torch.manual_seed(0)
    cfg = DPLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    native = NativeDPLM.random_init(cfg, seed=3, std=0.1)
    policy = FusedDPLM(native, IDENTITY, FusionConfig(
        context_dim=8, bottleneck=6, lora_rank=2,
        activation_checkpointing=False, compute_dtype="bfloat16"))
    policy.cuda().train()
    feats = torch.randn(len(view.coords()), 8)
    ctx_cpu = StaticContext(view, feats, IDENTITY)
    loss = policy.reconstruction_nll([view], [ctx_cpu], [targets])
    loss.backward()
    for name, p in policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), (
                f"non-finite or missing grad at {name}"
            )


def test_bf16_on_cpu_raises():
    policy = make_policy(compute_dtype="bfloat16")
    policy.eval()
    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    ctx = make_static_context(view)
    with torch.no_grad():
        with pytest.raises(ValueError, match="CUDA"):
            policy.forward_views([view], [ctx])


def test_identity_precision_mismatch_rejected():
    """Encoder identity with precision='bfloat16' must be refused by FrozenESMPPI."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(33, 8, dtype=torch.float32)
            self.num_layers = 1
            self.alphabet = FakeAlphabet()
        def forward(self, tokens, **kwargs):
            x = self.embedding(tokens)
            return {"representations": {1: x}}

    bad_identity = dict(IDENTITY, precision="bfloat16")
    with pytest.raises(ValueError, match="float32"):
        FrozenESMPPI(TinyModel(), bad_identity)


# ---------------------------------------------------------------------------
# DynamicContextProvider construction
# ---------------------------------------------------------------------------

def test_provider_rejects_ag_in_get_context():
    encoder = make_spy_encoder()
    ref_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    provider = DynamicContextProvider(encoder, ref_view)
    ag_view, _ = mask_complete({"H": "ACDE", "Ag": "FGH"}, [("H", 1)])
    with pytest.raises(ValueError, match="Ag"):
        provider.get_context(ag_view)


def test_provider_rejects_wrong_chain_length():
    encoder = make_spy_encoder()
    ref_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    provider = DynamicContextProvider(encoder, ref_view)
    wrong, _ = mask_complete({"H": "ACDEG"}, [("H", 1)])
    with pytest.raises(ValueError, match="length"):
        provider.get_context(wrong)


def test_provider_immutable_ag_appended():
    """Provider built from a view with Ag must append the Ag to every encoder call."""
    calls = []

    class RecordingEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(33, 8, dtype=torch.float32)
            self.num_layers = 2
            self.alphabet = FakeAlphabet()
        def forward(self, tokens, **kwargs):
            calls.append(tokens.clone())
            x = self.embedding(tokens)
            return {"representations": {2: x}}

    native = RecordingEncoder()
    encoder = FrozenESMPPI(native, IDENTITY)
    ref_view, _ = mask_complete({"H": "ACDE", "Ag": "FGH"}, [("H", 1)])
    provider = DynamicContextProvider(encoder, ref_view)

    ab_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    ctx = provider.get_context(ab_view)
    # The encoder received an H+Ag combined view: 4 H residues + 3 Ag + 2 specials per chain.
    assert calls, "encoder was not called"
    # StaticContext covers H + Ag residues; ab_features returns only H rows.
    assert ctx.features.shape == (7, 8)  # 4 H + 3 Ag
    assert ctx.ab_features().shape == (4, 8)


def test_provider_get_context_returns_fresh_context_each_call():
    """Same view -> same content_key but independent StaticContext objects."""
    encoder = make_spy_encoder()
    ref_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    provider = DynamicContextProvider(encoder, ref_view)
    ab_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    ctx1 = provider.get_context(ab_view)
    ctx2 = provider.get_context(ab_view)
    assert ctx1 is not ctx2
    assert ctx1.content_key == ctx2.content_key


# ---------------------------------------------------------------------------
# Spy encoder in whole_core_scores with dynamic provider
# ---------------------------------------------------------------------------

def test_dynamic_whole_core_scores_encoder_sees_changing_views():
    """Encoder must be called with the changing masked view, never with hidden targets."""
    spy_enc = make_spy_encoder(feature_dim=8)
    inner = spy_enc.model  # SpyEncoder nn.Module

    policy = make_policy(feature_dim=8)
    policy.eval()

    ref_view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    provider = DynamicContextProvider(spy_enc, ref_view)

    ab_view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    targets = [MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"})]
    order = (("H", 2), ("H", 3))

    inner.calls.clear()
    scores = whole_core_scores(policy, provider, ab_view, targets, order, state_microbatch=1)

    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()

    # 2 steps * 1 core = 2 encoder calls.
    assert len(inner.calls) == 2, f"expected 2 encoder calls, got {len(inner.calls)}"

    # Step 0: H:2 and H:3 both masked.
    toks_0 = inner.calls[0]["tokens"][0]  # [seq_len]
    # Step 1: H:2 revealed (D), H:3 still masked.
    toks_1 = inner.calls[1]["tokens"][0]

    # None of the tokens in step 0 should encode the gold "D" or "E" at positions H:2/H:3
    # (they're mask tokens).  Both calls must be under no_grad and eval.
    for call in inner.calls:
        assert not call["training"]
        assert not call["grad_enabled"]

    # Tokens at H:2 position differ between step 0 and step 1 (revealed vs masked).
    # Verify the second call has a non-mask token somewhere that was masked in the first.
    mask_count_0 = int(toks_0.eq(DPLM_MASK_ID).sum())
    mask_count_1 = int(toks_1.eq(DPLM_MASK_ID).sum())
    assert mask_count_0 > mask_count_1, "step 1 should have fewer masks than step 0"


def test_dynamic_whole_core_scores_encoder_identity_mismatch_rejected():
    different_identity = dict(IDENTITY, checkpoint_sha256="c" * 64)
    encoder = make_spy_encoder()
    ref_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    provider = DynamicContextProvider(encoder, ref_view)
    policy = make_policy(feature_dim=8)
    # Manually set policy's encoder_identity to something different.
    policy.encoder_identity = different_identity
    ab_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    tgt = MaskedTargets.from_mapping({("H", 1): "C"})
    with pytest.raises(ValueError, match="encoder_identity"):
        whole_core_scores(policy, provider, ab_view, [tgt], [("H", 1)])


# ---------------------------------------------------------------------------
# Dynamic generate: fresh context each step, immutable positions preserved
# ---------------------------------------------------------------------------

def test_dynamic_generate_calls_encoder_each_step():
    spy_enc = make_spy_encoder(feature_dim=8)
    inner = spy_enc.model

    policy = make_policy(feature_dim=8)
    policy.eval()

    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    provider = DynamicContextProvider(spy_enc, view)

    inner.calls.clear()
    result = policy.generate(view, provider, steps=3, generator=torch.Generator().manual_seed(0))

    # Encoder is called at each of the 3 decode steps.
    assert len(inner.calls) == 3, f"expected 3 encoder calls, got {len(inner.calls)}"
    # All encoder calls must be eval and no_grad.
    for call in inner.calls:
        assert not call["training"]
        assert not call["grad_enabled"]


def test_dynamic_generate_result_fully_revealed():
    spy_enc = make_spy_encoder(feature_dim=8)
    policy = make_policy(feature_dim=8)
    policy.eval()

    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    provider = DynamicContextProvider(spy_enc, view)

    result = policy.generate(view, provider, steps=4, generator=torch.Generator().manual_seed(1))
    assert not result.hidden_sites(), "generated view must have no masked positions"


def test_dynamic_generate_preserves_immutable_scaffold():
    spy_enc = make_spy_encoder(feature_dim=8)
    policy = make_policy(feature_dim=8)
    policy.eval()

    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    provider = DynamicContextProvider(spy_enc, view)

    result = policy.generate(view, provider, steps=4, generator=torch.Generator().manual_seed(2))
    # Scaffold positions (visible in original view) must be unchanged.
    for coord in view.coords():
        if view.token(coord) is not None:
            assert result.token(coord) == view.token(coord), \
                f"scaffold position {coord} was mutated"


def test_dynamic_generate_ag_swap_changes_encoder_input():
    """Providers with different Ag must produce different feature inputs to the encoder."""
    call_tokens = {}

    class TaggedEncoder(nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag
            self.embedding = nn.Embedding(33, 8, dtype=torch.float32)
            self.num_layers = 2
            self.alphabet = FakeAlphabet()
        def forward(self, tokens, **kwargs):
            call_tokens[self.tag] = tokens.clone()
            x = self.embedding(tokens)
            return {"representations": {2: x}}

    enc_a = FrozenESMPPI(TaggedEncoder("a"), IDENTITY)
    enc_b = FrozenESMPPI(TaggedEncoder("b"), IDENTITY)

    ref_with_ag_a, _ = mask_complete({"H": "ACDE", "Ag": "FGH"}, [("H", 1)])
    ref_with_ag_b, _ = mask_complete({"H": "ACDE", "Ag": "KLM"}, [("H", 1)])
    provider_a = DynamicContextProvider(enc_a, ref_with_ag_a)
    provider_b = DynamicContextProvider(enc_b, ref_with_ag_b)

    ab_view, _ = mask_complete({"H": "ACDE"}, [("H", 1)])
    provider_a.get_context(ab_view)
    provider_b.get_context(ab_view)

    # Token sequences differ because Ag differs.
    assert "a" in call_tokens and "b" in call_tokens
    assert not torch.equal(call_tokens["a"], call_tokens["b"])


# ---------------------------------------------------------------------------
# Static generate still works (regression guard)
# ---------------------------------------------------------------------------

def test_static_generate_still_works():
    policy = make_policy(feature_dim=8)
    policy.eval()
    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    ctx = make_static_context(view)
    result = policy.generate(view, ctx, steps=3, generator=torch.Generator().manual_seed(5))
    assert not result.hidden_sites()
    for coord in view.coords():
        if view.token(coord) is not None:
            assert result.token(coord) == view.token(coord)
