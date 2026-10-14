"""Regression tests for exact masking, preflight, precision, and real mask law."""
import itertools
import math
import random

import pytest
import torch

from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.exact_view_features import ExactViewFeatures
from smallAntibodyGen.models.antibody_antigen.esm_ppi import FrozenESMPPI
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.antibody_antigen.precision import native_bf16_supported
from smallAntibodyGen.models.antibody_antigen.views import VisibleView, mask_complete
from smallAntibodyGen.models.dplm_policy import DPLMConfig, NativeDPLM
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import (
    UniformMaskSpec, draw_mask_uniform, whole_core_scores,
)


IDENTITY = dict(source_sha256="a" * 64, checkpoint_sha256="b" * 64,
                extraction_layer="layer_1", precision="float32", tokenizer="test", eval_mode="true")


class Alphabet:
    cls_idx, eos_idx, mask_idx = 0, 2, 32

    def get_idx(self, residue):
        return 4 + "ACDEFGHIKLMNPQRSTVWY".index(residue)


class Encoder(torch.nn.Module):
    num_layers = 1
    alphabet = Alphabet()

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(33, 8)
        self.projection = torch.nn.Linear(8, 8)
        self.calls = []

    def forward(self, tokens, **kwargs):
        self.calls.append((tokens.detach().cpu().clone(), self.training, torch.is_grad_enabled()))
        return {"representations": {1: self.projection(self.embedding(tokens))}}


def fixture(update_mode="lora", *, checkpointing=False):
    torch.manual_seed(3)
    encoder = FrozenESMPPI(Encoder(), IDENTITY)
    native = NativeDPLM.random_init(DPLMConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=0), seed=4)
    policy = FusedDPLM(native, IDENTITY, FusionConfig(
        context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4,
        activation_checkpointing=checkpointing, update_mode=update_mode))
    combined, truth = mask_complete({"H": "ACDE", "L": "FG", "Ag": "HIK"}, [("H", 1)])
    return policy, encoder, combined.without_ag(), truth, DynamicContextProvider(encoder, combined)


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_all_public_paths_resolve_provider_and_encoder_stays_frozen(mode):
    policy, encoder, view, truth, provider = fixture(mode, checkpointing=True)
    policy.train()
    policy.forward_views([view], [provider])
    policy.queried_log_probs([view], [provider], [("H", 1)], ["C"])
    loss = policy.reconstruction_nll([view], [provider], [truth])
    loss.backward()
    assert len(encoder.model.calls) == 3  # checkpoint backward does not rerun encoder
    assert all(not training and not grad for _, training, grad in encoder.model.calls)
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    selected = [p for p in policy.parameters() if p.requires_grad]
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in selected)
    assert torch.count_nonzero(policy.up.weight.grad) > 0
    if mode == "full":
        assert torch.count_nonzero(policy.backbone.esm.embeddings.word_embeddings.weight.grad) > 0
    assert policy.bind(provider).forward(view).raw_logits.shape[-1] == 20


def test_one_site_twenty_step_generation_accepts_visible_intermediates():
    policy, encoder, view, _, provider = fixture()
    policy.eval()
    result = policy.generate(view, provider, steps=20, generator=torch.Generator().manual_seed(2))
    assert len(encoder.model.calls) == 20
    assert (encoder.model.calls[0][0] == 32).sum() == 1
    assert all((tokens == 32).sum() == 0 for tokens, _, _ in encoder.model.calls[1:])
    assert not result.hidden_sites()
    for coord in view.coords():
        if coord not in view.hidden_sites():
            assert result.token(coord) == view.token(coord)


def test_generation_rejects_exact_features_before_decoding():
    policy, encoder, view, _, provider = fixture()
    policy.eval()
    features = provider.get_context(view)
    before = len(encoder.model.calls)
    with pytest.raises(TypeError, match="cannot be reused"):
        policy.generate(view, features, steps=3)
    assert len(encoder.model.calls) == before


@pytest.mark.parametrize("path", ["forward", "reconstruction", "query", "generation", "whole_core"])
def test_foreign_encoder_identity_rejected_before_extraction(path):
    policy, encoder, view, truth, provider = fixture()
    policy.encoder_identity = {**IDENTITY, "checkpoint_sha256": "c" * 64}
    policy.eval()
    with pytest.raises(ValueError, match="identity"):
        if path == "forward":
            policy.forward_views([view], [provider])
        elif path == "reconstruction":
            policy.reconstruction_nll([view], [provider], [truth])
        elif path == "query":
            policy.queried_log_probs([view], [provider], [("H", 1)], ["C"])
        elif path == "generation":
            policy.generate(view, provider)
        else:
            whole_core_scores(policy, provider, view, [truth], view.hidden_sites())
    assert not encoder.model.calls


def test_invalid_later_batch_member_rejected_before_first_extraction():
    policy, encoder, view, truth, provider = fixture()
    wrong = VisibleView(view.roles, (("Y", None, "D", "E"), view.chain("L")))
    with pytest.raises(ValueError, match="scaffold"):
        policy.reconstruction_nll([view, wrong], [provider, provider], [truth, truth])
    assert not encoder.model.calls


def test_native_and_joint_length_limits_precede_extraction():
    policy, encoder, view, truth, provider = fixture()
    encoder.max_tokens = 3
    with pytest.raises(ValueError, match="joint sequence"):
        policy.forward_views([view], [provider])
    assert not encoder.model.calls
    encoder.max_tokens = 1024
    long, target = mask_complete({"H": "A" * 65}, [("H", 0)])
    long_provider = DynamicContextProvider(encoder, long)
    with pytest.raises(ValueError, match="native position"):
        whole_core_scores(policy, long_provider, long, [target], target.coords())
    assert not encoder.model.calls


def test_exact_features_reject_stale_reveal_and_foreign_antigen():
    _, _, view, _, provider = fixture()
    context = provider.get_context(view)
    with pytest.raises(ValueError, match="content mismatch"):
        context.check_query(view.reveal(("H", 1), "C"))
    with pytest.raises(ValueError, match="antigen"):
        context.check_query(VisibleView(view.roles + ("Ag",), view.chains + (("Y",),)))


def test_inference_extraction_can_be_reused_for_training():
    policy, _, view, truth, provider = fixture()
    with torch.inference_mode():
        exact = provider.get_context(view)
    assert not torch.is_inference(exact._features)
    policy.reconstruction_nll([view], [exact], [truth]).backward()
    assert torch.count_nonzero(policy.up.weight.grad) > 0


def test_hidden_gold_never_changes_encoder_input():
    _, encoder, view, _, provider = fixture()
    provider.get_context(view)
    alternate, _ = mask_complete({"H": "AYDE", "L": "FG", "Ag": "HIK"}, [("H", 1)])
    DynamicContextProvider(encoder, alternate).get_context(alternate.without_ag())
    torch.testing.assert_close(encoder.model.calls[0][0], encoder.model.calls[1][0], rtol=0, atol=0)


def test_frozen_encoder_fp32_is_immune_to_ambient_autocast():
    _, _, view, _, provider = fixture()
    expected = provider.get_context(view).features
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = provider.get_context(view).features
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32


def test_full_mode_unfreezes_native_and_rejects_actual_half_storage():
    cfg = DPLMConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=32)
    native = NativeDPLM.random_init(cfg)
    native.model.freeze_base()
    policy = FusedDPLM(native, IDENTITY, FusionConfig(context_dim=8, update_mode="full"))
    assert all(p.requires_grad for p in policy.backbone.parameters())
    half = NativeDPLM.random_init(cfg)
    half.model.half()
    with pytest.raises(ValueError, match="float32|FP32"):
        FusedDPLM(half, IDENTITY, FusionConfig(context_dim=8, update_mode="full"))


def test_encoder_bf16_rejects_cpu_before_freezing_parameters():
    native = Encoder()
    with pytest.raises(ValueError, match="native CUDA BF16"):
        FrozenESMPPI(native, {**IDENTITY, "precision": "bfloat16"}, compute_dtype="bfloat16")
    assert all(p.requires_grad for p in native.parameters())


@pytest.mark.skipif(not native_bf16_supported("cuda"), reason="native CUDA BF16 required")
def test_encoder_bf16_compute_and_backward_with_fp32_storage():
    torch.manual_seed(7)
    native = Encoder().cuda()
    reference = FrozenESMPPI(native, IDENTITY)
    view, truth = mask_complete({"H": "ACDE", "Ag": "FGH"}, [("H", 1)])
    expected = reference.extract_exact(view).features
    native2 = Encoder().cuda()
    native2.load_state_dict(native.state_dict())
    encoder = FrozenESMPPI(native2, {**IDENTITY, "precision": "bfloat16"}, compute_dtype="bfloat16")
    actual = encoder.extract_exact(view)
    torch.testing.assert_close(actual.features.float(), expected, rtol=.03, atol=.03)
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in encoder.parameters())
    policy, _, ab, _, _ = fixture()
    policy.encoder_identity = dict(encoder.encoder_identity)
    policy.cuda().train()
    provider = DynamicContextProvider(encoder, view)
    policy.reconstruction_nll([view.without_ag()], [provider], [truth]).backward()
    assert torch.isfinite(policy.up.weight.grad).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.1"])
def test_invalid_mask_weights_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        UniformMaskSpec(span_weight=bad)


def test_duplicate_disabled_region_rejected_without_advancing_rng():
    rng = random.Random(5)
    before = rng.getstate()
    with pytest.raises(ValueError, match="duplicate"):
        draw_mask_uniform((("H", 0),), rng,
                          UniformMaskSpec(span_region=(("H", 0), ("H", 0))))
    assert rng.getstate() == before


def test_actual_sampler_law_matches_random_order_loss():
    union = (("H", 0), ("H", 1), ("H", 2))

    class Choice:
        def __init__(self, subset):
            self.subset = subset

        def random(self):
            return .5

        def randint(self, low, high):
            assert (low, high) == (1, len(union))
            return len(self.subset)

        def sample(self, population, count):
            assert tuple(population) == union and count == len(self.subset)
            return list(self.subset)

    def site_loss(site, hidden):
        # Depends jointly on target site and the entire current mask, not just K.
        code = sum(2 ** coord[1] for coord in hidden)
        return ((13 * code + 7 * site[1]) % 23 + 1) / 7

    sampled = 0.0
    for k in range(1, len(union) + 1):
        for subset in itertools.combinations(union, k):
            draw = draw_mask_uniform(union, Choice(subset))
            assert draw.kind == "uniform_k" and draw.count == k
            sampled += sum(site_loss(site, draw.sites) for site in draw.sites) / (
                k * len(union) * math.comb(len(union), k))
    ordered = 0.0
    for order in itertools.permutations(union):
        ordered += sum(site_loss(site, order[i:]) for i, site in enumerate(order)) / (
            len(union) * math.factorial(len(union)))
    assert sampled == pytest.approx(ordered, abs=1e-12)
