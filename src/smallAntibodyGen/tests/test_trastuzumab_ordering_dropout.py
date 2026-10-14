"""Ordering fits deterministic evaluation scores without disabling checkpointing."""
import copy
from dataclasses import replace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc.factory import build_models
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import ordering_loss, whole_core_scores
from smallAntibodyGen.experiments.trastuzumab_poc.runner import deterministic_ordering
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.antibody_antigen.views import mask_complete
from smallAntibodyGen.models.dplm_policy import DPLMConfig, NativeDPLM


def models(mode):
    _, encoder, _ = build_models({"kind": "tiny"}, FusionConfig(
        context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4))
    native = NativeDPLM.random_init(DPLMConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=0.4), seed=4)
    policy = FusedDPLM(native, encoder.encoder_identity, FusionConfig(
        context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4,
        lora_dropout=0.3, update_mode=mode, activation_checkpointing=True))
    # Exercise every path beyond its zero-output initialization.
    with torch.no_grad():
        policy.up.weight.normal_(0, 0.01)
        for module in policy.backbone.lora_modules().values():
            module.lora_B.normal_(0, 0.01)
    return policy, encoder


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_ordering_checkpoint_recomputation_preserves_deterministic_scores_and_gradients(mode):
    checkpointed, encoder = models(mode)
    plain = copy.deepcopy(checkpointed)
    plain.config = replace(plain.config, activation_checkpointing=False)
    sites = tuple(("H", i) for i in range(1, 11))
    view, best = mask_complete({"H": "AACDEFGHIKLA", "L": "MNP", "Ag": "QRST"}, sites)
    _, worst = mask_complete({"H": "AACDEYGHIKLA", "L": "MNP", "Ag": "QRST"}, sites)
    results = []
    for policy in (checkpointed, plain):
        policy.train()
        dropout = [m for m in policy.modules() if isinstance(m, torch.nn.Dropout)]
        assert dropout and all(m.training for m in dropout)
        calls, encoder_calls = [], []

        def observe(module, args):
            calls.append((policy.training, policy.backbone.training,
                          all(not m.training for m in dropout)))

        handles = [policy.backbone.register_forward_pre_hook(observe),
                   encoder.model.register_forward_pre_hook(lambda module, args: encoder_calls.append(1))]
        rng_before = torch.random.get_rng_state().clone()
        try:
            with deterministic_ordering(policy):
                scores = whole_core_scores(policy, DynamicContextProvider(encoder, view),
                                           view.without_ag(), [best, worst], sites,
                                           state_microbatch=1)
                ordering_loss(scores, [(0, 1)], 0.1).backward()
        finally:
            for handle in handles:
                handle.remove()
        assert all(root and backbone and disabled for root, backbone, disabled in calls)
        assert all(m.training for m in dropout)
        assert torch.equal(rng_before, torch.random.get_rng_state())
        # Frozen extraction happens once per state, never in checkpoint backward.
        assert len(encoder_calls) == 20
        assert not encoder.training
        assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
        results.append((scores.detach(), len(calls)))

    assert [result[1] for result in results] == [40, 20]
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    plain_parameters = dict(plain.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        if parameter.requires_grad:
            reference = plain_parameters[name]
            assert parameter.grad is not None and reference.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("fail_backward", [False, True])
def test_ordering_restores_each_dropout_flag_even_after_backward_failure(mode, fail_backward):
    policy, _ = models(mode)
    policy.train()
    dropout = [m for m in policy.modules() if isinstance(m, torch.nn.Dropout)]
    for index, module in enumerate(dropout):
        module.training = bool(index % 2)
    before = [module.training for module in policy.modules()]

    class CheckedBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            assert policy.training and policy.backbone.training
            assert all(not module.training for module in dropout)
            if fail_backward:
                raise RuntimeError("synthetic backward failure")
            return gradient

    rng_before = torch.random.get_rng_state().clone()
    def run():
        with deterministic_ordering(policy):
            assert policy.training and policy.backbone.training
            assert all(not module.training for module in dropout)
            CheckedBackward.apply(policy.up.weight.sum()).backward()

    if fail_backward:
        with pytest.raises(RuntimeError, match="synthetic backward failure"):
            run()
    else:
        run()
    assert [module.training for module in policy.modules()] == before
    assert torch.equal(rng_before, torch.random.get_rng_state())
