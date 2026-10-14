"""Integration contracts on the actual native tiny DPLM implementation."""
import copy
from dataclasses import replace

import pytest
import torch

from smallAntibodyGen.models.dplm_policy import DPLMConfig, NativeDPLM, DPLM_MASK_ID
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig, serialize_antibody
from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext
from smallAntibodyGen.models.antibody_antigen.denoising import masked_nll, sequential_log_prob
from smallAntibodyGen.models.antibody_antigen.views import mask_complete, VisibleView


IDENTITY = {"source_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "extraction_layer": "layer_2", "precision": "float32", "tokenizer": "ESM-1b",
            "eval_mode": "true"}


@pytest.fixture(autouse=True)
def small_cpu_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup_policy(*, checkpointing=True, dropout=0.0):
    torch.manual_seed(7)
    config = DPLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, hidden_dropout_prob=dropout, max_position_embeddings=64)
    native = NativeDPLM.random_init(config, seed=9, std=0.1)
    policy = FusedDPLM(native, IDENTITY, FusionConfig(context_dim=8, bottleneck=6,
                                                    lora_rank=2, lora_alpha=4,
                                                    activation_checkpointing=checkpointing))
    view, targets = mask_complete({"H": "ACDEFG", "L": "HIK", "Ag": "LMNP"},
                                  [("H", 2), ("H", 3)])
    context = StaticContext(view, torch.randn(len(view.coords()), 8), IDENTITY)
    return policy, context, view.without_ag(), targets


def test_native_layout_and_zero_residual_parity():
    policy, context, view, _ = setup_policy()
    policy.eval()
    tokens, mapping = serialize_antibody(view)
    assert mapping[("H", 0)] == 1
    assert mapping[("L", 0)] == 10  # H length 6 + four native boundary tokens
    assert len(tokens) == 14
    h_tokens, h_map = serialize_antibody(view.select_roles(["H"]))
    assert len(h_tokens) == 9 and h_map[("H", 5)] == 6
    ids, _, _, _ = policy.prepare([view], [context])
    actual, _, _ = policy.forward_views([view], [context])
    expected = policy.backbone(ids)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_eval_native_construction_synchronizes_training_mode():
    native = NativeDPLM.random_init(DPLMConfig(hidden_size=16, num_hidden_layers=2,
                                    num_attention_heads=2, intermediate_size=32), seed=9)
    native.model.eval()
    policy = FusedDPLM(native, IDENTITY, FusionConfig(context_dim=8, bottleneck=6, lora_rank=2))
    assert policy.training and policy.backbone.training
    assert all(module.training for module in policy.modules())
    policy.eval()
    assert all(not module.training for module in policy.modules())


def test_native_fusion_fp32_ignores_ambient_autocast():
    policy, context, view, _ = setup_policy()
    policy.eval()
    with torch.no_grad():
        policy.up.weight.normal_(std=0.1)
    reference = policy.bind(context).forward(view).raw_logits
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = policy.bind(context).forward(view).raw_logits
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_residual_gradients_and_frozen_base():
    policy, context, view, truth = setup_policy()
    opt = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.1)
    policy.reconstruction_nll([view], [context], [truth]).backward()
    assert policy.up.weight.grad.abs().sum() > 0
    assert policy.down.weight.grad.abs().sum() == 0
    assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    opt.step(); opt.zero_grad()
    policy.reconstruction_nll([view], [context], [truth]).backward()
    assert policy.down.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in policy.named_parameters() if "lora_B" in n)


def test_whole_window_shared_residues_and_batch_gradient_parity():
    policy, context, view, a = setup_policy(checkpointing=False)
    _, b = mask_complete({"H": "ACDFFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    # Endpoints share H:2 but differ at H:3; both belong in the whole-window score.
    order = tuple(reversed(view.hidden_sites()))
    reference = sequential_log_prob(policy.bind(context), view, a, order).log_prob / 2
    states, coords, residues = [], [], []
    current = view
    for coord in order:
        states.append(current); coords.append(coord); residues.append(a.residue(coord))
        current = current.reveal(coord, a.residue(coord))
    actual = policy.queried_log_probs(states, [context] * 2, coords, residues).mean()
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    params = [p for p in policy.parameters() if p.requires_grad]
    ga = torch.autograd.grad(actual, params, retain_graph=True)
    gr = torch.autograd.grad(reference, params)
    for left, right in zip(ga, gr):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(sequential_log_prob(policy.bind(context), view, b, order).log_prob)


def test_reconstruction_batch_matches_literal_and_checkpoint_dropout_gradients():
    policy, context, view, truth = setup_policy(checkpointing=True, dropout=0.1)
    reference = copy.deepcopy(policy)
    reference.config = replace(reference.config, activation_checkpointing=False)
    torch.manual_seed(100)
    loss = policy.reconstruction_nll([view], [context], [truth])
    loss.backward()
    torch.manual_seed(100)
    expected = masked_nll(reference.bind(context), view, truth).total / len(truth)
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for (name, p), (_, q) in zip(policy.named_parameters(), reference.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_context_validation_precedes_backbone_and_queries_are_hidden(monkeypatch):
    policy, context, view, truth = setup_policy()
    def forbidden(*args, **kwargs):
        raise AssertionError("backbone must not be called")
    monkeypatch.setattr(policy.backbone, "forward", forbidden)
    bad = VisibleView(view.roles, ((None,) + view.chains[0][1:], view.chains[1]))
    with pytest.raises(ValueError, match="hidden union"):
        policy.forward_views([bad], [context])
    with pytest.raises(ValueError, match="remain masked"):
        policy.queried_log_probs([view], [context], [("H", 0)], ["A"])
    with pytest.raises(ValueError, match="exactly cover"):
        policy.reconstruction_nll([view], [context], [type(truth).from_mapping({("H", 2): "D"})])


def test_sampling_calls_fusion_and_preserves_scaffold(monkeypatch):
    policy, context, view, _ = setup_policy()
    policy.eval()
    calls = []
    def forced(ids, features, residue_mask):
        calls.append(ids.clone())
        logits = torch.full((*ids.shape, 33), -10000.0)
        logits[..., 5] = 10000  # native canonical A
        return logits, torch.zeros((*ids.shape, 16))
    monkeypatch.setattr(policy, "_forward_tensors", forced)
    result = policy.generate(view, context, steps=3, generator=torch.Generator().manual_seed(2))
    assert len(calls) == 3 and not result.hidden_sites()
    assert all(result.token(c) == "A" for c in view.hidden_sites())
    assert all(result.token(c) == view.token(c) for c in view.coords() if c not in view.hidden_sites())


def test_context_changes_outputs_after_nonzero_residual_and_state_roundtrip():
    policy, context, view, _ = setup_policy()
    policy.eval()
    with torch.no_grad():
        policy.up.weight.normal_(std=0.1)
    other_view = VisibleView(context.view.roles, context.view.chains[:-1] + (tuple("AAAA"),))
    other = StaticContext(other_view, context.features + 0.5, IDENTITY)
    a = policy.bind(context).forward(view).raw_logits
    b = policy.bind(other).forward(view).raw_logits
    assert not torch.allclose(a, b)
    state = policy.trainable_state()
    replica, _, _, _ = setup_policy()
    replica.load_trainable_state(state); replica.eval()
    torch.testing.assert_close(replica.bind(context).forward(view).raw_logits, a)
    bad = dict(state); bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="inventory"):
        replica.load_trainable_state(bad)
    state[next(iter(state))] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="invalid trainable"):
        replica.load_trainable_state(state)
