"""Standalone functional head, antibody pooling and pair loss on a real tiny FusedDPLM (M1)."""
from __future__ import annotations

import copy
import random
from dataclasses import replace

import pytest
import torch
import torch.utils.checkpoint

from smallAntibodyGen.experiments.trastuzumab_poc import functional_data, functional_protocol
from smallAntibodyGen.experiments.trastuzumab_poc import functional_train as ft
from smallAntibodyGen.experiments.trastuzumab_poc.functional_data import FunctionalConstruct
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import isolated_torch_rng, keyed_seed
from smallAntibodyGen.experiments.trastuzumab_poc.runner import deterministic_ordering
from smallAntibodyGen.models.antibody_antigen.dynamic_context import DynamicContextProvider
from smallAntibodyGen.models.antibody_antigen.fusion import serialize_antibody
from smallAntibodyGen.models.antibody_antigen.views import VisibleView, mask_complete
from smallAntibodyGen.tests.test_trastuzumab_functional_data import AG, LIGHT, heavy, imported_names
from smallAntibodyGen.tests.test_trastuzumab_ordering_dropout import models

MODES = ["lora", "full"]


def construct(construct_id, H, L=LIGHT, Ag=AG):
    return FunctionalConstruct(
        construct_id=construct_id, source_id="func", source_record_id=f"rec-{construct_id}", H=H,
        L=L, Ag=Ag, antibody_format="paired" if L is not None else "heavy_only",
        antibody_family="famF", target_family="her2", partition="train",
        evidence_reference="doi:10.0/test")


class RecordingConstruct:
    """Exposes only what is read; every attribute access is recorded."""

    def __init__(self, **values):
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "accessed", set())

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name not in values:
            raise AttributeError(name)
        object.__getattribute__(self, "accessed").add(name)
        return values[name]


def config(**overrides):
    return ft.FunctionalConfig(**{"weight": 2.0, "pairs_per_update": 4, "margin": 100.0, **overrides})


def fixed_head(values):
    head = ft.ObservationHead(len(values))
    with torch.no_grad():
        head.weight.copy_(torch.tensor([values], dtype=torch.float32))
    return head


def spy_on_forward(policy):
    """Record (views, contexts) of every forward_views call; returns the record list."""
    calls = []
    original = policy.forward_views

    def spying(views, contexts):
        calls.append((views, contexts))
        return original(views, contexts)

    policy.forward_views = spying
    return calls


def test_new_functional_modules_never_import_the_runner():
    for module in (functional_data, functional_protocol, ft):
        names = imported_names(module)
        assert not any(name.endswith(".runner") or name.endswith(".runner.deterministic_ordering")
                       for name in names)


# ---------------------------------------------------------------------------
# Config, head, loss: pure arithmetic with hand-computed answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(gradient_mode="partial"), dict(weight=0.0), dict(weight=-1.0), dict(weight=float("nan")),
    dict(weight=True), dict(weight="1"), dict(pairs_per_update=0), dict(pairs_per_update=True),
    dict(pairs_per_update=2.0), dict(margin=0.0), dict(margin=float("inf")), dict(head_lr_scale=0.0),
    dict(head_weight_decay=-0.1), dict(head_gradient_clip=0.0), dict(minimum_compared_fraction=1.5),
    dict(minimum_compared_fraction=-0.1)])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ft.FunctionalConfig(**kwargs)


def test_gradient_mode_only_selects_policy_gradient():
    assert ft.FunctionalConfig().policy_receives_gradient is True
    assert ft.FunctionalConfig(gradient_mode="detached").policy_receives_gradient is False


def test_head_is_bias_free_fp32_and_drawn_from_the_keyed_isolated_stream():
    head = ft.new_head(16, 7)
    assert [name for name, _ in head.named_parameters()] == ["weight"]
    assert head.bias is None and head.weight.shape == (1, 16) and head.weight.dtype == torch.float32
    with isolated_torch_rng(keyed_seed(7, 0, 0, "functional-head-init")):
        expected = torch.empty(1, 16, dtype=torch.float32).normal_(0.0, 16 ** -0.5)
    assert torch.equal(head.weight.detach(), expected)


def test_head_weights_follow_the_declared_distribution():
    weights = ft.new_head(4096, 3).weight.detach()
    scale = 4096 ** -0.5
    assert abs(weights.std().item() / scale - 1) < 0.1
    assert abs(weights.mean().item()) < 5 * scale / 4096 ** 0.5


def test_head_initialization_is_deterministic_and_leaves_external_rng_alone():
    random.seed(21)
    torch.manual_seed(22)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    first, again, other = ft.new_head(16, 5), ft.new_head(16, 5), ft.new_head(16, 6)
    assert torch.equal(first.weight, again.weight) and not torch.equal(first.weight, other.weight)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_head_weights_do_not_depend_on_the_device():
    assert torch.equal(ft.new_head(16, 5, "cuda").weight.detach().cpu(), ft.new_head(16, 5).weight.detach())


def test_head_rejects_wrong_width_or_dtype():
    head = ft.ObservationHead(4)
    for bad in (torch.zeros(4, dtype=torch.float64), torch.zeros(3)):
        with pytest.raises(ValueError):
            head(bad)
    with pytest.raises(ValueError):
        ft.ObservationHead(0)


BETTER = [0.5, 0.1, 0.2, 0.9]
WORSE = [0.1, 0.3, 0.2, 0.4]
WEIGHTS = [1.0, -2.0, 0.5, 0.0]
# gap = 1*0.4 + (-2)*(-0.2) + 0.5*0.0 + 0*0.5 = 0.8 ; difference of features = [0.4, -0.2, 0.0, 0.5]


def test_pair_loss_known_answer_for_an_active_hinge():
    head = fixed_head(WEIGHTS)
    loss, active = ft.pair_loss(head, torch.tensor(BETTER), torch.tensor(WORSE),
                                ft.FunctionalConfig(weight=3.0, pairs_per_update=4, margin=1.0))
    assert active is True
    # hinge = (1.0 - 0.8)^2 = 0.04, scaled by weight / pairs_per_update = 3 / 4
    assert loss.item() == pytest.approx(0.04 * 3.0 / 4.0, abs=1e-6)
    loss.backward()
    # d loss / d w = -(2 * 3 / 4) * (1.0 - 0.8) * (better - worse) = -0.3 * [0.4, -0.2, 0.0, 0.5]
    torch.testing.assert_close(head.weight.grad[0], torch.tensor([-0.12, 0.06, 0.0, -0.15]),
                               rtol=0, atol=1e-6)


def test_pair_loss_known_answer_for_an_inactive_hinge():
    head = fixed_head(WEIGHTS)
    loss, active = ft.pair_loss(head, torch.tensor(BETTER), torch.tensor(WORSE),
                                ft.FunctionalConfig(weight=3.0, pairs_per_update=4, margin=0.5))
    assert active is False and loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(head.weight.grad) == 0


def test_pair_loss_scales_with_weight_over_pairs_and_penalizes_a_reversed_pair():
    head = fixed_head(WEIGHTS)
    better, worse = torch.tensor(BETTER), torch.tensor(WORSE)
    one, _ = ft.pair_loss(head, better, worse, ft.FunctionalConfig(weight=2.0, pairs_per_update=1, margin=1.0))
    four, _ = ft.pair_loss(head, better, worse, ft.FunctionalConfig(weight=2.0, pairs_per_update=4, margin=1.0))
    assert one.item() == pytest.approx(4 * four.item(), rel=1e-6)
    reversed_loss, active = ft.pair_loss(head, worse, better,
                                         ft.FunctionalConfig(weight=2.0, pairs_per_update=1, margin=1.0))
    assert active and reversed_loss.item() == pytest.approx(2.0 * (1.0 + 0.8) ** 2, rel=1e-5)


def test_reversing_a_pair_with_both_hinges_active_opposes_the_gradient_with_a_different_magnitude():
    better, worse = torch.tensor(BETTER), torch.tensor(WORSE)
    settings = ft.FunctionalConfig(weight=3.0, pairs_per_update=4, margin=1.0)
    gradients = {}
    for name, (first, second) in {"forward": (better, worse), "reversed": (worse, better)}.items():
        head = fixed_head(WEIGHTS)
        loss, active = ft.pair_loss(head, first, second, settings)
        assert active                 # gap +0.8 leaves 1 - 0.8 = 0.2; gap -0.8 leaves 1 + 0.8 = 1.8
        loss.backward()
        gradients[name] = head.weight.grad[0].clone()
    # d loss / d w = -(2 * 3 / 4) * hinge * (first - second) with hinge 0.2 and 1.8 respectively.
    torch.testing.assert_close(gradients["forward"], torch.tensor([-0.12, 0.06, 0.0, -0.15]),
                               rtol=0, atol=1e-6)
    torch.testing.assert_close(gradients["reversed"], torch.tensor([1.08, -0.54, 0.0, 1.35]),
                               rtol=0, atol=1e-5)
    cosine = torch.nn.functional.cosine_similarity(gradients["forward"], gradients["reversed"], dim=0)
    assert cosine.item() == pytest.approx(-1.0, abs=1e-5)
    assert (gradients["reversed"].norm() / gradients["forward"].norm()).item() == pytest.approx(9.0, rel=1e-4)


AUTOCAST_CASES = [
    ("cpu", torch.bfloat16),
    pytest.param("cuda", torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
]


@pytest.mark.parametrize("device, low", AUTOCAST_CASES)
def test_head_scores_loss_and_gradients_stay_fp32_under_an_ambient_low_precision_autocast(device, low):
    settings = ft.FunctionalConfig(weight=3.0, pairs_per_update=4, margin=1.0)

    def forward():
        head = fixed_head(WEIGHTS).to(device)
        better = torch.tensor(BETTER, device=device, requires_grad=True)
        worse = torch.tensor(WORSE, device=device, requires_grad=True)
        scores = head(torch.stack([better, worse]))
        loss, active = ft.pair_loss(head, better, worse, settings)
        return head, better, worse, scores, loss, active

    def results(run):
        head, better, worse, scores, loss, active = run
        loss.backward()                       # outside autocast, as a trainer would
        return scores.detach(), loss.detach(), active, head.weight.grad, better.grad, worse.grad

    reference = results(forward())
    with torch.autocast(device, dtype=low):
        # The ambient autocast is real: a bare linear call is low precision here.
        probe = torch.nn.functional.linear(torch.ones(1, 4, device=device), torch.ones(1, 4, device=device))
        if probe.dtype != low:
            pytest.skip(f"ambient {device} {low} autocast is not active on this build")
        ambient = forward()
    assert ambient[3].dtype == ambient[4].dtype == torch.float32      # scores and loss
    ambient = results(ambient)
    assert reference[0].dtype == reference[1].dtype == torch.float32
    assert ambient[2] is True and reference[2] is True
    for value in (ambient[0], ambient[1], *ambient[3:]):
        assert value.dtype == torch.float32
    for left, right in zip((ambient[0], ambient[1], *ambient[3:]), (reference[0], reference[1], *reference[3:])):
        assert torch.equal(left, right)


def test_pair_loss_validates_its_inputs():
    head, good = fixed_head(WEIGHTS), torch.tensor(BETTER)
    for bad in (good.double(), good[:3], torch.stack([good, good]),
                torch.tensor([0.1, float("nan"), 0.2, 0.3]), BETTER):
        with pytest.raises(ValueError):
            ft.pair_loss(head, bad, good, ft.FunctionalConfig())
    with pytest.raises(TypeError):
        ft.pair_loss("head", good, good, ft.FunctionalConfig())
    with pytest.raises(TypeError):
        ft.pair_loss(head, good, good, "config")


def test_head_optimizer_and_gradient_clip_follow_the_config():
    head = fixed_head([0.0, 0.0])
    settings = ft.FunctionalConfig(head_lr_scale=0.5, head_weight_decay=0.02, head_gradient_clip=1.0)
    optimizer = ft.head_optimizer(head, settings, 0.01)
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (pytest.approx(0.005), pytest.approx(0.02))
    assert group["foreach"] is False and group["capturable"] is False
    head.weight.grad = torch.tensor([[3.0, 4.0]])
    assert ft.clip_head_gradient(head, settings) == pytest.approx(5.0)
    assert head.weight.grad.norm().item() == pytest.approx(1.0, rel=1e-3)
    with pytest.raises(ValueError):
        ft.head_optimizer(head, settings, 0.0)


# ---------------------------------------------------------------------------
# Pooling and information flow on the real tiny model
# ---------------------------------------------------------------------------

def pooled_reference(policy, encoder, item, arm):
    """Pooled feature computed from first principles, plus the layout facts."""
    ab = ft.complete_view(item, arm).without_ag()
    tokens, mapping = serialize_antibody(ab)
    provider = DynamicContextProvider(encoder, ft.complete_view(item, arm))
    _, hidden, mappings = policy.forward_views([ab], [provider])
    assert mappings[0] == mapping
    slots = [mapping[coord] for coord in ab.coords()]
    return hidden[0, slots].mean(dim=0), slots, tokens


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("H, L, specials", [(heavy(0), LIGHT, 5), (heavy(1) + "AAAAA", "DIQMT", 5),
                                            (heavy(2), None, 3)],
                         ids=["paired", "unequal_lengths", "heavy_only"])
def test_pooling_covers_exactly_the_antibody_residue_slots(mode, H, L, specials):
    policy, encoder = models(mode)
    policy.train()
    item = construct("c", H, L)
    with deterministic_ordering(policy):
        expected, slots, tokens = pooled_reference(policy, encoder, item, "paired")
        feature = ft.endpoint_features(policy, encoder, item, "paired", grad=False)
    assert len(slots) == len(H) + len(L or "") and len(set(slots)) == len(slots)
    assert len(tokens) - len(slots) == specials          # CLS/EOS tokens are never pooled
    assert feature.dtype == torch.float32 and feature.shape == (policy.up.out_features,)
    torch.testing.assert_close(feature, expected, rtol=0, atol=1e-6)


def test_special_padding_and_antigen_rows_cannot_change_the_feature():
    policy, encoder = models("lora")
    item = construct("c", heavy(0))
    with deterministic_ordering(policy):
        reference = ft.endpoint_features(policy, encoder, item, "paired", grad=False)
        original = policy.forward_views

        def perturbed(views, contexts):
            logits, hidden, mappings = original(views, contexts)
            noisy = torch.full_like(hidden, 1.0e6)
            for slot in mappings[0].values():
                noisy[:, slot] = hidden[:, slot]
            padding = torch.full((hidden.shape[0], 5, hidden.shape[2]), -1.0e6)
            return logits, torch.cat([noisy, padding], dim=1), mappings

        policy.forward_views = perturbed
        changed = ft.endpoint_features(policy, encoder, item, "paired", grad=False)
    torch.testing.assert_close(changed, reference, rtol=0, atol=1e-6)


def test_pooling_indexes_the_mapping_strictly_and_never_for_the_antigen():
    class StrictMapping(dict):
        def __init__(self, base):
            super().__init__(base)
            self.looked_up = []

        def __getitem__(self, key):
            self.looked_up.append(key)
            return super().__getitem__(key)

        def get(self, *args, **kwargs):
            raise AssertionError("pooling must index the mapping strictly")

    policy, encoder = models("lora")
    item = construct("c", heavy(0))
    spies, original = [], policy.forward_views

    def spying(views, contexts):
        logits, hidden, mappings = original(views, contexts)
        spies.append(StrictMapping(mappings[0]))
        return logits, hidden, [spies[-1]]

    policy.forward_views = spying
    with deterministic_ordering(policy):
        ft.endpoint_features(policy, encoder, item, "paired", grad=False)
    ab = ft.complete_view(item, "paired").without_ag()
    assert len(spies) == 1 and spies[0].looked_up == list(ab.coords())
    assert all(role != "Ag" for role, _ in spies[0].looked_up)


@pytest.mark.parametrize("arm", ["paired", "abonly"])
def test_only_sequences_reach_the_model_and_ids_groups_and_outcomes_do_not(arm):
    policy, encoder = models("lora")
    calls = spy_on_forward(policy)
    first = RecordingConstruct(construct_id="SECRET-A", group_id="g-secret", outcome=1.0,
                               H=heavy(0), L=LIGHT, Ag=AG)
    second = RecordingConstruct(construct_id="SECRET-B", group_id="g-other", outcome=-9.0,
                                H=heavy(0), L=LIGHT, Ag=AG)
    with deterministic_ordering(policy):
        feature_a = ft.endpoint_features(policy, encoder, first, arm, grad=False)
        feature_b = ft.endpoint_features(policy, encoder, second, arm, grad=False)
    assert first.accessed == {"H", "L", "Ag"} if arm == "paired" else first.accessed == {"H", "L"}
    assert second.accessed == first.accessed
    torch.testing.assert_close(feature_a, feature_b, rtol=0, atol=1e-6)
    for views, contexts in calls:
        assert len(views) == len(contexts) == 1
        assert isinstance(views[0], VisibleView) and views[0].roles == ("H", "L")
        assert views[0].chain("H") == tuple(heavy(0)) and views[0].chain("L") == tuple(LIGHT)
        assert isinstance(contexts[0], DynamicContextProvider) and "SECRET" not in repr(views)


def test_paired_and_abonly_arms_differ_only_through_the_antigen_context():
    policy, encoder = models("lora")
    item = construct("c", heavy(0))
    with deterministic_ordering(policy):
        paired = ft.endpoint_features(policy, encoder, item, "paired", grad=False)
        abonly = ft.endpoint_features(policy, encoder, item, "abonly", grad=False)
    assert not torch.equal(paired, abonly)
    with pytest.raises(ValueError):
        ft.complete_view(item, "both")


def test_every_endpoint_gets_a_fresh_complete_view_provider():
    policy, encoder = models("lora")
    calls = spy_on_forward(policy)
    item = construct("c", heavy(0))
    with deterministic_ordering(policy):
        for _ in range(3):
            ft.endpoint_features(policy, encoder, item, "paired", grad=False)
    providers = [contexts[0] for _, contexts in calls]
    assert len({id(provider) for provider in providers}) == 3
    assert all(provider.hidden_union == () for provider in providers)


def test_complete_and_masked_providers_cannot_be_shared():
    policy, encoder = models("lora")
    item = construct("c", heavy(0))
    complete = ft.complete_view(item, "paired")
    masked, _ = mask_complete({"H": item.H, "L": item.L, "Ag": item.Ag}, [("H", 3)])
    complete_provider = DynamicContextProvider(encoder, complete)
    masked_provider = DynamicContextProvider(encoder, masked)
    with deterministic_ordering(policy):
        with pytest.raises(ValueError, match="complete view"):
            ft.antibody_features(policy, masked_provider, complete)
        with pytest.raises(ValueError, match="complete view"):
            ft.antibody_features(policy, complete_provider, masked)
        with pytest.raises(ValueError):          # the policy itself refuses the reverse reuse
            policy.preflight([masked.without_ag()], [complete_provider])
    with pytest.raises(TypeError):
        ft.antibody_features(policy, "provider", complete)


def test_preflight_validates_the_complete_paired_and_abonly_views_of_every_construct():
    policy, encoder = models("lora")
    admitted = [construct("ok", heavy(0)), construct("ok2", heavy(1), L=None)]
    assert ft.preflight_complete_views(policy, encoder, admitted) == 4
    long_antigen = construct("long-ag", heavy(0), Ag="A" * 200)       # 10 + 10 + 200 + 2 > 128
    with pytest.raises(ValueError, match=r"long-ag.*context limit"):
        ft.preflight_complete_views(policy, encoder, admitted + [long_antigen])
    # The ab-only view alone would pass, yet the construct is rejected: the admitted set is
    # arm-independent.
    abonly = ft.complete_view(long_antigen, "abonly")
    policy.preflight([abonly.without_ag()], [DynamicContextProvider(encoder, abonly)])
    long_heavy = construct("long-h", heavy(0) + "A" * 60)             # 70 + 10 + 5 > 64 native slots
    with pytest.raises(ValueError, match=r"long-h.*native position limit"):
        ft.preflight_complete_views(policy, encoder, [long_heavy])


@pytest.mark.parametrize("mode", MODES)
def test_features_need_disabled_dropout_and_preserve_external_rng(mode):
    policy, encoder = models(mode)
    policy.train()
    item = construct("c", heavy(0))
    dropout = [m for m in policy.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropout and all(m.training for m in dropout)
    with pytest.raises(RuntimeError, match="deterministic_ordering"):
        ft.endpoint_features(policy, encoder, item, "paired", grad=False)
    random.seed(1)
    torch.manual_seed(2)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    with deterministic_ordering(policy):
        ft.endpoint_features(policy, encoder, item, "paired", grad=True)
        ft.endpoint_features(policy, encoder, item, "abonly", grad=False)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    assert all(m.training for m in dropout)
    with pytest.raises(TypeError):
        ft.endpoint_features(policy, encoder, item, "paired", grad=1)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

BETTER_CONSTRUCT, WORSE_CONSTRUCT = construct("better", heavy(0)), construct("worse", heavy(1))


def run_pair(policy, encoder, head, settings, *, grad, backward=True):
    with deterministic_ordering(policy):
        better = ft.endpoint_features(policy, encoder, BETTER_CONSTRUCT, "paired", grad=grad)
        worse = ft.endpoint_features(policy, encoder, WORSE_CONSTRUCT, "paired", grad=grad)
        loss, active = ft.pair_loss(head, better, worse, settings)
        if backward:
            loss.backward()
    return better, worse, loss, active


def pooled_activation(activation, item):
    ab = ft.complete_view(item, "paired").without_ag()
    _, mapping = serialize_antibody(ab)
    return activation[0, [mapping[coord] for coord in ab.coords()]].mean(dim=0)


@pytest.mark.parametrize("mode", MODES)
def test_shared_gradient_matches_the_analytic_up_projection_and_head_gradients(mode):
    policy, encoder = models(mode)
    policy.train()
    head, settings = ft.new_head(16, 3), config()
    captured = []
    handle = policy.up.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        better, worse, _, active = run_pair(policy, encoder, head, settings, grad=True)
    finally:
        handle.remove()
    assert active and len(captured) == 2
    assert better.requires_grad and worse.requires_grad
    # fused = hidden + up(act) * residue_mask and the pooled feature is the slot mean, so
    # d feature / d up.weight = mean_slots(act).  With gap = w . (f_b - f_w) the loss
    # (lambda / P) relu(m - gap)^2 gives d loss / d up.weight = -(2 lambda / P) relu(m - gap)
    # * outer(w, act_b - act_w), and d loss / d w = -(2 lambda / P) relu(m - gap) (f_b - f_w).
    w = head.weight.detach()[0]
    gap = float(w @ (better.detach() - worse.detach()))
    coefficient = -(2.0 * settings.weight / settings.pairs_per_update) * max(settings.margin - gap, 0.0)
    delta = pooled_activation(captured[0], BETTER_CONSTRUCT) - pooled_activation(captured[1], WORSE_CONSTRUCT)
    expected_up = coefficient * torch.outer(w, delta)
    torch.testing.assert_close(policy.up.weight.grad, expected_up, rtol=1e-3,
                               atol=1e-5 * float(expected_up.abs().max()))
    expected_head = coefficient * (better.detach() - worse.detach())
    torch.testing.assert_close(head.weight.grad[0], expected_head, rtol=1e-4,
                               atol=1e-5 * float(expected_head.abs().max()))
    if mode == "lora":
        retained = [p for name, p in policy.named_parameters() if "lora_B" in name]
    else:
        retained = [p for name, p in policy.named_parameters() if name.startswith("backbone.esm.")]
    assert retained and any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in retained)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())


@pytest.mark.parametrize("mode", MODES)
def test_detached_mode_trains_only_the_head_and_leaves_the_policy_untouched(mode):
    policy, encoder = models(mode)
    policy.train()
    head, settings = ft.new_head(16, 3), config(gradient_mode="detached")
    better, worse, _, active = run_pair(policy, encoder, head, settings,
                                        grad=settings.policy_receives_gradient)
    assert active and not better.requires_grad and not worse.requires_grad
    assert all(p.grad is None for p in policy.parameters())
    assert head.weight.grad is not None and bool(head.weight.grad.abs().sum() > 0)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())


@pytest.mark.parametrize("mode", MODES)
def test_gradient_mode_changes_only_detachment_never_data_or_values(mode):
    policy, encoder = models(mode)
    policy.train()
    head = ft.new_head(16, 3)
    calls = spy_on_forward(policy)
    shared = run_pair(policy, encoder, head, config(), grad=True, backward=False)
    shared_views = [views[0].content_digest() for views, _ in calls]
    del calls[:]
    detached = run_pair(policy, encoder, head, config(gradient_mode="detached"), grad=False, backward=False)
    assert shared_views == [views[0].content_digest() for views, _ in calls]
    assert len(shared_views) == 2 and shared[0].requires_grad and not detached[0].requires_grad
    for left, right in zip(shared[:3], detached[:3]):
        torch.testing.assert_close(left.detach(), right.detach(), rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_activation_checkpointing_does_not_change_features_or_gradients(mode, monkeypatch):
    checkpointed, encoder = models(mode)
    plain = copy.deepcopy(checkpointed)
    plain.config = replace(plain.config, activation_checkpointing=False)
    calls = []
    original = torch.utils.checkpoint.checkpoint

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting)
    results = []
    for policy in (checkpointed, plain):
        policy.train()
        before = len(calls)
        better, worse, loss, _ = run_pair(policy, encoder, ft.new_head(16, 3), config(), grad=True)
        results.append((better.detach(), worse.detach(), loss.detach(), len(calls) - before,
                        {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}))
    assert results[0][3] > 0 and results[1][3] == 0
    for left, right in zip(results[0][:3], results[1][:3]):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    assert set(results[0][4]) == set(results[1][4])
    for name, gradient in results[0][4].items():
        torch.testing.assert_close(gradient, results[1][4][name], rtol=1e-4, atol=1e-6)
