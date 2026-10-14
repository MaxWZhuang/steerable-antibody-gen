"""Tests for trastuzumab_poc masking and ranking objectives."""
from __future__ import annotations

import random
import copy

import pytest
import torch

from smallAntibodyGen.models.dplm_policy import DPLMConfig, NativeDPLM
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.antibody_antigen.static_context import StaticContext
from smallAntibodyGen.models.antibody_antigen.denoising import sequential_log_prob
from smallAntibodyGen.models.antibody_antigen.views import (
    mask_complete, MaskedTargets, VisibleView,
)
from smallAntibodyGen.experiments.trastuzumab_poc.objectives import (
    draw_mask, whole_core_scores, ordering_loss,
)

IDENTITY = {
    "source_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
    "extraction_layer": "layer_2", "precision": "float32",
    "tokenizer": "ESM-1b", "eval_mode": "true",
}


def make_policy(*, checkpointing=False, dropout=0.0):
    torch.manual_seed(42)
    cfg = DPLMConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, hidden_dropout_prob=dropout,
        max_position_embeddings=64,
    )
    native = NativeDPLM.random_init(cfg, seed=7, std=0.1)
    policy = FusedDPLM(
        native, IDENTITY,
        FusionConfig(context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4,
                     activation_checkpointing=checkpointing),
    )
    # hidden_union = {("H", 2), ("H", 3)}
    view, _ = mask_complete({"H": "ACDEFG", "L": "HIK"}, [("H", 2), ("H", 3)])
    ctx = StaticContext(view, torch.randn(len(view.coords()), 8), IDENTITY)
    return policy, ctx, view


# ── draw_mask tests ───────────────────────────────────────────────────────────

UNION_LARGE = [("H", i) for i in range(10)] + [("L", i) for i in range(5)]
UNION_GAPPED = [("H", 0), ("H", 1), ("H", 5), ("H", 6)]
UNION_SMALL = [("H", 0), ("H", 1)]


def test_draw_mask_all_four_modes_seen():
    rng = random.Random(0)
    seen = set()
    for _ in range(300):
        seen.add(draw_mask(UNION_LARGE, rng).mode)
    assert seen == {0, 1, 2, 3}


def test_draw_mask_sites_within_union():
    rng = random.Random(1)
    universe = set(map(tuple, UNION_LARGE))
    for _ in range(150):
        draw = draw_mask(UNION_LARGE, rng)
        assert draw.sites
        assert all(s in universe for s in draw.sites)


def test_draw_mask_reproducible():
    rng1, rng2 = random.Random(99), random.Random(99)
    draws1 = [draw_mask(UNION_LARGE, rng1) for _ in range(30)]
    draws2 = [draw_mask(UNION_LARGE, rng2) for _ in range(30)]
    for d1, d2 in zip(draws1, draws2):
        assert d1.mode == d2.mode and d1.sites == d2.sites


def test_draw_mask_span_does_not_bridge_gap():
    """Mode-2 spans must stay within a single consecutive run; no gap-bridging."""
    rng = random.Random(3)
    for _ in range(400):
        draw = draw_mask(UNION_GAPPED, rng)
        if draw.mode == 2 and len(draw.sites) > 1:
            indices = sorted(idx for _, idx in draw.sites)
            for a, b in zip(indices, indices[1:]):
                assert b == a + 1, f"span bridged a gap: {draw.sites}"


def test_draw_mask_small_region_does_not_crash():
    rng = random.Random(5)
    universe = set(map(tuple, UNION_SMALL))
    for _ in range(150):
        draw = draw_mask(UNION_SMALL, rng)
        assert 1 <= len(draw.sites) <= len(UNION_SMALL)
        assert all(s in universe for s in draw.sites)


def test_draw_mask_rejects_ag_coord():
    with pytest.raises(ValueError, match="H/L"):
        draw_mask([("Ag", 0)], random.Random(0))


def test_draw_mask_rejects_empty_union():
    with pytest.raises(ValueError):
        draw_mask([], random.Random(0))


def test_draw_mask_validation_precedes_rng():
    """Duplicate coord must be caught before any RNG is consumed."""
    rng = random.Random(0)
    state_before = rng.getstate()
    with pytest.raises(ValueError, match="duplicate"):
        draw_mask([("H", 0), ("H", 0)], rng)
    assert rng.getstate() == state_before, "RNG advanced before validation completed"


# ── whole_core_scores tests ───────────────────────────────────────────────────

def test_whole_core_scores_matches_sequential_log_prob():
    """Batched scores must equal sequential_log_prob / K for two endpoints sharing H:2."""
    policy, ctx, view = make_policy()
    policy.eval()

    tgt_a = MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"})
    tgt_b = MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "F"})
    order = (("H", 2), ("H", 3))

    scores = whole_core_scores(policy, ctx, view, [tgt_a, tgt_b], order, state_microbatch=1)
    assert scores.shape == (2,)

    bound = policy.bind(ctx)
    ref_a = sequential_log_prob(bound, view, tgt_a, order).log_prob / 2
    ref_b = sequential_log_prob(bound, view, tgt_b, order).log_prob / 2
    torch.testing.assert_close(scores[0], ref_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(scores[1], ref_b, atol=1e-5, rtol=1e-5)


def test_whole_core_scores_multi_microbatch_consistent():
    """microbatch=1 and microbatch=4 must give identical results for 3 targets."""
    policy, ctx, view = make_policy()
    policy.eval()

    targets = [
        MaskedTargets.from_mapping({("H", 2): "A", ("H", 3): "C"}),
        MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"}),
        MaskedTargets.from_mapping({("H", 2): "G", ("H", 3): "H"}),
    ]
    order = (("H", 2), ("H", 3))

    s1 = whole_core_scores(policy, ctx, view, targets, order, state_microbatch=1)
    s4 = whole_core_scores(policy, ctx, view, targets, order, state_microbatch=4)
    torch.testing.assert_close(s1, s4, atol=1e-6, rtol=1e-6)


def test_whole_core_scores_gradients_match_literal_reference():
    """Checkpointed chunked scoring matches every literal-path parameter gradient."""
    policy, ctx, view = make_policy(checkpointing=True)
    policy.train()
    # Exercise both sides of the adapters. At initialization, zero up/B matrices
    # intentionally give zero first-step gradients to down/A, which is not a bug.
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if "lora_B" in name or name == "up.weight":
                param.normal_(std=0.01)
    reference = copy.deepcopy(policy)
    targets = [MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): r}) for r in ("E", "F", "G")]
    order = (("H", 3), ("H", 2))  # reversed
    scores = whole_core_scores(policy, ctx, view, targets, order, state_microbatch=4)
    expected = torch.stack([sequential_log_prob(reference.bind(ctx), view, t, order).log_prob / 2
                            for t in targets])
    weights = torch.tensor([1.0, -0.5, 0.2])
    (scores * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(scores, expected, atol=2e-6, rtol=2e-5)
    for (name, p), (_, q) in zip(policy.named_parameters(), reference.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)


@pytest.mark.parametrize("index", [True, False])
def test_ordering_rejects_boolean_indices(index):
    with pytest.raises(TypeError, match="ints"):
        ordering_loss(torch.tensor([1.0, 2.0]), [(index, 1)])


@pytest.mark.parametrize("scores", [torch.tensor([float("nan"), 1.0]), torch.tensor([1, 2])])
def test_ordering_requires_finite_floating_scores(scores):
    with pytest.raises(ValueError, match="finite"):
        ordering_loss(scores, [(0, 1)])


@pytest.mark.parametrize("size", [True, 0, 1.5])
def test_whole_core_requires_integer_microbatch(size):
    policy, ctx, view = make_policy()
    tgt = MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"})
    with pytest.raises(ValueError, match="integer"):
        whole_core_scores(policy, ctx, view, [tgt], view.hidden_sites(), size)


def test_whole_core_scores_path_states_correct(monkeypatch):
    """At step k, the queried coord and all successors must still be masked."""
    policy, ctx, view = make_policy()
    tgt = MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"})
    other = MaskedTargets.from_mapping({("H", 2): "F", ("H", 3): "E"})
    order = (("H", 2), ("H", 3))

    captured: list = []
    orig_qrp = policy.queried_log_probs  # bound method captured before patch

    def capturing(views, contexts, coords, residues):
        for v, c in zip(views, coords):
            assert v.token(c) is None, f"queried coord {c} was not masked"
        captured.extend(list(zip(views, coords)))
        return orig_qrp(views, contexts, coords, residues)

    monkeypatch.setattr(policy, "queried_log_probs", capturing)
    whole_core_scores(policy, ctx, view, [tgt, other], order, state_microbatch=3)
    assert captured[0][0] == captured[2][0] == view
    assert captured[1][0].token(("H", 2)) == "D"
    assert captured[3][0].token(("H", 2)) == "F"

    # step 0 queries H:2 — H:3 must still be masked
    step0_view = captured[0][0]
    assert step0_view.token(("H", 3)) is None, "successor H:3 was prematurely revealed"
    # step 1 queries H:3 — H:2 must have been revealed
    step1_view = captured[1][0]
    assert step1_view.token(("H", 2)) is not None, "predecessor H:2 was not revealed"


def test_whole_core_scores_rejects_incomplete_order():
    policy, ctx, view = make_policy()
    tgt = MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"})
    with pytest.raises(ValueError):
        whole_core_scores(policy, ctx, view, [tgt], [("H", 2)], state_microbatch=4)


def test_whole_core_scores_rejects_ag_in_view():
    policy, ctx, _ = make_policy()
    ag_view, _ = mask_complete(
        {"H": "ACDEFG", "L": "HIK", "Ag": "MNP"}, [("H", 2), ("H", 3)]
    )
    tgt = MaskedTargets.from_mapping({("H", 2): "D", ("H", 3): "E"})
    with pytest.raises(ValueError, match="Ag"):
        whole_core_scores(policy, ctx, ag_view, [tgt], [("H", 2), ("H", 3)],
                          state_microbatch=4)


def test_whole_core_scores_rejects_partial_target():
    policy, ctx, view = make_policy()
    partial = MaskedTargets.from_mapping({("H", 2): "D"})
    with pytest.raises(ValueError):
        whole_core_scores(policy, ctx, view, [partial], [("H", 2), ("H", 3)],
                          state_microbatch=4)


# ── ordering_loss tests ───────────────────────────────────────────────────────

def test_ordering_loss_exact_zero_beyond_margin():
    scores = torch.tensor([1.0, 0.0])   # gap = 1.0 > margin = 0.5
    assert float(ordering_loss(scores, [(0, 1)], margin=0.5)) == 0.0


def test_ordering_loss_correct_gradient_direction():
    # scores[0] is labeled "better" but has a smaller value: gap < margin → loss > 0
    scores = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = ordering_loss(scores, [(0, 1)], margin=0.5)
    loss.backward()
    # gradient pushes scores[0] up (negative grad) and scores[1] down (positive grad)
    assert scores.grad[0] < 0
    assert scores.grad[1] > 0


def test_ordering_loss_rejects_self_pair():
    with pytest.raises(ValueError, match="self-pair"):
        ordering_loss(torch.tensor([1.0, 2.0]), [(0, 0)], margin=0.1)


def test_ordering_loss_rejects_empty_pairs():
    with pytest.raises(ValueError):
        ordering_loss(torch.tensor([1.0, 2.0]), [], margin=0.1)


def test_ordering_loss_rejects_nonpositive_margin():
    scores = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        ordering_loss(scores, [(0, 1)], margin=0.0)
    with pytest.raises(ValueError):
        ordering_loss(scores, [(0, 1)], margin=-0.5)
