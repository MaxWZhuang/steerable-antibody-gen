"""Preference direction, enumerated gradients, and immutable reference identity."""
import copy
import json
import math

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments.dpo import (
    dpo_per_pair_loss, load_reference_cache, write_reference_cache,
)


def test_identity_reference_has_log_two_loss_and_correct_gradient_direction():
    chosen = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    loss = dpo_per_pair_loss(chosen, rejected, chosen.detach(), rejected.detach(), beta=0.2)
    assert loss.item() == pytest.approx(math.log(2))
    loss.mean().backward()
    assert chosen.grad.item() == pytest.approx(-0.1)
    assert rejected.grad.item() == pytest.approx(0.1)


def test_exact_enumerable_policy_gradient_agrees_with_manual_binary_preference():
    # Four complete sequences, with different pair weights. The softmax
    # normalizer cancels within each ratio; no raw sequence-length normalization.
    logits = torch.tensor([0.2, -0.3, 0.7, -0.1], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float64).log_softmax(0)
    logq = logits.log_softmax(0)
    assert logq.exp().sum().item() == pytest.approx(1)
    chosen, rejected = torch.tensor([0, 2]), torch.tensor([1, 3])
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    losses = dpo_per_pair_loss(logq[chosen], logq[rejected], reference[chosen], reference[rejected], beta=0.4)
    gradient = torch.autograd.grad((weights * losses).sum(), logits)[0]
    margins = 0.4 * ((logits[chosen] - logits[rejected]) - (reference[chosen] - reference[rejected]))
    expected = torch.zeros_like(logits)
    direction = -weights * 0.4 * torch.sigmoid(-margins)
    expected[chosen], expected[rejected] = direction, -direction
    torch.testing.assert_close(gradient, expected)
    assert not reference.requires_grad


def test_extreme_margins_have_finite_loss_and_gradients():
    chosen = torch.tensor([-10000.0, 0.0], requires_grad=True)
    rejected = torch.tensor([0.0, -10000.0], requires_grad=True)
    zeros = torch.zeros(2)
    loss = dpo_per_pair_loss(chosen, rejected, zeros, zeros, beta=1.0).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(chosen.grad).all()


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf"), True])
def test_invalid_beta_rejected(beta):
    values = torch.zeros(2)
    with pytest.raises(ValueError, match="beta"):
        dpo_per_pair_loss(values, values, values, values, beta=beta)


def test_broadcasting_and_trainable_reference_are_rejected():
    vector = torch.zeros(2)
    with pytest.raises(ValueError, match="shape"):
        dpo_per_pair_loss(vector, vector[:, None], vector, vector, beta=0.1)
    with pytest.raises(ValueError, match="frozen"):
        dpo_per_pair_loss(vector, vector, vector.clone().requires_grad_(), vector, beta=0.1)


def cache_inputs():
    return {"checkpoint": "parent", "geometry": "5CJQ", "probability_contract": "binary16-v1"}, ["0" * 16, "1" * 16]


def test_cache_roundtrip_readonly_and_no_overwrite(tmp_path):
    identity, genotypes = cache_inputs()
    path = tmp_path / "cache.json"
    write_reference_cache(path, identity, genotypes, [-2.5, -0.5])
    values = load_reference_cache(path, identity, genotypes)
    np.testing.assert_array_equal(values, [-2.5, -0.5])
    assert not values.flags.writeable
    with pytest.raises(FileExistsError):
        write_reference_cache(path, identity, genotypes, [-2.5, -0.5])


@pytest.mark.parametrize("field", ["checkpoint", "geometry", "probability_contract"])
def test_reference_identity_changes_are_rejected(tmp_path, field):
    identity, genotypes = cache_inputs()
    path = tmp_path / "cache.json"
    write_reference_cache(path, identity, genotypes, [-2.5, -0.5])
    changed = copy.deepcopy(identity)
    changed[field] = "different"
    with pytest.raises(ValueError, match="identity"):
        load_reference_cache(path, changed, genotypes)
    with pytest.raises(ValueError, match="order"):
        load_reference_cache(path, identity, genotypes[::-1])


def test_cache_corruption_is_rejected(tmp_path):
    identity, genotypes = cache_inputs()
    path = tmp_path / "cache.json"
    write_reference_cache(path, identity, genotypes, [-2.5, -0.5])
    document = json.loads(path.read_text())
    document["log_probabilities"][0] = -1.0
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest"):
        load_reference_cache(path, identity, genotypes)


@pytest.mark.parametrize("values", [[float("nan"), -1], [float("inf"), -1], [0.1, -1], [-1]])
def test_invalid_cache_values_rejected(tmp_path, values):
    identity, genotypes = cache_inputs()
    with pytest.raises(ValueError, match="probability"):
        write_reference_cache(tmp_path / "cache.json", identity, genotypes, values)
