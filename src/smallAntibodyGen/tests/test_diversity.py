"""Enumerable distributions verify the actual entropy-gradient expectation."""
import itertools

import pytest
import torch

from smallAntibodyGen.experiments.diversity import negative_entropy_surrogate


@pytest.mark.parametrize("batch_size", [2, 3])
def test_expected_surrogate_gradient_equals_exact_negative_entropy(batch_size):
    logits = torch.tensor([1.4, -.6, .2], dtype=torch.float64, requires_grad=True)
    logq = logits.log_softmax(0)
    expected = torch.zeros_like(logits)
    for indices in itertools.product(range(3), repeat=batch_size):
        chosen = logq[list(indices)]
        weight = chosen.detach().sum().exp()
        grad, = torch.autograd.grad(negative_entropy_surrogate(chosen), logits, retain_graph=True)
        expected += weight * grad
    exact, = torch.autograd.grad((logq.exp() * logq).sum(), logits)
    torch.testing.assert_close(expected, exact, rtol=1e-12, atol=1e-12)


def test_gradient_descent_on_expected_negative_entropy_flattens_policy():
    logits = torch.tensor([3., 0.], requires_grad=True)
    logq = logits.log_softmax(0)
    gradient = torch.zeros_like(logits)
    for a, b in itertools.product(range(2), repeat=2):
        selected = logq[[a, b]]
        grad, = torch.autograd.grad(negative_entropy_surrogate(selected), logits, retain_graph=True)
        gradient += selected.detach().sum().exp() * grad
    assert gradient[0] > 0 and gradient[1] < 0


def test_uniform_batch_has_zero_entropy_gradient():
    logits = torch.zeros(3, requires_grad=True)
    negative_entropy_surrogate(logits.log_softmax(0)).backward()
    torch.testing.assert_close(logits.grad, torch.zeros(3))


@pytest.mark.parametrize("value", [torch.tensor([-1.]), torch.tensor([1., -1.]),
                                  torch.tensor([float('nan'), -1.]), torch.ones(2, 2),
                                  torch.tensor([-1, -2])])
def test_invalid_entropy_input_rejected(value):
    with pytest.raises(ValueError):
        negative_entropy_surrogate(value)
