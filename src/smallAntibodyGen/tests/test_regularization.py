"""Exact KL-gradient checks and independent representation calculations."""
import itertools

import pytest
import torch

from smallAntibodyGen.experiments.regularization import off_diagonal_cosine, pooled_embeddings, reverse_kl_surrogate


@pytest.mark.parametrize("batch_size", [2, 3])
def test_expected_reverse_kl_gradient_matches_exact_enumeration(batch_size):
    logits = torch.tensor([1.4, -.6, .2], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([-.7, 1.1, .3], dtype=torch.float64).log_softmax(0)
    logq = logits.log_softmax(0)
    expected = torch.zeros_like(logits)
    for indices in itertools.product(range(3), repeat=batch_size):
        chosen = logq[list(indices)]
        gradient, = torch.autograd.grad(reverse_kl_surrogate(chosen, reference[list(indices)]), logits, retain_graph=True)
        expected += chosen.detach().sum().exp() * gradient
    exact, = torch.autograd.grad((logq.exp() * (logq - reference)).sum(), logits)
    torch.testing.assert_close(expected, exact, rtol=1e-12, atol=1e-12)


def test_equal_reference_has_zero_gradient_and_reference_cannot_be_trainable():
    scores = torch.tensor([-1., -2., -1.], requires_grad=True)
    reverse_kl_surrogate(scores, scores.detach()).backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))
    with pytest.raises(ValueError): reverse_kl_surrogate(scores, scores)


@pytest.mark.parametrize("reference", [torch.tensor([-1.]), torch.tensor([float("nan"), -2.]), torch.tensor([1., -2.])])
def test_invalid_reference_rejected(reference):
    with pytest.raises(ValueError): reverse_kl_surrogate(torch.tensor([-1., -2.]), reference)


def test_masked_pooling_ignores_padding_and_backpropagates_only_to_valid_positions():
    features = torch.tensor([[[3., 0.], [0., 4.], [float("nan"), 2.]],
                             [[0., 2.], [2., 0.], [99., 99.]]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    z = pooled_embeddings(features, mask)
    torch.testing.assert_close(z, torch.tensor([[.6, .8], [0., 1.]]))
    loss = off_diagonal_cosine(z)
    assert loss.item() == pytest.approx(.8)
    loss.backward()
    assert features.grad[mask].abs().sum() > 0
    assert features.grad[~mask].abs().sum() == 0


def test_pairwise_cosine_matches_explicit_off_diagonal_average_and_keeps_duplicates():
    z = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]])
    expected = torch.stack([a @ b for i, a in enumerate(z) for j, b in enumerate(z) if i != j]).mean()
    torch.testing.assert_close(off_diagonal_cosine(z), expected)
    assert off_diagonal_cosine(z[:2]).item() == pytest.approx(1.)


def test_representation_change_alone_can_improve_cosine():
    # Same two identities can move apart without any policy-probability change.
    before = torch.tensor([[1., 0.], [1., 0.]])
    after = torch.tensor([[1., 0.], [-1., 0.]])
    assert off_diagonal_cosine(after) < off_diagonal_cosine(before)
