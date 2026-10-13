from __future__ import annotations

import dataclasses
import inspect
import math

import pytest
import torch

from smallAntibodyGen.models.antibody_antigen import denoising as dn
from smallAntibodyGen.models.antibody_antigen.views import (
    CANONICAL_RESIDUES,
    RESIDUE_INDEX,
    MaskedTargets,
    mask_complete,
)
from smallAntibodyGen.tests import aa_world as world

SEQ = world.SEQUENCES  # H=EVQLVESG (8)  L=DIQMTQ (6)  Ag=KVFGRCEL (8)  -> 22 residues


def approx(value, tol=1e-5):
    return pytest.approx(value, abs=tol, rel=tol)


def log_softmax_at(row, column):
    return row[column] - math.log(sum(math.exp(x) for x in row))


# ------------------------------------------------------------------ hidden truth


def test_hidden_truth_is_invisible_until_it_is_revealed():
    masked = [("H", 2), ("H", 3), ("L", 1)]
    view_a, targets_a = mask_complete(SEQ, masked)
    view_b, targets_b = mask_complete({"H": "EVAAVESG", "L": "DAQMTQ", "Ag": SEQ["Ag"]}, masked)
    assert view_a == view_b and targets_a != targets_b
    backend = world.ToyBackend(seed=2)
    assert torch.equal(backend.forward(view_a).raw_logits, backend.forward(view_b).raw_logits)
    # Unguided output cannot depend on the gold either.
    assert dn.complete(backend, view_a).residues == dn.complete(backend, view_b).residues

    order = [("H", 2), ("H", 3), ("L", 1)]
    spy_a, spy_b = world.RecordingBackend(backend), world.RecordingBackend(backend)
    dn.sequential_log_prob(spy_a, view_a, targets_a, order)
    dn.sequential_log_prob(spy_b, view_b, targets_b, order)
    assert spy_a.views[0] == spy_b.views[0] == view_a  # t0: identical, everything hidden
    assert spy_a.views[1] != spy_b.views[1]  # teacher forcing: each revealed its own gold
    assert not torch.equal(backend.forward(spy_a.views[1]).raw_logits, backend.forward(spy_b.views[1]).raw_logits)


# ------------------------------------------------------------------- masked NLL


def test_masked_nll_matches_a_hand_computed_oracle_and_scores_only_the_targets():
    view, targets = mask_complete(SEQ, [("H", 1), ("L", 0), ("H", 5)])
    only = MaskedTargets.from_mapping({("H", 1): "V", ("L", 0): "D"})  # H5 is hidden but not scored
    h_row, l_row = [0.0] * 20, [0.0] * 20
    h_row[RESIDUE_INDEX["V"]] = 2.0
    l_row[RESIDUE_INDEX["D"]] = -1.0
    backend = world.FixedLogitsBackend({("H", 1): h_row, ("L", 0): l_row, ("H", 5): [9.0] * 20}, mass=0.9)
    result = dn.masked_nll(backend, view, only)
    expected = -(2.0 - math.log(19 + math.exp(2.0))) - (-1.0 - math.log(19 + math.exp(-1.0)))
    assert result.total.item() == approx(expected)
    assert result.count == 2 and result.coords == (("H", 1), ("L", 0))
    assert result.site_nll.shape == (2,)
    assert torch.allclose(result.canonical_mass, torch.full((2,), 0.9))
    assert result.mask_ratio == 3 / 22


def test_scores_normalise_over_the_selected_20_logits_and_keep_outside_mass_visible():
    view, targets = mask_complete(SEQ, [("H", 2), ("L", 4)])
    backend = world.ToyBackend(seed=3)
    out = backend.forward(view)
    assert bool((out.canonical_mass < 1.0).all())  # a full-33 softmax would have given these 20 less mass
    result = dn.masked_nll(backend, view, targets)
    expected = sum(
        -torch.log_softmax(out.raw_logits[view.position(c)], dim=-1)[RESIDUE_INDEX[targets.residue(c)]]
        for c in targets.coords()
    )
    assert torch.allclose(result.total, expected, atol=1e-6)
    rows = [view.position(c) for c in targets.coords()]
    assert torch.allclose(result.canonical_mass, out.canonical_mass[rows])
    first = dn.sequential_log_prob(backend, view, targets, targets.coords())
    gold = RESIDUE_INDEX[targets.residue(targets.coords()[0])]
    assert first.step_log_probs[0].item() == approx(torch.log_softmax(out.raw_logits[rows[0]], dim=-1)[gold].item())
    assert first.step_mass[0].item() == approx(out.canonical_mass[rows[0]].item())


def test_masked_nll_refuses_targets_that_are_not_masked_or_are_empty():
    view, targets = mask_complete(SEQ, [("H", 2)])
    backend = world.FixedLogitsBackend()
    with pytest.raises(ValueError):
        dn.masked_nll(backend, view, MaskedTargets.from_mapping({("H", 3): "L"}))
    with pytest.raises(ValueError):
        dn.masked_nll(backend, view, MaskedTargets(()))
    assert dn.masked_nll(backend, view, targets).count == 1


def test_masked_nll_is_differentiable_through_the_backend():
    view, targets = mask_complete(SEQ, [("H", 2), ("L", 4)])
    backend = world.ToyBackend(seed=4)
    result = dn.masked_nll(backend, view, targets)
    assert result.total.requires_grad and result.site_nll.requires_grad
    result.total.backward()
    assert all(p.grad is not None for p in backend.parameters())
    assert not result.canonical_mass.requires_grad  # diagnostics are detached, the likelihood is not


# ----------------------------------------------------------------- sequential


ORDER = [("L", 1), ("H", 2), ("H", 5)]


def test_sequential_log_prob_matches_a_hand_computed_oracle_with_a_full_forward_per_step():
    view, targets = mask_complete(SEQ, ORDER)
    spy = world.RecordingBackend(world.StateDependentBackend())
    result = dn.sequential_log_prob(spy, view, targets, ORDER)
    weights = world.StateDependentBackend.WEIGHTS
    expected = []
    for step, coord in enumerate(ORDER):
        scaled = [(3 - step) * w for w in weights]  # (3 - step) masks are still hidden at this step
        expected.append(log_softmax_at(scaled, RESIDUE_INDEX[targets.residue(coord)]))
    assert result.step_log_probs.tolist() == approx(expected)
    assert result.log_prob.item() == approx(sum(expected))
    assert result.order == tuple(ORDER)
    assert result.step_mass.tolist() == approx([0.8] * 3)
    assert result.step_mask_ratio == (3 / 22, 2 / 22, 1 / 22)
    assert len(spy.views) == 3  # one full forward of the whole current state per step
    assert spy.views[0] == view
    assert spy.views[1] == view.reveal(("L", 1), "I")
    assert spy.views[2] == view.reveal(("L", 1), "I").reveal(("H", 2), "Q")


@pytest.mark.parametrize(
    "order",
    [
        [("H", 2)],  # too short
        [("L", 1), ("H", 2), ("H", 2)],  # repeats a site
        [("L", 1), ("H", 2), ("H", 0)],  # a visible site
        [("L", 1), ("H", 2), ("Ag", 0)],
        ORDER + [("H", 6)],  # too long
        [("L", 1), ("H", 2), ["H", 5]],  # not a coordinate tuple
    ],
)
def test_sequential_order_must_be_exactly_a_permutation_of_the_hidden_sites(order):
    view, targets = mask_complete(SEQ, ORDER)
    with pytest.raises(ValueError):
        dn.sequential_log_prob(world.StateDependentBackend(), view, targets, order)


def test_sequential_needs_a_gold_residue_for_every_hidden_site():
    view, _ = mask_complete(SEQ, ORDER)
    partial = MaskedTargets.from_mapping({("L", 1): "I", ("H", 2): "Q"})
    with pytest.raises(ValueError):
        dn.sequential_log_prob(world.StateDependentBackend(), view, partial, ORDER)


def test_gradient_reaches_every_sequential_step():
    view, targets = mask_complete(SEQ, ORDER)
    backend = world.ToyBackend(seed=1)
    result = dn.sequential_log_prob(backend, view, targets, ORDER)
    params = list(backend.parameters())
    assert result.log_prob.requires_grad and result.step_log_probs.requires_grad

    def grads(scalar):
        found = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for g, p in zip(found, params)]

    per_step = [grads(result.step_log_probs[t]) for t in range(3)]
    for step_grads in per_step:
        assert sum(float(g.abs().sum()) for g in step_grads) > 0.0
    for total, *parts in zip(grads(result.log_prob), *per_step):
        assert torch.allclose(total, sum(parts), atol=1e-6)


# -------------------------------------------------------------------- pair gap


def pair_case():
    changed = [("H", 3), ("H", 4)]  # L -> A, V -> K; every other site is shared
    view, targets_a = mask_complete(SEQ, changed)
    view_b, targets_b = mask_complete({**SEQ, "H": "EVQAKESG"}, changed)
    assert view == view_b
    return view, changed, targets_a, targets_b


def test_pair_gap_is_the_difference_of_two_sequential_scores_over_the_same_masks_and_order():
    view, order, targets_a, targets_b = pair_case()
    backend = world.ToyBackend(seed=6)
    spy = world.RecordingBackend(backend)
    pair = dn.sequential_gap(spy, view, order, targets_a, targets_b)
    a = dn.sequential_log_prob(backend, view, targets_a, order)
    b = dn.sequential_log_prob(backend, view, targets_b, order)
    assert pair.gap.item() == approx(b.log_prob.item() - a.log_prob.item())
    assert pair.a.order == pair.b.order == tuple(order)
    assert spy.views[0] == spy.views[2] == view  # both endpoints start from the same masks
    assert pair.gap.requires_grad


def test_pair_gap_refuses_to_mask_a_site_the_endpoints_share():
    view, targets_a = mask_complete(SEQ, [("H", 3), ("H", 4), ("L", 1)])
    _, targets_b = mask_complete({**SEQ, "H": "EVQAKESG"}, [("H", 3), ("H", 4), ("L", 1)])
    with pytest.raises(ValueError, match="unchanged"):
        dn.sequential_gap(world.ToyBackend(), view, view.hidden_sites(), targets_a, targets_b)


# A malformed pair is refused as a whole, before the backend is asked anything.


def gap_case(hidden):
    """AA vs CC masked at ``hidden``: the view, a valid order and the two valid endpoint targets."""
    view, targets_a = mask_complete({"H": "AA", "Ag": "GG"}, hidden)
    _, targets_b = mask_complete({"H": "CC", "Ag": "GG"}, hidden)
    return view, list(hidden), targets_a, targets_b


def assert_gap_refused_without_a_model_call(view, order, targets_a, targets_b, match):
    spy = world.RecordingBackend(world.FixedLogitsBackend())
    with pytest.raises(ValueError, match=match):
        dn.sequential_gap(spy, view, order, targets_a, targets_b)
    assert spy.views == []


@pytest.mark.parametrize("endpoint", ["a", "b"])
@pytest.mark.parametrize(
    "hidden, wrong",
    [
        ([("H", 0)], [("H", 0), ("H", 1)]),  # an extra coordinate that is visible in the view
        ([("H", 0)], [("H", 0), ("L", 0)]),  # an extra coordinate in a chain the view does not have
        ([("H", 0)], [("H", 1)]),  # as many coordinates as hidden sites, but the wrong one
        ([("H", 0), ("H", 1)], [("H", 0)]),  # missing the second hidden site
        ([("H", 0), ("H", 1)], [("H", 1)]),  # missing the first hidden site
        ([("H", 0)], []),  # no gold residue at all
    ],
    ids=["extra-visible", "extra-absent-chain", "wrong-site", "missing-last", "missing-first", "empty"],
)
def test_pair_gap_checks_both_endpoints_cover_exactly_the_hidden_sites_before_any_model_call(endpoint, hidden, wrong):
    view, order, targets_a, targets_b = gap_case(hidden)
    if endpoint == "a":
        targets_a = MaskedTargets.from_mapping({coord: "A" for coord in wrong})
    else:
        targets_b = MaskedTargets.from_mapping({coord: "C" for coord in wrong})
    assert_gap_refused_without_a_model_call(view, order, targets_a, targets_b, "exactly the hidden sites")


def test_pair_gap_refuses_an_empty_view_or_order_before_any_model_call():
    nothing_hidden, no_targets = mask_complete({"H": "AA", "Ag": "GG"})
    assert_gap_refused_without_a_model_call(nothing_hidden, [], no_targets, no_targets, "no hidden sites")
    view, _, targets_a, targets_b = gap_case([("H", 0)])
    assert_gap_refused_without_a_model_call(view, [], targets_a, targets_b, "permutation")
    assert_gap_refused_without_a_model_call(view, [("H", 0), ("H", 0)], targets_a, targets_b, "repeats")


def test_pair_gap_refuses_an_unchanged_site_before_any_model_call():
    view, order, targets_a, _ = gap_case([("H", 0), ("H", 1)])
    shares_h1 = MaskedTargets.from_mapping({("H", 0): "C", ("H", 1): "A"})
    assert_gap_refused_without_a_model_call(view, order, targets_a, shares_h1, "unchanged")


def test_pair_gap_scores_each_endpoint_once_along_the_validated_order():
    view, order, targets_a, targets_b = gap_case([("H", 0), ("H", 1)])
    spy = world.RecordingBackend(world.StateDependentBackend())
    pair = dn.sequential_gap(spy, view, order, targets_a, targets_b)
    assert pair.a.order == pair.b.order == tuple(order)
    assert len(spy.views) == 4  # two steps per endpoint, nothing else


def tight(value):
    return pytest.approx(value, abs=1e-10)


class TwoSiteToy(torch.nn.Module):
    """Float64 software toy with a closed-form likelihood; it models nothing real.

    Every row has C at 0, the other 18 non-A residues at -30 and A at ``a``. At H0 ``a`` is
    ``theta[0]``. At H1 it is ``theta[1]`` once H0 is visible as A, ``theta[2]`` once H0 is
    visible as C, and 0 while H0 is masked (or anything else).
    """

    def __init__(self, theta):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float64))

    def forward(self, view):
        coords = view.coords()
        zero = torch.zeros((), dtype=torch.float64)
        a_at = {
            ("H", 0): self.theta[0],
            ("H", 1): {"A": self.theta[1], "C": self.theta[2]}.get(view.token(("H", 0)), zero),
        }
        flat = torch.full((dn.N_CANONICAL,), -30.0, dtype=torch.float64)
        flat[RESIDUE_INDEX["C"]] = 0.0
        is_a = torch.arange(dn.N_CANONICAL) == RESIDUE_INDEX["A"]
        return dn.BackendOutput(
            coords=coords,
            raw_logits=torch.stack([torch.where(is_a, a_at[coord], flat) for coord in coords]),
            canonical_mass=torch.ones(len(coords), dtype=torch.float64),
            features=torch.zeros(len(coords), 1, dtype=torch.float64),
            mask_ratio=len(view.hidden_sites()) / (len(coords) + 2),  # + <cls> and <eos> of the one chain
        )


def two_site_case():
    sites = [("H", 0), ("H", 1)]
    view, targets_aa = mask_complete({"H": "AA"}, sites)
    view_cc, targets_cc = mask_complete({"H": "CC"}, sites)
    assert view == view_cc
    return view, sites, targets_aa, targets_cc


def two_site_oracle(theta, target=1.5):
    """Closed form of ``TwoSiteToy`` from the row partition function alone; calls no library code.

    A row has Z(x) = e^x + 1 + 18 e^-30 (A at x, C at 0, 18 others at -30), so log p(A) = x - log Z(x)
    and log p(C) = -log Z(x). Teacher forcing reveals the endpoint's own gold at H0: A for AA (H1 then
    uses theta[1]) and C for CC (H1 then uses theta[2]). Returns J(AA), J(CC), the squared-gap loss
    (gap - target)^2 with gap = J(CC) - J(AA), and its gradient in theta.
    """
    t0, t1, t2 = theta

    def z(x):
        return math.exp(x) + 1.0 + 18.0 * math.exp(-30.0)

    def p_a(x):
        return math.exp(x) / z(x)

    j_aa = (t0 - math.log(z(t0))) + (t1 - math.log(z(t1)))
    j_cc = -math.log(z(t0)) - math.log(z(t2))
    gap = j_cc - j_aa
    d_gap = (-1.0, -(1.0 - p_a(t1)), -p_a(t2))  # d(J_CC - J_AA)/d(theta0, theta1, theta2)
    return j_aa, j_cc, (gap - target) ** 2, [2.0 * (gap - target) * d for d in d_gap]


TWO_SITE_THETA = (0.2, -0.4, 0.7)
TWO_SITE_J_AA = -1.511154121783311
TWO_SITE_J_CC = -1.901324918268367
TWO_SITE_GAP = -0.3901707964850558  # J(CC) - J(AA)
TWO_SITE_LOSS = 3.57274563988495  # (gap - 1.5)^2
TWO_SITE_GRAD = [3.7803415929701116, 2.2632438627225855, 2.5259780270399435]
TWO_SITE_LOSS_AFTER_SGD = 3.319497550085768  # one SGD step, lr 0.01


def test_two_site_oracle_constants_follow_from_the_closed_form():
    j_aa, j_cc, loss, grad = two_site_oracle(TWO_SITE_THETA)
    assert [j_aa, j_cc, j_cc - j_aa, loss] == tight([TWO_SITE_J_AA, TWO_SITE_J_CC, TWO_SITE_GAP, TWO_SITE_LOSS])
    assert grad == tight(TWO_SITE_GRAD)
    stepped = [t - 0.01 * g for t, g in zip(TWO_SITE_THETA, TWO_SITE_GRAD)]
    assert two_site_oracle(stepped)[2] == tight(TWO_SITE_LOSS_AFTER_SGD)


def test_squared_gap_loss_matches_an_analytic_two_site_oracle_and_one_sgd_step():
    view, order, targets_aa, targets_cc = two_site_case()
    toy = TwoSiteToy(TWO_SITE_THETA)
    pair = dn.sequential_gap(toy, view, order, targets_aa, targets_cc)
    assert pair.a.log_prob.item() == tight(TWO_SITE_J_AA)
    assert pair.b.log_prob.item() == tight(TWO_SITE_J_CC)
    assert pair.gap.item() == tight(TWO_SITE_GAP)
    assert pair.a.step_mask_ratio == pair.b.step_mask_ratio == (2 / 4, 1 / 4)  # hidden / (residues + 2)
    loss = (pair.gap - 1.5) ** 2  # stands in for kappa * delta; the utility itself knows no measurements
    assert loss.item() == tight(TWO_SITE_LOSS)
    loss.backward()
    assert toy.theta.grad.tolist() == tight(TWO_SITE_GRAD)
    torch.optim.SGD(toy.parameters(), lr=0.01).step()
    after = (dn.sequential_gap(toy, view, order, targets_aa, targets_cc).gap - 1.5) ** 2
    assert after.item() == tight(TWO_SITE_LOSS_AFTER_SGD)


# ------------------------------------------------------------------ completion


def test_complete_signature_has_no_slot_for_hidden_truth():
    assert set(inspect.signature(dn.complete).parameters) == {"backend", "view", "order", "sample", "generator"}


def test_given_order_completion_is_deterministic_and_preserves_every_known_residue():
    view, _ = mask_complete(SEQ, ORDER)
    spy = world.RecordingBackend(world.ToyBackend(seed=7))
    done = dn.complete(spy, view, order=ORDER)
    assert done.view.hidden_sites() == () and done.order == tuple(ORDER)
    assert len(done.residues) == len(done.step_log_probs) == 3
    for coord in view.coords():
        if view.token(coord) is not None:
            assert done.view.token(coord) == view.token(coord)
    assert [len(v.hidden_sites()) for v in spy.views] == [3, 2, 1]  # both streams re-run after every reveal
    assert dn.complete(spy.inner, view, order=ORDER) == done
    assert all(residue in CANONICAL_RESIDUES for residue in done.residues)


def test_easy_first_reveals_the_most_confident_site_and_breaks_ties_by_coordinate():
    view, _ = mask_complete(SEQ, ORDER)
    peaked = [0.0] * 20
    peaked[RESIDUE_INDEX["V"]] = 10.0
    done = dn.complete(world.FixedLogitsBackend({("H", 5): peaked}), view)
    assert done.order == (("H", 5), ("H", 2), ("L", 1))
    assert done.residues == ("V", "A", "A")  # flat rows: ties go to the first column


def test_seeded_sampling_is_reproducible_needs_an_explicit_generator_and_varies_with_the_seed():
    view, _ = mask_complete(SEQ, [("H", 0), ("H", 1), ("H", 2), ("L", 0), ("L", 1), ("L", 2)])
    backend = world.FixedLogitsBackend()  # uniform: sampling decides everything

    def draw(seed, **kwargs):
        return dn.complete(backend, view, sample=True, generator=torch.Generator().manual_seed(seed), **kwargs)

    assert draw(11) == draw(11)
    assert draw(11, order=view.hidden_sites()) == draw(11, order=view.hidden_sites())
    assert len({draw(seed).residues for seed in range(8)}) > 1
    assert draw(11).view.hidden_sites() == ()
    with pytest.raises(ValueError):
        dn.complete(backend, view, sample=True)


def test_completion_supports_absent_light_and_antigen_chains_and_finishes_in_hidden_count_steps():
    heavy_only, _ = mask_complete({"H": SEQ["H"]}, [("H", 1), ("H", 2)])
    spy = world.RecordingBackend(world.ToyBackend(seed=8))
    done = dn.complete(spy, heavy_only)
    assert done.view.roles == ("H",) and done.view.hidden_sites() == () and len(spy.views) == 2
    light_antigen, _ = mask_complete({"L": SEQ["L"], "Ag": SEQ["Ag"]}, [("L", 0)])
    assert dn.complete(world.ToyBackend(seed=8), light_antigen).view.hidden_sites() == ()
    for complete_view in (done.view, mask_complete({"Ag": SEQ["Ag"]})[0]):
        zero_steps = dn.complete(spy, complete_view)
        assert zero_steps.view == complete_view and zero_steps.order == () and zero_steps.residues == ()


# ------------------------------------------------------------ output validation


class Corrupted:
    def __init__(self, edit):
        self.inner, self.edit = world.FixedLogitsBackend(), edit

    def forward(self, view):
        return self.edit(self.inner.forward(view))


@pytest.mark.parametrize(
    "edit",
    [
        lambda out: dataclasses.replace(out, coords=out.coords[::-1]),
        lambda out: dataclasses.replace(out, raw_logits=out.raw_logits[:-1]),
        lambda out: dataclasses.replace(out, raw_logits=out.raw_logits[:, :19]),
        lambda out: dataclasses.replace(out, raw_logits=out.raw_logits + float("nan")),
        lambda out: dataclasses.replace(out, canonical_mass=out.canonical_mass * 2),
        lambda out: dataclasses.replace(out, canonical_mass=out.canonical_mass[:-1]),
        lambda out: dataclasses.replace(out, features=out.features[:-1]),
        lambda out: dataclasses.replace(out, features=out.features / 0.0),
        lambda out: dataclasses.replace(out, mask_ratio=1.5),
        lambda out: dataclasses.replace(out, mask_ratio=0),  # an int is not the observed float ratio
    ],
)
def test_misaligned_or_non_finite_backend_output_is_refused(edit):
    view, targets = mask_complete(SEQ, [("H", 2)])
    with pytest.raises(ValueError):
        dn.masked_nll(Corrupted(edit), view, targets)
