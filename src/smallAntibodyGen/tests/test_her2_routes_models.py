"""Offline model tests: heads, the energy route, hooks, conditions and the ledger.

No pinned weights and no F: access. The native-weight checks live in
``test_her2_routes_native.py`` and are opt-in.

Two of these tests exist because the thing they check reports a plausible number when
it is wrong: an always-empty head metric reads as ``0.0``, and a detached head loss
trains nothing while the loss curve still moves.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import her2_routes_data as data_lib
from smallAntibodyGen.experiments.her2_routes_conditions import (
    ASSAY_NULL,
    BIN_HIGH,
    BIN_NULL,
    Condition,
    ConditionSpec,
)
from smallAntibodyGen.models import her2_routes_energy as energy_lib
from smallAntibodyGen.models import her2_routes_heads as heads_lib
from smallAntibodyGen.models import her2_routes_hooks as hooks_lib


def spec():
    return ConditionSpec(assays=("assay_a", "assay_b"), bin_dropout=0.1, assay_dropout=0.25)


# --- conditions -----------------------------------------------------------

def test_family_is_never_dropped_and_high_bins_cannot_collapse():
    table = spec()
    rng = np.random.default_rng(0)
    display = Condition(family="display", bin=BIN_HIGH, assay="assay_a")
    affinity = Condition(family="affinity", bin=BIN_HIGH, assay="assay_a")
    for _ in range(200):
        left = table.apply_dropout(display, rng)
        right = table.apply_dropout(affinity, rng)
        assert left.family == "display"
        assert right.family == "affinity"
        assert table.encode(left) != table.encode(right)


def test_dropout_reaches_both_components_at_their_own_rates():
    table = ConditionSpec(assays=("a",), bin_dropout=0.5, assay_dropout=0.5)
    rng = np.random.default_rng(1)
    condition = Condition(family="affinity", bin=BIN_HIGH, assay="a")
    outcomes = [table.apply_dropout(condition, rng) for _ in range(400)]
    assert any(c.bin == BIN_NULL for c in outcomes)
    assert any(c.assay == ASSAY_NULL for c in outcomes)
    assert any(c.bin == BIN_HIGH and c.assay == "a" for c in outcomes)


def test_an_unseen_assay_is_refused_not_given_an_untrained_row():
    table = spec()
    with pytest.raises(ValueError, match="served by the dropped-assay route"):
        table.encode(Condition(family="affinity", bin=BIN_HIGH, assay="never_enrolled"))
    # The dropped-assay slot IS available and is the declared route for an unseen study.
    assert table.encode(Condition(family="affinity", bin=BIN_HIGH, assay=ASSAY_NULL))


def test_answer_free_strips_only_the_bin():
    table = spec()
    condition = Condition(family="affinity", bin=BIN_HIGH, assay="assay_a")
    free = table.answer_free(condition)
    assert free.bin == BIN_NULL
    assert free.family == condition.family and free.assay == condition.assay
    assert free.is_answer_free


def test_condition_spec_refuses_the_reserved_null_assay_as_a_study():
    with pytest.raises(ValueError, match="reserved dropped-assay slot"):
        ConditionSpec(assays=("a", ASSAY_NULL))


# --- heads ---------------------------------------------------------------

def test_ordinal_thresholds_are_strictly_increasing_by_construction():
    heads = heads_lib.ObservationHeads(8, assays=("a", "b"), ordinal_classes=3)
    with torch.no_grad():
        heads.threshold_deltas.fill_(-20.0)      # softplus(-20) is tiny but positive
    thresholds = heads.thresholds()
    assert thresholds.numel() == 2
    assert float(thresholds[1]) > float(thresholds[0])
    with torch.no_grad():
        heads.threshold_deltas.fill_(3.0)
    assert float(heads.thresholds()[1]) > float(heads.thresholds()[0])


def test_ordinal_probabilities_sum_to_one_and_the_loss_is_finite():
    heads = heads_lib.ObservationHeads(8, assays=("a",), ordinal_classes=3)
    representation = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
    classes = torch.tensor([0, 1, 2, 1, 0])
    loss, rows = heads.ordinal_loss(representation, classes, torch.ones(5))
    assert rows == 5
    assert math.isfinite(float(loss))
    loss.backward()
    assert heads.threshold_base.grad is not None


def test_interval_likelihood_handles_both_censored_tails_and_a_point():
    mu = torch.zeros(3)
    scale = torch.ones(3)
    lower = torch.tensor([-1.0, -float("inf"), 0.0])
    upper = torch.tensor([1.0, 0.0, 0.0])
    finite_lower = torch.tensor([True, False, True])
    finite_upper = torch.tensor([True, True, True])
    values = heads_lib.interval_log_probability(mu, scale, lower, upper,
                                               finite_lower=finite_lower,
                                               finite_upper=finite_upper)
    assert torch.isfinite(values).all()
    # A two-sided interval around the mean holds about 68% of the mass.
    assert float(values[0].exp()) == pytest.approx(0.6827, abs=1e-3)
    # A left-censored bound at the mean holds exactly half.
    assert float(values[1].exp()) == pytest.approx(0.5, abs=1e-6)
    # The point case is the Gaussian density at the mean.
    assert float(values[2].exp()) == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=1e-6)


def test_interval_likelihood_stays_finite_deep_in_a_tail():
    values = heads_lib.interval_log_probability(
        torch.zeros(1), torch.ones(1), torch.tensor([20.0]), torch.tensor([21.0]),
        finite_lower=torch.tensor([True]), finite_upper=torch.tensor([True]))
    assert torch.isfinite(values).all()
    assert float(values[0]) < -100.0


def test_right_censoring_prefers_a_larger_prediction():
    heads = heads_lib.ObservationHeads(4, assays=("a",))
    representation = torch.zeros(1, 4)
    assay = torch.zeros(1, dtype=torch.long)
    # pKD >= 9.5 : the loss must fall as the prediction rises toward the bound.
    losses = []
    for bias in (0.0, 5.0, 9.0):
        with torch.no_grad():
            heads.score.bias.fill_(bias)
        loss, _ = heads.interval_loss(representation, torch.tensor([9.5]),
                                      torch.tensor([float("inf")]),
                                      torch.tensor([True]), torch.tensor([False]),
                                      assay, torch.ones(1))
        losses.append(float(loss))
    assert losses[0] > losses[1] > losses[2]


def test_a_masked_empty_head_returns_a_differentiable_zero():
    heads = heads_lib.ObservationHeads(4, assays=("a",))
    representation = torch.randn(3, 4, requires_grad=True)
    loss, rows = heads.ordinal_loss(representation, torch.zeros(3, dtype=torch.long),
                                    torch.zeros(3))
    assert rows == 0 and float(loss) == 0.0
    loss.backward()
    # A torch.zeros would have detached here and the head would silently stop training.
    assert representation.grad is not None
    assert heads.score.weight.grad is not None


def test_enrichment_trains_an_ordering_not_a_scale():
    heads = heads_lib.ObservationHeads(4, assays=("a",))
    representation = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))
    loss, pairs = heads.enrichment_loss(representation, torch.tensor([0, 2]),
                                        torch.tensor([1, 3]), torch.ones(2))
    assert pairs == 2 and math.isfinite(float(loss))
    # Adding a constant to every latent leaves a ranking loss unchanged.
    with torch.no_grad():
        heads.score.bias.fill_(7.0)
    shifted, _ = heads.enrichment_loss(representation, torch.tensor([0, 2]),
                                       torch.tensor([1, 3]), torch.ones(2))
    assert float(shifted) == pytest.approx(float(loss), rel=1e-5)


def test_accumulator_refuses_an_always_empty_metric():
    accumulator = heads_lib.HeadAccumulator()
    accumulator.add(heads_lib.KIND_ORDINAL, 1.5, 10)
    assert accumulator.mean(heads_lib.KIND_ORDINAL) == pytest.approx(1.5)
    assert accumulator.mean(heads_lib.KIND_INTERVAL) is None
    assert accumulator.require_nonempty([heads_lib.KIND_ORDINAL])
    with pytest.raises(ValueError, match="zero eligible rows"):
        accumulator.require_nonempty([heads_lib.KIND_INTERVAL])
    assert accumulator.document()["mean_loss"][heads_lib.KIND_INTERVAL] is None


# --- the energy route ----------------------------------------------------

def energy_model(**overrides):
    settings = {"window": 4, "sigma": 1.0, "reg_l2_norm": 1.0,
                "energy": energy_lib.EnergyNetwork(width=16, layers=2, context_width=8),
                "denoiser": energy_lib.Denoiser(width=16, layers=2, context_width=8),
                "context": energy_lib.ContextEncoder(width=8, layers=1, heads=2),
                "denoiser_context": energy_lib.ContextEncoder(width=8, layers=1, heads=2)}
    settings.update(overrides)
    return energy_lib.WalkJumpModel(**settings)


def clean_batch(rows=3, window=4, residues=20):
    indices = torch.arange(rows * window).reshape(rows, window) % residues
    return energy_lib.onehot_window(indices, residues=residues, scale=1.0)


def energy_context(rows=3, window=4):
    """A real paired context over constructs whose edit maps are ``window`` wide."""
    from smallAntibodyGen.experiments.her2_routes_serial import Construct
    constructs = []
    for index in range(rows):
        heavy = "ACDEFGHIKL"[:8] + "AV"[index % 2] + "M"
        constructs.append(Construct(heavy=heavy, light="WYTS",
                                    sites=tuple(("H", position) for position in range(window)),
                                    source="test", record_id=f"r{index}"))
    return energy_lib.build_energy_context(constructs)


def test_energy_loss_matches_the_reference_math():
    """The production loss must equal the standard-library transcription, exactly.

    Two generators seeded identically make the noise draw reproducible, so this is an
    exact comparison rather than a formula-shaped one. If Codex's reading of the pinned
    ``_noise_ebm.py`` bytes is wrong, this is where it surfaces.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "specs"
                          / "her2_generative_routes"))
    import reference_math as rm
    model = energy_model()
    model.freeze_denoiser()
    clean = clean_batch()
    fake = clean + 0.2
    production = torch.Generator().manual_seed(7)
    loss, diagnostics = model.energy_loss(clean, fake=fake, generator=production)
    mirror = torch.Generator().manual_seed(7)
    with torch.no_grad():
        real_energy = model.energy(model.apply_noise(clean.clone(), generator=mirror), None)
        fake_energy = model.energy(fake, None)
    expected = rm.dwjs_energy_loss([float(value) for value in real_energy],
                                   [float(value) for value in fake_energy],
                                   reg_l2_norm=1.0, cdiv_coeff=1.0)
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert diagnostics["cdiv_coeff"] == 1.0
    assert diagnostics["reg_l2_norm"] == 1.0
    assert "E(noisy_real).mean() - E(fake).mean()" in diagnostics["formula"]
    # And the recorded diagnostics reconstruct the same number, so an artifact reader
    # can re-derive the loss without rerunning the model.
    assert float(loss) == pytest.approx(
        diagnostics["cdiv_loss"] + 1.0 * diagnostics["regulariser"], rel=1e-5)


def test_energy_objective_refuses_an_unfrozen_denoiser():
    model = energy_model()
    with pytest.raises(ValueError, match="FROZEN denoiser"):
        model.energy_loss(clean_batch(), fake=clean_batch())
    model.freeze_denoiser()
    assert model.denoiser_is_frozen()
    loss, _ = model.energy_loss(clean_batch(), fake=clean_batch() + 0.1)
    assert torch.isfinite(loss)


def test_energy_gradient_does_not_reach_the_frozen_denoiser_or_its_context():
    """Freezing the denoiser has to freeze everything the denoiser reads.

    With a shared context tower the 'frozen' denoiser's input would drift under the
    energy's gradient while every parameter check still passed, so the two towers
    are independent objects and both are frozen together.
    """
    model = energy_model()
    model.freeze_denoiser()
    assert model.denoiser_context is not model.context_encoder
    loss, _ = model.energy_loss(clean_batch(), fake=clean_batch() + 0.1)
    model.zero_grad(set_to_none=True)
    loss.backward()
    for module in (model.denoiser, model.denoiser_context):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
               for parameter in model.energy.parameters())


def test_a_shared_context_tower_is_refused_at_construction():
    tower = energy_lib.ContextEncoder(width=8, layers=1, heads=2)
    with pytest.raises(ValueError, match="INDEPENDENT context towers"):
        energy_lib.WalkJumpModel(window=4, sigma=1.0, context=tower, denoiser_context=tower)


def test_the_contrastive_negative_is_detached():
    """A negative that still carries a graph trains a different objective.

    Upstream ``walk`` returns ``ys.detach()``. An earlier revision of this module
    returned a sample that retained the whole Langevin chain, so the energy loss
    back-propagated through it -- the gradient of a different functional, at a
    memory cost proportional to the walk length.
    """
    model = energy_model()
    model.freeze_denoiser()
    generator = torch.Generator().manual_seed(4)
    seeds = model.random_discrete_seeds(2, generator=generator)
    walked = model.walk(seeds, sampling_options={"steps": 2}, generator=generator)
    assert not walked.requires_grad
    assert walked.grad_fn is None
    live = clean_batch(rows=2).clone().requires_grad_(True)
    loss, record = model.energy_loss(clean_batch(rows=2), fake=live + 0.0)
    loss.backward()
    assert live.grad is None
    assert record["negatives_detached"]


def test_the_walk_noises_its_seed_before_the_chain():
    """Upstream's ``walk`` calls ``apply_noise`` on the seed. It is not decoration.

    A Langevin chain started exactly on the data manifold is not sampling the
    smoothed density the energy was trained on, and the first step's score is
    evaluated somewhere the model never saw noise.
    """
    model = energy_model()
    seeds = model.random_discrete_seeds(2, generator=torch.Generator().manual_seed(0))
    # Zero Langevin steps is refused, so the initial noise is observed through a
    # one-step walk with the score and the noise masked out at every position.
    walked = model.walk(seeds.clone(), sampling_options={"steps": 1},
                        generator=torch.Generator().manual_seed(0),
                        mask_positions=[1, 2, 3])
    assert not torch.allclose(walked[:, 0, :], seeds[:, 0, :])
    # A preserved position is restored exactly, which is what upstream's mask does.
    assert torch.equal(walked[:, 1:, :], seeds[:, 1:, :])


def test_denoiser_loss_is_its_own_objective_and_trains_only_the_denoiser():
    model = energy_model()
    clean = clean_batch()
    loss = model.denoiser_loss(clean)
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
               for parameter in model.denoiser.parameters())
    assert all(parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
               for parameter in model.energy.parameters())


def test_score_is_the_negative_energy_gradient():
    model = energy_model()
    y = clean_batch(rows=2).clone().requires_grad_(True)
    score = model.score(y, None)
    energy = model.energy(y, None)
    manual, = torch.autograd.grad(energy.sum(), y)
    assert torch.allclose(score, -manual, atol=1e-5)


def test_the_noise_level_is_one_declared_sigma_and_the_amendment_is_recorded():
    """``_base.py`` has been READ and defines no ``renorm_noise_factor``.

    The pinned ``_noise_ebm`` computes ``sigma_ebm / sigma_denoiser`` and feeds it
    to ``self.renorm_noise_factor``, which the released ``_base.py`` does not
    define -- an upstream defect, not a symbol to guess. This route carries one
    matched sigma, which is the only setting in which the two energies it compares
    are energies of the same density.
    """
    model = energy_model(sigma=1.0)
    assert model.sigma == pytest.approx(1.0)
    assert not hasattr(model, "sigma_renorm")
    assert not hasattr(energy_lib, "UNRESOLVED_TRANSCRIPTIONS")
    entry = next(e for e in energy_lib.AMENDMENTS if e["symbol"] == "renorm_noise_factor")
    assert "read" in entry["status"]
    assert "_base.py" in entry["declared_in"]
    assert "no renorm_noise_factor" in entry["evidence"]
    assert "one declared sigma" in entry["amendment"].lower() or \
        "ONE noise level" in entry["amendment"]
    # A mismatched pair is refused rather than silently renormalised.
    with pytest.raises(ValueError, match="ONE noise level"):
        energy_model(sigma=1.0, sigma_denoiser=0.5)
    # The production module and the standard-library reference agree on the record.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "specs"
                          / "her2_generative_routes"))
    import reference_math as rm
    assert ({e["symbol"] for e in energy_lib.AMENDMENTS}
            >= {e["symbol"] for e in rm.AMENDMENTS})


def test_walk_retains_the_upstream_zeta_and_moves_the_state():
    model = energy_model()
    generator = torch.Generator().manual_seed(0)
    seeds = model.random_discrete_seeds(2, generator=generator)
    walked = model.walk(seeds, sampling_options={"steps": 3}, generator=generator)
    assert walked.shape == seeds.shape
    assert not torch.allclose(walked, seeds)
    assert torch.isfinite(walked).all()


def test_walk_respects_masked_positions():
    model = energy_model()
    generator = torch.Generator().manual_seed(0)
    seeds = model.random_discrete_seeds(2, generator=generator)
    walked = model.walk(seeds.clone(), sampling_options={"steps": 3}, generator=generator,
                       mask_positions=[0])
    # A preserved position takes no score and no noise during the chain AND is
    # restored from the seed at the end, which is both halves of what upstream's
    # ``walkjump`` does with ``mask_idxs``.
    assert torch.equal(walked[:, 0, :], seeds[:, 0, :])
    assert not torch.allclose(walked[:, 1, :], seeds[:, 1, :])


# --- the energy route's context ------------------------------------------

def test_the_context_encoder_is_position_and_chain_aware():
    """A permutation-invariant bag of residues cannot represent a coordinate.

    The earlier encoder was a per-token MLP followed by a mean: reordering the
    construct changed its output by 1.19e-7, which is rounding. A variable edit map
    is nothing but coordinates, so an invariant encoder cannot learn what the route
    needs it to.
    """
    torch.manual_seed(0)
    encoder = energy_lib.ContextEncoder(width=8, layers=1, heads=2)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    report = encoder.permutation_sensitivity(energy_context())
    assert not report["permutation_invariant"]
    assert report["max_abs_difference"] > 1e-4
    assert "cannot represent a coordinate" in report["meaning"]


def test_the_context_carries_coordinates_chains_and_masked_edit_sites():
    context = energy_context(rows=2, window=3)
    tokens = context.tokens
    chains = context.chains
    # The heavy chain comes first, the light chain second, and both are marked.
    assert int(chains[0][0]) == energy_lib.CHAIN_HEAVY
    assert int(chains[0][10]) == energy_lib.CHAIN_LIGHT
    # The three edit coordinates are MASKED in the visible context, so the energy
    # cannot read the answer off its own conditioning input.
    for slot in context.site_index[0].tolist():
        assert int(tokens[0][slot]) == energy_lib.CONTEXT_MASK_ID
    assert context.sites == 3
    # Within-chain coordinates restart at the light chain.
    assert int(context.coordinates[0][10]) == 0


def test_the_energy_reads_the_context_and_changing_it_changes_the_energy():
    torch.manual_seed(1)
    model = energy_model()
    with torch.no_grad():
        for parameter in model.context_encoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    clean = clean_batch(rows=3, window=4)
    context = energy_context(rows=3, window=4)
    with torch.no_grad():
        conditioned = model.energy(clean, *model.encode_context(context))
        unconditioned = model.energy(clean, None, None, None)
    assert not torch.allclose(conditioned, unconditioned)
    # Two rows whose constructs differ get different energies for the same window.
    assert float((conditioned[0] - conditioned[1]).abs()) > 0.0


def test_the_density_only_arm_is_the_same_code_path():
    arm = energy_model().density_only_arm()
    assert arm["arm"] == "density_only"
    assert "context=None" in arm["difference_from_full"]
    assert "not implied" in arm["not"]


def test_jump_and_sample_return_canonical_indices():
    model = energy_model()
    model.freeze_denoiser()
    generator = torch.Generator().manual_seed(0)
    drawn = model.sample(3, sampling_options={"steps": 2}, generator=generator)
    assert drawn.shape == (3, model.window)
    assert int(drawn.min()) >= 0 and int(drawn.max()) < model.residues
    assert drawn.dtype == torch.int64


def test_the_energy_route_claims_no_likelihood():
    document = energy_model().document()
    assert "none" in document["likelihood"]
    assert "normalised discrete sequence likelihood" in document["likelihood"]
    assert document["sampler"].startswith("sachsetal")
    assert "constant for fixed-HER2" in document["fixed_her2_limitation"]


def test_memory_report_is_computed_and_names_what_it_excludes():
    report = energy_model().memory_report()
    assert report["parameters"] > 0
    assert report["fp32_weights_grads_two_moments_mib"] == pytest.approx(
        report["trainable_parameters"] * 16 / 2 ** 20)
    assert "Langevin trajectory" in report["excludes"]
    assert report["published_bytenet_profile_mb"] == 4735


# --- hooks ---------------------------------------------------------------

class _Block(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)

    def forward(self, hidden, *_args, **_kwargs):
        return self.linear(hidden), None


def test_activation_names_are_stable_and_validated():
    assert hooks_lib.activation_name("piggen", 3, "resid_pre") == "route/piggen/block3/resid_pre"
    with pytest.raises(ValueError, match="Unknown activation site"):
        hooks_lib.activation_name("piggen", 0, "invented")
    with pytest.raises(ValueError, match="non-negative int"):
        hooks_lib.activation_name("piggen", -1, "resid_pre")


def test_registry_captures_and_then_restores_the_module():
    block = _Block()
    registry = hooks_lib.HookRegistry(route="toy")
    registry.register("route/toy/block0/resid_post", block, site="resid_post")
    hidden = torch.randn(2, 3, 4)
    baseline = block(hidden)[0]
    registry.enable()
    output = block(hidden)[0]
    registry.disable()
    assert torch.allclose(output, baseline)
    assert "route/toy/block0/resid_post" in registry.captured
    after = block(hidden)[0]
    assert torch.allclose(after, baseline)


def test_zero_and_mean_ablation_differ_and_a_patch_overrides():
    block = _Block()
    registry = hooks_lib.HookRegistry(route="toy")
    name = "route/toy/block0/resid_post"
    registry.register(name, block, site="resid_post")
    hidden = torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(0))
    registry.plan(name, kind="zero_ablate")
    registry.enable([name])
    zeroed = block(hidden)[0]
    registry.disable()
    assert float(zeroed.abs().sum()) == 0.0
    # Mean ablation needs an explicit FROZEN donor mean (it no longer averages the batch it
    # is evaluating); the donors here are a separate set of rows.
    donors = torch.randn(5, 2, 4, generator=torch.Generator().manual_seed(2))
    reference = hooks_lib.reference_mean(donors, name=name, layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    registry.clear_plans().plan(name, kind="mean_ablate", reference=reference)
    registry.enable([name])
    meaned = block(hidden)[0]
    registry.disable()
    assert float(meaned.abs().sum()) > 0.0
    # Every batch row now carries the donor mean, so the rows are identical to it.
    assert torch.allclose(meaned[0], meaned[1])
    assert torch.allclose(meaned[0], donors.mean(dim=0), atol=1e-6)
    # A patch source carries its layout; a bare tensor is refused (see test_her2_functional_repairs).
    source = hooks_lib.PatchSource(tensor=torch.ones(1, 2, 4), layout=hooks_lib.LAYOUT_TOKENS_FIRST)
    registry.clear_plans().plan(name, kind="patch", source=source)
    registry.enable([name])
    patched = block(hidden)[0]
    registry.disable()
    assert torch.allclose(patched, torch.ones(3, 2, 4))


def test_a_positional_intervention_touches_only_those_positions():
    block = _Block()
    registry = hooks_lib.HookRegistry(route="toy")
    name = "route/toy/block0/resid_post"
    registry.register(name, block, site="resid_post")
    hidden = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(1))
    baseline = block(hidden)[0]
    registry.plan(name, kind="zero_ablate", positions=[1])
    registry.enable([name])
    output = block(hidden)[0]
    registry.disable()
    assert float(output[:, 1].abs().sum()) == 0.0
    assert torch.allclose(output[:, 0], baseline[:, 0])
    assert torch.allclose(output[:, 2], baseline[:, 2])


def test_token_map_must_cover_every_tensor_index():
    rows = [{"index": 0}, {"index": 1}]
    assert hooks_lib.require_token_map(rows, 2)
    with pytest.raises(ValueError, match="mapped indices against a tensor"):
        hooks_lib.require_token_map(rows, 3)
    with pytest.raises(ValueError, match="not 0..n-1"):
        hooks_lib.require_token_map([{"index": 1}, {"index": 0}], 2)


class _ConvBlock(torch.nn.Module):
    """A block whose output is ``(batch, channels, positions)``, like the energy stack."""

    def __init__(self, channels=4):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, hidden):
        return self.conv(hidden)


def test_a_positional_intervention_addresses_the_declared_sequence_axis():
    """On a (batch, channels, positions) tensor, dim 1 is a CHANNEL, not a residue.

    Addressing it as if it were tokens zeroes channel 1 at every residue and leaves
    a figure that looks exactly like a successful position ablation.
    """
    block = _ConvBlock(channels=4)
    registry = hooks_lib.HookRegistry(route="dwjs")
    name = "route/dwjs/block0/mlp_out"
    registry.register(name, block, site="mlp_out", layout=hooks_lib.LAYOUT_TOKENS_LAST)
    assert registry.position_axis(name) == 2
    hidden = torch.ones(2, 4, 3)
    registry.plan(name, kind="zero_ablate", positions=[1])
    registry.enable([name])
    output = block(hidden)
    registry.disable()
    # Residue 1 is zero at EVERY channel; every other residue is untouched.
    assert float(output[:, :, 1].abs().sum()) == 0.0
    assert float(output[:, :, 0].abs().sum()) > 0.0
    assert float(output[:, :, 2].abs().sum()) > 0.0
    # And channel 1 is NOT wholesale zero, which is what the axis bug produced.
    assert float(output[:, 1, :].abs().sum()) > 0.0


def test_a_tap_without_a_known_layout_is_refused_and_the_layout_travels():
    block = _Block()
    registry = hooks_lib.HookRegistry(route="toy")
    with pytest.raises(ValueError, match="Unknown tensor layout"):
        registry.register("route/toy/block0/resid_post", block, site="resid_post",
                          layout="whatever")
    name = registry.register("route/toy/block0/resid_post", block, site="resid_post")
    registry.enable([name])
    block(torch.randn(2, 3, 4))
    registry.disable()
    exported = registry.export(name)
    assert exported["layout"] == hooks_lib.LAYOUT_TOKENS_FIRST
    assert exported["position_axis"] == 1
    assert exported["shape"] == [2, 3, 4]
    assert registry.document()["position_axes"][name] == 1


def test_an_out_of_range_position_is_refused_rather_than_silently_clamped():
    block = _ConvBlock(channels=4)
    registry = hooks_lib.HookRegistry(route="dwjs")
    name = registry.register("route/dwjs/block0/mlp_out", block, site="mlp_out",
                             layout=hooks_lib.LAYOUT_TOKENS_LAST)
    registry.plan(name, kind="zero_ablate", positions=[9])
    registry.enable([name])
    with pytest.raises(ValueError, match="outside the"):
        block(torch.ones(1, 4, 3))
    registry.disable()


# --- the autoregressive token map ----------------------------------------

@pytest.fixture
def tiny_causal_serialization():
    """A four-residue heavy chain, a two-residue light chain and two edit sites.

    ``bind_piggen_vocabulary`` writes a module-global table, so the previous
    binding is restored afterwards: leaving a fake vocabulary behind would make
    every later test in the session depend on collection order.
    """
    from smallAntibodyGen.experiments import her2_routes_serial as serial
    previous = dict(serial.PIGGEN_ID_OF_RESIDUE)
    vocabulary = {residue: index + 3 for index, residue in enumerate(
        "ACDEFGHIKLMNPQRSTVWY")}
    vocabulary.update({"1": serial.PIGGEN_START_ID, "2": serial.PIGGEN_END_ID,
                       "B": serial.PIGGEN_VOCAB_SIZE - 1})
    serial.bind_piggen_vocabulary(vocabulary)
    construct = serial.Construct(heavy="ACDE", light="FG", sites=(("H", 1), ("L", 0)),
                                 source="toy", record_id="t")
    order = (("H", 1), ("L", 0))
    serialization = serial.causal_serialization(
        construct, order=order, residues=construct.residues(order))
    try:
        yield construct, serialization
    finally:
        serial.PIGGEN_ID_OF_RESIDUE.clear()
        serial.PIGGEN_ID_OF_RESIDUE.update(previous)


def test_the_causal_map_separates_the_token_present_from_the_coordinate_queried(
        tiny_causal_serialization):
    """Slot C predicts order[0]; slot C+1 holds order[0]'s answer and predicts order[1].

    The earlier map recorded slot C as a separator with no query at all and slot
    C+1 as querying order[0] -- so every query in the window was attributed one
    step late, and a 'site (H,1) feature' was actually the feature for (L,0).
    """
    construct, serialization = tiny_causal_serialization
    rows = hooks_lib.causal_token_map(construct, serialization)
    assert hooks_lib.require_token_map(rows, len(serialization.tokens))
    assert serialization.context_length == 8
    assert list(serialization.query_slots) == [8, 9]
    by_index = {row["index"]: row for row in rows}
    # Slot 8 is the separator AND the position whose logits are the conditional for
    # order[0] = (H, 1).
    assert by_index[8]["role"] == hooks_lib.ROLE_SEPARATOR
    assert by_index[8]["query_site"] == ["H", 1]
    assert by_index[8]["query_edit_slot"] == 0
    assert by_index[8]["is_scored_query"]
    # Slot 9 holds the residue emitted FOR (H, 1) and queries (L, 0).
    assert by_index[9]["input_edit_slot"] == 0
    assert by_index[9]["query_site"] == ["L", 0]
    assert by_index[9]["query_edit_slot"] == 1
    # The trailing answer token has an input and no scored query.
    assert by_index[10]["input_edit_slot"] == 1
    assert by_index[10]["query_site"] is None
    # The masked context position for (H, 1) is marked, at its own coordinate.
    assert by_index[2]["role"] == hooks_lib.ROLE_MASK
    assert (by_index[2]["chain"], by_index[2]["residue_index"]) == ("H", 1)
    report = hooks_lib.require_causal_query_alignment(rows, serialization)
    assert report["query_slots"] == 2
    assert report["trailing_unqueried_answers"] == 1


def test_a_shifted_map_is_caught_by_the_alignment_check(tiny_causal_serialization):
    construct, serialization = tiny_causal_serialization
    rows = hooks_lib.causal_token_map(construct, serialization)
    for row in rows:
        if row["query_edit_slot"] == 0:
            row["query_edit_slot"] = 1        # the off-by-one the map used to carry
    with pytest.raises(ValueError, match="the map records step"):
        hooks_lib.require_causal_query_alignment(rows, serialization)


# --- the exposure ledger ------------------------------------------------

def test_ledger_keeps_six_independent_counters():
    ledger = data_lib.ExposureLedger()
    ledger.record(task=data_lib.TASK_RECONSTRUCTION, source="oas", rows=4, forwards=4,
                  targets=40, identities=["a", "b"], update=True)
    ledger.record(task=data_lib.TASK_LOCAL, source="alphabind", rows=4, forwards=8,
                  targets=8, identities=["a", "c"], measured=True, update=True)
    document = ledger.document()
    totals = document["totals"]
    assert totals["optimizer_updates"] == 2
    assert totals["microbatch_rows"] == 8
    assert totals["student_forwards"] == 12        # the contrast is two forwards
    assert totals["unique_constructs"] == 3
    assert totals["measurement_presentations"] == 2
    assert totals["supervised_targets"] == 48
    assert document["forwards_per_row"] == pytest.approx(1.5)
    assert "none derived from another" in document["accounting_rule"]
    assert document["constructs_reused_across_tasks"] == 1


def test_a_task_with_a_big_share_and_no_gradient_is_visible():
    block = data_lib.task_gradient_fractions({"whole_window": 0.03, "reconstruction": 0.97})
    assert block["fractions"]["whole_window"] == pytest.approx(0.03)
    assert "is not teaching" in block["reading"]
    empty = data_lib.task_gradient_fractions({"a": 0.0, "b": 0.0})
    assert empty["fractions"]["a"] is None


def test_task_mixture_must_sum_to_one_and_names_every_task():
    mixture = data_lib.TaskMixture()
    assert sum(mixture.shares.values()) == pytest.approx(1.0)
    schedule = mixture.schedule(100, seed=1)
    assert len(schedule) == 100
    assert set(schedule) <= set(data_lib.TASKS)
    with pytest.raises(ValueError, match="sum to"):
        data_lib.TaskMixture(shares={name: 0.1 for name in data_lib.TASKS})
    with pytest.raises(ValueError, match="must name exactly"):
        data_lib.TaskMixture(shares={data_lib.TASK_RECONSTRUCTION: 1.0})


def test_head_views_are_refused_when_a_quality_bin_leaks_in():
    with pytest.raises(ValueError, match="answer-free stream"):
        data_lib.require_answer_free([Condition(family="affinity", bin=BIN_HIGH,
                                                assay="assay_a")])
    assert data_lib.require_answer_free([Condition(family="affinity", bin=BIN_NULL,
                                                   assay="assay_a")])


def test_pad_rows_refuses_a_width_that_would_truncate():
    padded, mask = data_lib.pad_rows([[1, 2], [1, 2, 3]], value=0)
    assert padded.shape == (2, 3)
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    with pytest.raises(ValueError, match="truncating here would drop a light chain"):
        data_lib.pad_rows([[1, 2, 3]], value=0, width=2)
