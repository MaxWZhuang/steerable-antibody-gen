"""Build B: the learned evidence/intervention model, its frozen tape, arms, evaluation, transfer and bridge.

These tests qualify MECHANICS on tiny CPU models: the student boundary, set-encoder invariances, actual
gradient paths through the shared encoder and the donor/base blocks, the matched A/I tape and compute,
the corrupted-supervision control, resume and tamper refusal, stage and label isolation, the transfer
branch and a REAL Build A matrix over the exported learned activations. They never assert that any arm
wins; a negative or inconclusive scientific result is a valid result. Everything is synthetic.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import os
import shutil
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from smallAntibodyGen.experiments import functional_baseline as fb
from smallAntibodyGen.experiments import functional_causal_learning as fcl
from smallAntibodyGen.experiments import functional_evidence as fe
from smallAntibodyGen.experiments import functional_matrix as fm
from smallAntibodyGen.experiments import functional_shared_transfer as fst
from smallAntibodyGen.experiments.her2_runtime import sha256 as file_sha256

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "functional_causal_learning.py"


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """Bit-reproducible CPU arithmetic for the exact parity assertions."""
    with fst.single_thread():
        yield
    assert not torch.cuda.is_initialized()


def small_config(**changes):
    values = dict(steps=4, batch=8, lr=3e-3, event_hidden=8, block_dim=8, program_embedding=4,
                  program_hidden=8, head_hidden=16, seeds=(0, 1), variants=3, eval_variant=1,
                  checkpoint_every=2, adapt_steps=5, adapter_hidden=4, export_label_budgets=(16, 32),
                  export_acquisition_seeds=(0, 1), export_bootstrap_replicates=10)
    values.update(changes)
    return fcl.CausalConfig(**values)


@pytest.fixture(scope="module")
def context():
    return fcl.TrainContext.build(small_config())


def brute_posterior(events):
    weights = []
    for w in range(16):
        weight = Fraction(1, 16)
        for e in events:
            r = Fraction(e.reliability)
            weight *= r if ((w >> e.fact) & 1) == e.bit else 1 - r
        weights.append(weight)
    total = sum(weights)
    return [x / total for x in weights]


def run_node(node, world):
    if isinstance(node, fe.Read):
        return (world >> node.index) & 1
    if isinstance(node, fe.Not):
        return 1 - run_node(node.child, world)
    if isinstance(node, fe.And):
        return run_node(node.left, world) * run_node(node.right, world)
    return (run_node(node.left, world) + run_node(node.right, world)) % 2


def brute_probability(program, events):
    posterior = brute_posterior(events)
    return sum(posterior[w] for w in range(16) if run_node(program, w))


def untrained_model(ctx, seed=0):
    net = fcl.net_from_state(ctx.spec, ctx.init_states[seed])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return fcl.FrozenModel(run_id=f"scratch_s{seed}", arm=fcl.SCRATCH, seed=seed, net=net, header={},
                           state_sha256=fst.tensor_state_digest(dict(net.state_dict())))


def snapshot(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*"))
            if p.is_file()}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_pilot_defaults_are_the_declared_bounded_pilot():
    config = fcl.pilot_config()
    assert (config.steps, config.batch, config.seeds) == (1000, 64, (0, 1, 2))
    assert config.arms == ("F", "M", "A", "I", "Ic") and config.event_width == 28
    assert config.block_dim in (8, 16) and config.transfer_budgets == (0, 16, 64)
    assert config.model_spec().block_dim == config.block_dim


@pytest.mark.parametrize("document", [
    {"surprise": 1}, {"arms": ["M", "A"]}, {"arms": ["F", "F"]}, {"arms": ["F", "Z"]},
    {"seeds": [0, 0]}, {"seeds": []}, {"block_dim": 12}, {"steps": 0}, {"lam_cf": 0},
    {"transfer_budgets": [16, 64]}, {"transfer_budgets": [0, 500]}, {"eval_variant": 9},
    {"export_label_budgets": [0]}, {"cancel_max": 9}, {"lr": -1.0}, {"variants": 0},
    {"export_bootstrap_replicates": 3}])
def test_config_is_strict(document):
    with pytest.raises(ValueError):
        fcl.CausalConfig.from_document(document)


def test_config_documents_round_trip_and_the_checkpoint_period_is_not_an_identity_choice():
    config = small_config()
    assert fcl.CausalConfig.from_document(config.document()) == config
    assert "checkpoint_every" not in config.identity_document()
    assert "arms" not in config.training_document() and "seeds" not in config.training_document()
    assert json.loads(json.dumps(config.document())) == config.document()


# ---------------------------------------------------------------------------
# the student boundary and the network
# ---------------------------------------------------------------------------

def test_the_student_sees_only_raw_events_and_program_tokens():
    assert {f.name for f in dataclasses.fields(fcl.EventBatch)} == {"fact", "bit", "reliability",
                                                                    "mask"}
    assert {f.name for f in dataclasses.fields(fcl.ProgramBatch)} == {"tokens", "length"}
    assert list(inspect.signature(fcl.CausalNet.forward).parameters) == ["self", "batch"]
    assert list(inspect.signature(fcl.CausalNet.encode).parameters) == ["self", "events"]
    with pytest.raises(ValueError):
        fcl.make_net(small_config().model_spec(), 0).encode("events")
    with pytest.raises(ValueError):
        fcl.make_net(small_config().model_spec(), 0)(object())


def test_event_identity_never_reaches_the_student(context):
    events = fe.raw_events((4, 2, 1, 5), 1, prefix="first")
    renamed = tuple(fe.Observation(f"other.{i}", e.fact, e.bit, e.reliability)
                    for i, e in enumerate(events))
    a, b = fcl.events_to_arrays([events], 28), fcl.events_to_arrays([renamed], 28)
    assert all(np.array_equal(a[k], b[k]) for k in fcl.EVENT_FIELDS)


def test_training_forwards_receive_only_event_and_program_batches(context, monkeypatch):
    seen = []
    encode, program = fcl.CausalNet.encode, fcl.CausalNet.encode_program

    def spy_encode(self, events):
        seen.append(type(events))
        return encode(self, events)

    def spy_program(self, batch):
        seen.append(type(batch))
        return program(self, batch)
    monkeypatch.setattr(fcl.CausalNet, "encode", spy_encode)
    monkeypatch.setattr(fcl.CausalNet, "encode_program", spy_program)
    for arm in fcl.ARMS:
        fcl.ArmRun(context, arm, 0).step()
    assert set(seen) == {fcl.EventBatch, fcl.ProgramBatch}


def test_the_global_rng_is_not_advanced_by_building_a_network():
    before = torch.get_rng_state().clone()
    fcl.initial_state(small_config().model_spec(), 3)
    fcl.make_net(small_config().model_spec(), 5)
    assert torch.equal(torch.get_rng_state(), before)


def test_soft_label_cross_entropy_is_stable_and_matches_the_reference():
    logits = torch.tensor([-100.0, -3.0, 0.0, 2.5, 100.0])
    targets = torch.tensor([0.3, 0.0, 0.5, 1.0, 0.9])
    value = fcl.soft_bce(logits, targets)
    reference = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.isfinite(value).all() and torch.allclose(value, reference, atol=1e-5)
    p = torch.sigmoid(torch.tensor([-3.0, 0.0, 2.5], dtype=torch.float64))
    t = torch.tensor([0.2, 0.5, 0.7], dtype=torch.float64)
    plain = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    assert torch.allclose(fcl.soft_bce(torch.tensor([-3.0, 0.0, 2.5], dtype=torch.float64), t), plain)


def make_batch(rows, width=28):
    arrays = fcl.events_to_arrays(rows, width)
    return arrays


def test_the_event_encoder_is_order_padding_and_batch_composition_invariant():
    net = fcl.make_net(small_config().model_spec(), 1)
    rows = [fe.raw_events((4, 2, 1, 5), 1), fe.raw_events((0, 6, 3, 3), 2), fe.raw_events((3, 3, 3, 3), 0)]
    arrays = make_batch(rows)
    with torch.no_grad():
        reference = net.encode(fcl.event_batch(*(arrays[k] for k in fcl.EVENT_FIELDS)))
        rng = np.random.default_rng(0)
        order = np.argsort(rng.random(arrays["mask"].shape) + (~arrays["mask"]) * 10.0, axis=1)
        permuted = {k: np.take_along_axis(arrays[k], order, axis=1) for k in fcl.EVENT_FIELDS}
        shuffled = net.encode(fcl.event_batch(*(permuted[k] for k in fcl.EVENT_FIELDS)))
        wide = fcl.events_to_arrays(rows, 40)
        padded = net.encode(fcl.event_batch(*(wide[k] for k in fcl.EVENT_FIELDS)))
        alone = torch.cat([net.encode(fcl.event_batch(*(arrays[k][i:i + 1] for k in fcl.EVENT_FIELDS)))
                           for i in range(3)])
        backwards = net.encode(fcl.event_batch(*(arrays[k][::-1].copy() for k in fcl.EVENT_FIELDS)))
    for other in (shuffled, padded, alone, backwards.flip(0)):
        assert torch.allclose(reference, other, atol=1e-4)


def test_the_program_encoder_ignores_right_padding_and_batch_composition():
    net = fcl.make_net(small_config().model_spec(), 1)
    functions = fcl.function_arrays()
    tokens, length = np.asarray(functions["tokens"]), np.asarray(functions["length"])
    with torch.no_grad():
        base = net.encode_program(fcl.program_batch(tokens, length))
        wide = net.encode_program(fcl.program_batch(
            np.concatenate([tokens, np.zeros((len(tokens), 4), dtype=tokens.dtype)], axis=1), length))
        half = net.encode_program(fcl.program_batch(tokens[:20], length[:20]))
    assert torch.allclose(base, wide, atol=1e-4) and torch.allclose(base[:20], half, atol=1e-4)


def test_identity_insertion_is_exactly_ordinary_prediction_and_removal_restores_it(context):
    run = fcl.ArmRun(context, "I", 0)
    update, _ = run.consumed(0)
    net = run.net
    with torch.no_grad():
        blocks = net.encode(fcl._stack(update, "base"))
        state = net.encode_program(fcl.program_batch(update["program_tokens"],
                                                      update["program_length"]))
        ordinary, _ = net.predict(blocks, state)
        for site in range(4):
            same = fcl.CausalNet.patch_blocks(blocks, blocks,
                                               torch.full((blocks.shape[0],), site, dtype=torch.int64))
            assert torch.equal(same, blocks)
            patched, _ = net.predict(same, state)
            assert torch.equal(patched, ordinary)
        donor = net.encode(fcl._stack(update, "donor"))
        site = torch.tensor(update["site"], dtype=torch.int64)
        patched_blocks = fcl.CausalNet.patch_blocks(blocks, donor, site)
        chosen = F.one_hot(site, 4).bool()
        assert torch.equal(patched_blocks[chosen], donor[chosen])
        assert torch.equal(patched_blocks[~chosen], blocks[~chosen])
        restored, _ = net.predict(blocks, state)
        assert torch.equal(restored, ordinary)


def test_removing_the_marginal_heads_does_not_change_ordinary_predictions(context):
    run = fcl.ArmRun(context, "M", 0)
    for _ in range(2):
        run.step()
    update, _ = run.consumed(2)
    batch = fcl.StudentBatch(events=fcl._stack(update, "base"),
                             program=fcl.program_batch(update["program_tokens"],
                                                       update["program_length"]))
    with torch.no_grad():
        before = run.net(batch)
    run.net.marginal_heads = torch.nn.ModuleList()
    with torch.no_grad():
        assert torch.equal(run.net(batch), before)


def test_actual_gradient_paths_through_the_shared_encoder_the_donor_and_the_base_blocks(context):
    run = fcl.ArmRun(context, "I", 0)
    update, _ = run.consumed(0)
    net = run.net
    blocks_base = net.encode(fcl._stack(update, "base"))
    blocks_base.retain_grad()
    donor = net.encode(fcl._stack(update, "donor"))
    donor.retain_grad()
    state = net.encode_program(fcl.program_batch(update["program_tokens"], update["program_length"]))
    site = torch.tensor(update["site"], dtype=torch.int64)
    logit, _ = net.predict(fcl.CausalNet.patch_blocks(blocks_base, donor, site), state)
    loss = fcl.soft_bce(logit, torch.tensor(update["p_cf"], dtype=torch.float32)).mean()
    loss.backward()
    chosen = F.one_hot(site, 4).bool()
    assert torch.all(blocks_base.grad[chosen] == 0), "the replaced base block must be bypassed"
    assert blocks_base.grad[~chosen].abs().sum() > 0
    assert donor.grad[chosen].abs().sum() > 0, "the donor block must stay differentiable"
    assert torch.all(donor.grad[~chosen] == 0)
    for name, parameter in net.named_parameters():
        if not name.startswith("marginal_heads"):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None


def test_input_augmentation_sends_its_counterfactual_gradient_through_the_hybrid_encode_only(context):
    run = fcl.ArmRun(context, "A", 0)
    update, _ = run.consumed(0)
    net = run.net
    blocks_base = net.encode(fcl._stack(update, "base"))
    blocks_base.retain_grad()
    hybrid = net.encode(fcl._stack(update, "hybrid"))
    hybrid.retain_grad()
    state = net.encode_program(fcl.program_batch(update["program_tokens"], update["program_length"]))
    logit, _ = net.predict(hybrid, state)
    fcl.soft_bce(logit, torch.tensor(update["p_cf"], dtype=torch.float32)).mean().backward()
    assert blocks_base.grad is None and hybrid.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# the event bank, the frozen tape and the world it is built from
# ---------------------------------------------------------------------------

def test_the_event_bank_holds_the_actual_raw_events(context):
    bank, events = context.bank, context.bank_events
    world = fe.q7_world()
    assert bank.arrays["mask"].shape == (1176 * 3, 28) and len(events) == 1176 * 3
    for row in (0, 1, 2, 500, 3527):
        decoded = fcl.arrays_to_events({k: bank.arrays[k] for k in fcl.EVENT_FIELDS}, row, "x")
        assert sorted((e.fact, e.bit, e.reliability) for e in decoded) == sorted(
            (e.fact, e.bit, e.reliability) for e in events[row])
        assert fe.signature_of_events(events[row]) == world.source[row // 3]
    with pytest.raises(ValueError):
        bank.arrays["fact"][0, 0] = 1                     # a read-only snapshot


def test_vectorized_hybrids_equal_the_tuple_hybrids(context):
    bank, events = context.bank, context.bank_events
    rng = np.random.default_rng(1)
    base_rows = rng.integers(0, len(events), 60)
    donor_rows = rng.integers(0, len(events), 60)
    site = rng.integers(0, 4, 60).astype(np.int8)
    arrays = fcl.compose_hybrid(bank.take(base_rows), bank.take(donor_rows), site)
    for i in range(60):
        expected = fe.hybrid_events(events[base_rows[i]], events[donor_rows[i]], int(site[i]))
        got = fcl.arrays_to_events(arrays, i, "h")
        assert sorted((e.fact, e.bit, e.reliability) for e in got) == sorted(
            (e.fact, e.bit, e.reliability) for e in expected)
        assert arrays["mask"][i].sum() == len(expected)


def test_donor_candidates_are_the_legal_levels_and_exactly_672():
    table = fcl.donor_candidates()
    world = fe.q7_world()
    assert table.shape == (1176, 4, 672)
    rng = np.random.default_rng(2)
    for b in rng.integers(0, 1176, 12):
        for site in range(4):
            legal = set(world.legal_levels(world.source[b], site))
            donors = table[b, site]
            assert len(set(donors.tolist())) == 672
            assert {world.source[d][site] for d in donors} == legal


def test_every_tape_row_is_source_only_and_every_hybrid_is_legal(context):
    world = fe.q7_world()
    arrays = context.tapes[0].arrays
    sites = set()
    for j in range(len(arrays["site"])):
        base, donor = world.source[int(arrays["base_sig"][j])], world.source[int(arrays["donor_sig"][j])]
        site = int(arrays["site"][j])
        sites.add(site)
        hybrid = fe.hybrid_signature(base, donor, site)
        assert hybrid in world.source_index and world.source_index[hybrid] == int(arrays["hybrid_sig"][j])
        assert donor[site] in world.legal_levels(base, site)
        assert 0 <= int(arrays["function"][j]) < 48
    assert sites == {0, 1, 2, 3}


def test_tape_labels_are_the_independent_teachers(context):
    registry = fe.function_registry()
    arrays, events = context.tapes[0].arrays, context.bank_events
    variants = context.config.variants
    for j in range(0, len(arrays["site"]), 5):
        brow = int(arrays["base_sig"][j]) * variants + int(arrays["base_var"][j])
        drow = int(arrays["donor_sig"][j]) * variants + int(arrays["donor_var"][j])
        program = registry.functions[int(arrays["function"][j])].program
        hybrid = fe.hybrid_events(events[brow], events[drow], int(arrays["site"][j]))
        assert float(brute_probability(program, events[brow])) == float(arrays["p_base"][j])
        assert float(brute_probability(program, hybrid)) == float(arrays["p_cf"][j])
        posterior = brute_posterior(events[brow])
        for i in range(4):
            assert float(sum(posterior[w] for w in range(16) if (w >> i) & 1)) == float(
                arrays["marg_base"][j, i])


def test_deep_verification_recomputes_every_digest_and_label(context):
    for seed in context.config.seeds:
        assert fcl.verify_tape(context.tapes[seed], context.bank, context.bank_events, context.config,
                               deep=True)


def test_tapes_are_deterministic_and_differ_by_seed_and_the_padded_width_is_fixed(context):
    again = fcl.build_tape(context.config, 0, context.bank, context.bank_events)
    assert again.sha256 == context.tapes[0].sha256 and context.tapes[0].sha256 != context.tapes[1].sha256
    update = fcl.update_arrays(context.bank, fcl.function_arrays(), context.tapes[0].arrays, 0, 8)
    for stream in ("base", "donor", "hybrid"):
        assert update[f"{stream}_mask"].shape == (8, 28)
    useful = context.tapes[0].arrays["useful"]
    assert useful.shape == (4, 3)
    assert int(useful[0, 0]) == int(update["base_mask"].sum())
    assert int(useful[0, 1]) == int(update["donor_mask"].sum())
    assert int(useful[0, 2]) == int(update["hybrid_mask"].sum())
    assert (useful[:, :3] <= 28 * 8).all() and (useful >= 0).all()


def stratum_column(counts):
    """A stratum's target column with the given ``{value: count}`` histogram in a fixed scrambled order."""
    column = np.array([value for value, count in counts.items() for _ in range(count)], dtype=np.float64)
    return column[np.random.default_rng(0).permutation(column.shape[0])]


def is_value_relabelling(old, new):
    """True when ``new`` is a FUNCTION of ``old`` (every old value always becomes one new value): the
    identity, a Boolean complement, a rotation of the values or any other fixed rule a model could learn."""
    mapping = {}
    for before, after in zip(np.round(old, 12), np.round(new, 12)):
        if mapping.setdefault(float(before), float(after)) != float(after):
            return False
    return True


def test_a_balanced_two_value_stratum_is_corrupted_partially_at_random_never_complemented():
    """The old derangement turned ``[.25, .25, .75, .75]`` into its Boolean complement every time while
    reporting it supported. A complement is a learnable rule; the control must randomize per example."""
    old = np.array([0.25, 0.25, 0.75, 0.75])
    outcomes = set()
    for seed in range(40):
        new, record = fcl._randomize_stratum(old, np.random.default_rng(seed))
        assert record["status"] == "supported" and record["classes_split"] >= 1
        assert sorted(new.tolist()) == sorted(old.tolist())
        assert not is_value_relabelling(old, new)
        assert not np.array_equal(new, 1 - old) and not np.array_equal(new, old)
        assert record["changed"] == 2 and record["changed_fraction"] == 0.5
        outcomes.add(tuple(new.tolist()))
    assert len(outcomes) > 1, "the corruption must be random, not one fixed rearrangement"
    assert record["descriptive_max_changed"] == 4 > record["changed"]     # the maximum is only a bound


@pytest.mark.parametrize("counts", [
    {0.1: 3, 0.2: 3, 0.3: 3}, {0.1: 4, 0.2: 4, 0.3: 4, 0.4: 4}, {0.1: 2, 0.5: 2, 0.9: 2},
    {0.2: 5, 0.6: 3, 0.8: 3}, {0.25: 6, 0.75: 6}])
def test_multi_value_strata_are_never_left_as_a_fixed_value_relabelling(counts):
    """Equal-size classes made the old rotation a deterministic value cycle (the probes reproduced a fixed
    three-value cycle for every seed). Whatever the draw, a supported stratum must split a repeated value."""
    old = stratum_column(counts)
    seen = set()
    for seed in range(30):
        new, record = fcl._randomize_stratum(old, np.random.default_rng(seed))
        assert record["status"] == "supported" and record["classes_split"] >= 1
        assert not is_value_relabelling(old, new)
        assert sorted(new.tolist()) == sorted(old.tolist())
        assert 2 <= record["changed"] <= old.shape[0]
        seen.add(tuple(new.tolist()))
    assert len(seen) > 1


def test_a_dominant_value_stratum_is_randomized_and_the_maximum_is_only_descriptive():
    old = stratum_column({0.5: 7, 0.2: 3})
    for seed in range(30):
        new, record = fcl._randomize_stratum(old, np.random.default_rng(seed))
        assert record["status"] == "supported" and record["descriptive_max_changed"] == 6
        assert sorted(new.tolist()) == sorted(old.tolist()) and (np.round(new, 12) == 0.5).sum() == 7
        assert not is_value_relabelling(old, new)
        assert 2 <= record["changed"] <= record["descriptive_max_changed"]
        assert record["changed"] == int((np.round(new, 12) != np.round(old, 12)).sum())


@pytest.mark.parametrize("old,kind", [
    (np.array([]), "unsupported_constant"), (np.array([0.3]), "unsupported_constant"),
    (np.array([0.4, 0.4, 0.4, 0.4]), "unsupported_constant"),
    (np.array([0.3, 0.3, 0.7]), "unsupported_small"),
    (np.array([0.1, 0.2, 0.3, 0.4]), "unsupported_singleton_classes"),
    (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), "unsupported_singleton_classes")])
def test_strata_that_cannot_be_randomized_meaningfully_are_reported_and_left_alone(old, kind):
    new, record = fcl._randomize_stratum(old, np.random.default_rng(0))
    assert record["status"] == "unsupported" and record["kind"] == kind and record["reason"]
    assert record["changed"] == 0 and record["classes_split"] == 0 and record["descriptive_max_changed"] == 0
    assert np.array_equal(new, old)


def test_corruption_is_reproducible_exact_in_every_histogram_and_seed_dependent():
    rng = np.random.default_rng(9)
    p = rng.choice([0.1, 0.25, 0.5, 0.75, 0.9], size=600)
    strata = rng.integers(0, 12, 600)
    orbit = {int(s): int(s) // 4 for s in np.unique(strata)}
    first, report = fcl.corrupt_targets(p, strata, orbit, np.random.default_rng(1))
    again, again_report = fcl.corrupt_targets(p, strata, orbit, np.random.default_rng(1))
    other, _ = fcl.corrupt_targets(p, strata, orbit, np.random.default_rng(2))
    assert np.array_equal(first, again) and report == again_report
    assert not np.array_equal(first, other)
    assert sorted(first.tolist()) == sorted(p.tolist())
    for s in np.unique(strata):
        mask = strata == s
        assert sorted(first[mask].tolist()) == sorted(p[mask].tolist())
    assert report["changed_values"] == int((np.round(first, 12) != np.round(p, 12)).sum()) > 0
    assert abs(report["changed_fraction"] + report["unchanged_fraction"] - 1.0) < 1e-12
    assert 0.3 < report["changed_fraction"] < 1.0            # randomized, so not every example changes
    assert report["mean_abs_difference"] > 0 and report["mean_kl_true_to_corrupted"] > 0
    assert report["histogram_preserved"] is True and report["definition"]
    assert report["supported_strata"] + len(report["unsupported_strata"]) == report["strata"] == 12
    assert report["events_in_supported_strata"] + report["events_in_unsupported_strata"] == 600
    assert "every_supported_stratum_at_its_maximum" not in report and report["descriptive_max_changed"] > 0
    for record in report["stratum_records"]:
        mask = strata == record["stratum"]
        assert record["n"] == int(mask.sum())
        if record["status"] == "supported":
            assert record["classes_split"] >= 1 and not is_value_relabelling(p[mask], first[mask])


def test_tiny_and_constant_strata_inside_a_tape_column_are_reported_unsupported():
    p = np.array([0.5] * 5 + [0.2, 0.3, 0.9] + [0.1, 0.1, 0.4])       # constant / all distinct / 3 events
    strata = np.array([0] * 5 + [1] * 3 + [2] * 3)
    corrupted, report = fcl.corrupt_targets(p, strata, {0: 0, 1: 0, 2: 0}, np.random.default_rng(0))
    assert np.array_equal(corrupted, p) and report["changed_values"] == 0
    kinds = {item["stratum"]: item["kind"] for item in report["unsupported_strata"]}
    assert kinds == {0: "unsupported_constant", 1: "unsupported_small", 2: "unsupported_small"}
    assert report["supported_strata"] == 0 and report["events_in_unsupported_strata"] == 11


def test_the_ordinary_source_tape_carries_real_nonzero_per_example_corruption_not_a_relabelling():
    config = small_config(steps=32, batch=32)
    bank, events = fcl.build_bank(config)
    tape = fcl.build_tape(config, 0, bank, events)
    report, arrays = tape.report["corruption"], tape.arrays
    assert report["events"] == 1024 and report["changed_values"] > 0
    assert report["supported_strata"] > 0 and report["classes_split"] >= report["supported_strata"]
    assert 0 < report["changed_fraction"] < 1
    assert not np.allclose(arrays["p_cf_corrupt"], 1 - arrays["p_cf"])
    assert sorted(arrays["p_cf_corrupt"].tolist()) == sorted(arrays["p_cf"].tolist())
    functions = fcl.function_arrays()
    strata = (functions["orbit_rank"][arrays["function"].astype(np.int64)].astype(np.int64) * 4
              + arrays["site"].astype(np.int64))
    for record in report["stratum_records"]:
        mask = strata == record["stratum"]
        old, new = arrays["p_cf"][mask], arrays["p_cf_corrupt"][mask]
        assert sorted(new.tolist()) == sorted(old.tolist())
        assert int((np.round(new, 12) != np.round(old, 12)).sum()) == record["changed"]
        if record["status"] == "supported":
            assert not is_value_relabelling(old, new) and record["changed"] >= 2
        else:
            assert np.array_equal(old, new) and record["changed"] == 0


def test_a_and_i_and_ic_consume_identical_events_labels_and_sites(context):
    digests = {}
    for arm in fcl.ARMS:
        run = fcl.ArmRun(context, arm, 0)
        run.step()
        run.step()
        digests[arm] = [e["update_sha256"] for e in run.log]
    assert all(digests[arm] == digests["F"] for arm in fcl.ARMS)
    first, _ = fcl.ArmRun(context, "A", 0).consumed(0)
    second, _ = fcl.ArmRun(context, "I", 0).consumed(0)
    assert all(np.array_equal(first[k], second[k]) for k in first)


def test_counterfactual_arms_read_the_right_target(context):
    update, _ = fcl.ArmRun(context, "I", 0).consumed(0)
    changed = {k: v.copy() for k, v in update.items()}
    changed["p_cf"] = np.clip(changed["p_cf"] * 0.5 + 0.2, 0, 1)
    changed_corrupt = {k: v.copy() for k, v in update.items()}
    changed_corrupt["p_cf_corrupt"] = np.clip(changed_corrupt["p_cf_corrupt"] * 0.5 + 0.2, 0, 1)
    values = {}
    for arm in ("A", "I", "Ic"):
        run = fcl.ArmRun(context, arm, 0)
        values[arm] = tuple(run.loss_terms(u)[1]["counterfactual"]
                            for u in (update, changed, changed_corrupt))
    for arm in ("A", "I"):
        assert values[arm][0] != values[arm][1] and values[arm][0] == values[arm][2]
    assert values["Ic"][0] != values["Ic"][2] and values["Ic"][0] == values["Ic"][1]


# ---------------------------------------------------------------------------
# arms, accounting, optimizer state
# ---------------------------------------------------------------------------

def test_every_arm_of_a_seed_starts_from_the_same_initialisation_and_seeds_differ(context):
    digests = {arm: fcl.ArmRun(context, arm, 0).state_sha256() for arm in fcl.ARMS}
    assert len(set(digests.values())) == 1
    assert fcl.ArmRun(context, "F", 1).state_sha256() != digests["F"]
    a, b = fcl.ArmRun(context, "A", 0), fcl.ArmRun(context, "I", 0)
    a.net.head_out.weight.data += 1.0
    assert not torch.equal(a.net.head_out.weight, b.net.head_out.weight)


def test_adam_state_belongs_only_to_parameters_a_loss_actually_reached(context):
    for arm in fcl.ARMS:
        run = fcl.ArmRun(context, arm, 0)
        for _ in range(3):
            run.step()
        names = [n for n, _ in run.net.named_parameters()]
        entries = run.optimizer.state_dict()["state"]
        expected = fcl.expected_update_counts(arm, 3, run.net)
        assert {names[i] for i in entries} == {n for n, c in expected.items() if c > 0}
        assert all(int(entry["step"]) == 3 for entry in entries.values())
        marginal = [i for i, n in enumerate(names) if n.startswith("marginal_heads")]
        assert (all(i in entries for i in marginal) if arm == "M"
                else all(i not in entries for i in marginal))


def test_incremental_accounting_equals_the_independent_derivation_and_a_i_match(context):
    accounting = {}
    for arm in fcl.ARMS:
        run = fcl.ArmRun(context, arm, 0)
        for _ in range(3):
            run.step()
            assert run.accounting == fcl.expected_accounting(arm, run.cursor, run.config, run.tape)
        accounting[arm] = run.accounting
    assert accounting["A"]["forward_calls"] == accounting["I"]["forward_calls"] == accounting["Ic"][
        "forward_calls"]
    assert accounting["A"]["rows"]["event_rows_padded"] == accounting["I"]["rows"]["event_rows_padded"]
    assert accounting["F"]["rows"]["event_rows_padded"] * 2 == accounting["I"]["rows"][
        "event_rows_padded"]
    assert accounting["M"]["presentations"]["marginal_targets"] == 3 * 8 * 4
    assert all(accounting[a]["presentations"]["marginal_targets"] == 0 for a in ("F", "A", "I", "Ic"))
    assert accounting["F"]["presentations"]["factual"] == accounting["I"]["presentations"]["factual"]
    for arm in ("A", "I"):
        rows = accounting[arm]["rows"]
        assert rows["event_rows_useful"] <= rows["event_rows_padded"]


def test_cost_report_keeps_units_separate(context):
    reports = {}
    for arm in fcl.ARMS:
        run = fcl.ArmRun(context, arm, 0)
        for _ in range(4):
            run.step()
        reports[arm] = run.cost_report()
    canonical = {arm: report["canonical_answers"] for arm, report in reports.items()}
    raw = {arm: report["raw_inputs"] for arm, report in reports.items()}
    encoded = {arm: report["distinct_raw_contexts_encoded"] for arm, report in reports.items()}
    assert canonical["F"]["counterfactual"] == 0 and canonical["I"]["counterfactual"] > 0
    assert canonical["M"]["marginal"] == 4 * len({
        int(s) for s in context.tapes[0].arrays["base_sig"][:32]})
    assert raw["M"]["marginal"] == 4 * encoded["M"]["base"]
    assert reports["Ic"]["corrupted_counterfactual_target_rows"] >= 0 and reports["I"][
        "corrupted_counterfactual_target_rows"] == 0
    assert canonical["I"]["factual"] == canonical["F"]["factual"]
    # a raw record is finer than the answer it belongs to: it can only be as many or more
    for arm in fcl.ARMS:
        assert raw[arm]["factual"] >= canonical[arm]["factual"]
        assert raw[arm]["counterfactual"] >= canonical[arm]["counterfactual"]
        reuse = reports[arm]["presentations_and_reuse"]
        for kind, entry in reuse.items():
            assert entry["reused_beyond_canonical"] == entry["presented"] - entry["canonical_answers"]
            assert entry["reused_beyond_canonical"] >= entry["reused_beyond_raw"] >= 0
    assert reports["I"]["raw_observation_rows"]["useful"] <= reports["I"]["raw_observation_rows"][
        "padded_dense"]
    # what each arm actually ENCODED: base for F and M, base + hybrid for A, base + donor for I and Ic
    assert encoded["F"] == encoded["M"] and encoded["F"]["hybrid"] == encoded["F"]["donor"] == 0
    assert encoded["A"]["hybrid"] > 0 and encoded["A"]["donor"] == 0
    assert encoded["I"]["donor"] > 0 and encoded["Ic"]["donor"] == encoded["I"]["donor"]
    assert encoded["A"]["union"] >= max(encoded["A"]["base"], encoded["A"]["hybrid"])
    assert encoded["I"]["union"] >= max(encoded["I"]["base"], encoded["I"]["donor"])
    assert encoded["A"]["union"] > encoded["F"]["union"], "A's raw hybrids are encoded inputs too"
    assert reports["I"]["parameters"]["reached_by_a_loss"] == reports["I"]["parameters"]["total"] - sum(
        p.numel() for n, p in fcl.make_net(context.spec, 0).named_parameters()
        if n.startswith("marginal_heads"))
    assert reports["M"]["parameters"]["reached_by_a_loss"] == reports["M"]["parameters"]["total"]
    assert "never added" in reports["I"]["note"] and reports["I"]["unit_definitions"]
    assert not any(key.startswith("unique_") and "label" in key for key in reports["I"])


def crafted_cost_report(context, arm):
    """The cost report of ONE update of eight hand-made tape rows over hand-made raw records: three
    variants of one context (one reordered and renamed, one with an extra cancelling pair), two donors that
    differ only at facts the hybrid does not take, exact repeats and a second program."""
    world, site, base_index, variants = fe.q7_world(), 2, 0, 3
    base = world.source[base_index]
    by_level = {}
    for donor in fcl.donor_candidates()[base_index, site].tolist():
        by_level.setdefault(world.source[donor][site], []).append(donor)
    level = next(lv for lv, members in sorted(by_level.items()) if lv != base[site] and len(members) >= 2)
    d1, d2 = by_level[level][:2]
    assert world.source[d1][site] == world.source[d2][site] and world.source[d1] != world.source[d2]
    arrays = {name: np.array(context.bank.arrays[name]) for name in fcl.EVENT_FIELDS}

    def put(signature_index, variant, events):
        packed = fcl.events_to_arrays([events], context.bank.width)
        for name in fcl.EVENT_FIELDS:
            arrays[name][signature_index * variants + variant] = packed[name][0]
    minimal = fe.raw_events(base, 0, cancel_max=1, null_max=2)
    reordered = tuple(fe.Observation(f"again.{i}", e.fact, e.bit, e.reliability)
                      for i, e in enumerate(reversed(minimal)))
    pair_fact = (site + 1) % 4
    padded = minimal + (fe.Observation("pad.0", pair_fact, 1, 0.75),
                        fe.Observation("pad.1", pair_fact, 0, 0.75))
    put(base_index, 0, minimal)
    put(base_index, 1, reordered)                  # the same raw input up to event ids and order
    put(base_index, 2, padded)                     # the same answer, a different raw input
    hybrid = world.source_index[fe.hybrid_signature(base, world.source[d1], site)]
    assert hybrid == world.source_index[fe.hybrid_signature(base, world.source[d2], site)]
    p_cf = np.linspace(0.1, 0.9, 8)
    corrupt = p_cf.copy()
    corrupt[[1, 3, 5]] = p_cf[[3, 5, 1]]
    tape = {"base_sig": np.full(8, base_index, np.int16), "base_var": np.array([0, 0, 1, 2, 0, 0, 0, 0], np.int8),
            "donor_sig": np.array([d1, d2, d1, d1, d1, d1, d1, d1], np.int16),
            "donor_var": np.zeros(8, np.int8), "site": np.full(8, site, np.int8),
            "function": np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int16),
            "hybrid_sig": np.full(8, hybrid, np.int16), "p_cf": p_cf, "p_cf_corrupt": corrupt}
    run = fcl.ArmRun(context, arm, 0)
    assert run.config.batch == 8
    run.bank = fcl.EventBank(arrays=fcl._owned(arrays), variants=variants, width=context.bank.width,
                             sha256="")
    run.tape = types.SimpleNamespace(arrays=tape, report=run.tape.report)
    run.cursor = 1
    run.accounting = fcl.expected_accounting(
        arm, 1, run.config, types.SimpleNamespace(arrays={"useful": np.zeros((1, 3), dtype=np.int32)}))
    return run.cost_report()


@pytest.mark.parametrize("arm", fcl.ARMS)
def test_costs_separate_canonical_answers_raw_inputs_interventions_and_encoded_contexts(context, arm):
    """Equivalent raw variants and donors that differ only at irrelevant facts are NOT new answers; the
    old accounting keyed everything by the raw bank row and counted four factual and five counterfactual
    labels for the very rows below, where only two canonical answers of each kind exist."""
    report = crafted_cost_report(context, arm)
    counterfactual = arm in fcl.CF_ARMS
    assert report["canonical_answers"] == {
        "factual": 2, "counterfactual": 2 if counterfactual else 0, "marginal": 4 if arm == "M" else 0}
    raw_cf = {"A": 3, "I": 4, "Ic": 4}.get(arm, 0)
    assert report["raw_inputs"] == {"factual": 3, "counterfactual": raw_cf,
                                    "marginal": 8 if arm == "M" else 0}
    assert report["intervention_tuples"] == ({"canonical": 3, "raw": 4} if counterfactual
                                             else {"canonical": 0, "raw": 0})
    encoded = report["distinct_raw_contexts_encoded"]
    assert encoded == {"F": {"base": 2, "hybrid": 0, "donor": 0, "union": 2},
                       "M": {"base": 2, "hybrid": 0, "donor": 0, "union": 2},
                       "A": {"base": 2, "hybrid": 2, "donor": 0, "union": 4},
                       "I": {"base": 2, "hybrid": 0, "donor": 2, "union": 4},
                       "Ic": {"base": 2, "hybrid": 0, "donor": 2, "union": 4}}[arm]
    reuse = report["presentations_and_reuse"]
    assert reuse["factual"] == {"presented": 8, "canonical_answers": 2, "raw_inputs": 3,
                                "reused_beyond_canonical": 6, "reused_beyond_raw": 5}
    if counterfactual:
        assert reuse["counterfactual"] == {"presented": 8, "canonical_answers": 2, "raw_inputs": raw_cf,
                                           "reused_beyond_canonical": 6, "reused_beyond_raw": 8 - raw_cf}
    if arm == "M":
        assert reuse["marginal"] == {"presented": 32, "canonical_answers": 4, "raw_inputs": 8,
                                     "reused_beyond_canonical": 28, "reused_beyond_raw": 24}
    assert report["corrupted_counterfactual_target_rows"] == (3 if arm == "Ic" else 0)
    assert report["unique_programs"] == 2


def test_tape_oracle_counters_are_actual_calls_and_marginals_follow_the_base_cache(monkeypatch):
    """The marginals are computed only for a base row not seen before, so they are NOT four per tape row;
    the hybrid posterior and both probabilities are computed for every row. Every counter must equal the
    calls the teacher really received."""
    config = small_config(steps=40, batch=32, variants=1, eval_variant=0)
    bank, events = fcl.build_bank(config)
    calls = {"posterior": 0, "marginals": 0, "probability": 0}
    posterior, marginals, probability = fe.world_posterior, fe.marginals, fe.table_probability

    def count(name, function):
        def spy(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return spy
    monkeypatch.setattr(fe, "world_posterior", count("posterior", posterior))
    monkeypatch.setattr(fe, "marginals", count("marginals", marginals))
    monkeypatch.setattr(fe, "table_probability", count("probability", probability))
    tape = fcl.build_tape(config, 0, bank, events)
    cost, arrays, rows = tape.report["teacher_cost"], tape.arrays, 40 * 32
    distinct = len({(int(s), int(v)) for s, v in zip(arrays["base_sig"], arrays["base_var"])})
    assert distinct < rows, "the fixture must contain repeated base rows"
    assert cost["base_posterior_computations"] == cost["marginal_computations"] == distinct
    assert cost["marginal_values"] == 4 * distinct != 4 * rows
    assert cost["hybrid_posterior_computations"] == cost["hybrid_event_materializations"] == rows
    assert cost["factual_probability_evaluations"] == cost["counterfactual_probability_evaluations"] == rows
    assert cost["probability_evaluations"] == 2 * rows and cost["tape_rows_materialized"] == rows
    assert cost["marginal_target_values_materialized"] == 4 * rows
    assert calls == {"posterior": distinct + rows, "marginals": distinct, "probability": 2 * rows}
    assert "marginal_evaluations" not in cost


def test_the_model_actually_learns_to_fit_a_fixed_batch(context):
    """A sanity check that the soft-label objective trains the network, not a scientific claim."""
    run = fcl.ArmRun(context, "F", 0)
    update, _ = run.consumed(0)
    floor = float(fcl.bernoulli_entropy(update["p_base"]).mean())     # the irreducible soft-label entropy
    optimizer = torch.optim.Adam(run.net.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss, _, _ = run.loss_terms(update)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[0] > floor and losses[-1] - floor < 0.5 * (losses[0] - floor)


# ---------------------------------------------------------------------------
# resume, checkpoints and tamper refusal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", fcl.ARMS)
def test_interrupted_and_resumed_runs_equal_the_uninterrupted_run_exactly(context, arm, tmp_path):
    straight = fcl.train_arm(context, arm, 0, tmp_path / "straight")
    fcl.train_arm(context, arm, 0, tmp_path / "resumed", stop_after=2)
    resumed = fcl.train_arm(context, arm, 0, tmp_path / "resumed")
    assert straight.cursor == resumed.cursor == 4
    assert straight.deterministic_result() == resumed.deterministic_result()


def reseal(directory, *, payload_edit=None, header_edit=None):
    directory = Path(directory)
    header = fb.read_json(directory / "checkpoint.json")
    weights = directory / header["weights_file"]
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if payload_edit is not None:
        payload_edit(payload)
    torch.save(payload, weights)
    header["weights_file_sha256"] = file_sha256(weights)
    header["state_sha256"] = fst.tensor_state_digest(payload)
    if header_edit is not None:
        header_edit(header)
    body = {k: v for k, v in header.items() if k != "header_sha256"}
    header["header_sha256"] = fb.digest(body)
    fb.write_json(directory / "checkpoint.json", header)


class Opaque:
    pass


@pytest.fixture
def interrupted(context, tmp_path):
    directory = tmp_path / "ckpt"
    fcl.train_arm(context, "I", 0, directory, stop_after=2)
    return directory


def refuse_restore(context, directory, arm="I", seed=0):
    run = fcl.ArmRun(context, arm, seed)
    before = (run.state_sha256(), run.cursor, json.dumps(run.accounting), len(run.log))
    with pytest.raises(ValueError):
        fcl.restore_checkpoint(run, directory)
    assert (run.state_sha256(), run.cursor, json.dumps(run.accounting), len(run.log)) == before


def test_a_valid_checkpoint_is_adopted(context, interrupted):
    run = fcl.ArmRun(context, "I", 0)
    assert fcl.restore_checkpoint(run, interrupted) is True and run.cursor == 2
    assert fcl.restore_checkpoint(fcl.ArmRun(context, "I", 0), interrupted.parent / "none") is False


def test_tampered_weights_headers_and_foreign_checkpoints_are_refused_without_mutation(
        context, interrupted, tmp_path):
    header = fb.read_json(interrupted / "checkpoint.json")
    weights = interrupted / header["weights_file"]
    # (a) a flipped byte in the weights file
    raw = bytearray(weights.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    weights.write_bytes(bytes(raw))
    refuse_restore(context, interrupted)
    # (b) a header edited without re-sealing
    other = tmp_path / "edited"
    fcl.train_arm(context, "I", 0, other, stop_after=2)
    document = fb.read_json(other / "checkpoint.json")
    document["cursor"] = 3
    fb.write_json(other / "checkpoint.json", document)
    refuse_restore(context, other)
    # (c) a re-sealed header with the wrong cursor
    third = tmp_path / "resealed"
    fcl.train_arm(context, "I", 0, third, stop_after=2)
    reseal(third, header_edit=lambda h: h.update(cursor=3))
    refuse_restore(context, third)
    # (d) another arm's checkpoint
    fourth = tmp_path / "foreign"
    fcl.train_arm(context, "A", 0, fourth, stop_after=2)
    refuse_restore(context, fourth, arm="I")
    # (e) another seed's checkpoint
    fifth = tmp_path / "seed1"
    fcl.train_arm(context, "I", 1, fifth, stop_after=2)
    refuse_restore(context, fifth, arm="I", seed=0)


def test_a_pickled_object_in_place_of_the_weights_is_refused(context, interrupted):
    header = fb.read_json(interrupted / "checkpoint.json")
    weights = interrupted / header["weights_file"]
    torch.save({"format": fcl.CHECKPOINT_SCHEMA, "net": Opaque(), "optimizer": {}}, weights)
    header["weights_file_sha256"] = file_sha256(weights)
    body = {k: v for k, v in header.items() if k != "header_sha256"}
    header["header_sha256"] = fb.digest(body)
    fb.write_json(interrupted / "checkpoint.json", header)
    refuse_restore(context, interrupted)


def test_malformed_tensors_and_optimizer_entries_are_refused_even_when_consistently_sealed(
        context, tmp_path):
    def malformed_net(payload):
        key = sorted(payload["net"])[0]
        payload["net"][key] = payload["net"][key][..., :1].clone()

    def nan_net(payload):
        key = sorted(payload["net"])[0]
        payload["net"][key] = payload["net"][key].clone().fill_(float("nan"))

    def bad_step(payload):
        entry = next(iter(payload["optimizer"]["state"].values()))
        entry["step"] = torch.tensor(99.0)

    def bad_moment(payload):
        entry = next(iter(payload["optimizer"]["state"].values()))
        entry["exp_avg"] = entry["exp_avg"][..., :1].clone()

    def negative_second_moment(payload):
        entry = next(iter(payload["optimizer"]["state"].values()))
        entry["exp_avg_sq"] = -entry["exp_avg_sq"].abs() - 1.0

    def stray_entry(payload):
        names = [n for n, _ in fcl.make_net(context.spec, 0).named_parameters()]
        index = next(i for i, n in enumerate(names) if n.startswith("marginal_heads"))
        payload["optimizer"]["state"][index] = dict(next(iter(payload["optimizer"]["state"].values())))
    for number, edit in enumerate((malformed_net, nan_net, bad_step, bad_moment, negative_second_moment,
                                   stray_entry)):
        directory = tmp_path / f"bad{number}"
        fcl.train_arm(context, "I", 0, directory, stop_after=2)
        reseal(directory, payload_edit=edit)
        refuse_restore(context, directory)


def test_save_checkpoint_never_overwrites_an_unknown_file_or_a_protected_input(context, tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    junk = directory / "ckpt_000002.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        fcl.train_arm(context, "I", 0, directory, stop_after=2)
    assert junk.read_bytes() == b"not a checkpoint"
    guarded = tmp_path / "guarded"
    fcl.train_arm(context, "I", 0, guarded, stop_after=2)
    kept = guarded / "ckpt_000002.pt"
    fcl.train_arm(context, "I", 0, guarded, protected=[kept])
    assert kept.exists() and (guarded / "ckpt_000004.pt").exists()
    plain = tmp_path / "plain"
    fcl.train_arm(context, "I", 0, plain, stop_after=2)
    fcl.train_arm(context, "I", 0, plain)
    assert not (plain / "ckpt_000002.pt").exists()
    aliased = tmp_path / "aliased"
    with pytest.raises(ValueError):
        fcl.train_arm(context, "I", 0, aliased, stop_after=2, protected=[aliased / "ckpt_000002.pt"])


@pytest.mark.parametrize("name,edit", [
    ("target", lambda run: run.tape.arrays["p_cf"].__setitem__(0, run.tape.arrays["p_cf"][0] + 0.25)),
    ("base_target", lambda run: run.tape.arrays["p_base"].__setitem__(1, 0.123)),
    ("corrupted_target", lambda run: run.tape.arrays["p_cf_corrupt"].__setitem__(2, 0.77)),
    ("marginal_target", lambda run: run.tape.arrays["marg_base"].__setitem__((3, 0), 0.31)),
    ("program", lambda run: run.tape.arrays["function"].__setitem__(
        0, (int(run.tape.arrays["function"][0]) + 1) % 48)),
    ("site", lambda run: run.tape.arrays["site"].__setitem__(0, (int(run.tape.arrays["site"][0]) + 1) % 4)),
    ("evidence_identity", lambda run: run.tape.arrays["base_var"].__setitem__(
        0, (int(run.tape.arrays["base_var"][0]) + 1) % 3)),
    ("donor", lambda run: run.tape.arrays["donor_sig"].__setitem__(
        0, (int(run.tape.arrays["donor_sig"][0]) + 1) % 1176)),
    ("bank_event", lambda run: run.bank.arrays["bit"].__setitem__(
        (int(run.tape.arrays["base_sig"][0]) * 3 + int(run.tape.arrays["base_var"][0]), 0),
        1 - run.bank.arrays["bit"][int(run.tape.arrays["base_sig"][0]) * 3
                                   + int(run.tape.arrays["base_var"][0]), 0])),
    ("program_tokens", lambda run: run.functions["tokens"].__setitem__(
        (int(run.tape.arrays["function"][0]), 0),
        (int(run.functions["tokens"][int(run.tape.arrays["function"][0]), 0]) + 1) % 8))])
def test_a_tampered_consumed_input_refuses_before_any_optimizer_mutation(context, name, edit):
    run = fcl.ArmRun(context, "I", 0)
    clean = fcl.ArmRun(context, "I", 0)
    for owner in (run.tape.arrays, run.bank.arrays, run.functions):
        for array in owner.values():
            array.setflags(write=True)
    before = (run.state_sha256(), run.cursor, json.dumps(run.accounting), len(run.log), run.cpu_seconds)
    edit(run)
    with pytest.raises(ValueError):
        run.step()
    assert (run.state_sha256(), run.cursor, json.dumps(run.accounting), len(run.log),
            run.cpu_seconds) == before
    assert run.state_sha256() == clean.state_sha256()
    with pytest.raises(ValueError):
        fcl.save_checkpoint(run, run_dir_for(name), protected=())
    clean.step()


def run_dir_for(name):
    return Path("unused") / name


def test_a_tampered_shared_input_is_refused_before_a_run_exists(context):
    array = context.tapes[0].arrays["p_cf"]
    original = float(array[0])
    array.setflags(write=True)
    array[0] = original + 0.3
    try:
        with pytest.raises(ValueError):
            fcl.ArmRun(context, "F", 0)
        with pytest.raises(ValueError):
            context.verify_live(0)
    finally:
        array[0] = original
        array.setflags(write=False)
    assert fcl.ArmRun(context, "F", 0).cursor == 0
    state = context.init_states[0]
    key = sorted(state)[0]
    saved = state[key].clone()
    state[key] += 1.0
    try:
        with pytest.raises(ValueError):
            fcl.ArmRun(context, "F", 0)
    finally:
        state[key].copy_(saved)


def test_invalid_initial_weights_are_refused_before_cloning(context):
    state = {k: v.clone() for k, v in context.init_states[0].items()}
    key = sorted(state)[0]
    for bad in (state[key][..., :1].clone(), state[key].clone().fill_(float("inf")),
                state[key].double()):
        broken = dict(state)
        broken[key] = bad
        with pytest.raises(ValueError):
            fcl.net_from_state(context.spec, broken)
    broken = dict(state)
    del broken[key]
    with pytest.raises(ValueError):
        fcl.net_from_state(context.spec, broken)


def test_a_finished_run_cannot_step_again(context):
    run = fcl.ArmRun(context, "F", 0)
    for _ in range(4):
        run.step()
    assert run.cursor == run.total_steps == 4
    before = run.state_sha256()
    with pytest.raises(ValueError):
        run.step()
    assert run.state_sha256() == before


# ---------------------------------------------------------------------------
# tape and initial-state files
# ---------------------------------------------------------------------------

def test_tape_and_init_files_round_trip_and_are_verified(context, tmp_path):
    tape = context.tapes[0]
    fcl.save_tape(tape, tmp_path / "tapes")
    fcl.save_tape(tape, tmp_path / "tapes")                          # an identical publication is adopted
    loaded = fcl.load_tape(tmp_path / "tapes", 0, context.config, bank_sha256=context.bank.sha256)
    assert loaded.sha256 == tape.sha256 and loaded.update_sha256 == tape.update_sha256
    fcl.save_init(context.init_states[0], 0, tmp_path / "init", context.spec)
    state = fcl.load_init(tmp_path / "init", 0, context.spec)
    assert fst.tensor_state_digest(state) == context.init_sha256[0]
    with pytest.raises(ValueError):
        fcl.load_tape(tmp_path / "tapes", 0, small_config(lr=1e-3), bank_sha256=context.bank.sha256)
    with pytest.raises(ValueError):
        fcl.load_tape(tmp_path / "tapes", 0, context.config, bank_sha256="0" * 64)
    with pytest.raises(ValueError):                                  # another state is never written over it
        fcl.save_init(context.init_states[1], 0, tmp_path / "init", context.spec)
    npz = tmp_path / "tapes" / "seed_0.npz"
    with np.load(npz) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["p_cf"][0] += 0.2
    np.savez(npz, **arrays)
    with pytest.raises(ValueError):
        fcl.load_tape(tmp_path / "tapes", 0, context.config, bank_sha256=context.bank.sha256)


# ---------------------------------------------------------------------------
# frozen states
# ---------------------------------------------------------------------------

def test_only_complete_runs_freeze_and_a_frozen_state_is_verified_on_load(context, tmp_path):
    run = fcl.ArmRun(context, "I", 0)
    run.step()
    with pytest.raises(ValueError):
        fcl.freeze_run(run, tmp_path / "frozen")
    done = fcl.train_arm(context, "I", 0, tmp_path / "run")
    header = fcl.freeze_run(done, tmp_path / "frozen")
    assert fcl.freeze_run(done, tmp_path / "frozen")["header_sha256"] == header["header_sha256"]
    model = fcl.load_frozen(tmp_path / "frozen", "I_s0", context.spec, bindings=context.bindings(0))
    assert model.state_sha256 == fst.tensor_state_digest(dict(done.net.state_dict()))
    assert model.verify_unchanged() and not any(p.requires_grad for p in model.net.parameters())
    with pytest.raises(ValueError):
        fcl.load_frozen(tmp_path / "frozen", "I_s0", context.spec, bindings=context.bindings(1))
    weights = tmp_path / "frozen" / "I_s0.pt"
    raw = bytearray(weights.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    weights.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        fcl.load_frozen(tmp_path / "frozen", "I_s0", context.spec)
    scratch = fcl.freeze_scratch(context, 0, tmp_path / "scratch")
    assert scratch["state_sha256"] == context.init_sha256[0] and scratch["updates"] == 0


# ---------------------------------------------------------------------------
# held-out evaluation of one (untrained) frozen model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluation(context):
    oracle = fcl.PanelOracle(context.config)
    model = untrained_model(context)
    return oracle, fcl.evaluate_model(model, oracle, context.config, context.bank)


def test_panels_have_the_declared_case_counts(evaluation):
    _, result = evaluation
    panels = result["panels"]
    assert panels["source_fit"]["cases"] == 56448
    assert panels["primary_composition"]["cases"] == 378
    assert panels["primary_source_preservation"]["cases"] == 672
    assert panels["validation_new_function"]["cases"] == 42
    assert panels["validation_source_function"]["cases"] == 336
    assert result["interventions"]["cases"] == 21168 == 14 * 14 * 4 * 27
    assert result["ablation_and_rescue"]["cases"] == 14 * 4 * 27


def test_the_oracle_is_the_independent_teacher(evaluation):
    oracle, _ = evaluation
    registry = fe.function_registry()
    cs = oracle.context_set("primary")
    functions = [f.index for f in registry.role("primary")]
    matrix = oracle.matrix("primary", functions)
    for c in (0, 5, 13):
        for j in (0, 13, 26):
            assert float(brute_probability(registry.functions[functions[j]].program, cs.events[c])
                         ) == matrix[c, j]
    hybrid = oracle.hybrids("primary")
    probabilities = oracle.probabilities(hybrid["posteriors"], functions)
    for row in (0, 100, 783):
        b, d, k = hybrid["index"][row]
        events = fe.hybrid_events(cs.events[b], cs.events[d], int(k))
        assert float(brute_probability(registry.functions[functions[7]].program, events)
                     ) == probabilities[row, 7]
        assert fe.signature_of_events(events) == fe.hybrid_signature(cs.signatures[b], cs.signatures[d],
                                                                      int(k))


def test_changed_weak_and_unchanged_are_defined_by_the_oracle_change_not_by_relevance(evaluation):
    _, result = evaluation
    summary = result["interventions"]["summary"]
    assert (summary["changed"]["n"] + summary["weak"]["n"] + summary["unchanged"]["n"]
            == summary["all"]["n"] == 21168)
    assert summary["irrelevant"]["n"] == summary["unchanged_irrelevant"]["n"] > 0
    assert summary["unchanged_relevant"]["n"] > 0 and summary["changed_relevant"]["n"] > 0
    assert summary["changed"]["n"] == summary["changed_relevant"]["n"]
    assert summary["relevant"]["n"] + summary["irrelevant"]["n"] == 21168
    assert summary["same_value_donor"]["n"] + summary["different_value_donor"]["n"] == 21168
    assert summary["identity_donor"]["n"] == 14 * 4 * 27
    assert min(summary["changed"]["metrics"]["oracle_response"]["mean"], 1.0) >= result[
        "interventions"]["changed_threshold"]


def test_both_comparisons_are_reported_and_agreement_is_not_fidelity(evaluation):
    _, result = evaluation
    changed = result["interventions"]["summary"]["changed"]["metrics"]
    for key in ("patched_kl", "patched_brier_excess", "patched_abs_error", "hybrid_kl", "hybrid_abs_error",
                "agreement_abs", "agreement_kl", "mismatched_abs_error", "patched_response",
                "hybrid_response", "oracle_response", "base_ordinary_kl"):
        assert key in changed
        for group in ("by_function_family", "by_base_context", "by_site"):
            assert group in changed[key]
        assert "mean_of_base_context_means" in changed[key]
    assert len(changed["patched_kl"]["by_base_context"]) == 14
    assert len(changed["patched_kl"]["by_function_family"]) == 3
    # an untrained network predicts nearly constant values: its patched and hybrid predictions agree
    # closely while both are far from the oracle, so agreement alone is NOT causal fidelity
    assert changed["agreement_abs"]["mean"] < changed["patched_abs_error"]["mean"]


def test_identity_insertion_and_invariance_are_measured(evaluation):
    _, result = evaluation
    assert result["interventions"]["identity_insertion_max_abs_logit_difference"] == 0.0
    assert result["headline"]["identity_insertion_max_abs_logit_difference"] == 0.0
    invariance = result["invariance"]
    assert invariance["all_within_tolerance"] and invariance["tolerance"] > 0
    assert set(invariance) >= {"event_permutation_max_abs_block_difference",
                               "right_padding_max_abs_block_difference",
                               "batch_composition_max_abs_block_difference",
                               "program_right_padding_max_abs_difference"}


def test_calibration_ambiguity_and_source_preservation_are_reported(evaluation):
    _, result = evaluation
    panel = result["panels"]["primary_composition"]
    assert 0.0 <= panel["calibration"]["expected_calibration_error"] <= 1.0
    assert panel["calibration"]["mean_oracle_entropy"] > 0
    assert set(result["headline"]) >= {"primary_composition_kl", "primary_source_preservation_kl",
                                       "intervention_changed_patched_kl",
                                       "intervention_irrelevant_patched_response"}
    assert panel["summary"]["all"]["metrics"]["kl"]["mean"] >= 0
    assert panel["summary"]["all"]["metrics"]["brier_excess"]["mean"] >= 0


def test_ablation_and_rescue_use_a_frozen_source_train_mean_not_the_evaluation_batch(context, evaluation):
    oracle, result = evaluation
    model = untrained_model(context)
    mean = fcl.source_train_mean_blocks(model.net, context.bank)
    blocks = fcl.encode_arrays(model.net, {k: context.bank.arrays[k] for k in fcl.EVENT_FIELDS})
    assert torch.allclose(mean, blocks.mean(dim=0)) and mean.shape == (4, 8)
    primary = fcl.encode_arrays(model.net, oracle.context_set("primary").arrays)
    assert not torch.allclose(mean, primary.mean(dim=0))
    again = fcl.ablation_panel(model.net, oracle, context.config, mean)
    assert again["summary"]["all"]["metrics"]["ablate_kl"]["mean"] == result["ablation_and_rescue"][
        "summary"]["all"]["metrics"]["ablate_kl"]["mean"]
    metrics = result["ablation_and_rescue"]["summary"]["relevant"]["metrics"]
    assert {"ablate_abs_error", "rescue_abs_error", "rescue_gain_abs", "ablation_effect",
            "full_ablation_empty_abs_error", "full_ablation_rescue_target_abs_error"} <= set(metrics)
    assert "full_ablation_abs_error" not in metrics, "the ambiguous name is gone"


def test_rescue_gain_scores_both_predictions_against_the_same_target():
    """The handcase: AND(X0, X1) has probability 1/4 with no evidence and 3/8 with only X0 observed. A full
    ablation that predicts 0.25 and a rescue that predicts 0.30 move the error to 3/8 from .125 to .075, a
    gain of +0.05; subtracting the error to the EMPTY answer from the error to the ONLY-k answer reported
    -0.075."""
    program = fe.And(fe.Read(0), fe.Read(1))
    empty = fe.program_probability(program, ())
    only_k = fe.program_probability(program, (fe.Observation("e", 0, 1, 0.75),))
    assert (empty, only_k) == (Fraction(1, 4), Fraction(3, 8))
    assert fcl.rescue_gain(0.25, 0.30, float(only_k)) == pytest.approx(0.05)
    assert abs(0.25 - float(empty)) - abs(0.30 - float(only_k)) == pytest.approx(-0.075)
    gains = fcl.rescue_gain(np.array([0.25, 0.25]), np.array([0.30, 0.10]), np.array([0.375, 0.375]))
    assert gains == pytest.approx([0.05, -0.15])               # a rescue that moves away is a loss


def test_the_ablation_panel_reports_the_common_target_gain_and_keeps_empty_fidelity_separate(
        context, evaluation):
    oracle, result = evaluation
    model = untrained_model(context)
    mean = fcl.source_train_mean_blocks(model.net, context.bank)
    cases = fcl.ablation_cases(model.net, oracle, context.config, mean)
    predicted, answers, metrics = cases["predictions"], cases["oracles"], cases["metrics"]
    expected = np.abs(predicted["full"] - answers["only"]) - np.abs(predicted["rescue"] - answers["only"])
    assert np.array_equal(metrics["rescue_gain_abs"], expected)
    assert np.array_equal(metrics["full_ablation_rescue_target_abs_error"],
                          np.abs(predicted["full"] - answers["only"]))
    assert np.array_equal(metrics["full_ablation_empty_abs_error"],
                          np.abs(predicted["full"] - answers["empty"]))
    assert np.array_equal(metrics["rescue_abs_error"], np.abs(predicted["rescue"] - answers["only"]))
    assert not np.allclose(answers["empty"], answers["only"]), "the two oracle targets really differ"
    assert not np.allclose(metrics["rescue_gain_abs"],
                           metrics["full_ablation_empty_abs_error"] - metrics["rescue_abs_error"])
    panel = fcl.ablation_panel(model.net, oracle, context.config, mean)
    overall = panel["summary"]["all"]["metrics"]["rescue_gain_abs"]
    assert overall["mean"] == pytest.approx(float(expected.mean()))
    assert result["headline"]["rescue_gain_abs"] == pytest.approx(float(expected.mean()))
    site = cases["groups"]["site"]
    for k in range(4):
        assert overall["by_site"][f"site{k}"] == pytest.approx(float(expected[site == k].mean()))


def test_held_out_contexts_open_lazily_and_are_recorded():
    oracle = fcl.PanelOracle(small_config())
    assert oracle.opened == []
    oracle.context_set("primary")
    oracle.context_set("primary")
    oracle.context_set("transfer_eval")
    assert oracle.opened == ["primary", "transfer_eval"]
    with pytest.raises(ValueError):
        oracle.context_set("nonsense")


def test_summaries_aggregate_exactly_by_group_and_never_report_an_interval():
    metrics = {"x": np.array([1.0, 2.0, 3.0, 5.0])}
    summary = fcl.summarize(metrics, {"all": np.ones(4, dtype=bool), "none": np.zeros(4, dtype=bool)},
                            {"g": np.array([0, 0, 1, 1])}, {"g": ["a", "b"]})
    assert summary["all"]["metrics"]["x"] == {"mean": 2.75, "by_g": {"a": 1.5, "b": 4.0},
                                              "mean_of_g_means": 2.75}
    assert summary["none"] == {"n": 0, "metrics": {}}


def test_bernoulli_metrics_match_hand_values():
    assert abs(float(fcl.bernoulli_kl(0.5, 0.5))) < 1e-12
    assert abs(float(fcl.bernoulli_kl(0.75, 0.5)) - (0.75 * np.log(1.5) + 0.25 * np.log(0.5))) < 1e-12
    assert float(fcl.bernoulli_kl(1.0, 0.5)) == pytest.approx(np.log(2.0))
    assert float(fcl.bernoulli_kl(0.0, 0.25)) == pytest.approx(-np.log(0.75))
    assert float(fcl.bernoulli_entropy(0.5)) == pytest.approx(np.log(2.0))
    assert float(fcl.bernoulli_entropy(1.0)) == 0.0
    assert fcl.sigmoid64(np.array([-800.0, 0.0, 800.0])).tolist() == [0.0, 0.5, 1.0]


def test_paired_seed_differences_are_arm_minus_arm_per_seed():
    headlines = {fcl.run_id(arm, s): {"m": base + (s * 0.1), "n": None}
                 for arm, base in (("F", 1.0), ("I", 0.7), ("A", 0.9), ("M", 0.8), ("Ic", 1.1))
                 for s in (0, 1)}
    table = fcl.paired_differences(headlines, ("F", "M", "A", "I", "Ic"), (0, 1))
    assert table["I-F"]["m"]["per_seed"] == pytest.approx([-0.3, -0.3])
    assert table["I-F"]["n"] is None and table["I-A"]["m"]["mean"] == pytest.approx(-0.2)
    assert set(table) == {"I-F", "I-A", "I-Ic", "I-M", "A-F", "M-F", "Ic-F", "A-Ic"}
    assert set(fcl.paired_differences(headlines, ("F", "I"), (0, 1))) == {"I-F"}


# ---------------------------------------------------------------------------
# the transfer branch
# ---------------------------------------------------------------------------

def test_transfer_selection_is_nested_round_robin_train_only_and_deterministic():
    config = small_config()
    sequence = fcl.transfer_selection(config)
    assert sequence == fcl.transfer_selection(config) and len(sequence) == len(set(sequence)) == 168
    assert {c for c, _ in sequence} == set(range(14))
    registry = fe.function_registry()
    assert {f for _, f in sequence} == {f.index for f in registry.role("transfer")}
    n16, n64 = fcl.selection_report(sequence, 16), fcl.selection_report(sequence, 64)
    assert n16["contexts_covered"] == 14 and n16["derived_labels"] == 16
    assert n64["contexts_covered"] == 14 and (n64["labels_per_context_min"],
                                              n64["labels_per_context_max"]) == (4, 5)
    assert fcl.selection_report(sequence, 0)["contexts_covered"] == 0
    assert sequence[:16] == sequence[:64][:16]
    other = fcl.transfer_selection(small_config(tape_seed=3))
    assert other != sequence and set(other) == set(sequence)


def test_an_untrained_adapter_is_the_frozen_source_and_adaptation_never_touches_the_source(context):
    model = untrained_model(context)
    oracle = fcl.PanelOracle(context.config)
    before = fcl.transfer_eval(model, None, oracle, context.config)
    zero = fcl.make_adapter(context.config, 0, context.config.head_hidden)
    assert fcl.transfer_eval(model, zero, oracle, context.config) == before
    sequence = fcl.transfer_selection(context.config)
    adapter, record = fcl.adapt_adapter(model, oracle, context.config, sequence, 16)
    assert record["steps"] == 5 and record["last_loss"] <= record["first_loss"]
    assert record["selection"]["contexts_covered"] == 14
    model.verify_unchanged()
    adapted = fcl.transfer_eval(model, adapter, oracle, context.config)
    assert adapted != before
    assert fcl.transfer_eval(model, None, oracle, context.config) == before     # removal restores N0
    assert "transfer_fit" in oracle.opened and set(adapted) == {"new_composition", "source_function"}
    assert adapted["new_composition"]["cases"] == 168 and adapted["source_function"]["cases"] == 672
    with pytest.raises(ValueError):
        fcl.adapt_adapter(model, oracle, context.config, sequence, 0)


def test_adapters_are_frozen_hashed_and_refuse_tampering(context, tmp_path):
    model = untrained_model(context)
    oracle = fcl.PanelOracle(context.config)
    sequence = fcl.transfer_selection(context.config)
    adapter, record = fcl.adapt_adapter(model, oracle, context.config, sequence, 16)
    bindings = {"source_state_sha256": model.state_sha256, "selection_sha256": "ab" * 32}
    header = fcl.publish_adapter(tmp_path, "scratch_s0", 16, adapter, bindings=bindings,
                                 record={**record, "seed": 0})
    assert fcl.publish_adapter(tmp_path, "scratch_s0", 16, adapter, bindings=bindings,
                               record={**record, "seed": 0})["header_sha256"] == header["header_sha256"]
    loaded = fcl.load_adapter(tmp_path, "scratch_s0", 16, context.config, context.config.head_hidden,
                              bindings=bindings)
    probe = torch.randn(3, context.config.head_hidden)
    assert torch.equal(loaded(probe), adapter(probe)) and not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError):
        fcl.load_adapter(tmp_path, "scratch_s0", 16, context.config, context.config.head_hidden,
                         bindings={**bindings, "selection_sha256": "cd" * 32})
    weights = tmp_path / "scratch_s0_n16.pt"
    raw = bytearray(weights.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    weights.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        fcl.load_adapter(tmp_path, "scratch_s0", 16, context.config, context.config.head_hidden)


def test_transfer_derived_arithmetic_and_pairing():
    def entry(new_n0, src_n0, new_n16, src_n16):
        return {"N0": {"new_composition": {"kl": new_n0}, "source_function": {"kl": src_n0}},
                "N16": {"new_composition": {"kl": new_n16}, "source_function": {"kl": src_n16}}}
    results = {"F_s0": entry(1.0, 0.2, 0.6, 0.3), "scratch_s0": entry(1.1, 0.5, 0.9, 0.5),
               "I_s0": entry(1.0, 0.25, 0.5, 0.2)}
    derived = fcl.CausalRun._transfer_derived(results, [16])
    assert derived["per_run"]["F_s0"]["N16"]["new_composition_kl_improvement"] == pytest.approx(0.4)
    assert derived["per_run"]["F_s0"]["N16"]["source_function_kl_degradation"] == pytest.approx(0.1)
    pair = derived["paired_vs_scratch_and_F"]["I-scratch"]["N16"][0]
    assert pair["new_composition_kl_difference"] == pytest.approx(0.5 - 0.9)
    assert derived["paired_vs_scratch_and_F"]["I-F"]["N16"][0]["new_composition_kl_difference"] == (
        pytest.approx(-0.1))
    assert "F-F" not in derived["paired_vs_scratch_and_F"]


# ---------------------------------------------------------------------------
# the Build A bridge
# ---------------------------------------------------------------------------

def test_the_bridge_roster_binds_context_and_program_and_groups_by_context(context):
    oracle = fcl.PanelOracle(context.config)
    rows = fcl.bridge_rows(oracle)
    assert len(rows) == 336 and len({r["id"] for r in rows}) == 336
    assert len({r["sequence"] for r in rows}) == 336
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(r["split"])
        signature, program = fe.parse_identity_sequence(r["sequence"])
        assert signature == r["signature"]
        assert program == fe.function_registry().functions[r["function"]].program
        assert len(r["sequence"]) == fe.identity_length()
    assert len(groups) == 28 and all(len(s) == 1 for s in groups.values())
    per_group = {}
    for r in rows:
        per_group.setdefault(r["group"], []).append(r["function"])
    assert all(len(v) == 12 == len(set(v)) for v in per_group.values())
    assert sum(1 for r in rows if r["split"] == "train") == 168
    assert all(r["label"] is not None for r in rows if r["split"] == "train")
    assert all(r["label"] is None for r in rows if r["split"] == "eval")
    assert oracle.opened == ["transfer_fit"]


def test_a_context_only_identity_is_refused_by_the_build_a_candidate_table(tmp_path):
    path = tmp_path / "candidates.csv"
    path.write_text("candidate_id,sequence,group_id,split,prior_log_weight\n"
                    "a,ABCDQQQQQQQ,g,train,0.0\nb,ABCDQQQQQQQ,g,train,0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="same physical sequence"):
        fb.read_candidates(path, alphabet=fe.IDENTITY_ALPHABET, length=fe.identity_length())


def test_bridge_features_are_the_penultimate_activation_and_not_a_posterior(context):
    model = untrained_model(context)
    oracle = fcl.PanelOracle(context.config)
    rows = fcl.bridge_rows(oracle)
    fused = fcl.bridge_features(model, rows, oracle)
    concat = fcl.bridge_features(model, rows, oracle, kind=fcl.FEATURE_CONCAT)
    assert fused.shape == (336, context.config.head_hidden)
    assert concat.shape == (336, 4 * context.config.block_dim + context.config.program_hidden)
    row = rows[17]
    cs = oracle.context_set(row["split_name"])
    blocks = fcl.encode_arrays(model.net, cs.arrays)[row["context"]].unsqueeze(0)
    state = fcl.program_states(model.net, [row["function"]])
    with torch.no_grad():
        logit, penultimate = model.net.predict(blocks, state)
    assert np.allclose(fused[17], penultimate[0].numpy().astype(np.float64), atol=1e-5)
    assert fused.shape[1] != concat.shape[1]
    with pytest.raises(ValueError):
        fcl.bridge_features(model, rows, oracle, kind="posterior")
    # the posterior is the label, never a column: no feature is a function of the label alone
    labels = np.array([r["label"] if r["label"] is not None else np.nan for r in rows])
    for column in range(fused.shape[1]):
        valid = ~np.isnan(labels)
        if fused[valid, column].std() > 0:
            assert abs(np.corrcoef(fused[valid, column], labels[valid])[0, 1]) < 0.999


def bridge_parts(context, *, arm="scratch"):
    """A model header, the roster, its features and its raw-event binding for the bridge helpers."""
    model = untrained_model(context)
    header = {"arm": arm, "seed": 0, "state_sha256": model.state_sha256, "bindings": context.bindings(0),
              "weights_file_sha256": "ab" * 32,
              "cost": {"canonical_answers": {"factual": 5, "counterfactual": 7, "marginal": 0}}}
    oracle = fcl.PanelOracle(context.config)
    rows = fcl.bridge_rows(oracle)
    features = fcl.bridge_features(model, rows, oracle)
    events = fcl.bridge_event_rows(rows, oracle)
    return header, rows, features, fcl.bridge_event_binding(rows, events, context.config)


def test_feature_extraction_needs_raw_events_only_and_materializes_no_held_out_answer(
        context, monkeypatch):
    """``bridge_features`` used to call ``context_set`` for the transfer-evaluation split just to read its
    events, which also computed every exact posterior of that split and recorded it as opened."""
    model = untrained_model(context)
    rows = fcl.bridge_rows(fcl.PanelOracle(context.config))
    fresh = fcl.PanelOracle(context.config)
    touched = []
    original = fe.world_posterior
    monkeypatch.setattr(fe, "world_posterior", lambda *a, **k: touched.append(1) or original(*a, **k))
    features = fcl.bridge_features(model, rows, fresh)
    events = fcl.bridge_event_rows(rows, fresh)
    assert touched == [] and fresh.opened == []
    assert features.shape == (336, context.config.head_hidden)
    assert set(events) == set(fcl.EVENT_FIELDS) and events["mask"].shape == (336, 28)
    raw = fresh.raw_context_set("transfer_eval")
    assert raw.posteriors is None and fresh.opened == [] and touched == []
    assert fresh.context_set("transfer_eval").posteriors is not None
    assert fresh.opened == ["transfer_eval"] and touched, "asking for the answers is what opens a split"
    again = fcl.PanelOracle(context.config)
    open_labels = fcl.bridge_rows(again, open_eval_labels=True)
    assert again.opened == ["transfer_fit", "transfer_eval"]
    assert all(r["label"] is not None for r in open_labels)


def test_cache_provenance_binds_the_variant_and_the_actual_event_arrays(context):
    oracle = fcl.PanelOracle(context.config)
    rows = fcl.bridge_rows(oracle)
    events = fcl.bridge_event_rows(rows, oracle)
    binding = fcl.bridge_event_binding(rows, events, context.config)
    assert binding["variant"] == context.config.eval_variant == 1 and binding["rows"] == 336
    assert binding["row_ids_sha256"] == fb.digest([r["id"] for r in rows])
    assert binding["row_context_sha256"] == fb.digest([[r["split_name"], r["context"]] for r in rows])
    for name in fcl.EVENT_FIELDS:
        entry = binding["event_arrays"][name]
        assert entry["dtype"] == events[name].dtype.str and entry["shape"] == [336, 28]
        assert entry["sha256"] == fcl._array_sha(events[name])
    assert "raw events only" in binding["teacher_quantities"]
    # the roster-ordered arrays really are each row's context record, not a re-derived summary
    for i in (0, 100, 335):
        record = oracle.raw_context_set(rows[i]["split_name"]).arrays
        assert all(np.array_equal(events[name][i], record[name][rows[i]["context"]])
                   for name in fcl.EVENT_FIELDS)
    other_config = dataclasses.replace(context.config, eval_variant=2)
    other = fcl.bridge_event_binding(rows, fcl.bridge_event_rows(rows, fcl.PanelOracle(other_config)),
                                     other_config)
    assert other["variant"] == 2 and other["events_sha256"] != binding["events_sha256"]
    header, _, features, _ = bridge_parts(context)
    schema = fcl._bridge_cache_args(rows, features, header, "scratch_s0", fcl.FEATURE_FUSED,
                                    context.config, binding)["space"].schema["raw_observation_variant"]
    assert schema["variant"] == 1 and schema["events_sha256"] == binding["events_sha256"]
    assert schema["event_arrays"] == binding["event_arrays"] and schema["row_ids_sha256"]
    with pytest.raises(ValueError):                  # a row that does not name its own context refuses
        broken = [dict(r) for r in rows]
        broken[0]["context"] = (broken[0]["context"] + 1) % 14
        fcl.bridge_event_rows(broken, fcl.PanelOracle(context.config))


def test_an_existing_cache_is_adopted_only_when_its_whole_semantic_contract_matches(context, tmp_path):
    header, rows, features, binding = bridge_parts(context)
    directory, rid, kind = tmp_path / "caches", "scratch_s0", fcl.FEATURE_FUSED
    args = (rows, features, header, rid, kind, context.config, binding)
    first = fcl.write_bridge_cache(directory, *args)
    assert first["adopted"] is False
    npz = directory / f"{rid}.cache.npz"
    before = snapshot(directory)
    again = fcl.write_bridge_cache(directory, *args)
    assert again["adopted"] is True and again["cache_sha256"] == first["cache_sha256"]
    assert again["file_sha256"] == file_sha256(npz) == first["file_sha256"], "the file hash is preserved"
    assert snapshot(directory) == before
    other_config = dataclasses.replace(context.config, eval_variant=2)
    other_binding = fcl.bridge_event_binding(
        rows, fcl.bridge_event_rows(rows, fcl.PanelOracle(other_config)), other_config)
    refusals = {
        "another raw evidence variant": (rows, features, header, rid, kind, context.config, other_binding),
        "a smaller roster": (rows[:-1], features[:-1], header, rid, kind, context.config, binding),
        "another learned state": (rows, features, {**header, "state_sha256": "cd" * 32}, rid, kind,
                                  context.config, binding),
        "another extractor file": (rows, features, {**header, "weights_file_sha256": "ef" * 32}, rid,
                                   kind, context.config, binding),
        "another arm and its supervision": (rows, features, {**header, "arm": "I"}, rid, kind,
                                            context.config, binding),
        "features that differ in the last digits": (rows, features + 1e-6, header, rid, kind,
                                                    context.config, binding)}
    for name, arguments in refusals.items():
        with pytest.raises(ValueError, match="never overwritten"):
            fcl.write_bridge_cache(directory, *arguments)
        assert snapshot(directory) == before, name


def test_a_numerically_equal_cache_with_falsely_label_free_provenance_is_refused(context, tmp_path):
    header, rows, features, binding = bridge_parts(context, arm="I")
    directory, rid = tmp_path / "caches", "I_s0"
    args = fcl._bridge_cache_args(rows, features, header, rid, fcl.FEATURE_FUSED, context.config, binding)
    assert args["space"].supervision.status == fb.HISTORY_DOCUMENTED
    false = dataclasses.replace(args["space"], supervision=fb.SupervisionHistory(
        status=fb.HISTORY_LABEL_FREE, description="claims it never saw a label", exposure_status="none"))
    directory.mkdir()
    fm.write_feature_cache(directory / f"{rid}.cache.json", directory / f"{rid}.cache.npz",
                           **{**args, "space": false})
    before = snapshot(directory)
    with pytest.raises(ValueError, match="never overwritten"):
        fcl.write_bridge_cache(directory, rows, features, header, rid, fcl.FEATURE_FUSED, context.config,
                               binding)
    assert snapshot(directory) == before
    honest = tmp_path / "honest"
    assert fcl.write_bridge_cache(honest, rows, features, header, rid, fcl.FEATURE_FUSED, context.config,
                                  binding)["adopted"] is False


def test_bridge_writers_never_replace_a_foreign_file_or_a_protected_input(context, tmp_path):
    _, rows, _, _ = bridge_parts(context)
    arguments = (rows, context.config, ["F_s0"], "F_s0")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    foreign = inputs / "candidates.csv"
    foreign.write_text("mine,keep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="never replaced"):
        fcl.write_bridge_inputs(inputs, *arguments)
    assert foreign.read_text(encoding="utf-8") == "mine,keep\n"
    assert [p.name for p in inputs.iterdir()] == ["candidates.csv"], "nothing else was written"
    fresh = tmp_path / "fresh"
    paths = fcl.write_bridge_inputs(fresh, *arguments)
    first = snapshot(fresh)
    assert set(paths) == {"candidates", "train_outcomes", "manifest", "plan"}
    fcl.write_bridge_inputs(fresh, *arguments)                       # identical text: adopted untouched
    assert snapshot(fresh) == first
    with pytest.raises(ValueError, match="same file"):               # a protected destination, directly
        fcl.write_bridge_inputs(tmp_path / "aliased", *arguments,
                                protected=[tmp_path / "aliased" / "candidates.csv"])
    assert not (tmp_path / "aliased").exists(), "refused before anything was created"
    keep = tmp_path / "precious.csv"
    keep.write_text("precious\n", encoding="utf-8")
    links = tmp_path / "links"
    links.mkdir()
    try:
        os.link(keep, links / "candidates.csv")
    except (OSError, NotImplementedError):
        pytest.skip("hard links are not available on this file system")
    with pytest.raises(ValueError, match="same file"):               # a hard link to a protected input
        fcl.write_bridge_inputs(links, *arguments, protected=[keep])
    assert keep.read_text(encoding="utf-8") == "precious\n" and [p.name for p in links.iterdir()] == [
        "candidates.csv"]


@pytest.mark.skipif(os.name != "nt", reason="letter-case aliases exist only on case-insensitive systems")
def test_a_letter_case_alias_of_a_protected_input_is_refused(context, tmp_path):
    _, rows, _, _ = bridge_parts(context)
    with pytest.raises(ValueError, match="same file"):
        fcl.write_bridge_inputs(tmp_path / "out", rows, context.config, ["F_s0"], "F_s0",
                                protected=[tmp_path / "out" / "CANDIDATES.CSV"])
    assert not (tmp_path / "out").exists()


def test_the_supervision_history_of_a_representation_is_documented(context):
    header = {"arm": "Ic", "cost": {"canonical_answers": {"factual": 30, "counterfactual": 28,
                                                          "marginal": 0}}}
    history = fcl.bridge_supervision(header)
    assert history.status == fb.HISTORY_DOCUMENTED and len(history.documented) == 2
    assert any("CORRUPTED" in entry["assay"] for entry in history.documented)
    assert [e["label_count"] for e in history.documented] == [30, 28]
    assert fcl.bridge_supervision({"arm": "scratch", "cost": {}}).status == fb.HISTORY_LABEL_FREE
    m = fcl.bridge_supervision({"arm": "M", "cost": {"canonical_answers": {
        "factual": 5, "marginal": 20, "counterfactual": 0}}})
    assert [e["label_count"] for e in m.documented] == [5, 20]
    assert all("distinct" in e["assay"] for e in m.documented)


# ---------------------------------------------------------------------------
# the whole pipeline: stage isolation, label isolation, the real Build A matrix
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    config = small_config(seeds=(0,))
    out = tmp_path_factory.mktemp("pipeline") / "run"
    run = fcl.CausalRun.open(out, config)
    run.prepare(deep_verify=True)
    run.run([s for s in fcl.STAGES if s != "prepare"])
    return run, out, config


def test_every_stage_completes_and_the_report_retains_every_arm(pipeline):
    run, out, config = pipeline
    assert all(run._complete(s) for s in fcl.STAGES if s != "report")
    report = fb.read_json(out / fcl.REPORT_NAME)
    assert set(report["costs"]) == {fcl.run_id(a, 0) for a in fcl.ARMS}
    assert set(report["evaluation"]["headlines"]) == {fcl.run_id(a, 0) for a in fcl.ARMS} | {"scratch_s0"}
    assert set(report["evaluation"]["paired_seed_differences"]) == {
        "I-F", "I-A", "I-Ic", "I-M", "A-F", "M-F", "Ic-F", "A-Ic"}
    assert all(report["integrity"]["stages"].values()) and report["integrity"]["frozen_states_unchanged"]
    assert report["claims"]["fidelity"] and report["interpretation"]
    assert report["tapes"]["corruption"]["0"]["events"] == 32


def test_the_corrupted_supervision_and_teacher_cost_are_in_the_prepare_record(pipeline):
    run, _, _ = pipeline
    record = run.state["stages"]["prepare"]
    assert record["world"]["source"] == 1176 and record["functions"] == {
        "source": 48, "primary": 27, "validation": 6, "transfer": 12}
    assert record["orbits"] == {"source": 10, "primary": 3, "validation": 1, "transfer": 1}
    assert record["deep_verified"] is True
    cost = record["teacher_cost"]["0"]
    assert cost["hybrid_posterior_computations"] == 32 and cost["probability_evaluations"] == 64


def test_held_out_panels_open_only_after_every_source_state_is_frozen(pipeline):
    run, out, _ = pipeline
    evaluation = fb.read_json(out / "evaluation" / "evaluation.json")
    freeze = run.state["stages"]["freeze"]["frozen"]
    assert evaluation["frozen_before_evaluation"] == {rid: v["state_sha256"] for rid, v in freeze.items()}
    assert "primary" in evaluation["opened_splits"] and "validation" in evaluation["opened_splits"]
    assert "transfer_eval" not in evaluation["opened_splits"]
    transfer = fb.read_json(out / "transfer" / "transfer.json")
    assert "transfer_eval" not in transfer["opened_before_adapter_freeze"]
    assert "transfer_eval" in transfer["opened_after_adapter_freeze"]
    assert transfer["integrity"]["adapter_removal_restores_N0"] is True
    recorded = run.state["stages"]["transfer"]["adapters_frozen"]
    assert set(recorded) == {f"{rid}|{n}" for rid in freeze for n in (16, 64)}
    for key, value in recorded.items():
        rid, budget = key.split("|")
        header = fb.read_json(out / "transfer" / "adapters" / f"{rid}_n{budget}.adapter.json")
        assert header["header_sha256"] == value["header_sha256"]


def test_transfer_reports_nested_budgets_contexts_and_labels_separately(pipeline):
    _, out, _ = pipeline
    transfer = fb.read_json(out / "transfer" / "transfer.json")
    assert transfer["budgets"] == [0, 16, 64]
    assert transfer["selection"]["N16"]["contexts_covered"] == 14
    assert transfer["selection"]["N64"]["labels_per_context_max"] == 5
    assert transfer["selection"]["N0"]["derived_labels"] == 0
    for rid, entry in transfer["results"].items():
        assert set(entry) == {"N0", "N16", "N64"}
        assert entry["N0"]["new_composition"]["cases"] == 168
    assert transfer["scratch_control"] and "scratch_s0" in transfer["results"]
    assert set(transfer["derived"]["per_run"]["I_s0"]) == {"N16", "N64"}
    assert "I-scratch" in transfer["derived"]["paired_vs_scratch_and_F"]
    for key, record in transfer["adapters"].items():
        assert record["steps"] == 5 and record["selection"]["derived_labels"] == record["budget"]


def test_the_exported_caches_pass_the_real_build_a_roster_join_and_comparison(pipeline):
    run, out, config = pipeline
    base = out / "export"
    summary = fb.read_json(base / "bridge_summary.json")
    assert summary["roster"] == {**summary["roster"], "candidates": 336, "train": 168, "eval": 168,
                                 "contexts": 28, "queries_per_context": 12}
    assert summary["feature"] == fcl.FEATURE_FUSED and summary["reference"] == "F_s0"
    counts = summary["bridge"]["status_counts"]
    assert counts["failed"] == 0 and counts["cells"] == 6 * 2 * 2
    cells = summary["bridge"]["cells"]
    assert len(cells) == counts["ok"] and all(np.isfinite(c["readout_mse"]) for c in cells)
    manifest = fm.load_matrix_manifest(base / "inputs" / "matrix_manifest.json")
    table = fb.read_candidates(base / "inputs" / "candidates.csv", alphabet=manifest.alphabet,
                               length=manifest.length)
    assert manifest.length == fe.identity_length() == 11
    for rid in summary["representations"]:
        cache = fm.load_feature_cache(base / "caches" / f"{rid}.cache.json")
        assert fm.check_cache_against_roster(cache, table, manifest)
        assert cache.representation_id == rid and cache.space.dimension == config.head_hidden
        assert cache.extractor["kind"] == "synthetic" and cache.extractor["checkpoint_sha256"]
        assert cache.space.schema["raw_observation_variant"]["variant"] == config.eval_variant
        assert "teacher_quantities_in_features" in cache.space.schema
        assert "penultimate" in cache.activation["location"]
        assert all(name.startswith("z") for name in cache.space.names)
        assert cache.conditioning["quality"]["values"] == [None] * 336
        expected = fb.HISTORY_LABEL_FREE if rid.startswith("scratch") else fb.HISTORY_DOCUMENTED
        assert cache.space.supervision.status == expected
    per_group = {}
    for sequence, group in zip(table.sequences, table.groups):
        per_group.setdefault(group, []).append(sequence)
    assert len(per_group) == 28 and all(len(set(v)) == 12 for v in per_group.values())
    results = fb.read_json(base / "matrix" / fm.RESULTS_NAME)
    assert results["status_counts"]["failed"] == 0 and all(
        item["comparison_to_reference"] is not None for item in results["cells"])
    assert (base / "inputs" / "eval_outcomes.csv").exists()
    freeze = run.state["stages"]["export"]["freeze"]
    assert fm.verify_freeze(base / "matrix")["index"]["index_sha256"] == freeze["index_sha256"]


def test_matrix_evaluation_outcomes_were_not_needed_to_freeze_the_exported_representations(pipeline):
    _, out, _ = pipeline
    base = out / "export"
    inputs = fb.read_json(base / "matrix" / fm.INPUTS_NAME)
    assert inputs["evaluation_outcomes"].startswith("not opened")
    assert "eval_outcomes" not in json.dumps(inputs["files"])


def test_rerunning_every_completed_stage_adopts_it_without_rewriting_a_byte(pipeline):
    run, out, config = pipeline
    before = snapshot(out)
    again = fcl.CausalRun.open(out, config)
    again.prepare(deep_verify=True)
    again.run([s for s in fcl.STAGES if s != "prepare"])
    after = snapshot(out)
    assert before.keys() == after.keys()
    assert [k for k in before if before[k] != after[k]] == []


def test_a_different_configuration_world_or_code_cannot_continue_an_output_directory(pipeline):
    _, out, config = pipeline
    with pytest.raises(ValueError):
        fcl.CausalRun.open(out, dataclasses.replace(config, lr=1e-3))
    with pytest.raises(ValueError):
        fcl.CausalRun.open(out, dataclasses.replace(config, seeds=(0, 1)))
    state = fb.read_json(out / "stage_state.json")
    assert {"config_sha256", "world_sha256", "code_sha256"} <= set(state)


def test_stages_cannot_run_out_of_order(tmp_path):
    run = fcl.CausalRun.open(tmp_path / "fresh", small_config(seeds=(0,), arms=("F",)))
    for stage in ("train", "freeze", "evaluate", "transfer", "export", "report"):
        with pytest.raises(ValueError):
            getattr(run, stage)()
    with pytest.raises(ValueError):
        run.run(["nonsense"])
    assert not (tmp_path / "fresh" / "evaluation").exists()


def test_a_tampered_frozen_state_is_refused_by_every_later_stage(pipeline, tmp_path):
    _, out, config = pipeline
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    weights = copy / "frozen" / "I_s0.pt"
    raw = bytearray(weights.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    weights.write_bytes(bytes(raw))
    run = fcl.CausalRun.open(copy, config)
    for stage in ("evaluate", "transfer", "export", "report"):
        with pytest.raises(ValueError):
            getattr(run, stage)()
    inspected = fcl.CausalRun.open(out, config)
    assert inspected.frozen_models()


def test_a_tampered_stored_tape_is_refused_when_the_context_is_rebuilt(pipeline, tmp_path):
    _, out, config = pipeline
    copy = tmp_path / "tamper_tape"
    shutil.copytree(out, copy)
    npz = copy / "tapes" / "seed_0.npz"
    with np.load(npz) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["function"][0] = (arrays["function"][0] + 1) % 48
    np.savez(npz, **arrays)
    with pytest.raises(ValueError):
        fcl.CausalRun.open(copy, config).train()


def test_held_out_contexts_are_labelled_only_after_the_source_states_are_frozen(tmp_path, monkeypatch):
    """A spy on the exact teacher records which labelled signatures it ever processed, stage by stage."""
    world = fe.q7_world()
    source = set(world.source)
    stages = []
    original = fe.world_posterior

    def spy(events, prior=fe.UNIFORM_PRIOR):
        events = tuple(events)
        if events:
            try:
                stages[-1][1].add(fe.signature_of_events(events))
            except ValueError:
                stages[-1][1].add(("restricted",))
        return original(events, prior)
    monkeypatch.setattr(fe, "world_posterior", spy)
    config = small_config(seeds=(0,), arms=("F", "I"))
    run = fcl.CausalRun.open(tmp_path / "isolation", config)
    for stage in ("prepare", "train", "freeze", "evaluate", "transfer"):
        stages.append((stage, set()))
        getattr(run, stage)()
    four = {name: {s for s in sigs if len(s) == 4} for name, sigs in stages}
    primary = set(world.centers_by_split["primary"])
    validation = set(world.centers_by_split["validation"])
    fit = set(world.centers_by_split["transfer_fit"])
    held = set(world.centers_by_split["transfer_eval"])
    # building the source tapes labels SOURCE signatures only; training and freezing label nothing
    assert four["prepare"] and four["prepare"] <= source
    assert not four["train"] and not four["freeze"]
    assert not (primary | validation | fit | held) & four["prepare"]
    # only the evaluate stage labels the primary and validation panels, and it never touches transfer
    assert primary <= four["evaluate"] and validation <= four["evaluate"]
    assert not (fit | held) & four["evaluate"]
    # the transfer stage labels the transfer panels (their order inside the stage is checked from the
    # recorded opening lists of the transfer artifact)
    assert fit <= four["transfer"] and held <= four["transfer"]
    assert not (primary | validation) & four["transfer"]
    transfer = fb.read_json(tmp_path / "isolation" / "transfer" / "transfer.json")
    assert transfer["opened_before_adapter_freeze"] == ["transfer_fit"]


# ---------------------------------------------------------------------------
# persistence on the PUBLIC stage paths: published bytes, export inputs, adoption, re-verification
# ---------------------------------------------------------------------------

def flip_byte(path):
    raw = bytearray(Path(path).read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    Path(path).write_bytes(bytes(raw))


def posix(files):
    return {key.replace("\\", "/"): value for key, value in files.items()}


def forget_export(work):
    """Make the next export a RESUME: forget the export stage record and its summary."""
    state = fb.read_json(work / "stage_state.json")
    state["stages"].pop("export", None)
    fb.write_json(work / "stage_state.json", state)
    summary = work / "export" / "bridge_summary.json"
    if summary.exists():
        summary.unlink()


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """prepare, train, freeze and transfer are done; the export has not started (two arms, one seed)."""
    config = small_config(seeds=(0,), arms=("F", "I"))
    out = tmp_path_factory.mktemp("staged") / "run"
    run = fcl.CausalRun.open(out, config)
    run.prepare()
    run.run(["train", "freeze", "transfer"])
    return out, config


def staged_copy(staged, tmp_path):
    out, config = staged
    work = tmp_path / "work"
    shutil.copytree(out, work)
    return work, config


def assert_nothing_published(out, state_before):
    assert (out / "stage_state.json").read_bytes() == state_before
    assert not (out / "config.json").exists() and not (out / "world.json").exists()


def test_prepare_refuses_an_orphan_foreign_tape_archive_and_publishes_nothing(tmp_path):
    """An orphan archive of another run used to receive a header that claimed it: ``prepare`` marked
    itself complete from its in-memory tape and training proceeded until a reopen exposed the lie."""
    config = small_config(seeds=(0,), arms=("F",))
    out = tmp_path / "orphan"
    (out / "tapes").mkdir(parents=True)
    foreign = out / "tapes" / "seed_0.npz"
    np.savez(foreign, p_cf=np.zeros(3))
    before = foreign.read_bytes()
    run = fcl.CausalRun.open(out, config)
    state_before = (out / "stage_state.json").read_bytes()
    with pytest.raises(ValueError, match="no header is published"):
        run.prepare()
    assert foreign.read_bytes() == before
    assert [p.name for p in (out / "tapes").iterdir()] == ["seed_0.npz"], "no header, no staging debris"
    assert not (out / "init").exists()
    assert_nothing_published(out, state_before)
    assert not run._complete("prepare") and run._context is None
    with pytest.raises(ValueError):
        run.train()
    with pytest.raises(ValueError, match="no header is published"):        # and a fresh object refuses too
        fcl.CausalRun.open(out, config).prepare()


def test_prepare_refuses_a_same_shape_archive_with_one_value_changed(pipeline, tmp_path):
    _, source, config = pipeline
    out = tmp_path / "corrupted"
    (out / "tapes").mkdir(parents=True)
    with np.load(source / "tapes" / "seed_0.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["p_cf"][0] += 0.125
    corrupted = out / "tapes" / "seed_0.npz"
    np.savez(corrupted, **arrays)
    before = corrupted.read_bytes()
    run = fcl.CausalRun.open(out, config)
    state_before = (out / "stage_state.json").read_bytes()
    with pytest.raises(ValueError, match="differs from this tape"):
        run.prepare()
    assert corrupted.read_bytes() == before and not (out / "tapes" / "seed_0.json").exists()
    assert not (out / "init").exists()
    assert_nothing_published(out, state_before)


def test_prepare_adopts_an_identical_orphan_archive_untouched_and_binds_its_actual_file_hash(
        pipeline, tmp_path):
    original, source, config = pipeline
    out = tmp_path / "adopt"
    (out / "tapes").mkdir(parents=True)
    copied = out / "tapes" / "seed_0.npz"
    shutil.copyfile(source / "tapes" / "seed_0.npz", copied)
    before, digest = copied.read_bytes(), file_sha256(copied)
    run = fcl.CausalRun.open(out, config)
    record = run.prepare()
    assert copied.read_bytes() == before and file_sha256(copied) == digest
    assert record["tape_file_sha256"] == {"0": digest} and (out / "tapes" / "seed_0.json").is_file()
    assert record["tape_sha256"] == original.state["stages"]["prepare"]["tape_sha256"]
    assert run.context().tapes[0].sha256 == record["tape_sha256"]["0"]
    flip_byte(copied)                       # bytes that are not the published ones refuse on reopen
    with pytest.raises(ValueError):
        fcl.CausalRun.open(out, config).context()


def test_prepare_decides_every_refusal_before_publishing_anything(tmp_path):
    config = small_config(seeds=(0,), arms=("F",))
    junk_state = tmp_path / "junk_state"
    (junk_state / "init").mkdir(parents=True)
    junk = junk_state / "init" / "seed_0.pt"
    junk.write_bytes(b"not a state")
    with pytest.raises(ValueError, match="never replaced"):
        fcl.CausalRun.open(junk_state, config).prepare()
    assert junk.read_bytes() == b"not a state"
    assert not (junk_state / "tapes").exists() and not (junk_state / "config.json").exists()
    foreign_config = tmp_path / "foreign_config"
    foreign_config.mkdir()
    (foreign_config / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="never replaced"):
        fcl.CausalRun.open(foreign_config, config).prepare()
    assert (foreign_config / "config.json").read_text(encoding="utf-8") == "{}"
    assert not (foreign_config / "tapes").exists() and not (foreign_config / "init").exists()
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    with pytest.raises(ValueError, match="same file"):
        fcl.CausalRun.open(guarded, config, protected=[guarded / "tapes" / "seed_0.npz"]).prepare()
    assert not (guarded / "tapes").exists() and not (guarded / "init").exists()


PROTECTED_DESTINATIONS = (
    "export/inputs/candidates.csv", "export/inputs/train_outcomes.csv", "export/inputs/matrix_plan.json",
    "export/inputs/matrix_manifest.json", "export/inputs/eval_outcomes.csv",
    "export/caches/I_s0.cache.json", "export/caches/F_s0.cache.npz",
    "export/matrix/matrix_freeze_index.json", "export/matrix/matrix_evaluation_report.json",
    "export/bridge_summary.json")


@pytest.mark.parametrize("relative", PROTECTED_DESTINATIONS)
def test_export_preflights_every_predictable_destination_against_protected_inputs(
        staged, tmp_path, relative):
    work, config = staged_copy(staged, tmp_path)
    before = snapshot(work)
    run = fcl.CausalRun.open(work, config, protected=[work / relative])
    with pytest.raises(ValueError, match="same file"):
        run.export()
    assert snapshot(work) == before, "nothing was created, replaced or recorded"
    assert not (work / "export").exists() and not run._complete("export")


def test_export_refuses_a_hard_linked_protected_input_unchanged(staged, tmp_path):
    work, config = staged_copy(staged, tmp_path)
    keep = tmp_path / "precious.csv"
    keep.write_text("precious\n", encoding="utf-8")
    (work / "export" / "inputs").mkdir(parents=True)
    try:
        os.link(keep, work / "export" / "inputs" / "candidates.csv")
    except (OSError, NotImplementedError):
        pytest.skip("hard links are not available on this file system")
    before = snapshot(work)
    with pytest.raises(ValueError, match="same file"):
        fcl.CausalRun.open(work, config, protected=[keep]).export()
    assert snapshot(work) == before and keep.read_text(encoding="utf-8") == "precious\n"


def test_export_never_replaces_a_foreign_input_file(staged, tmp_path):
    work, config = staged_copy(staged, tmp_path)
    (work / "export" / "inputs").mkdir(parents=True)
    foreign = work / "export" / "inputs" / "train_outcomes.csv"
    foreign.write_text("candidate_id,utility\nmine,0.5\n", encoding="utf-8")
    before = snapshot(work)
    with pytest.raises(ValueError, match="never replaced"):
        fcl.CausalRun.open(work, config).export()
    assert snapshot(work) == before, "refused in the preflight: no cache, freeze or record was written"


def test_a_foreign_evaluation_outcomes_file_is_refused_unchanged_after_the_freeze(staged, tmp_path):
    work, config = staged_copy(staged, tmp_path)
    (work / "export" / "inputs").mkdir(parents=True)
    foreign = work / "export" / "inputs" / "eval_outcomes.csv"
    foreign.write_text("candidate_id,utility\nmine,0.5\n", encoding="utf-8")
    run = fcl.CausalRun.open(work, config)
    with pytest.raises(ValueError, match="never replaced"):
        run.export()
    assert foreign.read_text(encoding="utf-8") == "candidate_id,utility\nmine,0.5\n"
    assert not (work / "export" / "bridge_summary.json").exists()
    assert not (work / "export" / "matrix" / fm.RESULTS_NAME).exists() and not run._complete("export")


def test_export_completes_with_unrelated_protected_paths_and_discloses_the_panel_reuse(staged, tmp_path):
    work, config = staged_copy(staged, tmp_path)
    unrelated = tmp_path / "unrelated_input.csv"
    unrelated.write_text("x\n", encoding="utf-8")
    run = fcl.CausalRun.open(work, config, protected=[unrelated])
    summary = run.export()
    assert unrelated.read_text(encoding="utf-8") == "x\n" and run._complete("export")
    reuse = summary["panel_reuse"]
    assert "ALREADY opened and scored" in reuse["statement"] and "not a fresh confirmation" in reuse[
        "statement"]
    assert reuse["statement"] == fcl.CLAIMS["panel_reuse"] == fcl.PANEL_REUSE
    assert reuse["transfer_evaluation_already_scored_by_the_transfer_stage"] is True
    assert reuse["opened_before_matrix_freeze"] == ["transfer_fit"]
    assert reuse["opened_by_this_stage"] == ["transfer_fit", "transfer_eval"]
    assert run.state["stages"]["export"]["opened_before_matrix_freeze"] == ["transfer_fit"]
    transfer = fb.read_json(work / "transfer" / "transfer.json")          # the earlier audit is kept as is
    assert transfer["opened_before_adapter_freeze"] == ["transfer_fit"]
    assert "transfer_eval" in transfer["opened_after_adapter_freeze"]
    assert summary["raw_evidence"]["variant"] == config.eval_variant
    for rid in summary["representations"]:
        cache = fm.load_feature_cache(work / "export" / "caches" / f"{rid}.cache.json")
        binding = cache.space.schema["raw_observation_variant"]
        assert binding["events_sha256"] == summary["raw_evidence"]["events_sha256"]
        assert binding["variant"] == config.eval_variant and binding["rows"] == 336


def test_held_out_answers_are_materialized_only_after_the_matrix_freeze_during_export(
        staged, tmp_path, monkeypatch):
    work, config = staged_copy(staged, tmp_path)
    held = set(fe.q7_world().centers_by_split["transfer_eval"])
    processed, before_freeze = set(), []
    posterior, run_matrix = fe.world_posterior, fm.run_matrix

    def spy(events, prior=fe.UNIFORM_PRIOR):
        events = tuple(events)
        if events:
            try:
                processed.add(fe.signature_of_events(events))
            except ValueError:
                pass
        return posterior(events, prior)

    def freeze_spy(**kwargs):
        before_freeze.append(bool(processed & held))
        return run_matrix(**kwargs)
    monkeypatch.setattr(fe, "world_posterior", spy)
    monkeypatch.setattr(fm, "run_matrix", freeze_spy)
    fcl.CausalRun.open(work, config).export()
    assert before_freeze == [False], "feature export and the freeze never touched a held-out answer"
    assert processed & held == held, "the held-out answers were opened afterwards, for the evaluation"


def test_a_valid_interrupted_export_resumes_and_adopts_everything_it_finds(pipeline, tmp_path):
    _, out, config = pipeline
    work = tmp_path / "resume"
    shutil.copytree(out, work)
    original = posix(snapshot(work))
    forget_export(work)
    (work / "export" / "matrix" / fm.RESULTS_NAME).unlink()
    (work / "export" / "inputs" / "eval_outcomes.csv").unlink()
    resumed = fcl.CausalRun.open(work, config)
    summary = resumed.export()
    after = posix(snapshot(work))
    assert after.keys() == original.keys()
    recomputed = {"export/bridge_summary.json", "export/matrix/matrix_evaluation_report.json",
                  "stage_state.json"}
    assert [k for k in original if k not in recomputed and original[k] != after[k]] == [], (
        "caches, inputs, the freeze and the report document were adopted byte for byte")
    for key in recomputed:
        assert json.loads(original[key]) == json.loads(after[key]), key
    assert json.loads(fb.canonical_json(summary)) == fb.read_json(work / "export" / "bridge_summary.json")
    assert resumed._complete("export")


def test_a_same_plan_freeze_from_different_learned_features_is_not_adopted(staged, tmp_path, monkeypatch):
    work, config = staged_copy(staged, tmp_path)
    real = fcl.bridge_features
    monkeypatch.setattr(fcl, "bridge_features",
                        lambda model, rows, oracle, **kw: real(model, rows, oracle, **kw) + 0.05)
    fcl.CausalRun.open(work, config).export()                  # a whole export over SHIFTED features
    monkeypatch.undo()
    plan = fb.read_json(work / "export" / "matrix" / fm.INDEX_NAME)["plan_sha256"]
    shutil.rmtree(work / "export" / "caches")                  # test-owned scratch: only the freeze remains
    forget_export(work)
    before = snapshot(work)
    with pytest.raises(ValueError, match="other feature caches"):
        fcl.CausalRun.open(work, config).export()
    assert snapshot(work) == before and not (work / "export" / "caches").exists()
    assert fb.read_json(work / "export" / "matrix" / fm.INDEX_NAME)["plan_sha256"] == plan


def test_a_freeze_over_other_offered_training_labels_is_not_adopted(staged, tmp_path, monkeypatch):
    """The plan and the feature caches are the same; only one offered training label differs, so only the
    freeze's own recorded common identity can tell (the old adoption compared the plan alone)."""
    work, config = staged_copy(staged, tmp_path)
    real = fcl.bridge_rows

    def altered(oracle, **kw):
        rows = real(oracle, **kw)
        first = next(r for r in rows if r["label"] is not None)
        first["label"] = 1.0 - first["label"]
        return rows
    monkeypatch.setattr(fcl, "bridge_rows", altered)
    fcl.CausalRun.open(work, config).export()                  # a whole export over one altered label
    monkeypatch.undo()
    shutil.rmtree(work / "export" / "inputs")                   # test-owned scratch: caches and freeze remain
    forget_export(work)
    before = snapshot(work)
    with pytest.raises(ValueError, match="offered training labels"):
        fcl.CausalRun.open(work, config).export()
    assert snapshot(work) == before and not (work / "export" / "inputs").exists()


def test_a_matrix_report_scored_under_other_truth_is_not_relabelled(staged, tmp_path):
    work, config = staged_copy(staged, tmp_path)
    fcl.CausalRun.open(work, config).export()
    genuine = (work / "export" / "inputs" / "eval_outcomes.csv").read_text(encoding="utf-8").splitlines()
    other = tmp_path / "other_truth.csv"
    other.write_text("candidate_id,utility\n" + "".join(
        f"{i},{1.0 - float(v)!r}\n" for i, v in (line.split(",") for line in genuine[1:])), encoding="utf-8")
    fm.evaluate_matrix(work / "export" / "matrix", other, overwrite=True)    # the SAME freeze, other truth
    forget_export(work)
    before = posix(snapshot(work))
    with pytest.raises(ValueError, match="other evaluation outcomes"):
        fcl.CausalRun.open(work, config).export()
    after = posix(snapshot(work))
    assert {k: v for k, v in before.items() if k != "stage_state.json"} == {
        k: v for k, v in after.items() if k != "stage_state.json"}, "the old result is neither rewritten nor relabelled"
    assert fb.read_json(work / "stage_state.json")["stages"]["export"]["complete"] is False
    assert not (work / "export" / "bridge_summary.json").exists()


def edit_json(path, change):
    document = fb.read_json(path)
    change(document)
    fb.write_json(path, document)


def edit_report(work):
    def change(document):
        next(cell for cell in document["cells"] if cell["status"] == "ok")["readout"]["mse"] += 0.01
    edit_json(work / "export" / "matrix" / fm.RESULTS_NAME, change)


EXPORT_TAMPERS = {
    "a_feature_cache_array_is_corrupted": lambda w: flip_byte(w / "export" / "caches" / "F_s0.cache.npz"),
    "a_feature_cache_header_is_deleted": lambda w: (w / "export" / "caches" / "I_s0.cache.json").unlink(),
    "an_input_file_is_deleted": lambda w: (w / "export" / "inputs" / "train_outcomes.csv").unlink(),
    "the_evaluation_outcomes_are_edited": lambda w: (
        w / "export" / "inputs" / "eval_outcomes.csv").write_text("candidate_id,utility\nx,0.5\n",
                                                                  encoding="utf-8"),
    "the_matrix_index_is_edited": lambda w: edit_json(
        w / "export" / "matrix" / fm.INDEX_NAME, lambda d: d["counts"].update(cells=d["counts"]["cells"] + 1)),
    "a_matrix_cell_is_deleted": lambda w: sorted((w / "export" / "matrix" / "cells").iterdir())[0].unlink(),
    "the_matrix_report_is_edited": edit_report,
    "the_stored_summary_is_edited": lambda w: edit_json(
        w / "export" / "bridge_summary.json", lambda d: d["bridge"].update(metric="edited"))}


@pytest.mark.parametrize("name", sorted(EXPORT_TAMPERS))
def test_a_completed_export_reverifies_every_tracked_artifact_before_anything_is_accepted(
        pipeline, tmp_path, name):
    _, out, config = pipeline
    work = tmp_path / "work"
    shutil.copytree(out, work)
    run = fcl.CausalRun.open(work, config)
    assert run.export()["bridge"] and run.report()["bridge"]               # clean: adopted, nothing rewritten
    EXPORT_TAMPERS[name](work)
    before = snapshot(work)
    for stage in ("export", "report"):
        with pytest.raises(ValueError):
            getattr(run, stage)()                                          # the SAME object, a second call
        assert snapshot(work) == before, (name, stage)
    with pytest.raises(ValueError):
        fcl.CausalRun.open(work, config).export()
    assert snapshot(work) == before


TRANSFER_TAMPERS = {
    "an_adapter_weights_file_is_corrupted": lambda w: flip_byte(w / "transfer" / "adapters" / "I_s0_n16.pt"),
    "an_adapter_weights_file_is_deleted": lambda w: (w / "transfer" / "adapters" / "F_s0_n64.pt").unlink(),
    "an_adapter_header_is_deleted": lambda w: (
        w / "transfer" / "adapters" / "scratch_s0_n16.adapter.json").unlink(),
    "an_adapter_header_is_edited": lambda w: edit_json(
        w / "transfer" / "adapters" / "A_s0_n64.adapter.json",
        lambda d: d["record"].update(last_loss=d["record"]["last_loss"] + 1.0))}


@pytest.mark.parametrize("name", sorted(TRANSFER_TAMPERS))
def test_a_completed_transfer_reverifies_every_tracked_adapter_on_every_entry(pipeline, tmp_path, name):
    _, out, config = pipeline
    work = tmp_path / "work"
    shutil.copytree(out, work)
    run = fcl.CausalRun.open(work, config)
    assert run.transfer()["results"] and run.report()["transfer"]          # clean: adopted
    TRANSFER_TAMPERS[name](work)
    before = snapshot(work)
    for stage in ("transfer", "report"):
        with pytest.raises(ValueError):
            getattr(run, stage)()                                          # the SAME object, a second call
        assert snapshot(work) == before, (name, stage)
    with pytest.raises(ValueError):
        fcl.CausalRun.open(work, config).transfer()
    assert snapshot(work) == before


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def load_cli():
    spec = importlib.util.spec_from_file_location("functional_causal_learning_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_world_prints_the_verified_frozen_world(capsys, tmp_path):
    cli = load_cli()
    target = tmp_path / "world.json"
    assert cli.main(["world", "--output", str(target)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["verified"]["source"] == 1176 and printed["functions"]["source"] == 48
    assert printed["world_sha256"] == fe.world_sha256() and fb.read_json(target)["functions"]
    assert cli.main(["world", "--output", str(target)]) == 2             # never overwrites a file


def test_cli_tape_run_and_inspect_use_the_same_pipeline(capsys, tmp_path):
    cli = load_cli()
    config = tmp_path / "config.json"
    fb.write_json(config, small_config().document())
    out = tmp_path / "out"
    assert cli.main(["tape", "--output-dir", str(out), "--config", str(config), "--seeds", "0",
                     "--arms", "F", "I"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["prepare"]["complete"] and printed["prepare"]["corruption"]["0"]["events"] == 32
    assert cli.main(["run", "--output-dir", str(out), "--config", str(config), "--seeds", "0",
                     "--arms", "F", "I", "--stages", "train", "freeze", "--single-thread"]) == 0
    capsys.readouterr()
    assert cli.main(["inspect", "--output-dir", str(out), "--config", str(config), "--seeds", "0",
                     "--arms", "F", "I", "--verify"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["stages"]["freeze"] is True and set(inspected["verified"]["freeze"]) == {
        "F_s0", "I_s0", "scratch_s0"}
    assert cli.main(["inspect", "--output-dir", str(tmp_path / "missing")]) == 2
    assert not (tmp_path / "missing").exists()


def test_cli_refuses_a_bad_configuration(capsys, tmp_path):
    cli = load_cli()
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"surprise": 1}), encoding="utf-8")
    assert cli.main(["tape", "--output-dir", str(tmp_path / "x"), "--config", str(config)]) == 2
    assert "error" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["run", "--output-dir", str(tmp_path / "y"), "--stages", "nonsense"])


def test_cli_inspect_verify_also_verifies_the_tracked_adapters(pipeline, capsys, tmp_path):
    _, out, config = pipeline
    cli = load_cli()
    path = tmp_path / "config.json"
    fb.write_json(path, config.document())
    assert cli.main(["inspect", "--output-dir", str(out), "--config", str(path), "--verify"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["verified"]["transfer"].startswith("tracked adapters verified")
    work = tmp_path / "copy"
    shutil.copytree(out, work)
    flip_byte(work / "transfer" / "adapters" / "I_s0_n16.pt")
    assert cli.main(["inspect", "--output-dir", str(work), "--config", str(path), "--verify"]) == 2
    assert "error" in capsys.readouterr().err


def test_the_selftest_runs_actual_learning_hidden_interventions_and_a_real_matrix(tmp_path):
    report = fcl.selftest(tmp_path)
    failed = [c["name"] for c in report["checks"] if not c["holds"]]
    assert report["passed"], failed
    names = {c["name"] for c in report["checks"]}
    assert any("tamper" in n for n in names) and any("resume" in n for n in names)
    assert any("bridge" in n for n in names) and any("transfer" in n for n in names)
    assert Path(report["pipeline_report"]).is_file()
