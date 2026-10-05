"""Build D: the shared-parameter learner, its frozen transfer into the matrix, and the contracts around it.

These tests qualify MECHANICS on a tiny, randomly initialised CPU DPLM's real token cache: the
student boundary, padding and batch-order invariance, shared-gradient wiring, exact F/S parity,
resume and tamper refusal, tape/label independence of target outcomes and the stage ordering of the
combined freeze. They never assert that the composed arm (or any representation) wins; a negative or
inconclusive result is a valid result.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.util
import json
import math
import os
import pickle
import shutil
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from smallAntibodyGen.experiments import _functional_transfer_synthetic as synth
from smallAntibodyGen.experiments import functional_baseline as fb
from smallAntibodyGen.experiments import functional_composition as fc
from smallAntibodyGen.experiments import functional_matrix as fm
from smallAntibodyGen.experiments import functional_native_export as fne
from smallAntibodyGen.experiments import functional_shared_transfer as fst

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "functional_shared_transfer.py"
SOURCE = fc.AssayDeclaration(target="src-target", assay="src-assay", units="u", scale="linear")


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """Bit-reproducible CPU arithmetic for the exact-parity assertions."""
    with fst.single_thread():
        yield


def small_config(**changes):
    values = dict(steps=4, batch=8, aux_batch=8, instances_per_sequence=2, lam=0.1, lr=3e-3,
                  clip_norm=1.0, width=16, heads=2, ffn=32, blocks=2, source_seeds=(0, 1),
                  acquisition_seeds=(0, 1), label_budgets=(4, 8), checkpoint_every=2,
                  probe_steps=(0, 2), panel_instances=1, panel_sequences=6,
                  bootstrap_replicates=20, extraction_batch=16)
    values.update(changes)
    return fst.TransferConfig(**values)


@pytest.fixture(scope="module")
def world_dir(tmp_path_factory):
    world = synth.build_world(synth.SELFTEST_SPEC)
    paths = synth.write_inputs(world, tmp_path_factory.mktemp("transfer_world") / "inputs")
    return world, paths


@pytest.fixture(scope="module")
def inputs(world_dir):
    return fst._inputs_from(world_dir[1])


@pytest.fixture(scope="module")
def opened(world_dir, inputs, tmp_path_factory):
    return fst.TransferRun.open(inputs, tmp_path_factory.mktemp("opened") / "run", small_config())


@pytest.fixture(scope="module")
def ctx(opened):
    return opened.ctx


def adam_steps(optimizer):
    return {int(entry["step"]) for entry in optimizer.state.values()}


def run_to_end(run):
    while run.cursor < run.total_steps:
        run.macrostep()
    return run


# ---------------------------------------------------------------------------
# configuration and declared arithmetic
# ---------------------------------------------------------------------------

def test_pilot_defaults_are_the_declared_bounded_pilot():
    config = fst.pilot_config()
    assert (config.steps, config.batch, config.lam) == (256, 32, 0.1)
    assert (config.width, config.heads, config.ffn, config.blocks, config.dropout) == (
        64, 4, 128, 2, 0.0)
    assert config.arms == ("F", "C", "S", "Q", "R") and config.source_seeds == (0, 1, 2)
    assert config.acquisition_seeds == (0, 1, 2) and config.label_budgets == (16, 64)
    assert config.aux_batch_size == 32 and config.instances_per_sequence * 4 == 16
    assert config.resolved_probe_steps() == (0, 128, 255)
    representations = 1 + len(config.arms) * len(config.source_seeds)
    assert representations * len(config.label_budgets) * len(config.acquisition_seeds) == 96


@pytest.mark.parametrize("document", [
    {"surprise": 1}, {"arms": ["C", "Q"]}, {"arms": ["F", "F"]}, {"arms": ["F", "Z"]},
    {"source_seeds": [0, 0]}, {"source_seeds": []}, {"label_budgets": [0]}, {"width": 10, "heads": 4},
    {"lam": 0}, {"steps": 0}, {"dropout": 1.0}, {"include_rbf": "yes"}, {"steps": 4,
                                                                      "probe_steps": [4]},
    {"bootstrap_replicates": 3}])
def test_config_is_strict(document):
    with pytest.raises(ValueError):
        fst.TransferConfig.from_document(document)


def test_config_documents_round_trip_and_the_checkpoint_period_is_not_an_identity_choice():
    config = small_config()
    assert fst.TransferConfig.from_document(config.document()) == config
    assert "checkpoint_every" not in config.identity_document()
    assert "arms" not in config.training_document() and "source_seeds" not in config.training_document()


def test_the_pilot_arithmetic_is_8192_derived_answers_and_8192_functional_presentations():
    world = synth.build_world(synth.PILOT_SPEC)
    layout = fc.Layout(4, 4, "ACDE")
    table = fc.generate_events(layout, world.sequences("source_aux"), universe=fc.UNIVERSE_TRAIN,
                               instances_per_sequence=4, seed=0)
    assert len(table) == 8192 and table.unique_sequences() == 512
    assert fc.coverage_report(table)["complete"]
    assert len({fst.answer_label_id(event) for event in table.events}) == 8192
    # 512 sequences x 16 UNIQUE canonical questions, and the same again on replay
    assert len({fc.event_id(e.candidate_id, e.program, e.query) for e in table.events}) == 8192
    assert {count for count in np.bincount(
        [world.roles["source_aux"].index(cid) for cid in table.candidate_ids()]).tolist()} == {16}
    assert world.excluded == ()                  # the seed-0 draw selects no constant sequence
    replay = fc.generate_events(layout, world.sequences("source_aux"), universe=fc.UNIVERSE_TRAIN,
                                instances_per_sequence=4, seed=0)
    assert replay.sha256 == table.sha256 and len(replay) == 8192
    ids = tuple(world.roles["source_train"])
    tape = fc.build_seed_tape(
        seed=0, steps=256, functional_steps=256, batch=32, aux_batch=32, functional_ids=ids,
        functional_labels=[world.outcomes["source"][cid] for cid in ids], table=table,
        mask_binding={"padded_tokens": 13, "padded_instructions": 3, "token_lengths": {}})
    assert tape.aux_batches.size == 8192 and len(set(tape.aux_batches.reshape(-1).tolist())) == 8192
    counts = np.bincount(tape.functional_batches.reshape(-1), minlength=128)
    assert tape.functional_batches.size == 8192 and set(counts.tolist()) == {64}   # 64 passes of 128
    assert 0 < tape.q_report["changed_values"] < 8192


# ---------------------------------------------------------------------------
# the student boundary
# ---------------------------------------------------------------------------

def test_the_student_receives_only_native_features_masks_metadata_and_typed_instructions(ctx):
    assert fst.STUDENT_FIELDS == (
        "features", "token_mask", "token_role", "token_chain", "token_index", "instr_kind",
        "instr_a", "instr_b", "instr_s", "instr_t", "instr_mask")
    batch, labels = fst.functional_batch(ctx, ctx.seed_tapes[0], 0)
    assert all(isinstance(getattr(batch, name), torch.Tensor) for name in fst.STUDENT_FIELDS)
    assert not any(hasattr(batch, name) for name in
                   ("labels", "codes", "candidate_id", "ids", "group", "split", "sequence",
                    "residues", "native_ids", "trace"))
    assert labels.shape == (8,)
    assert {f.name for f in dataclasses.fields(fst.TokenTable)} == {
        "ids", "position", "offsets", "features", "role", "chain", "index", "hidden", "max_tokens",
        "max_heavy", "max_light", "cache_sha256"}
    kinds = set(batch.instr_kind[batch.instr_mask].tolist())
    assert kinds == {fc.KIND_FUNCTION}


def test_token_metadata_carries_no_residue_information(opened):
    table, cache = opened.ctx.table, opened.cache
    first = slice(int(table.offsets[0]), int(table.offsets[1]))
    second = slice(int(table.offsets[1]), int(table.offsets[2]))
    assert (cache.heavy[0], cache.light[0]) != (cache.heavy[1], cache.light[1])
    for array in (table.role, table.chain, table.index):
        assert array[first].tolist() == array[second].tolist()
    assert not np.array_equal(table.features[first], table.features[second])
    assert set(table.role.tolist()) == {0, 1, 2, 3} and int(table.index.max()) == 3


def test_the_student_refuses_teacher_records_and_untyped_input(ctx):
    net = fst.build_student(ctx.spec, seed=1)
    with pytest.raises(ValueError, match="only a StudentBatch"):
        net.encode(ctx.events.events[0])
    with pytest.raises(ValueError, match="only a StudentBatch"):
        net.encode({"features": torch.zeros(1)})


def test_no_teacher_record_reaches_the_student_during_training(ctx, monkeypatch):
    seen, real = [], fst.StudentNet.encode

    def spy(self, batch):
        seen.append(batch)
        return real(self, batch)
    monkeypatch.setattr(fst.StudentNet, "encode", spy)
    for arm in ("C", "R", "S", "Q"):
        fst.ArmRun(ctx, arm, 0).macrostep()
    assert seen and all(type(batch) is fst.StudentBatch for batch in seen)
    teacher = (fc.TeacherEvent, fc.Program, fc.Resolution, fc.EventTable, fc.SeedTape)
    assert not any(isinstance(getattr(batch, name), teacher) for batch in seen
                   for name in fst.STUDENT_FIELDS)


# ---------------------------------------------------------------------------
# variable-length rows: padding and batch composition
# ---------------------------------------------------------------------------

VARIABLE_ROWS = (("v0", "EVQLV", "DIQM", "g0", "train"), ("v1", "EVQ", "DIQMTQ", "g1", "train"),
                 ("v2", "EVQLVES", "", "g2", "train"), ("v3", "QV", "D", "g3", "eval"),
                 ("v4", "ACDEAC", "ACD", "g4", "eval"))


@pytest.fixture(scope="module")
def variable(tmp_path_factory):
    directory = tmp_path_factory.mktemp("variable")
    document = {
        "schema": fne.PLAN_SCHEMA, "representation_id": "variable", "conditioning_mode": "none",
        "activation": {"kind": "final_hidden", "name": None},
        "feature_supervision": {"status": "label_free", "description": "random",
                                "sequence_only_exposure": {"status": "none"}},
        "extractor": {"kind": "synthetic", "checkpoint_sha256": None, "source_sha256": None,
                      "description": "tiny random-init CPU DPLM"},
        "rows": [{"candidate_id": i, "heavy": h, "light": l, "group_id": g, "split": s,
                  "condition": None} for i, h, l, g, s in VARIABLE_ROWS]}
    fne.export_dplm_tokens(fne.tiny_model(1), fne.parse_export_plan(document), directory / "v.json",
                           directory / "v.npz", batch_size=2)
    cache = fne.load_token_cache(directory / "v.json")
    table = fst.build_token_table(cache)
    spec = fst.StudentSpec(hidden_in=table.hidden, width=16, heads=2, ffn=32, blocks=2, dropout=0.0,
                           n_symbols=4, max_heavy=table.max_heavy, max_light=table.max_light)
    return table, fst.build_student(spec, seed=3)


def test_rows_really_have_unequal_lengths_and_masks_come_from_lengths(variable):
    table, net = variable
    lengths = [table.length(row) for row in range(len(table.ids))]
    assert len(set(lengths)) > 1 and table.max_tokens == max(lengths)
    batch = fst.function_batch_for(table, np.arange(len(table.ids)))
    assert batch.token_mask.sum(dim=1).tolist() == lengths
    assert table.max_heavy == 7 and table.max_light == 6


def test_function_features_are_invariant_to_batch_order_size_and_padding_amount(variable):
    table, net = variable
    rows = np.arange(len(table.ids))
    base = fst.extract_function_features(table, net, rows, batch_size=5)
    permutation = np.array([3, 0, 4, 1, 2])
    permuted = fst.extract_function_features(table, net, rows[permutation], batch_size=2)
    np.testing.assert_allclose(permuted, base[permutation], atol=2e-5)
    np.testing.assert_allclose(fst.extract_function_features(table, net, rows, batch_size=1), base,
                               atol=2e-5)
    padded = fst.extract_function_features(table, net, rows, token_pad=table.max_tokens + 9)
    np.testing.assert_allclose(padded, base, atol=2e-5)
    assert float(np.abs(base[0] - base[3]).max()) > 1e-4         # different rows differ


def test_padded_positions_cannot_influence_the_output_whatever_their_values(variable):
    table, net = variable
    batch = fst.function_batch_for(table, np.arange(len(table.ids)), token_pad=table.max_tokens + 4)
    with torch.no_grad():
        before = net.encode(batch).clone()
        noisy = batch.features.clone()
        noisy[~batch.token_mask] = 1.0e3 * torch.randn(int((~batch.token_mask).sum()), table.hidden)
        after = net.encode(dataclasses.replace(batch, features=noisy))
    torch.testing.assert_close(before, after, atol=2e-5, rtol=0)


def test_nothing_is_truncated_to_fit_a_padded_width(variable):
    table, _ = variable
    with pytest.raises(ValueError, match="nothing is truncated"):
        fst.function_batch_for(table, np.arange(len(table.ids)), token_pad=table.max_tokens - 1)


def test_removing_the_auxiliary_heads_leaves_function_outputs_unchanged(variable):
    table, net = variable
    batch = fst.function_batch_for(table, np.arange(len(table.ids)))
    with torch.no_grad():
        before, features = net.functional_forward(batch).clone(), net.encode(batch).clone()
        stripped = copy.deepcopy(net)
        stripped.remove_auxiliary_heads()
        assert torch.equal(stripped.functional_forward(batch), before)
        assert torch.equal(stripped.encode(batch), features)
    assert stripped.parameter_counts()["total"] < net.parameter_counts()["total"]
    assert not any(name.startswith(("res_head", "eq_head")) for name in stripped.state_dict())


def test_removing_heads_after_training_also_leaves_the_function_pathway_unchanged(ctx):
    run = run_to_end(fst.ArmRun(ctx, "C", 0))
    rows = np.arange(len(ctx.table.ids))[:6]
    before = fst.predict_function(ctx.table, run.func_net, rows)
    stripped = copy.deepcopy(run.func_net)
    stripped.remove_auxiliary_heads()
    np.testing.assert_array_equal(fst.predict_function(ctx.table, stripped, rows), before)


# ---------------------------------------------------------------------------
# loss definitions
# ---------------------------------------------------------------------------

def test_loss_definitions_by_hand():
    prediction, labels = torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0])
    assert float(fst.functional_loss(prediction, labels)) == pytest.approx((1.0 + 4.0) / 2)
    residue, equality = torch.zeros(3, 4), torch.zeros(3)
    targets = fst.AuxTargets(is_equality=torch.tensor([False, True, True]),
                             codes=torch.tensor([1, 1, 0]))
    assert float(fst.auxiliary_loss(residue, equality, targets)) == pytest.approx(
        (math.log(4) + 2 * math.log(2)) / 3)
    only_res = fst.AuxTargets(is_equality=torch.tensor([False, False]), codes=torch.tensor([0, 3]))
    only_eq = fst.AuxTargets(is_equality=torch.tensor([True, True]), codes=torch.tensor([0, 1]))
    assert float(fst.auxiliary_loss(torch.zeros(2, 4), torch.zeros(2), only_res)) \
        == pytest.approx(math.log(4))
    assert float(fst.auxiliary_loss(torch.zeros(2, 4), torch.zeros(2), only_eq)) \
        == pytest.approx(math.log(2))


def test_tensor_state_digest_is_content_sensitive_and_key_order_free():
    a = {"x": torch.arange(4.0), "y": {"z": torch.ones(2)}}
    b = {"y": {"z": torch.ones(2)}, "x": torch.arange(4.0)}
    assert fst.tensor_state_digest(a) == fst.tensor_state_digest(b)
    c = {"x": torch.arange(4.0), "y": {"z": torch.ones(2)}}
    c["x"][0] = 1e-7
    assert fst.tensor_state_digest(c) != fst.tensor_state_digest(a)
    assert fst.tensor_state_digest({"x": torch.arange(4.0).double()}) \
        != fst.tensor_state_digest({"x": torch.arange(4.0)})
    with pytest.raises(ValueError):
        fst.tensor_state_digest({"x": object()})


# ---------------------------------------------------------------------------
# the five arms
# ---------------------------------------------------------------------------

def test_every_arm_clones_the_same_initial_state_into_separate_parameter_objects(ctx):
    runs = {arm: fst.ArmRun(ctx, arm, 0) for arm in fst.ARMS}
    reference = fst.tensor_state_digest(runs["F"].state_payload()["nets"]["func"])
    assert reference == ctx.init_sha256[0] == fst.tensor_state_digest(ctx.init_states[0])
    owners = []
    for arm, run in runs.items():
        assert fst.tensor_state_digest(run.state_payload()["nets"]["func"]) == reference
        owners.append({id(p) for p in run.func_net.parameters()})
        if arm == "S":
            assert fst.tensor_state_digest(run.state_payload()["nets"]["aux"]) == reference
            assert not {id(p) for p in run.aux_net.parameters()} & {
                id(p) for p in run.func_net.parameters()}
            assert not {p.data_ptr() for p in run.aux_net.parameters()} & {
                p.data_ptr() for p in run.func_net.parameters()}
        elif arm in fst.SHARED_ARMS:
            assert run.aux_net is run.func_net
        else:
            assert run.aux_net is None
    for left in range(len(owners)):
        for right in range(left + 1, len(owners)):
            assert not owners[left] & owners[right]
    other_seed = fst.ArmRun(ctx, "F", 1)
    assert fst.tensor_state_digest(other_seed.state_payload()["nets"]["func"]) != reference


def test_all_arms_start_from_the_same_functional_loss_on_the_same_first_batch(ctx):
    losses = {}
    for arm in fst.ARMS:
        run = fst.ArmRun(ctx, arm, 0)
        run.macrostep()
        losses[arm] = run.log[0]["functional_loss"]
    assert len(set(losses.values())) == 1


def test_S_functional_pathway_and_optimizer_equal_F_exactly_after_every_macrostep(ctx):
    f_run, s_run = fst.ArmRun(ctx, "F", 0), fst.ArmRun(ctx, "S", 0)
    c_run = fst.ArmRun(ctx, "C", 0)
    assert f_run.functional_pathway_digest() == s_run.functional_pathway_digest()
    for _ in range(ctx.config.steps):
        f_run.macrostep()
        s_run.macrostep()
        c_run.macrostep()
        assert f_run.functional_pathway_digest() == s_run.functional_pathway_digest()
        assert f_run.log[-1]["functional_loss"] == s_run.log[-1]["functional_loss"]
        assert f_run.log[-1]["grad_norm"] == s_run.log[-1]["grad_norm"]
        assert f_run.log[-1]["clip_coefficient"] == s_run.log[-1]["clip_coefficient"]
    assert fst.tensor_state_digest(dict(s_run.aux_net.state_dict())) \
        != fst.tensor_state_digest(dict(s_run.func_net.state_dict()))
    assert c_run.functional_pathway_digest() != f_run.functional_pathway_digest()


def test_changing_S_auxiliary_labels_cannot_change_its_functional_weights(ctx, monkeypatch):
    reference = run_to_end(fst.ArmRun(ctx, "F", 0)).functional_pathway_digest()
    real = fst.auxiliary_batch

    def altered(ctx_, tape, arm, step):
        batch, targets = real(ctx_, tape, arm, step)
        return batch, fst.AuxTargets(is_equality=targets.is_equality,
                                     codes=torch.zeros_like(targets.codes))
    monkeypatch.setattr(fst, "auxiliary_batch", altered)
    changed = run_to_end(fst.ArmRun(ctx, "S", 0))
    assert changed.functional_pathway_digest() == reference
    monkeypatch.setattr(fst, "auxiliary_batch", real)
    honest = run_to_end(fst.ArmRun(ctx, "S", 0))
    assert fst.tensor_state_digest(dict(honest.aux_net.state_dict())) \
        != fst.tensor_state_digest(dict(changed.aux_net.state_dict()))


def test_auxiliary_gradients_reach_the_same_tensors_the_function_query_uses(ctx):
    run = fst.ArmRun(ctx, "C", 0)
    fbatch, labels = fst.functional_batch(ctx, run.tape, 0)
    abatch, atargets = fst.auxiliary_batch(ctx, run.tape, "C", 0)
    names = [n for n, _ in run.func_net.named_parameters() if not n.startswith(fst.HEAD_PREFIXES)]
    shared = run.func_net.shared_parameters()
    f_loss = fst.functional_loss(run.func_net.functional_forward(fbatch), labels)
    residue, equality = run.func_net.auxiliary_forward(abatch)
    a_loss = fst.auxiliary_loss(residue, equality, atargets)
    f_grads = dict(zip(names, torch.autograd.grad(f_loss, shared, allow_unused=True)))
    a_grads = dict(zip(names, torch.autograd.grad(a_loss, shared, allow_unused=True)))
    for name in ("proj.weight", "blocks.0.query.weight", "blocks.1.down.weight", "ln_f.weight",
                 "coord_emb.weight", "kind_emb.weight"):
        assert f_grads[name] is not None and float(f_grads[name].abs().sum()) > 0, name
        assert a_grads[name] is not None and float(a_grads[name].abs().sum()) > 0, name
    separate = fst.ArmRun(ctx, "S", 0)
    abatch, atargets = fst.auxiliary_batch(ctx, separate.tape, "S", 0)
    residue, equality = separate.aux_net.auxiliary_forward(abatch)
    fst.auxiliary_loss(residue, equality, atargets).backward()
    assert all(p.grad is None for p in separate.func_net.parameters())


def test_an_auxiliary_only_step_changes_C_function_features_but_not_S(ctx):
    rows = [ctx.table.position[cid] for cid in ctx.functional_ids[:4]]
    shared, separate = fst.ArmRun(ctx, "C", 0), fst.ArmRun(ctx, "S", 0)
    shared_before, separate_before = shared.function_features(rows), separate.function_features(rows)
    shared.auxiliary_only_step()
    separate.auxiliary_only_step()
    assert float(np.abs(shared.function_features(rows) - shared_before).max()) > 0.0
    np.testing.assert_array_equal(separate.function_features(rows), separate_before)


def test_the_gradient_probe_preserves_grads_optimizer_modes_buffers_and_rng(ctx):
    run = fst.ArmRun(ctx, "C", 0)
    run.macrostep()
    parameters = list(run.func_net.parameters())
    parameters[0].grad = torch.full_like(parameters[0], 7.0)
    parameters[1].grad = None
    grads = [None if p.grad is None else p.grad.clone() for p in parameters]
    run.func_net.train()
    run.func_net.blocks[0].eval()
    flags = [module.training for module in run.func_net.modules()]
    optimizer_before = fst.tensor_state_digest(run.func_opt.state_dict())
    weights_before = fst.tensor_state_digest(dict(run.func_net.state_dict()))
    rng_before = torch.get_rng_state()
    buffers_before = [b.clone() for b in run.func_net.buffers()]
    fbatch, labels = fst.functional_batch(ctx, run.tape, 1)
    abatch, atargets = fst.auxiliary_batch(ctx, run.tape, "C", 1)
    out = run._probe(fbatch, labels, abatch, atargets)
    assert out["functional_grad_norm_shared"] > 0 and out["auxiliary_grad_norm_shared"] > 0
    cosine = out["cosine_functional_auxiliary_shared"]
    assert cosine is None or -1.0 - 1e-6 <= cosine <= 1.0 + 1e-6
    for before, p in zip(grads, parameters):
        if before is None:
            assert p.grad is None
        else:
            assert torch.equal(p.grad, before)
    assert [module.training for module in run.func_net.modules()] == flags
    assert fst.tensor_state_digest(run.func_opt.state_dict()) == optimizer_before
    assert fst.tensor_state_digest(dict(run.func_net.state_dict())) == weights_before
    assert torch.equal(torch.get_rng_state(), rng_before) and torch.is_grad_enabled()
    assert all(torch.equal(a, b) for a, b in zip(buffers_before, run.func_net.buffers()))


def test_declared_steps_report_norms_cosines_clipping_and_update_norms(ctx):
    run = run_to_end(fst.ArmRun(ctx, "C", 0))
    assert [d["step"] for d in run.diagnostics] == list(ctx.config.resolved_probe_steps())
    for diagnostic in run.diagnostics:
        for key in ("functional_grad_norm_shared", "auxiliary_grad_norm_shared",
                    "auxiliary_weighted_grad_norm_shared", "grad_norm_before_clip",
                    "clip_coefficient", "update_norm_shared", "update_norm_functional_net",
                    "functional_loss", "auxiliary_loss"):
            assert math.isfinite(diagnostic[key]), key
        assert 0.0 < diagnostic["clip_coefficient"] <= 1.0 and diagnostic["update_norm_shared"] > 0
        assert diagnostic["clipped"] == (diagnostic["clip_coefficient"] < 1.0)
        assert diagnostic["shared_tensors"] == len(run.func_net.shared_parameters())
    separate = run_to_end(fst.ArmRun(ctx, "S", 0)).diagnostics[0]
    assert separate["cosine_functional_auxiliary_shared"] is None and "cosine_note" in separate
    assert separate["update_norm_auxiliary_net"] > 0 and "auxiliary_clip_coefficient" in separate
    plain = run_to_end(fst.ArmRun(ctx, "F", 0)).diagnostics[0]
    assert "auxiliary_grad_norm_shared" not in plain


@pytest.mark.parametrize("arm", ["F", "C", "S", "Q", "R"])
def test_one_joint_step_per_macrostep_for_shared_arms_and_independent_steps_for_S(ctx, arm):
    run = fst.ArmRun(ctx, arm, 0)
    for _ in range(3):
        run.macrostep()
    accounting = run.accounting
    expected = run.expected_adam_steps()
    assert accounting["macrosteps"] == 3
    # one Adam step per macrostep for everything the loss reached; a head whose answer type did not
    # occur in some batch lags by exactly the tape-implied count (never a manufactured zero gradient)
    assert adam_steps(run.func_opt) == {n for n in expected["func"].values() if n}
    assert int(run.func_opt.state[run.func_net.func_head.weight]["step"]) == 3
    assert int(run.func_opt.state[run.func_net.proj.weight]["step"]) == 3
    if arm == "S":
        assert adam_steps(run.aux_opt) == {n for n in expected["aux"].values() if n}
        assert (accounting["optimizer_steps"], accounting["backward_calls"]) == (6, 6)
    else:
        assert run.aux_opt is None
        assert (accounting["optimizer_steps"], accounting["backward_calls"]) == (3, 3)
    expected_aux = 0 if arm == "F" else 3
    assert accounting["forward_calls"] == {"functional": 3, "auxiliary": expected_aux}
    assert accounting["functional_presentations"] == 3 * ctx.config.batch
    assert accounting["auxiliary_presentations"] == expected_aux * ctx.config.aux_batch_size
    assert accounting["scalar_targets_presented"]["functional"] == 3 * ctx.config.batch


def test_compute_and_label_units_are_reported_separately(ctx):
    costs = {arm: run_to_end(fst.ArmRun(ctx, arm, 0)).cost_report() for arm in fst.ARMS}
    steps, batch = ctx.config.steps, ctx.config.batch
    assert {c["functional_presentations"] for c in costs.values()} == {steps * batch}
    presented = len(set(ctx.seed_tapes[0].functional_batches[:steps].reshape(-1).tolist()))
    assert {c["unique_measured_outcomes_directly_presented"] for c in costs.values()} == {presented}
    # the functional bias is the mean of ALL source labels: initialisation consumes every one of them
    assert {c["unique_measured_outcomes_consumed_by_initialization"] for c in costs.values()} \
        == {len(ctx.functional_ids)}
    assert {c["unique_measured_source_outcomes"] for c in costs.values()} == {len(ctx.functional_ids)}
    assert costs["F"]["auxiliary_presentations"] == 0 and costs["F"]["unique_computational_answers"] == 0
    aux = {arm: costs[arm] for arm in ("C", "S", "Q", "R")}
    assert len({c["auxiliary_presentations"] for c in aux.values()}) == 1
    assert len({c["unique_computational_answers"] for c in aux.values()}) == 1
    assert len({c["unique_auxiliary_sequences"] for c in aux.values()}) == 1
    assert costs["C"]["unique_computational_answers"] <= steps * ctx.config.aux_batch_size
    assert costs["C"]["unique_auxiliary_sequences"] <= len(ctx.cohorts.roles["source_aux"])
    assert costs["S"]["parameters"]["total_parameters"] \
        == 2 * costs["F"]["parameters"]["functional_net"]["total"]
    assert costs["F"]["parameters"]["updated_parameters"] < costs["F"]["parameters"]["total_parameters"]
    assert costs["C"]["parameters"]["updated_parameters"] == costs["C"]["parameters"]["total_parameters"]
    assert costs["S"]["total_forward_calls"] == costs["C"]["total_forward_calls"] \
        == 2 * steps and costs["F"]["total_forward_calls"] == steps
    assert costs["S"]["optimizer_steps"] == 2 * steps and costs["C"]["optimizer_steps"] == steps
    assert all(isinstance(c["elapsed_cpu_seconds"], float) and c["elapsed_cpu_seconds"] >= 0.0
               for c in costs.values())            # the Windows CPU timer is coarse: no lower bound
    assert "never added together" in costs["C"]["units"]


def test_Q_uses_the_frozen_shuffled_targets_and_R_the_resolved_direct_instructions(ctx):
    tape = ctx.seed_tapes[0]
    assert sorted(tape.q_targets.tolist()) == sorted(ctx.event_codes.tolist())
    assert not np.array_equal(tape.q_targets, ctx.event_codes)
    changed_any = False
    for step in range(ctx.config.steps):
        c_batch, c_targets = fst.auxiliary_batch(ctx, tape, "C", step)
        q_batch, q_targets = fst.auxiliary_batch(ctx, tape, "Q", step)
        r_batch, r_targets = fst.auxiliary_batch(ctx, tape, "R", step)
        assert torch.equal(q_batch.instr_kind, c_batch.instr_kind)
        assert torch.equal(q_targets.is_equality, c_targets.is_equality)
        changed_any |= not torch.equal(q_targets.codes, c_targets.codes)
        assert torch.equal(r_targets.codes, c_targets.codes)               # the SAME answers
        assert torch.equal(r_targets.is_equality, c_targets.is_equality)
        assert torch.equal(r_batch.features, c_batch.features)             # the SAME features
        assert r_batch.instr_kind.shape == c_batch.instr_kind.shape == (ctx.config.aux_batch_size, 3)
        assert r_batch.token_mask.shape == c_batch.token_mask.shape
        assert bool((r_batch.instr_mask.sum(dim=1) == 1).all())
        direct = {fc.KIND_READ_ORIGINAL, fc.KIND_READ_LITERAL, fc.KIND_REL_ORIGINAL_ORIGINAL,
                  fc.KIND_REL_ORIGINAL_LITERAL, fc.KIND_REL_LITERAL_LITERAL}
        assert set(r_batch.instr_kind[r_batch.instr_mask].tolist()) <= direct
        composed = {fc.KIND_SET, fc.KIND_SWAP, fc.KIND_RES, fc.KIND_EQ}
        assert set(c_batch.instr_kind[c_batch.instr_mask].tolist()) <= composed
    assert changed_any


def test_C_and_R_share_the_same_derived_answer_identities(ctx):
    assert len(ctx.answer_ids) == len(ctx.events) == len(set(ctx.answer_ids))
    assert all(answer.split("#")[0] == event.candidate_id
               for answer, event in zip(ctx.answer_ids, ctx.events.events))
    assert len({answer.split("#")[0] for answer in ctx.answer_ids}) \
        < len(ctx.answer_ids)                      # NOT the candidate id reused for every answer


def test_the_continued_F_control_extends_F_and_discloses_its_extra_presentations(world_dir, inputs,
                                                                                tmp_path):
    config = small_config(continued_f_extra_steps=2)
    opened_fx = fst.TransferRun.open(inputs, tmp_path / "fx", config)
    context = opened_fx.ctx
    assert config.arm_list() == ("F", "C", "S", "Q", "R", "FX")
    assert "FX_s0" in opened_fx.run_ids() and context.seed_tapes[0].functional_batches.shape == (6, 8)
    f_run, fx_run = run_to_end(fst.ArmRun(context, "F", 0)), run_to_end(fst.ArmRun(context, "FX", 0))
    assert fx_run.total_steps == f_run.total_steps + 2
    assert [e["functional_loss"] for e in fx_run.log[:4]] == [e["functional_loss"] for e in f_run.log]
    cost = fx_run.cost_report()
    assert cost["functional_presentations"] == (4 + 2) * 8 and "continued_f_disclosure" in cost
    assert "compute" in cost["continued_f_disclosure"] and "continued_f_disclosure" not in f_run.cost_report()


# ---------------------------------------------------------------------------
# checkpoints, resume and tamper refusal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["C", "S"])
def test_interrupted_and_resumed_runs_equal_uninterrupted_runs_exactly(ctx, tmp_path, arm):
    straight = fst.train_arm(ctx, arm, 0, tmp_path / "straight")
    interrupted = fst.train_arm(ctx, arm, 0, tmp_path / "resumed", stop_after=1)
    assert interrupted.cursor == 1
    resumed = fst.train_arm(ctx, arm, 0, tmp_path / "resumed")
    assert resumed.cursor == resumed.total_steps
    assert straight.deterministic_result() == resumed.deterministic_result()
    assert straight.state_sha256() == resumed.state_sha256()
    again = fst.train_arm(ctx, arm, 0, tmp_path / "resumed")                 # a finished run resumes
    assert again.deterministic_result() == straight.deterministic_result()


def reseal(path, **changes):
    header = json.loads(Path(path).read_text(encoding="utf-8"))
    header.update(changes)
    body = {key: value for key, value in header.items() if key != "header_sha256"}
    header["header_sha256"] = fb.digest(body)
    fb.write_json(path, header)


@pytest.fixture()
def checkpointed(ctx, tmp_path):
    fst.train_arm(ctx, "C", 0, tmp_path / "ckpt", stop_after=2)
    return tmp_path / "ckpt"


def refuses_without_touching(ctx, directory, match, arm="C"):
    run = fst.ArmRun(ctx, arm, 0)
    before = (run.state_sha256(), run.cursor, copy.deepcopy(run.accounting))
    with pytest.raises(ValueError, match=match):
        fst.restore_checkpoint(run, directory)
    assert (run.state_sha256(), run.cursor, run.accounting) == before


def test_a_flipped_weight_byte_is_refused_before_anything_is_mutated(ctx, checkpointed):
    weights = next(checkpointed.glob("ckpt_*.pt"))
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0x01
    weights.write_bytes(bytes(data))
    refuses_without_touching(ctx, checkpointed, "SHA-256")


class Boom:
    pass


def test_a_pickled_object_in_place_of_the_weights_is_refused_and_never_unpickled(ctx, checkpointed):
    weights = next(checkpointed.glob("ckpt_*.pt"))
    weights.write_bytes(pickle.dumps(Boom()))
    refuses_without_touching(ctx, checkpointed, "SHA-256")


def test_a_resealed_header_with_a_wrong_cursor_is_refused(ctx, checkpointed):
    reseal(checkpointed / fst.CHECKPOINT_NAME, cursor=1)
    refuses_without_touching(ctx, checkpointed, "cursor")


def test_an_unsealed_header_edit_is_refused(ctx, checkpointed):
    header = json.loads((checkpointed / fst.CHECKPOINT_NAME).read_text(encoding="utf-8"))
    header["cursor"] = 1
    fb.write_json(checkpointed / fst.CHECKPOINT_NAME, header)
    refuses_without_touching(ctx, checkpointed, "header_sha256")


def test_a_checkpoint_of_another_arm_is_refused(ctx, checkpointed):
    refuses_without_touching(ctx, checkpointed, "different run", arm="S")


def tampered_context(opened, kind):
    """A context whose bindings differ from the original in exactly one respect."""
    ctx = opened.ctx
    tape = ctx.seed_tapes[0]
    fields = dict(seed=tape.seed, steps=tape.steps, functional_steps=tape.functional_steps,
                  batch=tape.batch, aux_batch=tape.aux_batch, functional_ids=tape.functional_ids,
                  functional_labels=tape.functional_labels,
                  functional_batches=tape.functional_batches, aux_batches=tape.aux_batches,
                  q_targets=tape.q_targets, q_report=tape.q_report,
                  event_table_sha256=tape.event_table_sha256, mask_binding=tape.mask_binding)
    if kind == "accumulation_membership":
        swapped = tape.aux_batches.copy()
        swapped[[0, 1]] = swapped[[1, 0]]
        fields["aux_batches"] = swapped
    elif kind == "labels":
        fields["functional_labels"] = tape.functional_labels + 0.25
    elif kind == "masks":
        binding = copy.deepcopy(tape.mask_binding)
        first = next(iter(binding["token_lengths"]))
        binding["token_lengths"][first] += 1
        fields["mask_binding"] = binding
    elif kind == "shuffled_targets":
        fields["q_targets"] = (tape.q_targets + 1) % 4
    if kind in ("accumulation_membership", "labels", "masks", "shuffled_targets"):
        return dataclasses.replace(ctx, seed_tapes={**ctx.seed_tapes, 0: fc.SeedTape(**fields)})
    if kind == "programs":
        other = fc.generate_events(ctx.layout, opened.cohorts.rows(opened.cache, "source_aux"),
                                   universe=fc.UNIVERSE_TRAIN, instances_per_sequence=2, seed=99)
        return dataclasses.replace(ctx, events=other)
    if kind == "cache_identity":
        return dataclasses.replace(ctx, cache_sha256="0" * 64)
    if kind == "config":
        return dataclasses.replace(ctx, training_sha256="1" * 64)
    # a DIFFERENT initial state that is consistent with its own recorded digest: a run can be built from
    # it (an inconsistent one is refused at creation, see the initial-state test), and only the
    # checkpoint's binding then tells it apart
    other = {name: (value + 1.0 if value.is_floating_point() else value.clone())
             for name, value in ctx.init_states[0].items()}
    return dataclasses.replace(ctx, init_states={**ctx.init_states, 0: other},
                               init_sha256={**ctx.init_sha256, 0: fst.tensor_state_digest(other)})


@pytest.mark.parametrize("kind", ["accumulation_membership", "labels", "masks", "shuffled_targets",
                                  "programs", "cache_identity", "config", "init"])
def test_changed_labels_masks_programs_cache_or_tape_membership_are_refused(
        opened, checkpointed, kind):
    bad = tampered_context(opened, kind)
    run = fst.ArmRun(bad, "C", 0)
    before = (run.state_sha256(), run.cursor)
    with pytest.raises(ValueError, match="bindings"):
        fst.restore_checkpoint(run, checkpointed)
    assert (run.state_sha256(), run.cursor) == before


def test_own_checkpoint_and_adapter_loading_is_weights_only(ctx, checkpointed, tmp_path, monkeypatch):
    seen, real = [], torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real(*args, **kwargs)
    monkeypatch.setattr(torch, "load", spy)
    run = fst.ArmRun(ctx, "C", 0)
    assert fst.restore_checkpoint(run, checkpointed) and seen == [True]
    finished = run_to_end(run)
    fst.freeze_adapter(finished, tmp_path / "adapters")
    fst.load_adapter(tmp_path / "adapters", "C_s0", ctx)
    assert seen == [True, True]


def test_adapters_are_frozen_hashed_written_once_and_verified_on_load(ctx, tmp_path):
    run = run_to_end(fst.ArmRun(ctx, "S", 0))
    header = fst.freeze_adapter(run, tmp_path / "adapters")
    with pytest.raises(ValueError, match="already exists"):
        fst.freeze_adapter(run, tmp_path / "adapters")
    adapter = fst.load_adapter(tmp_path / "adapters", "S_s0", ctx)
    assert adapter.aux_net is not adapter.func_net and not adapter.func_net.training
    assert all(not p.requires_grad for p in adapter.func_net.parameters())
    assert header["state_sha256_note"].startswith("a hash of the parameter TENSORS")
    weights = tmp_path / "adapters" / header["weights_file"]
    data = bytearray(weights.read_bytes())
    data[len(data) // 3] ^= 0x01
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        fst.load_adapter(tmp_path / "adapters", "S_s0", ctx)
    with pytest.raises(ValueError, match="only a finished run"):
        fst.freeze_adapter(fst.ArmRun(ctx, "F", 0), tmp_path / "other")


# ---------------------------------------------------------------------------
# derived supervision history
# ---------------------------------------------------------------------------

def test_a_label_free_native_history_becomes_documented_with_separate_unit_counts():
    native = fb.SupervisionHistory(status=fb.HISTORY_LABEL_FREE, description="random",
                                   exposure_status="none")
    history, costs = fst.derived_supervision(
        native, source=SOURCE, measured_ids=["m1", "m2"],
        computational_ids=["m1#a", "m1#b", "m2#a"], arm="C")
    assert history.status == fb.HISTORY_DOCUMENTED and len(history.documented) == 2
    measured, computational = history.documented
    assert (measured["label_count"], measured["label_ids"]) == (2, ["m1", "m2"])
    assert computational["label_count"] == 3 and len(set(computational["label_ids"])) == 3
    assert not set(measured["label_ids"]) & set(computational["label_ids"])
    assert history.exposure_status == "declared"
    assert costs["measured_source"]["count"] == 2 and costs["computational"]["count"] == 3


def test_a_documented_native_history_keeps_its_entries_and_exposure():
    native = fb.SupervisionHistory(
        status=fb.HISTORY_DOCUMENTED, description="d", exposure_status="declared",
        exposure_description="seq", documented=({"assay": "x", "target": "y", "label_count": 5,
                                                 "label_ids": None},))
    history, _ = fst.derived_supervision(native, source=SOURCE, measured_ids=["m1"],
                                         computational_ids=None, arm="F")
    assert history.documented[0]["assay"] == "x" and len(history.documented) == 2
    assert history.exposure_status == "declared" and "seq" in history.exposure_description


def test_an_unknown_native_history_stays_unknown_and_the_known_costs_go_in_the_schema():
    native = fb.SupervisionHistory(status=fb.HISTORY_UNKNOWN, description="nobody knows")
    history, costs = fst.derived_supervision(native, source=SOURCE, measured_ids=["m1", "m2"],
                                             computational_ids=["m1#a"], arm="Q")
    assert history.status == fb.HISTORY_UNKNOWN and history.documented == ()
    assert costs["native_history_status"] == "unknown" and "KNOWN new costs" in costs[
        "recorded_in_schema_only"]
    assert costs["measured_source"]["label_ids"] == ["m1", "m2"]
    assert costs["computational"]["label_ids"] == ["m1#a"]


def test_arm_labels_in_the_computational_entry_are_truthful():
    native = fb.SupervisionHistory(status=fb.HISTORY_LABEL_FREE, exposure_status="none")
    texts = {}
    for arm in ("C", "Q", "R"):
        history, _ = fst.derived_supervision(native, source=SOURCE, measured_ids=["m"],
                                             computational_ids=["m#1"], arm=arm)
        texts[arm] = history.documented[1]["assay"]
    assert "SHUFFLED" in texts["Q"] and "direct-read" in texts["R"] and "SHUFFLED" not in texts["C"]
    with pytest.raises(ValueError):
        fst.derived_supervision(native, source=SOURCE, measured_ids=["m", "m"],
                                computational_ids=None, arm="F")


# ---------------------------------------------------------------------------
# the pipeline, end to end, with the evaluation outcomes withheld until after the freeze
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    base = tmp_path_factory.mktemp("pipeline")
    world = synth.build_world(synth.SELFTEST_SPEC)
    paths = synth.write_inputs(world, base / "inputs", withhold_eval=True)
    config, log, real = small_config(), [], fb.read_outcomes

    def spy(path, *, label):
        log.append(("read", label, str(path)))
        return real(path, label=label)
    out = {"world": world, "paths": paths, "config": config, "log": log,
           "inputs": fst._inputs_from(paths)}
    with mock.patch.object(fb, "read_outcomes", spy):
        run = fst.TransferRun.open(out["inputs"], base / "run", config)
        for stage in ("prepare", "train", "features", "freeze"):
            log.append(("stage", stage))
            getattr(run, stage)()
        out["eval_exists_at_freeze"] = Path(paths["target_eval_outcomes"]).exists()
        out["pending_exists_at_freeze"] = (run.matrix_inputs / "target_eval_outcomes.pending").exists()
        out["verified_at_freeze"] = fm.verify_freeze(run.freeze_dir)
        out["adapter_files_at_freeze"] = {
            run_id: fb.read_json(run.adapters / f"{run_id}.adapter.json")["weights_file_sha256"]
            for run_id in run.run_ids()}
        shutil.copy(paths["held_eval_outcomes"], paths["target_eval_outcomes"])
        for stage in ("evaluate", "panels", "report"):
            log.append(("stage", stage))
            out[stage] = getattr(run, stage)()
    out["run"] = run
    return out


def stage_index(log, name):
    return log.index(("stage", name))


def test_every_target_representation_is_frozen_before_any_evaluation_outcome_is_opened(pipeline):
    log = pipeline["log"]
    assert not pipeline["eval_exists_at_freeze"] and not pipeline["pending_exists_at_freeze"]
    assert pipeline["verified_at_freeze"]["index"]["counts"]["cells"] == 44
    real_reads = [item for item in log if item[0] == "read" and "evaluation" in item[1]
                  and not item[2].endswith(".pending")]
    assert real_reads and all(log.index(item) > stage_index(log, "evaluate") for item in real_reads)
    for item in log:
        if item[0] == "read" and item[2].endswith(".pending"):
            assert stage_index(log, "freeze") < log.index(item) < stage_index(log, "evaluate")
    train_reads = [log.index(item) for item in log if item[0] == "read"
                   and item[1] == "target training outcomes"]
    assert train_reads and min(train_reads) > stage_index(log, "freeze")
    dev_reads = [log.index(item) for item in log if item[0] == "read"
                 and item[1] == "source development outcomes"]
    assert dev_reads and stage_index(log, "train") < min(dev_reads) < stage_index(log, "features")
    early = [item for item in log[:stage_index(log, "freeze")] if item[0] == "read"]
    assert {item[1] for item in early} <= {"source training outcomes",
                                           "source development outcomes"}


def test_the_combined_matrix_is_one_native_plus_every_arm_and_seed(pipeline):
    run = pipeline["run"]
    index = pipeline["verified_at_freeze"]["index"]
    assert sorted(index["representations"]) == sorted(["native"] + run.run_ids())
    assert len(run.run_ids()) == 10 and index["counts"] == {"cells": 44, "ok": 44,
                                                            "unsupported": 0, "failed": 0}
    assert index["plan"]["seeds"] == [0, 1] and index["plan"]["label_budgets"] == [4, 8]
    assert index["plan"]["reference"] == {"representation": "native", "map": "linear"}
    assert [m["id"] for m in index["plan"]["maps"]] == ["linear"]
    support = index["common"]["support"]
    assert support["ids"] == sorted(pipeline["world"].roles["target_eval"])
    assert index["common"]["offered"]["count"] == len(pipeline["world"].roles["target_train"])


def load_rep(pipeline, rep):
    record = pipeline["run"].state["stages"]["features"]["caches"][rep]
    return fm.load_feature_cache(record["header"], npz_path=record["npz"])


def test_every_cache_covers_the_exact_sorted_target_roster_and_no_source_row(pipeline):
    run, world = pipeline["run"], pipeline["world"]
    expected = tuple(sorted(world.roles["target_train"] + world.roles["target_eval"]))
    assert run.target_ids() == expected
    forbidden = set(world.roles["source_aux"]) | set(world.roles["source_dev"])
    for rep in ["native"] + run.run_ids():
        cache = load_rep(pipeline, rep)
        assert cache.ids == expected and not set(cache.ids) & forbidden
        assert cache.identity["length"] == 9 and cache.identity["alphabet"] == "ACDE|"


def test_the_native_control_binds_its_parent_cache_runtime_and_selected_roster(pipeline):
    run = pipeline["run"]
    cache = load_rep(pipeline, "native")
    schema = cache.space.schema
    assert schema["parent_token_cache_sha256"] == run.cache.cache_sha256 \
        == schema["token_cache_sha256"]
    assert schema["runtime_sha256"] == run.cache.runtime_sha256
    assert schema["token_map_sha256"] == run.cache.token_map_sha256
    assert schema["selected_roster_sha256"] == fb.digest(list(run.target_ids()))
    assert schema["parent_rows"] == len(run.cache.ids) and schema["selected_rows"] == 32
    assert cache.extractor == run.cache.extractor and cache.space.supervision.status == "label_free"
    rows = [run.ctx.table.position[cid] for cid in run.target_ids()]
    pooled = fne.pool_token_features(run.cache, fne.POOL_ALL)[rows]
    np.testing.assert_allclose(cache.features, pooled, atol=1e-6, rtol=1e-6)


def test_adapted_caches_bind_native_runtime_adapter_tape_and_config_truthfully(pipeline):
    run, ctx = pipeline["run"], pipeline["run"].ctx
    header = fb.read_json(run.adapters / "C_s0.adapter.json")
    cache = load_rep(pipeline, "C_s0")
    assert cache.extractor["kind"] == "external_features"
    assert cache.extractor["checkpoint_sha256"] == header["weights_file_sha256"]
    assert cache.extractor["source_sha256"] == ctx.code_sha256
    schema = cache.space.schema
    assert cache.space.kind == "functional_shared_adapter_function_query"
    assert schema["source_adapted"] is True and schema["label_free"] is False
    assert schema["parent"]["runtime_sha256"] == run.cache.runtime_sha256
    assert schema["parent"]["token_cache_sha256"] == run.cache.cache_sha256
    assert schema["parent"]["token_map_sha256"] == run.cache.token_map_sha256
    assert schema["parent"]["activation"] == {"kind": "final_hidden", "name": None}
    adapter = schema["adapter"]
    assert adapter["tape_sha256"] == ctx.seed_tapes[0].sha256
    assert adapter["training_config_sha256"] == ctx.training_sha256
    assert adapter["adapter_state_sha256"] == header["adapter_state_sha256"] != adapter[
        "artifact_file_sha256"]
    assert "NOT a checkpoint" in adapter["adapter_state_sha256_note"]
    assert schema["selected_roster"]["rows"] == 32
    rows = [ctx.table.position[cid] for cid in run.target_ids()]
    adapted = fst.load_adapter(run.adapters, "C_s0", ctx)
    np.testing.assert_allclose(cache.features, fst.extract_function_features(
        ctx.table, adapted.func_net, rows), atol=1e-5)


def test_supervision_histories_count_measured_labels_and_derived_answers_separately(pipeline):
    run, ctx = pipeline["run"], pipeline["run"].ctx
    entries = {rep: load_rep(pipeline, rep).space.supervision
               for rep in ("F_s0", "C_s0", "S_s0", "Q_s0", "R_s0")}
    for rep, history in entries.items():
        assert history.status == fb.HISTORY_DOCUMENTED, rep
        measured = history.documented[0]
        assert measured["label_count"] == len(ctx.functional_ids) \
            and measured["label_ids"] == list(ctx.functional_ids)
    assert len(entries["F_s0"].documented) == 1 and len(entries["S_s0"].documented) == 1
    computational = {rep: entries[rep].documented[1] for rep in ("C_s0", "Q_s0", "R_s0")}
    for entry in computational.values():
        assert entry["label_count"] == len(ctx.events) and len(set(entry["label_ids"])) == len(
            ctx.events)
        assert not set(entry["label_ids"]) & set(ctx.functional_ids)
    assert computational["C_s0"]["label_ids"] == computational["R_s0"]["label_ids"] \
        == computational["Q_s0"]["label_ids"] == list(ctx.answer_ids)
    assert "SHUFFLED" in computational["Q_s0"]["assay"]
    parent = run.cache.space.supervision
    assert parent.status == fb.HISTORY_LABEL_FREE


def test_S_representation_is_exactly_F_and_so_are_its_target_cells(pipeline):
    for seed in (0, 1):
        np.testing.assert_array_equal(load_rep(pipeline, f"S_s{seed}").features,
                                      load_rep(pipeline, f"F_s{seed}").features)
    cells = {(c["representation"], c["label_budget"], c["acquisition_seed"]): c
             for c in pipeline["evaluate"]["descriptive_all_cells"]}
    for seed in (0, 1):
        for budget in (4, 8):
            assert cells[(f"S_s{seed}", budget, seed)]["mse"] == cells[(f"F_s{seed}", budget, seed)]["mse"]


def test_the_analysis_has_diagonal_paired_differences_and_descriptive_off_diagonal_cells(pipeline):
    analysis = pipeline["evaluate"]
    assert analysis["diagonal_seeds"] == [0, 1] and len(analysis["target_evaluation_groups"]) == 4
    cells = {(c["representation"], c["label_budget"], c["acquisition_seed"]): c
             for c in analysis["descriptive_all_cells"]}
    assert len(cells) == 44
    comparison = analysis["comparisons"]["C-minus-F|linear|b4"]
    assert comparison["n_seeds"] == 2 and comparison["n_groups"] == 4
    for item in comparison["per_seed"]:
        seed = item["seed"]
        assert item["mse_difference"] == pytest.approx(
            cells[(f"C_s{seed}", 4, seed)]["mse"] - cells[(f"F_s{seed}", 4, seed)]["mse"])
        for group, value in item["per_group"].items():
            assert value == pytest.approx(cells[(f"C_s{seed}", 4, seed)]["per_group_mse"][group]
                                          - cells[(f"F_s{seed}", 4, seed)]["per_group_mse"][group])
    assert set(comparison["per_group_mean_difference"]) == set(analysis["target_evaluation_groups"])
    for key in ("C-minus-Q", "C-minus-R", "C-minus-native", "F-minus-native", "S-minus-F",
                "Q-minus-F", "R-minus-F"):
        assert f"{key}|linear|b8" in analysis["comparisons"]
    off = [c for c in analysis["descriptive_all_cells"]
           if c["representation"] != "native" and not c["diagonal"]]
    on = [c for c in analysis["descriptive_all_cells"]
          if c["representation"] != "native" and c["diagonal"]]
    assert off and on and len(off) == len(on)
    assert "not independent" in analysis["off_diagonal_note"]
    assert "p_value" not in json.dumps(analysis)
    assert set(analysis["learning_curves"]) >= {"native|linear", "F|linear", "C|linear"}


def test_source_preservation_compares_source_dev_error_with_F(pipeline):
    report = pipeline["report"]
    preservation = report["source_preservation"]["arms"]
    for seed in (0, 1):
        dev = {arm: report["runs"][f"{arm}_s{seed}"]["source_report"]["source_dev"]["mse"]
               for arm in fst.ARMS}
        for arm in fst.ARMS:
            entry = preservation[arm]["per_seed"][str(seed)]
            assert entry["source_dev_mse"] == dev[arm]
            assert entry["difference_from_F"] == pytest.approx(dev[arm] - dev["F"])
        assert preservation["F"]["per_seed"][str(seed)]["difference_from_F"] == 0.0
        assert preservation["S"]["per_seed"][str(seed)]["difference_from_F"] == 0.0
    assert report["runs"]["F_s0"]["source_report"]["units_of_mse"] == "(arbitrary units (synthetic))^2"
    assert "integrity check only" in report["integrity"]["note"]


def test_frozen_adapters_are_unchanged_by_target_fitting_and_evaluation(pipeline):
    report, run = pipeline["report"], pipeline["run"]
    assert all(report["integrity"]["adapters_unchanged_after_target_fitting"].values())
    for run_id, before in pipeline["adapter_files_at_freeze"].items():
        assert fst.file_sha256(run.adapters / f"{run_id}.adapter.pt") == before


def test_the_report_keeps_interpretation_rules_units_and_executed_status(pipeline):
    report = pipeline["report"]
    assert report["executed"] is True and report["schema"] == fst.REPORT_SCHEMA
    rules = " ".join(report["claims"]["interpretation"])
    for phrase in ("auxiliary accuracy alone", "beating the shuffled-target arm",
                   "matching the direct-supervision arm", "continued functional-only",
                   "negative transfer and inconclusive results are retained"):
        assert phrase in rules
    assert "no cross-assay z-scoring" in report["claims"]["units"].lower()
    losses = report["inputs"]["loss_definitions"]
    assert "(arbitrary units (synthetic))^2" in losses["functional"] and "nats" in losses["auxiliary"]
    assert report["inputs"]["target_outcomes"].startswith("not opened by prepare, train or features")
    for run_id in ("C_s0", "S_s1"):
        assert report["training_curves"][run_id]["functional_loss"]


def test_auxiliary_panels_cross_sequence_and_dependency_and_report_operator_controls(pipeline):
    panels = pipeline["panels"]
    assert set(panels["panels"]) == {
        f"{s}__{d}" for s in ("seen_sequences", "new_sequences", "target_sequences")
        for d in ("seen_dependency", "held_out_dependency")}
    assert panels["unsupported_panels"] == {}
    held = panels["panels"]["new_sequences__held_out_dependency"]
    assert held["coverage"]["only_chained_swap_programs"]
    by_run = held["by_run"]
    assert set(by_run) == {f"{a}_s{s}" for a in ("C", "S", "Q", "R") for s in (0, 1)}
    assert set(by_run["R_s0"]) == {"composed", "direct"} and set(by_run["C_s0"]) == {"composed"}
    controls = by_run["C_s0"]["composed"]["operator_controls"]
    assert set(controls) == {"delete_first", "delete_second", "reversed"}
    assert all(item["events"] == held["events"] for item in controls.values())
    metrics = by_run["C_s0"]["composed"]
    assert {"residue", "equality"} == set(metrics["by_output_type"])
    assert "eq_primary" in metrics["by_stratum"] and "res_original" in metrics["by_stratum"]
    for group in metrics["by_output_type"].values():
        assert 0.0 <= group["accuracy"] <= 1.0 and 0.0 < group["majority_rate"] <= 1.0
    assert panels["panels"]["seen_sequences__seen_dependency"]["events"] > 0
    for panel in panels["panels"].values():
        # the notes are unsupported-control / fallback NOTES, not skipped instances: the panel still holds
        # its events, and the old (false) name is gone from the report
        assert "skipped_instances" not in panel
        assert panel["events"] > 0 and panel["unsupported_control_notes"] >= 0
        assert panel["skipped_instance_notes"] <= panel["unsupported_control_notes"]
        assert sum(panel["unsupported_control_notes_by_reason"].values()) == panel[
            "unsupported_control_notes"]
    note = panels["note"]
    assert "not a chance rate" in note and "best-constant" in note
    assert "Chance for each group" not in note and "unsupported_control_notes" in note
    training = set(pipeline["run"].ctx.events.ids())
    fresh = pipeline["run"].build_panels(include_target=False)[0]["seen_sequences__seen_dependency"]
    assert not training & set(fresh.table.ids())


def test_panel_note_summary_separates_fallback_notes_from_skipped_instances():
    skipped = "no complete instance could be drawn; the instance is skipped"
    fallback = "no secondary EQ pair; used the primary stratum"
    notes = ({"candidate": "a", "program": "p", "template": "t", "goal": "g", "reason": fallback},
             {"candidate": "b", "program": "p", "template": "t", "goal": "g", "reason": fallback},
             {"candidate": "c", "layout": "H4L3", "universe": "train", "reason": skipped,
              "rejections": {"x": 3}})
    assert fst.panel_note_summary(notes) == {
        "unsupported_control_notes": 3, "skipped_instance_notes": 1,
        "unsupported_control_notes_by_reason": {fallback: 2, skipped: 1}}
    assert fst.panel_note_summary(()) == {
        "unsupported_control_notes": 0, "skipped_instance_notes": 0,
        "unsupported_control_notes_by_reason": {}}


# ---------------------------------------------------------------------------
# target labels cannot reach adapters or tapes; interruption does not change them
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_b(tmp_path_factory, pipeline):
    base = tmp_path_factory.mktemp("pipeline_b")
    paths = synth.write_inputs(pipeline["world"], base / "inputs")
    for name in ("target_train_outcomes", "target_eval_outcomes"):
        path = Path(paths[name])
        rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()[1:]]
        path.write_text("candidate_id,utility\n" + "".join(
            f"{cid},{(1.0 - float(value))!r}\n" for cid, value in rows), encoding="utf-8")
    run = fst.TransferRun.open(fst._inputs_from(paths), base / "run", pipeline["config"])
    run.prepare()
    assert run.train(stop_after=2) is False
    assert not run._complete("train")
    assert run.train() is True
    run.features()
    return {"run": run, "paths": paths}


def test_mutating_target_labels_changes_no_tape_adapter_or_target_feature(pipeline, pipeline_b):
    a, b = pipeline["run"], pipeline_b["run"]
    for seed in (0, 1):
        assert a.ctx.seed_tapes[seed].sha256 == b.ctx.seed_tapes[seed].sha256
    for name in ["event_table.json", "seed_0.json", "seed_1.json"]:
        assert fb.digest(fb.read_json(a.tapes / name)) == fb.digest(fb.read_json(b.tapes / name))
    for run_id in a.run_ids():
        left = fb.read_json(a.adapters / f"{run_id}.adapter.json")
        right = fb.read_json(b.adapters / f"{run_id}.adapter.json")
        assert left["adapter_state_sha256"] == right["adapter_state_sha256"], run_id
    for rep in ["native"] + a.run_ids():
        record_a = a.state["stages"]["features"]["caches"][rep]
        record_b = b.state["stages"]["features"]["caches"][rep]
        np.testing.assert_array_equal(
            fm.load_feature_cache(record_a["header"], npz_path=record_a["npz"]).features,
            fm.load_feature_cache(record_b["header"], npz_path=record_b["npz"]).features)
    assert fst.file_sha256(pipeline_b["paths"]["target_train_outcomes"]) != fst.file_sha256(
        pipeline["paths"]["target_train_outcomes"])


def test_pipeline_level_interruption_and_resume_reproduce_the_same_adapters(pipeline, pipeline_b):
    assert pipeline_b["run"]._complete("train")
    a, b = pipeline["run"], pipeline_b["run"]
    for run_id in a.run_ids():
        assert fb.read_json(a.runs / run_id / fst.CHECKPOINT_NAME)["state_sha256"] \
            == fb.read_json(b.runs / run_id / fst.CHECKPOINT_NAME)["state_sha256"]


# ---------------------------------------------------------------------------
# stage order, idempotence and refusal of partial or foreign state
# ---------------------------------------------------------------------------

def test_stages_refuse_to_run_out_of_order(opened):
    for stage in ("train", "features", "freeze", "evaluate", "panels", "report"):
        with pytest.raises(ValueError, match="is not complete"):
            getattr(opened, stage)()
    with pytest.raises(ValueError, match="unknown stage"):
        opened.run(["nonsense"])


def test_a_changed_tape_event_table_or_mask_binding_is_refused_before_any_training(
        inputs, tmp_path):
    config = small_config()
    cases = (
        ("seed_0.json", lambda d: d["mask_binding"]["token_lengths"].__setitem__(
            next(iter(d["mask_binding"]["token_lengths"])), 1)),
        ("seed_0.json", lambda d: d["aux_batches"][0].reverse()),
        ("event_table.json", lambda d: d["events"][0].__setitem__("answer", "W")),
        ("seed_1.json", lambda d: d["functional_labels"].__setitem__(0, 9.0)))
    for number, (name, edit) in enumerate(cases):
        out = tmp_path / f"case{number}"
        run = fst.TransferRun.open(inputs, out, config)
        run.prepare()
        path = out / "tapes" / name
        document = fb.read_json(path)
        edit(document)
        fb.write_json(path, document)
        with pytest.raises(ValueError, match="differs from what the inputs imply"):
            run.train()
        assert not (out / "runs").exists() and not (out / "adapters").exists()


def test_a_directory_of_different_inputs_or_configuration_is_never_reused(inputs, tmp_path):
    out = tmp_path / "run"
    run = fst.TransferRun.open(inputs, out, small_config())
    run.prepare()
    with pytest.raises(ValueError, match="DIFFERENT inputs or configuration"):
        fst.TransferRun.open(inputs, out, small_config(lam=0.2))
    assert fst.TransferRun.open(inputs, out, small_config(checkpoint_every=1)).state["stages"][
        "prepare"]["complete"]


def copy_pipeline(pipeline, tmp_path, name="copy"):
    destination = tmp_path / name
    shutil.copytree(pipeline["run"].root, destination)
    return fst.TransferRun.open(pipeline["inputs"], destination, pipeline["config"]), destination


def drop_stages(destination, *names):
    state = fb.read_json(destination / "stage_state.json")
    for name in names:
        state["stages"].pop(name, None)
    fb.write_json(destination / "stage_state.json", state)


def test_a_complete_verified_freeze_is_adopted_not_refitted(pipeline, tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    drop_stages(destination, "freeze", "evaluate", "panels", "report")
    reopened = fst.TransferRun.open(pipeline["inputs"], destination, pipeline["config"])
    cells = sorted((destination / "matrix_freeze" / "cells").glob("*.json"))
    before = {path.name: fst.file_sha256(path) for path in cells}
    state = reopened.freeze()
    assert state["adopted_existing_freeze"] is True and state["complete"]
    assert {path.name: fst.file_sha256(path) for path in cells} == before
    assert state["index_sha256"] == pipeline["run"].state["stages"]["freeze"]["index_sha256"]


def test_a_partial_or_damaged_freeze_is_refused_and_never_overwritten(pipeline, tmp_path):
    run, destination = copy_pipeline(pipeline, tmp_path)
    cell = sorted((destination / "matrix_freeze" / "cells").glob("*.json"))[0]
    data = bytearray(cell.read_bytes())
    data[len(data) // 2] ^= 0x01
    cell.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        run.freeze()                                    # complete in the state, but not on disk
    drop_stages(destination, "freeze", "evaluate", "panels", "report")
    reopened = fst.TransferRun.open(pipeline["inputs"], destination, pipeline["config"])
    with pytest.raises(ValueError):
        reopened.freeze()                               # a damaged freeze is not adopted either
    assert cell.read_bytes() == bytes(data)


def test_an_interrupted_features_stage_is_refused_not_adopted(pipeline, tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    drop_stages(destination, "features", "freeze", "evaluate", "panels", "report")
    reopened = fst.TransferRun.open(pipeline["inputs"], destination, pipeline["config"])
    with pytest.raises(ValueError, match="interrupted features stage"):
        reopened.features()


def test_an_adapter_the_state_does_not_record_is_never_adopted(pipeline, tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    state = fb.read_json(destination / "stage_state.json")
    del state["stages"]["train"]["runs"]["F_s0"]
    state["stages"]["train"]["complete"] = False
    fb.write_json(destination / "stage_state.json", state)
    reopened = fst.TransferRun.open(pipeline["inputs"], destination, pipeline["config"])
    with pytest.raises(ValueError, match="never adopted"):
        reopened.train()


def test_a_damaged_adapter_is_refused_by_every_later_stage(pipeline, tmp_path):
    run, destination = copy_pipeline(pipeline, tmp_path)
    weights = destination / "adapters" / "C_s1.adapter.pt"
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0x01
    weights.write_bytes(bytes(data))
    drop_stages(destination, "panels", "report")
    reopened = fst.TransferRun.open(pipeline["inputs"], destination, pipeline["config"])
    with pytest.raises(ValueError, match="SHA-256"):
        reopened.panels()


def test_the_evaluation_file_must_name_exactly_the_target_evaluation_cohort(pipeline, tmp_path):
    run, destination = copy_pipeline(pipeline, tmp_path)
    drop_stages(destination, "evaluate", "panels", "report")
    bad = tmp_path / "bad_eval.csv"
    lines = Path(pipeline["paths"]["target_eval_outcomes"]).read_text(encoding="utf-8").splitlines()
    bad.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    inputs = dataclasses.replace(pipeline["inputs"], target_eval_outcomes=str(bad))
    reopened = fst.TransferRun.open(inputs, destination, pipeline["config"])
    with pytest.raises(ValueError, match="lacks"):
        reopened.evaluate()


# ---------------------------------------------------------------------------
# small analysis helpers on fabricated numbers
# ---------------------------------------------------------------------------

def test_source_preservation_is_a_paired_dev_error_difference_against_F():
    config = fst.TransferConfig(arms=("F", "C"), source_seeds=(0, 1), acquisition_seeds=(0,),
                                label_budgets=(4,), steps=4, probe_steps=())

    def report(dev, train):
        return {"source_report": {"source_dev": {"mse": dev, "constant_source_train_mean_mse": 2.0},
                                  "source_train": {"mse": train}}}
    runs = {"F_s0": report(1.0, 0.5), "C_s0": report(1.5, 0.4), "F_s1": report(1.0, 0.5),
            "C_s1": report(0.75, 0.4)}
    arms = fst.source_preservation(config, runs)["arms"]
    assert arms["C"]["per_seed"]["0"]["difference_from_F"] == pytest.approx(0.5)
    assert arms["C"]["per_seed"]["1"]["difference_from_F"] == pytest.approx(-0.25)
    assert arms["C"]["seeds_worse_than_F"] == 1 and arms["C"]["mean_difference_from_F"] == pytest.approx(0.125)
    assert arms["F"]["mean_difference_from_F"] == 0.0


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

def load_cli():
    spec = importlib.util.spec_from_file_location("functional_shared_transfer_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_inspect_validates_the_data_boundary(world_dir, capsys, tmp_path):
    _, paths = world_dir
    cli = load_cli()
    assert cli.main(["inspect", "--token-header", paths["token_header"], "--cohorts",
                     paths["cohorts"]]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    document = json.loads(Path(paths["cohorts"]).read_text(encoding="utf-8"))
    document["cohorts"]["target_eval"] = document["cohorts"]["target_eval"][1:]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(document), encoding="utf-8")
    assert cli.main(["inspect", "--token-header", paths["token_header"], "--cohorts", str(bad)]) == 2
    assert "outside the declared cohort union" in capsys.readouterr().err


def test_cli_synthetic_writes_a_real_cache_and_can_withhold_the_evaluation_outcomes(
        capsys, tmp_path):
    cli = load_cli()
    target = tmp_path / "synthetic"
    assert cli.main(["synthetic", "--output-dir", str(target), "--size", "selftest",
                     "--withhold-eval"]) == 0
    written = json.loads(capsys.readouterr().out)
    assert (target / "tokens.json").exists() and (target / "held" / "target_eval_outcomes.csv").exists()
    assert not (target / "target_eval_outcomes.csv").exists()
    assert written["counts"]["source_aux"] == 32 and "computational" in written["note"]
    assert cli.main(["inspect", "--token-header", str(target / "tokens.json"), "--cohorts",
                     str(target / "cohorts.json")]) == 0


def test_cli_run_executes_the_requested_stages_only(world_dir, capsys, tmp_path):
    _, paths = world_dir
    cli = load_cli()
    config = tmp_path / "config.json"
    fb.write_json(config, small_config().document())
    out = tmp_path / "run"
    argv = ["run", "--token-header", paths["token_header"], "--cohorts", paths["cohorts"],
            "--source-train-outcomes", paths["source_train_outcomes"],
            "--source-dev-outcomes", paths["source_dev_outcomes"],
            "--target-train-outcomes", paths["target_train_outcomes"],
            "--target-eval-outcomes", paths["target_eval_outcomes"],
            "--output-dir", str(out), "--config", str(config), "--stages", "prepare",
            "--single-thread", "--steps", "3"]
    assert cli.main(argv) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["stages_run"] == ["prepare"] and document["report"] is None
    state = json.loads((out / "stage_state.json").read_text(encoding="utf-8"))
    assert list(state["stages"]) == ["prepare"] and (out / "tapes" / "event_table.json").exists()
    with pytest.raises(SystemExit):
        cli.main(["nonsense" if item == "prepare" else item for item in argv])


def test_cli_surfaces_a_refusal_as_exit_code_two(world_dir, capsys, tmp_path):
    _, paths = world_dir
    cli = load_cli()
    argv = ["run", "--token-header", paths["token_header"], "--cohorts", paths["cohorts"],
            "--source-train-outcomes", paths["source_dev_outcomes"],       # the wrong cohort
            "--source-dev-outcomes", paths["source_dev_outcomes"],
            "--target-train-outcomes", paths["target_train_outcomes"],
            "--target-eval-outcomes", paths["target_eval_outcomes"],
            "--output-dir", str(tmp_path / "run"), "--stages", "prepare"]
    assert cli.main(argv) == 2
    assert "source training outcomes" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the selftest itself
# ---------------------------------------------------------------------------

def test_the_selftest_runs_every_named_check_and_passes(tmp_path):
    report = fst.selftest(tmp_path)
    failed = [check for check in report["checks"] if not check["holds"]]
    assert report["passed"] and not failed, failed
    names = {check["name"] for check in report["checks"]}
    for needed in ("parity::S_functional_pathway_and_optimizer_equal_F_after_every_macrostep",
                   "resume::interrupted_and_resumed_run_equals_the_uninterrupted_run_exactly",
                   "wiring::auxiliary_only_step_moves_C_function_features_but_not_S",
                   "matrix::all_representations_frozen_before_any_target_evaluation_outcome_existed",
                   "integrity::frozen_adapters_unchanged_after_target_fitting"):
        assert needed in names
    assert Path(report["pipeline_report"]).exists()
    assert (Path(report["work_directory"]) / fst.SELFTEST_REPORT_NAME).exists()
    assert "never a scientific win" in report["note"]


# ===========================================================================
# repair 15b
# ===========================================================================

# ---------------------------------------------------------------------------
# R1: valid Adam resume, atomic loading, one deterministic objective
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def head_use(world_dir, opened):
    """One auxiliary row per macrostep, so each step is RES or EQ and the residue/equality heads
    legitimately take FEWER Adam steps than the encoder."""
    config = small_config(steps=12, batch=4, aux_batch=1, checkpoint_every=4, probe_steps=(0, 6),
                          source_seeds=(0,), acquisition_seeds=(0,), label_budgets=(4,))
    return fst.build_context(opened.cache, opened.cohorts, config, world_dir[0].outcomes["source"])


def mixed_cut(context, limit=12):
    """The first cursor at which both answer types have occurred (so both heads lag the encoder)."""
    tape = context.seed_tapes[0]
    for cut in range(2, limit):
        kinds = [bool(context.event_is_equality[int(row[0])]) for row in tape.aux_batches[:cut]]
        if 0 < sum(kinds) < cut:
            return cut
    raise AssertionError("the first auxiliary steps of this tape are all one answer type")


def full_state(run):
    return (run.state_sha256(), run.cursor, copy.deepcopy(run.accounting), copy.deepcopy(run.log),
            copy.deepcopy(run.diagnostics), run.cpu_seconds, run.probe_seconds)


def reseal_checkpoint(directory, mutate):
    """Rewrite the weights with ``mutate(payload)`` and reseal every hash: only the CONTENT is bad."""
    header = json.loads((directory / fst.CHECKPOINT_NAME).read_text(encoding="utf-8"))
    weights = directory / header["weights_file"]
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, weights)
    reseal(directory / fst.CHECKPOINT_NAME, weights_file_sha256=fst.file_sha256(weights),
           state_sha256=fst.tensor_state_digest(payload))


@pytest.mark.parametrize("arm", ["F", "C", "S", "Q", "R"])
def test_per_parameter_adam_steps_equal_the_counts_the_frozen_tape_implies(head_use, arm):
    run, cut = fst.ArmRun(head_use, arm, 0), mixed_cut(head_use)
    while run.cursor < cut:
        run.macrostep()
    expected = run.expected_adam_steps()
    for key, net, optimizer in (("func", run.func_net, run.func_opt), ("aux", run.aux_net, run.aux_opt)):
        if optimizer is None:
            continue
        for name, parameter in net.named_parameters():
            steps = expected[key][name]
            if steps == 0:
                assert parameter not in optimizer.state, (key, name)   # no zero gradient is invented
            else:
                assert int(optimizer.state[parameter]["step"]) == steps, (key, name)
    if arm in fst.AUX_ARMS:
        counts = expected["aux" if arm == "S" else "func"]
        assert counts["res_head.weight"] < run.cursor and counts["eq_head.weight"] < run.cursor
        assert counts["res_head.weight"] + counts["eq_head.weight"] == run.cursor    # one row per step
        assert counts["proj.weight"] == run.cursor
    else:
        assert expected["func"]["res_head.weight"] == 0 == expected["func"]["eq_head.weight"]
    if arm == "S":
        assert expected["aux"]["func_head.weight"] == 0 and expected["func"]["func_head.weight"] == cut


@pytest.mark.parametrize("arm", ["F", "C", "S", "Q", "R"])
def test_resume_with_mixed_head_use_equals_the_uninterrupted_run_for_every_arm(head_use, tmp_path,
                                                                             arm):
    cut = mixed_cut(head_use)
    straight = fst.train_arm(head_use, arm, 0, tmp_path / "straight")
    interrupted = fst.train_arm(head_use, arm, 0, tmp_path / "resumed", stop_after=cut)
    assert interrupted.cursor == cut
    if arm in fst.AUX_ARMS:        # the legitimate checkpoint that the old step==cursor rule refused
        steps = adam_steps(interrupted.aux_opt if arm == "S" else interrupted.func_opt)
        assert len(steps) > 1
    fresh = fst.ArmRun(head_use, arm, 0)
    assert fst.restore_checkpoint(fresh, tmp_path / "resumed")
    assert fresh.state_sha256() == interrupted.state_sha256() and fresh.cursor == cut
    resumed = fst.train_arm(head_use, arm, 0, tmp_path / "resumed")
    assert resumed.cursor == resumed.total_steps
    assert resumed.deterministic_result() == straight.deterministic_result()
    assert resumed.state_sha256() == straight.state_sha256()


def test_a_consistently_hashed_malformed_tensor_is_refused_with_no_live_state_changed(head_use,
                                                                                     tmp_path):
    run = fst.ArmRun(head_use, "C", 0)
    for _ in range(5):
        run.macrostep()
    fst.save_checkpoint(run, tmp_path / "ck")

    def shrink(payload):
        weight = payload["nets"]["func"]["eq_head.weight"]
        payload["nets"]["func"]["eq_head.weight"] = weight[:, :-1].clone()
    reseal_checkpoint(tmp_path / "ck", shrink)
    fresh = fst.ArmRun(head_use, "C", 0)
    before = full_state(fresh)
    with pytest.raises(ValueError, match="eq_head.weight"):
        fst.restore_checkpoint(fresh, tmp_path / "ck")
    assert full_state(fresh) == before          # nothing, not even a partial load, reached the run


OPTIMIZER_CASES = {"moment_shape": "shape and dtype", "step": "Adam step", "nan": "not finite",
                   "negative": "negative", "extra_entry": "wrong parameters",
                   "missing_entry": "wrong parameters", "param_group": "parameter groups"}


@pytest.mark.parametrize("case", list(OPTIMIZER_CASES))
def test_malformed_optimizer_data_is_refused_with_no_live_state_changed(head_use, tmp_path, case):
    run = fst.ArmRun(head_use, "F" if case == "extra_entry" else "C", 0)
    for _ in range(4):
        run.macrostep()
    fst.save_checkpoint(run, tmp_path / "ck")
    index = {name: position for position, (name, _) in enumerate(run.func_net.named_parameters())}
    shapes = {name: tuple(parameter.shape) for name, parameter in run.func_net.named_parameters()}

    def edit(payload):
        state = payload["optimizers"]["func"]
        entry = None if case == "extra_entry" else state["state"][index["proj.weight"]]
        if case == "moment_shape":
            entry["exp_avg"] = torch.zeros(1)
        elif case == "step":
            entry["step"] = torch.tensor(99.0)
        elif case == "nan":
            entry["exp_avg_sq"] = torch.full(shapes["proj.weight"], float("nan"))
        elif case == "negative":
            entry["exp_avg_sq"] = torch.full(shapes["proj.weight"], -1.0)
        elif case == "extra_entry":                 # F never updates the residue head
            zeros = torch.zeros(shapes["res_head.weight"])
            state["state"][index["res_head.weight"]] = {"step": torch.tensor(4.0), "exp_avg": zeros,
                                                        "exp_avg_sq": zeros.clone()}
        elif case == "missing_entry":
            del state["state"][index["proj.weight"]]
        else:
            state["param_groups"][0]["lr"] = 0.5
    reseal_checkpoint(tmp_path / "ck", edit)
    fresh = fst.ArmRun(head_use, "F" if case == "extra_entry" else "C", 0)
    before = full_state(fresh)
    with pytest.raises(ValueError, match=OPTIMIZER_CASES[case]):
        fst.restore_checkpoint(fresh, tmp_path / "ck")
    assert full_state(fresh) == before


def test_nonzero_dropout_is_refused_at_the_configuration_boundary():
    with pytest.raises(ValueError, match="dropout"):
        fst.TransferConfig(dropout=0.3)
    with pytest.raises(ValueError, match="dropout"):
        fst.TransferConfig.from_document({"dropout": 0.3})
    assert fst.TransferConfig(dropout=0.0).dropout == 0.0


def test_the_training_step_and_the_diagnostic_probe_report_the_same_objective(ctx):
    run = fst.ArmRun(ctx, "C", 0)
    entry = run.macrostep()
    diagnostic = run.diagnostics[0]
    assert diagnostic["step"] == 0
    assert entry["functional_loss"] == pytest.approx(diagnostic["functional_loss"], abs=1e-6)
    assert entry["auxiliary_loss"] == pytest.approx(diagnostic["auxiliary_loss"], abs=1e-6)


def test_a_macrostep_runs_in_evaluation_mode_and_restores_mixed_prior_module_modes(ctx):
    run = fst.ArmRun(ctx, "C", 0)
    run.func_net.train()
    run.func_net.blocks[0].eval()
    flags = [module.training for module in run.func_net.modules()]
    seen, real = [], fst.StudentNet.encode

    def spy(self, batch):
        seen.append(self.training)
        return real(self, batch)
    with mock.patch.object(fst.StudentNet, "encode", spy):
        run.macrostep()
    assert seen and not any(seen)                  # every forward, step and probe alike, was eval mode
    assert [module.training for module in run.func_net.modules()] == flags


# ---------------------------------------------------------------------------
# R2: variable source and auxiliary sequence lengths on a real tiny-native cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mixed_source(world_dir, tmp_path_factory):
    """The synthetic world's real tiny-DPLM cache with 12 of the 32 auxiliary rows made H5/L4 (the
    other 20 stay H4/L4); every cohort, outcome and group is unchanged."""
    world, paths = world_dir
    odd = set(sorted(world.roles["source_aux"])[::2][:12])
    document = world.export_plan_document("mixed_source_layouts")
    for row in document["rows"]:
        if row["candidate_id"] in odd:
            row["heavy"] += "A"
    directory = tmp_path_factory.mktemp("mixed_source")
    fne.export_dplm_tokens(fne.tiny_model(0), fne.parse_export_plan(document),
                           directory / "tokens.json", directory / "tokens.npz", batch_size=16)
    inputs = dataclasses.replace(fst._inputs_from(paths), token_header=str(directory / "tokens.json"),
                                 token_npz=str(directory / "tokens.npz"))
    return {"odd": odd, "inputs": inputs}


def test_a_real_mixed_source_cache_runs_prepare_train_features_and_panels(mixed_source, tmp_path):
    config = small_config(steps=2, batch=4, aux_batch=4, arms=("F", "C"), source_seeds=(0,),
                          acquisition_seeds=(0,), label_budgets=(4,), checkpoint_every=1,
                          probe_steps=(0,), panel_sequences=32)
    run = fst.TransferRun.open(mixed_source["inputs"], tmp_path / "run", config)
    ctx = run.ctx
    assert ctx.layout is None and isinstance(ctx.layouts, fc.LayoutMap)
    assert ctx.space == fc.CoordSpace(5, 4) and (ctx.table.max_heavy, ctx.table.max_light) == (5, 4)
    assert ctx.events.config["layout_shapes"] == {"H4L4": 20, "H5L4": 12}
    assert ctx.coverage["complete"] and set(ctx.coverage["by_layout"]) == {"H4L4", "H5L4"}
    assert all(entry["complete"] for entry in ctx.coverage["by_layout"].values())
    rows = {cid: (h, l) for cid, h, l in run.cohorts.rows(run.cache, "source_aux")}
    for position, event in enumerate(ctx.events.events):
        heavy, light = rows[event.candidate_id]
        coords = [c for op in event.program.ops for c in op.coords()] + list(event.query.coords())
        assert all(c.index < (len(heavy) if c.chain == "H" else len(light)) for c in coords)
        length = int(ctx.event_composed_len[position])
        ids = {int(v) for row in ctx.event_composed[position][:length] for v in row[1:3] if v >= 0}
        # light ids sit after the SPACE's five heavy ids for every row, never after the row's own 4
        assert {("H", v) if v < 5 else ("L", v - 5) for v in ids} == {(c.chain, c.index) for c in coords}
    # the complete candidate -> layout map is part of the event table, the tape and the inputs identity
    assert len(ctx.events.document()["layout"]["candidates"]) == len(rows)
    assert ctx.seed_tapes[0].mask_binding["layout_sha256"] == fc.layouts_sha256(ctx.layouts)
    run.prepare()
    assert run.train() is True
    assert set(run.features()) == {"native", "F_s0", "C_s0"}
    panels, unsupported = run.build_panels(include_target=True)
    assert unsupported == {} and len(panels) == 6            # no variable-layout row is silently dropped
    assert set(panels["seen_sequences__held_out_dependency"].table.config["layout_shapes"]) \
        == {"H4L4", "H5L4"}


def test_row_bounds_and_padding_do_not_change_features_on_a_real_mixed_source_cache(mixed_source,
                                                                                     tmp_path):
    run = fst.TransferRun.open(mixed_source["inputs"], tmp_path / "run",
                               small_config(steps=1, source_seeds=(0,), acquisition_seeds=(0,),
                                            label_budgets=(4,), probe_steps=()))
    table = run.ctx.table
    odd = sorted(mixed_source["odd"])[:2]
    plain = [cid for cid in run.cohorts.roles["source_aux"] if cid not in mixed_source["odd"]][:2]
    rows = [table.position[cid] for cid in odd + plain]
    assert len({table.length(row) for row in rows}) > 1               # genuinely ragged
    net = fst.build_student(run.ctx.spec, seed=5)
    base = fst.extract_function_features(table, net, rows, batch_size=4)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(
        fst.extract_function_features(table, net, [rows[i] for i in order], batch_size=1),
        base[order], atol=2e-5)
    np.testing.assert_allclose(
        fst.extract_function_features(table, net, rows, token_pad=table.max_tokens + 7), base,
        atol=2e-5)


def test_a_sparse_layout_shape_is_refused_with_an_actionable_message(world_dir, tmp_path):
    world, paths = world_dir
    odd = sorted(world.roles["source_aux"])[0]
    document = world.export_plan_document("sparse_shape")
    for row in document["rows"]:
        if row["candidate_id"] == odd:
            row["heavy"] += "A"
    fne.export_dplm_tokens(fne.tiny_model(0), fne.parse_export_plan(document),
                           tmp_path / "tokens.json", tmp_path / "tokens.npz", batch_size=16)
    inputs = dataclasses.replace(fst._inputs_from(paths), token_header=str(tmp_path / "tokens.json"),
                                 token_npz=str(tmp_path / "tokens.npz"))
    with pytest.raises(ValueError, match="H5L4") as raised:
        fst.TransferRun.open(inputs, tmp_path / "run", small_config())
    assert "does not cover every training requirement" in str(raised.value)
    assert "remove those rows from source_aux" in str(raised.value)


# ---------------------------------------------------------------------------
# R3: one question, one answer id
# ---------------------------------------------------------------------------

def test_answer_ids_derive_from_the_canonical_question_not_the_stored_id(ctx):
    event = ctx.events.events[0]
    alias = copy.deepcopy(event)
    object.__setattr__(alias, "event_id", "alias")
    assert fst.answer_label_id(alias) == fst.answer_label_id(event)
    assert len(ctx.answer_ids) == len(set(ctx.answer_ids)) == len(ctx.events)


# ---------------------------------------------------------------------------
# R5: the arrays a run consumes are the verified ones, and the cost report counts initialisation
# ---------------------------------------------------------------------------

def private_tape_context(ctx):
    tape = ctx.seed_tapes[0].snapshot()
    return dataclasses.replace(ctx, seed_tapes={**ctx.seed_tapes, 0: tape}), tape


@pytest.mark.parametrize("which,arm", [("functional_labels", "F"), ("q_targets", "Q")])
def test_a_tape_array_edited_before_a_run_exists_is_refused_before_any_model_is_built(ctx, which,
                                                                                       arm):
    private, tape = private_tape_context(ctx)
    array = getattr(tape, which)
    array.setflags(write=True)
    array[0] = array[0] + (10.0 if which == "functional_labels" else 1)
    with pytest.raises(ValueError, match="recorded digest"):
        fst.ArmRun(private, arm, 0)


def test_a_tape_array_edited_after_a_run_exists_cannot_reach_that_run(ctx):
    private, tape = private_tape_context(ctx)
    reference = run_to_end(fst.ArmRun(ctx, "F", 0)).state_sha256()
    run = fst.ArmRun(private, "F", 0)
    labels = tape.functional_labels
    labels.setflags(write=True)
    labels += 50.0
    labels.setflags(write=False)
    assert run_to_end(run).state_sha256() == reference          # it trained on its own verified copy
    with pytest.raises(ValueError, match="recorded digest"):
        fst.ArmRun(private, "F", 0)                               # a NEW run refuses the edited tape


def test_an_edited_answer_array_is_refused_when_a_run_is_created(ctx):
    private = dataclasses.replace(ctx, event_codes=np.array(ctx.event_codes, copy=True))
    private.event_codes.setflags(write=True)
    private.event_codes[0] = (int(private.event_codes[0]) + 1) % 2
    with pytest.raises(ValueError, match="no longer match"):
        fst.ArmRun(private, "C", 0)


def private_table_context(ctx):
    """A context over a private writable copy of the token features (identical content, so the
    digests recorded when the context was built still hold) and that copy."""
    table = dataclasses.replace(ctx.table, features=np.array(ctx.table.features, copy=True))
    return dataclasses.replace(ctx, table=table), table


def test_the_shared_token_table_is_re_verified_before_a_checkpoint_is_published(ctx, tmp_path):
    private, table = private_table_context(ctx)
    run = fst.ArmRun(private, "F", 0)
    run.macrostep()
    table.features[0, 0] += 1.0                   # the (large) shared table is edited mid-run
    with pytest.raises(ValueError, match="no longer match"):
        fst.save_checkpoint(run, tmp_path / "ck")
    assert not (tmp_path / "ck").exists()         # refused before anything was written


def test_the_shared_token_table_is_re_verified_before_an_adapter_is_published(ctx, tmp_path):
    private, table = private_table_context(ctx)
    run = run_to_end(fst.ArmRun(private, "F", 0))
    table.features[0, 0] += 1.0
    with pytest.raises(ValueError, match="no longer match"):
        fst.freeze_adapter(run, tmp_path / "adapters")
    assert not (tmp_path / "adapters").exists()
    with pytest.raises(ValueError, match="no longer match"):
        fst.ArmRun(private, "F", 0)               # and a new run is refused at creation


def test_initialisation_consumes_every_source_label_and_the_cost_report_counts_it(opened, ctx):
    run = fst.ArmRun(ctx, "F", 0)
    n = len(ctx.functional_ids)
    before = run.cost_report()
    assert before["cursor"] == 0 and before["unique_measured_outcomes_directly_presented"] == 0
    assert before["unique_measured_outcomes_consumed_by_initialization"] == n
    assert before["unique_measured_source_outcomes"] == n
    assert before["measured_outcome_consumption"]["union_total"]["count"] == n
    run.macrostep()
    after = run.cost_report()
    assert 0 < after["unique_measured_outcomes_directly_presented"] <= ctx.config.batch
    assert after["unique_measured_source_outcomes"] == n and "separately" in after["units"]
    # ... and an unpresented label really does change the model before any batch is shown
    first = set(ctx.seed_tapes[0].functional_batches[0].tolist())
    unseen = next(i for i in range(n) if i not in first)
    labels = dict(zip(ctx.functional_ids, ctx.functional_labels.tolist()))
    labels[ctx.functional_ids[unseen]] += 8.0 * n                 # moves the mean by exactly 8
    other = fst.build_context(opened.cache, ctx.cohorts, ctx.config, labels)
    batch, _ = fst.functional_batch(ctx, ctx.seed_tapes[0], 0)
    with torch.no_grad():
        left = fst.ArmRun(ctx, "F", 0).func_net.functional_forward(batch)
        right = fst.ArmRun(other, "F", 0).func_net.functional_forward(batch)
    assert torch.allclose(right - left, torch.full_like(left, 8.0), atol=1e-4)


# ---------------------------------------------------------------------------
# R6: one bound outcome version per stage
# ---------------------------------------------------------------------------

def altered_outcomes(source, destination, *, offset=None, reverse=False):
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    rows = [line.split(",") for line in lines[1:]]
    if reverse:
        rows = rows[::-1]
    Path(destination).write_text("candidate_id,utility\n" + "".join(
        f"{cid},{(float(value) if offset is None else float(value) + offset)!r}\n"
        for cid, value in rows), encoding="utf-8")
    return str(destination)


def file_digests(root):
    return {str(path.relative_to(root)): fst.file_sha256(path)
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def test_a_completed_freeze_refuses_changed_target_training_outcomes(pipeline, tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    changed = altered_outcomes(pipeline["paths"]["target_train_outcomes"], tmp_path / "train_plus.csv",
                               offset=100.0)
    inputs = dataclasses.replace(pipeline["inputs"], target_train_outcomes=changed)
    reopened = fst.TransferRun.open(inputs, destination, pipeline["config"])
    before = file_digests(destination)
    with pytest.raises(ValueError, match="not the ones the matrix freeze was fitted on"):
        reopened.freeze()
    assert file_digests(destination) == before             # a failed continuation changes nothing


def test_an_adopted_freeze_is_compared_with_its_own_record_never_stamped_with_new_labels(pipeline,
                                                                                         tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    drop_stages(destination, "freeze", "evaluate", "panels", "report")
    changed = altered_outcomes(pipeline["paths"]["target_train_outcomes"], tmp_path / "train_plus.csv",
                               offset=100.0)
    refused = fst.TransferRun.open(dataclasses.replace(pipeline["inputs"], target_train_outcomes=changed),
                                   destination, pipeline["config"])
    before = file_digests(destination)
    with pytest.raises(ValueError, match="not the ones the matrix freeze was fitted on"):
        refused.freeze()
    assert file_digests(destination) == before
    state = fb.read_json(destination / "stage_state.json")
    assert "freeze" not in state["stages"]
    assert state["outcomes"]["target_train"] == pipeline["run"].state["outcomes"]["target_train"]


def test_a_freeze_is_adopted_for_the_same_target_training_data_in_another_row_order(pipeline,
                                                                                    tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    drop_stages(destination, "freeze", "evaluate", "panels", "report")
    reordered = altered_outcomes(pipeline["paths"]["target_train_outcomes"],
                                 tmp_path / "train_reordered.csv", reverse=True)
    inputs = dataclasses.replace(pipeline["inputs"], target_train_outcomes=reordered)
    state = fst.TransferRun.open(inputs, destination, pipeline["config"]).freeze()
    original = pipeline["run"].state["stages"]["freeze"]
    assert state["adopted_existing_freeze"] is True and state["complete"]
    assert state["target_training_outcomes_sha256"] == original["target_training_outcomes_sha256"]
    assert state["index_sha256"] == original["index_sha256"]
    assert state["fitted_training_outcomes_csv_sha256"]    # the hash of what the cells were fitted on


@pytest.mark.parametrize("dropped", [(), ("evaluate", "panels", "report")])
def test_a_repeated_evaluation_never_pairs_the_old_report_with_analysis_of_new_labels(
        pipeline, tmp_path, dropped):
    _, destination = copy_pipeline(pipeline, tmp_path)
    if dropped:
        drop_stages(destination, *dropped)
    changed = altered_outcomes(pipeline["paths"]["target_eval_outcomes"], tmp_path / "eval_plus.csv",
                               offset=100.0)
    inputs = dataclasses.replace(pipeline["inputs"], target_eval_outcomes=changed)
    reopened = fst.TransferRun.open(inputs, destination, pipeline["config"])
    before = file_digests(destination)
    with pytest.raises(ValueError, match="different target-evaluation outcomes"):
        reopened.evaluate()
    assert file_digests(destination) == before    # report bytes AND analysis bytes are untouched


def test_re_running_evaluate_on_the_same_outcomes_is_idempotent_whatever_the_row_order(pipeline,
                                                                                        tmp_path):
    _, destination = copy_pipeline(pipeline, tmp_path)
    drop_stages(destination, "evaluate", "panels", "report")
    reordered = altered_outcomes(pipeline["paths"]["target_eval_outcomes"],
                                 tmp_path / "eval_reordered.csv", reverse=True)
    inputs = dataclasses.replace(pipeline["inputs"], target_eval_outcomes=reordered)
    analysis = fst.TransferRun.open(inputs, destination, pipeline["config"]).evaluate()
    assert fb.digest(analysis) == fb.digest(pipeline["evaluate"])
    before = file_digests(destination)
    again = fst.TransferRun.open(inputs, destination, pipeline["config"]).evaluate()
    assert fb.digest(again) == fb.digest(analysis) and file_digests(destination) == before


def test_interrupted_source_training_never_mixes_source_development_versions(world_dir, inputs,
                                                                             tmp_path, monkeypatch):
    config = small_config(arms=("F", "C"), source_seeds=(0,), acquisition_seeds=(0,), steps=2,
                          probe_steps=(0,), checkpoint_every=1)
    run = fst.TransferRun.open(inputs, tmp_path / "run", config)
    run.prepare()
    real = fst.train_arm

    def stop_before_c(ctx_, arm, seed, directory, **kwargs):
        if arm == "C":
            raise RuntimeError("owned interruption before C, after F was committed")
        return real(ctx_, arm, seed, directory, **kwargs)
    monkeypatch.setattr(fst, "train_arm", stop_before_c)
    with pytest.raises(RuntimeError, match="owned interruption"):
        run.train()
    monkeypatch.setattr(fst, "train_arm", real)
    frozen_f = fb.read_json(run.adapters / "F_s0.adapter.json")
    assert frozen_f["source_report"]["outcome_bindings"]["source_dev"] \
        == run.state["outcomes"]["source_dev"]
    dev = world_dir[1]["source_dev_outcomes"]
    changed = altered_outcomes(dev, tmp_path / "dev_plus.csv", offset=100.0)
    refused = fst.TransferRun.open(dataclasses.replace(inputs, source_dev_outcomes=changed),
                                   tmp_path / "run", config)
    with pytest.raises(ValueError, match="source_dev outcomes differ"):
        refused.train()
    assert fb.read_json(run.adapters / "F_s0.adapter.json") == frozen_f
    assert not (run.adapters / "C_s0.adapter.json").exists() and not (run.runs / "C_s0").exists()
    # the same data in another row order is the same version: the interrupted run resumes
    reordered = altered_outcomes(dev, tmp_path / "dev_reordered.csv", reverse=True)
    resumed = fst.TransferRun.open(dataclasses.replace(inputs, source_dev_outcomes=reordered),
                                   tmp_path / "run", config)
    assert resumed.train() is True
    bound = {run_id: fb.read_json(resumed.adapters / f"{run_id}.adapter.json")["source_report"][
        "outcome_bindings"]["source_dev"] for run_id in resumed.run_ids()}
    assert len({json.dumps(value, sort_keys=True) for value in bound.values()}) == 1
    assert fb.read_json(resumed.adapters / "F_s0.adapter.json") == frozen_f


# ---------------------------------------------------------------------------
# R7: exclusive owned staging and protected inputs
# ---------------------------------------------------------------------------

def test_a_source_input_named_like_the_event_table_staging_file_is_refused_not_destroyed(
        world_dir, inputs, tmp_path):
    out = tmp_path / "run"
    alias = out / "tapes" / "event_table.json.tmp"
    alias.parent.mkdir(parents=True)
    shutil.copyfile(world_dir[1]["source_train_outcomes"], alias)
    before = alias.read_bytes()
    run = fst.TransferRun.open(dataclasses.replace(inputs, source_train_outcomes=str(alias)), out,
                               small_config())
    with pytest.raises(ValueError, match="resolves to the same file"):
        run.prepare()
    assert alias.read_bytes() == before                      # still the input, byte for byte
    assert not (out / "tapes" / "event_table.json").exists() and not (out / "stage_state.json").exists()


def test_a_source_dev_input_named_like_the_state_staging_file_is_refused_not_destroyed(
        world_dir, inputs, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    alias = out / "stage_state.json.tmp"
    shutil.copyfile(world_dir[1]["source_dev_outcomes"], alias)
    before = alias.read_bytes()
    run = fst.TransferRun.open(dataclasses.replace(inputs, source_dev_outcomes=str(alias)), out,
                               small_config())
    with pytest.raises(ValueError, match="resolves to the same file"):
        run.prepare()
    with pytest.raises(ValueError, match="resolves to the same file"):
        run._save_state()                                    # the state writer has the same guard
    assert alias.read_bytes() == before and not (out / "stage_state.json").exists()
    assert not (out / "tapes").exists()                      # nothing was committed before the refusal


def test_a_hard_linked_input_at_a_final_output_name_is_refused(world_dir, inputs, tmp_path):
    out = tmp_path / "run"
    (out / "tapes").mkdir(parents=True)
    source = Path(world_dir[1]["source_dev_outcomes"])
    try:
        os.link(source, out / "tapes" / "seed_0.json")
    except OSError:
        pytest.skip("hard links are not available here")
    before = source.read_bytes()
    run = fst.TransferRun.open(inputs, out, small_config())
    with pytest.raises(ValueError, match="resolves to the same file"):
        run.prepare()
    assert source.read_bytes() == before and not (out / "stage_state.json").exists()


def test_checkpoint_publication_never_deletes_or_truncates_unknown_files(ctx, tmp_path):
    directory = tmp_path / "ck"
    directory.mkdir()
    unknown = directory / "ckpt_unknown.pt"
    unknown.write_bytes(b"unrelated checkpoint-shaped file")
    victim = directory / "owned_source_sentinel.bin"
    victim.write_bytes(b"input sentinel hard linked to a staging name")
    legacy = directory / "ckpt_000000.pt.tmp"
    try:
        os.link(victim, legacy)
    except OSError:
        pytest.skip("hard links are not available here")
    run = fst.ArmRun(ctx, "F", 0)
    first = fst.save_checkpoint(run, directory)
    assert unknown.read_bytes() == b"unrelated checkpoint-shaped file"
    assert victim.read_bytes() == legacy.read_bytes() == b"input sentinel hard linked to a staging name"
    run.macrostep()
    second = fst.save_checkpoint(run, directory)
    assert not (directory / first["weights_file"]).exists()          # the TRACKED predecessor only
    assert (directory / second["weights_file"]).is_file() and unknown.exists() and legacy.exists()


def test_an_unknown_file_at_the_checkpoint_name_is_never_overwritten(ctx, tmp_path):
    directory = tmp_path / "ck"
    directory.mkdir()
    squatter = directory / "ckpt_000000.pt"
    squatter.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="never overwritten"):
        fst.save_checkpoint(fst.ArmRun(ctx, "F", 0), directory)
    assert squatter.read_bytes() == b"not a checkpoint"
    assert not (directory / fst.CHECKPOINT_NAME).exists()


def test_a_failed_publication_removes_only_the_staging_file_it_created(ctx, tmp_path, monkeypatch):
    directory = tmp_path / "ck"
    directory.mkdir()
    bystander = directory / ".ckpt_000000.pt.0badc0de.staging"
    bystander.write_bytes(b"someone else's staging file")
    run = fst.ArmRun(ctx, "F", 0)

    def refuse(source, destination):
        raise OSError("the disk says no")
    monkeypatch.setattr(fst.os, "replace", refuse)
    with pytest.raises(OSError, match="disk says no"):
        fst.save_checkpoint(run, directory)
    monkeypatch.undo()
    assert bystander.read_bytes() == b"someone else's staging file"
    assert [path.name for path in directory.iterdir()] == [bystander.name]


def test_an_interrupted_checkpoint_publication_recovers_to_the_uninterrupted_result(ctx, tmp_path,
                                                                                    monkeypatch):
    straight = fst.train_arm(ctx, "C", 0, tmp_path / "straight")
    directory = tmp_path / "interrupted"
    fst.train_arm(ctx, "C", 0, directory, stop_after=2)          # checkpoint at 2: weights + header
    bystander = directory / ".checkpoint.json.deadbeef.staging"
    bystander.write_bytes(b"someone else's staging file")
    real = fst.write_json_owned

    def crash_at_four(path, document, **kwargs):
        if Path(path).name == fst.CHECKPOINT_NAME and document["cursor"] == 4:
            raise RuntimeError("interrupted after the weights were published, before the header")
        return real(path, document, **kwargs)
    monkeypatch.setattr(fst, "write_json_owned", crash_at_four)
    with pytest.raises(RuntimeError, match="interrupted after the weights"):
        fst.train_arm(ctx, "C", 0, directory)
    monkeypatch.setattr(fst, "write_json_owned", real)
    assert (directory / "ckpt_000004.pt").exists()               # the orphan of the interrupted attempt
    resumed = fst.train_arm(ctx, "C", 0, directory)
    assert resumed.deterministic_result() == straight.deterministic_result()
    assert bystander.read_bytes() == b"someone else's staging file"
    assert sorted(path.name for path in directory.glob("ckpt_*.pt")) == ["ckpt_000004.pt"]


# ===========================================================================
# repair 15c
# ===========================================================================

# ---------------------------------------------------------------------------
# C3: the inputs an update ACTUALLY consumes are verified before anything changes
# ---------------------------------------------------------------------------

def unlock(array):
    """Re-enable the write flag of a copied array (numpy allows it for an array that owns its data)."""
    array.setflags(write=True)
    return array


def test_a_modified_initial_state_is_refused_before_any_model_is_built(ctx, monkeypatch):
    built, real = [], fst.student_from_state

    def spy(*args, **kwargs):
        built.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(fst, "student_from_state", spy)
    state = {name: value.clone() for name, value in ctx.init_states[0].items()}
    private = dataclasses.replace(ctx, init_states={**ctx.init_states, 0: state})  # digest is recorded
    intact = fst.ArmRun(private, "F", 0)                  # an unmodified copy is accepted ...
    assert built
    built.clear()
    state[next(name for name, value in state.items() if value.is_floating_point())].add_(1.0)
    with pytest.raises(ValueError, match="initial state"):
        fst.ArmRun(private, "F", 0)                       # ... a modified one never reaches a model
    assert built == []
    assert fst.tensor_state_digest(intact.state_payload()["nets"]["func"]) == ctx.init_sha256[0]
    with pytest.raises(ValueError, match="initial state"):
        intact.verify_inputs()                            # and the boundary check still refuses it


@pytest.mark.parametrize("arm,target", [
    ("F", "functional_labels"), ("C", "functional_labels"), ("C", "event_composed"),
    ("C", "event_codes"), ("C", "event_rows"), ("R", "event_direct"), ("Q", "event_composed_len")])
def test_a_run_owned_array_edited_after_construction_is_refused_before_the_update(ctx, arm, target):
    run, clean = fst.ArmRun(ctx, arm, 0), fst.ArmRun(ctx, arm, 0)
    run.macrostep()
    clean.macrostep()
    array = unlock(getattr(run.ctx, target))
    array[0] = array[0] + (7.0 if array.dtype.kind == "f" else 1)
    before = full_state(run)
    with pytest.raises(ValueError, match="no longer match"):
        run.macrostep()
    assert full_state(run) == before                       # no parameter, optimizer, cursor or counter
    assert run.state_sha256() == clean.state_sha256() and run.cursor == clean.cursor == 1


@pytest.mark.parametrize("which", ["functional_labels", "functional_batches", "aux_batches",
                                   "q_targets"])
def test_a_run_owned_tape_array_edited_after_construction_is_refused_before_the_update(ctx, which):
    run = fst.ArmRun(ctx, "Q", 0)
    run.macrostep()
    array = unlock(getattr(run.tape, which))
    array[0] = array[0] + (3.0 if array.dtype.kind == "f" else 1)
    before = full_state(run)
    with pytest.raises(ValueError, match="no longer match"):
        run.macrostep()
    assert full_state(run) == before


def test_a_native_feature_row_edited_after_construction_is_refused_before_the_update(ctx):
    private, table = private_table_context(ctx)
    run = fst.ArmRun(private, "C", 0)
    run.macrostep()
    rows = np.concatenate([run.ctx.functional_rows[run.tape.functional_batches[1]],
                           run.ctx.event_rows[run.tape.aux_batches[1]]])
    table.features[int(table.offsets[int(rows.min())]), 0] += 1.0     # a row the NEXT step reads
    before = full_state(run)
    with pytest.raises(ValueError, match="no longer match"):
        run.macrostep()
    assert full_state(run) == before


def test_an_edit_to_a_row_no_batch_reads_is_caught_at_the_next_boundary_not_by_the_update(
        ctx, tmp_path):
    private, table = private_table_context(ctx)
    run = fst.ArmRun(private, "F", 0)
    read = {int(row) for row in run.ctx.functional_rows[run.tape.functional_batches[0]]}
    unread = next(row for row in range(len(table.ids)) if row not in read)
    table.features[int(table.offsets[unread]), 0] += 1.0
    run.macrostep()                                        # every row this step reads is intact
    with pytest.raises(ValueError, match="no longer match"):
        fst.save_checkpoint(run, tmp_path / "ck")          # the whole table is re-verified here
    assert not (tmp_path / "ck").exists()


def test_an_auxiliary_only_step_verifies_its_inputs_before_it_moves_anything(ctx):
    run = fst.ArmRun(ctx, "C", 0)
    unlock(run.ctx.event_composed)[0, 0, 0] += 1
    before = full_state(run)
    with pytest.raises(ValueError, match="no longer match"):
        run.auxiliary_only_step()
    assert full_state(run) == before


def test_the_context_reports_the_actual_instance_mix_and_the_one_sided_literal_control(ctx):
    config, balance = ctx.events.config, ctx.balance
    shapes = config["instance_shapes"]
    assert set(shapes) <= {"2RES+2EQ", "1RES+3EQ"}
    assert sum(shapes.values()) == config["instances_per_sequence"] * config["sequences"]
    assert len(ctx.events) == config["nominal_events"]                      # 4 unique answers each
    output = balance["events_by_output"]
    assert output["residue"] + output["equality"] == len(ctx.events)
    assert output["residue"] == 2 * shapes.get("2RES+2EQ", 0) + shapes.get("1RES+3EQ", 0)
    literal = balance["eq_by_stratum"].get("eq_literal_literal", {"true": 0, "false": 0})
    assert literal["true"] + literal["false"] == shapes.get("1RES+3EQ", 0)  # one control per such instance
    if literal["true"] + literal["false"]:
        assert balance["secondary_eq_balance"]["eq_literal_literal"]["balance"] in (
            "one_sided_unsupported", "both_answers_present")


# ---------------------------------------------------------------------------
# C4: predecessor ownership and cleanup
# ---------------------------------------------------------------------------

def tree_bytes(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


@pytest.mark.parametrize("spelling", ["direct", "case_alias"])
def test_a_protected_predecessor_checkpoint_is_retained_not_deleted(ctx, tmp_path, spelling):
    directory = tmp_path / "ck"
    run = fst.ArmRun(ctx, "F", 0)
    first = fst.save_checkpoint(run, directory)
    predecessor = directory / first["weights_file"]
    protected = predecessor if spelling == "direct" else directory / first["weights_file"].upper()
    if not protected.exists():
        pytest.skip("this file system distinguishes letter case: there is no case alias to test")
    before = tree_bytes(directory)
    run.macrostep()
    second = fst.save_checkpoint(run, directory, protected=[protected])
    after = tree_bytes(directory)
    assert predecessor.is_file()                           # retaining an old checkpoint is fine
    unchanged = {name: data for name, data in before.items() if name != fst.CHECKPOINT_NAME}
    assert unchanged.items() <= after.items()              # every earlier file, byte for byte
    assert set(after) - set(before) == {second["weights_file"]}     # only the new weights were added
    assert fb.read_json(directory / fst.CHECKPOINT_NAME)["cursor"] == 1
    fresh = fst.ArmRun(ctx, "F", 0)
    assert fst.restore_checkpoint(fresh, directory) and fresh.cursor == 1     # and it still resumes


@pytest.mark.parametrize("protect", [False, True])
def test_a_hard_link_to_the_predecessor_is_data_safe_either_way(ctx, tmp_path, protect):
    directory = tmp_path / "ck"
    run = fst.ArmRun(ctx, "F", 0)
    first = fst.save_checkpoint(run, directory)
    alias = directory / "alias_of_the_first_weights.bin"
    try:
        os.link(directory / first["weights_file"], alias)
    except OSError:
        pytest.skip("hard links are not available here")
    kept = alias.read_bytes()
    run.macrostep()
    fst.save_checkpoint(run, directory, protected=[alias] if protect else ())
    assert alias.read_bytes() == kept                      # unlinking one NAME never destroys the other
    assert (directory / first["weights_file"]).exists() is protect    # a protected identity is retained


def test_a_tampered_predecessor_is_refused_before_any_file_changes(ctx, tmp_path):
    directory = tmp_path / "ck"
    run = fst.ArmRun(ctx, "F", 0)
    first = fst.save_checkpoint(run, directory)
    weights = directory / first["weights_file"]
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0x01
    weights.write_bytes(bytes(data))
    before = tree_bytes(directory)
    with pytest.raises(ValueError, match="SHA-256"):
        fst.save_checkpoint(run, directory)                # same cursor: it used to be overwritten
    assert tree_bytes(directory) == before
    run.macrostep()
    with pytest.raises(ValueError, match="SHA-256"):
        fst.save_checkpoint(run, directory)                # next cursor: it used to be deleted
    assert tree_bytes(directory) == before


def test_a_foreign_header_is_refused_before_any_file_changes(ctx, opened, tmp_path):
    directory = tmp_path / "ck"
    fst.save_checkpoint(fst.ArmRun(ctx, "F", 0), directory)
    before = tree_bytes(directory)
    with pytest.raises(ValueError, match="different run"):
        fst.save_checkpoint(fst.ArmRun(ctx, "C", 0), directory)    # identical cursor-0 weights, other arm
    with pytest.raises(ValueError, match="different run"):
        fst.save_checkpoint(fst.ArmRun(ctx, "F", 1), directory)    # another source seed
    other = tampered_context(opened, "config")
    with pytest.raises(ValueError, match="different inputs"):
        fst.save_checkpoint(fst.ArmRun(other, "F", 0), directory)  # same arm and seed, other bindings
    assert tree_bytes(directory) == before


def test_a_malformed_header_file_is_never_replaced(ctx, tmp_path):
    directory = tmp_path / "ck"
    directory.mkdir()
    (directory / fst.CHECKPOINT_NAME).write_text("this is not a checkpoint header", encoding="utf-8")
    before = tree_bytes(directory)
    with pytest.raises(ValueError):
        fst.save_checkpoint(fst.ArmRun(ctx, "F", 0), directory)
    assert tree_bytes(directory) == before


def test_valid_same_run_saves_and_resume_are_unchanged_by_the_ownership_check(ctx, tmp_path):
    straight = fst.train_arm(ctx, "C", 0, tmp_path / "straight")
    directory = tmp_path / "resumed"
    fst.train_arm(ctx, "C", 0, directory, stop_after=2)
    resumed = fst.train_arm(ctx, "C", 0, directory)
    assert resumed.deterministic_result() == straight.deterministic_result()
    assert sorted(path.name for path in directory.glob("ckpt_*.pt")) == [
        f"ckpt_{resumed.total_steps:06d}.pt"]               # only the newest, tracked weights remain


# ===========================================================================
# repair 15d
# ===========================================================================

# ---------------------------------------------------------------------------
# D1: the per-row digest binds the dtype and shape of every consumed slice
# ---------------------------------------------------------------------------

def private_metadata_context(ctx, kind):
    """A context over a private copy of one token-metadata array (identical content, so the digests
    recorded when the context was built still hold) and that copy: an in-place edit of the copy cannot
    reach the shared fixture."""
    array = np.array(getattr(ctx.table, kind), copy=True)
    array.setflags(write=False)
    table = dataclasses.replace(ctx.table, **{kind: array})
    return dataclasses.replace(ctx, table=table), array


def change_in_place(array, how, *, start=0):
    """Change what an array's consumers read WITHOUT replacing the object: one byte, or only the dtype
    or shape descriptor (the bytes stay identical)."""
    if how == "bytes":
        unlock(array)[start] += 1
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            if how == "dtype":
                array.dtype = np.dtype("float64")
            else:
                array.shape = (array.shape[0], 1)
        except (AttributeError, TypeError, ValueError) as error:
            pytest.skip(f"this numpy refuses an in-place {how} change ({error})")


@pytest.mark.parametrize("how", ["bytes", "dtype", "shape"])
@pytest.mark.parametrize("kind", ["role", "chain", "index"])
def test_a_metadata_array_changed_in_place_is_refused_before_the_update(ctx, kind, how):
    private, array = private_metadata_context(ctx, kind)
    run, clean = fst.ArmRun(private, "C", 0), fst.ArmRun(ctx, "C", 0)
    run.macrostep()
    clean.macrostep()
    assert run.state_sha256() == clean.state_sha256()      # an untouched private copy steps identically
    rows = np.concatenate([run.ctx.functional_rows[run.tape.functional_batches[1]],
                           run.ctx.event_rows[run.tape.aux_batches[1]]])
    row = int(rows.min())                                  # a row the NEXT step reads
    digest = fst._row_digest(private.table, row)
    change_in_place(array, how, start=int(private.table.offsets[row]))
    assert fst._row_digest(private.table, row) != digest   # the row digest itself sees the edit
    before = full_state(run)
    with pytest.raises(ValueError, match="no longer match"):
        run.macrostep()
    assert full_state(run) == before                       # no parameter, optimizer, cursor, counter, log
    assert run.state_sha256() == clean.state_sha256() and run.cursor == clean.cursor == 1
    with pytest.raises(ValueError, match="no longer match"):
        run.verify_inputs()                                # and the boundary check agrees


# ---------------------------------------------------------------------------
# D2: the public features stage verifies the in-memory context it consumes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(inputs, tmp_path_factory):
    """One run taken through prepare and train only; every test below works on its own copy of it."""
    config = small_config(steps=2, batch=4, aux_batch=4, arms=("F", "C"), source_seeds=(0,),
                          acquisition_seeds=(0,), label_budgets=(4,), checkpoint_every=1,
                          probe_steps=(0,), panel_sequences=32)
    run = fst.TransferRun.open(inputs, tmp_path_factory.mktemp("trained") / "run", config)
    run.prepare()
    assert run.train() is True
    return {"run": run, "config": config}


def open_trained_copy(trained, inputs, tmp_path):
    """A freshly opened pipeline over a copy of the trained output. Its in-memory context is what
    ``features`` consumes, and the tests change THAT object: reopening after an edit would rebuild the
    token metadata from the cache and hide it."""
    destination = tmp_path / "run"
    shutil.copytree(trained["run"].root, destination)
    return fst.TransferRun.open(inputs, destination, trained["config"])


@pytest.mark.parametrize("kind,how", [("role", "bytes"), ("chain", "bytes"), ("index", "bytes"),
                                      ("features", "bytes"), ("role", "dtype")])
def test_features_refuses_a_changed_in_memory_context_before_it_writes_anything(
        trained, inputs, tmp_path, kind, how):
    run = open_trained_copy(trained, inputs, tmp_path)
    table = run.ctx.table
    assert run._complete("train") and not run._complete("features")
    array = getattr(table, kind)
    if how == "dtype":
        change_in_place(array, how)
    else:
        unlock(array)
        for cid in run.target_ids():
            row = table.position[cid]
            start, end = int(table.offsets[row]), int(table.offsets[row + 1])
            if kind == "features":
                array[start, 0] += 1.0
            else:
                array[start:end] = 0                       # valid values, a different function of the row
    before_tree, before_state = tree_bytes(run.root), copy.deepcopy(run.state)
    with pytest.raises(ValueError, match="no longer match"):
        run.features()
    assert tree_bytes(run.root) == before_tree             # not one file, byte or staging file changed
    assert not run.features_dir.exists() and not run.matrix_inputs.exists()
    assert run.state == before_state and not run._complete("features")


def test_features_still_completes_and_is_idempotent_for_an_unchanged_context(trained, inputs,
                                                                             tmp_path):
    run = open_trained_copy(trained, inputs, tmp_path)
    records = run.features()
    assert set(records) == {"native", "F_s0", "C_s0"} and run._complete("features")
    settled = tree_bytes(run.root)
    assert run.features() == records and tree_bytes(run.root) == settled      # adopted, not rewritten
    reopened = fst.TransferRun.open(inputs, run.root, trained["config"])
    assert reopened.features() == records and tree_bytes(run.root) == settled


def test_features_does_not_hand_out_a_completed_stage_for_a_changed_context(trained, inputs,
                                                                           tmp_path):
    run = open_trained_copy(trained, inputs, tmp_path)
    run.features()
    unlock(run.ctx.table.role)[:] = 0
    settled, state = tree_bytes(run.root), copy.deepcopy(run.state)
    with pytest.raises(ValueError, match="no longer match"):
        run.features()                                     # the cached result is not adopted either
    assert tree_bytes(run.root) == settled and run.state == state


# ===========================================================================
# repair 15e
# ===========================================================================

# ---------------------------------------------------------------------------
# E1: the ordinary dicts that select rows and roles are bound, at every public stage
# ---------------------------------------------------------------------------

# one live-object change per kind -> the refusal it must raise
MAP_CHANGES = {
    "table_position_swap": "id-to-row map",         # two ids exchange rows: sequences under the wrong label
    "roles_member_removed": "cohort roles",         # 31 of 32 target rows
    "roles_member_moved": "cohort roles",           # eval -> train: the union is unchanged, the boundary is not
    "cohort_position_swap": "cohort row map",
    "manifest_role_edit": "cohort manifest"}


def folder_state(root):
    """Every file's bytes and every directory (an empty one counts) under ``root``: 'unchanged' means
    that, not only that the files that existed are the same."""
    root = Path(root)
    return {str(path.relative_to(root)): None if path.is_dir() else path.read_bytes()
            for path in sorted(root.rglob("*"))}


@contextlib.contextmanager
def mutated_maps(run, kind):
    """Apply ONE change to a map of this live pipeline object and put it back afterwards (a module-scoped
    run is shared). The table's own map is read-only and its dataclass frozen, so a changed map has to be
    REPLACED (the in-place edit is its own test below); the cohort dicts are edited in place."""
    table, cohorts = run.ctx.table, run.cohorts
    roles, position, manifest_roles = cohorts.roles, cohorts.position, cohorts.manifest.roles
    saved = (table.position, dict(roles), dict(position), dict(manifest_roles))
    first, second = sorted(roles["target_train"])[:2]
    try:
        if kind == "table_position_swap":
            swapped = dict(table.position)
            swapped[first], swapped[second] = swapped[second], swapped[first]
            object.__setattr__(table, "position", swapped)
        elif kind == "roles_member_removed":
            roles["target_eval"] = roles["target_eval"][:-1]
        elif kind == "roles_member_moved":
            moved = roles["target_eval"][-1]
            roles["target_eval"] = roles["target_eval"][:-1]
            roles["target_train"] = tuple(sorted(roles["target_train"] + (moved,)))
        elif kind == "cohort_position_swap":
            position[first], position[second] = position[second], position[first]
        else:
            manifest_roles["target_eval"] = manifest_roles["target_eval"][:-1]
        yield
    finally:
        object.__setattr__(table, "position", saved[0])
        for target, original in ((roles, saved[1]), (position, saved[2]), (manifest_roles, saved[3])):
            target.clear()
            target.update(original)


def assert_every_id_is_published_with_its_own_native_row(run, records):
    """Independent of ``table.position``: every id is looked up in the verified native cache's OWN id
    order, for the candidate table, the pooled native control and an adapted representation."""
    native_row = {cid: row for row, cid in enumerate(run.cache.ids)}
    identities = run.cache.identities
    lines = (run.matrix_inputs / "candidates.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert [line.split(",")[0] for line in lines] == list(run.target_ids())
    assert all(line.split(",")[1] == identities[native_row[line.split(",")[0]]] for line in lines)
    pooled = fne.pool_token_features(run.cache, fne.POOL_ALL)
    native = fm.load_feature_cache(records["native"]["header"], npz_path=records["native"]["npz"])
    np.testing.assert_allclose(native.features, pooled[[native_row[cid] for cid in native.ids]],
                               atol=1e-6, rtol=1e-6)
    adapted = fm.load_feature_cache(records["C_s0"]["header"], npz_path=records["C_s0"]["npz"])
    net = fst.load_adapter(run.adapters, "C_s0", run.ctx).func_net
    np.testing.assert_allclose(
        adapted.features, fst.extract_function_features(
            run.ctx.table, net, [native_row[cid] for cid in adapted.ids]), atol=1e-5)


@pytest.mark.parametrize("kind", list(MAP_CHANGES))
def test_a_changed_row_or_role_map_is_refused_by_every_public_stage_before_features_publishes(
        trained, inputs, tmp_path, kind):
    run = open_trained_copy(trained, inputs, tmp_path)
    assert run._complete("train") and not run._complete("features")
    before_tree, before_state = folder_state(run.root), copy.deepcopy(run.state)
    with mutated_maps(run, kind):
        for stage in ("prepare", "train", "features"):
            with pytest.raises(ValueError, match=MAP_CHANGES[kind]):
                getattr(run, stage)()
            assert folder_state(run.root) == before_tree, stage      # no file, directory or staging name
            assert run.state == before_state, stage
        with pytest.raises(ValueError, match=MAP_CHANGES[kind]):
            run.ctx.verify_all()                                     # the context gate itself agrees
    assert not run.features_dir.exists() and not run.matrix_inputs.exists()
    records = run.features()                         # the same object, its maps right again: publishes
    assert set(records) == {"native", "F_s0", "C_s0"} and run._complete("features")
    assert_every_id_is_published_with_its_own_native_row(run, records)


@pytest.mark.parametrize("kind", list(MAP_CHANGES))
def test_a_completed_pipeline_does_not_adopt_any_stage_over_a_changed_map(pipeline, kind):
    run = pipeline["run"]               # the SAME object that executed every stage and holds their bytes
    stages = ("prepare", "train", "features", "freeze", "evaluate", "panels", "report")
    assert all(run._complete(stage) for stage in stages)
    settled, state = folder_state(run.root), copy.deepcopy(run.state)
    with mutated_maps(run, kind):
        for stage in stages:
            with pytest.raises(ValueError, match=MAP_CHANGES[kind]):
                getattr(run, stage)()                # unchanged bytes on disk are not adopted blindly
            assert folder_state(run.root) == settled and run.state == state, stage
    for stage in ("features", "freeze", "evaluate"):                 # maps restored: adopted, not rewritten
        getattr(run, stage)()
        assert folder_state(run.root) == settled and run.state == state, stage


def test_the_token_table_row_map_cannot_be_edited_in_place_and_is_owned_by_the_table(ctx):
    table = ctx.table
    first, second = table.ids[:2]
    original = dict(table.position)
    with pytest.raises(TypeError):
        table.position[first], table.position[second] = original[second], original[first]
    with pytest.raises(TypeError):
        del table.position[first]
    assert dict(table.position) == original and table.verify_map()
    handed = dict(original)
    owned = dataclasses.replace(table, position=handed)
    handed[first], handed[second] = original[second], original[first]   # the caller's dict, edited later
    assert dict(owned.position) == original and owned.verify_map()
    with pytest.raises(ValueError, match="id-to-row map"):
        dataclasses.replace(table, position=handed)                      # a swapped map is refused at once
    with pytest.raises(ValueError, match="id-to-row map"):
        dataclasses.replace(table, position={cid: row for cid, row in original.items()
                                             if cid != first})


@pytest.mark.parametrize("field,match", [("roles", "cohort roles"), ("position", "cohort row map")])
def test_build_context_refuses_cohorts_that_no_longer_match_their_manifest_or_the_cache(
        opened, world_dir, field, match):
    cohorts = opened.cohorts
    if field == "roles":
        changed = {**cohorts.roles, "source_train": cohorts.roles["source_train"][:-1]}
    else:
        first, second = cohorts.roles["source_train"][:2]
        changed = {**cohorts.position, first: cohorts.position[second], second: cohorts.position[first]}
    bad = dataclasses.replace(cohorts, **{field: changed})
    with pytest.raises(ValueError, match=match):
        fst.build_context(opened.cache, bad, opened.config, world_dir[0].outcomes["source"])


def test_separately_supplied_cohorts_and_a_reordered_table_are_gated_too(trained, inputs, tmp_path):
    run = open_trained_copy(trained, inputs, tmp_path)
    cohorts, ctx = run.cohorts, run.ctx
    twin = dataclasses.replace(cohorts)                  # a distinct object with identical content
    assert fst.TransferRun(inputs, run.root, run.config, run.cache, twin, ctx).verify_maps() is True
    before = folder_state(run.root)
    shorter = dataclasses.replace(
        cohorts, roles={**cohorts.roles, "target_eval": cohorts.roles["target_eval"][:-1]})
    separate = fst.TransferRun(inputs, run.root, run.config, run.cache, shorter, ctx)
    for call in (separate.preflight, separate.features):
        with pytest.raises(ValueError, match="cohort roles"):
            call()
    # a table, cohorts and context that agree with one another but not with the verified cache's row order
    order = tuple(reversed(ctx.table.ids))
    table = dataclasses.replace(ctx.table, ids=order,
                                position={cid: row for row, cid in enumerate(order)})
    reordered = dataclasses.replace(cohorts, position=dict(table.position))
    foreign = fst.TransferRun(inputs, run.root, run.config, run.cache, reordered,
                              dataclasses.replace(ctx, table=table, cohorts=reordered))
    with pytest.raises(ValueError, match="verified token cache"):
        foreign.features()
    assert folder_state(run.root) == before


def test_unchanged_maps_pass_the_gate_for_fixed_and_variable_layout_pipelines(opened, mixed_source,
                                                                              tmp_path):
    assert opened.verify_maps() is True and opened.ctx.verify_all() is True
    opened.preflight()
    mixed = fst.TransferRun.open(mixed_source["inputs"], tmp_path / "run",
                                 small_config(steps=1, source_seeds=(0,), acquisition_seeds=(0,),
                                              label_budgets=(4,), probe_steps=()))
    assert isinstance(mixed.ctx.layouts, fc.LayoutMap)
    assert mixed.verify_maps() is True and mixed.ctx.verify_all() is True
    mixed.preflight()
