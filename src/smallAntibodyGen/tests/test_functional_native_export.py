"""Native token-feature export, ragged token cache and pooled matrix bridge, on a REAL tiny model.

Every extraction test runs the actual native forward of a tiny, randomly initialised, locally
constructed ``ConditionedDPLM`` on CPU (no weights, no downloads). Oracles are independent of the
code under test wherever one exists: hand-written native ids and role strings for small constructs,
the public AbLang serialisation plus the boundary translation, a direct single-row forward through
``model.native``, a direct ``HookRegistry`` capture, and the model's own ``representation``. What
this establishes is that the machinery computes what it says on a model it did not train; it says
nothing about pretrained weights or biology. No test asserts on source text.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.modules.module as torch_module_lib
from torch import nn

from smallAntibodyGen.experiments import functional_baseline as fb
from smallAntibodyGen.experiments import functional_matrix as fm
from smallAntibodyGen.experiments import functional_native_export as fne
from smallAntibodyGen.experiments import her2_routes_interp as interp
from smallAntibodyGen.experiments import her2_routes_serial as serial
from smallAntibodyGen.models import dplm_policy as dplm
from smallAntibodyGen.models import her2_routes_hooks as hooks_lib

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "functional_native_export.py"

# (candidate id, heavy, light, group, split): lengths 19, 19, 27, 10 (heavy-only), 8, 13 (heavy-only)
ROWS = (
    ("c0", "EVACDEGG", "DIQMTQ", "g0", "train"),
    ("c1", "EVADCEGG", "EIVLTQ", "g1", "train"),
    ("c2", "QVQLVESGGGLV", "DIQMTQSPSS", "g2", "eval"),
    ("c3", "EVACDEG", "", "g3", "train"),
    ("c4", "AC", "D", "g4", "eval"),
    ("c5", "EVQLVESGGG", "", "g5", "train"),
)
ROW_LENGTHS = [19, 19, 27, 10, 8, 13]
FINAL = {"kind": "final_hidden", "name": None}
EXTRACTOR = {"kind": "synthetic", "checkpoint_sha256": None, "source_sha256": None,
             "description": "tiny random-init CPU DPLM"}
SUPERVISION = {"status": "label_free", "description": "random init",
               "sequence_only_exposure": {"status": "none"}}
EXPLICIT = [("display", "a1"), ("affinity", None), ("enrichment", "a2"), ("display", None),
            ("affinity", "a2"), ("enrichment", "a1")]
# families display/affinity/enrichment = 0/1/2, bins ... null = 3, assays a1/a2/null = 0/1/2
EXPLICIT_INDICES = [[0, 3, 0], [1, 3, 2], [2, 3, 1], [0, 3, 2], [1, 3, 1], [2, 3, 0]]


def hook(name):
    return {"kind": "hook", "name": name}


def plan_doc(rows=ROWS, *, mode="none", activation=FINAL, representation_id="rep",
             supervision=SUPERVISION):
    plan_rows = []
    for position, (ident, heavy, light, group, split) in enumerate(rows):
        condition = None
        if mode == "explicit":
            family, assay = EXPLICIT[position % len(EXPLICIT)]
            condition = {"family": family, "assay": assay, "quality": None}
        plan_rows.append({"candidate_id": ident, "heavy": heavy, "light": light,
                          "group_id": group, "split": split, "condition": condition})
    return {"schema": fne.PLAN_SCHEMA, "representation_id": representation_id,
            "conditioning_mode": mode, "activation": copy.deepcopy(activation),
            "feature_supervision": copy.deepcopy(supervision), "extractor": dict(EXTRACTOR),
            "rows": plan_rows}


def write_plan(directory, document, name="plan.json"):
    path = Path(directory) / name
    fb.write_json(path, document)
    return path


def export(model, directory, document=None, *, name="t", batch_size=2, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = fne.read_export_plan(write_plan(directory, document or plan_doc(), f"{name}.plan.json"))
    summary = fne.export_dplm_tokens(model, plan, directory / f"{name}.json",
                                     directory / f"{name}.npz", batch_size=batch_size, **kwargs)
    return plan, summary, fne.load_token_cache(directory / f"{name}.json")


def public_native(heavy, light):
    """Native ids through the PUBLIC route: AbLang serialisation (nothing masked) then the boundary."""
    construct = serial.Construct(heavy=heavy, light=light, sites=(), source="test", record_id="x")
    tokens = serial.ablang_serialization(construct, masked_sites=(), order=()).tokens
    return dplm.ablang_to_dplm(list(tokens))


def reference_hidden(model, heavy, light, *, indices=None):
    """One unpadded row through ``model.native`` directly: the independent forward."""
    ids = public_native(heavy, light)[None, :]
    with interp.controlled_evaluation(model):
        condition = None if indices is None else model.condition_vector(
            torch.tensor([indices], dtype=torch.long))
        return model.native(ids, condition=condition)["last_hidden_state"][0].numpy()


def direct_capture(model, name, heavy, light):
    """A direct HookRegistry capture of one unpadded row, independent of the exporter."""
    registry = hooks_lib.bind_conditioned_dplm(model)
    registry.enable([name])
    try:
        with interp.controlled_evaluation(model):
            model.forward_native_ids(public_native(heavy, light)[None, :])
    finally:
        registry.disable()
    return registry.export(name)["tensor"][0].numpy()


@pytest.fixture
def forward_counter(monkeypatch):
    calls = []
    original = dplm.DPLMEncoder.forward

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(dplm.DPLMEncoder, "forward", counting)
    return calls


def snapshot(model):
    named = list(model.named_parameters(remove_duplicate=False))
    return {"params": {name: p.detach().clone() for name, p in named},
            "buffers": {name: b.detach().clone()
                        for name, b in model.named_buffers(remove_duplicate=False)},
            "flags": [(name, module.training)
                      for name, module in model.named_modules(remove_duplicate=False)],
            "grads": {name: None if p.grad is None else p.grad.detach().clone()
                      for name, p in named},
            "requires_grad": {name: p.requires_grad for name, p in named},
            "rng": torch.get_rng_state().clone(), "hooks": fne.hook_inventory(model)}


def assert_same_state(before, after):
    assert before["flags"] == after["flags"], "a module's training flag moved"
    assert before["requires_grad"] == after["requires_grad"]
    for kind in ("params", "buffers"):
        assert before[kind].keys() == after[kind].keys()
        for name in before[kind]:
            assert torch.equal(before[kind][name], after[kind][name]), (kind, name)
    for name, grad in before["grads"].items():
        assert (grad is None) == (after["grads"][name] is None), name
        if grad is not None:
            assert torch.equal(grad, after["grads"][name]), name
    assert torch.equal(before["rng"], after["rng"]), "the CPU RNG state moved"
    assert before["hooks"] == after["hooks"], "the hook inventory changed"


def listing(directory):
    return sorted(os.listdir(directory))


def cache_files(directory):
    """The cache files (and staging files) an export may have left in ``directory``."""
    return [name for name in listing(directory)
            if name in ("t.json", "t.npz", "t.json.tmp", "t.npz.tmp")]


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def test_a_plan_parses_in_canonical_order_keeps_its_path_and_is_input_order_invariant(tmp_path):
    forward = fne.read_export_plan(write_plan(tmp_path, plan_doc(ROWS)))
    backward = fne.parse_export_plan(plan_doc(tuple(reversed(ROWS))))
    assert [row.candidate_id for row in forward.rows] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert forward.fingerprint() == backward.fingerprint()
    assert forward.path == tmp_path / "plan.json" and backward.path is None
    assert forward.rows[3].identity == "EVACDEG|" and forward.rows[4].identity == "AC|D"


def test_identity_keeps_the_chain_boundary_and_heavy_only_keeps_its_trailing_delimiter():
    assert fne.identity_of("AC", "D") == "AC|D"
    assert fne.identity_of("AC", "D") != fne.identity_of("A", "CD")
    assert fne.identity_of("AC", "") == "AC|"
    boundary = plan_doc((("a", "AC", "D", "g0", "train"), ("b", "A", "CD", "g1", "train")))
    plan = fne.parse_export_plan(boundary)
    assert {row.identity for row in plan.rows} == {"AC|D", "A|CD"}, "ACD must not be the identity"
    framework = plan_doc((("a", "EVACDEGG", "DIQMTQ", "g0", "train"),
                          ("b", "QVACDEGG", "DIQMTQ", "g1", "train")))
    window = fne.parse_export_plan(framework)
    assert window.rows[0].identity != window.rows[1].identity, (
        "the same variable window in two frameworks is two identities")


def mutated(mode="none", **edits):
    document = plan_doc(mode=mode)
    for path, value in edits.items():
        target = document
        keys = path.split("__")
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        last = keys[-1]
        if isinstance(target, list):
            target[int(last)] = value
        else:
            target[last] = value
    return document


BAD_PLANS = {
    "unknown_top_key": (lambda: {**plan_doc(), "extra": 1}, "unknown key"),
    "missing_top_key": (lambda: {k: v for k, v in plan_doc().items() if k != "extractor"},
                        "is missing"),
    "wrong_schema": (lambda: mutated(schema="functional-native-export/2"), "not a functional"),
    "duplicate_id": (lambda: mutated(rows__1__candidate_id="c0"), "repeated"),
    "duplicate_identity": (lambda: mutated(rows__1__heavy="EVACDEGG", rows__1__light="DIQMTQ"),
                           "same full physical identity"),
    "group_crosses_the_split": (lambda: mutated(rows__1__group_id="g0", rows__1__split="eval"),
                                "both the training and the evaluation"),
    "lower_case_residue": (lambda: mutated(rows__0__heavy="evacdegg"), "non-canonical"),
    "unknown_residue": (lambda: mutated(rows__0__heavy="EVACXEGG"), "non-canonical"),
    "whitespace_in_chain": (lambda: mutated(rows__0__light="DIQ MTQ"), "non-canonical"),
    "empty_heavy": (lambda: mutated(rows__0__heavy=""), "must not be empty"),
    "bad_split": (lambda: mutated(rows__0__split="test"), "split must be"),
    "unknown_row_key": (lambda: mutated(rows__0__outcome=1.0), "unknown key"),
    "none_mode_with_a_condition": (
        lambda: mutated(rows__0__condition={"family": "display", "assay": None, "quality": None}),
        "NO condition vector"),
    "explicit_mode_with_a_null_condition": (
        lambda: {**plan_doc(mode="explicit"),
                 "rows": [dict(row, condition=None) for row in plan_doc(mode="explicit")["rows"]]},
        "needs a condition object"),
    "explicit_quality_missing": (
        lambda: {**plan_doc(mode="explicit"), "rows": [
            dict(row, condition={"family": "display", "assay": None})
            for row in plan_doc(mode="explicit")["rows"]]}, "is missing"),
    "explicit_quality_not_null": (
        lambda: {**plan_doc(mode="explicit"), "rows": [
            dict(row, condition={"family": "display", "assay": None, "quality": "high"})
            for row in plan_doc(mode="explicit")["rows"]]}, "explicit null"),
    "explicit_assay_string_null": (
        lambda: {**plan_doc(mode="explicit"), "rows": [
            dict(row, condition={"family": "display", "assay": "null", "quality": None})
            for row in plan_doc(mode="explicit")["rows"]]}, "reserved"),
    "explicit_unknown_family": (
        lambda: {**plan_doc(mode="explicit"), "rows": [
            dict(row, condition={"family": "binding", "assay": None, "quality": None})
            for row in plan_doc(mode="explicit")["rows"]]}, "family must be one of"),
    "bad_conditioning_mode": (lambda: mutated(conditioning_mode="zero"),
                              "conditioning_mode must be one of"),
    "final_hidden_with_a_name": (
        lambda: mutated(activation={"kind": "final_hidden", "name": "route/dplm/block0/resid_post"}),
        "names no tap"),
    "hook_without_a_name": (lambda: mutated(activation={"kind": "hook", "name": None}),
                            "exact name of a registered tap"),
    "hook_with_a_leading_zero_block": (
        lambda: mutated(activation={"kind": "hook", "name": "route/dplm/block01/resid_post"}),
        "exact name of a registered tap"),
    "hook_with_an_unknown_site": (
        lambda: mutated(activation={"kind": "hook", "name": "route/dplm/block0/final_ln"}),
        "exact name of a registered tap"),
    "synthetic_extractor_with_a_checkpoint_hash": (
        lambda: mutated(extractor={**EXTRACTOR, "checkpoint_sha256": "a" * 64}),
        "checkpoint_sha256 must be null"),
    "native_checkpoint_without_hashes": (
        lambda: mutated(extractor={**EXTRACTOR, "kind": "native_checkpoint"}), "must bind BOTH"),
    "external_features_extractor": (
        lambda: mutated(extractor={**EXTRACTOR, "kind": "external_features"}),
        "extractor.kind must be one of"),
    "no_rows": (lambda: mutated(rows=[]), "non-empty list"),
}


@pytest.mark.parametrize("name", sorted(BAD_PLANS))
def test_a_malformed_plan_is_refused_with_its_reason(name):
    build, message = BAD_PLANS[name]
    with pytest.raises(ValueError, match=message):
        fne.parse_export_plan(build())


def test_a_plan_file_with_a_duplicate_json_key_or_nan_is_refused(tmp_path):
    text = json.dumps(plan_doc())
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(text[:-1] + ', "schema": "other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        fne.read_export_plan(duplicate)
    nan = tmp_path / "nan.json"
    nan.write_text(text.replace('"representation_id"', '"x": NaN, "representation_id"'),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        fne.read_export_plan(nan)


def test_the_feature_supervision_history_is_parsed_and_unknown_is_kept_unknown():
    documented = {"status": "documented", "description": "trained on a labelled set",
                  "documented_supervision": [{"assay": "spr", "target": "her2", "label_count": 3,
                                              "label_ids": ["a", "b", "c"]}],
                  "sequence_only_exposure": {"status": "declared"}}
    assert fne.parse_export_plan(plan_doc(supervision=documented)).supervision.document() \
        == fb.SupervisionHistory.from_document(documented, label="t").document()
    unknown = fne.parse_export_plan(plan_doc(supervision={"status": "unknown"}))
    assert unknown.supervision.status == "unknown"


# ---------------------------------------------------------------------------
# native tokens and token maps: hand oracles
# ---------------------------------------------------------------------------

# A=5, C=23, D=13, E=9, F=18, G=6 in the native vocabulary; <cls>=0, <eos>=2
HAND = {
    ("AC", "D"): ([0, 5, 23, 2, 2, 0, 13, 2], "SHHEPSLE"),
    ("AC", ""): ([0, 5, 23, 2, 2], "SHHEP"),
    ("ACD", "EFG"): ([0, 5, 23, 13, 2, 2, 0, 9, 18, 6, 2], "SHHHEPSLLLE"),
    ("ACD", ""): ([0, 5, 23, 13, 2, 2], "SHHHEP"),
}


@pytest.mark.parametrize("chains", sorted(HAND))
def test_native_ids_and_the_encoded_layout_equal_the_hand_values_by_every_route(chains):
    heavy, light = chains
    ids, roles = HAND[chains]
    assert fne.native_token_ids(heavy, light) == tuple(ids)
    assert fne.independent_native_ids(heavy, light) == tuple(ids)
    assert public_native(heavy, light).tolist() == ids
    rows = fne.native_token_map(heavy, light)
    assert fne.encode_token_map(rows) == {"roles": roles, "native_ids": ids}
    assert fne.native_length(heavy, light) == len(ids) == len(rows)
    assert fne.validate_token_map(rows, heavy, light) is True
    assert fne.decode_token_map(fne.encode_token_map(rows), heavy, light) == rows


@pytest.mark.parametrize("chains", sorted(HAND))
def test_residue_positions_are_one_plus_i_and_len_h_plus_four_plus_j(chains):
    heavy, light = chains
    rows = fne.native_token_map(heavy, light)
    for row in rows:
        if row["chain"] == "H":
            assert row["index"] == 1 + row["residue_index"]
            assert row["residue"] == heavy[row["residue_index"]]
        elif row["chain"] == "L":
            assert row["index"] == len(heavy) + 4 + row["residue_index"]
            assert row["residue"] == light[row["residue_index"]]
    roles = {row["index"]: row["role"] for row in rows if row["chain"] is None}
    assert roles[0] == "start" and roles[len(heavy) + 1] == "end"
    assert roles[len(heavy) + 2] == "separator"
    if light:
        assert roles[len(heavy) + 3] == "start" and roles[len(rows) - 1] == "end"
        assert len(rows) == len(heavy) + len(light) + 5
    else:
        assert len(rows) == len(heavy) + 3 and max(roles) == len(rows) - 1
    assert all(row["residue"] is None and row["residue_index"] is None
               for row in rows if row["chain"] is None)


def _damage(**changes):
    def apply(rows):
        for index, edits in changes.items():
            rows[int(index.lstrip("r"))].update(edits)
        return rows
    return apply


TOKEN_MAP_DAMAGE = {
    "unknown_chain": (_damage(r1={"chain": "X"}), "unknown chain"),
    "negative_index": (_damage(r1={"residue_index": -1}), "non-negative integer"),
    "bool_index": (_damage(r1={"residue_index": True}), "non-negative integer"),
    "index_out_of_bounds": (_damage(r1={"residue_index": 99}), "outside the"),
    "duplicate_site": (_damage(r2={"residue_index": 0}), "appears twice"),
    "wrong_residue_letter": (_damage(r1={"residue": "C"}), "chain's residue"),
    "wrong_native_id_for_a_residue": (_damage(r1={"native_id": 23, "token": "C"}),
                                      "is not the id of residue"),
    "special_row_names_a_chain": (_damage(r0={"chain": "H"}), "names no chain"),
    "wrong_special_id": (_damage(r0={"native_id": 2, "token": "<eos>"}), "must carry native id"),
    "unknown_role": (_damage(r0={"role": "bogus"}), "unknown role"),
    "index_is_not_the_position": (_damage(r3={"index": 99}), "is not its position"),
    "token_disagrees_with_id": (_damage(r1={"token": "Z"}), "is not vocabulary entry"),
    "native_id_outside_vocabulary": (_damage(r1={"native_id": 40}), "inside the vocabulary"),
    "native_id_is_a_bool": (_damage(r1={"native_id": True}), "inside the vocabulary"),
}


@pytest.mark.parametrize("name", sorted(TOKEN_MAP_DAMAGE))
def test_a_damaged_token_map_is_refused_naming_the_defect(name):
    damage, message = TOKEN_MAP_DAMAGE[name]
    rows = damage(copy.deepcopy(fne.native_token_map("AC", "D")))
    with pytest.raises(ValueError, match=message):
        fne.validate_token_map(rows, "AC", "D")


def test_structurally_incomplete_token_maps_are_refused():
    rows = copy.deepcopy(fne.native_token_map("AC", "D"))
    with pytest.raises(ValueError, match="truncated or padded"):
        fne.validate_token_map(rows[:-1], "AC", "D")
    with pytest.raises(ValueError, match="truncated or padded"):
        fne.validate_token_map(rows + [dict(rows[-1], index=8)], "AC", "D")
    swapped = copy.deepcopy(rows)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(ValueError, match="is not its position"):
        fne.validate_token_map(swapped, "AC", "D")
    missing_key = copy.deepcopy(rows)
    del missing_key[1]["token"]
    with pytest.raises(ValueError, match="is missing"):
        fne.validate_token_map(missing_key, "AC", "D")
    extra_key = copy.deepcopy(rows)
    extra_key[1]["note"] = 1
    with pytest.raises(ValueError, match="unknown key"):
        fne.validate_token_map(extra_key, "AC", "D")
    with pytest.raises(ValueError, match="truncated or padded"):
        fne.decode_token_map({"roles": "SHHHEPSLE", "native_ids": [0, 5, 23, 5, 2, 2, 0, 13, 2]},
                             "AC", "D")
    with pytest.raises(ValueError, match="outside the"):       # right length, a third heavy residue
        fne.decode_token_map({"roles": "SHHHEPSE", "native_ids": [0, 5, 23, 5, 2, 2, 0, 2]},
                             "AC", "D")
    with pytest.raises(ValueError, match="unknown role code"):
        fne.decode_token_map({"roles": "SHHXPSLE", "native_ids": [0, 5, 23, 2, 2, 0, 13, 2]},
                             "AC", "D")
    with pytest.raises(ValueError, match="truncated or padded"):
        fne.validate_token_map(rows, "AC", "")                  # a paired map is not heavy-only
    reordered = copy.deepcopy(rows)                              # each row valid, the layout is not
    reordered[3]["role"], reordered[4]["role"] = "separator", "end"
    with pytest.raises(ValueError, match="not the native"):
        fne.validate_token_map(reordered, "AC", "D")


# ---------------------------------------------------------------------------
# the model, the export and its numerical oracles
# ---------------------------------------------------------------------------

def test_the_selftest_model_is_the_declared_tiny_random_cpu_model():
    model = fne.tiny_model(seed=4)
    config = model.config
    assert (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
            config.intermediate_size, config.max_position_embeddings,
            config.hidden_dropout_prob) == (32, 2, 4, 64, 64, 0.0)
    assert model.base_identity["pretrained"] is False and model.base_identity["seed"] == 4
    assert {str(p.device) for p in model.parameters()} == {"cpu"} and not model.is_lora
    assert any(float(table.weight.abs().max()) > 0 for table in model._tables())
    assert all(float(table.weight.abs().max()) == 0
               for table in fne.tiny_model(seed=4, table_std=0)._tables())


def test_final_hidden_export_equals_independent_unpadded_forwards_for_mixed_lengths(tmp_path):
    model = fne.tiny_model(seed=1)
    plan, summary, cache = export(model, tmp_path, batch_size=4)    # one batch mixes 27 and 8 tokens
    assert plan.document() == fne.parse_export_plan(plan_doc()).document(), (
        "the export must not edit the caller's plan")
    assert cache.lengths() == ROW_LENGTHS
    assert cache.offsets.tolist() == [0, 19, 38, 65, 75, 83, 96]
    assert cache.features.dtype == np.float32 and cache.features.shape == (96, 32)
    assert summary["token_rows"] == 96 and cache.verify() is True
    for position, (ident, heavy, light, _group, _split) in enumerate(ROWS):
        stored = cache.row_features(ident)
        assert stored.shape == (ROW_LENGTHS[position], 32)
        reference = reference_hidden(model, heavy, light)
        assert np.allclose(stored, reference, atol=1e-5), (ident, np.abs(stored - reference).max())
    assert float(np.abs(cache.features).max()) > 0.05, "a vanishing signal would make this weak"
    hand = reference_hidden(model, "AC", "D")                       # heavy 2 + light 1 -> T = 8
    hand_ids = torch.tensor([[0, 5, 23, 2, 2, 0, 13, 2]])
    with interp.controlled_evaluation(model):
        assert np.allclose(model.native(hand_ids)["last_hidden_state"][0].numpy(), hand, atol=1e-6)


def test_pooled_features_equal_the_native_representation_and_the_hand_means(tmp_path):
    model = fne.tiny_model(seed=1)
    _plan, _summary, cache = export(model, tmp_path, batch_size=3)
    everything = fne.pool_token_features(cache, "all_nonpad_mean")
    residues = fne.pool_token_features(cache, "residue_mean")
    assert everything.shape == residues.shape == (6, 32)
    for position, (_ident, heavy, light, _g, _s) in enumerate(ROWS):
        tokens = serial.ablang_serialization(
            serial.Construct(heavy=heavy, light=light, sites=(), source="t", record_id="x"),
            masked_sites=(), order=()).tokens
        with interp.controlled_evaluation(model):
            native_mean = model.representation(torch.tensor([list(tokens)]))[0].numpy()
        assert np.allclose(everything[position], native_mean, atol=1e-5), position
        reference = reference_hidden(model, heavy, light)
        kept = list(range(1, 1 + len(heavy)))
        if light:
            kept += list(range(len(heavy) + 4, len(heavy) + 4 + len(light)))
        assert np.allclose(residues[position], reference[kept].mean(axis=0), atol=1e-5), position
        assert not np.allclose(everything[position], residues[position], atol=1e-4), (
            "the specials must change the mean: the two poolings are different representations")


@pytest.mark.parametrize("name", ["route/dplm/block0/resid_pre", "route/dplm/block0/attn_out",
                                  "route/dplm/block0/mlp_out", "route/dplm/block1/resid_post"])
def test_a_hook_export_equals_a_direct_registry_capture_at_every_site(tmp_path, name):
    model = fne.tiny_model(seed=1)
    _plan, _summary, cache = export(model, tmp_path, plan_doc(activation=hook(name)),
                                    batch_size=4)
    assert cache.activation["kind"] == "hook" and cache.activation["name"] == name
    assert cache.activation["layout"] == "tokens_first" and cache.activation["position_axis"] == 1
    for ident, heavy, light, _g, _s in ROWS:
        assert np.allclose(cache.row_features(ident), direct_capture(model, name, heavy, light),
                           atol=1e-5), (name, ident)
    assert all(counts["_forward_hooks"] == 0 and counts["_forward_pre_hooks"] == 0
               for counts in fne.hook_inventory(model)["local"].values())


def test_the_final_layer_norm_separates_final_hidden_from_the_last_layers_resid_post(tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, final = export(model, tmp_path / "a", batch_size=3)
    _p, _s, last = export(model, tmp_path / "b",
                          plan_doc(activation=hook("route/dplm/block1/resid_post")), batch_size=3)
    with torch.no_grad():
        normalised = model.native.esm.encoder.emb_layer_norm_after(
            torch.from_numpy(np.array(last.features))).numpy()
    assert np.allclose(normalised, final.features, atol=1e-5), "final_hidden is LN(resid_post)"
    assert float(np.abs(last.features - final.features).max()) > 1e-3, (
        "resid_post is NOT final_hidden; they are distinct named features")
    _p, _s, again = export(model, tmp_path / "c", batch_size=3)
    assert np.array_equal(again.features, final.features), "a hook export left something behind"


def test_batch_size_batch_composition_and_row_order_do_not_change_features_beyond_float32(tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, reference = export(model, tmp_path / "bs1", batch_size=1)
    for size in (2, 4, 6):
        _p, _s, other = export(model, tmp_path / f"bs{size}", batch_size=size)
        assert other.ids == reference.ids and other.lengths() == reference.lengths()
        assert np.allclose(other.features, reference.features, atol=1e-5), size
    shuffled = plan_doc(tuple(reversed(ROWS)))
    _p, _s, reordered = export(model, tmp_path / "shuffled", shuffled, batch_size=4)
    assert reordered.ids == reference.ids, "a plan's row order is not part of the cache's identity"
    assert np.allclose(reordered.features, reference.features, atol=1e-5)
    longer = ROWS + (("c6", "Q" * 40, "D" * 12, "g6", "train"),)     # forces much more padding
    _p, _s, padded = export(model, tmp_path / "padded", plan_doc(longer), batch_size=7)
    assert np.allclose(padded.features[:96], reference.features, atol=1e-5), (
        "the amount of right-padding must not change a row")


def test_discarded_padding_activations_cannot_reach_the_cache(tmp_path, monkeypatch):
    model = fne.tiny_model(seed=1)
    _p, _s, clean = export(model, tmp_path / "clean", batch_size=6)
    real = fne._run_batch

    def poisoned(model_, ids, condition, *, registry, tap):
        activation = real(model_, ids, condition, registry=registry, tap=tap)
        padding = ids.eq(dplm.DPLM_PAD_ID)
        assert bool(padding.any()), "the batch must contain padding for this test to mean anything"
        activation[padding] = 1.0e6
        return activation
    monkeypatch.setattr(fne, "_run_batch", poisoned)
    _p, _s, dirty = export(model, tmp_path / "dirty", batch_size=6)
    assert np.array_equal(dirty.features, clean.features)
    assert float(np.abs(dirty.features).max()) < 1e3


def test_none_passes_no_condition_vector_and_explicit_null_is_a_distinct_condition(
        tmp_path, monkeypatch):
    seen, original = [], dplm.DPLMEncoder.forward

    def spy(self, input_ids, *, condition=None, activation_checkpointing=False):
        seen.append(None if condition is None else tuple(condition.shape))
        return original(self, input_ids, condition=condition,
                        activation_checkpointing=activation_checkpointing)
    monkeypatch.setattr(dplm.DPLMEncoder, "forward", spy)
    model = fne.tiny_model(seed=1)                                   # nonzero condition tables
    _p, _s, none = export(model, tmp_path / "none", batch_size=3)
    assert seen == [None, None], "'none' must pass NO condition vector to the native forward"
    plan, _s, explicit = export(model, tmp_path / "explicit", plan_doc(mode="explicit"),
                                batch_size=3)
    assert seen[2:] == [(3, 1, 32), (3, 1, 32)], "explicit passes one (batch, 1, hidden) vector"
    assert none.conditions == (None,) * 6 and none.condition_indices == (None,) * 6
    assert [item["family"] for item in explicit.conditions] == [e[0] for e in EXPLICIT]
    assert [item["assay"] for item in explicit.conditions] == [e[1] for e in EXPLICIT]
    assert all(item["quality"] is None for item in explicit.conditions)
    assert [list(item) for item in explicit.condition_indices] == EXPLICIT_INDICES
    assert float(np.abs(explicit.features - none.features).max()) > 1e-4
    for position, (ident, heavy, light, _g, _s2) in enumerate(ROWS):
        conditioned = reference_hidden(model, heavy, light, indices=EXPLICIT_INDICES[position])
        assert np.allclose(explicit.row_features(ident), conditioned, atol=1e-5), ident
    zero_indices = reference_hidden(model, "EVACDEGG", "DIQMTQ", indices=[0, 0, 0])
    assert float(np.abs(zero_indices - none.row_features("c0")).max()) > 1e-4, (
        "zero indices are a real display/low/a1 request, not 'no condition'")
    flat = fne.tiny_model(seed=1, table_std=0)                       # zero tables: no-op, bit-identical
    _p, _s, flat_none = export(flat, tmp_path / "flat_none", batch_size=3)
    _p, _s, flat_explicit = export(flat, tmp_path / "flat_explicit", plan_doc(mode="explicit"),
                                   batch_size=3)
    assert np.array_equal(flat_none.features, flat_explicit.features)
    assert explicit.conditioning_mode == "explicit" and plan.conditioning_mode == "explicit"


def test_an_unknown_assay_or_disabled_conditioning_is_refused_before_any_compute(
        tmp_path, forward_counter):
    model = fne.tiny_model(seed=1)
    unknown = plan_doc(mode="explicit")
    unknown["rows"][0]["condition"]["assay"] = "zz"
    with pytest.raises(ValueError, match="not enrolled"):
        export(model, tmp_path / "unknown", unknown)
    model.use_conditions = False
    with pytest.raises(ValueError, match="conditioning disabled"):
        export(model, tmp_path / "disabled", plan_doc(mode="explicit"))
    _p, _s, fine = export(model, tmp_path / "none_ok", plan_doc(mode="none"))   # none needs no vector
    assert fine.conditioning_mode == "none"
    assert len(forward_counter) == 3, "only the final 'none' export may run the forward (3 batches)"
    assert cache_files(tmp_path / "unknown") == [] and cache_files(tmp_path / "disabled") == []


def test_overlength_is_refused_before_compute_and_never_truncated(tmp_path, forward_counter):
    model = fne.tiny_model(seed=1)                                   # 64 positions
    overlong = plan_doc((("long", "A" * 70, "", "g0", "train"),))
    with pytest.raises(ValueError, match="nothing is truncated"):
        export(model, tmp_path / "long", overlong)
    exact = plan_doc((("fits", "A" * 61, "", "g0", "train"),))       # 61 + 3 = 64 tokens
    _p, _s, cache = export(model, tmp_path / "fits", exact)
    assert cache.lengths() == [64] and len(forward_counter) == 1


@pytest.mark.parametrize("kind", ["module_forward", "module_pre", "global_forward", "global_pre"])
def test_preexisting_forward_hooks_are_refused_before_compute_and_none_is_left_behind(
        tmp_path, forward_counter, kind):
    model = fne.tiny_model(seed=1)
    layer = model.native.esm.encoder.layer[0]
    handle = {"module_forward": lambda: layer.register_forward_hook(lambda m, i, o: None),
              "module_pre": lambda: layer.register_forward_pre_hook(lambda m, i: None),
              "global_forward": lambda: torch_module_lib.register_module_forward_hook(
                  lambda m, i, o: None),
              "global_pre": lambda: torch_module_lib.register_module_forward_pre_hook(
                  lambda m, i: None)}[kind]()
    try:
        with pytest.raises(ValueError, match="hook"):
            export(model, tmp_path / "refused", plan_doc(activation=hook("route/dplm/block1/mlp_out")))
        assert forward_counter == [] and cache_files(tmp_path / "refused") == []
    finally:
        handle.remove()
    _p, _s, cache = export(model, tmp_path / "ok", plan_doc(activation=hook("route/dplm/block1/mlp_out")))
    assert cache.verify() is True
    inventory = fne.hook_inventory(model)
    assert all(not any(counts.values()) for counts in inventory["local"].values())
    assert not any(inventory["global"].values()), "the exporter must remove its own handle"


def test_unrelated_backward_hooks_survive_a_successful_export(tmp_path):
    model = fne.tiny_model(seed=1)
    layer = model.native.esm.encoder.layer[0]
    handle = layer.register_full_backward_hook(lambda module, grad_in, grad_out: None)
    try:
        before = fne.hook_inventory(model)
        _p, _s, cache = export(model, tmp_path, plan_doc(activation=hook("route/dplm/block0/resid_post")))
        assert fne.hook_inventory(model) == before and len(layer._backward_hooks) == 1
        assert cache.verify() is True
    finally:
        handle.remove()


def test_unsupported_model_types_are_refused(tmp_path, forward_counter):
    class Sub(dplm.ConditionedDPLM):
        pass
    model = fne.tiny_model(seed=1)
    model.__class__ = Sub
    with pytest.raises(ValueError, match="unmodified ConditionedDPLM"):
        export(model, tmp_path / "subclass")
    model.__class__ = dplm.ConditionedDPLM
    model.native.forward = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="instance-level callable"):
        export(model, tmp_path / "patched")
    del model.native.forward
    model.native.extra = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="not a supported native"):
        export(model, tmp_path / "custom")
    del model.native.extra
    assert forward_counter == []
    _p, _s, cache = export(model, tmp_path / "restored")
    assert cache.verify() is True
    with pytest.raises(ValueError, match="needs an ExportPlan"):
        fne.export_dplm_tokens(model, plan_doc(), tmp_path / "x.json", tmp_path / "x.npz",
                               batch_size=2)


DOCUMENTED = {"status": "documented", "description": "trained on a labelled set",
              "documented_supervision": [{"assay": "spr", "target": "her2", "label_count": 2,
                                          "label_ids": ["a", "b"]}],
              "sequence_only_exposure": {"status": "declared"}}


def _bypass_condition(plan):
    """A frozen row cannot be reassigned normally; this is what a determined caller could do."""
    object.__setattr__(plan.rows[0], "condition",
                       {"family": "display", "assay": None, "quality": None})


# name -> (mode, supervision, edit applied to the PARSED plan, message the refusal must carry)
MUTATED_PLANS = {
    "condition_quality_becomes_a_bin": (
        "explicit", SUPERVISION, lambda p: p.rows[0].condition.__setitem__("quality", "high"),
        "explicit null"),
    "condition_gains_an_unknown_key": (
        "explicit", SUPERVISION, lambda p: p.rows[0].condition.__setitem__("note", 1),
        "unknown key"),
    "condition_family_becomes_unknown": (
        "explicit", SUPERVISION, lambda p: p.rows[0].condition.__setitem__("family", "binding"),
        "family must be one of"),
    "condition_assay_becomes_the_reserved_string": (
        "explicit", SUPERVISION, lambda p: p.rows[0].condition.__setitem__("assay", "null"),
        "reserved"),
    "a_condition_appears_in_none_mode": ("none", SUPERVISION, _bypass_condition,
                                         "NO condition vector"),
    "activation_kind_becomes_unknown": (
        "none", SUPERVISION, lambda p: p.activation.__setitem__("kind", "bogus"),
        "activation.kind must be one of"),
    "final_hidden_gains_a_tap_name": (
        "none", SUPERVISION,
        lambda p: p.activation.__setitem__("name", "route/dplm/block0/resid_post"), "names no tap"),
    "extractor_kind_becomes_external": (
        "none", SUPERVISION, lambda p: p.extractor.__setitem__("kind", "external_features"),
        "extractor.kind must be one of"),
    "synthetic_extractor_gains_a_checkpoint": (
        "none", SUPERVISION, lambda p: p.extractor.__setitem__("checkpoint_sha256", "a" * 64),
        "checkpoint_sha256 must be null"),
    "supervision_label_count_disagrees_with_its_ids": (
        "none", DOCUMENTED, lambda p: p.supervision.documented[0].__setitem__("label_count", 5),
        "label ids but declares"),
    "supervision_entry_gains_an_unknown_key": (
        "none", DOCUMENTED, lambda p: p.supervision.documented[0].__setitem__("bogus", 1),
        "unknown key"),
}


@pytest.mark.parametrize("name", sorted(MUTATED_PLANS))
def test_a_mutated_parsed_plan_is_refused_before_any_forward_or_write_and_is_not_normalised(
        tmp_path, forward_counter, name):
    mode, supervision, mutate, message = MUTATED_PLANS[name]
    model = fne.tiny_model(seed=1)
    plan = fne.read_export_plan(write_plan(tmp_path, plan_doc(mode=mode, supervision=supervision)))
    mutate(plan)
    edited, files, state = copy.deepcopy(plan.document()), listing(tmp_path), snapshot(model)
    with pytest.raises(ValueError, match=message):
        fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz", batch_size=2)
    assert forward_counter == [], "a refused plan must never reach the native forward"
    assert listing(tmp_path) == files and cache_files(tmp_path) == []
    assert plan.document() == edited, "the refusal must leave the caller's (invalid) plan as it was"
    assert_same_state(state, snapshot(model))


def _replace(**changes):
    return lambda plan: dataclasses.replace(plan, **changes)


def _first_row(**changes):
    return lambda plan: dataclasses.replace(
        plan, rows=(dataclasses.replace(plan.rows[0], **changes),) + plan.rows[1:])


# a plan built by hand (or by dataclasses.replace) rather than by the parser
MALFORMED_PLANS = {
    "supervision_is_not_a_history": (_replace(supervision=None), "must be a SupervisionHistory"),
    "activation_is_a_string": (_replace(activation="final_hidden"), "must be mappings"),
    "extractor_is_missing": (_replace(extractor=None), "must be mappings"),
    "rows_are_documents_not_plan_rows": (
        lambda plan: dataclasses.replace(plan, rows=tuple(row.document() for row in plan.rows)),
        "must hold PlanRow"),
    "no_rows": (_replace(rows=()), "non-empty list"),
    "a_repeated_row": (lambda plan: dataclasses.replace(plan, rows=plan.rows + plan.rows[:1]),
                       "repeated"),
    "a_heavy_chain_that_is_not_text": (_first_row(heavy=None), "must be a string"),
    "an_unsafe_representation_id": (_replace(representation_id="a/b"), "representation_id"),
    "an_unknown_conditioning_mode": (_replace(conditioning_mode="zero"),
                                     "conditioning_mode must be one of"),
    "a_path_that_is_not_a_path": (_replace(path=42), "plan.path must be None or a path"),
}


@pytest.mark.parametrize("name", sorted(MALFORMED_PLANS))
def test_a_hand_built_malformed_plan_is_refused_coherently_before_any_forward(
        tmp_path, forward_counter, name):
    build, message = MALFORMED_PLANS[name]
    model = fne.tiny_model(seed=1)
    plan = build(fne.parse_export_plan(plan_doc()))
    files = listing(tmp_path)
    with pytest.raises(ValueError, match=message):
        fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz", batch_size=2)
    assert forward_counter == [] and listing(tmp_path) == files


def test_a_valid_edit_of_a_parsed_plan_is_exported_as_it_now_stands_with_a_matching_fingerprint(
        tmp_path):
    model = fne.tiny_model(seed=1)
    plan = fne.read_export_plan(write_plan(tmp_path, plan_doc(mode="explicit")))
    parsed = plan.fingerprint()
    plan.rows[0].condition["assay"] = "a2"                  # still a valid, enrolled condition
    assert plan.fingerprint() != parsed
    summary = fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz",
                                     batch_size=2)
    cache = fne.load_token_cache(tmp_path / "t.json")
    assert cache.conditions[0]["assay"] == "a2" and list(cache.condition_indices[0]) == [0, 3, 1]
    assert summary["plan_sha256"] == plan.fingerprint() == fne.parse_export_plan(
        plan.document()).fingerprint(), "the reported plan and the exported request must be one"
    assert str(tmp_path / "plan.json") in summary["protected_input_paths"], (
        "the retained plan path still protects the original file")


def test_editing_the_callers_plan_during_the_export_cannot_change_what_is_exported(
        tmp_path, monkeypatch):
    model = fne.tiny_model(seed=1)
    plan = fne.read_export_plan(write_plan(tmp_path, plan_doc(mode="explicit")))
    parsed = plan.fingerprint()
    real = fne._run_batch

    def editing(model_, ids, condition, *, registry, tap):
        plan.rows[0].condition["assay"] = "a2"
        return real(model_, ids, condition, registry=registry, tap=tap)
    monkeypatch.setattr(fne, "_run_batch", editing)
    summary = fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz",
                                     batch_size=2)
    cache = fne.load_token_cache(tmp_path / "t.json")
    assert cache.conditions[0]["assay"] == "a1" and summary["plan_sha256"] == parsed, (
        "the export works from a detached snapshot of the plan")


def test_a_hand_built_plan_with_unsorted_rows_is_canonicalised_not_refused(tmp_path):
    model = fne.tiny_model(seed=1)
    parsed = fne.parse_export_plan(plan_doc())
    shuffled = dataclasses.replace(parsed, rows=tuple(reversed(parsed.rows)))
    summary = fne.export_dplm_tokens(model, shuffled, tmp_path / "t.json", tmp_path / "t.npz",
                                     batch_size=4)
    assert fne.load_token_cache(tmp_path / "t.json").ids == ("c0", "c1", "c2", "c3", "c4", "c5")
    assert summary["plan_sha256"] == parsed.fingerprint()


def test_the_export_leaves_every_piece_of_model_state_alone_on_success_and_on_native_failure(
        tmp_path, monkeypatch):
    model = fne.tiny_model(seed=1)
    model.train()
    model.native.esm.encoder.layer[1].eval()                          # a deliberately mixed model
    weight = model.native.esm.encoder.layer[0].output.dense.weight
    weight.grad = torch.full_like(weight, 0.25)                       # a pre-existing gradient
    model.condition_bin.weight.requires_grad_(False)
    torch.manual_seed(1234)
    before = snapshot(model)
    fingerprint = fne.runtime_fingerprint(model)["sha256"]
    _p, _s, cache = export(model, tmp_path / "ok", batch_size=2)
    assert_same_state(before, snapshot(model))
    assert fne.runtime_fingerprint(model)["sha256"] == fingerprint == cache.runtime_sha256
    assert any(training for _name, training in before["flags"]) and not all(
        training for _name, training in before["flags"]), "the fixture must stay mixed"
    calls, original = [], dplm.DPLMEncoder.forward

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected native failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(dplm.DPLMEncoder, "forward", flaky)
    with pytest.raises(RuntimeError, match="injected native failure"):
        export(model, tmp_path / "failed", plan_doc(activation=hook("route/dplm/block1/resid_post")),
               batch_size=2)
    assert_same_state(before, snapshot(model))
    assert cache_files(tmp_path / "failed") == [], "a failed export must publish nothing"
    assert torch.equal(weight.grad, torch.full_like(weight, 0.25))


def test_a_model_change_in_the_middle_of_an_export_is_refused_without_publishing(
        tmp_path, monkeypatch):
    model = fne.tiny_model(seed=1)
    real, state = fne._run_batch, {"calls": 0}

    def moving(model_, ids, condition, *, registry, tap):
        output = real(model_, ids, condition, registry=registry, tap=tap)
        state["calls"] += 1
        if state["calls"] == 1:
            with torch.no_grad():
                model_.condition_bin.weight[0, 0] += 1.0
        return output
    monkeypatch.setattr(fne, "_run_batch", moving)
    with pytest.raises(ValueError, match="changed during the export"):
        export(model, tmp_path, batch_size=2)
    assert state["calls"] >= 2 and cache_files(tmp_path) == []


def _lora_model():
    model = fne.tiny_model(seed=2, lora={"rank": 4, "alpha": 8.0, "dropout": 0.0})
    with torch.no_grad():
        for module in model.native.lora_modules().values():
            module.lora_B.normal_(0.0, 0.1)
    return model


def _first_lora(model):
    return next(iter(model.native.lora_modules().values()))


def _untie(model):
    decoder = model.native.lm_head.decoder
    decoder.weight = nn.Parameter(decoder.weight.detach().clone())


MUTATIONS = {
    "condition_table_weight": lambda m: m.condition_bin.weight.data.__setitem__((0, 0), 1.5),
    "adapter_weight": lambda m: _first_lora(m).lora_B.data.__setitem__((0, 0), 2.5),
    "base_weight": lambda m: m.native.esm.encoder.layer[0].attention.self.query.weight.data
        .__setitem__((0, 0), 2.5),
    "nonpersistent_rotary_buffer": lambda m: m.native.rotary_inv_freq.mul_(1.001),
    "id_table_buffer": lambda m: m._canonical_native.add_(1),
    "adapter_enable_flag": lambda m: setattr(_first_lora(m), "lora_enabled", False),
    "adapter_dropout_p": lambda m: setattr(_first_lora(m).lora_dropout, "p", 0.5),
    "adapter_scaling": lambda m: setattr(_first_lora(m), "lora_scaling", 3.0),
    "layer_norm_eps": lambda m: setattr(m.native.esm.encoder.emb_layer_norm_after, "eps", 1e-3),
    "config_value": lambda m: object.__setattr__(m.config, "rope_theta", 5000.0),
    "use_conditions": lambda m: setattr(m, "use_conditions", False),
    "requires_grad_flag": lambda m: m.condition_bin.weight.requires_grad_(False),
    "activation_checkpointing_flag": lambda m: setattr(m, "activation_checkpointing", True),
    "attention_head_count": lambda m: setattr(
        m.native.esm.encoder.layer[0].attention.self, "num_heads", 2),
    "tied_parameter_split": _untie,
    "tokenizer_table": lambda m: m._native_to_canonical.add_(1),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_the_runtime_fingerprint_changes_for_every_relevant_mutation(name):
    base = _lora_model()
    mutated_model = copy.deepcopy(base)
    assert fne.runtime_fingerprint(mutated_model)["sha256"] == fne.runtime_fingerprint(base)["sha256"]
    MUTATIONS[name](mutated_model)
    assert fne.runtime_fingerprint(mutated_model)["sha256"] != fne.runtime_fingerprint(base)["sha256"]


def test_the_fingerprint_is_content_based_and_ignores_only_execution_flags():
    model = _lora_model()
    digest = fne.runtime_fingerprint(model)["sha256"]
    assert fne.runtime_fingerprint(copy.deepcopy(model))["sha256"] == digest
    model.eval()
    assert fne.runtime_fingerprint(model)["sha256"] == digest, (
        "train/eval flags do not move an eval-mode forward and are checked separately")
    document = fne.runtime_fingerprint(model)["document"]
    assert ["native.esm.embeddings.word_embeddings.weight",
            "native.lm_head.decoder.weight"] in document["model"]["tied_parameter_names"]
    buffers = {entry["name"]: entry for entry in document["model"]["buffers"]}
    assert buffers["native.rotary_inv_freq"]["persistent"] is False
    assert document["execution"] == {
        "module_mode": "eval", "gradients": "disabled", "autocast": "disabled",
        "activation_checkpointing_active": False,
        "rng": "CPU and initialised-CUDA generators forked and restored"}
    assert document["model"]["lora"]["settings"]["rank"] == 4
    assert document["model"]["declared_identity"]["base_identity"]["pretrained"] is False


def test_bfloat16_parameters_are_hashed_by_their_raw_bytes():
    config = fne.dplm.DPLMConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                                 intermediate_size=64, max_position_embeddings=64,
                                 hidden_dropout_prob=0.0)
    native = dplm.NativeDPLM.random_init(config, seed=1, std=0.3, storage_dtype=torch.bfloat16)
    model = dplm.ConditionedDPLM(native, fne.conditions_lib.ConditionSpec(assays=("a1",)))
    document = fne.runtime_fingerprint(model)
    assert any(entry["dtype"] == "torch.bfloat16" for entry in document["document"]["model"]["parameters"])
    digest = document["sha256"]
    with torch.no_grad():
        model.native.esm.encoder.layer[0].attention.self.query.weight[0, 0] += 1.0
    assert fne.runtime_fingerprint(model)["sha256"] != digest


def _toggle(value):
    return not value


def _other_precision(value):
    return "high" if value == "highest" else "highest"


# record key -> (read, write, the other value): the host's own global switches, driven directly
BACKEND_FLAGS = {
    "float32_matmul_precision": (torch.get_float32_matmul_precision,
                                 torch.set_float32_matmul_precision, _other_precision),
    "mkldnn_enabled": (lambda: torch.backends.mkldnn.enabled,
                       lambda value: setattr(torch.backends.mkldnn, "enabled", value), _toggle),
    "sdp_math_enabled": (torch.backends.cuda.math_sdp_enabled,
                         torch.backends.cuda.enable_math_sdp, _toggle),
    "sdp_flash_enabled": (torch.backends.cuda.flash_sdp_enabled,
                          torch.backends.cuda.enable_flash_sdp, _toggle),
    "sdp_mem_efficient_enabled": (torch.backends.cuda.mem_efficient_sdp_enabled,
                                  torch.backends.cuda.enable_mem_efficient_sdp, _toggle),
    "sdp_cudnn_enabled": (torch.backends.cuda.cudnn_sdp_enabled,
                          torch.backends.cuda.enable_cudnn_sdp, _toggle),
    "cudnn_benchmark": (lambda: torch.backends.cudnn.benchmark,
                        lambda value: setattr(torch.backends.cudnn, "benchmark", value), _toggle),
}


# Torch >= 2.9 keeps per-context fp32 precision strings that ALIAS the legacy switches. A child
# context (cuda/matmul, mkldnn/matmul) starts as "none" (inherit); ``set_float32_matmul_precision``
# writes an explicit value into the children, and setting the legacy flag back leaves them "ieee"
# instead of "none". The fingerprint correctly sees that as a different state, so a fixture that
# only restores the NAMED flag leaks it. These (backend, op) pairs are saved and restored raw,
# parents before children. A torch without the C hooks has no such contexts and nothing to restore.
PRECISION_CONTEXTS = (("generic", "all"), ("cuda", "all"), ("mkldnn", "all"),
                      ("cuda", "matmul"), ("mkldnn", "matmul"))


def snapshot_precision_contexts():
    """``{(backend, op): raw precision string}`` including an inherited ``"none"``; ``{}`` when this
    torch does not expose the per-context getter (or a context)."""
    getter = getattr(torch._C, "_get_fp32_precision_getter", None)
    saved = {}
    if getter is None:
        return saved
    for context in PRECISION_CONTEXTS:
        try:
            saved[context] = getter(*context)
        except (RuntimeError, TypeError, ValueError):
            continue
    return saved


def restore_precision_contexts(saved):
    """Put every saved context back, parents (``all``) before children (``matmul``)."""
    setter = getattr(torch._C, "_set_fp32_precision_setter", None)
    if setter is None:
        return
    for context in PRECISION_CONTEXTS:
        if context in saved:
            setter(*context, saved[context])


@contextlib.contextmanager
def flipped_backend(name):
    """Change one host-global setting and ALWAYS put it back, even if the body or the check that the
    change took effect fails. The named legacy flag is restored first, then the aliased precision
    contexts exactly (including inherited ``"none"``); the restoration is itself asserted against a
    full snapshot of every recorded backend setting, so no flag can leak into a later test."""
    read, write, other = BACKEND_FLAGS[name]
    original = read()
    contexts = snapshot_precision_contexts()
    before = fne.backend_settings()
    try:
        write(other(original))
        assert read() != original, f"{name} cannot be changed on this build"
        yield original
    finally:
        write(original)
        restore_precision_contexts(contexts)
        assert read() == original, f"{name} was not restored"
        assert snapshot_precision_contexts() == contexts, (
            f"{name}: the aliased fp32 precision contexts were not restored exactly")
        assert fne.backend_settings() == before, f"{name}: a backend setting leaked"


def test_flipped_backend_restores_the_aliased_precision_contexts_exactly():
    contexts, before = snapshot_precision_contexts(), fne.backend_settings()
    with flipped_backend("float32_matmul_precision"):
        assert fne.backend_settings() != before
    assert snapshot_precision_contexts() == contexts
    assert fne.backend_settings() == before


def test_flipped_backend_restores_everything_when_the_body_raises():
    contexts, before = snapshot_precision_contexts(), fne.backend_settings()
    with pytest.raises(RuntimeError, match="body failed"):
        with flipped_backend("float32_matmul_precision"):
            raise RuntimeError("body failed")
    assert snapshot_precision_contexts() == contexts
    assert fne.backend_settings() == before


@pytest.mark.parametrize("name", sorted(BACKEND_FLAGS))
def test_the_runtime_fingerprint_changes_with_each_numerical_backend_setting_and_returns(name):
    model = fne.tiny_model(seed=1)
    cuda_before = torch.cuda.is_initialized()
    base = fne.runtime_fingerprint(model)
    with flipped_backend(name):
        moved = fne.runtime_fingerprint(model)
        assert moved["sha256"] != base["sha256"], f"{name} is not part of the runtime identity"
        assert (moved["document"]["environment"]["backend_settings"][name]
                != base["document"]["environment"]["backend_settings"][name])
    assert fne.runtime_fingerprint(model)["sha256"] == base["sha256"]
    assert torch.cuda.is_initialized() == cuda_before, "reading the flags must not initialise CUDA"


def test_recorded_backend_settings_are_real_reads_and_missing_flags_are_reported_not_invented(
        monkeypatch):
    settings = fne.backend_settings()
    assert settings["float32_matmul_precision"] == torch.get_float32_matmul_precision()
    assert settings["mkldnn_enabled"] == torch.backends.mkldnn.enabled
    assert settings["sdp_math_enabled"] == torch.backends.cuda.math_sdp_enabled()
    assert settings["sdp_flash_enabled"] == torch.backends.cuda.flash_sdp_enabled()
    assert settings["deterministic_algorithms"] == torch.are_deterministic_algorithms_enabled()
    assert settings["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
    assert all(value is None or isinstance(value, (bool, int, str)) for value in settings.values())
    model = fne.tiny_model(seed=1)
    base = fne.runtime_fingerprint(model)
    assert base["document"]["environment"]["backend_settings"] == settings
    monkeypatch.delattr(torch.backends.cuda, "cudnn_sdp_enabled")      # a torch without that switch
    older = fne.backend_settings()
    assert older["sdp_cudnn_enabled"] == fne.UNSUPPORTED
    assert ({key: value for key, value in older.items() if key != "sdp_cudnn_enabled"}
            == {key: value for key, value in settings.items() if key != "sdp_cudnn_enabled"})
    assert fne.runtime_fingerprint(model)["sha256"] != base["sha256"], (
        "an unsupported flag is a recorded fact, not the same identity as a present one")

    def unreadable():
        raise RuntimeError("legacy and new precision APIs disagree")
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", unreadable)
    assert fne.backend_settings()["sdp_math_enabled"] == "unreadable (RuntimeError)"


@pytest.mark.parametrize("restore_after", ["never", "second_batch"])
def test_a_backend_setting_changed_during_the_export_is_refused_without_publishing(
        tmp_path, monkeypatch, restore_after):
    model = fne.tiny_model(seed=1)
    real, calls = fne._run_batch, []
    original = torch.get_float32_matmul_precision()
    contexts, before = snapshot_precision_contexts(), fne.backend_settings()

    def moving(model_, ids, condition, *, registry, tap):
        output = real(model_, ids, condition, registry=registry, tap=tap)
        calls.append(1)
        if len(calls) == 1:
            torch.set_float32_matmul_precision(_other_precision(original))
        elif restore_after == "second_batch":
            torch.set_float32_matmul_precision(original)     # a flip that would hide from the end
        return output
    monkeypatch.setattr(fne, "_run_batch", moving)
    try:
        with pytest.raises(ValueError, match="changed during the export"):
            export(model, tmp_path, batch_size=2)             # six rows: three batches
    finally:
        torch.set_float32_matmul_precision(original)
        restore_precision_contexts(contexts)         # the legacy setter alone leaves children "ieee"
    assert torch.get_float32_matmul_precision() == original
    assert snapshot_precision_contexts() == contexts and fne.backend_settings() == before
    assert len(calls) == 1, "the change is caught right after the batch that saw it"
    assert cache_files(tmp_path) == []


# ---------------------------------------------------------------------------
# paths, aliases and publication
# ---------------------------------------------------------------------------

def test_paths_are_validated_before_any_compute_and_nothing_is_touched(tmp_path, forward_counter):
    model = fne.tiny_model(seed=1)
    plan_path = write_plan(tmp_path, plan_doc())
    plan = fne.read_export_plan(plan_path)
    sibling = tmp_path / "candidates.csv"
    sibling.write_text("unrelated", encoding="utf-8")
    other = tmp_path / "elsewhere"
    other.mkdir()
    stale = tmp_path / "stale.json.tmp"
    stale.write_bytes(b"not ours")
    before = listing(tmp_path)

    def attempt(header, npz, message, **kwargs):
        with pytest.raises(ValueError, match=message):
            fne.export_dplm_tokens(model, plan, header, npz, batch_size=2, **kwargs)
    attempt(tmp_path / "same.npz", tmp_path / "same.npz", "same physical file")
    attempt(tmp_path / "h.json", other / "h.npz", "header's directory")
    attempt(tmp_path / "h.json", tmp_path / "bad name.npz", "plain file name")
    attempt(tmp_path / "h.json", tmp_path / "h.txt", "plain file name")
    attempt(plan_path, tmp_path / "h.npz", "resolves to the same file")        # the plan retains its path
    attempt(sibling, tmp_path / "h.npz", "resolves to the same file",
            protected_input_paths=[sibling])
    attempt(tmp_path / "h.json", sibling, "plain file name|resolves to the same file",
            protected_input_paths=[sibling])
    attempt(tmp_path / "stale.json", tmp_path / "stale.npz", "already exists",
            overwrite=True)                                                    # stale staging file
    with pytest.raises(ValueError, match="not a single path"):
        fne.export_dplm_tokens(model, plan, tmp_path / "h.json", tmp_path / "h.npz", batch_size=2,
                               protected_input_paths=str(sibling))
    assert forward_counter == [] and listing(tmp_path) == before
    assert stale.read_bytes() == b"not ours" and sibling.read_text() == "unrelated"


def test_a_hard_link_to_a_protected_input_is_an_alias(tmp_path, forward_counter):
    model = fne.tiny_model(seed=1)
    plan_path = write_plan(tmp_path, plan_doc())
    link = tmp_path / "linked.json"
    try:
        os.link(plan_path, link)
    except (OSError, NotImplementedError):
        pytest.skip("this file system cannot make a hard link")
    plan = fne.read_export_plan(plan_path)
    with pytest.raises(ValueError, match="resolves to the same file"):
        fne.export_dplm_tokens(model, plan, link, tmp_path / "linked.npz", batch_size=2)
    assert forward_counter == [] and not (tmp_path / "linked.npz").exists()


@pytest.mark.skipif(os.name != "nt", reason="letter-case aliasing is a Windows file-system fact")
def test_a_letter_case_alias_of_a_protected_input_is_refused_on_windows(tmp_path):
    model = fne.tiny_model(seed=1)
    protected = tmp_path / "Candidates.CSV"
    protected.write_text("x", encoding="utf-8")
    plan = fne.read_export_plan(write_plan(tmp_path, plan_doc()))
    with pytest.raises(ValueError, match="resolves to the same file"):
        fne.export_dplm_tokens(model, plan, tmp_path / "candidates.csv", tmp_path / "c.npz",
                               batch_size=2, protected_input_paths=[protected])
    assert protected.read_text() == "x"


def test_overwrite_replaces_final_outputs_only_and_existing_outputs_are_refused_by_default(tmp_path):
    model = fne.tiny_model(seed=1)
    plan, first, cache = export(model, tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz", batch_size=2)
    again = fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz",
                                   batch_size=3, overwrite=True)
    reloaded = fne.load_token_cache(tmp_path / "t.json")
    assert np.allclose(reloaded.features, cache.features, atol=1e-5)
    assert again["batch_size"] == 3 and first["batch_size"] == 2
    assert not (tmp_path / "t.json.tmp").exists() and not (tmp_path / "t.npz.tmp").exists()


def test_a_failed_publish_leaves_an_incomplete_state_the_loader_refuses(tmp_path, monkeypatch):
    model = fne.tiny_model(seed=1)
    plan = fne.read_export_plan(write_plan(tmp_path, plan_doc()))
    original = Path.replace

    def flaky(self, target):
        if str(target).endswith(".json"):
            raise OSError("injected replace failure")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", flaky)
    with pytest.raises(OSError, match="injected replace failure"):
        fne.export_dplm_tokens(model, plan, tmp_path / "t.json", tmp_path / "t.npz", batch_size=2)
    monkeypatch.undo()
    assert (tmp_path / "t.npz").exists() and not (tmp_path / "t.json").exists()
    assert not (tmp_path / "t.json.tmp").exists() and not (tmp_path / "t.npz.tmp").exists(), (
        "only the staging files this call created are removed, and they ARE removed")
    with pytest.raises(ValueError, match="does not exist"):
        fne.load_token_cache(tmp_path / "t.json")


def test_a_new_npz_under_an_old_header_is_stale_and_refused(tmp_path):
    _p, _s, _cache = export(fne.tiny_model(seed=1), tmp_path / "a")
    _p, _s, _other = export(fne.tiny_model(seed=7), tmp_path / "b")
    (tmp_path / "a" / "t.npz").write_bytes((tmp_path / "b" / "t.npz").read_bytes())
    with pytest.raises(ValueError, match="STALE"):
        fne.load_token_cache(tmp_path / "a" / "t.json")


# ---------------------------------------------------------------------------
# the ragged token cache: loading, ownership and corruption
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def good(tmp_path_factory):
    """One 'none', one 'explicit' and one 'hook' (last block's resid_post) export of the six
    mixed-length rows: (header document, features, offsets) each, plus the loaded 'none' cache."""
    directory = tmp_path_factory.mktemp("good_cache")
    model = fne.tiny_model(seed=1)
    _p, none_summary, cache = export(model, directory / "none", batch_size=3)
    _p, explicit_summary, _explicit = export(model, directory / "explicit",
                                             plan_doc(mode="explicit"), batch_size=3)
    _p, hook_summary, _hook = export(model, directory / "hook", batch_size=3,
                                     document=plan_doc(activation=hook("route/dplm/block1/resid_post")))

    def parts(summary):
        header = json.loads(Path(summary["header"]).read_text(encoding="utf-8"))
        with np.load(summary["npz"]) as archive:
            return header, archive["features"].copy(), archive["offsets"].copy()
    return {"none": parts(none_summary), "explicit": parts(explicit_summary),
            "hook": parts(hook_summary), "cache": cache}


def test_a_loaded_cache_is_a_frozen_owned_copy_and_verify_catches_edits_to_the_arrays(tmp_path):
    model = fne.tiny_model(seed=1)
    plan, _summary, cache = export(model, tmp_path)
    assert cache.ids == ("c0", "c1", "c2", "c3", "c4", "c5") and isinstance(cache.ids, tuple)
    assert cache.identities[3] == "EVACDEG|" and cache.groups[4] == "g4"
    assert cache.splits == ("train", "train", "eval", "train", "eval", "train")
    assert cache.hidden_size == 32 and cache.header_path == tmp_path / "t.json"
    assert cache.npz_path == tmp_path / "t.npz"
    with pytest.raises(ValueError, match="read-only"):
        cache.features[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        cache.offsets[0] = 1
    row = cache.row_features("c4")
    row += 5.0                                                         # a copy: no effect on the cache
    snapshot_features, snapshot_offsets = cache.snapshot()
    assert cache.verify() is True and not snapshot_features.flags.writeable
    cache.extractor["description"] = "edited by a caller"            # accessors hand out copies
    assert cache.extractor["description"] == "tiny random-init CPU DPLM"
    cache.features.setflags(write=True)                                # a caller defeats the flag...
    cache.features[0, 0] += 1.0
    with pytest.raises(ValueError, match="no longer match"):
        cache.verify()                                                 # ...and verify still catches it
    assert snapshot_features[0, 0] != cache.features[0, 0], "snapshots are independent copies"
    assert np.array_equal(snapshot_offsets, cache.offsets)
    with pytest.raises(ValueError, match="no longer match"):
        fne.pool_token_features(cache, "all_nonpad_mean")
    assert plan.document()["rows"][0]["heavy"] == "EVACDEGG"


def test_a_token_cache_summary_and_header_carry_the_documented_contract(good):
    header, features, offsets = good["none"]
    assert header["schema"] == "functional-native-token-cache/1"
    assert sorted(header) == sorted(fne.HEADER_KEYS)
    assert sorted(header["storage"]["arrays"]) == ["features", "offsets"]
    assert header["storage"]["arrays"]["features"]["dtype"] == "float32"
    assert header["storage"]["arrays"]["features"]["shape"] == [96, 32]
    assert header["storage"]["arrays"]["offsets"]["dtype"] == "int64"
    assert header["storage"]["arrays"]["offsets"]["shape"] == [7]
    assert features.dtype == np.float32 and offsets.dtype == np.int64
    assert header["runtime"]["document"]["schema"] == "functional-native-runtime/1"
    assert header["extractor"]["kind"] == "synthetic" and header["extractor"]["checkpoint_sha256"] is None
    assert header["feature_space"]["provenance"]["status"] == "label_free"
    summary = good["cache"].summary()
    assert summary["rows"] == 6 and summary["heavy_only_rows"] == 2
    assert summary["tokens_per_row"] == {"min": 8, "max": 27} and summary["token_rows"] == 96


def write_cache(directory, document, features, offsets, *, payload=None):
    """Seal a (possibly semantically corrupted) cache so every HASH is consistent: only the targeted
    semantic check can then refuse it."""
    directory.mkdir(parents=True, exist_ok=True)
    sealed, sealed_payload = fne._seal(document, features, offsets)
    data = sealed_payload if payload is None else payload
    sealed["storage"]["file_sha256"] = hashlib.sha256(data).hexdigest()
    (directory / sealed["storage"]["file"]).write_bytes(data)
    header = directory / "h.json"
    header.write_text(json.dumps(sealed), encoding="utf-8")
    return header


def npz_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _offsets(edit):
    def build(document, features, offsets):
        edit(offsets)
        return document, features, offsets, None
    return build


def _swap_offsets(offsets):
    offsets[2], offsets[3] = offsets[3], offsets[2]


def _set(array, index, value):
    array[index] = value


ARRAY_CASES = {
    "offsets_not_increasing": (_offsets(_swap_offsets), "strictly increasing"),
    "offsets_do_not_start_at_zero": (_offsets(lambda o: _set(o, 0, 1)), "must start at 0"),
    "offsets_end_past_the_features": (_offsets(lambda o: _set(o, -1, o[-1] + 1)), "offsets end at"),
    "offsets_do_not_match_the_chains": (_offsets(lambda o: _set(o, 3, o[3] + 1)),
                                        "do not match the token counts"),
    "offsets_too_short": (lambda d, f, o: (d, f, o[:-1], None), "offsets.shape"),
    "features_with_nan": (lambda d, f, o: (d, _with(f, (0, 0), np.nan), o, None),
                          "NaN or infinity"),
    "features_with_inf": (lambda d, f, o: (d, _with(f, (1, 1), np.inf), o, None),
                          "NaN or infinity"),
    "features_missing_a_token_row": (lambda d, f, o: (d, f[:-1], o, None), "disagrees with the"),
    "features_stored_as_float64": (
        lambda d, f, o: (d, f, o, npz_bytes(features=f.astype(np.float64), offsets=o)),
        "features have dtype float64"),
    "offsets_stored_as_int32": (
        lambda d, f, o: (d, f, o, npz_bytes(features=f, offsets=o.astype(np.int32))),
        "offsets have dtype"),
    "object_array": (
        lambda d, f, o: (d, f, o, npz_bytes(features=np.array(f.tolist(), dtype=object), offsets=o)),
        "object arrays are refused"),
    "extra_member": (
        lambda d, f, o: (d, f, o, npz_bytes(features=f, offsets=o, extra=np.zeros(2))),
        "exactly the members"),
    "missing_member": (lambda d, f, o: (d, f, o, npz_bytes(features=f)), "exactly the members"),
    "truncated_archive": (
        lambda d, f, o: (d, f, o, npz_bytes(features=f, offsets=o)[:200]),
        "cannot be read as a numeric NPZ"),
    "not_an_archive": (lambda d, f, o: (d, f, o, b"definitely not an npz archive at all"),
                       "without pickle"),
}


def _with(array, index, value):
    array[index] = value
    return array


@pytest.mark.parametrize("name", sorted(ARRAY_CASES))
def test_array_level_corruption_is_refused_even_when_every_hash_is_consistent(tmp_path, good, name):
    document, features, offsets = (copy.deepcopy(good["none"][0]), good["none"][1].copy(),
                                   good["none"][2].copy())
    build, message = ARRAY_CASES[name]
    document, features, offsets, payload = build(document, features, offsets)
    header = write_cache(tmp_path / name, document, features, offsets, payload=payload)
    with pytest.raises(ValueError, match=message):
        fne.load_token_cache(header)


def _edit_rows(key, position, value):
    def build(document):
        document["rows"][key][position] = value
        return document
    return build


def _swap_ids(document):
    ids = document["rows"]["ids"]
    ids[0], ids[1] = ids[1], ids[0]
    return document


def _same_identity(document):
    rows = document["rows"]
    rows["heavy"][1], rows["light"][1] = rows["heavy"][0], rows["light"][0]
    rows["identities"][1] = rows["identities"][0]
    return document


def _crossing_group(document):
    document["rows"]["groups"][1] = document["rows"]["groups"][0]
    document["rows"]["splits"][1] = "eval"
    return document


def _map_edit(edit):
    def build(document):
        maps = document["rows"]["token_maps"]
        edit(maps[0])
        document["token_map_sha256"] = fb.digest(maps)       # keep the digest honest: only the map is wrong
        return document
    return build


def _swap_native(encoded):
    ids = encoded["native_ids"]
    ids[1], ids[2] = ids[2], ids[1]


def _flip_role(encoded):
    encoded["roles"] = encoded["roles"].replace("H", "L", 1)


def _runtime_edit(consistent):
    def build(document):
        document["runtime"]["document"]["model"]["use_conditions"] = False
        if consistent:
            document["runtime"]["sha256"] = fb.digest(document["runtime"]["document"])
        return document
    return build


def _nested(path, value):
    def build(document):
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return document
    return build


HEADER_CASES = {
    "unknown_header_key": (_nested(("extra",), 1), "unknown key"),
    "ids_out_of_order": (_swap_ids, "strictly increasing"),
    "identity_not_heavy_pipe_light": (_edit_rows("identities", 0, "AAAA|"),
                                      "identities are not heavy"),
    "two_rows_one_identity": (_same_identity, "same full physical identity"),
    "a_group_crosses_the_split": (_crossing_group, "both the training and the evaluation"),
    "token_map_native_ids_swapped": (_map_edit(_swap_native), "chain's residue"),
    "token_map_roles_edited": (_map_edit(_flip_role), "chain's residue"),
    "token_map_digest_wrong": (_nested(("token_map_sha256",), "0" * 64),
                               "do not match token_map_sha256"),
    "runtime_document_edited_without_its_digest": (_runtime_edit(False), "does not match its digest"),
    "runtime_edited_consistently_but_the_feature_space_names_the_old_one": (
        _runtime_edit(True), "feature space is not the one"),
    "activation_record_edited": (_nested(("activation", "location"), "somewhere"),
                                 "activation record"),
    "feature_space_edited": (_nested(("feature_space", "names"), ["x"]), "feature space is not"),
    "synthetic_extractor_with_a_checkpoint": (
        _nested(("extractor", "checkpoint_sha256"), "a" * 64), "checkpoint_sha256 must be null"),
    "none_mode_with_a_condition": (
        _nested(("rows", "conditions", 0), {"family": "display", "assay": None, "quality": None}),
        "passes no condition vector"),
    "unknown_schema": (_nested(("schema",), "functional-native-token-cache/9"), "not a functional"),
}


@pytest.mark.parametrize("name", sorted(HEADER_CASES))
def test_header_level_corruption_is_refused_even_when_every_hash_is_consistent(tmp_path, good, name):
    document, features, offsets = (copy.deepcopy(good["none"][0]), good["none"][1].copy(),
                                   good["none"][2].copy())
    build, message = HEADER_CASES[name]
    header = write_cache(tmp_path / name, build(document), features, offsets)
    with pytest.raises(ValueError, match=message):
        fne.load_token_cache(header)


EXPLICIT_CASES = {
    "condition_index_edited": (_nested(("rows", "condition_indices", 0, 2), 1),
                               "recorded condition_spec's encoding"),
    "assay_outside_the_recorded_spec": (_nested(("rows", "conditions", 0, "assay"), "zz"),
                                        "outside the recorded condition_spec"),
    "quality_not_null": (_nested(("rows", "conditions", 0, "quality"), "high"), "explicit null"),
    "mode_flipped_to_none_with_conditions_present": (
        _nested(("conditioning_mode",), "none"), "passes no condition vector"),
}


@pytest.mark.parametrize("name", sorted(EXPLICIT_CASES))
def test_condition_bindings_are_recomputed_against_the_recorded_condition_spec(tmp_path, good, name):
    document, features, offsets = (copy.deepcopy(good["explicit"][0]),
                                   good["explicit"][1].copy(), good["explicit"][2].copy())
    build, message = EXPLICIT_CASES[name]
    header = write_cache(tmp_path / name, build(document), features, offsets)
    with pytest.raises(ValueError, match=message):
        fne.load_token_cache(header)
    clean = write_cache(tmp_path / "clean", copy.deepcopy(good["explicit"][0]),
                        good["explicit"][1].copy(), good["explicit"][2].copy())
    assert fne.load_token_cache(clean).conditioning_mode == "explicit"


def reseal_runtime(document, edit):
    """Edit the RECORDED runtime and recompute everything that binds it (its digest, the digest the
    feature space names, then - in ``write_cache`` - the array and cache hashes): the hash chain is
    consistent, so only a semantic cross-check can still object."""
    edit(document["runtime"]["document"])
    digest = fb.digest(document["runtime"]["document"])
    document["runtime"]["sha256"] = digest
    document["feature_space"]["schema"]["runtime_sha256"] = digest
    return document


def resealed(tmp_path, good, kind, name, edit=None, *, document_edit=None):
    document, features, offsets = (copy.deepcopy(good[kind][0]), good[kind][1].copy(),
                                   good[kind][2].copy())
    document = reseal_runtime(document, edit or (lambda runtime: None))
    if document_edit is not None:
        document = document_edit(document)
    return write_cache(tmp_path / name, document, features, offsets)


def config_edit(which, field, value):
    def edit(runtime):
        if value is _DELETE:
            del runtime["model"][which][field]
        else:
            runtime["model"][which][field] = value
    return edit


_DELETE = object()


def both_configs(field, value):
    def edit(runtime):
        for which in ("config", "native_config"):
            config_edit(which, field, value)(runtime)
    return edit


def hook_block(block):
    """Point a hook cache's activation record and feature space at ``block`` (hashes resealed)."""
    def edit(document):
        name = f"route/dplm/block{block}/resid_post"
        document["activation"] = fne._activation_record({"kind": "hook", "name": name})
        document["feature_space"]["schema"]["activation"]["name"] = name
        return document
    return edit


# kind -> name -> (runtime edit, message): contradictions inside a header whose hashes all agree.
# The tiny model's rows are 8..27 native tokens; the recorded runtime says hidden 32, 2 layers, 64
# positions.
RUNTIME_CONTRADICTIONS = {
    "wrapper_hidden_size_differs_from_the_feature_width": (
        "none", config_edit("config", "hidden_size", 64),
        "recorded runtime model.config.hidden_size is 64"),
    "native_hidden_size_differs_from_the_feature_width": (
        "none", config_edit("native_config", "hidden_size", 64),
        "recorded runtime model.native_config.hidden_size is 64"),
    "hidden_size_is_a_bool": ("none", config_edit("config", "hidden_size", True),
                              "positive integer"),
    "hidden_size_is_a_float": ("none", config_edit("native_config", "hidden_size", 32.0),
                               "positive integer"),
    "hidden_size_is_missing": ("none", config_edit("config", "hidden_size", _DELETE),
                               "positive integer"),
    "layer_count_is_zero": ("none", config_edit("native_config", "num_hidden_layers", 0),
                            "positive integer"),
    "positions_are_a_string": ("none", config_edit("config", "max_position_embeddings", "64"),
                               "positive integer"),
    "a_row_is_longer_than_the_recorded_positions": (
        "none", config_edit("config", "max_position_embeddings", 26), "27 native tokens"),
    "a_row_is_longer_than_the_native_trunks_positions": (
        "none", config_edit("native_config", "max_position_embeddings", 20), "27 native tokens"),
    "the_config_is_not_an_object": (
        "none", lambda runtime: runtime["model"].__setitem__("config", 3), "must be an object"),
    "the_config_is_absent": ("none", lambda runtime: runtime["model"].pop("native_config"),
                             "is malformed"),
    "the_hook_block_is_beyond_the_wrapper_layer_count": (
        "hook", config_edit("config", "num_hidden_layers", 1), "taps block 1 but the recorded"),
    "the_hook_block_is_beyond_the_native_layer_count": (
        "hook", config_edit("native_config", "num_hidden_layers", 1),
        "taps block 1 but the recorded"),
    "explicit_conditioning_with_conditions_disabled": (
        "explicit", lambda runtime: runtime["model"].__setitem__("use_conditions", False),
        "use_conditions=true"),
    "explicit_conditioning_with_a_non_boolean_flag": (
        "explicit", lambda runtime: runtime["model"].__setitem__("use_conditions", 1),
        "use_conditions=true"),
    "explicit_conditioning_without_the_flag": (
        "explicit", lambda runtime: runtime["model"].pop("use_conditions"), "is malformed"),
}


@pytest.mark.parametrize("name", sorted(RUNTIME_CONTRADICTIONS))
def test_a_runtime_that_contradicts_the_rest_of_the_header_is_refused_with_every_hash_consistent(
        tmp_path, good, name):
    kind, edit, message = RUNTIME_CONTRADICTIONS[name]
    header = resealed(tmp_path, good, kind, name, edit)
    with pytest.raises(ValueError, match=message):
        fne.load_token_cache(header)


def test_a_hook_block_that_does_not_exist_is_refused_even_when_the_hashes_agree(tmp_path, good):
    header = resealed(tmp_path, good, "hook", "block999", document_edit=hook_block(999))
    with pytest.raises(ValueError, match="taps block 999 but the recorded"):
        fne.load_token_cache(header)
    last = resealed(tmp_path, good, "hook", "block1", document_edit=hook_block(1))
    assert fne.load_token_cache(last).activation["name"] == "route/dplm/block1/resid_post"
    first = resealed(tmp_path, good, "hook", "block0", document_edit=hook_block(0))
    assert fne.load_token_cache(first).activation["name"] == "route/dplm/block0/resid_post"


def test_the_contract_checks_accept_every_consistent_cache_and_only_read_the_recorded_runtime(
        tmp_path, good):
    for kind in ("none", "explicit", "hook"):                 # paired and heavy-only rows alike
        header = resealed(tmp_path, good, kind, f"as_exported_{kind}")
        loaded = fne.load_token_cache(header)
        assert loaded.cache_sha256 == good[kind][0]["cache_sha256"] and loaded.verify() is True
        assert any(not light for light in loaded.light) and any(loaded.light)
    longest = max(fne.load_token_cache(resealed(tmp_path, good, "none", "ref")).lengths())
    assert longest == 27
    snug = resealed(tmp_path, good, "none", "snug",
                    both_configs("max_position_embeddings", longest))
    assert fne.load_token_cache(snug).hidden_size == 32, "a row that exactly fills the positions fits"
    flag_off = resealed(
        tmp_path, good, "none", "unconditioned",
        lambda runtime: runtime["model"].__setitem__("use_conditions", False))
    assert fne.load_token_cache(flag_off).conditioning_mode == "none", (
        "conditioning_mode 'none' passes no vector, so a disabled-conditions runtime is consistent")


def test_final_hidden_is_not_a_block_tap_so_a_one_layer_runtime_still_loads_it(tmp_path, good):
    one_layer = both_configs("num_hidden_layers", 1)
    final = resealed(tmp_path, good, "none", "final_one_layer", one_layer)
    assert fne.load_token_cache(final).activation["kind"] == "final_hidden"
    block = resealed(tmp_path, good, "hook", "hook_one_layer", one_layer)
    with pytest.raises(ValueError, match="taps block 1 but the recorded"):
        fne.load_token_cache(block)


def test_hash_layer_corruption_is_refused(tmp_path, good):
    source = good["cache"]
    header_text = Path(source.header_path).read_text(encoding="utf-8")
    npz_bytes_ = Path(source.npz_path).read_bytes()
    flipped = tmp_path / "flipped"
    flipped.mkdir()
    (flipped / "t.json").write_text(header_text, encoding="utf-8")
    data = bytearray(npz_bytes_)
    data[len(data) // 2] ^= 0x01
    (flipped / "t.npz").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="STALE"):
        fne.load_token_cache(flipped / "t.json")
    edited = tmp_path / "edited"
    edited.mkdir()
    (edited / "t.npz").write_bytes(npz_bytes_)
    (edited / "t.json").write_text(
        header_text.replace('"representation_id":"rep"', '"representation_id":"nope"'),
        encoding="utf-8")
    assert '"representation_id":"nope"' in (edited / "t.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="cache_sha256 does not match"):
        fne.load_token_cache(edited / "t.json")
    nan = tmp_path / "nan"
    nan.mkdir()
    (nan / "t.npz").write_bytes(npz_bytes_)
    (nan / "t.json").write_text(header_text.replace('"schema"', '"x":NaN,"schema"', 1),
                                encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON"):
        fne.load_token_cache(nan / "t.json")
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "t.npz").write_bytes(npz_bytes_)
    (duplicate / "t.json").write_text(header_text.rstrip()[:-1] + ',"schema":"x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        fne.load_token_cache(duplicate / "t.json")
    absent = tmp_path / "absent"
    absent.mkdir()
    (absent / "t.json").write_text(header_text, encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        fne.load_token_cache(absent / "t.json")
    escaping = json.loads(header_text)
    escaping["storage"]["file"] = "../evil.npz"
    (absent / "escape.json").write_text(json.dumps(escaping), encoding="utf-8")
    with pytest.raises(ValueError, match="plain file name"):
        fne.load_token_cache(absent / "escape.json")


def test_an_explicit_npz_path_overrides_the_declared_name(tmp_path):
    _p, _s, cache = export(fne.tiny_model(seed=1), tmp_path / "a")
    moved = tmp_path / "somewhere_else.npz"
    moved.write_bytes(Path(cache.npz_path).read_bytes())
    loaded = fne.load_token_cache(tmp_path / "a" / "t.json", npz_path=moved)
    assert loaded.npz_path == moved and loaded.cache_sha256 == cache.cache_sha256
    with pytest.raises(ValueError, match="does not exist"):
        fne.load_token_cache(tmp_path / "a" / "t.json", npz_path=tmp_path / "missing.npz")


# ---------------------------------------------------------------------------
# the pooled bridge into the matrix
# ---------------------------------------------------------------------------

WINDOWS = [("r0", "ACDE", "DIQMTQ", "gA", "train"), ("r1", "ADCE", "EIVLTQ", "gA", "train"),
           ("r2", "CADE", "DIQMTQ", "gB", "train"), ("r3", "CDAE", "EIVLTQ", "gB", "train"),
           ("r4", "DACE", "DIQMTQ", "gC", "eval"), ("r5", "DCAE", "EIVLTQ", "gC", "eval"),
           ("r6", "EACD", "DIQMTQ", "gD", "eval"), ("r7", "EADC", "EIVLTQ", "gD", "eval")]
ROSTER = tuple((ident, "EV" + window + "GG", light, group, split)
               for ident, window, light, group, split in WINDOWS)


def identity_rows(rows):
    return [(ident, fne.identity_of(heavy, light), group, split)
            for ident, heavy, light, group, split in rows]


def manifest_document(identities, *, alphabet=None, length=None):
    alphabet = alphabet or "".join(sorted(set("".join(identity for _i, identity, _g, _s in identities))))
    return {"schema": fm.MANIFEST_SCHEMA, "target": "t", "assay": "a",
            "utility": {"name": "u", "units": "x", "scale": "linear", "higher_is_better": True,
                        "useful_threshold": 0.0},
            "grouping_rationale": "whole groups are held out",
            "candidates": {"identity_kind": "complete_sequence", "alphabet": alphabet,
                           "length": length or len(identities[0][1])},
            "prior": {"kind": "uniform", "semantics": "uniform",
                      "provenance": {"status": "label_free", "description": "none",
                                     "sequence_only_exposure": {"status": "none"}}}}


def write_world(directory, rows=ROSTER, *, identities=None):
    """A candidate CSV and a manifest for a roster; returns (csv, manifest file, table, manifest)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    listed = identities if identities is not None else identity_rows(rows)
    csv_path, manifest_path = directory / "candidates.csv", directory / "manifest.json"
    csv_path.write_text("candidate_id,sequence,group_id,split,prior_log_weight\n" + "\n".join(
        f"{i},{s},{g},{sp},0.0" for i, s, g, sp in listed) + "\n", encoding="utf-8")
    fb.write_json(manifest_path, manifest_document(listed))
    manifest = fm.load_matrix_manifest(manifest_path)
    table = fb.read_candidates(csv_path, alphabet=manifest.alphabet, length=manifest.length)
    return csv_path, manifest_path, table, manifest


def pool(cache, table, manifest, directory, name, pooling="all_nonpad_mean", **kwargs):
    return fne.write_pooled_matrix_cache(
        cache, table, manifest, Path(directory) / f"{name}.json", Path(directory) / f"{name}.npz",
        representation_id=name, pooling=pooling, **kwargs)


def test_pooled_caches_carry_provenance_join_the_matrix_loader_and_run_no_second_forward(
        tmp_path, forward_counter):
    model = fne.tiny_model(seed=1)
    supervision = {"status": "unknown"}
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(ROSTER, supervision=supervision),
                           batch_size=3)
    exports = len(forward_counter)
    csv_path, manifest_path, table, manifest = write_world(tmp_path / "world")
    roster_before, document_before = table.roster(), manifest.document()
    protected = [csv_path, manifest_path]
    cache_before = cache.snapshot()
    summaries = {}
    for pooling in fne.POOLINGS:
        summaries[pooling] = pool(cache, table, manifest, tmp_path / "pooled", pooling[:3],
                                  pooling=pooling, protected_input_paths=protected)
    assert len(forward_counter) == exports, "pooling must reuse the token cache: no second forward"
    assert table.roster() == roster_before and manifest.document() == document_before
    assert protected == [csv_path, manifest_path], "the caller's list is not edited"
    assert np.array_equal(cache.features, cache_before[0]) and cache.verify() is True
    for pooling, summary in summaries.items():
        loaded = fm.load_feature_cache(summary["header"])
        fm.check_cache_against_roster(loaded, table, manifest)
        assert loaded.ids == cache.ids and loaded.sequences == cache.identities
        assert np.array_equal(np.asarray(loaded.features),
                              fne.pool_token_features(cache, pooling).astype(np.float32).astype(np.float64))
        schema = loaded.space.schema
        assert schema["pooling"] == pooling and schema["runtime_sha256"] == cache.runtime_sha256
        assert schema["token_cache_sha256"] == cache.cache_sha256
        assert schema["token_map_sha256"] == cache.token_map_sha256
        assert schema["conditioning_mode"] == "none" and schema["activation"]["kind"] == "final_hidden"
        assert loaded.activation["pooling"].startswith(pooling)
        assert cache.token_map_sha256 in loaded.activation["token_map"]
        assert loaded.extractor == cache.extractor
        assert loaded.space.supervision.status == "unknown", "the supervision history is retained"
        assert loaded.identity["length"] == 15 and "|" in loaded.identity["alphabet"]
        assert loaded.conditioning_summary["quality"]["state"] == "explicit_null"
        assert loaded.conditioning_summary["family"]["state"] == "explicit_null"
        assert loaded.dtype == "float32"
        assert str(Path(csv_path)) in summary["protected_input_paths"]
        assert str(cache.header_path) in summary["protected_input_paths"]
        assert str(cache.npz_path) in summary["protected_input_paths"]
    both = [fm.load_feature_cache(s["header"]).features for s in summaries.values()]
    assert not np.allclose(both[0], both[1], atol=1e-4)


def test_explicit_conditioning_is_recorded_per_row_in_the_pooled_cache(tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(ROSTER, mode="explicit"), batch_size=4)
    _c, _m, table, manifest = write_world(tmp_path / "world")
    summary = pool(cache, table, manifest, tmp_path / "pooled", "explicit_rep")
    loaded = fm.load_feature_cache(summary["header"])
    assert loaded.conditioning["family"]["values"] == [EXPLICIT[i % 6][0] for i in range(8)]
    assert loaded.conditioning["assay"]["values"] == [EXPLICIT[i % 6][1] for i in range(8)]
    assert loaded.conditioning["quality"]["values"] == [None] * 8
    assert loaded.conditioning_summary["assay"]["state"] == "values"
    assert loaded.space.schema["conditioning_mode"] == "explicit"


def test_mutating_the_public_feature_space_cannot_rewrite_the_provenance_the_bridge_records(
        tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(ROSTER, supervision=DOCUMENTED),
                           batch_size=4)
    original = cache.space.document()
    assert original["provenance"]["documented_supervision"][0]["label_count"] == 2
    held = cache.space                                    # a caller keeps a reference and edits it
    held.supervision.documented[0]["label_count"] = 0
    held.supervision.documented[0]["label_ids"].clear()
    held.schema["activation"]["name"] = "route/dplm/block0/attn_out"
    held.schema["hidden_size"] = 999
    cache.space.supervision.documented[0]["label_count"] = 0      # and straight through the property
    cache.space.schema["conditioning_mode"] = "explicit"
    assert held.supervision.documented[0]["label_count"] == 0, "the detached copy really was edited"
    assert cache.space.document() == original, "nothing handed out may reach the cache's provenance"
    assert cache.summary()["feature_supervision"] == original["provenance"]
    assert cache.verify() is True
    _c, _m, table, manifest = write_world(tmp_path / "world")
    summary = pool(cache, table, manifest, tmp_path / "pooled", "p")
    loaded = fm.load_feature_cache(summary["header"])
    assert loaded.space.supervision.document() == original["provenance"]
    assert loaded.space.supervision.documented[0]["label_count"] == 2
    assert loaded.space.supervision.documented[0]["label_ids"] == ["a", "b"]
    assert loaded.space.schema["hidden_size"] == 32
    assert loaded.space.schema["conditioning_mode"] == "none"


def test_an_edit_of_the_private_header_is_refused_by_verification_before_anything_is_pooled(
        tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(ROSTER, supervision=DOCUMENTED),
                           batch_size=4)
    _c, _m, table, manifest = write_world(tmp_path / "world")
    cache._document["feature_space"]["provenance"]["description"] = "rewritten after loading"
    with pytest.raises(ValueError, match="no longer match"):
        cache.verify()
    with pytest.raises(ValueError, match="no longer match"):
        pool(cache, table, manifest, tmp_path / "pooled", "p")
    assert not (tmp_path / "pooled").exists() or listing(tmp_path / "pooled") == []


def test_the_bridge_refuses_each_inexact_roster_and_writes_nothing(tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(ROSTER), batch_size=4)
    identities = identity_rows(ROSTER)
    variants = {}
    extra_row = identities + [("r8", fne.identity_of("EVDCAEGG", "DIQMTQ"), "gE", "train")]
    variants["table_has_a_candidate_the_cache_lacks"] = (extra_row, "lacks")
    variants["cache_has_a_candidate_outside_the_table"] = (identities[:-1], "outside the roster")
    edited = list(identities)
    edited[0] = ("r0", fne.identity_of("EVDCAEGG", "DIQMTQ"), "gA", "train")
    variants["a_candidates_full_identity_differs"] = (edited, "have a different full")
    regrouped = list(identities)
    regrouped[0] = ("r0", identities[0][1], "gZ", "train")
    variants["a_candidates_group_differs"] = (regrouped, "have a different group")
    resplit = [(i, s, g, "eval" if g == "gA" else sp) for i, s, g, sp in identities]
    variants["candidates_split_differs"] = (resplit, "have a different split")
    for name, (listed, message) in variants.items():
        _c, _m, table, manifest = write_world(tmp_path / name, identities=listed)
        with pytest.raises(ValueError, match=message):
            pool(cache, table, manifest, tmp_path / name, "out")
        assert not (tmp_path / name / "out.json").exists() and not (tmp_path / name / "out.npz").exists()
    _c, _m, table, manifest = write_world(tmp_path / "base")
    longer = fm.parse_matrix_manifest(manifest_document(identities, length=16))
    with pytest.raises(ValueError, match="matrix manifest declares"):
        pool(cache, table, longer, tmp_path / "base", "out")
    narrow = fm.parse_matrix_manifest(manifest_document(
        identities, alphabet="".join(sorted(set("".join(i[1] for i in identities)) - {"|"}))))
    with pytest.raises(ValueError, match="does not admit"):
        pool(cache, table, narrow, tmp_path / "base", "out")
    missing_residue = fm.parse_matrix_manifest(manifest_document(
        identities, alphabet="".join(sorted(set("".join(i[1] for i in identities)) - {"Q"}))))
    with pytest.raises(ValueError, match="does not admit"):
        pool(cache, table, missing_residue, tmp_path / "base", "out")
    assert not (tmp_path / "base" / "out.json").exists()


def test_variable_length_identities_are_refused_never_trimmed_or_padded(tmp_path):
    model = fne.tiny_model(seed=1)
    mixed = ROSTER[:3] + (("r3", ROSTER[3][1], "", "gB", "train"),)
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(mixed), batch_size=2)
    assert cache.lengths() == [19, 19, 19, 11]
    _c, _m, table, manifest = write_world(tmp_path / "world", ROSTER[:4])
    with pytest.raises(ValueError, match="different full-identity lengths"):
        pool(cache, table, manifest, tmp_path / "out", "mixed")
    assert not (tmp_path / "out").exists() or listing(tmp_path / "out") == []


def test_pooled_outputs_are_never_overwritten_and_listed_inputs_are_protected(tmp_path):
    model = fne.tiny_model(seed=1)
    _p, _s, cache = export(model, tmp_path / "tok", plan_doc(ROSTER), batch_size=4)
    csv_path, manifest_path, table, manifest = write_world(tmp_path / "world")
    first = pool(cache, table, manifest, tmp_path / "pooled", "a")
    with pytest.raises(ValueError, match="already exists"):
        pool(cache, table, manifest, tmp_path / "pooled", "a")
    kept = Path(first["header"]).read_bytes()
    with pytest.raises(ValueError, match="resolves to the same file"):
        fne.write_pooled_matrix_cache(
            cache, table, manifest, manifest_path, tmp_path / "world" / "m.npz",
            representation_id="m", pooling="all_nonpad_mean",
            protected_input_paths=[csv_path, manifest_path])
    with pytest.raises(ValueError, match="resolves to the same file"):
        fne.write_pooled_matrix_cache(                                # the token cache's own NPZ
            cache, table, manifest, tmp_path / "tok" / "new.json", tmp_path / "tok" / "t.npz",
            representation_id="m", pooling="all_nonpad_mean")
    assert not (tmp_path / "tok" / "new.json").exists()
    assert Path(first["header"]).read_bytes() == kept
    assert manifest_path.read_text(encoding="utf-8").startswith("{")
    with pytest.raises(ValueError, match="pooling must be one of"):
        pool(cache, table, manifest, tmp_path / "pooled", "b", pooling="max")
    with pytest.raises(ValueError, match="needs a TokenCache"):
        fne.pool_token_features("not a cache", "all_nonpad_mean")


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def load_cli():
    spec = importlib.util.spec_from_file_location("functional_native_export_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_cli_inspects_verifies_and_pools_and_reports_refusals_with_status_two(tmp_path, capsys):
    cli = load_cli()
    model = fne.tiny_model(seed=1)
    (tmp_path / "tok").mkdir()
    plan_path = write_plan(tmp_path / "tok", plan_doc(ROSTER))
    plan = fne.read_export_plan(plan_path)
    fne.export_dplm_tokens(model, plan, tmp_path / "tok" / "t.json", tmp_path / "tok" / "t.npz",
                           batch_size=4)
    assert cli.main(["inspect", "--plan", str(plan_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "plan" and summary["rows"] == 8 and summary["eval_rows"] == 4
    assert cli.main(["inspect", "--header", str(tmp_path / "tok" / "t.json")]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "token_cache"
    assert cli.main(["verify", "--header", str(tmp_path / "tok" / "t.json")]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"] is True and verified["rows"] == 8
    csv_path, manifest_path, _t, _m = write_world(tmp_path / "world")
    arguments = ["pool", "--token-header", str(tmp_path / "tok" / "t.json"),
                 "--candidates", str(csv_path), "--manifest", str(manifest_path),
                 "--output-header", str(tmp_path / "p" / "a.json"),
                 "--output-npz", str(tmp_path / "p" / "a.npz"), "--representation-id", "a",
                 "--pooling", "residue_mean"]
    assert cli.main(arguments) == 0
    pooled = json.loads(capsys.readouterr().out)
    assert pooled["pooling"] == "residue_mean" and str(csv_path) in pooled["protected_input_paths"]
    assert fm.load_feature_cache(tmp_path / "p" / "a.json").representation_id == "a"
    assert cli.main(arguments) == 2 and "already exists" in capsys.readouterr().err
    data = bytearray((tmp_path / "tok" / "t.npz").read_bytes())
    data[len(data) // 2] ^= 0x01
    (tmp_path / "tok" / "t.npz").write_bytes(bytes(data))
    assert cli.main(["verify", "--header", str(tmp_path / "tok" / "t.json")]) == 2
    assert "STALE" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["inspect"])


def test_the_cli_selftest_runs_the_real_pipeline_in_a_fresh_directory_and_selects_no_winner(
        tmp_path, capsys):
    cli = load_cli()
    parent = tmp_path / "out"
    parent.mkdir()
    sentinel = parent / "keep_me.txt"
    sentinel.write_text("precious", encoding="utf-8")
    existing = parent / "existing_dir"
    existing.mkdir()
    (existing / "file.txt").write_text("also precious", encoding="utf-8")
    threads = torch.get_num_threads()
    assert cli.main(["selftest", "--output-dir", str(parent)]) == 0
    assert torch.get_num_threads() == threads, "the thread count must be restored"
    assert not torch.cuda.is_initialized()
    assert sentinel.read_text() == "precious" and (existing / "file.txt").read_text() == "also precious"
    created = [p for p in parent.iterdir() if p.name.startswith("native_export_selftest_")]
    assert len(created) == 1
    report = json.loads((created[0] / fne.SELFTEST_REPORT_NAME).read_text(encoding="utf-8"))
    failed = [c["name"] for c in report["checks"] if not c["holds"]]
    assert report["passed"] is True and not failed, failed
    gaps = next(c["detail"] for c in report["checks"]
                if c["name"].startswith("native::batch_size_and_composition"))
    assert sorted(gaps) == ["batch_size_1", "batch_size_40"], (
        "the report is strict JSON, so its keys are strings, and every real number is kept")
    assert all(isinstance(gap, float) and 0.0 <= gap < 1e-5 for gap in gaps.values())
    names = {c["name"] for c in report["checks"]}
    for expected in ("native::final_hidden_equals_an_independent_unpadded_forward_per_row",
                     "native::both_poolings_reuse_the_token_cache_with_no_second_forward",
                     "matrix::the_freeze_was_built_and_verified_before_any_evaluation_outcome_existed"):
        assert expected in names
    assert len(report["cell_results"]) == 4 * 2 * 2 * 2
    assert all(cell["status"] == "ok" and cell["readout_mse"] >= 0.0 for cell in report["cell_results"])
    forbidden = {"winner", "best", "ranking", "selected"}
    assert not any(any(word in key.lower() for word in forbidden) for key in report)
    assert "no cell is selected" in report["note"]
    printed = capsys.readouterr().out
    assert "selftest PASSED" in printed and "readout_mse=" in printed
