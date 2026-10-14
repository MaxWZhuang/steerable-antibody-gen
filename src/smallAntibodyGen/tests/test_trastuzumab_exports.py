"""Synthetic outcome-free export and explicit input-document contracts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import artifacts, inputs
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import (
    QueryPanel, canonical_bytes, digest, generate_attempts, read_json,
    score_queries, write_export,
)
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import select_budget
from smallAntibodyGen.tests.test_dynamic_contracts import fixture as tiny_fixture


def _panel_document(cores=("ACDEFGHIKL", "ACDEFGHIKM", "ACDEFGHIKL")):
    document = {
        "schema": "trastuzumab-queries/1", "protocol_id": "synthetic-protocol",
        "assay_id": "synthetic-assay", "heavy_prefix": "A", "heavy_suffix": "C",
        "light": "MNPQRS", "antigen": "HIK", "queries": [],
    }
    document["queries"] = [dict(case_id=f"case-{i}", core=core, construct_id="")
                           for i, core in enumerate(cores)]
    _rehash_constructs(document)
    return document


def _rehash_constructs(document):
    scaffold = digest({"kind": "scaffold", "segments": [document["heavy_prefix"],
                                                        document["heavy_suffix"], document["light"]]})
    for query in document["queries"]:
        query["construct_id"] = digest({"kind": "construct", "scaffold": scaffold, "core": query["core"]})


def _write(tmp_path, name, document):
    path = tmp_path / name
    path.write_bytes(canonical_bytes(document))
    return path


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy, encoder, *_ = tiny_fixture()
        yield policy, encoder
    finally:
        torch.set_num_threads(previous)


def test_query_wire_roundtrip_and_caller_list_mutation():
    document = _panel_document()
    panel = QueryPanel.from_document(document)
    original = panel.document()
    document["queries"].append(dict(document["queries"][0]))
    assert len(panel.queries) == 3
    assert panel.document() == original
    assert QueryPanel.from_document(json.loads(canonical_bytes(original))) == panel
    assert panel.queries[0].construct_id == panel.queries[2].construct_id


@pytest.mark.parametrize("scope,key", (("panel", "outcome"), ("query", "pKD"),
                                       ("panel", "unknown"), ("query", "role")))
def test_query_wire_rejects_unknown_and_outcome_fields(scope, key):
    document = _panel_document()
    target = document if scope == "panel" else document["queries"][0]
    target[key] = 1.0
    with pytest.raises(ValueError, match="expected exactly"):
        QueryPanel.from_document(document)


@pytest.mark.parametrize("field", ("light", "antigen", "core"))
def test_query_sequences_require_strings_even_when_array_residues_are_canonical(field):
    document = _panel_document()
    target = document["queries"][0] if field == "core" else document
    target[field] = list(target[field])
    _rehash_constructs(document)  # Do not mistake hash mismatch for scalar validation.
    with pytest.raises((TypeError, ValueError)):
        QueryPanel.from_document(document)


@pytest.mark.parametrize("field,value", (("heavy_prefix", ""), ("heavy_suffix", ""),
                                         ("light", "acdefg"), ("antigen", "HIX"),
                                         ("protocol_id", True)))
def test_query_wire_rejects_invalid_scaffold_and_scalar_fields(field, value):
    document = _panel_document()
    document[field] = value
    _rehash_constructs(document)
    with pytest.raises((TypeError, ValueError)):
        QueryPanel.from_document(document)


@pytest.mark.parametrize("problem", ("duplicate_id", "wrong_hash", "short_core", "empty"))
def test_query_identity_and_nonempty_population_are_required(problem):
    document = _panel_document()
    if problem == "duplicate_id":
        document["queries"][1]["case_id"] = document["queries"][0]["case_id"]
    elif problem == "wrong_hash":
        document["queries"][0]["construct_id"] = "0" * 64
    elif problem == "short_core":
        document["queries"][0]["core"] = "ACDE"
        _rehash_constructs(document)
    else:
        document["queries"] = []
    with pytest.raises(ValueError):
        QueryPanel.from_document(document)


@pytest.mark.parametrize("raw", ('{"x":1,"x":2}', '{"outer":{"x":1,"x":2}}',
                                 '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'))
def test_json_reader_rejects_duplicate_keys_and_all_nonfinite_spellings(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_canonical_json_matches_benchmark_signed_zero_and_ascii_convention():
    payload = {"z": (-0.0, "\u03bb"), "a": {"b": True, "n": None}}
    expected = b'{"a":{"b":true,"n":null},"z":[0.0,"\\u03bb"]}'
    assert canonical_bytes(payload) == expected
    assert digest(payload) == hashlib.sha256(expected).hexdigest()


def test_actual_tiny_scores_reuse_duplicate_cores_and_ignore_query_order(model):
    policy, encoder = model
    panel = QueryPanel.from_document(_panel_document())
    policy.train()
    first, meta = score_queries(policy, encoder, panel, checkpoint_identity={"run": "synthetic"},
                               arm="paired", orders=2, order_seed=81)
    assert policy.training
    assert len(encoder.model.calls) == 2 * 2 * 10  # Two unique cores, two complete ten-site orders.
    by_id = {row["case_id"]: row for row in first["scores"]}
    assert by_id["case-0"]["score"] == by_id["case-2"]["score"]
    assert all(row["failure"] is None and isinstance(row["score"], float) for row in first["scores"])
    reverse = replace(panel, queries=tuple(reversed(panel.queries)))
    second, meta2 = score_queries(policy, encoder, reverse, checkpoint_identity={"run": "synthetic"},
                                 arm="paired", orders=2, order_seed=81)
    assert first["scores"] == second["scores"]
    assert meta["readout"] == meta2["readout"]
    assert first["provenance"]["metadata"] == digest(meta)
    for schedule in meta["readout"]["orders"]:
        assert sorted(schedule) == [["H", i] for i in range(1, 11)]
    assert not meta["readout"]["is_affinity"]
    assert not meta["readout"]["is_sampler_likelihood"]


@pytest.mark.parametrize("failure_mode", ("exception", "nonfinite"))
def test_one_failed_order_invalidates_whole_core_and_all_duplicate_queries(model, monkeypatch, failure_mode):
    policy, encoder = model
    panel = QueryPanel.from_document(_panel_document(("A" * 10, "C" * 10, "A" * 10)))
    calls = Counter()

    def scoring(policy, provider, view, targets, order, **kwargs):
        core = "".join(residue for _, residue in targets[0].items)
        calls[core] += 1
        if core == "A" * 10 and calls[core] == 2:
            if failure_mode == "exception":
                raise FloatingPointError("synthetic order failed")
            return torch.tensor([float("nan")])
        return torch.tensor([-float(calls[core])])

    monkeypatch.setattr(artifacts, "whole_core_scores", scoring)
    result, _ = score_queries(policy, encoder, panel, checkpoint_identity={}, arm="paired", orders=3)
    first, healthy, duplicate = result["scores"]
    assert first["score"] is None and first["failure"]
    assert duplicate["score"] is None and duplicate["failure"] == first["failure"]
    assert healthy["score"] == -2.0 and healthy["failure"] is None
    assert calls["A" * 10] == (2 if failure_mode == "exception" else 3)
    assert calls["C" * 10] == 3


def test_structural_model_error_is_fatal_and_restores_training_mode(model, monkeypatch):
    policy, encoder = model
    policy.train()

    def broken(*args, **kwargs):
        raise ValueError("synthetic shape contract")

    monkeypatch.setattr(artifacts, "whole_core_scores", broken)
    with pytest.raises(ValueError, match="shape contract"):
        score_queries(policy, encoder, QueryPanel.from_document(_panel_document()),
                      checkpoint_identity={}, arm="paired", orders=1)
    assert policy.training


@pytest.mark.parametrize("count", (1, 2, 10))
def test_raw_attempts_preserve_zero_change_duplicates_failures_and_edit_positions(model, monkeypatch, count):
    policy, encoder = model
    panel = QueryPanel.from_document(_panel_document())
    reference = "ACDEFGHIKL"
    views = []

    def generating(view, provider, **kwargs):
        views.append(view)
        assert len(view.hidden_sites()) == count
        if len(views) == 2:
            raise FloatingPointError("synthetic generation failed")
        for role, index in view.hidden_sites():
            view = view.reveal((role, index), reference[index - len(panel.heavy_prefix)])
        return view

    monkeypatch.setattr(policy, "generate", generating)
    result, metadata = generate_attempts(
        policy, encoder, panel, checkpoint_identity={}, arm="paired", arm_id="test-arm",
        sampling_seeds=(9, 3), attempts_per_seed=3, steps=2,
        reference_core=reference, editable_count=count,
    )
    assert len(views) == len(result["attempts"]) == 6
    assert result["budgets"] == {"test-arm": {"3": 3, "9": 3}}
    assert [(a["seed"], a["index"]) for a in result["attempts"]] == [(s, i) for s in (3, 9) for i in range(3)]
    assert len({a["attempt_id"] for a in result["attempts"]}) == 6
    failed = result["attempts"][1]
    assert failed["raw_core"] is None and failed["failure_reason"]
    assert failed["source"]["hamming_distance"] is None
    for row, view in zip(result["attempts"], views):
        positions = row["source"]["editable_positions"]
        assert positions == sorted(set(positions)) and len(positions) == count
        assert view.hidden_sites() == tuple(("H", i + 1) for i in positions)
        assert row["source"]["metadata_sha256"] == digest(metadata)
        if row is not failed:
            assert row["raw_core"] == reference and row["failure_reason"] is None
            assert row["source"]["hamming_distance"] == 0
    assert result["attempt_set_sha256"] == digest({k: v for k, v in result.items() if k != "attempt_set_sha256"})


def test_actual_tiny_generation_respects_visible_residues_and_canonical_core(model):
    policy, encoder = model
    panel = QueryPanel.from_document(_panel_document())
    reference = "ACDEFGHIKL"
    result, _ = generate_attempts(policy, encoder, panel, checkpoint_identity={}, arm="paired",
                                  arm_id="actual-tiny", sampling_seeds=(5,), attempts_per_seed=1,
                                  steps=2, reference_core=reference, editable_count=2)
    row = result["attempts"][0]
    assert row["failure_reason"] is None
    assert len(row["raw_core"]) == 10 and set(row["raw_core"]) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert 0 <= row["source"]["hamming_distance"] <= 2
    for i, residue in enumerate(reference):
        if i not in row["source"]["editable_positions"]:
            assert row["raw_core"][i] == residue


@pytest.mark.parametrize("parameter,value", (("orders", True), ("orders", 0),
                                            ("order_seed", True), ("order_seed", -1)))
def test_bad_score_counts_and_seeds_reject_before_model_access(parameter, value):
    with pytest.raises(ValueError):
        score_queries(None, None, QueryPanel.from_document(_panel_document()),
                      checkpoint_identity={}, arm="paired", **{parameter: value})


@pytest.mark.parametrize("parameter,value", (("attempts_per_seed", True), ("steps", True),
                                            ("editable_count", True), ("editable_count", 0),
                                            ("editable_count", 11), ("sampling_seeds", (True,)),
                                            ("sampling_seeds", (3, 3)), ("reference_core", None)))
def test_bad_generation_parameters_reject_before_model_access(parameter, value):
    kwargs = dict(checkpoint_identity={}, arm="paired", arm_id="test", sampling_seeds=(3,),
                  attempts_per_seed=1, steps=2, reference_core="ACDEFGHIKL", editable_count=2)
    kwargs[parameter] = value
    with pytest.raises(ValueError):
        generate_attempts(None, None, QueryPanel.from_document(_panel_document()), **kwargs)


def _broad_document():
    return {
        "schema": "trastuzumab-pairs/1",
        "declaration": {
            "sources": [{"source_id": "source", "description": "synthetic", "priority": 0}],
            "rules": [{"rule_name": "positive", "description": "synthetic declared positive"}],
            "source_priority_order": ["source"], "grouping": {"separation_axes": ["antibody_family"]},
        },
        "records": [{
            "record_id": "r", "source_id": "source", "source_record_id": "row",
            "H": "ACDE", "L": "FG", "Ag": "HIK", "antibody_format": "paired",
            "antibody_family": "ab", "target_family": "target", "partition": "train",
            "h_eligible": [2, 0], "l_eligible": [0], "aux_region": [["H", 2]],
            "evidence_reference": "synthetic", "evidence_assay": "synthetic-positive",
            "evidence_criterion": "declared positive", "qualification_rule": "positive",
        }],
    }


def test_explicit_broad_document_loads_canonical_coordinates_and_file_identity(tmp_path):
    path = _write(tmp_path, "broad.json", _broad_document())
    population, identity = inputs.load_broad(path)
    assert population.records[0].h_eligible == (0, 2)
    assert population.records[0].aux_region == (("H", 2),)
    assert identity["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["population"] == population.population_digest


@pytest.mark.parametrize("scope", ("top", "row", "declaration", "grouping"))
def test_broad_document_rejects_unknown_or_outcome_fields_at_every_level(tmp_path, scope):
    document = _broad_document()
    target = {"top": document, "row": document["records"][0],
              "declaration": document["declaration"], "grouping": document["declaration"]["grouping"]}[scope]
    target["outcome"] = 1.0
    with pytest.raises(ValueError):
        inputs.load_broad(_write(tmp_path, "bad.json", document))


@pytest.mark.parametrize("scope", ("coordinate", "priority", "encoder_limit"))
def test_broad_document_rejects_boolean_numeric_contracts(tmp_path, scope):
    document = _broad_document()
    if scope == "coordinate":
        document["records"][0]["h_eligible"] = [True]
    elif scope == "priority":
        document["declaration"]["sources"][0]["priority"] = True
    else:
        document["declaration"]["encoder_max_tokens"] = True
    with pytest.raises((ValueError, TypeError)):
        inputs.load_broad(_write(tmp_path, "bad.json", document))


def _endpoint_document():
    panel = QueryPanel.from_document(_panel_document(("ACDEFGHIKL", "ACDEFGHIKM", "ACDEFGHIKN")))
    return {"schema": "trastuzumab-endpoints/1", "entries": [
        dict(endpoint_id=f"e{i}", construct=asdict(panel.construct(q.core)),
             role="reserved" if i == 2 else "post_pool", neighbourhood_group=f"g{i}",
             assay_id="SPR", endpoint_semantics="synthetic ordering")
        for i, q in enumerate(panel.queries)
    ]}


def _post_files(tmp_path):
    manifest_path = _write(tmp_path, "manifest.json", _endpoint_document())
    manifest = inputs.load_endpoint_manifest(manifest_path)
    budget = select_budget(manifest, 1, "budget")
    labels = {"schema": "trastuzumab-selected-labels/1", "selection_digest": budget.selection_digest,
              "measurements": [{"endpoint_id": budget.selected_ids[0], "value": 1.0,
                                "orientation": "higher_better", "assay_id": "SPR"}]}
    return manifest_path, labels


def test_explicit_post_document_loads_only_exact_selected_labels(tmp_path):
    manifest_path, labels = _post_files(tmp_path)
    label_path = _write(tmp_path, "labels.json", labels)
    data, identity = inputs.load_post(manifest_path, label_path, budget_n=1, budget_seed="budget")
    assert data.fitting_count == 1
    assert [m.endpoint_id for m in data.measured] == list(data.budget.selected_ids)
    assert identity["selected_labels_sha256"] == hashlib.sha256(label_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("problem", ("extra_reserved", "missing", "bool_value", "unknown", "wrong_digest"))
def test_selected_label_document_fails_closed(tmp_path, problem):
    manifest_path, labels = _post_files(tmp_path)
    if problem == "extra_reserved":
        labels["measurements"].append({**labels["measurements"][0], "endpoint_id": "e2"})
    elif problem == "missing":
        labels["measurements"] = []
    elif problem == "bool_value":
        labels["measurements"][0]["value"] = True
    elif problem == "unknown":
        labels["measurements"][0]["reserved_outcome"] = 2.0
    else:
        labels["selection_digest"] = "0" * 64
    with pytest.raises((ValueError, TypeError)):
        inputs.load_post(manifest_path, _write(tmp_path, "labels.json", labels), budget_n=1, budget_seed="budget")


def test_post_budget_validation_precedes_opening_label_path(tmp_path):
    manifest_path, _ = _post_files(tmp_path)
    with pytest.raises(ValueError, match="exceeds post_pool"):
        inputs.load_post(manifest_path, tmp_path / "must-not-open.json", budget_n=50, budget_seed="budget")


def test_export_completion_manifest_binds_exact_payload_and_metadata_bytes(tmp_path):
    directory = tmp_path / "export"
    metadata = {"checkpoint": {"sha256": "a" * 64}, "queries_sha256": "b" * 64}
    payload = {"scores": [], "provenance": {"metadata": digest(metadata)}}
    manifest = write_export(directory, payload_name="scores.json", payload=payload, metadata=metadata)
    assert set(manifest["files"]) == {"scores.json", "metadata.json"}
    for name, binding in manifest["files"].items():
        contents = (directory / name).read_bytes()
        assert binding == {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
    assert read_json(directory / "manifest.json") == manifest
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        write_export(directory, payload_name="scores.json", payload=payload, metadata=metadata)
    assert original == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("strategy", (
    "reparam-uncond-stochastic.-linear",
    "reparam-uncond-stochastic1..2-linear",
    "reparam-uncond-stochastic" + "9" * 400 + "-linear",
    "reparam-uncond-deterministic-linear\n",
))
def test_malformed_strategy_fails_before_any_inference_preflight(monkeypatch, strategy):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid strategy reached inference")

    monkeypatch.setattr(artifacts, "inference_metadata", forbidden)
    with pytest.raises(ValueError):
        generate_attempts(None, None, QueryPanel.from_document(_panel_document()),
                          checkpoint_identity={}, arm="paired", arm_id="test", sampling_seeds=(3,),
                          attempts_per_seed=1, steps=2, reference_core="ACDEFGHIKL", strategy=strategy)
    assert not calls


@pytest.mark.parametrize("limit", (True, 0, 2, 1024.0, "1024"))
def test_released_encoder_limit_rejected_before_file_or_model_loading(monkeypatch, limit):
    from smallAntibodyGen.experiments.trastuzumab_poc import factory
    from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid limit reached file/model loading")

    monkeypatch.setattr(factory, "read_json", forbidden)
    monkeypatch.setattr(factory, "load_released_encoder", forbidden)
    config = dict(kind="released", variant="dplm_650m", weights_dir="unused", pins_path="unused",
                  encoder_source_root="unused", encoder_source_pins_path="unused",
                  encoder_checkpoint="unused", encoder_checkpoint_sha256="a" * 64,
                  encoder_max_tokens=limit)
    with pytest.raises(ValueError):
        factory.build_models(config, FusionConfig())


def test_selected_labels_bind_literal_construct_manifest_and_ignore_row_order(tmp_path):
    manifest_path, labels = _post_files(tmp_path)
    label_path = _write(tmp_path, "labels.json", labels)
    original, _ = inputs.load_post(manifest_path, label_path, budget_n=1, budget_seed="budget")
    document = _endpoint_document()
    document["entries"].reverse()
    reordered_path = _write(tmp_path, "reordered.json", document)
    reordered, _ = inputs.load_post(reordered_path, label_path, budget_n=1, budget_seed="budget")
    assert reordered.data_digest() == original.data_digest()
    selected_id = original.budget.selected_ids[0]
    selected = next(entry for entry in document["entries"] if entry["endpoint_id"] == selected_id)
    selected["construct"]["core"] = "ACDEFGHIKQ"
    changed_path = _write(tmp_path, "changed.json", document)
    with pytest.raises(ValueError, match="selection digest mismatch"):
        inputs.load_post(changed_path, label_path, budget_n=1, budget_seed="budget")
