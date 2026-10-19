"""Outcome-free functional construct contracts and cross-manifest qualification (M1).

The builders at the top are imported by the protocol and head tests.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from smallAntibodyGen.experiments.trastuzumab_poc import artifacts
from smallAntibodyGen.experiments.trastuzumab_poc import functional_data as fd
from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    GroupingDeclaration, InputRecord, ManifestDeclaration, QualificationRule, SourceDeclaration,
    _content_key, _digest_manifest, qualify_manifest,
)
from smallAntibodyGen.experiments.trastuzumab_poc.sampling import FamilyFirstSampler, SamplerConfig

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
AG = "MKTAYIAKQR"
AG2 = "MKTAYIAKQK"
LIGHT = "DIQMTQSPSS"
KD = ("equilibrium dissociation constant", "nM", "lower_better")
ENRICH = ("sort enrichment", "log2 ratio", "higher_better")


def heavy(index):
    """Distinct canonical 10-residue heavy chains."""
    return "EVQLVESG" + RESIDUES[index] + "G"


# ---------------------------------------------------------------------------
# World builders
# ---------------------------------------------------------------------------

def declaration(**overrides):
    values = dict(sources=(SourceDeclaration("broad", "broad source", 0),),
                  rules=(QualificationRule("rule", "positive evidence"),),
                  source_priority_order=("broad",),
                  grouping=GroupingDeclaration(separation_axes=("antibody_family",)))
    values.update(overrides)
    return ManifestDeclaration(**values)


def broad(record_id, H, *, L=LIGHT, Ag=AG, family="famF", target="her2", partition="train",
          source="broad", source_record_id=None, assay="assay"):
    return InputRecord(
        record_id=record_id, source_id=source, source_record_id=source_record_id or record_id,
        H=H, L=L, Ag=Ag, antibody_format="paired" if L is not None else "heavy_only",
        antibody_family=family, target_family=target, partition=partition,
        h_eligible=(0, 1, 2), l_eligible=(0, 1) if L is not None else None, aux_region=None,
        evidence_reference="ref", evidence_assay=assay, evidence_criterion="criterion",
        qualification_rule="rule")


def broad_records():
    """b0 shares its content with functional t0 (consistent train/train overlap)."""
    return (broad("b0", heavy(0)),
            broad("b1", heavy(10)),
            broad("b2", heavy(11), family="famB"),
            broad("b3", heavy(12), family="famW", partition="val"))


def construct_row(construct_id, H, *, L=LIGHT, Ag=AG, family="famF", target="her2",
                  partition="train", source="func", source_record_id=None):
    return {"construct_id": construct_id, "source_id": source,
            "source_record_id": source_record_id or f"rec-{construct_id}", "H": H, "L": L, "Ag": Ag,
            "antibody_format": "paired" if L is not None else "heavy_only",
            "antibody_family": family, "target_family": target, "partition": partition,
            "evidence_reference": "doi:10.0/functional"}


def group_row(group_id, context_id, endpoint, members, *, Ag=AG):
    return {"group_id": group_id, "context_id": context_id,
            "endpoint": {"semantics": endpoint[0], "units": endpoint[1], "direction": endpoint[2]},
            "antigen_sha256": hashlib.sha256(Ag.encode("ascii")).hexdigest(),
            "member_construct_ids": list(members), "evidence_reference": "doi:10.0/group"}


def contract_block():
    return {"target_property": {"property_id": "binding-strength",
                                "description": "ordering of binding strength by one scalar head"},
            "condition_scope": {"scope_id": "scope-1", "description": "two assay contexts",
                                "context_ids": ["ctx-a", "ctx-b"]},
            "compatible_endpoints": [
                {"semantics": KD[0], "units": KD[1], "direction": KD[2],
                 "justification": "KD orders binding strength; lower is stronger"},
                {"semantics": ENRICH[0], "units": ENRICH[1], "direction": ENRICH[2],
                 "justification": "enrichment is asserted monotone in binding strength"}],
            "assertion": "curator asserts these endpoints order one property in these contexts"}


def standard_document(decl):
    """Train: t0..t5 in g-a (4), g-b (3, repeats t0 and t5), g-single (t5). Val: v0, v1 in g-val."""
    return {
        "schema": fd.CONSTRUCTS_SCHEMA,
        "pairs_declaration_digest": _digest_manifest(decl),
        "study_contract": contract_block(),
        "sources": [{"source_id": "func", "description": "functional measurement source"}],
        "constructs": [construct_row(f"t{i}", heavy(i)) for i in range(6)] + [
            construct_row("v0", heavy(6), family="famV", partition="val"),
            construct_row("v1", heavy(7), family="famV", partition="val")],
        "groups": [group_row("g-a", "ctx-a", KD, ["t0", "t1", "t2", "t3"]),
                   group_row("g-b", "ctx-b", ENRICH, ["t0", "t4", "t5"]),
                   group_row("g-single", "ctx-a", KD, ["t5"]),
                   group_row("g-val", "ctx-a", KD, ["v0", "v1"])]}


def standard_world():
    decl = declaration()
    return decl, broad_records(), standard_document(decl)


def qualify(doc, decl, records, **kwargs):
    return fd.qualify_functional(fd.parse_functional_constructs(doc), decl, records, **kwargs)


def standard_qualified():
    decl, records, doc = standard_world()
    return qualify(doc, decl, records)


def imported_modules(module):
    """Every module an import statement of ``module`` names (statically)."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def imported_names(module):
    """Like :func:`imported_modules`, plus ``module.name`` for every ``from module import name``."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(f"{node.module or ''}.{alias.name}" for alias in node.names)
    return names


def _group(doc, group_id):
    return next(g for g in doc["groups"] if g["group_id"] == group_id)


def _construct(doc, construct_id):
    return next(c for c in doc["constructs"] if c["construct_id"] == construct_id)


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    {"a": [1, 2.5, -0.0], "b": {"z": None, "y": "é"}}, [], {"x": (1, 2)}, {"n": 0.0, "t": True}])
def test_canonical_json_is_byte_compatible_with_the_torch_side_helpers(value):
    assert fd.canonical_bytes(value) == artifacts.canonical_bytes(value)
    assert fd.json_digest(value) == artifacts.digest(value)


@pytest.mark.parametrize("value, error", [({"x": float("nan")}, ValueError),
                                          ({"x": float("inf")}, ValueError),
                                          ({1: "a"}, TypeError), ({"x": {1, 2}}, TypeError)])
def test_canonical_json_rejects_nonfinite_and_unsupported_values(value, error):
    with pytest.raises(error):
        fd.canonical_bytes(value)


def test_strict_json_reader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    path = tmp_path / "doc.json"
    for text, match in (('{"a": 1, "a": 2}', "duplicate JSON key"), ('{"a": NaN}', "nonfinite JSON"),
                        ('{"a": Infinity}', "nonfinite JSON"), ('{"a": 1e999}', "finite")):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            fd.read_strict_json(path)
    path.write_text('﻿{"a": [1, 2]}', encoding="utf-8")
    assert fd.read_strict_json(path) == {"a": [1, 2]}


def test_modules_do_not_import_torch_or_the_torch_side_campaign_modules():
    from smallAntibodyGen.experiments.trastuzumab_poc import functional_protocol
    for module, allowed in (
            (fd, {"smallAntibodyGen.experiments.trastuzumab_poc.data",
                  "smallAntibodyGen.models.antibody_antigen.views"}),
            (functional_protocol, {"smallAntibodyGen.experiments.trastuzumab_poc.functional_data"})):
        names = imported_modules(module)
        assert not any(n == "torch" or n.startswith("torch.") for n in names)
        internal = {n for n in names if n.startswith("smallAntibodyGen")}
        assert internal <= allowed, internal - allowed


# ---------------------------------------------------------------------------
# Strict schema
# ---------------------------------------------------------------------------

def _duplicate_physical_construct(doc):
    clone = copy.deepcopy(_construct(doc, "t0"))
    clone.update(construct_id="t0-copy", source_record_id="rec-t0-copy")
    doc["constructs"].append(clone)
    _group(doc, "g-a")["member_construct_ids"].append("t0-copy")


MUTATIONS = {
    "unknown_top_level_key": lambda d: d.update(outcome=1.0),
    "missing_top_level_key": lambda d: d.pop("groups"),
    "wrong_schema": lambda d: d.update(schema="trastuzumab-functional-constructs/2"),
    "declaration_digest_not_hex": lambda d: d.update(pairs_declaration_digest="not-a-digest"),
    "declaration_digest_uppercase": lambda d: d.update(pairs_declaration_digest="A" * 64),
    "construct_unknown_key": lambda d: _construct(d, "t0").update(outcome=1.0),
    "construct_missing_key": lambda d: _construct(d, "t0").pop("evidence_reference"),
    "constructs_not_a_list": lambda d: d.update(constructs={"t0": 1}),
    "construct_row_not_an_object": lambda d: d["constructs"].append("t9"),
    "constructs_empty": lambda d: d.update(constructs=[]),
    "groups_empty": lambda d: d.update(groups=[]),
    "sources_empty": lambda d: d.update(sources=[]),
    "duplicate_group_id": lambda d: d["groups"].append(copy.deepcopy(d["groups"][0])),
    "duplicate_source_id": lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])),
    "undeclared_source": lambda d: _construct(d, "t0").update(source_id="ghost"),
    "shared_source_record": lambda d: _construct(d, "t1").update(source_record_id="rec-t0"),
    "same_physical_construct_under_two_ids": _duplicate_physical_construct,
    "group_duplicate_member": lambda d: _group(d, "g-a")["member_construct_ids"].append("t0"),
    "group_unknown_member": lambda d: _group(d, "g-a")["member_construct_ids"].append("ghost"),
    "group_without_members": lambda d: _group(d, "g-single").update(member_construct_ids=[]),
    "group_members_not_a_list": lambda d: _group(d, "g-a").update(member_construct_ids="t0"),
    "group_context_outside_scope": lambda d: _group(d, "g-a").update(context_id="ctx-z"),
    "group_endpoint_units_not_allowlisted": lambda d: _group(d, "g-a")["endpoint"].update(units="pM"),
    "group_endpoint_direction_not_allowlisted":
        lambda d: _group(d, "g-a")["endpoint"].update(direction="higher_better"),
    "group_endpoint_unknown_key": lambda d: _group(d, "g-a")["endpoint"].update(scale="log"),
    "group_antigen_digest_mismatch": lambda d: _group(d, "g-a").update(antigen_sha256="0" * 64),
    "group_antigen_digest_malformed": lambda d: _group(d, "g-a").update(antigen_sha256="abc"),
    "group_member_antigen_differs": lambda d: _construct(d, "t1").update(Ag=AG2),
    "group_spans_partitions": lambda d: _construct(d, "t2").update(partition="val"),
    "group_missing_provenance": lambda d: _group(d, "g-a").update(evidence_reference=" "),
    "contract_direction_invalid":
        lambda d: d["study_contract"]["compatible_endpoints"][0].update(direction="up"),
    "contract_empty_justification":
        lambda d: d["study_contract"]["compatible_endpoints"][0].update(justification=""),
    "contract_duplicate_endpoint": lambda d: d["study_contract"]["compatible_endpoints"].append(
        copy.deepcopy(d["study_contract"]["compatible_endpoints"][0])),
    "contract_no_endpoints": lambda d: d["study_contract"].update(compatible_endpoints=[]),
    "contract_duplicate_context":
        lambda d: d["study_contract"]["condition_scope"]["context_ids"].append("ctx-a"),
    "contract_no_contexts": lambda d: d["study_contract"]["condition_scope"].update(context_ids=[]),
    "contract_empty_assertion": lambda d: d["study_contract"].update(assertion=""),
    "contract_unknown_key": lambda d: d["study_contract"].update(extra=1),
    "noncanonical_residue": lambda d: _construct(d, "t0").update(H="EVQLVESGXG"),
    "lowercase_residue": lambda d: _construct(d, "t0").update(H="evqlvesgag"),
    "empty_heavy": lambda d: _construct(d, "t0").update(H=""),
    "paired_without_light": lambda d: _construct(d, "t0").update(L=None),
    "heavy_only_with_light": lambda d: _construct(d, "t0").update(antibody_format="heavy_only"),
    "unknown_format": lambda d: _construct(d, "t0").update(antibody_format="fab"),
    "unknown_partition": lambda d: _construct(d, "t0").update(partition="holdout"),
    "missing_provenance": lambda d: _construct(d, "t0").update(evidence_reference=""),
    "unreferenced_construct": lambda d: d["constructs"].append(construct_row("t-extra", heavy(9))),
    "id_with_nul": lambda d: _construct(d, "t0").update(construct_id="t0\x00"),
    "id_with_space": lambda d: _construct(d, "t0").update(construct_id=" t0"),
    "id_not_text": lambda d: _construct(d, "t0").update(construct_id=True),
    "source_record_not_text": lambda d: _construct(d, "t0").update(source_record_id=7),
    "target_family_conflict_for_same_antigen":
        lambda d: _construct(d, "t1").update(target_family="other"),
}


def test_the_unmutated_standard_document_is_valid():
    decl, records, doc = standard_world()
    assert fd.parse_functional_constructs(doc).document() == fd.parse_functional_constructs(
        fd.parse_functional_constructs(doc).document()).document()
    qualify(doc, decl, records)


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_schema_rejects_malformed_or_contradictory_documents(name):
    _, _, doc = standard_world()
    MUTATIONS[name](doc)
    with pytest.raises(ValueError):
        fd.parse_functional_constructs(doc)


def test_same_antibody_with_a_conflicting_family_is_rejected():
    _, _, doc = standard_world()
    doc["constructs"].append(construct_row("t-fam", heavy(0), Ag=AG2, family="famZ"))
    doc["groups"].append(group_row("g-fam", "ctx-a", KD, ["t-fam"], Ag=AG2))
    with pytest.raises(ValueError, match="conflicting antibody families"):
        fd.parse_functional_constructs(doc)


def test_parse_round_trip_and_row_order_independence():
    decl, records, doc = standard_world()
    parsed = fd.parse_functional_constructs(doc)
    assert fd.parse_functional_constructs(parsed.document()) == parsed
    shuffled = copy.deepcopy(doc)
    shuffled["constructs"].reverse()
    shuffled["groups"].reverse()
    shuffled["sources"].reverse()
    for group in shuffled["groups"]:
        group["member_construct_ids"].reverse()
    shuffled["study_contract"]["condition_scope"]["context_ids"].reverse()
    shuffled["study_contract"]["compatible_endpoints"].reverse()
    other = fd.parse_functional_constructs(shuffled)
    assert other == parsed
    assert other.document_digest() == parsed.document_digest()
    assert other.contract_digest() == parsed.contract_digest()
    first = fd.qualify_functional(parsed, decl, records)
    second = fd.qualify_functional(other, decl, tuple(reversed(records)))
    assert first.digest == second.digest and first.broad_input_digest == second.broad_input_digest


def test_load_functional_constructs_reads_a_strict_file(tmp_path):
    _, _, doc = standard_world()
    path = tmp_path / "functional.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert fd.load_functional_constructs(path) == fd.parse_functional_constructs(doc)
    path.write_text(json.dumps(doc)[:-1] + ', "schema": "again"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        fd.load_functional_constructs(path)


# ---------------------------------------------------------------------------
# Qualification against the broad manifest
# ---------------------------------------------------------------------------

def test_qualification_reports_observations_and_physical_constructs_separately():
    q = standard_qualified()
    assert [c.construct_id for c in q.train_constructs()] == [f"t{i}" for i in range(6)]
    assert {g.group_id for g in q.train_groups()} == {"g-a", "g-b", "g-single"}
    observations = q.train_observations()
    assert len(observations) == len(set(observations)) == 8
    assert all(group != "g-val" for group, _ in observations)
    report = q.report()
    assert report["unique_observations"] == 10
    assert report["unique_construct_ids"] == report["unique_physical_constructs"] == 8
    assert report["unique_antibodies"] == 8 and report["unique_antigens"] == 1
    assert report["by_partition"] == {"train": {"constructs": 6, "groups": 3, "observations": 8},
                                      "val": {"constructs": 2, "groups": 1, "observations": 2},
                                      "test": {"constructs": 0, "groups": 0, "observations": 0}}
    assert [q.direction(g) for g in ("g-a", "g-b")] == ["lower_better", "higher_better"]
    json.dumps(report)


def test_consistent_train_overlap_is_allowed_and_the_denoising_population_is_untouched():
    decl, records, doc = standard_world()
    population = qualify_manifest(decl, records)
    before = [tuple(i.record_id for i in FamilyFirstSampler(
        population, SamplerConfig(2, "family_first_v1", 11)).draw_batch(u).paired_items)
        for u in range(6)]
    q = qualify(doc, decl, records, population=population)
    after = [tuple(i.record_id for i in FamilyFirstSampler(
        population, SamplerConfig(2, "family_first_v1", 11)).draw_batch(u).paired_items)
        for u in range(6)]
    assert q.shared_with_denoising_train == ("t0",)
    assert before == after and {rid for batch in before for rid in batch} <= {"b0", "b1", "b2"}
    assert q.population_digest == population.population_digest
    assert [r.record_id for r in population.train_records()] == ["b0", "b1", "b2"]


def test_the_document_must_name_the_supplied_declaration():
    decl, records, doc = standard_world()
    doc["pairs_declaration_digest"] = _digest_manifest(declaration(excluded_antibody_families=("none",)))
    with pytest.raises(ValueError, match="pairs_declaration_digest"):
        qualify(doc, decl, records)


def test_a_supplied_population_must_be_the_qualification_of_the_broad_records():
    decl, records, doc = standard_world()
    good = qualify_manifest(decl, records)
    assert qualify(doc, decl, records, population=good).population_digest == good.population_digest
    with pytest.raises(ValueError, match="supplied population"):
        qualify(doc, decl, records, population=qualify_manifest(decl, records[:3]))
    with pytest.raises(TypeError):
        qualify(doc, decl, records, population="population")
    with pytest.raises(TypeError):
        qualify(doc, decl, ["not a record"])


@pytest.mark.parametrize("overrides, match", [
    (dict(excluded_content_keys=(_content_key(heavy(1), LIGHT, AG),)), "excluded content key"),
    (dict(excluded_antibody_families=("famF",)), "excluded antibody family"),
    (dict(excluded_target_families=("her2",)), "excluded target family"),
    (dict(encoder_max_tokens=20), "joint encoder limit"),
    (dict(native_max_tokens=22), "native token limit"),
])
def test_declaration_exclusions_and_limits_apply_to_functional_constructs_globally(overrides, match):
    decl = declaration(**overrides)
    with pytest.raises(ValueError, match=match):
        qualify(standard_document(decl), decl, broad_records())


def test_an_overlong_antigen_is_rejected_for_the_paired_view_that_both_arms_must_admit():
    decl, records, doc = standard_world()
    long_ag = "A" * 1100
    doc["constructs"].append(construct_row("t-long", heavy(8), Ag=long_ag))
    doc["groups"].append(group_row("g-long", "ctx-a", KD, ["t-long"], Ag=long_ag))
    with pytest.raises(ValueError, match="joint encoder limit"):
        qualify(doc, decl, records)


def test_shared_content_must_agree_even_when_the_broad_row_was_excluded():
    decl = declaration(excluded_record_ids=("b-val",))
    records = (broad("b-val", heavy(0), family="famF", partition="val"),
               broad("b1", heavy(10), family="famQ"))
    population = qualify_manifest(decl, records)
    assert {r.record_id for r in population.records} == {"b1"}
    assert [e.record_id for e in population.excluded] == ["b-val"]
    with pytest.raises(ValueError, match=r"disagrees with the broad manifest.*partition"):
        qualify(standard_document(decl), decl, records)


def test_a_family_on_a_separation_axis_cannot_span_partitions_through_an_excluded_row():
    decl = declaration(excluded_record_ids=("b-val",))
    records = (broad("b-val", heavy(13), family="famF", partition="val"),
               broad("b1", heavy(10), family="famQ"))
    with pytest.raises(ValueError, match="spans partitions"):
        qualify(standard_document(decl), decl, records)


def test_every_declared_separation_axis_is_checked_across_both_documents():
    decl = declaration(grouping=GroupingDeclaration(separation_axes=("target_family",)))
    records = (broad("b0", heavy(0)), broad("b1", heavy(10)),
               broad("b3", heavy(12), Ag=AG2, target="egfr", partition="val"))
    with pytest.raises(ValueError, match=r"target_family.*spans partitions"):
        qualify(standard_document(decl), decl, records)


def test_conflicting_families_for_the_same_sequences_are_rejected_across_documents():
    decl = declaration()
    records = (broad("bx", heavy(0), Ag=AG2, family="famOther"), broad("b1", heavy(10)))
    with pytest.raises(ValueError, match="antibody family conflicts"):
        qualify(standard_document(decl), decl, records)
    with pytest.raises(ValueError, match="target family conflicts"):
        qualify(standard_document(decl), decl, (broad("bx", heavy(14), target="egfr"),))


def test_construct_ids_and_broad_record_ids_are_different_namespaces():
    decl = declaration(excluded_record_ids=("t4",))
    records = broad_records() + (broad("t4", heavy(15), family="famB"),)
    q = qualify(standard_document(decl), decl, records)
    assert "t4" in {c.construct_id for c in q.train_constructs()}
    assert [e.record_id for e in qualify_manifest(decl, records).excluded] == ["t4"]


def test_source_ids_are_scoped_to_the_functional_document_unless_declared_in_the_broad_manifest():
    decl, records, doc = standard_world()
    # Same ID, same description: a shared source.
    shared = copy.deepcopy(doc)
    shared["sources"].append({"source_id": "broad", "description": "broad source"})
    qualify(shared, decl, records)
    # Same ID, different description: one ID cannot name two sources.
    renamed = copy.deepcopy(doc)
    renamed["sources"].append({"source_id": "broad", "description": "something else"})
    with pytest.raises(ValueError, match="different description"):
        qualify(renamed, decl, records)
    # Reusing a broad (source, record) identity for different content is a contradiction ...
    clash = copy.deepcopy(shared)
    _construct(clash, "t1").update(source_id="broad", source_record_id="b1")
    with pytest.raises(ValueError, match="broad source record identity"):
        qualify(clash, decl, records)
    # ... for the same content it is consistent ...
    same = copy.deepcopy(shared)
    _construct(same, "t0").update(source_id="broad", source_record_id="b0")
    qualify(same, decl, records)
    # ... and an undeclared functional source may reuse any record string.
    scoped = copy.deepcopy(doc)
    _construct(scoped, "t1").update(source_record_id="b1")
    qualify(scoped, decl, records)


def test_the_identity_binds_excluded_broad_rows_that_the_population_forgets():
    decl = declaration(excluded_record_ids=("b-ex",))
    doc = standard_document(decl)
    first = qualify(doc, decl, broad_records() + (broad("b-ex", heavy(16), family="famE", assay="a1"),))
    second = qualify(doc, decl, broad_records() + (broad("b-ex", heavy(16), family="famE", assay="a2"),))
    assert first.population_digest == second.population_digest
    assert first.broad_input_digest != second.broad_input_digest and first.digest != second.digest


def test_contract_digest_binds_only_the_study_contract():
    decl, records, doc = standard_world()
    base = qualify(doc, decl, records)
    claim = copy.deepcopy(doc)
    claim["study_contract"]["assertion"] = "a different curator claim"
    changed = qualify(claim, decl, records)
    assert changed.contract_digest != base.contract_digest and changed.digest != base.digest
    evidence = copy.deepcopy(doc)
    evidence["groups"][0]["evidence_reference"] = "doi:10.0/other"
    other = qualify(evidence, decl, records)
    assert other.contract_digest == base.contract_digest and other.digest != base.digest
