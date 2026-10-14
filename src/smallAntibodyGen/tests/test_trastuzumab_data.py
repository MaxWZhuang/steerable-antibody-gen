"""Tests for trastuzumab_poc data qualification layer."""
from __future__ import annotations

import json
from dataclasses import replace
from itertools import permutations

import pytest

from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    InputRecord, SourceDeclaration, QualificationRule,
    GroupingDeclaration, ManifestDeclaration,
    qualify_manifest, QualifiedPopulation, ExcludedRecord,
    _content_key,
    EXCL_EXPLICIT_RECORD, EXCL_CONTENT, EXCL_ANTIBODY_FAMILY,
    EXCL_TARGET_FAMILY, EXCL_OVERLENGTH_JOINT, EXCL_OVERLENGTH_PAIRED,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _src(sid="s1", priority=0):
    return SourceDeclaration(source_id=sid, description="test source", priority=priority)


def _rule(name="rule_A"):
    return QualificationRule(rule_name=name, description="positive binder")


def _decl(sources=None, rules=None, extra_excl_records=(), extra_excl_content=(),
          extra_excl_ab_fam=(), extra_excl_tgt_fam=(), axes=("antibody_family",),
          linked_groups=None):
    sources = sources or [_src("s1", 0), _src("s2", 1)]
    rules = rules or [_rule("rule_A")]
    src_ids = tuple(s.source_id for s in sources)
    return ManifestDeclaration(
        sources=tuple(sources),
        rules=tuple(rules),
        source_priority_order=src_ids,
        grouping=GroupingDeclaration(
            separation_axes=tuple(axes),
            linked_groups=tuple(frozenset(g) for g in linked_groups) if linked_groups else None,
        ),
        excluded_record_ids=tuple(extra_excl_records),
        excluded_content_keys=tuple(extra_excl_content),
        excluded_antibody_families=tuple(extra_excl_ab_fam),
        excluded_target_families=tuple(extra_excl_tgt_fam),
    )


def _rec(record_id="r1", source_id="s1", H="ACDEFGHIKL", L="MNPQRSTVWY", Ag="ACDE",
         ab_format="paired", ab_family="fam1", tgt_family="her2",
         partition="train", h_elig=None, l_elig=None, aux=None,
         evidence_ref="PDB:1ABC", assay="SPR", criterion="KD<10nM",
         rule="rule_A", source_record_id=None):
    h_elig = h_elig if h_elig is not None else (0, 1, 2)
    l_elig = l_elig if l_elig is not None else (0, 1)
    return InputRecord(
        record_id=record_id,
        source_id=source_id,
        source_record_id=source_record_id or record_id,
        H=H,
        L=L if ab_format == "paired" else None,
        Ag=Ag,
        antibody_format=ab_format,
        antibody_family=ab_family,
        target_family=tgt_family,
        partition=partition,
        h_eligible=tuple(h_elig),
        l_eligible=tuple(l_elig) if ab_format == "paired" and l_elig is not None else None,
        aux_region=aux,
        evidence_reference=evidence_ref,
        evidence_assay=assay,
        evidence_criterion=criterion,
        qualification_rule=rule,
    )


# ---------------------------------------------------------------------------
# InputRecord validation
# ---------------------------------------------------------------------------

def test_input_record_valid_paired():
    r = _rec()
    assert r.H == "ACDEFGHIKL"
    assert r.L is not None


def test_input_record_heavy_only_no_l():
    r = _rec(ab_format="heavy_only", L=None, l_elig=None)
    assert r.L is None
    assert r.l_eligible is None


def test_input_record_paired_requires_L():
    with pytest.raises(ValueError, match="paired"):
        InputRecord(
            record_id="r", source_id="s", source_record_id="s",
            H="ACDE", L=None, Ag="ACDE",
            antibody_format="paired", antibody_family="f", target_family="t",
            partition="train", h_eligible=(0,), l_eligible=None,
            aux_region=None, evidence_reference="X", evidence_assay="SPR",
            evidence_criterion="C", qualification_rule="rule_A",
        )


def test_input_record_noncanonical_residue():
    with pytest.raises(ValueError, match="non-canonical"):
        _rec(H="ACDEFGHIKB")  # B is not canonical


def test_input_record_empty_h_eligible():
    with pytest.raises(ValueError, match="nonempty"):
        _rec(h_elig=())


def test_input_record_out_of_bounds_h_eligible():
    with pytest.raises(ValueError, match="out of bounds"):
        _rec(H="AC", h_elig=(0, 99))


def test_input_record_duplicate_h_eligible():
    with pytest.raises(ValueError, match="duplicate"):
        _rec(H="ACDE", h_elig=(0, 0, 1))


def test_input_record_unknown_partition():
    with pytest.raises(ValueError, match="partition"):
        _rec(partition="unknown")


# ---------------------------------------------------------------------------
# ManifestDeclaration validation
# ---------------------------------------------------------------------------

def test_manifest_duplicate_source_ids():
    with pytest.raises(ValueError, match="Duplicate source_ids"):
        ManifestDeclaration(
            sources=(_src("s1"), _src("s1")),
            rules=(_rule(),),
            source_priority_order=("s1",),
            grouping=GroupingDeclaration(separation_axes=("antibody_family",)),
        )


def test_manifest_priority_order_mismatch():
    with pytest.raises(ValueError, match="source_priority_order"):
        ManifestDeclaration(
            sources=(_src("s1"), _src("s2")),
            rules=(_rule(),),
            source_priority_order=("s1",),  # missing s2
            grouping=GroupingDeclaration(separation_axes=("antibody_family",)),
        )


# ---------------------------------------------------------------------------
# Basic qualification pass
# ---------------------------------------------------------------------------

def test_qualify_single_record():
    decl = _decl()
    records = [_rec()]
    pop = qualify_manifest(decl, records)
    assert len(pop.records) == 1
    assert pop.records[0].record_id == "r1"
    assert pop.records[0].partition == "train"
    assert pop.coverage.input_rows == 1
    assert pop.coverage.unique_pairs == 1


# ---------------------------------------------------------------------------
# Duplicate provenance does not multiply items
# ---------------------------------------------------------------------------

def test_duplicate_provenance_single_qualified_item():
    """Two records with identical H/L/Ag (same content key) -> one qualified item."""
    decl = _decl()
    r1 = _rec(record_id="r1", source_id="s1", source_record_id="srcA")
    # Same content, different source, same family/partition
    r2 = InputRecord(
        record_id="r2",
        source_id="s2",
        source_record_id="srcB",
        H=r1.H, L=r1.L, Ag=r1.Ag,
        antibody_format=r1.antibody_format,
        antibody_family=r1.antibody_family,
        target_family=r1.target_family,
        partition=r1.partition,
        h_eligible=r1.h_eligible,
        l_eligible=r1.l_eligible,
        aux_region=None,
        evidence_reference=r1.evidence_reference,
        evidence_assay=r1.evidence_assay,
        evidence_criterion=r1.evidence_criterion,
        qualification_rule=r1.qualification_rule,
    )
    pop = qualify_manifest(decl, [r1, r2])
    assert len(pop.records) == 1
    # Owning source is s1 (lower priority index)
    assert pop.records[0].owning_source_id == "s1"
    refs = [json.loads(ref) for ref in pop.records[0].supporting_refs]
    assert {(ref["source"], ref["record"]) for ref in refs} == {
        ("s1", "srcA"), ("s2", "srcB"),
    }
    assert all(ref == {
        "source": source, "record": source_record, "reference": "PDB:1ABC",
        "assay": "SPR", "criterion": "KD<10nM", "rule": "rule_A",
    } for ref, source, source_record in zip(refs, ("s1", "s2"), ("srcA", "srcB")))


# ---------------------------------------------------------------------------
# Source-record conflicts
# ---------------------------------------------------------------------------

def test_conflict_cross_partition():
    """A conflicting declaration must fail before either row can disappear."""
    decl = _decl()
    r1 = _rec(record_id="r1", partition="train")
    r2 = InputRecord(
        record_id="r2",
        source_id="s1",
        source_record_id="s2",
        H=r1.H, L=r1.L, Ag=r1.Ag,
        antibody_format=r1.antibody_format,
        antibody_family=r1.antibody_family,
        target_family=r1.target_family,
        partition="val",   # CONFLICT
        h_eligible=r1.h_eligible,
        l_eligible=r1.l_eligible,
        aux_region=None,
        evidence_reference=r1.evidence_reference,
        evidence_assay=r1.evidence_assay,
        evidence_criterion=r1.evidence_criterion,
        qualification_rule=r1.qualification_rule,
    )
    with pytest.raises(ValueError, match="separation|conflicting"):
        qualify_manifest(decl, [r1, r2])


def test_conflict_cross_antibody_family():
    decl = _decl()
    r1 = _rec(record_id="r1", ab_family="famA")
    r2 = InputRecord(
        record_id="r2",
        source_id="s1",
        source_record_id="s2",
        H=r1.H, L=r1.L, Ag=r1.Ag,
        antibody_format=r1.antibody_format,
        antibody_family="famB",   # CONFLICT
        target_family=r1.target_family,
        partition=r1.partition,
        h_eligible=r1.h_eligible,
        l_eligible=r1.l_eligible,
        aux_region=None,
        evidence_reference=r1.evidence_reference,
        evidence_assay=r1.evidence_assay,
        evidence_criterion=r1.evidence_criterion,
        qualification_rule=r1.qualification_rule,
    )
    with pytest.raises(ValueError, match="conflicting"):
        qualify_manifest(decl, [r1, r2])


def test_distinct_positive_assays_preserve_provenance_without_multiplying_items():
    """Positive evidence from two assays is not a comparison of their values."""
    decl = _decl()
    r1 = _rec(record_id="r1", assay="SPR")
    r2 = InputRecord(
        record_id="r2",
        source_id="s2",
        source_record_id="s2",
        H=r1.H, L=r1.L, Ag=r1.Ag,
        antibody_format=r1.antibody_format,
        antibody_family=r1.antibody_family,
        target_family=r1.target_family,
        partition=r1.partition,
        h_eligible=r1.h_eligible,
        l_eligible=r1.l_eligible,
        aux_region=None,
        evidence_reference=r1.evidence_reference,
        evidence_assay="ITC",   # different assay, same content
        evidence_criterion=r1.evidence_criterion,
        qualification_rule=r1.qualification_rule,
    )
    pop = qualify_manifest(decl, [r1, r2])
    assert len(pop.records) == 1
    refs = [json.loads(ref) for ref in pop.records[0].supporting_refs]
    assert {ref["assay"] for ref in refs} == {"SPR", "ITC"}
    assert {ref["source"] for ref in refs} == {"s1", "s2"}


# ---------------------------------------------------------------------------
# Overlength excluded from BOTH arms
# ---------------------------------------------------------------------------

def test_overlength_excluded_from_both_arms():
    """A record that exceeds the joint ESM limit is excluded."""
    decl = _decl()
    # 600 + 200 + 300 + 2 = 1102 > 1024
    long_H = "A" * 600
    long_L = "C" * 200
    long_Ag = "D" * 300
    r = InputRecord(
        record_id="r_long",
        source_id="s1",
        source_record_id="s1",
        H=long_H, L=long_L, Ag=long_Ag,
        antibody_format="paired",
        antibody_family="fam1",
        target_family="her2",
        partition="train",
        h_eligible=(0,),
        l_eligible=(0,),
        aux_region=None,
        evidence_reference="X",
        evidence_assay="SPR",
        evidence_criterion="C",
        qualification_rule="rule_A",
    )
    pop = qualify_manifest(decl, [r])
    assert len(pop.records) == 0
    reasons = {e.reason for e in pop.excluded}
    assert EXCL_OVERLENGTH_JOINT in reasons


# ---------------------------------------------------------------------------
# Explicit exclusions
# ---------------------------------------------------------------------------

def test_explicit_record_exclusion():
    decl = _decl(extra_excl_records=("r1",))
    pop = qualify_manifest(decl, [_rec(record_id="r1"), _rec(record_id="r2", H="ACDEFGHIKL")])
    ids = {r.record_id for r in pop.records}
    assert "r1" not in ids
    assert any(e.record_id == "r1" and e.reason == EXCL_EXPLICIT_RECORD for e in pop.excluded)


def test_explicit_content_exclusion():
    r = _rec()
    ckey = _content_key(r.H, r.L, r.Ag)
    decl = _decl(extra_excl_content=(ckey,))
    pop = qualify_manifest(decl, [r])
    assert len(pop.records) == 0
    assert any(e.reason == EXCL_CONTENT for e in pop.excluded)


def test_explicit_antibody_family_exclusion():
    decl = _decl(extra_excl_ab_fam=("fam1",))
    pop = qualify_manifest(decl, [_rec(ab_family="fam1")])
    assert len(pop.records) == 0
    assert any(e.reason == EXCL_ANTIBODY_FAMILY for e in pop.excluded)


# ---------------------------------------------------------------------------
# Group separation
# ---------------------------------------------------------------------------

def test_group_separation_violation():
    """Family 'fam1' appearing in both train and val must raise ValueError."""
    decl = _decl(axes=("antibody_family",))
    r1 = _rec(record_id="r1", ab_family="fam1", partition="train")
    r2 = _rec(record_id="r2", ab_family="fam1", partition="val",
              H="CDWFGHIKLA", L="MNPQRSTVWY", Ag="ACDE")
    with pytest.raises(ValueError, match="Group separation violation"):
        qualify_manifest(decl, [r1, r2])


def test_linked_group_cross_partition_raises():
    decl = _decl(linked_groups=[["r1", "r2"]])
    r1 = _rec(record_id="r1", partition="train")
    r2 = _rec(record_id="r2", partition="val",
              H="CDWFGHIKLA", L="MNPQRSTVWY", Ag="ACDE",
              ab_family="fam2")
    with pytest.raises(ValueError, match="partition"):
        qualify_manifest(decl, [r1, r2])


# ---------------------------------------------------------------------------
# Empty train population
# ---------------------------------------------------------------------------

def test_no_train_population_can_be_reported():
    decl = _decl()
    r = _rec(partition="val")
    pop = qualify_manifest(decl, [r])
    assert pop.train_records() == ()
    assert len(pop.by_partition("val")) == 1


# ---------------------------------------------------------------------------
# Digest stability
# ---------------------------------------------------------------------------

def test_population_digest_stable():
    decl = _decl()
    pop1 = qualify_manifest(decl, [_rec()])
    pop2 = qualify_manifest(decl, [_rec()])
    assert pop1.population_digest == pop2.population_digest


def test_population_digest_changes_with_record():
    decl = _decl()
    pop1 = qualify_manifest(decl, [_rec(record_id="r1")])
    pop2 = qualify_manifest(decl, [_rec(record_id="r2", H="CDWFGHIKLA")])
    assert pop1.population_digest != pop2.population_digest


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def test_coverage_fields():
    decl = _decl()
    r1 = _rec(record_id="r1", ab_family="fam1")
    r2 = _rec(record_id="r2", ab_family="fam2", H="CDWFGHIKLA", L="MNPQRSTVWY", Ag="ACDE")
    pop = qualify_manifest(decl, [r1, r2])
    assert pop.coverage.input_rows == 2
    assert pop.coverage.unique_pairs == 2
    assert pop.coverage.unique_ab == 2
    assert pop.coverage.partitions.get("train", 0) == 2


# ---------------------------------------------------------------------------
# Unknown source / rule fail closed
# ---------------------------------------------------------------------------

def test_unknown_source_fails_closed():
    decl = _decl()
    r = _rec(source_id="unknown_source")
    with pytest.raises(ValueError, match="not declared"):
        qualify_manifest(decl, [r])


def test_unknown_rule_fails_closed():
    decl = _decl()
    r = _rec(rule="unknown_rule")
    with pytest.raises(ValueError, match="not declared"):
        qualify_manifest(decl, [r])


# ---------------------------------------------------------------------------
# Duplicate record_id in input
# ---------------------------------------------------------------------------

def test_duplicate_record_id_raises():
    decl = _decl()
    with pytest.raises(ValueError, match="Duplicate record_id"):
        qualify_manifest(decl, [_rec(record_id="r1"), _rec(record_id="r1", H="CDWFGHIKLA")])


# ---------------------------------------------------------------------------
# Immutability: caller mutation doesn't affect qualified items
# ---------------------------------------------------------------------------

def test_caller_mutation_doesnt_affect_qualified():
    """Qualified population is immutable; mutating input list has no effect."""
    decl = _decl()
    records = [_rec(record_id="r1")]
    pop = qualify_manifest(decl, records)
    records.append(_rec(record_id="r2", H="CDWFGHIKLA"))  # mutate after
    assert len(pop.records) == 1  # unchanged


@pytest.mark.parametrize("shared_axis", ("antibody_family", "target_family"))
@pytest.mark.parametrize("filter_kind", ("none", "record", "content", "family", "length"))
def test_each_separation_axis_checks_full_manifest_before_filters(shared_axis, filter_kind):
    train = _rec(record_id="train", ab_family="ab_train", tgt_family="target_train")
    heldout = _rec(
        record_id="heldout", H="CDEFGHIKLA", Ag="CDEF", partition="val",
        ab_family="ab_heldout", tgt_family="target_heldout",
    )
    heldout = replace(heldout, **{shared_axis: getattr(train, shared_axis)})
    decl = _decl(axes=("antibody_family", "target_family"))
    if filter_kind == "record":
        decl = replace(decl, excluded_record_ids=(heldout.record_id,))
    elif filter_kind == "content":
        decl = replace(decl, excluded_content_keys=(_content_key(heldout.H, heldout.L, heldout.Ag),))
    elif filter_kind == "family":
        field = "excluded_antibody_families" if shared_axis == "antibody_family" else "excluded_target_families"
        decl = replace(decl, **{field: (getattr(heldout, shared_axis),)})
    elif filter_kind == "length":
        heldout = replace(heldout, H="C" * 2000)
    with pytest.raises(ValueError, match=f"Group separation violation: {shared_axis}"):
        qualify_manifest(decl, [train, heldout])


def test_linked_components_are_validated_before_excluded_bridge_disappears():
    records = [
        _rec("a", H="AAAAAAAAAA", ab_family="a"),
        _rec("bridge", H="CCCCCCCCCC", ab_family="bridge"),
        _rec("b", H="DDDDDDDDDD", ab_family="b", partition="test"),
    ]
    decl = _decl(linked_groups=(("a", "bridge"), ("bridge", "b")),
                 extra_excl_records=("bridge", "b"))
    with pytest.raises(ValueError, match="Linked group.*partitions"):
        qualify_manifest(decl, records)


def test_linked_group_cannot_silently_reference_a_missing_record():
    with pytest.raises(ValueError, match="existing record IDs"):
        qualify_manifest(_decl(linked_groups=(("r1", "missing"),)), [_rec()])


def test_family_override_is_explicitly_unsupported_instead_of_ignored():
    with pytest.raises(ValueError, match="overrides are unsupported"):
        GroupingDeclaration(
            separation_axes=("antibody_family",),
            family_id_map=(("r1", ("replacement_ab", "replacement_target")),),
        )


@pytest.mark.parametrize("change", (
    {"H": "CDEFGHIKLA"}, {"h_eligible": (1, 2)},
    {"evidence_assay": "ITC"}, {"evidence_criterion": "different positive call"},
))
def test_conflicting_source_record_identity_fails_even_if_excluded(change):
    first = _rec(record_id="a", source_record_id="source-row")
    second = replace(first, record_id="b", **change)
    with pytest.raises(ValueError, match="conflicting source-record identity"):
        qualify_manifest(_decl(extra_excl_records=("b",)), [first, second])


def test_source_record_ids_are_scoped_to_their_declared_source():
    first = _rec(record_id="a", source_record_id="row-1")
    second = _rec(record_id="b", source_id="s2", source_record_id="row-1", H="CDEFGHIKLA")
    pop = qualify_manifest(_decl(), [first, second])
    assert len(pop.records) == 2


@pytest.mark.parametrize("change", (
    {"h_eligible": (1, 2)}, {"aux_region": (("H", 0),)},
))
def test_duplicate_content_contract_conflict_cannot_be_filtered_away(change):
    first = _rec(record_id="a")
    second = replace(first, record_id="b", source_id="s2", source_record_id="b", **change)
    with pytest.raises(ValueError, match="conflicting duplicate content"):
        qualify_manifest(_decl(extra_excl_records=("b",)), [first, second])


def test_duplicate_owner_and_all_provenance_are_stable_under_every_input_order():
    rows = [
        _rec(record_id="a", source_record_id="z"),
        _rec(record_id="z", source_record_id="a"),
        _rec(record_id="other-source", source_id="s2", source_record_id="0"),
    ]
    populations = [qualify_manifest(_decl(), order) for order in permutations(rows)]
    for pop in populations:
        assert pop.records[0].record_id == "z"
        assert pop.records[0].owning_source_id == "s1"
        assert len(pop.records[0].supporting_refs) == 3
        assert pop == populations[0]


@pytest.mark.parametrize("change", (
    {"partition": "val"}, {"antibody_family": "other-ab-family"},
    {"target_family": "other-target-family"}, {"h_eligible": (1, 2)},
    {"l_eligible": (1,)}, {"aux_region": (("H", 1),)},
    {"evidence_reference": "PDB:OTHER"}, {"evidence_assay": "ITC"},
    {"evidence_criterion": "other positive criterion"},
    {"source_record_id": "other-source-row"},
))
def test_population_identity_binds_metadata_even_when_sequences_and_id_match(change):
    original = _rec()
    old = qualify_manifest(_decl(), [original])
    new = qualify_manifest(_decl(), [replace(original, **change)])
    assert old.records[0].content_key == new.records[0].content_key
    assert old.population_digest != new.population_digest


@pytest.mark.parametrize("changed_field", (
    "source_description", "rule_description", "linked_groups",
    "encoder_limit", "native_limit", "source_priority",
))
def test_manifest_identity_binds_qualification_and_group_declarations(changed_field):
    original = _decl()
    if changed_field == "source_description":
        changed = replace(original, sources=(replace(original.sources[0], description="changed"), original.sources[1]))
    elif changed_field == "rule_description":
        changed = replace(original, rules=(replace(original.rules[0], description="changed criterion"),))
    elif changed_field == "linked_groups":
        changed = replace(original, grouping=replace(original.grouping, linked_groups=(frozenset({"r1"}),)))
    elif changed_field == "encoder_limit":
        changed = replace(original, encoder_max_tokens=1023)
    elif changed_field == "native_limit":
        changed = replace(original, native_max_tokens=1025)
    else:
        changed = replace(original, source_priority_order=("s2", "s1"))
    old = qualify_manifest(original, [_rec()])
    new = qualify_manifest(changed, [_rec()])
    assert old.records == new.records
    assert old.manifest_digest != new.manifest_digest


def test_manifest_identity_is_invariant_to_unordered_declaration_serialization():
    original = _decl(axes=("antibody_family", "target_family"),
                     linked_groups=(("r1",), ("r2",)), extra_excl_records=("missing-a", "missing-b"))
    changed = replace(
        original, sources=tuple(reversed(original.sources)),
        excluded_record_ids=tuple(reversed(original.excluded_record_ids)),
        grouping=replace(original.grouping, separation_axes=tuple(reversed(original.grouping.separation_axes)),
                         linked_groups=tuple(reversed(original.grouping.linked_groups))),
    )
    rows = [_rec("r1"), _rec("r2", H="CDEFGHIKLA")]
    assert qualify_manifest(original, rows) == qualify_manifest(changed, list(reversed(rows)))


@pytest.mark.parametrize("antigen_length, accepted", ((222, True), (223, False)))
def test_joint_encoder_boundary_counts_one_bos_and_one_eos(antigen_length, accepted):
    row = _rec(H="A" * 600, L="C" * 200, Ag="D" * antigen_length)
    pop = qualify_manifest(_decl(), [row])
    assert len(pop.records) == int(accepted)
    assert [e.reason for e in pop.excluded] == ([] if accepted else [EXCL_OVERLENGTH_JOINT])


@pytest.mark.parametrize("paired, residues, accepted", (
    (True, 1021, True), (True, 1022, False),
    (False, 1023, True), (False, 1024, False),
))
def test_native_1026_boundary_uses_actual_chain_special_counts(paired, residues, accepted):
    # Raise only the encoder cap to exercise the native boundary independently.
    decl = replace(_decl(), encoder_max_tokens=4096)
    row = _rec(H="A" * (residues - 400 if paired else residues),
               L="C" * 400 if paired else None, Ag="D",
               ab_format="paired" if paired else "heavy_only")
    pop = qualify_manifest(decl, [row])
    assert len(pop.records) == int(accepted)
    assert [e.reason for e in pop.excluded] == ([] if accepted else [EXCL_OVERLENGTH_PAIRED])


def test_eligible_and_auxiliary_coordinates_are_canonicalized_before_deduplication():
    first = _rec(record_id="a", h_elig=(2, 0, 1), l_elig=(1, 0),
                 aux=(("L", 0), ("H", 2)))
    second = _rec(record_id="b", source_id="s2", h_elig=(0, 1, 2), l_elig=(0, 1),
                  aux=(("H", 2), ("L", 0)))
    pop = qualify_manifest(_decl(), [first, second])
    assert len(pop.records) == 1
    assert pop.records[0].eligible_union() == (("H", 0), ("H", 1), ("H", 2), ("L", 0), ("L", 1))
    assert pop.records[0].aux_region == (("H", 2), ("L", 0))


@pytest.mark.parametrize("aux", ((), (("H", 5),), (("L", 3),)))
def test_auxiliary_region_must_be_nonempty_and_within_eligible_union(aux):
    with pytest.raises(ValueError, match="nonempty subset"):
        _rec(aux=aux)


def test_clearing_caller_axes_cannot_disable_validated_split_contract():
    axes = ["antibody_family", "target_family"]
    grouping = GroupingDeclaration(separation_axes=axes)
    axes.clear()
    assert grouping.separation_axes == ("antibody_family", "target_family")
    rows = [_rec("train"), _rec("val", H="CDEFGHIKLA", partition="val")]
    with pytest.raises(ValueError, match="Group separation violation"):
        qualify_manifest(replace(_decl(), grouping=grouping), rows)


def test_linked_group_aliases_cannot_remove_split_constraint():
    members = {"a", "b"}
    groups = [members]
    grouping = GroupingDeclaration(separation_axes=["antibody_family"], linked_groups=groups)
    members.clear()
    groups.clear()
    rows = [_rec("a", ab_family="a"),
            _rec("b", H="CDEFGHIKLA", ab_family="b", partition="test")]
    assert grouping.linked_groups == (frozenset({"a", "b"}),)
    with pytest.raises(ValueError, match="Linked group.*partitions"):
        qualify_manifest(replace(_decl(), grouping=grouping), rows)


def test_manifest_declaration_copies_source_rule_priority_and_exclusion_lists():
    sources, rules, priorities, excluded = [_src("s1")], [_rule()], ["s1"], ["r1"]
    declaration = ManifestDeclaration(
        sources=sources, rules=rules, source_priority_order=priorities,
        grouping=GroupingDeclaration(("antibody_family",)), excluded_record_ids=excluded,
    )
    before = qualify_manifest(declaration, [_rec()])
    sources.clear()
    rules.clear()
    priorities.clear()
    excluded.clear()
    after = qualify_manifest(declaration, [_rec()])
    assert before == after
    assert not after.records
    assert after.excluded[0].reason == EXCL_EXPLICIT_RECORD
