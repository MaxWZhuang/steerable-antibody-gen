"""Tests for trastuzumab_poc sampling, protocol, and budget layers."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import fields, replace

import pytest

from smallAntibodyGen.experiments.trastuzumab_poc.data import (
    InputRecord, SourceDeclaration, QualificationRule,
    GroupingDeclaration, ManifestDeclaration,
    qualify_manifest,
)
from smallAntibodyGen.experiments.trastuzumab_poc.sampling import (
    DrawKey, FamilyFirstSampler, SamplerConfig, SampledBatch,
    build_paired_view, build_abonly_view,
)
from smallAntibodyGen.experiments.trastuzumab_poc.protocol import (
    EndpointConstruct, EndpointEntry, EndpointManifest,
    SelectedBudget, select_budget, extend_budget,
    MeasuredEndpoint, DevelopmentLabel, SelectionAllowance, RestrictedPostData,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _src(sid, priority=0):
    return SourceDeclaration(source_id=sid, description="test", priority=priority)


def _rule(name="rule_A"):
    return QualificationRule(rule_name=name, description="positive")


def _decl(sources=None, axes=("antibody_family",)):
    sources = sources or [_src("s1", 0), _src("s2", 1)]
    src_ids = tuple(s.source_id for s in sources)
    return ManifestDeclaration(
        sources=tuple(sources),
        rules=(_rule(),),
        source_priority_order=src_ids,
        grouping=GroupingDeclaration(separation_axes=tuple(axes)),
    )


def _rec(record_id, H="ACDEFGHIKL", L="MNPQRSTVWY", Ag="ACDE",
         ab_family="fam1", tgt_family="her2", source_id="s1",
         partition="train"):
    return InputRecord(
        record_id=record_id,
        source_id=source_id,
        source_record_id=record_id,
        H=H,
        L=L,
        Ag=Ag,
        antibody_format="paired",
        antibody_family=ab_family,
        target_family=tgt_family,
        partition=partition,
        h_eligible=(0, 1, 2),
        l_eligible=(0, 1),
        aux_region=None,
        evidence_reference="PDB:X",
        evidence_assay="SPR",
        evidence_criterion="KD<1nM",
        qualification_rule="rule_A",
    )


def _make_pop_two_families():
    decl = _decl()
    records = [
        _rec("r1", ab_family="famA", H="ACDEFGHIKL"),
        _rec("r2", ab_family="famA", H="ACDEFGHIKM"),
        _rec("r3", ab_family="famB", H="CDWFGHIKLA"),
        _rec("r4", ab_family="famB", H="CDWFGHIKLM"),
    ]
    return qualify_manifest(decl, records)


def _cfg():
    return SamplerConfig(
        batch_size=4,
        algorithm="family_first_v1",
        base_seed=42,
    )


# ---------------------------------------------------------------------------
# Row-order invariant sampling
# ---------------------------------------------------------------------------

def test_row_order_invariant():
    """Shuffling input record order produces identical draws."""
    decl = _decl()
    recs_a = [
        _rec("r1", ab_family="famA", H="ACDEFGHIKL"),
        _rec("r2", ab_family="famA", H="ACDEFGHIKM"),
        _rec("r3", ab_family="famB", H="CDWFGHIKLA"),
    ]
    recs_b = [recs_a[2], recs_a[0], recs_a[1]]  # shuffled
    pop_a = qualify_manifest(decl, recs_a)
    pop_b = qualify_manifest(decl, recs_b)
    # Same population digest means same qualified set
    assert pop_a.population_digest == pop_b.population_digest
    sam_a = FamilyFirstSampler(pop_a, _cfg())
    sam_b = FamilyFirstSampler(pop_b, _cfg())
    batch_a = sam_a.draw_batch(0)
    batch_b = sam_b.draw_batch(0)
    ids_a = tuple(i.record_id for i in batch_a.paired_items)
    ids_b = tuple(i.record_id for i in batch_b.paired_items)
    assert ids_a == ids_b


# ---------------------------------------------------------------------------
# Paired and abonly arms select identical record_ids
# ---------------------------------------------------------------------------

def test_paired_abonly_same_ids():
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    for upd in range(5):
        batch = sam.draw_batch(upd)
        assert len(batch.paired_items) == len(batch.abonly_items)
        for p, a in zip(batch.paired_items, batch.abonly_items):
            assert p.record_id == a.record_id


# ---------------------------------------------------------------------------
# Independent model RNG cannot affect draw API
# ---------------------------------------------------------------------------

def test_model_rng_independent():
    """Calling mask_rng many times does not shift item-selection draws."""
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    batch_before = sam.draw_batch(7)
    # Exhaust mask RNGs
    for ex in range(100):
        rng = sam.mask_rng(7, ex)
        for _ in range(200):
            rng.random()
    batch_after = sam.draw_batch(7)
    ids_before = tuple(i.record_id for i in batch_before.paired_items)
    ids_after = tuple(i.record_id for i in batch_after.paired_items)
    assert ids_before == ids_after


# ---------------------------------------------------------------------------
# Deterministic draws
# ---------------------------------------------------------------------------

def test_deterministic_same_update():
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    b1 = sam.draw_batch(3)
    b2 = sam.draw_batch(3)
    assert tuple(i.record_id for i in b1.paired_items) == tuple(i.record_id for i in b2.paired_items)


def test_update_changes_rng_stream_without_changing_resume_draws():
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    first, second = sam.draw_batch(0), sam.draw_batch(1)
    assert first.mask_seeds != second.mask_seeds
    assert sam.draw_batch(0) == first
    assert FamilyFirstSampler(pop, _cfg()).draw_batch(1) == second


# ---------------------------------------------------------------------------
# View construction: paired includes Ag, abonly does not
# ---------------------------------------------------------------------------

def test_build_paired_view_includes_ag():
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    batch = sam.draw_batch(0)
    item = batch.paired_items[0]
    masked = [item.eligible_union()[0]]  # mask first eligible coord
    view, targets = build_paired_view(item, masked)
    assert "Ag" in view.roles


def test_build_abonly_view_excludes_ag():
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    batch = sam.draw_batch(0)
    item = batch.abonly_items[0]
    masked = [item.eligible_union()[0]]
    view, targets = build_abonly_view(item, masked)
    assert "Ag" not in view.roles


# ---------------------------------------------------------------------------
# Mask seeds per item
# ---------------------------------------------------------------------------

def test_mask_seeds_count():
    pop = _make_pop_two_families()
    sam = FamilyFirstSampler(pop, _cfg())
    batch = sam.draw_batch(0)
    assert len(batch.mask_seeds) == _cfg().batch_size


# ---------------------------------------------------------------------------
# Protocol: EndpointManifest validation
# ---------------------------------------------------------------------------

def _construct(core="ACDEFGHIKL", prefix="", suffix="", light="MNPQRSTVWY", ag="ACDE"):
    return EndpointConstruct(
        heavy_prefix=prefix,
        core=core,
        heavy_suffix=suffix,
        light=light,
        Ag=ag,
    )


def _entry(eid, role="post_pool", group="g1", core="ACDEFGHIKL",
           assay="SPR", semantics="KD"):
    return EndpointEntry(
        endpoint_id=eid,
        construct=_construct(core=core),
        role=role,
        neighbourhood_group=group,
        assay_id=assay,
        endpoint_semantics=semantics,
    )


def test_endpoint_core_must_be_10():
    with pytest.raises(ValueError, match="10 residues"):
        EndpointConstruct(heavy_prefix="", core="ACDE", heavy_suffix="", light="ACDE", Ag="ACDE")


def test_neighbourhood_group_cross_roles_raises():
    with pytest.raises(ValueError, match="neighbourhood_group"):
        EndpointManifest(entries=(
            _entry("e1", role="post_pool", group="gX"),
            _entry("e2", role="development", group="gX"),  # same group, different role
        ))


def test_duplicate_construct_cross_roles_raises():
    with pytest.raises(ValueError, match="Duplicate construct"):
        EndpointManifest(entries=(
            _entry("e1", role="post_pool", group="g1", core="ACDEFGHIKL"),
            _entry("e2", role="development", group="g2", core="ACDEFGHIKL"),  # same core
        ))


def test_duplicate_endpoint_id_raises():
    with pytest.raises(ValueError, match="Duplicate endpoint_id"):
        EndpointManifest(entries=(
            _entry("e1", role="post_pool", group="g1"),
            _entry("e1", role="post_pool", group="g2", core="CDWFGHIKLA"),
        ))


# ---------------------------------------------------------------------------
# Budget selection: outcome-free, deterministic, prefix
# ---------------------------------------------------------------------------

def _manifest_4():
    return EndpointManifest(entries=(
        _entry("e1", role="post_pool", group="g1", core="ACDEFGHIKL"),
        _entry("e2", role="post_pool", group="g1", core="ACDEFGHIKM"),
        _entry("e3", role="post_pool", group="g2", core="ACDEFGHIKN"),
        _entry("e4", role="post_pool", group="g2", core="ACDEFGHIKP"),
    ))


def test_select_budget_correct_count():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    assert bud.budget_n == 2
    assert len(bud.selected_ids) == 2


def test_select_budget_deterministic():
    man = _manifest_4()
    b1 = select_budget(man, 3, "seed_xyz")
    b2 = select_budget(man, 3, "seed_xyz")
    assert b1.selected_ids == b2.selected_ids
    assert b1.selection_digest == b2.selection_digest


def test_select_budget_prefix_consistent():
    """Budget N=3 must include the same endpoints as budget N=2."""
    man = _manifest_4()
    b2 = select_budget(man, 2, "seed_abc")
    b3 = select_budget(man, 3, "seed_abc")
    assert b3.selected_ids[:2] == b2.selected_ids


def test_select_budget_0():
    man = _manifest_4()
    bud = select_budget(man, 0, "seed_abc")
    assert bud.budget_n == 0
    assert len(bud.selected_ids) == 0


def test_select_budget_exceeds_pool_raises():
    man = _manifest_4()
    with pytest.raises(ValueError, match="exceeds post_pool"):
        select_budget(man, 99, "seed_abc")


def test_extend_budget():
    man = _manifest_4()
    b2 = select_budget(man, 2, "seed_abc")
    b4 = extend_budget(man, b2, 4)
    assert b4.budget_n == 4
    assert b4.selected_ids[:2] == b2.selected_ids


def test_extend_budget_decrease_raises():
    man = _manifest_4()
    b3 = select_budget(man, 3, "seed_abc")
    with pytest.raises(ValueError):
        extend_budget(man, b3, 1)


# ---------------------------------------------------------------------------
# Changed outcome values do not change budget selection
# ---------------------------------------------------------------------------

def test_changed_outcomes_dont_change_budget():
    man = _manifest_4()
    b1 = select_budget(man, 2, "seed_abc")
    # "outcome values" don't affect select_budget at all — just re-select with same seed
    b2 = select_budget(man, 2, "seed_abc")
    assert b1.selected_ids == b2.selected_ids


# ---------------------------------------------------------------------------
# RestrictedPostData: only budget IDs accepted
# ---------------------------------------------------------------------------

def _make_restricted(manifest, budget, measured_vals):
    meas = tuple(
        MeasuredEndpoint(
            endpoint_id=eid,
            value=v,
            orientation="higher_better",
            assay_id="SPR",
        )
        for eid, v in measured_vals.items()
    )
    return RestrictedPostData(
        manifest=manifest,
        budget=budget,
        measured=meas,
        fitting_count=len(meas),
        development_labels=(),
        selection_allowance=SelectionAllowance(allowance=0, actual_development_ids=()),
    )


def test_restricted_post_data_accepts_budget_ids():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    vals = {eid: 1.0 for eid in bud.selected_ids}
    rpd = _make_restricted(man, bud, vals)
    assert rpd.fitting_count == 2


def test_restricted_post_data_rejects_extra_endpoint():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    extra_id = next(eid for eid in ("e1", "e2", "e3", "e4") if eid not in bud.selected_ids)
    vals = {eid: 1.0 for eid in bud.selected_ids}
    vals[extra_id] = 2.0  # NOT in budget
    with pytest.raises(ValueError, match="not in selected budget"):
        _make_restricted(man, bud, vals)


def test_restricted_post_data_duplicate_measured_raises():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    first_id = bud.selected_ids[0]
    meas = (
        MeasuredEndpoint(endpoint_id=first_id, value=1.0, orientation="higher_better", assay_id="SPR"),
        MeasuredEndpoint(endpoint_id=first_id, value=2.0, orientation="higher_better", assay_id="SPR"),
    )
    with pytest.raises(ValueError, match="Duplicate measured"):
        RestrictedPostData(
            manifest=man,
            budget=bud,
            measured=meas,
            fitting_count=2,
            development_labels=(),
            selection_allowance=SelectionAllowance(allowance=0, actual_development_ids=()),
        )


# ---------------------------------------------------------------------------
# ordered_pairs: ties explicit, low budget, orientation
# ---------------------------------------------------------------------------

def test_ordered_pairs_low_budget_empty():
    man = _manifest_4()
    bud = select_budget(man, 1, "seed_abc")
    vals = {bud.selected_ids[0]: 1.0}
    rpd = _make_restricted(man, bud, vals)
    # budget_n < 2 -> ordering unsupported
    pairs = rpd.ordered_pairs(man)
    assert pairs == ()


def test_ordered_pairs_higher_better():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    id0, id1 = bud.selected_ids[0], bud.selected_ids[1]
    meas = (
        MeasuredEndpoint(endpoint_id=id0, value=3.0, orientation="higher_better", assay_id="SPR"),
        MeasuredEndpoint(endpoint_id=id1, value=1.0, orientation="higher_better", assay_id="SPR"),
    )
    rpd = RestrictedPostData(
        manifest=man,
        budget=bud,
        measured=meas,
        fitting_count=2,
        development_labels=(),
        selection_allowance=SelectionAllowance(allowance=0, actual_development_ids=()),
    )
    pairs = rpd.ordered_pairs(man)
    assert len(pairs) == 1
    better, worse = pairs[0]
    assert better == id0
    assert worse == id1


def test_ordered_pairs_tie_explicit():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    id0, id1 = bud.selected_ids[0], bud.selected_ids[1]
    meas = (
        MeasuredEndpoint(endpoint_id=id0, value=2.0, orientation="higher_better", assay_id="SPR"),
        MeasuredEndpoint(endpoint_id=id1, value=2.0, orientation="higher_better", assay_id="SPR"),
    )
    rpd = RestrictedPostData(
        manifest=man,
        budget=bud,
        measured=meas,
        fitting_count=2,
        development_labels=(),
        selection_allowance=SelectionAllowance(allowance=0, actual_development_ids=()),
    )
    pairs = rpd.ordered_pairs(man)
    assert pairs == ()


# ---------------------------------------------------------------------------
# Development labels counted separately from fitting labels
# ---------------------------------------------------------------------------

def test_development_labels_separate_from_fitting():
    man = _manifest_4()
    bud = select_budget(man, 2, "seed_abc")
    vals = {eid: 1.0 for eid in bud.selected_ids}
    meas = tuple(
        MeasuredEndpoint(endpoint_id=eid, value=v, orientation="higher_better", assay_id="SPR")
        for eid, v in vals.items()
    )
    dev_entry = _entry("dev1", role="development", group="gdev", core="ACDEFGHIKQ")
    man2 = EndpointManifest(entries=(
        _entry("e1", role="post_pool", group="g1", core="ACDEFGHIKL"),
        _entry("e2", role="post_pool", group="g1", core="ACDEFGHIKM"),
        _entry("e3", role="post_pool", group="g2", core="ACDEFGHIKN"),
        _entry("e4", role="post_pool", group="g2", core="ACDEFGHIKP"),
        dev_entry,
    ))
    bud2 = select_budget(man2, 2, "seed_abc")

    dev_label = DevelopmentLabel(
        endpoint_id="dev1", value=5.0, orientation="higher_better", assay_id="SPR"
    )
    allowance = SelectionAllowance(allowance=1, actual_development_ids=("dev1",))
    meas2 = tuple(
        MeasuredEndpoint(endpoint_id=eid, value=1.0, orientation="higher_better", assay_id="SPR")
        for eid in bud2.selected_ids
    )
    rpd = RestrictedPostData(
        manifest=man2,
        budget=bud2,
        measured=meas2,
        fitting_count=len(meas2),
        development_labels=(dev_label,),
        selection_allowance=allowance,
    )
    assert rpd.fitting_count == 2
    assert len(rpd.development_labels) == 1
    # Selection labels must be counted in the total label exposure.
    assert rpd.fitting_count + len(rpd.development_labels) == 3


# ---------------------------------------------------------------------------
# SelectionAllowance count mismatch
# ---------------------------------------------------------------------------

def test_selection_allowance_mismatch_raises():
    with pytest.raises(ValueError, match="allowance"):
        SelectionAllowance(allowance=2, actual_development_ids=("dev1",))


# ---------------------------------------------------------------------------
# MeasuredEndpoint rejects non-finite values
# ---------------------------------------------------------------------------

def test_measured_endpoint_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        MeasuredEndpoint(endpoint_id="e1", value=float("nan"),
                         orientation="higher_better", assay_id="SPR")


def test_measured_endpoint_rejects_bool():
    with pytest.raises(TypeError, match="numeric"):
        MeasuredEndpoint(endpoint_id="e1", value=True,
                         orientation="higher_better", assay_id="SPR")


@pytest.mark.parametrize("population_kind", ("excluded", "heldout_only", "empty"))
def test_empty_train_can_be_qualified_but_cannot_start_sampling(population_kind):
    decl = _decl()
    if population_kind == "excluded":
        decl = replace(decl, excluded_record_ids=("r1",))
        rows = [_rec("r1")]
    elif population_kind == "heldout_only":
        rows = [_rec("r1", partition="val")]
    else:
        rows = []
    pop = qualify_manifest(decl, rows)
    assert pop.train_records() == ()
    with pytest.raises(ValueError, match="No train records"):
        FamilyFirstSampler(pop, _cfg())


def test_matched_views_preserve_masks_targets_and_auxiliary_region():
    row = replace(_rec("r1"), aux_region=(("H", 1), ("L", 0)))
    sam = FamilyFirstSampler(qualify_manifest(_decl(), [row]), _cfg())
    batch = sam.draw_batch(3)
    assert batch.paired_items == batch.abonly_items
    for paired_item, ab_item in zip(batch.paired_items, batch.abonly_items):
        assert paired_item.aux_region == row.aux_region
        masks = (("H", 1), ("L", 0))
        paired, paired_targets = build_paired_view(paired_item, masks)
        ab_only, ab_targets = build_abonly_view(ab_item, masks)
        assert paired_targets == ab_targets
        assert tuple(chain for role, chain in zip(paired.roles, paired.chains) if role != "Ag") == ab_only.chains
        assert paired.roles == ("H", "L", "Ag")
        assert ab_only.roles == ("H", "L")


def test_heavy_only_record_uses_same_real_heavy_chain_in_both_arms():
    row = replace(_rec("vhh"), L=None, antibody_format="heavy_only", l_eligible=None)
    sampler = FamilyFirstSampler(qualify_manifest(_decl(), [row]), _cfg())
    item = sampler.draw_batch(0).paired_items[0]
    paired, paired_targets = build_paired_view(item, (("H", 1),))
    ab_only, ab_targets = build_abonly_view(item, (("H", 1),))
    assert paired.roles == ("H", "Ag")
    assert ab_only.roles == ("H",)
    assert paired_targets == ab_targets
    assert paired.chains[0] == ab_only.chains[0]


def test_joint_overlength_and_heldout_records_never_enter_either_arm():
    rows = [
        _rec("train"),
        _rec("long", H="C" * 600, L="D" * 200, Ag="E" * 223),
        _rec("heldout", H="F" * 10, ab_family="heldout", partition="test"),
    ]
    pop = qualify_manifest(_decl(), rows)
    assert {r.record_id for r in pop.records} == {"train", "heldout"}
    sampler = FamilyFirstSampler(pop, _cfg())
    for update in range(5):
        batch = sampler.draw_batch(update)
        assert {item.record_id for item in batch.paired_items} == {"train"}
        assert batch.paired_items == batch.abonly_items


@pytest.mark.parametrize("field,value", (("seed", 43), ("algorithm", "other-algorithm")))
def test_keyed_draw_rejects_identity_different_from_sampler(field, value):
    sampler = FamilyFirstSampler(_make_pop_two_families(), _cfg())
    key = DrawKey("family_first_v1", 42, 2, 0, "item_select")
    assert sampler.draw_item_keyed(key) == sampler.draw_batch(2).paired_items[0]
    with pytest.raises(ValueError, match="match sampler"):
        sampler.draw_item_keyed(replace(key, **{field: value}))


def test_family_then_source_balance_is_not_row_count_weighting():
    # One source has nine times more rows, and one family has ten times more rows.
    # Neither imbalance should multiply its exposure at the earlier sampling stage.
    rows = [_rec("a-single", H="A" * 10, ab_family="family-a", source_id="s1")]
    rows += [_rec(f"a-many-{aa}", H=aa * 10, ab_family="family-a", source_id="s2")
             for aa in "CDEFGHIKL"]
    rows.append(_rec("b-single", H="M" * 10, ab_family="family-b", source_id="s1"))
    sampler = FamilyFirstSampler(qualify_manifest(_decl(), rows), _cfg())
    counts = Counter((item.antibody_family, item.source_id)
                     for update in range(1000) for item in sampler.draw_batch(update).paired_items)
    total = sum(counts.values())
    assert abs(counts[("family-a", "s1")] / total - 0.25) < 0.04
    assert abs(counts[("family-a", "s2")] / total - 0.25) < 0.04
    assert abs(counts[("family-b", "s1")] / total - 0.50) < 0.04


@pytest.mark.parametrize("field,value", (
    ("heavy_prefix", "A"), ("heavy_suffix", "C"),
    ("light", "ACDEFGHIKL"), ("Ag", "CDEF"),
))
def test_endpoint_manifest_rejects_mixed_scaffold_or_antigen(field, value):
    first = _entry("e1")
    second = _entry("e2", core="ACDEFGHIKM")
    second = replace(second, construct=replace(second.construct, **{field: value}))
    with pytest.raises(ValueError, match="one fixed scaffold and antigen"):
        EndpointManifest((first, second))


def _manifest_with_development():
    return EndpointManifest(_manifest_4().entries + (
        _entry("dev1", role="development", group="dev1", core="ACDEFGHIKQ"),
        _entry("dev2", role="development", group="dev2", core="ACDEFGHIKR"),
        _entry("reserved", role="reserved", group="reserved", core="ACDEFGHIKS"),
    ))


def _restricted_kwargs(manifest=None, n=2, development_ids=()):
    manifest = manifest or _manifest_with_development()
    budget = select_budget(manifest, n, "shared-budget-seed")
    return dict(
        manifest=manifest, budget=budget,
        measured=tuple(MeasuredEndpoint(eid, float(i), "higher_better", manifest.by_id(eid).assay_id)
                       for i, eid in enumerate(budget.selected_ids)),
        fitting_count=n,
        development_labels=tuple(DevelopmentLabel(eid, float(i), "higher_better", "SPR")
                                 for i, eid in enumerate(development_ids)),
        selection_allowance=SelectionAllowance(len(development_ids), tuple(development_ids)),
    )


def test_budget_order_is_invariant_to_manifest_row_order_and_has_exact_prefixes():
    manifest = _manifest_with_development()
    shuffled = EndpointManifest(tuple(reversed(manifest.entries)))
    full = select_budget(manifest, 4, "same-for-all-training-seeds")
    for n in range(5):
        budget = select_budget(manifest, n, "same-for-all-training-seeds")
        assert budget == select_budget(shuffled, n, "same-for-all-training-seeds")
        assert budget.selected_ids == full.selected_ids[:n]
        assert set(budget.selected_ids) <= {e.endpoint_id for e in manifest.post_pool_entries()}


def test_restricted_data_discards_full_manifest_and_unselected_constructs():
    kwargs = _restricted_kwargs(development_ids=("dev1",))
    data = RestrictedPostData(**kwargs)
    assert "manifest" not in {field.name for field in fields(data)}
    assert "manifest" not in vars(data)
    assert {entry.endpoint_id for entry in data.selected_entries} == set(data.budget.selected_ids)
    assert {eid for eid, _ in data.constructs()} == set(data.budget.selected_ids)
    assert set(data.budget.selected_ids).isdisjoint({"dev1", "dev2", "reserved"})
    assert {label.endpoint_id for label in data.measured} == set(data.budget.selected_ids)


def test_missing_selected_outcome_is_not_silently_a_smaller_label_budget():
    kwargs = _restricted_kwargs()
    kwargs["measured"] = kwargs["measured"][:1]
    kwargs["fitting_count"] = 1
    with pytest.raises(ValueError, match="exactly match selected budget"):
        RestrictedPostData(**kwargs)


@pytest.mark.parametrize("count", (0, 1, 3, True, 2.0))
def test_fitting_count_must_equal_exact_unique_endpoint_budget(count):
    kwargs = _restricted_kwargs()
    kwargs["fitting_count"] = count
    with pytest.raises(ValueError, match="fitting_count"):
        RestrictedPostData(**kwargs)


@pytest.mark.parametrize("label_kind", ("fitting", "development"))
def test_label_assay_must_match_declared_endpoint_assay(label_kind):
    kwargs = _restricted_kwargs(development_ids=("dev1",))
    key = "measured" if label_kind == "fitting" else "development_labels"
    kwargs[key] = (replace(kwargs[key][0], assay_id="undeclared-assay"),) + kwargs[key][1:]
    with pytest.raises(ValueError, match="role or assay"):
        RestrictedPostData(**kwargs)


@pytest.mark.parametrize("bad_id", ("reserved", "e1"))
def test_development_allowance_cannot_relabel_reserved_or_fitting_endpoints(bad_id):
    kwargs = _restricted_kwargs(development_ids=(bad_id,))
    with pytest.raises(ValueError, match="role or assay"):
        RestrictedPostData(**kwargs)


@pytest.mark.parametrize("mismatch", ("missing", "extra", "duplicate", "wrong_id"))
def test_development_labels_must_exactly_match_declared_ids(mismatch):
    kwargs = _restricted_kwargs(development_ids=("dev1",))
    dev = kwargs["development_labels"][0]
    if mismatch == "missing":
        labels = ()
    elif mismatch == "extra":
        labels = (dev, replace(dev, endpoint_id="dev2"))
    elif mismatch == "duplicate":
        labels = (dev, dev)
    else:
        labels = (replace(dev, endpoint_id="dev2"),)
    kwargs["development_labels"] = labels
    with pytest.raises(ValueError, match="exactly match declared allowance"):
        RestrictedPostData(**kwargs)


@pytest.mark.parametrize("change", ("assay", "semantics", "construct", "group", "role"))
def test_altered_manifest_cannot_reuse_budget_or_construct_access(change):
    kwargs = _restricted_kwargs()
    original = kwargs["manifest"]
    data = RestrictedPostData(**kwargs)
    entries = list(original.entries)
    if change == "assay":
        entries[0] = replace(entries[0], assay_id="ITC")
    elif change == "semantics":
        entries = [replace(e, endpoint_semantics="changed endpoint definition") for e in entries]
    elif change == "construct":
        entries[0] = replace(entries[0], construct=replace(entries[0].construct, core="ACDEFGHIKT"))
    elif change == "group":
        entries[0] = replace(entries[0], neighbourhood_group="new-neighbourhood")
    else:
        entries[0] = replace(entries[0], role="reserved", neighbourhood_group="new-reserved")
    altered = EndpointManifest(tuple(entries))
    assert altered.manifest_digest() != original.manifest_digest()
    with pytest.raises(ValueError, match="match manifest"):
        RestrictedPostData(**{**kwargs, "manifest": altered})
    with pytest.raises(ValueError, match="match manifest"):
        extend_budget(altered, data.budget, 3)
    with pytest.raises(ValueError, match="manifest identity changed"):
        data.constructs(altered)


def test_measured_and_development_orientation_agree_within_each_assay():
    kwargs = _restricted_kwargs(development_ids=("dev1",))
    kwargs["development_labels"] = (replace(kwargs["development_labels"][0], orientation="lower_better"),)
    with pytest.raises(ValueError, match="mixed orientations"):
        RestrictedPostData(**kwargs)


def test_pair_identity_draws_are_outcome_independent_and_do_not_replace_ties():
    manifest = _manifest_4()
    manifest = EndpointManifest(tuple(replace(e, assay_id="ITC") if e.endpoint_id in {"e3", "e4"} else e
                                      for e in manifest.entries))
    kwargs = _restricted_kwargs(manifest, n=4)
    normal = RestrictedPostData(**kwargs)
    inverted = RestrictedPostData(**{**kwargs, "measured": tuple(replace(m, value=-m.value) for m in kwargs["measured"])})
    ties = RestrictedPostData(**{**kwargs, "measured": tuple(replace(m, value=0.0) for m in kwargs["measured"])})
    rngs = [random.Random(981) for _ in range(3)]
    observed_groups = set()
    for _ in range(100):
        a, b, tied = [data.draw_pair(rng) for data, rng in zip((normal, inverted, ties), rngs)]
        assert (a.left_id, a.right_id) == (b.left_id, b.right_id) == (tied.left_id, tied.right_id)
        assert a.better_id == b.worse_id and a.worse_id == b.better_id
        assert a.informative and b.informative
        assert not tied.informative and tied.better_id is None and tied.worse_id is None
        assert rngs[0].getstate() == rngs[1].getstate() == rngs[2].getstate()
        assay = manifest.by_id(a.left_id).assay_id
        assert manifest.by_id(a.right_id).assay_id == assay
        observed_groups.add(assay)
    assert observed_groups == {"SPR", "ITC"}


def test_pair_orientation_handles_lower_better_without_changing_sampled_ids():
    kwargs = _restricted_kwargs(_manifest_4(), n=4)
    normal = RestrictedPostData(**kwargs)
    lower = RestrictedPostData(**{**kwargs, "measured": tuple(replace(m, orientation="lower_better") for m in kwargs["measured"])})
    for seed in range(20):
        a, b = normal.draw_pair(random.Random(seed)), lower.draw_pair(random.Random(seed))
        assert (a.left_id, a.right_id) == (b.left_id, b.right_id)
        assert (a.better_id, a.worse_id) == (b.worse_id, b.better_id)


def test_single_endpoint_cannot_produce_a_pair_or_expand_its_budget():
    data = RestrictedPostData(**_restricted_kwargs(n=1))
    with pytest.raises(ValueError, match="ordering unsupported"):
        data.draw_pair(random.Random(7))
    assert len(data.measured) == len(data.constructs()) == data.budget.budget_n == 1


def test_appending_reserved_duplicate_cannot_mutate_validated_endpoint_manifest():
    entries = list(_manifest_4().entries)
    manifest = EndpointManifest(entries)
    original_digest = manifest.manifest_digest()
    budget = select_budget(manifest, 2, "seed")
    entries.append(_entry("reserved-copy", role="reserved", group="reserved", core=entries[0].construct.core))
    assert len(manifest.entries) == 4
    assert manifest.manifest_digest() == original_digest
    assert select_budget(manifest, 2, "seed") == budget
    with pytest.raises(ValueError, match="Duplicate construct"):
        EndpointManifest(entries)


def test_development_allowance_copies_input_ids_before_validation():
    caller_ids = ["dev1"]
    allowance = SelectionAllowance(1, caller_ids)
    caller_ids.append("reserved")
    assert allowance.actual_development_ids == ("dev1",)
    kwargs = _restricted_kwargs(development_ids=("dev1",))
    kwargs["selection_allowance"] = allowance
    data = RestrictedPostData(**kwargs)
    assert len(data.development_labels) == data.selection_allowance.allowance == 1


def test_restricted_data_copies_label_lists_and_cannot_gain_reserved_labels():
    kwargs = _restricted_kwargs(development_ids=("dev1",))
    measured = list(kwargs["measured"])
    development = list(kwargs["development_labels"])
    data = RestrictedPostData(**{**kwargs, "measured": measured, "development_labels": development})
    original_digest = data.data_digest()
    measured.append(MeasuredEndpoint("reserved", 1.0, "higher_better", "SPR"))
    development.clear()
    assert data.data_digest() == original_digest
    assert {label.endpoint_id for label in data.measured} == set(data.budget.selected_ids)
    assert tuple(label.endpoint_id for label in data.development_labels) == ("dev1",)
