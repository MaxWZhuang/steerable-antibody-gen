"""Regressions found at the real-source/runtime handoff of the HER2 campaign."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from smallAntibodyGen.experiments import her2_routes_data as data
from smallAntibodyGen.experiments import her2_routes_sources as sources
from smallAntibodyGen.experiments.her2_routes_conditions import ConditionSpec
from smallAntibodyGen.experiments.her2_routes_serial import Construct


def observation(heavy="ACDE", *, value=8.0, uncertainty=None, raw=None, assay="assay"):
    return sources.Observation(
        construct=Construct(heavy=heavy, light="FGH", sites=(("H", 2),),
                            source="fixture", record_id=heavy),
        source="fixture", family="affinity", kind=sources.KIND_POINT,
        target="target", assay=assay, parent="parent", scale=sources.SCALE_PKD,
        value=value, interval=(value, value), uncertainty=uncertainty, raw=raw or {})


def test_background_control_permutes_effects_without_changing_constructs_or_other_targets():
    from smallAntibodyGen.experiments.her2_routes_controls import permute_background_effects
    records = [observation(heavy, value=value) for heavy, value in
               [("ACDE", 1), ("ACWE", 3), ("FCDE", 7), ("FCWE", 2)]]
    rows = data.summarise_observations(records)
    by_heavy = {row.construct.heavy: row for row in rows}
    contrasts = []
    for first, second in [("ACWE", "ACDE"), ("FCWE", "FCDE")]:
        left, right = by_heavy[first], by_heavy[second]
        contrasts.append(data.LocalContrast(
            left, right, (("H", 2),), left.value, right.value,
            left.assay, left.target, True, False, margin=left.value - right.value))
    changed, audit = permute_background_effects(contrasts, seed=3)
    assert audit["effect_bundles_moved"] == 2
    assert sorted(c.margin for c in changed) == [-5, 2]
    assert changed[0].margin == -5 and changed[1].margin == 2
    for before, after in zip(contrasts, changed):
        assert before.left is after.left and before.right is after.right
        assert after.margin == after.left_value - after.right_value
    assert changed[0].left.value == 3


def test_energy_draws_have_no_fabricated_sequence_log_probability():
    from smallAntibodyGen.experiments.her2_routes_generate import DrawResult
    result = DrawResult(("AAAA",), None, 101, 1.0, "walk_jump")
    assert result.document()["mean_log_probability"] is None


def test_parent_geometry_does_not_depend_on_an_indel_in_the_first_row():
    records = [observation("AACDE"), observation("ACWE"), observation("ACDE")]
    annotated, unalignable, geometry = sources._assign_sites(
        records, {"heavy": "ACDE", "light": "FGH"})
    assert unalignable == 1
    assert annotated[0].construct.sites == ()
    assert annotated[1].construct.sites == (("H", 2),)
    assert annotated[2].construct.sites == (("H", 2),)
    assert geometry["aligned_rows"] == 2
    reordered, _, _ = sources._assign_sites(
        list(reversed(records)), {"heavy": "ACDE", "light": "FGH"})
    assert {r.identity: r.construct.sites for r in reordered} == {
        r.identity: r.construct.sites for r in annotated}


def test_compact_li_rows_retain_the_individual_replicates_in_the_likelihood():
    row = replace(observation(value=8.5, uncertainty=0.70710678),
                  source="li2023_s1_a", raw={
                      "measurement_id": "POI-1",
                      "replicates": [{"replicate": 1, "log10_kd_nm": 0.0},
                                     {"replicate": 2, "log10_kd_nm": 1.0}]})
    summary, = data.summarise_observations([row])
    assert summary.point_values == (9.0, 8.0)
    assert summary.replicates == (1, 2)
    assert summary.value == 8.5
    assert all(m.uncertainty is None for m in summary.measurements)
    payload = data.measurement_payload([summary], assay_index={summary.assay: 0})
    assert payload.replicates() == 2
    np.testing.assert_allclose(payload.replicate_weight, [0.5, 0.5])


def test_reported_uncertainty_is_per_measurement_without_adding_sample_spread():
    records = [observation(value=8.0, uncertainty=0.1),
               observation(value=9.0, uncertainty=0.8)]
    summary, = data.summarise_observations(records)
    payload = data.measurement_payload([summary], assay_index={summary.assay: 0})
    np.testing.assert_allclose(payload.replicate_uncertainty, [0.1, 0.8])
    np.testing.assert_allclose(payload.replicate_weight, [0.5, 0.5])


def test_assay_vocabulary_uses_training_rows_and_includes_lazy_display_counts():
    trained = observation(assay="seen")
    withheld = observation(assay="unseen", raw={"partition": "development",
                                               "withheld_as": "study"})
    vocabulary = sources.derive_assay_vocabulary(
        [trained, withheld], additional_training_counts={"display:display:HER2": 300})
    assert set(vocabulary["assays"]) == {trained.assay_key, "display:display:HER2"}
    assert vocabulary["counts"]["display:display:HER2"] == 300
    assert withheld.assay_key not in vocabulary["counts"]


def services_context(root):
    return SimpleNamespace(
        config={"conditions": {"bin_dropout": 0.1, "assay_dropout": 0.25}},
        condition_spec=lambda: ConditionSpec(assays=("preflight_only",)),
        path=lambda *parts: root.joinpath(*parts))


def test_frozen_vocabulary_reloads_before_model_construction_and_is_immutable(tmp_path):
    from smallAntibodyGen.experiments.her2_routes_services import FlightServices
    context = services_context(tmp_path)
    first = FlightServices(context, device="cpu")
    vocabulary = {"assays": ("trained_assay",), "counts": {"trained_assay": 3}}
    first.freeze_condition_vocabulary(vocabulary)
    frozen_path = tmp_path / "condition_vocabulary.json"
    original = frozen_path.read_bytes()
    restarted = FlightServices(context, device="cpu")
    assert restarted.condition_spec.assays == ("trained_assay",)
    restarted.freeze_condition_vocabulary(vocabulary)
    assert frozen_path.read_bytes() == original
    with pytest.raises(ValueError, match="vocabulary changed"):
        restarted.freeze_condition_vocabulary(
            {"assays": ("changed_assay",), "counts": {"changed_assay": 3}})


def test_training_identity_uses_the_actual_inherited_source_snapshot(tmp_path):
    import json
    from smallAntibodyGen.experiments import her2_routes_spec as spec
    from smallAntibodyGen.experiments.her2_routes_services import FlightServices
    context = services_context(tmp_path)
    context.config_path = tmp_path / "config.json"
    context.config_path.write_text('{}')
    (tmp_path / spec.FREEZE_MARKER).write_text(json.dumps({'snapshot_sha256': 'frozen-code'}))
    services = FlightServices(context, device='cpu')
    identity = services.training_identity('test', arm='full', seed=7)
    assert identity['snapshot_sha256'] == 'frozen-code'


def test_whole_window_training_applies_both_condition_dropouts():
    summary, = data.summarise_observations([observation()])
    condition_spec = ConditionSpec(assays=(summary.assay,), bin_dropout=0.99,
                                   assay_dropout=0.99)
    views = data.whole_window_views(
        [summary], condition_spec=condition_spec,
        bins={summary.population_key: "high"}, route="masked", rng=np.random.default_rng(0))
    assert views
    condition = views[0][2]
    assert condition.family == "affinity"
    assert condition.bin == "null"
    assert condition.assay == "null"


def test_single_mutant_reference_is_not_an_arbitrary_observed_mutant():
    pairs = [("WCDE", "FGH"), ("AWDE", "FGH"), ("ACWE", "FGH"),
             ("ACDE", "WGH"), ("ACDE", "FWH")]
    parent = sources.single_mutant_consensus(pairs)
    assert (parent["heavy"], parent["light"]) == ("ACDE", "FGH")
    assert parent["reference_measured"] is False
    assert parent["count"] == 0
    with pytest.raises(ValueError, match="ambiguous"):
        sources.single_mutant_consensus([("AC", "F"), ("WC", "F")])


def test_coding_parents_translate_complete_sequence_not_amplicon_offsets(tmp_path):
    path = tmp_path / "wt.csv"
    path.write_text("gene,vh_or_vl,seq,gene_start,gene_end\n"
                    "antibody,VH,ATGTGGGCT,39,48\n"
                    "antibody,VL,TGTGATGAA,54,63\n")
    parents = sources.load_petersen_coding_parents(path)["parents"]
    assert parents == {("antibody", "VH"): "MWA", ("antibody", "VL"): "CDE"}


def test_petersen_failed_fits_do_not_become_measurements(monkeypatch, tmp_path):
    path = tmp_path / "fixture.xlsx"
    path.write_bytes(b"fixture")
    rows = [{"Variant": "parent>VH:A1W-TGG|parent>VL:WT", "Kd": "1", "Success": "0"},
            {"Variant": "parent>VH:A1W-TGG|parent>VL:WT", "Kd": "100", "Success": "1"}]
    monkeypatch.setattr(sources, "read_xlsx_sheet", lambda *_: rows)
    result = sources.load_petersen_primary(
        path, parents={("parent", "VH"): "ACDE", ("parent", "VL"): "FGH"}, sheets=("pooled",))
    assert len(result["observations"]) == 1
    assert result["observations"][0].value == 7.0
    assert result["rejected_intervals"]["failed optimizer"] == 1


def test_joint_curriculum_preserves_both_marginals_and_replay_task_support():
    from collections import Counter
    mixture = data.TaskMixture()
    schedule = mixture.joint_schedule(10000, seed=71, microbatches=16)
    tasks = Counter(task for task, _ in schedule)
    families = Counter(family for _, family in schedule)
    for task, share in mixture.shares.items():
        assert tasks[task] / len(schedule) == pytest.approx(share, abs=0.004)
    for family, share in mixture.family_mix.items():
        assert families[family] / len(schedule) == pytest.approx(share, abs=0.004)
    assert all(task == data.TASK_RECONSTRUCTION for task, family in schedule
               if family == data.FAMILY_REPLAY)
    impossible = data.TaskMixture(shares={data.TASK_RECONSTRUCTION: 0.2, data.TASK_LOCAL: 0.3,
                                         data.TASK_WHOLE: 0.3, data.TASK_HEADS: 0.2})
    with pytest.raises(ValueError, match="smaller"):
        impossible.joint_schedule(10, seed=71)


def test_explicit_torch_and_python_random_streams_resume_exactly():
    import random
    import torch
    from smallAntibodyGen.experiments.her2_routes_checkpoints import RngBundle
    generator = torch.Generator().manual_seed(182)
    random.seed(913)
    bundle = RngBundle.capture({"energy_rng": generator})
    expected_tensor, expected_scalar = torch.rand(7, generator=generator), random.random()
    _ = torch.rand(13, generator=generator), random.random()
    bundle.restore({"energy_rng": generator})
    assert torch.equal(torch.rand(7, generator=generator), expected_tensor)
    assert random.random() == expected_scalar


def test_inference_load_preserves_rng_and_training_refuses_missing_optimizer(tmp_path):
    import random
    import torch
    from smallAntibodyGen.experiments import her2_routes_checkpoints as checkpoints
    model = torch.nn.Linear(3, 2)
    state = checkpoints.TrainingState()
    path = tmp_path / "weights.pt"
    identity = checkpoints.training_identity(route="fixture", arm="full", seed=29,
        config_sha256="config", snapshot_sha256="source", external_sha256="external",
        condition_sha256="condition", split_sha256="assignments")
    checkpoints.save(path, model=model, heads=None, optimizer=None, scheduler=None,
                     state=state, identity=identity)
    before_torch, before_python = torch.get_rng_state(), random.getstate()
    checkpoints.load_inference(path, model=model, heads=None, identity=identity)
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="optimizer state"):
        checkpoints.restore(path, model=model, heads=None, optimizer=optimizer,
                            scheduler=None, state=state, identity=identity)


def test_progressive_masks_keep_every_unrevealed_residue_hidden():
    from smallAntibodyGen.experiments import her2_routes_serial as serial
    summary, = data.summarise_observations([observation()])
    summary.region = (("H", 0), ("H", 1), ("H", 2), ("H", 3))
    condition_spec = ConditionSpec(assays=(summary.assay,))
    for seed in range(12):
        view, = data.whole_window_views(
            [summary], condition_spec=condition_spec, bins={}, route="masked",
            rng=np.random.default_rng(seed), progressive_probability=1.0)
        construct, serialized, _ = view
        assert 0 < len(serialized.order) < len(summary.region)
        assert all(serialized.tokens[index] == serial.ABLANG_MASK_ID
                   for index in serialized.query_slots)
        assert tuple(serialized.order) == summary.region[-len(serialized.order):]
