"""End-to-end runner tests on software backends: budget, run root, status, report and held-out hygiene.

Every run here uses an injected SOFTWARE backend through ``run_with_backend``. None of it runs the released
model, and the statuses these tests assert say exactly that.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import types
from pathlib import Path

import pytest
import torch  # noqa: F401  (the runner tests need torch for the software backends)

from smallAntibodyGen.experiments.interaction_qualification import diagnostics as dg
from smallAntibodyGen.experiments.interaction_qualification import runner
from smallAntibodyGen.tests import aa_qualification_world as world
from smallAntibodyGen.tests.aa_qualification_diagnostics_world import drow
from smallAntibodyGen.tests.aa_world import RecordingBackend, ToyBackend

NATIVE_SOURCE = os.environ.get("SMALLANTIBODYGEN_MINT_SOURCE")
needs_native = pytest.mark.skipif(
    not NATIVE_SOURCE, reason="SMALLANTIBODYGEN_MINT_SOURCE is not set; the tiny actual-native run is skipped"
)

SCIENTIFIC = {
    "antigen_specificity_of_joint_gap": "unestablished",
    "homology_leakage": "unverified",
    "upstream_pretraining_exclusion": "not_audited",
    "generative_success": "unestablished",
    "scientific_qualification": "not_assessed",
}
ARMS = ("joint", "antibody_only", "independent")


def make_cfg(**kwargs):
    return world.config_dict(
        **{
            "partitions": ["development"],
            "mask_counts": [1, 2, "full"],
            "geometries": ["suffix", "subset"],
            "max_examples": 4,
            "local_gap": {"enabled": True, "max_pairs": 4, "kappa": 1.0},
            **kwargs,
        }
    )


def standard_rows(shift=0.0, region="heavy"):
    """Train (4), development (3) and sealed test (2) rows; antibodies and antigens never cross partitions."""
    rows = [
        drow(f"train-{h}", h, partition="train", antigen="EFGK", value=6.0 + 0.5 * i, region=region)
        for i, h in enumerate(("ACDE", "ACDF", "ACGE", "WCDE"))
    ]
    rows += [
        drow(f"dev-{h}", h, partition="development", antigen="LMNP", value=7.0 + 0.4 * i + shift, region=region)
        for i, h in enumerate(("ACHI", "ACHK", "MCHI"))
    ]
    rows += [
        drow(f"test-{h}", h, partition="test", antigen="QRST", value=5.0 + 0.3 * i + shift, region=region)
        for i, h in enumerate(("WWWW", "WWWY"))
    ]
    return rows


class Factory:
    """A backend factory that counts how often it was called."""

    def __init__(self, backend=None):
        self.backend, self.calls = backend, 0

    def __call__(self):
        self.calls += 1
        return ToyBackend(seed=0) if self.backend is None else self.backend


def setup(tmp_path, rows, config=None, *, name="in"):
    config_path, manifest_path = world.write_inputs(tmp_path / name, rows, make_cfg() if config is None else config)
    code_root = tmp_path / "code"
    code_root.mkdir(exist_ok=True)
    (tmp_path / "out").mkdir(exist_ok=True)
    return runner.preflight(config_path, manifest_path), code_root


def go(tmp_path, rows, config=None, *, name="in", run="run", factory=None):
    pf, code_root = setup(tmp_path, rows, config, name=name)
    factory = Factory() if factory is None else factory
    result = runner.run_with_backend(pf, factory, run_root=tmp_path / "out" / run, code_root=code_root)
    return pf, result, factory


def snapshot(root):
    return sorted(str(path) for path in Path(root).rglob("*"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def refuse_constant(name):
    raise AssertionError(f"non-finite constant {name} in a written file")


# --- a complete software run ---------------------------------------------------------------------------------


def test_a_software_run_writes_three_lf_json_files_and_fixed_scientific_statuses(tmp_path):
    pf, result, factory = go(tmp_path, standard_rows())
    root = Path(result["run_root"])
    assert sorted(path.name for path in root.iterdir()) == sorted(runner.RUN_FILES) and factory.calls == 1
    for name in runner.RUN_FILES:
        data = (root / name).read_bytes()
        assert b"\r" not in data and data.endswith(b"\n")
        json.loads(data.decode("utf-8"), parse_constant=refuse_constant)
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed" and status["report_sha256"] == sha((root / "report.json").read_bytes())
    assert status["plan_sha256"] == sha((root / "plan.json").read_bytes()) == report["plan_summary"]["plan_sha256"]
    assert status["scientific_status"] == report["scientific_status"] == SCIENTIFIC
    execution = status["execution"]
    assert execution["software_test"] is True and execution["origin"] == "software_test"
    assert execution["released_execution"] == "not_run" and execution["released_weights_forward_completed"] is False
    assert "SOFTWARE TEST" in execution["description"] and execution["backend_constructed"] is True
    assert execution["forward_calls"] == pf.plan.budget.total > 0


def test_the_report_carries_provenance_support_and_probe_sections(tmp_path):
    pf, result, _ = go(tmp_path, standard_rows())
    report = result["report"]
    assert report["schema"] == "aa-qualification-report/1" and report["mode"] == "full_run"
    assert report["manifest"]["manifest_file_sha256"] == sha(pf.manifest_path.read_bytes())
    assert report["manifest"]["records_file_sha256"] == sha(pf.records_path.read_bytes())
    assert report["config_file_sha256"] == sha(pf.config_path.read_bytes())
    assert report["config"] == pf.config.as_dict() and report["config_content_sha256"] == pf.config.content_sha256()
    (source,) = report["manifest"]["sources"]
    assert source["citation"] and source["assays"][0]["declaration"]
    assert report["manifest"]["split_basis_status"] == "declared" and report["manifest"]["split_regime"] == "both"
    listed = {entry["record_id"]: entry for entry in report["support_report"]["records"]}
    assert set(listed) == {row["record_id"] for row in standard_rows()} and all(e["content_sha256"] for e in listed.values())
    assert report["sealed_test"] == {"evaluate_test": False, "records_withheld": 2}
    assert report["active_support"]["by_partition"]["test"]["n_records"] == 0
    assert report["active_support"]["by_partition"]["train"]["n_records"] == 4
    assert report["masked_support"]["singleton"]["tasks"] == len(pf.plan.singleton_tasks) == 3
    assert report["masked_support"]["pair"]["tasks"] == len(pf.plan.pair_tasks) > 0
    assert report["forward_budget"]["total_forwards"] == report["forward_budget_actual"]["total"]
    assert report["forward_budget_actual"]["matches_plan"] is True
    assert report["backend"] == {"constructed": True, "describe": {"backend": "ToyBackend"}}
    assert report["runtime"]["python"] and report["limitations"] and "DECLARED" in report["limitations"][0]
    probes = report["probes"]
    assert probes["status"] == "fitted" and set(probes["arms"]) == set(ARMS)
    assert {arm: entry["n_records"] for arm, entry in probes["arm_report"]["arms"].items()} == {arm: 7 for arm in ARMS}
    assert {arm: probes["arms"][arm]["width"] for arm in ARMS} == {"joint": 16, "antibody_only": 8, "independent": 16}
    fit = probes["arms"]["joint"]["fit"]
    assert fit["n_train"] == 4 and len(fit["coefficients"]) == 16
    assert {g["partition"] for g in probes["arms"]["joint"]["groups"]} == {"train", "development"}
    diagnostics = report["diagnostics"]
    assert diagnostics["singleton"]["status"] == "computed" and diagnostics["pair"]["status"] == "computed"
    assert diagnostics["singleton"]["summary"]["antigen_specificity_of_joint_gap"] == "unestablished"
    assert diagnostics["pair"]["summary"]["unique_pair_tasks"] == len(pf.plan.pair_tasks)


def test_two_identical_runs_write_byte_identical_reports(tmp_path):
    _, one, _ = go(tmp_path, standard_rows(), run="first")
    _, two, _ = go(tmp_path, standard_rows(), run="second")
    assert one["status"]["report_sha256"] == two["status"]["report_sha256"]


# --- budget before anything exists ------------------------------------------------------------------------------


def test_a_cap_of_232_rejects_before_any_backend_or_directory_and_233_runs_exactly_233(tmp_path):
    rows = [drow("a", "ACDE"), drow("b", "ACFG")]

    def config(cap):
        return make_cfg(mask_counts=[1, 2, 3, "full"], max_examples=8, max_forwards=cap)

    pf, code_root = setup(tmp_path, rows, config(232), name="in232")
    budget = pf.plan.budget
    assert (budget.feature, budget.singleton, budget.pair, budget.total) == (5, 212, 16, 233)
    factory = Factory()
    with pytest.raises(dg.BudgetExceeded):
        runner.run_with_backend(pf, factory, run_root=tmp_path / "out" / "r232", code_root=code_root)
    assert factory.calls == 0 and list((tmp_path / "out").iterdir()) == []
    pf, _ = setup(tmp_path, rows, config(233), name="in233")
    spy = RecordingBackend(ToyBackend(seed=0))
    result = runner.run_with_backend(pf, Factory(spy), run_root=tmp_path / "out" / "r233", code_root=code_root)
    assert len(spy.views) == 233 == result["report"]["forward_budget_actual"]["total"]
    assert result["report"]["forward_budget"]["max_forwards"] == 233 and result["report"]["forward_budget"]["within_budget"]


def test_an_understated_plan_is_refused_as_an_overrun_and_an_overstated_one_as_a_mismatch(tmp_path):
    pf, code_root = setup(tmp_path, standard_rows())
    budget = pf.plan.budget
    for name, delta, error in (("under", -1, dg.BudgetOverrun), ("over", +1, dg.BudgetMismatch)):
        plan = dataclasses.replace(pf.plan, budget=dataclasses.replace(budget, singleton=budget.singleton + delta))
        with pytest.raises(error):
            runner.run_with_backend(
                dataclasses.replace(pf, plan=plan), Factory(), run_root=tmp_path / "out" / name, code_root=code_root
            )
        status = json.loads((tmp_path / "out" / name / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "error" and status["error"]["type"] == error.__name__
        assert not (tmp_path / "out" / name / "report.json").exists()


def test_the_admission_report_prices_the_plan_and_writes_nothing(tmp_path):
    rows = standard_rows() + [drow("long", "ACDE", partition="train", antigen="K" * 40)]
    pf, _ = setup(tmp_path, rows, make_cfg(max_tokens=40))
    before = snapshot(tmp_path)
    report = runner.admission_report(pf)
    assert snapshot(tmp_path) == before
    json.loads(runner.canonical_text(report))
    assert report["schema"] == "aa-admission/1" and report["mode"] == "admission_only" and report["files_written"] == 0
    assert report["shared_length_admission"]["excluded"] == [{"record_id": "long", "joint_serialized_length": 48}]
    assert report["shared_length_admission"]["n_excluded"] == 1 and "long" not in {r.record_id for r in pf.eligible}
    assert report["sealed_test"]["records_withheld"] == 2 and report["scientific_status"] == SCIENTIFIC
    assert report["forward_budget"]["total_forwards"] == pf.plan.budget.total and report["forward_budget"]["within_budget"]
    assert report["backend"].startswith("not constructed")


# --- honest degenerate panels -----------------------------------------------------------------------------------


def test_an_empty_panel_is_a_valid_honest_run_with_no_calls(tmp_path):
    pf, result, factory = go(tmp_path, [])
    report = result["report"]
    assert pf.plan.budget.total == 0 and factory.calls == 0
    assert report["backend"] == {"constructed": False, "reason": "no forward was planned"}
    assert report["probes"]["status"] == "no_support" and report["probes"]["arms"] == {}
    for kind in ("singleton", "pair"):
        assert report["diagnostics"][kind]["status"] == "no_support"
    assert report["diagnostics"]["singleton"]["cells"] == [] and report["diagnostics"]["singleton"]["summary"]["groups"] == []
    assert report["support_report"]["no_support"] is True and report["forward_budget_actual"]["total"] == 0
    assert result["status"]["state"] == "completed" and result["status"]["execution"]["released_execution"] == "not_run"


def test_without_training_rows_no_probe_is_fitted_but_the_diagnostics_still_run(tmp_path):
    rows = [row for row in standard_rows() if row["partition"] == "development"]
    _, result, _ = go(tmp_path, rows)
    report = result["report"]
    assert report["probes"]["status"] == "no_training_support" and report["probes"]["arms"] == {}
    assert "no probe was fitted" in report["probes"]["reason"] and report["probes"]["arm_report"] is not None
    assert report["diagnostics"]["singleton"]["status"] == "computed" and report["diagnostics"]["singleton"]["cells"]


def test_records_without_a_region_feed_the_probes_but_not_the_masked_diagnostics(tmp_path):
    pf, result, _ = go(tmp_path, standard_rows(region=None))
    report = result["report"]
    assert report["probes"]["status"] == "fitted"
    assert report["diagnostics"]["singleton"]["status"] == "no_support" and report["diagnostics"]["pair"]["status"] == "no_support"
    assert pf.plan.singleton_stats["records_missing_region"] == 3
    assert report["forward_budget"]["singleton_forwards"] == 0 == report["forward_budget"]["pair_forwards"]
    assert report["forward_budget_actual"]["total"] == report["forward_budget"]["feature_forwards"] > 0


# --- held-out hygiene ---------------------------------------------------------------------------------------------


def test_held_out_labels_cannot_change_the_fit_or_any_prediction_and_the_sealed_test_is_never_forwarded(tmp_path):
    spy_one, spy_two, spy_three = (RecordingBackend(ToyBackend(seed=0)) for _ in range(3))
    _, one, _ = go(tmp_path, standard_rows(), name="in1", run="r1", factory=Factory(spy_one))
    _, two, _ = go(tmp_path, standard_rows(shift=3.0), name="in2", run="r2", factory=Factory(spy_two))
    a, b = one["report"]["probes"], two["report"]["probes"]
    for arm in ARMS:
        assert a["arms"][arm]["fit"] == b["arms"][arm]["fit"]
        assert a["arms"][arm]["predictions"] == b["arms"][arm]["predictions"] and a["arms"][arm]["predictions"]
        by_partition = lambda p, arm=arm: {g["partition"]: g for g in p["arms"][arm]["groups"]}  # noqa: E731
        assert by_partition(a)["train"] == by_partition(b)["train"]
        assert by_partition(a)["development"]["rmse"] != by_partition(b)["development"]["rmse"]
    chains = {chain for view in spy_one.views for chain in view.chains}
    assert tuple("WWWW") not in chains and tuple("WWWY") not in chains and tuple("QRST") not in chains
    assert len(spy_one.views) == len(spy_two.views) == one["report"]["forward_budget_actual"]["total"]
    unsealed = make_cfg(evaluate_test=True, partitions=["development", "test"])
    _, three, _ = go(tmp_path, standard_rows(), unsealed, name="in3", run="r3", factory=Factory(spy_three))
    assert tuple("QRST") in {chain for view in spy_three.views for chain in view.chains}
    assert {g["partition"] for g in three["report"]["probes"]["arms"]["joint"]["groups"]} == {"train", "development", "test"}
    assert three["report"]["sealed_test"] == {"evaluate_test": True, "records_withheld": 0}
    assert {c["partition"] for c in three["report"]["diagnostics"]["singleton"]["cells"]} == {"development", "test"}


def test_the_masked_diagnostics_never_run_an_unmasked_view_and_features_never_run_a_masked_one(tmp_path):
    spy = RecordingBackend(ToyBackend(seed=0))
    pf, _, _ = go(tmp_path, standard_rows(), factory=Factory(spy))
    feature_calls = pf.plan.budget.feature
    assert all(not view.hidden_sites() for view in spy.views[:feature_calls])
    assert all(view.hidden_sites() for view in spy.views[feature_calls:])
    assert len(spy.views) - feature_calls == pf.plan.budget.diagnostics


# --- seam, production entry, errors -------------------------------------------------------------------------------


def test_the_production_entry_has_no_injection_and_the_software_seam_is_explicitly_named():
    production = set(inspect.signature(runner.run_qualification).parameters)
    assert not production & {"backend", "backend_factory", "factory", "origin", "software_test"}
    assert {"mint_source", "mint_checkpoint", "device", "code_root"} <= production
    assert "backend_factory" in inspect.signature(runner.run_with_backend).parameters
    assert inspect.signature(runner.run_with_backend).parameters["origin"].default == runner.SOFTWARE_TEST


def test_an_unknown_origin_is_refused_before_anything_else(tmp_path):
    pf, code_root = setup(tmp_path, [])
    with pytest.raises(ValueError, match="origin"):
        runner.run_with_backend(pf, Factory(), run_root=tmp_path / "out" / "run", code_root=code_root, origin="released")
    assert list((tmp_path / "out").iterdir()) == []


def _fake_release_inputs(tmp_path):
    source = tmp_path / "mint-source"
    source.mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = weights / "mint.ckpt"
    checkpoint.write_bytes(b"")
    return source, checkpoint


def test_the_production_path_refuses_a_loader_result_that_is_not_the_pinned_release(tmp_path, monkeypatch):
    from smallAntibodyGen.models.antibody_antigen import mint

    monkeypatch.setattr(mint, "load_pinned_mint", lambda *args, **kwargs: ToyBackend(seed=0))
    source, checkpoint = _fake_release_inputs(tmp_path)
    config_path, manifest_path = world.write_inputs(tmp_path / "in", standard_rows(), make_cfg())
    code_root = tmp_path / "code"
    code_root.mkdir()
    (tmp_path / "out").mkdir()
    run = tmp_path / "out" / "run"
    with pytest.raises(ValueError, match="pinned release"):
        runner.run_qualification(
            config_path, manifest_path, run, mint_source=source, mint_checkpoint=checkpoint, code_root=code_root
        )
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error" and status["error"]["type"] == "ValueError"
    assert status["execution"]["origin"] == "pinned_release" and status["execution"]["released_execution"] == "not_run"
    assert status["execution"]["released_weights_forward_completed"] is False
    assert not (run / "report.json").exists()


def test_a_production_run_that_needs_no_forward_never_loads_the_model_and_claims_no_release_run(tmp_path, monkeypatch):
    from smallAntibodyGen.models.antibody_antigen import mint

    def refuse(*args, **kwargs):
        raise AssertionError("the loader must not be called when no forward is planned")

    monkeypatch.setattr(mint, "load_pinned_mint", refuse)
    source, checkpoint = _fake_release_inputs(tmp_path)
    config_path, manifest_path = world.write_inputs(tmp_path / "in", [], make_cfg())
    code_root = tmp_path / "code"
    code_root.mkdir()
    (tmp_path / "out").mkdir()
    result = runner.run_qualification(
        config_path, manifest_path, tmp_path / "out" / "run", mint_source=source, mint_checkpoint=checkpoint,
        code_root=code_root,
    )  # fmt: skip
    execution = result["status"]["execution"]
    assert execution["origin"] == "pinned_release" and execution["software_test"] is False
    assert execution["backend_constructed"] is False and execution["forward_calls"] == 0
    assert execution["released_weights_forward_completed"] is False and execution["released_execution"] == "not_run"
    assert "no forward was planned" in execution["description"]


def test_the_production_entry_needs_existing_explicit_local_paths(tmp_path):
    config_path, manifest_path = world.write_inputs(tmp_path / "in", [], make_cfg())
    (tmp_path / "out").mkdir()
    code_root = tmp_path / "code"
    code_root.mkdir()
    source, checkpoint = _fake_release_inputs(tmp_path)
    for bad_source, bad_checkpoint in ((tmp_path / "nope", checkpoint), (source, tmp_path / "nope.ckpt"), (source, source)):
        with pytest.raises(ValueError, match="mint"):
            runner.run_qualification(
                config_path, manifest_path, tmp_path / "out" / "run", mint_source=bad_source,
                mint_checkpoint=bad_checkpoint, code_root=code_root,
            )  # fmt: skip
    with pytest.raises(ValueError, match="device"):
        runner.run_qualification(
            config_path, manifest_path, tmp_path / "out" / "run", mint_source=source, mint_checkpoint=checkpoint,
            device="tpu", code_root=code_root,
        )  # fmt: skip
    assert list((tmp_path / "out").iterdir()) == []


def test_a_failing_backend_factory_leaves_an_error_status_a_plan_and_no_report(tmp_path):
    def broken():
        raise RuntimeError("loader exploded")

    with pytest.raises(RuntimeError, match="loader exploded"):
        go(tmp_path, standard_rows(), factory=broken)
    root = tmp_path / "out" / "run"
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error" and status["error"] == {"type": "RuntimeError", "message": "loader exploded"}
    assert (root / "plan.json").is_file() and not (root / "report.json").exists()
    assert status["execution"]["released_execution"] == "not_run" and status["execution"]["backend_constructed"] is False


# --- run root ---------------------------------------------------------------------------------------------------


@pytest.fixture
def scene(tmp_path):
    config_path, manifest_path = world.write_inputs(tmp_path / "in", standard_rows(), make_cfg())
    code = tmp_path / "code"
    code.mkdir()
    source, checkpoint = _fake_release_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "empty").mkdir()
    (out / "full").mkdir()
    (out / "full" / "keep.txt").write_text("keep", encoding="utf-8")
    return types.SimpleNamespace(
        root=tmp_path, inputs=tmp_path / "in", code=code, source=source, weights=checkpoint.parent, out=out,
        pf=runner.preflight(config_path, manifest_path),
        extra={"the MINT source": source, "the MINT checkpoint directory": checkpoint.parent},
    )  # fmt: skip


BAD_ROOTS = {
    "inside the input directory": lambda s: s.inputs / "run",
    "inside the code root": lambda s: s.code / "run",
    "inside the MINT source": lambda s: s.source / "run",
    "inside the checkpoint directory": lambda s: s.weights / "run",
    "an existing empty directory": lambda s: s.out / "empty",
    "an existing non-empty directory": lambda s: s.out / "full",
    "an ancestor of the inputs": lambda s: s.root,
    "the input directory itself": lambda s: s.inputs,
    "a relative path": lambda s: Path("relative-qualification-run"),
    "a missing parent": lambda s: s.out / "missing" / "run",
}


@pytest.mark.parametrize("label", sorted(BAD_ROOTS))
def test_a_bad_run_root_is_refused_before_the_backend_or_any_write(scene, label):
    before, factory = snapshot(scene.root), Factory()
    target = BAD_ROOTS[label](scene)
    with pytest.raises(ValueError):
        runner.run_with_backend(scene.pf, factory, run_root=target, code_root=scene.code, extra_protected=scene.extra)
    assert factory.calls == 0 and snapshot(scene.root) == before
    assert not Path("relative-qualification-run").exists()


def test_a_symlink_into_an_input_directory_is_resolved_before_the_run_root_is_accepted(scene):
    link = scene.out / "link"
    try:
        os.symlink(scene.inputs, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links cannot be created here")
    before, factory = snapshot(scene.root), Factory()
    for target in (link, link / "run"):
        with pytest.raises(ValueError):
            runner.run_with_backend(scene.pf, factory, run_root=target, code_root=scene.code, extra_protected=scene.extra)
    assert factory.calls == 0 and snapshot(scene.root) == before


def test_a_missing_protected_input_is_an_error_not_an_absent_overlap(scene):
    extra = dict(scene.extra, **{"the MINT source": scene.out / "no-such-source"})
    with pytest.raises(ValueError, match="do not exist"):
        runner.run_with_backend(
            scene.pf, Factory(), run_root=scene.out / "run", code_root=scene.code, extra_protected=extra
        )
    assert not (scene.out / "run").exists()


def test_a_disjoint_fresh_run_root_is_accepted_and_inputs_are_left_untouched(scene):
    inputs_before = {p.name: p.read_bytes() for p in scene.inputs.iterdir()}
    result = runner.run_with_backend(
        scene.pf, Factory(), run_root=scene.out / "run", code_root=scene.code, extra_protected=scene.extra
    )
    assert Path(result["run_root"]) == scene.out / "run"
    assert {p.name: p.read_bytes() for p in scene.inputs.iterdir()} == inputs_before
    with pytest.raises(ValueError, match="already exists"):  # no overwrite, ever
        runner.run_with_backend(
            scene.pf, Factory(), run_root=scene.out / "run", code_root=scene.code, extra_protected=scene.extra
        )


def test_non_finite_numbers_are_refused_before_a_file_exists(tmp_path):
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            runner.write_json_lf(tmp_path / "x.json", {"value": bad})
    assert list(tmp_path.iterdir()) == []
    digest = runner.write_json_lf(tmp_path / "ok.json", {"b": 1, "a": [1.5, None]})
    data = (tmp_path / "ok.json").read_bytes()
    assert data == b'{\n  "a": [\n    1.5,\n    null\n  ],\n  "b": 1\n}\n' and digest == sha(data)
    assert [p.name for p in tmp_path.iterdir()] == ["ok.json"]


# --- the tiny ACTUAL native model (software test only) ---------------------------------------------------------------


@needs_native
def test_a_tiny_actual_native_model_runs_the_whole_pipeline_as_a_software_test(tmp_path):
    from smallAntibodyGen.models.antibody_antigen import mint

    package = mint.load_native_package(Path(NATIVE_SOURCE))
    try:
        torch.manual_seed(1234)
        tiny = package.ESM2(
            num_layers=2, embed_dim=16, attention_heads=4, alphabet="ESM-1b", token_dropout=True, use_multimer=True
        )
        pf, result, factory = go(tmp_path, standard_rows(), factory=Factory(mint.MintBackend(tiny)))
        report = result["report"]
        assert report["backend"]["describe"]["release"].startswith("SOFTWARE TEST MODEL")
        assert report["backend"]["describe"]["code_commit"] is None
        execution = result["status"]["execution"]
        assert execution["software_test"] is True and execution["released_execution"] == "not_run"
        assert report["forward_budget_actual"]["total"] == pf.plan.budget.total and factory.calls == 1
        assert report["probes"]["status"] == "fitted" and report["diagnostics"]["singleton"]["cells"]
    finally:
        mint._forget(package.__name__)
