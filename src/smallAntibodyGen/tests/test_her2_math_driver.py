"""The numerical amendment must not reuse evidence from a different backend."""
import importlib.util
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.experiments.her2_runtime import load_json, save_json, sha256


@pytest.fixture
def driver(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("her2_math_driver", scripts / "evaluate_her2_math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(module, "expected_manifest", lambda *args: {"backend": "math", "hash": "a"})
    run = tmp_path / "run"
    (run / "continuation").mkdir(parents=True)
    save_json(run / "continuation/continuation_results.json", {"runs": {"unchanged": True}})
    return module, run


def test_evaluation_requires_same_manifest_bound_before_outcome_access(driver, monkeypatch):
    module, run = driver
    calls = []

    def validate(config, output, **kwargs):
        assert kwargs["stages"] == ("validate", "freeze")
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()
        assert load_json(output / "continuation_results.json")["runs"] == {"unchanged": True}
        save_json(output / "selection_frozen.json", {"stage": "final", "selected": {"policy": "a"}})
        calls.append("validate")

    def evaluate(config, output, **kwargs):
        freeze = load_json(kwargs["freeze_path"])
        assert freeze["numerical_evaluation"]["sha256"] == sha256(
            run / "validation_math/numerical_backend.json")
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled()
        save_json(output / "results.json", {"status": "completed", "selection_freeze_sha256":
                                           sha256(kwargs["freeze_path"])})
        calls.append("outcomes")

    monkeypatch.setattr(module.posttrain, "run", validate)
    monkeypatch.setattr(module.final_evaluation, "run", evaluate)
    module.run_stage("validate", run / "config.json", run)
    manifest = run / "validation_math/numerical_backend.json"
    original = load_json(manifest)
    save_json(manifest, dict(original, backend="automatic"))
    with pytest.raises(ValueError, match="identity changed"):
        module.run_stage("evaluate", run / "config.json", run)
    assert calls == ["validate"]
    save_json(manifest, original)
    module.run_stage("evaluate", run / "config.json", run)
    assert calls == ["validate", "outcomes"]
    assert load_json(run / "evaluation_math/results.json")["numerical_evaluation"]["sha256"] == sha256(manifest)
    with pytest.raises(ValueError, match="validation is now fixed"):
        module.run_stage("validate", run / "config.json", run)


def test_unidentified_validation_directory_is_not_adopted(driver):
    module, run = driver
    (run / "validation_math").mkdir()
    with pytest.raises(ValueError, match="without numerical provenance"):
        module.run_stage("validate", run / "config.json", run)


def test_freeze_cannot_be_rebound_to_another_backend(driver):
    module, run = driver
    freeze = run / "selection_frozen.json"
    manifest = run / "numerical_backend.json"
    save_json(manifest, {"backend": "math"})
    save_json(freeze, {"stage": "final", "selected": {"p": 1},
                       "numerical_evaluation": {"sha256": "other", "path": "other"}})
    original = sha256(freeze)
    with pytest.raises(ValueError, match="different manifest"):
        module.bind_freeze(freeze, manifest)
    assert sha256(freeze) == original
