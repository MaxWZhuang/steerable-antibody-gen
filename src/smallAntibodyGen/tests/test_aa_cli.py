"""The qualification command line, run as a subprocess behind an import blocker.

Nothing here runs the released model: the only full-run path exercised is a refusal at the pinned source
check, on a fake source directory that cannot match the pins.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from smallAntibodyGen.experiments.interaction_qualification.config import load_config
from smallAntibodyGen.tests import aa_qualification_world as world
from smallAntibodyGen.tests.aa_qualification_diagnostics_world import drow

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPT = ROOT / "scripts" / "qualify_antibody_antigen.py"
EXAMPLE = ROOT / "configs" / "experiments" / "antibody_antigen_qualification.example.json"

LAUNCHER = '''\
import runpy
import sys

blocked = {name for name in sys.argv[1].split(",") if name}
script = sys.argv[2]


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked:
            raise ImportError("blocked by the test: " + name)
        return None


sys.meta_path.insert(0, Block())
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
'''

LIBRARY_PROBE = """\
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            raise ImportError("blocked by the test: " + name)
        return None

sys.meta_path.insert(0, Block())
sys.path.insert(0, {src!r})
from smallAntibodyGen.experiments.interaction_qualification import arms, config, diagnostics, probes, records, runner
assert "torch" not in sys.modules
print("ok")
"""


@pytest.fixture(scope="module")
def launcher(tmp_path_factory):
    path = tmp_path_factory.mktemp("launcher") / "launch.py"
    path.write_text(LAUNCHER, encoding="utf-8")
    return path


def run_cli(launcher, blocked, args, cwd):
    return subprocess.run(
        [sys.executable, "-I", "-B", str(launcher), ",".join(blocked), str(SCRIPT), *map(str, args)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=900,
    )


def snapshot(root):
    return sorted(str(path) for path in Path(root).rglob("*"))


def rows():
    return [
        drow("t1", "ACDE", partition="train", antigen="EFGK", value=6.0),
        drow("t2", "ACDF", partition="train", antigen="EFGK", value=6.5),
        drow("d1", "ACHI", partition="development", antigen="LMNP", value=7.0),
        drow("d2", "ACHK", partition="development", antigen="LMNP", value=7.5),
        drow("x1", "WWWW", partition="test", antigen="QRST", value=5.0),
    ]


def config(**kwargs):
    return world.config_dict(mask_counts=[1, 2, "full"], geometries=["suffix", "subset"], **kwargs)


def make_inputs(tmp_path, panel_rows=None, cfg=None):
    config_path, manifest_path = world.write_inputs(tmp_path / "in", rows() if panel_rows is None else panel_rows, cfg or config())
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    return config_path, manifest_path, cwd


# --- help ----------------------------------------------------------------------------------------------------


def test_help_is_stdlib_only_creates_nothing_and_offers_no_fake_production_mode(launcher, tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    result = run_cli(launcher, ("torch", "numpy", "scipy", "pandas"), ["--help"], cwd)
    assert result.returncode == 0, result.stderr
    for flag in ("--config", "--manifest", "--run-root", "--mint-source", "--mint-checkpoint", "--device", "--admission-only"):
        assert flag in result.stdout
    for invented in ("--backend", "--software", "--fake", "--download", "--production", "--synthetic", "--tiny"):
        assert invented not in result.stdout
    assert list(cwd.iterdir()) == []


def test_the_library_modules_import_without_torch():
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", LIBRARY_PROBE.format(src=str(SRC))], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# --- admission-only --------------------------------------------------------------------------------------------


def test_admission_only_needs_no_torch_writes_nothing_and_reports_the_priced_plan(launcher, tmp_path):
    config_path, manifest_path, cwd = make_inputs(tmp_path)
    before = snapshot(tmp_path)
    result = run_cli(
        launcher,
        ("torch",),
        ["--config", config_path, "--manifest", manifest_path, "--admission-only", "--run-root", tmp_path / "out" / "run"],
        cwd,
    )
    assert result.returncode == 0, result.stderr
    assert snapshot(tmp_path) == before and not (tmp_path / "out").exists()
    report = json.loads(result.stdout)
    assert report["schema"] == "aa-admission/1" and report["mode"] == "admission_only" and report["files_written"] == 0
    assert report["manifest"]["manifest_file_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert report["manifest"]["records_file_sha256"] == hashlib.sha256((tmp_path / "in" / "panel.jsonl").read_bytes()).hexdigest()
    assert report["sealed_test"] == {"evaluate_test": False, "records_withheld": 1}
    assert report["forward_budget"]["within_budget"] is True and report["forward_budget"]["total_forwards"] > 0
    assert report["scientific_status"]["scientific_qualification"] == "not_assessed"
    assert report["support_report"]["split_checks"]["homology_separation"] == "not_audited"
    assert report["plan_summary"]["singleton_tasks"] == 2


def test_admission_only_over_budget_prints_the_numbers_and_exits_2(launcher, tmp_path):
    config_path, manifest_path, cwd = make_inputs(tmp_path, cfg=config(max_forwards=10))
    result = run_cli(launcher, ("torch",), ["--config", config_path, "--manifest", manifest_path, "--admission-only"], cwd)
    assert result.returncode == 2 and result.stderr.startswith("refused:") and "exceed" in result.stderr
    report = json.loads(result.stdout)
    assert report["forward_budget"]["within_budget"] is False and report["forward_budget"]["max_forwards"] == 10
    assert "Traceback" not in result.stderr


def _bad_manifest(inputs, **_):
    (inputs / "manifest.json").write_text("{not json", encoding="utf-8")


def _bad_hash(inputs, **_):
    data = (inputs / "panel.jsonl").read_bytes()
    (inputs / "panel.jsonl").write_bytes(data + b"\n")


def _bad_config(inputs, **_):
    document = json.loads((inputs / "config.json").read_text(encoding="utf-8"))
    document["extra_key"] = 1
    (inputs / "config.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("damage", [_bad_manifest, _bad_hash, _bad_config])
def test_malformed_inputs_fail_cleanly_with_no_traceback_and_no_output(launcher, tmp_path, damage):
    config_path, manifest_path, cwd = make_inputs(tmp_path)
    damage(tmp_path / "in")
    result = run_cli(launcher, ("torch",), ["--config", config_path, "--manifest", manifest_path, "--admission-only"], cwd)
    assert result.returncode == 2 and result.stderr.startswith("refused:")
    assert "Traceback" not in result.stderr and result.stdout == ""


def test_a_split_leak_is_refused_cleanly(launcher, tmp_path):
    leaky = [
        drow("t", "ACDE", partition="train", antigen="EFGK"),
        drow("d", "ACDE", partition="development", antigen="LMNP"),  # the same antibody on both sides
    ]
    config_path, manifest_path, cwd = make_inputs(tmp_path, leaky)
    result = run_cli(launcher, ("torch",), ["--config", config_path, "--manifest", manifest_path, "--admission-only"], cwd)
    assert result.returncode == 2 and "shared across partitions" in result.stderr and result.stdout == ""


# --- example config ---------------------------------------------------------------------------------------------


def _strings(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def test_the_example_config_is_the_exact_existing_schema_with_the_documented_defaults():
    loaded = load_config(EXAMPLE)  # the closed schema rejects any extra, missing or coerced key
    assert loaded.evaluate_test is False and "test" not in loaded.diagnostics.partitions
    assert loaded.diagnostics.max_forwards == 20000 and loaded.diagnostics.max_examples == 32
    assert loaded.diagnostics.local_gap.max_pairs == 32 and loaded.diagnostics.local_gap.enabled is True
    document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert document["schema"] == "aa-qualification/1"
    assert [s for s in _strings(document) if ("/" in s or "\\" in s) and s != document["schema"]] == []
    assert loaded.as_dict() == document


# --- the full command line ---------------------------------------------------------------------------------------


def test_a_full_run_needs_every_explicit_local_path_and_a_known_device(launcher, tmp_path):
    config_path, manifest_path, cwd = make_inputs(tmp_path)
    base = ["--config", config_path, "--manifest", manifest_path]
    run, source, checkpoint = tmp_path / "out" / "run", tmp_path / "mint", tmp_path / "mint.ckpt"
    cases = (
        (base, "--run-root, --mint-source, --mint-checkpoint"),
        (base + ["--run-root", run], "--mint-source, --mint-checkpoint"),
        (base + ["--run-root", run, "--mint-source", source], "--mint-checkpoint"),
    )
    for args, expected in cases:
        result = run_cli(launcher, (), args, cwd)
        assert result.returncode == 2 and f"missing {expected}" in result.stderr
    result = run_cli(launcher, (), base + ["--device", "tpu"], cwd)
    assert result.returncode == 2 and "invalid choice" in result.stderr
    assert not (tmp_path / "out").exists()


def test_a_full_run_refuses_missing_local_paths_before_creating_anything(launcher, tmp_path):
    config_path, manifest_path, cwd = make_inputs(tmp_path)
    (tmp_path / "out").mkdir()
    before = snapshot(tmp_path)
    result = run_cli(
        launcher,
        (),
        ["--config", config_path, "--manifest", manifest_path, "--run-root", tmp_path / "out" / "run",
         "--mint-source", tmp_path / "no-such-source", "--mint-checkpoint", tmp_path / "no-such.ckpt"],
        cwd,
    )  # fmt: skip
    assert result.returncode == 2 and result.stderr.startswith("refused:") and "mint-source" in result.stderr
    assert snapshot(tmp_path) == before


def test_a_full_run_refuses_overlapping_or_existing_run_roots_before_anything_is_written(launcher, tmp_path):
    config_path, manifest_path, cwd = make_inputs(tmp_path)
    (tmp_path / "out" / "taken").mkdir(parents=True)
    source = tmp_path / "mint"
    source.mkdir()
    checkpoint = tmp_path / "weights" / "mint.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"")
    before = snapshot(tmp_path)
    for run_root, reason in (
        (tmp_path / "in" / "run", "overlaps"),
        (source / "run", "overlaps"),
        (checkpoint.parent / "run", "overlaps"),
        (tmp_path / "out" / "taken", "already exists"),
        (tmp_path / "out", "already exists"),
    ):
        result = run_cli(
            launcher,
            (),
            ["--config", config_path, "--manifest", manifest_path, "--run-root", run_root,
             "--mint-source", source, "--mint-checkpoint", checkpoint],
            cwd,
        )  # fmt: skip
        assert result.returncode == 2 and result.stderr.startswith("refused:") and reason in result.stderr, result.stderr
        assert snapshot(tmp_path) == before


def test_a_fake_source_cannot_pass_the_pinned_closure_and_leaves_an_honest_error_status(launcher, tmp_path):
    config_path, manifest_path, cwd = make_inputs(tmp_path)
    (tmp_path / "out").mkdir()
    source = tmp_path / "mint"
    source.mkdir()
    checkpoint = tmp_path / "weights" / "mint.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"")
    run = tmp_path / "out" / "run"
    result = run_cli(
        launcher,
        (),
        ["--config", config_path, "--manifest", manifest_path, "--run-root", run,
         "--mint-source", source, "--mint-checkpoint", checkpoint],
        cwd,
    )  # fmt: skip
    assert result.returncode == 2 and "does not match the pinned release" in result.stderr
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error" and status["error"]["type"] == "MintLoadError"
    assert status["execution"]["origin"] == "pinned_release" and status["execution"]["released_execution"] == "not_run"
    assert status["execution"]["released_weights_forward_completed"] is False
    assert status["scientific_status"]["scientific_qualification"] == "not_assessed"
    assert not (run / "report.json").exists() and (run / "plan.json").is_file()
