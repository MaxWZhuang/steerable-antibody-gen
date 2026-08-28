"""
Tests for `scripts/gpu_memory_probe.py` (Open Decision 9).

All run on CPU. The probe's memory figures are inherently device-specific and
untestable off a GPU, so what is pinned here is everything else: that a
non-fitting configuration is RECORDED rather than raised, that a structural
rejection is distinguished from an out-of-memory one, and that the run order is
deterministic. Those are the properties that decide whether the recorded JSON
can be trusted as evidence.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.tokenizer import AminoAcidTokenizer


@pytest.fixture
def probe(project_root: Path):
    script_path = project_root.parents[1] / "scripts" / "gpu_memory_probe.py"
    spec = importlib.util.spec_from_file_location("gpu_memory_probe", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_runs_a_forward_and_backward_on_cpu(probe):
    """The happy path returns ok=True and reports the parameter count."""
    result = probe.probe_once(
        max_length=32,
        antigen_max_length=32,
        batch_size=2,
        use_amp=False,
        device=torch.device("cpu"),
        tokenizer=AminoAcidTokenizer(),
    )
    assert result["ok"] is True
    assert result["loss_finite"] is True
    assert result["parameters"] > 0
    # Memory figures are CUDA-only and must be absent rather than zero, so a
    # CPU run can never be mistaken for "this fits in 0 MiB".
    assert "peak_reserved_mib" not in result


def test_antigen_longer_than_max_length_is_recorded_as_a_structural_limit(probe):
    """
    On the scratch path the antigen encoder is built from the SAME config as the
    antibody encoder, so it inherits the antibody `max_length` and rejects any
    longer antigen. That is AB-07, and the probe's job is to record it as a
    result the owner can read -- not to crash the sweep partway through, and not
    to be confused with running out of memory. The distinction is what tells the
    owner whether a smaller batch would help (it would not).
    """
    result = probe.probe_once(
        max_length=32,
        antigen_max_length=64,
        batch_size=2,
        use_amp=False,
        device=torch.device("cpu"),
        tokenizer=AminoAcidTokenizer(),
    )
    assert result["ok"] is False
    assert result["error"] == "StructuralLimit"
    assert "64" in result["error_detail"]


def test_antigen_max_length_above_the_validator_cap_is_structural(probe):
    """`MLMConfig` caps antigen_max_length at 1024; the census measured antigens
    up to 2042 tokens, so this ceiling is reachable from real data."""
    result = probe.probe_once(
        max_length=2048,
        antigen_max_length=2048,
        batch_size=1,
        use_amp=False,
        device=torch.device("cpu"),
        tokenizer=AminoAcidTokenizer(),
    )
    assert result["ok"] is False
    assert result["error"] == "StructuralLimit"


def test_every_row_carries_the_configuration_it_measured(probe):
    """A memory figure detached from its shape is unusable as evidence."""
    result = probe.probe_once(
        max_length=32,
        antigen_max_length=32,
        batch_size=4,
        use_amp=False,
        device=torch.device("cpu"),
        tokenizer=AminoAcidTokenizer(),
    )
    assert result["max_length"] == 32
    assert result["antigen_max_length"] == 32
    assert result["batch_size"] == 4
    assert result["use_amp"] is False


def test_main_writes_deterministic_json(probe, tmp_path: Path):
    """
    Byte-identical output on a rerun. The probe's result is an owner input to
    Open Decision 1; if two runs of the same sweep disagree, neither can be
    cited.
    """
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    argv = [
        "--max-length", "32",
        "--antigen-max-length", "32",
        "--batch-size", "2",
        "--amp", "off",
        "--pairing", "coupled",
        "--device", "cpu",
    ]
    assert probe.main(argv + ["--output-json", str(out_a)]) == 0
    assert probe.main(argv + ["--output-json", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_decoupled_pairing_covers_every_combination(probe, tmp_path: Path, capsys):
    """
    `--pairing decoupled` must sweep the full cross product, because the number
    the owner needs is what raising the antigen limit ALONE would cost. Half of
    these rows are expected to be structural rejections today -- that is the
    finding, not a reason to skip them.
    """
    import json

    out = tmp_path / "decoupled.json"
    assert (
        probe.main(
            [
                "--max-length", "32", "48",
                "--antigen-max-length", "32", "64",
                "--batch-size", "2",
                "--amp", "off",
                "--pairing", "decoupled",
                "--device", "cpu",
                "--output-json", str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    measured = {(r["max_length"], r["antigen_max_length"]) for r in payload["results"]}
    assert measured == {(32, 32), (32, 64), (48, 32), (48, 64)}
