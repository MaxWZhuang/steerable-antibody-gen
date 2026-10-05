"""The adjacent RNG-capture helpers must not initialise CUDA in a CPU process.

``torch.cuda.get_rng_state_all`` lazily creates a device context, so a CPU-only run that
checkpointed would touch the driver. Capture is guarded on ``is_initialized`` alone; every CUDA
call here is mocked, so no test reaches a driver.
"""
from __future__ import annotations

import pytest
import torch

from smallAntibodyGen.experiments import her2_nf_trajectory as nf
from smallAntibodyGen.experiments import her2_routes_interp as interp


@pytest.fixture(autouse=True)
def restore_cpu_randomness():
    import random
    import numpy as np
    cpu, python, legacy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    yield
    torch.set_rng_state(cpu)
    random.setstate(python)
    np.random.set_state(legacy)


def mock_cuda(monkeypatch, *, initialized, states=(), devices=0):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: calls.append("available") or True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all",
                        lambda: calls.append("rng") or list(states))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: calls.append("count") or devices)
    return calls


def test_interp_checkpoint_state_skips_cuda_when_uninitialized(monkeypatch):
    calls = mock_cuda(monkeypatch, initialized=False)
    record = interp.checkpoint_state(model=torch.nn.Linear(2, 1))
    assert record["cuda_rng"] is None and calls == []
    assert isinstance(record["torch_rng"], torch.Tensor)


def test_interp_checkpoint_state_records_initialized_cuda(monkeypatch):
    states = [torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)]
    calls = mock_cuda(monkeypatch, initialized=True, states=states, devices=2)
    record = interp.checkpoint_state(model=torch.nn.Linear(2, 1))
    assert record["cuda_rng"] == states and calls == ["rng"]


def test_nf_capture_skips_cuda_when_uninitialized(monkeypatch):
    calls = mock_cuda(monkeypatch, initialized=False)
    record = nf.capture_rng_state()
    assert record["torch_cuda"] == [] and record["cuda_devices"] == 0 and calls == []
    assert isinstance(record["torch_cpu"], torch.Tensor)


def test_nf_capture_records_every_initialized_device(monkeypatch):
    states = [torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)]
    calls = mock_cuda(monkeypatch, initialized=True, states=states, devices=2)
    record = nf.capture_rng_state()
    assert record["torch_cuda"] == states and record["cuda_devices"] == 2
    assert "available" not in calls
