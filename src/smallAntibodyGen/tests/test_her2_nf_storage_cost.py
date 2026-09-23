import hashlib
from types import SimpleNamespace

import pytest
import torch

from smallAntibodyGen.experiments import her2_nf_storage as storage
from smallAntibodyGen.experiments.her2_nf_cost import BudgetExhausted, WorkBudget
from smallAntibodyGen.experiments import her2_support_paths as paths


def test_chunked_digest_is_the_historical_digest():
    model = torch.nn.Linear(400, 1000)
    expected = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        expected.update(name.encode())
        expected.update(tensor.detach().numpy().tobytes())
    assert storage.state_digest(model) == expected.hexdigest()


def test_checkpoint_publication_rejects_wrong_existing_weights_and_identity(tmp_path):
    policy = SimpleNamespace(model=torch.nn.Linear(3, 2))
    target = tmp_path / "model.pt"
    digest = storage.save_checkpoint(target, policy, {"identity": {"seed": 1}})
    assert storage.save_checkpoint(target, policy, {"identity": {"seed": 1}}) == digest
    with pytest.raises(ValueError, match="identity"):
        storage.save_checkpoint(target, policy, {"identity": {"seed": 2}})
    with torch.no_grad():
        policy.model.weight.add_(1.0)
    with pytest.raises(ValueError, match="digest"):
        storage.save_checkpoint(target, policy, {"identity": {"seed": 1}})


def test_budget_restart_keeps_measured_and_unknown_time_separate(tmp_path):
    clock = [0.0]
    target = tmp_path / "budget.json"
    meter = WorkBudget(target, identity={"pilot": 1}, limit_seconds=10, clock=lambda: clock[0])
    with meter.unit("step", reserve_seconds=3):
        clock[0] += 2
    saved = paths.read_json(target)
    saved["pending"] = {"category": "step", "reserved_seconds": 3}
    paths.write_json(target, saved)  # Kill after admission, before measured completion.
    resumed = WorkBudget(target, identity={"pilot": 1}, limit_seconds=10, clock=lambda: clock[0])
    assert resumed.data["measured_seconds"] == 2
    assert resumed.data["uncertainty_debit_seconds"] == 3
    assert resumed.remaining_seconds == 5
    with pytest.raises(BudgetExhausted):
        with resumed.unit("gate", reserve_seconds=6):
            pytest.fail("over-budget work started")
    again = WorkBudget(target, identity={"pilot": 1}, limit_seconds=10)
    assert again.charged_seconds == 5  # Reopening cannot double-charge the crash.


def test_budget_records_actual_overrun_and_admits_no_further_work(tmp_path):
    clock = [0.0]
    meter = WorkBudget(tmp_path / "budget.json", identity={}, limit_seconds=10,
                       clock=lambda: clock[0])
    with meter.unit("step", reserve_seconds=3):
        clock[0] = 4
    assert meter.data["measured_seconds"] == 4
    assert meter.data["overruns"]
    with pytest.raises(BudgetExhausted):
        with meter.unit("step", reserve_seconds=1):
            pass
