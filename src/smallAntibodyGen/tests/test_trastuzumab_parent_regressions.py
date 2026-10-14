"""Independent parent handoff regressions, using synthetic models only."""
import copy

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc import parent
from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import file_digest
from smallAntibodyGen.experiments.trastuzumab_poc.factory import build_models
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig


def _policy(mode="lora"):
    config = FusionConfig(context_dim=8, bottleneck=6, lora_rank=2,
                          lora_alpha=4, activation_checkpointing=False,
                          update_mode=mode)
    policy, encoder, _ = build_models({"kind": "tiny"}, config)
    with torch.no_grad():
        for module in policy.backbone.lora_modules().values():
            module.lora_B.copy_(torch.linspace(-0.123, 0.271,
                                               module.lora_B.numel()).reshape_as(module.lora_B))
        policy.up.weight.fill_(0.03)
    return policy, encoder


def _load(path, policy, encoder):
    return parent.load_parent(path, expected_sha256=file_digest(path),
                              encoder_identity=encoder.encoder_identity,
                              fusion_config=policy.config, seed=17)


@pytest.mark.parametrize("corruption", [
    "saved_update_mode", "saved_unknown_field", "missing_source_identity",
    "source_fusion_mismatch", "source_encoder_mismatch", "merge_semantics",
    "merge_mode_mismatch", "provenance_type", "provenance_nonfinite",
    "native_eps_nan", "native_eps_negative", "native_token_dropout_type",
    "native_config_source_mismatch",
])
def test_parent_rejects_inconsistent_metadata_before_rng_mutation(tmp_path, corruption):
    policy, encoder = _policy()
    path = tmp_path / "original.pt"
    parent.export_parent(policy, path, provenance={"completed_updates": 7})
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if corruption == "saved_update_mode":
        payload["fusion_config"]["update_mode"] = "invalid"
    elif corruption == "saved_unknown_field":
        payload["fusion_config"]["unrecognized"] = True
    elif corruption == "missing_source_identity":
        payload["source_policy_identity"] = {}
    elif corruption == "source_fusion_mismatch":
        payload["source_policy_identity"]["fusion"]["bottleneck"] += 1
    elif corruption == "source_encoder_mismatch":
        payload["source_policy_identity"]["encoder"]["checkpoint_sha256"] = "c" * 64
    elif corruption == "merge_semantics":
        payload["merge_semantics"] = "invalid"
    elif corruption == "merge_mode_mismatch":
        payload["merge_semantics"] = "full_fine_tuned"
    elif corruption == "provenance_type":
        payload["provenance"] = ["not a provenance mapping"]
    elif corruption == "provenance_nonfinite":
        payload["provenance"] = {"completed_updates": float("nan")}
    elif corruption == "native_eps_nan":
        payload["native_config"]["layer_norm_eps"] = float("nan")
    elif corruption == "native_eps_negative":
        payload["native_config"]["layer_norm_eps"] = -1.0
    elif corruption == "native_token_dropout_type":
        payload["native_config"]["token_dropout"] = "false"
    elif corruption == "native_config_source_mismatch":
        payload["native_config"]["hidden_dropout_prob"] = 0.25
    corrupted = tmp_path / "corrupted.pt"
    torch.save(payload, corrupted)
    before = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        _load(corrupted, policy, encoder)
    assert torch.equal(torch.get_rng_state(), before)


def test_parent_merge_is_independent_of_ambient_cpu_autocast(tmp_path):
    policy, _ = _policy()
    normal = tmp_path / "normal.pt"
    autocast = tmp_path / "autocast.pt"
    parent.export_parent(policy, normal, provenance={})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        parent.export_parent(policy, autocast, provenance={})
    expected = torch.load(normal, weights_only=True, map_location="cpu")
    actual = torch.load(autocast, weights_only=True, map_location="cpu")
    assert set(expected["backbone_weights"]) == set(actual["backbone_weights"])
    for name, weight in expected["backbone_weights"].items():
        torch.testing.assert_close(actual["backbone_weights"][name], weight, rtol=0, atol=0)


@pytest.mark.parametrize("component", ["fusion", "rotary"])
def test_export_rejects_half_storage_instead_of_silently_promoting(tmp_path, component):
    policy, _ = _policy()
    if component == "fusion":
        policy.up.half()
    else:
        policy.backbone._buffers["rotary_inv_freq"] = policy.backbone.rotary_inv_freq.half()
    destination = tmp_path / "parent.pt"
    with pytest.raises(ValueError, match="FP32|float32|fp32"):
        parent.export_parent(policy, destination, provenance={})
    assert not destination.exists()


def test_full_mode_unexpected_adapters_cannot_be_silently_dropped(tmp_path):
    policy, _ = _policy("full")
    policy.backbone.attach_lora(rank=2, alpha=4, dropout=0)
    with torch.no_grad():
        for module in policy.backbone.lora_modules().values():
            module.lora_B.fill_(0.1)
    with pytest.raises(ValueError):
        parent.export_parent(policy, tmp_path / "parent.pt", provenance={})


def test_export_does_not_overwrite_or_remove_an_existing_sibling_temp(tmp_path):
    policy, _ = _policy()
    unrelated = tmp_path / "parent.pt.tmp"
    unrelated.write_bytes(b"another operation owns this file")
    destination = tmp_path / "parent.pt"
    parent.export_parent(policy, destination, provenance={})
    assert destination.is_file()
    assert unrelated.read_bytes() == b"another operation owns this file"
    assert {path.name for path in tmp_path.iterdir()} == {"parent.pt", "parent.pt.tmp"}


def test_failed_export_cleans_only_its_own_temp(tmp_path, monkeypatch):
    policy, _ = _policy()
    unrelated = tmp_path / "parent.pt.tmp"
    unrelated.write_bytes(b"keep this unrelated file")

    def fail_save(*args, **kwargs):
        raise OSError("injected serialization failure")

    monkeypatch.setattr(parent.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        parent.export_parent(policy, tmp_path / "parent.pt", provenance={})
    assert unrelated.read_bytes() == b"keep this unrelated file"
    assert {path.name for path in tmp_path.iterdir()} == {"parent.pt.tmp"}


def test_loaded_parent_exposes_inherited_exposure_provenance(tmp_path):
    policy, encoder = _policy()
    provenance = {"completed_updates": 7, "examples": 14,
                  "exposure": {"record_ids": ["synthetic-record"]}}
    expected = copy.deepcopy(provenance)
    path = tmp_path / "parent.pt"
    parent.export_parent(policy, path, provenance=provenance)
    child = _load(path, policy, encoder)
    observed = child.base_identity.get("parent_provenance", child.base_identity.get("provenance"))
    assert observed == expected
