"""Tests for export_parent / load_parent (parent.py).

All tests use a tiny FusedDPLM; no released checkpoints or source weights are needed.
Codex runs the suite; this file makes no claims about test outcomes.
"""
import copy

import pytest
import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import file_digest
from smallAntibodyGen.experiments.trastuzumab_poc.parent import export_parent, load_parent
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.dplm_policy import DPLMConfig, NativeDPLM

_IDENTITY = dict(source_sha256="a" * 64, checkpoint_sha256="b" * 64,
                 extraction_layer="layer_1", precision="float32",
                 tokenizer="test", eval_mode="true")
_FUSION_CFG = FusionConfig(context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4,
                            lora_dropout=0.0, update_mode="lora")
_FULL_CFG = FusionConfig(context_dim=8, bottleneck=6, lora_rank=2, lora_alpha=4,
                          lora_dropout=0.0, update_mode="full")


def _make_dplm_config():
    return DPLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=64,
                      hidden_dropout_prob=0.0)


def _make_policy(fusion_cfg=_FUSION_CFG, *, seed=42):
    """Tiny FusedDPLM with nonzero LoRA B and nonzero learned fusion."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        native = NativeDPLM.random_init(_make_dplm_config(), seed=7)
        policy = FusedDPLM(native, _IDENTITY, fusion_cfg)
        if fusion_cfg.update_mode == "lora":
            for _, mod in policy.backbone.lora_modules().items():
                torch.nn.init.normal_(mod.lora_B)
        torch.nn.init.normal_(policy.up.weight)
        torch.nn.init.normal_(policy.up.bias)
        torch.nn.init.normal_(policy.down.weight)
    return policy


def _synthetic_forward_inputs(policy):
    """Minimal (B=1, L=5) input that exercises backbone + fusion."""
    # CLS=0, residues 4..7, EOS=2 — all non-PAD so attention mask is full.
    ids = torch.tensor([[0, 4, 5, 6, 2]], dtype=torch.long)
    ctx = torch.randn(1, 5, policy.config.context_dim, generator=torch.Generator().manual_seed(1))
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 1:4] = True
    return ids, ctx, mask


def _resave(payload, directory, name="modified.pt"):
    """Save a modified payload and return (path, sha256)."""
    path = directory / name
    torch.save(payload, str(path))
    return path, file_digest(path)


# ---------------------------------------------------------------------------
# core roundtrip
# ---------------------------------------------------------------------------

def test_roundtrip_eval_equivalence_lora(tmp_path):
    policy = _make_policy()
    policy.eval()

    result = export_parent(policy, tmp_path / "parent.pt", provenance={"step": 1})
    assert isinstance(result.get("sha256"), str) and len(result["sha256"]) == 64

    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)
    child.eval()

    ids, ctx, mask = _synthetic_forward_inputs(policy)
    with torch.no_grad():
        ref_logits, ref_hidden = policy._forward_tensors(ids, ctx, mask)
        new_logits, new_hidden = child._forward_tensors(ids, ctx, mask)

    torch.testing.assert_close(ref_logits, new_logits, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ref_hidden, new_hidden, rtol=1e-4, atol=1e-4)


def test_roundtrip_eval_equivalence_full_mode(tmp_path):
    policy = _make_policy(_FULL_CFG)
    policy.eval()

    result = export_parent(policy, tmp_path / "parent.pt", provenance={"step": 2})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FULL_CFG,
                        device="cpu", seed=0)
    child.eval()

    ids, ctx, mask = _synthetic_forward_inputs(policy)
    with torch.no_grad():
        ref_logits, _ = policy._forward_tensors(ids, ctx, mask)
        new_logits, _ = child._forward_tensors(ids, ctx, mask)

    torch.testing.assert_close(ref_logits, new_logits, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# rotary frequencies preserved
# ---------------------------------------------------------------------------

def test_nonanalytical_rotary_frequencies_preserved(tmp_path):
    policy = _make_policy()
    # Replace with non-analytical positive frequencies.
    custom_freq = torch.ones(policy.backbone.rotary_inv_freq.shape) * 0.123
    policy.backbone.set_rotary_inv_freq(custom_freq)

    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)

    torch.testing.assert_close(child.backbone.rotary_inv_freq, custom_freq,
                                rtol=0, atol=0)


# ---------------------------------------------------------------------------
# tied weights preserved
# ---------------------------------------------------------------------------

def test_tied_weights_preserved_after_load(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)

    assert (child.backbone.lm_head.decoder.weight
            is child.backbone.esm.embeddings.word_embeddings.weight), (
        "lm_head.decoder.weight must be the same object as word_embeddings.weight"
    )


# ---------------------------------------------------------------------------
# no live mutation during export
# ---------------------------------------------------------------------------

def test_export_does_not_mutate_policy(tmp_path):
    policy = _make_policy()
    policy.eval()

    # Snapshot params and flags before export.
    param_snapshot = {n: p.detach().clone() for n, p in policy.named_parameters()}
    lora_enabled_snapshot = {
        n: mod.lora_enabled for n, mod in policy.backbone.lora_modules().items()
    }
    training_before = policy.training
    rng_state_before = torch.get_rng_state()

    export_parent(policy, tmp_path / "parent.pt", provenance={})

    assert policy.training == training_before
    rng_state_after = torch.get_rng_state()
    assert torch.equal(rng_state_before, rng_state_after), "export mutated the global CPU RNG"

    for n, p in policy.named_parameters():
        torch.testing.assert_close(p, param_snapshot[n], rtol=0, atol=0,
                                   msg=f"parameter {n!r} was modified during export")

    for n, mod in policy.backbone.lora_modules().items():
        assert mod.lora_enabled == lora_enabled_snapshot[n], (
            f"lora_enabled flag at {n!r} was changed during export"
        )


# ---------------------------------------------------------------------------
# no source checkpoint needed for load
# ---------------------------------------------------------------------------

def test_load_requires_only_exported_file(tmp_path):
    """Load succeeds without any original NativeDPLM weight files."""
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={"note": "standalone"})
    del policy  # original model gone; only the exported file remains

    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)
    assert isinstance(child, FusedDPLM)


# ---------------------------------------------------------------------------
# fresh child has zero-output LoRA
# ---------------------------------------------------------------------------

def test_child_lora_is_zero_output(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)

    lora_mods = child.backbone.lora_modules()
    assert lora_mods, "expected LoRA modules on loaded child"
    for name, mod in lora_mods.items():
        assert torch.all(mod.lora_B == 0), f"lora_B at {name!r} is not zero after load"
        assert torch.any(mod.lora_A != 0), f"lora_A at {name!r} is unexpectedly all-zero"


# ---------------------------------------------------------------------------
# child identity records parent provenance, not original release
# ---------------------------------------------------------------------------

def test_child_identity_is_materialized_parent(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={"run": "x"})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)

    base_id = child.base_identity
    assert base_id.get("variant") == "materialized-parent"
    assert base_id.get("pretrained") is False
    assert base_id.get("parent_sha256") == result["sha256"]


# ---------------------------------------------------------------------------
# load-parent with different LoRA rank produces correct zero-output adapters
# ---------------------------------------------------------------------------

def test_child_lora_rank_from_new_fusion_config(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})

    new_cfg = FusionConfig(context_dim=8, bottleneck=6, lora_rank=4, lora_alpha=8,
                           lora_dropout=0.0, update_mode="lora")
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=new_cfg,
                        device="cpu", seed=0)

    for name, mod in child.backbone.lora_modules().items():
        assert mod.lora_A.shape[0] == 4, f"expected rank-4 lora_A at {name!r}"
        assert torch.all(mod.lora_B == 0)


# ---------------------------------------------------------------------------
# existing destination not overwritten
# ---------------------------------------------------------------------------

def test_existing_destination_not_overwritten(tmp_path):
    policy = _make_policy()
    dest = tmp_path / "parent.pt"
    dest.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="will not overwrite"):
        export_parent(policy, dest, provenance={})

    assert dest.read_bytes() == b"existing"


# ---------------------------------------------------------------------------
# corrupt sha rejected before deserializing
# ---------------------------------------------------------------------------

def test_corrupt_sha256_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    wrong_sha = "0" * 64

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_parent(tmp_path / "parent.pt", expected_sha256=wrong_sha,
                    encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                    device="cpu", seed=0)


# ---------------------------------------------------------------------------
# payload key inventory rejected
# ---------------------------------------------------------------------------

def test_missing_payload_key_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    payload = torch.load(str(tmp_path / "parent.pt"), map_location="cpu", weights_only=True)

    corrupt = {k: v for k, v in payload.items() if k != "backbone_weights"}
    path, sha = _resave(corrupt, tmp_path, "corrupt_missing.pt")
    with pytest.raises(ValueError, match="key mismatch|missing"):
        load_parent(path, expected_sha256=sha, encoder_identity=_IDENTITY,
                    fusion_config=_FUSION_CFG, device="cpu", seed=0)


def test_extra_payload_key_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    payload = torch.load(str(tmp_path / "parent.pt"), map_location="cpu", weights_only=True)

    corrupt = {**payload, "unexpected_key": "value"}
    path, sha = _resave(corrupt, tmp_path, "corrupt_extra.pt")
    with pytest.raises(ValueError, match="key mismatch|unexpected"):
        load_parent(path, expected_sha256=sha, encoder_identity=_IDENTITY,
                    fusion_config=_FUSION_CFG, device="cpu", seed=0)


# ---------------------------------------------------------------------------
# backbone tensor dtype / nonfinite rejected
# ---------------------------------------------------------------------------

def test_non_fp32_backbone_tensor_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    payload = torch.load(str(tmp_path / "parent.pt"), map_location="cpu", weights_only=True)

    first_key = next(iter(payload["backbone_weights"]))
    corrupt_backbone = {k: (v.half() if k == first_key else v)
                        for k, v in payload["backbone_weights"].items()}
    corrupt = {**payload, "backbone_weights": corrupt_backbone}
    path, sha = _resave(corrupt, tmp_path, "corrupt_dtype.pt")
    with pytest.raises(ValueError, match="FP32|float16"):
        load_parent(path, expected_sha256=sha, encoder_identity=_IDENTITY,
                    fusion_config=_FUSION_CFG, device="cpu", seed=0)


def test_nonfinite_backbone_tensor_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    payload = torch.load(str(tmp_path / "parent.pt"), map_location="cpu", weights_only=True)

    first_key = next(iter(payload["backbone_weights"]))
    bad = payload["backbone_weights"][first_key].clone()
    bad.view(-1)[0] = float("nan")
    corrupt_backbone = {k: (bad if k == first_key else v)
                        for k, v in payload["backbone_weights"].items()}
    corrupt = {**payload, "backbone_weights": corrupt_backbone}
    path, sha = _resave(corrupt, tmp_path, "corrupt_nan.pt")
    with pytest.raises(ValueError, match="nonfinite"):
        load_parent(path, expected_sha256=sha, encoder_identity=_IDENTITY,
                    fusion_config=_FUSION_CFG, device="cpu", seed=0)


# ---------------------------------------------------------------------------
# nonfinite / non-FP32 rotary rejected
# ---------------------------------------------------------------------------

def test_nonpositive_rotary_inv_freq_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    payload = torch.load(str(tmp_path / "parent.pt"), map_location="cpu", weights_only=True)

    bad_freq = payload["rotary_inv_freq"].clone()
    bad_freq[0] = -1.0
    corrupt = {**payload, "rotary_inv_freq": bad_freq}
    path, sha = _resave(corrupt, tmp_path, "corrupt_freq.pt")
    with pytest.raises(ValueError, match="non-positive"):
        load_parent(path, expected_sha256=sha, encoder_identity=_IDENTITY,
                    fusion_config=_FUSION_CFG, device="cpu", seed=0)


# ---------------------------------------------------------------------------
# wrong tied key value rejected
# ---------------------------------------------------------------------------

def test_wrong_tied_key_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    payload = torch.load(str(tmp_path / "parent.pt"), map_location="cpu", weights_only=True)

    corrupt = {**payload, "tied_input_key": "bad.key"}
    path, sha = _resave(corrupt, tmp_path, "corrupt_tied.pt")
    with pytest.raises(ValueError, match="tied"):
        load_parent(path, expected_sha256=sha, encoder_identity=_IDENTITY,
                    fusion_config=_FUSION_CFG, device="cpu", seed=0)


# ---------------------------------------------------------------------------
# encoder_identity mismatch rejected
# ---------------------------------------------------------------------------

def test_wrong_encoder_identity_rejected(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})

    wrong_identity = {**_IDENTITY, "checkpoint_sha256": "c" * 64}
    with pytest.raises(ValueError, match="encoder_identity"):
        load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                    encoder_identity=wrong_identity, fusion_config=_FUSION_CFG,
                    device="cpu", seed=0)


# ---------------------------------------------------------------------------
# fusion architecture mismatch rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_cfg", [
    FusionConfig(context_dim=16, bottleneck=6, lora_rank=2, lora_alpha=4),  # wrong context_dim
    FusionConfig(context_dim=8, bottleneck=12, lora_rank=2, lora_alpha=4),  # wrong bottleneck
])
def test_fusion_architecture_mismatch_rejected(tmp_path, bad_cfg):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})

    with pytest.raises(ValueError, match="context_dim|bottleneck"):
        load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                    encoder_identity=_IDENTITY, fusion_config=bad_cfg,
                    device="cpu", seed=0)


# ---------------------------------------------------------------------------
# outer RNG not mutated by load_parent
# ---------------------------------------------------------------------------

def test_load_does_not_mutate_outer_rng(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})

    rng_before = torch.get_rng_state()
    load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                device="cpu", seed=42)
    rng_after = torch.get_rng_state()

    assert torch.equal(rng_before, rng_after), "load_parent mutated the outer CPU RNG"


# ---------------------------------------------------------------------------
# provenance non-JSON-serializable rejected at export time
# ---------------------------------------------------------------------------

def test_non_json_provenance_rejected(tmp_path):
    policy = _make_policy()
    with pytest.raises((TypeError, ValueError)):
        export_parent(policy, tmp_path / "parent.pt", provenance={"bad": float("inf")})


# ---------------------------------------------------------------------------
# fusion weights correctly restored (not reset to zero)
# ---------------------------------------------------------------------------

def test_fusion_weights_restored_not_zero(tmp_path):
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)

    torch.testing.assert_close(child.up.weight, policy.up.weight, rtol=0, atol=0)
    torch.testing.assert_close(child.down.weight, policy.down.weight, rtol=0, atol=0)
    torch.testing.assert_close(child.norm.weight, policy.norm.weight, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# export of lora mode with lora B nonzero actually changes backbone vs base
# ---------------------------------------------------------------------------

def test_lora_merge_changes_backbone_weights(tmp_path):
    """Merged weights should differ from the base weights when lora_B is nonzero."""
    policy = _make_policy()
    result = export_parent(policy, tmp_path / "parent.pt", provenance={})
    child = load_parent(tmp_path / "parent.pt", expected_sha256=result["sha256"],
                        encoder_identity=_IDENTITY, fusion_config=_FUSION_CFG,
                        device="cpu", seed=0)

    lora_mods = policy.backbone.lora_modules()
    name = next(iter(lora_mods))
    orig_w = dict(policy.backbone.state_dict())[f"{name}.weight"]
    merged_w = dict(child.backbone.state_dict())[f"{name}.weight"]
    assert not torch.equal(orig_w, merged_w), (
        "merged weight should differ from base when lora_B is nonzero"
    )
