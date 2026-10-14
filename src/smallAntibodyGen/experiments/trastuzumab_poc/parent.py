"""Standalone midtraining-parent export and import for FusedDPLM.

Export merges all active LoRA adapters (W + scaling*(B@A)), drops adapter tensors,
and preserves learned norm/down/up fusion. Load reconstructs a fresh FusedDPLM with
zero-output new adapters and the restored fusion on top of the merged backbone.

Public API
----------
export_parent(policy, path, *, provenance) -> {"sha256": str}
load_parent(path, *, expected_sha256, encoder_identity, fusion_config, device, seed) -> FusedDPLM
"""
from __future__ import annotations

import copy
import os
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import canonical_bytes, file_digest, require_keys
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import isolated_torch_rng
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.antibody_antigen.precision import compute_context, validate_compute
from smallAntibodyGen.models.dplm_policy import (
    DPLMConfig, DPLMEncoder, DPLMLinear, NativeDPLM, expected_state_shapes,
)

_SCHEMA = "dplm-fused-parent/1"
_TIED_INPUT_KEY = "esm.embeddings.word_embeddings.weight"
_TIED_OUTPUT_KEY = "lm_head.decoder.weight"
_FUSION_KEYS = frozenset({
    "norm.weight", "norm.bias",
    "down.weight", "down.bias",
    "up.weight", "up.bias",
})
_PAYLOAD_KEYS = frozenset({
    "schema", "merge_semantics", "provenance", "source_policy_identity",
    "encoder_identity", "native_config", "tied_input_key", "tied_output_key",
    "rotary_inv_freq", "backbone_weights", "fusion_config", "fusion_weights",
})


def export_parent(policy: FusedDPLM, path, *, provenance: dict) -> dict:
    """Export a merged FP32 parent payload.  Never overwrites an existing destination.

    Does not mutate live parameters, enabled flags, train/eval mode, or RNG.
    Returns {"sha256": hex_digest}.
    """
    if not isinstance(policy, FusedDPLM):
        raise TypeError("policy must be a FusedDPLM")
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a JSON object")
    canonical_bytes(provenance)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"parent already exists at {path}; will not overwrite")

    # Synchronous export: no concurrent training. Retain detached references and
    # materialize each final CPU tensor once, rather than cloning two backbones.
    backbone_sd = policy.backbone.state_dict()

    lora_modules = policy.backbone.lora_modules()
    if policy.config.update_mode == "full" and lora_modules:
        raise ValueError("full-mode backbone unexpectedly contains adapters")
    if policy.config.update_mode == "lora" and not lora_modules:
        raise ValueError("LoRA-mode backbone is missing adapters")
    lora_param_keys: set[str] = set()
    for name in lora_modules:
        lora_param_keys.add(f"{name}.lora_A")
        lora_param_keys.add(f"{name}.lora_B")

    if policy.config.update_mode == "lora":
        for name, module in lora_modules.items():
            if not module.lora_enabled:
                raise ValueError(f"LoRA adapter at {name!r} is disabled; cannot export cleanly")

    for k, t in backbone_sd.items():
        if k in lora_param_keys:
            continue
        if t.dtype != torch.float32:
            raise ValueError(f"backbone tensor {k!r} has non-FP32 storage: {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"backbone tensor {k!r} contains nonfinite values")

    if (_TIED_INPUT_KEY in backbone_sd and _TIED_OUTPUT_KEY in backbone_sd
            and not torch.equal(backbone_sd[_TIED_INPUT_KEY], backbone_sd[_TIED_OUTPUT_KEY])):
        raise ValueError("tied embedding / output-projection weights disagree in state_dict")

    # Build merged backbone: drop LoRA params, drop tied-output duplicate, merge LoRA into base.
    merged_backbone: dict[str, torch.Tensor] = {
        k: t.detach().cpu() for k, t in backbone_sd.items()
        if k not in lora_param_keys and k != _TIED_OUTPUT_KEY
    }

    if policy.config.update_mode == "lora":
        for name, module in lora_modules.items():
            w_key = f"{name}.weight"
            if w_key not in merged_backbone:
                continue
            for suffix in ("lora_A", "lora_B"):
                a = backbone_sd[f"{name}.{suffix}"]
                if a.dtype != torch.float32:
                    raise ValueError(f"{name}.{suffix} is not FP32: {a.dtype}")
                if not bool(torch.isfinite(a).all()):
                    raise ValueError(f"{name}.{suffix} contains nonfinite values")
            A = backbone_sd[f"{name}.lora_A"].cpu()
            B = backbone_sd[f"{name}.lora_B"].cpu()
            with compute_context("float32", "cpu"):
                merged = merged_backbone[w_key] + module.lora_scaling * (B @ A)
            if not bool(torch.isfinite(merged).all()):
                raise ValueError(f"LoRA merge produced nonfinite weight for {name!r}")
            merged_backbone[w_key] = merged

    if policy.backbone.rotary_inv_freq.dtype != torch.float32:
        raise ValueError("rotary_inv_freq storage must be FP32")
    inv_freq = policy.backbone.rotary_inv_freq.detach().cpu().clone()
    if not bool((inv_freq > 0).all()):
        raise ValueError("rotary_inv_freq has non-positive frequencies")
    if not bool(torch.isfinite(inv_freq).all()):
        raise ValueError("rotary_inv_freq contains nonfinite values")

    fusion_weights: dict[str, torch.Tensor] = {}
    for attr, param_name in (("norm", "weight"), ("norm", "bias"),
                              ("down", "weight"), ("down", "bias"),
                              ("up", "weight"), ("up", "bias")):
        key = f"{attr}.{param_name}"
        parameter = getattr(getattr(policy, attr), param_name)
        if parameter.dtype != torch.float32:
            raise ValueError(f"fusion parameter {key!r} storage must be FP32")
        t = parameter.detach().cpu().clone()
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"fusion parameter {key!r} contains nonfinite values")
        fusion_weights[key] = t

    payload = {
        "schema": _SCHEMA,
        "merge_semantics": ("lora_merged_into_base" if policy.config.update_mode == "lora"
                            else "full_fine_tuned"),
        "provenance": provenance,
        "source_policy_identity": policy.identity(),
        "encoder_identity": copy.deepcopy(policy.encoder_identity),
        "native_config": policy.backbone.config.document(),
        "tied_input_key": _TIED_INPUT_KEY,
        "tied_output_key": _TIED_OUTPUT_KEY,
        "rotary_inv_freq": inv_freq,
        "backbone_weights": merged_backbone,
        "fusion_config": asdict(policy.config),
        "fusion_weights": fusion_weights,
    }

    _validate_metadata(payload, policy.config, policy.encoder_identity)
    expected = expected_state_shapes(policy.backbone.config)
    if set(merged_backbone) != set(expected) - {_TIED_OUTPUT_KEY}:
        raise ValueError("export backbone inventory mismatch")
    for key, tensor in merged_backbone.items():
        if tuple(tensor.shape) != expected[key]:
            raise ValueError(f"export backbone shape mismatch: {key}")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        # A same-filesystem hard link atomically publishes only if destination is
        # absent, on Windows and POSIX. No rename-overwrite race.
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

    return {"sha256": file_digest(path)}


def _validate_metadata(payload, fusion_config, encoder_identity):
    if not isinstance(fusion_config, FusionConfig):
        raise TypeError("fusion_config must be FusionConfig")
    for name in ("provenance", "source_policy_identity", "encoder_identity", "native_config", "fusion_config"):
        if not isinstance(payload[name], dict):
            raise ValueError(f"{name} must be a JSON object")
        canonical_bytes(payload[name])
    saved = FusionConfig(**payload["fusion_config"])
    require_keys(payload["fusion_config"], asdict(saved), "saved fusion config")
    if (saved.context_dim, saved.bottleneck) != (fusion_config.context_dim, fusion_config.bottleneck):
        raise ValueError("context_dim or bottleneck mismatch")
    expected_merge = "lora_merged_into_base" if saved.update_mode == "lora" else "full_fine_tuned"
    if payload["merge_semantics"] != expected_merge:
        raise ValueError("merge semantics differs from source update mode")
    doc = payload["native_config"]
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size",
            "layer_norm_eps", "hidden_dropout_prob", "max_position_embeddings", "rope_theta", "token_dropout")
    require_keys(doc, keys, "native config")
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                 "vocab_size", "max_position_embeddings"):
        if type(doc[name]) is not int or doc[name] < 1:
            raise ValueError(f"invalid native dimension {name}")
    for name in ("layer_norm_eps", "rope_theta", "hidden_dropout_prob"):
        value = doc[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"invalid native scalar {name}")
    if doc["layer_norm_eps"] <= 0 or doc["rope_theta"] <= 0 or type(doc["token_dropout"]) is not bool:
        raise ValueError("invalid native normalization, rotary or token-dropout setting")
    config = DPLMConfig(**doc)
    identity = payload["source_policy_identity"]
    require_keys(identity, ("schema", "base", "encoder", "fusion", "compute_dtype", "update_mode", "serialization"),
                 "source policy identity")
    if (identity["schema"] != "dplm-visible-fusion/2"
            or identity["serialization"] != "native-dplm-paired/1"
            or identity["fusion"] != payload["fusion_config"]
            or identity["encoder"] != payload["encoder_identity"]
            or identity["encoder"] != encoder_identity
            or identity["compute_dtype"] != saved.compute_dtype
            or identity["update_mode"] != saved.update_mode
            or encoder_identity.get("precision") != fusion_config.compute_dtype
            or not isinstance(identity["base"], dict)
            or identity["base"].get("config") != config.document()):
        raise ValueError("inconsistent source policy identity")
    return config


def load_parent(path, *, expected_sha256: str, encoder_identity: dict,
                fusion_config: FusionConfig, device="cpu", seed: int = 0) -> FusedDPLM:
    """Load a merged FP32 parent and return a fresh FusedDPLM with restored fusion.

    Verifies sha256 before deserializing.  New LoRA adapters are zero-output.
    Outer RNG state (CPU and selected CUDA if device is CUDA) is unchanged on return.
    """
    path = Path(path)
    actual = file_digest(path)
    if actual != expected_sha256:
        raise ValueError(f"sha256 mismatch: expected {expected_sha256!r}, got {actual!r}")

    payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)

    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"unrecognized parent payload schema (expected {_SCHEMA!r})")
    if frozenset(payload) != _PAYLOAD_KEYS:
        unexpected = sorted(frozenset(payload) - _PAYLOAD_KEYS)
        missing = sorted(_PAYLOAD_KEYS - frozenset(payload))
        raise ValueError(f"parent payload key mismatch; unexpected={unexpected}, missing={missing}")

    if payload["tied_input_key"] != _TIED_INPUT_KEY or payload["tied_output_key"] != _TIED_OUTPUT_KEY:
        raise ValueError("tied_input_key or tied_output_key do not match expected constants")

    if payload["encoder_identity"] != encoder_identity:
        raise ValueError("encoder_identity in payload does not match the supplied value")

    saved_fusion_cfg = payload["fusion_config"]
    if not isinstance(saved_fusion_cfg, dict):
        raise ValueError("fusion_config in payload must be a dict")
    if saved_fusion_cfg.get("context_dim") != fusion_config.context_dim:
        raise ValueError("context_dim mismatch between parent payload and requested fusion_config")
    if saved_fusion_cfg.get("bottleneck") != fusion_config.bottleneck:
        raise ValueError("bottleneck mismatch between parent payload and requested fusion_config")

    native_config_doc = payload["native_config"]
    if not isinstance(native_config_doc, dict):
        raise ValueError("native_config in payload must be a dict")
    config = _validate_metadata(payload, fusion_config, encoder_identity)

    validate_compute(fusion_config.compute_dtype, device)

    exp_shapes = expected_state_shapes(config)
    exp_backbone_keys = frozenset(k for k in exp_shapes if k != _TIED_OUTPUT_KEY)
    backbone_weights = payload["backbone_weights"]
    if not isinstance(backbone_weights, dict):
        raise ValueError("backbone_weights must be a dict")
    if frozenset(backbone_weights) != exp_backbone_keys:
        unexpected = sorted(frozenset(backbone_weights) - exp_backbone_keys)
        missing = sorted(exp_backbone_keys - frozenset(backbone_weights))
        raise ValueError(
            f"backbone_weights key inventory mismatch; unexpected={unexpected}, missing={missing}"
        )
    for k, t in backbone_weights.items():
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"backbone_weights[{k!r}] is not a tensor")
        if tuple(t.shape) != exp_shapes[k]:
            raise ValueError(
                f"shape mismatch for backbone {k!r}: expected {exp_shapes[k]}, got {tuple(t.shape)}"
            )
        if t.dtype != torch.float32:
            raise ValueError(f"backbone tensor {k!r} is not FP32: {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"backbone tensor {k!r} contains nonfinite values")

    inv_freq = payload["rotary_inv_freq"]
    exp_inv_shape = (int(config.head_dim) // 2,)
    if not isinstance(inv_freq, torch.Tensor):
        raise ValueError("rotary_inv_freq is not a tensor")
    if tuple(inv_freq.shape) != exp_inv_shape:
        raise ValueError(
            f"rotary_inv_freq shape: expected {exp_inv_shape}, got {tuple(inv_freq.shape)}"
        )
    if inv_freq.dtype != torch.float32:
        raise ValueError("rotary_inv_freq is not FP32")
    if not bool((inv_freq > 0).all()):
        raise ValueError("rotary_inv_freq has non-positive values")
    if not bool(torch.isfinite(inv_freq).all()):
        raise ValueError("rotary_inv_freq contains nonfinite values")

    fusion_weights = payload["fusion_weights"]
    if not isinstance(fusion_weights, dict):
        raise ValueError("fusion_weights must be a dict")
    if frozenset(fusion_weights) != _FUSION_KEYS:
        raise ValueError(f"fusion_weights key inventory mismatch; got {sorted(fusion_weights)}")
    hidden = config.hidden_size
    context_dim = fusion_config.context_dim
    bottleneck = fusion_config.bottleneck
    _fusion_shapes: dict[str, tuple[int, ...]] = {
        "norm.weight": (hidden,), "norm.bias": (hidden,),
        "down.weight": (bottleneck, hidden + context_dim), "down.bias": (bottleneck,),
        "up.weight": (hidden, bottleneck), "up.bias": (hidden,),
    }
    for k, t in fusion_weights.items():
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"fusion_weights[{k!r}] is not a tensor")
        if tuple(t.shape) != _fusion_shapes[k]:
            raise ValueError(
                f"fusion shape mismatch for {k!r}: expected {_fusion_shapes[k]}, got {tuple(t.shape)}"
            )
        if t.dtype != torch.float32:
            raise ValueError(f"fusion tensor {k!r} is not FP32: {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"fusion tensor {k!r} contains nonfinite values")

    with isolated_torch_rng(seed, device):
        with torch.device("meta"):
            backbone = DPLMEncoder(config, initialize=False)

        full_state = {k: v.to("cpu") for k, v in backbone_weights.items()}
        full_state[_TIED_OUTPUT_KEY] = full_state[_TIED_INPUT_KEY]
        backbone.load_state_dict(full_state, strict=True, assign=True)
        backbone.set_rotary_inv_freq(inv_freq.to("cpu"))
        backbone.tie_weights()

        if backbone.lm_head.decoder.weight is not backbone.esm.embeddings.word_embeddings.weight:
            raise RuntimeError("tied weight re-establishment failed after strict load")
        if any(p.device.type == "meta" for p in backbone.parameters()):
            raise RuntimeError("a backbone parameter remains on the meta device")

        backbone.eval()

        parent_identity = {
            "variant": "materialized-parent",
            "pretrained": False,
            "parent_sha256": expected_sha256,
            "source_policy_identity": payload["source_policy_identity"],
            "parent_provenance": copy.deepcopy(payload["provenance"]),
            "config": config.document(),
            "storage_dtype": "torch.float32",
            "compute_dtype": "torch.float32",
        }
        native = NativeDPLM(
            model=backbone,
            config=config,
            identity=parent_identity,
            load_record={"pretrained": False, "source": "materialized-parent",
                         "parent_sha256": expected_sha256},
        )

        policy = FusedDPLM(native, encoder_identity, fusion_config)

        with torch.no_grad():
            policy.norm.weight.copy_(fusion_weights["norm.weight"])
            policy.norm.bias.copy_(fusion_weights["norm.bias"])
            policy.down.weight.copy_(fusion_weights["down.weight"])
            policy.down.bias.copy_(fusion_weights["down.bias"])
            policy.up.weight.copy_(fusion_weights["up.weight"])
            policy.up.bias.copy_(fusion_weights["up.bias"])

        policy = policy.to(device)

    return policy
