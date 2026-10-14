"""Explicit released or test-only model construction. Never downloads or falls back."""
import torch
from torch import nn

from smallAntibodyGen.experiments.trastuzumab_poc.artifacts import read_json, require_keys
from smallAntibodyGen.experiments.trastuzumab_poc.randomness import isolated_torch_rng
from smallAntibodyGen.models.antibody_antigen.esm_ppi import FrozenESMPPI, load_released_encoder
from smallAntibodyGen.models.antibody_antigen.fusion import FusedDPLM, FusionConfig
from smallAntibodyGen.models.antibody_antigen.precision import validate_compute
from smallAntibodyGen.models.dplm_policy import DPLMConfig, DPLMEncoder, NativeDPLM, check_architecture, load_pinned_dplm


class _TinyAlphabet:
    cls_idx, eos_idx, mask_idx = 0, 2, 32

    def get_idx(self, residue):
        return 4 + "ACDEFGHIKLMNPQRSTVWY".index(residue)


class _TinyEncoder(nn.Module):
    """Synthetic context encoder; the pooled visible tokens include the antigen."""
    alphabet = _TinyAlphabet()
    num_layers = 1

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(33, 8)
        self.projection = nn.Linear(8, 8)

    def forward(self, tokens, **kwargs):
        features = self.embedding(tokens)
        return {"representations": {1: self.projection(features + features.mean(dim=1, keepdim=True))}}


def build_models(model, fusion_config, *, device="cpu", seed=0, initialization=None):
    """Return (policy, frozen_encoder, initialization_identity).

    A standalone parent needs its encoder, but never the original DPLM weight file.
    Tiny initialization is fixed across training seeds, just like a shared release.
    """
    if not isinstance(fusion_config, FusionConfig):
        raise TypeError("fusion_config must be FusionConfig")
    if not isinstance(model, dict) or (initialization is not None and not isinstance(initialization, dict)):
        raise ValueError("model and initialization must be objects")
    validate_compute(fusion_config.compute_dtype, device)
    initialization = {"kind": "release"} if initialization is None else initialization
    if initialization.get("kind") == "release":
        require_keys(initialization, ("kind",), "initialization")
    elif initialization.get("kind") == "parent":
        require_keys(initialization, ("kind", "path", "sha256"), "initialization")
    else:
        raise ValueError("initialization must be release or parent; resume is a separate operation")
    if model.get("kind") == "tiny":
        require_keys(model, ("kind",), "tiny model")
        if fusion_config.context_dim != 8:
            raise ValueError("tiny context_dim must equal 8")
        with isolated_torch_rng(1729):
            native_encoder = _TinyEncoder().to(device)
        identity = {"source_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                    "extraction_layer": "layer_1", "precision": fusion_config.compute_dtype,
                    "tokenizer": "synthetic-test-only", "eval_mode": "true"}
        encoder = FrozenESMPPI(native_encoder, identity, max_tokens=128,
                               compute_dtype=fusion_config.compute_dtype)
        if initialization["kind"] == "release":
            config = DPLMConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32, max_position_embeddings=128,
                                hidden_dropout_prob=0.0)
            with isolated_torch_rng(1729):
                backbone = DPLMEncoder(config).to(device)
            native = NativeDPLM(backbone, config,
                                {"variant": "synthetic-test-only", "pretrained": False,
                                 "seed": 1729, "config": config.document(),
                                 "storage_dtype": "torch.float32"}, {"pretrained": False})
    elif model.get("kind") == "released":
        require_keys(model, ("kind", "variant", "weights_dir", "pins_path", "encoder_source_root",
                            "encoder_source_pins_path", "encoder_checkpoint",
                            "encoder_checkpoint_sha256", "encoder_max_tokens"), "released model")
        if fusion_config.context_dim != 1280:
            raise ValueError("released ESM-PPI context_dim must equal 1280")
        if type(model["encoder_max_tokens"]) is not int or model["encoder_max_tokens"] < 3:
            raise ValueError("encoder_max_tokens must be an integer >= 3")
        pins = read_json(model["pins_path"])
        if model["variant"] not in ("dplm_150m", "dplm_650m"):
            raise ValueError("unsupported released DPLM variant")
        # The native encoder constructor initializes parameters before strict loading.
        with isolated_torch_rng(seed, device):
            encoder = load_released_encoder(model["encoder_source_root"],
                                            read_json(model["encoder_source_pins_path"]),
                                            model["encoder_checkpoint"], model["encoder_checkpoint_sha256"],
                                            device=device, max_tokens=model["encoder_max_tokens"],
                                            compute_dtype=fusion_config.compute_dtype)
        if initialization["kind"] == "release":
            native = load_pinned_dplm(model["weights_dir"], pins=pins,
                                      variant=model["variant"], device=device, storage_dtype=torch.float32)
    else:
        raise ValueError("model.kind must be explicitly released or tiny")
    if initialization["kind"] == "parent":
        from smallAntibodyGen.experiments.trastuzumab_poc.parent import load_parent
        policy = load_parent(initialization["path"], expected_sha256=initialization["sha256"],
                             encoder_identity=encoder.encoder_identity, fusion_config=fusion_config,
                             device=device, seed=seed)
        if model["kind"] == "released":
            check_architecture(policy.backbone.config, pins["variants"][model["variant"]]["architecture"])
        else:
            expected = DPLMConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=128,
                                  hidden_dropout_prob=0.0)
            if policy.backbone.config.document() != expected.document():
                raise ValueError("parent does not match tiny architecture")
        origin = {"kind": "parent", "sha256": initialization["sha256"], "base": policy.base_identity}
    else:
        with isolated_torch_rng(seed, device):
            policy = FusedDPLM(native, encoder.encoder_identity, fusion_config)
        origin = {"kind": "release", "base": policy.base_identity}
    return policy, encoder, origin
