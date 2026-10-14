"""Visible-only native encoder boundary; optional real upstream integration."""
import hashlib
import json
import os
from pathlib import Path
import types
import importlib.util
import sys

import pytest
import torch
from torch import nn

from smallAntibodyGen.models.antibody_antigen.esm_ppi import (
    FrozenESMPPI, import_native_esm2, serialize_visible,
    source_sha256, load_released_encoder,
)
from smallAntibodyGen.models.antibody_antigen.views import mask_complete


IDENTITY = {"source_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "extraction_layer": "layer_2", "precision": "float32", "tokenizer": "ESM-1b",
            "eval_mode": "true"}


class Alphabet:
    cls_idx, eos_idx, mask_idx = 0, 2, 32
    def get_idx(self, residue):
        return 4 + "ACDEFGHIKLMNPQRSTVWY".index(residue)


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(33, 8)
        self.num_layers = 2
        self.alphabet = Alphabet()
        self.calls = []

    def forward(self, tokens, **kwargs):
        self.calls.append((tokens.clone(), kwargs, self.training, torch.is_grad_enabled()))
        features = self.embedding(tokens)
        return {"representations": {2: features + features.mean(1, keepdim=True)}}


def test_true_masks_visible_only_entities_and_no_pair_output():
    view, _ = mask_complete({"H": "ACDE", "L": "ACFG", "Ag": "HIK"},
                            [("H", 2), ("H", 3), ("L", 2), ("L", 3)])
    model = RecordingEncoder()
    encoder = FrozenESMPPI(model, IDENTITY)
    encoder.train(True)
    context = encoder.extract(view)
    tokens, kwargs, training, grad = model.calls[0]
    assert not training and not grad and not encoder.training
    assert all(not p.requires_grad for p in model.parameters())
    assert int(tokens.eq(32).sum()) == 4
    assert kwargs["need_head_weights"] is False and kwargs["return_contacts"] is False
    assert kwargs["enty_ids"][0, 1] == kwargs["enty_ids"][0, 5]  # same visible chains
    assert kwargs["asym_ids"][0, 1] != kwargs["asym_ids"][0, 5]
    assert context.features.shape == (11, 8)
    nn.Linear(8, 1)(context.ab_features()).sum().backward()
    changed, _ = mask_complete({"H": "ACWW", "L": "ACYV", "Ag": "HIK"}, view.hidden_sites())
    other = encoder.extract(changed)
    assert other.content_key == context.content_key
    torch.testing.assert_close(other.features, context.features, rtol=0, atol=0)
    ab_context = encoder.extract(view.without_ag())
    assert ab_context.features.shape == (8, 8)
    assert ab_context.content_key != context.content_key


def test_joint_limit_refuses_before_forward():
    view, _ = mask_complete({"H": "ACDE", "Ag": "FGH"}, [("H", 0)])
    native = RecordingEncoder()
    with pytest.raises(ValueError, match="no truncation"):
        FrozenESMPPI(native, IDENTITY, max_tokens=8).extract(view)
    assert not native.calls


def test_source_mismatch_and_existing_package_rejected(tmp_path, monkeypatch):
    source = tmp_path / "tfold"; source.mkdir()
    init = source / "__init__.py"; init.write_text("# fixture only\n")
    pins = {"files": {"tfold/__init__.py": source_sha256(init)}}
    broken = {"files": {"tfold/__init__.py": "0" * 64}}
    with pytest.raises(ValueError, match="hash mismatch"):
        import_native_esm2(tmp_path, broken)
    monkeypatch.setitem(__import__("sys").modules, "tfold", types.ModuleType("tfold"))
    with pytest.raises(RuntimeError, match="unverified or conflicting"):
        import_native_esm2(tmp_path, pins)


def test_source_pins_normalize_checkout_newlines_and_refuse_bytecode(tmp_path):
    root = tmp_path / "tfold"; root.mkdir()
    init = root / "__init__.py"
    init.write_bytes(b"# fixture\r\n")
    digest = source_sha256(init)
    init.write_bytes(b"# fixture\n")
    assert source_sha256(init) == digest
    candidate = Path(importlib.util.cache_from_source(str(init)))
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"unverified bytecode")
    before = sys.dont_write_bytecode
    with pytest.raises(RuntimeError, match="cached bytecode"):
        import_native_esm2(tmp_path, {"files": {"tfold/__init__.py": digest}})
    assert sys.dont_write_bytecode == before


def test_extraction_ignores_ambient_autocast():
    class ResidualEncoder(RecordingEncoder):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
        def forward(self, tokens, **kwargs):
            x = self.embedding(tokens)
            return {"representations": {2: x + self.linear(x)}}
    native = ResidualEncoder()
    encoder = FrozenESMPPI(native, IDENTITY)
    view, _ = mask_complete({"H": "ACDE", "Ag": "FGHI"}, [("H", 1)])
    ordinary = encoder.extract(view)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        nested = encoder.extract(view)
    assert ordinary.content_key == nested.content_key
    torch.testing.assert_close(ordinary.features, nested.features, rtol=0, atol=0)


def test_loader_rejects_disagreeing_tied_entries_before_loading(tmp_path, monkeypatch):
    from smallAntibodyGen.models.antibody_antigen import esm_ppi
    class TinyTied(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.embed_tokens = nn.Embedding(3, 2)
            self.lm_head = nn.Linear(2, 3, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        def load_state_dict(self, *args, **kwargs):
            raise AssertionError("must reject before mutating the model")
    config = {"arch": "PPIModel", "attention_mode": "native", "embedding_dim": 1280,
              "num_heads": 20, "num_layers": 33, "token_dropout": True,
              "tokenizer": "ESM-1b", "use_crp_embeddings": True}
    payload = {"config": {"model": config}, "model": {
        "model.embed_tokens.weight": torch.zeros(3, 2), "model.lm_head.weight": torch.ones(3, 2)}}
    monkeypatch.setattr(esm_ppi, "file_sha256", lambda _: "a" * 64)
    monkeypatch.setattr(esm_ppi, "import_native_esm2", lambda *args: TinyTied)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match="tied.*disagree"):
        load_released_encoder(tmp_path, {"files": {}}, tmp_path / "fixture.pth", "a" * 64)


@pytest.mark.skipif(not os.environ.get("POC_TFOLD_SOURCE"), reason="optional pinned upstream checkout")
def test_real_native_tiny_encoder():
    root = Path(__file__).resolve().parents[3]
    pins = json.loads((root / "configs/experiments/trastuzumab_poc_tfold_source.json").read_text())
    native_class = import_native_esm2(os.environ["POC_TFOLD_SOURCE"], pins)
    native = native_class(num_layers=2, embed_dim=16, attention_heads=2, use_crp_embeddings=True)
    encoder = FrozenESMPPI(native, IDENTITY)
    view, _ = mask_complete({"H": "ACDE", "L": "FGHI", "Ag": "KLMNP"}, [("H", 1)])
    context = encoder.extract(view)
    assert context.features.shape == (13, 16)
    assert torch.isfinite(context.features).all()
    # Same true masks, native chain/entity IDs, and native ESM2 call with no pair stack.
    args = serialize_visible(view, native.alphabet)
    with torch.no_grad():
        reference = native(args[0], asym_ids=args[1], enty_ids=args[2], repr_layers=[2],
                           need_head_weights=False, return_contacts=False)["representations"][2][0, 1:-1]
    torch.testing.assert_close(context.features, reference, rtol=0, atol=0)
