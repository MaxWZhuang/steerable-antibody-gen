from __future__ import annotations

import importlib.util

import pytest
import torch

from smallAntibodyGen.antigen_tokenization import build_antigen_tokenizer
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


transformers = pytest.importorskip("transformers", reason="optional 'esm' extra not installed")

ESM_NAME = "facebook/esm2_t6_8M_UR50D"
ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDK"


def _esm_config(**overrides) -> MLMConfig:
    base = dict(
        vocab_size=AminoAcidTokenizer().vocab_size,
        pad_token_id=AminoAcidTokenizer().pad_id,
        max_length=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        antigen_encoder_type="esm",
        esm_model_name=ESM_NAME,
        antigen_max_length=16,
    )
    base.update(overrides)
    return MLMConfig(**base)


def _antigen_ids(max_length: int = 16) -> torch.Tensor:
    tok = build_antigen_tokenizer("esm", AminoAcidTokenizer(), ESM_NAME)
    return torch.tensor([tok.encode(ANTIGEN, max_length)], dtype=torch.long)


def _build_encoder(**overrides):
    from smallAntibodyGen.models.esm_antigen_encoder import ESMAntigenEncoder

    try:
        return ESMAntigenEncoder(_esm_config(**overrides))
    except OSError:
        pytest.skip("ESM weights unavailable (offline)")


def test_esm_encoder_forward_shape_and_mask():
    encoder = _build_encoder()
    encoder.eval()
    input_ids = _antigen_ids(16)
    hidden, mask = encoder(input_ids)

    assert hidden.shape == (1, input_ids.shape[1], 32)  # projected to d_model
    assert hidden.dtype == torch.float32
    assert mask.shape == input_ids.shape
    assert torch.isfinite(hidden).all()


def test_esm_encoder_infers_mask_from_pad_when_omitted():
    encoder = _build_encoder()
    encoder.eval()
    ids = _antigen_ids(16)
    # Append an explicit pad token; the inferred mask should mark it 0.
    padded = torch.cat([ids, torch.tensor([[encoder._pad_token_id]])], dim=1)
    _, mask = encoder(padded)
    assert mask[0, -1].item() == 0
    assert mask[0, 0].item() == 1


def test_frozen_esm_trains_only_projection():
    encoder = _build_encoder(antigen_encoder_finetune="frozen")
    esm_params = [p for p in encoder.esm.parameters()]
    proj_params = list(encoder.projection.parameters())

    assert all(not p.requires_grad for p in esm_params)  # backbone frozen
    assert all(p.requires_grad for p in proj_params)  # projection trainable

    # Gradient reaches the projection even though the ESM forward is under no_grad.
    hidden, _ = encoder(_antigen_ids(16))
    hidden.sum().backward()
    assert encoder.projection.weight.grad is not None
    assert all(p.grad is None for p in esm_params)


def test_esm_dual_stream_forward_produces_expected_shapes():
    config = _esm_config()
    model = AntibodyAntigenCrossAttention(config)
    model.eval()

    tokenizer = AminoAcidTokenizer()
    # Minimal antibody stream: [CLS][IGH] E V Q [EOS]
    antibody_ids = torch.tensor(
        [tokenizer.encode_sequence("EVQ", locus="IGH", max_length=config.max_length)],
        dtype=torch.long,
    )
    antibody_mask = torch.ones_like(antibody_ids)
    antigen_ids = _antigen_ids(16)
    antigen_mask = torch.ones_like(antigen_ids)

    mlm_logits, compatibility_logits = model(
        antibody_input_ids=antibody_ids,
        antibody_attention_mask=antibody_mask,
        antigen_input_ids=antigen_ids,
        antigen_attention_mask=antigen_mask,
    )

    assert mlm_logits.shape == (1, antibody_ids.shape[1], config.vocab_size)
    assert compatibility_logits.shape == (1, 2)
    assert torch.isfinite(mlm_logits).all()
    assert torch.isfinite(compatibility_logits).all()


def test_frozen_backbone_stays_in_eval_mode_under_model_train():
    """`model.train()` recurses into every submodule, including a FROZEN backbone.

    Without the `train()` override the ESM stack would run its dropout and
    train-mode normalization during training, making the "frozen features" arm's
    features stochastic. The Stage A ablation's entire claim is "pretrained ESM
    features vs from-scratch features"; a feature extractor whose output changes
    run-to-run is not a fixed feature extractor.
    """
    encoder = _build_encoder(antigen_encoder_finetune="frozen")

    encoder.train()
    assert not encoder.esm.training, "frozen ESM backbone must stay in eval mode"
    # The modules this class owns still follow the requested mode.
    assert encoder.projection_dropout.training
    assert encoder.training

    encoder.eval()
    assert not encoder.esm.training


def test_frozen_backbone_is_deterministic_while_the_parent_trains():
    encoder = _build_encoder(antigen_encoder_finetune="frozen", dropout=0.0)
    encoder.train()
    input_ids = _antigen_ids(16)
    with torch.no_grad():
        first, _ = encoder(input_ids)
        second, _ = encoder(input_ids)
    assert torch.allclose(first, second)


def test_dual_stream_train_call_keeps_the_frozen_antigen_stream_in_eval():
    """The path that actually matters: the trainer calls `model.train()` on the
    whole dual-stream model, not on the antigen encoder directly."""
    try:
        model = AntibodyAntigenCrossAttention(
            _esm_config(antigen_encoder_finetune="frozen")
        )
    except OSError:
        pytest.skip("ESM weights unavailable (offline)")

    model.train()
    assert model.antibody_encoder.training
    assert not model.antigen_encoder.esm.training


def test_dual_stream_init_does_not_clobber_pretrained_esm_weights():
    """The dual-stream __init__ initializes the fusion block and heads it owns.

    If it did that with a blanket `self.apply(...)`, the pass would recurse into
    `antigen_encoder.esm` and overwrite every pretrained ESM weight with
    N(0, 0.02) noise -- turning the "pretrained antigen encoder" arm into a
    randomly-initialized one that still trains, still converges, and still reports
    plausible metrics. That is why the init list is explicit and per-module.
    """
    from smallAntibodyGen.models.esm_antigen_encoder import ESMAntigenEncoder

    config = _esm_config()
    try:
        standalone = ESMAntigenEncoder(config)
        fused = AntibodyAntigenCrossAttention(config)
    except OSError:
        pytest.skip("ESM weights unavailable (offline)")

    reference = dict(standalone.esm.state_dict())
    embedded = dict(fused.antigen_encoder.esm.state_dict())
    assert reference.keys() == embedded.keys()
    for key, value in reference.items():
        assert torch.equal(value, embedded[key]), f"pretrained ESM weight overwritten: {key}"
