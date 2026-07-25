from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord
from smallAntibodyGen.models.mlm import (
    AntibodyAntigenCrossAttention,
    AntibodyMLM,
    MLMConfig,
)
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


def make_record(tokenizer, sequence: str, locus: str, chain_group: str):
    token_ids = tokenizer.encode_sequence(sequence, locus=locus, max_length=64)
    return OASRecord(
        sequence=sequence,
        token_ids=token_ids,
        locus=locus,
        chain_group=chain_group,
        split="train",
        length=len(sequence),
        token_length=len(token_ids),
    )


def make_paired_record(heavy_sequence: str, light_sequence: str, light_locus: str = "IGK"):
    return OASRecord(
        sequence=heavy_sequence,
        locus="PAIRED",
        chain_group="paired",
        split="train",
        length=len(heavy_sequence) + len(light_sequence),
        token_length=len(heavy_sequence) + len(light_sequence) + 5,
        sequence_heavy=heavy_sequence,
        sequence_light=light_sequence,
        heavy_locus="IGH",
        light_locus=light_locus,
        is_paired=True,
    )


def make_antigen_inputs(
    tokenizer: AminoAcidTokenizer,
    sequences: list[str],
    max_length: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        tokenizer.encode_sequence(sequence, locus=None, max_length=max_length)
        for sequence in sequences
    ]
    max_len = max(len(ids) for ids in encoded)
    padded = []
    attention_masks = []
    for ids in encoded:
        pad_len = max_len - len(ids)
        padded.append(ids + [tokenizer.pad_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
    )


def test_mlm_forward_shape(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model = AntibodyMLM(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([
        make_record(tokenizer, "CARDRST", "IGH", "heavy"),
        make_record(tokenizer, "QQYNSY", "IGK", "light"),
    ])

    logits = model(batch["input_ids"], batch["attention_mask"])

    assert logits.shape[0] == batch["input_ids"].shape[0]
    assert logits.shape[1] == batch["input_ids"].shape[1]
    assert logits.shape[2] == tokenizer.vocab_size


def test_mlm_loss_is_finite(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model = AntibodyMLM(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([
        make_record(tokenizer, "CARDRST", "IGH", "heavy"),
        make_record(tokenizer, "QQYNSY", "IGK", "light"),
    ])

    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = model.compute_loss(logits, batch["labels"])

    assert torch.isfinite(loss)


def test_mlm_backward_produces_gradients(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model = AntibodyMLM(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    batch = collator([
        make_record(tokenizer, "CARDRST", "IGH", "heavy"),
        make_record(tokenizer, "QQYNSY", "IGK", "light"),
    ])

    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = model.compute_loss(logits, batch["labels"])
    loss.backward()

    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_weight_tying_holds_when_enabled(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tie_weights=True,
    )
    model = AntibodyMLM(config)

    assert model.lm_head.weight.data_ptr() == model.sequence_encoder.token_embedding.weight.data_ptr()


def test_mlm_can_fit_one_fixed_batch(tokenizer):
    """
    Strong implementation proof:
    if the model cannot lower loss on one fixed batch, something is wrong.
    """
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    model = AntibodyMLM(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.5,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    fixed_batch = collator([
        make_record(tokenizer, "CARDRSTYWGQGTLV", "IGH", "heavy"),
        make_record(tokenizer, "QQYNSYPWTFGQGTK", "IGK", "light"),
    ])

    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    logits = model(fixed_batch["input_ids"], fixed_batch["attention_mask"])
    initial_loss = model.compute_loss(logits, fixed_batch["labels"]).item()

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed_batch["input_ids"], fixed_batch["attention_mask"])
        loss = model.compute_loss(logits, fixed_batch["labels"])
        loss.backward()
        optimizer.step()

    final_logits = model(fixed_batch["input_ids"], fixed_batch["attention_mask"])
    final_loss = model.compute_loss(final_logits, fixed_batch["labels"]).item()

    assert final_loss < initial_loss


def test_model_returns_pairing_logits_for_paired_batches(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=96,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model = AntibodyMLM(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=96,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        shuffle_pair_probability=1.0,
        rng_seed=42,
    )
    batch = collator([
        make_paired_record("CARDRSTYWGQGTLV", "QQYNSYPWTFGQGTK", light_locus="IGK"),
        make_paired_record("CVRDRSTYWGQGTLV", "AQYNSYPWTFGQGTA", light_locus="IGL"),
    ])

    mlm_logits, pair_logits = model.forward_with_pairing(batch["input_ids"], batch["attention_mask"])
    losses = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=batch["labels"],
        pair_logits=pair_logits,
        pair_labels=batch["pair_labels"],
        pair_mask=batch["pair_mask"],
        pair_loss_weight=1.0,
    )

    assert mlm_logits.shape[:2] == batch["input_ids"].shape
    assert pair_logits.shape == (2, 2)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["pair_loss"])


def test_antibody_antigen_cross_attention_returns_expected_shapes(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=96,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model = AntibodyAntigenCrossAttention(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=96,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    antibody_batch = collator([
        make_paired_record("CARDRSTYWGQGTLV", "QQYNSYPWTFGQGTK", light_locus="IGK"),
        make_paired_record("CVRDRSTYWGQGTLV", "AQYNSYPWTFGQGTA", light_locus="IGL"),
    ])
    antigen_input_ids, antigen_attention_mask = make_antigen_inputs(
        tokenizer,
        ["MKTIIALSYIFCLVFADYKDDDDK", "ACDEFGHIKLMNPQRSTVWY"],
        max_length=96,
    )

    mlm_logits, compatibility_logits = model(
        antibody_input_ids=antibody_batch["input_ids"],
        antibody_attention_mask=antibody_batch["attention_mask"],
        antigen_input_ids=antigen_input_ids,
        antigen_attention_mask=antigen_attention_mask,
    )

    assert mlm_logits.shape[:2] == antibody_batch["input_ids"].shape
    assert mlm_logits.shape[-1] == tokenizer.vocab_size
    assert compatibility_logits.shape == (2, 2)


def test_antibody_antigen_cross_attention_computes_joint_losses(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=96,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    model = AntibodyAntigenCrossAttention(config)

    collator = MLMCollator(
        tokenizer=tokenizer,
        max_length=96,
        mask_probability=0.3,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )
    antibody_batch = collator([
        make_paired_record("CARDRSTYWGQGTLV", "QQYNSYPWTFGQGTK"),
        make_paired_record("CVRDRSTYWGQGTLV", "AQYNSYPWTFGQGTA"),
    ])
    antigen_input_ids, antigen_attention_mask = make_antigen_inputs(
        tokenizer,
        ["MKTIIALSYIFCLVFADYKDDDDK", "ACDEFGHIKLMNPQRSTVWY"],
        max_length=96,
    )

    mlm_logits, compatibility_logits = model(
        antibody_input_ids=antibody_batch["input_ids"],
        antibody_attention_mask=antibody_batch["attention_mask"],
        antigen_input_ids=antigen_input_ids,
        antigen_attention_mask=antigen_attention_mask,
    )
    losses = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=antibody_batch["labels"],
        compatibility_logits=compatibility_logits,
        compatibility_labels=torch.tensor([1, 0], dtype=torch.long),
        compatibility_mask=torch.tensor([True, True], dtype=torch.bool),
        compatibility_loss_weight=0.5,
    )

    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["mlm_loss"])
    assert torch.isfinite(losses["compatibility_loss"])


def test_antibody_antigen_cross_attention_weight_tying_holds(tokenizer):
    config = MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        tie_weights=True,
    )
    model = AntibodyAntigenCrossAttention(config)

    assert model.lm_head.weight.data_ptr() == model.antibody_encoder.token_embedding.weight.data_ptr()


# --------------------------------------------------------------------------- #
# Mechanism-search knobs: mlm_loss_weight (ARM-A) + compat_readout (ARM-B)
#
# Ported from the sibling mirror repo (its commit "Add mlm_loss_weight +
# compat_readout knobs"). Both knobs default to the historical behavior
# byte-for-byte; these tests are what pins that.
# --------------------------------------------------------------------------- #
def _readout_config(tokenizer, compat_readout: str = "cls") -> MLMConfig:
    return MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        compat_readout=compat_readout,
    )


def _tiny_dual_forward(tokenizer, model):
    antibody_batch = MLMCollator(
        tokenizer=tokenizer,
        max_length=64,
        mask_probability=0.15,
        hcdr3_span_probability=0.0,
        rng_seed=42,
    )(
        [
            make_paired_record("CARDRSTYWGQGTLV", "QQYNSYPWTFGQGTK"),
            make_paired_record("CVRDRSTYWGQGTLV", "AQYNSYPWTFGQGTA"),
        ]
    )
    antigen_input_ids, antigen_attention_mask = make_antigen_inputs(
        tokenizer,
        ["EVQLVESGGGLVQPGGSLRLSCAAS", "DIQMTQSPSSLSASVGDRVTITC"],
        max_length=64,
    )
    mlm_logits, compat_logits = model(
        antibody_input_ids=antibody_batch["input_ids"],
        antibody_attention_mask=antibody_batch["attention_mask"],
        antigen_input_ids=antigen_input_ids,
        antigen_attention_mask=antigen_attention_mask,
    )
    return antibody_batch, mlm_logits, compat_logits


def test_compute_losses_mlm_loss_weight_default_matches_historical_total(tokenizer):
    """The default (weight 1.0) total is bit-exactly the historical
    ``mlm + w_c * compat`` reconstructed from the returned unweighted components,
    and passing 1.0 explicitly is identical to omitting it."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer))
    mlm_logits = torch.randn(2, 5, tokenizer.vocab_size)
    labels = torch.tensor([[-100, 3, -100, 4, -100], [10, -100, -100, -100, 11]])
    compat_logits = torch.randn(2, 2)
    compat_labels = torch.tensor([1, 0])
    compat_mask = torch.tensor([True, True])
    w_c = 0.5
    default = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=labels,
        compatibility_logits=compat_logits,
        compatibility_labels=compat_labels,
        compatibility_mask=compat_mask,
        compatibility_loss_weight=w_c,
    )
    expected = default["mlm_loss"] + w_c * default["compatibility_loss"]
    assert torch.equal(default["loss"], expected)
    explicit = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=labels,
        compatibility_logits=compat_logits,
        compatibility_labels=compat_labels,
        compatibility_mask=compat_mask,
        compatibility_loss_weight=w_c,
        mlm_loss_weight=1.0,
    )
    assert torch.equal(explicit["loss"], default["loss"])


def test_compute_losses_mlm_loss_weight_zero_drops_mlm_term(tokenizer):
    """``mlm_loss_weight=0.0`` leaves exactly the weighted compatibility term; the
    total still carries grad and backward reaches the compatibility head."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer))
    antibody_batch, mlm_logits, compat_logits = _tiny_dual_forward(tokenizer, model)
    w_c = 0.5
    losses = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=antibody_batch["labels"],
        compatibility_logits=compat_logits,
        compatibility_labels=torch.tensor([1, 0]),
        compatibility_mask=torch.tensor([True, True]),
        compatibility_loss_weight=w_c,
        mlm_loss_weight=0.0,
    )
    assert torch.equal(losses["loss"], w_c * losses["compatibility_loss"])
    assert losses["loss"].requires_grad
    losses["loss"].backward()
    assert model.compatibility_head.weight.grad is not None
    assert torch.any(model.compatibility_head.weight.grad != 0.0)


def test_compute_losses_mlm_loss_weight_zero_all_ignored_mlm_batch(tokenizer):
    """An all-ignored MLM batch AND ``mlm_loss_weight=0.0`` keeps the total finite
    (the differentiable-zero guard survives) and backward still runs."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer))
    antibody_batch, mlm_logits, compat_logits = _tiny_dual_forward(tokenizer, model)
    all_ignored = torch.full_like(antibody_batch["labels"], -100)
    losses = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=all_ignored,
        compatibility_logits=compat_logits,
        compatibility_labels=torch.tensor([1, 0]),
        compatibility_mask=torch.tensor([True, True]),
        compatibility_loss_weight=1.0,
        mlm_loss_weight=0.0,
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()  # must not raise


def test_joint_representation_cls_ignores_masks(tokenizer):
    """The default 'cls' readout is byte-for-byte the historical CLS-concat: the
    masks are ignored, so passing them changes nothing."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "cls"))
    model.eval()
    d = model.config.d_model
    ab = torch.randn(2, 4, d)
    ag = torch.randn(2, 3, d)
    mask_ab = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    mask_ag = torch.tensor([[1, 1, 0], [1, 1, 1]])
    without = model.joint_representation(ab, ag)
    with_masks = model.joint_representation(ab, ag, mask_ab, mask_ag)
    assert torch.equal(without, with_masks)


def test_joint_representation_mean_matches_hand_computation(tokenizer):
    """The 'mean' readout pools each stream over its unmasked positions ([CLS]
    included); pinned against an explicit per-row mean fed through fusion_mlp."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "mean"))
    model.eval()
    d = model.config.d_model
    ab = torch.randn(2, 4, d)
    ag = torch.randn(2, 4, d)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    def hand_pool(h):
        row0 = (h[0, 0] + h[0, 1] + h[0, 2]) / 3.0  # positions 0,1,2 unmasked
        row1 = (h[1, 0] + h[1, 1]) / 2.0  # positions 0,1 unmasked
        return torch.stack([row0, row1])

    expected = model.fusion_mlp(torch.cat([hand_pool(ab), hand_pool(ag)], dim=-1))
    actual = model.joint_representation(ab, ag, mask, mask)
    assert torch.equal(actual, expected)


def test_joint_representation_mean_ignores_padded_positions(tokenizer):
    """Perturbing ONLY masked positions leaves the mean-readout output unchanged."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "mean"))
    model.eval()
    d = model.config.d_model
    ab = torch.randn(2, 4, d)
    ag = torch.randn(2, 4, d)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    before = model.joint_representation(ab, ag, mask, mask)
    ab2, ag2 = ab.clone(), ag.clone()
    ab2[0, 3] += 5.0
    ab2[1, 2:] += 7.0
    ag2[0, 3] += 3.0
    ag2[1, 2:] += 9.0
    after = model.joint_representation(ab2, ag2, mask, mask)
    assert torch.equal(before, after)


def test_joint_representation_mean_requires_masks(tokenizer):
    """The 'mean' readout fails loud when a mask is missing (no silent CLS fallback)."""
    model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "mean"))
    model.eval()
    d = model.config.d_model
    ab = torch.randn(2, 4, d)
    ag = torch.randn(2, 4, d)
    with pytest.raises(ValueError, match="requires both attention masks"):
        model.joint_representation(ab, ag)
    with pytest.raises(ValueError, match="requires both attention masks"):
        model.joint_representation(ab, ag, torch.ones(2, 4, dtype=torch.long), None)


def test_compat_readout_adds_no_parameters(tokenizer):
    """``compat_readout`` adds ZERO parameters and draws ZERO extra init-RNG: at
    the same seed, 'cls' and 'mean' models have identical state_dict keys, shapes,
    and weights. This is exactly why a strict load cannot catch a readout
    mismatch, and why mlm_train validates the field against the checkpoint."""
    torch.manual_seed(20260717)
    cls_model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "cls"))
    torch.manual_seed(20260717)
    mean_model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "mean"))
    cls_sd = cls_model.state_dict()
    mean_sd = mean_model.state_dict()
    assert set(cls_sd.keys()) == set(mean_sd.keys())
    for key in cls_sd:
        assert cls_sd[key].shape == mean_sd[key].shape, key
        assert torch.equal(cls_sd[key], mean_sd[key]), key


def test_compat_readout_mean_changes_the_compatibility_logits(tokenizer):
    """Sanity: the two readouts are not accidentally the same function. Same
    weights, same inputs, different compatibility logits."""
    torch.manual_seed(7)
    cls_model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "cls"))
    torch.manual_seed(7)
    mean_model = AntibodyAntigenCrossAttention(_readout_config(tokenizer, "mean"))
    cls_model.eval()
    mean_model.eval()
    with torch.no_grad():
        cls_logits = _tiny_dual_forward(tokenizer, cls_model)[2]
        mean_logits = _tiny_dual_forward(tokenizer, mean_model)[2]
    assert not torch.allclose(cls_logits, mean_logits)


def test_invalid_compat_readout_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="compat_readout"):
        _readout_config(tokenizer, "bogus").validate()
