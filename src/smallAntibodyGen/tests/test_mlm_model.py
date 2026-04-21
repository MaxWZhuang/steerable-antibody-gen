from __future__ import annotations

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
