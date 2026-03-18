from __future__ import annotations

import torch
from torch.optim import AdamW

from smallAntibodyGen.data.MLMCollator import MLMCollator, OASRecord
from smallAntibodyGen.models.mlm import AntibodyMLM, MLMConfig
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

    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


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
    