"""Historical post-LN weights must retain their computation during scoring."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from hcdr3_infill import load_dual_stream_model
from mlm_train import TrainConfig, build_model, build_tokenizer


@pytest.mark.parametrize("norm_first,omit_key", [(False, True), (False, False), (True, False)])
def test_checkpoint_load_preserves_logits(tmp_path, norm_first, omit_key):
    cfg = TrainConfig(
        data_path="unused", training_stage="antigen_hcdr3_infill_refine", init_checkpoint="parent.pt",
        max_length=32, antigen_max_length=32, d_model=16, n_heads=2,
        n_layers=1, d_ff=32, dropout=0.0, norm_first=norm_first,
    )
    tokenizer = build_tokenizer()
    original = build_model(tokenizer, cfg, torch.device("cpu")).eval()
    saved = dataclasses.asdict(cfg)
    if omit_key:
        del saved["norm_first"]
    path = tmp_path / "checkpoint.pt"
    torch.save({"train_config": saved, "model_state_dict": original.state_dict()}, path)
    restored, restored_cfg = load_dual_stream_model(
        path, data_path="unused", device=torch.device("cpu"),
    )
    assert restored_cfg.norm_first is norm_first
    ids = torch.tensor([[tokenizer.cls_id, tokenizer.token_to_id["A"], tokenizer.mask_id,
                         tokenizer.token_to_id["G"], tokenizer.eos_id]])
    inputs = dict(antibody_input_ids=ids, antibody_attention_mask=torch.ones_like(ids),
                  antigen_input_ids=ids, antigen_attention_mask=torch.ones_like(ids))
    with torch.no_grad():
        before, after = original(**inputs), restored(**inputs)
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_missing_weights_still_fail_strict_loading(tmp_path):
    cfg = TrainConfig(data_path="unused", training_stage="antigen_hcdr3_infill_refine", init_checkpoint="parent.pt",
                      max_length=16, d_model=8, n_heads=2, n_layers=1, d_ff=16,
                      norm_first=False)
    model = build_model(build_tokenizer(), cfg, torch.device("cpu"))
    state = model.state_dict()
    del state["fusion_norm_antibody.weight"]
    path = tmp_path / "damaged.pt"
    saved = dataclasses.asdict(cfg)
    del saved["norm_first"]
    torch.save({"train_config": saved, "model_state_dict": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_dual_stream_model(path, data_path="unused", device=torch.device("cpu"))
