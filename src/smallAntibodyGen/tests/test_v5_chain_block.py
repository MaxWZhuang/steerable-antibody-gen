"""
The v5 chain carries ONE block, and it is the block J11 selected.

J11 chose a width. Until the chain configs actually carry it, that choice lives
only in a spec: `configs/pretrain_oas_small.yaml` and its three successors
declared `norm_first` and nothing else, so the canonical chain would have
trained the legacy block while the evidence described a modern one.

Two invariants, and the second is the one that bites:

- **The chain runs the promoted block.** RoPE, pre-RMSNorm, SwiGLU at 680, no
  attention/FFN biases.
- **All four stages agree on it, exactly.** Each stage warm-starts from the
  previous stage's `best.pt`. A block field that differs between two stages is a
  warm start across an architecture boundary; `strict=True` and the fingerprint
  check would catch it at run time, but only after the previous stage has been
  trained -- which on this card is hours to days. Catching it here costs
  milliseconds.

The stage-4 ESM ablation is held to the same block, because its ONE varied axis
is the antigen encoder (Rule 3). A block difference would make a
scratch-vs-ESM gap partly a block effect.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

#: The block J11 promoted. `swiglu_hidden_dim` is explicit and never inferred
#: from `d_ff`: accidental capacity is exactly what J11 measured the cost of.
PROMOTED_BLOCK = {
    "position_encoding": "rope",
    "norm_type": "rmsnorm",
    "norm_first": True,
    "ffn_type": "swiglu",
    "attention_bias": False,
    "ffn_bias": False,
    "swiglu_hidden_dim": 680,
}

#: Stage 1 -> 4, in warm-start order.
CHAIN = (
    "pretrain_oas_small.yaml",
    "refine_oas_paired.yaml",
    "refine_antigen_real_label.yaml",
    "refine_antigen_hcdr3_infill.yaml",
)
#: A stage-4 ablation, not a chain link -- but bound to the same block.
ABLATION = "refine_antigen_hcdr3_infill_esm.yaml"


def _config(project_root: Path, name: str) -> dict:
    path = project_root.parents[1] / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("name", CHAIN + (ABLATION,))
@pytest.mark.parametrize("field", sorted(PROMOTED_BLOCK))
def test_every_v5_config_declares_the_promoted_block(project_root: Path, name, field):
    """
    Declared, not defaulted. An absent key inherits whatever the dataclass
    default happens to be, which is the LEGACY block -- so a silent omission
    trains the thing J11 measured against rather than the thing it selected.
    """
    cfg = _config(project_root, name)
    assert field in cfg, f"{name} does not declare {field}"
    assert cfg[field] == PROMOTED_BLOCK[field], (
        f"{name}: {field}={cfg[field]!r}, expected {PROMOTED_BLOCK[field]!r}"
    )


@pytest.mark.parametrize("field", sorted(PROMOTED_BLOCK))
def test_the_chain_agrees_on_every_block_field(project_root: Path, field):
    """
    Each stage warm-starts from the previous one. A block field that differs
    between two stages is a warm start across an architecture boundary, and the
    cost of finding that at run time is the previous stage's training time.
    """
    values = {name: _config(project_root, name).get(field) for name in CHAIN}
    assert len(set(values.values())) == 1, f"{field} differs across the chain: {values}"


def test_the_esm_ablation_matches_the_scratch_stage_four_block(project_root: Path):
    """
    The ablation's one varied axis is the antigen encoder. If the block also
    moved, a scratch-vs-ESM gap would be partly a block effect -- the confound
    the stage-4 header already warns about for antigen context length.
    """
    scratch = _config(project_root, "refine_antigen_hcdr3_infill.yaml")
    esm = _config(project_root, ABLATION)
    for field in PROMOTED_BLOCK:
        assert esm.get(field) == scratch.get(field), (
            f"{ABLATION}: {field}={esm.get(field)!r} != scratch {scratch.get(field)!r}"
        )


def test_the_width_matches_the_one_j11_selected(project_root: Path):
    """
    Pinned against the evidence rather than against a remembered number. If the
    promotion report ever says something else, this fails instead of the chain
    quietly training a width nothing chose.
    """
    import json

    report = json.loads(
        (project_root.parents[1] / "specs/evidence/j11-comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["selected_width"] == PROMOTED_BLOCK["swiglu_hidden_dim"]
    for name in CHAIN:
        assert _config(project_root, name)["swiglu_hidden_dim"] == report["selected_width"]


@pytest.mark.parametrize("name", CHAIN + (ABLATION,))
def test_the_v5_generation_is_self_consistent(project_root: Path, name):
    """
    Output dirs and warm-start parents all carry the `_v5` suffix. A stage
    pointing at another generation's checkpoint is the mixing Rule 1 forbids,
    and it is a one-character typo away at all times.
    """
    cfg = _config(project_root, name)
    assert cfg["output_dir"].endswith("_v5"), cfg["output_dir"]
    parent = cfg.get("init_checkpoint")
    if parent is not None:
        assert "_v5/" in parent, f"{name} warm-starts outside the v5 generation: {parent}"
