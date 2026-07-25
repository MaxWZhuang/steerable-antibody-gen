"""Tests for the per-gamma steering-reachability probe (ported from the mirror)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data.MLMCollator import OASRecord
from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig

ANTIGEN = "MKTIIALSYIFCLVFADYKDDDDKAMDIGINSDPYQ"


def _load_probe(project_root: Path):
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "probe_steering_reachability", scripts_dir / "probe_steering_reachability.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(tokenizer, heavy_seq: str, heavy_cdr3: str) -> OASRecord:
    start = heavy_seq.index(heavy_cdr3)
    return OASRecord(
        sequence=heavy_seq,
        token_ids=tokenizer.encode_sequence(heavy_seq, locus="IGH", max_length=192),
        locus="IGH",
        chain_group="heavy",
        split="val",
        length=len(heavy_seq),
        token_length=len(heavy_seq) + 3,
        cdr3_start_aa=start,
        cdr3_end_aa=start + len(heavy_cdr3),
        cdr3_aa=heavy_cdr3,
        sequence_heavy=heavy_seq,
        heavy_locus="IGH",
        sequence_antigen=ANTIGEN,
        record_id="r0",
        target_key="uniprot:p1",
        is_strong_binder=True,
    )


def _infiller(tokenizer) -> FixedLengthHCDR3Infiller:
    torch.manual_seed(23)
    model = AntibodyAntigenCrossAttention(
        MLMConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_id,
            max_length=192,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.0,
        )
    )
    model.eval()
    return FixedLengthHCDR3Infiller(model, tokenizer, max_length=192, device="cpu")


def test_gamma_zero_is_exactly_the_unguided_distribution(
    project_root: Path, tokenizer, heavy_seq, heavy_cdr3
):
    """gamma == 0 must be a no-op by construction, not approximately."""
    probe = _load_probe(project_root)
    result = probe.reachability_at_position(
        _infiller(tokenizer),
        _record(tokenizer, heavy_seq, heavy_cdr3),
        position_index=0,
        gammas=[0.0, 1.0],
    )
    zero = result["gammas"][0]
    assert zero["gamma"] == 0.0
    assert zero["flipped"] is False
    assert zero["total_variation"] == pytest.approx(0.0, abs=1e-12)
    assert zero["delta_p_target"] == pytest.approx(0.0, abs=1e-12)


def test_probe_costs_exactly_two_forwards_regardless_of_gamma_count(
    project_root: Path, tokenizer, heavy_seq, heavy_cdr3
):
    """The whole point: every gamma comes from the same two cached vectors."""
    probe = _load_probe(project_root)
    infiller = _infiller(tokenizer)
    calls = {"n": 0}
    original = infiller.model.forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    infiller.model.forward = counting
    probe.reachability_at_position(
        infiller,
        _record(tokenizer, heavy_seq, heavy_cdr3),
        position_index=0,
        gammas=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
    )
    assert calls["n"] == 2


def test_probe_matches_the_infiller_combination_rule(
    project_root: Path, tokenizer, heavy_seq, heavy_cdr3
):
    """The probe must reproduce the SAME arithmetic guided_infill uses, or it is
    measuring a different function than the one that ships."""
    probe = _load_probe(project_root)
    infiller = _infiller(tokenizer)
    record = _record(tokenizer, heavy_seq, heavy_cdr3)
    gamma = 3.0

    result = probe.reachability_at_position(
        infiller, record, position_index=0, gammas=[gamma]
    )

    from smallAntibodyGen.infill.hcdr3 import HCDR3Span

    span = HCDR3Span.from_record(record)
    base_ids, base_attn, mask_positions, _, _ = (
        infiller._encode_antibody_with_masked_hcdr3(
            record, span, proposed_length=span.length
        )
    )
    antigen_ids, antigen_attn = infiller._encode_antigen(record)
    with torch.no_grad():
        guided, _ = infiller._guided_position_scores(
            base_ids,
            base_attn,
            antigen_ids,
            antigen_attn,
            mask_positions[0],
            guidance_strength=gamma,
        )
    expected_residue = infiller.tokenizer.id_to_token[
        infiller.canonical_token_ids[int(guided.argmax().item())]
    ]
    assert result["gammas"][0]["argmax_residue"] == expected_residue


def test_binder_spread_is_reported_as_the_ceiling(
    project_root: Path, tokenizer, heavy_seq, heavy_cdr3
):
    """A flat binder term means no gamma can reorder the residues; the spread is
    what makes that visible instead of inferred from a null result."""
    probe = _load_probe(project_root)
    result = probe.reachability_at_position(
        _infiller(tokenizer),
        _record(tokenizer, heavy_seq, heavy_cdr3),
        position_index=0,
        gammas=[0.0],
    )
    assert result["binder_spread"] >= 0.0
    assert result["binder_std"] >= 0.0


def test_out_of_range_position_fails_loud(
    project_root: Path, tokenizer, heavy_seq, heavy_cdr3
):
    probe = _load_probe(project_root)
    with pytest.raises(ValueError, match="out of range"):
        probe.reachability_at_position(
            _infiller(tokenizer),
            _record(tokenizer, heavy_seq, heavy_cdr3),
            position_index=999,
            gammas=[0.0],
        )


def test_summarize_curve_math(project_root: Path):
    probe = _load_probe(project_root)
    results = [
        {
            "binder_spread": 1.0,
            "gammas": [
                {"gamma": 0.0, "flipped": False, "total_variation": 0.0},
                {"gamma": 5.0, "flipped": True, "total_variation": 0.4},
            ],
        },
        {
            "binder_spread": 3.0,
            "gammas": [
                {"gamma": 0.0, "flipped": False, "total_variation": 0.0},
                {"gamma": 5.0, "flipped": False, "total_variation": 0.2},
            ],
        },
    ]
    summary = probe.summarize(results, [0.0, 5.0])
    assert summary["positions_probed"] == 2
    assert summary["binder_spread_median"] == pytest.approx(2.0)
    assert summary["binder_spread_max"] == pytest.approx(3.0)
    assert summary["curve"][0]["flip_fraction"] == pytest.approx(0.0)
    assert summary["curve"][1]["flip_fraction"] == pytest.approx(0.5)
    assert summary["curve"][1]["total_variation_median"] == pytest.approx(0.3)
