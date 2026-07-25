"""Tests for graded-affinity supervision, ported from the sibling mirror repo.

Covers three landings that must all be default-off:

- ``data/affinity.py``, the single home of the strong-binder decision tree that
  the producer and the reader now both delegate to;
- the conditional ``strength_head`` and its masked-MSE loss;
- ``scripts/annotate_affinity_targets.py``, the train-split-only quantile
  annotator.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from smallAntibodyGen.data import affinity as affinity_rules
from smallAntibodyGen.data.MLMCollator import (
    AntibodyAntigenCollator,
    OASRecord,
    OASSequenceDataset,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig


def _load_script(project_root: Path, name: str):
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- consolidation module


def test_producer_and_reader_share_one_decision_tree(project_root: Path):
    """The whole point of ``data/affinity.py``: the producer in
    prepare_antibody_antigen.py and the reader in MLMCollator.py must reach the
    same verdict for the same row, because they now call the same function."""
    prepare = _load_script(project_root, "prepare_antibody_antigen")
    cases = [
        # (affinity_type, affinity_raw, processed_measurement, expected)
        ("bool", None, 1, True),
        ("bool", None, 0, False),
        ("fuzzy", "h", None, True),
        ("fuzzy", "l", None, False),
        ("fuzzy", "l", "h", True),  # measurement wins over raw
        ("kd", None, 1e-10, True),
        ("kd", None, 1e-8, False),
        ("kd", None, 0.5, True),  # nanomolar encoding
        ("kd", None, 5.0, False),
        ("-log kd", None, 9.5, True),
        ("-log kd", None, 8.5, False),
        ("ddg", None, -3.0, False),
    ]
    for affinity_type, raw, measurement, expected in cases:
        assert prepare.infer_is_strong_binder(affinity_type, raw, measurement) is expected

        record = {
            "affinity_type": affinity_type,
            "affinity_raw": raw,
            "processed_measurement_raw": measurement,
            "processed_measurement_float": (
                measurement
                if isinstance(measurement, (int, float)) and not isinstance(measurement, bool)
                else None
            ),
            "binder_label": (
                (1 if measurement == 1 else 0) if affinity_type == "bool" else None
            ),
        }
        assert OASSequenceDataset._infer_is_strong_binder(record) is expected


def test_stored_is_strong_binder_wins_over_the_fallback():
    record = {"is_strong_binder": True, "affinity_type": "kd", "processed_measurement_float": 1.0}
    assert affinity_rules.infer_is_strong_binder(record) is True
    record["is_strong_binder"] = False
    assert affinity_rules.infer_is_strong_binder(record) is False


def test_kd_unit_encodings_produce_the_same_strength_score():
    """1 nM written in molar and in nanomolar must score identically; without the
    normalization the nanomolar row would sit at 0.0 and never clear 9.0."""
    molar = affinity_rules.base_affinity_strength_score(
        {"affinity_type": "kd", "processed_measurement_float": 1e-9}
    )
    nano = affinity_rules.base_affinity_strength_score(
        {"affinity_type": "kd", "processed_measurement_float": 1.0}
    )
    assert molar == pytest.approx(9.0)
    assert nano == pytest.approx(9.0)


# ----------------------------------------------------------------- the head


def _config(tokenizer, use_strength_head: bool) -> MLMConfig:
    return MLMConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        use_strength_head=use_strength_head,
    )


def test_strength_head_off_draws_zero_extra_init_rng(tokenizer):
    """Conditional construction, not a zeroed head: at the same seed the
    default model's weights must be bit-identical to what they were before the
    head existed, which means the OFF model must draw no extra RNG."""
    torch.manual_seed(31337)
    off = AntibodyAntigenCrossAttention(_config(tokenizer, False))
    torch.manual_seed(31337)
    on = AntibodyAntigenCrossAttention(_config(tokenizer, True))

    assert "strength_head.weight" not in off.state_dict()
    assert "strength_head.weight" in on.state_dict()
    off_sd = off.state_dict()
    for key, value in off_sd.items():
        # Every shared parameter is drawn before the conditional head, so ON and
        # OFF agree on all of them.
        assert torch.equal(value, on.state_dict()[key]), key


def test_forward_keeps_the_two_tuple_by_default(tokenizer):
    model = AntibodyAntigenCrossAttention(_config(tokenizer, True))
    model.eval()
    ab = torch.randint(4, 20, (2, 9))
    ag = torch.randint(4, 20, (2, 7))
    mask_ab = torch.ones_like(ab)
    mask_ag = torch.ones_like(ag)
    with torch.no_grad():
        default = model(ab, mask_ab, ag, mask_ag)
        opted_in = model(ab, mask_ab, ag, mask_ag, return_strength=True)
    assert len(default) == 2
    assert len(opted_in) == 3
    assert opted_in[2].shape == (2,)


def test_predict_strength_without_a_head_fails_loud(tokenizer):
    model = AntibodyAntigenCrossAttention(_config(tokenizer, False))
    with pytest.raises(RuntimeError, match="use_strength_head=True"):
        model.predict_strength(torch.randn(2, 32))
    # ...and the opt-in forward returns None rather than raising, so a caller
    # that asks for strength on a headless checkpoint gets a usable answer.
    with torch.no_grad():
        out = model(
            torch.randint(4, 20, (2, 9)),
            torch.ones(2, 9, dtype=torch.long),
            torch.randint(4, 20, (2, 7)),
            torch.ones(2, 7, dtype=torch.long),
            return_strength=True,
        )
    assert out[2] is None


def test_strength_loss_is_masked_and_hand_checkable(tokenizer):
    model = AntibodyAntigenCrossAttention(_config(tokenizer, True))
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 0.0, 3.5])
    mask = torch.tensor([True, False, True])
    loss = model.compute_strength_loss(predictions, targets, mask)
    # Only rows 0 and 2 contribute: mean((1.0-1.5)^2, (3.0-3.5)^2) = 0.25
    assert float(loss) == pytest.approx(0.25)


def test_all_masked_strength_batch_is_a_differentiable_zero(tokenizer):
    model = AntibodyAntigenCrossAttention(_config(tokenizer, True))
    predictions = (model.strength_head.weight.sum() * torch.ones(3)).float()
    targets = torch.zeros(3)
    mask = torch.zeros(3, dtype=torch.bool)
    loss = model.compute_strength_loss(predictions, targets, mask)
    assert float(loss) == 0.0
    assert loss.requires_grad
    loss.backward()  # must not raise, and must not be NaN


def test_compute_losses_default_off_is_numerically_identical(tokenizer):
    model = AntibodyAntigenCrossAttention(_config(tokenizer, False))
    mlm_logits = torch.randn(2, 5, tokenizer.vocab_size)
    labels = torch.tensor([[-100, 3, -100, 4, -100], [10, -100, -100, -100, 11]])
    compat_logits = torch.randn(2, 2)
    compat_labels = torch.tensor([1, 0])
    losses = model.compute_losses(
        mlm_logits=mlm_logits,
        labels=labels,
        compatibility_logits=compat_logits,
        compatibility_labels=compat_labels,
    )
    expected = losses["mlm_loss"] + losses["compatibility_loss"]
    assert torch.equal(losses["loss"], expected)
    assert float(losses["strength_loss"]) == 0.0


# ------------------------------------------------------------- collator side


def _antigen_record(
    sequence: str,
    quantile: float | None,
    strong: bool = True,
    antigen: str = "MKTIIALSYIFCLVFADYKDDDDK",
    target_key: str = "uniprot:p1",
) -> OASRecord:
    return OASRecord(
        sequence=sequence,
        locus="PAIRED_ANTIGEN",
        chain_group="paired_antigen",
        split="train",
        length=len(sequence),
        token_length=len(sequence) + 3,
        sequence_heavy=sequence,
        heavy_locus="IGH",
        sequence_antigen=antigen,
        is_paired=False,
        is_strong_binder=strong,
        affinity_family="ranking_regression",
        affinity_strength_quantile=quantile,
        target_key=target_key,
        record_id=f"{target_key}:{sequence[:4]}",
        dataset_name="ds",
    )


def test_batch_emits_strength_targets_and_masks_absent_quantiles(tokenizer):
    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=96,
        shuffle_antigen_probability=0.0,
        rng_seed=1,
    )
    batch = collator(
        [
            _antigen_record("EVQLVESGGGLVQPGGSLRLSCAAS", 0.9),
            _antigen_record("QVQLQESGPGLVKPSETLSLTCTVS", None),
        ]
    )
    assert batch["strength_mask"].tolist() == [True, False]
    assert batch["strength_targets"][0].item() == pytest.approx(0.9)
    # Absent quantile is stored as 0.0 -- the MASK is what makes it inert, and a
    # sentinel value would be indistinguishable from a genuine weakest row.
    assert batch["strength_targets"][1].item() == 0.0
    assert batch["affinity_family_ids"].tolist() == [
        affinity_rules.AFFINITY_FAMILY_IDS["ranking_regression"]
    ] * 2


def test_shuffled_antigen_rows_are_masked_out_of_strength_supervision(tokenizer):
    """A shuffled row keeps the antibody's affinity for its ORIGINAL antigen, so
    training the head on it teaches antigen-independent strength -- exactly what
    the dual stream exists to avoid."""
    collator = AntibodyAntigenCollator(
        tokenizer=tokenizer,
        max_length=96,
        shuffle_antigen_probability=1.0,
        rng_seed=1,
    )
    batch = collator(
        [
            _antigen_record(
                "EVQLVESGGGLVQPGGSLRLSCAAS",
                0.9,
                antigen="MKTIIALSYIFCLVFADYKDDDDK",
                target_key="uniprot:p1",
            ),
            _antigen_record(
                "QVQLQESGPGLVKPSETLSLTCTVS",
                0.8,
                antigen="AMDIGINSDPYQNVKLLTQFGWKA",
                target_key="uniprot:p2",
            ),
        ]
    )
    shuffled = batch["is_shuffled_antigen"]
    assert bool(shuffled.any())
    assert not bool((batch["strength_mask"] & shuffled).any())


# --------------------------------------------------------------- the annotator


def _write_corpus(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_corpus(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _kd_row(split: str, kd: float, idx: int) -> dict:
    return {
        "record_id": f"{split}{idx}",
        "split": split,
        "dataset_name": "ds",
        "affinity_type": "kd",
        "processed_measurement_float": kd,
        "sequence_heavy": "EVQLVESGGG",
        "sequence_antigen": "MKTIIALSY",
    }


def test_annotator_fits_on_train_only_and_orients_strong_high(
    tmp_path: Path, project_root: Path
):
    annotate = _load_script(project_root, "annotate_affinity_targets")
    rows = [_kd_row("train", 10.0 ** (-6 - i * 0.1), i) for i in range(30)]
    rows += [_kd_row("val", 1e-9, 100), _kd_row("val", 1e-6, 101)]
    src = tmp_path / "in.jsonl.gz"
    dst = tmp_path / "out.jsonl.gz"
    _write_corpus(src, rows)

    stats = annotate.annotate(src, dst, min_group_size=20)
    out = _read_corpus(dst)
    assert stats["rows_annotated"] == len(rows)

    by_id = {row["record_id"]: row for row in out}
    # Lower KD == stronger binder == higher quantile.
    assert by_id["val100"]["affinity_strength_quantile"] > by_id["val101"][
        "affinity_strength_quantile"
    ]
    for row in out:
        assert 0.0 <= row["affinity_strength_quantile"] <= 1.0


def test_annotator_excludes_small_groups(tmp_path: Path, project_root: Path):
    annotate = _load_script(project_root, "annotate_affinity_targets")
    rows = [_kd_row("train", 1e-9 * (i + 1), i) for i in range(5)]
    src = tmp_path / "in.jsonl.gz"
    dst = tmp_path / "out.jsonl.gz"
    _write_corpus(src, rows)
    stats = annotate.annotate(src, dst, min_group_size=20)
    assert stats["rows_annotated"] == 0
    assert all("affinity_strength_quantile" not in row for row in _read_corpus(dst))


def test_annotator_ties_share_one_quantile(tmp_path: Path, project_root: Path):
    annotate = _load_script(project_root, "annotate_affinity_targets")
    rows = [_kd_row("train", 1e-9, i) for i in range(25)]
    src = tmp_path / "in.jsonl.gz"
    dst = tmp_path / "out.jsonl.gz"
    _write_corpus(src, rows)
    annotate.annotate(src, dst, min_group_size=20)
    quantiles = {row["affinity_strength_quantile"] for row in _read_corpus(dst)}
    assert len(quantiles) == 1


def test_annotator_is_additive_only(tmp_path: Path, project_root: Path):
    """Stripping the added key must reproduce the input exactly -- otherwise the
    annotator is a data migration, not an annotation."""
    annotate = _load_script(project_root, "annotate_affinity_targets")
    rows = [_kd_row("train", 1e-9 * (i + 1), i) for i in range(25)]
    src = tmp_path / "in.jsonl.gz"
    dst = tmp_path / "out.jsonl.gz"
    _write_corpus(src, rows)
    annotate.annotate(src, dst, min_group_size=20)
    stripped = [
        {k: v for k, v in row.items() if k != "affinity_strength_quantile"}
        for row in _read_corpus(dst)
    ]
    assert stripped == rows


def test_annotator_refuses_to_write_in_place(tmp_path: Path, project_root: Path):
    annotate = _load_script(project_root, "annotate_affinity_targets")
    src = tmp_path / "corpus.jsonl.gz"
    _write_corpus(src, [_kd_row("train", 1e-9, 0)])
    with pytest.raises(ValueError, match="in place"):
        annotate.annotate(src, src, min_group_size=20)


# -------------------------------------------------------------- config knobs


def test_strength_knobs_default_off_and_validate(tmp_path: Path, project_root: Path):
    mlm_train = _load_script(project_root, "mlm_train")
    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")

    cfg = mlm_train.parse_args(["--data-path", str(data_path)])
    assert cfg.strength_loss_weight == 0.0
    assert cfg.include_strength_rows is False

    with pytest.raises(ValueError, match="only supported for antigen stages"):
        mlm_train.parse_args(
            ["--data-path", str(data_path), "--strength-loss-weight", "0.5"]
        )


def test_use_strength_head_is_derived_from_the_weight(tmp_path: Path, project_root: Path):
    mlm_train = _load_script(project_root, "mlm_train")
    from smallAntibodyGen.tokenizer import AminoAcidTokenizer

    data_path = tmp_path / "tiny.jsonl.gz"
    data_path.write_text("", encoding="utf-8")
    ckpt = tmp_path / "init.pt"
    ckpt.write_text("", encoding="utf-8")

    args = [
        "--data-path",
        str(data_path),
        "--training-stage",
        "antigen_real_label_refine",
        "--init-checkpoint",
        str(ckpt),
    ]
    off = mlm_train.parse_args(args)
    on = mlm_train.parse_args(args + ["--strength-loss-weight", "0.5"])
    tok = AminoAcidTokenizer()
    assert mlm_train.build_model(tok, off, torch.device("cpu")).config.use_strength_head is False
    assert mlm_train.build_model(tok, on, torch.device("cpu")).config.use_strength_head is True


def test_tie_aware_spearman_is_hand_verified(project_root: Path):
    mlm_train = _load_script(project_root, "mlm_train")
    # Perfectly monotone -> 1.0; reversed -> -1.0.
    assert mlm_train.tie_aware_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert mlm_train.tie_aware_spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # All-tied on one side is undefined, not zero.
    assert mlm_train.tie_aware_spearman([1, 1, 1], [1, 2, 3]) != mlm_train.tie_aware_spearman(
        [1, 1, 1], [1, 2, 3]
    )
    # Mid-ranks: x ties on the first pair, so order within the tie cannot matter.
    a = mlm_train.tie_aware_spearman([1, 1, 3], [5, 6, 7])
    b = mlm_train.tie_aware_spearman([1, 1, 3], [6, 5, 7])
    assert a == pytest.approx(b)
    assert mlm_train.tie_aware_spearman([1.0], [2.0]) != mlm_train.tie_aware_spearman([1.0], [2.0])
