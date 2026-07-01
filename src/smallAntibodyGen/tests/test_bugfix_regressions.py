"""
Regression tests for the bug-fix pass (Part 1 + Part 2).

Each test locks in one truth-preservation fix: labels mean what they say,
metrics pool correctly, resume restores RNG/scaler state, generation runs
trained weights and reports comparable scores, and lossy/edge cases fail loud
rather than silently. They are written to fail against the pre-fix behavior.
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_script(project_root: Path, name: str):
    """Import a scripts/*.py module by name (scripts/ added to sys.path)."""
    scripts_dir = project_root.parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_mlm_config(tokenizer, **overrides):
    from smallAntibodyGen.models.mlm import MLMConfig

    params = dict(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    params.update(overrides)
    return MLMConfig(**params)


# --------------------------------------------------------------------------- #
# KD units / label correctness
# --------------------------------------------------------------------------- #
def test_kd_strong_binder_nanomolar_and_molar(project_root):
    paa = _load_script(project_root, "prepare_antibody_antigen")
    f = paa.infer_is_strong_binder
    # Nanomolar-encoded (the ASD majority): a sub-/low-nM KD is a strong binder.
    assert f("kd", None, 0.39) is True
    assert f("kd", None, 0.5) is True
    assert f("kd", None, 5.0) is False
    # Molar-encoded subset: behavior preserved (1 nM strong, 10 nM not).
    assert f("kd", None, 1e-9) is True
    assert f("kd", None, 1e-8) is False
    # Degenerate: non-positive measurement is never a strong binder.
    assert f("kd", None, 0.0) is False


def test_fuzzy_zero_measurement_is_not_substituted(project_root):
    paa = _load_script(project_root, "prepare_antibody_antigen")
    # processed_measurement 0.0 must be used (-> "0.0" != "h"), NOT replaced by
    # affinity_raw="h" via a falsy `or`. Pre-fix this returned True.
    assert paa.infer_is_strong_binder("fuzzy", "h", 0.0) is False
    # A genuine high-binder marker still resolves to True.
    assert paa.infer_is_strong_binder("fuzzy", None, "h") is True


# --------------------------------------------------------------------------- #
# Leakage-aware split: canonical identity grouping
# --------------------------------------------------------------------------- #
def test_canonicalize_accession_collapses_variants(project_root):
    paa = _load_script(project_root, "prepare_antibody_antigen")
    c = paa.canonicalize_accession
    assert c("P12345") == c("P12345-2") == "p12345"
    assert c("6XYZ") == c("6xyz_A") == c("6xyz.A") == "6xyz"


def test_build_target_key_groups_isoform_variants(project_root):
    paa = _load_script(project_root, "prepare_antibody_antigen")
    key_a = paa.build_target_key({"target_uniprot": "P12345"}, "AAAA")
    key_b = paa.build_target_key({"target_uniprot": "P12345-2"}, "CCCC")
    assert key_a == key_b  # same biological target -> same split bucket


# --------------------------------------------------------------------------- #
# CDR3 span integrity
# --------------------------------------------------------------------------- #
def test_locate_cdr3_span_unique_and_duplicate(project_root):
    paa = _load_script(project_root, "prepare_antibody_antigen")
    seq = "AAAACDEFGHYYYY"
    start, end = paa.locate_cdr3_span(seq, "CDEFGH")
    assert (start, end) == (4, 10)
    assert seq[start:end] == "CDEFGH"  # invariant: span slices back to the CDR3
    # Ambiguous (appears twice) -> no span, never a wrong guess.
    assert paa.locate_cdr3_span("CDECDE", "CDE") == (None, None)


# --------------------------------------------------------------------------- #
# NaN-empty-batch loss guard
# --------------------------------------------------------------------------- #
def test_mlm_loss_all_ignored_is_finite_zero(tokenizer):
    from smallAntibodyGen.models.mlm import AntibodyMLM

    model = AntibodyMLM(_tiny_mlm_config(tokenizer))
    logits = torch.randn(2, 5, tokenizer.vocab_size, requires_grad=True)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    loss = model.compute_loss(logits, labels)
    assert torch.isfinite(loss)  # pre-fix: 0/0 == NaN
    assert float(loss.detach()) == 0.0
    loss.backward()  # differentiable zero, no NaN gradients


def test_cross_attention_mlm_loss_all_ignored_is_finite_zero(tokenizer):
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    model = AntibodyAntigenCrossAttention(_tiny_mlm_config(tokenizer))
    logits = torch.randn(2, 5, tokenizer.vocab_size, requires_grad=True)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    loss = model.compute_mlm_loss(logits, labels)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0


# --------------------------------------------------------------------------- #
# Metric pooling
# --------------------------------------------------------------------------- #
def test_masked_accuracy_counts_pool_correctly(project_root):
    mlm_train = _load_script(project_root, "mlm_train")
    vocab = 4
    # Batch 1: 1 masked token, correct. Batch 2: 3 masked tokens, all wrong.
    logits1 = torch.zeros(1, 1, vocab)
    logits1[0, 0, 2] = 10.0
    labels1 = torch.tensor([[2]])
    logits2 = torch.zeros(1, 3, vocab)
    logits2[:, :, 0] = 10.0
    labels2 = torch.tensor([[1, 1, 1]])

    c1, t1 = mlm_train.masked_accuracy_counts(logits1, labels1)
    c2, t2 = mlm_train.masked_accuracy_counts(logits2, labels2)
    assert (c1, t1) == (1, 1)
    assert (c2, t2) == (0, 3)
    pooled = (c1 + c2) / (t1 + t2)
    mean_of_means = (c1 / t1 + c2 / t2) / 2
    assert pooled == pytest.approx(0.25)      # token-pooled truth
    assert mean_of_means == pytest.approx(0.5)  # the biased value we removed
    assert pooled != mean_of_means


# --------------------------------------------------------------------------- #
# Optimizer: no weight decay on 1-D params
# --------------------------------------------------------------------------- #
def test_build_optimizer_excludes_norms_and_biases_from_decay(project_root):
    mlm_train = _load_script(project_root, "mlm_train")
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
    cfg = mlm_train.TrainConfig(data_path="x", weight_decay=0.123)
    opt = mlm_train.build_optimizer(model, cfg)

    groups = {g["weight_decay"]: g for g in opt.param_groups}
    assert set(groups) == {0.123, 0.0}
    n_no_decay = sum(p.dim() < 2 for p in model.parameters())
    n_decay = sum(p.dim() >= 2 for p in model.parameters())
    assert len(groups[0.0]["params"]) == n_no_decay  # both biases + LayerNorm w/b
    assert len(groups[0.123]["params"]) == n_decay


# --------------------------------------------------------------------------- #
# Warmup scheduler honored
# --------------------------------------------------------------------------- #
def test_warmup_scheduler_follows_linear_ramp(project_root):
    mlm_train = _load_script(project_root, "mlm_train")
    cfg = mlm_train.TrainConfig(data_path="x", learning_rate=0.1, warmup_steps=4)
    model = torch.nn.Linear(3, 3)
    opt = mlm_train.build_optimizer(model, cfg)
    sched = mlm_train.build_lr_scheduler(opt, cfg)
    assert sched is not None

    import warnings

    seen = []
    with warnings.catch_warnings():
        # The scheduler is stepped without a paired optimizer.step() in this
        # isolated test; that ordering warning is irrelevant to the LR values.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(5):
            seen.append(opt.param_groups[0]["lr"])
            sched.step()
    assert seen == pytest.approx([0.025, 0.05, 0.075, 0.1, 0.1])


def test_warmup_disabled_returns_no_scheduler(project_root):
    mlm_train = _load_script(project_root, "mlm_train")
    cfg = mlm_train.TrainConfig(data_path="x", warmup_steps=0)
    opt = mlm_train.build_optimizer(torch.nn.Linear(2, 2), cfg)
    assert mlm_train.build_lr_scheduler(opt, cfg) is None


# --------------------------------------------------------------------------- #
# Checkpoint state: scaler + RNG round-trip (faithful resume)
# --------------------------------------------------------------------------- #
def test_checkpoint_persists_and_restores_rng_and_scaler(project_root, tmp_path):
    mlm_train = _load_script(project_root, "mlm_train")
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cfg = mlm_train.TrainConfig(data_path="x")

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    _ = [random.random() for _ in range(5)]
    py_state, np_state, th_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected_py = [random.random() for _ in range(3)]
    expected_th = torch.rand(3)
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(th_state)

    ckpt_path = tmp_path / "last.pt"
    mlm_train.save_checkpoint(ckpt_path, model, opt, cfg, epoch=1, val_loss=0.5,
                              scaler=scaler, scheduler=None)

    payload = torch.load(ckpt_path, map_location="cpu")
    assert "scaler_state_dict" in payload and payload["scaler_state_dict"] is not None
    assert "scheduler_state_dict" in payload
    assert payload["rng_state"] is not None

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    mlm_train.restore_rng_state(payload["rng_state"])
    assert [random.random() for _ in range(3)] == expected_py
    assert torch.allclose(torch.rand(3), expected_th)


# --------------------------------------------------------------------------- #
# Tokenizer: lossy truncation must warn
# --------------------------------------------------------------------------- #
def test_tokenizer_truncation_warns(tokenizer):
    long_seq = "A" * 200
    with pytest.warns(UserWarning, match="truncated"):
        tokenizer.encode_sequence(long_seq, locus="IGH", max_length=32)
    # No warning when it fits.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tokenizer.encode_sequence("ACDEF", locus="IGH", max_length=64)


# --------------------------------------------------------------------------- #
# Generation: length-normalized, comparable ranking score
# --------------------------------------------------------------------------- #
def test_infill_candidate_reports_mean_log_probability(tokenizer):
    from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller
    from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention

    torch.manual_seed(0)
    model = AntibodyAntigenCrossAttention(_tiny_mlm_config(tokenizer))
    record = types.SimpleNamespace(
        sequence_heavy="AAAACDEFGHIKLMYYYY",
        sequence="AAAACDEFGHIKLMYYYY",
        sequence_light=None,
        sequence_antigen="ACDEFGHIKL",
        heavy_locus="IGH",
        light_locus=None,
        cdr3_aa_heavy="CDEFGH",
        cdr3_start_aa_heavy=4,
        cdr3_end_aa_heavy=10,
    )
    infiller = FixedLengthHCDR3Infiller(model, tokenizer, max_length=64, device="cpu")
    candidates = infiller.infill(record, num_samples=3, temperature=1.0, top_k=None)
    assert len(candidates) == 3
    for cand in candidates:
        assert len(cand.generated_hcdr3) == 6
        assert cand.length == 6
        assert cand.log_probability <= 0.0  # sum of log-probs
        assert cand.mean_log_probability == pytest.approx(cand.log_probability / cand.length)


# --------------------------------------------------------------------------- #
# Encoding: heavy-only chain token agrees across train and inference
# --------------------------------------------------------------------------- #
def test_heavy_only_chain_token_matches_train_and_inference(tokenizer):
    """
    Heavy-only antibody-antigen records (e.g. nanobodies) carry the generic
    locus "PAIRED_ANTIGEN". The training collator and the inference infiller
    must encode them with the *same* chain token at position 1. Pre-fix the
    collator tokenized "PAIRED_ANTIGEN" to [OTHER_CHAIN] while the infiller used
    heavy_locus -> [IGH], a first-position train/inference mismatch affecting
    every heavy-only row (~24% of the processed dataset).
    """
    from smallAntibodyGen.data.MLMCollator import (
        AntibodyAntigenRealLabelCollator,
        OASRecord,
    )
    from smallAntibodyGen.infill.hcdr3 import FixedLengthHCDR3Infiller, HCDR3Span

    rec = OASRecord(
        sequence="QVQLAAAACARDRSTYYYY",
        locus="PAIRED_ANTIGEN",
        chain_group="paired_antigen",
        split="train",
        length=19,
        sequence_heavy="QVQLAAAACARDRSTYYYY",
        sequence_light=None,
        heavy_locus="IGH",
        binder_label=1,
        is_strong_binder=True,
        sequence_antigen="MKTIIALSYIFCLVFA",
        cdr3_aa_heavy="CARDRST",
        cdr3_start_aa_heavy=8,
        cdr3_end_aa_heavy=15,
        record_id="r1",
        target_key="uniprot:p1",
        is_paired=False,
    )
    igh = tokenizer.token_to_id["[IGH]"]
    other = tokenizer.token_to_id["[OTHER_CHAIN]"]

    collator = AntibodyAntigenRealLabelCollator(
        tokenizer,
        max_length=64,
        hcdr3_mask_mode="full_span",
        mask_replacement_strategy="always_mask",
    )
    train_batch = collator([rec])
    # Position 0 is [CLS]; position 1 is the chain token.
    train_chain_token = int(train_batch["antibody_input_ids"][0, 1])

    infiller = FixedLengthHCDR3Infiller(torch.nn.Identity(), tokenizer, max_length=64)
    span = HCDR3Span.from_record(rec)
    ids, _, _, _, _ = infiller._encode_antibody_with_masked_hcdr3(
        rec, span, proposed_length=7
    )
    infer_chain_token = int(ids[0, 1])

    assert train_chain_token == igh        # not [OTHER_CHAIN]
    assert train_chain_token != other
    assert infer_chain_token == igh
    assert train_chain_token == infer_chain_token  # the core invariant


# --------------------------------------------------------------------------- #
# Generation: strict checkpoint load + arch guard
# --------------------------------------------------------------------------- #
def test_load_dual_stream_rejects_non_antigen_checkpoint(project_root, tmp_path):
    hcdr3_infill = _load_script(project_root, "hcdr3_infill")
    ckpt = tmp_path / "base.pt"
    torch.save({"model_state_dict": {}, "train_config": {"training_stage": "base"}}, ckpt)
    with pytest.raises(ValueError, match="dual-stream"):
        hcdr3_infill.load_dual_stream_model(ckpt, data_path="d", device=torch.device("cpu"))


def test_config_from_checkpoint_ignores_unknown_saved_fields(project_root):
    hcdr3_infill = _load_script(project_root, "hcdr3_infill")
    checkpoint = {"train_config": {"training_stage": "base", "removed_legacy_field": 123}}
    cfg = hcdr3_infill.config_from_checkpoint(checkpoint, data_path="d", device="cpu")
    assert cfg.training_stage == "base"  # reconstructs without crashing on the stale key


# --------------------------------------------------------------------------- #
# Generation: ProteinGuide-style guided infilling is opt-in and provenanced
# --------------------------------------------------------------------------- #
def test_hcdr3_infill_cli_guidance_flags_default_off_and_validate(project_root):
    # Guidance is opt-in: the flag defaults to 0.0 (original single-pass path),
    # order defaults to the easy-first schedule, and a negative strength is a
    # loud CLI error rather than a silently-ignored value.
    hcdr3_infill = _load_script(project_root, "hcdr3_infill")
    parser = hcdr3_infill.build_arg_parser()

    defaults = parser.parse_args(["--checkpoint", "c", "--data-path", "d"])
    assert defaults.guidance_strength == 0.0
    assert defaults.guidance_order == "confidence"
    assert defaults.guidance_target == 1

    enabled = parser.parse_args(
        ["--checkpoint", "c", "--data-path", "d",
         "--guidance-strength", "7.5", "--guidance-order", "left_to_right"]
    )
    assert enabled.guidance_strength == 7.5
    assert enabled.guidance_order == "left_to_right"

    with pytest.raises(SystemExit):
        hcdr3_infill.main(["--checkpoint", "c", "--data-path", "d", "--guidance-strength=-1"])


def test_candidate_to_json_records_guidance_provenance(project_root):
    # Output rows carry which schedule produced them; guidance_order is reported
    # only when guidance was actually active.
    hcdr3_infill = _load_script(project_root, "hcdr3_infill")
    from smallAntibodyGen.infill.hcdr3 import HCDR3InfillCandidate, HCDR3Span

    candidate = HCDR3InfillCandidate(
        generated_hcdr3="CARDRST",
        heavy_sequence="AAAACARDRSTYYYY",
        log_probability=-7.0,
        mean_log_probability=-1.0,
        length=7,
        compatibility_score=0.9,
    )
    span = HCDR3Span(aa_start=4, aa_end=11, original_hcdr3="CARDRST")

    class _Rec:
        record_id = "r"
        target_key = "t"
        target_name = "n"
        split = "val"

    guided = hcdr3_infill.candidate_to_json(
        record=_Rec(), true_span=span, length_mode="fixed", candidate=candidate,
        guidance_strength=5.0, guidance_order="confidence",
    )
    plain = hcdr3_infill.candidate_to_json(
        record=_Rec(), true_span=span, length_mode="fixed", candidate=candidate,
    )
    assert guided["guidance_strength"] == 5.0
    assert guided["guidance_order"] == "confidence"
    assert plain["guidance_strength"] == 0.0
    assert plain["guidance_order"] is None


# --------------------------------------------------------------------------- #
# End-to-end: the coordinated train-loop / checkpoint cluster
# --------------------------------------------------------------------------- #
def test_train_one_epoch_evaluate_checkpoint_integration(
    project_root, tmp_path, tokenizer, write_processed_jsonl_gz
):
    mlm_train = _load_script(project_root, "mlm_train")
    from smallAntibodyGen.data.MLMCollator import OASSequenceDataset

    rng = random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"

    def rec(split):
        seq = "".join(rng.choice(aa) for _ in range(30))
        return {"sequence": seq, "locus": "IGH", "chain_group": "heavy",
                "split": split, "length": 30}

    records = [rec("train") for _ in range(16)] + [rec("val") for _ in range(8)]
    data_path = write_processed_jsonl_gz(tmp_path / "tiny.jsonl.gz", records)

    cfg = mlm_train.TrainConfig(
        data_path=str(data_path), training_stage="base", epochs=1,
        batch_size=4, eval_batch_size=4, max_length=64,
        d_model=32, n_heads=4, n_layers=1, d_ff=64, dropout=0.0,
        warmup_steps=2, hcdr3_span_probability=0.0, learning_rate=0.01,
    )
    device = torch.device("cpu")
    train_ds = OASSequenceDataset(str(data_path), split="train")
    val_ds = OASSequenceDataset(str(data_path), split="val")

    model = mlm_train.build_model(tokenizer, cfg, device)
    opt = mlm_train.build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sched = mlm_train.build_lr_scheduler(opt, cfg)

    train_metrics = mlm_train.train_one_epoch(
        model=model, train_dataset=train_ds, tokenizer=tokenizer, optimizer=opt,
        scaler=scaler, scheduler=sched, cfg=cfg, device=device, epoch=0,
    )
    assert math.isfinite(train_metrics["loss"])
    assert 0.0 <= train_metrics["mlm_acc"] <= 1.0
    # 16/4 = 4 optimizer steps > warmup_steps=2, so LR has ramped to full.
    assert opt.param_groups[0]["lr"] == pytest.approx(cfg.learning_rate)

    val_metrics = mlm_train.evaluate(
        model=model, val_dataset=val_ds, tokenizer=tokenizer, cfg=cfg, device=device
    )
    assert math.isfinite(val_metrics["loss"])

    ckpt = tmp_path / "last.pt"
    mlm_train.save_checkpoint(
        ckpt, model, opt, cfg, epoch=1, val_loss=val_metrics["loss"],
        scaler=scaler, scheduler=sched,
    )
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["scheduler_state_dict"] is not None
    # Round-trip restore into fresh objects must not raise.
    model2 = mlm_train.build_model(tokenizer, cfg, device)
    opt2 = mlm_train.build_optimizer(model2, cfg)
    sched2 = mlm_train.build_lr_scheduler(opt2, cfg)
    scaler2 = torch.amp.GradScaler("cuda", enabled=False)
    mlm_train.load_checkpoint(
        ckpt, model2, optimizer=opt2, scaler=scaler2, scheduler=sched2, map_location="cpu"
    )
